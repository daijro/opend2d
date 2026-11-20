//! Geometry combining operations (`CombineWithGeometry`, `Outline`, `Simplify`, `CompareWithGeometry`).
//!
//! **Outline**, **Combine**, **Compare**, **ComputeArea** and a realization's rings run d2d1's region scanner, ported from WPF's MIL geometry code (MIT;
//! see THIRD-PARTY-NOTICES) and checked against d2d1's own build: both operands map onto a fixed-point
//! workspace grid, their figures are cut into monotone chains (`scanner`), swept (`sweep`, with exact
//! segment predicates in `intersect` over `exact`), and at each junction a module classifies the new
//! chains (`recon::Outline`, `recon::Boolean`, `recon::Relation`, `recon::Area`, `recon::Realize`) and
//! strings the surviving ones into figures, or records what it needs. Curves are
//! flattened into the grid by the double `CBezierFlattener` (`curveflat`), each edge tagged with its
//! curve fragment; consecutive fragments of one curve are re-emitted as the **original** curve trimmed
//! with de Casteljau (`CBezier<double>::TrimToStartAt`/`TrimToEndAt`), which is why a rect∪ellipse emits
//! beziers, not a flattened polygon.
//!
//! **Simplify** is separate (`CShapeBase`'s re-stream to the sink): it re-streams the geometry, keeping cubics
//! (`cubics_and_lines`) or flattening them with the float query `CBezierFlattener` (`lines`).
//!
//! **Tessellate** runs the same sweep with the `tess::Tessellator` module, on its own 64 grid.

use super::{flatten, Figure, FillMode, Geometry, Matrix, Point, Prim, Segment};
use curveflat::Bez;

pub(crate) mod curveflat;
pub(crate) mod exact;
pub(crate) mod intersect;
pub(crate) mod recon;
pub(crate) mod scanner;
pub(crate) mod sweep;
pub(crate) mod tess;
pub(crate) mod workspace;

use workspace::Workspace;

/// The workspace a scan of shapes with these tight bounds runs in, and its grid flattening tolerance
/// (the larger scale times the normalized tolerance, or the bounds' larger extent's `3e-12` share).
fn workspace(b: [f32; 4], tol: f32) -> (Workspace, f64) {
    let ws = Workspace::for_bounds(b);
    let empty = b[0] > b[2] || b[1] > b[3];
    // The extent is taken in double, so bounds spanning ±f32::MAX stay finite (SSE max: a NaN width
    // passes the height through, a NaN height wins).
    let (w, h) = (b[2] as f64 - b[0] as f64, b[3] as f64 - b[1] as f64);
    let ext = if empty { 1.0 } else if w > h { w } else { h };
    let ext = if ext.is_finite() { ext } else { 1.0 };
    (ws, ws.tol_grid(super::normalize_tolerance(tol) as f64, ext))
}

/// The union of two shapes' tight bounds as a scan of both takes it: an empty (inverted) one adds nothing.
/// `CombineWithGeometry(other, mode, transform, tol, sink)` — the region boolean. `mode` is
/// `D2D1_COMBINE_MODE` (0=union, 1=intersect, 2=xor, 3=exclude). The transform is the input
/// geometry's (`other`); `this` takes none. Returns a path geometry (fill `ALTERNATE`,
/// native's output convention), with curved boundary runs re-emitted as trimmed beziers.
pub fn combine_with_geometry(this: &Geometry, other: &Geometry, mode: u32, transform: &Matrix, tol: f32) -> Geometry {
    combine_transformed(this, &super::IDENTITY, other, transform, mode, tol)
}

/// [`combine_with_geometry`] with a transform for each operand: native's internal `CShapeBase::Combine`
/// takes one per shape (a layer pop's inverse mask fill passes the mask under the device transform and
/// the clip rect untransformed).
pub(crate) fn combine_transformed(this: &Geometry, mt: &Matrix, other: &Geometry, mo: &Matrix, mode: u32, tol: f32) -> Geometry {
    let mut catalog: Vec<Bez> = Vec::new();
    // Each shape's fill bounds under its matrix, hollow figures included.
    let (ws, tg) = workspace(super::scan_union(this.bounds_of(mt, true), other.bounds_of(mo, true)), tol);
    let mut list = scanner::ChainList::default();
    chains_into(&mut list, this, mt, &mut catalog, tg, &ws);
    list.base = sweep::SHAPE;
    chains_into(&mut list, other, mo, &mut catalog, tg, &ws);
    let winding = [this.fill_mode() == FillMode::Winding, other.fill_mode() == FillMode::Winding];
    let mut s = sweep::Scanner::new(list.chains(), winding);
    let mut b = recon::Boolean::new(&catalog, mode);
    // A scan failure yields no figures for now (native fails the call).
    let figures = if s.scan(&mut b).is_ok() { b.r.out.iter().map(|f| to_device(f, &ws)).collect() } else { Vec::new() };
    Geometry::Path { fill: FillMode::Alternate, figures }
}

/// `Outline(transform, tol, sink)` — the self-intersection-free, winding-normalized boundary of a
/// single geometry's filled region.
pub fn outline(this: &Geometry, transform: &Matrix, tol: f32) -> Geometry {
    let mut catalog: Vec<Bez> = Vec::new();
    // The workspace comes from the fill bounds under the transform, hollow figures included.
    let (ws, tg) = workspace(this.bounds_of(transform, true), tol);
    let mut list = scanner::ChainList::default();
    chains_into(&mut list, this, transform, &mut catalog, tg, &ws);
    let mut s = sweep::Scanner::new(list.chains(), [this.fill_mode() == FillMode::Winding; 2]);
    let mut o = recon::Outline::new(&catalog);
    // A scan failure yields no figures for now (native fails the call).
    let figures = if s.scan(&mut o).is_ok() { o.r.out.iter().map(|f| to_device(f, &ws)).collect() } else { Vec::new() };
    Geometry::Path { fill: FillMode::Alternate, figures }
}

/// `ComputeArea(transform, tol)` of a non-rectangle shape: native scans it in the workspace of its
/// bounds and sums its fill boundary's grid cross products (a negative sum is no area), scaled back by
/// both inverse scales.
pub fn area(this: &Geometry, transform: &Matrix, tol: f32) -> f32 {
    let mut catalog: Vec<Bez> = Vec::new();
    // The workspace comes from the fill bounds under the transform, hollow figures included.
    let (ws, tg) = workspace(this.bounds_of(transform, true), tol);
    let mut list = scanner::ChainList::default();
    list.skip_hollow = true;
    chains_into(&mut list, this, transform, &mut catalog, tg, &ws);
    let mut sc = sweep::Scanner::new(list.chains(), [this.fill_mode() == FillMode::Winding; 2]);
    let mut a = recon::Area::default();
    // A scan failure yields no area for now (native fails the call).
    let s = if sc.scan(&mut a).is_ok() { a.sum } else { 0.0 };
    let half = if s >= 0.0 { s * 0.5 } else { 0.0 };
    (ws.inv_scale_product() * half) as f32
}

/// `CompareWithGeometry(other, transform, tol)` → `D2D1_GEOMETRY_RELATION`
/// (0=unknown, 1=disjoint, 2=is_contained, 3=contains, 4=overlap), read off the relation module's
/// intersection-dimension matrix. The transform is the input geometry's (`other`). Native skips the
/// scan, reporting disjoint, when either shape's bounds are empty or the two bounds are strictly
/// apart (r23 28950: a shape collapsed to a point beside the other).
pub fn compare_with_geometry(this: &Geometry, other: &Geometry, transform: &Matrix, tol: f32) -> u32 {
    // Native's pre-check takes fill bounds with hollow figures included (`GetFillBounds` flag 2).
    let (ba, bb) = (this.bounds_of(&super::IDENTITY, true), other.bounds_of(transform, true));
    let empty = |r: [f32; 4]| r[0] > r[2] || r[1] > r[3];
    if empty(ba) || empty(bb) || ba[2] < bb[0] || bb[2] < ba[0] || ba[3] < bb[1] || bb[3] < ba[1] {
        return 1;
    }
    let mut cat = Vec::new();
    let (ws, tg) = workspace(super::scan_union(ba, bb), tol);
    let mut list = scanner::ChainList::default();
    chains_into(&mut list, this, &super::IDENTITY, &mut cat, tg, &ws);
    list.base = sweep::SHAPE;
    chains_into(&mut list, other, transform, &mut cat, tg, &ws);
    let winding = [this.fill_mode() == FillMode::Winding, other.fill_mode() == FillMode::Winding];
    let mut s = sweep::Scanner::new(list.chains(), winding);
    let mut r = recon::Relation::default();
    // A scan failure reports no relation for now (native fails the call).
    if s.scan(&mut r).is_err() {
        return 0;
    }
    r.finish();
    r.relation()
}

/// The rings of `g`'s filled region under `m` as a geometry realization's tessellator (the AA tessellator)
/// scans them on the fixed `2^20` grid (curves flattened at the normalized tolerance `tol`): the fill
/// rule applied, self-intersections split, coincident edges cancelled, each ring wound with the region
/// on its right (y-down) — the rings a realization feathers.
pub(crate) fn region_contours(g: &Geometry, m: &Matrix, tol: f64) -> Vec<Vec<Point>> {
    region_contours_exact(g, m, tol).into_iter().map(|(pts, _)| pts).collect()
}

/// [`region_contours`] with each point's exact grid position in device units beside it (`2^-20` steps,
/// held in f64): the tessellator that sweeps the region decides its turns on the grid, not on the f32
/// points it emits.
pub(crate) fn region_contours_exact(g: &Geometry, m: &Matrix, tol: f64) -> Vec<(Vec<Point>, Vec<[f64; 2]>)> {
    let ws = Workspace::FIXED;
    // the AA tessellator's scanner skips hollow figures and needs no curve runs.
    let mut list = scanner::ChainList::default();
    list.skip_hollow = true;
    chains_into(&mut list, g, m, &mut Vec::new(), 1_048_576.0 * tol, &ws);
    let chains: Vec<scanner::Chain> = list.chains().iter().map(|c| scanner::Chain { edges: vec![None; c.edges.len()], ..c.clone() }).collect();
    let mut s = sweep::Scanner::new(&chains, [g.fill_mode() == FillMode::Winding; 2]);
    // The tessellator reads no dropped chain once it retires, so the sweep recycles them.
    s.recycle = true;
    let mut t = recon::Realize::new(&[]);
    if s.scan(&mut t).is_err() {
        return Vec::new();
    }
    let d = |p: [f64; 2]| [ws.device_f(p[0], 0), ws.device_f(p[1], 1)];
    let e = |p: [f64; 2]| [ws.device_d(p[0], 0), ws.device_d(p[1], 1)];
    t.r.out
        .iter()
        .map(|f| {
            // Figure-vertex emission: the ring's points in f32, a point equal to the one before dropped,
            // and the closing points that return to the start.
            let mut pts: Vec<Point> = vec![d(f.start)];
            let mut exact = vec![e(f.start)];
            for seg in &f.segs {
                let (recon::OutSeg::Line(p) | recon::OutSeg::Bezier(_, _, p)) = *seg;
                let q = d(p);
                if q != *pts.last().unwrap() {
                    pts.push(q);
                    exact.push(e(p));
                }
            }
            while pts.len() > 1 && pts.last() == pts.first() {
                pts.pop();
                exact.pop();
            }
            (pts, exact)
        })
        .collect()
}

/// `ID2D1Geometry::Tessellate` (`CGeneralFillTessellator::SendToMglSink`): the shape's filled figures
/// under `m`, scanned on the tessellator's workspace with curves flattened at the scale times
/// `max(tolerance, extent·3e-12)` (the raw tolerance: zero, a negative or NaN takes the floor), each band
/// of the fill region triangulated by `CTessellator`, and the triangles mapped back to device f32.
/// Empty fill bounds, or a NaN on either axis, give none.
pub(crate) fn tessellate(g: &Geometry, m: &Matrix, tol: f32) -> Vec<[Point; 3]> {
    // The workspace comes from the shape's local bounds, hollow figures included; the points then
    // stream through `m`.
    let b = g.bounds_of(&super::IDENTITY, true);
    if !(b[2] >= b[0] && b[3] >= b[1]) {
        return Vec::new();
    }
    let (ws, extent) = Workspace::for_tessellation(b);
    let mut list = scanner::ChainList::default();
    list.skip_hollow = true;
    chains_into(&mut list, g, m, &mut Vec::new(), ws.tol_grid(tol as f64, extent), &ws);
    let chains: Vec<scanner::Chain> = list.chains().iter().map(|c| scanner::Chain { edges: vec![None; c.edges.len()], ..c.clone() }).collect();
    let mut s = sweep::Scanner::new(&chains, [g.fill_mode() == FillMode::Winding; 2]);
    let mut t = tess::Tessellator::default();
    // ponytail: a scan native fails returns its HRESULT after the triangles already sent; we keep none.
    if s.scan(&mut t).is_err() {
        return Vec::new();
    }
    t.tris.iter().map(|tri| tri.map(|p| [ws.device_f(p[0], 0), ws.device_f(p[1], 1)])).collect()
}

/// A reconstructed grid figure back in device space.
fn to_device(f: &recon::OutFig, ws: &Workspace) -> Figure {
    let d = |p: [f64; 2]| [ws.device_f(p[0], 0), ws.device_f(p[1], 1)];
    let segments = f
        .segs
        .iter()
        .map(|s| match *s {
            recon::OutSeg::Line(p) => Segment::Line(d(p)),
            recon::OutSeg::Bezier(a, b, c) => Segment::Cubic(d(a), d(b), d(c)),
        })
        .collect();
    Figure { start: d(f.start), filled: f.filled, closed: f.closed, segments, seg_flags: Vec::new() }
}

/// Sends `g`'s figures under `m` to `list`, cut into monotone chains in workspace `ws`: curves flattened
/// on the grid (`tol_grid`) with each edge tagged by its curve fragment, each original grid cubic landing
/// in `catalog`.
fn chains_into(list: &mut scanner::ChainList, g: &Geometry, m: &Matrix, catalog: &mut Vec<Bez>, tol_grid: f64, ws: &Workspace) {
    let gridf = |p: Point| ws.grid(p).map(|v| v as f64);
    let mut figs = Vec::new();
    g.fill_figures(m, &mut figs);
    for (start, prims, closed, filled) in figs {
        if list.skip_hollow && !filled {
            continue;
        }
        list.begin_figure(gridf(start), filled);
        for pr in &prims {
            match *pr {
                Prim::Line(_, e) => list.add(gridf(e)),
                Prim::Cubic(p0, p1, p2, p3) => {
                    let id = catalog.len() as u32;
                    let [g0, g1, g2, g3] = [p0, p1, p2, p3].map(gridf);
                    catalog.push([g0[0], g0[1], g1[0], g1[1], g2[0], g2[1], g3[0], g3[1]]);
                    let mut flat = Vec::new();
                    curveflat::flatten_grid(g0, g1, g2, g3, tol_grid, &mut flat);
                    let mut prev_t = 0.0;
                    for (gv, t) in flat {
                        list.add_edge(gv.map(|v| v as f64), Some((id, prev_t, t)));
                        prev_t = t;
                    }
                }
            }
        }
        list.end_figure(closed);
    }
}

/// `D2D1_GEOMETRY_SIMPLIFICATION_OPTION`.
pub const SIMPLIFY_CUBICS_AND_LINES: u32 = 0;
pub const SIMPLIFY_LINES: u32 = 1;

/// `Simplify(option, transform, tol, sink)`: re-stream `geom` to a path geometry. `cubics_and_lines`
/// keeps cubics (quadratics/arcs already raised by the shape lowering); `lines` flattens every curve
/// with the float `CBezierFlattener` at the normalized tolerance. The transform is applied first, so
/// the result is in post-transform space (query the result with identity).
pub fn simplify(geom: &Geometry, option: u32, transform: &Matrix, tolerance: f32) -> Geometry {
    // The lines-mode flattener takes the tolerance as given: a zero, negative or NaN one flattens at
    // the finest step.
    let tol = tolerance;
    // Native's sink stream emits `SetFillMode` with the source's fill mode; rect/rounded/ellipse
    // have none, so it stays the D2D default `ALTERNATE` (0). Paths/groups carry their own.
    let fill = sink_fill_mode(geom);
    // Native streams the shape to the sink as it is, non-finite figures included (a rectangle's
    // ORDERED rect: min/max per axis, a NaN never swaps; r23 28114, 28379), through the transform. It
    // emits each figure's defining segments and relies on `EndFigure(closed)` for the closing edge,
    // never streaming a synthetic edge back to the start. `sink_figures` appends that closing edge (a
    // `Line` ending at a copy of `start`, so bit-equal even when NaN), so drop it on closed contours;
    // a path figure's own last line back to its start is one of its segments and streams. The closing
    // edge is still counted by `GetSegmentCount` (see `segment_count`), just not streamed.
    let mut raw = Vec::new();
    geom.sink_figures(false, &mut raw);
    let mut appended = Vec::new();
    geom.figures_close_appended(&mut appended);
    // A hollow figure streams hollow (`D2D1_FIGURE_BEGIN_HOLLOW`).
    let mut filled = Vec::new();
    geom.figures_filled(&mut filled);
    // Each primitive's segment flags, as the shape streams them (D2D1_PATH_SEGMENT bits).
    let mut flags = Vec::new();
    geom.figures_prim_flags(&mut flags);
    let mut figures = Vec::new();
    for ((((start, prims, closed), filled), mut pflags), appended) in raw.into_iter().zip(filled).zip(flags).zip(appended) {
        let tp = |p| super::sink_point(transform, p);
        let start = tp(start);
        let mut prims: Vec<Prim> = prims
            .iter()
            .map(|p| match *p {
                Prim::Line(a, b) => Prim::Line(tp(a), tp(b)),
                Prim::Cubic(a, b, c, d) => Prim::Cubic(tp(a), tp(b), tp(c), tp(d)),
            })
            .collect();
        // The flags in force when the figure ends (its closing edge's): the flattener restores a
        // curve's own flags after its chords.
        let close_f = pflags.last().copied().unwrap_or(0);
        // `appended` marks exactly the figures `sink_figures` closed with a synthetic line. Under a
        // transformed shape the start takes the full product and the segments the matrix's form, so the
        // line's end need not equal the mapped start (r23 28775: a zero matrix maps a -inf start to NaN
        // and the line's end to the translation).
        if closed && appended {
            if let Some(Prim::Line(..)) = prims.last() {
                prims.pop();
                pflags.truncate(prims.len());
            }
        }
        let (mut segments, mut seg_flags) = (Vec::new(), Vec::new());
        for (k, pr) in prims.iter().enumerate() {
            let f = pflags.get(k).copied().unwrap_or(0);
            match *pr {
                Prim::Line(_, e) => {
                    segments.push(Segment::Line(e));
                    seg_flags.push(f);
                }
                Prim::Cubic(p0, p1, p2, p3) => {
                    if option == SIMPLIFY_LINES {
                        // The flattening sink streams a curve's chords after the first with
                        // D2D1_PATH_SEGMENT_FORCE_ROUND_LINE_JOIN, so a stroke turns round inside the curve.
                        let mut pts = Vec::new();
                        flatten::flatten_cubic_sink(p0, p1, p2, p3, tol, &mut pts);
                        seg_flags.extend((0..pts.len()).map(|i| if i == 0 { f } else { f | 2 }));
                        segments.extend(pts.into_iter().map(Segment::Line));
                    } else {
                        segments.push(Segment::Cubic(p1, p2, p3));
                        seg_flags.push(f);
                    }
                }
            }
        }
        if closed {
            seg_flags.push(close_f);
        }
        figures.push(Figure { start, filled, closed, segments, seg_flags });
    }
    Geometry::Path { fill, figures }
}

// The fill mode native's sink receives: a path/group's own, else the D2D default `ALTERNATE`.
fn sink_fill_mode(geom: &Geometry) -> FillMode {
    match geom {
        Geometry::Path { fill, .. } | Geometry::Group { fill, .. } => *fill,
        Geometry::Transformed { child, .. } => sink_fill_mode(child),
        _ => FillMode::Alternate,
    }
}

/// Native `GetSegmentCount` over an operation's output path: every streamed segment, plus one
/// implicit closing edge per closed figure (`EndFigure(closed)`). Matches rect=4 (3+1), ellipse=8
/// (7+1), rect-union=9 (8+1).
pub fn segment_count(geom: &Geometry) -> usize {
    match geom {
        Geometry::Path { figures, .. } => {
            figures.iter().map(|f| f.segments.len() + usize::from(f.closed)).sum()
        }
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::IDENTITY;

    // Huge coordinates (a -3.4e38 edge, a 1e9 edge) neither overflow nor wrap the scan (r23 28053, 28100),
    // and two overlapping squares still union to one eight-sided figure.
    #[test]
    fn huge_coordinates_do_not_overflow() {
        for r in [[36.43297, 2.741453, -3.4e38, 39.7398], [1e9, 29.026863, -420.0, 37.377567]] {
            match outline(&Geometry::Rect(r), &IDENTITY, 0.25) {
                Geometry::Path { figures, .. } => assert_eq!(figures.len(), 1),
                _ => unreachable!(),
            }
        }
        match combine_with_geometry(&Geometry::Rect([0.0, 0.0, 10.0, 10.0]), &Geometry::Rect([5.0, 5.0, 20.0, 20.0]), 0, &IDENTITY, 0.25) {
            Geometry::Path { figures, .. } => assert_eq!((figures.len(), figures[0].segments.len()), (1, 8)),
            _ => unreachable!(),
        }
    }

    // A rect simplifies to one closed figure of 3 streamed line segments (top/right/bottom); the
    // closing left edge is implicit. `GetSegmentCount` == 3 + 1 (closed) == 4.
    #[test]
    fn rect_simplify_stream_is_three_lines() {
        let g = Geometry::Rect([0.0, 0.0, 10.0, 8.0]);
        let s = simplify(&g, SIMPLIFY_CUBICS_AND_LINES, &IDENTITY, 0.25);
        match &s {
            Geometry::Path { figures, .. } => {
                assert_eq!(figures.len(), 1);
                assert!(figures[0].closed);
                assert_eq!(figures[0].segments.len(), 3);
            }
            _ => panic!("expected path"),
        }
        assert_eq!(segment_count(&s), 4);
    }

    // Simplify streams non-finite shapes as native does (VM probe, 26100): the NaN-radius rounded
    // rect and ellipse keep their figures, and LINES flattens a cubic whose second differences are NaN
    // in y to its endpoint (the flatness max is NaN, which no bound exceeds) but subdivides one NaN in x
    // by its y differences. Streamed segments exclude the closing edge.
    #[test]
    fn simplify_streams_nonfinite_shapes() {
        let streamed = |g: &Geometry, option| match simplify(g, option, &IDENTITY, 0.25) {
            Geometry::Path { figures, .. } => figures.iter().map(|f| f.segments.len()).collect::<Vec<_>>(),
            _ => panic!("expected path"),
        };
        let nan_ry = Geometry::Rounded { rect: [5.0, 5.0, 30.0, 30.0], radius: [3.0, f32::NAN] };
        let nan_rx = Geometry::Ellipse { center: [43.986_988, 13.845_563], radius: [f32::NAN, 1.691_258_4] };
        assert_eq!(streamed(&nan_ry, SIMPLIFY_CUBICS_AND_LINES), [7]);
        assert_eq!(streamed(&nan_ry, SIMPLIFY_LINES), [7]);
        assert_eq!(streamed(&nan_rx, SIMPLIFY_LINES), [11]);
        assert_eq!(streamed(&Geometry::Rect([f32::NAN, 20.2, 39.4, 63.0]), SIMPLIFY_LINES), [3]);
    }

    // comb-31: intersecting two edge-sharing rects yields the shared edge as ONE open hollow
    // 1-segment figure (start (60,70) → line (60,20)), oriented with `this` on its left.
    #[test]
    fn edge_sharing_intersect_is_open_shared_edge() {
        let a = Geometry::Rect([20.0, 20.0, 60.0, 70.0]);
        let b = Geometry::Rect([60.0, 20.0, 100.0, 70.0]);
        let r = combine_with_geometry(&a, &b, 1, &IDENTITY, 0.25);
        let Geometry::Path { figures, .. } = &r else { panic!("expected path") };
        assert_eq!(figures.len(), 1);
        let f = &figures[0];
        assert!(!f.filled && !f.closed, "shared edge is hollow + open");
        assert_eq!(f.start, [60.0, 70.0]);
        assert!(matches!(f.segments.as_slice(), [Segment::Line([60.0, 20.0])]));
        assert_eq!(segment_count(&r), 1);
        assert_eq!(r.get_bounds(&IDENTITY), [f32::INFINITY, f32::INFINITY, f32::MAX, f32::MAX]);
    }
}
