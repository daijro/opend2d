//! `ID2D1DeviceContext1` geometry realizations (phase 1C-8 / feather VS): a filled realization bakes
//! an antialiased triangle mesh at create time and replays it through WARP's realization vertex +
//! coverage pixel shaders under the draw-time transform.
//!
//! **Filled realizations — the feather mesh (`realize-01`/`-02`/`-03`/`-05`), as native bakes it.**
//!   1. The realization build step sweeps the shape with the software AA tessellator at an identity
//!      scale-and-shift (the clip-to-safe-device-bounds transform is identity), so the mesh is in the
//!      geometry's own DIP space. The sweep's robustness grid is 2^20 but does not scale the output.
//!   2. The tessellator's figure-vertex emission (after an impl pass transforms + dedups the outline)
//!      emits the **feather mesh** vertex
//!      `{f32 x, f32 y, f32 ax, f32 ay, u32 cov}`. For outline vertex `i` with unit edge directions
//!      `in = unit(p_i − p_{i-1})`, `out = unit(p_{i+1} − p_i)` (zero-length edge ⇒ all offsets 0):
//!        * interior ring `[0,n)`:  pos `p_i`, `(ax,ay)=p_i+(−(in.y+out.y), in.x+out.x)`, `cov=1`
//!        * outer ring 1 `[n,2n)`:  pos `p_i`,    `(ax,ay)=p_i+(out.y,−out.x)`, `cov=0`
//!        * outer ring 2 `[2n,3n)`: pos `p_{i+1}`,`(ax,ay)=p_{i+1}+(out.y,−out.x)`, `cov=0`
//!      Triangles per edge `i`: the strip `{int[i],out1[i],int[i+1]}` + `{out1[i],out2[i],int[i+1]}`,
//!      plus a corner triangle `{out2[i-1],out1[i],int[i]}` at convex vertices (`out.x·in.y −
//!      out.y·in.x ≤ 0`). `cov` is `0x1000000` = 8.24 one → `1.0` in the shader. The interior is
//!      filled by the base monotone triangulation of the interior ring (all `cov=1`); since that
//!      region is convex for these shapes a fan tiles the identical pixels.
//!   3. `DrawGeometryRealization` (opaque brush) → an internal helper → the static-buffer reset binds
//!      the draw transform (the render target's world∘DPI matrix) + the device AA-width scalar
//!      (0 when aliased). WARP's realization
//!      **VS** displaces each vertex `finalpos = transform·(pos + (a − pos)·aaWidth)` and the
//!      **PS** outputs `premul_brush · cov`, composited in integer UNORM8 — i.e. exactly
//!      [`crate::warp::fill_triangle`] with the coverage carried in the alpha channel (opaque brush ⇒
//!      premul = straight, so `premul·cov = straight·cov`). `aaWidth = 0.5` (the half-pixel feather),
//!      confirmed byte-exact across `realize-01/-03` (and `-02/-05` with curve flattening).
//!
//! A **stroked** realization (`realize-04`) bakes a pixel-aligned mesh with no feather; the re-raster
//! through the ordinary stroke coverage path is byte-exact and kept.

use super::{Geometry, Point, Prim, StrokeStyle};
use crate::pixels::Blend;
use crate::warp::{self, Vert};
use crate::{DeviceContext, SolidColorBrush};
use d2d_types::contracts::PixelBuffer;

/// Device AA half-width (`CD3DDeviceLevel1+0xF0`): the feather spans ±`AA_WIDTH` device pixels around
/// the silhouette, a one-pixel band, at any draw scale.
pub(crate) const AA_WIDTH: f32 = 0.5;

/// Realization sweep robustness grid (`CScannerModuleD::Set(…, shift=20)` ⇒ scale 2^20); curves are
/// flattened on this grid, then converted back to device space.
const SWEEP_SCALE: f64 = (1u64 << 20) as f64;

impl DeviceContext {
    /// Draw a baked geometry realization under the current transform; `stroke` present ⇒ a stroked
    /// realization (re-rastered, no feather — `realize-04`), else the filled feather mesh.
    pub fn draw_geometry_realization(&mut self, geom: &Geometry, stroke: Option<&(f32, StrokeStyle)>, tolerance: f32, brush: &SolidColorBrush) {
        let _a = self.api(40 + 24, 0, None);
        // A latched draw error (or not being in a draw) makes this a silent no-op, as every other draw:
        // native's sticky WRONG_STATE skips the op (fuzz 16792: a per-primitive FillMesh latches WRONG_STATE,
        // then a later realization draw is skipped — we were flattening the stroke first and hanging on a
        // NaN tolerance). Checked before the DIP widen so the hang never runs.
        if !self.ready_to_draw() {
            return;
        }
        // Native's CreateFilled/StrokedGeometryRealization rejects a geometry with ANY non-finite point
        // (empty realization → draws nothing), unlike direct FillGeometry/DrawGeometry which drop only the
        // bad figures (fuzz r11 16021: a NaN figure voids the whole stroked realization; per-figure filtering
        // would stroke the sibling finite figure at width 1e9 over the target). A filled realization bakes
        // only the filled figures, so a hollow one neither fills nor voids it.
        let filled = if stroke.is_none() { filled_only(geom) } else { None };
        let geom = filled.as_ref().unwrap_or(geom);
        if geom.any_nonfinite() {
            return;
        }
        let dip = match stroke {
            Some((w, st)) => dip_stroke(geom, *w, st, tolerance),
            None => safe_clipped(geom),
        };
        let geom = match (stroke, &dip) {
            (_, Some(g)) => g,
            (Some((w, st)), None) => return self.draw_geometry(geom, brush, *w, st),
            (None, None) => geom,
        };
        if !self.is_in_draw() {
            return;
        }
        let m = device_matrix(self);
        let blend = blend_of(self.get_primitive_blend());
        let modifier = self.opacity_modifier(stroke, &m);
        let cull = realization_cull(&m);
        let Some(mut buf) = self.take_target() else {
            return;
        };
        // An opaque brush under SOURCE_OVER draws the mesh straight onto the target; anything else goes
        // through the A8 coverage intermediate (`fill_feather_mask`).
        let op = brush.opacity;
        if op == 1.0 && brush.color[3].clamp(0.0, 1.0) == 1.0 && blend == Blend::SourceOver {
            // The realization PS outputs `premul_brush · cov`; the brush opacity carries the stroke's
            // opacity modifier.
            let a = brush.color[3] * (op * modifier);
            let premul = [brush.color[0] * a, brush.color[1] * a, brush.color[2] * a, a];
            fill_feather_tol(&mut buf, geom, premul, &m, blend, self.aa_width(), tolerance, cull);
        } else {
            let a = brush.color[3] * op;
            let src = [brush.color[2] * a * 255.0, brush.color[1] * a * 255.0, brush.color[0] * a * 255.0, a * 255.0]; // BGRA
            fill_feather_mask(&mut buf, geom, &m, blend, self.aa_width(), tolerance, cull, modifier, |_, _| src);
        }
        self.set_target_bitmap(buf);
    }

    /// `DrawGeometryRealization` with a gradient/bitmap/image brush. A fill realization replays the
    /// baked feather mesh (`fill_feather_mask`) with the brush as a per-pixel source — the same mesh
    /// and coverage as the solid [`draw_geometry_realization`], so the AA rim matches native. A stroke
    /// realization re-rasters through the stroke-brush path (native strokes the realized outline).
    pub fn draw_geometry_realization_brush(&mut self, geom: &Geometry, stroke: Option<&(f32, StrokeStyle)>, tolerance: f32, brush: &crate::PixelBrush) {
        let _a = self.api(40 + brush.batch, 0, None);
        if !self.ready_to_draw() {
            return; // latched error / not in a draw → no-op before the DIP widen (see the solid path)
        }
        let filled = if stroke.is_none() { filled_only(geom) } else { None };
        let geom = filled.as_ref().unwrap_or(geom);
        if geom.any_nonfinite() {
            return; // non-finite geometry → empty realization (see `draw_geometry_realization`)
        }
        let dip = match stroke {
            Some((w, st)) => dip_stroke(geom, *w, st, tolerance),
            None => safe_clipped(geom),
        };
        let geom = match (stroke, &dip) {
            (_, Some(g)) => g,
            (Some((w, st)), None) => return self.draw_geometry_brush(geom, brush, *w, st),
            (None, None) => geom,
        };
        if !self.is_in_draw() || !self.check_brush(brush) {
            return;
        }
        // The brush is placed under the device transform current at the draw, as every fill realizes it.
        let placed = self.realize_brush_for_fill(brush);
        let brush = placed.as_ref().unwrap_or(brush);
        let m = device_matrix(self);
        let blend = blend_of(self.get_primitive_blend());
        let modifier = self.opacity_modifier(stroke, &m);
        let cull = realization_cull(&m);
        let Some(mut buf) = self.take_target() else {
            return;
        };
        // ponytail: an opaque gradient/bitmap brush takes native's direct path (no A8 intermediate); it
        // shares the mask path here, which only differs where feather triangles overlap.
        fill_feather_mask(&mut buf, geom, &m, blend, self.aa_width(), tolerance, cull, modifier, |x, y| brush.brush.source_at_frac(x, y));
        self.set_target_bitmap(buf);
    }

    /// The opacity factor native applies to the brush under per-primitive antialiasing only: a stroked
    /// realization thinner than a device pixel draws at `max_scale(transform) · |width|` opacity (its
    /// baked mesh is that thin and its feather a full pixel); a filled one, or any aliased draw, at 1.
    fn opacity_modifier(&self, stroke: Option<&(f32, StrokeStyle)>, m: &[f64; 6]) -> f32 {
        let Some((w, _)) = stroke else { return 1.0 };
        if self.get_antialias_mode() == 1 {
            return 1.0;
        }
        let t = crate::extras::print::compute_maximum_scale_factor(m.map(|v| v as f32)) * w.abs();
        if t >= 1.0 { 1.0 } else { t }
    }

    /// The device AA-width scalar the static-buffer reset binds: 0 under an aliased draw,
    /// which collapses the feather rings onto the outline and leaves the hard-edged interior (r13 18746).
    fn aa_width(&self) -> f32 {
        if self.get_antialias_mode() == 1 { 0.0 } else { AA_WIDTH }
    }
}

/// The rasterizer cull mode of a realization draw under device transform `m`: back faces under a
/// non-negative determinant, front faces under a mirroring one (a NaN determinant compares false and
/// culls front). A feather band thinner than its feather inverts its interior triangles, which this
/// culls, leaving only the fringe strips.
fn realization_cull(m: &[f64; 6]) -> warp::Cull {
    let f = m.map(|v| v as f32);
    if f[3] * f[0] - f[2] * f[1] >= 0.0 { warp::Cull::Back } else { warp::Cull::Front }
}

/// A stroked realization's outline as native bakes it: the realization build step widens at
/// create time in DIP space (identity, 96 dpi) at the realization's own flattening tolerance and
/// tessellates the widened shape like a filled realization, whatever the stroke transform type: a
/// HAIRLINE or FIXED stroke's 1-unit width and dash lengths are laid out in DIPs (r13 18729), and a
/// NORMAL one gets the realization feather (r14 19242, 19326). The scan's region resolution
/// (`combine::region_contours`, in the feather) unions the widened rings under nonzero winding,
/// swallowtails, self-overlaps and the inverted inner offset of a stroke wider than its shape included
/// (r15 20344 resolves to its outer ring); here a rectangle's rings lose their collinear vertices
/// (native's realize-04 mesh: two 4-vertex rings).
fn dip_stroke(geom: &Geometry, width: f32, style: &StrokeStyle, tolerance: f32) -> Option<Geometry> {
    if let Some(rings) = stroke_special_case(geom, width, style) {
        return Some(rings);
    }
    let clipped = stroke_safe_clipped(geom, width, style);
    let geom = clipped.as_ref().unwrap_or(geom);
    let polys = geom.widen_realization(width, style, tolerance);
    // A clipped source bakes its widen as is, the pen reaching past the box at the cut included.
    // ponytail: an unclipped NORMAL stroke whose widen still leaves ±524287 (a non-finite shape, which
    // skips the clip) keeps the draw-time stroke; bake it once the clip takes non-finite shapes.
    if clipped.is_none() && style.transform_type == crate::geometry::stroke::TransformType::Normal
        && polys.iter().flatten().any(|v| !(v[0].abs() < 524287.0 && v[1].abs() < 524287.0))
    {
        return None;
    }
    // A rectangle the special case passes over (a dashed or fixed stroke) loses its rings' collinear
    // vertices; any other shape (a rectangle inside a group included) keeps every vertex the widener
    // gives it.
    let rect = matches!(geom, Geometry::Rect(_));
    let rings: Vec<Vec<Point>> = polys.into_iter().map(|p| if rect { drop_collinear(p) } else { p }).filter(|p| p.len() >= 3).collect();
    let figures = rings.into_iter().map(|p| crate::geometry::Figure {
        start: p[0],
        filled: true,
        closed: true,
        segments: p[1..].iter().map(|&q| crate::geometry::Segment::Line(q)).collect(),
        seg_flags: Vec::new(),
    }).collect();
    Some(Geometry::Path { fill: crate::geometry::FillMode::Winding, figures })
}

/// A stroked realization's widen when the shape widens itself (`CCompactShape::WidenToShape` runs a
/// special-case-widen test before any pen is built): a rectangle, or a rounded rect or ellipse with equal
/// radii, stroked solid with a normal pen, that the safe-box clip leaves alone. The analytic rings, curves
/// kept, tessellate as a filled realization under the rings' alternate fill; no pen, so the tolerance
/// never voids them.
pub(crate) fn stroke_special_case(geom: &Geometry, width: f32, style: &StrokeStyle) -> Option<Geometry> {
    if !matches!(geom, Geometry::Rect(_) | Geometry::Rounded { .. } | Geometry::Ellipse { .. }) || stroke_safe_clipped(geom, width, style).is_some() {
        return None;
    }
    geom.special_case_widen(width, style, &super::IDENTITY)
}

/// The source shape a stroked realization widens when its stroke's loose bounds pass the |524287| safe
/// box: every figure strip-clipped to `[-524287, 524287]²` first (each keeps its end type), so a huge
/// shape is widened, and dashed, only inside the box (r22 27668: a 1e9-tall dashed rect).
/// ponytail: shapes with a non-finite point keep the unclipped widen; the strip clip of an ±inf edge
/// would overflow (see [`Geometry::realize_empty`]).
pub(crate) fn stroke_safe_clipped(geom: &Geometry, width: f32, style: &StrokeStyle) -> Option<Geometry> {
    let [l, t, r, b] = stroke_loose_bounds(geom, width, style)?;
    if r <= l || b <= t {
        return None;
    }
    let corners = [[l, t], [r, t], [l, b], [r, b]].map(|p| super::transform_point(&super::IDENTITY, p));
    if corners.iter().flatten().all(|v| v.abs() <= 524287.0) {
        return None;
    }
    // A rectangle's shape holds its rect in order, so the clipped figure starts at the top-left corner.
    if let Geometry::Rect([l, t, r, b]) = *geom {
        return clipped_to_keep(&Geometry::Rect([l.min(r), t.min(b), l.max(r), t.max(b)]), 524287.0, true);
    }
    clipped_to_keep(geom, 524287.0, true)
}

/// Whether a geometry realization's create bakes an EMPTY realization (S_OK, draws nothing) at its
/// bounds tests. A filled realization takes its fill bounds, a stroked one its loose bounds (hollow
/// figures included), and bakes empty when they compare inverted or unordered (a NaN width, say). When
/// they pass the |524287| safe box it bakes the strip-clipped shape instead, empty when that shape's
/// fill bounds (filled figures only) are degenerate or NaN (r22 27847: a quad to y=-3.4e38 clips to NaN
/// crossings; 27856: an ellipse 3.4e38 tall). A NaN crossing in a hollow figure does not count there,
/// and the stroke's widen of it fails instead (r11 16683).
pub fn realization_clips_empty(geom: &Geometry, stroke: Option<(f32, &StrokeStyle)>) -> bool {
    let filled = filled_only(geom);
    let fill_geom = filled.as_ref().unwrap_or(geom);
    let ordered = |g: &Geometry| match *g {
        Geometry::Rect([l, t, r, b]) => Geometry::Rect([l.min(r), t.min(b), l.max(r), t.max(b)]),
        _ => g.clone(),
    };
    // The shape keeps a path figure with an infinite point (the sink does not drop it).
    let kept = |g: &Geometry| ordered(g).prims_per_figure_keep(&super::IDENTITY, f32::INFINITY);
    let mut hollow_nan = false;
    let [l, t, r, b] = match stroke {
        Some((w, st)) => {
            if geom.any_nonfinite() && !geom.path_nonfinite() {
                return true;
            }
            // A filled figure that converts to NaN bakes the realization empty, beside finite ones too (r11 16021,
            // r24 29835). A NaN only in a hollow figure leaves native's loose bounds all NaN, which is no inverted box:
            // the shape goes on to the safe-box clip and the widen, which fails (VM p753: a hollow line to NaN beside a
            // finite fill is BAD_NUMBER, the same line in the lone filled figure is S_OK).
            if fill_geom.raw_shape_has_nan() {
                return true;
            }
            if geom.raw_shape_has_nan() {
                hollow_nan = true;
                [f32::NAN; 4]
            } else {
                loose_bounds_of(&kept(geom), w, st)
            }
        }
        None => {
            if fill_geom.realize_empty() {
                return true;
            }
            fill_geom.get_bounds(&super::IDENTITY)
        }
    };
    if r < l || b < t || (!hollow_nan && !(r >= l && b >= t)) {
        return true;
    }
    if [[l, t], [r, t], [l, b], [r, b]].iter().flat_map(|&p| super::transform_point(&super::IDENTITY, p)).all(|v| v.abs() <= 524287.0) {
        return false;
    }
    use crate::pixels::Seg;
    let mut pts: Vec<Point> = Vec::new();
    for (start, prims, _, _) in &kept(fill_geom) {
        let segs: Vec<Seg> = prims.iter().map(|p| match *p { Prim::Line(_, b) => Seg::Line(b), Prim::Cubic(_, a, b, c) => Seg::Cubic(a, b, c) }).collect();
        if let Some((s0, out)) = crate::pixels::clip_figure_to(*start, &segs, 524287.0) {
            pts.push(s0);
            pts.extend(out.iter().flat_map(|g| match *g { Seg::Line(q) => vec![q], Seg::Cubic(a, c, q) => vec![a, c, q] }));
        }
    }
    if pts.iter().flatten().any(|v| v.is_nan()) {
        return true;
    }
    // The clipped shape's fill bounds start at ±FLT_MAX and take a point only past them (`lo > p`,
    // `p > hi`), so an infinite coordinate never moves the near side: a figure clipped to y = +inf on
    // every vertex spans FLT_MAX..inf, not empty, and goes on to fail the scan (r26 31389).
    let mut lo = [f32::MAX; 2];
    let mut hi = [-f32::MAX; 2];
    for p in &pts {
        for k in 0..2 {
            if lo[k] > p[k] {
                lo[k] = p[k];
            }
            if p[k] > hi[k] {
                hi[k] = p[k];
            }
        }
    }
    !(hi[0] > lo[0] && hi[1] > lo[1])
}

/// A stroke's loose bounds with no transform at 96 dpi: the shape's control-point bounds grown by half
/// the width (one for a hairline) times the miter limit·√2 for a MITER or MITER_OR_BEVEL join, else √2
/// when any cap is SQUARE, else 1. `None` for a shape with a non-finite point.
fn stroke_loose_bounds(geom: &Geometry, width: f32, style: &StrokeStyle) -> Option<[f32; 4]> {
    let figs = geom.prims_per_figure(&super::IDENTITY);
    if figs.iter().any(|(s, p, _, _)| !shape_points(*s, p).all(|q| q[0].is_finite() && q[1].is_finite())) {
        return None;
    }
    Some(loose_bounds_of(&figs, width, style))
}

/// A figure's start and every segment point, controls included.
fn shape_points(start: Point, prims: &[Prim]) -> impl Iterator<Item = Point> + '_ {
    std::iter::once(start).chain(prims.iter().flat_map(|p| match *p {
        Prim::Line(_, b) => vec![b],
        Prim::Cubic(_, a, b, c) => vec![a, b, c],
    }))
}

/// [`stroke_loose_bounds`] of these figures, an infinite point included.
fn loose_bounds_of(figs: &[(Point, Vec<Prim>, bool, Vec<Point>)], width: f32, style: &StrokeStyle) -> [f32; 4] {
    use super::stroke::{Cap, Join, TransformType};
    let mut lo = [f32::INFINITY; 2];
    let mut hi = [f32::NEG_INFINITY; 2];
    for (start, prims, _, _) in figs {
        for q in shape_points(*start, prims) {
            for k in 0..2 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
            }
        }
    }
    if lo[0] > hi[0] || lo[1] > hi[1] {
        return [lo[0], lo[1], hi[0], hi[1]];
    }
    let w = if style.transform_type == TransformType::Hairline { 1.0 } else { (width * 0.5).abs() + (width * 0.5).abs() };
    let half = (96.0 * w / 96.0) * 0.5;
    let k = match style.join {
        Join::Miter | Join::MiterOrBevel => (style.miter_limit.max(1.0) as f64 * std::f64::consts::SQRT_2) as f32,
        _ if [style.start_cap, style.end_cap, style.dash_cap].contains(&Cap::Square) => std::f32::consts::SQRT_2,
        _ => 1.0,
    };
    let d = k * half;
    [lo[0] - d, lo[1] - d, hi[0] + d, hi[1] + d]
}

/// The figures a filled realization bakes: a hollow (`filled: false`) path figure never reaches the fill
/// sweep (r19 24358: a hollow figure with NaN points beside a filled one still realizes the filled one).
/// `None` when there is no hollow figure to drop.
pub(crate) fn filled_only(g: &Geometry) -> Option<Geometry> {
    match g {
        Geometry::Path { fill, figures } if figures.iter().any(|f| !f.filled) => {
            Some(Geometry::Path { fill: *fill, figures: figures.iter().filter(|f| f.filled).cloned().collect() })
        }
        Geometry::Group { fill, children } => {
            let kept: Vec<Option<Geometry>> = children.iter().map(filled_only).collect();
            kept.iter().any(Option::is_some).then(|| Geometry::Group {
                fill: *fill,
                children: kept.into_iter().zip(children).map(|(k, c)| k.unwrap_or_else(|| c.clone())).collect(),
            })
        }
        Geometry::Transformed { matrix, child } => filled_only(child).map(|c| Geometry::Transformed { matrix: *matrix, child: Box::new(c) }),
        _ => None,
    }
}

/// A filled realization's shape as native bakes it when its tight bounds pass the |524287| safe box:
/// each figure cut to `[-524287, 524287]²` by the same strip clip a huge fill takes, so a huge or
/// far-off control point no longer reaches the sweep. `None` when the bounds stay inside (bake as is).
/// A stroked realization's source takes the same clip by its stroke's loose bounds ([`stroke_safe_clipped`]).
fn safe_clipped(geom: &Geometry) -> Option<Geometry> {
    if !geom.clips_huge(&crate::geometry::IDENTITY) {
        return None;
    }
    clipped_to(geom, 524287.0)
}

/// Every figure of `geom` strip-clipped to `[-bound, bound]²` (a filled path of lines and cubics).
fn clipped_to(geom: &Geometry, bound: f32) -> Option<Geometry> {
    clipped_to_keep(geom, bound, false)
}

/// [`clipped_to`]; `keep_end` keeps each figure's open/closed end, as the strip clip passes it through
/// (a stroke needs it), where a fill closes every figure.
/// `keep_end`: the shape is to be stroked, so it is clipped as the widener takes it (split at its
/// FORCE_UNSTROKED segments, open figures kept open); otherwise it is filled, every figure whole.
fn clipped_to_keep(geom: &Geometry, bound: f32, keep_end: bool) -> Option<Geometry> {
    use crate::pixels::Seg;
    // A stroke also keeps each figure's FORCE_ROUND_LINE_JOIN vertices.
    let figs: Vec<(Point, Vec<Prim>, bool, Vec<Point>)> = if keep_end {
        geom.prims_per_figure(&crate::geometry::IDENTITY)
    } else {
        let mut f = Vec::new();
        geom.fill_figures(&crate::geometry::IDENTITY, &mut f);
        f.into_iter().map(|(s, p, c, _)| (s, p, c, Vec::new())).collect()
    };
    let figures = figs
        .into_iter()
        .filter_map(|(start, prims, closed, round)| {
            let segs: Vec<Seg> = prims.iter().map(|p| match *p { Prim::Line(_, b) => Seg::Line(b), Prim::Cubic(_, a, b, c) => Seg::Cubic(a, b, c) }).collect();
            let (start, segs) = crate::pixels::clip_figure_to(start, &segs, bound)?;
            let segments: Vec<crate::geometry::Segment> = segs.iter().map(|g| match *g { Seg::Line(q) => crate::geometry::Segment::Line(q), Seg::Cubic(a, b, c) => crate::geometry::Segment::Cubic(a, b, c) }).collect();
            // The forced-round joins go back on the segments that start at them (the clip keeps a vertex
            // inside the box where it was), and a closed figure's closing edge carries its own flag.
            let mut seg_flags = Vec::new();
            if !round.is_empty() {
                let mut cur = start;
                for seg in &segments {
                    seg_flags.push(if !seg_flags.is_empty() && round.contains(&cur) { 2 } else { 0 });
                    cur = match *seg { crate::geometry::Segment::Line(q) | crate::geometry::Segment::Cubic(_, _, q) => q, _ => cur };
                }
                if closed {
                    seg_flags.push(if round.contains(&start) && (cur == start || round.contains(&cur)) { 2 } else { 0 });
                }
            }
            Some(crate::geometry::Figure { start, filled: true, closed: closed || !keep_end, segments, seg_flags })
        })
        .collect();
    Some(Geometry::Path { fill: geom.fill_mode(), figures })
}



/// Drop each vertex exactly on the straight run between its neighbours.
fn drop_collinear(mut p: Vec<Point>) -> Vec<Point> {
    while let Some(i) = (0..p.len()).find(|&i| {
        let n = p.len();
        let (a, b, c) = (p[(i + n - 1) % n], p[i], p[(i + 1) % n]);
        let (u, v) = ([b[0] - a[0], b[1] - a[1]], [c[0] - b[0], c[1] - b[1]]);
        n > 3 && u[0] * v[1] - u[1] * v[0] == 0.0 && u[0] * v[0] + u[1] * v[1] >= 0.0
    }) {
        p.remove(i);
    }
    p
}

/// Build the feather mesh for `geom`, displace each vertex by `AA_WIDTH` along its device-space AA
/// direction (the realization VS), and return the device positions with their coverage per vertex,
/// plus the triangle index list. Shared by the solid [`fill_feather`] and the brush
/// [`fill_feather_mask`] so both replay the *same* baked mesh and coverage.
fn feather_device(geom: &Geometry, m: &[f64; 6], aa_width: f32, tolerance: f32) -> (Vec<([f32; 2], f32)>, Vec<[u32; 3]>) {
    let mut verts: Vec<FVert> = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut rings: Vec<(u32, usize)> = Vec::new();
    let tol = scan_tolerance(geom, tolerance);
    // The feather rims the region the sweep resolves, not the raw figures: the fill rule applied,
    // self-intersections split, coincident edges cancelled, every contour wound with the region on its
    // right — so a reversed figure feathers as its forward twin and an even-odd duplicate draws nothing.
    // The whole shape is resolved, as far out as the safe-box clip left it.
    let (outlines, exact): (Vec<Vec<Point>>, Vec<Vec<[f64; 2]>>) = super::combine::region_contours_exact(geom, &super::IDENTITY, tol).into_iter().unzip();
    // The sweep decides its turns on each ring point's exact grid position (as native's tessellator,
    // a module of the same scanner, does); the mesh keeps the f32 points.
    let mut sweep: Vec<[f64; 2]> = Vec::new();
    for (i, outline) in outlines.iter().enumerate() {
        if let Some(r) = feather_figure(outline, &mut verts, &mut tris, false) {
            sweep.resize(verts.len(), [0.0; 2]);
            for (j, p) in outline.iter().enumerate() {
                sweep[r.0 as usize + j] = exact.get(i).and_then(|e| e.get(j).copied()).unwrap_or([p[0] as f64, p[1] as f64]);
            }
            rings.push(r);
        }
    }
    let mut dev: Vec<([f32; 2], f32)> = verts
        .iter()
        .map(|v| {
            // The realization VS (captured from d2d1, fx7k/rsn1/rvs.asm) displaces in *device* space by
            // the unit AA direction × aaWidth, so the feather stays aaWidth pixels wide at any draw scale.
            // The vertex stores the extruded point `pos + off` (f32); the VS transforms both (dp3, f32),
            // takes `d = E − P`, and adds `d·k·(0.5/|d|)` (0 below |d| = 1e-4), where `k = cb0[2].w` is
            // aaWidth/0.5. Each op rounds in f32 as WARP's SSE shader does (no FMA): a facet vertex near a
            // 1/256 snap boundary needs native's exact bits, or one pixel's coverage flips by an LSB.
            let (m0, m1, m2, m3, m4, m5) = (m[0] as f32, m[1] as f32, m[2] as f32, m[3] as f32, m[4] as f32, m[5] as f32);
            let dp3 = |q: [f32; 2]| [q[0] * m0 + q[1] * m2 + m4, q[0] * m1 + q[1] * m3 + m5];
            let p = dp3(v.pos);
            let e = dp3([v.pos[0] + v.off[0], v.pos[1] + v.off[1]]);
            let d = [e[0] - p[0], e[1] - p[1]];
            let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
            let s = if len < 1e-4 { 0.0 } else { 0.5 / len };
            let k = aa_width * 2.0;
            let pos = [d[0] * k * s + p[0], d[1] * k * s + p[1]];
            (pos, v.cov)
        })
        .collect();
    // One AA tessellator sweep for the interior where it runs clean, else the trapezoid fallback. Native
    // sweeps the UNDISPLACED rings and the VS displaces afterwards, so an interior chord whose ends move
    // into the band can overlap the fringe (native's sr-4 mesh: an outer-corner chord lands at y 9.848,
    // past the hole edge at 9.5, and covers the fringe's half pixel in full).
    let undisplaced: Vec<([f32; 2], f32)> = verts.iter().map(|v| (v.pos, v.cov)).collect();
    if rings.is_empty() || !(aatess_interior(&rings, &undisplaced, Some(&sweep), true, &mut tris) || aatess_interior(&rings, &dev, None, true, &mut tris)) {
        winding_interior(&rings, &mut dev, &mut tris);
    }
    (dev, tris)
}

/// Build the feather mesh for `geom` (create space), displace by `AA_WIDTH`, transform by `m`, and
/// rasterize each triangle through the WARP triangle pipeline with coverage in the alpha channel.
pub(crate) fn fill_feather(buf: &mut PixelBuffer, geom: &Geometry, premul: [f32; 4], m: &[f64; 6], blend: Blend, aa_width: f32) {
    fill_feather_tol(buf, geom, premul, m, blend, aa_width, super::DEFAULT_TOLERANCE, warp::Cull::None)
}

/// [`fill_feather`] flattened at a realization's own `tolerance` (`CScannerModuleD::Set`, `CreationParams+8`).
#[allow(clippy::too_many_arguments)]
fn fill_feather_tol(buf: &mut PixelBuffer, geom: &Geometry, premul: [f32; 4], m: &[f64; 6], blend: Blend, aa_width: f32, tolerance: f32, cull: warp::Cull) {
    let (dev, tris) = feather_device(geom, m, aa_width, tolerance);
    // Carry `premultiplied_colour · coverage` as one attribute, exactly the realization VS's `o1`, so
    // the outer ring (cov 0) is all zero and the plane interpolates `premul·cov` directly (premul path
    // in `warp::fill_triangle`).
    let vert = |i: u32| {
        let (pos, cov) = dev[i as usize];
        Vert { pos, rgba: premul.map(|c| c * cov) }
    };
    for t in &tris {
        warp::fill_triangle_cull(buf, [vert(t[0]), vert(t[1]), vert(t[2])], blend, true, cull);
    }
}

/// `DrawGeometryRealization` through native's non-opaque path: the mesh is drawn white (opacity =
/// the stroke's opacity `modifier`) into an A8 coverage intermediate — source-over, or saturating ADD
/// under any other primitive blend, both in float — and the brush (`src(px, py)`, premultiplied BGRA 0..255) is then
/// composited once through that coverage under `blend`. Where feather triangles overlap (a band
/// thinner than the feather) the coverage accumulates before the brush is applied.
///
/// The intermediate is a 512² tile (r19 24358: the viewport of a 16×64 target's mask draw is 512²)
/// laid from the draw's loose bounds — the realized shape's bounds through the transform, 1 px out,
/// cut to the target and widened to whole pixels. Each tile draws the mesh with the transform's
/// translation moved to the tile origin (in f32) and clips and maps it on the tile's own viewport.
#[allow(clippy::too_many_arguments)]
fn fill_feather_mask(
    buf: &mut PixelBuffer,
    geom: &Geometry,
    m: &[f64; 6],
    blend: Blend,
    aa_width: f32,
    tolerance: f32,
    cull: warp::Cull,
    modifier: f32,
    mut src: impl FnMut(f32, f32) -> [f32; 4],
) {
    const TILE: usize = 512;
    let (w, h) = (buf.desc.width as usize, buf.desc.height as usize);
    let [l, t, r, b] = geom.get_bounds(&super::IDENTITY);
    if r < l || b < t {
        return;
    }
    let mf = m.map(|v| v as f32);
    let corners = [[l, t], [r, t], [l, b], [r, b]].map(|[x, y]| [x * mf[0] + y * mf[2] + mf[4], x * mf[1] + y * mf[3] + mf[5]]);
    let lo = |k: usize| corners.iter().fold(f32::INFINITY, |a, p| a.min(p[k])) - 1.0;
    let hi = |k: usize| corners.iter().fold(f32::NEG_INFINITY, |a, p| a.max(p[k])) + 1.0;
    let clip = [lo(0).max(0.0), lo(1).max(0.0), hi(0).min(w as f32), hi(1).min(h as f32)];
    if !(clip[2] > clip[0] && clip[3] > clip[1]) {
        return;
    }
    let [x0, y0, x1, y1] = [clip[0].floor(), clip[1].floor(), clip[2].ceil(), clip[3].ceil()].map(|v| v as usize);
    let add = blend != Blend::SourceOver;
    for ty in (y0..y1).step_by(TILE) {
        for tx in (x0..x1).step_by(TILE) {
            let mut mt = *m;
            mt[4] = (mf[4] - tx as f32) as f64;
            mt[5] = (mf[5] - ty as f32) as f64;
            let (dev, tris) = feather_device(geom, &mt, aa_width, tolerance);
            let mut mask = vec![0u8; TILE * TILE];
            for t in &tris {
                let pos = t.map(|i| dev[i as usize].0);
                let cov = t.map(|i| dev[i as usize].1 * modifier);
                // Every blend clips to the tile's viewport, COPY included: 27012's COPY stroke realization
                // crossing x = 0 is one LSB off on 5 px when its triangles stay whole (VM probes).
                warp::visit_triangle_brush_cov(TILE, TILE, pos, cov, true, cull, |x, y, cv| {
                    // The A8 target blends in float — the interpolated coverage over the stored value
                    // read back as d/255 — and rounds once on write; not the BGRA target's fused
                    // 8-bit kernel (r21 26529: the mask holds 106 where that kernel gives 105 and 107).
                    let d = f32::from(mask[y * TILE + x]) / 255.0;
                    let r = if add { cv + d } else { cv + d * (1.0 - cv) };
                    mask[y * TILE + x] = (r.clamp(0.0, 1.0) * 255.0).round_ties_even() as u8;
                });
            }
            for y in ty..(ty + TILE).min(y1) {
                for x in tx..(tx + TILE).min(x1) {
                    let c = mask[(y - ty) * TILE + (x - tx)];
                    if c != 0 {
                        warp::composite_cov(buf, x, y, src(x as f32 + 0.5, y as f32 + 0.5), f32::from(c) / 255.0, blend);
                    }
                }
            }
        }
    }
}

/// `2^⌈log2(v)⌉` for `v > 0` — the smallest power of two ≥ `v` — computed exactly from the f32 bits
/// (no `log2`/`exp2` libm call), so it is bit-identical on every platform. Equivalent to the former
/// `v.log2().ceil().exp2()` but without the cross-platform transcendental: a power of two keeps its
/// exponent, anything else rounds the exponent up by one.
fn pow2_ceil_log2(v: f32) -> f32 {
    if !(v > 0.0) {
        return 0.0;
    }
    let bits = v.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127; // v ∈ [2^exp, 2^(exp+1))
    let k = if bits & 0x007F_FFFF == 0 { exp } else { exp + 1 }; // exact power of two → exp, else next
    // 2^k as f32 (k stays in the normal range for realistic em/300 ratios).
    f32::from_bits((((k + 127).clamp(1, 254)) as u32) << 23)
}

/// Outline-text realization feather (the text-outline realizer's cached geometry, `texte2e-04`): the glyph
/// mesh native caches for an opaque solid run, baked on a per-glyph **1/8-unit fixed-point** grid in
/// the glyph's cache space and mapped to device by the outline-text VS ([`feather_fixed_device`]).
/// `aa_width` is the device feather half-width: [`AA_WIDTH`], or 0 for an aliased run, whose collapsed fringe leaves the
/// hard-edged interior (VM DDI capture, r25 30063: an aliased OUTLINE-mode run draws the cached mesh, its constant AA 0).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_feather_fixed(buf: &mut PixelBuffer, geom: &Geometry, premul: [f32; 4], lin: [f32; 4], origin: [f32; 2], blend: Blend, aa_width: f32) {
    let turned = SIDEWAYS_CACHE.with(|c| c.get()).map(|k| (sideways(geom, k), k));
    let (geom, lin) = match &turned {
        Some((g, k)) => {
            let (a, b, c, d) = (lin[2], lin[3], -lin[0], -lin[1]);
            (g, [a, b, c + k * a, d + k * b])
        }
        None => (geom, lin),
    };
    let (dev, tris) = feather_fixed_device(geom, lin, origin, aa_width);
    let vert = |i: u32| {
        let (pos, cov) = dev[i as usize];
        Vert { pos, rgba: premul.map(|c| c * cov) }
    };
    // The cached outline-text draw culls as a realization draw does: back faces under a non-negative
    // transform determinant, front faces under a mirroring one (`lin` is the transform times em/bucket, so
    // its determinant has the transform's sign). A stroke thinner than its feather turns the triangles whose
    // displaced vertices fold over, and those are not drawn (r25 30178: Yu Gothic UI Light U+2122 at 12 px,
    // its 0.48 px stems).
    let cull = if lin[3] * lin[0] - lin[2] * lin[1] >= 0.0 { warp::Cull::Back } else { warp::Cull::Front };
    for t in &tris {
        let vp = FEATHER_VIEWPORT.with(|c| c.get()).unwrap_or([0.0, 0.0, buf.desc.width as f32, buf.desc.height as f32]);
        warp::fill_triangle_cull_vp(buf, [vert(t[0]), vert(t[1]), vert(t[2])], blend, true, cull, vp);
    }
}

// Provenance: native's glyph-outline renderer (glyph-run init) and text-outline realizer (cached geometry).
pub(crate) fn colr_outline_bucket(device_em: f32) -> f32 { 75.0 * pow2_ceil_log2(device_em / 75.0) }

thread_local! {
    static SIDEWAYS_CACHE: std::cell::Cell<Option<f32>> = const { std::cell::Cell::new(None) };
}

/// While alive, the cached outline mesh is built for the glyph turned a quarter to the left and drawn turned back: an
/// upright glyph of a vertical layout is a sideways run to DirectWrite (the run turned a quarter right), so d2d1 caches
/// its outline in that turned frame, where the fixed-point shift and the 1/8 snap take the other bound (VM DDI capture,
/// r25 30046: Segoe UI Emoji's flag upright in a column, `lin` [0, s, -s, 0], the mesh our ring turned, its left edge
/// 0.026 px right of an upright cache's). `Some(k)` turns it; a nonzero `k` is the oblique simulation's slant, which the
/// draw's transform carries as a vertical shear but DirectWrite applies to the sideways outline, so the cache holds the
/// turned glyph slanted, x − k·y, and the shear leaves the mesh transform (VM r25 30046 probes: the flag oblique).
pub(crate) struct SidewaysCache(Option<f32>);

impl SidewaysCache {
    pub(crate) fn set(k: Option<f32>) -> Self {
        SidewaysCache(SIDEWAYS_CACHE.with(|c| c.replace(k)))
    }
}

impl Drop for SidewaysCache {
    fn drop(&mut self) {
        SIDEWAYS_CACHE.with(|c| c.set(self.0));
    }
}

/// `geom` turned a quarter, (x, y) to (y, -x) (a quarter to the left on the y-down device), then slanted by `k`: x − k·y.
/// ponytail: an arc only turns; glyph outlines have none.
fn sideways(geom: &Geometry, k: f32) -> Geometry {
    let Geometry::Path { fill, figures } = geom else { return geom.clone() };
    let t = |p: &Point| -> Point { [p[1] + k * p[0], -p[0]] };
    let figures = figures
        .iter()
        .map(|f| super::Figure {
            start: t(&f.start),
            segments: f
                .segments
                .iter()
                .map(|sg| match sg {
                    super::Segment::Line(p) => super::Segment::Line(t(p)),
                    super::Segment::Quadratic(c, p) => super::Segment::Quadratic(t(c), t(p)),
                    super::Segment::Cubic(a, c, p) => super::Segment::Cubic(t(a), t(c), t(p)),
                    super::Segment::Arc { to, radius, rotation, clockwise, large } => {
                        super::Segment::Arc { to: t(to), radius: *radius, rotation: *rotation - 90.0, clockwise: *clockwise, large: *large }
                    }
                })
                .collect(),
            ..f.clone()
        })
        .collect();
    Geometry::Path { fill: *fill, figures }
}

/// Fill `geom`'s cached outline feather mesh mapped by `m` with the premultiplied colour `rgba` (RGBA, 0..1). Each
/// vertex carries the colour times its feather coverage, so a texel is the colour times its float coverage,
/// rounded once. A solid colour-glyph leaf is the same cached outline draw as large text (VM DDI capture of r25
/// 30994: the outline-text mesh instances, `lin` the leaf transform's linear part), so it takes [`fill_feather_fixed`].
thread_local! {
    /// The viewport [`fill_feather_fixed`] draws through when it is not the buffer ([`fill_colr_outline`]'s `vp`).
    static FEATHER_VIEWPORT: std::cell::Cell<Option<[f32; 4]>> = const { std::cell::Cell::new(None) };
}

/// `vp` is the viewport the leaf is drawn through when it is not `buf` itself (a composite group's atlas cell).
pub(crate) fn fill_colr_outline(buf: &mut PixelBuffer, geom: &Geometry, m: &[f64; 6], rgba: [f32; 4], vp: Option<[f32; 4]>) {
    let lin = [m[0] as f32, m[1] as f32, m[2] as f32, m[3] as f32];
    let prev = FEATHER_VIEWPORT.with(|c| c.replace(vp));
    fill_feather_fixed(buf, geom, rgba, lin, [m[4] as f32, m[5] as f32], Blend::SourceOver, AA_WIDTH);
    FEATHER_VIEWPORT.with(|c| c.set(prev));
}

/// The outline-text feather mesh in device space. `geom` is the glyph in its cache space (em = bucket,
/// origin at the glyph origin); `lin` maps cache to device and `origin` is the glyph origin's device
/// position. Direct2D caches the glyph's AA tessellation once per bucket: shifted so its bounds min sits
/// at (1,1) on a 1/8 fixed-point grid, or in float in its own frame when it is too large for that grid.
/// The outline vertex shader maps the cached positions by `lin` plus the shift back, then displaces each
/// by `aa / |lin·n|` along `lin·n`, n being the cache-space AA direction. Under a shear `lin·n` is not the
/// device edge normal, which is why the feather is built in cache space.
/// Reverse each raw ring whose ink lies on the side the feather fringes outward: the glyph's nonzero winding is
/// inked a hair outside its longest edge and not inside. TrueType winds every contour with its ink on the feather's
/// inner side; a font that winds some the other way (Segoe UI Emoji's face rings, inline COLR test fonts) would fold
/// its fringe inward and fill bi-level. Holes and overlapping contours keep their order.
fn orient_rings(rings: &mut [Vec<Point>]) {
    let winding = |rings: &[Vec<Point>], p: [f64; 2]| -> i32 {
        let mut w = 0;
        for c in rings {
            for i in 0..c.len() {
                let (a, b) = (c[i].map(f64::from), c[(i + 1) % c.len()].map(f64::from));
                let cross = (b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1]);
                if a[1] <= p[1] && b[1] > p[1] && cross > 0.0 { w += 1; }
                if a[1] > p[1] && b[1] <= p[1] && cross < 0.0 { w -= 1; }
            }
        }
        w
    };
    let flip: Vec<bool> = rings.iter().map(|c| {
        let n = c.len();
        let len2 = |i: usize| { let (a, b) = (c[i], c[(i + 1) % n]); (b[0] - a[0]) as f64 * (b[0] - a[0]) as f64 + (b[1] - a[1]) as f64 * (b[1] - a[1]) as f64 };
        let Some(i) = (0..n).max_by(|&i, &j| len2(i).total_cmp(&len2(j))) else { return false };
        let (a, b) = (c[i].map(f64::from), c[(i + 1) % n].map(f64::from));
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = (dx * dx + dy * dy).sqrt();
        if !(len > 0.0) { return false; }
        let m = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
        // `feather_figure` offsets the outer ring by (dy, -dx) per edge.
        let e = 1e-3 / len;
        let outer = winding(rings, [m[0] + dy * e, m[1] - dx * e]) != 0;
        let inner = winding(rings, [m[0] - dy * e, m[1] + dx * e]) != 0;
        outer && !inner
    }).collect();
    for (c, f) in rings.iter_mut().zip(flip) {
        if f {
            c.reverse();
        }
    }
}

/// The bounds the text-outline realizer shifts a glyph's cache by: the box of the path's points, Bezier controls
/// included, not its curves' extrema. They agree for an upright TrueType glyph, whose extrema are on-curve points;
/// a slanted one (the oblique simulation) reaches past its curves at its controls (VM: Leelawadee UI 'c' bold and
/// oblique at em 126.55, every cached vertex 9/8 unit left of the extrema-bounds grid).
fn control_bounds(geom: &Geometry) -> [f32; 4] {
    let Geometry::Path { figures, .. } = geom else { return geom.get_bounds(&super::IDENTITY) };
    let mut b = [f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
    let mut acc = |p: &Point| b = [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])];
    for f in figures {
        acc(&f.start);
        for sg in &f.segments {
            match sg {
                super::Segment::Line(p) => acc(p),
                super::Segment::Quadratic(c, p) => { acc(c); acc(p); }
                super::Segment::Cubic(a, c, p) => { acc(a); acc(c); acc(p); }
                super::Segment::Arc { to, .. } => acc(to),
            }
        }
    }
    b
}

fn feather_fixed_device(geom: &Geometry, lin: [f32; 4], origin: [f32; 2], aa_width: f32) -> (Vec<([f32; 2], f32)>, Vec<[u32; 3]>) {
    // The glyph's control-point bounds ([`control_bounds`]) decide the vertex format: a glyph at least 8190
    // units wide or tall keeps float positions in its own frame; any other is shifted so its bounds
    // min sits at (1,1) and stored on the 1/8 fixed-point grid.
    let b = control_bounds(geom);
    if !b.iter().all(|v| v.is_finite()) {
        return (Vec::new(), Vec::new());
    }
    let float = (b[0] <= b[2] && b[2] - b[0] >= 8190.0) || (b[1] <= b[3] && b[3] - b[1] >= 8190.0);
    let shift = if float { [0.0, 0.0] } else { [1.0 - b[0], 1.0 - b[1]] };
    // The realizer's tessellator sweeps the shifted glyph through the scanner (tolerance 0.25), which resolves its
    // region (nonzero winding, self-intersections split, a loop the embolden folds out of a concave corner dropped),
    // and only then snaps each boundary vertex to the 1/8 grid (VM textout meshes, Leelawadee UI 'a' bold: the
    // two-vertex spur at its bowl joint is not in native's ring; Calibri 'O', Leelawadee 'c' and 'a', every vertex).
    // Should the sweep fail, the raw rings are fed through, wound ink-inward.
    let m = [1.0, 0.0, 0.0, 1.0, shift[0], shift[1]];
    let (mut region, exact): (Vec<Vec<Point>>, Vec<Vec<[f64; 2]>>) =
        super::combine::region_contours_exact(geom, &m, super::DEFAULT_TOLERANCE as f64).into_iter().unzip();
    let resolved = !region.is_empty();
    if !resolved {
        region = outlines_transformed(geom, &m, super::DEFAULT_TOLERANCE as f64);
    }
    let mut verts: Vec<FVert> = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut rings: Vec<(u32, usize)> = Vec::new();
    // Two boundary vertices the snap lands on one grid point are one figure vertex, as the realizer's figure
    // builder keeps only a vertex unequal to the one before it (Leelawadee UI 'a' bold: its bowl joint and the
    // crossing that closes the embolden's loop there, both at (386, 577)/8). `at[k][i]` is the figure vertex
    // region vertex `i` of ring `k` lands on.
    let mut snapped: Vec<Vec<Point>> = Vec::with_capacity(region.len());
    let mut at: Vec<Vec<usize>> = Vec::with_capacity(region.len());
    for poly in &region {
        let (mut ring, mut to): (Vec<Point>, Vec<usize>) = (Vec::with_capacity(poly.len()), Vec::with_capacity(poly.len()));
        for p in poly {
            let q = if float { *p } else { p.map(|v| (v * 8.0 + 0.5).floor() / 8.0) };
            if ring.last() != Some(&q) {
                ring.push(q);
            }
            to.push(ring.len() - 1);
        }
        while ring.len() > 1 && ring.first() == ring.last() {
            ring.pop();
            to.iter_mut().filter(|i| **i == ring.len()).for_each(|i| *i = 0);
        }
        snapped.push(ring);
        at.push(to);
    }
    if !resolved {
        orient_rings(&mut snapped);
    }
    for poly in &snapped {
        if let Some(r) = feather_figure(poly, &mut verts, &mut tris, true) {
            rings.push(r);
        }
    }
    // The cached shift undone, composed with the glyph matrix.
    let (ox, oy) = (-shift[0], -shift[1]);
    let dev_org = [(ox * lin[0] + lin[2] * oy) + origin[0], (ox * lin[1] + lin[3] * oy) + origin[1]];
    let mut dev: Vec<([f32; 2], f32)> = verts
        .iter()
        .map(|v| {
            let (c, n) = (v.pos, v.off);
            let mut pos = [(c[0] * lin[0] + c[1] * lin[2]) + dev_org[0], (c[0] * lin[1] + c[1] * lin[3]) + dev_org[1]];
            // The VS skips an inner vertex whose summed normals nearly cancel (|n|² < 1e-4).
            if n[0] * n[0] + n[1] * n[1] >= 1e-4 {
                let d = [n[0] * lin[0] + n[1] * lin[2], n[0] * lin[1] + n[1] * lin[3]];
                let s = (1.0 / (d[0] * d[0] + d[1] * d[1]).sqrt()) * aa_width;
                pos = [d[0] * s + pos[0], d[1] * s + pos[1]];
            }
            (pos, v.cov)
        })
        .collect();
    // Interior fill. Native draws fringe+interior in ONE AA tessellator CScanner sweep, so every pixel
    // belongs to exactly one triangle. Our fringe (`feather_figure`) + a SEPARATE interior sweep
    // (`winding_interior`, trapezoids whose band boundaries add Steiner points on the inner-ring edges)
    // leave a 1-px seam hole wherever a ring edge passes through a pixel centre (large-glyph outline text:
    // Arial W em 4000/8000 flipped interior-edge pixels to full white). Tessellating the interior ring
    // through the monotone AA tessellator sweep ([`aatess`]) instead uses ONLY the ring vertices — no
    // Steiner points — so the interior triangles share the inner-ring edges with the fringe exactly and
    // the seam closes. The interior is uniformly cov=1, so this changes no interior pixel, only removes
    // the holes (em 500 stays byte-exact). A single ring only; counters/holes need the multi-contour
    // winding sweep (not yet ported), so they keep `winding_interior`.
    // As the geometry realization, the sweep runs on the undisplaced (cache-space) rings; the VS displaces
    // each vertex afterwards, so a thin part whose displaced ring folds keeps native's triangles.
    let undisplaced: Vec<([f32; 2], f32)> = verts.iter().map(|v| (v.pos, v.cov)).collect();
    let first = tris.len();
    // The cache's tessellator sweeps the region on the scanner's grid and stores the snapped figure vertices, so
    // the triangles come from the exact rings and land on the figure vertices their corners snap to; one whose
    // corners merge is dropped (r25 30923, Leelawadee UI Semilight 'ย' at em 0.81: its counter snaps to a spur,
    // and native's interior still fans the outer ring onto it). A sweep of the raw rings keeps the snapped ones.
    let ids: Vec<Vec<u32>> = rings.iter().zip(&at).map(|(&(base, _), to)| to.iter().map(|&i| base + i as u32).collect()).collect();
    let rings_big: Vec<(u32, usize)> = rings.iter().copied().filter(|r| r.1 >= 3).collect();
    let swept = if resolved && rings.len() == region.len() {
        sweep_interior(&rings, &exact, &ids, true, &mut tris)
    } else {
        !rings_big.is_empty() && aatess_interior(&rings_big, &undisplaced, None, resolved, &mut tris)
    };
    if swept {
        // The glyph cache's tessellator then flips collapsed interior triangles against their outline
        // neighbours, on the cached positions.
        let pos: Vec<[f32; 2]> = verts.iter().map(|v| v.pos).collect();
        fixup_collapsed_interior(&rings, &pos, &mut tris[first..]);
        let interior = tris.split_off(first);
        tris = cache_draw_order(&rings, &interior);
    } else {
        if !(!rings_big.is_empty() && aatess_interior(&rings_big, &dev, None, resolved, &mut tris)) {
            winding_interior(&rings_big, &mut dev, &mut tris);
        }
        // Neither fallback is native's triangulation, and both sweep the displaced rings, so their triangles
        // carry no fold to cull: wind each front-facing so the culled draw keeps all of them.
        let front = lin[3] * lin[0] - lin[2] * lin[1] >= 0.0;
        for t in &mut tris[first..] {
            let [a, b, c] = t.map(|i| dev[i as usize].0);
            let area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
            if area != 0.0 && (area > 0.0) != front {
                t.swap(1, 2);
            }
        }
    }
    (dev, tris)
}

/// Fill the figures' interior rings through the AA tessellator single sweep ([`crate::geometry::aatess`])
/// so the interior triangles share the inner-ring edges with the fringe (no seam T-junction). All rings
/// are swept together (nonzero winding, so counters/holes are left empty). Returns `false` — leaving the
/// caller to use `winding_interior` — when the sweep bails or its triangles do not cleanly tile the
/// filled region (a mapping miss, or total area off by >0.1% of the net signed area), so an anomalous
/// glyph outline never silently mis-fills. On `true`, the interior triangles have been pushed to `tris`.
/// `region`: the rings are a resolved region's contours, swept by [`crate::geometry::aatess::tessellate_region`].
/// `exact`: per-vertex sweep positions overriding `dev` (the rings' exact grid positions), if known.
fn aatess_interior(rings: &[(u32, usize)], dev: &[([f32; 2], f32)], exact: Option<&[[f64; 2]]>, region: bool, tris: &mut Vec<[u32; 3]>) -> bool {
    let contours: Vec<Vec<[f64; 2]>> = rings
        .iter()
        .map(|&(base, n)| (0..n).map(|i| { let k = base as usize + i; exact.map_or_else(|| { let p = dev[k].0; [p[0] as f64, p[1] as f64] }, |e| e[k]) }).collect())
        .collect();
    let ids: Vec<Vec<u32>> = rings.iter().map(|&(base, n)| (0..n as u32).map(|i| base + i).collect()).collect();
    sweep_interior(rings, &contours, &ids, region, tris)
}

/// [`aatess_interior`] over sweep contours whose vertices land on the ring vertices `ids` (contour `k` on ring
/// `rings[k]`, `ids[k][i]` the vertex its point `i` lands on, several points of a contour possibly on one): a
/// triangle whose corners land on fewer than three vertices is dropped.
fn sweep_interior(rings: &[(u32, usize)], contours: &[Vec<[f64; 2]>], ids: &[Vec<u32>], region: bool, tris: &mut Vec<[u32; 3]>) -> bool {
    let crefs: Vec<&[[f64; 2]]> = contours.iter().map(|c| c.as_slice()).collect();
    let swept = if region { crate::geometry::aatess::tessellate_region(&crefs) } else { crate::geometry::aatess::tessellate_contours(&crefs) };
    if swept.is_empty() {
        return false;
    }
    // A point two contours share (resolved loops touching at a crossing) has one ring vertex per
    // contour, each with that contour's feather offset; a triangle takes the vertex of the contour its
    // other corners belong to.
    let mut idx: std::collections::HashMap<[u64; 2], Vec<(usize, u32)>> = std::collections::HashMap::new();
    for (ci, c) in contours.iter().enumerate() {
        for (i, p) in c.iter().enumerate() {
            let e = idx.entry([p[0].to_bits(), p[1].to_bits()]).or_default();
            if !e.contains(&(ci, ids[ci][i])) {
                e.push((ci, ids[ci][i]));
            }
        }
    }
    let tri_area = |t: &[[f64; 2]; 3]| 0.5 * ((t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[2][0] - t[0][0]) * (t[1][1] - t[0][1])).abs();
    // Net filled area = |sum of signed contour areas| (outer minus counters), nonzero-winding.
    let ring_area = 0.5
        * contours
            .iter()
            .map(|c| { let m = c.len(); (0..m).map(|i| { let a = c[i]; let b = c[(i + 1) % m]; a[0] * b[1] - b[0] * a[1] }).sum::<f64>() })
            .sum::<f64>()
            .abs();
    let mut mapped: Vec<[u32; 3]> = Vec::with_capacity(swept.len());
    let mut area = 0.0;
    for t in &swept {
        let mut cands = [&[][..]; 3];
        for (k, p) in t.iter().enumerate() {
            match idx.get(&[p[0].to_bits(), p[1].to_bits()]) {
                Some(c) => cands[k] = c.as_slice(),
                None => return false, // sweep produced a point not on any ring — bail
            }
        }
        // A corner several contours share takes the copy of a contour the triangle's single-contour
        // corners lie on, else of one every corner can take, else its first: a triangle lies against its
        // own contours, and a copy of a contour it does not touch would push the corner into the wrong
        // feather.
        let unique: Vec<usize> = cands.iter().filter(|c| c.len() == 1).map(|c| c[0].0).collect();
        // A copy joined by its ring's edge to another corner's copy on the same ring.
        let edge = |k: usize, ci: usize, vix: u32| {
            let (base, n) = rings[ci];
            let li = vix - base;
            (0..3).filter(|&o| o != k).any(|o| {
                cands[o].iter().any(|&(oc, ov)| oc == ci && { let lo = ov - base; (lo + 1) % n as u32 == li || (li + 1) % n as u32 == lo })
            })
        };
        let vi: [u32; 3] = core::array::from_fn(|k| {
            let c = cands[k];
            c.iter()
                .find(|&&(ci, vix)| c.len() > 1 && edge(k, ci, vix))
                .or_else(|| c.iter().find(|&&(ci, _)| unique.contains(&ci)))
                .or_else(|| c.iter().find(|&&(ci, _)| cands.iter().all(|o| o.iter().any(|&(oc, _)| oc == ci))))
                .unwrap_or(&c[0])
                .1
        });
        area += tri_area(t);
        if vi[0] != vi[1] && vi[1] != vi[2] && vi[0] != vi[2] {
            mapped.push(vi);
        }
    }
    if (area - ring_area).abs() > ring_area * 1e-3 + 1e-3 {
        return false; // triangles do not tile the filled region (gap/overlap) — bail
    }
    tris.extend(mapped);
    true
}

/// The outline-text cache's triangles in the order its draw rasterizes them, as the cached feather mesh from
/// [`feather_figure`] plus the interior sweep's triangles `interior`. The fringe instances walk each ring's
/// vertices, drawing at vertex `i` the outline triangle of the edge `i → i+1` (to its apex, as `[apex, i, i+1]`),
/// the edge's two feather strips and the corner at `i+1`; the interior instance then draws the triangles that
/// have no outline edge, in sweep order. With overlapping translucent fringes the order moves an 8-bit result
/// (r25 30714, Times New Roman 'e' at 19 px, 144 dpi).
fn cache_draw_order(rings: &[(u32, usize)], interior: &[[u32; 3]]) -> Vec<[u32; 3]> {
    let ring_of = |v: u32| rings.iter().copied().find(|&(b, n)| v >= b && v < b + n as u32);
    let next_on_ring = |a: u32, b: u32| ring_of(a).is_some_and(|(base, n)| b >= base && b < base + n as u32 && b - base == (a - base + 1) % n as u32);
    // An outline triangle is keyed by its edge's start; a second one on that edge draws with the interior ones.
    let mut outline: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut rest: Vec<[u32; 3]> = Vec::new();
    for t in interior {
        let edge = (0..3).find_map(|k| {
            let (a, b, c) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
            if next_on_ring(a, b) { Some((a, c)) } else if next_on_ring(b, a) { Some((b, c)) } else { None }
        });
        match edge {
            Some((e, apex)) if !outline.contains_key(&e) => {
                outline.insert(e, apex);
            }
            _ => rest.push(*t),
        }
    }
    let mut out = Vec::with_capacity(interior.len() + rings.iter().map(|r| 3 * r.1).sum::<usize>());
    for &(base, n) in rings {
        let n = n as u32;
        let (int, out1, out2) = (|i: u32| base + i, |i: u32| base + n + i, |i: u32| base + 2 * n + i);
        for i in 0..n {
            let j = (i + 1) % n;
            if let Some(&apex) = outline.get(&int(i)) {
                out.push([apex, int(i), int(j)]);
            }
            out.push([int(i), out1(i), int(j)]);
            out.push([out1(i), out2(i), int(j)]);
            out.push([out2(i), out1(j), int(j)]);
        }
    }
    out.extend(rest);
    out
}

/// The outline-text glyph cache's post-sweep pass over the interior sweep's triangles `tris` (ring
/// vertex indices into `pos`, the cached positions). The cache stores a triangle that has an outline
/// edge as its edge's start vertex pointing at the apex (`apex[e]` for the edge `e → e+1`), and every
/// other triangle as an interior one. An interior triangle whose inset triangle — each corner moved one
/// unit inward along its outline vertex's summed edge normals — no longer winds positively would invert
/// under the AA displacement, so its shared diagonal with an outline triangle is flipped, while the
/// flip keeps both triangles positive and passes the incircle test, up to 101 times or until the
/// triangle no longer collapses. Each ring is laid out as the cache stores it, `[L, L, F, V1 … L, F, V1,
/// V1]` (L the last vertex, F the first), so a vertex's neighbours and the closing edge come from the
/// padding copies, which never carry an apex. The flipped pairs are replaced in place.
fn fixup_collapsed_interior(rings: &[(u32, usize)], pos: &[[f32; 2]], tris: &mut [[u32; 3]]) {
    // The cache layout: hw index -> ring vertex, and each ring vertex's own (non-padding) hw index.
    let mut hw: Vec<u32> = Vec::new();
    let mut real: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut ring_of: std::collections::HashMap<u32, (u32, usize)> = std::collections::HashMap::new();
    for &(base, n) in rings {
        let off = hw.len() as u32;
        let last = base + n as u32 - 1;
        hw.extend([last, last]);
        for i in 0..n as u32 {
            hw.push(base + i);
            real.insert(base + i, off + 2 + i);
            ring_of.insert(base + i, (base, n));
        }
        hw.extend([base, base + 1.min(n as u32 - 1), base + 1.min(n as u32 - 1)]);
    }
    let p = |h: u32| pos[hw[h as usize] as usize];
    let next_on_ring = |a: u32, b: u32| ring_of.get(&a).is_some_and(|&(base, n)| ring_of.get(&b).is_some_and(|&(bb, _)| bb == base) && b - base == (a - base + 1) % n as u32);
    // The outline triangles (an edge along the ring) set `apex`; the rest are interior.
    let mut apex: Vec<u32> = (0..hw.len() as u32).collect();
    let mut owner: std::collections::HashMap<u32, usize> = std::collections::HashMap::new(); // e -> tris slot
    let mut interior: Vec<usize> = Vec::new();
    for (slot, t) in tris.iter().enumerate() {
        if t[0] == t[1] || t[1] == t[2] || t[2] == t[0] {
            continue;
        }
        match (0..3).find(|&k| next_on_ring(t[k], t[(k + 1) % 3]) || next_on_ring(t[(k + 1) % 3], t[k])) {
            Some(k) => {
                let e = real[&t[k]];
                apex[e as usize] = real[&t[(k + 2) % 3]];
                owner.insert(e, slot);
            }
            None => interior.push(slot),
        }
    }
    let inset = |h: u32| -> [f32; 2] {
        let (a, c, b) = (p(h - 1), p(h), p(h + 1));
        match (unitize([c[0] - a[0], c[1] - a[1]]), unitize([b[0] - c[0], b[1] - c[1]])) {
            (Some(i), Some(o)) => [-(o[1] + i[1]) + c[0], -(-i[0] - o[0]) + c[1]],
            _ => c,
        }
    };
    let orient = |a: [f32; 2], b: [f32; 2], c: [f32; 2]| (c[1] - b[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - b[0]);
    let collapsed = |t: [u32; 3]| orient(inset(t[0]), inset(t[1]), inset(t[2])) <= 0.0;
    // A flip: edge start `e`, which side (`side`), the old apex, the interior triangle's far vertex,
    // and the interior triangle it leaves.
    struct Flip { e: u32, side: bool, old: u32, far: u32, tri: [u32; 3] }
    let flip_ok = |f: &Flip| -> bool {
        let (a, b, d, c) = (p(f.e), p(f.e + 1), p(f.old), p(f.far));
        let (s0, s1, s2) = if f.side { (a, c, d) } else { (b, d, c) };
        if orient(a, b, c) <= 0.0 || orient(s0, s1, s2) <= 0.0 {
            return false;
        }
        let row = |q: [f32; 2]| { let (x, y) = (d[0] - q[0], d[1] - q[1]); [x, y, y * y + x * x] };
        let m = [row(a), row(b), row(c)];
        let det = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * m[0][0] - (m[1][0] * m[2][2] - m[2][0] * m[1][2]) * m[0][1]
            + (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * m[0][2];
        (det as f64) < -0.000001
    };
    for &slot in &interior {
        let mut t = tris[slot].map(|v| real[&v]);
        if !collapsed(t) {
            continue;
        }
        for _ in 0..=100 {
            let mut cands: Vec<Flip> = Vec::new();
            for k in 0..3 {
                let (a, b, far) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
                for side in 0..2u32 {
                    let e = if apex[(a - side) as usize] == b { a - side } else if apex[(b - side) as usize] == a { b - side } else { continue };
                    let old = apex[e as usize];
                    let tri = if side == 1 { [e, far, old] } else { [e + 1, old, far] };
                    let f = Flip { e, side: side == 1, old, far, tri };
                    if flip_ok(&f) {
                        cands.push(f);
                    }
                    break;
                }
            }
            if cands.is_empty() {
                break;
            }
            let done = cands.iter().position(|f| !collapsed(f.tri));
            let f = &cands[done.unwrap_or(0)];
            apex[f.e as usize] = f.far;
            t = f.tri;
            if let Some(&o) = owner.get(&f.e) {
                tris[o] = [hw[f.e as usize], hw[(f.e + 1) as usize], hw[f.far as usize]];
            }
            if done.is_some() {
                break;
            }
        }
        tris[slot] = t.map(|h| hw[h as usize]);
    }
}

/// One feather-mesh vertex in create space: position, AA displacement direction `a − pos`, coverage.
#[derive(Clone, Copy)]
struct FVert {
    pos: [f32; 2],
    off: [f32; 2],
    cov: f32,
}

/// Each figure's closed outline polygon in create (DIP) space (identity transform), deduped as
/// The outline dedup pass drops consecutive-equal points and the closing wrap.
fn outlines(geom: &Geometry) -> Vec<Vec<Point>> {
    outlines_transformed(geom, &super::IDENTITY, super::DEFAULT_TOLERANCE as f64)
}

/// The realization sweep's flattening tolerance (`CScannerModuleD::Set`): the realization's own
/// tolerance as given, but no finer than `3e-12` of the shape's larger extent, which also stands in for
/// a non-positive or NaN tolerance (`maxsd` keeps the extent term when either is NaN).
fn scan_tolerance(geom: &Geometry, tolerance: f32) -> f64 {
    let [l, t, r, b] = geom.get_bounds(&super::IDENTITY);
    let (w, h) = (r as f64 - l as f64, b as f64 - t as f64);
    let ext = if w > h { w } else { h };
    let floor = if ext.is_finite() { ext } else { 1.0 } * 3e-12;
    let tol = tolerance as f64;
    if tol > floor { tol } else { floor }
}

/// `tolerance`: the sweep's flattening tolerance in DIPs, already resolved ([`scan_tolerance`]).
fn outlines_transformed(geom: &Geometry, m: &[f32; 6], tolerance: f64) -> Vec<Vec<Point>> {
    let mut out = Vec::new();
    // Every point reaches the sweep on its 2^20 grid (`CScannerModuleD::Set(…, 20)`), line ends too: a
    // rounded rect's zero-length bottom edge then folds onto its corner cubic's end (r15 20821).
    let snap = |p: Point| [((p[0] as f64 * SWEEP_SCALE).round() / SWEEP_SCALE) as f32, ((p[1] as f64 * SWEEP_SCALE).round() / SWEEP_SCALE) as f32];
    // The sweep fills the shape: every figure whole, a FORCE_UNSTROKED segment included.
    let mut figs = Vec::new();
    geom.fill_figures(m, &mut figs);
    for (start, prims, _closed, _filled) in figs {
        let mut poly = vec![snap(start)];
        for pr in &prims {
            match *pr {
                Prim::Line(_, e) => poly.push(snap(e)),
                Prim::Cubic(p0, p1, p2, p3) => flatten_cubic(p0, p1, p2, p3, &mut poly, tolerance),
            }
        }
        dedup_wrap(&mut poly);
        if poly.len() >= 2 {
            out.push(poly);
        }
    }
    out
}

/// Drop consecutive-duplicate points and the closing vertex (first == last).
fn dedup_wrap(poly: &mut Vec<Point>) {
    poly.dedup();
    if poly.len() > 1 && poly.first() == poly.last() {
        poly.pop();
    }
}

/// Emit the feather rim mesh for one deduped closed outline into `verts`/`tris`, and return the
/// interior ring's `(base, n)` vertex range so the caller can triangulate the combined nonzero-winding
/// interior across all figures (holes). Returns `None` for a degenerate figure (under three points, or none
/// for the cache). `cache`: the outline-text cache's mesh template, which keeps a figure of one or two vertices (the snap can
/// fold a small counter onto a segment, which the cache still feathers) and emits the corner triangle at every
/// vertex (its draw culls the reflex corners' back-facing ones); otherwise only convex corners, from three vertices.
fn feather_figure(outline: &[Point], verts: &mut Vec<FVert>, tris: &mut Vec<[u32; 3]>, cache: bool) -> Option<(u32, usize)> {
    let n = outline.len();
    if n < if cache { 1 } else { 3 } {
        return None;
    }
    let base = verts.len() as u32;
    let interior = |i: usize| base + i as u32; // ring [base, base+n)
    let outer1 = |i: usize| base + (n + i) as u32; // ring [base+n, base+2n), at p_i
    let outer2 = |i: usize| base + (2 * n + i) as u32; // ring [base+2n, base+3n), at p_{i+1}

    // Per-vertex unit edge directions and the three offsets.
    let mut int_off = vec![[0.0f32; 2]; n];
    let mut out_off = vec![[0.0f32; 2]; n];
    let mut convex = vec![false; n];
    for i in 0..n {
        let prev = outline[(i + n - 1) % n];
        let cur = outline[i];
        let next = outline[(i + 1) % n];
        let ind = unitize([cur[0] - prev[0], cur[1] - prev[1]]);
        let outd = unitize([next[0] - cur[0], next[1] - cur[1]]);
        if let (Some(id), Some(od)) = (ind, outd) {
            int_off[i] = [-(id[1] + od[1]), id[0] + od[0]];
            out_off[i] = [od[1], -od[0]];
            // Convex-corner test (`out.x·in.y − out.y·in.x ≤ 0`).
            convex[i] = od[0] * id[1] - od[1] * id[0] <= 0.0;
        } else {
            convex[i] = true; // test on zeros is 0 ≤ 0 (corner tri degenerates away)
        }
    }

    verts.reserve(3 * n);
    for i in 0..n {
        verts.push(FVert { pos: outline[i], off: int_off[i], cov: 1.0 });
    }
    for i in 0..n {
        verts.push(FVert { pos: outline[i], off: out_off[i], cov: 0.0 });
    }
    for i in 0..n {
        verts.push(FVert { pos: outline[(i + 1) % n], off: out_off[i], cov: 0.0 });
    }

    // Per-edge feather strip + per-convex-corner triangle.
    for i in 0..n {
        let j = (i + 1) % n;
        tris.push([interior(i), outer1(i), interior(j)]);
        tris.push([outer1(i), outer2(i), interior(j)]);
        if cache || convex[i] {
            let p = (i + n - 1) % n;
            tris.push([outer2(p), outer1(i), interior(i)]);
        }
    }
    // Interior fill is deferred to the caller: the interior rings of ALL figures are triangulated
    // together by a nonzero-winding trapezoidal sweep ([`winding_interior`]), so a glyph with a counter
    // (the "0" ring) leaves the hole empty instead of a per-figure fan filling it solid. The covered
    // pixels equal native's monotone-triangulation interior regardless of the triangulation (all cov=1).
    Some((base, n))
}

/// Triangulate the combined nonzero-winding interior of all figures' (device-space, AA-displaced)
/// interior rings into trapezoids (all `cov = 1`), appending Steiner corner vertices to `dev` and the
/// triangles to `tris`. A horizontal sweep between consecutive ring-vertex y's: within each band every
/// active edge spans it fully, so sort them by x, accumulate winding, and emit one trapezoid per
/// nonzero-winding span. The union of trapezoids is exactly the fill region (holes excluded), so the
/// rasterised interior pixels match native's — the exact triangulation is irrelevant at `cov = 1`.
fn winding_interior(rings: &[(u32, usize)], dev: &mut Vec<([f32; 2], f32)>, tris: &mut Vec<[u32; 3]>) {
    // Non-horizontal ring edges (device positions), plus the sorted set of distinct vertex y's.
    let mut edges: Vec<([f32; 2], [f32; 2])> = Vec::new();
    let mut ys: Vec<f32> = Vec::new();
    for &(base, n) in rings {
        for i in 0..n {
            let a = dev[base as usize + i].0;
            let b = dev[base as usize + (i + 1) % n].0;
            ys.push(a[1]);
            if a[1] != b[1] {
                edges.push((a, b));
            }
        }
    }
    // total_cmp (not partial_cmp().unwrap()) so a non-finite device y — from a nan/inf input coord or
    // transform reaching here — sorts deterministically instead of panicking; the `y1 > y0` band guard
    // below is false for any NaN, so such bands are skipped and realization proceeds on the finite ones.
    ys.sort_by(f32::total_cmp);
    ys.dedup();
    for w in ys.windows(2) {
        let (y0, y1) = (w[0], w[1]);
        if !(y1 > y0) {
            continue;
        }
        let ym = 0.5 * (y0 + y1);
        // Active edges crossing the whole band: (x@ym, x@y0, x@y1, winding dir).
        let mut act: Vec<(f32, f32, f32, i32)> = Vec::new();
        for &(a, b) in &edges {
            let (lo, hi) = (a[1].min(b[1]), a[1].max(b[1]));
            if lo <= y0 && hi >= y1 {
                let xat = |y: f32| a[0] + (y - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
                act.push((xat(ym), xat(y0), xat(y1), if b[1] > a[1] { 1 } else { -1 }));
            }
        }
        act.sort_by(|p, q| p.0.total_cmp(&q.0)); // total_cmp: a NaN span x never panics the sort
        let mut wind = 0i32;
        let mut left: Option<(f32, f32, f32, i32)> = None;
        for e in act {
            let prev = wind;
            wind += e.3;
            if prev == 0 && wind != 0 {
                left = Some(e);
            } else if prev != 0 && wind == 0 {
                if let Some(l) = left.take() {
                    let mut push = |x: f32, y: f32| -> u32 {
                        dev.push(([x, y], 1.0));
                        dev.len() as u32 - 1
                    };
                    let c0 = push(l.1, y0); // left @ y0
                    let c1 = push(e.1, y0); // right @ y0
                    let c2 = push(e.2, y1); // right @ y1
                    let c3 = push(l.2, y1); // left @ y1
                    tris.push([c0, c1, c2]);
                    tris.push([c0, c2, c3]);
                }
            }
        }
    }
}

/// `CPoint2F::Unitize`: normalize to unit length; `None` when the f32 length is below 1e-6 (compared
/// in f64) or NaN. An infinite length passes and scales by `1/inf`, as native does.
fn unitize(v: [f32; 2]) -> Option<[f32; 2]> {
    // d2d1 scales by the reciprocal length (fx7k/rsn1: every baked EXTRUDED point matches `d·(1/|d|)`).
    let len = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if len as f64 >= 1e-6 {
        let inv = 1.0 / len;
        Some([v[0] * inv, v[1] * inv])
    } else {
        None
    }
}

/// Flatten one cubic into `poly` (excluding `p0`, including `p3`) through the realization sweep's
/// grid forward-differencer (`CScanner::AddBeziers`, `combine::curveflat`) at the sweep scale;
/// `tolerance` in DIPs, already resolved.
pub(super) fn flatten_cubic(p0: Point, p1: Point, p2: Point, p3: Point, poly: &mut Vec<Point>, tolerance: f64) {
    let g = |p: Point| [(p[0] as f64 * SWEEP_SCALE).round() as i64, (p[1] as f64 * SWEEP_SCALE).round() as i64];
    let gf = |q: [i64; 2]| [q[0] as f64, q[1] as f64];
    let dev = |q: [i64; 2]| [(q[0] as f64 / SWEEP_SCALE) as f32, (q[1] as f64 / SWEEP_SCALE) as f32];
    let tol = SWEEP_SCALE * tolerance;
    let mut grid = Vec::new();
    super::combine::curveflat::flatten_grid(gf(g(p0)), gf(g(p1)), gf(g(p2)), gf(g(p3)), tol, &mut grid);
    poly.extend(grid.iter().map(|&(q, _)| dev(q)));
}

/// The device transform the draw maps through: the context's, each world element scaled by its axis's f32 DPI
/// factor (native's f32 composition; an f64 product differs at a DPI whose /96 is no f32 value, and a full
/// matrix product would meet an infinite element with the scale's zeros).
fn device_matrix(dc: &DeviceContext) -> [f64; 6] {
    dc.device_transform()
}

fn blend_of(mode: u32) -> Blend {
    match mode {
        1 => Blend::Copy,
        2 => Blend::Min,
        3 => Blend::Add,
        4 => Blend::Max,
        _ => Blend::SourceOver,
    }
}
#[cfg(test)]
mod tests {
    use super::super::{Figure, Segment};
    use super::*;
    use crate::{bitmap_target, Factory};

    fn draw(geom: &Geometry, rgb: [f32; 4], world: [f32; 6]) -> Vec<u8> {
        let factory = Factory::new(false);
        let mut dc = factory.create_device().create_device_context();
        dc.set_dpi(96.0, 96.0);
        dc.set_target_bitmap(bitmap_target(120, 90, 96.0));
        dc.begin_draw();
        dc.clear(Some([1.0, 1.0, 1.0, 1.0]));
        dc.set_transform(world);
        let brush = dc.create_solid_color_brush(rgb, None);
        dc.draw_geometry_realization(geom, None, crate::geometry::DEFAULT_TOLERANCE, &brush);
        let _ = dc.end_draw();
        dc.take_target().unwrap().data
    }

    // The cached glyph feather is winding-agnostic: a counter-clockwise square fills exactly as the clockwise one,
    // antialiased, not with its fringe folded inward.
    #[test]
    fn fixed_feather_ignores_ring_winding() {
        let square = |cw: bool| {
            let mut pts = vec![[10.3f32, 10.6], [40.2, 12.1], [38.7, 41.4], [9.1, 39.8]];
            if !cw { pts.reverse(); }
            Geometry::Path { fill: super::super::FillMode::Winding, figures: vec![Figure { start: pts[0], filled: true, closed: true, segments: pts[1..].iter().map(|&p| Segment::Line(p)).collect(), seg_flags: Vec::new() }] }
        };
        let paint = |g: &Geometry| {
            let mut buf = bitmap_target(64, 64, 96.0);
            fill_feather_fixed(&mut buf, g, [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 0.0], Blend::SourceOver, AA_WIDTH);
            buf.data
        };
        let (a, b) = (paint(&square(true)), paint(&square(false)));
        assert_eq!(a, b);
        assert!(a.chunks(4).any(|p| p[3] > 0 && p[3] < 255), "antialiased edges");
    }

    fn oracle(name: &str) -> Vec<u8> {
        std::fs::read(d2d_reference::suite_dir().unwrap().join(format!("oracle/win11-26100/{name}.bgra"))).unwrap()
    }

    // The filled-realization feather is byte-exact vs native WARP for straight shapes (rect/triangle)
    // and a magnified ellipse: the mesh (the AA tessellator), the device-space unit-direction feather
    // (`aaWidth=0.5`), and the coverage PS (premul·cov) are all reproduced.
    #[test]
    fn feather_byte_exact() {
        let Some(_suite) = d2d_reference::suite_dir() else { return };
        let id = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let tri = Geometry::Path {
            fill: super::super::FillMode::Winding,
            figures: vec![Figure {
                start: [60.0, 15.0], filled: true, closed: true,
                segments: vec![Segment::Line([100.0, 75.0]), Segment::Line([20.0, 75.0])],
                seg_flags: Vec::new(),
            }],
        };
        let scale2 = [2.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let cases = [
            ("realize-01-fill-rect", draw(&Geometry::Rect([20.3, 20.7, 100.6, 70.2]), [0.2, 0.5, 0.85, 1.0], id)),
            ("realize-03-fill-triangle", draw(&tri, [0.2, 0.6, 0.3, 1.0], id)),
            ("realize-05-fill-scaled", draw(&Geometry::Ellipse { center: [30.0, 22.0], radius: [18.0, 12.0] }, [0.2, 0.5, 0.85, 1.0], scale2)),
        ];
        for (name, px) in &cases {
            let o = oracle(name);
            let diffs = px.chunks_exact(4).zip(o.chunks_exact(4)).filter(|(a, b)| a != b).count();
            assert_eq!(diffs, 0, "{name}: {diffs} px differ from the WARP feather oracle");
        }
    }

    // The feather rims the sweep-resolved region, so a figure wound the other way realizes exactly as
    // its forward twin (native: byte-identical), and an even-odd pair of identical figures cancels to
    // nothing (r13 18840's doubled inverted rect).
    #[test]
    fn realization_feathers_the_resolved_region() {
        let id = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let tri = |pts: [Point; 3]| Geometry::Path {
            fill: super::super::FillMode::Winding,
            figures: vec![Figure { start: pts[0], filled: true, closed: true, segments: vec![Segment::Line(pts[1]), Segment::Line(pts[2])], seg_flags: Vec::new() }],
        };
        let fwd = draw(&tri([[60.0, 15.0], [100.0, 75.0], [20.0, 75.0]]), [0.2, 0.6, 0.3, 1.0], id);
        let rev = draw(&tri([[60.0, 15.0], [20.0, 75.0], [100.0, 75.0]]), [0.2, 0.6, 0.3, 1.0], id);
        assert!(fwd == rev, "reversed triangle realizes differently");
        let rect = Figure { start: [17.86, 110.0 - 80.0], filled: true, closed: true, segments: vec![Segment::Line([14.93, 30.0]), Segment::Line([14.93, 42.18]), Segment::Line([17.86, 42.18])], seg_flags: Vec::new() };
        let pair = Geometry::Path { fill: super::super::FillMode::Alternate, figures: vec![rect.clone(), rect] };
        let blank = draw(&Geometry::Rect([0.0, 0.0, 0.0, 0.0]), [0.2, 0.6, 0.3, 1.0], id);
        assert!(draw(&pair, [0.2, 0.6, 0.3, 1.0], id) == blank, "even-odd duplicate pair should draw nothing");
    }

    // A filled realization whose bounds pass the safe box is cut to [-524287, 524287]² first; one that
    // stays inside is baked unchanged.
    #[test]
    fn huge_fill_realization_is_safe_clipped() {
        let fig = |c: Point| Figure { start: [0.0, 0.0], filled: true, closed: true, segments: vec![Segment::Quadratic(c, [80.0, 32.0]), Segment::Line([0.0, 40.0])], seg_flags: Vec::new() };
        let small = Geometry::Path { fill: super::super::FillMode::Winding, figures: vec![fig([40.0, 10.0])] };
        assert!(safe_clipped(&small).is_none());
        let huge = Geometry::Path { fill: super::super::FillMode::Winding, figures: vec![fig([-1e30, 45.0])] };
        let Some(Geometry::Path { figures, .. }) = safe_clipped(&huge) else { panic!("huge fill not clipped") };
        let pts = figures.iter().flat_map(|f| std::iter::once(f.start).chain(f.segments.iter().map(|s| match *s { Segment::Line(p) | Segment::Cubic(_, _, p) => p, _ => unreachable!() })));
        assert!(pts.into_iter().all(|p| p[0].abs() <= 524287.0 && p[1].abs() <= 524287.0));
    }

    // A stroked realization of a shape far past the safe box widens only the clipped shape: a dashed
    // rectangle a million DIPs tall is cut at 524287 before dashing, starting from its top-left corner
    // even when its rect is given inverted.
    #[test]
    fn huge_stroke_source_clips_to_safe_box() {
        let style = StrokeStyle { dashes: vec![2.0, 1.5], ..StrokeStyle::default() };
        for rect in [[5.3, 10.2, 50.7, 2e6], [50.7, 2e6, 5.3, 10.2]] {
            let Some(Geometry::Path { figures, .. }) = stroke_safe_clipped(&Geometry::Rect(rect), 3.0, &style) else { panic!("not clipped") };
            assert_eq!(figures.len(), 1);
            assert_eq!(figures[0].start, [5.3, 10.2]);
            let ends = figures[0].segments.iter().map(|s| match *s { Segment::Line(p) => p, _ => unreachable!() });
            assert!(ends.into_iter().all(|p| p[1] <= 524287.0));
        }
        assert!(stroke_safe_clipped(&Geometry::Rect([5.3, 10.2, 50.7, 5e5]), 3.0, &style).is_none());
    }

    // A realization bakes empty when its bounds come out unordered, or when the strip clip of a shape past
    // the safe box gives its filled figures a NaN crossing; a hollow figure's NaN crossing does not count.
    #[test]
    fn realization_bounds_tests_bake_empty() {
        use super::super::FillMode;
        let fig = |start: Point, segments: Vec<Segment>, filled: bool| Figure { start, filled, closed: false, segments, seg_flags: Vec::new() };
        let path = |figures| Geometry::Path { fill: FillMode::Winding, figures };
        let pen = StrokeStyle::default();
        let quad = path(vec![fig([76.6, 113.9], vec![Segment::Quadratic([0.5, 22.1], [32.4, -3.4e38])], true)]);
        assert!(realization_clips_empty(&quad, None));
        let tall = Geometry::Ellipse { center: [18.4, 11.2], radius: [4.0, 3.4e38] };
        assert!(realization_clips_empty(&tall, Some((0.34, &pen))));
        let small = Geometry::Rect([1.0, 2.0, 30.0, 40.0]);
        assert!(realization_clips_empty(&small, Some((f32::NAN, &pen))));
        assert!(!realization_clips_empty(&small, Some((2.0, &pen))));
        let hollow = path(vec![fig([46.3, 12.2], vec![Segment::Cubic([21.7, 18.6], [0.5, 23.3], [38.4, 23.2])], false)]);
        assert!(!realization_clips_empty(&hollow, Some((1.5, &pen))));
        let mixed = path(vec![
            fig([73.4, 91.5], vec![Segment::Cubic([32.3, 126.7], [46.1, -1e9], [78.1, 11.0])], true),
            fig([64.7, 114.8], vec![Segment::Line([17.4, f32::NEG_INFINITY]), Segment::Line([29.5, 0.0])], false),
        ]);
        assert!(!realization_clips_empty(&mixed, Some((6.4, &pen))));
    }

    // An edge shorter than 1e-6 has no direction, so its vertices take no feather offset (the 1-ulp
    // seam where a closed stroke's rails meet their cap); an infinite one still normalizes.
    #[test]
    fn unitize_rejects_sub_micro_edges() {
        assert_eq!(unitize([2.4e-7, 0.0]), None);
        assert_eq!(unitize([0.0, 9.0e-7]), None);
        assert_eq!(unitize([f32::NAN, 1.0]), None);
        assert_eq!(unitize([2.0e-6, 0.0]), Some([1.0, 0.0]));
        assert_eq!(unitize([f32::INFINITY, 1.0]).map(|v| v[1]), Some(0.0));
    }

    // A stroked realization thinner than a pixel: its interior triangles invert under the feather and
    // are culled, and the stroke draws at `width` opacity, so the band peaks near 1−(1−0.64·w)² (native
    // r21 26529's 0.273-wide ellipse: ≈0.32), never the full coverage the uncull'd interior gave.
    #[test]
    fn thin_stroke_realization_culls_and_fades() {
        let factory = Factory::new(false);
        let mut dc = factory.create_device().create_device_context();
        dc.set_target_bitmap(bitmap_target(48, 16, 96.0));
        dc.begin_draw();
        dc.clear(Some([0.0, 0.0, 0.0, 0.0]));
        let brush = dc.create_solid_color_brush([1.0, 1.0, 1.0, 1.0], None);
        let ellipse = Geometry::Ellipse { center: [31.311554, 7.793466], radius: [27.482342, 12.865635] };
        dc.draw_geometry_realization(&ellipse, Some(&(0.273_375_93, StrokeStyle::default())), 0.753_047_2, &brush);
        let _ = dc.end_draw();
        let max_a = dc.take_target().unwrap().data.chunks_exact(4).map(|p| p[3]).max().unwrap();
        assert!((60..128).contains(&max_a), "thin stroke peak alpha {max_a}");
    }

    // `winding_interior`: the feather's interior fill honours nonzero winding across figures, so a hole
    // (nested opposite winding) stays empty, a nested same-winding contour still fills, and two
    // overlapping contours fill their union — the behaviour COLR emoji leaves and holed glyphs ("0")
    // need. Checks the fill at representative pixels (B channel: ~217 blue filled, 255 white empty).
    #[test]
    fn winding_interior_holes_and_overlap() {
        let id = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let blue = [0.2, 0.5, 0.85, 1.0];
        let b = |px: &[u8], x: usize, y: usize| px[(y * 120 + x) * 4] as i32;
        let sq = |a: [f32; 2], c: [f32; 2], cw: bool| -> Figure {
            // Axis-aligned rectangle a..c, wound CW or CCW (y-down device space).
            let (p1, p2, p3) = if cw {
                ([c[0], a[1]], [c[0], c[1]], [a[0], c[1]])
            } else {
                ([a[0], c[1]], [c[0], c[1]], [c[0], a[1]])
            };
            Figure {
                start: a, filled: true, closed: true,
                segments: vec![Segment::Line(p1), Segment::Line(p2), Segment::Line(p3)],
                seg_flags: Vec::new(),
            }
        };
        // Outer CW square with an inner CCW square ⇒ a hole.
        let holed = Geometry::Path {
            fill: super::super::FillMode::Winding,
            figures: vec![sq([20.0, 20.0], [80.0, 80.0], true), sq([40.0, 40.0], [60.0, 60.0], false)],
        };
        let px = draw(&holed, blue, id);
        assert!(b(&px, 50, 50) > 240, "hole centre should be empty (white), got B={}", b(&px, 50, 50));
        assert!(b(&px, 25, 50) < 240, "ring should be filled, got B={}", b(&px, 25, 50));
        // Two overlapping CW squares ⇒ union filled (winding 2 in the overlap is still inside).
        let overlap = Geometry::Path {
            fill: super::super::FillMode::Winding,
            figures: vec![sq([20.0, 20.0], [50.0, 80.0], true), sq([40.0, 20.0], [70.0, 80.0], true)],
        };
        let px = draw(&overlap, blue, id);
        for (x, y) in [(25usize, 50usize), (45, 50), (65, 50)] {
            assert!(b(&px, x, y) < 240, "overlap/union ({x},{y}) should be filled, got B={}", b(&px, x, y));
        }
    }
}
