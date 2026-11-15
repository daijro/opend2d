//! COPY-blend (opaque) geometry-fill coverage: native's `CHwRasterizer` trapezoid-fringe AA, not the
//! 8×8 area coverage the SOURCE_OVER path (our [`super::rasterize_edges`]) uses. `IsAlphaBlendingRequired`
//! is false for `D2D1_PRIMITIVE_BLEND_COPY`, so native takes the opaque path: WPF's hwrasterizer
//! (`RasterizeEdges` → `AddTrapezoid`) emits, per span between two active edges, an 8-vertex triangle
//! strip whose coverage attribute ramps **linearly in x** across a fringe of half-width
//! `dL = |dx/dy|·0.5 + 0.5` device px along each slanted edge (0→1 left fringe, 1 interior, 1→0 right
//! fringe). We feed those triangles through the byte-exact mesh coverage path
//! ([`crate::warp::fill_triangle_cov`]: snap 1/256, the triangle-attribute plane, fused-FMA eval, MAX
//! accumulate), so there is no new GPU emulation — the GPU's attribute interpolation IS that path.
//!
//! This is one faithful `RasterizeEdges` active-edge-list walk (transliterated from the MIT WPF source
//! `WpfGfx/core/hw/{hwrasterizer,hwvertexbuffer}.cpp` + `aacoverage.{h,cpp}`, reconciled against native's
//! DDI capture). The SAME edge list drives both native outputs: `OutputTrapezoids`
//! for pixel-aligned bands where the two edges don't converge, and `FillEdges`→`CCoverageBuffer` (the 8×8
//! AddInterval complex scan) for the cap/vertex rows. Stage 1 is convex single-contour fills (one left +
//! one right edge per band — rects, ellipses/rounded-rects/triangles via `FillGeometry`, canvas `copy`).
//!
//! Three conventions match native bit-for-bit (verified against captured CEdges):
//!   1. 28.4 endpoint quantisation is `floor(device·16)`, not round-to-nearest — it sets each edge's
//!      integer DDA slope `dm/dn`, hence the fringe half-width and rail slope ([`to_28_4`]).
//!   2. The AA edge-scale offset: `FillRectangle` uses `(28.4)<<3` (`off=0`), while `FillGeometry`/
//!      `FillEllipse` shift the input points −0.5 device (pixel-centre) and InitializeEdges adds +7, net
//!      `off=-1`. Trapped from four native CEdge captures (rect SY=55 vs ellipse/triangle/rr SY shifted).
//!   3. d2d1's `ConvertSubpixel{X,Y}ToPixel` adds `+0.5` subpixel in both axes (it samples the subpixel
//!      centre, dropping WPF's `(v+8)` fixup), so trapezoid rails/tops land in device space exactly
//!      ([`CEdge::rail`]); the integer complex scan ([`Intervals::add`]) uses the raw subpixel X, no `+0.5`.
//!
//! Byte-exact vs the native rot-rect COPY oracle (the unit test). Finely-flattened curves (ellipse/rr)
//! are byte-exact on the straight trapezoid rails; the cap/vertex complex-scan rows carry a small (≤10)
//! residual that is the shared grayscale scan-converter edge residual, not a COPY-path issue.

/// 28.4 fixed-point of a device coordinate: `floor(device·16)`. WPF's `TransformRasterizerPointsTo28_4`
/// nominally rounds, but d2d1's capture shows it truncates toward −∞ (e.g. 96.997→96, 364.517→364): the
/// endpoint floor is what sets each edge's integer DDA slope `dm/dn`, so it must match native bit-for-bit.
fn to_28_4(v: f32) -> i32 {
    (v * 16.0).floor() as i32
}

/// A `CEdge` exactly as `InitializeEdges` builds it. `x`/`error` are the edge's
/// subpixel x at the current scanline (x in 1/8-px units; `x>>3` = pixel, `x&7` = sub-column); the
/// per-scanline step is `dx`+`error_up`/`error_down`; `start_y`/`end_y` bound the subpixel scanlines.
/// The AA scaling is `(to_28_4(v) + off) << 3`: `off = 0` for a `FillRectangle` primitive (native uses
/// no pixel-centre shift there — rot-rect trapped SY=55), `off = -1` for `FillGeometry`/`FillEllipse`
/// (native's -0.5 device pixel-centre shift = -8 in 28.4, plus WPF's +7 InitializeEdges = net -1;
/// ellipse/triangle/rounded-rect all trapped this way). `dm/dn` (hence dx/err_up/err_down and `delta`)
/// are offset-independent — only the snapped absolute position (`x`/`error`/`start_y`) depends on `off`.
#[derive(Clone)]
struct CEdge {
    x: i32,
    dx: i32,
    error: i32,
    error_up: i32,
    error_down: i32,
    start_y: i32,
    end_y: i32,
    /// Winding direction: +1 if the edge runs downward (p1.y ≥ p0.y), −1 upward. InitializeEdges
    /// normalizes every edge to run top→bottom (so `dn ≥ 0`), discarding the sign; the fill rule needs
    /// it back — the active-edge-list walk sums `wind` left→right and fills where the sum is non-zero
    /// (Winding) or odd (Alternate). For a single convex contour there are two active edges of opposite
    /// winding, so this reduces to the old one-left/one-right pairing.
    wind: i32,
}

/// A device point on InitializeEdges' AA grid (1/128 px, `8·x28.4 + 56`). A geometry fill's points reach
/// it through `TransformRasterizerPointsTo28_4` with the −0.5 pixel-centre shift, `ceil(16·v − 8)` (the
/// fill path's own conversion, `flatten_fill_prims_scale`), plus WPF's +7: equal to `floor(16·v) − 1` on
/// the curve cracker's grid points, but not on an exact line endpoint, where the floor dropped the
/// ellipse's top extreme a subpixel row (native's 32-point ellipse capture, copyddi/ellinit: (17.3, 5.9)
/// → 87, not 86). The 28.4 value is clamped to ±2²³ first, as InitializeEdges does.
fn aa_point(p: [f32; 2], off: i32) -> [i32; 2] {
    let aa = |v: f32| {
        let q = if off == 0 { to_28_4(v) - 7 } else { (v * 16.0 - 8.0).ceil() as i32 };
        (q.clamp(-8_388_608, 8_388_608) + 7) << 3
    };
    [aa(p[0]), aa(p[1])]
}

/// InitializeEdges' clip walk over one closed figure's AA points (`pts`, the closing point repeated):
/// an edge wholly above the clip top or below `bottom − 16` is skipped; any but the last edge whose
/// end, start and following point all lie left of the clip (or all right of it) is folded away — its
/// end point takes the start point's place, so the next edge spans the gap — and the rest are kept.
/// An off-target figure thus contributes no edges to the walk (r3 2128: an off-target dash's short edge
/// had cut the on-target dash's trapezoid band into complex scans).
fn clipped_edges(mut pts: Vec<[i32; 2]>, clip: [i32; 4], out: &mut Vec<CEdge>) {
    let [l, t, r, b] = clip;
    let n = pts.len();
    for i in 1..n {
        let (prev, cur) = (pts[i - 1], pts[i]);
        if (prev[1] <= t && cur[1] <= t) || (prev[1] > b && cur[1] > b) {
            continue;
        }
        if i + 1 < n {
            let next = pts[i + 1];
            if (prev[0] < l && cur[0] < l && next[0] < l) || (prev[0] > r && cur[0] > r && next[0] > r) {
                pts[i] = prev;
                continue;
            }
        }
        if let Some(e) = init_edge(prev, cur) {
            out.push(e);
        }
    }
}

fn init_edge(p0: [i32; 2], p1: [i32; 2]) -> Option<CEdge> {
    let (mut dm, mut dn) = (p1[0] - p0[0], p1[1] - p0[1]);
    let wind = if dn >= 0 { 1 } else { -1 }; // winding direction before the top→bottom normalization
    let (x_start0, y_start, y_start_int, y_end_int) = if dn >= 0 {
        (p0[0], p0[1], (p0[1] + 15) >> 4, (p1[1] + 15) >> 4)
    } else {
        dn = -dn;
        dm = -dm;
        (p1[0], p1[1], (p1[1] + 15) >> 4, (p0[1] + 15) >> 4)
    };
    if y_end_int <= y_start_int {
        return None; // spans no integer scanline (horizontal) — dropped
    }
    let (dx, error_up);
    if dm < 0 {
        let dma = -dm;
        if dma < dn {
            dx = -1;
            error_up = dn - dma;
        } else {
            let q = (dma as u32 / dn as u32) as i32;
            let r = (dma as u32 % dn as u32) as i32;
            if r > 0 {
                dx = -q - 1;
                error_up = dn - r;
            } else {
                dx = -q;
                error_up = r;
            }
        }
    } else if dm < dn {
        dx = 0;
        error_up = dm;
    } else {
        dx = (dm as u32 / dn as u32) as i32;
        error_up = (dm as u32 % dn as u32) as i32;
    }
    let mut error = -1i32;
    let mut x = x_start0;
    if (y_start & 15) != 0 {
        for _ in 0..(16 - (y_start & 15)) {
            x += dx;
            error += error_up;
            if error >= 0 {
                error -= dn;
                x += 1;
            }
        }
    }
    if (x & 15) != 0 {
        error -= dn * (16 - (x & 15));
        x += 15;
    }
    x >>= 4;
    error >>= 4;
    Some(CEdge { x, dx, error, error_up, error_down: dn, start_y: y_start_int, end_y: y_end_int, wind })
}

impl CEdge {
    /// Advance the DDA in place by `n` subpixel scanlines (AdvanceDDAMultipleSteps, hwrasterizer.cpp).
    fn step(&mut self, n: i32) {
        self.x += n * self.dx;
        let mut err = self.error as i64 + n as i64 * self.error_up as i64;
        if err >= 0 {
            let d = err / self.error_down as i64 + 1;
            self.x += d as i32;
            err -= self.error_down as i64 * d;
        }
        self.error = err as i32;
    }
    /// Device-x rail at the current scanline: ConvertSubpixelXToPixel with d2d1's +0.5 subpixel.
    fn rail(&self) -> f32 {
        (self.x as f32 + 0.5 + self.error as f32 / self.error_down as f32) * 0.125
    }
    /// `rPixelXDelta` = 0.5 + 0.5·|Dx + ErrorUp/ErrorDown| (OutputTrapezoids fringe half-width).
    fn delta(&self) -> f32 {
        0.5 + 0.5 * (self.dx as f32 + self.error_up as f32 / self.error_down as f32).abs()
    }
}

/// `ComputeDeltaUpperBound(pEdge, nAdv)` (hwrasterizer.cpp): ≥ `nAdv·|1/m|`, for the self-intersection test.
fn delta_upper_bound(e: &CEdge, n_adv: i32) -> i32 {
    if e.error_up == 0 {
        n_adv * e.dx.abs()
    } else {
        let (abs_dx, abs_eup) = if e.dx >= 0 { (e.dx, e.error_up) } else { (-e.dx - 1, e.error_down - e.error_up) };
        n_adv * abs_dx + (n_adv * abs_eup) / e.error_down + 1
    }
}

/// `ComputeDistanceLowerBound(l, r)` (hwrasterizer.cpp): ≤ the subpixel x-distance between `l` and `r`.
fn distance_lower_bound(l: &CEdge, r: &CEdge) -> i32 {
    let mut d = r.x - l.x;
    if (l.error_down as i64) * (r.error as i64 + 1) < (r.error_down as i64) * (l.error as i64 + 1) {
        d -= 1;
    }
    d
}

/// `ComputeTrapezoidsEndScan`: the furthest subpixel-Y (snapped to a pixel boundary) a trapezoid band can
/// reach from `cur`, or `cur` (⇒ complex scan). For every x-adjacent pair, the running bottom takes the
/// left edge's end; a pair whose expanded top distance is negative aborts; a converging pair (left inverse
/// slope steeper than the right's) whose expanded bottom positions cross ends the band at the integer
/// crossing estimate `cur + adv·d1/(d1 + d2)` (d1 the top lower bound, d2 the bottom overlap + 1), and a
/// band under one pixel high aborts.
fn compute_end_scan(active: &[CEdge], cur: i32, next_inactive: i32) -> i32 {
    let mut bottom = next_inactive;
    for i in 0..active.len() {
        bottom = bottom.min(active[i].end_y);
        let Some(r) = active.get(i + 1) else { break };
        let l = &active[i];
        let expand = 8 + delta_upper_bound(l, 4) + delta_upper_bound(r, 4);
        let top = distance_lower_bound(l, r) - expand;
        if top < 0 {
            return cur;
        }
        let converging = l.dx > r.dx
            || (l.dx == r.dx && (r.error_down as i64) * (l.error_up as i64) > (r.error_up as i64) * (l.error_down as i64));
        if !converging {
            continue;
        }
        let adv = bottom - cur;
        let (mut lb, mut rb) = (l.clone(), r.clone());
        lb.step(adv);
        rb.step(adv);
        let left_adj = lb.x + expand;
        if left_adj < rb.x {
            continue;
        }
        bottom = cur + adv * top / (top + left_adj - rb.x + 1);
        if bottom < cur + 8 {
            return cur;
        }
    }
    bottom & !7
}

/// A triangle in native's vertex-buffer order (its last vertex is WARP's attribute reference).
pub(crate) type Tri = [[f32; 2]; 3];

/// `CCoverageBuffer` (aacoverage.h): the complex-scan row's interval list, `(pixel x, coverage)` from
/// an `i32::MIN` head to an `i32::MAX` tail. `AddInterval` splits at the touched pixels and never merges,
/// and the breakpoints decide `AddComplexScan`'s triangles.
struct Intervals {
    iv: Vec<(i32, i32)>,
}

impl Intervals {
    fn new() -> Self {
        Intervals { iv: vec![(i32::MIN, 0), (i32::MAX, 0)] }
    }

    fn clear(&mut self) {
        self.iv.truncate(1);
        self.iv.push((i32::MAX, 0));
    }

    /// `CCoverageBuffer::AddInterval`: add one sub-scanline interval `[xl, xr)` (1/8-px units).
    fn add(&mut self, xl: i32, xr: i32) {
        if xl >= xr {
            return;
        }
        let iv = &mut self.iv;
        let (pl, pr) = (xl >> 3, xr >> 3);
        let mut i = 0;
        while iv[i + 1].0 < pl {
            i += 1;
        }
        if iv[i + 1].0 != pl {
            iv.insert(i + 1, (pl, iv[i].1));
        }
        i += 1;
        let left = 8 - (xl & 7);
        if (left < 8 || pl == pr) && pl + 1 != iv[i + 1].0 {
            iv.insert(i + 1, (pl + 1, iv[i].1));
        }
        if pl == pr {
            iv[i].1 += xr - xl;
            return;
        }
        iv[i].1 += left;
        while iv[i + 1].0 < pr {
            i += 1;
            iv[i].1 += 8;
        }
        if iv[i + 1].0 != pr {
            iv.insert(i + 1, (pr, iv[i].1 - 8));
        }
        i += 1;
        let right = xr & 7;
        if right > 0 {
            if pr + 1 != iv[i + 1].0 {
                iv.insert(i + 1, (pr + 1, iv[i].1));
            }
            iv[i].1 += right;
        }
    }

    /// d2d1 `CHwVertexBufferWriter::AddComplexScan`: a multi-pixel interval is one triangle
    /// `(a, y+½), (b, y), (b, y+1)`; two adjacent single-pixel intervals share one triangle (a linear
    /// coverage ramp through both centres) when it ends inside `xmax`, else each is its own. `emit` gets
    /// each in-target pixel with its interval coverage and triangle.
    fn emit(&self, row: usize, w: usize, emit: &mut dyn FnMut(usize, usize, Cov, Tri)) {
        let (y, xmax) = (row as f32, w as i32);
        let tri = |a: i32, b: i32| -> Tri {
            let r = (xmax as f32).min(b as f32);
            [[a as f32, y + 0.5], [r, y], [r, y + 1.0]]
        };
        let mut put = |a: i32, b: i32, t: Tri, c: &dyn Fn(i32) -> i32| {
            for x in a.max(0)..b.min(xmax) {
                emit(x as usize, row, Cov::Area(c(x).clamp(0, 64) as u32), t);
            }
        };
        let mut pending: Option<(i32, i32)> = None;
        let iv = &self.iv;
        for k in 0..iv.len() - 1 {
            let ((x, c), next) = (iv[k], iv[k + 1].0);
            if x == i32::MAX {
                break;
            }
            if c == 0 {
                continue;
            }
            if next - x != 1 {
                if let Some((px, pc)) = pending.take() {
                    put(px, px + 1, tri(px, px + 1), &|_| pc);
                }
                put(x, next, tri(x, next), &|_| c);
                continue;
            }
            match pending.take() {
                Some((px, pc)) if px == x - 1 && next <= xmax => {
                    put(px, px + 2, tri(px, px + 2), &|q| if q == px { pc } else { c });
                }
                Some((px, pc)) => {
                    put(px, px + 1, tri(px, px + 1), &|_| pc);
                    pending = Some((x, c));
                }
                None => pending = Some((x, c)),
            }
        }
        if let Some((px, pc)) = pending {
            put(px, px + 1, tri(px, px + 1), &|_| pc);
        }
    }
}

/// Per-pixel coverage of a COPY-blend convex fill: `Area` for complex-scan (cap/vertex) rows — an 8×8
/// sub-sample count in `0..=64` — and `Trapezoid` for the fringe-band rows — a continuous `0..=1`
/// coverage. Both are drawn as triangles with a coverage attribute, so both blend through [`Cov::lerp`].
pub(crate) enum Cov {
    Area(u32),
    Trapezoid(f32),
}

impl Cov {
    /// The coverage as the kernel's 8-bit attribute: an area count rounds `n·255/64`, a band coverage
    /// `c·255`, each to nearest.
    pub(crate) fn c8(&self) -> i32 {
        match *self {
            Cov::Area(n) => (n.min(64) as f32 * 255.0 / 64.0).round_ties_even() as i32,
            Cov::Trapezoid(c) => (c.clamp(0.0, 1.0) * 255.0).round_ties_even() as i32,
        }
    }

    /// Lerp the target toward the byte source by the coverage (the trapezoid/span triangles' kernel;
    /// cpg-*-b1 probes, solid and brush alike): [`lerp8`].
    pub(crate) fn lerp(&self, dst: [u8; 4], src: [u8; 4]) -> [u8; 4] {
        lerp8(dst, src, self.c8())
    }

    /// MIN of an opaque source: the source over white by the unquantised coverage, rounded once, then
    /// the minimum with the target (VM primitive-blend probes: MIN of an opaque brush fills geometry
    /// through this walk, not the 8×8 area).
    pub(crate) fn min_white(&self, dst: [u8; 4], src: [u8; 4]) -> [u8; 4] {
        let c = match *self {
            Cov::Area(n) => n.min(64) as f32 / 64.0,
            Cov::Trapezoid(c) => c.clamp(0.0, 1.0),
        };
        core::array::from_fn(|k| ((src[k] as f32 * c + 255.0 - 255.0 * c).round_ties_even().clamp(0.0, 255.0) as u8).min(dst[k]))
    }
}

/// WARP's COPY coverage lerp in the high byte, each product rounded `x + (x>>8)`: `(s·c + d·(255−c))/255`
/// with `c` the 8-bit coverage (cpg-lin-rr-b1 (12,26): R 61, where one f32 rounding gives 62).
pub(crate) fn lerp8(dst: [u8; 4], src: [u8; 4], c: i32) -> [u8; 4] {
    core::array::from_fn(|k| {
        let (ps, pd) = (src[k] as i32 * c, dst[k] as i32 * (255 - c));
        ((ps + (ps >> 8) + pd + (pd >> 8) + 128) >> 8).min(255) as u8
    })
}

/// Rasterize a CONVEX device polygon the way native's `CHwRasterizer::RasterizeEdges` does: one AEL walk
/// driving both outputs off the same edge list — `OutputTrapezoids` for pixel-aligned bands where the
/// edges don't converge, and `FillEdges`→`CCoverageBuffer` (complex-scan) for the cap/vertex rows. `emit`
/// is called once per covered pixel. `geometry` selects native's pixel-centre convention: `true` for
/// FillGeometry/FillEllipse (-0.5 device shift ⇒ edge off = -1), `false` for a FillRectangle primitive.
/// Walk the x-sorted active edge list left→right summing winding, and return the maximal `(left, right)`
/// active-edge index pairs of the "inside" runs — where the running sum is non-zero (Winding) or odd
/// (Alternate). Interior edges within a run are not coverage boundaries (a self-intersecting winding
/// fill has no fringe at the crossing), so only the entering/leaving edges bound each emitted span. For
/// a convex contour there are two edges of opposite winding ⇒ one span `(0, 1)` = the old pairing.
fn inside_spans(active: &[CEdge], alternate: bool) -> Vec<(usize, usize)> {
    let inside = |w: i32| if alternate { w & 1 != 0 } else { w != 0 };
    let mut spans = Vec::new();
    let mut wind = 0i32;
    let mut enter: Option<usize> = None;
    for (i, e) in active.iter().enumerate() {
        let was = inside(wind);
        wind += e.wind;
        let now = inside(wind);
        if !was && now {
            enter = Some(i);
        } else if was && !now {
            if let Some(l) = enter.take() {
                spans.push((l, i));
            }
        }
    }
    spans
}

/// Single convex contour (back-compat wrapper): a convex shape has two active edges of opposite
/// winding per band, so Winding and Alternate agree — pass Alternate.
pub(crate) fn fill_convex_copy(
    poly: &[[f32; 2]],
    w: usize,
    h: usize,
    geometry: bool,
    emit: impl FnMut(usize, usize, Cov, Tri),
) {
    fill_copy(&[poly], true, w, h, geometry, emit);
}

/// General COPY-blend fill: the full `RasterizeEdges` active-edge-list walk over one or more contours
/// under `alternate` (true = even-odd, false = non-zero winding). Non-convex and multi-contour shapes
/// pair their x-sorted active edges by winding ([`inside_spans`]) instead of the convex one-left/one-right
/// assumption, so self-intersections (a winding star's doubly-wound core) and overlaps match native.
pub(crate) fn fill_copy(
    polys: &[&[[f32; 2]]],
    alternate: bool,
    w: usize,
    h: usize,
    geometry: bool,
    emit: impl FnMut(usize, usize, Cov, Tri),
) {
    fill_copy_in(polys, alternate, w, h, geometry, [0, 0, w as i32, h as i32], emit);
}

/// [`fill_copy`] under the draw's clip: `clip` is the pixel rect `[l, t, r, b]` the pushed clips and layer
/// bounds leave on the target. The rasterizer's edge walk takes it as its clip rect — the walk's edge
/// folding and its first and last scanlines — rather than the whole target (a trace of InitializeEdges
/// under layer bounds [4.54, 15.76, 24.5, 33.07] reads [4, 15, 25, 34]).
pub(crate) fn fill_copy_in(
    polys: &[&[[f32; 2]]],
    alternate: bool,
    w: usize,
    h: usize,
    geometry: bool,
    clip: [i32; 4],
    mut emit: impl FnMut(usize, usize, Cov, Tri),
) {
    let [cl, ct, cr, cb] = [clip[0].max(0), clip[1].max(0), clip[2].min(w as i32), clip[3].min(h as i32)];
    if cr <= cl || cb <= ct {
        return;
    }
    if !polys.iter().flat_map(|p| p.iter()).all(|p| p[0].is_finite() && p[1].is_finite()) {
        return;
    }
    let off = if geometry { -1 } else { 0 };
    let y_bottom = cb * 8;
    // The clip rect in the AA grid units (28.4 × 8), with InitializeEdges' `bottom − 16`.
    let clip = [cl << 7, ct << 7, cr << 7, (cb << 7) - 16];
    let mut edges: Vec<CEdge> = Vec::new();
    for poly in polys {
        if poly.len() < 3 {
            continue;
        }
        // The edge-list sink hands InitializeEdges a figure in 32-point chunks (each chunk starting at
        // the previous chunk's last point, the start point appended at the figure's end), so the clip
        // walk's folds and its unfolded last edge apply per chunk.
        let pts: Vec<[i32; 2]> = poly.iter().chain(std::iter::once(&poly[0])).map(|&p| aa_point(p, off)).collect();
        let mut i = 0;
        while i + 1 < pts.len() {
            let end = (i + 32).min(pts.len());
            clipped_edges(pts[i..end].to_vec(), clip, &mut edges);
            i = end - 1;
        }
    }
    if edges.is_empty() {
        return;
    }
    edges.sort_by_key(|e| e.start_y);
    let mut next_idx = 0usize; // next inactive edge to activate
    let mut active: Vec<CEdge> = Vec::new();
    let mut cur = edges[0].start_y.max(ct * 8);
    let mut band = vec![0.0f32; w * h]; // scratch for the trapezoid triangles
    let mut band_tri: Vec<Tri> = vec![[[0.0; 2]; 3]; w * h]; // the triangle each band pixel came from
    let mut ivs = Intervals::new(); // complex-scan intervals for the pixel row being accumulated
    let mut cov_row: i32 = -1; // pixel row `ivs` is accumulating for (-1 = empty)
    let flush = |ivs: &mut Intervals, row: i32, emit: &mut dyn FnMut(usize, usize, Cov, Tri)| {
        if row >= 0 {
            ivs.emit(row as usize, w, emit);
        }
        ivs.clear();
    };
    while cur < y_bottom {
        while next_idx < edges.len() && edges[next_idx].start_y <= cur {
            let mut e = edges[next_idx].clone();
            e.step(cur - e.start_y); // advance a freshly-activated edge to the current scanline
            e.start_y = cur;
            active.push(e);
            next_idx += 1;
        }
        active.retain(|e| e.end_y > cur);
        // The inactive list ends in a sentinel whose start is INT_MAX, so after the last edge a band may
        // run to its edges' ends, past the target (cpg-lin-quad-clip-b1, once its off-target edges are
        // clipped away).
        let next_inactive = edges.get(next_idx).map(|e| e.start_y).unwrap_or(i32::MAX);
        if active.is_empty() {
            cur = next_inactive.max(cur + 1);
            continue;
        }
        active.sort_by(|a, b| {
            let xa = a.x as f64 + a.error as f64 / a.error_down as f64;
            let xb = b.x as f64 + b.error as f64 / b.error_down as f64;
            xa.partial_cmp(&xb).unwrap()
        });
        // Trapezoid attempt (OutputTrapezoids): pixel-aligned, ≥8 before the next edge event, non-crossing.
        // Only when every inside run is a SIMPLE two-edge span (r == l+1): a run with interior edges comes
        // from self-intersection/overlap (winding |≥2|), where the span rails can cross an interior edge
        // within the band and the fixed-rail trapezoid collapses — those bands take the complex scan,
        // which re-pairs by winding every sub-scanline.
        let simple_spans = inside_spans(&active, alternate).iter().all(|&(l, r)| r == l + 1);
        if (cur & 7) == 0 && active.len() >= 2 && next_inactive >= cur + 8 && simple_spans {
            let next = compute_end_scan(&active, cur, next_inactive);
            if next >= cur + 8 {
                // OutputTrapezoids: trapezoid spans device y [yt, yb] with top rails at `cur`, bottom
                // rails at `next` (no extrapolation — the DDA rail line is exact). d2d1's convert adds the
                // +0.5 subpixel in both axes (`rail()` for x; here for y), so yt = (cur+0.5)/8.
                let (yt, yb) = ((cur as f32 + 0.5) * 0.125, (next as f32 + 0.5) * 0.125);
                // A band may run past the target bottom: native keeps its edges whole (the triangles, hence
                // their attribute planes, reach the edge's end; WARP clips them — lrrq-b1's DDI band ends at
                // y 37.06 on a 32-px target), and only rows inside the target are drawn (r9 14009 indexed
                // `band` past `h`).
                let (r0, r1) = ((cur >> 3) as usize, ((next >> 3) as usize).min(h));
                for px in &mut band[r0 * w..r1 * w] {
                    *px = 0.0;
                }
                for (li, ri) in inside_spans(&active, alternate) {
                    let (l, r) = (&active[li], &active[ri]);
                    let (lt, rt) = (l.rail(), r.rail());
                    let (lb, rb) = {
                        let (mut el, mut er) = (l.clone(), r.clone());
                        el.step(next - cur);
                        er.step(next - cur);
                        (el.rail(), er.rail())
                    };
                    emit_trapezoid(yt, yb, lt, lb, l.delta(), rt, rb, r.delta(), w, h, &mut band, &mut band_tri);
                }
                for row in r0..r1 {
                    for x in 0..w {
                        let c = band[row * w + x];
                        if c > 0.0 {
                            emit(x, row, Cov::Trapezoid(c), band_tri[row * w + x]);
                        }
                    }
                }
                for e in &mut active {
                    e.step(next - cur);
                    e.start_y = next;
                }
                cur = next;
                continue;
            }
        }
        // Complex scanline (FillEdges → CCoverageBuffer): accumulate this sub-scanline's interval(s).
        let row = cur >> 3;
        if cov_row != row {
            flush(&mut ivs, cov_row, &mut emit);
            cov_row = row;
        }
        for (li, ri) in inside_spans(&active, alternate) {
            ivs.add(active[li].x, active[ri].x);
        }
        // Flush the pixel row once its last (8th) sub-scanline is in (GenerateOutputAndClearCoverage).
        if cur & 7 == 7 {
            flush(&mut ivs, cov_row, &mut emit);
            cov_row = -1;
        }
        for e in &mut active {
            e.step(1);
            e.start_y = cur + 1;
        }
        cur += 1;
    }
    flush(&mut ivs, cov_row, &mut emit); // last partial-coverage row (shape bottom vertex)
}

/// WPF `AddTrapezoid`'s 8-vertex strip, as six triangles fed to [`crate::warp::fill_triangle_cov`]
/// (coverage in the interpolated attribute). Left fringe `x∈[xl−dl, xl+dl]` ramps 0→1, interior is 1,
/// right fringe `x∈[xr−dr, xr+dr]` ramps 1→0.
#[allow(clippy::too_many_arguments)]
fn emit_trapezoid(
    yt: f32, yb: f32,
    xl_t: f32, xl_b: f32, dl: f32,
    xr_t: f32, xr_b: f32, dr: f32,
    w: usize, h: usize, mask: &mut [f32], owner: &mut [Tri],
) {
    // A coverage quad (corners top-left, bottom-left, top-right, bottom-right) as two triangles. Each pixel
    // keeps the higher coverage and the triangle it came from, in native's strip order [tl, tr, bl],
    // [tr, br, bl] (the DDI capture of cpg-solid-ell-b1); the coverage itself is order-independent.
    let mut quad = |tl: ([f32; 2], f32), bl: ([f32; 2], f32), tr: ([f32; 2], f32), br: ([f32; 2], f32)| {
        for (pos, cov, tri) in [
            ([tl.0, bl.0, tr.0], [tl.1, bl.1, tr.1], [tl.0, tr.0, bl.0]),
            ([bl.0, br.0, tr.0], [bl.1, br.1, tr.1], [tr.0, br.0, bl.0]),
        ] {
            // Native submits the quad as a strip, [tl, tr, bl] then [tr, br, bl], and WARP clips each triangle
            // to the viewport, fanning the clipped polygon before its plane fit: a band running past the
            // target bottom tilts the plane of the triangle cut there (r8 9705's COPY layer pop).
            let ord: [usize; 3] = core::array::from_fn(|k| (0..3).find(|&j| pos[j] == tri[k]).unwrap_or(k));
            crate::warp::visit_triangle_brush_cov(w, h, tri, ord.map(|j| cov[j]), true, crate::warp::Cull::None, |x, y, v| {
                let i = y * w + x;
                if v > mask[i] {
                    mask[i] = v;
                    owner[i] = tri;
                }
            });
        }
    };
    // Left fringe: x ∈ [xl−dl, xl+dl], coverage 0→1 (full band y-range).
    quad(([xl_t - dl, yt], 0.0), ([xl_b - dl, yb], 0.0), ([xl_t + dl, yt], 1.0), ([xl_b + dl, yb], 1.0));
    // Right fringe: x ∈ [xr−dr, xr+dr], coverage 1→0.
    quad(([xr_t - dr, yt], 1.0), ([xr_b - dr, yb], 1.0), ([xr_t + dr, yt], 0.0), ([xr_b + dr, yb], 0.0));
    // Interior (coverage 1) between the fringes' inner rails xLI = xl+dl and xRI = xr−dr. Where the band
    // tapers to narrower than dl+dr (near a vertex / thin shape) the rails cross; WPF emits no interior
    // there. Both rails are linear in y, so clamp the interior to the y-sub-range where xLI ≤ xRI — else
    // the inverted quad MAX-paints the gap to 1 (over-paint) and the overlapping fringes already cover it.
    let (xli_t, xli_b) = (xl_t + dl, xl_b + dl);
    let (xri_t, xri_b) = (xr_t - dr, xr_b - dr);
    let (gt, gb) = (xri_t - xli_t, xri_b - xli_b);
    if gt >= 0.0 && gb >= 0.0 {
        quad(([xli_t, yt], 1.0), ([xli_b, yb], 1.0), ([xri_t, yt], 1.0), ([xri_b, yb], 1.0));
    } else if gt < 0.0 && gb < 0.0 {
        // interior entirely collapsed — fringes cover the span
    } else {
        // Rails cross within the band; emit only the valid (wide) sub-range, a point at the crossover.
        let t = gt / (gt - gb);
        let ys = yt + t * (yb - yt);
        let xs = xli_t + t * (xli_b - xli_t);
        if gt >= 0.0 {
            quad(([xli_t, yt], 1.0), ([xs, ys], 1.0), ([xri_t, yt], 1.0), ([xs, ys], 1.0));
        } else {
            quad(([xs, ys], 1.0), ([xli_b, yb], 1.0), ([xs, ys], 1.0), ([xri_b, yb], 1.0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Native rot-rect COPY oracle: rect [9,11,31,29] under M=[c,s,-s,c,tx,ty] (c=0.88295, s=0.46947,
    // tx=11.73048, ty=-7.04838 — 28° about (20,20)), dpi 96, solid black, D2D1_PRIMITIVE_BLEND_COPY.
    // Coverage values captured from the VM (alpha = 102 + 128·cov over a 0.4-dst). Our trapezoid-fringe
    // generator reproduces every one byte-exact: interior 1.0, both slanted fringes, and the band
    // top/bottom rows (row 8, row 31) that distinguish the dispatch from a plain area rasterizer.
    #[test]
    fn rot_rect_copy_matches_native() {
        let (c, s, tx, ty) = (0.88295f32, 0.46947, 11.73048, -7.04838);
        let m = |x: f32, y: f32| [c * x - s * y + tx, s * x + c * y + ty];
        let poly = [m(9.0, 11.0), m(31.0, 11.0), m(31.0, 29.0), m(9.0, 29.0)];
        let (w, h) = (40usize, 40usize);
        let mut mask = vec![0.0f32; w * h];
        fill_convex_copy(&poly, w, h, false, |x, y, cov, _| {
            mask[y * w + x] = match cov {
                Cov::Area(s) => s as f32 / 64.0,
                Cov::Trapezoid(c) => c,
            };
        });
        // (x, y, native coverage). Tolerance 0.012 ≈ 1.5 LSB of the dst-alpha quantisation (1/128).
        let samples = [
            (20, 20, 1.0),
            (14, 8, 1.0),
            (15, 8, 1.0),
            (16, 8, 0.851562),
            (13, 8, 0.351562),
            (18, 8, 0.15625),
            (21, 31, 0.179688),
            (23, 31, 0.867188),
            (26, 31, 0.414062),
            (10, 25, 0.28125),
            (30, 24, 0.25),
            (29, 14, 0.265625),
            (7, 20, 0.632812),
            (31, 16, 0.875),
        ];
        for (x, y, want) in samples {
            let got = mask[y * w + x];
            assert!((got - want).abs() <= 0.012, "({x},{y}) got {got} want {want}");
        }
    }

    // cpg-solid-ell-b1's top cap row against native's DDI vertex buffer: per-pixel coverage (n/64) and the
    // AddComplexScan triangle each pixel is drawn with (pairs of single-pixel intervals share one).
    #[test]
    fn ellipse_cap_row_matches_native_complex_scan() {
        let poly = [[6.0999994f32, 15.599999], [6.375, 13.6875], [7.0, 11.875], [8.0625, 10.1875], [9.375, 8.75],
            [12.9375, 6.6875], [15.0625, 6.125], [17.3125, 5.9375], [17.3, 5.9000006], [19.5625, 6.125],
            [21.6875, 6.6875], [25.25, 8.75], [26.5625, 10.1875], [27.625, 11.875], [28.25, 13.6875], [28.5, 15.625],
            [28.5, 15.6], [28.25, 17.5625], [27.625, 19.4375], [26.5625, 21.0625], [25.25, 22.5], [21.6875, 24.5625],
            [19.5625, 25.125], [17.3125, 25.3125], [17.3, 25.3], [15.0625, 25.125], [12.9375, 24.5625], [9.375, 22.5],
            [8.0625, 21.0625], [7.0, 19.4375], [6.375, 17.5625], [6.125, 15.625], [6.0999994, 15.599999]];
        let native = [(12, 7, 12, 14), (13, 30, 12, 14), (14, 46, 14, 16), (15, 58, 14, 16), (16, 64, 16, 18),
            (17, 64, 16, 18), (18, 62, 18, 20), (19, 54, 18, 20), (20, 40, 20, 22), (21, 22, 20, 22), (22, 1, 22, 23)];
        let mut row6 = Vec::new();
        let mut above = 0;
        fill_convex_copy(&poly, 36, 32, true, |x, y, cov, tri| match (y, cov) {
            (6, Cov::Area(n)) => row6.push((x, n, tri[0][0] as usize, tri[1][0] as usize)),
            (0..=5, _) => above += 1,
            _ => {}
        });
        assert_eq!(above, 0, "native draws nothing above row 6");
        assert_eq!(row6, native.to_vec());
    }

    // r9 14009: a convex fill running past the target bottom, with a vertex (an inactive edge) below it,
    // grew a trapezoid band past `h` and sliced `band` out of range. Rows past the target are not emitted.
    #[test]
    fn band_past_target_bottom_is_clipped() {
        let poly = [[2.3f32, 3.1], [13.7, 2.6], [15.2, 45.0], [9.0, 120.0], [1.4, 60.0]];
        let (w, h) = (16usize, 20usize);
        let mut rows = 0usize;
        fill_convex_copy(&poly, w, h, true, |x, y, _, _| {
            assert!(x < w && y < h, "emitted ({x},{y}) outside {w}x{h}");
            rows = rows.max(y + 1);
        });
        assert_eq!(rows, h);
    }
}
