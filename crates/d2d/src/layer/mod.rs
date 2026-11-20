//! Phase 2A layers, clips and composition: the composite-back arithmetic for `PopAxisAlignedClip`
//! and `PopLayer`, and the `DrawImage` composite table, all in WARP's UNORM8 fixed point.
//! `crate::context` owns the push/pop stack and
//! the target redirection; this module owns the pixel math so it stays out of the 1A state machine.
//!
//! A pushed clip/layer redirects drawing into a working [`PixelBuffer`]; `Pop` composites it back:
//!   * **clip** — working starts as a copy of the backdrop (draws blend against the real target),
//!     so the pop reveals it through the clip coverage: `out = lerp(saved, working, clipCov)`.
//!   * **layer** — working starts transparent (group isolation), so the pop composites it over the
//!     backdrop modulated by `opacity × contentBounds coverage × geometric-mask × opacity-brush`.
//!
//! Coverage rules read from the oracle (`layers-01`/`02`/`06`): an **aliased** axis-aligned clip is a
//! hard device scissor whose edges snap round-half-down (`ceil(x−0.5)`); an **antialiased** clip (and
//! a layer's fractional content bounds) uses the analytic per-axis fractional overlap quantised
//! round-half-down — distinct from the fill rasteriser's 8×8 round-to-even (`crate::pixels`).

use crate::bitmap::Bitmap;
use crate::sampling::{Extend, Texture};
use d2d_types::contracts::PixelBuffer;

/// WARP's UNORM8 multiply `round(v·m/255)`: `t = v·m + 128; (t + (t>>8)) >> 8` — the fixed-point
/// form the whole pixel pipeline uses (shared with `crate::bitmap`'s sampler dump).
pub(crate) fn unorm_mul(v: i32, m: i32) -> i32 {
    let t = v * m + 128;
    (t + (t >> 8)) >> 8
}

/// WARP's **blend-unit** single UNORM8 multiply: `p = v·m; (p + (p>>8) + 128) >> 8`. The PopLayer
/// composite kernel (layerfill JIT capture, k885000) modulates and composites with this — the high-byte
/// correction `p>>8` is taken *before* the single `+128` round, unlike `unorm_mul`, which folds `+128`
/// into the correction and over-rounds ~1 LSB per factor (the error compounds across the opacity /
/// bounds-coverage / brush / mask chain — fuzz r5 5814's d12).
fn blend_mul(v: i32, m: i32) -> i32 {
    let p = v * m;
    (p + (p >> 8) + 128) >> 8
}

/// Snap a device coordinate to the aliased-clip grid: round half **down** = `ceil(x − 0.5)`.
fn snap(x: f32) -> i32 {
    (x - 0.5).ceil() as i32
}

/// Whether a content-bounds axis `[lo, hi]` (device-space, `device_rect`-sorted so `lo ≤ hi`)
/// composites to nothing. Native converts the bounds to its fixed-point clip with x86 `cvttss2si`,
/// which sends a finite but huge **positive** max edge (≥ i32::MAX, e.g. f32::MAX = 3.4e38) to
/// `i32::MIN` — collapsing it left of the min edge → that axis is empty. The exception is when the
/// min edge also −overflows (`[-3.4e38, +3.4e38]`): native then keeps the full unbounded span. Rust's
/// saturating `as i32` instead sends +overflow to `i32::MAX`, so we drew the whole target width.
/// VM-pinned (fuzz 6923 + probes): `[16.46, 3.4e38]`/`[-10, 3.4e38]`/`[20, 3.4e38]` → empty;
/// `[-3.4e38, 3.4e38]` → full; `[-3.4e38, 16.46]` → kept. Negatives already matched (Rust saturates
/// −overflow to `i32::MIN` too), so only the finite +overflow max edge changes.
fn axis_empty(lo: f32, hi: f32) -> bool {
    hi >= 2147483648.0 && !(lo < -2147483648.0)
}

/// Native's strip clipper (`CTransformedBezierFlattener`, tolerance 0.25) turns a device rect edge that
/// overflows the negative fixed-point clip (`x < -2^31`, from a huge-finite / −inf coordinate the DPI
/// scale did not resolve) into a 45° DIAGONAL instead of a vertical edge. So a rect with such an X edge
/// clips to a band/triangle whose boundary at device scanline `y` is `finite_x - (bottom - y)` (slope 1,
/// meeting the finite bound at `y == bottom`). This geometry is shared by the layer triangle, the
/// `FillOpacityMask` band and the `DrawBitmap` band; each surface decides WHEN to invoke it (the layer
/// only when the raw rect is inverted — a valid huge-negative-left layer stays a plain rect, fuzz r12
/// 17157 — while the textured surfaces apply it on any huge-negative dest edge) and supplies its own
/// content (a solid clip vs the degenerate source→dest texcoord sample). `[strip_neg_x]` detects the edge.
pub(crate) fn strip_neg_x(edge: f32) -> bool {
    edge < -2147483648.0
}
/// The strip clip's diagonal left-boundary x at device scanline centre `y` (see [`strip_neg_x`]).
pub(crate) fn strip_diag_left_x(finite_x: f32, bottom: f32, y: f32) -> f32 {
    finite_x - (bottom - y)
}

/// Coverage byte (0..=255) of a layer's axis-aligned content bounds at pixel `(px,py)`. d2d1 draws
/// the ring between the bounds' integer rect and the bounds themselves as one alternate-fill WARP
/// geometry: each point snaps to 28.4 as `ceil(16·x − 8)`, and WARP's 8×8 coverage rasterizer puts
/// it on sub-pixel `ceil((X + 7) / 2)`. The ring gets `(255·(64 − c) + 32) >> 6`, so the content keeps
/// `255 −` that, which is `(255·c + 31) >> 6` (measured via VM probes, layers-13/-14).
// d2d1 clips the bounds to the target first; the clamp keeps huge/infinite bounds in range.
fn eighth_q(v: f32) -> i64 {
    let x16 = (v.clamp(-1048576.0, 1048576.0) * 16.0 - 8.0).ceil() as i64;
    (x16 + 8).div_euclid(2)
}

/// Eighth-pixel coverage (0..=8) of the `[lo,hi]` edge pair (in eighth units) at pixel column/row `p`.
fn eighth_axis(lo: i64, hi: i64, p: i64) -> i64 {
    if p < lo >> 3 || p >= (hi + 7) >> 3 {
        return 0;
    }
    if lo >> 3 == hi >> 3 {
        return hi - lo; // both edges inside this pixel
    }
    if p == lo >> 3 && lo & 7 != 0 {
        8 - (lo & 7)
    } else if p == hi >> 3 {
        hi & 7
    } else {
        8
    }
}

fn layer_bounds_cova(rect: [f32; 4], px: i32, py: i32) -> i32 {
    let [l, t, r, b] = rect.map(eighth_q);
    let c = eighth_axis(l, r, px as i64) * eighth_axis(t, b, py as i64);
    ((255 * c + 31) >> 6) as i32
}

/// Content weight (0..=255) of an antialiased axis-aligned clip at pixel `(px,py)`: the product of its
/// four edge half-plane coverages, quantised as one strip pass of [`clip_strip_passes`] would be. Zero
/// tells a pixel the clip leaves as the backdrop.
fn clip_strip_cova(rect: [f32; 4], px: i32, py: i32) -> i32 {
    let [l, t, r, b] = rect;
    let (x, y) = (px as f32, py as f32);
    let (cl, cr) = ((x + 1.0 - l).clamp(0.0, 1.0), (r - x).clamp(0.0, 1.0));
    let (ct, cb) = ((y + 1.0 - t).clamp(0.0, 1.0), (b - y).clamp(0.0, 1.0));
    let cov = cl * cr * ct * cb;
    if cov >= 1.0 {
        return 255;
    }
    let b256 = ((1.0 - cov) * 256.0 + 0.5).floor() as i32;
    255 - (255 * b256 + 128) / 256
}

/// The background coverages with which d2d1's `ClipRectSubTarget` redraws the saved backdrop over pixel
/// `(px, py)` when an antialiased clip pops, in its drawing order. The interior is the working surface as
/// it is; then the left and right edge strips (over the rows the clip covers whole), the top and bottom
/// strips (over the columns it covers whole) and the four corners each redraw the backdrop at the share
/// of their pixel outside the clip: `l − ⌊l⌋`, `⌈r⌉ − r`, `t − ⌊t⌋`, `⌈b⌉ − b`, and for a corner
/// `1 − (1 − h)(1 − v)` of its two sides. A clip with both edges of an axis in one pixel draws both of
/// them there, one after the other (a sliver clip y [5.664, 5.780] redraws row 5 at 0.664, then 0.220).
fn clip_strip_passes(rect: [f32; 4], px: i32, py: i32) -> ([f32; 8], usize) {
    let [l, t, r, b] = rect;
    let (fl, cl, ft, ct) = (l.floor(), l.ceil(), t.floor(), t.ceil());
    let (fr, cr, fb, cb) = (r.floor(), r.ceil(), b.floor(), b.ceil());
    let (x, y) = (px as f32, py as f32);
    let (gl, gr, gt, gb) = ((fl - l).abs(), (cr - r).abs(), (ft - t).abs(), (cb - b).abs());
    let corner = |h: f32, v: f32| 1.0 - (1.0 - h) * (1.0 - v);
    let rows = ct <= y && y < fb;
    let cols = cl <= x && x < fr;
    let passes = [
        (cl > fl && x == fl && rows, gl),
        (cr > fr && x == fr && rows, gr),
        (ct > ft && y == ft && cols, gt),
        (cb > fb && y == fb && cols, gb),
        (cl > fl && ct > ft && x == fl && y == ft, corner(gl, gt)),
        (cr > fr && ct > ft && x == fr && y == ft, corner(gr, gt)),
        (cl > fl && cb > fb && x == fl && y == fb, corner(gl, gb)),
        (cr > fr && cb > fb && x == fr && y == fb, corner(gr, gb)),
    ];
    let mut out = ([0.0; 8], 0);
    for (on, g) in passes {
        if on {
            out.0[out.1] = g;
            out.1 += 1;
        }
    }
    out
}

/// One entry on the context's push/pop stack: the backdrop it will composite back into plus the
/// parameters of the clip or layer. `saved` is the target that was bound before the push.
pub(crate) enum Frame {
    /// `min_rule`: per-pixel marks (target-sized, empty = none) of pixels a gradient FillOpacityMask
    /// drew, which composite back with the box-distance clip coverage ([`clip_min_cova`]).
    /// `corner`: the one pixel a zero-area clip passes ([`degenerate_clip_pixel`]).
    Clip { saved: PixelBuffer, rect: [f32; 4], aliased: bool, min_rule: Vec<u8>, corner: Option<(usize, usize)> },
    Layer { saved: PixelBuffer, params: LayerParams },
}

/// An opacity brush and the device rect of the pixels its pop fills, with what the pop needs to place it again: its
/// brush→device matrix at the push (before a bitmap's DPI), the unit mode then and the layer opacity.
#[derive(Clone)]
pub(crate) struct PopBrush {
    pub brush: crate::PixelBrush,
    pub region: [f32; 4],
    pub b2d: [f32; 6],
    pub unit: u32,
    pub opacity: f32,
}

impl std::fmt::Debug for PopBrush {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PopBrush({:?})", self.region)
    }
}

/// A pushed layer's composite-back parameters: the device-space content bounds, the constant opacity
/// (0..=255) and the optional per-pixel geometric-mask coverage / opacity-brush alpha (target-sized).
#[derive(Clone, Debug)]
pub(crate) struct LayerParams {
    pub bounds: [f32; 4],
    /// Per-pixel content-bounds coverage when the layer bounds are NOT an axis-aligned device rect
    /// (a rotated/sheared/singular device transform): native clips the layer to the transformed
    /// bounds QUAD as geometry, so this holds that quad's rasterised coverage and `bounds` is just its
    /// AABB (the loop extent). `None` ⇒ the axis-aligned `layer_bounds_cova(bounds)` coverage.
    pub bounds_mask: Option<Vec<u8>>,
    /// The device corners of those non-axis-aligned bounds on the layer an axis-aligned clip becomes
    /// when a command list replays it under a rotation or skew: its pop restores the backdrop through
    /// the quad's inverse (see `layer_back`).
    pub init_quad: Option<[[f32; 2]; 4]>,
    /// Any RAW (pre-transform) layer bound was NaN: the layer clips nothing for later draws' reach.
    pub bounds_nan: bool,
    /// The opacity brush paints nothing (a singular bitmap or image brush matrix): the pop draws nothing
    /// and the backdrop stays as it was before the push (r27 32420).
    pub void_pop: bool,
    /// The pop draws nothing (a singular opacity brush, see `push_layer_as`): the layer's pixels keep its content as
    /// drawn on the cleared (or initialized) layer, the backdrop left out.
    pub keep_pop: bool,
    /// D2D1_LAYER_OPTIONS1_INITIALIZE_FROM_BACKGROUND: the layer's content starts from the backdrop.
    pub init: bool,
    /// The opacity brush the pop fills the layer's pixels with, which it checks under the pop's transform.
    pub pop_brush: Option<PopBrush>,
    pub opacity8: i32,
    /// The raw constant opacity (0..=1). The 8-bit composite rounds it to `opacity8`; the float
    /// composite (`crate::float::layer_back_f`) applies this verbatim in f32 (oracle `hdr-12`).
    pub opacity: f32,
    pub mask: Option<Vec<u8>>,
    /// The geometric mask's device bounds: the layer's extent is its content bounds within them.
    pub mask_bounds: Option<[f32; 4]>,
    pub brush_alpha: Option<Vec<u8>>,
    /// The opacity-brush alpha × opacity unrounded, for the IGNORE_ALPHA pop's float blend.
    pub brush_alpha_f: Option<Vec<f32>>,
    /// `D2D1_LAYER_OPTIONS1_IGNORE_ALPHA`: the layer is treated as opaque, so the composite-back
    /// forces the working pixel's alpha to 255 before the opacity/mask factors (undrawn regions of
    /// the bounds become opaque black × the factors), per Direct2D's layer setup.
    pub ignore_alpha: bool,
    /// The layer's `maskAntialiasMode == ALIASED`: the axis-aligned content-bounds rect is a hard
    /// centre-sample clip, not the eighth-pixel antialiased ring (`bounds_mask`/`layer_bounds_cova`).
    pub aliased: bool,
    /// The RAW (pre-transform) layer bounds rect was inverted (`l > r` or `t > b`); `device_rect` sorts
    /// it, and the overflow shapes below tell the two apart.
    pub inverted: bool,
    /// The primitive blend active when the layer was PUSHED — PopLayer composites the layer back under
    /// it, not under the blend current at pop (fuzz: COPY set before push replaces the backdrop, COPY
    /// set inside the layer does not). Only COPY (at full coverage) diverges from source-over so far.
    pub blend: crate::pixels::Blend,
    /// Direct2D's own layer clip (see [`NativeClip`]); `None` for a layer with a geometric mask, whose
    /// clip geometry (already cut to the content bounds) is the whole coverage in `mask`.
    pub native: Option<NativeClip>,
    /// The one pixel a zero-width or zero-height layer passes as a zero-area antialiased clip does
    /// ([`degenerate_clip_pixel`]), and whether it is live: once a command list records while the layer is
    /// pushed, the layer's draws reach that pixel whole and the pop leaves it as drawn (probe-zwh: SetTarget
    /// to a list and back inside a zero-width layer; r27 32747, 32842).
    pub corner: Option<(usize, usize)>,
    pub corner_live: bool,
}

/// How Direct2D's layer setup and pop treat this layer. Direct2D draws a layer's content straight onto
/// the target: the push saves the backdrop under the layer's pixel rect and clears that rect (unless
/// INITIALIZE_FROM_BACKGROUND), and the pop fades the content by the opacity (a DEST_IN rect fill),
/// erases it outside the content-bounds geometry (a DEST_OUT fill of the geometry's inverse), then puts
/// the saved backdrop back underneath (DEST_OVER). The pop stops at the first draw that fails, and the
/// batch executor drops a `D2DERR_BAD_NUMBER` silently — so when the bounds geometry cannot be filled,
/// the rect keeps the faded content on the cleared backdrop.
#[derive(Clone, Debug)]
pub(crate) struct NativeClip {
    /// The layer's device pixel rect `[x0, y0, x1, y1]`, clamped to the target (empty: the layer
    /// composites nothing).
    pub rect: [i32; 4],
    /// The rect has a NaN edge: the push clip is empty and the pop composites nothing.
    pub nan: bool,
    /// The pop's bounds-geometry fill fails, so the rect is left holding the faded content.
    pub fail: bool,
    /// The axis-aligned bounds rect leaves the safe device box but clips to it with every crossing
    /// finite, so it still rasterises as the plain clipped rect.
    pub exact: bool,
    /// No bounds geometry clips the pop (bounds unbounded on both ends of an axis, or an axis-kept rect
    /// on whole pixels): the rect itself is the hard edge, at full coverage.
    pub hard: bool,
    /// The axis-aligned bounds rect's coverage as its clipped figure, when the safe-box clip leaves
    /// infinite (not NaN) crossings: they sit at the far corner, as the fill path rasterises them.
    pub far: Option<Vec<u8>>,
}

/// Direct2D's device bounds of the layer's bounds rectangle under the layer transform: the rect is
/// normalised (`l > r` swaps; a NaN edge does not), and only a valid one is transformed — its four corners `(m21·y + m11·x) + dx` in
/// f32 (a full multiply, so an infinite edge times a zero matrix term is NaN). The per-axis min/max is
/// NaN-PROPAGATING: one NaN corner coordinate makes both bounds of that axis NaN. An invalid (NaN) rect
/// comes back untransformed.
pub(crate) fn rect_fill_bounds(b: [f32; 4], m: &[f32; 6]) -> [f32; 4] {
    let (l, r) = if b[0] > b[2] { (b[2], b[0]) } else { (b[0], b[2]) };
    let (t, bt) = if b[1] > b[3] { (b[3], b[1]) } else { (b[1], b[3]) };
    if !(l <= r && t <= bt) {
        return [l, t, r, bt];
    }
    let c = rect_corners([l, t, r, bt], m);
    let fold = |v: [f32; 4]| -> (f32, f32) {
        if v.iter().any(|x| x.is_nan()) {
            return (f32::NAN, f32::NAN);
        }
        (v.iter().copied().fold(f32::INFINITY, f32::min), v.iter().copied().fold(f32::NEG_INFINITY, f32::max))
    };
    let (x0, x1) = fold(c.map(|q| q[0]));
    let (y0, y1) = fold(c.map(|q| q[1]));
    [x0, y0, x1, y1]
}

/// The four corners of a normalised rect under the full f32 multiply `(m21·y + m11·x) + dx` that
/// Direct2D's bounds and fill-overflow checks use.
pub(crate) fn rect_corners([l, t, r, b]: [f32; 4], m: &[f32; 6]) -> [[f32; 2]; 4] {
    let p = |x: f32, y: f32| [(m[2] * y + m[0] * x) + m[4], (m[3] * y + m[1] * x) + m[5]];
    [p(l, t), p(r, t), p(l, b), p(r, b)]
}

/// Whether a transform keeps rects axis-aligned: no shear, or a pure 90° swap.
pub(crate) fn axis_preserving(m: &[f32; 6]) -> bool {
    m[1] == 0.0 && m[2] == 0.0 || m[0] == 0.0 && m[3] == 0.0
}

/// Direct2D's round-halves-down of one rect edge: `ceil(x − 0.5)` saturated to the i32 range (a NaN
/// lands on `i32::MIN + 1`).
fn round_halves_down(x: f32) -> f32 {
    let v = x - 0.5;
    if v < -2147483600.0 {
        -2147483648.0
    } else if v >= 2147483600.0 {
        2147483647.0
    } else {
        let i = if v.is_nan() { i32::MIN } else { v as i32 };
        (if v <= i as f32 { i } else { i + 1 }) as f32
    }
}

/// A valid rect whose edges already sit on the integer grid (they survive round-halves-down).
pub(crate) fn pixel_aligned(r: [f32; 4]) -> bool {
    r[0] <= r[2] && r[1] <= r[3] && r.iter().all(|&v| round_halves_down(v) == v)
}

/// The layer's device rect as Direct2D's layer setup derives it from the bounds geometry's device
/// bounds `dev`: clamped to the `w×h` target, intersected with the current clip `clip`, then rounded
/// out (floor the min edges, ceil the max ones; a value past `2^23`, ±inf or NaN passes through).
/// `None` when the intersection is empty. NaN-safe in the same way as native's compare chain.
pub(crate) fn layer_rect(dev: [f32; 4], w: f32, h: f32, clip: [f32; 4]) -> Option<[f32; 4]> {
    let [mut l, mut t, mut r, mut b] = dev;
    if l < 0.0 {
        l = 0.0;
    }
    if t < 0.0 {
        t = 0.0;
    }
    if w < r {
        r = w;
    }
    if h < b {
        b = h;
    }
    if clip[2] < l {
        l = clip[2];
    }
    if clip[3] < t {
        t = clip[3];
    }
    if r < clip[0] {
        r = clip[0];
    }
    if b < clip[1] {
        b = clip[1];
    }
    if l < clip[0] {
        l = clip[0];
    }
    if t < clip[1] {
        t = clip[1];
    }
    if clip[2] < r {
        r = clip[2];
    }
    if clip[3] < b {
        b = clip[3];
    }
    if r <= l || b <= t {
        return None;
    }
    let keep = |v: f32| v.is_nan() || v.abs() > 8388608.0;
    let lo = |v: f32| if keep(v) { v } else { v.floor() };
    let hi = |v: f32| if keep(v) { v } else { v.ceil() };
    Some([lo(l), lo(t), hi(r), hi(b)])
}

/// `PopAxisAlignedClip` composite-back. Aliased: a hard scissor copying the
/// working pixels back inside the round-half-down integer rect. Antialiased: reveal the working
/// result through the analytic edge coverage in WARP's blend unit, the fused high-byte form
/// `(s·c + (s·c >> 8) + d·(255−c) + (d·(255−c) >> 8) + 128) >> 8`, whatever blend drew the content (r6
/// 6575 under COPY; r22 27633 a source-over mesh).
/// The clip-edge coverage a gradient-brush FillOpacityMask takes: the edge product with each axis
/// ramp the NEARER edge, `min(cl, cr)`, instead of the product of both edges (the both-edges extra
/// factors stay). Native applies the AA clip per draw with a draw-specific rule: a VM sweep of clip edges
/// over one pixel (fx7k/v18771c) fits all radial-FOM cases with the minimum — including the corner row,
/// `min(cl, cr)·cy²` — while a solid FOM and a gradient FillRectangle at the same pixel fit the product
/// (r13 18771).
fn clip_min_cova(rect: [f32; 4], px: i32, py: i32) -> i32 {
    let [l, t, r, b] = rect;
    let (x, y) = (px as f32, py as f32);
    let (cl, cr) = ((x + 1.0 - l).clamp(0.0, 1.0), (r - x).clamp(0.0, 1.0));
    let (ct, cb) = ((y + 1.0 - t).clamp(0.0, 1.0), (b - y).clamp(0.0, 1.0));
    let (cx, cy) = (cl.min(cr), ct.min(cb));
    let mut cov = cx * cy;
    if ct < 1.0 && cb < 1.0 {
        cov *= cx;
    }
    if cl < 1.0 && cr < 1.0 {
        cov *= cy;
    }
    if cov >= 1.0 {
        return 255;
    }
    let b256 = ((1.0 - cov) * 256.0 + 0.5).floor() as i32;
    255 - (255 * b256 + 128) / 256
}

/// The device pixel span `opacity_mask_fill_brush` fills for a device `dest` on a `w×h` target.
pub(crate) fn opacity_mask_span([dl, dt, dr, db]: [f32; 4], w: i32, h: i32) -> (i32, i32, i32, i32) {
    (snap(dl).clamp(0, w), snap(dr).clamp(0, w), snap(dt).clamp(0, h), snap(db).clamp(0, h))
}

/// A zero-width or zero-height antialiased clip passes the one pixel holding its top-left corner whole, whatever draws
/// there, when both of that corner's coordinates are fractional; a corner on the pixel grid passes nothing (probe-zwd,
/// probe-zwe, probe-zwf: every blend and composite, inverted or not). `rect` is the device rect of a finite clip: one
/// made degenerate by a NaN or infinite coordinate passes nothing (r27 32142, r13 18651).
pub(crate) fn degenerate_clip_pixel(rect: [f32; 4], aliased: bool, w: i32, h: i32) -> Option<(usize, usize)> {
    let [l, t, r, b] = rect;
    if aliased || !(l == r || t == b) {
        return None;
    }
    let (x, y) = (l.min(r), t.min(b));
    let (px, py) = (x.floor(), y.floor());
    (x != px && y != py && px >= 0.0 && py >= 0.0 && px < w as f32 && py < h as f32).then_some((px as usize, py as usize))
}

/// Copy one pixel of `src` into `dst` (same size and format).
pub(crate) fn copy_pixel(dst: &mut PixelBuffer, src: &PixelBuffer, x: usize, y: usize) {
    let Some(bpp) = crate::bitmap::bytes_per_pixel(dst.desc.format) else { return };
    let i = y * dst.stride + x * bpp;
    if let (Some(d), Some(s)) = (dst.data.get_mut(i..i + bpp), src.data.get(i..i + bpp)) { d.copy_from_slice(s); }
}

pub(crate) fn clip_back(saved: &mut PixelBuffer, working: &PixelBuffer, rect: [f32; 4], aliased: bool, min_rule: &[u8], ignore: bool, corner: Option<(usize, usize)>) {
    let (w, h, st) = (saved.desc.width as i32, saved.desc.height as i32, saved.stride);
    if let Some((x, y)) = corner {
        let i = y * st + x * 4;
        saved.data[i..i + 4].copy_from_slice(&working.data[i..i + 4]);
        return;
    }
    let [l, t, r, b] = rect;
    // An empty clip passes nothing: the edge-product coverage of a zero-width strip would leak
    // `cl·cr` into the pixel it sits in (r16 21988: an inverted AA clip, native unchanged).
    if !(r > l && b > t) {
        return;
    }
    let (x0, x1, y0, y1) = if aliased {
        (snap(l).clamp(0, w), snap(r).clamp(0, w), snap(t).clamp(0, h), snap(b).clamp(0, h))
    } else {
        (
            (l.floor() as i32).clamp(0, w),
            (r.ceil() as i32).clamp(0, w),
            (t.floor() as i32).clamp(0, h),
            (b.ceil() as i32).clamp(0, h),
        )
    };
    for py in y0..y1 {
        for px in x0..x1 {
            let cova = if aliased {
                255
            } else if min_rule.get(py as usize * w as usize + px as usize).is_some_and(|&m| m != 0) {
                clip_min_cova(rect, px, py)
            } else {
                clip_strip_cova(rect, px, py)
            };
            if cova == 0 {
                continue;
            }
            let i = py as usize * st + px as usize * 4;
            let min = min_rule.get(py as usize * w as usize + px as usize).is_some_and(|&m| m != 0);
            if !aliased && !min && !ignore {
                let (passes, n) = clip_strip_passes(rect, px, py);
                let mut cur: [i32; 4] = core::array::from_fn(|k| working.data[i + k] as i32);
                for &g in &passes[..n] {
                    let ws = (255 * (g * 256.0 + 0.5).floor() as i32 + 128) / 256;
                    for k in 0..4 {
                        let (sv, dv) = (saved.data[i + k] as i32 * ws, cur[k] * (255 - ws));
                        cur[k] = ((sv + (sv >> 8) + dv + (dv >> 8) + 128) >> 8).min(255);
                    }
                }
                for k in 0..4 {
                    saved.data[i + k] = cur[k] as u8;
                }
                continue;
            }
            // Inside an IGNORE_ALPHA layer the pop draws the saved backdrop back over the working surface
            // through quads at the INVERSE coverage (256ths): the left and right edge columns over the rows
            // the clip covers fully, the top and bottom edge rows over the columns it covers fully, then the
            // four corners at the product of their edges. Each quad samples the saved backdrop: colour
            // rne(saved/255·w·255), alpha rne(w·255), through WARP's source-over unit, so two edges in one
            // pixel are two passes (r18 23729; clip-in-layer JIT and DDI captures).
            if ignore && !aliased && !min_rule.get(py as usize * w as usize + px as usize).is_some_and(|&m| m != 0) {
                let (x, y) = (px as f32, py as f32);
                let (cl, cr) = ((x + 1.0 - l).clamp(0.0, 1.0), (r - x).clamp(0.0, 1.0));
                let (ct, cb) = ((y + 1.0 - t).clamp(0.0, 1.0), (b - y).clamp(0.0, 1.0));
                let (xl, xr, yt, yb) = (cl < 1.0, cr < 1.0, ct < 1.0, cb < 1.0);
                let passes = [
                    (!yt && !yb && xl, cl), (!yt && !yb && xr, cr),
                    (!xl && !xr && yt, ct), (!xl && !xr && yb, cb),
                    (xl && yt, cl * ct), (xr && yt, cr * ct), (xl && yb, cl * cb), (xr && yb, cr * cb),
                ];
                let mut cur: [i32; 4] = core::array::from_fn(|k| working.data[i + k] as i32);
                for (_, cov) in passes.into_iter().filter(|p| p.0) {
                    let wt = ((1.0 - cov) * 256.0 + 0.5).floor() / 256.0;
                    let sa = (wt.clamp(0.0, 1.0) * 255.0).round_ties_even() as i32;
                    for k in 0..4 {
                        let v = saved.data[i + k] as f32;
                        let s = if k == 3 { sa } else { (((v * (1.0 / 256.0) + v * f32::from_bits(0x3780_8081)) * wt).clamp(0.0, 1.0) * 255.0).round_ties_even() as i32 };
                        let p = cur[k] * (255 - sa);
                        cur[k] = (((s << 8) + p + (p >> 8) + 128) >> 8).min(255);
                    }
                }
                for k in 0..4 { saved.data[i + k] = cur[k] as u8; }
                continue;
            }
            for k in 0..4 {
                let wv = working.data[i + k] as i32;
                let sv = saved.data[i + k] as i32;
                let (ws, ss) = (wv * cova, sv * (255 - cova));
                saved.data[i + k] = ((ws + (ws >> 8) + ss + (ss >> 8) + 128) >> 8).min(255) as u8;
            }
        }
    }
}

/// What a layer pop that stops after its DEST_IN fade leaves in `rect`: the layer's content (drawn on
/// the cleared backdrop, or for INITIALIZE_FROM_BACKGROUND on the backdrop itself) times the opacity and
/// opacity brush, replacing the backdrop. An IGNORE_ALPHA layer fades with its alpha taken as opaque
/// (fuzz r16 21161: an empty one leaves opaque black × opacity), through the float blend.
fn fade(saved: &mut PixelBuffer, working: &PixelBuffer, f: &LayerParams, [x0, y0, x1, y1]: [i32; 4], tile: [u32; 2]) {
    let (w, h, st) = (saved.desc.width as i32, saved.desc.height as i32, saved.stride);
    // The pop walks the layer in rendering-controls tiles, anchored at the bounds' min edge rounded from
    // the pixel centre and held to the target, and stops in the first tile that meets the rect: only that
    // tile fades; the rest keep the layer surface (r24 29094's 1×1 tiles).
    let [l, t, ..] = f.bounds;
    let (ax, ay) = (((l - 0.5).floor() as i32).clamp(0, w), ((t - 0.5).floor() as i32).clamp(0, h));
    let (tw, th) = (tile[0].min(1 << 20) as i32, tile[1].min(1 << 20) as i32);
    let (tx, ty) = (ax + (x0.max(0) - ax).max(0) / tw * tw, ay + (y0.max(0) - ay).max(0) / th * th);
    let in_tile = |px: i32, py: i32| (tx..tx + tw).contains(&px) && (ty..ty + th).contains(&py);
    for py in y0.max(0)..y1.min(h) {
        for px in x0.max(0)..x1.min(w) {
            let p = (py * w + px) as usize;
            let i = py as usize * st + px as usize * 4;
            if !in_tile(px, py) {
                saved.data[i..i + 4].copy_from_slice(&working.data[i..i + 4]);
                continue;
            }
            // IGNORE_ALPHA: the float fade, destination alpha taken as opaque, stored round-half-even.
            if f.ignore_alpha {
                // The float blend fades by the brush alpha × opacity unrounded (r17 22261, r21 26531).
                let a = match (&f.brush_alpha_f, &f.brush_alpha) {
                    (Some(bf), _) => bf[p],
                    (_, Some(b)) => b[p] as f32 / 255.0,
                    _ => f.opacity,
                };
                for k in 0..4 {
                    let d = if k == 3 { 1.0 } else { working.data[i + k] as f32 / 255.0 };
                    saved.data[i + k] = (d * a * 255.0).round_ties_even().clamp(0.0, 255.0) as u8;
                }
                continue;
            }
            let mut src: [i32; 4] = core::array::from_fn(|k| working.data[i + k] as i32);
            for m in [Some(f.opacity8), f.brush_alpha.as_ref().map(|b| b[p] as i32)].into_iter().flatten() {
                src = src.map(|c| blend_mul(c, m));
            }
            saved.data[i..i + 4].copy_from_slice(&src.map(|c| c as u8));
        }
    }
}

/// `PopLayer` composite-back: source-over the transparent-initialised working
/// buffer onto the backdrop, each pixel pre-multiplied by `opacity × contentBounds coverage ×
/// geometric-mask coverage × opacity-brush alpha`. Restricted to the content bounds.
/// An aliased layer's coverage of its rotated/sheared device bounds `quad`: Direct2D erases outside
/// it with an aliased fill of the quad's inverse (WARP's aliased walk, the inverse running past every
/// side of the target), so a pixel is in the layer when that fill skips it (clrot2 probes).
pub(crate) fn aliased_quad_cov(quad: &[[f32; 2]; 4], w: usize, h: usize) -> Vec<u8> {
    const SAFE: f32 = 524287.0;
    let outer = [[-SAFE, -SAFE], [SAFE + 2.0, -SAFE], [SAFE + 2.0, SAFE + 2.0], [-SAFE, SAFE + 2.0]];
    let fixed = |p: &[[f32; 2]]| p.iter().map(|p| p.map(|v| ((v.clamp(-SAFE, SAFE) - 0.5) * 16.0).ceil() as i32)).collect::<Vec<_>>();
    let mut cov = vec![255u8; w * h];
    crate::pixels::warp_geom::fill_aliased(&[fixed(&outer), fixed(quad)], false, w, h, |y, a, b| cov[y * w + a..y * w + b].fill(0));
    cov
}

/// A masked layer whose singular CUBIC or HIGH_QUALITY_CUBIC opacity brush voids its pop (`void_pop`) replaces the
/// backdrop with its content over the geometric mask's bounds within the layer's bounds, where an unmasked one leaves
/// the backdrop (VM probes: a masked layer with the singular brush of r27 32222, its bounds finite, inverted or with
/// an infinite edge: every pixel whose centre is in that rect takes the layer's content, transparent when empty, the
/// mask's shape aside; r27 32222's inner IGNORE_ALPHA layer shows there; no opacity brush, or a regular one, keeps
/// the backdrop).
// ponytail: probed with an ALIASED mask only; an antialiased one may ring the rect's fractional edges.
pub(crate) fn void_pop_erase(saved: &mut PixelBuffer, working: &PixelBuffer, f: &LayerParams) {
    let (Some(_), Some(mb)) = (&f.mask, f.mask_bounds) else { return };
    let [l, t, r, b] = f.bounds;
    // The rect rounds out to whole pixels (32222: an ellipse top at 17.85 takes row 17, a layer left at 7.59 column 7).
    let rect = [l.max(mb[0]).floor(), t.max(mb[1]).floor(), r.min(mb[2]).ceil(), b.min(mb[3]).ceil()];
    let (w, h, st) = (saved.desc.width as i32, saved.desc.height as i32, saved.stride);
    for py in 0..h {
        for px in 0..w {
            let (cx, cy) = (px as f32 + 0.5, py as f32 + 0.5);
            if rect[0] <= cx && cx < rect[2] && rect[1] <= cy && cy < rect[3] {
                let i = py as usize * st + px as usize * 4;
                saved.data[i..i + 4].copy_from_slice(&working.data[i..i + 4]);
            }
        }
    }
}

pub(crate) fn layer_back(saved: &mut PixelBuffer, working: &PixelBuffer, f: &LayerParams, tile: [u32; 2]) {
    let (w, h, st) = (saved.desc.width as i32, saved.desc.height as i32, saved.stride);
    if let Some(nc) = &f.native {
        let [x0, y0, x1, y1] = nc.rect;
        if x1 <= x0 || y1 <= y0 || nc.nan {
            return;
        }
        // A pop that draws nothing leaves the layer as drawn within the pixel rect its push clipped to, at full weight:
        // the bounds geometry (its antialiased edge, a rotated quad) only acts in the pop's own fills (r27 32560: a quad
        // whose corners overflow covers nothing, yet the cleared layer stays; r27 32771). An aliased layer's rect keeps
        // the pixels whose centres it holds (r27 32862).
        if f.keep_pop {
            let [l, t, r, b] = f.bounds;
            for py in y0..y1 {
                for px in x0..x1 {
                    let (cx, cy) = (px as f32 + 0.5, py as f32 + 0.5);
                    if f.aliased && !(l <= cx && cx < r && t <= cy && cy < b) {
                        continue;
                    }
                    let i = py as usize * st + px as usize * 4;
                    let under = if f.ignore_alpha && !f.init { 255 - working.data[i + 3] as i32 } else { 0 };
                    for k in 0..4 {
                        saved.data[i + k] = (working.data[i + k] as i32 + blend_mul(saved.data[i + k] as i32, under)) as u8;
                    }
                }
            }
            return;
        }
        if nc.fail {
            fade(saved, working, f, nc.rect, tile);
            return;
        }
    }
    let hard = f.native.as_ref().filter(|n| n.hard).map(|n| n.rect).or(f.mask.is_some().then_some([0, 0, w, h]));
    let [l, t, r, b] = f.bounds;
    let [x0, y0, x1, y1] = hard.unwrap_or([
        (l.floor() as i32).clamp(0, w),
        (t.floor() as i32).clamp(0, h),
        (r.ceil() as i32).clamp(0, w),
        (b.ceil() as i32).clamp(0, h),
    ]);
    // The pop's fills run under the layer's own pixel rect as their edge-walk clip (a trace of the pop's
    // inverse fill reads the bounds rounded out, cut to the target).
    let pop_clip = [x0, y0, x1, y1];
    // A non-inverted rect whose only out-of-range edge is a finite-huge bottom composites a hard pixel
    // rect down to the target bottom (fuzz r17 22432).
    let in_range = |x: f32| x.is_finite() && x.abs() < 2147483648.0;
    let clip_to_target = f.bounds_mask.is_none() && !f.inverted
        && in_range(l) && in_range(t) && in_range(r)
        && b.is_finite() && b.abs() >= 2147483648.0;
    // The overflow shapes below approximate a bounds rect's safe-box clip; they remain only for the
    // sheared quad's AABB (fuzz r19 24134). An axis-aligned rect that clips exactly, one rasterised as its
    // clipped figure (`far`), one with no bounds geometry at all, or a masked layer — whose clip geometry
    // is the mask cut by the bounds, with infinite bounds held to ±FLT_MAX — never takes them (r19 24248).
    let far = f.native.as_ref().and_then(|n| n.far.as_ref());
    let exact = f.native.as_ref().map_or(true, |n| n.exact) || hard.is_some() || far.is_some();
    // A finite +overflow max bound (f32::MAX in the right/bottom slot after the device_rect sort)
    // collapses that axis under native's cvttss2si clip conversion: the interior vanishes (backdrop
    // shows through, unlike the collapse-clear above which zeroes it). Only the min-edge pixel survives,
    // and only when that edge is sub-pixel: native rasterizes the collapsed rect's fractional min edge
    // as a thin antialiased sliver (fuzz 6923: l=16.46 keeps col 16 with its ring coverage), but an
    // integer-aligned min edge leaves nothing (probes l=0/20/-10). So keep just the min-edge column
    // when the edge is fractional, else drop the axis entirely.
    // A sheared quad's AABB never collapses: its bounds geometry clips (VM shear probes, r24 29259).
    let quad = f.bounds_mask.is_some();
    let x1 = if axis_empty(l, r) && !exact && !quad {
        if l.fract() != 0.0 { (l.floor() as i32).saturating_add(1).clamp(0, w).min(x1).max(x0) } else { x0 }
    } else {
        x1
    };
    let y1 = if axis_empty(t, b) && !clip_to_target && !exact && !quad {
        if t.fract() != 0.0 { (t.floor() as i32).saturating_add(1).clamp(0, h).min(y1).max(y0) } else { y0 }
    } else {
        y1
    };
    // A huge-negative device RIGHT bound (sorted l < -2^31, from an INVERTED raw rect whose right edge is
    // a huge-finite DIP value the DPI scale did not push to ±inf) is flattened by native's strip clipper
    // (CTransformedBezierFlattener, tolerance 0.25) into a 45° DIAGONAL, so the axis-aligned layer clips
    // to a right TRIANGLE: vertical edge at the finite bound `r`, apex (r, b), diagonal x = r - (b - y).
    // The INVERSION gates it: a VALID rect with a huge-negative LEFT clamps to a plain [0, r] rect and
    // composites (fuzz r12 17157: left=-3.4e38 < right -> rect, native keeps it). The +2^31 max side
    // collapses to empty via axis_empty. Probe-confirmed vs native and the layer-pop
    // composite gate; fuzz r17 22633. Hard edge for now — the diagonal's sub-pixel AA is a ≤1-LSB residual.
    // The layer invokes the shared strip-clip diagonal only when the raw rect is INVERTED (a valid
    // huge-negative-left layer keeps a plain rect, fuzz r12 17157); the diagonal's vertical edge is the
    // finite bound `r`, apex (r, b). (Textured surfaces reuse `strip_diag_left_x` with their own texcoord.)
    let tri_neg_x = f.bounds_mask.is_none() && strip_neg_x(l) && f.inverted && !exact;
    // COPY puts the backdrop back through the inverse of the bounds rect, filled by the COPY rasterizer's
    // walk (DDI of r8 9705's pop): x from −524287 to 524289 and the layer's pixel rows, less the bounds
    // strip-clipped to the safe box (±524287). So a bound past the left edge lands on the inverse's own
    // (a zero-width span: the walk scans complex rows, r13 18436) while one past the right stays two
    // pixels in (r17 22667). `inv` is that fill's coverage, the backdrop's lerp weight.
    // Rotated or skewed bounds pushed under COPY take the same walk through the inverse of their device quad (VM probes
    // geo/c396 of a 53-degree layer at 192 dpi: native's edge coverage is the walk's fractional trapezoid, not the bounds
    // mask's eighths; 138 px with or without content, and r25 30396's INITIALIZE_FROM_BACKGROUND edge).
    let copy_quad = f.init_quad.filter(|_| f.blend == crate::pixels::Blend::Copy);
    let copy_walk = f.blend == crate::pixels::Blend::Copy && (f.bounds_mask.is_none() || copy_quad.is_some()) && !f.aliased && hard.is_none() && far.is_none();
    // The layer a command-list clip becomes under a rotation or skew pops in one draw: the saved
    // backdrop back through the quad's inverse by the same walk, a lerp (DDI of clclip p2); an aliased
    // clip's inverse is its aliased bounds mask's complement. A plain INITIALIZE_FROM_BACKGROUND layer
    // with the same bounds keeps the three-step pop (clrot probes).
    let quad_walk = f.init_quad.filter(|_| {
        f.blend == crate::pixels::Blend::SourceOver && f.opacity8 == 255 && f.mask.is_none() && f.brush_alpha.is_none() && !f.ignore_alpha
    });
    let inv = (copy_walk || quad_walk.is_some()).then(|| {
        const SAFE: f32 = 524287.0;
        let (oy0, oy1) = (y0 as f32, y1 as f32);
        let outer = [[-SAFE, oy0], [SAFE + 2.0, oy0], [SAFE + 2.0, oy1], [-SAFE, oy1]];
        let [cl, ct, cr, cb] = [l, t, r, b].map(|v| v.clamp(-SAFE, SAFE));
        let inner = quad_walk.or(copy_quad).map_or([[cl, ct], [cl, cb], [cr, cb], [cr, ct]], |q| q.map(|p| p.map(|v| v.clamp(-SAFE, SAFE))));
        let mut inv = vec![0u8; (w * h) as usize];
        if let (Some(_), true, Some(m)) = (quad_walk, f.aliased, &f.bounds_mask) {
            return m.iter().map(|&c| 255 - c).collect();
        }
        crate::pixels::copy_tess::fill_copy_in(&[&outer, &inner], true, w as usize, h as usize, true, pop_clip, |x, y, cov, _| {
            inv[y * w as usize + x] = cov.c8() as u8;
        });
        inv
    });
    for py in y0..y1 {
        for px in x0..x1 {
            let p = (py * w + px) as usize;
            if tri_neg_x && (px as f32 + 0.5) < strip_diag_left_x(r, b, py as f32 + 0.5) {
                continue;
            }
            // Content-bounds coverage: the transformed-quad mask when the device transform was not
            // axis-aligned, else the WARP geometry ring coverage.
            let bc = match &f.bounds_mask {
                _ if hard.is_some() => 255,
                _ if far.is_some() => far.unwrap()[p] as i32,
                // A finite-huge BOTTOM layer falls onto native's integer cvttss2si clip: a HARD pixel rect
                // (full coverage over the floor/ceil integer range this loop walks, no antialiased edge
                // ring, on both axes — fuzz r17 22432: top row and right col paint the full interior value).
                _ if clip_to_target && !exact => 255,
                Some(m) => m[p] as i32,
                // ALIASED layer (maskAntialiasMode == ALIASED): the content-bounds rect is a hard
                // clip — a pixel is fully in or fully out by its centre, no eighth-pixel ring
                // (fuzz r16 21972/21592: a top edge mid-pixel keeps the whole row at full coverage).
                None if f.aliased => {
                    let [l, t, r, b] = f.bounds;
                    let (cx, cy) = (px as f32 + 0.5, py as f32 + 0.5);
                    if l <= cx && cx < r && t <= cy && cy < b { 255 } else { 0 }
                }
                None => layer_bounds_cova(f.bounds, px, py),
            };
            // COPY composites the bounds region as CHwRasterizer's 1/16 edge coverage (its "keep"),
            // not the SOURCE_OVER eighth-pixel ring; the axis-aligned case has a dedicated coverage.
            let is_copy = f.blend == crate::pixels::Blend::Copy || quad_walk.is_some();
            let copy_bc = inv.as_ref().map_or(bc, |inv| 255 - inv[p] as i32);
            if (if is_copy { copy_bc } else { bc }) == 0 {
                continue;
            }
            let i = py as usize * st + px as usize * 4;
            if f.keep_pop {
                let c = f.mask.as_ref().map_or(bc, |m| blend_mul(bc, m[p] as i32));
                // An IGNORE_ALPHA layer's pop skips only its fade: the content still goes over the backdrop, where it
                // has alpha of its own (r27 32862: a SOURCE_IN draw in the layer shows the backdrop through). An
                // INITIALIZE_FROM_BACKGROUND layer's content already holds the backdrop (r27 32771: left as it was).
                let under = if f.ignore_alpha && !f.init { 255 - blend_mul(working.data[i + 3] as i32, c) } else { 255 - c };
                for k in 0..4 {
                    saved.data[i + k] = (blend_mul(working.data[i + k] as i32, c) + blend_mul(saved.data[i + k] as i32, under)) as u8;
                }
                continue;
            }
            // An IGNORE_ALPHA layer's pop runs its three steps through WARP's float blend with each result
            // stored to UNORM8 (round half to even) between them: the fade by the raw opacity with the
            // destination alpha taken as opaque, the erase outside the bounds by the inverse coverage as an
            // 8-bit alpha, then the saved backdrop under it (fuzz r16..r22 INIT|IGNORE_ALPHA ±1 family).
            // Under ADD the last step adds the backdrop instead (r20 25602).
            let add = f.blend == crate::pixels::Blend::Add;
            if f.ignore_alpha && (f.blend == crate::pixels::Blend::SourceOver || add) {
                let cov = 1.0 - (255 - bc) as f32 / 255.0;
                let cov = cov * f.mask.as_ref().map_or(1.0, |m| m[p] as f32 / 255.0);
                // The float blend fades by the brush alpha × opacity unrounded (r17 22261, r21 26531).
                let a = match (&f.brush_alpha_f, &f.brush_alpha) {
                    (Some(bf), _) => bf[p],
                    (_, Some(b)) => b[p] as f32 / 255.0,
                    _ => f.opacity,
                };
                let q = |v: f32| (v * 255.0).round_ties_even().clamp(0.0, 255.0) / 255.0;
                let x: [f32; 4] = core::array::from_fn(|k| q(if k == 3 { 1.0 } else { working.data[i + k] as f32 / 255.0 } * a));
                let y: [f32; 4] = core::array::from_fn(|k| q(x[k] * cov));
                for k in 0..4 {
                    let d = saved.data[i + k] as f32 / 255.0;
                    saved.data[i + k] = (q(y[k] + if add { d } else { d * (1.0 - y[3]) }) * 255.0) as u8;
                }
                continue;
            }
            // An opacity-brush pop runs the same float blend with the brush alpha as a byte: fade by it,
            // erase outside the bounds (and mask), then the saved backdrop under it (r18 23736).
            if let (Some(ba), crate::pixels::Blend::SourceOver) = (&f.brush_alpha, f.blend) {
                let cov = bc as f32 / 255.0 * f.mask.as_ref().map_or(1.0, |m| m[p] as f32 / 255.0);
                let a = ba[p] as f32 / 255.0;
                let q = |v: f32| (v * 255.0).round_ties_even().clamp(0.0, 255.0) / 255.0;
                let x: [f32; 4] = core::array::from_fn(|k| q(working.data[i + k] as f32 / 255.0 * a));
                let y: [f32; 4] = core::array::from_fn(|k| q(x[k] * cov));
                for k in 0..4 {
                    let d = saved.data[i + k] as f32 / 255.0;
                    saved.data[i + k] = (q(y[k] + d * (1.0 - y[3])) * 255.0) as u8;
                }
                continue;
            }
            // Modulate the working (premultiplied) pixel by each factor as a *separate* UNORM8
            // multiply — opacity, content-bounds coverage, the opacity-brush alpha, then the geometric
            // mask coverage. Native applies these as distinct texture/constant multiplies, each
            // rounding to UNORM8; the order (opacity mask before geometric mask) and the separate
            // rounds both matter by a LSB (`layers-11` opacity, `15-layer-mask` all three combined).
            let mut src: [i32; 4] = core::array::from_fn(|k| working.data[i + k] as i32);
            if f.ignore_alpha {
                // Under any other blend an IGNORE_ALPHA layer still fades in float by the brush alpha × opacity
                // unrounded, stored round-half-even (r27 32574: COPY).
                let a = match (&f.brush_alpha_f, &f.brush_alpha) {
                    (Some(bf), _) => bf[p],
                    (_, Some(b)) => b[p] as f32 / 255.0,
                    _ => f.opacity,
                };
                src = core::array::from_fn(|k| {
                    let d = if k == 3 { 1.0 } else { src[k] as f32 / 255.0 };
                    (d * a * 255.0).round_ties_even().clamp(0.0, 255.0) as i32
                });
            }
            // COPY restores the backdrop through the inverse of the layer's clip geometry (the bounds
            // clipped by any geometric mask), so its bounds and mask coverage are the lerp weight below,
            // not source multiplies.
            let mask = f.mask.as_ref().map(|m| m[p] as i32);
            let faded = f.ignore_alpha;
            let opt = [f.brush_alpha.as_ref().map(|b| b[p] as i32).filter(|_| !faded), if is_copy { None } else { mask }];
            let bounds = if is_copy { None } else { Some(bc) };
            let copy_bc = match mask {
                Some(m) if is_copy => blend_mul(copy_bc, m),
                _ => copy_bc,
            };
            // The pop fades by the opacity brush (DEST_IN) before it erases outside the bounds, so the
            // brush multiplies ahead of the bounds coverage (r21 26784's edge row and column).
            for m in [Some(f.opacity8).filter(|_| !faded), opt[0], bounds, opt[1]].into_iter().flatten() {
                src = src.map(|c| blend_mul(c, m));
            }
            // PopLayer composites the (modulated, premultiplied) layer onto the backdrop under the
            // primitive blend active at PUSH. COPY lerps layer and backdrop by the content-bounds
            // coverage c in the kernel's fused form `(s·c + (s·c>>8) + d·(255−c) + (d·(255−c)>>8) +
            // 128) >> 8` (VM-pinned lc_copyedge / the COPY edge probes: at c=0.5 the backdrop halves,
            // at c=1 fully replaced, at c=0 kept; q9499-b1-*-l1: the layer is weighted by c, not by the
            // eighths ring) — NOT source-over's `1−srcAlpha`. MAX takes the per-channel max of the
            // coverage-modulated premultiplied layer src and the backdrop (fuzz r16 21967). MIN is the
            // Porter-Duff MIN primitive blend `min(dst, src + (255 − src_a))` per premultiplied channel
            // (the same formula native/paint.rs use for a MIN-blend DRAW): a transparent src → src+255 ≥
            // dst → keeps the backdrop (r16 21339 opacity 0.04 ≈ backdrop), an opaque src → min(dst,src).
            // ADD stays source-over (unpinned).
            let sa = src[3];
            if is_copy {
                // COPY lerps the backdrop out by the bounds coverage (its own VM-pinned keep).
                for (s, d) in src.iter().zip(&mut saved.data[i..i + 4]) {
                    let (ps, pd) = (s * copy_bc, *d as i32 * (255 - copy_bc));
                    *d = ((ps + (ps >> 8) + pd + (pd >> 8) + 128) >> 8).min(255) as u8;
                }
            } else if f.blend == crate::pixels::Blend::Max {
                for k in 0..4 { saved.data[i + k] = src[k].max(saved.data[i + k] as i32) as u8; }
            } else if f.blend == crate::pixels::Blend::Min {
                for k in 0..4 { saved.data[i + k] = (saved.data[i + k] as i32).min(src[k] + 255 - sa) as u8; }
            } else if f.blend == crate::pixels::Blend::Add {
                // ADD / PLUS primitive blend: out = min(src + dst, 255) per premultiplied channel — the
                // coverage-modulated layer is ADDED to the backdrop, NOT source-over's src + dst·(1−srcA)
                // (which under-adds by dst·srcA). Fuzz 26274/26692/26903: a push-blend ADD layer over a
                // semi-transparent backdrop.
                for k in 0..4 { saved.data[i + k] = (src[k] + saved.data[i + k] as i32).min(255) as u8; }
            } else {
                // SOURCE_OVER stores the modulated layer plus the backdrop's `dst·(255 − src_alpha)/255`
                // rounded to nearest, ties down (65536 VM probe pixels over eight opacities, both layer
                // options; WARP's blend-unit store misses ~0.2%: r8 9180).
                let ia = 255 - sa;
                for k in 0..4 {
                    let dv = saved.data[i + k] as i32;
                    saved.data[i + k] = (src[k] + (dv * ia + 127) / 255).min(255) as u8;
                }
            }
        }
    }
}

/// `DrawImage` composite of a premultiplied source over `dst` (the full
/// `D2D1_COMPOSITE_MODE` table). Modes 0..=11 are the fixed-function Porter-Duff blends
/// `out = Fs·src + Fd·dst` per premultiplied channel (Direct2D's composite blend description,
/// D3D `BlendOperation.Add`): WARP's blend unit scales each term (exactly for ONE, with the high-byte
/// correction for an alpha factor) and rounds the sum once to UNORM8 (`layers-10` and the all-mode
/// composite probes). Every `Fs`/`Fd` is one of `{0,1,αs,αd,1−αs,
/// 1−αd}`. BOUNDED_SOURCE_COPY (11) shares COPY's formula (its image-rect bounding is the caller's;
/// with no source rect native copies over the whole target — `DoDrawImageFastPath` SOURCE_COPY).
/// MASK_INVERT (12) is not a linear blend: it is native's `MaskInvertShader`,
/// `out_rgb = (1−αs)·d_rgb + (1−d_rgb)·s_rgb`, `out_a = αd`, in premultiplied floats.
pub(crate) fn composite(src: [i32; 4], dst: [u8; 4], mode: u32) -> Option<[u8; 4]> {
    let d = dst.map(|v| v as i32);
    let (sa, da) = (src[3], d[3]);
    if mode == 12 {
        let (q, sa_f) = (|v: i32| v as f32 / 255.0, sa as f32 / 255.0);
        let mut out = [0u8; 4];
        for c in 0..3 {
            let (dc, sc) = (q(d[c]), q(src[c]));
            out[c] = (((1.0 - sa_f) * dc + (1.0 - dc) * sc) * 255.0).round_ties_even().clamp(0.0, 255.0) as u8;
        }
        out[3] = da as u8;
        return Some(out);
    }
    // WARP's blend unit: a ONE factor passes its operand exactly (`x·256`), an alpha factor `f`
    // multiplies with the high-byte correction (`p + (p >> 8)`, `p = x·f`), ZERO drops it; the sum
    // rounds once, `(Σ + 128) >> 8` (all-mode composite probes, nearest and LINEAR: 16512 values exact).
    // With a ONE source term this is the kernel's fused source-over.
    enum F { One, Zero, A(i32) }
    let (fs, fd) = match mode {
        0 => (F::One, F::A(255 - sa)),         // SOURCE_OVER
        1 => (F::A(255 - da), F::One),         // DESTINATION_OVER
        2 => (F::A(da), F::Zero),              // SOURCE_IN
        3 => (F::Zero, F::A(sa)),              // DESTINATION_IN
        4 => (F::A(255 - da), F::Zero),        // SOURCE_OUT
        5 => (F::Zero, F::A(255 - sa)),        // DESTINATION_OUT
        6 => (F::A(da), F::A(255 - sa)),       // SOURCE_ATOP
        7 => (F::A(255 - da), F::A(sa)),       // DESTINATION_ATOP
        8 => (F::A(255 - da), F::A(255 - sa)), // XOR
        9 => (F::One, F::One),                 // PLUS (saturating)
        10 | 11 => (F::One, F::Zero),          // SOURCE_COPY / BOUNDED_SOURCE_COPY
        _ => return None,
    };
    let term = |x: i32, f: &F| match *f { F::One => x * 256, F::Zero => 0, F::A(f) => { let p = x * f; p + (p >> 8) } };
    Some(core::array::from_fn(|i| ((term(src[i], &fs) + term(d[i], &fd) + 128) >> 8).clamp(0, 255) as u8))
}

// WARP mask JIT: power-of-two addresses use floor; general-size addresses use cvtps2dq (mask48/maskcheck captures).
fn mask_axis(d: f32, c: f32, dim: u32, pos: f32, power_of_two: bool) -> (i32, i32) {
    if power_of_two {
        crate::sampling::plane_quant(d, c, dim as f32, pos)
    } else {
        let q = (((d * pos + c).clamp(0.0, 1.0) * (dim as f32 * 256.0)).round_ties_even() as i32) - 128;
        (q >> 8, q & 255)
    }
}

/// `FillOpacityMask`: aliased
/// fill of the device rect `dest` with the premultiplied solid brush modulated by the opacity mask's
/// per-pixel alpha, source-over the backdrop. Byte-for-byte reproduces the WARP software
/// rasterizer's opacity-mask fill, verified against the captured readback:
/// the mask texcoord is the `uv` plane ([`fom_uv_plane`]) quantized at the pixel centre
/// ([`mask_axis`]), the alpha is a `value·128`
/// bilinear kept at sub-byte precision (`sampling::bilinear_v128`→`norm_v128`), and the whole modulate
/// runs in **f32** — each channel `round_ties_even(clamp(m·brush, 0, 1)·255)` (WARP multiplies by 255
/// with `mulps` before the `cvtps2dq` round, so an exact `.5` falls to even) — then source-over in the
/// kernel's exact integer order `((src<<8) + t + (t>>8) + 128) >> 8`, `t = dst·(255−srcα)`. `brush_premul` is the
/// premultiplied brush in **RGBA** f32 (full precision; WARP never quantises it to a byte first).
/// Aliased: mask alpha is the only coverage and `dest` edges snap round-half-down (no geometry AA).
/// The composite store honours the primitive blend active at the call ([`mask_blend_store`]).

/// FillOpacityMask's composite store under the primitive blend active at the call, per premultiplied
/// channel of the `(mask × brush)` source `src` over backdrop `dst`. SOURCE_OVER is WARP's over kernel
/// `((src<<8) + t + (t>>8) + 128) >> 8` (t = dst·(255−srcα)); COPY replaces; MIN/ADD/MAX are the
/// Porter-Duff primitive blends — the fom honours SetPrimitiveBlend like a draw (fuzz r16 21102: a fom
/// under ADD stores `src + dst`, byte-exact where source-over was ~8 % low over a semi-transparent
/// backdrop).
fn mask_blend_store(dst: &mut [u8], src: [i32; 4], blend: crate::pixels::Blend) {
    use crate::pixels::Blend::*;
    let sa = src[3];
    for c in 0..4 {
        let d = dst[c] as i32;
        dst[c] = match blend {
            Copy => src[c],
            Add => (src[c] + d).min(255),
            Min => d.min(src[c] + 255 - sa),
            Max => d.max(src[c]),
            SourceOver => {
                let t = d * (255 - sa);
                ((src[c] << 8) + t + (t >> 8) + 128) >> 8
            }
        }
        .clamp(0, 255) as u8;
    }
}

/// FillOpacityMask's mask texcoord plane, device → normalized mask uv, as d2d1 builds it: the
/// `source` → `dest` rect mapping (both DIPs; source sorted and clipped to the mask's DIP `size`, dest
/// sorted) divides in f64 and stores f32, composes with the f32 `world` (`Matrix3x2F::SetProduct`), is
/// scaled by the mask's DIP size and inverted with a reciprocal-determinant multiply (the bitmap
/// brush's texture transform). WARP then fits it over the unit triangle. `None` = a singular mapping,
/// which native drops.
pub(crate) fn fom_uv_plane(source: [f32; 4], dest: [f32; 4], world: [f32; 6], size: [f32; 2]) -> Option<[f32; 6]> {
    let (l, t) = (source[0].min(source[2]).max(0.0) as f64, source[1].min(source[3]).max(0.0) as f64);
    let (r, b) = (source[0].max(source[2]).min(size[0]) as f64, source[1].max(source[3]).min(size[1]) as f64);
    let [dl, dt] = [dest[0].min(dest[2]) as f64, dest[1].min(dest[3]) as f64];
    let [dr, db] = [dest[0].max(dest[2]) as f64, dest[1].max(dest[3]) as f64];
    let (sx, sy) = ((dr - dl) / (r - l), (db - dt) / (b - t));
    let m = [sx as f32, 0.0, 0.0, sy as f32, (dl - l * sx) as f32, (dt - t * sy) as f32];
    let product = |a: &[f32; 6], b: &[f32; 6]| {
        [a[0] * b[0] + b[2] * a[1], b[3] * a[1] + b[1] * a[0], a[2] * b[0] + b[2] * a[3], a[2] * b[1] + b[3] * a[3],
            (a[4] * b[0] + b[2] * a[5]) + b[4], (a[4] * b[1] + b[3] * a[5]) + b[5]]
    };
    let c = product(&[size[0], 0.0, 0.0, size[1], 0.0, 0.0], &product(&m, &world));
    let det = c[3] * c[0] - c[2] * c[1];
    if det == 0.0 || !(1.0 / det).is_finite() {
        return None;
    }
    Some(crate::warp::plane::warp_fit_affine(&crate::brush::invert(&c)))
}

pub(crate) fn opacity_mask_fill(
    target: &mut PixelBuffer,
    mask: &Bitmap,
    brush_premul: [f32; 4],
    dest: [f32; 4],
    uv: [f32; 6],
    // FillOpacityMask runs under the current primitive blend ([`mask_blend_store`]): COPY replaces the
    // dest rect with mask×brush (transparent where the mask is 0), ADD/MIN/MAX apply their Porter-Duff
    // blend, else source-over (fuzz 5112 COPY writes mask·brush = [7,5,8,22]; r16 21102 ADD = src+dst).
    blend: crate::pixels::Blend,
) {
    let (w, h, st) = (target.desc.width as i32, target.desc.height as i32, target.stride);
    let [dl, dt, dr, db] = dest;
    if !(dr - dl > 0.0 && db - dt > 0.0) {
        return;
    }
    // A huge/non-finite dest bound (device |x·256| overflow) is a degenerate quad native draws nothing
    // for (fuzz 7218: dest top -3.4e38); the texcoord plane would otherwise be huge
    // and paint garbage. Same degenerate-bounds rule as layer/clip.
    if [dl, dt, dr, db].iter().any(|&v| !(v * 256.0).is_finite()) {
        return;
    }
    let tex = Texture { data: &mask.premul_bgra, width: mask.width, height: mask.height };
    // WARP's float-family address takes the power-of-two path per axis, but v only when both
    // dimensions are: a 2×6 mask's u floors while its v rounds, a 12×8's v rounds (msk-* probes,
    // r9 14517/14264).
    let pow2 = [mask.width.is_power_of_two(), mask.width.is_power_of_two() && mask.height.is_power_of_two()];
    // Brush premultiplied colour in the target's BGRA channel order (f32, kept full precision).
    let [br, bg, bb, ba] = brush_premul;
    let brush = [bb, bg, br, ba];
    let [udx, _, _, vdy, uc, vc] = uv;
    let (x0, x1) = (snap(dl).clamp(0, w), snap(dr).clamp(0, w));
    let (y0, y1) = (snap(dt).clamp(0, h), snap(db).clamp(0, h));
    for py in y0..y1 {
        let yq = mask_axis(vdy, vc, mask.height, py as f32 + 0.5, pow2[1]);
        for px in x0..x1 {
            let xq = mask_axis(udx, uc, mask.width, px as f32 + 0.5, pow2[0]);
            let m = crate::sampling::norm_v128(tex.bilinear_v128([Extend::Clamp; 2], 3, xq, yq));
            let src: [i32; 4] =
                core::array::from_fn(|c| ((m * brush[c]).clamp(0.0, 1.0) * 255.0).round_ties_even() as i32);
            let i = py as usize * st + px as usize * 4;
            // MIN takes the masked source over white in float and rounds once (VM primitive-blend probes).
            if blend == crate::pixels::Blend::Min {
                let a = (m * brush[3]).clamp(0.0, 1.0);
                for c in 0..4 {
                    let v = (((m * brush[c]).clamp(0.0, 1.0) + 1.0 - a) * 255.0).round_ties_even().clamp(0.0, 255.0) as u8;
                    target.data[i + c] = target.data[i + c].min(v);
                }
                continue;
            }
            mask_blend_store(&mut target.data[i..i + 4], src, blend);
        }
    }
}

/// `FillOpacityMask` with a gradient/bitmap/image brush: identical to [`opacity_mask_fill`] but the
/// modulated colour is the brush's per-pixel premultiplied source (device pixel centre, WARP's
/// gradient PS 8-bit ramp) instead of a VS-constant solid. `mask × brushSource` runs in f32 with the
/// same single round-ties-even and the same integer source-over the `maskfill` kernel uses.
pub(crate) fn opacity_mask_fill_brush(
    target: &mut PixelBuffer,
    mask: &Bitmap,
    src_at: impl Fn(f32, f32) -> [f32; 4], // the brush's premultiplied BGRA (0..1) at a pixel centre
    dest: [f32; 4],
    uv: [f32; 6], // [`fom_uv_plane`]
    blend: crate::pixels::Blend, // the primitive blend active at the call (see mask_blend_store).
) {
    let (w, h, st) = (target.desc.width as i32, target.desc.height as i32, target.stride);
    let [dl, dt, dr, db] = dest;
    if !(dr - dl > 0.0 && db - dt > 0.0) {
        return;
    }
    // Degenerate huge/non-finite dest bound → native draws nothing (see opacity_mask_fill, fuzz 7218).
    if [dl, dt, dr, db].iter().any(|&v| !(v * 256.0).is_finite()) {
        return;
    }
    let tex = Texture { data: &mask.premul_bgra, width: mask.width, height: mask.height };
    // WARP's float-family address takes the power-of-two path per axis, but v only when both
    // dimensions are: a 2×6 mask's u floors while its v rounds, a 12×8's v rounds (msk-* probes,
    // r9 14517/14264).
    let pow2 = [mask.width.is_power_of_two(), mask.width.is_power_of_two() && mask.height.is_power_of_two()];
    let [udx, _, _, vdy, uc, vc] = uv;
    let (x0, x1) = (snap(dl).clamp(0, w), snap(dr).clamp(0, w));
    let (y0, y1) = (snap(dt).clamp(0, h), snap(db).clamp(0, h));
    for py in y0..y1 {
        let yq = mask_axis(vdy, vc, mask.height, py as f32 + 0.5, pow2[1]);
        for px in x0..x1 {
            let xq = mask_axis(udx, uc, mask.width, px as f32 + 0.5, pow2[0]);
            let m = crate::sampling::norm_v128(tex.bilinear_v128([Extend::Clamp; 2], 3, xq, yq));
            // Per-pixel premultiplied brush source in the target's BGRA order (0..1), before quantization.
            let s = src_at(px as f32 + 0.5, py as f32 + 0.5);
            let src: [i32; 4] =
                core::array::from_fn(|c| ((m * s[c]).clamp(0.0, 1.0) * 255.0).round_ties_even() as i32);
            let i = py as usize * st + px as usize * 4;
            mask_blend_store(&mut target.data[i..i + 4], src, blend);
        }
    }
}

/// `FillOpacityMask` under a **non-axis-aligned** device transform (rotation/skew). Native maps the
/// DIP `dest` rect through the full transform to a rotated/skewed quad, so a device pixel is covered
/// only when its centre maps back inside the DIP rect — not the diagonal-AABB `device_rect` returns
/// (fuzz 14312: a skew sends the 0.5px dest strip off-screen, yet our axis-aligned fill painted a
/// 7px column over the top-left where native draws nothing). The mask texel/coverage come from the
/// inverse-mapped `(u, v)`; `src_at` is the per-pixel premultiplied BGRA source at the device centre
/// (a constant for a solid brush, the realised brush sample otherwise). SOURCE_OVER / COPY and the
/// f32 modulate match [`opacity_mask_fill`]; the axis-aligned case keeps that byte-exact path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn opacity_mask_fill_xform(
    target: &mut PixelBuffer,
    mask: &Bitmap,
    dip: [f32; 4],
    m: [f64; 6],
    source: [f32; 4],
    uv: Option<[f32; 6]>,
    blend: crate::pixels::Blend,
    src_at: impl Fn(f32, f32) -> [f32; 4],
) {
    let [dl, dt, dr, db] = dip;
    if !(dr - dl > 0.0 && db - dt > 0.0) {
        return;
    }
    let det = m[0] * m[3] - m[1] * m[2];
    if det == 0.0 || !det.is_finite() {
        return;
    }
    let inv = 1.0 / det;
    // Inverse affine (device → DIP), f64 to match device_transform.
    let mi = [m[3] * inv, -m[1] * inv, -m[2] * inv, m[0] * inv, (m[2] * m[5] - m[3] * m[4]) * inv, (m[1] * m[4] - m[0] * m[5]) * inv];
    let fwd = |x: f32, y: f32| [(m[0] * x as f64 + m[2] * y as f64 + m[4]) as f32, (m[1] * x as f64 + m[3] * y as f64 + m[5]) as f32];
    let corners = [fwd(dl, dt), fwd(dr, dt), fwd(dr, db), fwd(dl, db)];
    if corners.iter().any(|p| !(p[0] * 256.0).is_finite() || !(p[1] * 256.0).is_finite()) {
        return; // degenerate huge/non-finite quad: native draws nothing (as opacity_mask_fill)
    }
    let (w, h, st) = (target.desc.width as i32, target.desc.height as i32, target.stride);
    let minx = corners.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
    let maxx = corners.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
    let miny = corners.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
    let maxy = corners.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);
    let (x0, x1) = ((snap(minx) - 1).clamp(0, w), (snap(maxx) + 1).clamp(0, w));
    let (y0, y1) = ((snap(miny) - 1).clamp(0, h), (snap(maxy) + 1).clamp(0, h));
    // The quad fills as aliased geometry: its corners go to 28.4 on the pixel-centre grid (⌈16·(v − ½)⌉),
    // so a pixel centre a hair past the true edge can be inside the snapped one (r16 21852).
    let mut inside = vec![false; (w * h) as usize];
    let quad = vec![corners.map(|p| p.map(|v| ((v - 0.5) * 16.0).ceil() as i32)).to_vec()];
    crate::pixels::warp_geom::fill_aliased(&quad, true, w as usize, h as usize, |y, a, b| inside[y * w as usize + a..y * w as usize + b].fill(true));
    let tex = Texture { data: &mask.premul_bgra, width: mask.width, height: mask.height };
    let power_of_two = mask.width.is_power_of_two() && mask.height.is_power_of_two();
    let [sl, stop, sr, sb] = source;
    let (tw, th) = (mask.width as f32, mask.height as f32);
    for py in y0..y1 {
        for px in x0..x1 {
            let (cx, cy) = (px as f32 + 0.5, py as f32 + 0.5);
            if !inside[(py * w + px) as usize] {
                continue;
            }
            // The mask kernel evaluates the same fitted texcoord plane as the axis-aligned fill ([`fom_uv_plane`]: the
            // source clamped to the mask), with the same per-axis power-of-two rule (VM span capture of r27 32826's
            // rotated FillOpacityMask: its planes are this one's, a source past the mask's bottom clamped to it).
            let (xq, yq) = match uv {
                Some(p) => (
                    mask_axis(0.0, p[0] * cx + (p[2] * cy + p[4]), mask.width, 0.0, mask.width.is_power_of_two()),
                    mask_axis(0.0, p[1] * cx + (p[3] * cy + p[5]), mask.height, 0.0, power_of_two),
                ),
                None => {
                    let ux = (mi[0] * cx as f64 + mi[2] * cy as f64 + mi[4]) as f32;
                    let uy = (mi[1] * cx as f64 + mi[3] * cy as f64 + mi[5]) as f32;
                    let (u, v) = ((ux - dl) / (dr - dl), (uy - dt) / (db - dt));
                    (mask_axis(0.0, (sl + u * (sr - sl)) / tw, mask.width, 0.0, power_of_two), mask_axis(0.0, (stop + v * (sb - stop)) / th, mask.height, 0.0, power_of_two))
                }
            };
            let mcov = crate::sampling::norm_v128(tex.bilinear_v128([Extend::Clamp; 2], 3, xq, yq));
            let s = src_at(cx, cy);
            let src: [i32; 4] = core::array::from_fn(|c| ((mcov * s[c]).clamp(0.0, 1.0) * 255.0).round_ties_even() as i32);
            let i = py as usize * st + px as usize * 4;
            mask_blend_store(&mut target.data[i..i + 4], src, blend);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Clip edges snap round-half-down (`layers-01`); a layer's half-covered column is 127 (layers-06),
    // an edge at 8·x = 542.502 lands on the next eighth (fxclip-08), and an edge at k + 1/16 rounds
    // down to k (layers-14: x = 4.0625 covers pixel 4 fully, 30.1875 leaves it 7/8).
    // The layer rect: an infinite bound under an axis-keeping transform poisons the other axis with
    // `0·inf`, which NaN-propagates through the bounds and survives the round-out; finite bounds clamp
    // to the target and round out; an infinite edge is never on the pixel grid.
    #[test]
    fn layer_rect_bounds() {
        let id = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let d = rect_fill_bounds([f32::INFINITY, 5.5, 14.5, 39.2], &id);
        assert!(d[0] == 14.5 && d[2] == f32::INFINITY && d[1].is_nan() && d[3].is_nan());
        assert!(layer_rect(d, 16.0, 64.0, [0.0, 0.0, 16.0, 64.0]).is_some_and(|r| r[1].is_nan()));
        let d = rect_fill_bounds([-2.5, 3.25, 100.0, 7.0], &[2.0, 0.0, 0.0, 2.0, 1.0, 0.0]);
        assert_eq!(d, [-4.0, 6.5, 201.0, 14.0]);
        assert_eq!(layer_rect(d, 64.0, 48.0, [0.0, 0.0, 64.0, 48.0]), Some([0.0, 6.0, 64.0, 14.0]));
        assert_eq!(layer_rect([10.0, 0.0, 5.0, 4.0], 64.0, 48.0, [0.0, 0.0, 64.0, 48.0]), None);
        assert!(pixel_aligned([0.0, 6.0, 64.0, 14.0]) && !pixel_aligned([0.0, 6.5, 64.0, 14.0]));
        assert!(!pixel_aligned([0.0, 0.0, f32::INFINITY, 4.0]));
    }

    #[test]
    fn coverage_rounds_half_down() {
        assert_eq!([snap(16.5), snap(24.25), snap(100.75), snap(72.5)], [16, 24, 101, 72]);
        assert_eq!(layer_bounds_cova([0.5, 0.0, 10.0, 10.0], 0, 5), 127);
        assert_eq!(layer_bounds_cova([0.0, 0.0, 10.0, 10.0], 3, 3), 255);
        assert_eq!(layer_bounds_cova([67.81274, 0.0, 90.0, 10.0], 67, 5), 32);
        assert_eq!(layer_bounds_cova([4.0625, 0.0, 20.0625, 10.0], 4, 5), 255);
        assert_eq!(layer_bounds_cova([4.0625, 0.0, 20.0625, 10.0], 20, 5), 0);
        assert_eq!(layer_bounds_cova([30.1875, 0.0, 50.0, 10.0], 30, 5), 223);
    }

    // The §2.7 composite table, against the hand-verified flood source (0,128,255,153) over an
    // opaque teal backdrop (`layers-10`).
    #[test]
    fn composite_modes() {
        let (src, teal) = ([0, 128, 255, 153], [153, 140, 26, 255]);
        assert_eq!(composite(src, teal, 0), Some([61, 184, 255, 255])); // SOURCE_OVER
        assert_eq!(composite(src, teal, 1), Some([153, 140, 26, 255])); // DESTINATION_OVER (opaque dst)
        assert_eq!(composite(src, teal, 9), Some([153, 255, 255, 255])); // PLUS (saturating)
        assert_eq!(composite(src, teal, 10), Some([0, 128, 255, 153])); // SOURCE_COPY
        assert_eq!(composite(src, teal, 11), Some([0, 128, 255, 153])); // BOUNDED_SOURCE_COPY (same formula)
        assert_eq!(composite(src, teal, 12), Some([61, 114, 239, 255])); // MASK_INVERT (shader)
        // XOR over a translucent dst exercises both factors (1−αd and 1−αs), summed then rounded once.
        assert_eq!(composite([40, 20, 10, 80], [60, 30, 15, 120], 8), Some([62, 31, 16, 125]));
        assert_eq!(composite(src, teal, 13), None); // out of range
    }
}
