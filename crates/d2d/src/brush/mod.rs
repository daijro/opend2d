//! Phase 1D brushes: solid colour brushes and gradient (linear/radial) brushes, reproducing
//! Direct2D.s gradient texture realization and the WARP sampler byte-for-byte.
//!
//! The gradient texture is built exactly as native's gradient-texture generator
//! (Direct2D): stops are stably sorted and clamped to a
//! well-formed `[0,1]` ramp with flat extension; `CalcTextureSize` picks a power-of-two,
//! optionally bordered width from the device footprint; a box-filter pass averages the
//! piecewise-linear ramp over each interior texel (centre value when a texel lies in one stop
//! pair, area average when it straddles), and `Core_PostProcess` premultiplies, applies the
//! colour-space/gamma conversion and quantises round-half-up to premultiplied BGRA8.
//!
//! Each brush produces a premultiplied source colour per device pixel ([`Brush::source_at`],
//! BGRA float 0..255, already multiplied by brush opacity). Coverage weighting and source-over
//! compositing belong to the pixel layer (`crate::pixels`); the brush tests drive the gradient
//! scenes (pixel-aligned rectangle fills, full coverage) directly.

use crate::context::{DeviceContext, PixelBrush};
use crate::geometry::Geometry;
use crate::pixels::{rasterize_rect, Aa, Blend, FillMode, Target};
use crate::sampling::{Extend, Filter, Texture};
use d2d_types::contracts::PixelBuffer;
use std::sync::OnceLock;

// 3×2 affine `[m11,m12,m21,m22,dx,dy]`; `apply(m,p)` maps a point through it.
type Mat = [f32; 6];
fn apply(m: &Mat, x: f32, y: f32) -> [f32; 2] {
    [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]
}
/// The 3×2 matrix inverse: one reciprocal determinant.
pub(crate) fn invert(m: &Mat) -> Mat {
    let det = m[0] * m[3] - m[1] * m[2];
    let i = 1.0 / det;
    [m[3] * i, -m[1] * i, -m[2] * i, m[0] * i, (m[2] * m[5] - m[3] * m[4]) * i, (m[1] * m[4] - m[0] * m[5]) * i]
}

fn bitmap_footprint(quad: Option<[[f32; 3]; 2]>, m: &[f32; 6], dims: [u32; 2], x: f32, y: f32) -> ([f32; 2], f32, f32) {
    let (dx, dy) = if let Some(p) = quad {
        let (x, y) = ((x * 0.5).floor() * 2.0 + 0.5, (y * 0.5).floor() * 2.0 + 0.5);
        let eval = |x, y| p.map(|a| a[0] * x + (a[1] * y + a[2]));
        let (v, vx, vy) = (eval(x, y), eval(x + 1.0, y), eval(x, y + 1.0));
        (core::array::from_fn::<_, 2, _>(|i| (vx[i] - v[i]) * dims[i] as f32), core::array::from_fn::<_, 2, _>(|i| (vy[i] - v[i]) * dims[i] as f32))
    } else { ([m[0], m[1]], [m[2], m[3]]) };
    let (dx2, dy2) = (dx[0] * dx[0] + dx[1] * dx[1], dy[0] * dy[0] + dy[1] * dy[1]);
    let major2 = dx2.max(dy2);
    let major = crate::warpmath::sqrt(major2);
    let area = (dx[0] * dy[1] - dy[0] * dx[1]).abs();
    let minor = (area / major).max(major / 16.0);
    let ratio = if minor < 1.0 { major.max(1.0) } else { (major2 / area).min(16.0) };
    (if dx2 >= dy2 { dx } else { dy }.map(|v| v * (1.0 / major)), ratio, minor.max(1.0) * minor.max(1.0))
}

// Native d2d1 10.0.26100 brush VS: normalized coordinates retain the device-coordinate cancellation.
pub(crate) fn bitmap_quad(m: &[f32; 6], size: [u32; 2], rect: [f32; 4], x: f32, y: f32) -> [[f32; 3]; 2] {
    bitmap_quad_at(m, size, [0; 2], rect, x, y, [0.0; 2])
}

/// WARP's reciprocal: `1/scale` as the float-family MULTI_SAMPLE_LINEAR kernel forms it — the SSE
/// RCPPS 12-bit seed refined by one Newton–Raphson step (`r·(2 − a·r)`). It UNDERSHOOTS the
/// correctly-rounded `1.0/a` by a ULP (`1/f32(1e-6)` is 999999.9375, not 1000000.0), which the
/// extreme-minification MSL bitmap brush's per-pixel plane needs bit-for-bit.
///
/// The seed is reproduced in software — NOT a hardware intrinsic, so the browser build matches the
/// native builds bit-for-bit — and is exact to the reference machine's RCPPS over every normal
/// input. RCPPS's 12-bit mantissa approximation is a pure lookup (no interpolation); its seed is
/// CPU-vendor-specific, and the reference is AMD Zen 3. [`RCP_SEED_CORR`] encodes that table as a
/// 2-bit-per-entry correction to a midpoint-reciprocal base, verified exhaustively against the
/// hardware across all 2^23 mantissas at every normal exponent.
#[inline]
pub(crate) fn warp_rcp(a: f32) -> f32 {
    let r = rcpps_seed(a);
    r * (2.0 - a * r)
}

/// The RCPPS 12-bit seed (see [`warp_rcp`]): base `round(4096·(8191−2·i)/(8193+2·i))` for the
/// 12-bit mantissa index `i`, plus the correction from [`RCP_SEED_CORR`], with the exponent negated.
/// Non-normal inputs (denormal/zero/±inf/NaN, or a reciprocal that would denormalise) return the
/// correctly-rounded reciprocal — deterministic on every target, and a brush scale never reaches it.
#[inline]
fn rcpps_seed(a: f32) -> f32 {
    let b = a.to_bits();
    let exp = (b >> 23) & 0xff;
    if exp == 0 || exp > 252 {
        return 1.0 / a;
    }
    let i = ((b & 0x7f_ffff) >> 11) as i64;
    let base = (4096 * (8191 - 2 * i) + (8193 + 2 * i) / 2) / (8193 + 2 * i);
    let corr = ((RCP_SEED_CORR[(i >> 2) as usize] >> ((i & 3) * 2)) & 3) as i64 - 1;
    f32::from_bits((b & 0x8000_0000) | ((253 - exp) << 23) | (((base + corr) as u32) << 11))
}

/// AMD Zen 3 RCPPS seed correction (2 bits/entry, value+1, for the 4096 mantissa indices); see
/// [`rcpps_seed`]. Dumped exhaustively from the reference CPU.
#[rustfmt::skip]
const RCP_SEED_CORR: [u8; 1024] = [
    80,85,165,170,80,85,165,170,80,85,165,170,80,85,80,85,165,170,165,170,165,90,165,170,81,85,165,85,165,86,165,90,
    165,90,165,90,165,86,165,85,101,85,85,85,165,90,165,85,170,106,165,86,170,106,165,85,166,90,101,85,165,85,81,1,
    4,69,85,89,20,85,101,105,20,69,85,89,89,154,166,170,84,149,101,89,89,150,166,105,105,154,166,105,105,154,166,105,
    89,150,101,89,85,85,85,21,89,150,101,85,105,154,101,89,105,154,101,89,105,150,85,21,89,86,81,4,106,154,101,21,
    68,85,102,166,68,101,102,170,68,101,102,166,68,85,101,102,85,102,170,170,68,85,85,102,85,102,102,102,85,102,102,102,
    85,102,102,102,85,102,85,85,153,170,170,102,85,102,86,85,153,102,102,102,153,170,102,102,154,170,102,102,153,106,102,86,
    20,85,150,105,101,153,166,170,20,85,149,101,85,89,150,105,101,89,166,105,101,89,150,105,85,85,150,101,20,85,85,85,
    101,89,150,101,85,85,85,85,85,85,85,85,101,89,150,85,101,89,150,85,101,89,85,85,85,85,81,20,105,154,150,85,
    80,81,85,89,84,86,86,90,149,150,150,154,149,150,150,154,149,150,150,154,149,150,86,90,85,86,85,89,169,166,166,154,
    149,86,86,89,165,166,150,90,85,85,81,69,85,86,85,69,85,86,85,69,85,86,81,69,85,85,65,5,169,150,86,89,
    85,85,165,85,106,149,170,169,21,85,149,85,101,85,169,85,105,85,170,85,106,85,170,85,106,85,169,85,105,85,165,85,
    85,85,85,84,106,149,170,85,105,85,85,85,106,85,170,85,21,85,85,0,106,85,85,80,106,85,85,84,106,85,85,84,
    85,85,85,149,85,85,85,85,149,170,170,170,85,85,85,170,85,85,85,85,85,85,85,149,170,170,170,170,85,85,85,85,
    85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,170,170,170,170,106,85,85,85,85,85,85,85,
    85,101,85,85,85,165,170,170,1,80,85,85,85,85,85,85,85,165,86,85,85,165,90,85,85,165,90,85,85,165,86,85,
    85,85,85,85,21,80,85,85,85,165,170,85,85,101,85,85,90,165,170,86,85,101,85,85,86,165,90,85,85,80,85,1,
    149,90,165,106,149,106,165,170,64,85,80,85,64,85,80,85,149,106,165,106,149,90,165,90,149,86,165,85,85,85,85,85,
    150,170,165,90,149,86,101,85,69,85,80,5,149,86,101,85,150,106,165,86,85,85,81,21,149,85,85,85,149,90,101,85,
    81,69,85,85,81,69,85,85,81,69,85,85,81,69,21,85,165,150,90,106,165,149,86,90,101,149,85,86,81,69,21,85,
    165,149,86,90,85,85,21,85,165,150,86,86,85,69,21,21,165,149,85,85,166,150,86,90,85,69,5,5,101,85,21,21,
    85,101,101,101,85,165,165,165,149,165,165,101,85,101,101,101,85,101,101,101,69,85,85,85,69,81,81,81,150,165,165,165,
    85,101,101,85,69,85,81,81,149,165,101,101,69,85,81,81,149,101,101,85,69,81,81,80,85,101,85,81,150,165,101,85,
    85,85,149,101,85,89,150,165,101,89,150,166,101,89,154,166,101,89,154,166,101,89,154,166,101,89,150,166,101,89,150,101,
    85,85,149,101,85,85,85,85,81,20,69,81,101,89,150,101,85,85,85,85,166,105,154,166,101,85,85,85,81,20,69,81,
    4,81,20,85,85,101,85,101,89,166,89,166,20,85,20,85,85,85,85,101,85,101,89,101,89,102,89,166,89,166,105,166,
    20,81,20,81,20,81,20,85,20,81,20,81,20,81,20,81,105,166,89,166,89,102,89,102,89,101,89,101,85,85,85,85,
    21,85,84,81,105,166,153,102,89,101,149,85,21,85,84,81,89,102,153,101,85,85,85,81,89,102,153,101,85,85,84,81,
    89,102,149,85,21,81,68,17,89,85,85,81,89,102,149,85,106,166,153,101,21,85,68,17,89,85,85,81,89,101,85,85,
    68,84,17,85,68,85,81,85,68,85,81,85,68,85,81,85,68,85,17,85,68,84,17,81,153,153,102,102,153,153,101,102,
    85,153,85,101,85,85,85,85,68,85,17,81,153,153,101,102,85,149,85,85,68,85,17,17,153,153,85,101,85,85,17,85,
    153,89,85,102,85,21,17,85,153,89,85,101,85,21,17,17,149,89,85,85,153,89,102,102,85,21,17,17,149,89,85,85,
    153,89,85,85,169,106,102,102,85,21,17,17,85,21,17,17,153,89,85,85,153,89,85,85,153,89,85,85,153,89,85,85,
];

pub(crate) fn bitmap_quad_at(m: &[f32; 6], size: [u32; 2], origin: [i32; 2], rect: [f32; 4], x: f32, y: f32, physical_origin: [f32; 2]) -> [[f32; 3]; 2] {
    let [l, t, r, b] = rect;
    let pos = if (y - t) * (r - l) > (x - l) * (b - t) { [[l, b], [l, t], [r, b]] }
        else { [[l, t], [r, t], [r, b]] };
    let det = m[0] * m[3] - m[1] * m[2];
    let uv = pos.map(|p| {
        let phase = [(m[3] * p[0] + -m[2] * p[1] + (m[2] * m[5] - m[3] * m[4])) / det,
            (-m[1] * p[0] + m[0] * p[1] + (m[1] * m[4] - m[0] * m[5])) / det];
        core::array::from_fn::<_, 2, _>(|c| (phase[c] - p[c]) * (1.0 / size[c] as f32) + (p[c] - origin[c] as f32) / size[c] as f32)
    });
    let sx = pos.map(|p| ((p[0] - physical_origin[0]) * 256.0).round_ties_even() as i64);
    let sy = pos.map(|p| ((p[1] - physical_origin[1]) * 256.0).round_ties_even() as i64);
    core::array::from_fn(|c| crate::warp::plane::setup_triangle_attr(sx, sy, uv.map(|v| v[c])))
}

fn bitmap_fetch(bitmap: &crate::bitmap::Bitmap, extend: [Extend; 2], x: i32, y: i32) -> [f32; 4] {
    if (extend[0] == Extend::Transparent && (x < 0 || x >= bitmap.width as i32))
        || (extend[1] == Extend::Transparent && (y < 0 || y >= bitmap.height as i32)) { return [0.0; 4]; }
    bitmap.texel_rgba(extend[0].wrap(x, bitmap.width as i32), extend[1].wrap(y, bitmap.height as i32))
}

/// The realized vertical pass of a high-quality bitmap resample.
#[derive(Clone)]
pub struct BitmapHq {
    size: [i32; 2],
    origin: [i32; 2],
    samples: Vec<[f32; 4]>,
    precision: u32,
    /// The 2x2 placing the intermediate after its downscale passes.
    residual: Option<[f32; 4]>,
    /// The dest the intermediate was realized for; a rotated placement draws its quad over it.
    rect: [f32; 4],
    /// The placement's own intermediate's slot in the frame's atlas (its texel at the quad's origin).
    slot: [i32; 2],
}

impl BitmapHq {
    fn rotated(&self) -> bool { self.residual.is_some_and(|r| r[1] != 0.0 || r[2] != 0.0) }
}

/// `whole_direct`: a whole-pixel translation draws direct (the rect path); a geometry fill passes `false`, as native
/// places it with the residual cubic all the same (DDI of a minified HQ ellipse fill at a whole offset: the residual
/// 1.000002 scale and the translation -20.00004).
fn capped_hq_request(m: &[f32; 6], rect: [f32; 4], whole_direct: bool) -> Option<([f32; 2], [f32; 4], [f32; 4], [f32; 4])> {
    let minimum = crate::sampling::HQ_MIN_SCALE;
    // Native downscales onto a grid anchored at the bitmap's origin, then places that intermediate with a cubic at
    // the device translation (DDI: a source sub-rect moves only that last offset). At a whole-pixel translation the
    // placement is exact and the direct downscale at the device pixel is the same (the sampler suite), so only a
    // fractional translation, a scale past the minimum, or an x pass that runs first takes the intermediate.
    let capped = m[0] < minimum || m[3] < minimum;
    let whole = whole_direct && m[4].fract() == 0.0 && m[5].fract() == 0.0;
    // A mixed scale (one axis magnifying) keeps its own downscale-then-magnify path (r23 28030).
    let mixed = m[0] > 1.0 || m[3] > 1.0;
    let scales = hq_scales(m).map(|s| s.clamp(minimum, 1.0));
    // Native downscales y first unless x shrinks more; the direct path only runs y first.
    let x_first = scales[0] < scales[1] && scales[1] < 1.0;
    // A rotated or skewed matrix downscales each shrinking column the same way, then places the intermediate with
    // the residual 2x2 (probe-hqrot).
    let rotated = m[1] != 0.0 || m[2] != 0.0;
    if rotated { if !(scales[0] < 1.0 || scales[1] < 1.0) { return None; } }
    else if m[0] <= 0.0 || m[3] <= 0.0 || (m[0] >= 1.0 && m[3] >= 1.0) || (!capped && ((whole && !x_first) || mixed)) { return None; }
    let r = [m[0] * (1.0 / scales[0]), m[1] * (1.0 / scales[0]), m[2] * (1.0 / scales[1]), m[3] * (1.0 / scales[1])];
    let out = hq_tap_rect(m, r, rect);
    // The first pass covers the second pass's taps along the second pass's axis.
    let b = usize::from(x_first);
    let mut first = out;
    if scales[b] < 1.0 {
        let taps = ((2.0 / scales[b]) as i32 + 3) & !3;
        first[b] = ((out[b] + 0.5) / scales[b] - 0.5).floor() - (taps - 1) as f32;
        first[b + 2] = ((out[b + 2] - 0.5) / scales[b] - 0.5).floor() + (taps + 1) as f32;
    }
    Some((scales, r, out, first))
}

// The intermediate a residual `r` places covers the cubic taps of every pixel the dest's whole-pixel bounds hold.
fn hq_tap_rect(m: &[f32; 6], r: [f32; 4], rect: [f32; 4]) -> [f32; 4] {
    let rotated = m[1] != 0.0 || m[2] != 0.0;
    let rect = [rect[0].floor(), rect[1].floor(), rect[2].ceil(), rect[3].ceil()];
    let det = r[0] * r[3] - r[1] * r[2];
    let corners = [[rect[0] + 0.5, rect[1] + 0.5], [rect[2] - 0.5, rect[1] + 0.5], [rect[2] - 0.5, rect[3] - 0.5], [rect[0] + 0.5, rect[3] - 0.5]];
    let at = |p: [f32; 2]| if rotated {
        [(r[3] * (p[0] - m[4]) - r[2] * (p[1] - m[5])) / det, (r[0] * (p[1] - m[5]) - r[1] * (p[0] - m[4])) / det]
    } else { [(p[0] - m[4]) / r[0], (p[1] - m[5]) / r[3]] };
    let mut out = [0.0; 4];
    for i in 0..2 {
        out[i] = (corners.iter().map(|p| at(*p)[i]).fold(f32::INFINITY, f32::min) - 0.5).floor() - 1.0;
        out[i+2] = (corners.iter().map(|p| at(*p)[i]).fold(f32::NEG_INFINITY, f32::max) - 0.5).ceil() + 2.0;
    }
    out
}

/// A HIGH_QUALITY_CUBIC flood placed by a translation whose affine pre-scale shrinks an axis (an offset so large that
/// (m11 + dx) − dx loses the unit in f32): native pre-scales the flood into an intermediate covering the placement's
/// taps over `reach`, then places it with the residual cubic. That intermediate is the flood colour with a transparent
/// apron, and the placement's interpolated texcoord and its per-pixel phase are rounded apart at such offsets, so a tap
/// can reach the apron: an outer (negative-weight) tap there brightens the pixel (VM DDI of geo/flw at 120 dpi: native's
/// own three draws, replayed in f32, give its ±2 band exactly). `colour` is the flood's BGRA in 0..1; `None` when
/// neither axis is pre-scaled. The closure gives the 8-bit placed source at a device pixel centre.
pub(crate) fn flood_hq(colour: [f32; 4], m: [f32; 6], reach: [f32; 4]) -> Option<impl Fn(f32, f32) -> [f32; 4]> {
    let scales = hq_scales(&m);
    if scales == [1.0; 2] {
        return None;
    }
    let r = [m[0] * (1.0 / scales[0]), 0.0, 0.0, m[3] * (1.0 / scales[1])];
    let out = hq_tap_rect(&m, r, reach).map(|v| v as i32);
    let size = [out[2] - out[0], out[3] - out[1]];
    if size[0] <= 0 || size[1] <= 0 || size[0].max(size[1]) > 4096 {
        return None;
    }
    let c = BitmapHq { size, origin: [out[0], out[1]], samples: vec![colour; (size[0] * size[1]) as usize], precision: 1, residual: Some(r), rect: reach, slot: [0; 2] };
    Some(move |x: f32, y: f32| crate::effects::quantize_to(1, hq_place(&c, &m, reach, x, y, false).map(|v| v.clamp(0.0, 1.0))))
}

// Native's HQ downscale factors (the affine transform effect's pre-scale). A pure scale passes through; otherwise each
// axis is the length of the transformed unit vector, (m11 + dx) - dx in f32, clamped to [0.01, 1] and sent through
// 0.5^(ln s / ln 0.5) with the system runtime's logf and powf, the exponent snapped to an integer within 1e-5. The
// translation perturbs a factor by a few ulps (0.4 at dx 2.6 is 0.4000001), which can drop a tap or swap the passes.
pub(crate) fn hq_scales(m: &[f32; 6]) -> [f32; 2] {
    if m[1] == 0.0 && m[2] == 0.0 && m[4] == 0.0 && m[5] == 0.0 && m[0] as f64 > 0.01 && m[3] as f64 > 0.01 { return [m[0], m[3]]; }
    let pre = |unit: f32| {
        let s = unit.abs().min(1.0).max(f32::from_bits(0x3c23_d70a));
        let e = (crate::crt::system_crt::logf(s) as f64 / -core::f64::consts::LN_2) as f32;
        let r = e.round_ties_even();
        crate::crt::system_crt::powf(0.5, if (e - r).abs() < 1e-5 { r } else { e })
    };
    // A column's length: the affine effect's `hq_prescale_factor` (|(m11 + dx) - dx| for an axis-aligned matrix).
    let len = |a: f32, b: f32| { let (u, v) = (m[4] - (a + m[4]), m[5] - (b + m[5])); (v * v + u * u).sqrt() };
    [pre(len(m[0], m[1])), pre(len(m[2], m[3]))]
}

// A high-quality-cubic dest with a fractional edge is placed as a quad over its whole-pixel bounds, drawn into its own
// target at their origin, so its position and texcoord planes are fit on those corners there; a pixel-aligned dest is
// placed where it is. Returns the quad and the origin its planes are evaluated from.
fn hq_place_quad(rect: [f32; 4], origin: [f32; 2]) -> ([f32; 4], [f32; 2]) {
    let q = [rect[0].floor(), rect[1].floor(), rect[2].ceil(), rect[3].ceil()];
    (q, if q == rect { origin } else { [q[0], q[1]] })
}

fn hq_vertical(bitmap: &crate::bitmap::Bitmap, extend: [Extend; 2], scale: f32, py: f32, uv: [f32; 2], precision: u32) -> [f32; 4] {
    crate::effects::quantize_to(precision, hq_axis_sample(bitmap, extend, 1, scale, py, uv))
}

fn hq_axis_sample(bitmap: &crate::bitmap::Bitmap, extend: [Extend; 2], axis: usize, scale: f32, position: f32, uv: [f32; 2]) -> [f32; 4] {
    let highp = crate::float::is_highp(bitmap.format);
    let tex = Texture { data: &bitmap.premul_bgra, width: bitmap.width, height: bitmap.height };
    // WARP.s HQ axis shader uses point texture reads, including at rounded phases.
    let q = |v: f32| ((v * 256.0).floor() as i32) >> 8;
    let fetch = |x, y| { let p = bitmap_fetch(bitmap, extend, x, y); [p[2], p[1], p[0], p[3]] };
    let dims = [bitmap.width, bitmap.height];
    crate::sampling::hq_downscale_axis(position / scale, position, uv[axis], 1.0 / dims[axis] as f32, scale, 0.0, !highp, |v| {
        let mut at = uv; at[axis] = v;
        let (qx, qy) = (q(at[0] * bitmap.width as f32), q(at[1] * bitmap.height as f32));
        if highp { fetch(qx, qy) }
        else { core::array::from_fn(|ch| crate::sampling::norm_v128(tex.bilinear_v128(extend, ch, (qx, 0), (qy, 0)))) }
    })
}

// The residual cubic placing a realized intermediate `c` at device pixel centre (x, y).
fn hq_place(c: &BitmapHq, m: &[f32; 6], rect: [f32; 4], x: f32, y: f32, local: bool) -> [f32; 4] {
    let dim = (c.size[0].max(c.size[1]) as u32).next_power_of_two().max(512);
    let step = 1.0 / dim as f32;
    let mut remaining = *m;
    remaining[..4].copy_from_slice(&c.residual.unwrap());
    let (q, o) = hq_place_quad(rect, [0.0; 2]);
    // A draw the placement cannot take directly (a geometry, a mask, a layer) draws the brush into its own
    // intermediate at the quad's origin first (DDI: a rounded-rect fill's placement renders to an atlas from (6, 4)).
    let o = if local { [q[0], q[1]] } else { o };
    let o = [o[0] - c.slot[0] as f32, o[1] - c.slot[1] as f32];
    let (lx, ly) = (x - o[0], y - o[1]);
    let planes = bitmap_quad_at(&remaining, [dim; 2], c.origin, q, x, y, o);
    let uv = planes.map(|p| p[0] * lx + (p[1] * ly + p[2]));
    let [l, t, r, b] = q;
    let pos = if (y - t) * (r - l) > (x - l) * (b - t) { [[l, b], [l, t], [r, b]] } else { [[l, t], [r, t], [r, b]] };
    let scene: [f32; 2] = core::array::from_fn(|ch| {
        let p = crate::warp::plane::setup_triangle_attr(pos.map(|p| ((p[0] - o[0]) * 256.0) as i64), pos.map(|p| ((p[1] - o[1]) * 256.0) as i64), pos.map(|p| p[ch]));
        p[0] * lx + (p[1] * ly + p[2])
    });
    let r = remaining;
    let det = r[0] * r[3] - r[1] * r[2];
    let phase = [((r[3] * scene[0] - r[2] * scene[1]) + (r[2] * m[5] - r[3] * m[4])) / det,
        ((-r[1] * scene[0] + r[0] * scene[1]) + (r[1] * m[4] - r[0] * m[5])) / det];
    let axis = |p: f32, u: f32| {
        let f = (p - 0.5) - (p - 0.5).floor();
        let w = crate::sampling::cubic_weights(f, 0.0); let g = w[1] + w[2];
        let base = u - f * step;
        ([base - step, base + (w[2] / g) * step, base + 2.0 * step].map(|u| {
            let q = ((u * dim as f32 - 0.5) * 256.0).floor() as i32; (q >> 8, q & 255)
        }), [w[0], g, w[3]])
    };
    let (xs, wx) = axis(phase[0], uv[0]); let (ys, wy) = axis(phase[1], uv[1]);
    let fetch = |x: i32, y: i32| {
        // Native's realized atlas is clamped; its unused positive apron remains transparent.
        let (x, y) = (x.clamp(0, dim as i32 - 1), y.clamp(0, dim as i32 - 1));
        if x >= c.size[0] || y >= c.size[1] { [0.0; 4] }
        else { c.samples[(y * c.size[0] + x) as usize] }
    };
    if c.precision >= 2 { crate::sampling::cubic_windows_f32(xs, ys, wx, wy, fetch) }
        else { crate::sampling::hqcubic_windows_unclamped(xs, ys, wx, wy, |x, y| fetch(x, y).map(|v| (v * 255.0).round_ties_even() as i32)) }
}

// Native HQ minification realizes a vertical pass in a 512-wide atlas before the horizontal pass.
fn bitmap_hq_min(bitmap: &crate::bitmap::Bitmap, extend: [Extend; 2], m: &[f32; 6], rect: [f32; 4], x: f32, y: f32, cache: Option<&BitmapHq>, local: bool) -> [f32; 4] {
    if (m[0] == 1.0 || m[3] == 1.0) && cache.is_none_or(|c| c.residual.is_none()) {
        // Native one-axis HQ DDI emits a single downscale pass, with no intermediate store.
        let axis = usize::from(m[3] < 1.0);
        let p = [x - m[4], y - m[5]];
        return hq_axis_sample(bitmap, extend, axis, hq_scales(m)[axis], p[axis], [p[0] / bitmap.width as f32, p[1] / bitmap.height as f32]);
    }
    if let Some(c) = cache.filter(|c| c.residual.is_some()) {
        return hq_place(c, m, rect, x, y, local);
    }
    let precision = cache.map_or(if crate::float::is_highp(bitmap.format) { 5 } else { 1 }, |c| c.precision);
    let q = |v: f32| ((v * 256.0).floor() as i32) >> 8;
    let s = hq_scales(m);
    let taps = ((2.0 / s[0]) as i32 + 3) & !3;
    let left = cache.map_or_else(|| (((rect[0].floor() + 0.5 - m[4]) / s[0] - 0.5).floor() as i32) - (taps - 1), |c| c.origin[0]);
    let top = cache.map_or_else(|| (rect[1].floor() - m[5]).floor() as i32, |c| c.origin[1]);
    let intermediate = |ix: i32, iy: i32| {
        if let Some(c) = cache {
            if ix < 0 || iy < 0 || ix >= c.size[0] || iy >= c.size[1] { return [0.0; 4]; }
            return c.samples[(iy * c.size[0] + ix) as usize];
        }
        hq_vertical(bitmap, extend, s[1], iy as f32 + top as f32 + 0.5, [(ix as f32 + left as f32 + 0.5) / bitmap.width as f32, (iy as f32 + top as f32 + 0.5) / bitmap.height as f32], precision)
    };
    let px = x - m[4];
    let py = y - m[5];
    crate::sampling::hq_downscale_axis(px / s[0], px, (px - left as f32) / 512.0, 1.0 / 512.0, s[0], 0.0, precision < 2, |u| {
        let (qx, qy) = (q(u * 512.0), q(py - top as f32));
        if precision >= 2 { intermediate(qx, qy) }
        else {
            intermediate(qx, qy).map(|v| crate::sampling::norm_v128((v * 255.0).round_ties_even() as i32 * 128))
        }
    })
}

// Colour spaces (D2D1_COLOR_SPACE / native collection fields): sRGB=1, scRGB(linear)=2.
const SRGB: u32 = 1;
const SCRGB: u32 = 2;

/// Round half up to `[0,255]` — WARP's gradient-texel quantiser (`cvtss2si` round-to-nearest-even
/// with exact `.5` bumped up equals `floor(x+0.5)` for `x>=0`).
fn q8(x: f32) -> u8 {
    (x + 0.5).floor().clamp(0.0, 255.0) as u8
}

/// sRGB→scRGB decode (`sRGBToscRGB`), clamping to `[0,1]` (native returns 0 below, 1 at/above 1).
fn srgb_to_scrgb(x: f32) -> f32 {
    if x <= 0.0 {
        0.0
    } else if x <= 0.04045 {
        x / 12.92
    } else if x < 1.0 {
        crate::crt::system_crt::powf((x + 0.055) / 1.055, 2.4)
    } else {
        1.0
    }
}

/// scRGB(linear)→sRGB byte via native's 3355-entry `GammaLUT_scRGB_to_sRGB`. The table is
/// `round_half_up(255·srgb_encode(i/3354))`; generated here from the sRGB transfer function
/// (verified byte-identical to the shipped table) rather than copied from `ref/`.
fn scrgb_to_srgb_byte(x: f32) -> u8 {
    static TABLE: OnceLock<[u8; 3355]> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u8; 3355];
        for (i, slot) in t.iter_mut().enumerate() {
            let l = i as f64 / 3354.0;
            let s = if l <= 0.003_130_8 { 12.92 * l } else { 1.055 * crate::crt::pow(l, 1.0 / 2.4) - 0.055 };
            *slot = (255.0 * s + 0.5).floor() as u8;
        }
        t
    });
    if x <= 0.0 {
        0
    } else if x >= 1.0 {
        255
    } else {
        table[(x * 3354.0 + 0.5).floor() as usize]
    }
}

/// Integer premultiply (`Premultiply`): `round(a·c/255)` per channel via the `+128` trick.
fn premultiply_int(bgra: [u8; 4]) -> [u8; 4] {
    let a = bgra[3] as u32;
    if a == 255 {
        return bgra;
    }
    if a == 0 {
        return [0, 0, 0, 0];
    }
    let f = |c: u8| {
        let x = a * c as u32;
        ((x + 128 + ((x + 128) >> 8)) >> 8) as u8
    };
    [f(bgra[0]), f(bgra[1]), f(bgra[2]), bgra[3]]
}

/// Convert a straight interp-space colour `[r,g,b,a]` to a premultiplied BGRA8 texel, matching
/// the stop post-process for the legacy 8bpc/straight path.
fn post_process(c: [f32; 4], interp_space: u32, post_space: u32) -> [u8; 4] {
    // 8BPC_UNORM buffer-precision clamp to [0,1] on every channel.
    let [r, g, b, a] = c.map(|v| v.clamp(0.0, 1.0));
    if interp_space == post_space {
        // No gamma conversion (e.g. gamma 2.2: interp sRGB == post sRGB): float premultiply.
        [q8(b * a * 255.0), q8(g * a * 255.0), q8(r * a * 255.0), q8(a * 255.0)]
    } else if interp_space == SCRGB {
        // scRGB→premultiplied sRGB (gamma 1.0): table encode straight, then integer premultiply.
        premultiply_int([scrgb_to_srgb_byte(b), scrgb_to_srgb_byte(g), scrgb_to_srgb_byte(r), q8(a * 255.0)])
    } else {
        // interp sRGB → post scRGB (decode then premultiply); not reachable from legacy scenes.
        let d = |ch: f32| q8(srgb_to_scrgb(ch) * a * 255.0);
        [d(b), d(g), d(r), q8(a * 255.0)]
    }
}

/// A realized gradient texture. Native builds a 1024-wide 11-level mip chain
/// , but at draw the brush pool binds the **single**
/// power-of-two level matching the device footprint as the SRV's
/// most-detailed mip and WARP samples that one level — there is no in-shader mip blend. So we realize
/// only that level here, box-filtered per the gradient-texture realize. `u0`/`u1` are the texel span
/// `t∈[0,1]` maps to.
#[derive(Clone)]
pub struct Gradient {
    texels: Vec<u8>, // `width` premultiplied BGRA8 texels, tightly packed, one row
    width: usize,    // chosen realization level
    extend: u32,
    u0: f32,
    u1: f32,
    coarsest: [u8; 4], // the mip chain's 1×1 level
}

fn extend_of(mode: u32) -> Extend {
    match mode {
        1 => Extend::Wrap,
        2 => Extend::Mirror,
        // 3 is not a D2D1_EXTEND_MODE; it is our internal signal for a transparent border (the
        // anisotropic source-rect crop's realize), which also builds the mip chain unpadded.
        3 => Extend::Transparent,
        _ => Extend::Clamp,
    }
}

/// A well-formed stop (position in `[0,1]`, straight colour in interp space).
type WfStop = (f32, [f32; 4]);

/// Near-equal test: `|(a − b) / (b == 0 ? 1 : b)| < FLT_EPSILON·10`-ish (`0x35a00000`), false on NaN.
fn is_close(a: f32, b: f32) -> bool {
    ((a - b) / if b == 0.0 { 1.0 } else { b }).abs() < CLOSE_EPS
}
/// `__real@35a00000`, d2d1's gradient position tolerance.
const CLOSE_EPS: f32 = f32::from_bits(0x35a0_0000);

/// Colour interpolation: the colour at `t` between `c0` (at `p0`) and `c1`,
/// `f = (t − p0) / span`.
fn interpolate_colors(c0: [f32; 4], c1: [f32; 4], t: f32, span: f32, p0: f32) -> [f32; 4] {
    let f = (t - p0) / span;
    core::array::from_fn(|j| (1.0 - f) * c0[j] + f * c1[j])
}

/// The gradient-array builder, op for op against Direct2D:
/// the stops (converted to the interpolation space) become a ramp from exactly 0 to exactly 1, plus
/// the leading/trailing border colours. Every compare keeps its x86 NaN semantics (`comiss` + `ja` /
/// `jbe`), which is what makes a NaN position behave as native: the insertion sort never moves a NaN
/// stop or anything past it, a NaN first stop is "positive" (copied to 0), and the main walk stops at
/// a NaN, so the stop at 1 interpolates toward it into a NaN colour that `fill_span` stores as 0
/// (e009b/rs-nan* probes: transparent from the NaN stop on).
// The negated compares are the x86 NaN semantics (`!(a > b)` is `jbe`, true when unordered).
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn well_formed(stops: &[(f32, [f32; 4])], interp_space: u32) -> (Vec<WfStop>, [f32; 4], [f32; 4]) {
    let mut s: Vec<WfStop> = stops
        .iter()
        .map(|&(p, c)| {
            let c = c.map(|v| v.clamp(0.0, 1.0)); // 8bpc precision clamp
            let c = if interp_space == SCRGB {
                [srgb_to_scrgb(c[0]), srgb_to_scrgb(c[1]), srgb_to_scrgb(c[2]), c[3]]
            } else {
                c
            };
            (p, c)
        })
        .collect();
    // Insertion sort: shift while `prev > cur` (`comiss; ja`), so an unordered (NaN) compare stops it.
    for i in 1..s.len() {
        let cur = s[i];
        let mut k = i;
        while k > 0 && s[k - 1].0 > cur.0 {
            s[k] = s[k - 1];
            k -= 1;
        }
        s[k] = cur;
    }
    let n = s.len();
    // SetFirstStop: a first stop not below 0 and not ≈0 (a NaN included) is copied to t=0.
    let p0 = s[0].0;
    let (first, mut src) = if !(0.0 > p0) && !(CLOSE_EPS > (p0 - 0.0).abs()) {
        s.insert(0, (0.0, s[0].1));
        (s[0].1, 1)
    } else {
        // The first stop at or above 0 (a NaN or ≈0 one ends the scan too).
        let mut i = 0;
        while i < n && 0.0 > s[i].0 && !is_close(s[i].0, 0.0) {
            i += 1;
        }
        if i == n {
            // All below 0: the ramp is the last stop's colour from 0.
            s[0] = (0.0, s[n - 1].1);
            (s[0].1, n)
        } else if !(CLOSE_EPS > (s[i].0 - 0.0).abs()) {
            let c = interpolate_colors(s[i - 1].1, s[i].1, 0.0, s[i].0 - s[i - 1].0, s[i - 1].0);
            s[0] = (0.0, c);
            (c, i)
        } else {
            // ≈0: the first of the run is the leading colour, the ramp starts from the last of it.
            let first = s[i].1;
            let mut j = i + 1;
            while j < n && CLOSE_EPS > (s[j].0 - 0.0).abs() {
                j += 1;
            }
            s[0] = (0.0, s[j - 1].1);
            (first, j)
        }
    };
    let n = s.len();
    let mut dst = 1;
    // Copy the interior stops, collapsing near-duplicate runs to their last one; stop at 1, ≈1 or NaN.
    while src < n {
        let p = s[src].0;
        if !(1.0 > p) || CLOSE_EPS > (p - 1.0).abs() {
            break;
        }
        let prev = s[src - 1].0;
        let den = if p == 0.0 { 1.0 } else { p };
        if CLOSE_EPS > ((prev - p) / den).abs() {
            let mut i = src + 1;
            while i < n && 1.0 > s[i].0 && !is_close(s[i].0, 1.0) && is_close(prev, s[i].0) {
                i += 1;
            }
            src = i - 1;
            s[src].0 = prev;
        }
        s[dst] = s[src];
        dst += 1;
        src += 1;
    }
    let last = if src == n {
        let c = s[n - 1].1;
        if dst < n { s[dst] = (1.0, c) } else { s.push((1.0, c)) }
        c
    } else if !(CLOSE_EPS > (s[src].0 - 1.0).abs()) {
        let c = interpolate_colors(s[src - 1].1, s[src].1, 1.0, s[src].0 - s[src - 1].0, s[src - 1].0);
        s[dst] = (1.0, c);
        c
    } else {
        s[dst] = (1.0, s[src].1);
        let mut j = src + 1;
        while j < n && is_close(s[j].0, 1.0) {
            j += 1;
        }
        s[j - 1].1
    };
    s.truncate(dst + 1);
    (s, first, last)
}

/// The gradient box-filter: box-filter the well-formed ramp over `out.len()` texels, writing
/// premultiplied BGRA8. Texels wholly inside a stop pair take the centre lerp; the texel holding a
/// stop is the area average of the pairs it overlaps (skipping zero-length pairs). Ported with the
/// x86 conversions and compares: the pair end `floor(p1·n)` is `cvttss2si` (a NaN position gives
/// `i32::MIN`, so 0 texels), and the straddle walk leaves on `!(t_end > p1)`, so a NaN pair feeds a
/// NaN weight into the average (stored as 0).
#[allow(clippy::neg_cmp_op_on_partial_ord)] // x86 NaN semantics, as in `well_formed`
fn fill_span(out: &mut [[u8; 4]], wf: &[WfStop], interp_space: u32, post_space: u32) {
    let n = out.len() as i32;
    let nf = n as f32;
    let inv = 1.0_f32 / nf;
    let half = inv * 0.5;
    let post = |c: [f32; 4]| post_process(c, interp_space, post_space);
    let last = wf.len() - 1;
    let (mut a, mut b) = (0usize, 1usize); // the current pair wf[a], wf[b]
    let mut k = 0i32;
    while k < n {
        let v = wf[b].0 * nf;
        let t = if v.is_nan() || v.abs() >= 2_147_483_648.0 { i32::MIN } else { v as i32 };
        let fl = if !((t as f32) > v) { t } else { t.wrapping_sub(1) };
        let end = if fl > n { n } else { fl.max(0) };
        let seg = wf[b].0 - wf[a].0;
        while k < end {
            let f = ((k as f32 * inv + half) - wf[a].0) / seg;
            out[k as usize] = post(core::array::from_fn(|j| f * wf[b].1[j] + (1.0 - f) * wf[a].1[j]));
            k += 1;
        }
        if k >= n {
            break;
        }
        let (t0, t1) = (k as f32 * inv, (k + 1) as f32 * inv);
        k += 1;
        let mut acc = [0.0_f32; 4];
        let add = |acc: &mut [f32; 4], a: usize, b: usize| {
            let (q0, q1) = (wf[a].0, wf[b].0);
            let lo = if t0 > q0 { t0 } else { q0 }; // maxss t0, q0
            let ov = (if t1 < q1 { t1 } else { q1 }) - lo; // minss t1, q1
            if !(0.0 > ov) && !(CLOSE_EPS > (ov - 0.0).abs()) {
                let w = ov * nf;
                let mid = ((ov * 0.5 + lo) - q0) / (q1 - q0);
                for j in 0..4 {
                    acc[j] += (mid * wf[b].1[j] + (1.0 - mid) * wf[a].1[j]) * w;
                }
            }
        };
        loop {
            if !(t1 > wf[b].0) {
                add(&mut acc, a, b); // the texel ends inside this pair
                break;
            }
            add(&mut acc, a, b);
            if b == last {
                break;
            }
            a += 1;
            b += 1;
            if wf[a].0 == wf[b].0 {
                if b == last {
                    break;
                }
                a += 1;
                b += 1;
            }
        }
        out[(k - 1) as usize] = post(acc);
    }
}

/// Device-footprint → texture width (`CalcTextureSize`): `floor(clamp(footprint,0,1024))` plus a
/// two-texel clamp border (none for wrap/mirror), rounded up to a power of two, cap 1024. (Native
/// gives a clamped radial only one border texel, but that never changes the power-of-two width for
/// any captured radial scene, and the radial span is baked into its realization coefficients.)
fn view_width(footprint: f32, extend: u32) -> usize {
    // Degenerate axis (a NaN footprint, from a non-finite endpoint sanitised to NaN in Brush::linear):
    // native's GetRealizationInfoAndTextureMapping sizes the realization at width 1 — GetRealizationSize
    // returns NaN, `fmaxf(NaN,0)=0`, and the width logic collapses to 1 (fuzz 17006:
    // width=1). A FINITE endpoint whose axis-length SQUARE overflows to +inf (e.g. end.y=3.4e38,
    // grad-degen-265) is NOT this case: native's `fmaxf(inf,0)=inf`, `fminf(1024,inf)=1024` → full width
    // 1024, so only a NaN footprint collapses to 1 (an `is_finite` test here wrongly caught +inf too).
    if footprint.is_nan() {
        return 1;
    }
    let mut v = footprint.clamp(0.0, 1024.0) as u32; // truncate == floor for a finite ≥0 footprint
    if extend == 0 {
        // Clamp (Pad): a leading + trailing border texel bracket the interior ramp (`t`→span `[1, W−1]`).
        // A sub-pixel axis (footprint < 1 → `W = 2`) has ZERO interior texels; the W==2 span is fixed up
        // in `Gradient::new` so it still Pad-clamps (see the `width == 2` arm there) rather than washing
        // flat. The texture generator never fills a linear span narrower than 2 texels (VM gradspan).
        v += 2;
    } else if v <= 1 {
        v = 1;
    }
    if v >= 1024 {
        return 1024;
    }
    v.max(1).next_power_of_two() as usize
}

/// The rect→parallelogram transform, op for op: the general three-point
/// affine solve from rect `[l, t, r, b]` to the parallelogram `{p0, p1, p2}`. The terms that multiply a
/// zero rect difference (`(t−t)·p2`, `(l−l)·p1`, the `0·p` translation terms) stay in: with an infinite
/// endpoint they are NaN, which fails `Invert` and collapses the radial to one texel (r13 18746's
/// −3.4e38 radius at dpi 192; native uploads a zero map and W = 1).
// The `(t − t)` and `(l − l)` terms (and the `0·p` translations) are DELIBERATE, not redundant: for an
// infinite rect endpoint they evaluate to NaN (not 0), which propagates through `Invert` and collapses the
// radial to one texel exactly as native does (see the doc comment above, r13 18746). They must keep the
// literal self-subtraction form, so silence clippy::eq_op rather than simplifying it away. For the same
// reason silence clippy::erasing_op: the `(t − t)·p2` and `(l − l)·p1` products are not erasable zeros —
// with an infinite endpoint the `(t − t)`/`(l − l)` factor is NaN, so the product is NaN, not 0.
#[allow(clippy::eq_op, clippy::erasing_op)]
fn rect_to_parallelogram([l, t, r, b]: [f32; 4], [p0, p1, p2]: [[f32; 2]; 3]) -> Mat {
    let w = r - l;
    let tl = t * l;
    let d0 = b * r - tl;
    let inv_det = 1.0 / ((((t - b) * l) - w * t) + d0);
    let d1 = tl - b * l;
    let d2 = tl - r * t;
    [
        (((b - t) * p1[0] + (t - b) * p0[0]) + (t - t) * p2[0]) * inv_det,
        (((b - t) * p1[1] + (t - b) * p0[1]) + (t - t) * p2[1]) * inv_det,
        (((l - l) * p1[0] + (l - r) * p0[0]) + w * p2[0]) * inv_det,
        (((l - l) * p1[1] + (l - r) * p0[1]) + w * p2[1]) * inv_det,
        ((d1 * p1[0] + d0 * p0[0]) + d2 * p2[0]) * inv_det,
        ((d1 * p1[1] + d0 * p0[1]) + d2 * p2[1]) * inv_det,
    ]
}

/// Radial device-footprint → texture width. Native's `GetRealizationInfoAndTextureMapping` radial
/// branch (the border branch) sizes `W = next_pow2(floor(footprint) + 1)`,
/// cap 1024 — the `+1` is the trailing clamp-border texel. Verified against WARP
/// sweep (`0.5/W`): r∈{1,2,4,8,16}→W∈{2,4,8,16,32}. (canvas-radial footprints are non-integer
/// with focal offsets, so `next_pow2(floor f)` happened to equal `next_pow2(floor f + 1)` for them;
/// the emoji's many small integer-ish footprints expose the missing `+1`, halving W and collapsing
/// the gradient to a flat wash.)
pub(crate) fn radial_width(footprint: f32, extend: u32) -> usize {
    // Native rounds floor(footprint) up to a power of two, after adding the border texels that precede
    // that round: a clamped radial reserves one trailing border texel, while a wrap/mirror ramp wraps
    // and reserves none — so a footprint whose floor is already a power of two keeps that width instead
    // of doubling (r20 25764: a WRAP radial with footprint 32.38 realizes at W 32, not 64).
    let fp = footprint.clamp(0.0, 1024.0).floor() as u32;
    let v = if extend == 0 { fp + 1 } else { fp };
    (v.next_power_of_two() as usize).min(1024)
}

/// Fold a texel index onto `0..n` for the extend mode — WARP's integer index clamp/wrap/mirror
/// after the half-texel shift (`crate::sampling::Extend::wrap` is the same, but lives in a module
/// this task does not own, and the gradient filter's lerp differs from the bitmap one anyway).
fn fold(ex: Extend, c: i32, n: i32) -> i32 {
    match ex {
        Extend::Clamp => c.clamp(0, n - 1),
        Extend::Wrap => c.rem_euclid(n),
        Extend::Mirror => {
            let p = 2 * n;
            let m = c.rem_euclid(p);
            if m < n { m } else { p - 1 - m }
        }
        Extend::Transparent => c.clamp(0, n - 1), // never used by gradient brushes (blur-magnify only)
    }
}

/// WARP's `cvttss2si` texel index from a texel-space coord `u`, folded under `ex`: floor `u` (the
/// sampler's round-toward-−∞), but a non-finite coord or one past the i32 range yields INT_MIN, so a
/// huge or degenerate texcoord collapses to the ORIGIN texel under every extend. A finite in-range
/// coord is the ordinary `fold(floor(u))`.
pub(crate) fn texel_index(u: f32, ex: Extend, n: i32) -> i32 {
    let f = u.floor();
    let raw = if f.is_finite() && f.abs() < 2147483648.0 { f as i32 } else { i32::MIN };
    fold(ex, raw, n)
}

/// The effective 16.16 texcoord for a fixed-point DDA sampler. In the normal range the DDA value is
/// exact; once the texel-space coord `u` reaches 2¹⁵ the i32 16.16 accumulator wraps (and native does
/// not), so substitute the folded texel with a zero sub-texel fraction (a huge coord has no fraction).
/// Native's truncation overflows to INT_MIN (→ origin) at different points per filter: NEAREST truncates
/// the texel index (2³¹), LINEAR the `u·256` Q8 address (2³¹/256), so a `|u|` in between is a finite far
/// texel for NEAREST but the origin for LINEAR.
pub(crate) fn dda_texcoord(u: f32, dda: i32, ex: Extend, n: i32, linear: bool) -> i32 {
    if u.is_finite() && u.abs() < 32768.0 {
        return dda;
    }
    // A non-finite texcoord off the power-of-two path takes texcoord 0's address, half-texel bias
    // included: VM probes of a 16×10 brush with a NaN or infinite offset blend rows 9 and 0 under
    // WRAP + LINEAR and read row 0 otherwise.
    if !u.is_finite() && !(n as u32).is_power_of_two() {
        return -0x7f80;
    }
    let thresh = if linear { 2147483648.0 / 256.0 } else { 2147483648.0 };
    let f = u.floor();
    let raw = if f.is_finite() && f.abs() < thresh { f as i32 } else { i32::MIN };
    fold(ex, raw, n) << 16
}

impl Gradient {
    /// Realize the one selected mip level from stops, the legacy `D2D1_GAMMA` (0 = 2.2 / interpolate
    /// in sRGB, 1 = 1.0 / interpolate in scRGB), the extend mode and the device-space footprint.
    /// `radial` picks the texture width: a radial brush's `GetRealizationInfoAndTextureMapping`
    /// radial branch sizes with [`radial_width`] (no `+2` width border — the border for clamp is
    /// instead the two endpoint texels below), everything else is the shared gradient-texture build.
    pub fn new(stops: &[(f32, [f32; 4])], gamma: u32, extend: u32, footprint: f32, radial: bool) -> Gradient {
        let interp = if gamma != 0 { SCRGB } else { SRGB };
        let post = SRGB;
        let (wf, first, last) = well_formed(stops, interp);
        let width = if radial { radial_width(footprint, extend) } else { view_width(footprint, extend) };
        // Box-filter the texels with Direct2D.s gradient-texture border layout
        // (the leading border is gated by `brushType==3`, i.e. linear only; and the WARP
        // texture dump):
        // - Linear (BrushTypes==3) clamp: leading first-stop texel + trailing last-stop texel, box-
        //   filter the inner `W−2` texels over `t∈[0,1]`, `t`→span `[1, W−1]`.
        // - Radial (BrushTypes==4) clamp: **trailing** last-stop texel only (no leading border), box-
        //   filter the first `W−1` texels over `t∈[0,1]`; the sampler scale is the full `W` with the
        //   index clamp-fold, so `t`→span `[0, W]` (the half-texel is `sample_jit`'s own `u − 0.5`).
        // - Wrap/mirror (either type): no border, full-width box-filter, span `[0, W]`.
        let level = |width: usize| {
        let mut tx = vec![[0u8; 4]; width];
        if radial && extend == 0 {
            // Radial clamp (the texture generator, brushType!=3): **no** leading border, just a
            // trailing last-stop texel, box-filter the first `W−1`. At `W==1` the span is empty and the
            // lone texel is the last stop — the midpoint average is linear-only (linear brush type),
            // so a sub-pixel radial collapses to a flat **last-stop** wash, not the ramp midpoint.
            fill_span(&mut tx[0..width - 1], &wf, interp, post);
            tx[width - 1] = post_process(last, interp, post);
        } else if radial || extend != 0 {
            fill_span(&mut tx, &wf, interp, post);
        } else if width == 1 {
            let m: [f32; 4] = core::array::from_fn(|j| last[j] * 0.5 + first[j] * 0.5);
            tx[0] = post_process(m, interp, post);
        } else {
            tx[0] = post_process(first, interp, post);
            tx[width - 1] = post_process(last, interp, post);
            fill_span(&mut tx[1..width - 1], &wf, interp, post);
        }
        tx
        };
        let tx = level(width);
        // The chain's 1×1 level: `EnsureRealizedForBrushType` generates every level straight from the
        // stops ([`Brush::nan_lod`] samples it).
        let coarsest = if width == 1 { tx[0] } else { level(1)[0] };
        // Gradient `t∈[0,1] → texel` span (the sampler's texcoord scale): a linear clamp gradient's
        // endpoints sit in the two border texels, so `t` maps across the interior `[1, W−1]`. A radial
        // clamp texture has only the trailing border, so `t` maps across `[0, W−1]` (this is native's
        // `span_right = W−1`, equivalently its `(W−1)/W` texcoord scaling). Wrap/mirror tile the
        // whole width `[0, W]`.
        let (u0, u1) = match (extend == 0 && width > 1, radial) {
            // Linear clamp: `t` spans the interior `[1, W−1]`. At W==2 this collapses to `[1,1]`
            // (`u0==u1`); `linear_transform` detects that degenerate span and reproduces native's
            // unit-direction one-device-pixel ramp (see its `u1-u0==0` branch) rather than a flat wash.
            (true, false) => (1.0, (width - 1) as f32),
            (true, true) => (0.0, (width - 1) as f32),
            _ => (0.0, width as f32),
        };
        Gradient { texels: tx.into_iter().flatten().collect(), width, extend, u0, u1, coarsest }
    }

    /// Geometry-fill sampler — WARP.s gradient pixel shader: form
    /// the 24.8 fixed coordinate `floor((u − 0.5)·256)`, take the integer texel (`>>8`) and 8-bit
    /// sub-texel weight (`&0xff`), fold both taps by the extend mode, and lerp each channel by
    /// **truncation** — `(256·T0 + (T1 − T0)·w) >> 8`, no rounding bias. Returns premultiplied BGRA
    /// as float 0..255 (the result is already integral).
    fn sample_jit(&self, u: f32) -> [f32; 4] {
        let ex = extend_of(self.extend);
        let n = self.width as i32;
        let fixed = ((u - 0.5) * 256.0).floor() as i32;
        let w = fixed & 0xff;
        let i0 = fold(ex, fixed >> 8, n) as usize * 4;
        let i1 = fold(ex, (fixed >> 8) + 1, n) as usize * 4;
        core::array::from_fn(|c| {
            let (a, b) = (self.texels[i0 + c] as i32, self.texels[i1 + c] as i32);
            ((256 * a + (b - a) * w) >> 8) as f32
        })
    }

    /// Geometry-fill sampler that keeps the lerp **fractional**. WARP's gradient PS only truncates the
    /// lerp to an integer texel when it stores the result directly (opacity 1 → [`sample_jit`]); when
    /// the brush opacity scales the sample the PS keeps the lerp in its sampler's 8.7 fixed point (the
    /// Q8 product `>> 1`, as the JIT'd bilinear: `psllw 8` + `pmullw` + `psrlw 1`), multiplies by
    /// opacity, and the UNORM8 store rounds once. So an opacity brush must retain the sub-texel
    /// fraction here (e.g. `G=1.35` not `1`), or the ×opacity rounds a texel short at an interior ramp
    /// pixel (fuzz r4 gradient-over-fill interior ±1), but only to 1/128 (r9 14434's 76 + 173/256).
    fn sample_jit_f(&self, u: f32) -> [f32; 4] {
        let ex = extend_of(self.extend);
        let n = self.width as i32;
        let fixed = ((u - 0.5) * 256.0).floor() as i32;
        let w = fixed & 0xff;
        let i0 = fold(ex, fixed >> 8, n) as usize * 4;
        let i1 = fold(ex, (fixed >> 8) + 1, n) as usize * 4;
        core::array::from_fn(|c| {
            let (a, b) = (self.texels[i0 + c] as i32, self.texels[i1 + c] as i32);
            ((256 * a + (b - a) * w) >> 1) as f32 / 128.0
        })
    }

    /// COLR colorizer ramp sampler — the geometry gradient shader's `floor((u−0.5)·256)` address
    /// (same as [`sample_jit`]) but a **round-half-up** lerp (`+128` before `>>8`), matching the COLR
    /// place-back's `Sample`. The truncating [`sample_jit`] is systematically −1 and the float
    /// [`sample_f`] +1 on a sub-texel ramp transition; this is the exact native rounding (probe: an
    /// angled W=2 transition pixel wants B=177 AND R=78, sum 255, which only `+128` rounding yields).
    fn sample_jit_round(&self, u: f32) -> [f32; 4] {
        let ex = extend_of(self.extend);
        let n = self.width as i32;
        let fixed = ((u - 0.5) * 256.0).floor() as i32;
        let w = fixed & 0xff;
        let i0 = fold(ex, fixed >> 8, n) as usize * 4;
        let i1 = fold(ex, (fixed >> 8) + 1, n) as usize * 4;
        core::array::from_fn(|c| {
            let (a, b) = (self.texels[i0 + c] as i32, self.texels[i1 + c] as i32);
            ((256 * a + (b - a) * w + 128) >> 8) as f32
        })
    }

    // Provenance: WARP gradient-mask JIT (emoji-108, +0x63c..0x654): the Q8 lerp
    // shifts right once before conversion to float, retaining seven fractional bits.
    fn sample_text(&self, u: f32) -> [f32; 4] {
        let ex = extend_of(self.extend);
        let fixed = ((u - 0.5) * 256.0).floor() as i32;
        let w = fixed & 255;
        let i0 = fold(ex, fixed >> 8, self.width as i32) as usize * 4;
        let i1 = fold(ex, (fixed >> 8) + 1, self.width as i32) as usize * 4;
        core::array::from_fn(|c| {
            let a = self.texels[i0 + c] as i32;
            let b = self.texels[i1 + c] as i32;
            ((a * 256 + (b - a) * w) >> 1) as f32 * (1.0 / 128.0)
        })
    }

    /// Pick the gradient sampler for the fill primitive (see [`Brush::source_at`]).
    fn sample(&self, u: f32, jit: bool) -> [f32; 4] {
        if jit {
            self.sample_jit(u)
        } else {
            self.sample_rect(u)
        }
    }

    /// Rectangle fast-path sampler — `FillNonOverlappingRectangles_FastPath_Warp_BitmapBrushes`
    /// realizes the gradient as a 1-row bitmap brush and samples it through WARP's **bitmap** UNORM8
    /// filter (`crate::sampling`, round-half-up `+128` lerp), not the gradient pixel shader. Returns
    /// premultiplied BGRA as float 0..255.
    fn sample_rect(&self, u: f32) -> [f32; 4] {
        let tex = Texture { data: &self.texels, width: self.width as u32, height: 1 };
        tex.sample(Filter::Linear, [extend_of(self.extend), Extend::Clamp], u, 0.5).map(|v| v * 255.0)
    }

    /// Float render-target sampler: the *same* 8-bit gradient texels, bilinear-interpolated with
    /// **true f32 weights** (`1−frac`, `frac`) on `byte/255` taps — not the bitmap sampler's UNORM8
    /// weights — so a float target composites the ramp in full precision (oracle `hdr-11`). Returns
    /// premultiplied BGRA float 0..1. The extend fold and texel span are shared with the 8-bit path.
    fn sample_f(&self, u: f32) -> [f32; 4] {
        let ex = extend_of(self.extend);
        let n = self.width as i32;
        if !u.is_finite() {
            // WARP's address conversion (`cvtps2dq`) turns NaN/±inf into 0x80000000: texel
            // `i32::MIN >> 8`, weight 0 (r10 15098's −inf radial focal under a FillOpacityMask).
            let i = fold(ex, i32::MIN >> 8, n) as usize * 4;
            return core::array::from_fn(|c| self.texels[i + c] as f32 / 255.0);
        }
        let uu = u - 0.5;
        let x0 = uu.floor();
        let f = uu - x0;
        let (i0, i1) = (fold(ex, x0 as i32, n) as usize * 4, fold(ex, x0 as i32 + 1, n) as usize * 4);
        core::array::from_fn(|c| {
            let (a, b) = (self.texels[i0 + c] as f32 / 255.0, self.texels[i1 + c] as f32 / 255.0);
            a * (1.0 - f) + b * f
        })
    }
}

/// Where a realized bitmap brush came from — it decides how a non-finite brush matrix voids the draw.
/// A plain bitmap brush samples through AlphaBlt (a non-finite translate only shifts the texcoord, so
/// it paints the fallback texel), whereas an image brush realizes through the full matrix (a non-finite
/// entry poisons its determinant and voids the draw). The two are otherwise the same `Brush::Bitmap`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BrushOrigin {
    /// `CreateBitmapBrush` on a bitmap: voids only on a singular/non-finite 2×2.
    PlainBitmap,
    /// `CreateImageBrush` over a whole bitmap: voids on any non-finite matrix entry.
    BitmapImage,
    /// `CreateImageBrush` over an effect or a cropped bitmap, which renders through an intermediate.
    StagedImage,
    /// `CreateImageBrush` over a command list.
    OtherImage,
}

/// A realized brush producing premultiplied source colour per device pixel. Gradient textures and
/// the device-space mapping are resolved at construction from the brush→device matrix (native
/// realizes lazily at draw; the result is identical when the transform is unchanged between
/// create and draw, which is the only case the scenes exercise).
#[derive(Clone)]
pub enum Brush {
    /// Premultiplied BGRA float (already multiplied by opacity).
    Solid([f32; 4]),
    /// Linear gradient: device-space `start`/`end`, the texture (which carries the extend mode), and
    /// `xform` = native's precomputed device→texel texcoord transform `[dir_x, dir_y, offset]` (the
    /// per-pixel texcoord is `u = dir_x·px + dir_y·py + offset`). See [`linear_transform`].
    Linear { grad: Gradient, start: [f32; 2], end: [f32; 2], opacity: f32, xform: [f32; 3] },
    /// Radial gradient: `to_unit` maps a device pixel straight into unit-circle space (one affine,
    /// as WARP's precomputed realization does — fewer f32 roundings than inverse-then-normalize), and
    /// `focal` is the gradient origin there.
    Radial { grad: Gradient, to_unit: [f32; 6], to_unit_raw: [f32; 6], focal: [f32; 2], opacity: f32, centered: bool },
    /// Bitmap/image brush preserving its byte or float source format, device mapping, sampler
    /// modes, opacity, and realized minification textures. The texel fetch reuses WARP's sampler
    /// (`crate::sampling`), shared with 1E's `DrawBitmap`.
    Bitmap {
        bitmap: crate::bitmap::Bitmap,
        to_source: [f32; 6],
        transform: [f32; 6],
        mips: Vec<(u32, u32, Vec<u8>)>,
        float_mips: Vec<(u32, u32, Vec<[f32; 4]>)>,
        mip_origin: [i32; 2],
        mip_size: [u32; 2],
        /// Size of the texture native samples (AlphaBlt normalizes its texcoord plane by it): the
        /// bitmap's own, or the power-of-two atlas an image brush is realized into.
        texture_size: [u32; 2],
        /// Drawn by a pixel shader from a cells atlas (a bitmap image brush), not by AlphaBlt.
        shader_draw: bool,
        /// Which `Create*Brush` produced this brush — governs the non-finite-matrix void rule.
        origin: BrushOrigin,
        /// An image brush over a cropped bitmap: the texels of `bitmap` (its atlas) that hold the source bitmap, the rest
        /// being the crop's transparent padding (see `DeviceContext::brush_samples_padding_only`).
        content: Option<[i32; 4]>,
        /// The device→normalized-texcoord row the vertex shader evaluates, when the draw built it itself (a sampled
        /// DrawBitmap: d2d1's source→dest matrix through the bitmap brush's row); `None`: `invert(transform·size)`.
        vs_row: Option<[f32; 6]>,
        /// An image brush's border cells in the frame's 512x512 atlas: the atlas texel that holds `bitmap`'s texel (0, 0).
        /// `vs_row` then addresses the atlas.
        cell: Option<[i32; 2]>,
        /// With `cell`: the tile texel → device matrix whose adjugate the CUBIC pixel shader phases its taps by, and the
        /// device quad the fill draws, over whose vertices its vertex shader forms the texcoord through that matrix.
        cell_quad: Option<([f32; 6], [f32; 4])>,
        /// The side of the square texture `cell` addresses: the frame's 512 atlas, or past 25 tiles the need's own
        /// power-of-two intermediate.
        cell_size: u32,
        /// A sub-rect bitmap brush (d2d1 draws an image brush over a whole-pixel sub-rect of its bitmap as one): the
        /// normalized texel-centre box `[l, t, r, b]` the shader clamps its texcoord to.
        clamp: Option<[f32; 4]>,
        intermediate_precision: u32,
        hq: Option<BitmapHq>,
        extend: [Extend; 2],
        filter: Filter,
        interpolation: u32,
        opacity: f32,
        /// Read as the text kernels read a filtered brush: whole-byte texels of the brush at opacity 1, the opacity
        /// applied after ([`Brush::as_text_texels`]).
        text_texels: bool,
        /// A text outline fill's intermediate: the brush drawn over the fill's whole-pixel quad, as [`Brush::text_texels`]
        /// draws one for a glyph texture ([`Brush::with_text_quad`]).
        text_quad: Option<Box<TextQuad>>,
    },
    /// A radial-realized 1-D ramp with no geometry: the caller supplies the parameter `t∈[0,1]` (the
    /// COLR conical two-circle solve or the sweep angle), sampled at `u = t·u1` — native realizes the
    /// conical/sweep ramp with the RADIAL `CalcTextureSize` (trailing-border, `radial_width`), NOT the
    /// linear `view_width`.
    Ramp { grad: Gradient },
}

/// Premultiply a straight RGBA colour (float 0..1) to BGRA float 0..255 — for a solid brush the
/// quantisation happens at the blend/store, so this stays in float.
pub(crate) fn premul_f(c: [f32; 4], opacity: f32) -> [f32; 4] {
    let o = opacity.clamp(0.0, 1.0);
    let a = c[3].clamp(0.0, 1.0);
    [c[2] * a * 255.0 * o, c[1] * a * 255.0 * o, c[0] * a * 255.0 * o, a * 255.0 * o]
}

/// Native's linear-gradient device→texel texcoord transform, byte-for-byte. Two d2d1 steps on the
/// device-space endpoints `start`/`end` and the realized texel span `[u0, u1]`:
///   1. `GetRealizationPlacementData` (d2d1+0x13aa40) renormalizes the gradient axis through the
///      perpendicular's length (a `sqrt`), nudging `end` so the axis is a hair longer. For a
///      perfect-square `|axis|²` (axis-aligned gradients) the sqrt is exact and this is a no-op; for an
///      arbitrary angle it shifts the axis a ≤1-ULP — which native then carries into `dir`, so EVERY
///      non-axis-aligned linear gradient (CSS `linear-gradient` at an angle) gets it. Our old per-pixel
///      `dot(p−start,d)/|d|²` projection skipped it (a cross-platform fingerprint on angled gradients).
///   2. the texture-mapping: `det = dy²+dx²`, the per-span scale `(u1−u0)/det`, `dir = d·scale`,
///      `offset = u0 − (dy·start.y + dx·start.x)·scale` — the texcoord is `u = dir_x·px + dir_y·py + offset`
///      (texel space; native's later ×1/W and the sampler's ×W cancel for a power-of-two W).
/// Returns `[dir_x, dir_y, offset]`. (`sqrt` is algebraic — stays out of `crate::crt`, matching native's
/// `fsqrt`.) Verified byte-exact vs native dir on diagonal `0x3c27fffe` and horizontal `0x3ca55555`.
pub(crate) fn linear_transform(start: [f32; 2], end: [f32; 2], u0: f32, u1: f32) -> [f32; 3] {
    let (p0x, p0y) = (start[0], start[1]);
    let (mut p1x, mut p1y) = (end[0], end[1]);
    // GetRealizationPlacementData's perpendicular-length renormalization.
    let ny = -(p1y - p0y);
    let nx = -(p1x - p0x);
    let nlen = (nx * nx + ny * ny).sqrt();
    if nlen != 0.0 && !nlen.is_nan() {
        let inv = 1.0 / nlen;
        let uy = ny * inv;
        let ux = nx * inv;
        let proj = (p1y - p0y) * uy + (p1x - p0x) * ux;
        let qy = proj * uy;
        let qx = proj * ux;
        let qlen = (qy * qy + qx * qx).sqrt();
        if qlen != 0.0 && !qlen.is_nan() {
            p1x = qx + p0x;
            p1y = qy + p0y;
        }
    }
    linear_mapping([p0x, p0y], [p1x, p1y], u0, u1)
}

/// the texture-mapping alone, for device endpoints `GetRealizationPlacementData` already
/// re-projected ([`Brush::linear`]): running [`linear_transform`]'s renormalization on top shifts the
/// texcoord row ≤2 ulp off native's (VS cbuffer of r9 14434 at dpi 192 and r5 5379 at dpi 120).
fn linear_mapping(start: [f32; 2], end: [f32; 2], u0: f32, u1: f32) -> [f32; 3] {
    let ([p0x, p0y], [p1x, p1y]) = (start, end);
    // CalculateTextureMapping (det = dy²+dx², one reciprocal span/det). Branch structure mirrors native's
    // the texture-mapping path exactly: the `u1−u0 != 0` main branch sets the degenerate flag (⇒ zeroed
    // transform, `u = 0` everywhere) when `det == 0 || isnan(det)`; the `u1==u0` branch (W==2 clamp) uses
    // a unit direction and zeroes on `sqrt(det) == 0 || isnan`. A NON-FINITE gradient endpoint is stored by
    // Direct2D as NaN (a non-finite endpoint stores NaN), so
    // `det` comes out NaN here and we take the degenerate `[0,0,0]` wash — NOT the `[NaN,…]` that a raw inf
    // endpoint (det=inf, which native never sees) would give. `det==inf` from a huge-but-FINITE endpoint
    // stays non-degenerate, matching Direct2D (the degenerate flag keys on `isnan`, not `isinf`).
    let dy = p1y - p0y;
    let dx = p1x - p0x;
    let det = dy * dy + dx * dx;
    if u1 - u0 != 0.0 {
        if det != 0.0 && !det.is_nan() {
            let scale = (u1 - u0) / det;
            [dx * scale, dy * scale, u0 - (dy * p0y + dx * p0x) * scale]
        } else {
            [0.0, 0.0, 0.0] // degenerate axis wash (samples texel 0 of the realization)
        }
    } else {
        // Degenerate SPAN (`u0 == u1` — a W==2 clamp texture: two border texels, no interior). Native's
        // the degenerate-span branch builds a UNIT direction `(dx,dy)/|axis|` (`|dir| = 1`, a one-device-pixel ramp)
        // with `offset = u0 − dir·P0`, so a sub-pixel linear Pad-clamps to its endpoints and transitions
        // over exactly one device pixel — NOT scaled by the (sub-pixel) axis length. Verified against
        // Direct2D.s realize texcoords: every W==2 leaf has `|dir| == 1`.
        let axis_len = det.sqrt();
        if axis_len == 0.0 || axis_len.is_nan() {
            return [0.0, 0.0, 0.0];
        }
        let inv = 1.0 / axis_len; // 1/|axis|
        let (dir_x, dir_y) = (dx * inv, dy * inv);
        [dir_x, dir_y, u0 - (dir_y * p0y + dir_x * p0x)]
    }
}

impl Brush {
    /// A radial-realized ramp for a caller-parameterised gradient (COLR conical / sweep). `footprint`
    /// is the device-space extent (outer-circle radius / sweep radius); the ramp uses the radial
    /// `radial_width` + trailing-border span, matching native's gradient-texture realize for these.
    pub fn ramp(stops: &[(f32, [f32; 4])], gamma: u32, extend: u32, footprint: f32) -> Brush {
        Brush::Ramp { grad: Gradient::new(stops, gamma, extend, footprint, true) }
    }

    /// Sample a [`Brush::Ramp`] at `t∈[0,1]` the way native's radial sampler does: `u = t·u1`
    /// (`u1 = W−1` clamp / `W` wrap-mirror), through the full-float `sample_f`. Premultiplied BGRA 0..1.
    pub fn sample_ramp_f(&self, t: f32) -> [f32; 4] {
        match self {
            Brush::Ramp { grad } => grad.sample_f(t * grad.u1),
            _ => [0.0; 4],
        }
    }

    /// Sample a [`Brush::Ramp`] at `u = t·W` (the full realized width, not `u1 = W−1`): native's COLR
    /// **conical** colorizer scales `t` by the texture width like the plain radial brush, a ≤1-LSB
    /// difference from the `t·u1` the sweep colorizer uses (colorfont2-04: 384→8 bytes). Sweep keeps
    /// [`Self::sample_ramp_f`].
    pub fn sample_ramp_f_w(&self, t: f32) -> [f32; 4] {
        match self {
            Brush::Ramp { grad } => grad.sample_f(t * grad.width as f32),
            _ => [0.0; 4],
        }
    }

    /// The paint-tree text shader samples normalized ramp coordinates, retaining Q8-filtered
    /// colour in float until its gamma mask and output-merger store.
    pub(crate) fn sample_ramp_text(&self, t: f32) -> [f32; 4] {
        match self {
            Brush::Ramp { grad } => grad.sample_text(t * grad.width as f32),
            _ => [0.0; 4],
        }
    }

    /// The brush a geometry or rect fill actually samples for a linear gradient whose texcoord has a NaN
    /// y DERIVATIVE: WARP's LOD takes `maxss(|du/dx|, |du/dy|)` (its second operand on NaN), so a NaN
    /// du/dy gives a NaN LOD and the sampler reads the bound chain's coarsest, 1×1 level — the stop
    /// midpoint for clamp, the whole-ramp average for wrap/mirror (VM ladder fx7k/nanlod; r12 17124,
    /// whose device start.y of +inf makes the texture-mapping `[0, NaN, NaN]`). A NaN x derivative
    /// alone (`[NaN, -0, NaN]`, an end.x of 3.4e38: fuzz 6095/17139/17404) or a NaN offset alone
    /// (grad-degen-265) keeps level 0. Text and opacity-mask draws sample without it (17616, 5022).
    pub(crate) fn nan_lod(&self) -> Option<Brush> {
        match self {
            // WARP's LOD = maxss(|du/dx|, |du/dy|) reads the coarsest 1×1 ONLY when du/dy (`xform[1]`) is
            // NaN while du/dx (`xform[0]`) stays FINITE (r12 17124's `[0, NaN, NaN]`). A NaN du/dx as well
            // (`[NaN, NaN, NaN]`: an inf END under a skew, 17267) keeps level 0, so the full-width ramp's
            // u≡NaN folds to texel 0 — the first stop — not the whole-ramp average.
            Brush::Linear { grad, start, end, opacity, xform } if xform[1].is_nan() && xform[0].is_finite() => {
                let g = Gradient { texels: grad.coarsest.to_vec(), width: 1, extend: grad.extend, u0: 0.0, u1: 1.0, coarsest: grad.coarsest };
                Some(Brush::Linear { grad: g, start: *start, end: *end, opacity: *opacity, xform: [0.0; 3] })
            }
            _ => None,
        }
    }

    pub fn solid(color: [f32; 4], opacity: f32) -> Brush {
        Brush::Solid(premul_f(color, opacity))
    }

    /// This brush with its device→texel mapping pre-translated by `(dx, dy)` device pixels, so sampling
    /// at `(x, y)` now returns what the original returned at `(x−dx, y−dy)`. Used when a recorded brush
    /// fill replays at a command-list offset: the gradient/bitmap pattern must move with the geometry,
    /// as native's translated replay does. Only the translation terms of the precomputed device→texel
    /// transforms shift (see [`Brush::source_at`]); a solid or 1-D ramp brush is unaffected. [G6]
    pub(crate) fn translated(&self, dx: f32, dy: f32) -> Brush {
        let mut b = self.clone();
        match &mut b {
            Brush::Solid(_) | Brush::Ramp { .. } => {}
            Brush::Linear { xform, .. } => xform[2] -= xform[0] * dx + xform[1] * dy,
            Brush::Radial { to_unit, .. } => {
                to_unit[4] -= to_unit[0] * dx + to_unit[2] * dy;
                to_unit[5] -= to_unit[1] * dx + to_unit[3] * dy;
            }
            Brush::Bitmap { to_source, .. } => {
                to_source[4] -= to_source[0] * dx + to_source[2] * dy;
                to_source[5] -= to_source[1] * dx + to_source[3] * dy;
            }
        }
        b
    }

    /// Whether every pixel this brush produces is fully opaque (α ≡ 1): the gate for draw paths whose
    /// native fast path assumes an opaque source (the ink ISM gamma LUT). A gradient is opaque when
    /// its whole ramp is opaque and brush opacity is 1; a bitmap/image brush when every texel is and
    /// opacity is 1. Premultiplied texels/colours store α in the 4th byte/lane.
    pub fn is_opaque(&self) -> bool {
        match self {
            Brush::Solid(p) => p[3] >= 1.0,
            Brush::Linear { grad, opacity, .. } | Brush::Radial { grad, opacity, .. } => {
                *opacity >= 1.0 && grad.texels.chunks_exact(4).all(|t| t[3] == 255)
            }
            Brush::Bitmap { bitmap, opacity, .. } => *opacity >= 1.0 && !crate::float::is_highp(bitmap.format) && bitmap.premul_bgra.chunks_exact(4).all(|t| t[3] == 255),
            // The COLR-leaf 1-D ramp (`Brush::ramp`): opaque iff every realized texel is.
            Brush::Ramp { grad } => grad.texels.chunks_exact(4).all(|t| t[3] == 255),
        }
    }

    /// Linear gradient from brush-space `start`/`end` and the `brush→device` matrix `b2d`.
    #[allow(clippy::too_many_arguments)]
    pub fn linear(stops: &[(f32, [f32; 4])], gamma: u32, extend: u32, opacity: f32, start: [f32; 2], end: [f32; 2], b2d: &[f32; 6]) -> Brush {
        // d2d1 `LinearGradientBrushDataWrapper::GetRealizationPlacementData`: p0 = start_dev, and the
        // device axis end p1 = p0 + L·(end − start) — the linear part L of `b2d` applied to the brush
        // axis vector and ADDED to the device origin (native adds the device axis vector to the device start), NOT
        // L·end + translation. For finite endpoints the two are byte-equal; for a non-finite endpoint
        // the origin-relative add is what native computes, and its ±inf/NaN propagation picks the
        // realization width (first stop vs midpoint wash):
        //  • huge START (device +inf), finite end → vec1_dev.y = −inf → p1.y = (+inf)+(−inf) = NaN →
        //    NaN footprint → width 1 → midpoint wash (r13 18830 start.y=3.4e38).
        //  • inf END under a SKEW → nonzero off-diagonal, m·inf = ±inf (no cancellation) → p1 = ±inf →
        //    inf footprint → width 1024 → the fill maps to u≈0 = the FIRST stop (17267 skew).
        //  • inf END under a pure SCALE → zero off-diagonal, 0·inf = NaN → p1 = NaN → width 1 → midpoint
        //    (17267 scale). The perp-isoline below routes the NaN cases through `len.is_nan()`.
        // The endpoint is kept as-is (native SetStart/EndPoint store the raw point; no inf→NaN sanitise),
        // then the end is re-projected onto the device normal of the perpendicular isoline `start +
        // rot90(end − start)` (a non-uniform transform does not keep them perpendicular).
        let mut s = apply(b2d, start[0], start[1]);
        let e_raw = apply(b2d, end[0], end[1]);
        // For finite device endpoints native's origin-relative `p1 = start_dev + L·(end−start)` equals
        // `apply(b2d,end)` to the last bit, so use that form (what the suite ratchet encodes). The
        // origin-relative add only MATTERS once a device coordinate is non-finite — then its inf/NaN
        // cancellation picks the degenerate width: 18830's start→device +inf with L·(end−start).y = −inf
        // cancels to NaN (width 1, midpoint), while 17267's inf end under a skew gives ±inf (width 1024,
        // first stop) and under a pure scale gives 0·inf = NaN (width 1, midpoint).
        let v1 = [end[0] - start[0], end[1] - start[1]];
        let mut e = if s.iter().all(|v| v.is_finite()) && e_raw.iter().all(|v| v.is_finite()) {
            e_raw
        } else {
            [s[0] + (b2d[0] * v1[0] + b2d[2] * v1[1]), s[1] + (b2d[1] * v1[0] + b2d[3] * v1[1])]
        };
        let perp = apply(b2d, start[0] - (end[1] - start[1]), (end[0] - start[0]) + start[1]);
        let (nx, ny) = (perp[0] - s[0], -(perp[1] - s[1]));
        let len = (ny * ny + nx * nx).sqrt();
        if len != 0.0 && !len.is_nan() {
            let (a, b) = (nx * (1.0 / len), ny * (1.0 / len));
            let t = (e[1] - s[1]) * a + (e[0] - s[0]) * b;
            let (dy, dx) = (t * a, t * b);
            let l2 = (dy * dy + dx * dx).sqrt();
            if l2 != 0.0 && !l2.is_nan() {
                e = [dx + s[0], dy + s[1]];
            }
        } else if len.is_nan() || (len == 0.0 && b2d[0] * b2d[3] - b2d[1] * b2d[2] == 0.0) {
            // A perp-isoline normal that cancels to NaN (an endpoint whose device coord OVERFLOWS to ±inf,
            // so `perp − s` is inf − inf) is native's degenerate placement: GetRealizationPlacementData
            // yields a NaN start, GetRealizationSize `fmaxf(NaN,0)=0` → realization width 1, a flat
            // midpoint-of-stops wash (fuzz r13 18830 start.y=3.4e38 → device +inf → NaN normal; native
            // REALIZEINFO width=1, texcoord NaN). A finite-but-huge END endpoint (grad-degen-265 end.y=
            // 3.4e38) instead gives len=+inf (not NaN) → the branch above runs and width stays 1024, as
            // native. Propagate the degeneracy to a NaN footprint/texcoord by NaN-ing the endpoints.
            // A SINGULAR `brush→device` (det 0) collapses the perpendicular to the start (`len == 0`) with
            // finite endpoints: native's placement is just as degenerate, so it takes the same flat
            // midpoint-of-stops wash — the collapsed aliased fill then paints that `t = ½` colour (probe
            // 7800232). A zero-length gradient axis under a non-singular transform keeps its own path.
            s = [f32::NAN; 2];
            e = [f32::NAN; 2];
        }
        Self::linear_mapped(stops, gamma, extend, opacity, s, e, linear_mapping)
    }

    /// Linear gradient from endpoints already in **device space** (`s`/`e`), for callers that must
    /// compute the COLR p2 rotation projection in device space before realizing — native projects the
    /// COLRv1 `PaintLinearGradient` points after the leaf transform, so a non-uniform leaf transform
    /// (which does not preserve perpendicularity) must be applied before the projection. See [`linear`].
    pub fn linear_device(stops: &[(f32, [f32; 4])], gamma: u32, extend: u32, opacity: f32, s: [f32; 2], e: [f32; 2]) -> Brush {
        Self::linear_mapped(stops, gamma, extend, opacity, s, e, linear_transform)
    }

    fn linear_mapped(stops: &[(f32, [f32; 4])], gamma: u32, extend: u32, opacity: f32, s: [f32; 2], e: [f32; 2],
        mapping: fn([f32; 2], [f32; 2], f32, f32) -> [f32; 3]) -> Brush {
        let footprint = ((e[0] - s[0]).powi(2) + (e[1] - s[1]).powi(2)).sqrt();
        let grad = Gradient::new(stops, gamma, extend, footprint, false);
        let xform = mapping(s, e, grad.u0, grad.u1);
        Brush::Linear { grad, start: s, end: e, opacity, xform }
    }

    /// Radial gradient from brush-space ellipse (`center`, `radii`, focal `offset`) and `b2d`.
    ///
    /// Reproduces Direct2D's precomputed realization: the placement data
    /// maps the ellipse centre, its +X/+Y endpoints and the focal point
    /// (`centre+offset`) into device space; `GetRealizationSize` takes the footprint as the max of the
    /// device X-radius, Y-radius and focal distance; the texture-mapping builds the device→unit
    /// map as `invert(rect{0,0,W,W}→parallelogram{C,X,Y})`, which `SetBrushAndContextInternal` scales
    /// by `1/W`. The focal is that map applied to the device focal. (Native keeps the texture's
    /// `span_right/W = (W−1)/W` scaling in the span-scale term; here it is folded into the clamp texcoord span
    /// `[0, W−1]` — see `Gradient::new` — which leaves the coordinate affine as powers-of-two-clean.)
    #[allow(clippy::too_many_arguments)]
    pub fn radial(stops: &[(f32, [f32; 4])], gamma: u32, extend: u32, opacity: f32, center: [f32; 2], radii: [f32; 2], offset: [f32; 2], b2d: &[f32; 6]) -> Brush {
        Self::radial_split(stops, gamma, extend, opacity, center, radii, offset, b2d, None)
    }

    /// [`Brush::radial`] for a brush drawn under the world→device matrix `device` with its own brush
    /// transform `xform` (`b2d = device · xform`): the focal constant is then formed as native's colour
    /// source does, the world-space focal through the world→texture map (see `radial_split`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn radial_world(stops: &[(f32, [f32; 4])], gamma: u32, extend: u32, opacity: f32, center: [f32; 2], radii: [f32; 2], offset: [f32; 2], b2d: &[f32; 6], device: &[f32; 6], xform: &[f32; 6]) -> Brush {
        Self::radial_split(stops, gamma, extend, opacity, center, radii, offset, b2d, Some((device, xform)))
    }

    #[allow(clippy::too_many_arguments)]
    fn radial_split(stops: &[(f32, [f32; 4])], gamma: u32, extend: u32, opacity: f32, center: [f32; 2], radii: [f32; 2], offset: [f32; 2], b2d: &[f32; 6], split: Option<(&[f32; 6], &[f32; 6])>) -> Brush {
        let c = apply(b2d, center[0], center[1]);
        let xe = apply(b2d, center[0] + radii[0], center[1]);
        let ye = apply(b2d, center[0], center[1] + radii[1]);
        let f = apply(b2d, center[0] + offset[0], center[1] + offset[1]);
        let dist = |p: [f32; 2]| {
            let (dx, dy) = (c[0] - p[0], c[1] - p[1]);
            (dx * dx + dy * dy).sqrt()
        };
        // `GetRealizationSize`: a NaN |C−X| stays NaN, then two MAXSS (each selecting its second operand
        // on NaN) with |C−Y| and |C−F|; a NaN footprint is one texel (r10 15233: an infinite x radius
        // makes X's y `−inf·0`).
        let maxss = |a: f32, b: f32| if a > b { a } else { b };
        let (dx, dy, df) = (dist(xe), dist(ye), dist(f));
        let footprint = if dx.is_nan() { dx } else { let r = maxss(dx, dy); if r.is_nan() { r } else { maxss(r, df) } };
        let w = radial_width(footprint, extend) as f32;
        // the texture-mapping = invert(SetRectToParallelogram(rect{0,0,span,span}, {C,X,Y})) with
        // span = the texture's `span_right` — `W−1` for clamp (trailing border texel), `W` for wrap/mirror
        // (no border) — basis `(span·P − span·C)·(1/span²)` and translation `(span²·C)·(1/span²)` (= C up
        // to the native 1-ULP rounding, span not a power of two), then `SetBrushAndContextInternal` scales
        // by 1/W. Matches native's rect-to-parallelogram transform + inverse (d2d1, f32) op-for-op
        // and `ref/radial5-dump`'s captured M; `span` tracks `Gradient::new`'s `u1`.
        let span = if extend == 0 && w > 1.0 { w - 1.0 } else { w };
        let rp = rect_to_parallelogram([0.0, 0.0, span, span], [c, xe, ye]);
        let iw = 1.0 / w;
        // The 3×2 matrix inverse fails on a zero determinant or a non-finite reciprocal (a zero or 1e-20
        // radius, a centre so large that adding the radius is lost); the realization then collapses to
        // ONE texel, so every pixel reads it: the last stop under clamp (a lone radial texel), the whole
        // box-filtered ramp under wrap/mirror (rs-singx/e009b/r3-ext* probes).
        let det = rp[0] * rp[3] - rp[1] * rp[2];
        let unmapped = det == 0.0 || !(1.0 / det).is_finite();
        let to_unit: Mat = if unmapped { [0.0; 6] } else { invert(&rp).map(|v| v * iw) };
        let footprint = if unmapped { 0.0 } else { footprint };
        // WARP's VS emits the per-vertex texcoord `o1 = M·v`, which `SetupTriangleInternal2` fits into a
        // screen-space plane over the fill's one unit triangle — dropping the diagonal a few ULP
        // (`warp_fit_affine`, trace-confirmed `ref/radial5-dump`). That fitted plane drives the per-pixel
        // `pc`. The focal is a *constant* VS output (the uploaded realize `M·focal_dev`, plain mul+add),
        // so it keeps the unfitted map. Both verified against the captured plane/focal/readback.
        // With the world matrix known, the focal follows the colour source: the brush-space `center +
        // offset` through the brush transform alone (world space), then through the world→texture map
        // `device · M`, each product in native's operation order. At a far focal the large translation
        // cancels, so this order, not `M · focal_device`, sets its last bits (r20 25950, dpi 120).
        let focal = match split {
            Some((d, bx)) if !unmapped => {
                let (fx, fy) = (center[0] + offset[0], center[1] + offset[1]);
                let fw = [(fy * bx[2] + fx * bx[0]) + bx[4], (fy * bx[3] + fx * bx[1]) + bx[5]];
                let m = &to_unit;
                let w2t = [
                    (m[2] * d[1]) + (m[0] * d[0]),
                    (m[3] * d[1]) + (m[1] * d[0]),
                    (m[2] * d[3]) + (m[0] * d[2]),
                    (m[3] * d[3]) + (m[1] * d[2]),
                    ((m[2] * d[5]) + (m[0] * d[4])) + m[4],
                    ((m[3] * d[5]) + (m[1] * d[4])) + m[5],
                ];
                [(fw[1] * w2t[2] + fw[0] * w2t[0]) + w2t[4], (fw[1] * w2t[3] + fw[0] * w2t[1]) + w2t[5]]
            }
            _ => apply(&to_unit, f[0], f[1]),
        };
        // `to_unit` is the UNIT-triangle plane fit for the per-pixel (source_at) path; the geometry
        // fill refits the RAW device→unit map per rasterized triangle at its provoking vertex
        // (`fill_triangle_brush`), so keep the raw matrix too.
        let to_unit_raw = to_unit;
        let to_unit = crate::warp::plane::warp_fit_affine(&to_unit);
        Brush::Radial { grad: Gradient::new(stops, gamma, extend, footprint, true), to_unit, to_unit_raw, focal, opacity, centered: offset == [0.0; 2] }
    }

    /// Bitmap/image brush from a format-aware bitmap, its brush→device matrix, extend modes,
    /// D2D interpolation mode, and opacity.
    pub fn bitmap(texels: Vec<u8>, size: [u32; 2], b2d: &[f32; 6], extend: [u32; 2], interpolation: u32, opacity: f32) -> Brush {
        Self::bitmap_source(crate::bitmap::Bitmap {
            width: size[0], height: size[1], premul_bgra: texels,
            format: d2d_types::abi::DXGI_FORMAT_B8G8R8A8_UNORM,
            alpha: d2d_types::abi::D2D1_ALPHA_MODE_PREMULTIPLIED, dpi: (96.0, 96.0),
        }, b2d, extend, interpolation, opacity)
    }

    /// d2d1's exact device→normalized texcoord plane = `invert(diag(W,H) ∘ b2d)`, where `b2d` is the
    /// brush's texel→device matrix (`transform`, set by `bitmap_placement` for every draw context: world
    /// transform, unit mode, layers). The dim fold is inside the source scale BEFORE the inverse, matching
    /// WARP's case-7 composition (byte-for-byte on the obs6 `[rax+0x558]+0x30` REFINED coefficients). Derived
    /// from the CURRENT `transform` each sample, so a re-placed draw never reads a stale plane (r27 32580).
    pub(crate) fn plane_of_b2d(b2d: &[f32; 6], w: u32, h: u32) -> [f32; 6] {
        invert(&compose_f32(b2d, &[w as f32, 0.0, 0.0, h as f32, 0.0, 0.0]))
    }

    pub(crate) fn bitmap_source(bitmap: crate::bitmap::Bitmap, b2d: &[f32; 6], extend: [u32; 2], interpolation: u32, opacity: f32) -> Brush {
        let padded = extend == [0, 0];
        // interp 5 (HQ cubic) also needs a mip chain for EXTREME minification: a direct 2/scale-tap
        // downscale is unbounded (fuzz r24 29309/29494 1e-6 → 2e6 taps → hang), so the extreme-min
        // sampler cascades to a mip level instead (bounded), as native does.
        // An A8 bitmap has one byte a texel and is rejected where the brush draws: it gets no mip chain (r24 29421).
        let want_mips = (interpolation == 4 || interpolation == 5) && bitmap.format.0 != 65;
        let mips = if want_mips && !crate::float::is_highp(bitmap.format) {
            let tex = Texture { data: &bitmap.premul_bgra, width: bitmap.width, height: bitmap.height };
            if padded { crate::bitmap::clamped_mip_chain(&tex) } else { crate::bitmap::mip_chain(&tex) }
        } else { Vec::new() };
        let float_mips = if want_mips && crate::float::is_highp(bitmap.format) {
            crate::bitmap::float_mip_chain(&bitmap, padded)
        } else { Vec::new() };
        let mip_size = [bitmap.width, bitmap.height];
        Brush::Bitmap {
            intermediate_precision: if crate::float::is_highp(bitmap.format) { 5 } else { 1 },
            bitmap,
            to_source: invert(b2d),
            transform: *b2d,
            mips,
            float_mips,
            mip_origin: [0; 2],
            mip_size,
            texture_size: mip_size,
            shader_draw: false,
            origin: BrushOrigin::PlainBitmap,
            content: None,
            vs_row: None,
            cell: None,
            cell_quad: None,
            cell_size: 512,
            clamp: None,
            hq: None,
            extend: [extend_of(extend[0]), extend_of(extend[1])],
            filter: crate::bitmap::filter_of(interpolation),
            interpolation,
            opacity,
            text_texels: false,
            text_quad: None,
        }
    }

    /// A bitmap/image brush whose brush→device matrix doesn't invert in f32: native draws nothing with
    /// it (singular when `det == 0` or `1/det` is non-finite), so a
    /// NaN matrix from a `-inf` image-brush source rect is singular too (r11 16043). A **non-finite
    /// translation** (dx/dy = ±inf/NaN) also voids the brush: native folds the whole matrix into that
    /// determinant and multiplies the translation by `0.0` on the way (multiplying the x delta by 0), so
    /// an inf dx becomes `inf·0 = NaN`, poisons the determinant, and trips `!_finite` (r13 18020, an
    /// image brush with `dx = inf` — native draws nothing, we sampled a wrapped texel). So require every
    /// matrix entry finite, not just an invertible 2×2.
    pub(crate) fn singular(&self) -> bool {
        let Brush::Bitmap { transform: m, origin, .. } = self else { return false };
        let det = m[0] * m[3] - m[1] * m[2];
        // The 2×2 must invert finitely whatever the origin. A det that overflows to ±inf (a near-FLT_MAX
        // scale — r19 24160) voids too, which `(1/det).is_finite()` alone misses since `1/inf = 0`.
        let core_ok = [m[0], m[1], m[2], m[3]].iter().all(|v| v.is_finite())
            && det != 0.0 && det.is_finite() && (1.0 / det).is_finite();
        match origin {
            // A plain bitmap brush samples through AlphaBlt: a non-finite translate only shifts the
            // texcoord (painted as WARP's non-finite fallback), it does NOT void (r17 22870, dx = inf
            // paints the fallback texel, not nothing).
            BrushOrigin::PlainBitmap => !core_ok,
            // An image brush realizes through the full matrix, so a non-finite translate poisons its
            // determinant and voids the draw (r13 18020/18740, r14 19147, r15 20842).
            BrushOrigin::BitmapImage | BrushOrigin::StagedImage | BrushOrigin::OtherImage => !(core_ok && [m[4], m[5]].iter().all(|v| v.is_finite())),
        }
    }

    /// Place a bitmap brush through the brush→device matrix `m` (a draw's placement); no-op on other brushes.
    /// A new brush→device matrix drops the VS row taken for the old one (a layer pop's opacity brush re-placed under the
    /// pop's unit mode samples its own placement; r27 32291, 32580).
    pub(crate) fn set_placement(&mut self, m: [f32; 6]) {
        if let Brush::Bitmap { transform, to_source, vs_row, .. } = self {
            *transform = m;
            *to_source = invert(&m);
            *vs_row = None;
        }
    }

    /// Mark where a realized bitmap brush came from (see [`BrushOrigin`]); no-op on other brushes.
    pub(crate) fn set_origin(&mut self, o: BrushOrigin) {
        if let Brush::Bitmap { origin, .. } = self {
            *origin = o;
        }
    }

    /// Set the vertex shader's device→normalized-texcoord row a bitmap brush samples through (see `vs_row`).
    pub(crate) fn set_vs_row(&mut self, row: [f32; 6]) {
        if let Brush::Bitmap { vs_row, .. } = self { *vs_row = Some(row); }
    }

    /// Places a prepared HIGH_QUALITY_CUBIC brush's intermediate at slot `at` of the frame's atlas.
    pub(crate) fn set_hq_slot(&mut self, at: [i32; 2]) {
        if let Brush::Bitmap { hq: Some(c), .. } = self { c.slot = at; }
    }

    /// Samples the brush from its border cells at atlas texel `at` through the atlas row `row`.
    pub(crate) fn set_clamp(&mut self, c: [f32; 4]) {
        if let Brush::Bitmap { clamp, .. } = self { *clamp = Some(c); }
    }

    /// Keep the border cells' CUBIC quad only for a fill native draws as that quad: a whole-pixel, axis-aligned rect at
    /// full opacity (VM DDI: an integer rect fill is one 6-vertex quad, the rect cut to the target, through the cubic
    /// shader; a fractional one draws no such quad, and keeps the atlas row's plane, exact on geo/c703q 9703300/03/04;
    /// below full opacity the rect takes the walk's or the row's planes, geo/c703 9703009/10 41 and 7 px to 0).
    pub(crate) fn keep_cell_quad(&mut self, keep: bool) {
        if let (false, Brush::Bitmap { cell_quad, .. }) = (keep, self) { *cell_quad = None; }
    }

    pub(crate) fn set_atlas_cell(&mut self, row: [f32; 6], at: [i32; 2], size: u32, quad: ([f32; 6], [f32; 4])) {
        if let Brush::Bitmap { vs_row, cell, cell_quad, cell_size, .. } = self { (*vs_row, *cell, *cell_size, *cell_quad) = (Some(row), Some(at), size, Some(quad)); }
    }

    pub(crate) fn set_intermediate_precision(&mut self, precision: u32) {
        if precision == 0 { return; }
        if let Brush::Bitmap { bitmap, to_source, intermediate_precision, interpolation, mips, float_mips, extend, .. } = self {
            if *intermediate_precision == precision { return; }
            *intermediate_precision = precision;
            let minifying = (to_source[0] * to_source[0] + to_source[1] * to_source[1]).max(to_source[2] * to_source[2] + to_source[3] * to_source[3]) > 1.0;
            if *interpolation == 4 && minifying {
                let padded = *extend == [Extend::Clamp; 2];
                if precision >= 2 {
                    *float_mips = crate::bitmap::float_mip_chain_precision(bitmap, padded, precision);
                    mips.clear();
                } else {
                    let mut bytes = Vec::with_capacity(bitmap.width as usize * bitmap.height as usize * 4);
                    for y in 0..bitmap.height { for x in 0..bitmap.width {
                        let p = bitmap.texel_rgba(x as i32, y as i32).map(|v| (v * 255.0).round_ties_even().clamp(0.0, 255.0) as u8);
                        bytes.extend_from_slice(&[p[2], p[1], p[0], p[3]]);
                    }}
                    let tex = Texture { data: &bytes, width: bitmap.width, height: bitmap.height };
                    *mips = if padded { crate::bitmap::clamped_mip_chain(&tex) } else { crate::bitmap::mip_chain(&tex) };
                    float_mips.clear();
                }
            }
        }
    }

    fn aniso_bounds(&self, rect: [f32; 4]) -> Option<[i32; 4]> {
        let Brush::Bitmap { transform, to_source, extend, interpolation: 4, .. } = self else { return None };
        if !extend.iter().any(|e| matches!(e, Extend::Wrap | Extend::Mirror)) { return None; }
        let coverage = crate::warpmath::sqrt((to_source[0] * to_source[0] + to_source[1] * to_source[1]).max(to_source[2] * to_source[2] + to_source[3] * to_source[3]));
        if coverage <= 1.0 { return None; }
        let mut grid = 1i32;
        while (grid as f32) < coverage && grid < 128 { grid *= 2; }
        let det = transform[0] * transform[3] - transform[1] * transform[2];
        let points = [[rect[0], rect[1]], [rect[2], rect[1]], [rect[2], rect[3]], [rect[0], rect[3]]].map(|p| [
            ((transform[3] * p[0] - transform[2] * p[1]) + (transform[2] * transform[5] - transform[3] * transform[4])) / det,
            ((-transform[1] * p[0] + transform[0] * p[1]) + (transform[1] * transform[4] - transform[0] * transform[5])) / det,
        ]);
        let mut bounds = [0; 4];
        for c in 0..2 {
            let lo = points.iter().map(|p| p[c]).fold(f32::INFINITY, f32::min).floor() as i32;
            let hi = points.iter().map(|p| p[c]).fold(f32::NEG_INFINITY, f32::max).ceil() as i32;
            bounds[c] = lo - lo.rem_euclid(grid);
            bounds[c + 2] = hi + (-hi).rem_euclid(grid);
            if lo - bounds[c] <= grid / 2 { bounds[c] -= grid; }
            if bounds[c + 2] - hi <= grid / 2 { bounds[c + 2] += grid; }
        }
        // An extreme minification (e.g. a 1e-6 downscale) maps the device rect onto an enormous source
        // extent, which would subdivide into millions of realization tiles — an effective hang (fuzz r21
        // 26999). Native does not realize that extent; the anisotropic probe samples a coarse mip, so a
        // realization past what a bounded atlas can hold falls back to the per-pixel mip sampler (which is
        // itself bounded). Verified byte-exact against native's own render of 26999.
        if (bounds[2] as i64 - bounds[0] as i64) * (bounds[3] as i64 - bounds[1] as i64) > 1 << 22 {
            return None;
        }
        Some(bounds)
    }

    /// Native d2d1 10.0.26100 splits an oversized anisotropic realization along its longest source
    /// extent, visiting the high half first, until each padded tile fits a 4096-pixel atlas.
    pub(crate) fn split_rect(&self, rect: [f32; 4]) -> Option<[[f32; 4]; 2]> {
        // Extreme minification: a per-tile HQ intermediate stays oversized even at a single device
        // pixel (`1/scale > 4096`), so tiling recurses to sub-pixel tiles — an effective hang (fuzz r24
        // 29309/29494 at 1e-6). Don't tile: the per-pixel mip sampler (the interp-5 min guard in
        // `source_at_rect_impl`) handles it bounded. Matches that guard's `2/scale > 4096` threshold.
        if let Brush::Bitmap { transform, interpolation: 5, .. } = self {
            if 2.0 / transform[0].abs().min(transform[3].abs()) > 4096.0 { return None; }
        }
        let size = if let Some(bounds) = self.hq_request_bounds(rect) {
            [bounds[2] as f64 - bounds[0] as f64, bounds[3] as f64 - bounds[1] as f64]
        } else {
            let bounds = self.aniso_bounds(rect)?;
            [bounds[2] as f64 - bounds[0] as f64, bounds[3] as f64 - bounds[1] as f64]
        };
        if size[0].max(size[1]) <= 4096.0 { return None; }
        let axis = usize::from(size[1] > size[0]);
        let mid = (rect[axis] + rect[axis + 2]) * 0.5;
        let mid = if mid.floor() > rect[axis] { mid.floor() } else { mid };
        if mid <= rect[axis] || mid >= rect[axis + 2] { return None; }
        let (mut high, mut low) = (rect, rect);
        high[axis] = mid;
        low[axis + 2] = mid;
        Some([high, low])
    }

    /// Whether an ANISOTROPIC bitmap brush drawn over the device pixels `region` needs a mip intermediate wider or
    /// taller than 4096 px, which fails the draw with INTERMEDIATE_TOO_LARGE. Direct2D realizes the brush per tile
    /// (`context::halves`). A tile needs the source under its pixel centres, rounded out to the mip grid of
    /// [`aniso_bounds`](Self::aniso_bounds) at the source origin. A side with half a block or less to spare gains a
    /// block. Any extend mode (sc119-sc121: a 16 px tile fails below a 3.9e-3 scale, two 12 px tiles draw).
    pub(crate) fn aniso_need_too_large(&self, region: [i32; 4]) -> bool {
        let Brush::Bitmap { transform: t, to_source, interpolation: 4, .. } = self else { return false };
        let coverage = crate::warpmath::sqrt((to_source[0] * to_source[0] + to_source[1] * to_source[1]).max(to_source[2] * to_source[2] + to_source[3] * to_source[3]));
        if !(coverage > 1.0) {
            return false;
        }
        let mut grid = 1i64;
        while (grid as f32) < coverage && grid < 128 { grid *= 2; }
        let det = t[0] * t[3] - t[1] * t[2];
        let source = |x: f32, y: f32| [
            ((t[3] * x - t[2] * y) + (t[2] * t[5] - t[3] * t[4])) / det,
            ((-t[1] * x + t[0] * y) + (t[1] * t[4] - t[0] * t[5])) / det,
        ];
        let (mut xs, mut ys) = (Vec::new(), Vec::new());
        crate::context::halves(region[0], region[2], &mut xs);
        crate::context::halves(region[1], region[3], &mut ys);
        xs.iter().any(|&(x0, x1)| ys.iter().any(|&(y0, y1)| {
            let (l, r, tp, b) = (x0 as f32 + 0.5, x1 as f32 - 0.5, y0 as f32 + 0.5, y1 as f32 - 0.5);
            let p = [source(l, tp), source(r, tp), source(r, b), source(l, b)];
            (0..2).any(|c| {
                let lo = p.iter().map(|q| q[c]).fold(f32::INFINITY, f32::min).floor() as i64;
                let hi = p.iter().map(|q| q[c]).fold(f32::NEG_INFINITY, f32::max).ceil() as i64;
                let (mut b0, mut b1) = (lo - lo.rem_euclid(grid), hi + (-hi).rem_euclid(grid));
                if lo - b0 <= grid / 2 { b0 -= grid; }
                if b1 - hi <= grid / 2 { b1 += grid; }
                b1 - b0 > 4096
            })
        }))
    }

    pub(crate) fn prepare_aniso_rect(&self, rect: [f32; 4]) -> Option<Brush> {
        let bounds = self.aniso_bounds(rect)?;
        let Brush::Bitmap { bitmap, transform, to_source, extend, filter, opacity, intermediate_precision, origin, content, .. } = self else { return None };
        let dims = [(bounds[2] - bounds[0]) as u32, (bounds[3] - bounds[1]) as u32].map(|v| v.next_power_of_two().max(512));
        let highp = *intermediate_precision >= 2;
        let bpp = if highp { 16 } else { 4 };
        let mut data = vec![0; dims[0] as usize * dims[1] as usize * bpp];
        for y in bounds[1]..bounds[3] {
            for x in bounds[0]..bounds[2] {
                let (sx, sy) = (extend[0].wrap(x, bitmap.width as i32), extend[1].wrap(y, bitmap.height as i32));
                let dst = ((y - bounds[1]) as usize * dims[0] as usize + (x - bounds[0]) as usize) * bpp;
                if highp {
                    let value = bitmap.texel_rgba(sx, sy);
                    let value = crate::effects::quantize_to(*intermediate_precision, value);
                    for (ch, value) in value.iter().enumerate() { data[dst + 4*ch..dst + 4*ch + 4].copy_from_slice(&value.to_le_bytes()); }
                } else if crate::float::is_highp(bitmap.format) {
                    let p = bitmap.texel_rgba(sx, sy).map(|v| (v * 255.0).round_ties_even().clamp(0.0, 255.0) as u8);
                    data[dst..dst + 4].copy_from_slice(&[p[2], p[1], p[0], p[3]]);
                } else {
                    let src = (sy as usize * bitmap.width as usize + sx as usize) * 4;
                    data[dst..dst + 4].copy_from_slice(&bitmap.premul_bgra[src..src + 4]);
                }
            }
        }
        let atlas = crate::bitmap::Bitmap {
            width: dims[0], height: dims[1], premul_bgra: data, dpi: bitmap.dpi, alpha: bitmap.alpha,
            format: if highp { d2d_types::abi::DXGI_FORMAT_R32G32B32A32_FLOAT } else { d2d_types::abi::DXGI_FORMAT_B8G8R8A8_UNORM },
        };
        Some(Brush::Bitmap {
            bitmap: bitmap.clone(), transform: *transform, to_source: *to_source, extend: *extend,
            filter: *filter, interpolation: 4, opacity: *opacity, hq: None, text_texels: false, text_quad: None,
            intermediate_precision: *intermediate_precision,
            mip_origin: [bounds[0], bounds[1]], mip_size: dims, texture_size: [bitmap.width, bitmap.height], shader_draw: false, origin: *origin, content: *content, vs_row: None, cell: None, cell_quad: None, cell_size: 512, clamp: None,
            mips: if highp { Vec::new() } else { crate::bitmap::mip_chain(&Texture { data: &atlas.premul_bgra, width: dims[0], height: dims[1] }) },
            float_mips: if highp { crate::bitmap::float_mip_chain_precision(&atlas, false, *intermediate_precision) } else { Vec::new() },
        })
    }

    /// First intermediate request before integer conversion; used by the context's native error latch.
    pub(crate) fn hq_request_bounds(&self, rect: [f32; 4]) -> Option<[f32; 4]> {
        let Brush::Bitmap { transform, interpolation: 5, .. } = self else { return None; };
        capped_hq_request(transform, rect, true).map(|r| r.3)
    }

    /// A rotated or skewed HIGH_QUALITY_CUBIC bitmap brush's downscaled intermediate for a draw over `rect`.
    pub(crate) fn prepare_rotated(&self, rect: [f32; 4]) -> Option<Brush> {
        let Brush::Bitmap { transform: m, interpolation: 5, .. } = self else { return None; };
        if m[1] == 0.0 && m[2] == 0.0 { return None; }
        self.prepare_rect(rect)
    }

    /// Realize the intermediate texture once for the rectangle's minification passes.
    pub(crate) fn prepare_rect(&self, rect: [f32; 4]) -> Option<Brush> {
        self.prepare_rect_in(rect, true)
    }

    /// [`Self::prepare_rect`] for a geometry fill over `rect`, which takes the residual placement at a whole offset too.
    pub(crate) fn prepare_geometry(&self, rect: [f32; 4]) -> Option<Brush> {
        self.prepare_rect_in(rect, false)
    }

    fn prepare_rect_in(&self, rect: [f32; 4], whole_direct: bool) -> Option<Brush> {
        if let Some(brush) = self.prepare_aniso_rect(rect) { return Some(brush); }
        let Brush::Bitmap { bitmap, transform: m, extend, interpolation: 5, opacity, intermediate_precision, hq: None, .. } = self else { return None; };
        if let Some((scales, residual, out, first)) = capped_hq_request(m, rect, whole_direct) {
            if first.iter().chain(out.iter()).any(|v| !v.is_finite() || *v <= i32::MIN as f32 || *v >= i32::MAX as f32) { return None; }
            let out = out.map(|v| v as i32); let first = first.map(|v| v as i32);
            let size = [out[2].checked_sub(out[0])?, out[3].checked_sub(out[1])?];
            let first_size = [first[2].checked_sub(first[0])?, first[3].checked_sub(first[1])?];
            if first_size.iter().any(|v| *v <= 0 || *v > 4096) || size.iter().any(|v| *v <= 0 || *v > 4096) { return None; }
            // Pass axes: `a` runs on the bitmap, `b` on its atlas (y then x unless x shrinks more).
            let b = usize::from(scales[0] < scales[1] && scales[1] < 1.0); let a = 1 - b;
            let pass1: Vec<[f32; 4]> = (first[1]..first[3]).flat_map(|y| (first[0]..first[2]).map(move |x| {
                let p = [x as f32 + 0.5, y as f32 + 0.5];
                crate::effects::quantize_to(*intermediate_precision, hq_axis_sample(bitmap, *extend, a, scales[a], p[a], [p[0] / bitmap.width as f32, p[1] / bitmap.height as f32]))
            })).collect();
            let dim = (first_size[0].max(first_size[1]) as u32).next_power_of_two().max(512) as i32;
            let samples = (out[1]..out[3]).flat_map(|y| (out[0]..out[2]).map(move |x| (x,y))).map(|(x,y)| {
                let p = [x as f32 + 0.5, y as f32 + 0.5]; let step = 1.0 / dim as f32;
                let s = crate::sampling::hq_downscale_axis(p[b] / scales[b], p[b], (p[b] - first[b] as f32) * step, step, scales[b], 0.0, *intermediate_precision < 2, |u| {
                    let mut i = [0; 2];
                    i[b] = (((u * dim as f32 * 256.0).floor() as i32) >> 8).clamp(0, dim-1);
                    i[a] = [x, y][a] - first[a];
                    if i[b] >= first_size[b] || i[a] < 0 || i[a] >= first_size[a] { return [0.0; 4]; }
                    let v = pass1[(i[1] * first_size[0] + i[0]) as usize];
                    if *intermediate_precision >= 2 { v } else { v.map(|v| crate::sampling::norm_v128((v * 255.0).round_ties_even() as i32 * 128)) }
                });
                crate::effects::quantize_to(*intermediate_precision, s)
            }).collect();
            let mut prepared = Brush::bitmap_source(bitmap.clone(), m, [0; 2], 5, *opacity);
            prepared.set_intermediate_precision(*intermediate_precision);
            if let Brush::Bitmap { extend: out_extend, hq, .. } = &mut prepared {
                *out_extend = *extend; *hq = Some(BitmapHq { size, origin: [out[0], out[1]], samples, precision: *intermediate_precision, residual: Some(residual), rect, slot: [0; 2] });
            }
            return Some(prepared);
        }
        if m[1] == 0.0 && m[2] == 0.0 && m[0] > 0.0 && m[3] > 0.0 && (m[0] < 1.0 && m[3] > 1.0 || m[3] < 1.0 && m[0] > 1.0) {
            // Native mixed HQ: prefilter the shrinking axis, then place its atlas with cubic.
            let rect = [rect[0].floor(), rect[1].floor(), rect[2].ceil(), rect[3].ceil()];
            let axis = usize::from(m[3] < 1.0);
            let mut bounds = [0; 4];
            // The atlas covers the cubic taps of every pixel centre the dest's whole-pixel bounds hold, under the residual
            // scale, as a capped request's does (DDI: a 0.75 x 1.7 brush's atlas spans x -2..30, y -2..49). The residual
            // is each scale over its prefilter factor, as a capped request's, so the shrinking axis keeps the factor's
            // few ulps (DDI of a 1.7 x 0.6 fill: its y pass at 0.5999999, the placement's y scale 1.0000002).
            let pre = hq_scales(m).map(|s| s.clamp(crate::sampling::HQ_MIN_SCALE, 1.0));
            let res = [m[0] * (1.0 / pre[0]), m[3] * (1.0 / pre[1])];
            for i in 0..2 {
                bounds[i] = ((((rect[i] + 0.5 - m[4 + i]) / res[i]) - 0.5).floor() - 1.0) as i32;
                bounds[i + 2] = ((((rect[i + 2] - 0.5 - m[4 + i]) / res[i]) - 0.5).ceil() + 2.0) as i32;
            }
            let size = [bounds[2] - bounds[0], bounds[3] - bounds[1]];
            let planes = [
                [[0, 0], [size[0], 0], [size[0], size[1]]],
                [[0, size[1]], [0, 0], [size[0], size[1]]],
            ].map(|p| {
                let values = p.map(|p| {
                    let scene = [(p[0] + bounds[0]) as f32, (p[1] + bounds[1]) as f32];
                    [scene[0], scene[1], scene[0] / bitmap.width as f32, scene[1] / bitmap.height as f32]
                });
                core::array::from_fn::<_, 4, _>(|ch| crate::warp::plane::setup_triangle_attr(p.map(|p| p[0] as i64 * 256), p.map(|p| p[1] as i64 * 256), values.map(|p| p[ch])))
            });
            let samples = (0..size[1]).flat_map(|y| (0..size[0]).map(move |x| {
                let (x, y) = (x as f32 + 0.5, y as f32 + 0.5);
                let triangle = usize::from(y * size[0] as f32 > x * size[1] as f32);
                let v = planes[triangle].map(|p| p[0] * x + (p[1] * y + p[2]));
                crate::effects::quantize_to(*intermediate_precision, hq_axis_sample(bitmap, *extend, axis, hq_scales(m)[axis], v[axis], [v[2], v[3]]))
            })).collect();
            let mut prepared = Brush::bitmap_source(bitmap.clone(), m, [0; 2], 5, *opacity);
            prepared.set_intermediate_precision(*intermediate_precision);
            if let Brush::Bitmap { extend: out_extend, hq, .. } = &mut prepared {
                *out_extend = *extend; *hq = Some(BitmapHq { size, origin: [bounds[0], bounds[1]], samples, precision: *intermediate_precision, residual: Some([res[0], 0.0, 0.0, res[1]]), rect, slot: [0; 2] });
            }
            return Some(prepared);
        }
        if !(m[1] == 0.0 && m[2] == 0.0 && m[0] > 0.0 && m[0] < 1.0 && m[3] > 0.0 && m[3] < 1.0) { return None; }
        let rect = [rect[0].floor(), rect[1].floor(), rect[2].ceil(), rect[3].ceil()];
        let s = hq_scales(m);
        let taps = ((2.0 / s[0]) as i32 + 3) & !3;
        let left = (((rect[0] + 0.5 - m[4]) / s[0] - 0.5).floor() as i32) - (taps - 1);
        let right = (((rect[2] - 0.5 - m[4]) / s[0] - 0.5).floor() as i32) + taps + 1;
        let top = (rect[1] - m[5]).floor() as i32;
        let bottom = (rect[3] - m[5]).ceil() as i32;
        let size = [right - left, bottom - top];
        if size[0] <= 0 || size[1] <= 0 { return None; }
        let planes = [
            [[0, 0], [size[0], 0], [size[0], size[1]]],
            [[0, size[1]], [0, 0], [size[0], size[1]]],
        ].map(|p| {
            let values = p.map(|p| {
                let scene = [(p[0] + left) as f32, (p[1] + top) as f32];
                [scene[0], scene[1], scene[0] / bitmap.width as f32, scene[1] / bitmap.height as f32]
            });
            core::array::from_fn::<_, 4, _>(|ch| crate::warp::plane::setup_triangle_attr(p.map(|p| p[0] as i64 * 256), p.map(|p| p[1] as i64 * 256), values.map(|p| p[ch])))
        });
        let samples = (0..size[1]).flat_map(|y| (0..size[0]).map(move |x| {
            let (x, y) = (x as f32 + 0.5, y as f32 + 0.5);
            let triangle = usize::from(y * size[0] as f32 > x * size[1] as f32);
            let v = planes[triangle].map(|p| p[0] * x + (p[1] * y + p[2]));
            hq_vertical(bitmap, *extend, s[1], v[1], [v[2], v[3]], *intermediate_precision)
        })).collect();
        let mut prepared = Brush::bitmap_source(bitmap.clone(), m, [0; 2], 5, *opacity);
        prepared.set_intermediate_precision(*intermediate_precision);
        if let Brush::Bitmap { extend: out_extend, hq, .. } = &mut prepared {
            *out_extend = *extend;
            *hq = Some(BitmapHq { size, origin: [left, top], samples, precision: *intermediate_precision, residual: None, rect, slot: [0; 2] });
        }
        Some(prepared)
    }

    /// Premultiplied source colour (BGRA float 0..255) at device pixel centre `(x, y)`. `jit` picks
    /// the gradient sampler: a pixel-aligned rectangle fast-path fill (`jit=false`) realizes the
    /// gradient as a bitmap brush (round-half-up bilinear), a geometry fill (`jit=true`) runs WARP's
    /// gradient pixel shader (truncating lerp) — the two round a half-texel apart (solids are
    /// unaffected; geometry bitmap sampling floors the Q8 address). A linear gradient over a rect is bitmap-sampled; a radial gradient has no 1-row
    /// bitmap form, so it always uses its pixel shader.
    pub fn source_at(&self, x: f32, y: f32, jit: bool) -> [f32; 4] {
        self.source_at_rect_impl(x, y, jit, None, [0.0; 2], false, false)
    }

    /// [`Self::source_at`] for a pixel shader that multiplies the filtered sample by a coverage mask as well
    /// as the opacity (a geometry realization's tile composite): a gradient's lerp stays fractional and is
    /// read the way the shader unpacks it, even at full opacity.
    pub(crate) fn source_at_frac(&self, x: f32, y: f32) -> [f32; 4] {
        self.source_at_rect_impl(x, y, true, None, [0.0; 2], false, true)
    }

    pub(crate) fn source_at_rect(&self, x: f32, y: f32, jit: bool, rect: [f32; 4]) -> [f32; 4] {
        self.source_at_rect_impl(x, y, jit, Some(rect), [0.0; 2], false, false)
    }

    /// [`Self::source_at_rect`] with the texcoord planes fitted and evaluated in the frame at `origin`.
    pub(crate) fn source_at_rect_framed(&self, x: f32, y: f32, jit: bool, rect: [f32; 4], origin: [f32; 2]) -> [f32; 4] {
        self.source_at_rect_impl(x, y, jit, Some(rect), origin, false, false)
    }

    pub(crate) fn source_at_rect_origin(&self, x: f32, y: f32, rect: [f32; 4], origin: [f32; 2]) -> [f32; 4] {
        self.source_at_rect_impl(x, y, true, Some(rect), origin, false, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn source_at_rect_impl(&self, x: f32, y: f32, jit: bool, rect: Option<[f32; 4]>, origin: [f32; 2], normalized: bool, frac: bool) -> [f32; 4] {
        self.source_at_rect_base(x, y, jit, rect, origin, normalized, frac, None)
    }

    /// [`Self::source_at_rect`] with the texcoord plane `base` (`[du/dx, dv/dx, du/dy, dv/dy, u0, v0]`, normalized) that
    /// the triangle under the pixel was set up with, in place of the unit-triangle fit: a CUBIC or MULTI_SAMPLE_LINEAR
    /// bitmap drawn as several triangles samples each through its own plane.
    pub(crate) fn source_at_rect_plane(&self, x: f32, y: f32, rect: [f32; 4], base: [f32; 6]) -> [f32; 4] {
        self.source_at_rect_base(x, y, true, Some(rect), [0.0; 2], false, false, Some(base))
    }

    #[allow(clippy::too_many_arguments)]
    fn source_at_rect_base(&self, x: f32, y: f32, jit: bool, rect: Option<[f32; 4]>, origin: [f32; 2], normalized: bool, frac: bool, base: Option<[f32; 6]>) -> [f32; 4] {
        match self {
            Brush::Solid(p) => *p,
            Brush::Linear { grad, xform, opacity, .. } => {
                // Native's gradient PS computes the texcoord as `x·step_x + (y·step_y + offset)` — the
                // offset is grouped with the y-term (the 5302 gradient PS: x =
                // (y+0.5)·step_y + c is formed first, then x·step_x is added). f32 add is not associative,
                // so our old left-assoc `(x·a + y·b) + c` was ~2 ULP off and flipped near-texel-boundary
                // samples (the "DDA floor"); the radial path already groups this way. See `linear_transform`.
                let u = x * xform[0] + (y * xform[1] + xform[2]);
                // Opacity≠1 keeps the Q8 lerp fractional (the PS scales then rounds once); opacity 1
                // stores the truncated texel directly (`sample_jit`). See [`Gradient::sample_jit_f`].
                // The SOURCE_OVER/COPY rect fast path draws the ramp as a 1-row bitmap brush, so opacity
                // is WARP.s byte modulation (the bitmap rect path below), not a float scale:
                // the float product rounded a level low across the whole ramp (fuzz 7461, gradop-*).
                if !jit && *opacity != 1.0 {
                    let o8 = (opacity.clamp(0.0, 1.0) * 255.0).round_ties_even() as i32;
                    return grad.sample(u, false).map(|v| crate::layer::unorm_mul(v as i32, o8) as f32);
                }
                if frac {
                    // The tile composite is a WARP geometry draw: its ramp coordinate is the unit-triangle plane
                    // fit, as a geometry fill's is.
                    let p = crate::warp::plane::warp_fit_linear(xform);
                    let u = x * p[0] + (y * p[1] + p[2]);
                    return scale_unpacked(grad.sample_jit_f(u), opacity.clamp(0.0, 1.0));
                }
                let s = if jit && *opacity != 1.0 { grad.sample_jit_f(u) } else { grad.sample(u, jit) };
                scale_jit(s, opacity.clamp(0.0, 1.0))
            }
            Brush::Radial { grad, to_unit, focal, opacity, centered, .. } => {
                // Device pixel → gradient space. The radial PS forms the texcoord as `x·m00 + (y·m01 +
                // dx)` with **separate** multiplies/adds (not fused — the FMA plane eval is the coverage
                // rasteriser's, not this shader), then the scaled focal quadratic (`(W−1)/W`) yields
                // `t`, sampled at `u = t·W`. Matches `ref/radial-vs-dump` captures byte-exact.
                let pc = [to_unit[0] * x + (to_unit[2] * y + to_unit[4]), to_unit[1] * x + (to_unit[3] * y + to_unit[5])];
                let w = grad.width as f32;
                let t = focal_t(pc, *focal, *centered, grad.u1 / w, 0.5 / w);
                // Opacity≠1 keeps the Q8 lerp fractional (see [`Gradient::sample_jit_f`]).
                if frac {
                    return scale_unpacked(grad.sample_jit_f(t * w), opacity.clamp(0.0, 1.0));
                }
                let s = if *opacity != 1.0 { grad.sample_jit_f(t * w) } else { grad.sample_jit(t * w) };
                scale_jit(s, opacity.clamp(0.0, 1.0))
            }
            Brush::Bitmap { bitmap, to_source, transform, mips, float_mips, mip_origin, mip_size, texture_size, hq, extend, filter, interpolation, opacity, intermediate_precision, vs_row, cell, cell_quad, cell_size, text_texels, text_quad, clamp, .. } => {
                if let Some(tq) = text_quad.as_ref().filter(|_| !normalized) {
                    let (q, planes, adj) = &**tq;
                    return intermediate_texel(bitmap, *extend, *interpolation, q, planes, adj, x, y).map(|v| v * opacity.clamp(0.0, 1.0));
                }
                let output_scale = if normalized { 1.0 } else { 255.0 };
                // A HIGH_QUALITY_CUBIC brush on an 8-bit target is drawn into an intermediate of its buffer precision at
                // full opacity: its sample is stored at that precision before the brush opacity scales it (probe-hro,
                // probe-hrn: 253-993 px Δ1-2 at opacity 0.13 and 0.6 -> 0; r27 32017 at 16 bpc).
                // Quantized as a pixel, so that at sRGB precision the colour takes the sRGB store and the alpha a linear one.
                let hq_byte = |s: [f32; 4]| if *interpolation == 5 && !normalized { crate::effects::quantize_to(*intermediate_precision, s.map(|v| v.clamp(0.0, 1.0))) } else { s };
                if let Some(c) = hq.as_ref().filter(|c| c.rotated() || (rect.is_none() && c.residual.is_some())) {
                    return hq_byte(bitmap_hq_min(bitmap, *extend, transform, rect.unwrap_or(c.rect), x, y, Some(c), rect.is_none())).map(|v| v * opacity.clamp(0.0, 1.0) * output_scale);
                }
                if *interpolation == 5 && transform[1] == 0.0 && transform[2] == 0.0 && transform[0] > 0.0 && transform[3] > 0.0 && (transform[0] < 1.0 || transform[3] < 1.0) {
                    // Extreme minification: a direct `2/scale`-tap downscale is unbounded (fuzz r24
                    // 29309/29494 at 1e-6 → 2e6 taps → hang). Past the realization cap, cascade to the
                    // mip chain (bounded), as native does for a sub-pixel source; moderate minification
                    // keeps the exact direct downscale below. ponytail: samples the coarsest level
                    // (sub-pixel source ⇒ the whole-source average); a mid-mip lod refinement can follow
                    // once native's exact level is pinned.
                    if 2.0 / transform[0].min(transform[3]) > 4096.0 && !mips.is_empty() {
                        let (mw, mh, data) = mips.last().unwrap();
                        let tex = Texture { data, width: *mw, height: *mh };
                        let sx = (x - transform[4]) / transform[0] * (*mw as f32 / bitmap.width as f32);
                        let sy = (y - transform[5]) / transform[3] * (*mh as f32 / bitmap.height as f32);
                        return hq_byte(tex.sample_hqcubic(*extend, sx, sy)).map(|v| v * opacity.clamp(0.0, 1.0) * output_scale);
                    }
                    if let Some(rect) = rect { return hq_byte(bitmap_hq_min(bitmap, *extend, transform, rect, x, y, hq.as_ref(), false)).map(|v| v * opacity.clamp(0.0, 1.0) * output_scale); }
                }
                if (*interpolation != 4 && crate::float::is_highp(bitmap.format)) || (*interpolation == 4 && !float_mips.is_empty()) {
                    return self.source_at_f_rect_impl(x, y, rect, origin).map(|v| v * output_scale);
                }
                let (rect, origin) = match rect { Some(r) if *interpolation == 5 => { let (q, o) = hq_place_quad(r, origin); (Some(q), o) } _ => (rect, origin) };
                let gradients = *to_source;
                let fitted = crate::warp::plane::warp_fit_affine(to_source);
                let det = transform[0] * transform[3] - transform[1] * transform[2];
                let divided = [transform[3] / det, -transform[1] / det, -transform[2] / det, transform[0] / det,
                    (transform[2] * transform[5] - transform[3] * transform[4]) / det,
                    (transform[1] * transform[4] - transform[0] * transform[5]) / det];
                let to_source = if *interpolation == 4 { &divided } else if *interpolation >= 2 || normalized { &fitted } else { to_source };
                // interp 4 (aniso): the rasterizer fits the realization plane over the fill's INTEGER-snapped
                // device-rect corners (floor/ceil), not the fractional rect — confirmed by a native per-triangle
                // vertex capture (32825 corners [15,7,62,81], sweep-mag [24,16,80,72]). Snapping is a no-op when
                // the rect is already integer (the sampler-sweep scenes stay bit-exact) and matches native's
                // plane on a fractional placement (r27 32825 24 -> 0 px).
                let quad = rect.filter(|_| *interpolation >= 4).map(|r| if *interpolation == 4 { let bb = [r[0].floor(), r[1].floor(), r[2].ceil(), r[3].ceil()]; bitmap_quad_at(transform, *mip_size, *mip_origin, bb, x, y, origin) } else { bitmap_quad_at(transform, [bitmap.width, bitmap.height], [0; 2], r, x, y, origin) });
                let uv = quad.map(|p| p.map(|a| a[0] * (x - origin[0]) + (a[1] * (y - origin[1]) + a[2])));
                let tx = uv.map_or_else(|| to_source[0] * x + (to_source[2] * y + to_source[4]), |u| u[0] * bitmap.width as f32);
                let ty = uv.map_or_else(|| to_source[1] * x + (to_source[3] * y + to_source[5]), |u| u[1] * bitmap.height as f32);
                // ANISO (interp 4): WARP's anisotropic sampler maps a non-finite texcoord to a TRANSPARENT
                // sample, not the N/L origin-texel fold — a degenerate (inf/NaN) brush-matrix texcoord
                // contributes nothing. VM probes aniso_infX/infY on .8875: over a gray clear the fill leaves
                // the background showing through. A huge-but-FINITE texcoord stays on the coarse-mip path.
                if *interpolation == 4 && !(tx.is_finite() && ty.is_finite()) {
                    return [0.0; 4];
                }
                let det = transform[0] * transform[3] - transform[1] * transform[2];
                let phase = [(transform[3] * x - transform[2] * y + (transform[2] * transform[5] - transform[3] * transform[4])) / det,
                             (-transform[1] * x + transform[0] * y + (transform[1] * transform[4] - transform[0] * transform[5])) / det];
                let tex = Texture { data: &bitmap.premul_bgra, width: bitmap.width, height: bitmap.height };
                // Extreme-scale NEAREST bitmap brush (fuzz 22998): WARP's 16.16 fixed-point rasterizer
                // accumulator overflows i32 over the fill extent (`(1/s)·extent·2^16 ≥ 2^31`), so it falls
                // back to the f32 attribute plane rasterized in 8-ROW TILES with an incremental per-span DDA
                // (jit-dump `bb` capture, probebm 1e-6..1e-3; X accumulates 4-px SIMD blocks from `coeff·½`,
                // Y re-bases each 8-row tile to `coeff·8·tile + coeff·½` then accumulates per row). The direct
                // texel-space product `floor((1/s)·x)` loses all sub-texel precision at ~1e8 magnitude; the
                // tiled DDA reproduces native's exact low bits. Plane = WARP's attribute fit over the actual
                // fill-rect triangle (`bitmap_quad_at`, as the HQ/aniso arms use). Axis-aligned only; the
                // moderate (fixed-point) regime keeps the exact direct path below (byte-exact there).
                // ponytail: the 16.16↔f32 switch is the accumulator-overflow condition — the exact point at
                // which the brush realization stores its device→texcoord step as an f32 plane instead of a
                // 16.16 integer DDA is capture-groundable; the probes only bracket it to s∈(1e-3,1e-2), so pin
                // it if a mid-scale scene ever needs it.
                if *texture_size == [bitmap.width, bitmap.height] && transform[1] == 0.0 && transform[2] == 0.0
                    && ((*interpolation == 0 && rect.is_some_and(|r| {
                            let g = transform[0].abs().max(transform[3].abs()).recip();
                            (r[2] - r[0]).abs().max((r[3] - r[1]).abs()) * g >= 32768.0
                        }))
                        // interp 3 (MSL) has no rect on the opacity-mask/float path, so it triggers on the
                        // per-pixel texcoord step itself overflowing WARP's 16.16 accumulator (|1/scale| ≥ 2^15).
                        || (*interpolation == 3 && (1.0 / transform[0]).abs().max((1.0 / transform[3]).abs()) >= 32768.0))
                {
                    if *interpolation == 3 {
                        // MULTI_SAMPLE_LINEAR at extreme minification: the WARP MSL pixel-shader evaluates the
                        // f32 attribute plane DIRECTLY per pixel (device·the raw f32-reciprocal gradient
                        // `(1/transform)/dim` — the kernel's per-span `rdi` values, e.g. 1e-6 brush → 999999.9375,
                        // /48), NOT the NEAREST kernel's 8-row-tiled 16.16 accumulator. Four MSAA samples at the
                        // rotated-grid offsets through that gradient, each wrap/mirror-folded in NORMALIZED space
                        // before ×dimension (the texcoord is ~1e8, so folding the frac keeps the low bits), and
                        // the Q8 texel address FLOORS on every axis (the kernel's roundps-toward-−∞), then a
                        // value·128 bilinear averaged ×¼. Verified 96/96 against a native per-source-row delta
                        // sweep (jit kernel read + DDI). Axis-aligned only (cross terms 0 by the guard above).
                        let (bw, bh) = (bitmap.width as f32, bitmap.height as f32);
                        let (gu, gv) = (warp_rcp(transform[0]) / bw, warp_rcp(transform[3]) / bh);
                        let base = [gu * (x - transform[4]), gv * (y - transform[5])];
                        let qa = |v: f32| { let q = ((v - 0.5) * 256.0).floor() as i32; (q >> 8, q & 255) };
                        let reduce = |u: f32, e: Extend, n: u32| -> f32 {
                            (match e {
                                Extend::Wrap => u - u.floor(),
                                Extend::Mirror => { let m = u * 0.5; u - 2.0 * m.floor() }
                                _ => u.clamp(-10.0, 10.0),
                            }) * n as f32
                        };
                        let mut acc = [0.0f32; 4];
                        for (ox, oy) in [(-0.125, -0.375), (0.375, -0.125), (-0.375, 0.125), (0.125, 0.375)] {
                            let (qx, qy) = (qa(reduce(base[0] + ox * gu, extend[0], bitmap.width)),
                                            qa(reduce(base[1] + oy * gv, extend[1], bitmap.height)));
                            for (ch, a) in acc.iter_mut().enumerate() {
                                *a += crate::sampling::norm_v128(tex.bilinear_v128(*extend, ch, qx, qy)) * 0.25;
                            }
                        }
                        return scale(acc.map(|v| v * output_scale), opacity.clamp(0.0, 1.0));
                    }
                    let r = rect.unwrap();
                    let quad = bitmap_quad_at(transform, [bitmap.width, bitmap.height], [0; 2], r, x, y, origin);
                    let (px0, py0) = (r[0].floor(), r[1].floor());
                    let (col, row_i) = ((x - 0.5 - px0).round() as i32, (y - 0.5 - py0).round() as i32);
                    let base = [quad[0][2] + quad[0][0] * (px0 + 0.5) + quad[0][1] * (py0 + 0.5),
                                quad[1][2] + quad[1][0] * (px0 + 0.5) + quad[1][1] * (py0 + 0.5)];
                    let un = [crate::warp::plane::tiled_dda_x(quad[0][0], base[0], col),
                              crate::warp::plane::tiled_dda_y(quad[1][1], base[1], row_i)];
                    let pow2 = [bitmap.width.is_power_of_two(), bitmap.width.is_power_of_two() && bitmap.height.is_power_of_two()];
                    let idx = |ax: usize, n: u32| -> i32 {
                        let (t, e, nf) = (un[ax], extend[ax], n as f32);
                        let raw = if pow2[ax] {
                            let t = if matches!(e, Extend::Wrap | Extend::Mirror) { t } else { t.clamp(-10.0, 10.0) };
                            (t * nf).floor() as i32
                        } else {
                            (match e {
                                Extend::Wrap => (t - t.floor()) * nf,
                                Extend::Mirror => { let m = t * 0.5; (m - m.floor()) * (2.0 * nf) }
                                _ => t.clamp(-10.0, 10.0) * nf,
                            }).floor() as i32
                        };
                        fold(e, raw, n as i32)
                    };
                    let (ix, iy) = (idx(0, bitmap.width), idx(1, bitmap.height));
                    let s = tex.sample(Filter::Nearest, [Extend::Clamp; 2], ix as f32 + 0.5, iy as f32 + 0.5);
                    return scale(s.map(|v| v * output_scale), opacity.clamp(0.0, 1.0));
                }
                // The shader bilinear keeps value·128 precision until the final UNORM8 store.
                let s = match *interpolation {
                    3 => {
                        let mut acc = [0.0f32; 4];
                        // WARP's Q8 address floors a power-of-two dimension and rounds any other (the MSL
                        // JIT's cvtps2dq on a 7×5 bitmap; dimg-sc-over-i3-*). The dimension is the texture's: a
                        // cropped image brush samples its power-of-two atlas, so it floors (probe-ibb, probe-ibc).
                        let qa = |v: f32, n: u32| {
                            let a = (v - 0.5) * 256.0;
                            let q = if n.is_power_of_two() { a.floor() } else { a.round_ties_even() } as i32;
                            (q >> 8, q & 255)
                        };
                        if *texture_size == [bitmap.width, bitmap.height] {
                            // Byte-exact MSL (plain bitmap): WARP evaluates the device→NORMALIZED realization
                            // plane, folds the extend in that normalized space, THEN ×dim. The BASE texcoord
                            // uses WARP's unit-triangle attribute fit of the plane (`rp` = the raw `+0x18`
                            // plane: each gradient round-tripped `(g+c)-c` in f32, translation unchanged); the
                            // 4 MSAA sub-pixel offsets use the REFINED gradients `np` (the `+0x38` plane). The
                            // plane is derived from THIS draw's `transform` each sample, never stale (r27 32580).
                            // Confirmed 92/92 byte-exact on obs6 + 22-scene [rax+0x558] capture spread.
                            let np = vs_row.unwrap_or_else(|| Self::plane_of_b2d(transform, bitmap.width, bitmap.height));
                            let rp = base.unwrap_or_else(|| crate::warp::plane::warp_fit_affine(&np));
                            let (bw, bh) = (bitmap.width as f32, bitmap.height as f32);
                            let bu = rp[0] * x + (rp[2] * y + rp[4]);
                            let bv = rp[1] * x + (rp[3] * y + rp[5]);
                            let pow2 = [bitmap.width.is_power_of_two(), bitmap.width.is_power_of_two() && bitmap.height.is_power_of_two()];
                            let q = |ax: usize, t: f32, n: f32| -> (i32, i32) {
                                let e = extend[ax];
                                let qv = if pow2[ax] {
                                    let t = if matches!(e, Extend::Wrap | Extend::Mirror) { t } else { t.clamp(-10.0, 10.0) };
                                    crate::sampling::q8_floor((t * n - 0.5) * 256.0)
                                } else {
                                    (match e {
                                        Extend::Wrap => (t - t.floor()) * (256.0 * n),
                                        Extend::Mirror => { let m = t * 0.5; (m - m.floor()) * (512.0 * n) }
                                        _ => t.clamp(-10.0, 10.0) * (256.0 * n),
                                    }).round_ties_even() as i32 - 128
                                };
                                (qv >> 8, qv & 255)
                            };
                            for (ox, oy) in [(-0.125, -0.375), (0.375, -0.125), (-0.375, 0.125), (0.125, 0.375)] {
                                let pu = (bu + ox * np[0]) + oy * np[2];
                                let pv = (bv + ox * np[1]) + oy * np[3];
                                let (qx, qy) = (q(0, pu, bw), q(1, pv, bh));
                                for (ch, a) in acc.iter_mut().enumerate() { *a += crate::sampling::norm_v128(tex.bilinear_v128(*extend, ch, qx, qy)) * 0.25; }
                            }
                        } else if let (Some(c), Some(row)) = (cell, vs_row) {
                            for off in [[-0.125, -0.375], [0.375, -0.125], [-0.375, 0.125], [0.125, 0.375]] {
                                let [qx, qy] = atlas_cell_q8(row, *c, *cell_size as f32, x, y, off);
                                for (ch, a) in acc.iter_mut().enumerate() { *a += crate::sampling::norm_v128(tex.bilinear_v128(*extend, ch, qx, qy)) * 0.25; }
                            }
                        } else {
                            for (ox, oy) in [(-0.125, -0.375), (0.375, -0.125), (-0.375, 0.125), (0.125, 0.375)] {
                                // Image-brush atlas (power-of-two): keep the legacy texel-space path.
                                let px = (tx + ox * gradients[0]) + oy * gradients[2];
                                let py = (ty + ox * gradients[1]) + oy * gradients[3];
                                let (qx, qy) = (qa(px, texture_size[0]), qa(py, texture_size[1]));
                                for (ch, a) in acc.iter_mut().enumerate() { *a += crate::sampling::norm_v128(tex.bilinear_v128(*extend, ch, qx, qy)) * 0.25; }
                            }
                        }
                        acc
                    }
                    4 => {
                        let (lo, hi, lod) = if let (Some(c), Some(row)) = (cell, vs_row) {
                            // A cropped image brush's border cells in the frame atlas, which has one level: the atlas row's
                            // plane addresses the 512 atlas and the footprint is taken over its texels (DDI: the aniso draw
                            // reads atlas A, its texcoords normalized by 512; an isotropic footprint is one bilinear of A).
                            let r = crate::warp::plane::warp_fit_affine(row);
                            let (direction, ratio, _) = bitmap_footprint(Some([[r[0], r[2], r[4]], [r[1], r[3], r[5]]]), &r, [512; 2], x, y);
                            let uv = [r[0] * x + (r[2] * y + r[4]), r[1] * x + (r[3] * y + r[5])];
                            let v = tex.anisotropic_v128_shifted(uv, direction, ratio, *extend, [512.0; 2], [true; 2], *c);
                            (v, v, 0)
                        } else {
                        let (direction, ratio, footprint) = bitmap_footprint(quad, to_source, *mip_size, x - origin[0], y - origin[1]);
                        let lod = (crate::warpmath::mip_lod_aniso(footprint, mips.len().saturating_sub(1) as f32) * 256.0).round_ties_even() as usize;
                        let sample = |level: usize| {
                            if mips.is_empty() { return [0; 4]; }
                            let level = level.min(mips.len() - 1);
                            let (w, h, data) = &mips[level];
                            let mt = Texture { data, width: *w, height: *h };
                            // Normalise the Q8 by the CONTENT size at this mip (`content/2^level`, fractional
                            // for a non-pow2 content), floor on a pow2 axis / round on any other — WARP's one
                            // texture-unit rule (54a90d61), here per mip level. The integer mip width `>>level`
                            // mis-scales a non-pow2 content (9>>1=4 vs 9/2=4.5), the X error that grows across
                            // the dest (r24 28960/29094/29916; iso9_4 maxd 52).
                            let dims = [mip_size[0] as f32 / (1u32 << level) as f32, mip_size[1] as f32 / (1u32 << level) as f32];
                            let pow2 = [mip_size[0].is_power_of_two(), mip_size[0].is_power_of_two() && mip_size[1].is_power_of_two()];
                            let uvp = uv.unwrap_or([tx / bitmap.width as f32, ty / bitmap.height as f32]);
                            // The source-rect crop's transparent border lives in the UNPADDED mip chain + the
                            // Transparent extend (an image_rect edge past the bitmap fades through it); the fill
                            // rect's own AA edge covers an edge inside the bitmap (draw_image_placed aniso arm).
                            mt.anisotropic_v128_size(uvp, direction, ratio, *extend, dims, pow2)
                        };
                        (sample(lod >> 8), sample((lod >> 8) + 1), lod)
                        };
                        // Onto an 8bpc target the pixel-aligned rect fast-path fill (`rect=Some`) floors the
                        // value·128 bilinear to UNORM8 and modulates it by WARP's integer `unorm_mul` with the
                        // 8-bit opacity `round(opacity·255)` — the geometry fill's byte modulation, not a float
                        // multiply (r27 32825 brush fill at opacity 0.78: 1615 → 24; 32658/32666/32706). The
                        // general shader path keeps the float-opacity floor fall-through below (the caller's
                        // `scale`), as a DrawBitmap/non-rect fill does.
                        if !normalized && rect.is_some() {
                            let o8 = (opacity.clamp(0.0, 1.0) * 255.0).round_ties_even() as i32;
                            return core::array::from_fn(|ch| {
                                let v = lo[ch] + (((hi[ch] - lo[ch]) * (lod & 255) as i32) >> 8);
                                crate::bitmap::unorm_mul((v >> 7).clamp(0, 255), o8) as f32
                            });
                        }
                        // Above 8bpc native draws the brush into an intermediate of the buffer precision first (DDI: one
                        // draw into a 512 atlas over the fill's bounds), so the filtered value is stored at that precision,
                        // not truncated to a byte (VM probes of magnified ANISOTROPIC fills at precision 3 and 4).
                        let at_precision = !normalized && *intermediate_precision >= 2;
                        core::array::from_fn(|ch| {
                            let v = lo[ch] + (((hi[ch] - lo[ch]) * (lod & 255) as i32) >> 8);
                            if at_precision {
                                // An sRGB intermediate converts the colour from the float and stores the alpha as the
                                // value·128 byte truncated, as the 8bpc path does (VM probes at precision 2).
                                let q = crate::effects::quantize_to(*intermediate_precision, [crate::sampling::norm_v128(v); 4]);
                                if ch != 3 { q[0] } else if *intermediate_precision == 2 { (v >> 7) as f32 / 255.0 } else { q[3] }
                            } else if normalized { crate::sampling::norm_v128(v) } else { (v >> 7) as f32 / 255.0 }
                        })
                    },
                    2 if cell.is_some() => {
                        // Border cells in the frame atlas: the atlas row's plane addresses the atlas, phased in its texels.
                        let (u, phase) = cell_cubic_uv(vs_row.as_ref().unwrap(), cell.unwrap(), *cell_size, cell_quad.as_ref(), x, y);
                        let s = tex.sample_hqcubic_unclamped_at(*extend, u, phase, [*cell_size; 2], cell.unwrap());
                        if normalized { s } else { s.map(|v| v.clamp(0.0, 1.0)) }
                    }
                    // Native brush cubic JIT normalizes each bilinear window before its X/Y sums.
                    5 if normalized => tex.sample_hqcubic_at_unclamped(*extend, uv.unwrap_or([tx / bitmap.width as f32, ty / bitmap.height as f32]), phase),
                    // A draw that built its own row (a rotated DrawBitmap's, d2d1's source-to-dest matrix) samples it over its rect too.
                    // A triangle's own texcoord plane (a COPY draw's pieces): the addresses take `u`, the phase `u·n`.
                    2 if base.is_some() && *texture_size == [bitmap.width, bitmap.height] => {
                        let r = base.unwrap();
                        let (w, h) = (bitmap.width as f32, bitmap.height as f32);
                        let u = [r[0] * x + (r[2] * y + r[4]), r[1] * x + (r[3] * y + r[5])];
                        if normalized { tex.sample_hqcubic_at_unclamped(*extend, u, [u[0] * w, u[1] * h]) } else { tex.sample_hqcubic_at(*extend, u, [u[0] * w, u[1] * h]) }
                    }
                    2 if *texture_size == [bitmap.width, bitmap.height] && (rect.is_none() || vs_row.is_some()) => {
                        // The cubic JIT addresses all nine bilinear windows and takes the polynomial phase off
                        // the brush's normalized VS rows (the LINEAR arm's plane below): `u` for the addresses and
                        // `u·n` for the phase, not the texel plane over the size (r27 32539: a rotated 15×9
                        // WRAP/MIRROR brush was a step off at scattered pixels).
                        let (w, h) = (bitmap.width as f32, bitmap.height as f32);
                        let row = crate::warp::plane::warp_fit_affine(&vs_row.unwrap_or_else(|| invert(&[transform[0] * w, transform[1] * w, transform[2] * h, transform[3] * h, transform[4], transform[5]])));
                        let u = [row[0] * x + (row[2] * y + row[4]), row[1] * x + (row[3] * y + row[5])];
                        let phase = [u[0] * w, u[1] * h];
                        if normalized { tex.sample_hqcubic_at_unclamped(*extend, u, phase) } else { tex.sample_hqcubic_at(*extend, u, phase) }
                    }
                    // A realized image brush's atlas, and a DrawImage's own kernel, keep the texel plane.
                    2 if normalized => tex.sample_hqcubic_at_unclamped(*extend, [tx / bitmap.width as f32, ty / bitmap.height as f32], [tx, ty]),
                    2 => tex.sample_hqcubic(*extend, tx, ty),
                    5 => tex.sample_hqcubic_at(*extend, uv.unwrap_or([tx / bitmap.width as f32, ty / bitmap.height as f32]), phase),
                    1 if jit && cell.is_some() => {
                        let [qx, qy] = atlas_cell_q8(vs_row.as_ref().unwrap(), cell.unwrap(), *cell_size as f32, x, y, [0.0; 2]);
                        core::array::from_fn(|ch| crate::sampling::norm_v128(tex.bilinear_v128(*extend, ch, qx, qy)))
                    }
                    1 if jit && *texture_size != [bitmap.width, bitmap.height] => {
                        // A realized image brush's power-of-two atlas: the texel-space floor address.
                        let q = |v: f32| {
                            let q = ((v - 0.5) * 256.0).floor() as i32;
                            (q >> 8, q & 255)
                        };
                        core::array::from_fn(|ch| crate::sampling::norm_v128(tex.bilinear_v128(*extend, ch, q(tx), q(ty))))
                    }
                    0 if jit && *texture_size == [bitmap.width, bitmap.height] => {
                        // Geometry's NEAREST bitmap-brush pixel shader (the `meshbitmap`/`meshbitmap_cm`
                        // kernels, each byte-exact for WRAP/CLAMP/MIRROR). The extend fold runs in
                        // NORMALIZED texcoord space off the LINEAR jit arm's `row` plane: WRAP folds the
                        // frac (u − ⌊u⌋) BEFORE ×dimension so a huge-magnitude texcoord keeps [0,1)·n
                        // precision (fuzz r17 22971: 1e-6 brush → texcoords ~1e8), CLAMP floors u·n and
                        // clamps, MIRROR reflects over 2n. Guarded to a true bitmap brush (NOT a realized
                        // image-brush pow2 atlas, whose `texture_size` ≠ the bitmap — that path keeps `_`).
                        let (bw, bh) = (bitmap.width as f32, bitmap.height as f32);
                        let row = crate::warp::plane::warp_fit_affine(&vs_row.unwrap_or_else(|| invert(&[transform[0] * bw, transform[1] * bw, transform[2] * bh, transform[3] * bh, transform[4], transform[5]])));
                        let u = [row[0] * x + (row[2] * y + row[4]), row[1] * x + (row[3] * y + row[5])];
                        let u = clamp.map_or(u, |c| [u[0].clamp(c[0], c[2]), u[1].clamp(c[1], c[3])]);
                        let pow2 = [bitmap.width.is_power_of_two(), bitmap.width.is_power_of_two() && bitmap.height.is_power_of_two()];
                        let idx = |ax: usize, n: u32| -> i32 {
                            let (t, e) = (u[ax], extend[ax]);
                            let nf = n as f32;
                            let raw = if pow2[ax] {
                                let t = if matches!(e, Extend::Wrap | Extend::Mirror) { t } else { t.clamp(-10.0, 10.0) };
                                (t * nf).floor() as i32
                            } else {
                                (match e {
                                    Extend::Wrap => (t - t.floor()) * nf,
                                    Extend::Mirror => { let m = t * 0.5; (m - m.floor()) * (2.0 * nf) }
                                    _ => t.clamp(-10.0, 10.0) * nf,
                                }).floor() as i32
                            };
                            fold(e, raw, n as i32)
                        };
                        let (ix, iy) = (idx(0, bitmap.width), idx(1, bitmap.height));
                        tex.sample(Filter::Nearest, [Extend::Clamp; 2], ix as f32 + 0.5, iy as f32 + 0.5)
                    }
                    1 if jit => {
                        // Geometry's sampler reads the brush's VS rows (traced r9 14261 stroke): the brush→device
                        // matrix with the bitmap size folded in first, inverted by one reciprocal determinant (not
                        // invert-then-divide), fitted over the unit triangle. The address follows the shader JIT
                        // (gbb-* probes): an axis on WARP's power-of-two path (u when the width is a power of two,
                        // v when both dimensions are) floors `(u·n − ½)·256`, clamped to ±10 under CLAMP only, and
                        // WRAP/MIRROR fold the texel index; any other axis folds WRAP (`frac u`) or MIRROR
                        // (`frac(u·½)` over 2n) in normalized space and rounds `·256n` to nearest, less ½ texel.
                        // The opacity scales the normalized sample before ×255.
                        let (w, h) = (bitmap.width as f32, bitmap.height as f32);
                        let row = crate::warp::plane::warp_fit_affine(&vs_row.unwrap_or_else(|| invert(&[transform[0] * w, transform[1] * w, transform[2] * h, transform[3] * h, transform[4], transform[5]])));
                        let u = [row[0] * x + (row[2] * y + row[4]), row[1] * x + (row[3] * y + row[5])];
                        let u = clamp.map_or(u, |c| [u[0].clamp(c[0], c[2]), u[1].clamp(c[1], c[3])]);
                        let pow2 = [bitmap.width.is_power_of_two(), bitmap.width.is_power_of_two() && bitmap.height.is_power_of_two()];
                        let q = |ax: usize, n: f32| -> (i32, i32) {
                            let (t, e) = (u[ax], extend[ax]);
                            if !pow2[ax] {
                                return crate::sampling::q8_address_npow2_geom(t, n, e);
                            }
                            let t = if matches!(e, Extend::Wrap | Extend::Mirror) { t } else { t.clamp(-10.0, 10.0) };
                            let q = crate::sampling::q8_floor((t * n - 0.5) * 256.0);
                            (q >> 8, q & 255)
                        };
                        let (qx, qy, op) = (q(0, w), q(1, h), opacity.clamp(0.0, 1.0));
                        // At full opacity onto an 8bpc target the kernel stores the value·128 bilinear truncated (`>> 7`),
                        // as the colourless sprite PS does: a rotated image-rectangle fill (sc113, sc114) is byte-exact
                        // only so; rounding it puts most pixels a step high.
                        if op == 1.0 && !normalized {
                            return core::array::from_fn(|ch| ((tex.bilinear_v128(*extend, ch, qx, qy) >> 7) as f32 / 255.0) * output_scale);
                        }
                        return core::array::from_fn(|ch| (crate::sampling::norm_v128(tex.bilinear_v128(*extend, ch, qx, qy)) * op).clamp(0.0, 1.0) * output_scale);
                    }
                    1 => tex.linear_v128(*extend, tx, ty),
                    _ => tex.sample(*filter, *extend, tx, ty),
                };
                // A text fill reads a CUBIC or MULTI_SAMPLE_LINEAR brush's whole-byte texels at opacity 1 and applies the opacity
                // after ([`Brush::as_text_texels`]).
                if *text_texels && matches!(interpolation, 2 | 3) && !normalized {
                    return s.map(|v| (v.clamp(0.0, 1.0) * 255.0).round_ties_even() * opacity.clamp(0.0, 1.0));
                }
                // The MULTI_SAMPLE_LINEAR kernel clamps the four taps' average, then scales it by the opacity before
                // ×255 (JIT dump); the other filters scale the stored-range sample.
                if *interpolation == 3 { return s.map(|v| (v.clamp(0.0, 1.0) * opacity.clamp(0.0, 1.0)) * output_scale); }
                scale(hq_byte(s).map(|v| v * output_scale), opacity.clamp(0.0, 1.0))
            }
            // No geometry: sampled only through `sample_ramp_f(t)` (COLR conical/sweep).
            Brush::Ramp { .. } => [0.0; 4],
        }
    }

    /// Premultiplied source colour (BGRA, **float 0..1**) at a device pixel centre for a **float
    /// render target** (`crate::float`). Gradients retain their ramp precision; bitmap brushes
    /// decode the source format and use WARP's float bilinear and cubic arithmetic.
    pub(crate) fn source_at_f(&self, x: f32, y: f32) -> [f32; 4] {
        self.source_at_f_rect_impl(x, y, None, [0.0; 2])
    }

    pub(crate) fn source_at_f_rect(&self, x: f32, y: f32, rect: [f32; 4]) -> [f32; 4] {
        self.source_at_f_rect_impl(x, y, Some(rect), [0.0; 2])
    }

    pub(crate) fn source_at_f_rect_origin(&self, x: f32, y: f32, rect: [f32; 4], origin: [f32; 2]) -> [f32; 4] {
        self.source_at_f_rect_impl(x, y, Some(rect), origin)
    }

    fn source_at_f_rect_impl(&self, x: f32, y: f32, rect: Option<[f32; 4]>, origin: [f32; 2]) -> [f32; 4] {
        match self {
            Brush::Linear { grad, xform, opacity, .. } => {
                // A float target stores the gradient PS's sample unrounded: the BGRA8 path's texcoord
                // (`x·a + (y·b + c)`) and Q8 address, its lerp kept in the sampler's 8.7 fixed point
                // ([`Gradient::sample_jit_f`]; Rainbow's 361-stop ramp into a Float32 target, native probe).
                let u = x * xform[0] + (y * xform[1] + xform[2]);
                scale(grad.sample_jit_f(u).map(|v| v / 255.0), opacity.clamp(0.0, 1.0))
            }
            Brush::Radial { grad, to_unit, focal, opacity, centered, .. } => {
                let pc = [to_unit[0] * x + (to_unit[2] * y + to_unit[4]), to_unit[1] * x + (to_unit[3] * y + to_unit[5])];
                let w = grad.width as f32;
                let u = focal_t(pc, *focal, *centered, grad.u1 / w, 0.5 / w) * w;
                scale(grad.sample_jit_f(u).map(|v| v / 255.0), opacity.clamp(0.0, 1.0))
            }
            Brush::Bitmap { bitmap, to_source, transform, float_mips, mip_origin, mip_size, hq, extend, interpolation, opacity, vs_row, cell, cell_quad, cell_size, .. } => {
                if (*interpolation == 4 && float_mips.is_empty()) || (*interpolation != 4 && !crate::float::is_highp(bitmap.format)) {
                    return self.source_at_rect_impl(x, y, true, rect, origin, true, false);
                }
                if let Some(c) = hq.as_ref().filter(|c| c.rotated() || (rect.is_none() && c.residual.is_some())) {
                    return bitmap_hq_min(bitmap, *extend, transform, rect.unwrap_or(c.rect), x, y, Some(c), rect.is_none()).map(|v| v * opacity.clamp(0.0, 1.0));
                }
                if *interpolation == 5 && transform[1] == 0.0 && transform[2] == 0.0 && transform[0] > 0.0 && transform[3] > 0.0 && (transform[0] < 1.0 || transform[3] < 1.0) {
                    if let Some(rect) = rect { return bitmap_hq_min(bitmap, *extend, transform, rect, x, y, hq.as_ref(), false).map(|v| v * opacity.clamp(0.0, 1.0)); }
                }
                let (rect, origin) = match rect { Some(r) if *interpolation == 5 => { let (q, o) = hq_place_quad(r, origin); (Some(q), o) } _ => (rect, origin) };
                let m = crate::warp::plane::warp_fit_affine(to_source);
                let quad = rect.filter(|_| *interpolation >= 4).map(|r| if *interpolation == 4 { bitmap_quad_at(transform, *mip_size, *mip_origin, r, x, y, origin) } else { bitmap_quad_at(transform, [bitmap.width, bitmap.height], [0; 2], r, x, y, origin) });
                let uv = quad.map(|p| p.map(|a| a[0] * (x - origin[0]) + (a[1] * (y - origin[1]) + a[2])));
                let tx = uv.map_or_else(|| m[0] * x + (m[2] * y + m[4]), |u| u[0] * bitmap.width as f32);
                let ty = uv.map_or_else(|| m[1] * x + (m[3] * y + m[5]), |u| u[1] * bitmap.height as f32);
                let det = transform[0] * transform[3] - transform[1] * transform[2];
                let phase = if *interpolation == 5 { [(transform[3] * x - transform[2] * y + (transform[2] * transform[5] - transform[3] * transform[4])) / det,
                    (-transform[1] * x + transform[0] * y + (transform[1] * transform[4] - transform[0] * transform[5])) / det] } else { [tx, ty] };
                let fetch = |x, y| bitmap_fetch(bitmap, *extend, x, y);
                let q = |v: f32| { let q = ((v - 0.5) * 256.0).floor() as i32; (q >> 8, q & 255) };
                let s = match *interpolation {
                    0 => fetch(tx.floor() as i32, ty.floor() as i32),
                    4 => {
                        let (direction, ratio, footprint) = bitmap_footprint(quad, to_source, *mip_size, x - origin[0], y - origin[1]);
                        let lod = (crate::warpmath::mip_lod_aniso(footprint, float_mips.len().saturating_sub(1) as f32) * 256.0).round_ties_even() as usize;
                        let sample = |level: usize| {
                            let level = level.min(float_mips.len() - 1);
                            let (w, h, data) = &float_mips[level];
                            let dims = [(mip_size[0] >> level).max(1) as f32, (mip_size[1] >> level).max(1) as f32];
                            crate::sampling::anisotropic_f32(uv.unwrap_or([tx / bitmap.width as f32, ty / bitmap.height as f32]), dims, direction, ratio, |x, y| {
                                if (extend[0] == Extend::Transparent && (x < 0 || x >= *w as i32)) || (extend[1] == Extend::Transparent && (y < 0 || y >= *h as i32)) { return [0.0; 4]; }
                                data[(extend[1].wrap(y, *h as i32) as u32 * w + extend[0].wrap(x, *w as i32) as u32) as usize]
                            })
                        };
                        let (lo, hi) = (sample(lod >> 8), sample((lod >> 8) + 1));
                        core::array::from_fn(|ch| lo[ch] + (hi[ch] - lo[ch]) * ((lod & 255) as f32 / 256.0))
                    }
                    3 if cell.is_some() => {
                        let mut acc = [0.0; 4];
                        for off in [[-0.125, -0.375], [0.375, -0.125], [-0.375, 0.125], [0.125, 0.375]] {
                            let [qx, qy] = atlas_cell_q8(vs_row.as_ref().unwrap(), cell.unwrap(), *cell_size as f32, x, y, off);
                            let tap = crate::sampling::bilinear_f32(qx, qy, fetch);
                            for ch in 0..4 { acc[ch] += tap[ch] * 0.25; }
                        }
                        acc
                    }
                    3 => {
                        let mut acc = [0.0; 4];
                        for (ox, oy) in [(-0.125, -0.375), (0.375, -0.125), (-0.375, 0.125), (0.125, 0.375)] {
                            let px = (tx + ox * to_source[0]) + oy * to_source[2];
                            let py = (ty + ox * to_source[1]) + oy * to_source[3];
                            let tap = crate::sampling::bilinear_f32(q(px), q(py), fetch);
                            for ch in 0..4 { acc[ch] += tap[ch] * 0.25; }
                        }
                        acc
                    }
                    2 | 5 => {
                        // Border cells in the frame atlas (CUBIC): the atlas row's plane addresses the atlas, phased in its
                        // texels, each address stepped back by the cell's offset.
                        let (u, phase, dims, at) = match (cell, vs_row) {
                            (Some(c), Some(row)) if *interpolation == 2 => {
                                let (u, phase) = cell_cubic_uv(row, *c, *cell_size, cell_quad.as_ref(), x, y);
                                (u, phase, [*cell_size; 2], *c)
                            }
                            _ => ([uv.map_or(tx / bitmap.width as f32, |u| u[0]), uv.map_or(ty / bitmap.height as f32, |u| u[1])], phase, [bitmap.width, bitmap.height], [0; 2]),
                        };
                        let axis = |t: f32, u: f32, dim: u32, at: i32| {
                            let f = (t - 0.5) - (t - 0.5).floor();
                            let w = crate::sampling::cubic_weights(f, 0.0);
                            let g = w[1] + w[2];
                            let step = 1.0 / dim as f32;
                            let base = u - f * step;
                            ([base - step, base + (w[2] / g) * step, base + 2.0 * step].map(|v| { let (i, f) = q(v * dim as f32); (i.wrapping_sub(at), f) }), [w[0], g, w[3]])
                        };
                        let (xs, wx) = axis(phase[0], u[0], dims[0], at[0]);
                        let (ys, wy) = axis(phase[1], u[1], dims[1], at[1]);
                        crate::sampling::cubic_windows_f32(xs, ys, wx, wy, fetch)
                    }
                    1 if cell.is_some() => {
                        let [qx, qy] = atlas_cell_q8(vs_row.as_ref().unwrap(), cell.unwrap(), *cell_size as f32, x, y, [0.0; 2]);
                        crate::sampling::bilinear_f32(qx, qy, fetch)
                    }
                    _ => crate::sampling::bilinear_f32(q(tx), q(ty), fetch),
                };
                [s[2], s[1], s[0], s[3]].map(|v| v * opacity.clamp(0.0, 1.0))
            }
            _ => self.source_at(x, y, false).map(|v| v / 255.0),
        }
    }

    // Native masklinear JIT keeps normalized unit-triangle UVs and value*128 filtering until mask modulation.
    pub(crate) fn source_for_mask(&self, x: f32, y: f32) -> [f32; 4] {
        match self {
            Brush::Linear { grad, xform, opacity, .. } => {
                let w = grad.width as f32;
                let [dx, dy, c] = xform.map(|v| v / w);
                let plane = crate::warp::plane::warp_fit_affine(&[dx, 0.0, dy, 0.0, c, 0.5]);
                let u = (x * plane[0] + (y * plane[2] + plane[4])) * w;
                let q = ((u - 0.5) * 256.0).floor() as i32;
                let tex = Texture { data: &grad.texels, width: grad.width as u32, height: 1 };
                core::array::from_fn(|ch| crate::sampling::norm_v128(tex.bilinear_v128(
                    [extend_of(grad.extend), Extend::Clamp], ch, (q >> 8, q & 255), (0, 0),
                )) * opacity.clamp(0.0, 1.0))
            }
            // The radial mask kernel filters the same way: the focal `t` at the fitted unit-plane `pc`
            // (as `source_at`), then the 8.7 bilinear, value*128 until the mask modulation (r4/r3 mask probes).
            Brush::Radial { grad, to_unit, focal, opacity, centered, .. } => {
                let pc = [to_unit[0] * x + (to_unit[2] * y + to_unit[4]), to_unit[1] * x + (to_unit[3] * y + to_unit[5])];
                let w = grad.width as f32;
                let u = focal_t(pc, *focal, *centered, grad.u1 / w, 0.5 / w) * w;
                let q = ((u - 0.5) * 256.0).floor() as i32;
                let tex = Texture { data: &grad.texels, width: grad.width as u32, height: 1 };
                core::array::from_fn(|ch| crate::sampling::norm_v128(tex.bilinear_v128(
                    [extend_of(grad.extend), Extend::Clamp], ch, (q >> 8, q & 255), (0, 0),
                )) * opacity.clamp(0.0, 1.0))
            }
            _ => self.source_at_f(x, y),
        }
    }

    /// The brush's maximally-minified sample — the gradient's 1×1 coarsest mip, a CONSTANT. An extreme
    /// minification footprint drives WARP's LOD to the coarsest level (as [`Brush::nan_lod`]); the
    /// degenerate FillOpacityMask strip (a dest edge past the |524287| safe box) is that case, so its
    /// gradient brush samples the 1×1 level, not the per-pixel full-res ramp. Mirrors `source_for_mask`'s
    /// `value·128`→norm×opacity on the single coarsest texel.
    pub(crate) fn coarse_sample(&self) -> [f32; 4] {
        match self {
            Brush::Linear { grad, opacity, .. } | Brush::Radial { grad, opacity, .. } => {
                let tex = Texture { data: &grad.coarsest, width: 1, height: 1 };
                core::array::from_fn(|ch| crate::sampling::norm_v128(tex.bilinear_v128([Extend::Clamp; 2], ch, (0, 0), (0, 0))) * opacity.clamp(0.0, 1.0))
            }
            _ => self.source_for_mask(0.5, 0.5),
        }
    }

    pub(crate) fn source_for_text(&self, x: f32, y: f32) -> [f32; 4] {
        let (s, opacity) = self.text_sample(x, y);
        scale(s, opacity)
    }

    /// The text kernels' brush sample before the brush opacity, and that opacity (a gradient's 8.7 ramp
    /// sample; other brushes come premultiplied by theirs, opacity 1).
    pub(crate) fn text_sample(&self, x: f32, y: f32) -> ([f32; 4], f32) {
        match self {
            Brush::Linear { grad, xform, opacity, .. } => {
                // Text/COLR fills are WARP geometry draws: the ramp coordinate is the unit-triangle plane
                // fit, evaluated `x·a + (y·b + c)` like the kernel (textx-emoji ±1s).
                let p = crate::warp::plane::warp_fit_linear(xform);
                (grad.sample_text(p[0] * x + (p[1] * y + p[2])), opacity.clamp(0.0, 1.0))
            }
            Brush::Radial { grad, to_unit, focal, opacity, centered, .. } => {
                let pc = [to_unit[0] * x + (to_unit[2] * y + to_unit[4]), to_unit[1] * x + (to_unit[3] * y + to_unit[5])];
                let w = grad.width as f32;
                // Provenance: WARP centered radial gradient-mask JIT (emoji-001, +0x51a..0x541).
                let t = if *centered { (pc[0] * pc[0] + pc[1] * pc[1]).sqrt() }
                    else { focal_t(pc, *focal, *centered, grad.u1 / w, 0.5 / w) };
                (grad.sample_text(t * w), opacity.clamp(0.0, 1.0))
            }
            // The text kernel reads a LINEAR bitmap's value·128 bilinear in float: rounded, not truncated as the
            // full-opacity sprite store is (VM probe-cb: Arial and Courier New 'Wlg' under a magnified 2×2 LINEAR
            // bitmap brush, ±1 on 157-348 px truncated).
            Brush::Bitmap { interpolation: 1, .. } => (self.source_at_rect_impl(x, y, true, None, [0.0; 2], true, false).map(|v| v * 255.0), 1.0),
            _ => (self.source_at_round(x, y), 1.0),
        }
    }

    /// The text kernels read a filtered (CUBIC, MULTI_SAMPLE_LINEAR, ANISOTROPIC, HIGH_QUALITY_CUBIC) bitmap or
    /// image brush as a UNORM8 texel and apply the brush opacity after, in float (r25 30331 and kin: exact at
    /// opacity 1, and rounding after the opacity was worse below it). LINEAR keeps its kernel, which samples the
    /// bitmap with the opacity inside; NEAREST's texel is whole bytes already. The sampler for one draw:
    /// whole-byte samples of the brush at opacity 1, paired with its opacity.
    /// This text-texel brush ([`Brush::as_text_texels`]) read through an intermediate over `quad` cut to `limit`.
    pub(crate) fn with_text_quad(&self, quad: [f32; 4], limit: [f32; 4]) -> Brush {
        let mut b = self.clone();
        let planes = intermediate_planes(self, clip_quad(quad, limit));
        if let Brush::Bitmap { text_quad, .. } = &mut b {
            *text_quad = planes.map(Box::new);
        }
        b
    }

    /// This brush for a text outline filled as geometry: a CUBIC or MULTI_SAMPLE_LINEAR bitmap brush is read as the
    /// text kernels read it ([`Brush::text_texels`]). Direct2D draws a CUBIC or MULTI_SAMPLE_LINEAR bitmap brush through
    /// an intermediate for a text primitive (its brush-redirection test lets NEAREST and LINEAR through for every
    /// primitive, these two only for the low primitive kinds the geometry fills are), so the glyph fill reads whole
    /// bytes. VM probes of a 537-px OUTLINE run
    /// (r25 30178) and a 314-px one (r25 30556): the same brushes fill a rect, an ellipse and two paths exactly as
    /// geometry samples them, and the run only this way.
    pub(crate) fn as_text_texels(&self) -> Option<Brush> {
        let mut b = self.clone();
        let Brush::Bitmap { interpolation: 2 | 3, text_texels, .. } = &mut b else { return None };
        *text_texels = true;
        Some(b)
    }

    /// `quad`: the glyph texture's whole-pixel bounds the text draws over; `limit`: the device rect an intermediate is
    /// drawn within (the target within the pushed clips).
    pub(crate) fn text_texels(&self, quad: [f32; 4], limit: [f32; 4]) -> Option<impl Fn(f32, f32) -> ([f32; 4], f32)> {
        let Brush::Bitmap { opacity, interpolation, intermediate_precision, .. } = self else { return None };
        if *interpolation <= 1 {
            return None;
        }
        let (op, hq, precision) = (opacity.clamp(0.0, 1.0), *interpolation == 5, *intermediate_precision);
        let mut unit = self.clone();
        if let Brush::Bitmap { opacity, .. } = &mut unit {
            *opacity = 1.0;
        }
        let staged = matches!(*interpolation, 2 | 3).then(|| intermediate_planes(self, clip_quad(quad, limit))).flatten();
        // An ANISOTROPIC brush's intermediate is the brush drawn over the quad in the atlas, as its rect fill draws one
        // (VM DDI of r25 30150: a quad with the brush's vertex shader, the pixel shader one anisotropic sample).
        let aniso = (*interpolation == 4).then(|| {
            let q = clip_quad(quad, limit);
            (q[2] > q[0] && q[3] > q[1] && q.iter().all(|v| v.is_finite())).then(|| (q, unit.prepare_rect(q).unwrap_or_else(|| unit.clone())))
        }).flatten();
        Some(move |x: f32, y: f32| {
            if let (Some((q, planes, adj)), Brush::Bitmap { bitmap, extend, interpolation, .. }) = (&staged, &unit) {
                return (intermediate_texel(bitmap, *extend, *interpolation, q, planes, adj, x, y), op);
            }
            if let Some((q, b)) = &aniso {
                return (b.source_at_rect_framed(x, y, true, *q, [q[0], q[1]]).map(f32::round_ties_even), op);
            }
            // A HIGH_QUALITY_CUBIC brush reaches the text through its intermediate, placed over the draw's whole-pixel
            // quad from that quad's origin and stored at full opacity, as it reaches a mask (r25 30179, 30961: a
            // magnified 7×4 brush's texel-boundary row ±1 sampled straight; r25 30231 4 px → 2 from the quad's origin).
            let s = if hq {
                crate::effects::quantize_to(precision, unit.source_at_f_rect_origin(x, y, quad, [quad[0], quad[1]]).map(|v| v.clamp(0.0, 1.0))).map(|v| v * 255.0)
            } else {
                unit.source_at_round(x, y)
            };
            (s.map(f32::round_ties_even), op)
        })
    }

    /// Premultiplied source colour (BGRA float 0..255) sampling the ramp through the COLR colorizer's
    /// round-half-up lerp ([`Gradient::sample_jit_round`]) — the exact native rounding for COLR leaf
    /// gradient fills. Linear/radial only; others defer to [`source_at`].
    pub(crate) fn source_at_round(&self, x: f32, y: f32) -> [f32; 4] {
        match self {
            Brush::Linear { grad, xform, opacity, .. } => {
                let u = x * xform[0] + y * xform[1] + xform[2];
                scale(grad.sample_jit_round(u), opacity.clamp(0.0, 1.0))
            }
            Brush::Radial { grad, to_unit, focal, opacity, centered, .. } => {
                let pc = [to_unit[0] * x + (to_unit[2] * y + to_unit[4]), to_unit[1] * x + (to_unit[3] * y + to_unit[5])];
                let w = grad.width as f32;
                let t = focal_t(pc, *focal, *centered, grad.u1 / w, 0.5 / w);
                scale(grad.sample_jit_round(t * w), opacity.clamp(0.0, 1.0))
            }
            _ => self.source_at(x, y, true),
        }
    }
}

/// The device-independent definition of a gradient/bitmap brush, retained so a command list can
/// re-realize it under the replay transform — native records brushes device-independently and
/// realizes them against the draw-time transform. `xform` is the brush's own transform (identity when
/// the app passed none); [`realize`](BrushDef::realize) rebuilds the device-space [`Brush`] against the
/// `brush→device` matrix `device · xform`, matching `DeviceContext::brush_to_device` byte-for-byte
/// (compose in f64, then down to f32).
#[derive(Clone)]
pub enum BrushDef {
    Linear { stops: Vec<(f32, [f32; 4])>, gamma: u32, extend: u32, opacity: f32, start: [f32; 2], end: [f32; 2], xform: [f32; 6] },
    Radial { stops: Vec<(f32, [f32; 4])>, gamma: u32, extend: u32, opacity: f32, center: [f32; 2], radii: [f32; 2], offset: [f32; 2], xform: [f32; 6] },
    Bitmap { texels: Vec<u8>, size: [u32; 2], extend: [u32; 2], interpolation: u32, opacity: f32, xform: [f32; 6], dpi: (f32, f32) },
}

/// `a ∘ b` (b applied first, then a), f64, identical to `context::compose`.
/// `brush→device` = `device · brush_xform`, composed in **f32** exactly as native's
/// `D2D1::Matrix3x2F::SetProduct` (brush·world, `GetRealizationPlacementData`): each element is two
/// f32 products summed (f32 `+` is commutative, so native's per-element operand order is immaterial),
/// the translation grouped `((p0 + p1) + t)`. We previously composed in f64 then rounded, which
/// diverged from native for any brush drawn under a non-identity brush transform.
fn compose_f32(a: &[f32; 6], b: &[f32; 6]) -> [f32; 6] {
    [
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
        a[0] * b[4] + a[2] * b[5] + a[4],
        a[1] * b[4] + a[3] * b[5] + a[5],
    ]
}

impl BrushDef {
    /// The `brush→device` matrix for this brush under device transform `device`.
    fn b2d(&self, device: &[f32; 6], xform: &[f32; 6]) -> [f32; 6] {
        compose_f32(device, xform)
    }
    /// Re-realize the device-space brush under `device` (the draw-time device transform).
    pub(crate) fn realize(&self, device: &[f32; 6]) -> Brush {
        match self {
            BrushDef::Linear { stops, gamma, extend, opacity, start, end, xform } => {
                Brush::linear(stops, *gamma, *extend, *opacity, *start, *end, &self.b2d(device, xform))
            }
            BrushDef::Radial { stops, gamma, extend, opacity, center, radii, offset, xform } => {
                Brush::radial_world(stops, *gamma, *extend, *opacity, *center, *radii, *offset, &self.b2d(device, xform), device, xform)
            }
            BrushDef::Bitmap { texels, size, extend, interpolation, opacity, xform, dpi } => {
                // Fold the bitmap's texel→DIP scale into brush→device, as `create_bitmap_brush` does, so
                // a replayed non-96-dpi bitmap brush samples the right texels (96-dpi → ×1.0, unchanged).
                let (sx, sy) = (96.0 / dpi.0, 96.0 / dpi.1);
                let b = self.b2d(device, xform);
                let b = [b[0] * sx, b[1] * sx, b[2] * sy, b[3] * sy, b[4], b[5]];
                Brush::bitmap(texels.clone(), *size, &b, *extend, *interpolation, *opacity)
            }
        }
    }
}

fn scale(c: [f32; 4], o: f32) -> [f32; 4] {
    c.map(|v| v * o)
}

/// [`scale`] for a gradient kernel's fractional ramp sample: the JIT multiplies the opacity in the
/// normalised domain, the byte unpacked as `c·(1/256) + c·(1/65280)` then `×o ×255`, which lands a
/// `.5` byte tie a hair high (cpg-lin-rr-b0's 18.125·0.8 → 14.500001, stored 15).
fn scale_jit(c: [f32; 4], o: f32) -> [f32; 4] {
    if o == 1.0 {
        return scale(c, o);
    }
    scale_unpacked(c, o)
}

/// The shader's normalised multiply of a fractional ramp sample by `o`, at any opacity (see [`scale_jit`]).
fn scale_unpacked(c: [f32; 4], o: f32) -> [f32; 4] {
    c.map(|v| ((v * 0.003_906_25 + v * f32::from_bits(0x3780_8081)) * o) * 255.0)
}

/// Source-over a premultiplied float source over a premultiplied BGRA8 destination at full
/// coverage, matching `crate::pixels`: the shader source is quantised to UNORM8 (WARP converts the
/// pixel-shader output to the render-target format before blending), then composited with a final
/// round-to-nearest-even store.
fn over(dst: [u8; 4], src: [f32; 4]) -> [u8; 4] {
    let s = src.map(|v| v.round_ties_even().clamp(0.0, 255.0));
    // Opaque source: `s[3] == 255` makes `sa == 1`, so every `dst·(1 − sa)` term is exactly 0 and the
    // result is the rounded source, independent of dst. Skip the per-channel destination arithmetic.
    // (`s` is already integral in `[0, 255]`, so `as u8` is exact.) Byte-identical.
    if s[3] == 255.0 {
        return [s[0] as u8, s[1] as u8, s[2] as u8, 255];
    }
    let sa = s[3] / 255.0;
    core::array::from_fn(|i| (s[i] + (dst[i] as f32 - dst[i] as f32 * sa)).round_ties_even().clamp(0.0, 255.0) as u8)
}

/// SOURCE_OVER through WARP's output merger (ONE, INV_SRC_ALPHA) at full coverage: the UNORM8 source plus `d·ia/255`
/// rounded as `(t + (t>>8) + 128) >> 8`. A MULTI_SAMPLE_LINEAR bitmap brush's rect fill composites so (an MSL rect
/// over a destination alpha of 244: 345 of 345 covered pixels, where [`over`] gets 340).
fn over_om(dst: [u8; 4], src: [f32; 4]) -> [u8; 4] {
    let s = src.map(|v| v.round_ties_even().clamp(0.0, 255.0) as i32);
    core::array::from_fn(|k| {
        let t = dst[k] as i32 * (255 - s[3]);
        (s[k] + ((t + (t >> 8) + 128) >> 8)).clamp(0, 255) as u8
    })
}

/// Non-SOURCE_OVER primitive blend of a premultiplied brush source over a BGRA8 destination at
/// coverage `c ∈ 0..=1`, mirroring the solid rasterizer's per-mode formulas (`pixels::composite`):
/// WARP converts the shader float output to the target UNORM8 format *before* a non-source-over
/// blend, so the float source (already × opacity) is quantised first, then combined per `blend`.
/// Callers keep their own LSB-exact source-over path and only route the other modes here.
/// ponytail: COPY/MIN/ADD/MAX brush fringe has no oracle scene — same ceiling `pixels::composite`
/// already documents (aliased is exact; the coverage-weighted fringe is the natural lerp).
pub(crate) fn blend_src(dst: [u8; 4], src: [f32; 4], c: f32, blend: Blend) -> [u8; 4] {
    let re = |v: f32| v.round_ties_even().clamp(0.0, 255.0) as u8;
    let s8 = src.map(|v| v.round_ties_even().clamp(0.0, 255.0));
    let cs: [f32; 4] = core::array::from_fn(|i| s8[i] * c);
    match blend {
        Blend::Copy => core::array::from_fn(|i| re(cs[i] + dst[i] as f32 * (1.0 - c))),
        // MIN composites the float source over WHITE before the UNORM store: `min(dst, src + 1 − src.a)`.
        Blend::Min => core::array::from_fn(|i| re(src[i] * c + (255.0 - src[3] * c)).min(dst[i])),
        Blend::Add => core::array::from_fn(|i| (re(cs[i]) as u16 + dst[i] as u16).min(255) as u8),
        Blend::Max => core::array::from_fn(|i| re(cs[i]).max(dst[i])),
        // Callers handle SOURCE_OVER themselves; the 8-bit premul form keeps this total (no panic).
        Blend::SourceOver => core::array::from_fn(|i| re(cs[i] + dst[i] as f32 * (1.0 - cs[3] / 255.0))),
    }
}

/// Composite one brush source sample under rect coverage `cov` (64 = full): full pixels take the
/// LSB-exact `over`/`blend_src`; a fractional edge the UNORM8 source times the 8-bit coverage, as
/// [`fill_geometry`] does.
fn cov_composite(dst: [u8; 4], src: [f32; 4], cov: u32, blend: Blend) -> [u8; 4] {
    if cov >= 64 {
        if blend == Blend::SourceOver { over(dst, src) } else { blend_src(dst, src, 1.0, blend) }
    } else if matches!(blend, Blend::SourceOver | Blend::Copy | Blend::Add) {
        let cov8 = (cov as f32 * 255.0 / 64.0).round_ties_even() as i32;
        let sc: [i32; 4] = core::array::from_fn(|k| {
            let p = src[k].round_ties_even().clamp(0.0, 255.0) as i32 * cov8;
            (p + (p >> 8) + 128) >> 8
        });
        // ADD sums that same 8-bit coverage-weighted source (an image-brush rect fringe under PLUS, sc64 g-*-c9).
        if blend == Blend::Add {
            return core::array::from_fn(|k| (sc[k] + dst[k] as i32).min(255) as u8);
        }
        // COPY keeps the destination's uncovered share instead of its source-alpha share (winv
        // nrm-*-b1: a 3/8 Flood edge lands 151, not the float lerp's 150).
        let ia = 255 - if blend == Blend::Copy { cov8 } else { sc[3] };
        core::array::from_fn(|k| (sc[k] + crate::bitmap::unorm_mul(dst[k] as i32, ia)).clamp(0, 255) as u8)
    } else {
        blend_src(dst, src, cov as f32 / 64.0, blend)
    }
}

/// A bitmap rect fill's pixel span and coverage: the rounded rect at full coverage, or under
/// `Aa::PerPrimitive` with a fractional edge every pixel [`rasterize_rect`]'s eighths touch.
struct BitmapRectCov {
    span: [usize; 4],
    map: Option<Vec<u32>>,
}

impl BitmapRectCov {
    fn new(rect: [f32; 4], aa: Aa, w: usize, h: usize, rounded: [usize; 4]) -> Self {
        if aa != Aa::PerPrimitive || rect.iter().all(|v| v.fract() == 0.0) {
            return Self { span: rounded, map: None };
        }
        let mut px = Vec::new();
        rasterize_rect(rect, Aa::PerPrimitive, w, h, |x, y, c| px.push((x, y, c)));
        let Some(&(fx, fy, _)) = px.first() else { return Self { span: [0; 4], map: None } };
        let span = px.iter().fold([fx, fy, fx + 1, fy + 1], |s, &(x, y, _)| [s[0].min(x), s[1].min(y), s[2].max(x + 1), s[3].max(y + 1)]);
        let mut map = vec![0; (span[2] - span[0]) * (span[3] - span[1])];
        for (x, y, c) in px { map[(y - span[1]) * (span[2] - span[0]) + x - span[0]] = c; }
        Self { span, map: Some(map) }
    }

    /// The fully covered columns and rows `[x0, y0, x1, y1]` inside the span (empty past the span when
    /// there are none), read across the middle row and column.
    fn inner(&self) -> [usize; 4] {
        let Some(m) = &self.map else { return self.span };
        let [x0, y0, x1, y1] = self.span;
        let (w, h) = (x1 - x0, y1 - y0);
        let (mx, my) = (w / 2, h / 2);
        let col = |x: usize| m[my * w + x] == 64;
        let row = |y: usize| m[y * w + mx] == 64;
        let (cx0, cx1) = match ((0..w).find(|&x| col(x)), (0..w).rev().find(|&x| col(x))) {
            (Some(a), Some(b)) => (x0 + a, x0 + b + 1),
            _ => (x1, x1),
        };
        let (cy0, cy1) = match ((0..h).find(|&y| row(y)), (0..h).rev().find(|&y| row(y))) {
            (Some(a), Some(b)) => (y0 + a, y0 + b + 1),
            _ => (y1, y1),
        };
        [cx0, cy0, cx1, cy1]
    }

    fn put(&self, buf: &mut PixelBuffer, x: usize, y: usize, src: [f32; 4], blend: Blend) {
        let c = self.map.as_ref().map_or(64, |m| m[(y - self.span[1]) * (self.span[2] - self.span[0]) + x - self.span[0]]);
        if c == 0 { return; }
        let i = y * buf.stride + x * 4;
        let dst = buf.data[i..i + 4].try_into().unwrap();
        buf.data[i..i + 4].copy_from_slice(&cov_composite(dst, src, c, blend));
    }
}

/// Fill a device-space rectangle with `brush` under antialias mode `aa`. The pixel-aligned interior
/// is full coverage (the LSB-exact single-round `over`/`blend_src`); with `Aa::PerPrimitive` a
/// fractional edge is antialiased by the shared 8×8 scan converter ([`rasterize_rect`]) and composited
/// coverage-weighted exactly as [`fill_geometry`] does (native converts the shader output to UNORM8,
/// then multiplies by the 8-bit coverage). `Aa::Aliased` keeps the pixel-centre round-to-bounds fill.
/// The bitmap fast paths below sample the same way; with `Aa::PerPrimitive` and a fractional edge they
/// span every pixel the eighths edges touch, each weighted by its coverage (bbaa-*/ibaa-*, r6 6875).
pub fn fill_rect(buf: &mut PixelBuffer, rect: [f32; 4], brush: &Brush, blend: Blend, aa: Aa) {
    // Whether every unclipped device edge sits within 1/32 of a whole pixel: part of d2d1's gate for drawing an image
    // brush's rect through its effect graph (below).
    let px_edges = rect.iter().all(|v| (v.round() - v).abs() < 1.0 / 32.0);
    let unclipped = rect;
    // A LARGE anti-aliased rect fills by the plain 8×8 scan, not the COPY trapezoid walk: native's rect
    // path leaves the small-rect walk once the SNAPPED interior area exceeds 47863 px² —
    // (floor(right) − ceil(left)) · (floor(bottom) − ceil(top)), on the UNCLIPPED device rect (so compute
    // it here, before the clip below). VM-probed: the big-rect edge is the 8×8 eighths scan (left frac
    // .0625→full, .1875→7/8, .3125→6/8, .4375→5/8, .5625→4/8), whereas the small rect's COPY walk is the
    // 1/16 trapezoid coverage. So a big COPY radial/CUBIC/MSL rect skips the walk and uses the scan below.
    let big = {
        let iw = rect[2].floor() - rect[0].ceil();
        let ih = rect[3].floor() - rect[1].ceil();
        iw > 0.0 && ih > 0.0 && iw * ih > 47863.0
    };
    let whole = rect;
    // Native clips the rectangle before realizing the sampling atlas and fitting its quad.
    let rect = [rect[0].max(0.0), rect[1].max(0.0), rect[2].min(buf.desc.width as f32), rect[3].min(buf.desc.height as f32)];
    if rect[2] <= rect[0] || rect[3] <= rect[1] { return; }
    if let Some(tiles) = brush.split_rect(rect) {
        for tile in tiles { fill_rect(buf, tile, brush, blend, aa); }
        return;
    }
    let (w, h, stride) = (buf.desc.width as usize, buf.desc.height as usize, buf.stride);
    // WARP's aliased fill rule: snap each edge to the 24.8 device grid (round-half-up) and cover a
    // pixel iff its centre is inside the snapped span — the first covered column is the smallest `x`
    // with centre `256·x+128 ≥ edge_24_8`. This is `f32::round` of the raw edge everywhere except an
    // edge landing within 1/256 above a pixel centre, where `round` snaps it to the next pixel and
    // drops the column native keeps (clal-d: a 2.50195 edge covers x=2, not x=3).
    let aa_edge = |v: f32| -> usize {
        let e = (v * 256.0 + 0.5).floor(); // edge in 24.8 fixed point, half-up
        (((e - 128.0) / 256.0).ceil().max(0.0) as usize).min(w.max(h))
    };
    // An aliased HIGH_QUALITY_CUBIC rect that is not blitted from its atlas (below) draws its last pass straight into
    // the target over its aliased pixel span, and realizes its intermediates for that span, not for the rect's whole-
    // pixel bounds (DDI of aliased and antialiased fractional HQ rects: the aliased last pass has the target viewport,
    // its quad the span and its first pass's quad the span's taps; VM probes geo/hqr, geo/hqx).
    let hq_span = (aa == Aa::Aliased && matches!(brush, Brush::Bitmap { interpolation: 5, opacity, .. }
        if *opacity >= 1.0 || !matches!(blend, Blend::SourceOver | Blend::Copy)))
        .then(|| [aa_edge(rect[0]).min(w), aa_edge(rect[1]).min(h), aa_edge(rect[2]).min(w), aa_edge(rect[3]).min(h)].map(|v| v as f32));
    let prepared = brush.prepare_rect(hq_span.unwrap_or(rect));
    let brush = prepared.as_ref().unwrap_or(brush);
    // A MULTI_SAMPLE_LINEAR bitmap brush's covered pixels are composited by the output merger.
    let om = blend == Blend::SourceOver && matches!(brush, Brush::Bitmap { interpolation: 3, .. });
    // Under COPY a radial gradient and a CUBIC or MULTI_SAMPLE_LINEAR bitmap cover the rect as any COPY primitive does,
    // by the trapezoid walk, not the eighths or the 8x8 scan (probe-crs: SOURCE_OVER and COPY rects with fractional
    // edges). NEAREST and LINEAR keep their AlphaBlt fills; ANISOTROPIC and HIGH_QUALITY_CUBIC, drawn from their
    // intermediates, keep the eighths (probe-crh).
    if blend == Blend::Copy && aa == Aa::PerPrimitive && !big
        && (matches!(brush, Brush::Radial { .. }) || matches!(brush, Brush::Bitmap { interpolation: 2 | 3, .. })) {
        // The walk traces the unclipped rect; the target clips its triangles (VM DDI of a radial COPY rect past the
        // target's right edge: its right rail at 96.875 on a 96 px target).
        let poly = [[whole[0], whole[1]], [whole[2], whole[1]], [whole[2], whole[3]], [whole[0], whole[3]]];
        crate::pixels::copy_tess::fill_copy(&[&poly], true, w, h, true, |x, y, cov, tri| {
            // The walk is a triangle draw whose vertex shader maps each vertex through the radial's device-to-unit
            // matrix, so the sample interpolates over each walk triangle (VM DDI of r12 17021's fill: 24 vertices,
            // the matrix in the VS constants), as the geometry walk's does.
            let s = if matches!(brush, Brush::Radial { .. }) {
                triangle_brush_sampler(tri, brush, false, [0.0, 0.0, w as f32, h as f32])(x, y)
            } else {
                brush.source_at_rect(x as f32 + 0.5, y as f32 + 0.5, false, rect)
            }.map(|v| v.round_ties_even().clamp(0.0, 255.0) as u8);
            let i = y * stride + x * 4;
            let dst: [u8; 4] = buf.data[i..i + 4].try_into().unwrap();
            buf.data[i..i + 4].copy_from_slice(&cov.lerp(dst, s));
        });
        return;
    }
    let x0 = aa_edge(rect[0]).min(w);
    let y0 = aa_edge(rect[1]).min(h);
    let x1 = aa_edge(rect[2]).min(w);
    let y1 = aa_edge(rect[3]).min(h);
    // The bitmap paths' per-pixel coverage: full over the (24.8-snapped) aliased rect, or — with a
    // PerPrimitive fractional edge — the AA rect's eighths, which `BitmapRectCov` computes itself.
    let cov = BitmapRectCov::new(rect, aa, w, h, [x0, y0, x1, y1]);
    let [x0, y0, x1, y1] = cov.span;
    // Axis-aligned nearest/linear byte brushes use WARP's 16.16 attribute DDA. Nearest removes
    // the bilinear half-texel bias before selecting the texel; float coordinates shift wrap seams.
    // A sub-rect bitmap brush has no blit: its rect is drawn by the bitmap brush's pixel shader (the generic fill below).
    let clamped = matches!(brush, Brush::Bitmap { clamp: Some(_), .. });
    if let (false, Brush::Bitmap { bitmap, to_source: ts, transform, extend, filter, interpolation, opacity, texture_size, shader_draw, cell, cell_size, vs_row, .. }) = (clamped, brush) {
        // An image brush's atlas draws through the pixel shader below on d2d1's effect-graph route for a rect: full
        // opacity, an aliased fill or whole-pixel edges, and SOURCE_OVER, COPY or ADD. Any other fill redirects the brush
        // into a bitmap, which WARP blits with a byte opacity under SOURCE_OVER or COPY (VM DDI: a fractional
        // per-primitive fill issues no draw for the rect, an aliased one a pixel-shader draw). Under MIN, ADD and MAX it
        // is drawn, through the pixel shader below, whatever the brush's rotation or opacity.
        let graph = *opacity == 1.0 && (aa == Aa::Aliased || px_edges) && matches!(blend, Blend::SourceOver | Blend::Copy | Blend::Add);
        let blit = matches!(blend, Blend::SourceOver | Blend::Copy);
        let shader = *shader_draw && *interpolation == 1 && *filter == Filter::Linear && !crate::float::is_highp(bitmap.format)
            && (graph || !blit);
        // A texcoord start past 16.16 (2^15 texels) leaves the DDA for the float alpha-blit (VM JIT and probes of a
        // 16×10 brush offset 1e9).
        let far = [ts[0] * rect[0] + ts[4], ts[0] * rect[2] + ts[4], ts[3] * rect[1] + ts[5], ts[3] * rect[3] + ts[5]]
            .iter().any(|u| u.is_finite() && u.abs() >= 32768.0);
        if !shader && (ts[1] != 0.0 || ts[2] != 0.0 || far) && *interpolation <= 1 && matches!(filter, Filter::Linear | Filter::Nearest) && !crate::float::is_highp(bitmap.format) {
            let tex = Texture { data: &bitmap.premul_bgra, width: bitmap.width, height: bitmap.height };
            let o8 = (opacity.clamp(0.0, 1.0) * 255.0).round_ties_even() as i32;
            crate::sampling::alphablt_float_iter_ex(ts, *texture_size, [bitmap.width, bitmap.height], [x0, y0, x1, y1], cov.inner(), *extend, *filter == Filter::Nearest, buf.desc.height, |x, y, qx, qy| {
                let s = tex.sample_alphablt_q8(*extend, qx, qy, *filter == Filter::Nearest);
                let src = s.map(|v| crate::layer::unorm_mul(v, o8) as f32);
                cov.put(buf, x, y, src, blend);
            });
            return;
        }
        // A bitmap image brush, realized into a power-of-two cells atlas, is drawn by a pixel shader:
        // the quad over the rect is split (LT, RB, RT) / (LB, LT, RB), each triangle fits its texcoord
        // plane in atlas coordinates (the atlas starts at the need region, `RoundSat(u(edge+½) − 1)`),
        // the address floors, the opacity multiplies in float and the UNORM store truncates (WRAP image
        // brushes over a bitmap: imgbrush-bm-*, traced against the final draw's post-VS attributes). A rotated brush
        // anchors the atlas at its need region's minimum corner (probe-ibb, probe-ibc).
        if shader {
            let tex = Texture { data: &bitmap.premul_bgra, width: bitmap.width, height: bitmap.height };
            let q = |v: f32| { let q = ((v - 0.5) * 256.0).floor() as i32; (q >> 8, q & 255) };
            let rs = |v: f32| { let r = v.round_ties_even(); if r - v == -0.5 { r + 1.0 } else { r } };
            let at = |x: f32, y: f32| [ts[0] * x + (ts[2] * y + ts[4]), ts[1] * x + (ts[3] * y + ts[5])];
            let cs = [at(x0 as f32 + 0.5, y0 as f32 + 0.5), at(x1 as f32 - 0.5, y0 as f32 + 0.5), at(x0 as f32 + 0.5, y1 as f32 - 0.5), at(x1 as f32 - 0.5, y1 as f32 - 0.5)];
            let n0 = [0, 1].map(|k| rs(cs.iter().map(|c| c[k]).fold(f32::INFINITY, f32::min) - 1.0));
            let (l, t, r, b) = (x0 as i32, y0 as i32, x1 as i32, y1 as i32);
            let uv = |x: i32, y: i32| { let p = at(x as f32, y as f32); [p[0] - n0[0], p[1] - n0[1]] };
            let planes = [[(l, t), (r, b), (r, t)], [(l, b), (l, t), (r, b)]].map(|c| {
                let xs = c.map(|(x, _)| x as i64 * 256);
                let ys = c.map(|(_, y)| y as i64 * 256);
                [0, 1].map(|ax| crate::warp::plane::setup_triangle_attr(xs, ys, c.map(|(x, y)| uv(x, y)[ax])))
            });
            let (qw, qh) = ((r - l) as f32, (b - t) as f32);
            let sample = |x: usize, y: usize| -> [f32; 4] {
                let p = [x as f32 + 0.5, y as f32 + 0.5];
                let k = usize::from((p[1] - t as f32) * qw > (p[0] - l as f32) * qh);
                let [tx, ty] = planes[k].map(|a| a[0] * p[0] + (a[1] * p[1] + a[2]));
                let ((ix, wx), (iy, wy)) = (q(tx), q(ty));
                // Off the graph under ADD or MAX the redirected bitmap's rectangle is one WARP geometry draw whose texcoord is
                // the atlas row fitted once over the unit triangle, not per rect triangle (JIT plane block of a rotated ADD
                // fill: every call carries the same six coefficients, bit-identical to `atlas_cell_q8`'s fit). MIN walks.
                let (qx, qy) = match (graph || blend == Blend::Min, cell, vs_row) {
                    (false, Some(c), Some(row)) => {
                        let [a, b] = atlas_cell_q8(row, *c, *cell_size as f32, p[0], p[1], [0.0; 2]);
                        (a, b)
                    }
                    _ => ((ix + n0[0] as i32, wx), (iy + n0[1] as i32, wy)),
                };
                core::array::from_fn(|ch| crate::sampling::norm_v128(tex.bilinear_v128(*extend, ch, qx, qy)) * opacity.clamp(0.0, 1.0) * 255.0)
            };
            // Off the effect graph (MIN, ADD, MAX) the redirected bitmap is drawn as a rectangle shape. Under MIN that shape
            // takes the trapezoid walk and lerps the unquantised source toward white ([`fill_geometry_cov`]'s rule; p28w).
            if !graph && blend == Blend::Min && aa == Aa::PerPrimitive && !big {
                let poly = [[unclipped[0], unclipped[1]], [unclipped[2], unclipped[1]], [unclipped[2], unclipped[3]], [unclipped[0], unclipped[3]]];
                let vp = [0.0, 0.0, w as f32, h as f32];
                crate::pixels::copy_tess::fill_copy(&[&poly], true, w, h, true, |x, y, c, tri| {
                    let i = y * stride + x * 4;
                    let dst: [u8; 4] = buf.data[i..i + 4].try_into().unwrap();
                    // The walk is a DDI draw whose vertex shader maps each vertex through the brush's atlas row (VS constant
                    // rows of a MIN image-brush rect's walk draw, SRV the 512 atlas), so each walk triangle fits its own
                    // texcoord plane, as a geometry walk's bitmap brush does.
                    let s = match (cell, vs_row) {
                        (Some(cl), Some(row)) => {
                            let at = |q: [[f32; 2]; 3]| Some(q.map(|v| [row[0] * v[0] + (row[2] * v[1] + row[4]), row[1] * v[0] + (row[3] * v[1] + row[5])]));
                            let (xc, yc) = (x as f32 + 0.5, y as f32 + 0.5);
                            match triangle_attr_planes(tri, at, vp)(xc, yc) {
                                Some(pl) => {
                                    let [qx, qy] = [0, 1].map(|k| {
                                        let u = xc * pl[k][0] + (yc * pl[k][1] + pl[k][2]);
                                        let q = ((u * *cell_size as f32 - 0.5) * 256.0).floor() as i32;
                                        ((q >> 8) - cl[k], q & 255)
                                    });
                                    core::array::from_fn(|ch| crate::sampling::norm_v128(tex.bilinear_v128(*extend, ch, qx, qy)) * opacity.clamp(0.0, 1.0) * 255.0)
                                }
                                None => sample(x, y),
                            }
                        }
                        _ => sample(x, y),
                    };
                    buf.data[i..i + 4].copy_from_slice(&c.min_white_f(dst, s));
                });
                return;
            }
            for y in y0..y1 {
                for x in x0..x1 {
                    // The effect graph's store truncates; off it, the redirected bitmap's pixel shader truncates at full
                    // opacity and keeps the opacity-scaled sample to round in the blend otherwise (the bitmap brush's).
                    let s = sample(x, y);
                    cov.put(buf, x, y, if graph || *opacity >= 1.0 { s.map(f32::floor) } else { s }, blend);
                }
            }
            return;
        }
        if ts[1] == 0.0 && ts[2] == 0.0 && *interpolation <= 1 && matches!(filter, Filter::Linear | Filter::Nearest) && !crate::float::is_highp(bitmap.format) {
            let tex = Texture { data: &bitmap.premul_bgra, width: bitmap.width, height: bitmap.height };
            // WARP modulates the sampled byte by the opacity byte (imgbrush-04).
            let o8 = (opacity.clamp(0.0, 1.0) * 255.0).round_ties_even() as i32;
            // Extreme scale (fuzz 22998): the 16.16 texcoord DDA step `sx` overflows i32 over the fill
            // extent (`(1/s)·extent·2^16 ≥ 2^31`), so WARP rasterizes the f32 attribute plane in 8-ROW
            // TILES with an incremental per-span DDA (jit-dump `bb` capture, probebm 1e-6..1e-3): X
            // accumulates 4-px SIMD blocks from `coeff·½`, Y re-bases each 8-row tile to `coeff·8·tile +
            // coeff·½` then accumulates per row. The plane is WARP's attribute fit over the actual fill-rect
            // triangle (`bitmap_quad_at`). NEAREST only; the moderate (fixed-point) regime keeps the 16.16
            // DDA below (byte-exact). ponytail: switch = the i32-accumulator-overflow condition; the exact
            // scale at which the brush realization stores its texcoord step as an f32 plane rather than a
            // 16.16 integer DDA is capture-groundable and the probes only bracket it to s∈(1e-3,1e-2) — pin
            // it if a mid-scale scene ever needs it.
            let ext = (rect[2] - rect[0]).max(rect[3] - rect[1]);
            if *filter == Filter::Nearest && ts[0].abs().max(ts[3].abs()) * ext >= 32768.0 {
                // Normalized texcoord gradient = `(1/s)·(1/size)` (matches the captured plane coeff bit-exact
                // at the true-extreme scales; a 1-ULP error there compounds over the 8-row-tile Y re-base and
                // wrecks the whole map). base = coeff·½ + the normalized device-origin offset (captured
                // base = coeff·½ exactly for a zero-offset brush).
                let (bw, bh) = (bitmap.width as f32, bitmap.height as f32);
                let (px0, py0) = (rect[0].floor(), rect[1].floor());
                let (cu, cv) = ((1.0 / transform[0]) * (1.0 / bw), (1.0 / transform[3]) * (1.0 / bh));
                let base_u = cu * (px0 + 0.5) + ts[4] * (1.0 / bw);
                let base_v = cv * (py0 + 0.5) + ts[5] * (1.0 / bh);
                let pow2 = [bitmap.width.is_power_of_two(), bitmap.width.is_power_of_two() && bitmap.height.is_power_of_two()];
                let idx = |t: f32, n: u32, e: Extend, p2: bool| -> i32 {
                    let nf = n as f32;
                    let raw = if p2 {
                        let t = if matches!(e, Extend::Wrap | Extend::Mirror) { t } else { t.clamp(-10.0, 10.0) };
                        (t * nf).floor() as i32
                    } else {
                        (match e {
                            Extend::Wrap => (t - t.floor()) * nf,
                            Extend::Mirror => { let m = t * 0.5; (m - m.floor()) * (2.0 * nf) }
                            _ => t.clamp(-10.0, 10.0) * nf,
                        }).floor() as i32
                    };
                    fold(e, raw, n as i32)
                };
                for y in y0..y1 {
                    let v = crate::warp::plane::tiled_dda_y(cv, base_v, y as i32 - py0 as i32);
                    let iy = idx(v, bitmap.height, extend[1], pow2[1]);
                    for x in x0..x1 {
                        let u = crate::warp::plane::tiled_dda_x(cu, base_u, x as i32 - px0 as i32);
                        let ix = idx(u, bitmap.width, extend[0], pow2[0]);
                        let src = tex.sample(Filter::Nearest, [Extend::Clamp; 2], ix as f32 + 0.5, iy as f32 + 0.5)
                            .map(|v| crate::layer::unorm_mul((v * 255.0).round() as i32, o8) as f32);
                        cov.put(buf, x, y, src, blend);
                    }
                }
                return;
            }
            let (bx, sx) = crate::sampling::axis_fixed(ts[0] * rect[0] + ts[4], ts[0] * rect[2] + ts[4], rect[0], rect[2]);
            let (by, sy) = crate::sampling::axis_fixed(ts[3] * rect[1] + ts[5], ts[3] * rect[3] + ts[5], rect[1], rect[3]);
            for y in y0..y1 {
                // WARP's 16.16 texcoord DDA is 32-bit and wraps; a degenerate brush matrix makes the
                // fixed step huge, so wrap (matches hardware, no debug overflow panic — fuzz seed 800).
                // Past 2¹⁵ texels the wrap diverges from native's cvttss2si (origin on overflow); fix it
                // with the float texcoord through [`dda_texcoord`].
                let ufy = ts[3] * (y as f32 + 0.5) + ts[5];
                let uy = dda_texcoord(ufy, by.wrapping_add((y as i32).wrapping_mul(sy)), extend[1], tex.height as i32, *filter == Filter::Linear);
                for x in x0..x1 {
                    let ufx = ts[0] * (x as f32 + 0.5) + ts[4];
                    let ux = dda_texcoord(ufx, bx.wrapping_add((x as i32).wrapping_mul(sx)), extend[0], tex.width as i32, *filter == Filter::Linear);
                    let src = if *filter == Filter::Nearest {
                        tex.sample(Filter::Nearest, *extend, (ux.wrapping_add(0x7f80) >> 16) as f32, (uy.wrapping_add(0x7f80) >> 16) as f32)
                    } else {
                        tex.sample_fixed(*extend, ux, uy)
                    }.map(|v| crate::layer::unorm_mul((v * 255.0).round() as i32, o8) as f32);
                    cov.put(buf, x, y, src, blend);
                }
            }
            return;
        }
    }
    // A linear gradient's SOURCE_OVER/COPY rect fast path draws its ramp as a one-row bitmap through WARP's float
    // alpha-blit (the VM JIT of r12 17657's rect fill: the blit kernel over the ramp texture), so the texel address
    // takes the blit's lanes and spans, and the opacity modulates the byte, as a bitmap brush's rect does.
    if let Brush::Linear { grad, xform, opacity, .. } = brush {
        if matches!(blend, Blend::SourceOver | Blend::Copy) {
            let tex = Texture { data: &grad.texels, width: grad.width as u32, height: 1 };
            let ex = [extend_of(grad.extend), Extend::Clamp];
            let o8 = (opacity.clamp(0.0, 1.0) * 255.0).round_ties_even() as i32;
            let ts = [xform[0], 0.0, xform[1], 0.0, xform[2], 0.5];
            crate::sampling::alphablt_float_iter_ex(&ts, [grad.width as u32, 1], [grad.width as u32, 1], [x0, y0, x1, y1], cov.inner(), ex, false, buf.desc.height, |x, y, qx, qy| {
                let s = tex.sample_alphablt_q8(ex, qx, qy, false);
                cov.put(buf, x, y, s.map(|v| crate::layer::unorm_mul(v, o8) as f32), blend);
            });
            return;
        }
    }
    // A HIGH_QUALITY_CUBIC rect is drawn from its realized intermediate at full opacity and blitted with the opacity as
    // a byte, as a bitmap rect's AlphaBlt is (DDI: the last cubic pass renders into the atlas, not the target; VM probes
    // of HQ rect fills below opacity 1, aliased and per-primitive, ±1 on 2-397 px with the opacity in float).
    let hq_blit = match brush {
        Brush::Bitmap { interpolation: 5, opacity, intermediate_precision, .. } if *opacity < 1.0 && matches!(blend, Blend::SourceOver | Blend::Copy) => {
            let mut unit = brush.clone();
            if let Brush::Bitmap { opacity, .. } = &mut unit { *opacity = 1.0; }
            Some((unit, (opacity.clamp(0.0, 1.0) * 255.0).round() as i32, *intermediate_precision, opacity.clamp(0.0, 1.0)))
        }
        _ => None,
    };
    // An ANISOTROPIC rect below full opacity, or an antialiased one with a fractional edge, is drawn into an atlas
    // intermediate whose origin is the rect's floored corner, so its texcoord planes are fitted and evaluated in that
    // frame; any other draws into the target in the device frame (VM DDI and plane captures of axis-aligned, skewed,
    // quarter-turned and rotated anisotropic rects, whole-pixel and fractional, aliased and antialiased, at opacity 1 and
    // 0.5: the atlas viewport and its corner's constants for exactly those, the target otherwise).
    let frame = if matches!(brush, Brush::Bitmap { interpolation: 4, opacity, .. }
        if *opacity < 1.0 || (aa == Aa::PerPrimitive && rect.iter().any(|v| v.fract() != 0.0))) {
        [rect[0].floor(), rect[1].floor()]
    } else {
        [0.0; 2]
    };
    let rect_src = |x: usize, y: usize| match &hq_blit {
        // Above 8bpc the intermediate holds the sample at its precision, which the fill scales by the opacity in float
        // and rounds once (geo/c641c: rect fills at precision 2, 3 and 4, 799-941 px through the byte opacity, exact).
        Some((unit, _, p, op)) if *p >= 2 => unit.source_at_rect(x as f32 + 0.5, y as f32 + 0.5, false, rect)
            .map(|v| (v * op).round_ties_even().clamp(0.0, 255.0)),
        Some((unit, o8, _, _)) => unit.source_at_rect(x as f32 + 0.5, y as f32 + 0.5, false, rect)
            .map(|v| crate::layer::unorm_mul(v.round_ties_even().clamp(0.0, 255.0) as i32, *o8) as f32),
        None => brush.source_at_rect_framed(x as f32 + 0.5, y as f32 + 0.5, clamped || matches!(blend, Blend::Min | Blend::Add | Blend::Max), hq_span.unwrap_or(rect), frame),
    };
    if aa == Aa::PerPrimitive {
        // Antialiased edges: the shared 8×8 rasterizer gives per-pixel samples (64 = full). Full-
        // coverage pixels keep the LSB-exact `over`/`blend_src` path (byte-identical to the aliased
        // interior, so integer-aligned rects are unchanged); fractional edges composite the UNORM8
        // brush source times the 8-bit coverage, the same arithmetic as `fill_geometry`.
        let put = |x: usize, y: usize, cov: u32| {
            // Native's round-up bitmap-brush rect fast path serves SOURCE_OVER and COPY (both are a
            // straight brush write); the accumulating/comparing blends (MIN/ADD/MAX) take the general
            // gradient pixel shader, which truncates. So truncate (`sample_jit`) only for those three —
            // inv-addgrad (ADD) is +1 with the round-up sampler, inv-brush-b1 (COPY) is −1 with truncate.
            let src = rect_src(x, y);
            let i = y * stride + x * 4;
            let dst: [u8; 4] = buf.data[i..i + 4].try_into().unwrap();
            // A radial gradient's COPY fringe lerps the target toward the sample by the 8-bit coverage in one rounding, as
            // its COPY walk does on a small rect (r18 23725: a big rect's 6/8 top row; VM probe with the row fully covered).
            let out = if blend == Blend::Copy && cov < 64 && matches!(brush, Brush::Radial { .. }) {
                let cov8 = (cov as f32 * 255.0 / 64.0).round_ties_even() as i32;
                crate::pixels::copy_tess::lerp8(dst, src.map(|v| v.round_ties_even().clamp(0.0, 255.0) as u8), cov8)
            } else if om && cov >= 64 {
                over_om(dst, src)
            } else {
                cov_composite(dst, src, cov, blend)
            };
            buf.data[i..i + 4].copy_from_slice(&out);
        };
        // A radial gradient's rect is drawn by its pixel shader under the 8×8 scan, not the eighths of
        // the rect path that solid, linear and bitmap brushes take (r27 32574: a 7.688 edge covers 2/8
        // under a radial brush, 3/8 under a solid or linear one).
        if matches!(brush, Brush::Radial { .. }) {
            let [l, t, r, b] = rect;
            let edges = [[l, t, r, t], [r, t, r, b], [r, b, l, b], [l, b, l, t]];
            crate::pixels::rasterize_edges(&edges, FillMode::Winding, Aa::PerPrimitive, w, h, put);
        } else {
            rasterize_rect(rect, Aa::PerPrimitive, w, h, put);
        }
        return;
    }
    for y in y0..y1 {
        for x in x0..x1 {
            // MIN/ADD/MAX gradient rect fills take the truncating gradient PS; SOURCE_OVER/COPY keep the
            // round-up bitmap-brush rect fast path — see the PerPrimitive branch above (inv-addgrad/b1).
            let src = rect_src(x, y);
            let i = y * stride + x * 4;
            let dst = buf.data[i..i + 4].try_into().unwrap();
            let out = if om { over_om(dst, src) } else if blend == Blend::SourceOver { over(dst, src) } else { blend_src(dst, src, 1.0, blend) };
            buf.data[i..i + 4].copy_from_slice(&out);
        }
    }
}

/// Fill the device-space `figures` (closed polylines) under `winding`/`aa` with `brush`,
/// source-over. Coverage is the native 8×8 scan conversion — produced by the *shared* pixel
/// rasterizer (`crate::pixels`) so there is only one implementation of it: the figures are filled
/// opaque-white into a zeroed scratch, whose read-back alpha is exactly the 8-bit coverage a solid
/// fill would use. Each covered pixel then gets `coverage · brushSource` (the brush's per-pixel
/// premultiplied source quantised to UNORM8, as native converts the shader output before compositing)
/// source-over the destination — identical arithmetic to the solid path (`pixels::composite`), just
/// with a per-pixel source. `brushes.md` §2: the colour-texturing shader multiplies the brush source
/// by AA coverage, then source-over blends.
/// A figure (flattened polyline) as a clean simple closed loop for [`crate::geometry::fill_tess`]:
/// consecutive duplicates and a repeated closing vertex dropped, `None` if it has fewer than 3
/// distinct vertices or self-intersects (then the caller uses the general scan-converter path; the
/// robust self-intersecting tessellation is a later increment). O(n²) intersection test, capped.
fn simple_closed_loop(fig: &[[f32; 2]]) -> Option<Vec<[f32; 2]>> {
    let mut p: Vec<[f32; 2]> = Vec::with_capacity(fig.len());
    for &v in fig {
        if p.last() != Some(&v) {
            p.push(v);
        }
    }
    if p.first() == p.last() {
        p.pop();
    }
    let n = p.len();
    if n < 3 || n > 4096 {
        return None;
    }
    // Reject self-intersection: any two non-adjacent edges that properly cross.
    let orient = |a: [f32; 2], b: [f32; 2], c: [f32; 2]| {
        ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).partial_cmp(&0.0)
    };
    let seg_cross = |a: [f32; 2], b: [f32; 2], c: [f32; 2], d: [f32; 2]| {
        use core::cmp::Ordering::*;
        match (orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)) {
            (Some(o1), Some(o2), Some(o3), Some(o4)) => {
                (o1 != Equal && o2 != Equal && o1 != o2) && (o3 != Equal && o4 != Equal && o3 != o4)
            }
            _ => false,
        }
    };
    for i in 0..n {
        let (a, b) = (p[i], p[(i + 1) % n]);
        for j in (i + 1)..n {
            // skip adjacent (shared-vertex) edges
            if j == i || (j + 1) % n == i || (i + 1) % n == j {
                continue;
            }
            let (c, d) = (p[j], p[(j + 1) % n]);
            if seg_cross(a, b, c, d) {
                return None;
            }
        }
    }
    Some(p)
}

pub fn fill_geometry(buf: &mut PixelBuffer, figures: &[Vec<[f32; 2]>], winding: bool, aa: Aa, brush: &Brush, blend: Blend) {
    fill_geometry_cov(buf, figures, winding, aa, brush, blend, false, true, None, false)
}

/// [`fill_geometry`] whose aliased coverage can come from WARP's geometry rasterizer
/// ([`crate::pixels::Target::fill_warp28`]) when `warp28` is set.
/// `rect_fixed`: see [`Target::fill_path_in`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_geometry_cov(buf: &mut PixelBuffer, figures: &[Vec<[f32; 2]>], winding: bool, aa: Aa, brush: &Brush, blend: Blend, warp28: bool, rect_fixed: bool, clip: Option<[i32; 4]>, glyphs: bool) {
    // A shape past |524287| is clipped by the safe-box clip before any fill path sees it, the COPY trapezoid
    // walk below included ([`crate::pixels::clip_huge`]).
    let Some(figures) = crate::pixels::clip_huge(figures) else { return };
    let figures = &*figures;
    let bounds = {
        let pts = figures.iter().flatten();
        let lo = |k: usize| pts.clone().map(|p| p[k]).fold(f32::INFINITY, f32::min).floor().max(0.0);
        let hi = |k: usize, n: usize| pts.clone().map(|p| p[k]).fold(f32::NEG_INFINITY, f32::max).ceil().min(n as f32);
        [lo(0), lo(1), hi(0, buf.desc.width as usize), hi(1, buf.desc.height as usize)]
    };
    // An axis-aligned HIGH_QUALITY_CUBIC minification is realized over the fill's bounds before the fill samples it,
    // as the rect path's is ([`Brush::prepare_rect`]): the axis downscale passes, then a cubic resample through the
    // residual transform at the fill's own origin. Without it the sampler took the magnifying cubic (VM probes of
    // minified HQ ellipse fills; DDI: two downscale passes over the fill's bounds, then the 9-tap cubic fill).
    let hq_min = matches!(brush, Brush::Bitmap { interpolation: 5, transform: m, hq: None, .. }
        if m[1] == 0.0 && m[2] == 0.0 && m[0] > 0.0 && m[3] > 0.0 && (m[0] < 1.0 || m[3] < 1.0));
    let hq = if hq_min { brush.prepare_geometry(bounds) } else { brush.prepare_rotated(bounds) };
    let brush = hq.as_ref().unwrap_or(brush);
    // A realization with a residual placement samples at the fill's own origin; any other samples over the fill's bounds.
    let hq_rect = hq_min && !matches!(brush, Brush::Bitmap { hq: Some(c), .. } if c.residual.is_some());
    // Native rasterizes an ALIASED gradient fill of a simple contour as a triangle list (the
    // the scan/tessellation sweep), fitting the gradient texcoord plane PER TRIANGLE at its
    // provoking vertex (`fill_triangle_brush` = WARP's `SetupTriangleAttr`), not from one global
    // plane. For the cases native takes that path (one simple, non-self-intersecting contour),
    // reproduce it byte-exact; otherwise fall through to the shared scan-converter + global sample.
    // A gradient's rectangle shape (`!rect_fixed`: a rect past the rect path that Direct2D fills through its
    // general rasterizer) samples the unit-triangle plane instead (VM captures: r18 23822's 1e9-wide DrawRectangle,
    // one plane for every span, bit for bit `warp_fit_linear`; r17 22111's radial, the same plane for a ±1e9 rect
    // fill, a 1e6-wide aliased DrawRectangle and a small rect fill). An outline glyph run's fill (`glyphs`) samples it
    // too, a one-contour glyph included: Direct2D fills the run's outline on WARP's span path with no triangle set up
    // (VM DDI of r25 30625's 396 px aliased run under a linear brush: no D3D draw; 2 px off through per-triangle planes).
    let shape_plane = (!rect_fixed || glyphs) && matches!(brush, Brush::Linear { .. } | Brush::Radial { .. });
    if aa == Aa::Aliased
        && !warp28
        && !shape_plane
        && blend == Blend::SourceOver
        && matches!(brush, Brush::Linear { .. } | Brush::Radial { .. })
        && figures.len() == 1
    {
        if let Some(poly) = simple_closed_loop(&figures[0]) {
            for tri in crate::geometry::fill_tess::tessellate_simple(&poly) {
                fill_triangle_brush(buf, tri, brush, blend);
            }
            return;
        }
    }
    let fill = if winding { FillMode::Winding } else { FillMode::Alternate };
    // A linear-gradient WARP geometry draw — per-primitive, OR an aliased non-rect fill/stroke through the
    // geometry rasterizer (`warp28`) — fits its texcoord plane once over the unit triangle at the screen
    // origin (traced: SetupTriangleInternal2 sees exactly that triangle for r9 14434's stroke and r5 5379's),
    // as text/COLR fills. An aliased stroke that falls through the per-triangle simple-contour branch (e.g. a
    // self-intersecting widened outline, fuzz r6 6722 op13) took the RAW `xform` here, one 16.16-fit short of
    // native's `warp_fit_linear` (byte-exact to the captured realization plane).
    let unit = match brush {
        Brush::Linear { xform, .. } if aa == Aa::PerPrimitive || warp28 || shape_plane => Some(crate::warp::plane::warp_fit_linear(xform)),
        _ => None,
    };
    let sample = |x: usize, y: usize| match (unit, brush) {
        (Some(pl), Brush::Linear { grad, opacity, .. }) => {
            let (xc, yc) = (x as f32 + 0.5, y as f32 + 0.5);
            let u = xc * pl[0] + (yc * pl[1] + pl[2]);
            let s = if *opacity != 1.0 { grad.sample_jit_f(u) } else { grad.sample(u, true) };
            scale_jit(s, opacity.clamp(0.0, 1.0))
        }
        _ if hq_rect => brush.source_at_rect(x as f32 + 0.5, y as f32 + 0.5, true, bounds),
        // An unrealized HIGH_QUALITY_CUBIC brush (a magnification) renders into an atlas over the fill's bounds, drawn as
        // that quad from the atlas origin with the 9-tap cubic shader, which the fill then samples (VM DDI of r27 32641's
        // rounded-rect stroke: the quad (3,54)-(49,96) at atlas (0,0); geo/c641 strokes 3-11 px and the seed 10 -> 0).
        (_, Brush::Bitmap { interpolation: 5, hq: None, .. }) => brush.source_at_rect_framed(x as f32 + 0.5, y as f32 + 0.5, true, bounds, [bounds[0], bounds[1]]),
        _ => brush.source_at(x as f32 + 0.5, y as f32 + 0.5, std::env::var_os("X_RECT").is_none()),
    };
    let (w, h, stride) = (buf.desc.width as usize, buf.desc.height as usize, buf.stride);
    // A COPY-blend per-primitive convex fill takes Direct2D.s opaque trapezoid coverage, as
    // the solid fill does ([`crate::pixels::copy_tess`]), lerping toward the brush's byte-quantised
    // source per pixel (cpg-*-b1 probes: linear, flood and solid share the same residual).
    // Every COPY fill except a single axis-aligned rect, as the solid one ([`crate::pixels::Target`]'s
    // `fill_copy` route): r16 21296's radial rounded-rect stroke is two rings, which the area path
    // fringed at 0.97 where native's walk gives 0.88. A rect that left the rect path (`!rect_fixed`: a CUBIC brush, a
    // corner past the device grid) is a rectangle shape, which takes the walk too (VM DDI of r22 27012's CUBIC rect
    // fill: the walk's edge slivers and interior as one draw; COPY rect probes with CUBIC brushes and huge corners).
    // MIN takes the walk as COPY does, its coverage lerping the source toward white (`Target`'s fill).
    let single_axis_rect = crate::pixels::is_single_axis_rect(figures);
    if matches!(blend, Blend::Copy | Blend::Min) && aa == Aa::PerPrimitive && (!single_axis_rect || !rect_fixed) {
        walk_fill(buf, figures, fill == FillMode::Alternate, brush, blend, clip);
        return;
    }
    let mut cov = PixelBuffer { desc: buf.desc, stride: buf.stride, data: vec![0u8; buf.data.len()] };
    if warp28 {
        Target::new(&mut cov).fill_warp28(figures, winding, [255, 255, 255, 255], Blend::SourceOver);
    } else {
        Target::new(&mut cov).fill_path_in(figures, fill, [255, 255, 255, 255], aa, Blend::SourceOver, true, rect_fixed);
    }
    for y in 0..h {
        for x in 0..w {
            let i = y * stride + x * 4;
            let cov8 = cov.data[i + 3] as i32; // read-back alpha = 8-bit coverage
            if cov8 == 0 {
                continue;
            }
            let sf = sample(x, y);
            let cov = cov8 as f32 / 255.0;
            let out: [u8; 4] = if blend == Blend::SourceOver {
                // Every brush PS (gradient and bitmap alike) writes its opacity-scaled, premultiplied
                // output as UNORM8 *before* the coverage/output-merger blend — so quantize the source
                // to a byte first, then multiply by the 8-bit coverage with the integer /255 rounding
                // (`(p + (p>>8) + 128) >> 8`, the same tail as DrawBitmap). Keeping the gradient source
                // float through the coverage multiply is a native mismatch: at a partial-coverage pixel
                // `round(float·cov)` differs from `round(round(float)·cov)` (fuzz r4 gradient-over-fill).
                let sc: [i32; 4] = core::array::from_fn(|k| {
                    let p = sf[k].round_ties_even().clamp(0.0, 255.0) as i32 * cov8;
                    (p + (p >> 8) + 128) >> 8
                });
                let ia = 255 - sc[3];
                // A bitmap or gradient brush's draw kernel fuses source-over in the high-byte form,
                // `(s·256 + p + (p >> 8) + 128) >> 8` with `p = d·(255 − sα)` (r5 5782; r12 17066's
                // one-texel radial: d = 191, sα = 2 gives p = 48323 → 189, where the exact /255 rounds to 190).
                if matches!(brush, Brush::Bitmap { .. } | Brush::Linear { .. } | Brush::Radial { .. }) {
                    core::array::from_fn(|k| { let p = buf.data[i + k] as i32 * ia; ((sc[k] << 8) + p + (p >> 8) + 128) >> 8 }.clamp(0, 255) as u8)
                } else {
                    core::array::from_fn(|k| (sc[k] + crate::bitmap::unorm_mul(buf.data[i + k] as i32, ia)).clamp(0, 255) as u8)
                }
            } else {
                blend_src(buf.data[i..i + 4].try_into().unwrap(), sf, cov, blend)
            };
            buf.data[i..i + 4].copy_from_slice(&out);
        }
    }
}

/// A per-primitive fill through the trapezoid walk (Direct2D's own rasterizer, not WARP's tessellator), the
/// row triangles each fitting their own brush plane.
pub(crate) fn walk_fill(buf: &mut PixelBuffer, figures: &[Vec<[f32; 2]>], alternate: bool, brush: &Brush, blend: Blend, clip: Option<[i32; 4]>) {
    if figures.iter().all(|f| f.is_empty()) {
        return;
    }
    let (w, h, stride) = (buf.desc.width as usize, buf.desc.height as usize, buf.stride);
    let slices: Vec<&[[f32; 2]]> = figures.iter().map(|f| f.as_slice()).collect();
    let clip = clip.unwrap_or([0, 0, w as i32, h as i32]);
    let vp = [0.0, 0.0, w as f32, h as f32];
    crate::pixels::copy_tess::fill_copy_in(&slices, alternate, w, h, true, clip, |x, y, cov, tri| {
        let i = y * stride + x * 4;
        // The walk is a triangle draw whose vertex shader maps each vertex to the brush's texcoord, so a gradient's
        // or bitmap brush's texcoord is fitted per walk triangle ([`triangle_brush_sampler`]; VM DDI of r6 6794's COPY
        // fill: a 414-vertex draw with the bitmap brush's device-to-texcoord rows in the VS constants). Other
        // brushes sample their global source.
        let gradient = matches!(brush, Brush::Linear { .. } | Brush::Radial { .. });
        let s = triangle_brush_sampler(tri, brush, false, vp)(x, y);
        let dst: [u8; 4] = buf.data[i..i + 4].try_into().unwrap();
        // A gradient's shader works in 0..1 on the normalized ramp sample (JIT, p224d and the ADD ink kernel).
        let unit = if matches!(blend, Blend::Min | Blend::Add) && gradient { plane_sample_unit(triangle_planes(tri, brush, vp)(x as f32 + 0.5, y as f32 + 0.5), brush, x, y) } else { None };
        let c = cov.unit();
        let out = match (blend, unit) {
            // MIN lerps the unquantised source toward white and rounds once (p224c: a 0.648-opacity brush's fringe):
            // source·opacity·coverage + (1 − alpha·opacity·coverage), clamped, ×255.
            (Blend::Min, Some((p, o))) => {
                let p = p.map(|v| v * o);
                let keep = 1.0 - p[3] * c;
                core::array::from_fn(|k| (((p[k] * c + keep).clamp(0.0, 1.0) * 255.0).round_ties_even() as u8).min(dst[k]))
            }
            (Blend::Min, None) => cov.min_white_f(dst, s),
            // ADD scales the sample by one interpolated opacity·coverage attribute, clamps, ×255, rounds, then adds
            // saturating (JIT of a per-primitive ADD radial ink).
            // ponytail: a·(o·c) with c our walk coverage; native interpolates o·c from the vertices (same inside a row).
            (Blend::Add, Some((p, o))) => add_unit(dst, p, o * c),
            // Other brushes the same on their unquantised source: a solid's colour attribute, a bitmap's texel/255 (JIT).
            (Blend::Add, None) => core::array::from_fn(|k| (dst[k] as f32 + (s[k] * c).round_ties_even().clamp(0.0, 255.0)).min(255.0) as u8),
            (Blend::Copy, _) => cov.lerp(dst, s.map(|v| v.round_ties_even().clamp(0.0, 255.0) as u8)),
            _ => blend_src(dst, s, c, blend),
        };
        buf.data[i..i + 4].copy_from_slice(&out);
    });
}

/// `FillMesh` with a gradient/bitmap/image brush: rasterize one aliased triangle (device-space
/// `pos`) through WARP's D3D triangle rule — the *same* 1/256-snap + top-left membership as
/// [`crate::warp::fill_triangle`], so the solid and brush mesh fills place edges identically — and
/// composite the per-pixel brush source at full coverage. The composite is byte-identical to
/// [`fill_geometry`]'s `cov8 == 255` branch (one round-ties-even of the premultiplied source, then
/// WARP's integer source-over / `blend_src`), just with triangle coverage instead of scan coverage.
pub fn fill_triangle_brush(buf: &mut PixelBuffer, pos: [[f32; 2]; 3], brush: &Brush, blend: Blend) {
    let (w, h) = (buf.desc.width as f32, buf.desc.height as f32);
    // A gradient under ADD below full opacity: the shader's normalized sample × opacity, rounded once (as `walk_fill`
    // at full coverage). At opacity 1 the kernel has no attribute to scale by and stores the integer ramp lerp.
    if blend == Blend::Add && matches!(brush, Brush::Linear { opacity, .. } | Brush::Radial { opacity, .. } if *opacity < 1.0) {
        let planes = triangle_planes(pos, brush, [0.0, 0.0, w, h]);
        let (p, area) = crate::warp::snap_tri(pos);
        crate::warp::cover_tri(p, area, w as i64, h as i64, |x, y| {
            if let Some((xn, o)) = plane_sample_unit(planes(x as f32 + 0.5, y as f32 + 0.5), brush, x, y) {
                let i = y * buf.stride + x * 4;
                let dst: [u8; 4] = buf.data[i..i + 4].try_into().unwrap();
                buf.data[i..i + 4].copy_from_slice(&add_unit(dst, xn, o));
            }
        });
        return;
    }
    fill_triangle_sampled(buf, pos, triangle_brush_sampler(pos, brush, false, [0.0, 0.0, w, h]), blend, fused_kernel(brush));
}

/// ADD of a gradient's normalized sample `xn` scaled by the shader's opacity·coverage attribute `a`: clamped, ×255,
/// rounded once, added saturating (JIT of the ADD ink kernels).
fn add_unit(dst: [u8; 4], xn: [f32; 4], a: f32) -> [u8; 4] {
    core::array::from_fn(|k| (dst[k] as f32 + ((xn[k] * a).clamp(0.0, 1.0) * 255.0).round_ties_even()).min(255.0) as u8)
}

/// One FillMesh triangle piece as WARP sets it up: each vertex `[x, y, _, a0, a1]` carries the brush values
/// ([`brush_vertex_attrs`]) the clipper interpolated onto it, and the planes are fitted from them with
/// `order`'s first vertex as the reference.
pub fn fill_mesh_piece_brush(buf: &mut PixelBuffer, piece: [[f32; 5]; 3], order: [usize; 3], brush: &Brush, blend: Blend) {
    let o = order.map(|k| piece[k]);
    let (sp, _) = crate::warp::snap_tri(o.map(|v| [v[0], v[1]]));
    let pl = [3, 4].map(|c| crate::warp::plane::setup_triangle_attr(sp.map(|q| q[0]), sp.map(|q| q[1]), o.map(|v| v[c])));
    fill_triangle_sampled(buf, piece.map(|v| [v[0], v[1]]), |x, y| plane_sample(Some(pl), brush, false, x, y), blend, fused_kernel(brush));
}

/// Whether the brush's draw kernel fuses source-over in the high-byte form (bitmap and gradient
/// brushes; see [`fill_geometry`]), rather than adding WARP's separate `unorm_mul` of the backdrop.
fn fused_kernel(brush: &Brush) -> bool {
    matches!(brush, Brush::Bitmap { .. } | Brush::Linear { .. } | Brush::Radial { .. })
}

/// A FillMesh fan triangle whose clipped vertex carries NaN clip attributes: WARP rasterizes it at the
/// `cvtss2si` positions `pos`, but its interpolated texcoord is NaN, whose 24.8 sampler address is
/// `i32::MIN`: texel `−2^23`, which every extend folds to texel 0 of a power-of-two ramp, at weight 0
/// (r14 19786: the whole 943 px fan is the ramp's first texel).
pub fn fill_triangle_brush_nan_texcoord(buf: &mut PixelBuffer, pos: [[f32; 2]; 3], brush: &Brush, blend: Blend) {
    let s0 = match brush {
        Brush::Linear { grad, opacity, .. } | Brush::Radial { grad, opacity, .. } => {
            let t0: [f32; 4] = core::array::from_fn(|c| grad.texels[c] as f32);
            Some(scale_jit(t0, opacity.clamp(0.0, 1.0)))
        }
        _ => None,
    };
    fill_triangle_sampled(buf, pos, |x, y| s0.unwrap_or_else(|| brush.source_at(x as f32 + 0.5, y as f32 + 0.5, true)), blend, fused_kernel(brush));
}

fn fill_triangle_sampled(buf: &mut PixelBuffer, pos: [[f32; 2]; 3], sample: impl Fn(usize, usize) -> [f32; 4], blend: Blend, fused: bool) {
    let (w, h) = (buf.desc.width as i64, buf.desc.height as i64);
    let (p, area) = crate::warp::snap_tri(pos);
    crate::warp::cover_tri(p, area, w, h, |x, y| {
        blend_sampled_px(buf, x, y, sample(x, y), blend, fused);
    });
}

/// Composite one sampled premultiplied brush texel (0..255) over pixel `(x, y)` as a brush-mesh
/// triangle fill does.
fn blend_sampled_px(buf: &mut PixelBuffer, x: usize, y: usize, sf: [f32; 4], blend: Blend, fused: bool) {
    let i = y * buf.stride + x * 4;
    let out: [u8; 4] = if blend == Blend::SourceOver {
        let sc: [i32; 4] = core::array::from_fn(|k| sf[k].round_ties_even().clamp(0.0, 255.0) as i32);
        let ia = 255 - sc[3];
        // r23 28723: a gradient mesh fill one LSB high on 10 px with the separate `unorm_mul`.
        if fused {
            core::array::from_fn(|k| { let p = buf.data[i + k] as i32 * ia; ((sc[k] << 8) + p + (p >> 8) + 128) >> 8 }.clamp(0, 255) as u8)
        } else {
            core::array::from_fn(|k| (sc[k] + crate::bitmap::unorm_mul(buf.data[i + k] as i32, ia)).clamp(0, 255) as u8)
        }
    } else {
        blend_src(buf.data[i..i + 4].try_into().unwrap(), sf, 1.0, blend)
    };
    buf.data[i..i + 4].copy_from_slice(&out);
}

/// One pixel of a FillMesh triangle's inverted quad: the brush at the NaN texcoord the clipped `+inf`
/// vertex carries (a gradient's first texel, a bitmap's corner texel), composited as the fill does.
pub fn fill_px_brush_nan_texcoord(buf: &mut PixelBuffer, x: usize, y: usize, brush: &Brush, blend: Blend) {
    let sf = match brush {
        Brush::Linear { grad, opacity, .. } | Brush::Radial { grad, opacity, .. } => {
            let t0: [f32; 4] = core::array::from_fn(|c| grad.texels[c] as f32);
            scale_jit(t0, opacity.clamp(0.0, 1.0))
        }
        _ => brush.source_at(f32::NAN, f32::NAN, true),
    };
    blend_sampled_px(buf, x, y, sf, blend, fused_kernel(brush));
}

fn clip_quad(q: [f32; 4], limit: [f32; 4]) -> [f32; 4] {
    [q[0].max(limit[0]), q[1].max(limit[1]), q[2].min(limit[2]), q[3].min(limit[3])]
}

/// An intermediate's quad, per triangle and per tap its u and v planes, and the brush inverse ([`intermediate_planes`]).
type TextQuad = ([f32; 4], [[[[f32; 3]; 2]; 4]; 2], [[f32; 3]; 3]);

/// A CUBIC or MULTI_SAMPLE_LINEAR bitmap brush reaches text through its intermediate, as a HIGH_QUALITY_CUBIC one
/// does: d2d1 renders the brush over the draw's quad (the glyph texture's bounds cut to the target) into its atlas at
/// the origin, and the text reads the stored bytes. The vertex shader forms each texcoord at each quad corner (MSL:
/// each of its four rotated-grid taps, the corner plus the tap offset) through the brush's inverse in adjugate form
/// over its determinant, then `u = tex − (corner − X)·step` with `tex = corner·step`, `step = 1/size`; the rasterizer
/// interpolates them over the quad's two triangles (VM DDI of r25 30242 and 30374: the shader pairs, the quads and the
/// intermediates read back). CUBIC's pixel shader takes its kernel's phase from the interpolated device position
/// mapped through that same inverse, the texcoord only placing its taps. Returns the quad, per triangle and per tap
/// the u and v planes in the atlas frame (CUBIC: the texcoord, then the device position), and the inverse.
fn intermediate_planes(brush: &Brush, q: [f32; 4]) -> Option<TextQuad> {
    let Brush::Bitmap { bitmap, transform: m, interpolation, .. } = brush else { return None };
    if !(q[2] > q[0] && q[3] > q[1]) || !q.iter().all(|v| v.is_finite()) {
        return None;
    }
    let (a, b, c, d, tx, ty) = (m[0], m[1], m[2], m[3], m[4], m[5]);
    let adj = [[d, -c, c * ty - d * tx], [-b, a, b * tx - a * ty], [0.0f32, 0.0, a * d - b * c]];
    let step = [1.0 / bitmap.width as f32, 1.0 / bitmap.height as f32];
    let taps: [(f32, f32); 4] = if *interpolation == 3 { [(0.375, -0.125), (-0.125, -0.375), (-0.375, 0.125), (0.125, 0.375)] } else { [(0.0, 0.0); 4] };
    let corner = |p: [f32; 2]| -> [[f32; 2]; 4] {
        let tex = [p[0] * step[0], p[1] * step[1]];
        let mut v = taps.map(|(ox, oy)| {
            let (px, py) = (p[0] + ox, p[1] + oy);
            let [x, y] = device_to_texel(&adj, px, py);
            [-(-x + p[0]) * step[0] + tex[0], -(-y + p[1]) * step[1] + tex[1]]
        });
        // The cubic kernel takes one texcoord, and its pixel shader re-maps the interpolated device position for the
        // kernel's phase, so the second slot carries the corner.
        if *interpolation == 2 {
            v[1] = p;
        }
        v
    };
    let [l, t, r, bt] = q;
    // The quad's two triangles in the shader's vertex order (first vertex the plane's reference).
    let tris = [[[r, t], [l, t], [r, bt]], [[l, bt], [l, t], [r, bt]]];
    let planes = tris.map(|tri| {
        let vals = tri.map(corner);
        let (sx, sy) = (tri.map(|p| ((p[0] - l) * 256.0) as i64), tri.map(|p| ((p[1] - t) * 256.0) as i64));
        core::array::from_fn(|k| core::array::from_fn(|ch| crate::warp::plane::setup_triangle_attr(sx, sy, vals.map(|v| v[k][ch]))))
    });
    Some((q, planes, adj))
}

/// A device position through the brush's inverse in adjugate form (rows: the x, y and w terms), over its w.
fn device_to_texel(adj: &[[f32; 3]; 3], px: f32, py: f32) -> [f32; 2] {
    let [x, y, w] = adj.map(|r| (r[0] * px + py * r[1]) + r[2]);
    [x / w, y / w]
}

/// The intermediate's byte at device pixel centre `(x, y)` ([`intermediate_planes`]), stored ×255 to nearest
/// (premultiplied BGRA, 0..255): MSL sums its four taps' bilinear samples and scales by 0.25; CUBIC takes its kernel
/// at the one texcoord, phased by the device position's texel.
#[allow(clippy::too_many_arguments)]
fn intermediate_texel(bitmap: &crate::bitmap::Bitmap, extend: [Extend; 2], interpolation: u32, q: &[f32; 4], planes: &[[[[f32; 3]; 2]; 4]; 2], adj: &[[f32; 3]; 3], x: f32, y: f32) -> [f32; 4] {
    let (lx, ly) = (x - q[0], y - q[1]);
    // Above the diagonal from the top-left to the bottom-right corner is the first triangle.
    let tri = if (q[3] - q[1]) * lx >= (q[2] - q[0]) * ly { 0 } else { 1 };
    let tex = Texture { data: &bitmap.premul_bgra, width: bitmap.width, height: bitmap.height };
    let (bw, bh) = (bitmap.width as f32, bitmap.height as f32);
    if interpolation == 2 {
        let [u, p] = [0, 1].map(|k| planes[tri][k].map(|p| lx * p[0] + (ly * p[1] + p[2])));
        let t = device_to_texel(adj, p[0], p[1]);
        return tex.sample_hqcubic_at(extend, u, t).map(|v| (v.clamp(0.0, 1.0) * 255.0).round_ties_even());
    }
    let pow2 = [bitmap.width.is_power_of_two(), bitmap.width.is_power_of_two() && bitmap.height.is_power_of_two()];
    let mut sum = [0.0f32; 4];
    for k in 0..4 {
        let [pu, pv] = planes[tri][k].map(|p| lx * p[0] + (ly * p[1] + p[2]));
        let (qx, qy) = (bitmap_brush_q8(pu, bw, extend[0], pow2[0]), bitmap_brush_q8(pv, bh, extend[1], pow2[1]));
        for (ch, s) in sum.iter_mut().enumerate() {
            *s += crate::sampling::norm_v128(tex.bilinear_v128(extend, ch, qx, qy));
        }
    }
    sum.map(|v| ((v * 0.25).clamp(0.0, 1.0) * 255.0).round_ties_even())
}

/// Premultiplied BGRA (0..255) of a geometry bitmap brush from a per-triangle NORMALIZED texcoord `u`
/// (one piece's plane already evaluated at the pixel centre). The extend-fold address matches
/// `source_at`'s geometry arms byte-for-byte: NEAREST = `floor(fold(u)·n)` fetch (eb4177a); LINEAR = the
/// `1 if jit` arm's Q8 `bilinear_v128` (WRAP folds the frac before ×n/×256n, CLAMP/MIRROR likewise).
fn bitmap_geom_fold_sample(bitmap: &crate::bitmap::Bitmap, extend: [Extend; 2], interp: u32, opacity: f32, u: [f32; 2], np: [f32; 6]) -> [f32; 4] {
    let tex = Texture { data: &bitmap.premul_bgra, width: bitmap.width, height: bitmap.height };
    let (bw, bh) = (bitmap.width as f32, bitmap.height as f32);
    let pow2 = [bitmap.width.is_power_of_two(), bitmap.width.is_power_of_two() && bitmap.height.is_power_of_two()];
    let op = opacity.clamp(0.0, 1.0);
    if interp == 3 {
        // MULTI_SAMPLE_LINEAR: four bilinear taps at the sub-pixel offsets, stepped along the brush's exact normalized
        // gradients `np` from the triangle's base texcoord, averaged (as `source_at`'s plain-bitmap MSL arm).
        let mut acc = [0.0f32; 4];
        for (ox, oy) in [(-0.125, -0.375), (0.375, -0.125), (-0.375, 0.125), (0.125, 0.375)] {
            let (pu, pv) = ((u[0] + ox * np[0]) + oy * np[2], (u[1] + ox * np[1]) + oy * np[3]);
            let (qx, qy) = (bitmap_brush_q8(pu, bw, extend[0], pow2[0]), bitmap_brush_q8(pv, bh, extend[1], pow2[1]));
            for (ch, a) in acc.iter_mut().enumerate() { *a += crate::sampling::norm_v128(tex.bilinear_v128(extend, ch, qx, qy)) * 0.25; }
        }
        return acc.map(|v| (v.clamp(0.0, 1.0) * op) * 255.0);
    }
    if interp == 2 {
        // The cubic kernel addresses its windows from the normalized plane and takes the phase as `u·n`.
        let s = tex.sample_hqcubic_at(extend, u, [u[0] * bw, u[1] * bh]);
        return s.map(|v| (v * op).clamp(0.0, 1.0) * 255.0);
    }
    if interp == 0 {
        let idx = |ax: usize, n: u32| -> i32 {
            let (t, e) = (u[ax], extend[ax]);
            let nf = n as f32;
            let raw = if pow2[ax] {
                let t = if matches!(e, Extend::Wrap | Extend::Mirror) { t } else { t.clamp(-10.0, 10.0) };
                (t * nf).floor() as i32
            } else {
                (match e {
                    Extend::Wrap => (t - t.floor()) * nf,
                    Extend::Mirror => { let m = t * 0.5; (m - m.floor()) * (2.0 * nf) }
                    _ => t.clamp(-10.0, 10.0) * nf,
                }).floor() as i32
            };
            fold(e, raw, n as i32)
        };
        let (ix, iy) = (idx(0, bitmap.width), idx(1, bitmap.height));
        // A transparent border reads 0 past the texture rather than folding back onto it.
        let past = |ax: usize, n: u32| extend[ax] == Extend::Transparent && {
            let t = u[ax].clamp(-10.0, 10.0) * n as f32;
            !(0.0..n as f32).contains(&t.floor())
        };
        if past(0, bitmap.width) || past(1, bitmap.height) {
            return [0.0; 4];
        }
        scale(tex.sample(Filter::Nearest, [Extend::Clamp; 2], ix as f32 + 0.5, iy as f32 + 0.5).map(|v| v * 255.0), op)
    } else {
        // Off the power-of-two path the geometry and mesh kernels zero a NaN address lane after the half texel.
        let q = |t: f32, n: f32, e: Extend, p2: bool| if p2 { bitmap_brush_q8(t, n, e, true) } else { crate::sampling::q8_address_npow2_geom(t, n, e) };
        let (qx, qy) = (q(u[0], bw, extend[0], pow2[0]), q(u[1], bh, extend[1], pow2[1]));
        // At full opacity the kernel stores the value·128 bilinear truncated, as `source_at`'s LINEAR arm does (VM probes
        // of COPY walk fills and FillMesh with a full-opacity LINEAR bitmap brush: rounding put most pixels a step high).
        if op == 1.0 {
            return core::array::from_fn(|ch| (tex.bilinear_v128(extend, ch, qx, qy) >> 7) as f32);
        }
        core::array::from_fn(|ch| (crate::sampling::norm_v128(tex.bilinear_v128(extend, ch, qx, qy)) * op).clamp(0.0, 1.0) * 255.0)
    }
}

/// The brush values FillMesh's vertex shader outputs at the device vertices `pos`: a linear gradient's
/// normalized `u`, a radial gradient's normalized `pc`, a bitmap brush's normalized texcoord (`None`: a
/// brush sampled globally).
pub(crate) fn brush_vertex_attrs(pos: [[f32; 2]; 3], brush: &Brush) -> Option<[[f32; 2]; 3]> {
    match brush {
        Brush::Linear { grad, xform, .. } => {
            let inv_w = 1.0 / grad.width as f32; // W is a power of two ⇒ exact (Direct2D keeps `1/W`)
            let (cx, cy, cc) = (xform[0] * inv_w, xform[1] * inv_w, xform[2] * inv_w);
            Some(pos.map(|p| [(p[0] * cx + p[1] * cy) + cc, 0.0]))
        }
        Brush::Radial { to_unit_raw: m, .. } => Some(pos.map(|p| [m[0] * p[0] + (m[2] * p[1] + m[4]), m[1] * p[0] + (m[3] * p[1] + m[5])])),
        // NEAREST/LINEAR bitmap brush on geometry: WARP fits the NORMALIZED texcoord plane PER rasterized
        // triangle (SetupTriangleAttr keyed on the provoking vertex pos[2]), NOT a global unit-triangle
        // plane. At a huge-magnitude texcoord (fuzz r17 22971: 1e-6 brush → ~1e8) the fit's
        // `c0 = a0 − dadx·refx − dady·refy` loses ULPs differently for the actual provoking vertex than
        // for the unit triangle, flipping the sampled texel. Per-vertex = the RAW device→normalized map
        // (`invert(transform·size)`); the per-triangle/clip fit recovers the plane. Guarded to a
        // true bitmap brush (NOT a realized image-brush pow2 atlas, whose texture_size ≠ the bitmap).
        // A cropped image brush sampled from its border cells: the walk's vertex shader forms each vertex's atlas texcoord
        // from the atlas row, `dp2` then `add` (VM DDI of r22 27703's COPY arc fill: 324 walk vertices, the row in the VS
        // constants, the CUBIC pixel shader phasing by texcoord times the atlas size). A whole-pixel rect keeps the quad its
        // cubic pass draws ([`Brush::keep_cell_quad`]).
        Brush::Bitmap { vs_row: Some(m), cell: Some(_), cell_quad: None, interpolation: 1 | 2, .. } => {
            Some(pos.map(|p| [(p[0] * m[0] + p[1] * m[2]) + m[4], (p[0] * m[1] + p[1] * m[3]) + m[5]]))
        }
        Brush::Bitmap { transform, bitmap, texture_size, interpolation, vs_row, cell: None, .. }
            if *texture_size == [bitmap.width, bitmap.height] && *interpolation <= 3 => {
            let (bw, bh) = (bitmap.width as f32, bitmap.height as f32);
            let m = vs_row.unwrap_or_else(|| invert(&[transform[0] * bw, transform[1] * bw, transform[2] * bh, transform[3] * bh, transform[4], transform[5]]));
            // The vertex shader forms each texcoord as a two-term dot product plus the translation (DDI of a COPY walk
            // fill's bitmap-brush VS: `dp2` then `add`).
            Some(pos.map(|p| [(p[0] * m[0] + p[1] * m[2]) + m[4], (p[0] * m[1] + p[1] * m[3]) + m[5]]))
        }
        _ => None,
    }
}

/// Whether the device polygon `pts` is an axis-aligned rect on whole pixels.
pub(crate) fn whole_px_rect(pts: &[[f32; 2]]) -> bool {
    let distinct = |k: usize| { let mut v: Vec<f32> = pts.iter().map(|p| p[k]).collect(); v.sort_by(f32::total_cmp); v.dedup(); v.len() };
    !pts.is_empty() && pts.iter().all(|p| p[0].fract() == 0.0 && p[1].fract() == 0.0) && distinct(0) == 2 && distinct(1) == 2
}

/// A CUBIC fill's atlas texcoord and tap phase at device `(x, y)` over border cells at `cell`. d2d1 draws the fill's
/// quad with its 9-tap cubic shader: the vertex shader forms each vertex's texcoord through the adjugate of the tile's
/// matrix `m` (cb0 = adj(m), cb0[2].z = det), the rasterizer interpolates it over the quad, and the pixel shader phases
/// the taps by the pixel centre through that adjugate (VM DDI of rotated CUBIC crop fills: cb0 bit for bit; geo/c703d
/// 9703203/04/06, 5-13 px each through the atlas row's fitted plane, exact). Without a quad: the atlas row's plane.
fn cell_cubic_uv(row: &[f32; 6], cell: [i32; 2], size: u32, quad: Option<&([f32; 6], [f32; 4])>, x: f32, y: f32) -> ([f32; 2], [f32; 2]) {
    match quad {
        Some((m, q)) => {
            let det = m[0] * m[3] - m[1] * m[2];
            let phase = [((m[3] * x - m[2] * y) + (m[2] * m[5] - m[3] * m[4])) / det, ((-m[1] * x + m[0] * y) + (m[1] * m[4] - m[0] * m[5])) / det];
            let planes = bitmap_quad_at(m, [size; 2], [-cell[0], -cell[1]], *q, x, y, [0.0; 2]);
            (planes.map(|a| a[0] * x + (a[1] * y + a[2])), phase)
        }
        None => {
            let r = crate::warp::plane::warp_fit_affine(row);
            let u = [r[0] * x + (r[2] * y + r[4]), r[1] * x + (r[3] * y + r[5])];
            (u, u.map(|v| v * size as f32))
        }
    }
}

/// The Q8 address, on the brush bitmap's own texels, of a brush sampled from its border cells at atlas texel `cell`:
/// the atlas row fitted over the unit triangle, moved by `off` pixels along the row's own gradients (a
/// MULTI_SAMPLE_LINEAR tap), then the power-of-two atlas's floor address (as [`bitmap_brush_q8`]).
fn atlas_cell_q8(row: &[f32; 6], cell: [i32; 2], n: f32, x: f32, y: f32, off: [f32; 2]) -> [(i32, i32); 2] {
    let r = crate::warp::plane::warp_fit_affine(row);
    core::array::from_fn(|k| {
        let u = (r[k] * x + (r[2 + k] * y + r[4 + k]) + off[0] * row[k]) + off[1] * row[2 + k];
        let q = ((u * n - 0.5) * 256.0).floor() as i32;
        ((q >> 8) - cell[k], q & 255)
    })
}

/// WARP's LINEAR bitmap-brush address of one axis: the Q8 texel and weight of the normalized texcoord `t` on an
/// `n`-texel axis under extend `e`. `pow2` is x of a power-of-two-wide texture or y of one that is a power of two on
/// both axes, which floors `(t·n − ½)·256`; any other axis rounds `t·256n` to nearest-even, less 128.
pub(crate) fn bitmap_brush_q8(t: f32, n: f32, e: Extend, pow2: bool) -> (i32, i32) {
    let q = if pow2 {
        let t = if matches!(e, Extend::Wrap | Extend::Mirror) { t } else { t.clamp(-10.0, 10.0) };
        crate::sampling::q8_floor((t * n - 0.5) * 256.0)
    } else {
        (match e {
            Extend::Wrap => (t - t.floor()) * (256.0 * n),
            Extend::Mirror => { let m = t * 0.5; (m - m.floor()) * (512.0 * n) }
            _ => t.clamp(-10.0, 10.0) * (256.0 * n),
        }).round_ties_even() as i32 - 128
    };
    (q >> 8, q & 255)
}

/// The premultiplied BGRA (0..255) brush source of a WARP triangle fill at pixel `(x, y)`, for the device
/// triangle `pos` drawn into `viewport`: gradients sample their texcoord plane fitted per rasterized
/// triangle, any other brush its global source. `frac` keeps a gradient's lerp fractional, as for an opacity
/// brush, for a kernel that computes on the sample instead of storing it.
pub(crate) fn triangle_brush_sampler(pos: [[f32; 2]; 3], brush: &Brush, frac: bool, viewport: [f32; 4]) -> impl Fn(usize, usize) -> [f32; 4] + '_ {
    let plane_at = triangle_planes(pos, brush, viewport);
    move |x, y| plane_sample(plane_at(x as f32 + 0.5, y as f32 + 0.5), brush, frac, x, y)
}

/// The attribute planes WARP sets up for the triangle `pos` drawn with `brush` (per clipped piece), looked up at a
/// pixel centre: see [`triangle_brush_sampler`].
fn triangle_planes(pos: [[f32; 2]; 3], brush: &Brush, viewport: [f32; 4]) -> impl Fn(f32, f32) -> Option<[[f32; 3]; 2]> {
    // Linear gradient: native fills each triangle with WARP's NORMALIZED texcoord plane, fitted per
    // rasterized triangle by WARP's attribute setup keyed on the **provoking (last) vertex**
    // (`pos[2]`), then the pixel shader scales by the gradient width `W` for the sampler. d2d1 hands
    // WARP the normalized transform (`xform/W`, [`linear_transform`] × 1/W, exact for pow2 W): e.g.
    // meshbrush-01-linear `c = 0xbe1ffffe`. WARP's per-triangle `c0 = a0 − dadx·refx − dady·refy`
    // (ref = `pos[2]`, snapped 1/256) re-rounds it to native's `0xbe1ffffc`, which flips a near-texel-
    // boundary pixel ((15,8): u·W 11.4999→11.5000, texel 10→11). Reproduced byte-exact from the VM JIT
    // dump of the gradient PS + the d2d1 realize chain (GetRealizationInfoAndTextureMapping →
    // CalculateTextureMapping; a4=identity). Our old texel-domain direct eval and a vertex-0 fit both
    // recovered the input `c` exactly (no-op) — the key is the **normalized domain + provoking vertex**.
    // Radial: the normalized device→unit texcoord `pc` is a TWO-component varying, `o1 = M·v` per vertex
    // (M = the RAW device→unit map, already ÷W), each component fitted the same way; the focal is the
    // unfitted constant `M·focal`.
    triangle_attr_planes(pos, |p| brush_vertex_attrs(p, brush), viewport)
}

/// [`triangle_planes`] for any per-vertex two-component attribute: `attrs` maps the triangle's vertices, provoking
/// vertex first, to their attribute values (`None`: no plane).
fn triangle_attr_planes(pos: [[f32; 2]; 3], attrs: impl Fn([[f32; 2]; 3]) -> Option<[[f32; 2]; 3]>, viewport: [f32; 4]) -> impl Fn(f32, f32) -> Option<[[f32; 3]; 2]> {
    let ord = [2usize, 0, 1]; // provoking (last) vertex first = WARP's SetupTriangleAttr reference
    let attrs = attrs(ord.map(|k| pos[k]));
    // (piece corners, snapped, the two attribute planes).
    type Piece = ([[f32; 2]; 3], [[f32; 3]; 2]);
    let fit = |sx: [i64; 3], sy: [i64; 3], a: [[f32; 2]; 3]| -> [[f32; 3]; 2] {
        [0, 1].map(|c| crate::warp::plane::setup_triangle_attr(sx, sy, a.map(|v| v[c])))
    };
    // WARP frustum-clips a triangle with a vertex outside the viewport before its plane setup: the
    // attributes are interpolated onto the clipped vertices (in setup order, provoking vertex first) and
    // each fan piece gets its own plane (lrrq-b1: a COPY band triangle past the target on three sides).
    let [l, t, r, b] = viewport;
    let outside = pos.iter().any(|v| v[0] < l || v[0] > r || v[1] < t || v[1] > b);
    let pieces: Vec<Piece> = match attrs {
        None => Vec::new(),
        Some(a) if outside => {
            let tri: [[f32; 4]; 3] = core::array::from_fn(|i| [pos[ord[i]][0], pos[ord[i]][1], a[i][0], a[i][1]]);
            // FillMesh's clipper ([`crate::warp::clip_triangle`]): its interpolated plane distances and clamped screen
            // mapping are what native's planes carry (COPY walk triangles across the target's left edge, VM probes).
            let fan = match crate::warp::clip_triangle(tri, viewport) {
                crate::warp::Clipped::Rejected => Vec::new(),
                crate::warp::Clipped::Inside => vec![tri],
                crate::warp::Clipped::Fan(f) => f,
            };
            fan.into_iter().map(|t| {
                let corners = t.map(|v| [v[0], v[1]]);
                let (sp, _) = crate::warp::snap_tri(corners);
                (corners, fit(sp.map(|q| q[0]), sp.map(|q| q[1]), t.map(|v| [v[2], v[3]])))
            }).collect()
        }
        Some(a) => {
            let p = crate::warp::snap_tri(pos).0;
            vec![(pos, fit(ord.map(|k| p[k][0]), ord.map(|k| p[k][1]), a))]
        }
    };
    let plane_at = move |xc: f32, yc: f32| -> Option<[[f32; 3]; 2]> {
        let inside = |c: &[[f32; 2]; 3]| {
            let e = |a: [f32; 2], b: [f32; 2]| (b[0] - a[0]) * (yc - a[1]) - (b[1] - a[1]) * (xc - a[0]);
            let (e0, e1, e2) = (e(c[0], c[1]), e(c[1], c[2]), e(c[2], c[0]));
            (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) || (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0)
        };
        pieces.iter().find(|(c, _)| inside(c)).or(pieces.first()).map(|&(_, pl)| pl)
    };
    plane_at
}

/// A gradient's sample through a walk triangle's plane as the MIN walk's pixel shader forms it: the fractional ramp
/// lerp normalized `v/256 + v/65280` and scaled by the opacity, in 0..1 (JIT of a MIN linear-gradient walk).
/// `None` for other brushes.
fn plane_sample_unit(planes: Option<[[f32; 3]; 2]>, brush: &Brush, x: usize, y: usize) -> Option<([f32; 4], f32)> {
    let (xc, yc) = (x as f32 + 0.5, y as f32 + 0.5);
    let (grad, t, opacity) = match (planes, brush) {
        (Some([pl, _]), Brush::Linear { grad, opacity, .. }) => (grad, xc * pl[0] + (yc * pl[1] + pl[2]), opacity),
        (Some([plx, ply]), Brush::Radial { grad, focal, opacity, centered, .. }) => {
            let pc = [xc * plx[0] + (yc * plx[1] + plx[2]), xc * ply[0] + (yc * ply[1] + ply[2])];
            let w = grad.width as f32;
            (grad, focal_t(pc, *focal, *centered, grad.u1 / w, 0.5 / w), opacity)
        }
        _ => return None,
    };
    Some((grad.sample_jit_f(t * grad.width as f32).map(|v| v * 0.003_906_25 + v * f32::from_bits(0x3780_8081)), opacity.clamp(0.0, 1.0)))
}

/// The brush source at pixel `(x, y)` from a triangle's fitted attribute planes (`None`: a brush sampled
/// globally).
fn plane_sample(planes: Option<[[f32; 3]; 2]>, brush: &Brush, frac: bool, x: usize, y: usize) -> [f32; 4] {
    let (xc, yc) = (x as f32 + 0.5, y as f32 + 0.5);
    match (planes, brush) {
        (Some([pl, _]), Brush::Linear { grad, opacity, .. }) => {
            // WARP PS per-pixel: u_norm = dadx·xc + (dady·yc + c0) at the pixel centre, then ×W.
            let u = (xc * pl[0] + (yc * pl[1] + pl[2])) * grad.width as f32;
            // A brush opacity < 1 keeps the ramp lerp in its Q8 fraction (the PS truncates to a byte
            // only when it stores the sample directly); the single store-round then applies, so the
            // fractional sampler matches native (fuzz fill_mesh seeds, all opacity<1). Same rule as the
            // radial direct path (`source_at`).
            let s = if *opacity != 1.0 || frac { grad.sample_jit_f(u) } else { grad.sample(u, true) };
            scale_jit(s, opacity.clamp(0.0, 1.0))
        }
        (Some([plx, ply]), Brush::Radial { grad, focal, opacity, centered, .. }) => {
            // The per-triangle-fitted normalized `pc`, evaluated with the radial PS's separate
            // multiply-adds, then the scaled focal quadratic and `u = t·W` sampler (as `source_at`).
            let pc = [xc * plx[0] + (yc * plx[1] + plx[2]), xc * ply[0] + (yc * ply[1] + ply[2])];
            let w = grad.width as f32;
            let t = focal_t(pc, *focal, *centered, grad.u1 / w, 0.5 / w);
            let s = if *opacity != 1.0 || frac { grad.sample_jit_f(t * w) } else { grad.sample_jit(t * w) };
            scale_jit(s, opacity.clamp(0.0, 1.0))
        }
        (Some([plx, ply]), Brush::Bitmap { bitmap, extend, interpolation: interp @ (1 | 2), opacity, cell: Some(c), cell_size, .. }) => {
            // Border cells: the per-triangle atlas texcoord, addressed on the atlas less the cell offset (as `atlas_cell_q8`).
            let u = [xc * plx[0] + (yc * plx[1] + plx[2]), xc * ply[0] + (yc * ply[1] + ply[2])];
            let tex = Texture { data: &bitmap.premul_bgra, width: bitmap.width, height: bitmap.height };
            let (n, op) = (*cell_size as f32, opacity.clamp(0.0, 1.0));
            if *interp == 2 {
                tex.sample_hqcubic_unclamped_at(*extend, u, u.map(|v| v * n), [*cell_size; 2], *c).map(|v| (v * op).clamp(0.0, 1.0) * 255.0)
            } else {
                let [qx, qy] = [0, 1].map(|k| { let q = ((u[k] * n - 0.5) * 256.0).floor() as i32; ((q >> 8) - c[k], q & 255) });
                if op == 1.0 { core::array::from_fn(|ch| (tex.bilinear_v128(*extend, ch, qx, qy) >> 7) as f32) }
                else { core::array::from_fn(|ch| (crate::sampling::norm_v128(tex.bilinear_v128(*extend, ch, qx, qy)) * op).clamp(0.0, 1.0) * 255.0) }
            }
        }
        (Some([plx, ply]), Brush::Bitmap { bitmap, extend, interpolation, opacity, transform, vs_row, .. }) => {
            // The per-triangle (/clipped-piece) normalized texcoord, then the same extend-fold address
            // + fetch as `source_at`'s geometry bitmap arms (WRAP frac-fold / CLAMP / MIRROR; N / L).
            let u = [xc * plx[0] + (yc * plx[1] + plx[2]), xc * ply[0] + (yc * ply[1] + ply[2])];
            bitmap_geom_fold_sample(bitmap, *extend, *interpolation, *opacity, u, vs_row.unwrap_or_else(|| Brush::plane_of_b2d(transform, bitmap.width, bitmap.height)))
        }
        _ => brush.source_at(x as f32 + 0.5, y as f32 + 0.5, true),
    }
}

impl DeviceContext {
    /// `FillGeometry`/`FillEllipse` with a gradient or bitmap/image brush. Flattens the geometry the
    /// way the solid `fill_geometry` does (native MIL curve cracker) and composites the brush under
    /// the geometry's coverage ([`fill_geometry`]). Recording and non-SOURCE_OVER primitive blend are
    /// not wired (as `fill_rectangle_brush`): they latch `E_NOTIMPL`.
    pub fn fill_geometry_brush(&mut self, geom: &Geometry, brush: &PixelBrush) {
        let _a = self.api(128 + brush.batch, 0, Some(brush.queued()));
        if !self.ready_to_draw() || !self.check_brush(brush) {
            return;
        }
        if geom.fill_nonfinite_void() {
            return; // native's geometry sink voids a path with ANY non-finite point (fuzz r11 16347)
        }
        // While recording, record the DIP geometry + the brush (with its device-independent definition);
        // replay re-flattens the geometry and re-realizes the brush under the replay transform. [G6]
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            self.record(crate::command::RecCmd::FillGeomBrush { state, dpi, geom: geom.clone(), brush: brush.clone() });
            return;
        }
        // Gradient brushes realize lazily per draw under the current device transform (native runs
        // GetRealizationPlacementData at fill time), so a world transform set after CreateBrush applies
        // (fuzz r5 5011/5862). Non-gradient brushes keep their baked realization.
        let grad_rr = self.realize_brush_for_fill(brush);
        let brush = grad_rr.as_ref().unwrap_or(brush);
        let m = self.device_transform();
        let singular = m[0] * m[3] - m[1] * m[2] == 0.0;
        let dm: crate::geometry::Matrix = [m[0] as f32, m[1] as f32, m[2] as f32, m[3] as f32, m[4] as f32, m[5] as f32];
        // Fills use native's 28.4 edge list (aliased and per-primitive alike); 26.6 is `DrawInk`-only.
        let figures: Vec<Vec<[f32; 2]>> =
            geom.flatten_fill_on(&dm, self.copy_target()).into_iter().filter(|p| p.filled).map(|p| p.pts).collect();
        let winding = geom.fill_mode() == crate::geometry::FillMode::Winding;
        let aa = if self.state.aa == 1 { Aa::Aliased } else { Aa::PerPrimitive };
        let blend = crate::context::blend_of(self.state.blend);
        // A singular (det 0) device transform collapses the fill onto a line. An ALIASED non-rect fill is
        // a WARP geometry draw whose aliased edge walk still lights the pixel centres the collapsed outline
        // passes through — for a solid (`context.rs::fill_geometry`, 26191) or a GRADIENT brush, which
        // realises a degenerate `t = ½` wash there. A bitmap/image brush paints nothing on that line, and
        // every other fill (per-primitive, rect) has zero area (fuzz r12 17584) — all draw nothing. Probes
        // 7800231/232/233.
        let warp = aa == Aa::Aliased && !matches!(geom, Geometry::Rect(_));
        if singular && (!warp || matches!(brush.brush, Brush::Bitmap { .. })) {
            return;
        }
        let whole = matches!(geom, Geometry::Rect(_)) && figures.len() == 1 && whole_px_rect(&figures[0])
            && matches!(brush.brush, Brush::Bitmap { opacity, .. } if opacity >= 1.0);
        let per_draw = self.figures_image_brush(brush, &figures, whole);
        // An image brush whose per-draw realization native cannot place (a degenerate/huge brush
        // transform) paints nothing (VM-probed: a 3.4e38-matrix effect image brush voids this fill and
        // FillEllipse, which routes here). `None` means void for an image brush, not the fallback.
        if brush.image.is_some() && per_draw.is_none() {
            return;
        }
        let brush = per_draw.as_ref().unwrap_or(brush);
        let clip = self.raster_clip();
        if let Some(buf) = &mut self.target {
            // An aliased non-rect geometry fill is a WARP geometry draw (`CreateGeometryGeneratorFromShape`
            // → the WARP geometry adapter), so its coverage is WARP's aliased edge walk (9711).
            fill_geometry_cov(buf, &figures, winding, aa, &brush.brush, blend, warp, true, clip, false);
        }
    }

    /// An image or cropped-bitmap brush realized for one draw over the device bounds of its `figures`, as native
    /// realizes it per draw (`whole` keeps the cell quad). `None` for any other brush; for an image brush, `None`
    /// means the realization failed and the draw paints nothing.
    fn figures_image_brush(&mut self, brush: &PixelBrush, figures: &[Vec<[f32; 2]>], whole: bool) -> Option<PixelBrush> {
        if brush.image.is_none() && brush.crop.is_none() {
            return None;
        }
        let pts = figures.iter().flatten();
        let lo = |k: usize| pts.clone().map(|p| p[k]).fold(f32::INFINITY, f32::min);
        let hi = |k: usize| pts.clone().map(|p| p[k]).fold(f32::NEG_INFINITY, f32::max);
        let t = self.target.as_ref()?;
        let d = [lo(0).floor().max(0.0) as i32, lo(1).floor().max(0.0) as i32, hi(0).ceil().min(t.desc.width as f32) as i32, hi(1).ceil().min(t.desc.height as f32) as i32];
        self.image_brush_for_draw(brush, d, false, true).map(|mut pb| { pb.brush.keep_cell_quad(whole); pb })
    }

    /// `DrawGeometry`/`DrawLine`/`DrawRectangle`/… with a gradient or bitmap/image brush: widen the
    /// stroke to device-space fill polygons (exactly as the solid `draw_geometry`) and composite the
    /// brush under their coverage (nonzero winding, as `CWidener`'s pieces union). Shares the brush
    /// colour-texturing path with `fill_geometry_brush`.
    pub fn draw_geometry_brush(&mut self, geom: &Geometry, brush: &PixelBrush, width: f32, style: &crate::geometry::StrokeStyle) {
        let _a = self.api(144 + brush.batch, 0, Some(brush.queued()));
        // A rectangle geometry strokes as `DrawRectangle` ([`DeviceContext::draw_geometry`]).
        match geom {
            Geometry::Rect(r) if self.recording.is_none() => self.draw_rectangle_brush(*r, brush, width, style),
            _ => self.draw_geometry_brush_impl(geom, brush, width, style, None),
        }
    }

    /// [`Self::draw_geometry_brush`]; `cells` are `DrawRectangle`'s fast-widened border cells, which an aliased
    /// stroke fills as a small region ([`DeviceContext::draw_rectangle`]).
    pub(crate) fn draw_geometry_brush_impl(&mut self, geom: &Geometry, brush: &PixelBrush, width: f32, style: &crate::geometry::StrokeStyle, cells: Option<&[[f32; 4]]>) {
        let warp = cells.is_none();
        if !self.ready_to_draw() || !self.check_brush(brush) {
            return;
        }
        // A non-finite STROKE is NOT blanket-voided like a fill: native widens per figure and only voids
        // the whole DrawGeometry on an edge-list overflow or a pen BAD_NUMBER tangent (see `widen_inner`).
        // Record the DIP geometry + stroke + brush; replay re-widens and re-realizes under the replay
        // transform (a stroke fills with nonzero winding). [G6]
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            self.record(crate::command::RecCmd::StrokeGeomBrush { state, dpi, geom: geom.clone(), brush: brush.clone(), width, style: style.clone() });
            return;
        }
        // Gradient brushes realize lazily per draw under the current device transform (as in
        // `fill_geometry_brush`): a world transform set after CreateBrush applies to the stroke too.
        let grad_rr = self.realize_brush_for_fill(brush);
        let brush = grad_rr.as_ref().unwrap_or(brush);
        let m = self.device_transform();
        let dm: crate::geometry::Matrix = [m[0] as f32, m[1] as f32, m[2] as f32, m[3] as f32, m[4] as f32, m[5] as f32];
        // A brush too large to realize over the stroke's bounds fails the draw, as a fill's does (r11 16676: a
        // zero-width DrawRectangle with a 1e-6 image brush latches INTERMEDIATE_TOO_LARGE).
        if let Some(buf) = &self.target {
            // A NaN bound (a non-finite figure) leaves the check out (r14 19511: a rect to -inf under a 1e-6 transform).
            let b = geom.widened_bounds(width, style, &dm, crate::geometry::DEFAULT_TOLERANCE);
            let visible = [b[0].max(0.0), b[1].max(0.0), b[2].min(buf.desc.width as f32), b[3].min(buf.desc.height as f32)];
            if !b.iter().any(|v| v.is_nan()) && visible[0] < visible[2] && visible[1] < visible[3] && self.brush_need_too_large(brush, visible) {
                self.process_new_error(crate::state::INTERMEDIATE_TOO_LARGE);
                return;
            }
        }
        let dpi = self.get_dpi();
        let dpi_scale = [dpi.0 / 96.0, dpi.1 / 96.0];
        let figures = geom.widen_scaled(width, style, &dm, dpi_scale, crate::geometry::DEFAULT_TOLERANCE, 16.0, self.copy_target(), self.target_size());
        // An image brush realizes over the stroke's widened bounds, as a fill's over its outline (r27 32161: an effect
        // image brush's creation-time realization, made at the creation DPI and tile, sampled a stroke at 120 dpi).
        let per_draw = self.figures_image_brush(brush, &figures, false);
        if brush.image.is_some() && per_draw.is_none() {
            return;
        }
        let brush = per_draw.as_ref().unwrap_or(brush);
        let aa = if self.state.aa == 1 { Aa::Aliased } else { Aa::PerPrimitive };
        let blend = crate::context::blend_of(self.state.blend);
        // An aliased bitmap/image-brush stroke takes the same WARP geometry rasterizer as the solid one
        // (`DeviceContext::draw_geometry`, aastroke-02). Gradient strokes keep the per-triangle path
        // `fill_geometry` fits for them.
        let warp28 = aa == Aa::Aliased && warp;
        let clip = self.raster_clip();
        // ponytail: gradients only; a bitmap brush over the cells keeps the union's scan until a probe shows its fit.
        let region = cells.filter(|_| aa == Aa::Aliased && matches!(brush.brush, Brush::Linear { .. } | Brush::Radial { .. }))
            .and_then(|c| small_region_triangles(c, &dm));
        if let Some(buf) = &mut self.target {
            match region {
                Some(tris) => tris.into_iter().for_each(|tri| fill_triangle_brush(buf, tri, &brush.brush, blend)),
                // Cells past the fast path's range fill as rectangle shapes through the general rasterizer.
                None => fill_geometry_cov(buf, &figures, true, aa, &brush.brush, blend, warp28, !(cells.is_some() && aa == Aa::Aliased), clip, false),
            }
        }
    }
}

/// The triangles an aliased fill of non-overlapping rectangles (DIPs, under the device transform `m`) draws: each
/// rect's corners go to device space clockwise from the top-left (counter-clockwise under a mirroring `m`) and are
/// sent as one triangle strip `q0 q0 q3 q1 q2 q2`, so its two real triangles are an odd one on `q0 q3 q1` and an
/// even one on `q3 q1 q2`, in the order the rasterizer sets them up. `None` where a device coordinate passes
/// ±524287.97, which Direct2D draws another way.
fn small_region_triangles(rects: &[[f32; 4]], m: &[f32; 6]) -> Option<Vec<[[f32; 2]; 3]>> {
    let flip = m[0] * m[3] - m[2] * m[1] <= 0.0;
    let dev = |x: f32, y: f32| [(x * m[0] + y * m[2]) + m[4], (x * m[1] + y * m[3]) + m[5]];
    let mut tris = Vec::with_capacity(rects.len() * 2);
    for &[l, t, r, b] in rects {
        let q = if flip { [dev(l, t), dev(l, b), dev(r, b), dev(r, t)] } else { [dev(l, t), dev(r, t), dev(r, b), dev(l, b)] };
        if q.iter().flatten().any(|v| !(v.abs() <= f32::from_bits(0x48ff_ffe0))) {
            return None;
        }
        tris.push([q[3], q[0], q[1]]);
        tris.push([q[3], q[1], q[2]]);
    }
    Some(tris)
}

/// Radial gradient parameter `t` for a point `pc` in the gradient's scaled space with focal `fc` and
/// the `span_right/W` scale WARP keeps in the colour-source constants: cast the ray
/// `fc + k·(pc−fc)` to the (scaled) circle and return the positive root of the focal quadratic, fed
/// straight to the sampler as `u = t·W`.
///
/// Op-for-op against the native radial pixel shader (DXBC `ref/windows/shaders/0x49a130.asm`, and the
/// JIT'd kernel emulated against `ref/radial5-dump`'s 128×96 readback). WARP is a no-FMA reference
/// rasterizer with exact `sqrtps`/`divps` (no `rcp`/`rsqrt`), so the `dp2` dot products are
/// **non-fused** two-`mulss`+`addss`: `cross = (dvx·fy) + (dvy·−fx)` (the fused form landed `u` 1 ULP
/// high — `canvas-radial-050` (23,29) exactly on texel-weight boundary 2646.0, weight 86 vs native
/// 85, confirmed against WARP.s plane struct which this crate's realize reproduces
/// bit-exact). The DXBC `mad`s *are* fused: `D2 = fma(dvx,dvx, dvy²)`, `disc = fma(span², D2, −cross²)`;
/// `num = 2·span·D2`, `fdot` the plain `dp4` reduction. the span-scale term is `1` only for the unbordered
/// (wrap/mirror) span; radial clamp carries `(W−1)/W`. `brushes.md` §2.5.
///
/// A centered gradient (no origin offset) takes the kernel without the focal: `t = |pc|`, raised to the first texel
/// centre `0.5/W` (a NaN `t` reads it too). VM JIT dump of the centered radial fill kernel: `sqrt(pc.x² + pc.y²)`
/// then a max with that constant, and no overflow clamp before the ramp address. The focal quadratic's
/// `2·d²/√(4·d²)` rounds a ULP away from it (VM probes: 3 ±1 pixels in 46 random centered fills, now exact).
fn focal_t(pc: [f32; 2], fc: [f32; 2], centered: bool, vz: f32, v2x: f32) -> f32 {
    if centered {
        return ((pc[0] * pc[0]) + (pc[1] * pc[1])).sqrt().max(v2x);
    }
    let dvx = pc[0] - fc[0];
    let dvy = pc[1] - fc[1];
    let cross = (dvx * fc[1]) + (dvy * -fc[0]);
    let cross2 = cross * cross;
    let d2 = (dvx * dvx) + (dvy * dvy);
    let disc = ((vz * vz) * d2) - cross2;
    let num = (vz * d2) + (vz * d2);
    let root = (disc * 4.0).sqrt();
    let fdot = ((fc[0] * dvx + fc[0] * dvx) + fc[1] * dvy) + fc[1] * dvy;
    let t = num / (root - fdot);
    // Clamp/extend fold of the texcoord, matching the DXBC tail (the shader tail) with `0.5/W` the
    // half-texel and Direct2D.s radial-gradient colour-source constants: the w term = −1
    // when the focal is on or outside the circle (`span² − 1e-6 ≤ |fc|²`), the focal-shift term `fc·(1 − 0.5/⌈u1⌉)`.
    // A missed ray (`disc<0`) → the last texel centre `1−0.5/W`; an overflow (`t>32768`) → `1`; below the
    // first texel centre → `0.5/W`, or `1−0.5/W` for a negative `t` of an outside focal; within `0.5/W` of
    // the focal-shift term (the focal singularity) → `0.5/W` (r10 15098's −inf focal, rad098-* probes).
    let u1 = vz * (0.5 / v2x);
    let pull = 1.0 - 0.5 / u1.ceil();
    let (sx, sy) = (pc[0] - fc[0] * pull, pc[1] - fc[1] * pull);
    let singular = (sx * sx + sy * sy) < v2x * v2x;
    let outside = (vz * vz) - 0.000_001 <= (fc[1] * fc[1]) + (fc[0] * fc[0]);
    let low = if outside && t < 0.0 { 1.0 - v2x } else { v2x };
    let folded = if t < v2x { low } else if t > 32768.0 { 1.0 } else { t };
    if singular {
        v2x
    } else if disc < 0.0 {
        1.0 - v2x
    } else {
        folded
    }
}

#[cfg(test)]
mod tests {
    // The software RCPPS seed (+ one Newton–Raphson) must be bit-for-bit the reference CPU's RCPPS on every
    // target, or the browser build fingerprints against the native builds. These expected bits were dumped
    // from the reference machine's hardware RCPPS; the extreme-MSL bitmap brush's gradient depends on them.
    #[test]
    fn warp_rcp_matches_reference_hardware() {
        let cases: [(f32, u32); 9] = [
            (1e-6, 0x4974_23ff), (1.0, 0x3f7f_ffff), (2.0, 0x3eff_ffff), (3.0, 0x3eaa_aaab),
            (48.0, 0x3caa_aaab), (0.5, 0x3fff_ffff), (7.0, 0x3e12_4924), (1e-5, 0x47c3_5000),
            (3.0517578e-5, 0x46ff_ffff),
        ];
        for (a, want) in cases {
            assert_eq!(super::warp_rcp(a).to_bits(), want, "warp_rcp({a:e})");
        }
    }

    // The Scale effect's per-pass factors native's DDI shows for HQ DrawBitmap minifications (lyr probes 6 and
    // 10): the translation and the logf/powf round trip move 0.4 by a few ulps.
    #[test]
    fn hq_scales_match_native_prescale() {
        let f = f32::from_bits;
        let s = |m: [f32; 6]| super::hq_scales(&m).map(f32::to_bits);
        assert_eq!(s([f(0x3ecccccd), 0.0, 0.0, f(0x3ecccccc), f(0x40266666), f(0x404ccccd)]), [0x3eccccd0, 0x3eccccc8]);
        assert_eq!(s([f(0x3ecccccd), 0.0, 0.0, f(0x3ecccccd), f(0x400f4a85), f(0x3fc0af1e)]), [0x3eccccd0, 0x3ecccccb]);
        // An exponent within 1e-5 of an integer snaps to a power of two; a pure scale passes through.
        assert_eq!(s([0.5, 0.0, 0.0, 0.25, 2.3, 1.7]), [0.5f32.to_bits(), 0.25f32.to_bits()]);
        assert_eq!(s([0.4, 0.0, 0.0, 0.3, 0.0, 0.0]), [0.4f32.to_bits(), 0.3f32.to_bits()]);
    }

    use super::*;

    // Non-SOURCE_OVER brush composite mirrors `pixels::composite`: full coverage COPY replaces the
    // destination; MIN takes the per-channel minimum; ADD saturates. (Browser-reachable blend of a
    // gradient/bitmap fill — the fuzz `brushes-08/09/10` class.)
    #[test]
    fn blend_src_modes() {
        let dst = [10u8, 200, 50, 255];
        let src = [100.0f32, 20.0, 240.0, 128.0]; // premul float 0..255
        assert_eq!(blend_src(dst, src, 1.0, Blend::Copy), [100, 20, 240, 128]); // replace
        assert_eq!(blend_src(dst, src, 1.0, Blend::Min), [10, 147, 50, 255]); // min(dst, src over white)
        assert_eq!(blend_src(dst, src, 1.0, Blend::Add), [110, 220, 255, 255]); // saturating add
        // Half coverage COPY lerps dst→src by c (= pixels::composite's Copy arm).
        assert_eq!(blend_src(dst, src, 0.5, Blend::Copy), [55, 110, 145, 192]);
    }

    // `over`'s opaque-source fast path (rounded source alpha 255 → the rounded source, independent of
    // the destination) must equal the general source-over formula for EVERY input it triggers on, not
    // just the corpora's. The general form is evaluated inline and compared to `over`'s shortcut.
    #[test]
    fn over_opaque_matches_general() {
        let general = |dst: [u8; 4], src: [f32; 4]| -> [u8; 4] {
            let s = src.map(|v: f32| v.round_ties_even().clamp(0.0, 255.0));
            let sa = s[3] / 255.0;
            core::array::from_fn(|i| (s[i] + (dst[i] as f32 - dst[i] as f32 * sa)).round_ties_even().clamp(0.0, 255.0) as u8)
        };
        // Alphas that round_ties_even().clamp to exactly 255 (so the fast path fires). 254.5 ties to
        // the even 254 and is deliberately excluded.
        let alphas = [255.0f32, 254.6, 255.4, 255.5, 256.0, 300.0, 1e9];
        let chans = [-10.0f32, -0.4, 0.0, 0.5, 1.5, 127.4, 127.5, 128.5, 254.5, 255.0, 255.6, 1000.0];
        let dsts = [[0u8, 0, 0, 0], [255, 255, 255, 255], [13, 200, 77, 128], [128, 129, 254, 1]];
        for &a in &alphas {
            assert_eq!(a.round_ties_even().clamp(0.0, 255.0), 255.0, "alpha {a} must round to 255");
            for &r in &chans {
                for &g in &chans {
                    let src = [r, g, (r + g) * 0.3, a];
                    for dst in dsts {
                        assert_eq!(over(dst, src), general(dst, src), "src={src:?} dst={dst:?}");
                    }
                }
            }
        }
    }

    // A bitmap brush scaled 40× and offset (24,16) samples the right texel centre: at device
    // (44,36) — the centre of the 40×40 cell over texel (0,0) — nearest picks texel (0,0).
    #[test]
    fn bitmap_brush_mapping() {
        // 2×2 premultiplied-opaque BGRA: texel(0,0)=red, (1,0)=green, (0,1)=blue, (1,1)=white.
        let texels = vec![
            0, 0, 255, 255, 0, 255, 0, 255, // row 0: red, green
            255, 0, 0, 255, 255, 255, 255, 255, // row 1: blue, white
        ];
        let b2d = [40.0, 0.0, 0.0, 40.0, 24.0, 16.0]; // brush→device
        let b = Brush::bitmap(texels, [2, 2], &b2d, [0, 0], 0 /*nearest*/, 1.0);
        // Cell centres: texel (0,0) spans device x∈[24,64), centre 44; y∈[16,56), centre 36.
        assert_eq!(b.source_at(44.0, 36.0, false), [0.0, 0.0, 255.0, 255.0]); // red
        assert_eq!(b.source_at(84.0, 36.0, false), [0.0, 255.0, 0.0, 255.0]); // green (texel 1,0)
        assert_eq!(b.source_at(44.0, 76.0, false), [255.0, 0.0, 0.0, 255.0]); // blue (texel 0,1)
    }

    // VALIDATION (inkbrush-02-linear, scratch): native's exact 99 swept-nib triangles (captured via
    // the DDI VB dump, 26.6 grid) rasterized through the per-triangle gradient fit must reproduce the
    // oracle. Proves the ink ±1 is tessellation (which triangles), not the fit. Remove once the
    // tessellator port lands.
    #[test]
    fn inkbrush02_native_tris_reproduce_oracle() {
        use d2d_types::contracts::{ImageDesc, PixelBuffer};
        let native_tris: [[[f32;2];3];99] = [
    [[12.0,44.0],[10.4531,43.6875],[13.5469,43.6875]],
    [[14.8281,42.8281],[13.5469,43.6875],[10.4531,43.6875]],
    [[10.4531,43.6875],[9.17188,42.8281],[14.8281,42.8281]],
    [[15.2769,42.1591],[14.8281,42.8281],[9.17188,42.8281]],
    [[9.17188,42.8281],[8.3125,41.5469],[15.2769,42.1591]],
    [[8.3125,41.5469],[8.0,40.0],[15.2769,42.1591]],
    [[8.0,40.0],[8.3125,38.4531],[15.2769,42.1591]],
    [[18.1255,37.9789],[15.3438,42.2031],[15.2769,42.1591]],
    [[18.1255,37.9789],[15.2769,42.1591],[8.3125,38.4531]],
    [[8.3125,38.4531],[8.72312,37.8409],[18.1255,37.9789]],
    [[8.72312,37.8409],[8.65625,37.7969],[18.1255,37.9789]],
    [[41.4354,37.0098],[40.625,37.1719],[40.3478,37.1159]],
    [[40.3438,37.1719],[38.0,37.0],[40.3478,37.1159]],
    [[40.3478,37.1159],[38.0,37.0],[41.4354,37.0098]],
    [[38.0,37.0],[38.0031,36.9569],[41.4354,37.0098]],
    [[38.0031,36.9569],[37.1094,36.7781],[41.4354,37.0098]],
    [[43.7656,36.5938],[41.4531,37.0938],[41.4354,37.0098]],
    [[43.7656,36.5938],[41.4354,37.0098],[37.1094,36.7781]],
    [[43.7507,36.5232],[43.7656,36.5938],[37.1094,36.7781]],
    [[44.4844,36.375],[43.7507,36.5232],[37.1094,36.7781]],
    [[44.9481,36.064],[44.4844,36.375],[37.1094,36.7781]],
    [[20.4308,35.4721],[18.1255,37.9789],[8.65625,37.7969]],
    [[37.0938,36.8281],[32.2969,35.3125],[37.1094,36.7781]],
    [[37.1094,36.7781],[32.2969,35.3125],[44.9481,36.064]],
    [[32.2969,35.3125],[32.3134,35.2603],[44.9481,36.064]],
    [[32.3134,35.2603],[32.217,35.2408],[44.9481,36.064]],
    [[47.2656,34.7656],[44.9844,36.125],[44.9481,36.064]],
    [[47.2656,34.7656],[44.9481,36.064],[32.217,35.2408]],
    [[47.2323,34.7094],[47.2656,34.7656],[32.217,35.2408]],
    [[48.0469,34.1562],[47.2323,34.7094],[32.217,35.2408]],
    [[22.4336,34.0994],[20.4308,35.4721],[8.65625,37.7969]],
    [[48.1217,34.0448],[48.0469,34.1562],[32.217,35.2408]],
    [[32.2031,35.2812],[27.5111,33.6877],[32.217,35.2408]],
    [[32.217,35.2408],[27.5111,33.6877],[48.1217,34.0448]],
    [[24.0902,33.5281],[22.4336,34.0994],[8.65625,37.7969]],
    [[27.5111,33.6877],[25.7944,33.4172],[48.1217,34.0448]],
    [[11.6724,33.354],[25.7944,33.4172],[24.0902,33.5281]],
    [[11.6724,33.354],[24.0902,33.5281],[8.65625,37.7969]],
    [[8.65625,37.7969],[11.6094,33.3125],[11.6724,33.354]],
    [[11.6724,33.354],[12.0372,32.8168],[25.7944,33.4172]],
    [[25.7944,33.4172],[12.0372,32.8168],[48.1217,34.0448]],
    [[12.0372,32.8168],[12.0156,32.7969],[48.1217,34.0448]],
    [[50.305,31.7499],[48.1406,34.0625],[48.1217,34.0448]],
    [[50.305,31.7499],[48.1217,34.0448],[12.0156,32.7969]],
    [[50.3281,31.7344],[50.305,31.7499],[12.0156,32.7969]],
    [[50.3499,31.7019],[50.3281,31.7344],[12.0156,32.7969]],
    [[50.4219,31.625],[50.3499,31.7019],[12.0156,32.7969]],
    [[50.4093,31.6133],[50.4219,31.625],[12.0156,32.7969]],
    [[50.853,30.9518],[50.4093,31.6133],[12.0156,32.7969]],
    [[14.9122,29.7231],[50.853,30.9518],[12.0156,32.7969]],
    [[12.0156,32.7969],[14.8906,29.7031],[14.9122,29.7231]],
    [[14.9122,29.7231],[15.0,29.5938],[50.853,30.9518]],
    [[15.0,29.5938],[15.5927,29.1913],[50.853,30.9518]],
    [[15.5927,29.1913],[40.3649,29.1467],[50.853,30.9518]],
    [[15.5927,29.1913],[15.5469,29.125],[40.3649,29.1467]],
    [[39.0459,29.0499],[40.3649,29.1467],[15.5469,29.125]],
    [[40.3649,29.1467],[41.4489,28.9196],[50.853,30.9518]],
    [[41.4489,28.9196],[42.6658,28.1994],[50.853,30.9518]],
    [[34.7311,27.6963],[39.0459,29.0499],[15.5469,29.125]],
    [[53.1719,27.2656],[50.9062,30.9844],[50.853,30.9518]],
    [[53.1719,27.2656],[50.853,30.9518],[42.6658,28.1994]],
    [[18.3844,27.2623],[34.7311,27.6963],[15.5469,29.125]],
    [[53.1121,27.2293],[53.1719,27.2656],[42.6658,28.1994]],
    [[15.5469,29.125],[18.3438,27.2031],[18.3844,27.2623]],
    [[18.3844,27.2623],[19.0469,26.8125],[34.7311,27.6963]],
    [[53.4176,26.7794],[53.1121,27.2293],[42.6658,28.1994]],
    [[19.0469,26.8125],[19.311,26.7597],[34.7311,27.6963]],
    [[53.4318,26.7584],[53.4219,26.7812],[53.4176,26.7794]],
    [[53.4318,26.7584],[53.4176,26.7794],[42.6658,28.1994]],
    [[53.4375,26.75],[53.4318,26.7584],[42.6658,28.1994]],
    [[53.4393,26.7411],[53.4375,26.75],[42.6658,28.1994]],
    [[19.311,26.7597],[19.2969,26.7188],[34.7311,27.6963]],
    [[42.6658,28.1994],[44.2911,26.462],[53.4393,26.7411]],
    [[19.2969,26.7188],[29.7948,26.0678],[34.7311,27.6963]],
    [[29.7948,26.0678],[29.8125,26.0156],[34.7311,27.6963]],
    [[29.1291,25.9333],[29.7948,26.0678],[19.2969,26.7188]],
    [[29.1406,25.8594],[29.1291,25.9333],[19.2969,26.7188]],
    [[22.0151,25.8246],[29.1406,25.8594],[19.2969,26.7188]],
    [[19.2969,26.7188],[22.0,25.7812],[22.0151,25.8246]],
    [[22.0151,25.8246],[23.0656,25.6124],[29.1406,25.8594]],
    [[23.0656,25.6124],[23.0625,25.5625],[29.1406,25.8594]],
    [[23.0625,25.5625],[26.5659,25.5301],[29.1406,25.8594]],
    [[25.7062,25.4556],[26.5659,25.5301],[23.0625,25.5625]],
    [[26.5659,25.5301],[26.5781,25.4531],[29.1406,25.8594]],
    [[25.7062,25.4556],[25.9531,25.4062],[26.5659,25.5301]],
    [[23.0625,25.5625],[25.7031,25.4062],[25.7062,25.4556]],
    [[44.2911,26.462],[46.1887,23.3389],[53.4393,26.7411]],
    [[55.6719,21.5938],[53.4393,26.7411],[46.1887,23.3389]],
    [[55.6596,21.5884],[55.6719,21.5938],[46.1887,23.3389]],
    [[55.6875,21.5469],[55.6596,21.5884],[46.1887,23.3389]],
    [[56.0,20.0],[55.6875,21.5469],[46.1887,23.3389]],
    [[55.6875,18.4531],[56.0,20.0],[46.1887,23.3389]],
    [[48.3404,18.4116],[55.6875,18.4531],[46.1887,23.3389]],
    [[46.1887,23.3389],[48.3281,18.4062],[48.3404,18.4116]],
    [[54.8281,17.1719],[55.6875,18.4531],[48.3404,18.4116]],
    [[48.3404,18.4116],[49.1719,17.1719],[54.8281,17.1719]],
    [[53.5469,16.3125],[54.8281,17.1719],[49.1719,17.1719]],
    [[49.1719,17.1719],[50.4531,16.3125],[53.5469,16.3125]],
    [[50.4531,16.3125],[52.0,16.0],[53.5469,16.3125]],
    ];
        let desc = ImageDesc { width: 64, height: 64,
            format: d2d_types::abi::DXGI_FORMAT_B8G8R8A8_UNORM, alpha_mode: d2d_types::abi::D2D1_ALPHA_MODE_PREMULTIPLIED,
            dpi_x: 96.0, dpi_y: 96.0 };
        let mut buf = PixelBuffer { desc, stride: 64*4, data: vec![255u8; 64*64*4] };
        let stops = [(0.0f32, [1.0,0.1,0.05,1.0]), (1.0, [0.1,0.2,1.0,1.0])];
        let brush = Brush::linear(&stops, 0, 0, 1.0, [8.0,8.0], [56.0,56.0], &[1.0,0.0,0.0,1.0,0.0,0.0]);
        for t in native_tris { fill_triangle_brush(&mut buf, t, &brush, Blend::SourceOver); }
        let px = |x:usize,y:usize| { let i=y*256+x*4; [buf.data[i],buf.data[i+1],buf.data[i+2],buf.data[i+3]] };
        assert_eq!(px(54,17), [154,40,121,255], "(54,17)");
        assert_eq!(px(29,26), [114,36,159,255], "(29,26)");
        assert_eq!(px(53,26), [174,42,102,255], "(53,26)");
    }
}
