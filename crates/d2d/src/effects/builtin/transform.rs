//! Phase 2C built-in effects — transform / generator / lighting family, reproduced from
//! native Direct2D 10.0.26100 on WARP. Each effect is a
//! pixel kernel `eval(scenePos) -> premultiplied-linear RGBA f32` over the effect's own device
//! space; the engine (2B) positions the output. [`to_bitmap`] bakes a kernel into a
//! premultiplied-BGRA8 [`Bitmap`] (round-half-even, WARP's 8bpc intermediate store) so the
//! existing byte-exact `DrawImage` path composites it.
//!
//! Owned here: Flood, Atlas, Turbulence (SVG feTurbulence), the six lighting effects, and the
//! resampler shared by Scale / DPI compensation / 3D transform / 3D perspective. Crop, Border,
//! Tile and the 2D affine effect are 2B's. Transform sampling uses WARP's triangle attributes,
//! cubic bilinear windows, multisample offsets, and anisotropic probe arithmetic.

use crate::bitmap::Bitmap;
use d2d_types::abi::{D2D1_ALPHA_MODE_PREMULTIPLIED, DXGI_FORMAT_B8G8R8A8_UNORM};

/// Interpolation mode shared by the transform effects.
#[derive(Clone, Copy, PartialEq)]
pub enum Interp {
    Nearest,
    Linear,
    Cubic,
    Anisotropic,
    MultiSampleLinear,
    HqCubic(f32),
}

impl Interp {
    pub fn from_mode(m: u32) -> Self {
        match m {
            0 => Interp::Nearest,
            2 => Interp::Cubic,
            3 => Interp::MultiSampleLinear,
            4 => Interp::Anisotropic,
            5 => Interp::HqCubic(0.0), // HIGH_QUALITY_CUBIC, default sharpness
            _ => Interp::Linear,
        }
    }
}

/// How the resampler treats sample positions outside the input: SOFT zero-fills (transparent
/// black), HARD clamps the sample to the edge texel.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Border {
    Soft,
    Hard,
}

impl Border {
    pub fn from_mode(m: u32) -> Self {
        if m == 1 { Border::Hard } else { Border::Soft }
    }
}

/// A byte or float input image at device origin `(ox, oy)`, as the effect graph hands an
/// effect its single input. Samples come back as premultiplied RGBA, preserving float range.
pub struct Input<'a> {
    data: &'a [u8],
    bitmap: &'a Bitmap,
    w: i32,
    h: i32,
    ox: i32,
    oy: i32,
    texture_size: [i32; 2],
    texture_offset: [i32; 2],
    output_precision: u32,
    rne_address: bool,
}

impl<'a> Input<'a> {
    pub fn new(bitmap: &'a Bitmap, ox: i32, oy: i32) -> Self {
        Input { bitmap, data: &bitmap.premul_bgra, w: bitmap.width as i32, h: bitmap.height as i32, ox, oy, texture_size: [bitmap.width as i32, bitmap.height as i32], texture_offset: [0; 2], output_precision: 1, rne_address: false }
    }

    /// Preserve the UNORM sampler's fractional result when the shader writes a float target.
    pub(crate) fn with_float_output(self) -> Self { self.with_output_precision(5) }

    pub(crate) fn with_output_precision(mut self, precision: u32) -> Self { self.output_precision = precision; self }

    /// The rotated DrawImage composite resample: WARP's geometry setup kernel rounds the Q8 texel
    /// address to nearest (round-to-nearest of `(u·W + offset)·256`, traced on a rotated DrawImage),
    /// where an effect's own `Sample()` floors.
    pub(crate) fn with_rne_address(mut self) -> Self { self.rne_address = true; self }

    // Native generated intermediates use power-of-two atlas dimensions with a 512-texel minimum.
    pub(crate) fn in_atlas(mut self) -> Self {
        self.texture_size = self.texture_size.map(|n| (n as u32).next_power_of_two().max(512) as i32);
        self
    }

    pub(crate) fn with_texture_offset(mut self, offset: [i32; 2]) -> Self {
        self.texture_offset = offset;
        self
    }

    fn highp(&self) -> bool { crate::float::is_highp(self.bitmap.format) }

    fn bilinear_float(&self, x: (i32, i32), y: (i32, i32), border: Border) -> [f32; 4] {
        crate::sampling::bilinear_f32(x, y, |x, y| {
            let (x, y) = if border == Border::Hard { (x.clamp(0, self.w - 1), y.clamp(0, self.h - 1)) } else { (x, y) };
            self.bitmap.texel_rgba(x, y)
        })
    }

    /// Premultiplied-RGBA bytes of texel `(ix, iy)` in input space. Soft: `None` outside the image
    /// (zero-fill); Hard: edge texel (clamp).
    fn texel(&self, ix: i32, iy: i32, border: Border) -> Option<[i32; 4]> {
        let (x, y) = match border {
            Border::Hard => (ix.clamp(0, self.w - 1), iy.clamp(0, self.h - 1)),
            Border::Soft => {
                if ix < 0 || iy < 0 || ix >= self.w || iy >= self.h {
                    return None;
                }
                (ix, iy)
            }
        };
        let i = ((y * self.w + x) * 4) as usize;
        // stored BGRA -> RGBA
        Some([self.data[i + 2] as i32, self.data[i + 1] as i32, self.data[i] as i32, self.data[i + 3] as i32])
    }

    /// Whether device point `(x, y)` falls on a texel of this image (the lighting output rect).
    fn covers(&self, x: f32, y: f32) -> bool {
        let (ix, iy) = (x.floor() as i32 - self.ox, y.floor() as i32 - self.oy);
        ix >= 0 && iy >= 0 && ix < self.w && iy < self.h
    }

    /// Input alpha (premultiplied = straight for the alpha channel) at device point `(x, y)`,
    /// point-sampled, zero outside — the lighting surface-height read (point centre tap).
    fn alpha_at(&self, x: f32, y: f32) -> f32 {
        let ix = x.floor() as i32 - self.ox;
        let iy = y.floor() as i32 - self.oy;
        match self.texel(ix, iy, Border::Soft) {
            Some(t) => t[3] as f32 / 255.0,
            None => 0.0,
        }
    }

    /// Bilinear alpha at input-local texel-space `(lx, ly)` (texel centres at integer+0.5), edge
    /// texel clamped — the LINEAR `sample` the lighting pixel shader does on the height map. Only the
    /// alpha channel is reconstructed. Uses WARP's JIT-confirmed effect sampler ([`crate::effects::sample`]).
    fn bilinear_alpha(&self, lx: f32, ly: f32) -> f32 {
        crate::effects::sample::sample_linear_f32(lx, ly, |x, y| {
            let (cx, cy) = (x.clamp(0, self.w - 1), y.clamp(0, self.h - 1));
            let i = ((cy * self.w + cx) * 4) as usize;
            [0.0, 0.0, 0.0, crate::effects::sample::unorm_to_f32(self.data[i + 3])]
        })[3]
    }

    /// Nearest sample (R2) at device sample position `(sx, sy)`.
    fn point(&self, sx: f32, sy: f32, border: Border) -> [f32; 4] {
        let ix = sx.floor() as i32 - self.ox;
        let iy = sy.floor() as i32 - self.oy;
        match self.texel(ix, iy, border) {
            Some(t) => t.map(|c| c as f32 / 255.0),
            None => [0.0; 4],
        }
    }

    /// Bilinear sample (R3): the transform effect is a pixel shader, so it drives WARP's in-shader
    /// linear `sample` ([`crate::effects::sample`]: Q8 texcoord weight, UNORM8 corners via
    /// `·257/65536`, f32 product-sum) — *not* DrawBitmap's fixed-function integer `lerp8`.
    fn bilinear(&self, sx: f32, sy: f32, border: Border) -> [f32; 4] {
        crate::effects::sample::sample_linear(sx - self.ox as f32, sy - self.oy as f32, |x, y| {
            self.texel(x, y, border).unwrap_or([0; 4])
        })
    }

    /// Resample at device sample position with the chosen filter/border.
    fn sample(&self, sx: f32, sy: f32, interp: Interp, border: Border) -> [f32; 4] {
        match interp {
            Interp::Nearest => self.point(sx, sy, border),
            Interp::Linear | Interp::Anisotropic | Interp::MultiSampleLinear => self.bilinear(sx, sy, border),
            Interp::Cubic => self.cubic([sx - self.ox as f32, sy - self.oy as f32], [(sx - self.ox as f32) / self.w as f32, (sy - self.oy as f32) / self.h as f32], border),
            // HIGH_QUALITY_CUBIC: WARP's Catmull-Rom combine ([`crate::sampling::cubic_combine`]) over
            // the border-aware texel fetch (SOFT → 0 outside, so the cubic negative lobes feather the
            // edge; HARD → clamp). Texel centres at integer+0.5, matching the bilinear path.
            Interp::HqCubic(sharpness) => crate::sampling::cubic_combine_sharpness(true, sx - self.ox as f32, sy - self.oy as f32, sharpness, |x, y| {
                self.texel(x, y, border).unwrap_or([0; 4])
            }),
        }
    }

    pub(crate) fn cubic(&self, phase: [f32; 2], uv: [f32; 2], border: Border) -> [f32; 4] {
        self.cubic_with_steps(phase, uv, self.texture_size.map(|v| 1.0 / v as f32), border)
    }

    fn cubic_with_steps(&self, phase: [f32; 2], uv: [f32; 2], steps: [f32; 2], border: Border) -> [f32; 4] {
        // WARP's Q8 address floors x of a power-of-two-wide texture and y of one that is a power of two on both axes,
        // and rounds any other axis (cub-i5-7x5-*; a 3x16 HIGH_QUALITY_CUBIC magnification rounds its y).
        let [tw, th] = self.texture_size.map(|v| (v as u32).is_power_of_two());
        let axis = |p: f32, u: f32, dim: i32, step: f32, offset: i32, pow2: bool| {
            let f = (p - 0.5) - (p - 0.5).floor();
            let w = crate::sampling::cubic_weights(f, 0.0);
            let g = w[1] + w[2];
            let base = u - f * step;
            let qs = [base - step, base + (w[2] / g) * step, base + 2.0 * step].map(|v| {
                let a = (v.clamp(-10.0, 10.0) * dim as f32 - 0.5) * 256.0;
                let q = if pow2 { a.floor() } else { a.round_ties_even() } as i32;
                ((q >> 8) + offset, q & 255)
            });
            (qs, [w[0], g, w[3]])
        };
        let (xs, wx) = axis(phase[0], uv[0], self.texture_size[0], steps[0], self.texture_offset[0], tw);
        let (ys, wy) = axis(phase[1], uv[1], self.texture_size[1], steps[1], self.texture_offset[1], tw && th);
        self.cubic_windows(xs, ys, wx, wy, border)
    }

    fn cubic_windows(&self, xs: [(i32, i32); 3], ys: [(i32, i32); 3], wx: [f32; 3], wy: [f32; 3], border: Border) -> [f32; 4] {
        if self.highp() {
            crate::sampling::cubic_windows_f32(xs, ys, wx, wy, |x, y| {
                let (x, y) = if border == Border::Hard { (x.clamp(0, self.w - 1), y.clamp(0, self.h - 1)) } else { (x, y) };
                self.bitmap.texel_rgba(x, y)
            })
        } else {
            let fetch = |x, y| self.texel(x, y, border).unwrap_or([0; 4]);
            if self.output_precision > 1 { crate::sampling::hqcubic_windows_unclamped(xs, ys, wx, wy, fetch) }
            else { crate::sampling::hqcubic_windows(xs, ys, wx, wy, fetch) }
        }
    }

    /// Resample at an input-**local** texel-space position (texel centres at integer+0.5, origin
    /// already removed), the entry the affine resampler uses after computing WARP's normalized
    /// texcoord plane. Same per-mode kernels as [`sample`] without the device `-ox,-oy` shift.
    fn sample_local(&self, cx: f32, cy: f32, interp: Interp, border: Border) -> [f32; 4] {
        match interp {
            Interp::Nearest => match self.texel(cx.floor() as i32, cy.floor() as i32, border) {
                Some(t) => t.map(|c| c as f32 / 255.0),
                None => [0.0; 4],
            },
            Interp::Linear | Interp::Anisotropic | Interp::MultiSampleLinear => crate::effects::sample::sample_linear(cx, cy, |x, y| self.texel(x, y, border).unwrap_or([0; 4])),
            Interp::Cubic => self.cubic([cx, cy], [cx / self.w as f32, cy / self.h as f32], border),
            Interp::HqCubic(sharpness) => crate::sampling::cubic_combine_sharpness(true, cx, cy, sharpness, |x, y| self.texel(x, y, border).unwrap_or([0; 4])),
        }
    }
}

// --- output ----------------------------------------------------------------------------------

/// Bake a kernel into a premultiplied-BGRA8 bitmap of `width × height`, evaluating it at the
/// device scene position of each texel centre offset by `(ox, oy)` (the effect's content origin
/// relative to the bitmap). Channels are clamped and rounded half-to-even — WARP's 8bpc store.
pub fn to_bitmap(width: u32, height: u32, ox: f32, oy: f32, eval: impl FnMut(f32, f32) -> [f32; 4]) -> Bitmap {
    to_bitmap_precision(width, height, ox, oy, false, eval)
}

pub(crate) fn to_bitmap_precision(width: u32, height: u32, ox: f32, oy: f32, highp: bool, mut eval: impl FnMut(f32, f32) -> [f32; 4]) -> Bitmap {
    if highp {
        return super::super::float_bitmap(width, height, (0..height).flat_map(|y| (0..width).map(move |x| (x, y))).map(|(x, y)| eval(x as f32 + 0.5 + ox, y as f32 + 0.5 + oy)));
    }
    let mut px = vec![0u8; (width * height * 4) as usize];
    let q = |v: f32| (v * 255.0).round_ties_even().clamp(0.0, 255.0) as u8;
    for y in 0..height {
        for x in 0..width {
            let [r, g, b, a] = eval(x as f32 + 0.5 + ox, y as f32 + 0.5 + oy);
            let i = ((y * width + x) * 4) as usize;
            px[i..i + 4].copy_from_slice(&[q(b), q(g), q(r), q(a)]);
        }
    }
    Bitmap {
        width,
        height,
        premul_bgra: px,
        format: DXGI_FORMAT_B8G8R8A8_UNORM,
        alpha: D2D1_ALPHA_MODE_PREMULTIPLIED,
        dpi: (96.0, 96.0),
    }
}

// --- Flood -----------------------------------------------------------------------------------

/// Flood: every texel is the color constant verbatim (no premultiply step — the stored
/// premultiplied texel is the raw RGBA, D2D permits superluminous). Output logically infinite.
pub fn flood(color: [f32; 4]) -> impl Fn(f32, f32) -> [f32; 4] {
    move |_x, _y| color
}

// --- Atlas -----------------------------------------------------------------------------------

/// Atlas: passthrough of the input shifted so the (rounded) input rect's origin maps to 0, then
/// clipped to `cropRect = (0,0,w,h)`. `rect` is the rounded INPUT_RECT in device space.
pub fn atlas<'a>(input: &'a Input<'a>, rect: [i32; 4]) -> impl Fn(f32, f32) -> [f32; 4] + 'a {
    let [l, t, r, b] = rect;
    let (w, h) = (r - l, b - t);
    move |x, y| {
        let (cx, cy) = (x.floor() as i32, y.floor() as i32);
        if cx < 0 || cy < 0 || cx >= w || cy >= h {
            return [0.0; 4];
        }
        input.point(x + l as f32, y + t as f32, Border::Soft)
    }
}

// Expand the transformed edges by their pixel-square support, then intersect them. The
// endpoint order is significant: the inverse/forward round trip preserves D2D's vertex rounding.
fn expand_sampler_quad(p: [[f32; 2]; 4]) -> [[f32; 2]; 4] {
    let sub = |a: [f32; 2], b: [f32; 2]| [a[0] - b[0], a[1] - b[1]];
    let edge = |a: usize, b: usize, toward: [f32; 2]| {
        let d = sub(p[b], p[a]);
        let inv = 1.0 / crate::warpmath::sqrt(d[0] * d[0] + d[1] * d[1]);
        let n = [d[1] * inv, -d[0] * inv];
        let dot = n[1] * toward[1] + n[0] * toward[0];
        let sign = dot / dot.abs();
        let angle = crate::crt::system_crt::acosf((d[0] * inv + d[1] * inv * 0.0).abs()) as f64;
        let amount = (libm::sin(angle + core::f64::consts::FRAC_PI_4) * libm::sin(core::f64::consts::FRAC_PI_4)) as f32;
        let shift = n.map(|v| (sign * v) * amount);
        [p[a], p[b]].map(|v| [v[0] + shift[0], v[1] + shift[1]])
    };
    let top = edge(0, 1, sub(p[1], p[3]));
    let left = edge(0, 3, sub(p[3], p[1]));
    let right = edge(2, 1, sub(p[1], p[3]));
    let bottom = edge(2, 3, sub(p[3], p[1]));
    let intersect = |a: [[f32; 2]; 2], b: [[f32; 2]; 2]| {
        let da = sub(a[1], a[0]);
        let db = sub(b[1], b[0]);
        let d = sub(a[0], b[0]);
        let t = (d[1] * db[0] - d[0] * db[1]) / (da[0] * db[1] - db[0] * da[1]);
        [da[0] * t + a[0][0], da[1] * t + a[0][1]]
    };
    [intersect(top, left), intersect(top, right), intersect(bottom, right), intersect(bottom, left)]
}

/// Transform-filter attributes over WARP's two triangles. Cubic carries phase and UV separately;
/// MSL carries four offset UVs; anisotropy computes derivatives across each 2×2 pixel quad.
struct SamplerQuad {
    triangles: Vec<([[i64; 2]; 3], [[f32; 3]; 16])>,
    inverse: [f32; 9],
    origin: [f32; 2],
    mips: Option<Vec<(u32, u32, Vec<u8>)>>,
    float_mips: Option<Vec<(u32, u32, Vec<[f32; 4]>)>>,
    interp: Interp,
    projective_samples: bool,
    /// The sampled texture's size and the device position of its texel (0, 0): the texcoords are normalized by it.
    dims: [u32; 2],
    texture_origin: [i64; 2],
}

fn sampler_geometry(input: &Input, m: [f32; 9], scale: bool, border: Border, interp: Interp) -> ([f32; 9], [[f32; 2]; 4], [[f32; 3]; 4], bool) {
    let inverse = [
        m[4] * m[8] - m[5] * m[7], m[2] * m[7] - m[1] * m[8], m[1] * m[5] - m[2] * m[4],
        m[5] * m[6] - m[3] * m[8], m[0] * m[8] - m[2] * m[6], m[2] * m[3] - m[0] * m[5],
        m[3] * m[7] - m[4] * m[6], m[1] * m[6] - m[0] * m[7], m[0] * m[4] - m[1] * m[3],
    ];
    let map = |p: [f32; 2], m: [f32; 9]| -> [f32; 3] {
        core::array::from_fn(|c| (p[0] * m[c] + p[1] * m[3 + c]) + m[6 + c])
    };
    let original = [[input.ox as f32, input.oy as f32], [input.ox as f32 + input.w as f32, input.oy as f32],
        [input.ox as f32 + input.w as f32, input.oy as f32 + input.h as f32], [input.ox as f32, input.oy as f32 + input.h as f32]];
    let horizon = original.iter().any(|&p| map(p, m)[2] <= 0.0);
    let pad = if border == Border::Hard || interp == Interp::Nearest { [0.0; 2] } else if interp == Interp::Linear { [0.5; 2] } else if matches!(interp, Interp::Anisotropic | Interp::MultiSampleLinear) {
        [(inverse[0].abs() + inverse[3].abs()) / inverse[8].abs() * 0.5 + 0.5,
         (inverse[1].abs() + inverse[4].abs()) / inverse[8].abs() * 0.5 + 0.5]
    } else { [1.5; 2] };
    let (l, t) = (input.ox as f32 - pad[0], input.oy as f32 - pad[1]);
    let (r, b) = (input.ox as f32 + input.w as f32 + pad[0], input.oy as f32 + input.h as f32 + pad[1]);
    let mut source = [[l, t], [r, t], [r, b], [l, b]];
    if !horizon && !scale && border == Border::Soft && matches!(interp, Interp::Anisotropic | Interp::MultiSampleLinear) {
        let l = input.ox as f32 - 0.5;
        let t = input.oy as f32 - 0.5;
        let r = input.ox as f32 + input.w as f32 + 0.5;
        let b = input.oy as f32 + input.h as f32 + 0.5;
        let quad = [[l, t], [r, t], [r, b], [l, b]].map(|p| {
            let q = map(p, m);
            [q[0] / q[2], q[1] / q[2]]
        });
        source = expand_sampler_quad(quad).map(|p| {
            let q = map(p, inverse);
            [q[0] / q[2], q[1] / q[2]]
        });
    }
    // System32 WARP's sampler-placement-linear-apron-horizon capture drops an apron crossing W=0.
    if horizon || source.iter().any(|&p| map(p, m)[2] <= 0.0) { source = original; }
    let position = source.map(|p| map(p, m));
    (inverse, source, position, horizon)
}

/// The device-space extent `(min, max)` of the sampler quad, `None` past the horizon.
pub(crate) fn projective_extent(input: &Input, m: [f64; 16], interp: Interp, border: Border) -> Option<([f32; 2], [f32; 2])> {
    let h = [m[0], m[1], m[3], m[4], m[5], m[7], m[12], m[13], m[15]].map(|v| v as f32);
    let (_, _, p, horizon) = sampler_geometry(input, h, false, border, interp);
    if horizon { return None; }
    let p = p.map(|p| [p[0] / p[2], p[1] / p[2]]);
    if p.iter().flatten().any(|v| !v.is_finite() || v.abs() >= i32::MAX as f32) { return None; }
    let lo = core::array::from_fn(|c| p.iter().map(|p| p[c]).fold(f32::INFINITY, f32::min));
    let hi = core::array::from_fn(|c| p.iter().map(|p| p[c]).fold(f32::NEG_INFINITY, f32::max));
    Some((lo, hi))
}

pub(crate) fn projective_bounds(input: &Input, m: [f64; 16], interp: Interp, border: Border) -> Option<[i32; 4]> {
    let (lo, hi) = projective_extent(input, m, interp, border)?;
    Some([(lo[0] + 0.5).floor() as i32, (lo[1] + 0.5).floor() as i32, (hi[0] - 0.5).ceil() as i32, (hi[1] - 0.5).ceil() as i32])
}

impl SamplerQuad {
    fn new(input: &Input, m: [f32; 9], scale: bool, border: Border, origin: [f32; 2], interp: Interp, clip: Option<[f32; 4]>, viewport: Option<[f32; 4]>) -> Option<Self> {
        if input.w == 0 || input.h == 0 { return None; }
        let (inverse, source, mut position, horizon) = sampler_geometry(input, m, scale, border, interp);
        let map = |p: [f32; 2], m: [f32; 9]| -> [f32; 3] {
            core::array::from_fn(|c| (p[0] * m[c] + p[1] * m[3 + c]) + m[6 + c])
        };
        let mut texture_origin = [(input.ox + input.texture_offset[0]) as i64, (input.oy + input.texture_offset[1]) as i64];
        let mut dims = input.texture_size.map(|v| v as u32);
        let mut mip_data = None;
        let mut float_mip_data = None;
        if interp == Interp::Anisotropic && border == Border::Soft {
            // The graph requests enough mip levels for the largest inverse derivative at a
            // primitive corner. Its mip atlas aligns a horizontally centered input to that grid.
            let mut coverage = 1.0f32;
            for p in position {
                let p = [p[0] / p[2], p[1] / p[2]];
                let a = map(p, inverse);
                let a = [a[0] / a[2], a[1] / a[2]];
                for delta in [[1.0, 0.0], [0.0, 1.0]] {
                    let b = map([p[0] + delta[0], p[1] + delta[1]], inverse);
                    let d = [a[0] - b[0] / b[2], a[1] - b[1] / b[2]];
                    coverage = coverage.max(crate::warpmath::sqrt(d[0] * d[0] + d[1] * d[1]));
                }
            }
            let mut grid = 1i64;
            while (grid as f32) < coverage && grid < 128 { grid *= 2; }
            if grid > 1 {
                let shift = if input.w <= 0x20_0000 { (grid - input.w as i64 % grid) / 2 - (input.ox as i64).rem_euclid(grid) } else { 0 };
                let mut rect = [input.ox as i64, input.oy as i64, input.ox as i64 + input.w as i64, input.oy as i64 + input.h as i64];
                let original = rect;
                for c in 0..2 {
                    let offset = if c == 0 { shift } else { 0 };
                    rect[c] -= (rect[c] + offset).rem_euclid(grid);
                    rect[c + 2] += (-(rect[c + 2] + offset)).rem_euclid(grid);
                    if original[c] - rect[c] <= grid / 2 { rect[c] -= grid; }
                    if rect[c + 2] - original[c + 2] <= grid / 2 { rect[c + 2] += grid; }
                }
                texture_origin = [rect[0], rect[1]];
                dims = [(rect[2] - rect[0]) as u32, (rect[3] - rect[1]) as u32].map(|n| n.next_power_of_two().max(512));
                if input.highp() {
                    let mut data = vec![[0.0; 4]; dims[0] as usize * dims[1] as usize];
                    for y in 0..input.h {
                        for x in 0..input.w {
                            let dst = (y as i64 + input.oy as i64 - rect[1]) as usize * dims[0] as usize + (x as i64 + input.ox as i64 - rect[0]) as usize;
                            data[dst] = input.bitmap.texel_rgba(x, y);
                        }
                    }
                    float_mip_data = Some(data);
                } else {
                    let mut data = vec![0; dims[0] as usize * dims[1] as usize * 4];
                    for y in 0..input.h {
                        let dst = ((y as i64 + input.oy as i64 - rect[1]) as usize * dims[0] as usize + (input.ox as i64 - rect[0]) as usize) * 4;
                        let src = y as usize * input.w as usize * 4;
                        data[dst..dst + input.w as usize * 4].copy_from_slice(&input.data[src..src + input.w as usize * 4]);
                    }
                    mip_data = Some(data);
                }
            }
        }
        let mut uv = source.map(|p| [(p[0] - texture_origin[0] as f32) / dims[0] as f32, (p[1] - texture_origin[1] as f32) / dims[1] as f32]);
        // Anisotropic MAGNIFICATION (no mip atlas): native draws the bitmap direct (TryDrawImageFastPath →
        // DoDrawImageFastPath); its vertex buffer snaps the quad corners in DEVICE space and forms the texcoord
        // from cb0 = adj(full M) per vertex (offset inside the matrix, `/det` per vertex, base = v1·step). We split
        // it in local space with an offset-free adj — the same value, a different f32 grouping that fit a plane
        // ~1 Q8 off at a tie (r27 32530/32712/25657; proven vs the op8cap DDI cb0 + vertex buffer). Minification
        // builds a mip atlas and tessellates differently, so it (and MSL, other filters) keep the local split.
        // The Scale/Dpi EFFECT graph realizes aniso through its own pipeline (a different vertex buffer than the
        // DrawImage fast path this cb0/VB was captured from), so it keeps the local split too; only the direct
        // DrawImage path passes a `viewport` and takes this branch.
        let aniso_mag = interp == Interp::Anisotropic && mip_data.is_none() && float_mip_data.is_none() && viewport.is_some();
        if scale {
            // The graph's rectangular Scale primitive is rounded to integer device pixels.
            for i in 0..4 {
                let p = if aniso_mag {
                    // Device-space snap of the corner, then clamp to the viewport: native's vertex buffer caps the
                    // quad at the viewport edge (DDI g667: an 11x1 at 1.5x whose apron reaches x -6 has its left
                    // vertex at the viewport's left, x 0). Clamping, not a Sutherland-Hodgman fan — the rect stays a
                    // rect. The attribute plane is affine, so the clamp does not move it (an interior quad is a
                    // no-op, r27 32530 exact); it keeps the far corner's plane precision bounded and matches native.
                    let [vl, vt, vr, vb] = viewport.unwrap_or([f32::NEG_INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::INFINITY]);
                    let d = [position[i][0] + origin[0], position[i][1] + origin[1]];
                    let s = [if i == 0 || i == 3 { d[0].floor() } else { d[0].ceil() },
                             if i < 2 { d[1].floor() } else { d[1].ceil() }];
                    let c = [s[0].clamp(vl, vr), s[1].clamp(vt, vb)];
                    [c[0] - origin[0], c[1] - origin[1]]
                } else if matches!(interp, Interp::Anisotropic | Interp::MultiSampleLinear) {
                    [if i == 0 || i == 3 { position[i][0].floor() } else { position[i][0].ceil() },
                     if i < 2 { position[i][1].floor() } else { position[i][1].ceil() }]
                } else { [position[i][0].round_ties_even(), position[i][1].round_ties_even()] };
                position[i] = [p[0], p[1], 1.0];
                uv[i] = if aniso_mag {
                    let v1 = [p[0] + origin[0], p[1] + origin[1]];
                    let mut full = inverse;
                    full[6] = -inverse[3] * origin[1] - inverse[0] * origin[0];
                    full[7] = -inverse[1] * origin[0] - inverse[4] * origin[1];
                    let q = map(v1, full);
                    core::array::from_fn(|c| {
                        let step = 1.0 / dims[c] as f32;
                        (q[c] / q[2] - v1[c]) * step + (v1[c] - texture_origin[c] as f32) * step
                    })
                } else {
                    let q = map(p, inverse);
                    core::array::from_fn(|c| {
                        let step = 1.0 / dims[c] as f32;
                        (q[c] / q[2] - p[c]) * step + (p[c] - texture_origin[c] as f32) * step
                    })
                };
            }
        }
        let indices = if scale { [[1, 0, 2], [3, 0, 2]] } else { [[0, 3, 2], [0, 1, 2]] };
        let extreme = position.iter().any(|p| (p[0] / p[2] + origin[0]).abs() > 1_048_576.0 || (p[1] / p[2] + origin[1]).abs() > 1_048_576.0);
        if extreme && !horizon {
            let [l, t, r, b] = clip?;
            position = [[l, t, 1.0], [r, t, 1.0], [r, b, 1.0], [l, b, 1.0]];
            uv = position.map(|p| {
                let q = map([p[0], p[1]], inverse);
                core::array::from_fn(|c| {
                    let step = 1.0 / dims[c] as f32;
                    (q[c] / q[2] - p[c]) * step + (p[c] - texture_origin[c] as f32) * step
                })
            });
        }
        let clipped = horizon || position.iter().any(|p| p[2] <= 0.0)
            || clip.is_some_and(|[l,t,r,b]| position.iter().any(|p| p[0] / p[2] < l || p[0] / p[2] > r || p[1] / p[2] < t || p[1] / p[2] > b));
        let mut triangles = Vec::new();
        if aniso_mag && scale {
            // Native draws the direct anisotropic magnification quad with its corners snapped in DEVICE space and
            // the texcoord from cb0 = adj(M) (done in the scale block above). The quad's affine texcoord plane is
            // clip-invariant — a viewport clip only reorders/refits the same plane, and WARP's f32 refit landed a
            // Q8 step off at a bottom edge (r27 32095), fixing no draw while regressing that one — so we fit the
            // plane over the unclipped device corners and let sample() cover only in-target pixels.
            for ids in indices {
                let tri: [[f32; 4]; 3] = core::array::from_fn(|k| {
                    let i = ids[k];
                    [position[i][0] + origin[0], position[i][1] + origin[1], uv[i][0], uv[i][1]]
                });
                let snapped: [[i64; 2]; 3] = tri.map(|v| [(v[0] * 256.0).round_ties_even() as i64, (v[1] * 256.0).round_ties_even() as i64]);
                // Native fits the plane from the provoking (min-x) vertex, as the existing path does.
                let mut o = [0usize, 1, 2];
                o.rotate_left((0..3).min_by_key(|&i| snapped[i][0]).unwrap());
                let q0 = o.map(|i| snapped[i]);
                let area = (q0[1][0] - q0[0][0]) * (q0[2][1] - q0[0][1]) - (q0[2][0] - q0[0][0]) * (q0[1][1] - q0[0][1]);
                if area == 0 { continue; }
                if area < 0 { o.swap(1, 2); }
                let attrs: [[f32; 16]; 3] = tri.map(|v| {
                    let mut a = [0.0; 16];
                    a[0] = v[0];
                    a[1] = v[1];
                    a[2] = 1.0;
                    a[3] = v[2];
                    a[4] = v[3];
                    a[13] = 1.0;
                    a[14] = 1.0 / dims[0] as f32;
                    a[15] = 1.0 / dims[1] as f32;
                    a
                });
                let p = o.map(|i| snapped[i]);
                let planes = core::array::from_fn(|c| crate::warp::plane::setup_triangle_attr(p.map(|v| v[0]), p.map(|v| v[1]), o.map(|i| attrs[i][c])));
                triangles.push((p, planes));
            }
        } else {
        for ids in indices {
            let mut polygon: Vec<[f32; 20]> = ids.map(|i| {
                let p = position[i];
                let mut v = [0.0; 20];
                v[..7].copy_from_slice(&[p[0], p[1], p[2], uv[i][0], uv[i][1], p[0], p[1]]);
                for (j, offset) in [[0.375, -0.125], [-0.125, -0.375], [-0.375, 0.125], [0.125, 0.375]].iter().enumerate() {
                    let xy = [p[0] / p[2], p[1] / p[2]];
                    let q = map([xy[0] + offset[0], xy[1] + offset[1]], inverse);
                    for c in 0..2 {
                        let step = 1.0 / [input.w, input.h][c] as f32;
                        v[12 + 2*j + c] = (xy[c] - [input.ox, input.oy][c] as f32) * step - (xy[c] - q[c] / q[2]) * step;
                    }
                }
                v
            }).to_vec();
            if clipped {
                let [l, t, r, b] = clip?;
                for v in &mut polygon {
                    v[0] = v[0] * (2.0 / (r - l)) + v[2] * (-(r + l) / (r - l));
                    v[1] = v[1] * (-2.0 / (b - t)) + v[2] * ((b + t) / (b - t));
                    v[7] = v[2];
                    v[8] = v[0] + v[2];
                    v[9] = v[2] - v[0];
                    v[10] = v[2] - v[1];
                    v[11] = v[1] + v[2];
                }
                let anchor = polygon.iter().rev().find(|v| v[7..12].iter().all(|&d| d >= 0.0)).copied();
                for plane in 7..12 {
                    let mut result = Vec::new();
                    for i in 0..polygon.len() {
                        let a = polygon[i];
                        let b = polygon[(i + 1) % polygon.len()];
                        let (da, db) = (a[plane], b[plane]);
                        if da >= 0.0 { result.push(a); }
                        if (da >= 0.0) != (db >= 0.0) {
                            // WARP always interpolates from the inside endpoint. Differences
                            // round in f32; the multiply/divide/add then run in f64.
                            let (a, b, da, db) = if da >= 0.0 { (a, b, da, db) } else { (b, a, db, da) };
                            result.push(core::array::from_fn(|c| ((b[c] - a[c]) as f64 * da as f64 / (da - db) as f64 + a[c] as f64) as f32));
                        }
                    }
                    polygon = result;
                }
                if let Some(anchor) = anchor.filter(|_| horizon) {
                    if let Some(first) = polygon.iter().position(|v| *v == anchor) { polygon.rotate_left(first); }
                }
            }
            for i in 1..polygon.len().saturating_sub(1) {
                let vertices = [polygon[0], polygon[i], polygon[i + 1]];
                if vertices.iter().any(|p| p[2] <= 0.0 || !(p[0] / p[2]).is_finite() || !(p[1] / p[2]).is_finite()) { continue; }
                let snapped = vertices.map(|p| {
                    let xy = if clipped {
                        let [l, t, r, b] = clip.unwrap();
                        let rw = 1.0 / p[2];
                        [(p[0] * rw) * ((r - l) * 0.5) + ((r + l) * 0.5 + origin[0]),
                         (p[1] * rw) * ((t - b) * 0.5) + ((b + t) * 0.5 + origin[1])]
                    } else { let rw = 1.0 / p[2]; [p[0] * rw + origin[0], p[1] * rw + origin[1]] };
                    xy.map(|v| (v * 256.0).round_ties_even() as i64)
                });
                let attrs: [[f32; 16]; 3] = vertices.map(|v| {
                    let w = v[2];
                    let mut a = [0.0; 16];
                    let rw = 1.0 / w;
                    a[..5].copy_from_slice(&if clipped { [v[5] * rw, v[6] * rw, rw, v[3] * rw, v[4] * rw] }
                        else { [v[0] * rw, v[1] * rw, rw, v[3] * rw, v[4] * rw] });
                    for j in 0..8 { a[5 + j] = v[12 + j] * rw; }
                    // Native WARP interpolates scene W and texel steps too; w*(1/w) can differ from one.
                    a[13] = w * rw;
                    a[14] = (1.0 / dims[0] as f32) * rw;
                    a[15] = (1.0 / dims[1] as f32) * rw;
                    a
                });
                let mut ids = [0, 1, 2];
                let p = snapped;
                let area = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
                if area == 0 { continue; }
                let first = (0..3).min_by_key(|&i| snapped[i][0]).unwrap();
                // Native's attribute-plane setup preserves the provoking vertex for un-clipped projective primitives.
                if (!clipped && m[2] == 0.0 && m[5] == 0.0) || horizon { ids.rotate_left(first); }
                if area < 0 { ids.swap(1, 2); }
                let p = ids.map(|i| snapped[i]);
                let planes = core::array::from_fn(|c| crate::warp::plane::setup_triangle_attr(p.map(|v| v[0]), p.map(|v| v[1]), ids.map(|i| attrs[i][c])));
                triangles.push((p, planes));
            }
        }
        }
        let mips = (!input.highp() && interp == Interp::Anisotropic).then(|| crate::bitmap::mip_chain(&crate::sampling::Texture { data: mip_data.as_deref().unwrap_or(input.data), width: dims[0], height: dims[1] }));
        let float_mips = (input.highp() && interp == Interp::Anisotropic).then(|| {
            let data = float_mip_data.unwrap_or_else(|| (0..input.h).flat_map(|y| (0..input.w).map(move |x| input.bitmap.texel_rgba(x, y))).collect());
            crate::bitmap::float_mip_levels(dims[0], dims[1], false, 5, data)
        });
        Some(Self { triangles, inverse, origin, mips, float_mips, interp, projective_samples: m[2] != 0.0 || m[5] != 0.0, dims, texture_origin })
    }

    fn sample(&self, input: &Input, x: f32, y: f32, border: Border) -> [f32; 4] {
        let p = [x + self.origin[0], y + self.origin[1]];
        for (vertices, planes) in self.triangles.iter().rev() {
            let inside = (0..3).all(|i| {
                let a = vertices[i].map(|v| v as f32 / 256.0);
                let b = vertices[(i + 1) % 3].map(|v| v as f32 / 256.0);
                let cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
                cross > 0.0 || (cross == 0.0 && (b[1] < a[1] || (b[1] == a[1] && b[0] > a[0])))
            });
            if !inside { continue; }
            let v = planes.map(|a| a[0] * p[0] + (a[1] * p[1] + a[2]));
            let reciprocal = 1.0 / v[2];
            let scene_w = v[13] * reciprocal;
            let scene = [(v[0] * reciprocal) / scene_w, (v[1] * reciprocal) / scene_w];
            let steps = [v[14] * reciprocal, v[15] * reciprocal];
            if self.interp == Interp::MultiSampleLinear {
                let texture = crate::sampling::Texture { data: input.data, width: input.w as u32, height: input.h as u32 };
                let extend = if border == Border::Soft { crate::sampling::Extend::Transparent } else { crate::sampling::Extend::Clamp };
                let mut acc = [0.0; 4];
                for j in 0..4 {
                    let q: [(i32, i32); 2] = core::array::from_fn(|c| {
                        let u = if self.projective_samples {
                            let offset = [[0.375, -0.125], [-0.125, -0.375], [-0.375, 0.125], [0.125, 0.375]][j];
                            let map = |x, y| -> [f32; 3] { core::array::from_fn(|c| (y * self.inverse[3 + c] + x * self.inverse[c]) + self.inverse[6 + c]) };
                            let a = map(scene[0], scene[1]);
                            let b = map(scene[0] + offset[0], scene[1] + offset[1]);
                            v[3 + c] * reciprocal - (a[c] / a[2] - b[c] / b[2]) * steps[c]
                        } else { v[5 + 2 * j + c] * reciprocal };
                        let q = (((u).clamp(-10.0, 10.0) * input.texture_size[c] as f32 - 0.5) * 256.0).floor() as i32;
                        (q >> 8, q & 255)
                    });
                    if input.highp() {
                        let c = input.bilinear_float(q[0], q[1], border);
                        for ch in 0..4 { acc[ch] += c[ch]; }
                    } else {
                        let v = texture.bilinear_v128_all([extend; 2], q[0], q[1]);
                        for ch in 0..4 {
                            acc[ch] += crate::sampling::norm_v128(v[ch]);
                        }
                    }
                }
                return if input.highp() { acc.map(|v| v * 0.25) } else { [acc[2] * 0.25, acc[1] * 0.25, acc[0] * 0.25, acc[3] * 0.25] };
            }
            if !input.highp() && self.interp == Interp::Linear {
                let texture = crate::sampling::Texture { data: input.data, width: input.w as u32, height: input.h as u32 };
                let extend = if border == Border::Soft { crate::sampling::Extend::Transparent } else { crate::sampling::Extend::Clamp };
                let q: [(i32, i32); 2] = core::array::from_fn(|c| {
                    // The normalized texcoord is recovered at the realized-texture size (an intermediate's
                    // power-of-two atlas, else content) via `self.dims` (from `input.texture_size`), with
                    // the crop's `texture_origin` offset; the fetch stays content-sized.
                    let t = ((v[3 + c] * reciprocal).clamp(-10.0, 10.0) * self.dims[c] as f32 - 0.5) * 256.0;
                    let q = if input.rne_address { t.round_ties_even() as i32 } else { t.floor() as i32 };
                    let q = q + ((self.texture_origin[c] - [input.ox, input.oy][c] as i64) * 256) as i32;
                    (q >> 8, q & 255)
                });
                let v4 = texture.bilinear_v128_all([extend; 2], q[0], q[1]);
                let c: [f32; 4] = core::array::from_fn(|ch| {
                    let v = v4[ch];
                    if input.output_precision > 1 && !(input.output_precision == 2 && ch == 3) { crate::sampling::norm_v128(v) } else { (v >> 7) as f32 / 255.0 }
                });
                return [c[2], c[1], c[0], c[3]];
            }
            if self.interp == Interp::Nearest || (input.highp() && self.interp == Interp::Linear) {
                let uv = [v[3] * reciprocal, v[4] * reciprocal];
                if self.interp == Interp::Nearest {
                    // The texel under the texcoord, in the (atlas) texture, then in the image.
                    let at = |c: usize| (uv[c] * self.dims[c] as f32).floor() as i32 + (self.texture_origin[c] - [input.ox, input.oy][c] as i64) as i32;
                    let (ix, iy) = (at(0), at(1));
                    let (ix, iy) = if border == Border::Hard { (ix.clamp(0, input.w - 1), iy.clamp(0, input.h - 1)) } else { (ix, iy) };
                    return super::super::bitmap_texel(input.bitmap, ix, iy);
                }
                // A non-power-of-two direct bitmap rounds (rne) its Q8 sub-texel address under float LINEAR, as
                // the 8-bit `bilinear` path does (:700) — floor reads 1/256 toward the up-left neighbour and
                // leaves a ≤0.6/255 residual on a float target (gb2 Affine2D-LINEAR koh/sc29 to exact).
                let q = core::array::from_fn::<_, 2, _>(|c| {
                    let t = (uv[c].clamp(-10.0, 10.0) * input.texture_size[c] as f32 - 0.5) * 256.0;
                    let q = if input.rne_address { t.round_ties_even() } else { t.floor() } as i32;
                    (q >> 8, q & 255)
                });
                return input.bilinear_float(q[0], q[1], border);
            }
            if self.mips.is_some() || self.float_mips.is_some() {
                let (mw, mh, levels) = if let Some(mips) = &self.mips { (mips[0].0, mips[0].1, mips.len()) } else { let mips = self.float_mips.as_ref().unwrap(); (mips[0].0, mips[0].1, mips.len()) };
                let eval = |x, y| {
                    let a = planes.map(|a| a[0] * x + (a[1] * y + a[2]));
                    { let rw = 1.0 / a[2]; [a[3] * rw, a[4] * rw] }
                };
                let (qx, qy) = ((p[0] * 0.5).floor() * 2.0 + 0.5, (p[1] * 0.5).floor() * 2.0 + 0.5);
                let (v0, vx, vy) = (eval(qx, qy), eval(qx + 1.0, qy), eval(qx, qy + 1.0));
                let dims = [mw as f32, mh as f32];
                let dx: [f32; 2] = core::array::from_fn(|i| (vx[i] - v0[i]) * dims[i]);
                let dy: [f32; 2] = core::array::from_fn(|i| (vy[i] - v0[i]) * dims[i]);
                let (dx2, dy2) = (dx[0] * dx[0] + dx[1] * dx[1], dy[0] * dy[0] + dy[1] * dy[1]);
                let major2 = dx2.max(dy2);
                let major = crate::warpmath::sqrt(major2);
                let area = (dx[0] * dy[1] - dy[0] * dx[1]).abs();
                let minor = (area / major).max(major / 16.0);
                let ratio = if minor < 1.0 { major.max(1.0) } else { (major2 / area).min(16.0) };
                let direction = if dx2 >= dy2 { dx } else { dy }.map(|v| v * (1.0 / major));
                let lod = crate::warpmath::mip_lod(minor.max(1.0) * minor.max(1.0), (levels - 1) as f32);
                let lodq = (lod * 256.0).round_ties_even() as usize;
                let uv = [v[3] * reciprocal, v[4] * reciprocal];
                let extend = if border == Border::Soft { crate::sampling::Extend::Transparent } else { crate::sampling::Extend::Clamp };
                if let Some(mips) = &self.float_mips {
                    let sample = |level: usize| {
                        let (w, h, data) = &mips[level.min(mips.len() - 1)];
                        crate::sampling::anisotropic_f32(uv, [*w as f32, *h as f32], direction, ratio, |x, y| {
                            if border == Border::Soft && (x < 0 || y < 0 || x >= *w as i32 || y >= *h as i32) { return [0.0; 4]; }
                            let (x, y) = (extend.wrap(x, *w as i32), extend.wrap(y, *h as i32));
                            data[(y as u32 * w + x as u32) as usize]
                        })
                    };
                    let lo = sample(lodq >> 8);
                    let hi = sample((lodq >> 8) + 1);
                    let f = (lodq & 255) as f32 / 256.0;
                    return core::array::from_fn(|ch| lo[ch] + (hi[ch] - lo[ch]) * f);
                }
                let mips = self.mips.as_ref().unwrap();
                let sample = |level: usize| {
                    let (width, height, data) = &mips[level.min(mips.len() - 1)];
                    crate::sampling::Texture { data, width: *width, height: *height }.anisotropic_v128_extend(uv, direction, ratio, [extend; 2])
                };
                let lo = sample(lodq >> 8);
                let hi = sample((lodq >> 8) + 1);
                let c: [f32; 4] = core::array::from_fn(|ch| {
                    let v = lo[ch] + (((hi[ch] - lo[ch]) * (lodq & 255) as i32) >> 8);
                    if input.output_precision > 1 && !(input.output_precision == 2 && ch == 3) { crate::sampling::norm_v128(v) } else { (v >> 7) as f32 / 255.0 }
                });
                return [c[2], c[1], c[0], c[3]];
            }
            let m = self.inverse;
            let q: [f32; 3] = core::array::from_fn(|c| (scene[0] * m[c] + scene[1] * m[3 + c]) + m[6 + c]);
            return input.cubic_with_steps([q[0] / q[2], q[1] / q[2]], [v[3] * reciprocal, v[4] * reciprocal], steps, border);
        }
        [0.0; 4]
    }
}

// --- resampler (Scale / DPI compensation / 3D) -----------------------------------------------

/// Invert a D2D 3×2 affine `[a,b,c,d,e,f]` (`x' = a·x + c·y + e`, `y' = b·x + d·y + f`). `None`
/// if singular (output then empty/transparent per the spec).
fn invert_affine(m: [f32; 6]) -> Option<[f32; 6]> {
    let [a, b, c, d, e, f] = m;
    let det = a * d - b * c;
    if det == 0.0 {
        return None;
    }
    let (ia, ib, ic, id) = (d / det, -b / det, -c / det, a / det);
    Some([ia, ib, ic, id, -(e * ia + f * ic), -(e * ib + f * id)])
}

/// WARP's effect texcoord: it carries the inverse map as a texture-**normalized** plane
/// (coefficients `inv/texSize`, rounded to f32) and recovers the input-local texel coordinate as
/// `(dUdx·px + dUdy·py + Ubase)·texSize`. That `÷texSize … ×texSize` round-trip, evaluated in f32,
/// lands exactly on integer texel boundaries where the texel-direct `inv·p` falls ±1 ULP off and
/// flips the Q8 `floor` (captured off a native 3DTransform draw).
/// `iu,iv` are the texel-space inverse rows `(ia,ic,ie)`,`(ib,id,if_)`; the input origin folds into
/// the base so the result is already input-local (feed [`Input::sample_local`]).
#[inline]
fn warp_local_texcoord(iu: [f64; 3], iv: [f64; 3], input: &Input, px: f32, py: f32) -> (f32, f32) {
    let (w, h) = (input.w as f64, input.h as f64);
    let (nux, nuy, nub) = ((iu[0] / w) as f32, (iu[1] / w) as f32, ((iu[2] - input.ox as f64) / w) as f32);
    let (nvx, nvy, nvb) = ((iv[0] / h) as f32, (iv[1] / h) as f32, ((iv[2] - input.oy as f64) / h) as f32);
    ((nux * px + nuy * py + nub) * input.w as f32, (nvx * px + nvy * py + nvb) * input.h as f32)
}

/// Affine resampler (R1 + R2/R3): inverse-map each output scene position to the input, then
/// sample. Singular matrix ⇒ all transparent.
pub fn affine<'a>(input: &'a Input<'a>, matrix: [f32; 6], interp: Interp, border: Border) -> impl Fn(f32, f32) -> [f32; 4] + 'a {
    affine_at(input, matrix, interp, border, [0.0; 2], true)
}

pub(crate) fn affine_at<'a>(input: &'a Input<'a>, matrix: [f32; 6], interp: Interp, border: Border, origin: [f32; 2], scale: bool) -> impl Fn(f32, f32) -> [f32; 4] + 'a {
    affine_at_vp(input, matrix, interp, border, origin, scale, None)
}

/// As [`affine_at`], with the device viewport (target extent). An ANISOTROPIC magnification quad is then
/// clipped to the viewport the way native's vertex buffer is (its clip is against the viewport), fixing the
/// texcoord/winding at a clipped edge (x=0). Device-space; only used for the aniso magnification path.
pub(crate) fn affine_at_vp<'a>(input: &'a Input<'a>, matrix: [f32; 6], interp: Interp, border: Border, origin: [f32; 2], scale: bool, viewport: Option<[f32; 4]>) -> impl Fn(f32, f32) -> [f32; 4] + 'a {
    let inv = invert_affine(matrix).filter(|_| input.w > 0 && input.h > 0);
    let cubic = inv.and_then(|_| (!matches!(interp, Interp::HqCubic(_)) || !scale)
        .then(|| SamplerQuad::new(input, [matrix[0], matrix[1], 0.0, matrix[2], matrix[3], 0.0, matrix[4], matrix[5], 1.0], scale, border, origin, interp, None, viewport.filter(|_| interp == Interp::Anisotropic))).flatten());
    move |x, y| match inv {
        None => [0.0; 4],
        Some([ia, ib, ic, id, ie, if_]) => {
            if let Some(cubic) = &cubic { return cubic.sample(input, x, y, border); }
            if let Interp::HqCubic(sharpness) = interp {
                // Scale's shader divides the phase separately from its normalized varying. Keep
                // the subtraction order: even the outer windows retain their Q8 bilinear address.
                if matrix[1] == 0.0 && matrix[2] == 0.0 {
                    let axis = |p: f32, scale: f32, offset: f32, origin: i32, dim: i32| {
                        let phase = (p - offset) / scale;
                        let f = (phase - 0.5) - (phase - 0.5).floor();
                        let w = crate::sampling::cubic_weights(f, sharpness);
                        let g = w[1] + w[2];
                        let step = 1.0 / dim as f32;
                        let base = ((phase - f) - p) * step + (p - origin as f32) * step;
                        let qs = [base - step, base + (w[2] / g) * step, base + 2.0 * step].map(|u| {
                            let q = ((u.clamp(-10.0, 10.0) * dim as f32 - 0.5) * 256.0).floor() as i32;
                            (q >> 8, q & 255)
                        });
                        (qs, [w[0], g, w[3]])
                    };
                    let (mut xs, wx) = axis(x, matrix[0], matrix[4], input.ox + input.texture_offset[0], input.texture_size[0]);
                    let (mut ys, wy) = axis(y, matrix[3], matrix[5], input.oy + input.texture_offset[1], input.texture_size[1]);
                    for q in &mut xs { q.0 += input.texture_offset[0]; }
                    for q in &mut ys { q.0 += input.texture_offset[1]; }
                    return input.cubic_windows(xs, ys, wx, wy, border);
                }
            }
            let (cx, cy) = warp_local_texcoord([ia as f64, ic as f64, ie as f64], [ib as f64, id as f64, if_ as f64], input, x, y);
            // HARD keeps to the transformed input rect, the pixels whose centres it holds. The sampler quad bounds that
            // itself; this path is taken when the quad runs past WARP's ±2^20 guard band with no viewport to clip it to
            // (r27 32796: a HARD x1e5 Scale of a 9-row bitmap filled every row with its clamped edge).
            if border == Border::Hard && !(cx >= 0.0 && cx < input.w as f32 && cy >= 0.0 && cy < input.h as f32) {
                return [0.0; 4];
            }
            input.sample_local(cx, cy, interp, border)
        }
    }
}

// Native Direct2D 10.0.26100 HQ downscale shaders: four taps per loop side, normalize their sum.
pub(crate) fn hq_downscale_axis(input: &Input, axis: usize, scale: f32, offset: f32, sharpness: f32, border: Border, out: [i32; 4], precision: u32) -> Bitmap {
    let filter = |input: &Input, x: f32, y: f32, axis: usize, scale: f32, offset: f32| {
        let p = [x, y];
        let phase = (p[axis] - offset) / scale;
        let dims = input.texture_size;
        let origins = [input.ox, input.oy];
        let step = 1.0 / dims[axis] as f32;
        let other = 1 - axis;
        let uq = (((p[other] - origins[other] as f32) * 256.0).floor() as i32) >> 8;
        crate::sampling::hq_downscale_axis(phase, p[axis], (p[axis] - origins[axis] as f32) * step, step, scale, sharpness, !input.highp(), |u| {
            // Native HQ axis shader's WARP point fetch (hqdraw-v.asm, 0xadf..0xbe4).
            let q = (((u.clamp(-10.0, 10.0) * dims[axis] as f32) * 256.0).floor() as i32) >> 8;
            let mut at = [uq; 2]; at[axis] = q;
            if input.highp() {
                if border == Border::Hard { at = [at[0].clamp(0, input.w - 1), at[1].clamp(0, input.h - 1)]; }
                input.bitmap.texel_rgba(at[0], at[1])
            } else { input.texel(at[0], at[1], border).unwrap_or([0; 4]).map(|v| crate::sampling::norm_v128(v * 128)) }
        })
    };
    to_bitmap_precision((out[2] - out[0]) as u32, (out[3] - out[1]) as u32, out[0] as f32, out[1] as f32, precision > 1,
        |x, y| super::super::quantize_to(precision, filter(input, x, y, axis, scale, offset)))
}

pub(crate) fn hq_downscale(input: &Input, matrix: [f32; 6], sharpness: f32, border: Border, out: [i32; 4], precision: u32) -> Bitmap {
    if matrix[0] > 1.0 || matrix[3] > 1.0 {
        // Native mixed-scale DDI: downsample the shrinking axis, then run the magnifying Scale shader.
        let axis = usize::from(matrix[3] < 1.0);
        let mut bounds = [input.ox, input.oy, input.ox + input.w, input.oy + input.h];
        bounds[axis] = (bounds[axis] as f32 * matrix[axis * 3] + matrix[axis + 4] - 2.0).floor() as i32;
        bounds[axis + 2] = (bounds[axis + 2] as f32 * matrix[axis * 3] + matrix[axis + 4] + 2.0).ceil() as i32;
        let intermediate = hq_downscale_axis(input, axis, matrix[axis * 3], matrix[axis + 4], sharpness, border, bounds, precision);
        let mut offset = [0; 2]; offset[1 - axis] = -3;
        let mid = Input::new(&intermediate, bounds[0], bounds[1]).in_atlas().with_texture_offset(offset);
        let mid = if precision > 1 { mid.with_float_output() } else { mid };
        let mut remaining = matrix; remaining[axis * 3] = 1.0; remaining[axis + 4] = 0.0;
        let sample = affine_at(&mid, remaining, Interp::HqCubic(sharpness), border, [0.0; 2], true);
        return to_bitmap_precision((out[2] - out[0]) as u32, (out[3] - out[1]) as u32, out[0] as f32, out[1] as f32, precision > 1, sample);
    }
    let top = (input.oy as f32 * matrix[3] + matrix[5] - 2.0).floor() as i32;
    let bottom = ((input.oy + input.h) as f32 * matrix[3] + matrix[5] + 2.0).ceil() as i32;
    let intermediate = hq_downscale_axis(input, 1, matrix[3], matrix[5], sharpness, border, [input.ox, top, input.ox + input.w, bottom], precision);
    let mid = Input::new(&intermediate, input.ox, top).in_atlas();
    hq_downscale_axis(&mid, 0, matrix[0], matrix[4], sharpness, border, out, precision)
}

/// Scale `M = [sx,0, 0,sy, (1−sx)·cx, (1−sy)·cy]`.
pub fn scale_matrix(scale: [f32; 2], center: [f32; 2]) -> [f32; 6] {
    [scale[0], 0.0, 0.0, scale[1], (1.0 - scale[0]) * center[0], (1.0 - scale[1]) * center[1]]
}

/// DPI compensation `M = [ctxDpi/inDpi, …]` (factor 1 on an axis whose input DPI ≤ 0).
pub fn dpi_matrix(ctx_dpi: [f32; 2], in_dpi: [f32; 2]) -> [f32; 6] {
    let k = |c: f32, i: f32| if i > 0.0 { c / i } else { 1.0 };
    [k(ctx_dpi[0], in_dpi[0]), 0.0, 0.0, k(ctx_dpi[1], in_dpi[1]), 0.0, 0.0]
}

/// Projective resampler (3D transform / 3D perspective). D2D uses row vectors with a row-major
/// 4×4 `m` (translation in indices 12..14). The input lies at `z = 0`, so the forward map reduces
/// to the 2D homography `(ix,iy,1)·H = (X,Y,W)`, `screen = (X/W, Y/W)` with
/// `H = [[m0,m1,m3],[m4,m5,m7],[m12,m13,m15]]`. We inverse-map each output scene position through
/// `H⁻¹` and sample.
pub fn projective<'a>(input: &'a Input<'a>, m: [f64; 16], interp: Interp, border: Border) -> impl Fn(f32, f32) -> [f32; 4] + 'a {
    projective_at(input, m, interp, border, [0.0; 2])
}

pub(crate) fn projective_at<'a>(input: &'a Input<'a>, m: [f64; 16], interp: Interp, border: Border, origin: [f32; 2]) -> impl Fn(f32, f32) -> [f32; 4] + 'a {
    projective_rect_at(input, m, interp, border, origin, None)
}

pub(crate) fn projective_rect_at<'a>(input: &'a Input<'a>, m: [f64; 16], interp: Interp, border: Border, origin: [f32; 2], clip: Option<[f32; 4]>) -> impl Fn(f32, f32) -> [f32; 4] + 'a {
    let h = [m[0], m[1], m[3], m[4], m[5], m[7], m[12], m[13], m[15]];
    let inv = invert3x3(h).filter(|_| input.w > 0 && input.h > 0);
    let cubic = inv.and_then(|_| (input.highp() || matches!(interp, Interp::Nearest) || (interp == Interp::Linear && border == Border::Soft) || matches!(interp, Interp::Cubic | Interp::Anisotropic | Interp::MultiSampleLinear))
        .then(|| SamplerQuad::new(input, h.map(|v| v as f32), false, border, origin, interp, clip, None)).flatten());
    move |x, y| match inv {
        None => [0.0; 4],
        Some(ih) => {
            if let Some(cubic) = &cubic { return cubic.sample(input, x, y, border); }
            // Affine case (projective row maps every point to the same w = ih[8]): the inverse map
            // is a plane, so use WARP's normalized texcoord (the divide folds into the coefficients).
            if ih[2] == 0.0 && ih[5] == 0.0 {
                let wd = ih[8];
                let (cx, cy) = warp_local_texcoord([ih[0] / wd, ih[3] / wd, ih[6] / wd], [ih[1] / wd, ih[4] / wd, ih[7] / wd], input, x, y);
                return input.sample_local(cx, cy, interp, border);
            }
            let (px, py) = (x as f64, y as f64);
            // (sx,sy,1) · H⁻¹  (row vector), then divide by the homogeneous component.
            let a = px * ih[0] + py * ih[3] + ih[6];
            let b = px * ih[1] + py * ih[4] + ih[7];
            let w = px * ih[2] + py * ih[5] + ih[8];
            if w == 0.0 {
                return [0.0; 4];
            }
            input.sample((a / w) as f32, (b / w) as f32, interp, border)
        }
    }
}

// --- 3D perspective matrix build -------------------------------------------------------------

/// Row-major 4×4 multiply (row-vector convention, so `p·(A·B) = (p·A)·B`).
fn mat_mul(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut r = [0.0f32; 16];
    for i in 0..4 {
        for j in 0..4 {
            r[i * 4 + j] = (0..4).map(|k| a[i * 4 + k] * b[k * 4 + j]).sum();
        }
    }
    r
}

fn translation(x: f32, y: f32, z: f32) -> [f32; 16] {
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, z, 1.0]
}

/// D2D `Matrix4x4F` rotations take degrees; row-vector convention.
fn rot_x(deg: f32) -> [f32; 16] {
    let (s, c) = (crate::crt::system_crt::sinf(deg * f32::from_bits(0x3c8e_fa35)), crate::crt::system_crt::cosf(deg * f32::from_bits(0x3c8e_fa35)));
    [1.0, 0.0, 0.0, 0.0, 0.0, c, s, 0.0, 0.0, -s, c, 0.0, 0.0, 0.0, 0.0, 1.0]
}
fn rot_y(deg: f32) -> [f32; 16] {
    let (s, c) = (crate::crt::system_crt::sinf(deg * f32::from_bits(0x3c8e_fa35)), crate::crt::system_crt::cosf(deg * f32::from_bits(0x3c8e_fa35)));
    [c, 0.0, -s, 0.0, 0.0, 1.0, 0.0, 0.0, s, 0.0, c, 0.0, 0.0, 0.0, 0.0, 1.0]
}
fn rot_z(deg: f32) -> [f32; 16] {
    let (s, c) = (crate::crt::system_crt::sinf(deg * f32::from_bits(0x3c8e_fa35)), crate::crt::system_crt::cosf(deg * f32::from_bits(0x3c8e_fa35)));
    [c, s, 0.0, 0.0, -s, c, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
}

/// `Matrix4x4F::PerspectiveProjection(depth)`: `m[2][3] = −1/depth` (so `w' = 1 − z/depth`).
fn perspective(depth: f32) -> [f32; 16] {
    let p = if depth != 0.0 { -1.0 / depth } else { 0.0 };
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, p, 0.0, 0.0, 0.0, 1.0]
}

/// 3D-perspective transform matrix (`PerspectiveTransform3DEffect.BuildTransformMatrix`):
/// `T(local)·T(−rotOrigin)·Rx·Ry·Rz·T(rotOrigin)·T(global)·[T(−pOrigin)·Persp(depth)·T(pOrigin)]`.
pub fn perspective_matrix(depth: f32, p_origin: [f32; 2], local: [f32; 3], global: [f32; 3], rot_origin: [f32; 3], rotation_deg: [f32; 3]) -> [f64; 16] {
    let p = |v: [f32; 3]| (v[0], v[1], v[2]);
    let (lx, ly, lz) = p(local);
    let (rx, ry, rz) = p(rot_origin);
    let (gx, gy, gz) = p(global);
    let (pox, poy) = (p_origin[0], p_origin[1]);
    let persp = mat_mul(mat_mul(translation(-pox, -poy, 0.0), perspective(depth)), translation(pox, poy, 0.0));
    let mut m = translation(lx, ly, lz);
    m = mat_mul(m, translation(-rx, -ry, -rz));
    m = mat_mul(m, rot_x(rotation_deg[0]));
    m = mat_mul(m, rot_y(rotation_deg[1]));
    m = mat_mul(m, rot_z(rotation_deg[2]));
    m = mat_mul(m, translation(rx, ry, rz));
    m = mat_mul(m, translation(gx, gy, gz));
    mat_mul(m, persp).map(|v| v as f64)
}

/// 3×3 inverse (row-major), `None` if singular.
fn invert3x3(m: [f64; 9]) -> Option<[f64; 9]> {
    let c0 = m[4] * m[8] - m[5] * m[7];
    let c1 = m[5] * m[6] - m[3] * m[8];
    let c2 = m[3] * m[7] - m[4] * m[6];
    let det = m[0] * c0 + m[1] * c1 + m[2] * c2;
    if det == 0.0 {
        return None;
    }
    Some([
        c0 / det,
        (m[2] * m[7] - m[1] * m[8]) / det,
        (m[1] * m[5] - m[2] * m[4]) / det,
        c1 / det,
        (m[0] * m[8] - m[2] * m[6]) / det,
        (m[2] * m[3] - m[0] * m[5]) / det,
        c2 / det,
        (m[1] * m[6] - m[0] * m[7]) / det,
        (m[0] * m[4] - m[1] * m[3]) / det,
    ])
}

// --- lighting --------------------------------------------------------------------------------

/// In-shader `normalize`: `dp3`+`rsq`+`mul` on WARP. `rsq` is `1/sqrt` (exact square root, then divide) and the
/// vector is scaled by it, not divided per lane — see [`crate::warpmath`].
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    crate::warpmath::normalize3(v)
}

fn safe_normalize3(v: [f32; 3]) -> [f32; 3] {
    if v[0] * v[0] + v[1] * v[1] + v[2] * v[2] > 0.0 {
        normalize3(v)
    } else {
        [0.0; 3]
    }
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// SVG edge/corner gradient factors per region (row-major TL,T,TR,L,C,R,BL,B,BR), folded into
/// `−surfaceScale` to scale gx/gy, as native's lighting vertex data carries them: interior ¼,
/// straight edges ½/⅓, corners ⅔.
const SOBEL_REGION_SCALE: [(f32, f32); 9] = [
    (2.0 / 3.0, 2.0 / 3.0), (1.0 / 3.0, 0.5), (2.0 / 3.0, 2.0 / 3.0),
    (0.5, 1.0 / 3.0), (0.25, 0.25), (0.5, 1.0 / 3.0),
    (2.0 / 3.0, 2.0 / 3.0), (1.0 / 3.0, 0.5), (2.0 / 3.0, 2.0 / 3.0),
];

/// The 9 region kernels, each up to 6 bilinear taps `(dx, dy, gx-weight, gy-weight)`, as native's
/// lighting vertex data carries them. Each 3×3 Sobel is realised
/// as bilinear alpha taps at fractional (0.33, exactly f32 `0x3ea8f5c3`) offsets — distinct from an
/// exact integer Sobel, which is why the point-sampled form was ≤1 LSB off at steep surface scales.
#[rustfmt::skip]
const SOBEL_TAPS: [[(f32, f32, f32, f32); 6]; 9] = [
    [(0.0, 0.33, -3.0, 0.0), (1.0, 0.33, 3.0, 0.0), (0.33, 0.0, 0.0, -3.0), (0.33, 1.0, 0.0, 3.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)],
    [(-1.0, 0.33, -3.0, 0.0), (1.0, 0.33, 3.0, 0.0), (-0.33, 0.0, 0.0, -3.0), (-0.33, 1.0, 0.0, 3.0), (1.0, 0.0, 0.0, -1.0), (1.0, 1.0, 0.0, 1.0)],
    [(-1.0, 0.33, -3.0, 0.0), (0.0, 0.33, 3.0, 0.0), (-0.33, 0.0, 0.0, -3.0), (-0.33, 1.0, 0.0, 3.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)],
    [(0.0, 0.33, -3.0, 0.0), (1.0, 0.33, 3.0, 0.0), (0.0, -1.0, -1.0, 0.0), (1.0, -1.0, 1.0, 0.0), (0.33, -1.0, 0.0, -3.0), (0.33, 1.0, 0.0, 3.0)],
    [(-1.0, -0.33, -3.0, 0.0), (1.0, 0.33, 3.0, 0.0), (-0.33, -1.0, 0.0, -3.0), (0.33, 1.0, 0.0, 3.0), (-1.0, 1.0, -1.0, 1.0), (1.0, -1.0, 1.0, -1.0)],
    [(-1.0, -0.33, -3.0, 0.0), (0.0, -0.33, 3.0, 0.0), (-1.0, 1.0, -1.0, 0.0), (0.0, 1.0, 1.0, 0.0), (-0.33, -1.0, 0.0, -3.0), (-0.33, 1.0, 0.0, 3.0)],
    [(0.0, -0.33, -3.0, 0.0), (1.0, -0.33, 3.0, 0.0), (0.33, -1.0, 0.0, -3.0), (0.33, 0.0, 0.0, 3.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)],
    [(-1.0, -0.33, -3.0, 0.0), (1.0, -0.33, 3.0, 0.0), (-0.33, -1.0, 0.0, -3.0), (-0.33, 0.0, 0.0, 3.0), (1.0, -1.0, 0.0, -1.0), (1.0, 0.0, 0.0, 1.0)],
    [(-1.0, -0.33, -3.0, 0.0), (0.0, -0.33, 3.0, 0.0), (-0.33, -1.0, 0.0, -3.0), (-0.33, 0.0, 0.0, 3.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)],
];

/// Surface normal (L1), as native's lighting effects compute it: pick the
/// edge/corner/interior region for the pixel's texel, accumulate the region's bilinear alpha taps
/// into gx/gy (weights pre-scaled by `−surfaceScale · regionFactor`), and `normalize(gx, gy, 1)`.
fn surface_normal(input: &Input, x: f32, y: f32, surface_scale: f32) -> [f32; 3] {
    let (lx, ly) = (x - input.ox as f32, y - input.oy as f32); // input-local texel-space coord
    let (tx, ty) = (lx.floor() as i32, ly.floor() as i32);
    let col = if tx <= 0 { 0 } else if tx >= input.w - 1 { 2 } else { 1 };
    let row = if ty <= 0 { 0 } else if ty >= input.h - 1 { 2 } else { 1 };
    let region = (row * 3 + col) as usize;
    let (fx, fy) = SOBEL_REGION_SCALE[region];
    let (mx, my) = (-surface_scale * fx, -surface_scale * fy);
    let (mut gx, mut gy) = (0.0f32, 0.0f32);
    for &(dx, dy, wz, ww) in &SOBEL_TAPS[region] {
        let a = input.bilinear_alpha(lx + dx, ly + dy);
        gx += a * (wz * mx);
        gy += a * (ww * my);
    }
    normalize3([gx, gy, 1.0])
}

/// The light setup for a pixel: the light direction `l` and, for spot lights, its attenuation.
pub enum Light {
    /// Point: light position in device space, surface z = surfaceScale·alpha.
    Point { position: [f32; 3], surface_scale: f32 },
    /// Distant: constant light direction.
    Distant { direction: [f32; 3] },
    /// Spot: like Point, plus axis / focus and the two cone cosines (see [`spot_cone_cosines`]).
    Spot { position: [f32; 3], surface_scale: f32, axis: [f32; 3], focus: f32, cos_outer: f32, cos_inner: f32 },
}

/// Spot-cone attenuation, as the native spot-light pixel shader computes it: `falloff · cone`,
/// both in **cosine space**. `lDotS = saturate(−L·axis)`; `falloff = lDotS^focus`; `cone` is the
/// Hermite smoothstep of `lDotS` from the inner cone cosine up to the outer — *not* the managed /
/// SVG `acos`-angle smoothstep, which differs by ~1 LSB at the cone edge (`fx-transform-19`).
fn spot_attenuation(l: [f32; 3], axis: [f32; 3], focus: f32, cos_outer: f32, cos_inner: f32) -> f32 {
    let l_dot_s = (-dot3(l, axis)).clamp(0.0, 1.0);
    let falloff = crate::warpmath::pow(l_dot_s, focus);
    let t = ((l_dot_s - cos_inner) / (cos_outer - cos_inner)).clamp(0.0, 1.0);
    falloff * (t * t * (3.0 - 2.0 * t))
}

/// The two spot-cone cosines from the limiting-cone-angle degrees, as native's cone-angle setter
/// computes them: clamp to `[−90, 90]°`, take `|angle|`, then the outer cosine
/// `cos(angle°)` and the inner `cos((angle+1)°)` (`cos(1.2·angle°)` below 5°). Radians are
/// `(f64)deg·π/180` cast to f32, as native (`cosf` for the f32 branches, `cos` for the <5° inner).
pub fn spot_cone_cosines(angle_deg: f32) -> (f32, f32) {
    let a = angle_deg.clamp(-90.0, 90.0).abs() as f64;
    let rad = |d: f64| d * std::f64::consts::PI / 180.0;
    let outer = crate::crt::system_crt::cosf(rad(a) as f32);
    let inner = if a >= 5.0 { crate::crt::system_crt::cosf(rad(a + 1.0) as f32) } else { crate::crt::system_crt::cosf(rad(a * 1.2) as f32) };
    (outer, inner)
}

/// Light direction and attenuation at a surface point.
fn light_vec(light: &Light, x: f32, y: f32, center_alpha: f32) -> ([f32; 3], f32) {
    match *light {
        Light::Distant { direction } => (direction, 1.0),
        Light::Point { position, surface_scale } => {
            let sp = [x, y, surface_scale * center_alpha];
            (safe_normalize3([position[0] - sp[0], position[1] - sp[1], position[2] - sp[2]]), 1.0)
        }
        Light::Spot { position, surface_scale, axis, focus, cos_outer, cos_inner } => {
            let sp = [x, y, surface_scale * center_alpha];
            let l = safe_normalize3([position[0] - sp[0], position[1] - sp[1], position[2] - sp[2]]);
            (l, spot_attenuation(l, axis, focus, cos_outer, cos_inner))
        }
    }
}

/// Diffuse lighting (L3): `out.rgb = max(N·L,0)·diffuseColor·att`, `out.a = 1` (opaque).
/// `diffuse_color` is already `color·diffuseConstant`.
pub fn diffuse<'a>(input: &'a Input<'a>, light: Light, diffuse_color: [f32; 3], surface_scale: f32) -> impl Fn(f32, f32) -> [f32; 4] + 'a {
    move |x, y| {
        if !input.covers(x, y) {
            return [0.0; 4];
        }
        let n = surface_normal(input, x, y, surface_scale);
        let (l, att) = light_vec(&light, x, y, input.alpha_at(x, y));
        let f = dot3(n, l).max(0.0) * att;
        [f * diffuse_color[0], f * diffuse_color[1], f * diffuse_color[2], 1.0]
    }
}

/// Specular lighting (L4): `s = specularColor·pow(max(N·H,0),exp)·att`; `a = max(s.rgb)`;
/// `out = (s·a, a)` premultiplied. `specular_color` is already `color·specularConstant`.
/// For a distant light the halfway vector is precomputed into `Light::Distant{direction=H}`.
pub fn specular<'a>(input: &'a Input<'a>, light: Light, specular_color: [f32; 3], exponent: f32, surface_scale: f32) -> impl Fn(f32, f32) -> [f32; 4] + 'a {
    let distant = matches!(light, Light::Distant { .. });
    move |x, y| {
        if !input.covers(x, y) {
            return [0.0; 4];
        }
        let n = surface_normal(input, x, y, surface_scale);
        let (l, att) = light_vec(&light, x, y, input.alpha_at(x, y));
        // Distant precomputes H; point/spot form H = safeNorm(L + z).
        let h = if distant { l } else { safe_normalize3([l[0], l[1], l[2] + 1.0]) };
        let f = crate::warpmath::pow(dot3(n, h).max(0.0), exponent) * att;
        let s = [specular_color[0] * f, specular_color[1] * f, specular_color[2] * f];
        let a = s[0].max(s[1]).max(s[2]);
        [s[0] * a, s[1] * a, s[2] * a, a]
    }
}

/// Distant-light direction from azimuth/elevation degrees (L2).
pub fn distant_direction(azimuth_deg: f32, elevation_deg: f32) -> [f32; 3] {
    let az = azimuth_deg * std::f32::consts::PI / 180.0;
    let el = elevation_deg * std::f32::consts::PI / 180.0;
    let ce = crate::crt::system_crt::cosf(el);
    [crate::crt::system_crt::cosf(az) * ce, crate::crt::system_crt::sinf(az) * ce, crate::crt::system_crt::sinf(el)]
}

/// Distant specular halfway vector: `normalize(Lx, Ly, Lz+1)` with an explicit length guard,
/// as the managed effect precomputes it.
pub fn distant_halfway(direction: [f32; 3]) -> [f32; 3] {
    let v = [direction[0], direction[1], direction[2] + 1.0];
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 { [v[0] / len, v[1] / len, v[2] / len] } else { [0.0; 3] }
}

/// Spot-light axis `normalize(pointsAt − position)` (explicit-length guard; L5).
pub fn spot_axis(position: [f32; 3], points_at: [f32; 3]) -> [f32; 3] {
    let v = [points_at[0] - position[0], points_at[1] - position[1], points_at[2] - position[2]];
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 { [v[0] / len, v[1] / len, v[2] / len] } else { [0.0; 3] }
}

// --- Turbulence (SVG feTurbulence) -----------------------------------------------------------

const B_SIZE: usize = 256;
const BM: i32 = 255;
const STRIDE: usize = 514;
const PERLIN_N: f32 = 4096.0;

/// The seed-derived Perlin lattice (`514` permutation entries) and gradient table
/// (`4 × 514` 2-vectors), built exactly as `TurbulenceEffect.BuildLattice`.
pub struct Turbulence {
    lattice: [i32; STRIDE],
    gradient: Vec<[f32; 2]>, // 4 channels × 514, gradient[c*514 + i]
    base_freq: [f32; 2],
    octaves: i32,
    fractal: bool,
    rect: [f32; 4], // output rect L,T,R,B (ints, as f32)
}

fn setup_seed(mut s: i64) -> i64 {
    if s <= 0 {
        s = -(s % 2147483646) + 1;
    }
    if s > 2147483646 {
        s = 2147483646;
    }
    s
}

fn random_next(s: i64) -> i64 {
    let mut n = 16807 * (s % 127773) - 2836 * (s / 127773);
    if n <= 0 {
        n += 2147483647;
    }
    n
}

impl Turbulence {
    pub fn new(seed: i32, base_freq: [f32; 2], octaves: i32, fractal: bool, rect: [i32; 4]) -> Self {
        let mut lat = [0i32; STRIDE];
        let mut grad = vec![[0.0f32; 2]; 4 * STRIDE];
        let mut arr = [0i32; STRIDE];
        let mut gd = vec![[0.0f64; 2]; 4 * STRIDE];
        let mut num = setup_seed(seed as i64);
        for i in 0..4 {
            for j in 0..B_SIZE {
                arr[j] = j as i32;
                num = random_next(num);
                let g0 = (num % 512 - 256) as f64 / 256.0;
                num = random_next(num);
                let g1 = (num % 512 - 256) as f64 / 256.0;
                let len = (g0 * g0 + g1 * g1).sqrt();
                gd[i * STRIDE + j] = if len != 0.0 { [g0 / len, g1 / len] } else { [0.0, 0.0] };
            }
        }
        let mut j = B_SIZE;
        while {
            j -= 1;
            j > 0
        } {
            let k = arr[j];
            num = random_next(num);
            let m = (num % 256) as usize;
            arr[j] = arr[m];
            arr[m] = k;
        }
        for j in 0..258 {
            arr[B_SIZE + j] = arr[j];
            for i in 0..4 {
                gd[i * STRIDE + B_SIZE + j] = gd[i * STRIDE + j];
            }
        }
        lat.copy_from_slice(&arr);
        for (dst, src) in grad.iter_mut().zip(gd.iter()) {
            *dst = [src[0] as f32, src[1] as f32];
        }
        let octaves = octaves.max(1);
        Turbulence { lattice: lat, gradient: grad, base_freq, octaves, fractal, rect: rect.map(|v| v as f32) }
    }

    /// Noise for one channel at lattice coordinates, returning the signed 2D Perlin value.
    fn noise(&self, chan: usize, px: f32, py: f32) -> f32 {
        let tx = px + PERLIN_N;
        let bx0 = tx as i32;
        let rx0 = tx - bx0 as f32;
        let rx1 = rx0 - 1.0;
        let ty = py + PERLIN_N;
        let by0 = ty as i32;
        let ry0 = ty - by0 as f32;
        let ry1 = ry0 - 1.0;
        let (ix0, ix1) = ((bx0 & BM) as usize, ((bx0 + 1) & BM) as usize);
        let (iy0, iy1) = ((by0 & BM) as usize, ((by0 + 1) & BM) as usize);
        let i = self.lattice[ix0];
        let j = self.lattice[ix1];
        let b00 = self.lattice[(i + iy0 as i32) as usize] as usize;
        let b10 = self.lattice[(j + iy0 as i32) as usize] as usize;
        let b01 = self.lattice[(i + iy1 as i32) as usize] as usize;
        let b11 = self.lattice[(j + iy1 as i32) as usize] as usize;
        let sx = rx0 * rx0 * (3.0 - 2.0 * rx0);
        let sy = ry0 * ry0 * (3.0 - 2.0 * ry0);
        let base = chan * STRIDE;
        let g = self.gradient[base + b00];
        let u = rx0 * g[0] + ry0 * g[1];
        let g = self.gradient[base + b10];
        let v = rx1 * g[0] + ry0 * g[1];
        let a = u + sx * (v - u);
        let g = self.gradient[base + b01];
        let u = rx0 * g[0] + ry1 * g[1];
        let g = self.gradient[base + b11];
        let v = rx1 * g[0] + ry1 * g[1];
        let b = u + sx * (v - u);
        a + sy * (b - a)
    }

    /// `eval` kernel.
    pub fn eval(&self, x: f32, y: f32) -> [f32; 4] {
        if x < self.rect[0] || x >= self.rect[2] || y < self.rect[1] || y >= self.rect[3] {
            return [0.0; 4];
        }
        let mut acc = [0.0f32; 4];
        let mut px = (x - 0.5) * self.base_freq[0];
        let mut py = (y - 0.5) * self.base_freq[1];
        let mut div = 1.0f32;
        for _ in 0..self.octaves {
            for c in 0..4 {
                let n = self.noise(c, px, py);
                acc[c] += (if self.fractal { n } else { n.abs() }) / div;
            }
            px *= 2.0;
            py *= 2.0;
            div *= 2.0;
        }
        let ch = |v: f32| (if self.fractal { v * 0.5 + 0.5 } else { v }).clamp(0.0, 1.0);
        let (r, g, b, a) = (ch(acc[0]), ch(acc[1]), ch(acc[2]), ch(acc[3]));
        [r * a, g * a, b * a, a]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flood_is_verbatim() {
        assert_eq!(flood([0.3, 0.6, 0.9, 0.5])(10.0, 10.0), [0.3, 0.6, 0.9, 0.5]);
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // std cos as the oracle for crt::cosf (test only)
    fn spot_cone_is_cosine_space() {
        // SetConeAngle: outer = cos(angle°), inner = cos((angle+1)°) for angle ≥ 5°.
        let (outer, inner) = spot_cone_cosines(30.0);
        assert_eq!(outer.to_bits(), (((30.0f64).to_radians() as f32).cos()).to_bits());
        assert_eq!(inner.to_bits(), (((31.0f64).to_radians() as f32).cos()).to_bits());
        let axis = [0.0, 0.0, -1.0];
        // L straight down the axis (lDotS = 1) is fully inside the cone; focus 1 ⇒ att = 1.
        assert_eq!(spot_attenuation([0.0, 0.0, 1.0], axis, 1.0, outer, inner), 1.0);
        // L perpendicular (lDotS = 0) is outside ⇒ att = 0.
        assert_eq!(spot_attenuation([1.0, 0.0, 0.0], axis, 1.0, outer, inner), 0.0);
    }

    #[test]
    fn filtered_projective_handles_singular_and_horizon_vertices() {
        let bm = Bitmap::from_straight_rgba(2, 2, &[255; 16]);
        let input = Input::new(&bm, 0, 0);
        let mut m = [0.0; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[10] = 1.0;
        for mode in [Interp::Cubic, Interp::MultiSampleLinear, Interp::Anisotropic] {
            assert_eq!(projective_at(&input, m, mode, Border::Soft, [0.0; 2])(0.5, 0.5), [0.0; 4]);
        }
        // Invertible, but the cubic apron at x=-1.5 meets the projective horizon.
        m[15] = 1.0;
        m[3] = 2.0 / 3.0;
        let c = projective_at(&input, m, Interp::Cubic, Border::Soft, [0.0; 2])(0.5, 0.5);
        assert!(c.iter().all(|v| v.is_finite() && *v > 0.0));
    }

    #[test]
    fn linear_apron_crossing_horizon_uses_input_footprint() {
        let bm = Bitmap::from_straight_rgba(1, 1, &[255; 4]);
        let input = Input::new(&bm, 0, 0);
        // W is positive on the input, but negative on LINEAR's left half-texel apron.
        // Native drops this apron and rasterizes the unexpanded input footprint.
        let m = [1.0, 6.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.5, 0.25];
        let quad = SamplerQuad::new(&input, m, false, Border::Soft, [0.0; 2], Interp::Linear, Some([0.0, 0.0, 16.0, 16.0]), None).unwrap();
        assert_eq!(quad.sample(&input, 0.5, 5.5, Border::Soft), [0.0; 4]);
        assert_eq!(quad.sample(&input, 0.5, 6.5, Border::Soft).map(|v| (v * 255.0).round_ties_even() as u8), [141; 4]);
        assert_eq!(quad.sample(&input, 0.5, 8.5, Border::Soft), [0.0; 4]);
    }

    #[test]
    fn mip_atlas_extreme_origins_do_not_overflow() {
        let bm = Bitmap::from_straight_rgba(16, 16, &[255; 16 * 16 * 4]);
        for origin in [[0, i32::MAX - 8], [i32::MIN + 1, 0], [0, i32::MIN + 1]] {
            let input = Input::new(&bm, origin[0], origin[1]);
            assert!(SamplerQuad::new(&input, [0.625, 0.0, 0.0, 0.0, 0.625, 0.0, 0.0, 0.0, 1.0], true, Border::Soft, [0.0; 2], Interp::Anisotropic, None, None).is_none());
        }
    }

    #[test]
    fn filtered_empty_input_is_transparent() {
        let bm = Bitmap::from_straight_rgba(0, 5, &[]);
        let input = Input::new(&bm, 0, 0);
        for mode in [Interp::Cubic, Interp::MultiSampleLinear, Interp::Anisotropic] {
            assert_eq!(affine_at(&input, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], mode, Border::Soft, [0.0; 2], true)(0.5, 0.5), [0.0; 4]);
        }
    }

    #[test]
    fn affine_identity_roundtrips_a_texel() {
        // 2x2 input, sample dead-centre of texel (0,0) under identity -> that texel back.
        let bm = Bitmap::from_straight_rgba(2, 2, &[10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255]);
        let input = Input::new(&bm, 0, 0);
        let k = affine_at(&input, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], Interp::Nearest, Border::Soft, [0.0; 2], true);
        let c = k(0.5, 0.5);
        assert_eq!(c.map(|v| (v * 255.0).round() as u8), [10, 20, 30, 255]);
    }
}
