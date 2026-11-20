//! Reusable texture sampler reproducing WARP's UNORM8 texture filters
//! byte-for-byte. The brushes phase (1D) uses this for bitmap and image brushes; the bitmap
//! phase (1E) uses it for `DrawBitmap`/`DrawImage`.
//!
//! This is the filter WARP runs for a linear `DrawBitmap`, confirmed against the oracle. The
//! 4-channel UNORM8 linear filter is one 16-bit SIMD lane per channel
//! (zero-extended bytes, no cross-channel packing — the earlier "packed-pair carry" guess was wrong) with a
//! single lerp step `a + (((b − a)·w + 128) >> 8)` ([`lerp8`]), applied **vertically first, then
//! horizontally**. The 8-bit sub-texel weight is bits `[15:8]` of a 16.16 texcoord biased by
//! `−0x7f80` ([`split`]).
//!
//! Rasterized coordinates are supplied by the draw path: triangle attribute planes and quad
//! derivatives determine the Q8 sample addresses. Filtering and coordinate generation must both
//! match WARP, including intermediate texture placement and format conversion.
//!
//! Textures here are premultiplied BGRA8, tightly packed, top-down. Coordinates are in texel
//! space (`u·width`, `v·height`; texel centres at integer+0.5), so callers apply their own
//! device→texture transform first. Samples come back as premultiplied float BGRA in `[0,1]`
//! (an exact UNORM8 value over 255); callers that need the fixed-point byte round `v·255`.

pub(crate) mod srgb;

/// `D2D1_EXTEND_MODE`: how out-of-range texel coordinates wrap.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Extend {
    Clamp,
    Wrap,
    Mirror,
    /// Out-of-bounds texels contribute 0 (transparent). Used by the SOFT effect-output magnify, whose
    /// LINEAR sampler reads transparent past the realized bitmap so the edge ramps to 0 (native's
    /// DrawImage of a blur/effect output), where a plain bitmap uses Clamp.
    Transparent,
}

/// `D2D1_INTERPOLATION_MODE` subset this sampler reproduces. Nearest and linear are byte-exact
/// from WARP. Cubic uses WARP's Catmull-Rom weights and 8-bit fractional texcoord (both confirmed
/// against the emitted kernel and the native impulse captures), but its 4×4 tap combine is a float
/// accumulation that is ≤1 LSB off WARP's exact fixed-point refinement — see [`Texture::sample_cubic`]
/// for the pinned residual and the blocker. The remaining modes (multisample-linear's minification
/// multisampling, anisotropic, high-quality cubic) are separate WARP kernels not reproduced here.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    Nearest,
    Linear,
    Cubic,
}

impl Extend {
    /// Fold a texel index onto `0..n` (`n ≥ 1`).
    pub(crate) fn wrap(self, c: i32, n: i32) -> i32 {
        if n <= 0 {
            return 0; // empty axis: no texel to fold onto (clamp/rem_euclid would panic)
        }
        match self {
            Extend::Clamp => c.clamp(0, n - 1),
            Extend::Wrap => c.rem_euclid(n),
            Extend::Mirror => {
                let period = 2 * n;
                let m = c.rem_euclid(period);
                if m < n { m } else { period - 1 - m }
            }
            Extend::Transparent => c.clamp(0, n - 1), // in-bounds identity; OOB handled in `texel`
        }
    }
}

/// WARP float-texture bilinear: Q8 weights, Y lerps followed by the X lerp.
pub(crate) fn bilinear_f32(x: (i32, i32), y: (i32, i32), fetch: impl Fn(i32, i32) -> [f32; 4]) -> [f32; 4] {
    let (a, b, c, d) = (fetch(x.0, y.0), fetch(x.0 + 1, y.0), fetch(x.0, y.0 + 1), fetch(x.0 + 1, y.0 + 1));
    let (fx, fy) = (x.1 as f32 / 256.0, y.1 as f32 / 256.0);
    core::array::from_fn(|ch| {
        let left = a[ch] + (c[ch] - a[ch]) * fy;
        let right = b[ch] + (d[ch] - b[ch]) * fy;
        left + (right - left) * fx
    })
}

/// WARP float anisotropic taps, with independent symmetric positions for the two-tap case.
// WARP 10.0.26100.1301's anisotropic sampler: two-tap offsets.
pub(crate) fn anisotropic_f32(uv: [f32; 2], dims: [f32; 2], direction: [f32; 2], ratio: f32, fetch: impl Fn(i32, i32) -> [f32; 4]) -> [f32; 4] {
    let (mut count, start, step) = if ratio <= 1.0 { (1, 0.0, 0.0) }
        else if ratio <= 2.0 { (2, -(ratio - 1.0) * 0.5, ratio - 1.0) }
        else { let step = ratio * (1.0 / 17.0); (16, ratio * -0.5 + step, step) };
    let axis = [direction[0] * (1.0 / dims[0]), direction[1] * (1.0 / dims[1])];
    let pair = core::array::from_fn::<_, 2, _>(|c| (((ratio - 1.0) * 0.5) * direction[c]) * (1.0 / dims[c]));
    let mut pos = if count == 2 { [uv[0] + pair[0], uv[1] + pair[1]] } else { [uv[0] + axis[0] * start, uv[1] + axis[1] * start] };
    let delta = [axis[0] * step, axis[1] * step];
    let mut taps = [[0.0; 4]; 16];
    for tap in &mut taps[..count] {
        let q: [(i32, i32); 2] = core::array::from_fn(|c| {
            let q = ((pos[c] * dims[c] - 0.5) * 256.0).floor() as i32;
            (q >> 8, q & 255)
        });
        *tap = bilinear_f32(q[0], q[1], &fetch);
        if count == 2 { pos = [uv[0] - pair[0], uv[1] - pair[1]]; }
        else { pos[0] += delta[0]; pos[1] += delta[1]; }
    }
    while count > 1 {
        count /= 2;
        for i in 0..count { taps[i] = core::array::from_fn(|ch| (taps[2*i][ch] + taps[2*i+1][ch]) * 0.5); }
    }
    taps[0]
}

/// A Q8 texel address from its scaled float, floored: an address past i32 (or infinite) converts to i32::MIN as the
/// x86 float-to-int conversion does, which every extend folds onto the origin texel with no blend (VM probes of a
/// 1e-6-scale WRAP brush on CUBIC and LINEAR geometry fills, r22 27012: native reads texel 0 from the pixel where
/// `(u·n − ½)·256` passes 2^31). A NaN keeps address 0.
pub(crate) fn q8_floor(v: f32) -> i32 {
    let f = v.floor();
    if f.abs() < 2147483648.0 { f as i32 } else if f.is_nan() { 0 } else { i32::MIN }
}

/// WARP's Q8 texel address (texel, weight) of the normalized coordinate `u` on an axis of `n` texels
/// off its power-of-two path (an axis that is not a power of two, or v unless both are): WRAP folds
/// `u − ⌊u⌋` and MIRROR `frac(u·½)` over `2n` in normalized space, CLAMP holds `u` to ±10, then
/// `·256n` rounds to nearest, less half a texel (the LINEAR and CUBIC shader JITs).
pub(crate) fn q8_address_npow2(u: f32, n: f32, e: Extend) -> (i32, i32) {
    let q = npow2_fold(u, n, e).round_ties_even() as i32 - 128;
    (q >> 8, q & 255)
}

/// [`q8_address_npow2`] as the geometry and mesh bitmap kernels take it: they zero a NaN lane after
/// taking the half texel off, so a NaN address (a NaN texcoord, or an infinite one under WRAP/MIRROR)
/// reads texel 0 unblended (VM probes: 21261's brush on a mesh, an ellipse and a path; the JIT of its
/// mesh kernel). An axis-aligned rect fill blends there instead.
pub(crate) fn q8_address_npow2_geom(u: f32, n: f32, e: Extend) -> (i32, i32) {
    let v = npow2_fold(u, n, e);
    if v.is_nan() {
        return (0, 0);
    }
    let q = v.round_ties_even() as i32 - 128;
    (q >> 8, q & 255)
}

fn npow2_fold(u: f32, n: f32, e: Extend) -> f32 {
    match e {
        Extend::Wrap => (u - u.floor()) * (256.0 * n),
        Extend::Mirror => { let m = u * 0.5; (m - m.floor()) * (512.0 * n) }
        _ => u.clamp(-10.0, 10.0) * (256.0 * n),
    }
}

/// WARP's nine float bilinear windows, combined in shader X-then-Y order.
pub(crate) fn cubic_windows_f32(
    xs: [(i32, i32); 3], ys: [(i32, i32); 3], wx: [f32; 3], wy: [f32; 3],
    fetch: impl Fn(i32, i32) -> [f32; 4],
) -> [f32; 4] {
    let rows = ys.map(|y| {
        let c = xs.map(|x| bilinear_f32(x, y, &fetch));
        core::array::from_fn::<_, 4, _>(|ch| (c[0][ch] * wx[0] + c[1][ch] * wx[1]) + c[2][ch] * wx[2])
    });
    core::array::from_fn(|ch| rows[1][ch] * wy[1] + (rows[2][ch] * wy[2] + rows[0][ch] * wy[0]))
}

// Native Direct2D 10.0.26100 hq-limit-{256,512,1024,2048} DDI prefilter constants.
pub(crate) const HQ_MIN_SCALE: f32 = f32::from_bits(0x3c23_d70b);

// Native Direct2D 10.0.26100 HQ downscale PS: four taps on each side per loop, then divide by total weight. `byte`: the
// sampled texture is 8-bit (WARP's bilinear lerps it in value·128 fixed point).
pub(crate) fn hq_downscale_axis(
    phase: f32, position: f32, uv: f32, step: f32, scale: f32, sharpness: f32, byte: bool,
    sample: impl Fn(f32) -> [f32; 4],
) -> [f32; 4] {
    let f = (phase - 0.5) - (phase - 0.5).floor();
    let base = ((phase - f) - position) * step + uv;
    let a = -0.5 - 1.5 * sharpness;
    let weight = |t: f32| {
        let inner = t.clamp(0.0, 1.0);
        let outer = t.clamp(1.0, 2.0);
        let inner = (inner * inner) * (inner * (a + 2.0) - (a + 3.0)) + 1.0;
        let outer = outer * (outer * (outer * a - 5.0 * a) + 8.0 * a) - 4.0 * a;
        outer + inner
    };
    if scale == 0.5 || scale == 0.25 || scale == 0.125 {
        return hq_downscale_pairs(phase, position, uv, step, scale, f, byte, weight, sample);
    }
    let mut acc = [0.0; 4];
    let mut total = 0.0;
    for i in (0..(2.0 / scale) as usize).step_by(4) {
        for side in 0..2 {
            let ws = core::array::from_fn::<_, 4, _>(|j| weight(((if side == 0 { f } else { 1.0 - f }) + (i + j) as f32) * scale));
            for j in 0..4 {
                let u = if side == 0 { -((i + j) as f32) * step + base } else { (i + j) as f32 * step + (base + step) };
                let c = sample(u);
                for ch in 0..4 { acc[ch] = ws[j] * c[ch] + acc[ch]; }
            }
            total = ((ws[0] + ws[1]) + ws[2]) + ws[3] + total;
        }
    }
    acc.map(|c| c / total)
}

// Native's HQ downscale by exactly 1/2, 1/4 or 1/8 (VM-captured pixel shaders): every pixel takes its taps at the same
// phase, so d2d1 folds each two neighbouring taps into one bilinear sample at their weighted position, with the pair's
// share of the total weight as a constant. The texel coordinate is `((1/s - 1)·position + offset)·step + uv`, each
// side's samples sit at ±(pair offset)·step from it, the bilinear weight is WARP's 8-bit fraction, and the samples
// accumulate in each shader's own order. Smaller scales run the general loop (probe-hqrot: 2304 px Δ1 -> 0).
fn hq_downscale_pairs(
    phase: f32, position: f32, uv: f32, step: f32, scale: f32, f: f32, byte: bool,
    weight: impl Fn(f32) -> f32, sample: impl Fn(f32) -> [f32; 4],
) -> [f32; 4] {
    let taps = (2.0 / scale) as usize;
    // Distances from the centre: the + side's taps at (1 - f) + j, the - side's at f + j.
    let side = |d0: f32| -> Vec<(f32, f32)> { (0..taps).map(|j| { let d = d0 + j as f32; (d, weight(d * scale)) }).collect() };
    let (plus, minus) = (side(1.0 - f), side(f));
    let total = minus.iter().chain(plus.iter()).fold(0.0f32, |t, &(_, w)| t + w);
    let pairs = |taps: &[(f32, f32)]| -> Vec<(f32, f32)> {
        taps.chunks_exact(2).map(|p| {
            let w = p[0].1 + p[1].1;
            (w / total, (p[0].0 * p[0].1 + p[1].0 * p[1].1) / w)
        }).collect()
    };
    let (plus, minus) = (pairs(&plus), pairs(&minus));
    let k = 1.0 / scale - 1.0;
    let c = (k * position + (phase - position / scale)) * step + uv;
    let dim = (1.0 / step).round();
    let pow2 = (dim as u32).is_power_of_two();
    let bilinear = |u: f32| -> [f32; 4] {
        let t = (u * dim - 0.5) * 256.0;
        let q = if pow2 { t.floor() } else { t.round_ties_even() } as i32;
        let (k, w) = ((q >> 8) as f32, (q & 255) as f32 / 256.0);
        let (lo, hi) = (sample((k + 0.5) * step), sample((k + 1.5) * step));
        if byte {
            // An 8-bit texture lerps in WARP's value·128 fixed point.
            let wi = q & 255;
            return core::array::from_fn(|ch| {
                let (a, b) = ((lo[ch] * 255.0).round() as i32 * 128, (hi[ch] * 255.0).round() as i32 * 128);
                norm_v128((a * (256 - wi) + b * wi) >> 8)
            });
        }
        core::array::from_fn(|ch| lo[ch] + (hi[ch] - lo[ch]) * w)
    };
    let at = |(w, o): (f32, f32), sign: f32| -> [f32; 4] { bilinear((sign * o) * step + c).map(|v| v * w) };
    let add = |acc: [f32; 4], (w, o): (f32, f32), sign: f32| -> [f32; 4] {
        let s = bilinear((sign * o) * step + c);
        core::array::from_fn(|ch| w * s[ch] + acc[ch])
    };
    let mut acc;
    if scale == 0.5 {
        // -0, +0, +1, -1
        acc = at(minus[0], -1.0);
        acc = add(acc, plus[0], 1.0);
        acc = add(acc, plus[1], 1.0);
        acc = add(acc, minus[1], -1.0);
    } else {
        // +1, +0, +2.., then -0, -1..
        acc = at(plus[1], 1.0);
        acc = add(acc, plus[0], 1.0);
        for &p in &plus[2..] { acc = add(acc, p, 1.0); }
        for &p in &minus { acc = add(acc, p, -1.0); }
    }
    acc
}

/// A premultiplied BGRA8 texture view: `data` is `width*height*4` bytes, tightly packed.
#[derive(Clone, Copy)]
pub struct Texture<'a> {
    pub data: &'a [u8],
    pub width: u32,
    pub height: u32,
}

impl Texture<'_> {
    fn texel(&self, x: i32, y: i32, ex: [Extend; 2]) -> [i32; 4] {
        // A zero-size texture (a degenerate image/effect realized to 0×0) has no texels — sampling it
        // is transparent, and `wrap(_, 0)` would panic (clamp(0,-1) / rem_euclid(0)). (fuzz 297/1525/…)
        if self.width == 0 || self.height == 0 {
            return [0; 4];
        }
        // Transparent: a texel outside the bitmap contributes 0 (the ramp-to-transparent border).
        if (ex[0] == Extend::Transparent && (x < 0 || x >= self.width as i32))
            || (ex[1] == Extend::Transparent && (y < 0 || y >= self.height as i32))
        {
            return [0; 4];
        }
        let x = ex[0].wrap(x, self.width as i32) as usize;
        let y = ex[1].wrap(y, self.height as i32) as usize;
        let i = (y * self.width as usize + x) * 4;
        // The data may be shorter than width*height*4 (a sub-4-byte-per-pixel format, e.g. an A8 mask
        // whose bytes are 1/px) — sampling past it is transparent, not a panic.
        if i + 3 >= self.data.len() {
            return [0; 4];
        }
        [self.data[i] as i32, self.data[i + 1] as i32, self.data[i + 2] as i32, self.data[i + 3] as i32]
    }

    /// Sample at texel-space coordinates `(tx, ty)` and return premultiplied float BGRA in
    /// `[0,1]`. `tx = u·width`, `ty = v·height` (texel centres at integer+0.5). Convenience over
    /// [`Texture::sample_fixed`] for callers that have the coordinate in float (the 1-D gradient
    /// LUT); forms WARP's 16.16 texcoord per axis with [`split`].
    pub fn sample(&self, filter: Filter, ex: [Extend; 2], tx: f32, ty: f32) -> [f32; 4] {
        match filter {
            Filter::Nearest => self.texel(tx.floor() as i32, ty.floor() as i32, ex).map(|c| c as f32 / 255.0),
            Filter::Linear => {
                let (x, ux) = split(tx);
                let (y, uy) = split(ty);
                self.filtered(ex, x, ux, y, uy)
            }
            Filter::Cubic => self.sample_cubic(ex, tx, ty),
        }
    }

    /// LINEAR bilinear via WARP's `value·128` combine (single higher-precision round), normalised to
    /// `[0,1]`. Unlike [`sample`]'s `Filter::Linear` (nested UNORM8 `lerp8`, which some callers —
    /// BitmapSource effects, the glyph atlas — depend on), this is the bit-exact HW-sampler bilinear
    /// (probed on WARP). The bitmap-brush sampler uses it (the nested-8-bit double-round was ≤2 LSB
    /// off on magnified brush fills: 13-bitmap-brush etc.).
    pub fn linear_v128(&self, ex: [Extend; 2], tx: f32, ty: f32) -> [f32; 4] {
        let (x, ux) = split(tx);
        let (y, uy) = split(ty);
        core::array::from_fn(|ch| norm_v128(self.bilinear_v128(ex, ch, (x, ux), (y, uy))))
    }

    /// `D2D1_INTERPOLATION_MODE_CUBIC`: separable Catmull-Rom (a = −0.5), reproducing WARP's cubic
    /// kernel's premultiplied output byte-for-byte (confirmed on a local readback). WARP does NOT do a plain 4×4 convolution — per axis the four taps
    /// `i−1,i,i+1,i+2` are a **Sigg-interpolated centre pair** `{i,i+1}` (combine weight `gC = w1+w2`,
    /// 8-bit within-pair offset `floor(256·w2/gC)`) plus two **direct outer taps** `i−1` (weight `w0`)
    /// and `i+2` (weight `w3`); the nine `x∈{outerL,centre,outerR} × y∈{…}` samples are each a
    /// `value·128` fixed-point 2×2/column lerp ([`lerp_v`]/[`lerp_h`]), normalised `×1/128 ×1/255`, and
    /// summed in f32 with the Catmull-Rom products. The float combine matches WARP's traced premul
    /// intermediates to f32 noise (traced at its final float→int conversion inputs). Returns premultiplied
    /// `[0,1]` (callers apply opacity/coverage and the `×255` round / source-over).
    pub fn sample_cubic(&self, ex: [Extend; 2], tx: f32, ty: f32) -> [f32; 4] {
        self.sample_cubic_impl(ex, tx, ty, false)
    }

    /// HQ cubic at a texel-space position. DrawBitmap supplies its separately interpolated
    /// normalized coordinate to [`Self::sample_hqcubic_at`] to preserve address rounding.
    pub fn sample_hqcubic(&self, ex: [Extend; 2], tx: f32, ty: f32) -> [f32; 4] {
        self.sample_hqcubic_at(ex, [tx / self.width as f32, ty / self.height as f32], [tx, ty])
    }

    /// WARP HQ cubic uses the inverse transform for the polynomial phase and an independent
    /// normalized varying for all nine bilinear addresses, including the two outer taps per axis.
    /// Replacing an outer bilinear by a direct texel loses its Q8 boundary rounding.
    pub(crate) fn sample_hqcubic_at(&self, ex: [Extend; 2], uv: [f32; 2], phase: [f32; 2]) -> [f32; 4] {
        self.sample_hqcubic_in(ex, uv, phase, [self.width, self.height])
    }

    /// [`Self::sample_hqcubic_at`] addressing a texture of `dims` that holds this one at its origin, transparent
    /// past it (an effect output in its atlas).
    pub(crate) fn sample_hqcubic_in(&self, ex: [Extend; 2], uv: [f32; 2], phase: [f32; 2], dims: [u32; 2]) -> [f32; 4] {
        self.sample_hqcubic_unclamped_in(ex, uv, phase, dims).map(|v| v.clamp(0.0, 1.0))
    }

    pub(crate) fn sample_hqcubic_at_unclamped(&self, ex: [Extend; 2], uv: [f32; 2], phase: [f32; 2]) -> [f32; 4] {
        self.sample_hqcubic_unclamped_in(ex, uv, phase, [self.width, self.height])
    }

    pub(crate) fn sample_hqcubic_unclamped_in(&self, ex: [Extend; 2], uv: [f32; 2], phase: [f32; 2], dims: [u32; 2]) -> [f32; 4] {
        self.sample_hqcubic_unclamped_at(ex, uv, phase, dims, [0; 2])
    }

    /// [`Self::sample_hqcubic_unclamped_in`] for a texture of `dims` that holds this one at texel `at` (an image brush's
    /// border cells in the frame atlas), unclamped.
    pub(crate) fn sample_hqcubic_unclamped_at(&self, ex: [Extend; 2], uv: [f32; 2], phase: [f32; 2], dims: [u32; 2], at: [i32; 2]) -> [f32; 4] {
        // WARP's power-of-two address path: u when the width is a power of two, v only when both are.
        let pow2 = [dims[0].is_power_of_two(), dims[0].is_power_of_two() && dims[1].is_power_of_two()];
        let axis = |u: f32, t: f32, dim: f32, e: Extend, pow2: bool| {
            let (_, f) = cubic_axis(t);
            let w = cubic_weights(f, 0.0);
            let g = w[1] + w[2];
            let step = 1.0 / dim;
            let base = u - f * step;
            // WARP's Q8 address, as the LINEAR unit-triangle draw's (cub-i5-7x5-* probes). On the
            // power-of-two path it floors `(c·dim − ½)·256` of the raw `c` (a CLAMP axis clamped to ±10 first),
            // the integer texel fold in `texel` applying a WRAP/MIRROR period; an address past i32 reads the
            // origin texel ([`q8_floor`]). Any other axis folds WRAP (`frac c`) or MIRROR (`frac(c·½)` over 2·dim) in
            // normalized space and rounds `·256·dim` to nearest, less half a texel: a rotated 15×9
            // WRAP/MIRROR brush sampled `c·dim` unfolded was a step off at scattered pixels (r27 32539).
            let q = [base - step, base + (w[2] / g) * step, base + 2.0 * step].map(|c| {
                let q = if pow2 {
                    let c = if matches!(e, Extend::Wrap | Extend::Mirror) { c } else { c.clamp(-10.0, 10.0) };
                    q8_floor((c * dim - 0.5) * 256.0)
                } else {
                    return q8_address_npow2(c, dim, e);
                };
                (q >> 8, q & 0xff)
            });
            (q, [w[0], g, w[3]])
        };
        let (xs, wx) = axis(uv[0], phase[0], dims[0] as f32, ex[0], pow2[0]);
        let (ys, wy) = axis(uv[1], phase[1], dims[1] as f32, ex[1], pow2[1]);
        hqcubic_windows_unclamped(xs, ys, wx, wy, |x, y| self.texel(x.wrapping_sub(at[0]), y.wrapping_sub(at[1]), ex))
    }

    fn sample_cubic_impl(&self, ex: [Extend; 2], tx: f32, ty: f32, hq: bool) -> [f32; 4] {
        cubic_combine(hq, tx, ty, |x, y| self.texel(x, y, ex))
    }

    /// Linear-sample from WARP's rasterized 16.16 texcoords directly. `ux`/`uy` are the biased 16.16
    /// the attribute interpolator produces (`base + px·step` with the `−0x7f80` sampling bias folded
    /// in, see [`axis_fixed`]): the integer texel is `u >> 16` and the 8-bit lerp weight is bits
    /// `[15:8]`. This is the exact per-pixel input the emitted bilinear sampler reads, so no float
    /// texcoord is rounded per pixel. (Nearest keeps its own coordinate convention via [`sample`].)
    pub fn sample_fixed(&self, ex: [Extend; 2], ux: i32, uy: i32) -> [f32; 4] {
        self.filtered(ex, ux >> 16, (ux >> 8) & 0xff, uy >> 16, (uy >> 8) & 0xff)
    }

    /// [`Self::sample_fixed`] at Q8 addresses (texel `q >> 8`, weight `q & 0xff`).
    pub(crate) fn sample_q8(&self, ex: [Extend; 2], qx: i32, qy: i32) -> [f32; 4] {
        self.filtered(ex, qx >> 8, qx & 0xff, qy >> 8, qy & 0xff)
    }

    /// NEAREST from WARP's rasterized 16.16 texcoord DDA (`base + px·step`, same [`axis_fixed`] as
    /// [`sample_fixed`]). The texel is `floor` of the UNBIASED coordinate: [`axis_fixed`] folds a
    /// `−0x7f80` half-texel bias in for linear's sub-texel weight, so nearest adds it back before the
    /// `>> 16` floor. This floors WARP's own 16.16 value, which differs from a per-pixel float
    /// `floor(u·dim)` at non-power-of-two device scales: a DrawBitmap at dpi 120 (×1.25) whose float
    /// coordinate lands exactly on a texel boundary (u = 3.0 → float floor 3) is 2.99995 in the DDA and
    /// floors to 2, matching native. Integer-aligned / pow2-scale draws are unchanged (u is exact).
    pub fn sample_fixed_nearest(&self, ex: [Extend; 2], ux: i32, uy: i32) -> [f32; 4] {
        let xi = ux.wrapping_add(0x7f80) >> 16;
        let yi = uy.wrapping_add(0x7f80) >> 16;
        self.texel(xi, yi, ex).map(|c| c as f32 / 255.0)
    }

    /// The alpha-blit bilinear at a Q8 address (texel `q >> 8`, weight `q & 0xff`, the −½-texel bias
    /// already in): each row lerps first, then the column; nearest takes the texel its weight rounds to.
    pub fn sample_alphablt_q8(&self, ex: [Extend; 2], qx: i32, qy: i32, nearest: bool) -> [i32; 4] {
        let ((x0, fx), (y0, fy)) = ((qx >> 8, qx & 0xff), (qy >> 8, qy & 0xff));
        if nearest {
            return self.texel(x0.wrapping_add((fx >= 128) as i32), y0.wrapping_add((fy >= 128) as i32), ex);
        }
        let (x1, y1) = (x0.wrapping_add(1), y0.wrapping_add(1));
        let (c00, c10, c01, c11) = (self.texel(x0, y0, ex), self.texel(x1, y0, ex), self.texel(x0, y1, ex), self.texel(x1, y1, ex));
        core::array::from_fn(|ch| lerp8(lerp8(c00[ch], c10[ch], fx), lerp8(c01[ch], c11[ch], fx), fy))
    }

    /// The 2×2 bilinear of WARP's emitted sampler: vertical lerp of each column, then horizontal.
    /// `x0`/`y0` are the lower texel indices, `fx8`/`fy8` the 8-bit sub-texel weights.
    fn filtered(&self, ex: [Extend; 2], x0: i32, fx8: i32, y0: i32, fy8: i32) -> [f32; 4] {
        let c00 = self.texel(x0, y0, ex);
        let c10 = self.texel(x0 + 1, y0, ex);
        let c01 = self.texel(x0, y0 + 1, ex);
        let c11 = self.texel(x0 + 1, y0 + 1, ex);
        core::array::from_fn(|ch| {
            let col0 = lerp8(c00[ch], c01[ch], fy8);
            let col1 = lerp8(c10[ch], c11[ch], fy8);
            lerp8(col0, col1, fx8) as f32 / 255.0
        })
    }
}

/// The height of the row bands WARP's rasterizer hands a blit on a render target `height` rows tall: that height over 32
/// to the nearest even row, ties up, and at least 8 (VM: a rotated LINEAR Border's blit onto targets 24 to 2048 rows tall,
/// each exact at one band only: 8 up to 257 rows, 10 at 288 and 320, 16 from 480 to 513, 24 at 768, 64 at 2048).
pub(crate) fn alphablt_band(height: u32) -> usize {
    (2 * ((height as usize + 32) / 64)).max(8)
}

/// [`alphablt_float_iter_ex`] for a LINEAR draw whose extend never folds (CLAMP or a transparent border).
pub(crate) fn alphablt_float_iter(ts: &[f32; 6], size: [u32; 2], rect: [usize; 4], target_h: u32, put: impl FnMut(usize, usize, i32, i32)) {
    alphablt_float_iter_ex(ts, size, size, rect, rect, [Extend::Clamp; 2], false, target_h, put)
}

/// [`alphablt_float_iter`] for the blit of a float texture, which floors the Q8 address instead of rounding it
/// (VM probes of a float bitmap's exact scale drawn LINEAR at (0.3, 0.7) and (10.6, 3.2) onto a float target).
pub(crate) fn alphablt_float_iter_floor(ts: &[f32; 6], size: [u32; 2], rect: [usize; 4], target_h: u32, put: impl FnMut(usize, usize, i32, i32)) {
    alphablt_iter(ts, size, size, rect, rect, [Extend::Clamp; 2], false, true, target_h, put)
}

/// WARP's float alpha-blit iterator over the device rect `[x0, y0, x1, y1]`, calling `put(x, y, qx, qy)`
/// with each pixel's Q8 texel address (−½-texel bias in). `ts` is the device→texel matrix and `size` the
/// texture's (traced on imgbrush-01's fill). WARP's blit attribute setup evaluates `ts` at the rect's first
/// pixel centre and normalizes the plane by the texture size; the kernel scales it back by size·256,
/// starts each row at `row·size·256 + (k·step − 128)` for SIMD lane k, adds `4·step` per four pixels
/// (`step` per leftover pixel), advances the row by `dY` and rounds the address to nearest-even. The
/// rasterizer hands it row bands ([`alphablt_band`] of the render target's height: 8 rows on one up to 257 tall;
/// absolute rows, full width), each restarting its row from `dY·(band top − rect top)` (traced band rects). An axis whose texture dimension is not a power of two
/// (each axis by its own, unlike the geometry kernels) under WRAP or MIRROR is interpolated NORMALIZED instead and
/// folded before it is scaled: WRAP `frac(v)·n`, MIRROR `frac(v·½)·2n`; LINEAR rounds that ×256 less half a
/// texel ([`q8_address_npow2`]), NEAREST truncates it to the texel (the VM JIT of a 16×10 brush's rect fill:
/// a texcoord ~1e8 texels out keeps its period there, where the scaled address overflows).
/// `period` is the sampled bitmap's own size, which an image brush's power-of-two atlas `size` is not.
#[allow(clippy::too_many_arguments)]
pub(crate) fn alphablt_float_iter_ex(ts: &[f32; 6], size: [u32; 2], period: [u32; 2], rect: [usize; 4], inner: [usize; 4], ex: [Extend; 2], nearest: bool, target_h: u32, put: impl FnMut(usize, usize, i32, i32)) {
    alphablt_iter(ts, size, period, rect, inner, ex, nearest, false, target_h, put)
}

/// [`alphablt_float_iter_ex`], its scaled Q8 address floored (`floor`) or rounded to nearest-even.
#[allow(clippy::too_many_arguments)]
fn alphablt_iter(ts: &[f32; 6], size: [u32; 2], period: [u32; 2], [x0, y0, x1, y1]: [usize; 4], inner: [usize; 4], ex: [Extend; 2], nearest: bool, floor: bool, target_h: u32, mut put: impl FnMut(usize, usize, i32, i32)) {
    let band = alphablt_band(target_h);
    let pow2 = size.map(|v| v.is_power_of_two());
    let fold: [bool; 2] = core::array::from_fn(|a| !pow2[a] && matches!(ex[a], Extend::Wrap | Extend::Mirror));
    let size = size.map(|v| v as f32);
    let inv = size.map(|v| 1.0 / v);
    let (fx, fy) = (x0 as f32 + 0.5, y0 as f32 + 0.5);
    let o = [(ts[0] * fx + ts[2] * fy) + ts[4], (ts[1] * fx + ts[3] * fy) + ts[5]];
    let (dx, dy, o) = ([inv[0] * ts[0], inv[1] * ts[1]], [inv[0] * ts[2], inv[1] * ts[3]], [inv[0] * o[0], inv[1] * o[1]]);
    let scale = size.map(|v| v * 256.0);
    let step = [dx[0] * scale[0], dx[1] * scale[1]];
    let tstep = [dx[0] * size[0], dx[1] * size[1]];
    let lanes: [[f32; 4]; 2] = core::array::from_fn(|a| core::array::from_fn(|k| k as f32 * step[a] - 128.0));
    let cvt = |v: f32| if v.is_nan() { 0 } else if v.abs() < 2147483648.0 { (if floor { v.floor() } else { v.round_ties_even() }) as i32 } else { i32::MIN };
    // The address of axis `a` from its value scaled to Q8 (`s`), normalized (`n`) and in texels (`t`).
    let addr = |a: usize, s: f32, n: f32, t: f32| -> i32 {
        if !fold[a] && nearest {
            // NEAREST floors the coordinate in texels, then masks or clamps it; reduce the texel to its
            // period here so a far one survives the Q8 encoding (every rotated NEAREST probe picked a
            // texel-edge pixel one texel off with the rounded Q8 address). The period is the bitmap's, not
            // an image brush's atlas: the sampler wraps over the bitmap (VM probes of rotated NEAREST image
            // brushes on 14×11, 6×23, 10×6 and 13×8 bitmaps, r27 32702 and 32326).
            let i = cvt(t.floor());
            let len = period[a] as i32;
            return match ex[a] { Extend::Wrap => i.rem_euclid(len), Extend::Mirror => i.rem_euclid(2 * len), _ => i.clamp(-1, len) } << 8;
        }
        if !fold[a] {
            return cvt(s);
        }
        if nearest {
            let t = match ex[a] { Extend::Wrap => (n - n.floor()) * size[a], _ => { let m = n * 0.5; (m - m.floor()) * (2.0 * size[a]) } };
            return (t as i32) << 8; // truncates; a NaN lane reads 0
        }
        let (t, w) = q8_address_npow2(n, size[a], ex[a]);
        (t << 8) | w
    };
    // An antialiased rect's partial edge columns and top row are spans of their own, each starting its lanes
    // and its row at its own first pixel from the rect's plane, `(dX·x + dY·y) + c` (the INT3 trace of a
    // rotated 16×16 brush's rect: a pixel of the interior span is one LSB off with lanes from the edge column).
    let ix = [inner[0].clamp(x0, x1), inner[2].clamp(x0, x1).max(inner[0].clamp(x0, x1))];
    let iy = inner[1].clamp(y0, y1);
    for (ra, rb) in [(y0, iy), (iy, y1)] {
        for (ca, cb) in [(x0, ix[0]), (ix[0], ix[1]), (ix[1], x1)] {
            if ca >= cb || ra >= rb {
                continue;
            }
            let xr = (ca - x0) as f32;
            let mut top = ra;
            while top < rb {
                let bottom = ((top / band + 1) * band).min(rb);
                let yr = (top - y0) as f32;
                let mut row = [(dx[0] * xr + dy[0] * yr) + o[0], (dx[1] * xr + dy[1] * yr) + o[1]];
                for y in top..bottom {
                    let mut u: [f32; 4] = core::array::from_fn(|k| row[0] * scale[0] + lanes[0][k]);
                    let mut v: [f32; 4] = core::array::from_fn(|k| row[1] * scale[1] + lanes[1][k]);
                    let mut un: [f32; 4] = core::array::from_fn(|k| row[0] + k as f32 * dx[0]);
                    let mut vn: [f32; 4] = core::array::from_fn(|k| row[1] + k as f32 * dx[1]);
                    let mut ut: [f32; 4] = core::array::from_fn(|k| row[0] * size[0] + k as f32 * tstep[0]);
                    let mut vt: [f32; 4] = core::array::from_fn(|k| row[1] * size[1] + k as f32 * tstep[1]);
                    let mut x = ca;
                    while x + 4 <= cb {
                        for k in 0..4 { put(x + k, y, addr(0, u[k], un[k], ut[k]), addr(1, v[k], vn[k], vt[k])); }
                        (u, v) = (u.map(|a| a + 4.0 * step[0]), v.map(|a| a + 4.0 * step[1]));
                        (un, vn) = (un.map(|a| a + 4.0 * dx[0]), vn.map(|a| a + 4.0 * dx[1]));
                        (ut, vt) = (ut.map(|a| a + 4.0 * tstep[0]), vt.map(|a| a + 4.0 * tstep[1]));
                        x += 4;
                    }
                    for x in x..cb {
                        put(x, y, addr(0, u[0], un[0], ut[0]), addr(1, v[0], vn[0], vt[0]));
                        (u[0], v[0]) = (u[0] + step[0], v[0] + step[1]);
                        (un[0], vn[0]) = (un[0] + dx[0], vn[0] + dx[1]);
                        (ut[0], vt[0]) = (ut[0] + tstep[0], vt[0] + tstep[1]);
                    }
                    row = [row[0] + dy[0], row[1] + dy[1]];
                }
                top = bottom;
            }
        }
    }
}

/// The texel a pixel samples on a 16.16 fixed-point DDA, rounded to nearest: the texel coordinate at the first pixel's
/// centre (`m0·(first + ½) + m4`) and the per-pixel step `m0` each rounded to 1/65536, the step accumulated from `first`
/// to `px` and the sum shifted down 16. An exact texel tie falls low at some pixels and high at others as the rounded
/// step accumulates (NEAREST DrawImage of a bitmap off its own DPI: all 35 native tie picks, sc156/sc174).
pub(crate) fn dda_texel(m0: f32, m4: f32, first: f32, px: f32) -> i32 {
    let start = ((m0 * (first + 0.5) + m4) * 65536.0).round() as i64;
    let step = (m0 * 65536.0).round() as i64;
    ((start + (px.floor() - first) as i64 * step) >> 16) as i32
}

/// One axis of WARP's per-pixel attribute interpolation for an axis-aligned textured draw, as the
/// 16.16 fixed-point DDA the texture sampler reads: `u16(px) = base + px·step`, texel `u16 >> 16`,
/// weight bits `[15:8]`. Thin re-export of the shared [`crate::warp::linear_attr_axis`] (the texcoord
/// counterpart of WARP's Gouraud attribute plane), so every textured draw — `DrawBitmap`, the bitmap
/// brush, `FillOpacityMask` — routes its source-coordinate interpolation through one function.
pub fn axis_fixed(src0: f32, src1: f32, dst0: f32, dst1: f32) -> (i32, i32) {
    crate::warp::linear_attr_axis(src0, src1, dst0, dst1)
}

/// One axis of WARP's **float-family** texcoord plane (the cubic/HQ-cubic/MSL/anisotropic and
/// `FillOpacityMask` kernels, distinct from the integer DDA [`axis_fixed`]), returning `(u_d, u_c)`
/// such that the *normalised* texcoord at device position `pos` is `u_d·pos + u_c`: Direct2D realises the
/// affine source-sub-rect→device-dest map and divides by the texture size `dim`. [`plane_quant`] then
/// turns a position into a texel + sub-texel weight; evaluate this once per draw and quantise many
/// positions per pixel (the multi-sample filters) or just the pixel centre (`FillOpacityMask`).
pub fn float_plane(src0: f32, src1: f32, dst0: f32, dst1: f32, dim: f32) -> (f32, f32) {
    let scale = (src1 - src0) / (dst1 - dst0);
    (scale / dim, (src0 - dst0 * scale) / dim)
}

/// Quantise a normalised plane `(u_d, u_c)` ([`float_plane`]) at device sub-pixel position `pos` to
/// `(lower texel, 8-bit weight)`, reproducing WARP's evaluation: `u = u_d·pos + u_c` in f32
/// (separate multiply and add), clamp to `[−10,10]` (WARP's guard), scale by `dim`, bias `−0.5` for
/// texel-centre sampling, then `floor(·×256)` → `>>8` texel, `&0xff` weight.
pub fn plane_quant(u_d: f32, u_c: f32, dim: f32, pos: f32) -> (i32, i32) {
    let u = (u_d * pos + u_c).clamp(-10.0, 10.0);
    let q = ((u * dim - 0.5) * 256.0).floor() as i32;
    (q >> 8, q & 0xff)
}

/// Normalise a `value·128` fixed-point sample (the float-family bilinear's output, [`lerp_v`]/
/// [`lerp_h`]) to `[0,1]` exactly as the kernels do: `×1/128`, then the two f32 products `×1/256` and
/// `×1/65280` summed — `1/256 + 1/65280 = 1/255` exactly, WARP's reciprocal of 255.
pub fn norm_v128(v: i32) -> f32 {
    let a = v as f32 * (1.0 / 128.0);
    a * (1.0 / 256.0) + a * (1.0 / 65280.0)
}

impl Texture<'_> {
    /// WARP's anisotropic probe line at one mip: 1, 2, or 16 bilinear samples, reduced by
    /// pairwise shifts before addition. The direction is the longer screen-space derivative.
    pub(crate) fn anisotropic_v128_extend(&self, uv: [f32; 2], direction: [f32; 2], ratio: f32, extend: [Extend; 2]) -> [i32; 4] {
        // WARP's one texture-unit rule: the x address floors iff the width is a power of two; the y address
        // floors iff BOTH dimensions are (same as the cubic sampler, sprites and the bitmap brush). A
        // non-power-of-two width therefore rounds the y address too — the aniso path had keyed y on the
        // height alone, so a one-texel-tall non-pow2-wide source (r27 32667, 11x1) floored y where native
        // rounds it, a ±1 that surfaces once every pixel sits on that border step.
        let (w2, h2) = ((self.width as u32).is_power_of_two(), (self.height as u32).is_power_of_two());
        let pow2 = [w2, w2 && h2];
        self.anisotropic_v128_size(uv, direction, ratio, extend, [self.width as f32, self.height as f32], pow2)
    }

    // `dims` is the CONTENT-normalising size per axis — for a mip level it is `content / 2^level` (a
    // fractional value for a non-power-of-two content, e.g. 9→4.5 at mip 1), NOT the integer stored mip
    // width (`content >> level`). `pow2` is the BASE content's power-of-two-ness per axis (WARP's texture
    // unit floors the Q8 address on a power-of-two axis and rounds it (cvtps2dq) on any other, normalised by
    // the content size — one rule for every filter, 54a90d61). The extra clamped border texels the mip chain
    // stores hold the odd-column/row content the fractional `dims` addresses.
    pub(crate) fn anisotropic_v128_size(&self, uv: [f32; 2], direction: [f32; 2], ratio: f32, extend: [Extend; 2], dims: [f32; 2], pow2: [bool; 2]) -> [i32; 4] {
        self.anisotropic_v128_shifted(uv, direction, ratio, extend, dims, pow2, [0, 0])
    }

    /// [`Self::anisotropic_v128_size`] over a texture placed at texel `shift` of the one `uv` addresses (a brush's border
    /// cells in a frame atlas: `uv` is normalized over the atlas, the taps land on the brush's own texels).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn anisotropic_v128_shifted(&self, uv: [f32; 2], direction: [f32; 2], ratio: f32, extend: [Extend; 2], dims: [f32; 2], pow2: [bool; 2], shift: [i32; 2]) -> [i32; 4] {
        let (mut count, start, step) = if ratio <= 1.0 {
            (1, 0.0, 0.0)
        } else if ratio <= 2.0 {
            (2, -(ratio - 1.0) * 0.5, ratio - 1.0)
        } else {
            let step = ratio * (1.0 / 17.0);
            (16, ratio * -0.5 + step, step)
        };
        let axis = [direction[0] * (1.0 / dims[0]), direction[1] * (1.0 / dims[1])];
        let pair = core::array::from_fn::<_, 2, _>(|c| (((ratio - 1.0) * 0.5) * direction[c]) * (1.0 / dims[c]));
        let mut pos = if count == 2 { [uv[0] + pair[0], uv[1] + pair[1]] } else { [uv[0] + axis[0] * start, uv[1] + axis[1] * start] };
        let delta = [axis[0] * step, axis[1] * step];
        let mut taps = [[0; 4]; 16];
        for tap in &mut taps[..count] {
            // WARP's Q8 address floors a power-of-two (mip) dimension and rounds any other (dimg-sc-*-i4-*).
            let q: [(i32, i32); 2] = core::array::from_fn(|i| {
                let a = (pos[i] * dims[i] - 0.5) * 256.0;
                let q = if pow2[i] { a.floor() } else { a.round_ties_even() } as i32;
                ((q >> 8) - shift[i], q & 0xff)
            });
            *tap = core::array::from_fn(|ch| self.bilinear_v128(extend, ch, q[0], q[1]));
            if count == 2 { pos = [uv[0] - pair[0], uv[1] - pair[1]]; }
            else { pos[0] += delta[0]; pos[1] += delta[1]; }
        }
        while count > 1 {
            count /= 2;
            for i in 0..count {
                taps[i] = core::array::from_fn(|ch| (taps[2 * i][ch] >> 1) + (taps[2 * i + 1][ch] >> 1));
            }
        }
        taps[0]
    }

    /// Float-family 2×2 bilinear of channel `ch`, returning the `value·128` fixed point (callers
    /// normalise with [`norm_v128`]): vertical [`lerp_v`] of each column with weight `wy`, then
    /// horizontal [`lerp_h`] with weight `wx`. `x`/`y` are `(lower texel, 8-bit weight)` from
    /// [`float_axis`]. This is the sampler `FillOpacityMask` (and the smooth filters) read, keeping
    /// sub-byte precision the integer [`sample_fixed`] path discards.
    pub fn bilinear_v128(&self, ex: [Extend; 2], ch: usize, x: (i32, i32), y: (i32, i32)) -> i32 {
        let ((x0, wx), (y0, wy)) = (x, y);
        let t = |xx, yy| self.texel(xx, yy, ex)[ch];
        let cl = lerp_v(t(x0, y0), t(x0, y0 + 1), wy);
        let cr = lerp_v(t(x0 + 1, y0), t(x0 + 1, y0 + 1), wy);
        lerp_h(cl, cr, wx)
    }

    /// All four channels of [`Self::bilinear_v128`] at once: fetch the 2×2 texels ONE time (not once per
    /// channel) and run the identical per-channel `value·128` lerp across the four lanes. Byte-identical.
    pub fn bilinear_v128_all(&self, ex: [Extend; 2], x: (i32, i32), y: (i32, i32)) -> [i32; 4] {
        let ((x0, wx), (y0, wy)) = (x, y);
        let (c00, c01, c10, c11) = (
            self.texel(x0, y0, ex),
            self.texel(x0, y0 + 1, ex),
            self.texel(x0 + 1, y0, ex),
            self.texel(x0 + 1, y0 + 1, ex),
        );
        core::array::from_fn(|ch| {
            let cl = lerp_v(c00[ch], c01[ch], wy);
            let cr = lerp_v(c10[ch], c11[ch], wy);
            lerp_h(cl, cr, wx)
        })
    }

}

/// Split a texel coordinate into the lower texel index and the 8-bit sub-texel weight, exactly as
/// WARP's emitted code does: form the 16.16 fixed-point texcoord, subtract `0x7f80`
/// (`= 0.5 texel − 0.5 of an 8-bit fraction`, so the half-texel centre offset and the fraction's
/// round-to-nearest bias are folded into one bias), then take the integer part (arithmetic shift
/// right 16) and fraction bits `[15:8]`.
pub(crate) fn split(t: f32) -> (i32, i32) {
    // WARP converts the texcoord to 16.16 with `cvtps2dq`, which maps a non-finite coordinate to
    // 0x80000000 (SSE "integer indefinite"): the address is i32::MIN, so the texel is i32::MIN>>16
    // with zero sub-texel weight. The integer bias subtract below would otherwise overflow i64 on a
    // non-finite texcoord and sample a stray texel (r12 17139).
    if !t.is_finite() {
        let c = i32::MIN;
        return (c >> 16, (c >> 8) & 0xff);
    }
    let c = (t * 65536.0).round_ties_even() as i64 - 0x7f80;
    ((c >> 16) as i32, ((c >> 8) & 0xff) as i32)
}

/// WARP's separable Catmull-Rom (a = −0.5) combine over a caller-supplied texel `fetch` (so both the
/// [`Texture`] sampler and the effect resampler — which has its own soft/hard border — share one
/// implementation). `fetch(x, y)` returns the premultiplied channels of texel `(x, y)` as `i32`
/// (any channel order, preserved on output; the caller applies extend/border). `hq` selects the
/// HIGH_QUALITY_CUBIC per-window `÷255` normalisation ([`Texture::sample_hqcubic`]) vs mode-2 cubic's
/// normalise-once-at-the-end ([`Texture::sample_cubic`]); see those methods for the WARP derivation.
pub fn cubic_combine(hq: bool, tx: f32, ty: f32, fetch: impl Fn(i32, i32) -> [i32; 4]) -> [f32; 4] {
    cubic_combine_sharpness(hq, tx, ty, 0.0, fetch)
}

pub(crate) fn cubic_combine_sharpness(hq: bool, tx: f32, ty: f32, sharpness: f32, fetch: impl Fn(i32, i32) -> [i32; 4]) -> [f32; 4] {
    let (ix, fx) = cubic_axis(tx);
    let (iy, fy) = cubic_axis(ty);
    let (wx, wy) = (cubic_weights(fx, sharpness), cubic_weights(fy, sharpness));
    let (gcx, gcy) = (wx[1] + wx[2], wy[1] + wy[2]);
    let (mx, my) = (sigg_q8(wx[2], gcx), sigg_q8(wy[2], gcy));
    // Per axis: (is_centre, first-tap index, combine weight). Outer taps are a single column/row (the
    // degenerate lerp picks that tap); the centre is the 2×2 Sigg bilinear.
    let xs = [(false, ix - 1, wx[0]), (true, ix, gcx), (false, ix + 2, wx[3])];
    let ys = [(false, iy - 1, wy[0]), (true, iy, gcy), (false, iy + 2, wy[3])];
    core::array::from_fn(|ch| {
        // Separable f32 combine in WARP's order: sum the three y-samples per x-column, then sum the
        // columns. HQ-cubic normalises each window here; mode-2 cubic normalises `acc` once.
        let mut acc = 0.0f32;
        for &(xc, bx, wxv) in &xs {
            let mut col = 0.0f32;
            for &(yc, by, wyv) in &ys {
                let m = if yc { my } else { 0 };
                let t = |x, y| fetch(x, y)[ch];
                let v = if xc {
                    let cl = lerp_v(t(bx, by), t(bx, by + 1), m);
                    let cr = lerp_v(t(bx + 1, by), t(bx + 1, by + 1), m);
                    lerp_h(cl, cr, mx)
                } else {
                    lerp_v(t(bx, by), t(bx, by + 1), m)
                };
                col += wyv * if hq { norm_v128(v) } else { v as f32 * (1.0 / 128.0) };
            }
            acc += wxv * col;
        }
        // `×1/256 + ×1/65280 = 1/255` exactly in f32 (65280 = 256·255); HQ already did it per window.
        let n = if hq { acc } else { acc * (1.0 / 256.0) + acc * (1.0 / 65280.0) };
        n.clamp(0.0, 1.0)
    })
}

/// Cubic texcoord → (centre texel index `i`, sub-texel fraction `f ∈ [0,1)`). WARP biases the
/// texel-space coordinate by −0.5 (centres at integer), so the cubic's centre pair is `{i, i+1}`;
/// the Catmull-Rom weights ([`cubic_weights`]) use the full-precision fraction.
fn cubic_axis(t: f32) -> (i32, f32) {
    let i = (t - 0.5).floor();
    (i as i32, (t - 0.5) - i)
}

/// Sigg within-pair offset for the centre pair as an 8-bit weight: `floor(256·w2/(w1+w2))`. `w1,w2`
/// are both ≥ 0 so the offset stays in `[0,256)`; a degenerate `g == 0` yields 0.
fn sigg_q8(w2: f32, g: f32) -> i32 {
    if g == 0.0 { 0 } else { ((w2 / g) * 256.0).floor() as i32 & 0xff }
}

/// Vertical UNORM8 lerp of two texel bytes, `((b−a)·m + (a<<8)) >> 1` → `value·128` (WARP's cubic
/// kernel at `0xcd0`; `m` is the 8-bit weight). A single tap (outer sample) passes `m = 0`, giving
/// `a·128`.
fn lerp_v(a: i32, b: i32, m: i32) -> i32 {
    ((b - a) * m + (a << 8)) >> 1
}

/// Horizontal lerp of two `value·128` columns, `l + ((w·(r−l)) >> 8)` (WARP's 16-bit high-multiply
/// step); `w` is the 8-bit weight. Stays in `value·128`.
fn lerp_h(l: i32, r: i32, w: i32) -> i32 {
    l + ((w * (r - l)) >> 8)
}

/// WARP's cubic weights in Horner order. DrawBitmap uses Catmull-Rom (sharpness zero);
/// Scale maps its sharpness property to a = -0.5 - 1.5*sharpness (traced on WARP's high-quality scale).
pub(crate) fn cubic_weights(t: f32, sharpness: f32) -> [f32; 4] {
    let a = -0.5 - 1.5 * sharpness;
    [
        ((a * t - 2.0 * a) * t + a) * t,
        (((a + 2.0) * t - (a + 3.0)) * t) * t + 1.0,
        ((-(a + 2.0) * t + (2.0 * a + 3.0)) * t - a) * t,
        ((-a * t + a) * t) * t,
    ]
}

/// Combine nine Q8 bilinear windows, normalizing each before the X then Y weighted sums.
pub(crate) fn hqcubic_windows(
    xs: [(i32, i32); 3], ys: [(i32, i32); 3], wx: [f32; 3], wy: [f32; 3],
    fetch: impl Fn(i32, i32) -> [i32; 4],
) -> [f32; 4] {
    hqcubic_windows_unclamped(xs, ys, wx, wy, fetch).map(|v| v.clamp(0.0, 1.0))
}

pub(crate) fn hqcubic_windows_unclamped(
    xs: [(i32, i32); 3], ys: [(i32, i32); 3], wx: [f32; 3], wy: [f32; 3],
    fetch: impl Fn(i32, i32) -> [i32; 4],
) -> [f32; 4] {
    // Fetch each texel once (all four channels at once) rather than once per channel, then run the
    // identical per-channel lerp/normalise/weight arithmetic across the four lanes. Each window's 2×2
    // was previously fetched four times (one channel each); this is the same math, four times fewer
    // texel reads.
    let row = |(y, fy): (i32, i32)| -> [f32; 4] {
        let s = xs.map(|(x, fx)| {
            let (a, b, c, d) = (fetch(x, y), fetch(x, y + 1), fetch(x + 1, y), fetch(x + 1, y + 1));
            core::array::from_fn::<f32, 4, _>(|ch| {
                let left = lerp_v(a[ch], b[ch], fy);
                let right = lerp_v(c[ch], d[ch], fy);
                norm_v128(lerp_h(left, right, fx))
            })
        });
        core::array::from_fn(|ch| (s[0][ch] * wx[0] + s[1][ch] * wx[1]) + s[2][ch] * wx[2])
    };
    let (r0, r1, r2) = (row(ys[0]), row(ys[1]), row(ys[2]));
    core::array::from_fn(|ch| r1[ch] * wy[1] + (r2[ch] * wy[2] + r0[ch] * wy[0]))
}

/// WARP's emitted UNORM8 bilinear lerp step: `a + (((b − a)·w + 128) >> 8)`, in 16-bit lanes with a
/// logical shift. `a`,`b` ∈ `0..=255`, `w` ∈ `0..=255`; done here in `i32` (an arithmetic shift of
/// the signed delta reproduces the two's-complement + logical-shift result bit-for-bit) so it is a
/// round-half-up lerp between two bytes.
fn lerp8(a: i32, b: i32, w: i32) -> i32 {
    a + (((b - a) * w + 128) >> 8)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A 144-dpi bitmap at 120 dpi: 0.8333334 device px per texel from 12.500001. The pixel centre 17.5 is texel 6.0
    // exactly; native takes texel 5 there, and 6 at 18.5 (sc156 dip-b144-near).
    #[test]
    fn dda_texel_rounds_ties_by_accumulation() {
        let (s, o) = (0.8333334f32, 12.500001f32);
        let i = 1.0 / (s * s);
        let (m0, m4) = (i * s, -(s * o) * i);
        let t: Vec<i32> = (12..19).map(|x| dda_texel(m0, m4, 12.0, x as f32)).collect();
        assert_eq!(t, [0, 1, 2, 3, 4, 5, 7]);
    }

    // `bilinear_v128_all` fetches each window's 2×2 once and must equal the per-channel `bilinear_v128`
    // for every channel, address and extend mode — the all-inputs proof for the fetch-once refactor.
    #[test]
    fn bilinear_v128_all_matches_per_channel() {
        let data: Vec<u8> = (0..12 * 4).map(|i| ((i * 53 + 17) % 256) as u8).collect();
        let tex = Texture { data: &data, width: 4, height: 3 };
        for ex0 in [Extend::Clamp, Extend::Wrap, Extend::Mirror, Extend::Transparent] {
            for ex1 in [Extend::Clamp, Extend::Wrap, Extend::Mirror, Extend::Transparent] {
                let ex = [ex0, ex1];
                for x0 in [-1i32, 0, 1, 3, 4] {
                    for wx in [0i32, 1, 127, 255] {
                        for y0 in [-1i32, 0, 2, 3] {
                            for wy in [0i32, 128, 255] {
                                let all = tex.bilinear_v128_all(ex, (x0, wx), (y0, wy));
                                for ch in 0..4 {
                                    assert_eq!(all[ch], tex.bilinear_v128(ex, ch, (x0, wx), (y0, wy)), "x0={x0} wx={wx} y0={y0} wy={wy} ch={ch}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
