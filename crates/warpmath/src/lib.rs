//! WARP's shader-math intrinsics, reproduced bit-for-bit as WARP 10.0.26100 evaluates DXBC `ps_5_0`,
//! verified on the reference VM against the native Direct2D lighting shaders; the same canonical
//! constant block
//! (`0.6931381` ln2-hi, the fdlibm P1..P5 series, the `2/ln2` log2 rational, the sincos minimax)
//! appears across shaders, so these are WARP's one math library, not per-shader code.
//!
//! Everything is `f32` in WARP's round-to-nearest-even with the exact operation order WARP emits —
//! `pow`, for instance, is `exp2(y * log2(x))` because DXBC lowers `pow` to `log`+`mul`+`exp`, and
//! each of `log2`/`exp2` is WARP's own polynomial, not libm's. The CPU shader-executor
//! (`d2d-shader`) and the lighting effect kernels call these so their transcendentals match WARP.
//!
//! `exp2`/`log2` reproduce the fdlibm-style decompositions WARP uses, with WARP's exact f32
//! constants (`2/ln2` log2 rational, the `P1..P5` exp series, the `ln2` hi/lo split); `rsq`/`rcp`/
//! `sqrt` are exact IEEE square root and division (WARP uses no reciprocal approximations).
//! `pow`/`exp2`/`log2` are the intrinsics the `d2d-shader` executor routes through here. (The
//! lighting scenes `fx-transform-18/19/20` still differ ≤1 LSB, but that residual is *not* these
//! intrinsics — it is Direct2D's CPU-side distant light-direction constant and WARP's sampler
//! texcoord quantisation.)

/// `rcp` — WARP computes an exact division (`1.0 / x`), not a reciprocal approximation.
/// Correctly-rounded, so CPU-vendor independent.
#[inline]
pub fn rcp(x: f32) -> f32 {
    1.0 / x
}

/// `sqrt` — exact square root (IEEE correctly-rounded).
#[inline]
pub fn sqrt(x: f32) -> f32 {
    x.sqrt()
}

/// `rsq` — `1.0 / sqrt(x)` as WARP computes it: square root then divide (two correctly-rounded steps),
/// *not* a reciprocal-sqrt approximation. `normalize()` is `dp3`+`rsq`+`mul`, so a vector is scaled
/// by this reciprocal, never divided component-wise — reproduce that to match the last LSB.
#[inline]
pub fn rsqrt(x: f32) -> f32 {
    1.0 / x.sqrt()
}

/// `normalize` of a 3-vector exactly as the shader does it: `dp3` (with WARP's add order), one
/// `rsq`, then `mul` each lane by that scalar.
#[inline]
pub fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let d = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let inv = rsqrt(d);
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

// --- log2 -------------------------------------------------------------------------------------

/// Breakpoint table WARP's log2 looks up: `log2(c)` for `c ∈ {1.0, 1.25, 1.5, 1.75}` (the top two
/// mantissa bits select `c`), stored as f32.
const LOG2_TAB: [f32; 4] = [
    0.0,
    0.321_928_09,  // 0x3ea4d3c2
    0.584_962_5,   // 0x3f15c01a
    0.807_354_9,   // 0x3f4eaed0
];

/// `log2` — WARP's `log` opcode. `x = 2^E·m`, `m∈[1,2)`; the top two mantissa bits pick a breakpoint
/// `c∈{1,1.25,1.5,1.75}`; `t=(m−c)/(m+c)`; `log2 = E + log2(c) + t·(2/ln2 + (2/3ln2)t² + (2/5ln2)t⁴)`.
/// `x=0 → −inf`, `x<0/NaN → NaN`, `x=+inf → +inf` (exactly the shader's blends).
pub fn log2(x: f32) -> f32 {
    let bits = x.to_bits();
    let ibits = bits as i32;
    let e = (ibits >> 23) - 127;
    let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    let idx = ((ibits >> 21) & 3) as usize;
    let c = f32::from_bits(((idx as u32) << 21) | 0x3f80_0000);
    let t = (m - c) / (m + c);
    let t2 = t * t;
    let b = 0.577_078_04_f32 * t2 + 0.961_796_7; // 0x3f13bb63, 0x3f76384f
    let poly = t2 * b + 2.885_390_04; // 0x4038aa3b = 2/ln2
    let r = (e as f32 + LOG2_TAB[idx]) + t * poly;
    // Special cases, matching WARP's select chain (order: zero, negative, infinity, NaN).
    if x < 0.0 {
        f32::NAN
    } else if x == 0.0 {
        f32::NEG_INFINITY
    } else if x == f32::INFINITY {
        f32::INFINITY
    } else if x.is_nan() {
        x
    } else {
        r
    }
}

/// WARP's texture-sampler trilinear mip LOD for a squared derivative magnitude `rho2` (= the max over
/// the two screen axes of `|d(texcoord·dim)/d(axis)|²`, the 2×2-quad coarse derivative), clamped to
/// `[0, max_lod]`. There is NO sqrt — WARP's trilinear sampler takes `0.5·log2(rho2)`
/// (= `log2(sqrt(rho2))`) through this same [`log2`] routine (the exponent split + the
/// `2/ln2` rational), then clamps. Shared by the d2d-shader executor's sampler and the COLR gradient fills.
pub fn mip_lod(rho2: f32, max_lod: f32) -> f32 {
    if !(rho2 > 0.0) {
        return 0.0;
    }
    (0.5 * log2(rho2)).clamp(0.0, max_lod)
}

/// WARP's **anisotropic-sampler** log2, which is a lower-order approximation than [`log2`] (the shared
/// COLR/shader one): a single `c = 1` with `t = (m−1)/(m+1)` and only a 2-term polynomial
/// `e + t·(2/ln2 + 0.961797·t²)` — no quadrant table, no `t⁴` term. Decoded from WARP's aniso pixel
/// kernel (const pool c0=`0x4038aa3b`=2/ln2, c1=`0x3f76384f`). Kept separate from [`log2`] so the COLR
/// gradient / shader sampler is untouched (`aniso-store-round-vs-floor`).
pub fn log2_aniso(x: f32) -> f32 {
    let bits = x.to_bits();
    let e = (bits as i32 >> 23) - 127;
    let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    let t = (m - 1.0) / (m + 1.0);
    let poly = 0.961_796_7_f32 * (t * t) + 2.885_390_04; // 0x3f76384f, 0x4038aa3b
    e as f32 + t * poly
}

/// [`mip_lod`] using the anisotropic sampler's own [`log2_aniso`]. For the anisotropic DrawBitmap/brush
/// trilinear LOD only.
pub fn mip_lod_aniso(rho2: f32, max_lod: f32) -> f32 {
    if !(rho2 > 0.0) {
        return 0.0;
    }
    (0.5 * log2_aniso(rho2)).clamp(0.0, max_lod)
}

// --- exp2 -------------------------------------------------------------------------------------

const EXP2_LN2_HI: f32 = 0.693_138_12; // 0x3f317180
const EXP2_LN2_LO: f32 = 9.058_001e-6; // 0x3717f7d1
// fdlibm expf minimax series P1..P5.
const EXP2_P1: f32 = 0.166_666_67; // 0x3e2aaaab
const EXP2_P2: f32 = -0.002_777_777_8; // 0xbb360b61
const EXP2_P3: f32 = 6.613_756e-5; // 0x388ab355
const EXP2_P4: f32 = -1.653_390_2e-6; // 0xb5ddea0e
const EXP2_P5: f32 = 4.138_137e-8; // 0x3331bb4c

/// `exp2` — WARP's `exp` opcode: `n = round(y)` (half-up), `f = y−n ∈ [−0.5,0.5]`; the fractional
/// part is `exp(f·ln2)` via the fdlibm `expf` rational, then scaled by `2^n` with an integer add to
/// the exponent. `y ≥ 128 → +inf`, `y < −126 → 0`, NaN propagates.
pub fn exp2(y: f32) -> f32 {
    // The float→int conversion is round-to-nearest-even; WARP then bumps a "round-half-even-down" tie up by 1,
    // making the integer part round-half-up.
    let mut n = y.round_ties_even() as i32;
    if (n as f32) - y == -0.5 {
        n += 1;
    }
    let nf = n as f32;
    let f = y - nf;
    let hi = f * EXP2_LN2_HI;
    let flo = f * EXP2_LN2_LO;
    let r = hi + flo;
    let neg_lo = -flo;
    let r2 = r * r;
    let p = (((EXP2_P5 * r2 + EXP2_P4) * r2 + EXP2_P3) * r2 + EXP2_P2) * r2 + EXP2_P1;
    let c = r - r2 * p;
    let rc = (r * c) / (2.0 - c);
    let yv = 1.0 - ((neg_lo - rc) - hi);
    let scaled = f32::from_bits(((yv.to_bits() as i32).wrapping_add(n << 23)) as u32);
    if y < -126.0 {
        0.0
    } else if 128.0 <= y {
        f32::INFINITY
    } else if y.is_nan() {
        y
    } else {
        scaled
    }
}

/// `pow(x, y)` as DXBC lowers it: `exp2(y * log2(x))`.
#[inline]
pub fn pow(x: f32, y: f32) -> f32 {
    exp2(y * log2(x))
}

/// `frc` — fractional part `x - floor(x)` (floor rounds toward −∞), result clamped to the
/// largest float below 1 (`0x3f7fffff`) so it is always `< 1`, exactly as WARP computes it.
#[inline]
pub fn frc(x: f32) -> f32 {
    let f = x - x.floor();
    if f >= 1.0 {
        f32::from_bits(0x3f7f_ffff)
    } else {
        f
    }
}

// --- sincos -----------------------------------------------------------------------------------

const SINCOS_INV_2PI: f32 = f32::from_bits(0x3e22_f983); // 1/(2π)
const SINCOS_2PI: f32 = f32::from_bits(0x40c9_0fdb); // 2π

/// `(cos(kπ/16), sin(kπ/16))` for k=0..31 as raw f32 bits — the exact segment table WARP's `sincos`
/// looks up (interleaved cos/sin). These are *not* the correctly-rounded cos/sin values (e.g.
/// `sin(π)` is a tiny non-zero), so they are kept verbatim.
const SINCOS_TAB: [(u32, u32); 32] = [
    (0x3f80_0000, 0x0000_0000), (0x3f7b_14be, 0x3e47_c5c2), (0x3f6c_835e, 0x3ec3_ef16),
    (0x3f54_db31, 0x3f0e_39da), (0x3f35_04f3, 0x3f35_04f3), (0x3f0e_39d9, 0x3f54_db32),
    (0x3ec3_ef15, 0x3f6c_835e), (0x3e47_c5bc, 0x3f7b_14bf), (0xb33b_bd2e, 0x3f80_0000),
    (0xbe47_c5c2, 0x3f7b_14be), (0xbec3_ef18, 0x3f6c_835e), (0xbf0e_39dc, 0x3f54_db30),
    (0xbf35_04f3, 0x3f35_04f3), (0xbf54_db32, 0x3f0e_39d9), (0xbf6c_8360, 0x3ec3_ef10),
    (0xbf7b_14bf, 0x3e47_c5c1), (0xbf80_0000, 0xb3bb_bd2e), (0xbf7b_14be, 0xbe47_c5cd),
    (0xbf6c_835e, 0xbec3_ef15), (0xbf54_db30, 0xbf0e_39db), (0xbf35_04f1, 0xbf35_04f5),
    (0xbf0e_39d6, 0xbf54_db34), (0xbec3_ef0b, 0xbf6c_8361), (0xbe47_c5c6, 0xbf7b_14be),
    (0x324c_de2e, 0xbf80_0000), (0x3e47_c5c8, 0xbf7b_14be), (0x3ec3_ef1b, 0xbf6c_835d),
    (0x3f0e_39dd, 0xbf54_db2f), (0x3f35_04f7, 0xbf35_04ef), (0x3f54_db31, 0xbf0e_39db),
    (0x3f6c_835f, 0xbec3_ef15), (0x3f7b_14bf, 0xbe47_c5bc),
];

/// `sincos` — WARP's `sincos` opcode, returning `(sin x, cos x)`. Reduces `x` to one of 32 turn
/// segments (`k = floor(32·frac(x/2π))`, residual angle `r = (frac − k/32)·2π`), evaluates `sin r`
/// and `cos r` with WARP's degree-3/4 Taylor, then recombines against the segment's tabulated
/// `cos(kπ/16)`/`sin(kπ/16)` by angle addition — WARP's exact operation order.
pub fn sincos(x: f32) -> (f32, f32) {
    let t = x * SINCOS_INV_2PI;
    let frac = t - t.floor();
    let kf = (32.0 * frac).floor();
    let k = (kf as i32 & 31) as usize;
    let r = (frac - kf * 0.03125) * SINCOS_2PI;
    let r2 = r * r;
    let sin_r = r + (r * r2) * (-1.0 / 6.0);
    let cos_r = r2 * ((1.0 / 24.0) * r2 + (-0.5)) + 1.0;
    let (c_a, s_a) = (f32::from_bits(SINCOS_TAB[k].0), f32::from_bits(SINCOS_TAB[k].1));
    let cos = c_a * cos_r - s_a * sin_r;
    let mut sin = s_a * cos_r + c_a * sin_r;
    // sin(-0) = -0, and negative subnormals keep their sign (WARP blends -0.0 where sign+exp == 0x80000000).
    if x.to_bits() & 0xff80_0000 == 0x8000_0000 {
        sin = -0.0;
    }
    (sin, cos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rsqrt_is_sqrt_then_div() {
        assert_eq!(rsqrt(4.0), 0.5);
        assert_eq!(rsqrt(2.0), 1.0 / 2.0f32.sqrt());
        assert_eq!(frc(3.25), 0.25);
        assert_eq!(frc(-0.75), 0.25);
    }

    #[test]
    fn log2_exp2_roundtrip_exact_powers() {
        for e in -20..20 {
            let x = 2.0f32.powi(e);
            assert_eq!(log2(x), e as f32, "log2 2^{e}");
            assert_eq!(exp2(e as f32), x, "exp2 {e}");
        }
        assert_eq!(log2(0.0), f32::NEG_INFINITY);
        assert!(log2(-1.0).is_nan());
    }

    #[test]
    fn sincos_matches_libm_and_zero_signs() {
        assert_eq!(sincos(0.0), (0.0, 1.0));
        assert_eq!(sincos(-0.0).0.to_bits(), (-0.0f32).to_bits()); // sin(-0) keeps the sign
        // Close to libm (WARP's segmented minimax, not correctly rounded) across several periods.
        for &x in &[0.3f32, 1.5, -2.0, 3.14159, 6.0, -12.5, 20.0] {
            let (s, c) = sincos(x);
            assert!((s - x.sin()).abs() <= 1e-5, "sin({x})={s} vs {}", x.sin());
            assert!((c - x.cos()).abs() <= 1e-5, "cos({x})={c} vs {}", x.cos());
        }
    }

    #[test]
    fn pow_matches_within_a_couple_ulp_of_libm() {
        // Sanity only (not the parity oracle): WARP's pow is close to libm but not identical.
        for &(x, y) in &[(0.5f32, 8.0f32), (0.8, 3.0), (0.3, 2.0), (0.95, 50.0)] {
            let got = pow(x, y);
            let libm = x.powf(y);
            assert!((got - libm).abs() <= 4.0 * libm.abs() * f32::EPSILON + 1e-7, "pow({x},{y})={got} vs {libm}");
        }
    }
}
