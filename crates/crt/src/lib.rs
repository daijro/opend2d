//! d2d1's own CPU-side maths, reproduced bit-for-bit from the MSVC CRT (`ucrtbase.dll`).
//!
//! Direct2D's GPU work runs through WARP's exact intrinsics (`warpmath`), but d2d1 also
//! computes a lot on the CPU — arc→Bézier conversion, rounded rects/ellipses, rotations, SVG
//! transforms, brush/colour setup, lighting-effect setup and so on — by calling the C runtime's
//! `sinf`/`cosf`/`tanf`/`acosf`/`atan2f`/`powf`/`expf`/`logf` (and their `double` forms). Those are
//! `ucrtbase.dll`'s implementations, not the platform libm, so Rust's `f32::sin()` (glibc on Linux,
//! musl on wasm, mingw on Windows) disagrees in the last ULP — enough to shift a curve's control
//! points and flip an edge pixel, and enough to break cross-platform digest parity.
//!
//! This module reproduces exactly the ucrtbase functions d2d1 imports, in pure Rust, so every platform
//! computes the identical bits native WARP's host CRT does. ucrtbase's FMA3 math is built from
//! published open source, so these are ports of that source (not clean-room from the binary), with the
//! provenance verified against `ucrtbase.dll` on the reference VM:
//!   - **AMD aocl-libm-ose** (BSD-3-Clause, © AMD 2008-2022): `sinf`/`cosf` (+ their Cody–Waite
//!     [`reduce`], [`sin_poly`]/[`cos_poly`], and [`remainder_piby2`] Payne–Hanek) and [`expf`].
//!     Verified bit-exact to ucrtbase over all 2³² f32 inputs (`harness/src/bin/{trig,expf}-verify.rs`).
//!   - **Arm optimized-routines** (MIT): `pow`/`powf` ([`pow_log`]/[`pow_exp`] + the `POW_LOG`/`EXP_TAB`
//!     tables and `POW_OFF`) — the constants match `__pow_log_data`/`__exp_data` bit-for-bit; parity
//!     verified on the gamma domain d2d reaches (`scripts/conformance.sh parity`).
//!   - **No published open-source equivalent — reimplemented to match the platform CRT (`ucrtbase.dll`)
//!     bit-exact, verified over all 2³² inputs:** `atan2f`/[`atan_unit`] and [`asinf`]. ucrtbase's atan uses a 256-entry `atan(c/256)`
//!     anchor table that matches neither AMD's `atanf` (6-segment RANGE/VALUE) nor Arm's (plain
//!     polynomial); its `asinf` is a single-precision rational `g·P(g)/Q(g)` (AMD aocl's `asinf` is a
//!     double-degree-8 polynomial — 1 ULP off on 0.12% of [−1,1]). Both verified bit-exact to ucrtbase
//!     over all 2³² f32 inputs (`harness/src/bin/{atan,asinf}-verify.rs`).
//! `sqrt`/`sqrtf` are omitted on purpose: IEEE-754 square root is correctly rounded and `f32::sqrt`
//! already matches the hardware `sqrtss`/`sqrtsd` ucrtbase uses.
//!
//! Which C runtime each component calls, from their import tables (Windows 11 26100):
//!
//! | Component   | C runtime                                  | Maths it calls |
//! |-------------|--------------------------------------------|----------------|
//! | Direct2D    | the system runtime, by name                | acosf, atan, atan2f, ceil(f), cos(f), expf, floor(f), fmodf, log(f), modff, powf, sin(f), sqrt(f), tanf |
//! | DirectWrite | the universal runtime's private entry points | atan2f, ceilf, cosf, expf, floor(f), log10f, logf, pow, roundf, sinf, sqrt(f), tanf |
//! | WIC         | the universal runtime's private entry points | atan2, ceilf, exp, floorf, ldexp, log, powf, sqrt |
//! | WARP        | the universal runtime's private entry points | acos, asin, atan, atan2, ceil(f), cos, cosh, exp, exp2, floor(f), fmod, log, log10, log2, pow, powf, roundf, sin, sinh, sqrt(f), tan, tanh |
//!
//! The private entry points forward to the universal runtime's public functions (bit-exact for `expf` and `sinf` over
//! every f32), so this crate's root ports serve DirectWrite, WIC and WARP. Direct2D's are the system runtime's, a
//! separate build that differs in every function here: they are in [`system_crt`] (`sinf`, `cosf`, `tanf`, `acosf`,
//! `atan2f`, `expf`, `powf`). Rounding, `floor`, `ceil`, `fmod` and `sqrt` are exact in both.

pub mod system_crt;

// --- sinf / cosf (ucrtbase FMA3 path) --------------------------------------------------------------
//
// ucrtbase picks an FMA3 implementation on any CPU with FMA3 (the reference VM), reproduced here with
// Rust's `f64::mul_add` (a fused multiply-add, correctly rounded and identical on every target — so
// the result is the same bits on Linux/wasm/aarch64/Windows). `|x| ≤ π/4` evaluates the polynomial
// directly; `π/4 < |x| < 2²⁴` does one Cody–Waite reduction `r = (|x| − n·π/2_hi) − n·π/2_lo` with
// the quadrant `n&3` selecting the sin or cos polynomial and the sign.
// ponytail: the `|x| ≥ 2²⁴` Payne–Hanek path (ucrtbase's `_L_2_by_pi_bits` table) is not reproduced —
// no d2d1 CPU angle is that large (rotations/azimuth are a few radians); the medium reduction still
// runs there, deterministic but not ucrtbase-exact for astronomically large arguments.

const TWO_OVER_PI: f64 = f64::from_bits(0x3fe4_5f30_6dc9_c883);
const PIO2_HI: f64 = f64::from_bits(0x3ff9_21fb_5440_0000);
const PIO2_LO: f64 = f64::from_bits(0x3dd0_b461_1a62_6331);
const SIXTH: f64 = f64::from_bits(0x3fc5_5555_5555_5555); // 1/6, the tiny-argument sin term
// sin(r) ≈ r + r³·(S0 + r²·(S1 + r²·(S2 + r²·S3)))   (−1/6, 1/120, −1/5040, 1/362880)
const S0: f64 = f64::from_bits(0xbfc5_5555_5555_5555);
const S1: f64 = f64::from_bits(0x3f81_1111_1111_1111);
const S2: f64 = f64::from_bits(0xbf2a_01a0_1a01_a01a);
const S3: f64 = f64::from_bits(0x3ec7_1de3_a556_c734);
// cos(r) ≈ (1 + r²·C0) + r⁴·(C1 + r²·(C2 + r²·(C3 + r²·C4)))  (−1/2,1/24,−1/720,1/40320,−1/3628800)
const C0: f64 = f64::from_bits(0xbfe0_0000_0000_0000);
const C1: f64 = f64::from_bits(0x3fa5_5555_5555_5555);
const C2: f64 = f64::from_bits(0xbf56_c16c_16c1_6c16);
const C3: f64 = f64::from_bits(0x3efa_01a0_1a01_a019);
const C4: f64 = f64::from_bits(0xbe92_7e4f_b778_9f5c);

fn sin_poly(r: f64) -> f64 {
    let r2 = r * r;
    let p = r2.mul_add(S3, S2);
    let p = p.mul_add(r2, S1);
    let p = p.mul_add(r2, S0);
    p.mul_add(r * r2, r)
}
fn cos_poly(r: f64) -> f64 {
    let r2 = r * r;
    let lo = r2.mul_add(C0, 1.0);
    let p = r2.mul_add(C4, C3);
    let p = p.mul_add(r2, C2);
    let p = p.mul_add(r2, C1);
    p.mul_add(r2 * r2, lo)
}
/// Cody–Waite reduce `ax = |x|` (π/4 < ax) to `(r, n&3)`. Exact while `n = ax·2/π` fits i32, i.e.
/// `ax < PH_THRESH` (≈2³¹·⁶⁵); above that [`remainder_piby2`] (Payne–Hanek) takes over.
fn reduce(ax: f64) -> (f64, i32) {
    let ni = ax.mul_add(TWO_OVER_PI, 0.5) as i32;
    let n = ni as f64;
    let t = (-n).mul_add(PIO2_HI, ax); // ax − n·π/2_hi  (fused)
    (t - n * PIO2_LO, ni & 3)
}

// --- Payne–Hanek large-argument reduction (ucrtbase FMA3) ------------------------------------------
//
// The Cody–Waite `reduce` above loses all precision once `n = ax·2/π` overflows i32 (|x| ≳ 2³¹·⁶⁵),
// so for large angles — e.g. SVG `rotate(1e30)` — sinf/cosf must reduce mod π/2 against a stored 2/π.
// Ported from AMD aocl-libm-ose `call_remainder_piby2_fma3` (`src/isa/avx2/include/trig_func.h`, with
// the 2/π bits + π/2 parts from `remainder_piby2.c`; BSD-3-Clause, © AMD 2008-2022), the same library
// ucrtbase's FMA3 path is built from. `mul_add` stands in for each `vfmadd` so every target computes
// identical bits; verified bit-exact against the reference VM's ucrtbase sinf AND cosf over all 2³²
// f32 inputs (`harness/src/bin/trig-verify.rs`). The reduction yields `(r, region)` that feed the same
// `sin_poly`/`cos_poly` the medium path uses (they consume only `r`).

/// `|x|` f32 bits at which the Cody–Waite `n = ax·2/π` overflows i32 — the medium↔Payne–Hanek cut.
const PH_THRESH: u32 = 0x4f49_0fdb;

/// 2/π bits — AMD aocl-libm-ose `remainder_piby2.c` `L__2_by_pi_bits`.
#[rustfmt::skip]
static TWO_BY_PI_BITS: [u8; 160] = [
    224,241,27,193,12,88,33,116,53,126,196,126,237,175,169,75,74,41,222,231,
    28,244,236,197,151,175,31,235,158,212,181,168,127,121,154,253,24,61,221,38,
    44,159,60,251,217,180,125,180,41,104,45,70,188,188,63,96,22,120,255,95,
    226,127,236,160,228,247,46,126,17,114,210,231,76,13,230,88,71,230,4,249,
    125,209,154,192,113,166,19,18,237,186,212,215,8,162,251,156,166,196,114,172,
    119,248,115,72,70,39,168,187,36,25,128,75,55,9,233,184,145,220,134,21,
    239,122,175,142,69,249,7,65,14,241,100,86,138,109,3,119,211,212,71,95,
    157,240,167,84,16,57,185,13,230,139,2,0,0,0,0,0,0,0,0,0,
];
const PIBY2_LEAD: f64 = f64::from_bits(0x3ff9_21fb_5444_2d18);
const PIBY2_PART1: f64 = f64::from_bits(0x3ff9_21fb_5000_0000);
const PIBY2_PART2: f64 = f64::from_bits(0x3e51_10b4_6000_0000);
const PIBY2_PART3: f64 = f64::from_bits(0x3c91_a626_3314_5c06);

/// Payne–Hanek mod-π/2 reduction of large `x` (x>0); returns `(r, region&3)` for `sin_poly`/`cos_poly`.
fn remainder_piby2(x: f64) -> (f64, i32) {
    let xbits = x.to_bits();
    let xexp = ((xbits >> 52) as i64) - 1023;
    let last = (134 - (xexp >> 3)) as usize; // byte offset into the 2/π table
    let mant = (xbits & 0x000f_ffff_ffff_ffff) | (1u64 << 52);
    let ru64 = |o: usize| u64::from_le_bytes(TWO_BY_PI_BITS[o..o + 8].try_into().unwrap());
    let (m0, m1, m2) = (ru64(last), ru64(last + 8), ru64(last + 16));
    // 192-bit mantissa × 2/π (res1[0] most-significant … res1[2] least)
    let p0 = m0 as u128 * mant as u128;
    let r8 = p0 as u64;
    let carry0 = (p0 >> 64) as u64;
    let p1 = m1 as u128 * mant as u128;
    let (rax1, cf1) = (p1 as u64).overflowing_add(carry0);
    let r9 = rax1;
    let carry1 = (p1 >> 64) as u64 + cf1 as u64;
    let r10 = (m2 as u128 * mant as u128) as u64 + carry1;
    let resexp = (xexp & 7) as u32;
    // region = round(res1[0] >> (54−resexp)); a `point_five` carry negates the fraction
    let cl = 54 - resexp;
    let rax = r10 >> cl;
    let point_five = (r10 >> (cl - 1)) & 1;
    let (mut r10m, mut r9m, mut r8m, sign) = if point_five == 1 {
        (!r10, !r9, !r8, 0x8000_0000_0000_0000u64)
    } else {
        (r10, r9, r8, 0u64)
    };
    let region = ((rax + point_five) & 3) as i32;
    // strip the integer bits, leaving the fraction across r10m:r9m:r8m, then renormalise to x + xx
    let cl2 = resexp + 10;
    r10m = r10m.wrapping_shl(cl2).wrapping_shr(cl2);
    let mut r11e: i64 = cl2 as i64 - 64;
    let hsb = if r10m != 0 {
        63 - r10m.leading_zeros()
    } else {
        r10m = r9m;
        r9m = r8m;
        r8m = 0;
        r11e -= 64;
        63 - r10m.leading_zeros()
    };
    r11e += hsb as i64;
    let shift = hsb as i64 - 52;
    if shift < 0 {
        let c = (-shift) as u32;
        let c3 = 64 - c;
        let saved = r9m;
        r10m = r10m.wrapping_shl(c) | saved.wrapping_shr(c3);
        r9m = r9m.wrapping_shl(c) | r8m.wrapping_shr(c3);
    } else if shift > 0 {
        let c = shift as u32;
        let c3 = 64 - c;
        r8m = r10m;
        r10m = r10m.wrapping_shr(c);
        r9m = r9m.wrapping_shr(c) | r8m.wrapping_shl(c3);
    }
    let xexp_b = (r11e + 1023) as u64;
    let x_d = f64::from_bits((r10m & !(1u64 << 52)) | sign | (xexp_b << 52));
    let hsb9 = 63 - r9m.leading_zeros();
    let c9 = 64 - hsb9;
    let r9x = r9m.wrapping_shl(c9) >> 12;
    let xx_exp = xexp_b as i64 - (c9 as i64 + 52);
    let xx_d = f64::from_bits(r9x | sign | ((xx_exp as u64) << 52));
    // FMA3 refinement: r = (x+xx)·(π/2) carried in extended precision
    let hx = f64::from_bits(x_d.to_bits() & 0xffff_ffff_f800_0000);
    let tx = x_d - hx;
    let c = x_d * PIBY2_LEAD;
    let mut cc = hx.mul_add(PIBY2_PART1, -c);
    cc = tx.mul_add(PIBY2_PART1, cc);
    cc = hx.mul_add(PIBY2_PART2, cc);
    cc = tx.mul_add(PIBY2_PART2, cc);
    cc += x_d.mul_add(PIBY2_PART3, xx_d * PIBY2_LEAD);
    (c + cc, region)
}

/// `sinf` (ucrtbase).
pub fn sinf(x: f32) -> f32 {
    let v = x.to_bits() & 0x7fff_ffff;
    if v >= 0x7f80_0000 {
        return f32::NAN; // ±inf / nan (unreached by d2d1's finite angles)
    }
    let xd = x as f64;
    if v > 0x3f49_0fdb {
        // |x| > π/4: reduce |x|, pick polynomial by quadrant, reapply sin's odd sign
        let (r, q) = if v >= PH_THRESH { remainder_piby2(xd.abs()) } else { reduce(xd.abs()) };
        let mut p = if q & 1 == 1 { cos_poly(r) } else { sin_poly(r) };
        if q >= 2 {
            p = -p;
        }
        if xd < 0.0 {
            p = -p;
        }
        p as f32
    } else if v >= 0x3c00_0000 {
        sin_poly(xd) as f32 // π/4 ≥ |x|: direct (quadrant 0)
    } else if v >= 0x3900_0000 {
        let x3 = (xd * xd) * xd;
        (-x3).mul_add(SIXTH, xd) as f32 // x − x³/6
    } else {
        x // |x| tiny: sin x ≈ x
    }
}

/// `cosf` (ucrtbase).
pub fn cosf(x: f32) -> f32 {
    let v = x.to_bits() & 0x7fff_ffff;
    if v >= 0x7f80_0000 {
        return f32::NAN;
    }
    let xd = x as f64;
    if v > 0x3f49_0fdb {
        let (r, q) = if v >= PH_THRESH { remainder_piby2(xd.abs()) } else { reduce(xd.abs()) };
        let mut p = if q & 1 == 1 { sin_poly(r) } else { cos_poly(r) };
        if ((q + 1) >> 1) & 1 == 1 {
            p = -p; // negate for quadrants 1 and 2
        }
        p as f32
    } else if v >= 0x3c00_0000 {
        cos_poly(xd) as f32
    } else if v >= 0x3900_0000 {
        (xd * xd).mul_add(-0.5, 1.0) as f32 // 1 − x²/2
    } else {
        1.0
    }
}

// --- tanf (ucrtbase FMA3 path) ---------------------------------------------------------------------
//
// PROVENANCE: reproduced from `ucrtbase.dll`'s `tanf` FMA3 path (the rational
// constants + 2/π, π/2 hi/lo, 0.5 bias, −1 and 1/3 match ucrtbase's). ucrtbase
// reduces `|x|` to `r ∈ [−π/4, π/4]` with the SAME Cody–Waite [`reduce`] / Payne–Hanek [`remainder_piby2`]
// as sinf/cosf (same 2/π, π/2 parts, `n = |x|·2/π + 0.5`), then evaluates a single-precision-range RATIONAL
// `tan(r) ≈ r + r³·(num(r²)/den(r²))`, takes `−1/tan(r)` for an odd quadrant (`tan` has period π), and
// reapplies `x`'s sign (`tan` is odd). `mul_add` reproduces each `vfmadd`/`vfnmadd`; the one `vdivsd`
// (`num/den`) and the odd-quadrant `−1/p` are plain f64 divisions — so every target computes ucrtbase's bits.
// tan(r) rational coefficients: num = TAN_N1·r² + TAN_N0, den = (TAN_D2·r² + TAN_D1)·r² + TAN_D0.
const TAN_N0: f64 = f64::from_bits(0x3fd8_a8b0_da56_cb17);
const TAN_N1: f64 = f64::from_bits(0xbf91_9dba_6efd_6aad);
const TAN_D0: f64 = f64::from_bits(0x3ff2_7e84_a3e7_3a2e);
const TAN_D1: f64 = f64::from_bits(0xbfe0_7266_d7b3_511b);
const TAN_D2: f64 = f64::from_bits(0x3f92_e290_03c6_92d9);
const THIRD: f64 = f64::from_bits(0x3fd5_5555_5555_5555); // 1/3, the tiny-argument r³ term

fn tan_poly(r: f64) -> f64 {
    let r2 = r * r;
    let num = TAN_N1.mul_add(r2, TAN_N0);
    let den = TAN_D2.mul_add(r2, TAN_D1).mul_add(r2, TAN_D0);
    (r2 * r).mul_add(num / den, r) // r + r³·(num/den)
}

/// `tanf` (ucrtbase).
pub fn tanf(x: f32) -> f32 {
    let v = x.to_bits() & 0x7fff_ffff;
    if v >= 0x7f80_0000 {
        return f32::NAN; // ±inf / nan (unreached by d2d1's finite angles)
    }
    let xd = x as f64;
    if v > 0x3f49_0fdb {
        // |x| > π/4: reduce |x|, tan poly, cotangent for an odd quadrant, then reapply x's sign.
        let (r, q) = if v >= PH_THRESH { remainder_piby2(xd.abs()) } else { reduce(xd.abs()) };
        let mut p = tan_poly(r);
        if q & 1 == 1 {
            p = -1.0 / p; // odd quadrant: tan(x) = −cot(r)
        }
        if xd < 0.0 {
            p = -p; // tan is odd
        }
        p as f32
    } else if v >= 0x3900_0000 {
        tan_poly(xd) as f32 // π/4 ≥ |x|: direct (quadrant 0, signed — the poly is odd)
    } else if v >= 0x3200_0000 {
        let x3 = (xd * xd) * xd;
        x3.mul_add(THIRD, xd) as f32 // x + x³/3
    } else {
        x // |x| tiny: tan x ≈ x
    }
}

// --- atan2f (ucrtbase FMA3 path) -------------------------------------------------------------------
//
// PROVENANCE: now bit-exact to ucrtbase (`atan2f(y,1)` = ucrtbase for all 2^32, atan-verify.rs); almost
// all math, four minimax constants (AT_C13/AT_C15/AT_C17/AT_C13B) matched to ucrtbase bit-for-bit.
//  - ATAN_ANCHOR table is **generated math**: `truncate(atan(c/256))` toward zero, bit-exact for all 241
//    entries (c∈[16,256]) — verified with mpmath, so it is not opaque Windows data.
//  - Algorithm matches ucrtbase's atan2f_fma: first-octant ratio; ratio≤0.0625 uses a
//    sub-cutoff (ratio<~1e-4 → atan(r)=r, else a 3-term poly C13/C15/C17); larger ratios use the anchor
//    table + a 1-term residual (C13B). Exactly 3 poly terms — NO extra terms. (This is neither AMD aocl
//    `atanf`, 6-segment RANGE/VALUE, nor Arm optimized-routines `atanf`, plain poly.)
//  - `AT_C13`/`AT_C15`/`AT_C17`/`AT_C13B` are that atan MINIMAX, matched to ucrtbase bit-for-bit.
//    They are tweaks of Taylor (1/3,1/5,1/7) — NOT the correctly-rounded Taylor values
//    (29 / 2.2e5 / 2.6e8 ULP off) — and a clean Remez of the unknown metric/interval does not
//    reproduce their exact bits. The ~1e-4 cutoff `0x3f1a36e2eb1c432d` is a design threshold, and
//    π/2, π and the table are math. The −0-sign fold and the <1e-4 cutoff close the last 21/2^32 residuals.
//
// Reduce to the first octant `α = atan(min/max)`, then fold by the quadrant. `min/max < 0.0625` uses
// the degree-7 polynomial directly; larger ratios subtract a tabulated `atan(c/256)` anchor (c =
// round(256·ratio), c∈[16,256]) and add `atan` of the small residual. ATAN_ANCHOR[c-16] = atan(c/256).
const AT_C13: f64 = f64::from_bits(0x3fd5_5555_5555_5538);
const AT_C15: f64 = f64::from_bits(0x3fc9_9999_9996_43a3);
const AT_C17: f64 = f64::from_bits(0x3fc2_4924_82bd_6be1);
const AT_C13B: f64 = f64::from_bits(0x3fd5_5555_5550_0877); // residual term after a table anchor
const PI_2: f64 = f64::from_bits(0x3ff9_21fb_5444_2d18);
const PI_1: f64 = f64::from_bits(0x4009_21fb_5444_2d18);

/// atan of the first-octant ratio `r = num/den ≤ 1`, with `num`/`den` the ordered magnitudes.
fn atan_unit(num: f64, den: f64) -> f64 {
    let ratio = num / den;
    if ratio <= 0.0625 {
        // ratio below ~1e-4 (ucrtbase's cutoff): atan(r) = r to f32 significance — skip the poly, so
        // the omitted r³ term can't tip an f32 rounding tie (fixes the ~20 near-π/2 inputs).
        if ratio < f64::from_bits(0x3f1a_36e2_eb1c_432d) {
            return ratio;
        }
        let z2 = ratio * ratio;
        let p = z2.mul_add(-AT_C17, AT_C15);
        let p = z2.mul_add(-p, AT_C13);
        (ratio * z2).mul_add(-p, ratio)
    } else {
        let c = ((256.0f64).mul_add(ratio, 0.5) as i32).min(256); // round(256·ratio), c∈[16,256]
        let cf = c as f64;
        let z = (256.0f64).mul_add(num, -(cf * den)) / num.mul_add(cf, den * 256.0);
        let anchor = f64::from_bits(ATAN_ANCHOR[(c - 16) as usize]);
        (z * (z * z)).mul_add(-AT_C13B, z + anchor)
    }
}

/// `atan2f` (ucrtbase).
pub fn atan2f(y: f32, x: f32) -> f32 {
    // ucrtbase returns NaN if either argument is NaN. At infinities C99 `atan2` is the angle of the
    // unit direction (an infinite axis → ±1, a finite axis → 0), so reduce to that and fall through to
    // the finite path — otherwise a NaN ratio (e.g. inf/inf) would index `ATAN_ANCHOR` out of bounds
    // and panic (fuzz 30027 reached it through a non-finite glyph coordinate).
    if y.is_nan() || x.is_nan() {
        return f32::NAN;
    }
    let (mut yd, mut xd) = (y as f64, x as f64);
    if yd.is_infinite() || xd.is_infinite() {
        yd = if yd.is_infinite() { yd.signum() } else { 0.0 };
        xd = if xd.is_infinite() { xd.signum() } else { 0.0 };
    }
    let (ay, ax) = (yd.abs(), xd.abs());
    if ay == 0.0 && ax == 0.0 {
        return 0.0; // degenerate (unreached by real arcs)
    }
    let mut a = if ay > ax { PI_2 - atan_unit(ax, ay) } else { atan_unit(ay, ax) };
    if xd < 0.0 {
        a = PI_1 - a;
    }
    // fold on y's SIGN BIT, not `yd < 0.0`, so atan2f(−0, x) keeps the −0 (matches ucrtbase).
    (if yd.is_sign_negative() { -a } else { a }) as f32
}


// --- asinf (ucrtbase FMA3 `asinf_fma`) -------------------------------------------------------------
//
// Reproduced from `ucrtbase.dll`'s `asinf_fma`: a SINGLE-precision rational approximation,
// NOT AMD aocl's double polynomial (aocl is 1 ULP off on 0.12% of [−1,1]; see the header). Everything
// is f32, with the `vfmadd/vfnmadd/vfmsub` ops reproduced by `f32::mul_add` (fused) exactly where the
// native fuses (the two `*`/`/` steps are plain).
//
// |x| < 2⁻¹⁴ → x; |x| ≥ 1 → ±π/2 at 1, NaN beyond. Otherwise `g = x²` (|x|<0.5) or `g = ½(1−|x|)`,
// `s = √g` (0.5 ≤ |x| < 1), and `R = g·P(g)/Q(g)` with P (3 FMAs) and Q (1 FMA). The small branch
// returns `|x| + |x|·R`; the sqrt branch reconstructs `π/2 − 2s(1+R)` in extended precision (an
// `s = sh + sl` head/tail split and a low-order π/2 correction constant) so it stays exact near π/2.
// The `P`/`Q` coefficients and the `0x33a22168` correction match ucrtbase bit-for-bit,
// as with `atan2f`. π/4, π/2 and ½ are math.
/// `asinf` (ucrtbase). Domain [−1, 1]; verified bit-exact to ucrtbase over all 2³² f32 inputs.
pub fn asinf(x: f32) -> f32 {
    let bits = x.to_bits();
    let ax_exp = (bits >> 23) & 0xff;
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return f32::from_bits(bits | 0x40_0000); // NaN → quiet NaN
    }
    if ax_exp < 0x71 {
        return x; // |x| < 2⁻¹⁴ → asin(x) = x
    }
    if ax_exp >= 0x7f {
        // |x| ≥ 1: ±π/2 at exactly 1, NaN beyond
        let a = f32::from_bits(bits & 0x7fff_ffff);
        if a == 1.0 {
            return if (bits as i32) < 0 { f32::from_bits(0xbfc9_0fdb) } else { f32::from_bits(0x3fc9_0fdb) };
        }
        return f32::NAN;
    }
    let a = f32::from_bits(bits & 0x7fff_ffff); // |x|
    let neg = (bits as i32) < 0;
    let (g, s, sqrt_branch) = if ax_exp < 0x7e {
        (a * a, 0.0f32, false) // |x| < 0.5
    } else {
        let g = (1.0f32 - a) * 0.5; // 0.5 ≤ |x| < 1
        (g, g.sqrt(), true)
    };
    // numerator P(g) (3 fused FMAs), num = P(g)·g
    let p = g.mul_add(-f32::from_bits(0x3b81_ce6b), f32::from_bits(0xbc5b_3fe1)); // −(g·c0)+c1
    let p = g.mul_add(p, -f32::from_bits(0x3d67_8bdd)); // (g·p)−c2
    let p = g.mul_add(p, f32::from_bits(0x3e3c_94dc)); // (g·p)+c3
    let num = p * g;
    // denominator Q(g) (1 fused FMA), R = num/den
    let den = g.mul_add(-f32::from_bits(0x3f56_1f0d), f32::from_bits(0x3f8d_6fa5));
    let r = num / den;
    let res = if sqrt_branch {
        let sh = f32::from_bits(s.to_bits() & 0xffff_0000); // head of √g
        let sl = sh.mul_add(-sh, g) / (sh + s); // low part (g − sh²)/(sh + s) = s − sh
        let mut x4 = f32::from_bits(0x33a2_2168) - (sl + sl); // c − 2sl
        x4 = r.mul_add(s + s, -x4); // R·2s − x4
        let pi4 = f32::from_bits(0x3f49_0fda);
        let t = x4 - (pi4 - (sh + sh));
        pi4 - t // π/2 − 2s(1+R), reconstructed
    } else {
        r.mul_add(a, a) // |x| + |x|·R
    };
    if neg { -res } else { res }
}


/// ucrtbase's FMA3 `acosf`, including its head/tail reconstruction near ±1.
pub fn acosf(x: f32) -> f32 {
    let bits = x.to_bits();
    let a = x.abs();
    if a.is_nan() { return f32::from_bits(bits | 0x40_0000); }
    if a < f32::from_bits(0x3280_0000) { return core::f32::consts::FRAC_PI_2; }
    if a >= 1.0 {
        return if x == 1.0 { 0.0 } else if x == -1.0 { core::f32::consts::PI } else { f32::from_bits(0xffc0_0000) };
    }
    let g = if a < 0.5 { a * a } else { (1.0 - a) * 0.5 };
    let p = g.mul_add(-f32::from_bits(0x3b81_ce6b), f32::from_bits(0xbc5b_3fe1));
    let p = g.mul_add(p, -f32::from_bits(0x3d67_8bdd));
    let p = g.mul_add(p, f32::from_bits(0x3e3c_94dc));
    let r = (p * g) / g.mul_add(-f32::from_bits(0x3f56_1f0d), f32::from_bits(0x3f8d_6fa5));
    const PI_2_LOW: f64 = 6.123233995736766e-17;
    if a < 0.5 {
        return (core::f64::consts::FRAC_PI_2 - (x as f64 - (PI_2_LOW - (r * x) as f64))) as f32;
    }
    let s = g.sqrt();
    if x < 0.0 {
        let t = ((s * r) as f64 - PI_2_LOW) + s as f64;
        (core::f64::consts::PI - (t + t)) as f32
    } else {
        let sh = f32::from_bits(s.to_bits() & 0xffff_0000);
        let sl = sh.mul_add(-sh, g) / (sh + s);
        r.mul_add(s + s, sl + sl) + (sh + sh)
    }
}

// --- pow / powf (ucrtbase FMA3 `pow`) --------------------------------------------------------------
//
// d2d1's sRGB/CMS gamma (`color_context`/`brush`/`bitmap` `powf(2.4)` and `powf(1/2.4)`) is byte-exact
// on native via glibc's `pow` but DIVERGES on wasm (the `libm` crate), breaking cross-platform digest
// parity (cms2-04/05). ucrtbase's FMA3 `pow` is Arm optimized-routines' `pow` (MIT); this is a port of
// it — `POW_OFF`, the 128-entry `POW_LOG` table and the 256-entry `EXP_TAB`/poly match `__pow_log_data`
// and `__exp_data` bit-for-bit (tab[0]=0x1.6a00p0/−0x1.62c82f2b9c800p-2/0x1.ab42428375680p-48, ln2hi,
// the log1p/exp coefficients). `exp(y·ln x)`: a 128-entry log reduction and a 256-entry exp reduction,
// each using `mul_add` exactly where the native uses `vfmadd` (plain mul/add elsewhere), deterministic
// on every platform.
// ponytail: only the finite domain d2d's gamma reaches (base ∈ (0,1], exp ≈ 2.4 or 0.417) is
// bit-reproduced; x≤0, x==1, and `|y·ln x|` outside [2⁻⁵⁴, ~709) take simple correct branches rather
// than the native's gradual-underflow/overflow/NaN bit patterns (no gamma argument reaches them).

const POW_OFF: u64 = 0x3fe6_9555_0000_0000;

/// ucrtbase `pow`'s log reduction: returns `(w, hi, lo)` with `w = ln x` and `(hi, lo)` its split.
fn pow_log(ix: u64) -> (f64, f64, f64) {
    const LN2HI: f64 = f64::from_bits(0x3fe6_2e42_fefa_3800);
    const LN2LO: f64 = f64::from_bits(0x3d2e_f357_93c7_6730);
    let tmp = ix.wrapping_sub(POW_OFF);
    let i = ((tmp >> 45) & 0x7f) as usize;
    let k = (tmp as i64) >> 52;
    let z = f64::from_bits(ix.wrapping_sub(tmp & (0xfff_u64 << 52)));
    let kd = k as f64;
    let invc = f64::from_bits(POW_LOG[i][0]);
    let logc = f64::from_bits(POW_LOG[i][1]);
    let logctail = f64::from_bits(POW_LOG[i][2]);
    let r = invc.mul_add(z, -1.0);
    let t1 = kd * LN2HI + logc;
    let ar = r * f64::from_bits(0xbfe0_0000_0000_0000); // A0·r = −0.5·r
    let ar2 = ar * r;
    let t2 = t1 + r;
    let hi = ar2 + t2;
    let lo3 = ar.mul_add(r, -ar2);
    let lo1 = kd * LN2LO + logctail;
    let lo2 = (t1 - t2) + r;
    let lo_terms = (lo3 + (lo1 + lo2)) + ((t2 - hi) + ar2);
    // log1p polynomial in r (native immediates), evaluated with ar2 = −0.5·r² as the step:
    let pa = f64::from_bits(0x3fe5_5555_5529_a47a);
    let pb = f64::from_bits(0x3fe9_9999_9959_554e);
    let pc = f64::from_bits(0x3ff0_002b_8b26_3fc3);
    let pd = f64::from_bits(0x3ff2_495b_9b48_45e9);
    let pe = f64::from_bits(0x3fe0_0000_0000_0006);
    let pf = f64::from_bits(0x3fe5_5555_5555_5560);
    let p = (ar2 * r) * ((pe * r - pf) + ar2 * ((pb - pa * r) + ar2 * (pc * r - pd)));
    let lo = lo_terms + p;
    (lo + hi, hi, lo)
}

/// ucrtbase `pow`'s exp reduction of `ehi + elo` (`sign_bias` is 0 for a positive base).
fn pow_exp(ehi: f64, elo: f64) -> f64 {
    const INVLN2N: f64 = f64::from_bits(0x4077_1547_652b_82fe);
    const SHIFT: f64 = f64::from_bits(0x4238_0000_0000_8000);
    const LN2HIN: f64 = f64::from_bits(0x3f66_2e42_fefc_0000);
    const LN2LON: f64 = f64::from_bits(0x3d2c_610c_a86c_3899);
    let ki = (ehi * INVLN2N + SHIFT).to_bits() >> 16;
    let kd = (ki as u32 as i32) as f64;
    let r = ((ehi - kd * LN2HIN) + kd * LN2LON) + elo;
    let r2 = r * r;
    let idx = (ki & 0xff) as usize;
    let tail = f64::from_bits(EXP_TAB[2 * idx]);
    let scale = f64::from_bits(EXP_TAB[2 * idx + 1].wrapping_add(ki << 44));
    let ca = f64::from_bits(0x3fc5_5555_5555_543c);
    let cb = f64::from_bits(0x3fdf_ffff_ffff_fdbd);
    let cc = f64::from_bits(0x3f81_1111_67a4_b553);
    let cd = f64::from_bits(0x3fa5_5555_cf16_e1ed);
    let tmp = (tail + r) + r2 * (cb + ca * r) + (r2 * r2) * (cd + cc * r);
    scale.mul_add(tmp, scale)
}

/// `pow` (ucrtbase, double). Deterministic on every target; byte-exact for d2d's gamma domain.
pub fn pow(x: f64, y: f64) -> f64 {
    if x == 1.0 || y == 0.0 {
        return 1.0;
    }
    if !(x > 0.0) || !x.is_finite() || !y.is_finite() {
        return x.powf(y); // exotic: unreached by d2d's gamma
    }
    let (w, hi, lo) = pow_log(x.to_bits());
    let ehi = w * y;
    let elo = y.mul_add(w, -ehi) + ((hi - w) + lo) * y;
    if ehi.abs() < f64::from_bits(0x3c90_0000_0000_0000) {
        1.0 + ehi // |y·ln x| < 2⁻⁵⁴: exp ≈ 1 + ehi
    } else if ehi > 709.782_712_893_384 {
        f64::INFINITY
    } else if ehi < -745.133_219_101_941_1 {
        0.0
    } else {
        pow_exp(ehi, elo)
    }
}

/// `powf` (single): d2d's f32 gamma routed through the deterministic double `pow`.
pub fn powf(x: f32, y: f32) -> f32 {
    pow(x as f64, y as f64) as f32
}

// --- expf (ucrtbase FMA3 `expf`) -------------------------------------------------------------------
//
// d2d1's Gaussian-blur kernel-weight generation (the blur effect's cbuffer fill) calls the CRT `expf`;
// `libm::expf` is correctly-rounded but ucrtbase's is not, and the 1-ULP gap flips one blur weight and
// one output pixel (pdnv5-gaussianblur (3,0), dropshadow/shadow-only rim). ucrtbase's FMA3 math comes
// from AMD's aocl-libm-ose (BSD-3-Clause, © 2008-2022 Advanced Micro Devices, Inc.); this is a direct
// port of `src/isa/avx2/masm/expf_fma3.asm`, with `mul_add` for each `vfmadd` so every target computes
// the identical bits (verified bit-exact against the reference VM's `ucrtbase!expf` over all 2^32 f32
// inputs — `harness/src/bin/expf-verify.rs`). The `-103.277` early-out in the AMD asm is replaced by a
// `-104` one so the compute path produces ucrtbase's denormal tail (the asm's would floor it to 0).

/// 2^(j/64), j=0..63 — the plain table the FMA3 path indexes (correctly-rounded doubles).
#[rustfmt::skip]
static TWO_TO_JBY64: [u64; 64] = [
    0x3ff0000000000000, 0x3ff02c9a3e778061, 0x3ff059b0d3158574, 0x3ff0874518759bc8,
    0x3ff0b5586cf9890f, 0x3ff0e3ec32d3d1a2, 0x3ff11301d0125b51, 0x3ff1429aaea92de0,
    0x3ff172b83c7d517b, 0x3ff1a35beb6fcb75, 0x3ff1d4873168b9aa, 0x3ff2063b88628cd6,
    0x3ff2387a6e756238, 0x3ff26b4565e27cdd, 0x3ff29e9df51fdee1, 0x3ff2d285a6e4030b,
    0x3ff306fe0a31b715, 0x3ff33c08b26416ff, 0x3ff371a7373aa9cb, 0x3ff3a7db34e59ff7,
    0x3ff3dea64c123422, 0x3ff4160a21f72e2a, 0x3ff44e086061892d, 0x3ff486a2b5c13cd0,
    0x3ff4bfdad5362a27, 0x3ff4f9b2769d2ca7, 0x3ff5342b569d4f82, 0x3ff56f4736b527da,
    0x3ff5ab07dd485429, 0x3ff5e76f15ad2148, 0x3ff6247eb03a5585, 0x3ff6623882552225,
    0x3ff6a09e667f3bcd, 0x3ff6dfb23c651a2f, 0x3ff71f75e8ec5f74, 0x3ff75feb564267c9,
    0x3ff7a11473eb0187, 0x3ff7e2f336cf4e62, 0x3ff82589994cce13, 0x3ff868d99b4492ed,
    0x3ff8ace5422aa0db, 0x3ff8f1ae99157736, 0x3ff93737b0cdc5e5, 0x3ff97d829fde4e50,
    0x3ff9c49182a3f090, 0x3ffa0c667b5de565, 0x3ffa5503b23e255d, 0x3ffa9e6b5579fdbf,
    0x3ffae89f995ad3ad, 0x3ffb33a2b84f15fb, 0x3ffb7f76f2fb5e47, 0x3ffbcc1e904bc1d2,
    0x3ffc199bdd85529c, 0x3ffc67f12e57d14b, 0x3ffcb720dcef9069, 0x3ffd072d4a07897c,
    0x3ffd5818dcfba487, 0x3ffda9e603db3285, 0x3ffdfc97337b9b5f, 0x3ffe502ee78b3ff6,
    0x3ffea4afa2a490da, 0x3ffefa1bee615a27, 0x3fff50765b6e4540, 0x3fffa7c1819e90d8,
];

/// `expf` (ucrtbase FMA3). Deterministic on every target; bit-exact to `ucrtbase!expf` over all f32.
/// Ported from AMD aocl-libm-ose `expf_fma3.asm` (BSD-3-Clause, © AMD 2008-2022).
pub fn expf(x: f32) -> f32 {
    let xb = x.to_bits();
    if xb & 0x7f80_0000 == 0x7f80_0000 {
        // exponent all ones: +inf→+inf, -inf→0, nan→quiet(nan)
        return if xb == 0x7f80_0000 {
            x
        } else if xb == 0xff80_0000 {
            0.0
        } else {
            f32::from_bits(xb | 0x0040_0000)
        };
    }
    if x >= f32::from_bits(0x42b1_7218) {
        return f32::INFINITY; // L__max_expf_arg
    }
    if x < f32::from_bits(0xc2d0_0000) {
        return 0.0; // -104: below this exp(x) f32-rounds to 0 (the asm's -103.277 would floor the denormal tail)
    }
    let xd = x as f64;
    let z = xd * f64::from_bits(0x4057_1547_652b_82fe); // x * 64/ln2
    let n = z.round_ties_even() as i32; // vcvtpd2dq (round-half-even)
    let r = (n as f64).mul_add(f64::from_bits(0xbf86_2e42_fefa_39ef), xd); // fma(n, -ln2/64, x)
    let half = r.mul_add(f64::from_bits(0x3fc5_5555_5555_5555), 0.5); // fma(r, 1/6, 1/2)
    let q = (r * r).mul_add(half, r); // fma(r², half, r)
    let f = f64::from_bits(TWO_TO_JBY64[(n & 0x3f) as usize]);
    let res = q.mul_add(f, f); // fma(q, f, f) = f·(1+q)
    let m = (n >> 6) as i64;
    f64::from_bits(res.to_bits().wrapping_add((m << 52) as u64)) as f32 // ·2^m, then vcvtpd2ps
}
#[rustfmt::skip]
static POW_LOG: [[u64; 3]; 128] = [
    [0x3ff6a00000000000, 0xbfd62c82f2b9c800, 0x3cfab42428375680],
    [0x3ff6800000000000, 0xbfd5d1bdbf580800, 0xbd1ca508d8e0f720],
    [0x3ff6600000000000, 0xbfd5767717455800, 0xbd2362a4d5b6506d],
    [0x3ff6400000000000, 0xbfd51aad872df800, 0xbce684e49eb067d5],
    [0x3ff6200000000000, 0xbfd4be5f95777800, 0xbd041b6993293ee0],
    [0x3ff6000000000000, 0xbfd4618bc21c6000, 0x3d13d82f484c84cc],
    [0x3ff5e00000000000, 0xbfd404308686a800, 0x3cdc42f3ed820b3a],
    [0x3ff5c00000000000, 0xbfd3a64c55694800, 0x3d20b1c686519460],
    [0x3ff5a00000000000, 0xbfd347dd9a988000, 0x3d25594dd4c58092],
    [0x3ff5800000000000, 0xbfd2e8e2bae12000, 0x3d267b1e99b72bd8],
    [0x3ff5600000000000, 0xbfd2895a13de8800, 0x3d15ca14b6cfb03f],
    [0x3ff5600000000000, 0xbfd2895a13de8800, 0x3d15ca14b6cfb03f],
    [0x3ff5400000000000, 0xbfd22941fbcf7800, 0xbd165a242853da76],
    [0x3ff5200000000000, 0xbfd1c898c1699800, 0xbd1fafbc68e75404],
    [0x3ff5000000000000, 0xbfd1675cababa800, 0x3d1f1fc63382a8f0],
    [0x3ff4e00000000000, 0xbfd1058bf9ae4800, 0xbd26a8c4fd055a66],
    [0x3ff4c00000000000, 0xbfd0a324e2739000, 0xbd0c6bee7ef4030e],
    [0x3ff4a00000000000, 0xbfd0402594b4d000, 0xbcf036b89ef42d7f],
    [0x3ff4a00000000000, 0xbfd0402594b4d000, 0xbcf036b89ef42d7f],
    [0x3ff4800000000000, 0xbfcfb9186d5e4000, 0x3d0d572aab993c87],
    [0x3ff4600000000000, 0xbfcef0adcbdc6000, 0x3d2b26b79c86af24],
    [0x3ff4400000000000, 0xbfce27076e2af000, 0xbd172f4f543fff10],
    [0x3ff4200000000000, 0xbfcd5c216b4fc000, 0x3d21ba91bbca681b],
    [0x3ff4000000000000, 0xbfcc8ff7c79aa000, 0x3d27794f689f8434],
    [0x3ff4000000000000, 0xbfcc8ff7c79aa000, 0x3d27794f689f8434],
    [0x3ff3e00000000000, 0xbfcbc286742d9000, 0x3d194eb0318bb78f],
    [0x3ff3c00000000000, 0xbfcaf3c94e80c000, 0x3cba4e633fcd9066],
    [0x3ff3a00000000000, 0xbfca23bc1fe2b000, 0xbd258c64dc46c1ea],
    [0x3ff3a00000000000, 0xbfca23bc1fe2b000, 0xbd258c64dc46c1ea],
    [0x3ff3800000000000, 0xbfc9525a9cf45000, 0xbd2ad1d904c1d4e3],
    [0x3ff3600000000000, 0xbfc87fa06520d000, 0x3d2bbdbf7fdbfa09],
    [0x3ff3400000000000, 0xbfc7ab890210e000, 0x3d2bdb9072534a58],
    [0x3ff3400000000000, 0xbfc7ab890210e000, 0x3d2bdb9072534a58],
    [0x3ff3200000000000, 0xbfc6d60fe719d000, 0xbd10e46aa3b2e266],
    [0x3ff3000000000000, 0xbfc5ff3070a79000, 0xbd1e9e439f105039],
    [0x3ff3000000000000, 0xbfc5ff3070a79000, 0xbd1e9e439f105039],
    [0x3ff2e00000000000, 0xbfc526e5e3a1b000, 0xbd20de8b90075b8f],
    [0x3ff2c00000000000, 0xbfc44d2b6ccb8000, 0x3d170cc16135783c],
    [0x3ff2c00000000000, 0xbfc44d2b6ccb8000, 0x3d170cc16135783c],
    [0x3ff2a00000000000, 0xbfc371fc201e9000, 0x3cf178864d27543a],
    [0x3ff2800000000000, 0xbfc29552f81ff000, 0xbd248d301771c408],
    [0x3ff2600000000000, 0xbfc1b72ad52f6000, 0xbd2e80a41811a396],
    [0x3ff2600000000000, 0xbfc1b72ad52f6000, 0xbd2e80a41811a396],
    [0x3ff2400000000000, 0xbfc0d77e7cd09000, 0x3d0a699688e85bf4],
    [0x3ff2400000000000, 0xbfc0d77e7cd09000, 0x3d0a699688e85bf4],
    [0x3ff2200000000000, 0xbfbfec9131dbe000, 0xbd2575545ca333f2],
    [0x3ff2000000000000, 0xbfbe27076e2b0000, 0x3d2a342c2af0003c],
    [0x3ff2000000000000, 0xbfbe27076e2b0000, 0x3d2a342c2af0003c],
    [0x3ff1e00000000000, 0xbfbc5e548f5bc000, 0xbd1d0c57585fbe06],
    [0x3ff1c00000000000, 0xbfba926d3a4ae000, 0x3d253935e85baac8],
    [0x3ff1c00000000000, 0xbfba926d3a4ae000, 0x3d253935e85baac8],
    [0x3ff1a00000000000, 0xbfb8c345d631a000, 0x3d137c294d2f5668],
    [0x3ff1a00000000000, 0xbfb8c345d631a000, 0x3d137c294d2f5668],
    [0x3ff1800000000000, 0xbfb6f0d28ae56000, 0xbd269737c93373da],
    [0x3ff1600000000000, 0xbfb51b073f062000, 0x3d1f025b61c65e57],
    [0x3ff1600000000000, 0xbfb51b073f062000, 0x3d1f025b61c65e57],
    [0x3ff1400000000000, 0xbfb341d7961be000, 0x3d2c5edaccf913df],
    [0x3ff1400000000000, 0xbfb341d7961be000, 0x3d2c5edaccf913df],
    [0x3ff1200000000000, 0xbfb16536eea38000, 0x3d147c5e768fa309],
    [0x3ff1000000000000, 0xbfaf0a30c0118000, 0x3d2d599e83368e91],
    [0x3ff1000000000000, 0xbfaf0a30c0118000, 0x3d2d599e83368e91],
    [0x3ff0e00000000000, 0xbfab42dd71198000, 0x3d1c827ae5d6704c],
    [0x3ff0e00000000000, 0xbfab42dd71198000, 0x3d1c827ae5d6704c],
    [0x3ff0c00000000000, 0xbfa77458f632c000, 0xbd2cfc4634f2a1ee],
    [0x3ff0c00000000000, 0xbfa77458f632c000, 0xbd2cfc4634f2a1ee],
    [0x3ff0a00000000000, 0xbfa39e87b9fec000, 0x3cf502b7f526feaa],
    [0x3ff0a00000000000, 0xbfa39e87b9fec000, 0x3cf502b7f526feaa],
    [0x3ff0800000000000, 0xbf9f829b0e780000, 0xbd2980267c7e09e4],
    [0x3ff0800000000000, 0xbf9f829b0e780000, 0xbd2980267c7e09e4],
    [0x3ff0600000000000, 0xbf97b91b07d58000, 0xbd288d5493faa639],
    [0x3ff0400000000000, 0xbf8fc0a8b0fc0000, 0xbcdf1e7cf6d3a69c],
    [0x3ff0400000000000, 0xbf8fc0a8b0fc0000, 0xbcdf1e7cf6d3a69c],
    [0x3ff0200000000000, 0xbf7fe02a6b100000, 0xbd19e23f0dda40e4],
    [0x3ff0200000000000, 0xbf7fe02a6b100000, 0xbd19e23f0dda40e4],
    [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
    [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
    [0x3fefc00000000000, 0x3f80101575890000, 0xbd10c76b999d2be8],
    [0x3fef800000000000, 0x3f90205658938000, 0xbd23dc5b06e2f7d2],
    [0x3fef400000000000, 0x3f98492528c90000, 0xbd2aa0ba325a0c34],
    [0x3fef000000000000, 0x3fa0415d89e74000, 0x3d0111c05cf1d753],
    [0x3feec00000000000, 0x3fa466aed42e0000, 0xbd2c167375bdfd28],
    [0x3fee800000000000, 0x3fa894aa149fc000, 0xbd197995d05a267d],
    [0x3fee400000000000, 0x3faccb73cdddc000, 0xbd1a68f247d82807],
    [0x3fee200000000000, 0x3faeea31c006c000, 0xbd0e113e4fc93b7b],
    [0x3fede00000000000, 0x3fb1973bd1466000, 0xbd25325d560d9e9b],
    [0x3feda00000000000, 0x3fb3bdf5a7d1e000, 0x3d2cc85ea5db4ed7],
    [0x3fed600000000000, 0x3fb5e95a4d97a000, 0xbd2c69063c5d1d1e],
    [0x3fed400000000000, 0x3fb700d30aeac000, 0x3cec1e8da99ded32],
    [0x3fed000000000000, 0x3fb9335e5d594000, 0x3d23115c3abd47da],
    [0x3fecc00000000000, 0x3fbb6ac88dad6000, 0xbd1390802bf768e5],
    [0x3feca00000000000, 0x3fbc885801bc4000, 0x3d2646d1c65aacd3],
    [0x3fec600000000000, 0x3fbec739830a2000, 0xbd2dc068afe645e0],
    [0x3fec400000000000, 0x3fbfe89139dbe000, 0xbd2534d64fa10afd],
    [0x3fec000000000000, 0x3fc1178e8227e000, 0x3d21ef78ce2d07f2],
    [0x3febe00000000000, 0x3fc1aa2b7e23f000, 0x3d2ca78e44389934],
    [0x3feba00000000000, 0x3fc2d1610c868000, 0x3d039d6ccb81b4a1],
    [0x3feb800000000000, 0x3fc365fcb0159000, 0x3cc62fa8234b7289],
    [0x3feb400000000000, 0x3fc4913d8333b000, 0x3d25837954fdb678],
    [0x3feb200000000000, 0x3fc527e5e4a1b000, 0x3d2633e8e5697dc7],
    [0x3feae00000000000, 0x3fc6574ebe8c1000, 0x3d19cf8b2c3c2e78],
    [0x3feac00000000000, 0x3fc6f0128b757000, 0xbd25118de59c21e1],
    [0x3feaa00000000000, 0x3fc7898d85445000, 0xbd1c661070914305],
    [0x3fea600000000000, 0x3fc8beafeb390000, 0xbd073d54aae92cd1],
    [0x3fea400000000000, 0x3fc95a5adcf70000, 0x3d07f22858a0ff6f],
    [0x3fea000000000000, 0x3fca93ed3c8ae000, 0xbd28724350562169],
    [0x3fe9e00000000000, 0x3fcb31d8575bd000, 0xbd0c358d4eace1aa],
    [0x3fe9c00000000000, 0x3fcbd087383be000, 0xbd2d4bc4595412b6],
    [0x3fe9a00000000000, 0x3fcc6ffbc6f01000, 0xbcf1ec72c5962bd2],
    [0x3fe9600000000000, 0x3fcdb13db0d49000, 0xbd2aff2af715b035],
    [0x3fe9400000000000, 0x3fce530effe71000, 0x3cc212276041f430],
    [0x3fe9200000000000, 0x3fcef5ade4dd0000, 0xbcca211565bb8e11],
    [0x3fe9000000000000, 0x3fcf991c6cb3b000, 0x3d1bcbecca0cdf30],
    [0x3fe8c00000000000, 0x3fd07138604d5800, 0x3cf89cdb16ed4e91],
    [0x3fe8a00000000000, 0x3fd0c42d67616000, 0x3d27188b163ceae9],
    [0x3fe8800000000000, 0x3fd1178e8227e800, 0xbd2c210e63a5f01c],
    [0x3fe8600000000000, 0x3fd16b5ccbacf800, 0x3d2b9acdf7a51681],
    [0x3fe8400000000000, 0x3fd1bf99635a6800, 0x3d2ca6ed5147bdb7],
    [0x3fe8200000000000, 0x3fd214456d0eb800, 0x3d0a87deba46baea],
    [0x3fe7e00000000000, 0x3fd2bef07cdc9000, 0x3d2a9cfa4a5004f4],
    [0x3fe7c00000000000, 0x3fd314f1e1d36000, 0xbd28e27ad3213cb8],
    [0x3fe7a00000000000, 0x3fd36b6776be1000, 0x3d116ecdb0f177c8],
    [0x3fe7800000000000, 0x3fd3c25277333000, 0x3d183b54b606bd5c],
    [0x3fe7600000000000, 0x3fd419b423d5e800, 0x3d08e436ec90e09d],
    [0x3fe7400000000000, 0x3fd4718dc271c800, 0xbd2f27ce0967d675],
    [0x3fe7200000000000, 0x3fd4c9e09e173000, 0xbd2e20891b0ad8a4],
    [0x3fe7000000000000, 0x3fd522ae0738a000, 0x3d2ebe708164c759],
    [0x3fe6e00000000000, 0x3fd57bf753c8d000, 0x3d1fadedee5d40ef],
    [0x3fe6c00000000000, 0x3fd5d5bddf596000, 0xbd0a0b2a08a465dc],
];
#[rustfmt::skip]
static EXP_TAB: [u64; 512] = [
    0x0000000000000000, 0x3ff0000000000000, 0xbc84e82fc61851ac, 0x3feffb1afa5abcbf,
    0x3c9b3b4f1a88bf6e, 0x3feff63da9fb3335, 0xbc82985dd8521d32, 0x3feff168143b0281,
    0xbc7160139cd8dc5d, 0x3fefec9a3e778061, 0x3c651e617061bfbd, 0x3fefe7d42e11bbcc,
    0xbc905e7a108766d1, 0x3fefe315e86e7f85, 0x3c845fad437fa426, 0x3fefde5f72f654b1,
    0x3c8cd2523567f613, 0x3fefd9b0d3158574, 0xbc954529642b232f, 0x3fefd50a0e3c1f89,
    0xbc8bce8023f98efa, 0x3fefd06b29ddf6de, 0x3c8293708ef5c32e, 0x3fefcbd42b72a836,
    0x3c60f74e61e6c861, 0x3fefc74518759bc8, 0xbc95b9280905b2a4, 0x3fefc2bdf66607e0,
    0x3c90a3e45b33d399, 0x3fefbe3ecac6f383, 0x3c84f31f32c4b7e7, 0x3fefb9c79b1f3919,
    0x3c979aa65d837b6d, 0x3fefb5586cf9890f, 0x3c9407fb30d06420, 0x3fefb0f145e46c85,
    0x3c8eb51a92fdeffc, 0x3fefac922b7247f7, 0xbc9a5d04b3b9911b, 0x3fefa83b23395dec,
    0x3c3ebe3d702f9cd1, 0x3fefa3ec32d3d1a2, 0xbc937a01f0739546, 0x3fef9fa55fdfa9c5,
    0xbc6a033489906e0b, 0x3fef9b66affed31b, 0x3c8b8268b04ef0a5, 0x3fef973028d7233e,
    0xbc9556522a2fbd0e, 0x3fef9301d0125b51, 0xbc9ac46e44a2ebcc, 0x3fef8edbab5e2ab6,
    0xbc5080ef8c4eea55, 0x3fef8abdc06c31cc, 0xbc65704e90c9f860, 0x3fef86a814f204ab,
    0xbc91c923b9d5f416, 0x3fef829aaea92de0, 0xbc897cea57e46280, 0x3fef7e95934f312e,
    0x3c80d3e3e95c55af, 0x3fef7a98c8a58e51, 0x3c56f01429e2b9d2, 0x3fef76a45471c3c2,
    0xbc801b15eaa59348, 0x3fef72b83c7d517b, 0x3c6e653b2459034b, 0x3fef6ed48695bbc0,
    0xbc8f1ff055de323d, 0x3fef6af9388c8dea, 0x3c92cc7ea345b7dc, 0x3fef672658375d2f,
    0x3c8b898c3f1353bf, 0x3fef635beb6fcb75, 0x3c957bfb2876ea9e, 0x3fef5f99f8138a1c,
    0xbc96d99c7611eb26, 0x3fef5be084045cd4, 0x3c8cdc1873af2155, 0x3fef582f95281c6b,
    0x3c9aecf73e3a2f60, 0x3fef54873168b9aa, 0xbc9493684653a131, 0x3fef50e75eb44027,
    0xbc8fe782cb86389d, 0x3fef4d5022fcd91d, 0xbc98e2899077520a, 0x3fef49c18438ce4d,
    0x3c8a6f4144a6c38d, 0x3fef463b88628cd6, 0x3c9120fcd4f59273, 0x3fef42be3578a819,
    0x3c807a05b0e4047d, 0x3fef3f49917ddc96, 0x3c89b788c188c9b8, 0x3fef3bdda27912d1,
    0x3c968efde3a8a894, 0x3fef387a6e756238, 0x3c877afbca90ef84, 0x3fef351ffb82140a,
    0x3c875e18f274487d, 0x3fef31ce4fb2a63f, 0x3c91512f082876ee, 0x3fef2e85711ece75,
    0x3c80472b981fe7f2, 0x3fef2b4565e27cdd, 0x3c9a02f0c7d75ec6, 0x3fef280e341ddf29,
    0xbc96b87b3f71085e, 0x3fef24dfe1f56381, 0xbc803297e78260bf, 0x3fef21ba7591bb70,
    0x3c82f7e16d09ab31, 0x3fef1e9df51fdee1, 0xbc95b77e5ccd9fbf, 0x3fef1b8a66d10f13,
    0xbc3d219b1a6fbffa, 0x3fef187fd0dad990, 0xbc91e75c40b4251e, 0x3fef157e39771b2f,
    0x3c8b3782720c0ab4, 0x3fef1285a6e4030b, 0x3c98a911f1f7785a, 0x3fef0f961f641589,
    0x3c6e149289cecb8f, 0x3fef0cafa93e2f56, 0xbc61e7c998db7dbb, 0x3fef09d24abd886b,
    0x3c834d754db0abb6, 0x3fef06fe0a31b715, 0x3c85425c11faadf4, 0x3fef0432edeeb2fd,
    0x3c864201e2ac744c, 0x3fef0170fc4cd831, 0xbc979517a03e2847, 0x3feefeb83ba8ea32,
    0x3c8fdd395dd3f84a, 0x3feefc08b26416ff, 0xbc800e2a46da4bee, 0x3feef96266e3fa2d,
    0xbc86a3803b8e5b04, 0x3feef6c55f929ff1, 0xbc87430803972b34, 0x3feef431a2de883b,
    0xbc924aedcc4b5068, 0x3feef1a7373aa9cb, 0xbc954de30ae02d94, 0x3feeef26231e754a,
    0xbc9907f81b512d8e, 0x3feeecae6d05d866, 0xbc94f2487e1c03ec, 0x3feeea401b7140ef,
    0xbc71d1e83e9436d2, 0x3feee7db34e59ff7, 0x3c914a5432fcb2f4, 0x3feee57fbfec6cf4,
    0xbc991919b3ce1b15, 0x3feee32dc313a8e5, 0x3c79c3bba5562a2f, 0x3feee0e544ede173,
    0x3c859f48a72a4c6d, 0x3feedea64c123422, 0xbc85a71612e21658, 0x3feedc70df1c5175,
    0xbc9312607a28698a, 0x3feeda4504ac801c, 0x3c86421f6f1d24d6, 0x3feed822c367a024,
    0xbc58a78f4817895b, 0x3feed60a21f72e2a, 0xbc9348a6815fce65, 0x3feed3fb2709468a,
    0xbc7c2c9b67499a1b, 0x3feed1f5d950a897, 0x3c835c43984d9871, 0x3feecffa3f84b9d4,
    0x3c4363ed60c2ac11, 0x3feece086061892d, 0xbc632afc8d9473a0, 0x3feecc2042a7d232,
    0x3c9666093b0664ef, 0x3feeca41ed1d0057, 0xbc95fc5e44de020e, 0x3feec86d668b3237,
    0x3c6ecce1daa10379, 0x3feec6a2b5c13cd0, 0xbc7ea0148327c42f, 0x3feec4e1e192aed2,
    0x3c93ff8e3f0f1230, 0x3feec32af0d7d3de, 0xbc7a843ad1a88022, 0x3feec17dea6db7d7,
    0x3c7690cebb7aafb0, 0x3feebfdad5362a27, 0x3c892ca3bf144e63, 0x3feebe41b817c114,
    0x3c931dbdeb54e077, 0x3feebcb299fddd0d, 0xbc902c99b04aa8b0, 0x3feebb2d81d8abff,
    0xbc8f94340071a38e, 0x3feeb9b2769d2ca7, 0x3c73e34f67e67118, 0x3feeb8417f4531ee,
    0xbc87deccdc93a349, 0x3feeb6daa2cf6642, 0xbc75a3b1197ba0f0, 0x3feeb57de83f4eef,
    0xbc78dec6bd0f385f, 0x3feeb42b569d4f82, 0x3c81bd2888075068, 0x3feeb2e2f4f6ad27,
    0xbc861246ec7b5cf6, 0x3feeb1a4ca5d920f, 0xbc896be8ae89ef8f, 0x3feeb070dde910d2,
    0x3c93350518fdd78e, 0x3feeaf4736b527da, 0xbc88e6ac90348602, 0x3feeae27dbe2c4cf,
    0x3c7b98b72f8a9b05, 0x3feead12d497c7fd, 0xbc91af7f1365c3ac, 0x3feeac0827ff07cc,
    0x3c9063e1e21c5409, 0x3feeab07dd485429, 0xbc943a3540d1898a, 0x3feeaa11fba87a03,
    0x3c34c7855019c6ea, 0x3feea9268a5946b7, 0xbc951f58ddaa8090, 0x3feea84590998b93,
    0x3c9432e62b64c035, 0x3feea76f15ad2148, 0xbc82e1648e50a17c, 0x3feea6a320dceb71,
    0xbc8ce44a6199769f, 0x3feea5e1b976dc09, 0x3c95f30eda98a575, 0x3feea52ae6cdf6f4,
    0xbc8c33c53bef4da8, 0x3feea47eb03a5585, 0x3c917ecda8a72159, 0x3feea3dd1d1929fd,
    0xbc845378892be9ae, 0x3feea34634ccc320, 0xbc9345f3cee1ae6e, 0x3feea2b9febc8fb7,
    0xbc93cedd78565858, 0x3feea23882552225, 0xbc85c33fdf910406, 0x3feea1c1c70833f6,
    0x3c5710aa807e1964, 0x3feea155d44ca973, 0x3c81079ab5789604, 0x3feea0f4b19e9538,
    0xbc93b3efbf5e2228, 0x3feea09e667f3bcd, 0x3c727df161cd7778, 0x3feea052fa75173e,
    0xbc6a12ad8734b982, 0x3feea012750bdabf, 0x3c93f9924a05b767, 0x3fee9fdcddd47645,
    0xbc6367efb86da9ee, 0x3fee9fb23c651a2f, 0xbc87557939a8b5ef, 0x3fee9f9298593ae5,
    0xbc80dc3d54e08851, 0x3fee9f7df9519484, 0x3c51ed2f56fa9d1a, 0x3fee9f7466f42e87,
    0xbc781f647e5a3ecf, 0x3fee9f75e8ec5f74, 0xbc88e67a9006c909, 0x3fee9f8286ead08a,
    0xbc86ee4ac08b7db0, 0x3fee9f9a48a58174, 0x3c86597566977ac8, 0x3fee9fbd35d7cbfd,
    0xbc8619321e55e68a, 0x3fee9feb564267c9, 0x3c92c0b7028a5c3a, 0x3feea024b1ab6e09,
    0x3c909ccb5e09d4d3, 0x3feea0694fde5d3f, 0x3c8a30faf49cc78c, 0x3feea0b938ac1cf6,
    0xbc7b32dcb94da51d, 0x3feea11473eb0187, 0xbc92dad3519d7b5b, 0x3feea17b0976cfdb,
    0x3c94ecfd5467c06b, 0x3feea1ed0130c132, 0x3c87d51410fd15c2, 0x3feea26a62ff86f0,
    0x3c65ebe1abd66c55, 0x3feea2f336cf4e62, 0xbc760a3629969871, 0x3feea3878491c491,
    0xbc88a1c52fb3cf42, 0x3feea427543e1a12, 0x3c8b18c6e3fdef5d, 0x3feea4d2add106d9,
    0xbc9369b6f13b3734, 0x3feea589994cce13, 0x3c90ec1ddcb1390a, 0x3feea64c1eb941f7,
    0xbc805e843a19ff1e, 0x3feea71a4623c7ad, 0xbc522cea4f3afa1e, 0x3feea7f4179f5b21,
    0xbc94d450d872576e, 0x3feea8d99b4492ed, 0x3c7c88549b958471, 0x3feea9cad931a436,
    0x3c90ad675b0e8a00, 0x3feeaac7d98a6699, 0x3c931143962f7877, 0x3feeabd0a478580f,
    0x3c8db72fc1f0eab4, 0x3feeace5422aa0db, 0x3c93e9e96f112479, 0x3feeae05bad61778,
    0xbc65b6609cc5e7ff, 0x3feeaf3216b5448c, 0xbc8dac42a4a38df0, 0x3feeb06a5e0866d9,
    0x3c7bf68359f35f44, 0x3feeb1ae99157736, 0x3c8b99dd98b1ed84, 0x3feeb2fed0282c8a,
    0xbc93091fa71e3d83, 0x3feeb45b0b91ffc6, 0xbc7885ad50cbb750, 0x3feeb5c353aa2fe2,
    0xbc5da9b88b6c1e29, 0x3feeb737b0cdc5e5, 0xbc82d5e85f3e0301, 0x3feeb8b82b5f98e5,
    0xbc6c23f97c90b959, 0x3feeba44cbc8520f, 0xbc51669428996971, 0x3feebbdd9a7670b3,
    0xbc92434322f4f9aa, 0x3feebd829fde4e50, 0x3c71f2b2c1c4c014, 0x3feebf33e47a22a2,
    0xbc85ca6cd7668e4b, 0x3feec0f170ca07ba, 0xbc9294f304f166b6, 0x3feec2bb4d53fe0d,
    0x3c71affc2b91ce27, 0x3feec49182a3f090, 0xbc8a1e58414c07d3, 0x3feec674194bb8d5,
    0x3c6dd235e10a73bb, 0x3feec86319e32323, 0xbc79740b58a20091, 0x3feeca5e8d07f29e,
    0xbc87c50422622263, 0x3feecc667b5de565, 0x3c9165830a2b96c2, 0x3feece7aed8eb8bb,
    0x3c8b1c86e3e231d5, 0x3feed09bec4a2d33, 0xbc903d5cbe27874b, 0x3feed2c980460ad8,
    0xbc91bbd1d3bcbb15, 0x3feed503b23e255d, 0x3c5986178980fce0, 0x3feed74a8af46052,
    0x3c90cc319cee31d2, 0x3feed99e1330b358, 0xbc89472975b1f2a5, 0x3feedbfe53c12e59,
    0x3c8469846e735ab3, 0x3feede6b5579fdbf, 0x3c7d8157a34b7e7f, 0x3feee0e521356eba,
    0xbc82dfcd978e9db4, 0x3feee36bbfd3f37a, 0x3c8c8a4e231ebb7d, 0x3feee5ff3a3c2774,
    0x3c8c1a7792cb3387, 0x3feee89f995ad3ad, 0xbc888c8d11a142e5, 0x3feeeb4ce622f2ff,
    0xbc907b8f4ad1d9fa, 0x3feeee07298db666, 0x3c889c2ea41433c7, 0x3feef0ce6c9a8952,
    0xbc55c3d956dcaeba, 0x3feef3a2b84f15fb, 0xbc7274aedac8ff80, 0x3feef68415b749b1,
    0xbc90a40e3da6f640, 0x3feef9728de5593a, 0x3c85c620ce76df06, 0x3feefc6e29f1c52a,
    0xbc68d6f438ad9334, 0x3feeff76f2fb5e47, 0xbc8fda52e1b51e41, 0x3fef028cf22749e4,
    0xbc91eee26b588a35, 0x3fef05b030a1064a, 0xbc32141a7b3e2cd8, 0x3fef08e0b79a6f1f,
    0x3c74ffd70a5fddcd, 0x3fef0c1e904bc1d2, 0xbc302899507554e5, 0x3fef0f69c3f3a207,
    0xbc91bdfbfa9298ac, 0x3fef12c25bd71e09, 0xbc80dda2d4c0010c, 0x3fef16286141b33d,
    0x3c736eae30af0cb3, 0x3fef199bdd85529c, 0xbc8a007daadf8d68, 0x3fef1d1cd9fa652c,
    0x3c8ee3325c9ffd94, 0x3fef20ab5fffd07a, 0x3c836909391181d3, 0x3fef244778fafb22,
    0x3c84e08fd10959ac, 0x3fef27f12e57d14b, 0xbc811cd7dbdf9547, 0x3fef2ba88988c933,
    0x3c63cdaf384e1a67, 0x3fef2f6d9406e7b5, 0xbc7ac28b7bef6621, 0x3fef33405751c4db,
    0x3c676b2c6c921968, 0x3fef3720dcef9069, 0xbc7030587207b9e1, 0x3fef3b0f2e6d1675,
    0xbc808a1883ccb5d2, 0x3fef3f0b555dc3fa, 0xbc8cc734592af7fc, 0x3fef43155b5bab74,
    0xbc8fad5d3ffffa6f, 0x3fef472d4a07897c, 0x3c87752a44f587e8, 0x3fef4b532b08c968,
    0xbc900dae3875a949, 0x3fef4f87080d89f2, 0x3c85b66fefeef52e, 0x3fef53c8eacaa1d6,
    0x3c74a385a63d07a7, 0x3fef5818dcfba487, 0x3c5159d9d908a96e, 0x3fef5c76e862e6d3,
    0xbc82919e2040220f, 0x3fef60e316c98398, 0x3c8c254d16117a68, 0x3fef655d71ff6075,
    0x3c8e5a50d5c192ac, 0x3fef69e603db3285, 0xbc8d8c329fbd0e03, 0x3fef6e7cd63a8315,
    0x3c843a59ac016b4b, 0x3fef7321f301b460, 0xbc8ea6e6fbd5f2a6, 0x3fef77d5641c0658,
    0xbc82d52107b43e1f, 0x3fef7c97337b9b5f, 0xbc63e8e3eab2cbb4, 0x3fef81676b197d17,
    0xbc892ab93b470dc9, 0x3fef864614f5a129, 0xbc8b7966cd0d2cd9, 0x3fef8b333b16ee12,
    0x3c74b604603a88d3, 0x3fef902ee78b3ff6, 0xbc776caa4c2ff1cf, 0x3fef953924676d76,
    0x3c83c5ec519d7271, 0x3fef9a51fbc74c83, 0xbc81d5fc525d9940, 0x3fef9f7977cdb740,
    0xbc8ff7128fd391f0, 0x3fefa4afa2a490da, 0x3c855cd8aaea3d21, 0x3fefa9f4867cca6e,
    0xbc8dae98e223747d, 0x3fefaf482d8e67f1, 0x3c8269947c2bed4a, 0x3fefb4aaa2188510,
    0x3c8ec3bc41aa2008, 0x3fefba1bee615a27, 0xbc83b6137e9afe9e, 0x3fefbf9c1cb6412a,
    0x3c842b94c3a9eb32, 0x3fefc52b376bba97, 0xbc69fa74878ba7c7, 0x3fefcac948dd7274,
    0x3c8a64a931d185ee, 0x3fefd0765b6e4540, 0x3c901f3a75ee0efe, 0x3fefd632798844f8,
    0xbc8e37bae43be3ed, 0x3fefdbfdad9cbe14, 0xbc516a9ce6ed84fa, 0x3fefe1d802243c89,
    0x3c77893b4d91cd9d, 0x3fefe7c1819e90d8, 0xbc699c7db2effc76, 0x3fefedba3692d514,
    0x3c5305c14160cc89, 0x3feff3c22b8f71f1, 0x3c64b458677f9840, 0x3feff9d96b2a23d9,
];

// Still on std (deliberate): the `double` trig — their call sites move no captured pixel and the
// digest already matches; reproduced only when a scene or parity diff needs one. (`tanf` is now
// ported above — the exported `D2D1Tan`/`D2D1MakeSkewMatrix` return its value to callers.)

/// `atan(c/256)` for `c = 16..=256` (ucrtbase's atan2 reduction anchors), as raw f64 bits.
#[rustfmt::skip]
static ATAN_ANCHOR: [u64; 241] = [
    0x3faff55bb72cfde9, 0x3fb0f99ea71d52a6, 0x3fb1f86dbf082d58, 0x3fb2f719318a4a9a,
    0x3fb3f59f0e7c559d, 0x3fb4f3fd677292fb, 0x3fb5f2324fd2d7b2, 0x3fb6f03bdcea4b0c,
    0x3fb7ee182602f10e, 0x3fb8ebc54478fb28, 0x3fb9e94153cfdcf1, 0x3fbae68a71c722b8,
    0x3fbbe39ebe6f07c3, 0x3fbce07c5c3cca32, 0x3fbddd21701eba6e, 0x3fbed98c2190043a,
    0x3fbfd5ba9aac2f6d, 0x3fc068d584212b3d, 0x3fc0e6adccf40881, 0x3fc1646541060850,
    0x3fc1e1fafb043726, 0x3fc25f6e171a535c, 0x3fc2dcbdb2fba1ff, 0x3fc359e8edeb99a3,
    0x3fc3d6eee8c6626c, 0x3fc453cec6092a9e, 0x3fc4d087a9da4f17, 0x3fc54d18ba11570a,
    0x3fc5c9811e3ec269, 0x3fc645bfffb3aa73, 0x3fc6c1d4898933d8, 0x3fc73dbde8a7d201,
    0x3fc7b97b4bce5b02, 0x3fc8350be398ebc7, 0x3fc8b06ee2879c28, 0x3fc92ba37d050271,
    0x3fc9a6a8e96c8626, 0x3fca217e601081a5, 0x3fca9c231b403279, 0x3fcb1696574d780b,
    0x3fcb90d7529260a2, 0x3fcc0ae54d768466, 0x3fcc84bf8a742e6d, 0x3fccfe654e1d5395,
    0x3fcd77d5df205736, 0x3fcdf110864c9d9d, 0x3fce6a148e96ec4d, 0x3fcee2e1451d980c,
    0x3fcf5b75f92c80dd, 0x3fcfd3d1fc40dbe4, 0x3fd025fa510665b5, 0x3fd061eea03d6290,
    0x3fd09dc597d86362, 0x3fd0d97ee509acb3, 0x3fd1151a362431c9, 0x3fd150973a9ce546,
    0x3fd18bf5a30bf178, 0x3fd1c735212dd883, 0x3fd2025567e47c95, 0x3fd23d562b381041,
    0x3fd278372057ef45, 0x3fd2b2f7fd9b5fe2, 0x3fd2ed987a823cfe, 0x3fd328184fb58951,
    0x3fd362773707ebcb, 0x3fd39cb4eb76157b, 0x3fd3d6d129271134, 0x3fd410cbad6c7d32,
    0x3fd44aa436c2af09, 0x3fd4845a84d0c21b, 0x3fd4bdee586890e6, 0x3fd4f75f73869978,
    0x3fd530ad9951cd49, 0x3fd569d88e1b4cd7, 0x3fd5a2e0175e0f4e, 0x3fd5dbc3fbbe768d,
    0x3fd614840309cfe1, 0x3fd64d1ff635c1c5, 0x3fd685979f5fa6fd, 0x3fd6bdeac9cbd76c,
    0x3fd6f61941e4def0, 0x3fd72e22d53aa2a9, 0x3fd7660752817501, 0x3fd79dc6899118d1,
    0x3fd7d5604b63b3f7, 0x3fd80cd46a14b1d0, 0x3fd84422b8df95d7, 0x3fd87b4b0c1ebedb,
    0x3fd8b24d394a1b25, 0x3fd8e92916f5cde8, 0x3fd91fde7cd0c662, 0x3fd9566d43a34907,
    0x3fd98cd5454d6b18, 0x3fd9c3165cc58107, 0x3fd9f93066168001, 0x3fda2f233e5e530b,
    0x3fda64eec3cc23fc, 0x3fda9a92d59e98cf, 0x3fdad00f5422058b, 0x3fdb056420ae9343,
    0x3fdb3a911da65c6c, 0x3fdb6f962e737efb, 0x3fdba473378624a5, 0x3fdbd9281e528191,
    0x3fdc0db4c94ec9ef, 0x3fdc42191ff11eb6, 0x3fdc76550aad71f8, 0x3fdcaa6872f3631b,
    0x3fdcde53432c1350, 0x3fdd121566b7f2ad, 0x3fdd45aec9ec862b, 0x3fdd791f5a1226f4,
    0x3fddac670561bb4f, 0x3fdddf85bb026974, 0x3fde127b6b0744af, 0x3fde4548066cf51a,
    0x3fde77eb7f175a34, 0x3fdeaa65c7cf28c4, 0x3fdedcb6d43f8434, 0x3fdf0ede98f393cf,
    0x3fdf40dd0b541417, 0x3fdf72b221a4e495, 0x3fdfa45dd3029258, 0x3fdfd5e0175fdf83,
    0x3fe0039c73c1a40b, 0x3fe01c341e82422d, 0x3fe034b709250488, 0x3fe04d25314342e5,
    0x3fe0657e94db30cf, 0x3fe07dc3324e9b38, 0x3fe095f30861a58f, 0x3fe0ae0e1639866c,
    0x3fe0c6145b5b43da, 0x3fe0de05d7aa6f7c, 0x3fe0f5e28b67e295, 0x3fe10daa77307a0d,
    0x3fe1255d9bfbd2a8, 0x3fe13cfbfb1b056e, 0x3fe1548596376469, 0x3fe16bfa6f5137e1,
    0x3fe1835a88be7c13, 0x3fe19aa5e5299f99, 0x3fe1b1dc87904284, 0x3fe1c8fe7341f64f,
    0x3fe1e00babdefeb3, 0x3fe1f7043557138a, 0x3fe20de813e823b1, 0x3fe224b74c1d192a,
    0x3fe23b71e2cc9e6a, 0x3fe25217dd17e501, 0x3fe268a940696da6, 0x3fe27f261273d1b3,
    0x3fe2958e59308e30, 0x3fe2abe21aded073, 0x3fe2c2215e024465, 0x3fe2d84c2961e48b,
    0x3fe2ee628406cbca, 0x3fe30464753b090a, 0x3fe31a52048874be, 0x3fe3302b39b78856,
    0x3fe345f01cce37bb, 0x3fe35ba0b60eccce, 0x3fe3713d0df6c503, 0x3fe386c52d3db11e,
    0x3fe39c391cd41719, 0x3fe3b198e5e2564a, 0x3fe3c6e491c78dc4, 0x3fe3dc1c2a188504,
    0x3fe3f13fb89e96f4, 0x3fe4064f47569f48, 0x3fe41b4ae06fea41, 0x3fe430328e4b26d5,
    0x3fe445065b795b55, 0x3fe459c652badc7f, 0x3fe46e727efe4715, 0x3fe4830aeb5f7bfd,
    0x3fe4978fa3269ee1, 0x3fe4ac00b1c71762, 0x3fe4c05e22de94e4, 0x3fe4d4a8023414e8,
    0x3fe4e8de5bb6ec04, 0x3fe4fd013b7dd17e, 0x3fe51110adc5ed81, 0x3fe5250cbef1e9fa,
    0x3fe538f57b89061e, 0x3fe54ccaf0362c8f, 0x3fe5608d29c70c34, 0x3fe5743c352b33b9,
    0x3fe587d81f732fba, 0x3fe59b60f5cfab9d, 0x3fe5aed6c5909517, 0x3fe5c2399c244260,
    0x3fe5d58987169b18, 0x3fe5e8c6941043cf, 0x3fe5fbf0d0d5cc49, 0x3fe60f084b46e05e,
    0x3fe6220d115d7b8d, 0x3fe634ff312d1f3b, 0x3fe647deb8e20b8f, 0x3fe65aabb6c07b02,
    0x3fe66d663923e086, 0x3fe6800e4e7e2857, 0x3fe692a40556fb6a, 0x3fe6a5276c4b0575,
    0x3fe6b798920b3d98, 0x3fe6c9f7855c3198, 0x3fe6dc44551553ae, 0x3fe6ee7f10204aef,
    0x3fe700a7c5784633, 0x3fe712be84295198, 0x3fe724c35b4fae7b, 0x3fe736b65a172dff,
    0x3fe748978fba8e0f, 0x3fe75a670b82d8d8, 0x3fe76c24dcc6c6c0, 0x3fe77dd112ea22c7,
    0x3fe78f6bbd5d315e, 0x3fe7a0f4eb9c19a2, 0x3fe7b26cad2e50fd, 0x3fe7c3d311a6092b,
    0x3fe7d528289fa093, 0x3fe7e66c01c114fd, 0x3fe7f79eacb97898, 0x3fe808c03940694a,
    0x3fe819d0b7158a4c, 0x3fe82ad036000005, 0x3fe83bbec5cdee22, 0x3fe84c9c7653f7ea,
    0x3fe85d69576cc2c5, 0x3fe86e2578f87ae5, 0x3fe87ed0eadc5a2a, 0x3fe88f6bbd023118,
    0x3fe89ff5ff57f1f7, 0x3fe8b06fc1cf3dfe, 0x3fe8c0d9145cf49d, 0x3fe8d13206f8c4ca,
    0x3fe8e17aa99cc05d, 0x3fe8f1b30c44f167, 0x3fe901db3eeef187, 0x3fe911f35199833b,
    0x3fe921fb54442d18,
];

#[cfg(test)]
mod tests {
    use super::*;

    // crt matches ucrtbase, not libm, so the check is accuracy (within a couple of ULP of the true
    // value computed in f64) plus the exact edge values — enough to catch a wrong branch or constant.
    fn close(a: f32, truth: f64) {
        let ulp = (truth.abs() as f32).max(1.0) * f32::EPSILON * 2.0;
        assert!((a as f64 - truth).abs() as f32 <= ulp, "got {a}, want ~{truth}");
    }

    #[test]
    fn sincos_accurate() {
        assert_eq!(sinf(0.0), 0.0);
        assert_eq!(cosf(0.0), 1.0);
        let mut x = -20.0f32;
        while x < 20.0 {
            close(sinf(x), (x as f64).sin());
            close(cosf(x), (x as f64).cos());
            x += 0.013;
        }
    }

    // `tanf` is the ucrtbase port; this is the accuracy sanity
    // check — full bit-exact verification over all 2³² inputs belongs in a trig-verify-style harness.
    #[test]
    fn tan_accurate() {
        assert_eq!(tanf(0.0), 0.0);
        // Sweep the open interval between the ±π/2 poles, plus larger angles in other quadrants.
        let mut x = -1.5f32;
        while x < 1.5 {
            close(tanf(x), (x as f64).tan());
            x += 0.013;
        }
        for x in [2.0f32, -2.0, 3.0, -3.1, 6.5, 10.0, 0.785_398_1, -0.785_398_1] {
            close(tanf(x), (x as f64).tan());
        }
    }

    #[test]
    fn acos_edges_and_accuracy() {
        assert_eq!(acosf(1.0).to_bits(), 0);
        assert_eq!(acosf(-1.0), core::f32::consts::PI);
        assert_eq!(acosf(0.0), core::f32::consts::FRAC_PI_2);
        for x in [2.0, -2.0, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(acosf(x).to_bits(), 0xffc0_0000);
        }
        assert_eq!(acosf(f32::from_bits(0xff81_2345)).to_bits(), 0xffc1_2345);
        for i in -1000..=1000 {
            let x = i as f32 / 1000.0;
            close(acosf(x), (x as f64).acos());
        }
    }

    #[test]
    fn atan2_octants() {
        // all four quadrants + the axes
        for &(y, x) in &[(1.0f32, 1.0f32), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0), (0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0), (0.3, -0.9), (0.9, 0.2)] {
            close(atan2f(y, x), (y as f64).atan2(x as f64));
        }
    }

    // Non-finite inputs must not panic (fuzz 30027 indexed ATAN_ANCHOR out of bounds via an inf/inf
    // ratio); NaN → NaN, infinities fold to the C99 unit-direction angle.
    #[test]
    fn atan2_non_finite() {
        let inf = f32::INFINITY;
        assert!(atan2f(f32::NAN, 1.0).is_nan());
        assert!(atan2f(1.0, f32::NAN).is_nan());
        close(atan2f(inf, inf), std::f64::consts::FRAC_PI_4); // π/4
        close(atan2f(inf, -inf), 3.0 * std::f64::consts::FRAC_PI_4); // 3π/4
        close(atan2f(inf, 5.0), std::f64::consts::FRAC_PI_2); // ±inf over finite → ±π/2
        close(atan2f(5.0, inf), 0.0); // finite over +inf → 0
        close(atan2f(5.0, -inf), std::f64::consts::PI); // finite over -inf → π
    }

    #[test]
    fn asin_domain() {
        assert_eq!(asinf(0.0), 0.0);
        assert_eq!(asinf(1.0).to_bits(), 0x3fc9_0fdb); // π/2
        assert_eq!(asinf(-1.0).to_bits(), 0xbfc9_0fdb); // −π/2
        assert!(asinf(1.5).is_nan()); // out of domain
        let mut x = -1.0f32;
        while x <= 1.0 {
            close(asinf(x), (x as f64).asin()); // crosses the 0.5 poly/sqrt split
            x += 0.013;
        }
    }
}
