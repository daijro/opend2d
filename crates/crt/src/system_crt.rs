//! Windows' system C runtime, where its maths differ from the universal C runtime's.
//!
//! Direct2D calls the C runtime that Windows' own components link, not the universal one .NET and most applications
//! call; the two are separate builds. The system runtime's maths are older AMD-style algorithms: `expf` table-driven
//! in f32; `sinf`, `cosf` and `tanf` double-precision kernels on [−π/4, π/4] with a three-piece π/2 reduction and a
//! 36-bit-chunk one for large arguments; `acosf` a single-precision rational; `atan2f` double with `atan(c/256)`
//! anchors; `powf` table-driven log and exp in double. Each reproduces the system runtime of Windows 11 26100 on the
//! reference VM and is verified bit-exact against it (`harness/src/bin/crt-verify.rs RUNTIME FUNC system`): every
//! f32 input for the one-argument functions; for `atan2f` 2^26 and for `powf` 2^28 pseudo-random pairs, plus `powf`
//! over every x in [0, 2] at thirteen gamma exponents. `logf` reuses `powf`'s log and is one ulp off at 22 inputs.

/// 2^(j/32) as `LEAD[j] + TRAIL[j]`: a lead with its low mantissa bits clear, and the rest.
const LEAD: [u32; 32] = [
    0x3f800000, 0x3f82c000, 0x3f858000, 0x3f888000, 0x3f8b8000, 0x3f8e8000, 0x3f91c000, 0x3f94c000, 0x3f980000, 0x3f9b8000,
    0x3f9ec000, 0x3fa24000, 0x3fa5c000, 0x3fa98000, 0x3fad4000, 0x3fb10000, 0x3fb50000, 0x3fb8c000, 0x3fbd0000, 0x3fc10000,
    0x3fc54000, 0x3fc98000, 0x3fce0000, 0x3fd28000, 0x3fd74000, 0x3fdbc000, 0x3fe0c000, 0x3fe58000, 0x3feac000, 0x3fefc000,
    0x3ff50000, 0x3ffa8000,
];
const TRAIL: [u32; 32] = [
    0x00000000, 0x39d86988, 0x3aab0d9f, 0x3a407404, 0x3a2e0f1e, 0x3a90e62d, 0x38f4dce0, 0x3ad3bea3, 0x3adfc146, 0x39d39b9c,
    0x3ad4c982, 0x3ac10c0c, 0x3afb5aa6, 0x3a856ad3, 0x3a41f752, 0x3a8fd607, 0x391e6678, 0x3aeebd1d, 0x398a39f4, 0x3ab13329,
    0x3a9ca845, 0x3ae6f619, 0x3a923054, 0x3aa07647, 0x391f9958, 0x3aeede5f, 0x39cdeec0, 0x3ae41b9d, 0x37c6e7c0, 0x3a92e66f,
    0x3a95f454, 0x38ecb6d0,
];

/// `expf` (the system runtime's). Plain f32 arithmetic, so the same bits on every target:
/// - `|x| < 2⁻²⁶`: `x + 1`; `|x| < 2⁻⁶`: the degree-7 Taylor polynomial in Horner form;
/// - otherwise `n` = `x·32/ln2` rounded half away from zero, `r = (x − n·L₁) + (−n·L₂)` with ln2/32 split into
///   `L₁ + L₂`, and `2^(j/32)·(1 + r + r²/2 + r³/6)` as `lead + (p·(trail + lead) + trail)`, `j = n mod 32`;
/// - scaled by `2^((n − j)/32)`, in two steps when that leaves the normal range (subnormal results);
/// - NaN quietened, +∞ kept, −∞ to 0; `x ≥ 88.722839` overflows to +∞, `x < −103.27893` underflows to 0.
pub fn expf(x: f32) -> f32 {
    let f = f32::from_bits;
    let bits = x.to_bits();
    let ax = bits & 0x7fff_ffff;
    if ax < 0x42b1_7218 {
        if ax < 0x3c80_0000 {
            if ax < 0x3280_0000 {
                return x + 1.0;
            }
            let mut p = x * f(0x3950_0d01) + f(0x3ab6_0b61);
            for c in [0x3c08_8889, 0x3d2a_aaab, 0x3e2a_aaab, 0x3f00_0000, 0x3f80_0000] {
                p = p * x + f(c);
            }
            return p * x + 1.0;
        }
    } else {
        if ax >= 0x7f80_0000 {
            if bits & 0x007f_ffff != 0 {
                return f(bits | 0x0040_0000);
            }
            return if bits >> 31 == 0 { x } else { 0.0 };
        }
        if x >= f(0x42b1_7218) {
            return f32::INFINITY;
        }
        if x < f(0xc2ce_8ed0) {
            return 0.0;
        }
    }
    let t = x * f(0x4238_aa3b);
    let n = (if t > 0.0 { t + 0.5 } else { t - 0.5 }) as i32;
    let j = (n & 31) as usize;
    let r1 = x - n as f32 * f(0x3cb1_7000);
    let r2 = (-n) as f32 * f(0x3585_fdf4);
    let m = (n - j as i32) >> 5;
    let r = r2 + r1;
    let p = (r * f(0x3e2a_aaab) + 0.5) * (r * r) + r2 + r1;
    let (lead, trail) = (f(LEAD[j]), f(TRAIL[j]));
    let v = lead + (p * (trail + lead) + trail);
    if (m + 126) as u32 > 0xfd {
        let k = (n - j as i32) / 64;
        return v * f(((k + 127) as u32) << 23) * f(((m - k + 127) as u32) << 23);
    }
    v * f(((m + 127) as u32) << 23)
}

// --- sinf / cosf ----------------------------------------------------------------------------------------------------
//
// The system runtime evaluates both in double: |x| ≤ π/4 directly (a cubic Taylor step below 2⁻⁷, x itself below
// 2⁻¹³), otherwise reduced to r ∈ [−π/4, π/4] and a quadrant, by subtracting n·π/2 in up to three pieces below 5·10⁵ and
// by a 36-bit-chunk reduction against 2/π above.

const fn d(bits: u64) -> f64 {
    f64::from_bits(bits)
}

/// sin r on [−π/4, π/4], in double: `((x²·S4 − S3)·x² + S2)·x² − S1`, times x³, plus x.
fn sin_piby4(x: f64) -> f64 {
    let x2 = x * x;
    let mut p = x2 * d(0x3ec6_dbe4_ad15_72d5) - d(0x3f2a_013a_88a3_7196);
    p = p * x2 + d(0x3f81_1110_df01_232d);
    p *= x2;
    let x3 = x2 * x;
    p -= d(0x3fc5_5555_5545_e87d);
    p * x3 + x
}

/// cos r on [−π/4, π/4], in double: `(((C4 − x²·C5)·x² − C3)·x² + C2)·x⁴ + (1 − x²/2)`.
fn cos_piby4(x: f64) -> f64 {
    let x2 = x * x;
    let mut p = d(0x3efa_015c_50a9_3b49) - x2 * d(0x3e92_5247_43cc_46b8);
    let x4 = x2 * x2;
    p *= x2;
    p -= d(0x3f56_c16b_f556_99d7);
    p *= x2;
    let h = x2 * 0.5;
    p += d(0x3fa5_5555_5550_2f31);
    p *= x4;
    p + (1.0 - h)
}

/// 2/π in 36-bit chunks, for [`remainder_piby2f`].
const PIBITS: [u64; 12] = [
    0x0, 0x145f, 0x306dc9c88, 0x2a53f84ea, 0xfa3ea69bb, 0x81b6c52b3, 0x278872083, 0xfca2c757b, 0xd778ac36e, 0x48dc74849,
    0xba5c00c92, 0x5dd413a32,
];

/// `|x|` (as double bits, `|x| ≥ 5·10⁵`) mod π/2: the f32 mantissa times 2/π in 36-bit chunks, the fraction rebuilt
/// as a double and scaled by π/2. Returns `(r, quadrant)`.
fn remainder_piby2f(ux: u64) -> (f64, u32) {
    const M36: u64 = 0xf_ffff_ffff;
    let xexp = ((ux >> 52) & 0x7ff) as i64 - 1023;
    let m = ((ux & 0x000f_ffff_e000_0000) | 0x0010_0000_0000_0000) >> 29;
    let (i, j) = ((xexp / 36) as usize, (xexp % 36) as u32);
    let mut res = [0u64; 4];
    let p3 = PIBITS[i + 3].wrapping_mul(m);
    res[2] = p3 & M36;
    let p2 = (p3 >> 36).wrapping_add(PIBITS[i + 2].wrapping_mul(m));
    res[1] = p2 & M36;
    let p0 = PIBITS[i].wrapping_mul(m);
    let p1 = (p2 >> 36).wrapping_add(PIBITS[i + 1].wrapping_mul(m));
    res[0] = p1 & M36;
    let top = p1 ^ ((p1 ^ (p0.wrapping_add(p1 >> 36) << 36)) & 0xffff_fff0_0000_0000);
    let t = ((top >> (35 - j)) & 7) as u32;
    let mask = (1u64 << (36 - j)) - 1;
    let odd = t & 1 != 0;
    let (mut f, mut k, region);
    k = 1usize;
    if odd {
        region = ((t >> 1) + 1) & 3;
        f = mask & !res[0];
        while f < 0x10000 {
            f = (f << 36) | (!res[k] & M36);
            k += 1;
        }
    } else {
        region = t >> 1;
        f = res[0] & mask;
        while f < 0x10000 {
            f = res[k] | (f << 36);
            k += 1;
        }
    }
    let next = if odd { !res[k] & M36 } else { res[k] };
    let mut shift = 0i64;
    while f < 0x4000_0000_0000 {
        f <<= 6;
        shift += 6;
    }
    while f < 0x0010_0000_0000_0000 {
        f <<= 1;
        shift += 1;
    }
    let e = j as i64 - 36 * k as i64 - shift + 52;
    let mant = (f | (next >> (36 - shift))) & 0x000f_ffff_ffff_ffff;
    let mut bits = mant | (((e + 1023) as u64) << 52);
    if odd {
        bits |= 0x8000_0000_0000_0000;
    }
    (f64::from_bits(bits) * d(0x3ff9_21fb_5444_2d18), region)
}

/// `dx = |x| > π/4` to `(r, quadrant)`.
fn reduce_piby2(dx: f64) -> (f64, u32) {
    let ax = dx.to_bits();
    if dx >= 5.0e5 {
        return remainder_piby2f(ax);
    }
    let n: i32 = if ax <= 0x400f_6a7a_2955_385e {
        1 + (ax > 0x4002_d97c_7f33_21d2) as i32
    } else if ax <= 0x401c_463a_becc_b2bb {
        3 + (ax > 0x4015_fdbb_e9bb_a775) as i32
    } else {
        (dx * d(0x3fe4_5f30_6dc9_c883) + 0.5) as i32
    };
    let nf = n as f64;
    let mut head = dx - nf * d(0x3ff9_21fb_5440_0000);
    let mut tail = nf * d(0x3dd0_b461_1a62_6331);
    let expdiff = (ax >> 52).wrapping_sub((head.to_bits() >> 52) & 0x7ff);
    if expdiff > 15 {
        let t = head;
        let piece = nf * d(0x3dd0_b461_1a60_0000);
        head = t - piece;
        tail = nf * d(0x3ba3_198a_2e03_7073) - ((t - head) - piece);
        if expdiff > 48 {
            let t = head;
            let piece = nf * d(0x3ba3_198a_2e00_0000);
            head = t - piece;
            tail = nf * d(0x397b_839a_2520_49c1) - ((t - head) - piece);
        }
    }
    (head - tail, (n & 3) as u32)
}

/// The NaN the system runtime returns for an infinite argument.
const DEFAULT_NAN: u32 = 0xffc0_0000;

/// `sinf` (the system runtime's).
pub fn sinf(x: f32) -> f32 {
    let dx = x as f64;
    let ax = dx.to_bits() & 0x7fff_ffff_ffff_ffff;
    if ax <= 0x3fe9_21fb_5444_2d18 {
        if ax < 0x3f80_0000_0000_0000 {
            if ax < 0x3f20_0000_0000_0000 {
                return x;
            }
            return (dx - dx * dx * dx * d(0x3fc5_5555_5555_5555)) as f32;
        }
        return sin_piby4(dx) as f32;
    }
    if ax & 0x7ff0_0000_0000_0000 == 0x7ff0_0000_0000_0000 {
        return if ax & 0x000f_ffff_ffff_ffff != 0 { f32::from_bits(x.to_bits() | 0x0040_0000) } else { f32::from_bits(DEFAULT_NAN) };
    }
    let neg = x.is_sign_negative();
    let (r, region) = reduce_piby2(dx.abs());
    let v = match (region, neg) {
        (0, false) | (2, true) => sin_piby4(r),
        (1, false) | (3, true) => cos_piby4(r),
        (2, false) | (0, true) => -sin_piby4(r),
        _ => -cos_piby4(r),
    };
    v as f32
}

/// `cosf` (the system runtime's).
pub fn cosf(x: f32) -> f32 {
    let dx = x as f64;
    let ax = dx.to_bits() & 0x7fff_ffff_ffff_ffff;
    if ax <= 0x3fe9_21fb_5444_2d18 {
        if ax < 0x3f80_0000_0000_0000 {
            if ax < 0x3f20_0000_0000_0000 {
                return 1.0;
            }
            return (1.0 - dx * dx * 0.5) as f32;
        }
        return cos_piby4(dx) as f32;
    }
    if ax & 0x7ff0_0000_0000_0000 == 0x7ff0_0000_0000_0000 {
        return if ax & 0x000f_ffff_ffff_ffff != 0 { f32::from_bits(x.to_bits() | 0x0040_0000) } else { f32::from_bits(DEFAULT_NAN) };
    }
    let (r, region) = reduce_piby2(dx.abs());
    let v = match region {
        0 => cos_piby4(r),
        1 => -sin_piby4(r),
        2 => -cos_piby4(r),
        _ => sin_piby4(r),
    };
    v as f32
}

/// tan r on [−π/4, π/4], in double: `(A − x²·B)·x³ / ((x²·C − E)·x² + F) + x`; `recip` gives −1/tan.
fn tan_piby4(x: f64, recip: bool) -> f64 {
    let x2 = x * x;
    let num = (d(0x3fd8_a8b0_da56_cb17) - x2 * d(0x3f91_9dba_6efd_6aad)) * (x2 * x);
    let den = (x2 * d(0x3f92_e290_03c6_92d9) - d(0x3fe0_7266_d7b3_511b)) * x2 + d(0x3ff2_7e84_a3e7_3a2e);
    let t = num / den + x;
    if recip {
        -1.0 / t
    } else {
        t
    }
}

/// `tanf` (the system runtime's).
pub fn tanf(x: f32) -> f32 {
    let dx = x as f64;
    let ax = dx.to_bits() & 0x7fff_ffff_ffff_ffff;
    if ax <= 0x3fe9_21fb_5444_2d18 {
        if ax < 0x3f80_0000_0000_0000 {
            if ax < 0x3f20_0000_0000_0000 {
                return x;
            }
            return (dx * dx * dx * d(0x3fd5_5555_5555_5555) + dx) as f32;
        }
        return tan_piby4(dx, false) as f32;
    }
    if ax & 0x7ff0_0000_0000_0000 == 0x7ff0_0000_0000_0000 {
        return if ax & 0x000f_ffff_ffff_ffff != 0 { f32::from_bits(x.to_bits() | 0x0040_0000) } else { f32::from_bits(DEFAULT_NAN) };
    }
    let (r, region) = reduce_piby2(dx.abs());
    let t = tan_piby4(r, region & 1 != 0);
    (if x.is_sign_negative() { -t } else { t }) as f32
}

/// `acosf` (the system runtime's): a single-precision rational approximation on [0, ½], reached through
/// `r = (1 − |x|)/2`, `s = √r` above ½, and the result rebuilt around π/2 or π in double (π/2's tail kept apart).
pub fn acosf(x: f32) -> f32 {
    let f = f32::from_bits;
    let ux = x.to_bits();
    let aux = ux & 0x7fff_ffff;
    let bexp = (ux >> 23) & 0xff;
    if aux > 0x7f80_0000 {
        return f(ux | 0x0040_0000);
    }
    if bexp < 0x65 {
        return f(0x3fc9_0fdb);
    }
    if bexp >= 0x7f {
        return if x == 1.0 {
            0.0
        } else if x == -1.0 {
            f(0x4049_0fdb)
        } else {
            f(DEFAULT_NAN)
        };
    }
    let neg = ux >> 31 != 0;
    let transform = bexp >= 0x7e;
    let mut y = f(aux);
    let (r, s) = if transform {
        let r = (1.0 - y) * 0.5;
        let s = r.sqrt();
        y = s;
        (r, s)
    } else {
        (y * y, 0.0)
    };
    let mut p = f(0xbc5b_3fe1) - r * f(0x3b81_ce6b);
    p = p * r - f(0x3d67_8bdd);
    p = p * r + f(0x3e3c_94dc);
    p *= r;
    let u = p / (f(0x3f8d_6fa5) - r * f(0x3f56_1f0d));
    let tail = d(0x3c91_a626_3314_5c07);
    if !transform {
        let xu = (x * u) as f64;
        return (d(0x3ff9_21fb_5444_2d18) - (x as f64 - (tail - xu))) as f32;
    }
    if neg {
        let t = ((y * u) as f64 - tail) + s as f64;
        return (d(0x4009_21fb_5444_2d18) - (t + t)) as f32;
    }
    let s1 = f(s.to_bits() & 0xffff_0000);
    let c = (r - s1 * s1) / (s1 + s);
    ((y + y) * u + (c + c)) + (s1 + s1)
}

/// `atan2f` (the system runtime's): in double, with the universal runtime's `atan(c/256)` anchors but plain (unfused)
/// arithmetic and its own polynomials. The quadrant comes from the signs (sign bits, so −0 counts), and the
/// exponent difference picks the shortcuts: ±π/2 when y dwarfs x, y/x (scaled exactly below the normal range) or ±0
/// when x dwarfs it on the positive side, ±π on the negative.
pub fn atan2f(y: f32, x: f32) -> f32 {
    const PI: f64 = d(0x4009_21fb_5444_2d18);
    const PI_2: f64 = d(0x3ff9_21fb_5444_2d18);
    let (yd, xd) = (y as f64, x as f64);
    let (ub, xb) = (yd.to_bits(), xd.to_bits());
    let (ay, ax) = (ub & 0x7fff_ffff_ffff_ffff, xb & 0x7fff_ffff_ffff_ffff);
    let (yneg, xneg) = (ub >> 63 != 0, xb >> 63 != 0);
    let expdiff = ((ub >> 52) & 0x7ff) as i32 - ((xb >> 52) & 0x7ff) as i32;
    let signed = |v: f64| if yneg { -v } else { v } as f32;
    if ax > 0x7ff0_0000_0000_0000 {
        return f32::from_bits(x.to_bits() | 0x0040_0000);
    }
    if ay > 0x7ff0_0000_0000_0000 {
        return f32::from_bits(y.to_bits() | 0x0040_0000);
    }
    if ay == 0 {
        return if xneg { signed(PI) } else { y };
    }
    if ax == 0 && yneg {
        return -PI_2 as f32;
    }
    if expdiff > 26 {
        return signed(PI_2);
    }
    if expdiff < -13 && !xneg {
        if expdiff < -150 {
            return if yneg { -0.0 } else { 0.0 };
        }
        if expdiff < -126 {
            return ((d(0x4630_0000_0000_0000) * yd) / xd * d(0x39b0_0000_0000_0000)) as f32;
        }
        return (yd / xd) as f32;
    }
    if expdiff < -26 && xneg {
        return signed(PI);
    }
    if ay == 0x7ff0_0000_0000_0000 && ax == 0x7ff0_0000_0000_0000 {
        return signed(if xneg { d(0x4002_d97c_7f33_21d2) } else { d(0x3fe9_21fb_5444_2d18) });
    }
    let (mut num, mut den) = (f64::from_bits(ay), f64::from_bits(ax));
    let swap = num > den;
    if swap {
        (num, den) = (den, num);
    }
    let r = num / den;
    let mut a = if r > 0.0625 {
        let c = (r * 256.0 + 0.5) as i32;
        let cf = c as f64;
        let z = (num * 256.0 - cf * den) / (cf * num + den * 256.0);
        let a = z + f64::from_bits(crate::ATAN_ANCHOR[(c - 16) as usize]);
        a - z * z * z * d(0x3fd5_5555_5555_0877)
    } else if r < d(0x3f1a_36e2_eb1c_432d) {
        r
    } else {
        let r2 = r * r;
        let p = (d(0x3fc9_9999_9996_43a3) - r2 * d(0x3fc2_4924_82bd_6be1)) * r2;
        r - (d(0x3fd5_5555_5555_5538) - p) * (r2 * r)
    };
    if swap {
        a = PI_2 - a;
    }
    if xneg {
        a = PI - a;
    }
    signed(a)
}

// --- powf -----------------------------------------------------------------------------------------------------------

/// ln(2·f₁) for the 129 f₁ = ½ + k/256 (k = 0..=128): [`powf`]'s log anchors.
const POW_LN: [u64; 129] = [
    0x0000000000000000, 0x3f7fe02a6b106789, 0x3f8fc0a8b0fc03e4, 0x3f97b91b07d5b11b, 0x3f9f829b0e783300, 0x3fa39e87b9febd60,
    0x3fa77458f632dcfc, 0x3fab42dd711971bf, 0x3faf0a30c01162a6, 0x3fb16536eea37ae1, 0x3fb341d7961bd1d1, 0x3fb51b073f06183f,
    0x3fb6f0d28ae56b4c, 0x3fb8c345d6319b21, 0x3fba926d3a4ad563, 0x3fbc5e548f5bc743, 0x3fbe27076e2af2e6, 0x3fbfec9131dbeabb,
    0x3fc0d77e7cd08e59, 0x3fc1b72ad52f67a0, 0x3fc29552f81ff523, 0x3fc371fc201e8f74, 0x3fc44d2b6ccb7d1e, 0x3fc526e5e3a1b438,
    0x3fc5ff3070a793d4, 0x3fc6d60fe719d21d, 0x3fc7ab890210d909, 0x3fc87fa06520c911, 0x3fc9525a9cf456b4, 0x3fca23bc1fe2b563,
    0x3fcaf3c94e80bff3, 0x3fcbc286742d8cd6, 0x3fcc8ff7c79a9a22, 0x3fcd5c216b4fbb91, 0x3fce27076e2af2e6, 0x3fcef0adcbdc5936,
    0x3fcfb9186d5e3e2b, 0x3fd0402594b4d041, 0x3fd0a324e27390e3, 0x3fd1058bf9ae4ad5, 0x3fd1675cababa60e, 0x3fd1c898c16999fb,
    0x3fd22941fbcf7966, 0x3fd2895a13de86a3, 0x3fd2e8e2bae11d31, 0x3fd347dd9a987d55, 0x3fd3a64c556945ea, 0x3fd404308686a7e4,
    0x3fd4618bc21c5ec2, 0x3fd4be5f957778a1, 0x3fd51aad872df82d, 0x3fd5767717455a6c, 0x3fd5d1bdbf5809ca, 0x3fd62c82f2b9c795,
    0x3fd686c81e9b14af, 0x3fd6e08eaa2ba1e4, 0x3fd739d7f6bbd007, 0x3fd792a55fdd47a2, 0x3fd7eaf83b82afc3, 0x3fd842d1da1e8b17,
    0x3fd89a3386c1425b, 0x3fd8f11e873662c8, 0x3fd947941c2116fb, 0x3fd99d958117e08b, 0x3fd9f323ecbf984c, 0x3fda484090e5bb0a,
    0x3fda9cec9a9a084a, 0x3fdaf1293247786b, 0x3fdb44f77bcc8f63, 0x3fdb9858969310fb, 0x3fdbeb4d9da71b7c, 0x3fdc3dd7a7cdad4d,
    0x3fdc8ff7c79a9a22, 0x3fdce1af0b85f3eb, 0x3fdd32fe7e00ebd5, 0x3fdd83e7258a2f3e, 0x3fddd46a04c1c4a1, 0x3fde24881a7c6c26,
    0x3fde744261d68788, 0x3fdec399d2468cc0, 0x3fdf128f5faf06ed, 0x3fdf6123fa7028ac, 0x3fdfaf588f78f31f, 0x3fdffd2e0857f498,
    0x3fe02552a5a5d0ff, 0x3fe04bdf9da926d2, 0x3fe0723e5c1cdf40, 0x3fe0986f4f573521, 0x3fe0be72e4252a83, 0x3fe0e44985d1cc8c,
    0x3fe109f39e2d4c97, 0x3fe12f719593efbc, 0x3fe154c3d2f4d5ea, 0x3fe179eabbd899a1, 0x3fe19ee6b467c96f, 0x3fe1c3b81f713c25,
    0x3fe1e85f5e7040d0, 0x3fe20cdcd192ab6e, 0x3fe23130d7bebf43, 0x3fe2555bce98f7cb, 0x3fe2795e1289b11b, 0x3fe29d37fec2b08b,
    0x3fe2c0e9ed448e8c, 0x3fe2e47436e40268, 0x3fe307d7334f10be, 0x3fe32b1339121d71, 0x3fe34e289d9ce1d3, 0x3fe37117b54747b6,
    0x3fe393e0d3562a1a, 0x3fe3b68449fffc23, 0x3fe3d9026a7156fb, 0x3fe3fb5b84d16f42, 0x3fe41d8fe84672ae, 0x3fe43f9fe2f9ce67,
    0x3fe4618bc21c5ec2, 0x3fe48353d1ea88df, 0x3fe4a4f85db03ebb, 0x3fe4c679afccee3a, 0x3fe4e7d811b75bb1, 0x3fe50913cc01686b,
    0x3fe52a2d265bc5ab, 0x3fe54b2467999498, 0x3fe56bf9d5b3f399, 0x3fe58cadb5cd7989, 0x3fe5ad404c359f2d, 0x3fe5cdb1dc6c1765,
    0x3fe5ee02a9241675, 0x3fe60e32f44788d9, 0x3fe62e42fefa39ef,
];
/// 1/f₁ for the same anchors.
const POW_RCP: [u64; 129] = [
    0x4000000000000000, 0x3fffc07f01fc07f0, 0x3fff81f81f81f820, 0x3fff44659e4a4271, 0x3fff07c1f07c1f08, 0x3ffecc07b301ecc0,
    0x3ffe9131abf0b767, 0x3ffe573ac901e574, 0x3ffe1e1e1e1e1e1e, 0x3ffde5d6e3f8868a, 0x3ffdae6076b981db, 0x3ffd77b654b82c34,
    0x3ffd41d41d41d41d, 0x3ffd0cb58f6ec074, 0x3ffcd85689039b0b, 0x3ffca4b3055ee191, 0x3ffc71c71c71c71c, 0x3ffc3f8f01c3f8f0,
    0x3ffc0e070381c0e0, 0x3ffbdd2b899406f7, 0x3ffbacf914c1bad0, 0x3ffb7d6c3dda338b, 0x3ffb4e81b4e81b4f, 0x3ffb2036406c80d9,
    0x3ffaf286bca1af28, 0x3ffac5701ac5701b, 0x3ffa98ef606a63be, 0x3ffa6d01a6d01a6d, 0x3ffa41a41a41a41a, 0x3ffa16d3f97a4b02,
    0x3ff9ec8e951033d9, 0x3ff9c2d14ee4a102, 0x3ff999999999999a, 0x3ff970e4f80cb872, 0x3ff948b0fcd6e9e0, 0x3ff920fb49d0e229,
    0x3ff8f9c18f9c18fa, 0x3ff8d3018d3018d3, 0x3ff8acb90f6bf3aa, 0x3ff886e5f0abb04a, 0x3ff8618618618618, 0x3ff83c977ab2bedd,
    0x3ff8181818181818, 0x3ff7f405fd017f40, 0x3ff7d05f417d05f4, 0x3ff7ad2208e0ecc3, 0x3ff78a4c8178a4c8, 0x3ff767dce434a9b1,
    0x3ff745d1745d1746, 0x3ff724287f46debc, 0x3ff702e05c0b8170, 0x3ff6e1f76b4337c7, 0x3ff6c16c16c16c17, 0x3ff6a13cd1537290,
    0x3ff6816816816817, 0x3ff661ec6a5122f9, 0x3ff642c8590b2164, 0x3ff623fa77016240, 0x3ff6058160581606, 0x3ff5e75bb8d015e7,
    0x3ff5c9882b931057, 0x3ff5ac056b015ac0, 0x3ff58ed2308158ed, 0x3ff571ed3c506b3a, 0x3ff5555555555555, 0x3ff5390948f40feb,
    0x3ff51d07eae2f815, 0x3ff5015015015015, 0x3ff4e5e0a72f0539, 0x3ff4cab88725af6e, 0x3ff4afd6a052bf5b, 0x3ff49539e3b2d067,
    0x3ff47ae147ae147b, 0x3ff460cbc7f5cf9a, 0x3ff446f86562d9fb, 0x3ff42d6625d51f87, 0x3ff4141414141414, 0x3ff3fb013fb013fb,
    0x3ff3e22cbce4a902, 0x3ff3c995a47babe7, 0x3ff3b13b13b13b14, 0x3ff3991c2c187f63, 0x3ff3813813813814, 0x3ff3698df3de0748,
    0x3ff3521cfb2b78c1, 0x3ff33ae45b57bcb2, 0x3ff323e34a2b10bf, 0x3ff30d190130d190, 0x3ff2f684bda12f68, 0x3ff2e025c04b8097,
    0x3ff2c9fb4d812ca0, 0x3ff2b404ad012b40, 0x3ff29e4129e4129e, 0x3ff288b01288b013, 0x3ff27350b8812735, 0x3ff25e22708092f1,
    0x3ff2492492492492, 0x3ff23456789abcdf, 0x3ff21fb78121fb78, 0x3ff20b470c67c0d9, 0x3ff1f7047dc11f70, 0x3ff1e2ef3b3fb874,
    0x3ff1cf06ada2811d, 0x3ff1bb4a4046ed29, 0x3ff1a7b9611a7b96, 0x3ff19453808ca29c, 0x3ff1811811811812, 0x3ff16e0689427379,
    0x3ff15b1e5f75270d, 0x3ff1485f0e0acd3b, 0x3ff135c81135c811, 0x3ff12358e75d3033, 0x3ff1111111111111, 0x3ff0fef010fef011,
    0x3ff0ecf56be69c90, 0x3ff0db20a88f4696, 0x3ff0c9714fbcda3b, 0x3ff0b7e6ec259dc8, 0x3ff0a6810a6810a7, 0x3ff0953f39010954,
    0x3ff0842108421084, 0x3ff073260a47f7c6, 0x3ff0624dd2f1a9fc, 0x3ff05197f7d73404, 0x3ff0410410410410, 0x3ff03091b51f5e1a,
    0x3ff0204081020408, 0x3ff0101010101010, 0x3ff0000000000000,
];
/// 2^(j/32), j = 0..32.
const POW_EXP2: [u64; 32] = [
    0x3ff0000000000000, 0x3ff059b0d3158574, 0x3ff0b5586cf9890f, 0x3ff11301d0125b51, 0x3ff172b83c7d517b, 0x3ff1d4873168b9aa,
    0x3ff2387a6e756238, 0x3ff29e9df51fdee1, 0x3ff306fe0a31b715, 0x3ff371a7373aa9cb, 0x3ff3dea64c123422, 0x3ff44e086061892d,
    0x3ff4bfdad5362a27, 0x3ff5342b569d4f82, 0x3ff5ab07dd485429, 0x3ff6247eb03a5585, 0x3ff6a09e667f3bcd, 0x3ff71f75e8ec5f74,
    0x3ff7a11473eb0187, 0x3ff82589994cce13, 0x3ff8ace5422aa0db, 0x3ff93737b0cdc5e5, 0x3ff9c49182a3f090, 0x3ffa5503b23e255d,
    0x3ffae89f995ad3ad, 0x3ffb7f76f2fb5e47, 0x3ffc199bdd85529c, 0x3ffcb720dcef9069, 0x3ffd5818dcfba487, 0x3ffdfc97337b9b5f,
    0x3ffea4afa2a490da, 0x3fff50765b6e4540,
];

/// `powf` (the system runtime's): ln x from a 129-entry table and a quartic, times y, then e^· from a 32-entry table
/// and a quartic, all in double, rounded to f32 once. Special operands follow the runtime, including its own choices:
/// a NaN x stays NaN even for y = 0 or 1; (−1)^−∞ = −1; |y| < 2⁻³⁴ gives 1 + y; a negative x with a non-integer y gives
/// the default NaN.
pub fn powf(x: f32, y: f32) -> f32 {
    const INF: u32 = 0x7f80_0000;
    let (ux, uy) = (x.to_bits(), y.to_bits());
    let is_nan = |u: u32| u & INF == INF && u & 0x007f_ffff != 0;
    let is_inf_or_nan = |u: u32| u & INF == INF;
    let f = f32::from_bits;
    // A NaN x wins unless y is a NaN too (then x's, or y's for the default NaN).
    let x_nan = |r11: u32| if is_nan(uy) && r11 == DEFAULT_NAN { f(uy) } else { f(r11) };
    if uy & 0x7fff_ffff == 0 {
        return if is_nan(ux) { x_nan(ux) } else { 1.0 };
    }
    if uy == 0x3f80_0000 {
        return if is_nan(ux) { x_nan(ux) } else { x };
    }
    let mut sign = 0u32;
    let very_large = |sign: u32| -> f32 {
        if is_inf_or_nan(ux) {
            return inf_or_nan_x(ux, uy, sign);
        }
        if ux & 0x7fff_ffff == 0 {
            return zero_x(uy, sign);
        }
        if ux == 0xbf80_0000 {
            return neg_one_x(uy, sign);
        }
        let neg_y = uy >> 31 != 0;
        let r = if (ux & 0x7fff_ffff) < 0x3f80_0000 { if neg_y { INF } else { 0 } } else if neg_y { 0 } else { INF };
        if is_nan(uy) {
            return f(uy);
        }
        f(r)
    };
    let ax = if ux >> 31 != 0 {
        if (uy & INF) > 0x4f00_0000 {
            return very_large(sign);
        }
        let ay = uy & 0x7fff_ffff;
        let e = (ay >> 23) as i32 - 127;
        let not_int = || -> f32 {
            if is_inf_or_nan(ux) {
                return inf_or_nan_x(ux, uy, sign);
            }
            if ux == 0x8000_0000 {
                return zero_x(uy, sign);
            }
            f(DEFAULT_NAN)
        };
        if e < 0 {
            return not_int();
        }
        if e <= 24 {
            if (0x007f_ffff >> e) & ay != 0 {
                return not_int();
            }
            if (0x0080_0000 >> e) & ay != 0 {
                sign = 0x8000_0000;
            }
        }
        if ux == 0x8000_0000 {
            return zero_x(uy, sign);
        }
        if ux == 0xbf80_0000 {
            return neg_one_x(uy, sign);
        }
        if is_inf_or_nan(ux) {
            return inf_or_nan_x(ux, uy, sign);
        }
        f(ux & 0x7fff_ffff)
    } else {
        if ux == 0x3f80_0000 {
            return 1.0;
        }
        if ux == 0 {
            return zero_x(uy, sign);
        }
        if is_inf_or_nan(ux) {
            return inf_or_nan_x(ux, uy, sign);
        }
        if (uy & INF) > 0x4f00_0000 {
            return very_large(sign);
        }
        if (uy & INF) < 0x2e80_0000 {
            return 1.0 + y;
        }
        x
    };
    let w = ln_table(ax) * y as f64;
    // e^w = 2^(m + j/32) · e^r, r = w − n·ln2/32.
    let t = d(0x4047_1547_652b_82fe) * w;
    if t >= 4096.0 || t.is_nan() {
        return f(INF | sign);
    }
    if t < d(0xc0b2_a000_0000_0000) {
        return f(sign);
    }
    let n = t.round_ties_even() as i32;
    let r = w - d(0x3f96_2e42_fefa_39ef) * n as f64;
    let j = (n & 31) as usize;
    let m = (n - j as i32) >> 5;
    let r3 = r * r * r;
    let q = (0.5 * r + 1.0) * r + (d(0x3fa5_5555_5555_5555) * r + d(0x3fc5_5555_5555_5555)) * r3;
    let tj = d(POW_EXP2[j]);
    let v = (q * tj + tj) * d(((m + 1023) as u64) << 52);
    f((v as f32).to_bits() | sign)
}

/// ln x of a positive finite x, in double: f ∈ [½, 1) and its 7-bit rounding f₁, u = (f₁ − f)/f₁,
/// ln f = ln f₁ − (u + u²/2 + u³/3 + u⁴/4) ([`powf`]'s log).
fn ln_table(ax: f32) -> f64 {
    let xd = ax as f64;
    let bits = xd.to_bits();
    let e = (((bits >> 52) as i64 - 1023) as i32) as f64;
    let fr = d((bits & 0x000f_ffff_ffff_ffff) | 0x3fe0_0000_0000_0000);
    let r8 = (bits & 0x000f_e000_0000_0000) + ((bits & 0x0000_1000_0000_0000) << 1);
    let k = (r8 >> 45) as usize;
    let u = (d(r8 | 0x3fe0_0000_0000_0000) - fr) * d(POW_RCP[k]);
    let u2 = u * u;
    let u3 = u2 * u;
    let p = (0.5 * u + 1.0) * u + (0.25 * u + d(0x3fd5_5555_5555_5555)) * u3;
    (d(POW_LN[k]) - p) + d(0x3fe6_2e42_fefa_39ef) * e
}

/// `logf` (the system runtime's): [`powf`]'s double ln, rounded to f32 once. 1 → +0; 0 → −∞; negative → the default
/// NaN; +∞ and NaN pass through. Not bit-exact everywhere: 22 of the 2^31 − 2^24 positive normal inputs differ from
/// the runtime by one ulp (`crt-verify system logf`), 5 of them in [0.01, 1], the range Direct2D takes it over.
pub fn logf(x: f32) -> f32 {
    let ux = x.to_bits();
    if ux == 0x3f80_0000 {
        return 0.0;
    }
    if ux & 0x7fff_ffff == 0 {
        return f32::NEG_INFINITY;
    }
    if ux >> 31 != 0 && !x.is_nan() {
        return f32::from_bits(DEFAULT_NAN);
    }
    if ux & 0x7f80_0000 == 0x7f80_0000 {
        return x;
    }
    ln_table(x) as f32
}


/// [`powf`] of an infinite or NaN x: ±∞ or 0 by y's sign (with the odd-integer `sign`), a NaN operand returned as is.
fn inf_or_nan_x(ux: u32, uy: u32, sign: u32) -> f32 {
    let r = if uy >> 31 == 0 { 0x7f80_0000 } else { 0 };
    if ux & 0x007f_ffff != 0 {
        return f32::from_bits(if uy & 0x7f80_0000 == 0x7f80_0000 && uy & 0x007f_ffff != 0 && ux == DEFAULT_NAN { uy } else { ux });
    }
    if uy & 0x7f80_0000 == 0x7f80_0000 && uy & 0x007f_ffff != 0 {
        return f32::from_bits(uy);
    }
    f32::from_bits(r | sign)
}

/// [`powf`] of a zero x: 0 for a positive y, ∞ for a negative one (signed for odd integers), and +0 for y = +∞.
fn zero_x(uy: u32, sign: u32) -> f32 {
    if uy & 0x7f80_0000 == 0x7f80_0000 {
        if uy == 0xff80_0000 {
            return f32::from_bits(0x7f80_0000 | sign);
        }
        if uy & 0x007f_ffff != 0 {
            return f32::from_bits(uy);
        }
        return 0.0;
    }
    f32::from_bits(if uy >> 31 != 0 { 0x7f80_0000 } else { 0 } | sign)
}

/// [`powf`] of x = −1: ±1 by the odd-integer `sign`, −1 for y = −∞, y itself when it is a NaN.
fn neg_one_x(uy: u32, sign: u32) -> f32 {
    if uy & 0x7f80_0000 == 0x7f80_0000 && uy & 0x007f_ffff != 0 {
        return f32::from_bits(uy);
    }
    f32::from_bits(if uy == 0xff80_0000 { 0xbf80_0000 } else { 0x3f80_0000 } | sign)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trig_and_pow_match_the_system_runtime_where_the_universal_one_differs() {
        // (input bits, the system runtime's result bits) read on the reference VM.
        let one = |f: fn(f32) -> f32, x: u32, want: u32| assert_eq!(f(f32::from_bits(x)).to_bits(), want, "{x:#x}");
        one(sinf, 0x3da7_4168, 0x3da7_11d4);
        one(cosf, 0x3ef8_ff88, 0x3f62_518a);
        one(tanf, 0x3ab4_adcc, 0x3ab4_add3);
        one(acosf, 0x3e1b_d5b4, 0x3fb5_81b0);
        assert_eq!(atan2f(f32::from_bits(0x4770_7d80), f32::from_bits(0x4e08_dcd2)).to_bits(), 0x38e0_eacf);
        assert_eq!(powf(f32::from_bits(0x0011_9e86), f32::from_bits(0x3ee8_ba2e)).to_bits(), 0x222c_120f);
        // The large-argument reduction.
        assert!((sinf(1.0e6) - 1.0e6f64.sin() as f32).abs() < 1e-6);
        assert!((cosf(3.0e7) - 3.0e7f64.cos() as f32).abs() < 1e-6);
    }

    #[test]
    fn expf_matches_the_system_runtime_where_the_universal_one_differs() {
        // (input bits, the system runtime's result bits) read on the reference VM; the second is where the universal
        // runtime gives one ulp less (the outer weight of a σ ≈ 7.2 Gaussian).
        for (x, want) in [(3229018591u32, 1017966124u32), (3230145649, 1013507374), (1065353216, 1076754516), (3212836864, 1052531378), (1118925335, 2139094916), (3263090384, 294187734)] {
            assert_eq!(expf(f32::from_bits(x)).to_bits(), want, "x = {x:#x}");
        }
        assert_eq!(expf(f32::NEG_INFINITY), 0.0);
        assert_eq!(expf(f32::INFINITY), f32::INFINITY);
        assert_eq!(expf(100.0), f32::INFINITY);
        assert_eq!(expf(-110.0), 0.0);
        assert!(expf(f32::NAN).is_nan());
    }
}
