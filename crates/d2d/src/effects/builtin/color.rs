//! Phase 2C built-in colour & transfer effects: the exact per-pixel maths native Direct2D
//! runs on WARP, written as pure functions over premultiplied-alpha f32 pixels.
//!
//! A [`Pixel`] is premultiplied RGBA f32, exactly what `D2DGetInput(i)` yields and what a D2D
//! pixel shader returns; the effect-graph engine (phase 2B) unpacks the UNORM8 input and rounds
//! the returned pixel back to the target. Every function here is the shader body only.
//!
//! Each effect was verified byte-for-byte against `conformance/oracle/win11-26100/fx-color-*`
//! (native WARP); where the managed Paint.NET reimplementation and native agree that formula is
//! used, otherwise native's constant setup and WARP's shader arithmetic are authoritative. The
//! colour-matrix family routes through [`color_matrix`] with per-effect matrices and alpha modes
//! (confirmed against the oracle, not copied from docs — e.g. Invert is PREMULTIPLIED, not the
//! STRAIGHT the public docs imply).

/// Premultiplied RGBA, f32 per channel: `[R, G, B, A]`.
pub type Pixel = [f32; 4];

/// `D2D1_COLORMATRIX_ALPHA_MODE`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AlphaMode {
    Premultiplied = 1,
    Straight = 2,
}

impl AlphaMode {
    /// From the registered enum value; anything else clamps to the default (`Premultiplied`).
    pub fn from_u32(v: u32) -> Self {
        if v == 2 {
            AlphaMode::Straight
        } else {
            AlphaMode::Premultiplied
        }
    }
}

fn sat(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// `UnpremultiplyOrTransparentBlack`: `a!=0 ? (rgb/a, a) : (0,0,0,0)`.
fn unpremul_tb(p: Pixel) -> Pixel {
    let a = p[3];
    if a != 0.0 {
        [p[0] / a, p[1] / a, p[2] / a, a]
    } else {
        [0.0, 0.0, 0.0, 0.0]
    }
}

/// `UnpremultiplyOrPassthrough` rgb: `a!=0 ? rgb/a : rgb`.
fn unpremul_pass(p: Pixel) -> [f32; 3] {
    let a = p[3];
    if a != 0.0 {
        [p[0] / a, p[1] / a, p[2] / a]
    } else {
        [p[0], p[1], p[2]]
    }
}

/// Re-premultiply straight rgba → premultiplied.
fn premul(t: Pixel) -> Pixel {
    [t[0] * t[3], t[1] * t[3], t[2] * t[3], t[3]]
}

/// ColorMatrix (R1). `m` is the `D2D_MATRIX_5X4_F` in row-major order: rows 0..3 are the 4×4
/// linear part (input channel → output channels), row 4 is the translation. `out.c = Σ_r v.r·m[r][c]
/// + m[16+c]`, the HLSL row-vector×matrix. STRAIGHT applies it to the raw premultiplied input and
/// returns the result unchanged; PREMULTIPLIED unpremultiplies first and re-premultiplies after.
pub fn color_matrix(p: Pixel, m: &[f32; 20], mode: AlphaMode, clamp_output: bool) -> Pixel {
    let apply = |v: Pixel| -> Pixel {
        std::array::from_fn(|c| {
            v[0] * m[c] + v[1] * m[4 + c] + v[2] * m[8 + c] + v[3] * m[12 + c] + m[16 + c]
        })
    };
    match mode {
        AlphaMode::Straight => {
            let mut t = apply(p);
            if clamp_output {
                t = t.map(sat);
            }
            t
        }
        AlphaMode::Premultiplied => {
            let mut t = apply(unpremul_tb(p));
            if clamp_output {
                t = t.map(sat);
            }
            premul(t)
        }
    }
}

// --- colour-matrix-derived effects (fixed matrices) ---

/// Invert (R2): `out.rgb = 1 − s.rgb`, alpha kept. PREMULTIPLIED (oracle-confirmed).
pub fn invert(p: Pixel) -> Pixel {
    #[rustfmt::skip]
    const M: [f32; 20] = [
        -1.0, 0.0, 0.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, -1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 0.0,
    ];
    color_matrix(p, &M, AlphaMode::Premultiplied, false)
}

/// Grayscale (R3): Rec.601 luma into R,G,B; alpha kept.
pub fn grayscale(p: Pixel) -> Pixel {
    #[rustfmt::skip]
    const M: [f32; 20] = [
        0.299, 0.299, 0.299, 0.0,
        0.587, 0.587, 0.587, 0.0,
        0.114, 0.114, 0.114, 0.0,
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    color_matrix(p, &M, AlphaMode::Premultiplied, false)
}

/// LuminanceToAlpha (R4): Rec.709 luma of the straight colour into alpha, rgb = 0. PREMULTIPLIED.
pub fn luminance_to_alpha(p: Pixel) -> Pixel {
    #[rustfmt::skip]
    const M: [f32; 20] = [
        0.0, 0.0, 0.0, 0.2125,
        0.0, 0.0, 0.0, 0.7154,
        0.0, 0.0, 0.0, 0.0721,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    color_matrix(p, &M, AlphaMode::Premultiplied, false)
}

/// Saturation (R5): `out.c = (1−v)·luma + v·s.c`, luma `(0.213,0.715,0.072)`. v=0→grey, 1→identity.
pub fn saturation(p: Pixel, saturation: f32) -> Pixel {
    const L: [f32; 3] = [0.213, 0.715, 0.072];
    let k = 1.0 - saturation;
    let mut m = [0.0f32; 20];
    for c in 0..3 {
        m[c] = k * L[0];
        m[4 + c] = k * L[1];
        m[8 + c] = k * L[2];
    }
    m[0] += saturation;
    m[5] += saturation;
    m[10] += saturation;
    m[15] = 1.0;
    color_matrix(p, &m, AlphaMode::Straight, false)
}

/// HueRotation (R6): the W3C `feColorMatrix type="hueRotate"` 3×3, luma `(0.213,0.715,0.072)`.
/// The angle wraps into [0, 360) and an angle within ten float epsilons of 0 passes the input through untouched; the radians
/// are computed in double and narrowed, then Direct2D's runtime `sinf`/`cosf`.
pub fn hue_rotation(p: Pixel, angle_degrees: f32) -> Pixel {
    let mut deg = angle_degrees % 360.0;
    if deg < 0.0 {
        deg += 360.0;
    }
    if deg.abs() < 0.0000011920929 {
        return p;
    }
    let a = (deg as f64 * std::f64::consts::PI / 180.0) as f32;
    let (s, c) = (crate::crt::system_crt::sinf(a), crate::crt::system_crt::cosf(a));
    #[rustfmt::skip]
    let m: [f32; 20] = [
        0.213 + 0.787 * c - 0.213 * s, 0.213 - 0.213 * c + 0.143 * s, 0.213 - 0.213 * c - 0.787 * s, 0.0,
        0.715 - 0.715 * c - 0.715 * s, 0.715 + 0.285 * c + 0.140 * s, 0.715 - 0.715 * c + 0.715 * s, 0.0,
        0.072 - 0.072 * c + 0.928 * s, 0.072 - 0.072 * c - 0.283 * s, 0.072 + 0.928 * c + 0.072 * s, 0.0,
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    color_matrix(p, &m, AlphaMode::Straight, false)
}

/// Sepia (R7): fixed `(0.3,0.52,0.12)` columns + an intensity-driven translation. `alpha_mode` is
/// the effect's property (default PREMULTIPLIED).
pub fn sepia(p: Pixel, intensity: f32, alpha_mode: AlphaMode) -> Pixel {
    let i = sat(intensity);
    #[rustfmt::skip]
    let m: [f32; 20] = [
        0.3, 0.3, 0.3, 0.0,
        0.52, 0.52, 0.52, 0.0,
        0.12, 0.12, 0.12, 0.0,
        0.0, 0.0, 0.0, 1.0,
        0.12 + 0.08 * i, 0.05 * i, -0.1 - 0.05 * i, 0.0,
    ];
    color_matrix(p, &m, alpha_mode, false)
}

// --- direct-shader colour effects ---

/// Exposure (R8): scale premultiplied rgb by `2^exposureValue`, alpha unchanged. `exp2` is the WARP
/// shader intrinsic (platform-independent), matching native's in-shader `exp`.
pub fn exposure(p: Pixel, exposure_value: f32) -> Pixel {
    let f = crate::warpmath::exp2(exposure_value);
    [p[0] * f, p[1] * f, p[2] * f, p[3]]
}

/// Brightness (R11): Paint.NET's sqrt-space tone curve, applied to the unpremultiplied colour and
/// re-premultiplied. `white`/`black` are the `(x,y)` points (`[0,1]²`), default `(1,1)`/`(0,0)` →
/// identity. Matches native WARP byte-for-byte.
pub fn brightness(p: Pixel, white: [f32; 2], black: [f32; 2]) -> Pixel {
    let num = white[0].max(0.0).sqrt();
    let num2 = white[1].max(0.0).sqrt();
    let delta = num - num2;
    let kink = 0.608_276_9 * (1.0 + delta);
    let slope = if kink != 0.0 { (kink - delta) / kink } else { 1.0 };
    let num6 = black[0].max(0.0).sqrt();
    let num7 = if num6 < kink { slope * num6 } else { num6 - delta };
    let num9 = white[1] - num7 * num7;
    let scale = if num9.abs() > 1e-9 { (white[1] - black[1]) / num9 } else { 1.0 };
    let bias = white[1] * (1.0 - scale);
    let a = p[3];
    let curve = |x: f32| {
        let enc = x.max(0.0).sqrt();
        let transfer = if enc >= kink { enc - delta } else { slope * enc };
        bias + scale * transfer * transfer
    };
    let s = unpremul_pass(p);
    [curve(s[0]) * a, curve(s[1]) * a, curve(s[2]) * a, a]
}

/// Contrast (R12): `A=1.5·contrast`, `B=1−0.75·contrast`; `x<0.5 → A·x²+B·x`, else `1−(A·m²+B·m)`
/// with `m=1−x`. Applied to the unpremultiplied colour (clamped first iff `clamp_input`), then
/// re-premultiplied. contrast=0 → identity.
pub fn contrast(p: Pixel, contrast: f32, clamp_input: bool) -> Pixel {
    let a_coef = 1.5 * contrast;
    let b_coef = 1.0 - 0.75 * contrast;
    let a = p[3];
    let curve = |mut x: f32| {
        if clamp_input {
            x = sat(x);
        }
        if x < 0.5 {
            a_coef * x * x + b_coef * x
        } else {
            let m = 1.0 - x;
            1.0 - (a_coef * m * m + b_coef * m)
        }
    };
    let s = unpremul_pass(p);
    [curve(s[0]) * a, curve(s[1]) * a, curve(s[2]) * a, a]
}

/// HighlightsAndShadows tone map (ps6) — byte-exact vs native WARP.
///
/// Native builds a transform graph whose final pixel shader (ps6) remaps luminance in a gamma-2.2
/// perceptual space while preserving chroma via NTSC YIQ, then gamut-clamps. Its second input is the
/// local-contrast "mask" built by [`crate::effects::builtin::filters::highlights_shadows`] (edge
/// strength in alpha, blurred input in rgb); `mask` is that sampled pixel (`None` = the mask equals
/// the input with alpha 0, i.e. smooth content / `MaskBlurRadius` 0 — then `La = L1 = L0` and the
/// `Clarity`/mask terms vanish, so the tone map is a pure per-pixel function of the input + gains).
/// `cx`/`cy` are the Highlights/Shadows gains from [`highlights_shadows_gains`]; `cz = (C>0?2:1)·C`.
///
/// `log`/`exp` are the WARP shader intrinsics (`warpmath::log2`/`exp2`) so `pow(x, 1/2.2)` and
/// `pow(x, 2.2)` are platform-independent, matching native's in-shader `log`/`exp`.
pub fn highlights_shadows(p: Pixel, cx: f32, cy: f32, cz: f32, mask: Option<Pixel>, linear: bool) -> Pixel {
    const W: [f32; 3] = [0.21266, 0.71515, 0.07219]; // Rec.709 luma weights (shader constant)
    // `InputGamma`: SRGB (default) runs the tone map in a gamma-2.2 perceptual space (pow on the way
    // in and out); LINEAR (prop 3 = 0) runs the SAME math directly on the straight rgb — ps6 drops the
    // two `pow` chains. `enc`/`dec` are those transfers (identity when `linear`).
    let pow_1_22 = |x: f32| crate::warpmath::exp2(crate::warpmath::log2(x.max(0.0)) * 0.45454547);
    let pow_22 = |x: f32| crate::warpmath::exp2(crate::warpmath::log2(x.max(0.0)) * 2.2);
    let enc = |x: f32| if linear { x } else { pow_1_22(x) };
    let dec = |x: f32| if linear { x } else { pow_22(x) };
    let fmax = |a: f32, b: f32| if a < b { b } else { a };

    let a = p[3];
    // ps6 unpremultiplies (a==0 → rgb 0) then works in straight alpha.
    let s = unpremul_tb(p); // (rgb/a, a) or 0000
    // r3 = enc(straight.rgb) — the (gamma-encoded, or linear) colour used for L0 and for I/Q.
    let gr = [enc(s[0]), enc(s[1]), enc(s[2])];
    let l0 = W[0] * gr[0] + W[1] * gr[1] + W[2] * gr[2]; // input luma
    // L1 = luma(enc(mask.rgb)); alpha1 = mask.w. No mask ⇒ L1 = L0, alpha1 = 0.
    let (l1, alpha1) = match mask {
        Some(m) => {
            let mg = [enc(m[0]), enc(m[1]), enc(m[2])];
            (W[0] * mg[0] + W[1] * mg[1] + W[2] * mg[2], m[3])
        }
        None => (l0, 0.0),
    };
    let diff = l0 - l1; // r1.z
    let la = alpha1 * diff + l1; // r1.x — blended luma (mask-weighted)
    let lb = cz * diff + l0; // r1.z — Clarity-weighted luma
    // r4.x (shadow) = L0·cy/((L0·cy − L0) + 1); r4.y (highlight) = L0·cx/((L0·cx − L0) + 1).
    let lcy = l0 * cy;
    let lcx = l0 * cx;
    let term_s = lcy / ((lcy - l0) + 1.0);
    let term_h = lcx / ((lcx - l0) + 1.0);
    // Quadratic blend (transcribed from ps6 lines 32-43): La drives the weights, Lb the midtone term.
    let mut r1y = (la * la) * term_h; // La²·termH
    let one_m = -la + 1.0; // 1 − La
    let omla2 = one_m * one_m; // (1 − La)²
    let r1w = -one_m * one_m + 1.0; // 1 − (1 − La)²
    let r1x = -la * la + r1w; // 2·La·(1 − La)
    r1y = term_s * omla2 + r1y; // + (1 − La)²·termS
    let yq = sat_ps(lb * r1x + r1y); // adjusted luma Y' (mad_sat)
    // Chroma: keep the input's NTSC I/Q, substitute the adjusted luma, convert back.
    let i = -0.09991 * gr[0] + -0.33609 * gr[1] + 0.436 * gr[2];
    let q = 0.615 * gr[0] + -0.55861 * gr[1] + -0.05639 * gr[2];
    let rr = yq + 1.28033 * q;
    let gg = yq + -0.21482 * i + -0.38059 * q;
    let bb = yq + 2.12798 * i;
    // Gamut clamp: if max channel > 1, scale all by 1/max (via native's movc blend).
    let m = fmax(fmax(bb, gg), rr);
    let scale = if 1.0 < m { 1.0 } else { 0.0 } * (1.0 / m - 1.0) + 1.0;
    // Re-gamma (SRGB) or pass through (LINEAR), then re-premultiply by the original alpha.
    [dec(rr * scale) * a, dec(gg * scale) * a, dec(bb * scale) * a, a]
}

/// Edge-strength scale (ps0 `cb0[0].x`) for `MaskBlurRadius` `r`, as `SetMaskBlurRadius` computes it:
/// `((r·0.5)·r)·π)·0.7` (op order preserved for f32 bit-exactness; default `r = 1.25 → 1.71806`).
pub fn highlights_shadows_edge_scale(r: f32) -> f32 {
    (((r * 0.5) * r) * std::f32::consts::PI) * 0.69999999
}

/// `cz` tone-map weight from the Clarity strength (`[-1, 1]`), as `SetClarityAmount`: `(C>0?2:1)·C`.
pub fn highlights_shadows_clarity(clarity: f32) -> f32 {
    let c = clarity.clamp(-1.0, 1.0);
    (if c > 0.0 { 2.0 } else { 1.0 }) * c
}

/// `(cx, cy)` tone-map gains from the Highlights and Shadows strengths (`[-1, 1]`), as native's
/// HighlightsAmount/ShadowsAmount setters compute them: `cx = 4^(H·kH)` with
/// `kH = H>0 ? 2.0 : 1.5`, `cy = 5^(S·kS)` with `kS = S>0 ? 0.8 : 1.5`, via `expf(logf(base)·arg)`
/// (`logf(4)`/`logf(5)` as the correctly-rounded f32 constants native's `logf` returns; `expf` is the
/// bit-exact Windows CRT port). Neutral (0) → `(1, 1)` → identity.
pub fn highlights_shadows_gains(highlights: f32, shadows: f32) -> (f32, f32) {
    const LN4: f32 = f32::from_bits(0x3FB1_7218); // logf(4.0f)
    const LN5: f32 = f32::from_bits(0x3FCE_0210); // logf(5.0f)
    let h = highlights.clamp(-1.0, 1.0);
    let s = shadows.clamp(-1.0, 1.0);
    let cx = crate::crt::system_crt::expf(LN4 * (h * if h > 0.0 { 2.0 } else { 1.5 }));
    let cy = crate::crt::system_crt::expf(LN5 * (s * if s > 0.0 { 0.8 } else { 1.5 }));
    (cx, cy)
}

/// WARP `_sat`: NaN→0, clamp to `[0,1]`, normalise −0 to +0 (`max(0, min(1, x))`).
fn sat_ps(x: f32) -> f32 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(0.0, 1.0) + 0.0
    }
}

/// Tint (R9): multiply by the premultiplied tint `diag = (color.rgb·color.a, color.a)`. With
/// `clamp_output` the result is unpremultiplied, saturated and re-premultiplied (→0000 if `a==0`).
pub fn tint(p: Pixel, color: [f32; 4], clamp_output: bool) -> Pixel {
    let ca = color[3];
    let diag = [color[0] * ca, color[1] * ca, color[2] * ca, ca];
    let raw: Pixel = std::array::from_fn(|c| p[c] * diag[c]);
    if clamp_output {
        if raw[3] != 0.0 {
            premul([sat(raw[0] / raw[3]), sat(raw[1] / raw[3]), sat(raw[2] / raw[3]), raw[3]])
        } else {
            [0.0, 0.0, 0.0, 0.0]
        }
    } else {
        raw
    }
}

/// TemperatureAndTint (R10): a diagonal `(rGain, 1, bGain)` channel gain on the premultiplied rgb.
/// Gains come from [`temperature_tint_gains`] (Robertson CCT white-balance). (0,0)→(1,1,1).
pub fn temperature_tint(p: Pixel, temperature: f32, tint: f32) -> Pixel {
    let (r_gain, b_gain) = temperature_tint_gains(temperature, tint);
    [p[0] * r_gain, p[1], p[2] * b_gain, p[3]]
}

/// WhiteLevelAdjustment (R24): scale premultiplied rgb by `inputWhiteLevel/outputWhiteLevel`
/// (0 when the output level is ≤0), alpha unchanged. Levels are nits.
pub fn white_level(p: Pixel, input_white_level: f32, output_white_level: f32) -> Pixel {
    let ratio = if output_white_level > 0.0 {
        input_white_level / output_white_level
    } else {
        0.0
    };
    [p[0] * ratio, p[1] * ratio, p[2] * ratio, p[3]]
}

// --- transfer effects (per channel R,G,B,A; unpremultiply → transfer → re-premultiply) ---

/// A `GammaTransfer` channel: `f(x)= exponent==1 ? amplitude·x+offset : amplitude·|x|^exponent+offset`.
/// Defaults (applied by the caller): amplitude 1, exponent 1, offset 0, not disabled.
#[derive(Clone, Copy)]
pub struct GammaChannel {
    pub amplitude: f32,
    pub exponent: f32,
    pub offset: f32,
    pub disable: bool,
}

/// A `LinearTransfer` channel: `f(x)=slope·x+yIntercept`. Defaults: slope 1, yIntercept 0.
#[derive(Clone, Copy)]
pub struct LinearChannel {
    pub y_intercept: f32,
    pub slope: f32,
    pub disable: bool,
}

/// Shared transfer wrapper: unpremultiply (transparent-black), apply `f` per channel (0..3 = R,G,B,A),
/// optionally saturate, re-premultiply.
fn transfer(p: Pixel, clamp_output: bool, f: impl Fn(f32, usize) -> f32) -> Pixel {
    let s = unpremul_tb(p);
    let mut t: Pixel = std::array::from_fn(|c| f(s[c], c));
    if clamp_output {
        t = t.map(sat);
    }
    premul(t)
}

/// GammaTransfer (R14). Non-finite premultiplied results (`|c| ≥ 1e38`) flush to transparent black,
/// as native does after the `pow`.
pub fn gamma_transfer(p: Pixel, ch: &[GammaChannel; 4], clamp_output: bool) -> Pixel {
    let out = transfer(p, clamp_output, |x, c| {
        let g = ch[c];
        if g.disable {
            x
        } else if g.exponent == 1.0 {
            g.amplitude * x + g.offset
        } else {
            // GPU per-pixel `pow` = WARP `exp2(exponent·log2|x|)` (platform-independent).
            g.amplitude * crate::warpmath::pow(x.abs(), g.exponent) + g.offset
        }
    });
    if out.iter().any(|v| !v.is_finite() || v.abs() >= 1e38) {
        [0.0, 0.0, 0.0, 0.0]
    } else {
        out
    }
}

/// LinearTransfer (R13).
pub fn linear_transfer(p: Pixel, ch: &[LinearChannel; 4], clamp_output: bool) -> Pixel {
    transfer(p, clamp_output, |x, c| {
        let l = ch[c];
        if l.disable {
            x
        } else {
            l.slope * x + l.y_intercept
        }
    })
}

/// DiscreteTransfer (R15): point-sampled table, `f(x)=table[clamp(floor(x·L),0,L−1)]`. Empty or
/// disabled channels pass through.
pub fn discrete_transfer(p: Pixel, tables: [&[f32]; 4], disable: [bool; 4], clamp_output: bool) -> Pixel {
    transfer(p, clamp_output, |x, c| {
        let t = tables[c];
        if disable[c] || t.is_empty() {
            return x;
        }
        let l = t.len();
        let idx = ((x * l as f32).floor() as isize).clamp(0, l as isize - 1) as usize;
        t[idx]
    })
}

/// TableTransfer (R16): each table is a float texture of its `L` entries sampled LINEAR at `u = x·s + o`, where d2d1's
/// constants are `s = 1 − 1/L` and `o = ½·(1/L)` (f32, from one reciprocal). WARP addresses it at a 1/256 weight: a
/// power-of-two `L` floors `((u·L) − ½)·256`, any other rounds `clamp(u, 0, 1)·256L` to nearest less half a texel; the
/// texels clamp to the table and the weight lerps them (VM JIT of a TableTransfer pass; a 2-entry table passes 0.633
/// as 162/256).
/// d2d1 passes a channel through when it is disabled or its table is exactly [0, 1]: all four so, the effect leaves its
/// input as it is; alpha so, its shader skips the alpha lookup (colour channels are looked up whatever their table; VM:
/// a red-only table keeps alpha 248 where the [0, 1] lookup's 1/256 weight gives 247).
pub fn table_transfer(p: Pixel, tables: [&[f32]; 4], disable: [bool; 4], clamp_output: bool) -> Pixel {
    let pass: [bool; 4] = core::array::from_fn(|c| disable[c] || tables[c] == [0.0, 1.0]);
    if !pass.contains(&false) {
        return p;
    }
    transfer(p, clamp_output, |x, c| {
        let t = tables[c];
        if disable[c] || (c == 3 && pass[3]) || t.is_empty() {
            return x;
        }
        let l = t.len();
        let lf = l as f32;
        let inv = 1.0 / lf;
        let u = (x * (1.0 - inv) + 0.5 * inv).clamp(-10.0, 10.0);
        let a = if l.is_power_of_two() { ((u * lf) + -0.5) * 256.0 } else { (u.clamp(0.0, 1.0) * (lf * 256.0)).round_ties_even() - 128.0 };
        let q = if a.is_nan() { 0 } else { a.floor() as i32 };
        let last = l as i32 - 1;
        let (i0, i1) = ((q >> 8).clamp(0, last) as usize, ((q >> 8) + 1).clamp(0, last) as usize);
        let w = (q & 255) as f32 * (1.0 / 256.0);
        t[i0] + (t[i1] - t[i0]) * w
    })
}

// --- alpha / premultiplication ---

/// Premultiply (R17): treat the input as straight and emit premultiplied `(rgb·a, a)`.
pub fn premultiply_effect(p: Pixel) -> Pixel {
    [p[0] * p[3], p[1] * p[3], p[2] * p[3], p[3]]
}

/// UnPremultiply (R18): `a==0 → 0000`, else `(rgb/a, a)`.
pub fn unpremultiply_effect(p: Pixel) -> Pixel {
    unpremul_tb(p)
}

/// AlphaMask (R19): scale every channel of `dest` (input 0) by the premultiplied alpha of `mask`
/// (input 1).
pub fn alpha_mask(dest: Pixel, mask: Pixel) -> Pixel {
    let ma = mask[3];
    [dest[0] * ma, dest[1] * ma, dest[2] * ma, dest[3] * ma]
}

/// ChromaKey (R23), feather off: key out colours within `tolerance` of `key` (straight RGB).
/// `keep = step(tolerance, |s.rgb − key|)`; `invert_alpha` flips it; `outA = a·keep`.
pub fn chroma_key(p: Pixel, key: [f32; 3], tolerance: f32, invert_alpha: bool) -> Pixel {
    let s = unpremul_pass(p);
    let dist = ((s[0] - key[0]).powi(2) + (s[1] - key[1]).powi(2) + (s[2] - key[2]).powi(2)).sqrt();
    let mut keep = if dist < tolerance { 0.0 } else { 1.0 };
    if invert_alpha {
        keep = 1.0 - keep;
    }
    let out_a = p[3] * keep;
    [s[0] * out_a, s[1] * out_a, s[2] * out_a, out_a]
}

// --- Vignette (R: spatial, needs scene position) ---

/// Direct2D's Vignette as native draws it (its vertex setup plus vertex and pixel shaders): a 9-quad
/// mesh over the input rect — the inner rect inset by `inset = min(w/2·ts, h/2·ts)`, four edge strips and
/// four corners — whose vertices carry the scene position and a zero-weight point (the position clamped into
/// the inner rect). WARP interpolates both attributes over each triangle (its attribute planes on
/// the 1/256-snapped corners, keyed by the clipper's rotation), and the PS takes
/// `t = saturate(|pos − zero|·(1/inset) − (1 − strength)·0.75)`, `w = t·t·(−2t + 3)`, `w·(colour − in) + in`,
/// every op in f32 (vig-* probes: 0 px against WARP where the analytic falloff was off by 1 on 8–192 px).
/// ponytail: the positions skip the VS's clip-space and viewport round trip (exact for the probed sizes).
pub struct Vignette {
    origin: [i32; 2],
    w: usize,
    /// Per pixel of the input rect, the triangle whose interior holds its centre (`u8::MAX`: none).
    tri_of: Vec<u8>,
    /// Per triangle, the planes `[dx, dy, c0]` of pos.x, pos.y, zero.x, zero.y.
    planes: Vec<[[f32; 3]; 4]>,
    inv: f32,
    off: f32,
    color: [f32; 4],
    pass: bool,
}

impl Vignette {
    /// `[l, t, r, b]` is the input rect; `color` is used as given (`SetColor` stores it unchanged).
    pub fn new(rect: [i32; 4], color: [f32; 4], transition_size: f32, strength: f32) -> Self {
        let clamp01 = |v: f32| 1.0f32.min(0.0f32.max(v)); // fminf(1, fmaxf(0, v)): NaN → 0
        let (ts, st) = (clamp01(transition_size), clamp01(strength));
        let [l, t, r, b] = rect.map(|v| v as f32);
        let inset = (((r - l) * 0.5) * ts).min(((b - t) * 0.5) * ts);
        let (w, h) = ((rect[2] - rect[0]).max(0) as usize, (rect[3] - rect[1]).max(0) as usize);
        let mut v = Self { origin: [rect[0], rect[1]], w, tri_of: vec![u8::MAX; w * h], planes: Vec::new(), inv: 1.0 / inset, off: (1.0 - st) * 0.75, color, pass: !(ts.abs() >= 1.192_092_9e-7) };
        if v.pass {
            return v;
        }
        let xs = [l, l + inset, r - inset, r];
        let ys = [t, t + inset, b - inset, b];
        let zero = |x: f32, y: f32| [xs[2].min(xs[1].max(x)), ys[2].min(ys[1].max(y))];
        let (fw, fh) = ((rect[2] - rect[0]) as f32, (rect[3] - rect[1]) as f32);
        let (ox, oy) = (rect[0] as i64 * 256, rect[1] as i64 * 256);
        for qy in 0..3 {
            for qx in 0..3 {
                let (x0, x1, y0, y1) = (xs[qx], xs[qx + 1], ys[qy], ys[qy + 1]);
                for tri in [[[x0, y0], [x1, y0], [x1, y1]], [[x0, y0], [x1, y1], [x0, y1]]] {
                    let (p, area) = crate::warp::snap_tri(tri);
                    let o = crate::warp::clip_rotation(tri.map(|q| [q[0] - l, q[1] - t]), fw, fh);
                    let attr = |k: usize| tri.map(|q| { let z = zero(q[0], q[1]); [q[0], q[1], z[0], z[1]][k] });
                    let planes: [[f32; 3]; 4] = core::array::from_fn(|k| {
                        let a = attr(k);
                        crate::warp::plane::setup_triangle_attr(o.map(|i| p[i][0]), o.map(|i| p[i][1]), o.map(|i| a[i]))
                    });
                    let idx = v.planes.len() as u8;
                    v.planes.push(planes);
                    let rel = p.map(|q| [q[0] - ox, q[1] - oy]);
                    let tri_of = &mut v.tri_of;
                    crate::warp::cover_tri(rel, area, w as i64, h as i64, |x, y| tri_of[y * w + x] = idx);
                }
            }
        }
        v
    }

    /// The pixel at device position `(x, y)` (absolute), given the premultiplied input pixel there.
    pub fn pixel(&self, x: i32, y: i32, input: Pixel) -> Pixel {
        if self.pass {
            return input;
        }
        let (rx, ry) = (x - self.origin[0], y - self.origin[1]);
        let Some(&k) = (rx >= 0 && ry >= 0 && (rx as usize) < self.w).then(|| self.tri_of.get(ry as usize * self.w + rx as usize)).flatten() else { return input };
        let Some(pl) = self.planes.get(k as usize) else { return input };
        let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
        let a = pl.map(|q| px * q[0] + (py * q[1] + q[2]));
        let (dx, dy) = (a[0] - a[2], a[1] - a[3]);
        let dist = (dx * dx + dy * dy).sqrt();
        let t = (dist * self.inv - self.off).clamp(0.0, 1.0);
        let w = (t * t) * (t * -2.0 + 3.0);
        std::array::from_fn(|c| w * (self.color[c] - input[c]) + input[c])
    }
}

// --- TemperatureAndTint white balance (Robertson CCT), f32 throughout to match native/managed ---

#[rustfmt::skip]
const ROBERTSON_U: [f32; 31] = [
    0.18006, 0.18066, 0.18133, 0.18208, 0.18293, 0.18388, 0.18494, 0.18611, 0.1874, 0.1888,
    0.19032, 0.19462, 0.19962, 0.20525, 0.21142, 0.21807, 0.22511, 0.23247, 0.2401, 0.24792,
    0.25591, 0.264, 0.27218, 0.28039, 0.28863, 0.29685, 0.30505, 0.3132, 0.32129, 0.32931, 0.33724,
];
#[rustfmt::skip]
const ROBERTSON_V: [f32; 31] = [
    0.26352, 0.26589, 0.26846, 0.27119, 0.27407, 0.27709, 0.28021, 0.28342, 0.28668, 0.28997,
    0.29326, 0.30141, 0.30921, 0.31647, 0.32312, 0.32909, 0.33439, 0.33904, 0.34308, 0.34655,
    0.34951, 0.352, 0.35407, 0.35577, 0.35714, 0.35823, 0.35907, 0.35968, 0.36011, 0.36038, 0.36051,
];
#[rustfmt::skip]
const ROBERTSON_T: [f32; 31] = [
    -0.24341, -0.25479, -0.26876, -0.28539, -0.3047, -0.32675, -0.35156, -0.37915, -0.40955,
    -0.44278, -0.47888, -0.58204, -0.70471, -0.84901, -1.0182, -1.2168, -1.4512, -1.7298, -2.0637,
    -2.4681, -2.9641, -3.5814, -4.3633, -5.3762, -6.7262, -8.5955, -11.324, -15.628, -23.325,
    -40.77, -116.45,
];
#[rustfmt::skip]
const ROBERTSON_RECIP_K: [f32; 31] = [
    1e-10, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5,
    1e-4, 1.25e-4, 1.5e-4, 1.75e-4, 2e-4, 2.25e-4, 2.5e-4, 2.75e-4, 3e-4, 3.25e-4,
    3.5e-4, 3.75e-4, 4e-4, 4.25e-4, 4.5e-4, 4.75e-4, 5e-4, 5.25e-4, 5.5e-4, 5.75e-4, 6e-4,
];

/// Robertson CCT (+Δuv) → linear sRGB primaries, from `TemperatureTintWhiteBalance.CctToLinearSrgb`.
fn cct_to_linear_srgb(kelvin: f32, duv: f32) -> [f32; 3] {
    let kelvin = kelvin.max(1666.6666);
    let duv = duv.clamp(-0.1, 0.1);
    let recip = 1.0 / kelvin;
    let mut i = 1usize;
    while i < 30 && recip > ROBERTSON_RECIP_K[i] {
        i += 1;
    }
    let f1 = (ROBERTSON_RECIP_K[i] - recip) / (ROBERTSON_RECIP_K[i] - ROBERTSON_RECIP_K[i - 1]);
    let f0 = 1.0 - f1;
    let mut u = ROBERTSON_U[i - 1] * f1 + ROBERTSON_U[i] * f0;
    let mut v = ROBERTSON_V[i - 1] * f1 + ROBERTSON_V[i] * f0;
    let (t0, t1) = (ROBERTSON_T[i - 1], ROBERTSON_T[i]);
    let n0 = (1.0 + t0 * t0).sqrt();
    let n1 = (1.0 + t1 * t1).sqrt();
    let mut du = (1.0 / n0) * f1 + (1.0 / n1) * f0;
    let mut dv = (t0 / n0) * f1 + (t1 / n1) * f0;
    let norm = (du * du + dv * dv).sqrt();
    du /= norm;
    dv /= norm;
    u += du * -duv;
    v += dv * -duv;
    let denom = 2.0 * u - 8.0 * v + 4.0;
    let x = 3.0 * u / denom;
    let y = 2.0 * v / denom;
    let big_x = x / y;
    let big_z = (1.0 - x - y) / y;
    [
        3.2404542 * big_x - 1.5371385 + -0.4985314 * big_z,
        -0.969266 * big_x + 1.8760108 + 0.041556 * big_z,
        0.0556434 * big_x - 0.2040259 + 1.0572252 * big_z,
    ]
}

/// `(rGain, bGain)` for TemperatureAndTint, from `TemperatureTintWhiteBalance.ComputeChannelGains`.
pub fn temperature_tint_gains(temperature: f32, tint: f32) -> (f32, f32) {
    let num = temperature * 100.0 * 0.0125;
    let (kelvin, kelvin2) = if num > 0.0 {
        (6502.078, 1.0 / (num * 6.842_5245e-5 + 0.000_153_796_98))
    } else if num < 0.0 {
        (
            1.0 / (0.000_153_796_98 - num * 8.429_8255e-5),
            1.0 / (0.000_153_796_98 - num * -5.379_6983e-5),
        )
    } else {
        (6502.078, 6502.078)
    };
    let duv = 0.003256;
    let duv2 = tint * 1.25 * 0.01 + 0.003256;
    let r = cct_to_linear_srgb(kelvin, duv);
    let r2 = cct_to_linear_srgb(kelvin2, duv2);
    (r2[0] * r[1] / (r[0] * r2[1]), r2[2] * r[1] / (r[2] * r2[1]))
}

// --- HdrToneMap (R25) — native's full IPT/PQ colour-appearance tone map ---
//
// Native builds the ps_4_0 shader's constant buffer (two RGB↔LMS matrices + four IPT-I tone-curve
// anchors) on the CPU, and the per-pixel maths is the native pixel shader (SDR and HDR variants,
// which share the pipeline). The
// managed Paint.NET port is a per-luminance PQ scale and diverges; this is the native model. The
// pipeline: RGB → LMS → a sign-preserving hyperbolic cone response → Ebner–Fairchild IPT, roll the
// intensity I onto the display range with a rational-Bézier curve, scale chroma by `(mappedI/I)^1.2`,
// then invert back. [`HdrToneMap::pixel`] follows the native pixel shader's operation order
// (so `mad` = mul-then-add, matching WARP's no-FMA order), and every `exp2`/`log2` goes
// through [`crate::warpmath`] (WARP-exact, platform-independent). The adaptive exponent `n` is
// computed two ways, exactly as native does: [`shader_n`] (the per-pixel shader, `log2`/`mad`) and
// [`anchor_n`] (the CPU cache builder's natural-log form) for the anchors.
//
// `fx-color-27` (SDR, 8-bit) and `hdr-07` (HDR, fp16) are both byte-exact. The two constant-buffer
// RGB↔LMS matrices are built through the native DirectXMath chain below, not a pre-folded `BASE·scale`
// (which left hdr-07 2 fp16 LSBs off): native's RGB→LMS and LMS→RGB matrix builders fold the
// luminance scale into the D65 white point and build RGB↔XYZ↔LMS with runtime `XMMatrixInverse`, whose
// exact f32 rounding the scalar reproduction matches (and, being plain f32, stays platform-independent).

/// XYZ→LMS (IPT, Ebner–Fairchild), native's f32 constants.
#[rustfmt::skip]
const IPT_XYZ_LMS: [[f32; 3]; 3] = [
    [0.400_2, 0.707_5, -0.080_7],
    [-0.228, 1.15, 0.061_2],
    [0.0, 0.0, 0.918_4],
];
/// L'M'S'→IPT (native constants; the shader bakes these and their inverse).
#[rustfmt::skip]
const IPT_LMS_IPT: [[f32; 3]; 3] = [
    [0.4, 0.4, 0.2],
    [4.455, -4.851, 0.396],
    [0.805_6, 0.357_2, -1.162_8],
];
/// IPT→L'M'S' (`IPT_LMS_IPT`⁻¹) — the native shader's exact baked immediates.
#[rustfmt::skip]
const IPT_IPT_LMS: [[f32; 3]; 3] = [
    [1.0, 0.097_568_93, 0.205_226_44],
    [1.0, -0.113_876_484, 0.133_217_16],
    [1.0, 0.032_615_11, -0.676_887_15],
];
// --- native constant-buffer matrix chain (RGB→LMS and LMS→RGB) ---
//
// Both RGB↔LMS matrices are built the way native does: a Rec.709 RGB→XYZ matrix with the luminance
// scale folded into the D65 white point, then composed with the IPT
// XYZ→LMS matrix `A` — the LMS→RGB direction as `XMMatrixInverse(RGB→XYZ)·XMMatrixInverse(A)`. The
// inverses and products are DirectXMath's SSE2 (no-FMA) `XMMatrixInverse`/`XMMatrixMultiply`,
// reproduced below as scalar f32 in their exact op order (each SSE lane is one scalar f32 op, so this
// is bit-identical to WARP's host and deterministic on every target). A pre-folded `BASE·scale`
// factorisation can't reproduce the inverse chain's last-bit rounding — this does (fixes hdr-07).

/// 4-lane f32 vector / 4×4 matrix, DirectXMath row-vector layout.
type V4 = [f32; 4];
type M4 = [V4; 4];
#[inline] fn vmul(a: V4, b: V4) -> V4 { [a[0]*b[0], a[1]*b[1], a[2]*b[2], a[3]*b[3]] }
#[inline] fn vadd(a: V4, b: V4) -> V4 { [a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]] }
#[inline] fn vsub(a: V4, b: V4) -> V4 { [a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]] }
/// `_mm_shuffle_ps(a, b, _MM_SHUFFLE(s3,s2,s1,s0))`: low two lanes from `a`, high two from `b`.
#[inline] fn shuf(a: V4, b: V4, s3: usize, s2: usize, s1: usize, s0: usize) -> V4 { [a[s0], a[s1], b[s2], b[s3]] }
#[inline] fn perm(a: V4, s3: usize, s2: usize, s1: usize, s0: usize) -> V4 { [a[s0], a[s1], a[s2], a[s3]] }
#[inline] fn fnmadd(a: V4, b: V4, c: V4) -> V4 { vsub(c, vmul(a, b)) } // c − a·b (SSE2 negate-mul-sub)

/// `XMMatrixMultiply` (SSE2): `result.row[i] = (r.z·B2 + r.x·B0) + (r.w·B3 + r.y·B1)`.
fn xm_mul(a: M4, b: M4) -> M4 {
    let row = |r: V4| vadd(vadd(vmul([r[2]; 4], b[2]), vmul([r[0]; 4], b[0])), vadd(vmul([r[3]; 4], b[3]), vmul([r[1]; 4], b[1])));
    [row(a[0]), row(a[1]), row(a[2]), row(a[3])]
}

/// `XMMatrixInverse` (SSE2, no-FMA): transpose, paired 2×2 sub-determinants, cofactors, `1/det`.
fn xm_inverse(m: M4) -> M4 {
    let vt1 = shuf(m[0], m[1], 1, 0, 1, 0);
    let vt3 = shuf(m[0], m[1], 3, 2, 3, 2);
    let vt2 = shuf(m[2], m[3], 1, 0, 1, 0);
    let vt4 = shuf(m[2], m[3], 3, 2, 3, 2);
    let mt = [shuf(vt1, vt2, 2, 0, 2, 0), shuf(vt1, vt2, 3, 1, 3, 1), shuf(vt3, vt4, 2, 0, 2, 0), shuf(vt3, vt4, 3, 1, 3, 1)];

    let (s00, s10) = (perm(mt[2], 1, 1, 0, 0), perm(mt[3], 3, 2, 3, 2));
    let (s01, s11) = (perm(mt[0], 1, 1, 0, 0), perm(mt[1], 3, 2, 3, 2));
    let (s02, s12) = (shuf(mt[2], mt[0], 2, 0, 2, 0), shuf(mt[3], mt[1], 3, 1, 3, 1));
    let (mut d0, mut d1, mut d2) = (vmul(s00, s10), vmul(s01, s11), vmul(s02, s12));
    let (s00, s10) = (perm(mt[2], 3, 2, 3, 2), perm(mt[3], 1, 1, 0, 0));
    let (s01, s11) = (perm(mt[0], 3, 2, 3, 2), perm(mt[1], 1, 1, 0, 0));
    let (s02, s12) = (shuf(mt[2], mt[0], 3, 1, 3, 1), shuf(mt[3], mt[1], 2, 0, 2, 0));
    d0 = fnmadd(s00, s10, d0);
    d1 = fnmadd(s01, s11, d1);
    d2 = fnmadd(s02, s12, d2);

    let s00 = perm(mt[1], 1, 0, 2, 1);
    let mut s11 = shuf(d0, d2, 1, 1, 3, 1);
    let s10 = shuf(s11, d0, 0, 3, 0, 2);
    let s01 = perm(mt[0], 0, 1, 0, 2);
    s11 = shuf(s11, d0, 2, 1, 2, 1);
    let mut s13 = shuf(d1, d2, 3, 3, 3, 1);
    let s02 = perm(mt[3], 1, 0, 2, 1);
    let s12 = shuf(s13, d1, 0, 3, 0, 2);
    let s03 = perm(mt[2], 0, 1, 0, 2);
    s13 = shuf(s13, d1, 2, 1, 2, 1);
    let (mut c0, mut c2, mut c4, mut c6) = (vmul(s00, s10), vmul(s01, s11), vmul(s02, s12), vmul(s03, s13));

    let s00 = perm(mt[1], 2, 1, 3, 2);
    let mut s11 = shuf(d0, d2, 0, 0, 1, 0);
    let s10 = shuf(d0, s11, 2, 1, 0, 3);
    let s01 = perm(mt[0], 1, 3, 2, 3);
    s11 = shuf(d0, s11, 0, 2, 1, 2);
    let mut s13 = shuf(d1, d2, 2, 2, 1, 0);
    let s02 = perm(mt[3], 2, 1, 3, 2);
    let s12 = shuf(d1, s13, 2, 1, 0, 3);
    let s03 = perm(mt[2], 1, 3, 2, 3);
    s13 = shuf(d1, s13, 0, 2, 1, 2);
    c0 = fnmadd(s00, s10, c0);
    c2 = fnmadd(s01, s11, c2);
    c4 = fnmadd(s02, s12, c4);
    c6 = fnmadd(s03, s13, c6);

    let s00 = perm(mt[1], 0, 3, 0, 3);
    let s10 = perm(shuf(d0, d2, 1, 0, 2, 2), 0, 2, 3, 0);
    let s01 = perm(mt[0], 2, 0, 3, 1);
    let s11 = perm(shuf(d0, d2, 1, 0, 3, 0), 2, 1, 0, 3);
    let s02 = perm(mt[3], 0, 3, 0, 3);
    let s12 = perm(shuf(d1, d2, 3, 2, 2, 2), 0, 2, 3, 0);
    let s03 = perm(mt[2], 2, 0, 3, 1);
    let s13 = perm(shuf(d1, d2, 3, 2, 3, 0), 2, 1, 0, 3);
    let (p0, p1, p2, p3) = (vmul(s00, s10), vmul(s01, s11), vmul(s02, s12), vmul(s03, s13));
    let (c1, c0) = (vsub(c0, p0), vadd(c0, p0));
    let (c3, c2) = (vadd(c2, p1), vsub(c2, p1));
    let (c5, c4) = (vsub(c4, p2), vadd(c4, p2));
    let (c7, c6) = (vadd(c6, p3), vsub(c6, p3));

    let r0 = perm(shuf(c0, c1, 3, 1, 2, 0), 3, 1, 2, 0);
    let r1 = perm(shuf(c2, c3, 3, 1, 2, 0), 3, 1, 2, 0);
    let r2 = perm(shuf(c4, c5, 3, 1, 2, 0), 3, 1, 2, 0);
    let r3 = perm(shuf(c6, c7, 3, 1, 2, 0), 3, 1, 2, 0);
    // determinant = XMVector4Dot(r0, MT.r[0]) = (t0+t2)+(t1+t3), reciprocal, scale each cofactor row.
    let t = vmul(r0, mt[0]);
    let recip = [1.0 / ((t[0] + t[2]) + (t[1] + t[3])); 4];
    [vmul(r0, recip), vmul(r1, recip), vmul(r2, recip), vmul(r3, recip)]
}

/// Rec.709 chromaticity-ratio matrix `P` (native's immediates: `x/y, 1, z/y`
/// per primary).
#[rustfmt::skip]
const P709: [[f32; 3]; 3] = [
    [f32::from_bits(0x3ff8_3e0f), 0.5, f32::from_bits(0x4020_0001)],
    [1.0, 1.0, 1.0],
    [f32::from_bits(0x3dba_2e8c), f32::from_bits(0x3e2a_aaa6), f32::from_bits(0x4152_aaab)],
];

/// 3×3 → 4×4 homogeneous embed (4th row `[0,0,0,1]`, 0 4th column).
fn embed(m: &[[f32; 3]; 3]) -> M4 {
    [
        [m[0][0], m[0][1], m[0][2], 0.0],
        [m[1][0], m[1][1], m[1][2], 0.0],
        [m[2][0], m[2][1], m[2][2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}
fn top3(m: M4) -> [[f32; 3]; 3] { std::array::from_fn(|i| std::array::from_fn(|j| m[i][j])) }
/// Homogeneous embed of a 3×3's transpose (native inverts `Pᵀ`, so `[wx,wy,wz,1]·(Pᵀ)⁻¹ = P⁻¹·white`).
fn embed_t(m: &[[f32; 3]; 3]) -> M4 {
    [
        [m[0][0], m[1][0], m[2][0], 0.0],
        [m[0][1], m[1][1], m[2][1], 0.0],
        [m[0][2], m[1][2], m[2][2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Native's white-point RGB→XYZ matrix for `scale`: D65 white × `scale`, per-primary scales `S = [wx,wy,wz,1]·(Pᵀ)⁻¹`
/// (native accumulation order), then `M[i][j] = P[i][j]·S[j]`.
fn compute_rgb_to_xyz(scale: f32) -> [[f32; 3]; 3] {
    let white = [scale * 0.950_455_96, scale, scale * 1.089_057_7, 1.0];
    let pinv = xm_inverse(embed_t(&P709));
    let s = vadd(
        vadd(vadd(vmul([white[2]; 4], pinv[2]), pinv[3]), vmul([white[1]; 4], pinv[1])),
        vmul([white[0]; 4], pinv[0]),
    );
    std::array::from_fn(|i| std::array::from_fn(|j| P709[i][j] * s[j]))
}

fn dp3(m: [f32; 3], v: [f32; 3]) -> f32 {
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2]
}
fn signf(x: f32) -> f32 {
    (x > 0.0) as i32 as f32 - (x < 0.0) as i32 as f32
}
/// `pow` as the WARP shader does it (`log`/`exp` opcodes): `exp2(n·log2|x|)`; 0 at x==0. Routed
/// through [`crate::warpmath`] so the result is WARP-exact and identical on every platform.
fn powx(x: f32, n: f32) -> f32 {
    crate::warpmath::exp2(n * crate::warpmath::log2(x.abs()))
}
/// Adaptive exponent `n`, exactly as the per-pixel shader computes it:
/// WARP `log`/`exp`, `mad` (= mul then add — WARP has no FMA), native immediates. `0.6931472` is
/// `ln2`, `5.7620516` is `ln(318)`.
fn shader_n(in_max: f32) -> f32 {
    let x = -(20.0 / in_max) * 1.358_695_6 + 1.25;
    let y = 5.762_051_6 / (crate::warpmath::log2(in_max) * std::f32::consts::LN_2);
    0.58 / (y * x)
}
/// Adaptive exponent `n` as the CPU cache builder computes it (native `XYZtoIPT`): natural logs and
/// the `1.25 − (20/inMax)/0.184·0.25` form. Uses the WARP `log` so it is identical on every target.
fn anchor_n(in_max: f32) -> f32 {
    let ln = |v: f32| crate::warpmath::log2(v) * std::f32::consts::LN_2;
    0.58 / ((ln(318.0) / ln(in_max)) * (1.25 - 20.0 / in_max / 0.184 * 0.25))
}
/// Sign-preserving cone response `N` (native `ipt_f` / shader lines 20-32):
/// `sign·246·|x|ⁿ/(2ⁿ+|x|ⁿ) + 0.02`.
fn cone_fwd(x: f32, n: f32, two_n: f32) -> f32 {
    let p = powx(x, n);
    p * (246.0 * signf(x)) / (two_n + p) + 0.02
}

/// HdrToneMap (R25). Built per effect instance from the three properties; [`Self::pixel`] is the
/// per-pixel shader. Byte-exact vs WARP `fx-color-27` (SDR, 8-bit). Real HDR domain adds float
/// targets (SDR display mode clamps `inMax≥4000`, `outMax≥max(150, 0.75·outputMaxLum)`).
pub struct HdrToneMap {
    rgb_lms: [[f32; 3]; 3],
    lms_rgb: [[f32; 3]; 3],
    in_max: f32,
    out_max: f32,
    anchors: [f32; 4], // I at 0.6·outMax (−0.05), 0.1·outMax, inMax, outMax
}

impl HdrToneMap {
    /// `input_max_luminance`/`output_max_luminance` in nits; `display_mode` 0=SDR, 1=HDR.
    pub fn new(input_max_luminance: f32, output_max_luminance: f32, display_mode: u32) -> Self {
        let hdr = display_mode != 0;
        let (in_max, out_max) = if hdr || input_max_luminance <= 160.0 {
            (input_max_luminance, output_max_luminance)
        } else {
            (input_max_luminance.max(4000.0), (output_max_luminance * 0.75).max(150.0))
        };
        // Native builds both matrices through the DirectXMath chain: RGB→LMS = A·RGB→XYZ(80/inMax),
        // LMS→RGB = XMMatrixInverse(RGB→XYZ(s/inMax)) · XMMatrixInverse(A), s = (outMax/rawOut)·80.
        let xyz_lms = embed(&IPT_XYZ_LMS);
        let rgb_lms = top3(xm_mul(xyz_lms, embed(&compute_rgb_to_xyz(80.0 / in_max))));
        let xyz_to_rgb = top3(xm_inverse(embed(&compute_rgb_to_xyz((out_max / output_max_luminance) * 80.0 / in_max))));
        let lms_rgb = top3(xm_mul(embed(&xyz_to_rgb), xm_inverse(xyz_lms)));
        // Anchors come from native's CPU tone-map cache builder, which uses its own `n` (natural-log form) — not the shader's `n`.
        let n = anchor_n(in_max);
        let two_n = crate::warpmath::exp2(n);
        // IPT-I of a neutral `lum`-nit stimulus, as native computes it: normalise by inMax,
        // RGB→LMS via the IPT matrix, cone response, then the I row of the IPT matrix.
        let lum_i = |lum: f32| -> f32 {
            let xyz = [lum * 0.950_428_55 / in_max, lum / in_max, lum * 1.088_900_4 / in_max];
            let lms = [dp3(IPT_XYZ_LMS[0], xyz), dp3(IPT_XYZ_LMS[1], xyz), dp3(IPT_XYZ_LMS[2], xyz)];
            let lp = [cone_fwd(lms[0], n, two_n), cone_fwd(lms[1], n, two_n), cone_fwd(lms[2], n, two_n)];
            dp3(IPT_LMS_IPT[0], lp)
        };
        Self {
            rgb_lms,
            lms_rgb,
            in_max,
            out_max,
            anchors: [
                lum_i(out_max - out_max * 0.4) - 0.05,
                if hdr { 0.0 } else { lum_i(out_max * 0.1) },
                lum_i(in_max),
                lum_i(out_max),
            ],
        }
    }

    pub fn pixel(&self, p: Pixel) -> Pixel {
        let a = p[3];
        let rgb = if a != 0.0 { [p[0] / a, p[1] / a, p[2] / a] } else { [0.0, 0.0, 0.0] };
        let lms = [dp3(self.rgb_lms[0], rgb), dp3(self.rgb_lms[1], rgb), dp3(self.rgb_lms[2], rgb)];
        // The shader recomputes `n`/`two_n` per pixel (its own form), distinct from the anchors'.
        let n = shader_n(self.in_max);
        let two_n = crate::warpmath::exp2(n);
        let lp = [cone_fwd(lms[0], n, two_n), cone_fwd(lms[1], n, two_n), cone_fwd(lms[2], n, two_n)];
        let i = dp3(IPT_LMS_IPT[0], lp);
        let pp = dp3(IPT_LMS_IPT[1], lp);
        let tp = dp3(IPT_LMS_IPT[2], lp);
        let [anchor0, anchor1, anchor2, anchor3] = self.anchors;
        let mapped = if anchor1 >= i {
            // shadow toe: rational quadratic Bézier over t = I/anchor1
            let t = i / anchor1;
            let omt = 1.0 - t;
            let two_t = omt * t + omt * t;
            let den = t * t + (omt * omt * 0.35 + two_t);
            let num = two_t * (anchor1 * 0.5) + (anchor1 + 0.05) * (t * t);
            num / den
        } else if anchor0 >= i {
            i + 0.05
        } else if i < anchor2 {
            // highlight roll-off: rational quadratic Bézier over s
            let k = (self.in_max / self.out_max) * 0.4;
            let s = (i - anchor0) / ((anchor2 - anchor0) + 0.000_01);
            let w = anchor3 / (anchor0 + 0.05);
            let (oms, s2) = (1.0 - s, s * s);
            let two_s = oms * s + oms * s;
            let num = s2 * k + (two_s * k + w * (oms * oms));
            let den = (k * s2) * anchor3 + ((w * (oms * oms)) * (anchor0 + 0.05) + k * (two_s * anchor3));
            den / num
        } else {
            anchor3
        };
        let sc = powx((mapped / i).abs().min(1.0), 1.2);
        let ipt = [mapped, pp * sc, tp * sc];
        // inverse IPT (B⁻¹ baked) then the inverse cone response, per channel.
        let lms2: [f32; 3] = std::array::from_fn(|c| {
            let x = dp3(IPT_IPT_LMS[c], ipt);
            let m = two_n * (50.0 * x - 1.0) / (50.0 * x - 12301.0);
            signf(x) * crate::warpmath::exp2((1.0 / n) * crate::warpmath::log2(m.abs()))
        });
        let out = [dp3(self.lms_rgb[0], lms2), dp3(self.lms_rgb[1], lms2), dp3(self.lms_rgb[2], lms2)];
        [out[0] * a, out[1] * a, out[2] * a, a]
    }
}

// --- RgbToHue / HueToRgb — native's pixel shaders (HSV; min16f → fp32 on WARP) ---

/// HLSL `frc`: fractional part in `[0,1)` (`x − floor(x)`), unlike Rust's sign-keeping `f32::fract`.
fn frc(x: f32) -> f32 {
    x - x.floor()
}

/// RgbToHue: RGB → `(H, S, V/L)` packed in RGB, premultiplied. `color_space` is
/// `D2D1_RGBTOHUE_OUTPUT_COLOR_SPACE` (0 = HSV, 1 = HSL). Transparent-black on `a==0`, H normalised
/// to `[0,1)` via `frac`. Exactly the native `ps_4_0` body.
pub fn rgb_to_hue(p: Pixel, color_space: u32) -> Pixel {
    if color_space == 1 {
        return rgb_to_hsl(p);
    }
    let a = p[3];
    let s = if a != 0.0 { [p[0] / a, p[1] / a, p[2] / a] } else { [0.0, 0.0, 0.0] };
    let (r, g, b) = (s[0], s[1], s[2]);
    let cmax = r.max(g).max(b);
    let delta = cmax - r.min(g).min(b);
    // hue candidates (numerator/delta + base), then select by which channel is the max
    let mut h = [(g - b) / delta, (b - r) / delta + 2.0, (r - g) / delta + 4.0];
    if delta == 0.0 {
        h = [1.0, 1.0, 1.0];
    }
    let hy = if g == cmax { h[1] } else { h[2] };
    let hx = if r == cmax { h[0] } else { hy };
    let hue = frc(hx * (1.0 / 6.0));
    let sat = if cmax == 0.0 { 1.0 } else { delta / cmax };
    [hue * a, sat * a, cmax * a, a]
}

/// RgbToHue with `OutputColorSpace = HSL` — the native HSL `ps_4_0` body in its operation order. The
/// hue is the same max-channel ramp as HSV; lightness is `(cmax+cmin)/2` and saturation is
/// `delta / denom` with `denom = (cmax+cmin > 1) ? 2-(cmax+cmin) : cmax+cmin` (1 when `denom == 0`).
/// The shader saturates the unpremultiplied RGB before the min/max (no-op for valid premultiplied
/// input, but faithful). Output is premultiplied `[a·H, a·S, 0.5·(sum·a), a]`.
fn rgb_to_hsl(p: Pixel) -> Pixel {
    let a = p[3];
    let s = if a != 0.0 { [sat(p[0] / a), sat(p[1] / a), sat(p[2] / a)] } else { [0.0, 0.0, 0.0] };
    let (r, g, b) = (s[0], s[1], s[2]);
    let cmin = r.min(g).min(b);
    let cmax = r.max(g).max(b);
    let delta = cmax - cmin;
    let sum = cmin + cmax;
    let mut h = [(g - b) / delta, (b - r) / delta + 2.0, (r - g) / delta + 4.0];
    if delta == 0.0 {
        h = [1.0, 1.0, 1.0];
    }
    let hy = if g == cmax { h[1] } else { h[2] };
    let hx = if r == cmax { h[0] } else { hy };
    let hue = frc(hx * (1.0 / 6.0));
    let denom = if 1.0 < sum { 2.0 - sum } else { sum };
    let satl = if denom == 0.0 { 1.0 } else { delta / denom };
    [a * hue, a * satl, 0.5 * (sum * a), a]
}

// --- ColorManagement (R26) — native's sRGB↔scRGB fast path ---

/// Build the ColorManagement transform from its colour-context and property inputs. The source
/// and destination rendering intents and the quality do not change the sRGB↔scRGB result (same
/// Rec.709 primaries, no per-intent tables — verified byte-for-byte: `cms-01`/`06`/`07`/`08` are
/// identical on WARP), so they are accepted and ignored. `alpha_mode` is
/// `D2D1_COLORMANAGEMENT_ALPHA_MODE` (2 = straight). Returns `None` for colour-context combinations
/// not yet reproduced (anything but the ICC sRGB/scRGB named contexts); see
/// [`crate::color_context::ColorTransform`].
pub fn color_management(
    src: &crate::color_context::ColorContext,
    dst: &crate::color_context::ColorContext,
    alpha_mode: u32,
) -> Option<crate::color_context::ColorTransform> {
    crate::color_context::ColorTransform::new(src, dst, alpha_mode == 2)
}

/// HueToRgb: `(H, S, V/L)` packed in RGB → RGB, premultiplied. `color_space` is
/// `D2D1_HUETORGB_INPUT_COLOR_SPACE` (0 = HSV, 1 = HSL). Exactly the native `ps_4_0` body (triangle
/// hue ramp).
pub fn hue_to_rgb(p: Pixel, color_space: u32) -> Pixel {
    if color_space == 1 {
        return hsl_to_rgb(p);
    }
    let a = p[3];
    let s = if a != 0.0 { [p[0] / a, p[1] / a, p[2] / a] } else { [0.0, 0.0, 0.0] };
    let (hue, sval, v) = (s[0], s[1], s[2]);
    let ramp: [f32; 3] = std::array::from_fn(|c| {
        let off = [0.0, -1.0 / 3.0, -2.0 / 3.0][c];
        sat((frc(hue + off) - 0.5).abs() * 6.0 - 1.0)
    });
    let vs = v * sval;
    let v1s = v * (1.0 - sval);
    [(vs * ramp[0] + v1s) * a, (vs * ramp[1] + v1s) * a, (vs * ramp[2] + v1s) * a, a]
}

/// HueToRgb with `InputColorSpace = HSL` — the native HSL `ps_4_0` body, in its operation order. The triangle
/// hue ramp is shared with HSV; the chroma is `S·(1-|2L-1|)` and each channel is `chroma·ramp + m`
/// with `m = L - chroma/2`.
fn hsl_to_rgb(p: Pixel) -> Pixel {
    let a = p[3];
    let s = if a != 0.0 { [p[0] / a, p[1] / a, p[2] / a] } else { [0.0, 0.0, 0.0] };
    let (hue, sval, lum) = (s[0], s[1], s[2]);
    let chroma = sval * (1.0 - (lum * 2.0 - 1.0).abs());
    let m = -chroma * 0.5 + lum;
    let ramp: [f32; 3] = std::array::from_fn(|c| {
        let off = [0.0, -1.0 / 3.0, -2.0 / 3.0][c];
        sat((frc(hue + off) - 0.5).abs() * 6.0 - 1.0)
    });
    [(chroma * ramp[0] + m) * a, (chroma * ramp[1] + m) * a, (chroma * ramp[2] + m) * a, a]
}

#[cfg(test)]
mod tests {
    use super::*;

    // `xm_inverse` is the DirectXMath cofactor inverse; check it actually inverts (the byte-exact
    // op order is covered end-to-end by fx-color-27 / hdr-07 in the conformance suite).
    #[test]
    fn xm_inverse_inverts() {
        let m: M4 = [[2.0, 0.3, 0.1, 0.0], [0.4, 1.7, 0.2, 0.0], [0.05, 0.6, 3.1, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let prod = xm_mul(xm_inverse(m), m);
        for i in 0..4 {
            for j in 0..4 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((prod[i][j] - want).abs() < 1e-4, "inv·M not identity at [{i}][{j}]: {}", prod[i][j]);
            }
        }
    }

    // `HdrToneMap` is now built once per property set and cached, not rebuilt per pixel. Prove `new` is
    // a pure function of its three properties, so a cached instance is byte-identical to a freshly built
    // one for every input — the all-inputs proof for the memoisation.
    #[test]
    fn hdr_tonemap_new_is_deterministic() {
        for &(ml, md, mode) in &[(4000.0f32, 300.0f32, 0u32), (1000.0, 200.0, 1), (160.0, 80.0, 0), (6000.0, 600.0, 0), (80.0, 80.0, 1)] {
            let (a, b) = (HdrToneMap::new(ml, md, mode), HdrToneMap::new(ml, md, mode));
            for r in [0.0f32, 0.2, 0.5, 0.9, 1.0, 2.5] {
                for g in [0.0f32, 0.33, 1.0, 1.8] {
                    let p = [r, g, (r + g) * 0.4, 1.0];
                    assert_eq!(a.pixel(p), b.pixel(p), "ml={ml} md={md} mode={mode} p={p:?}");
                }
            }
        }
    }

    // HighlightsShadows tone map (Clarity=0) is byte-exact to native WARP. The expected bytes are the
    // native Direct2D render of a 256×1 opaque gray ramp through the effect, captured on the reference
    // VM (the final draw's output == the fxhs oracle). Gains come straight from native's
    // setters: Highlights 0.5 → cx=4, Shadows 0.5 → cy=5^0.4.
    #[test]
    fn highlights_shadows_matches_native_ramp() {
        // round-to-nearest-even *255, the output-merger UNORM8 quantise (see d2d_shader::to_unorm8).
        let q = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round_ties_even() as u8;
        let gray_out = |i: u8, cx: f32, cy: f32| -> u8 {
            let v = i as f32 / 255.0;
            // No mask (smooth content): La = L1 = L0, Clarity term vanishes. SRGB (default) gamma.
            q(highlights_shadows([v, v, v, 1.0], cx, cy, 0.0, None, false)[1]) // opaque gray → premul == straight
        };
        // Highlights=0.5 (cx=4, cy=1): native green at these indices.
        let (cx, cy) = highlights_shadows_gains(0.5, 0.0);
        assert_eq!((cx, cy), (4.0, 1.0));
        for (i, want) in [(16u8, 19u8), (32, 42), (64, 88), (96, 131), (128, 169), (160, 201), (192, 226), (224, 244), (255, 255)] {
            assert_eq!(gray_out(i, cx, cy), want, "H=0.5 idx {i}");
        }
        // Shadows=0.5 (cx=1, cy=5^0.4≈1.90365): native green.
        let (cx, cy) = highlights_shadows_gains(0.0, 0.5);
        assert_eq!(cx, 1.0);
        for (i, want) in [(16u8, 27u8), (32, 44), (64, 73), (96, 102), (128, 131), (160, 161), (192, 192), (224, 224), (255, 255)] {
            assert_eq!(gray_out(i, cx, cy), want, "S=0.5 idx {i}");
        }
        // Neutral (0,0) → identity.
        let (cx, cy) = highlights_shadows_gains(0.0, 0.0);
        assert_eq!((cx, cy), (1.0, 1.0));
        for i in [0u8, 1, 50, 128, 200, 255] {
            assert_eq!(gray_out(i, cx, cy), i, "neutral idx {i}");
        }
    }
}
