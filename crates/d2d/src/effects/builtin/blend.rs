//! Blending-family built-in effects: `Blend` (26 modes), `ArithmeticComposite`, `CrossFade` and
//! `Opacity`. Pure pixel math over premultiplied-alpha RGBA f32 input buffers plus the effect
//! properties — exactly the native Direct2D Blend/ArithmeticComposite/CrossFade/Opacity
//! shaders WARP runs. Inputs and outputs are premultiplied `[R, G, B, A]`; the caller stores each
//! channel to the target format (`B8G8R8A8_UNORM` → `round_ties_even`). Verified byte-exact against
//! `conformance/oracle/win11-26100` (see `tests/effects/blend`). The 2B effect-graph engine wires
//! these in once it is on main; until then they are tested directly against the captured scenes.

/// Premultiplied RGBA, f32 per channel.
pub type Rgba = [f32; 4];

/// HLSL `saturate`: clamp to `[0, 1]`, mapping NaN → 0 (matches `max(0, min(1, x))`).
fn sat(x: f32) -> f32 {
    x.max(0.0).min(1.0)
}

/// HLSL `UnpremultiplyOrZero`: straight rgb, or 0 when alpha is 0.
fn straight(p: Rgba) -> [f32; 3] {
    let a = p[3];
    if a != 0.0 {
        [p[0] / a, p[1] / a, p[2] / a]
    } else {
        [0.0; 3]
    }
}

/// W3C non-separable luminosity, `dot(c, (0.3, 0.59, 0.11))` with the exact shader constants.
fn lum(c: [f32; 3]) -> f32 {
    let k = [
        f32::from_bits(0x3E99999A), // 0.3
        f32::from_bits(0x3F170A3D), // 0.59
        f32::from_bits(0x3DE147AE), // 0.11
    ];
    c[0] * k[0] + c[1] * k[1] + c[2] * k[2]
}

/// The tail every separable blend shares, in the shaders' order: the source `Cs·(1 − αb)`, plus `(αb·B)·αs`, then
/// the backdrop `Cb·(1 − αs)` added; alpha `(αs + αb) − αs·αb`. `B` is the blend of the straight colours; `Cs`
/// and `Cb` stay premultiplied as sampled, so a colour above its alpha passes through as it is.
fn tail(dst: Rgba, src: Rgba, b: [f32; 3]) -> Rgba {
    let (ab, as_) = (dst[3], src[3]);
    let rgb = |i: usize| dst[i] * (1.0 - as_) + ((ab * b[i]) * as_ + src[i] * (1.0 - ab));
    [rgb(0), rgb(1), rgb(2), alpha(dst, src)]
}

fn alpha(dst: Rgba, src: Rgba) -> f32 {
    -src[3] * dst[3] + (src[3] + dst[3])
}

/// HLSL `round` (to nearest, ties to even): the shaders' branch select, a 0/1 weight for an in-range value.
fn rne(x: f32) -> f32 {
    x.round_ties_even()
}

/// The 3-channel `d2d1` blend `B(b, s)` per channel, `b` the straight backdrop and `s` the straight source, op for op
/// as the native Blend shaders evaluate it (their selects are roundings and comparisons of the unclamped straight
/// values, which matters for a colour above its alpha).
fn separable(mode: u32, b: f32, s: f32) -> f32 {
    match mode {
        0 => s * b, // MULTIPLY
        5 => {
            // COLOR_BURN
            let t = (s + b) + -1.0;
            let q = if 0.0 < t { t / s } else { 0.0 };
            sat(if b >= 1.0 { 1.0 } else { q })
        }
        6 => sat((s + b) + -1.0), // LINEAR_BURN
        9 => sat(if s < 1.0 { b / (1.0 - s) } else { b.ceil() }), // COLOR_DODGE
        10 => sat(s + b),          // LINEAR_DODGE
        11 => {
            // OVERLAY: the backdrop rounds to its branch.
            let screen = b * (1.0 - s) + s;
            let prod = s * b;
            let high = -prod * 2.0 + (screen * 2.0 + -1.0);
            rne(b) * high + (prod + prod)
        }
        12 => {
            // SOFT_LIGHT
            let k = s * 2.0 + -1.0;
            let mut v = (b.sqrt() - b) * k + b;
            let q = (-s * f32::from_bits(0x4190_0000) + f32::from_bits(0x4110_0000)) * b + s * f32::from_bits(0x40B8_51EC);
            let q = q + f32::from_bits(0xBFF0_A3D7);
            v = -q * b + v;
            let quad = b * q;
            v = rne(b + f32::from_bits(0x3EB7_7CEE)) * v + quad;
            let low = (1.0 - b) * k + 1.0;
            v = -b * low + v;
            rne(s) * v + low * b
        }
        13 => {
            // HARD_LIGHT: the source rounds to its branch.
            let screen = s * (1.0 - b) + b;
            let twice = b * 2.0;
            let high = -twice * s + (2.0 * screen + -1.0);
            rne(s) * high + s * twice
        }
        14 => {
            // VIVID_LIGHT
            let burn = if 0.0 < s { ((b + -1.0) * 0.5 + s) / s } else { 0.0 };
            let dodge = if s < 1.0 { (b * 0.5) / (1.0 - s) } else { 1.0 };
            sat(if s < 0.5 { burn } else { dodge })
        }
        15 => sat((s + s + b) + -1.0), // LINEAR_LIGHT
        16 => {
            // PIN_LIGHT
            let hi = (s * 2.0 + -1.0).max(b);
            let lo = b.min(s + s);
            rne(s) * (hi - lo) + lo
        }
        17 => {
            // HARD_MIX
            let sum = s + b;
            let one = b >= s || f32::from_bits(0x3F80_20C5) < sum;
            if sum < f32::from_bits(0x3F7F_BE77) { 0.0 } else if one { 1.0 } else { 0.0 }
        }
        18 => (s - b).abs(),   // DIFFERENCE
        24 => sat(b - s),      // SUBTRACT
        25 => sat(b / s) * b.ceil(), // DIVISION
        _ => s,
    }
}

/// HUE and SATURATION: `base`'s hue scaled to the saturation `sat_of` and lit to the backdrop's luminosity, as the
/// shaders build it: the max/min/mid channel picked by `floor(1 − distance)` masks, the result's clipping chosen by
/// arithmetic selects. HUE's base is the source as sampled (premultiplied); SATURATION's the straight backdrop.
fn hue_sat(base: [f32; 3], sat_of: [f32; 3], l_of: [f32; 3]) -> [f32; 3] {
    let (mn, mx) = (|c: [f32; 3]| c[0].min(c[1].min(c[2])), |c: [f32; 3]| c[0].max(c[1].max(c[2])));
    let (s_b, s_c) = (mx(sat_of) - mn(sat_of), mx(base) - mn(base));
    if !(0.0 < s_b * s_c) {
        let l = lum(l_of);
        return [l; 3];
    }
    let (cmax, cmin) = (mx(base), mn(base));
    let ismax0: [f32; 3] = core::array::from_fn(|i| (1.0 - (cmax - base[i])).floor());
    let ismin0: [f32; 3] = core::array::from_fn(|i| (1.0 - (base[i] - cmin)).floor());
    let ismax: [f32; 3] = core::array::from_fn(|i| sat(ismax0[i] - ismax0[(i + 1) % 3]));
    let ismin: [f32; 3] = core::array::from_fn(|i| sat(ismin0[i] - ismin0[(i + 1) % 3]));
    let mid: [f32; 3] = core::array::from_fn(|i| (1.0 - ismax[i]) - ismin[i]);
    let ratio = ((mid[0] * base[0] + mid[1] * base[1] + mid[2] * base[2]) - cmin) / s_c;
    let unit: [f32; 3] = core::array::from_fn(|i| mid[i] * ratio + ismax[i]);
    let scaled: [f32; 3] = core::array::from_fn(|i| s_b * unit[i]);
    let l = lum(l_of);
    let lc = lum(scaled);
    let d = l - lc;
    let c: [f32; 3] = core::array::from_fn(|i| unit[i] * s_b + d);
    let xl = s_b - lc;
    let low: [f32; 3] = core::array::from_fn(|i| l + (l * (c[i] - l)) / lc);
    let high: [f32; 3] = core::array::from_fn(|i| l + ((1.0 - l) * (c[i] - l)) / xl);
    let keep_hi = if (1.0 - l) >= xl { 1.0 } else { 0.0 };
    let keep_lo = if d >= 0.0 { 1.0 } else { 0.0 };
    core::array::from_fn(|i| {
        let v = keep_hi * (c[i] - high[i]) + high[i];
        keep_lo * (v - low[i]) + low[i]
    })
}

/// COLOR and LUMINOSITY: `c` (`base` shifted by the luminosity difference) pulled toward the luminosity `l` by the
/// largest clip factor any channel past `[0, 1]` needs, in the shaders' channel order.
fn set_lum_clip(base: [f32; 3], l_base: f32, l: f32) -> [f32; 3] {
    let d = -l_base + l;
    let c: [f32; 3] = core::array::from_fn(|i| d + base[i]);
    let to_l: [f32; 3] = core::array::from_fn(|i| -c[i] + l);
    let over: [f32; 3] = core::array::from_fn(|i| (1.0 - c[i]) / to_l[i]);
    let under: [f32; 3] = core::array::from_fn(|i| -c[i] / to_l[i]);
    let mut t = if 1.0 < c[0] { over[0].max(0.0) } else { 0.0 };
    t = if c[0] < 0.0 { under[0].max(0.0) } else { t };
    for i in 1..3 {
        let o = if 1.0 < c[i] { over[i].max(t) } else { t };
        t = if c[i] < 0.0 { under[i].max(t) } else { o };
    }
    core::array::from_fn(|i| t * to_l[i] + c[i])
}

/// `D2D1Blend` — `mode` is `D2D1_BLEND_MODE` (0..=25). `dst` = input 0 (backdrop), `src` = input 1
/// (source), both premultiplied. `scene_pos` is the output pixel's scene position (device pixel
/// centre), used only by Dissolve. Each mode is its native shader op for op.
pub fn blend(mode: u32, dst: Rgba, src: Rgba, scene_pos: [f32; 2]) -> Rgba {
    let (cb, cs) = (dst, src);
    match mode {
        4 => dissolve(dst, src, scene_pos),
        // SCREEN, DARKEN, LIGHTEN and EXCLUSION run on the premultiplied colours.
        1 => {
            let a = alpha(dst, src);
            [cs[0] * (1.0 - cb[0]) + cb[0], cs[1] * (1.0 - cb[1]) + cb[1], cs[2] * (1.0 - cb[2]) + cb[2], a]
        }
        2 | 3 => {
            let a = alpha(dst, src);
            let pick = |i: usize| {
                let s_over = (1.0 - src[3]) * cb[i] + cs[i];
                let b_over = (1.0 - dst[3]) * cs[i] + cb[i];
                if mode == 2 { b_over.min(s_over) } else { b_over.max(s_over) }
            };
            [pick(0), pick(1), pick(2), a]
        }
        19 => {
            let a = alpha(dst, src);
            let e = |i: usize| -(cb[i] * cs[i]) * 2.0 + (cb[i] + cs[i]);
            [e(0), e(1), e(2), a]
        }
        7 | 8 => {
            let (b, s) = (straight(dst), straight(src));
            let (lb, ls) = (lum(b), lum(s));
            let take_s = if mode == 7 { ls < lb } else { lb < ls };
            tail(dst, src, if take_s { s } else { b })
        }
        20 => tail(dst, src, hue_sat([cs[0], cs[1], cs[2]], straight(dst), straight(dst))),
        21 => tail(dst, src, hue_sat(straight(dst), straight(src), straight(dst))),
        22 => {
            let (b, s) = (straight(dst), straight(src));
            tail(dst, src, set_lum_clip(s, lum(s), lum(b)))
        }
        23 => {
            let (b, s) = (straight(dst), straight(src));
            tail(dst, src, set_lum_clip(b, lum(b), lum(s)))
        }
        _ => {
            let (b, s) = (straight(dst), straight(src));
            tail(dst, src, core::array::from_fn(|i| separable(mode, b[i], s[i])))
        }
    }
}

/// Dissolve (mode 4): `out = (src.a >= hash(scenePos) && src.a > 0) ? (straightSrc, 1) : dst`,
/// as native Direct2D's Blend Dissolve pixel shader computes it. `dst`/`src` are the premultiplied inputs; `scene_pos` is the
/// output pixel centre `(x+0.5, y+0.5)`, which the shader uses as the noise coordinate. The chosen
/// source colour is unpremultiplied (`div`) with alpha forced to 1, exactly as the shader does.
fn dissolve(dst: Rgba, src: Rgba, scene_pos: [f32; 2]) -> Rgba {
    let h = dissolve_hash(scene_pos[0], scene_pos[1]);
    if src[3] >= h && src[3] > 0.0 {
        let a = src[3];
        [src[0] / a, src[1] / a, src[2] / a, 1.0]
    } else {
        dst
    }
}

/// The Dissolve positional noise, reproduced instruction-for-instruction from the native shader: a
/// chain of `frc(dot4(state, C))` seeded from the scene position `(u, v)`, each `frc` fed back into
/// the state in round-robin `x,y,z,w` order (9 rounds). `C` is the shader's `dp4` immediate; `frc`
/// and the left-to-right `dp4` are WARP's (`crate::warpmath`).
fn dissolve_hash(u: f32, v: f32) -> f32 {
    // dp4 immediate operand of the native shader (hash constants).
    const C: [f32; 4] = [
        f32::from_bits(0x42c2_d174), // 97.40909
        f32::from_bits(0x425a_6481), // 54.59815
        f32::from_bits(0x4260_d257), // 56.20541
        f32::from_bits(0x4232_c050), // 44.687805
    ];
    let frc = crate::warpmath::frc;
    let dp4 = |a: [f32; 4]| a[0] * C[0] + a[1] * C[1] + a[2] * C[2] + a[3] * C[3];
    let mut r = [0.0f32; 4];
    r[0] = frc(dp4([u, v, u, v]));
    r[2] = u;
    r[3] = v;
    r[1] = frc(dp4([r[0], r[3], r[2], r[3]]));
    r[2] = frc(dp4(r));
    r[3] = frc(dp4(r));
    r[0] = frc(dp4(r));
    r[1] = frc(dp4(r));
    r[2] = frc(dp4(r));
    r[3] = frc(dp4(r));
    frc(dp4(r))
}

/// `D2D1ArithmeticComposite` — `out = c.x·dst·src + c.y·dst + c.z·src + c.w` per channel on the
/// premultiplied inputs (`dst` = input 0, `src` = input 1); `saturate` when `clamp_output`.
pub fn arithmetic_composite(coefficients: [f32; 4], clamp_output: bool, dst: Rgba, src: Rgba) -> Rgba {
    let [cx, cy, cz, cw] = coefficients;
    let mut out = [0.0f32; 4];
    for i in 0..4 {
        let r = (cx * dst[i] * src[i]) + (cy * dst[i]) + (cz * src[i]) + cw;
        out[i] = if clamp_output { sat(r) } else { r };
    }
    out
}

/// `D2D1CrossFade` — `out = w·dst + (1−w)·src` per channel on the premultiplied inputs
/// (`dst` = input 0, `src` = input 1): the weight is input 0's, as Direct2D applies it.
pub fn cross_fade(weight: f32, dst: Rgba, src: Rgba) -> Rgba {
    let mut out = [0.0f32; 4];
    for i in 0..4 {
        out[i] = (weight * dst[i]) + ((1.0 - weight) * src[i]);
    }
    out
}

/// `D2D1Opacity` — multiply the premultiplied input by `opacity` (clamped to `[0, 1]`).
pub fn opacity(opacity: f32, input: Rgba) -> Rgba {
    let o = sat(opacity);
    [input[0] * o, input[1] * o, input[2] * o, input[3] * o]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dissolve_hash_matches_native_shader() {
        // Pinned against the native Blend Dissolve shader run on the oracle's scene positions.
        assert_eq!(dissolve_hash(0.5, 0.5).to_bits(), 0x3e51f400);
        assert_eq!(dissolve_hash(2.5, 2.5).to_bits(), 0x3f07e080);
        assert_eq!(dissolve_hash(3.5, 2.5).to_bits(), 0x3f223000);
    }

    #[test]
    fn dissolve_picks_source_when_alpha_clears_hash() {
        // (2.5,2.5): hash≈0.531; src.a=200/255≈0.784 ≥ hash ⇒ straight src, opaque.
        let src = [160.0 / 255.0, 160.0 / 255.0, 0.0, 200.0 / 255.0];
        let dst = [0.0, 0.0, 0.0, 1.0];
        let out = dissolve(dst, src, [2.5, 2.5]);
        assert_eq!(out[3], 1.0);
        assert!((out[0] - src[0] / src[3]).abs() < 1e-6);
        // (1.5,2.5): hash≈0.870 > src.a ⇒ backdrop passes through.
        assert_eq!(dissolve(dst, src, [1.5, 2.5]), dst);
    }
}
