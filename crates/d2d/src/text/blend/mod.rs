//! Direct2D's text blend, exactly as Direct2D + WARP composite a DirectWrite coverage
//! texture with a gamma/enhanced-contrast-corrected foreground colour.
//!
//! This is the back end of glyph rendering *after* coverage exists: DirectWrite's
//! `IDWriteGlyphRunAnalysis` produces an alpha texture (1 byte/pixel grayscale, or 3 bytes/pixel
//! B,G,R ClearType), and Direct2D turns the foreground colour + rendering params into a 256-entry
//! lookup table, samples it by coverage, and composites it over the
//! target, matching Direct2D:
//!
//! * The per-coverage gamma curve, a 2-byte
//!   entry `(base, contrast-delta)` built from a 4-ratio row times the two fixed constants
//!   `4.0314341` / `4.0156865` ([`gamma_table`]). The row index comes from the blend gamma
//!   (`D2D1_GAMMA`: 1.0 → row 0 = identity, 2.2 → row 12).
//! * The lookup-table build: the enhanced-contrast curve
//!   `ec[i] = round(i·(1+k)/(1 + k·i/255))` from the contrast scalar `k`, then each LUT entry =
//!   `premul_fg · (base + (ecFactor·delta)>>8)`, 8-bit round-half-up throughout
//!   ([`TextLut::grayscale`]). The ClearType sibling is per-channel
//!   coverage, result weighted by foreground alpha, dual-source over the target.
//! * The light-on-dark alpha-blend params: the contrast scalar is `enhancedContrast · clamp(3 − 4·L, 0, 1)` with
//!   luminance `L = 0.30·R + 0.59·G + 0.11·B`; this is the only place the foreground colour's
//!   luminance enters ([`BlendParams::from`]).
//!
//! The composite is WARP's fixed-point premultiplied source-over, `out = src + round(dst·(255−a)/255)`
//! (the same `unorm_mul` the rest of the pixel pipeline uses). Target is premultiplied
//! `B8G8R8A8_UNORM` — the only format the oracle exercises; other targets/alpha modes are a table of
//! output conversions away but have no oracle scene yet (see the final report).
//!
//! Coverage bytes are fed straight from `IDWriteGlyphRunAnalysis::CreateAlphaTexture`: the resolved
//! byte equals the value native's own internal coverage resolves to, so the LUT indexes it directly
//! (verified byte-for-byte against `DrawGlyphRun` on WARP — see `tests/text_blend`).

use d2d_types::contracts::PixelBuffer;

/// `D2D1_TEXT_ANTIALIAS_MODE_ALIASED`: the one AA value the grayscale blend branches on — it skips
/// the gamma/contrast correction (binary coverage). DEFAULT(0)/CLEARTYPE(1)/GRAYSCALE(2) all take
/// the corrected path.
pub const AA_ALIASED: u32 = 3;

/// The two fixed multipliers applied to the ratio row:
/// ratios 0/2 (the contrast-delta curve) scale by the first, 1/3 (the base curve)
/// by the second.
const GAMMA_C0: f32 = 4.0314341;
const GAMMA_C1: f32 = 4.0156865;

/// Round half up (`floor(x + 0.5)`), the net effect of native's round-to-nearest conversions plus
/// the `== -0.5` tie-up correction that appears throughout the LUT build.
#[inline]
fn rhu(x: f32) -> i32 {
    (x + 0.5).floor() as i32
}

/// WARP's UNORM8 multiply `round(v·m/255)` = `(t + (t>>8)) >> 8`, `t = v·m + 128` — the identity the
/// whole fixed-point pixel pipeline uses (see `bitmap::unorm_mul`, verified in the sampler dump).
#[inline]
fn unorm_mul(v: i32, m: i32) -> i32 {
    let t = v * m + 128;
    (t + (t >> 8)) >> 8
}

/// `fg·255` rounded to nearest-even — WARP's float→UNORM8 rule for the ClearType foreground byte.
#[inline]
fn to_unorm8(c: f32) -> i32 {
    (c * 255.0).round_ties_even().clamp(0.0, 255.0) as i32
}

/// ClearType's SOURCE_OVER merge at one byte channel: the blend factor `fg` (the colour byte) times the
/// weight `w`, plus `dst·(255−w)`. WARP's output merger (from the JIT kernel) takes each product's
/// `/255` as `t + (t>>8)`, sums both, adds 128 and shifts by 8, all in 16-bit lanes. Close to
/// `(fg·w + dst·(255−w) + 126) / 255`, but a remainder-128 tie can round up (r20 25023).
#[inline]
fn lerp8(fg: i32, w: u8, dst: u8) -> u8 {
    let w = w as i32;
    let (t1, t2) = (fg * w, dst as i32 * (255 - w));
    ((t1 + (t1 >> 8) + t2 + (t2 >> 8) + 128) >> 8).clamp(0, 255) as u8
}

/// DirectWrite's per-gamma "incorrect target" correction ratios, gamma 1.0 (row 0) to 2.2 (row 12) in
/// 0.1 steps. Sourced verbatim (as `value / 4.0`) from Microsoft's MIT-licensed Windows Terminal
/// (`src/renderer/atlas/dwrite.cpp`, `gammaIncorrectTargetRatios`; Copyright (c) Microsoft Corporation,
/// Licensed under the MIT license — recorded in the repository's third-party license notes). Per-row
/// order is `[a1, b1, a2, b2]` as [`gamma_table`] consumes it (base curve ← indices 1/3, contrast-delta
/// ← 0/2). Row 12 (gamma 2.2) reproduces the previously hand-entered values bit-for-bit.
#[rustfmt::skip]
const GAMMA_INCORRECT_TARGET_RATIOS: [[f32; 4]; 13] = [
    [0.0000 / 4.0,  0.0000 / 4.0, 0.0000 / 4.0,  0.0000 / 4.0], // gamma 1.0
    [0.0166 / 4.0, -0.0807 / 4.0, 0.2227 / 4.0, -0.0751 / 4.0], // gamma 1.1
    [0.0350 / 4.0, -0.1760 / 4.0, 0.4325 / 4.0, -0.1370 / 4.0], // gamma 1.2
    [0.0543 / 4.0, -0.2821 / 4.0, 0.6302 / 4.0, -0.1876 / 4.0], // gamma 1.3
    [0.0739 / 4.0, -0.3963 / 4.0, 0.8167 / 4.0, -0.2287 / 4.0], // gamma 1.4
    [0.0933 / 4.0, -0.5161 / 4.0, 0.9926 / 4.0, -0.2616 / 4.0], // gamma 1.5
    [0.1121 / 4.0, -0.6395 / 4.0, 1.1588 / 4.0, -0.2877 / 4.0], // gamma 1.6
    [0.1300 / 4.0, -0.7649 / 4.0, 1.3159 / 4.0, -0.3080 / 4.0], // gamma 1.7
    [0.1469 / 4.0, -0.8911 / 4.0, 1.4644 / 4.0, -0.3234 / 4.0], // gamma 1.8  (COLR paint-tree leaf)
    [0.1627 / 4.0, -1.0170 / 4.0, 1.6051 / 4.0, -0.3347 / 4.0], // gamma 1.9
    [0.1773 / 4.0, -1.1420 / 4.0, 1.7385 / 4.0, -0.3426 / 4.0], // gamma 2.0
    [0.1908 / 4.0, -1.2652 / 4.0, 1.8650 / 4.0, -0.3476 / 4.0], // gamma 2.1
    [0.2031 / 4.0, -1.3864 / 4.0, 1.9851 / 4.0, -0.3501 / 4.0], // gamma 2.2  (main text path)
];

/// The `(base, contrast-delta)` gamma curve for a `D2D1_GAMMA` row.
/// `base(c) = ((c·b1 + b2)·c·(1−c) + c)·255`,
/// `delta(c) = ((c·a1 + a2)·c·(1−c))·255`, both round-half-up, `c = i/255`.
///
/// `gamma_row` indexes [`GAMMA_INCORRECT_TARGET_RATIOS`]: gamma 1.0 (row 0) … 2.2 (row 12) in 0.1
/// steps. Row 12 (gamma 2.2) is the main grayscale/ClearType text path; row 8 (gamma 1.8) is
/// DirectWrite's default — the gamma the COLR paint-tree leaf blend uses (d2d1 `FillRows` capture
/// confirmed the leaf's LUT is byte-for-byte this row, see `text/color`).
fn gamma_table(gamma_row: u32) -> [[u8; 2]; 256] {
    let ratios = GAMMA_INCORRECT_TARGET_RATIOS[(gamma_row as usize).min(12)];
    let (a1, a2) = (ratios[0] * GAMMA_C0, ratios[2] * GAMMA_C0);
    let (b1, b2) = (ratios[1] * GAMMA_C1, ratios[3] * GAMMA_C1);
    let mut t = [[0u8; 2]; 256];
    for (i, e) in t.iter_mut().enumerate() {
        let c = i as f32 * 0.0039215689;
        let base = ((c * b1 + b2) * ((1.0 - c) * c) + c) * 255.0;
        let delta = ((c * a1 + a2) * ((1.0 - c) * c)) * 255.0;
        *e = [rhu(base) as u8, rhu(delta) as u8];
    }
    t
}

/// The COLR paint-tree leaf's grayscale gamma blend weight: coverage `cov` corrected for the straight
/// foreground `fg` (RGBA, 0..1), returned as the 0..255 weight to multiply the premultiplied source by
/// (matching `TextLut::grayscale`'s `premul_fg · corrected`). Native renders each paint-tree leaf
/// through a fresh internal DC whose DWrite defaults are **gamma 1.8 (row 8)** and grayscale enhanced
/// contrast **1.0** — confirmed by capturing Direct2D's leaf-draw lookup table
/// (its LUT is this row, vs the main text path's row 12 / gamma 2.2, on byte-identical dwrite coverage).
/// `cov` varies per pixel; `fg` too across a gradient leaf, so this computes one entry, not a LUT.
pub(crate) fn leaf_gamma_coverage(cov: u8, fg: [f32; 4]) -> i32 {
    use std::sync::OnceLock;
    static G8: OnceLock<[[u8; 2]; 256]> = OnceLock::new();
    let g = G8.get_or_init(|| gamma_table(8));
    // Contrast scalar = enhanced_contrast(1.0) · luminance ramp (as GetAlphaBlendParams...), ec_factor
    // = round(intensity·255), intensity = ¼R+½G+¼B — identical to `BlendParams::resolve`/`TextLut`.
    let l = 0.30000001 * fg[0] + 0.58999997 * fg[1] + 0.11 * fg[2];
    let k = if l >= 0.75 { 0.0 } else if l > 0.5 { (0.75 - l) * 4.0 } else { 1.0 };
    let ef = rhu((0.25 * fg[0] + 0.5 * fg[1] + 0.25 * fg[2]) * 255.0);
    let c = cov as f32 * 0.0039215689;
    let ecv = if cov == 0 {
        0usize
    } else if cov == 255 {
        255
    } else {
        ((((k + 1.0) * c) / (c * k + 1.0) * 255.0 + 0.5).floor() as usize).min(255)
    };
    let [base, delta] = g[ecv];
    base as i32 + ((ef * delta as i32) >> 8)
}

/// Gradient glyphs use a fixed-contrast mask and apply gamma to the sampled brush
/// colour in the pixel shader. Unlike solid text, this correction is not a byte LUT.
pub(crate) fn leaf_brush_coverage(cov: u8, fg: [f32; 4]) -> f32 {
    let c = contrast_at(1.0, cov) as f32;
    let c = c * (1.0 / 256.0) + c * (1.0 / 65280.0);
    let intensity = (fg[0] * 0.25 + fg[2] * 0.25) + fg[1] * 0.5;
    let [a1, b1, a2, b2] = GAMMA_INCORRECT_TARGET_RATIOS[8];
    let a = a1 * intensity + b1;
    let b = a2 * intensity + b2;
    ((a * c + b) * ((1.0 - c) * c)) * 4.0 + c
}

/// The enhanced-contrast curve `FillRows` builds from the contrast scalar `k`:
/// `ec[0]=0`, `ec[255]=255`, `ec[i] = floor(i·(1+k)/(1 + k·i/255) + 0.5)` for the rest.
fn contrast_curve(k: f32) -> [u8; 256] {
    let mut ec = [0u8; 256];
    for i in 1..255 {
        let c = i as f32 * 0.0039215689;
        ec[i] = (((k + 1.0) * c) / (c * k + 1.0) * 255.0 + 0.5).floor() as u8;
    }
    ec[255] = 255;
    ec
}

/// Single-index enhanced-contrast value (see [`contrast_curve`]) without building the 256-table —
/// for the per-pixel-brush text blends, whose contrast scalar `k` varies with the sampled colour.
fn contrast_at(k: f32, i: u8) -> u8 {
    match i {
        0 => 0,
        255 => 255,
        _ => {
            let c = i as f32 * 0.0039215689;
            ((((k + 1.0) * c) / (c * k + 1.0)) * 255.0 + 0.5).floor() as u8
        }
    }
}

/// The three scalars Direct2D's blend reads from the active rendering params, resolved for one
/// foreground colour the way `GetAlphaBlendParamsWithLightOnDarkAdjustment` does.
#[derive(Clone, Copy, Debug)]
pub struct BlendParams {
    /// `D2D1_GAMMA` row for the gamma table (1.0 → 0, 2.2 → 12).
    pub gamma_row: u32,
    /// Enhanced-contrast scalar `k` already multiplied by the luminance ramp `clamp(3−4L,0,1)`.
    pub contrast: f32,
    /// The same `k` without the ramp: a non-SOURCE_OVER blend's coverage curve (see [`blend_grayscale`]).
    pub enhanced_contrast: f32,
    /// ClearType level (1.0 = full ClearType, 0 = grayscale intensity). Grayscale ignores it.
    pub cleartype_level: f32,
}

impl BlendParams {
    /// Resolve for `fg` (straight RGBA, the foreground brush colour). `gamma` is the blend gamma
    /// (2.2 / 1.0), `enhanced_contrast` and `cleartype_level` come from the rendering params. The
    /// luminance ramp — the only colour dependence of the contrast scalar — is native's
    /// `L = 0.30·R + 0.59·G + 0.11·B`, `clamp(4·(0.75−L), 0, 1)`.
    pub fn resolve(fg: [f32; 4], gamma: f32, enhanced_contrast: f32, cleartype_level: f32) -> BlendParams {
        let l = 0.30000001 * fg[0] + 0.58999997 * fg[1] + 0.11 * fg[2];
        let ramp = if l >= 0.75 {
            0.0
        } else if l > 0.5 {
            (0.75 - l) * 4.0
        } else {
            1.0
        };
        BlendParams {
            gamma_row: gamma_row_for(gamma),
            contrast: enhanced_contrast * ramp,
            enhanced_contrast,
            cleartype_level,
        }
    }
}

/// `D2D1_GAMMA` → gamma-table row: the native index `clamp(round(gamma·10), 10, 22) − 10`.
fn gamma_row_for(gamma: f32) -> u32 {
    ((gamma * 10.0).round() as i32).clamp(10, 22) as u32 - 10
}

/// The per-colour grayscale lookup table: 256 premultiplied BGRA entries, one per coverage byte,
/// exactly as Direct2D fills them. Built once per run, then sampled per pixel.
struct TextLut {
    /// `premul_fg · corrected_coverage`, BGRA8, indexed by the coverage byte.
    lut: [[u8; 4]; 256],
}

impl TextLut {
    fn grayscale(fg: [f32; 4], bp: &BlendParams) -> TextLut {
        let gamma = gamma_table(bp.gamma_row);
        let ec = contrast_curve(bp.contrast);
        // Premultiplied foreground (straight → premultiplied); the LUT holds premul_fg · coverage.
        let a = fg[3];
        let premul = [fg[2] * a, fg[1] * a, fg[0] * a, a]; // B, G, R, A
        // Grayscale enhanced-contrast factor = round(intensity·255), intensity = ¼R + ½G + ¼B.
        let intensity = 0.25 * fg[0] + 0.5 * fg[1] + 0.25 * fg[2];
        let ec_factor = rhu(intensity * 255.0);
        let mut lut = [[0u8; 4]; 256];
        for (cov, out) in lut.iter_mut().enumerate() {
            let [base, delta] = gamma[ec[cov] as usize];
            let corrected = (base as i32 + ((ec_factor * delta as i32) >> 8)) as f32;
            *out = core::array::from_fn(|c| rhu(premul[c] * corrected).clamp(0, 255) as u8);
        }
        TextLut { lut }
    }
}

/// A coverage texture `IDWriteGlyphRunAnalysis::CreateAlphaTexture` produced: `bytes` is the alpha
/// data at `[left, top, right, bottom]` device pixels (`bpp` bytes per pixel: 1 grayscale, 3
/// ClearType B,G,R). Rows are tightly packed, `width·bpp` bytes each.
#[derive(Clone, Copy)]
pub struct AlphaTexture<'a> {
    pub bytes: &'a [u8],
    pub bounds: [i32; 4],
}

impl AlphaTexture<'_> {
    fn width(&self) -> i32 {
        self.bounds[2] - self.bounds[0]
    }
    fn height(&self) -> i32 {
        self.bounds[3] - self.bounds[1]
    }
}

/// Composite `src` (premultiplied BGRA) over the pixel at `(x, y)` in a premultiplied `B8G8R8A8`
/// target, WARP's fixed-point source-over.
#[inline]
fn over(target: &mut PixelBuffer, x: i32, y: i32, src: [u8; 4]) {
    if x < 0 || y < 0 || x >= target.desc.width as i32 || y >= target.desc.height as i32 {
        return;
    }
    if float_px(target, x, y, src, crate::pixels::Blend::SourceOver) {
        return;
    }
    let i = y as usize * target.stride + x as usize * 4;
    let dst = &mut target.data[i..i + 4];
    let ia = 255 - src[3] as i32;
    for c in 0..4 {
        dst[c] = (src[c] as i32 + unorm_mul(dst[c] as i32, ia)).clamp(0, 255) as u8;
    }
}

/// SOURCE_OVER for a brush-text pixel: the gradient-text shader's output goes through WARP's output merger
/// (ONE, INV_SRC_ALPHA), which rounds `d·ia/255` as `(t + (t>>8) + 128) >> 8`, the form the bitmap, layer,
/// ink and ClearType-brush composites use. Solid text keeps [`over`]'s `unorm_mul`. They differ on 24 of the
/// 65536 pairs: r16 21584 (radial brush) gives 239 over dest alpha 229, where `unorm_mul` gives 240, while
/// solid seeds (r3 2558, r8 9687, ...) need `unorm_mul`.
fn over_om(target: &mut PixelBuffer, x: i32, y: i32, src: [u8; 4]) {
    if x < 0 || y < 0 || x >= target.desc.width as i32 || y >= target.desc.height as i32 {
        return;
    }
    if float_px(target, x, y, src, crate::pixels::Blend::SourceOver) {
        return;
    }
    let i = y as usize * target.stride + x as usize * 4;
    let dst = &mut target.data[i..i + 4];
    let ia = 255 - src[3] as i32;
    for c in 0..4 {
        let t = dst[c] as i32 * ia;
        dst[c] = (src[c] as i32 + ((t + (t >> 8) + 128) >> 8)).clamp(0, 255) as u8;
    }
}

/// Composite one premultiplied-BGRA text pixel under the `SetPrimitiveBlend` mode. SOURCE_OVER keeps
/// the byte-exact text [`over`]; COPY/MIN/ADD/MAX reuse the fill composite at full coverage
/// (`pixels::composite(.., 64, ..)` → `c = 1`, so COPY replaces the destination and the others go
/// per-channel on the 8-bit premultiplied source) — the same path fills use, so text now honours the
/// primitive blend like every other primitive (canvas `fillText` under `globalCompositeOperation`).
fn composite_px(target: &mut PixelBuffer, x: i32, y: i32, src: [u8; 4], blend: crate::pixels::Blend) {
    if matches!(blend, crate::pixels::Blend::SourceOver) {
        return over(target, x, y, src);
    }
    if x < 0 || y < 0 || x >= target.desc.width as i32 || y >= target.desc.height as i32 {
        return;
    }
    if float_px(target, x, y, src, blend) {
        return;
    }
    let i = y as usize * target.stride + x as usize * 4;
    let d = &mut target.data[i..i + 4];
    let out = crate::pixels::composite([d[0], d[1], d[2], d[3]], src, 64, blend);
    d.copy_from_slice(&out);
}

/// Composite one grayscale/brush text pixel under a non-SOURCE_OVER primitive blend. `src_full` is the
/// FULL-coverage premultiplied foreground (no gamma, no coverage baked in) and `cov` the coverage
/// byte the caller already ran through the enhanced-contrast curve: native lerps the dest toward the
/// blended opaque source by it — COPY = `src·c + dst·(1−c)`, MIN/ADD/MAX = `mode(src,dst)·c +
/// dst·(1−c)` (the ctb/wb oracle rule). The gamma LUT only drives SOURCE_OVER, so the earlier
/// "replace with the gamma-premul × coverage" path wrongly dropped partial-coverage edges to a low
/// alpha instead of keeping them blended over the dest.
///
/// `src_f` is the foreground's unquantised premultiplied BGRA (0..255; a solid colour or the brush sample): the text shader multiplies
/// the float colour constant by the coverage and the output merger MIN/ADD/MAXes that float against
/// the dest, rounding once (r15 20504: 'Ag' under MAX, alpha 113.41·c not 113·c).
fn composite_cov_px(target: &mut PixelBuffer, x: i32, y: i32, src_full: [u8; 4], src_f: Option<[f32; 4]>, cov: u8, blend: crate::pixels::Blend) {
    use crate::pixels::Blend;
    if x < 0 || y < 0 || x >= target.desc.width as i32 || y >= target.desc.height as i32 {
        return;
    }
    if crate::float::is_highp(target.desc.format) {
        // ponytail: a float target takes the byte source at this coverage through the f32 blend, not native's float
        // shader (no float-target oracle under a primitive blend yet).
        let c = cov as u32 * 64 / 255;
        let src = [src_full[2], src_full[1], src_full[0], src_full[3]].map(|v| v as f32 / 255.0);
        crate::float::FloatTarget::new(target).blend_px(x as usize, y as usize, src, c, blend);
        return;
    }
    let i = y as usize * target.stride + x as usize * 4;
    let d = &mut target.data[i..i + 4];
    let dst = [d[0], d[1], d[2], d[3]];
    let re = |v: f32| v.round_ties_even().clamp(0.0, 255.0) as u8;
    let c = cov as f32 / 255.0;
    // COPY is the fused high-byte lerp, as the COPY fills' kernel (w951 probes over a coloured dest).
    let out: [u8; 4] = match (blend, src_f.map(|f| f.map(|v| v * c))) {
        (Blend::Copy, _) => crate::pixels::copy_tess::lerp8(dst, src_full, cov as i32),
        (Blend::Max, Some(cs)) => core::array::from_fn(|k| re(cs[k]).max(dst[k])),
        (Blend::Add, Some(cs)) => core::array::from_fn(|k| re(cs[k] + dst[k] as f32)),
        (Blend::Min, Some(cs)) => core::array::from_fn(|k| re(cs[k] + (255.0 - cs[3])).min(dst[k])),
        _ => crate::pixels::blend_cov(dst, src_full, c, blend),
    };
    d.copy_from_slice(&out);
}

/// Composite one ClearType pixel. `fg_bgr`/`w_bgr` are the straight foreground bytes and per-subpixel
/// weights in B,G,R order; SOURCE_OVER is the per-channel [`lerp8`] (dual-source, dest alpha preserved).
///
/// Under a non-SOURCE_OVER `SetPrimitiveBlend` ClearType has no per-subpixel destination to lerp against
/// and no separate alpha channel, so native falls back to the GRAYSCALE coverage: the green subpixel
/// weight `w_bgr[1]` is CT's coverage·fgAlpha (it is also the weight CT's merger writes to the alpha
/// channel), so the source is the single premultiplied `fg·vg` = `[fgB·wg, fgG·wg, fgR·wg, wg]`, run
/// through the SAME primitive-blend composite the grayscale and fill paths use. Verified byte-exact on
/// fuzz r6 6577 (CT 'Ag' under COPY): (5,47) = `[41,85,110,138]` = `fg_premul` at full coverage. This is
/// the one-rule-for-all-text form; the earlier premul-`fg·w` per-channel attempt (that made r5 5596
/// worse) was wrong because it kept the three independent weights instead of collapsing to vg.
fn composite_ct_px(target: &mut PixelBuffer, x: i32, y: i32, fg_bgr: [i32; 3], w_bgr: [u8; 3], blend: crate::pixels::Blend) {
    if crate::float::is_highp(target.desc.format) {
        // ponytail: ClearType on a float target lerps each channel by its byte weight in f32; native's float-target
        // ClearType (atlasplan sc29 scene 6) is per channel too but its exact weights are not ported yet.
        let mut ft = crate::float::FloatTarget::new(target);
        let mut d = ft.load_px(x as usize, y as usize);
        for (c, k) in [(0, 2), (1, 1), (2, 0)] {
            let w = w_bgr[k] as f32 / 255.0;
            d[c] = fg_bgr[k] as f32 / 255.0 * w + d[c] * (1.0 - w);
        }
        ft.store_px(x as usize, y as usize, d);
        let _ = blend;
        return;
    }
    let i = y as usize * target.stride + x as usize * 4;
    let dst = &mut target.data[i..i + 4];
    if matches!(blend, crate::pixels::Blend::SourceOver) {
        dst[0] = lerp8(fg_bgr[0], w_bgr[0], dst[0]);
        dst[1] = lerp8(fg_bgr[1], w_bgr[1], dst[1]);
        dst[2] = lerp8(fg_bgr[2], w_bgr[2], dst[2]);
        return;
    }
    let wg = w_bgr[1]; // green weight = grayscale coverage · fgAlpha (premultiplied)
    let src = [
        unorm_mul(fg_bgr[0], wg as i32) as u8,
        unorm_mul(fg_bgr[1], wg as i32) as u8,
        unorm_mul(fg_bgr[2], wg as i32) as u8,
        wg,
    ];
    let out = crate::pixels::composite([dst[0], dst[1], dst[2], dst[3]], src, 64, blend);
    dst.copy_from_slice(&out);
}

/// A byte text pixel (premultiplied BGRA) composited onto a float target in f32; false on a byte target.
fn float_px(target: &mut PixelBuffer, x: i32, y: i32, src: [u8; 4], blend: crate::pixels::Blend) -> bool {
    if !crate::float::is_highp(target.desc.format) {
        return false;
    }
    let src = [src[2], src[1], src[0], src[3]].map(|v| v as f32 / 255.0);
    crate::float::FloatTarget::new(target).blend_px(x as usize, y as usize, src, 64, blend);
    true
}

/// The alpha-correction ratios for a gamma-correct target (an R16G16B16A16_FLOAT or R32G32B32A32_FLOAT target, where
/// blending is linear): Direct2D's companion to [`GAMMA_INCORRECT_TARGET_RATIOS`], same rows (gamma 1.0 to 2.2 in 0.1
/// steps) and the same `[a1, b1, a2, b2]` order. Text on such a target is not corrected through the byte table: the
/// coverage through the enhanced-contrast table becomes `a + 4·a·(1−a)·((a1·I + b1)·a + (a2·I + b2))` in f32, I the
/// ink's ¼R + ½G + ¼B. Verified against VM captures at several gammas and contrasts.
#[rustfmt::skip]
const GAMMA_CORRECT_TARGET_RATIOS: [[f32; 4]; 13] = [
    [0.0256 / 4.0, -0.1020 / 4.0, -1.5787 / 4.0, 0.8339 / 4.0], // gamma 1.0
    [0.0105 / 4.0, -0.0428 / 4.0, -1.4340 / 4.0, 0.7358 / 4.0], // gamma 1.1
    [-0.0009 / 4.0, 0.0038 / 4.0, -1.2934 / 4.0, 0.6450 / 4.0], // gamma 1.2
    [-0.0092 / 4.0, 0.0392 / 4.0, -1.1567 / 4.0, 0.5610 / 4.0], // gamma 1.3
    [-0.0147 / 4.0, 0.0644 / 4.0, -1.0238 / 4.0, 0.4834 / 4.0], // gamma 1.4
    [-0.0178 / 4.0, 0.0804 / 4.0, -0.8946 / 4.0, 0.4115 / 4.0], // gamma 1.5
    [-0.0189 / 4.0, 0.0881 / 4.0, -0.7689 / 4.0, 0.3451 / 4.0], // gamma 1.6
    [-0.0182 / 4.0, 0.0882 / 4.0, -0.6467 / 4.0, 0.2838 / 4.0], // gamma 1.7
    [-0.0160 / 4.0, 0.0816 / 4.0, -0.5278 / 4.0, 0.2271 / 4.0], // gamma 1.8
    [-0.0124 / 4.0, 0.0687 / 4.0, -0.4122 / 4.0, 0.1749 / 4.0], // gamma 1.9
    [-0.0078 / 4.0, 0.0503 / 4.0, -0.2998 / 4.0, 0.1267 / 4.0], // gamma 2.0
    [-0.0022 / 4.0, 0.0270 / 4.0, -0.1904 / 4.0, 0.0823 / 4.0], // gamma 2.1
    [0.0042 / 4.0, -0.0009 / 4.0, -0.0840 / 4.0, 0.0414 / 4.0], // gamma 2.2
];

/// A float target's correction coefficients for `gamma_row`, scaled by the 4 the correction carries.
fn float_ratios(gamma_row: u32) -> [f32; 4] {
    GAMMA_CORRECT_TARGET_RATIOS[(gamma_row as usize).min(12)].map(|v| v * 4.0)
}

/// Grayscale / aliased text on a float target: the premultiplied ink times the float-evaluated coverage
/// (aliased: full), composited in f32.
fn blend_grayscale_f(target: &mut PixelBuffer, alpha: AlphaTexture, fg: [f32; 4], bp: &BlendParams, aa_mode: u32, blend: crate::pixels::Blend) {
    let (w, h) = (alpha.width(), alpha.height());
    let ec = contrast_curve(bp.contrast);
    let i = 0.25 * fg[0] + 0.5 * fg[1] + 0.25 * fg[2];
    let [g0, g1, g2, g3] = float_ratios(bp.gamma_row);
    let (ga, gb) = (g0 * i + g1, g2 * i + g3);
    let premul = [fg[0] * fg[3], fg[1] * fg[3], fg[2] * fg[3], fg[3]];
    let (tw, th) = (target.desc.width as i32, target.desc.height as i32);
    let mut ft = crate::float::FloatTarget::new(target);
    for row in 0..h {
        for col in 0..w {
            let cov = alpha.bytes[(row * w + col) as usize];
            let (x, y) = (alpha.bounds[0] + col, alpha.bounds[1] + row);
            if cov == 0 || x < 0 || y < 0 || x >= tw || y >= th {
                continue;
            }
            let v = if aa_mode == AA_ALIASED {
                1.0
            } else {
                let a = ec[cov as usize] as f32 / 255.0;
                a + a * (1.0 - a) * (ga * a + gb)
            };
            ft.blend_px(x as usize, y as usize, premul.map(|c| c * v), 64, blend);
        }
    }
}

/// ClearType on a float target: each subpixel's coverage through the contrast table and the float correction with
/// that channel's ink as I, then the channel lerps toward the ink by it times the ink's alpha; alpha is kept.
fn blend_cleartype_f(target: &mut PixelBuffer, alpha: AlphaTexture, fg: [f32; 4], bp: &BlendParams) {
    let (w, h) = (alpha.width(), alpha.height());
    let ec = contrast_curve(bp.contrast);
    let [g0, g1, g2, g3] = float_ratios(bp.gamma_row);
    let (tw, th) = (target.desc.width as i32, target.desc.height as i32);
    let mut ft = crate::float::FloatTarget::new(target);
    for row in 0..h {
        for col in 0..w {
            let b = ((row * w + col) * 3) as usize;
            let cov = [alpha.bytes[b], alpha.bytes[b + 1], alpha.bytes[b + 2]]; // R, G, B
            let (x, y) = (alpha.bounds[0] + col, alpha.bounds[1] + row);
            if cov == [0; 3] || x < 0 || y < 0 || x >= tw || y >= th {
                continue;
            }
            let mut d = ft.load_px(x as usize, y as usize);
            for c in 0..3 {
                let a = ec[cov[c] as usize] as f32 / 255.0;
                let v = (a + a * (1.0 - a) * ((g0 * fg[c] + g1) * a + (g2 * fg[c] + g3))) * fg[3];
                d[c] = fg[c] * v + d[c] * (1.0 - v);
            }
            ft.store_px(x as usize, y as usize, d);
        }
    }
}

/// Grayscale / aliased text blend (R3 / R5): sample the LUT by coverage, source-over.
///
/// `aa_mode` picks the correction: `AA_GRAYSCALE` (and `AA_DEFAULT`, which resolves to grayscale on
/// our premultiplied target) applies the gamma/contrast LUT; `AA_ALIASED` skips it — binary
/// coverage composites the premultiplied foreground directly.
pub fn blend_grayscale(
    target: &mut PixelBuffer,
    alpha: AlphaTexture,
    fg: [f32; 4],
    bp: &BlendParams,
    aa_mode: u32,
    blend: crate::pixels::Blend,
) {
    let (w, h) = (alpha.width(), alpha.height());
    if w <= 0 || h <= 0 || alpha.bytes.len() < (w * h) as usize {
        return;
    }
    if crate::float::is_highp(target.desc.format) {
        return blend_grayscale_f(target, alpha, fg, bp, aa_mode, blend);
    }
    if aa_mode == AA_ALIASED {
        // The solid colour converts to UNORM8 rounding ties to even (VM geo/alc: 0.3 at alpha 1 is 76, alpha 0.3 is 76).
        let premul = {
            let a = fg[3];
            let u = |v: f32| (v * 255.0).round_ties_even().clamp(0.0, 255.0) as u8;
            [u(fg[2] * a), u(fg[1] * a), u(fg[0] * a), u(a)]
        };
        for row in 0..h {
            for col in 0..w {
                if alpha.bytes[(row * w + col) as usize] != 0 {
                    composite_px(target, alpha.bounds[0] + col, alpha.bounds[1] + row, premul, blend);
                }
            }
        }
        return;
    }
    // A non-over primitive blend lerps the full-coverage foreground over the dest by the coverage
    // through the enhanced-contrast curve alone: no gamma LUT and no luminance ramp, so every colour
    // lerps by the same `contrast_curve(k)[cov]` (w951 probes: CT/grayscale × 5 colours under COPY,
    // native's COPY alphas 57..238 are that curve of the k/8 coverages its SOURCE_OVER LUT implies).
    if !matches!(blend, crate::pixels::Blend::SourceOver) {
        let ec = contrast_curve(bp.enhanced_contrast);
        let a = fg[3];
        let src_f = [fg[2] * a * 255.0, fg[1] * a * 255.0, fg[0] * a * 255.0, a * 255.0];
        let src_full = [
            rhu(fg[2] * a * 255.0).clamp(0, 255) as u8,
            rhu(fg[1] * a * 255.0).clamp(0, 255) as u8,
            rhu(fg[0] * a * 255.0).clamp(0, 255) as u8,
            rhu(a * 255.0).clamp(0, 255) as u8,
        ];
        for row in 0..h {
            for col in 0..w {
                let cov = alpha.bytes[(row * w + col) as usize];
                if cov != 0 {
                    composite_cov_px(target, alpha.bounds[0] + col, alpha.bounds[1] + row, src_full, Some(src_f), ec[cov as usize], blend);
                }
            }
        }
        return;
    }
    let lut = TextLut::grayscale(fg, bp);
    for row in 0..h {
        for col in 0..w {
            let cov = alpha.bytes[(row * w + col) as usize];
            if cov != 0 {
                composite_px(target, alpha.bounds[0] + col, alpha.bounds[1] + row, lut.lut[cov as usize], blend);
            }
        }
    }
}

/// ClearType text blend (R4): per-subpixel coverage, dual-source lerp over the target.
///
/// `CreateAlphaTexture(CLEARTYPE_3x1)` is 3 bytes/pixel in **R, G, B** subpixel order. Per channel
/// the coverage weight is `w = round(corrected · fgAlpha)` from the same gamma/contrast LUT as
/// grayscale, with `ecFactor` the straight foreground of that channel (R for the R subpixel, …).
/// WARP's output merger then blends per channel, `out.c = fg.c·w_c + dst.c·(1 − w_c)`, in 16-bit
/// fixed point ([`lerp8`]), with destination alpha preserved.
/// For the pinned ClearType level 1.0 the channels are independent (no mix toward grayscale).
pub fn blend_cleartype(target: &mut PixelBuffer, alpha: AlphaTexture, fg: [f32; 4], bp: &BlendParams, blend: crate::pixels::Blend) {
    let (w, h) = (alpha.width(), alpha.height());
    if w <= 0 || h <= 0 || alpha.bytes.len() < (w * h * 3) as usize {
        return;
    }
    // At ClearType level 0 the sub-pixel coverage is equal across channels and every channel's weight takes the ink's
    // intensity as its contrast factor (VM, sc33 level-0 scenes); on a float target that is the grayscale blend.
    let level0 = bp.cleartype_level == 0.0;
    if level0 && crate::float::is_highp(target.desc.format) {
        let green: Vec<u8> = alpha.bytes.chunks_exact(3).map(|c| c[1]).collect();
        return blend_grayscale(target, AlphaTexture { bytes: &green, bounds: alpha.bounds }, fg, bp, AA_ALIASED + 1, blend); // 2 = grayscale
    }
    if crate::float::is_highp(target.desc.format) && matches!(blend, crate::pixels::Blend::SourceOver) {
        return blend_cleartype_f(target, alpha, fg, bp);
    }
    let gamma = gamma_table(bp.gamma_row);
    let ec = contrast_curve(bp.contrast);
    let a = fg[3];
    // Per-subpixel weight tables, indexed [0=R, 1=G, 2=B][coverage]; each uses its own foreground.
    let mut wlut = [[0u8; 256]; 3];
    let intensity = 0.25 * fg[0] + 0.5 * fg[1] + 0.25 * fg[2];
    let ec_srcs = if level0 { [intensity; 3] } else { [fg[0], fg[1], fg[2]] };
    for (ch, ec_src) in ec_srcs.into_iter().enumerate() {
        let ec_factor = rhu(ec_src * 255.0);
        for cov in 0..256 {
            let [base, delta] = gamma[ec[cov] as usize];
            let corrected = (base as i32 + ((ec_factor * delta as i32) >> 8)) as f32;
            wlut[ch][cov] = rhu(corrected * a).clamp(0, 255) as u8;
        }
    }
    let fg_byte = [to_unorm8(fg[0]), to_unorm8(fg[1]), to_unorm8(fg[2])]; // R, G, B
    // Under a non-SOURCE_OVER blend the dual-source merger has no per-subpixel dest to lerp against,
    // so ClearType collapses to the single LINEAR grayscale coverage = the green-subpixel coverage
    // `cg` (the pixel-centre column), composited over the dest at full-coverage foreground (ctb/wb
    // oracle: COPY edge = `fg_full · cg/255`, R=G=B for a white fg). CT geometry/placement is kept.
    let non_over = !matches!(blend, crate::pixels::Blend::SourceOver);
    let fg_f = [fg[2] * a * 255.0, fg[1] * a * 255.0, fg[0] * a * 255.0, a * 255.0];
    let fg_full = {
        let a = fg[3];
        [
            rhu(fg[2] * a * 255.0).clamp(0, 255) as u8,
            rhu(fg[1] * a * 255.0).clamp(0, 255) as u8,
            rhu(fg[0] * a * 255.0).clamp(0, 255) as u8,
            rhu(a * 255.0).clamp(0, 255) as u8,
        ]
    };
    for row in 0..h {
        for col in 0..w {
            let base = ((row * w + col) * 3) as usize;
            let (cr, cg, cb) = (alpha.bytes[base], alpha.bytes[base + 1], alpha.bytes[base + 2]);
            if cr == 0 && cg == 0 && cb == 0 {
                continue;
            }
            let (x, y) = (alpha.bounds[0] + col, alpha.bounds[1] + row);
            if x < 0 || y < 0 || x >= target.desc.width as i32 || y >= target.desc.height as i32 {
                continue;
            }
            if non_over {
                // Collapse to the green-subpixel (pixel-centre) coverage: it keeps the interior
                // byte-exact (the mean of all three dilutes it). The residual edge coverage differs
                // from native's grayscale-under-CT footprint — a CT raster-geometry floor, not a
                // blend one.
                composite_cov_px(target, x, y, fg_full, Some(fg_f), cg, blend);
                continue;
            }
            let (wr, wg, wb) = (wlut[0][cr as usize], wlut[1][cg as usize], wlut[2][cb as usize]);
            // Native d2d1.dll 26100 ClearType SOURCE_OVER preserves destination alpha (nativegap-text-ct-alpha
            // captures); the dual-source per-channel lerp runs here.
            composite_ct_px(target, x, y, [fg_byte[2], fg_byte[1], fg_byte[0]], [wb, wg, wr], blend);
        }
    }
}

/// The device rect a bitmap brush's text intermediate is drawn within: the target, within the pushed clips by the
/// pixel-centre rule.
fn intermediate_limit(target: &PixelBuffer, clip_px: Option<[i32; 4]>) -> [f32; 4] {
    let t = [0.0, 0.0, target.desc.width as f32, target.desc.height as f32];
    clip_px.map_or(t, |c| [t[0].max(c[0] as f32), t[1].max(c[1] as f32), t[2].min(c[2] as f32), t[3].min(c[3] as f32)])
}

/// Grayscale / aliased text blend (R3 / R5) with a gradient/bitmap/image **brush**: the per-colour
/// LUT [`TextLut::grayscale`] computes, evaluated per pixel against the brush's sampled colour (so
/// the gamma/enhanced-contrast correction follows the gradient), then premultiplied source-over.
/// The brush is sampled at the device pixel centre. `gamma_row`/`enhanced_contrast` are the run's.
/// ponytail: renders gradient/bitmap text; native's exact per-pixel gamma is pending the mask-brush
/// text-shader DXBC capture, tracked with the other brush-text floors.
pub fn blend_grayscale_brush(
    target: &mut PixelBuffer,
    alpha: AlphaTexture,
    brush: &crate::brush::Brush,
    gamma_row: u32,
    enhanced_contrast: f32,
    aa_mode: u32,
    blend: crate::pixels::Blend,
    clip_px: Option<[i32; 4]>,
) {
    let (w, h) = (alpha.width(), alpha.height());
    if w <= 0 || h <= 0 || alpha.bytes.len() < (w * h) as usize {
        return;
    }
    let [a1,b1,a2,b2] = GAMMA_INCORRECT_TARGET_RATIOS[gamma_row.min(12) as usize];
    let texels = brush.text_texels([alpha.bounds[0] as f32, alpha.bounds[1] as f32, alpha.bounds[2] as f32, alpha.bounds[3] as f32], intermediate_limit(target, clip_px));
    for row in 0..h {
        for col in 0..w {
            let cov = alpha.bytes[(row * w + col) as usize];
            if cov == 0 {
                continue;
            }
            let (x, y) = (alpha.bounds[0] + col, alpha.bounds[1] + row);
            // premultiplied BGRA, 0..255
            let s = match &texels {
                Some(t) => { let (s, op) = t(x as f32 + 0.5, y as f32 + 0.5); s.map(|v| v * op) }
                None => brush.source_for_text(x as f32 + 0.5, y as f32 + 0.5),
            };
            let put = |target: &mut PixelBuffer, premul: [u8; 4]| if matches!(blend, crate::pixels::Blend::SourceOver) { over_om(target, x, y, premul) } else { composite_px(target, x, y, premul, blend) };
            if aa_mode == AA_ALIASED {
                // The kernel converts the sample to bytes to nearest, ties to even (a gradient's 8.7 ramp lerp lands on
                // .5: r25 30808's [18.5, 18, 10.5, 22.5] stores [18, 18, 10, 22]).
                let premul = core::array::from_fn(|c| (s[c].round_ties_even() as i32).clamp(0, 255) as u8);
                put(target, premul);
                continue;
            }
            if !matches!(blend, crate::pixels::Blend::SourceOver) {
                // A non-over primitive blend lerps the full-coverage brush sample toward the dest by
                // the enhanced-contrast coverage alone, as the solid run does ([`blend_grayscale`]).
                let src_full = core::array::from_fn(|c| rhu(s[c]).clamp(0, 255) as u8);
                composite_cov_px(target, x, y, src_full, Some(s), contrast_at(enhanced_contrast, cov), blend);
                continue;
            }
            let a = s[3] / 255.0;
            // Straight RGB (the gamma/contrast ramps are on the straight foreground colour).
            let (r, g, b) = if a > 0.0 { (s[2] / 255.0 / a, s[1] / 255.0 / a, s[0] / 255.0 / a) } else { (0.0, 0.0, 0.0) };
            // Native d2d1.dll gradient-text PS (captured ps3.dxbc): fixed-contrast mask,
            // then a float gamma polynomial; the solid-text byte LUT is not used here.
            let c = contrast_at(enhanced_contrast,cov) as f32 * (1.0/255.0);
            let intensity = (r*0.25+b*0.25)+g*0.5;
            let av = a1*intensity+b1;
            let bv = a2*intensity+b2;
            let corrected = ((av*c+bv)*((1.0-c)*c))*4.0+c;
            let premul = core::array::from_fn(|i| (s[i]*corrected).round_ties_even().clamp(0.0,255.0) as u8);
            put(target, premul);
        }
    }
}

/// ClearType text blend (R4) with a gradient/bitmap/image **brush**: native draws it in three passes,
/// one per colour channel, each writing only that channel's byte (dst alpha is kept). Pass `k` takes
/// subpixel `k`'s coverage and the brush's straight channel `k` as the intensity, then the gradient
/// text kernel's f32 alpha correction (JIT of r9 14910's radial "fjord"; its cbuffer holds the one-hot
/// channel weights, the gamma row's ratios and the brush opacity): each byte unpacks as
/// `c·(1/256) + c·(1/65280)`, `a' = ((P·a + Q)·((1−a)·a))·4 + a` with `P = L·g0 + g1`,
/// `Q = L·g2 + g3`, the 8.7 ramp sample is scaled by `a'·opacity`, stored ×255 to nearest and
/// source-over'd as `s + ((d·ia + (d·ia >> 8) + 128) >> 8)`.
pub fn blend_cleartype_brush(target: &mut PixelBuffer, alpha: AlphaTexture, brush: &crate::brush::Brush, gamma_row: u32, enhanced_contrast: f32, blend: crate::pixels::Blend, clip_px: Option<[i32; 4]>) {
    let (w, h) = (alpha.width(), alpha.height());
    if w <= 0 || h <= 0 || alpha.bytes.len() < (w * h * 3) as usize {
        return;
    }
    let [g0, g1, g2, g3] = GAMMA_INCORRECT_TARGET_RATIOS[gamma_row.min(12) as usize];
    let unorm = |c: f32| c * 0.003_906_25 + c * f32::from_bits(0x3780_8081);
    let texels = brush.text_texels([alpha.bounds[0] as f32, alpha.bounds[1] as f32, alpha.bounds[2] as f32, alpha.bounds[3] as f32], intermediate_limit(target, clip_px));
    for row in 0..h {
        for col in 0..w {
            let base_i = ((row * w + col) * 3) as usize;
            let (cr, cg, cb) = (alpha.bytes[base_i], alpha.bytes[base_i + 1], alpha.bytes[base_i + 2]);
            if cr == 0 && cg == 0 && cb == 0 {
                continue;
            }
            let (x, y) = (alpha.bounds[0] + col, alpha.bounds[1] + row);
            if x < 0 || y < 0 || x >= target.desc.width as i32 || y >= target.desc.height as i32 {
                continue;
            }
            if !matches!(blend, crate::pixels::Blend::SourceOver) {
                // Under a non-over blend ClearType collapses to the green-subpixel (centre) coverage
                // over the dest (same rule as solid CT); the full-coverage source is the brush sample.
                let s = brush.source_at(x as f32 + 0.5, y as f32 + 0.5, true); // premultiplied BGRA, 0..255
                let src_full = core::array::from_fn(|c| rhu(s[c]).clamp(0, 255) as u8);
                composite_cov_px(target, x, y, src_full, Some(s), cg, blend);
                continue;
            }
            let (s, opacity) = match &texels { Some(t) => t(x as f32 + 0.5, y as f32 + 0.5), None => brush.text_sample(x as f32 + 0.5, y as f32 + 0.5) };
            let s = s.map(unorm); // premultiplied BGRA
            let st = if s[3] > 0.0 { [s[0] / s[3], s[1] / s[3], s[2] / s[3]] } else { [s[0], s[1], s[2]] };
            let i = y as usize * target.stride + x as usize * 4;
            // (subpixel coverage, BGRA byte) per pass, R first.
            for (cov, k) in [(cr, 2usize), (cg, 1), (cb, 0)] {
                // ponytail: the fixed-contrast mask as the grayscale brush kernel's; r9 14910 is GDI
                // ClearType (contrast 0), so a contrast ≠ 0 capture is still owed.
                let a = unorm(contrast_at(enhanced_contrast, cov) as f32);
                let l = st[k];
                let (pp, q) = (l * g0 + g1, l * g2 + g3);
                let a2 = ((pp * a + q) * ((1.0 - a) * a)) * 4.0 + a;
                let store = |v: f32| (((v * a2) * opacity).clamp(0.0, 1.0) * 255.0).round_ties_even() as i32;
                let (sk, sa) = (store(s[k]), store(s[3]));
                let t = target.data[i + k] as i32 * (255 - sa);
                target.data[i + k] = (sk + ((t + (t >> 8) + 128) >> 8)).clamp(0, 255) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contrast_curve_endpoints_and_monotone() {
        let ec = contrast_curve(1.0);
        assert_eq!(ec[0], 0);
        assert_eq!(ec[255], 255);
        // k=1 lifts low coverage: 128 → round(2·0.502/1.502·255) = 170.
        assert_eq!(ec[128], 170);
        assert!(ec.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn black_grayscale_lut_endpoints() {
        // Black text: coverage 0 → transparent, full coverage → opaque black. gamma 2.2, ramp 1.0.
        let bp = BlendParams::resolve([0.0, 0.0, 0.0, 1.0], 2.2, 1.0, 1.0);
        assert_eq!((bp.gamma_row, bp.contrast), (12, 1.0));
        let lut = TextLut::grayscale([0.0, 0.0, 0.0, 1.0], &bp);
        assert_eq!((lut.lut[0], lut.lut[255]), ([0, 0, 0, 0], [0, 0, 0, 255]));
    }

    #[test]
    fn white_text_gets_no_contrast_ramp() {
        // White (L=1) → ramp 0 → contrast scalar 0.
        let bp = BlendParams::resolve([1.0, 1.0, 1.0, 1.0], 2.2, 1.0, 1.0);
        assert_eq!(bp.contrast, 0.0);
    }

    #[test]
    fn over_is_premultiplied_source_over() {
        use d2d_types::abi::{D2D1_ALPHA_MODE_PREMULTIPLIED, DXGI_FORMAT_B8G8R8A8_UNORM};
        use d2d_types::contracts::ImageDesc;
        let mut buf = PixelBuffer {
            desc: ImageDesc {
                width: 1,
                height: 1,
                format: DXGI_FORMAT_B8G8R8A8_UNORM,
                alpha_mode: D2D1_ALPHA_MODE_PREMULTIPLIED,
                dpi_x: 96.0,
                dpi_y: 96.0,
            },
            stride: 4,
            data: vec![255, 255, 255, 255], // opaque white
        };
        over(&mut buf, 0, 0, [0, 0, 0, 128]); // half-coverage black
        // out = 0 + round(255·127/255) = 127 for RGB; alpha 128 + round(255·127/255) = 255.
        assert_eq!(buf.data, vec![127, 127, 127, 255]);
    }
}
