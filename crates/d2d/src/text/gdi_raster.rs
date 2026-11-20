//! DirectWrite's **GDI bitmap render target** rasteriser — `IDWriteBitmapRenderTarget::DrawGlyphRun`
//! onto a top-down BGRA32 DIB, byte-exact with DirectWrite.
//!
//! This is a *distinct* pipeline from `IDWriteGlyphRunAnalysis::CreateAlphaTexture` + the Direct2D
//! text blend (`text::raster` + `text::blend`). DirectWrite's bitmap render target builds a
//! glyph-run analysis and draws it, which for a **grayscale** target runs the grayscale path:
//!
//! * **Coverage** is an **8 horizontal × 1 vertical** sub-sample mask per device pixel (per the
//!   texture-type table `[8·aa + mode]`: grayscale NATURAL / GDI_* → 8×1; NATURAL_SYMMETRIC → 4×4),
//!   popcounted 0..8. That is a different oversample from the 4×4 `raster::grayscale`,
//!   so we scan the *same* hinted outline through the shared scan converter at 8×1
//!   ([`raster::coverage_bitmap`]) and popcount each pixel's row of 8.
//! * **Alpha curve** (per sub-sample count `n`):
//!   `c = n/8`, contrast `a = (k+1)c/(kc+1)` (grayscale enhanced-contrast `k`, after the light-on-dark
//!   ramp), then the gamma parabola `a += ((i·g0+g1)a + (i·g2+g3))·(1−a)·a` with the fg intensity
//!   `i = (2G+R+B)/1020` and the gamma ratios (gamma 2.2 → row 11), then
//!   `round(a·256)` (truncate, tie → up). Count 8 (full) writes the opaque fg directly.
//! * **Composite**: straight-alpha `(dst·(256−a) + fg·a)/256`
//!   per channel, packed (bias `0x7F007F`) — *not* the premultiplied source-over `text::blend` uses.
//!
//! ClearType (the 6×1 path) is a separate follow-up; grayscale is wired here.

use super::raster::{self, dropout_mode};
use super::{GlyphRun, RenderingParams, recommended_rendering_mode};
use skrifa::FontRef;

// D2D1_TEXT_ANTIALIAS_MODE values the bitmap render target resolves to (see `text::render_glyph_run`).
const AA_CLEARTYPE: u32 = 1;
const AA_GRAYSCALE: u32 = 2;
const AA_ALIASED: u32 = 3;

// DWRITE_RENDERING_MODE values the mode→hinting mapping branches on.
const MODE_ALIASED: u32 = 1;
const MODE_GDI_CLASSIC: u32 = 2;
const MODE_GDI_NATURAL: u32 = 3;
const MODE_NATURAL_SYMMETRIC: u32 = 5;
const MODE_OUTLINE: u32 = 6;

/// The 4 gamma-parabola coefficients for `gamma`,
/// or `None` for gamma ≤ ~1.05 (identity — no parabola). Row = `trunc((gamma−1.05)·10)`, clamped 11.
///
/// Only row 11 (gamma 2.2, the single gamma every GDI scene pins) is baked, read from the native
/// table; it is a distinct set from the Direct2D blend's gamma-2.2 row.
// ponytail: other gamma rows read from the native table when a scene uses a gamma other than 2.2 / 1.0.
fn gamma_ratios(gamma: f32) -> Option<[f32; 4]> {
    if ((gamma - 1.05) * 10.0) as i32 <= 0 {
        return None;
    }
    Some([0.20310020, -1.3864021, 1.9850283, -0.35011005])
}

/// The light-on-dark contrast adjustment: the enhanced-contrast
/// scalar `k` scaled by the fg-luminance ramp `clamp(4·(0.75−L), 0, 1)`, `L = 0.30R+0.59G+0.11B`.
fn light_on_dark(k: f32, r: f32, g: f32, b: f32) -> f32 {
    let l = (r * 0.30000001 + g * 0.58999997) + b * 0.11;
    let ramp = if l >= 0.75 {
        0.0
    } else if l > 0.5 {
        (0.75 - l) * 4.0
    } else {
        1.0
    };
    ramp * k
}

/// The grayscale alpha-by-sub-sample-count for an N-sub-sample mask: the alpha byte for each
/// covered-count 1..N−1 (count 0 → transparent, count N → opaque, both handled by the caller).
/// `k` is the light-on-dark-adjusted contrast, `gamma` the blend gamma, `color` the straight fg COLORREF
/// (`0x00BBGGRR`) for the intensity term. All float maths mirror native exactly.
fn alpha_table(k: f32, gamma: f32, color: u32, subsamples: u32) -> Vec<u8> {
    let ratios = gamma_ratios(gamma);
    let (r, g, b) = ((color & 0xFF) as f32, ((color >> 8) & 0xFF) as f32, ((color >> 16) & 0xFF) as f32);
    // Intensity i = (2G + R + B)/1020 = ¼R + ½G + ¼B, in native's add order (2/1020, 1/1020, 1/1020).
    let i = (g * 0.0019607844 + r * 0.00098039221) + b * 0.00098039221;
    let step = 1.0 / subsamples as f32;
    let mut c_base = step;
    let mut out = Vec::with_capacity(subsamples as usize - 1);
    for _ in 1..subsamples {
        let mut c = c_base;
        if k != 0.0 {
            c = ((k + 1.0) * c) / (c * k + 1.0);
        }
        if let Some([g0, g1, g2, g3]) = ratios {
            c += (((i * g0 + g1) * c) + (i * g2 + g3)) * ((1.0 - c) * c);
        }
        // Native `cvtss2si` rounds to nearest, and the `== -0.5` fixup rounds exact halves up — i.e.
        // round-half-up of `c''·256` (not truncation).
        let n = (c * 256.0 + 0.5).floor() as i32;
        out.push(n.clamp(0, 255) as u8);
        c_base += step;
    }
    out
}

/// Straight-alpha composite of opaque fg `s` (a BGRA32 pixel)
/// over dst `d` by alpha `a` (0..255), packed per channel exactly as native (bias `0x7F007F`, α/256).
#[inline]
fn gray_blend(d: u32, s: u32, a: u32) -> u32 {
    let ia = 256 - a;
    (((ia * (d & 0xFF00FF) + 0x7F007F) >> 8) & 0xFF00FF)
        + ((a * ((s >> 8) & 0xFF00FF) + 0x7F007F) & 0xFF00FF00)
        + (((a * (s & 0xFF00FF) + 0x7F007F) >> 8) & 0xFF00FF)
        + ((ia * ((d >> 8) & 0xFF00FF) + 0x7F007F) & 0xFF00FF00)
}

// --- ClearType DIB path (6×1) --------------------------------------------------

/// GDI ClearType gamma table: `[0..256)` is the inverse
/// gamma `round((i/255)^(1/γ)·255)` (gamma-space → display), `[256..512)` the forward
/// `round((i/255)^γ·255)` (display → gamma-space); endpoints pinned 0/255.
fn gamma_table(gamma: f32) -> [u8; 512] {
    let mut t = [0u8; 512];
    for i in 1..255usize {
        let x = i as f32 / 255.0;
        t[i] = (crate::crt::powf(x, 1.0 / gamma) * 255.0 + 0.5).floor() as u8;
        t[256 + i] = (crate::crt::powf(x, gamma) * 255.0 + 0.5).floor() as u8;
    }
    t[255] = 255;
    t[511] = 255;
    t
}

/// GDI enhanced-contrast table: the contrast-adjusted
/// coverage weight `round((k+1)c/(kc+1)·256)` (`c=i/255`, 0..256), for the light-on-dark-adjusted
/// ClearType enhanced-contrast scalar `k`.
fn ec_table(k: f32) -> [u16; 256] {
    let mut t = [0u16; 256];
    for i in 1..255usize {
        let c = i as f32 / 255.0;
        t[i] = (((k + 1.0) * c) / (c * k + 1.0) * 256.0 + 0.5).floor() as u16;
    }
    t[255] = 256;
    t
}

/// `g_AlphaNormalizationTable6x1[0..=6]`: a flat 6-box coverage count → normalized coverage
/// `round(count·255/6)` (`= {0,43,85,128,170,213,255}`).
fn norm6(count: u32) -> u16 {
    ((count * 255 + 3) / 6) as u16
}

/// One ClearType channel composite: gamma-space blend `inv[(w·fwd[fg] + 128 + (256−w)·fwd[dst]) >> 8]`,
/// `w` the EC-corrected coverage weight. The dst alpha byte is left untouched (ClearType is RGB-only).
#[inline]
fn ct_blend(dst_ch: u32, fg_ch: u32, w: u32, g: &[u8; 512]) -> u32 {
    let idx = ((w * g[256 + fg_ch as usize] as u32 + 128 + (256 - w) * g[256 + dst_ch as usize] as u32) >> 8) as usize;
    g[idx] as u32
}

/// The ClearType path with **flat** pixel geometry: the GDI bitmap render target's
/// *default*. Coverage is a 6-horizontal × 1-vertical sub-sample count per pixel; flat geometry makes
/// the three colour sub-pixels share that count, so the blend composites an equal weight onto B/G/R
/// through the ClearType gamma + enhanced-contrast tables (RGB only, dst alpha preserved).
fn draw_cleartype_flat(dib: &mut [u32], width: u32, height: u32, dt: &[f64; 6], run: &GlyphRun, color: u32, params: &RenderingParams) -> bool {
    let (fg_r, fg_g, fg_b) = (color & 0xFF, (color >> 8) & 0xFF, (color >> 16) & 0xFF);
    let gamma = gamma_table(params.gamma);
    // ClearType enhanced contrast, light-on-dark-adjusted (0 for white — which drops out below).
    let k = light_on_dark(params.enhanced_contrast, fg_r as f32 / 255.0, fg_g as f32 / 255.0, fg_b as f32 / 255.0);
    let ec = ec_table(k);
    composite(dib, width, height, dt, run, AA_CLEARTYPE, params, 6, |count, d| {
        let w = ec[norm6(count) as usize] as u32;
        let out_b = ct_blend(d & 0xFF, fg_b, w, &gamma);
        let out_g = ct_blend((d >> 8) & 0xFF, fg_g, w, &gamma);
        let out_r = ct_blend((d >> 16) & 0xFF, fg_r, w, &gamma);
        (d & 0xFF000000) | (out_r << 16) | (out_g << 8) | out_b
    })
}

/// Snap the baseline origin (DIPs) to the device pixel grid perpendicular to the baseline, as
/// `DrawGlyphRun` does (mirrors `text::snap_baseline`): axis-aligned snaps device y, rotated unsnapped.
fn snap_baseline(origin: [f32; 2], dt: &[f64; 6], no_snap: bool) -> [f32; 2] {
    if no_snap || dt[1] != 0.0 || dt[2] != 0.0 || dt[3] == 0.0 {
        return origin;
    }
    super::snap_baseline(origin, dt, false)
}

/// Quantize a device pen x to the mode's sub-pixel phase grid.
fn quantize_phase(x: f64, symmetric: bool) -> f64 {
    let base = x.floor();
    let phases = if symmetric { 4.0 } else { 1.0 };
    base + ((x - base) * phases).round() / phases
}

/// Grid-fit the run through DirectWrite's TrueType hinting interpreter and lay the hinted outlines out
/// in device space — the same hinted quadratic contours `text::render_glyph_run` scan-converts, so the
/// covered-pixel *shape* matches; we then resolve them at the GDI 8×1 oversample. Mirrors
/// `text::hinted_contours`. Returns the contours and the dropout-control mode word.
fn hinted_contours(dt: &[f64; 6], run: &GlyphRun, font: &FontRef, mode: u32, aa: u32) -> Option<(Vec<(Vec<[f64; 2]>, Vec<bool>)>, u16)> {
    use super::hinting;
    let scale = dt[0].abs().max(dt[3].abs());
    let device_em = (run.em_size as f64 * scale).round();
    let hmode = match (mode, aa) {
        (MODE_GDI_CLASSIC, _) => hinting::Mode::GdiClassic,
        (MODE_GDI_NATURAL, _) => hinting::Mode::GdiNatural,
        (_, AA_ALIASED) => hinting::Mode::Aliased,
        (_, AA_CLEARTYPE) => hinting::Mode::ClearType,
        _ => hinting::Mode::Grayscale,
    };
    let symmetric = mode == MODE_NATURAL_SYMMETRIC;
    let cfg = hinting::Config {
        rasterizer: hinting::Rasterizer::FontCache,
        mode: hmode,
        symmetric,
        bgr: false,
        is_rotated: dt[1] != 0.0 || dt[2] != 0.0,
        is_stretched: (dt[0].abs() - dt[3].abs()).abs() > 1e-6,
        em_y: None,
        axes_swapped: false, // ponytail: DIB path renders unrotated; pass dt[0] == 0.0 if a rotated DIB scene appears
        rotation_kind: 0, // ponytail: DIB path renders unrotated; its composite offsets stay on the base rule
        version: 42,
        gdi_render: false,
        cleartype: false,
        subpixel_advance: false, // DIB raster path is byte-exact as-is
        grid_fit: true,          // GDI DIB raster always grid-fits (GDI-compatible hinting)
        bold_fu: 0,
    };
    let inst = hinting::Instance::new(font, device_em, cfg)?;
    #[cfg(feature = "inject-contours")]
    if std::env::var_os("D2D_INJECT_DBG").is_some() { eprintln!("GDIHC aa={} mode={} em={} inject={:?}", aa, mode, device_em, std::env::var_os("D2D_INJECT_CONTOURS").is_some()); }
    let subpixel = !matches!(hmode, hinting::Mode::Aliased | hinting::Mode::GdiClassic);
    let origin = snap_baseline(run.baseline_origin, dt, run.no_snap);
    let mut out: Vec<(Vec<[f64; 2]>, Vec<bool>)> = Vec::new();
    // Diagnostic (off-by-default `inject-contours` feature): D2D_INJECT_CONTOURS swaps each glyph's hinted
    // output for native's captured shape at OUR pen origin, to split a hinting residual from a raster one in
    // the GDI DIB path. (Record alignment/units here still predate the mod.rs fixes — update before using
    // this path for 5029.)
    #[cfg(feature = "inject-contours")]
    let inject = super::inject_contours();
    #[cfg(feature = "inject-contours")]
    let mut inject_idx = 0usize;
    let mut pen_x = origin[0] as f64;
    let origin_y = origin[1] as f64;
    for glyph in run.glyphs {
        let gx = pen_x + glyph.offset[0] as f64;
        let gy = origin_y - glyph.offset[1] as f64;
        let pen_dev_x = dt[0] * gx + dt[2] * gy + dt[4];
        let pen_dev_y = dt[1] * gx + dt[3] * gy + dt[5];
        let py = pen_dev_y.round();
        let px = if subpixel { quantize_phase(pen_dev_x, symmetric) } else { pen_dev_x.round() };
        if let Some(h) = inst.hint(glyph.id) {
            #[cfg(feature = "inject-contours")]
            if let Some(recs) = &inject {
                if let Some(contours) = recs.get(inject_idx) {
                    inject_idx += 1;
                    for c in contours {
                        let pts: Vec<[f64; 2]> = c.iter().map(|&(x, y, _)| [px + x / 512.0, py - y / 64.0]).collect();
                        let onc: Vec<bool> = c.iter().map(|&(_, _, o)| o).collect();
                        out.push((pts, onc));
                    }
                }
                pen_x += glyph.advance as f64;
                continue;
            }
            let mut s = 0usize;
            for &e in &h.contour_ends {
                let e = e as usize;
                if e >= s && e < h.points.len() {
                    let pts = h.points[s..=e].iter().map(|p| [px + p.x as f64 / 64.0, py - p.y as f64 / 64.0]).collect();
                    out.push((pts, h.on_curve[s..=e].to_vec()));
                }
                s = e + 1;
            }
        }
        pen_x += glyph.advance as f64;
    }
    let (enabled, scan_type) = inst.scan_control();
    Some((out, dropout_mode(enabled, scan_type)))
}

/// Scan-convert the run's hinted outline at `sx`×1 (the GDI DIB oversample) and composite each covered
/// pixel onto the DIB via `pixel(count, dst) -> new_dst`, `count` the pixel's `sx`-sub-sample popcount
/// (1..sx). Shared by the ClearType (6×1) and grayscale (8×1) paths. Returns false if the run needs a
/// path not wired here (outline / aliased).
fn composite(dib: &mut [u32], width: u32, height: u32, dt: &[f64; 6], run: &GlyphRun, aa: u32, params: &RenderingParams, sx: i32, pixel: impl Fn(u32, u32) -> u32) -> bool {
    let Ok(font) = FontRef::from_index(run.font_data, run.face_index) else { return false };
    let mode = recommended_rendering_mode(run, dt, aa, params);
    if mode == MODE_OUTLINE || mode == MODE_ALIASED {
        return false;
    }
    let Some((contours, mode_word)) = hinted_contours(dt, run, &font, mode, aa) else { return false };
    let Some((bounds, bits)) = raster::coverage_bitmap(&contours, mode_word, sx, 1) else { return true };
    let (l, t, r, b) = (bounds[0], bounds[1], bounds[2], bounds[3]);
    let ow = ((r - l) * sx) as usize;
    for row in 0..(b - t) {
        let dy = t + row;
        if dy < 0 || dy >= height as i32 {
            continue;
        }
        for cx in 0..(r - l) {
            let dx = l + cx;
            if dx < 0 || dx >= width as i32 {
                continue;
            }
            let o = row as usize * ow + (cx * sx) as usize;
            let count = bits[o..o + sx as usize].iter().filter(|&&v| v).count() as u32;
            if count == 0 {
                continue;
            }
            let i = dy as usize * width as usize + dx as usize;
            dib[i] = pixel(count, dib[i]);
        }
    }
    true
}

/// The grayscale path: the run's 8×1 grayscale coverage composited with the
/// GDI alpha curve + straight-alpha composite (opaque fg = `0xFF<<24 | R<<16 | G<<8 | B`).
fn draw_grayscale(dib: &mut [u32], width: u32, height: u32, dt: &[f64; 6], run: &GlyphRun, color: u32, params: &RenderingParams) -> bool {
    let (rr, gg, bb) = (color & 0xFF, (color >> 8) & 0xFF, (color >> 16) & 0xFF);
    let fg = 0xFF000000 | (rr << 16) | (gg << 8) | bb;
    let k = light_on_dark(params.grayscale_contrast, rr as f32 / 255.0, gg as f32 / 255.0, bb as f32 / 255.0);
    let table = alpha_table(k, params.gamma, color, 8);
    composite(dib, width, height, dt, run, AA_GRAYSCALE, params, 8, |count, dst| {
        if count >= 8 { fg } else { gray_blend(dst, fg, table[count as usize - 1] as u32) }
    })
}

/// Composite one positioned glyph run onto a top-down BGRA32 DIB exactly as
/// `IDWriteBitmapRenderTarget::DrawGlyphRun`. `color` is the text COLORREF (`0x00BBGGRR`); `grayscale`
/// selects the grayscale target (the default) vs ClearType. Returns false if the run needs a path not
/// yet wired here (ClearType, outline, aliased), so the caller can fall back.
pub fn draw_glyph_run_gdi(dib: &mut [u32], width: u32, height: u32, dt: &[f64; 6], run: &GlyphRun, color: u32, grayscale: bool, params: &RenderingParams) -> bool {
    if run.glyphs.is_empty() {
        return true;
    }
    if grayscale {
        draw_grayscale(dib, width, height, dt, run, color, params)
    } else {
        draw_cleartype_flat(dib, width, height, dt, run, color, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_white_on_black() {
        // White fg over black by half alpha: every channel ≈ grey 0x7F (packed /256 straight-alpha).
        assert_eq!(gray_blend(0x00000000, 0xFFFFFFFF, 128), 0x7F7F7F7F);
        // Zero alpha leaves the destination untouched.
        assert_eq!(gray_blend(0x12345678, 0xFFFFFFFF, 0) & 0x00FFFFFF, 0x00345678);
    }

    #[test]
    fn cleartype_flat_curve_matches_native() {
        // White fg on black, γ2.2, EC0: the GDI ClearType-flat grey for each 6×1 coverage count,
        // verified byte-for-byte against native DrawGlyphRun (dwrite-probe/GetBitmapData on the VM,
        // Arial 'H'): counts 1/2/4/5 → 114/155/212/235, full → opaque white.
        let g = gamma_table(2.2);
        let ec = ec_table(0.0);
        let grey = |count: u32| ct_blend(0, 255, ec[norm6(count) as usize] as u32, &g) as u8;
        assert_eq!([grey(1), grey(2), grey(4), grey(5), grey(6)], [114, 155, 212, 235, 255]);
    }

    #[test]
    fn grayscale_curve_matches_native() {
        // White fg on black, γ2.2: the 8×1 grayscale grey for coverage counts 2 and 6, verified against
        // native's explicit-grayscale DrawGlyphRun (dwrite-probe, Arial 'H'): count2→127, count6→227.
        let t = alpha_table(0.0, 2.2, 0xFFFFFF, 8);
        let grey = |count: usize| (gray_blend(0, 0xFFFFFFFF, t[count - 1] as u32) & 0xFF) as u8;
        assert_eq!((grey(2), grey(6)), (127, 227));
    }

    #[test]
    fn alpha_table_monotone_and_bounded() {
        // Black text, gamma 2.2, full contrast: 7 entries (counts 1..7), strictly increasing, in 0..255.
        let t = alpha_table(1.0, 2.2, 0x000000, 8);
        assert_eq!(t.len(), 7);
        assert!(t.windows(2).all(|w| w[0] < w[1]));
        // Gamma 1.0 → no parabola; white → ramp 0 so contrast drops out (identity curve × 256).
        let id = alpha_table(0.0, 1.0, 0xFFFFFF, 8);
        assert_eq!(id, vec![32, 64, 96, 128, 160, 192, 224]);
    }
}
