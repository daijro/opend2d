//! `ID2D1DeviceContext4::DrawColorBitmapGlyphRun`: CBDT/CBLC and sbix PNG (and raw PBGRA) bitmap
//! strikes. Reproduces Direct2D's strike
//! selection and placement transform, with `skrifa::bitmap` supplying the strike data and the
//! workspace `png` decoder the PNG images. The glyph image composites source-over (LINEAR), as native
//! does via a DrawImage of the realized strike.
//!
//! Placement (native `GetColorBitmapGlyphImage`): the strike is chosen at device ppem
//! `GetMaxFactor(deviceTransform) * emSize` (truncated, +1 only on an exact .5), then a local matrix
//! scales by `s = emSize / strikePpem` and places the glyph so its left-origin pixel sits on the
//! baseline; the device transform maps that into the target. Verified byte-exact against the native
//! `colorglyph-sbix-00` oracle (System32 d2d1/dwrite 10.0.26100).

use super::{GlyphRun, RenderingParams};
use crate::bitmap::Bitmap;
use d2d_types::abi::{D2D1_ALPHA_MODE_PREMULTIPLIED, DXGI_FORMAT_B8G8R8A8_UNORM};
use d2d_types::contracts::PixelBuffer;
use skrifa::bitmap::{BitmapData, BitmapStrikes, Origin};
use skrifa::instance::Size;
use skrifa::{FontRef, GlyphId};

/// The larger singular value of the 2×2 part (the matrix max-factor) — the
/// device pixels-per-em scale. Ported f32-for-f32 (`m = [m11, m12, m21, m22]`).
pub(crate) fn max_factor(m: [f32; 4]) -> f32 {
    if m[1] == 0.0 && m[2] == 0.0 {
        m[3].abs().max(m[0].abs())
    } else {
        let v4 = m[0] * m[0] + m[1] * m[1];
        let v5 = m[2] * m[0] + m[3] * m[1];
        let v6 = m[2] * m[2] + m[3] * m[3];
        let v7 = ((v5 * 4.0) * v5 + (v4 - v6) * (v4 - v6)).sqrt();
        ((v7 + (v6 + v4)) * 0.5).sqrt()
    }
}

/// Decode a bitmap-glyph image to premultiplied BGRA8 `(width, height, pixels)`. `Bgra` is already
/// premultiplied (skrifa/OpenType); a `Png` is decoded (straight BGRA) and premultiplied here. `Mask`
/// (monochrome EBDT) is not a colour-bitmap format and is skipped.
fn decode_premul(data: &BitmapData, w: u32, h: u32) -> Option<Vec<u8>> {
    match data {
        BitmapData::Bgra(bytes) => (bytes.len() >= (w * h * 4) as usize).then(|| bytes[..(w * h * 4) as usize].to_vec()),
        BitmapData::Png(bytes) => {
            let img = crate::wic::decode_png(bytes).ok()?;
            if img.width != w || img.height != h {
                return None;
            }
            // decode_png emits straight BGRA8 (32bppBGRA); premultiply for source-over.
            let mut out = vec![0u8; (w * h * 4) as usize];
            for (o, px) in out.chunks_exact_mut(4).zip(img.pixels.chunks_exact(4)) {
                let a = px[3] as u32;
                o[0] = ((px[0] as u32 * a + 127) / 255) as u8;
                o[1] = ((px[1] as u32 * a + 127) / 255) as u8;
                o[2] = ((px[2] as u32 * a + 127) / 255) as u8;
                o[3] = px[3];
            }
            Some(out)
        }
        BitmapData::Mask(_) => None,
    }
}

/// `ID2D1DeviceContext4::GetColorBitmapGlyphImage`: the single-glyph form of the run renderer. Returns
/// the decoded strike bitmap and the image→world transform (`*glyphTransform`) the caller draws it
/// under, so `DrawImage(image)` with that transform reproduces the run renderer's placement exactly.
/// The strike is chosen
/// at device ppem `maxFactor(world × dpi/96) · emSize`; the glyph matrix scales by `s = emSize /
/// strikePpem` and offsets to the glyph origin, is pre-rotated −90° when sideways and post-skewed by
/// −0.33984375 when the face carries OBLIQUE simulation (`GetSimulations()==2`), then the baseline
/// origin is added and the whole is composed with the caller's world transform.
#[allow(clippy::too_many_arguments)]
pub fn color_bitmap_glyph_image(
    font_data: &[u8],
    face_index: u32,
    glyph: u16,
    em_size: f32,
    sims: u16,
    world: [f32; 6],
    dpi: (f32, f32),
    origin: [f32; 2],
    is_sideways: bool,
) -> Option<(Bitmap, [f32; 6])> {
    let font = FontRef::from_index(font_data, face_index).ok()?;
    let strikes = BitmapStrikes::new(&font);
    // Device ppem for strike selection: GetMaxFactor of the world transform scaled by dpi/96.
    let (sx, sy) = (dpi.0 / 96.0, dpi.1 / 96.0);
    let dev = [world[0] * sx, world[1] * sy, world[2] * sx, world[3] * sy];
    let dev_em = max_factor(dev) * em_size;
    let t = dev_em as i32;
    let dev_ppem = if (t as f32 - dev_em) == -0.5 { t + 1 } else { t };
    let bg = strikes.glyph_for_size(Size::new(dev_ppem as f32), GlyphId::from(glyph))?;
    let px = decode_premul(&bg.data, bg.width, bg.height)?;
    let (w, h) = (bg.width as f32, bg.height as f32);
    let s = em_size / bg.ppem_y;
    // The image top-left relative to the glyph origin (same placement as the run renderer): a sbix
    // bottom-left origin lifts the image by its height.
    let (gx, gy) = match bg.placement_origin {
        Origin::BottomLeft => (bg.inner_bearing_x * s, -(bg.inner_bearing_y + h) * s),
        Origin::TopLeft => (bg.inner_bearing_x * s, -bg.inner_bearing_y * s),
    };
    // Image-pixel → glyph-local matrix: scale by s, translate to the top-left.
    let mut m = [s, 0.0, 0.0, s, gx, gy];
    if is_sideways {
        // Native pre-rotates the scale matrix by −90° (glyphScale × rotate).
        m = crate::geometry::mat_mul(&m, &rotate_neg90());
    }
    if sims & 2 != 0 {
        // OBLIQUE simulation: native post-multiplies the −0.33984375 oblique skew.
        m = crate::geometry::mat_mul(&m, &[1.0, 0.0, -0.339_843_75, 1.0, 0.0, 0.0]);
    }
    // Add the baseline origin to the glyph matrix, then compose with the caller's world transform.
    m[4] += origin[0];
    m[5] += origin[1];
    let gt = crate::geometry::mat_mul(&m, &world);
    let bmp = Bitmap { width: bg.width, height: bg.height, premul_bgra: px, format: DXGI_FORMAT_B8G8R8A8_UNORM, alpha: D2D1_ALPHA_MODE_PREMULTIPLIED, dpi: (96.0, 96.0) };
    Some((bmp, gt))
}

/// A −90° rotation about the origin (`D2D1MakeRotateMatrix(-90)`): `[cos, sin, -sin, cos, 0, 0]` with
/// cos(−90)=0, sin(−90)=−1 → `[0, -1, 1, 0, 0, 0]`.
fn rotate_neg90() -> [f32; 6] {
    [0.0, -1.0, 1.0, 0.0, 0.0, 0.0]
}

/// Render a colour-bitmap glyph run. Returns `true` when the run was handled (even if a glyph had no
/// strike — matching native, which simply draws nothing for a glyph absent from the requested format).
pub fn render_bitmap_glyph_run(target: &mut PixelBuffer, dt: &[f64; 6], run: &GlyphRun, formats: u32, params: &RenderingParams) -> bool {
    crate::text::via_bgra8(target, |t| render_bitmap_glyph_run_u8(t, dt, run, formats, params))
}

fn render_bitmap_glyph_run_u8(target: &mut PixelBuffer, dt: &[f64; 6], run: &GlyphRun, _formats: u32, params: &RenderingParams) -> bool {
    let Ok(font) = FontRef::from_index(run.font_data, run.face_index) else { return false };
    let strikes = BitmapStrikes::new(&font);
    if strikes.is_empty() {
        return false;
    }
    // Device pixels-per-em for strike selection (native GetColorBitmapGlyphImage: GetMaxFactor·em,
    // truncate toward zero, +1 only on an exact half).
    let mf = max_factor([dt[0] as f32, dt[1] as f32, dt[2] as f32, dt[3] as f32]);
    let dev_em = mf * run.em_size;
    let t = dev_em as i32;
    let dev_ppem = if (t as f32 - dev_em) == -0.5 { t + 1 } else { t };

    let (mut pen_x, pen_y) = (run.baseline_origin[0], run.baseline_origin[1]);
    for g in run.glyphs {
        let ox = pen_x + g.offset[0];
        let oy = pen_y - g.offset[1];
        pen_x += g.advance;
        let Some(bg) = strikes.glyph_for_size(Size::new(dev_ppem as f32), GlyphId::from(g.id)) else { continue };
        let Some(px) = decode_premul(&bg.data, bg.width, bg.height) else { continue };
        let (w, h) = (bg.width as f32, bg.height as f32);
        let s = run.em_size / bg.ppem_y;
        // Image top-left in DIP space. sbix carries a bottom-left origin (the inner bearing is the
        // sbix graphicOffset); CBDT/EBDT a top-left origin (inner bearing = the BDT left/top bearing).
        let (tlx, tly) = match bg.placement_origin {
            Origin::BottomLeft => (ox + bg.inner_bearing_x * s, oy - (bg.inner_bearing_y + h) * s),
            Origin::TopLeft => (ox + bg.inner_bearing_x * s, oy - bg.inner_bearing_y * s),
        };
        // Map the DIP placement rect through the device transform.
        let map = |x: f32, y: f32| [(dt[0] * x as f64 + dt[2] * y as f64 + dt[4]) as f32, (dt[1] * x as f64 + dt[3] * y as f64 + dt[5]) as f32];
        let (x1, y1) = (tlx + w * s, tly + h * s);
        let bmp = Bitmap { width: bg.width, height: bg.height, premul_bgra: px, format: DXGI_FORMAT_B8G8R8A8_UNORM, alpha: D2D1_ALPHA_MODE_PREMULTIPLIED, dpi: (96.0, 96.0) };
        // Native composites the strike with a LINEAR-sampled DrawImage under the full transform. Any axis-aligned
        // scale is a device blit, a flip included: it blits the image mirrored (r25 30563, 30666, 30010), where the
        // general affine sampler is a texel phase off. A rotation or shear maps the four corners.
        if dt[1] == 0.0 && dt[2] == 0.0 && dt[0] != 0.0 && dt[3] != 0.0 {
            let (a, c) = (map(tlx, tly), map(x1, y1));
            let (fx, fy) = (dt[0] < 0.0, dt[3] < 0.0);
            let bmp = if fx || fy {
                let (bw, bh) = (bmp.width as usize, bmp.height as usize);
                let mut px = vec![0u8; bmp.premul_bgra.len()];
                for y in 0..bh {
                    for x in 0..bw {
                        let (sx, sy) = (if fx { bw - 1 - x } else { x }, if fy { bh - 1 - y } else { y });
                        px[(y * bw + x) * 4..][..4].copy_from_slice(&bmp.premul_bgra[(sy * bw + sx) * 4..][..4]);
                    }
                }
                Bitmap { premul_bgra: px, ..bmp }
            } else {
                bmp
            };
            // The image draw takes the primitive blend, as any DrawImage does (r25 30534: ADD).
            let dest = [a[0].min(c[0]), a[1].min(c[1]), a[0].max(c[0]), a[1].max(c[1])];
            crate::bitmap::draw_bitmap_with(target, &bmp, dest, [0.0, 0.0, w, h], 1.0, 1, params.blend, crate::pixels::Aa::PerPrimitive);
        } else {
            let corners = [map(tlx, tly), map(x1, tly), map(x1, y1), map(tlx, y1)];
            crate::bitmap::draw_bitmap_affine(target, &bmp, corners, [0.0, 0.0, w, h], 1.0, 1);
        }
    }
    true
}
