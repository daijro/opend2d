//! Image sources (`ID2D1ImageSourceFromWic`, `ID2D1TransformedImageSource`; D2D 1.1–1.3).
//!
//! Byte-exact here: an image source drawn through `DrawImage` at unit scale reduces to the proven
//! 1E bitmap blit. `ImageSourceViewProcessing::GetImage` (d2d1.i64) selects mip 0 at unit scale and
//! textures with the normal bitmap path, so `DrawImage(imageSource)` equals `DrawImage(bitmap)`. A
//! transformed image source's orientation is an axis-aligned reindex of the source texels (the 8
//! `D2D1_ORIENTATION` values); at scale 1 with nearest sampling every output texel centre lands on
//! a source texel centre, so the permutation is exact.
//!
//! Not reproduced (no byte-exact path, no scene): `D2D1_ALPHA_MODE` conversion of a
//! non-premultiplied WIC source (`AlphaConvertSource`), planar/YCbCr JPEG sources, and non-unit
//! `scaleX/scaleY` (which resample through the interpolation mode — the sampler's job, deferred to
//! the brush/sampling spec's residual).

use crate::bitmap::Bitmap;
use crate::context::DeviceContext;

/// A realized image source: its mip-0 pixels as a premultiplied BGRA8 bitmap, usable as a
/// `DrawImage`/`BlendImage` input exactly like a bitmap.
pub struct ImageSource {
    pub bitmap: Bitmap,
}

impl DeviceContext {
    /// `CreateImageSourceFromWic`: realize a (already-decoded, premultiplied) WIC source into the
    /// image source's backing bitmap. For a premultiplied/opaque source this is the identity
    /// realization — the mip-0 pixels are the source pixels.
    pub fn create_image_source_from_wic(&self, bitmap: Bitmap) -> ImageSource {
        ImageSource { bitmap }
    }

    /// `CreateTransformedImageSource`: apply `orientation` (a `D2D1_ORIENTATION` value) at unit
    /// scale. `scale`/`interpolation` are honoured only at scale 1 (the byte-exact case); the
    /// orientation is an axis-aligned reindex of the source texels.
    pub fn create_transformed_image_source(&self, source: &ImageSource, orientation: u32, _scale: [f32; 2], _interpolation: u32) -> ImageSource {
        ImageSource { bitmap: orient(&source.bitmap, orientation) }
    }
}

/// Reindex a bitmap's premultiplied BGRA8 texels under a `D2D1_ORIENTATION`. The enum values are
/// the Win32 metadata's, which are *not* sequential; the 90°/270°/transpose cases swap dimensions.
fn orient(src: &Bitmap, orientation: u32) -> Bitmap {
    let (w, h) = (src.width as i32, src.height as i32);
    let (ow, oh): (i32, i32) = match orientation {
        5 | 6 | 7 | 8 => (h, w), // 90 / 270 / transpose / anti-transpose swap axes
        _ => (w, h),
    };
    let mut out = vec![0u8; (ow * oh * 4) as usize];
    for oy in 0..oh {
        for ox in 0..ow {
            // (ox,oy) -> source (sx,sy) per orientation.
            let (sx, sy) = match orientation {
                2 => (w - 1 - ox, oy),         // FLIP_HORIZONTAL
                3 => (w - 1 - ox, h - 1 - oy), // ROTATE_CLOCKWISE180
                4 => (ox, h - 1 - oy),         // ROTATE180_FLIP_H == flip vertical
                5 => (oy, ox),                 // ROTATE90_FLIP_H == transpose
                6 => (w - 1 - oy, ox),         // ROTATE_CLOCKWISE270
                7 => (w - 1 - oy, h - 1 - ox), // ROTATE270_FLIP_H == anti-transpose
                8 => (oy, h - 1 - ox),         // ROTATE_CLOCKWISE90
                _ => (ox, oy),                 // DEFAULT (1) / unknown
            };
            let si = ((sy * w + sx) * 4) as usize;
            let di = ((oy * ow + ox) * 4) as usize;
            out[di..di + 4].copy_from_slice(&src.premul_bgra[si..si + 4]);
        }
    }
    Bitmap { width: ow as u32, height: oh as u32, premul_bgra: out, format: src.format, alpha: src.alpha, dpi: src.dpi }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile() -> Bitmap {
        // 2×1 straight RGBA: red, green.
        Bitmap::from_straight_rgba(2, 1, &[255, 0, 0, 255, 0, 255, 0, 255])
    }

    #[test]
    fn orientation_reindex() {
        let b = tile();
        // 180° reverses a 2×1 row; dims unchanged.
        let r = orient(&b, 3);
        assert_eq!((r.width, r.height), (2, 1));
        assert_eq!(&r.premul_bgra[0..4], &b.premul_bgra[4..8]);
        // 90° clockwise of a 2×1 is a 1×2 column; dims swap.
        let r90 = orient(&b, 8);
        assert_eq!((r90.width, r90.height), (1, 2));
        // DEFAULT is identity.
        assert_eq!(orient(&b, 1).premul_bgra, b.premul_bgra);
    }
}
