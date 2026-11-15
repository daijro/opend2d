//! `IWICBitmapFlipRotator` and `IWICBitmapClipper`, as WIC does them. Both are pure
//! geometric rearrangements of whole pixels: the pixel format and each pixel's bytes are preserved,
//! only positions change, so the output is byte-identical to the real codec by construction (spot
//! checks against it under `wic-dump fliprotate`/`clip`).
//!
//! Flip-rotate order matches native: the horizontal/vertical flips are applied first, then the
//! rotation. Rotations are clockwise. Transform options are the `WICBitmapTransformOptions` bitfield
//! (rotation in the low two bits; `FlipHorizontal = 8`, `FlipVertical = 16`).
//!
//! These operate on whole-byte pixel formats (bpp a multiple of 8 — every format Direct2D flips,
//! rotates or clips is). Sub-byte (1/2/4bpp indexed/gray) would need bit-level shifting and is left
//! unsupported rather than guessed at.

use super::WicError;

/// Flip and/or rotate `w`×`h` pixels (`bpp` bits each, a multiple of 8). Returns the rearranged
/// bytes with the new `(width, height)` — swapped for the 90°/270° rotations.
pub fn flip_rotate(
    pixels: &[u8],
    w: u32,
    h: u32,
    bpp: u32,
    options: u32,
) -> Result<(Vec<u8>, u32, u32), WicError> {
    if bpp == 0 || bpp % 8 != 0 {
        return Err(WicError::Unsupported);
    }
    let bytes = (bpp / 8) as usize;
    let (w, h) = (w as usize, h as usize);
    if pixels.len() < w * h * bytes {
        return Err(WicError::BadImage);
    }
    let rot = options & 3;
    let (flip_h, flip_v) = (options & 8 != 0, options & 16 != 0);
    let (ow, oh) = if rot == 1 || rot == 3 { (h, w) } else { (w, h) };

    let mut out = vec![0u8; ow * oh * bytes];
    for oy in 0..oh {
        for ox in 0..ow {
            // Undo the rotation to the pre-rotation coordinate, then undo the flips.
            let (mut sx, mut sy) = match rot {
                0 => (ox, oy),
                1 => (oy, h - 1 - ox),         // 90° clockwise
                2 => (w - 1 - ox, h - 1 - oy), // 180°
                _ => (w - 1 - oy, ox),         // 270° clockwise
            };
            if flip_h {
                sx = w - 1 - sx;
            }
            if flip_v {
                sy = h - 1 - sy;
            }
            let s = (sy * w + sx) * bytes;
            let d = (oy * ow + ox) * bytes;
            out[d..d + bytes].copy_from_slice(&pixels[s..s + bytes]);
        }
    }
    Ok((out, ow as u32, oh as u32))
}

/// Crop `w`×`h` pixels (`bpp` bits each, a multiple of 8) to the rectangle `(x, y, cw, ch)`, exactly
/// as `IWICBitmapClipper`. Returns the cropped bytes (tightly strided to `cw`).
pub fn clip(
    pixels: &[u8],
    w: u32,
    h: u32,
    bpp: u32,
    x: u32,
    y: u32,
    cw: u32,
    ch: u32,
) -> Result<Vec<u8>, WicError> {
    if bpp == 0 || bpp % 8 != 0 {
        return Err(WicError::Unsupported);
    }
    if x + cw > w || y + ch > h {
        return Err(WicError::BadImage);
    }
    let bytes = (bpp / 8) as usize;
    let (src_stride, dst_stride) = (w as usize * bytes, cw as usize * bytes);
    let mut out = vec![0u8; dst_stride * ch as usize];
    for row in 0..ch as usize {
        let s = (y as usize + row) * src_stride + x as usize * bytes;
        out[row * dst_stride..(row + 1) * dst_stride].copy_from_slice(&pixels[s..s + dst_stride]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2×2, 1 byte/pixel, labelled a b / c d.
    const IMG: &[u8] = &[b'a', b'b', b'c', b'd'];

    #[test]
    fn rotate_180_reverses() {
        let (out, w, h) = flip_rotate(IMG, 2, 2, 8, 2).unwrap();
        assert_eq!((w, h, &out[..]), (2, 2, &b"dcba"[..]));
    }

    #[test]
    fn rotate_90_swaps_dims() {
        // 90° clockwise of a b / c d  ->  c a / d b
        let (out, w, h) = flip_rotate(IMG, 2, 2, 8, 1).unwrap();
        assert_eq!((w, h, &out[..]), (2, 2, &b"cadb"[..]));
    }

    #[test]
    fn flip_horizontal() {
        let (out, _, _) = flip_rotate(IMG, 2, 2, 8, 8).unwrap();
        assert_eq!(&out[..], b"badc");
    }

    #[test]
    fn clip_subrect() {
        // right column of a b / c d -> b / d
        let out = clip(IMG, 2, 2, 8, 1, 0, 1, 2).unwrap();
        assert_eq!(&out[..], b"bd");
    }
}
