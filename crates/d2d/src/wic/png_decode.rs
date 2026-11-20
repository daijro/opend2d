//! PNG → WIC native frame. The `png` crate gives inflated, unfiltered, de-interlaced samples in the
//! file's own color type / bit depth (`Transformations::IDENTITY`, row-byte-padded exactly like
//! WIC). All that is left is the WIC-specific native-format mapping and byte layout, read off the VM
//! oracle:
//!   * grayscale 1/2/4/8 and palette 1/2/4/8 → the matching BlackWhite/Gray/Indexed format, bytes
//!     copied through unchanged (same packing as WIC);
//!   * grayscale 16 → 16bppGray, each sample byte-swapped to little-endian;
//!   * truecolor 8 → 24bppBGR, 16 → 48bppRGB (channels stay RGB, each sample → LE);
//!   * truecolor+alpha 8 → 32bppBGRA (straight), 16 → 64bppRGBA (→ LE);
//!   * grayscale+alpha 8 → 32bppBGRA with B=G=R=gray, 16 → 64bppRGBA with R=G=B=gray.
//! Palette entries become `0xAARRGGBB`, alpha from `tRNS` (255 beyond its length).

use super::{DecodedImage, WicError, WicFormat};
use png::{BitDepth, ColorType, Decoder, Transformations};

/// The decompressed iCCP ICC profile, if the PNG embeds one (the `png` crate inflates it). WIC's
/// `IWICBitmapFrameDecode::GetColorContexts` returns these bytes for a PNG with an iCCP chunk.
pub fn png_icc_profile(bytes: &[u8]) -> Option<Vec<u8>> {
    let mut decoder = Decoder::new(bytes);
    decoder.set_transformations(Transformations::IDENTITY);
    let reader = decoder.read_info().ok()?;
    reader.info().icc_profile.as_ref().map(|p| p.to_vec())
}

pub fn decode_png(bytes: &[u8]) -> Result<DecodedImage, WicError> {
    let mut decoder = Decoder::new(bytes);
    decoder.set_transformations(Transformations::IDENTITY);
    // The header/structure scan (signature, IHDR, and every chunk up to the first IDAT, including the
    // palette) is the decoder-creation phase: a malformed or truncated chunk here — a short PLTE, a
    // missing IDAT — makes WIC reject the stream as an unrecognised image, not a corrupt one. Only a
    // failure decompressing the IDAT scanlines below is reported as a bad image.
    let mut reader = decoder.read_info().map_err(|_| WicError::UnknownImageFormat)?;

    // Snapshot everything needed before the mutable `next_frame` borrow.
    let (width, height, color, depth, plte, trns) = {
        let info = reader.info();
        (
            info.width,
            info.height,
            info.color_type,
            info.bit_depth,
            info.palette.as_ref().map(|p| p.to_vec()),
            info.trns.as_ref().map(|t| t.to_vec()),
        )
    };

    // Decode the scanlines into a zero-initialised buffer. A truncated IDAT (the stream ends partway
    // through the image data) is decoded leniently, exactly as WIC does: the complete scanlines are
    // kept and every row that could not be decoded stays zero — the image comes back fully sized with
    // its tail filled with zeros rather than failing. `next_frame` writes each row in turn and stops
    // at the truncation point, so the already-decoded rows are left in `buf`.
    let mut buf = vec![0u8; reader.output_buffer_size()];
    if let Ok(frame) = reader.next_frame(&mut buf) {
        buf.truncate(frame.buffer_size());
    }

    let (format, pixels) = map(color, depth, buf)?;
    let palette = build_palette(format, plte, trns);
    Ok(DecodedImage {
        width,
        height,
        format,
        stride: DecodedImage::stride_for(width, format),
        pixels,
        palette,
    })
}

/// Swap every 16-bit sample from PNG big-endian to WIC little-endian, in place.
fn swap16(mut v: Vec<u8>) -> Vec<u8> {
    for pair in v.chunks_exact_mut(2) {
        pair.swap(0, 1);
    }
    v
}

fn map(color: ColorType, depth: BitDepth, buf: Vec<u8>) -> Result<(WicFormat, Vec<u8>), WicError> {
    use BitDepth::*;
    use ColorType::*;
    Ok(match (color, depth) {
        // Grayscale and indexed sub-byte/8-bit: WIC copies the packed samples through unchanged.
        (Grayscale, One) => (WicFormat::BlackWhite, buf),
        (Grayscale, Two) => (WicFormat::Gray2, buf),
        (Grayscale, Four) => (WicFormat::Gray4, buf),
        (Grayscale, Eight) => (WicFormat::Gray8, buf),
        (Grayscale, Sixteen) => (WicFormat::Gray16, swap16(buf)),
        (Indexed, One) => (WicFormat::Indexed1, buf),
        (Indexed, Two) => (WicFormat::Indexed2, buf),
        (Indexed, Four) => (WicFormat::Indexed4, buf),
        (Indexed, Eight) => (WicFormat::Indexed8, buf),

        // Truecolor → BGR (8) / RGB (16, little-endian samples).
        (Rgb, Eight) => (WicFormat::Bgr24, reorder(&buf, 3, &[2, 1, 0])),
        (Rgb, Sixteen) => (WicFormat::Rgb48, swap16(buf)),
        (Rgba, Eight) => (WicFormat::Bgra32, reorder(&buf, 4, &[2, 1, 0, 3])),
        (Rgba, Sixteen) => (WicFormat::Rgba64, swap16(buf)),

        // Grayscale+alpha is expanded: gray replicated across RGB, alpha kept straight.
        (GrayscaleAlpha, Eight) => (
            WicFormat::Bgra32,
            buf.chunks_exact(2).flat_map(|p| [p[0], p[0], p[0], p[1]]).collect(),
        ),
        (GrayscaleAlpha, Sixteen) => (
            WicFormat::Rgba64,
            buf.chunks_exact(4)
                .flat_map(|p| [p[1], p[0], p[1], p[0], p[1], p[0], p[3], p[2]])
                .collect(),
        ),
        _ => return Err(WicError::Unsupported),
    })
}

/// Permute the channels of each `n`-byte pixel by `order` (byte indices into the source pixel).
fn reorder(buf: &[u8], n: usize, order: &[usize]) -> Vec<u8> {
    buf.chunks_exact(n).flat_map(|p| order.iter().map(|&i| p[i])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncated_structure_is_unknown_image_format() {
        // Valid signature + IHDR (8x5, 1bpp indexed), then a PLTE chunk that claims 3 data bytes but
        // the stream ends after 2, with no IDAT: the chunk scan up to IDAT fails. WIC reports this as
        // an unrecognised image during decoder creation, not a bad image.
        let bytes = [
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, // signature
            0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52, // IHDR len + type
            0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x05, 0x01, 0x03, 0x00, 0x00, 0x00, // IHDR data
            0x42, 0x5f, 0x3f, 0x16, // IHDR CRC
            0x00, 0x00, 0x00, 0x03, 0x50, 0x4c, 0x54, 0x45, 0x00, 0x00, // PLTE len+type, 2/3 bytes
        ];
        assert!(matches!(decode_png(&bytes), Err(WicError::UnknownImageFormat)));
    }
}

/// WIC palette: `0xAARRGGBB` per `PLTE` entry, alpha from `tRNS` (255 where `tRNS` is shorter).
fn build_palette(format: WicFormat, plte: Option<Vec<u8>>, trns: Option<Vec<u8>>) -> Vec<u32> {
    use WicFormat::*;
    if !matches!(format, Indexed1 | Indexed2 | Indexed4 | Indexed8) {
        return Vec::new();
    }
    let plte = plte.unwrap_or_default();
    let trns = trns.unwrap_or_default();
    plte.chunks_exact(3)
        .enumerate()
        .map(|(i, c)| {
            let a = trns.get(i).copied().unwrap_or(255) as u32;
            (a << 24) | ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | c[2] as u32
        })
        .collect()
}
