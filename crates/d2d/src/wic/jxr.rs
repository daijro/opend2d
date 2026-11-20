//! WIC JPEG-XR (HD Photo) decoding as the WIC JPEG-XR codec does it. JPEG-XR decode is fully specified by
//! ITU-T T.832, so a conformant decoder's samples are bit-identical to WIC's by construction. We
//! reuse `openreadout-jpegxr` (a `forbid(unsafe)` port of the decode path of Microsoft's own jxrlib,
//! bit-exact with it) and verified on the reference VM that `wmphoto`'s decode equals it byte-for-byte
//! across every sample type (8/16-bit unsigned, 16/32-bit fixed point, 16-bit half, 32-bit float) and
//! colour layout (grey, RGB, BGR).
//!
//! The one WIC-specific piece we write here is the container + alpha handling. A `.jxr` file is a
//! TIFF-like directory (`II` `0xBC` magic) whose tags point at the pixel-format GUID and the primary
//! image codestream; images with alpha store the alpha as a *separate* Y-only codestream. We parse the
//! directory, decode the primary (and, for alpha formats, the alpha) codestream with `openreadout`,
//! reorder colour channels to the WIC byte order and interleave the alpha plane. The result matches
//! `IWICBitmapFrameDecode::CopyPixels` byte-for-byte.
//!
//! The *encoder* is not reproduced: byte-identical `.jxr` output across WIC's option matrix is decided
//! by `wmphoto`'s own quantiser/mode choices, which no reusable crate reproduces — see the module docs
//! in `mod.rs` and notes. Unsupported-but-rare native formats (CMYK, RGBE, YCC, packed 5/6/10-bit,
//! bilevel, N-component, and 3-channel-in-a-padded-unit like `32bppBGR`) return [`WicError::Unsupported`].

use super::WicError;
use openreadout_jpegxr::{decode, Image};

/// Cap on a decoded plane's allocation, passed to `openreadout` so a malformed header cannot ask for
/// an unbounded buffer. Larger than any real frame WIC produces.
const DECODE_LIMIT: usize = 1 << 30;

/// A decoded JPEG-XR frame in its WIC native pixel format: the `GUID_WICPixelFormat*` WIC reports (the
/// raw 16 little-endian bytes), the size, the WIC stride (`⌈width·bpp/8⌉`) and the tightly-strided
/// pixels, byte-identical to `IWICBitmapFrameDecode::CopyPixels`.
#[derive(Debug, Clone)]
pub struct JxrImage {
    pub width: u32,
    pub height: u32,
    pub format: [u8; 16],
    pub bpp: u32,
    pub stride: usize,
    pub pixels: Vec<u8>,
}

impl JxrImage {
    /// The canonical `GUID_WICPixelFormat*` string, e.g. `6fddc324-4e03-4bfe-b185-3d77768dc90f`.
    pub fn guid(&self) -> String {
        let g = &self.format;
        format!(
            "{:08x}-{:04x}-{:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            u32::from_le_bytes([g[0], g[1], g[2], g[3]]),
            u16::from_le_bytes([g[4], g[5]]),
            u16::from_le_bytes([g[6], g[7]]),
            g[8], g[9], g[10], g[11], g[12], g[13], g[14], g[15],
        )
    }
}

/// How a supported format's WIC pixels are built from the decoded codestream(s).
#[derive(Clone, Copy)]
enum Layout {
    /// A single codestream whose bytes already match WIC (grey/RGB/BGR, tightly packed).
    Direct,
    /// Four channels B,G,R,A: colour codestream decodes as R,G,B (reorder) + separate alpha plane.
    Bgra,
    /// Four channels R,G,B,A: colour codestream decodes as R,G,B + separate alpha plane.
    Rgba,
}

/// WIC native format descriptor, keyed by the final byte of the `6fddc324-…` GUID family.
struct Format {
    bpp: u32,
    layout: Layout,
}

/// Map a WIC pixel-format GUID to how we decode it, or `None` for formats `openreadout` does not
/// decode (CMYK/RGBE/YCC/packed/bilevel/N-component) or that pad three channels into a wider unit.
fn describe(guid: &[u8; 16]) -> Option<Format> {
    // All WIC JPEG-XR native formats share this GUID but for the final byte (the stored GUID is
    // `{6fddc324-4e03-4bfe-b185-3d77768dc9XX}`: data1/2/3 little-endian, data4 as-is).
    const PREFIX: [u8; 15] =
        [0x24, 0xc3, 0xdd, 0x6f, 0x03, 0x4e, 0xfe, 0x4b, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9];
    if guid[..15] != PREFIX {
        return None;
    }
    use Layout::*;
    let (bpp, layout) = match guid[15] {
        0x08 => (8, Direct),   // 8bppGray
        0x0b => (16, Direct),  // 16bppGray
        0x13 => (16, Direct),  // 16bppGrayFixedPoint
        0x3e => (16, Direct),  // 16bppGrayHalf
        0x3f => (32, Direct),  // 32bppGrayFixedPoint
        0x11 => (32, Direct),  // 32bppGrayFloat
        0x0c => (24, Direct),  // 24bppBGR
        0x0d => (24, Direct),  // 24bppRGB
        0x15 => (48, Direct),  // 48bppRGB
        0x12 => (48, Direct),  // 48bppRGBFixedPoint
        0x3b => (48, Direct),  // 48bppRGBHalf
        0x18 => (96, Direct),  // 96bppRGBFixedPoint
        0x0f => (32, Bgra),    // 32bppBGRA
        0x10 => (32, Bgra),    // 32bppPBGRA
        0x16 => (64, Rgba),    // 64bppRGBA
        0x17 => (64, Rgba),    // 64bppPRGBA
        0x1d => (64, Rgba),    // 64bppRGBAFixedPoint
        0x3a => (64, Rgba),    // 64bppRGBAHalf
        0x19 => (128, Rgba),   // 128bppRGBAFloat
        0x1a => (128, Rgba),   // 128bppPRGBAFloat
        0x1e => (128, Rgba),   // 128bppRGBAFixedPoint
        _ => return None,
    };
    Some(Format { bpp, layout })
}

/// The pixel-format GUID and codestream slices read from a `.jxr` container directory.
struct Container {
    guid: [u8; 16],
    image: (usize, usize),
    alpha: Option<(usize, usize)>,
}

fn le16(d: &[u8], at: usize) -> Result<u16, WicError> {
    d.get(at..at + 2).map(|b| u16::from_le_bytes([b[0], b[1]])).ok_or(WicError::BadImage)
}

fn le32(d: &[u8], at: usize) -> Result<u32, WicError> {
    d.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).ok_or(WicError::BadImage)
}

/// Parse the HD Photo container (TIFF-like `II` `0xBC`): first image directory only. Tags: `0xBC01`
/// pixel-format GUID (offset), `0xBCC0/0xBCC1` image codestream offset/length, `0xBCC2/0xBCC3` alpha
/// codestream offset/length.
fn parse_container(data: &[u8]) -> Result<Container, WicError> {
    if data.len() < 8 || &data[..2] != b"II" || data[2] != 0xBC || data[3] > 1 {
        return Err(WicError::BadImage);
    }
    let dir = le32(data, 4)? as usize;
    let count = le16(data, dir)?;
    if count == 0 || count == u16::MAX {
        return Err(WicError::BadImage);
    }
    let (mut guid, mut img_off, mut img_len, mut alp_off, mut alp_len) = (None, None, 0u32, 0usize, 0u32);
    for i in 0..usize::from(count) {
        let at = dir + 2 + 12 * i;
        let value = le32(data, at + 8)?;
        match le16(data, at)? {
            0xBC01 => {
                let g = data.get(value as usize..value as usize + 16).ok_or(WicError::BadImage)?;
                guid = Some(<[u8; 16]>::try_from(g).unwrap());
            }
            0xBCC0 => img_off = Some(value as usize),
            0xBCC1 => img_len = value,
            0xBCC2 => alp_off = value as usize,
            0xBCC3 => alp_len = value,
            _ => {}
        }
    }
    let guid = guid.ok_or(WicError::BadImage)?;
    let image = slice_bounds(data, img_off.ok_or(WicError::BadImage)?, img_len)?;
    let alpha = if alp_off != 0 { Some(slice_bounds(data, alp_off, alp_len)?) } else { None };
    Ok(Container { guid, image, alpha })
}

/// A `(offset, len)` codestream range, clamped to the file (a zero byte-count means "to end").
fn slice_bounds(data: &[u8], off: usize, len: u32) -> Result<(usize, usize), WicError> {
    if off >= data.len() {
        return Err(WicError::BadImage);
    }
    let end = if len == 0 { data.len() } else { data.len().min(off + len as usize) };
    Ok((off, end))
}

fn map_err(e: openreadout_jpegxr::Error) -> WicError {
    match e {
        openreadout_jpegxr::Error::Unsupported(_) => WicError::Unsupported,
        _ => WicError::BadImage,
    }
}

/// Decode a JPEG-XR / HD Photo file to its WIC native pixel format, byte-for-byte as the WIC JPEG-XR codec.
pub fn decode_jxr(data: &[u8]) -> Result<JxrImage, WicError> {
    let c = parse_container(data)?;
    let fmt = describe(&c.guid).ok_or(WicError::Unsupported)?;

    let (width, height, pixels) = match fmt.layout {
        Layout::Direct => {
            // openreadout parses the container itself and emits WIC-ordered bytes for these formats.
            let img = decode(data, DECODE_LIMIT).map_err(map_err)?;
            (img.width, img.height, img.data)
        }
        Layout::Bgra | Layout::Rgba => {
            let color = decode(&data[c.image.0..c.image.1], DECODE_LIMIT).map_err(map_err)?;
            let (ao, ae) = c.alpha.ok_or(WicError::Unsupported)?;
            let alpha = decode(&data[ao..ae], DECODE_LIMIT).map_err(map_err)?;
            let bgr = matches!(fmt.layout, Layout::Bgra);
            (color.width, color.height, interleave_alpha(&color, &alpha, bgr)?)
        }
    };

    let bpp = fmt.bpp;
    let stride = ((width * bpp + 7) / 8) as usize;
    Ok(JxrImage { width, height, format: c.guid, bpp, stride, pixels })
}

/// Build 4-channel WIC pixels from a 3-channel colour plane and a 1-channel alpha plane. `bgr` swaps
/// the colour channels (R,G,B → B,G,R) for the BGRA formats; the alpha goes last. All channels share
/// the colour plane's sample width.
fn interleave_alpha(color: &Image, alpha: &Image, bgr: bool) -> Result<Vec<u8>, WicError> {
    if color.channels != 3 || alpha.channels != 1 || alpha.sample != color.sample
        || alpha.width != color.width || alpha.height != color.height
    {
        return Err(WicError::BadImage);
    }
    let sb = color.sample.bytes();
    let n = color.width as usize * color.height as usize;
    if color.data.len() != n * 3 * sb || alpha.data.len() != n * sb {
        return Err(WicError::BadImage);
    }
    let mut out = vec![0u8; n * 4 * sb];
    for p in 0..n {
        for ch in 0..3 {
            let src = if bgr { 2 - ch } else { ch };
            out[(p * 4 + ch) * sb..(p * 4 + ch) * sb + sb]
                .copy_from_slice(&color.data[(p * 3 + src) * sb..(p * 3 + src) * sb + sb]);
        }
        out[(p * 4 + 3) * sb..(p * 4 + 3) * sb + sb]
            .copy_from_slice(&alpha.data[p * sb..p * sb + sb]);
    }
    Ok(out)
}
