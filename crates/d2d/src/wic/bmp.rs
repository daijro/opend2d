//! BMP → WIC native frame, reproducing WIC's BMP decoder.
//!
//! The native pixel format is decided exactly as WIC's BMP decoder does it:
//! bit depth plus `biCompression`/masks/header-size select one of
//! 1/4/8bppIndexed, 16bppBGR555/BGR565, 24bppBGR, 32bppBGR/BGRA or 64bppRGBAFixedPoint, with any
//! compressed (RLE) or non-standard-bitfield source collapsing to 32bppBGR.
//!
//! Pixels come straight from the file: the BI_RGB read path is a vertically-flipped copy of the tight
//! `⌈w·bpp/8⌉` row bytes (64bpp additionally swaps R/B), and RLE4/RLE8 expand through the palette
//! into 32bppBGR via `ConvertRLEto32RGBA` (buffer pre-filled with 0xFF000000, bottom-up).
//!
//! Not reproduced: the GDI fallback WIC uses for non-standard 16/32bpp bitfield masks
//! (`ReadLine_GDI`) and OS/2 v1 core headers — both return `Unsupported`.

use super::convert::{convert, Format};
use super::WicError;

/// WIC `GUID_WICPixelFormat*` GUID strings (shared family prefix `6fddc324-…-3d77768dc9`).
pub mod fmt {
    pub const IDX1: &str = "6fddc324-4e03-4bfe-b185-3d77768dc901";
    pub const IDX4: &str = "6fddc324-4e03-4bfe-b185-3d77768dc903";
    pub const IDX8: &str = "6fddc324-4e03-4bfe-b185-3d77768dc904";
    pub const BGR555: &str = "6fddc324-4e03-4bfe-b185-3d77768dc909";
    pub const BGR565: &str = "6fddc324-4e03-4bfe-b185-3d77768dc90a";
    pub const BGR24: &str = "6fddc324-4e03-4bfe-b185-3d77768dc90c";
    pub const BGR32: &str = "6fddc324-4e03-4bfe-b185-3d77768dc90e";
    pub const BGRA32: &str = "6fddc324-4e03-4bfe-b185-3d77768dc90f";
    pub const RGB24: &str = "6fddc324-4e03-4bfe-b185-3d77768dc90d";
    pub const PBGRA32: &str = "6fddc324-4e03-4bfe-b185-3d77768dc910";
    pub const RGBA64FP: &str = "6fddc324-4e03-4bfe-b185-3d77768dc91d";
    // Grayscale / wide-RGB formats WIC's TIFF decoder also reports.
    pub const BLACKWHITE: &str = "6fddc324-4e03-4bfe-b185-3d77768dc905";
    pub const GRAY4: &str = "6fddc324-4e03-4bfe-b185-3d77768dc907";
    pub const GRAY8: &str = "6fddc324-4e03-4bfe-b185-3d77768dc908";
    pub const GRAY16: &str = "6fddc324-4e03-4bfe-b185-3d77768dc90b";
    pub const RGB48: &str = "6fddc324-4e03-4bfe-b185-3d77768dc915";
    pub const RGBA64: &str = "6fddc324-4e03-4bfe-b185-3d77768dc916";

    pub fn bits(format: &str) -> u32 {
        match format {
            IDX1 | BLACKWHITE => 1,
            IDX4 | GRAY4 => 4,
            IDX8 | GRAY8 => 8,
            BGR555 | BGR565 | GRAY16 => 16,
            BGR24 => 24,
            BGR32 | BGRA32 | PBGRA32 => 32,
            RGB48 => 48,
            RGBA64 | RGBA64FP => 64,
            _ => 0,
        }
    }
}

/// A decoded WIC frame (shared by the bmp/gif/ico/tiff decoders).
#[derive(Debug, Clone)]
pub struct Decoded {
    pub width: u32,
    pub height: u32,
    /// WIC `GUID_WICPixelFormat*` string.
    pub format: String,
    pub stride: usize,
    pub pixels: Vec<u8>,
    /// Palette as `0xAARRGGBB` (indexed formats only; empty otherwise).
    pub palette: Vec<u32>,
}

impl Decoded {
    pub fn bpp(&self) -> u32 {
        fmt::bits(&self.format)
    }
}

fn u16(b: &[u8], o: usize) -> Option<u32> {
    Some(u16::from_le_bytes([*b.get(o)?, *b.get(o + 1)?]) as u32)
}
fn u32(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes([*b.get(o)?, *b.get(o + 1)?, *b.get(o + 2)?, *b.get(o + 3)?]))
}
fn i32(b: &[u8], o: usize) -> Option<i32> {
    u32(b, o).map(|v| v as i32)
}

struct Header {
    width: u32,
    height: u32,
    top_down: bool,
    bpp: u32,
    comp: u32,
    clr_used: u32,
    bi_size: u32,
    rmask: u32,
    gmask: u32,
    bmask: u32,
    amask: u32,
    off_bits: usize,
    pal_off: usize,
}

fn parse_header(b: &[u8]) -> Result<Header, WicError> {
    // WIC reads the 14-byte BITMAPFILEHEADER first; a shorter stream fails the read (STREAMREAD),
    // and a readable header with the wrong `bfType` is UNKNOWNIMAGEFORMAT (both confirmed via Wine).
    if b.len() < 14 {
        return Err(WicError::StreamRead);
    }
    if &b[0..2] != b"BM" {
        return Err(WicError::UnknownImageFormat);
    }
    let off_bits = u32(b, 10).ok_or(WicError::StreamRead)? as usize;
    let bi_size = u32(b, 14).ok_or(WicError::StreamRead)?;
    // Only BITMAPINFOHEADER (40) and its V2/V3/V4/V5 extensions. OS/2 v1 core (12) uses the GDI
    // path in WIC; not reproduced here.
    if bi_size < 40 {
        return Err(WicError::Unsupported);
    }
    let w = i32(b, 18).ok_or(WicError::StreamRead)?;
    let h = i32(b, 22).ok_or(WicError::StreamRead)?;
    let bpp = u16(b, 28).ok_or(WicError::StreamRead)?;
    let comp = u32(b, 30).ok_or(WicError::StreamRead)?;
    let clr_used = u32(b, 46).ok_or(WicError::StreamRead)?;
    if w <= 0 {
        return Err(WicError::BadImage);
    }
    let top_down = h < 0;
    let height = h.unsigned_abs();
    // Reject absurd dimensions before any decode path allocates its output. The RLE and bitfields
    // paths size their buffer from `width * height * 4` independently of the file length, so a
    // crafted header (e.g. height near i32::MAX in a tiny file) would attempt a multi-GB allocation
    // and abort the process. No BMP expands by more than ~2^20× from file bytes to decoded pixels.
    match (w as u64).checked_mul(height as u64).and_then(|n| n.checked_mul(4)) {
        Some(n) if n <= (b.len() as u64).saturating_mul(1 << 20) => {}
        _ => return Err(WicError::BadImage),
    }
    // Bitfield masks (only consulted for BI_BITFIELDS). For a 40-byte header they follow the
    // header; for V2+ they live inside it. Either way they start at byte 54.
    let (mut rmask, mut gmask, mut bmask, mut amask) = (0, 0, 0, 0);
    if comp == 3 {
        rmask = u32(b, 54).ok_or(WicError::StreamRead)?;
        gmask = u32(b, 58).ok_or(WicError::StreamRead)?;
        bmask = u32(b, 62).ok_or(WicError::StreamRead)?;
        if bi_size >= 56 {
            amask = u32(b, 66).ok_or(WicError::StreamRead)?;
        }
    }
    // Colour table begins after the header (+ the 3 inline masks when comp==3 && bi_size==40).
    let pal_off = 14 + bi_size as usize + if comp == 3 && bi_size == 40 { 12 } else { 0 };
    Ok(Header {
        width: w as u32,
        height,
        top_down,
        bpp,
        comp,
        clr_used,
        bi_size,
        rmask,
        gmask,
        bmask,
        amask,
        off_bits,
        pal_off,
    })
}

/// WIC's BMP pixel-format selection.
fn pixel_format(h: &Header) -> Result<&'static str, WicError> {
    use fmt::*;
    let base = match h.bpp {
        1 => IDX1,
        4 => IDX4,
        8 => IDX8,
        16 => {
            if h.comp == 3 {
                if h.rmask == 0xF800 {
                    if h.gmask == 0x07E0 && h.bmask == 0x001F { BGR565 } else { BGR32 }
                } else if h.rmask == 0x7C00 && h.gmask == 0x03E0 && h.bmask == 0x001F {
                    BGR555
                } else {
                    BGR32
                }
            } else {
                BGR555
            }
        }
        24 => BGR24,
        32 => {
            if h.comp == 3
                && (h.bi_size == 108 || h.bi_size == 124)
                && h.bmask == 0x0000_00FF
                && h.gmask == 0x0000_FF00
                && h.rmask == 0x00FF_0000
                && h.amask == 0xFF00_0000
            {
                BGRA32
            } else {
                BGR32
            }
        }
        _ => return Err(WicError::UnknownImageFormat),
    };
    // Post-pass: any compression collapses to 32bppBGR except standard bitfields.
    if h.comp != 0 {
        if h.comp != 3 {
            return Ok(BGR32);
        }
        if base != BGR555 && base != BGR565 && base != BGRA32 {
            return Ok(BGR32);
        }
    }
    Ok(base)
}

/// Colour-table entry count WIC reads (`GetColorTableCount`): `biClrUsed` clamped to `1<<bpp`,
/// or the full `1<<bpp` when `biClrUsed` is zero.
fn color_count(h: &Header) -> u32 {
    if !matches!(h.bpp, 1 | 2 | 4 | 8) {
        return 0;
    }
    let max = 1u32 << h.bpp;
    match h.clr_used {
        0 => max,
        n if n > max => max,
        n => n,
    }
}

fn read_palette(b: &[u8], h: &Header) -> Vec<u32> {
    let count = color_count(h) as usize;
    let mut pal = Vec::with_capacity(count);
    for i in 0..count {
        let o = h.pal_off + i * 4;
        let (bl, gr, rd) = (
            b.get(o).copied().unwrap_or(0),
            b.get(o + 1).copied().unwrap_or(0),
            b.get(o + 2).copied().unwrap_or(0),
        );
        pal.push(0xFF00_0000 | (rd as u32) << 16 | (gr as u32) << 8 | bl as u32);
    }
    pal
}

/// Raw 4-byte BGRX palette entries, as the RLE expander consumes them.
fn read_palette_raw(b: &[u8], h: &Header) -> Vec<[u8; 4]> {
    let count = color_count(h) as usize;
    (0..count)
        .map(|i| {
            let o = h.pal_off + i * 4;
            [
                b.get(o).copied().unwrap_or(0),
                b.get(o + 1).copied().unwrap_or(0),
                b.get(o + 2).copied().unwrap_or(0),
                b.get(o + 3).copied().unwrap_or(0),
            ]
        })
        .collect()
}

pub fn decode_bmp(bytes: &[u8]) -> Result<Decoded, WicError> {
    let h = parse_header(bytes)?;
    let format = pixel_format(&h)?;

    if h.comp == 1 || h.comp == 2 {
        return decode_rle(bytes, &h, format);
    }
    // Direct BI_RGB / standard-bitfields copy. Non-standard bitfields would need WIC's GDI path.
    if h.comp == 3 {
        let standard = matches!(format, fmt::BGR555 | fmt::BGR565 | fmt::BGRA32)
            || (format == fmt::BGR32
                && h.bpp == 32
                && h.bmask == 0x0000_00FF
                && h.gmask == 0x0000_FF00
                && h.rmask == 0x00FF_0000
                && h.amask == 0);
        if !standard {
            return Err(WicError::Unsupported);
        }
    }

    let stride = ((h.width * h.bpp + 7) / 8) as usize;
    let src_stride = ((((h.bpp * h.width + 7) >> 3) + 3) & !3) as usize;
    // The pixel data must be present before we allocate `stride * height`: a crafted BMP declaring huge
    // dimensions (e.g. width/height near i32::MAX in a tiny file) would otherwise attempt a multi-TB
    // allocation and abort the process. If the file does not hold `off_bits + src_stride * height`
    // bytes, the image is truncated/absurd — reject it (native returns BADIMAGE), which also bounds the
    // allocation below (stride ≤ src_stride ⇒ stride·height ≤ the file length).
    match src_stride.checked_mul(h.height as usize).and_then(|n| n.checked_add(h.off_bits)) {
        Some(end) if end <= bytes.len() => {}
        _ => return Err(WicError::BadImage),
    }
    let mut pixels = vec![0u8; stride * h.height as usize];
    for y in 0..h.height as usize {
        let src_row = if h.top_down { y } else { h.height as usize - 1 - y };
        let so = h.off_bits + src_row * src_stride;
        let src = bytes.get(so..so + stride).ok_or(WicError::BadImage)?;
        pixels[y * stride..y * stride + stride].copy_from_slice(src);
    }
    if format == fmt::RGBA64FP {
        for px in pixels.chunks_exact_mut(8) {
            px.swap(0, 4);
            px.swap(1, 5);
        }
    }

    let palette = if matches!(format, fmt::IDX1 | fmt::IDX4 | fmt::IDX8) {
        read_palette(bytes, &h)
    } else {
        Vec::new()
    };
    Ok(Decoded { width: h.width, height: h.height, format: format.into(), stride, pixels, palette })
}

/// Encode a frame as WIC's BMP encoder (`CBmpFrameEncode`) does: a BITMAPFILEHEADER +
/// 40-byte BITMAPINFOHEADER (BI_BITFIELDS + 3 masks for 16bppBGR565), optional BGRX palette, then
/// bottom-up DWORD-padded scanlines. `dpi` sets the pels-per-metre fields (round(dpi·100/2.54)).
/// Supports the uncompressed formats the encoder accepts: 1/4/8bppIndexed, 16bppBGR555/565,
/// 24bppBGR, 32bppBGR/BGRA.
pub fn encode_bmp(img: &Decoded, dpi: f32) -> Result<Vec<u8>, WicError> {
    // Native's BMP WriteSource maps sources BMP can't store directly: 24bppRGB → 24bppBGR (swap),
    // 8bppGray → 8bppIndexed (same median-cut + error-diffusion converter as GIF — BMP treats 8bpp as
    // a palettised image). 32bppPBGRA is stored verbatim as 32bpp (native does NOT un-premultiply —
    // it copies the premultiplied bytes), handled by the direct 32bpp arm below.
    let conv;
    let img = match img.format.as_str() {
        fmt::RGB24 => {
            let sf = Format::from_guid(&img.format).ok_or(WicError::Unsupported)?;
            let px = convert(sf, Format::Bgr24, &img.pixels, img.width, img.height, &img.palette)?;
            conv = Decoded {
                width: img.width,
                height: img.height,
                format: fmt::BGR24.into(),
                stride: img.width as usize * 3,
                pixels: px,
                palette: Vec::new(),
            };
            &conv
        }
        fmt::GRAY8 => {
            conv = super::gif::error_diffuse_to_indexed8(img)?;
            &conv
        }
        // A 64bppRGBA source is stored as 64bppRGBAFixedPoint: colour channels decoded to linear scRGB,
        // alpha kept linear, both on the s2.13 fixed-point grid, in B,G,R,A order.
        fmt::RGBA64 => {
            conv = Decoded {
                width: img.width,
                height: img.height,
                format: fmt::RGBA64FP.into(),
                stride: img.width as usize * 8,
                pixels: super::convert::rgba64_to_fixedpoint(&img.pixels),
                palette: Vec::new(),
            };
            &conv
        }
        _ => img,
    };
    let bpp = fmt::bits(&img.format);
    let (comp, masks): (u32, Option<[u32; 3]>) = match img.format.as_str() {
        fmt::BGR565 => (3, Some([0xF800, 0x07E0, 0x001F])),
        fmt::IDX1 | fmt::IDX4 | fmt::IDX8 | fmt::BGR555 | fmt::BGR24 | fmt::BGR32 | fmt::BGRA32 | fmt::PBGRA32
        | fmt::RGBA64FP => (0, None),
        _ => return Err(WicError::Unsupported),
    };
    if bpp == 0 {
        return Err(WicError::Unsupported);
    }
    let clr_used = img.palette.len() as u32; // 0 for bitfields; the masks aren't a colour table
    let dpm = {
        let v = (dpi * 100.0 / 2.54) + 0.5;
        if v.is_nan() { 3780 } else { v as i32 }
    };
    let dib_extra = masks.map_or(0, |_| 12); // 3 bitfield masks stored where the palette would go
    let off_bits = 14 + 40 + dib_extra + 4 * img.palette.len();
    let dst_stride = (((bpp * img.width + 7) >> 3) + 3) & !3;
    let image_size = dst_stride as usize * img.height as usize;

    let mut out = Vec::with_capacity(off_bits + image_size);
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&((off_bits + image_size) as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(off_bits as u32).to_le_bytes());
    // BITMAPINFOHEADER.
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&img.width.to_le_bytes());
    out.extend_from_slice(&img.height.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(bpp as u16).to_le_bytes());
    out.extend_from_slice(&comp.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // biSizeImage
    out.extend_from_slice(&dpm.to_le_bytes());
    out.extend_from_slice(&dpm.to_le_bytes());
    out.extend_from_slice(&clr_used.to_le_bytes()); // biClrUsed
    out.extend_from_slice(&clr_used.to_le_bytes()); // biClrImportant
    if let Some(m) = masks {
        for mask in m {
            out.extend_from_slice(&mask.to_le_bytes());
        }
    } else {
        for &c in &img.palette {
            // 0xAARRGGBB -> B,G,R,A bytes (WIC copies the palette DWORD straight out).
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
    // Scanlines, bottom-up, each padded to the DWORD stride with zeroes.
    let src_stride = img.stride;
    for y in (0..img.height as usize).rev() {
        let row = img.pixels.get(y * src_stride..y * src_stride + src_stride).ok_or(WicError::BadImage)?;
        out.extend_from_slice(row);
        out.extend(std::iter::repeat(0u8).take(dst_stride as usize - src_stride));
    }
    Ok(out)
}

/// RLE4/RLE8 → 32bppBGR (`ConvertRLEto32RGBA` + the bottom-up flip `ReadLine_RLE` applies).
fn decode_rle(bytes: &[u8], h: &Header, format: &'static str) -> Result<Decoded, WicError> {
    let (w, ht) = (h.width as usize, h.height as usize);
    let pal = read_palette_raw(bytes, h);
    let src = bytes.get(h.off_bits..).ok_or(WicError::BadImage)?;
    let rle4 = h.comp == 2;

    // Stream-order buffer. RLE pixels not covered by the stream stay 0 (confirmed by the oracle).
    let mut buf = vec![0u8; w * ht * 4];
    let color = |idx: usize| -> [u8; 4] { pal.get(idx).copied().unwrap_or([0, 0, 0, 0]) };
    let put = |buf: &mut [u8], row: usize, x: usize, c: [u8; 4]| {
        if row < ht && x < w {
            let p = (row * w + x) * 4;
            buf[p..p + 4].copy_from_slice(&c);
        }
    };

    let (mut i, mut row, mut x) = (0usize, 0usize, 0usize);
    while row < ht && i + 1 < src.len() {
        let count = src[i] as usize;
        let val = src[i + 1] as usize;
        i += 2;
        if count != 0 {
            // Encoded run.
            let n = if x < w { (x + count).min(w) - x } else { 0 };
            for k in 0..n {
                let c = if rle4 {
                    color(if k % 2 == 0 { val >> 4 } else { val & 0xF })
                } else {
                    color(val)
                };
                put(&mut buf, row, x + k, c);
            }
            x += count;
        } else {
            match val {
                0 => {
                    row += 1;
                    x = 0;
                }
                1 => break,
                2 => {
                    let dx = *src.get(i).ok_or(WicError::BadImage)? as usize;
                    let dy = *src.get(i + 1).ok_or(WicError::BadImage)? as usize;
                    i += 2;
                    x += dx;
                    row += dy;
                }
                len => {
                    let data_bytes = if rle4 { (len + 1) / 2 } else { len };
                    if i + data_bytes > src.len() {
                        return Err(WicError::BadImage);
                    }
                    for k in 0..len {
                        let idx = if rle4 {
                            let byte = src[i + k / 2];
                            if k % 2 == 0 { byte >> 4 } else { byte & 0xF }
                        } else {
                            src[i + k]
                        } as usize;
                        put(&mut buf, row, x + k, color(idx));
                    }
                    x += len;
                    // Absolute data is padded to a 16-bit boundary.
                    i += data_bytes + (data_bytes & 1);
                }
            }
        }
    }

    // ReadLine_RLE serves rows bottom-up for a normal BMP, top-down when flagged.
    let stride = w * 4;
    let mut pixels = vec![0u8; stride * ht];
    for y in 0..ht {
        let sr = if h.top_down { y } else { ht - 1 - y };
        pixels[y * stride..(y + 1) * stride].copy_from_slice(&buf[sr * stride..(sr + 1) * stride]);
    }
    // WIC still exposes the source colour table for an RLE (32bppBGR) frame.
    Ok(Decoded {
        width: h.width,
        height: h.height,
        format: format.into(),
        stride,
        pixels,
        palette: read_palette(bytes, h),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_rgba64_fixedpoint_matches_native() {
        // Seed 100165: a 1x5 64bppRGBA source → the native BMP encoder's exact 94-byte 64bppRGBAFixedPoint
        // output (no DPI set ⇒ ppm 0). Exercises the RGBA64 → 64bppRGBAFixedPoint encode path end to end.
        let src: [[u16; 4]; 5] = [
            [14181, 28291, 17434, 40848],
            [996, 12645, 49011, 40904],
            [4172, 32270, 5164, 33862],
            [49409, 29812, 36892, 52909],
            [41313, 36770, 9250, 20976],
        ];
        let pixels: Vec<u8> = src.iter().flat_map(|p| p.iter().flat_map(|c| c.to_le_bytes())).collect();
        let img = Decoded { width: 1, height: 5, format: fmt::RGBA64.into(), stride: 8, pixels, palette: vec![] };
        let want: Vec<u8> = vec![
            66, 77, 94, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 1, 0, 64, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 144, 0, 205, 8, 94, 11,
            62, 10, 221, 8, 151, 5, 235, 16, 214, 25, 58, 0, 160, 6, 43, 0, 137, 16, 157, 16, 254, 0, 10, 0,
            249, 19, 215, 1, 255, 4, 59, 1, 242, 19,
        ];
        assert_eq!(encode_bmp(&img, 0.0).unwrap(), want);
    }

    // Regression (fuzz): width=3, height≈469M (0x1C000002) in a tiny file. The RLE path sizes a
    // `width*height*4` (~5.6GB) buffer independently of the file length; it must be rejected rather
    // than aborting the process on the allocation.
    #[test]
    fn rejects_absurd_dimensions() {
        let mut b = vec![0u8; 54];
        b[0..2].copy_from_slice(b"BM");
        b[10] = 54; // off_bits
        b[14] = 40; // bi_size (BITMAPINFOHEADER)
        b[18..22].copy_from_slice(&3u32.to_le_bytes()); // width
        b[22..26].copy_from_slice(&0x1C00_0002u32.to_le_bytes()); // height
        b[28] = 32; // bpp
        b[30] = 2; // compression = BI_RLE4 (the unguarded RLE path)
        assert!(decode_bmp(&b).is_err());
    }
}
