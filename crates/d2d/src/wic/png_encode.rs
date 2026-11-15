//! The exact PNG WIC's encoder writes. The container is fixed — signature, then
//! `IHDR`, `sRGB` (intent 0), `gAMA` (45455), the `IDAT`(s) and `IEND` — and the compressed stream
//! comes from [`super::deflate`]. The two WIC-specific pieces here are the
//! per-format compression settings and the adaptive per-row PNG filter
//! heuristic: minimum sum of |signed filtered byte|, candidates gated by
//! row position, with Paeth tried only when the best-so-far still exceeds `16·rowlen`.
//!
//! Byte-identical to native WIC (under Wine) for every non-indexed format it encodes.

use super::deflate::zlib_compress;
use super::{Decoded, WicError};

/// CRC-32 (PNG/zlib, poly 0xEDB88320).
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

fn chunk(out: &mut Vec<u8>, typ: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(typ);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// One complete PNG chunk (length + type + data + CRC) — for the metadata chunk writers.
pub fn png_chunk(typ: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    chunk(&mut out, typ, data);
    out
}

/// How a WIC source row maps to a PNG scanline (byte transform of the sample bytes).
#[derive(Clone, Copy, PartialEq)]
enum Xform {
    Copy,      // gray / sub-byte / already-RGB(A): identity
    Bswap16,   // 16-bit samples little-endian -> big-endian
    Bgr24,     // [B,G,R] -> [R,G,B]
    Bgra32,    // [B,G,R,A] -> [R,G,B,A]
    Bgr32,     // [B,G,R,X] -> [R,G,B] (drop the ignored X byte)
    Bgr565,    // 16bpp 5-6-5 -> [R,G,B] 8-bit (bit-replication expand)
    Bgr555,    // 16bpp 5-5-5 (top bit ignored) -> [R,G,B] 8-bit (bit-replication expand)
}

struct Cfg {
    colortype: u8,
    bitdepth: u8,
    bpp: usize,       // PNG output bytes per pixel (>=1), for filtering
    src_bits: usize,  // SOURCE bits per pixel (for the source row stride; differs from output when a
                      // format is widened/narrowed, e.g. BGR32 32->24, BGR565 16->24)
    xform: Xform,
    level: u32,
    strategy: u32,
    adaptive: bool,
}

/// Per-format PNG settings: ≥8-bit non-palette formats use level 4 + `Z_FILTERED`
/// + adaptive filtering; sub-byte and palette formats use filter None + `Z_DEFAULT` (level 7 sub-byte,
/// 4 for 8-bit palette).
fn config(fmt: &str) -> Result<Cfg, WicError> {
    let suffix = u8::from_str_radix(&fmt[fmt.len() - 2..], 16).map_err(|_| WicError::Unsupported)?;
    // (colortype, bitdepth, bpp_out, src_bits, xform). colortype 3 = palette (PLTE/tRNS + filter None).
    let (ct, bd, bpp, src_bits, xf) = match suffix {
        0x01 => (3, 1, 1, 1, Xform::Copy),   // Indexed1
        0x02 => (3, 2, 1, 2, Xform::Copy),   // Indexed2
        0x03 => (3, 4, 1, 4, Xform::Copy),   // Indexed4
        0x04 => (3, 8, 1, 8, Xform::Copy),   // Indexed8
        0x05 => (0, 1, 1, 1, Xform::Copy),   // BlackWhite
        0x06 => (0, 2, 1, 2, Xform::Copy),   // Gray2
        0x07 => (0, 4, 1, 4, Xform::Copy),   // Gray4
        0x08 => (0, 8, 1, 8, Xform::Copy),   // Gray8
        0x0b => (0, 16, 2, 16, Xform::Bswap16), // Gray16
        0x09 => (2, 8, 3, 16, Xform::Bgr555), // Bgr555 -> RGB8
        0x0a => (2, 8, 3, 16, Xform::Bgr565), // Bgr565 -> RGB8
        0x0c => (2, 8, 3, 24, Xform::Bgr24), // Bgr24
        0x0d => (2, 8, 3, 24, Xform::Copy),  // Rgb24 (already R,G,B)
        0x0e => (2, 8, 3, 32, Xform::Bgr32), // Bgr32 -> RGB8 (drop X)
        0x0f => (6, 8, 4, 32, Xform::Bgra32), // Bgra32
        0xe9 => (6, 8, 4, 32, Xform::Copy),  // Rgba32 (already R,G,B,A)
        0x15 => (2, 16, 6, 48, Xform::Bswap16), // Rgb48
        0x16 => (6, 16, 8, 64, Xform::Bswap16), // Rgba64
        _ => return Err(WicError::Unsupported),
    };
    let subbyte = bd < 8;
    let palette = ct == 3;
    Ok(Cfg {
        colortype: ct,
        bitdepth: bd,
        bpp,
        src_bits,
        xform: xf,
        level: if subbyte { 7 } else { 4 },
        strategy: if subbyte || palette { 0 } else { 1 },
        adaptive: !subbyte && !palette,
    })
}

/// Transform one WIC source row into PNG scanline bytes (channel reorder / endianness).
fn transform_row(src: &[u8], xf: Xform, out: &mut Vec<u8>) {
    match xf {
        Xform::Copy => out.extend_from_slice(src),
        Xform::Bswap16 => {
            for p in src.chunks_exact(2) {
                out.push(p[1]);
                out.push(p[0]);
            }
        }
        Xform::Bgr24 => {
            for p in src.chunks_exact(3) {
                out.extend_from_slice(&[p[2], p[1], p[0]]);
            }
        }
        Xform::Bgra32 => {
            for p in src.chunks_exact(4) {
                out.extend_from_slice(&[p[2], p[1], p[0], p[3]]);
            }
        }
        Xform::Bgr32 => {
            for p in src.chunks_exact(4) {
                out.extend_from_slice(&[p[2], p[1], p[0]]); // drop the ignored X byte
            }
        }
        Xform::Bgr565 => {
            for p in src.chunks_exact(2) {
                let v = u16::from_le_bytes([p[0], p[1]]);
                let (r5, g6, b5) = (((v >> 11) & 0x1f) as u8, ((v >> 5) & 0x3f) as u8, (v & 0x1f) as u8);
                out.extend_from_slice(&[(r5 << 3) | (r5 >> 2), (g6 << 2) | (g6 >> 4), (b5 << 3) | (b5 >> 2)]);
            }
        }
        Xform::Bgr555 => {
            for p in src.chunks_exact(2) {
                let v = u16::from_le_bytes([p[0], p[1]]);
                let (r5, g5, b5) = (((v >> 10) & 0x1f) as u8, ((v >> 5) & 0x1f) as u8, (v & 0x1f) as u8);
                out.extend_from_slice(&[(r5 << 3) | (r5 >> 2), (g5 << 3) | (g5 >> 2), (b5 << 3) | (b5 >> 2)]);
            }
        }
    }
}

fn abs_i8(b: u8) -> u32 {
    (b as i8).unsigned_abs() as u32
}

/// Sum of |signed byte| over `row` — the cost metric the filter heuristic minimises.
fn cost(row: &[u8]) -> u32 {
    row.iter().map(|&b| abs_i8(b)).sum()
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (a, b, c) = (a as i32, b as i32, c as i32);
    let p = a + b - c;
    let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
    (if pa <= pb && pa <= pc { a } else if pb <= pc { b } else { c }) as u8
}

/// Apply PNG filter `f` to `cur` (previous row `prev`, `None` for row 0), returning the filtered bytes.
fn apply(f: u8, cur: &[u8], prev: Option<&[u8]>, bpp: usize) -> Vec<u8> {
    let n = cur.len();
    let mut out = vec![0u8; n];
    let left = |i: usize| if i >= bpp { cur[i - bpp] } else { 0 };
    let up = |i: usize| prev.map_or(0, |p| p[i]);
    let ul = |i: usize| if i >= bpp { prev.map_or(0, |p| p[i - bpp]) } else { 0 };
    for i in 0..n {
        out[i] = match f {
            0 => cur[i],
            1 => cur[i].wrapping_sub(left(i)),
            2 => cur[i].wrapping_sub(up(i)),
            3 => cur[i].wrapping_sub(((left(i) as u32 + up(i) as u32) >> 1) as u8),
            _ => cur[i].wrapping_sub(paeth(left(i), up(i), ul(i))),
        };
    }
    out
}

/// Choose the adaptive filter for one row exactly as native's filter heuristic does (all five candidates enabled,
/// i.e. mode 0xF8): minimum cost, ties to the lower filter number, with the row-0 and Paeth gates.
fn choose(cur: &[u8], prev: Option<&[u8]>, bpp: usize) -> u8 {
    let rowlen = cur.len();
    let mut best_f = 0u8;
    let mut best = cost(cur);
    if best == 0 {
        return 0;
    }
    if rowlen > bpp {
        let c = cost(&apply(1, cur, prev, bpp));
        if c < best {
            best_f = 1;
            best = c;
            if c == 0 {
                return 1;
            }
        }
    }
    match prev {
        None => {
            // row 0: only None/Sub above, plus Average (with prev = 0).
            if rowlen > bpp {
                let c = cost(&apply(3, cur, None, bpp));
                if c < best {
                    return 3;
                }
            }
            best_f
        }
        Some(_) => {
            let c = cost(&apply(2, cur, prev, bpp));
            if c < best {
                best_f = 2;
                best = c;
                if c == 0 {
                    return 2;
                }
            }
            let c = cost(&apply(3, cur, prev, bpp));
            if c < best {
                best_f = 3;
                best = c;
                if c == 0 {
                    return 3;
                }
            }
            // Paeth is tried only when the best so far still exceeds 16·rowlen, and it carries a
            // 4·(rowlen-bpp) penalty (WIC biases against the costlier-to-decode Paeth filter).
            if rowlen > bpp && best > 16 * rowlen as u32 {
                let c = cost(&apply(4, cur, prev, bpp)) + 4 * (rowlen - bpp) as u32;
                if c < best {
                    return 4;
                }
            }
            best_f
        }
    }
}

/// Encode with no added metadata.
pub fn encode_png(img: &Decoded) -> Result<Vec<u8>, WicError> {
    encode_png_meta(img, None)
}

/// Encode `img` to a PNG byte-for-byte as WIC does. Non-indexed formats only.
/// `xmp`, when set, is an XMP packet embedded as an `iTXt` chunk (`XML:com.adobe.xmp`) right after
/// `gAMA`, as native's PNG encoder does when serialising XMP into an iTXt chunk.
pub fn encode_png_meta(img: &Decoded, xmp: Option<&[u8]>) -> Result<Vec<u8>, WicError> {
    let cfg = config(&img.format)?;
    if img.width == 0 || img.height == 0 {
        return Err(WicError::BadImage);
    }
    let src_rowbytes = ((img.width as usize * cfg.src_bits + 7) / 8).max(1);
    if img.pixels.len() < img.stride * img.height as usize {
        return Err(WicError::BadImage);
    }

    // Build the filtered scanline stream (filter byte + transformed, filtered row data).
    let png_rowbytes = ((img.width as usize * cfg.colortype_channels() * cfg.bitdepth as usize) + 7) / 8;
    let mut filtered: Vec<u8> = Vec::with_capacity((png_rowbytes + 1) * img.height as usize);
    let mut prev: Option<Vec<u8>> = None;
    for y in 0..img.height as usize {
        let so = y * img.stride;
        let mut row = Vec::with_capacity(png_rowbytes);
        transform_row(&img.pixels[so..so + src_rowbytes], cfg.xform, &mut row);
        let f = if cfg.adaptive { choose(&row, prev.as_deref(), cfg.bpp) } else { 0 };
        filtered.push(f);
        filtered.extend_from_slice(&apply(f, &row, prev.as_deref(), cfg.bpp));
        prev = Some(row);
    }

    // windowBits = clamp(ceil_log2(idat_raw_size + 256), 8, 15), as native computes it.
    let data_size = filtered.len() as u32;
    let padded_size = data_size + 256;
    let ceil_log2 = 32 - (padded_size - 1).leading_zeros();
    let window_bits = ceil_log2.clamp(8, 15);
    let zlib = zlib_compress(&filtered, cfg.level, window_bits, cfg.strategy);

    let mut out = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&img.width.to_be_bytes());
    ihdr.extend_from_slice(&img.height.to_be_bytes());
    ihdr.extend_from_slice(&[cfg.bitdepth, cfg.colortype, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"sRGB", &[0]);
    chunk(&mut out, b"gAMA", &45455u32.to_be_bytes());
    if let Some(packet) = xmp {
        // iTXt: keyword "XML:com.adobe.xmp", compression 0, empty language/translated keyword, then
        // the uncompressed XMP packet as the text.
        let mut data = b"XML:com.adobe.xmp\0\0\0\0\0".to_vec();
        data.extend_from_slice(packet);
        chunk(&mut out, b"iTXt", &data);
    }
    if cfg.colortype == 3 {
        // PLTE: each palette entry's R,G,B (0xAARRGGBB -> R,G,B, alpha dropped).
        let mut plte = Vec::with_capacity(img.palette.len() * 3);
        for &c in &img.palette {
            plte.extend_from_slice(&[(c >> 16) as u8, (c >> 8) as u8, c as u8]);
        }
        chunk(&mut out, b"PLTE", &plte);
        // tRNS: palette alphas, truncated after the last non-opaque entry (omitted if all opaque).
        let last = img.palette.iter().rposition(|&c| (c >> 24) as u8 != 0xff);
        if let Some(last) = last {
            let trns: Vec<u8> = img.palette[..=last].iter().map(|&c| (c >> 24) as u8).collect();
            chunk(&mut out, b"tRNS", &trns);
        }
    }
    // IDAT chunks fill WIC's 65536-byte output buffer: each non-final chunk's 8-byte header + data
    // ends exactly on a buffer boundary, so its data length is `65528 - (output_pos % 65536)`.
    let mut off = 0;
    while off < zlib.len() {
        let avail = 65528 - out.len() % 65536;
        let take = (zlib.len() - off).min(avail);
        chunk(&mut out, b"IDAT", &zlib[off..off + take]);
        off += take;
    }
    chunk(&mut out, b"IEND", &[]);
    Ok(out)
}

impl Cfg {
    fn bpp_bits(&self) -> usize {
        self.colortype_channels() * self.bitdepth as usize
    }
    fn colortype_channels(&self) -> usize {
        match self.colortype {
            0 => 1,
            2 => 3,
            6 => 4,
            _ => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_source_formats_accepted_with_native_colortype() {
        // BGR32/RGB24 → RGB (ct 2), RGBA32 → RGBA (ct 6), BGR565 → RGB8 (ct 2), each with the right
        // source stride. Suffixes are the GUID's last byte (RGBA32 = ...261ae9 → 0xe9).
        let c = |g: &str| config(g).unwrap();
        let bgr32 = c("6fddc324-4e03-4bfe-b185-3d77768dc90e");
        assert_eq!((bgr32.colortype, bgr32.src_bits, bgr32.bpp), (2, 32, 3));
        let rgb24 = c("6fddc324-4e03-4bfe-b185-3d77768dc90d");
        assert_eq!((rgb24.colortype, rgb24.src_bits), (2, 24));
        let rgba32 = c("f5c7ad2d-6a8d-43dd-a7a8-a29935261ae9");
        assert_eq!((rgba32.colortype, rgba32.src_bits, rgba32.bpp), (6, 32, 4));
        let bgr565 = c("6fddc324-4e03-4bfe-b185-3d77768dc90a");
        assert_eq!((bgr565.colortype, bgr565.src_bits), (2, 16));
        let bgr555 = c("6fddc324-4e03-4bfe-b185-3d77768dc909");
        assert_eq!((bgr555.colortype, bgr555.src_bits, bgr555.bpp), (2, 16, 3));
    }

    #[test]
    fn transforms_match_native() {
        // BGR32 [B,G,R,X] → [R,G,B]; BGR565 0xAAD1 → (173,89,140) (native-r3 seed 100326 px0).
        let mut o = Vec::new();
        transform_row(&[139, 108, 210, 41], Xform::Bgr32, &mut o);
        assert_eq!(o, [210, 108, 139]);
        o.clear();
        transform_row(&[0xD1, 0xAA], Xform::Bgr565, &mut o);
        assert_eq!(o, [173, 89, 140]);
        o.clear();
        // BGR555 0xA377 → R=8 G=27 B=23 → (66,222,189) (native-r5 seed 500055 px0).
        transform_row(&[0x77, 0xA3], Xform::Bgr555, &mut o);
        assert_eq!(o, [66, 222, 189]);
    }
}
