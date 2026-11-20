//! TIFF → WIC frames, reproducing WIC's TIFF decoder (bundled libtiff).
//!
//! Native format follows `ReadPixelFormat`: photometric + bit depth + sample count select
//! BlackWhite / 4bpp/8bpp/16bppGray, 1/4/8bppIndexed, 24bppBGR / 48bppRGB, 32bppBGRA / 64bppRGBA.
//! Byte layout follows the `HrProcess*` line processors: RGB(8/8/8) and RGBA swap R/B
//! (`HrProcess24Bpp`/`HrProcess32Bpp`), MinIsWhite (photometric 0) inverts every byte
//! (`HrProcessCopy`), everything else copies. Each IFD is a frame.
//!
//! Scope: strips (not tiles), PlanarConfiguration 1, compression none/PackBits/LZW/Deflate with
//! horizontal predictor; photometric MinIsWhite/MinIsBlack/RGB/Palette. CMYK, YCbCr, float, tiled,
//! planar and separate alpha/gray+alpha return `Unsupported`.

use super::bmp::{fmt, Decoded};
use super::convert::{convert, Format};
use super::WicError;
use std::collections::HashMap;

fn u16(b: &[u8], o: usize, le: bool) -> Option<u32> {
    let (a, c) = (*b.get(o)?, *b.get(o + 1)?);
    Some(if le { u16::from_le_bytes([a, c]) } else { u16::from_be_bytes([a, c]) } as u32)
}
fn u32(b: &[u8], o: usize, le: bool) -> Option<u32> {
    let v = [*b.get(o)?, *b.get(o + 1)?, *b.get(o + 2)?, *b.get(o + 3)?];
    Some(if le { u32::from_le_bytes(v) } else { u32::from_be_bytes(v) })
}

/// A parsed IFD entry: field type, count, and the raw 4-byte value/offset.
struct Entry {
    typ: u16,
    count: u32,
    val: [u8; 4],
}
type Ifd = HashMap<u16, Entry>;

fn parse_ifd(b: &[u8], off: usize, le: bool) -> Result<(Ifd, usize), WicError> {
    // Reading the IFD at an offset the header points past EOF is a STREAMREAD (the header itself is
    // already validated as a real TIFF by the caller; confirmed via Wine).
    let n = u16(b, off, le).ok_or(WicError::StreamRead)? as usize;
    let mut ifd = HashMap::new();
    for i in 0..n {
        let e = off + 2 + i * 12;
        let tag = u16(b, e, le).ok_or(WicError::StreamRead)? as u16;
        let typ = u16(b, e + 2, le).ok_or(WicError::StreamRead)? as u16;
        let count = u32(b, e + 4, le).ok_or(WicError::StreamRead)?;
        let v = b.get(e + 8..e + 12).ok_or(WicError::StreamRead)?;
        ifd.insert(tag, Entry { typ, count, val: [v[0], v[1], v[2], v[3]] });
    }
    let next = u32(b, off + 2 + n * 12, le).ok_or(WicError::StreamRead)? as usize;
    Ok((ifd, next))
}

fn type_size(typ: u16) -> usize {
    match typ {
        1 | 2 | 6 | 7 => 1, // BYTE/ASCII/SBYTE/UNDEFINED
        3 | 8 => 2,         // SHORT/SSHORT
        _ => 4,             // LONG/SLONG/... (treated as 4)
    }
}

/// All values of a tag as u32 (SHORT/LONG/BYTE), inline or via offset.
fn values(b: &[u8], le: bool, ifd: &Ifd, tag: u16) -> Option<Vec<u32>> {
    let e = ifd.get(&tag)?;
    let sz = type_size(e.typ);
    let total = sz * e.count as usize;
    let src = if total <= 4 {
        &e.val[..total]
    } else {
        let off = u32(&e.val, 0, le).unwrap() as usize;
        b.get(off..off + total)?
    };
    Some(
        (0..e.count as usize)
            .map(|i| match sz {
                1 => src[i] as u32,
                2 => u16(src, i * 2, le).unwrap(),
                _ => u32(src, i * 4, le).unwrap(),
            })
            .collect(),
    )
}

fn scalar(b: &[u8], le: bool, ifd: &Ifd, tag: u16, default: u32) -> u32 {
    values(b, le, ifd, tag).and_then(|v| v.first().copied()).unwrap_or(default)
}

pub fn decode_tiff(bytes: &[u8]) -> Result<Vec<Decoded>, WicError> {
    let le = match bytes.get(0..2) {
        Some(b"II") => true,
        Some(b"MM") => false,
        _ => return Err(WicError::BadImage),
    };
    if u16(bytes, 2, le) != Some(42) {
        return Err(WicError::BadImage);
    }
    let mut off = u32(bytes, 4, le).ok_or(WicError::BadImage)? as usize;
    let mut frames = Vec::new();
    // A crafted TIFF can chain the IFDs into a cycle (a next-IFD pointer back to an earlier IFD),
    // which would loop forever and decode a frame each pass (CPU spin + unbounded memory). Stop as
    // soon as an IFD offset repeats; a revisited offset also bounds the total to the distinct offsets.
    let mut seen = std::collections::HashSet::new();
    while off != 0 && seen.insert(off) {
        let (ifd, next) = parse_ifd(bytes, off, le)?;
        frames.push(decode_ifd(bytes, le, &ifd)?);
        off = next;
    }
    if frames.is_empty() {
        return Err(WicError::BadImage);
    }
    Ok(frames)
}

fn decode_ifd(b: &[u8], le: bool, ifd: &Ifd) -> Result<Decoded, WicError> {
    if ifd.contains_key(&322) {
        return Err(WicError::Unsupported); // tiled
    }
    let width = scalar(b, le, ifd, 256, 0);
    let height = scalar(b, le, ifd, 257, 0);
    // A crafted TIFF can declare a dimension near u32::MAX; `width * bps * samples` (row_bytes, below)
    // and `width * bits` (stride, in `transform`) are u32 arithmetic and then wrap to a tiny stride
    // inconsistent with the `0..width` pixel loops, indexing out of bounds. Cap each dimension well
    // beyond any real image (16M px) so the arithmetic stays within u32; native rejects such dims too.
    if width > (1 << 24) || height > (1 << 24) {
        return Err(WicError::BadImage);
    }
    let bps = scalar(b, le, ifd, 258, 1);
    let comp = scalar(b, le, ifd, 259, 1);
    let photo = scalar(b, le, ifd, 262, 1);
    let samples = scalar(b, le, ifd, 277, 1);
    let predictor = scalar(b, le, ifd, 317, 1);
    if scalar(b, le, ifd, 284, 1) != 1 {
        return Err(WicError::Unsupported); // planar
    }

    let format = pixel_format(photo, bps, samples)?;

    // Decode every strip into one packed, top-down scanline buffer.
    let row_bytes = ((width * bps * samples + 7) / 8) as usize;
    // Reject absurd declared dimensions before any allocation (the strip buffer and the `transform`
    // output are both `row_bytes * height`): a crafted TIFF can claim dimensions whose decoded size is
    // many GB from a tiny file, aborting the process. No real codec expands by more than ~2^20×, so a
    // declared decoded size beyond that of the file is malformed (native returns BADIMAGE).
    match (row_bytes as u64).checked_mul(height as u64) {
        Some(n) if n <= (b.len() as u64).saturating_mul(1 << 20) => {}
        _ => return Err(WicError::BadImage),
    }
    let rows_per_strip = scalar(b, le, ifd, 278, height).max(1);
    let offs = values(b, le, ifd, 273).ok_or(WicError::BadImage)?;
    let counts = values(b, le, ifd, 279).ok_or(WicError::BadImage)?;
    // Cap the pre-allocation hint: a crafted TIFF can declare dimensions whose product is many GB
    // (an upfront alloc that aborts the process). The buffer still grows to the real decoded size as
    // strips append — each strip's `want` is validated below — so a legitimate large image is unaffected.
    let mut raw = Vec::with_capacity((row_bytes.saturating_mul(height as usize)).min(1 << 24));
    let mut y = 0u32;
    for (o, c) in offs.iter().zip(&counts) {
        let strip_rows = rows_per_strip.min(height - y) as usize;
        let want = row_bytes * strip_rows;
        let src = b.get(*o as usize..*o as usize + *c as usize).ok_or(WicError::BadImage)?;
        let mut dec = decompress(comp, src, want)?;
        dec.truncate(want);
        if dec.len() < want {
            return Err(WicError::BadImage);
        }
        if predictor == 2 {
            undo_predictor(&mut dec, strip_rows, row_bytes, width as usize, samples as usize, bps);
        }
        raw.extend_from_slice(&dec);
        y += strip_rows as u32;
        if y >= height {
            break;
        }
    }

    let pixels = transform(&raw, format, photo, width as usize, height as usize, row_bytes);
    let palette = if matches!(format, fmt::IDX1 | fmt::IDX4 | fmt::IDX8) {
        read_colormap(b, le, ifd, bps)?
    } else {
        Vec::new()
    };
    let stride = ((width * fmt::bits(format) + 7) / 8) as usize;
    Ok(Decoded { width, height, format: format.into(), stride, pixels, palette })
}

/// `ReadPixelFormat` for the supported photometric/depth/sample combinations.
fn pixel_format(photo: u32, bps: u32, samples: u32) -> Result<&'static str, WicError> {
    use fmt::*;
    Ok(match photo {
        0 | 1 => match (bps, samples) {
            (1, 1) => BLACKWHITE,
            (2..=4, 1) => GRAY4,
            (5..=8, 1) => GRAY8,
            (9..=16, 1) => GRAY16,
            _ => return Err(WicError::Unsupported), // gray+alpha (samples 2) out of scope
        },
        2 => match (bps, samples) {
            (8, 3) => BGR24,
            (8, 4) => BGRA32,
            (16, 3) => RGB48,
            (16, 4) => RGBA64,
            _ => return Err(WicError::Unsupported),
        },
        3 => match bps {
            1 => IDX1,
            2..=4 => IDX4,
            5..=8 => IDX8,
            _ => return Err(WicError::Unsupported),
        },
        _ => return Err(WicError::Unsupported),
    })
}

/// Pack the raw scanlines into the WIC native byte layout.
fn transform(raw: &[u8], format: &str, photo: u32, w: usize, h: usize, row_bytes: usize) -> Vec<u8> {
    let stride = ((w as u32 * fmt::bits(format) + 7) / 8) as usize;
    let mut out = vec![0u8; stride * h];
    for y in 0..h {
        // A crafted TIFF can declare more rows than its strip data provides; `raw` is then shorter
        // than `h * row_bytes`. Guard both the slice start (out-of-range would panic) and the per-row
        // length: a short/missing row stays zero-filled rather than indexing out of bounds.
        let src = raw.get(y * row_bytes..).unwrap_or(&[]);
        let dst = &mut out[y * stride..y * stride + stride];
        if src.len() < stride {
            continue;
        }
        match format {
            fmt::BGR24 => {
                for x in 0..w {
                    dst[x * 3] = src[x * 3 + 2];
                    dst[x * 3 + 1] = src[x * 3 + 1];
                    dst[x * 3 + 2] = src[x * 3];
                }
            }
            fmt::BGRA32 => {
                for x in 0..w {
                    dst[x * 4] = src[x * 4 + 2];
                    dst[x * 4 + 1] = src[x * 4 + 1];
                    dst[x * 4 + 2] = src[x * 4];
                    dst[x * 4 + 3] = src[x * 4 + 3];
                }
            }
            _ => {
                dst.copy_from_slice(&src[..stride]);
                if photo == 0 {
                    for byte in dst.iter_mut() {
                        *byte = !*byte;
                    }
                }
            }
        }
    }
    out
}

fn read_colormap(b: &[u8], le: bool, ifd: &Ifd, bps: u32) -> Result<Vec<u32>, WicError> {
    let cm = values(b, le, ifd, 320).ok_or(WicError::BadImage)?;
    let n = 1usize << bps;
    if cm.len() < 3 * n {
        return Err(WicError::BadImage);
    }
    // ColorMap is 3 consecutive 16-bit arrays (R, G, B); WIC scales each to 8-bit (>>8).
    Ok((0..n)
        .map(|i| {
            let r = cm[i] >> 8;
            let g = cm[n + i] >> 8;
            let bl = cm[2 * n + i] >> 8;
            0xFF00_0000 | (r << 16) | (g << 8) | bl
        })
        .collect())
}

/// Horizontal differencing predictor (TIFF predictor 2) for 8- and 16-bit samples.
fn undo_predictor(d: &mut [u8], rows: usize, row_bytes: usize, w: usize, samples: usize, bps: u32) {
    // Cap the column count to what fits in `row_bytes`: a crafted TIFF can declare a width/samples/bps
    // whose per-row span (w·samples·⌈bps/8⌉) exceeds row_bytes, which would index past the row slice.
    // A consistent image has w·samples·bytes == row_bytes, so this is inert for valid input.
    let bytes_per_sample = (bps as usize).div_ceil(8).max(1);
    let w = w.min(row_bytes / samples.max(1) / bytes_per_sample);
    for r in 0..rows {
        let row = &mut d[r * row_bytes..r * row_bytes + row_bytes];
        if bps == 8 {
            for x in 1..w {
                for c in 0..samples {
                    row[x * samples + c] = row[x * samples + c].wrapping_add(row[(x - 1) * samples + c]);
                }
            }
        } else if bps == 16 {
            for x in 1..w {
                for c in 0..samples {
                    let i = (x * samples + c) * 2;
                    let p = ((x - 1) * samples + c) * 2;
                    let v = u16::from_le_bytes([row[i], row[i + 1]])
                        .wrapping_add(u16::from_le_bytes([row[p], row[p + 1]]));
                    row[i..i + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
}

fn decompress(comp: u32, src: &[u8], want: usize) -> Result<Vec<u8>, WicError> {
    match comp {
        1 => Ok(src.to_vec()),
        5 => tiff_lzw(src, want),
        8 | 32946 => {
            use std::io::Read;
            let mut out = Vec::with_capacity(want.min(1 << 24)); // cap the hint; a huge declared `want` must not pre-alloc GB (output grows to the stream's real size, bounded by `src`)
            flate2::read::ZlibDecoder::new(src)
                .read_to_end(&mut out)
                .map_err(|_| WicError::BadImage)?;
            Ok(out)
        }
        32773 => packbits(src, want),
        _ => Err(WicError::Unsupported),
    }
}

/// PackBits RLE.
fn packbits(src: &[u8], want: usize) -> Result<Vec<u8>, WicError> {
    let mut out = Vec::with_capacity(want.min(1 << 24)); // cap the hint; a huge declared `want` must not pre-alloc GB (output grows to the stream's real size, bounded by `src`)
    let mut i = 0;
    while i < src.len() && out.len() < want {
        let n = src[i] as i8;
        i += 1;
        if n >= 0 {
            let len = n as usize + 1;
            out.extend_from_slice(src.get(i..i + len).ok_or(WicError::BadImage)?);
            i += len;
        } else if n != -128 {
            let len = (1 - n as i32) as usize;
            let b = *src.get(i).ok_or(WicError::BadImage)?;
            i += 1;
            out.extend(std::iter::repeat(b).take(len));
        }
    }
    Ok(out)
}

/// TIFF LZW (MSB-first codes, early code-size change, clear 256 / EOI 257).
fn tiff_lzw(data: &[u8], want: usize) -> Result<Vec<u8>, WicError> {
    const CLEAR: u32 = 256;
    const EOI: u32 = 257;
    let mut dict: Vec<Vec<u8>> = Vec::new();
    let reset = |dict: &mut Vec<Vec<u8>>| {
        dict.clear();
        for i in 0..256u32 {
            dict.push(vec![i as u8]);
        }
        dict.push(Vec::new());
        dict.push(Vec::new());
    };
    reset(&mut dict);
    let mut code_size = 9u8;
    let mut out = Vec::with_capacity(want.min(1 << 24)); // cap the hint; a huge declared `want` must not pre-alloc GB (output grows to the stream's real size, bounded by `src`)
    let (mut acc, mut nbits, mut pos) = (0u32, 0u8, 0usize);
    let mut prev: Option<usize> = None;
    loop {
        while nbits < code_size {
            match data.get(pos) {
                Some(&byte) => {
                    acc = (acc << 8) | byte as u32;
                    nbits += 8;
                    pos += 1;
                }
                None => return Ok(out),
            }
        }
        let code = (acc >> (nbits - code_size)) & ((1 << code_size) - 1);
        nbits -= code_size;
        if code == CLEAR {
            reset(&mut dict);
            code_size = 9;
            prev = None;
            continue;
        }
        if code == EOI {
            return Ok(out);
        }
        let entry = if (code as usize) < dict.len() {
            dict[code as usize].clone()
        } else if code as usize == dict.len() {
            let p = dict[prev.ok_or(WicError::BadImage)?].clone();
            let mut e = p.clone();
            e.push(p[0]);
            e
        } else {
            return Err(WicError::BadImage);
        };
        out.extend_from_slice(&entry);
        if let Some(p) = prev {
            let mut new = dict[p].clone();
            new.push(entry[0]);
            dict.push(new);
            // Early change: grow one code before the width would overflow.
            if dict.len() + 1 == (1 << code_size) && code_size < 12 {
                code_size += 1;
            }
        }
        prev = Some(code as usize);
    }
}

/// One IFD field to write: a typed value (`data` = little-endian value bytes, any length — the
/// serialiser inlines it when ≤4 bytes, else places it out of line) or an Exif/GPS sub-IFD pointer
/// with its child fields. Used to merge `GetMetadataQueryWriter` tags into the encoder's directory.
#[derive(Clone)]
pub enum TiffField {
    Value { tag: u16, ty: u16, count: u32, data: Vec<u8> },
    Sub { tag: u16, children: Vec<TiffField> },
}

impl TiffField {
    fn v(tag: u16, ty: u16, count: u32, data: Vec<u8>) -> Self {
        TiffField::Value { tag, ty, count, data }
    }
    fn tag(&self) -> u16 {
        match self {
            TiffField::Value { tag, .. } | TiffField::Sub { tag, .. } => *tag,
        }
    }
}

/// The structural tags libtiff owns; the encoder drops any of these
/// set through the metadata query writer (native's sorted critical-tag table).
const CRITICAL_TAGS: [u16; 35] = [
    254, 255, 256, 257, 258, 259, 262, 263, 264, 265, 266, 273, 277, 278, 279, 282, 283, 284, 288,
    289, 290, 291, 292, 293, 296, 317, 320, 322, 323, 324, 325, 338, 513, 514, 33006,
];

/// Serialise an IFD (the metadata IFD writer's layout): entries sorted ascending by
/// tag, then the out-of-line data area (values >4 bytes, each even-padded, in tag order), then nested
/// sub-IFD blocks (each padded to a multiple of 4). Every offset is relative to the TIFF start, with
/// this IFD's table beginning at `base`.
fn emit_tiff_ifd(fields: &[TiffField], base: usize) -> Vec<u8> {
    let mut fs: Vec<&TiffField> = fields.iter().collect();
    fs.sort_by_key(|f| f.tag());
    let n = fs.len();
    let data_start = base + 2 + 12 * n + 4;
    let mut data = Vec::new();
    let value_off: Vec<Option<usize>> = fs
        .iter()
        .map(|f| match f {
            TiffField::Value { data: d, .. } if d.len() > 4 => {
                let off = data_start + data.len();
                data.extend_from_slice(d);
                if d.len() % 2 == 1 {
                    data.push(0);
                }
                Some(off)
            }
            _ => None,
        })
        .collect();
    let mut subs = Vec::new();
    let mut cursor = data_start + data.len();
    let sub_off: Vec<Option<usize>> = fs
        .iter()
        .map(|f| match f {
            TiffField::Sub { children, .. } => {
                let off = cursor;
                let mut block = emit_tiff_ifd(children, off);
                while block.len() % 4 != 0 {
                    block.push(0);
                }
                cursor += block.len();
                subs.extend_from_slice(&block);
                Some(off)
            }
            _ => None,
        })
        .collect();
    let mut out = Vec::new();
    out.extend_from_slice(&(n as u16).to_le_bytes());
    for (i, f) in fs.iter().enumerate() {
        match f {
            TiffField::Value { tag, ty, count, data: d } => {
                out.extend_from_slice(&tag.to_le_bytes());
                out.extend_from_slice(&ty.to_le_bytes());
                out.extend_from_slice(&count.to_le_bytes());
                match value_off[i] {
                    Some(off) => out.extend_from_slice(&(off as u32).to_le_bytes()),
                    None => {
                        let mut f4 = d.clone();
                        f4.resize(4, 0);
                        out.extend_from_slice(&f4);
                    }
                }
            }
            TiffField::Sub { tag, .. } => {
                out.extend_from_slice(&tag.to_le_bytes());
                out.extend_from_slice(&4u16.to_le_bytes()); // LONG pointer
                out.extend_from_slice(&1u32.to_le_bytes());
                out.extend_from_slice(&(sub_off[i].unwrap() as u32).to_le_bytes());
            }
        }
    }
    out.extend_from_slice(&0u32.to_le_bytes()); // next-IFD offset (none)
    out.extend_from_slice(&data);
    out.extend_from_slice(&subs);
    out
}

/// Encode with no added metadata (the plain core encoder).
pub fn encode_tiff(img: &Decoded) -> Result<Vec<u8>, WicError> {
    encode_tiff_meta(img, &[])
}

/// Encode a `Decoded` exactly as WIC's TIFF encoder does (bundled libtiff),
/// confirmed byte-identical against the real codec. Little-endian
/// (`II*`), the strip data at offset 8, then the IFD; LZW compression (tag 5) with a horizontal
/// predictor (tag 2); 96 dpi (`96000/1000`, ResolutionUnit inch). Rows are split into strips of
/// `min(height, 12288 / bytes_per_line)` rows, each strip predictor-differenced per row and LZW-
/// compressed independently; RGB/RGBA swap R↔B and RGBA adds `ExtraSamples = 2`, palette formats
/// emit a 256-entry `ColorMap` (`v*257`). Out-of-line tag values follow the IFD in ascending tag
/// order. Supports the formats WIC decodes to: 8bppGray, 24bppBGR, 32bppBGRA and 8bppIndexed.
/// `extra` holds metadata-query-writer tags merged into IFD0 (native's merge with offset fixups);
/// structural tags libtiff owns ([`CRITICAL_TAGS`]) are dropped.
pub fn encode_tiff_meta(img: &Decoded, extra: &[TiffField]) -> Result<Vec<u8>, WicError> {
    // Native's TIFF `WriteSource` converts packed/BGRX 8-bpc colour sources to 24bppBGR (its RGB
    // path) first; 16-bpc sources (16bppGray, 64bppRGBA) keep their depth.
    let conv;
    let img = match img.format.as_str() {
        fmt::BGR32 | fmt::BGR555 | fmt::BGR565 => {
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
        _ => img,
    };
    // (samples-per-pixel, photometric, bits-per-sample, swap-R↔B). TIFF stores RGB(A); BGR(A) sources
    // swap R↔B, but 64bppRGBA is already R,G,B,A so it must NOT swap.
    let (spp, photo, bits, swap_rb): (usize, u16, u16, bool) = match img.format.as_str() {
        fmt::GRAY8 => (1, 1, 8, false),
        fmt::GRAY16 => (1, 1, 16, false),
        fmt::BGR24 => (3, 2, 8, true),
        fmt::BGRA32 => (4, 2, 8, true),
        fmt::RGBA64 => (4, 2, 16, false),
        fmt::IDX8 => (1, 3, 8, false),
        _ => return Err(WicError::Unsupported),
    };
    let bs = (bits / 8) as usize; // bytes per sample
    let (w, h) = (img.width as usize, img.height as usize);
    if w == 0 || h == 0 {
        return Err(WicError::Unsupported);
    }
    let bpr = w * spp * bs;
    if img.stride < bpr || img.pixels.len() < img.stride * h {
        return Err(WicError::BadImage);
    }

    // Tight sample buffer; RGB/RGBA swap R↔B (TIFF stores RGB(A), the source is BGR(A)) — per
    // `bs`-byte sample, so a 16-bpc channel swaps its two bytes as a unit.
    let mut s = vec![0u8; bpr * h];
    for y in 0..h {
        let (src, dst) = (&img.pixels[y * img.stride..], &mut s[y * bpr..y * bpr + bpr]);
        dst.copy_from_slice(&src[..bpr]);
        if swap_rb {
            for px in dst.chunks_mut(spp * bs) {
                for k in 0..bs {
                    px.swap(k, 2 * bs + k);
                }
            }
        }
    }

    let rps = (12288 / bpr).clamp(1, h);
    let nstrips = h.div_ceil(rps);
    let mut strips: Vec<Vec<u8>> = Vec::with_capacity(nstrips);
    for st in 0..nstrips {
        let (r0, r1) = (st * rps, ((st + 1) * rps).min(h));
        let mut seg = s[r0 * bpr..r1 * bpr].to_vec();
        // Horizontal predictor (tag 2): each sample minus its left neighbour, per row. Native enables
        // it only for 8-bpc samples; 16-bpc TIFF uses Predictor NONE (verified: 16bppGray tag 317 = 1).
        if bs == 1 {
            for row in seg.chunks_mut(bpr) {
                for x in (1..w).rev() {
                    for ch in 0..spp {
                        let (i, j) = (x * spp + ch, (x - 1) * spp + ch);
                        row[i] = row[i].wrapping_sub(row[j]);
                    }
                }
            }
        }
        strips.push(tiff_lzw_encode(&seg));
    }

    // Strip data starts at offset 8; the IFD is word-aligned after it.
    let body_len: usize = strips.iter().map(Vec::len).sum();
    let mut strip_offs = Vec::with_capacity(nstrips);
    let mut o = 8;
    for stp in &strips {
        strip_offs.push(o as u32);
        o += stp.len();
    }
    let ifd_off = (8 + body_len + 1) & !1;

    // Baseline IFD fields (`data` = little-endian value bytes). `sh`/`lo` build SHORT/LONG values.
    let sh = |v: u16| v.to_le_bytes().to_vec();
    let lo = |v: u32| v.to_le_bytes().to_vec();
    let res: Vec<u8> = [96000u32.to_le_bytes(), 1000u32.to_le_bytes()].concat();
    let mut fields: Vec<TiffField> = vec![
        TiffField::v(254, 4, 1, lo(0)),
        TiffField::v(256, 4, 1, lo(w as u32)),
        TiffField::v(257, 4, 1, lo(h as u32)),
        if spp == 1 { TiffField::v(258, 3, 1, sh(bits)) } else { TiffField::v(258, 3, spp as u32, (0..spp).flat_map(|_| bits.to_le_bytes()).collect()) },
        TiffField::v(259, 3, 1, sh(5)), // LZW
        TiffField::v(262, 3, 1, sh(photo)),
        if nstrips == 1 { TiffField::v(273, 4, 1, lo(strip_offs[0])) } else { TiffField::v(273, 4, nstrips as u32, strip_offs.iter().flat_map(|v| v.to_le_bytes()).collect()) },
        TiffField::v(277, 3, 1, sh(spp as u16)),
        TiffField::v(278, 4, 1, lo(rps as u32)),
        if nstrips == 1 { TiffField::v(279, 4, 1, lo(strips[0].len() as u32)) } else { TiffField::v(279, 4, nstrips as u32, strips.iter().flat_map(|s| (s.len() as u32).to_le_bytes()).collect()) },
        TiffField::v(282, 5, 1, res.clone()),
        TiffField::v(283, 5, 1, res),
        TiffField::v(296, 3, 1, sh(2)), // ResolutionUnit = inch
        TiffField::v(317, 3, 1, sh(if bits == 8 { 2 } else { 1 })), // Predictor: horizontal (8-bpc) / none (16-bpc)
    ];
    if spp > 1 {
        fields.push(TiffField::v(284, 3, 1, sh(1))); // PlanarConfiguration = chunky
    }
    if photo == 3 {
        let mut cm = vec![0u8; 768 * 2];
        for (i, &c) in img.palette.iter().take(256).enumerate() {
            for (ch, chan) in [16u32, 8, 0].into_iter().enumerate() {
                let v = (((c >> chan) & 0xff) as u16) * 257;
                cm[(ch * 256 + i) * 2..(ch * 256 + i) * 2 + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        fields.push(TiffField::v(320, 3, 768, cm));
    }
    if spp == 4 {
        fields.push(TiffField::v(338, 3, 1, sh(2))); // ExtraSamples = unassociated alpha
    }
    // Merge the caller's metadata tags, dropping the structural tags libtiff owns.
    fields.extend(extra.iter().filter(|f| !CRITICAL_TAGS.contains(&f.tag())).cloned());

    let mut out = Vec::new();
    out.extend_from_slice(b"II");
    out.extend_from_slice(&42u16.to_le_bytes());
    out.extend_from_slice(&(ifd_off as u32).to_le_bytes());
    for stp in &strips {
        out.extend_from_slice(stp);
    }
    out.resize(ifd_off, 0); // word-align pad before the IFD
    out.extend_from_slice(&emit_tiff_ifd(&fields, ifd_off));
    Ok(out)
}

/// TIFF LZW (libtiff `LZWEncode`): MSB-first codes, clear 256 / end 257, 9–12 bit width growing when
/// the next free code reaches `2^width`, the table cleared when it reaches 4094.
fn tiff_lzw_encode(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let (mut acc, mut nout) = (0u32, 0u8);
    let mut put = |code: u32, width: u8, out: &mut Vec<u8>| {
        acc = (acc << width) | code;
        nout += width;
        while nout >= 8 {
            nout -= 8;
            out.push((acc >> nout) as u8);
        }
    };
    let mut nbits = 9u8;
    let mut table: std::collections::HashMap<(u32, u8), u32> = std::collections::HashMap::new();
    let mut free = 258u32;
    put(256, nbits, &mut out);
    let mut ent = data[0] as u32;
    for &c in &data[1..] {
        if let Some(&code) = table.get(&(ent, c)) {
            ent = code;
        } else {
            put(ent, nbits, &mut out);
            table.insert((ent, c), free);
            free += 1;
            if free == (1 << nbits) {
                nbits += 1;
            }
            if free == 4094 {
                put(256, nbits, &mut out);
                table.clear();
                free = 258;
                nbits = 9;
            }
            ent = c as u32;
        }
    }
    put(ent, nbits, &mut out);
    put(257, nbits, &mut out);
    if nout > 0 {
        out.push((acc << (8 - nout)) as u8);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Regression (fuzz): ImageWidth near u32::MAX. `width*bps*samples` (row_bytes) and `width*bits`
    // (stride) are u32 arithmetic and wrap to a tiny stride inconsistent with the `0..width` pixel
    // loops, indexing out of bounds. Must be rejected before that arithmetic runs.
    #[test]
    fn rejects_absurd_width() {
        fn entry(tag: u16, val: u32) -> Vec<u8> {
            let mut e = tag.to_le_bytes().to_vec();
            e.extend_from_slice(&4u16.to_le_bytes()); // type LONG
            e.extend_from_slice(&1u32.to_le_bytes()); // count
            e.extend_from_slice(&val.to_le_bytes());
            e
        }
        let mut b = b"II".to_vec();
        b.extend_from_slice(&0x2Au16.to_le_bytes());
        b.extend_from_slice(&8u32.to_le_bytes()); // IFD at offset 8
        b.extend_from_slice(&2u16.to_le_bytes()); // 2 entries (tags ascending)
        b.extend_from_slice(&entry(256, 2_281_701_379)); // ImageWidth
        b.extend_from_slice(&entry(257, 2)); // ImageLength
        b.extend_from_slice(&0u32.to_le_bytes()); // next IFD = none
        assert!(decode_tiff(&b).is_err());
    }

    // Regression (fuzz): a next-IFD pointer that cycles back to an earlier IFD would loop forever,
    // decoding a frame each pass (CPU spin + unbounded memory). Decoding must terminate — the cycle is
    // cut after the first valid frame.
    #[test]
    fn cyclic_ifd_terminates() {
        fn entry(tag: u16, val: u32) -> Vec<u8> {
            let mut e = tag.to_le_bytes().to_vec();
            e.extend_from_slice(&4u16.to_le_bytes()); // type LONG
            e.extend_from_slice(&1u32.to_le_bytes()); // count 1
            e.extend_from_slice(&val.to_le_bytes());
            e
        }
        let mut b = b"II".to_vec();
        b.extend_from_slice(&0x2Au16.to_le_bytes());
        b.extend_from_slice(&8u32.to_le_bytes()); // first IFD at offset 8
        b.extend_from_slice(&8u16.to_le_bytes()); // 8 entries (tags ascending)
        for (tag, val) in [(256, 1), (257, 1), (258, 8), (259, 1), (262, 1), (273, 110), (277, 1), (279, 1)] {
            b.extend_from_slice(&entry(tag, val)); // 1×1 gray8, uncompressed, strip at offset 110
        }
        b.extend_from_slice(&8u32.to_le_bytes()); // next-IFD = 8 → cycle back onto itself
        b.push(0); // one byte of strip pixel data at offset 110
        let frames = decode_tiff(&b).expect("decodes the one valid 1×1 frame");
        assert_eq!(frames.len(), 1); // terminated: the cycle was cut, not followed forever
    }
}
