//! ICO/CUR → WIC frames, reproducing WIC's ICO and CUR decoders.
//!
//! Each directory entry is one frame. A PNG entry is decoded by the PNG codec and keeps the PNG's
//! own native format (`HrPreparePngIcon`). A DIB entry is a headerless BITMAPINFOHEADER whose
//! `biHeight` is twice the real height (XOR colour bitmap over a 1bpp AND mask); WIC rebuilds a BMP
//! from it (`HrPrepareIcon`/`DecodeDib`), decodes the colour to 32bppBGRA, then either keeps the
//! source's own alpha (32bpp icons whose alpha channel is non-zero — `HrCheckAlphaIcon`) or applies
//! the AND mask (masked texels → 0x00000000, the rest get alpha forced to 0xFF).
//! Every DIB frame is therefore 32bppBGRA, top-down.
//!
//! Colour depths 16 and 64 in a DIB entry return `Unsupported` (16bpp icons are vanishingly rare
//! and would need WIC's 565→8888 expansion).

use super::bmp::{decode_bmp, fmt, Decoded};
use super::{decode_png, WicError};

fn u16(b: &[u8], o: usize) -> Option<u32> {
    Some(u16::from_le_bytes([*b.get(o)?, *b.get(o + 1)?]) as u32)
}
fn u32(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes([*b.get(o)?, *b.get(o + 1)?, *b.get(o + 2)?, *b.get(o + 3)?]))
}

const PNG_SIG: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

/// `GetPaletteEntries`: colour-table size for a DIB icon entry.
fn palette_entries(bpp: u32, clr_used: u32, comp: u32) -> u32 {
    if comp == 3 {
        return if bpp == 16 || bpp == 32 { 3 } else { 0 };
    }
    if matches!(bpp, 1 | 4 | 8) {
        let max = 1u32 << bpp;
        return if clr_used == 0 || clr_used > max { max } else { clr_used };
    }
    0
}

fn dword_stride(bits: u32) -> usize {
    ((((bits + 7) >> 3) + 3) & !3) as usize
}

/// One directory entry's sort key fields (as `MatchImage` reads them).
struct Entry {
    off: usize,
    end: usize, // data end, clamped to the stream length (a truncated entry keeps what is present)
    png: bool,
    width: i64,
    height: i64, // DIB: the doubled biHeight; PNG: image height
    bpp: i64,
}

pub fn decode_ico(bytes: &[u8]) -> Result<Vec<Decoded>, WicError> {
    // ICONDIR: reserved(2)=0, type(2)=1 ICO / 2 CUR, count(2). A stream too short for the 6-byte
    // directory is STREAMREAD; any readable-but-invalid directory/entry is BADIMAGE (confirmed via
    // Wine — WIC returns BADIMAGE, not STREAMREAD, even for truncated entries).
    let count = u16(bytes, 4).ok_or(WicError::StreamRead)? as usize;
    if count == 0 {
        return Err(WicError::BadImage);
    }
    let mut ents = Vec::with_capacity(count);
    for i in 0..count {
        let e = 6 + i * 16; // ICONDIRENTRY
        let off = u32(bytes, e + 12).ok_or(WicError::BadImage)? as usize;
        let size = u32(bytes, e + 8).ok_or(WicError::BadImage)? as usize;
        // A truncated entry declares more data than the stream holds; WIC keeps what is present
        // (a PNG icon whose IDAT runs off the end still decodes, filling the missing rows with zeros).
        let end = off.saturating_add(size).min(bytes.len());
        let d = bytes.get(off..end).ok_or(WicError::BadImage)?;
        let png = d.len() >= 8 && d[..8] == PNG_SIG;
        // The entry data must be long enough for the header the branch reads: a PNG frame's IHDR
        // (width/height/colour at d[16..26]) or a BMP frame's BITMAPINFOHEADER (d[4..16]). A crafted
        // ICO can declare an entry size of 0 or a few bytes; reject it rather than index out of bounds.
        if d.len() < if png { 26 } else { 16 } {
            return Err(WicError::BadImage);
        }
        let (width, height, bpp) = if png {
            let w = u32::from_be_bytes([d[16], d[17], d[18], d[19]]) as i64;
            let h = u32::from_be_bytes([d[20], d[21], d[22], d[23]]) as i64;
            let chans = match d[25] {
                2 => 3,
                4 => 2,
                6 => 4,
                _ => 1,
            };
            (w, h, d[24] as i64 * chans)
        } else {
            (
                i32::from_le_bytes([d[4], d[5], d[6], d[7]]) as i64,
                i32::from_le_bytes([d[8], d[9], d[10], d[11]]) as i64,
                u16(d, 14).ok_or(WicError::BadImage)? as i64,
            )
        };
        ents.push(Entry { off, end, png, width, height, bpp });
    }

    // WIC presents frames sorted by MatchImage distance to the largest/deepest entry.
    let (mw, mh, mb) = ents.iter().fold((0, 0, 0), |(w, h, b), e| {
        (w.max(e.width), h.max(e.height), b.max(e.bpp))
    });
    let term = |v: i64, t: i64| -> i64 {
        let d = v - t;
        if d < 0 { 2 * -d } else { d }
    };
    let score = |e: &Entry| term(e.width, mw) + term(e.height, mh >> 1) + 2 * term(e.bpp, mb);
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by_key(|&i| score(&ents[i]));

    let mut frames = Vec::with_capacity(count);
    for i in order {
        let e = &ents[i];
        let d = &bytes[e.off..e.end];
        frames.push(if e.png {
            let p = decode_png(d)?;
            Decoded {
                width: p.width,
                height: p.height,
                format: p.format.guid(),
                stride: p.stride,
                pixels: p.pixels,
                palette: p.palette,
            }
        } else {
            decode_dib(d)?
        });
    }
    Ok(frames)
}

/// Decode one DIB icon entry (BITMAPINFOHEADER + palette + XOR colour + AND mask) → 32bppBGRA.
fn decode_dib(d: &[u8]) -> Result<Decoded, WicError> {
    let bi_size = u32(d, 0).ok_or(WicError::BadImage)?;
    if bi_size < 40 {
        return Err(WicError::Unsupported);
    }
    let width = u32(d, 4).ok_or(WicError::BadImage)?;
    let dbl_height = i32::from_le_bytes([d[8], d[9], d[10], d[11]]);
    let bpp = u16(d, 14).ok_or(WicError::BadImage)?;
    let comp = u32(d, 16).ok_or(WicError::BadImage)?;
    let clr_used = u32(d, 32).ok_or(WicError::BadImage)?;
    if !matches!(bpp, 1 | 4 | 8 | 24 | 32) {
        return Err(WicError::Unsupported);
    }
    let (w, h) = (width as usize, (dbl_height / 2) as usize);
    let pal = palette_entries(bpp, clr_used, comp) as usize;
    let color_stride = dword_stride(bpp * width);
    let color_off = bi_size as usize + pal * 4;
    let color_size = color_stride * h;
    let mask_off = color_off + color_size;
    let mask_stride = dword_stride(width);

    // Rebuild a BMP (file header + DIB with the real height) and decode the colour bitmap.
    let mut bmp = Vec::with_capacity(14 + color_off + color_size);
    bmp.extend_from_slice(b"BM");
    bmp.extend_from_slice(&0u32.to_le_bytes());
    bmp.extend_from_slice(&0u32.to_le_bytes());
    bmp.extend_from_slice(&((14 + color_off) as u32).to_le_bytes());
    let dib_end = bi_size as usize + pal * 4;
    bmp.extend_from_slice(d.get(0..dib_end).ok_or(WicError::BadImage)?);
    bmp[14 + 8..14 + 12].copy_from_slice(&(h as i32).to_le_bytes()); // real height
    bmp.extend_from_slice(d.get(color_off..color_off + color_size).ok_or(WicError::BadImage)?);
    let color = decode_bmp(&bmp)?;

    // Unpack the native colour frame to straight BGRA (4th byte = source alpha candidate).
    let mut px = unpack_bgra(&color, w, h);

    let has_alpha = bpp == 32 && px.chunks_exact(4).any(|p| p[3] != 0);
    if !has_alpha {
        let mask = d.get(mask_off..mask_off + mask_stride * h);
        for y in 0..h {
            for x in 0..w {
                let p = (y * w + x) * 4;
                let bit = mask
                    .map(|m| (m[(h - 1 - y) * mask_stride + x / 8] >> (7 - (x & 7))) & 1)
                    .unwrap_or(0);
                if bit != 0 {
                    px[p..p + 4].copy_from_slice(&[0, 0, 0, 0]);
                } else {
                    px[p + 3] = 0xFF;
                }
            }
        }
    }
    Ok(Decoded {
        width,
        height: h as u32,
        format: fmt::BGRA32.into(),
        stride: w * 4,
        pixels: px,
        palette: Vec::new(),
    })
}

/// Native colour frame → straight BGRA, top-down (the 4th byte carries the source alpha for 32bpp).
fn unpack_bgra(c: &Decoded, w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 4];
    for y in 0..h {
        let row = &c.pixels[y * c.stride..];
        for x in 0..w {
            let o = (y * w + x) * 4;
            let (b, g, r, a) = match c.format.as_str() {
                fmt::BGR24 => (row[x * 3], row[x * 3 + 1], row[x * 3 + 2], 0),
                fmt::BGR32 | fmt::BGRA32 => (row[x * 4], row[x * 4 + 1], row[x * 4 + 2], row[x * 4 + 3]),
                fmt::IDX1 | fmt::IDX4 | fmt::IDX8 => {
                    let idx = index_at(row, x, &c.format);
                    let col = c.palette.get(idx).copied().unwrap_or(0);
                    ((col & 0xFF) as u8, (col >> 8 & 0xFF) as u8, (col >> 16 & 0xFF) as u8, 0)
                }
                _ => (0, 0, 0, 0),
            };
            out[o..o + 4].copy_from_slice(&[b, g, r, a]);
        }
    }
    out
}

fn index_at(row: &[u8], x: usize, format: &str) -> usize {
    match format {
        fmt::IDX8 => row[x] as usize,
        fmt::IDX4 => {
            let byte = row[x / 2];
            (if x & 1 == 0 { byte >> 4 } else { byte & 0xF }) as usize
        }
        fmt::IDX1 => ((row[x / 8] >> (7 - (x & 7))) & 1) as usize,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Regression (fuzz): a directory entry whose declared data is too short for the BMP/PNG header
    // the decoder reads must be rejected, not indexed out of bounds.
    #[test]
    fn rejects_short_entry() {
        let mut b = vec![0u8; 26];
        b[2] = 1; // type = ICO
        b[4] = 1; // count = 1
        b[14..18].copy_from_slice(&4u32.to_le_bytes()); // entry data size = 4 (< 16-byte BMP header)
        b[18..22].copy_from_slice(&22u32.to_le_bytes()); // entry data offset = 22
        assert!(decode_ico(&b).is_err());
    }

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    // native-r5 seeds 500002 / 500162: a PNG icon whose IDAT is declared longer than the stream holds.
    // WIC keeps the complete scanlines it could decode and zero-fills the rest; the frame is byte-exact.
    #[test]
    fn truncated_png_icon_fills_tail_with_zeros() {
        let ico = unhex("00000100010008060000000020000a0100001600000089504e470d0a1a0a0000000d4948445200000008000000060806000000fe05dffb000000d149444154789c01c60039ff0178941dffdd19450019970a004a319f0053d25a00b041e000a627a7000cb2610002ddeb10001816de00f2ae6d00ca38fb0003898e009f2a48007b944b00e002f2000015a316ff9c1f19ff1d7d61ffa736e8ffb59821ff6143b3ff1cfc");
        let native = unhex("1d9478ff62ad55ff6c446eff0b75b8ff65470bff4588bbffecaf61ff4d616dff2d7f55ff40c36dffd9f260ff06ad82fff3d00eff8db25aff3743dcff3f634dff0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");
        let f = &decode_ico(&ico).unwrap()[0];
        assert_eq!((f.width, f.height, f.pixels.len()), (8, 6, 192));
        assert_eq!(f.pixels, native);

        let ico = unhex("0000010001000506000000002000b40000001600000089504e470d0a1a0a0000000d49484452000000050000000608060000000bfb544b0000007b49444154789c05c1b10d85201405d0cb2fa82d2d0c610b07c0380d0dbd4b503b8125219674ac426967f21221b9ff1c755d17cff3c43ccfd05a639a26a86559b86d1bf67d8788e0791ea8d61a4504ebbac21803ef3d7ebd77e49ce19c434a09b5568024dff765088131467edf47945238c6e07ddf");
        let native = unhex("a9a9a9ff444444ff5c5c5cff626262ff727272ff1c1c1cff606060ffa6a6a6ff9d9d9dff8a8a8affe5e5e5ffdcdcdcff131313ff313131ff9d9d9dffe0e0e0ff8f8f8fff565656ffe2e2e2ff616161fffffffffff4f4f4ff717171ff919191fffafafaff0000000000000000000000000000000000000000");
        let f = &decode_ico(&ico).unwrap()[0];
        assert_eq!((f.width, f.height, f.pixels.len()), (5, 6, 120));
        assert_eq!(f.pixels, native);
    }
}
