//! GIF → WIC frames, reproducing WIC's GIF decoder.
//!
//! Each image descriptor is one 8bppIndexed frame at its own sub-rectangle size, holding the
//! LZW-decoded, de-interlaced colour indices (`DecodeCurrentFrame`/`GetLine`). The frame palette
//! (`CopyFramePalette`/`HrInitPalette`) is the local colour table if the image has one, else the
//! global table, as `0xFFRRGGBB`; the active graphic-control transparent index is stored as
//! `0x00RRGGBB` (alpha 0), and the table is extended with zeroes up to that index when it lies
//! beyond the table.

use super::bmp::{fmt, Decoded};
use super::convert::{convert, Format};
use super::quantize::median_cut;
use super::WicError;

fn u16(b: &[u8], o: usize) -> Option<usize> {
    Some(u16::from_le_bytes([*b.get(o)?, *b.get(o + 1)?]) as usize)
}

/// Read length-prefixed GIF sub-blocks starting at `*pos`, concatenating their data. A stream that
/// ends partway through the sub-blocks (a truncated frame) is read leniently, exactly as WIC does:
/// the bytes present are kept and the run of blocks ends at the stream end instead of failing.
fn read_subblocks(b: &[u8], pos: &mut usize) -> Result<Vec<u8>, WicError> {
    let mut out = Vec::new();
    loop {
        let len = match b.get(*pos) {
            Some(&l) => l as usize,
            None => return Ok(out), // truncated before the next block header
        };
        *pos += 1;
        if len == 0 {
            return Ok(out);
        }
        let end = (*pos + len).min(b.len());
        out.extend_from_slice(&b[*pos..end]);
        *pos = end; // if the block ran off the end, the next read hits EOF and returns
    }
}

pub fn decode_gif(bytes: &[u8]) -> Result<Vec<Decoded>, WicError> {
    // WIC reads the 13-byte header (signature + logical screen descriptor): a shorter stream is
    // STREAMREAD, a readable but wrong signature is BADHEADER (both confirmed via Wine).
    if bytes.len() < 13 {
        return Err(WicError::StreamRead);
    }
    if &bytes[0..6] != b"GIF87a" && &bytes[0..6] != b"GIF89a" {
        return Err(WicError::BadHeader);
    }
    // Logical Screen Descriptor.
    let packed = bytes[10];
    let mut pos = 13;
    let global_pal = if packed & 0x80 != 0 {
        let n = 2usize << (packed & 7);
        let p = bytes.get(13..13 + 3 * n).ok_or(WicError::BadImage)?.to_vec();
        pos = 13 + 3 * n;
        Some(p)
    } else {
        None
    };

    let mut frames = Vec::new();
    let mut transparent: Option<u8> = None; // from the most recent Graphic Control Extension
    loop {
        // A stream that ends without a trailer (a truncated animation) stops here with the frames
        // decoded so far, as WIC does — the later frames are still counted and partially decoded.
        let marker = match bytes.get(pos) {
            Some(&m) => m,
            None => break,
        };
        match marker {
            0x3B => break, // trailer
            0x21 => {
                // Extension. 0xF9 = Graphic Control (transparent index); others skipped.
                let label = *bytes.get(pos + 1).ok_or(WicError::BadImage)?;
                pos += 2;
                if label == 0xF9 {
                    let size = *bytes.get(pos).ok_or(WicError::BadImage)? as usize;
                    let gce = bytes.get(pos + 1..pos + 1 + size).ok_or(WicError::BadImage)?;
                    transparent = (gce[0] & 1 != 0).then(|| gce[3]);
                    pos += 1 + size;
                    let _ = read_subblocks(bytes, &mut pos)?; // block terminator
                } else {
                    let _ = read_subblocks(bytes, &mut pos)?;
                }
            }
            0x2C => {
                // Image Descriptor.
                let w = u16(bytes, pos + 5).ok_or(WicError::BadImage)?;
                let h = u16(bytes, pos + 7).ok_or(WicError::BadImage)?;
                let ipacked = *bytes.get(pos + 9).ok_or(WicError::BadImage)?;
                pos += 10;
                let local_pal = if ipacked & 0x80 != 0 {
                    let n = 2usize << (ipacked & 7);
                    let p = bytes.get(pos..pos + 3 * n).ok_or(WicError::BadImage)?.to_vec();
                    pos += 3 * n;
                    Some(p)
                } else {
                    None
                };
                let min_code = *bytes.get(pos).ok_or(WicError::BadImage)?;
                pos += 1;
                let lzw = read_subblocks(bytes, &mut pos)?;
                // Bound the declared image against the file size (as the tiff/bmp/ico decoders do):
                // a tiny stream cannot describe a huge image, so this caps the pixel buffer and keeps
                // `w * h` from wrapping `usize` on a 32-bit target.
                let pixel_count = w.checked_mul(h).ok_or(WicError::BadImage)?;
                if pixel_count > bytes.len().saturating_mul(1 << 20) {
                    return Err(WicError::BadImage);
                }
                let mut pixels = lzw_decode(min_code, &lzw, pixel_count)?;
                if ipacked & 0x40 != 0 {
                    pixels = deinterlace(&pixels, w, h);
                }
                let table = local_pal.as_ref().or(global_pal.as_ref()).ok_or(WicError::BadImage)?;
                frames.push(Decoded {
                    width: w as u32,
                    height: h as u32,
                    format: fmt::IDX8.into(),
                    stride: w,
                    pixels,
                    palette: build_palette(table, transparent),
                });
                transparent = None; // a GCE governs only the next image
            }
            _ => return Err(WicError::BadImage),
        }
    }
    Ok(frames)
}

/// Colour table → `0xFFRRGGBB` (transparent index → `0x00RRGGBB`), extended past the table if
/// the transparent index lies beyond it.
fn build_palette(table: &[u8], transparent: Option<u8>) -> Vec<u32> {
    let mut n = table.len() / 3;
    if let Some(t) = transparent {
        n = n.max(t as usize + 1);
    }
    (0..n)
        .map(|i| {
            let (r, g, b) = (
                table.get(i * 3).copied().unwrap_or(0) as u32,
                table.get(i * 3 + 1).copied().unwrap_or(0) as u32,
                table.get(i * 3 + 2).copied().unwrap_or(0) as u32,
            );
            let a = if transparent == Some(i as u8) { 0 } else { 0xFF };
            (a << 24) | (r << 16) | (g << 8) | b
        })
        .collect()
}

/// Reorder interlaced rows (passes 0/8, 4/8, 2/4, 1/2) into sequential order.
fn deinterlace(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    let mut k = 0;
    for &(start, step) in &[(0usize, 8usize), (4, 8), (2, 4), (1, 2)] {
        let mut y = start;
        while y < h {
            if let (Some(s), Some(d)) = (src.get(k * w..k * w + w), out.get_mut(y * w..y * w + w)) {
                d.copy_from_slice(s);
            }
            k += 1;
            y += step;
        }
    }
    out
}

/// Metadata blocks set through a GIF frame encoder's query writer, written between the logical
/// screen descriptor and the image descriptor. `comment` is the comment-extension text;
/// `(app_id, app_data)` the application extension (identifier then verbatim sub-block data); `gce`
/// the graphic-control extension; `left`/`top`/`width`/`height` override the image descriptor.
#[derive(Default)]
pub struct GifMeta {
    pub comment: Option<Vec<u8>>,
    pub gce: Option<GifGce>,
    pub app_id: Option<Vec<u8>>,
    pub app_data: Vec<u8>,
    pub left: Option<u16>,
    pub top: Option<u16>,
    pub width: Option<u16>,
    pub height: Option<u16>,
}

/// A GIF graphic-control extension: `packed` = `disposal<<2 | userInput<<1 | transparency`.
pub struct GifGce {
    pub packed: u8,
    pub delay: u16,
    pub trans_index: u8,
}

/// Encode with no added metadata (the plain core encoder).
pub fn encode_gif(img: &Decoded) -> Result<Vec<u8>, WicError> {
    encode_gif_meta(img, &GifMeta::default())
}

/// Down-convert a non-indexed source to `8bppIndexed`, exactly as native's GIF `WriteSource`
/// does when no palette is set: it runs an
/// `IWICFormatConverter` to `8bppIndexed` with a `WICBitmapPaletteTypeMedianCut` palette and
/// `WICBitmapDitherTypeErrorDiffusion`, then copies the computed palette back. Ported byte-exact
/// (validated against native-r3's 22 GIF scenes):
///   * palette = `median_cut` over the 24bpp-BGR pixels (same path as `InitializeFromBitmap`);
///   * the nearest-entry table: a 4096-cell inverse map — cell
///     `(R>>4,G>>4,B>>4)` keyed `(Rc<<8)|(Gc<<4)|Bc`, each holding the nearest palette colour for the cell
///     representative `(Rc*17,Gc*17,Bc*17)` (min RGB squared distance, first index on a tie);
///   * the error-diffusion pass: serpentine Floyd-Steinberg — odd rows L→R, even R→L;
///     the working row holds `16*source + diffused error` (`eff = acc>>4`, clamped 0..=255), the
///     chosen palette colour's error spreads 7/16 ahead (same row) + 1/16 ahead, 5/16 down, 3/16
///     behind (next row).
///
/// ponytail: a source WITH alpha (none in the GIF corpus — all opaque BGR/Gray) would also get a
/// 50%-threshold transparent palette entry; not reproduced here. Add it when such a source needs it.
pub(crate) fn error_diffuse_to_indexed8(img: &Decoded) -> Result<Decoded, WicError> {
    let (w, h) = (img.width as usize, img.height as usize);
    let sf = Format::from_guid(&img.format).ok_or(WicError::Unsupported)?;
    let bgr = convert(sf, Format::Bgr24, &img.pixels, img.width, img.height, &img.palette)?;
    let palette = median_cut(&bgr, 256);
    if palette.is_empty() {
        return Err(WicError::Unsupported);
    }
    let rgb: Vec<(i32, i32, i32)> = palette
        .iter()
        .map(|&c| (((c >> 16) & 0xff) as i32, ((c >> 8) & 0xff) as i32, (c & 0xff) as i32))
        .collect();
    let nearest = |tr: i32, tg: i32, tb: i32| -> u8 {
        let (mut best, mut best_d) = (0u8, i32::MAX);
        for (i, &(r, g, b)) in rgb.iter().enumerate() {
            let d = (r - tr).pow(2) + (g - tg).pow(2) + (b - tb).pow(2);
            if d < best_d {
                (best, best_d) = (i as u8, d);
                if d == 0 {
                    break;
                }
            }
        }
        best
    };
    let mut table = [0u8; 4096];
    for rc in 0..16usize {
        for gc in 0..16usize {
            for bc in 0..16usize {
                table[(rc << 8) | (gc << 4) | bc] = nearest((rc * 17) as i32, (gc * 17) as i32, (bc * 17) as i32);
            }
        }
    }
    // 3 signed accumulators per pixel (R,G,B) with a 2-pixel front pad so the behind/ahead spread at
    // the row edges lands in padding, never out of bounds.
    let row = 3 * (w + 6);
    let base = 6usize;
    let clamp = |v: i32| -> i32 { (v >> 4).clamp(0, 255) };
    let mut next = vec![0i32; row];
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        let mut work = next.clone();
        next.iter_mut().for_each(|e| *e = 0);
        for x in 0..w {
            let p = &bgr[(y * w + x) * 3..];
            let o = base + 3 * x;
            work[o] += 16 * p[2] as i32; // R
            work[o + 1] += 16 * p[1] as i32; // G
            work[o + 2] += 16 * p[0] as i32; // B
        }
        let ltor = y & 1 == 1;
        for i in 0..w {
            let x = if ltor { i } else { w - 1 - i };
            let o = base + 3 * x;
            let (r, g, b) = (clamp(work[o]), clamp(work[o + 1]), clamp(work[o + 2]));
            let idx = table[(((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4)) as usize];
            let (pr, pg, pb) = rgb[idx as usize];
            let err = [r - pr, g - pg, b - pb];
            out[y * w + x] = idx;
            let ahead = (base as isize + 3 * (x as isize + if ltor { 1 } else { -1 })) as usize;
            let behind = (base as isize + 3 * (x as isize - if ltor { 1 } else { -1 })) as usize;
            for (off, &e) in err.iter().enumerate() {
                work[ahead + off] += 7 * e;
                next[behind + off] += 3 * e;
                next[o + off] += 5 * e;
                next[ahead + off] += e;
            }
        }
    }
    Ok(Decoded { width: img.width, height: img.height, format: fmt::IDX8.into(), stride: w, pixels: out, palette })
}

/// Encode an 8bppIndexed `Decoded` exactly as WIC's GIF encoder does
/// (confirmed byte-identical against the real codec):
/// `GIF89a`, a logical screen descriptor with packed byte `0x70` (no global colour table, colour
/// resolution 7), then a single image at `(0,0)` with a **local** colour table rounded up to a power
/// of two (≥4 entries, zero-padded past the palette) and `min_code = log2(table)`. Pixels are GIF
/// LZW (standard variable-width, LSB-first). WIC compresses the image in batches of
/// `12000 / bytes_per_line` scanlines — a 12000-byte input buffer fed a line
/// at a time, each batch compressed and emitted as 255-byte
/// sub-blocks plus a partial; the LZW bit/dictionary state carries across batches. `meta` holds any
/// graphic-control/application/comment extensions and image-descriptor overrides set on the frame
/// query writer, written in native's order (GCE, application, comment) before the image descriptor.
pub fn encode_gif_meta(img: &Decoded, meta: &GifMeta) -> Result<Vec<u8>, WicError> {
    // A non-indexed source is down-converted first, exactly as native's GIF `WriteSource` does.
    let downconv;
    let img = if img.format == fmt::IDX8 {
        img
    } else {
        downconv = error_diffuse_to_indexed8(img)?;
        &downconv
    };
    let ncol = img.palette.len();
    let (w, h) = (img.width as usize, img.height as usize);
    if ncol == 0 || ncol > 256 || w == 0 || h == 0 {
        return Err(WicError::Unsupported);
    }
    if img.pixels.len() < img.stride * h || img.stride < w {
        return Err(WicError::BadImage);
    }
    // Local-colour-table size code: smallest k≥1 with 2^(k+1) ≥ ncol. min_code = k+1.
    let mut code = 1u32;
    while (2 << code) < ncol as u32 {
        code += 1;
    }
    let (lct_n, min_code) = (2usize << code, (code + 1) as u8);

    // Tight width×height index buffer (WIC's encoder works on `w` significant bytes per line).
    let mut px = Vec::with_capacity(w * h);
    for y in 0..h {
        px.extend_from_slice(&img.pixels[y * img.stride..y * img.stride + w]);
    }

    // LZW, batched as WIC does: `12000 / w` lines per CompressData call.
    let batch_px = (12000 / w).max(1) * w;
    let lzw = lzw_encode(min_code, &px, batch_px);

    let mut out = Vec::with_capacity(13 + 10 + 3 * lct_n + lzw.len() + 16);
    out.extend_from_slice(b"GIF89a");
    out.extend_from_slice(&(img.width as u16).to_le_bytes());
    out.extend_from_slice(&(img.height as u16).to_le_bytes());
    out.extend_from_slice(&[0x70, 0, 0]); // LSD packed (no GCT), background, aspect ratio
    // Metadata extensions in native's fixed order: graphic control, application, comment.
    if let Some(g) = &meta.gce {
        out.extend_from_slice(&[0x21, 0xF9, 0x04, g.packed, g.delay as u8, (g.delay >> 8) as u8, g.trans_index, 0]);
    }
    if let (Some(app), false) = (&meta.app_id, meta.app_data.is_empty()) {
        out.extend_from_slice(&[0x21, 0xFF, app.len() as u8]);
        out.extend_from_slice(app);
        // ponytail: app_data is written verbatim assuming it is a valid sub-block stream (native
        // drops the block if it isn't); validate/reframe here if a caller passes raw data.
        out.extend_from_slice(&meta.app_data);
    }
    if let Some(c) = &meta.comment {
        out.extend_from_slice(&[0x21, 0xFE]);
        for chunk in c.chunks(255) {
            out.push(chunk.len() as u8);
            out.extend_from_slice(chunk);
        }
        out.push(0); // block terminator
    }
    out.push(0x2C); // image descriptor
    out.extend_from_slice(&meta.left.unwrap_or(0).to_le_bytes());
    out.extend_from_slice(&meta.top.unwrap_or(0).to_le_bytes());
    out.extend_from_slice(&meta.width.unwrap_or(img.width as u16).to_le_bytes());
    out.extend_from_slice(&meta.height.unwrap_or(img.height as u16).to_le_bytes());
    out.push(0x80 | code as u8); // local colour table present, size code
    for i in 0..lct_n {
        let c = img.palette.get(i).copied().unwrap_or(0);
        out.extend_from_slice(&[(c >> 16) as u8, (c >> 8) as u8, c as u8]);
    }
    out.push(min_code);
    out.extend_from_slice(&lzw); // already framed into sub-blocks (incl. the 0 terminator)
    out.push(0x3B); // trailer
    Ok(out)
}

/// GIF LZW encode with WIC's batched sub-block framing. `batch_px` pixels are compressed per
/// `CompressData` call; each call's complete output bytes are written as 255-byte sub-blocks plus a
/// partial. The returned bytes include every sub-block length prefix and the final `0` terminator.
fn lzw_encode(min_code: u8, px: &[u8], batch_px: usize) -> Vec<u8> {
    let clear = 1u32 << min_code;
    let end = clear + 1;
    let (mut bitbuf, mut bits) = (0u32, 0u8);
    let mut raw: Vec<u8> = Vec::new(); // continuous LZW byte stream
    let mut emit = |code: u32, width: u8, raw: &mut Vec<u8>| {
        bitbuf |= code << bits;
        bits += width;
        while bits >= 8 {
            raw.push(bitbuf as u8);
            bitbuf >>= 8;
            bits -= 8;
        }
    };
    let mut table: std::collections::HashMap<(u32, u8), u32> = std::collections::HashMap::new();
    let mut width = min_code + 1;
    let mut maxc = 1u32 << width;
    let mut running = end + 1;
    emit(clear, width, &mut raw);
    let mut cur = px[0] as u32;
    // Byte count at the end of each full batch (complete bytes emitted so far).
    let mut batch_bytes = Vec::new();
    for i in 1..px.len() {
        let p = px[i];
        if let Some(&c) = table.get(&(cur, p)) {
            cur = c;
        } else {
            emit(cur, width, &mut raw);
            if running == 4096 {
                emit(clear, width, &mut raw);
                table.clear();
                running = end + 1;
                width = min_code + 1;
                maxc = 1 << width;
            } else {
                table.insert((cur, p), running);
                running += 1;
                if running > maxc && width < 12 {
                    width += 1;
                    maxc = 1 << width;
                }
            }
            cur = p as u32;
        }
        if (i + 1) % batch_px == 0 && i + 1 < px.len() {
            batch_bytes.push(raw.len());
        }
    }
    emit(cur, width, &mut raw);
    emit(end, width, &mut raw);
    if bits > 0 {
        raw.push(bitbuf as u8);
    }
    batch_bytes.push(raw.len());

    // Frame each batch's bytes into 255-byte sub-blocks (a partial block per batch).
    let mut out = Vec::with_capacity(raw.len() + raw.len() / 255 + 8);
    let mut prev = 0;
    for &end_byte in &batch_bytes {
        for chunk in raw[prev..end_byte].chunks(255) {
            out.push(chunk.len() as u8);
            out.extend_from_slice(chunk);
        }
        prev = end_byte;
    }
    out.push(0); // block terminator
    out
}

/// GIF LZW (variable-width, LSB-first codes, 12-bit max).
fn lzw_decode(min_code: u8, data: &[u8], pixel_count: usize) -> Result<Vec<u8>, WicError> {
    // The LZW minimum code size is at most 11 (GIF codes are 12 bits wide). A larger value would
    // overflow the `1 << min_code` and `min_code + 1` arithmetic below and size an enormous
    // dictionary, so reject it instead of wrapping (silently) or panicking (with overflow checks).
    if min_code > 11 {
        return Err(WicError::BadImage);
    }
    let clear = 1u32 << min_code;
    let end = clear + 1;
    let mut dict: Vec<Vec<u8>> = Vec::new();
    let reset = |dict: &mut Vec<Vec<u8>>| {
        dict.clear();
        for i in 0..clear {
            dict.push(vec![i as u8]);
        }
        dict.push(Vec::new()); // clear
        dict.push(Vec::new()); // end
    };
    reset(&mut dict);
    let mut code_size = min_code + 1;
    let mut out = Vec::with_capacity(pixel_count.min(1 << 24));
    let (mut bitbuf, mut bits, mut pos) = (0u32, 0u8, 0usize);
    let mut prev: Option<usize> = None;
    loop {
        while bits < code_size {
            let byte = match data.get(pos) {
                Some(&b) => b,
                None => {
                    out.resize(pixel_count, 0); // truncated: fill the undecoded tail with zeros
                    return Ok(out);
                }
            };
            bitbuf |= (byte as u32) << bits;
            bits += 8;
            pos += 1;
        }
        let code = (bitbuf & ((1 << code_size) - 1)) as u32;
        bitbuf >>= code_size;
        bits -= code_size;

        if code == clear {
            reset(&mut dict);
            code_size = min_code + 1;
            prev = None;
            continue;
        }
        if code == end {
            out.resize(pixel_count, 0); // an early end code leaves the rest of the frame zero
            return Ok(out);
        }
        let entry: Vec<u8> = if (code as usize) < dict.len() {
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
        // The image is exactly `pixel_count` pixels; stop once it is full so a crafted LZW stream
        // with long dictionary entries cannot grow the output without bound.
        if out.len() >= pixel_count {
            out.truncate(pixel_count);
            return Ok(out);
        }
        if let Some(p) = prev {
            if dict.len() < 4096 {
                let mut new = dict[p].clone();
                new.push(entry[0]);
                dict.push(new);
                if dict.len() == (1 << code_size) && code_size < 12 {
                    code_size += 1;
                }
            }
        }
        prev = Some(code as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    // native-r5 seed 500122: a 3-frame GIF87a whose last frame's LZW data runs off the end. WIC
    // counts all three frames and decodes frame 0 (complete) byte-exact; the truncated frame is
    // decoded leniently rather than failing the whole image.
    #[test]
    fn truncated_multiframe_counts_all_frames() {
        let gif = unhex("47494638376108000800850000051fab1232141d510b0b6b6510940918c031300a122b15b326363e3337c93c76465a04084c2e00582ab24252045a60c848724e4cba1d51ed9e55e7a451f9747a004a66329e6e56fe6f55e56963a1607eda61859467993872aca769bb167ac8d17cfcfb821bd58a433d8d508c8861cb9ac75e88d38e89d2d18fe00182e3dabc0133bf2254a42d79a03df1a952d8b653efae990ab2dd12adf4e1c00729d20fbfc92254cd2cfacd3a06c15030d15ae6dd665ada850efd4c46f28f04f9a2efe9e6a72c000000000800080000084b00772408d1c1c605143d783018a0a1c4821c137ef85071e3c40a18060040288043868e001c30b888e1e18305022c40206891a1810405332890a86122420a010f5e8c7050e1000d111b02020021f90401000040002c00000000080008008512069017273715278d1f494b1a687a1e6cba117b3b1b99b215b5ba1cd2ed08f0af3c100232145630227222340a363c26223ed02ef0b1552bf149358950982b4bc33a4af0495ff7567c3048634d23644c966442e27a661a6fb3da75c3d863f102970e74901dc38429899a4f6b895cbe8b84379d968796fb5099f28db03f31bf4a02b27329b87b6db3962dbcc96aaaffa8c54c4ace4de1df5816d55408d89fecdeaf2cc8b9eed5d4ade312cee514f8f9446cfe5b0bf98edfec9ddaf7eefde5f06d084b001f1808d021c68b110d425898d1628504132836d8f8c122c28114352840c8012041091a155c7058304086021d1e267c8001028108150e34dcd8c183048e1e270a101090c10706061702020021f9040100003f002c000000000800080085add2a315090b0e100c1296e70db71a16c00506e05b10fc8f05f3b22b03b12917e53a5a12389ccb3abe4b32b6975628ac5020d25f570352aa9d57bf0257b9624dd5e6642e747f2fa372462e6c5e347765a76b894a6684637ebaad945d718556c6986b8f8aa5dc8cc58292dfb089fa15a93a58ba8306bc8fa0b883c4a0cfdac13c88d7484edb9077c7b65fcec75cced960c3f00fd1fe37cdf65bd0f3a8f411d1e013e3f8276bf524a8fd629ceb62c2e67931ea7b95e073d9f5ea5de6f568000000084b00496cd8a103018514234ed898");
        let frames = decode_gif(&gif).unwrap();
        assert_eq!(frames.len(), 3);
        let f0 = &frames[0];
        assert_eq!((f0.width, f0.height, f0.pixels.len()), (8, 8, 64));
        assert_eq!(f0.pixels, unhex("3b09211d3617283d3c0c031a250b39133f3e2a37272b300600100538323a011c182e311e1f16042c20082d190d120a33142435261129020f2f230e150734221b"));
    }

    #[test]
    fn malformed_gif_rejected_not_panic() {
        // LZW minimum code size out of range (255) — `1 << min_code` would overflow / wrap.
        let mut bad_code = b"GIF89a\x01\x00\x01\x00\x00\x00\x00".to_vec(); // 13-byte header, no GCT
        bad_code.extend_from_slice(&[0x2C, 0, 0, 0, 0, 1, 0, 1, 0, 0x00]); // 1×1 image, no LCT
        bad_code.extend_from_slice(&[0xFF, 0x00, 0x3B]); // min_code=255, empty LZW, trailer
        assert_eq!(decode_gif(&bad_code).err(), Some(WicError::BadImage));

        // Huge declared dimensions in a tiny file — would pre-size a multi-gigabyte buffer.
        let mut big = b"GIF89a\xFF\xFF\xFF\xFF\x00\x00\x00".to_vec();
        big.extend_from_slice(&[0x2C, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0x00]); // 65535×65535
        big.extend_from_slice(&[0x02, 0x00, 0x3B]); // min_code=2 (valid), empty LZW, trailer
        assert_eq!(decode_gif(&big).err(), Some(WicError::BadImage));
    }

    #[test]
    fn down_convert_non_indexed_matches_native() {
        // Gray8 2×1 [163,146] → native-r3 seed 100321's exact 41-byte GIF (median-cut palette
        // [163,146] padded to 4, error-diffused indices [0,1]). Exercises gif_down_convert end-to-end.
        let img = Decoded {
            width: 2,
            height: 1,
            format: fmt::GRAY8.into(),
            stride: 2,
            pixels: vec![163, 146],
            palette: vec![],
        };
        let want: Vec<u8> = vec![
            71, 73, 70, 56, 57, 97, 2, 0, 1, 0, 112, 0, 0, 44, 0, 0, 0, 0, 2, 0, 1, 0, 129, 163, 163,
            163, 146, 146, 146, 0, 0, 0, 0, 0, 0, 2, 2, 68, 10, 0, 59,
        ];
        assert_eq!(encode_gif(&img).unwrap(), want);
    }
}
