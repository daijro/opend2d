//! Frame colour-context extraction for `IWICBitmapFrameDecode::GetColorContexts`: the embedded ICC
//! profile and/or the EXIF colour-space id, exactly as native's per-codec `GetColorContexts`
//! produces them (JPEG: assembled APP2 `ICC_PROFILE` chunks + EXIF `ColorSpace`; TIFF: tag 34675 +
//! EXIF `ColorSpace`; PNG: the iCCP profile, inflated by the `png` crate). The caller supplies the
//! `IWICColorContext` objects; `GetColorContexts` initialises them.

const PNG: u128 = 0x1b7cfaf4_713f_473c_bbcd_6137425faeaf;
const JPEG: u128 = 0x19e4a5aa_5662_4fc5_a0c0_1758028e1057;
const TIFF: u128 = 0x163bcc30_e2e9_4f0b_961d_a3e9fdb788a3;

/// One colour context: an ICC profile, or an EXIF colour-space id.
pub(crate) enum ColorData {
    Profile(Vec<u8>),
    Exif(u32),
}

fn rd16(b: &[u8], o: usize, le: bool) -> u16 {
    match b.get(o..o + 2) {
        Some(x) if le => u16::from_le_bytes([x[0], x[1]]),
        Some(x) => u16::from_be_bytes([x[0], x[1]]),
        None => 0,
    }
}
fn rd32(b: &[u8], o: usize, le: bool) -> u32 {
    match b.get(o..o + 4) {
        Some(x) if le => u32::from_le_bytes([x[0], x[1], x[2], x[3]]),
        Some(x) => u32::from_be_bytes([x[0], x[1], x[2], x[3]]),
        None => 0,
    }
}

/// Find an IFD entry's (type, count, value-or-offset bytes) by tag, at IFD offset `off`.
fn ifd_entry(t: &[u8], off: u32, le: bool, tag: u16) -> Option<(u16, usize, &[u8])> {
    let o = off as usize;
    let n = rd16(t, o, le) as usize;
    for i in 0..n {
        let e = o + 2 + i * 12;
        if e + 12 > t.len() {
            break;
        }
        if rd16(t, e, le) == tag {
            let typ = rd16(t, e + 2, le);
            let cnt = rd32(t, e + 4, le) as usize;
            let size = type_size(typ) * cnt;
            let dpos = if size <= 4 { e + 8 } else { rd32(t, e + 8, le) as usize };
            return t.get(dpos..dpos + size).map(|d| (typ, cnt, d));
        }
    }
    None
}

fn type_size(t: u16) -> usize {
    match t {
        3 | 8 => 2,
        4 | 9 => 4,
        5 | 10 => 8,
        _ => 1,
    }
}

/// The EXIF `ColorSpace` (tag 0xA001) inside the Exif sub-IFD (tag 0x8769) of IFD0, if present.
fn exif_colorspace(tiff: &[u8]) -> Option<u32> {
    let le = tiff.get(0..2) != Some(b"MM");
    let ifd0 = rd32(tiff, 4, le);
    let (_, _, exif_ptr) = ifd_entry(tiff, ifd0, le, 0x8769)?;
    let (_, _, cs) = ifd_entry(tiff, rd32(exif_ptr, 0, le), le, 0xA001)?;
    Some(rd16(cs, 0, le) as u32)
}

/// Assemble a JPEG APP2 ICC profile from its `ICC_PROFILE\0` chunks (ordered by sequence number),
/// and read the EXIF `ColorSpace` from an APP1 Exif segment.
fn jpeg(raw: &[u8]) -> Vec<ColorData> {
    let mut chunks: Vec<(u8, Vec<u8>)> = Vec::new();
    let mut exif: Option<u32> = None;
    let mut o = 2;
    while o + 1 < raw.len() && raw[o] == 0xFF {
        let mut m = o + 1;
        while m < raw.len() && raw[m] == 0xFF {
            m += 1;
        }
        let marker = raw[m];
        o = m + 1;
        if marker == 0xD9 || marker == 0xDA {
            break;
        }
        if marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            continue;
        }
        let seglen = rd16(raw, o, false) as usize;
        if seglen < 2 || o + seglen > raw.len() {
            break;
        }
        let payload = &raw[o + 2..o + seglen];
        if marker == 0xE2 && payload.starts_with(b"ICC_PROFILE\0") && payload.len() >= 14 {
            chunks.push((payload[12], payload[14..].to_vec()));
        } else if marker == 0xE1 && payload.starts_with(b"Exif\0\0") {
            exif = exif_colorspace(&payload[6..]);
        }
        o += seglen;
    }
    let mut out = Vec::new();
    if !chunks.is_empty() {
        chunks.sort_by_key(|c| c.0);
        out.push(ColorData::Profile(chunks.iter().flat_map(|c| c.1.clone()).collect()));
    }
    if let Some(v) = exif {
        out.push(ColorData::Exif(v));
    }
    out
}

fn tiff(raw: &[u8], index: u32) -> Vec<ColorData> {
    let le = raw.get(0..2) != Some(b"MM");
    let mut ifd = rd32(raw, 4, le);
    for _ in 0..index {
        if ifd == 0 {
            return Vec::new();
        }
        let n = rd16(raw, ifd as usize, le) as usize;
        ifd = rd32(raw, ifd as usize + 2 + n * 12, le);
    }
    let mut out = Vec::new();
    if let Some((_, _, icc)) = ifd_entry(raw, ifd, le, 34675) {
        out.push(ColorData::Profile(icc.to_vec()));
    }
    if let Some(v) = exif_colorspace(raw) {
        out.push(ColorData::Exif(v));
    }
    out
}

/// The colour contexts for a decoded frame, in native's order (ICC first, then EXIF colour space).
pub(crate) fn frame_color_contexts(container: u128, raw: &[u8], index: u32) -> Vec<ColorData> {
    match container {
        JPEG => jpeg(raw),
        TIFF => tiff(raw, index),
        PNG => d2d::wic::png_icc_profile(raw).map(|p| vec![ColorData::Profile(p)]).unwrap_or_default(),
        _ => Vec::new(),
    }
}
