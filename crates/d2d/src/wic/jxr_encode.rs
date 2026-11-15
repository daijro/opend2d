//! WIC JPEG-XR (HD Photo) *encoding* as the WIC JPEG-XR codec does it, byte-for-byte. The codestream is
//! produced by statically-linked JXRLib (`ImageStrEnc{Init,Encode,Term}`); the only genuinely
//! wmphoto-specific logic is the WIC option → JXRLib `CWMIStrCodecParam` mapping (quality → QP) and
//! the WIC IFD container the frame writes around the codestream(s).
//!
//! We reuse the vendored, patched `jpegxr-pure-rs` (see `crates/vendor/jpegxr-pure-rs`) as the
//! codestream engine: [`jpegxr::encode_wmp_container`] drives its glue encoder with an explicit
//! `CWMIStrCodecParam` and returns a JXRLib container; we extract the raw image (and planar-alpha)
//! codestreams — identical to the native codec's with the two byte-parity patches applied — and rewrap them in
//! WIC's own IFD container (template read off the committed oracle). Verified byte-for-byte against
//! the WIC JPEG-XR codec (oracle captured from native WIC; fixtures under `tests/wic/jxr/encode/`).
//!
//! The quality→QP map (from the JPEG-XR codec's frame encoder): only the single luma
//! QP index is derived from `ImageQuality`; the chroma-QP scaling and `208−192q²`/`255−240q²` tables
//! are keyed on *bit depth* and never fire for 8-bit, so every 8-bit format (gray8/bgr24/bgra32)
//! shares the one luma QP and leaves the other nine QP indices zero (strenc reuses the luma QP for
//! them). The planar-alpha codestream is always lossless (QP 1). `UseCodecOptions` false (WIC's
//! default) makes `Overlap`, `Subsampling`, `FrequencyOrder` and `InterleavedAlpha` no-ops: the
//! encoder uses the quality-tier overlap, YUV444, FREQUENCY order and planar alpha regardless.

use super::WicError;
use jpegxr::{BitstreamFormat, Overlap, StrCodecParam, StrColorFormat, Subband};

/// WIC JPEG-XR encoder options (the `IPropertyBag2` names `wmphoto` validates). Fields left at these
/// defaults reproduce WIC's default-lossy encode.
#[derive(Debug, Clone)]
pub struct JxrEncodeOptions {
    /// `ImageQuality` (VT_R4): 0.0–1.0; 1.0 (or `Lossless`) means lossless. WIC default 0.9.
    pub image_quality: f32,
    /// `Lossless` (VT_BOOL): force lossless regardless of `ImageQuality`.
    pub lossless: bool,
    /// `HorizontalTileSlices` / `VerticalTileSlices` (VT_UI2): number of tile slices (1 = none).
    pub h_tile_slices: u16,
    pub v_tile_slices: u16,
    // ponytail: `Overlap`, `Subsampling`, `FrequencyOrder` and `InterleavedAlpha` are no-ops unless
    // `UseCodecOptions` is set, which no committed oracle does — so the encoder uses the quality-tier
    // overlap, YUV444, FREQUENCY order and planar alpha regardless, and those options are accepted but
    // ignored (matching native at the default). Honouring UseCodecOptions needs VM-captured cases.
}

impl Default for JxrEncodeOptions {
    fn default() -> Self {
        JxrEncodeOptions { image_quality: 0.9, lossless: false, h_tile_slices: 1, v_tile_slices: 1 }
    }
}

/// How a WIC pixel format maps to the encoder: colour model, separate-alpha plane, and whether the
/// samples are 16-bit (which changes the quantization). Only the formats the oracle covers are wired
/// (8bppGray, 16bppBGR565, 24bppBGR/RGB, 32bppBGR/BGRA, 64bppRGBA).
struct EncodeFormat {
    bpp: u32,
    gray: bool,
    alpha: bool,
    /// A 16-bit-per-channel integer colour format, which the encoder quantizes more coarsely (see
    /// [`luma_qp_16bit`]) and with per-macroblock adaptive quantization.
    bits16: bool,
}

/// Map a WIC pixel-format GUID (16 raw little-endian bytes) to its encode descriptor. All JPEG-XR WIC
/// formats share the `6fddc324-…-3d77768dc9XX` family; only the final byte varies.
fn describe(format: &[u8; 16]) -> Option<EncodeFormat> {
    const PREFIX: [u8; 15] =
        [0x24, 0xc3, 0xdd, 0x6f, 0x03, 0x4e, 0xfe, 0x4b, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9];
    if format[..15] != PREFIX {
        return None;
    }
    let (bpp, gray, alpha) = match format[15] {
        0x08 => (8, true, false),   // 8bppGray
        0x0a => (16, false, false), // 16bppBGR565
        0x0c => (24, false, false), // 24bppBGR
        0x0d => (24, false, false), // 24bppRGB
        0x0e => (32, false, false), // 32bppBGR (BGRX, no alpha)
        0x0f => (32, false, true),  // 32bppBGRA
        0x16 => (64, false, true),  // 64bppRGBA (16-bit)
        _ => return None,
    };
    let bits16 = matches!(format[15], 0x16); // 16-bit-per-channel integer colour formats
    Some(EncodeFormat { bpp, gray, alpha, bits16 })
}

/// Luma QP index from `ImageQuality` `q`, exactly as the codec's frame encoder computes it. The constants are
/// 32-bit floats, the comparisons strict `>`, and the final `(int)` truncates toward zero.
fn luma_qp(q: f32) -> u8 {
    let qp: f32 = if q >= 1.0 || q < 0.0 {
        1.0
    } else if q > 0.9 {
        142.00999 - 140.0 * q
    } else if q > 0.8 {
        160.00999 - 160.0 * q
    } else if q > 0.4 {
        64.010002 - 40.0 * q
    } else if q > 0.1 {
        69.010002 - 51.0 * q
    } else {
        112.01 - 480.0 * q
    };
    qp as i32 as u8
}

/// Luma QP index for a 16-bit-per-channel integer colour format. The frame encoder selects this path
/// by pixel class (16-bit integer RGBA/RGB): the base QP is `(int)(208 - q²·192)` rather than the
/// 8-bit `luma_qp` curve, which for the default quality 0.9 gives 52 (vs 16 at 8-bit). `q` outside
/// `[0, 1)` means lossless, QP 1.
fn luma_qp_16bit(q: f32) -> u8 {
    if q >= 1.0 || q < 0.0 {
        1
    } else {
        (208.0 - q * q * 192.0) as i32 as u8
    }
}

/// The quality `q` the encoder uses after applying `Lossless` / `ImageQuality == 1.0`.
fn effective_quality(opts: &JxrEncodeOptions) -> f32 {
    if opts.lossless || opts.image_quality >= 1.0 {
        1.0
    } else {
        opts.image_quality
    }
}

/// Quality-tier overlap default (the quality-tier overlap table at `tier + 4`) with `tier = (q>0.4) + (q>=0.9)`
/// (tier0 → 2, tier1 → 1, tier2 → 0).
fn overlap_default(q: f32) -> Overlap {
    let tier = (q > 0.4) as usize + (q >= 0.9) as usize;
    match [2u8, 1, 0][tier] {
        0 => Overlap::None,
        1 => Overlap::One,
        _ => Overlap::Two,
    }
}

/// JXR quantization step for a QP index (T.832): `q < 16 ? q : (16 + (q&15)) << ((q>>4) - 1)`.
fn qp_step(q: i32) -> i32 {
    if q < 16 {
        q
    } else {
        (16 + (q & 15)) << ((q >> 4) - 1)
    }
}

/// The codec's luma→chroma QP step table entry (`table[i] = 2·qp_step(i+1)`).
fn chroma_step(luma: u8) -> i32 {
    2 * qp_step(luma as i32 + 1)
}

/// Nearest QP index in 1..=254 whose `chroma_step` is closest to `target` (`StepSize2QPLUT`), the
/// smaller index winning ties.
fn step_to_qp(target: i32) -> u8 {
    let (mut best, mut best_dist) = (0u8, i32::MAX);
    for i in 1..=254u8 {
        let d = (2 * qp_step(i as i32 + 1) - target).abs();
        if d < best_dist {
            best_dist = d;
            best = i;
        }
    }
    best
}

/// Set the chroma (U/V) DC/LP/HP and the luma HP QP indices for a lossy YUV444 image, exactly as
/// The codec's encoder init: from the luma step `s = chroma_step(luma)`, the QP indices are
/// `StepSize2QPLUT` of `s` scaled by 1.65 (chroma DC), 2.5 (chroma LP), 3.2 (chroma HP) and 1.5
/// (luma HP); HP indices clamp to a minimum of 2. Luma DC/LP stay at the luma QP (left as the param
/// default). The channel mode (cChMode 2, writing Y/U/V per subband) comes from the vendored crate's
/// `uQPMode` patch.
fn set_chroma_qp(scp: &mut StrCodecParam) {
    let s = chroma_step(scp.uiDefaultQPIndex) as f64;
    let cdc = step_to_qp((s * 1.65) as i32);
    let clp = step_to_qp((s * 2.5) as i32);
    let chp = step_to_qp((s * 3.2) as i32).max(2);
    scp.uiDefaultQPIndexYHP = step_to_qp((s * 1.5) as i32).max(2);
    scp.uiDefaultQPIndexU = cdc;
    scp.uiDefaultQPIndexV = cdc;
    scp.uiDefaultQPIndexULP = clp;
    scp.uiDefaultQPIndexVLP = clp;
    scp.uiDefaultQPIndexUHP = chp;
    scp.uiDefaultQPIndexVHP = chp;
}

/// The HD Photo container (`II` `0xBC`) written exactly as WIC's frame encoder does: first-IFD at
/// offset 8, 9 tags (or 11 with alpha) ascending, the 16-byte pixel-format GUID right after the IFD,
/// then the image codestream, then (if present) the alpha codestream. `res` is the (X, Y) resolution
/// as IEEE-754 f32 bit patterns — the frame's DPI (0.0 when `SetResolution` was never called, which
/// is what native writes; not a hardcoded 96).
fn build_container(
    format: &[u8; 16],
    width: u32,
    height: u32,
    image: &[u8],
    alpha: Option<&[u8]>,
    res: (u32, u32),
) -> Vec<u8> {
    let count: u16 = if alpha.is_some() { 11 } else { 9 };
    let guid_off = 8 + 2 + 12 * count as usize + 4; // past the next-IFD offset
    let image_off = guid_off + 16;
    let alpha_off = image_off + image.len();

    let mut out = Vec::with_capacity(alpha_off + alpha.map_or(0, <[u8]>::len));
    out.extend_from_slice(b"II");
    out.push(0xBC);
    out.push(0x01);
    out.extend_from_slice(&8u32.to_le_bytes()); // first IFD offset
    out.extend_from_slice(&count.to_le_bytes());

    let entry = |out: &mut Vec<u8>, tag: u16, typ: u16, cnt: u32, val: u32| {
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&typ.to_le_bytes());
        out.extend_from_slice(&cnt.to_le_bytes());
        out.extend_from_slice(&val.to_le_bytes());
    };
    const BYTE: u16 = 1;
    const LONG: u16 = 4;
    const FLOAT: u16 = 11;
    entry(&mut out, 0xBC01, BYTE, 16, guid_off as u32); // pixel-format GUID (offset)
    entry(&mut out, 0xBC02, LONG, 1, 0); // transformation
    entry(&mut out, 0xBC04, LONG, 1, 0); // image type flags
    entry(&mut out, 0xBC80, LONG, 1, width);
    entry(&mut out, 0xBC81, LONG, 1, height);
    entry(&mut out, 0xBC82, FLOAT, 1, res.0); // WidthResolution (frame DPI, f32 bits)
    entry(&mut out, 0xBC83, FLOAT, 1, res.1); // HeightResolution
    entry(&mut out, 0xBCC0, LONG, 1, image_off as u32);
    entry(&mut out, 0xBCC1, LONG, 1, image.len() as u32);
    if let Some(alpha) = alpha {
        entry(&mut out, 0xBCC2, LONG, 1, alpha_off as u32);
        entry(&mut out, 0xBCC3, LONG, 1, alpha.len() as u32);
    }
    out.extend_from_slice(&0u32.to_le_bytes()); // next IFD
    out.extend_from_slice(format); // pixel-format GUID
    out.extend_from_slice(image);
    if let Some(alpha) = alpha {
        out.extend_from_slice(alpha);
    }
    out
}

/// Pull the image (and planar-alpha) codestream slices out of a JXRLib container by its directory
/// tags (`0xBCC0`/`0xBCC1` image, `0xBCC2`/`0xBCC3` alpha) — the glue container we rewrap.
type Slices = (std::ops::Range<usize>, Option<std::ops::Range<usize>>);
fn extract_codestreams(c: &[u8]) -> Result<Slices, WicError> {
    let g = |at: usize| c.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).ok_or(WicError::BadImage);
    let h = |at: usize| c.get(at..at + 2).map(|b| u16::from_le_bytes([b[0], b[1]])).ok_or(WicError::BadImage);
    if c.len() < 8 || &c[..2] != b"II" || c[2] != 0xBC {
        return Err(WicError::BadImage);
    }
    let dir = g(4)? as usize;
    let count = h(dir)?;
    let (mut io, mut il, mut ao) = (0usize, 0u32, 0usize);
    for i in 0..usize::from(count) {
        let at = dir + 2 + 12 * i;
        let val = g(at + 8)?;
        match h(at)? {
            0xBCC0 => io = val as usize,
            0xBCC1 => il = val,
            0xBCC2 => ao = val as usize,
            _ => {}
        }
    }
    if io == 0 || il == 0 || io + il as usize > c.len() {
        return Err(WicError::BadImage);
    }
    // The glue writes the alpha byte-count tag (0xBCC3) as an absolute end offset, not a length, so
    // read the planar-alpha codestream from its offset to the end of the container.
    let alpha = (ao != 0 && ao < c.len()).then(|| ao..c.len());
    Ok((io..io + il as usize, alpha))
}

/// Encode a WIC pixel buffer to a `.jxr` file byte-for-byte as the WIC JPEG-XR codec does, under the given
/// WIC encoder options. `pixels` is the tightly-WIC-strided source buffer (`stride = ⌈w·bpp/8⌉`).
pub fn encode_jxr(
    format: &[u8; 16],
    width: u32,
    height: u32,
    pixels: &[u8],
    stride: usize,
    opts: &JxrEncodeOptions,
    dpi: (f32, f32),
) -> Result<Vec<u8>, WicError> {
    let fmt = describe(format).ok_or(WicError::Unsupported)?;
    let min_stride = ((width * fmt.bpp + 7) / 8) as usize;
    if width == 0 || height == 0 || stride < min_stride || pixels.len() < stride * height as usize {
        return Err(WicError::BadImage);
    }

    let q = effective_quality(opts);
    let mut scp = StrCodecParam {
        uiDefaultQPIndex: if fmt.bits16 { luma_qp_16bit(q) } else { luma_qp(q) },
        cfColorFormat: if fmt.gray { StrColorFormat::YOnly } else { StrColorFormat::Yuv444 },
        olOverlap: overlap_default(q),
        sbSubband: Subband::All,
        bfBitstreamFormat: BitstreamFormat::Frequency,
        bWriteProfileLevel: 1, // wmphoto stamps the PROFILE_LEVEL_INFO trailer on every codestream
        ..Default::default()
    };
    // HorizontalTileSlices / VerticalTileSlices → extra slice boundaries; 1 slice = no tiling. The
    // uiTile* offsets stay zero, so validate_tiling fills a uniform split.
    // ponytail: uniform tiling only — the oracle exercises 1 slice; revisit the H/V ↔ column/row
    // mapping if a multi-tile oracle lands.
    scp.cNumOfSliceMinus1V = opts.h_tile_slices.saturating_sub(1) as u32;
    scp.cNumOfSliceMinus1H = opts.v_tile_slices.saturating_sub(1) as u32;

    // Lossy colour: derive the per-subband chroma (and luma-HP) QP indices wmphoto scales from the
    // luma QP. Gray and lossless keep the single luma QP (the other nine indices stay zero).
    if !fmt.gray && scp.uiDefaultQPIndex > 1 {
        set_chroma_qp(&mut scp);
    }

    // A 4-channel image carries a planar, always-lossless alpha codestream.
    if fmt.alpha {
        scp.uAlphaMode = 2;
    }
    let alpha_qp = fmt.alpha.then_some(1u8);

    let container = jpegxr::encode_wmp_container(width, height, *format, pixels, stride, &scp, alpha_qp)
        .map_err(|_| WicError::BadImage)?;
    let (image, alpha) = extract_codestreams(&container)?;
    let alpha = alpha.filter(|_| fmt.alpha);
    Ok(build_container(format, width, height, &container[image], alpha.map(|a| &container[a]), (dpi.0.to_bits(), dpi.1.to_bits())))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 16-bit colour formats quantize with the `(int)(208 - q²·192)` base QP, not the 8-bit curve.
    #[test]
    fn luma_qp_16bit_default_quality() {
        assert_eq!(luma_qp_16bit(0.9), 52); // default quality 0.9 → 208 - 0.81·192 = 52.48 → 52
        assert_eq!(luma_qp_16bit(1.0), 1); // q ≥ 1 is lossless
        assert_eq!(luma_qp_16bit(-0.5), 1);
    }

    /// A `64bppRGBA` source encodes byte-for-byte like the native JPEG-XR encoder, including the
    /// per-tile adaptive (differential) quantization 16-bit colour uses above QP 16. The fixture is a
    /// 2×7 image at the default quality with no resolution set (native reference, decoded identically).
    #[test]
    fn encode_rgba64_matches_native() {
        const RGBA64: [u8; 16] = [
            0x24, 0xc3, 0xdd, 0x6f, 0x03, 0x4e, 0xfe, 0x4b, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9,
            0x16,
        ];
        // `.to_vec()` aligns the pixels to 2 bytes (the encoder reads 16-bit samples); `include_bytes!`
        // alone has no alignment guarantee. Real callers pass aligned WIC bitmap pixels.
        let input = include_bytes!("testdata/jxr_rgba64_2x7_in.bin").to_vec();
        let expected = include_bytes!("testdata/jxr_rgba64_2x7_native.jxr");
        let out =
            encode_jxr(&RGBA64, 2, 7, &input, 16, &JxrEncodeOptions::default(), (0.0, 0.0)).unwrap();
        assert_eq!(out.as_slice(), &expected[..]);
    }
}
