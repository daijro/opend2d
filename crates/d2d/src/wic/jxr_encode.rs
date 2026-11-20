//! WIC JPEG-XR (HD Photo) *encoding* as the WIC JPEG-XR codec does it, byte-for-byte. The codestream is
//! produced by statically-linked JXRLib (`ImageStrEnc{Init,Encode,Term}`); the only genuinely
//! wmphoto-specific logic is the WIC option → JXRLib `CWMIStrCodecParam` mapping (quality → QP) and
//! the WIC IFD container the frame writes around the codestream(s).
//!
//! We reuse the vendored, patched `jpegxr-pure-rs` (see `crates/vendor/jpegxr-pure-rs`) as the
//! codestream engine: [`jpegxr::encode_wmp_container`] drives its glue encoder with an explicit
//! `CWMIStrCodecParam` and returns a JXRLib container, built in memory (no temporary file, so it runs
//! on wasm32 too); we extract the raw image (and planar-alpha)
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
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    /// The encoder's bytes for generated images too large for one 4 KiB stream packet, in every
    /// wired format and in the default, lossless, low-quality and tiled configurations. The
    /// expected lengths and FNV-1a hashes were written by the encoder when it still went through a
    /// temporary file, so they pin the in-memory stream (and the wasm32 build) to those bytes.
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn encode_large_images_keeps_its_bytes() {
        let fnv = |b: &[u8]| b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100_0000_01b3));
        let mut seed = 0x2545_f491u32;
        let mut noise = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let mut got = Vec::new();
        for (last, bpp) in [(0x08u8, 1usize), (0x0a, 2), (0x0c, 3), (0x0d, 3), (0x0e, 4), (0x0f, 4), (0x16, 8)] {
            let mut format = [
                0x24, 0xc3, 0xdd, 0x6f, 0x03, 0x4e, 0xfe, 0x4b, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9, 0x00,
            ];
            format[15] = last;
            for (w, h) in [(1u32, 1u32), (37, 21), (200, 120)] {
                // A gradient under a little noise, so lossy and lossless outputs both stay large.
                let stride = w as usize * bpp;
                let pixels: Vec<u8> = (0..h as usize * stride)
                    .map(|i| ((i % stride) as u32 * 3 + (i / stride) as u32 * 5 + (noise() & 15)) as u8)
                    .collect();
                for (name, opts) in [
                    ("default", JxrEncodeOptions::default()),
                    ("lossless", JxrEncodeOptions { lossless: true, ..Default::default() }),
                    ("q030", JxrEncodeOptions { image_quality: 0.3, ..Default::default() }),
                    ("tiles", JxrEncodeOptions { h_tile_slices: 3, v_tile_slices: 2, ..Default::default() }),
                ] {
                    let out = encode_jxr(&format, w, h, &pixels, stride, &opts, (96.0, 72.0)).unwrap();
                    got.push(format!("{last:02x} {w}x{h} {name} {} {:016x}", out.len(), fnv(&out)));
                }
            }
        }
        // format, size, options, length, FNV-1a 64.
        let expected = [
            "08 1x1 default 197 c22e0eebf2e1f765",
            "08 1x1 lossless 197 0ecdbdaa07748b4b",
            "08 1x1 q030 197 9d28813538bdbdaa",
            "08 1x1 tiles 197 c22e0eebf2e1f765",
            "08 37x21 default 645 3ddeb94a20d2b1c5",
            "08 37x21 lossless 1058 9695aa9d1a839f09",
            "08 37x21 q030 263 2d11952f0a9c5189",
            "08 37x21 tiles 729 413fdfd93fdcb10f",
            "08 200x120 default 12175 856e80562b39003e",
            "08 200x120 lossless 19145 b4b75f3b2d55a609",
            "08 200x120 q030 3782 67362c36c16c7ae0",
            "08 200x120 tiles 12496 5281943acf4323a0",
            "0a 1x1 default 223 0f264f3caa04ab71",
            "0a 1x1 lossless 220 904dd3b829f13b43",
            "0a 1x1 q030 223 bfe98804be9d8c44",
            "0a 1x1 tiles 223 0f264f3caa04ab71",
            "0a 37x21 default 1450 5502255b0b601225",
            "0a 37x21 lossless 3447 e41ca891b52d9738",
            "0a 37x21 q030 525 5afcecd6c1b49819",
            "0a 37x21 tiles 1520 3d7905146bfc6832",
            "0a 200x120 default 29640 571e09a8b1a58561",
            "0a 200x120 lossless 63773 00e0714795e219ea",
            "0a 200x120 q030 6149 aef15c1c289924b0",
            "0a 200x120 tiles 30137 a4f67bf77dd55b86",
            "0c 1x1 default 223 a92721227d3a671a",
            "0c 1x1 lossless 221 0dc1b28f46bab177",
            "0c 1x1 q030 223 85239c26005d8a89",
            "0c 1x1 tiles 223 a92721227d3a671a",
            "0c 37x21 default 1133 98ed4f733bb263ea",
            "0c 37x21 lossless 3101 00987b161c7a334e",
            "0c 37x21 q030 507 ce15ef0587b1bdc8",
            "0c 37x21 tiles 1229 88cd80ab27238399",
            "0c 200x120 default 24283 059e7726d41d9437",
            "0c 200x120 lossless 58960 df37244375ded388",
            "0c 200x120 q030 8477 35206840ca215f87",
            "0c 200x120 tiles 24769 4b5289a32713097b",
            "0d 1x1 default 223 8cda20dd1e187239",
            "0d 1x1 lossless 221 fe562706822d35be",
            "0d 1x1 q030 223 b7480d3d3b35b5bc",
            "0d 1x1 tiles 223 8cda20dd1e187239",
            "0d 37x21 default 1161 7891ccd94ac9fab6",
            "0d 37x21 lossless 3123 e3060d5d8acac4ca",
            "0d 37x21 q030 501 f0719d6c85aa3c16",
            "0d 37x21 tiles 1280 8f195bf7243656e4",
            "0d 200x120 default 24065 cb0fc8ed484b2ae0",
            "0d 200x120 lossless 58964 33cfa1e78e36eb58",
            "0d 200x120 q030 8449 9326c48e809e6bdb",
            "0d 200x120 tiles 24652 e2a529208ee81536",
            "0e 1x1 default 223 542ff876019e4067",
            "0e 1x1 lossless 221 ec7f7df75291127d",
            "0e 1x1 q030 223 e45211db63e1c51f",
            "0e 1x1 tiles 223 542ff876019e4067",
            "0e 37x21 default 1106 a8c9d753190ff195",
            "0e 37x21 lossless 3106 02cc0e42d3422db5",
            "0e 37x21 q030 573 7bf5dd760d5ba55e",
            "0e 37x21 tiles 1245 152c076ccf886d7f",
            "0e 200x120 default 25608 842c4ab622a93c88",
            "0e 200x120 lossless 60720 7a5d59eb7610c324",
            "0e 200x120 q030 9675 357301124659d235",
            "0e 200x120 tiles 26230 8c6c77a5c62f0cfb",
            "0f 1x1 default 306 20a93253fdcf2e41",
            "0f 1x1 lossless 304 2ac0ba29f4498c31",
            "0f 1x1 q030 369 217ba253d3940eaf",
            "0f 1x1 tiles 306 20a93253fdcf2e41",
            "0f 37x21 default 2274 f334ef085acde19b",
            "0f 37x21 lossless 4189 5e2c76e912580e03",
            "0f 37x21 q030 1898 0790c6a34b65ff93",
            "0f 37x21 tiles 2519 23753443c1c4b089",
            "0f 200x120 default 47033 16ebe03d26b70008",
            "0f 200x120 lossless 81966 b35d5848c38bc028",
            "0f 200x120 q030 31244 6dad1db1530b1efc",
            "0f 200x120 tiles 48634 7b44cf18867c092a",
            "16 1x1 default 328 939674b39cc4abc0",
            "16 1x1 lossless 313 b1b8e3d1aa4cec06",
            "16 1x1 q030 325 3749a09bff1c02e2",
            "16 1x1 tiles 328 939674b39cc4abc0",
            "16 37x21 default 7002 a7b331b7b0bdd551",
            "16 37x21 lossless 10540 e88bb471cb111b7c",
            "16 37x21 q030 4000 bb1995d23ea2d6b8",
            "16 37x21 tiles 7485 9ef18e821e3f8ccb",
            "16 200x120 default 130440 333ed7407f3b9873",
            "16 200x120 lossless 201390 8b4ff062a5f5409a",
            "16 200x120 q030 57435 adb3bc8efc002422",
            "16 200x120 tiles 144719 0cd9478bd6e15d38",
        ];
        assert_eq!(got, expected);
    }
}
