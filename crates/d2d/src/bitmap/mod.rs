//! Phase 1E bitmaps & imaging: the native pixel-format/alpha-mode validity table,
//! `CreateBitmap` option/DPI validation, premultiplied upload,
//! `DrawBitmap`/`DrawImage` compositing over the CPU target via [`crate::sampling`], and the
//! per-colour `D2D1ConvertColorSpace` / `Premultiply` / `UnPremultiply` arithmetic.
//!
//! The samplers share WARP's fixed-point texel arithmetic and float attribute planes. The smooth
//! axis-aligned modes use the bitmap-brush sampler; rotated anisotropic draws use the textured
//! quad and its per-primitive coverage in [`draw_bitmap_affine`].

use crate::sampling::{Extend, Filter, Texture};
use d2d_types::abi::*;
use d2d_types::contracts::PixelBuffer;
use windows_core::HRESULT;

const fn hr(code: u32) -> HRESULT {
    HRESULT(code as i32)
}
// HRESULTs specific to bitmap creation/mapping. The 1A set lives in
// `crate::state`; these are the extra ones this area can return.
pub const E_INVALIDARG: HRESULT = hr(0x8007_0057);
pub const WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT: HRESULT = hr(0x8898_2F80);

// --- pixel-format × alpha-mode validity ------------------------------------------------------

// Flag bits in native's 28-entry pixel-format table.
const VALID: u32 = 0x2; // a legal D2D bitmap format
const PREMUL: u32 = 0x4; // PREMULTIPLIED alpha legal
const STRAIGHT: u32 = 0x8; // STRAIGHT alpha legal
const IGNORE: u32 = 0x10; // IGNORE alpha legal
// (0x20 = GDI-compatible, 0x2_0000 = render-target-capable: carried in the table, not branched on here)
const SOURCE: u32 = 0x4_0000; // drawable as a DrawBitmap source (when 0x80 clear)
const NOT_SOURCE: u32 = 0x80; // clears SOURCE eligibility
const BLOCK_COMPRESSED: u32 = 0x8_0000; // 4-pixel-aligned copy rects

/// The usable rows of the native table: `(DXGI_FORMAT, flags)`. A format absent here — or present
/// with `VALID` clear — is unsupported, exactly as native treats the 15 non-usable rows. Values are
/// native's format validation table.
const FORMAT_FLAGS: &[(i32, u32)] = &[
    (65, 0x17_A78E),  // A8_UNORM
    (61, 0x10_A212),  // R8_UNORM
    (49, 0x10_A212),  // R8G8_UNORM
    (28, 0x17_A356),  // R8G8B8A8_UNORM
    (29, 0x16_0816),  // R8G8B8A8_UNORM_SRGB
    (87, 0x17_2376),  // B8G8R8A8_UNORM
    (91, 0x16_0836),  // B8G8R8A8_UNORM_SRGB
    (88, 0x15_0212),  // B8G8R8X8_UNORM
    (10, 0x16_9816),  // R16G16B16A16_FLOAT
    (11, 0x16_8816),  // R16G16B16A16_UNORM
    (2, 0x16_9816),   // R32G32B32A32_FLOAT
    (24, 0x14_2016),  // R10G10B10A2_UNORM
    (71, 0x1D_2216),  // BC1_UNORM
    (74, 0x1D_2216),  // BC2_UNORM
    (77, 0x1D_2216),  // BC3_UNORM
];

/// Flags for `format`, or `None` if it is not a usable D2D bitmap format.
pub fn format_flags(format: DXGI_FORMAT) -> Option<u32> {
    FORMAT_FLAGS
        .iter()
        .find(|(f, flags)| *f == format.0 && flags & VALID != 0)
        .map(|(_, flags)| *flags)
}

/// Pixel-format validation: a format is legal iff it carries `VALID`, and the alpha mode is
/// legal iff its flag bit is set — PREMULTIPLIED→`0x4`, STRAIGHT→`0x8`, IGNORE→`0x10`; UNKNOWN is
/// always rejected. Any failure maps to `WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT`. Returns the flags.
pub fn validate_pixel_format(format: DXGI_FORMAT, alpha: D2D1_ALPHA_MODE) -> Result<u32, HRESULT> {
    let flags = format_flags(format).ok_or(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT)?;
    let need = match alpha {
        D2D1_ALPHA_MODE_PREMULTIPLIED => PREMUL,
        D2D1_ALPHA_MODE_STRAIGHT => STRAIGHT,
        D2D1_ALPHA_MODE_IGNORE => IGNORE,
        _ => return Err(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT), // UNKNOWN(0) and any out-of-range
    };
    if flags & need == 0 {
        return Err(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT);
    }
    Ok(flags)
}

/// True if `format` may be a `DrawBitmap`/`DrawImage` source (`SOURCE` set and
/// `NOT_SOURCE` clear).
pub fn is_drawable_source(flags: u32) -> bool {
    flags & SOURCE != 0 && flags & NOT_SOURCE == 0
}

// --- CreateBitmap option / DPI validation ----------------------------------------------------

/// Bitmap-option validation (step 1): reject unknown option bits and the illegal combinations.
/// `CPU_READ ⇔ CANNOT_DRAW`, `TARGET` and `CPU_READ` are mutually exclusive, `GDI_COMPATIBLE`
/// requires `TARGET`.
pub fn validate_bitmap_options(options: D2D1_BITMAP_OPTIONS) -> Result<(), HRESULT> {
    let o = options.0;
    let target = o & D2D1_BITMAP_OPTIONS_TARGET.0 != 0;
    let cpu_read = o & D2D1_BITMAP_OPTIONS_CPU_READ.0 != 0;
    let cannot_draw = o & D2D1_BITMAP_OPTIONS_CANNOT_DRAW.0 != 0;
    let gdi = o & D2D1_BITMAP_OPTIONS_GDI_COMPATIBLE.0 != 0;
    // Any bit at or above 0x10 (beyond TARGET|CANNOT_DRAW|CPU_READ|GDI = 0xF) is invalid.
    if o & !0xF != 0
        || (cpu_read && target)
        || (cpu_read != cannot_draw)
        || (gdi && !target)
    {
        return Err(E_INVALIDARG);
    }
    Ok(())
}

/// `CreateBitmap` DPI rule (step 2): `(0,0)` inherits the context DPI; exactly one `≤ 0` is
/// invalid; otherwise the pair is kept.
pub fn resolve_dpi(dpi_x: f32, dpi_y: f32, context: (f32, f32)) -> Result<(f32, f32), HRESULT> {
    if dpi_x == 0.0 && dpi_y == 0.0 {
        Ok(context)
    } else if dpi_x > 0.0 && dpi_y > 0.0 {
        Ok((dpi_x, dpi_y))
    } else {
        Err(E_INVALIDARG)
    }
}

/// The full `CreateBitmap` validation chain: options, DPI, format/alpha. Returns the
/// resolved DPI and format flags. `CreateBitmap` errors are observable, so order matches native.
pub fn validate_create_bitmap(
    format: DXGI_FORMAT,
    alpha: D2D1_ALPHA_MODE,
    options: D2D1_BITMAP_OPTIONS,
    dpi_x: f32,
    dpi_y: f32,
    context_dpi: (f32, f32),
) -> Result<((f32, f32), u32), HRESULT> {
    validate_bitmap_options(options)?;
    let dpi = resolve_dpi(dpi_x, dpi_y, context_dpi)?;
    let flags = validate_pixel_format(format, alpha)?;
    // Step 5: block-compressed with any TARGET/CANNOT_DRAW option is invalid.
    if flags & BLOCK_COMPRESSED != 0
        && options.0 & (D2D1_BITMAP_OPTIONS_TARGET.0 | D2D1_BITMAP_OPTIONS_CANNOT_DRAW.0) != 0
    {
        return Err(E_INVALIDARG);
    }
    Ok((dpi, flags))
}

// --- premultiplied upload ---------------------------------------------------------------------

/// Straight RGBA8 → tightly packed premultiplied BGRA8, rounding each channel half-up
/// (`(c·a+127)/255`). This matches the harness upload that produced the oracle bitmaps
/// so an identity-scale `DrawBitmap` round-trips byte-for-byte.
pub fn premultiply_rgba(rgba: &[u8]) -> Vec<u8> {
    rgba.as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| {
            let a = p[3] as u32;
            let c = |v: u8| ((v as u32 * a + 127) / 255) as u8;
            [c(p[2]), c(p[1]), c(p[0]), p[3]]
        })
        .collect()
}

/// Byte width of an uncompressed texel supported by the bitmap storage paths.
pub fn bytes_per_pixel(format: DXGI_FORMAT) -> Option<usize> {
    match format.0 {
        2 => Some(16),
        10 | 11 => Some(8),
        24 | 28 | 29 | 87 | 88 | 91 => Some(4),
        49 => Some(2),
        61 | 65 => Some(1),
        _ => None,
    }
}

/// A realized D2D bitmap: premultiplied BGRA8 pixels plus the stored format/alpha/DPI/options.
#[derive(Clone)]
pub struct Bitmap {
    pub width: u32,
    pub height: u32,
    pub premul_bgra: Vec<u8>,
    pub format: DXGI_FORMAT,
    pub alpha: D2D1_ALPHA_MODE,
    pub dpi: (f32, f32),
}

impl Bitmap {
    /// `ID2D1Bitmap::CopyFromBitmap`: `src`'s texels in `rect` (default: all of it) copied to `point`
    /// (default: the origin) of this bitmap. The pixel formats and alpha modes must match, else
    /// E_INVALIDARG; otherwise the rect is clamped to the source and the copy clipped to this bitmap, and
    /// an empty or inverted rect copies nothing (VM probes: all S_OK).
    pub fn copy_from_bitmap(&mut self, src: &Bitmap, point: Option<[u32; 2]>, rect: Option<[u32; 4]>) -> Result<(), windows_core::HRESULT> {
        let Some(bpp) = bytes_per_pixel(self.format).filter(|_| self.format.0 == src.format.0 && self.alpha.0 == src.alpha.0) else {
            return Err(crate::state::E_INVALIDARG);
        };
        let [l, t, r, b] = rect.unwrap_or([0, 0, src.width, src.height]);
        let (r, b) = (r.min(src.width), b.min(src.height));
        let [x, y] = point.unwrap_or([0, 0]);
        if l >= r || t >= b || x >= self.width || y >= self.height {
            return Ok(());
        }
        let (w, h) = ((r - l).min(self.width - x), (b - t).min(self.height - y));
        let row = w as usize * bpp;
        for k in 0..h as usize {
            let s = ((t as usize + k) * src.width as usize + l as usize) * bpp;
            let d = ((y as usize + k) * self.width as usize + x as usize) * bpp;
            self.premul_bgra[d..d + row].copy_from_slice(&src.premul_bgra[s..s + row]);
        }
        Ok(())
    }

    /// Premultiplied RGBA at an integer texel, preserving float source precision.
    pub(crate) fn texel_rgba(&self, x: i32, y: i32) -> [f32; 4] {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 { return [0.0; 4]; }
        let n = y as usize * self.width as usize + x as usize;
        let d = &self.premul_bgra;
        // Decode by the real format, not an assumed 4-bpp BGRA8: a 1-bpp A8/R8 or 2-bpp R8G8/R16 source
        // read as BGRA8 would index past its buffer, and an R8G8B8A8 read as BGRA8 swaps R/B. Mirrors the
        // effects decoder so the brush/draw path agrees with it (r23 A8 sources).
        match self.format.0 {
            2 => core::array::from_fn(|c| f32::from_le_bytes(d[n * 16 + c * 4..n * 16 + c * 4 + 4].try_into().unwrap())), // R32G32B32A32_FLOAT
            10 => core::array::from_fn(|c| crate::float::f16_to_f32(u16::from_le_bytes(d[n * 8 + c * 2..n * 8 + c * 2 + 2].try_into().unwrap()))), // R16G16B16A16_FLOAT
            11 => core::array::from_fn(|c| u16::from_le_bytes(d[n * 8 + c * 2..n * 8 + c * 2 + 2].try_into().unwrap()) as f32 / 65535.0), // R16G16B16A16_UNORM
            65 => [0.0, 0.0, 0.0, d[n] as f32 / 255.0],                                  // A8_UNORM
            61 => [d[n] as f32 / 255.0, 0.0, 0.0, 1.0],                                  // R8_UNORM
            49 => [d[n * 2] as f32 / 255.0, d[n * 2 + 1] as f32 / 255.0, 0.0, 1.0],      // R8G8_UNORM
            28 | 29 => core::array::from_fn(|c| d[n * 4 + c] as f32 / 255.0),            // R8G8B8A8_UNORM(_SRGB): RGBA order
            _ => [d[n * 4 + 2] as f32 / 255.0, d[n * 4 + 1] as f32 / 255.0, d[n * 4] as f32 / 255.0, d[n * 4 + 3] as f32 / 255.0], // BGRA8
        }
    }

    /// This bitmap as premultiplied BGRA8 for the colour-sampling path, normalising the two things the
    /// BGRA8 sampler assumes: channel order and premultiplied alpha. R8G8B8A8 stores RGBA, so R and B are
    /// exchanged. A STRAIGHT source is premultiplied; an IGNORE source is made opaque (its colour is
    /// already straight, i.e. premultiplied at alpha 1). Only 4-bpp byte colour formats are normalised —
    /// A8 (1 bpp, rejected before any sampler), the float formats and R16 pass through so their own paths
    /// handle them. The stored bitmap keeps its created format/alpha so a pixel-format query is unchanged.
    pub(crate) fn as_sampled_bgra8(&self) -> std::borrow::Cow<'_, Bitmap> {
        if self.premul_bgra.len() != self.width as usize * self.height as usize * 4 {
            return std::borrow::Cow::Borrowed(self);
        }
        let rgba_order = self.format.0 == 28 || self.format.0 == 29; // R8G8B8A8(_SRGB)
        let ignore = self.alpha == D2D1_ALPHA_MODE_IGNORE;
        let straight = self.alpha == D2D1_ALPHA_MODE_STRAIGHT;
        if !rgba_order && !ignore && !straight {
            return std::borrow::Cow::Borrowed(self); // already premultiplied BGRA8
        }
        let mut px = self.premul_bgra.clone();
        for c in px.chunks_exact_mut(4) {
            if rgba_order {
                c.swap(0, 2);
            }
            if ignore {
                c[3] = 255;
            } else if straight {
                let a = c[3] as u32;
                for k in 0..3 {
                    c[k] = ((c[k] as u32 * a + 127) / 255) as u8;
                }
            }
        }
        std::borrow::Cow::Owned(Bitmap { premul_bgra: px, format: DXGI_FORMAT_B8G8R8A8_UNORM, alpha: D2D1_ALPHA_MODE_PREMULTIPLIED, ..self.clone() })
    }

    /// [`as_sampled_bgra8`](Self::as_sampled_bgra8) for an image brush's source-rect crop: an IGNORE_ALPHA bitmap keeps
    /// its stored alpha bytes, its colour as stored, where a draw of the whole bitmap reads it opaque (VM probe-ign: a
    /// crop inside or past the bitmap; r27 32455).
    pub(crate) fn as_cropped_bgra8(&self) -> std::borrow::Cow<'_, Bitmap> {
        if self.alpha != D2D1_ALPHA_MODE_IGNORE || self.premul_bgra.len() != self.width as usize * self.height as usize * 4 {
            return self.as_sampled_bgra8();
        }
        let mut px = self.premul_bgra.clone();
        if self.format.0 == 28 || self.format.0 == 29 {
            px.chunks_exact_mut(4).for_each(|c| c.swap(0, 2));
        }
        std::borrow::Cow::Owned(Bitmap { premul_bgra: px, format: DXGI_FORMAT_B8G8R8A8_UNORM, alpha: D2D1_ALPHA_MODE_PREMULTIPLIED, ..self.clone() })
    }

    /// An A8 (alpha-only) bitmap presented as BGRA8 for the sprite sampler: the single alpha byte is
    /// broadcast to all four channels, (a,a,a,a), keeping the source alpha mode (so a later
    /// [`Self::as_sampled_bgra8`] premultiplies a STRAIGHT source and opaques an IGNORE one). Native
    /// samples an A8 sprite source this way — an untinted A8 sprite is invisible (white at coverage a)
    /// and a tinted one is tint·a — rather than rejecting it like DrawBitmap/DrawImage (VM-probed,
    /// direct and command-list-recorded; both S_OK).
    pub(crate) fn a8_as_bgra(&self) -> Bitmap {
        let mut px = vec![0u8; self.premul_bgra.len() * 4];
        for (i, &a) in self.premul_bgra.iter().enumerate() {
            px[i * 4..i * 4 + 4].copy_from_slice(&[a, a, a, a]);
        }
        Bitmap { premul_bgra: px, format: DXGI_FORMAT_B8G8R8A8_UNORM, ..self.clone() }
    }

    /// Whether every texel is fully opaque (alpha ≡ 1): an IGNORE-alpha format is opaque by definition,
    /// otherwise all stored alphas must be max. Used to gate the COPY huge-draw trapezoid path — only a
    /// fully-opaque draw (this AND `opacity == 1`) replaces the dest exterior, so only then does native
    /// build the `NeedOutsideGeometry` trapezoid mesh (`IsBrushOpaque`'s opacity·source-opaque test).
    pub(crate) fn is_fully_opaque(&self) -> bool {
        if self.alpha == crate::abi::D2D1_ALPHA_MODE_IGNORE {
            return true;
        }
        match self.format {
            DXGI_FORMAT_R32G32B32A32_FLOAT | DXGI_FORMAT_R16G16B16A16_FLOAT => (0..self.height as i32)
                .all(|y| (0..self.width as i32).all(|x| self.texel_rgba(x, y)[3] >= 1.0)),
            _ => self.premul_bgra.chunks_exact(4).all(|px| px[3] == 255),
        }
    }

    /// Copy of the texel sub-rect `[l, t, r, b)`, each edge rounded: the source rectangle of image
    /// brushes and BlendImage, which native realises as a crop. Texels past the bitmap are transparent,
    /// so a WRAP image brush tiles the whole source rect (r5 5930: a 7×8 bitmap under a 24×39 source).
    pub fn crop(&self, rect: [f32; 4]) -> Bitmap {
        // ponytail: a window past 2^24 texels keeps only the bitmap's part; native tiles it lazily.
        let [l, t, r, b] = rect.map(|v| v.round().clamp(-(1 << 24) as f32, (1 << 24) as f32) as i64);
        let (mut l, mut t, mut r, mut b) = (l, t, r.max(l), b.max(t));
        if (r - l) * (b - t) > 1 << 24 {
            (l, t, r, b) = (l.clamp(0, self.width as i64), t.clamp(0, self.height as i64), r.clamp(0, self.width as i64), b.clamp(0, self.height as i64));
        }
        let (w, h) = ((r - l) as u32, (b - t) as u32);
        // Copy by the format's real stride, not an assumed 4 bpp — a 1-bpp A8 (or 2-bpp R8G8/R16) source
        // reaching the crop (an image-brush/BlendImage source rect) would otherwise read 4× past its
        // backing buffer and panic (r23 28307/28733/28749/28860). `premul_bgra` holds whatever the
        // source format stored; `bytes_per_pixel` knows every format we accept.
        let bpp = bytes_per_pixel(self.format).unwrap_or(4) as u32;
        let mut px = vec![0u8; (w * h * bpp) as usize];
        let (x0, x1) = (l.max(0), r.min(self.width as i64));
        for y in t.max(0)..b.min(self.height as i64) {
            if x1 <= x0 { break; }
            let s0 = ((y as u32 * self.width + x0 as u32) * bpp) as usize;
            let d0 = ((((y - t) as u32) * w + (x0 - l) as u32) * bpp) as usize;
            let n = ((x1 - x0) as u32 * bpp) as usize;
            px[d0..d0 + n].copy_from_slice(&self.premul_bgra[s0..s0 + n]);
        }
        Bitmap { width: w, height: h, premul_bgra: px, format: self.format, alpha: self.alpha, dpi: self.dpi }
    }

    /// A premultiplied-BGRA8 bitmap built from inline straight RGBA8 (the `bitmap` scene op), at
    /// 96 DPI — the harness's upload.
    pub fn from_straight_rgba(width: u32, height: u32, rgba: &[u8]) -> Self {
        Self {
            width,
            height,
            premul_bgra: premultiply_rgba(rgba),
            format: DXGI_FORMAT_B8G8R8A8_UNORM,
            alpha: D2D1_ALPHA_MODE_PREMULTIPLIED,
            dpi: (96.0, 96.0),
        }
    }

    /// A float (`R32G32B32A32_FLOAT`) bitmap from inline RGBA f32 — e.g. a scene-position image a
    /// SampleMap shader samples at full precision (an 8bpc bitmap would clamp the coordinates to [0,1]).
    /// Texels are stored as raw little-endian f32, RGBA order (16 bytes each), decoded by `bitmap_texel`.
    pub fn from_f32_rgba(width: u32, height: u32, rgba: &[f32]) -> Self {
        let mut px = Vec::with_capacity(rgba.len() * 4);
        for v in rgba {
            px.extend_from_slice(&v.to_le_bytes());
        }
        Self {
            width,
            height,
            premul_bgra: px,
            format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            alpha: D2D1_ALPHA_MODE_PREMULTIPLIED,
            dpi: (96.0, 96.0),
        }
    }

    fn texture(&self) -> Texture<'_> {
        Texture { data: &self.premul_bgra, width: self.width, height: self.height }
    }
}

// --- drawing ----------------------------------------------------------------------------------

/// Composite modes this phase reproduces (`D2D1_COMPOSITE_MODE`). The rest await extension.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Composite {
    SourceOver,
    SourceCopy,
    Plus,
}

/// Select the basic point, linear, or cubic filter. Callers dispatch MSL, anisotropic and
/// high-quality cubic to their multi-tap kernels before using this helper.
/// How d2d1 draws an image brush over a bitmap (`TryCreateFastPathBitmapBrush`): as a plain bitmap brush over the whole
/// bitmap, as one sampling a sub-rect of it, or not as a bitmap brush at all (the image brush's effect graph).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum FastPathBitmap {
    None,
    /// The source rect is the whole bitmap: the brush keeps its extend modes and interpolation.
    Copy,
    /// A whole-pixel sub-rect `[l, t, r, b]` (pixels), sampled CLAMPed to its texel centres.
    Scaled([f32; 4]),
}

/// d2d1's test for drawing an image brush over a bitmap as a plain bitmap brush. `source_px` is the brush's (sorted)
/// source rect in the bitmap's pixels, `size` the bitmap's, `extend` the D2D extend modes (0 = CLAMP). `masked`: the
/// fill also takes an opacity mask; `sub_rect`: the caller accepts a sub-rect brush (a FillRectangle does).
/// - The source rect, rounded with halves down, must stay within 1/64 of itself on every edge, be non-empty, and lie
///   inside the bitmap.
/// - The whole bitmap converts at NEAREST or LINEAR with any extend, and at CUBIC or MULTI_SAMPLE_LINEAR unless masked
///   (WARP's device allows their WRAP and MIRROR: a full-source CUBIC WRAP fill draws no atlas cells).
/// - A sub-rect converts only at NEAREST or LINEAR with both extends CLAMP.
pub(crate) fn fast_path_bitmap(source_px: [f32; 4], size: [u32; 2], extend: [u32; 2], interpolation: u32, masked: bool, sub_rect: bool) -> FastPathBitmap {
    if !source_px.iter().all(|v| v.is_finite()) {
        return FastPathBitmap::None;
    }
    let r = source_px.map(|v| (v - 0.5).ceil());
    if !(r[2] > r[0] && r[3] > r[1]) || (0..4).any(|k| (r[k] - source_px[k]).abs() >= 1.0 / 64.0) {
        return FastPathBitmap::None;
    }
    let (w, h) = (size[0] as f32, size[1] as f32);
    if r[0] < 0.0 || r[1] < 0.0 || w < r[2] || h < r[3] {
        return FastPathBitmap::None;
    }
    if r == [0.0, 0.0, w, h] {
        return if interpolation <= 1 || (matches!(interpolation, 2 | 3) && !masked) { FastPathBitmap::Copy } else { FastPathBitmap::None };
    }
    if sub_rect && extend == [0, 0] && interpolation <= 1 { FastPathBitmap::Scaled(r) } else { FastPathBitmap::None }
}

pub fn filter_of(interpolation: u32) -> Filter {
    match interpolation {
        0 => Filter::Nearest,
        2 => Filter::Cubic,
        _ => Filter::Linear,
    }
}

/// WARP's UNORM8 multiply `round(v·m/255)`, emitted as `t = v·m + 128; (t + (t >> 8)) >> 8`
/// (verified in the sampler dump, same formula the whole fixed-point pixel pipeline uses). Shared
/// with the brush colour-texturing composite (`crate::brush::fill_geometry`).
pub(crate) fn unorm_mul(v: i32, m: i32) -> i32 {
    let t = v * m + 128;
    (t + (t >> 8)) >> 8
}

/// Composite one UNORM8 premultiplied source (`sc`, already opacity- and coverage-scaled) onto a
/// UNORM8 destination, in the integer fixed point WARP uses for a UNORM8 target:
/// copy replaces; additive saturates; source-over is `sc + round(dst·(255−sc.a)/255)`, saturating.
fn composite(sc: [i32; 4], dst: [u8; 4], mode: Composite) -> [u8; 4] {
    let out: [i32; 4] = match mode {
        Composite::SourceCopy => sc,
        Composite::Plus => core::array::from_fn(|i| sc[i] + dst[i] as i32),
        Composite::SourceOver => {
            let ia = 255 - sc[3];
            core::array::from_fn(|i| sc[i] + unorm_mul(dst[i] as i32, ia))
        }
    };
    out.map(|v| v.clamp(0, 255) as u8)
}

/// `DrawBitmap`: composite `bitmap` from `source` (texels) into the axis-aligned device-space
/// `dest` rectangle with premultiplied source-over, scaling opacity into the sample. Fractional
/// `dest` edges get WARP's axis-aligned anti-aliased-rectangle coverage (exact per-axis fractional
/// overlap, confirmed byte-exact against the oracle), applied to the premultiplied source.
pub fn draw_bitmap(
    target: &mut PixelBuffer,
    bitmap: &Bitmap,
    dest: [f32; 4],
    source: [f32; 4],
    opacity: f32,
    interpolation: u32,
) {
    draw_bitmap_with(target, bitmap, dest, source, opacity, interpolation, crate::pixels::Blend::SourceOver, crate::pixels::Aa::PerPrimitive, None);
}

/// [`draw_bitmap`] under a primitive blend and antialias mode.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_bitmap_with(
    target: &mut PixelBuffer,
    bitmap: &Bitmap,
    dest: [f32; 4],
    source: [f32; 4],
    opacity: f32,
    interpolation: u32,
    blend: crate::pixels::Blend,
    aa: crate::pixels::Aa,
    brush_plane: Option<[f32; 6]>,
) {
    // The sampled modes map the source as native's DrawBitmap affine does: per axis [min, max], the min held
    // at >= 0 and the max at <= the bitmap's size, so a source wholly past an edge keeps a reversed span and
    // samples the clamped edge backwards (r24 29713: rows (3.81, 1) of a 1-high bitmap).
    let (w, h) = (bitmap.width as f32, bitmap.height as f32);
    let half = [source[0].min(source[2]).max(0.0), source[1].min(source[3]).max(0.0), source[0].max(source[2]).min(w), source[1].max(source[3]).min(h)];
    if matches!(interpolation, 2..=5) {
        crate::float::draw_sampled_bitmap(target, bitmap, dest, half, opacity, interpolation, blend, aa);
        return;
    }
    let source = norm_source(source, bitmap.width as f32, bitmap.height as f32);
    if crate::float::is_highp(bitmap.format) {
        crate::float::draw_sampled_bitmap(target, bitmap, dest, source, opacity, interpolation, blend, aa);
        return;
    }
    // NEAREST and LINEAR iterate the same mapping: a source wholly past an edge runs its reversed span backwards across
    // the dest, reading the clamped edge until it crosses the bitmap (VM: rows [1e9, 1600] of a 12-row bitmap read the
    // last row, then the first on the dest's bottom row).
    blit(target, bitmap, dest, half, opacity, filter_of(interpolation), Composite::SourceOver, Extend::Clamp, false, blend, aa, Some(half), brush_plane);
}

/// Realize a DrawImage source sub-rect into `target` with SOURCE_OVER, using DrawImage's NEAREST
/// POINT placement (not DrawBitmap's edge AA) — the slow-path composite realizes into a transparent
/// intermediate here, then applies the composite mode separately.
pub(crate) fn draw_image_rect(target: &mut PixelBuffer, bitmap: &Bitmap, dest: [f32; 4], source: [f32; 4], interpolation: u32) {
    let source = norm_source(source, bitmap.width as f32, bitmap.height as f32);
    if crate::float::is_highp(bitmap.format) || matches!(interpolation, 2..=5) {
        crate::float::draw_sampled_bitmap(target, bitmap, dest, source, 1.0, interpolation, crate::pixels::Blend::SourceOver, crate::pixels::Aa::PerPrimitive);
        return;
    }
    blit(target, bitmap, dest, source, 1.0, filter_of(interpolation), Composite::SourceOver, Extend::Clamp, true, crate::pixels::Blend::SourceOver, crate::pixels::Aa::PerPrimitive, None, None);
}

/// Native normalizes the `DrawBitmap` source rectangle — per axis `[min, max]` — and clamps it to the
/// bitmap bounds `[0, dim]` before sampling, so a flipped or out-of-range source rect samples the same
/// texels as the equivalent in-bounds forward rect. VM-probed: flip/clamp of the source produced
/// byte-identical output to the forward in-bounds rect (fuzz 20008/20025).
pub(crate) fn norm_source(s: [f32; 4], w: f32, h: f32) -> [f32; 4] {
    [
        s[0].min(s[2]).clamp(0.0, w),
        s[1].min(s[3]).clamp(0.0, h),
        s[0].max(s[2]).clamp(0.0, w),
        s[1].max(s[3]).clamp(0.0, h),
    ]
}

/// WARP's recursive 2×2 box mip chain of a premultiplied BGRA8 texture, for the minifying sampler.
/// Each level is `round_ties_even((a+b+c+d)/4)` per channel over the previous level's 2×2 blocks —
/// pinned against WARP's `GenerateMips` readback (mip1 of a `B=x` gradient is `2i`,
/// mip2 is `4j+1`, i.e. each level built from the previous, ties-to-even, in the stored premul
/// space). `levels[0]` is the base; it runs down to 1×1. Power-of-two sources (the only minify
/// scenes) halve cleanly; odd dims would drop the last row/col.
pub(crate) fn mip_chain(base: &Texture) -> Vec<(u32, u32, Vec<u8>)> {
    // Every level here is 4-bpp BGRA8. A caller that hands a non-4-bpp texture (an A8/R8 source that
    // slipped into the anisotropic brush path, r23 28262/28345/28363) would make the 2×2 average read
    // past the backing buffer, so refuse to mip it and return just the base level unchanged.
    if base.data.len() < (base.width as usize * base.height as usize * 4) {
        return vec![(base.width, base.height, base.data.to_vec())];
    }
    let mut levels = vec![(base.width, base.height, base.data.to_vec())];
    loop {
        let (pw, ph, prev) = levels.last().unwrap();
        let (pw, ph) = (*pw, *ph);
        if pw <= 1 && ph <= 1 {
            break;
        }
        let (nw, nh) = ((pw / 2).max(1), (ph / 2).max(1));
        let mut data = vec![0u8; (nw * nh * 4) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let (x1, y1) = ((2 * x + 1).min(pw - 1), (2 * y + 1).min(ph - 1));
                let p = |xx: u32, yy: u32, ch: u32| prev[((yy * pw + xx) * 4 + ch) as usize] as f32;
                for ch in 0..4 {
                    let avg = (p(2 * x, 2 * y, ch) + p(x1, 2 * y, ch) + p(2 * x, y1, ch) + p(x1, y1, ch)) * 0.25;
                    data[((y * nw + x) * 4 + ch) as usize] = avg.round_ties_even() as u8;
                }
            }
        }
        levels.push((nw, nh, data));
    }
    levels
}

// The atlas copies the last source column and row before generating mips. A single extra
// texel at each level retains those independently averaged borders; clamping the unpadded
// mip instead would repeat its last 2x2 average and lose the original edge colour.
pub(crate) fn float_mip_chain(bitmap: &Bitmap, padded: bool) -> Vec<(u32, u32, Vec<[f32; 4]>)> {
    float_mip_chain_precision(bitmap, padded, 5)
}

pub(crate) fn float_mip_chain_precision(bitmap: &Bitmap, padded: bool, precision: u32) -> Vec<(u32, u32, Vec<[f32; 4]>)> {
    let (w, h) = (bitmap.width, bitmap.height);
    let p = u32::from(padded);
    let mut base = Vec::with_capacity(((w + p) * (h + p)) as usize);
    for y in 0..h + p {
        for x in 0..w + p {
            let value = bitmap.texel_rgba(x.min(w - 1) as i32, y.min(h - 1) as i32);
            base.push(crate::effects::quantize_to(precision, value));
        }
    }
    float_mip_levels(w, h, padded, precision, base)
}

pub(crate) fn float_mip_levels(mut w: u32, mut h: u32, padded: bool, precision: u32, base: Vec<[f32; 4]>) -> Vec<(u32, u32, Vec<[f32; 4]>)> {
    let p = u32::from(padded);
    let mut levels = vec![(w + p, h + p, base)];
    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let prev = &levels.last().unwrap().2;
        let fetch = |x: u32, y: u32| prev[(y.min(h + p - 1) * (w + p) + x.min(w + p - 1)) as usize];
        let mut data = Vec::with_capacity(((nw + p) * (nh + p)) as usize);
        for y in 0..nh + p {
            for x in 0..nw + p {
                let (a, b, c, d) = (fetch(2*x, 2*y), fetch(2*x+1, 2*y), fetch(2*x, 2*y+1), fetch(2*x+1, 2*y+1));
                if precision == 3 {
                    // WARP 10.0.26100's R16_UNORM GenerateMips: stored integer mean, (sum + 2) >> 2.
                    data.push(core::array::from_fn(|ch| {
                        let sum: u32 = [a[ch], b[ch], c[ch], d[ch]].into_iter().map(|v| (v * 65535.0).round_ties_even() as u32).sum();
                        ((sum + 2) >> 2) as f32 / 65535.0
                    }));
                    continue;
                }
                // WARP's float GenerateMips shader adds these four samples left to right.
                let avg = core::array::from_fn(|ch| (((a[ch] + b[ch]) + c[ch]) + d[ch]) * 0.25);
                data.push(crate::effects::quantize_to(precision, avg));
            }
        }
        levels.push((nw + p, nh + p, data));
        (w, h) = (nw, nh);
    }
    levels
}

pub(crate) fn clamped_mip_chain(tex: &Texture<'_>) -> Vec<(u32, u32, Vec<u8>)> {
    let (mut w, mut h) = (tex.width, tex.height);
    let mut data = vec![0; ((w + 1) * (h + 1) * 4) as usize];
    for y in 0..=h {
        for x in 0..=w {
            let src = ((y.min(h - 1) * w + x.min(w - 1)) * 4) as usize;
            let dst = ((y * (w + 1) + x) * 4) as usize;
            data[dst..dst + 4].copy_from_slice(&tex.data[src..src + 4]);
        }
    }
    let mut levels = vec![(w + 1, h + 1, data)];
    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let data = &levels.last().unwrap().2;
        let mut next = vec![0; ((nw + 1) * (nh + 1) * 4) as usize];
        for y in 0..=nh {
            for x in 0..=nw {
                for ch in 0..4 {
                    let p = |x: u32, y: u32| data[((y.min(h) * (w + 1) + x.min(w)) * 4 + ch) as usize] as f32;
                    let avg = ((p(2 * x, 2 * y) + p(2 * x + 1, 2 * y)) + (p(2 * x, 2 * y + 1) + p(2 * x + 1, 2 * y + 1))) * 0.25;
                    next[((y * (nw + 1) + x) * 4 + ch) as usize] = avg.round_ties_even() as u8;
                }
            }
        }
        levels.push((nw + 1, nh + 1, next));
        (w, h) = (nw, nh);
    }
    levels
}

/// `DrawBitmap` under a rotation/shear transform (`bitmaps-09`): the axis-aligned [`blit`] can't
/// express a rotated quad, so rasterize the transformed dest rectangle with WARP's per-primitive 8×8
/// analytic coverage ([`crate::pixels::rasterize_edges`], the same coverage scan converter the
/// geometry fills use) and sample the source through the device→source-texel affine plane, evaluated
/// with [`crate::warp::plane::eval`] (the same attribute-plane interpolation the radial brush / effect
/// sampler use). `corners` are the device-space images of the dest rect's TL,TR,BR,BL; `source` maps
/// to them linearly. The anti-aliased quad edge is the coverage (confirmed byte-exact against the WARP
/// oracle — WARP AAs the DrawBitmap primitive boundary as a separate pass; the sampler clamps taps, so
/// the transparency is coverage, not a footprint straddle). Interior pixels sample the byte-exact
/// integer bilinear ([`crate::sampling`] linear) which the anisotropic footprint reduces to on
/// magnification. Minification uses a mip chain and WARP's anisotropic probe line; the remaining
/// coordinate and edge-coverage differences are small residuals.
pub fn draw_bitmap_affine(
    target: &mut PixelBuffer,
    bitmap: &Bitmap,
    corners: [[f32; 2]; 4],
    source: [f32; 4],
    opacity: f32,
    interpolation: u32,
) {
    draw_bitmap_affine_precision(target, bitmap, corners, source, opacity, interpolation, 0, None, crate::pixels::Aa::PerPrimitive, crate::pixels::Blend::SourceOver, None);
}

pub(crate) fn draw_bitmap_affine_precision(
    target: &mut PixelBuffer, bitmap: &Bitmap, corners: [[f32; 2]; 4], source: [f32; 4],
    opacity: f32, interpolation: u32, precision: u32, source_transform: Option<[f32; 6]>, aa: crate::pixels::Aa,
    blend: crate::pixels::Blend, vs_row: Option<[f32; 6]>,
) {
    let source = norm_source(source, bitmap.width as f32, bitmap.height as f32);
    draw_sampled_bitmap_affine(target, bitmap, corners, source, opacity, interpolation, precision, source_transform, aa, blend, vs_row);
}

fn realize_hq_axis(bitmap: &Bitmap, origin: [i32; 2], atlas: bool, axis: usize, scale: f32, out: [i32; 4], precision: u32) -> Bitmap {
    let size = [out[2] - out[0], out[3] - out[1]];
    let dims = [bitmap.width, bitmap.height].map(|v| if atlas { v.next_power_of_two().max(512) } else { v });
    let planes = [
        [[0,0],[size[0],0],[size[0],size[1]]],
        [[0,size[1]],[0,0],[size[0],size[1]]],
    ].map(|p| {
        let values = p.map(|p| {
            let scene = [(p[0]+out[0]) as f32,(p[1]+out[1]) as f32];
            [scene[0],scene[1],(scene[0]-origin[0] as f32)/dims[0] as f32,(scene[1]-origin[1] as f32)/dims[1] as f32]
        });
        core::array::from_fn::<_,4,_>(|ch| crate::warp::plane::setup_triangle_attr(p.map(|p| p[0] as i64*256),p.map(|p| p[1] as i64*256),values.map(|p| p[ch])))
    });
    crate::effects::builtin::transform::to_bitmap_precision(size[0] as u32,size[1] as u32,0.0,0.0,precision>1,|x,y| {
        let triangle = usize::from(y*size[0] as f32 > x*size[1] as f32);
        let v = planes[triangle].map(|p| p[0]*x+(p[1]*y+p[2]));
        let s = crate::sampling::hq_downscale_axis(v[axis]/scale,v[axis],v[axis+2],1.0/dims[axis] as f32,scale,0.0,!crate::float::is_highp(bitmap.format),|u| {
            let mut uv = [v[2],v[3]]; uv[axis] = u;
            let at = core::array::from_fn::<_,2,_>(|c| ((uv[c]*dims[c] as f32*256.0).floor() as i32)>>8);
            let x = at[0].clamp(0,bitmap.width as i32-1); let y = at[1].clamp(0,bitmap.height as i32-1);
            if crate::float::is_highp(bitmap.format) { bitmap.texel_rgba(x,y) } else {
                let i = (y as usize*bitmap.width as usize+x as usize)*4;
                [2,1,0,3].map(|ch| crate::sampling::norm_v128(bitmap.premul_bgra[i+ch] as i32*128))
            }
        });
        crate::effects::quantize_to(precision,s)
    })
}

// Native DrawBitmap HQ DDI: unit-axis footprint, Y then X prefilter, then residual affine cubic.
/// The scene position the HQ pass's pixel shader reads at device point `p`: the attribute interpolated over the slot rect
/// `visible` drawn as d2d1's two triangles (W,0),(0,0),(W,H) and (0,H),(0,0),(W,H), each plane fitted by WARP at its
/// left-most vertex, which lands an ulp off the pixel centre (VM: the HQ kernel's phase at a skewed DrawBitmap's pixel).
fn slot_scene(visible: [f32; 4], p: [f32; 2]) -> [f32; 2] {
    let (w, h) = (visible[2] - visible[0], visible[3] - visible[1]);
    let q = [p[0] - visible[0], p[1] - visible[1]];
    let mut tri: [[f32; 2]; 3] = if q[0] * h >= q[1] * w { [[w, 0.0], [0.0, 0.0], [w, h]] } else { [[0.0, h], [0.0, 0.0], [w, h]] };
    // Fitted at its left-most vertex (the first on a tie), the winding kept, as WARP's setup does.
    let first = (0..3).min_by(|&i, &j| tri[i][0].total_cmp(&tri[j][0])).unwrap_or(0);
    tri.rotate_left(first);
    let sx = tri.map(|v| (v[0] * 256.0).round_ties_even() as i64);
    let sy = tri.map(|v| (v[1] * 256.0).round_ties_even() as i64);
    let ax = crate::warp::plane::setup_triangle_attr(sx, sy, tri.map(|v| v[0] + visible[0]));
    let ay = crate::warp::plane::setup_triangle_attr(sx, sy, tri.map(|v| v[1] + visible[1]));
    [ax[0] * q[0] + (ax[1] * q[1] + ax[2]), ay[0] * q[0] + (ay[1] * q[1] + ay[2])]
}

fn prepare_rotated_hq(bitmap: &Bitmap, matrix: [f32; 6], bounds: [f32; 4], precision: u32) -> Option<(Bitmap, [i32; 2], [f32; 6])> {
    let scales: [f32; 2] = core::array::from_fn(|i| {
        let dx = (matrix[2*i] + matrix[4]) - matrix[4];
        let dy = (matrix[2*i+1] + matrix[5]) - matrix[5];
        crate::warpmath::sqrt(dx * dx + dy * dy).max(crate::sampling::HQ_MIN_SCALE)
    });
    if !scales.iter().any(|&v| v > 0.0 && v < 1.0) { return None; }
    let mut residual = matrix;
    for i in 0..2 { if scales[i] > 0.0 && scales[i] < 1.0 {
        let r = 1.0 / scales[i]; residual[2*i] *= r; residual[2*i+1] *= r;
    }}
    let [a,b,c,d,e,f] = residual;
    let det = a*d-b*c;
    let points = [[bounds[0]+0.5,bounds[1]+0.5],[bounds[2]-0.5,bounds[1]+0.5],[bounds[2]-0.5,bounds[3]-0.5],[bounds[0]+0.5,bounds[3]-0.5]].map(|p| {
        [((p[0]*d-p[1]*c)+(c*f-d*e))/det, ((-p[0]*b+p[1]*a)+(b*e-a*f))/det]
    });
    let rect = [
        ((points.iter().map(|p| p[0]).fold(f32::INFINITY,f32::min)-0.5).floor() as i32).saturating_sub(1),
        ((points.iter().map(|p| p[1]).fold(f32::INFINITY,f32::min)-0.5).floor() as i32).saturating_sub(1),
        ((points.iter().map(|p| p[0]).fold(f32::NEG_INFINITY,f32::max)-0.5).ceil() as i32).saturating_add(2),
        ((points.iter().map(|p| p[1]).fold(f32::NEG_INFINITY,f32::max)-0.5).ceil() as i32).saturating_add(2),
    ];
    let mut first = rect;
    if scales[0] < 1.0 {
        let taps = ((2.0/scales[0]) as i32).saturating_add(3)&!3;
        first[0] = (((rect[0] as f32+0.5)/scales[0]-0.5).floor() as i32).saturating_sub(taps.saturating_sub(1));
        first[2] = (((rect[2] as f32-0.5)/scales[0]-0.5).floor() as i32).saturating_add(taps).saturating_add(1);
    }
    let precision = if precision == 0 { if crate::float::is_highp(bitmap.format) {5} else {1} } else {precision};
    let mut intermediate = bitmap.clone();
    let mut origin = [0,0];
    if scales[1] < 1.0 {
        intermediate = realize_hq_axis(bitmap,[0,0],false,1,scales[1],first,precision);
        origin = [first[0],first[1]];
    }
    if scales[0] < 1.0 {
        intermediate = realize_hq_axis(&intermediate,origin,scales[1]<1.0,0,scales[0],rect,precision);
        origin = [rect[0],rect[1]];
    }
    Some((intermediate,origin,residual))
}

// System32 rotated DrawBitmap captures share brush kernels and quantize UNORM samples before edge coverage.
fn draw_sampled_bitmap_affine(
    target: &mut PixelBuffer, bitmap: &Bitmap, corners: [[f32; 2]; 4], source: [f32; 4],
    opacity: f32, interpolation: u32, precision: u32, source_transform: Option<[f32; 6]>, aa: crate::pixels::Aa,
    blend: crate::pixels::Blend, vs_row: Option<[f32; 6]>,
) {
    let a = (corners[1][0] - corners[0][0]) / (source[2] - source[0]);
    let b = (corners[1][1] - corners[0][1]) / (source[2] - source[0]);
    let c = (corners[3][0] - corners[0][0]) / (source[3] - source[1]);
    let d = (corners[3][1] - corners[0][1]) / (source[3] - source[1]);
    let matrix = source_transform.unwrap_or([a, b, c, d, (corners[0][0] - a * source[0]) - c * source[1], (corners[0][1] - b * source[0]) - d * source[1]]);
    if matrix.iter().any(|v| !v.is_finite()) || a * d - b * c == 0.0 { return; }
    let snapped = corners.map(|p| p.map(|v| (v as f64*256.0).round_ties_even()));
    let area = (snapped[1][0]-snapped[0][0])*(snapped[3][1]-snapped[0][1])-(snapped[3][0]-snapped[0][0])*(snapped[1][1]-snapped[0][1]);
    if area == 0.0 { return; }
    let bounds = [
        corners.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min).floor(),
        corners.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min).floor(),
        corners.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max).ceil(),
        corners.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max).ceil(),
    ];
    let visible = [bounds[0].max(0.0), bounds[1].max(0.0), bounds[2].min(target.desc.width as f32), bounds[3].min(target.desc.height as f32)];
    if visible[2] <= visible[0] || visible[3] <= visible[1] { return; }
    let mut brush = crate::brush::Brush::bitmap_source(bitmap.clone(), &matrix, [0; 2], interpolation, opacity);
    if let Some(row) = vs_row { brush.set_vs_row(row); }
    brush.set_intermediate_precision(precision);
    if interpolation <= 1 && !crate::float::is_highp(bitmap.format) {
        if let crate::brush::Brush::Bitmap { to_source, .. } = &mut brush { *to_source = crate::warp::plane::warp_fit_affine(to_source); }
    }
    // Under a COPY primitive blend the quad is a COPY textured mesh from the edge walk, as a cropped DrawImage under
    // COPY is, and the target lerps toward the sample by the walk's coverage (VM probes of rotated and sheared
    // DrawBitmaps, every interpolation mode; the fill kernel rounds the opacity-scaled sample once, then lerps). A
    // NEAREST/LINEAR triangle samples through its own texcoord plane.
    let copy = blend == crate::pixels::Blend::Copy && aa == crate::pixels::Aa::PerPrimitive;
    let per_triangle = (copy && interpolation <= 3 && !crate::float::is_highp(bitmap.format)).then(|| brush.clone());
    let prepared = brush.prepare_rect(bounds);
    let brush = prepared.as_ref().unwrap_or(&brush);
    let hq = (interpolation == 5).then(|| prepare_rotated_hq(bitmap, matrix, visible, precision)).flatten();
    let hq_input = hq.as_ref().map(|(bitmap, origin, _)| {
        let input = crate::effects::builtin::transform::Input::new(bitmap, origin[0], origin[1]).in_atlas();
        if crate::float::is_highp(target.desc.format) { input.with_float_output() } else { input }
    });
    let (w, h) = (target.desc.width as usize, target.desc.height as usize);
    let Some(mut polygon) = crate::pixels::clip_huge(&[corners.to_vec()]).and_then(|f| f.first().cloned()) else { return };
    if aa == crate::pixels::Aa::Aliased {
        // The aliased quad's vertices snap to 28.4 (`ceil(16·v − 8)`) before the centre test (r5 5608).
        for p in &mut polygon { *p = p.map(|v| ((16.0 * v - 8.0).ceil() + 8.0) / 16.0); }
    }
    let float_target = crate::float::is_highp(target.desc.format);
    // The premultiplied BGRA sample (0..1) at pixel (`x`, `y`).
    let sample_at = |x: usize, y: usize| -> [f32; 4] {
        let p = [x as f32 + 0.5, y as f32 + 0.5];
        // Native HQ/ANISO intermediate triangles start at physical atlas (0,0), retaining world UVs.
        let origin = if hq.is_some() { [visible[0], visible[1]] } else if interpolation >= 4 { [bounds[0], bounds[1]] } else { [0.0; 2] };
        if let (Some((bitmap, texture_origin, m)), Some(input)) = (&hq, &hq_input) {
            let dims = [bitmap.width,bitmap.height].map(|v| v.next_power_of_two().max(512));
            let planes = crate::brush::bitmap_quad_at(m,dims,*texture_origin,visible,p[0],p[1],origin);
            let uv = planes.map(|a| a[0]*(p[0]-origin[0])+(a[1]*(p[1]-origin[1])+a[2]));
            let det = m[0]*m[3]-m[1]*m[2];
            let q = slot_scene(visible, p);
            let phase = [((m[3]*q[0]-m[2]*q[1])+(m[2]*m[5]-m[3]*m[4]))/det,
                ((-m[1]*q[0]+m[0]*q[1])+(m[1]*m[4]-m[0]*m[5]))/det];
            let s = input.cubic(phase,uv,crate::effects::builtin::transform::Border::Hard);
            // An 8-bit cubic pass renders into its own atlas at full opacity, and that stored texel is filled onto the
            // target scaled by the float opacity and rounded (VM DDI captures and atlas readbacks of r27 32522's sheared
            // magnification: the atlas is byte-exact, the opacity applied in float before the store is ±2).
            if float_target || crate::float::is_highp(bitmap.format) { [s[2],s[1],s[0],s[3]].map(|v| v*opacity) }
            else { [s[2],s[1],s[0],s[3]].map(|v| ((v.clamp(0.0,1.0)*255.0).round_ties_even()*opacity).round_ties_even()/255.0) }
        } else if crate::float::is_highp(bitmap.format) || float_target { brush.source_at_f_rect_origin(p[0], p[1], bounds, origin) }
            else { brush.source_at_rect_origin(p[0], p[1], bounds, origin).map(|v| v / 255.0) }
    };
    // The stored sample (0..255): an opacity below 1 multiplies the LINEAR sample in float and the store rounds; at full
    // opacity it truncates (r5 5608).
    let store = |v: f32| if interpolation == 1 && opacity >= 1.0 && !crate::float::is_highp(bitmap.format) { v.floor() } else { v.round_ties_even() };
    if copy && float_target {
        // A float target lerps in float by the walk's coverage, unquantized.
        let vp = [0.0, 0.0, w as f32, h as f32];
        let mut ft = crate::float::FloatTarget::new(target);
        crate::pixels::copy_tess::fill_copy(&[polygon.as_slice()], false, w, h, true, |x, y, cov, tri| {
            let c = match cov { crate::pixels::copy_tess::Cov::Area(n) => n.min(64) as f32 / 64.0, crate::pixels::copy_tess::Cov::Trapezoid(c) => c.clamp(0.0, 1.0) };
            let s = match &per_triangle {
                Some(raw) => crate::brush::triangle_brush_sampler(tri, raw, false, vp)(x, y).map(|v| v / 255.0),
                None => sample_at(x, y),
            };
            let d = ft.load_px(x, y);
            let src = [s[2], s[1], s[0], s[3]];
            ft.store_px(x, y, core::array::from_fn(|i| src[i] * c + d[i] * (1.0 - c)));
        });
        return;
    }
    if copy {
        let (stride, vp) = (target.stride, [0.0, 0.0, w as f32, h as f32]);
        crate::pixels::copy_tess::fill_copy(&[polygon.as_slice()], false, w, h, true, |x, y, cov, tri| {
            let s = match &per_triangle {
                Some(raw) => crate::brush::triangle_brush_sampler(tri, raw, false, vp)(x, y),
                None => sample_at(x, y).map(|v| v.clamp(0.0, 1.0) * 255.0),
            };
            let src = s.map(|v| store(v).clamp(0.0, 255.0) as u8);
            let i = y * stride + x * 4;
            let out = cov.lerp(target.data[i..i + 4].try_into().unwrap(), src);
            target.data[i..i + 4].copy_from_slice(&out);
        });
        return;
    }
    let edges = crate::pixels::figure_edges(&[polygon]);
    crate::pixels::rasterize_edges(&edges, crate::pixels::FillMode::Winding, aa, w, h, |x, y, cov| {
        let sample = sample_at(x, y);
        if float_target {
            crate::float::FloatTarget::new(target).blend_px(x, y, [sample[2], sample[1], sample[0], sample[3]], cov, blend);
        } else {
            let i = y * target.stride + x * 4;
            let cov8 = (cov.min(64) as f32 * 255.0 / 64.0).round_ties_even() as i32;
            let so = sample.map(|v| store(v.clamp(0.0, 1.0) * 255.0));
            let dst = [target.data[i], target.data[i + 1], target.data[i + 2], target.data[i + 3]];
            let out = if matches!(blend, crate::pixels::Blend::Min | crate::pixels::Blend::Add | crate::pixels::Blend::Max) {
                // MIN/ADD/MAX take the geometry brush fill's blend of the stored sample under the scan's coverage.
                crate::brush::blend_src(dst, so, cov8 as f32 / 255.0, blend)
            } else {
                composite(so.map(|v| { let v = v as i32 * cov8; (v + (v >> 8) + 128) >> 8 }), dst, Composite::SourceOver)
            };
            target.data[i..i + 4].copy_from_slice(&out);
        }
    });
}

/// Source-over for a float shader sample on a UNORM8 target. Quantize the source before
/// WARP's integer destination blend; opacity and fractional coverage multiply the sample first.
pub(crate) fn over_premul_f(target: &mut PixelBuffer, i: usize, s: [f32; 4], m: f32) {
    let src: [i32; 4] = core::array::from_fn(|c| ((s[c] * m).clamp(0.0, 1.0) * 255.0).round_ties_even() as i32);
    let ia = 255 - src[3];
    for c in 0..4 {
        let t = target.data[i + c] as i32 * ia;
        target.data[i + c] = (((src[c] << 8) + t + (t >> 8) + 128) >> 8).min(255) as u8;
    }
}

/// PLUS of one premultiplied float sample: `dst + rne(s·255)`, saturating.
pub(crate) fn add_premul_f(target: &mut PixelBuffer, i: usize, s: [f32; 4]) {
    for (d, v) in target.data[i..i + 4].iter_mut().zip(s) {
        *d = (*d as i32 + (v.clamp(0.0, 1.0) * 255.0).round_ties_even() as i32).min(255) as u8;
    }
}

/// `DrawImage` of a bitmap: draw at `offset` (device space), whole-bitmap source, under
/// `composite`. SOURCE_COPY replaces the entire target — transparent black outside the image —
/// so it clears the rest; the others leave the target untouched outside the image.
pub fn draw_image(target: &mut PixelBuffer, bitmap: &Bitmap, offset: [f32; 2], interpolation: u32, composite: Composite) {
    // A float bitmap lands as its texels rounded to 8 bits, then the integer composite (fbitmap-04).
    let decoded;
    let bitmap = if crate::float::is_highp(bitmap.format) {
        decoded = crate::effects::float_input_to_bgra8(bitmap);
        &decoded
    } else {
        bitmap
    };
    if composite == Composite::SourceCopy {
        target.data.fill(0);
    }
    let dest = [offset[0], offset[1], offset[0] + bitmap.width as f32, offset[1] + bitmap.height as f32];
    let source = [0.0, 0.0, bitmap.width as f32, bitmap.height as f32];
    blit(target, bitmap, dest, source, 1.0, filter_of(interpolation), composite, Extend::Clamp, true, crate::pixels::Blend::SourceOver, crate::pixels::Aa::PerPrimitive, None, None);
}

/// LINEAR magnify blit with a TRANSPARENT border (OOB texels = 0) — the SOFT effect-output magnify
/// (`draw_image_effect`). `dest` is the DEVICE rect, `source` the texel sub-rect (may extend past the
/// bitmap by the resample reach so the edge ramps to transparent, matching native's DrawImage of a
/// blur/effect output). SOURCE_OVER, opacity 1.
pub fn draw_bitmap_transparent(target: &mut PixelBuffer, bitmap: &Bitmap, dest: [f32; 4], source: [f32; 4]) {
    blit(target, bitmap, dest, source, 1.0, Filter::Linear, Composite::SourceOver, Extend::Transparent, false, crate::pixels::Blend::SourceOver, crate::pixels::Aa::PerPrimitive, None, None);
}

// DrawBitmap/DrawImage sample the source rect with clamp addressing; the brush extend modes
// (wrap/mirror) are 1D's, which call the sampler directly. Fractional `dest` edges get the
// axis-aligned AA-rectangle coverage: each pixel's premultiplied source is scaled by the fraction
// of the pixel the rect covers on each axis (their product). The texcoord is still evaluated at the
// pixel centre, matching WARP's attribute interpolation; integer-aligned rects get coverage 1.
#[allow(clippy::too_many_arguments)]
fn blit(
    target: &mut PixelBuffer,
    bitmap: &Bitmap,
    dest: [f32; 4],
    source: [f32; 4],
    opacity: f32,
    filter: Filter,
    mode: Composite,
    ex: Extend,
    // NEAREST edge rule: DrawImage point-samples at the pixel centre (full/none, no fractional edge AA
    // — VM di-* probes), while DrawBitmap anti-aliases the fractional rect edge (11-bitmap-nearest,
    // sampling-01/02). Only affects the NEAREST fractional-edge pixels; LINEAR ignores it.
    point: bool,
    // The primitive blend (`D2D1_PRIMITIVE_BLEND`): COPY/MIN/ADD/MAX apply the opacity-scaled sample at
    // the edge coverage. COPY rounds its two terms apart, `src·cov + dst·(255 − cov)` each `/255`
    // (r5 5762's DrawBitmap edges); MIN/ADD/MAX take a geometry fill's form.
    blend: crate::pixels::Blend,
    // ALIASED: the rect's edges round to 24.8 (half up) and a pixel is whole iff its centre is inside
    // `[l, r) × [t, b)` (r5 5672; 21 aliased probes, ties and 1/10000-px edges included; the C API
    // fuzz's 10.5123/63.5214 edges, which 28.4 rounds onto the pixel centres).
    aa: crate::pixels::Aa,
    // DrawBitmap's own source mapping, each axis's low edge held at >= 0 and high edge at <= the size apart (a source
    // wholly past an edge keeps its reversed span): what decides the NEAREST plane below. `None`: `source` itself.
    native_source: Option<[f32; 4]>,
    // The device→normalized-texcoord plane (`[du/dx, dv/dx, du/dy, dv/dy, u0, v0]`) of the bitmap brush a LINEAR
    // DrawBitmap fills its dest with off this alpha-blit path; `None` keeps the float iterator.
    brush_plane: Option<[f32; 6]>,
) {
    let [dl, dt, dr, db] = dest;
    let snap = |v: f32| (256.0 * v + 0.5).floor() / 256.0;
    let aliased = (aa == crate::pixels::Aa::Aliased).then(|| [snap(dl), snap(dt), snap(dr), snap(db)]);
    let (dw, dh) = (dr - dl, db - dt);
    if !(dw > 0.0 && dh > 0.0) {
        return;
    }
    let (w, h) = (target.desc.width as i32, target.desc.height as i32);
    // Past |524287| native leaves this alpha-blit rect path
    // and fills the dest as rectangle geometry with the bitmap brush: the strip clip and 8×8 scan of
    // [`crate::pixels::clip_huge`], which can leave a slope-1 sliver or nothing (w_bm dbl/dbx).
    let huge = if dest.iter().any(|v| v.abs() > 524287.0) {
        let rect = [vec![[dl, dt], [dr, dt], [dr, db], [dl, db]]];
        let Some(figures) = crate::pixels::clip_huge(&rect) else { return };
        // Native fills the huge dest as the textured rectangle's geometry. A COPY draw writes the shape's
        // EXTERIOR, so it takes the geometry trapezoid-fringe AA: an OPAQUE COPY emits the full trapezoid
        // mesh (the cov-0 outer ring + 32/64 diagonal); a translucent/non-opaque COPY takes the same
        // trapezoid-fringe through the faithful edge-walk (`copy_tess`), its sixteenths edge
        // (28.4 floor) — byte-exact on an off-integer edge (18388 left 62.53 reads 7/16, where the 8×8
        // subsample read 0.5; 17023/20620, whose edges land where the two agree, stay exact). SOURCE_OVER
        // never writes the exterior, so it keeps the 8×8 complex-scan. Match that split.
        if blend == crate::pixels::Blend::Copy && opacity >= 1.0 && bitmap.is_fully_opaque() {
            Some(crate::pixels::trapezoid_coverage(&figures, w as usize, h as usize, aa))
        } else if blend == crate::pixels::Blend::Copy {
            let mut cov = vec![0u8; (w * h) as usize];
            let slices: Vec<&[[f32; 2]]> = figures.iter().map(|f| f.as_slice()).collect();
            crate::pixels::copy_tess::fill_copy(&slices, false, w as usize, h as usize, true, |x, y, c, _| {
                cov[y * w as usize + x] = c.c8().clamp(0, 255) as u8;
            });
            Some(cov)
        } else {
            let mut cov = vec![0u8; (w * h) as usize];
            let edges = crate::pixels::figure_edges(&figures);
            crate::pixels::rasterize_edges(&edges, crate::pixels::FillMode::Winding, aa, w as usize, h as usize, |x, y, c| {
                cov[y * w as usize + x] = (c.min(64) as f32 * 255.0 / 64.0).round_ties_even() as u8;
            });
            Some(cov)
        }
    } else {
        None
    };
    // The huge COPY fill routes through native's geometry/textured-FillRectangle path, whose COPY kernel
    // is the single-rounding `copy_tess::lerp8`, not the DrawBitmap blit's two-term `unorm_mul` round.
    let huge_copy_tess = huge.is_some()
        && blend == crate::pixels::Blend::Copy
        && !(opacity >= 1.0 && bitmap.is_fully_opaque());
    let tex = bitmap.texture();
    let op8 = (opacity * 255.0).round_ties_even().clamp(0.0, 255.0) as i32;
    // WARP's rasterizer interpolates the source texcoord as a 16.16 fixed-point accumulation
    // `base + px·step` per axis (`crate::sampling::axis_fixed`); the filtered sampler reads that
    // directly. Nearest keeps the float floor (`Texture::sample`), which the oracle already matches.
    // A dest that starts off the target anchors the DDA at the first visible pixel (the clipped
    // draw's left/top), so the rounded step accumulates from there (18-gaussian-blur's ×6 magnify
    // of a blur output whose bounds start further left).
    let clip_axis = |s0: f32, s1: f32, d0: f32, d1: f32| {
        if d0 < 0.0 {
            let r = (s1 - s0) / (d1 - d0);
            crate::sampling::axis_fixed(s0 + (0.0 - d0) * r, s1, 0.0, d1)
        } else { crate::sampling::axis_fixed(s0, s1, d0, d1) }
    };
    let (base_x, step_x) = clip_axis(source[0], source[2], dl, dr);
    let (base_y, step_y) = clip_axis(source[1], source[3], dt, db);
    // Native's texcoord realization collapses to the extend origin only when the dest span PRODUCT
    // (dr−dl)·(db−dt) overflows f32 to non-finite: that product is the scale in its 2×2 source→dest
    // matrix, so once it is ±inf the inverse determinant takes BOTH axes to the origin texel (coupled —
    // 29567's finite Y pins to row 0, not the source.top=1600 clamp). This is a true overflow at the
    // FLT_MAX scale, not merely a "huge" dest: a VM magnitude sweep collapses at 3.4e38 but samples
    // normally at 1e36 and every ±1e9 corpus dest (whose product stays finite), so gate on the overflow,
    // not on |dest| > 524287. Sample the origin texel across the clip_huge wedge (the 16.16 DDA otherwise
    // wraps the |source|≈1e9 edge to garbage). VM probes fp_lin/fp_nn/vB/vC/mag_* on .8875.
    let huge_origin = (!(dw * dh).is_finite()).then(|| {
        ((crate::brush::texel_index(f32::INFINITY, ex, tex.width as i32)) << 16,
         (crate::brush::texel_index(f32::INFINITY, ex, tex.height as i32)) << 16)
    });
    // WARP iterates the texcoord on its 16.16 DDA only while each axis's start, at the first pixel centre, fits an
    // i32; past that the draw runs on a float plane, whose NEAREST address converts floor(u) with cvtps2dq, so a
    // texcoord out of the i32 range (or NaN) reads INT_MIN, clamped to the first texel (VM probes and the kernel's
    // values at its address: the kernel switches between a source edge of 33280 and 33290, starts 2146962432 and
    // 2147607552; a texcoord near 6e36 reads the first row, one near 5e6 the clamped last row).
    let plane_axis = |s0: f32, s1: f32, d0: f32, d1: f32| {
        let (s0, d0) = if d0 < 0.0 { (s0 + (0.0 - d0) * ((s1 - s0) / (d1 - d0)), 0.0) } else { (s0, d0) };
        let ratio = (s1 as f64 - s0 as f64) / (d1 as f64 - d0 as f64);
        let start = s0 as f64 + (d0.floor() as f64 + 0.5 - d0 as f64) * ratio;
        let fits = (start * 65536.0).abs() < 2147483648.0;
        let du = (s1 - s0) / (d1 - d0);
        (fits, du, s0 - d0 * du)
    };
    let ps = native_source.unwrap_or(source);
    let (px_plane, py_plane) = (plane_axis(ps[0], ps[2], dl, dr), plane_axis(ps[1], ps[3], dt, db));
    let on_plane = huge_origin.is_none() && !(px_plane.0 && py_plane.0);
    let nearest_plane = filter == Filter::Nearest && on_plane;
    // LINEAR's plane carries the Q8 address `u·256 − 128`, rounded to nearest by cvtps2dq, INT_MIN out of range.
    let linear_plane = filter == Filter::Linear && on_plane;
    let plane_q8 = |(_, du, c): (bool, f32, f32), p: f32| -> i32 {
        let q = ((p * du + c) * 256.0 - 128.0).round_ties_even();
        if q >= -2147483648.0 && q < 2147483648.0 { q as i32 } else { i32::MIN }
    };
    let plane_texel = |(_, du, c): (bool, f32, f32), p: f32, n: u32| -> i32 {
        let t = (p * du + c).floor();
        if t >= -2147483648.0 && t < 2147483648.0 { (t as i64).clamp(0, n as i64 - 1) as i32 } else { 0 }
    };
    let (x0, x1, y0, y1) = match huge {
        Some(_) => (0, w, 0, h),
        None => ((dl.floor() as i32).clamp(0, w), (dr.ceil() as i32).clamp(0, w), (dt.floor() as i32).clamp(0, h), (db.ceil() as i32).clamp(0, h)),
    };
    // Native leaves this alpha-blit path for a dest past |524287| or a MIN/ADD/MAX primitive blend and fills
    // the rect with the bitmap brush. Its LINEAR sampler keeps the bilinear in value·128 and scales by the
    // float opacity (r22 27610: all 840 pixels), addressed on the brush's texcoord plane (`brush_plane`; the
    // float iterator without one) rather than the 16.16 DDA (r22 27261's ADD draw). At full opacity the fill
    // kernel has no opacity multiply and stores the bilinear truncated, `>> 7` (JIT dump).
    let brush_path = filter == Filter::Linear && (opacity < 1.0 || brush_plane.is_some())
        && (huge.is_some() || matches!(blend, crate::pixels::Blend::Min | crate::pixels::Blend::Add | crate::pixels::Blend::Max));
    // A NEAREST draw takes the same brush fill there, its texel still the iterator's (VM: a dest past |524287| on
    // either axis, or an ADD primitive blend, at opacities 0.22, 0.37 and 0.61).
    let nearest_brush = filter == Filter::Nearest && opacity < 1.0
        && (huge.is_some() || matches!(blend, crate::pixels::Blend::Min | crate::pixels::Blend::Add | crate::pixels::Blend::Max));
    // Under MIN the brush fill takes the geometry walk's coverage, as a MIN geometry fill does, kept as the kernel's float.
    let min_cov: Option<Vec<f32>> = (brush_path && brush_plane.is_some() && blend == crate::pixels::Blend::Min).then(|| {
        let mut c = vec![0.0f32; (w * h) as usize];
        let rect = [vec![[dl, dt], [dr, dt], [dr, db], [dl, db]]];
        let Some(figures) = crate::pixels::clip_huge(&rect) else { return c };
        let slices: Vec<&[[f32; 2]]> = figures.iter().map(|f| f.as_slice()).collect();
        crate::pixels::copy_tess::fill_copy(&slices, false, w as usize, h as usize, true, |x, y, v, _| {
            c[y * w as usize + x] = match v { crate::pixels::copy_tess::Cov::Area(n) => n.min(64) as f32 / 64.0, crate::pixels::copy_tess::Cov::Trapezoid(t) => t.clamp(0.0, 1.0) };
        });
        c
    });
    let float_q: Option<Vec<(i32, i32)>> = (brush_path && brush_plane.is_none() && huge.is_none() && x1 > x0 && y1 > y0).then(|| {
        let (sw, sh) = (source[2] - source[0], source[3] - source[1]);
        let ts = [sw / dw, 0.0, 0.0, sh / dh, source[0] - dl * (sw / dw), source[1] - dt * (sh / dh)];
        let n = (x1 - x0) as usize;
        let mut q = vec![(0, 0); n * (y1 - y0) as usize];
        crate::sampling::alphablt_float_iter(&ts, [bitmap.width, bitmap.height], [x0 as usize, y0 as usize, x1 as usize, y1 as usize], h as u32, |x, y, qx, qy| {
            q[(y - y0 as usize) * n + (x - x0 as usize)] = (qx, qy);
        });
        q
    });
    for py in y0..y1 {
        let cy = py as f32 + 0.5;
        let cov_y = (db.min(py as f32 + 1.0) - dt.max(py as f32)).clamp(0.0, 1.0);
        if cov_y <= 0.0 && huge.is_none() {
            continue;
        }
        // WARP's 16.16 texcoord DDA is i32 and wraps on overflow (imul/add); match it explicitly so a
        // huge/degenerate dest (r21 26668) wraps instead of panicking in a debug build.
        let uy = match huge_origin { Some((_, oy)) => oy, None => base_y.wrapping_add(py.wrapping_mul(step_y)) };
        let ty = source[1] + (cy - dt) / dh * (source[3] - source[1]);
        for px in x0..x1 {
            let cx = px as f32 + 0.5;
            let cov_x = (dr.min(px as f32 + 1.0) - dl.max(px as f32)).clamp(0.0, 1.0);
            if cov_x <= 0.0 && huge.is_none() {
                continue;
            }
            // 8-bit AA coverage of the axis-aligned rect (1.0 for an integer-aligned draw).
            let cov8 = if let Some(cov) = &huge {
                match cov[(py * w + px) as usize] {
                    0 => continue,
                    c => c as i32,
                }
            } else if let Some([sl, st, sr, sb]) = aliased {
                if cx >= sl && cx < sr && cy >= st && cy < sb { 255 } else { 0 }
            } else if filter == Filter::Nearest && point {
                // DrawImage NEAREST point sample: full iff the centre is (strictly) inside the rect.
                if cx > dl && cx < dr && cy > dt && cy < db { 255 } else { 0 }
            } else if matches!(filter, Filter::Nearest | Filter::Linear) {
                // DrawBitmap draws the rect as two triangles through WARP's 8×-supersampled rasterizer,
                // so each axis' edge coverage is the count of the 8 sub-samples (at `p + (k+0.5)/8`,
                // k = 0..8) inside the half-open device span `[lo, hi)`, and the pixel coverage is their
                // product over 64. Both sub-samples and edges are evaluated in 8.8 fixed-point (×256):
                // the sub-sample is `p·256 + 32k + 16`, the edges `round(·256)`. The fixed grid avoids
                // the float instability when a device edge lands right on a sub-sample — e.g. a bottom
                // edge at 44.1875410 (sampling-02) whose f32 is 44.1875076, just ABOVE sub-sample
                // 44.1875, which float `s < hi` wrongly includes (2/8 vs native 1/8); ×256 snaps it to
                // 44.1875 and excludes it. 1/16 was too coarse (mis-snapped a 5.2 top edge to 5.1875).
                // Byte-exact against clean probes at dpi 96/120/192 (top 2.6→3/8, left 3.3→6/8, right
                // 15.3→2/8, dpi-192 top 5.2→6/8, sampling-02 bottom 44.1875→1/8).
                let sub = |lo: f32, hi: f32, p: i32| {
                    let (lo8, hi8) = ((lo * 256.0).round_ties_even() as i32, (hi * 256.0).round_ties_even() as i32);
                    (0..8).filter(|k| { let s = p * 256 + 32 * k + 16; s >= lo8 && s < hi8 }).count() as i32
                };
                (sub(dl, dr, px) * sub(dt, db, py) * 255 + 32) / 64
            } else {
                (cov_x * cov_y * 255.0).round_ties_even() as i32
            };
            let ux = match huge_origin { Some((ox, _)) => ox, None => base_x.wrapping_add(px.wrapping_mul(step_x)) };
            // Sample → UNORM8, then the fixed-point colour-texturing pipeline: × opacity, × coverage.
            // Nearest and linear read WARP's rasterized 16.16 texcoord DDA (`base + px·step`) directly,
            // so the texel is bit-exact to native at any device scale (a per-pixel float `floor(u·dim)`
            // picked a neighbouring texel at a texel boundary under dpi 120 = ×1.25); cubic samples from
            // the float texel-space centre (it re-quantises to its own 8-bit fraction).
            let sample = match filter {
                Filter::Linear if linear_plane => tex.sample_q8([ex; 2], plane_q8(px_plane, cx), plane_q8(py_plane, cy)),
                Filter::Linear => tex.sample_fixed([ex; 2], ux, uy),
                Filter::Nearest if nearest_plane => {
                    let (ix, iy) = (plane_texel(px_plane, cx, tex.width), plane_texel(py_plane, cy, tex.height));
                    tex.sample_fixed_nearest([ex; 2], (ix << 16) - 0x7f80, (iy << 16) - 0x7f80)
                }
                Filter::Nearest => tex.sample_fixed_nearest([ex; 2], ux, uy),
                _ => {
                    let tx = source[0] + (cx - dl) / dw * (source[2] - source[0]);
                    tex.sample(filter, [ex; 2], tx, ty)
                }
            };
            let s = sample.map(|v| (v * 255.0).round_ties_even() as i32);
            // The brush fill's normalized sample (opacity in), which its MIN kernel composites over white in float.
            let mut fill_f: Option<[f32; 4]> = None;
            let so: [i32; 4] = if brush_path {
                let (qx, qy) = match (&float_q, brush_plane) {
                    // WARP's brush fill kernel evaluates the plane at the pixel centre, `x·du/dx + (y·du/dy + u0)`
                    // (JIT dump), and addresses it as the bitmap brush does.
                    (_, Some(p)) => {
                        let pow2 = [tex.width.is_power_of_two(), tex.width.is_power_of_two() && tex.height.is_power_of_two()];
                        let u = [p[0] * cx + (p[2] * cy + p[4]), p[1] * cx + (p[3] * cy + p[5])];
                        (crate::brush::bitmap_brush_q8(u[0], tex.width as f32, ex, pow2[0]), crate::brush::bitmap_brush_q8(u[1], tex.height as f32, ex, pow2[1]))
                    }
                    (Some(q), None) => {
                        let (qx, qy) = q[(py - y0) as usize * (x1 - x0) as usize + (px - x0) as usize];
                        ((qx >> 8, qx & 0xff), (qy >> 8, qy & 0xff))
                    }
                    _ => ((ux >> 16, (ux >> 8) & 0xff), (uy >> 16, (uy >> 8) & 0xff)),
                };
                let v = tex.bilinear_v128_all([ex; 2], qx, qy);
                // The fill kernel scales the normalized sample by the opacity before ×255 (JIT dump).
                if brush_plane.is_some() {
                    fill_f = Some(v.map(|v| if opacity >= 1.0 { crate::sampling::norm_v128(v) } else { crate::sampling::norm_v128(v) * opacity }));
                }
                if brush_plane.is_some() && opacity >= 1.0 { v.map(|v| v >> 7) }
                else if brush_plane.is_some() { v.map(|v| ((crate::sampling::norm_v128(v) * opacity).clamp(0.0, 1.0) * 255.0).round_ties_even() as i32) }
                else { v.map(|v| ((crate::sampling::norm_v128(v) * 255.0) * opacity.max(0.0)).round_ties_even() as i32) }
            } else if nearest_brush {
                // The bitmap-brush fill scales the NEAREST texel by the float opacity and rounds.
                s.map(|v| (v as f32 * opacity.max(0.0)).round_ties_even() as i32)
            } else {
                s.map(|v| unorm_mul(v, op8))
            };
            let i = py as usize * target.stride + px as usize * 4;
            let dst = [target.data[i], target.data[i + 1], target.data[i + 2], target.data[i + 3]];
            if blend != crate::pixels::Blend::SourceOver {
                let so = so.map(|v| v.clamp(0, 255) as u8);
                let out = if blend == crate::pixels::Blend::Copy {
                    if huge_copy_tess {
                        crate::pixels::copy_tess::lerp8(dst, so, cov8)
                    } else {
                        core::array::from_fn(|c| (unorm_mul(so[c] as i32, cov8) + unorm_mul(dst[c] as i32, 255 - cov8)).clamp(0, 255) as u8)
                    }
                } else if let (crate::pixels::Blend::Min, Some(n)) = (blend, fill_f) {
                    // MIN: `n·c + (1 − nα·c)` in [0, 1], ×255 rounded once, then the minimum (JIT dumps at opacity 1 and 0.55).
                    let c = min_cov.as_ref().map_or(cov8 as f32 / 255.0, |m| m[(py * w + px) as usize]);
                    let white = 1.0 - n[3] * c;
                    core::array::from_fn(|k| (((n[k] * c + white).clamp(0.0, 1.0) * 255.0).round_ties_even() as u8).min(dst[k]))
                } else {
                    crate::pixels::blend_cov(dst, so, cov8 as f32 / 255.0, blend)
                };
                target.data[i..i + 4].copy_from_slice(&out);
                continue;
            }
            let sc: [i32; 4] = core::array::from_fn(|c| unorm_mul(so[c], cov8));
            target.data[i..i + 4].copy_from_slice(&composite(sc, dst, mode));
        }
    }
}

// --- per-colour arithmetic --------------------------------------------------------------------

/// `D2D1_COLOR_SPACE`.
pub const COLOR_SPACE_CUSTOM: u32 = 0;
pub const COLOR_SPACE_SRGB: u32 = 1;
pub const COLOR_SPACE_SCRGB: u32 = 2;

/// `D2D1ConvertColorSpace`: per-colour sRGB⇄scRGB with alpha passed through. `src == dst`
/// returns the colour unchanged; CUSTOM or any out-of-range space yields transparent black.
pub fn convert_color_space(src: u32, dst: u32, color: [f32; 4]) -> [f32; 4] {
    if src == dst && (src == COLOR_SPACE_SRGB || src == COLOR_SPACE_SCRGB) {
        return color;
    }
    let [r, g, b, a] = color;
    match (src, dst) {
        (COLOR_SPACE_SRGB, COLOR_SPACE_SCRGB) => [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a],
        (COLOR_SPACE_SCRGB, COLOR_SPACE_SRGB) => [linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b), a],
        _ => [0.0, 0.0, 0.0, 0.0],
    }
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { crate::crt::system_crt::powf((c + 0.055) / 1.055, 2.4) }
}
fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * crate::crt::system_crt::powf(c, 1.0 / 2.4) - 0.055 }
}

/// `Premultiply` effect: `out.rgb = in.rgb·in.a`, `out.a = in.a`. Straight → premultiplied.
pub fn premultiply_color([r, g, b, a]: [f32; 4]) -> [f32; 4] {
    [r * a, g * a, b * a, a]
}

/// `UnPremultiply` effect: `out.rgb = a > 0 ? in.rgb/in.a : 0`, `out.a = in.a`.
pub fn unpremultiply_color([r, g, b, a]: [f32; 4]) -> [f32; 4] {
    if a > 0.0 { [r / a, g / a, b / a, a] } else { [0.0, 0.0, 0.0, a] }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::norm_source;

    #[test]
    fn anisotropic_thin_source_rejects_nonfinite_mapping() {
        let rgba: Vec<u8> = (0..21).flat_map(|y| [if y >= 10 { 255 } else { 0 }, 0, 0, 255]).collect();
        let bitmap = Bitmap::from_straight_rgba(1, 21, &rgba);
        let mut target = crate::context::bitmap_target(102, 102, 96.0);
        draw_bitmap_affine(
            &mut target, &bitmap,
            [[0.0, 0.0], [100.0, 1.0], [100.0, 101.0], [0.0, 100.0]],
            [0.0, 10.0, 1e-38, 20.0], 1.0, 4,
        );
        assert!(target.data.iter().all(|&v| v == 0));
    }

    // Coordinator probe: is the COLR leaf's edge fattening an ss-raster + HQ-cubic 1/ss downscale?
    // Build an ss4 binary red square (left edge at ss column E), downscale 1/4 via interp=5 (HQ-cubic),
    // read device col-12 coverage. Box/geometric gives {64,128,191} at E=49/50/51; the native leaf is
    // {83,140,196}. Also print interp=1 (linear) as the box reference.
    #[test]
    fn hqcubic_downscale_edge() {
        let (sw, sh) = (64u32, 64u32); // 16 device px at ss4
        for interp in [1u32, 5u32] {
            let mut line = format!("interp={interp}: ");
            for e in [49usize, 50, 51] {
                let mut rgba = vec![0u8; (sw * sh) as usize * 4];
                for y in 0..sh as usize {
                    for x in 0..sw as usize {
                        if x >= e {
                            let p = (y * sw as usize + x) * 4;
                            rgba[p] = 255; // R
                            rgba[p + 3] = 255; // A (opaque red)
                        }
                    }
                }
                let bmp = Bitmap::from_straight_rgba(sw, sh, &rgba);
                let (dw, dh) = (16u32, 16u32);
                let mut tgt = PixelBuffer {
                    desc: d2d_types::contracts::ImageDesc { width: dw, height: dh, format: DXGI_FORMAT_B8G8R8A8_UNORM, alpha_mode: D2D1_ALPHA_MODE_PREMULTIPLIED, dpi_x: 96.0, dpi_y: 96.0 },
                    stride: (dw * 4) as usize,
                    data: vec![0u8; (dw * dh * 4) as usize],
                };
                draw_bitmap(&mut tgt, &bmp, [0.0, 0.0, dw as f32, dh as f32], [0.0, 0.0, sw as f32, sh as f32], 1.0, interp);
                // red premul: R==A==cov. read device cols 11..14 at row 8.
                let row = 8usize;
                let cov: Vec<u8> = (11..15).map(|x| tgt.data[(row * dw as usize + x) * 4 + 2]).collect();
                line += &format!("E{e}={cov:?}  ");
            }
            eprintln!("{line}");
        }
        eprintln!("(box/geometric E49/50/51 col12 ~ 191/128/64 reversed by edge dir; native leaf partial ~ 83/140/196)");
    }

    // Native normalizes + clamps the DrawBitmap source rect to [0,dim]: a flipped or out-of-range
    // rect becomes the equivalent in-bounds forward rect (VM-probed; fuzz 20008/20025).
    #[test]
    fn source_rect_normalized_and_clamped() {
        assert_eq!(norm_source([1.0, 2.0, 3.0, 4.0], 4.0, 6.0), [1.0, 2.0, 3.0, 4.0]); // forward in-bounds: unchanged
        assert_eq!(norm_source([3.0, 4.0, 1.0, 2.0], 4.0, 6.0), [1.0, 2.0, 3.0, 4.0]); // flipped both axes
        assert_eq!(norm_source([6.0, 0.0, -2.0, 1.0], 4.0, 6.0), [0.0, 0.0, 4.0, 1.0]); // clamp x to [0,4]
    }
}

#[cfg(test)]
mod fast_path_tests {
    use super::{fast_path_bitmap, FastPathBitmap};

    #[test]
    fn fast_path_bitmap_gate() {
        let f = |src: [f32; 4], ex: [u32; 2], ip: u32| fast_path_bitmap(src, [6, 7], ex, ip, false, true);
        assert_eq!(f([0.0, 0.0, 6.0, 7.0], [1, 2], 1), FastPathBitmap::Copy);
        assert_eq!(f([0.0, 0.0, 6.0, 7.0], [1, 2], 2), FastPathBitmap::Copy);
        assert_eq!(f([0.0, 0.0, 6.0, 7.0], [0, 0], 4), FastPathBitmap::None);
        assert_eq!(f([1.0, 2.0, 5.0, 6.0], [0, 0], 1), FastPathBitmap::Scaled([1.0, 2.0, 5.0, 6.0]));
        assert_eq!(f([1.01, 2.0, 5.0, 6.0], [0, 0], 0), FastPathBitmap::Scaled([1.0, 2.0, 5.0, 6.0]));
        assert_eq!(f([1.1, 2.0, 5.0, 6.0], [0, 0], 1), FastPathBitmap::None);
        assert_eq!(f([1.0, 2.0, 5.0, 6.0], [1, 1], 1), FastPathBitmap::None);
        assert_eq!(f([1.0, 2.0, 5.0, 6.0], [0, 0], 2), FastPathBitmap::None);
        assert_eq!(f([-1.0, 2.0, 5.0, 6.0], [0, 0], 1), FastPathBitmap::None);
        assert_eq!(f([1.5, 2.0, 5.0, 6.0], [0, 0], 1), FastPathBitmap::None);
        assert_eq!(fast_path_bitmap([0.0, 0.0, 6.0, 7.0], [6, 7], [0, 0], 2, true, true), FastPathBitmap::None);
    }
}
