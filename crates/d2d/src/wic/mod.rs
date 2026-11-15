//! WIC decoding, exactly as the native Windows codec does it. Direct2D apps load images through WIC, so exact
//! pixels start here: each decoder frame's native pixel format and bytes must match
//! `IWICBitmapFrameDecode` byte-for-byte (the oracle is captured from native WIC on the reference
//! system; fixtures under `crates/d2d/tests/wic/`).
//!
//! PNG is the dominant Direct2D image format and lossless, so a conformant decoder's samples are
//! bit-identical to WIC's by construction (PNG/inflate are fully specified). We reuse the `png`
//! crate for inflate + unfilter + de-interlace, then map the native color type / bit depth to the
//! WIC native frame format and byte layout — the one piece that is WIC-specific, read off the VM
//! oracle and confirmed against `windowscodecs` (PNG decoder uses `CopyPixels` straight through).

mod dds;
mod deflate;
mod jpeg;
mod png_encode;
pub use png_encode::{encode_png, encode_png_meta, png_chunk};
mod xmp;
pub use xmp::xmp_packet;
mod jpeg_encode;
mod jxr;
// WIC JPEG-XR *encoder*, native-only: drives the vendored patched jpegxr-pure-rs.
#[cfg(not(target_arch = "wasm32"))]
mod jxr_encode;
mod png_decode;
pub mod color_transform;
pub mod convert;
pub mod quantize;
pub mod dds_encode;
pub mod fliprotate;
pub mod scale;

pub use dds::{decode_dds, DdsFormat, DdsFrame};
pub use jpeg::{decode_jpeg, decode_jpeg_scaled, jpeg_closest_scale};
pub use jpeg_encode::{encode_jpeg, Subsampling};
pub use jxr::{decode_jxr, JxrImage};
#[cfg(not(target_arch = "wasm32"))]
pub use jxr_encode::{encode_jxr, JxrEncodeOptions};
pub use png_decode::{decode_png, png_icc_profile};

mod bmp;

pub use bmp::{decode_bmp, encode_bmp, Decoded};

mod ico;

pub use ico::decode_ico;

mod gif;

pub use gif::{decode_gif, encode_gif, encode_gif_meta, GifGce, GifMeta};

mod tiff;

pub use tiff::{decode_tiff, encode_tiff, encode_tiff_meta, TiffField};

/// A WIC decode/process failure, carrying the exact `WINCODEC_ERR_*` HRESULT the real codec returns
/// so an application sees the same error at the same call. The named codes are the canonical
/// `winerror.h` values (verified against native WIC). `BadImage` is the generic
/// malformed-content code; the more specific variants are used where the native codec returns them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WicError {
    /// `WINCODEC_ERR_BADIMAGE` — the image content is corrupt / not a valid image of this codec.
    BadImage,
    /// `WINCODEC_ERR_BADHEADER` — the file header/signature is malformed (GIF uses this).
    BadHeader,
    /// `WINCODEC_ERR_UNKNOWNIMAGEFORMAT` — signature or pixel-format descriptor unrecognised (BMP).
    UnknownImageFormat,
    /// `WINCODEC_ERR_STREAMREAD` — a required header field could not be read (stream too short).
    StreamRead,
    /// `WINCODEC_ERR_UNSUPPORTEDOPERATION` — a valid image using a feature this decoder does not yet
    /// reproduce (a opend2d gap, not necessarily a native rejection).
    Unsupported,
}

impl WicError {
    /// The `WINCODEC_ERR_*` HRESULT, as WIC returns it (canonical `winerror.h`
    /// values, confirmed against the real codec under Wine for each malformed-input class).
    pub const fn hresult(self) -> u32 {
        match self {
            WicError::UnknownImageFormat => 0x8898_2F07,
            WicError::BadImage => 0x8898_2F60,
            WicError::BadHeader => 0x8898_2F61,
            WicError::StreamRead => 0x8898_2F72,
            WicError::Unsupported => 0x8898_2F81,
        }
    }
}

/// A WIC native pixel format (the `GUID_WICPixelFormat*` a decoder frame reports). Only the formats
/// the PNG decoder produces are listed; the family shares a GUID prefix, so a variant is just its
/// final byte plus a bits-per-pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WicFormat {
    Indexed1,
    Indexed2,
    Indexed4,
    Indexed8,
    BlackWhite,
    Gray2,
    Gray4,
    Gray8,
    Gray16,
    Bgr24,
    Bgra32,
    Cmyk32,
    Rgb48,
    Rgba64,
}

impl WicFormat {
    /// Final byte of the `GUID_WICPixelFormat*` GUID (all share prefix `6fddc324-…-3d77768dc9`).
    const fn guid_suffix(self) -> u8 {
        use WicFormat::*;
        match self {
            Indexed1 => 0x01,
            Indexed2 => 0x02,
            Indexed4 => 0x03,
            Indexed8 => 0x04,
            BlackWhite => 0x05,
            Gray2 => 0x06,
            Gray4 => 0x07,
            Gray8 => 0x08,
            Gray16 => 0x0b,
            Bgr24 => 0x0c,
            Bgra32 => 0x0f,
            Cmyk32 => 0x1c,
            Rgb48 => 0x15,
            Rgba64 => 0x16,
        }
    }

    /// The canonical `GUID_WICPixelFormat*` string, e.g. `6fddc324-4e03-4bfe-b185-3d77768dc90c`.
    pub fn guid(self) -> String {
        format!("6fddc324-4e03-4bfe-b185-3d77768dc9{:02x}", self.guid_suffix())
    }

    pub const fn bits_per_pixel(self) -> u32 {
        use WicFormat::*;
        match self {
            Indexed1 | BlackWhite => 1,
            Indexed2 | Gray2 => 2,
            Indexed4 | Gray4 => 4,
            Indexed8 | Gray8 => 8,
            Gray16 => 16,
            Bgr24 => 24,
            Bgra32 | Cmyk32 => 32,
            Rgb48 => 48,
            Rgba64 => 64,
        }
    }
}

/// A decoded WIC frame: the native pixel format, size, tightly WIC-strided bytes
/// (`stride = ⌈width·bpp/8⌉`) and, for indexed formats, the palette as `0xAARRGGBB` entries
/// (`CopyPalette`), with per-entry alpha from `tRNS` (255 for entries it does not cover).
#[derive(Debug, Clone)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub format: WicFormat,
    pub stride: usize,
    pub pixels: Vec<u8>,
    pub palette: Vec<u32>,
}

impl DecodedImage {
    pub fn stride_for(width: u32, format: WicFormat) -> usize {
        ((width * format.bits_per_pixel() + 7) / 8) as usize
    }
}
