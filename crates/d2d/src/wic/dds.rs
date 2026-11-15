//! DDS → WIC native frames, exactly as WIC's DDS decoder does it: read the header,
//! derive the pixel format, enumerate frames, and unpack BC1/2/3 block rows.
//!
//! WIC's `CopyPixels` path decodes a *very* small set: only `BC1/BC2/BC3` in their **TYPELESS** DXGI
//! form (71/74/77) — format validation accepts exactly those three. Output is 32bppBGRA, or
//! 32bppPBGRA when AlphaMode∈{Premultiplied(2),Opaque(3)}. Legacy FourCC maps DXT1/DXT2/DXT4 → the
//! premultiplied label (PBGRA) and DXT3/DXT5 → BGRA.
//!
//! Everything else errors with `WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT` (0x88982f61) at decoder create,
//! even BC1_UNORM(72), the uncompressed formats, and BC4/5/6H/7. (The pixel-format derivation has dead cases
//! for UNORM/R32G32B32A32_FLOAT/B8G8R8X8 that `CopyPixels` can never reach; BC4–7 are only readable as
//! raw blocks via `IWICDdsFrameDecode::CopyBlocks`, which this does not model.)
//!
//! Each subresource (array slice → mip → depth slice, in that order) is one frame, matching WIC's
//! `GetFrameCount`/`GetFrame`. Block decode reproduces WIC's exact fixed-point arithmetic.

use super::WicError;

/// A DDS output pixel format. Kept local to this module so the shared `WicFormat` (PNG) stays
/// untouched; only the final-byte GUID suffix differs (both are 32bpp).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DdsFormat {
    Bgra32,
    Pbgra32,
}

impl DdsFormat {
    pub fn guid(self) -> String {
        let suffix = match self {
            DdsFormat::Bgra32 => 0x0f,
            DdsFormat::Pbgra32 => 0x10,
        };
        format!("6fddc324-4e03-4bfe-b185-3d77768dc9{suffix:02x}")
    }

    pub const fn bits_per_pixel(self) -> u32 {
        32
    }
}

/// One decoded DDS subresource (no palette for DDS).
#[derive(Debug, Clone)]
pub struct DdsFrame {
    pub width: u32,
    pub height: u32,
    pub format: DdsFormat,
    pub stride: usize,
    pub pixels: Vec<u8>,
}

// --- header parse (DdsDecodeParameters + GetDXGIFormat) ------------------------------------------

/// Which block codec decodes a frame's bytes.
#[derive(Clone, Copy)]
enum Codec {
    Bc1,
    Bc2,
    Bc3,
}

struct Mapping {
    format: DdsFormat,
    codec: Codec,
    /// Input bytes per 4×4 block.
    block_bytes: u32,
}

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    mip_levels: u32,
    array_size: u32,
    /// 1=1D, 2=2D, 3=3D (WICDdsDimension; cube is a 2D array of 6).
    dimension: u32,
    map: Mapping,
    data_offset: usize,
}

fn rd_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// Format validation + pixel-format derivation: WIC accepts only the TYPELESS BC1/BC2/BC3 DXGI formats
/// (71/74/77); anything else — including the UNORM variants and uncompressed formats — is rejected.
fn derive(dxgi: u32, alpha_mode: u32) -> Result<Mapping, WicError> {
    // 32bppPBGRA iff AlphaMode is Premultiplied(2) or Opaque(3), else 32bppBGRA (a WIC quirk).
    let format = if alpha_mode.wrapping_sub(2) <= 1 { DdsFormat::Pbgra32 } else { DdsFormat::Bgra32 };
    let (codec, block_bytes) = match dxgi {
        71 => (Codec::Bc1, 8),
        74 => (Codec::Bc2, 16),
        77 => (Codec::Bc3, 16),
        _ => return Err(WicError::Unsupported),
    };
    Ok(Mapping { format, codec, block_bytes })
}

/// `GetDXGIFormat` for the legacy (non-DX10) formats that can actually decode: the DXT1–DXT5 FourCCs,
/// → `(dxgiFormat, alphaMode)`. The DDS_PIXELFORMAT is at byte 72 of the header. (GetDXGIFormat also
/// maps BGRX→88 and FourCC 116→R32G32B32A32_FLOAT, but `derive`'s format gate rejects both, so those
/// legacy entries are indistinguishable here from an unrecognised format: all give `Unsupported`.)
fn legacy_dxgi(pf: &[u8]) -> Result<(u32, u32), WicError> {
    if rd_u32(pf, 4) & 4 == 0 {
        return Err(WicError::Unsupported); // not DDPF_FOURCC
    }
    Ok(match &pf[8..12] {
        b"DXT1" => (71, 2),
        b"DXT2" => (74, 2),
        b"DXT3" => (74, 0),
        b"DXT4" => (77, 2),
        b"DXT5" => (77, 0),
        _ => return Err(WicError::Unsupported),
    })
}

fn parse_header(b: &[u8]) -> Result<Params, WicError> {
    if b.len() < 128 || rd_u32(b, 0) != 0x2053_4444 {
        return Err(WicError::BadImage); // not "DDS "
    }
    let h = &b[4..]; // DDS_HEADER (dwSize at h[0])
    let flags = rd_u32(h, 4);
    let height = rd_u32(h, 8);
    let width = rd_u32(h, 12);
    let hdr_depth = rd_u32(h, 20);
    let mut mip_levels = rd_u32(h, 24);
    if mip_levels == 0 {
        mip_levels = 1;
    }
    let caps2 = rd_u32(h, 108);
    let pf = &h[72..]; // DDS_PIXELFORMAT

    // Dimension uses WICDdsDimension codes: 1D=0, 2D=1, 3D=2, Cube=3.
    let has_dx10 = (rd_u32(pf, 4) & 4) != 0 && rd_u32(pf, 8) == u32::from_le_bytes(*b"DX10");
    let (array_size, dimension, depth, dxgi, alpha_mode, data_offset);
    if has_dx10 {
        if b.len() < 148 {
            return Err(WicError::BadImage);
        }
        let d = &b[128..]; // DDS_HEADER_DXT10
        dxgi = rd_u32(d, 0);
        let res_dim = rd_u32(d, 4); // D3D10_RESOURCE_DIMENSION: 2=1D, 3=2D, 4=3D
        let misc = rd_u32(d, 8);
        let arr = rd_u32(d, 12);
        alpha_mode = rd_u32(d, 16) & 7;
        data_offset = 148;
        match res_dim {
            2 => (array_size, dimension, depth) = (arr, 0, 1),
            3 if misc & 4 != 0 => (array_size, dimension, depth) = (6 * arr, 3, 1), // cube
            3 => (array_size, dimension, depth) = (arr, 1, 1),
            4 => {
                if flags & 0x80_0000 == 0 || arr > 1 {
                    return Err(WicError::BadImage);
                }
                (array_size, dimension, depth) = (arr, 2, hdr_depth);
            }
            _ => return Err(WicError::BadImage),
        }
    } else {
        data_offset = 128;
        let (dx, am) = legacy_dxgi(pf)?;
        dxgi = dx;
        alpha_mode = am;
        if flags & 0x80_0000 != 0 {
            (array_size, dimension, depth) = (1, 2, hdr_depth); // DDSD_DEPTH → volume
        } else if caps2 & 0x200 != 0 {
            // DDSCAPS2_CUBEMAP; all six faces required (0xFE00).
            if caps2 & 0xFE00 != 0xFE00 {
                return Err(WicError::BadImage);
            }
            (array_size, dimension, depth) = (6, 3, 1);
        } else {
            (array_size, dimension, depth) = (1, 1, 1);
        }
    }

    let map = derive(dxgi, alpha_mode)?;
    let p = Params {
        width,
        height,
        depth: depth.max(1),
        mip_levels,
        array_size,
        dimension,
        map,
        data_offset,
    };
    validate(&p, alpha_mode)?;
    Ok(p)
}

/// `DdsValidateParameters`: WIC only accepts Texture2D(1) and Texture3D(2) — 1D and cube are rejected.
fn validate(p: &Params, alpha_mode: u32) -> Result<(), WicError> {
    if p.width == 0 || p.height == 0 || p.depth == 0 || p.array_size == 0 || p.mip_levels == 0 {
        return Err(WicError::BadImage);
    }
    if alpha_mode > 4 || p.mip_levels > 15 {
        return Err(WicError::Unsupported);
    }
    let ok = match p.dimension {
        1 => p.width <= 16384 && p.height <= 16384 && p.depth == 1 && p.array_size <= 2048,
        2 => p.width <= 2048 && p.height <= 2048 && p.depth <= 2048 && p.array_size == 1,
        _ => false, // 1D / cube
    };
    if ok { Ok(()) } else { Err(WicError::Unsupported) }
}

// --- frame enumeration + decode ------------------------------------------------------------------

fn subresource_bytes(map: &Mapping, w: u32, h: u32) -> usize {
    (map.block_bytes * ((w + 3) >> 2) * ((h + 3) >> 2)) as usize
}

/// Decode a DDS stream into its subresource frames, in WIC's `GetFrame` order
/// (array slice → mip level → depth slice).
pub fn decode_dds(bytes: &[u8]) -> Result<Vec<DdsFrame>, WicError> {
    let p = parse_header(bytes)?;
    let mut frames = Vec::new();
    let mut offset = p.data_offset;
    for _ in 0..p.array_size {
        for mip in 0..p.mip_levels {
            let w = (p.width >> mip).max(1);
            let h = (p.height >> mip).max(1);
            let d = if p.dimension == 2 { (p.depth >> mip).max(1) } else { 1 }; // 3D volume
            for _ in 0..d {
                let size = subresource_bytes(&p.map, w, h);
                let data = bytes.get(offset..offset + size).ok_or(WicError::BadImage)?;
                frames.push(decode_frame(&p.map, w, h, data)?);
                offset += size;
            }
        }
    }
    Ok(frames)
}

fn decode_frame(map: &Mapping, w: u32, h: u32, data: &[u8]) -> Result<DdsFrame, WicError> {
    let (wu, hu) = (w as usize, h as usize);
    let stride = wu * 4; // 32bpp BGRA
    let mut out = vec![0u8; stride * hu];
    let bx = (wu + 3) / 4;
    let bb = map.block_bytes as usize;
    for by in 0..((hu + 3) / 4) {
        for cx in 0..bx {
            let blk = &data[(by * bx + cx) * bb..][..bb];
            let texels = match map.codec {
                Codec::Bc1 => decode_bc1(blk),
                Codec::Bc2 => decode_bc2(blk),
                Codec::Bc3 => decode_bc3(blk),
            };
            for ty in 0..4 {
                let y = by * 4 + ty;
                if y >= hu {
                    break;
                }
                for tx in 0..4 {
                    let x = cx * 4 + tx;
                    if x >= wu {
                        break;
                    }
                    out[(y * wu + x) * 4..][..4].copy_from_slice(&texels[ty * 4 + tx]);
                }
            }
        }
    }
    Ok(DdsFrame { width: w, height: h, format: map.format, stride, pixels: out })
}

// --- exact block decode (windowscodecs DecodeBC1/2/3) --------------------------------------------
//
// Colours come from RGB565 endpoints scaled to 8-bit with WIC's fixed-point constants:
//   s5(v) = (138006131*v + 0x800000) >> 24   (5-bit channel, v≤31)
//   s6(v) = (67907779*v  + 0x800000) >> 24   (6-bit channel, v≤63)
// Interpolation runs on the raw 5/6-bit endpoints (scale and blend folded into one multiply):
//   third5(x)=(46002043*x+0x800000)>>24  third6(x)=(22635926*x+..)   for (2a+b) and (a+2b)
//   half5(x) =(69003065*x+0x800000)>>24  half6(x) =(33953889*x+..)   for (a+b)
// Each product fits in u32, so plain arithmetic reproduces the DLL exactly.

#[inline]
fn fx(coef: u32, v: u32) -> u8 {
    ((coef * v + 0x0080_0000) >> 24) as u8
}

/// Build a BGRA texel (bytes B,G,R,A) from 5/6/5 endpoint channels, with the given alpha.
#[inline]
fn bgra(r5: u32, g6: u32, b5: u32, a: u8) -> [u8; 4] {
    [fx(138006131, b5), fx(67907779, g6), fx(138006131, r5), a]
}

/// BGRA from an interpolated endpoint combination `(coef5 for R/B, coef6 for G)`.
#[inline]
fn bgra_interp(cr: u32, cb: u32, cg: u32, coef5: u32, coef6: u32, a: u8) -> [u8; 4] {
    [fx(coef5, cb), fx(coef6, cg), fx(coef5, cr), a]
}

/// The four candidate colours of a BC colour block at `bytes[color_off..]` (endpoints + 2 blended).
/// `four_color_only=true` forces the 4-colour mode (BC2/BC3); otherwise BC1's `c0<=c1` 3-colour rule
/// applies (index 3 = transparent black).
fn bc_colors(bytes: &[u8], off: usize, four_color_only: bool) -> [[u8; 4]; 4] {
    let c0 = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32;
    let c1 = u16::from_le_bytes([bytes[off + 2], bytes[off + 3]]) as u32;
    let (r0, g0, b0) = (c0 >> 11, (c0 >> 5) & 0x3f, c0 & 0x1f);
    let (r1, g1, b1) = (c1 >> 11, (c1 >> 5) & 0x3f, c1 & 0x1f);
    let mut col = [[0u8; 4]; 4];
    col[0] = bgra(r0, g0, b0, 255);
    col[1] = bgra(r1, g1, b1, 255);
    if !four_color_only && c0 <= c1 {
        col[2] = bgra_interp(r0 + r1, b0 + b1, g0 + g1, 69003065, 33953889, 255);
        col[3] = [0, 0, 0, 0]; // transparent black
    } else {
        col[2] = bgra_interp(2 * r0 + r1, 2 * b0 + b1, 2 * g0 + g1, 46002043, 22635926, 255);
        col[3] = bgra_interp(r0 + 2 * r1, b0 + 2 * b1, g0 + 2 * g1, 46002043, 22635926, 255);
    }
    col
}

fn decode_bc1(b: &[u8]) -> [[u8; 4]; 16] {
    let col = bc_colors(b, 0, false);
    let idx = rd_u32(b, 4);
    std::array::from_fn(|t| col[((idx >> (2 * t)) & 3) as usize])
}

fn decode_bc2(b: &[u8]) -> [[u8; 4]; 16] {
    // Bytes 0..8 = explicit 4-bit alpha (LSB = texel 0); bytes 8..16 = BC1-style colour (4-colour).
    let col = bc_colors(b, 8, true);
    let idx = rd_u32(b, 12);
    let alpha_bits = u64::from_le_bytes(b[0..8].try_into().unwrap());
    std::array::from_fn(|t| {
        let mut px = col[((idx >> (2 * t)) & 3) as usize];
        px[3] = (((alpha_bits >> (4 * t)) & 0xf) as u8) * 0x11;
        px
    })
}

/// BC3's 8 alpha values from endpoints `a0,a1` (bytes 0,1): 8-value mode when `a0>a1`
/// (step `= ((a0-a1)<<24)/7`), else 6-value mode plus 0 and 255.
fn bc3_alpha(a0: u32, a1: u32) -> [u8; 8] {
    let mut alpha = [0u8; 8];
    alpha[0] = a0 as u8;
    alpha[1] = a1 as u8;
    if a0 > a1 {
        let step = ((a0 - a1) << 24) / 7;
        let base = (a0 << 24) + 0x0080_0000;
        for k in 1..=6u32 {
            alpha[(k + 1) as usize] = ((base - k * step) >> 24) as u8;
        }
    } else {
        let step = ((a1 - a0) << 24) / 5;
        let base = (a0 << 24) + 0x0080_0000;
        for k in 1..=4u32 {
            alpha[(k + 1) as usize] = ((base + k * step) >> 24) as u8;
        }
        alpha[6] = 0;
        alpha[7] = 255;
    }
    alpha
}

fn decode_bc3(b: &[u8]) -> [[u8; 4]; 16] {
    let col = bc_colors(b, 8, true);
    let idx = rd_u32(b, 12);
    let alpha = bc3_alpha(b[0] as u32, b[1] as u32);
    // 3-bit alpha indices: texels 0..7 in bytes 2..5, texels 8..15 in bytes 5..8 (LSB = first texel).
    let alo = b[2] as u32 | (b[3] as u32) << 8 | (b[4] as u32) << 16;
    let ahi = b[5] as u32 | (b[6] as u32) << 8 | (b[7] as u32) << 16;
    std::array::from_fn(|t| {
        let mut px = col[((idx >> (2 * t)) & 3) as usize];
        let ai = if t < 8 { (alo >> (3 * t)) & 7 } else { (ahi >> (3 * (t - 8))) & 7 };
        px[3] = alpha[ai as usize];
        px
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Self-check: the fixed-point endpoint scaling hits the exact 8-bit values WIC produces for the
    // RGB565 extremes, and BC1's c0<=c1 rule yields transparent black on index 3.
    #[test]
    fn bc1_endpoints_and_3color() {
        // c0 = white (0xFFFF), c1 = 0 → c0 > c1, 4-colour, opaque.
        let mut blk = [0u8; 8];
        blk[0] = 0xff;
        blk[1] = 0xff;
        let col = bc_colors(&blk, 0, false);
        assert_eq!(col[0], [255, 255, 255, 255]); // 5/6-bit max → 255
        assert_eq!(col[1], [0, 0, 0, 255]);
        assert_eq!(col[3][3], 255);
        // c0 (0) <= c1 (white) → 3-colour, index 3 transparent.
        blk[0] = 0;
        blk[1] = 0;
        blk[2] = 0xff;
        blk[3] = 0xff;
        let col = bc_colors(&blk, 0, false);
        assert_eq!(col[3], [0, 0, 0, 0]);
    }

    #[test]
    fn bc3_alpha_modes() {
        // 8-alpha mode (a0 > a1): endpoints exact, interpolated monotone decreasing.
        let a = bc3_alpha(255, 0);
        assert_eq!(a[0], 255);
        assert_eq!(a[1], 0);
        assert!(a[2] > a[7]);
        // 6-alpha mode (a0 <= a1): index 6 = 0, index 7 = 255.
        let a = bc3_alpha(0, 255);
        assert_eq!(a[6], 0);
        assert_eq!(a[7], 255);
    }
}
