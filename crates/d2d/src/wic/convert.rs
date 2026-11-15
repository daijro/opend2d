//! `IWICFormatConverter`, as WIC does it. Direct2D loads images by decoding to a
//! WIC frame then converting to the format its device bitmap wants (overwhelmingly `32bppPBGRA`, the
//! premultiplied BGRA D2D composites in; also plain `32bppBGRA`). We reproduce WIC's own conversion,
//! matching WIC's own converter and verified byte-for-byte against it (the premultiply and
//! 64bpp-RGBA → 32bpp-BGRA scanline paths).
//!
//! Model (confirmed against the oracle for every pair below): unpack each source channel to the
//! destination's 8-bit channel depth, then pack in the destination's channel order, premultiplying
//! when the destination is premultiplied.
//!   * sub-8-bit gray / indexed samples expand by bit-replication (`v * 255/(2^n-1)`);
//!   * 48/64-bit colour channels reduce to 8-bit by **truncation** (the high byte, `v >> 8`) — this
//!     is the native 64bpp-RGBA → 32bpp-BGRA / RGB48 path, verified exact on a sweep;
//!   * premultiply is `p = (a*c + 128 + ((a*c+128)>>8)) >> 8` = `round(a*c/255)` (the native magic).
//!
//! Scope: 8-bit channel destinations (24/32bpp) go through the `Rgba8` intermediate; 16-bpc-channel
//! destinations (`16bppGray`, `48bppRGB`, `64bppRGBA`) go through the parallel 16-bit `Rgba16` path
//! (`unpack16`/`pack16`), which native reaches by unpacking to the destination's channel depth: 8-bpc
//! sources expand by bit-replication (`v*257`), 48/64-bit sources are identity, `16bppGray` round-trips
//! through linear. `16bppGray` as a SOURCE into 8-bit reduces through the same gamma round-trip
//! (`gray16_to_u8`, byte-exact on all 65536 inputs).
//!
//! Down-conversion to indexed/low-bit-depth destinations (dithering, palette translation, the alpha
//! threshold) is not reproduced here — Direct2D's load path never asks for it; see `color.rs` notes.

use super::WicError;

/// A WIC pixel format this converter can read or write. Each maps to a `GUID_WICPixelFormat*`.
/// Premultiplied variants (`P*`) carry alpha already folded into the colour channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    BlackWhite,
    Gray2,
    Gray4,
    Gray8,
    Gray16,
    Bgr555,
    Bgr565,
    Indexed1,
    Indexed2,
    Indexed4,
    Indexed8,
    Bgr24,
    Rgb24,
    Bgr32,
    Bgra32,
    Rgba32,
    Pbgra32,
    Prgba32,
    Rgb48,
    Rgba64,
}

impl Format {
    pub const fn bits_per_pixel(self) -> u32 {
        use Format::*;
        match self {
            BlackWhite | Indexed1 => 1,
            Gray2 | Indexed2 => 2,
            Gray4 | Indexed4 => 4,
            Gray8 | Indexed8 => 8,
            Gray16 | Bgr555 | Bgr565 => 16,
            Bgr24 | Rgb24 => 24,
            Bgr32 | Bgra32 | Rgba32 | Pbgra32 | Prgba32 => 32,
            Rgb48 => 48,
            Rgba64 => 64,
        }
    }

    /// Canonical `GUID_WICPixelFormat*` string (used to label / cross-reference oracle captures).
    pub fn guid(self) -> &'static str {
        use Format::*;
        match self {
            BlackWhite => "6fddc324-4e03-4bfe-b185-3d77768dc905",
            Gray2 => "6fddc324-4e03-4bfe-b185-3d77768dc906",
            Gray4 => "6fddc324-4e03-4bfe-b185-3d77768dc907",
            Gray8 => "6fddc324-4e03-4bfe-b185-3d77768dc908",
            Gray16 => "6fddc324-4e03-4bfe-b185-3d77768dc90b",
            Bgr555 => "6fddc324-4e03-4bfe-b185-3d77768dc909",
            Bgr565 => "6fddc324-4e03-4bfe-b185-3d77768dc90a",
            Indexed1 => "6fddc324-4e03-4bfe-b185-3d77768dc901",
            Indexed2 => "6fddc324-4e03-4bfe-b185-3d77768dc902",
            Indexed4 => "6fddc324-4e03-4bfe-b185-3d77768dc903",
            Indexed8 => "6fddc324-4e03-4bfe-b185-3d77768dc904",
            Bgr24 => "6fddc324-4e03-4bfe-b185-3d77768dc90c",
            Rgb24 => "6fddc324-4e03-4bfe-b185-3d77768dc90d",
            Bgr32 => "6fddc324-4e03-4bfe-b185-3d77768dc90e",
            Bgra32 => "6fddc324-4e03-4bfe-b185-3d77768dc90f",
            Rgba32 => "f5c7ad2d-6a8d-43dd-a7a8-a29935261ae9",
            Pbgra32 => "6fddc324-4e03-4bfe-b185-3d77768dc910",
            Prgba32 => "3cc4a650-a527-4d37-a916-3142c7ebedba",
            Rgb48 => "6fddc324-4e03-4bfe-b185-3d77768dc915",
            Rgba64 => "6fddc324-4e03-4bfe-b185-3d77768dc916",
        }
    }

    /// The `Format` whose `guid()` equals `s` (case-insensitive), if any.
    pub fn from_guid(s: &str) -> Option<Format> {
        use Format::*;
        [
            BlackWhite, Gray2, Gray4, Gray8, Gray16, Bgr555, Bgr565, Indexed1, Indexed2, Indexed4,
            Indexed8, Bgr24, Rgb24, Bgr32, Bgra32, Rgba32, Pbgra32, Prgba32, Rgb48, Rgba64,
        ]
        .into_iter()
        .find(|f| f.guid().eq_ignore_ascii_case(s))
    }

    fn is_indexed(self) -> bool {
        use Format::*;
        matches!(self, Indexed1 | Indexed2 | Indexed4 | Indexed8)
    }
}

/// Destination tightly-packed stride, `ceil(width * bpp / 8)`.
pub fn stride(width: u32, fmt: Format) -> usize {
    ((width * fmt.bits_per_pixel() + 7) / 8) as usize
}

/// `round(a*c/255)` exactly as native WIC premultiplies (`AlphaMultiply_32bppBGRA`): no float, no
/// division — `(a*c + 128 + ((a*c+128)>>8)) >> 8`.
fn premul8(c: u8, a: u8) -> u8 {
    let x = a as u32 * c as u32 + 128;
    ((x + (x >> 8)) >> 8) as u8
}

/// Un-premultiply a channel the way native's WIC format converter does: a **16.16 fixed-point
/// reciprocal** of alpha (floored), multiplied, shifted back with truncation, clamped to 255; `a==0 ⇒ 0`.
/// `straight = min(255, (c · ((255<<16)/a)) >> 16)`. VM-derived byte-exact over native-r1's pBGRA/pRGBA
/// → straight pixels (0/408 mismatch; plain `c·255/a` floor or an f32 reciprocal miss 15 at the
/// exact-ratio boundaries, an 8.8 reciprocal misses 85 — the 16.16 reciprocal is the one that matches).
fn unpremul8(c: u8, a: u8) -> u8 {
    if a == 0 {
        0
    } else {
        let recip = (255u32 << 16) / a as u32;
        ((c as u32 * recip) >> 16).min(255) as u8
    }
}

/// Reduce a 48/64-bit colour channel to 8-bit as WIC does: take the high byte (truncation), exactly
/// as the native 64bpp-RGBA → 32bpp-BGRA path. Verified byte-exact against the codec on a full sweep.
fn trunc16(v: u16) -> u8 {
    (v >> 8) as u8
}

/// Reduce a `16bppGray` sample to the 8-bit sRGB byte exactly as native WIC's format converter does
/// (byte-for-byte on all 65536 inputs, confirmed against WIC). It is the two-step gamma path
/// 16bpp-gray → 128bpp-RGBA(linear) → 32bpp-BGRA:
/// decode the sRGB sample to a linear (scRGB) f32, then encode that back to an sRGB byte. R=G=B share
/// this byte; alpha is 255 (premul is a no-op at full alpha, so `Pbgra32` matches `Bgra32`).
fn gray16_to_u8(v: u16) -> u8 {
    scrgb_to_srgb8(srgb16_to_scrgb(v))
}

/// Convert a `16bppGray` sample to the `32bppGrayFloat` value native WIC's format converter produces
/// (byte-for-byte, confirmed against WIC). It is the two-step gamma path 16bpp-gray →
/// 128bpp-RGBA(linear, `R=G=B`) → 32bpp-gray-float: decode to the scRGB linear value, then collapse
/// the equal RGB channels to grey with the luma weights, clamped to `[0, 1]`. The luma weights do not
/// sum to exactly 1.0 in single precision, so this is NOT the identity — it is a few ULP off the raw
/// scRGB value, which the downstream float resampler then sees.
pub(crate) fn gray16_to_grayfloat(v: u16) -> f32 {
    let g = srgb16_to_scrgb(v);
    let y = (g * 0.715_200_01) + (g * 0.212_599_99) + (g * 0.0722);
    y.clamp(0.0, 1.0)
}

/// Decode a 16-bit sRGB sample to a linear (scRGB) f32: linear-interpolate the sRGB→linear LUT at the
/// sample position `v/257` (`idx.frac` from `(65281*v)>>8`), in f64, then scale by `1/255`. The LUT
/// holds `255·srgb_to_linear`; the result is the scRGB linear value in `[0, 1]`.
fn srgb16_to_scrgb(v: u16) -> f32 {
    let p = (65281u32 * v as u32) >> 8;
    let idx = (p >> 16) as usize;
    let frac = (p & 0xFFFF) as f64;
    let lo = f32::from_bits(SRGB16_TO_SCRGB_BITS[idx]) as f64;
    let mut x = lo;
    if frac != 0.0 {
        let hi = f32::from_bits(SRGB16_TO_SCRGB_BITS[idx + 1]) as f64;
        x += (hi - lo) * frac * (1.0 / 65536.0);
    }
    (x * (1.0 / 255.0)) as f32
}

/// Convert a `64bppRGBA` buffer (straight sRGB, `R,G,B,A` u16 LE per pixel) to the
/// `64bppRGBAFixedPoint` bytes WIC's BMP encoder writes for a 64bpp source: each colour channel is
/// decoded to its linear scRGB value, alpha is kept linear, and both are scaled to the s2.13
/// fixed-point grid (`1.0 == 8192`, round half up) and stored in `B,G,R,A` order (the 64bpp BMP byte
/// order). Verified byte-for-byte against the native encoder.
pub(crate) fn rgba64_to_fixedpoint(px: &[u8]) -> Vec<u8> {
    let fix = |x: f32| (x * 8192.0 + 0.5) as i16;
    let mut out = Vec::with_capacity(px.len());
    for p in px.chunks_exact(8) {
        let ch = |i: usize| u16::from_le_bytes([p[i], p[i + 1]]);
        let (r, g, b, a) = (ch(0), ch(2), ch(4), ch(6));
        for v in [fix(srgb16_to_scrgb(b)), fix(srgb16_to_scrgb(g)), fix(srgb16_to_scrgb(r)), fix(a as f32 / 65535.0)] {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Encode a linear value to an 8-bit sRGB byte: quantise the linear value to one of 3354 buckets
/// with round-ties-to-even plus the native exact-`.5` bump, then look up the sRGB
/// byte. Everything is single-precision, matching the native's rounding.
fn scrgb_to_srgb8(a: f32) -> u8 {
    if a <= 0.0 {
        return 0;
    }
    if a >= 1.0 {
        return 255;
    }
    let t = a * 3354.0;
    let mut i = t.round_ties_even();
    if (i - t) == -0.5 {
        i += 1.0;
    }
    SCRGB_TO_SRGB8[i as usize]
}

/// Encode a linear scRGB value (in `[0,1]`) to a 16-bit sRGB code, exactly as native WIC's
/// `64bppRGBA` / `16bppGray` writers do. Seed the 8-bit result as a bracket index into the
/// sRGB→linear LUT, walk it so `LUT[bracket] <= 255·a < LUT[bracket+1]`, linear-interpolate to 8
/// fractional bits, then scale `(bracket<<8)+t` by `0x10101/0x10000` (= 256/255) with a 32-bit
/// multiply + `>>16`. Precision matters: `255·a` and the compares are f64, but the bracket width
/// is an f32 subtraction before widening; `t` truncates toward zero after the `+0.5`. Endpoints
/// clamp to `0` / `0xFFFF`.
fn scrgb_to_srgb16(a: f32) -> u16 {
    let d = a as f64 * 255.0;
    if d <= 0.0 {
        return 0;
    }
    if d >= 255.0 {
        return 0xFFFF;
    }
    let lut = |i: usize| f32::from_bits(SRGB16_TO_SCRGB_BITS[i]);
    let mut bracket = scrgb_to_srgb8(a) as usize;
    if bracket > 0xFE {
        bracket = 0xFE;
    }
    while lut(bracket) as f64 > d {
        bracket -= 1;
    }
    while d >= lut(bracket + 1) as f64 {
        bracket += 1;
    }
    let num = d - lut(bracket) as f64;
    let den = (lut(bracket + 1) - lut(bracket)) as f64; // f32 difference, then widened
    let t = (num / den * 256.0 + 0.5) as i64; // truncate toward zero
    let idx = ((bracket as u32) << 8).wrapping_add(t as u32);
    (idx.wrapping_mul(0x10101) >> 16) as u16
}

/// Expand an `n`-bit gray/index sample to 8-bit by bit-replication (`v * 255/(2^n-1)`), exact for
/// n = 1,2,4,8.
fn widen(v: u8, bits: u32) -> u8 {
    (v as u32 * 255 / ((1 << bits) - 1)) as u8
}

/// One straight-alpha pixel at 8-bit channel depth: (r, g, b, a).
type Rgba8 = [u8; 4];

/// Read the `w`×`h` source as straight-alpha `Rgba8` pixels, row-major. Sub-byte formats are
/// unpacked MSB-first within each `stride`-padded row, exactly like WIC scanline reads.
fn unpack(src: Format, px: &[u8], w: u32, h: u32, palette: &[u32]) -> Result<Vec<Rgba8>, WicError> {
    use Format::*;
    let (w, h) = (w as usize, h as usize);
    let srow = stride(w as u32, src);
    if px.len() < srow * h {
        return Err(WicError::BadImage);
    }
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        let row = &px[y * srow..];
        match src {
            BlackWhite | Gray2 | Gray4 | Indexed1 | Indexed2 | Indexed4 => {
                let bits = src.bits_per_pixel();
                for x in 0..w {
                    let bitpos = x as u32 * bits;
                    let byte = row[(bitpos / 8) as usize];
                    let shift = 8 - bits - (bitpos % 8);
                    let v = (byte >> shift) & ((1 << bits) - 1);
                    out.push(sample(src, v, palette)?);
                }
            }
            Gray8 | Indexed8 => {
                for x in 0..w {
                    out.push(sample(src, row[x], palette)?);
                }
            }
            Gray16 => {
                for x in 0..w {
                    let g = gray16_to_u8(u16::from_le_bytes([row[2 * x], row[2 * x + 1]]));
                    out.push([g, g, g, 255]);
                }
            }
            // 16bpp packed BGR → 8bpc by native's bit-replication expand (VM-derived byte-exact):
            // r8=(r<<3)|(r>>2) for a 5-bit channel, (g<<2)|(g>>4) for the 6-bit green.
            Bgr565 => {
                for x in 0..w {
                    let v = u16::from_le_bytes([row[2 * x], row[2 * x + 1]]);
                    let (r5, g6, b5) = (((v >> 11) & 0x1f) as u8, ((v >> 5) & 0x3f) as u8, (v & 0x1f) as u8);
                    out.push([(r5 << 3) | (r5 >> 2), (g6 << 2) | (g6 >> 4), (b5 << 3) | (b5 >> 2), 255]);
                }
            }
            Bgr555 => {
                for x in 0..w {
                    let v = u16::from_le_bytes([row[2 * x], row[2 * x + 1]]);
                    let (r5, g5, b5) = (((v >> 10) & 0x1f) as u8, ((v >> 5) & 0x1f) as u8, (v & 0x1f) as u8);
                    out.push([(r5 << 3) | (r5 >> 2), (g5 << 3) | (g5 >> 2), (b5 << 3) | (b5 >> 2), 255]);
                }
            }
            Bgr24 => {
                for x in 0..w {
                    let p = &row[3 * x..];
                    out.push([p[2], p[1], p[0], 255]);
                }
            }
            Rgb24 => {
                for x in 0..w {
                    let p = &row[3 * x..];
                    out.push([p[0], p[1], p[2], 255]);
                }
            }
            Bgra32 => {
                for x in 0..w {
                    let p = &row[4 * x..];
                    out.push([p[2], p[1], p[0], p[3]]);
                }
            }
            Rgba32 => {
                for x in 0..w {
                    let p = &row[4 * x..];
                    out.push([p[0], p[1], p[2], p[3]]);
                }
            }
            // 32bppBGR: opaque, the 4th byte is ignored on read (alpha ⇒ 255).
            Bgr32 => {
                for x in 0..w {
                    let p = &row[4 * x..];
                    out.push([p[2], p[1], p[0], 255]);
                }
            }
            Rgb48 => {
                for x in 0..w {
                    let p = &row[6 * x..];
                    out.push([
                        trunc16(u16::from_le_bytes([p[0], p[1]])),
                        trunc16(u16::from_le_bytes([p[2], p[3]])),
                        trunc16(u16::from_le_bytes([p[4], p[5]])),
                        255,
                    ]);
                }
            }
            Rgba64 => {
                for x in 0..w {
                    let p = &row[8 * x..];
                    out.push([
                        trunc16(u16::from_le_bytes([p[0], p[1]])),
                        trunc16(u16::from_le_bytes([p[2], p[3]])),
                        trunc16(u16::from_le_bytes([p[4], p[5]])),
                        trunc16(u16::from_le_bytes([p[6], p[7]])),
                    ]);
                }
            }
            // Premultiplied sources → straight alpha via native's FLOOR un-premultiply (unpremul8).
            Pbgra32 => {
                for x in 0..w {
                    let p = &row[4 * x..];
                    let a = p[3];
                    out.push([unpremul8(p[2], a), unpremul8(p[1], a), unpremul8(p[0], a), a]);
                }
            }
            Prgba32 => {
                for x in 0..w {
                    let p = &row[4 * x..];
                    let a = p[3];
                    out.push([unpremul8(p[0], a), unpremul8(p[1], a), unpremul8(p[2], a), a]);
                }
            }
        }
    }
    Ok(out)
}

/// One sub-byte / 8-bit gray or indexed sample → straight `Rgba8`.
fn sample(src: Format, v: u8, palette: &[u32]) -> Result<Rgba8, WicError> {
    Ok(if src.is_indexed() {
        let e = *palette.get(v as usize).ok_or(WicError::BadImage)?;
        [(e >> 16) as u8, (e >> 8) as u8, e as u8, (e >> 24) as u8]
    } else {
        let g = widen(v, src.bits_per_pixel());
        [g, g, g, 255]
    })
}

/// Pack straight-alpha `Rgba8` pixels into `dst`'s channel order and bit depth, premultiplying for
/// premultiplied destinations. 16-bit destinations expand each 8-bit channel by `v*257`.
fn pack(dst: Format, pix: &[Rgba8], w: u32) -> Result<Vec<u8>, WicError> {
    use Format::*;
    let drow = stride(w, dst);
    let h = pix.len() / w as usize;
    let mut out = vec![0u8; drow * h];
    for (i, &[r, g, b, a]) in pix.iter().enumerate() {
        let (y, x) = (i / w as usize, i % w as usize);
        let o = &mut out[y * drow..];
        match dst {
            Bgr24 => o[3 * x..3 * x + 3].copy_from_slice(&[b, g, r]),
            Rgb24 => o[3 * x..3 * x + 3].copy_from_slice(&[r, g, b]),
            Bgra32 => o[4 * x..4 * x + 4].copy_from_slice(&[b, g, r, a]),
            Rgba32 => o[4 * x..4 * x + 4].copy_from_slice(&[r, g, b, a]),
            // 32bppBGR: native carries the (intermediate) alpha in the unused X byte (VM-derived:
            // RGBA32→BGR32 writes X == source alpha), so the bytes equal a BGRA32 write.
            Bgr32 => o[4 * x..4 * x + 4].copy_from_slice(&[b, g, r, a]),
            // 8bpc → 16bpp packed BGR by native's TRUNCATION (VM-derived: r>>3 / g>>2 / b>>3, not
            // round); BGR555 leaves bit 15 clear.
            // RGB → gray the way native's converter does: to scRGB (linear) via the 8→16 sRGB rep,
            // Rec.709 linear luma in f32, back to sRGB8 (WIC's gamma-aware gray path, not an sRGB luma).
            Gray8 => {
                let y = 0.212_6 * srgb16_to_scrgb(r as u16 * 257)
                    + 0.715_2 * srgb16_to_scrgb(g as u16 * 257)
                    + 0.072_2 * srgb16_to_scrgb(b as u16 * 257);
                o[x] = scrgb_to_srgb8(y);
            }
            Bgr565 => {
                let v = (((r >> 3) as u16) << 11) | (((g >> 2) as u16) << 5) | ((b >> 3) as u16);
                o[2 * x..2 * x + 2].copy_from_slice(&v.to_le_bytes());
            }
            Bgr555 => {
                let v = (((r >> 3) as u16) << 10) | (((g >> 3) as u16) << 5) | ((b >> 3) as u16);
                o[2 * x..2 * x + 2].copy_from_slice(&v.to_le_bytes());
            }
            Pbgra32 => {
                o[4 * x..4 * x + 4].copy_from_slice(&[premul8(b, a), premul8(g, a), premul8(r, a), a])
            }
            Prgba32 => {
                o[4 * x..4 * x + 4].copy_from_slice(&[premul8(r, a), premul8(g, a), premul8(b, a), a])
            }
            // 48/64-bit destinations are out of D2D's load-path scope.
            _ => return Err(WicError::Unsupported),
        }
    }
    Ok(out)
}

/// One straight-alpha pixel at 16-bit channel depth, sRGB-encoded: (r, g, b, a).
type Rgba16 = [u16; 4];

/// Read the source as straight-alpha 16-bit sRGB `Rgba16`, native's intermediate for 16-bpc-channel
/// destinations (it unpacks to the destination's channel depth). 8-bpc / sub-8 / 16bpp-packed sources
/// go through the `Rgba8` unpack then expand by bit-replication (`v*257`, native's 8→16); 48/64-bit
/// colour sources are identity; `16bppGray` decodes to linear and re-encodes each channel to sRGB16
/// (native's gray→colour gamma round-trip: decode sRGB16→linear, then re-encode linear→sRGB16).
fn unpack16(src: Format, px: &[u8], w: u32, h: u32, palette: &[u32]) -> Result<Vec<Rgba16>, WicError> {
    use Format::*;
    let (wz, hz) = (w as usize, h as usize);
    let srow = stride(w, src);
    let read16 = |p: &[u8]| u16::from_le_bytes([p[0], p[1]]);
    match src {
        Rgba64 | Rgb48 | Gray16 => {
            if px.len() < srow * hz {
                return Err(WicError::BadImage);
            }
            let mut out = Vec::with_capacity(wz * hz);
            for y in 0..hz {
                let row = &px[y * srow..];
                for x in 0..wz {
                    out.push(match src {
                        Rgba64 => {
                            let p = &row[8 * x..];
                            [read16(p), read16(&p[2..]), read16(&p[4..]), read16(&p[6..])]
                        }
                        Rgb48 => {
                            let p = &row[6 * x..];
                            [read16(p), read16(&p[2..]), read16(&p[4..]), 0xFFFF]
                        }
                        // 16bppGray → colour: sRGB gray → linear scRGB → sRGB16 per channel (R=G=B).
                        _ => {
                            let g = scrgb_to_srgb16(srgb16_to_scrgb(read16(&row[2 * x..])));
                            [g, g, g, 0xFFFF]
                        }
                    });
                }
            }
            Ok(out)
        }
        // Everything 8-bpc-equivalent: unpack to straight Rgba8, replicate each channel to 16 bits.
        _ => Ok(unpack(src, px, w, h, palette)?
            .into_iter()
            .map(|[r, g, b, a]| [r as u16 * 257, g as u16 * 257, b as u16 * 257, a as u16 * 257])
            .collect()),
    }
}

/// Pack straight-alpha `Rgba16` pixels into a 16-bpc-channel destination. `64bppRGBA`/`48bppRGB` write
/// the channels little-endian (straight alpha); `16bppGray` is the gamma-aware Rec.709 luma encoded to
/// 16 bits (same weights + encode native uses, just the 16-bit linear→sRGB encode).
fn pack16(dst: Format, pix: &[Rgba16], w: u32) -> Result<Vec<u8>, WicError> {
    use Format::*;
    let drow = stride(w, dst);
    let h = pix.len() / w as usize;
    let mut out = vec![0u8; drow * h];
    for (i, &[r, g, b, a]) in pix.iter().enumerate() {
        let (y, x) = (i / w as usize, i % w as usize);
        let o = &mut out[y * drow..];
        match dst {
            Rgba64 => {
                let p = &mut o[8 * x..8 * x + 8];
                p[0..2].copy_from_slice(&r.to_le_bytes());
                p[2..4].copy_from_slice(&g.to_le_bytes());
                p[4..6].copy_from_slice(&b.to_le_bytes());
                p[6..8].copy_from_slice(&a.to_le_bytes());
            }
            Rgb48 => {
                let p = &mut o[6 * x..6 * x + 6];
                p[0..2].copy_from_slice(&r.to_le_bytes());
                p[2..4].copy_from_slice(&g.to_le_bytes());
                p[4..6].copy_from_slice(&b.to_le_bytes());
            }
            Gray16 => {
                let y = 0.212_6 * srgb16_to_scrgb(r)
                    + 0.715_2 * srgb16_to_scrgb(g)
                    + 0.072_2 * srgb16_to_scrgb(b);
                o[2 * x..2 * x + 2].copy_from_slice(&scrgb_to_srgb16(y).to_le_bytes());
            }
            _ => return Err(WicError::Unsupported),
        }
    }
    Ok(out)
}

/// Convert `w`×`h` `src` pixels (tightly strided, with `palette` for indexed sources, each entry
/// `0xAARRGGBB`) to `dst`. Returns tightly-strided destination bytes.
pub fn convert(
    src: Format,
    dst: Format,
    pixels: &[u8],
    w: u32,
    h: u32,
    palette: &[u32],
) -> Result<Vec<u8>, WicError> {
    // Same format in and out: native's converter is a straight passthrough (VM-verified identity for
    // pBGRA→pBGRA / pRGBA→pRGBA, native-r1 seeds 13/307), so copy the (tightly-strided) source rather
    // than round-tripping a premultiplied format through unpremultiply→premultiply (which would lose a
    // bit). The rest of this module assumes a tight source stride, as `unpack` does.
    if src == dst {
        let n = stride(w, src) * h as usize;
        if pixels.len() < n {
            return Err(WicError::BadImage);
        }
        return Ok(pixels[..n].to_vec());
    }
    // 16-bpc-channel destinations unpack to the destination's channel depth (native's model).
    if matches!(dst, Format::Gray16 | Format::Rgb48 | Format::Rgba64) {
        let pix = unpack16(src, pixels, w, h, palette)?;
        return pack16(dst, &pix, w);
    }
    let pix = unpack(src, pixels, w, h, palette)?;
    pack(dst, &pix, w)
}

/// WIC's sRGB→linear gamma LUT: 257 f32 = `255·srgb_to_linear(i/255)`,
/// stored as bit patterns for exact reproduction (computing it is 1 ULP off at index 15).
#[rustfmt::skip]
static SRGB16_TO_SCRGB_BITS: [u32; 257] = [
    0x00000000, 0x3d9e8391, 0x3e1e8391, 0x3e6dc55a, 0x3e9e8391, 0x3ec62475, 0x3eedc55a, 0x3f0ab31f,
    0x3f1e8391, 0x3f325403, 0x3f462475, 0x3f5a763c, 0x3f70009d, 0x3f835de4, 0x3f8f5630, 0x3f9beb76,
    0x3fa91fed, 0x3fb6f5c1, 0x3fc56f15, 0x3fd48dfd, 0x3fe45487, 0x3ff4c4b6, 0x4002f042, 0x400bd4f2,
    0x4015115e, 0x401ea678, 0x4028952d, 0x4032de65, 0x403d8308, 0x404883f7, 0x4053e212, 0x405f9e35,
    0x406bb939, 0x407833f5, 0x4082879e, 0x408925f0, 0x408ff556, 0x4096f638, 0x409e28fa, 0x40a58dff,
    0x40ad25aa, 0x40b4f05c, 0x40bcee75, 0x40c52054, 0x40cd8658, 0x40d620dd, 0x40def040, 0x40e7f4dc,
    0x40f12f0b, 0x40fa9f28, 0x410222c5, 0x41071144, 0x410c1b3e, 0x411140dd, 0x4116824b, 0x411bdfb4,
    0x41215940, 0x4126ef1b, 0x412ca16c, 0x4132705d, 0x41385c16, 0x413e64c0, 0x41448a82, 0x414acd84,
    0x41512ded, 0x4157abe3, 0x415e478f, 0x41650116, 0x416bd89e, 0x4172ce4c, 0x4179e247, 0x41808a5a,
    0x418432dc, 0x4187eabb, 0x418bb20a, 0x418f88db, 0x41936f40, 0x4197654b, 0x419b6b0e, 0x419f8099,
    0x41a3a600, 0x41a7db52, 0x41ac20a2, 0x41b07600, 0x41b4db7e, 0x41b9512d, 0x41bdd71d, 0x41c26d5f,
    0x41c71404, 0x41cbcb1d, 0x41d092b9, 0x41d56aea, 0x41da53bf, 0x41df4d49, 0x41e45797, 0x41e972bb,
    0x41ee9ec3, 0x41f3dbc0, 0x41f929c1, 0x41fe88d6, 0x4201fc87, 0x4204bd3d, 0x42078693, 0x420a5893,
    0x420d3343, 0x421016ac, 0x421302d4, 0x4215f7c3, 0x4218f580, 0x421bfc14, 0x421f0b85, 0x422223da,
    0x4225451c, 0x42286f52, 0x422ba281, 0x422edeb3, 0x423223ee, 0x42357239, 0x4238c99c, 0x423c2a1c,
    0x423f93c3, 0x42430695, 0x4246829c, 0x424a07dc, 0x424d965e, 0x42512e29, 0x4254cf42, 0x425879b2,
    0x425c2d7e, 0x425feaae, 0x4263b149, 0x42678154, 0x426b5ad7, 0x426f3dd9, 0x42732a60, 0x42772072,
    0x427b2017, 0x427f2955, 0x42819e19, 0x4283ac5a, 0x4285bf72, 0x4287d763, 0x4289f431, 0x428c15df,
    0x428e3c6f, 0x429067e6, 0x42929847, 0x4294cd93, 0x429707d0, 0x429946ff, 0x429b8b25, 0x429dd443,
    0x42a0225e, 0x42a27578, 0x42a4cd94, 0x42a72ab5, 0x42a98cdf, 0x42abf415, 0x42ae6059, 0x42b0d1ae,
    0x42b34819, 0x42b5c39a, 0x42b84437, 0x42bac9f0, 0x42bd54cb, 0x42bfe4c8, 0x42c279ed, 0x42c5143a,
    0x42c7b3b4, 0x42ca585d, 0x42cd0238, 0x42cfb148, 0x42d26590, 0x42d51f13, 0x42d7ddd4, 0x42daa1d5,
    0x42dd6b19, 0x42e039a4, 0x42e30d78, 0x42e5e698, 0x42e8c506, 0x42eba8c6, 0x42ee91da, 0x42f18046,
    0x42f4740b, 0x42f76d2d, 0x42fa6baf, 0x42fd6f92, 0x43003c6e, 0x4301c3c6, 0x43034dd3, 0x4304da97,
    0x43066a13, 0x4307fc49, 0x43099139, 0x430b28e5, 0x430cc34e, 0x430e6076, 0x4310005e, 0x4311a307,
    0x43134873, 0x4314f0a4, 0x43169b99, 0x43184955, 0x4319f9d9, 0x431bad27, 0x431d633f, 0x431f1c22,
    0x4320d7d3, 0x43229653, 0x432457a2, 0x43261bc3, 0x4327e2b5, 0x4329ac7c, 0x432b7917, 0x432d4888,
    0x432f1ad1, 0x4330eff3, 0x4332c7ee, 0x4334a2c5, 0x43368079, 0x4338610a, 0x433a447a, 0x433c2acb,
    0x433e13fd, 0x43400011, 0x4341ef0a, 0x4343e0e8, 0x4345d5ad, 0x4347cd59, 0x4349c7ee, 0x434bc56d,
    0x434dc5d8, 0x434fc92f, 0x4351cf74, 0x4353d8a8, 0x4355e4cd, 0x4357f3e2, 0x435a05eb, 0x435c1ae7,
    0x435e32d8, 0x43604dc0, 0x43626b9f, 0x43648c77, 0x4366b048, 0x4368d714, 0x436b00dd, 0x436d2da3,
    0x436f5d68, 0x4371902c, 0x4373c5f1, 0x4375feb8, 0x43783a82, 0x437a7951, 0x437cbb25, 0x437f0000,
    0x437f0000,
];

/// WIC's linear→sRGB gamma LUT: 3355 bytes =
/// `round_ties_even(255·srgb_encode(i/3354))`, indexed by `round_ties_even(linear·3354)` (which can
/// reach 3354 for a linear value just below 1.0, hence the trailing 0xFF entry).
#[rustfmt::skip]
static SCRGB_TO_SRGB8: [u8; 3355] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 14, 15, 16, 16, 17, 18, 18, 19, 20, 20, 21, 21, 22, 23, 23, 24, 24,
    25, 25, 26, 26, 27, 27, 28, 28, 28, 29, 29, 30, 30, 31, 31, 31, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 36, 36, 36, 37, 37, 37,
    38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 44, 45, 45, 45, 46, 46, 46, 46, 47, 47,
    47, 47, 48, 48, 48, 48, 49, 49, 49, 50, 50, 50, 50, 51, 51, 51, 51, 51, 52, 52, 52, 52, 53, 53, 53, 53, 54, 54, 54, 54, 55, 55,
    55, 55, 55, 56, 56, 56, 56, 57, 57, 57, 57, 57, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61,
    62, 62, 62, 62, 62, 63, 63, 63, 63, 63, 64, 64, 64, 64, 64, 65, 65, 65, 65, 65, 66, 66, 66, 66, 66, 66, 67, 67, 67, 67, 67, 67,
    68, 68, 68, 68, 68, 69, 69, 69, 69, 69, 69, 70, 70, 70, 70, 70, 70, 71, 71, 71, 71, 71, 71, 72, 72, 72, 72, 72, 72, 73, 73, 73,
    73, 73, 73, 74, 74, 74, 74, 74, 74, 75, 75, 75, 75, 75, 75, 75, 76, 76, 76, 76, 76, 76, 77, 77, 77, 77, 77, 77, 77, 78, 78, 78,
    78, 78, 78, 79, 79, 79, 79, 79, 79, 79, 80, 80, 80, 80, 80, 80, 80, 81, 81, 81, 81, 81, 81, 81, 82, 82, 82, 82, 82, 82, 82, 83,
    83, 83, 83, 83, 83, 83, 84, 84, 84, 84, 84, 84, 84, 84, 85, 85, 85, 85, 85, 85, 85, 86, 86, 86, 86, 86, 86, 86, 87, 87, 87, 87,
    87, 87, 87, 87, 88, 88, 88, 88, 88, 88, 88, 88, 89, 89, 89, 89, 89, 89, 89, 90, 90, 90, 90, 90, 90, 90, 90, 91, 91, 91, 91, 91,
    91, 91, 91, 92, 92, 92, 92, 92, 92, 92, 92, 92, 93, 93, 93, 93, 93, 93, 93, 93, 94, 94, 94, 94, 94, 94, 94, 94, 95, 95, 95, 95,
    95, 95, 95, 95, 95, 96, 96, 96, 96, 96, 96, 96, 96, 97, 97, 97, 97, 97, 97, 97, 97, 97, 98, 98, 98, 98, 98, 98, 98, 98, 98, 99,
    99, 99, 99, 99, 99, 99, 99, 100, 100, 100, 100, 100, 100, 100, 100, 100, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 102, 102, 102, 102, 102, 102,
    102, 102, 102, 103, 103, 103, 103, 103, 103, 103, 103, 103, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 105, 105, 105, 105, 105, 105, 105, 105, 105, 106,
    106, 106, 106, 106, 106, 106, 106, 106, 106, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 108, 108, 108, 108, 108, 108, 108, 108, 108, 109, 109, 109, 109,
    109, 109, 109, 109, 109, 109, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 112, 112, 112, 112, 112,
    112, 112, 112, 112, 112, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 115, 115, 115, 115, 115, 115,
    115, 115, 115, 115, 115, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 118, 118, 118, 118, 118,
    118, 118, 118, 118, 118, 118, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 121, 121, 121, 121,
    121, 121, 121, 121, 121, 121, 121, 121, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 124,
    124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
    126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 129, 129, 129, 129, 129,
    129, 129, 129, 129, 129, 129, 129, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131,
    132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 134, 134, 134, 134, 134, 134,
    134, 134, 134, 134, 134, 134, 134, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136,
    136, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 137, 138, 138, 138, 138, 138, 138, 138, 138, 138, 138, 138, 138, 138, 138, 139, 139, 139, 139,
    139, 139, 139, 139, 139, 139, 139, 139, 139, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 141, 141, 141, 141, 141, 141, 141, 141, 141,
    141, 141, 141, 141, 141, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143,
    143, 144, 144, 144, 144, 144, 144, 144, 144, 144, 144, 144, 144, 144, 144, 145, 145, 145, 145, 145, 145, 145, 145, 145, 145, 145, 145, 145, 145, 146, 146, 146,
    146, 146, 146, 146, 146, 146, 146, 146, 146, 146, 146, 146, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 148, 148, 148, 148, 148, 148,
    148, 148, 148, 148, 148, 148, 148, 148, 148, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 150, 150, 150, 150, 150, 150, 150, 150,
    150, 150, 150, 150, 150, 150, 150, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 152, 152, 152, 152, 152, 152, 152, 152, 152, 152,
    152, 152, 152, 152, 152, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
    154, 154, 154, 154, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156,
    156, 156, 156, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 158, 158, 158, 158, 158, 158, 158, 158, 158, 158, 158, 158, 158,
    158, 158, 158, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
    160, 160, 160, 160, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 161, 162, 162, 162, 162, 162, 162, 162, 162, 162, 162, 162, 162,
    162, 162, 162, 162, 162, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 163, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164, 164,
    164, 164, 164, 164, 164, 164, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 166, 166, 166, 166, 166, 166, 166, 166, 166,
    166, 166, 166, 166, 166, 166, 166, 166, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 167, 168, 168, 168, 168, 168, 168, 168,
    168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 169, 170, 170, 170, 170,
    170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 171, 172,
    172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 172, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173, 173,
    173, 173, 173, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 174, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175,
    175, 175, 175, 175, 175, 175, 175, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 177, 177, 177, 177, 177, 177,
    177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 178, 179,
    179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180,
    180, 180, 180, 180, 180, 181, 181, 181, 181, 181, 181, 181, 181, 181, 181, 181, 181, 181, 181, 181, 181, 181, 181, 181, 182, 182, 182, 182, 182, 182, 182, 182,
    182, 182, 182, 182, 182, 182, 182, 182, 182, 182, 182, 183, 183, 183, 183, 183, 183, 183, 183, 183, 183, 183, 183, 183, 183, 183, 183, 183, 183, 183, 184, 184,
    184, 184, 184, 184, 184, 184, 184, 184, 184, 184, 184, 184, 184, 184, 184, 184, 184, 184, 185, 185, 185, 185, 185, 185, 185, 185, 185, 185, 185, 185, 185, 185,
    185, 185, 185, 185, 185, 185, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 186, 187, 187, 187, 187, 187, 187, 187,
    187, 187, 187, 187, 187, 187, 187, 187, 187, 187, 187, 187, 187, 188, 188, 188, 188, 188, 188, 188, 188, 188, 188, 188, 188, 188, 188, 188, 188, 188, 188, 188,
    188, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 189, 190, 190, 190, 190, 190, 190, 190, 190, 190, 190, 190,
    190, 190, 190, 190, 190, 190, 190, 190, 190, 190, 191, 191, 191, 191, 191, 191, 191, 191, 191, 191, 191, 191, 191, 191, 191, 191, 191, 191, 191, 191, 192, 192,
    192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193,
    193, 193, 193, 193, 193, 193, 193, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 195, 195, 195, 195,
    195, 195, 195, 195, 195, 195, 195, 195, 195, 195, 195, 195, 195, 195, 195, 195, 195, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196,
    196, 196, 196, 196, 196, 196, 196, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 198, 198, 198, 198,
    198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 199, 199, 199, 199, 199, 199, 199, 199, 199, 199, 199, 199, 199, 199, 199,
    199, 199, 199, 199, 199, 199, 199, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 201, 201, 201,
    201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 202, 202, 202, 202, 202, 202, 202, 202, 202, 202, 202, 202, 202, 202,
    202, 202, 202, 202, 202, 202, 202, 202, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 203, 204,
    204, 204, 204, 204, 204, 204, 204, 204, 204, 204, 204, 204, 204, 204, 204, 204, 204, 204, 204, 204, 204, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205,
    205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 207, 207, 207, 207, 207, 207, 207, 207, 207, 207, 207, 207, 207, 207, 207, 207, 207, 207, 207, 207, 207, 207, 207, 208, 208, 208, 208, 208, 208, 208,
    208, 208, 208, 208, 208, 208, 208, 208, 208, 208, 208, 208, 208, 208, 208, 208, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209, 209,
    209, 209, 209, 209, 209, 209, 209, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 211, 211,
    211, 211, 211, 211, 211, 211, 211, 211, 211, 211, 211, 211, 211, 211, 211, 211, 211, 211, 211, 211, 211, 212, 212, 212, 212, 212, 212, 212, 212, 212, 212, 212,
    212, 212, 212, 212, 212, 212, 212, 212, 212, 212, 212, 212, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213,
    213, 213, 213, 213, 214, 214, 214, 214, 214, 214, 214, 214, 214, 214, 214, 214, 214, 214, 214, 214, 214, 214, 214, 214, 214, 214, 214, 214, 215, 215, 215, 215,
    215, 215, 215, 215, 215, 215, 215, 215, 215, 215, 215, 215, 215, 215, 215, 215, 215, 215, 215, 215, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216,
    216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 217, 217, 217, 217, 217, 217, 217, 217, 217, 217, 217, 217, 217, 217, 217, 217, 217, 217, 217, 217,
    217, 217, 217, 217, 218, 218, 218, 218, 218, 218, 218, 218, 218, 218, 218, 218, 218, 218, 218, 218, 218, 218, 218, 218, 218, 218, 218, 218, 219, 219, 219, 219,
    219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220,
    220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221,
    221, 221, 221, 221, 221, 221, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 223,
    223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 224, 224, 224, 224, 224, 224, 224, 224,
    224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225,
    225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226, 226,
    226, 226, 226, 226, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 228, 228,
    228, 228, 228, 228, 228, 228, 228, 228, 228, 228, 228, 228, 228, 228, 228, 228, 228, 228, 228, 228, 228, 228, 228, 228, 229, 229, 229, 229, 229, 229, 229, 229,
    229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230,
    230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 230, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231,
    231, 231, 231, 231, 231, 231, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232,
    233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 234, 234, 234, 234, 234,
    234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235,
    235, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236,
    236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 237, 237, 237, 237, 237, 237, 237, 237, 237, 237, 237, 237, 237, 237, 237, 237, 237, 237, 237, 237, 237,
    237, 237, 237, 237, 237, 237, 237, 238, 238, 238, 238, 238, 238, 238, 238, 238, 238, 238, 238, 238, 238, 238, 238, 238, 238, 238, 238, 238, 238, 238, 238, 238,
    238, 238, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239, 240, 240, 240,
    240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 241, 241, 241, 241, 241, 241, 241,
    241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242,
    242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243,
    243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 244, 244, 244, 244, 244, 244, 244, 244, 244, 244, 244, 244, 244, 244, 244, 244, 244, 244, 244,
    244, 244, 244, 244, 244, 244, 244, 244, 244, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245,
    245, 245, 245, 245, 245, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246,
    246, 246, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 248, 248,
    248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 249, 249, 249, 249, 249,
    249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 250, 250, 250, 250, 250, 250, 250, 250,
    250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251,
    251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
    252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253,
    253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba64_to_fixedpoint_matches_native() {
        // Seed 100165 (64bppRGBA → BMP 64bppRGBAFixedPoint): the five native-oracle pixels.
        let inputs: [[u16; 4]; 5] = [
            [14181, 28291, 17434, 40848], // R,G,B,A
            [996, 12645, 49011, 40904],
            [4172, 32270, 5164, 33862],
            [49409, 29812, 36892, 52909],
            [41313, 36770, 9250, 20976],
        ];
        let expected: [[i16; 4]; 5] = [
            [471, 1279, 315, 5106], // stored B,G,R,A
            [4253, 254, 10, 5113],
            [58, 1696, 43, 4233],
            [2269, 1431, 4331, 6614],
            [144, 2253, 2910, 2622],
        ];
        for (inp, exp) in inputs.iter().zip(expected.iter()) {
            let px: Vec<u8> = inp.iter().flat_map(|c| c.to_le_bytes()).collect();
            let got: Vec<i16> =
                rgba64_to_fixedpoint(&px).chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect();
            assert_eq!(got, exp.to_vec(), "input {inp:?}");
        }
    }

    #[test]
    fn bgr32_carries_alpha_in_x() {
        // Native's 32bppBGR output carries the (intermediate) alpha in the unused X byte, and the read
        // ignores it (opaque ⇒ alpha 255). VM-derived: RGBA32→BGR32 writes X == source alpha.
        assert_eq!(pack(Format::Bgr32, &[[10, 20, 30, 58]], 1).unwrap(), vec![30, 20, 10, 58]);
        assert_eq!(unpack(Format::Bgr32, &[30, 20, 10, 99], 1, 1, &[]).unwrap()[0], [10, 20, 30, 255]);
    }

    #[test]
    fn gray8_rec709_linear_luma() {
        // RGB → gray is gamma-aware: sRGB→linear, Rec.709 luma, linear→sRGB (NOT an sRGB-domain luma).
        // Native-verified spot values (native-r1): a pure-channel green reads much brighter than 0.587·g.
        let g8 = |r, g, b| pack(Format::Gray8, &[[r, g, b, 255]], 1).unwrap()[0];
        assert_eq!(g8(23, 194, 72), 168);
        assert_eq!(g8(251, 3, 134), 130);
        assert_eq!(g8(44, 165, 109), 146);
        // A neutral gray maps to itself (weights sum to 1; the transfer round-trips).
        for c in [0u8, 1, 64, 128, 200, 255] {
            assert_eq!(g8(c, c, c), c, "gray {c} must pass through");
        }
    }

    #[test]
    fn rgba64_rgb48_gray16_destinations() {
        // 16-bpc-channel destinations (native-r1 byte-exact, verified through the Wine A/B pipeline).
        let le = |c: u16| c.to_le_bytes();
        // 8bpc → 64bppRGBA: straight channel expand ×257, opaque alpha 65535 (seed 214 px0, BGR24 in).
        let rgba64 = convert(Format::Bgr24, Format::Rgba64, &[15, 253, 96], 1, 1, &[]).unwrap();
        let want: Vec<u8> = [96u16 * 257, 253 * 257, 15 * 257, 65535].iter().flat_map(|&c| le(c)).collect();
        assert_eq!(rgba64, want);
        // 8bpc → 48bppRGB: same expand, no alpha channel.
        let rgb48 = convert(Format::Bgr24, Format::Rgb48, &[15, 253, 96], 1, 1, &[]).unwrap();
        assert_eq!(rgb48, [96u16 * 257, 253 * 257, 15 * 257].iter().flat_map(|&c| le(c)).collect::<Vec<_>>());
        // RGB → 16bppGray: gamma-aware Rec.709 luma encoded to 16 bits (seed 272 px0, RGB(107,59,182)).
        let g16 = convert(Format::Bgr24, Format::Gray16, &[182, 59, 107], 1, 1, &[]).unwrap();
        assert_eq!(u16::from_le_bytes([g16[0], g16[1]]), 22472);
        // 16bppGray → 64bppRGBA: sRGB gray → linear scRGB → sRGB16 round-trip (seed 205 px0: 51250→51251).
        let o = convert(Format::Gray16, Format::Rgba64, &le(51250), 1, 1, &[]).unwrap();
        assert_eq!(u16::from_le_bytes([o[0], o[1]]), 51251);
        assert_eq!(u16::from_le_bytes([o[6], o[7]]), 65535, "opaque alpha");
        // 64bppRGBA → 16bppGray: luma on the FULL 16-bit channels, not truncated (seed 464 px0).
        let rgba: Vec<u8> = [63544u16, 48467, 23086, 65535].iter().flat_map(|&c| le(c)).collect();
        let y = convert(Format::Rgba64, Format::Gray16, &rgba, 1, 1, &[]).unwrap();
        assert_eq!(u16::from_le_bytes([y[0], y[1]]), 51008);
        // scrgb_to_srgb16 endpoints clamp.
        assert_eq!(scrgb_to_srgb16(-1.0), 0);
        assert_eq!(scrgb_to_srgb16(0.0), 0);
        assert_eq!(scrgb_to_srgb16(2.0), 0xFFFF);
    }

    #[test]
    fn bgr565_555_expand_trunc() {
        // Expand = 5/6-bit replication, pack = truncation (VM-derived byte-exact). trunc(expand(x))==x,
        // so a packed 16bpp value survives the unpack→pack round-trip; BGR555 clears bit 15.
        for v in [0u16, 0xFFFF, 0x1234, 0xABCD, 0x07E0, 0xF800, 0x001F] {
            let b = v.to_le_bytes();
            let px = unpack(Format::Bgr565, &b, 1, 1, &[]).unwrap();
            assert_eq!(pack(Format::Bgr565, &px, 1).unwrap(), b, "565 roundtrip {v:#06x}");
        }
        for v in [0u16, 0x7FFF, 0x1234, 0x7C00, 0x001F] {
            let px = unpack(Format::Bgr555, &v.to_le_bytes(), 1, 1, &[]).unwrap();
            let out = pack(Format::Bgr555, &px, 1).unwrap();
            assert_eq!(u16::from_le_bytes([out[0], out[1]]), v & 0x7FFF, "555 roundtrip {v:#06x}");
        }
        // Spot expand (565): r5=19,g6=50,b5=20 → (156,203,165) native.
        let v: u16 = (19 << 11) | (50 << 5) | 20;
        assert_eq!(unpack(Format::Bgr565, &v.to_le_bytes(), 1, 1, &[]).unwrap()[0], [156, 203, 165, 255]);
    }

    #[test]
    fn unpremultiply_16_16_reciprocal() {
        // Native's un-premultiply is a 16.16 fixed-point reciprocal of alpha, floored + clamped
        // (VM-derived byte-exact over native-r1; see `unpremul8`). Full alpha is identity, zero alpha
        // is zero, and an over-unit ratio clamps to 255.
        assert_eq!(unpremul8(0, 0), 0);
        assert_eq!(unpremul8(200, 0), 0);
        for c in [0u8, 1, 84, 128, 254, 255] {
            assert_eq!(unpremul8(c, 255), c, "full alpha must be identity");
        }
        assert_eq!(unpremul8(200, 100), 255, "ratio > 1 clamps to 255");
        // Exact-ratio boundary where plain c*255/a (=102) and the 16.16 reciprocal agree here:
        assert_eq!(unpremul8(84, 210), ((84u32 * ((255 << 16) / 210)) >> 16).min(255) as u8);
    }

    #[test]
    fn premultiply_matches_native_formula() {
        // Spot values confirmed against the real codec (rgba8 capture → 32bppPBGRA).
        assert_eq!(premul8(0x3c, 0x80), 0x1e);
        assert_eq!(premul8(0x32, 0x80), 0x19);
        assert_eq!(premul8(0x78, 0xc8), 0x5e);
        assert_eq!(premul8(0, 0), 0);
        assert_eq!(premul8(0xff, 0xff), 0xff);
    }

    #[test]
    fn colour16_truncates_gray16_gamma() {
        // 48/64-bit colour channels take the high byte (truncation).
        assert_eq!(trunc16(0x00ff), 0x00);
        assert_eq!(trunc16(0xff80), 0xff);
        // 16bppGray → 8-bit via the native sRGB↔scRGB gamma round-trip. Spot values confirmed
        // byte-for-byte against `windowscodecs` (all 65536 match; these include the boundary cases
        // where the old `round(v*255/65535)` approximation was off, e.g. 0x007f and 0x8000).
        assert_eq!(gray16_to_u8(0x0000), 0);
        assert_eq!(gray16_to_u8(0x007f), 1);
        assert_eq!(gray16_to_u8(0x0080), 1);
        assert_eq!(gray16_to_u8(0x7fff), 128);
        assert_eq!(gray16_to_u8(0x8000), 128);
        assert_eq!(gray16_to_u8(0xff80), 254);
        assert_eq!(gray16_to_u8(0xffff), 255);
    }

    #[test]
    fn subbyte_bit_replication() {
        assert_eq!(widen(1, 1), 255);
        assert_eq!(widen(2, 2), 170);
        assert_eq!(widen(0x8, 4), 0x88);
    }
}
