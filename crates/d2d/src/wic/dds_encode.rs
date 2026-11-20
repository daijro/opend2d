//! DDS container encoding, byte-for-byte as the WIC DDS encoder produces it: the `DDS ` magic, a
//! `DDS_HEADER` (optionally a `DDS_HEADER_DXT10`), then the block data. [`encode_dds`] builds the
//! header for caller-supplied blocks; [`encode_dds_bc3`] is the full source path — it converts any
//! source to 32bppBGRA and DXT5/BC3-compresses it (the WIC encoder always emits DXT5), then wraps it.
//!
//! The BC3 compressor ([`encode_dds_bc3`] and its helpers) is a byte-exact reproduction of the WIC
//! encoder's own block coder, verified against the reference codec: two iterated least-squares block
//! fits — BC1 for colour (covariance endpoint seed, projected indices, f32 2×2 refine) and BC4 for
//! alpha (min/max endpoints, 6- or 8-interpolation index, f32 refine) — over the image clamp-padded
//! to a 4×4 grid.
//!
//! Scope: a single 2D texture (the case Direct2D-adjacent apps encode); 1D/3D/cube and arrays are a
//! direct extension off the same header fields but are left out until exercised.

use super::bmp::{fmt, Decoded};
use super::convert::{convert, Format};
use super::WicError;

/// The 2D-texture subset of `WICDdsParameters` the encoder needs. `dxgi_format` is a `DXGI_FORMAT`
/// value, `alpha_mode` a `WICDdsAlphaMode` (0 = unknown/straight; non-zero forces the DX10 header).
#[derive(Debug, Clone, Copy)]
pub struct DdsParams {
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    pub dxgi_format: u32,
    pub alpha_mode: u32,
}

// DXGI_FORMAT values this encoder recognises for sizing / legacy-FourCC selection.
const BC1_UNORM: u32 = 71;
const BC2_UNORM: u32 = 74;
const BC3_UNORM: u32 = 77;
const B8G8R8X8_UNORM: u32 = 88;

/// 32-byte `DDS_PIXELFORMAT` blobs, exactly as the native constants (`DDSPF_*`).
fn ddspf_fourcc(tag: &[u8; 4]) -> [u8; 32] {
    let mut p = [0u8; 32];
    p[0] = 32; // dwSize
    p[4] = 4; // DDPF_FOURCC
    p[8..12].copy_from_slice(tag);
    p
}
fn ddspf_x8r8g8b8() -> [u8; 32] {
    let mut p = [0u8; 32];
    p[..20].copy_from_slice(&[
        32, 0, 0, 0, // dwSize
        0x40, 0, 0, 0, // DDPF_RGB
        0, 0, 0, 0, // dwFourCC
        32, 0, 0, 0, // dwRGBBitCount
        0, 0, 0xff, 0, // dwRBitMask = 0x00ff0000
    ]);
    p[20..24].copy_from_slice(&0x0000ff00u32.to_le_bytes()); // G
    p[24..28].copy_from_slice(&0x000000ffu32.to_le_bytes()); // B
    p // A mask 0
}

fn is_compressed(fmt: u32) -> bool {
    matches!(fmt, 70..=84 | 94..=99)
}

/// `ComputePitch`: returns (pitch, linear_size) for the top surface.
fn compute_pitch(fmt: u32, w: u32, h: u32) -> Result<(u32, u32), WicError> {
    if is_compressed(fmt) {
        // Bytes per 4x4 block: BC1/BC4 = 8, others = 16.
        let bb = if (70..=72).contains(&fmt) || (79..=81).contains(&fmt) { 8 } else { 16 };
        let (bw, bh) = (((w + 3) / 4).max(1), ((h + 3) / 4).max(1));
        let pitch = bb * bw;
        Ok((pitch, pitch * bh))
    } else {
        // Uncompressed: only the 32bpp formats this encoder sizes (R8G8B8A8/B8G8R8A8/B8G8R8X8).
        let bpp = match fmt {
            28 | 87 | 88 => 32,
            _ => return Err(WicError::Unsupported),
        };
        let pitch = (w * bpp + 7) / 8;
        Ok((pitch, pitch * h))
    }
}

fn put(buf: &mut [u8], dword_index: usize, v: u32) {
    buf[dword_index * 4..dword_index * 4 + 4].copy_from_slice(&v.to_le_bytes());
}

/// Encode a 2D-texture DDS file: `DDS ` magic + header (+ DX10 header when required) + `blocks`.
pub fn encode_dds(p: &DdsParams, blocks: &[u8]) -> Result<Vec<u8>, WicError> {
    let legacy_fmt =
        matches!(p.dxgi_format, BC1_UNORM | BC2_UNORM | BC3_UNORM | B8G8R8X8_UNORM);
    let use_dx10 = p.alpha_mode != 0 || !legacy_fmt;

    let mut h = [0u8; 124];
    put(&mut h, 0, 124); // dwSize
    put(&mut h, 2, p.height); // dwHeight
    put(&mut h, 3, p.width); // dwWidth
    put(&mut h, 5, 1); // dwDepth
    put(&mut h, 26, 0x1000); // dwCaps = DDSCAPS_TEXTURE
    let mut flags = 0x1007u32; // CAPS|HEIGHT|WIDTH|PIXELFORMAT
    if p.mip_levels != 0 {
        flags |= 0x20000; // DDSD_MIPMAPCOUNT
        put(&mut h, 6, p.mip_levels);
        if p.mip_levels > 1 {
            put(&mut h, 26, 0x401008); // TEXTURE|COMPLEX|MIPMAP
        }
    }

    let (pitch, linear) = compute_pitch(p.dxgi_format, p.width, p.height)?;
    if is_compressed(p.dxgi_format) {
        put(&mut h, 4, linear);
        flags |= 0x80000; // DDSD_LINEARSIZE
    } else {
        put(&mut h, 4, pitch);
        flags |= 8; // DDSD_PITCH
    }
    put(&mut h, 1, flags);

    // DDS_PIXELFORMAT at byte offset 72 (dword 18).
    let ddspf: [u8; 32] = if use_dx10 {
        ddspf_fourcc(b"DX10")
    } else {
        match p.dxgi_format {
            BC1_UNORM => ddspf_fourcc(b"DXT1"),
            BC2_UNORM => ddspf_fourcc(b"DXT3"),
            BC3_UNORM => ddspf_fourcc(b"DXT5"),
            B8G8R8X8_UNORM => ddspf_x8r8g8b8(),
            _ => return Err(WicError::Unsupported),
        }
    };
    h[72..104].copy_from_slice(&ddspf);

    let mut out = Vec::with_capacity(4 + 124 + if use_dx10 { 20 } else { 0 } + blocks.len());
    out.extend_from_slice(b"DDS "); // magic 0x20534444
    out.extend_from_slice(&h);
    if use_dx10 {
        let mut dx10 = [0u8; 20];
        put(&mut dx10, 0, p.dxgi_format);
        put(&mut dx10, 1, 3); // D3D11_RESOURCE_DIMENSION_TEXTURE2D
        put(&mut dx10, 2, 0); // miscFlag
        put(&mut dx10, 3, 1); // arraySize
        put(&mut dx10, 4, p.alpha_mode); // miscFlags2
        out.extend_from_slice(&dx10);
    }
    out.extend_from_slice(blocks);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(fmt: u32) -> DdsParams {
        DdsParams { width: 8, height: 8, mip_levels: 1, dxgi_format: fmt, alpha_mode: 0 }
    }

    #[test]
    fn bc1_legacy_header() {
        // 8×8 BC1 = 2×2 blocks × 8 bytes = 32 bytes of data.
        let out = encode_dds(&params(BC1_UNORM), &[0xAB; 32]).unwrap();
        assert_eq!(&out[0..4], b"DDS ");
        assert_eq!(&out[4..8], &124u32.to_le_bytes()); // dwSize
        assert_eq!(&out[8..12], &0xA1007u32.to_le_bytes()); // flags: CAPS|HWH|PF|MIPCOUNT|LINEARSIZE
        assert_eq!(&out[20..24], &32u32.to_le_bytes()); // linearSize = 8*2*2
        assert_eq!(&out[84..88], b"DXT1"); // ddspf.dwFourCC (offset 8+72+8)
        assert_eq!(out.len(), 4 + 124 + 32); // no DX10 header
        assert_eq!(&out[128..], &[0xAB; 32]);
    }

    #[test]
    fn bc7_uses_dx10_header() {
        let out = encode_dds(&params(98), &[0u8; 64]).unwrap(); // BC7_UNORM, 16B blocks
        assert_eq!(&out[84..88], b"DX10");
        assert_eq!(&out[128..132], &98u32.to_le_bytes()); // dxgiFormat
        assert_eq!(&out[132..136], &3u32.to_le_bytes()); // resourceDimension = 2D
        assert_eq!(out.len(), 4 + 124 + 20 + 64);
    }
}

// ---- DXT5 / BC3 block compressor (byte-exact to the WIC DDS encoder, validated vs the reference) ----
// Fixed lookup tables the encoder uses. The `ENC*` map a projection bucket to a block index; the
// `C0*/C1*` are the two barycentric weights per index for the least-squares endpoint solve.
const ENC4: [u32; 4] = [0, 2, 3, 1];
const C0_4: [i32; 4] = [3, 0, 2, 1];
const C1_4: [i32; 4] = [0, 3, 1, 2];
const ENC6: [usize; 6] = [0, 2, 3, 4, 5, 1];
const C0_6: [i32; 6] = [5, 0, 4, 3, 2, 1];
const C1_6: [i32; 6] = [0, 5, 1, 2, 3, 4];
const ENC8: [usize; 8] = [1, 7, 6, 5, 4, 3, 2, 0];
const C0_8: [i32; 8] = [7, 0, 6, 5, 4, 3, 2, 1];
const C1_8: [i32; 8] = [0, 7, 1, 2, 3, 4, 5, 6];

// BC1 colour endpoint seed: per-channel bounding box plus the sign of each channel's covariance with
// green decides which extreme goes to c0 vs c1. Channels are perceptually weighted B×1 / G×10 / R×3.
fn choose_endpoints(wb: &[i32; 16], wg: &[i32; 16], wr: &[i32; 16]) -> (u32, u32) {
    let (mut min_b, mut max_b) = (255i32, 0i32);
    let (mut min_g, mut max_g) = (2550i32, 0i32);
    let (mut min_r, mut max_r) = (765i32, 0i32);
    let (mut s_b, mut s_g, mut s_r) = (0i32, 0i32, 0i32);
    let (mut cov_bg, mut cov_gr) = (0i64, 0i64);
    for i in 0..16 {
        let (b, g, r) = (wb[i], wg[i], wr[i]);
        min_b = min_b.min(b); max_b = max_b.max(b);
        min_g = min_g.min(g); max_g = max_g.max(g);
        min_r = min_r.min(r); max_r = max_r.max(r);
        s_b += b; s_g += g; s_r += r;
        cov_bg += (b * g) as i64; cov_gr += (g * r) as i64;
    }
    let corr_bg = 16 * cov_bg >= (s_b as i64) * (s_g as i64);
    let corr_gr = 16 * cov_gr >= (s_g as i64) * (s_r as i64);
    let qb = |x: i32| ((7967 * x + 0x8000) >> 16) as u32;
    let qg = |x: i32| ((1619 * x + 0x8000) >> 16) as u32;
    let qr = |x: i32| ((2655 * x + 0x8000) >> 16) as u32;
    let c0 = qb(if corr_bg { min_b } else { max_b }) | (qg(min_g) << 5) | (qr(if corr_gr { min_r } else { max_r }) << 11);
    let c1 = qb(if corr_bg { max_b } else { min_b }) | (qg(max_g) << 5) | (qr(if corr_gr { max_r } else { min_r }) << 11);
    (c0, c1)
}

// Decode a 565 endpoint back to the weighted 8-bit channel space used for projection.
fn dec565(c: u32) -> (i32, i32, i32) {
    let (b5, g6, r5) = ((c & 0x1f) as i32, ((c >> 5) & 0x3f) as i32, ((c >> 11) & 0x1f) as i32);
    (((539086 * b5 + 0x8000) >> 16), ((2652647 * g6 + 0x8000) >> 16), ((1617259 * r5 + 0x8000) >> 16))
}

// Assign each pixel a 2-bit index by projecting it onto the c0→c1 axis (4-colour mode, c0 ≥ c1).
fn encode_colors(mut c0: u32, mut c1: u32, wb: &[i32; 16], wg: &[i32; 16], wr: &[i32; 16]) -> (u32, u32, u32) {
    if c0 < c1 {
        std::mem::swap(&mut c0, &mut c1);
    }
    if c0 == c1 {
        return (c0, c1, 0);
    }
    let (b0, g0, r0) = dec565(c0);
    let (b1, g1, r1) = dec565(c1);
    let (db, dg, dr) = (b1 - b0, g1 - g0, r1 - r0);
    let d2 = db * db + dg * dg + dr * dr;
    let mut idx = 0u32;
    for i in 0..16 {
        let proj = dg * (wg[i] - g0) + db * (wb[i] - b0) + dr * (wr[i] - r0);
        let v = if proj > d2 {
            1
        } else if proj < 0 {
            0
        } else {
            // Fixed-point ratio, computed in wrapping 32-bit as the encoder does.
            let t = (0xC000_0000u32 / d2 as u32).wrapping_mul(proj as u32).wrapping_add(0x2000_0000);
            ENC4[(t >> 30) as usize]
        };
        idx |= v << (2 * i);
    }
    (c0, c1, idx)
}

// Quantise a weighted (B, 10·G, 3·R) colour to 565, rounding to nearest and saturating.
fn pack_saturate(b: f32, g: f32, r: f32) -> u32 {
    let q = |x: f32, lt: f32, fac: f32, mx: u32| -> u32 {
        if x > 0.0 {
            if x >= lt { mx } else { (x * fac + 0.5) as u32 }
        } else {
            0
        }
    };
    let b5 = q(b, 255.0, 0.121_568_63, 31);
    let g6 = q(g, 2550.0, 0.024_705_883, 63);
    let r5 = q(r, 765.0, 0.040_522_877, 31);
    b5 | (g6 << 5) | (r5 << 11)
}

// Given fixed indices, solve the 2×2 normal equations for the optimal endpoints (f32, ×3/det).
fn optimize_colors(idx: u32, wb: &[i32; 16], wg: &[i32; 16], wr: &[i32; 16]) -> Option<(u32, u32)> {
    let (mut a00, mut a01, mut a11) = (0i32, 0i32, 0i32);
    let (mut cb, mut cg, mut cr) = ([0i32; 2], [0i32; 2], [0i32; 2]);
    for i in 0..16 {
        let k = ((idx >> (2 * i)) & 3) as usize;
        let (c0, c1) = (C0_4[k], C1_4[k]);
        a00 += c0 * c0; a01 += c0 * c1; a11 += c1 * c1;
        cb[0] += c0 * wb[i]; cb[1] += c1 * wb[i];
        cg[0] += c0 * wg[i]; cg[1] += c1 * wg[i];
        cr[0] += c0 * wr[i]; cr[1] += c1 * wr[i];
    }
    let det = a00 * a11 - a01 * a01;
    if det == 0 {
        return None;
    }
    let f = 3.0f32 / det as f32;
    let e = |x0: i32, x1: i32, first: bool| -> f32 {
        if first { (a11 * x0 - a01 * x1) as f32 * f } else { (a00 * x1 - a01 * x0) as f32 * f }
    };
    let c0 = pack_saturate(e(cb[0], cb[1], true), e(cg[0], cg[1], true), e(cr[0], cr[1], true));
    let c1 = pack_saturate(e(cb[0], cb[1], false), e(cg[0], cg[1], false), e(cr[0], cr[1], false));
    Some((c0, c1))
}

fn encode_color_block(px: &[[u8; 4]; 16]) -> (u32, u32, u32) {
    let mut wb = [0i32; 16];
    let mut wg = [0i32; 16];
    let mut wr = [0i32; 16];
    for i in 0..16 {
        wb[i] = px[i][0] as i32;
        wg[i] = 10 * px[i][1] as i32;
        wr[i] = 3 * px[i][2] as i32;
    }
    let (mut c0, mut c1) = choose_endpoints(&wb, &wg, &wr);
    let mut idx;
    (c0, c1, idx) = encode_colors(c0, c1, &wb, &wg, &wr);
    if c0 == c1 {
        return (c0, c1, 0);
    }
    // Refine up to three times; stop when neither the endpoint pair nor the indices change.
    for _ in 0..3 {
        let prev = (c0, c1);
        match optimize_colors(idx, &wb, &wg, &wr) {
            None => break,
            Some((n0, n1)) => {
                c0 = n0;
                c1 = n1;
            }
        }
        if (c0, c1) == prev {
            break;
        }
        if c0 == c1 {
            return (c0, c1, 0);
        }
        let pidx = idx;
        (c0, c1, idx) = encode_colors(c0, c1, &wb, &wg, &wr);
        if idx == pidx {
            break;
        }
    }
    (c0, c1, idx)
}

// BC4 alpha, 6-interpolation mode (endpoints a0=min, a1=max; indices 6/7 are an explicit 0/255).
fn encode_alphas6(a: &[i32; 16], amin: i32, amax: i32) -> u64 {
    let mut idx = 0u64;
    for (i, &v) in a.iter().enumerate() {
        let k: u64 = if v >= amin {
            if v <= amax {
                ENC6[(((0x50000u32 / (amax - amin) as u32) * (v - amin) as u32 + 0x8000) >> 16) as usize] as u64
            } else if (amax + 255) >> 1 < v {
                7
            } else {
                1
            }
        } else if amin >> 1 >= v {
            6
        } else {
            0
        };
        idx |= k << (3 * i);
    }
    idx
}

// BC4 alpha, 8-interpolation mode (endpoints a0=max, a1=min).
fn encode_alphas8(a: &[i32; 16], amax: i32, amin: i32) -> u64 {
    let mut idx = 0u64;
    for (i, &v) in a.iter().enumerate() {
        let k: u64 = if v >= amin {
            if v <= amax {
                ENC8[(((0x70000u32 / (amax - amin) as u32) * (v - amin) as u32 + 0x8000) >> 16) as usize] as u64
            } else {
                0
            }
        } else {
            1
        };
        idx |= k << (3 * i);
    }
    idx
}

fn alpha_clamp(x: f32) -> i32 {
    if x >= 0.0 {
        if x > 255.0 { 255 } else { (x + 0.5) as i32 }
    } else {
        0
    }
}

// Least-squares alpha endpoints given fixed indices (f32, ×denom/det). In 6-interpolation mode the
// explicit-0/255 indices (6, 7) are excluded from the fit.
fn optimize_alpha(a: &[i32; 16], idx: u64, c0t: &[i32], c1t: &[i32], denom: f32, mode6: bool) -> Option<(i32, i32)> {
    let (mut a00, mut a01, mut a11) = (0i32, 0i32, 0i32);
    let (mut b0, mut b1) = (0i32, 0i32);
    for i in 0..16 {
        let k = ((idx >> (3 * i)) & 7) as usize;
        if mode6 && k >= 6 {
            continue;
        }
        let (c0, c1) = (c0t[k], c1t[k]);
        a00 += c0 * c0; a01 += c0 * c1; a11 += c1 * c1;
        b0 += c0 * a[i]; b1 += c1 * a[i];
    }
    let det = a00 * a11 - a01 * a01;
    if det == 0 {
        return None;
    }
    let f = denom / det as f32;
    Some((alpha_clamp((a11 * b0 - a01 * b1) as f32 * f), alpha_clamp((a00 * b1 - a01 * b0) as f32 * f)))
}

fn refine_alpha(mut a0: i32, mut a1: i32, a: &[i32; 16], mode6: bool) -> (i32, i32, u64) {
    let enc = |a0: i32, a1: i32| if mode6 { encode_alphas6(a, a0, a1) } else { encode_alphas8(a, a0, a1) };
    let (c0t, c1t, denom): (&[i32], &[i32], f32) =
        if mode6 { (&C0_6, &C1_6, 5.0) } else { (&C0_8, &C1_8, 7.0) };
    let mut idx = enc(a0, a1);
    for _ in 0..3 {
        let prev = (a0, a1);
        match optimize_alpha(a, idx, c0t, c1t, denom, mode6) {
            None => break,
            Some((n0, n1)) => {
                a0 = n0;
                a1 = n1;
            }
        }
        if (a0, a1) == prev {
            break;
        }
        if a0 == a1 {
            return (a0, a1, 0);
        }
        let pidx = idx;
        idx = enc(a0, a1);
        if idx == pidx {
            break;
        }
    }
    (a0, a1, idx)
}

// BC4 alpha block → (a0, a1, 48-bit index). Endpoints come from the block's min/max of the mid
// (non-0/255) alpha values; which interpolation mode is used depends on whether 0/255 are present.
fn encode_alpha_block(a: &[i32; 16]) -> (u8, u8, u64) {
    let (mut flags, mut amin, mut amax) = (0u32, 0i32, 0i32);
    for &v in a {
        if v == 0 {
            flags |= 1;
        } else if v == 0xFF {
            flags |= 2;
        } else if flags & 4 != 0 {
            amin = amin.min(v);
            amax = amax.max(v);
        } else {
            amin = v;
            amax = v;
            flags |= 4;
        }
    }
    if flags > 3 {
        if amin == amax {
            return (amin as u8, amax as u8, 0); // one distinct mid value → solid
        }
        let (a0, a1, idx) = if flags & 3 != 0 {
            refine_alpha(amin, amax, a, true) // 0/255 present → 6-interpolation
        } else {
            refine_alpha(amax, amin, a, false) // mid only → 8-interpolation
        };
        (a0 as u8, a1 as u8, idx)
    } else if flags == 1 {
        (0, 0, 0) // all transparent
    } else if flags == 2 {
        (255, 255, 0) // all opaque
    } else {
        // Only 0 and 255: endpoints (255,255), the explicit-0 index (6) for every transparent texel.
        let mut idx = 0u64;
        for (i, &v) in a.iter().enumerate() {
            if v == 0 {
                idx |= 6u64 << (3 * i);
            }
        }
        (255, 255, idx)
    }
}

// One 16-byte BC3 block for a 4×4 BGRA tile: 8 bytes BC4 alpha then 8 bytes BC1 colour.
fn encode_bc3_block(px: &[[u8; 4]; 16]) -> [u8; 16] {
    let (c0, c1, cidx) = encode_color_block(px);
    let mut a = [0i32; 16];
    for i in 0..16 {
        a[i] = px[i][3] as i32;
    }
    let (a0, a1, aidx) = encode_alpha_block(&a);
    let mut out = [0u8; 16];
    out[0] = a0;
    out[1] = a1;
    out[2..8].copy_from_slice(&aidx.to_le_bytes()[..6]);
    out[8..10].copy_from_slice(&(c0 as u16).to_le_bytes());
    out[10..12].copy_from_slice(&(c1 as u16).to_le_bytes());
    out[12..16].copy_from_slice(&cidx.to_le_bytes());
    out
}

/// Encode any source `Decoded` to a DXT5/BC3 DDS, byte-for-byte as the WIC DDS encoder does: convert
/// to 32bppBGRA, clamp-pad to a 4×4 grid, BC3-compress each block row-major, wrap in the header.
pub fn encode_dds_bc3(img: &Decoded) -> Result<Vec<u8>, WicError> {
    let (w, h) = (img.width as usize, img.height as usize);
    if w == 0 || h == 0 {
        return Err(WicError::Unsupported);
    }
    let sf = Format::from_guid(&img.format).ok_or(WicError::Unsupported)?;
    let bgra = convert(sf, Format::Bgra32, &img.pixels, img.width, img.height, &img.palette)?;
    let at = |x: usize, y: usize| -> [u8; 4] {
        let (cx, cy) = (x.min(w - 1), y.min(h - 1)); // clamp-to-edge padding
        let o = (cy * w + cx) * 4;
        [bgra[o], bgra[o + 1], bgra[o + 2], bgra[o + 3]]
    };
    let (bw, bh) = (w.div_ceil(4), h.div_ceil(4));
    let mut blocks = Vec::with_capacity(bw * bh * 16);
    for by in 0..bh {
        for bx in 0..bw {
            let mut px = [[0u8; 4]; 16];
            for yy in 0..4 {
                for xx in 0..4 {
                    px[yy * 4 + xx] = at(bx * 4 + xx, by * 4 + yy);
                }
            }
            blocks.extend_from_slice(&encode_bc3_block(&px));
        }
    }
    encode_dds(
        &DdsParams { width: img.width, height: img.height, mip_levels: 1, dxgi_format: BC3_UNORM, alpha_mode: 0 },
        &blocks,
    )
}

#[cfg(test)]
mod bc3_tests {
    use super::*;

    #[test]
    fn opaque_grey_is_dxt5() {
        // 4×4 opaque mid-grey: alpha all 255 → (255,255,0); one colour → c0==c1 → zero indices.
        let img = Decoded {
            width: 4,
            height: 4,
            format: fmt::BGRA32.into(),
            stride: 16,
            pixels: vec![128, 128, 128, 255].repeat(16),
            palette: vec![],
        };
        let dds = encode_dds_bc3(&img).unwrap();
        assert_eq!(&dds[..4], b"DDS ");
        assert_eq!(&dds[84..88], b"DXT5");
        assert_eq!(dds.len(), 128 + 16);
        assert_eq!(&dds[128..136], &[255, 255, 0, 0, 0, 0, 0, 0]);
    }
}
