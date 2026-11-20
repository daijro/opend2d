//! Baseline JPEG encoding, byte-identical to WIC's `IWICBitmapEncoder`.
//!
//! WIC's JPEG encoder is libjpeg-turbo (its frame-header setup calls
//! `jpeg_set_defaults` + `jpeg_set_quality(90, force_baseline)`): baseline sequential, standard
//! Huffman tables, quality 90, 4:2:0 subsampling by default (its ChromaSubsampling option picks
//! 4:2:0/4:2:2/4:4:4/4:4:0). Verified: the output is byte-for-byte identical to libjpeg-turbo
//! 3.2.0's `cjpeg -quality <q> -sample <s>` for the DQT/DHT/SOF/SOS/entropy stream; the only
//! WIC-specific bytes are the JFIF APP0 density (unit = 1 "inches", X/Y = the source DPI), which
//! this writes. So this reproduces that encoder in fresh Rust (fdct islow, quality→quant scaling,
//! IJG forward DCT quantisation rounding, standard Huffman, JFIF markers).

use super::WicFormat;

use std::num::Wrapping;
type W = Wrapping<i32>;

/// zig-zag scan order → natural (row-major) 8×8 index (same table the decoder uses).
#[rustfmt::skip]
const ZIGZAG: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// IJG standard luminance / chrominance base quantization tables (natural order).
#[rustfmt::skip]
const STD_LUMA_QUANT: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];
#[rustfmt::skip]
const STD_CHROMA_QUANT: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

// IJG standard Huffman tables (Annex K.3): counts per code length 1..16, then the symbol list.
const DC_LUMA_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const DC_LUMA_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const DC_CHROMA_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const DC_CHROMA_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const AC_LUMA_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
#[rustfmt::skip]
const AC_LUMA_VALS: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];
const AC_CHROMA_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
#[rustfmt::skip]
const AC_CHROMA_VALS: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

const CONST_BITS: u32 = 13;
const PASS1_BITS: u32 = 2;
/// islow fixed-point constants `FIX(x) = round(x·2¹³)`, the same set the IDCT uses.
const FIX: [i32; 12] =
    [2446, 3196, 4433, 6270, 7373, 9633, 12299, 15137, 16069, 16819, 20995, 25172];

#[inline]
fn descale(x: W, n: u32) -> W {
    (x + Wrapping(1 << (n - 1))) >> n as usize
}

/// jpeg_fdct_islow (jfdctint.c): forward DCT of level-shifted 8×8 `blk` (row-major), output scaled
/// up by 8 as libjpeg leaves it for the quantiser.
fn fdct_islow(blk: &mut [W; 64]) {
    let m = |a: W, b: i32| a * Wrapping(b);
    // Pass 1: rows.
    for r in 0..8 {
        let d = &mut blk[r * 8..r * 8 + 8];
        let tmp0 = d[0] + d[7];
        let tmp7 = d[0] - d[7];
        let tmp1 = d[1] + d[6];
        let tmp6 = d[1] - d[6];
        let tmp2 = d[2] + d[5];
        let tmp5 = d[2] - d[5];
        let tmp3 = d[3] + d[4];
        let tmp4 = d[3] - d[4];
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;
        d[0] = (tmp10 + tmp11) << PASS1_BITS as usize;
        d[4] = (tmp10 - tmp11) << PASS1_BITS as usize;
        let z1 = m(tmp12 + tmp13, FIX[2]);
        d[2] = descale(z1 + m(tmp13, FIX[3]), CONST_BITS - PASS1_BITS);
        d[6] = descale(z1 + m(tmp12, -FIX[7]), CONST_BITS - PASS1_BITS);
        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = m(z3 + z4, FIX[5]);
        let t4 = m(tmp4, FIX[0]);
        let t5 = m(tmp5, FIX[9]);
        let t6 = m(tmp6, FIX[11]);
        let t7 = m(tmp7, FIX[6]);
        let z1 = m(z1, -FIX[4]);
        let z2 = m(z2, -FIX[10]);
        let z3 = m(z3, -FIX[8]) + z5;
        let z4 = m(z4, -FIX[1]) + z5;
        d[7] = descale(t4 + z1 + z3, CONST_BITS - PASS1_BITS);
        d[5] = descale(t5 + z2 + z4, CONST_BITS - PASS1_BITS);
        d[3] = descale(t6 + z2 + z3, CONST_BITS - PASS1_BITS);
        d[1] = descale(t7 + z1 + z4, CONST_BITS - PASS1_BITS);
    }
    // Pass 2: columns.
    for c in 0..8 {
        let g = |i: usize| blk[c + i * 8];
        let tmp0 = g(0) + g(7);
        let tmp7 = g(0) - g(7);
        let tmp1 = g(1) + g(6);
        let tmp6 = g(1) - g(6);
        let tmp2 = g(2) + g(5);
        let tmp5 = g(2) - g(5);
        let tmp3 = g(3) + g(4);
        let tmp4 = g(3) - g(4);
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;
        blk[c] = descale(tmp10 + tmp11, PASS1_BITS);
        blk[c + 32] = descale(tmp10 - tmp11, PASS1_BITS);
        let z1 = m(tmp12 + tmp13, FIX[2]);
        blk[c + 16] = descale(z1 + m(tmp13, FIX[3]), CONST_BITS + PASS1_BITS);
        blk[c + 48] = descale(z1 + m(tmp12, -FIX[7]), CONST_BITS + PASS1_BITS);
        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = m(z3 + z4, FIX[5]);
        let t4 = m(tmp4, FIX[0]);
        let t5 = m(tmp5, FIX[9]);
        let t6 = m(tmp6, FIX[11]);
        let t7 = m(tmp7, FIX[6]);
        let z1 = m(z1, -FIX[4]);
        let z2 = m(z2, -FIX[10]);
        let z3 = m(z3, -FIX[8]) + z5;
        let z4 = m(z4, -FIX[1]) + z5;
        blk[c + 56] = descale(t4 + z1 + z3, CONST_BITS + PASS1_BITS);
        blk[c + 40] = descale(t5 + z2 + z4, CONST_BITS + PASS1_BITS);
        blk[c + 24] = descale(t6 + z2 + z3, CONST_BITS + PASS1_BITS);
        blk[c + 8] = descale(t7 + z1 + z4, CONST_BITS + PASS1_BITS);
    }
}

/// IJG forward-DCT quantisation: `round(coef / (quant·8))`, half rounded away from zero — matching
/// libjpeg's reciprocal quantiser bit-for-bit.
#[inline]
fn quantize(coef: i32, quant: u16) -> i32 {
    let q = (quant as i32) << 3;
    if coef < 0 {
        -(((-coef) + (q >> 1)) / q)
    } else {
        (coef + (q >> 1)) / q
    }
}

/// Scale a base quantization table by quality (`jpeg_quality_scaling` + `jpeg_add_quant_table`,
/// force_baseline so values clamp to 1..=255).
fn scale_quant(base: &[u16; 64], quality: u32) -> [u16; 64] {
    let q = quality.clamp(1, 100);
    let scale = if q < 50 { 5000 / q } else { 200 - q * 2 } as i32;
    let mut out = [0u16; 64];
    for (o, &b) in out.iter_mut().zip(base.iter()) {
        *o = (((b as i32 * scale) + 50) / 100).clamp(1, 255) as u16;
    }
    out
}

/// A canonical Huffman code table for encoding: `(code, size)` per symbol value (0..=255).
struct Ehuff {
    code: [u16; 256],
    size: [u8; 256],
}

impl Ehuff {
    fn build(bits: &[u8; 16], vals: &[u8]) -> Ehuff {
        let mut e = Ehuff { code: [0; 256], size: [0; 256] };
        let (mut code, mut k) = (0u16, 0usize);
        for l in 0..16 {
            for _ in 0..bits[l] {
                e.code[vals[k] as usize] = code;
                e.size[vals[k] as usize] = l as u8 + 1;
                code += 1;
                k += 1;
            }
            code <<= 1;
        }
        e
    }
}

/// MSB-first entropy bit writer with 0xFF byte-stuffing.
struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    bits: u32,
}

impl BitWriter {
    fn put(&mut self, code: u32, size: u32) {
        self.acc = (self.acc << size) | (code & ((1u32 << size) - 1));
        self.bits += size;
        while self.bits >= 8 {
            self.bits -= 8;
            let b = ((self.acc >> self.bits) & 0xFF) as u8;
            self.out.push(b);
            if b == 0xFF {
                self.out.push(0x00);
            }
        }
    }

    /// Pad the final partial byte with 1-bits (libjpeg's `flush_bits`).
    fn flush(&mut self) {
        if self.bits > 0 {
            let pad = 8 - self.bits;
            self.put((1 << pad) - 1, pad);
        }
    }
}

/// Magnitude category (number of bits) of a nonzero coefficient difference.
#[inline]
fn category(v: i32) -> u32 {
    32 - v.unsigned_abs().leading_zeros()
}

/// Huffman-encode one quantized 8×8 block (natural order) against its DC/AC tables, updating the
/// DC predictor — libjpeg's `encode_one_block`.
fn encode_block(bw: &mut BitWriter, blk: &[i32; 64], pred: &mut i32, dc: &Ehuff, ac: &Ehuff) {
    // DC difference.
    let diff = blk[0] - *pred;
    *pred = blk[0];
    let n = category(diff);
    bw.put(dc.code[n as usize] as u32, dc.size[n as usize] as u32);
    if n > 0 {
        let v = if diff < 0 { diff - 1 } else { diff };
        bw.put((v as u32) & ((1 << n) - 1), n);
    }
    // AC coefficients in zig-zag order, run-length + magnitude.
    let mut run = 0u32;
    for k in 1..64 {
        let v = blk[ZIGZAG[k]];
        if v == 0 {
            run += 1;
            continue;
        }
        while run > 15 {
            bw.put(ac.code[0xF0] as u32, ac.size[0xF0] as u32); // ZRL
            run -= 16;
        }
        let n = category(v);
        let sym = (run << 4) | n;
        bw.put(ac.code[sym as usize] as u32, ac.size[sym as usize] as u32);
        let m = if v < 0 { v - 1 } else { v };
        bw.put((m as u32) & ((1 << n) - 1), n);
        run = 0;
    }
    if run > 0 {
        bw.put(ac.code[0x00] as u32, ac.size[0x00] as u32); // EOB
    }
}

fn be16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_be_bytes());
}

/// Chroma subsampling, matching WIC's `WICJpegYCrCbSubsampling*` option (default 4:2:0).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Subsampling {
    S420,
    S422,
    S444,
    S440,
}

impl Subsampling {
    fn factors(self) -> (usize, usize) {
        match self {
            Subsampling::S420 => (2, 2),
            Subsampling::S422 => (2, 1),
            Subsampling::S444 => (1, 1),
            Subsampling::S440 => (1, 2),
        }
    }
}

struct Plane {
    data: Vec<u8>,
    w: usize,
    h: usize,
}

impl Plane {
    #[inline]
    fn at(&self, x: isize, y: isize) -> i32 {
        let xx = x.clamp(0, self.w as isize - 1) as usize;
        let yy = y.clamp(0, self.h as isize - 1) as usize;
        self.data[yy * self.w + xx] as i32
    }
}

/// Encode a 24bpp BGR (`WicFormat::Bgr24`) or 8bpp gray image to a baseline JPEG, byte-identical to
/// WIC's encoder. `quality` 1..=100 (WIC default 90); `dpi` is written into the JFIF APP0 density
/// (unit = inches, as WIC does — pass the source bitmap's DPI, e.g. (96, 96), or (0, 0) for none).
pub fn encode_jpeg(
    format: WicFormat,
    width: u32,
    height: u32,
    pixels: &[u8],
    quality: u32,
    subsampling: Subsampling,
    dpi: (u16, u16),
) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let gray = format == WicFormat::Gray8;
    let (sh, sv) = if gray { (1, 1) } else { subsampling.factors() };

    // Colour-convert to full-resolution component planes (Y, or Y/Cb/Cr).
    let mut planes: Vec<Plane> = Vec::new();
    if gray {
        planes.push(Plane { data: pixels[..w * h].to_vec(), w, h });
    } else {
        let (mut y, mut cb, mut cr) = (vec![0u8; w * h], vec![0u8; w * h], vec![0u8; w * h]);
        for i in 0..w * h {
            let b = pixels[i * 3] as i32;
            let g = pixels[i * 3 + 1] as i32;
            let r = pixels[i * 3 + 2] as i32;
            y[i] = ((19595 * r + 38470 * g + 7471 * b + 32768) >> 16) as u8;
            cb[i] = ((-11059 * r - 21709 * g + 32768 * b + (128 << 16) + 32767) >> 16) as u8;
            cr[i] = ((32768 * r - 27439 * g - 5329 * b + (128 << 16) + 32767) >> 16) as u8;
        }
        planes.push(Plane { data: y, w, h });
        planes.push(Plane { data: cb, w, h });
        planes.push(Plane { data: cr, w, h });
    }

    // MCU grid: luma at (sh,sv), chroma at (1,1). Image is padded to whole MCUs by edge replication.
    let mcus_x = w.div_ceil(sh * 8);
    let mcus_y = h.div_ceil(sv * 8);

    // Downsample chroma (h2v* box filter with libjpeg's alternating 1/2 bias) to its block grid.
    // The luma plane is used at full resolution; a non-max component is averaged `sh`×`sv`.
    struct Comp {
        plane: Plane, // sample plane at this component's resolution, edge-padded to its block grid
        quant: [u16; 64],
        wib: usize, // real blocks wide / tall; blocks beyond these in the MCU grid are dummies
        hib: usize,
        ch: usize, // sampling factors (MCU block layout)
        cv: usize,
    }
    let luma_q = scale_quant(&STD_LUMA_QUANT, quality);
    let chroma_q = scale_quant(&STD_CHROMA_QUANT, quality);

    let mut comps: Vec<Comp> = Vec::new();
    for (ci, p) in planes.iter().enumerate() {
        let (ch, cv) = if ci == 0 { (sh, sv) } else { (1, 1) };
        let (pw, ph) = (mcus_x * ch * 8, mcus_y * cv * 8); // padded sample plane for this component
        let mut data = vec![0u8; pw * ph];
        let dx = sh / ch; // downsample ratio relative to luma
        let dy = sv / cv;
        // Columns: libjpeg expand_right_edge replicates the last *column* of the full-res input, so
        // padded columns re-downsample the clamped edge (`p.at` clamps x). Rows: the prep controller
        // replicates the last *row group*, so padded output rows copy the last real downsampled row
        // — clamp the source row to the real downsampled height.
        let cdh = h.div_ceil(dy);
        for oy in 0..ph {
            let sy = oy.min(cdh - 1);
            for ox in 0..pw {
                data[oy * pw + ox] = match (dx, dy) {
                    (1, 1) => p.at(ox as isize, sy as isize) as u8,
                    // libjpeg h2v2_downsample: 2×2 box, bias alternating 1,2 per column, ÷4.
                    (2, 2) => {
                        let s = p.at((ox * 2) as isize, (sy * 2) as isize)
                            + p.at((ox * 2 + 1) as isize, (sy * 2) as isize)
                            + p.at((ox * 2) as isize, (sy * 2 + 1) as isize)
                            + p.at((ox * 2 + 1) as isize, (sy * 2 + 1) as isize);
                        ((s + 1 + (ox & 1) as i32) >> 2) as u8
                    }
                    // h2v1_downsample: 1×2 box, bias alternating 0,1 per column, ÷2.
                    (2, 1) => {
                        let s = p.at((ox * 2) as isize, sy as isize)
                            + p.at((ox * 2 + 1) as isize, sy as isize);
                        ((s + (ox & 1) as i32) >> 1) as u8
                    }
                    // other ratios: libjpeg int_downsample, fixed +numpix/2 rounding, no alternation.
                    _ => {
                        let mut sum = 0i32;
                        for j in 0..dy {
                            for i in 0..dx {
                                sum += p.at((ox * dx + i) as isize, (sy * dy + j) as isize);
                            }
                        }
                        let numpix = (dx * dy) as i32;
                        ((sum + numpix / 2) / numpix) as u8
                    }
                };
            }
        }
        comps.push(Comp {
            plane: Plane { data, w: pw, h: ph },
            quant: if ci == 0 { luma_q } else { chroma_q },
            wib: (w * ch).div_ceil(sh).div_ceil(8),
            hib: (h * cv).div_ceil(sv).div_ceil(8),
            ch,
            cv,
        });
    }

    // --- Markers ---
    let mut out = Vec::new();
    out.extend_from_slice(&[0xFF, 0xD8]); // SOI
    // JFIF APP0: version 1.1, density unit = inches (WIC), X/Y = DPI, no thumbnail.
    out.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x10]);
    out.extend_from_slice(b"JFIF\0");
    out.extend_from_slice(&[0x01, 0x01, 0x01]);
    be16(&mut out, dpi.0);
    be16(&mut out, dpi.1);
    out.extend_from_slice(&[0x00, 0x00]);
    // DQT (one per distinct table), written in zig-zag order.
    let dqt = |out: &mut Vec<u8>, id: u8, q: &[u16; 64]| {
        out.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x43, id]);
        for k in 0..64 {
            out.push(q[ZIGZAG[k]] as u8);
        }
    };
    dqt(&mut out, 0, &luma_q);
    if !gray {
        dqt(&mut out, 1, &chroma_q);
    }
    // SOF0 baseline.
    let nc = comps.len() as u8;
    out.extend_from_slice(&[0xFF, 0xC0]);
    be16(&mut out, 8 + 3 * nc as u16);
    out.push(8);
    be16(&mut out, height as u16);
    be16(&mut out, width as u16);
    out.push(nc);
    for (ci, _) in comps.iter().enumerate() {
        let (ch, cv) = if ci == 0 { (sh, sv) } else { (1, 1) };
        let tq = if ci == 0 { 0 } else { 1 };
        out.extend_from_slice(&[ci as u8 + 1, ((ch as u8) << 4) | cv as u8, tq]);
    }
    // DHT (standard tables).
    let dht = |out: &mut Vec<u8>, class_id: u8, bits: &[u8; 16], vals: &[u8]| {
        out.extend_from_slice(&[0xFF, 0xC4]);
        be16(out, 19 + vals.len() as u16);
        out.push(class_id);
        out.extend_from_slice(bits);
        out.extend_from_slice(vals);
    };
    dht(&mut out, 0x00, &DC_LUMA_BITS, &DC_LUMA_VALS);
    dht(&mut out, 0x10, &AC_LUMA_BITS, &AC_LUMA_VALS);
    if !gray {
        dht(&mut out, 0x01, &DC_CHROMA_BITS, &DC_CHROMA_VALS);
        dht(&mut out, 0x11, &AC_CHROMA_BITS, &AC_CHROMA_VALS);
    }
    // SOS.
    out.extend_from_slice(&[0xFF, 0xDA]);
    be16(&mut out, 6 + 2 * nc as u16);
    out.push(nc);
    for ci in 0..comps.len() {
        let td = if ci == 0 { 0x00 } else { 0x11 }; // DC table | AC table nibbles
        out.extend_from_slice(&[ci as u8 + 1, td]);
    }
    out.extend_from_slice(&[0x00, 0x3F, 0x00]); // Ss=0, Se=63, Ah=Al=0

    // --- Entropy-coded scan (interleaved MCUs) ---
    let dc_l = Ehuff::build(&DC_LUMA_BITS, &DC_LUMA_VALS);
    let ac_l = Ehuff::build(&AC_LUMA_BITS, &AC_LUMA_VALS);
    let dc_c = Ehuff::build(&DC_CHROMA_BITS, &DC_CHROMA_VALS);
    let ac_c = Ehuff::build(&AC_CHROMA_BITS, &AC_CHROMA_VALS);
    let mut bw = BitWriter { out: Vec::new(), acc: 0, bits: 0 };
    let mut pred = vec![0i32; comps.len()];
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            for (ci, c) in comps.iter().enumerate() {
                let (dc, ac) = if ci == 0 { (&dc_l, &ac_l) } else { (&dc_c, &ac_c) };
                for by in 0..c.cv {
                    for bx in 0..c.ch {
                        let (bcol, brow) = (mx * c.ch + bx, my * c.cv + by);
                        let mut q = [0i32; 64];
                        if bcol < c.wib && brow < c.hib {
                            // Real block: forward-DCT the (edge-expanded) samples and quantize.
                            let (px, py) = (bcol * 8, brow * 8);
                            let mut blk = [Wrapping(0); 64];
                            for r in 0..8 {
                                for col in 0..8 {
                                    let s = c.plane.data[(py + r) * c.plane.w + px + col] as i32;
                                    blk[r * 8 + col] = Wrapping(s - 128);
                                }
                            }
                            fdct_islow(&mut blk);
                            for i in 0..64 {
                                q[i] = quantize(blk[i].0, c.quant[i]);
                            }
                        } else {
                            // Dummy block (beyond the real block grid): zero AC and DC equal to the
                            // running predictor, so its DC difference is 0 — libjpeg's cheapest fill.
                            q[0] = pred[ci];
                        }
                        encode_block(&mut bw, &q, &mut pred[ci], dc, ac);
                    }
                }
            }
        }
    }
    bw.flush();
    out.extend_from_slice(&bw.out);
    out.extend_from_slice(&[0xFF, 0xD9]); // EOI
    out
}
