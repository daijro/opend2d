//! Baseline JPEG → WIC native frame, as WIC (libjpeg-turbo) decodes it.
//!
//! WIC's JPEG codec is libjpeg-turbo with the entropy/marker machinery intact and the lossy
//! back-end swapped for IPP primitives that reproduce IJG's algorithms exactly: the islow integer
//! IDCT (`ippiDCTQuantInv8x8LS_IJG_JPEG`), fancy triangle chroma upsampling
//! (`ippiSampleUpRow{H2V1,H2V2}_Triangle_JPEG`) and a 14-bit-fixed-point YCbCr→BGR
//! (`ippiYCbCrToBGR_JPEG`, constants read from the binary below). Verified: `djpeg` from
//! libjpeg-turbo 3.2.0 is byte-identical to WIC across the whole corpus, so this reproduces that
//! default decode. `GetPixelFormat` reports 8bppGray for grayscale and 24bppBGR for YCbCr/RGB.
//!
//! Scope: baseline sequential (SOF0) and progressive (SOF2), grayscale / YCbCr / RGB / CMYK / YCCK,
//! every subsampling, restart markers and EXIF orientation (metadata only — pixels are not rotated).
//!
//! Rejections reproduce WIC's exact `WINCODEC_ERR_*` HRESULTs (captured from native WIC
//! on the reference VM; see tests/wic_jpeg/errors.json). WIC reads the header lazily and runs the
//! libjpeg decompress at `CopyPixels`; every libjpeg error there raises one SEH code that WIC maps
//! to an HRESULT per call site. `decode_jpeg` is a single call, so it returns the HRESULT whatever
//! WIC's call site:
//!   * [`WINCODEC_ERR_COMPONENTNOTFOUND`] — not a JPEG (no SOI); WIC matches no codec (CreateDecoder).
//!   * [`WINCODEC_ERR_FRAMEMISSING`] — SOI present but no complete, non-zero-size frame (CreateDecoder).
//!   * [`WINCODEC_ERR_BADHEADER`] — the frame parses but can't be decompressed: data precision ≠ 8,
//!     an undefined quantization table, or an unsupported SOF (CopyPixels).
//! Truncated entropy data, even a missing EOI, is tolerated and decodes, exactly as libjpeg does.

use super::{DecodedImage, WicFormat};
use std::num::Wrapping;

/// Stream is not a JPEG (no SOI marker); WIC can select no codec.
pub const WINCODEC_ERR_COMPONENTNOTFOUND: u32 = 0x8898_2F50;
/// SOI present, but no complete frame header with non-zero dimensions.
pub const WINCODEC_ERR_FRAMEMISSING: u32 = 0x8898_2F61;
/// Frame header parses but the image cannot be decompressed.
pub const WINCODEC_ERR_BADHEADER: u32 = 0x8898_2F60;

type W = Wrapping<i32>;

/// zig-zag index → natural (row-major) 8×8 index.
#[rustfmt::skip]
const ZIGZAG: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

struct Huff {
    mincode: [i32; 17],
    maxcode: [i32; 18], // [17] = sentinel 0xFFFFF
    valptr: [i32; 17],
    values: Vec<u8>,
}

impl Huff {
    /// Build decode tables from the DHT `bits` (counts per code length 1..16) and symbol list.
    fn build(bits: &[u8; 16], values: Vec<u8>) -> Huff {
        let mut huffsize = [0u8; 257];
        let mut k = 0;
        for l in 0..16 {
            for _ in 0..bits[l] {
                huffsize[k] = l as u8 + 1;
                k += 1;
            }
        }
        let mut huffcode = [0i32; 257];
        let (mut code, mut si, mut p) = (0i32, huffsize[0] as i32, 0usize);
        while huffsize[p] != 0 {
            while huffsize[p] as i32 == si {
                huffcode[p] = code;
                code += 1;
                p += 1;
            }
            code <<= 1;
            si += 1;
        }
        let mut h = Huff { mincode: [0; 17], maxcode: [0; 18], valptr: [0; 17], values };
        p = 0;
        for l in 1..=16 {
            if bits[l - 1] != 0 {
                h.valptr[l] = p as i32;
                h.mincode[l] = huffcode[p];
                p += bits[l - 1] as usize;
                h.maxcode[l] = huffcode[p - 1];
            } else {
                h.maxcode[l] = -1;
            }
        }
        h.maxcode[17] = 0xFFFFF;
        h
    }
}

/// MSB-first bit reader over an entropy-coded segment, handling 0xFF00 byte-stuffing and stopping
/// (feeding zero bits, libjpeg-style) at any marker.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u32,
    cnt: u32,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8], pos: usize) -> Bits<'a> {
        Bits { data, pos, acc: 0, cnt: 0 }
    }

    /// Next entropy byte, or 0 without advancing once a marker is reached.
    fn byte(&mut self) -> u8 {
        if self.pos >= self.data.len() {
            return 0;
        }
        let b = self.data[self.pos];
        if b == 0xFF {
            match self.data.get(self.pos + 1) {
                Some(0x00) => {
                    self.pos += 2;
                    0xFF
                }
                _ => 0, // marker: stop here, leave pos at the 0xFF
            }
        } else {
            self.pos += 1;
            b
        }
    }

    fn fill(&mut self) {
        while self.cnt <= 24 {
            let b = self.byte() as u32;
            self.acc |= b << (24 - self.cnt);
            self.cnt += 8;
        }
    }

    fn get(&mut self, n: u32) -> i32 {
        if n == 0 || n > 16 {
            return 0; // n > 16 means a corrupt size symbol; keep the shift in range
        }
        self.fill();
        let v = self.acc >> (32 - n);
        self.acc <<= n;
        self.cnt -= n;
        v as i32
    }

    fn receive_extend(&mut self, n: u32) -> i32 {
        let v = self.get(n);
        if n != 0 && n <= 16 && v < (1 << (n - 1)) {
            v - (1 << n) + 1
        } else {
            v
        }
    }

    fn decode(&mut self, t: &Huff) -> u8 {
        let mut code = 0i32;
        for l in 1..=16 {
            code = (code << 1) | self.get(1);
            if code <= t.maxcode[l] {
                let i = t.valptr[l] + code - t.mincode[l];
                return *t.values.get(i as usize).unwrap_or(&0);
            }
        }
        0
    }

    /// Byte-align and consume the next RSTn marker (scanning past any padding/stuffing).
    fn restart(&mut self) {
        self.acc = 0;
        self.cnt = 0;
        while self.pos + 1 < self.data.len() {
            if self.data[self.pos] == 0xFF {
                let m = self.data[self.pos + 1];
                if (0xD0..=0xD7).contains(&m) {
                    self.pos += 2;
                    return;
                }
                if m == 0x00 {
                    self.pos += 2;
                    continue;
                }
                return; // some other marker: stop resync here
            }
            self.pos += 1;
        }
    }
}

// islow IDCT constants (CONST_BITS = 13).
const C: [i32; 12] = [
    2446, 3196, 4433, 6270, 7373, 9633, 12299, 15137, 16069, 16819, 20995, 25172,
];
const CONST_BITS: u32 = 13;
const PASS1_BITS: u32 = 2;

#[inline]
fn descale(x: W, n: u32) -> W {
    (x + Wrapping(1 << (n - 1))) >> n as usize
}

/// jpeg_idct_islow: dequantise `coef` (natural order) by `quant` (natural order) and inverse-DCT
/// into `out` (row-major 8×8), level-shifted by 128 and clamped to u8 — exactly as IPP's
/// `DCTQuantInv8x8LS` finishes with `packus(x + 128)`.
fn idct_islow(coef: &[i32; 64], quant: &[i32; 64], out: &mut [u8; 64]) {
    let f = |i: usize| C[i];
    let mut ws = [Wrapping(0i32); 64];
    // Pass 1: columns.
    for col in 0..8 {
        let d = |i: usize| Wrapping(coef[col + i * 8] * quant[col + i * 8]);
        let z2 = d(2);
        let z3 = d(6);
        let z1 = (z2 + z3) * Wrapping(f(2));
        let tmp2 = z1 + z3 * Wrapping(-f(7));
        let tmp3 = z1 + z2 * Wrapping(f(3));
        let z2 = d(0);
        let z3 = d(4);
        let tmp0 = (z2 + z3) << CONST_BITS as usize;
        let tmp1 = (z2 - z3) << CONST_BITS as usize;
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;
        let t0 = d(7);
        let t1 = d(5);
        let t2 = d(3);
        let t3 = d(1);
        let z1 = t0 + t3;
        let z2 = t1 + t2;
        let z3 = t0 + t2;
        let z4 = t1 + t3;
        let z5 = (z3 + z4) * Wrapping(f(5));
        let t0 = t0 * Wrapping(f(0));
        let t1 = t1 * Wrapping(f(9));
        let t2 = t2 * Wrapping(f(11));
        let t3 = t3 * Wrapping(f(6));
        let z1 = z1 * Wrapping(-f(4));
        let z2 = z2 * Wrapping(-f(10));
        let z3 = z3 * Wrapping(-f(8)) + z5;
        let z4 = z4 * Wrapping(-f(1)) + z5;
        let t0 = t0 + z1 + z3;
        let t1 = t1 + z2 + z4;
        let t2 = t2 + z2 + z3;
        let t3 = t3 + z1 + z4;
        let sh = CONST_BITS - PASS1_BITS;
        ws[col] = descale(tmp10 + t3, sh);
        ws[col + 56] = descale(tmp10 - t3, sh);
        ws[col + 8] = descale(tmp11 + t2, sh);
        ws[col + 48] = descale(tmp11 - t2, sh);
        ws[col + 16] = descale(tmp12 + t1, sh);
        ws[col + 40] = descale(tmp12 - t1, sh);
        ws[col + 24] = descale(tmp13 + t0, sh);
        ws[col + 32] = descale(tmp13 - t0, sh);
    }
    // Pass 2: rows.
    for row in 0..8 {
        let r = row * 8;
        let w = |i: usize| ws[r + i];
        let z2 = w(2);
        let z3 = w(6);
        let z1 = (z2 + z3) * Wrapping(f(2));
        let tmp2 = z1 + z3 * Wrapping(-f(7));
        let tmp3 = z1 + z2 * Wrapping(f(3));
        let tmp0 = (w(0) + w(4)) << CONST_BITS as usize;
        let tmp1 = (w(0) - w(4)) << CONST_BITS as usize;
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;
        let t0 = w(7);
        let t1 = w(5);
        let t2 = w(3);
        let t3 = w(1);
        let z1 = t0 + t3;
        let z2 = t1 + t2;
        let z3 = t0 + t2;
        let z4 = t1 + t3;
        let z5 = (z3 + z4) * Wrapping(f(5));
        let t0 = t0 * Wrapping(f(0));
        let t1 = t1 * Wrapping(f(9));
        let t2 = t2 * Wrapping(f(11));
        let t3 = t3 * Wrapping(f(6));
        let z1 = z1 * Wrapping(-f(4));
        let z2 = z2 * Wrapping(-f(10));
        let z3 = z3 * Wrapping(-f(8)) + z5;
        let z4 = z4 * Wrapping(-f(1)) + z5;
        let t0 = t0 + z1 + z3;
        let t1 = t1 + z2 + z4;
        let t2 = t2 + z2 + z3;
        let t3 = t3 + z1 + z4;
        let sh = CONST_BITS + PASS1_BITS + 3;
        let px = |v: W| -> u8 { (descale(v, sh).0 + 128).clamp(0, 255) as u8 };
        out[r] = px(tmp10 + t3);
        out[r + 7] = px(tmp10 - t3);
        out[r + 1] = px(tmp11 + t2);
        out[r + 6] = px(tmp11 - t2);
        out[r + 2] = px(tmp12 + t1);
        out[r + 5] = px(tmp12 - t1);
        out[r + 3] = px(tmp13 + t0);
        out[r + 4] = px(tmp13 - t0);
    }
}

struct Comp {
    id: u8,
    h: usize,
    v: usize,
    tq: usize,
    td: usize,
    ta: usize,
    // DCT coefficients (natural order), one 64-block per block in the MCU-padded `bw`×`bh` grid.
    coeffs: Vec<i16>,
    bw: usize, // MCU-padded blocks wide (= mcus_x * h); matches the coeff grid
    bh: usize,
    bw_img: usize, // meaningful blocks (= ceil(dw/8)); bounds for non-interleaved scans
    bh_img: usize,
    // block-padded sample plane: `stride` wide, filled by the IDCT; `dw`/`dh` meaningful.
    plane: Vec<u8>,
    stride: usize,
    dw: usize,
    dh: usize,
    cblk: usize,  // this component's IDCT output block edge (1/2/4/8); ≥ the frame min for
    up_h: usize,  // doubly-subsampled chroma, so it needs less upsampling (`up_h`/`up_v`).
    up_v: usize,
}

struct Frame {
    out_w: usize, // scaled output = ⌈width/denom⌉; equals width at denom 1
    out_h: usize,
    comps: Vec<Comp>,
    quant: [[i32; 64]; 4],
    progressive: bool,
    mcus_x: usize,
    mcus_y: usize,
}

pub fn decode_jpeg(bytes: &[u8]) -> Result<DecodedImage, u32> {
    decode_jpeg_scaled(bytes, 1)
}

/// The closest scale WIC's `IWICBitmapSourceTransform` snaps a requested `w`×`h` to for a
/// `full_w`×`full_h` image: the largest denominator in {1,2,4,8} whose ⌈dim/denom⌉ still covers
/// the request, and the resulting output size. Reproduces `GetClosestScaleDenominator` +
/// WIC's JPEG output-size calculation.
pub fn jpeg_closest_scale(full_w: u32, full_h: u32, w: u32, h: u32) -> (u32, (u32, u32)) {
    let axis = |dim: u32, req: u32| if req <= 1 { 8 } else { (dim - 1) / (req - 1) };
    let m = axis(full_w, w).min(axis(full_h, h));
    let denom = if m >= 8 { 8 } else if m >= 4 { 4 } else if m >= 2 { 2 } else { 1 };
    (denom, (full_w.div_ceil(denom), full_h.div_ceil(denom)))
}

/// Decode at a native reduced scale (`denom` ∈ {1,2,4,8}), as WIC's `IWICBitmapSourceTransform`
/// does with libjpeg's reduced-size IDCTs. `denom` 1 is the full decode.
pub fn decode_jpeg_scaled(bytes: &[u8], denom: usize) -> Result<DecodedImage, u32> {
    debug_assert!(matches!(denom, 1 | 2 | 4 | 8));
    let rd16 = |b: &[u8], i: usize| -> usize { ((b[i] as usize) << 8) | b[i + 1] as usize };
    if bytes.len() < 2 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return Err(WINCODEC_ERR_COMPONENTNOTFOUND); // no SOI: WIC matches no codec
    }
    let mut p = 2usize;

    let mut frame: Option<Frame> = None;
    let mut quant = [[0i32; 64]; 4];
    let mut quant_defined = [false; 4];
    let mut dc_tbl: [Option<Huff>; 4] = [None, None, None, None];
    let mut ac_tbl: [Option<Huff>; 4] = [None, None, None, None];
    let mut restart = 0usize;
    let mut scanned = false;
    let mut adobe_transform: Option<u8> = None;

    loop {
        // Seek the next marker.
        while p < bytes.len() && bytes[p] != 0xFF {
            p += 1;
        }
        while p < bytes.len() && bytes[p] == 0xFF {
            p += 1;
        }
        // A frame header makes the image presentable: before one, structural problems read as a
        // missing frame (CreateDecoder); after one, as a bad header (CopyPixels).
        let structural = || if frame.is_some() { WINCODEC_ERR_BADHEADER } else { WINCODEC_ERR_FRAMEMISSING };
        if p >= bytes.len() {
            return Err(structural());
        }
        let marker = bytes[p];
        p += 1;
        match marker {
            0xD9 => {
                // EOI: realize the image if any scan ran, else no frame was ever decoded.
                let f = frame.as_mut().filter(|_| scanned).ok_or(WINCODEC_ERR_FRAMEMISSING)?;
                validate_decodable(f, &quant_defined)?;
                f.quant = quant;
                realize(f);
                return assemble(f, adobe_transform);
            }
            0x01 | 0xD0..=0xD7 => continue, // standalone markers
            _ => {}
        }
        if p + 2 > bytes.len() {
            return Err(structural());
        }
        let len = rd16(bytes, p);
        if len < 2 || p + len > bytes.len() {
            return Err(structural());
        }
        let seg = &bytes[p + 2..p + len];
        let seg_end = p + len;

        match marker {
            0xC0 => frame = Some(parse_sof(seg, false, denom)?), // SOF0 baseline
            0xC2 => frame = Some(parse_sof(seg, true, denom)?),  // SOF2 progressive
            0xC4 => parse_dht(seg, &mut dc_tbl, &mut ac_tbl)?,
            // extended/lossless/arithmetic SOF and DAC: valid frame, undecodable → BADHEADER.
            0xC1 | 0xC3 | 0xC5..=0xC7 | 0xC9..=0xCF => return Err(WINCODEC_ERR_BADHEADER),
            0xDB => parse_dqt(seg, &mut quant, &mut quant_defined)?,
            0xDD => {
                if seg.len() < 2 {
                    return Err(WINCODEC_ERR_BADHEADER);
                }
                restart = rd16(seg, 0);
            }
            0xDA => {
                let f = frame.as_mut().ok_or(WINCODEC_ERR_FRAMEMISSING)?;
                let scan = parse_sos(seg, f)?;
                p = decode_scan(bytes, p + 2 + scan.hdr_len, f, &scan, &dc_tbl, &ac_tbl, restart)?;
                scanned = true;
                if !f.progressive {
                    validate_decodable(f, &quant_defined)?;
                    f.quant = quant;
                    realize(f);
                    return assemble(f, adobe_transform);
                }
                continue; // more progressive scans follow
            }
            0xEE if seg.len() >= 12 && &seg[0..5] == b"Adobe" => {
                adobe_transform = Some(seg[11]); // 0=CMYK/RGB, 1=YCbCr, 2=YCCK
            }
            _ => {} // APPn/COM/DNL/etc.: skip
        }
        p = seg_end;
    }
}

fn parse_sof(seg: &[u8], progressive: bool, denom: usize) -> Result<Frame, u32> {
    // A frame header too short or with no components / zero size is not a presentable frame.
    if seg.len() < 6 {
        return Err(WINCODEC_ERR_FRAMEMISSING);
    }
    let prec = seg[0];
    if prec != 8 {
        return Err(WINCODEC_ERR_BADHEADER); // only 8-bit precision is decodable
    }
    let height = ((seg[1] as usize) << 8) | seg[2] as usize;
    let width = ((seg[3] as usize) << 8) | seg[4] as usize;
    let nc = seg[5] as usize;
    if nc == 0 || seg.len() < 6 + nc * 3 || width == 0 || height == 0 {
        return Err(WINCODEC_ERR_FRAMEMISSING);
    }
    if nc != 1 && nc != 3 && nc != 4 {
        return Err(WINCODEC_ERR_BADHEADER); // no WIC output format for this component count
    }
    let mut comps = Vec::with_capacity(nc);
    let (mut max_h, mut max_v) = (1usize, 1usize);
    for i in 0..nc {
        let o = 6 + i * 3;
        let h = (seg[o + 1] >> 4) as usize;
        let v = (seg[o + 1] & 15) as usize;
        if h == 0 || v == 0 || h > 4 || v > 4 {
            return Err(WINCODEC_ERR_BADHEADER);
        }
        max_h = max_h.max(h);
        max_v = max_v.max(v);
        comps.push(Comp {
            id: seg[o],
            h,
            v,
            tq: (seg[o + 2] & 3) as usize,
            td: 0,
            ta: 0,
            coeffs: Vec::new(),
            bw: 0,
            bh: 0,
            bw_img: 0,
            bh_img: 0,
            plane: Vec::new(),
            stride: 0,
            dw: 0,
            dh: 0,
            cblk: 0,
            up_h: 1,
            up_v: 1,
        });
    }
    let mcus_x = width.div_ceil(max_h * 8);
    let mcus_y = height.div_ceil(max_v * 8);
    // Scaled output = ⌈dim/denom⌉; the frame's minimum IDCT block edge is `min_blk` = 8/denom.
    let (out_w, out_h) = (width.div_ceil(denom), height.div_ceil(denom));
    let min_blk = 8 / denom;
    for c in &mut comps {
        c.bw = mcus_x * c.h;
        c.bh = mcus_y * c.v;
        // libjpeg gives each component the largest power-of-two IDCT block (≤8) that still fits
        // within the frame's downscale: a component subsampled in *both* axes (e.g. 4:2:0 chroma)
        // is transformed at a bigger block and upsampled less (or not at all). `g` is that factor.
        let mut g = 1;
        while g * 2 <= (max_h / c.h).min(max_v / c.v) && min_blk * g * 2 <= 8 {
            g *= 2;
        }
        c.cblk = min_blk * g;
        c.up_h = (max_h / c.h) / g;
        c.up_v = (max_v / c.v) / g;
        // Full-resolution component block counts drive entropy decode (coefficients, scale-free).
        c.bw_img = (width * c.h).div_ceil(max_h).div_ceil(8);
        c.bh_img = (height * c.v).div_ceil(max_v).div_ceil(8);
        c.coeffs = vec![0i16; c.bw * c.bh * 64];
        // Scaled sample plane: meaningful region `dw`×`dh` (libjpeg's downsampled size for this
        // component's block edge), block-padded to `bw*cblk`×`bh*cblk`.
        c.dw = (width * c.h * c.cblk).div_ceil(max_h * 8);
        c.dh = (height * c.v * c.cblk).div_ceil(max_v * 8);
        c.stride = c.bw * c.cblk;
        c.plane = vec![0u8; c.stride * c.bh * c.cblk];
    }
    Ok(Frame {
        out_w,
        out_h,
        comps,
        quant: [[0; 64]; 4],
        progressive,
        mcus_x,
        mcus_y,
    })
}

fn parse_dqt(seg: &[u8], quant: &mut [[i32; 64]; 4], defined: &mut [bool; 4]) -> Result<(), u32> {
    let mut i = 0;
    while i < seg.len() {
        let pq = seg[i] >> 4;
        let tq = (seg[i] & 15) as usize;
        i += 1;
        if tq > 3 {
            return Err(WINCODEC_ERR_BADHEADER);
        }
        if pq == 0 {
            if i + 64 > seg.len() {
                return Err(WINCODEC_ERR_BADHEADER);
            }
            for k in 0..64 {
                quant[tq][ZIGZAG[k]] = seg[i + k] as i32;
            }
            i += 64;
        } else {
            if i + 128 > seg.len() {
                return Err(WINCODEC_ERR_BADHEADER);
            }
            for k in 0..64 {
                quant[tq][ZIGZAG[k]] = ((seg[i + 2 * k] as i32) << 8) | seg[i + 2 * k + 1] as i32;
            }
            i += 128;
        }
        defined[tq] = true;
    }
    Ok(())
}

/// Reject a frame that cannot be decompressed because a component selects a quantization table that
/// no DQT ever defined — libjpeg's `JERR_NO_QUANT_TABLE`, which WIC surfaces as a bad header.
fn validate_decodable(f: &Frame, quant_defined: &[bool; 4]) -> Result<(), u32> {
    if f.comps.iter().all(|c| quant_defined[c.tq]) {
        Ok(())
    } else {
        Err(WINCODEC_ERR_BADHEADER)
    }
}

fn parse_dht(
    seg: &[u8],
    dc: &mut [Option<Huff>; 4],
    ac: &mut [Option<Huff>; 4],
) -> Result<(), u32> {
    let mut i = 0;
    while i < seg.len() {
        if i + 17 > seg.len() {
            return Err(WINCODEC_ERR_BADHEADER);
        }
        let class = seg[i] >> 4;
        let id = (seg[i] & 15) as usize;
        i += 1;
        let mut bits = [0u8; 16];
        let mut total = 0usize;
        for b in 0..16 {
            bits[b] = seg[i + b];
            total += bits[b] as usize;
        }
        i += 16;
        if id > 3 || i + total > seg.len() {
            return Err(WINCODEC_ERR_BADHEADER);
        }
        let values = seg[i..i + total].to_vec();
        i += total;
        let h = Huff::build(&bits, values);
        if class == 0 {
            dc[id] = Some(h);
        } else {
            ac[id] = Some(h);
        }
    }
    Ok(())
}

struct Scan {
    comps: Vec<usize>, // indices into Frame.comps, in scan order
    ss: usize,
    se: usize,
    ah: u32,
    al: u32,
    hdr_len: usize, // bytes of header within the SOS segment (entropy data follows)
}

/// Parse the SOS header, wiring each scan component's DC/AC table selectors.
fn parse_sos(seg: &[u8], f: &mut Frame) -> Result<Scan, u32> {
    if seg.is_empty() {
        return Err(WINCODEC_ERR_BADHEADER);
    }
    let ns = seg[0] as usize;
    if ns == 0 || ns > f.comps.len() || seg.len() < 1 + ns * 2 + 3 {
        return Err(WINCODEC_ERR_BADHEADER);
    }
    let mut comps = Vec::with_capacity(ns);
    for i in 0..ns {
        let cs = seg[1 + i * 2];
        let t = seg[2 + i * 2];
        let idx = f.comps.iter().position(|c| c.id == cs).ok_or(WINCODEC_ERR_BADHEADER)?;
        let (td, ta) = ((t >> 4) as usize, (t & 15) as usize);
        if td > 3 || ta > 3 {
            return Err(WINCODEC_ERR_BADHEADER);
        }
        f.comps[idx].td = td;
        f.comps[idx].ta = ta;
        comps.push(idx);
    }
    let o = 1 + ns * 2;
    Ok(Scan {
        comps,
        ss: seg[o] as usize,
        se: (seg[o + 1] as usize).min(63),
        ah: (seg[o + 2] >> 4) as u32,
        al: (seg[o + 2] & 15) as u32,
        hdr_len: o + 3,
    })
}

/// Decode one scan into the components' coefficient buffers (baseline sequential, or one of the four
/// progressive sub-scans). Returns the byte position just after the entropy data (at the next
/// marker). A single-component scan iterates that component's own blocks; a multi-component scan
/// iterates MCUs.
fn decode_scan(
    bytes: &[u8],
    start: usize,
    f: &mut Frame,
    scan: &Scan,
    dc: &[Option<Huff>; 4],
    ac: &[Option<Huff>; 4],
    restart: usize,
) -> Result<usize, u32> {
    let mut br = Bits::new(bytes, start);
    let mut pred = [0i32; 4];
    let mut eobrun = 0u32;
    let mut until = restart;
    let baseline = !f.progressive;

    if scan.comps.len() > 1 {
        for my in 0..f.mcus_y {
            for mx in 0..f.mcus_x {
                if restart != 0 && until == 0 {
                    br.restart();
                    pred = [0; 4];
                    eobrun = 0;
                    until = restart;
                }
                for (si, &ci) in scan.comps.iter().enumerate() {
                    let (h, v, bw) = {
                        let c = &f.comps[ci];
                        (c.h, c.v, c.bw)
                    };
                    for cy in 0..v {
                        for cx in 0..h {
                            let bi = ((my * v + cy) * bw + mx * h + cx) * 64;
                            decode_block(&mut br, f, ci, bi, scan, baseline, dc, ac, &mut pred[si], &mut eobrun)?;
                        }
                    }
                }
                if restart != 0 {
                    until -= 1;
                }
            }
        }
    } else {
        let ci = scan.comps[0];
        let (bw, bw_img, bh_img) = {
            let c = &f.comps[ci];
            (c.bw, c.bw_img, c.bh_img)
        };
        for by in 0..bh_img {
            for bx in 0..bw_img {
                if restart != 0 && until == 0 {
                    br.restart();
                    pred = [0; 4];
                    eobrun = 0;
                    until = restart;
                }
                let bi = (by * bw + bx) * 64;
                decode_block(&mut br, f, ci, bi, scan, baseline, dc, ac, &mut pred[0], &mut eobrun)?;
                if restart != 0 {
                    until -= 1;
                }
            }
        }
    }
    Ok(br.pos)
}

#[allow(clippy::too_many_arguments)]
fn decode_block(
    br: &mut Bits,
    f: &mut Frame,
    ci: usize,
    bi: usize,
    scan: &Scan,
    baseline: bool,
    dc: &[Option<Huff>; 4],
    ac: &[Option<Huff>; 4],
    pred: &mut i32,
    eobrun: &mut u32,
) -> Result<(), u32> {
    let (td, ta) = (f.comps[ci].td, f.comps[ci].ta);
    let blk = &mut f.comps[ci].coeffs[bi..bi + 64];
    if baseline {
        let dct = dc[td].as_ref().ok_or(WINCODEC_ERR_BADHEADER)?;
        let act = ac[ta].as_ref().ok_or(WINCODEC_ERR_BADHEADER)?;
        let s = br.decode(dct);
        *pred += br.receive_extend(s as u32);
        blk[0] = *pred as i16;
        let mut k = 1usize;
        while k < 64 {
            let rs = br.decode(act);
            let r = (rs >> 4) as usize;
            let sz = (rs & 15) as u32;
            if sz == 0 {
                if r != 15 {
                    break;
                }
                k += 16;
            } else {
                k += r;
                if k >= 64 {
                    break;
                }
                blk[ZIGZAG[k]] = br.receive_extend(sz) as i16;
                k += 1;
            }
        }
    } else if scan.ss == 0 {
        if scan.ah == 0 {
            let dct = dc[td].as_ref().ok_or(WINCODEC_ERR_BADHEADER)?;
            let s = br.decode(dct);
            *pred += br.receive_extend(s as u32);
            blk[0] = (*pred << scan.al) as i16;
        } else if br.get(1) != 0 {
            blk[0] |= (1 << scan.al) as i16;
        }
    } else {
        let act = ac[ta].as_ref().ok_or(WINCODEC_ERR_BADHEADER)?;
        if scan.ah == 0 {
            ac_first(br, act, scan.ss, scan.se, scan.al, eobrun, blk);
        } else {
            ac_refine(br, act, scan.ss, scan.se, scan.al, eobrun, blk);
        }
    }
    Ok(())
}

/// Progressive AC first scan (spectral band [ss,se], new coefficients at bit `al`).
fn ac_first(br: &mut Bits, act: &Huff, ss: usize, se: usize, al: u32, eobrun: &mut u32, blk: &mut [i16]) {
    if *eobrun > 0 {
        *eobrun -= 1;
        return;
    }
    let mut k = ss;
    while k <= se {
        let rs = br.decode(act);
        let r = (rs >> 4) as usize;
        let s = (rs & 15) as u32;
        if s != 0 {
            k += r;
            if k > se {
                break;
            }
            blk[ZIGZAG[k]] = (br.receive_extend(s) << al) as i16;
            k += 1;
        } else {
            if r != 15 {
                if r != 0 {
                    *eobrun = (1 << r) + br.get(r as u32) as u32;
                }
                break;
            }
            k += 16;
        }
    }
}

/// Progressive AC refinement scan (correction bits for existing coefficients, plus newly nonzero
/// ones), reproducing libjpeg's `decode_mcu_AC_refine`.
fn ac_refine(br: &mut Bits, act: &Huff, ss: usize, se: usize, al: u32, eobrun: &mut u32, blk: &mut [i16]) {
    let p1: i16 = 1 << al;
    let m1: i16 = -1i16 << al;
    let bit = |br: &mut Bits, v: &mut i16| {
        if br.get(1) != 0 && (*v & p1) == 0 {
            *v = v.wrapping_add(if *v >= 0 { p1 } else { m1 });
        }
    };
    let mut k = ss;
    if *eobrun == 0 {
        while k <= se {
            let rs = br.decode(act);
            let mut r = (rs >> 4) as i32;
            let s = rs & 15;
            let newval: i16 = if s != 0 {
                if br.get(1) != 0 { p1 } else { m1 }
            } else {
                if r != 15 {
                    *eobrun = 1 << r;
                    if r != 0 {
                        *eobrun += br.get(r as u32) as u32;
                    }
                    break;
                }
                0
            };
            loop {
                let pos = ZIGZAG[k];
                if blk[pos] != 0 {
                    let mut v = blk[pos];
                    bit(br, &mut v);
                    blk[pos] = v;
                } else {
                    r -= 1;
                    if r < 0 {
                        break;
                    }
                }
                k += 1;
                if k > se {
                    break;
                }
            }
            if s != 0 && k <= se {
                blk[ZIGZAG[k]] = newval;
            }
            k += 1;
        }
    }
    if *eobrun > 0 {
        while k <= se {
            let pos = ZIGZAG[k];
            if blk[pos] != 0 {
                let mut v = blk[pos];
                bit(br, &mut v);
                blk[pos] = v;
            }
            k += 1;
        }
        *eobrun -= 1;
    }
}

/// jpeg_idct_4x4 (jidctred.c): 8×8 coefficients → top-left 4×4 samples, written row-major into the
/// first 16 bytes of `out`. Same CONST_BITS/PASS1_BITS fixed point as islow; MULTIPLY is 32-bit.
fn idct_4x4(coef: &[i32; 64], quant: &[i32; 64], out: &mut [u8; 64]) {
    let m = |a: W, b: i32| a * Wrapping(b);
    let mut ws = [Wrapping(0i32); 32]; // 8 cols × 4 rows
    for col in (0..8).filter(|&c| c != 4) {
        let d = |i: usize| Wrapping(coef[col + i * 8] * quant[col + i * 8]);
        let tmp0 = d(0) << (CONST_BITS as usize + 1);
        let tmp2 = m(d(2), 15137) + m(d(6), -6270);
        let (tmp10, tmp12) = (tmp0 + tmp2, tmp0 - tmp2);
        let o0 = m(d(7), -1730) + m(d(5), 11893) + m(d(3), -17799) + m(d(1), 8697);
        let o2 = m(d(7), -4176) + m(d(5), -4926) + m(d(3), 7373) + m(d(1), 20995);
        let sh = CONST_BITS - PASS1_BITS + 1;
        ws[col] = descale(tmp10 + o2, sh);
        ws[col + 24] = descale(tmp10 - o2, sh);
        ws[col + 8] = descale(tmp12 + o0, sh);
        ws[col + 16] = descale(tmp12 - o0, sh);
    }
    for row in 0..4 {
        let w = |i: usize| ws[row * 8 + i];
        let tmp0 = w(0) << (CONST_BITS as usize + 1);
        let tmp2 = m(w(2), 15137) + m(w(6), -6270);
        let (tmp10, tmp12) = (tmp0 + tmp2, tmp0 - tmp2);
        let o0 = m(w(7), -1730) + m(w(5), 11893) + m(w(3), -17799) + m(w(1), 8697);
        let o2 = m(w(7), -4176) + m(w(5), -4926) + m(w(3), 7373) + m(w(1), 20995);
        let sh = CONST_BITS + PASS1_BITS + 3 + 1;
        let px = |v: W| -> u8 { (descale(v, sh).0 + 128).clamp(0, 255) as u8 };
        out[row * 4] = px(tmp10 + o2);
        out[row * 4 + 3] = px(tmp10 - o2);
        out[row * 4 + 1] = px(tmp12 + o0);
        out[row * 4 + 2] = px(tmp12 - o0);
    }
}

/// jpeg_idct_2x2 (jidctred.c): 8×8 coefficients → top-left 2×2 samples.
fn idct_2x2(coef: &[i32; 64], quant: &[i32; 64], out: &mut [u8; 64]) {
    let m = |a: W, b: i32| a * Wrapping(b);
    let mut ws = [Wrapping(0i32); 16]; // 8 cols × 2 rows
    for col in [0usize, 1, 3, 5, 7] {
        let d = |i: usize| Wrapping(coef[col + i * 8] * quant[col + i * 8]);
        let tmp10 = d(0) << (CONST_BITS as usize + 2);
        let tmp0 = m(d(7), -5906) + m(d(5), 6967) + m(d(3), -10426) + m(d(1), 29692);
        let sh = CONST_BITS - PASS1_BITS + 2;
        ws[col] = descale(tmp10 + tmp0, sh);
        ws[col + 8] = descale(tmp10 - tmp0, sh);
    }
    for row in 0..2 {
        let w = |i: usize| ws[row * 8 + i];
        let tmp10 = w(0) << (CONST_BITS as usize + 2);
        let tmp0 = m(w(7), -5906) + m(w(5), 6967) + m(w(3), -10426) + m(w(1), 29692);
        let sh = CONST_BITS + PASS1_BITS + 3 + 2;
        let px = |v: W| -> u8 { (descale(v, sh).0 + 128).clamp(0, 255) as u8 };
        out[row * 2] = px(tmp10 + tmp0);
        out[row * 2 + 1] = px(tmp10 - tmp0);
    }
}

/// jpeg_idct_1x1 (jidctred.c): one sample = ⅛ of the DC coefficient.
fn idct_1x1(coef: &[i32; 64], quant: &[i32; 64], out: &mut [u8; 64]) {
    out[0] = (descale(Wrapping(coef[0] * quant[0]), 3).0 + 128).clamp(0, 255) as u8;
}

/// Dequantise and inverse-DCT every block into each component's sample plane, at the frame's scale
/// (`denom` 1 = full islow; 2/4/8 = jidctred's 4×4 / 2×2 / 1×1 reduced IDCTs).
fn realize(f: &mut Frame) {
    let quant = f.quant;
    for c in &mut f.comps {
        let q = &quant[c.tq];
        let blk = c.cblk;
        for by in 0..c.bh {
            for bx in 0..c.bw {
                let bi = (by * c.bw + bx) * 64;
                let mut coef = [0i32; 64];
                for (i, v) in coef.iter_mut().enumerate() {
                    *v = c.coeffs[bi + i] as i32;
                }
                let mut tile = [0u8; 64];
                match blk {
                    1 => idct_1x1(&coef, q, &mut tile),
                    2 => idct_2x2(&coef, q, &mut tile),
                    4 => idct_4x4(&coef, q, &mut tile),
                    _ => idct_islow(&coef, q, &mut tile),
                }
                let (px, py) = (bx * blk, by * blk);
                for r in 0..blk {
                    let dst = (py + r) * c.stride + px;
                    c.plane[dst..dst + blk].copy_from_slice(&tile[r * blk..r * blk + blk]);
                }
            }
        }
    }
}

/// Expand one component to the scaled output resolution, reproducing libjpeg's upsamplers: fancy
/// triangle for a 2× axis (h2v1 / h2v2 / h1v2) when the component is wider than 2 samples, plain
/// integer replication otherwise, copy at 1:1. Ratios are per-component (`up_h`/`up_v`), already
/// reduced by this component's larger IDCT block.
fn upsample(c: &Comp, out_w: usize, out_h: usize) -> Vec<u8> {
    let hx = c.up_h;
    let vx = c.up_v;
    let fancy = c.dw > 2; // libjpeg falls back to box upsampling for ≤2-wide components
    let g = |x: isize, y: isize| -> i32 {
        let xx = x.clamp(0, c.dw as isize - 1) as usize;
        let yy = y.clamp(0, c.dh as isize - 1) as usize;
        c.plane[yy * c.stride + xx] as i32
    };
    let mut out = vec![0u8; out_w * out_h];
    if hx == 1 && vx == 1 {
        for y in 0..out_h {
            for x in 0..out_w {
                out[y * out_w + x] = g(x as isize, y as isize) as u8;
            }
        }
    } else if hx == 2 && vx == 1 && fancy {
        for y in 0..out_h {
            let row = &mut out[y * out_w..y * out_w + out_w];
            for i in 0..c.dw {
                let v3 = g(i as isize, y as isize) * 3;
                let a = ((v3 + g(i as isize - 1, y as isize) + 1) >> 2) as u8;
                let b = ((v3 + g(i as isize + 1, y as isize) + 2) >> 2) as u8;
                if 2 * i < out_w {
                    row[2 * i] = a;
                }
                if 2 * i + 1 < out_w {
                    row[2 * i + 1] = b;
                }
            }
        }
    } else if hx == 2 && vx == 2 && fancy {
        for iy in 0..c.dh {
            for sub in 0..2 {
                let oy = iy * 2 + sub;
                if oy >= out_h {
                    break;
                }
                let far = if sub == 0 { iy as isize - 1 } else { iy as isize + 1 };
                let col = |i: isize| -> i32 { g(i, iy as isize) * 3 + g(i, far) };
                let row = &mut out[oy * out_w..oy * out_w + out_w];
                for i in 0..c.dw {
                    let this = col(i as isize);
                    let left = col(i as isize - 1);
                    let right = col(i as isize + 1);
                    let a = ((this * 3 + left + 8) >> 4) as u8;
                    let b = ((this * 3 + right + 7) >> 4) as u8;
                    if 2 * i < out_w {
                        row[2 * i] = a;
                    }
                    if 2 * i + 1 < out_w {
                        row[2 * i + 1] = b;
                    }
                }
            }
        }
    } else if hx == 1 && vx == 2 {
        // fancy vertical triangle (libjpeg-turbo h1v2): near row weight 3, adjacent weight 1.
        for iy in 0..c.dh {
            for sub in 0..2 {
                let oy = iy * 2 + sub;
                if oy >= out_h {
                    break;
                }
                let far = if sub == 0 { iy as isize - 1 } else { iy as isize + 1 };
                let bias = if sub == 0 { 1 } else { 2 };
                let row = &mut out[oy * out_w..oy * out_w + out_w];
                for x in 0..out_w {
                    let v = g(x as isize, iy as isize) * 3 + g(x as isize, far) + bias;
                    row[x] = (v >> 2) as u8;
                }
            }
        }
    } else {
        // integer replication (nearest), e.g. 4:1:1.
        for y in 0..out_h {
            let sy = (y / vx).min(c.dh - 1);
            for x in 0..out_w {
                let sx = (x / hx).min(c.dw - 1);
                out[y * out_w + x] = c.plane[sy * c.stride + sx];
            }
        }
    }
    out
}

/// YCbCr → BGR, libjpeg's `ycc_rgb_convert` 16-bit-scaled fixed point (`jdcolor.c`): the oracle
/// (WIC == libjpeg-turbo 3.2.0's `djpeg`) follows these constants exactly. (windowscodecs' IPP
/// `ippiYCbCrToBGR_JPEG` has a 14-bit scalar tail with different constants, but the decode output
/// matches this 16-bit path on every pixel.)
#[inline]
fn ycbcr_to_bgr(y: i32, cb: i32, cr: i32, out: &mut [u8]) {
    let cb = cb - 128;
    let cr = cr - 128;
    let r = y + ((91881 * cr + 32768) >> 16);
    let g = y + ((-22554 * cb - 46802 * cr + 32768) >> 16);
    let b = y + ((116130 * cb + 32768) >> 16);
    out[0] = b.clamp(0, 255) as u8;
    out[1] = g.clamp(0, 255) as u8;
    out[2] = r.clamp(0, 255) as u8;
}

fn assemble(f: &Frame, adobe: Option<u8>) -> Result<DecodedImage, u32> {
    let (w, h) = (f.out_w, f.out_h); // scaled output (= width/height at denom 1)
    let planes: Vec<Vec<u8>> = f
        .comps
        .iter()
        .map(|c| upsample(c, w, h))
        .collect();

    let (format, stride, pixels) = match f.comps.len() {
        1 => (WicFormat::Gray8, w, planes[0].clone()),
        3 => {
            // 24bppBGR. Adobe transform 0 ⇒ components are already RGB; otherwise YCbCr.
            let rgb = adobe == Some(0);
            let mut px = vec![0u8; w * h * 3];
            for i in 0..w * h {
                let o = &mut px[i * 3..i * 3 + 3];
                if rgb {
                    o.copy_from_slice(&[planes[2][i], planes[1][i], planes[0][i]]); // B,G,R
                } else {
                    ycbcr_to_bgr(planes[0][i] as i32, planes[1][i] as i32, planes[2][i] as i32, o);
                }
            }
            (WicFormat::Bgr24, w * 3, px)
        }
        4 => {
            // 32bppCMYK. Transform 2 ⇒ YCCK: YCbCr→RGB then libjpeg's CMY = 255−RGB, K kept;
            // otherwise the four components are CMYK directly. Adobe files store inverted samples,
            // and WIC reports them un-inverted (255 − libjpeg's CMYK on every channel).
            let ycck = adobe == Some(2);
            let invert = adobe.is_some();
            let mut px = vec![0u8; w * h * 4];
            for i in 0..w * h {
                let cmyk = if ycck {
                    let mut bgr = [0u8; 3];
                    ycbcr_to_bgr(planes[0][i] as i32, planes[1][i] as i32, planes[2][i] as i32, &mut bgr);
                    [255 - bgr[2], 255 - bgr[1], 255 - bgr[0], planes[3][i]]
                } else {
                    [planes[0][i], planes[1][i], planes[2][i], planes[3][i]]
                };
                let o = &mut px[i * 4..i * 4 + 4];
                for (j, &v) in cmyk.iter().enumerate() {
                    o[j] = if invert { 255 - v } else { v };
                }
            }
            (WicFormat::Cmyk32, w * 4, px)
        }
        _ => return Err(WINCODEC_ERR_BADHEADER),
    };
    Ok(DecodedImage {
        width: w as u32,
        height: h as u32,
        format,
        stride,
        pixels,
        palette: Vec::new(),
    })
}
