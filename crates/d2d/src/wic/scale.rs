//! `IWICBitmapScaler`, as WIC does it. Scales a source to a new size with one of
//! WIC's interpolation modes.
//!
//! NearestNeighbor is a Bresenham DDA, and
//! confirmed byte-exact against the real codec on both axes: the source sample for destination
//! index `d` is `floor(((d+1)*src - 1) / dst)` — i.e. the last source pixel whose left edge is at or
//! before the destination pixel's right edge. Whole pixels are copied, so any whole-byte opaque
//! format (bpp a multiple of 8) is exact — but a straight-alpha 32bpp source still routes through
//! the premultiplied working buffer (like the filtering modes), so its copied pixels come back
//! premultiply→unpremultiply round-tripped, ±1 off a verbatim copy.
//!
//! Linear is reproduced for the alpha-less whole-byte formats (`Gray8`, `Bgr24`/`Rgb24`) and is
//! byte-exact against the real codec there. It is a separable bilinear with a
//! center-aligned 16.16 source position `pos = step*(d + 0.5) - 0.5` (`step = ⌊src·65536/dst⌋`,
//! truncated as a double→int), split into an integer sample + an 8-bit fraction `frac>>8`. The two
//! neighbours (sample and sample+1, each clamped to `[0, src-1]` — the native padded-scanline edge
//! replication) blend as `(w_hi·right + (256-w_hi)·left) >> 8`, i.e. **floored**, not rounded
//! (8bpp bilinear, and the `0xFF00FF` SWAR trick for 32bpp). X runs
//! first into byte temps, then Y blends two temp rows the same way (double floor).
//!
//! For straight-alpha 32bpp (`straight_alpha`), WIC blends on a **premultiplied** 32bpp working
//! buffer and converts back: it premultiplies each source channel by alpha (round, the shared
//! premultiply op `(a·c+128+((a·c+128)>>8))>>8`), runs the same floored bilinear, then
//! unpremultiplies each output pixel via `UnpremultiplyTable` — `out = min(255, (tbl[a]·c)>>16)` with
//! `tbl[a] = (255<<16)/a` and a fully-transparent pixel forced to all-zero. This is byte-exact
//! against the real codec.
//!
//! Cubic (mode 2) is a separable **fixed-point** 4-tap bicubic (Catmull-Rom, a=-0.5).
//! A 16.16 source position `pos = ⌊src·65536/dst⌋·(d+½)` carried as `init=(step-0x10000)>>1`, with the
//! integer part `pos>>16` the centre pixel and the 6-bit phase `(pos&0xffff)>>10` indexing a 129-entry
//! coefficient table `T[j]=round(catmull(j/64)·65536)`. The four taps `ipart-1..ipart+2` use weights
//! `T[phase+64], T[phase], T[64-phase], T[128-phase]` (symmetry), `(Σ wᵢ·pᵢ)>>16` clamped to `[0,255]`;
//! X runs into byte temps (clamped), then Y the same way. Edges clamp to `[0,src-1]`
//! (per-axis X/Y channel scaling against a shared cubic coefficient table).
//!
//! HighQualityCubic (mode 4) is a polyphase resampler with a Catmull-Rom (a=-0.5) kernel: the weight
//! generator builds f32 weight rows — `scale=src/dst`, `support=max(scale,1)`, per dest `d` a
//! centre `(d+½)·scale` and start `⌊centre-2·support⌋`, iterating `numTaps` kernel samples at distance
//! `(start+tap+½)-centre` normalised by `1/support`, folding out-of-range taps into the edge columns of
//! a `windowSize=min(numTaps,src)` row and normalising by their sum (`numTaps=src≥dst ? ⌈4·src/dst⌉+1 :
//! 6`). The apply is a separable f32 weighted sum: X per source row into
//! float rows (`out = Σ src·w`, init with the first product), Y accumulates `acc += w·xrow` over
//! non-zero weights, then `(int)(v+0.5)` saturated to `[0,255]`.
//! Below windowSize 5 the native X path is scalar; the vectorised Y path computes the same.
//!
//! Fant (mode 3) is a separable triangle-coverage area resampler. The setup
//! builds per-dest f32 weight rows by integer coverage accounting (each source pixel's
//! overlap `rem` of `dst` units contributes `rem²` to its slot and `dst·rem-rem²` to the next — the
//! triangle split), normalised by `1/(src·dst)`, with `windowSize = src≤dst ? 2 : src/dst+2` and start
//! `src·d/dst`. The apply is the same separable f32 weighted sum + saturating round as HighQualityCubic.
//!
//! All three scale in native channel count (`Gray8`→1, `Bgr24`→3, `Bgra32`→4); the padded-scanline
//! buffer premultiplies a straight-alpha 32bpp source to PBGRA before scaling, so (like Linear) the
//! `straight_alpha` round trip is premultiply → scale → unpremultiply. Byte-exact against the real codec
//! across up/down/same/1×1 sizes, all four byte formats and the straight-alpha cases.

use super::convert::gray16_to_grayfloat;
use super::WicError;

/// `WICBitmapInterpolationMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interp {
    NearestNeighbor = 0,
    Linear = 1,
    Cubic = 2,
    Fant = 3,
    HighQualityCubic = 4,
}

impl Interp {
    pub fn from_mode(m: i32) -> Option<Interp> {
        Some(match m {
            0 => Interp::NearestNeighbor,
            1 => Interp::Linear,
            2 => Interp::Cubic,
            3 => Interp::Fant,
            4 => Interp::HighQualityCubic,
            _ => return None,
        })
    }
}

/// Nearest-neighbour source index for destination index `d` (native DDA result).
#[inline]
fn nn_index(d: u32, src: u32, dst: u32) -> usize {
    (((d + 1) * src - 1) / dst) as usize
}

/// Scale `srcw`×`srch` pixels (`bpp` bits each) to `dstw`×`dsth` with `mode`. `straight_alpha` must be
/// set for a 32bpp straight-alpha format (`32bppBGRA`/`RGBA`): it enables WIC's premultiply→blend→
/// unpremultiply round trip. Returns tightly-strided destination bytes.
pub fn scale(
    pixels: &[u8],
    srcw: u32,
    srch: u32,
    bpp: u32,
    dstw: u32,
    dsth: u32,
    mode: Interp,
    straight_alpha: bool,
) -> Result<Vec<u8>, WicError> {
    if bpp == 0 || bpp % 8 != 0 {
        return Err(WicError::Unsupported);
    }
    if srcw == 0 || srch == 0 || dstw == 0 || dsth == 0 {
        return Err(WicError::BadImage);
    }
    let bytes = (bpp / 8) as usize;
    if pixels.len() < srcw as usize * srch as usize * bytes {
        return Err(WicError::BadImage);
    }
    if mode == Interp::NearestNeighbor {
        // Even nearest-neighbour routes a straight-alpha 32bpp source through WIC's premultiplied
        // working buffer, so the copied pixels come back premultiply→unpremultiply round-tripped
        // (off by ±1 from a verbatim copy); opaque / non-32bpp formats copy whole pixels exactly.
        if bpp == 32 && straight_alpha {
            let pm = premultiply_bgra(pixels);
            let mut out = nearest(&pm, srcw, srch, bytes, dstw, dsth);
            unpremultiply_bgra(&mut out);
            return Ok(out);
        }
        return Ok(nearest(pixels, srcw, srch, bytes, dstw, dsth));
    }
    // Linear / Cubic / Fant / HighQualityCubic all scale whole-byte formats in native channel count
    // (`bytes` channels). For a straight-alpha 32bpp source the native padded-scanline buffer
    // premultiplies to PBGRA before scaling, so wrap the core with the premultiply round trip
    // (identity for opaque pixels); opaque formats scale raw.
    if bpp != 8 && bpp != 24 && bpp != 32 {
        return Err(WicError::Unsupported);
    }
    let core = |src: &[u8]| -> Vec<u8> {
        match mode {
            Interp::Linear => linear(src, srcw, srch, bytes, dstw, dsth),
            Interp::Cubic => cubic(src, srcw, srch, bytes, dstw, dsth),
            Interp::Fant => fant(src, srcw, srch, bytes, dstw, dsth),
            Interp::HighQualityCubic => hqcubic(src, srcw, srch, bytes, dstw, dsth),
            Interp::NearestNeighbor => unreachable!(),
        }
    };
    if bpp == 32 && straight_alpha {
        let pm = premultiply_bgra(pixels);
        let mut out = core(&pm);
        unpremultiply_bgra(&mut out);
        Ok(out)
    } else {
        Ok(core(pixels))
    }
}

/// Float → byte with saturation and round-half-up:
/// `v<0 → 0`, `v≥255 → 255`, else `(int)(v + 0.5)` (truncate toward zero of the non-negative sum).
#[inline]
fn sat(v: f32) -> u8 {
    if v < 0.0 {
        0
    } else if v >= 255.0 {
        255
    } else {
        (v + 0.5) as u8
    }
}

/// Premultiply each 32bpp pixel's first three channels by its alpha (the shared round op); alpha kept.
fn premultiply_bgra(px: &[u8]) -> Vec<u8> {
    let mut out = px.to_vec();
    for p in out.chunks_exact_mut(4) {
        let a = p[3] as u32;
        for c in &mut p[..3] {
            let v = a * *c as u32 + 128;
            *c = ((v + (v >> 8)) >> 8) as u8;
        }
    }
    out
}

/// Unpremultiply each 32bpp pixel in place (`WicError`-free, `CopyPixels` case 15/68): a fully
/// transparent pixel becomes all-zero, otherwise each channel is `min(255, (tbl[a]·c)>>16)`.
fn unpremultiply_bgra(px: &mut [u8]) {
    for p in px.chunks_exact_mut(4) {
        let a = p[3] as u32;
        if a == 0 {
            p.fill(0);
        } else if a != 255 {
            let t = (255u32 << 16) / a;
            for c in &mut p[..3] {
                *c = (((t * *c as u32) >> 16).min(255)) as u8;
            }
        }
    }
}

/// Per-destination-pixel `(integer source index, right-neighbour weight 0..=255)` for one axis, as
/// The linear scaler computes them: the 16.16 position `step*(d+0.5)-0.5` carried in
/// fixed point, with the fraction's high byte as the weight. The index may be -1 (clamped on read).
fn axis_weights(src: u32, dst: u32) -> Vec<(i32, u32)> {
    let step = ((src as f64 * 65536.0) / dst as f64) as i32;
    let init = (step - 0x10000) >> 1; // position of d=0 (signed 16.16)
    let mut idx = init >> 16;
    let mut frac = (init as u32) & 0xFFFF;
    (0..dst)
        .map(|_| {
            let r = (idx, frac >> 8);
            let acc = frac + step as u32;
            idx += (acc >> 16) as i32;
            frac = acc & 0xFFFF;
            r
        })
        .collect()
}

fn linear(pixels: &[u8], srcw: u32, srch: u32, bytes: usize, dstw: u32, dsth: u32) -> Vec<u8> {
    let xw = axis_weights(srcw, dstw);
    let yw = axis_weights(srch, dsth);
    let (src_stride, dst_stride) = (srcw as usize * bytes, dstw as usize * bytes);
    let clampx = |i: i32| i.clamp(0, srcw as i32 - 1) as usize;
    let clampy = |i: i32| i.clamp(0, srch as i32 - 1) as usize;
    // X-scale source row `sy` into a tightly-packed dst-width byte row.
    let xrow = |sy: usize| -> Vec<u8> {
        let row = &pixels[sy * src_stride..];
        let mut out = vec![0u8; dst_stride];
        for (ox, &(ix, whi)) in xw.iter().enumerate() {
            let (l, r, wlo) = (clampx(ix) * bytes, clampx(ix + 1) * bytes, 256 - whi);
            for b in 0..bytes {
                out[ox * bytes + b] =
                    ((whi * row[r + b] as u32 + wlo * row[l + b] as u32) >> 8) as u8;
            }
        }
        out
    };
    let mut out = vec![0u8; dst_stride * dsth as usize];
    for oy in 0..dsth as usize {
        let (iy, why) = yw[oy];
        // ponytail: re-X-scales the two rows per output line (no scanline cache); fine at WIC sizes.
        let (t0, t1) = (xrow(clampy(iy)), xrow(clampy(iy + 1)));
        let (wlo, drow) = (256 - why, &mut out[oy * dst_stride..(oy + 1) * dst_stride]);
        for i in 0..dst_stride {
            drow[i] = ((why * t1[i] as u32 + wlo * t0[i] as u32) >> 8) as u8;
        }
    }
    out
}

fn nearest(pixels: &[u8], srcw: u32, srch: u32, bytes: usize, dstw: u32, dsth: u32) -> Vec<u8> {
    let (src_stride, dst_stride) = (srcw as usize * bytes, dstw as usize * bytes);
    let mut out = vec![0u8; dst_stride * dsth as usize];
    for oy in 0..dsth {
        let sy = nn_index(oy, srch, dsth);
        let srow = &pixels[sy * src_stride..];
        let drow = &mut out[oy as usize * dst_stride..];
        for ox in 0..dstw as usize {
            let sx = nn_index(ox as u32, srcw, dstw);
            drow[ox * bytes..ox * bytes + bytes]
                .copy_from_slice(&srow[sx * bytes..sx * bytes + bytes]);
        }
    }
    out
}

// --- HighQualityCubic (mode 4): polyphase resampler with a Catmull-Rom kernel ---

/// Catmull-Rom (a=-0.5) weight for a signed sample distance `dist`, distance normalised by
/// `inv = 1/support`. Native f32 op order.
#[inline]
fn hq_kernel(dist: f32, inv: f32) -> f32 {
    if dist == 0.0 {
        return 1.0;
    }
    let t = dist.abs() * inv;
    if t >= 2.0 {
        0.0
    } else if t < 1.0 {
        (t * 1.5 - 2.5) * t * t + 1.0
    } else {
        ((t * -0.5 + 2.5) * t - 4.0) * t + 2.0
    }
}

/// `numTaps` for one axis: `src≥dst ? ⌈4·src/dst⌉+1 : 6`, as the
/// native `ceilf((2x)+(2x))+1` with `x = src/dst`.
fn hq_num_taps(src: u32, dst: u32) -> u32 {
    if src >= dst {
        let x = src as f32 / dst as f32;
        let xx = x + x;
        (xx + xx).ceil() as u32 + 1
    } else {
        6
    }
}

/// `GenerateWeights`: the flat `dst×ws` f32 weight rows plus the `dst` start indices for one axis.
fn hq_weights(src: u32, dst: u32) -> (Vec<f32>, Vec<i32>, usize) {
    let nt = hq_num_taps(src, dst);
    let ws = nt.min(src) as usize;
    let scale = src as f32 / dst as f32;
    let support = scale.max(1.0);
    let inv = 1.0 / support;
    let offset = support + support;
    let srci = src as i32;
    let mut weights = vec![0.0f32; dst as usize * ws];
    let mut starts = vec![0i32; dst as usize];
    for d in 0..dst as usize {
        let center = (d as f32 + 0.5) * scale;
        let start = (center - offset).floor() as i32;
        let (base, mut shiftc) = if start < 0 {
            (0, 0)
        } else if start + ws as i32 > srci {
            (srci - ws as i32, ws as i32 + start - srci)
        } else {
            (start, 0)
        };
        starts[d] = base;
        let row = &mut weights[d * ws..d * ws + ws];
        let mut wsum = 0.0f32;
        let s_plus = start as f32 + 0.5;
        for tap in 0..nt as i32 {
            let w = hq_kernel((tap as f32 + s_plus) - center, inv);
            let idx = start + tap;
            if idx < 0 {
                row[0] += w;
            } else if idx >= srci || shiftc >= ws as i32 {
                row[ws - 1] += w;
            } else {
                row[shiftc as usize] += w;
                shiftc += 1;
            }
            wsum += w;
        }
        if wsum > 0.0 {
            for w in row.iter_mut() {
                *w /= wsum;
            }
        }
    }
    (weights, starts, ws)
}

/// HighQualityCubic: separable f32 polyphase. X scales each source row into interleaved float rows
/// (`out = Σ src·w`, init with the first product); Y accumulates `acc += w·xrow` over non-zero
/// weights; then saturating round. Native channel count `n` (Gray8→1, Bgr24→3, Bgra32→4).
fn hqcubic(px: &[u8], srcw: u32, srch: u32, n: usize, dstw: u32, dsth: u32) -> Vec<u8> {
    let (wx, sx, wsx) = hq_weights(srcw, dstw);
    let (wy, sy, wsy) = hq_weights(srch, dsth);
    let (dw, dh, sw, row_f) = (dstw as usize, dsth as usize, srcw as usize, dstw as usize * n);
    let mut xrows = vec![0.0f32; srch as usize * row_f];
    for r in 0..srch as usize {
        let srow = &px[r * sw * n..];
        let frow = &mut xrows[r * row_f..(r + 1) * row_f];
        for x in 0..dw {
            let (st, wrow) = (sx[x] as usize, &wx[x * wsx..x * wsx + wsx]);
            for c in 0..n {
                let mut acc = 0.0f32;
                for (k, &w) in wrow.iter().enumerate() {
                    let t = srow[(st + k) * n + c] as f32 * w;
                    acc = if k == 0 { t } else { acc + t };
                }
                frow[x * n + c] = acc;
            }
        }
    }
    let mut out = vec![0u8; dh * row_f];
    for y in 0..dh {
        let mut accum = vec![0.0f32; row_f];
        for (k, &w) in wy[y * wsy..y * wsy + wsy].iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let xr = &xrows[(sy[y] as usize + k) * row_f..];
            for (a, &v) in accum.iter_mut().zip(&xr[..row_f]) {
                *a += w * v;
            }
        }
        for (o, &a) in out[y * row_f..(y + 1) * row_f].iter_mut().zip(&accum) {
            *o = sat(a);
        }
    }
    out
}

// --- Cubic (mode 2): fixed-point 4-tap bicubic ---

/// 129-entry Catmull-Rom (a=-0.5) coefficient table `T[j] = round(catmull(j/64)·65536)`, 16.16 fixed
/// point. Computed fresh from the kernel; byte-for-byte identical to the native `cubicCoeffTable`.
fn cubic_table() -> [i32; 129] {
    let mut t = [0i32; 129];
    for (j, e) in t.iter_mut().enumerate() {
        let x = j as f64 / 64.0;
        let k = if x >= 2.0 {
            0.0
        } else if x < 1.0 {
            (1.5 * x - 2.5) * x * x + 1.0
        } else {
            ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
        };
        *e = (k * 65536.0).round() as i32;
    }
    t
}

/// Fixed-point 1D bicubic resample to `dst` clamped bytes. `get(i)` returns the source value at an
/// unclamped index (it clamps internally). 16.16 position `init=(step-0x10000)>>1`, 6-bit phase, four
/// taps `ipart-1..ipart+2` weighted by the symmetric table, `(Σ wᵢ·pᵢ)>>16` saturated.
fn cubic_1d(get: impl Fn(i32) -> i32, src: u32, dst: u32, t: &[i32; 129]) -> Vec<u8> {
    let step = (src as f64 * 65536.0 / dst as f64) as i32;
    let mut pos = (step - 0x10000) >> 1;
    (0..dst)
        .map(|_| {
            let ipart = pos >> 16;
            let phase = ((pos & 0xFFFF) >> 10) as usize;
            let (cl, c0, c1, c2) = (t[phase + 64], t[phase], t[64 - phase], t[128 - phase]);
            let v = (c0 as i64 * get(ipart) as i64
                + cl as i64 * get(ipart - 1) as i64
                + c1 as i64 * get(ipart + 1) as i64
                + c2 as i64 * get(ipart + 2) as i64)
                >> 16;
            pos = pos.wrapping_add(step);
            v.clamp(0, 255) as u8
        })
        .collect()
}

/// Cubic: separable fixed-point bicubic. X pass → clamped byte temps, then Y pass; edges clamp.
fn cubic(px: &[u8], srcw: u32, srch: u32, n: usize, dstw: u32, dsth: u32) -> Vec<u8> {
    let t = cubic_table();
    let (dw, dh, sw, sh) = (dstw as usize, dsth as usize, srcw as usize, srch as usize);
    let cc = |i: i32| i.clamp(0, srcw as i32 - 1) as usize;
    let cr = |i: i32| i.clamp(0, srch as i32 - 1) as usize;
    let mut xr = vec![0u8; sh * dw * n];
    for r in 0..sh {
        let srow = &px[r * sw * n..];
        for c in 0..n {
            let col = cubic_1d(|i| srow[cc(i) * n + c] as i32, srcw, dstw, &t);
            for x in 0..dw {
                xr[(r * dw + x) * n + c] = col[x];
            }
        }
    }
    let mut out = vec![0u8; dw * dh * n];
    for x in 0..dw {
        for c in 0..n {
            let col = cubic_1d(|i| xr[(cr(i) * dw + x) * n + c] as i32, srch, dsth, &t);
            for y in 0..dh {
                out[(y * dw + x) * n + c] = col[y];
            }
        }
    }
    out
}

// --- Fant (mode 3): triangle-coverage area resampler ---

/// Fant per-dest weight rows, start indices and counts for one axis (`HrInitScaling`): integer
/// triangle-coverage accounting (each source pixel's overlap `rem` of `dst` units adds `rem²` to its
/// slot and `dst·rem-rem²` to the next), normalised by `1/(src·dst)`. `windowSize = src≤dst ? 2 :
/// src/dst+2`.
fn fant_weights(src: u32, dst: u32) -> (Vec<Vec<f32>>, Vec<usize>, Vec<usize>) {
    let (s, d) = (src as i32, dst as i32);
    let w_size = if src <= dst { 2 } else { (src / dst) as usize + 2 };
    let recip = 1.0f32 / (src * dst) as f32;
    let mut weights = Vec::with_capacity(dst as usize);
    let mut starts = Vec::with_capacity(dst as usize);
    let mut counts = Vec::with_capacity(dst as usize);
    for out in 0..d {
        let mut row = vec![0.0f32; w_size];
        let mut wi = 0usize;
        let mut rem = d - (out * s) % d;
        let sp0 = (out * s) / d;
        let mut k = usize::from(sp0 < s - 1); // native `sp0 + 1 <= src - 1`
        let mut span = s;
        while rem < span {
            span -= rem;
            row[wi] += rem as f32 * rem as f32;
            row[k] += d as f32 * rem as f32 - rem as f32 * rem as f32;
            wi += 1;
            k += 1;
            rem = d;
            if k as i32 + sp0 > s - 1 {
                if wi as i32 + sp0 > s - 1 {
                    wi = (s - sp0) as usize;
                }
                k = wi;
            }
        }
        let w_frac = rem as f32 * span as f32;
        row[wi] += w_frac;
        let w_rest = d as f32 * span as f32 - w_frac;
        if k >= w_size {
            row[wi] += w_rest;
        } else {
            row[k] += w_rest;
            wi = k;
        }
        while wi > 0 && row[wi] == 0.0 {
            wi -= 1;
        }
        let count = wi + 1;
        for w in row.iter_mut().take(count) {
            *w *= recip;
        }
        weights.push(row);
        starts.push(sp0 as usize);
        counts.push(count);
    }
    (weights, starts, counts)
}

/// Fant: separable f32 weighted sum (same apply + saturating round as HighQualityCubic).
fn fant(px: &[u8], srcw: u32, srch: u32, n: usize, dstw: u32, dsth: u32) -> Vec<u8> {
    let (wx, sx, cx) = fant_weights(srcw, dstw);
    let (wy, sy, cy) = fant_weights(srch, dsth);
    let (dw, dh, sw, row_f) = (dstw as usize, dsth as usize, srcw as usize, dstw as usize * n);
    let cc = |i: i32| i.clamp(0, srcw as i32 - 1) as usize;
    let cr = |i: i32| i.clamp(0, srch as i32 - 1) as usize;
    let mut xrows = vec![0.0f32; srch as usize * row_f];
    for r in 0..srch as usize {
        let srow = &px[r * sw * n..];
        let frow = &mut xrows[r * row_f..(r + 1) * row_f];
        for x in 0..dw {
            let (st, wrow) = (sx[x] as i32, &wx[x]);
            for c in 0..n {
                let mut acc = 0.0f32;
                for (k, &w) in wrow[..cx[x]].iter().enumerate() {
                    let t = srow[cc(st + k as i32) * n + c] as f32 * w;
                    acc = if k == 0 { t } else { acc + t };
                }
                frow[x * n + c] = acc;
            }
        }
    }
    let mut out = vec![0u8; dh * row_f];
    for y in 0..dh {
        let (st, wrow) = (sy[y] as i32, &wy[y]);
        for x in 0..dw {
            for c in 0..n {
                let mut acc = 0.0f32;
                for (k, &w) in wrow[..cy[y]].iter().enumerate() {
                    let t = w * xrows[cr(st + k as i32) * row_f + x * n + c];
                    acc = if k == 0 { t } else { acc + t };
                }
                out[(y * dw + x) * n + c] = sat(acc);
            }
        }
    }
    out
}

/// Fant separable f32 weighted sum for a single-channel float image (no byte quantisation). Same
/// weights and accumulation order as `fant`, but the source and destination samples stay f32.
fn fant_f32(src: &[f32], srcw: u32, srch: u32, dstw: u32, dsth: u32) -> Vec<f32> {
    let (wx, sx, cx) = fant_weights(srcw, dstw);
    let (wy, sy, cy) = fant_weights(srch, dsth);
    let (dw, dh, sw) = (dstw as usize, dsth as usize, srcw as usize);
    let cc = |i: i32| i.clamp(0, srcw as i32 - 1) as usize;
    let cr = |i: i32| i.clamp(0, srch as i32 - 1) as usize;
    let mut xrows = vec![0.0f32; srch as usize * dw];
    for r in 0..srch as usize {
        let srow = &src[r * sw..];
        for x in 0..dw {
            let (st, wrow) = (sx[x] as i32, &wx[x]);
            let mut acc = 0.0f32;
            for (k, &w) in wrow[..cx[x]].iter().enumerate() {
                let t = srow[cc(st + k as i32)] * w;
                acc = if k == 0 { t } else { acc + t };
            }
            xrows[r * dw + x] = acc;
        }
    }
    let mut out = vec![0.0f32; dw * dh];
    for y in 0..dh {
        let (st, wrow) = (sy[y] as i32, &wy[y]);
        for x in 0..dw {
            let mut acc = 0.0f32;
            for (k, &w) in wrow[..cy[y]].iter().enumerate() {
                let t = w * xrows[cr(st + k as i32) * dw + x];
                acc = if k == 0 { t } else { acc + t };
            }
            out[y * dw + x] = acc;
        }
    }
    out
}

/// Fant-scale a `16bppGray` source to `32bppGrayFloat`, as native's BitmapScaler does for Gray16
/// under the Fant resampler: convert each sample to its scRGB-linear grey float, resample in f32, and
/// emit little-endian f32 samples. Returns tightly-strided (4 bytes/pixel) destination bytes.
pub fn scale_gray16_to_grayfloat(src: &[u8], srcw: u32, srch: u32, dstw: u32, dsth: u32) -> Vec<u8> {
    let n = (srcw * srch) as usize;
    let gray: Vec<f32> =
        (0..n).map(|i| gray16_to_grayfloat(u16::from_le_bytes([src[2 * i], src[2 * i + 1]]))).collect();
    fant_f32(&gray, srcw, srch, dstw, dsth).iter().flat_map(|f| f.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray16_fant_grayfloat_matches_native() {
        // native-r5 seed 500031: 2x5 Gray16 → 1x3 GrayFloat (Fant). Byte-exact little-endian f32.
        let src: Vec<u8> = [32830u16, 26871, 54702, 45156, 63231, 6011, 54539, 14774, 20261, 29859]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let got = scale_gray16_to_grayfloat(&src, 2, 5, 1, 3);
        let want: Vec<u8> = [0.3258529603481293f32, 0.44795435667037964, 0.1863327920436859]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn nearest_index_matches_native_dda() {
        // src width 4 → dst 2,3,5 (confirmed against the real codec).
        let got: Vec<_> = (0..2).map(|d| nn_index(d, 4, 2)).collect();
        assert_eq!(got, [1, 3]);
        let got: Vec<_> = (0..3).map(|d| nn_index(d, 4, 3)).collect();
        assert_eq!(got, [1, 2, 3]);
        let got: Vec<_> = (0..5).map(|d| nn_index(d, 4, 5)).collect();
        assert_eq!(got, [0, 1, 2, 3, 3]);
    }

    // NearestNeighbor routes a straight-alpha 32bpp source through the premultiplied working buffer,
    // so a copied pixel comes back premultiply→unpremultiply round-tripped (bgra 138,90,148,110 →
    // 139,90,148,110 on the VM's native scaler), not a verbatim copy. Opaque (a=255) and premultiplied
    // (straight_alpha=false) sources copy exactly.
    #[test]
    fn nearest_straight_alpha_roundtrips_like_native() {
        let px = [138u8, 90, 148, 110];
        let got = scale(&px, 1, 1, 32, 1, 1, Interp::NearestNeighbor, true).unwrap();
        assert_eq!(got, [139, 90, 148, 110], "NN straight-alpha must premul round-trip (native)");
        // premultiplied source: verbatim copy.
        assert_eq!(scale(&px, 1, 1, 32, 1, 1, Interp::NearestNeighbor, false).unwrap(), px);
        // opaque straight-alpha: round trip is identity → verbatim.
        let opaque = [138u8, 90, 148, 255];
        assert_eq!(scale(&opaque, 1, 1, 32, 1, 1, Interp::NearestNeighbor, true).unwrap(), opaque);
    }

    #[test]
    fn cubic_table_matches_native() {
        // Known anchors of the native `cubicCoeffTable` (16.16 Catmull-Rom at j/64).
        let t = cubic_table();
        assert_eq!(t[0], 65536); // kernel(0) = 1.0
        assert_eq!(t[64], 0); // kernel(1) = 0
        assert_eq!(t[128], 0); // kernel(2) = 0
        assert_eq!(t[80], -4608); // kernel(1.25), negative lobe
    }
}
