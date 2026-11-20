//! The filter family at a high buffer precision: a 16/32-bit float or 16-bit UNORM device context
//! (`SetRenderingControls` bufferPrecision). Native keeps an effect's intermediates at the context's
//! precision, so a float graph carries values outside [0,1] (HDR, an emboss's negative side) and
//! sub-LSB fractions through these effects, which the UNORM8 node stores of [`super::filters`] (WARP's
//! 8bpc storage and fixed-point sampling, byte-exact against the 8bpc oracle) cannot. Each effect keeps
//! that module's geometry (output rect and origin) and kernel tables; the arithmetic is plain f32, and
//! a multi-pass effect stores between its passes at the context precision ([`quantize_to`]). Inputs are
//! read in their own format (BGRA8, R16 or R32 float), so a float source keeps its range.

use super::filters::{blur_params, discrete_lookup, discrete_table, gaussian_weights, kernel_taps, LUMA, SHARP_G};
use crate::bitmap::Bitmap;
use crate::effects::{bitmap_texel, quantize_to};

/// Premultiplied f32 pixels with the logical origin of their top-left, in input-pixel space.
pub struct FOut {
    pub w: i32,
    pub h: i32,
    pub ox: i32,
    pub oy: i32,
    pub px: Vec<[f32; 4]>,
}

impl FOut {
    fn new(w: i32, h: i32, ox: i32, oy: i32) -> Self {
        FOut { w, h, ox, oy, px: vec![[0.0; 4]; (w.max(0) * h.max(0)) as usize] }
    }
    /// `input` with its top-left at `org` in the effect's space.
    pub fn of_at(input: &Bitmap, org: [i32; 2]) -> Self {
        let mut f = Self::of(input);
        (f.ox, f.oy) = (org[0], org[1]);
        f
    }
    fn of(input: &Bitmap) -> Self {
        let (w, h) = (input.width as i32, input.height as i32);
        let px = (0..w * h).map(|i| bitmap_texel(input, i % w, i / w)).collect();
        FOut { w, h, ox: 0, oy: 0, px }
    }
    /// The pixel at absolute (input-space) `(x,y)`, off the edge per `edge`.
    fn at(&self, x: i32, y: i32, edge: Edge) -> [f32; 4] {
        let (lx, ly) = (x - self.ox, y - self.oy);
        let refl = |v: i32, n: i32| {
            let p = v.rem_euclid(2 * n);
            if p >= n { 2 * n - 1 - p } else { p }
        };
        let (lx, ly) = match edge {
            Edge::Clamp => (lx.clamp(0, self.w - 1), ly.clamp(0, self.h - 1)),
            Edge::Mirror => (refl(lx, self.w), refl(ly, self.h)),
            Edge::Transparent if lx < 0 || ly < 0 || lx >= self.w || ly >= self.h => return [0.0; 4],
            Edge::Transparent => (lx, ly),
        };
        self.px[(ly * self.w + lx) as usize]
    }
}

/// How a blur reads past its input's edge.
#[derive(Clone, Copy)]
enum Edge {
    Clamp,
    Mirror,
    Transparent,
}

/// Premultiplied texel of `bm`: HARD clamps to the edge, SOFT is transparent outside.
fn texel(bm: &Bitmap, x: i32, y: i32, hard: bool) -> [f32; 4] {
    if hard {
        bitmap_texel(bm, x.clamp(0, bm.width as i32 - 1), y.clamp(0, bm.height as i32 - 1))
    } else {
        bitmap_texel(bm, x, y)
    }
}

/// Straight RGB (input border clamped, as the ±N neighbourhood effects read it).
fn straight_rgb(bm: &Bitmap, x: i32, y: i32, is_straight: bool) -> [f32; 3] {
    let [r, g, b, a] = texel(bm, x, y, true);
    if is_straight || a == 0.0 { [r, g, b] } else { [r / a, g / a, b / a] }
}

fn luminance(bm: &Bitmap, x: i32, y: i32) -> f32 {
    let s = straight_rgb(bm, x, y, false);
    LUMA[0] * s[0] + LUMA[1] * s[1] + LUMA[2] * s[2]
}

/// Bilinear premultiplied sample at scene position `(px,py)` (texel centres at +0.5), transparent
/// outside, with 8-bit fractional weights as the GPU texture filter takes them.
fn bilinear(bm: &Bitmap, px: f32, py: f32) -> [f32; 4] {
    let (fx, fy) = (px - 0.5, py - 0.5);
    let (x0, y0) = (fx.floor(), fy.floor());
    let q = |f: f32| (f * 256.0).floor() / 256.0;
    let (wx, wy) = (q(fx - x0), q(fy - y0));
    let (x0, y0) = (x0 as i32, y0 as i32);
    let (c00, c10, c01, c11) =
        (bitmap_texel(bm, x0, y0), bitmap_texel(bm, x0 + 1, y0), bitmap_texel(bm, x0, y0 + 1), bitmap_texel(bm, x0 + 1, y0 + 1));
    core::array::from_fn(|k| {
        let top = c00[k] + (c10[k] - c00[k]) * wx;
        let bot = c01[k] + (c11[k] - c01[k]) * wx;
        top + (bot - top) * wy
    })
}

/// The discrete Gaussian behind native's bilinear tap pairs ([`kernel_taps`]): texel weights over
/// `−reach..=reach` (index `reach` is the centre), exactly the pairs' two-texel split.
fn gaussian_texel_weights(sigma: f32, support: f32) -> (Vec<f32>, i32) {
    let taps = kernel_taps(sigma, support);
    let reach = 2 * taps.len() as i32 - 1;
    let mut w = vec![0.0f32; (2 * reach + 1) as usize];
    for (i, &(off, wt)) in taps.iter().enumerate() {
        let f = off - (2 * i) as f32;
        for sign in [1i32, -1] {
            w[(reach + sign * 2 * i as i32) as usize] += wt * (1.0 - f);
            w[(reach + sign * (2 * i as i32 + 1)) as usize] += wt * f;
        }
    }
    (w, reach)
}

/// GaussianBlur: the discrete Gaussian of σ (native's kernel table) applied separably, X then Y.
/// Output rect as [`super::filters::gaussian_blur`] (SOFT grows by the kernel reach). At full
/// resolution HARD clamps at the edge; once σ takes native's halving path HARD runs that graph
/// ([`hard_graph`]), as native does at every buffer precision.
/// ponytail: SOFT's halving and either border's pre-downscale (`s < 1`, large σ) still run at full
/// resolution, their ideal limit; port them from [`super::filters`]' `blur_graph` when a float
/// effect needs them.
pub fn gaussian_blur(input: &Bitmap, sigma: f32, mode: u32, hard: bool, texture: bool, p: u32, place: &BlurPlace) -> FOut {
    gaussian_blur_ex(input, [0, 0], sigma, mode, hard, texture, p, place, None).0
}

/// [`gaussian_blur`] of an input whose top-left sits at `org` in the effect's space, also giving what a HARD blur's
/// last pass reads ([`LastPass`]), and taking its realized input ready-made (`apron`, over [`HardRects::input`])
/// instead of drawn from `input` by the border's copy. A HARD blur computes `place.need` of its output (all of it
/// when unset), as Direct2D sizes its passes from the rect asked of it; the output's origin is relative to the input.
#[allow(clippy::too_many_arguments)]
pub fn gaussian_blur_ex(input: &Bitmap, org: [i32; 2], sigma: f32, mode: u32, hard: bool, texture: bool, p: u32, place: &BlurPlace, apron: Option<&FOut>) -> (FOut, Option<LastPass>) {
    if sigma <= 0.0 {
        return (FOut::of(input), None);
    }
    let (k, c, threshold, support, msl) = blur_params(mode);
    if hard {
        let mut src = FOut::of(input);
        (src.ox, src.oy) = (org[0], org[1]);
        let bounds = [org[0], org[1], org[0] + src.w, org[1] + src.h];
        let r = hard_rects(bounds, blur_need(bounds, place.need), sigma, mode, texture).expect("σ > 0");
        let (mut out, last) = if sigma <= threshold {
            hard_full(&src, &r, sigma, support, texture, p, place, apron)
        } else if k / (sigma + c) >= 1.0 {
            hard_graph(src, true, &r, bounds, sigma, support, 0.5, p, place, apron)
        } else {
            let unorm8 = texture && input.format == crate::abi::DXGI_FORMAT_B8G8R8A8_UNORM;
            hard_downscaled(&src, &r, bounds, sigma, (k / (sigma + c)).clamp(0.1, 1.0), support, msl, unorm8, p, place, apron)
        };
        out.ox -= org[0];
        out.oy -= org[1];
        return (out, Some(last));
    }
    if sigma > threshold && k / (sigma + c) >= 1.0 {
        return (soft_halving(input, sigma, support, p), None);
    }
    // ponytail: a SOFT σ whose scale k/(σ+c) is below 1 pre-downscales natively ([`super::filters`]'
    // `soft_downscaled`); this full-resolution blur stands in until that geometry is ported at float.
    let edge = match (hard, sigma > threshold) {
        (false, _) => Edge::Transparent,
        (true, false) => Edge::Clamp,
        (true, true) => Edge::Mirror,
    };
    let (iw, ih) = (input.width as i32, input.height as i32);
    let r0 = if hard {
        0
    } else {
        let r = libm::ceilf(sigma * support).max(1.0);
        (2.0 * libm::ceilf(r * 0.5) - 1.0).max(1.0) as i32
    };
    // Accumulate in native's blur-PS order ([`tap_sum`]), one bilinear sample per tap side. Summing
    // the flattened symmetric texel weights end-to-end instead drifts the f32 sum by ~1 ULP, so a
    // constant (DC) HDR field blurs to 2.0000002 rather than native's exact 2.0 (precision-10).
    let taps = kernel_taps(sigma, support);
    let src = FOut::of(input);
    let mut xp = FOut::new(iw + 2 * r0, ih + 2 * r0, -r0, -r0);
    for jy in 0..xp.h {
        for jx in 0..xp.w {
            let (x, y) = (xp.ox + jx, xp.oy + jy);
            xp.px[(jy * xp.w + jx) as usize] = quantize_to(p, tap_sum(&taps, |d| src.at(x + d, y, edge)));
        }
    }
    let mut yp = FOut::new(xp.w, xp.h, xp.ox, xp.oy);
    for jy in 0..yp.h {
        for jx in 0..yp.w {
            let (x, y) = (yp.ox + jx, yp.oy + jy);
            yp.px[(jy * yp.w + jx) as usize] = tap_sum(&taps, |d| xp.at(x, y + d, edge));
        }
    }
    (yp, None)
}

/// Where Direct2D's intermediate allocator put a HARD blur's intermediates in their atlases: the atlas pixel of
/// each one's top-left corner (the realized input, the pre-downscale, the X pass, the Y pass) and of the
/// output the last pass draws into. The default, all at the origin, is a blur alone in its graph.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlurPlace {
    pub input: [i32; 2],
    /// The shrunk image a pre-downscale whose apron [`HardRects::shrink`] splits draws first.
    pub shrink: [i32; 2],
    pub pre: [i32; 2],
    pub x: [i32; 2],
    pub y: [i32; 2],
    pub out: [i32; 2],
    /// The input is a HARD blur whose last pass runs in this blur's border ([`border_fold`]).
    pub fold: bool,
    /// The rect of the output its consumers read, in the effect's space; `None` is all of it.
    pub need: Option<[i32; 4]>,
    /// The intermediate the border samples when its input is drawn: its atlas place and its rect in the effect's
    /// space.
    pub src: Option<([i32; 2], [i32; 4])>,
}

/// The part of a HARD blur's output (its input's `bounds`) that is computed for `need` (all of it when unset or
/// empty).
pub fn blur_need(bounds: [i32; 4], need: Option<[i32; 4]>) -> [i32; 4] {
    let Some(n) = need else { return bounds };
    let r = [n[0].max(bounds[0]), n[1].max(bounds[1]), n[2].min(bounds[2]), n[3].min(bounds[3])];
    if r[2] <= r[0] || r[3] <= r[1] { bounds } else { r }
}

/// The rects of a HARD blur's intermediates, in the effect's space: the input realized for it (absent
/// when the input is a bitmap, which the first pass samples itself), the pre-downscale, the X and Y passes,
/// and the output the last pass draws. `y` is absent at full resolution, where the Y pass draws the output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HardRects {
    pub input: Option<[i32; 4]>,
    /// The shrunk image alone, `[0, p)`, when the pre-downscale's mirrored apron spans more than 25 copies of it:
    /// Direct2D then shrinks first and mirrors the apron in a second pass.
    pub shrink: Option<[i32; 4]>,
    pub pre: Option<[i32; 4]>,
    pub x: [i32; 4],
    pub y: Option<[i32; 4]>,
    pub out: [i32; 4],
}

/// [`HardRects`] for a HARD blur of an input over `bounds` whose output rect `need` is asked for, both in the effect's
/// space, as Direct2D lays them out: every pass covers what the next one reads of it, on grids anchored at the
/// effect's origin (a halved texel `j` spans pixels `2j` and `2j + 1`; the shrunk grid scales the effect's space
/// by `s`), reflecting at the input's bounds. `None` for σ ≤ 0 (no blur).
pub fn hard_rects(bounds: [i32; 4], need: [i32; 4], sigma: f32, mode: u32, texture: bool) -> Option<HardRects> {
    if sigma <= 0.0 {
        return None;
    }
    let (k, c, threshold, support, _) = blur_params(mode);
    let realized = |r: [i32; 4]| (!texture).then_some(r);
    let [nl, nt, nr, nb] = need;
    if sigma <= threshold {
        let r = 2 * kernel_taps(sigma, support).len() as i32 - 1;
        return Some(HardRects { input: realized([nl - r, nt - r, nr + r, nb + r]), shrink: None, pre: None, x: [nl, nt - r, nr, nb + r], y: None, out: need });
    }
    let ratio = k / (sigma + c);
    let s = ratio.clamp(0.1, 1.0);
    let (sh, sig) = if ratio >= 1.0 { (0.5, sigma) } else { (s * 0.5, sigma * s) };
    let r = 2 * kernel_taps(sig, support).len() as i32 - 1;
    let y = upscale_reads(need, sh);
    // A pass halving an axis reads the two texels it averages and the kernel's reach `r` past them.
    let apron = [2 * y[0] - r, 2 * y[1] - r, 2 * (y[2] - 1) + r + 1, 2 * (y[3] - 1) + r + 1];
    let x = [y[0], apron[1], y[2], apron[3]];
    if ratio >= 1.0 {
        return Some(HardRects { input: realized(apron), shrink: None, pre: None, x, y: Some(y), out: need });
    }
    let j = shrunk_image(bounds, s);
    // The shrunk texels the pre-downscale reads: its apron folded into the image, which repeats mirrored about its
    // edges (all of it once the apron spans a full mirror period).
    let reads = |k: usize| -> (i32, i32) {
        let (lo, hi, a, b) = (j[k], j[k + 2], apron[k], apron[k + 2]);
        let n = hi - lo;
        if n <= 0 || b - a >= 2 * n { return (lo, hi); }
        // A texel past an edge reads its mirror image about that edge, `2·edge − q − 1`.
        let fold = |q: i32| {
            let m = (q - lo).rem_euclid(2 * n);
            lo + if m < n { m } else { 2 * n - 1 - m }
        };
        (a..b).map(fold).fold((hi, lo), |(l, h), q| (l.min(q), h.max(q + 1)))
    };
    let (rx, ry) = (reads(0), reads(1));
    let read = [rx.0, ry.0, rx.1, ry.1];
    // What the shrink samples: those texels mapped back through the scale as its vertex shader inverts it,
    // `(s·j)/s²`, half a texel wider for the bilinear, rounded out (DDI: a clipped draw's input covers only the
    // part of the image its apron reads, at x 7..49 of a 48-wide image for apron texels 5..42 of 32).
    let back = |v: i32| (s * v as f32) / (s * s);
    let input = [
        libm::floorf(back(read[0]) - 0.5) as i32,
        libm::floorf(back(read[1]) - 0.5) as i32,
        libm::ceilf(back(read[2]) + 0.5) as i32,
        libm::ceilf(back(read[3]) + 0.5) as i32,
    ];
    let split = split_apron(apron[2] - apron[0], apron[3] - apron[1], j[2] - j[0], j[3] - j[1]);
    Some(HardRects { input: realized(input), shrink: split.then_some(j), pre: Some(apron), x, y: Some(y), out: need })
}

/// The shrunk image a HARD blur's pre-downscale makes of an input over `bounds` (its output's bounds), or `None` when
/// the blur does not pre-downscale.
pub fn hard_shrunk(bounds: [i32; 4], sigma: f32, mode: u32) -> Option<[i32; 4]> {
    let (k, c, threshold, _, _) = blur_params(mode);
    let ratio = k / (sigma + c);
    (sigma > threshold && ratio < 1.0).then(|| shrunk_image(bounds, ratio.clamp(0.1, 1.0)))
}

/// Whether a pre-downscale of `w × h` texels (apron included) over a `pw × ph` shrunk image is drawn in two passes:
/// when the apron spans more than 25 whole copies of the image, Direct2D shrinks into `[0, p)` first and mirrors
/// the apron from that.
fn split_apron(w: i32, h: i32, pw: i32, ph: i32) -> bool {
    (w / pw) * (h / ph) > 25
}

/// The texels of a halved Y pass (pre-downscaled first when `sh < ½`) the upscale reads for the output rect `need`:
/// its vertex mapping `(m·x + c)/z` (`m = 1/sh`, `z = m²`, `c = z/4`) puts an output pixel's centre `x` at
/// `x·sh + ¼`, and the bilinear reads the texel half a texel before that and the next. Direct2D sizes the passes
/// (and the pre-downscale) from this.
fn upscale_reads(need: [i32; 4], sh: f32) -> [i32; 4] {
    let m = 1.0 / sh;
    let (z, c) = (m * m, m * m * 0.25);
    let at = |x: f32| (m * x + c) / z - 0.5;
    [
        libm::floorf(at(need[0] as f32 + 0.5)) as i32,
        libm::floorf(at(need[1] as f32 + 0.5)) as i32,
        libm::ceilf(at(need[2] as f32 - 0.5)) as i32 + 1,
        libm::ceilf(at(need[3] as f32 - 0.5)) as i32 + 1,
    ]
}

/// The shrunk image a pre-downscale by `s` makes of an input over `bounds`: each edge scaled and rounded to the
/// nearest texel, ties to even, on the grid anchored at the effect's origin (at least one texel wide).
fn shrunk_image(bounds: [i32; 4], s: f32) -> [i32; 4] {
    let [l, t, r, b] = bounds.map(|v| (v as f32 * s).round_ties_even() as i32);
    [l, t, r.max(l + 1), b.max(t + 1)]
}

/// A quad's texcoords at its edges `e` (`[left, top, right, bottom]`) when the effect's origin sits at texel `org`
/// of a texture `inv` = 1/size.
fn edge_tex(e: [i32; 4], org: [i32; 2], inv: [f32; 2]) -> [f32; 4] {
    [(e[0] + org[0]) as f32 * inv[0], (e[1] + org[1]) as f32 * inv[1], (e[2] + org[0]) as f32 * inv[0], (e[3] + org[1]) as f32 * inv[1]]
}

/// SOFT GaussianBlur on native's halving path without a pre-downscale (`σ > threshold`, `s == 1`): [`super::filters`]'
/// device-anchored SOFT graph at the context precision. The X pass subsamples X by 2 into an intermediate grown by the
/// reach `rb = r + 2` (Y full-res) with a half-res near-side apron `ap = ⌊rb/2⌋`, transparent outside; the Y pass
/// subsamples Y the same way; a LINEAR ×2 upscale (Y lerp, then X) over Direct2D's SOFT output span returns to device
/// pixels at `(d + ½)·½ + ¼ + ap`. Each pass is the blur PS ([`tap_sum_at`]); intermediates store at precision `p`.
fn soft_halving(input: &Bitmap, sigma: f32, support: f32, p: u32) -> FOut {
    let taps = kernel_taps(sigma, support);
    let src = FOut::of(input);
    let (iw, ih) = (src.w, src.h);
    let half = |v: i32| (v + 1) / 2;
    let r = libm::ceilf(sigma * support).max(1.0) as i32;
    let rb = r + 2;
    let ap = rb / 2;
    let mut xp = FOut::new(half(iw + 2 * rb), ih + 2 * rb, 0, 0);
    for jy in 0..xp.h {
        for jx in 0..xp.w {
            let v = tap_sum_at(&taps, (2 * (jx - ap)) as f32, false, |i| src.at(i, jy - rb, Edge::Transparent));
            xp.px[(jy * xp.w + jx) as usize] = quantize_to(p, v);
        }
    }
    let mut yp = FOut::new(xp.w, half(ih + 2 * rb), 0, 0);
    for jy in 0..yp.h {
        for jx in 0..yp.w {
            let v = tap_sum_at(&taps, (2 * (jy - ap) + rb) as f32, false, |i| xp.at(jx, i, Edge::Transparent));
            yp.px[(jy * yp.w + jx) as usize] = quantize_to(p, v);
        }
    }
    let m = taps.last().map_or(0.0, |t| t.0);
    let ((ol, ow), (ot, oh)) = (super::filters::soft_bounds(iw, 1.0, 0.5, false, m), super::filters::soft_bounds(ih, 1.0, 0.5, false, m));
    let mut out = FOut::new(ow, oh, ol, ot);
    // Texel coordinate of a device pixel in yp: centre (d + ½)·½ + ¼ + ap, minus the texel half.
    let q = |d: i32| {
        let u = (d as f32 + 0.5) * 0.5 + 0.25 + ap as f32 - 0.5;
        (u.floor() as i32, ((u - u.floor()) * 256.0).floor() / 256.0)
    };
    for jy in 0..oh {
        let (y0, wy) = q(ot + jy);
        for jx in 0..ow {
            let (x0, wx) = q(ol + jx);
            let col = |x: i32| {
                let (a, b) = (yp.at(x, y0, Edge::Transparent), yp.at(x, y0 + 1, Edge::Transparent));
                core::array::from_fn::<f32, 4, _>(|k| a[k] + (b[k] - a[k]) * wy)
            };
            let (a, b) = (col(x0), col(x0 + 1));
            out.px[(jy * ow + jx) as usize] = core::array::from_fn(|k| a[k] + (b[k] - a[k]) * wx);
        }
    }
    out
}

/// HARD GaussianBlur at full resolution (`σ ≤ threshold`), as Direct2D draws it: an X pass over the needed columns and
/// the rows `r` past them (`r = 2n − 1` for `n` tap pairs), reading the bitmap itself (mirror addressing) or another
/// effect's output realized into an atlas with an `r` apron mirrored around it; then a Y pass straight into the
/// output. Both run Direct2D's blur shaders over their quads ([`Pass`]).
#[allow(clippy::too_many_arguments)]
fn hard_full(src: &FOut, rects: &HardRects, sigma: f32, support: f32, texture: bool, p: u32, place: &BlurPlace, apron: Option<&FOut>) -> (FOut, LastPass) {
    let taps = kernel_taps(sigma, support);
    let atlas = |e: i32| (e as u32).next_power_of_two().max(512) as f32;
    let (x, o) = (rects.x, rects.out);
    // The texture the X pass reads, and the texel of the effect's origin in it: the bitmap itself, or the atlas
    // holding the input realized with its apron.
    let (x_size, org, rz) = match rects.input {
        Some(rz) => ([atlas(rz[2] - rz[0]), atlas(rz[3] - rz[1])], [place.input[0] - rz[0], place.input[1] - rz[1]], rz),
        None => ([src.w as f32, src.h as f32], [-src.ox, -src.oy], [0; 4]),
    };
    let x_geom = Geom { w: x[2] - x[0], h: x[3] - x[1], at: place.x, scene: x.map(|v| v as f32), tex: edge_tex(x, org, [1.0 / x_size[0], 1.0 / x_size[1]]), size: x_size };
    let x_pass = Pass::new(&x_geom, 0, 1.0, &taps);
    let mut xp = FOut::new(x_geom.w, x_geom.h, x[0], x[1]);
    for cy in 0..xp.h {
        for cx in 0..xp.w {
            let v = x_pass.px(cx, cy, &taps, texture, |ax, ay| {
                let (x, y) = (ax - org[0], ay - org[1]);
                if texture {
                    return src.at(x, y, Edge::Mirror);
                }
                let (x, y) = (x.clamp(rz[0], rz[2] - 1), y.clamp(rz[1], rz[3] - 1));
                match apron {
                    Some(a) => a.at(x, y, Edge::Clamp),
                    None => src.at(x, y, Edge::Mirror),
                }
            });
            xp.px[(cy * xp.w + cx) as usize] = quantize_to(p, v);
        }
    }
    // The texel of the effect's origin in the X pass's atlas.
    let xo = [place.x[0] - x[0], place.x[1] - x[1]];
    let y_size = [atlas(x_geom.w), atlas(x_geom.h)];
    let y_geom = Geom { w: o[2] - o[0], h: o[3] - o[1], at: place.out, scene: o.map(|v| v as f32), tex: edge_tex(o, xo, [1.0 / y_size[0], 1.0 / y_size[1]]), size: y_size };
    let y_pass = Pass::new(&y_geom, 1, 1.0, &taps);
    let mut out = FOut::new(y_geom.w, y_geom.h, o[0], o[1]);
    for cy in 0..out.h {
        for cx in 0..out.w {
            out.px[(cy * out.w + cx) as usize] = y_pass.px(cx, cy, &taps, false, |ax, ay| xp.at(ax - xo[0], ay - xo[1], Edge::Clamp));
        }
    }
    let bounds = [src.ox, src.oy, src.ox + src.w, src.oy + src.h];
    (out, LastPass { kind: LastKind::Y(taps), src: xp, org: xo, size: y_size, bounds })
}

/// HARD GaussianBlur on Direct2D's pre-downscale path (`σ > threshold`, `s = k/(σ+c) < 1`): the input first shrinks by
/// `s` through WARP's minify sampler (normalised texcoord planes and Q8 addressing, a float bilinear with an 8-bit
/// fraction, averaging the four rotated-grid sub-samples for MULTI_SAMPLE_LINEAR) into the shrunk image ([`shrunk_image`]),
/// whose edges are where the passes reflect, stored at precision `p`; the kernel
/// at `σ·s` then runs on that as [`hard_graph`], and the LINEAR upscale by `1/(s·½)` returns to the input.
#[allow(clippy::too_many_arguments)]

fn hard_downscaled(src: &FOut, rects: &HardRects, bounds: [i32; 4], sigma: f32, s: f32, support: f32, msl: bool, unorm8: bool, p: u32, place: &BlurPlace, apron: Option<&FOut>) -> (FOut, LastPass) {
    const MSAA4: [(f32, f32); 4] = [(0.375, -0.125), (-0.125, -0.375), (-0.375, 0.125), (0.125, 0.375)];
    let texture = rects.input.is_none();
    let j = shrunk_image(bounds, s);
    let n = [j[2] - j[0], j[3] - j[1]];
    // The pre-downscale carries the X pass's reach around the texels the passes centre on, as copies of the shrunk
    // image mirrored about its edges (repeatedly, when the reach is wider than the image). Every texel is resampled
    // from the input: its mirrored centre plus the sub-sample offsets (the same offsets in every copy), scaled by
    // `1/s`.
    let pr = rects.pre.expect("a pre-downscale");
    let mut pre = FOut::new(pr[2] - pr[0], pr[3] - pr[1], pr[0], pr[1]);
    let atlas = |e: i32| (e as u32).next_power_of_two().max(512) as f32;
    // A wide apron is mirrored in a second pass from the shrunk image alone ([`split_apron`]).
    let split = rects.shrink.is_some();
    let mut shrunk_img = FOut::new(n[0], n[1], j[0], j[1]);
    // The minify sampler lerps Y before X; reading the bitmap itself it rounds the Q8 address to nearest,
    // reading another effect's output (realized into an atlas with a clamped border) it floors it.
    // The texel of the effect's origin: in the bitmap itself, or in the atlas holding the input realized.
    let (org, size) = match rects.input {
        None => ([-src.ox, -src.oy], [src.w as f32, src.h as f32]),
        Some(rz) => ([place.input[0] - rz[0], place.input[1] - rz[1]], [atlas(rz[2] - rz[0]), atlas(rz[3] - rz[1])]),
    };
    let sample = |u: f32, v: f32| -> [f32; 4] {
        let q8 = |t: f32, k: usize| {
            let q = (t * size[k] - 0.5) * 256.0;
            let q = if texture { q.round_ties_even() } else { q.floor() } as i32;
            (q >> 8, (q & 0xff) as f32 / 256.0)
        };
        let ((x0, fx), (y0, fy)) = (q8(u, 0), q8(v, 1));
        let t = |x: i32, y: i32| match apron {
            Some(a) if !texture => a.at(x - org[0], y - org[1], Edge::Clamp),
            _ => src.at(x - org[0], y - org[1], Edge::Clamp),
        };
        if unorm8 {
            // A UNORM8 bitmap goes through WARP's integer filter (`value·128`, X lerped before Y).
            let byte = |x: i32, y: i32| t(x, y).map(|v| (v * 255.0).round() as i32);
            let w = |f: f32| (f * 256.0) as i32;
            return crate::effects::sample::bilin_v128_at(x0, w(fx), y0, w(fy), byte).map(crate::sampling::norm_v128);
        }
        let (c00, c10, c01, c11) = (t(x0, y0), t(x0 + 1, y0), t(x0, y0 + 1), t(x0 + 1, y0 + 1));
        core::array::from_fn(|c| {
            let left = c00[c] + (c01[c] - c00[c]) * fy;
            let right = c10[c] + (c11[c] - c10[c]) * fy;
            left + (right - left) * fx
        })
    };
    let inv = size.map(|v| 1.0 / v);
    // A shrunk position folded into the image, which repeats mirrored about its edges.
    let mirror = |q: i32, k: usize| {
        let m = (q - j[k]).rem_euclid(2 * n[k]);
        j[k] + if m > n[k] { 2 * n[k] - m } else { m }
    };
    if !msl {
        // LINEAR: one sample at the mirrored texel centre, scaled by `1/s`.
        let mirror_f = |q: f32, k: usize| {
            let (o, n) = (j[k] as f32, n[k] as f32);
            let m = (q - o).rem_euclid(2.0 * n);
            o + if m > n { 2.0 * n - m } else { m }
        };
        let dst = if split { &mut shrunk_img } else { &mut pre };
        for jy in 0..dst.h {
            for jx in 0..dst.w {
                let (tx, ty) = (mirror_f((dst.ox + jx) as f32 + 0.5, 0), mirror_f((dst.oy + jy) as f32 + 0.5, 1));
                let v = sample((tx / s + org[0] as f32) * inv[0], (ty / s + org[1] as f32) * inv[1]);
                dst.px[(jy * dst.w + jx) as usize] = quantize_to(p, v);
            }
        }
    } else {
        // MULTI_SAMPLE_LINEAR: Direct2D draws the shrunk image as quads split at the mirror lines (the image's edges
        // repeated), each carrying its mirrored shrunk-space position `a` and texcoord `(a + border)/W` at the
        // corners. Per corner its vertex shader maps the four rotated-grid offsets `o` back into the input,
        // `(a + o)·s / s²`, and emits `t + (that − a)/W`; WARP interpolates each over the quad's two triangles.
        // The pixel shader averages the four bilinear samples.
        let s2 = s * s;
        let corner = |a: [i32; 2]| -> [[f32; 2]; 4] {
            let (ax, ay) = (a[0] as f32, a[1] as f32);
            let t = [(ax + org[0] as f32) * inv[0], (ay + org[1] as f32) * inv[1]];
            core::array::from_fn(|i| {
                let (ox, oy) = MSAA4[i];
                let x = (s * (ax + ox)) / s2;
                let y = ((ay + oy) * s) / s2;
                [-(ax - x) * inv[0] + t[0], -(ay - y) * inv[1] + t[1]]
            })
        };
        // Split, the shrunk image alone is one quad at its own place; otherwise the pre-downscale's quads are cut at
        // the mirror lines.
        let cuts = |k: usize| -> Vec<i32> {
            if split {
                return vec![j[k], j[k + 2]];
            }
            let (lo, hi) = (pr[k], pr[k + 2]);
            let mut e = vec![lo];
            e.extend((((lo - j[k]).div_euclid(n[k]) + 1)..).map(|m| j[k] + m * n[k]).take_while(|&v| v < hi).filter(|&v| v > lo));
            e.push(hi);
            e
        };
        let (xs, ys) = (cuts(0), cuts(1));
        // The atlas pixel of the effect's origin in the target the quads draw.
        let po = if split { [place.shrink[0] - j[0], place.shrink[1] - j[1]] } else { [place.pre[0] - pr[0], place.pre[1] - pr[1]] };
        let dst = if split { &mut shrunk_img } else { &mut pre };
        use crate::warp::plane::setup_triangle_attr;
        for cy in ys.windows(2) {
            for cx in xs.windows(2) {
                let (l, t, r, b) = (cx[0] + po[0], cy[0] + po[1], cx[1] + po[0], cy[1] + po[1]);
                let at = |x: i32, y: i32| corner([mirror(x - po[0], 0), mirror(y - po[1], 1)]);
                let tris = [[(l, t), (r, b), (r, t)], [(l, b), (l, t), (r, b)]];
                let planes = tris.map(|tri| {
                    let (sx, sy) = (tri.map(|(x, _)| x as i64 * 256), tri.map(|(_, y)| y as i64 * 256));
                    let v = tri.map(|(x, y)| at(x, y));
                    core::array::from_fn::<[[f32; 3]; 2], 4, _>(|i| core::array::from_fn(|k| setup_triangle_attr(sx, sy, v.map(|c| c[i][k]))))
                });
                let (w, h) = ((r - l) as f32, (b - t) as f32);
                for ay in t..b {
                    for ax in l..r {
                        let pc = [ax as f32 + 0.5, ay as f32 + 0.5];
                        let k = usize::from((pc[1] - t as f32) * w > (pc[0] - l as f32) * h);
                        let mut acc = [0.0f32; 4];
                        for plane in &planes[k] {
                            let e = |a: &[f32; 3]| a[0] * pc[0] + (a[1] * pc[1] + a[2]);
                            let v = sample(e(&plane[0]), e(&plane[1]));
                            (0..4).for_each(|c| acc[c] += v[c]);
                        }
                        dst.px[((ay - po[1] - dst.oy) * dst.w + ax - po[0] - dst.ox) as usize] = quantize_to(p, acc.map(|v| v * 0.25));
                    }
                }
            }
        }
    }
    if split {
        mirror_apron(&shrunk_img, &mut pre, place, p);
    }
    hard_graph(pre, false, rects, bounds, sigma * s, support, s * 0.5, p, place, None)
}

/// Direct2D's mirror pass over a split pre-downscale's quad (`pre`, at its place), from the shrunk image `img` at
/// its own place. The vertex shader emits each corner's position as a fraction of the image, `x/p`, and the image's
/// origin texcoord, `(0 − x)/size + t`; WARP interpolates them, and the pixel shader folds the fraction into the
/// image (`|round(frac(u/2)) − frac(u)|`, even copies forward, odd ones back) and point-samples it there.
fn mirror_apron(img: &FOut, pre: &mut FOut, place: &BlurPlace, p: u32) {
    use crate::warp::plane::setup_triangle_attr;
    let size = [img.w, img.h].map(|e| (e as u32).next_power_of_two().max(512) as f32);
    let inv = size.map(|v| 1.0 / v);
    let (w, h) = (pre.w, pre.h);
    let [l, t] = place.pre;
    // Positions from the image's top-left, where its mirror lines start.
    let scene = [pre.ox - img.ox, pre.oy - img.oy, pre.ox - img.ox + w, pre.oy - img.oy + h];
    let corner = |(cx, cy): (i32, i32)| -> [f32; 4] {
        let (x, y) = (if cx == 0 { scene[0] } else { scene[2] } as f32, if cy == 0 { scene[1] } else { scene[3] } as f32);
        let tx = (x + place.shrink[0] as f32) * inv[0];
        let ty = (y + place.shrink[1] as f32) * inv[1];
        [x / img.w as f32, y / img.h as f32, -x * inv[0] + tx, -y * inv[1] + ty]
    };
    let tris = [[(0, 0), (w, h), (w, 0)], [(0, h), (0, 0), (w, h)]];
    let planes = tris.map(|tri| {
        let (sx, sy) = (tri.map(|(x, _)| (l + x) as i64 * 256), tri.map(|(_, y)| (t + y) as i64 * 256));
        let v = tri.map(corner);
        core::array::from_fn::<[f32; 3], 4, _>(|i| setup_triangle_attr(sx, sy, v.map(|c| c[i])))
    });
    let scale = [img.w as f32 * inv[0], img.h as f32 * inv[1]];
    let fold = |u: f32| {
        let frac = |v: f32| v - v.floor();
        ((frac(u * 0.5)).round_ties_even() - frac(u)).abs()
    };
    for jy in 0..h {
        for jx in 0..w {
            let pc = [jx as f32 + 0.5, jy as f32 + 0.5];
            let k = usize::from(pc[1] * w as f32 > pc[0] * h as f32);
            let q = [(l + jx) as f32 + 0.5, (t + jy) as f32 + 0.5];
            let a: [f32; 4] = core::array::from_fn(|i| planes[k][i][0] * q[0] + (planes[k][i][1] * q[1] + planes[k][i][2]));
            let tc = [fold(a[0]) * scale[0] + a[2], fold(a[1]) * scale[1] + a[3]];
            let texel = |k: usize| (tc[k] * size[k]).floor() as i32 - place.shrink[k];
            pre.px[(jy * w + jx) as usize] = quantize_to(p, img.at(texel(0) + img.ox, texel(1) + img.oy, Edge::Clamp));
        }
    }
}

/// HARD GaussianBlur's halving graph, as Direct2D draws it, over `src`: the input when `direct` (native's halving path,
/// `σ > threshold`, `s == 1`), else its pre-downscale. The X pass subsamples X by 2 into an intermediate aproned by
/// the kernel reach `r` in Y; the Y pass subsamples Y by 2 from it; a LINEAR upscale (Y lerp, then X, clamped)
/// samples it at `x·sh + ¼` for each output pixel centre `x`. The passes cover what the next one reads ([`HardRects`]),
/// reflecting at the input's `bounds`, and run Direct2D's blur shaders over their quads ([`Pass`]); the direct input
/// is a bitmap when `rects` has no realized input. Intermediates store at precision `p`. On a float context this
/// is what native runs for Paint.NET's Glow (σ 2.5, QUALITY): the full-resolution blur was up to 2/255 off it.
#[allow(clippy::too_many_arguments)]
fn hard_graph(src: FOut, direct: bool, rects: &HardRects, bounds: [i32; 4], sigma: f32, support: f32, sh: f32, p: u32, place: &BlurPlace, apron: Option<&FOut>) -> (FOut, LastPass) {
    let taps = kernel_taps(sigma, support);
    let atlas = |e: i32| (e as u32).next_power_of_two().max(512) as f32;
    let (x, y, o) = (rects.x, rects.y.expect("a halving graph has a Y pass"), rects.out);
    let texture = direct && rects.input.is_none();
    // The X pass's texture: its size, and the texel of the effect's origin in it. Another effect's output is
    // realized with its apron; a tap past that reads the apron's outermost texel.
    let (x_size, x_org, rz) = match (direct, rects.input) {
        (true, None) => ([src.w as f32, src.h as f32], [-src.ox, -src.oy], [0; 4]),
        (true, Some(rz)) => ([atlas(rz[2] - rz[0]), atlas(rz[3] - rz[1])], [place.input[0] - rz[0], place.input[1] - rz[1]], rz),
        (false, _) => {
            let pr = rects.pre.expect("a pre-downscale");
            ([atlas(src.w), atlas(src.h)], [place.pre[0] - pr[0], place.pre[1] - pr[1]], [0; 4])
        }
    };
    let x_texel = |ax: i32, ay: i32| {
        let (x, y) = (ax - x_org[0], ay - x_org[1]);
        if direct && !texture {
            let (x, y) = (x.clamp(rz[0], rz[2] - 1), y.clamp(rz[1], rz[3] - 1));
            return match apron {
                Some(a) => a.at(x, y, Edge::Clamp),
                None => src.at(x, y, Edge::Mirror),
            };
        }
        src.at(x, y, if direct { Edge::Mirror } else { Edge::Clamp })
    };
    let x_geom = Geom { w: x[2] - x[0], h: x[3] - x[1], at: place.x, scene: x.map(|v| v as f32), tex: edge_tex(x, x_org, [1.0 / x_size[0], 1.0 / x_size[1]]), size: x_size };
    // The texel of the effect's origin in the X pass's atlas.
    let xo = [place.x[0] - x[0], place.x[1] - x[1]];
    let y_size = [atlas(x_geom.w), atlas(x_geom.h)];
    let y_geom = Geom { w: y[2] - y[0], h: y[3] - y[1], at: place.y, scene: y.map(|v| v as f32), tex: edge_tex(y, xo, [1.0 / y_size[0], 1.0 / y_size[1]]), size: y_size };
    // Our buffers: the passes' quads, with one more column and row before them (two X rows) when direct.
    let e = direct as i32;
    let mut xp = FOut::new(x_geom.w + e, x_geom.h + 2 * e, x[0] - e, x[1] - 2 * e);
    let x_pass = Pass::new(&x_geom, 0, 2.0, &taps);
    for jy in 0..xp.h {
        for jx in 0..xp.w {
            let v = x_pass.px(jx + xp.ox - x[0], jy + xp.oy - x[1], &taps, texture, x_texel);
            xp.px[(jy * xp.w + jx) as usize] = quantize_to(p, v);
        }
    }
    let mut yp = FOut::new(y_geom.w + e, y_geom.h + e, y[0] - e, y[1] - e);
    // A texel before the atlas's first column or row clamps to it, as the texture's address mode does. Our buffers keep
    // one more column and row before a pass for the atlas texels a pass placed further in reads there; at the atlas
    // edge there are none (fblur-tile-thin-w2100: an odd tile's Y pass at atlas x 0 leaks 1/256 of its left
    // neighbour, which is its own first column).
    let y_texel = |ax: i32, ay: i32| xp.at(ax.max(0) - xo[0], ay.max(0) - xo[1], Edge::Clamp);
    let y_pass = Pass::new(&y_geom, 1, 2.0, &taps);
    for jy in 0..yp.h {
        for jx in 0..yp.w {
            let v = y_pass.px(jx + yp.ox - y[0], jy + yp.oy - y[1], &taps, false, y_texel);
            yp.px[(jy * yp.w + jx) as usize] = quantize_to(p, v);
        }
    }
    // The upscale draws the output quad at its place. Its vertex shader takes each corner `x` to `(m·x + c)/z`
    // (`m = 1/sh`, `z = m²`, `c = z/4`: the Y pass's texel `x·sh + ¼`) and emits `(that − x)/size + t` from the
    // corner's texcoord `t` (the Y pass's texel `x`); WARP interpolates it over the quad and the pixel shader
    // samples the Y pass linearly there.
    let yo = [place.y[0] - y[0], place.y[1] - y[1]];
    let up_size = [atlas(y_geom.w), atlas(y_geom.h)];
    let m = 1.0 / sh;
    let (z, c) = (m * m, m * m * 0.25);
    let corner = |e: i32, k: usize| {
        let x = e as f32;
        ((m * x + c) / z - x) * (1.0 / up_size[k]) + (e + yo[k]) as f32 * (1.0 / up_size[k])
    };
    let (ow, oh) = (o[2] - o[0], o[3] - o[1]);
    let up = PassQuad::new(ow, oh, place.out, o.map(|v| v as f32), [corner(o[0], 0), corner(o[1], 1), corner(o[2], 0), corner(o[3], 1)], 0, 1.0);
    let mut out = FOut::new(ow, oh, o[0], o[1]);
    for jy in 0..oh {
        for jx in 0..ow {
            let a = up.at(jx, jy);
            out.px[(jy * ow + jx) as usize] = bilinear_at(a[2], a[3], up_size, false, |ax, ay| yp.at(ax.max(0) - yo[0], ay.max(0) - yo[1], Edge::Clamp));
        }
    }
    (out, LastPass { kind: LastKind::Up(sh), src: yp, org: yo, size: up_size, bounds })
}

/// What a HARD blur's last pass reads, to run that pass again inside another blur's border: Direct2D folds it
/// there when the other blur reads this one, drawing it over the border's quads at the border's place.
pub struct LastPass {
    kind: LastKind,
    /// The intermediate the pass samples, the atlas texel of the effect's origin, and the atlas size.
    src: FOut,
    org: [i32; 2],
    size: [f32; 2],
    /// The blur's output bounds (its input's), in the effect's space.
    bounds: [i32; 4],
}

enum LastKind {
    /// The full-resolution Y pass, with its taps.
    Y(Vec<(f32, f32)>),
    /// The upscale from the Y pass, by `1/sh`.
    Up(f32),
    /// A drawn input's own intermediate, sampled linearly, its output clamped to [0, 1] when `clamp` (a
    /// ConvolveMatrix's clamp links into the border).
    Copy { clamp: bool },
    /// A ConvolveMatrix over its padded copy: each tap a linear sample at its offset in texels from the quad's
    /// texcoord, summed as [`sum_taps`] does.
    Convolve { taps: Vec<(i32, i32, f32)>, clamp: bool },
}

impl LastPass {
    /// What a blur's border draws from a drawn input: `src` (its intermediate in the effect's space) sampled
    /// directly, or the ConvolveMatrix `taps` over it, the texture at atlas place `at` sized `size`, and the
    /// blur's input over `bounds`.
    pub fn drawn(src: FOut, at: [i32; 2], bounds: [i32; 4], taps: Option<Vec<(i32, i32, f32)>>, clamp: bool) -> Self {
        let atlas = |e: i32| (e as u32).next_power_of_two().max(512) as f32;
        let (org, size) = ([at[0] - src.ox, at[1] - src.oy], [atlas(src.w), atlas(src.h)]);
        let kind = match taps {
            Some(taps) => LastKind::Convolve { taps, clamp },
            None => LastKind::Copy { clamp },
        };
        LastPass { kind, src, org, size, bounds }
    }
}

/// `last`'s pass drawn as a blur's border over `rect` (in the effect's space), whose top-left sits at atlas
/// pixel `at`: Direct2D's quads for that border, each corner carrying the output position it shows and its
/// texcoord. A mirror border cuts the rect at every multiple of the output size and reflects there; a clamp border
/// (`clamp`) keeps the output itself and extends its edge texels' centres over one-pixel strips.
pub fn border_fold(last: &LastPass, rect: [i32; 4], at: [i32; 2], clamp: bool, p: u32) -> FOut {
    // Per axis: the segments `[c0, c1)` and the output position shown at each end, the output spanning `[lo, hi)`.
    let segs = |c0: i32, c1: i32, k: usize| -> Vec<(i32, i32, f32, f32)> {
        let (lo, hi) = (last.bounds[k], last.bounds[k + 2]);
        if clamp {
            let mut v = Vec::new();
            if c0 < lo {
                v.push((c0, lo, lo as f32 + 0.5, lo as f32 + 0.5));
            }
            v.push((c0.max(lo), c1.min(hi), c0.max(lo) as f32, c1.min(hi) as f32));
            if c1 > hi {
                v.push((hi, c1, hi as f32 - 0.5, hi as f32 - 0.5));
            }
            return v;
        }
        let n = hi - lo;
        let mirror = |c: i32| {
            let m = (c - lo).rem_euclid(2 * n);
            (lo + if m > n { 2 * n - m } else { m }) as f32
        };
        let mut cuts = vec![c0];
        cuts.extend(((c0 - lo).div_euclid(n) + 1..).map(|m| lo + m * n).take_while(|&v| v < c1));
        cuts.push(c1);
        cuts.windows(2).map(|c| (c[0], c[1], mirror(c[0]), mirror(c[1]))).collect()
    };
    let [w, h] = [rect[2] - rect[0], rect[3] - rect[1]];
    let mut out = FOut::new(w, h, rect[0], rect[1]);
    let inv = last.size.map(|v| 1.0 / v);
    let texel = |ax: i32, ay: i32| last.src.at(ax.max(0) - last.org[0], ay.max(0) - last.org[1], Edge::Clamp);
    for &(y0, y1, sy0, sy1) in &segs(rect[1], rect[3], 1) {
        for &(x0, x1, sx0, sx1) in &segs(rect[0], rect[2], 0) {
            let (qw, qh) = (x1 - x0, y1 - y0);
            let qat = [at[0] + x0 - rect[0], at[1] + y0 - rect[1]];
            let t = |v: f32, k: usize| (v + last.org[k] as f32) * inv[k];
            let scene = [sx0, sy0, sx1, sy1];
            match &last.kind {
                LastKind::Y(taps) => {
                    let g = Geom { w: qw, h: qh, at: qat, scene, tex: [t(sx0, 0), t(sy0, 1), t(sx1, 0), t(sy1, 1)], size: last.size };
                    let pass = Pass::new(&g, 1, 1.0, taps);
                    for cy in 0..qh {
                        for cx in 0..qw {
                            let v = pass.px(cx, cy, taps, false, texel);
                            out.px[((y0 + cy - rect[1]) * w + x0 + cx - rect[0]) as usize] = quantize_to(p, v);
                        }
                    }
                }
                LastKind::Copy { clamp } | LastKind::Convolve { clamp, .. } => {
                    let q = PassQuad::new(qw, qh, qat, scene, [t(sx0, 0), t(sy0, 1), t(sx1, 0), t(sy1, 1)], 0, 1.0);
                    let sample = |u: f32, v: f32| bilinear_at(u, v, last.size, false, texel);
                    for cy in 0..qh {
                        for cx in 0..qw {
                            let a = q.at(cx, cy);
                            let v = match &last.kind {
                                LastKind::Convolve { taps, .. } => sum_taps(taps, false, |dx, dy| sample(dx as f32 * inv[0] + a[2], dy as f32 * inv[1] + a[3])),
                                _ => sample(a[2], a[3]),
                            };
                            let v = if *clamp { v.map(|c| c.clamp(0.0, 1.0)) } else { v };
                            out.px[((y0 + cy - rect[1]) * w + x0 + cx - rect[0]) as usize] = quantize_to(p, v);
                        }
                    }
                }
                LastKind::Up(sh) => {
                    let m = 1.0 / sh;
                    let (z, c) = (m * m, m * m * 0.25);
                    let corner = |x: f32, k: usize| ((m * x + c) / z - x) * inv[k] + t(x, k);
                    let q = PassQuad::new(qw, qh, qat, scene, [corner(sx0, 0), corner(sy0, 1), corner(sx1, 0), corner(sy1, 1)], 0, 1.0);
                    for cy in 0..qh {
                        for cx in 0..qw {
                            let a = q.at(cx, cy);
                            let v = bilinear_at(a[2], a[3], last.size, false, texel);
                            out.px[((y0 + cy - rect[1]) * w + x0 + cx - rect[0]) as usize] = quantize_to(p, v);
                        }
                    }
                }
            }
        }
    }
    out
}

/// [`tap_sum`] about texel-space position `base` (texel index units: texel `i` spans `[i, i+1)`'s
/// centre at `i`) instead of an output texel: each tap reads the bilinear of `sample` at `base ± off`.
fn tap_sum_at(taps: &[(f32, f32)], base: f32, round: bool, sample: impl Fn(i32) -> [f32; 4]) -> [f32; 4] {
    if taps.len() > 8 {
        return tap_sum_long(taps, base, round, sample);
    }
    let mut order: Vec<(f32, f32)> = taps.iter().copied().chain(taps.iter().map(|&(off, wt)| (-off, wt))).collect();
    order.swap(0, 1);
    let mut acc = [0.0f32; 4];
    for (d, wt) in order {
        let (i0, f) = q8_texel(base + d, round);
        let (a, b) = (sample(i0), sample(i0 + 1));
        (0..4).for_each(|c| acc[c] += wt * (a[c] + (b[c] - a[c]) * f));
    }
    acc
}

/// One blur pass's quad: `w × h` pixels of its intermediate, the scene position and texcoord at its edges
/// (`[left, top, right, bottom]`), and the size of the texture it samples.
struct Geom {
    w: i32,
    h: i32,
    /// The atlas pixel of the quad's top-left corner.
    at: [i32; 2],
    scene: [f32; 4],
    tex: [f32; 4],
    size: [f32; 2],
}

/// A blur pass over a [`Geom`], with the shader Direct2D picks for the kernel's size.
enum Pass {
    Long(PassQuad, [f32; 2]),
    Short(TapQuad, [f32; 2]),
}

impl Pass {
    /// `scale` is the blurred axis's subsampling (2 when the pass halves it, else 1).
    fn new(g: &Geom, axis: usize, scale: f32, taps: &[(f32, f32)]) -> Self {
        if taps.len() > 8 {
            Pass::Long(PassQuad::new(g.w, g.h, g.at, g.scene, g.tex, axis, scale), g.size)
        } else {
            Pass::Short(TapQuad::new(g, axis, scale, taps), g.size)
        }
    }

    fn px(&self, cx: i32, cy: i32, taps: &[(f32, f32)], round: bool, texel: impl Fn(i32, i32) -> [f32; 4]) -> [f32; 4] {
        match self {
            Pass::Long(q, size) => long_pass_px(q, cx, cy, taps, *size, round, texel),
            Pass::Short(q, size) => {
                // The fixed shader's order: `+off₁, +off₀, +off₂ … +offₙ₋₁, −off₀ … −offₙ₋₁`, each `w·sample`
                // added on its own.
                let n = taps.len();
                let mut order: Vec<usize> = (0..n).map(|i| 2 * i).chain((0..n).map(|i| 2 * i + 1)).collect();
                order.swap(0, 1);
                let mut acc = [0.0f32; 4];
                for t in order {
                    let [u, v] = q.at(t, cx, cy);
                    let s = bilinear_at(u, v, *size, round, &texel);
                    (0..4).for_each(|c| acc[c] += taps[t / 2].1 * s[c]);
                }
                acc
            }
        }
    }
}

/// A fixed-kernel blur pass's quad: Direct2D's vertex shader emits one texcoord per tap,
/// `((p ± off) − a)/size + t` with `p = (a − ½)(scale − 1) + a` from the corner's scene position `a` and
/// texcoord `t`, and WARP interpolates each over the quad's two triangles.
struct TapQuad {
    w: f32,
    h: f32,
    at: [i32; 2],
    /// Per tap (`+off₀, −off₀, +off₁, …`), per triangle (LT, RB, RT) / (LB, LT, RB): the `u` and `v` planes.
    planes: Vec<[[[f32; 3]; 2]; 2]>,
}

impl TapQuad {
    fn new(g: &Geom, axis: usize, scale: f32, taps: &[(f32, f32)]) -> Self {
        use crate::warp::plane::setup_triangle_attr;
        let inv = g.size.map(|s| 1.0 / s);
        let (w, h) = (g.w, g.h);
        let corners = [[(0, 0), (w, h), (w, 0)], [(0, h), (0, 0), (w, h)]];
        let [l, t] = g.at;
        let attr = |(x, y): (i32, i32), off: f32, k: usize| -> f32 {
            let pick = |e: [f32; 4]| if k == 0 { if x == 0 { e[0] } else { e[2] } } else if y == 0 { e[1] } else { e[3] };
            let (a, t) = (pick(g.scene), pick(g.tex));
            let p = (a - 0.5) * (if k == axis { scale } else { 1.0 } - 1.0) + a;
            ((p + if k == axis { off } else { 0.0 }) - a) * inv[k] + t
        };
        let planes = taps
            .iter()
            .flat_map(|&(off, _)| [off, -off])
            .map(|off| {
                core::array::from_fn(|tri| {
                    let c = corners[tri];
                    let (sx, sy) = (c.map(|(x, _)| (l + x) as i64 * 256), c.map(|(_, y)| (t + y) as i64 * 256));
                    core::array::from_fn(|k| setup_triangle_attr(sx, sy, c.map(|p| attr(p, off, k))))
                })
            })
            .collect();
        TapQuad { w: w as f32, h: h as f32, at: g.at, planes }
    }

    /// Tap `t`'s texcoord at the centre of quad pixel `(cx, cy)`.
    fn at(&self, t: usize, cx: i32, cy: i32) -> [f32; 2] {
        let p = [cx as f32 + 0.5, cy as f32 + 0.5];
        let k = usize::from(p[1] * self.w > p[0] * self.h);
        let q = [(self.at[0] + cx) as f32 + 0.5, (self.at[1] + cy) as f32 + 0.5];
        self.planes[t][k].map(|a| a[0] * q[0] + (a[1] * q[1] + a[2]))
    }
}

/// WARP's float bilinear of `texel` (a texture of `size` texels) at normalized `(u, v)`: the Q8 address
/// (rounded or floored), Y lerped before X.
fn bilinear_at(u: f32, v: f32, size: [f32; 2], round: bool, texel: impl Fn(i32, i32) -> [f32; 4]) -> [f32; 4] {
    let address = |tc: f32, k: usize| {
        let q = (tc * size[k] - 0.5) * 256.0;
        let q = if round { q.round_ties_even() } else { q.floor() } as i32;
        (q >> 8, (q & 0xff) as f32 / 256.0)
    };
    let ((x0, fx), (y0, fy)) = (address(u, 0), address(v, 1));
    let (c00, c10, c01, c11) = (texel(x0, y0), texel(x0 + 1, y0), texel(x0, y0 + 1), texel(x0 + 1, y0 + 1));
    core::array::from_fn(|c| {
        let left = c00[c] + (c01[c] - c00[c]) * fy;
        let right = c10[c] + (c11[c] - c10[c]) * fy;
        left + (right - left) * fx
    })
}

/// One blur pass's quad as Direct2D draws it, `w × h` pixels of its intermediate split along the LT→RB
/// diagonal: the scene position and the texcoord ride WARP's attribute planes over the two triangles.
struct PassQuad {
    w: f32,
    h: f32,
    at: [i32; 2],
    /// The blurred axis, and its subsampling (its scene position scales by it).
    axis: usize,
    scale: f32,
    /// Per triangle (LT, RB, RT) / (LB, LT, RB): the planes of scene x, scene y, texcoord u and v.
    planes: [[[f32; 3]; 4]; 2],
}

impl PassQuad {
    /// `at` is the atlas pixel of the quad's top-left corner; `scene` and `tex` give `[x at left, y at top,
    /// x at right, y at bottom]`.
    fn new(w: i32, h: i32, at: [i32; 2], scene: [f32; 4], tex: [f32; 4], axis: usize, scale: f32) -> Self {
        use crate::warp::plane::setup_triangle_attr;
        let corners = [[(0, 0), (w, h), (w, 0)], [(0, h), (0, 0), (w, h)]];
        let value = |(x, y): (i32, i32), k: usize| match k {
            0 => if x == 0 { scene[0] } else { scene[2] },
            1 => if y == 0 { scene[1] } else { scene[3] },
            2 => if x == 0 { tex[0] } else { tex[2] },
            _ => if y == 0 { tex[1] } else { tex[3] },
        };
        let planes = corners.map(|c| {
            let (sx, sy) = (c.map(|(x, _)| (at[0] + x) as i64 * 256), c.map(|(_, y)| (at[1] + y) as i64 * 256));
            core::array::from_fn(|k| setup_triangle_attr(sx, sy, c.map(|p| value(p, k))))
        });
        PassQuad { w: w as f32, h: h as f32, at, axis, scale, planes }
    }

    /// Scene x, scene y, u and v at the centre of quad pixel `(cx, cy)`, in WARP's evaluation order.
    fn at(&self, cx: i32, cy: i32) -> [f32; 4] {
        let p = [cx as f32 + 0.5, cy as f32 + 0.5];
        let k = usize::from(p[1] * self.w > p[0] * self.h);
        let q = [(self.at[0] + cx) as f32 + 0.5, (self.at[1] + cy) as f32 + 0.5];
        core::array::from_fn(|i| {
            let a = &self.planes[k][i];
            a[0] * q[0] + (a[1] * q[1] + a[2])
        })
    }
}

/// A pixel of a blur pass with more than 8 tap pairs, as Direct2D's looping blur shader computes it: from
/// the interpolated scene position `a` and texcoord `t`, the centre is `((a − ½)·scale − (a − ½))/size + t`
/// (`scale` 2 on the blurred `axis`), each tap `±off/size` from it. A tap samples `texel` (a texture of
/// `size` texels) bilinearly, Y before X, at its Q8 address (rounded for the effect's own input, else
/// floored), and the pairs accumulate as [`tap_sum_long`] does.
#[allow(clippy::too_many_arguments)]
fn long_pass_px(q: &PassQuad, cx: i32, cy: i32, taps: &[(f32, f32)], size: [f32; 2], round: bool, texel: impl Fn(i32, i32) -> [f32; 4]) -> [f32; 4] {
    let axis = q.axis;
    let a = q.at(cx, cy);
    let inv = size.map(|s| 1.0 / s);
    let base: [f32; 2] = core::array::from_fn(|k| {
        let r = a[k] - 0.5;
        let r = r * if k == axis { q.scale } else { 1.0 } - r;
        r * inv[k] + a[2 + k]
    });
    let sample = |off: f32| -> [f32; 4] {
        let tc: [f32; 2] = core::array::from_fn(|k| if k == axis { off } else { 0.0 } * inv[k] + base[k]);
        bilinear_at(tc[0], tc[1], size, round, &texel)
    };
    let pair = |i: usize| -> [f32; 4] {
        let (plus, minus) = (sample(taps[i].0), sample(-taps[i].0));
        core::array::from_fn(|c| plus[c] + minus[c])
    };
    let chain = |from: usize, to: usize, start: [f32; 4]| {
        (from..to).fold(start, |acc, i| {
            let p = pair(i);
            core::array::from_fn(|c| p[c] * taps[i].1 + acc[c])
        })
    };
    let n = taps.len();
    let (mut sum, mut i0) = ([0.0f32; 4], 0);
    while n - i0 > 8 {
        let (p0, p1) = (pair(i0), pair(i0 + 1));
        let head: [f32; 4] = core::array::from_fn(|c| p0[c] * taps[i0].1 + p1[c] * taps[i0 + 1].1);
        let block = chain(i0 + 2, i0 + 8, head);
        sum = core::array::from_fn(|c| sum[c] + block[c]);
        i0 += 8;
    }
    let tail = chain(i0, n, [0.0; 4]);
    core::array::from_fn(|c| sum[c] + tail[c])
}

/// A sample position (texel index units, texel `i` centred at `i`) as WARP's sampler addresses it: the texel
/// and its 8-bit fraction. A blur pass reading the effect's own input texture (mirror-addressed under HARD)
/// rounds the Q8 address to nearest; one reading an intermediate floors it.
fn q8_texel(u: f32, round: bool) -> (i32, f32) {
    let q = u * 256.0;
    let q = if round { q.round_ties_even() } else { q.floor() } as i32;
    (q >> 8, (q & 0xff) as f32 / 256.0)
}

/// [`tap_sum_at`] for more than 8 tap pairs, where Direct2D's blur shader loops over the kernel: each pair's
/// two bilinear samples are added first (`+off`, then `−off`), then weighted. While more than 8 pairs remain,
/// a block of 8 is chained as `p₀·w₀ + p₁·w₁`, then `+ pᵢ·wᵢ` for the rest, and the block's total is added
/// to the running sum; the last pairs chain the same way from zero, and the output is the running sum plus
/// that tail. Every multiply and add rounds on its own.
fn tap_sum_long(taps: &[(f32, f32)], base: f32, round: bool, sample: impl Fn(i32) -> [f32; 4]) -> [f32; 4] {
    let bilinear = |d: f32| -> [f32; 4] {
        let (i0, f) = q8_texel(base + d, round);
        let (a, b) = (sample(i0), sample(i0 + 1));
        core::array::from_fn(|c| a[c] + (b[c] - a[c]) * f)
    };
    let pair = |i: usize| -> [f32; 4] {
        let (plus, minus) = (bilinear(taps[i].0), bilinear(-taps[i].0));
        core::array::from_fn(|c| plus[c] + minus[c])
    };
    let chain = |from: usize, to: usize, start: [f32; 4]| {
        (from..to).fold(start, |acc, i| {
            let p = pair(i);
            core::array::from_fn(|c| p[c] * taps[i].1 + acc[c])
        })
    };
    let n = taps.len();
    let (mut sum, mut i0) = ([0.0f32; 4], 0);
    while n - i0 > 8 {
        let (p0, p1) = (pair(i0), pair(i0 + 1));
        let head: [f32; 4] = core::array::from_fn(|c| p0[c] * taps[i0].1 + p1[c] * taps[i0 + 1].1);
        let block = chain(i0 + 2, i0 + 8, head);
        sum = core::array::from_fn(|c| sum[c] + block[c]);
        i0 += 8;
    }
    let tail = chain(i0, n, [0.0; 4]);
    core::array::from_fn(|c| sum[c] + tail[c])
}

/// A separable Gaussian pass in native's blur-PS order. The VS emits the texcoords `+off₀ … +offₙ₋₁`
/// then `−off₀ … −offₙ₋₁`, and the PS (DDI-captured for 1, 2 and 3 pairs, Direct2D 10.0.26100.8875) sums
/// them in that order with the first two swapped, each an unfused `mul`/`mad` of a two-texel bilinear
/// (`a + (b−a)·f`) by its tap weight; fbitmap-10's X pass, Y pass and SourceOver then match native
/// bit for bit. `sample(d)` fetches the texel `d` away from the output centre along the blur axis.
/// The bilinear fraction is truncated to 8 bits, as WARP's float-texture filter takes it (the f32
/// split was ~4e-4 off native on a Float32 target, fbitmap-10; [`bilinear`] does the same).
fn tap_sum(taps: &[(f32, f32)], sample: impl Fn(i32) -> [f32; 4]) -> [f32; 4] {
    let mut order: Vec<(f32, f32)> = taps.iter().copied().chain(taps.iter().map(|&(off, wt)| (-off, wt))).collect();
    order.swap(0, 1);
    let mut acc = [0.0f32; 4];
    for (d, wt) in order {
        let i0 = d.floor();
        let f = ((d - i0) * 256.0).floor() / 256.0;
        let (a, b) = (sample(i0 as i32), sample(i0 as i32 + 1));
        (0..4).for_each(|c| acc[c] += wt * (a[c] + (b[c] - a[c]) * f));
    }
    acc
}

/// Shadow: the premultiplied silhouette `premul(color)·α` (stored at the context precision), then a
/// SOFT Gaussian blur of it.
pub fn shadow(input: &Bitmap, sigma: f32, color: [f32; 4], mode: u32, p: u32) -> FOut {
    let pc = [color[0] * color[3], color[1] * color[3], color[2] * color[3], color[3]];
    let src = FOut::of(input);
    let silhouette: Vec<u8> = src
        .px
        .iter()
        .flat_map(|c| quantize_to(p, pc.map(|v| v * c[3])).into_iter().flat_map(f32::to_le_bytes))
        .collect();
    let bm = Bitmap {
        width: input.width,
        height: input.height,
        premul_bgra: silhouette,
        format: crate::abi::DXGI_FORMAT_R32G32B32A32_FLOAT,
        alpha: crate::abi::D2D1_ALPHA_MODE_PREMULTIPLIED,
        dpi: (96.0, 96.0),
    };
    gaussian_blur(&bm, sigma, mode, false, false, p, &BlurPlace::default())
}

/// DirectionalBlur: a 1-D Gaussian along `angle` (deg, CCW from +X), bilinear 1-px steps; SOFT grows
/// by the projected radius (as [`super::filters::directional_blur`]).
pub fn directional_blur(input: &Bitmap, sigma: f32, angle_deg: f32, hard: bool) -> FOut {
    if sigma <= 0.0 {
        return FOut::of(input);
    }
    let (wv, r) = gaussian_weights(sigma);
    let th = angle_deg * std::f32::consts::PI / 180.0;
    let (dx, dy) = (crate::crt::system_crt::cosf(th), -crate::crt::system_crt::sinf(th));
    let grow = if hard { 0 } else { ((r as f32) * dx.abs().max(dy.abs())).ceil() as i32 };
    let mut out = FOut::new(input.width as i32 + 2 * grow, input.height as i32 + 2 * grow, -grow, -grow);
    for yi in 0..out.h {
        for xi in 0..out.w {
            let (cx, cy) = ((out.ox + xi) as f32 + 0.5, (out.oy + yi) as f32 + 0.5);
            let mut acc = [0f32; 4];
            for (k, &wk) in wv.iter().enumerate() {
                let t = (k as i32 - r) as f32;
                let s = bilinear(input, cx + t * dx, cy + t * dy);
                (0..4).for_each(|c| acc[c] += wk * s[c]);
            }
            out.px[(yi * out.w + xi) as usize] = acc;
        }
    }
    out
}

/// ConvolveMatrix (as [`super::filters::convolve`], unclamped unless `clamp_output`): the flipped
/// kernel over premultiplied (or, preserving alpha, straight) samples, `/divisor + bias`.
#[allow(clippy::too_many_arguments)]
pub fn convolve(
    input: &Bitmap,
    ksx: u32,
    ksy: u32,
    kernel: &[f32],
    divisor: f32,
    bias: f32,
    offset: [f32; 2],
    preserve_alpha: bool,
    clamp_output: bool,
    hard: bool,
) -> FOut {
    let (ksx, ksy) = (ksx.max(1) as i32, ksy.max(1) as i32);
    let n = (ksx * ksy) as usize;
    let center = (ksy / 2 * ksx + ksx / 2) as usize;
    let is_identity = ksx % 2 == 1
        && ksy % 2 == 1
        && kernel.len() >= n
        && kernel[..n].iter().enumerate().all(|(i, &k)| k == if i == center { 1.0 } else { 0.0 });
    if is_identity && !preserve_alpha && offset == [0.0, 0.0] {
        return FOut::of(input);
    }
    let bx = -((ksx as f32 - 1.0) * 0.5) + offset[0];
    let by = -((ksy as f32 - 1.0) * 0.5) + offset[1];
    let grow_l = (-bx).ceil().max(0.0) as i32;
    let grow_r = (bx + (ksx - 1) as f32).ceil().max(0.0) as i32;
    let grow_t = (-by).ceil().max(0.0) as i32;
    let grow_b = (by + (ksy - 1) as f32).ceil().max(0.0) as i32;
    let (iw, ih) = (input.width as i32, input.height as i32);
    let mut out = if hard {
        FOut::new(iw, ih, 0, 0)
    } else {
        FOut::new(iw + grow_l + grow_r, ih + grow_t + grow_b, -grow_l, -grow_t)
    };
    let native = native_taps(ksx, ksy, kernel, divisor, bias, offset, clamp_output);
    for yi in 0..out.h {
        for xi in 0..out.w {
            let (px, py) = (out.ox + xi, out.oy + yi);
            let sample = |sx: i32, sy: i32| {
                let [r, g, b, a] = texel(input, sx, sy, hard);
                if preserve_alpha && a > 0.0 { [r / a, g / a, b / a, 0.0] } else { [r, g, b, a] }
            };
            let mut acc = [0f32; 4];
            if let Some(taps) = &native {
                acc = sum_taps(taps, ksx == 3 && ksy == 3, |dx, dy| sample(px + dx, py + dy));
            } else {
                for row in 0..ksy {
                    for col in 0..ksx {
                        let k = kernel[(ksx * (ksy - 1 - row) + (ksx - 1 - col)) as usize];
                        let t = sample(px + (bx + col as f32).round() as i32, py + (by + row as f32).round() as i32);
                        (0..4).for_each(|c| acc[c] += t[c] * k);
                    }
                }
            }
            // The native taps carry the divisor already.
            let div = if native.is_some() { acc.map(|v| v + bias) } else { acc.map(|v| v / divisor + bias) };
            let clamp = |v: f32| if clamp_output { v.clamp(0.0, 1.0) } else { v };
            out.px[(yi * out.w + xi) as usize] = if preserve_alpha {
                let ca = texel(input, px, py, hard)[3];
                [clamp(div[0]) * ca, clamp(div[1]) * ca, clamp(div[2]) * ca, ca]
            } else {
                div.map(clamp)
            };
        }
    }
    out
}

/// Direct2D's single-pass ConvolveMatrix tap list `(dx, dy, weight)`: the flipped kernel's non-zero taps listed
/// column-major (dx outer, dy inner), each weight divided by the divisor, positive weights from the front and
/// negative ones from the back (so the negatives come out reversed). The pixel shader sums them as [`sum_taps`] does.
/// One pass holds up to 256 taps.
/// `None` (or fewer than two taps) keeps the row-major form.
/// ponytail: bias and a kernel offset draw as further passes natively (taps split, bias as a constant pass),
/// and an even kernel size centres between texels; those keep the row-major form.
pub(crate) fn native_taps(ksx: i32, ksy: i32, kernel: &[f32], divisor: f32, bias: f32, offset: [f32; 2], clamp: bool) -> Option<Vec<(i32, i32, f32)>> {
    let n = (ksx * ksy) as usize;
    let _ = clamp; // applied after the sum, as the native post transform does
    if ksx % 2 == 0 || ksy % 2 == 0 || kernel.len() < n || divisor == 0.0 || bias != 0.0 || offset != [0.0, 0.0] {
        return None;
    }
    let (mut pos, mut neg) = (Vec::new(), Vec::new());
    let (hx, hy) = (ksx / 2, ksy / 2);
    for x in 0..ksx {
        for y in 0..ksy {
            let k = kernel[((ksy - 1 - y) * ksx + (ksx - 1 - x)) as usize];
            if k == 0.0 {
                continue;
            }
            let w = k / divisor;
            if w < 0.0 { neg.push((x - hx, y - hy, w)) } else { pos.push((x - hx, y - hy, w)) }
        }
    }
    pos.extend(neg.into_iter().rev());
    (pos.len() >= 2 && pos.len() <= 256).then_some(pos)
}

/// Sum the taps as the native pixel shaders do, with unfused `mul`/`mad`s. The 3×3 shader sums all its taps as one run:
/// the second tap's product, then the first's, then the rest. The general kernel shader sums runs of 16 taps that
/// way while more than 16 remain, adding each run's sum to a total that starts at zero, then adds the remaining taps
/// in order onto a zero.
pub(super) fn sum_taps(taps: &[(i32, i32, f32)], three: bool, at: impl Fn(i32, i32) -> [f32; 4]) -> [f32; 4] {
    let mad = |acc: [f32; 4], (dx, dy, w): (i32, i32, f32)| {
        let s = at(dx, dy);
        [0, 1, 2, 3].map(|c| s[c] * w + acc[c])
    };
    let run = |t: &[(i32, i32, f32)]| {
        let (dx, dy, w) = t[1];
        let first = at(dx, dy).map(|v| v * w);
        t[2..].iter().fold(mad(first, t[0]), |acc, &tap| mad(acc, tap))
    };
    if three {
        return run(taps);
    }
    let (mut total, mut start) = ([0f32; 4], 0);
    while taps.len() - start > 16 {
        let r = run(&taps[start..start + 16]);
        total = [0, 1, 2, 3].map(|c| total[c] + r[c]);
        start += 16;
    }
    let rest = taps[start..].iter().fold([0f32; 4], |acc, &tap| mad(acc, tap));
    [0, 1, 2, 3].map(|c| total[c] + rest[c])
}

/// `src` over `rect` (the effect's space), transparent where it has nothing: an intermediate as drawn.
pub fn regrid(src: &FOut, rect: [i32; 4]) -> FOut {
    let mut out = FOut::new(rect[2] - rect[0], rect[3] - rect[1], rect[0], rect[1]);
    for y in 0..out.h {
        for x in 0..out.w {
            let (sx, sy) = (rect[0] + x - src.ox, rect[1] + y - src.oy);
            if (0..src.w).contains(&sx) && (0..src.h).contains(&sy) {
                out.px[(y * out.w + x) as usize] = src.px[(sy * src.w + sx) as usize];
            }
        }
    }
    out
}

/// Morphology: erode (min) / dilate (max) per premultiplied channel; grows by ⌊w/2⌋,⌊h/2⌋.
pub fn morphology(input: &Bitmap, dilate: bool, width: u32, height: u32) -> FOut {
    let (wpx, hpx) = (width.clamp(1, 100) as i32, height.clamp(1, 100) as i32);
    let (rx, ry) = (wpx / 2, hpx / 2);
    let mut out = FOut::new(input.width as i32 + 2 * rx, input.height as i32 + 2 * ry, -rx, -ry);
    for oy in 0..out.h {
        for ox in 0..out.w {
            let (cx, cy) = (ox - rx, oy - ry);
            let mut acc = if dilate { [f32::NEG_INFINITY; 4] } else { [f32::INFINITY; 4] };
            for ky in 0..hpx {
                for kx in 0..wpx {
                    let s = texel(input, cx - rx + kx, cy - ry + ky, false);
                    (0..4).for_each(|c| acc[c] = if dilate { acc[c].max(s[c]) } else { acc[c].min(s[c]) });
                }
            }
            out.px[(oy * out.w + ox) as usize] = acc;
        }
    }
    out
}

/// Posterize: per-channel DiscreteTransfer on straight colour, alpha kept.
pub fn posterize(input: &Bitmap, red: u32, green: u32, blue: u32) -> FOut {
    let t = [discrete_table(red), discrete_table(green), discrete_table(blue)];
    let mut out = FOut::of(input);
    for y in 0..out.h {
        for x in 0..out.w {
            let [r, g, b] = straight_rgb(input, x, y, false);
            let a = texel(input, x, y, true)[3].clamp(0.0, 1.0);
            let s = [discrete_lookup(&t[0], r), discrete_lookup(&t[1], g), discrete_lookup(&t[2], b)];
            out.px[(y * out.w + x) as usize] = [s[0] * a, s[1] * a, s[2] * a, a];
        }
    }
    out
}

/// EdgeDetection: Sobel/Prewitt magnitude on straight colour, `clamp(strength·(gx²+gy²), 0, 1)`.
pub fn edge_detection(input: &Bitmap, strength: f32, prewitt: bool, overlay: bool, alpha_straight: bool) -> FOut {
    let mw = if prewitt { 1.0 } else { 2.0 };
    let mut out = FOut::of(input);
    for y in 0..out.h {
        for x in 0..out.w {
            let c = straight_rgb(input, x, y, alpha_straight);
            let s = |dx, dy| straight_rgb(input, x + dx, y + dy, alpha_straight);
            let (tl, tm, tr) = (s(-1, -1), s(0, -1), s(1, -1));
            let (ml, mr) = (s(-1, 0), s(1, 0));
            let (bl, bm, br) = (s(-1, 1), s(0, 1), s(1, 1));
            let edge: [f32; 3] = core::array::from_fn(|k| {
                let gx = (tr[k] + br[k] - tl[k] - bl[k]) + mw * (mr[k] - ml[k]);
                let gy = (bl[k] + br[k] - tl[k] - tr[k]) + mw * (bm[k] - tm[k]);
                (strength * (gx * gx + gy * gy)).clamp(0.0, 1.0)
            });
            let a = texel(input, x, y, false)[3];
            out.px[(y * out.w + x) as usize] = if overlay {
                [(c[0] - edge[0]) * a, (c[1] - edge[1]) * a, (c[2] - edge[2]) * a, a]
            } else {
                [edge[0], edge[1], edge[2], 1.0]
            };
        }
    }
    out
}

/// Emboss: a luminance height field lit by a distant light (elevation 50°, azimuth = direction).
pub fn emboss(input: &Bitmap, height: f32, direction_deg: f32) -> FOut {
    let dir = direction_deg * std::f32::consts::PI / 180.0;
    let elev = 0.87266463_f32;
    let (cz, sz) = (crate::crt::system_crt::cosf(elev), crate::crt::system_crt::sinf(elev));
    let light = [crate::crt::system_crt::cosf(dir) * cz, crate::crt::system_crt::sinf(dir) * cz, sz];
    let nscale = -height * 0.25;
    let mut out = FOut::of(input);
    for y in 0..out.h {
        for x in 0..out.w {
            let l = |dx, dy| luminance(input, x + dx, y + dy);
            let (tl, tm, tr) = (l(-1, -1), l(0, -1), l(1, -1));
            let (ml, mr) = (l(-1, 0), l(1, 0));
            let (bl, bm, br) = (l(-1, 1), l(0, 1), l(1, 1));
            let n = [nscale * ((tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl)), nscale * ((bl + 2.0 * bm + br) - (tl + 2.0 * tm + tr)), 1.0];
            let inv = 1.0 / (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            let gray = (n[0] * inv * light[0] + n[1] * inv * light[1] + n[2] * inv * light[2]).max(0.0);
            out.px[(y * out.w + x) as usize] = [gray, gray, gray, 1.0];
        }
    }
    out
}

/// Sharpen: unsharp mask on luminance added to straight RGB (clamped to [0,1]), alpha kept.
pub fn sharpen(input: &Bitmap, sharpness: f32, threshold: f32) -> FOut {
    let mut out = FOut::of(input);
    for y in 0..out.h {
        for x in 0..out.w {
            let center = straight_rgb(input, x, y, false);
            let center_l = LUMA[0] * center[0] + LUMA[1] * center[1] + LUMA[2] * center[2];
            let mut blur = 0.0;
            for dy in -2..=2i32 {
                for dx in -2..=2i32 {
                    let l = if dx == 0 && dy == 0 { center_l } else { luminance(input, x + dx, y + dy) };
                    blur += SHARP_G[dx.unsigned_abs() as usize] * SHARP_G[dy.unsigned_abs() as usize] * l;
                }
            }
            let d = center_l - blur;
            let factor = if threshold > 0.0 { (d.abs() / threshold).min(1.0) } else { 1.0 };
            let corr = sharpness * d * factor;
            let a = texel(input, x, y, false)[3];
            let rgb = [0, 1, 2].map(|k| (center[k] + corr).clamp(0.0, 1.0) * a);
            out.px[(y * out.w + x) as usize] = [rgb[0], rgb[1], rgb[2], a];
        }
    }
    out
}

/// DisplacementMap: input0 sampled bilinearly at `scale·(channel(dmap) − 0.5)` off each pixel; grows by
/// ⌈|scale|/2⌉ each side.
pub fn displacement_map(image: &Bitmap, dmap: &Bitmap, scale: f32, x_channel: u32, y_channel: u32) -> FOut {
    let r = (scale.abs() / 2.0).ceil() as i32;
    let mut out = FOut::new(image.width as i32 + 2 * r, image.height as i32 + 2 * r, -r, -r);
    for oy in 0..out.h {
        for ox in 0..out.w {
            let (sx, sy) = ((ox - r) as f32 + 0.5, (oy - r) as f32 + 0.5);
            let d = bitmap_texel(dmap, ox - r, oy - r);
            let dx = scale * (d[x_channel.min(3) as usize] - 0.5);
            let dy = scale * (d[y_channel.min(3) as usize] - 0.5);
            out.px[(oy * out.w + ox) as usize] = bilinear(image, sx + dx, sy + dy);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn float_bitmap(w: u32, h: u32, px: &[[f32; 4]]) -> Bitmap {
        Bitmap {
            width: w,
            height: h,
            premul_bgra: px.iter().flat_map(|c| c.iter().flat_map(|v| v.to_le_bytes())).collect(),
            format: crate::abi::DXGI_FORMAT_R32G32B32A32_FLOAT,
            alpha: crate::abi::D2D1_ALPHA_MODE_PREMULTIPLIED,
            dpi: (96.0, 96.0),
        }
    }

    /// The pair split reproduces the Gaussian the tap pairs encode: weights sum to 1, symmetric.
    #[test]
    fn texel_weights_sum_to_one_and_mirror() {
        for sigma in [0.6f32, 2.0, 7.5] {
            let (w, reach) = gaussian_texel_weights(sigma, 3.0);
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5, "σ {sigma}");
            for k in 0..reach as usize {
                assert!((w[k] - w[w.len() - 1 - k]).abs() < 1e-7);
            }
        }
    }

    /// At float precision a uniform HDR image blurs to itself (no clamp to 1), and a sub-LSB value
    /// survives (8bpc would round it away).
    #[test]
    fn float_blur_keeps_hdr_and_sub_lsb() {
        let px = vec![[2.0, 0.002, 0.5, 1.0]; 16 * 12];
        let out = gaussian_blur(&float_bitmap(16, 12, &px), 2.0, 2, true, true, 5, &BlurPlace::default());
        for c in &out.px {
            assert!((c[0] - 2.0).abs() < 1e-4 && (c[1] - 0.002).abs() < 1e-6, "{c:?}");
        }
    }

    /// An unclamped convolve keeps its negative side at float precision (the kernel is applied
    /// flipped: out = p(x−1,y−1) + p − p(x+1,y+1), negative just left of the step up).
    #[test]
    fn float_convolve_keeps_negative() {
        let px: Vec<[f32; 4]> = (0..8 * 8).map(|i| if i % 8 < 4 { [0.2, 0.2, 0.2, 1.0] } else { [0.6, 0.6, 0.6, 1.0] }).collect();
        let k = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let out = convolve(&float_bitmap(8, 8, &px), 3, 3, &k, 1.0, 0.0, [0.0, 0.0], false, false, true);
        assert!(out.px.iter().any(|c| c[0] < -0.1), "no negative output");
    }
}


