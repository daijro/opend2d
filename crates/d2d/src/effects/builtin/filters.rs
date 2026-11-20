//! Phase 2C filter-family built-in effects. Each effect's native passes (cross-checked against the
//! WARP shader math) are written here as a function over premultiplied
//! -BGRA8 input image(s) and the effect's D2D properties, returning the premultiplied-BGRA8 output
//! plus its logical pixel origin — negative when a SOFT border grows the bounds. Intermediate math
//! is f32; the output is the 8bpc UNORM premultiplied node buffer the graph stores between nodes.
//! 2B's effect graph calls these once it lands on main; until then they are tested directly against
//! the oracle (`tests/effects/filters`, and dispatched in `tests/conformance.rs`).

use crate::bitmap::Bitmap;

/// An effect's output: the 8bpc premultiplied-BGRA node buffer and its top-left in input-pixel
/// space. At 96 DPI logical pixels equal DIPs, so `(ox,oy)` is also the DIP offset for drawing.
pub struct Output {
    pub bitmap: Bitmap,
    pub ox: i32,
    pub oy: i32,
}

impl Output {
    fn new(width: u32, height: u32, ox: i32, oy: i32, premul_bgra: Vec<u8>) -> Self {
        Output {
            bitmap: Bitmap {
                width,
                height,
                premul_bgra,
                format: crate::abi::DXGI_FORMAT_B8G8R8A8_UNORM,
                alpha: crate::abi::D2D1_ALPHA_MODE_PREMULTIPLIED,
                dpi: (96.0, 96.0),
            },
            ox,
            oy,
        }
    }
}

// --- sampling / packing -------------------------------------------------------------------------

/// Premultiplied RGBA (f32, [0,1]) of texel `(x,y)`. HARD border clamps to the edge; SOFT returns
/// transparent black outside the input (the `ID2D1BorderTransform` the blurs/convolve prepend).
fn premul_at(bm: &Bitmap, x: i32, y: i32, hard: bool) -> [f32; 4] {
    let (w, h) = (bm.width as i32, bm.height as i32);
    let (sx, sy) = if hard {
        (x.clamp(0, w - 1), y.clamp(0, h - 1))
    } else if x < 0 || y < 0 || x >= w || y >= h {
        return [0.0; 4];
    } else {
        (x, y)
    };
    let i = (sy as usize * w as usize + sx as usize) * 4;
    let p = &bm.premul_bgra;
    [p[i + 2] as f32 / 255.0, p[i + 1] as f32 / 255.0, p[i] as f32 / 255.0, p[i + 3] as f32 / 255.0]
}

/// Round an f32 colour component in [0,1] to UNORM8 the way WARP stores a pixel-shader result to a
/// UNORM8 render target: `round_ties_even(saturate(v)·255)`.
fn unorm8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round_ties_even() as u8
}

/// Premultiplied RGBA (f32) of texel `(x,y)` the way WARP's effect sampler reads a UNORM8 input
/// (`byte·257/65536`); SOFT border returns transparent black outside. Used for the point-sampled
/// displacement channel and the linear-sampled image corners.
fn sample_premul_f32(bm: &Bitmap, x: i32, y: i32) -> [f32; 4] {
    use crate::effects::sample::unorm_to_f32;
    let (w, h) = (bm.width as i32, bm.height as i32);
    if x < 0 || y < 0 || x >= w || y >= h {
        return [0.0; 4];
    }
    let p = &bm.premul_bgra;
    let i = (y as usize * w as usize + x as usize) * 4;
    [unorm_to_f32(p[i + 2]), unorm_to_f32(p[i + 1]), unorm_to_f32(p[i]), unorm_to_f32(p[i + 3])]
}

/// Pack a premultiplied RGBA (f32) output pixel into BGRA8 node-buffer bytes.
fn pack_premul([r, g, b, a]: [f32; 4]) -> [u8; 4] {
    [unorm8(b), unorm8(g), unorm8(r), unorm8(a)]
}

/// Re-premultiply a straight RGBA result, then pack (the unpremultiply→process→premultiply effects).
fn pack_straight([r, g, b, a]: [f32; 4]) -> [u8; 4] {
    let a = a.clamp(0.0, 1.0);
    pack_premul([r * a, g * a, b * a, a])
}

// --- Posterize (wraps DiscreteTransfer) ---------------------------------------------------------

/// DiscreteTransfer table for `n` levels: `T[i] = i/(n-1)` (straight value in [0,1]).
pub(super) fn discrete_table(n: u32) -> Vec<f32> {
    let n = n.clamp(2, 16);
    (0..n).map(|i| i as f32 / (n - 1) as f32).collect()
}

/// `out = T[clamp(floor(c·n), 0, n-1)]` (DiscreteTransfer on a straight channel).
pub(super) fn discrete_lookup(table: &[f32], c: f32) -> f32 {
    let n = table.len() as i32;
    let idx = (c * n as f32).floor() as i32;
    table[idx.clamp(0, n - 1) as usize]
}

/// Posterize: per-channel DiscreteTransfer on straight colour, alpha unchanged (AlphaDisable=TRUE).
pub fn posterize(input: &Bitmap, red: u32, green: u32, blue: u32) -> Output {
    let t = [discrete_table(red), discrete_table(green), discrete_table(blue)];
    let (w, h) = (input.width, input.height);
    let mut out = vec![0u8; (w * h * 4) as usize];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let [r, g, b] = straight_rgb(input, x, y, false);
            let a = premul_at(input, x, y, true)[3];
            let s = [discrete_lookup(&t[0], r), discrete_lookup(&t[1], g), discrete_lookup(&t[2], b), a];
            let i = ((y as u32 * w + x as u32) * 4) as usize;
            out[i..i + 4].copy_from_slice(&pack_straight(s));
        }
    }
    Output::new(w, h, 0, 0, out)
}

// --- Morphology --------------------------------------------------------------------------------

/// Erode (min) / dilate (max) over a `width`×`height` window per channel of premultiplied colour.
/// Separable in native (X then Y) but min/max is associative, so one window pass is identical.
/// Output grows by ⌊w/2⌋,⌊h/2⌋ each side; samples outside the input are transparent black.
pub fn morphology(input: &Bitmap, dilate: bool, width: u32, height: u32) -> Output {
    let (wpx, hpx) = (width.clamp(1, 100) as i32, height.clamp(1, 100) as i32);
    let (rx, ry) = (wpx / 2, hpx / 2);
    let (iw, ih) = (input.width as i32, input.height as i32);
    let (ow, oh) = (iw + 2 * rx, ih + 2 * ry);
    let mut out = vec![0u8; (ow * oh * 4) as usize];
    for oy in 0..oh {
        for ox in 0..ow {
            let (cx, cy) = (ox - rx, oy - ry);
            let mut acc = if dilate { [0f32; 4] } else { [f32::INFINITY; 4] };
            for ky in 0..hpx {
                for kx in 0..wpx {
                    let s = premul_at(input, cx - rx + kx, cy - ry + ky, false);
                    for c in 0..4 {
                        acc[c] = if dilate { acc[c].max(s[c]) } else { acc[c].min(s[c]) };
                    }
                }
            }
            let i = ((oy * ow + ox) * 4) as usize;
            out[i..i + 4].copy_from_slice(&pack_premul(acc));
        }
    }
    Output::new(ow as u32, oh as u32, -rx, -ry, out)
}

// --- EdgeDetection -----------------------------------------------------------------------------

/// Straight RGB of texel `(x,y)` for the ±N neighbourhood effects (unpremultiply unless the colour
/// is already straight). These effects (output rect = input rect) clamp the input border — native's
/// `D2DSampleInputAtOffset` confines the read to the input rect — confirmed by the sharpen oracle.
fn straight_rgb(bm: &Bitmap, x: i32, y: i32, is_straight: bool) -> [f32; 3] {
    let [r, g, b, a] = premul_at(bm, x, y, true);
    if is_straight || a == 0.0 { [r, g, b] } else { [r / a, g / a, b / a] }
}

/// EdgeDetection (output rect = input rect, confirmed by capture): a Sobel/Prewitt gradient
/// magnitude on straight colour, `edge = clamp(strength·(gx²+gy²), 0, 1)`; overlay darkens the
/// original, else an opaque edge map. `prewitt` sets the middle weight (Prewitt 1 / Sobel 2).
pub fn edge_detection(input: &Bitmap, strength: f32, prewitt: bool, overlay: bool, alpha_straight: bool) -> Output {
    let (w, h) = (input.width, input.height);
    let (wi, hi) = (w as i32, h as i32);
    let mw = if prewitt { 1.0 } else { 2.0 };
    // Precompute the straight (unpremultiplied) input once; the 3×3 Sobel/Prewitt reads each input texel
    // for up to nine output pixels and `straight_rgb` unpremultiplies (a divide) each read. It hard-clamps
    // the border, so a clamped read into this table matches an out-of-bounds neighbour. Byte-identical.
    let str3: Vec<[f32; 3]> = (0..(w * h) as usize)
        .map(|i| straight_rgb(input, (i as u32 % w) as i32, (i as u32 / w) as i32, alpha_straight))
        .collect();
    let at = |x: i32, y: i32| str3[y.clamp(0, hi - 1) as usize * w as usize + x.clamp(0, wi - 1) as usize];
    let mut out = vec![0u8; (w * h * 4) as usize];
    for y in 0..hi {
        for x in 0..wi {
            let c = at(x, y);
            let s = |dx: i32, dy: i32| at(x + dx, y + dy);
            let (tl, tm, tr) = (s(-1, -1), s(0, -1), s(1, -1));
            let (ml, mr) = (s(-1, 0), s(1, 0));
            let (bl, bm, br) = (s(-1, 1), s(0, 1), s(1, 1));
            let mut edge = [0f32; 3];
            for k in 0..3 {
                let gx = (tr[k] + br[k] - tl[k] - bl[k]) + mw * (mr[k] - ml[k]);
                let gy = (bl[k] + br[k] - tl[k] - tr[k]) + mw * (bm[k] - tm[k]);
                edge[k] = (strength * (gx * gx + gy * gy)).clamp(0.0, 1.0);
            }
            let a = premul_at(input, x, y, false)[3];
            let px = if overlay {
                pack_premul([(c[0] - edge[0]) * a, (c[1] - edge[1]) * a, (c[2] - edge[2]) * a, a])
            } else {
                pack_premul([edge[0], edge[1], edge[2], 1.0])
            };
            let i = ((y as u32 * w + x as u32) * 4) as usize;
            out[i..i + 4].copy_from_slice(&px);
        }
    }
    Output::new(w, h, 0, 0, out)
}

// --- Emboss ------------------------------------------------------------------------------------

// Rec.601 luma weights, exact f32 bits from the managed shader.
pub(super) const LUMA: [f32; 3] =
    [f32::from_bits(1050220167), f32::from_bits(1058424226), f32::from_bits(1038710997)];

fn luminance(bm: &Bitmap, x: i32, y: i32) -> f32 {
    let s = straight_rgb(bm, x, y, false);
    LUMA[0] * s[0] + LUMA[1] * s[1] + LUMA[2] * s[2]
}

/// Emboss (managed lighting guide; native kernel is an open question): a luminance height field
/// shaded by a distant light at elevation 50°, azimuth = DIRECTION, relief depth HEIGHT. Opaque
/// grayscale output, rect = input rect (confirmed by capture).
pub fn emboss(input: &Bitmap, height: f32, direction_deg: f32) -> Output {
    let dir = direction_deg * std::f32::consts::PI / 180.0;
    let elev = 0.87266463_f32;
    let (cz, sz) = (crate::crt::system_crt::cosf(elev), crate::crt::system_crt::sinf(elev));
    let light = [crate::crt::system_crt::cosf(dir) * cz, crate::crt::system_crt::sinf(dir) * cz, sz];
    let nscale = -height * 0.25;
    let (w, h) = (input.width, input.height);
    let (wi, hi) = (w as i32, h as i32);
    // Precompute the per-texel luminance once; the 3×3 Sobel reads each input texel for up to eight
    // output pixels, and `luminance` unpremultiplies (a divide) each read. `luminance` hard-clamps the
    // border (via `straight_rgb`), so a clamped read into this table matches an out-of-bounds neighbour.
    // Byte-identical.
    let lum: Vec<f32> = (0..(w * h) as usize)
        .map(|i| luminance(input, (i as u32 % w) as i32, (i as u32 / w) as i32))
        .collect();
    let mut out = vec![0u8; (w * h * 4) as usize];
    for y in 0..hi {
        for x in 0..wi {
            let l = |dx: i32, dy: i32| lum[(y + dy).clamp(0, hi - 1) as usize * w as usize + (x + dx).clamp(0, wi - 1) as usize];
            let (tl, tm, tr) = (l(-1, -1), l(0, -1), l(1, -1));
            let (ml, mr) = (l(-1, 0), l(1, 0));
            let (bl, bm, br) = (l(-1, 1), l(0, 1), l(1, 1));
            let sx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
            let sy = (bl + 2.0 * bm + br) - (tl + 2.0 * tm + tr);
            let n = [nscale * sx, nscale * sy, 1.0];
            let inv = 1.0 / (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            let gray = (n[0] * inv * light[0] + n[1] * inv * light[1] + n[2] * inv * light[2]).max(0.0);
            let i = ((y as u32 * w + x as u32) * 4) as usize;
            out[i..i + 4].copy_from_slice(&pack_premul([gray, gray, gray, 1.0]));
        }
    }
    Output::new(w, h, 0, 0, out)
}

// --- Sharpen -----------------------------------------------------------------------------------

// Separable 5-tap Gaussian weights (center = 0.6021547² = 0.36259025), from the managed shader.
pub(super) const SHARP_G: [f32; 3] = [0.6021547, 0.191929, 0.0069936];

/// Sharpen (unsharp mask on luminance): `Δ = sharpness·(L − blur(L))·factor` added equally to the
/// straight RGB, α unchanged. `factor = threshold>0 ? min(1,|h|/threshold) : 1`. The 5×5 blur is the
/// outer product of `SHARP_G`, summed row-major like the shader. Output rect = input rect.
pub fn sharpen(input: &Bitmap, sharpness: f32, threshold: f32) -> Output {
    let (w, h) = (input.width, input.height);
    let mut out = vec![0u8; (w * h * 4) as usize];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let center = straight_rgb(input, x, y, false);
            let center_l = LUMA[0] * center[0] + LUMA[1] * center[1] + LUMA[2] * center[2];
            // Direct2D 10.0.26100.8875's Sharpen uses nine bilinear taps at ±1.033, not a 25-tap Gaussian.
            let tap = |dx: f32, dy: f32| {
                crate::effects::sample::bilin_v128(x as f32 + 0.5 + dx, y as f32 + 0.5 + dy, |sx, sy| {
                    let c = straight_rgb(input, sx, sy, false);
                    [unorm8(c[0]) as i32, unorm8(c[1]) as i32, unorm8(c[2]) as i32, 255]
                }).map(crate::sampling::norm_v128)
            };
            let cross = [tap(0.0, 1.033), tap(1.033, 0.0), tap(0.0, -1.033), tap(-1.033, 0.0)];
            let diagonal = [tap(1.033, 1.033), tap(1.033, -1.033), tap(-1.033, 1.033), tap(-1.033, -1.033)];
            let blurred: [f32; 3] = core::array::from_fn(|c| {
                let d = ((diagonal[0][c] + diagonal[1][c]) + diagonal[2][c]) + diagonal[3][c];
                let a = ((cross[0][c] + cross[1][c]) + cross[2][c]) + cross[3][c];
                a * 0.120269 + (center[c] * 0.36 + d * 0.039731)
            });
            let blur = (LUMA[0] * blurred[0] + LUMA[1] * blurred[1]) + LUMA[2] * blurred[2];
            let d = center_l - blur;
            let factor = if threshold > 0.0 { (d.abs() / threshold).min(1.0) } else { 1.0 };
            let corr = sharpness * d * factor;
            let a = premul_at(input, x, y, false)[3];
            let rgb = [
                (center[0] + corr).clamp(0.0, 1.0),
                (center[1] + corr).clamp(0.0, 1.0),
                (center[2] + corr).clamp(0.0, 1.0),
            ];
            let i = ((y as u32 * w + x as u32) * 4) as usize;
            out[i..i + 4].copy_from_slice(&pack_premul([rgb[0] * a, rgb[1] * a, rgb[2] * a, a]));
        }
    }
    Output::new(w, h, 0, 0, out)
}

// --- DisplacementMap ---------------------------------------------------------------------------

/// Linear-`sample` premultiplied `bm` at scene position `(px,py)` (texel centres at integer+0.5),
/// soft border (transparent outside), using WARP's effect-shader sampler ([`crate::effects::sample`]):
/// the shader's float texcoord → Q8 fixed point → f32 2×2 combine, UNORM8 corners via `·257/65536`.
///
/// ponytail/BLOCKED residual (fx-filters-17, fxsample-01, ≤1 LSB): native's bilinear displacement
/// shader (sample the map, saturate, channel-select, subtract 0.5, scale, then offset the interpolated
/// base and sample the image) computes the image texcoord in **normalised** coordinates —
/// `offset·(1/texSize) + normBase`, with the base and inverse size interpolated from the vertex
/// shader — then the sampler
/// multiplies back by `texSize`. That normalised round-trip rounds differently from this texel-space
/// `base + offset`, flipping the Q8 `floor` within ~1/256 texel of a weight boundary. Reproducing it
/// byte-exact needs WARP's interpolation of that vertex attribute and the realised input0 texture size;
/// a texel-space recompute would risk regressing the passing `fx-filters-12`. Same shared WARP
/// texcoord sub-LSB as [`crate::effects::sample`]; bidirectional.
fn bilinear_premul(bm: &Bitmap, px: f32, py: f32, round: [bool; 2]) -> [f32; 4] {
    let (w, h) = (bm.width as i32, bm.height as i32);
    crate::effects::sample::sample_linear_sel(px, py, round[0], round[1], |x, y| {
        if x < 0 || y < 0 || x >= w || y >= h {
            return [0; 4]; // SOFT border: transparent black (the prepended border transform)
        }
        let i = (y as usize * w as usize + x as usize) * 4;
        let p = &bm.premul_bgra;
        [p[i + 2] as i32, p[i + 1] as i32, p[i] as i32, p[i + 3] as i32] // R,G,B,A
    })
}

/// DisplacementMap: offset input0's bilinear sample by `scalePx·(channel(displacement) − 0.5)`,
/// reading the displacement (input1) point-sampled premultiplied. Output = input0 inflated by
/// ⌈|scalePx|/2⌉ each side (SCALE is in DIPs; 96 DPI ⇒ px). Channel masks pick R/G/B/A.
pub fn displacement_map(image: &Bitmap, dmap: &Bitmap, scale: f32, x_channel: u32, y_channel: u32, round: [bool; 2]) -> Output {
    let r = (scale.abs() / 2.0).ceil() as i32;
    let (iw, ih) = (image.width as i32, image.height as i32);
    let (ow, oh) = (iw + 2 * r, ih + 2 * r);
    let chan = |px: [f32; 4], c: u32| px[c.min(3) as usize];
    let mut out = vec![0u8; (ow * oh * 4) as usize];
    for oy in 0..oh {
        for ox in 0..ow {
            let (sx, sy) = ((ox - r) as f32 + 0.5, (oy - r) as f32 + 0.5);
            let d = sample_premul_f32(dmap, ox - r, oy - r);
            let dx = scale * (chan(d, x_channel) - 0.5);
            let dy = scale * (chan(d, y_channel) - 0.5);
            let px = bilinear_premul(image, sx + dx, sy + dy, round);
            let i = ((oy * ow + ox) * 4) as usize;
            out[i..i + 4].copy_from_slice(&pack_premul(px));
        }
    }
    Output::new(ow as u32, oh as u32, -r, -r, out)
}

// --- ConvolveMatrix ----------------------------------------------------------------------------

/// ConvolveMatrix on straight colour (managed-shader form; native divergence is an open question):
/// true convolution with the flipped kernel, `(Σ tap·sample)/divisor + bias`, optional preserve
/// alpha and clamp. Output grows by the kernel reach; `hard` clamps the border instead of
/// transparent. Taps are spacing-1 (KernelUnitLength resampling not modelled).
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
) -> Output {
    let (ksx, ksy) = (ksx.max(1) as i32, ksy.max(1) as i32);
    // Native passthrough (confirmed by the oracle): an identity (centre-only) kernel with no offset
    // and no preserve-alpha connects the input straight through, ignoring divisor/bias/clamp.
    let n = (ksx * ksy) as usize;
    let center = (ksy / 2 * ksx + ksx / 2) as usize;
    let is_identity = ksx % 2 == 1
        && ksy % 2 == 1
        && kernel.len() >= n
        && kernel[..n].iter().enumerate().all(|(i, &k)| k == if i == center { 1.0 } else { 0.0 });
    if is_identity && !preserve_alpha && offset == [0.0, 0.0] {
        return Output::new(input.width, input.height, 0, 0, input.premul_bgra.clone());
    }
    // baseOffset = -(size-1)/2 + kernelOffset (tap col/row add to this).
    let bx = -((ksx as f32 - 1.0) * 0.5) + offset[0];
    let by = -((ksy as f32 - 1.0) * 0.5) + offset[1];
    let grow_l = (-bx).ceil().max(0.0) as i32;
    let grow_r = (bx + (ksx - 1) as f32).ceil().max(0.0) as i32;
    let grow_t = (-by).ceil().max(0.0) as i32;
    let grow_b = (by + (ksy - 1) as f32).ceil().max(0.0) as i32;
    let (iw, ih) = (input.width as i32, input.height as i32);
    let (ox, oy) = (if hard { 0 } else { -grow_l }, if hard { 0 } else { -grow_t });
    let (ow, oh) = (
        if hard { iw } else { iw + grow_l + grow_r },
        if hard { ih } else { ih + grow_t + grow_b },
    );
    let mut out = vec![0u8; (ow * oh * 4) as usize];
    if let Some(taps) = super::filters_hp::native_taps(ksx, ksy, kernel, divisor, bias, offset, clamp_output) {
        // Native's passes over 8bpc intermediates: (preserve alpha) un-premultiply → UNORM8, the convolve
        // PS (native tap order, unfused mul/mad, first two swapped) → UNORM8 (saturated), then × the
        // original alpha; without preserve alpha the convolve PS runs on the premultiplied input straight
        // into the target.
        let rd = |b: u8| b as f32 / 255.0; // the point sample of a UNORM8 texel (b·257/65536 is ±1 on 7 probes)
        let src = |sx: i32, sy: i32| -> [f32; 4] {
            let p = premul_at(input, sx, sy, hard).map(unorm8).map(rd);
            if !preserve_alpha {
                return p;
            }
            let a = p[3];
            let st = if a == 0.0 { [0.0; 3] } else { [p[0] / a, p[1] / a, p[2] / a] };
            [rd(unorm8(st[0])), rd(unorm8(st[1])), rd(unorm8(st[2])), a]
        };
        for yi in 0..oh {
            for xi in 0..ow {
                let (px, py) = (ox + xi, oy + yi);
                let acc = super::filters_hp::sum_taps(&taps, ksx == 3 && ksy == 3, |dx, dy| src(px + dx, py + dy));
                let px8 = if preserve_alpha {
                    let a = rd(unorm8(premul_at(input, px, py, hard)[3]));
                    let q = acc.map(|v| rd(unorm8(v)));
                    pack_premul([q[0] * a, q[1] * a, q[2] * a, a])
                } else {
                    pack_premul(acc)
                };
                let i = ((yi * ow + xi) * 4) as usize;
                out[i..i + 4].copy_from_slice(&px8);
            }
        }
        return Output::new(ow as u32, oh as u32, ox, oy, out);
    }
    for yi in 0..oh {
        for xi in 0..ow {
            let (px, py) = (ox + xi, oy + yi);
            let mut acc = [0f32; 4];
            for row in 0..ksy {
                for col in 0..ksx {
                    let k = kernel[(ksx * (ksy - 1 - row) + (ksx - 1 - col)) as usize];
                    let sx = px + (bx + col as f32).round() as i32;
                    let sy = py + (by + row as f32).round() as i32;
                    let [r, g, b, a] = premul_at(input, sx, sy, hard);
                    let tap = if preserve_alpha && a > 0.0 { [r / a, g / a, b / a, 0.0] } else { [r, g, b, a] };
                    for c in 0..4 {
                        acc[c] += tap[c] * k;
                    }
                }
            }
            let div = acc.map(|v| v / divisor + bias);
            let px8 = if preserve_alpha {
                let ca = premul_at(input, px, py, hard)[3];
                let rgb = if clamp_output { [div[0].clamp(0.0, 1.0), div[1].clamp(0.0, 1.0), div[2].clamp(0.0, 1.0)] } else { [div[0], div[1], div[2]] };
                pack_premul([rgb[0] * ca, rgb[1] * ca, rgb[2] * ca, ca])
            } else {
                // Non-preserve convolves the premultiplied samples and the shader result IS the
                // premultiplied node value — stored directly, not re-premultiplied (oracle-confirmed).
                let s = if clamp_output { div.map(|v| v.clamp(0.0, 1.0)) } else { div };
                pack_premul(s)
            };
            let i = ((yi * ow + xi) * 4) as usize;
            out[i..i + 4].copy_from_slice(&px8);
        }
    }
    Output::new(ow as u32, oh as u32, ox, oy, out)
}

// --- GaussianBlur / Shadow / DirectionalBlur ---------------------------------------------------

/// The blur's optimization parameters per `D2D1_GAUSSIAN_BLUR_OPTIMIZATION`: `(k, c, threshold,
/// support, msl)`. `s = clamp(k/(σpx+c), 0.1, 1)`; `halving = σpx > threshold ? 0.5 : 1`; the
/// pre-downscale uses MULTI_SAMPLE_LINEAR when `msl` (BALANCED/QUALITY), else LINEAR (SPEED). `support`
/// (the per-mode tap radius the kernel generator reads) is 2.5 for SPEED/BALANCED, 3.0 for QUALITY.
pub(super) fn blur_params(mode: u32) -> (f32, f32, f32, f32, bool) {
    match mode {
        0 => (8.0, 6.0, 1.5, 2.5, false), // SPEED
        2 => (12.0, 6.0, 2.0, 3.0, true), // QUALITY
        _ => (8.0, 6.0, 1.5, 2.5, true),  // BALANCED (default)
    }
}

/// A node buffer in the blur graph: an 8bpc premultiplied-BGRA store (native keeps one between every
/// transform node) plus the **absolute texel index** of its top-left pixel in this node's own grid
/// (`ox,oy`), so the device-anchored SOFT graph can carry negative origins while keeping the texel
/// grid phase fixed to scaled device-0. SOFT fetches transparent outside the stored rect (= native's
/// clip-to-rect then transparent-black); HARD clamps to the edge.
struct Buf {
    w: i32,
    h: i32,
    ox: i32,
    oy: i32,
    hard: bool,
    px: Vec<u8>,
    /// The Shadow's premultiplied colour, multiplied into this pass's float result before its store, `o = r · colour`
    /// (`r` the red, which carries the broadcast alpha). With a pre-downscale Direct2D links its colour matrix into the
    /// expand (DXBC of a BALANCED shadow: `Mul o0, r0.xxxx, cb0[0]`, a rounded store); without one the X pass
    /// blurs the alpha alone and multiplies it by the colour (`Mul o0, r0.xxxx, cb0[1]`), and the upscale is plain.
    tint: Option<[f32; 4]>,
    /// A HARD read past the stored rect mirrors back into it (Direct2D's HARD full-res blur samples its input and its X
    /// pass through a mirror border) instead of clamping to the edge.
    mirror: bool,
}

impl Buf {
    fn new(w: i32, h: i32, ox: i32, oy: i32, hard: bool) -> Self {
        Buf { w, h, ox, oy, hard, px: vec![0u8; (w.max(0) * h.max(0) * 4) as usize], tint: None, mirror: false }
    }
    fn from_bitmap(bm: &Bitmap, hard: bool) -> Self {
        Buf { w: bm.width as i32, h: bm.height as i32, ox: 0, oy: 0, hard, px: bm.premul_bgra.clone(), tint: None, mirror: false }
    }
    /// Premultiplied UNORM8 `[R,G,B,A]` (i32) of the **absolute** texel `(xa,ya)` in this node's grid;
    /// SOFT → 0 outside the stored rect, HARD → clamp to its edge.
    fn fetch(&self, xa: i32, ya: i32) -> [i32; 4] {
        let (x, y) = (xa - self.ox, ya - self.oy);
        let reflect = |v: i32, n: i32| { let p = v.rem_euclid(2 * n); if p >= n { 2 * n - 1 - p } else { p } };
        let (cx, cy) = if self.hard && self.mirror {
            (reflect(x, self.w), reflect(y, self.h))
        } else if self.hard {
            (x.clamp(0, self.w - 1), y.clamp(0, self.h - 1))
        } else if x < 0 || y < 0 || x >= self.w || y >= self.h {
            return [0; 4];
        } else {
            (x, y)
        };
        let i = ((cy * self.w + cx) * 4) as usize;
        [self.px[i + 2] as i32, self.px[i + 1] as i32, self.px[i] as i32, self.px[i + 3] as i32]
    }
    /// Whether absolute texel column `xa` / row `ya` is inside this node's stored rect (the blur's
    /// "source" — a tap outside it reads the transparent border, where WARP floors the Q8 weight).
    fn in_x(&self, xa: i32) -> bool {
        xa >= self.ox && xa < self.ox + self.w
    }
    fn in_y(&self, ya: i32) -> bool {
        ya >= self.oy && ya < self.oy + self.h
    }
    /// Like [`fetch`] but BOTH axes MIRROR (reflect, period `2w`/`2h`) — the native blur's intermediate
    /// sampler on the halving path. The small per-node intermediate reflects, so the content replicates
    /// into the apron (the Y-blur's reach) with no opposite-edge pollution (which a WRAP would cause on a
    /// gradient). Byte-exact vs the captured X-blur output A. `ox/oy` are 0 on this path.
    fn fetch_mir(&self, xa: i32, ya: i32) -> [i32; 4] {
        let refl = |v: i32, n: i32| {
            let p = v.rem_euclid(2 * n);
            if p >= n { 2 * n - 1 - p } else { p }
        };
        let (x, y) = (refl(xa - self.ox, self.w), refl(ya - self.oy, self.h));
        let i = ((y * self.w + x) * 4) as usize;
        [self.px[i + 2] as i32, self.px[i + 1] as i32, self.px[i] as i32, self.px[i + 3] as i32]
    }
    fn put(&mut self, jx: i32, jy: i32, c: [f32; 4]) {
        let i = ((jy * self.w + jx) * 4) as usize;
        let c = self.tint.map_or(c, |pc| pc.map(|v| c[0] * v));
        self.px[i..i + 4].copy_from_slice(&pack_premul(c));
    }
    fn output(self) -> Output {
        Output::new(self.w as u32, self.h as u32, self.ox, self.oy, self.px)
    }
}


/// Native `GenerateKernelTaps`: bilinear tap pairs for a 1-D Gaussian of `sigma` (the reduced σpx·s),
/// returning `(offset, weight)` for the positive half (the kernel applies each at ±offset). Pair `i`:
/// `w0 = exp(−(2i)²/2σ²)`, `w1 = exp(−(2i+1)²/2σ²)`, centre `w0` halved; `offset = 2i + w1/(w0+w1)`,
/// `weight = w0+w1`, all normalised so Σ (incl. mirror) = 1. `powf`/`expf` match the Windows CRT's.
/// `support` is the per-mode optimization tap radius (3.0 QUALITY, 2.5 SPEED/BALANCED) — `R = max(1, ceil(σ·support))`.
pub(super) fn kernel_taps(sigma: f32, support: f32) -> Vec<(f32, f32)> {
    let r = libm::ceilf(sigma * support).max(1.0);
    let n = libm::ceilf(r * 0.5) as usize;
    let s2 = sigma * sigma;
    let two_s2 = s2 + s2;
    let mut taps = Vec::with_capacity(n);
    let mut sum = 0.0f32;
    for i in 0..n {
        let x = (2 * i) as f32;
        // Native computes the Gaussian with an EXACT `x*x` square and Windows' system C runtime f32 `expf`
        // (`crate::crt::system_crt`, the runtime Direct2D calls). `libm::powf(x,2)` is sub-ULP off `x*x` (it's
        // exp2(2·log2 x)), `libm::expf` is correctly rounded where the CRT's is not, and the universal C runtime's
        // `expf` is another algorithm — each gap flips kernel weights (σ2's dropshadow rim; σ 7.2's outermost
        // pair). Verified exact vs the native cbuffer.
        let mut w0 = crate::crt::system_crt::expf(-(x * x) / two_s2);
        let w1 = crate::crt::system_crt::expf(-((x + 1.0) * (x + 1.0)) / two_s2);
        if i == 0 {
            w0 *= 0.5;
        }
        sum += w1 + w0;
        taps.push((w1 / (w1 + w0) + x, w1 + w0));
    }
    let norm = sum + sum;
    for t in &mut taps {
        t.1 /= norm;
    }
    taps
}

/// One separable kernel blur pass, subsampling its axis by `halving`. `out` is pre-sized with
/// its absolute origin; for output pixel `(jx,jy)` the absolute index along the blur axis maps to the
/// source centre `(a+0.5)/halving` and the other axis passes through at `(a+0.5)` (an exact texel), so
/// each tap samples `src` at `centre ± off` (in `src`'s absolute grid). Each tap is a value·128 bilinear
/// normalised by f32 `norm_v128` (== the WARP float-family HW `Sample`), weighted and summed, then stored
/// 8bpc (×255 ties-to-even), summed in the blur PS's own order (below), with `kernel_taps` offsets+weights
/// bit-exact to native's captured cbuffers (N=1,2,3,5,9). So any
/// residual blur ±1 is the graph GEOMETRY (SOFT downscale/halving, HARD-σ2 upscale phase), not this pass.
/// `planes` (the full-res passes) takes each tap's Q8 address from WARP's interpolated texcoord
/// ([`PassPlanes`]); without it the address is the floored `centre ± off`.
fn blur_pass(src: &Buf, out: &mut Buf, taps: &[(f32, f32)], halving: f32, vertical: bool, planes: Option<&PassPlanes>) {
    for jy in 0..out.h {
        for jx in 0..out.w {
            let (ax, ay) = (out.ox + jx, out.oy + jy);
            let (cx, cy) = if vertical {
                (ax as f32 + 0.5, (ay as f32 + 0.5) / halving)
            } else {
                ((ax as f32 + 0.5) / halving, ay as f32 + 0.5)
            };
            // One tap pair's `sample(+off) + sample(−off)`.
            let pair = |i: usize| -> [f32; 4] {
                let off = taps[i].0;
                let v: [[i32; 4]; 2] = core::array::from_fn(|k| {
                    let sign = if k == 0 { off } else { -off };
                    match planes {
                        Some(p) => {
                            let ((x0, wx), (y0, wy)) = p.address(ax - p.at[0], ay - p.at[1], 2 * i + k);
                            let lo = if p.clamp_org { p.org } else { [i32::MIN; 2] };
                            crate::effects::sample::bilin_v128_at(x0, wx, y0, wy, |x, y| src.fetch(x.max(lo[0]), y.max(lo[1])))
                        }
                        None => {
                            let (px, py) = if vertical { (cx, cy + sign) } else { (cx + sign, cy) };
                            crate::effects::sample::bilin_v128(px, py, |x, y| src.fetch(x, y))
                        }
                    }
                });
                core::array::from_fn(|c| crate::sampling::norm_v128(v[0][c]) + crate::sampling::norm_v128(v[1][c]))
            };
            let n = taps.len();
            // The unrolled 1- to 4-pair shader sums its samples singly: tap +1 (−0 for one pair), +0, the other +taps,
            // then every −tap in order, each a multiply then an add onto the running sum (its DXBC).
            if n <= 4 {
                let tap = |i: usize, k: usize| -> [f32; 4] {
                    let sign = if k == 0 { taps[i].0 } else { -taps[i].0 };
                    let v = match planes {
                        Some(p) => {
                            let ((x0, wx), (y0, wy)) = p.address(ax - p.at[0], ay - p.at[1], 2 * i + k);
                            let lo = if p.clamp_org { p.org } else { [i32::MIN; 2] };
                            crate::effects::sample::bilin_v128_at(x0, wx, y0, wy, |x, y| src.fetch(x.max(lo[0]), y.max(lo[1])))
                        }
                        None => {
                            let (px, py) = if vertical { (cx, cy + sign) } else { (cx + sign, cy) };
                            crate::effects::sample::bilin_v128(px, py, |x, y| src.fetch(x, y))
                        }
                    };
                    v.map(crate::sampling::norm_v128)
                };
                let order: Vec<(usize, usize)> = if n == 1 {
                    vec![(0, 1), (0, 0)]
                } else {
                    [(1, 0), (0, 0)].into_iter().chain((2..n).map(|i| (i, 0))).chain((0..n).map(|i| (i, 1))).collect()
                };
                let mut acc = [0.0f32; 4];
                for (j, &(i, k)) in order.iter().enumerate() {
                    let v = tap(i, k);
                    for c in 0..4 { acc[c] = if j == 0 { v[c] * taps[i].1 } else { v[c] * taps[i].1 + acc[c] }; }
                }
                out.put(jx, jy, acc);
                continue;
            }
            // The looping shader (five pairs or more, its DXBC): while more than 8 pairs remain, a group of 8 sums as
            // `p0·w0 + p1·w1`, then `+ pk·wk` for the next six, into the running total; the last ≤8 pairs chain the
            // same way from zero, and the two totals add at the end. Each multiply-add is a multiply then an add.
            let mut total = [0.0f32; 4];
            let mut i0 = 0;
            while n - i0 > 8 {
                let (p0, p1) = (pair(i0), pair(i0 + 1));
                let mut g: [f32; 4] = core::array::from_fn(|c| p0[c] * taps[i0].1 + p1[c] * taps[i0 + 1].1);
                for i in i0 + 2..i0 + 8 {
                    let p = pair(i);
                    for c in 0..4 { g[c] = p[c] * taps[i].1 + g[c]; }
                }
                for c in 0..4 { total[c] += g[c]; }
                i0 += 8;
            }
            let mut rest = [0.0f32; 4];
            for i in i0..n {
                let p = pair(i);
                for c in 0..4 { rest[c] = p[c] * taps[i].1 + rest[c]; }
            }
            let acc: [f32; 4] = core::array::from_fn(|c| total[c] + rest[c]);
            out.put(jx, jy, acc);
        }
    }
}

/// A full-res blur pass's per-tap Q8 texel addresses as WARP interpolates them (WARP's triangle setup
/// traced on pdnv5-dropshadow and imgbrush-02, plus vertex-buffer captures). Direct2D
/// draws each pass as screen quads whose corners carry `((coord − org) + off)·(1/size)` (the tap
/// offset on the blurred axis only); WARP fits each triangle's plane on the snapped corners
/// ([`crate::warp::plane::setup_triangle_attr`]) and the kernel floors `(u·size − 0.5)·256` per lane at
/// the pixel centre. The f32 planes are why some taps land a Q8 step off `centre ± off`.
///   * X pass: one quad over the requested output columns × the input rows, into an atlas at column
///     `req_l` and row `floor(req_t − reach)`, sampling the input.
///   * Y pass: one quad over the requested rect, at its place in the draw's viewport (the target, or the
///     atlas at the requested top-left), sampling the X pass's atlas (power of two, ≥512).
///   * Y pass under a mirror/wrap image brush: [`blur_cells`] — one quad per border cell.
struct PassPlanes {
    /// The input coordinate drawn at screen pixel 0.
    at: [i32; 2],
    /// The sampled texture's texel 0 (in input coordinates) and its size.
    org: [i32; 2],
    size: [f32; 2],
    /// The quad's screen rect `[l, t, w, h]` (the LT→RB diagonal splits the triangles).
    quad: [f32; 4],
    /// Per tap `2·i + (sign < 0)`, per triangle (RT, LT, RB) / (LB, LT, RB): the `u` and `v` planes.
    uv: Vec<[[[f32; 3]; 2]; 2]>,
    /// The texture is an atlas with texel 0 at its edge (the Y pass's): taps before it clamp there.
    clamp_org: bool,
    /// The blur pass's pixel shader forms its taps itself from the interpolated scene position, texcoord and
    /// `1/size` (its vertex shader passes them through): per triangle the scene-position and texcoord planes,
    /// then the scales and the signed tap offsets.
    ps: Option<PsTaps>,
}

struct PsTaps {
    /// Per triangle, per axis: the scene-position plane and the texcoord plane.
    pos: [[[f32; 3]; 2]; 2],
    tex: [[[f32; 3]; 2]; 2],
    inv: [f32; 2],
    scale: [f32; 2],
    offs: Vec<f32>,
    axis: usize,
}

impl PassPlanes {
    /// The quad over screen rect `scr` whose edges sample input coordinates `inp` (`[x at l, y at t, x at
    /// r, y at b]`, reversed for a mirrored cell); `axis` is the blurred one, which the pass scales by
    /// `scale` (`1/halving`).
    #[allow(clippy::too_many_arguments)]
    fn new(at: [i32; 2], scr: [i32; 4], inp: [f32; 4], org: [i32; 2], size: [f32; 2], taps: &[(f32, f32)], axis: usize, scale: f32) -> Self {
        let offs: Vec<f32> = taps.iter().flat_map(|&(off, _)| [off, -off]).collect();
        // Up to four tap pairs the blur shader is unrolled and its vertex shader forms every tap's texcoord,
        // `((p + off) − a)·(1/size) + texcoord` with `p = (a − ½)(scale − 1) + a`, which WARP interpolates per triangle
        // (DXBC of the 1- to 4-pair variants).
        if taps.len() <= 4 {
            let attr = |i: usize, ax: usize, a: f32| {
                let (sc, s) = if ax == axis { (scale, offs[i]) } else { (1.0, 0.0) };
                let p = (a - 0.5) * (sc - 1.0) + a;
                ((p + s) - a) * (1.0 / size[ax]) + (a - org[ax] as f32) * (1.0 / size[ax])
            };
            return Self::with_attr(at, scr, inp, org, size, offs.len(), attr, axis == 1);
        }
        let mut pl = Self::with_attr(at, scr, inp, org, size, 0, |_, _, _| 0.0, axis == 1);
        // The vertex buffer carries each corner's scene position and texcoord `(a − org)·(1/size)`; WARP fits a plane to
        // each per triangle (DDI and JIT plane block of a QUALITY SOFT blur's X pass).
        use crate::warp::plane::setup_triangle_attr;
        let [l, t, r, b] = scr;
        let corners = [[(l, t), (r, b), (r, t)], [(l, b), (l, t), (r, b)]];
        let coord = |c: (i32, i32), ax: usize| if ax == 0 { if c.0 == l { inp[0] } else { inp[2] } } else if c.1 == t { inp[1] } else { inp[3] };
        let inv = size.map(|v| 1.0 / v);
        let fit = |k: usize, f: &dyn Fn(usize, (i32, i32)) -> f32| -> [[f32; 3]; 2] {
            let (sx, sy) = (corners[k].map(|(x, _)| x as i64 * 256), corners[k].map(|(_, y)| y as i64 * 256));
            core::array::from_fn(|ax| setup_triangle_attr(sx, sy, corners[k].map(|c| f(ax, c))))
        };
        let pos = [0, 1].map(|k| fit(k, &|ax, c| coord(c, ax)));
        let tex = [0, 1].map(|k| fit(k, &|ax, c| (coord(c, ax) - org[ax] as f32) * inv[ax]));
        let sc = core::array::from_fn(|ax| if ax == axis { scale } else { 1.0 });
        pl.ps = Some(PsTaps { pos, tex, inv, scale: sc, offs, axis });
        pl
    }

    /// The quad's planes for `n` texcoords whose value at a corner is `attr(i, axis, input coordinate)`.
    #[allow(clippy::too_many_arguments)]
    fn with_attr(at: [i32; 2], scr: [i32; 4], inp: [f32; 4], org: [i32; 2], size: [f32; 2], n: usize, attr: impl Fn(usize, usize, f32) -> f32, clamp_org: bool) -> Self {
        use crate::warp::plane::setup_triangle_attr;
        let [l, t, r, b] = scr;
        // WARP sets the upper triangle up from LT.
        let corners = [[(l, t), (r, b), (r, t)], [(l, b), (l, t), (r, b)]];
        let screen = corners.map(|c| (c.map(|(x, _)| x as i64 * 256), c.map(|(_, y)| y as i64 * 256)));
        let coord = |c: (i32, i32)| [if c.0 == l { inp[0] } else { inp[2] }, if c.1 == t { inp[1] } else { inp[3] }];
        let uv = (0..n)
            .map(|i| core::array::from_fn(|ax| core::array::from_fn(|k| setup_triangle_attr(screen[k].0, screen[k].1, corners[k].map(|c| attr(i, ax, coord(c)[ax]))))))
            .collect();
        PassPlanes { at, org, size, quad: [l as f32, t as f32, (r - l) as f32, (b - t) as f32], uv, clamp_org, ps: None }
    }

    /// The X pass's planes: requested output columns `[l, r)` from `out` ∩ `req`.
    fn x_pass(src: &Buf, taps: &[(f32, f32)], out: [i32; 3], req: Option<[i32; 4]>) -> Option<Self> {
        let [out_l, out_t, ow] = out;
        let req = req.unwrap_or([i32::MIN, i32::MIN, i32::MAX, i32::MAX]);
        let (l, r, top) = (out_l.max(req[0]), (out_l + ow).min(req[2]), out_t.max(req[1]));
        if src.w <= 0 || src.h <= 0 || r <= l {
            return None;
        }
        let at = [l, (top as f32 - reach(taps)).floor() as i32];
        let (t, b) = (src.oy, src.oy + src.h);
        Some(Self::new(at, [0, t - at[1], r - l, b - at[1]], [l, t, r, b].map(|v| v as f32), [src.ox, src.oy], [src.w as f32, src.h as f32], taps, 0, 1.0))
    }

    /// The Y pass's planes over the requested rect ∩ the output, drawn at its place in the draw's
    /// viewport, sampling the X pass laid out by `x`.
    fn y_pass(x: &PassPlanes, taps: &[(f32, f32)], out: [i32; 4], req: Request) -> Option<Self> {
        let Request { rect: q, origin: at, .. } = req;
        let [l, t, r, b] = [out[0].max(q[0]), out[1].max(q[1]), out[2].min(q[2]), out[3].min(q[3])];
        if r <= l || b <= t {
            return None;
        }
        Some(Self::new(at, [l - at[0], t - at[1], r - at[0], b - at[1]], [l, t, r, b].map(|v| v as f32), x.at, x.atlas_size(), taps, 1, 1.0))
    }

    /// The X pass's atlas: the next power of two of the quad's extent, at least 512 per axis.
    /// ponytail: per-axis power of two is inferred from 512² captures; a >512 extent is unverified.
    fn atlas_size(&self) -> [f32; 2] {
        [self.quad[2], self.quad[1] + self.quad[3]].map(|e| (e as u32).next_power_of_two().max(512) as f32)
    }

    /// `((x0, wx), (y0, wy))` in absolute texels for screen pixel `(sx, sy)` and tap `tap`.
    fn address(&self, sx: i32, sy: i32, tap: usize) -> ((i32, i32), (i32, i32)) {
        let p = [sx as f32 + 0.5, sy as f32 + 0.5];
        let [ql, qt, w, h] = self.quad;
        let k = usize::from((p[1] - qt) * w > (p[0] - ql) * h);
        let eval = |a: &[f32; 3]| p[0] * a[0] + (p[1] * a[1] + a[2]);
        let q = |ax: usize| {
            let u = match &self.ps {
                // The pixel shader: `base = ((a − ½)·scale − (a − ½))·(1/size) + texcoord`, tap `±off·(1/size) + base`.
                Some(ps) => {
                    let a = eval(&ps.pos[k][ax]) - 0.5;
                    let base = (a * ps.scale[ax] - a) * ps.inv[ax] + eval(&ps.tex[k][ax]);
                    if ax == ps.axis { ps.offs[tap] * ps.inv[ax] + base } else { base }
                }
                None => eval(&self.uv[tap][ax][k]),
            }
            .clamp(-10.0, 10.0);
            // WARP's sampler floors `(u·n − ½)·256` on a power-of-two axis (y only when both are) and, on any other,
            // clamps u to [−1, 2] and rounds `u·256n` to nearest-even, less 128 (JIT of the X pass on a 17×23 input,
            // whose first row sits on the tie). A Shadow pass reading only the alpha keeps the same rule (a 16×23
            // input's X pass floors its x).
            let pow2 = self.size[0].fract() == 0.0 && (self.size[0] as u32).is_power_of_two() && (ax == 0 || (self.size[1] as u32).is_power_of_two());
            let q = if !pow2 {
                (u.clamp(-1.0, 2.0) * (256.0 * self.size[ax])).round_ties_even() as i32 - 128
            } else {
                ((u * self.size[ax] - 0.5) * 256.0).floor() as i32
            };
            ((q >> 8) + self.org[ax], q & 0xff)
        };
        (q(0), q(1))
    }
}

fn reach(taps: &[(f32, f32)]) -> f32 {
    taps.iter().fold(0.0f32, |m, t| m.max(t.0))
}

/// What a draw asks of a blur, in input pixels: its passes cover `rect`, and the last pass lands at
/// `rect − origin` in the draw's viewport (the target for a direct draw, else an atlas at `rect`). `input` is the
/// input's device origin, where Direct2D anchors the graph's grids.
#[derive(Clone, Copy)]
pub(crate) struct Request {
    pub rect: [i32; 4],
    pub origin: [i32; 2],
    pub input: [i32; 2],
}

/// An image brush's border cells (Direct2D's repeating-cell and nine-grid vertex generation):
/// the brush's need region (`origin`, `size`, input coordinates) and, per cell, its screen rect in that
/// region's intermediate and the input coordinates at its edges (`[x at l, y at t, x at r, y at b]`).
pub(crate) struct BorderCells {
    pub origin: [i32; 2],
    pub size: [i32; 2],
    pub cells: Vec<([i32; 4], [f32; 4])>,
}

/// The Y pass drawn as border cells ([`BorderCells`]): Direct2D collapses the border into the blur's last
/// pass, so each cell is its own quad (and WARP plane) over the intermediate, sampling the X pass laid
/// out by `x` (imgbrush-02's 54-vertex draw).
fn blur_cells(xp: &Buf, x: &PassPlanes, taps: &[(f32, f32)], c: &BorderCells, hard: bool) -> Buf {
    let mut out = Buf::new(c.size[0], c.size[1], c.origin[0], c.origin[1], hard);
    for &(scr, inp) in &c.cells {
        let p = PassPlanes::new([0, 0], scr, inp, x.at, x.atlas_size(), taps, 1, 1.0);
        let mut cell = Buf::new(scr[2] - scr[0], scr[3] - scr[1], scr[0], scr[1], hard);
        blur_pass(xp, &mut cell, taps, 1.0, true, Some(&p));
        let row = cell.w as usize * 4;
        for y in 0..cell.h as usize {
            let o = ((scr[1] as usize + y) * out.w as usize + scr[0] as usize) * 4;
            out.px[o..o + row].copy_from_slice(&cell.px[y * row..(y + 1) * row]);
        }
    }
    out
}

/// Direct2D's Border shader pass for a mirror/wrap image brush whose need region spans more than 25
/// tiles (where it switches from per-cell quads to a vertex+pixel shader border): one quad over the need region `n` whose
/// corners carry the tile coordinate `t = (scene − tile_lt) / size`, then per pixel the PS
/// `|rne(frc(t/2)) − frc(t)|` (mirror; `frc(t)` for wrap) scaled back into the tile, which sits at the
/// origin of a 512 atlas, and bilinear-sampled there (ibf-intid DDI capture).
pub(crate) fn border_ps(tile: &Bitmap, tile_lt: [i32; 2], n: [i32; 4], extend: [u32; 2]) -> Output {
    use crate::warp::plane::setup_triangle_attr;
    let src = Buf::from_bitmap(tile, false);
    let size = [tile.width as f32, tile.height as f32];
    let (w, h) = (n[2] - n[0], n[3] - n[1]);
    let mut out = Buf::new(w, h, n[0], n[1], false);
    let corners = [[(0, 0), (w, h), (w, 0)], [(0, h), (0, 0), (w, h)]];
    let screen = corners.map(|c| (c.map(|(x, _)| x as i64 * 256), c.map(|(_, y)| y as i64 * 256)));
    let planes: [[[f32; 3]; 2]; 2] = core::array::from_fn(|ax| {
        core::array::from_fn(|k| setup_triangle_attr(screen[k].0, screen[k].1, corners[k].map(|c| [c.0 + n[0] - tile_lt[0], c.1 + n[1] - tile_lt[1]][ax] as f32 / size[ax])))
    });
    for jy in 0..h {
        for jx in 0..w {
            let p = [jx as f32 + 0.5, jy as f32 + 0.5];
            let k = usize::from(p[1] * w as f32 > p[0] * h as f32);
            let q = |ax: usize| {
                let a = &planes[ax][k];
                let t = a[0] * p[0] + (a[1] * p[1] + a[2]);
                // `frc` clamps to the largest float below one, as WARP's border kernel does.
                let f = (t - t.floor()).min(f32::from_bits(0x3f7f_ffff));
                let m = if extend[ax] == 2 { let half = t * 0.5; ((half - half.floor()).round_ties_even() - f).abs() } else { f };
                // A point sample (one fetch per lane): `floor(t·W·256) >> 8`, clamped to the atlas.
                ((m.clamp(-10.0, 10.0) * size[ax] * 256.0).floor() as i32 >> 8).max(0)
            };
            // The tile sits at the atlas origin.
            let c = src.fetch(q(0), q(1));
            out.put(jx, jy, c.map(|v| v as f32 / 255.0));
        }
    }
    out.output()
}

/// Whether a GaussianBlur of `sigma`/`mode` runs the full-res X/Y passes (no pre-scale, no halving),
/// the graph whose Y pass [`BorderCells`] can replace.
pub(crate) fn blur_full_res(sigma: f32, mode: u32) -> bool {
    let (k, c, thr, _, _) = blur_params(mode);
    sigma > 0.0 && (k / (sigma + c)).clamp(0.1, 1.0) * if sigma > thr { 0.5 } else { 1.0 } >= 1.0
}

/// Whether Direct2D draws a full-res blur's Y pass as an image brush's border cells ([`BorderCells`]) rather than
/// rendering the pass and sampling it: its resolver folds the pass shader into the parent border only when the border's
/// input `request` lies within the pass's output rect `out` (or is empty) and, for a shader past 12 instructions (three
/// or more tap pairs, an instruction hint over 50 above 0.25), when the border's need `n` is at most twice the request's
/// area.
pub(crate) fn blur_folds_into_border(sigma: f32, mode: u32, out: [i32; 4], request: [i32; 4], n: [i32; 4]) -> bool {
    let area = |r: [i32; 4]| (r[2] - r[0]).max(0) as u64 * (r[3] - r[1]).max(0) as u64;
    let inside = request[0] >= out[0] && request[1] >= out[1] && request[2] <= out[2] && request[3] <= out[3];
    if !inside && area(request) > 0 {
        return false;
    }
    kernel_taps(sigma, blur_params(mode).3).len() <= 2 || (area(request) > 0 && area(n) / area(request) <= 2)
}

/// Pre-downscale affine transform by `scale` (<1): MULTI_SAMPLE_LINEAR when `msl` (BALANCED/QUALITY),
/// else LINEAR. Output pixel `(jx,jy)` (absolute index via `out`'s origin) maps to source centre
/// `(a+0.5)/scale` in `src`'s (= the input's) absolute grid; the MSL footprint is `1/scale` texels.
fn resample(src: &Buf, out: &mut Buf, scale: f32, msl: bool, place: Option<([i32; 4], [i32; 2])>) {
    use crate::sampling::{float_plane, norm_v128, plane_quant};
    // Native's affine shrink = WARP's minify sampler: a normalized texcoord plane + Q8
    // `plane_quant` addressing (same as `bitmap::blit_msl`) and the 4 rotated-grid MSAA sub-samples for
    // MULTI_SAMPLE_LINEAR (LINEAR = the single centre tap) — but the border is the input's own
    // (transparent outside for SOFT via `src.fetch`), not Clamp. The `value·128` bilinear combine is
    // WARP's (`crate::effects::sample::bilin_v128`), inlined here to take the plane_quant texel/weight.
    const MSAA4: [(f32, f32); 4] = [(0.375, -0.125), (-0.125, -0.375), (-0.375, 0.125), (0.125, 0.375)];
    let center = [(0.0f32, 0.0f32)];
    let grid: &[(f32, f32)] = if msl { &MSAA4 } else { &center };
    let wt = 1.0 / grid.len() as f32;
    let (tw, th) = (src.w as f32, src.h as f32);
    let (udx, uc) = float_plane(0.0, tw, 0.0, tw * scale, tw);
    let (vdy, vc) = float_plane(0.0, th, 0.0, th * scale, th);
    // `place`: the SOFT pre's drawn quad `[l, t, r, b]` and its render target's origin (the X pass's
    // need rect), which move WARP's plane intercepts. A HARD pre-downscale has no `place`; it draws the
    // whole output rect, and native maps its MSL samples through the SAME homogeneous plane (vs35/ps34,
    // `s·(pos+off)/s²`, rounded s²), so build the plane for it too — the net-linear `plane_quant` is a
    // ULP off at some sub-texel phases (p16-s5: 10→2 pre diffs).
    //
    // The HARD raw+refined plane is restricted to `scale < 0.5`: it is the exact pre only for the
    // co-port's device-anchored FIVE-node graph (the extreme-downscale regime the blur_graph co-port owns,
    // where native materialises the separate MSL-mip + resample). In the moderate inline regime
    // (0.5 ≤ scale < 1, the fused four-node graph) the inline conv/upscale carry a compensating ½-LSB that
    // the legacy `plane_quant` pre cancels to byte-exact (e.g. blur-hard-content-np14-s8, σ=8 s=0.571); the
    // exact pre would break that cancellation (maxd 1). So HARD scale≥0.5 keeps the legacy `plane_quant`.
    // A LINEAR SOFT pre (SPEED) is the same draw with one sample: its VS maps each corner to `(s·a)/s²` and
    // outputs `(mapped − a)·(1/size) + texcoord` (DXBC of a SPEED shadow's pre).
    let sample_planes = ((msl && ((src.hard && scale < 0.5) || place.is_some())) || (!msl && place.is_some())).then(|| {
        use crate::warp::plane::{setup_triangle_attr, warp_fit_linear};
        let (q, org) = place.unwrap_or(([out.ox, out.oy, out.ox + out.w, out.oy + out.h], [out.ox, out.oy]));
        let positions = [[q[0], q[1]], [q[2], q[1]], [q[2], q[3]], [q[0], q[3]]].map(|p| p.map(|v| v as f32));
        let screen = positions.map(|p| [(p[0] as i64 - org[0] as i64) * 256, (p[1] as i64 - org[1] as i64) * 256]);
        // The HARD pre-downscale matches WARP's MSL via the raw+refined texcoord planes (bitmap::blit_msl,
        // 8bb48cc0): per triangle, per axis, the BASE (offset-0) plane `np` (refined gradients) and its raw
        // round-tripped `rp` (`(g+c)-c` in f32) — base texcoord from `rp`, the 4 MSAA offsets from `np`.
        // The SOFT pre-downscale (`place`) keeps the legacy offset-baked per-sample plane (the raw+refined
        // regresses its sub-texel ties, e.g. recap 1850); `hard_planes` None → the else arm uses `soft`.
        let hard_planes = src.hard.then(|| {
            [[0usize, 1, 2], [3, 0, 2]].map(|ids| {
                let sx = ids.map(|i| screen[i][0]);
                let sy = ids.map(|i| screen[i][1]);
                core::array::from_fn::<([f32; 3], [f32; 3]), 2, _>(|c| {
                    let np = setup_triangle_attr(sx, sy, ids.map(|i| (positions[i][c] / scale) / [tw, th][c]));
                    (np, warp_fit_linear(&np))
                })
            })
        });
        let det = scale * scale;
        let soft_planes = (!src.hard).then(|| {
            let attrs = positions.map(|p| core::array::from_fn::<_, 8, _>(|i| {
                let c = i % 2;
                if !msl {
                    let inv = 1.0 / [tw, th][c];
                    return ((p[c] * scale) / det - p[c]) * inv + p[c] * inv;
                }
                let off = if c == 0 { MSAA4[i / 2].0 } else { MSAA4[i / 2].1 };
                let mapped = ((p[c] + off) * scale) / det;
                p[c] / [tw, th][c] - (p[c] - mapped) / [tw, th][c]
            }));
            [[0usize, 1, 2], [3, 0, 2]].map(|ids| {
                let p = ids.map(|i| screen[i]);
                core::array::from_fn::<_, 8, _>(|c| setup_triangle_attr(p.map(|v| v[0]), p.map(|v| v[1]), ids.map(|i| attrs[i][c])))
            })
        });
        (q, org, hard_planes, soft_planes)
    });
    for jy in 0..out.h {
        for jx in 0..out.w {
            let (ax, ay) = ((out.ox + jx) as f32, (out.oy + jy) as f32);
            let mut acc = [0.0f32; 4];
            let mut raw = [0i32; 4];
            for (j, &(ox, oy)) in grid.iter().enumerate() {
                let ((x0, wx), (y0, wy)) = if let Some((q, org, hard_planes, soft_planes)) = &sample_planes {
                    let p = [ax + 0.5 - org[0] as f32, ay + 0.5 - org[1] as f32];
                    let tri = usize::from((ay + 0.5 - q[1] as f32) * (q[2] - q[0]) as f32 > (ax + 0.5 - q[0] as f32) * (q[3] - q[1]) as f32);
                    let qf = |c: usize| {
                        let u = if let Some(hp) = hard_planes {
                            // HARD: base texcoord from the raw plane; MSAA offset from the refined gradients.
                            let (np, rp) = &hp[tri][c];
                            (rp[0] * p[0] + (rp[1] * p[1] + rp[2])) + (ox * np[0] + oy * np[1])
                        } else {
                            // SOFT: legacy offset-baked per-sample plane.
                            let a = &soft_planes.as_ref().unwrap()[tri][2 * j + c];
                            a[0] * p[0] + (a[1] * p[1] + a[2])
                        };
                        // WARP's sampler rounds the address on a non-power-of-two axis (see `PassPlanes::address`).
                        let n = [tw, th][c];
                        let pow2 = (tw as u32).is_power_of_two() && (c == 0 || (th as u32).is_power_of_two());
                        let q = if pow2 { ((u * n - 0.5) * 256.0).floor() as i32 } else { (u * (256.0 * n)).round_ties_even() as i32 - 128 };
                        (q >> 8, q & 255)
                    };
                    (qf(0), qf(1))
                } else {
                    (plane_quant(udx, uc, tw, ax + 0.5 + ox), plane_quant(vdy, vc, th, ay + 0.5 + oy))
                };
                let (c00, c10, c01, c11) = (src.fetch(x0, y0), src.fetch(x0 + 1, y0), src.fetch(x0, y0 + 1), src.fetch(x0 + 1, y0 + 1));
                for (c, a) in acc.iter_mut().enumerate() {
                    let (p, q, r, s) = (c00[c], c10[c], c01[c], c11[c]);
                    let top = ((q - p) * wx + (p << 8)) >> 1;
                    let bot = ((s - r) * wx + (r << 8)) >> 1;
                    let v = top + (((wy as i64) << 8) * (bot - top) as i64 >> 16) as i32;
                    raw[c] = v;
                    *a += norm_v128(v) * wt;
                }
            }
            // A single-sample SOFT pre is a plain copy of its sample, which WARP keeps in integer and stores as the
            // value·128 lerp shifted down 7, truncating (JIT of a SPEED shadow's pre).
            if !msl && place.is_some() {
                let i = ((jy * out.w + jx) * 4) as usize;
                out.px[i..i + 4].copy_from_slice(&[raw[2], raw[1], raw[0], raw[3]].map(|v| (v >> 7).clamp(0, 255) as u8));
                continue;
            }
            out.put(jx, jy, acc);
        }
    }
}

/// Post-upscale affine transform by `1/sh` (LINEAR) with the `halving·0.5−0.5` half-texel translate,
/// `sh = s·halving`. Output device pixel `(jx,jy)` (absolute device index `d = out.ox+jx`) samples `src`
/// back at `(d+0.5)·sh − (halving·0.5−0.5)` in `src`'s absolute grid. The value·128 bilinear is normalised
/// in f32 (`norm_v128`); the 8bpc store **truncates toward zero** for BOTH borders — proven
/// byte-exact by node copy-out: SOFT via `D2D_BLUR_CLEAR_TRANSPARENT` (round was +1 at the blur tail), and
/// HARD via the per-node capture of pdnv5-gaussianblur (native's gradient pass vs our `D2D_BLUR_NODES`
/// — X-blur and Y-blur are byte-exact, and truncating the upscale is 0/188 vs native, round was 188/188).
fn upscale(src: &Buf, out: &mut Buf, sh: f32, halving: f32, ap: f32, planes: Option<&PassPlanes>) {
    use crate::effects::sample::bilin_v128;
    use crate::sampling::norm_v128;
    let tpost = halving * 0.5 - 0.5;
    // `as u8` truncates toward zero and saturates — native's truncating final conversion (both borders).
    let t = |x: f32| (x.clamp(0.0, 1.0) * 255.0) as u8;
    for jy in 0..out.h {
        for jx in 0..out.w {
            // The upscale's sampler lerps each column (Y) first, then across (fxdpi-blur3-d96's
            // transposed pair; the blur passes' one-axis taps can't tell the order).
            let v = if let Some(p) = planes {
                let ((x0, wx), (y0, wy)) = p.address(out.ox + jx - p.at[0], out.oy + jy - p.at[1], 0);
                crate::effects::sample::bilin_v128_at(y0, wy, x0, wx, |y, x| src.fetch(x, y))
            } else {
                let cx = ((out.ox + jx) as f32 + 0.5) * sh - tpost + ap;
                let cy = ((out.oy + jy) as f32 + 0.5) * sh - tpost + ap;
                bilin_v128(cy, cx, |y, x| src.fetch(x, y))
            };
            let c = core::array::from_fn::<f32, 4, _>(|k| norm_v128(v[k]));
            let i = ((jy * out.w + jx) * 4) as usize;
            // The expand with the shadow colour linked in stores rounded, like any colour-matrix pass (VM captures of
            // BALANCED and sigma-9 shadows); the plain upscale truncates.
            if let Some(pc) = out.tint {
                out.px[i..i + 4].copy_from_slice(&pack_premul(pc.map(|v| c[0] * v)));
                continue;
            }
            out.px[i..i + 4].copy_from_slice(&[t(c[2]), t(c[1]), t(c[0]), t(c[3])]);
        }
    }
}

/// Direct2D's SOFT blur output span `(start, len)` over an input of `n` pixels, mapped node by node:
/// the pre-downscale affine transform (scale `s`; MULTI_SAMPLE_LINEAR expands its quad a pixel),
/// each kernel pass (its input-rect mapping with the max tap offset `m` and scale
/// `1/h`), then the post affine transform (`1/(s·h)` after the `−(½ − h/2)` shift). A transform
/// inflates by half a pixel and keeps the pixels whose centres land inside (`RoundSat` left,
/// `⌈x − ½⌉` right). Exact on 363 GetImageLocalBounds probes (blur-bounds-survey).
pub(super) fn soft_bounds(n: i32, s: f32, h: f32, msl: bool, m: f32) -> (i32, i32) {
    let (l, r) = soft_chain(n, s, h, msl, m)[2];
    (l, r - l)
}

/// [`soft_bounds`] per node: `[pre, kernel pass, output]` as `[l, r)` spans.
fn soft_chain(n: i32, s: f32, h: f32, msl: bool, m: f32) -> [(i32, i32); 3] {
    soft_chain_at(0, n, s, h, msl, m)
}

/// [`soft_chain`] over the input span `[l0, r0)` in device pixels: every node's grid is anchored at scaled device 0.
fn soft_chain_at(l0: i32, r0: i32, s: f32, h: f32, msl: bool, m: f32) -> [(i32, i32); 3] {
    let transform = |l: i32, r: i32, k: f32, dx: f32| (round_sat((l as f32 - 0.5) * k + dx), (((r as f32 + 0.5) * k + dx) - 0.5).ceil() as i32);
    let (mut l, mut r) = (l0, r0);
    if s < 1.0 {
        (l, r) = transform(l, r, s, 0.0);
        if msl {
            (l, r) = (l - 1, r + 1);
        }
    }
    let pre = (l, r);
    let sx = 1.0 / h;
    (l, r) = (round_sat((((l as f32 - 0.5) - m) - 0.5) / sx + 0.5), (((((r as f32 + 0.5) + m) - 0.5) / sx + 0.5) - 0.5).ceil() as i32);
    let kernel = (l, r);
    if s * h != 1.0 {
        let k = 1.0 / (s * h);
        (l, r) = transform(l, r, k, -(0.5 - h * 0.5) * k);
    }
    [pre, kernel, (l, r)]
}

/// Round-saturate to i32: nearest, ties up.
fn round_sat(x: f32) -> i32 {
    let r = x.round_ties_even();
    (if r - x == -0.5 { r + 1.0 } else { r }) as i32
}

/// A kernel pass's output-to-input rect mapping: the input rect (`[l, t, r, b]`) a pass scaling by `sc`
/// with max tap offset `m` (per axis) samples for output rect `o`.
fn kernel_need(o: [i32; 4], sc: [f32; 2], m: [f32; 2]) -> [i32; 4] {
    let lo = |v: i32, a: usize| {
        let p = v as f32 + 0.5;
        round_sat((((p - 0.5) * (sc[a] - 1.0) + p) + -m[a]) - 1.0)
    };
    let hi = |v: i32, a: usize| {
        let p = v as f32 - 0.5;
        (((((p - 0.5) * (sc[a] - 1.0) + p) + m[a]) + 1.0) - 0.5).ceil() as i32
    };
    [lo(o[0], 0), lo(o[1], 1), hi(o[2], 0), hi(o[3], 1)]
}

/// The input rect the SOFT upscale (the affine `x·k + d`, LINEAR) needs for output rect `o`:
/// for its whole output `ob` the cached input rect `ib` grown a texel, else Direct2D's need-rect
/// bounding box over the pixel centres through the adjugate inverse.
fn upscale_need(o: [i32; 4], ob: [i32; 4], ib: [i32; 4], k: f32, d: f32) -> [i32; 4] {
    if o == ob {
        return [ib[0] - 1, ib[1] - 1, ib[2] + 1, ib[3] + 1];
    }
    let inv = |v: f32| (k * v + -(k * d)) / (k * k);
    let lo = |v: i32| round_sat(inv(v as f32 + 0.5) - 1.0);
    let hi = |v: i32| ((inv(v as f32 - 0.5) + 1.0) - 0.5).ceil() as i32;
    [lo(o[0]), lo(o[1]), hi(o[2]), hi(o[3])]
}

/// The SOFT pre-downscaled graph (s<1, or s=1 without the pre: the X pass samples the input texture
/// itself, at its own size): pre ×s → X/Y passes halving their axis → upscale, laid out
/// as Direct2D lays out a request (DDI captures of fxdpi-blur3-d192 drawn whole, clipped and moved): a
/// node's render target starts at its consumer's need rect and the node draws that rect ∩ its
/// bounds, so WARP's texcoord planes (and the pre's MSL sample planes) move with the request.
/// Direct2D anchors every node's grid at scaled device 0, so the graph runs in device pixels, the input at its device
/// origin (`Request::input`): an input off the origin pre-downscales at the phase its position gives (r19 24502: a
/// Shadow of a command list at (2, −52), whose passes moved a pixel against native's).
fn soft_downscaled(src: &Buf, taps: &[(f32, f32)], s: f32, h: f32, msl: bool, req: Option<Request>, tint: Option<[f32; 4]>) -> Output {
    let org = req.map_or([0, 0], |q| q.input);
    let add = |v: i32, k: usize| v.saturating_add(org[k]);
    let req = req.map(|q| Request {
        rect: [add(q.rect[0], 0), add(q.rect[1], 1), add(q.rect[2], 0), add(q.rect[3], 1)],
        origin: [add(q.origin[0], 0), add(q.origin[1], 1)],
        input: org,
    });
    let moved;
    let src = if org == [0, 0] {
        src
    } else {
        moved = Buf { w: src.w, h: src.h, ox: add(src.ox, 0), oy: add(src.oy, 1), hard: src.hard, px: src.px.clone(), tint: src.tint, mirror: src.mirror };
        &moved
    };
    let mut out = soft_downscaled_at(src, taps, s, h, msl, req, tint);
    (out.ox, out.oy) = (out.ox.saturating_sub(org[0]), out.oy.saturating_sub(org[1]));
    out
}

fn soft_downscaled_at(src: &Buf, taps: &[(f32, f32)], s: f32, h: f32, msl: bool, req: Option<Request>, tint: Option<[f32; 4]>) -> Output {
    let m = reach(taps);
    let (cx, cy) = (soft_chain_at(src.ox, src.ox + src.w, s, h, msl, m), soft_chain_at(src.oy, src.oy + src.h, s, h, msl, m));
    let node = |i: usize| [cx[i].0, cy[i].0, cx[i].1, cy[i].1];
    let (pb, kb, ob) = (node(0), node(1), node(2));
    let isect = |a: [i32; 4], b: [i32; 4]| [a[0].max(b[0]), a[1].max(b[1]), a[2].min(b[2]), a[3].min(b[3])];
    let buf = |r: [i32; 4]| Buf::new(r[2] - r[0], r[3] - r[1], r[0], r[1], false);
    let mut out = buf(ob);
    out.tint = tint.filter(|_| s < 1.0);
    let o = req.map_or(ob, |q| isect(q.rect, ob));
    let k = 1.0 / (s * h);
    let d = -(0.5 - h * 0.5) * k;
    let yneed = upscale_need(o, ob, kb, k, d);
    let ydraw = isect(yneed, kb);
    let xneed = kernel_need(ydraw, [1.0, 1.0 / h], [0.0, m]);
    let xdraw = isect(xneed, [kb[0], pb[1], kb[2], pb[3]]);
    let pneed = kernel_need(xdraw, [1.0 / h, 1.0], [m, 0.0]);
    let pdraw = isect(pneed, pb);
    if [o, ydraw, xdraw, pdraw].iter().any(|r| r[2] <= r[0] || r[3] <= r[1]) {
        return out.output();
    }
    // Each pass reads the previous node's render target: texel 0 at its need rect, power of two ≥512.
    let lt = |r: [i32; 4]| [r[0], r[1]];
    let local = |r: [i32; 4], at: [i32; 4]| [r[0] - at[0], r[1] - at[1], r[2] - at[0], r[3] - at[1]];
    let atlas = |r: [i32; 4]| [r[2] - r[0], r[3] - r[1]].map(|e| (e.max(1) as u32).next_power_of_two().max(512) as f32);
    let resampled;
    let (pre, porg, psize) = if s < 1.0 {
        let mut p = buf(pdraw);
        resample(src, &mut p, s, msl, Some((pdraw, lt(pneed))));
        resampled = p;
        (&resampled, lt(pneed), atlas(pneed))
    } else {
        (src, [src.ox, src.oy], [src.w as f32, src.h as f32])
    };
    // Without a pre-downscale the Shadow's X pass reads the alpha alone (WARP's single-channel sampler) and multiplies
    // the colour in; after one it samples all four channels of the pre's broadcast alpha (DXBC of both).
    let mut xp = buf(xdraw);
    xp.tint = tint.filter(|_| s >= 1.0);
    let xpl = PassPlanes::new(lt(xneed), local(xdraw, xneed), xdraw.map(|v| v as f32), porg, psize, taps, 0, 1.0 / h);
    blur_pass(pre, &mut xp, taps, h, false, Some(&xpl));
    let mut yp = buf(ydraw);
    let ypl = PassPlanes::new(lt(yneed), local(ydraw, yneed), ydraw.map(|v| v as f32), lt(xneed), atlas(xneed), taps, 1, 1.0 / h);
    blur_pass(&xp, &mut yp, taps, h, true, Some(&ypl));
    // The upscale draws the visible output at its place in the draw's viewport through Direct2D's
    // texel-space transform vertex shader: `u = (k·a + c)/k²` (the adjugate inverse, `c = −d·k`), texcoord
    // `(u − a)/size + (a − org)/size`.
    let at = req.map_or(lt(o), |q| q.origin);
    let size = atlas(yneed);
    let vs = |_: usize, ax: usize, a: f32| {
        let u = (k * a + -(d * k)) / (k * k);
        (u - a) * (1.0 / size[ax]) + (a - yneed[ax] as f32) * (1.0 / size[ax])
    };
    let upl = PassPlanes::with_attr(at, local(o, [at[0], at[1], 0, 0]), o.map(|v| v as f32), lt(yneed), size, 1, vs, false);
    upscale(&yp, &mut out, s * h, h, 0.0, Some(&upl));
    if let Ok(d) = std::env::var("D2D_BLUR_NODES") {
        let dump = |name: &str, b: &Buf| {
            let mut o = format!("{} {} {} {}\n", b.w, b.h, b.ox, b.oy).into_bytes();
            o.extend_from_slice(&b.px);
            let _ = std::fs::write(format!("{d}/{name}_softs1_{}x{}.bin", src.w, src.h), o);
        };
        dump("pre", pre);
        dump("xblur", &xp);
        dump("yblur", &yp);
        dump("out", &out);
    }
    out.output()
}

/// The blur's optimization transform graph, device-anchored so it serves both
/// borders: pre-downscale by `s` (MSL/LINEAR, when s<1), a separable bilinear-tap-pair blur subsampling
/// by `halving` per pass, then post-upscale by `1/(s·halving)` (LINEAR, when sh<1), with an 8bpc
/// premultiplied store between every node. `out_l/out_t` is the output's top-left device index and
/// `ow/oh` its size — HARD uses `[0,0,iw,ih]`; SOFT (full-res only, here) the input inflated by the
/// kernel reach. Every node's texel grid stays anchored at scaled device-0 (the `dx=0` matrices), so
/// the SOFT growth is just a negative output origin with transparent fetches outside the input.
#[allow(clippy::too_many_arguments)]
fn blur_graph(
    src: &Buf,
    sigma: f32,
    s: f32,
    halving: f32,
    support: f32,
    msl: bool,
    hard: bool,
    out_l: i32,
    out_t: i32,
    ow: i32,
    oh: i32,
    req: Option<Request>,
    tint: Option<[f32; 4]>,
    cells: Option<&BorderCells>,
) -> Output {
    let taps = kernel_taps(sigma * s, support);
    let sh = s * halving;

    if sh >= 1.0 {
        // Full-res (s==1, halving==1): X-blur then Y-blur produce the output directly (no up/downscale).
        // `out_l/out_t`+`ow/oh` carry the SOFT growth; HARD passes `[0,0,iw,ih]`. HARD samples its input through a mirror
        // border, and its X pass covers the Y taps' reach past the output, the mirrored rows (VM DDI of r27 32897's σ1
        // blur: the X pass drawn over rows -3..18 of a 15-row bitmap, rows -1, -2, -3 being rows 0, 1, 2, and its last
        // columns reading the input mirrored), so the Y pass reads that pass mirrored rather than clamped.
        let mirrored;
        let src = if hard {
            mirrored = Buf { w: src.w, h: src.h, ox: src.ox, oy: src.oy, hard, px: src.px.clone(), tint: src.tint, mirror: true };
            &mirrored
        } else {
            src
        };
        let mut xp = Buf::new(ow, oh, out_l, out_t, hard);
        xp.tint = tint;
        let xpl = PassPlanes::x_pass(src, &taps, [out_l, out_t, ow], req.map(|q| q.rect));
        blur_pass(src, &mut xp, &taps, 1.0, false, xpl.as_ref());
        if let (Some(c), Some(x)) = (cells, xpl.as_ref()) {
            return blur_cells(&xp, x, &taps, c, hard).output();
        }
        xp.mirror = hard;
        let mut yp = Buf::new(ow, oh, out_l, out_t, hard);
        let ypl = xpl.as_ref().zip(req).and_then(|(x, req)| PassPlanes::y_pass(x, &taps, [out_l, out_t, out_l + ow, out_t + oh], req));
        blur_pass(&xp, &mut yp, &taps, 1.0, true, ypl.as_ref());
        if let Ok(d) = std::env::var("D2D_BLUR_NODES") {
            let dump = |name: &str, b: &Buf| {
                let mut o = format!("{} {} {} {}\n", b.w, b.h, b.ox, b.oy).into_bytes();
                o.extend_from_slice(&b.px);
                let _ = std::fs::write(format!("{d}/{name}_fr_{}x{}.bin", src.w, src.h), o);
            };
            dump("xblur", &xp);
            dump("yblur", &yp);
        }
        return yp.output();
    }

    // Downscale path (halving==0.5): pre ×s → X/Y blur subsampling by `halving` → upscale. SOFT s<1 is
    // [`soft_downscaled`]; SOFT s==1 is the halving grow below; only HARD falls through to the origin-0
    // input-rect grid (`out_l==out_t==0, ow==iw`).
    // A SOFT halving blur asked for a rect lays its passes out by the request, as the pre-downscaled one does
    // (its last pass is the draw's own quad: an image brush's fill over a crop, sc104).
    if !hard && (s < 1.0 || req.is_some()) {
        return soft_downscaled(src, &taps, s, halving, msl, req, tint);
    }
    let (iw, ih) = (src.w, src.h);
    let hi = |v: i32, f: f32| libm::ceilf(v as f32 * f).max(1.0) as i32;
    // The pre-downscale shrinks each axis to round(n·s) texels (ties to even), as native does — NOT
    // ceil, which leaves an extra partial texel when n·s is non-integer and shifts the whole blur.
    let shrink = |v: i32, f: f32| (v as f32 * f).round_ties_even().max(1.0) as i32;
    let pre_owned;
    let pre: &Buf = if s < 1.0 {
        let mut p = Buf::new(shrink(iw, s), shrink(ih, s), 0, 0, hard);
        resample(src, &mut p, s, msl, None);
        pre_owned = p;
        &pre_owned
    } else {
        src
    };
    // Native's device-anchored halving geometry (decoded via the mid-EndDraw copy-out): the X-blur
    // renders an APRON'd intermediate whose pass-through (Y) axis MIRRORs (reflects) mod the input
    // height — the small intermediate reflects, replicating the content into the apron bands the Y-blur
    // integrates (fixing the edge under-spread the old tight+clamp path produced, with no gradient
    // pollution). `r` = kernel reach (apron); blurred-axis centre = `idx/halving + 0.5`, pass-through-Y
    // centre = `jy − r + 0.5` (reflected).
    let (pw, ph) = (pre.w, pre.h);
    let r = libm::ceilf(sigma * s * support).max(1.0) as i32;
    if !hard {
        // SOFT device-anchored halving: the output grows by the kernel reach `r` (origin −r, transparent
        // outside), and the blurred axis additionally grows a half-res apron `ap = ⌊r·halving⌋` on the near
        // side — native's X-blur intermediate is the tight blur shifted by exactly `ap` (verified byte-exact
        // vs the copy-out `[1,8,29,70,114,125,94,48,16,4]`). Same tap-pair kernel; MIRROR → transparent
        // (`pre.fetch`). Each grown device pixel centre then maps exactly onto B's half-res grid at texel
        // `(device_index)·sh + ap`, so the upscale is a pure bilinear with no half-texel fudge.
        // (This branch is `s==1`, `halving<1`; the `s<1` downscale is the grown-apron branch above.)
        // The intermediates reach two texels past the kernel so the output's tail samples real data.
        let rb = r + 2;
        let ap = libm::floorf(rb as f32 * halving) as i32;
        // X-blur: blurred X half-res (grown by `ap`), pass-through Y full-res (grown by `r`), transparent.
        let (xw, xh) = (hi(iw + 2 * rb, halving), ih + 2 * rb);
        let mut xp = Buf::new(xw, xh, 0, 0, hard);
        xp.tint = tint;
        for jy in 0..xh {
            let cy = (jy - rb) as f32 + 0.5;
            for jx in 0..xw {
                let cx = (jx - ap) as f32 / halving + 0.5;
                let mut acc = [0.0f32; 4];
                for &(off, wt) in &taps {
                    for sgn in [off, -off] {
                        let v = crate::effects::sample::bilin_v128(cx + sgn, cy, |x, y| pre.fetch(x, y));
                        for (k, a) in acc.iter_mut().enumerate() {
                            *a += wt * crate::sampling::norm_v128(v[k]);
                        }
                    }
                }
                xp.put(jx, jy, acc);
            }
        }
        // Y-blur: blurred Y half-res (grown by `ap`), pass-through X, reads A, transparent.
        let (yw, yh) = (xw, hi(ih + 2 * rb, halving));
        let mut yp = Buf::new(yw, yh, 0, 0, hard);
        for jy in 0..yh {
            let cy = (jy - ap) as f32 / halving + rb as f32 + 0.5;
            for jx in 0..yw {
                let cx = jx as f32 + 0.5;
                let mut acc = [0.0f32; 4];
                for &(off, wt) in &taps {
                    for sgn in [off, -off] {
                        let v = crate::effects::sample::bilin_v128(cx, cy + sgn, |x, y| xp.fetch(x, y));
                        for (k, a) in acc.iter_mut().enumerate() {
                            *a += wt * crate::sampling::norm_v128(v[k]);
                        }
                    }
                }
                yp.put(jx, jy, acc);
            }
        }
        // Upscale over native's output bounds.
        let m = taps.last().map_or(0.0, |t| t.0);
        let ((ol, ow2), (ot, oh2)) = (soft_bounds(iw, s, halving, msl, m), soft_bounds(ih, s, halving, msl, m));
        let mut out = Buf::new(ow2, oh2, ol, ot, hard);
        upscale(&yp, &mut out, sh, halving, ap as f32, None); // SOFT upscale truncates toward zero
        if let Ok(d) = std::env::var("D2D_BLUR_NODES") {
            let dump = |name: &str, b: &Buf| {
                let mut o = format!("{} {} {} {}\n", b.w, b.h, b.ox, b.oy).into_bytes();
                o.extend_from_slice(&b.px);
                let _ = std::fs::write(format!("{d}/{name}_softs1_{}x{}_s{:.2}.bin", iw, ih, sigma), o);
            };
            dump("pre", pre);
            dump("xblur", &xp);
            dump("yblur", &yp);
            dump("out", &out);
        }
        return out.output();
    }
    // HARD pre-downscaled blur = native's device-anchored FIVE-node graph: a separate downscale (→ tiny
    // `pre`) THEN a separate resample node mirror-expanding `pre` to RS×RS, conv-X, conv-Y (all via WARP's
    // `setup_triangle_attr` texcoord plane + floor-Q8 + integer bilinear), then the shared `upscale`
    // (native-correct). Byte-exact end-to-end vs native on σ=30/50/100 (copy-draw01/02/03/04 node-verified;
    // resample + conv-X core byte-exact, the apron col0 and conv-Y carry a ½-LSB the upscale averages out).
    //
    // Gated on `s < 0.5`, NOT `halving<1`: native's downscale optimisation reduces by 2× mip steps only
    // while the needed scale is below 0.5, which is exactly when it materialises the separate downscaled
    // `pre` + resample-apron node this co-port models (σ=100 s=0.1 draw-00 → 2×2, draw-01 → 55×55; σ=30
    // s=0.22 likewise). At `s ≥ 0.5` native fuses the downscale and the mirror-apron into ONE draw (no
    // separate resample node — verified: σ=8 s=0.571 capture is 4 draws, draw-00 reads the 19×16 input and
    // writes the 37×35 apron directly), so those keep the inline fused path below, which is byte-exact
    // there (the suite blur-hard-*-s5/s8 scenes: σ=5 s=0.73, σ=8 s=0.57 — both halving=0.5, so the old
    // `halving<1` gate wrongly captured them). `s < 0.5` ⟹ σ > 10 ⟹ halving==0.5 always; HARD s==1 and the
    // moderate 0.5 ≤ s < 1 band both take the inline path. (Verified via the full suite map: with the
    // co-port widened to all `s<1`, the ONLY HARD-downscale scenes reaching it are the σ=5/8 blur-hard-*
    // probes — s∈{0.571,0.727}, which it renders wrong by maxd 30–77 — and they all have s>0.5; no suite
    // scene reaches it with s<0.5, so this gate is a zero-regression suite-wide. The σ=30/50/100 fuzz
    // family that the co-port fixes all have s≤0.22<0.5.)
    if s < 0.5 {
        use crate::warp::plane::setup_triangle_attr as stria;
        let ap = r + 2;
        let ub = (ap - 1) as f32; // conv output centre = ub + idx + 0.5 (native tc u∈[ap-1, ap-1+cw])
        let (rw, rh) = (2 * ap + pw - 1, 2 * ap + ph - 1);
        let mut res = Buf::new(rw, rh, 0, 0, true);
        for jy in 0..rh {
            for jx in 0..rw {
                // pre texel 0 maps to resample texel `ap` (native draw-01 f45 [-ap, ap+1]).
                let v = pre.fetch_mir(jx - ap, jy - ap);
                let i = ((jy * rw + jx) * 4) as usize;
                res.px[i..i + 4].copy_from_slice(&[v[2] as u8, v[1] as u8, v[0] as u8, v[3] as u8]);
            }
        }
        // one separable pass over `src`, blurring `axis` (0=X,1=Y) centred at ub, passing the other axis.
        let pass = |src: &Buf, cw: i32, ch: i32, axis: usize| -> Buf {
            let atlas = (src.w.max(src.h).max(1) as u32).next_power_of_two().max(512) as f32;
            let (ql, qt, qr, qb) = (0i64, 0i64, cw as i64, ch as i64);
            let corners = [[(ql, qt), (qr, qb), (qr, qt)], [(ql, qb), (ql, qt), (qr, qb)]];
            let scr: [([i64; 3], [i64; 3]); 2] = core::array::from_fn(|k| (corners[k].map(|c| c.0 * 256), corners[k].map(|c| c.1 * 256)));
            let plane = |blur_off: Option<f32>| -> [[f32; 3]; 2] {
                core::array::from_fn(|k| stria(scr[k].0, scr[k].1, corners[k].map(|c| match blur_off {
                    Some(off) => (ub + (if axis == 0 { c.0 } else { c.1 }) as f32 + off) / atlas, // blurred axis
                    None => (if axis == 0 { c.1 } else { c.0 }) as f32 / atlas,                   // pass-through axis
                })))
            };
            let passpl = plane(None);
            let blurpls: Vec<([[f32; 3]; 2], f32)> = taps.iter().flat_map(|&(o, w)| [(plane(Some(o)), w), (plane(Some(-o)), w)]).collect();
            let q8 = |u: f32| {
                let q = ((u.clamp(-10.0, 10.0) * atlas - 0.5) * 256.0).floor() as i32;
                (q >> 8, q & 0xff)
            };
            let mut dst = Buf::new(cw, ch, 0, 0, true);
            for jy in 0..ch {
                for jx in 0..cw {
                    let (p0, p1) = (jx as f32 + 0.5, jy as f32 + 0.5);
                    let k = usize::from(p1 * cw as f32 > p0 * ch as f32);
                    let (pt, pw_) = q8(passpl[k][0] * p0 + (passpl[k][1] * p1 + passpl[k][2]));
                    let mut acc = [0.0f32; 4];
                    for (pl, wt) in &blurpls {
                        let (bt, bw) = q8(pl[k][0] * p0 + (pl[k][1] * p1 + pl[k][2]));
                        let (x0, wx, y0, wy) = if axis == 0 { (bt, bw, pt, pw_) } else { (pt, pw_, bt, bw) };
                        let v = crate::effects::sample::bilin_v128_at(x0, wx, y0, wy, |x, y| src.fetch(x, y));
                        for (c, a) in acc.iter_mut().enumerate() {
                            *a += wt * crate::sampling::norm_v128(v[c]);
                        }
                    }
                    dst.put(jx, jy, acc);
                }
            }
            dst
        };
        let (cw, ch) = (hi(pw, halving) + 2, hi(ph, halving) + 2);
        let xp2 = pass(&res, cw, rh, 0);
        let yp2 = pass(&xp2, cw, ch, 1);
        let mut out = Buf::new(iw, ih, 0, 0, true);
        upscale(&yp2, &mut out, sh, halving, 0.0, None);
        if let Ok(d) = std::env::var("D2D_BLUR_NODES") {
            for (nm, b) in [("resample_v2", &res), ("xblur_v2", &xp2), ("yblur_v2", &yp2), ("upscale_v2", &out)] {
                let mut o = format!("{} {} {} {}\n", b.w, b.h, b.ox, b.oy).into_bytes();
                o.extend_from_slice(&b.px);
                let _ = std::fs::write(format!("{d}/{nm}.bin"), o);
            }
        }
        return out.output();
    }
    // X-blur: apron'd (xw × xh), pass-through Y reflects mod ph. A leading subsampled-X apron texel
    // (origin −1) is computed too — the upscale's edge bilinear samples just left of texel 0, and native
    // stores that blurred apron rather than clamping to the core edge.
    let (xw, xh) = (hi(pw, halving) + 2, ph + 2 * r + 1);
    let mut xp = Buf::new(xw, xh, -1, 0, hard);
    for jy in 0..xh {
        let cy = (jy - r) as f32 + 0.5;
        for jx in 0..xw {
            let cx = (jx - 1) as f32 / halving + 0.5;
            let mut acc = [0.0f32; 4];
            for &(off, wt) in &taps {
                for sgn in [off, -off] {
                    let v = crate::effects::sample::bilin_v128(cx + sgn, cy, |x, y| pre.fetch_mir(x, y));
                    for (k, a) in acc.iter_mut().enumerate() {
                        *a += wt * crate::sampling::norm_v128(v[k]);
                    }
                }
            }
            xp.put(jx, jy, acc);
        }
    }
    // Y-blur: blurred Y reads the apron'd A (clamp), pass-through X. Centre = jy/halving + r + 0.5.
    // Carries xp's leading-X apron (origin −1) and computes a leading subsampled-Y apron row, for the
    // upscale's edge bilinear (same reason as the X pass).
    let (yw, yh) = (xw, hi(ph, halving) + 2);
    let mut yp = Buf::new(yw, yh, -1, -1, hard);
    for jy in 0..yh {
        let cy = (jy - 1) as f32 / halving + r as f32 + 0.5;
        for jx in 0..yw {
            let cx = (jx - 1) as f32 + 0.5;
            let mut acc = [0.0f32; 4];
            for &(off, wt) in &taps {
                for sgn in [off, -off] {
                    let v = crate::effects::sample::bilin_v128(cx, cy + sgn, |x, y| xp.fetch_mir(x, y));
                    for (k, a) in acc.iter_mut().enumerate() {
                        *a += wt * crate::sampling::norm_v128(v[k]);
                    }
                }
            }
            yp.put(jx, jy, acc);
        }
    }
    let mut out = Buf::new(iw, ih, 0, 0, hard);
    upscale(&yp, &mut out, sh, halving, 0.0, None); // upscale truncates toward zero — both borders
    if let Ok(d) = std::env::var("D2D_BLUR_NODES") {
        let dump = |name: &str, b: &Buf| {
            let mut o = format!("{} {}\n", b.w, b.h).into_bytes();
            o.extend_from_slice(&b.px);
            let _ = std::fs::write(format!("{d}/{name}_{iw}x{ih}.bin"), o);
        };
        dump("xblur", &xp);
        dump("yblur", &yp);
        dump("upscale", &out);
    }
    out.output()
}

/// Separable Gaussian weights for `sigma` (px): `exp(−k²/2σ²)` over ±⌈3σ⌉, normalised to sum 1.
pub(super) fn gaussian_weights(sigma: f32) -> (Vec<f32>, i32) {
    let r = (sigma * 3.0).ceil().max(1.0) as i32;
    let mut w: Vec<f32> = (-r..=r).map(|k| libm::expf(-((k * k) as f32) / (2.0 * sigma * sigma))).collect();
    let sum: f32 = w.iter().sum();
    for x in &mut w {
        *x /= sum;
    }
    (w, r)
}

/// Separable Gaussian blur of premultiplied colour, through the native optimization transform
/// graph ([`blur_graph`]): the bilinear-tap-pair kernel (byte-verified against the native constant
/// buffer) with an 8bpc premultiplied store between every node. HARD runs the full
/// optimization (pre-downscale by `s` → blur subsampling by `halving` → LINEAR upscale) and is
/// byte-exact. SOFT grows the output by the kernel radius `r0=⌈σ·support⌉` (transparent outside) and
/// runs the SAME device-anchored graph on both the `s==1` halving branch and the `s<1` pre-downscale
/// branch (the X/Y blur grown to the full output apron so the Gaussian tail is computed, + the decoded
/// `yp_texel = sh·d + (1+sh)` upscale; see [`blur_graph`]). The large-σ end-to-end residual is the ±1
/// inter-node rounding floor. `mode` is `D2D1_GAUSSIAN_BLUR_OPTIMIZATION`.
/// HighlightsAndShadows with the local-contrast mask — byte-exact vs native.
///
/// When any amount is active, native builds a graph: unpremultiply (ps4) → edge-detect (ps0) →
/// separable Gaussian blur (ps2, H then V) → tone map (ps6). The mask carries the blurred straight
/// input in rgb and the blurred edge strength in alpha; ps6 reads the ORIGINAL premultiplied input
/// (t0) + the mask (t1) ([`color::highlights_shadows`]). `MaskBlurRadius` `r` (default 1.25, clamp
/// `[0,10]`) drives the blur σ (`== r`) AND the edge scale (`1.0995574·r²`); `r == 0` ⇒ no mask
/// (smooth tone map). All intermediates are 8bpc B8G8R8A8_UNORM (no >8bpc stage); samplers are WARP's
/// float-family (`norm_v128`, byte/255 exact). The blur REUSES the GaussianBlur engine (`kernel_taps`
/// at σ=r, QUALITY support 3.0, `blur_pass` with the f313598 round-in-bounds rule); clamp border.
pub fn highlights_shadows(input: &Bitmap, cx: f32, cy: f32, cz: f32, radius: f32, linear: bool) -> Output {
    use crate::effects::builtin::color;
    use crate::sampling::norm_v128;
    let (w, h) = (input.width as i32, input.height as i32);
    let nf = |b: u8| norm_v128(b as i32 * 128); // WARP UNORM8 → f32 (byte/255, exact two-term)
    // ps6's t0 is the ORIGINAL premultiplied input: [R,G,B,A] f32 of texel (x,y).
    let inp_px = |x: i32, y: i32| -> [f32; 4] {
        let i = ((y * w + x) * 4) as usize;
        let p = &input.premul_bgra;
        [nf(p[i + 2]), nf(p[i + 1]), nf(p[i]), nf(p[i + 3])]
    };
    let mask = hs_mask(input, radius);
    // ps6 tone map: original premultiplied input + the mask (1:1), per pixel.
    let mut out = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let m = mask.as_ref().map(|mk| {
                let c = mk.fetch(x, y);
                [nf(c[0] as u8), nf(c[1] as u8), nf(c[2] as u8), nf(c[3] as u8)]
            });
            let o = color::highlights_shadows(inp_px(x, y), cx, cy, cz, m, linear);
            let i = ((y * w + x) * 4) as usize;
            out[i..i + 4].copy_from_slice(&pack_premul(o));
        }
    }
    Output::new(input.width, input.height, 0, 0, out)
}

/// The HighlightsShadows local-contrast mask (`None` when `MaskBlurRadius` 0). Native's order (proven
/// by per-stage DDI capture): ps4 unpremultiply → ps2 separable Gaussian (σ = radius, H then V) → ps0
/// edge-detect. The edge runs LAST, on the BLURRED rgb; the mask is rgb = blurred rgb, alpha = the
/// edge strength of the blurred rgb (NOT re-blurred). All 8bpc UNORM, clamp border; the blur reuses
/// the GaussianBlur `kernel_taps`/`blur_pass` engine (f313598 round-in-bounds).
fn hs_mask(input: &Bitmap, radius: f32) -> Option<Buf> {
    use crate::effects::builtin::color;
    use crate::sampling::norm_v128;
    let (w, h) = (input.width as i32, input.height as i32);
    let nf = |b: u8| norm_v128(b as i32 * 128);
    let luma = |r: f32, g: f32, b: f32| 0.21266 * r + 0.71515 * g + 0.07219 * b; // Rec.709 (ps0)
    let radius = radius.clamp(0.0, 10.0);
    if radius <= f32::EPSILON {
        return None;
    }
    // ps4 unpremultiply → an 8bpc straight-alpha buffer (clamp border).
    let mut straight = Buf::new(w, h, 0, 0, true);
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let p = &input.premul_bgra;
            let (b, g, r, a) = (nf(p[i]), nf(p[i + 1]), nf(p[i + 2]), nf(p[i + 3]));
            let s = if a > 0.0 { [r / a, g / a, b / a, a] } else { [0.0, 0.0, 0.0, a] };
            straight.put(x, y, s);
        }
    }
    // ps2 separable Gaussian (σ = radius), H then V — blurs the straight rgb (alpha stays constant).
    // `force_floor`: HighlightsAndShadows' blur floors the Q8 weight on both passes (no GaussianBlur +1-LSB round).
    let blurred = if radius > 1.5 {
        let bm = Output::new(w as u32, h as u32, 0, 0, straight.px.clone()).bitmap;
        Buf::from_bitmap(&gaussian_blur_extended(&bm, radius, 1, [2,2], [0,0,w,h], true).bitmap, true)
    } else {
        let taps = kernel_taps(radius, 2.5);
        let mut tmp = Buf::new(w, h, 0, 0, true);
        blur_pass(&straight, &mut tmp, &taps, 1.0, false, None);
        let mut blurred = Buf::new(w, h, 0, 0, true);
        blur_pass(&tmp, &mut blurred, &taps, 1.0, true, None);
        blurred
    };
    // ps0 edge-detect ON THE BLURRED rgb: mask rgb = blurred(center) [exact 8bpc passthrough], mask
    // alpha = (ΔLx² + ΔLy²)·scale on the blurred luma of the four ±1-texel neighbours (clamped).
    let scale = color::highlights_shadows_edge_scale(radius);
    let bl = |x: i32, y: i32| {
        let c = blurred.fetch(x, y); // clamped [R,G,B,A] bytes (i32)
        luma(nf(c[0] as u8), nf(c[1] as u8), nf(c[2] as u8))
    };
    let mut mask = Buf::new(w, h, 0, 0, true);
    for y in 0..h {
        for x in 0..w {
            let dyl = bl(x, y + 1) - bl(x, y - 1);
            let dxl = bl(x + 1, y) - bl(x - 1, y);
            let e = (dxl * dxl + dyl * dyl) * scale;
            let i = ((y * w + x) * 4) as usize;
            mask.px[i] = blurred.px[i]; // B (blurred, passthrough)
            mask.px[i + 1] = blurred.px[i + 1]; // G
            mask.px[i + 2] = blurred.px[i + 2]; // R
            mask.px[i + 3] = unorm8(e); // A = edge strength of the blurred rgb
        }
    }
    Some(mask)
}

pub fn gaussian_blur(input: &Bitmap, sigma: f32, mode: u32, hard: bool) -> Output {
    gaussian_blur_in(input, sigma, mode, hard, None, None, None)
}

/// [`gaussian_blur`] realizing only `req` (input-relative `[l, t, r, b]`, the draw's visible region):
/// native's passes cover just that, which moves WARP's texcoord planes ([`PassPlanes`]).
/// `tint`: the Shadow's premultiplied colour, multiplied into the last pass (see [`Buf::tint`]).
pub(crate) fn gaussian_blur_in(input: &Bitmap, sigma: f32, mode: u32, hard: bool, req: Option<Request>, tint: Option<[f32; 4]>, cells: Option<&BorderCells>) -> Output {
    if sigma <= 0.0 {
        let mut px = input.premul_bgra.clone();
        // A zero-radius Shadow is its colour matrix alone: alpha times the premultiplied colour.
        if let Some(pc) = tint {
            for p in px.chunks_exact_mut(4) {
                let a = p[3] as f32 / 255.0;
                p.copy_from_slice(&pack_premul(pc.map(|v| v * a)));
            }
        }
        return Output::new(input.width, input.height, 0, 0, px);
    }
    // Clamp a huge/non-finite stdDev before any sigma-scaled sizing: the kernel-tap count and grown
    // dimensions scale with sigma, so an inf/1e9/3.4e38 stdDev (fuzz r19 24664/24872, via the inf-accepting
    // effect properties) overflows the tap Vec (capacity overflow) or the grown buffer. Native clamps the
    // blur to a bounded kernel and renders a near-uniform result; cap sigma so everything stays
    // renderable. Normal sigma (≤~10) is unaffected; the exact huge-sigma output is a deferred parity class.
    let sigma = sigma.min(256.0);
    let (k, c, thr, support, msl) = blur_params(mode);
    let s = (k / (sigma + c)).clamp(0.1, 1.0);
    let halving = if sigma > thr { 0.5 } else { 1.0 };
    // Both borders now run the device-anchored graph: s==1 SOFT is the grown-origin halving branch, and
    // s<1 SOFT uses the pre-downscale apron and X/Y blur, followed by
    // `yp_texel=(d+0.5)*sh - tpost + 1` (decoded from the blur/upscale VSs).
    let (iw, ih) = (input.width as i32, input.height as i32);
    // SOFT output grows by the KERNEL REACH (furthest texel the bilinear tap-pairs touch = 2n−1 for n
    // pairs), not the Gaussian radius ⌈σ·support⌉ — native's σ0.6 blur output is 10×10 (grown 1, n=1),
    // captured from native. (Odd radii like σ2.5's r=7 give 2n−1=7, unchanged.)
    let r0 = if hard {
        0
    } else {
        let r = libm::ceilf(sigma * support).max(1.0);
        // Clamp the grown half-size: a huge/non-finite blur stdDev (fuzz r19 24664 Shadow 3.4e38, 24872
        // DirectionalBlur 1e9) makes `r` overflow to +inf, so `2*r0` would overflow i32 and the grown
        // buffer (`iw+2*r0`) would be unrenderable (OOM/panic). Native clamps the blur to a bounded
        // kernel and renders a near-uniform result; cap r0 so the dimensions stay finite and renderable.
        // Normal sigma (≤~10 → r0≈30) is far below the cap, so unaffected; the exact huge-sigma output is
        // a deferred parity class (these degenerate stdDev seeds then diff, not crash).
        (2.0 * libm::ceilf(r * 0.5) - 1.0).clamp(1.0, 512.0) as i32
    };
    let src = Buf::from_bitmap(input, hard);
    blur_graph(&src, sigma, s, halving, support, msl, hard, -r0, -r0, iw + 2 * r0, ih + 2 * r0, req, tint, cells)
}

/// Native's device-anchored blur graph for an extended source (Direct2D 10.0.26100.8875 DDI capture).
pub fn gaussian_blur_extended(input: &Bitmap, sigma: f32, mode: u32, extend: [u32; 2], rect: [i32; 4], mirror_scaled_edges: bool) -> Output {
    use crate::sampling::norm_v128;
    if input.width == 0 || input.height == 0 || rect[2] <= rect[0] || rect[3] <= rect[1] {
        return Output::new(0, 0, rect[0], rect[1], Vec::new());
    }
    let (k, c, thr, support, msl) = blur_params(mode);
    let s = (k / (sigma + c)).clamp(0.1, 1.0);
    let half = if sigma > thr { 0.5 } else { 1.0 };
    let sh = s * half;
    let taps = kernel_taps(sigma * s, support);
    let fetch = |x: i32, y: i32| {
        let ext = |v: i32, n: i32, m: u32| match m {
            1 => v.rem_euclid(n),
            2 => { let p = v.rem_euclid(n * 2); if p >= n { n * 2 - p - 1 } else { p } },
            _ => v.clamp(0, n - 1),
        };
        let x = ext(x, input.width as i32, extend[0]);
        let y = ext(y, input.height as i32, extend[1]);
        let i = ((y as u32 * input.width + x as u32) * 4) as usize;
        let p = &input.premul_bgra[i..i + 4];
        [p[2] as i32, p[1] as i32, p[0] as i32, p[3] as i32]
    };
    if sigma <= 0.0 {
        let mut out = Buf::new(rect[2] - rect[0], rect[3] - rect[1], rect[0], rect[1], false);
        for y in 0..out.h {
            for x in 0..out.w {
                out.put(x, y, fetch(out.ox + x, out.oy + y).map(|v| v as f32 / 255.0));
            }
        }
        return out.output();
    }
    let shift = 0.5 - half * 0.5;
    let lo = [rect[0], rect[1]].map(|v| (v as f32 * sh + shift).floor() as i32 - 1);
    let hi = [rect[2], rect[3]].map(|v| (v as f32 * sh + shift).ceil() as i32);
    let reach = taps.last().unwrap().0;
    let plo = lo.map(|v| (v as f32 / half - reach).floor() as i32);
    let phi = hi.map(|v| (((v - 1) as f32 / half) + reach).floor() as i32 + 2);
    let mut pre = Buf::new(phi[0] - plo[0], phi[1] - plo[1], plo[0], plo[1], false);
    let offsets: &[[f32; 2]] = if msl && s < 1.0 { &[[0.375,-0.125],[-0.125,-0.375],[-0.375,0.125],[0.125,0.375]] } else { &[[0.0,0.0]] };
    let positions = [[plo[0] as f32,plo[1] as f32],[phi[0] as f32,plo[1] as f32],[phi[0] as f32,phi[1] as f32],[plo[0] as f32,phi[1] as f32]];
    let planes: Vec<_> = offsets.iter().map(|off| {
        let attrs = positions.map(|p| core::array::from_fn::<_,2,_>(|c| {
            let mapped = ((p[c] + off[c]) * s) / (s * s);
            let dim = [input.width, input.height][c] as f32;
            p[c] / dim - (p[c] - mapped) / dim
        }));
        [[0,1,2],[3,0,2]].map(|ids| core::array::from_fn::<_,2,_>(|c| {
            crate::warp::plane::setup_triangle_attr(ids.map(|i| ((positions[i][0] - plo[0] as f32) * 256.0) as i64), ids.map(|i| ((positions[i][1] - plo[1] as f32) * 256.0) as i64), ids.map(|i| attrs[i][c]))
        }))
    }).collect();
    for y in 0..pre.h { for x in 0..pre.w {
        let mut acc = [0.0;4];
        let p = [x as f32 + 0.5,y as f32 + 0.5];
        let tri = usize::from(p[1] * pre.w as f32 > p[0] * pre.h as f32);
        for (j, plane) in planes.iter().enumerate() {
            let q: [f32;2] = core::array::from_fn(|c| {
                if mirror_scaled_edges {
                    let n = (([input.width, input.height][c] as f32 * s).round_ties_even() as i32).max(1);
                    let a = [pre.ox+x,pre.oy+y][c].rem_euclid(2*n);
                    let a = if a >= n { 2*n-1-a } else { a };
                    return (a as f32 + 0.5 + offsets[j][c]) / s;
                }
                let a = plane[tri][c];
                (a[0] * p[0] + (a[1] * p[1] + a[2])) * [input.width,input.height][c] as f32
            });
            // Native WARP Border/MSL JIT interpolates Y before X (10.0.26100.8875, gauss-border capture).
            let v = crate::effects::sample::bilin_v128(q[1], q[0], |y, x| fetch(x, y));
            for c in 0..4 { acc[c] += norm_v128(v[c]) / offsets.len() as f32; }
        }
        pre.put(x,y,acc);
    }}
    let mut xp = Buf::new(hi[0]-lo[0],pre.h,lo[0],pre.oy,false);
    for y in 0..xp.h { for x in 0..xp.w {
        let cx = (xp.ox+x) as f32 / half + 0.5;
        let cy = (xp.oy+y) as f32 + 0.5;
        let mut acc = [0.0;4];
        for &(off,wt) in &taps { for off in [off,-off] {
            let v = crate::effects::sample::bilin_v128(cx+off,cy,|x,y| pre.fetch(x,y));
            for c in 0..4 { acc[c] += wt * norm_v128(v[c]); }
        }}
        xp.put(x,y,acc);
    }}
    let mut yp = Buf::new(xp.w,hi[1]-lo[1],lo[0],lo[1],false);
    for y in 0..yp.h { for x in 0..yp.w {
        let cx = (yp.ox+x) as f32 + 0.5;
        let cy = (yp.oy+y) as f32 / half + 0.5;
        let mut acc = [0.0;4];
        for &(off,wt) in &taps { for off in [off,-off] {
            let v = crate::effects::sample::bilin_v128(cx,cy+off,|x,y| xp.fetch(x,y));
            for c in 0..4 { acc[c] += wt * norm_v128(v[c]); }
        }}
        yp.put(x,y,acc);
    }}
    let mut out = Buf::new(rect[2]-rect[0],rect[3]-rect[1],rect[0],rect[1],false);
    let pos = [[rect[0],rect[1]],[rect[2],rect[1]],[rect[2],rect[3]],[rect[0],rect[3]]];
    let scale = 1.0/sh;
    let attrs = pos.map(|p| core::array::from_fn::<_,2,_>(|c| {
        let p=p[c] as f32;
        let mapped = (p*scale + (shift/sh)*scale)/(scale*scale);
        (p - [yp.ox,yp.oy][c] as f32)/512.0 - (p-mapped)/512.0
    }));
    let postplanes = [[0,1,2],[3,0,2]].map(|ids| core::array::from_fn::<_,2,_>(|c| {
        crate::warp::plane::setup_triangle_attr(ids.map(|i|((pos[i][0]-rect[0])*256) as i64),ids.map(|i|((pos[i][1]-rect[1])*256) as i64),ids.map(|i|attrs[i][c]))
    }));
    for y in 0..out.h { for x in 0..out.w {
        let p=[x as f32+0.5,y as f32+0.5];
        let tri=usize::from(p[1]*out.w as f32>p[0]*out.h as f32);
        let q=postplanes[tri].map(|a|(a[0]*p[0]+(a[1]*p[1]+a[2]))*512.0);
        let v=crate::effects::sample::bilin_v128(q[1],q[0],|y,x|yp.fetch(x+yp.ox,y+yp.oy));
        let i=((y*out.w+x)*4) as usize;
        let c=v.map(|v|(v>>7).clamp(0,255) as u8);
        out.px[i..i+4].copy_from_slice(&[c[2],c[1],c[0],c[3]]);
    }}
    if let Ok(dir) = std::env::var("D2D_BLUR_NODES") {
        for (name, b) in [("pre", &pre), ("x", &xp), ("y", &yp), ("out", &out)] {
            let mut bytes = format!("{} {} {} {}\n", b.w, b.h, b.ox, b.oy).into_bytes();
            bytes.extend_from_slice(&b.px);
            let _ = std::fs::write(format!("{dir}/extended-{name}.bin"),bytes);
        }
    }
    out.output()
}

/// Premultiplied f32 image with a logical origin — the directional-blur working buffer.
struct FImg {
    w: i32,
    h: i32,
    ox: i32,
    oy: i32,
    p: Vec<[f32; 4]>,
}
fn from_bitmap(bm: &Bitmap) -> FImg {
    let (w, h) = (bm.width as i32, bm.height as i32);
    let p = (0..w * h).map(|i| premul_at(bm, i % w, i / w, false)).collect();
    FImg { w, h, ox: 0, oy: 0, p }
}
fn pack_fimg(f: &FImg) -> Output {
    let mut out = vec![0u8; (f.w * f.h * 4) as usize];
    for (i, px) in f.p.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&pack_premul(*px));
    }
    Output::new(f.w as u32, f.h as u32, f.ox, f.oy, out)
}

/// Shadow: colorize the input to a premultiplied silhouette `premul(color)·inα` (an 8bpc
/// node), then Gaussian-blur it (always SOFT) through the same optimization graph. Default color opaque
/// black ⇒ `(0,0,0,blur(α))`. `mode` is the shadow's `D2D1_GAUSSIAN_BLUR_OPTIMIZATION`.
pub fn shadow(input: &Bitmap, sigma: f32, color: [f32; 4], mode: u32) -> Output {
    shadow_in(input, sigma, color, mode, None, false)
}

/// [`shadow`] over the requested region `req` (see [`gaussian_blur_in`]). `stored` colours the stored blur instead:
/// a Shadow read through a mirror border (a HARD DirectionalBlur's) keeps its expand plain and the border's cells
/// multiply the colour into the stored alpha as they copy it, so the colour rounds once more (VM DDI of r19 24502's
/// HARD DirectionalBlur over a Shadow: a plain expand, then the cells' `Mul o0, r0.xxxx, cb0[0]` over the needed
/// columns; a SOFT one keeps the coloured expand).
/// ponytail: only after a pre-downscale; without one the colour sits in the X pass, whose border case is unprobed.
pub(crate) fn shadow_in(input: &Bitmap, sigma: f32, color: [f32; 4], mode: u32, req: Option<Request>, stored: bool) -> Output {
    // Native's Shadow blurs the input's ALPHA (full 0..255 range), THEN multiplies the blurred
    // alpha by the premultiplied shadow colour. Colourising first (pc·a stored 8bpc before the
    // blur) quantises the small colour channels away and makes the output systematically 1 LSB
    // below native in the penumbra (deferring the colourise closes it).
    let pc = [color[0] * color[3], color[1] * color[3], color[2] * color[3], color[3]];
    let (w, h) = (input.width, input.height);
    let mut alpha = vec![0u8; (w * h * 4) as usize];
    for i in 0..(w * h) as usize {
        let a = unorm8(premul_at(input, (i as u32 % w) as i32, (i as u32 / w) as i32, false)[3]);
        alpha[i * 4..i * 4 + 4].copy_from_slice(&[a, a, a, a]);
    }
    let alpha_bm = Bitmap {
        width: w,
        height: h,
        premul_bgra: alpha,
        format: crate::abi::DXGI_FORMAT_B8G8R8A8_UNORM,
        alpha: crate::abi::D2D1_ALPHA_MODE_PREMULTIPLIED,
        dpi: (96.0, 96.0),
    };
    // The colour multiplies a pass's float result before its store: the X pass, or the expand after a
    // pre-downscale (see `Buf::tint`).
    let (k, c, ..) = blur_params(mode);
    if !stored || sigma <= 0.0 || (k / (sigma.min(256.0) + c)).clamp(0.1, 1.0) >= 1.0 {
        return gaussian_blur_in(&alpha_bm, sigma, mode, false, req, Some(pc), None);
    }
    let mut out = gaussian_blur_in(&alpha_bm, sigma, mode, false, req, None, None);
    for p in out.bitmap.premul_bgra.chunks_exact_mut(4) {
        let a = p[2] as f32 / 255.0;
        p.copy_from_slice(&pack_premul(pc.map(|v| v * a)));
    }
    out
}

/// `org` is the input's device origin. `hard` is the axis-aligned HARD graph: the pre-transform clamps
/// and keeps the transformed input rect itself (top-left rounded), and the taps read it mirrored.
fn directional_graph(input: &Bitmap, org: [i32; 2], sigma: f32, angle_deg: f32, mode: u32, clip: Option<[i32; 4]>, hard: bool) -> Output {
    use super::transform::{self as t, Border, Interp, Input};
    let (k,c,thr,support,msl) = blur_params(mode);
    let s = (k / (sigma+c)).clamp(0.1,1.0);
    let half = if sigma > thr {0.5} else {1.0};
    // Native builds both matrices as D2D1::Matrix3x2F products of D2D1MakeRotateMatrix (which reduces
    // the angle into [0, 360) first), so the inverse rotation's bits are sin/cos of 360−θ, not −sin/cos θ.
    let rotate = make_rotate_matrix(angle_deg);
    let (sn,cs) = (rotate[1],rotate[0]);
    let matrix = mul3x2(rotate,[s,0.0,0.0,1.0,0.0,0.0]);
    // An identity pre-transform is no node at all: the taps read the input itself, over its own rect.
    let identity = matrix == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let pad = if hard || identity { 0.0 } else { 0.5 + (cs.abs()+sn.abs()) / s * 0.5 };
    let (x0, y0, x1, y1) = (org[0] as f32, org[1] as f32, (org[0] + input.width as i32) as f32, (org[1] + input.height as i32) as f32);
    let corners = [[x0-pad,y0-pad],[x1+pad,y0-pad],[x1+pad,y1+pad],[x0-pad,y1+pad]];
    let mapped = corners.map(|p| [matrix[0]*p[0]+matrix[2]*p[1],matrix[1]*p[0]+matrix[3]*p[1]]);
    let round = |v: f32, up: bool| if hard || identity { (v - 0.5).ceil() as i32 } else if up { v.ceil() as i32 } else { v.floor() as i32 };
    let lo: [i32;2] = core::array::from_fn(|c| round(mapped.iter().map(|p|p[c]).fold(f32::INFINITY,f32::min), false));
    let hi: [i32;2] = core::array::from_fn(|c| round(mapped.iter().map(|p|p[c]).fold(f32::NEG_INFINITY,f32::max), true));
    let taps = kernel_taps(sigma*s,support);
    let reach = 2*taps.len() as i32-1;
    let left = (((lo[0]-reach) as f32)*half+0.5).floor() as i32;
    let right = (((hi[0]+reach) as f32)*half).ceil() as i32;
    let need_left = clip.map(|b| {
        let min = [[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]].map(|p| (cs*p[0] as f32-sn*p[1] as f32)*s*half + 0.5-half*0.5).into_iter().fold(f32::INFINITY,f32::min);
        (min - 0.5).floor() as i32
    }).unwrap_or(left).max(left);
    let pre_origin = [-need_left as f32 / half + reach as f32,-lo[1] as f32];
    let inp = Input::new(input,org[0],org[1]);
    // The pre-transform resamples with MULTI_SAMPLE_LINEAR, LINEAR at SPEED.
    let mut pre = if identity {
        Buf::from_bitmap(&pack_fimg(&from_bitmap(input)).bitmap, false)
    } else {
        let interp = if msl { Interp::MultiSampleLinear } else { Interp::Linear };
        let eval = t::affine_at(&inp,matrix,interp,if hard {Border::Hard} else {Border::Soft},pre_origin,false);
        Buf::from_bitmap(&t::to_bitmap((hi[0]-lo[0]) as u32,(hi[1]-lo[1]) as u32,lo[0] as f32,lo[1] as f32,eval),false)
    };
    (pre.ox, pre.oy) = (lo[0], lo[1]);
    let mut blurred = Buf::new(right-left,pre.h,left,pre.oy,false);
    for y in 0..blurred.h { for x in 0..blurred.w {
        let cx=(left+x) as f32/half+0.5;
        let cy=(pre.oy+y) as f32+0.5;
        let mut acc=[0.0;4];
        for &(off,wt) in &taps { for off in [off,-off] {
            let v=crate::effects::sample::bilin_v128(cx+off,cy,|x,y|if hard {pre.fetch_mir(x,y)} else {pre.fetch(x,y)});
            for c in 0..4 {acc[c]+=wt*crate::sampling::norm_v128(v[c]);}
        }}
        blurred.put(x,y,acc);
    }}
    let post=mul3x2(mul3x2([1.0,0.0,0.0,1.0,-(0.5-half*0.5),0.0],[1.0/(s*half),0.0,0.0,1.0,0.0,0.0]),make_rotate_matrix(-angle_deg));
    let corners=[[left as f32-0.5,pre.oy as f32-0.5],[right as f32+0.5,pre.oy as f32-0.5],[right as f32+0.5,(pre.oy+pre.h) as f32+0.5],[left as f32-0.5,(pre.oy+pre.h) as f32+0.5]];
    let mapped=corners.map(|p|[post[0]*p[0]+post[2]*p[1]+post[4],post[1]*p[0]+post[3]*p[1]+post[5]]);
    let olo:[i32;2]=core::array::from_fn(|c|mapped.iter().map(|p|p[c]).fold(f32::INFINITY,f32::min).floor() as i32);
    let ohi:[i32;2]=core::array::from_fn(|c|mapped.iter().map(|p|p[c]).fold(f32::NEG_INFINITY,f32::max).ceil() as i32);
    let bm=blurred.output().bitmap;
    let inp=Input::new(&bm,left,lo[1]);
    let origin=clip.map(|b|[-b[0] as f32,-b[1] as f32]).unwrap_or([-olo[0] as f32,-olo[1] as f32]);
    // The post is the draw into the caller's viewport (`clip`); WARP clips its triangles there before
    // fitting the texcoord planes, which moves their snapped vertices off the rotated corners.
    let m=post.map(f64::from);
    let eval=t::projective_rect_at(&inp,[m[0],m[1],0.0,0.0,m[2],m[3],0.0,0.0,0.0,0.0,1.0,0.0,m[4],m[5],0.0,1.0],Interp::Linear,Border::Soft,origin,clip.map(|b|b.map(|v|v as f32)));
    let output=t::to_bitmap((ohi[0]-olo[0]) as u32,(ohi[1]-olo[1]) as u32,olo[0] as f32,olo[1] as f32,eval);
    if let Ok(dir)=std::env::var("D2D_BLUR_NODES") {
        for (name,w,h,ox,oy,bytes) in [("pre",pre.w,pre.h,pre.ox,pre.oy,&pre.px),("blur",bm.width as i32,bm.height as i32,left,lo[1],&bm.premul_bgra),("out",output.width as i32,output.height as i32,olo[0],olo[1],&output.premul_bgra)] {
            let mut data=format!("{w} {h} {ox} {oy}\n").into_bytes();data.extend_from_slice(bytes);let _=std::fs::write(format!("{dir}/direction-{name}.bin"),data);
        }
    }
    Output {bitmap:output,ox:olo[0],oy:olo[1]}
}

/// What a DirectionalBlur reads of its input for output rect `v` (effect pixels), as d2d1's nodes map a need back: the
/// inverse rotation (a LINEAR matrix transform: `v`'s pixel-centre inset corners through the adjugate of its matrix over
/// the determinant, each bound snapped within 1e-4 of a whole pixel, the floor and ceiling one pixel wider), the
/// kernel at its x scale (its taps' reach, a pixel more each side), then the pre-rotation (inset none under
/// MULTI_SAMPLE_LINEAR, a half pixel under LINEAR, one pixel wider). A HARD blur reads that through a mirror border over
/// the input rect `input`. `None` where no rotation runs (an axis-aligned or a 90°/270° blur), whose need is not modelled.
/// ponytail: the need is not cut by each transform's output quad, so a viewport reaching past the blurred content asks
/// a little more; and a HARD need overhanging the input by more than it overlaps takes the whole input on that axis
/// rather than its exact mirror fold.
pub(crate) fn directional_need(sigma: f32, angle_deg: f32, mode: u32, hard: bool, v: [i32; 4], input: [i32; 4]) -> Option<[i32; 4]> {
    let a = libm::fmodf(angle_deg, 360.0);
    let a = if a < 0.0 { a + 360.0 } else { a };
    if sigma <= 0.0 || matches!(a, 0.0 | 90.0 | 180.0 | 270.0) {
        return None;
    }
    let (k, c, thr, support, msl) = blur_params(mode);
    let s = (k / (sigma + c)).clamp(0.1, 1.0);
    let half = if sigma > thr { 0.5 } else { 1.0 };
    let pre = mul3x2(make_rotate_matrix(angle_deg), [s, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let post = mul3x2(mul3x2([1.0, 0.0, 0.0, 1.0, -(0.5 - half * 0.5), 0.0], [1.0 / (s * half), 0.0, 0.0, 1.0, 0.0, 0.0]), make_rotate_matrix(-angle_deg));
    let snap = |v: f32| { let n = v.round(); if (v - n).abs() < 1e-4 { n } else { v } };
    let back = |m: [f32; 6], r: [i32; 4], inset: f32| -> [i32; 4] {
        let [a, b, c, d, e, f] = m;
        let (det, tx, ty) = ((a * d) - (c * b), (c * f) - (e * d), (e * b) - (a * f));
        let (l, t, rr, bb) = (r[0] as f32 + inset, r[1] as f32 + inset, r[2] as f32 - inset, r[3] as f32 - inset);
        let pts = [(l, t), (l, bb), (rr, bb), (rr, t)].map(|(x, y)| [((x * d + y * -c) + tx) / det, ((x * -b + y * a) + ty) / det]);
        let lo = |k: usize| (snap(pts.iter().map(|p| p[k]).fold(f32::INFINITY, f32::min)).floor() as i32).saturating_sub(1);
        let hi = |k: usize| (snap(pts.iter().map(|p| p[k]).fold(f32::NEG_INFINITY, f32::max)).ceil() as i32).saturating_add(1);
        [lo(0), lo(1), hi(0), hi(1)]
    };
    let n1 = back(post, v, 0.5);
    let reach = kernel_taps(sigma * s, support).last().map_or(0.0, |t| t.0);
    let g = 1.0 / half - 1.0;
    let round = |v: f32| (v + 0.5).floor() as i32;
    let (l, r) = (n1[0] as f32 + 0.5, n1[2] as f32 - 0.5);
    let n2 = [round((((l - 0.5) * g) + l) - reach - 1.0), n1[1], ((((((r - 0.5) * g) + r) + reach) + 1.0) - 0.5).ceil() as i32, n1[3]];
    let n3 = back(pre, n2, if msl { 0.0 } else { 0.5 });
    if !hard {
        return Some(n3);
    }
    let axis = |k: usize| -> (i32, i32) {
        let (lo, hi) = (n3[k].max(input[k]), n3[k + 2].min(input[k + 2]));
        let (n, i, span) = (n3.map(i64::from), input.map(i64::from), hi as i64 - lo as i64);
        if lo >= hi || i[k] - n[k] > span || n[k + 2] - i[k + 2] > span { (input[k], input[k + 2]) } else { (lo, hi) }
    };
    let ((l, r), (t, b)) = (axis(0), axis(1));
    Some([l, t, r, b])
}

/// `D2D1MakeRotateMatrix(angle, (0,0))` as Direct2D 10.0.26100.8875 computes it: the angle reduced into
/// `[0, 360)`, exact at the quarter turns, else the Windows CRT's `sinf`/`cosf` of `angle · 0.017453292`.
fn make_rotate_matrix(angle: f32) -> [f32; 6] {
    let mut a = libm::fmodf(angle, 360.0);
    if a < 0.0 {
        a += 360.0;
    }
    let (sn, cs) = match a {
        0.0 => (0.0, 1.0),
        90.0 => (1.0, 0.0),
        180.0 => (0.0, -1.0),
        270.0 => (-1.0, 0.0),
        _ => {
            let r = a * 0.017453292;
            (crate::crt::system_crt::sinf(r), crate::crt::system_crt::cosf(r))
        }
    };
    [cs, sn, -sn, cs, 0.0, 0.0]
}

/// `D2D1::Matrix3x2F::SetProduct(a, b)`: `a·b` in the row-vector convention, f32 as native rounds it.
fn mul3x2(a: [f32; 6], b: [f32; 6]) -> [f32; 6] {
    [a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3], a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4], a[4] * b[1] + a[5] * b[3] + b[5]]
}

/// DirectionalBlur: rotate/downsample, blur the horizontal axis, then inverse-rotate.
/// The SOFT graph follows native Direct2D 10.0.26100.8875 DDI/JIT captures. HARD retains the direct kernel.
pub fn directional_blur(input: &Bitmap, sigma: f32, angle_deg: f32, hard: bool) -> Output {
    directional_blur_with_options(input,sigma,angle_deg,1,hard,None,[0,0])
}

/// `at` is the input's device origin, where Direct2D anchors the graph's grids (`clip` and the output stay relative to
/// the input).
pub(crate) fn directional_blur_with_options(input: &Bitmap, sigma: f32, angle_deg: f32, mode: u32, hard: bool, clip: Option<[i32; 4]>, at: [i32; 2]) -> Output {
    if sigma <= 0.0 {
        return pack_fimg(&from_bitmap(input));
    }
    let graph = |inp: &Bitmap, org: [i32; 2], hard_axis: bool| {
        let add = |v: i32, k: usize| v.saturating_add(at[k]);
        let clip = clip.map(|c| [add(c[0], 0), add(c[1], 1), add(c[2], 0), add(c[3], 1)]);
        let mut o = directional_graph(inp, [add(org[0], 0), add(org[1], 1)], sigma, angle_deg, mode, clip, hard_axis);
        (o.ox, o.oy) = (o.ox.saturating_sub(at[0]), o.oy.saturating_sub(at[1]));
        o
    };
    // Native DirectionalBlur (Direct2D 10.0.26100.8875) builds its optimization transforms as: rotate and
    // pre-scale the blur axis, blur it at half resolution above the level's σ threshold, rotate back. At 90° and
    // 270° it doesn't rotate: it scales and blurs along y, which is this graph on the transposed image.
    let a = libm::fmodf(angle_deg, 360.0);
    let a = if a < 0.0 { a + 360.0 } else { a };
    if a == 90.0 || a == 270.0 {
        let out = directional_blur_with_options(&transpose(input), sigma, 0.0, mode, hard, clip.map(|c| [c[1], c[0], c[3], c[2]]), [at[1], at[0]]);
        return Output { bitmap: transpose(&out.bitmap), ox: out.oy, oy: out.ox };
    }
    if !hard {
        return graph(input, [0, 0], false);
    }
    if input.width == 0 || input.height == 0 {
        return pack_fimg(&from_bitmap(input));
    }
    // HARD runs the same graph over the input mirrored past its edges, and clips the result to the input.
    // At 0° and 180° the pre-transform clamps (HARD) and the taps read its output mirrored. Otherwise the
    // mirrored input is cut to the input grown by the kernel's reach along each axis, ceil(|cos θ|·σ·support) by
    // ceil(|sin θ|·σ·support), and goes through the SOFT graph.
    let (iw, ih) = (input.width as i32, input.height as i32);
    let out = if a == 0.0 || a == 180.0 {
        graph(input, [0, 0], true)
    } else {
        let support = blur_params(mode).3;
        let th = (a as f64 * std::f64::consts::PI / 180.0) as f32;
        let ix = (crate::crt::system_crt::cosf(th).abs() * (sigma * support)).ceil() as i32;
        let iy = (crate::crt::system_crt::sinf(th).abs() * (sigma * support)).ceil() as i32;
        let (w, h) = (iw + 2 * ix, ih + 2 * iy);
        let refl = |v: i32, n: i32| { let p = v.rem_euclid(2 * n); if p >= n { 2 * n - 1 - p } else { p } };
        let mut px = vec![0u8; (w * h * 4) as usize];
        for y in 0..h { for x in 0..w {
            let si = ((refl(y - iy, ih) * iw + refl(x - ix, iw)) * 4) as usize;
            px[((y * w + x) * 4) as usize..][..4].copy_from_slice(&input.premul_bgra[si..si + 4]);
        }}
        let src = Bitmap { width: w as u32, height: h as u32, premul_bgra: px, format: input.format, alpha: input.alpha, dpi: input.dpi };
        graph(&src, [-ix, -iy], false)
    };
    let mut px = vec![0u8; (iw * ih * 4) as usize];
    let ow = out.bitmap.width as i32;
    for y in 0..ih { for x in 0..iw {
        let (sx, sy) = (x - out.ox, y - out.oy);
        if sx >= 0 && sy >= 0 && sx < ow && sy < out.bitmap.height as i32 {
            let si = ((sy * ow + sx) * 4) as usize;
            px[((y * iw + x) * 4) as usize..][..4].copy_from_slice(&out.bitmap.premul_bgra[si..si + 4]);
        }
    }}
    Output::new(iw as u32, ih as u32, 0, 0, px)
}

/// `bm` with x and y swapped (4-byte texels).
fn transpose(bm: &Bitmap) -> Bitmap {
    let (w, h) = (bm.width as usize, bm.height as usize);
    let mut px = vec![0u8; w * h * 4];
    for y in 0..h { for x in 0..w {
        px[(x * h + y) * 4..][..4].copy_from_slice(&bm.premul_bgra[(y * w + x) * 4..][..4]);
    }}
    Bitmap { width: h as u32, height: w as u32, premul_bgra: px, format: bm.format, alpha: bm.alpha, dpi: bm.dpi }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_table_values() {
        assert_eq!(discrete_table(2), vec![0.0, 1.0]);
        assert_eq!(discrete_table(3), vec![0.0, 0.5, 1.0]);
        assert_eq!(discrete_table(4), vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]);
        // Clamp out-of-range level counts to [2,16].
        assert_eq!(discrete_table(1).len(), 2);
        assert_eq!(discrete_table(99).len(), 16);
    }

    #[test]
    fn posterize_ramp() {
        // A 1-D grayscale ramp (0,17,…,255) at R=2,G=3,B=4 matches the hand-computed oracle values.
        let rgba: Vec<u8> = (0..16u32).flat_map(|i| [(i * 17) as u8; 3].into_iter().chain([255])).collect();
        let bm = Bitmap::from_straight_rgba(16, 1, &rgba);
        let out = posterize(&bm, 2, 3, 4);
        let px = |i: usize| &out.bitmap.premul_bgra[i * 4..i * 4 + 4];
        assert_eq!(px(4), [85, 0, 0, 255]); // gray 68: B floor(1.06)=1→85, G floor(0.8)=0, R floor(0.53)=0
        assert_eq!(px(8), [170, 128, 255, 255]); // gray 136: B→170, G floor(1.6)=1→128, R floor(1.06)=1→255
        assert_eq!(px(15), [255, 255, 255, 255]); // gray 255 clamps to the top level everywhere
    }

    // Sharpen precomputes the straight (unpremultiplied) + UNORM8 input once and reads it border-clamped,
    // instead of unpremultiplying per tap. Prove that table equals the per-tap unpremultiply at EVERY
    // sample position, in and out of bounds (where `straight_rgb`'s hard clamp and the table's clamp must
    // agree) — the all-inputs proof for the refactor, beyond the corpora the identity gate covers.
    #[test]
    fn sharpen_straight_table_matches_per_tap() {
        let (w, h) = (5u32, 4u32);
        let rgba: Vec<u8> = (0..(w * h) as usize)
            .flat_map(|i| [(i * 37 % 256) as u8, (i * 53 % 256) as u8, (i * 71 % 256) as u8, ((i * 23 + 30) % 256) as u8])
            .collect();
        let bm = Bitmap::from_straight_rgba(w, h, &rgba);
        let straight8: Vec<[i32; 4]> = (0..(w * h) as usize)
            .map(|i| {
                let (x, y) = ((i as u32 % w) as i32, (i as u32 / w) as i32);
                let c = straight_rgb(&bm, x, y, false);
                [unorm8(c[0]) as i32, unorm8(c[1]) as i32, unorm8(c[2]) as i32, 255]
            })
            .collect();
        let (wi, hi) = (w as i32, h as i32);
        let table = |sx: i32, sy: i32| straight8[sy.clamp(0, hi - 1) as usize * w as usize + sx.clamp(0, wi - 1) as usize];
        let per_tap = |sx: i32, sy: i32| {
            let c = straight_rgb(&bm, sx, sy, false);
            [unorm8(c[0]) as i32, unorm8(c[1]) as i32, unorm8(c[2]) as i32, 255]
        };
        for sy in -3..hi + 3 {
            for sx in -3..wi + 3 {
                assert_eq!(table(sx, sy), per_tap(sx, sy), "sx={sx} sy={sy}");
            }
        }
    }

    // Emboss precomputes the per-texel luminance once and reads it border-clamped. Prove the table
    // equals the per-neighbour `luminance` at every position, in and out of bounds.
    #[test]
    fn emboss_luminance_table_matches_per_neighbour() {
        let (w, h) = (5u32, 4u32);
        let rgba: Vec<u8> = (0..(w * h) as usize)
            .flat_map(|i| [(i * 37 % 256) as u8, (i * 53 % 256) as u8, (i * 71 % 256) as u8, ((i * 23 + 30) % 256) as u8])
            .collect();
        let bm = Bitmap::from_straight_rgba(w, h, &rgba);
        let lum: Vec<f32> = (0..(w * h) as usize)
            .map(|i| luminance(&bm, (i as u32 % w) as i32, (i as u32 / w) as i32))
            .collect();
        let (wi, hi) = (w as i32, h as i32);
        for sy in -3..hi + 3 {
            for sx in -3..wi + 3 {
                let table = lum[sy.clamp(0, hi - 1) as usize * w as usize + sx.clamp(0, wi - 1) as usize];
                assert_eq!(table, luminance(&bm, sx, sy), "sx={sx} sy={sy}");
            }
        }
    }

    // EdgeDetection precomputes the straight input once and reads it border-clamped, for either alpha
    // mode. Prove the table equals the per-neighbour `straight_rgb` at every position, in and out of bounds.
    #[test]
    fn edge_straight_table_matches_per_neighbour() {
        let (w, h) = (5u32, 4u32);
        let rgba: Vec<u8> = (0..(w * h) as usize)
            .flat_map(|i| [(i * 41 % 256) as u8, (i * 59 % 256) as u8, (i * 67 % 256) as u8, ((i * 29 + 40) % 256) as u8])
            .collect();
        let bm = Bitmap::from_straight_rgba(w, h, &rgba);
        let (wi, hi) = (w as i32, h as i32);
        for alpha_straight in [false, true] {
            let str3: Vec<[f32; 3]> = (0..(w * h) as usize)
                .map(|i| straight_rgb(&bm, (i as u32 % w) as i32, (i as u32 / w) as i32, alpha_straight))
                .collect();
            for sy in -3..hi + 3 {
                for sx in -3..wi + 3 {
                    let table = str3[sy.clamp(0, hi - 1) as usize * w as usize + sx.clamp(0, wi - 1) as usize];
                    assert_eq!(table, straight_rgb(&bm, sx, sy, alpha_straight), "a={alpha_straight} sx={sx} sy={sy}");
                }
            }
        }
    }
}

