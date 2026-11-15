//! WARP's HLSL pixel-shader linear `sample` intrinsic, as WARP runs it for a `DisplacementMap`
//! effect draw (confirmed against `fxsample-01-displacement-probe`). This is the `sample` an effect shader
//! calls on an input with a texcoord it computed in float — distinct from [`crate::sampling`], which
//! reproduces the fixed-function DrawBitmap texture filter (integer `lerp8` on a rasterized 16.16
//! texcoord). Here the texcoord arrives as a shader float and the 2×2 is combined in **f32**.
//!
//! The kernel (per axis): take the texel-space coordinate `c` (texel centres at integer+0.5),
//! `q = floor((c − 0.5)·256)` as a signed i32 (WARP: multiply by 256, floor, convert to int), the
//! lower texel is `q >> 8` (arithmetic) and the 8-bit sub-texel weight is `q & 0xff`. The four corner
//! texels are converted from UNORM8 to f32 by `byte·257/65536` (WARP's `·1/256 + ·1/65536`, ≈ 1/255
//! but not exactly), and the sample is the f32 bilinear product sum
//! `Σ corner·w`, `w ∈ {(1−fx)(1−fy), fx(1−fy), (1−fx)fy, fx·fy}`, accumulated in the kernel's order.
//!
//! Residual (≤1-LSB, bidirectional, OPEN). The displacement kernel's combine is **not**
//! the f32 product-sum above — it is WARP's **integer** stage, deterministic: u8 corners → an x-lerp
//! `col = ((b−a)·wx + (a<<8)) >> 1` (16-bit, `wx=q&0xff`, giving value·128), then a y-lerp
//! `res = top + ((wy·(bot−top)) >> 16)` (an unsigned 16-bit high multiply with sign-fix, `wy=(q&0xff)<<8`), stored
//! `byte = round(res/128)`; the float tail only does opacity/clamp/×255/round. Porting that exact form
//! reproduces the captured checker readback to **±1 on 40/100 px** (and the same 20 on `fxsample-01`)
//! — i.e. it still carries the one-step residual, so it is **not** the combine choice (f32 vs integer
//! give the identical diff set): the ≤1-LSB is in the per-pixel Q8 **weight/address** (the exact
//! integer truncation/rounding of `floor(clamp(U)·texSize·256)`), still being pinned
//! against the readback. `crate::warp::plane` (radial agent) owns the plane; this kernel owns the
//! combine+address. Same floor the DrawBitmap/blur tracks hit.

/// UNORM8 byte → f32 the way WARP's sampler does it (`b·1/256 + b·1/65536 = b·257/65536`).
#[inline]
pub fn unorm_to_f32(b: u8) -> f32 {
    b as f32 * (257.0 / 65536.0)
}

/// Texel-space coordinate `c` → (lower texel index, 8-bit sub-texel weight integer in `[0,256)`),
/// exactly as WARP's `sample` quantises it: `q = floor((c − 0.5)·256)`, texel `q >> 8`, weight
/// `q & 0xff`. Computed in f32 so the `floor` lands where WARP's does.
#[inline]
fn axis(c: f32) -> (i32, i32) {
    let q = ((c - 0.5) * 256.0).floor() as i32;
    (q >> 8, q & 0xff)
}

/// Like [`axis`] but rounds the Q8 address to nearest (ties to even) — WARP's `cvtps2dq`. Used only by
/// the built-in displacement's [`sample_linear_sel`] (keyed on the realized texture size). A custom
/// pixel-shader's input `Sample` keeps [`axis`]'s floor — the standard LINEAR sub-texel address — so a
/// displaced/dependent custom tap matches genuine Direct2D (the round shifted the Q8 weight one, +1 in
/// the steepest channel: PdnV5 Ripple B / Wavey Transform G).
#[inline]
fn axis_round(c: f32) -> (i32, i32) {
    let q = ((c - 0.5) * 256.0).round_ties_even() as i32;
    (q >> 8, q & 0xff)
}

/// [`sample_linear`] choosing the Q8 address per axis: the round [`axis_round`] (cvtps2dq) when the
/// realized texture is non-pow2 **content**, the floor [`axis`] when it is the pow2 realization atlas —
/// WARP's `Sample` keyed on the realized texture size (the built-in filters' counterpart of the custom
/// `Sample` round). `rx`/`ry` request the round per axis.
#[inline]
pub fn sample_linear_sel(cx: f32, cy: f32, rx: bool, ry: bool, fetch: impl Fn(i32, i32) -> [i32; 4]) -> [f32; 4] {
    let (x0, wx) = if rx { axis_round(cx) } else { axis(cx) };
    let (y0, wy) = if ry { axis_round(cy) } else { axis(cy) };
    bilin_v128_at(x0, wx, y0, wy, fetch).map(|res| (res >> 7).clamp(0, 255) as f32 / 255.0)
}

/// Linear-`sample` premultiplied RGBA at texel-space `(cx, cy)`, reproducing WARP's effect-shader
/// `sample` kernel in its operation order (byte-identical on a captured affine readback). `fetch(x, y)` returns the premultiplied
/// UNORM8 channels of texel `(x, y)` as `i32`, with the caller's extend/border already applied
/// (out-of-range → 0 for SOFT, clamped for HARD). The 2×2 is the kernel's **integer `value·128`**
/// two-stage lerp — x: `top = ((b−a)·wx + (a<<8)) >> 1`, y: `top + ((wy<<8)·(bot−top)) >> 16` — then
/// `>> 7` (**truncate**, as WARP shifts) to the premultiplied byte. (The earlier f32 product-sum /
/// `×1/255` normalize rounded a step off on ~0.3 % of texels; this truncating form is ≤1 LSB only at
/// exact texel-boundary coords.) Returned as `byte/255` f32 so the 8bpc node store round-trips exactly.
#[inline]
pub fn sample_linear(cx: f32, cy: f32, fetch: impl Fn(i32, i32) -> [i32; 4]) -> [f32; 4] {
    bilin_v128(cx, cy, fetch).map(|res| (res >> 7).clamp(0, 255) as f32 / 255.0)
}

/// The `value·128` 2×2 bilinear per channel at texel-space `(cx, cy)` — the shared integer core of
/// WARP's effect LINEAR/MSL samplers: Q8 [`axis`] sub-texel weights, x-lerp `((b−a)·wx + (a<<8)) >> 1`
/// then y-lerp `+ ((wy<<8)·(bot−top)) >> 16` (16-bit high multiply with sign-fix). `fetch` supplies premul UNORM8
/// corners (border already applied). [`sample_linear`] truncates `>> 7`; the MSL and blur paths
/// normalise each tap with [`crate::sampling::norm_v128`] and accumulate in f32.
#[inline]
pub fn bilin_v128(cx: f32, cy: f32, fetch: impl Fn(i32, i32) -> [i32; 4]) -> [i32; 4] {
    let ((x0, wx), (y0, wy)) = (axis(cx), axis(cy));
    bilin_v128_at(x0, wx, y0, wy, fetch)
}

/// [`bilin_v128`] at an explicit Q8 texel address: lower texels `(x0, y0)` and 8-bit weights.
#[inline]
pub fn bilin_v128_at(x0: i32, wx: i32, y0: i32, wy: i32, fetch: impl Fn(i32, i32) -> [i32; 4]) -> [i32; 4] {
    let (c00, c10, c01, c11) = (fetch(x0, y0), fetch(x0 + 1, y0), fetch(x0, y0 + 1), fetch(x0 + 1, y0 + 1));
    core::array::from_fn(|k| {
        let (a, b, c, d) = (c00[k], c10[k], c01[k], c11[k]);
        let top = ((b - a) * wx + (a << 8)) >> 1;
        let bot = ((d - c) * wx + (c << 8)) >> 1;
        top + ((((wy as i64) << 8) * (bot - top) as i64) >> 16) as i32
    })
}

/// D3D11 4× rotated-grid sub-sample offsets (texel offsets per unit footprint), the `MULTI_SAMPLE_LINEAR`
/// minification pattern. Same four points as the DrawBitmap `MSAA4` grid; here scaled by the texcoord
/// footprint (minify ratio) so each tap lands within the output pixel's input coverage.
const MSL_GRID: [(f32, f32); 4] =
    [(3.0 / 8.0, -1.0 / 8.0), (-1.0 / 8.0, -3.0 / 8.0), (-3.0 / 8.0, 1.0 / 8.0), (1.0 / 8.0, 3.0 / 8.0)];

/// Effect `MULTI_SAMPLE_LINEAR` minification `sample`, reproducing WARP's minify kernel in its operation
/// order (byte-identical on the ×0.5 readback, 0/64). Four [`MSL_GRID`]
/// taps around texel-space `(cx, cy)`, each offset by the footprint `(foot_x, foot_y)` (the minify ratio,
/// = `1/scale` texels per output pixel). Each tap is the integer `value·128` 2×2 bilinear (same as
/// [`sample_linear`] before its `>>7`), normalised to `[0,1]` by WARP's f32 [`crate::sampling::norm_v128`]
/// (`·1/128`, then `·1/256 + ·1/65280`) and averaged `×0.25`, accumulated in f32. Returns premultiplied
/// `[0,1]`; the 8bpc node store rounds `×255` ties-to-even (the kernel's `×255` and round-to-nearest-even conversion). `fetch`
/// applies the caller's border (0 for SOFT, clamped for HARD).
#[inline]
pub fn sample_msl(cx: f32, cy: f32, foot_x: f32, foot_y: f32, fetch: impl Fn(i32, i32) -> [i32; 4]) -> [f32; 4] {
    let mut acc = [0.0f32; 4];
    for (gx, gy) in MSL_GRID {
        let tap = bilin_v128(cx + foot_x * gx, cy + foot_y * gy, &fetch);
        for (k, a) in acc.iter_mut().enumerate() {
            *a += crate::sampling::norm_v128(tap[k]) * 0.25;
        }
    }
    acc
}

/// f32 product-sum bilinear (UNORM8 corners via [`unorm_to_f32`]) — the sampler the **lighting**
/// Sobel taps, which native evaluates in float, distinct from the resample's
/// integer `value·128` [`sample_linear`]. Keeping it separate is deliberate: the truncating integer
/// combine is ≤1 LSB off on the lighting alpha gradient (regresses `fx-transform-18`).
#[inline]
pub fn sample_linear_f32(cx: f32, cy: f32, fetch: impl Fn(i32, i32) -> [f32; 4]) -> [f32; 4] {
    sample_linear_f32_sel(cx, cy, false, false, fetch)
}

/// [`sample_linear_f32`] choosing the Q8 address per axis like [`sample_linear_sel`]: round
/// ([`axis_round`], cvtps2dq) when that axis's realized texture is non-pow2 **content**, floor
/// ([`axis`]) when it is a power of two. `rx`/`ry` request the round per axis.
#[inline]
pub fn sample_linear_f32_sel(cx: f32, cy: f32, rx: bool, ry: bool, fetch: impl Fn(i32, i32) -> [f32; 4]) -> [f32; 4] {
    let (x0, wx) = if rx { axis_round(cx) } else { axis(cx) };
    let (y0, wy) = if ry { axis_round(cy) } else { axis(cy) };
    let (fx, fy) = (wx as f32 / 256.0, wy as f32 / 256.0);
    let (c00, c10, c01, c11) = (fetch(x0, y0), fetch(x0 + 1, y0), fetch(x0, y0 + 1), fetch(x0 + 1, y0 + 1));
    let (w00, w10, w01, w11) = ((1.0 - fx) * (1.0 - fy), fx * (1.0 - fy), (1.0 - fx) * fy, fx * fy);
    core::array::from_fn(|k| c00[k] * w00 + c10[k] * w10 + c01[k] * w01 + c11[k] * w11)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_quantises_like_warp() {
        // Texel centre: weight 0 at integer+0.5 (point), lower texel = the integer.
        assert_eq!(axis(1.5), (1, 0));
        // Probe-confirmed native weights: frac 144/256 at coord 0.0647.
        assert_eq!(axis(0.064_706), (-1, 144));
        // Negative coordinate: arithmetic shift keeps the lower texel correct.
        assert_eq!(axis(-0.25).0, -1);
    }

    #[test]
    fn msl_minify_matches_warp_tie() {
        // The native minify capture: 16×16 input `[B=x·16, G=y·16, R=(y·16+x)&0xff, A=255]`,
        // ×0.5 affine MSL. Output (5,5) is an exact `.5` tie whose per-tap f32 `norm_v128` accumulate
        // rounds R up to 179 (not 178) — the last of the 8 ties the integer `Σ/512` model missed.
        let fetch = |x: i32, y: i32| {
            if !(0..16).contains(&x) || !(0..16).contains(&y) {
                return [0; 4]; // SOFT
            }
            [((y * 16 + x) & 0xff), (y * 16) & 0xff, (x * 16) & 0xff, 255] // R,G,B,A
        };
        // Output pixel (5,5): centre texel (2·5+1, 2·5+1), footprint 2 (= 1/0.5).
        let s = sample_msl(11.0, 11.0, 2.0, 2.0, fetch);
        let byte = |v: f32| (v * 255.0).round_ties_even() as i32;
        assert_eq!([byte(s[0]), byte(s[1]), byte(s[2]), byte(s[3])], [179, 168, 168, 255]);
    }

    #[test]
    fn sample_is_a_truncating_value128_lerp() {
        // Opaque white corner (255) at (0,0), 0 elsewhere; halfway both axes (c=1.0 → wx=wy=128):
        // top=((0−255)·128+(255<<8))>>1=16320 (255·128·½), bot=0, res=top+((128<<8)·(0−16320))>>16=
        // 16320−8160=8160=63.75·128, >>7 (trunc)=63 → 63/255. (The float quarter-weight was 0.25·255=63.75.)
        let fetch = |x: i32, y: i32| if x == 0 && y == 0 { [255; 4] } else { [0; 4] };
        let s = sample_linear(1.0, 1.0, fetch);
        for ch in s {
            assert_eq!((ch * 255.0).round() as i32, 63, "truncated quarter, got {ch}");
        }
    }
}
