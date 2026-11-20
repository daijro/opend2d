//! WARP's screen-space attribute-plane setup and the sampler's Q8 texel address, as WARP (Win11 26100)
//! computes them. This is the "plane setup" shared by the radial
//! gradient brush (the per-pixel texcoord `pc = M·pixel` is a plane-interpolated varying; the fitted
//! plane *is* `M`) and the effect-transform / resample sampler (the normalized input texcoord `U`).
//! Both route their per-pixel coordinate through [`setup_triangle_attr`]; the effect sampler then
//! takes the per-pixel texel address through [`q8_address`]. Keeping one implementation here means the
//! WARP attribute floor has one home.
//!
//! What WARP's setup does (correcting the earlier "rcp"/"snapped-plane-fit" guesses):
//! - Positions enter as the **1/256-subpixel SNAPPED screen integers** converted to f32 then
//!   `·(1/256)` — the same per-triangle terms WARP precomputes.
//! - The reciprocal of twice the signed area is **exact**: `inv = 1 / (area2 · 2^-16)` by a true
//!   division on the int64 subpixel area — *not* a reciprocal approximation. So the plane arithmetic is plain f32; a
//!   residual (e.g. radial's −4 ULP diagonal) is in the per-vertex attribute *values* fed in (the
//!   vertex shader's `M·v0` summation), not in this fit.

/// `2^-16` as f32 (`0x37800000`), WARP's subpixel-area scale.
const SUBPIXEL_AREA_SCALE: f32 = f32::from_bits(0x3780_0000);
/// `1/256` as f32 (`0x3b800000`), the subpixel→pixel scale WARP multiplies the snapped positions by.
const SUBPIXEL_TO_PIXEL: f32 = f32::from_bits(0x3b80_0000);

/// Fit the screen-space plane `[dadx, dady, c0]` of a scalar attribute `a`, given the triangle's
/// three vertices' **snapped 1/256-subpixel integer** screen positions (`sx`,`sy`), with vertex 0 the
/// reference (provoking) vertex WARP's setup keys on. The attribute at pixel centre `(px, py)` is then
/// `c0 + px·dadx + py·dady`. Matches WARP's attribute-plane setup in its operation order (edge terms as
/// `(d1·Δ + d2·Δ)·inv`, `c0 = (a0 − dadx·refx) − dady·refy`); degenerate (zero-area) → flat `a[0]`.
pub fn setup_triangle_attr(sx: [i64; 3], sy: [i64; 3], a: [f32; 3]) -> [f32; 3] {
    // Twice the signed area in subpixels (exact i64), then the exact f32 reciprocal WARP's division
    // forms: 1 / (area2 · 2^-16).
    // (In i128: a vertex far outside any guard band, e.g. a bitmap replayed from a command list under a 1e6 scale,
    // overflows the i64 product; r23 28111.)
    let area2 = (sx[1] - sx[0]) as i128 * (sy[2] - sy[0]) as i128 - (sx[2] - sx[0]) as i128 * (sy[1] - sy[0]) as i128;
    if area2 == 0 {
        return [0.0, 0.0, a[0]];
    }
    let inv = 1.0f32 / (area2 as f32 * SUBPIXEL_AREA_SCALE);
    // Snapped position deltas in pixels (WARP's per-triangle terms); `as i32 as f32` is its int→f32 conversion.
    let s = SUBPIXEL_TO_PIXEL;
    let dy_20 = (sy[2] - sy[0]) as i32 as f32 * s; // y2 − y0
    let dy_01 = (sy[0] - sy[1]) as i32 as f32 * s; // y0 − y1
    let dx_01 = (sx[1] - sx[0]) as i32 as f32 * s; // x1 − x0
    let dx_02 = (sx[0] - sx[2]) as i32 as f32 * s; // x0 − x2
    let ref_x = sx[0] as i32 as f32 * s;
    let ref_y = sy[0] as i32 as f32 * s;
    let d1 = a[1] - a[0];
    let d2 = a[2] - a[0];
    let dadx = (d1 * dy_20 + d2 * dy_01) * inv;
    let dady = (d1 * dx_02 + d2 * dx_01) * inv;
    let c0 = (a[0] - dadx * ref_x) - dady * ref_y;
    [dadx, dady, c0]
}

/// WARP's radial-fill plane fit, applied to a device→texcoord affine `m` (brush `apply` layout
/// `[m00, m10, m01, m11, dx, dy]`). Confirmed by tracing WARP's triangle setup: WARP rasterises the whole fill from a single **unit triangle** at screen
/// `(0,0),(1,0),(1,1)` (snapped subpixel `(0,256,256)`/`(0,0,256)`, area2 = 65536), fits the attribute
/// plane over it once (VS `o1 = M·v` per vertex, then the attribute-plane fit), and extrapolates
/// that plane across every pixel. The fit's `(m·1 + c) − c` f32 round-trip drops each diagonal a few
/// ULP against the translation (radial5: `0x3d044444`→`0x3d044440`) while the translation returns
/// unchanged; the per-pixel texcoord is then `c0 + px·dadx + py·dady` with those coefficients. A clean
/// matrix where `(m+c)−c == m` is a no-op.
pub fn warp_fit_affine(m: &[f32; 6]) -> [f32; 6] {
    const SX: [i64; 3] = [0, 256, 256];
    const SY: [i64; 3] = [0, 0, 256];
    let verts = [(0.0f32, 0.0f32), (1.0, 0.0), (1.0, 1.0)];
    let ox = verts.map(|(px, py)| m[0] * px + m[2] * py + m[4]); // o1.x = M·v per vertex
    let oy = verts.map(|(px, py)| m[1] * px + m[3] * py + m[5]); // o1.y = M·v per vertex
    let x = setup_triangle_attr(SX, SY, ox); // [dadx, dady, c0] → m00, m01, dx
    let y = setup_triangle_attr(SX, SY, oy); // → m10, m11, dy
    [x[0], y[0], x[1], y[1], x[2], y[2]]
}

/// [`warp_fit_affine`] for a scalar attribute `u = a·x + b·y + c` (`xf = [a, b, c]`): the
/// unit-triangle fit WARP's geometry draws use, returned as `[dadx, dady, c0]`.
pub fn warp_fit_linear(xf: &[f32; 3]) -> [f32; 3] {
    setup_triangle_attr([0, 256, 256], [0, 0, 256], [xf[2], xf[0] + xf[2], (xf[0] + xf[1]) + xf[2]])
}

/// Evaluate a plane `[dadx, dady, c0]` at pixel centre `(px, py)` (integers), in WARP's order
/// `c0 + px·dadx + py·dady`.
#[inline]
pub fn eval(plane: &[f32; 3], px: i32, py: i32) -> f32 {
    plane[2] + px as f32 * plane[0] + py as f32 * plane[1]
}

/// WARP's **extreme-scale** bitmap-brush per-span attribute DDA for one axis, reproduced from the
/// captured plane (jit-dump `bb` multi-span, probebm 1e-6..1e-3). When the texel-space gradient
/// overflows WARP's 16.16 fixed-point rasterizer accumulator over the fill extent, WARP rasterises
/// the fill in **8-row tiles** carrying one global f32 plane and accumulates the normalised texcoord
/// incrementally so the low bits survive (a direct `coeff·px` product loses them at ~1e8 magnitude):
/// - **X (within a scanline):** `u = base + coeff·(col&3)`, then `+= 4·coeff` per 4-pixel SIMD block
///   (`col>>2` times); restarts from `base` each row. (`base = coeff·½`, matching the captured plane.)
/// - **Y (per 8-row tile):** re-base `v = coeff·(8·tile) + base` (direct from the global plane) at each
///   tile start, then `+= coeff` per row within the tile.
///
/// `col`/`row` are 0-based within the fill's device bounding box; `base` is the texcoord at the
/// fill-origin pixel centre. Returns the normalised texcoord (caller folds `frac·size`). Confirmed
/// bit-exact against the captured per-span planes at s∈{1e-6,1e-5,1e-4,1e-3}.
#[inline]
pub fn tiled_dda_x(coeff: f32, base: f32, col: i32) -> f32 {
    let mut u = base + coeff * (col & 3) as f32;
    for _ in 0..(col >> 2) {
        u += 4.0 * coeff;
    }
    u
}

#[inline]
pub fn tiled_dda_y(coeff: f32, base: f32, row: i32) -> f32 {
    let t0 = row & !7;
    let mut v = coeff * t0 as f32 + base;
    for _ in 0..(row & 7) {
        v += coeff;
    }
    v
}

/// The effect sampler's per-pixel Q8 texel address from a plane-interpolated **normalized** texcoord
/// `u` (one axis): WARP clamps `u ∈ [−10, 10]`, then forms `q = floor(clamp(u) · tex_size · 256)` in
/// one scale (multiply, floor, convert to int), giving `texel = q >> 8` and 8-bit sub-texel
/// `weight = q & 0xff`. The `−0.5` half-texel centring is already folded into the plane's `c0`
/// (so callers pass the raw interpolated `u`).
#[inline]
pub fn q8_address(u: f32, tex_size: f32) -> (i32, i32) {
    let q = (u.clamp(-10.0, 10.0) * tex_size * 256.0).floor() as i32;
    (q >> 8, q & 0xff)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The fit recovers an exactly-linear attribute's gradients. A triangle at integer pixels
    // (snapped = pixel·256) carrying a(x,y) = 3x + 5y + 7 must come back [3, 5, 7].
    #[test]
    fn recovers_linear_gradient() {
        let px = [2i64, 40, 6];
        let py = [3i64, 5, 48];
        let lin = |x: i64, y: i64| 3.0 * x as f32 + 5.0 * y as f32 + 7.0;
        let sx = px.map(|v| v * 256);
        let sy = py.map(|v| v * 256);
        let a = [lin(px[0], py[0]), lin(px[1], py[1]), lin(px[2], py[2])];
        let plane = setup_triangle_attr(sx, sy, a);
        assert!((plane[0] - 3.0).abs() < 1e-4, "dadx {}", plane[0]);
        assert!((plane[1] - 5.0).abs() < 1e-4, "dady {}", plane[1]);
        // Evaluates back to the attribute at each vertex.
        for i in 0..3 {
            assert!((eval(&plane, px[i] as i32, py[i] as i32) - a[i]).abs() < 1e-3);
        }
    }

    // Degenerate triangle → flat.
    #[test]
    fn zero_area_is_flat() {
        let p = setup_triangle_attr([0, 256, 512], [0, 0, 0], [1.0, 2.0, 3.0]);
        assert_eq!(p, [0.0, 0.0, 1.0]);
    }

    // Fitting the captured radial texmap (diagonal 0x3d044444, translation dx/dy) over the unit
    // triangle WARP rasterizes for the fill (`screen (0,0),(1,0),(1,1)`)
    // reproduces its stored plane struct's diagonal bit-exact (0x3d044440, −4 ULP) — evidence that
    // WARP fits the attribute plane per small tile, not over the full fill rect. (Not wired into the
    // radial brush: the shift is tile-origin-dependent, so a uniform origin-0 fit regresses the
    // corpus.)
    #[test]
    fn plane_fit_over_unit_tile_drops_diagonal_4ulp() {
        let m00 = f32::from_bits(0x3d04_4444);
        let dx = f32::from_bits(0xc004_4444);
        let ox = [(0.0f32, 0.0f32), (1.0, 0.0), (1.0, 1.0)].map(|(px, py)| m00 * px + 0.0 * py + dx);
        let x = setup_triangle_attr([0, 256, 256], [0, 0, 256], ox);
        assert_eq!(x[0].to_bits(), 0x3d04_4440, "diagonal −4 ULP");
        assert_eq!(x[1], 0.0, "off-diagonal stays zero");
        assert_eq!(x[2].to_bits(), 0xc004_4444, "translation unchanged");
    }

    // Q8 address matches the probe-confirmed case from `effects::sample` (normalized u, tex_size 1 is
    // texel space): texel centre (u=1.5, size 1) → texel 1, weight 128; a mid coordinate floors down.
    #[test]
    fn q8_matches_probe() {
        assert_eq!(q8_address(1.5, 1.0), (1, 128));
        // u·size·256 just below an integer texel floors to texel−1 with a high weight.
        let (t, w) = q8_address(0.999, 1.0);
        assert_eq!(t, 0);
        assert!(w > 240, "weight {w}");
    }
}
