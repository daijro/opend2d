//! Phase 2C filter-family focused tests: the effect functions exercised directly on synthetic
//! inputs (impulse, step, checkerboard) for properties the full-scene oracle meter (in
//! `tests/conformance.rs`) does not isolate, plus a full-frame parity diagnostic for the blur-based
//! Shadow effect that reports the residual against the WARP oracle.

use d2d::bitmap::Bitmap;
use d2d::effects::builtin::filters;
use d2d::{bitmap_target, Factory};
use std::path::PathBuf;

fn oracle(name: &str) -> Vec<u8> {
    let p = d2d_reference::suite_dir().unwrap().join(format!("oracle/win11-26100/{name}.bgra"));
    std::fs::read(p).unwrap()
}

/// A solid opaque straight-RGBA bitmap builder for synthetic inputs.
fn solid(w: u32, h: u32, px: impl Fn(u32, u32) -> [u8; 4]) -> Bitmap {
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            rgba.extend_from_slice(&px(x, y));
        }
    }
    Bitmap::from_straight_rgba(w, h, &rgba)
}

/// Render an effect output the way the scenes do: clear white, draw it at its logical origin under
/// a scale+translate transform with nearest-neighbour `DrawImage`, return the premultiplied BGRA.
fn render(out: &filters::Output, size: [u32; 2], m: [f32; 6]) -> Vec<u8> {
    let factory = Factory::new(false);
    let mut dc = factory.create_device().create_device_context();
    dc.set_dpi(96.0, 96.0);
    dc.set_target_bitmap(bitmap_target(size[0], size[1], 96.0));
    dc.begin_draw();
    dc.clear(Some([1.0, 1.0, 1.0, 1.0]));
    dc.set_transform(m);
    let (w, h) = (out.bitmap.width as f32, out.bitmap.height as f32);
    let (ox, oy) = (out.ox as f32, out.oy as f32);
    dc.draw_bitmap(&out.bitmap, [ox, oy, ox + w, oy + h], [0.0, 0.0, w, h], 1.0, 0);
    let _ = dc.end_draw();
    dc.take_target().unwrap().data
}

/// |diff| histogram between a render and the oracle, and the worst pixel.
fn diff_stats(got: &[u8], want: &[u8]) -> (u32, i32, usize) {
    let (mut n, mut max, mut worst) = (0u32, 0i32, 0usize);
    for i in 0..got.len() {
        let d = (got[i] as i32 - want[i] as i32).abs();
        if d != 0 {
            n += 1;
        }
        if d > max {
            max = d;
            worst = i;
        }
    }
    (n, max, worst)
}

// The scenes all draw an 8×8 effect output at scale 10, translate (24,8), into a 128×96 target.
const M10: [f32; 6] = [10.0, 0.0, 0.0, 10.0, 24.0, 8.0];
const SIZE: [u32; 2] = [128, 96];

/// The 4×4 opaque-white-on-transparent 8×8 input shared by the degenerate scenes.
fn white_square_8() -> Bitmap {
    solid(8, 8, |x, y| if (2..6).contains(&x) && (2..6).contains(&y) { [255; 4] } else { [0; 4] })
}

#[test]
fn shadow_parity_residual() {
    if d2d_reference::suite_dir().is_none() { return; }
    // Scene 04: σ=3, red shadow, QUALITY optimization (the scene's prop 2 = enum 2), matching the
    // fx-filters-04 oracle. Blurs via the native tap-pair graph (SOFT forces full-res / even-pad).
    let out = filters::shadow(&white_square_8(), 3.0, [1.0, 0.0, 0.0, 1.0], 2);
    let got = render(&out, SIZE, M10);
    let (n, max, worst) = diff_stats(&got, &oracle("fx-filters-04-shadow"));
    println!("shadow: {n} px differ, max |Δ|={max}, worst byte {worst}");
    // Documents the current residual: the native bilinear-tap-pair/optimization blur gap, a small
    // LSB difference, never a gross error. Tighten this bound when the native taps are reproduced.
    assert!(max <= 6, "shadow residual grew beyond the known blur gap: max |Δ|={max}");
}

/// 4×4 opaque magenta square on transparent (the large-σ blur scenes' input).
fn magenta_square_8() -> Bitmap {
    solid(8, 8, |x, y| if (2..6).contains(&x) && (2..6).contains(&y) { [220, 30, 180, 255] } else { [0; 4] })
}

#[test]
fn gaussian_large_parity_residual() {
    if d2d_reference::suite_dir().is_none() { return; }
    // fxsample-02: σ=6 QUALITY SOFT. σ>2 ⇒ native downsamples (CalculateOptimizationTransforms) and
    // resamples through WARP's minification sampler, which is unread; the plain full-res convolution
    // is the ceiling until that kernel is reversed. Documents the downsample-path residual.
    let out = filters::gaussian_blur(&magenta_square_8(), 6.0, 2, false);
    let got = render(&out, SIZE, M10);
    let (n, max, _) = diff_stats(&got, &oracle("fxsample-02-gaussian-large"));
    println!("gaussian σ=6: {n} px differ, max |Δ|={max}");
}

#[test]
fn directional_large_parity_residual() {
    if d2d_reference::suite_dir().is_none() { return; }
    // fxsample-03: σ=6 angle 30 QUALITY SOFT. Same downsample-path residual as the gaussian.
    let out = filters::directional_blur(&magenta_square_8(), 6.0, 30.0, false);
    let got = render(&out, SIZE, M10);
    let (n, max, _) = diff_stats(&got, &oracle("fxsample-03-directional-large"));
    println!("directional σ=6: {n} px differ, max |Δ|={max}");
}

#[test]
fn displacement_parity_residual() {
    if d2d_reference::suite_dir().is_none() { return; }
    // Scene 17: scale 4, X=R, Y=G. f32 bilinear resample; residual is a ≤1-LSB f32 rounding gap.
    let img = solid(8, 8, |x, y| [(x * 36).min(255) as u8, (y * 36).min(255) as u8, 128, 255]);
    let dmap = solid(8, 8, |x, y| [(x * 36).min(255) as u8, (y * 36).min(255) as u8, 0, 255]);
    let out = filters::displacement_map(&img, &dmap, 4.0, 0, 1, [false, false]); // 8×8 pow2 content ⇒ floor
    let got = render(&out, SIZE, M10);
    let (n, max, _) = diff_stats(&got, &oracle("fx-filters-17-displacement-map"));
    println!("displacement: {n} bytes differ, max |Δ|={max}");
    assert!(max <= 1, "displacement residual grew beyond the f32 rounding gap: max |Δ|={max}");
}

#[test]
fn gaussian_impulse_is_symmetric_and_conserves_alpha() {
    // A single opaque-white centre pixel on transparent: the blurred alpha must be symmetric and
    // its premultiplied-white relation (rgb==a) preserved everywhere.
    let input = solid(9, 9, |x, y| if x == 4 && y == 4 { [255; 4] } else { [0; 4] });
    let out = filters::gaussian_blur(&input, 2.0, 1, false);
    let bm = &out.bitmap;
    let at = |x: i32, y: i32| {
        let i = (((y - out.oy) * bm.width as i32 + (x - out.ox)) * 4) as usize;
        [bm.premul_bgra[i], bm.premul_bgra[i + 1], bm.premul_bgra[i + 2], bm.premul_bgra[i + 3]]
    };
    let c = at(4, 4);
    assert_eq!(at(3, 4), at(5, 4), "x-symmetry");
    assert_eq!(at(4, 3), at(4, 5), "y-symmetry");
    assert_eq!(at(3, 3), at(5, 5), "diagonal symmetry");
    assert_eq!([c[0], c[1], c[2]], [c[3], c[3], c[3]], "white stays premultiplied-white");
}

#[test]
fn zero_sigma_is_passthrough() {
    let input = white_square_8();
    let out = filters::gaussian_blur(&input, 0.0, 1, false);
    assert_eq!(out.bitmap.premul_bgra, input.premul_bgra);
    assert_eq!((out.ox, out.oy, out.bitmap.width), (0, 0, 8));
}

#[test]
fn morphology_dilate_grows_erode_shrinks() {
    let input = white_square_8();
    let dil = filters::morphology(&input, true, 3, 3);
    let ero = filters::morphology(&input, false, 3, 3);
    // 3×3 window ⇒ output grows 1px each side; origin (-1,-1).
    assert_eq!((dil.ox, dil.oy, dil.bitmap.width, dil.bitmap.height), (-1, -1, 10, 10));
    let opaque = |o: &filters::Output, x: i32, y: i32| {
        let i = (((y - o.oy) * o.bitmap.width as i32 + (x - o.ox)) * 4) as usize;
        o.bitmap.premul_bgra[i + 3] == 255
    };
    // Dilation reaches one ring outside the 4×4 square (e.g. (1,2)); erosion clears the square's rim.
    assert!(opaque(&dil, 1, 3), "dilate grows left");
    assert!(!opaque(&ero, 2, 2), "erode clears the corner");
    assert!(opaque(&ero, 3, 3), "erode keeps the interior");
}

#[test]
fn convolve_identity_is_passthrough_even_with_divisor_bias() {
    // Native quirk confirmed by scene 05: an identity kernel passes through, ignoring divisor/bias.
    let input = white_square_8();
    let id = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let out = filters::convolve(&input, 3, 3, &id, 2.0, 0.25, [0.0, 0.0], false, true, false);
    assert_eq!(out.bitmap.premul_bgra, input.premul_bgra);
}

#[test]
fn posterize_matches_oracle_bytes() {
    if d2d_reference::suite_dir().is_none() { return; }
    // Full-scene byte parity is covered by the conformance meter; here confirm the table directly.
    let ramp = solid(16, 4, |x, _| [(x * 17) as u8, (x * 17) as u8, (x * 17) as u8, 255]);
    let out = filters::posterize(&ramp, 2, 3, 4);
    let got = render(&out, SIZE, [7.0, 0.0, 0.0, 7.0, 8.0, 32.0]);
    let (n, max, _) = diff_stats(&got, &oracle("fx-filters-13-posterize"));
    assert_eq!((n, max), (0, 0), "posterize must be byte-exact");
}
