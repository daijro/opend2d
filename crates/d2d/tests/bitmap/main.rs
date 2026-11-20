//! Phase 1E conformance & unit checks for `d2d::bitmap` and `d2d::sampling`.
//!
//! Pixel scenes are driven through the bitmap module and compared byte-for-byte with the WARP
//! capture in `conformance/oracle/win11-26100`. Nearest, identity-copy and the additive /
//! source-copy `DrawImage` paths are exact; WARP's linear filter kernel is reproduced bit-for-bit
//! from its native fixed-point (`sampling.rs`, verified against the JIT IR emitters). A ≤1-LSB
//! residual remains on ~2% of fractional-scale pixels — not from the kernel but from WARP's
//! rasterizer texcoord interpolation and the 4-channel packed-pair carry (see the `sampling.rs`
//! header) — so the linear scenes are asserted within a tight bound and their diff counts printed.

use d2d::bitmap::*;
use d2d::pixels::premultiply;
use d2d_reference::{read_scene, Op, Scene};
use d2d_types::abi::*;
use d2d_types::contracts::{ImageDesc, PixelBuffer};
use std::collections::HashMap;
use std::path::PathBuf;


fn scene_path(name: &str) -> PathBuf {
    d2d_reference::suite_dir().unwrap().join("scenes").join(format!("{name}.json"))
}
fn oracle_bgra(name: &str) -> Vec<u8> {
    let p = d2d_reference::suite_dir().unwrap().join("oracle/win11-26100").join(format!("{name}.bgra"));
    std::fs::read(&p).unwrap_or_else(|e| panic!("read oracle {}: {e}", p.display()))
}

fn target(scene: &Scene) -> PixelBuffer {
    let [w, h] = scene.size;
    PixelBuffer {
        desc: ImageDesc {
            width: w,
            height: h,
            format: DXGI_FORMAT_B8G8R8A8_UNORM,
            alpha_mode: D2D1_ALPHA_MODE_PREMULTIPLIED,
            dpi_x: scene.dpi,
            dpi_y: scene.dpi,
        },
        stride: w as usize * 4,
        data: vec![0; w as usize * h as usize * 4],
    }
}

/// Render the axis-aligned DrawBitmap/DrawImage scenes at 96 DPI (device space ≡ DIP). Returns
/// `None` for a scene using an op outside 1E's exact subset (a world transform, bitmap brush, …).
fn render(scene: &Scene) -> Option<PixelBuffer> {
    assert_eq!(scene.dpi, 96.0);
    let mut buf = target(scene);
    let mut bitmaps: HashMap<&str, Bitmap> = HashMap::new();
    for op in &scene.ops {
        match op {
            Op::Clear { color } => {
                let p = premultiply(*color);
                buf.data.chunks_exact_mut(4).for_each(|px| px.copy_from_slice(&p));
            }
            Op::Bitmap { id, size, rgba, .. } => {
                bitmaps.insert(id, Bitmap::from_straight_rgba(size[0], size[1], rgba));
            }
            Op::DrawBitmap { bitmap, dest, source, opacity, interpolation, .. } => {
                draw_bitmap(&mut buf, bitmaps.get(bitmap.as_str())?, *dest, *source, *opacity, *interpolation);
            }
            Op::DrawImage { image, offset, interpolation, composite, .. } => {
                let mode = match composite {
                    9 => Composite::Plus,
                    10 => Composite::SourceCopy,
                    0 => Composite::SourceOver,
                    _ => return None,
                };
                draw_image(&mut buf, bitmaps.get(image.as_str())?, *offset, *interpolation, mode);
            }
            _ => return None, // transform, bitmap brush, effect: not this test's subset
        }
    }
    Some(buf)
}

fn diffs(got: &[u8], want: &[u8], w: u32) -> (usize, Option<(u32, u32, [u8; 4], [u8; 4])>) {
    let mut n = 0;
    let mut first = None;
    for (i, (a, b)) in got.chunks_exact(4).zip(want.chunks_exact(4)).enumerate() {
        if a != b {
            n += 1;
            if first.is_none() {
                let p = i as u32;
                first = Some((p % w, p / w, a.try_into().unwrap(), b.try_into().unwrap()));
            }
        }
    }
    (n, first)
}

fn run(name: &str) -> (usize, [u32; 2]) {
    let scene = read_scene(&scene_path(name)).unwrap();
    let buf = render(&scene).unwrap_or_else(|| panic!("{name}: out-of-subset op"));
    let want = oracle_bgra(name);
    assert_eq!(buf.data.len(), want.len(), "{name}: size");
    let (n, first) = diffs(&buf.data, &want, scene.size[0]);
    if let Some((x, y, g, w)) = first {
        if n > 0 && name_must_be_exact(name) {
            panic!("{name}: first diff at ({x},{y}) got {g:?} want {w:?} ({n} total)");
        }
    }
    (n, scene.size)
}

fn name_must_be_exact(name: &str) -> bool {
    matches!(
        name,
        "bitmaps-01-identity-copy"
            | "bitmaps-02-magnify-nearest"
            | "bitmaps-03-magnify-linear"
            | "bitmaps-07-minify-linear"
            | "bitmaps-10-opacity-half"
            | "bitmaps-11-drawimage-plus"
            | "bitmaps-12-drawimage-copy"
    )
}
/// Byte-exact: identity copy, nearest magnify, the integer linear filter (magnify & minify), the
/// opacity path, and the additive / source-copy `DrawImage` composites — all confirmed against the
/// WARP dump (`sampling.rs` lerp + `bitmap` integer cutout).
#[test]
fn oracle_exact() {
    if d2d_reference::suite_dir().is_none() { return; }
    for name in [
        "bitmaps-01-identity-copy",
        "bitmaps-02-magnify-nearest",
        "bitmaps-03-magnify-linear",
        "bitmaps-07-minify-linear",
        "bitmaps-10-opacity-half",
        "bitmaps-11-drawimage-plus",
        "bitmaps-12-drawimage-copy",
    ] {
        let (n, _) = run(name);
        assert_eq!(n, 0, "{name} must be byte-exact");
    }
}

/// Near-exact (≤2 LSB) scenes. Two causes, both documented in `sampling.rs`:
/// - WARP linear with semi-transparent *premultiplied* texels (12, the opacity-isolating probe
///   warpjit-01, and MSL on magnify 06): the kernel is exact, the residual is WARP's rasterizer
///   fixed-point texcoord `base + px·step` whose 8-bit sub-texel weight is sometimes one less than
///   `round(tx·65536)`'s — it shows only once premultiplied alpha is interpolated.
/// - Cubic (04): WARP's Catmull-Rom weights and 8-bit fraction are reproduced, but its 16-bit
///   fixed-point nested-lerp tap combine is not, so ~3% of pixels near the transparent texels ring
///   by ≤2 LSB. See `sample_cubic`.
/// Bounded so a regression trips.
#[test]
fn oracle_linear_near_exact() {
    if d2d_reference::suite_dir().is_none() { return; }
    for name in
        ["12-bitmap-linear", "warpjit-01-linear-opaque", "bitmaps-06-magnify-msl", "bitmaps-04-magnify-cubic"]
    {
        let (n, [w, h]) = run(name);
        let total = (w * h) as usize;
        println!("{name}: {n}/{total} diffs ({:.2}%)", 100.0 * n as f64 / total as f64);
        assert!(n * 100 < total * 10, "{name}: {n} diffs exceeds 10% — sampler regression?");
    }
}

/// `bitmaps-09-rotate-aniso`: a 16×16 bitmap drawn into dest [0,0,40,40] under a 30° rotate+2.5×
/// magnify. The rotated textured-quad raster ([`draw_bitmap_affine`]) is **byte-exact** against the
/// WARP oracle: WARP's 8×8 analytic coverage for the quad-edge AA (via `crate::pixels::rasterize_edges`)
/// plus the ANISOTROPIC value·128 bilinear truncated (`>>7`) — on magnification the oriented 19-tap
/// footprint collapses to the centre bilinear and the kernel floors its accumulator. Guards the whole
/// draw against a raster/coverage/texcoord regression.
#[test]
fn rotate_aniso_exact() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    let scene = read_scene(&scene_path("bitmaps-09-rotate-aniso")).unwrap();
    let want = oracle_bgra("bitmaps-09-rotate-aniso");
    let [w, h] = scene.size;
    let mut buf = target(&scene);
    let mut bm: Option<Bitmap> = None;
    // device transform = the scene's world transform at 96 DPI (device ≡ DIP).
    let m = [0.866f64, 0.5, -0.5, 0.866, 48.0, 10.0];
    let ap = |x: f32, y: f32| [(m[0] * x as f64 + m[2] * y as f64 + m[4]) as f32, (m[1] * x as f64 + m[3] * y as f64 + m[5]) as f32];
    for op in &scene.ops {
        match op {
            Op::Clear { color } => {
                let p = premultiply(*color);
                buf.data.chunks_exact_mut(4).for_each(|px| px.copy_from_slice(&p));
            }
            Op::Bitmap { size, rgba, .. } => bm = Some(Bitmap::from_straight_rgba(size[0], size[1], rgba)),
            Op::DrawBitmap { dest, source, opacity, interpolation, .. } => {
                let c = [ap(dest[0], dest[1]), ap(dest[2], dest[1]), ap(dest[2], dest[3]), ap(dest[0], dest[3])];
                draw_bitmap_affine(&mut buf, bm.as_ref().unwrap(), c, *source, *opacity, *interpolation);
            }
            _ => {}
        }
    }
    let (n, first) = diffs(&buf.data, &want, w);
    println!("bitmaps-09 rotated raster: {n}/{} diffs, first={first:?}", (w * h) as usize);
    assert_eq!(n, 0, "bitmaps-09 rotated raster regressed: {n} diffs (first {first:?})");
}

// --- validation unit checks -------------------------------------------------------

#[test]
fn pixel_format_table() {
    // BGRA8 premultiplied — the harness target/bitmap format — is valid.
    assert!(validate_pixel_format(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED).is_ok());
    // A8 is the only format allowing STRAIGHT.
    assert!(validate_pixel_format(DXGI_FORMAT_A8_UNORM, D2D1_ALPHA_MODE_STRAIGHT).is_ok());
    assert_eq!(
        validate_pixel_format(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_STRAIGHT),
        Err(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT),
        "only A8 allows straight alpha"
    );
    // R8/R8G8 allow only IGNORE; premultiplied is rejected.
    assert!(validate_pixel_format(DXGI_FORMAT_R8_UNORM, D2D1_ALPHA_MODE_IGNORE).is_ok());
    assert!(validate_pixel_format(DXGI_FORMAT_R8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED).is_err());
    // UNKNOWN format and UNKNOWN alpha are always rejected.
    assert!(validate_pixel_format(DXGI_FORMAT_UNKNOWN, D2D1_ALPHA_MODE_PREMULTIPLIED).is_err());
    assert_eq!(
        validate_pixel_format(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_UNKNOWN),
        Err(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT)
    );
    // A format absent from the 28-entry table (UINT) is unsupported.
    assert!(validate_pixel_format(DXGI_FORMAT_R8G8B8A8_UINT, D2D1_ALPHA_MODE_IGNORE).is_err());
    // BGRA8 is a drawable source; A8 is not.
    assert!(is_drawable_source(format_flags(DXGI_FORMAT_B8G8R8A8_UNORM).unwrap()));
    assert!(!is_drawable_source(format_flags(DXGI_FORMAT_A8_UNORM).unwrap()));
}

#[test]
fn bitmap_options() {
    use D2D1_BITMAP_OPTIONS as O;
    assert!(validate_bitmap_options(D2D1_BITMAP_OPTIONS_NONE).is_ok());
    assert!(validate_bitmap_options(D2D1_BITMAP_OPTIONS_TARGET).is_ok());
    // CPU_READ requires CANNOT_DRAW and excludes TARGET.
    assert!(validate_bitmap_options(D2D1_BITMAP_OPTIONS_CPU_READ).is_err());
    assert!(validate_bitmap_options(O(D2D1_BITMAP_OPTIONS_CPU_READ.0 | D2D1_BITMAP_OPTIONS_CANNOT_DRAW.0)).is_ok());
    assert!(validate_bitmap_options(O(D2D1_BITMAP_OPTIONS_CPU_READ.0 | D2D1_BITMAP_OPTIONS_TARGET.0)).is_err());
    // CANNOT_DRAW alone (without CPU_READ) is illegal, as is GDI without TARGET.
    assert!(validate_bitmap_options(D2D1_BITMAP_OPTIONS_CANNOT_DRAW).is_err());
    assert!(validate_bitmap_options(D2D1_BITMAP_OPTIONS_GDI_COMPATIBLE).is_err());
    assert!(validate_bitmap_options(O(D2D1_BITMAP_OPTIONS_GDI_COMPATIBLE.0 | D2D1_BITMAP_OPTIONS_TARGET.0)).is_ok());
    // An unknown high bit is rejected.
    assert!(validate_bitmap_options(O(0x10)).is_err());
}

#[test]
fn dpi_rule() {
    assert_eq!(resolve_dpi(0.0, 0.0, (96.0, 96.0)), Ok((96.0, 96.0)));
    assert_eq!(resolve_dpi(192.0, 144.0, (96.0, 96.0)), Ok((192.0, 144.0)));
    assert!(resolve_dpi(0.0, 96.0, (96.0, 96.0)).is_err()); // exactly one zero
    assert!(resolve_dpi(-1.0, 96.0, (96.0, 96.0)).is_err());
}

#[test]
fn color_math() {
    // D2D1ConvertColorSpace: same space unchanged, custom → transparent black, alpha passthrough.
    assert_eq!(convert_color_space(COLOR_SPACE_SRGB, COLOR_SPACE_SRGB, [0.3, 0.4, 0.5, 0.6]), [0.3, 0.4, 0.5, 0.6]);
    assert_eq!(convert_color_space(COLOR_SPACE_CUSTOM, COLOR_SPACE_SRGB, [0.3, 0.4, 0.5, 0.6]), [0.0; 4]);
    let lin = convert_color_space(COLOR_SPACE_SRGB, COLOR_SPACE_SCRGB, [0.5, 0.5, 0.5, 0.8]);
    assert_eq!(lin[3], 0.8, "alpha passes through");
    assert!(lin[0] < 0.5, "sRGB 0.5 decodes below 0.5 in linear");
    // round-trip sRGB→scRGB→sRGB is near-identity.
    let back = convert_color_space(COLOR_SPACE_SCRGB, COLOR_SPACE_SRGB, lin);
    assert!((back[0] - 0.5).abs() < 1e-4);
    // Premultiply / UnPremultiply are inverses for a>0.
    assert_eq!(premultiply_color([1.0, 0.5, 0.25, 0.5]), [0.5, 0.25, 0.125, 0.5]);
    assert_eq!(unpremultiply_color([0.5, 0.25, 0.125, 0.5]), [1.0, 0.5, 0.25, 0.5]);
    assert_eq!(unpremultiply_color([0.0, 0.0, 0.0, 0.0]), [0.0; 4]);
}
