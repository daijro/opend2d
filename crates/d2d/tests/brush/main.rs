//! Phase 1D conformance: render the brush oracle scenes with `d2d::brush` and compare byte-for-byte
//! to the WARP capture in `conformance/oracle/win11-26100`.
//!
//! The brush scenes are pixel-aligned rectangle fills (full coverage), so this harness maps each
//! brush op into device space, builds the realized brush and fills the rectangle via
//! `brush::fill_rect` (full-coverage source-over, matching `d2d::pixels`). Scene 13's rounded-rect
//! geometry needs 1C and is not rendered here.

use d2d::bitmap::premultiply_rgba;
use d2d::brush::{fill_rect, Brush};
use d2d::pixels::premultiply;
use d2d::pixels::{Aa, Blend};
use d2d::{bitmap_target, Factory};
use d2d_reference::{read_scene, Op, Scene, Shape};
use d2d_types::abi::{D2D1_ALPHA_MODE_PREMULTIPLIED, DXGI_FORMAT_B8G8R8A8_UNORM};
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

// 3×2 affine [m11,m12,m21,m22,dx,dy]; compose(a,b) applies b then a.
type Mat = [f32; 6];
fn apply(m: &Mat, x: f32, y: f32) -> [f32; 2] {
    [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]
}
fn compose(a: &Mat, b: &Mat) -> Mat {
    [
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
        a[0] * b[4] + a[2] * b[5] + a[4],
        a[1] * b[4] + a[3] * b[5] + a[5],
    ]
}

fn stops(raw: &[d2d_reference::Stop]) -> Vec<(f32, [f32; 4])> {
    raw.iter().map(|s| (s.position, s.color)).collect()
}

fn render(scene: &Scene) -> Option<Vec<u8>> {
    let [w, h] = scene.size;
    let mut buf = PixelBuffer {
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
    };
    let ds = scene.dpi / 96.0;
    let dpi: Mat = [ds, 0.0, 0.0, ds, 0.0, 0.0];
    let ident: Mat = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let mut world: Mat = ident;
    let mut brushes: HashMap<&str, Brush> = HashMap::new();
    let mut geoms: HashMap<&str, [f32; 4]> = HashMap::new();
    let mut bitmaps: HashMap<&str, ([u32; 2], Vec<u8>)> = HashMap::new();

    for op in &scene.ops {
        // brush→device for a brush with brush transform `bt`.
        let b2d = |bt: &Mat| compose(&compose(&dpi, &world), bt);
        match op {
            Op::Clear { color } => {
                let p = premultiply(*color);
                for px in buf.data.chunks_exact_mut(4) {
                    px.copy_from_slice(&p);
                }
            }
            Op::Transform { matrix } => world = *matrix,
            Op::Solid { id, color, opacity, transform } => {
                if transform.is_some() {
                    return None; // solid brush transform only affects gradients/bitmaps; no-op here but keep honest
                }
                brushes.insert(id, Brush::solid(*color, opacity.unwrap_or(1.0)));
            }
            Op::Linear { id, start, end, stops: st, gamma, extend, opacity, transform } => {
                let m = b2d(&transform.unwrap_or(ident));
                brushes.insert(id, Brush::linear(&stops(st), *gamma, *extend, opacity.unwrap_or(1.0), *start, *end, &m));
            }
            Op::Radial { id, center, offset, radius, stops: st, gamma, extend, opacity, transform } => {
                let m = b2d(&transform.unwrap_or(ident));
                brushes.insert(id, Brush::radial(&stops(st), *gamma, *extend, opacity.unwrap_or(1.0), *center, *radius, *offset, &m));
            }
            Op::Bitmap { id, size, rgba, .. } => {
                bitmaps.insert(id, (*size, premultiply_rgba(rgba)));
            }
            Op::BitmapBrush { id, bitmap, extend, interpolation, matrix, opacity } => {
                let (size, texels) = bitmaps.get(bitmap.as_str())?.clone();
                let m = b2d(matrix);
                brushes.insert(id, Brush::bitmap(texels, size, &m, *extend, *interpolation, *opacity));
            }
            Op::Geometry { id, shape } => match shape {
                Shape::Rect { rect } => {
                    geoms.insert(id, *rect);
                }
                _ => return None, // non-rect geometry is out of this harness (1C)
            },
            Op::Fill { geometry, brush } => {
                let rect = geoms.get(geometry.as_str())?;
                let brush = brushes.get(brush.as_str())?;
                let a = apply(&compose(&dpi, &world), rect[0], rect[1]);
                let c = apply(&compose(&dpi, &world), rect[2], rect[3]);
                fill_rect(&mut buf, [a[0], a[1], c[0], c[1]], brush, Blend::SourceOver, Aa::PerPrimitive);
            }
            Op::Primitive { shape: Shape::Rect { rect }, brush, width: None } => {
                let brush = brushes.get(brush.as_str())?;
                let a = apply(&compose(&dpi, &world), rect[0], rect[1]);
                let c = apply(&compose(&dpi, &world), rect[2], rect[3]);
                fill_rect(&mut buf, [a[0], a[1], c[0], c[1]], brush, Blend::SourceOver, Aa::PerPrimitive);
            }
            _ => return None,
        }
    }
    Some(buf.data)
}

fn first_diff(got: &[u8], want: &[u8], w: u32) -> Option<(u32, u32, [u8; 4], [u8; 4])> {
    for i in (0..got.len()).step_by(4) {
        if got[i..i + 4] != want[i..i + 4] {
            let p = (i / 4) as u32;
            return Some((p % w, p / w, got[i..i + 4].try_into().unwrap(), want[i..i + 4].try_into().unwrap()));
        }
    }
    None
}

/// Solid and linear gradient scenes are byte-identical to the WARP oracle: the texture build
/// (box filter, borders, gamma), the extend modes, and the affine texture mapping are exact.
#[test]
fn oracle_exact() {
    if d2d_reference::suite_dir().is_none() { return; }
    let scenes = [
        "brushes-01-gradient-hardstop",
        "brushes-02-gamma-midpoint",
        "brushes-03-linear-extend",
        "brushes-04-stop-edges",
        "brushes-05-gradient-alpha",
        "09-linear-gamma",
    ];
    for name in scenes {
        let scene = read_scene(&scene_path(name)).unwrap();
        let buf = render(&scene).unwrap_or_else(|| panic!("{name}: uses an out-of-scope op"));
        let want = oracle_bgra(name);
        assert_eq!(buf.len(), want.len(), "{name}: size mismatch");
        if let Some((x, y, g, w)) = first_diff(&buf, &want, scene.size[0]) {
            let total: usize = buf.chunks_exact(4).zip(want.chunks_exact(4)).filter(|(a, b)| a != b).count();
            panic!("{name}: {total} px differ; first at ({x},{y}) got BGRA {g:?} want {w:?}");
        }
    }
}

/// The same linear gradient scenes, driven through the **public API** (`Factory` → `Device` →
/// `DeviceContext`: `create_linear_gradient_brush` + `fill_rectangle_brush`) exactly as the
/// conformance harness drives native, prove the 1D wiring into `DeviceContext` is byte-exact.
#[test]
fn api_wiring_exact() {
    if d2d_reference::suite_dir().is_none() { return; }
    let device = Factory::new(false).create_device();
    for name in ["brushes-01-gradient-hardstop", "brushes-02-gamma-midpoint", "09-linear-gamma"] {
        let scene = read_scene(&scene_path(name)).unwrap();
        let [w, h] = scene.size;
        let mut dc = device.create_device_context();
        dc.set_dpi(scene.dpi, scene.dpi);
        dc.set_target_bitmap(bitmap_target(w, h, scene.dpi));
        dc.begin_draw();
        let mut geoms: HashMap<&str, [f32; 4]> = HashMap::new();
        let mut brushes: HashMap<&str, d2d::PixelBrush> = HashMap::new();
        for op in &scene.ops {
            match op {
                Op::Clear { color } => dc.clear(Some(*color)),
                Op::Transform { matrix } => dc.set_transform(*matrix),
                Op::Linear { id, start, end, stops: st, gamma, extend, opacity, transform } => {
                    brushes.insert(id, dc.create_linear_gradient_brush(*start, *end, &stops(st), *gamma, *extend, opacity.unwrap_or(1.0), *transform));
                }
                Op::Geometry { id, shape: Shape::Rect { rect } } => {
                    geoms.insert(id, *rect);
                }
                Op::Fill { geometry, brush } => {
                    dc.fill_rectangle_brush(geoms[geometry.as_str()], &brushes[brush.as_str()]);
                }
                _ => unreachable!("{name}: unexpected op for the API wiring test"),
            }
        }
        let _ = dc.end_draw();
        let got = dc.take_target().unwrap().data;
        let want = oracle_bgra(name);
        if let Some((x, y, g, w)) = first_diff(&got, &want, scene.size[0]) {
            panic!("{name} (via API): first diff at ({x},{y}) got BGRA {g:?} want {w:?}");
        }
    }
}

/// Radial gradients and bitmap brushes match native's structure, mapping, extend modes, borders and
/// sampling. The gradient **texture + sampler** are now reproduced byte-for-byte (read from the WARP
/// texture dump and VM probes of its build): a radial clamp texture has a
/// trailing last-stop border only (no leading border, unlike linear), its gradient box-filtered over
/// `W−1` texels with the stop's texel as the straddle (`end = floor(p·N)`, **not** `floor(p·N+0.5)`),
/// and the clamp texcoord maps `t` across `[0, W−1]` (native's `span_right = W−1` / `v2.z = (W−1)/W`).
/// `23/80 canvas-radial` are byte-exact and listed in `conformance/passing.txt`.
///
/// The residual on the rest is ≤1 LSB on gradient-transition pixels and is **not** the realization
/// matrix `M` (an earlier guess, disproven here). Tracing the whole brush → realization → texcoord-
/// mapping → invert → ×1/W → compose chain against native shows our
/// `M` already equals native's for every radial scene here (all identity-transform at 96 DPI, so the
/// final compose with the context matrix is a no-op). The 4-ULP gap seen against the captured `M` was
/// a property of the *dump* scene only: its context matrix carries a non-unit DPI scale `s`, and
/// native's `(invert·1/W)·s` with the s-scaled placement accumulates the extra ULPs — no identity-`s`
/// input reproduces the captured `0x3d044440`, a DPI-scaled `s` does.
///
/// The true residual is the per-pixel coordinate `v1` at knife-edge texel-weight boundaries: `focal_t`
/// is exact (exact `pc` → exact pixel), but WARP forms `v1` by interpolating its VS output across the
/// fill triangles (`crate::warp::fill_triangle`'s attribute plane from snapped vertices), and every f32
/// coordinate we can form (`to_unit·pixel`, the attribute plane, or the texel-space `M`+`v2.z`) lands a
/// sub-ULP (~5e-8) short of WARP's on pixels whose `u` falls exactly on a texel-weight boundary (e.g.
/// `canvas-radial-000` (30,13): exact `u = 27.125`, ours `27.124996` → weight 159 not 160 → B 190 not
/// 191). Closing it needs WARP's VS→rasterizer→PS f32 pipeline bit-for-bit, not a brush-side change.
/// Bitmap-06 is the separate ≤2 WARP rasterizer texcoord (`base + px·step`) gap.
#[test]
fn oracle_near() {
    if d2d_reference::suite_dir().is_none() { return; }
    let mut cases: Vec<(String, i32)> = (0..80).map(|i| (format!("canvas-radial-{i:03}"), 1)).collect();
    cases.push(("brushes-07-radial-extend-clamp".into(), 1));
    cases.push(("10-radial-focal".into(), 1));
    cases.push(("brushes-06-bitmap-brush-clamp".into(), 2));
    for (name, tol) in &cases {
        let scene = read_scene(&scene_path(name)).unwrap();
        let buf = render(&scene).unwrap();
        let want = oracle_bgra(name);
        let maxe = buf.iter().zip(&want).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap();
        assert!(maxe <= *tol, "{name}: max channel error {maxe} exceeds the known ±{tol} WARP coordinate-interpolation floor");
    }
}
