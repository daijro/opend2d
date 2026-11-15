//! Phase 1B conformance: render the raster/core oracle scenes with `d2d::pixels` and assert
//! the output is byte-identical to the WARP capture in `conformance/oracle/win11-26100`.
//!
//! The scene JSON is parsed with the harness's parser; this file only maps the in-scope ops
//! (Clear, solid brush, antialias/blend/transform state, rectangle and line-path fills) into
//! device space and drives the rasterizer. Scenes needing curves, strokes, gradients, bitmaps,
//! text or effects are out of 1B's scope and not listed here.

use d2d::pixels::{premultiply, Aa, Blend, FillMode, Target};
use d2d_reference::{read_scene, Op, Scene, Segment, Shape};
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

// 3×2 affine [m11,m12,m21,m22,dx,dy]; compose(a, b) applies b then a.
type Mat = [f64; 6];
fn apply(m: &Mat, x: f64, y: f64) -> [f32; 2] {
    [(m[0] * x + m[2] * y + m[4]) as f32, (m[1] * x + m[3] * y + m[5]) as f32]
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

/// Render an in-scope scene into a fresh premultiplied-BGRA buffer. Returns `None` if the scene
/// uses an op/shape outside 1B (so the caller can skip it rather than fail).
fn render(scene: &Scene) -> Option<PixelBuffer> {
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
    let ds = scene.dpi as f64 / 96.0;
    let dp: Mat = [ds, 0.0, 0.0, ds, 0.0, 0.0];
    let mut world: Mat = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let mut aa = Aa::PerPrimitive;
    let mut blend = Blend::SourceOver;
    let mut brushes: HashMap<&str, [u8; 4]> = HashMap::new();
    let mut geoms: HashMap<&str, &Shape> = HashMap::new();
    let mut t = Target::new(&mut buf);

    for op in &scene.ops {
        match op {
            Op::Clear { color } => t.clear(premultiply(*color)),
            Op::Antialias { mode } => aa = if *mode == 1 { Aa::Aliased } else { Aa::PerPrimitive },
            Op::Blend { mode } => {
                blend = match mode {
                    0 => Blend::SourceOver,
                    1 => Blend::Copy,
                    2 => Blend::Min,
                    3 => Blend::Add,
                    4 => Blend::Max,
                    _ => return None,
                }
            }
            Op::Transform { matrix } => world = matrix.map(|v| v as f64),
            Op::Solid { id, color, opacity, transform } => {
                if opacity.is_some() || transform.is_some() {
                    return None; // brush opacity/transform is 1D's concern
                }
                brushes.insert(id, premultiply(*color));
            }
            Op::Geometry { id, shape } => {
                geoms.insert(id, shape);
            }
            Op::Primitive { shape, brush, width } => {
                if width.is_some() {
                    return None; // stroking is geometry's (1C) concern
                }
                let m = compose(&dp, &world);
                fill_shape(&mut t, shape, *brushes.get(brush.as_str())?, aa, blend, &m)?;
            }
            Op::Fill { geometry, brush } => {
                let m = compose(&dp, &world);
                fill_shape(&mut t, geoms.get(geometry.as_str())?, *brushes.get(brush.as_str())?, aa, blend, &m)?;
            }
            _ => return None,
        }
    }
    Some(buf)
}

fn fill_shape(t: &mut Target, shape: &Shape, src: [u8; 4], aa: Aa, blend: Blend, m: &Mat) -> Option<()> {
    match shape {
        Shape::Rect { rect } => {
            let a = apply(m, rect[0] as f64, rect[1] as f64);
            let b = apply(m, rect[2] as f64, rect[3] as f64);
            t.fill_rect([a[0], a[1], b[0], b[1]], src, aa, blend);
        }
        Shape::Path { fill, figures } => {
            let mut figs = Vec::new();
            for f in figures {
                let mut pts = vec![apply(m, f.start[0] as f64, f.start[1] as f64)];
                for s in &f.segments {
                    match s {
                        Segment::Line { to } => pts.push(apply(m, to[0] as f64, to[1] as f64)),
                        _ => return None, // curves belong to geometry (1C)
                    }
                }
                figs.push(pts);
            }
            let fm = if *fill == 1 { FillMode::Winding } else { FillMode::Alternate };
            t.fill_path(&figs, fm, src, aa, blend, true); // path primitive = FillGeometry (pixel-centre shift)
        }
        _ => return None, // rounded/ellipse flatten in geometry (1C)
    }
    Some(())
}

fn first_diff(got: &[u8], want: &[u8], w: u32) -> Option<(u32, u32, [u8; 4], [u8; 4])> {
    for i in (0..got.len()).step_by(4) {
        if got[i..i + 4] != want[i..i + 4] {
            let p = (i / 4) as u32;
            return Some((
                p % w,
                p / w,
                got[i..i + 4].try_into().unwrap(),
                want[i..i + 4].try_into().unwrap(),
            ));
        }
    }
    None
}

/// Byte-exact scenes: Clear, pixel-aligned + fractional rectangles (8×8 coverage), aliased
/// top-left, DPI scaling, the shallow coverage ramp, aliased triangle, and all blend modes.
#[test]
fn oracle_exact() {
    if d2d_reference::suite_dir().is_none() { return; }
    let scenes = [
        "01-clear-alpha",
        "raster-02-pixel-aligned-rect",
        "raster-03-half-pixel-rect",
        "raster-08-subpixel-offsets",
        "02-fractional-rect",
        "03-aliased-copy",
        "core-01-blend-min",
        "core-02-blend-add",
        "core-03-blend-max",
        "core-04-dpi-pixels",
        "raster-01-coverage-ramp",
        "raster-04-aliased-half-pixel-rect",
        "raster-05-aliased-triangle",
        "raster-06-fillmode-star",
        "raster-09-subpixel-slivers",
        "06-fill-rules",
        "1b2-01-sixteenth-offsets",
    ];
    for name in scenes {
        let scene = read_scene(&scene_path(name)).unwrap();
        let buf = render(&scene).unwrap_or_else(|| panic!("{name}: uses an out-of-scope op"));
        let want = oracle_bgra(name);
        assert_eq!(buf.data.len(), want.len(), "{name}: size mismatch");
        if let Some((x, y, g, w)) = first_diff(&buf.data, &want, scene.size[0]) {
            panic!("{name}: first diff at ({x},{y}) got BGRA {g:?} want {w:?}");
        }
    }
}

// raster-07 (thin strokes) and raster-10 (curves) stay out of this set: stroking and curve
// flattening are geometry's (1C), so `render` reports them out of scope. Everything that only
// fills polygons or rectangles with a solid colour — including the steep star, sub-pixel slivers
// and the 1/16-offset Q2 capture — is byte-exact via the native `CHwRasterizer` port in `pixels`.
