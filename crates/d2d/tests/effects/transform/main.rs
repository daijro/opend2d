//! Phase 2C conformance: drive each `fx-transform-*` scene for the effects this phase owns
//! (Flood, Atlas, Scale, DPI compensation, 3D transform / perspective, Turbulence, the six
//! lighting effects) and compare byte-for-byte against the captured WARP oracle.
//!
//! The effect-graph engine (2B) is not on main yet, so this test stands in for it: it evaluates
//! the kernel at every target texel (baking in the `draw_image` offset), realizes a premultiplied
//! bitmap, and composites it through the real `DeviceContext` (`Clear` + `DrawImage`, nearest,
//! source-over) exactly as the scenes do. When 2B lands, the kernels register through its engine
//! and the plain `oracle_conformance` test covers these scenes instead.

use d2d::bitmap::{Bitmap, Composite};
use d2d::effects::builtin::transform as fx;
use d2d::{bitmap_target, Device, Factory};
use d2d_reference::{read_scene, Op, Scene, Value};
use std::path::PathBuf;


fn oracle_dir() -> PathBuf {
    d2d_reference::suite_dir().unwrap().join("oracle/win11-26100")
}
fn scene_path(name: &str) -> PathBuf {
    d2d_reference::suite_dir().unwrap().join("scenes").join(format!("{name}.json"))
}

fn prop(props: &[d2d_reference::Property], idx: u32) -> Option<&Value> {
    props.iter().find(|p| p.index == idx).map(|p| &p.value)
}
fn f(props: &[d2d_reference::Property], idx: u32, d: f32) -> f32 {
    match prop(props, idx) {
        Some(Value::Float(v)) => *v,
        _ => d,
    }
}
fn u(props: &[d2d_reference::Property], idx: u32, d: u32) -> u32 {
    match prop(props, idx) {
        Some(Value::Enum(v)) | Some(Value::Uint32(v)) => *v,
        _ => d,
    }
}
fn vec2(props: &[d2d_reference::Property], idx: u32, d: [f32; 2]) -> [f32; 2] {
    match prop(props, idx) {
        Some(Value::Vector2(v)) => *v,
        _ => d,
    }
}
fn vec3(props: &[d2d_reference::Property], idx: u32, d: [f32; 3]) -> [f32; 3] {
    match prop(props, idx) {
        Some(Value::Vector3(v)) => *v,
        _ => d,
    }
}
fn vec4(props: &[d2d_reference::Property], idx: u32, d: [f32; 4]) -> [f32; 4] {
    match prop(props, idx) {
        Some(Value::Vector4(v)) => *v,
        _ => d,
    }
}

fn scale3(c: [f32; 3], k: f32) -> [f32; 3] {
    [c[0] * k, c[1] * k, c[2] * k]
}

/// Build the effect's premultiplied-BGRA8 output, the full target size, with the `draw_image`
/// offset baked into the scene position. `None` ⇒ an effect/mode 2C does not reproduce.
fn effect_output(scene: &Scene) -> Option<(Bitmap, [f32; 2], u32, Composite)> {
    let [w, h] = scene.size;
    let input_bitmap = scene.ops.iter().find_map(|op| match op {
        Op::Bitmap { size, rgba, .. } => Some(Bitmap::from_straight_rgba(size[0], size[1], rgba)),
        _ => None,
    });
    let (clsid, props) = scene.ops.iter().find_map(|op| match op {
        Op::Effect { clsid, properties, .. } => Some((clsid.as_str(), properties.as_slice())),
        _ => None,
    })?;
    let (offset, interp, composite) = scene.ops.iter().find_map(|op| match op {
        Op::DrawImage { offset, interpolation, composite, .. } => Some((*offset, *interpolation, *composite)),
        _ => None,
    })?;
    let mode = match composite {
        0 => Composite::SourceOver,
        _ => return None,
    };
    let (ox, oy) = (-offset[0], -offset[1]);
    let input = input_bitmap.as_ref().map(|b| fx::Input::new(b, 0, 0));
    let interp_of = fx::Interp::from_mode;
    let border_of = fx::Border::from_mode;

    let bitmap = match clsid {
        // Flood
        "61c23c20-ae69-4d8e-94cf-50078df638f2" => {
            fx::to_bitmap(w, h, ox, oy, fx::flood(vec4(props, 0, [0.0, 0.0, 0.0, 1.0])))
        }
        // Atlas
        "913e2be4-fdcf-4fe2-a5f0-2454f14ff408" => {
            let r = vec4(props, 0, [f32::MIN, f32::MIN, f32::MAX, f32::MAX]).map(|v| v.round() as i32);
            let input = input.as_ref()?;
            fx::to_bitmap(w, h, ox, oy, fx::atlas(input, r))
        }
        // Scale
        "9daf9369-3846-4d0e-a44e-0c607934a5d7" => {
            let m = fx::scale_matrix(vec2(props, 0, [1.0, 1.0]), vec2(props, 1, [0.0, 0.0]));
            let input = input.as_ref()?;
            fx::to_bitmap(w, h, ox, oy, fx::affine(input, m, interp_of(u(props, 2, 1)), border_of(u(props, 3, 0))))
        }
        // DPI compensation
        "6c26c5c7-34e0-46fc-9cfd-e5823706e228" => {
            let m = fx::dpi_matrix([scene.dpi, scene.dpi], vec2(props, 2, [96.0, 96.0]));
            let input = input.as_ref()?;
            fx::to_bitmap(w, h, ox, oy, fx::affine(input, m, interp_of(u(props, 0, 1)), border_of(u(props, 1, 0))))
        }
        // 3D transform (explicit 4×4)
        "e8467b04-ec61-4b8a-b5de-d4d73debea5a" => {
            let m = match prop(props, 2) {
                Some(Value::Matrix4x4(m)) => m.map(|v| v as f64),
                _ => return None,
            };
            let input = input.as_ref()?;
            fx::to_bitmap(w, h, ox, oy, fx::projective(input, m, interp_of(u(props, 0, 1)), border_of(u(props, 1, 0))))
        }
        // 3D perspective transform
        "c2844d0b-3d86-46e7-85ba-526c9240f3fb" => {
            let m = fx::perspective_matrix(
                f(props, 2, 1000.0),
                vec2(props, 3, [0.0, 0.0]),
                vec3(props, 4, [0.0, 0.0, 0.0]),
                vec3(props, 5, [0.0, 0.0, 0.0]),
                vec3(props, 6, [0.0, 0.0, 0.0]),
                vec3(props, 7, [0.0, 0.0, 0.0]),
            );
            let input = input.as_ref()?;
            fx::to_bitmap(w, h, ox, oy, fx::projective(input, m, interp_of(u(props, 0, 1)), border_of(u(props, 1, 0))))
        }
        // Turbulence
        "cf2bb6ae-889a-4ad7-ba29-a2fd732c9fc9" => {
            let offset = vec2(props, 0, [0.0, 0.0]);
            let size = vec2(props, 1, [0.0, 0.0]);
            let bf = vec2(props, 2, [0.01, 0.01]);
            let octaves = u(props, 3, 1) as i32;
            let seed = match prop(props, 4) {
                Some(Value::Int32(v)) => *v,
                _ => 0,
            };
            let fractal = u(props, 5, 0) == 0;
            let rect = [
                offset[0].round() as i32,
                offset[1].round() as i32,
                (offset[0] + size[0]).round() as i32,
                (offset[1] + size[1]).round() as i32,
            ];
            let turb = fx::Turbulence::new(seed, bf, octaves, fractal, rect);
            fx::to_bitmap(w, h, ox, oy, |x, y| turb.eval(x, y))
        }
        // Point-diffuse
        "b9e303c3-c08c-4f91-8b7b-38656bc48c20" => {
            let pos = vec3(props, 0, [0.0, 0.0, 0.0]);
            let diffuse = scale3(vec3(props, 3, [1.0, 1.0, 1.0]), f(props, 1, 1.0));
            let ss = f(props, 2, 1.0);
            let input = input.as_ref()?;
            let light = fx::Light::Point { position: pos, surface_scale: ss };
            fx::to_bitmap(w, h, ox, oy, fx::diffuse(input, light, diffuse, ss))
        }
        // Distant-diffuse
        "3e7efd62-a32d-46d4-a83c-5278889ac954" => {
            let dir = fx::distant_direction(f(props, 0, 0.0), f(props, 1, 0.0));
            let diffuse = scale3(vec3(props, 4, [1.0, 1.0, 1.0]), f(props, 2, 1.0));
            let ss = f(props, 3, 1.0);
            let input = input.as_ref()?;
            fx::to_bitmap(w, h, ox, oy, fx::diffuse(input, fx::Light::Distant { direction: dir }, diffuse, ss))
        }
        // Spot-diffuse
        "818a1105-7932-44f4-aa86-08ae7b2f2c93" => {
            let pos = vec3(props, 0, [0.0, 0.0, 0.0]);
            let points_at = vec3(props, 1, [0.0, 0.0, 0.0]);
            let focus = f(props, 2, 1.0);
            let (cos_outer, cos_inner) = fx::spot_cone_cosines(f(props, 3, 90.0));
            let diffuse = scale3(vec3(props, 6, [1.0, 1.0, 1.0]), f(props, 4, 1.0));
            let ss = f(props, 5, 1.0);
            let input = input.as_ref()?;
            let light = fx::Light::Spot { position: pos, surface_scale: ss, axis: fx::spot_axis(pos, points_at), focus, cos_outer, cos_inner };
            fx::to_bitmap(w, h, ox, oy, fx::diffuse(input, light, diffuse, ss))
        }
        // Distant-specular
        "428c1ee5-77b8-4450-8ab5-72219c21abda" => {
            let dir = fx::distant_direction(f(props, 0, 0.0), f(props, 1, 0.0));
            let exp = f(props, 2, 1.0);
            let spec = scale3(vec3(props, 5, [1.0, 1.0, 1.0]), f(props, 3, 1.0));
            let ss = f(props, 4, 1.0);
            let input = input.as_ref()?;
            let light = fx::Light::Distant { direction: fx::distant_halfway(dir) };
            fx::to_bitmap(w, h, ox, oy, fx::specular(input, light, spec, exp, ss))
        }
        _ => return None,
    };
    Some((bitmap, offset, interp, mode))
}

/// Render the scene (clear + the baked effect bitmap drawn at origin) and return BGRA8 pixels.
fn render(scene: &Scene, bitmap: &Bitmap, interp: u32, mode: Composite) -> Vec<u8> {
    let [w, h] = scene.size;
    let factory = Factory::new(false);
    let device: Device = factory.create_device();
    let mut dc = device.create_device_context();
    dc.set_dpi(scene.dpi, scene.dpi);
    dc.set_target_bitmap(bitmap_target(w, h, scene.dpi));
    dc.begin_draw();
    dc.clear(Some([0.0; 4]));
    for op in &scene.ops {
        if let Op::Clear { color } = op {
            dc.clear(Some(*color));
        }
    }
    let composite = match mode { Composite::SourceOver => 0, Composite::SourceCopy => 10, Composite::Plus => 9 };
    dc.draw_image_bitmap(bitmap, [0.0, 0.0], None, interp, composite);
    let _ = dc.end_draw();
    dc.take_target().unwrap().data
}

fn first_diff(got: &[u8], want: &[u8], w: u32) -> Option<String> {
    let mut first = None;
    let (mut count, mut maxd) = (0u32, 0i32);
    for (i, (a, b)) in got.as_chunks::<4>().0.iter().zip(want.as_chunks::<4>().0).enumerate() {
        if a != b {
            count += 1;
            let d = (0..4).map(|c| (a[c] as i32 - b[c] as i32).abs()).max().unwrap();
            maxd = maxd.max(d);
            if first.is_none() {
                let p = i as u32;
                let ch = (0..4).find(|&c| a[c] != b[c]).unwrap();
                first = Some(format!("pixel ({},{}) ch {} got {a:?} want {b:?}", p % w, p / w, ["B", "G", "R", "A"][ch]));
            }
        }
    }
    first.map(|f| format!("{count} px differ (≤{maxd} LSB); first {f}"))
}

const SCENES: &[&str] = &[
    "fx-transform-01-flood-default",
    "fx-transform-02-flood-translucent",
    "fx-transform-08-atlas",
    "fx-transform-11-scale-hqcubic",
    "fx-transform-12-perspective",
    "fx-transform-13-transform3d",
    "fx-transform-14-dpi",
    "fx-transform-15-turbulence-fractal",
    "fx-transform-16-turbulence-turb",
    "fx-transform-17-point-diffuse",
    "fx-transform-18-distant-diffuse",
    "fx-transform-19-spot-diffuse",
    "fx-transform-20-distant-specular",
];

/// Scenes reproduced byte-for-byte; these must never regress. The rest are near-exact (≤2 LSB at
/// interpolation gradients — WARP's per-draw texcoord fixed point, same residual as 1E's
/// DrawBitmap — or at transcendental results, cos/sin/acos/pow) or not yet reproduced
/// (high-quality-cubic Scale: kernel not extracted from WARP). They print their first difference.
const EXACT: &[&str] = &[
    "fx-transform-01-flood-default",
    "fx-transform-02-flood-translucent",
    "fx-transform-08-atlas",
    "fx-transform-15-turbulence-fractal",
    "fx-transform-16-turbulence-turb",
    "fx-transform-17-point-diffuse",
];

#[test]
fn fx_transform_conformance() {
    if d2d_reference::suite_dir().is_none() { return; }
    let (mut pass, mut skip, mut fail) = (0u32, 0u32, 0u32);
    let mut fails = Vec::new();
    let mut exact_regressions = Vec::new();
    for name in SCENES {
        let scene = read_scene(&scene_path(name)).expect("scene");
        let Some((bitmap, _off, interp, mode)) = effect_output(&scene) else {
            println!("SKIP {name}: effect/mode not reproduced");
            skip += 1;
            continue;
        };
        let got = render(&scene, &bitmap, interp, mode);
        let want = std::fs::read(oracle_dir().join(format!("{name}.bgra"))).expect("oracle");
        match first_diff(&got, &want, scene.size[0]) {
            None => {
                println!("PASS {name}");
                pass += 1;
            }
            Some(d) => {
                println!("FAIL {name}: {d}");
                if EXACT.contains(name) {
                    exact_regressions.push(format!("{name}: {d}"));
                }
                fails.push(format!("{name}: {d}"));
                fail += 1;
            }
        }
    }
    println!("\nfx-transform: {pass} pass, {skip} skip, {fail} fail");
    for d in &fails {
        println!("  {d}");
    }
    assert!(exact_regressions.is_empty(), "byte-exact scenes regressed:\n  {}", exact_regressions.join("\n  "));
}
