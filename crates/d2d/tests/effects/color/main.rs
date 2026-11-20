//! Phase 2C conformance: drive every `fx-color-*` scene's colour/transfer effect through the pure
//! pixel maths in `d2d::effects::builtin::color` and compare byte-for-byte against the native WARP
//! oracle in `conformance/oracle/win11-26100`.
//!
//! The effect-graph engine is phase 2B; until it lands this test replicates the one thing the
//! engine does around the shader for these simple 1:1 effects: premultiply the straight-RGBA8
//! input exactly as the harness uploads it (`(v·a+127)/255`), run the shader body in f32, and store
//! the premultiplied result to the `B8G8R8A8_UNORM` target with WARP's round-ties-even float→UNORM8
//! rule. `draw_image` copies the effect output 1:1 at the origin over the transparent-black clear,
//! so the target equals the effect result in the input's rectangle and 0 elsewhere.

use d2d::effects::builtin::color::{self as fx, AlphaMode, GammaChannel, HdrToneMap, LinearChannel, Pixel, Vignette};
use d2d_reference::{read_scene, Op, Value};
use std::collections::HashMap;
use std::path::PathBuf;


fn scene_path(name: &str) -> PathBuf {
    d2d_reference::suite_dir().unwrap().join("scenes").join(format!("{name}.json"))
}
fn oracle_bgra(name: &str) -> Vec<u8> {
    let p = d2d_reference::suite_dir().unwrap().join("oracle/win11-26100").join(format!("{name}.bgra"));
    std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// Straight RGBA8 pixel → premultiplied f32, matching `harness::premultiply_bgra`.
fn premul_input(px: &[u8]) -> Pixel {
    let a = px[3] as u32;
    let c = |v: u8| ((v as u32 * a + 127) / 255) as f32 / 255.0;
    [c(px[0]), c(px[1]), c(px[2]), px[3] as f32 / 255.0]
}

/// Premultiplied f32 → premultiplied `[B,G,R,A]` UNORM8 (round-ties-even, WARP's float→UNORM8 rule).
fn store(p: Pixel) -> [u8; 4] {
    let q = |c: f32| (c * 255.0).round_ties_even().clamp(0.0, 255.0) as u8;
    [q(p[2]), q(p[1]), q(p[0]), q(p[3])]
}

/// Effect properties keyed by index, with typed accessors that fall back to a default.
struct Props<'a>(HashMap<u32, &'a Value>);
impl<'a> Props<'a> {
    fn f(&self, i: u32, d: f32) -> f32 {
        match self.0.get(&i) {
            Some(Value::Float(v)) => *v,
            _ => d,
        }
    }
    fn b(&self, i: u32, d: bool) -> bool {
        match self.0.get(&i) {
            Some(Value::Bool(v)) => *v,
            _ => d,
        }
    }
    fn e(&self, i: u32, d: u32) -> u32 {
        match self.0.get(&i) {
            Some(Value::Enum(v)) => *v,
            _ => d,
        }
    }
    fn vec2(&self, i: u32, d: [f32; 2]) -> [f32; 2] {
        match self.0.get(&i) {
            Some(Value::Vector2(v)) => *v,
            _ => d,
        }
    }
    fn vec3(&self, i: u32, d: [f32; 3]) -> [f32; 3] {
        match self.0.get(&i) {
            Some(Value::Vector3(v)) => *v,
            _ => d,
        }
    }
    fn vec4(&self, i: u32, d: [f32; 4]) -> [f32; 4] {
        match self.0.get(&i) {
            Some(Value::Vector4(v)) => *v,
            _ => d,
        }
    }
    fn m54(&self, i: u32) -> [f32; 20] {
        match self.0.get(&i) {
            Some(Value::Matrix5x4(v)) => *v,
            _ => panic!("missing matrix5x4 property {i}"),
        }
    }
}

/// Run one scene's effect over its input bitmap(s), returning the full target buffer (128×96 BGRA8,
/// transparent black outside the effect's rectangle), or `None` if the effect is not yet exact.
fn render(name: &str) -> Option<(Vec<u8>, [u32; 2])> {
    let scene = read_scene(&scene_path(name)).unwrap();
    let [tw, th] = scene.size;
    let bitmaps: HashMap<&str, (&[u32; 2], &Vec<u8>)> = scene
        .ops
        .iter()
        .filter_map(|op| match op {
            Op::Bitmap { id, size, rgba, .. } => Some((id.as_str(), (size, rgba))),
            _ => None,
        })
        .collect();
    let effect = scene.ops.iter().find_map(|op| match op {
        Op::Effect { clsid, inputs, properties, .. } => Some((clsid, inputs, properties)),
        _ => None,
    })?;
    let (clsid, inputs, properties) = effect;
    let props = Props(properties.iter().map(|p| (p.index, &p.value)).collect());

    // Inputs premultiplied to f32, in scene order; all effect inputs share the first's size.
    let (size, _) = bitmaps[inputs[0].as_str()];
    let [w, h] = *size;
    let ins: Vec<Vec<Pixel>> = inputs
        .iter()
        .map(|id| {
            let (_, rgba) = bitmaps[id.as_str()];
            rgba.chunks_exact(4).map(premul_input).collect()
        })
        .collect();

    let tag = &clsid[..8];
    let vignette = (tag == "c00c40be").then(|| {
        Vignette::new([0, 0, w as i32, h as i32], props.vec4(0, [0.0, 0.0, 0.0, 1.0]), props.f(1, 0.1), props.f(2, 0.5))
    });
    let hdr = (tag == "7b0b748d").then(|| HdrToneMap::new(props.f(0, 4000.0), props.f(1, 300.0), props.e(2, 0)));

    let pixel = |i: usize, x: u32, y: u32| -> Option<Pixel> {
        let p = ins[0][i];
        Some(match tag {
            "921f03d6" => fx::color_matrix(p, &props.m54(0), AlphaMode::from_u32(props.e(1, 1)), props.b(2, false)),
            "e0c3784d" => fx::invert(p),
            "36dde0eb" => fx::grayscale(p),
            "41251ab7" => fx::luminance_to_alpha(p),
            "b56c8cfa" => fx::exposure(p, props.f(0, 0.0)),
            "8cea8d1e" => fx::brightness(p, props.vec2(0, [1.0, 1.0]), props.vec2(1, [0.0, 0.0])),
            "b648a78a" => fx::contrast(p, props.f(0, 0.0), props.b(1, false)),
            "0f4458ec" => fx::hue_rotation(p, props.f(0, 0.0)),
            "5cb2d9cf" => fx::saturation(p, props.f(0, 0.5)),
            "3a1af410" => fx::sepia(p, props.f(0, 0.5), AlphaMode::from_u32(props.e(1, 1))),
            "36312b17" => fx::tint(p, props.vec4(0, [1.0, 1.0, 1.0, 1.0]), props.b(1, false)),
            "89176087" => fx::temperature_tint(p, props.f(0, 0.0), props.f(1, 0.0)),
            "409444c4" => {
                let ch = std::array::from_fn(|c| GammaChannel {
                    amplitude: props.f(c as u32 * 4, 1.0),
                    exponent: props.f(c as u32 * 4 + 1, 1.0),
                    offset: props.f(c as u32 * 4 + 2, 0.0),
                    disable: props.b(c as u32 * 4 + 3, false),
                });
                fx::gamma_transfer(p, &ch, props.b(16, false))
            }
            "ad47c8fd" => {
                let ch = std::array::from_fn(|c| LinearChannel {
                    y_intercept: props.f(c as u32 * 3, 0.0),
                    slope: props.f(c as u32 * 3 + 1, 1.0),
                    disable: props.b(c as u32 * 3 + 2, false),
                });
                fx::linear_transfer(p, &ch, props.b(12, false))
            }
            "90866fcd" => {
                let def = [0.0f32, 1.0];
                let disable = std::array::from_fn(|c| props.b(c as u32 * 2 + 1, false));
                fx::discrete_transfer(p, [&def, &def, &def, &def], disable, props.b(8, false))
            }
            "5bf818c3" => {
                let def = [0.0f32, 1.0];
                let disable = std::array::from_fn(|c| props.b(c as u32 * 2 + 1, false));
                fx::table_transfer(p, [&def, &def, &def, &def], disable, props.b(8, false))
            }
            "06eab419" => fx::premultiply_effect(p),
            "fb9ac489" => fx::unpremultiply_effect(p),
            "c80ecff0" => fx::alpha_mask(p, ins[1][i]),
            "74c01f5b" => fx::chroma_key(p, props.vec3(0, [0.0, 0.0, 0.0]), props.f(1, 0.1), props.b(2, false)),
            "44a1cadb" => fx::white_level(p, props.f(0, 80.0), props.f(1, 80.0)),
            "c00c40be" => vignette.as_ref().unwrap().pixel(x as i32, y as i32, p),
            "7b0b748d" => hdr.as_ref().unwrap().pixel(p),
            _ => return None,
        })
    };

    let mut target = vec![0u8; (tw * th) as usize * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let out = pixel(i, x, y)?;
            let o = ((y * tw + x) as usize) * 4;
            target[o..o + 4].copy_from_slice(&store(out));
        }
    }
    Some((target, [tw, th]))
}

/// Every colour/transfer scene this task owns that is byte-exact against the oracle.
const EXACT: &[&str] = &[
    "fx-color-01-colormatrix-premul",
    "fx-color-02-colormatrix-straight",
    "fx-color-03-invert",
    "fx-color-04-grayscale",
    "fx-color-05-luminance-to-alpha",
    "fx-color-06-exposure",
    "fx-color-07-brightness",
    "fx-color-08-contrast",
    "fx-color-09-hue-rotation",
    "fx-color-10-saturation",
    "fx-color-11-sepia",
    "fx-color-12-tint",
    "fx-color-13-temperature-tint",
    "fx-color-14-vignette",
    "fx-color-15-gamma-transfer",
    "fx-color-16-linear-transfer",
    "fx-color-17-discrete-transfer",
    "fx-color-18-table-transfer",
    "fx-color-19-premultiply",
    "fx-color-20-unpremultiply",
    "fx-color-21-alpha-mask",
    "fx-color-22-chroma-key",
    "fx-color-23-white-level",
    "fx-color-27-hdr-tonemap",
];

#[test]
fn colour_effects_byte_exact() {
    if d2d_reference::suite_dir().is_none() { return; }
    let mut failures = Vec::new();
    for &name in EXACT {
        let (target, [tw, _]) = render(name).unwrap_or_else(|| panic!("{name}: no effect rendered"));
        let oracle = oracle_bgra(name);
        assert_eq!(target.len(), oracle.len(), "{name}: size mismatch");
        if let Some(i) = (0..target.len()).step_by(4).find(|&i| target[i..i + 4] != oracle[i..i + 4]) {
            let (px, x, y) = (i / 4, (i / 4) as u32 % tw, (i / 4) as u32 / tw);
            failures.push(format!(
                "{name}: first diff at px{px} ({x},{y}) got {:?} want {:?}",
                &target[i..i + 4],
                &oracle[i..i + 4]
            ));
        }
    }
    assert!(failures.is_empty(), "byte-exact regressions:\n{}", failures.join("\n"));
}

// HdrToneMap (`fx-color-27`, CLSID 7b0b748d) is now byte-exact: native HdrToneMap is the full
// Ebner–Fairchild IPT appearance tone map (RGB→LMS→hyperbolic cone response→IPT, rational-Bézier
// roll-off on I, chroma scale, inverse) reproduced in `color::HdrToneMap`. Native defaults are
// inputMaxLuminance=4000, outputMaxLuminance=300, SDR (from `Create`; SDR clamps inMax≥4000). Its
// real HDR domain (FP16/FP32 targets) is only expressible in v2 scenes — see the float note below.
