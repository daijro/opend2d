//! Phase 2C blend-family conformance: run the pure pixel math in `d2d::effects::builtin::blend`
//! over each captured scene's input bitmaps and compare byte-for-byte against the native WARP
//! oracle in `conformance/oracle/win11-26100`. Each scene clears to transparent, uploads 4×4
//! premultiplied input bitmap(s), applies one effect and copies it 1:1 to the origin, so the
//! top-left 4×4 of the oracle is exactly the premultiplied effect output rounded to UNORM8.
//!
//! This bypasses the effect-graph engine (phase 2B, not yet on main): it tests the math directly,
//! as the task requires. Registration/engine dispatch is added once 2B merges.

use d2d::effects::builtin::blend;
use d2d_reference::{read_scene, Op, Value};
use std::collections::HashMap;
use std::path::PathBuf;

const BLEND: &str = "81c5b77b-13f8-4cdd-ad20-c890547ac65d";
const ARITH: &str = "fc151437-049a-4784-a24a-f1c4daf20987";
const CROSS: &str = "12f575e8-4db1-485f-9a84-03a07dd3829f";
const OPAC: &str = "811d79a4-de28-4454-8094-c64685f8bd4c";

fn oracle(name: &str) -> Vec<u8> {
    let p = d2d_reference::suite_dir().unwrap().join("oracle/win11-26100").join(format!("{name}.bgra"));
    std::fs::read(&p).unwrap_or_else(|e| panic!("read oracle {}: {e}", p.display()))
}
fn scene_file(name: &str) -> PathBuf {
    d2d_reference::suite_dir().unwrap().join("scenes").join(format!("{name}.json"))
}

/// Straight RGBA8 → premultiplied f32 `[R, G, B, A]`, matching the harness upload
/// (`premult = (v·a + 127) / 255`) and the sampler's UNORM8→f32 (`byte / 255`).
fn upload(rgba: &[u8]) -> Vec<[f32; 4]> {
    rgba.chunks_exact(4)
        .map(|p| {
            let a = p[3] as u32;
            let c = |v: u8| ((v as u32 * a + 127) / 255) as f32 / 255.0;
            [c(p[0]), c(p[1]), c(p[2]), a as f32 / 255.0]
        })
        .collect()
}

fn store(px: [f32; 4]) -> [u8; 4] {
    let q = |v: f32| (v * 255.0).round_ties_even().max(0.0).min(255.0) as u8;
    [q(px[2]), q(px[1]), q(px[0]), q(px[3])] // BGRA
}

fn prop_f32(props: &[d2d_reference::Property], index: u32, default: f32) -> f32 {
    props.iter().find(|p| p.index == index).map_or(default, |p| match p.value {
        Value::Float(f) => f,
        _ => panic!("property {index} not a float"),
    })
}

/// Compute the scene's effect output for pixel `(x, y)` (0..4), premultiplied f32.
fn apply(clsid: &str, props: &[d2d_reference::Property], ins: &[&Vec<[f32; 4]>], i: usize, x: usize, y: usize) -> [f32; 4] {
    match clsid {
        c if c == BLEND => {
            let mode = props.iter().find(|p| p.index == 0).map(|p| match p.value {
                Value::Enum(e) => e,
                Value::Uint32(e) => e,
                _ => panic!("blend mode not enum"),
            }).unwrap_or(0);
            blend::blend(mode, ins[0][i], ins[1][i], [x as f32 + 0.5, y as f32 + 0.5])
        }
        c if c == ARITH => {
            let coeffs = props.iter().find(|p| p.index == 0).map(|p| match p.value {
                Value::Vector4(v) => v,
                _ => panic!("arith coeffs not vector4"),
            }).unwrap();
            let clamp = props.iter().find(|p| p.index == 1).map_or(false, |p| matches!(p.value, Value::Bool(true)));
            blend::arithmetic_composite(coeffs, clamp, ins[0][i], ins[1][i])
        }
        c if c == CROSS => blend::cross_fade(prop_f32(props, 0, 0.0), ins[0][i], ins[1][i]),
        c if c == OPAC => blend::opacity(prop_f32(props, 0, 1.0), ins[0][i]),
        _ => panic!("unexpected clsid {clsid}"),
    }
}

/// Returns `Err(message)` with the first differing pixel, or `Ok(())` when byte-exact.
fn check(name: &str) -> Result<(), String> {
    let scene = read_scene(&scene_file(name)).map_err(|e| e.to_string())?;
    let w = scene.size[0] as usize;
    let mut bitmaps: HashMap<String, Vec<[f32; 4]>> = HashMap::new();
    let mut effect = None;
    for op in &scene.ops {
        match op {
            Op::Bitmap { id, rgba, .. } => {
                bitmaps.insert(id.clone(), upload(rgba));
            }
            Op::Effect { clsid, inputs, properties, .. } => effect = Some((clsid, inputs, properties)),
            _ => {}
        }
    }
    let (clsid, inputs, props) = effect.ok_or("no effect op")?;
    let ins: Vec<&Vec<[f32; 4]>> = inputs.iter().map(|id| &bitmaps[id]).collect();
    let oracle = oracle(name);
    for y in 0..4 {
        for x in 0..4 {
            let i = y * 4 + x;
            let got = store(apply(clsid, props, &ins, i, x, y));
            let o = (y * w + x) * 4;
            let want = &oracle[o..o + 4];
            if got != want {
                return Err(format!("first diff at ({x},{y}) [BGRA]: got {got:?} want {want:?}"));
            }
        }
    }
    Ok(())
}

const BLEND_SCENES: &[&str] = &[
    "fx-color-26-blend-multiply", // mode 0 (captured by the colour task)
    "fxblend-01-screen",
    "fxblend-02-darken",
    "fxblend-03-lighten",
    "fxblend-04-dissolve",
    "fxblend-05-colorburn",
    "fxblend-06-linearburn",
    "fxblend-07-darkercolor",
    "fxblend-08-lightercolor",
    "fxblend-09-colordodge",
    "fxblend-10-lineardodge",
    "fxblend-11-overlay",
    "fxblend-12-softlight",
    "fxblend-13-hardlight",
    "fxblend-14-vividlight",
    "fxblend-15-linearlight",
    "fxblend-16-pinlight",
    "fxblend-17-hardmix",
    "fxblend-18-difference",
    "fxblend-19-exclusion",
    "fxblend-20-hue",
    "fxblend-21-saturation",
    "fxblend-22-color",
    "fxblend-23-luminosity",
    "fxblend-24-subtract",
    "fxblend-25-division",
];
const OTHER_SCENES: &[&str] = &[
    "fx-color-24-arithmetic-composite",
    "fxblend-33-arithmetic-clamp",
    "fxblend-30-crossfade-half",
    "fxblend-31-opacity-half",
    "fxblend-32-opacity-zero",
];

/// Drives every scene, prints a PASS/FAIL/PENDING line each, and fails only if a scene whose
/// oracle exists regresses. Missing oracle (not yet captured) or Dissolve (pending native) are
/// reported, not failed.
#[test]
fn blend_family_conformance() {
    if d2d_reference::suite_dir().is_none() { return; }
    let mut failures = Vec::new();
    for name in BLEND_SCENES.iter().chain(OTHER_SCENES) {
        let oracle_path = d2d_reference::suite_dir().unwrap().join("oracle/win11-26100").join(format!("{name}.bgra"));
        if !oracle_path.exists() {
            println!("PENDING-CAPTURE {name}");
            continue;
        }
        if name.contains("dissolve") {
            println!("PENDING-NATIVE  {name} (Dissolve noise source)");
            continue;
        }
        match check(name) {
            Ok(()) => println!("PASS            {name}"),
            Err(e) => {
                println!("FAIL            {name}: {e}");
                failures.push(name.to_string());
            }
        }
    }
    assert!(failures.is_empty(), "blend scenes not byte-exact: {failures:?}");
}
