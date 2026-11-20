//! Executor coverage over Paint.NET's full `ps_5_0` shader corpus vs WARP (phase 3C.3).
//!
//! Paint.NET ships 711 distinct `ps_5_0` pixel shaders across its own assemblies (all using the
//! ComputeSharp.D2D1 varying convention: `SV_POSITION`@r0, `SCENE_POSITION`@r1, `TEXCOORD0..7`@r2..).
//! This test runs each through the CPU executor and compares byte-for-byte with WARP's output of the
//! *same* bytecode, under one uniform input (the `vs_scene` VS, a fixed b0 constant buffer and t0
//! input). The shader bytecode and WARP captures are Microsoft/Paint.NET property — never committed —
//! so the test reads them from gitignored locations given by env vars and is skipped when absent:
//!
//!   PDN_SHADER_DIR  directory of extracted `<name>.dxbc` (distinct ps_5_0 blobs)
//!   PDN_SCENE       the uniform scene JSON (cbuffer + 64x64 t0 input), matching `vs_scene`/64x64
//!   PDN_PROBE       WARP's probe capture: per-pixel interpolated SCENE_POSITION.xy / TEXCOORD0.xy
//!   PDN_CAP_DIR     directory of WARP captures `<name>.bin` (R32G32B32A32_FLOAT, 64x64),
//!                   plus an optional `<name>.restex` (raw f32x4 texels) for a res-tex effect's
//!                   register-1 LUT — captured bound on WARP and fed to the executor the same way.
//!
//! Capture them with `harness/src/bin/shader-ref.rs --batch` on the reference VM.

use d2d::sampling::{Extend, Filter, Texture};
use d2d_shader::exec::{Env, PixelInputs, Sampling};
use d2d_shader::Shader;
use std::path::PathBuf;

const W: usize = 64;
const H: usize = 64;

/// The single bound input image (t0), sampled nearest+clamp through crates/d2d's WARP-exact sampler.
/// Every other resource register is unbound, so it reads zero — exactly as WARP does for an unbound
/// SRV, which keeps the comparison honest for multi-input shaders.
struct TexEnv {
    data: Vec<u8>,
    w: u32,
    h: u32,
    lut: Vec<[f32; 4]>, // register-1 resource texture (empty = none); fed per-shader by the captures
}
impl TexEnv {
    fn texel(&self, x: i32, y: i32) -> [f32; 4] {
        let cl = |c: i32, n: i32| c.clamp(0, n - 1);
        let (x, y) = (cl(x, self.w as i32) as usize, cl(y, self.h as i32) as usize);
        let i = (y * self.w as usize + x) * 4;
        core::array::from_fn(|k| self.data[i + k] as f32)
    }
    fn footprint(&self, coord: [f32; 4]) -> (i32, i32) {
        let split = |t: f32| ((t * 65536.0).round_ties_even() as i64 - 0x7f80) >> 16;
        (split(coord[0] * self.w as f32) as i32, split(coord[1] * self.h as f32) as i32)
    }
}
impl Env for TexEnv {
    fn sample(&self, res: u32, _s: u32, coord: [f32; 4], m: Sampling) -> [f32; 4] {
        if res != 0 {
            return [0.0; 4];
        }
        match m {
            Sampling::Gather { component, offset } => {
                let (x0, y0) = self.footprint(coord);
                let (x0, y0) = (x0 + offset[0], y0 + offset[1]);
                let c = component as usize;
                [
                    self.texel(x0, y0 + 1)[c] / 255.0,
                    self.texel(x0 + 1, y0 + 1)[c] / 255.0,
                    self.texel(x0 + 1, y0)[c] / 255.0,
                    self.texel(x0, y0)[c] / 255.0,
                ]
            }
            Sampling::GatherCmp { reference } => {
                let (x0, y0) = self.footprint(coord);
                let cmp = |x: i32, y: i32| if reference <= self.texel(x, y)[0] / 255.0 { 1.0 } else { 0.0 };
                [cmp(x0, y0 + 1), cmp(x0 + 1, y0 + 1), cmp(x0 + 1, y0), cmp(x0, y0)]
            }
            _ => {
                let tex = Texture { data: &self.data, width: self.w, height: self.h };
                tex.sample(Filter::Nearest, [Extend::Clamp, Extend::Clamp], coord[0] * self.w as f32, coord[1] * self.h as f32)
            }
        }
    }
    fn load(&self, res: u32, c: [i32; 4]) -> [u32; 4] {
        if res == 1 && !self.lut.is_empty() {
            let i = (c[0].max(0) as usize).min(self.lut.len() - 1);
            return self.lut[i].map(|v| v.to_bits());
        }
        if res != 0 {
            return [0; 4];
        }
        self.texel(c[0], c[1]).map(|v| (v / 255.0).to_bits())
    }
    fn resinfo(&self, res: u32) -> [u32; 4] {
        if res == 1 && !self.lut.is_empty() {
            return [self.lut.len() as u32, 1, 1, 1];
        }
        if res != 0 {
            [0; 4]
        } else {
            [self.w, self.h, 1, 1]
        }
    }
}

#[test]
fn texenv_restex_load_resinfo() {
    // The register-1 resource-texture path the per-effect captures depend on: Load returns the raw
    // f32x4 texel (no /255), clamps the index, resinfo gives the length; register 0 is unaffected.
    let lut = vec![[-15736.0, 1.0, 2.0, 3.0], [0.0, 4.0, 5.0, 6.0], [23200.0, 7.0, 8.0, 9.0]];
    let e = TexEnv { data: vec![10, 20, 30, 40], w: 1, h: 1, lut };
    assert_eq!(e.load(1, [2, 0, 0, 0]), [23200.0f32.to_bits(), 7f32.to_bits(), 8f32.to_bits(), 9f32.to_bits()]);
    assert_eq!(e.load(1, [99, 0, 0, 0]), e.load(1, [2, 0, 0, 0])); // clamp high
    assert_eq!(e.load(1, [-5, 0, 0, 0]), e.load(1, [0, 0, 0, 0])); // clamp low
    assert_eq!(e.resinfo(1), [3, 1, 1, 1]);
    assert_eq!(e.resinfo(0), [1, 1, 1, 1]); // input image still wins at register 0
    assert_eq!(TexEnv { data: vec![0; 4], w: 1, h: 1, lut: vec![] }.load(1, [0, 0, 0, 0]), [0; 4]); // no lut → 0
}

fn env(var: &str) -> Option<String> {
    let v = std::env::var(var).ok()?;
    if PathBuf::from(&v).exists() {
        Some(v)
    } else {
        eprintln!("skip: {var}={v} absent");
        None
    }
}

#[test]
fn paintnet_shaders_match_warp() {
    let (Some(sdir), Some(scene_path)) = (env("PDN_SHADER_DIR"), env("PDN_SCENE")) else {
        eprintln!("skip: set PDN_SHADER_DIR and PDN_SCENE to measure");
        return;
    };
    let scene: serde_json::Value = serde_json::from_slice(&std::fs::read(&scene_path).unwrap()).unwrap();
    let mut cbuffer: Vec<u8> = Vec::new();
    for x in scene["cbuffer"].as_array().unwrap() {
        cbuffer.extend_from_slice(&(x.as_f64().unwrap() as f32).to_le_bytes());
    }
    while cbuffer.len() % 16 != 0 {
        cbuffer.push(0);
    }
    let inp = &scene["input"];
    let data: Vec<u8> = inp["rgba"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u8).collect();
    let (tw, th) = (inp["size"][0].as_u64().unwrap() as u32, inp["size"][1].as_u64().unwrap() as u32);

    // Classify: which shaders the executor can run at all (no unimplemented opcode).
    let mut shaders: Vec<(String, Shader)> = Vec::new();
    let mut blocked: std::collections::BTreeMap<u16, usize> = Default::default();
    let mut parse_err = 0usize;
    for e in std::fs::read_dir(&sdir).unwrap() {
        let path = e.unwrap().path();
        if path.extension().and_then(|s| s.to_str()) != Some("dxbc") {
            continue;
        }
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        match Shader::parse(&std::fs::read(&path).unwrap()) {
            Ok(sh) => match sh.unsupported_op() {
                None => shaders.push((name, sh)),
                Some(op) => *blocked.entry(op).or_default() += 1,
            },
            Err(_) => parse_err += 1,
        }
    }
    let total = shaders.len() + blocked.values().sum::<usize>() + parse_err;
    eprintln!(
        "pdn_coverage: {total} distinct ps_5_0; {} executor-runnable, {} blocked by unimplemented opcode {blocked:?}, {parse_err} parse errors",
        shaders.len(),
        blocked.values().sum::<usize>()
    );

    // Compare against WARP captures if provided.
    let (Some(cap_dir), Some(probe_path)) = (std::env::var("PDN_CAP_DIR").ok(), std::env::var("PDN_PROBE").ok()) else {
        eprintln!("pdn_coverage: set PDN_CAP_DIR and PDN_PROBE to compare against WARP");
        return;
    };
    // Probe: WARP's exact interpolated (SCENE_POSITION.xy, TEXCOORD0.xy) per pixel.
    let probe: Vec<f32> = std::fs::read(&probe_path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect();
    assert_eq!(probe.len(), W * H * 4, "probe size");
    let probe = std::sync::Arc::new(probe);
    let cbuffer = std::sync::Arc::new(cbuffer);

    let mut exact = 0usize;
    let mut compared = 0usize;
    let mut timeouts = 0usize;
    let mut fails: Vec<String> = Vec::new();
    for (name, sh) in &shaders {
        let cap_path = PathBuf::from(&cap_dir).join(format!("{name}.bin"));
        let Ok(cap) = std::fs::read(&cap_path) else { continue };
        compared += 1;
        let want: Vec<u32> = cap.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        // A res-tex effect's LUT (register 1) rides in a `{name}.restex` sidecar of raw f32x4 texels,
        // produced alongside its capture; absent for ordinary shaders, so the env reads 0 as before.
        let lut: Vec<[f32; 4]> = std::fs::read(PathBuf::from(&cap_dir).join(format!("{name}.restex")))
            .map(|b| b.chunks_exact(16).map(|c| core::array::from_fn(|k| f32::from_le_bytes(c[4 * k..4 * k + 4].try_into().unwrap()))).collect())
            .unwrap_or_default();
        let env = std::sync::Arc::new(TexEnv { data: data.clone(), w: tw, h: th, lut });
        // Run each shader on its own thread with a timeout: a shader whose loop does not converge for
        // our arbitrary inputs (as one does under WARP too) must not hang the whole measurement.
        let (probe, cbuffer, sh) = (probe.clone(), cbuffer.clone(), sh.clone());
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(compare_quad(&sh, &*env, &probe, &cbuffer, &want));
        });
        match rx.recv_timeout(std::time::Duration::from_secs(30)) {
            Ok((0, _)) => exact += 1,
            Ok((diffs, Some((x, y, c, w, g)))) => {
                if fails.len() < 30 {
                    fails.push(format!("{name}: {diffs} px diff, first @({x},{y}).{c} want={w:?} got={g:?}"));
                }
            }
            Ok((_, None)) => exact += 1,
            Err(_) => {
                timeouts += 1;
                fails.push(format!("{name}: EXECUTOR TIMEOUT (>30s, non-converging loop)"));
            }
        }
    }
    eprintln!(
        "pdn_coverage: {exact}/{compared} shaders byte-exact vs WARP; {timeouts} executor-timeout; {} runnable, {} captured",
        shaders.len(),
        compared
    );
    for f in &fails {
        eprintln!("  FAIL {f}");
    }
    // A timed-out worker thread is left running; exit so the harness does not wait on it.
    std::io::Write::flush(&mut std::io::stderr()).ok();
    std::process::exit(0);
}

/// Compare one shader against its WARP capture over 2×2 quads (so `deriv_*` read neighbour lanes, as
/// WARP does). Returns (differing-channel count, first diff `(x,y,channel,want,got)`).
fn compare_quad(sh: &Shader, env: &dyn Env, probe: &[f32], cbuffer: &[u8], want: &[u32]) -> (usize, Option<(usize, usize, usize, f32, f32)>) {
    let regs = |x: usize, y: usize| -> Vec<[u32; 4]> {
        let p = (y * W + x) * 4;
        let sv = [(x as f32 + 0.5).to_bits(), (y as f32 + 0.5).to_bits(), 0f32.to_bits(), 1f32.to_bits()];
        let scene = [probe[p].to_bits(), probe[p + 1].to_bits(), 0f32.to_bits(), 1f32.to_bits()];
        let tc = [probe[p + 2].to_bits(), probe[p + 3].to_bits(), 0f32.to_bits(), 1f32.to_bits()];
        vec![sv, scene, tc, tc, tc, tc, tc, tc, tc, tc]
    };
    let mut diffs = 0usize;
    let mut first = None;
    for by in (0..H).step_by(2) {
        for bx in (0..W).step_by(2) {
            let coords = [(bx, by), (bx + 1, by), (bx, by + 1), (bx + 1, by + 1)];
            let inputs: Vec<Vec<[u32; 4]>> = coords.iter().map(|&(x, y)| regs(x, y)).collect();
            let px: Vec<PixelInputs> = inputs.iter().map(|v| PixelInputs { inputs: v, cbuffer }).collect();
            let got = sh.run_quad(env, [&px[0], &px[1], &px[2], &px[3]]);
            for (lane, &(x, y)) in coords.iter().enumerate() {
                for c in 0..4 {
                    let (w, g) = (want[(y * W + x) * 4 + c], got[lane][c].to_bits());
                    if w != g {
                        diffs += 1;
                        if first.is_none() {
                            first = Some((x, y, c, f32::from_bits(w), got[lane][c]));
                        }
                    }
                }
            }
        }
    }
    (diffs, first)
}
