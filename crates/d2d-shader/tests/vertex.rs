//! Vertex-shader execution (`Shader::run_vertex`): run the committed fullscreen effect VS and check
//! the `SV_Position` it emits for each `SV_VertexID`. Proves the executor runs `vs_5_0` and returns
//! the output registers, the foundation of the custom-vertex-shader draw path.

use d2d_shader::exec::Sampling;
use d2d_shader::{Env, Shader};

struct NoEnv;
impl Env for NoEnv {
    fn sample(&self, _r: u32, _s: u32, _c: [f32; 4], _m: Sampling) -> [f32; 4] {
        [0.0; 4]
    }
}

#[test]
fn fullscreen_vs_emits_varyings() {
    // Read the committed fullscreen VS at run time (not include_bytes!): the public tree ships
    // without the harness instrumentation blobs, so skip when it is absent.
    let Ok(vs) = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../harness/src/bin/vs_fullscreen.dxbc")) else {
        eprintln!("vs_fullscreen.dxbc not present, skipped");
        return;
    };
    let shader = Shader::parse(&vs).expect("parse vs");
    assert_eq!(shader.program.program_type, 1, "vertex shader");
    let pos = shader.outputs.iter().find(|e| e.system_value == 1).map(|e| e.register as usize).unwrap_or(0);
    let run = |vid: u32| shader.run_vertex(&NoEnv, &[[vid, 0, 0, 0]], &[]);
    // Fullscreen triangle: uv ∈ {(0,0),(2,0),(0,2)}; pos = uv·(2,−2) + (−1,1).
    assert_eq!(run(0)[pos], [-1.0, 1.0, 0.0, 1.0], "vid0 SV_Position");
    assert_eq!(run(1)[pos], [3.0, 1.0, 0.0, 1.0], "vid1 SV_Position");
    assert_eq!(run(2)[pos], [-1.0, -3.0, 0.0, 1.0], "vid2 SV_Position");
}

// Gated: run a captured VS (DXBC) with a captured cbuffer (vscb0) + vertex buffer (vb0, 48B stride)
// and print the per-vertex output registers. D2D_VS_RUN="<dxbc>:<vscb0>:<vb0>".
#[test]
fn captured_vs_run() {
    let Ok(spec) = std::env::var("D2D_VS_RUN") else { return };
    let mut it = spec.rsplitn(3, ':');
    let vb_p = it.next().unwrap();
    let cb_p = it.next().unwrap();
    let dxbc_p = it.next().unwrap();
    let dxbc = std::fs::read(dxbc_p).unwrap();
    let cb = std::fs::read(cb_p).unwrap();
    let vb = std::fs::read(vb_p).unwrap();
    let shader = Shader::parse(&dxbc).expect("parse");
    eprintln!("VS program_type={} n_in_regs(48B/vert=3)", shader.program.program_type);
    let rd = |o: usize| u32::from_le_bytes([vb[o], vb[o + 1], vb[o + 2], vb[o + 3]]);
    for v in 0..6usize {
        let b = v * 48;
        let inputs: Vec<[u32; 4]> = (0..3).map(|r| [rd(b + r * 16), rd(b + r * 16 + 4), rd(b + r * 16 + 8), rd(b + r * 16 + 12)]).collect();
        let out = shader.run_vertex_bufs(&NoEnv, &inputs, &[&cb]);
        let f = |o: &[f32; 4]| format!("[{:.4} {:.4} {:.4} {:.4}]", o[0], o[1], o[2], o[3]);
        eprintln!("  v{v}: {}", out.iter().map(f).collect::<Vec<_>>().join(" "));
        // D2D_VS_RAW=<path>: append each vertex's output regs as raw f32 (little-endian, 4·nregs floats).
        if let Ok(p) = std::env::var("D2D_VS_RAW") {
            use std::io::Write;
            let mut fh = std::fs::OpenOptions::new().create(true).append(true).open(&p).unwrap();
            for o in &out {
                for x in o {
                    fh.write_all(&x.to_le_bytes()).unwrap();
                }
            }
        }
    }
}

// Reproduce the X-blur knee at output px `PX` the WARP way: vs1's per-vertex tap texcoords →
// f32 attribute-plane interpolation → `plane_quant` (normalised u·dim, −0.5, ·256, floor) addressing
// of the value·128 bilinear — vs filters.rs's direct `axisq(cx±off)` on the texel coord. Decides
// whether the frac lands one 1/128 bucket over at the knee (the dropshadow/shadow-only ±1).
// D2D_VS_BLUR="<vs1.dxbc>:<cb0.bin>:<vb0.bin>:<tex1.bin>" (tex1 = 48×48 BGRA8 realized input).
#[test]
fn blur_knee_probe() {
    let Ok(spec) = std::env::var("D2D_VS_BLUR") else { return };
    let mut it = spec.rsplitn(4, ':');
    let tex_p = it.next().unwrap();
    let vb_p = it.next().unwrap();
    let cb_p = it.next().unwrap();
    let dxbc_p = it.next().unwrap();
    let (dxbc, cb, vb, tex) = (std::fs::read(dxbc_p).unwrap(), std::fs::read(cb_p).unwrap(), std::fs::read(vb_p).unwrap(), std::fs::read(tex_p).unwrap());
    let shader = Shader::parse(&dxbc).unwrap();
    let rd = |o: usize| u32::from_le_bytes([vb[o], vb[o + 1], vb[o + 2], vb[o + 3]]);
    // Per-vertex outputs for the 6 verts (o0=clip pos, o1..o4 = tap texcoords).
    let vo: Vec<Vec<[f32; 4]>> = (0..6usize).map(|v| {
        let b = v * 48;
        let inputs: Vec<[u32; 4]> = (0..3).map(|r| [rd(b + r * 16), rd(b + r * 16 + 4), rd(b + r * 16 + 8), rd(b + r * 16 + 12)]).collect();
        shader.run_vertex_bufs(&NoEnv, &inputs, &[&cb])
    }).collect();
    // clip → device px (viewport 512). Quad corners: find TL(min px,min py) and TR(max px,min py).
    let vp = 512.0f32;
    let dpx = |o: &[[f32; 4]]| (o[0][0] + 1.0) * 0.5 * vp;
    let dpy = |o: &[[f32; 4]]| (1.0 - o[0][1]) * 0.5 * vp;
    // Top-edge endpoints: the two verts with the smallest device py; left=min px, right=max px.
    let mut top: Vec<usize> = (0..6).collect();
    top.sort_by(|&a, &b| dpy(&vo[a]).partial_cmp(&dpy(&vo[b])).unwrap());
    let (mut l, mut r) = (top[0], top[1]);
    if dpx(&vo[l]) > dpx(&vo[r]) { std::mem::swap(&mut l, &mut r); }
    let (pxl, pxr) = (dpx(&vo[l]), dpx(&vo[r]));
    eprintln!("quad top edge: left vert px={pxl:.3} right px={pxr:.3}");
    // o.x linear in device px between left,right. For each tap reg (1..=4), eval at device px PX+0.5.
    let px: f32 = std::env::var("PX").ok().and_then(|s| s.parse().ok()).unwrap_or(12.0);
    let dev = px + 0.5;
    let t = (dev - pxl) / (pxr - pxl); // f32 lerp param
    let dim = 48.0f32;
    // cb0 weights: cb0[2].z inner, cb0[3].z outer (float4 regs, 16B each).
    let cbf = |reg: usize, c: usize| f32::from_le_bytes([cb[reg * 16 + c * 4], cb[reg * 16 + c * 4 + 1], cb[reg * 16 + c * 4 + 2], cb[reg * 16 + c * 4 + 3]]);
    let (w_in, w_out) = (cbf(2, 2), cbf(3, 2));
    // tex1 alpha at (x,y) clamped, as value·128 bilinear along x at a fixed interior row (y=20).
    let w = 48usize;
    let ta = |x: i32, y: i32| -> i32 { let x = x.clamp(0, 47); let y = y.clamp(0, 47); tex[(y as usize * w + x as usize) * 4 + 3] as i32 };
    let bil_x = |fx: f32, y: i32| -> f32 {
        // plane_quant addressing: u (normalised) already; here fx = texel coord. value·128 bilinear in x.
        let q = ((fx - 0.5) * 256.0).floor() as i32;
        let (x0, wx) = (q >> 8, q & 0xff);
        let (a, b) = (ta(x0, y), ta(x0 + 1, y));
        let v128 = ((b - a) * wx + (a << 8)) >> 1; // y-weight 0 (centre row)
        (v128 as f32 * (1.0 / 128.0)) * (1.0 / 256.0) + (v128 as f32 * (1.0 / 128.0)) * (1.0 / 65280.0)
    };
    // tap x-texels via (A) plane_quant: u = o_k.x (normalised, f32-lerped) → u·dim ; (B) axisq direct.
    let yrow = 20;
    let tap = |reg: usize, warp: bool| -> f32 {
        let ux = vo[l][reg][0] + t * (vo[r][reg][0] - vo[l][reg][0]); // f32 lerp of normalised texcoord
        let texel = if warp { ux * dim } else { ux * dim }; // same; difference is the lerp space below
        bil_x(texel, yrow)
    };
    // (A) WARP: lerp the NORMALISED texcoord (o.x) in f32, then ·dim.  (B) direct: lerp the TEXEL coord.
    let tap_warp = |reg: usize| -> f32 {
        let ux = vo[l][reg][0] + t * (vo[r][reg][0] - vo[l][reg][0]);
        bil_x(ux * dim, yrow)
    };
    let tap_direct = |reg: usize| -> f32 {
        let tl = vo[l][reg][0] * dim; let tr = vo[r][reg][0] * dim;
        bil_x(tl + t * (tr - tl), yrow)
    };
    let _ = tap;
    // ps0 order: Mul(outer+ = o2) + Mad(inner+ = o1) + Mad(inner- = o3) + Mad(outer- = o4).
    for (name, f) in [("WARP plane_quant (lerp normalised)", &tap_warp as &dyn Fn(usize) -> f32), ("direct (lerp texel)", &tap_direct)] {
        let acc = f(2) * w_out + f(1) * w_in + f(3) * w_in + f(4) * w_out;
        let v = (acc * 255.0).round() as i32; // output alpha (×colour black = alpha)
        eprintln!("{name}: taps[o1..o4]=[{:.5} {:.5} {:.5} {:.5}] acc×255={:.4} -> {v}", f(1), f(2), f(3), f(4), acc * 255.0);
    }
    eprintln!("(native X-blur at x{} = 179; filters.rs value·128 = 178)", px as i32);
}
