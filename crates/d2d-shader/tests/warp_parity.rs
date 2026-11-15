//! Byte-exact parity of the CPU executor against WARP captures (phase 3C.1).
//!
//! For each `scenes/shader-NN-*.{json,dxbc}` the executor runs per output pixel with the
//! pixel-centre `SV_Position` the fullscreen-triangle VS supplies, and its float output is compared
//! bit-for-bit with the `R32G32B32A32_FLOAT` capture WARP produced (`shader-ref`, run on the VM).
//! The arithmetic/comparison/conversion/flow-control shaders must match exactly; the transcendental
//! shader is reported (WARP's `exp`/`log`/`rsq`/`rcp` approximations are the documented residual).

use d2d::sampling::{Extend, Filter, Texture};
use d2d_shader::exec::{self, Env, PixelInputs, Sampling, UavDesc};
use d2d_shader::sm4::Mem;
use d2d_shader::Shader;
use std::path::PathBuf;

/// Conformance suite root (the `conformance/suite` submodule): the `D2D_SUITE` override, else the
/// in-tree submodule, else the main checkout's copy. `None` ⇒ absent (public clone), so the compare
/// helpers skip and every byte-exact assert passes.
fn suite_dir() -> Option<PathBuf> {
    d2d_reference::suite_dir()
}
fn root() -> PathBuf {
    suite_dir().expect("conformance suite present (compare helpers gate on suite_dir)")
}

/// An environment with no textures (the arithmetic shaders do not sample).
struct NoTextures;
impl Env for NoTextures {
    fn sample(&self, _r: u32, _s: u32, _c: [f32; 4], _m: Sampling) -> [f32; 4] {
        [0.0; 4]
    }
}

/// A single input texture sampled through crates/d2d's WARP-exact `sampling.rs`. The shader passes
/// a normalized coordinate; WARP (and this) scale it by the texture size into texel space.
struct TexEnv {
    data: Vec<u8>,
    w: u32,
    h: u32,
    filter: Filter,
    extend: Extend,
}
impl TexEnv {
    /// Raw UNORM8 byte values (0..=255) as floats; the /255 normalisation is deferred to the end.
    fn texel(&self, x: i32, y: i32) -> [f32; 4] {
        let cl = |c: i32, n: i32| c.clamp(0, n - 1);
        let (x, y) = (cl(x, self.w as i32) as usize, cl(y, self.h as i32) as usize);
        let i = (y * self.w as usize + x) * 4;
        core::array::from_fn(|k| self.data[i + k] as f32)
    }
}
impl TexEnv {
    /// Bilinear-footprint corner `(x0, y0)` for a normalized coord, matching `sampling.rs`'s split
    /// (a 16.16 coordinate biased by -0x7f80).
    fn footprint(&self, coord: [f32; 4]) -> (i32, i32) {
        let split = |t: f32| ((t * 65536.0).round_ties_even() as i64 - 0x7f80) >> 16;
        (split(coord[0] * self.w as f32) as i32, split(coord[1] * self.h as f32) as i32)
    }
}

impl Env for TexEnv {
    fn sample(&self, _r: u32, _s: u32, coord: [f32; 4], m: Sampling) -> [f32; 4] {
        // gather4 / gather4_po: the chosen channel of the four footprint texels, in D3D order
        // (.x lower-left, .y lower-right, .z upper-right, .w upper-left), raw texel bytes / 255. The
        // integer `offset` (gather4_po) shifts the footprint corner.
        if let Sampling::Gather { component, offset } = m {
            let (x0, y0) = self.footprint(coord);
            let (x0, y0) = (x0 + offset[0], y0 + offset[1]);
            let c = component as usize;
            return [
                self.texel(x0, y0 + 1)[c] / 255.0,
                self.texel(x0 + 1, y0 + 1)[c] / 255.0,
                self.texel(x0 + 1, y0)[c] / 255.0,
                self.texel(x0, y0)[c] / 255.0,
            ];
        }
        // gather4_c: each footprint texel's red channel compared against the reference (0/1), same
        // D3D order; comparison direction matches the scene's sampler (LESS_EQUAL here).
        if let Sampling::GatherCmp { reference } = m {
            let (x0, y0) = self.footprint(coord);
            let cmp = |x: i32, y: i32| if reference <= self.texel(x, y)[0] / 255.0 { 1.0 } else { 0.0 };
            return [cmp(x0, y0 + 1), cmp(x0 + 1, y0 + 1), cmp(x0 + 1, y0), cmp(x0, y0)];
        }
        // sample_c (point, LESS_EQUAL): compare the reference against the nearest texel's red; 0/1.
        if let Sampling::Cmp { reference, .. } = m {
            let tex = Texture { data: &self.data, width: self.w, height: self.h };
            let t = tex.sample(Filter::Nearest, [self.extend, self.extend], coord[0] * self.w as f32, coord[1] * self.h as f32);
            return [if reference <= t[0] { 1.0 } else { 0.0 }; 4];
        }
        match self.filter {
            f @ (Filter::Nearest | Filter::Cubic) => {
                let tex = Texture { data: &self.data, width: self.w, height: self.h };
                tex.sample(f, [self.extend, self.extend], coord[0] * self.w as f32, coord[1] * self.h as f32)
            }
            Filter::Linear => {
                // WARP's fixed-point sub-texel weight (sampling.rs's `split`: the 8-bit field of a
                // 16.16 coord biased by -0x7f80), but blended in float (the shader writes to a float
                // target, so the filter is not quantised to UNORM8 as the DrawBitmap path is).
                let split = |t: f32| -> (i32, f32) {
                    let c = (t * 65536.0).round_ties_even() as i64 - 0x7f80;
                    ((c >> 16) as i32, ((c >> 8) & 0xff) as f32 / 256.0)
                };
                let (x0, fx) = split(coord[0] * self.w as f32);
                let (y0, fy) = split(coord[1] * self.h as f32);
                let c00 = self.texel(x0, y0);
                let c10 = self.texel(x0 + 1, y0);
                let c01 = self.texel(x0, y0 + 1);
                let c11 = self.texel(x0 + 1, y0 + 1);
                core::array::from_fn(|k| {
                    let a = c00[k] + (c10[k] - c00[k]) * fx;
                    let b = c01[k] + (c11[k] - c01[k]) * fx;
                    (a + (b - a) * fy) / 255.0
                })
            }
        }
    }

    fn resinfo(&self, _r: u32) -> [u32; 4] {
        // (width, height, array-size/depth, mip count); this texture is single-mip.
        [self.w, self.h, 1, 1]
    }
}

/// A 1-D resource-texture LUT bound at register 1 (RGBA32F texels), read by a shader's `ld`/`resinfo`
/// — the executor's resource-register path (the gap crates/d2d's pixel draw transform and d2d-ref
/// both had to close). `sample` is unused: a resource texture is index-loaded, not filtered.
struct LutEnv {
    lut: Vec<[f32; 4]>,
}
impl Env for LutEnv {
    fn sample(&self, _r: u32, _s: u32, _c: [f32; 4], _m: Sampling) -> [f32; 4] {
        [0.0; 4]
    }
    fn load(&self, r: u32, c: [i32; 4]) -> [u32; 4] {
        if r == 1 && !self.lut.is_empty() {
            self.lut[(c[0].max(0) as usize).min(self.lut.len() - 1)].map(f32::to_bits)
        } else {
            [0; 4]
        }
    }
    fn resinfo(&self, r: u32) -> [u32; 4] {
        if r == 1 {
            [self.lut.len() as u32, 1, 1, 1]
        } else {
            [0; 4]
        }
    }
}

fn make_env(scene: &serde_json::Value) -> Box<dyn Env> {
    if let Some(rt) = scene.get("resource_texture").filter(|t| !t.is_null()) {
        let lut = rt["data"]
            .as_array()
            .unwrap()
            .chunks_exact(4)
            .map(|c| core::array::from_fn(|i| c[i].as_f64().unwrap() as f32))
            .collect::<Vec<[f32; 4]>>();
        return Box::new(LutEnv { lut });
    }
    match scene.get("input").filter(|i| !i.is_null()) {
        None => Box::new(NoTextures),
        Some(i) => {
            let w = i["size"][0].as_u64().unwrap() as u32;
            let h = i["size"][1].as_u64().unwrap() as u32;
            let data = i["rgba"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u8).collect();
            let filter = if i["filter"].as_str() == Some("linear") { Filter::Linear } else { Filter::Nearest };
            let extend = match i["extend"].as_str() {
                Some("wrap") => Extend::Wrap,
                Some("mirror") => Extend::Mirror,
                _ => Extend::Clamp,
            };
            Box::new(TexEnv { data, w, h, filter, extend })
        }
    }
}

struct Stats {
    pixels: usize,
    channel_mismatches: usize,
    first_diff: Option<(usize, usize, usize, f32, f32)>,
    max_ulp: u32,
    per_channel: [usize; 4],
    per_channel_max_ulp: [u32; 4],
    per_channel_max_abs: [f32; 4],
}
impl Stats {
    /// A zero / byte-exact result — returned when the suite is absent so every assert passes (skip).
    fn skip() -> Self {
        Stats { pixels: 0, channel_mismatches: 0, first_diff: None, max_ulp: 0, per_channel: [0; 4], per_channel_max_ulp: [0; 4], per_channel_max_abs: [0.0; 4] }
    }
}

/// Run the executor over a scene and compare each channel's raw bits against the capture.
fn compare(name: &str) -> Stats {
    if suite_dir().is_none() {
        return Stats::skip();
    }
    let root = root();
    let scene: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join(format!("scenes/{name}.json"))).unwrap()).unwrap();
    let w = scene["size"][0].as_u64().unwrap() as usize;
    let h = scene["size"][1].as_u64().unwrap() as usize;
    let mut cbuffer: Vec<u8> = Vec::new();
    if let Some(arr) = scene["cbuffer"].as_array() {
        for x in arr {
            cbuffer.extend_from_slice(&(x.as_f64().unwrap() as f32).to_le_bytes());
        }
        while cbuffer.len() % 16 != 0 {
            cbuffer.push(0);
        }
    }
    // A scene may name a shared pixel shader (`"shader"`); default to its own stem.
    let stem = scene["shader"].as_str().unwrap_or(name);
    let shader = Shader::parse(&std::fs::read(root.join(format!("scenes/{stem}.dxbc"))).unwrap()).unwrap();

    let cap = std::fs::read(root.join(format!("oracle/win11-26100/{name}.bin"))).unwrap();
    assert_eq!(cap.len(), w * h * 16, "{name}: capture size");
    let want: Vec<f32> = cap.chunks_exact(4).map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect();

    let env = make_env(&scene);
    let mut st = Stats {
        pixels: w * h,
        channel_mismatches: 0,
        first_diff: None,
        max_ulp: 0,
        per_channel: [0; 4],
        per_channel_max_ulp: [0; 4],
        per_channel_max_abs: [0.0; 4],
    };
    for y in 0..h {
        for x in 0..w {
            // SV_Position: pixel centre, z=0, w=1 (the fullscreen quad is at z=0, w_clip=1).
            let sv = [(x as f32 + 0.5).to_bits(), (y as f32 + 0.5).to_bits(), 0.0f32.to_bits(), 1.0f32.to_bits()];
            let px = PixelInputs { inputs: &[sv], cbuffer: &cbuffer };
            let got = exec::run(&shader, &*env, &px);
            for c in 0..4 {
                let w_bits = want[(y * w + x) * 4 + c].to_bits();
                let g_bits = got[c].to_bits();
                if w_bits != g_bits {
                    st.channel_mismatches += 1;
                    st.per_channel[c] += 1;
                    let ulp = (w_bits as i64 - g_bits as i64).unsigned_abs() as u32;
                    st.max_ulp = st.max_ulp.max(ulp);
                    st.per_channel_max_ulp[c] = st.per_channel_max_ulp[c].max(ulp);
                    let abs = (want[(y * w + x) * 4 + c] - got[c]).abs();
                    st.per_channel_max_abs[c] = st.per_channel_max_abs[c].max(abs);
                    if st.first_diff.is_none() {
                        st.first_diff = Some((x, y, c, want[(y * w + x) * 4 + c], got[c]));
                    }
                }
            }
        }
    }
    st
}

fn report(name: &str) -> Stats {
    let st = compare(name);
    eprintln!(
        "{name}: {}/{} channels differ (max {} ULP){}",
        st.channel_mismatches,
        st.pixels * 4,
        st.max_ulp,
        match st.first_diff {
            Some((x, y, c, w, g)) => format!(
                "; per-channel {:?} (max ULP {:?}); first @({x},{y}).{c} want={w:?} got={g:?}",
                st.per_channel, st.per_channel_max_ulp
            ),
            None => " — byte-exact".into(),
        }
    );
    st
}

/// Like [`compare`] but runs the shader over 2×2 quads (lanes TL, TR, BL, BR) so `deriv_*` ops read
/// neighbour lanes. Requires even dimensions.
fn compare_quad(name: &str) -> Stats {
    if suite_dir().is_none() {
        return Stats::skip();
    }
    let root = root();
    let scene: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join(format!("scenes/{name}.json"))).unwrap()).unwrap();
    let w = scene["size"][0].as_u64().unwrap() as usize;
    let h = scene["size"][1].as_u64().unwrap() as usize;
    let mut cbuffer: Vec<u8> = Vec::new();
    if let Some(arr) = scene["cbuffer"].as_array() {
        for x in arr {
            cbuffer.extend_from_slice(&(x.as_f64().unwrap() as f32).to_le_bytes());
        }
        while cbuffer.len() % 16 != 0 {
            cbuffer.push(0);
        }
    }
    let shader = Shader::parse(&std::fs::read(root.join(format!("scenes/{name}.dxbc"))).unwrap()).unwrap();
    let cap = std::fs::read(root.join(format!("oracle/win11-26100/{name}.bin"))).unwrap();
    let want: Vec<f32> = cap.chunks_exact(4).map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect();
    let env = make_env(&scene);
    let mut st = Stats { pixels: w * h, channel_mismatches: 0, first_diff: None, max_ulp: 0, per_channel: [0; 4], per_channel_max_ulp: [0; 4], per_channel_max_abs: [0.0; 4] };
    let sv = |x: usize, y: usize| [(x as f32 + 0.5).to_bits(), (y as f32 + 0.5).to_bits(), 0.0f32.to_bits(), 1.0f32.to_bits()];
    for by in (0..h).step_by(2) {
        for bx in (0..w).step_by(2) {
            // Lane order: TL, TR, BL, BR.
            let coords = [(bx, by), (bx + 1, by), (bx, by + 1), (bx + 1, by + 1)];
            let inputs: Vec<_> = coords.iter().map(|&(x, y)| [sv(x, y)]).collect();
            let px: Vec<_> = inputs.iter().map(|v| PixelInputs { inputs: v, cbuffer: &cbuffer }).collect();
            let got = shader.run_quad(&*env, [&px[0], &px[1], &px[2], &px[3]]);
            for (lane, &(x, y)) in coords.iter().enumerate() {
                for c in 0..4 {
                    let w_bits = want[(y * w + x) * 4 + c].to_bits();
                    let g_bits = got[lane][c].to_bits();
                    if w_bits != g_bits {
                        st.channel_mismatches += 1;
                        st.per_channel[c] += 1;
                        let ulp = (w_bits as i64 - g_bits as i64).unsigned_abs() as u32;
                        st.max_ulp = st.max_ulp.max(ulp);
                        st.per_channel_max_ulp[c] = st.per_channel_max_ulp[c].max(ulp);
                        if st.first_diff.is_none() {
                            st.first_diff = Some((x, y, c, want[(y * w + x) * 4 + c], got[lane][c]));
                        }
                    }
                }
            }
        }
    }
    st
}

#[test]
fn int_completion_is_byte_exact() {
    assert_eq!(report("shader2-01-intops").channel_mismatches, 0);
}

#[test]
fn half_conversions_are_byte_exact() {
    assert_eq!(report("shader2-02-f16").channel_mismatches, 0);
}

#[test]
fn doubles_are_byte_exact() {
    assert_eq!(report("shader2-03-double").channel_mismatches, 0);
}

#[test]
fn derivatives_are_byte_exact() {
    let st = compare_quad("shader2-04-deriv");
    eprintln!("shader2-04-deriv (quad): {}/{} channels differ (max {} ULP)", st.channel_mismatches, st.pixels * 4, st.max_ulp);
    assert_eq!(st.channel_mismatches, 0);
}

#[test]
fn sample_variants_are_byte_exact() {
    assert_eq!(report("shader2-05-sample").channel_mismatches, 0);
}

#[test]
fn gather4_is_byte_exact() {
    assert_eq!(report("shader2-06-gather").channel_mismatches, 0);
}

#[test]
fn resinfo_is_byte_exact() {
    assert_eq!(report("shader2-07-resinfo").channel_mismatches, 0);
}

#[test]
fn sincos_is_byte_exact() {
    // sin/cos over a wide input sweep through WARP's `sincos` kernel (d2d::warpmath::sincos).
    assert_eq!(report("shader3-01-sincos").channel_mismatches, 0);
}

#[test]
fn switch_is_byte_exact() {
    // switch/case/default/endswitch with diverging per-pixel selectors: empty labels falling through,
    // a conditional break inside a case, a nested switch, a switch inside a loop (G15: BoltBait Black
    // and White+). fxc 10.0.26100 /O3, WARP capture.
    assert_eq!(report("shader3-06-switch").channel_mismatches, 0);
}

#[test]
fn restex_is_byte_exact() {
    // A 1-D LUT bound at t1, read per column by `resinfo`+`ld` (no input texture). Guards the
    // executor's resource-register load/resinfo against WARP — the capability crates/d2d's pixel
    // draw transform (1f486eb) and d2d-ref (f12feb9) restored.
    assert_eq!(report("shader3-05-restex").channel_mismatches, 0);
}

#[test]
fn sampleinfo_is_byte_exact() {
    assert_eq!(report("shader3-02-sampleinfo").channel_mismatches, 0);
}

#[test]
fn gather4_po_is_byte_exact() {
    assert_eq!(report("shader3-03-gather4po").channel_mismatches, 0);
}

#[test]
fn gather4_c_is_byte_exact() {
    assert_eq!(report("shader3-04-gather4c").channel_mismatches, 0);
}

#[test]
fn samplecmp_is_byte_exact() {
    assert_eq!(report("shader2-10-samplecmp").channel_mismatches, 0);
}

#[test]
fn samplegrad_is_byte_exact() {
    // SampleGrad: the shader's ddx/ddy of uv run on the quad; single-mip → base-level nearest texel.
    let st = compare_quad("shader2-11-samplegrad");
    eprintln!("shader2-11-samplegrad (quad): {}/{} channels differ (max {} ULP)", st.channel_mismatches, st.pixels * 4, st.max_ulp);
    assert_eq!(st.channel_mismatches, 0);
}

/// Run a compute scene through `run_compute` and compare the output UAV bytes to the WARP capture.
fn compare_compute(name: &str) -> (usize, Option<(usize, u8, u8)>) {
    if suite_dir().is_none() {
        return (0, None);
    }
    let root = root();
    let scene: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join(format!("scenes/{name}.json"))).unwrap()).unwrap();
    let mut cbuffer: Vec<u8> = Vec::new();
    if let Some(arr) = scene["cbuffer"].as_array() {
        for x in arr {
            cbuffer.extend_from_slice(&(x.as_f64().unwrap() as f32).to_le_bytes());
        }
        while cbuffer.len() % 16 != 0 {
            cbuffer.push(0);
        }
    }
    let shader = Shader::parse(&std::fs::read(root.join(format!("scenes/{name}.dxbc"))).unwrap()).unwrap();
    let dj = &scene["compute"]["dispatch"];
    let groups = [dj[0].as_u64().unwrap() as u32, dj[1].as_u64().unwrap() as u32, dj[2].as_u64().unwrap() as u32];
    // Build the output UAV at u0 from the shader's own declaration, sized by the scene.
    let decl = shader.program.uavs[0];
    let u = &scene["uav"];
    let (dims, len) = match decl.mem {
        Mem::Typed(_) => {
            let s = &u["size"];
            ([s[0].as_u64().unwrap() as u32, s[1].as_u64().unwrap() as u32, 1], (s[0].as_u64().unwrap() * s[1].as_u64().unwrap() * 4) as usize)
        }
        Mem::Structured { stride, .. } => ([0, 0, 0], stride as usize * u["count"].as_u64().unwrap() as usize),
        Mem::Raw { .. } => ([0, 0, 0], u["bytes"].as_u64().unwrap() as usize),
    };
    let mut uavs = vec![UavDesc { reg: decl.reg, kind: decl.mem, dims, data: vec![0u8; len] }];
    shader.run_compute(&NoTextures, &cbuffer, groups, &mut uavs, &[]);

    let cap = std::fs::read(root.join(format!("oracle/win11-26100/{name}.bin"))).unwrap();
    assert_eq!(cap.len(), uavs[0].data.len(), "{name}: capture size");
    let mut diffs = 0;
    let mut first = None;
    for (i, (&g, &w)) in uavs[0].data.iter().zip(&cap).enumerate() {
        if g != w {
            diffs += 1;
            if first.is_none() {
                first = Some((i, w, g));
            }
        }
    }
    eprintln!("{name}: {diffs}/{} bytes differ{}", cap.len(), match first {
        Some((i, w, g)) => format!("; first @{i} want={w} got={g}"),
        None => " — byte-exact".into(),
    });
    (diffs, first)
}

#[test]
fn compute_groupshared_atomic_is_byte_exact() {
    assert_eq!(compare_compute("shader2-08-compute").0, 0);
}

#[test]
fn compute_typed_uav_is_byte_exact() {
    assert_eq!(compare_compute("shader2-09-compute-uav").0, 0);
}

#[test]
fn arithmetic_is_byte_exact() {
    assert_eq!(report("shader-01-arith").channel_mismatches, 0);
}

#[test]
fn minmax_compare_is_byte_exact() {
    assert_eq!(report("shader-03-minmaxcmp").channel_mismatches, 0);
}

#[test]
fn conversions_are_byte_exact() {
    assert_eq!(report("shader-04-convert").channel_mismatches, 0);
}

#[test]
fn flow_control_is_byte_exact() {
    assert_eq!(report("shader-05-flow").channel_mismatches, 0);
}

#[test]
fn nearest_sample_is_byte_exact() {
    assert_eq!(report("shader-06-sample-nearest").channel_mismatches, 0);
}

#[test]
fn linear_sample_rgb_exact_alpha_within_one_lsb() {
    // Byte-domain bilinear (interpolate UNORM8 values, divide by 255 once): RGB is byte-exact.
    // Alpha carries WARP's documented ≤1/256 (≤1 LSB at 8-bit) filter-weight rounding residual.
    let st = report("shader-06-sample-linear");
    assert_eq!([st.per_channel[0], st.per_channel[1], st.per_channel[2]], [0, 0, 0], "RGB must be byte-exact");
    assert!(st.per_channel_max_abs[3] <= 1.0 / 256.0 + 1e-6, "alpha residual {} > 1 LSB", st.per_channel_max_abs[3]);
}

#[test]
fn premultiply_rgb_within_one_lsb() {
    // Sample (linear) + multiply colour by alpha; the only error is alpha's ≤1-LSB sampler residual
    // propagated through the multiply, so every channel stays within ~1 LSB.
    let st = report("shader-07-premul");
    assert!(st.per_channel_max_abs.iter().all(|&e| e <= 2.0 / 256.0 + 1e-6), "residual {:?}", st.per_channel_max_abs);
}

#[test]
fn unpremultiply_is_byte_exact() {
    // UnPremultiply's pattern (sample + guarded divide) over a nearest sample: byte-exact.
    assert_eq!(report("shader-08-unpremul").channel_mismatches, 0);
}

#[test]
fn transcendentals_are_byte_exact() {
    // exp2/log2/rsq/rcp now route through d2d::warpmath (WARP's emitted x86), so byte-exact.
    assert_eq!(report("shader-02-trans").channel_mismatches, 0);
}
