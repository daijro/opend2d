//! Seeded random scene generator for the differential fuzzer (the `fuzz-gen` bin and the fuzz runner).
//! Emits scenes in the harness JSON format (see [`crate::Scene`]) as `serde_json::Value`, so every
//! generated scene round-trips through [`crate::read_scene`] (the generator's own self-check) and is
//! renderable by both the reference implementation and ours. The goal is
//! divergence hunting for the browser (wasm) build, where any pixel/HRESULT difference from the
//! reference is a cross-site fingerprint — so the generator leans on edge values: 0, ±huge finite, tiny, negative
//! sizes, degenerate geometry and singular/huge transforms. (NaN/inf coordinates need a scene-format
//! extension — JSON has no NaN — tracked separately.)

use serde_json::{json, Value};

mod r27;

/// Deterministic SplitMix64 — one seed → one scene, reproducibly, with no `rand` dependency.
/// The `bool` is REALISTIC mode: finite, moderate, well-conditioned values only (what real pages/apps
/// produce), as opposed to the default adversarial mode's edge/degenerate values. Browsers sanitise
/// non-finite and extreme canvas args before D2D, so realistic divergences are the fingerprints that
/// matter; the degenerate ones are low risk (see fuzz-buckets.md "KNOWN DEGENERATE-INPUT GAPS").
/// The second `bool` is EXTENDED mode (seeds from [`EXT_SEED`]): the round-23 op and parameter set.
/// The third is the round-24 set (seeds from [`EXT24_SEED`]).
pub struct Rng(u64, bool, bool, bool);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15, false, false, false)
    }
    pub fn new_realistic(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15, true, false, false)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// `0..n`.
    fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % n as u64) as u32
    }
    fn chance(&mut self, pct: u32) -> bool {
        self.below(100) < pct
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len() as u32) as usize]
    }
    /// A coordinate as a JSON value: usually a uniform float in `[lo, hi]`, but ~6% of the time a
    /// non-finite `"nan"`/`"inf"`/`"-inf"` string (the scene deserialiser accepts those — JSON has no
    /// NaN) and ~15% a finite edge (0, ±tiny, ±huge, far off-canvas). These are the inputs that expose
    /// rounding/clamp/overflow/NaN-propagation divergences — a browser fingerprint if they differ.
    fn coord(&mut self, lo: f32, hi: f32) -> Value {
        if self.1 {
            // Realistic: always a finite uniform value in range (no non-finite/edge/huge).
            let t = (self.next_u64() as f64 / u64::MAX as f64) as f32;
            return json!(lo + t * (hi - lo));
        }
        match self.below(100) {
            0..=5 => Value::String((*self.pick(&["nan", "inf", "-inf"])).to_string()),
            6..=20 => json!(*self.pick(&[0.0f32, -0.0, 1e-20, -1e-20, 1e9, -1e9, 3.4e38, -3.4e38, 1e-4, hi * 100.0, lo - hi * 10.0, 0.5, 0.25])),
            _ => {
                let t = (self.next_u64() as f64 / u64::MAX as f64) as f32;
                json!(lo + t * (hi - lo))
            }
        }
    }
    /// A non-negative coordinate: realistic radii/lengths never go negative. In adversarial mode this
    /// still returns the edge/degenerate `coord` values (including negatives) to keep that coverage.
    fn coord_pos(&mut self, lo: f32, hi: f32) -> Value {
        self.coord(if self.1 { lo.max(0.0) } else { lo }, hi)
    }
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

const EXTENDS: [u32; 3] = [0, 1, 2]; // CLAMP, WRAP, MIRROR
const INTERPS: [u32; 2] = [0, 1]; // NEAREST, LINEAR (D2D1_BITMAP_INTERPOLATION_MODE: DrawBitmap/sprite)
// D2D1_INTERPOLATION_MODE (DrawImage / image brush / bitmap-brush SetInterpolationMode1): the full set,
// incl. the float-family samplers CUBIC(2), MULTI_SAMPLE_LINEAR(3), ANISOTROPIC(4), HIGH_QUALITY_CUBIC(5).
const INTERPS6: [u32; 6] = [0, 1, 2, 3, 4, 5];
const AA: [u32; 2] = [0, 1]; // PER_PRIMITIVE, ALIASED

/// First seed of the extended generator (round 23 on). Every seed below it generates exactly the scene
/// it always did, so pinned corpora stay reproducible; the extended ops and parameters reach only seeds
/// from here on. Every extended draw is behind `r.2`, so it never consumes a random number for an older
/// seed.
pub const EXT_SEED: u64 = 28_000;
/// First seed of the round-24 additions (bitmap copies and readback, mid-frame DPI and rendering controls,
/// the remaining effects, image brushes over any image, layer-heavy lists, every glyph-run mode,
/// degenerate parameter values). Like [`EXT_SEED`], everything new is behind `r.3`.
pub const EXT24_SEED: u64 = 29_000;
/// First seed of round 25, the text round: every scene from here is built by [`gen_text_scene`]. Older
/// seeds never reach it.
pub const EXT25_SEED: u64 = 30_000;
const FONTS: [&str; 4] = ["Arial", "Segoe UI", "Times New Roman", "Consolas"];
// Extended-mode strings: Latin with accents, Greek and Cyrillic, which every face of the cold capture set
// covers (no fallback to a face outside the capture's font collection).
const EXT_TEXT: [&str; 10] = ["Ag", "Wx1!", "fjord", "x", "The quick", "", " ", "café résumé", "Ωμέγα λ", "Привет"];

fn color(r: &mut Rng) -> Value {
    json!([r.unit(), r.unit(), r.unit(), if r.chance(70) { r.unit() } else { *r.pick(&[0.0f32, 1.0]) }])
}

/// A 3x2 affine: mostly well-conditioned, but sometimes singular (det 0 / collinear rows) or huge.
/// Realistic mode: scale 0.1..10 on each axis, any rotation, moderate translation — always invertible.
fn transform(r: &mut Rng) -> Value {
    if r.1 {
        let (sx, sy) = (0.1 + r.unit() * 9.9, 0.1 + r.unit() * 9.9);
        let a = r.unit() * std::f32::consts::TAU;
        let (c, s) = (a.cos(), a.sin());
        return json!([sx * c, sx * s, -sy * s, sy * c, (r.unit() * 2.0 - 1.0) * 200.0, (r.unit() * 2.0 - 1.0) * 200.0]);
    }
    match r.below(5) {
        0 => json!([0.0, 0.0, 0.0, 0.0, r.coord(-50.0, 50.0), r.coord(-50.0, 50.0)]), // fully singular
        1 => { let s = r.coord(-2.0, 2.0); json!([s.clone(), s.clone(), s.clone(), s, r.coord(-50.0, 50.0), r.coord(-50.0, 50.0)]) } // collinear → det 0
        2 => json!([1e6, 0.0, 0.0, 1e6, r.coord(-1e6, 1e6), r.coord(-1e6, 1e6)]), // huge scale
        3 => json!([1e-6, 0.0, 0.0, 1e-6, 0.0, 0.0]), // near-zero scale
        _ => json!([r.coord(-3.0, 3.0), r.coord(-1.0, 1.0), r.coord(-1.0, 1.0), r.coord(-3.0, 3.0), r.coord(-80.0, 80.0), r.coord(-80.0, 80.0)]),
    }
}

fn stops(r: &mut Rng) -> Value {
    let n = 2 + r.below(4);
    // Positions span 0..1 but sometimes out of order / out of range (native clamps/sorts its way).
    let v: Vec<Value> = (0..n).map(|_| json!({"position": r.coord(if r.1 { 0.0 } else { -0.2 }, if r.1 { 1.0 } else { 1.2 }), "color": color(r)})).collect();
    Value::Array(v)
}

fn shape(r: &mut Rng, w: f32, h: f32) -> Value {
    match r.below(4) {
        0 => json!({"kind": "rect", "rect": rect(r, w, h)}),
        1 => json!({"kind": "rounded", "rect": rect(r, w, h), "radius": [r.coord_pos(-5.0, 30.0), r.coord_pos(-5.0, 30.0)]}),
        2 => json!({"kind": "ellipse", "center": [r.coord(0.0, w), r.coord(0.0, h)], "radius": [r.coord_pos(-5.0, w), r.coord_pos(-5.0, h)]}),
        _ => path(r, w, h),
    }
}

fn rect(r: &mut Rng, w: f32, h: f32) -> Value {
    if r.1 {
        // Realistic: a valid, in-canvas rect (left<=right, top<=bottom, within the target).
        let (x0, x1, y0, y1) = (r.unit() * w, r.unit() * w, r.unit() * h, r.unit() * h);
        return json!([x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1)]);
    }
    // Deliberately allow left>right / top>bottom (inverted/empty rects) and off-canvas corners.
    json!([r.coord(-10.0, w), r.coord(-10.0, h), r.coord(0.0, w + 10.0), r.coord(0.0, h + 10.0)])
}

fn path(r: &mut Rng, w: f32, h: f32) -> Value {
    let nfig = 1 + r.below(3);
    let figs: Vec<Value> = (0..nfig).map(|_| {
        let nseg = 1 + r.below(5);
        let segs: Vec<Value> = (0..nseg).map(|_| match r.below(4) {
            0 => json!({"kind": "line", "to": [r.coord(0.0, w), r.coord(0.0, h)]}),
            1 => json!({"kind": "cubic", "p1": [r.coord(0.0, w), r.coord(0.0, h)], "p2": [r.coord(0.0, w), r.coord(0.0, h)], "to": [r.coord(0.0, w), r.coord(0.0, h)]}),
            2 => json!({"kind": "quadratic", "control": [r.coord(0.0, w), r.coord(0.0, h)], "to": [r.coord(0.0, w), r.coord(0.0, h)]}),
            _ => json!({"kind": "arc", "to": [r.coord(0.0, w), r.coord(0.0, h)], "radius": [r.coord_pos(-5.0, w), r.coord_pos(-5.0, h)], "rotation": r.coord(-360.0, 360.0), "sweep": r.below(2), "size": r.below(2)}),
        }).collect();
        let mut fig = json!({"start": [r.coord(0.0, w), r.coord(0.0, h)], "filled": r.chance(80), "closed": r.chance(60), "segments": segs});
        if r.2 && r.chance(25) {
            // D2D1_PATH_SEGMENT flags per segment: FORCE_UNSTROKED (1), FORCE_ROUND_LINE_JOIN (2).
            let flags: Vec<u32> = (0..nseg).map(|_| r.below(4)).collect();
            fig.as_object_mut().unwrap().insert("seg_flags".into(), json!(flags));
        }
        fig
    }).collect();
    json!({"kind": "path", "fill": r.below(2), "figures": figs})
}

/// A random built-in effect → (clsid, input_count 0/1/2, properties with correct D2D indices/types).
/// Expands coverage from the old 3 effects/empty-props to a curated ~24 our core implements + native
/// renders. Property `{index,value:{type,value}}` matches the harness `Property`.
fn effect(r: &mut Rng) -> (&'static str, u32, Vec<Value>) {
    let pf = |i: u32, v: Value| json!({"index": i, "value": {"type": "float", "value": v}});
    let pe = |i: u32, v: u32| json!({"index": i, "value": {"type": "enum", "value": v}});
    let pu = |i: u32, v: u32| json!({"index": i, "value": {"type": "uint32", "value": v}});
    let pb = |i: u32, v: bool| json!({"index": i, "value": {"type": "bool", "value": v}});
    let p2 = |i: u32, v: Value| json!({"index": i, "value": {"type": "vector2", "value": v}});
    let p4 = |i: u32, v: Value| json!({"index": i, "value": {"type": "vector4", "value": v}});
    let c4 = |r: &mut Rng| json!([r.unit(), r.unit(), r.unit(), r.unit()]);
    match r.below(if r.3 { 55 } else if r.2 { 46 } else { 24 }) {
        0 => ("61c23c20-ae69-4d8e-94cf-50078df638f2", 0, vec![p4(0, c4(r))]), // Flood: Color
        1 => ("cf2bb6ae-889a-4ad7-ba29-a2fd732c9fc9", 0, vec![ // Turbulence
            p2(0, json!([r.coord(0.0, 64.0), r.coord(0.0, 64.0)])), p2(1, json!([r.coord(1.0, 64.0), r.coord(1.0, 64.0)])),
            p2(2, json!([r.unit(), r.unit()])), pu(3, 1 + r.below(6)), pu(4, r.below(100)), pe(5, r.below(2)), pb(6, r.chance(50))]),
        2 => ("8cea8d1e-77b0-4986-b3b9-2f0c0eae7887", 1, vec![p2(0, json!([r.unit(), r.unit()])), p2(1, json!([r.unit(), r.unit()]))]), // Brightness: White/Black
        3 => ("5cb2d9cf-327d-459f-a0ce-40c0b2086bf7", 1, vec![pf(0, json!(r.unit()))]), // Saturation
        4 => ("b648a78a-0ed5-4f80-a94a-8e825aca6b77", 1, vec![pf(0, r.coord(-1.0, 1.0)), pb(1, r.chance(50))]), // Contrast
        5 => ("b56c8cfa-f634-41ee-bee0-ffa617106004", 1, vec![pf(0, r.coord(-2.0, 2.0))]), // Exposure
        6 => ("0f4458ec-4b32-491b-9e85-bd73f44d3eb6", 1, vec![pf(0, r.coord(0.0, 360.0))]), // HueRotation
        7 => ("3a1af410-5f1d-4dbe-84df-915da79b7153", 1, vec![pf(0, json!(r.unit())), pe(1, r.below(2))]), // Sepia
        8 => ("36dde0eb-3725-42e0-836d-52fb20aee644", 1, vec![]), // Grayscale
        9 => ("e0c3784d-cb39-4e84-b6fd-6b72f0810263", 1, vec![]), // Invert
        10 => ("89176087-8af9-4a08-aeb1-895f38db1766", 1, vec![pf(0, r.coord(-1.0, 1.0)), pf(1, r.coord(-1.0, 1.0))]), // TemperatureTint
        11 => ("36312b17-f7dd-4014-915d-ffca768cf211", 1, vec![p4(0, c4(r)), pb(1, r.chance(50))]), // Tint
        12 => ("2188945e-33a3-4366-b7bc-086bd02d0884", 1, vec![pu(0, 2 + r.below(14)), pu(1, 2 + r.below(14)), pu(2, 2 + r.below(14))]), // Posterize
        13 => ("811d79a4-de28-4454-8094-c64685f8bd4c", 1, vec![pf(0, json!(r.unit()))]), // Opacity
        14 => ("921f03d6-641c-47df-852d-b4bb6153ae11", 1, vec![]), // ColorMatrix (identity default)
        15 => ("1feb6d69-2fe6-4ac9-8c58-1d7f93e7a6a5", 1, vec![pf(0, r.coord(0.0, 10.0)), pe(1, r.below(3)), pe(2, r.below(2))]), // GaussianBlur
        16 => ("174319a6-58e9-49b2-bb63-caf2c811a3db", 1, vec![pf(0, r.coord(0.0, 10.0)), pf(1, r.coord(0.0, 360.0)), pe(2, r.below(3)), pe(3, r.below(2))]), // DirectionalBlur
        17 => ("c67ea361-1863-4e69-89db-695d3e9a5b6b", 1, vec![pf(0, r.coord(0.0, 10.0)), p4(1, c4(r)), pe(2, r.below(3))]), // Shadow
        18 => ("c9b887cb-c5ff-4dc5-9779-273dcf417c7d", 1, vec![pf(0, r.coord(0.0, 10.0)), pf(1, json!(r.unit()))]), // Sharpen
        19 => ("eff583ca-cb07-4aa9-ac5d-2cc44c76460f", 1, vec![pf(0, json!(r.unit())), pf(1, r.coord(0.0, 10.0)), pe(2, r.below(2)), pb(3, r.chance(50)), pe(4, r.below(2))]), // EdgeDetection
        20 => ("b1c5eb2b-0348-43f0-8107-4957cacba2ae", 1, vec![pf(0, r.coord(0.0, 10.0)), pf(1, r.coord(0.0, 360.0))]), // Emboss
        21 => ("eae6c40d-626a-4c2d-bfcb-391001abe202", 1, vec![pe(0, r.below(2)), pu(1, 1 + r.below(20)), pu(2, 1 + r.below(20))]), // Morphology
        22 => ("48fc9f51-f6ac-48f1-8b58-3b28ac46f76d", 2, vec![pe(0, r.below(14))]), // Composite: Mode
        23 => ("81c5b77b-13f8-4cdd-ad20-c890547ac65d", 2, vec![pe(0, r.below(26))]), // Blend: Mode
        24..=45 => effect_ext(r),
        _ => effect_ext24(r),
    }
}

/// The round-24 built-in effects (seeds from [`EXT24_SEED`]): lighting, cross-fade, 3D transforms and the
/// hue/RGB and HDR conversions, with the property indices and types the conformance suite sets them by.
fn effect_ext24(r: &mut Rng) -> (&'static str, u32, Vec<Value>) {
    let pf = |i: u32, v: Value| json!({"index": i, "value": {"type": "float", "value": v}});
    let pe = |i: u32, v: u32| json!({"index": i, "value": {"type": "enum", "value": v}});
    let pv = |i: u32, t: &str, v: Value| json!({"index": i, "value": {"type": t, "value": v}});
    let v3 = |r: &mut Rng, lo: f32, hi: f32| json!([r.coord(lo, hi), r.coord(lo, hi), r.coord(lo, hi)]);
    let unit3 = |r: &mut Rng| json!([r.unit(), r.unit(), r.unit()]);
    let kul = |r: &mut Rng| json!([r.coord(0.1, 4.0), r.coord(0.1, 4.0)]);
    match r.below(9) {
        0 => ("428c1ee5-77b8-4450-8ab5-72219c21abda", 1, vec![pf(0, r.coord(0.0, 360.0)), pf(1, r.coord(0.0, 90.0)), pf(2, r.coord(1.0, 128.0)), pf(3, r.coord(0.0, 4.0)), // DistantSpecular
            pf(4, r.coord(-10.0, 10.0)), pv(5, "vector3", unit3(r)), pv(6, "vector2", kul(r)), pe(7, r.below(6))]),
        1 => ("b9e303c3-c08c-4f91-8b7b-38656bc48c20", 1, vec![pv(0, "vector3", v3(r, -20.0, 40.0)), pf(1, r.coord(0.0, 4.0)), pf(2, r.coord(-10.0, 10.0)), // PointDiffuse
            pv(3, "vector3", unit3(r)), pv(4, "vector2", kul(r)), pe(5, r.below(6))]),
        2 => ("818a1105-7932-44f4-aa86-08ae7b2f2c93", 1, vec![pv(0, "vector3", v3(r, -20.0, 40.0)), pv(1, "vector3", v3(r, -20.0, 40.0)), pf(2, r.coord(0.0, 200.0)), // SpotDiffuse
            pf(3, r.coord(0.0, 90.0)), pf(4, r.coord(0.0, 4.0)), pf(5, r.coord(-10.0, 10.0)), pv(6, "vector3", unit3(r)), pv(7, "vector2", kul(r)), pe(8, r.below(6))]),
        3 => ("12f575e8-4db1-485f-9a84-03a07dd3829f", 2, vec![pf(0, r.coord(0.0, 1.0))]), // CrossFade: Weight
        4 => { // 3D transform: a 4x4 matrix, mostly an affine one, sometimes perspective or degenerate
            let mut m: Vec<Value> = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0].iter().map(|v| json!(v)).collect();
            for k in [0usize, 1, 4, 5, 12, 13] { m[k] = r.coord(-4.0, 4.0); }
            if r.chance(30) { m[3] = r.coord(-0.05, 0.05); m[7] = r.coord(-0.05, 0.05); }
            ("e8467b04-ec61-4b8a-b5de-d4d73debea5a", 1, vec![pe(0, r.below(6)), pe(1, r.below(2)), pv(2, "matrix4x4", json!(m))])
        }
        5 => ("c2844d0b-3d86-46e7-85ba-526c9240f3fb", 1, vec![pe(0, r.below(6)), pe(1, r.below(2)), pf(2, r.coord(1.0, 1000.0)), // 3D perspective
            pv(3, "vector2", json!([r.coord(-20.0, 40.0), r.coord(-20.0, 40.0)])), pv(4, "vector3", v3(r, -20.0, 20.0)), pv(5, "vector3", v3(r, -20.0, 20.0)),
            pv(6, "vector3", v3(r, -20.0, 40.0)), pv(7, "vector3", v3(r, -90.0, 90.0))]),
        6 => ("7b78a6bd-0141-4def-8a52-6356ee0cbdd5", 1, vec![pe(0, r.below(2))]), // HueToRgb: input colour space
        7 => ("23f3e5ec-91e8-4d3d-ad0a-afadc1004aa1", 1, vec![pe(0, r.below(2))]), // RgbToHue: output colour space
        _ => ("7b0b748d-4610-4486-a90c-999d9a2e2b11", 1, vec![pf(0, r.coord(0.0, 4000.0)), pf(1, r.coord(0.0, 4000.0)), pe(2, r.below(2))]), // HdrToneMap
    }
}

/// The extended-mode built-in effects (seeds from [`EXT_SEED`]): geometric, convolution, transfer,
/// lighting, keying and compositing effects the first table never drew, with the property indices and
/// types the conformance suite sets them by.
fn effect_ext(r: &mut Rng) -> (&'static str, u32, Vec<Value>) {
    let pf = |i: u32, v: Value| json!({"index": i, "value": {"type": "float", "value": v}});
    let pe = |i: u32, v: u32| json!({"index": i, "value": {"type": "enum", "value": v}});
    let pu = |i: u32, v: u32| json!({"index": i, "value": {"type": "uint32", "value": v}});
    let pb = |i: u32, v: bool| json!({"index": i, "value": {"type": "bool", "value": v}});
    let pv = |i: u32, t: &str, v: Value| json!({"index": i, "value": {"type": t, "value": v}});
    match r.below(22) {
        0 => ("6aa97485-6354-4cfc-908c-e4a74f62c96c", 1, vec![pe(0, r.below(6)), pe(1, r.below(2)), pv(2, "matrix3x2", transform(r)), pf(3, json!(r.unit()))]), // 2D affine transform
        1 => ("9daf9369-3846-4d0e-a44e-0c607934a5d7", 1, vec![pv(0, "vector2", json!([r.coord(0.0, 4.0), r.coord(0.0, 4.0)])), pv(1, "vector2", json!([r.coord(-20.0, 20.0), r.coord(-20.0, 20.0)])), pe(2, r.below(6)), pe(3, r.below(2)), pf(4, json!(r.unit()))]), // Scale
        2 => ("e23f7110-0e9a-4324-af47-6a2c0c46f35b", 1, vec![pv(0, "vector4", json!([r.coord(-10.0, 40.0), r.coord(-10.0, 40.0), r.coord(0.0, 80.0), r.coord(0.0, 80.0)])), pe(1, r.below(2))]), // Crop
        3 => ("2a2d49c0-4acf-43c7-8c6a-7c4a27874d27", 1, vec![pe(0, r.below(3)), pe(1, r.below(3))]), // Border
        4 => ("b0784138-3b76-4bc5-b13b-0fa2ad02659f", 1, vec![pv(0, "vector4", json!([r.coord(0.0, 8.0), r.coord(0.0, 8.0), r.coord(1.0, 24.0), r.coord(1.0, 24.0)]))]), // Tile
        5 => { // ConvolveMatrix: kernel size first, then a matrix of exactly that many taps
            let (kx, ky) = (1 + r.below(5), 1 + r.below(5));
            let k: Vec<Value> = (0..kx * ky).map(|_| r.coord(-2.0, 2.0)).collect();
            ("407f8c08-5533-4331-a341-23cc3877843e", 1, vec![pu(2, kx), pu(3, ky), pv(4, "floats", json!(k)), pf(5, r.coord(0.1, 4.0)), pf(6, r.coord(-0.5, 0.5)), pb(8, r.chance(50)), pe(9, r.below(2)), pb(10, r.chance(50))])
        }
        6 => ("edc48364-0417-4111-9450-43845fa9f890", 2, vec![pf(0, r.coord(-16.0, 16.0)), pe(1, r.below(4)), pe(2, r.below(4))]), // DisplacementMap
        7 => ("fc151437-049a-4784-a24a-f1c4daf20987", 2, vec![pv(0, "vector4", json!([r.coord(-1.0, 2.0), r.coord(-1.0, 2.0), r.coord(-1.0, 2.0), r.coord(-0.5, 0.5)])), pb(1, r.chance(50))]), // ArithmeticComposite
        8 => ("41251ab7-0beb-46f8-9da7-59e93fcce5de", 1, vec![]), // LuminanceToAlpha
        9 => ("74c01f5b-2a0d-408c-88e2-c7a3c7197742", 1, vec![pv(0, "vector3", json!([r.unit(), r.unit(), r.unit()])), pf(1, json!(r.unit())), pb(2, r.chance(50)), pb(3, r.chance(50))]), // ChromaKey
        10 => ("c00c40be-5e67-4ca3-95b4-f4b02c115135", 1, vec![pv(0, "vector4", json!([r.unit(), r.unit(), r.unit(), r.unit()])), pf(1, json!(r.unit())), pf(2, json!(r.unit()))]), // Vignette
        11 => ("cadc8384-323f-4c7e-a361-2e2b24df6ee4", 1, vec![pf(0, r.coord(-1.0, 1.0)), pf(1, r.coord(-1.0, 1.0)), pf(2, r.coord(-1.0, 1.0)), pe(3, r.below(2)), pf(4, r.coord(0.0, 10.0))]), // HighlightsShadows
        12 => ("06eab419-deed-4018-80d2-3e1d471adeb2", 1, vec![]), // Premultiply
        13 => ("fb9ac489-ad8d-41ed-9999-bb6347d110f7", 1, vec![]), // UnPremultiply
        14 => ("c80ecff0-3fd5-4f05-8328-c5d1724b4f0a", 2, vec![]), // AlphaMask
        15 => ("ad47c8fd-63ef-4acc-9b51-67979c036c06", 1, vec![pf(0, r.coord(-1.0, 1.0)), pf(1, r.coord(-2.0, 3.0)), pf(3, r.coord(-1.0, 1.0)), pf(4, r.coord(-2.0, 3.0))]), // LinearTransfer
        16 => ("409444c4-c419-41a0-b0c1-8cd0c0a18e42", 1, vec![pf(0, r.coord(0.0, 2.0)), pf(1, r.coord(0.0, 4.0)), pf(2, r.coord(-0.5, 0.5))]), // GammaTransfer
        17 => ("3e7efd62-a32d-46d4-a83c-5278889ac954", 1, vec![pf(0, r.coord(0.0, 360.0)), pf(1, r.coord(0.0, 90.0)), pf(2, r.coord(0.0, 4.0)), pf(3, r.coord(-10.0, 10.0)),
            pv(4, "vector3", json!([r.unit(), r.unit(), r.unit()])), pv(5, "vector2", json!([r.coord(0.1, 4.0), r.coord(0.1, 4.0)])), pe(6, r.below(6))]), // DistantDiffuse
        18 => ("4da47b12-79a3-4fb0-8237-bbc3b2a4de08", 1, vec![pf(0, r.coord(-45.0, 45.0)), pb(1, r.chance(50)), pe(2, r.below(6))]), // Straighten
        19 => { // ColorMatrix with a real 5x4 matrix, alpha mode and clamp
            let m: Vec<Value> = (0..20).map(|_| r.coord(-1.0, 1.5)).collect();
            ("921f03d6-641c-47df-852d-b4bb6153ae11", 1, vec![pv(0, "matrix5x4", json!(m)), pe(1, 1 + r.below(2)), pb(2, r.chance(50))])
        }
        20 => { // TableTransfer: a red table of 2..6 entries
            let t: Vec<Value> = (0..2 + r.below(5)).map(|_| json!(r.unit())).collect();
            ("5bf818c3-5e43-48cb-b631-868396d6a1d4", 1, vec![pv(0, "floats", json!(t))])
        }
        _ => { // DiscreteTransfer: a red table of 2..6 entries
            let t: Vec<Value> = (0..2 + r.below(5)).map(|_| json!(r.unit())).collect();
            ("90866fcd-488e-454b-af06-e5041b66c36c", 1, vec![pv(0, "floats", json!(t))])
        }
    }
}

/// `v` with each non-finite `"nan"`/`"inf"`/`"-inf"` string replaced by a finite stand-in (0, ±3.4e38):
/// for scene fields that only take plain numbers (the string forms are accepted on coordinates only).
fn finite(v: Value) -> Value {
    match v {
        Value::String(s) => json!(match s.as_str() { "inf" => 3.4e38f32, "-inf" => -3.4e38, _ => 0.0 }),
        Value::Array(a) => Value::Array(a.into_iter().map(finite).collect()),
        v => v,
    }
}

/// A DrawGlyphRun of a short string at a baseline, with the run's rendering mode, grid fit, measuring
/// mode and text antialias mode (the rendering-mode set DWRITE_RENDERING_MODE1 0..7).
fn glyph_run(r: &mut Rng, w: f32, h: f32, brush: &str, in_list: bool) -> Value {
    // Mostly the DEFAULT rendering mode: an explicit mode that conflicts with the text antialias mode
    // fails validation (E_INVALIDARG), which latches into EndDraw and loses the frame's pixels.
    let mode = if r.chance(60) { 0 } else { r.below(8) };
    let aa = if in_list { *r.pick(&[0u32, 2, 3]) } else { r.below(4) };
    json!({"op": "glyph_run", "text": *r.pick(&EXT_TEXT), "font": *r.pick(&FONTS), "size": finite(r.coord(2.0, 48.0)),
        "weight": *r.pick(&[400u32, 700]), "style": r.below(3), "baseline": finite(json!([r.coord(0.0, w), r.coord(0.0, h)])),
        "brush": brush, "antialias": aa, "measuring": r.below(3), "rendering_mode": mode,
        "grid_fit": r.below(3), "simulations": 0})
}

/// The resources a recorded command list may draw with: everything defined before it.
struct Pools<'a> {
    brushes: &'a [String],
    geoms: &'a [String],
    bitmaps: &'a [String],
    images: &'a [String],
    strokes: &'a [String],
}

/// The body of an extended-mode command list: a mix of every draw kind (fills, strokes, bitmaps, sprite
/// batches, text, glyph runs, gradients under brush transforms, clips, layers, transforms, blend and
/// antialias modes, images), balanced, recorded so its replay path is diffed op by op. Most lists take
/// realistic values: a recording that fails validation (an adversarial value, a mesh or sprite batch
/// outside ALIASED) fails its Close and the whole frame's EndDraw, so it would replay nothing.
fn rich_list(r: &mut Rng, w: f32, h: f32, uid: &mut u32, p: &Pools) -> Vec<Value> {
    let outer = r.1;
    if r.chance(85) { r.1 = true; }
    let ops = rich_list_body(r, w, h, uid, p);
    r.1 = outer;
    ops
}

fn rich_list_body(r: &mut Rng, w: f32, h: f32, uid: &mut u32, p: &Pools) -> Vec<Value> {
    let mut id = |pre: &str| { *uid += 1; format!("{pre}{uid}") };
    let cs = id("cls");
    let mut ops = vec![json!({"op": "solid", "id": cs, "color": color(r), "opacity": r.unit()})];
    let mut brushes: Vec<String> = p.brushes.to_vec();
    brushes.push(cs);
    let mut scope: Vec<&str> = Vec::new();
    for _ in 0..2 + r.below(8) {
        match r.below(16) {
            0 => ops.push(json!({"op": "transform", "matrix": transform(r)})),
            1 => {
                let b = id("cll");
                if r.chance(50) {
                    ops.push(json!({"op": "linear", "id": b, "start": [r.coord(0.0, w), r.coord(0.0, h)], "end": [r.coord(0.0, w), r.coord(0.0, h)],
                        "stops": stops(r), "gamma": r.below(2), "extend": r.pick(&EXTENDS), "opacity": r.unit(), "transform": finite(transform(r))}));
                } else {
                    ops.push(json!({"op": "radial", "id": b, "center": [r.coord(0.0, w), r.coord(0.0, h)], "offset": [r.coord(-w, w), r.coord(-h, h)],
                        "radius": [r.coord_pos(-5.0, w), r.coord_pos(-5.0, h)], "stops": stops(r), "gamma": r.below(2), "extend": r.pick(&EXTENDS),
                        "opacity": r.unit(), "transform": finite(transform(r))}));
                }
                brushes.push(b);
            }
            2 => ops.push(json!({"op": "primitive", "shape": shape(r, w, h), "brush": r.pick(&brushes).clone(),
                "width": if r.chance(40) { json!(r.coord(0.0, 8.0)) } else { Value::Null }})),
            3 if !p.geoms.is_empty() => ops.push(json!({"op": "fill", "geometry": r.pick(p.geoms).clone(), "brush": r.pick(&brushes).clone()})),
            4 if !p.geoms.is_empty() => {
                let style = if !p.strokes.is_empty() && r.chance(50) { json!(r.pick(p.strokes).clone()) } else { Value::Null };
                ops.push(json!({"op": "draw", "geometry": r.pick(p.geoms).clone(), "brush": r.pick(&brushes).clone(), "width": r.coord(0.0, 8.0), "style": style}));
            }
            5 if !p.bitmaps.is_empty() => ops.push(json!({"op": "draw_bitmap", "bitmap": r.pick(p.bitmaps).clone(), "dest": rect(r, w, h),
                "source": [r.coord(0.0, 16.0), r.coord(0.0, 16.0), r.coord(0.0, 16.0), r.coord(0.0, 16.0)], "opacity": r.unit(), "interpolation": r.pick(&INTERPS6)})),
            6 if !p.bitmaps.is_empty() => { // a sprite batch, usually under the ALIASED mode it needs
                if r.chance(90) { ops.push(json!({"op": "antialias", "mode": 1})); }
                let sb = id("csb");
                let sprites: Vec<Value> = (0..1 + r.below(4)).map(|_| json!({"dest": rect(r, w, h),
                    "source": if r.chance(50) { json!([r.coord(0.0, 16.0), r.coord(0.0, 16.0), r.coord(0.0, 16.0), r.coord(0.0, 16.0)]) } else { Value::Null },
                    "color": if r.chance(50) { json!(color(r)) } else { Value::Null },
                    "transform": if r.chance(40) { json!(transform(r)) } else { Value::Null }})).collect();
                ops.push(json!({"op": "sprite_batch", "id": sb, "sprites": sprites}));
                ops.push(json!({"op": "draw_sprite_batch", "batch": sb, "bitmap": r.pick(p.bitmaps).clone(), "interpolation": r.pick(&INTERPS), "options": r.below(2)}));
            }
            7 => ops.push(json!({"op": "text", "text": *r.pick(&EXT_TEXT), "font": *r.pick(&FONTS), "size": r.coord(2.0, 48.0), "rect": rect(r, w, h),
                "brush": r.pick(&brushes).clone(), "weight": *r.pick(&[400u32, 700]), "style": r.below(3), "locale": "en-us",
                "antialias": *r.pick(&[0u32, 2, 3]), "measuring": r.below(3)})),
            8 => { let b = r.pick(&brushes).clone(); ops.push(glyph_run(r, w, h, &b, true)); }
            9 => { ops.push(json!({"op": "push_clip", "rect": rect(r, w, h), "antialias": r.pick(&AA)})); scope.push("clip"); }
            10 => {
                let mask = if !p.geoms.is_empty() && r.chance(40) { json!(r.pick(p.geoms).clone()) } else { Value::Null };
                ops.push(json!({"op": "push_layer", "bounds": rect(r, w, h), "mask": mask, "opacity": r.unit(), "opacity_brush": Value::Null,
                    "mask_antialias": r.below(2), "mask_transform": Value::Null, "layer_options": if r.chance(70) { 0 } else { r.below(4) }}));
                scope.push("layer");
            }
            11 if !scope.is_empty() => ops.push(json!({"op": if scope.pop() == Some("clip") { "pop_clip" } else { "pop_layer" }})),
            12 => ops.push(json!({"op": "blend", "mode": r.below(2)})),
            13 => ops.push(json!({"op": "antialias", "mode": r.pick(&AA)})),
            14 if !p.images.is_empty() => ops.push(json!({"op": "draw_image", "image": r.pick(p.images).clone(),
                "offset": [r.coord(-10.0, w), r.coord(-10.0, h)], "interpolation": r.pick(&INTERPS6), "composite": r.below(14)})),
            15 => {
                if r.chance(90) { ops.push(json!({"op": "antialias", "mode": 1})); }
                let tris: Vec<Value> = (0..1 + r.below(3)).map(|_| json!([[r.coord(0.0, w), r.coord(0.0, h)], [r.coord(0.0, w), r.coord(0.0, h)], [r.coord(0.0, w), r.coord(0.0, h)]])).collect();
                ops.push(json!({"op": "fill_mesh", "triangles": tris, "brush": r.pick(&brushes).clone()}));
            }
            _ => {}
        }
    }
    while let Some(kind) = scope.pop() {
        ops.push(json!({"op": if kind == "clip" { "pop_clip" } else { "pop_layer" }}));
    }
    ops
}

/// Generate one scene for `seed`. Guarantees validity: balanced clip/layer, ids defined before use,
/// inline bitmaps with correct byte counts.
pub fn gen_scene(seed: u64) -> Value {
    if (r27::EXT27_SEED..r27::EXT27_END).contains(&seed) {
        return r27::gen_layer_scene(seed);
    }
    // REALISTIC mode (FUZZ_REALISTIC set): finite/moderate/well-conditioned values only — what real
    // pages and apps produce, the divergences that are actual cross-site fingerprints. Default mode
    // stays adversarial (edge/degenerate), whose remaining gaps are low-risk (fuzz-buckets.md).
    // BROWSER mode (FUZZ_BROWSER set): realistic values AND only the ops a browser canvas emits — fills,
    // strokes, text, images, linear/radial gradients, clips, opacity layers, blur/colour effects,
    // patterns. Drops gradient mesh / ink / sprite batch / geometry realization (ops 26..29), which
    // canvas never produces, so every divergence found is a direct browser fingerprint.
    if seed >= EXT26_SEED {
        return gen_geom_scene(seed);
    }
    if seed >= EXT25_SEED {
        return gen_text_scene(seed);
    }
    let browser = std::env::var_os("FUZZ_BROWSER").is_some();
    let realistic = browser || std::env::var_os("FUZZ_REALISTIC").is_some();
    let mut r = if realistic { Rng::new_realistic(seed) } else { Rng::new(seed) };
    let ext = seed >= EXT_SEED;
    r.2 = ext;
    let ext24 = seed >= EXT24_SEED;
    r.3 = ext24;
    // Extended seeds append the round-23 cases after the existing ones (`k >= nbase`).
    let nbase = if browser { 26 } else { 31 };
    let ncases = nbase + if ext { 9 } else { 0 } + if ext24 { 9 } else { 0 };
    // Mostly sane target sizes, but ~4% a tiny degenerate dimension (1..3 — the scene format/D2D require
    // ≥1, so a true 0-size target is an API-rejection case, not a renderable scene). Realistic: normal only.
    let dim = |r: &mut Rng| if !r.1 && r.chance(4) { *r.pick(&[1u32, 2, 3]) } else { *r.pick(&[16u32, 32, 48, 64, 96, 128]) };
    let w = dim(&mut r);
    let h = dim(&mut r);
    let (wf, hf) = (w as f32, h as f32);
    let mut ops: Vec<Value> = vec![json!({"op": "clear", "color": color(&mut r)})];
    let mut brushes: Vec<String> = Vec::new();
    let mut geoms: Vec<String> = Vec::new();
    let mut bitmaps: Vec<String> = Vec::new();
    let mut strokes: Vec<String> = Vec::new();
    let mut images: Vec<String> = Vec::new(); // DrawImage targets: command lists, effect outputs, bitmaps
    let mut pbrushes: Vec<String> = Vec::new(); // non-solid brushes (gradients) usable as a layer opacity brush
    let mut scope: Vec<&'static str> = Vec::new(); // open clip/layer kinds, innermost last
    let mut uid = 0u32;
    let id = |uid: &mut u32, p: &str| { *uid += 1; format!("{p}{uid}") };

    let nops = 6 + r.below(22);
    for _ in 0..nops {
        match r.below(ncases) {
            k if k >= nbase + 9 => match k - nbase - 9 { // round 24
                0 => { // ID2D1Bitmap::CopyFromMemory into a fresh bitmap
                    let b = id(&mut uid, "bmm");
                    let (bw, bh) = (1 + r.below(16), 1 + r.below(16));
                    let rgba: Vec<u8> = (0..bw * bh * 4).map(|_| r.below(256) as u8).collect();
                    ops.push(json!({"op": "copy_memory_bitmap", "id": b, "size": [bw, bh], "rgba": rgba}));
                    bitmaps.push(b.clone());
                    images.push(b);
                }
                1 if !bitmaps.is_empty() => { // ID2D1Bitmap::CopyFromBitmap, sometimes out of range or across formats
                    let b = id(&mut uid, "bmc");
                    let mut o = json!({"op": "copy_from_bitmap", "id": b, "size": [1 + r.below(16), 1 + r.below(16)], "src": r.pick(&bitmaps).clone(), "bgra": r.chance(20)});
                    let m = o.as_object_mut().unwrap();
                    if r.chance(70) { m.insert("point".into(), json!([r.below(20), r.below(20)])); }
                    if r.chance(60) { m.insert("rect".into(), json!([r.below(20), r.below(20), r.below(20), r.below(20)])); }
                    ops.push(o);
                    bitmaps.push(b.clone());
                    images.push(b);
                }
                2 => { // a mid-frame CopyFromRenderTarget + Map readback, mostly a small in-range rect
                    let rc = if w * h <= 1024 && r.chance(30) { Value::Null } else {
                        let (l, t) = (r.below(w), r.below(h));
                        let (rr, b) = if r.chance(85) { ((l + 1 + r.below(16)).min(w), (t + 1 + r.below(16)).min(h)) } else { (l + r.below(40), t + r.below(40)) };
                        json!([l, t, rr, b])
                    };
                    ops.push(json!({"op": "readback", "rect": rc}));
                }
                3 => { // SetDpi mid-frame, sometimes degenerate
                    let dpi = if r.chance(75) { let d = *r.pick(&[96.0f32, 120.0, 144.0, 192.0, 72.0, 48.0]); json!([d, if r.chance(80) { d } else { *r.pick(&[96.0f32, 120.0, 192.0]) }]) }
                        else { r.pick(&[json!([0.0, 0.0]), json!([-96.0, 96.0]), json!(["nan", "nan"]), json!(["inf", 96.0]), json!([1e-3, 1e-3]), json!([1e6, 1e6]), json!([96.0, 0.0])]).clone() };
                    ops.push(json!({"op": "set_dpi", "dpi": dpi}));
                }
                4 => ops.push(json!({"op": "rendering_controls", "precision": r.below(7), // D2D1_BUFFER_PRECISION 0..5, and one past it
                    "tile": *r.pick(&[[0u32, 0], [1, 1], [64, 64], [256, 256], [512, 1024], [4096, 4096], [100_000, 100_000]])})),
                5 if !images.is_empty() && !geoms.is_empty() => { // an image brush over an effect, list or bitmap, filling a geometry
                    let b = id(&mut uid, "ibx");
                    ops.push(json!({"op": "image_brush", "id": b, "image": r.pick(&images).clone(), "source": rect(&mut r, wf, hf),
                        "extend": [r.pick(&EXTENDS), r.pick(&EXTENDS)], "interpolation": r.pick(&INTERPS6), "opacity": r.unit(), "matrix": transform(&mut r)}));
                    ops.push(json!({"op": "fill", "geometry": r.pick(&geoms).clone(), "brush": b}));
                    brushes.push(b);
                }
                6 if !brushes.is_empty() => { // DrawGlyphRun with any rendering, measuring, grid-fit and antialias mode
                    let b = r.pick(&brushes).clone();
                    let mut g = glyph_run(&mut r, wf, hf, &b, false);
                    let m = g.as_object_mut().unwrap();
                    m.insert("rendering_mode".into(), json!(r.below(8)));
                    m.insert("measuring".into(), json!(r.below(3)));
                    m.insert("grid_fit".into(), json!(r.below(3)));
                    m.insert("antialias".into(), json!(r.below(4)));
                    ops.push(g);
                }
                7 => { // a command list built around layers (options, masks, opacity brushes, nested clips), replayed
                    let c = id(&mut uid, "cly");
                    let cs = id(&mut uid, "cls");
                    // Mostly realistic values: a recording that fails validation fails its Close and the frame.
                    let outer = r.1;
                    if r.chance(85) { r.1 = true; }
                    let mut sub = vec![json!({"op": "solid", "id": cs, "color": color(&mut r), "opacity": r.unit()})];
                    for _ in 0..1 + r.below(3) {
                        let mask = if !geoms.is_empty() && r.chance(40) { json!(r.pick(&geoms).clone()) } else { Value::Null };
                        let ob = if !pbrushes.is_empty() && r.chance(25) { json!(r.pick(&pbrushes).clone()) } else { Value::Null };
                        sub.push(json!({"op": "push_layer", "bounds": rect(&mut r, wf, hf), "mask": mask, "opacity": r.unit(), "opacity_brush": ob,
                            "mask_antialias": r.below(2), "mask_transform": if r.chance(30) { finite(transform(&mut r)) } else { Value::Null }, "layer_options": r.below(4)}));
                        if r.chance(40) { sub.push(json!({"op": "push_clip", "rect": rect(&mut r, wf, hf), "antialias": r.pick(&AA)})); sub.push(json!({"op": "primitive", "shape": shape(&mut r, wf, hf), "brush": cs, "width": Value::Null})); sub.push(json!({"op": "pop_clip"})); }
                        sub.push(json!({"op": "primitive", "shape": shape(&mut r, wf, hf), "brush": cs, "width": if r.chance(30) { json!(r.coord(0.0, 8.0)) } else { Value::Null }}));
                        sub.push(json!({"op": "pop_layer"}));
                    }
                    r.1 = outer;
                    ops.push(json!({"op": "command_list", "id": c, "ops": sub}));
                    images.push(c.clone());
                    if r.chance(50) { ops.push(json!({"op": "transform", "matrix": transform(&mut r)})); }
                    ops.push(json!({"op": "draw_image", "image": c, "offset": [r.coord(-10.0, wf), r.coord(-10.0, hf)], "interpolation": r.pick(&INTERPS6), "composite": r.below(13)}));
                }
                8 if !brushes.is_empty() => { // degenerate parameter values on ops that take them
                    let bad = *r.pick(&[-1.0f32, 2.0, 1e30, -1e30, 0.0, 1e-30, 3.4e38]);
                    match r.below(4) {
                        0 => { let b = id(&mut uid, "sd"); let col = finite(json!([r.coord(-1.0, 2.0), r.coord(-1.0, 2.0), r.coord(-1.0, 2.0), r.coord(-1.0, 2.0)]));
                            ops.push(json!({"op": "solid", "id": b, "color": col, "opacity": bad}));
                            ops.push(json!({"op": "primitive", "shape": shape(&mut r, wf, hf), "brush": b, "width": Value::Null})); brushes.push(b); }
                        1 => { ops.push(json!({"op": "push_layer", "bounds": rect(&mut r, wf, hf), "mask": Value::Null, "opacity": bad, "opacity_brush": Value::Null, "mask_antialias": r.below(2), "mask_transform": Value::Null, "layer_options": r.below(4)})); scope.push("layer"); }
                        2 if !bitmaps.is_empty() => ops.push(json!({"op": "draw_bitmap", "bitmap": r.pick(&bitmaps).clone(), "dest": rect(&mut r, wf, hf), "source": [0.0, 0.0, 16.0, 16.0], "opacity": bad, "interpolation": r.pick(&INTERPS)})),
                        _ => ops.push(json!({"op": "primitive", "shape": shape(&mut r, wf, hf), "brush": r.pick(&brushes).clone(), "width": r.pick(&[json!(-5.0), json!(0.0), json!(1e9), json!(3.4e38), json!("nan"), json!("inf"), json!(-1e-30)]).clone()})),
                    }
                }
                _ => {}
            },
            k if k >= nbase => match k - nbase {
                0 if !brushes.is_empty() => { let b = r.pick(&brushes).clone(); ops.push(glyph_run(&mut r, wf, hf, &b, false)); }
                1 if !geoms.is_empty() => { // a transformed geometry, usable by every geometry op after it
                    let g = id(&mut uid, "gt");
                    ops.push(json!({"op": "geometry_transform", "id": g, "geometry": r.pick(&geoms).clone(), "matrix": finite(transform(&mut r))}));
                    geoms.push(g);
                }
                2 if !brushes.is_empty() => ops.push(json!({"op": "draw_line", "p0": [r.coord(0.0, wf), r.coord(0.0, hf)],
                    "p1": [r.coord(0.0, wf), r.coord(0.0, hf)], "brush": r.pick(&brushes).clone(), "width": r.coord(0.0, 8.0)})),
                3 | 4 => { // a command list of every draw kind, replayed under a different transform
                    let c = id(&mut uid, "cl");
                    let pools = Pools { brushes: &brushes, geoms: &geoms, bitmaps: &bitmaps, images: &images, strokes: &strokes };
                    let sub = rich_list(&mut r, wf, hf, &mut uid, &pools);
                    ops.push(json!({"op": "command_list", "id": c, "ops": sub}));
                    images.push(c.clone());
                    if r.chance(70) {
                        ops.push(json!({"op": "transform", "matrix": transform(&mut r)}));
                        let mut d = json!({"op": "draw_image", "image": c, "offset": [r.coord(-10.0, wf), r.coord(-10.0, hf)],
                            "interpolation": r.pick(&INTERPS6), "composite": r.below(14)});
                        if r.chance(30) { d.as_object_mut().unwrap().insert("image_rect".into(), finite(rect(&mut r, wf, hf))); }
                        ops.push(d);
                    } else {
                        let b = id(&mut uid, "ib");
                        ops.push(json!({"op": "image_brush", "id": b, "image": c, "source": [0.0, 0.0, r.coord(1.0, wf), r.coord(1.0, hf)],
                            "extend": [r.pick(&EXTENDS), r.pick(&EXTENDS)], "interpolation": r.pick(&INTERPS6), "opacity": r.unit(), "matrix": transform(&mut r)}));
                        ops.push(json!({"op": "primitive", "shape": shape(&mut r, wf, hf), "brush": b, "width": Value::Null}));
                        brushes.push(b);
                    }
                }
                5 if !geoms.is_empty() => { // a geometry query on the result channel
                    let g = r.pick(&geoms).clone();
                    let kind = *r.pick(&["area", "length", "bounds", "contains_point", "point_at_length", "stroke_contains", "widened_bounds", "compare"]);
                    let op = if ["area", "length", "bounds", "contains_point"].contains(&kind) { "query" } else { "query_geom" };
                    let mut q = json!({"op": op, "geometry": g, "kind": kind});
                    let m = q.as_object_mut().unwrap();
                    if kind == "contains_point" || kind == "stroke_contains" { m.insert("point".into(), finite(json!([r.coord(0.0, wf), r.coord(0.0, hf)]))); }
                    if kind == "point_at_length" { m.insert("length".into(), finite(r.coord(0.0, 200.0))); }
                    if kind == "stroke_contains" || kind == "widened_bounds" {
                        m.insert("width".into(), finite(r.coord(0.0, 8.0)));
                        if !strokes.is_empty() && r.chance(50) { m.insert("style".into(), json!(r.pick(&strokes).clone())); }
                    }
                    if kind == "compare" { m.insert("other".into(), json!(r.pick(&geoms).clone())); }
                    ops.push(q);
                }
                6 if !geoms.is_empty() => { // widen / outline / simplify / combine into a new, drawable geometry
                    let g = id(&mut uid, "go");
                    let operation = *r.pick(&["widen", "outline", "simplify", "combine"]);
                    let mut o = json!({"op": "geometry_op", "id": g, "source": r.pick(&geoms).clone(), "operation": operation});
                    let m = o.as_object_mut().unwrap();
                    match operation {
                        "widen" => {
                            m.insert("width".into(), finite(r.coord(0.0, 8.0)));
                            if !strokes.is_empty() && r.chance(50) { m.insert("style".into(), json!(r.pick(&strokes).clone())); }
                            if r.chance(30) { m.insert("transform".into(), finite(transform(&mut r))); }
                        }
                        "simplify" => { m.insert("mode".into(), json!(r.below(2))); }
                        "combine" => { m.insert("mode".into(), json!(r.below(4))); m.insert("other".into(), json!(r.pick(&geoms).clone())); }
                        _ => {}
                    }
                    ops.push(o);
                    geoms.push(g);
                }
                7 => { // raw-texel bitmap with a pixel format, alpha mode and DPI
                    let b = id(&mut uid, "bmx");
                    let (bw, bh) = (1 + r.below(16), 1 + r.below(16));
                    // (DXGI_FORMAT, D2D1_ALPHA_MODE, bytes/px): B8G8R8A8 / R8G8B8A8 premultiplied or ignore, A8 premultiplied or straight.
                    let (fmt, alpha, bpp) = *r.pick(&[(87u32, 1u32, 4u32), (87, 1, 4), (87, 3, 4), (28, 1, 4), (28, 3, 4), (65, 1, 1), (65, 2, 1), (28, 1, 4)]);
                    let data: Vec<u8> = (0..bw * bh * bpp).map(|_| r.below(256) as u8).collect();
                    ops.push(json!({"op": "bitmap", "id": b, "size": [bw, bh], "rgba": [], "data": data, "format": fmt, "alpha": alpha,
                        "dpi": *r.pick(&[96.0f32, 48.0, 144.0, 192.0])}));
                    bitmaps.push(b.clone());
                    images.push(b);
                }
                _ => {}
            },
            0 => ops.push(json!({"op": "transform", "matrix": transform(&mut r)})),
            1 => ops.push(json!({"op": "antialias", "mode": r.pick(&AA)})),
            2 => { // solid brush
                let b = id(&mut uid, "s");
                ops.push(json!({"op": "solid", "id": b, "color": color(&mut r), "opacity": r.unit()}));
                if ext && r.chance(15) { ops.last_mut().unwrap().as_object_mut().unwrap().insert("transform".into(), finite(transform(&mut r))); }
                brushes.push(b);
            }
            3 => { // linear gradient
                let b = id(&mut uid, "l");
                ops.push(json!({"op": "linear", "id": b, "start": [r.coord(0.0, wf), r.coord(0.0, hf)], "end": [r.coord(0.0, wf), r.coord(0.0, hf)], "stops": stops(&mut r), "gamma": r.below(2), "extend": r.pick(&EXTENDS), "opacity": r.unit()}));
                if ext && r.chance(40) { ops.last_mut().unwrap().as_object_mut().unwrap().insert("transform".into(), finite(transform(&mut r))); }
                brushes.push(b.clone());
                pbrushes.push(b);
            }
            4 => { // radial gradient
                let b = id(&mut uid, "r");
                ops.push(json!({"op": "radial", "id": b, "center": [r.coord(0.0, wf), r.coord(0.0, hf)], "offset": [r.coord(-wf, wf), r.coord(-hf, hf)], "radius": [r.coord_pos(-5.0, wf), r.coord_pos(-5.0, hf)], "stops": stops(&mut r), "gamma": r.below(2), "extend": r.pick(&EXTENDS), "opacity": r.unit()}));
                if ext && r.chance(40) { ops.last_mut().unwrap().as_object_mut().unwrap().insert("transform".into(), finite(transform(&mut r))); }
                brushes.push(b.clone());
                pbrushes.push(b);
            }
            5 => { // inline bitmap
                let b = id(&mut uid, "bm");
                let (bw, bh) = (1 + r.below(16), 1 + r.below(16));
                let rgba: Vec<u8> = (0..bw * bh * 4).map(|_| r.below(256) as u8).collect();
                ops.push(json!({"op": "bitmap", "id": b, "size": [bw, bh], "rgba": rgba}));
                bitmaps.push(b.clone());
                images.push(b);
            }
            6 => { // geometry
                let g = id(&mut uid, "g");
                ops.push(json!({"op": "geometry", "id": g, "shape": shape(&mut r, wf, hf)}));
                geoms.push(g);
            }
            7 if !brushes.is_empty() => { // primitive (fill or stroke)
                let brush = r.pick(&brushes).clone();
                let width = if r.chance(50) { json!(r.coord(0.0, 10.0)) } else { Value::Null };
                ops.push(json!({"op": "primitive", "shape": shape(&mut r, wf, hf), "brush": brush, "width": width}));
            }
            8 if !geoms.is_empty() && !brushes.is_empty() => {
                ops.push(json!({"op": "fill", "geometry": r.pick(&geoms).clone(), "brush": r.pick(&brushes).clone()}));
            }
            9 if !geoms.is_empty() && !brushes.is_empty() => {
                let style = if !strokes.is_empty() && r.chance(50) { json!(r.pick(&strokes).clone()) } else { Value::Null };
                ops.push(json!({"op": "draw", "geometry": r.pick(&geoms).clone(), "brush": r.pick(&brushes).clone(), "width": r.coord(0.0, 8.0), "style": style}));
            }
            10 if !bitmaps.is_empty() => {
                ops.push(json!({"op": "draw_bitmap", "bitmap": r.pick(&bitmaps).clone(),
                    "dest": rect(&mut r, wf, hf), "source": [r.coord(0.0, 16.0), r.coord(0.0, 16.0), r.coord(0.0, 16.0), r.coord(0.0, 16.0)],
                    "opacity": r.unit(), "interpolation": if ext { *r.pick(&INTERPS6) } else { *r.pick(&INTERPS) }}));
                if ext && r.chance(20) { // ID2D1RenderTarget::DrawBitmap: D2D1_BITMAP_INTERPOLATION_MODE only
                    let m = ops.last_mut().unwrap().as_object_mut().unwrap();
                    m.insert("overload".into(), json!("rt"));
                    m.insert("interpolation".into(), json!(*r.pick(&INTERPS)));
                }
            }
            11 => { // push clip
                ops.push(json!({"op": "push_clip", "rect": rect(&mut r, wf, hf), "antialias": r.pick(&AA)}));
                scope.push("clip");
            }
            12 if !scope.is_empty() => { // pop whatever is innermost (keeps balance)
                ops.push(json!({"op": if scope.pop() == Some("clip") { "pop_clip" } else { "pop_layer" }}));
            }
            13 => { // push layer (geometric mask, opacity brush, mask AA/transform, options)
                // A geometric mask clips the layer to a geometry; an opacity brush supplies per-pixel alpha;
                // mask_antialias/mask_transform shape the mask; layer_options bit0=INIT_FROM_BACKGROUND,
                // bit1=IGNORE_ALPHA. All previously null/absent — the whole mask/brush/options path was unfuzzed.
                let mask = if !geoms.is_empty() && r.chance(40) { json!(r.pick(&geoms).clone()) } else { Value::Null };
                let ob = if !pbrushes.is_empty() && r.chance(35) { json!(r.pick(&pbrushes).clone()) } else { Value::Null };
                let mt = if mask != Value::Null && r.chance(40) { json!(transform(&mut r)) } else { Value::Null };
                ops.push(json!({"op": "push_layer", "bounds": rect(&mut r, wf, hf), "mask": mask, "opacity": r.unit(),
                    "opacity_brush": ob, "mask_antialias": r.below(2), "mask_transform": mt, "layer_options": r.below(4)}));
                scope.push("layer");
            }
            14 => { // text
                let brush = if brushes.is_empty() { None } else { Some(r.pick(&brushes).clone()) };
                if let Some(brush) = brush {
                    let font = r.pick(&["Arial", "Segoe UI", "Times New Roman", "Consolas"]).to_string();
                    let txt = if ext { r.pick(&EXT_TEXT) } else { r.pick(&["Ag", "Wx1!", "fjord", "x", "The quick", "", " "]) }.to_string();
                    ops.push(json!({"op": "text", "text": txt, "font": font, "size": r.coord(2.0, 48.0),
                        "rect": rect(&mut r, wf, hf), "brush": brush, "weight": r.pick(&[400u32, 700]), "style": if ext { r.below(3) } else { 0 },
                        "locale": "en-us", "antialias": r.below(if ext { 4 } else { 3 }), "measuring": r.below(3)}));
                    if ext {
                        // D2D1_DRAW_TEXT_OPTIONS (NO_SNAP 1, CLIP 2, ENABLE_COLOR_FONT 4, DISABLE_COLOR_BITMAP_SNAPPING 8)
                        // and the DrawTextLayout overload.
                        let opts = if r.chance(35) { Some(r.below(16)) } else { None };
                        let layout = r.chance(15);
                        let m = ops.last_mut().unwrap().as_object_mut().unwrap();
                        if let Some(o) = opts { m.insert("options".into(), json!(o)); }
                        if layout { m.insert("overload".into(), json!("layout")); }
                    }
                }
            }
            15 => ops.push(json!({"op": "blend", "mode": r.below(5)})), // SOURCE_OVER/COPY/MIN/ADD/MAX
            16 => { // stroke style (dashes/caps/join) — exercises the stroker/widener via `draw`
                let s = id(&mut uid, "ss");
                // A named D2D1_DASH_STYLE (DASH/DOT/DASH_DOT/DASH_DOT_DOT) part of the time, else a
                // custom dash array. Named styles interact with the dash cap, offset and join, so all
                // are varied together.
                let named = r.chance(45);
                let dash_style = if named { 1 + r.below(4) } else { 0 };
                let dashes: Vec<Value> =
                    if named { vec![] } else { (0..r.below(5)).map(|_| json!(r.coord(0.0, 8.0))).collect() };
                ops.push(json!({"op": "stroke", "id": s, "caps": [r.below(4), r.below(4), r.below(4)],
                    "join": r.below(4), "miter": r.coord(0.0, 20.0), "offset": r.coord(-4.0, 8.0),
                    "dashes": dashes, "transform_type": r.below(3), "dash_style": dash_style}));
                strokes.push(s);
            }
            17 if !bitmaps.is_empty() => { // bitmap brush (a whole sampler path gen_scene missed)
                let b = id(&mut uid, "bb");
                ops.push(json!({"op": "bitmap_brush", "id": b, "bitmap": r.pick(&bitmaps).clone(),
                    "extend": [r.pick(&EXTENDS), r.pick(&EXTENDS)], "interpolation": r.pick(&INTERPS6),
                    "matrix": transform(&mut r), "opacity": r.unit()}));
                brushes.push(b);
            }
            18 if !geoms.is_empty() => { // geometry group (combined fill rule over several geometries)
                let g = id(&mut uid, "gg");
                let members: Vec<Value> = (0..1 + r.below(3)).map(|_| json!(r.pick(&geoms).clone())).collect();
                ops.push(json!({"op": "geometry_group", "id": g, "fill": r.below(2), "geometries": members}));
                geoms.push(g);
            }
            19 if !brushes.is_empty() => { // fill mesh (triangle list)
                let tris: Vec<Value> = (0..1 + r.below(4)).map(|_| json!([
                    [r.coord(0.0, wf), r.coord(0.0, hf)], [r.coord(0.0, wf), r.coord(0.0, hf)],
                    [r.coord(0.0, wf), r.coord(0.0, hf)]])).collect();
                ops.push(json!({"op": "fill_mesh", "triangles": tris, "brush": r.pick(&brushes).clone()}));
            }
            20 => { // command list: a varied, sometimes-NESTED op list → a drawable image
                // (command-list nesting + the DrawImageList slow path paint.rs drives)
                let c = id(&mut uid, "cl");
                let cs = id(&mut uid, "cls");
                let mut sub = vec![
                    json!({"op": "solid", "id": cs, "color": color(&mut r), "opacity": r.unit()}),
                    // fill OR stroke (a width turns the primitive into a stroked draw inside the recording)
                    json!({"op": "primitive", "shape": shape(&mut r, wf, hf), "brush": cs,
                        "width": if r.chance(40) { json!(r.coord(0.5, 6.0)) } else { Value::Null }}),
                ];
                // A NESTED command list recorded + drawn (under a composite mode) INSIDE this one — the
                // recursive record + DrawImageList slow path (one level deep; sub2 does not nest further).
                if r.chance(45) {
                    let c2 = id(&mut uid, "cl");
                    let cs2 = id(&mut uid, "cls");
                    sub.push(json!({"op": "command_list", "id": c2, "ops": [
                        {"op": "solid", "id": cs2, "color": color(&mut r), "opacity": r.unit()},
                        {"op": "primitive", "shape": shape(&mut r, wf, hf), "brush": cs2, "width": Value::Null}
                    ]}));
                    sub.push(json!({"op": "draw_image", "image": c2,
                        "offset": [r.coord(-10.0, wf), r.coord(-10.0, hf)], "interpolation": 0, "composite": r.below(14)}));
                }
                ops.push(json!({"op": "command_list", "id": c, "ops": json!(sub)}));
                images.push(c);
            }
            21 if !images.is_empty() => { // DrawImage of a command list / effect output / bitmap
                // Sometimes under a non-identity world transform: the scaled/rotated/skewed
                // DrawImage(commandList) replay path (command.rs: a transformed list replays its ops).
                if r.chance(35) { ops.push(json!({"op": "transform", "matrix": transform(&mut r)})); }
                ops.push(json!({"op": "draw_image", "image": r.pick(&images).clone(),
                    "offset": [r.coord(-10.0, wf), r.coord(-10.0, hf)], "interpolation": r.pick(&INTERPS6), "composite": r.below(14)}));
                if ext && r.chance(35) { ops.last_mut().unwrap().as_object_mut().unwrap().insert("image_rect".into(), finite(rect(&mut r, wf, hf))); }
            }
            22 => { // built-in effect graph → a drawable image (~24 effects + real props)
                let e = id(&mut uid, "fx");
                let (clsid, nin, props) = effect(&mut r);
                // Input candidates: bitmaps + prior effect/command-list images (enables effect CHAINING,
                // an effect reading another's output). Fall back to Flood when there are too few inputs.
                let cands: Vec<String> = bitmaps.iter().chain(images.iter()).cloned().collect();
                let (clsid, props, inputs): (&str, Vec<Value>, Vec<Value>) = if (nin as usize) <= cands.len() {
                    (clsid, props, (0..nin).map(|_| json!(r.pick(&cands).clone())).collect())
                } else {
                    ("61c23c20-ae69-4d8e-94cf-50078df638f2", vec![json!({"index": 0, "value": {"type": "vector4", "value": [r.unit(), r.unit(), r.unit(), r.unit()]}})], vec![]) // Flood
                };
                ops.push(json!({"op": "effect", "id": e, "clsid": clsid, "inputs": inputs, "properties": props}));
                images.push(e);
            }
            23 if !images.is_empty() => { // image brush over an image
                let b = id(&mut uid, "ib");
                ops.push(json!({"op": "image_brush", "id": b, "image": r.pick(&images).clone(),
                    "source": [0.0, 0.0, r.coord(1.0, wf), r.coord(1.0, hf)], "extend": [r.pick(&EXTENDS), r.pick(&EXTENDS)],
                    "interpolation": r.pick(&INTERPS6), "opacity": r.unit(), "matrix": transform(&mut r)}));
                brushes.push(b);
            }
            24 if !brushes.is_empty() => { // DrawTextLayout: wrapped layout + optional per-range formatting,
                // trimming, typography features and inline objects (the rich-text paths browsers drive).
                let txt = r.pick(&["Ag", "Wx1!", "fjord", "x", "The quick brown fox", "", "café résumé"]).to_string();
                let n = txt.encode_utf16().count().max(1) as u32;
                let mut o = json!({"op": "text_layout", "text": txt,
                    "font": r.pick(&["Arial", "Segoe UI", "Times New Roman", "Consolas"]).to_string(),
                    "size": r.coord(2.0, 48.0), "max_width": r.coord(0.0, wf), "max_height": r.coord(0.0, hf),
                    "weight": r.pick(&[400u32, 700]), "style": r.below(3), "stretch": 1 + r.below(9), "locale": "en-us",
                    "measuring": r.below(3), "wrap": r.below(2), "align": r.below(3), "para": r.below(3),
                    "reading": r.below(2), "flow": r.below(2),
                    "brush": r.pick(&brushes).clone(), "origin": [r.coord(0.0, wf), r.coord(0.0, hf)]});
                let m = o.as_object_mut().unwrap();
                if r.chance(55) { // per-range character formatting over sub-ranges
                    let k = 1 + r.below(3);
                    let ranges: Vec<Value> = (0..k).map(|_| json!({
                        "position": r.below(n), "length": 1 + r.below(n),
                        "size": if r.chance(60) { r.coord(2.0, 48.0) } else { Value::Null },
                        "weight": if r.chance(50) { json!(r.pick(&[300u32, 400, 700, 900])) } else { Value::Null },
                        "style": if r.chance(40) { json!(r.below(3)) } else { Value::Null },
                        "stretch": if r.chance(25) { json!(1 + r.below(9)) } else { Value::Null },
                        "underline": if r.chance(30) { json!(r.chance(50)) } else { Value::Null },
                        "strikethrough": if r.chance(30) { json!(r.chance(50)) } else { Value::Null }})).collect();
                    let mut ranges = ranges;
                    if ext {
                        for rg in &mut ranges {
                            if r.chance(30) { rg.as_object_mut().unwrap().insert("family".into(), json!(*r.pick(&FONTS))); }
                        }
                    }
                    m.insert("ranges".into(), json!(ranges));
                }
                if r.chance(25) { // trimming (+ ellipsis sign)
                    m.insert("trimming".into(), json!({"granularity": 1 + r.below(2), "ellipsis": r.chance(60)}));
                }
                if r.chance(25) { // typography feature over the whole run
                    let tag = r.pick(&["liga", "kern", "smcp", "frac", "onum", "dlig", "swsh", "salt"]);
                    m.insert("features".into(), json!([{"position": 0u32, "length": n, "tag": tag, "value": r.below(2)}]));
                }
                if r.chance(20) { // fixed-size inline-object placeholder
                    m.insert("inline_objects".into(), json!([{"position": r.below(n), "length": 1u32,
                        "size": [r.coord(2.0, 30.0), r.coord(2.0, 30.0)], "baseline": r.coord(0.0, 20.0)}]));
                }
                ops.push(o);
            }
            25 if !brushes.is_empty() => { // FillOpacityMask — a BGRA mask (alpha channel) + ALIASED mode,
                // content GRAPHICS (as the suite's opacitymask scenes; the mask is read from the A channel)
                let m = id(&mut uid, "mask");
                let (mw, mh) = (1 + r.below(16), 1 + r.below(16));
                let rgba: Vec<u8> = (0..mw * mh * 4).map(|_| r.below(256) as u8).collect();
                ops.push(json!({"op": "bitmap", "id": m, "size": [mw, mh], "rgba": rgba}));
                ops.push(json!({"op": "antialias", "mode": 1})); // ALIASED (FillOpacityMask precondition)
                ops.push(json!({"op": "fill_opacity_mask", "bitmap": m, "brush": r.pick(&brushes).clone(),
                    "content": 0, "dest": rect(&mut r, wf, hf), "source": [0.0, 0.0, mw as f32, mh as f32]}));
                if ext { // the DeviceContext overload, and null dest / source rects (whole target / whole mask)
                    let (dc, nd, ns) = (r.chance(30), r.chance(15), r.chance(15));
                    let o = ops.last_mut().unwrap().as_object_mut().unwrap();
                    if dc { o.insert("overload".into(), json!("dc")); }
                    if nd { o.insert("dest".into(), Value::Null); }
                    if ns { o.insert("source".into(), Value::Null); }
                }
            }
            26 if !geoms.is_empty() && !brushes.is_empty() => { // geometry realization (fill or stroke) then draw it
                let g = id(&mut uid, "real");
                let stroked = r.chance(50);
                ops.push(json!({"op": "realize", "id": g, "geometry": r.pick(&geoms).clone(),
                    "stroke_width": if stroked { json!(r.coord(0.0, 8.0)) } else { Value::Null },
                    "style": if stroked && !strokes.is_empty() && r.chance(40) { json!(r.pick(&strokes).clone()) } else { Value::Null },
                    "tolerance": if r.chance(40) { json!(r.coord(0.1, 2.0)) } else { Value::Null }}));
                ops.push(json!({"op": "draw_realization", "realization": g, "brush": r.pick(&brushes).clone()}));
            }
            27 if !brushes.is_empty() => { // ink: an InkStyle + a bezier ink path, then DrawInk
                let istyle = id(&mut uid, "istyle");
                ops.push(json!({"op": "ink_style", "id": istyle, "nib_shape": r.below(2), "nib_transform": transform(&mut r)}));
                let ink = id(&mut uid, "ink");
                let pt3 = |r: &mut Rng| json!([r.coord(0.0, wf), r.coord(0.0, hf), r.coord(0.0, 4.0)]);
                let segs: Vec<Value> = (0..1 + r.below(4)).map(|_| json!({"p1": pt3(&mut r), "p2": pt3(&mut r), "end": pt3(&mut r)})).collect();
                ops.push(json!({"op": "ink", "id": ink, "start": pt3(&mut r), "segments": segs}));
                ops.push(json!({"op": "draw_ink", "ink": ink, "brush": r.pick(&brushes).clone(),
                    "style": if r.chance(70) { json!(istyle) } else { Value::Null }}));
            }
            28 if !bitmaps.is_empty() => { // sprite batch of 1..4 sprites, then DrawSpriteBatch
                let sb = id(&mut uid, "sb");
                let sprites: Vec<Value> = (0..1 + r.below(4)).map(|_| json!({
                    "dest": rect(&mut r, wf, hf),
                    "source": if r.chance(50) { json!([r.coord(0.0, 16.0), r.coord(0.0, 16.0), r.coord(0.0, 16.0), r.coord(0.0, 16.0)]) } else { Value::Null },
                    "color": if r.chance(50) { json!(color(&mut r)) } else { Value::Null },
                    "transform": if r.chance(40) { json!(transform(&mut r)) } else { Value::Null }})).collect();
                ops.push(json!({"op": "sprite_batch", "id": sb, "sprites": sprites}));
                ops.push(json!({"op": "draw_sprite_batch", "batch": sb, "bitmap": r.pick(&bitmaps).clone(),
                    "interpolation": r.pick(&INTERPS), "options": r.below(2)}));
            }
            29 => { // gradient mesh: 1..2 Coons patches (16 points, 4 corner colors), then DrawGradientMesh
                let gm = id(&mut uid, "gm");
                let patches: Vec<Value> = (0..1 + r.below(2)).map(|_| json!({
                    "points": (0..16).map(|_| json!([r.coord(0.0, wf), r.coord(0.0, hf)])).collect::<Vec<_>>(),
                    "colors": (0..4).map(|_| color(&mut r)).collect::<Vec<_>>(),
                    "extend": [r.pick(&EXTENDS), r.pick(&EXTENDS), r.pick(&EXTENDS), r.pick(&EXTENDS)]})).collect();
                ops.push(json!({"op": "gradient_mesh", "id": gm, "patches": patches}));
                ops.push(json!({"op": "draw_gradient_mesh", "mesh": gm}));
            }
            30 => ops.push(json!({"op": "unit_mode", "mode": r.below(2)})), // PIXELS vs DIPS (D2D1_UNIT_MODE)
            _ => {} // no-op this slot (a guarded case above wasn't eligible)
        }
    }
    // Close any still-open clips/layers in reverse, matching each kind.
    while let Some(kind) = scope.pop() {
        ops.push(json!({"op": if kind == "clip" { "pop_clip" } else { "pop_layer" }}));
    }
    let dpi = if ext { *r.pick(&[96.0f32, 120.0, 144.0, 192.0, 72.0]) } else { *r.pick(&[96.0f32, 120.0, 192.0]) };
    json!({"version": 2, "size": [w, h], "dpi": dpi, "ops": ops})
}


// ---- round 25: text ---------------------------------------------------------------------------------

/// Round-25 families. Each resolves to a face of the round's cold collection (see `tools/fuzz/vmcap.py`
/// `FONTS25`); the last two are names the collection does not hold as families ("Arial Black" groups under
/// Arial; "NoSuchFont" does not exist).
const FONTS25: [&str; 20] = ["Arial", "Times New Roman", "Consolas", "Segoe UI", "Calibri", "Comic Sans MS", "Courier New",
    "Georgia", "Impact", "Lucida Console", "Palatino Linotype", "Tahoma", "Trebuchet MS", "Verdana", "Bahnschrift",
    "Leelawadee UI", "Malgun Gothic", "Segoe UI Emoji", "Arial Black", "NoSuchFont"];
/// Round-25 strings: Latin with kerning pairs, ligatures, accents and combining marks, Greek and Cyrillic,
/// bidi (Hebrew, Arabic, mixed with digits), Thai, Devanagari, Korean, Japanese, emoji (ZWJ sequences, skin
/// tones, flags), symbols, and whitespace (tabs, line breaks).
const TEXT25: [&str; 30] = ["Ag", "The quick brown fox", "fjord ffi fl", "AVAWAY To Tj", "Wx1!", "café résumé naïve",
    "e\u{301} a\u{308} o\u{338} n\u{303}", "0123456789", "Ωμέγα λόγος", "Привет мир", "abc אבג 123", "שלום עולם",
    "مرحبا بالعالم", "Hello مرحبا 42", "ภาษาไทย", "नमस्ते दुनिया", "한국어 텍스트", "日本語のテキスト", "🙂👍🏽❤️",
    "👨\u{200d}👩\u{200d}👧 🇺🇸", "A😀B", "→ € ™ ½", "line one\nline two", "tab\there", " lead", "trail ", "", "x",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "iiiiillll WWWWMMMM"];
const LOCALES25: [&str; 8] = ["en-us", "en-us", "ja-jp", "ko-kr", "he-il", "ar-eg", "th-th", "hi-in"];

/// A text em: mostly ordinary, a fifth tiny (0.5-4 px), some large or huge (300+ px, the outline path) and
/// a few degenerate values.
fn size25(r: &mut Rng) -> f32 {
    match r.below(100) {
        0..=19 => 0.5 + r.unit() * 3.5,
        20..=79 => 6.0 + r.unit() * 42.0,
        80..=90 => 48.0 + r.unit() * 100.0,
        91..=97 => 300.0 + r.unit() * 300.0,
        _ => *r.pick(&[0.0f32, -2.0, 1e-3, 5000.0]),
    }
}

/// A transform for text: identity, a rotation (any angle or a quarter turn), a skew, a mirror, a
/// non-uniform scale or a mix, pivoted about a point of the target so the text stays in view.
fn transform25(r: &mut Rng, w: f32, h: f32) -> Value {
    let rot = |a: f32| [a.cos(), a.sin(), -a.sin(), a.cos()];
    let m: [f32; 4] = match r.below(10) {
        0 => [1.0, 0.0, 0.0, 1.0],
        1 | 2 => rot(r.unit() * std::f32::consts::TAU),
        3 => rot(r.pick(&[90.0f32, 180.0, 270.0]).to_radians()),
        4 => [1.0, 0.0, r.unit() * 1.6 - 0.8, 1.0],
        5 => [1.0, r.unit() * 1.6 - 0.8, 0.0, 1.0],
        6 => [-1.0, 0.0, 0.0, 1.0],
        7 => [1.0, 0.0, 0.0, -1.0],
        8 => [0.5 + r.unit() * 2.5, 0.0, 0.0, 0.5 + r.unit() * 2.5],
        _ => {
            let a = rot(r.unit() * std::f32::consts::TAU);
            let (k, sx, sy) = (r.unit() - 0.5, 0.5 + r.unit() * 1.5, 0.5 + r.unit() * 1.5);
            [a[0] * sx, a[1] * sx, (a[2] + k * a[0]) * sy, (a[3] + k * a[1]) * sy]
        }
    };
    let (cx, cy) = (r.unit() * w, r.unit() * h);
    json!([m[0], m[1], m[2], m[3], cx - (m[0] * cx + m[2] * cy), cy - (m[1] * cx + m[3] * cy)])
}

/// One text brush: a solid colour (sometimes translucent), a linear or radial gradient, or a bitmap brush.
fn brush25(r: &mut Rng, ops: &mut Vec<Value>, id: String, w: f32, h: f32) -> String {
    let opacity = if r.chance(65) { 1.0 } else { r.unit() };
    match r.below(10) {
        0..=5 => {
            let c = color(r);
            let c = json!([c[0], c[1], c[2], if r.chance(75) { json!(1.0) } else { c[3].clone() }]);
            ops.push(json!({"op": "solid", "id": id, "color": c, "opacity": opacity}));
        }
        6 => ops.push(json!({"op": "linear", "id": id, "start": [r.unit() * w, r.unit() * h], "end": [r.unit() * w, r.unit() * h],
            "stops": stops(r), "gamma": r.below(2), "extend": r.pick(&EXTENDS), "opacity": opacity})),
        7 => ops.push(json!({"op": "radial", "id": id, "center": [r.unit() * w, r.unit() * h], "offset": [r.unit() * 10.0 - 5.0, r.unit() * 10.0 - 5.0],
            "radius": [1.0 + r.unit() * w, 1.0 + r.unit() * h], "stops": stops(r), "gamma": r.below(2), "extend": r.pick(&EXTENDS), "opacity": opacity})),
        _ => {
            let (bw, bh) = (1 + r.below(8), 1 + r.below(8));
            let rgba: Vec<u8> = (0..bw * bh * 4).map(|_| r.below(256) as u8).collect();
            let b = format!("{id}b");
            ops.push(json!({"op": "bitmap", "id": b, "size": [bw, bh], "rgba": rgba}));
            let s = 1.0 + r.unit() * 6.0;
            ops.push(json!({"op": "bitmap_brush", "id": id, "bitmap": b, "extend": [r.pick(&EXTENDS), r.pick(&EXTENDS)],
                "interpolation": r.pick(&INTERPS6), "matrix": [s, 0.0, 0.0, s, r.unit() * w, r.unit() * h], "opacity": opacity}));
        }
    }
    id
}

/// Whether a text antialias mode accepts a rendering mode (DWRITE_RENDERING_MODE1; NATURAL_SYMMETRIC_DOWNSAMPLED
/// reads as NATURAL_SYMMETRIC): ClearType refuses ALIASED and OUTLINE, grayscale refuses ALIASED, aliased refuses
/// the GDI and natural modes, DEFAULT accepts all. A conflict fails the draw with E_INVALIDARG.
fn aa_accepts(aa: u32, mode: u32) -> bool {
    let m = if mode == 7 { 5 } else { mode };
    match aa {
        1 => m != 1 && m != 6,
        2 => m != 1,
        3 => !(2..=5).contains(&m),
        _ => true,
    }
}

/// A text antialias mode that accepts `mode` (usually; sometimes any, to keep the conflict covered).
fn aa25(r: &mut Rng, mode: u32) -> u32 {
    let aa = r.below(4);
    if aa_accepts(aa, mode) || r.chance(8) { aa } else { *r.pick(&[0u32, 0, 2]) }
}

fn font_bytes(name: &str) -> Value {
    let b: &[u8] = match name {
        "sbix" => include_bytes!("fuzz_fonts/sbix.ttf"),
        "svg" => include_bytes!("fuzz_fonts/svg.ttf"),
        "colr3" => include_bytes!("fuzz_fonts/colr3.ttf"),
        _ => include_bytes!("fuzz_fonts/colr2.ttf"),
    };
    json!(b)
}

/// A round-25 scene: text in every API form, interleaved with the state that changes how text renders.
fn gen_text_scene(seed: u64) -> Value {
    let mut r = Rng::new_realistic(seed);
    let (w, h) = (*r.pick(&[32u32, 48, 64, 96, 128]), *r.pick(&[32u32, 48, 64, 96, 128]));
    let (wf, hf) = (w as f32, h as f32);
    let clear = color(&mut r);
    let clear = if r.chance(60) { json!([clear[0], clear[1], clear[2], 1.0]) } else { clear };
    let mut ops: Vec<Value> = vec![json!({"op": "clear", "color": clear})];
    let mut brushes = Vec::new();
    for i in 0..1 + r.below(3) {
        let b = brush25(&mut r, &mut ops, format!("t{i}"), wf, hf);
        brushes.push(b);
    }
    let mut scope: Vec<&'static str> = Vec::new();
    let pt = |r: &mut Rng, w: f32, h: f32| json!([r.unit() * w, r.unit() * h]);
    // A glyph run's rendering mode stays on the context for later text, so the text antialias mode of a
    // later draw is chosen against it.
    let mut mode_now = 0u32;
    for _ in 0..3 + r.below(8) {
        match r.below(20) {
            0..=1 => ops.push(json!({"op": "transform", "matrix": transform25(&mut r, wf, hf)})),
            2 => { ops.push(json!({"op": "push_clip", "rect": rect(&mut r, wf, hf), "antialias": r.below(2)})); scope.push("clip"); }
            3 => {
                ops.push(json!({"op": "push_layer", "bounds": rect(&mut r, wf, hf), "mask": Value::Null, "opacity": 0.2 + r.unit() * 0.8,
                    "opacity_brush": Value::Null, "mask_antialias": 0, "mask_transform": Value::Null, "layer_options": r.below(2)}));
                scope.push("layer");
            }
            4 if !scope.is_empty() => { let k = scope.pop().unwrap(); ops.push(json!({"op": if k == "clip" { "pop_clip" } else { "pop_layer" }})); }
            5 if r.chance(40) => ops.push(json!({"op": "blend", "mode": r.below(5)})),
            6 if r.chance(25) => { let d = *r.pick(&[96.0f32, 120.0, 144.0, 192.0]); ops.push(json!({"op": "set_dpi", "dpi": [d, d]})); }
            7..=10 => { // DrawText
                let t = *r.pick(&TEXT25);
                let emoji = t.chars().any(|c| c as u32 >= 0x1F000);
                let font = if emoji && r.chance(50) { "Segoe UI Emoji" } else { *r.pick(&FONTS25) };
                let (x, y) = (r.unit() * wf, r.unit() * hf);
                let mut o = json!({"op": "text", "text": t, "font": font, "size": size25(&mut r),
                    "rect": [x, y, x + r.unit() * wf * 1.5, y + r.unit() * hf * 1.5], "brush": r.pick(&brushes).clone(),
                    "weight": *r.pick(&[100u32, 300, 400, 400, 600, 700, 900]), "style": r.below(3), "locale": *r.pick(&LOCALES25),
                    "antialias": aa25(&mut r, mode_now), "measuring": r.below(3)});
                let m = o.as_object_mut().unwrap();
                // D2D1_DRAW_TEXT_OPTIONS: NO_SNAP 1, CLIP 2, ENABLE_COLOR_FONT 4, DISABLE_COLOR_BITMAP_SNAPPING 8.
                if emoji && r.chance(60) { m.insert("options".into(), json!(4 | (r.below(4) & 3) | if r.chance(20) { 8 } else { 0 })); }
                else if r.chance(30) { m.insert("options".into(), json!(r.below(16))); }
                if r.chance(10) { m.insert("overload".into(), json!("layout")); }
                if r.chance(20) {
                    // Explicit params draw in the DEFAULT mode, then the pinned default params come back.
                    m.insert("params".into(), json!([1.0 + r.unit() * 2.0, r.unit() * 2.0, r.unit() * 2.0, r.unit()]));
                    mode_now = 0;
                }
                ops.push(o);
            }
            11..=14 => { // DrawTextLayout
                let t = *r.pick(&TEXT25);
                let n = t.encode_utf16().count().max(1) as u32;
                let (reading, flow) = if r.chance(90) {
                    if r.chance(75) { (r.below(2), r.below(2)) } else { (2 + r.below(2), 2 + r.below(2)) }
                } else { (r.below(4), r.below(4)) };
                let em = size25(&mut r);
                let mut o = json!({"op": "text_layout", "text": t, "font": *r.pick(&FONTS25), "size": em,
                    "max_width": r.unit() * wf * 1.5, "max_height": r.unit() * hf * 1.5,
                    "weight": *r.pick(&[300u32, 400, 700, 900]), "style": r.below(3), "stretch": 1 + r.below(9), "locale": *r.pick(&LOCALES25),
                    "measuring": r.below(3), "wrap": r.below(5), "align": r.below(4), "para": r.below(3), "reading": reading, "flow": flow,
                    "brush": r.pick(&brushes).clone(), "origin": pt(&mut r, wf, hf)});
                let m = o.as_object_mut().unwrap();
                if r.chance(30) { m.insert("trimming".into(), json!({"granularity": r.below(3), "ellipsis": r.chance(60),
                    "delimiter": if r.chance(20) { 32 } else { 0 }, "delimiter_count": r.below(2)})); }
                if r.chance(30) {
                    let sp = em.abs().max(1.0) * (0.5 + r.unit() * 1.5);
                    m.insert("line_spacing".into(), json!([r.below(3), sp, sp * r.unit()]));
                }
                if r.chance(60) {
                    let ranges: Vec<Value> = (0..1 + r.below(3)).map(|_| json!({"position": r.below(n), "length": 1 + r.below(n),
                        "size": if r.chance(40) { json!(size25(&mut r)) } else { Value::Null },
                        "weight": if r.chance(40) { json!(r.pick(&[300u32, 400, 700, 900])) } else { Value::Null },
                        "style": if r.chance(30) { json!(r.below(3)) } else { Value::Null },
                        "stretch": Value::Null,
                        "underline": if r.chance(45) { json!(r.chance(70)) } else { Value::Null },
                        "strikethrough": if r.chance(35) { json!(r.chance(70)) } else { Value::Null },
                        "family": if r.chance(25) { json!(*r.pick(&FONTS25)) } else { Value::Null }})).collect();
                    m.insert("ranges".into(), json!(ranges));
                }
                if r.chance(15) {
                    let tag = r.pick(&["liga", "kern", "smcp", "frac", "onum", "dlig", "case", "ss01"]);
                    m.insert("features".into(), json!([{"position": 0u32, "length": n, "tag": tag, "value": r.below(2)}]));
                }
                ops.push(o);
            }
            15..=17 => { // DrawGlyphRun, every DWRITE_RENDERING_MODE1 and grid fit
                let mode = if r.chance(45) { 0 } else { r.below(8) };
                let aa = aa25(&mut r, mode);
                mode_now = mode;
                // A glyph run names a family of the collection outright (no matcher fallback), so it skips the
                // two families a font-file collection does not hold by that name (Arial Black groups under Arial).
                ops.push(json!({"op": "glyph_run", "text": *r.pick(&TEXT25), "font": *r.pick(&FONTS25[..18]), "size": size25(&mut r),
                    "weight": *r.pick(&[400u32, 700]), "style": r.below(3), "baseline": pt(&mut r, wf, hf),
                    "brush": r.pick(&brushes).clone(), "antialias": aa, "measuring": r.below(3), "rendering_mode": mode,
                    "grid_fit": r.below(3), "simulations": 0}));
            }
            18 => { // a COLR font
                let colr2 = r.chance(50);
                let t: String = (0..1 + r.below(4)).map(|_| *r.pick(if colr2 { &['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'][..] } else { &['a', 'b', 'c', 'd', 'e', 'f'][..] })).collect();
                let c = color(&mut r);
                ops.push(json!({"op": "color_glyph_run", "font": font_bytes(if colr2 { "colr2" } else { "colr3" }), "text": t,
                    "em_size": 4.0 + r.unit() * 120.0, "baseline": pt(&mut r, wf, hf), "palette": 0,
                    "color": [c[0], c[1], c[2], 1.0], "with_color_support": r.chance(50)}));
            }
            19 => { // a colour bitmap (sbix) or OT-SVG font
                let t = *r.pick(&["A", "AA", "A A"]);
                if r.chance(50) {
                    ops.push(json!({"op": "color_bitmap_glyph_run", "font": font_bytes("sbix"), "text": t, "em_size": 4.0 + r.unit() * 150.0,
                        "baseline": pt(&mut r, wf, hf), "glyph_format": "png"}));
                } else {
                    let c = color(&mut r);
                    ops.push(json!({"op": "svg_glyph_run", "font": font_bytes("svg"), "text": t, "em_size": 4.0 + r.unit() * 150.0,
                        "baseline": pt(&mut r, wf, hf), "color": [c[0], c[1], c[2], 1.0]}));
                }
            }
            _ => {}
        }
    }
    while let Some(k) = scope.pop() {
        ops.push(json!({"op": if k == "clip" { "pop_clip" } else { "pop_layer" }}));
    }
    let dpi = *r.pick(&[96.0f32, 96.0, 120.0, 144.0, 192.0]);
    json!({"version": 2, "size": [w, h], "dpi": dpi, "ops": ops})
}

// --- round 26: geometry ---------------------------------------------------------------------------

/// First seed of round 26, the geometry round: every scene from here is built by [`gen_geom_scene`].
/// Older seeds never reach it.
pub const EXT26_SEED: u64 = 31_000;

/// What a round-26 geometry spans: its largest finite coordinate, whether any coordinate is not finite,
/// and a few of its finite points (where hit tests and lengths get interesting).
#[derive(Clone, Default)]
struct Ext26 {
    max: f32,
    nonfinite: bool,
    pts: Vec<[f32; 2]>,
}
impl Ext26 {
    fn add(&mut self, v: f32) {
        if v.is_finite() { self.max = self.max.max(v.abs()) } else { self.nonfinite = true }
    }
    fn join(&mut self, o: &Ext26) {
        self.max = self.max.max(o.max);
        self.nonfinite |= o.nonfinite;
        self.pts.extend(o.pts.iter().take(6));
    }
}

fn num_json(v: f32) -> Value {
    if v.is_nan() { json!("nan") } else if v == f32::INFINITY { json!("inf") } else if v == f32::NEG_INFINITY { json!("-inf") } else { json!(v) }
}

/// A round-26 number in `[lo, hi]`: `edge` percent of the time an edge value (signed zeros, denormals,
/// tiny, huge up to FLT_MAX, NaN and infinities, far outside the range), sometimes a pixel-grid or
/// eighth-pixel value, else uniform.
fn num26(r: &mut Rng, lo: f32, hi: f32, edge: u32, e: &mut Ext26) -> (Value, f32) {
    let v: f32 = if r.below(100) < edge {
        *r.pick(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0, 1.4e-45, -1e-40, 1.1754944e-38, 1e-20, -1e-7, 1e-4,
            1e9, -1e9, 1e20, -1e30, 3.4e38, -3.4e38, f32::MAX, 524287.5, -8191.75, 16777217.0, 0.5, -0.5])
    } else if r.chance(15) {
        let t = lo + r.unit() * (hi - lo);
        (t * *r.pick(&[1.0f32, 2.0, 8.0, 16.0])).round() / *r.pick(&[1.0f32, 2.0, 8.0, 16.0])
    } else {
        lo + r.unit() * (hi - lo)
    };
    e.add(v);
    (num_json(v), v)
}

fn pt26(r: &mut Rng, w: f32, h: f32, edge: u32, e: &mut Ext26) -> Value {
    let (x, xv) = num26(r, -0.25 * w, 1.25 * w, edge, e);
    let (y, yv) = num26(r, -0.25 * h, 1.25 * h, edge, e);
    if xv.is_finite() && yv.is_finite() && e.pts.len() < 12 { e.pts.push([xv, yv]); }
    json!([x, y])
}

/// A round-26 3x2 transform and its largest linear entry: mostly a rotation, scale, skew or mirror, now
/// and then singular, huge, tiny or with a non-finite entry.
fn transform26(r: &mut Rng, w: f32, h: f32, edge: u32) -> (Value, f32) {
    let t = |r: &mut Rng| (r.unit() * 2.0 - 1.0) * w.max(h);
    let (m, norm): ([f32; 6], f32) = match r.below(14) {
        0..=4 => {
            let a = *r.pick(&[0.0f32, 30.0, 45.0, 90.0, 180.0, 270.0]) + if r.chance(50) { r.unit() * 360.0 } else { 0.0 };
            let (s, c) = a.to_radians().sin_cos();
            let (sx, sy) = (0.25 + r.unit() * 3.0, 0.25 + r.unit() * 3.0);
            ([sx * c, sx * s, -sy * s, sy * c, t(r), t(r)], sx.max(sy))
        }
        5..=6 => { let (sx, sy) = (*r.pick(&[1.0f32, -1.0, 0.5, 2.0, 1.25, 3.0]), *r.pick(&[1.0f32, -1.0, 0.75, 2.0, 1.5])); ([sx, 0.0, 0.0, sy, t(r), t(r)], sx.abs().max(sy.abs())) }
        7 => { let k = r.unit() * 2.0 - 1.0; ([1.0, 0.0, k, 1.0, t(r), t(r)], 1.0 + k.abs()) }
        8 => ([1.0, 0.0, 0.0, 1.0, t(r), t(r)], 1.0),
        9 => { let s = *r.pick(&[1e3f32, 1e6, 1e20, 3.4e38]); ([s, 0.0, 0.0, s, t(r), t(r)], s) }
        10 => { let s = *r.pick(&[1e-3f32, 1e-6, 1e-20, 1.4e-45]); ([s, 0.0, 0.0, s, t(r), t(r)], 1.0) }
        11 => { let s = r.unit() * 2.0; ([s, s, s, s, t(r), t(r)], s) } // collinear rows: det 0
        12 => ([0.0, 0.0, 0.0, 0.0, t(r), t(r)], 1.0),
        _ => {
            let mut m = [1.0f32, 0.0, 0.0, 1.0, t(r), t(r)];
            let k = r.below(6) as usize;
            m[k] = *r.pick(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 3.4e38, -0.0]);
            (m, if k < 4 && m[k].is_finite() { m[k].abs().max(1.0) } else { 1.0 })
        }
    };
    let _ = edge;
    (Value::Array(m.iter().map(|&v| num_json(v)).collect()), norm)
}

fn shape26(r: &mut Rng, w: f32, h: f32, edge: u32, e: &mut Ext26) -> Value {
    match r.below(12) {
        0..=1 => { let (a, b) = (pt26(r, w, h, edge, e), pt26(r, w, h, edge, e)); json!({"kind": "rect", "rect": [a[0], a[1], b[0], b[1]]}) }
        2..=3 => {
            let (a, b) = (pt26(r, w, h, edge, e), pt26(r, w, h, edge, e));
            let (rx, _) = num26(r, -2.0, 0.4 * w, edge, e);
            let (ry, _) = if r.chance(70) { num26(r, -2.0, 0.4 * h, edge, e) } else { (rx.clone(), 0.0) };
            json!({"kind": "rounded", "rect": [a[0], a[1], b[0], b[1]], "radius": [rx, ry]})
        }
        4..=5 => {
            let c = pt26(r, w, h, edge, e);
            let (rx, _) = num26(r, -2.0, 0.6 * w, edge, e);
            let (ry, _) = if r.chance(70) { num26(r, -2.0, 0.6 * h, edge, e) } else { (rx.clone(), 0.0) };
            json!({"kind": "ellipse", "center": c, "radius": [rx, ry]})
        }
        _ => path26(r, w, h, edge, e),
    }
}

fn path26(r: &mut Rng, w: f32, h: f32, edge: u32, e: &mut Ext26) -> Value {
    let figs: Vec<Value> = (0..1 + r.below(4)).map(|_| {
        let start = pt26(r, w, h, edge, e);
        let nseg = if r.chance(5) { 0 } else { 1 + r.below(7) };
        let mut segs = Vec::new();
        let mut last = start.clone();
        for _ in 0..nseg {
            let s = match r.below(20) {
                // Degenerate on purpose: back to the previous point, or onto it again.
                0 => json!({"kind": "line", "to": last.clone()}),
                1..=7 => json!({"kind": "line", "to": pt26(r, w, h, edge, e)}),
                8..=11 => json!({"kind": "cubic", "p1": pt26(r, w, h, edge, e), "p2": pt26(r, w, h, edge, e), "to": pt26(r, w, h, edge, e)}),
                12..=14 => json!({"kind": "quadratic", "control": pt26(r, w, h, edge, e), "to": pt26(r, w, h, edge, e)}),
                _ => {
                    let (rx, _) = num26(r, -5.0, w, edge, e);
                    let (ry, _) = num26(r, -5.0, h, edge, e);
                    let (rot, _) = num26(r, -720.0, 720.0, edge, e);
                    json!({"kind": "arc", "to": pt26(r, w, h, edge, e), "radius": [rx, ry], "rotation": rot, "sweep": r.below(2), "size": r.below(2)})
                }
            };
            last = s["to"].clone();
            segs.push(s);
        }
        let mut fig = json!({"start": start, "filled": r.chance(75), "closed": r.chance(55), "segments": segs});
        if nseg > 0 && r.chance(15) {
            let flags: Vec<u32> = (0..nseg).map(|_| r.below(4)).collect();
            fig.as_object_mut().unwrap().insert("seg_flags".into(), json!(flags));
        }
        fig
    }).collect();
    json!({"kind": "path", "fill": r.below(2), "figures": figs})
}

/// A stroke style and whether it dashes.
fn stroke26(r: &mut Rng, id: &str) -> (Value, bool) {
    let style = *r.pick(&[0u32, 0, 0, 1, 2, 3, 4, 5, 5]);
    let dashes: Vec<Value> = if style == 5 {
        (0..1 + r.below(6)).map(|_| num_json(*r.pick(&[0.0f32, 0.25, 0.5, 1.0, 2.0, 3.5, 8.0, -2.0, 1.0, 2.0]))).collect()
    } else { Vec::new() };
    let miter = *r.pick(&[0.0f32, 0.5, 1.0, 1.4142135, 2.0, 4.0, 10.0, 10.0, 100.0, 1e6, f32::NAN]);
    let offset = *r.pick(&[0.0f32, 0.0, 0.5, -1.5, 2.5, 7.25, 1e3, -1e3]);
    let o = json!({"op": "stroke", "id": id, "caps": [r.below(4), r.below(4), r.below(4)], "join": r.below(4),
        "miter": num_json(miter), "offset": offset, "dashes": dashes, "transform_type": *r.pick(&[0u32, 0, 0, 1, 2]), "dash_style": style});
    (o, style != 0)
}

/// A stroke width: mostly ordinary, sometimes zero, negative, sub-pixel, huge or not finite.
fn width26(r: &mut Rng) -> f32 {
    if r.chance(75) { *r.pick(&[0.5f32, 1.0, 1.0, 1.5, 2.0, 2.25, 3.0, 4.75, 8.0, 20.0]) } else {
        *r.pick(&[0.0f32, -1.0, -3.0, 1e-6, 0.1, 1e3, 1e9, 3.4e38, f32::NAN, f32::INFINITY])
    }
}

/// A flattening tolerance for a query whose device extent is `dev`: anything from NaN and tiny to huge
/// where the extent is small, and only coarser ones as it grows, so no stream runs to millions of
/// segments. `None` is the default tolerance.
fn tolerance26(r: &mut Rng, dev: f32) -> Option<Value> {
    if r.chance(30) { return None; }
    let v = if dev <= 1e3 {
        *r.pick(&[f32::NAN, 0.0, -1.0, 1e-30, 1e-7, 1e-6, 1e-3, 0.01, 0.1, 0.25, 0.5, 1.0, 3.0, 10.0, 100.0, 1e6, 3.4e38, f32::INFINITY])
    } else if dev <= 1e7 {
        *r.pick(&[0.0f32, -1.0, 0.25, 0.5, 1.0, 3.0, 10.0, 100.0, 1e6, 3.4e38, f32::INFINITY])
    } else {
        *r.pick(&[10.0f32, 100.0, 1e6, 3.4e38, f32::INFINITY])
    };
    Some(num_json(v))
}

/// A point for a hit test: near one of the geometry's points (within a few tolerances) or anywhere.
fn hit26(r: &mut Rng, e: &Ext26, w: f32, h: f32) -> Value {
    if !e.pts.is_empty() && r.chance(60) {
        let p = *r.pick(&e.pts);
        let d = *r.pick(&[0.0f32, 0.1, 0.25, 0.5, 1.0, 2.0]);
        json!([p[0] + (r.unit() * 2.0 - 1.0) * d, p[1] + (r.unit() * 2.0 - 1.0) * d])
    } else if r.chance(5) {
        json!([num_json(*r.pick(&[f32::NAN, f32::INFINITY, 3.4e38])), r.unit() * h])
    } else {
        json!([r.unit() * w, r.unit() * h])
    }
}

/// One round-26 scene: geometry of every kind, from ordinary to degenerate, every query and geometry
/// operation on it at tolerances from NaN to huge and under world transforms, and draws of it all under
/// both antialias modes at fractional DPI. A dashed stroke and a fine tolerance go only where the device
/// extent stays modest: dashing or flattening a 1e9-long edge finely runs native for minutes.
fn gen_geom_scene(seed: u64) -> Value {
    let mut r = Rng::new(seed);
    let (w, h) = (*r.pick(&[16u32, 32, 48, 64, 96, 128]), *r.pick(&[16u32, 32, 48, 64, 96, 128]));
    let (wf, hf) = (w as f32, h as f32);
    // The share of edge values in this scene's numbers: clean scenes test the geometry itself.
    let edge = *r.pick(&[0u32, 0, 3, 8, 20]);
    let mut ops: Vec<Value> = vec![json!({"op": "clear", "color": [1.0, 1.0, 1.0, 1.0]})];
    let mut brushes = Vec::new();
    for i in 0..1 + r.below(3) {
        let b = format!("s{i}");
        let c = color(&mut r);
        ops.push(json!({"op": "solid", "id": b, "color": [c[0], c[1], c[2], if r.chance(70) { 1.0 } else { 0.3 + r.unit() * 0.7 }], "opacity": if r.chance(80) { 1.0 } else { r.unit() }}));
        brushes.push(b);
    }
    let mut geoms: Vec<(String, Ext26)> = Vec::new();
    let mut strokes: Vec<(String, bool)> = Vec::new();
    let mut uid = 0u32;
    let mut next = |p: &str| { uid += 1; format!("{p}{uid}") };
    // The current world transform's largest entry, and the DPI scale, for the dashing limit of draws.
    let mut world_norm = 1.0f32;
    let dpi = *r.pick(&[96.0f32, 96.0, 120.0, 144.0, 72.0, 100.8, 134.4, 37.5, 96.5, 250.0]);
    let mut dpi_now = dpi;
    for _ in 0..8 + r.below(18) {
        match r.below(40) {
            0..=8 => { // a geometry of any kind
                let g = next("g");
                let mut e = Ext26::default();
                let s = shape26(&mut r, wf, hf, edge, &mut e);
                ops.push(json!({"op": "geometry", "id": g, "shape": s}));
                geoms.push((g, e));
            }
            9 if !geoms.is_empty() => { // a group of earlier geometries
                let g = next("gg");
                let mut e = Ext26::default();
                let members: Vec<Value> = (0..1 + r.below(4)).map(|_| { let (id, m) = r.pick(&geoms).clone(); e.join(&m); json!(id) }).collect();
                ops.push(json!({"op": "geometry_group", "id": g, "fill": r.below(2), "geometries": members}));
                geoms.push((g, e));
            }
            10..=11 if !geoms.is_empty() => { // a transformed geometry
                let g = next("gt");
                let (src, se) = r.pick(&geoms).clone();
                let (m, norm) = transform26(&mut r, wf, hf, edge);
                let mut e = Ext26 { max: se.max * norm + wf.max(hf) * 2.0, nonfinite: se.nonfinite || m.as_array().unwrap().iter().any(|v| v.is_string()), pts: Vec::new() };
                if norm > 1e6 { e.max = f32::MAX; }
                ops.push(json!({"op": "geometry_transform", "id": g, "geometry": src, "matrix": m}));
                geoms.push((g, e));
            }
            12..=13 => { let s = next("ss"); let (o, dashed) = stroke26(&mut r, &s); ops.push(o); strokes.push((s, dashed)); }
            14..=22 if !geoms.is_empty() => { // a query
                let (g, e) = r.pick(&geoms).clone();
                let world = if r.chance(35) { Some(transform26(&mut r, wf, hf, edge)) } else { None };
                let wn = world.as_ref().map_or(1.0, |w| w.1);
                let dev = e.max * wn;
                let kind = *r.pick(&["area", "length", "bounds", "contains_point", "contains_point", "point_at_length", "stroke_contains",
                    "stroke_contains", "widened_bounds", "compare", "tessellate"]);
                let tol = tolerance26(&mut r, dev);
                let mut q = match kind {
                    "tessellate" => json!({"op": "tessellate", "geometry": g}),
                    "area" | "length" | "bounds" | "contains_point" => json!({"op": "query", "geometry": g, "kind": kind}),
                    _ => json!({"op": "query_geom", "geometry": g, "kind": kind}),
                };
                let m = q.as_object_mut().unwrap();
                if let Some(t) = tol { if kind != "bounds" { m.insert("tolerance".into(), t); } }
                if let Some((wv, _)) = world { if kind != "compare" { m.insert("world".into(), wv); } }
                if kind == "contains_point" || kind == "stroke_contains" { m.insert("point".into(), hit26(&mut r, &e, wf, hf)); }
                if kind == "point_at_length" {
                    m.insert("length".into(), num_json(*r.pick(&[0.0f32, -1.0, 0.5, 3.0, 10.0, 25.0, 60.0, 200.0, 1e9, f32::NAN, f32::INFINITY, 1e-7])));
                }
                if kind == "stroke_contains" || kind == "widened_bounds" {
                    let wd = width26(&mut r);
                    m.insert("width".into(), num_json(wd));
                    let small = dev + wd.abs() * wn <= 2e3 && wd.abs() >= 0.25;
                    let usable: Vec<&(String, bool)> = strokes.iter().filter(|(_, d)| !*d || small).collect();
                    if !usable.is_empty() && r.chance(60) { m.insert("style".into(), json!(r.pick(&usable).0.clone())); }
                }
                if kind == "compare" {
                    m.insert("other".into(), json!(r.pick(&geoms).0.clone()));
                    if r.chance(40) { m.insert("transform".into(), transform26(&mut r, wf, hf, edge).0); }
                }
                ops.push(q);
            }
            23..=26 if !geoms.is_empty() => { // widen / outline / simplify / combine into a new geometry
                let g = next("go");
                let (src, se) = r.pick(&geoms).clone();
                let operation = *r.pick(&["widen", "widen", "outline", "simplify", "combine", "combine"]);
                let world = if r.chance(30) { Some(transform26(&mut r, wf, hf, edge)) } else { None };
                let wn = world.as_ref().map_or(1.0, |w| w.1);
                let mut e = Ext26 { max: se.max * wn, nonfinite: se.nonfinite, pts: se.pts.clone() };
                let mut o = json!({"op": "geometry_op", "id": g, "source": src, "operation": operation});
                let m = o.as_object_mut().unwrap();
                match operation {
                    "widen" => {
                        let wd = width26(&mut r);
                        m.insert("width".into(), num_json(wd));
                        let small = e.max + wd.abs() * wn <= 2e3 && wd.abs() >= 0.25;
                        let usable: Vec<&(String, bool)> = strokes.iter().filter(|(_, d)| !*d || small).collect();
                        if !usable.is_empty() && r.chance(60) { m.insert("style".into(), json!(r.pick(&usable).0.clone())); }
                        if let Some((wv, _)) = &world { m.insert("transform".into(), wv.clone()); }
                        e.max += if wd.is_finite() { wd.abs() * wn } else { f32::MAX };
                    }
                    "outline" | "simplify" => {
                        if operation == "simplify" { m.insert("mode".into(), json!(r.below(2))); }
                        if let Some((wv, _)) = &world { m.insert("world".into(), wv.clone()); }
                    }
                    _ => {
                        let (other, oe) = r.pick(&geoms).clone();
                        m.insert("mode".into(), json!(r.below(4)));
                        m.insert("other".into(), json!(other));
                        if let Some((wv, _)) = &world { m.insert("transform".into(), wv.clone()); }
                        e.max = se.max.max(oe.max * wn);
                        e.nonfinite |= oe.nonfinite;
                    }
                }
                if let Some(t) = tolerance26(&mut r, e.max) { m.insert("tolerance".into(), t); }
                ops.push(o);
                geoms.push((g, e));
            }
            27..=31 if !geoms.is_empty() => { // fill or stroke a geometry
                let (g, e) = r.pick(&geoms).clone();
                let b = r.pick(&brushes).clone();
                if r.chance(40) {
                    ops.push(json!({"op": "fill", "geometry": g, "brush": b}));
                } else {
                    let wd = width26(&mut r);
                    let scale = world_norm * dpi_now / 96.0;
                    let small = (e.max + wd.abs()) * scale <= 2e3 && wd.abs() * scale >= 0.25 && !e.nonfinite;
                    let usable: Vec<&(String, bool)> = strokes.iter().filter(|(_, d)| !*d || small).collect();
                    let style = if !usable.is_empty() && r.chance(60) { json!(r.pick(&usable).0.clone()) } else { Value::Null };
                    ops.push(json!({"op": "draw", "geometry": g, "brush": b, "width": num_json(wd), "style": style}));
                }
            }
            32 => { // a primitive: FillRectangle / DrawEllipse / FillRoundedRectangle and the like
                let mut e = Ext26::default();
                let s = shape26(&mut r, wf, hf, edge, &mut e);
                let s = if s["kind"] == "path" { json!({"kind": "rect", "rect": [r.unit() * wf, r.unit() * hf, r.unit() * wf, r.unit() * hf]}) } else { s };
                let wd = if r.chance(50) { Value::Null } else { num_json(width26(&mut r)) };
                ops.push(json!({"op": "primitive", "shape": s, "brush": r.pick(&brushes).clone(), "width": wd}));
            }
            33 if !geoms.is_empty() => { // a filled or stroked realization, drawn
                let (g, e) = r.pick(&geoms).clone();
                let id = next("real");
                let stroked = r.chance(50);
                let wd = width26(&mut r);
                let small = e.max + wd.abs() <= 2e3 && wd.abs() >= 0.25 && !e.nonfinite;
                let usable: Vec<&(String, bool)> = strokes.iter().filter(|(_, d)| !*d || small).collect();
                let style = if stroked && !usable.is_empty() && r.chance(50) { json!(r.pick(&usable).0.clone()) } else { Value::Null };
                ops.push(json!({"op": "realize", "id": id, "geometry": g, "stroke_width": if stroked { num_json(wd) } else { Value::Null },
                    "style": style, "tolerance": tolerance26(&mut r, e.max + wd.abs().min(1e9)).unwrap_or(Value::Null)}));
                ops.push(json!({"op": "draw_realization", "realization": id, "brush": r.pick(&brushes).clone()}));
            }
            34..=35 => {
                let (m, norm) = transform26(&mut r, wf, hf, edge);
                world_norm = if m.as_array().unwrap().iter().any(|v| v.is_string()) { f32::MAX } else { norm };
                ops.push(json!({"op": "transform", "matrix": m}));
            }
            36..=37 => ops.push(json!({"op": "antialias", "mode": r.below(2)})),
            38 if r.chance(40) => {
                let d = *r.pick(&[96.0f32, 120.0, 144.0, 72.0, 100.8, 134.4, 37.5]);
                dpi_now = d;
                ops.push(json!({"op": "set_dpi", "dpi": [d, d]}));
            }
            _ => {}
        }
    }
    json!({"version": 2, "size": [w, h], "dpi": dpi, "ops": ops})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seeds: impl Iterator<Item = u64>) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64; // FNV-1a over the scenes' JSON
        for s in seeds {
            for b in serde_json::to_string(&gen_scene(s)).unwrap().bytes() {
                h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
            }
        }
        h
    }

    /// Seeds before EXT_SEED generate the scenes the pinned corpora were captured from.
    #[test]
    fn pre_ext_seeds_unchanged() {
        assert_eq!(digest((0..40).chain(EXT_SEED - 40..EXT_SEED)), PRE_EXT_DIGEST);
    }
    const PRE_EXT_DIGEST: u64 = 2_749_055_875_428_931_918;

    /// Round-25 scenes parse as harness scenes and stay balanced (every push popped).
    #[test]
    fn r25_scenes_parse() {
        for s in EXT25_SEED..EXT25_SEED + 300 {
            let v = gen_scene(s);
            crate::read_scene_value(&v).unwrap_or_else(|e| panic!("seed {s}: {e}"));
            let ops = v["ops"].as_array().unwrap();
            let depth = ops.iter().fold(0i32, |d, o| match o["op"].as_str() { Some("push_clip" | "push_layer") => d + 1, Some("pop_clip" | "pop_layer") => d - 1, _ => d });
            assert_eq!(depth, 0, "seed {s} unbalanced");
        }
    }

    /// Rounds 23 and 24 keep generating the scenes their corpora were captured from.
    #[test]
    fn r23_r24_seeds_unchanged() {
        assert_eq!(digest((EXT_SEED..EXT_SEED + 40).chain(EXT25_SEED - 40..EXT25_SEED)), R23_R24_DIGEST);
    }
    const R23_R24_DIGEST: u64 = 6_360_592_279_060_608_394;

    /// Round 25 keeps generating the scenes its corpus was captured from.
    #[test]
    fn r25_seeds_unchanged() {
        assert_eq!(digest((EXT25_SEED..EXT25_SEED + 40).chain(EXT26_SEED - 40..EXT26_SEED)), R25_DIGEST);
    }
    const R25_DIGEST: u64 = 10_092_220_903_728_145_247;

    /// Round-26 scenes parse as harness scenes.
    #[test]
    fn r26_scenes_parse() {
        for s in EXT26_SEED..EXT26_SEED + 1000 {
            let v = gen_scene(s);
            crate::read_scene_value(&v).unwrap_or_else(|e| panic!("seed {s}: {e}"));
        }
    }
}
