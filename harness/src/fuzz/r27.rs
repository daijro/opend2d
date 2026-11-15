//! Round 27, layers, composite and effects: every seed in [`EXT27_SEED`]..[`EXT27_END`] is built by
//! [`gen_layer_scene`]. Older seeds never reach it, and the range is closed, so later rounds keep theirs.
//!
//! A scene defines brushes of every kind, bitmaps, geometries, effects and command lists, then draws
//! through nested layers (every option, opacity brushes of every kind, geometric masks under both
//! antialias modes and a mask transform, inverted, empty, huge, infinite and NaN bounds) and clips
//! (aligned, nested, sometimes left unpopped), under every primitive blend mode, with DrawImage of
//! bitmaps, effects and command lists under every composite and interpolation mode, opacity masks,
//! mid-frame DPI and unit mode, rendering controls, and command lists nested and replayed under
//! transforms. Most scenes take realistic values; the rest the adversarial ones.

use super::*;

pub const EXT27_SEED: u64 = 32_000;
pub const EXT27_END: u64 = 33_000;

/// What a round-27 scene has defined so far.
#[derive(Default)]
struct Res {
    brushes: Vec<String>,
    geoms: Vec<String>,
    bitmaps: Vec<String>,
    images: Vec<String>, // bitmaps, effect outputs and command lists
    lists: Vec<String>,
    uid: u32,
}
impl Res {
    fn id(&mut self, p: &str) -> String {
        self.uid += 1;
        format!("{p}{}", self.uid)
    }
}

/// A layer or clip rectangle: mostly an ordinary one, else inverted, empty, huge, infinite, NaN or
/// infinite on one axis only.
fn bounds27(r: &mut Rng, w: f32, h: f32) -> Value {
    let (x, y) = (r.unit() * w, r.unit() * h);
    match r.below(24) {
        0 => json!([x + 1.0 + r.unit() * w, y + 1.0 + r.unit() * h, x, y]), // inverted
        1 => json!([x + 1.0 + r.unit() * w, 0.0, x, h]),                      // inverted in x
        2 => json!([x, y, x, y + r.unit() * h]),                              // empty
        3 => json!([-1e9, -1e9, 1e9, 1e9]),
        4 => json!(["-inf", "-inf", "inf", "inf"]),
        5 => json!([-3.4e38, -3.4e38, 3.4e38, 3.4e38]),
        6 => { let mut b = vec![json!(x), json!(y), json!(x + r.unit() * w), json!(y + r.unit() * h)]; b[r.below(4) as usize] = json!("nan"); json!(b) }
        7 => json!(["-inf", y, "inf", y + r.unit() * h]),
        8 => json!([x, y, "inf", "-inf"]), // infinite and inverted in y
        9 => json!([x, y, x + 0.25 + r.unit() * 3.0, y + 0.25 + r.unit() * 3.0]), // sub-pixel to a few pixels
        _ => rect(r, w, h),
    }
}

/// A layer opacity: mostly in range, sometimes at or past its ends.
fn opacity27(r: &mut Rng) -> f32 {
    if r.chance(80) { r.unit() } else { *r.pick(&[0.0f32, 1.0, 1.0, -0.5, 1.5, 1e30, 1e-30]) }
}

/// A brush of any kind, defined into `ops`.
fn brush27(r: &mut Rng, ops: &mut Vec<Value>, res: &mut Res, w: f32, h: f32) {
    let k = r.below(if res.images.is_empty() { 4 } else { 5 });
    let b = res.id(["s", "l", "r", "bb", "ib"][k as usize]);
    let mut o = match k {
        0 => json!({"op": "solid", "id": b, "color": color(r), "opacity": r.unit()}),
        1 => json!({"op": "linear", "id": b, "start": [r.coord(0.0, w), r.coord(0.0, h)], "end": [r.coord(0.0, w), r.coord(0.0, h)],
            "stops": stops(r), "gamma": r.below(2), "extend": r.pick(&EXTENDS), "opacity": r.unit()}),
        2 => json!({"op": "radial", "id": b, "center": [r.coord(0.0, w), r.coord(0.0, h)], "offset": [r.coord(-w, w), r.coord(-h, h)],
            "radius": [r.coord_pos(-5.0, w), r.coord_pos(-5.0, h)], "stops": stops(r), "gamma": r.below(2), "extend": r.pick(&EXTENDS), "opacity": r.unit()}),
        3 => {
            while res.bitmaps.is_empty() { bitmap27(r, ops, res); }
            json!({"op": "bitmap_brush", "id": b, "bitmap": r.pick(&res.bitmaps).clone(), "extend": [r.pick(&EXTENDS), r.pick(&EXTENDS)],
                "interpolation": r.pick(&INTERPS6), "matrix": transform(r), "opacity": r.unit()})
        }
        _ => json!({"op": "image_brush", "id": b, "image": r.pick(&res.images).clone(), "source": [r.coord(-4.0, 4.0), r.coord(-4.0, 4.0), r.coord(1.0, w), r.coord(1.0, h)],
            "extend": [r.pick(&EXTENDS), r.pick(&EXTENDS)], "interpolation": r.pick(&INTERPS6), "opacity": r.unit(), "matrix": transform(r)}),
    };
    if k < 3 && r.chance(35) { o.as_object_mut().unwrap().insert("transform".into(), finite(transform(r))); }
    ops.push(o);
    res.brushes.push(b);
}

/// A bitmap: straight RGBA mostly, else raw texels in another format, alpha mode or DPI.
fn bitmap27(r: &mut Rng, ops: &mut Vec<Value>, res: &mut Res) {
    let b = res.id("bm");
    let (bw, bh) = if r.chance(85) { (1 + r.below(24), 1 + r.below(24)) } else { (*r.pick(&[1u32, 2, 64]), *r.pick(&[1u32, 3, 48])) };
    if r.chance(65) {
        let rgba: Vec<u8> = (0..bw * bh * 4).map(|_| r.below(256) as u8).collect();
        ops.push(json!({"op": "bitmap", "id": b, "size": [bw, bh], "rgba": rgba}));
    } else {
        // (DXGI_FORMAT, D2D1_ALPHA_MODE, bytes/px): B8G8R8A8 / R8G8B8A8 premultiplied or ignore, A8 premultiplied or straight.
        let (fmt, alpha, bpp) = *r.pick(&[(87u32, 1u32, 4u32), (87, 3, 4), (28, 1, 4), (28, 3, 4), (65, 1, 1), (65, 2, 1)]);
        let data: Vec<u8> = (0..bw * bh * bpp).map(|_| r.below(256) as u8).collect();
        ops.push(json!({"op": "bitmap", "id": b, "size": [bw, bh], "rgba": [], "data": data, "format": fmt, "alpha": alpha,
            "dpi": *r.pick(&[96.0f32, 96.0, 48.0, 144.0, 192.0])}));
    }
    // An A8 bitmap is no colour source for DrawBitmap, a bitmap brush or BlendImage (they fail the frame);
    // it stays an image for DrawImage and effects.
    if !ops.last().unwrap()["format"].as_u64().is_some_and(|f| f == 65) { res.bitmaps.push(b.clone()); }
    res.images.push(b);
}

/// A composite mode: every valid one, and MASK_INVERT (which DrawImage rejects) now and then.
fn composite27(r: &mut Rng) -> u32 {
    if r.chance(1) { 13 } else { r.below(13) }
}

/// A built-in effect over the images defined so far: the whole fuzz table, or one of the common effects
/// with a property at or past the edge of its range.
fn effect27(r: &mut Rng, ops: &mut Vec<Value>, res: &mut Res) {
    let pf = |i: u32, v: Value| json!({"index": i, "value": {"type": "float", "value": v}});
    let pe = |i: u32, v: u32| json!({"index": i, "value": {"type": "enum", "value": v}});
    let pv = |i: u32, t: &str, v: Value| json!({"index": i, "value": {"type": t, "value": v}});
    let e = res.id("fx");
    let (clsid, nin, props) = match r.below(12) {
        0 => ("1feb6d69-2fe6-4ac9-8c58-1d7f93e7a6a5", 1, vec![ // GaussianBlur
            pf(0, json!(*r.pick(&[0.0f32, 1e-3, 0.3, 1.0, 3.0, 12.0, 100.0, 250.0]))), pe(1, r.below(3)), pe(2, r.below(2))]),
        1 => ("c67ea361-1863-4e69-89db-695d3e9a5b6b", 1, vec![ // Shadow
            pf(0, json!(*r.pick(&[0.0f32, 0.5, 3.0, 40.0]))), pv(1, "vector4", json!([r.unit(), r.unit(), r.unit(), r.unit()])), pe(2, r.below(3))]),
        2 => ("e23f7110-0e9a-4324-af47-6a2c0c46f35b", 1, vec![pv(0, "vector4", r.pick(&[ // Crop
            json!(["-inf", "-inf", "inf", "inf"]), json!([2.5, 1.25, 9.75, 14.5]), json!([10.0, 10.0, 2.0, 2.0]), json!([0.0, 0.0, 0.0, 0.0]),
            json!([-1e9, -1e9, 1e9, 1e9]), json!([1.0, "nan", 8.0, 8.0]), json!([0.0, 0.0, 1.0, 1.0])]).clone()), pe(1, r.below(2))]),
        3 => { // 2D affine transform: fractional, singular, huge, mirrored
            let m = r.pick(&[json!([1.0, 0.0, 0.0, 1.0, 0.5, 0.25]), json!([0.0, 0.0, 0.0, 0.0, 3.0, 3.0]), json!([1e6, 0.0, 0.0, 1e6, 0.0, 0.0]),
                json!([-1.0, 0.0, 0.0, 1.0, 16.0, 0.0]), json!([0.7071, 0.7071, -0.7071, 0.7071, 8.0, -2.0]), json!([0.3, 0.0, 0.0, 2.5, 0.0, 0.0])]).clone();
            ("6aa97485-6354-4cfc-908c-e4a74f62c96c", 1, vec![pe(0, r.below(6)), pe(1, r.below(2)), pv(2, "matrix3x2", m), pf(3, json!(r.unit()))])
        }
        4 => ("9daf9369-3846-4d0e-a44e-0c607934a5d7", 1, vec![ // Scale
            pv(0, "vector2", r.pick(&[json!([0.0, 1.0]), json!([1e-4, 1e-4]), json!([0.5, 0.5]), json!([3.0, 0.25]), json!([1e5, 1.0])]).clone()),
            pv(1, "vector2", json!([r.coord(-8.0, 8.0), r.coord(-8.0, 8.0)])), pe(2, r.below(6)), pe(3, r.below(2)), pf(4, json!(r.unit()))]),
        5 => ("811d79a4-de28-4454-8094-c64685f8bd4c", 1, vec![pf(0, json!(*r.pick(&[0.0f32, 1.0, 1.0 / 255.0, 0.5, 0.999])))]), // Opacity
        6 => ("48fc9f51-f6ac-48f1-8b58-3b28ac46f76d", 2, vec![pe(0, r.below(14))]), // Composite
        7 => ("81c5b77b-13f8-4cdd-ad20-c890547ac65d", 2, vec![pe(0, r.below(26))]), // Blend
        8 => { // Flood, out-of-range colours too
            let c = json!([r.unit(), r.unit(), r.unit(), r.unit()]);
            ("61c23c20-ae69-4d8e-94cf-50078df638f2", 0, vec![pv(0, "vector4", r.pick(&[c, json!([2.0, -1.0, 0.5, 0.5]), json!([1.0, 1.0, 1.0, 0.0]), json!([0.5, 0.5, 0.5, 1.5])]).clone())])
        }
        9 => ("2a2d49c0-4acf-43c7-8c6a-7c4a27874d27", 1, vec![pe(0, r.below(3)), pe(1, r.below(3))]), // Border
        _ => effect(r),
    };
    // Input candidates: every image so far (bitmaps, effect outputs and command lists, so effects chain
    // and read recordings); Flood stands in when there are too few.
    let (clsid, props, inputs): (&str, Vec<Value>, Vec<Value>) = if nin as usize <= res.images.len() {
        (clsid, props, (0..nin).map(|_| json!(r.pick(&res.images).clone())).collect())
    } else {
        ("61c23c20-ae69-4d8e-94cf-50078df638f2", vec![pv(0, "vector4", json!([r.unit(), r.unit(), r.unit(), r.unit()]))], vec![])
    };
    ops.push(json!({"op": "effect", "id": e, "clsid": clsid, "inputs": inputs, "properties": props}));
    res.images.push(e);
}

fn push_layer27(r: &mut Rng, res: &Res, w: f32, h: f32) -> Value {
    let mask = if !res.geoms.is_empty() && r.chance(45) { json!(r.pick(&res.geoms).clone()) } else { Value::Null };
    let ob = if !res.brushes.is_empty() && r.chance(40) { json!(r.pick(&res.brushes).clone()) } else { Value::Null };
    let mt = match r.below(12) {
        0..=5 => Value::Null,
        6..=8 => transform(r),
        9 => json!([0.0, 0.0, 0.0, 0.0, 4.0, 4.0]),
        10 => json!([1.0, 0.0, 0.0, 1.0, 0.5, 0.25]),
        _ => json!([1e6, 0.0, 0.0, 1e6, 0.0, 0.0]),
    };
    json!({"op": "push_layer", "bounds": bounds27(r, w, h), "mask": mask, "opacity": opacity27(r), "opacity_brush": ob,
        "mask_antialias": if r.chance(99) { r.below(2) } else { 2 }, "mask_transform": mt, "layer_options": r.below(4)})
}

/// A draw of any kind with the resources so far, or nothing when they lack what it needs.
fn draw27(r: &mut Rng, ops: &mut Vec<Value>, res: &Res, w: f32, h: f32) {
    let b = |r: &mut Rng| r.pick(&res.brushes).clone();
    match r.below(9) {
        0..=2 => ops.push(json!({"op": "primitive", "shape": shape(r, w, h), "brush": b(r), "width": if r.chance(25) { json!(r.coord(0.0, 8.0)) } else { Value::Null }})),
        3 if !res.geoms.is_empty() => ops.push(json!({"op": "fill", "geometry": r.pick(&res.geoms).clone(), "brush": b(r)})),
        4 if !res.bitmaps.is_empty() => ops.push(json!({"op": "draw_bitmap", "bitmap": r.pick(&res.bitmaps).clone(), "dest": rect(r, w, h),
            "source": [r.coord(-2.0, 8.0), r.coord(-2.0, 8.0), r.coord(0.0, 24.0), r.coord(0.0, 24.0)], "opacity": r.unit(), "interpolation": r.pick(&INTERPS6)})),
        5..=7 if !res.images.is_empty() => {
            let mut d = json!({"op": "draw_image", "image": r.pick(&res.images).clone(), "offset": [r.coord(-10.0, w), r.coord(-10.0, h)],
                "interpolation": r.pick(&INTERPS6), "composite": composite27(r)});
            if r.chance(30) { d.as_object_mut().unwrap().insert("image_rect".into(), finite(rect(r, w, h))); }
            ops.push(d);
        }
        _ => ops.push(json!({"op": "clear", "color": color(r)})),
    }
}

/// A command list body: layers (every option, masks, opacity brushes), nested clips, blend modes and
/// draws, sometimes replaying an earlier list under its own transform. Mostly realistic values, since a
/// recording that fails validation fails its Close and the frame.
fn list27(r: &mut Rng, res: &Res, w: f32, h: f32) -> Vec<Value> {
    let outer = r.1;
    if r.chance(85) { r.1 = true; }
    let mut ops = Vec::new();
    let mut scope: Vec<&'static str> = Vec::new();
    for _ in 0..2 + r.below(8) {
        match r.below(12) {
            0 | 1 => { ops.push(push_layer27(r, res, w, h)); scope.push("layer"); }
            2 => { ops.push(json!({"op": "push_clip", "rect": bounds27(r, w, h), "antialias": r.below(2)})); scope.push("clip"); }
            3 if !scope.is_empty() => { let k = scope.pop().unwrap(); ops.push(json!({"op": if k == "clip" { "pop_clip" } else { "pop_layer" }})); }
            4 => ops.push(json!({"op": "blend", "mode": r.below(5)})),
            5 => ops.push(json!({"op": "transform", "matrix": transform(r)})),
            6 if !res.lists.is_empty() => { // replay an earlier list inside this one, under its own transform
                ops.push(json!({"op": "transform", "matrix": transform(r)}));
                ops.push(json!({"op": "draw_image", "image": r.pick(&res.lists).clone(), "offset": [r.coord(-10.0, w), r.coord(-10.0, h)],
                    "interpolation": r.pick(&INTERPS6), "composite": r.below(13)}));
            }
            _ => draw27(r, &mut ops, res, w, h),
        }
    }
    while let Some(k) = scope.pop() { ops.push(json!({"op": if k == "clip" { "pop_clip" } else { "pop_layer" }})); }
    // A list keeps to what records: an op that fails validation (a NaN clip or layer bound, an invalid mask
    // antialias or composite mode) fails its Close, which aborts the scene.
    for o in &mut ops {
        let m = o.as_object_mut().unwrap();
        for k in ["bounds", "rect"] { if let Some(v) = m.get_mut(k) { *v = finite(v.take()); } }
        if m.get("mask_antialias") == Some(&json!(2)) { m.insert("mask_antialias".into(), json!(0)); }
        if m.get("composite") == Some(&json!(13)) { m.insert("composite".into(), json!(12)); }
    }
    // The list's state is the context's: put back what the main frame draws with.
    ops.extend([json!({"op": "blend", "mode": 0}), json!({"op": "transform", "matrix": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]})]);
    r.1 = outer;
    ops
}

pub fn gen_layer_scene(seed: u64) -> Value {
    let mut r = Rng::new(seed);
    (r.1, r.2, r.3) = (r.chance(70), true, true);
    let (w, h) = (*r.pick(&[24u32, 32, 48, 64, 96, 128]), *r.pick(&[24u32, 32, 48, 64, 96, 128]));
    let (wf, hf) = (w as f32, h as f32);
    let mut ops = vec![json!({"op": "clear", "color": color(&mut r)})];
    let mut res = Res::default();
    for _ in 0..1 + r.below(2) { bitmap27(&mut r, &mut ops, &mut res); }
    for _ in 0..r.below(3) {
        let g = res.id("g");
        ops.push(json!({"op": "geometry", "id": g, "shape": shape(&mut r, wf, hf)}));
        res.geoms.push(g);
    }
    if res.geoms.len() > 1 && r.chance(30) {
        let g = res.id("gg");
        ops.push(json!({"op": "geometry_group", "id": g, "fill": r.below(2), "geometries": [res.geoms[0].clone(), res.geoms[1].clone()]}));
        res.geoms.push(g);
    }
    for _ in 0..r.below(3) { effect27(&mut r, &mut ops, &mut res); }
    for _ in 0..2 + r.below(3) { brush27(&mut r, &mut ops, &mut res, wf, hf); }

    let mut scope: Vec<&'static str> = Vec::new();
    for _ in 0..6 + r.below(18) {
        match r.below(30) {
            0..=4 => { ops.push(push_layer27(&mut r, &res, wf, hf)); scope.push("layer"); }
            5 | 6 => { ops.push(json!({"op": "push_clip", "rect": bounds27(&mut r, wf, hf), "antialias": r.below(2)})); scope.push("clip"); }
            7..=9 if !scope.is_empty() => {
                // A pop of the wrong kind now and then (it fails the frame's EndDraw).
                let k = scope.pop().unwrap();
                let clip = (k == "clip") != r.chance(2);
                ops.push(json!({"op": if clip { "pop_clip" } else { "pop_layer" }}));
            }
            10..=17 => draw27(&mut r, &mut ops, &res, wf, hf),
            18 => ops.push(json!({"op": "blend", "mode": r.below(5)})),
            19 => ops.push(json!({"op": "antialias", "mode": r.below(2)})),
            20 => ops.push(json!({"op": "transform", "matrix": if r.chance(25) { json!([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]) } else { transform(&mut r) }})),
            21 => { // SetDpi mid-frame, sometimes degenerate
                let dpi = if r.chance(95) { let d = *r.pick(&[96.0f32, 120.0, 144.0, 192.0, 72.0, 48.0]); json!([d, if r.chance(80) { d } else { *r.pick(&[96.0f32, 120.0, 192.0]) }]) }
                    else { r.pick(&[json!([0.0, 0.0]), json!([-96.0, 96.0]), json!(["nan", "nan"]), json!([1e-3, 1e-3]), json!([1e6, 1e6])]).clone() };
                ops.push(json!({"op": "set_dpi", "dpi": dpi}));
            }
            22 => ops.push(json!({"op": "unit_mode", "mode": r.below(2)})),
            23 => ops.push(json!({"op": "rendering_controls", "precision": r.below(7), // D2D1_BUFFER_PRECISION 0..5, and one past it
                "tile": *r.pick(&[[1024u32, 1024], [0, 0], [1, 1], [7, 5], [16, 16], [64, 64], [256, 256], [512, 1024], [4096, 4096], [100_000, 100_000]])})),
            24 => { // FillOpacityMask, mostly under the ALIASED mode it needs
                let m = res.id("om");
                let (mw, mh) = (1 + r.below(16), 1 + r.below(16));
                let rgba: Vec<u8> = (0..mw * mh * 4).map(|_| r.below(256) as u8).collect();
                ops.push(json!({"op": "bitmap", "id": m, "size": [mw, mh], "rgba": rgba}));
                if r.chance(98) { ops.push(json!({"op": "antialias", "mode": 1})); }
                let mut o = json!({"op": "fill_opacity_mask", "bitmap": m, "brush": r.pick(&res.brushes).clone(), "content": r.below(3),
                    "dest": if r.chance(85) { rect(&mut r, wf, hf) } else { Value::Null },
                    "source": if r.chance(80) { json!([r.coord(-1.0, 2.0), r.coord(-1.0, 2.0), r.coord(1.0, mw as f32 + 2.0), r.coord(1.0, mh as f32 + 2.0)]) } else { Value::Null }});
                if r.chance(30) { o.as_object_mut().unwrap().insert("overload".into(), json!("dc")); }
                ops.push(o);
                res.bitmaps.push(m.clone());
                res.images.push(m);
            }
            25 => effect27(&mut r, &mut ops, &mut res),
            26 => brush27(&mut r, &mut ops, &mut res, wf, hf),
            27 | 28 => { // a command list, sometimes nesting an earlier one, replayed under a transform
                let c = res.id("cl");
                let sub = list27(&mut r, &res, wf, hf);
                ops.push(json!({"op": "command_list", "id": c, "ops": sub}));
                res.images.push(c.clone());
                res.lists.push(c.clone());
                if r.chance(60) { ops.push(json!({"op": "transform", "matrix": transform(&mut r)})); }
                let mut d = json!({"op": "draw_image", "image": c, "offset": [r.coord(-10.0, wf), r.coord(-10.0, hf)],
                    "interpolation": r.pick(&INTERPS6), "composite": composite27(&mut r)});
                if r.chance(25) { d.as_object_mut().unwrap().insert("image_rect".into(), finite(rect(&mut r, wf, hf))); }
                ops.push(d);
            }
            29 if !res.bitmaps.is_empty() => { // BlendImage of a bitmap (every W3C blend mode but dissolve)
                let mode = *r.pick(&[0u32, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]);
                ops.push(json!({"op": "blend_image", "image": r.pick(&res.bitmaps).clone(), "mode": mode,
                    "offset": finite(json!([r.coord(-8.0, wf), r.coord(-8.0, hf)])), "interpolation": r.below(6)}));
            }
            _ => {}
        }
    }
    // Every push popped, except now and then a frame that ends with a clip or layer still open.
    let keep = if !scope.is_empty() && r.chance(4) { 1 } else { 0 };
    while scope.len() > keep {
        let k = scope.pop().unwrap();
        ops.push(json!({"op": if k == "clip" { "pop_clip" } else { "pop_layer" }}));
    }
    let dpi = *r.pick(&[96.0f32, 96.0, 96.0, 120.0, 144.0, 192.0]);
    json!({"version": 2, "size": [w, h], "dpi": dpi, "ops": ops})
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-27 scenes parse as harness scenes, the same each time.
    #[test]
    fn r27_scenes_parse() {
        for s in EXT27_SEED..EXT27_END {
            let v = gen_scene(s);
            crate::read_scene_value(&v).unwrap_or_else(|e| panic!("seed {s}: {e}"));
            assert_eq!(v, gen_scene(s), "seed {s} not deterministic");
        }
    }
}
