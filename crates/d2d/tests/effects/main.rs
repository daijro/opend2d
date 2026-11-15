//! Phase 2B effect-graph conformance: drive `custom-fx-01..06` through the public effect API
//! (`create_effect` → `set_input`/`set_value` → `draw_image_effect`) exactly as the harness drives
//! native `CreateEffect`/`SetInput`/`SetValue`/`DrawImage`, and compare the rendered target
//! byte-for-byte against `conformance/oracle/win11-26100`. These six scenes exercise the whole engine:
//! crop (hard output rect), border (per-axis extend), flood+composite (union bounds), 2D affine
//! (inverse-map + 8-bit bilinear), the precision chain (shader-linked ColorMatrix), and tiling.

use d2d::bitmap::Bitmap;
use d2d::effects::{EffectContext, EffectImpl, Input, Lut3d, Prop, Props, Transform, TransformGraph};
use d2d::state::{E_INVALIDARG, S_OK};
use d2d::{bitmap_target, Device, DeviceContext, Factory};
use d2d_reference::{read_scene, Op, Scene, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use windows_core::{GUID, HRESULT};

/// Byte-exact against the WARP oracle.
const EXACT: &[&str] = &[
    "custom-fx-01-crop-output-rect",
    "custom-fx-02-border-extend",
    "custom-fx-03-composite-union",
    "custom-fx-05-precision-chain",
    "custom-fx-06-tile",
];
/// 2D-affine resample. The R3 bilinear is correct, but the output carries the WARP rasterizer's
/// texcoord-gradient residual (its per-draw texcoord is accumulated in a reduced-precision fixed
/// point that Direct2D's gradient realization — not yet reproduced, the same gap 1E documents for
/// `DrawBitmap`) — so the soft-halo/seam pixels land ≤1 LSB off. Watched here so it can't regress
/// past that bound; not byte-exact, so not in `conformance/passing.txt`.
const AFFINE: &str = "custom-fx-04-affine-resample";

fn scene_path(name: &str) -> PathBuf {
    d2d_reference::suite_dir().unwrap().join("scenes").join(format!("{name}.json"))
}
fn oracle_bgra(name: &str) -> Vec<u8> {
    let p = d2d_reference::suite_dir().unwrap().join("oracle/win11-26100").join(format!("{name}.bgra"));
    std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

fn guid(clsid: &str) -> GUID {
    GUID::from_u128(u128::from_str_radix(&clsid.replace('-', ""), 16).expect("clsid hex"))
}

/// Scene `Value` → engine `Prop` (1:1).
fn prop(v: &Value) -> Prop {
    match *v {
        Value::Float(x) => Prop::F32(x),
        Value::Uint32(x) => Prop::U32(x),
        Value::Int32(x) => Prop::I32(x),
        Value::Enum(x) => Prop::U32(x),
        Value::Bool(x) => Prop::Bool(x),
        Value::Vector2(x) => Prop::V2(x),
        Value::Vector3(x) => Prop::V3(x),
        Value::Vector4(x) => Prop::V4(x),
        Value::Matrix3x2(x) => Prop::M32(x),
        Value::Matrix4x4(x) => Prop::M44(x),
        Value::Matrix5x4(x) => Prop::M54(x),
        Value::Floats(ref _x) => panic!("float-array property not used by custom-fx scenes"),
        Value::Iunknown(_) => panic!("iunknown property not used by custom-fx scenes"),
    }
}

/// Drive one scene through the public API and return the target's premultiplied BGRA8.
fn run(scene: &Scene) -> Vec<u8> {
    let factory = Factory::new(false);
    let device: Device = factory.create_device();
    let mut dc: DeviceContext = device.create_device_context();
    let [w, h] = scene.size;
    dc.set_dpi(scene.dpi, scene.dpi);
    dc.set_target_bitmap(bitmap_target(w, h, scene.dpi));
    dc.begin_draw();
    dc.clear(Some([0.0; 4]));

    let mut images: HashMap<&str, Input> = HashMap::new();
    for op in &scene.ops {
        match op {
            Op::Clear { color } => dc.clear(Some(*color)),
            Op::Transform { matrix } => dc.set_transform(*matrix),
            Op::Bitmap { id, size, rgba, .. } => {
                let b = Rc::new(Bitmap::from_straight_rgba(size[0], size[1], rgba));
                images.insert(id, Input::Bitmap(b));
            }
            Op::Effect { id, clsid, inputs, properties } => {
                let eff = dc.create_effect(guid(clsid)).expect("create_effect");
                for (i, input) in inputs.iter().enumerate() {
                    eff.set_input(i as u32, images[input.as_str()].clone());
                }
                for p in properties {
                    eff.set_value(p.index, prop(&p.value)).expect("set_value");
                }
                images.insert(id, Input::Effect(eff));
            }
            Op::DrawImage { image, offset, interpolation, composite, .. } => match &images[image.as_str()] {
                Input::Effect(e) => dc.draw_image_effect(e, *offset, *interpolation, *composite),
                Input::Bitmap(b) => dc.draw_image_bitmap(b, *offset, None, *interpolation, 0), // SOURCE_OVER
                Input::List(_) => unreachable!("custom-fx scenes bind no command list"),
            },
            other => panic!("custom-fx scene used an unexpected op: {other:?}"),
        }
    }
    let _ = dc.end_draw();
    dc.take_target().unwrap().data
}

/// First differing pixel `(x, y, channel)` between two BGRA buffers, or `None` if identical.
fn first_diff(got: &[u8], want: &[u8], width: u32) -> Option<String> {
    if got.len() != want.len() {
        return Some(format!("length {} vs oracle {}", got.len(), want.len()));
    }
    for (i, (a, b)) in got.chunks_exact(4).zip(want.chunks_exact(4)).enumerate() {
        if a != b {
            let ch = (0..4).find(|&c| a[c] != b[c]).unwrap();
            let name = ["B", "G", "R", "A"][ch];
            let p = i as u32;
            return Some(format!("pixel ({},{}) ch {name} got {a:?} want {b:?}", p % width, p / width));
        }
    }
    None
}

/// (#differing pixels, max abs channel delta) — characterises a near-miss.
fn diff_stats(got: &[u8], want: &[u8]) -> (usize, i32) {
    let mut n = 0;
    let mut md = 0;
    for (a, b) in got.chunks_exact(4).zip(want.chunks_exact(4)) {
        if a != b {
            n += 1;
            md = md.max((0..4).map(|c| (a[c] as i32 - b[c] as i32).abs()).max().unwrap());
        }
    }
    (n, md)
}

#[test]
fn effects_oracle() {
    if d2d_reference::suite_dir().is_none() { return; }
    let mut fails = Vec::new();
    for name in EXACT {
        let scene = read_scene(&scene_path(name)).unwrap_or_else(|e| panic!("read {name}: {e}"));
        let want = oracle_bgra(name);
        match first_diff(&run(&scene), &want, scene.size[0]) {
            None => println!("PASS {name}"),
            Some(diff) => fails.push(format!("{name}: {diff}")),
        }
    }
    assert!(fails.is_empty(), "effect-graph scenes not byte-exact:\n  {}", fails.join("\n  "));

    // Affine near-miss watchdog: correct algorithm, ≤1-LSB WARP texcoord-gradient residual.
    let scene = read_scene(&scene_path(AFFINE)).unwrap_or_else(|e| panic!("read {AFFINE}: {e}"));
    let (n, md) = diff_stats(&run(&scene), &oracle_bgra(AFFINE));
    println!("NEAR {AFFINE}: {n} px differ, max Δ{md} (WARP texcoord residual)");
    assert!(md <= 1, "{AFFINE} regressed past the ≤1-LSB residual: max Δ{md}");
}

/// Registration and property validation — specified from native, not
/// expressible as v1 scenes (no HRESULT assertions), so checked directly here.
#[test]
fn registration_and_validation() {
    use d2d::effects::NOTFOUND;
    let dc = Factory::new(false).create_device().create_device_context();
    // Unregistered CLSID → raw NOTFOUND (not D2DERR_EFFECT_IS_NOT_REGISTERED).
    assert_eq!(dc.create_effect(GUID::from_u128(0)).err(), Some(NOTFOUND));
    // Registered effect accepts a correctly-typed property and rejects a type mismatch.
    let crop = dc.create_effect(guid("e23f7110-0e9a-4324-af47-6a2c0c46f35b")).unwrap();
    assert!(crop.set_value(0, Prop::V4([0.0, 0.0, 4.0, 4.0])).is_ok());
    assert!(crop.set_value(0, Prop::U32(1)).is_err()); // RECT slot is vec4, not enum
    assert!(crop.set_value(0x8000_0007, Prop::U32(1)).is_ok()); // PRECISION system prop: accepted, ignored
}

// ================= custom effects (phase 2B part 2) ==========================================

/// A custom effect authored in Rust: one input + a `Color` (vector4) property. Its `Initialize`
/// builds a transform graph `Composite( BoundsAdjustment( Source(flood) ), input0 )` — exactly the
/// `Flood → Crop → Composite` graph of the `custom-fx-03` built-in scene, so its output must match
/// that scene's WARP oracle byte-for-byte (end-to-end: registration → schema → graph → realise).
struct CompositeFlood;
impl EffectImpl for CompositeFlood {
    fn initialize(&self, props: &Props, ctx: &EffectContext, graph: &mut TransformGraph) -> Result<(), HRESULT> {
        let color = props.get_v4(0, [0.0, 0.0, 0.0, 1.0]);
        let flood = graph.add_node(Transform::source(move |_, _| color));
        let crop = graph.add_node(ctx.create_bounds_adjustment_transform([0, 0, 6, 6]));
        graph.connect_node(flood, crop, 0)?;
        let comp = graph.add_node(ctx.create_blend_transform(4)); // D2D1_COMPOSITE_MODE 4
        graph.connect_node(crop, comp, 0)?; // dst = cropped flood
        graph.connect_to_effect_input(0, comp, 1)?; // src = effect input 0 (the bitmap)
        graph.set_output_node(comp)
    }
}

#[test]
fn custom_effect_reproduces_oracle() {
    if d2d_reference::suite_dir().is_none() { return; }
    let scene = read_scene(&scene_path("custom-fx-03-composite-union")).unwrap();
    let (bw, bh, rgba) = scene
        .ops
        .iter()
        .find_map(|op| match op {
            Op::Bitmap { size, rgba, .. } => Some((size[0], size[1], rgba.clone())),
            _ => None,
        })
        .unwrap();
    let transform = scene.ops.iter().find_map(|op| if let Op::Transform { matrix } = op { Some(*matrix) } else { None }).unwrap();

    let factory = Factory::new(false);
    let clsid = GUID::from_u128(0x0000_1111_2222_3333_4444_5555_6666_7777);
    factory.register_effect_from_string(clsid, CUSTOM_XML, Rc::new(|| Box::new(CompositeFlood))).unwrap();

    let device: Device = factory.create_device();
    let mut dc: DeviceContext = device.create_device_context();
    let [w, h] = scene.size;
    dc.set_dpi(scene.dpi, scene.dpi);
    dc.set_target_bitmap(bitmap_target(w, h, scene.dpi));
    dc.begin_draw();
    dc.clear(Some([1.0, 1.0, 1.0, 1.0]));

    let eff = dc.create_effect(clsid).expect("create custom effect");
    assert_eq!(eff.input_count(), 1);
    eff.set_input(0, Input::Bitmap(Rc::new(Bitmap::from_straight_rgba(bw, bh, &rgba))));
    eff.set_value_by_name("Color", Prop::V4([0.2, 0.8, 0.3, 0.75])).expect("set Color");
    dc.set_transform(transform);
    dc.draw_image_effect(&eff, [0.0, 0.0], 0, 0);
    let _ = dc.end_draw();
    let got = dc.take_target().unwrap().data;

    factory.unregister_effect(clsid).unwrap();
    assert_eq!(first_diff(&got, &oracle_bgra("custom-fx-03-composite-union"), w), None, "custom effect graph must match the Flood→Crop→Composite oracle");
}

const CUSTOM_XML: &str = r#"<?xml version='1.0'?>
<Effect>
  <Property name='DisplayName' type='string' value='CompositeFlood'/>
  <Inputs><Input name='Source'/></Inputs>
  <Property name='Color' type='vector4'/>
</Effect>"#;

#[test]
fn custom_registration_lifecycle() {
    use d2d::effects::NOTFOUND;
    let factory = Factory::new(false);
    let clsid = GUID::from_u128(0x1234_5678_9abc_def0_1122_3344_5566_7788);
    let make = || -> Rc<dyn Fn() -> Box<dyn EffectImpl>> { Rc::new(|| Box::new(CompositeFlood)) };
    let dc = factory.create_device().create_device_context();

    assert_eq!(dc.create_effect(clsid).err(), Some(NOTFOUND)); // not yet registered
    factory.register_effect_from_string(clsid, CUSTOM_XML, make()).unwrap();
    assert!(dc.create_effect(clsid).is_ok());
    // Re-register bumps a refcount; one unregister still leaves it live.
    factory.register_effect_from_string(clsid, CUSTOM_XML, make()).unwrap();
    factory.unregister_effect(clsid).unwrap();
    assert!(dc.create_effect(clsid).is_ok());
    factory.unregister_effect(clsid).unwrap();
    assert_eq!(dc.create_effect(clsid).err(), Some(NOTFOUND)); // refcount hit zero
    assert_eq!(factory.unregister_effect(clsid).err(), Some(NOTFOUND));
    // A built-in CLSID can't be overridden.
    assert!(factory.register_effect_from_string(guid("e23f7110-0e9a-4324-af47-6a2c0c46f35b"), CUSTOM_XML, make()).is_err());
}

// ================= LookupTable3D + resource validation (phase 2B part 2) =====================

#[test]
fn lookup_table_3d_and_resources() {
    let factory = Factory::new(false);
    let device: Device = factory.create_device();
    let mut dc: DeviceContext = device.create_device_context();
    dc.set_dpi(96.0, 96.0);
    dc.set_target_bitmap(bitmap_target(4, 4, 96.0));
    dc.begin_draw();
    dc.clear(Some([0.0, 0.0, 0.0, 0.0]));

    // A 2³ LUT that swaps red and blue: corner (r,g,b) -> (b,g,r). WARP's layout: red slowest,
    // blue fastest.
    let mut data = Vec::new();
    for r in 0..2 {
        for g in 0..2 {
            for b in 0..2 {
                data.push([b as f32, g as f32, r as f32, 1.0]);
            }
        }
    }
    let lut = dc.create_lookup_table3d([2, 2, 2], data).unwrap();
    let src = Bitmap::from_straight_rgba(4, 4, &[200u8, 10, 40, 255].repeat(16)); // opaque R=200,G=10,B=40
    let eff = dc.create_effect(d2d::abi::CLSID_D2D1LookupTable3D).expect("LookupTable3D");
    eff.set_input(0, Input::Bitmap(Rc::new(src)));
    eff.set_lookup_table(lut);
    dc.draw_image_effect(&eff, [0.0, 0.0], 0, 0);
    let _ = dc.end_draw();
    let out = dc.take_target().unwrap().data; // BGRA, top-left pixel
    // R and B swapped through WARP's Q8 trilinear sampler: input R=200 -> weight floor(200/255·256)/256
    // = 0.78125 -> 199 after the UNORM8 store; B=40 -> 40. Stored BGRA = [199,10,40,255].
    assert_eq!(&out[0..4], &[199, 10, 40, 255], "LUT3D should swap R and B");

    assert!(Lut3d::new([1, 2, 2], vec![[0.0; 4]; 4]).is_err()); // axis < 2
    assert!(dc.create_lookup_table3d([2, 2, 2], vec![[0.0; 4]; 3]).is_err()); // wrong data length
}

/// `CreateResourceTexture` validation runs inside `Initialize`; a bad texture fails `Initialize`,
/// which fails `CreateEffect`, while valid args succeed.
struct ResourceEffect {
    dimensions: u32,
}
impl EffectImpl for ResourceEffect {
    fn initialize(&self, _p: &Props, ctx: &EffectContext, graph: &mut TransformGraph) -> Result<(), HRESULT> {
        let ext = [2u32, 2, 2];
        let modes = [0u32, 0, 0];
        let d = self.dimensions as usize;
        ctx.create_resource_texture(self.dimensions, &ext[..d.min(3)], 1 /*8BPC_UNORM*/, 1 /*CHANNEL 1*/, &modes[..d.min(3)], &[0u8; 8])?;
        graph.set_passthrough_graph(0)
    }
}

#[test]
fn custom_resource_texture_validation() {
    let factory = Factory::new(false);
    let good = GUID::from_u128(0xaaaa_0000_0000_0000_0000_0000_0000_0001);
    let bad = GUID::from_u128(0xaaaa_0000_0000_0000_0000_0000_0000_0002);
    factory.register_effect_from_string(good, CUSTOM_XML, Rc::new(|| Box::new(ResourceEffect { dimensions: 2 }))).unwrap();
    factory.register_effect_from_string(bad, CUSTOM_XML, Rc::new(|| Box::new(ResourceEffect { dimensions: 4 }))).unwrap();
    let dc = factory.create_device().create_device_context();
    assert!(dc.create_effect(good).is_ok()); // valid 2-D texture
    assert!(dc.create_effect(bad).is_err()); // dimensions 4 -> E_INVALIDARG fails Initialize -> CreateEffect
    factory.unregister_effect(good).unwrap();
    factory.unregister_effect(bad).unwrap();
}

/// `set_single_transform_node` + a border (extend) node: wrap the input in both axes.
struct BorderWrap;
impl EffectImpl for BorderWrap {
    fn initialize(&self, _p: &Props, ctx: &EffectContext, g: &mut TransformGraph) -> Result<(), HRESULT> {
        g.set_single_transform_node(ctx.create_border_transform(1, 1)) // wrap X, wrap Y
    }
}
/// `set_single_transform_node` + an offset node: shift the input +1 in x.
struct OffsetBy1;
impl EffectImpl for OffsetBy1 {
    fn initialize(&self, _p: &Props, ctx: &EffectContext, g: &mut TransformGraph) -> Result<(), HRESULT> {
        g.set_single_transform_node(ctx.create_offset_transform([1, 0]))
    }
}
/// A nested built-in (`CreateEffect`) wrapped as a graph node (`CreateTransformNodeFromEffect`).
struct NestedFlood;
impl EffectImpl for NestedFlood {
    fn initialize(&self, _p: &Props, ctx: &EffectContext, g: &mut TransformGraph) -> Result<(), HRESULT> {
        let flood = ctx.create_effect(d2d::abi::CLSID_D2D1Flood)?;
        flood.set_value(0, Prop::V4([0.5, 0.25, 0.125, 1.0]))?;
        let n = g.add_node(ctx.create_transform_node_from_effect(flood));
        g.set_output_node(n)
    }
}

fn render_custom(clsid: GUID, imp: impl Fn() -> Box<dyn EffectImpl> + 'static, input: Option<Bitmap>) -> Vec<u8> {
    let factory = Factory::new(false);
    factory.register_effect_from_string(clsid, CUSTOM_XML, Rc::new(imp)).unwrap();
    let mut dc = factory.create_device().create_device_context();
    dc.set_dpi(96.0, 96.0);
    dc.set_target_bitmap(bitmap_target(4, 4, 96.0));
    dc.begin_draw();
    dc.clear(Some([0.0; 4]));
    let eff = dc.create_effect(clsid).unwrap();
    if let Some(b) = input {
        eff.set_input(0, Input::Bitmap(Rc::new(b)));
    }
    dc.draw_image_effect(&eff, [0.0, 0.0], 0, 0);
    let _ = dc.end_draw();
    factory.unregister_effect(clsid).unwrap();
    dc.take_target().unwrap().data
}

#[test]
fn custom_graph_node_types() {
    // 2×2 input: R, G / B, Y (premultiplied-opaque == straight).
    let input = || Bitmap::from_straight_rgba(2, 2, &[255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255]);
    let px = |buf: &[u8], x: usize, y: usize| [buf[(y * 4 + x) * 4], buf[(y * 4 + x) * 4 + 1], buf[(y * 4 + x) * 4 + 2], buf[(y * 4 + x) * 4 + 3]];

    // Border wrap: target(x,y) = input(x%2, y%2). (2,0)==(0,0)=R; (3,1)==(1,1)=Y.
    let b = render_custom(GUID::from_u128(0xb0b0_0001), || Box::new(BorderWrap), Some(input()));
    assert_eq!(px(&b, 2, 0), [0, 0, 255, 255]); // R (BGRA)
    assert_eq!(px(&b, 3, 1), [0, 255, 255, 255]); // Y

    // Offset +1 in x: (0,0) transparent, (1,0)==input(0,0)=R, (2,0)==input(1,0)=G.
    let o = render_custom(GUID::from_u128(0xb0b0_0002), || Box::new(OffsetBy1), Some(input()));
    assert_eq!(px(&o, 0, 0), [0, 0, 0, 0]);
    assert_eq!(px(&o, 1, 0), [0, 0, 255, 255]); // R
    assert_eq!(px(&o, 2, 0), [0, 255, 0, 255]); // G

    // Nested flood: every pixel is the flood colour (0.5,0.25,0.125,1) -> BGRA [32,64,128,255].
    let f = render_custom(GUID::from_u128(0xb0b0_0003), || Box::new(NestedFlood), None);
    assert_eq!(px(&f, 1, 2), [32, 64, 128, 255]);
}

/// G5: a custom effect's draw-time input requirement is the inputs its transform graph actually
/// references, not the declared schema `<Inputs>` count. A self-contained generator renders with no
/// input bound (though its XML declares one); a graph that wires an effect input left unbound latches
/// native's E_INVALIDARG instead of indexing a missing input (the panic `graph_valid` forecloses).
#[test]
fn custom_referenced_input_requirement() {
    let draw = |clsid: GUID, imp: fn() -> Box<dyn EffectImpl>, bind: bool| -> (HRESULT, Vec<u8>) {
        let factory = Factory::new(false);
        factory.register_effect_from_string(clsid, CUSTOM_XML, Rc::new(imp)).unwrap();
        let mut dc = factory.create_device().create_device_context();
        dc.set_dpi(96.0, 96.0);
        dc.set_target_bitmap(bitmap_target(4, 4, 96.0));
        dc.begin_draw();
        dc.clear(Some([0.0; 4]));
        let eff = dc.create_effect(clsid).unwrap();
        if bind {
            eff.set_input(0, Input::Bitmap(Rc::new(Bitmap::from_straight_rgba(2, 2, &[255u8; 16]))));
        }
        dc.draw_image_effect(&eff, [0.0, 0.0], 0, 0);
        let hr = dc.end_draw().0;
        factory.unregister_effect(clsid).unwrap();
        (hr, dc.take_target().unwrap().data)
    };
    let px = |buf: &[u8], x: usize, y: usize| [buf[(y * 4 + x) * 4], buf[(y * 4 + x) * 4 + 1], buf[(y * 4 + x) * 4 + 2], buf[(y * 4 + x) * 4 + 3]];

    // Generator: NestedFlood's XML declares one input but its graph is self-sourced — drawing it with
    // NO input bound succeeds and paints the flood colour, exactly as native does.
    let (hr, g) = draw(GUID::from_u128(0xb0b0_0010), || Box::new(NestedFlood), false);
    assert_eq!(hr, S_OK);
    assert_eq!(px(&g, 1, 2), [32, 64, 128, 255]);

    // Referenced-but-unset: OffsetBy1's graph wires effect input 0 (set_single_transform_node). With no
    // input bound it references an input that isn't there → E_INVALIDARG latched (no missing-index draw).
    assert_eq!(draw(GUID::from_u128(0xb0b0_0011), || Box::new(OffsetBy1), false).0, E_INVALIDARG);
    // ...and with the input bound the same effect renders fine.
    assert_eq!(draw(GUID::from_u128(0xb0b0_0012), || Box::new(OffsetBy1), true).0, S_OK);
}
