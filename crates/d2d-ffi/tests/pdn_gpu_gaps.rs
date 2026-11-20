//! Paint.NET 5 GPU-plugin gaps exercised over the real COM surface (no oracle; behaviour assertions).
//!
//! G14 `RegisterEffectFromString` accepts `type='iunknown'` properties (ComputeSharp.D2D1's
//!     `TransformMapper`/`ResourceTextureManager0..15`), and setting one on a custom effect is accepted.
//! G16 A command list set as an effect input *after* its output image was taken is still realised
//!     (native's output image is live, not a snapshot) — EndDraw does not latch E_INVALIDARG and the
//!     output is not blank.
//! G18 A gradient/bitmap-brush fill recorded into a command list with no bitmap target bound first is
//!     not dropped — the recorded content draws when the list is later composited.
use windows::Win32::Graphics::Direct2D::Common::*;
use windows::Win32::Graphics::Direct2D::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows_core::{Interface, OutRef, PCWSTR, HRESULT, IUnknown};
use windows_numerics::Vector2;

fn context() -> ID2D1DeviceContext {
    unsafe {
        let mut raw = std::ptr::null_mut();
        d2d1::win::D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &ID2D1Factory1::IID, std::ptr::null(), &mut raw).ok().unwrap();
        let f = ID2D1Factory1::from_raw(raw);
        f.CreateDevice(None).unwrap().CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE).unwrap()
    }
}
fn props(options: D2D1_BITMAP_OPTIONS) -> D2D1_BITMAP_PROPERTIES1 {
    D2D1_BITMAP_PROPERTIES1 {
        pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT(87), alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED },
        dpiX: 96.0,
        dpiY: 96.0,
        bitmapOptions: options,
        ..Default::default()
    }
}
/// Read back an 8×8 target and report whether any pixel is non-zero.
unsafe fn drew_anything(dc: &ID2D1DeviceContext, target: &ID2D1Bitmap1) -> bool {
    unsafe {
        let staging = dc.CreateBitmap(D2D_SIZE_U { width: 8, height: 8 }, None, 0, &props(D2D1_BITMAP_OPTIONS_CPU_READ | D2D1_BITMAP_OPTIONS_CANNOT_DRAW)).unwrap();
        staging.CopyFromBitmap(None, target, None).unwrap();
        let m = staging.Map(D2D1_MAP_OPTIONS_READ).unwrap();
        let any = (0..8usize).any(|y| std::slice::from_raw_parts(m.bits.add(y * m.pitch as usize), 8 * 4).iter().any(|&b| b != 0));
        staging.Unmap().unwrap();
        any
    }
}

#[test]
fn g18_brush_fill_recorded_without_a_bound_target_is_not_dropped() {
    unsafe {
        let dc = context();
        // SetTarget(commandList) with NO prior bitmap target — Paint.NET's GpuDrawingEffect records
        // OnDraw on a context with no target.
        let list = dc.CreateCommandList().unwrap();
        dc.SetTarget(&list);
        dc.BeginDraw();
        let stops = [
            D2D1_GRADIENT_STOP { position: 0.0, color: D2D1_COLOR_F { r: 1.0, g: 0.0, b: 0.0, a: 1.0 } },
            D2D1_GRADIENT_STOP { position: 1.0, color: D2D1_COLOR_F { r: 0.0, g: 0.0, b: 1.0, a: 1.0 } },
        ];
        // The ID2D1DeviceContext `…1` overload (exercises CreateGradientStopCollection1 too).
        let coll = dc
            .CreateGradientStopCollection(&stops, D2D1_COLOR_SPACE_SRGB, D2D1_COLOR_SPACE_SRGB, D2D1_BUFFER_PRECISION_8BPC_UNORM, D2D1_EXTEND_MODE_CLAMP, D2D1_COLOR_INTERPOLATION_MODE_STRAIGHT)
            .unwrap();
        let brush = dc
            .CreateLinearGradientBrush(
                &D2D1_LINEAR_GRADIENT_BRUSH_PROPERTIES { startPoint: Vector2 { X: 0.0, Y: 0.0 }, endPoint: Vector2 { X: 8.0, Y: 0.0 } },
                None,
                &coll,
            )
            .unwrap();
        dc.FillRectangle(&D2D_RECT_F { left: 0.0, top: 0.0, right: 8.0, bottom: 8.0 }, &brush);
        dc.EndDraw(None, None).expect("EndDraw must succeed for a brush fill recorded with no bound target (G18)");
        list.Close().unwrap();
        // Composite the list into a real target: the recorded gradient must appear (not blank).
        let target = dc.CreateBitmap(D2D_SIZE_U { width: 8, height: 8 }, None, 0, &props(D2D1_BITMAP_OPTIONS_TARGET)).unwrap();
        dc.SetTarget(&target);
        dc.BeginDraw();
        dc.Clear(Some(&D2D1_COLOR_F { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }));
        dc.DrawImage(&list, None, None, D2D1_INTERPOLATION_MODE_LINEAR, D2D1_COMPOSITE_MODE_SOURCE_OVER);
        dc.EndDraw(None, None).unwrap();
        assert!(drew_anything(&dc, &target), "gradient FillRectangle recorded with no bound target drew blank (G18)");
    }
}

#[test]
fn g16_command_list_input_set_after_getoutput_is_realised() {
    unsafe {
        let dc = context();
        // A command list with opaque content.
        let content = dc.CreateCommandList().unwrap();
        dc.SetTarget(&content);
        dc.BeginDraw();
        let solid = dc.CreateSolidColorBrush(&D2D1_COLOR_F { r: 1.0, g: 0.0, b: 0.0, a: 1.0 }, None).unwrap();
        dc.FillRectangle(&D2D_RECT_F { left: 0.0, top: 0.0, right: 8.0, bottom: 8.0 }, &solid);
        dc.EndDraw(None, None).unwrap();
        content.Close().unwrap();
        // Compose the graph and take the OUTPUT image FIRST, then set the command-list input — the order
        // Paint.NET's Meme Maker / Blur Selection Edge use. A snapshot-at-GetOutput would keep 0 inputs.
        let effect = dc.CreateEffect(&CLSID_D2D1GaussianBlur).unwrap();
        let output = effect.GetOutput().unwrap();
        effect.SetInput(0, &content, false);
        let target = dc.CreateBitmap(D2D_SIZE_U { width: 8, height: 8 }, None, 0, &props(D2D1_BITMAP_OPTIONS_TARGET)).unwrap();
        dc.SetTarget(&target);
        dc.BeginDraw();
        dc.Clear(Some(&D2D1_COLOR_F { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }));
        dc.DrawImage(&output, None, None, D2D1_INTERPOLATION_MODE_LINEAR, D2D1_COMPOSITE_MODE_SOURCE_OVER);
        dc.EndDraw(None, None).expect("EndDraw must not latch E_INVALIDARG from a 0-input effect (G16)");
        assert!(drew_anything(&dc, &target), "effect whose command-list input was set after GetOutput drew blank (G16)");
    }
}

// --- G14: a custom effect with an `iunknown` property ------------------------------------------

const G14_CLSID: windows_core::GUID = windows_core::GUID::from_u128(0x7c1e_0001_0002_0003_0004_000500060007);
const G14_XML: &str = "<?xml version='1.0'?><Effect>\
    <Property name='DisplayName' type='string' value='G14'/>\
    <Inputs><Input name='Source'/></Inputs>\
    <Property name='TransformMapper' type='iunknown'/></Effect>";

/// A minimal 1-input passthrough: a single clamp Border transform wired as the whole graph.
#[windows_core::implement(ID2D1EffectImpl)]
struct PassEffect;
#[allow(non_snake_case)]
impl ID2D1EffectImpl_Impl for PassEffect_Impl {
    fn Initialize(&self, ctx: windows_core::Ref<ID2D1EffectContext>, graph: windows_core::Ref<ID2D1TransformGraph>) -> windows_core::Result<()> {
        unsafe {
            let border: ID2D1TransformNode = ctx.as_ref().unwrap().CreateBorderTransform(D2D1_EXTEND_MODE_CLAMP, D2D1_EXTEND_MODE_CLAMP)?.cast()?;
            let g = graph.as_ref().unwrap();
            g.AddNode(&border)?;
            g.ConnectToEffectInput(0, &border, 0)?;
            g.SetOutputNode(&border)
        }
    }
    fn PrepareForRender(&self, _c: D2D1_CHANGE_TYPE) -> windows_core::Result<()> {
        Ok(())
    }
    fn SetGraph(&self, _g: windows_core::Ref<ID2D1TransformGraph>) -> windows_core::Result<()> {
        Ok(())
    }
}
unsafe extern "system" fn pass_factory(out: OutRef<IUnknown>) -> HRESULT {
    let e: ID2D1EffectImpl = PassEffect.into();
    match e.cast::<IUnknown>() {
        Ok(u) => {
            let _ = out.write(Some(u));
            HRESULT(0)
        }
        Err(e) => e.code(),
    }
}

#[test]
fn g14_iunknown_property_registers_and_accepts_a_value() {
    unsafe {
        let mut raw = std::ptr::null_mut();
        d2d1::win::D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &ID2D1Factory1::IID, std::ptr::null(), &mut raw).ok().unwrap();
        let factory = ID2D1Factory1::from_raw(raw);
        let dc = factory.CreateDevice(None).unwrap().CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE).unwrap();

        // Registration must not reject the `type='iunknown'` property (G14 — would be E_INVALIDARG).
        let xml: Vec<u16> = G14_XML.encode_utf16().chain(std::iter::once(0)).collect();
        factory
            .RegisterEffectFromString(&G14_CLSID, PCWSTR(xml.as_ptr()), None, Some(pass_factory))
            .expect("an effect with a type='iunknown' property must register (G14)");

        let effect = dc.CreateEffect(&G14_CLSID).unwrap();
        // Setting the IUnknown property is accepted (our executor ignores TransformMapper, as it does
        // not consume a transform mapper — native stores it for PrepareForRender).
        let value: IUnknown = factory.cast().unwrap();
        let ptr_bytes = (value.as_raw() as usize).to_ne_bytes();
        let name: Vec<u16> = "TransformMapper".encode_utf16().chain(std::iter::once(0)).collect();
        effect
            .SetValueByName(PCWSTR(name.as_ptr()), D2D1_PROPERTY_TYPE_IUNKNOWN, &ptr_bytes)
            .expect("setting an iunknown property on a custom effect must be accepted (G14)");

        factory.UnregisterEffect(&G14_CLSID).unwrap();
    }
}
