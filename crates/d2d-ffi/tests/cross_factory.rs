//! A resource created by one ID2D1Factory, used on a context from another factory, must be rejected
//! with D2DERR_WRONG_FACTORY (0x88990012) — never an out-of-bounds access. Plug-ins are untrusted, so
//! a cross-factory resource must not reach the engine. Draw calls latch the error (surfaced at EndDraw);
//! Create* calls return it directly. These cover the resource-taking calls whose factory check was a gap
//! (FillMesh, DrawGeometryRealization, DrawInk, DrawSpriteBatch, Create{Filled,Stroked}GeometryRealization)
//! plus the already-guarded DrawGeometry as the canonical repro.
#![allow(unsafe_op_in_unsafe_fn)] // COM-setup test helpers are wholly unsafe; a per-call block adds noise.
use windows::Win32::Graphics::Direct2D::Common::*;
use windows::Win32::Graphics::Direct2D::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows_core::Interface;
use core::mem::ManuallyDrop;

const WRONG_FACTORY: u32 = 0x8899_0012;

unsafe fn factory() -> ID2D1Factory1 {
    let mut raw = std::ptr::null_mut();
    d2d1::win::D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &ID2D1Factory1::IID, std::ptr::null(), &mut raw)
        .ok()
        .unwrap();
    ID2D1Factory1::from_raw(raw)
}

unsafe fn context(f: &ID2D1Factory1) -> ID2D1DeviceContext {
    let dc = f
        .CreateDevice(None)
        .unwrap()
        .CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE)
        .unwrap();
    let props = D2D1_BITMAP_PROPERTIES1 {
        pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT(87), alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED },
        dpiX: 96.0,
        dpiY: 96.0,
        bitmapOptions: D2D1_BITMAP_OPTIONS_TARGET,
        ..Default::default()
    };
    let target = dc.CreateBitmap(D2D_SIZE_U { width: 4, height: 4 }, None, 0, &props).unwrap();
    dc.SetTarget(&target);
    dc
}

unsafe fn rect_geom(f: &ID2D1Factory1) -> ID2D1RectangleGeometry {
    f.CreateRectangleGeometry(&D2D_RECT_F { left: 0.0, top: 0.0, right: 2.0, bottom: 2.0 }).unwrap()
}

// DrawGeometry with a geometry from another factory latches WRONG_FACTORY (the canonical repro, already
// guarded — a regression guard).
#[test]
fn draw_geometry_cross_factory() {
    unsafe {
        let (fa, fb) = (factory(), factory());
        let dc = context(&fa);
        let geo_b = rect_geom(&fb);
        let brush = dc.CreateSolidColorBrush(&D2D1_COLOR_F { r: 1.0, g: 0.0, b: 0.0, a: 1.0 }, None).unwrap();
        dc.BeginDraw();
        dc.DrawGeometry(&geo_b, &brush, 1.0, None);
        assert_eq!(dc.EndDraw(None, None).unwrap_err().code().0 as u32, WRONG_FACTORY);
    }
}

// FillMesh with a mesh from another factory latches WRONG_FACTORY (after the aliased-state check).
#[test]
fn fill_mesh_cross_factory() {
    unsafe {
        let (fa, fb) = (factory(), factory());
        let dc = context(&fa);
        let dc_b = context(&fb);
        let mesh_b = dc_b.CreateMesh().unwrap();
        let brush = dc.CreateSolidColorBrush(&D2D1_COLOR_F { r: 1.0, g: 0.0, b: 0.0, a: 1.0 }, None).unwrap();
        dc.BeginDraw();
        dc.SetAntialiasMode(D2D1_ANTIALIAS_MODE_ALIASED);
        dc.FillMesh(&mesh_b, &brush);
        assert_eq!(dc.EndDraw(None, None).unwrap_err().code().0 as u32, WRONG_FACTORY);
    }
}

// DrawGeometryRealization with a realization from another factory latches WRONG_FACTORY.
#[test]
fn draw_geometry_realization_cross_factory() {
    unsafe {
        let (fa, fb) = (factory(), factory());
        let dc = context(&fa);
        let dc1_b: ID2D1DeviceContext1 = context(&fb).cast().unwrap();
        let geo_b = rect_geom(&fb);
        let real_b = dc1_b.CreateFilledGeometryRealization(&geo_b, 0.25).unwrap();
        let brush = dc.CreateSolidColorBrush(&D2D1_COLOR_F { r: 1.0, g: 0.0, b: 0.0, a: 1.0 }, None).unwrap();
        let dc1: ID2D1DeviceContext1 = dc.cast().unwrap();
        dc.BeginDraw();
        dc1.DrawGeometryRealization(&real_b, &brush);
        assert_eq!(dc.EndDraw(None, None).unwrap_err().code().0 as u32, WRONG_FACTORY);
    }
}

// DrawInk with an ink from another factory latches WRONG_FACTORY.
#[test]
fn draw_ink_cross_factory() {
    unsafe {
        let (fa, fb) = (factory(), factory());
        let dc = context(&fa);
        let dc2_b: ID2D1DeviceContext2 = context(&fb).cast().unwrap();
        let ink_b = dc2_b.CreateInk(&D2D1_INK_POINT { x: 1.0, y: 1.0, radius: 1.0 }).unwrap();
        let brush = dc.CreateSolidColorBrush(&D2D1_COLOR_F { r: 1.0, g: 0.0, b: 0.0, a: 1.0 }, None).unwrap();
        let dc2: ID2D1DeviceContext2 = dc.cast().unwrap();
        dc.BeginDraw();
        dc2.DrawInk(&ink_b, &brush, None);
        assert_eq!(dc.EndDraw(None, None).unwrap_err().code().0 as u32, WRONG_FACTORY);
    }
}

// DrawSpriteBatch with a sprite batch from another factory latches WRONG_FACTORY (after the aliased check).
#[test]
fn draw_sprite_batch_cross_factory() {
    unsafe {
        let (fa, fb) = (factory(), factory());
        let dc = context(&fa);
        let dc3_b: ID2D1DeviceContext3 = context(&fb).cast().unwrap();
        let batch_b = dc3_b.CreateSpriteBatch().unwrap();
        let props = D2D1_BITMAP_PROPERTIES1 {
            pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT(87), alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED },
            dpiX: 96.0,
            dpiY: 96.0,
            ..Default::default()
        };
        let bmp = dc.CreateBitmap(D2D_SIZE_U { width: 2, height: 2 }, None, 0, &props).unwrap();
        let dc3: ID2D1DeviceContext3 = dc.cast().unwrap();
        dc.BeginDraw();
        dc.SetAntialiasMode(D2D1_ANTIALIAS_MODE_ALIASED);
        dc3.DrawSpriteBatch(&batch_b, 0, 0, &bmp, D2D1_BITMAP_INTERPOLATION_MODE_LINEAR, D2D1_SPRITE_OPTIONS_NONE);
        assert_eq!(dc.EndDraw(None, None).unwrap_err().code().0 as u32, WRONG_FACTORY);
    }
}

// CreateFilledGeometryRealization returns WRONG_FACTORY directly for a geometry from another factory.
#[test]
fn create_filled_realization_cross_factory() {
    unsafe {
        let (fa, fb) = (factory(), factory());
        let dc1: ID2D1DeviceContext1 = context(&fa).cast().unwrap();
        let geo_b = rect_geom(&fb);
        assert_eq!(dc1.CreateFilledGeometryRealization(&geo_b, 0.25).unwrap_err().code().0 as u32, WRONG_FACTORY);
    }
}

// CreateStrokedGeometryRealization returns WRONG_FACTORY directly for a geometry from another factory.
#[test]
fn create_stroked_realization_cross_factory() {
    unsafe {
        let (fa, fb) = (factory(), factory());
        let dc1: ID2D1DeviceContext1 = context(&fa).cast().unwrap();
        let geo_b = rect_geom(&fb);
        assert_eq!(
            dc1.CreateStrokedGeometryRealization(&geo_b, 0.25, 1.0, None).unwrap_err().code().0 as u32,
            WRONG_FACTORY
        );
    }
}

// (b) An allocation that cannot be satisfied returns E_OUTOFMEMORY (or E_INVALIDARG on size overflow),
// never an abort or null deref — an untrusted plug-in must not be able to trap the wasm instance with an
// absurd resource size. A huge single request exercises the fallible-allocation path (checked_mul +
// try_reserve_exact) deterministically; a real leak-to-exhaust loop would consume all RAM on the shared
// host, so the huge request is the safe proxy for the same guarantee.
#[test]
fn create_bitmap_absurd_size_is_oom_not_crash() {
    unsafe {
        let f = factory();
        let dc = context(&f);
        let props = D2D1_BITMAP_PROPERTIES1 {
            pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT(87), alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED },
            dpiX: 96.0,
            dpiY: 96.0,
            ..Default::default()
        };
        // 2^30 x 2^30 x 4bpp = 2^62 bytes: within isize::MAX (so it reaches the fallible reserve) yet no
        // allocator can satisfy it, so try_reserve_exact fails -> E_OUTOFMEMORY, without trapping.
        let huge = D2D_SIZE_U { width: 1 << 30, height: 1 << 30 };
        let code = dc.CreateBitmap(huge, None, 0, &props).unwrap_err().code().0 as u32;
        assert!(code == 0x8007_000E || code == 0x8007_0057, "expected E_OUTOFMEMORY/E_INVALIDARG, got {code:#x}");
    }
}

unsafe fn stroke_style(f: &ID2D1Factory1) -> ID2D1StrokeStyle1 {
    f.CreateStrokeStyle(&D2D1_STROKE_STYLE_PROPERTIES1::default(), None).unwrap()
}

// DrawGeometry with a stroke style from another factory latches WRONG_FACTORY (the geometry is same-factory).
#[test]
fn draw_geometry_stroke_style_cross_factory() {
    unsafe {
        let (fa, fb) = (factory(), factory());
        let dc = context(&fa);
        let geo_a = rect_geom(&fa);
        let stroke_b = stroke_style(&fb);
        let brush = dc.CreateSolidColorBrush(&D2D1_COLOR_F { r: 1.0, g: 0.0, b: 0.0, a: 1.0 }, None).unwrap();
        dc.BeginDraw();
        dc.DrawGeometry(&geo_a, &brush, 1.0, &stroke_b);
        assert_eq!(dc.EndDraw(None, None).unwrap_err().code().0 as u32, WRONG_FACTORY);
    }
}

// CreateStrokedGeometryRealization returns WRONG_FACTORY for a stroke style from another factory.
#[test]
fn create_stroked_realization_stroke_style_cross_factory() {
    unsafe {
        let (fa, fb) = (factory(), factory());
        let dc1: ID2D1DeviceContext1 = context(&fa).cast().unwrap();
        let geo_a = rect_geom(&fa);
        let stroke_b = stroke_style(&fb);
        assert_eq!(
            dc1.CreateStrokedGeometryRealization(&geo_a, 0.25, 1.0, &stroke_b).unwrap_err().code().0 as u32,
            WRONG_FACTORY
        );
    }
}

// DrawInk with an ink style from another factory latches WRONG_FACTORY (the ink is same-factory).
#[test]
fn draw_ink_style_cross_factory() {
    unsafe {
        let (fa, fb) = (factory(), factory());
        let dc = context(&fa);
        let dc2: ID2D1DeviceContext2 = dc.cast().unwrap();
        let ink_a = dc2.CreateInk(&D2D1_INK_POINT { x: 1.0, y: 1.0, radius: 1.0 }).unwrap();
        let dc2_b: ID2D1DeviceContext2 = context(&fb).cast().unwrap();
        let style_b = dc2_b.CreateInkStyle(None).unwrap();
        let brush = dc.CreateSolidColorBrush(&D2D1_COLOR_F { r: 1.0, g: 0.0, b: 0.0, a: 1.0 }, None).unwrap();
        dc.BeginDraw();
        dc2.DrawInk(&ink_a, &brush, &style_b);
        assert_eq!(dc.EndDraw(None, None).unwrap_err().code().0 as u32, WRONG_FACTORY);
    }
}

// PushLayer with a geometric mask from another factory latches WRONG_FACTORY; the layer still pops cleanly
// (the stack stays balanced).
#[test]
fn push_layer_mask_cross_factory() {
    unsafe {
        let (fa, fb) = (factory(), factory());
        let dc = context(&fa);
        let mask_b = rect_geom(&fb);
        let params = D2D1_LAYER_PARAMETERS1 {
            contentBounds: D2D_RECT_F { left: 0.0, top: 0.0, right: 4.0, bottom: 4.0 },
            opacity: 1.0,
            geometricMask: ManuallyDrop::new(Some(mask_b.cast().unwrap())),
            ..Default::default()
        };
        dc.BeginDraw();
        dc.PushLayer(&params, None);
        dc.PopLayer();
        assert_eq!(dc.EndDraw(None, None).unwrap_err().code().0 as u32, WRONG_FACTORY);
    }
}

// PushLayer with an opacity brush from another factory latches WRONG_FACTORY.
#[test]
fn push_layer_opacity_brush_cross_factory() {
    unsafe {
        let (fa, fb) = (factory(), factory());
        let dc = context(&fa);
        let brush_b = context(&fb)
            .CreateSolidColorBrush(&D2D1_COLOR_F { r: 1.0, g: 0.0, b: 0.0, a: 1.0 }, None)
            .unwrap();
        let params = D2D1_LAYER_PARAMETERS1 {
            contentBounds: D2D_RECT_F { left: 0.0, top: 0.0, right: 4.0, bottom: 4.0 },
            opacity: 1.0,
            opacityBrush: ManuallyDrop::new(Some(brush_b.cast().unwrap())),
            ..Default::default()
        };
        dc.BeginDraw();
        dc.PushLayer(&params, None);
        dc.PopLayer();
        assert_eq!(dc.EndDraw(None, None).unwrap_err().code().0 as u32, WRONG_FACTORY);
    }
}
