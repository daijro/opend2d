//! Negative-case HRESULT census for the D2D draw methods: trigger each draw guard that's reachable
//! through the public API (a resource from another device/factory, a null resource), capture the
//! deferred EndDraw HRESULT + tags, and print one line per case. Run with the system d2d1.dll (native
//! oracle) and with the drop-in to compare each HRESULT and which call reports it. Dev tool; Windows.
#![cfg_attr(not(windows), allow(unused_imports, dead_code))]

#[cfg(not(windows))]
fn main() {
    eprintln!("hresult-census only runs on Windows");
}

#[cfg(windows)]
use windows::{
    core::Interface,
    Win32::{
        Foundation::HMODULE,
        Graphics::{
            Direct2D::{Common::*, *},
            Direct3D::*,
            Direct3D11::*,
            Dxgi::Common::*,
            Dxgi::*,
        },
        System::Com::*,
    },
};

#[cfg(windows)]
unsafe fn make_dc() -> windows::core::Result<(ID2D1Factory1, ID2D1DeviceContext)> {
    let mut d3d = None;
    D3D11CreateDevice(None, D3D_DRIVER_TYPE_WARP, HMODULE::default(), D3D11_CREATE_DEVICE_BGRA_SUPPORT, Some(&[D3D_FEATURE_LEVEL_11_0]), D3D11_SDK_VERSION, Some(&mut d3d), None, None)?;
    let dxgi: IDXGIDevice = d3d.unwrap().cast()?;
    let factory: ID2D1Factory1 = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, None)?;
    let dc = factory.CreateDevice(&dxgi)?.CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE)?;
    Ok((factory, dc))
}

// A sibling device context on the SAME factory but a fresh D3D/D2D device — its resources are the
// same-factory, different-device case (native: D2DERR_WRONG_RESOURCE_DOMAIN, not WRONG_FACTORY).
#[cfg(windows)]
unsafe fn sibling_dc(factory: &ID2D1Factory1) -> windows::core::Result<ID2D1DeviceContext> {
    let mut d3d = None;
    D3D11CreateDevice(None, D3D_DRIVER_TYPE_WARP, HMODULE::default(), D3D11_CREATE_DEVICE_BGRA_SUPPORT, Some(&[D3D_FEATURE_LEVEL_11_0]), D3D11_SDK_VERSION, Some(&mut d3d), None, None)?;
    let dxgi: IDXGIDevice = d3d.unwrap().cast()?;
    factory.CreateDevice(&dxgi)?.CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE)
}

#[cfg(windows)]
fn bp() -> D2D1_BITMAP_PROPERTIES1 {
    D2D1_BITMAP_PROPERTIES1 {
        pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT_B8G8R8A8_UNORM, alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED },
        dpiX: 96.0, dpiY: 96.0, bitmapOptions: D2D1_BITMAP_OPTIONS_TARGET, ..Default::default()
    }
}

#[cfg(windows)]
fn main() -> anyhow::Result<()> {
    unsafe {
        CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;
        let (fa, dc) = make_dc()?; // device/context A — the render target
        let (fb, dcb) = make_dc()?; // device/factory B — foreign resources
        let size = D2D_SIZE_U { width: 16, height: 16 };
        let target = dc.CreateBitmap(size, None, 0, &bp())?;
        dc.SetTarget(&target);

        // Foreign resources (device/factory B).
        let px = [0u8; 4 * 4 * 4];
        let foreign_bitmap = dcb.CreateBitmap(D2D_SIZE_U { width: 4, height: 4 }, Some(px.as_ptr().cast()), 16, &D2D1_BITMAP_PROPERTIES1 {
            pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT_B8G8R8A8_UNORM, alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED }, dpiX: 96.0, dpiY: 96.0, ..Default::default()
        })?;
        let foreign_brush = dcb.CreateSolidColorBrush(&D2D1_COLOR_F { r: 1.0, g: 0.0, b: 0.0, a: 1.0 }, None)?;
        let foreign_rect_geo: ID2D1RectangleGeometry = fb.CreateRectangleGeometry(&D2D_RECT_F { left: 0.0, top: 0.0, right: 4.0, bottom: 4.0 })?;
        let own_brush = dc.CreateSolidColorBrush(&D2D1_COLOR_F { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }, None)?;
        let own_geo: ID2D1RectangleGeometry = fa.CreateRectangleGeometry(&D2D_RECT_F { left: 0.0, top: 0.0, right: 4.0, bottom: 4.0 })?;
        let own_bitmap = dc.CreateBitmap(D2D_SIZE_U { width: 4, height: 4 }, Some(px.as_ptr().cast()), 16, &D2D1_BITMAP_PROPERTIES1 {
            pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT_B8G8R8A8_UNORM, alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED }, dpiX: 96.0, dpiY: 96.0, ..Default::default()
        })?;

        // Sibling device on the SAME factory fa (different D2D/D3D device) — its bitmap/brush are the
        // same-factory, different-device case: native D2DERR_WRONG_RESOURCE_DOMAIN (0x88990015).
        let dc_sib = sibling_dc(&fa)?;
        let sib_bitmap = dc_sib.CreateBitmap(D2D_SIZE_U { width: 4, height: 4 }, Some(px.as_ptr().cast()), 16, &D2D1_BITMAP_PROPERTIES1 {
            pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT_B8G8R8A8_UNORM, alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED }, dpiX: 96.0, dpiY: 96.0, ..Default::default()
        })?;
        let sib_brush = dc_sib.CreateSolidColorBrush(&D2D1_COLOR_F { r: 0.0, g: 1.0, b: 0.0, a: 1.0 }, None)?;

        // Each case: (name, closure issuing the draw). Each runs in its own BeginDraw/EndDraw so the
        // latched HRESULT + tags are isolated. SetTags before the draw tags the failing op.
        let rect = D2D_RECT_F { left: 0.0, top: 0.0, right: 8.0, bottom: 8.0 };
        let cases: Vec<(&str, Box<dyn Fn()>)> = vec![
            ("DrawBitmap(foreign-device bitmap)", Box::new({ let dc = dc.clone(); let b = foreign_bitmap.clone(); move || { dc.DrawBitmap(&b, Some(&rect), 1.0, D2D1_INTERPOLATION_MODE_LINEAR, None, None); } })),
            ("FillGeometry(foreign-factory geometry)", Box::new({ let dc = dc.clone(); let g = foreign_rect_geo.clone(); let br = own_brush.clone(); move || { dc.FillGeometry(&g, &br, None); } })),
            ("FillGeometry(foreign-device brush)", Box::new({ let dc = dc.clone(); let g = own_geo.clone(); let br = foreign_brush.clone(); move || { dc.FillGeometry(&g, &br, None); } })),
            ("DrawGeometry(foreign-factory geometry)", Box::new({ let dc = dc.clone(); let g = foreign_rect_geo.clone(); let br = own_brush.clone(); move || { dc.DrawGeometry(&g, &br, 1.0, None); } })),
            ("FillRectangle(foreign-device brush)", Box::new({ let dc = dc.clone(); let br = foreign_brush.clone(); move || { dc.FillRectangle(&rect, &br); } })),
            // Same-factory, different-device resources (expect D2DERR_WRONG_RESOURCE_DOMAIN).
            ("DrawBitmap(same-factory other-device bitmap)", Box::new({ let dc = dc.clone(); let b = sib_bitmap.clone(); move || { dc.DrawBitmap(&b, Some(&rect), 1.0, D2D1_INTERPOLATION_MODE_LINEAR, None, None); } })),
            ("FillRectangle(same-factory other-device brush)", Box::new({ let dc = dc.clone(); let br = sib_brush.clone(); move || { dc.FillRectangle(&rect, &br); } })),
            // Wrong-type / invalid-value guards (expect E_INVALIDARG or a specific code, not an AV).
            ("DrawBitmap(invalid interpolation mode 99)", Box::new({ let dc = dc.clone(); let b = own_bitmap.clone(); move || { dc.DrawBitmap(&b, Some(&rect), 1.0, D2D1_INTERPOLATION_MODE(99), None, None); } })),
            ("FillRectangle(NaN rect)", Box::new({ let dc = dc.clone(); let br = own_brush.clone(); move || { dc.FillRectangle(&D2D_RECT_F { left: f32::NAN, top: 0.0, right: 8.0, bottom: 8.0 }, &br); } })),
            // DrawImage of a cross-domain bitmap (the dominant 1.1 draw path).
            ("DrawImage(foreign-factory bitmap)", Box::new({ let dc = dc.clone(); let img: ID2D1Image = foreign_bitmap.cast().unwrap(); move || { dc.DrawImage(&img, None, None, D2D1_INTERPOLATION_MODE_LINEAR, D2D1_COMPOSITE_MODE_SOURCE_OVER); } })),
            ("DrawImage(same-factory other-device bitmap)", Box::new({ let dc = dc.clone(); let img: ID2D1Image = sib_bitmap.cast().unwrap(); move || { dc.DrawImage(&img, None, None, D2D1_INTERPOLATION_MODE_LINEAR, D2D1_COMPOSITE_MODE_SOURCE_OVER); } })),
        ];

        for (i, (name, run)) in cases.iter().enumerate() {
            dc.BeginDraw();
            dc.SetTags(i as u64 + 1, 0);
            run();
            let (mut t1, mut t2) = (0u64, 0u64);
            let hr = dc.EndDraw(Some(&mut t1), Some(&mut t2));
            let code = hr.as_ref().err().map(|e| e.code().0).unwrap_or(0);
            println!("CASE {name}: hr=0x{:08x} tag1={t1} tag2={t2}", code as u32);
        }

        // Getter round-trips / defaults (SetMaximumTextureMemory, Set/GetRenderingControls).
        let dev: ID2D1Device = dc.GetDevice().unwrap().cast().unwrap();
        println!("GET MaxTextureMemory default = {}", dev.GetMaximumTextureMemory());
        dev.SetMaximumTextureMemory(123_456_789);
        println!("GET MaxTextureMemory after Set(123456789) = {}", dev.GetMaximumTextureMemory());
        let rc = dc.GetRenderingControls();
        println!("GET RenderingControls default: bufferPrecision={} tileW={} tileH={}", rc.bufferPrecision.0, rc.tileSize.width, rc.tileSize.height);
        let set = D2D1_RENDERING_CONTROLS { bufferPrecision: D2D1_BUFFER_PRECISION_16BPC_FLOAT, tileSize: D2D_SIZE_U { width: 256, height: 128 } };
        dc.SetRenderingControls(&set);
        let rc2 = dc.GetRenderingControls();
        println!("GET RenderingControls after Set(16bpc,256x128): bufferPrecision={} tileW={} tileH={}", rc2.bufferPrecision.0, rc2.tileSize.width, rc2.tileSize.height);

        // Text validation: CreateTextFormat font size (synchronous) + DrawText layout rect (deferred).
        use windows::Win32::Graphics::DirectWrite::*;
        let dw: IDWriteFactory = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED)?;
        for sz in [-3.4e38f32, 0.0, -1.0, f32::NAN, f32::INFINITY, 1e-9, 12.0] {
            let r = dw.CreateTextFormat(windows_core::w!("Segoe UI"), None, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, sz, windows_core::w!("en-us"));
            let code = r.as_ref().err().map(|e| e.code().0).unwrap_or(0);
            println!("CreateTextFormat(size={sz:e}) hr=0x{:08x}", code as u32);
        }
        let txt: Vec<u16> = "A".encode_utf16().collect();
        dc.SetDpi(96.0, 96.0); // device em == logical size
        // Undo the 16BPC_FLOAT the getter probe set (a float buffer breaks text rendering → E_NOINTERFACE).
        dc.SetRenderingControls(&D2D1_RENDERING_CONTROLS { bufferPrecision: D2D1_BUFFER_PRECISION_UNKNOWN, tileSize: D2D_SIZE_U { width: 1024, height: 1024 } });
        // DrawTextLayout huge em: is it rejected at CreateTextLayout (sync) or DrawTextLayout (deferred)?
        for em in [1e9f32, 12.0] {
            let fmt = dw.CreateTextFormat(windows_core::w!("Arial"), None, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, em, windows_core::w!("en-us"))?;
            match dw.CreateTextLayout(&txt, &fmt, 100.0, 100.0) {
                Ok(layout) => {
                    dc.BeginDraw();
                    dc.SetTags(88, 0);
                    dc.DrawTextLayout(D2D_POINT_2F { x: 0.0, y: 0.0 }, &layout, &own_brush, D2D1_DRAW_TEXT_OPTIONS_NONE);
                    let (mut a, _) = (0u64, 0u64);
                    let hr = dc.EndDraw(Some(&mut a), None);
                    println!("DrawTextLayout(em={em:e}) createlayout=OK enddraw=0x{:08x} tag1={a}", hr.as_ref().err().map(|e| e.code().0).unwrap_or(0) as u32);
                }
                Err(e) => println!("DrawTextLayout(em={em:e}) createlayout=0x{:08x}", e.code().0 as u32),
            }
        }
        // DrawGlyphRun huge em: build a run from a real Arial face.
        let mut coll: Option<IDWriteFontCollection> = None;
        dw.GetSystemFontCollection(&mut coll, false)?;
        let coll = coll.unwrap();
        let (mut fi, mut ex) = (0u32, windows::Win32::Foundation::BOOL(0));
        coll.FindFamilyName(windows_core::w!("Arial"), &mut fi, &mut ex)?;
        let face = coll.GetFontFamily(fi)?.GetFont(0)?.CreateFontFace()?;
        let mut gid = [0u16; 1];
        face.GetGlyphIndices(['A' as u32].as_ptr(), 1, gid.as_mut_ptr())?;
        for em in [1e9f32, 12.0] {
            let adv = [0.0f32];
            let run = DWRITE_GLYPH_RUN {
                fontFace: std::mem::ManuallyDrop::new(Some(face.clone())),
                fontEmSize: em, glyphCount: 1, glyphIndices: gid.as_ptr(), glyphAdvances: adv.as_ptr(),
                glyphOffsets: std::ptr::null(), isSideways: windows::Win32::Foundation::BOOL(0), bidiLevel: 0,
            };
            dc.BeginDraw();
            dc.SetTags(77, 0);
            dc.DrawGlyphRun(D2D_POINT_2F { x: 10.0, y: 30.0 }, &run, None, &own_brush, DWRITE_MEASURING_MODE_NATURAL);
            let (mut a, _) = (0u64, 0u64);
            let hr = dc.EndDraw(Some(&mut a), None);
            println!("DrawGlyphRun(em={em:e}) enddraw=0x{:08x} tag1={a}", hr.as_ref().err().map(|e| e.code().0).unwrap_or(0) as u32);
        }

        CoUninitialize();
    }
    Ok(())
}
