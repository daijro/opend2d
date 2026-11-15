//! A command-list target switch shares the caller's BeginDraw/EndDraw scope.
use windows::Win32::Graphics::Direct2D::Common::*;
use windows::Win32::Graphics::Direct2D::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows_core::Interface;

#[test]
fn command_list_scope_preserves_native_draw_lifecycle() {
    let suite = std::env::var_os("D2D_SUITE").map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/suite"));
    if !suite.join("scenes").is_dir() { return; }
    let Ok(expected) = std::fs::read(suite.join("oracle/win11-26100/com-g9-scope.bgra")) else { return };
    unsafe {
        let mut raw = std::ptr::null_mut();
        d2d1::win::D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &ID2D1Factory1::IID, std::ptr::null(), &mut raw).ok().unwrap();
        let factory = ID2D1Factory1::from_raw(raw);
        let dc = factory.CreateDevice(None).unwrap().CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE).unwrap();
        let props = |format, options| D2D1_BITMAP_PROPERTIES1 { pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT(format), alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED }, dpiX: 96.0, dpiY: 96.0, bitmapOptions: options, ..Default::default() };
        let size = D2D_SIZE_U { width: 4, height: 4 };
        let target = dc.CreateBitmap(size,None,0,&props(87,D2D1_BITMAP_OPTIONS_TARGET)).unwrap();
        dc.SetTarget(&target);
        // Feather Object enters its drawing scope before its command-list scope. A second
        // BeginDraw is an error on native too; suppressing this error in OpenD2D would be wrong.
        dc.BeginDraw();
        let bad = dc.CreateCommandList().unwrap();
        dc.SetTarget(&bad);
        dc.BeginDraw();
        assert_eq!(dc.EndDraw(None,None).unwrap_err().code().0 as u32,0x88990001);
        dc.SetTarget(&target);
        // Close surfaces the error latched while the list was the recording target, even though EndDraw
        // already reported it — native returns D2DERR_WRONG_STATE here (clscope-probe, 2026-11-09).
        assert_eq!(bad.Close().unwrap_err().code().0 as u32,0x88990001);
        assert_eq!(dc.EndDraw(None,None).unwrap_err().code().0 as u32,0x88990001);

        dc.BeginDraw();
        dc.Clear(Some(&D2D1_COLOR_F { r:1.0,g:1.0,b:1.0,a:1.0 }));
        let list = dc.CreateCommandList().unwrap();
        dc.SetTarget(&list);
        let brush = dc.CreateSolidColorBrush(&D2D1_COLOR_F { r:1.0,g:0.0,b:0.0,a:1.0 },None).unwrap();
        dc.FillRectangle(&D2D_RECT_F { left:1.0,top:1.0,right:3.0,bottom:3.0 },&brush);
        dc.SetTarget(&target);
        list.Close().unwrap();
        dc.DrawImage(&list,None,None,D2D1_INTERPOLATION_MODE_LINEAR,D2D1_COMPOSITE_MODE_SOURCE_OVER);
        dc.EndDraw(None,None).unwrap();
        // A target remains copyable after EndDraw while it is still bound.
        let read = dc.CreateBitmap(size,None,0,&props(87,D2D1_BITMAP_OPTIONS_CPU_READ | D2D1_BITMAP_OPTIONS_CANNOT_DRAW)).unwrap();
        read.CopyFromBitmap(None,&target,None).unwrap();
        let m = read.Map(D2D1_MAP_OPTIONS_READ).unwrap();
        for y in 0..4 { assert_eq!(std::slice::from_raw_parts(m.bits.add(y*m.pitch as usize),16), &expected[y*16..(y+1)*16]); }
        read.Unmap().unwrap();
    }
}
