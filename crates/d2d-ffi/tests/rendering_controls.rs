//! The graph's intermediate precision must survive the COM facade.
use windows::Win32::Graphics::Direct2D::Common::*;
use windows::Win32::Graphics::Direct2D::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows_core::Interface;

#[test]
fn rendering_controls_drive_blur_intermediate_precision() {
    let suite = std::env::var_os("D2D_SUITE").map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/suite"));
    if !suite.join("scenes").is_dir() { return; }
    if [0,1,3,4,5].iter().any(|p| !suite.join("oracle/win11-26100").join(format!("com-g11-precision-{p}.bgra")).is_file()) { return; }
    unsafe {
        let mut raw = std::ptr::null_mut();
        d2d1::win::D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &ID2D1Factory1::IID, std::ptr::null(), &mut raw).ok().unwrap();
        let factory = ID2D1Factory1::from_raw(raw);
        let dc = factory.CreateDevice(None).unwrap().CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE).unwrap();
        let props = |format, options| D2D1_BITMAP_PROPERTIES1 { pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT(format), alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED }, dpiX: 96.0, dpiY: 96.0, bitmapOptions: options, ..Default::default() };
        let size = D2D_SIZE_U { width: 16, height: 16 };
        let px: Vec<f32> = (0..256).flat_map(|_| [0.0021,0.0021,0.0021,1.0]).collect();
        let src = dc.CreateBitmap(size,Some(px.as_ptr().cast()),256,&props(2,D2D1_BITMAP_OPTIONS_NONE)).unwrap();
        let blur = dc.CreateEffect(&d2d1::abi::CLSID_D2D1GaussianBlur).unwrap();
        blur.SetInput(0,&src,true);
        blur.SetValue(0,D2D1_PROPERTY_TYPE_FLOAT,&2f32.to_le_bytes()).unwrap();
        blur.SetValue(1,D2D1_PROPERTY_TYPE_ENUM,&1u32.to_le_bytes()).unwrap();
        blur.SetValue(2,D2D1_PROPERTY_TYPE_ENUM,&1u32.to_le_bytes()).unwrap();
        let gain = dc.CreateEffect(&d2d1::abi::CLSID_D2D1ColorMatrix).unwrap();
        gain.SetInput(0,&blur.GetOutput().unwrap(),true);
        let matrix: [f32;20] = [50.,0.,0.,0.,0.,50.,0.,0.,0.,0.,50.,0.,0.,0.,0.,1.,0.,0.,0.,0.];
        let bytes: Vec<u8> = matrix.into_iter().flat_map(f32::to_le_bytes).collect();
        gain.SetValue(0,D2D1_PROPERTY_TYPE_MATRIX_5X4,&bytes).unwrap();
        gain.SetValue(1,D2D1_PROPERTY_TYPE_ENUM,&1u32.to_le_bytes()).unwrap();
        gain.SetValue(2,D2D1_PROPERTY_TYPE_BOOL,&0u32.to_le_bytes()).unwrap();
        // Reuse the same graph while raising and lowering the context precision.
        for precision in [0,1,3,4,5,0] {
            let controls = D2D1_RENDERING_CONTROLS { bufferPrecision: D2D1_BUFFER_PRECISION(precision), tileSize: D2D_SIZE_U { width: 32, height: 64 } };
            dc.SetRenderingControls(&controls);
            let got = dc.GetRenderingControls();
            assert_eq!((got.bufferPrecision.0,got.tileSize.width,got.tileSize.height),(precision,32,64));
            let target = dc.CreateBitmap(size,None,0,&props(87,D2D1_BITMAP_OPTIONS_TARGET)).unwrap();
            dc.SetTarget(&target);
            dc.BeginDraw();
            dc.DrawImage(&gain.GetOutput().unwrap(),None,None,D2D1_INTERPOLATION_MODE_NEAREST_NEIGHBOR,D2D1_COMPOSITE_MODE_SOURCE_OVER);
            dc.EndDraw(None,None).unwrap();
            dc.SetTarget(None);
            let read = dc.CreateBitmap(size,None,0,&props(87,D2D1_BITMAP_OPTIONS_CPU_READ | D2D1_BITMAP_OPTIONS_CANNOT_DRAW)).unwrap();
            read.CopyFromBitmap(None,&target,None).unwrap();
            let m = read.Map(D2D1_MAP_OPTIONS_READ).unwrap();
            let expected = std::fs::read(suite.join("oracle/win11-26100").join(format!("com-g11-precision-{precision}.bgra"))).unwrap();
            for y in 0..16 { assert_eq!(std::slice::from_raw_parts(m.bits.add(y*m.pitch as usize),64), &expected[y*64..(y+1)*64], "precision {precision} row {y}"); }
            read.Unmap().unwrap();
        }
    }
}
