//! Native WARP probes: uploads/copies preserve texels, including ignored alpha.
use windows::Win32::Graphics::Direct2D::Common::*;
use windows::Win32::Graphics::Direct2D::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows_core::Interface;

fn context() -> ID2D1DeviceContext {
    unsafe {
        let mut raw = std::ptr::null_mut();
        d2d1::win::D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &ID2D1Factory1::IID, std::ptr::null(), &mut raw).ok().unwrap();
        let f = ID2D1Factory1::from_raw(raw);
        f.CreateDevice(None).unwrap().CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE).unwrap()
    }
}
fn props(format: i32, alpha: i32, options: D2D1_BITMAP_OPTIONS) -> D2D1_BITMAP_PROPERTIES1 {
    D2D1_BITMAP_PROPERTIES1 { pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT(format), alphaMode: D2D1_ALPHA_MODE(alpha) }, dpiX: 96.0, dpiY: 96.0, bitmapOptions: options, ..Default::default() }
}
#[test]
fn upload_copy_map_preserves_format_and_raw_rows() {
    unsafe {
        let dc = context();
        for (format, bpp) in [(2,16), (10,8), (11,8), (65,1), (28,4), (87,4)] {
            for alpha in if format == 65 { [1,2] } else { [1,3] } {
                let pitch = 2*bpp+3;
                let data: Vec<u8> = (0..2*pitch).map(|i| (i*13+5) as u8).collect();
                let src = dc.CreateBitmap(D2D_SIZE_U { width: 2, height: 2 }, Some(data.as_ptr().cast()), pitch, &props(format, alpha, D2D1_BITMAP_OPTIONS_NONE)).unwrap();
                let pf = src.GetPixelFormat();
                assert_eq!((pf.format.0,pf.alphaMode.0), (format,alpha));
                let dst = dc.CreateBitmap(D2D_SIZE_U { width: 2, height: 2 }, None, 0, &props(format, alpha, D2D1_BITMAP_OPTIONS_CPU_READ | D2D1_BITMAP_OPTIONS_CANNOT_DRAW)).unwrap();
                dst.CopyFromBitmap(None, &src, None).unwrap();
                let check = |bmp: &ID2D1Bitmap1, expected: &[u8]| {
                    let m = bmp.Map(D2D1_MAP_OPTIONS_READ).unwrap();
                    for y in 0..2usize {
                        assert_eq!(std::slice::from_raw_parts(m.bits.add(y*m.pitch as usize), 2*bpp as usize), &expected[y*pitch as usize..y*pitch as usize+2*bpp as usize], "format {format} alpha {alpha} row {y}");
                    }
                    bmp.Unmap().unwrap();
                };
                check(&dst, &data);
                let replaced: Vec<u8> = data.iter().map(|v| v.wrapping_add(17)).collect();
                dst.CopyFromMemory(None, replaced.as_ptr().cast(), pitch).unwrap();
                check(&dst, &replaced);
                if format != 65 {
                    let target = dc.CreateBitmap(D2D_SIZE_U { width: 2, height: 2 },Some(data.as_ptr().cast()),pitch,&props(format,alpha,D2D1_BITMAP_OPTIONS_TARGET)).unwrap();
                    dc.SetTarget(&target);
                    assert_eq!(target.GetPixelFormat().format.0,format);
                    dc.BeginDraw(); dc.EndDraw(None,None).unwrap();
                    dst.CopyFromBitmap(None,&target,None).unwrap();
                    check(&dst,&data);
                    target.CopyFromMemory(None,replaced.as_ptr().cast(),pitch).unwrap();
                    dc.BeginDraw(); dc.EndDraw(None,None).unwrap();
                    dst.CopyFromBitmap(None,&target,None).unwrap();
                    check(&dst,&replaced);
                    dc.SetTarget(None);
                }
            }
        }
    }
}
#[test]
fn rgba_straight_alpha_is_rejected_like_native() {
    unsafe {
        let dc = context();
        for format in [2,10,11,28,87] {
            let result = dc.CreateBitmap(D2D_SIZE_U { width: 1, height: 1 }, Some([0u8;16].as_ptr().cast()), 16, &props(format, 2, D2D1_BITMAP_OPTIONS_NONE));
            assert_eq!(result.err().map(|e| e.code().0 as u32), Some(0x88982f80), "format {format}");
        }
    }
}

#[test]
fn bitmap_effect_pixels_match_native() {
    let suite = std::env::var_os("D2D_SUITE").map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/suite"));
    if !suite.join("scenes").is_dir() { return; }
    unsafe {
        for entry in std::fs::read_dir(suite.join("scenes")).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_stem().unwrap().to_str().unwrap();
            if !name.starts_with("com-g10-") { continue; }
            let scene: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            let op = &scene["ops"][0];
            let data: Vec<u8> = op["data"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u8).collect();
            let fmt = op["format"].as_i64().unwrap() as i32;
            let alpha = op["alpha"].as_i64().unwrap() as i32;
            let dc = context();
            let size = D2D_SIZE_U { width: 4, height: 1 };
            let src = dc.CreateBitmap(size, Some(data.as_ptr().cast()), data.len() as u32, &props(fmt, alpha, D2D1_BITMAP_OPTIONS_NONE)).unwrap();
            let target = dc.CreateBitmap(size, None, 0, &props(87, 1, D2D1_BITMAP_OPTIONS_TARGET)).unwrap();
            dc.SetTarget(&target);
            dc.BeginDraw();
            let fx = dc.CreateEffect(&d2d1::abi::CLSID_D2D1ColorMatrix).unwrap();
            fx.SetInput(0, &src, true);
            dc.DrawImage(&fx.GetOutput().unwrap(), None, None, D2D1_INTERPOLATION_MODE_NEAREST_NEIGHBOR, D2D1_COMPOSITE_MODE_SOURCE_OVER);
            dc.EndDraw(None,None).unwrap();
            dc.SetTarget(None);
            let read = dc.CreateBitmap(size, None, 0, &props(87,1,D2D1_BITMAP_OPTIONS_CPU_READ | D2D1_BITMAP_OPTIONS_CANNOT_DRAW)).unwrap();
            read.CopyFromBitmap(None,&target,None).unwrap();
            let m = read.Map(D2D1_MAP_OPTIONS_READ).unwrap();
            let expected = std::fs::read(suite.join("oracle/win11-26100").join(format!("{name}.bgra"))).unwrap();
            assert_eq!(std::slice::from_raw_parts(m.bits,16), expected, "{name}");
            read.Unmap().unwrap();
        }
    }
}

#[test]
fn empty_bitmap_remains_drawable_after_copy_from_memory() {
    unsafe {
        let dc = context();
        let size = D2D_SIZE_U { width: 2, height: 2 };
        let source = dc.CreateBitmap(size,None,0,&props(87,1,D2D1_BITMAP_OPTIONS_NONE)).unwrap();
        let target = dc.CreateBitmap(size,None,0,&props(87,1,D2D1_BITMAP_OPTIONS_TARGET)).unwrap();
        let read = dc.CreateBitmap(size,None,0,&props(87,1,D2D1_BITMAP_OPTIONS_CPU_READ | D2D1_BITMAP_OPTIONS_CANNOT_DRAW)).unwrap();
        // Match ffirects-04: no initial data, then upload and draw through ID2D1DeviceContext.
        // Repeat with new texels to ensure drawing reads the updated storage rather than a snapshot.
        for pixels in [[0u8;16], [0,0,255,255,0,255,0,255,255,0,0,255,255,255,255,255]] {
            source.CopyFromMemory(None,pixels.as_ptr().cast(),8).unwrap();
            dc.SetTarget(&target);
            dc.BeginDraw();
            dc.Clear(Some(&D2D1_COLOR_F::default()));
            dc.DrawBitmap(&source,None,1.0,D2D1_INTERPOLATION_MODE_NEAREST_NEIGHBOR,None,None);
            dc.EndDraw(None,None).unwrap();
            dc.SetTarget(None);
            read.CopyFromBitmap(None,&target,None).unwrap();
            let mapped = read.Map(D2D1_MAP_OPTIONS_READ).unwrap();
            for y in 0..2 {
                assert_eq!(std::slice::from_raw_parts(mapped.bits.add(y*mapped.pitch as usize),8),&pixels[y*8..y*8+8]);
            }
            read.Unmap().unwrap();
        }
    }
}
