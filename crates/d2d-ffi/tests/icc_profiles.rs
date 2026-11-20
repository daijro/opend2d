use windows::Win32::Graphics::Direct2D::Common::*;
use windows::Win32::Graphics::Direct2D::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows_core::Interface;

#[test]
fn custom_icc_contexts_match_native_color_management() {
    let suite = std::env::var_os("D2D_SUITE").map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/suite"));
    if !suite.join("scenes").is_dir() { return; }
    unsafe {
        let mut raw = std::ptr::null_mut();
        d2d1::win::D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &ID2D1Factory1::IID, std::ptr::null(), &mut raw).ok().unwrap();
        let f = ID2D1Factory1::from_raw(raw);
        let dc = f.CreateDevice(None).unwrap().CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE).unwrap();
        let props = |options| D2D1_BITMAP_PROPERTIES1 { pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT(87), alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED }, dpiX: 96.0, dpiY: 96.0, bitmapOptions: options, ..Default::default() };
        let mut paths: Vec<_> = std::fs::read_dir(suite.join("scenes")).unwrap().flatten().map(|e|e.path()).filter(|p|p.file_stem().unwrap().to_string_lossy().starts_with("com-g13-") || p.file_stem().unwrap().to_string_lossy().starts_with("com-g14-")).collect();
        paths.sort();
        let mut failed = Vec::new();
        for path in paths {
            let name = path.file_stem().unwrap().to_str().unwrap();
            let scene: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            let ops = scene["ops"].as_array().unwrap();
            let rgba: Vec<u8> = ops[0]["rgba"].as_array().unwrap().iter().map(|v|v.as_u64().unwrap() as u8).collect();
            let data = ops[0]["data"].as_array().map(|a|a.iter().map(|v|v.as_u64().unwrap() as u8).collect::<Vec<_>>()).unwrap_or_else(||d2d::bitmap::premultiply_rgba(&rgba));
            let size = D2D_SIZE_U { width: scene["size"][0].as_u64().unwrap() as u32, height: scene["size"][1].as_u64().unwrap() as u32 };
            let src = dc.CreateBitmap(size, Some(data.as_ptr().cast()),size.width*4,&props(D2D1_BITMAP_OPTIONS_NONE)).unwrap();
            let mut contexts = std::collections::HashMap::new();
            for op in &ops[1..3] {
                let profile: Vec<u8> = op["profile"].as_array().map(|a|a.iter().map(|v|v.as_u64().unwrap() as u8).collect()).unwrap_or_default();
                let space = if profile.is_empty() { D2D1_COLOR_SPACE_SRGB } else { D2D1_COLOR_SPACE_CUSTOM };
                let context = dc.CreateColorContext(space, if profile.is_empty() { None } else { Some(&profile) }).unwrap();
                contexts.insert(op["id"].as_str().unwrap(),context);
            }
            let effect = dc.CreateEffect(&d2d1::abi::CLSID_D2D1ColorManagement).unwrap();
            for (index,expected) in [(1,0u32),(3,0),(4,1),(5,1)] {
                let mut value = [0u8;4];
                effect.GetValue(index,D2D1_PROPERTY_TYPE_ENUM,&mut value).unwrap();
                assert_eq!(u32::from_le_bytes(value),expected);
            }
            effect.SetInput(0,&src,true);
            for p in ops[3]["properties"].as_array().unwrap() {
                let index = p["index"].as_u64().unwrap() as u32;
                if p["value"]["type"] == "iunknown" {
                    let raw = contexts[p["value"]["value"].as_str().unwrap()].as_raw() as usize;
                    effect.SetValue(index,D2D1_PROPERTY_TYPE_COLOR_CONTEXT,&raw.to_ne_bytes()).unwrap();
                } else {
                    let v = p["value"]["value"].as_u64().unwrap() as u32;
                    effect.SetValue(index,D2D1_PROPERTY_TYPE_ENUM,&v.to_le_bytes()).unwrap();
                    let mut value = [0u8;4];
                    effect.GetValue(index,D2D1_PROPERTY_TYPE_ENUM,&mut value).unwrap();
                    assert_eq!(u32::from_le_bytes(value),v);
                }
            }
            let target = dc.CreateBitmap(size,None,0,&props(D2D1_BITMAP_OPTIONS_TARGET)).unwrap();
            dc.SetTarget(&target); dc.BeginDraw();
            dc.DrawImage(&effect.GetOutput().unwrap(),None,None,D2D1_INTERPOLATION_MODE_NEAREST_NEIGHBOR,D2D1_COMPOSITE_MODE_SOURCE_OVER);
            dc.EndDraw(None,None).unwrap(); dc.SetTarget(None);
            let read = dc.CreateBitmap(size,None,0,&props(D2D1_BITMAP_OPTIONS_CPU_READ | D2D1_BITMAP_OPTIONS_CANNOT_DRAW)).unwrap();
            read.CopyFromBitmap(None,&target,None).unwrap();
            let m = read.Map(D2D1_MAP_OPTIONS_READ).unwrap();
            let expected = std::fs::read(suite.join("oracle/win11-26100").join(format!("{name}.bgra"))).unwrap();
            let actual: Vec<u8> = (0..size.height as usize).flat_map(|y|std::slice::from_raw_parts(m.bits.add(y*m.pitch as usize),size.width as usize*4).iter().copied()).collect();
            if actual != expected {
                eprintln!("{name}: {:?}", actual.iter().zip(&expected).enumerate().filter(|(_, (a,b))|a!=b).map(|(i,(a,b))|(i,*a,*b)).collect::<Vec<_>>());
                failed.push(name.to_owned());
            }
            read.Unmap().unwrap();
        }
        assert!(failed.is_empty(), "ICC mismatches: {failed:?}");
    }
}
