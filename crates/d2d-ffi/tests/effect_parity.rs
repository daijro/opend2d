use windows::Win32::Graphics::Direct2D::{Common::*, *};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows_core::Interface;

#[test]
fn native_effect_graphs_match_through_com() {
    let suite = std::env::var_os("D2D_SUITE").map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/suite"));
    if !suite.join("scenes").is_dir() { return; }
    unsafe {
        let mut raw = std::ptr::null_mut();
        d2d1::win::D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &ID2D1Factory1::IID, std::ptr::null(), &mut raw).ok().unwrap();
        let factory = ID2D1Factory1::from_raw(raw);
        let props = |options| D2D1_BITMAP_PROPERTIES1 {
            pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT(87), alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED },
            dpiX: 96.0, dpiY: 96.0, bitmapOptions: options, ..Default::default()
        };
        for name in [
            "custom-fx-04-affine-resample", "fx-gaussian-border-extend", "fx-transform-12-perspective",
            "fx-transform-13-transform3d", "fxhs-05-radius3",
            "fxstraighten-01-rotate-linear", "fxstraighten-02-maintain-size", "pdnv5-sharpenglitch",
        ] {
            let scene: serde_json::Value = serde_json::from_slice(&std::fs::read(suite.join("scenes").join(format!("{name}.json"))).unwrap()).unwrap();
            let size = D2D_SIZE_U { width: scene["size"][0].as_u64().unwrap() as u32, height: scene["size"][1].as_u64().unwrap() as u32 };
            let dc = factory.CreateDevice(None).unwrap().CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE).unwrap();
            let target = dc.CreateBitmap(size, None, 0, &props(D2D1_BITMAP_OPTIONS_TARGET)).unwrap();
            dc.SetTarget(&target);
            dc.BeginDraw();
            let mut images = std::collections::HashMap::<String, ID2D1Image>::new();
            for op in scene["ops"].as_array().unwrap() {
                match op["op"].as_str().unwrap() {
                    "clear" => {
                        let c: Vec<_> = op["color"].as_array().unwrap().iter().map(|v|v.as_f64().unwrap() as f32).collect();
                        dc.Clear(Some(&D2D1_COLOR_F { r: c[0], g: c[1], b: c[2], a: c[3] }));
                    }
                    "bitmap" => {
                        let s = D2D_SIZE_U { width: op["size"][0].as_u64().unwrap() as u32, height: op["size"][1].as_u64().unwrap() as u32 };
                        let rgba: Vec<_> = op["rgba"].as_array().unwrap().iter().map(|v|v.as_u64().unwrap() as u8).collect();
                        let bgra = d2d::bitmap::premultiply_rgba(&rgba);
                        let bm = dc.CreateBitmap(s, Some(bgra.as_ptr().cast()), s.width * 4, &props(D2D1_BITMAP_OPTIONS_NONE)).unwrap();
                        images.insert(op["id"].as_str().unwrap().to_owned(), bm.cast().unwrap());
                    }
                    "effect" => {
                        let guid = u128::from_str_radix(&op["clsid"].as_str().unwrap().replace('-', ""), 16).unwrap();
                        let fx = dc.CreateEffect(&windows_core::GUID::from_u128(guid)).unwrap();
                        for (i, input) in op["inputs"].as_array().unwrap().iter().enumerate() {
                            fx.SetInput(i as u32, &images[input.as_str().unwrap()], true);
                        }
                        for p in op["properties"].as_array().unwrap() {
                            let v = &p["value"]["value"];
                            let (ty, data) = match p["value"]["type"].as_str().unwrap() {
                                "enum" => (D2D1_PROPERTY_TYPE_ENUM, (v.as_u64().unwrap() as u32).to_le_bytes().to_vec()),
                                "bool" => (D2D1_PROPERTY_TYPE_BOOL, (v.as_bool().unwrap() as u32).to_le_bytes().to_vec()),
                                "float" => (D2D1_PROPERTY_TYPE_FLOAT, (v.as_f64().unwrap() as f32).to_le_bytes().to_vec()),
                                kind => {
                                    let ty = match kind {
                                        "vector2" => D2D1_PROPERTY_TYPE_VECTOR2,
                                        "vector3" => D2D1_PROPERTY_TYPE_VECTOR3,
                                        "matrix3x2" => D2D1_PROPERTY_TYPE_MATRIX_3X2,
                                        "matrix4x4" => D2D1_PROPERTY_TYPE_MATRIX_4X4,
                                        _ => panic!("unexpected property {kind}"),
                                    };
                                    (ty, v.as_array().unwrap().iter().flat_map(|v|(v.as_f64().unwrap() as f32).to_le_bytes()).collect())
                                }
                            };
                            fx.SetValue(p["index"].as_u64().unwrap() as u32, ty, &data).unwrap();
                        }
                        images.insert(op["id"].as_str().unwrap().to_owned(), fx.GetOutput().unwrap());
                    }
                    "transform" => {
                        let m: Vec<_> = op["matrix"].as_array().unwrap().iter().map(|v|v.as_f64().unwrap() as f32).collect();
                        dc.SetTransform(&windows_numerics::Matrix3x2 { M11:m[0], M12:m[1], M21:m[2], M22:m[3], M31:m[4], M32:m[5] });
                    }
                    "draw_image" => {
                        let p = windows_numerics::Vector2 { X:op["offset"][0].as_f64().unwrap() as f32, Y:op["offset"][1].as_f64().unwrap() as f32 };
                        dc.DrawImage(&images[op["image"].as_str().unwrap()], Some(&p), None,
                            D2D1_INTERPOLATION_MODE(op["interpolation"].as_i64().unwrap() as i32), D2D1_COMPOSITE_MODE(op["composite"].as_i64().unwrap() as i32));
                    }
                    kind => panic!("unexpected operation {kind}"),
                }
            }
            dc.EndDraw(None, None).unwrap();
            dc.SetTarget(None);
            let read = dc.CreateBitmap(size, None, 0, &props(D2D1_BITMAP_OPTIONS_CPU_READ | D2D1_BITMAP_OPTIONS_CANNOT_DRAW)).unwrap();
            read.CopyFromBitmap(None, &target, None).unwrap();
            let mapped = read.Map(D2D1_MAP_OPTIONS_READ).unwrap();
            let expected = std::fs::read(suite.join("oracle/win11-26100").join(format!("{name}.bgra"))).unwrap();
            for y in 0..size.height as usize {
                let actual = std::slice::from_raw_parts(mapped.bits.add(y * mapped.pitch as usize), size.width as usize * 4);
                let row = &expected[y * size.width as usize * 4..(y + 1) * size.width as usize * 4];
                assert_eq!(actual.iter().zip(row).position(|(a,b)|a != b), None, "{name} row{y}");
            }
            read.Unmap().unwrap();
        }
    }
}
