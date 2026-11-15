use windows::Win32::Graphics::Direct2D::{Common::*, *};
use windows::Win32::Graphics::DirectWrite::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows_core::{PCWSTR, Interface};

fn wide(s: &str) -> Vec<u16> { s.encode_utf16().chain(Some(0)).collect() }
use windows_numerics::{Matrix3x2, Vector2};

fn color(v: &serde_json::Value) -> D2D1_COLOR_F {
    D2D1_COLOR_F { r:v[0].as_f64().unwrap() as f32, g:v[1].as_f64().unwrap() as f32,
        b:v[2].as_f64().unwrap() as f32, a:v[3].as_f64().unwrap() as f32 }
}
fn point(v: &serde_json::Value) -> Vector2 {
    Vector2 { X:v[0].as_f64().unwrap() as f32, Y:v[1].as_f64().unwrap() as f32 }
}
fn rect(v: &serde_json::Value) -> D2D_RECT_F {
    let a=point(v);
    let b=Vector2 {X:v[2].as_f64().unwrap() as f32,Y:v[3].as_f64().unwrap() as f32};
    D2D_RECT_F {left:a.X.min(b.X),top:a.Y.min(b.Y),right:a.X.max(b.X),bottom:a.Y.max(b.Y)}
}

#[test]
fn native_text_pixels_through_com() {
    let suite=std::env::var_os("D2D_SUITE").map(std::path::PathBuf::from)
        .unwrap_or_else(||std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/suite"));
    if !suite.join("scenes").is_dir() || std::env::var_os("D2D_FONTS").is_none() { return; }
    unsafe {
        let mut raw=std::ptr::null_mut();
        d2d1::win::D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED,&ID2D1Factory1::IID,std::ptr::null(),&mut raw).ok().unwrap();
        let factory=ID2D1Factory1::from_raw(raw);
        dwrite::win::DWriteCreateFactory(DWRITE_FACTORY_TYPE_ISOLATED,&IDWriteFactory3::IID,&mut raw).ok().unwrap();
        let write3=IDWriteFactory3::from_raw(raw);
        let write=write3.cast::<IDWriteFactory>().unwrap();
        let mut failures=Vec::new();
        for name in ["text-05-cleartype-gdi-classic","text-georgia-ital-gray-48","text-07-rotated-cleartype","text-08-outline-large","text-aliased-02-small-linear",
            "text-grayscale-01-arial-x","text-impact-bold-gray-48","textbrush-01-grayscale-linear",
            "textmodes-04-cleartype-mmm","16-text-grayscale","17-text-aliased-gdi",
            "text-rot45-gray-arial","text-rot45-ct-arial","text-rot45-gray-consolas","text-rot45-gray-times","text-rot45-gray-neg","text-rot45-aliased-arial","text-near45-in-arial","text-near45-out-arial"] {
            let scene:serde_json::Value=serde_json::from_slice(&std::fs::read(suite.join("scenes").join(format!("{name}.json"))).unwrap()).unwrap();
            let size=D2D_SIZE_U {width:scene["size"][0].as_u64().unwrap() as u32,height:scene["size"][1].as_u64().unwrap() as u32};
            let dpi=scene["dpi"].as_f64().unwrap_or(96.0) as f32;
            let props=|options|D2D1_BITMAP_PROPERTIES1 {pixelFormat:D2D1_PIXEL_FORMAT {format:DXGI_FORMAT(87),alphaMode:D2D1_ALPHA_MODE_PREMULTIPLIED},dpiX:dpi,dpiY:dpi,bitmapOptions:options,..Default::default()};
            let dc=factory.CreateDevice(None).unwrap().CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE).unwrap();
            let rt=dc.cast::<ID2D1RenderTarget>().unwrap();
            let target=dc.CreateBitmap(size,None,0,&props(D2D1_BITMAP_OPTIONS_TARGET)).unwrap();
            dc.SetTarget(&target); dc.SetDpi(dpi,dpi);
            let params=write.CreateCustomRenderingParams(2.2,1.0,1.0,DWRITE_PIXEL_GEOMETRY_FLAT,DWRITE_RENDERING_MODE_DEFAULT).unwrap();
            dc.SetTextRenderingParams(&params); dc.BeginDraw();
            let (mut dx,mut dy)=(0.0,0.0);dc.GetDpi(&mut dx,&mut dy);assert_eq!((dx,dy),(dpi,dpi));
            let mut brushes=std::collections::HashMap::<String,ID2D1Brush>::new();
            for op in scene["ops"].as_array().unwrap() {
                match op["op"].as_str().unwrap() {
                    "clear"=>dc.Clear(Some(&color(&op["color"]))),
                    "solid"=>{
                        let b=dc.CreateSolidColorBrush(&color(&op["color"]),Some(&D2D1_BRUSH_PROPERTIES {opacity:op["opacity"].as_f64().unwrap_or(1.0) as f32,transform:Matrix3x2::identity()})).unwrap();
                        brushes.insert(op["id"].as_str().unwrap().to_owned(),b.cast().unwrap());
                    }
                    "linear"=>{
                        let stops:Vec<_>=op["stops"].as_array().unwrap().iter().map(|s|D2D1_GRADIENT_STOP {position:s["position"].as_f64().unwrap() as f32,color:color(&s["color"])}).collect();
                        let stops=rt.CreateGradientStopCollection(&stops,D2D1_GAMMA(op["gamma"].as_i64().unwrap() as i32),D2D1_EXTEND_MODE(op["extend"].as_i64().unwrap() as i32)).unwrap();
                        let b=dc.CreateLinearGradientBrush(&D2D1_LINEAR_GRADIENT_BRUSH_PROPERTIES {startPoint:point(&op["start"]),endPoint:point(&op["end"])},Some(&D2D1_BRUSH_PROPERTIES {opacity:op["opacity"].as_f64().unwrap_or(1.0) as f32,transform:Matrix3x2::identity()}),&stops).unwrap();
                        brushes.insert(op["id"].as_str().unwrap().to_owned(),b.cast().unwrap());
                    }
                    "transform"=>{
                        let m:Vec<_>=op["matrix"].as_array().unwrap().iter().map(|v|v.as_f64().unwrap() as f32).collect();
                        dc.SetTransform(&Matrix3x2 {M11:m[0],M12:m[1],M21:m[2],M22:m[3],M31:m[4],M32:m[5]});
                    }
                    "text"=>{
                        let format=write.CreateTextFormat(PCWSTR(wide(op["font"].as_str().unwrap()).as_ptr()),None,
                            DWRITE_FONT_WEIGHT(op["weight"].as_i64().unwrap_or(400) as i32),DWRITE_FONT_STYLE(op["style"].as_i64().unwrap_or(0) as i32),DWRITE_FONT_STRETCH_NORMAL,
                            op["size"].as_f64().unwrap() as f32,PCWSTR(wide(op["locale"].as_str().unwrap_or("en-us")).as_ptr())).unwrap();
                        dc.SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE(op["antialias"].as_i64().unwrap() as i32));
                        let text:Vec<_>=op["text"].as_str().unwrap().encode_utf16().collect();
                        rt.DrawText(&text,&format,&rect(&op["rect"]),&brushes[op["brush"].as_str().unwrap()],D2D1_DRAW_TEXT_OPTIONS_NONE,DWRITE_MEASURING_MODE(op["measuring"].as_i64().unwrap() as i32));
                    }
                    "glyph_run"=>{
                        let mut collection=None;write.GetSystemFontCollection(&mut collection,false).unwrap();let collection=collection.unwrap();
                        let(mut index,mut exists)=(0,Default::default());collection.FindFamilyName(PCWSTR(wide(op["font"].as_str().unwrap()).as_ptr()),&mut index,&mut exists).unwrap();assert!(exists.as_bool());
                        let face=collection.GetFontFamily(index).unwrap().GetFirstMatchingFont(DWRITE_FONT_WEIGHT(op["weight"].as_i64().unwrap_or(400) as i32),DWRITE_FONT_STRETCH_NORMAL,DWRITE_FONT_STYLE(op["style"].as_i64().unwrap_or(0) as i32)).unwrap().CreateFontFace().unwrap();
                        let codes:Vec<_>=op["text"].as_str().unwrap().chars().map(|c|c as u32).collect();let mut ids=vec![0;codes.len()];face.GetGlyphIndices(codes.as_ptr(),codes.len() as u32,ids.as_mut_ptr()).unwrap();
                        let mut fm=DWRITE_FONT_METRICS::default();face.GetMetrics(&mut fm);let mut gm=vec![DWRITE_GLYPH_METRICS::default();codes.len()];face.GetDesignGlyphMetrics(ids.as_ptr(),codes.len() as u32,gm.as_mut_ptr(),false).unwrap();
                        let em=op["size"].as_f64().unwrap() as f32;let advances:Vec<_>=gm.iter().map(|m|m.advanceWidth as f32*em/fm.designUnitsPerEm as f32).collect();
                        let run=DWRITE_GLYPH_RUN {fontFace:std::mem::ManuallyDrop::new(Some(face)),fontEmSize:em,glyphCount:ids.len() as u32,glyphIndices:ids.as_ptr(),glyphAdvances:advances.as_ptr(),glyphOffsets:std::ptr::null(),isSideways:false.into(),bidiLevel:0};
                        let params=write3.CreateCustomRenderingParams(2.2,1.0,1.0,1.0,DWRITE_PIXEL_GEOMETRY_FLAT,DWRITE_RENDERING_MODE1(op["rendering_mode"].as_i64().unwrap() as i32),DWRITE_GRID_FIT_MODE(op["grid_fit"].as_i64().unwrap() as i32)).unwrap();
                        dc.SetTextRenderingParams(&params);dc.SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE(op["antialias"].as_i64().unwrap() as i32));
                        rt.DrawGlyphRun(point(&op["baseline"]),&run,&brushes[op["brush"].as_str().unwrap()],DWRITE_MEASURING_MODE(op["measuring"].as_i64().unwrap() as i32));
                        drop(std::mem::ManuallyDrop::into_inner(run.fontFace));
                    }
                    "push_layer"=>{
                        let p=D2D1_LAYER_PARAMETERS1 {contentBounds:rect(&op["bounds"]),opacity:op["opacity"].as_f64().unwrap() as f32,maskTransform:Matrix3x2::identity(),..Default::default()};
                        dc.PushLayer(&p,None);
                    }
                    "pop_layer"=>dc.PopLayer(),
                    other=>panic!("unexpected operation {other}"),
                }
            }
            dc.EndDraw(None,None).unwrap();dc.SetTarget(None);
            let read=dc.CreateBitmap(size,None,0,&props(D2D1_BITMAP_OPTIONS_CPU_READ|D2D1_BITMAP_OPTIONS_CANNOT_DRAW)).unwrap();read.CopyFromBitmap(None,&target,None).unwrap();
            let mapped=read.Map(D2D1_MAP_OPTIONS_READ).unwrap();let expected=std::fs::read(suite.join("oracle/win11-26100").join(format!("{name}.bgra"))).unwrap();
            let mut diffs=0;
            for y in 0..size.height as usize {
                let row=std::slice::from_raw_parts(mapped.bits.add(y*mapped.pitch as usize),size.width as usize*4);
                diffs+=row.iter().zip(&expected[y*size.width as usize*4..]).filter(|(a,b)|a!=b).count();
            }
            read.Unmap().unwrap();if diffs!=0 {failures.push((name,diffs));}
        }
        assert!(failures.is_empty(),"native text COM mismatches: {failures:?}");
    }
}
