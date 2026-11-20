//! opend2d Rust example — the standard Direct2D / DirectWrite / WIC API via the `windows` crate.
//!
//! This is how a Rust project consumes opend2d as a drop-in: the exact same `windows`-crate COM
//! calls it would make against the OS DLLs, resolved at runtime to our d2d1.dll / dwrite.dll /
//! windowscodecs.dll (under Wine via WINEDLLOVERRIDES, or beside the app on Windows). It renders
//! harness scene `02-fractional-rect` op-for-op, reads the target back through a CPU staging bitmap,
//! asserts byte-equality with the committed native oracle, then exercises path geometry, DrawText
//! and a Flood effect and saves both passes to PNG through WIC.
//!
//! `cargo run -p d2d-ffi --example example --target x86_64-pc-windows-gnu -- <oracle.bgra> <out-dir>`
//! under Wine; see examples/run-examples.sh. Prints "RUST EXAMPLE OK" and exits 0 on success.

fn main() {
    #[cfg(windows)]
    ex::run();
}

#[cfg(windows)]
mod ex {
    use windows::Win32::Graphics::Direct2D::Common::*;
    use windows::Win32::Graphics::Direct2D::*;
    use windows::Win32::Graphics::DirectWrite::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM;
    use windows::Win32::Graphics::Dxgi::IDXGIDevice;
    use windows::Win32::Graphics::Imaging::*;
    use windows::Win32::System::Com::*;
    use windows::core::{PCWSTR, w};
    use windows_numerics::Vector2;

    const W: u32 = 128;
    const H: u32 = 96;

    fn color(r: f32, g: f32, b: f32, a: f32) -> D2D1_COLOR_F {
        D2D1_COLOR_F { r, g, b, a }
    }
    fn rect(l: f32, t: f32, r: f32, b: f32) -> D2D_RECT_F {
        D2D_RECT_F { left: l, top: t, right: r, bottom: b }
    }

    unsafe fn make_bitmap(dc: &ID2D1DeviceContext, opts: D2D1_BITMAP_OPTIONS) -> windows::core::Result<ID2D1Bitmap1> {
        let props = D2D1_BITMAP_PROPERTIES1 {
            pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT_B8G8R8A8_UNORM, alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED },
            dpiX: 96.0,
            dpiY: 96.0,
            bitmapOptions: opts,
            ..Default::default()
        };
        unsafe { dc.CreateBitmap(D2D_SIZE_U { width: W, height: H }, None, 0, &props) }
    }

    /// Copy a rendered target into a CPU-read staging bitmap and read it out tightly packed.
    unsafe fn readback(dc: &ID2D1DeviceContext, target: &ID2D1Bitmap1) -> windows::core::Result<Vec<u8>> {
        unsafe {
            let staging = make_bitmap(dc, D2D1_BITMAP_OPTIONS_CPU_READ | D2D1_BITMAP_OPTIONS_CANNOT_DRAW)?;
            staging.CopyFromBitmap(None, target, None)?;
            let m = staging.Map(D2D1_MAP_OPTIONS_READ)?;
            let mut out = vec![0u8; (W * H * 4) as usize];
            let row = (W * 4) as usize;
            for y in 0..H as usize {
                let src = std::slice::from_raw_parts(m.bits.add(y * m.pitch as usize), row);
                out[y * row..(y + 1) * row].copy_from_slice(src);
            }
            staging.Unmap()?;
            Ok(out)
        }
    }

    /// Save tightly-packed BGRA8 pixels to `path` as a PNG through WIC — the standard encoder chain.
    unsafe fn save_png(wic: &IWICImagingFactory, path: &str, bgra: &[u8]) -> windows::core::Result<()> {
        unsafe {
            let wpath: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
            let stream = wic.CreateStream()?;
            stream.InitializeFromFilename(PCWSTR(wpath.as_ptr()), 0x4000_0000)?; // GENERIC_WRITE
            let enc = wic.CreateEncoder(&GUID_ContainerFormatPng, std::ptr::null())?;
            enc.Initialize(&stream, WICBitmapEncoderNoCache)?;
            let mut frame = None;
            enc.CreateNewFrame(&mut frame, std::ptr::null_mut())?;
            let frame = frame.unwrap();
            frame.Initialize(None)?;
            frame.SetSize(W, H)?;
            let mut fmt = GUID_WICPixelFormat32bppBGRA;
            frame.SetPixelFormat(&mut fmt)?;
            frame.WritePixels(H, W * 4, bgra)?;
            frame.Commit()?;
            enc.Commit()?;
            Ok(())
        }
    }

    pub fn run() {
        let args: Vec<String> = std::env::args().collect();
        let oracle_path = args.get(1).cloned();
        let out_dir = args.get(2).cloned().unwrap_or_else(|| ".".into());
        if let Err(e) = unsafe { render(oracle_path, &out_dir) } {
            eprintln!("FAIL {e:?}");
            std::process::exit(1);
        }
    }

    unsafe fn render(oracle_path: Option<String>, out_dir: &str) -> windows::core::Result<()> {
        unsafe {
            CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;

            let factory: ID2D1Factory1 = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, None)?;
            let no_dxgi: Option<&IDXGIDevice> = None; // our CreateDevice ignores it (CPU renderer)
            let device = factory.CreateDevice(no_dxgi)?;
            let dc = device.CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE)?;
            dc.SetDpi(96.0, 96.0);

            let wic: IWICImagingFactory = CoCreateInstance(&CLSID_WICImagingFactory, None, CLSCTX_INPROC_SERVER)?;

            // ---- pass 1: parity — op-for-op copy of scene 02-fractional-rect ----
            let target = make_bitmap(&dc, D2D1_BITMAP_OPTIONS_TARGET)?;
            dc.SetTarget(&target);
            dc.BeginDraw();
            dc.Clear(Some(&color(0.0, 0.0, 0.0, 0.0))); // harness clears transparent first
            dc.Clear(Some(&color(1.0, 1.0, 1.0, 1.0))); // scene op: clear white
            let ink = dc.CreateSolidColorBrush(&color(0.12, 0.35, 0.8, 1.0), None)?;
            let red = dc.CreateSolidColorBrush(&color(1.0, 0.0, 0.0, 0.5), None)?;
            dc.FillRectangle(&rect(10.25, 12.5, 110.75, 82.125), &ink);
            dc.FillRectangle(&rect(30.0, 25.0, 126.0, 55.0), &red);
            let (mut t1, mut t2) = (0u64, 0u64);
            dc.EndDraw(Some(&mut t1), Some(&mut t2))?;
            dc.SetTarget(None);

            let pixels = readback(&dc, &target)?;
            let parity = oracle_path.as_ref().map(|p| {
                std::fs::read(p).map(|o| o == pixels).unwrap_or(false)
            });
            save_png(&wic, &format!("{out_dir}/rust-parity.png"), &pixels)?;

            // ---- pass 2: showcase — path geometry, text, effect ----
            let canvas = make_bitmap(&dc, D2D1_BITMAP_OPTIONS_TARGET)?;
            dc.SetTarget(&canvas);
            dc.BeginDraw();
            dc.Clear(Some(&color(1.0, 1.0, 1.0, 1.0)));

            let tri = factory.CreatePathGeometry()?; // ID2D1Factory1 → ID2D1PathGeometry1
            let sink = tri.Open()?;
            sink.BeginFigure(Vector2 { X: 20.0, Y: 20.0 }, D2D1_FIGURE_BEGIN_FILLED);
            sink.AddLines(&[Vector2 { X: 100.0, Y: 40.0 }, Vector2 { X: 30.0, Y: 90.0 }]);
            sink.EndFigure(D2D1_FIGURE_END_CLOSED);
            sink.Close()?;
            let fill = dc.CreateSolidColorBrush(&color(0.0, 0.5, 0.5, 1.0), None)?;
            dc.FillGeometry(&tri, &fill, None);

            let dw: IDWriteFactory = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED)?;
            let tf = dw.CreateTextFormat(w!("Arial"), None, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL,
                                         DWRITE_FONT_STRETCH_NORMAL, 18.0, w!("en-us"))?;
            let text_brush = dc.CreateSolidColorBrush(&color(0.0, 0.0, 0.0, 1.0), None)?;
            let msg: Vec<u16> = "opend2d".encode_utf16().collect();
            dc.DrawText(&msg, &tf, &rect(8.0, 60.0, 126.0, 94.0), &text_brush, D2D1_DRAW_TEXT_OPTIONS_NONE, DWRITE_MEASURING_MODE_NATURAL);

            let flood = dc.CreateEffect(&CLSID_D2D1Flood)?;
            let fc = [1.0f32, 0.6, 0.0, 0.35]; // D2D1_FLOOD_PROP_COLOR, a VECTOR_4F
            flood.SetValue(0, D2D1_PROPERTY_TYPE_VECTOR4, std::slice::from_raw_parts(fc.as_ptr() as *const u8, 16))?;
            let flood_out = flood.GetOutput()?;
            dc.DrawImage(&flood_out, Some(&Vector2 { X: 0.0, Y: 0.0 }), Some(&rect(0.0, 0.0, 64.0, 24.0)),
                         D2D1_INTERPOLATION_MODE_LINEAR, D2D1_COMPOSITE_MODE_SOURCE_OVER);

            dc.EndDraw(Some(&mut t1), Some(&mut t2))?;
            dc.SetTarget(None);
            let shot = readback(&dc, &canvas)?;
            save_png(&wic, &format!("{out_dir}/rust-showcase.png"), &shot)?;

            CoUninitialize();
            println!(
                "RUST EXAMPLE OK (parity: {}; geometry+text+effect+WIC PNG saved)",
                match parity { Some(true) => "byte-exact vs native oracle", Some(false) => "MISMATCH", None => "no oracle given" }
            );
            if parity == Some(false) {
                std::process::exit(1);
            }
            Ok(())
        }
    }
}
