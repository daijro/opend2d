//! `ID2D1RenderTarget::CreateSharedBitmap` (item 4): sharing another ID2D1Bitmap carries its pixels.
use windows::Win32::Graphics::Direct2D::Common::*;
use windows::Win32::Graphics::Direct2D::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows_core::Interface;

#[test]
fn shared_bitmap_from_d2d_bitmap_carries_its_pixels() {
    unsafe {
        let mut raw = std::ptr::null_mut();
        d2d1::win::D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &ID2D1Factory1::IID, std::ptr::null(), &mut raw).ok().unwrap();
        let f = ID2D1Factory1::from_raw(raw);
        let dc = f.CreateDevice(None).unwrap().CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE).unwrap();
        let props = |options| D2D1_BITMAP_PROPERTIES1 {
            pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT(87), alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED },
            dpiX: 96.0,
            dpiY: 96.0,
            bitmapOptions: options,
            ..Default::default()
        };
        // A source bitmap with known pixels.
        let data: Vec<u8> = (0..2 * 2 * 4).map(|i| (i * 7 + 1) as u8).collect();
        let source = dc.CreateBitmap(D2D_SIZE_U { width: 2, height: 2 }, Some(data.as_ptr().cast()), 2 * 4, &props(D2D1_BITMAP_OPTIONS_NONE)).unwrap();

        // Share it as a 1.0 ID2D1Bitmap (riid identifies the resource; our impl recovers it by QI).
        let legacy: D2D1_BITMAP_PROPERTIES = D2D1_BITMAP_PROPERTIES { pixelFormat: props(D2D1_BITMAP_OPTIONS_NONE).pixelFormat, dpiX: 96.0, dpiY: 96.0 };
        let mut shared: Option<ID2D1Bitmap> = None;
        dc.CreateSharedBitmap(&ID2D1Bitmap::IID, source.as_raw(), Some(&legacy), &mut shared).unwrap();
        let shared = shared.expect("shared bitmap written");
        let shared1: ID2D1Bitmap1 = shared.cast().unwrap();
        let sz = shared.GetSize();
        assert_eq!((sz.width, sz.height), (2.0, 2.0));

        // Read the shared bitmap back: it must carry the source's pixels.
        let staging = dc.CreateBitmap(D2D_SIZE_U { width: 2, height: 2 }, None, 0, &props(D2D1_BITMAP_OPTIONS_CPU_READ | D2D1_BITMAP_OPTIONS_CANNOT_DRAW)).unwrap();
        staging.CopyFromBitmap(None, &shared1, None).unwrap();
        let m = staging.Map(D2D1_MAP_OPTIONS_READ).unwrap();
        for y in 0..2usize {
            let row = std::slice::from_raw_parts(m.bits.add(y * m.pitch as usize), 2 * 4);
            assert_eq!(row, &data[y * 8..y * 8 + 8], "row {y}");
        }
        staging.Unmap().unwrap();
    }
}
