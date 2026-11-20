//! A foreign (non-opend2d) `IWICBitmapSource` — e.g. the system / Wine-builtin windowscodecs — fed to
//! the D2D1BitmapSource effect must render through the source's public vtable, not stay empty. Native
//! accepts any `IWICBitmapSource` implementation; we must not require it to be one of our own WIC
//! objects. Regression test for the Wine FFI sweep's empty fxbmsrc-01..09.

use windows::Win32::Graphics::Direct2D::Common::*;
use windows::Win32::Graphics::Direct2D::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::Win32::Graphics::Imaging::*;
use windows_core::{implement, Interface, GUID};

/// A minimal foreign `IWICBitmapSource`: a 2×2 32bpp premultiplied-BGRA image with four known opaque
/// texels. A plain `windows_core` object, NOT a wic-ffi type, so d2d-ffi cannot downcast it to a
/// concrete WIC object — it must read it via GetSize / GetPixelFormat / CopyPixels like native.
#[implement(IWICBitmapSource)]
struct ForeignSource;

const TEXELS: [[u8; 4]; 4] = [[10, 20, 30, 255], [40, 50, 60, 255], [70, 80, 90, 255], [100, 110, 120, 255]];

impl IWICBitmapSource_Impl for ForeignSource_Impl {
    fn GetSize(&self, pw: *mut u32, ph: *mut u32) -> windows_core::Result<()> {
        unsafe {
            *pw = 2;
            *ph = 2;
        }
        Ok(())
    }
    fn GetPixelFormat(&self) -> windows_core::Result<GUID> {
        Ok(GUID_WICPixelFormat32bppPBGRA)
    }
    fn GetResolution(&self, dx: *mut f64, dy: *mut f64) -> windows_core::Result<()> {
        unsafe {
            *dx = 96.0;
            *dy = 96.0;
        }
        Ok(())
    }
    fn CopyPalette(&self, _p: windows_core::Ref<IWICPalette>) -> windows_core::Result<()> {
        Err(windows_core::Error::from(windows_core::HRESULT(0x8898_2F04u32 as i32))) // WINCODEC_ERR_WRONGSTATE: no palette
    }
    fn CopyPixels(&self, _rc: *const WICRect, stride: u32, size: u32, buf: *mut u8) -> windows_core::Result<()> {
        let out = unsafe { std::slice::from_raw_parts_mut(buf, size as usize) };
        for y in 0..2usize {
            for x in 0..2usize {
                let o = y * stride as usize + x * 4;
                out[o..o + 4].copy_from_slice(&TEXELS[y * 2 + x]);
            }
        }
        Ok(())
    }
}

fn context() -> ID2D1DeviceContext {
    unsafe {
        let mut raw = std::ptr::null_mut();
        d2d1::win::D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &ID2D1Factory1::IID, std::ptr::null(), &mut raw).ok().unwrap();
        let f = ID2D1Factory1::from_raw(raw);
        f.CreateDevice(None).unwrap().CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE).unwrap()
    }
}
fn bgra_props(options: D2D1_BITMAP_OPTIONS) -> D2D1_BITMAP_PROPERTIES1 {
    D2D1_BITMAP_PROPERTIES1 {
        pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT(87), alphaMode: D2D1_ALPHA_MODE(1) }, // B8G8R8A8_UNORM premul
        dpiX: 96.0,
        dpiY: 96.0,
        bitmapOptions: options,
        ..Default::default()
    }
}

#[test]
fn bitmapsource_effect_renders_a_foreign_wic_source() {
    unsafe {
        let dc = context();
        let foreign: IWICBitmapSource = ForeignSource.into();

        let effect = dc.CreateEffect(&CLSID_D2D1BitmapSource).unwrap();
        // Property 0 = WicBitmapSource (an IUnknown). SetValue must accept the foreign source (before
        // the fix it fell through to E_NOTIMPL / was ignored, so the effect had no input → empty).
        let ptr = (foreign.as_raw() as usize as u64).to_le_bytes();
        effect.SetValue(0, D2D1_PROPERTY_TYPE_IUNKNOWN, &ptr).unwrap();
        let output = effect.GetOutput().unwrap();

        let target = dc.CreateBitmap(D2D_SIZE_U { width: 2, height: 2 }, None, 0, &bgra_props(D2D1_BITMAP_OPTIONS_TARGET)).unwrap();
        dc.SetTarget(&target);
        dc.BeginDraw();
        dc.Clear(Some(&D2D1_COLOR_F { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }));
        dc.DrawImage(&output, None, None, D2D1_INTERPOLATION_MODE_LINEAR, D2D1_COMPOSITE_MODE_SOURCE_COPY);
        dc.EndDraw(None, None).unwrap();
        dc.SetTarget(None);

        let cpu = dc.CreateBitmap(D2D_SIZE_U { width: 2, height: 2 }, None, 0, &bgra_props(D2D1_BITMAP_OPTIONS_CPU_READ | D2D1_BITMAP_OPTIONS_CANNOT_DRAW)).unwrap();
        cpu.CopyFromBitmap(None, &target, None).unwrap();
        let m = cpu.Map(D2D1_MAP_OPTIONS_READ).unwrap();
        let row0 = std::slice::from_raw_parts(m.bits, 8);
        let row1 = std::slice::from_raw_parts(m.bits.add(m.pitch as usize), 8);
        cpu.Unmap().unwrap();

        // Not empty, and a 1:1 SOURCE_COPY passthrough reproduces the source's opaque BGRA texels.
        assert!(row0.iter().chain(row1).any(|&b| b != 0), "BitmapSource effect rendered EMPTY from a foreign WIC source");
        assert_eq!(&row0[0..4], &TEXELS[0], "pixel (0,0)");
        assert_eq!(&row0[4..8], &TEXELS[1], "pixel (1,0)");
        assert_eq!(&row1[0..4], &TEXELS[2], "pixel (0,1)");
        assert_eq!(&row1[4..8], &TEXELS[3], "pixel (1,1)");
    }
}
