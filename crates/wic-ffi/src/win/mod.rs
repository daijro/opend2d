//! The Windows COM façade: real WIC objects built with the official `windows` IIDs/vtables via
//! `#[implement]`, each method forwarding to `d2d::wic`. Only the surface the harness `wic-dump.exe`
//! exercises for the WIC fixture corpus is wired; every other method returns `E_NOTIMPL`/a default.

#![allow(non_snake_case)]

use core::ffi::c_void;
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Imaging::*;
use windows::Win32::System::Com::IClassFactory;
use windows_core::{GUID, HRESULT, IUnknown, Interface, Ref, Result as WResult, implement};

mod color;
mod components;
mod decode;
mod encode;
mod factory;
#[cfg(test)]
mod bitmap_tests;
mod meta;
#[cfg(test)]
mod compinfo_tests;
mod process;
mod stream;

use factory::{FactoryObj, PaletteObj};

/// A `windows_core::Error` from an HRESULT (the exact `WINCODEC_ERR_*` an application should see).
pub(crate) fn e(hr: HRESULT) -> windows_core::Error {
    windows_core::Error::from(hr)
}

/// A `WINCODEC_ERR_*` HRESULT code (`u32`) → `windows_core::Error`. Core decoders report these codes.
pub(crate) fn ecode(hr: u32) -> windows_core::Error {
    e(HRESULT(hr as i32))
}

/// `d2d::wic::WicError` → the exact `WINCODEC_ERR_*` HRESULT error.
pub(crate) fn wicerr(err: d2d::wic::WicError) -> windows_core::Error {
    ecode(err.hresult())
}

/// Canonical `GUID_WICPixelFormat*` string (or any GUID string) → `GUID`.
pub(crate) fn guid_from_str(s: &str) -> GUID {
    let hex: String = s.chars().filter(|c| c.is_ascii_hexdigit()).collect();
    GUID::from_u128(u128::from_str_radix(&hex, 16).unwrap_or(0))
}

/// `GUID` → its canonical lowercase string (to look a pixel format up via `d2d::wic`'s string API).
pub(crate) fn guid_to_str(g: &GUID) -> String {
    let d = g.data4;
    format!(
        "{:08x}-{:04x}-{:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        g.data1, g.data2, g.data3, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]
    )
}

/// Bits-per-pixel of a WIC pixel format. The standard `6fddc324-…` family's bpp is a fixed function
/// of the final GUID byte (the only formats our decoders / converter emit); `32bppPRGBA` is the one
/// non-family format. 0 for anything else.
pub(crate) fn bpp_of(g: &GUID) -> u32 {
    // 32bppPRGBA = {3cc4a650-a527-4d37-a916-3142c7ebedba}
    if g.data1 == 0x3cc4a650 {
        return 32;
    }
    // 32bppRGBA (straight) = {f5c7ad2d-6a8d-43dd-a7a8-a29935261ae9}
    if g.data1 == 0xf5c7ad2d {
        return 32;
    }
    if g.data1 != 0x6fddc324 {
        return 0;
    }
    match g.data4[7] {
        0x01 | 0x05 => 1,
        0x02 | 0x06 => 2,
        0x03 | 0x07 => 4,
        0x04 | 0x08 => 8,
        0x09 | 0x0a | 0x0b => 16,
        0x0c | 0x0d => 24,
        0x0e | 0x0f | 0x10 | 0x11 | 0x1c => 32, // …dc911 = 32bppGrayFloat
        0x15 => 48,
        0x16 => 64,
        _ => 0,
    }
}

/// The universal read-side payload: every WIC source object (decoder frame, bitmap, converter,
/// scaler, flip-rotator, clipper) holds one of these and exposes it through `IWICBitmapSource`.
/// Pixels are tightly strided (`stride = ⌈width·bpp/8⌉`), exactly as `CopyPixels` returns them.
#[derive(Clone)]
pub(crate) struct Source {
    pub format: GUID,
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub pixels: Vec<u8>,
    /// Palette as `0xAARRGGBB` (indexed formats only; empty → no palette).
    pub palette: Vec<u32>,
    pub dpi: (f64, f64),
}

impl Source {
    pub(crate) fn get_size(&self, pw: *mut u32, ph: *mut u32) -> WResult<()> {
        unsafe {
            *pw = self.width;
            *ph = self.height;
        }
        Ok(())
    }

    pub(crate) fn copy_palette(&self, pal: Ref<IWICPalette>) -> WResult<()> {
        if self.palette.is_empty() {
            return Err(e(WINCODEC_ERR_PALETTEUNAVAILABLE));
        }
        unsafe { pal.as_ref().ok_or(e(E_POINTER))?.InitializeCustom(&self.palette) }
    }

    pub(crate) fn copy_pixels(&self, prc: *const WICRect, stride: u32, buf: &mut [u8]) -> WResult<()> {
        let bpp = bpp_of(&self.format);
        let (x, y, w, h) = pixel_rect(prc, self.width, self.height)?;
        let row = buffer_size(w, h, bpp, stride, buf.len())?.0;
        let bit_offset = x as usize * bpp as usize;
        for r in 0..h as usize {
            let source = &self.pixels[(y as usize + r) * self.stride..];
            let dest = &mut buf[r * stride as usize..r * stride as usize + row];
            if bit_offset & 7 == 0 {
                dest.copy_from_slice(&source[bit_offset / 8..bit_offset / 8 + row]);
            } else {
                copy_bits(source, bit_offset, dest, 0, w as usize * bpp as usize);
            }
        }
        Ok(())
    }
}

// Native windowscodecs!HrValidatePixelRects and HrCheckBufferSize (Windows 11 26100).
pub(crate) fn pixel_rect(prc: *const WICRect, width: u32, height: u32) -> WResult<(u32,u32,u32,u32)> {
    let overflow = || e(HRESULT(0x80070216u32 as i32));
    if width > i32::MAX as u32 || height > i32::MAX as u32 { return Err(overflow()); }
    if prc.is_null() { return Ok((0,0,width,height)); }
    let r = unsafe { &*prc };
    if r.X < 0 || r.Y < 0 || r.Width < 0 || r.Height < 0 { return Err(overflow()); }
    let right = r.X.checked_add(r.Width).ok_or_else(overflow)?;
    let bottom = r.Y.checked_add(r.Height).ok_or_else(overflow)?;
    if r.Width == 0 || r.Height == 0 || right as u32 > width || bottom as u32 > height { return Err(e(E_INVALIDARG)); }
    Ok((r.X as u32,r.Y as u32,r.Width as u32,r.Height as u32))
}

pub(crate) fn buffer_size(w: u32, h: u32, bpp: u32, stride: u32, size: usize) -> WResult<(usize,usize)> {
    let overflow = || e(HRESULT(0x80070216u32 as i32));
    let row = w.checked_mul(bpp).and_then(|n|n.checked_add(7)).ok_or_else(overflow)? / 8;
    if stride < row { return Err(e(E_INVALIDARG)); }
    let needed = if h == 0 { 0 } else { stride.checked_mul(h-1).and_then(|n|n.checked_add(row)).ok_or_else(overflow)? };
    if size < needed as usize { return Err(e(WINCODEC_ERR_INSUFFICIENTBUFFER)); }
    Ok((row as usize,needed as usize))
}

pub(crate) fn copy_bits(src: &[u8], src_bit: usize, dst: &mut [u8], dst_bit: usize, count: usize) {
    for bit in 0..count {
        let s=src_bit+bit;
        let d=dst_bit+bit;
        let mask=0x80 >> (d&7);
        dst[d/8]=(dst[d/8]&!mask) | (((src[s/8] >> (7-(s&7)))&1) << (7-(d&7)));
    }
}

/// Emit `impl IWICBitmapSource_Impl` for an object whose `src: RefCell<Option<Source>>` field holds
/// its pixels (filled at construction or `Initialize`). Removes the five-method boilerplate repeated
/// across every WIC source object.
macro_rules! impl_source {
    ($ty:ty) => { impl_source!($ty, _this, Ok::<(),windows_core::Error>(())); };
    ($ty:ty, $this:ident, $check:expr) => {
        impl windows::Win32::Graphics::Imaging::IWICBitmapSource_Impl for $ty {
            fn GetSize(&self, pw: *mut u32, ph: *mut u32) -> windows_core::Result<()> {
                self.src.borrow().as_ref().ok_or($crate::win::e(WINCODEC_ERR_NOTINITIALIZED))?.get_size(pw, ph)
            }
            fn GetPixelFormat(&self) -> windows_core::Result<windows_core::GUID> {
                Ok(self.src.borrow().as_ref().ok_or($crate::win::e(WINCODEC_ERR_NOTINITIALIZED))?.format)
            }
            fn GetResolution(&self, pdpix: *mut f64, pdpiy: *mut f64) -> windows_core::Result<()> {
                let b = self.src.borrow();
                let s = b.as_ref().ok_or($crate::win::e(WINCODEC_ERR_NOTINITIALIZED))?;
                unsafe {
                    *pdpix = s.dpi.0;
                    *pdpiy = s.dpi.1;
                }
                Ok(())
            }
            fn CopyPalette(&self, pal: windows_core::Ref<IWICPalette>) -> windows_core::Result<()> {
                self.src.borrow().as_ref().ok_or($crate::win::e(WINCODEC_ERR_NOTINITIALIZED))?.copy_palette(pal)
            }
            fn CopyPixels(&self, prc: *const WICRect, stride: u32, size: u32, buf: *mut u8) -> windows_core::Result<()> {
                let $this = self;
                $check?;
                let b = self.src.borrow();
                let s = b.as_ref().ok_or($crate::win::e(WINCODEC_ERR_NOTINITIALIZED))?;
                if buf.is_null() { return Err($crate::win::e(E_INVALIDARG)); }
                let slice = unsafe { core::slice::from_raw_parts_mut(buf, size as usize) };
                s.copy_pixels(prc, stride, slice)
            }
        }
    };
}
pub(crate) use impl_source;

/// Build a [`Source`] by reading any `IWICBitmapSource` through the public WIC ABI (used by the
/// transforms' `Initialize` and the encoder's `WriteSource` to consume an arbitrary source).
pub(crate) fn read_source(src: &IWICBitmapSource) -> WResult<Source> {
    let format = unsafe { src.GetPixelFormat()? };
    let (mut w, mut h) = (0u32, 0u32);
    unsafe { src.GetSize(&mut w, &mut h)? };
    let bpp = bpp_of(&format);
    if bpp == 0 {
        return Err(e(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT));
    }
    let stride = ((w * bpp + 7) / 8) as usize;
    let mut pixels = vec![0u8; stride * h as usize];
    unsafe { src.CopyPixels(core::ptr::null(), stride as u32, &mut pixels)? };
    let (mut dx, mut dy) = (0.0f64, 0.0f64);
    let _ = unsafe { src.GetResolution(&mut dx, &mut dy) };
    Ok(Source { format, width: w, height: h, stride, pixels, palette: read_palette(src), dpi: (dx, dy) })
}

/// Read an indexed source's palette through the public ABI (empty if the source has none).
pub(crate) fn read_palette(src: &IWICBitmapSource) -> Vec<u32> {
    let pal: IWICPalette = PaletteObj::default().into();
    if unsafe { src.CopyPalette(&pal) }.is_err() {
        return Vec::new();
    }
    let n = unsafe { pal.GetColorCount() }.unwrap_or(0);
    if n == 0 {
        return Vec::new();
    }
    let mut colors = vec![0u32; n as usize];
    let mut got = 0u32;
    if unsafe { pal.GetColors(&mut colors, &mut got) }.is_ok() {
        colors.truncate(got as usize);
        colors
    } else {
        Vec::new()
    }
}

// --- COM in-proc server -------------------------------------------------------------------------

/// The class factory for our CLSIDs: the imaging factory and the PNG decoder (the only objects
/// created by `CoCreateInstance`; every other object is created by the imaging factory's methods).
#[implement(IClassFactory)]
struct ClassFactory {
    clsid: GUID,
}

impl windows::Win32::System::Com::IClassFactory_Impl for ClassFactory_Impl {
    fn CreateInstance(&self, outer: Ref<IUnknown>, riid: *const GUID, ppv: *mut *mut c_void) -> WResult<()> {
        if outer.as_ref().is_some() {
            return Err(e(CLASS_E_NOAGGREGATION));
        }
        let unk: IUnknown = if self.clsid == CLSID_WICImagingFactory || self.clsid == CLSID_WICImagingFactory2 {
            FactoryObj::new().into()
        } else if components::is_codec_class(&self.clsid) {
            components::create_codec_instance(&self.clsid)?
        } else {
            return Err(e(CLASS_E_CLASSNOTAVAILABLE));
        };
        unsafe { unk.query(riid, ppv).ok() }
    }
    fn LockServer(&self, _flock: windows_core::BOOL) -> WResult<()> {
        Ok(())
    }
}

/// `DllGetClassObject`: hand back a class factory for a CLSID we serve.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn DllGetClassObject(rclsid: *const GUID, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let clsid = unsafe { *rclsid };
    if clsid != CLSID_WICImagingFactory && clsid != CLSID_WICImagingFactory2 && !components::is_codec_class(&clsid) {
        return CLASS_E_CLASSNOTAVAILABLE;
    }
    let cf: IClassFactory = ClassFactory { clsid }.into();
    unsafe { cf.query(riid, ppv) }
}

/// `DllCanUnloadNow`: we never unload (objects outlive calls).
#[unsafe(no_mangle)]
pub unsafe extern "system" fn DllCanUnloadNow() -> HRESULT {
    S_FALSE
}

/// `WICCreateImagingFactory_Proxy`: the SDK's registration-free way to obtain an
/// `IWICImagingFactory` (what `wincodec.h`'s inline factory creation calls). Lets an app load an
/// app-local drop-in and get the factory with no `CoCreateInstance`, registry entry or manifest.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn WICCreateImagingFactory_Proxy(_sdk_version: u32, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    let f: IWICImagingFactory = FactoryObj::new().into();
    unsafe { *out = f.into_raw() };
    S_OK
}
