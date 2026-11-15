//! Presenting the CPU render target to real windows and DXGI surfaces (Windows only). The core
//! renders into its own premultiplied-BGRA8 [`PixelBuffer`]; at `EndDraw` that buffer is copied to
//! the bound target's present destination — GDI `StretchDIBits` for HWND/DC render targets,
//! `UpdateSubresource` for a DXGI surface (a swap-chain back buffer or any `IDXGISurface`). The
//! render-target COM objects are plain [`ContextObj`]s (so they QI to `ID2D1DeviceContext`, as a
//! native 1.1 render target does) carrying a present destination on their bound bitmap.

use super::context::ContextObj;
use super::resources::BitmapObj;
use super::rt_dpi;
use d2d::bitmap::Bitmap;
use d2d::contracts::PixelBuffer;
use std::cell::RefCell;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct2D::Common::D2D_SIZE_U;
use windows::Win32::Graphics::Direct2D::{D2D1_BITMAP_OPTIONS_CANNOT_DRAW, D2D1_BITMAP_OPTIONS_TARGET, D2D1_HWND_RENDER_TARGET_PROPERTIES, D2D1_RENDER_TARGET_PROPERTIES};
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Resource, ID3D11Texture2D, D3D11_CPU_ACCESS_READ, D3D11_MAPPED_SUBRESOURCE, D3D11_MAP_READ,
    D3D11_TEXTURE2D_DESC, D3D11_USAGE_STAGING,
};
use windows::Win32::Graphics::Dxgi::IDXGISurface;
use windows::Win32::Graphics::Gdi::{GetDC, ReleaseDC, StretchDIBits, BITMAPINFO, BITMAPINFOHEADER, BI_RGB, DIB_RGB_COLORS, HDC, SRCCOPY};
use windows_core::{ComObject, Interface, Ref, Result as WResult};

/// `D2DERR_RECREATE_TARGET`: the target's device was lost; the caller must rebuild it.
pub(crate) const D2DERR_RECREATE_TARGET: windows_core::HRESULT = windows_core::HRESULT(0x8899_000Cu32 as i32);

/// Where a target's pixels go at `EndDraw`.
pub(crate) enum Present {
    /// GDI blit to a window (`GetDC(hwnd)` each present) or a bound DC, at a device-pixel origin.
    Gdi { hwnd: Option<HWND>, hdc: Option<HDC>, origin: (i32, i32) },
    /// `UpdateSubresource` into a D3D11 texture (a DXGI surface / swap-chain back buffer).
    Dxgi { ctx: windows::Win32::Graphics::Direct3D11::ID3D11DeviceContext, tex: ID3D11Resource },
}

impl Present {
    /// Copy the finished `buf` (premultiplied BGRA8, top-down, tightly packed) to the destination.
    pub(crate) fn flush(&self, buf: &PixelBuffer) -> WResult<()> {
        let (w, h) = (buf.desc.width, buf.desc.height);
        match self {
            Present::Gdi { hwnd, hdc, origin } => {
                let bmi = bitmapinfo(w, h);
                let dc = match hdc {
                    Some(h) => *h,
                    None => unsafe { GetDC(*hwnd) },
                };
                let rows = unsafe {
                    StretchDIBits(dc, origin.0, origin.1, w as i32, h as i32, 0, 0, w as i32, h as i32,
                        Some(buf.data.as_ptr().cast()), &bmi, DIB_RGB_COLORS, SRCCOPY)
                };
                if hdc.is_none() {
                    unsafe { ReleaseDC(*hwnd, dc) };
                }
                // Native's HWND/DC render target EndDraw returns S_OK even when the window is gone or
                // the blit is a no-op (verified on the VM: EndDraw with a destroyed HWND is S_OK, not
                // D2DERR_RECREATE_TARGET — RECREATE is reserved for D3D device loss, which a GDI present
                // has none of). So a 0-row StretchDIBits is still success.
                let _ = rows;
                Ok(())
            }
            Present::Dxgi { ctx, tex } => {
                unsafe {
                    ctx.UpdateSubresource(tex, 0, None, buf.data.as_ptr().cast(), w * 4, 0);
                    // Device loss surfaces as D2DERR_RECREATE_TARGET, as native does.
                    ctx.GetDevice()?.GetDeviceRemovedReason().map_err(|_| windows_core::Error::from(D2DERR_RECREATE_TARGET))
                }
            }
        }
    }
}

/// Top-down 32bpp BGRA `BITMAPINFO` for `StretchDIBits`/`CreateDIBSection` (BI_RGB ignores alpha).
pub(crate) fn bitmapinfo(w: u32, h: u32) -> BITMAPINFO {
    BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: core::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: w as i32,
            biHeight: -(h as i32),
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB.0,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Inspect a DXGI surface as a D3D11 texture: `(width, height, present-to-texture destination)`.
fn dxgi_dest(surface: &IDXGISurface) -> WResult<(u32, u32, Present)> {
    let tex: ID3D11Texture2D = surface.cast()?;
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    unsafe { tex.GetDesc(&mut desc) };
    let ctx = unsafe { tex.GetDevice()?.GetImmediateContext()? };
    Ok((desc.Width, desc.Height, Present::Dxgi { ctx, tex: tex.cast()? }))
}

/// Read a DXGI surface's current pixels back into a core premultiplied-BGRA8 [`Bitmap`] — the read
/// half of the DXGI bridge (the rest of this file is the write half). Native keeps a surface's pixels
/// on the GPU; the CPU core needs them in memory to use the surface as a drawable source (a bitmap or
/// image source), so copy the surface's texture to a CPU-readable staging texture and map it. This is
/// the same staging copy the harness's own `read_texture` does.
pub(crate) fn read_dxgi(surface: &IDXGISurface) -> WResult<Bitmap> {
    let tex: ID3D11Texture2D = surface.cast()?;
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    unsafe { tex.GetDesc(&mut desc) };
    let dev = unsafe { tex.GetDevice()? };
    let ctx = unsafe { dev.GetImmediateContext()? };
    let staging_desc = D3D11_TEXTURE2D_DESC {
        BindFlags: 0,
        MiscFlags: 0,
        Usage: D3D11_USAGE_STAGING,
        CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
        ..desc
    };
    let mut staging = None;
    unsafe { dev.CreateTexture2D(&staging_desc, None, Some(&mut staging))? };
    let staging = staging.ok_or_else(|| windows_core::Error::from(super::E_NOTIMPL))?;
    let (w, h) = (desc.Width, desc.Height);
    let mut premul = vec![0u8; (w * h * 4) as usize];
    unsafe {
        ctx.CopyResource(&staging, &tex);
        let mut m = D3D11_MAPPED_SUBRESOURCE::default();
        ctx.Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut m))?;
        for row in 0..h as usize {
            let s = (m.pData as *const u8).add(row * m.RowPitch as usize);
            let d = (row * w as usize) * 4;
            premul[d..d + w as usize * 4].copy_from_slice(core::slice::from_raw_parts(s, w as usize * 4));
        }
        ctx.Unmap(&staging, 0);
    }
    Ok(Bitmap {
        width: w,
        height: h,
        premul_bgra: premul,
        format: d2d::abi::DXGI_FORMAT_B8G8R8A8_UNORM,
        alpha: d2d::abi::D2D1_ALPHA_MODE_PREMULTIPLIED,
        dpi: (96.0, 96.0),
    })
}

/// Build a render-target `ContextObj` with an internal pixel buffer of `w`×`h` and a present
/// destination (`None` ⇒ a DC target before `BindDC`).
fn render_target(factory: &d2d::Factory, factory_iface: windows::Win32::Graphics::Direct2D::ID2D1Factory, w: u32, h: u32, dpi: f32, present: Option<Present>) -> ComObject<ContextObj> {
    let mut dc = factory.create_device().create_device_context();
    dc.set_dpi(dpi, dpi);
    let buf = d2d::bitmap_target(w, h, dpi);
    let bm = ComObject::new(BitmapObj {
        pixels: D2D_SIZE_U { width: w, height: h },
        dpi: (dpi, dpi),
        buf: RefCell::new(None),
        src: RefCell::new(None),
        options: D2D1_BITMAP_OPTIONS_TARGET,
        present: RefCell::new(present),
        factory: Some(factory_iface.clone()),
        domain_ids: None,
        color_context: None,
    });
    ContextObj::with_target(dc, bm, buf, factory_iface)
}

/// `DXGI_ERROR_NOT_CURRENTLY_AVAILABLE` — native's result when a desktop (hwnd) render target has no
/// display-capable adapter to present to.
const DXGI_ERROR_NOT_CURRENTLY_AVAILABLE: windows_core::HRESULT = windows_core::HRESULT(0x887A_0022u32 as i32);

/// Does any DXGI adapter drive at least one display output?
///
/// Native's hwnd (desktop) render target acquires its D3D device through the *display* adapter class
/// (`Displays::CreateDevice` → `D3D11CreateDevice`). On a session with no display output — a headless
/// server, Session 0, the Basic Render Driver on a display-less VM — that device acquisition fails
/// with `DXGI_ERROR_NOT_CURRENTLY_AVAILABLE`, and the hardware→software fallback can't recover it
/// (VM-verified: native `CreateHwndRenderTarget` returns 0x887A0022 there). Our render target blits
/// through GDI and needs no device, so it would otherwise succeed where native fails. Probing for a
/// live output reproduces native's gate: a real desktop (the only place a browser fingerprints) has
/// an output and both succeed; a display-less host has none and both fail with the same HRESULT.
fn has_display_output() -> bool {
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};
    unsafe {
        let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory1>() else { return false };
        let mut i = 0u32;
        while let Ok(adapter) = factory.EnumAdapters1(i) {
            // Any adapter with EnumOutputs(0) == Ok drives a display (EnumOutputs is inherited from
            // IDXGIAdapter).
            if adapter.EnumOutputs(0).is_ok() {
                return true;
            }
            i += 1;
        }
        false
    }
}

/// `ID2D1Factory::CreateHwndRenderTarget`.
pub(crate) fn hwnd_render_target(factory: &d2d::Factory, factory_iface: windows::Win32::Graphics::Direct2D::ID2D1Factory, rtprops: *const D2D1_RENDER_TARGET_PROPERTIES, hp: *const D2D1_HWND_RENDER_TARGET_PROPERTIES) -> WResult<windows::Win32::Graphics::Direct2D::ID2D1HwndRenderTarget> {
    // A desktop (DEFAULT/HARDWARE) target needs a display-capable device; native fails with
    // DXGI_ERROR_NOT_CURRENTLY_AVAILABLE when none exists. A SOFTWARE target uses WARP and is exempt.
    let rt_type = unsafe { rtprops.as_ref() }.map(|p| p.r#type.0).unwrap_or(0);
    let is_software = rt_type == windows::Win32::Graphics::Direct2D::D2D1_RENDER_TARGET_TYPE_SOFTWARE.0;
    if !is_software && !has_display_output() {
        return Err(windows_core::Error::from(DXGI_ERROR_NOT_CURRENTLY_AVAILABLE));
    }
    let hp = unsafe { &*hp };
    let present = Present::Gdi { hwnd: Some(hp.hwnd), hdc: None, origin: (0, 0) };
    render_target(factory, factory_iface, hp.pixelSize.width, hp.pixelSize.height, rt_dpi(rtprops), Some(present)).cast()
}

/// `ID2D1Factory::CreateDCRenderTarget` — unbound until `BindDC` supplies the DC and rect.
pub(crate) fn dc_render_target(factory: &d2d::Factory, factory_iface: windows::Win32::Graphics::Direct2D::ID2D1Factory, rtprops: *const D2D1_RENDER_TARGET_PROPERTIES) -> WResult<windows::Win32::Graphics::Direct2D::ID2D1DCRenderTarget> {
    // A DC render target must be the GDI-compatible format DXGI_FORMAT_B8G8R8A8_UNORM; native rejects
    // any other format — including the default DXGI_FORMAT_UNKNOWN — with D2DERR_UNSUPPORTED_PIXEL_FORMAT
    // (VM-verified). Our target otherwise accepted UNKNOWN and silently rendered.
    let fmt = unsafe { rtprops.as_ref() }.map(|p| p.pixelFormat.format).unwrap_or(windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(0));
    if fmt != windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM {
        return Err(windows_core::Error::from(windows_core::HRESULT(0x8898_2F80u32 as i32)));
    }
    render_target(factory, factory_iface, 0, 0, rt_dpi(rtprops), None).cast()
}

/// `ID2D1Factory::CreateDxgiSurfaceRenderTarget`.
pub(crate) fn dxgi_render_target(factory: &d2d::Factory, factory_iface: windows::Win32::Graphics::Direct2D::ID2D1Factory, surface: Ref<IDXGISurface>, rtprops: *const D2D1_RENDER_TARGET_PROPERTIES) -> WResult<windows::Win32::Graphics::Direct2D::ID2D1RenderTarget> {
    let s = surface.as_ref().ok_or(windows_core::Error::from(super::E_NOTIMPL))?;
    let (w, h, present) = dxgi_dest(s)?;
    render_target(factory, factory_iface, w, h, rt_dpi(rtprops), Some(present)).cast()
}

/// `ID2D1DeviceContext::CreateBitmapFromDxgiSurface`: a drawable bitmap backed by the DXGI surface.
/// It is a render target — its pixels are flushed to the surface when it is the target at `EndDraw`
/// — and, unless `CANNOT_DRAW` was requested, also a readable source: `readable` reads the surface's
/// current pixels into `src` so the bitmap can be drawn from (and `GetSurface` returns the surface).
pub(crate) fn bitmap_from_dxgi(surface: Ref<IDXGISurface>, dpi: (f32, f32), readable: bool, factory: windows::Win32::Graphics::Direct2D::ID2D1Factory) -> WResult<BitmapObj> {
    let s = surface.as_ref().ok_or(windows_core::Error::from(super::E_NOTIMPL))?;
    let src = if readable { Some(read_dxgi(s)?) } else { None };
    let (w, h, present) = dxgi_dest(s)?;
    // A DXGI-surface bitmap is a render target; CANNOT_DRAW (not readable) keeps it write-only.
    let options = if readable { D2D1_BITMAP_OPTIONS_TARGET } else { D2D1_BITMAP_OPTIONS_TARGET | D2D1_BITMAP_OPTIONS_CANNOT_DRAW };
    Ok(BitmapObj {
        pixels: D2D_SIZE_U { width: w, height: h },
        dpi,
        buf: RefCell::new(Some(d2d::bitmap_target(w, h, dpi.0))),
        src: RefCell::new(src),
        options,
        present: RefCell::new(Some(present)),
        factory: Some(factory),
        domain_ids: None,
        color_context: None,
    })
}
