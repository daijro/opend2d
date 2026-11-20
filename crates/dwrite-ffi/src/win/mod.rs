//! The Windows COM façade: real DirectWrite objects built with the official `windows` IIDs/vtables
//! via `#[implement]`, each method forwarding to `d2d::text`. Only the surface the harness
//! `d2d-ref.exe` exercises for the text scene corpus is wired; every other method returns
//! `E_NOTIMPL`/a default.

#![allow(non_snake_case)]

use core::ffi::c_void;
use windows::Win32::Graphics::DirectWrite::*;
use windows_core::{GUID, HRESULT, IUnknown, IUnknown_Vtbl, Interface, PWSTR, interface};

mod analysis;
mod colorrun;
#[cfg(test)]
mod enum_tests;
mod factory;
pub use factory::default_rendering_params;
mod fontfile;
mod fonts;
pub use fonts::{resolve_face, resolve_face_sims, resolve_face_sims_covering, resolve_face_sims_covering_typographic, resolve_face_sims_in, resolve_face_sims_typographic, wss_to_axis_values};
#[cfg(test)]
mod face_image_tests;
#[cfg(windows)] // GDI interop / bitmap render target — GDI (HDC/HFONT/DIB) is Windows-only.
mod gdi;
mod fontset;
mod misc;
mod paint;
mod text;
#[cfg(test)]
mod text_tests;
#[cfg(test)]
mod var_tests;
mod textanalyzer;

pub(crate) const E_NOTIMPL: HRESULT = HRESULT(0x8000_4001u32 as i32);
pub(crate) const E_INVALIDARG: HRESULT = HRESULT(0x8007_0057u32 as i32);

/// Private interop interface our text objects implement and our `d2d1.dll` QIs: it hands the Rust
/// `d2d::text` inputs (font file bytes, text-format fields) across the DLL boundary so `DrawText`/
/// `DrawGlyphRun` render through the shared `d2d` core. A `windows` custom COM interface, so both
/// DLLs agree on its IID and vtable (same generated code) and QueryInterface matches across them.
#[interface("6d7a2c14-9e2b-4a5f-8c31-7b0d4e2f1a62")]
pub unsafe trait ITextInterop: IUnknown {
    /// Native rasterizer identity: 0 in-process DirectWrite, 1 system FontCache.
    pub unsafe fn FontRasterizer(&self, rasterizer: *mut u32) -> HRESULT;
    /// The resolved font file bytes (`*data`/`*len`, valid while this object lives) and face index.
    pub unsafe fn FontBlob(&self, data: *mut *const u8, len: *mut usize, index: *mut u32) -> HRESULT;
    /// The text-format fields a `DrawText` needs: em size, alignment, word-wrap, base RTL, and the
    /// resolved face's `DWRITE_FONT_SIMULATIONS` (so bold-sim text spaces/embolden-renders like
    /// native) (`d2d::text::layout::TextFormat`). `E_NOTIMPL` on a font face. The IID carries this
    /// vtable; a version-mismatched pair QIs E_NOINTERFACE and falls back to the foreign-format path.
    pub unsafe fn FormatParams(&self, em: *mut f32, align: *mut u32, wrap: *mut u32, base_rtl: *mut i32, sims: *mut u32) -> HRESULT;
}

/// Rasterizer identity for an installed or app-provided font file, shared with foreign-face interop.
#[doc(hidden)]
pub fn font_file_rasterizer(path: &str) -> d2d::text::hinting::Rasterizer {
    fonts::FontSource::Path(path.into()).rasterizer()
}

/// `dwrite.dll!DWriteCreateFactory`: build a factory and hand it back through the requested `iid`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn DWriteCreateFactory(
    _factory_type: DWRITE_FACTORY_TYPE,
    iid: *const GUID,
    factory: *mut *mut c_void,
) -> HRESULT {
    let f: IDWriteFactory7 = factory::FactoryObj::new().into();
    unsafe { f.query(iid, factory) }
}

// --- helpers -----------------------------------------------------------------------------------

/// A null-terminated or length-counted wide string → Rust `String` (the family/locale names the
/// factory receives). `len` is the UTF-16 unit count, or `None` for null-terminated.
pub(crate) fn wide_to_string(p: *const u16, len: Option<usize>) -> String {
    if p.is_null() {
        return String::new();
    }
    let units = match len {
        Some(n) => unsafe { core::slice::from_raw_parts(p, n) }.to_vec(),
        None => {
            let mut n = 0usize;
            while unsafe { *p.add(n) } != 0 {
                n += 1;
            }
            unsafe { core::slice::from_raw_parts(p, n) }.to_vec()
        }
    };
    String::from_utf16_lossy(&units)
}

/// Copy `s` (as UTF-16, null-terminated) into a caller `PWSTR` buffer of `size` units, as the
/// DirectWrite `Get*Name(buffer, size)` methods do. Truncates to fit; always null-terminates.
pub(crate) fn write_wide(s: &str, buf: PWSTR, size: u32) -> windows_core::Result<()> {
    if buf.is_null() || size == 0 {
        return Ok(());
    }
    let units: Vec<u16> = s.encode_utf16().collect();
    let n = units.len().min(size as usize - 1);
    unsafe {
        core::ptr::copy_nonoverlapping(units.as_ptr(), buf.0, n);
        *buf.0.add(n) = 0;
    }
    Ok(())
}

#[cfg(test)]
mod outline_tests;

mod outline;

#[cfg(all(test, target_os = "linux"))]
mod font_discovery_tests;

#[cfg(target_os = "linux")]
mod font_discovery;
