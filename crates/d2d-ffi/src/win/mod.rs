//! The Windows COM façade: real Direct2D objects built with the official `windows` IIDs/vtables via
//! `#[implement]`, each method forwarding to the `d2d` Rust core. Only the surface the harness
//! `d2d-ref.exe` exercises for the conformance scene corpus is wired; every other method returns
//! `E_NOTIMPL` (see [`NOT_IMPL`]).

#![allow(non_snake_case)]

use core::ffi::c_void;
use windows::Win32::Graphics::Direct2D::Common::{D2D1_COLOR_F, D2D_RECT_F};
use windows::Win32::Graphics::Direct2D::{D2D1_FACTORY_OPTIONS, D2D1_FACTORY_TYPE};
use windows_core::{GUID, HRESULT, Interface};
use windows_numerics::{Matrix3x2, Vector2};

mod author;
mod context;
mod factory;
mod geometry;
mod image;
mod metafile;
#[cfg(windows)] // HWND / DXGI-surface / D3D11 / GDI-DC render targets — Windows presentation only.
mod present;
#[cfg(windows)] // XPS-OM print package (ID2D1PrintControl) — Windows only.
mod print;
mod resources;
mod svg;

pub(crate) const E_NOTIMPL: HRESULT = HRESULT(0x8000_4001u32 as i32);
pub(crate) const E_INVALIDARG: HRESULT = HRESULT(0x8007_0057u32 as i32);

/// Effective DPI from `D2D1_RENDER_TARGET_PROPERTIES` (null or 0 ⇒ 96, as native resolves it).
/// Used by the Windows presenting render targets (`win::present`) and the portable WIC render target.
pub(crate) fn rt_dpi(p: *const windows::Win32::Graphics::Direct2D::D2D1_RENDER_TARGET_PROPERTIES) -> f32 {
    if p.is_null() { return 96.0; }
    let d = unsafe { &*p }.dpiX;
    if d == 0.0 { 96.0 } else { d }
}

/// Methods present in the vtables (so QueryInterface identity and layout are exact) but not yet
/// forwarded to the core — they return `E_NOTIMPL`/no-op. The honest gap list for phase 3A: the
/// drawing/geometry/brush/bitmap/effect/layer/SVG/image-source/sprite-batch/command-list/gradient-
/// mesh/opacity-mask surface the conformance corpus exercises is wired (`ID2D1DeviceContext6` and its
/// whole base chain, `ID2D1Effect`, `ID2D1CommandList`, `ID2D1GradientMesh`, `ID2D1ImageSourceFromWic`,
/// `ID2D1TransformedImageSource`, `ID2D1SvgDocument`, `ID2D1SpriteBatch`); the rest is stubbed because
/// the core has no metadata table (built-in effect property names/types/counts) or no passing scene
/// reaches it (colour-glyph / SVG-glyph runs, some legacy render-target creators, partial Widen).
pub static NOT_IMPL: &[&str] = &[
    // Widened-region stream is partially wired (byte-exact for rects + flat-cap/miter-bevel strokes);
    // round joins/caps (beziers) and square caps (separate figure) still differ — a richer widener
    // contour is 1C core work. (GetGlyphRunWorldBounds, GetEffect{Invalid,RequiredInput}Rectangles and
    // GetImage{Local,World}Bounds are now wired — see text::bounds / effects::rects.)
    "ID2D1Geometry::Widen: round joins/caps (beziers) and square caps (separate figure) still differ \
     from native's widened-region stream — a richer widener contour is 1C core work",
    // Implemented (COM gaps part 3) with documented ceilings, not E_NOTIMPL: readable DXGI surfaces
    // (ID2D1Bitmap1::GetSurface, CreateBitmapFromDxgiSurface read-back, CreateImageSourceFromDxgi —
    // single BGRA plane; multi-plane YUV is not); ID2D1GdiMetafile (CreateGdiMetafile/GetBounds/GetDpi/
    // GetSourceBounds/Stream are the deterministic EMF-header + record surface, DrawGdiMetafile plays a
    // minimal primitive record set — native delegates the whole metafile to the external
    // CreateMetafileRenderer apiset: when it is absent CreateGdiMetafile returns D2DERR_UNSUPPORTED_OPERATION,
    // and when present even GetBounds/GetSourceBounds forward to that renderer (which interprets records,
    // not the header — VM: native bounds width 40.005 DIP vs the rclFrame's 39.005), so neither the
    // playback pixels nor the bounds are reproducible here, see d2d::metafile); ID2D1PrintControl
    // (XPS package via the OS XPS OM, native's whole-page-raster form — vector serialisation of arbitrary
    // command-list primitives is the ceiling, see print.rs).
    // Text/glyph color-run + SVG-glyph surface (no core counterpart):
    "ID2D1DeviceContext4::{CreateSvgGlyphStyle,GetColorBitmapGlyphImage,GetSvgGlyphImage,DrawColorBitmapGlyphRun}",
    // Still stubbed readback / small forwards (no core table or no passing scene):
    "ID2D1Properties::{GetPropertyName,GetPropertyCount,GetType,GetSubProperties} — built-in effects' property \
     metadata (names/types/counts) live in Direct2D's registration XML, not modelled in the core; GetValue/\
     GetValueSize do read back the stored value. Concrete geometry shape getters; old \
     ID2D1RenderTarget::{CreateSharedBitmap,CreateLayer,CreateMesh,\
     CreateBitmap(base),CreateBitmapBrush(base)}; versioned Factory1 CreateDrawingStateBlock1; \
     GetRegisteredEffects/GetEffectProperties; CreateWicBitmapRenderTarget",
];

// --- value conversions (windows value structs ⇄ the core's plain `[f32; N]`) -------------------

pub(crate) fn mat(m: &Matrix3x2) -> [f32; 6] {
    [m.M11, m.M12, m.M21, m.M22, m.M31, m.M32]
}
pub(crate) fn mat_opt(m: *const Matrix3x2) -> [f32; 6] {
    if m.is_null() { [1.0, 0.0, 0.0, 1.0, 0.0, 0.0] } else { mat(unsafe { &*m }) }
}
pub(crate) fn col(c: &D2D1_COLOR_F) -> [f32; 4] {
    [c.r, c.g, c.b, c.a]
}
pub(crate) fn rectf(r: &D2D_RECT_F) -> [f32; 4] {
    [r.left, r.top, r.right, r.bottom]
}
pub(crate) fn vec2(v: &Vector2) -> [f32; 2] {
    [v.X, v.Y]
}

// --- exports -----------------------------------------------------------------------------------

/// `D2D1CreateFactory`: build a factory and hand it back through the requested `riid`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn D2D1CreateFactory(
    factory_type: D2D1_FACTORY_TYPE,
    riid: *const GUID,
    _options: *const D2D1_FACTORY_OPTIONS,
    out: *mut *mut c_void,
) -> HRESULT {
    let f: windows::Win32::Graphics::Direct2D::ID2D1Factory1 =
        factory::FactoryObj::new(factory_type.0 == 1).into();
    unsafe { f.query(riid, out) }
}

/// `D2D1CreateDevice`: create a device without a factory handle (the DXGI device is
/// ignored — the core renders on the CPU).
#[unsafe(no_mangle)]
pub unsafe extern "system" fn D2D1CreateDevice(
    _dxgi: *mut c_void,
    _props: *const c_void,
    out: *mut *mut c_void,
) -> HRESULT {
    // No user factory on this path; mint one so the device's GetFactory is a valid factory (native's
    // D2D1CreateDevice device likewise reports an internal factory).
    let f1: windows::Win32::Graphics::Direct2D::ID2D1Factory1 = factory::FactoryObj::new(false).into();
    let fac: windows::Win32::Graphics::Direct2D::ID2D1Factory = f1.cast().unwrap();
    let dev: windows::Win32::Graphics::Direct2D::ID2D1Device =
        factory::DeviceObj::new(d2d::Factory::new(false).create_device(), fac).into();
    unsafe { *out = dev.into_raw() };
    HRESULT(0)
}

/// `D2D1CreateDeviceContext` is not wired (it needs a DXGI surface target).
#[unsafe(no_mangle)]
pub unsafe extern "system" fn D2D1CreateDeviceContext(
    _surface: *mut c_void,
    _props: *const c_void,
    _out: *mut *mut c_void,
) -> HRESULT {
    E_NOTIMPL
}
