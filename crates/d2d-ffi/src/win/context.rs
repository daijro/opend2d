//! The device context: `ID2D1DeviceContext` (over `ID2D1RenderTarget`/`ID2D1Resource`). Drawing,
//! state and resource-creation methods forward to one `d2d::DeviceContext`; target binding moves the
//! `PixelBuffer` between the bound `BitmapObj` and the core so pixels survive `SetTarget(nullptr)`
//! for the staging read-back. Everything else is `E_NOTIMPL`/no-op (see [`super::NOT_IMPL`]).

use super::geometry::{GeometryObj, MeshObj};
use super::image::{bitmap_from_wic, image_target, CmFailObj, ColorContextObj, CommandListObj, DrawTarget, EffectObj, GradientMeshObj, ImageSrcObj, LutObj, SpriteBatchObj};
use super::metafile::MetafileObj;
#[cfg(windows)] // HWND/DC/GDI render targets are Windows-only (see win/present, win/mod).
use super::present::Present;
use super::svg::SvgDocObj;
use d2d::color_context::{ColorContext, SimpleProfile};
use super::resources::{BitmapObj, BrushParams, InkObj, InkStyleObj, PixelBrushObj, RealizationObj, SolidBrushObj, StateBlockObj, StopsObj, StrokeStyleObj};
use super::{E_NOTIMPL, col, mat, rectf, vec2};
use d2d::bitmap::Bitmap;
#[cfg(windows)] // only the Windows-only `with_target` (presenting render targets) moves a `PixelBuffer`.
use d2d::contracts::PixelBuffer;
#[cfg(windows)] // only the GDI-interop GetDC below uses the bare `c_void` (via CreateDIBSection).
use core::ffi::c_void;
#[cfg(windows)] // only the GDI-interop / HWND / DC render targets below use these.
use windows::Win32::Foundation::{HWND, RECT};
#[cfg(windows)]
use windows::Win32::Graphics::Gdi::{CreateCompatibleDC, CreateDIBSection, DeleteDC, DeleteObject, SelectObject, DIB_RGB_COLORS, HBITMAP, HDC, HGDIOBJ};
use d2d::extras::mesh::MeshPatch;
use d2d::geometry::{Figure, FillMode, Geometry, Segment, StrokeStyle};
use d2d::text::layout::TextFormat;
use d2d::text::{Glyph, GlyphRun, RenderingParams};
use dwrite::ITextInterop;
use std::cell::RefCell;
use windows::Win32::Graphics::Direct2D::Common::*;
use windows::Win32::Graphics::Direct2D::*;
use windows::Win32::Graphics::DirectWrite::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM};
use windows::Win32::System::Com::IStream;
use windows_core::{BOOL, ComObject, IUnknownImpl, Interface, Ref, Result as WResult, implement};
use windows_numerics::{Matrix3x2, Vector2};

struct Ctx {
    dc: d2d::DeviceContext,
    /// The bitmap currently bound as target, holding its storage while unbound.
    target: Option<ComObject<BitmapObj>>,
    /// Text rendering params set by `SetTextRenderingParams` (read from the `IDWriteRenderingParams`),
    /// passed to the core's `DrawText`/`DrawGlyphRun`.
    text_params: RenderingParams,
    /// The `IDWriteRenderingParams` object last set, so `GetTextRenderingParams` hands back exactly
    /// what was set (native returns NULL only when none was ever set).
    text_params_obj: Option<IDWriteRenderingParams>,
    /// Tags of a ColorManagement draw native fails to realise (SIMPLE profile); `EndDraw` reports
    /// E_INVALIDARG latched at these tags, as native does (cms2-04/05).
    cm_fail: Option<(u64, u64)>,
    /// `CreateWicBitmapRenderTarget` destination: the rendered pixels are copied into this IWICBitmap
    /// at `EndDraw`. Portable (IWICBitmap is a WIC COM object, not an OS surface).
    wic: Option<windows::Win32::Graphics::Imaging::IWICBitmap>,
}

// The one object is both the device context AND a render target: a factory-created Hwnd/DC/DXGI
// render target is a `ContextObj` carrying a present destination on its bound bitmap, and (as on a
// native 1.1 render target) QIs to `ID2D1DeviceContext`.
// The HWND/DC/GDI-interop render-target interfaces exist only on Windows (no HWND/GDI elsewhere);
// off Windows the object is just the device context (over its render-target base chain).
#[cfg_attr(windows, implement(ID2D1DeviceContext7, ID2D1HwndRenderTarget, ID2D1DCRenderTarget, ID2D1GdiInteropRenderTarget, ID2D1BitmapRenderTarget))]
#[cfg_attr(not(windows), implement(ID2D1DeviceContext7, ID2D1BitmapRenderTarget))]
pub(crate) struct ContextObj {
    inner: RefCell<Ctx>,
    /// The factory this context (and every resource it creates) belongs to, for `GetFactory`.
    factory: ID2D1Factory,
    /// The device that created this context (`GetDevice`); `None` for the 1.0 render targets, which
    /// native does not expose a device on. Apps use `GetDevice` to spin up sibling contexts.
    device: Option<ID2D1Device>,
    /// `SetRenderingControls` state, round-tripped by the getter. Native's default is buffer precision
    /// UNKNOWN (effects pick per-node) and a 1024×1024 tile.
    rendering_controls: std::cell::Cell<D2D1_RENDERING_CONTROLS>,
    /// True while a `DrawTextLayout` is driving `IDWriteTextLayout::Draw` through `LayoutRenderer`. Its
    /// glyph runs re-enter `DrawGlyphRun`, but native renders a *layout* at any device em (only a direct
    /// `DrawText`/`DrawGlyphRun` rejects ≥ 2^15 under a GDI measuring mode), so the em-reject guard is
    /// skipped while this is set.
    in_layout_draw: std::cell::Cell<bool>,
    /// Scratch memory DC + DIB section live between `ID2D1GdiInteropRenderTarget::GetDC` and `ReleaseDC`.
    #[cfg(windows)]
    gdi: RefCell<Option<GdiScratch>>,
}

/// GDI-interop scratch: the DIB-backed memory DC handed out by `GetDC`, torn down by `ReleaseDC`.
#[cfg(windows)]
struct GdiScratch {
    dc: HDC,
    bmp: HBITMAP,
    old: HGDIOBJ,
    bits: *mut u8,
}

/// The internal `IDWriteTextRenderer` that `DrawTextLayout` drives `IDWriteTextLayout::Draw` with,
/// exactly as native's D2D text renderer does: each glyph run is drawn with its per-range
/// `SetDrawingEffect` brush (an `ID2D1Brush`), else the default; underlines/strikethroughs become
/// filled rects; inline objects draw themselves through this same renderer. Re-enters the owning
/// render target's `DrawGlyphRun`/`FillRectangle`.
#[implement(IDWriteTextRenderer)]
struct LayoutRenderer {
    rt: ID2D1RenderTarget,
    brush: ID2D1Brush,
    transform: DWRITE_MATRIX,
    // The device transform (world ∘ DPI), for baseline pixel-snapping in device space — the same dt
    // the core layout path snaps with. `transform` above stays the world matrix dwrite's `Draw` reads.
    dev_dt: [f64; 6],
    ppd: f32,
    /// D2D1_DRAW_TEXT_OPTIONS_NO_SNAP: the baselines stay sub-pixel.
    no_snap: bool,
}

impl LayoutRenderer_Impl {
    /// The per-call drawing effect as an `ID2D1Brush` (native's convention), else the default brush.
    fn eff_brush(&self, effect: Ref<windows_core::IUnknown>) -> ID2D1Brush {
        effect.as_ref().and_then(|e| e.cast::<ID2D1Brush>().ok()).unwrap_or_else(|| self.brush.clone())
    }

    /// An underline / strikethrough rect `thickness` tall from `top`. D2D's text renderer fills it under
    /// its own antialias mode: ALIASED, and at least one device pixel thick, when the text is aliased
    /// (text antialias mode or rendering params) or measured in a GDI mode; per-primitive otherwise. The
    /// target's mode comes back after.
    fn fill_decoration(&self, x: f32, top: f32, width: f32, thickness: f32, measuring: DWRITE_MEASURING_MODE, brush: &ID2D1Brush) {
        let rt = &self.rt;
        let params_aliased = unsafe { rt.GetTextRenderingParams() }.is_ok_and(|p| unsafe { p.GetRenderingMode() } == DWRITE_RENDERING_MODE_ALIASED);
        let aliased = unsafe { rt.GetTextAntialiasMode() } == D2D1_TEXT_ANTIALIAS_MODE_ALIASED || params_aliased || measuring != DWRITE_MEASURING_MODE_NATURAL;
        // One device pixel in DIPs: |row 1| / |det| of the world transform scaled by dpi/96 (96 under
        // the pixel unit mode).
        let pixels = rt.cast::<ID2D1DeviceContext>().is_ok_and(|dc| unsafe { dc.GetUnitMode() } == D2D1_UNIT_MODE_PIXELS);
        let (mut dx, mut dy) = (96.0f32, 96.0f32);
        if !pixels {
            unsafe { rt.GetDpi(&mut dx, &mut dy) };
        }
        let (sx, sy) = (dx / 96.0, dy / 96.0);
        let mut m = Matrix3x2::default();
        unsafe { rt.GetTransform(&mut m) };
        let (a, b) = (m.M11 * sx, m.M12 * sy);
        let device_px = (b * b + a * a).sqrt() / ((m.M22 * sy) * a - (m.M21 * sx) * b).abs();
        let thickness = if aliased { device_px.max(thickness) } else { thickness };
        unsafe {
            let saved = rt.GetAntialiasMode();
            rt.SetAntialiasMode(if aliased { D2D1_ANTIALIAS_MODE_ALIASED } else { D2D1_ANTIALIAS_MODE_PER_PRIMITIVE });
            rt.FillRectangle(&D2D_RECT_F { left: x, top, right: x + width, bottom: top + thickness }, brush);
            rt.SetAntialiasMode(saved);
        }
    }
}

impl IDWritePixelSnapping_Impl for LayoutRenderer_Impl {
    fn IsPixelSnappingDisabled(&self, _c: *const core::ffi::c_void) -> WResult<BOOL> {
        Ok(self.no_snap.into())
    }
    fn GetCurrentTransform(&self, _c: *const core::ffi::c_void, transform: *mut DWRITE_MATRIX) -> WResult<()> {
        unsafe { *transform = self.transform };
        Ok(())
    }
    fn GetPixelsPerDip(&self, _c: *const core::ffi::c_void) -> WResult<f32> {
        Ok(self.ppd)
    }
}

impl IDWriteTextRenderer_Impl for LayoutRenderer_Impl {
    fn DrawGlyphRun(&self, _c: *const core::ffi::c_void, x: f32, y: f32, measuring: DWRITE_MEASURING_MODE, run: *const DWRITE_GLYPH_RUN, _desc: *const DWRITE_GLYPH_RUN_DESCRIPTION, effect: Ref<windows_core::IUnknown>) -> WResult<()> {
        let brush = self.eff_brush(effect);
        // Baseline pixel-snapping is the layout's job (unless NO_SNAP disables it): snap here,
        // before re-entering DrawGlyphRun, which itself draws at the exact origin. As the core layout path,
        // a horizontal run snaps on the device grid of the scaled linear part plus the world translation
        // unscaled (DirectWrite's pixel snapping), so DrawTextLayout matches core/native.
        let sideways = !run.is_null() && unsafe { (*run).isSideways.as_bool() };
        let d = &self.dev_dt;
        let t = if sideways { *d } else { [d[0], d[1], d[2], d[3], self.transform.dx as f64, self.transform.dy as f64] };
        let o = d2d::text::snap_baseline([x, y], &t, self.no_snap);
        unsafe { self.rt.DrawGlyphRun(Vector2 { X: o[0], Y: o[1] }, run, &brush, measuring) };
        Ok(())
    }
    fn DrawUnderline(&self, _c: *const core::ffi::c_void, x: f32, y: f32, u: *const DWRITE_UNDERLINE, effect: Ref<windows_core::IUnknown>) -> WResult<()> {
        let u = unsafe { &*u };
        let brush = self.eff_brush(effect);
        // The underline is a rect below the baseline: offset (+down) from the baseline, `thickness` tall.
        self.fill_decoration(x, y + u.offset, u.width, u.thickness, u.measuringMode, &brush);
        Ok(())
    }
    fn DrawStrikethrough(&self, _c: *const core::ffi::c_void, x: f32, y: f32, s: *const DWRITE_STRIKETHROUGH, effect: Ref<windows_core::IUnknown>) -> WResult<()> {
        let s = unsafe { &*s };
        let brush = self.eff_brush(effect);
        self.fill_decoration(x, y + s.offset, s.width, s.thickness, s.measuringMode, &brush);
        Ok(())
    }
    fn DrawInlineObject(&self, c: *const core::ffi::c_void, x: f32, y: f32, o: Ref<IDWriteInlineObject>, sideways: BOOL, rtl: BOOL, effect: Ref<windows_core::IUnknown>) -> WResult<()> {
        if let Some(o) = o.as_ref() {
            let r: IDWriteTextRenderer = self.to_interface();
            unsafe { o.Draw(Some(c), &r, x, y, sideways.as_bool(), rtl.as_bool(), effect.as_ref())? };
        }
        Ok(())
    }
}

impl ContextObj {
    pub(crate) fn new(dc: d2d::DeviceContext, factory: ID2D1Factory, device: Option<ID2D1Device>) -> Self {
        Self {
            inner: RefCell::new(Ctx { dc, target: None, text_params: system_text_params(), text_params_obj: None, cm_fail: None, wic: None }),
            factory,
            device,
            rendering_controls: std::cell::Cell::new(D2D1_RENDERING_CONTROLS {
                bufferPrecision: D2D1_BUFFER_PRECISION_UNKNOWN,
                tileSize: D2D_SIZE_U { width: 1024, height: 1024 },
            }),
            in_layout_draw: std::cell::Cell::new(false),
            #[cfg(windows)]
            gdi: RefCell::new(None),
        }
    }

    /// A counted `ID2D1Factory` reference to this context's factory, for the resources it creates.
    pub(crate) fn factory_iface(&self) -> ID2D1Factory {
        self.factory.clone()
    }

    /// `ID2D1Factory::CreateWicBitmapRenderTarget`: a render target whose pixels land in `wic` at
    /// `EndDraw`. Portable — unlike the Hwnd/DC/DXGI targets it needs no OS surface, just WIC memory.
    pub(crate) fn with_wic_target(factory: &d2d::Factory, factory_iface: ID2D1Factory, wic: windows::Win32::Graphics::Imaging::IWICBitmap, w: u32, h: u32, dpi: f32) -> ComObject<Self> {
        let mut dc = factory.create_device().create_device_context();
        dc.set_dpi(dpi, dpi);
        dc.set_target_bitmap(d2d::bitmap_target(w, h, dpi));
        let obj = ComObject::new(Self::new(dc, factory_iface, None));
        obj.inner.borrow_mut().wic = Some(wic);
        obj
    }

    /// `ID2D1RenderTarget::CreateCompatibleRenderTarget`: an offscreen bitmap render target in the
    /// same resource domain as its parent (`dc` is a [`DeviceContext::sibling`]). It presents nowhere
    /// (`present: None`); `EndDraw` publishes its pixels onto the `target` `BitmapObj`, which `GetBitmap`
    /// hands back for the parent to `DrawBitmap`. Portable — no OS surface, like the WIC target.
    pub(crate) fn with_compatible_target(mut dc: d2d::DeviceContext, factory_iface: ID2D1Factory, w: u32, h: u32, dpi: f32) -> ComObject<Self> {
        dc.set_dpi(dpi, dpi);
        let (fid, did) = dc.domain_ids();
        dc.set_target_bitmap(d2d::bitmap_target(w, h, dpi));
        let bm = ComObject::new(BitmapObj {
            pixels: D2D_SIZE_U { width: w, height: h },
            dpi: (dpi, dpi),
            buf: RefCell::new(None),
            src: RefCell::new(None),
            options: D2D1_BITMAP_OPTIONS_TARGET,
            present: RefCell::new(None),
            factory: Some(factory_iface.clone()),
            domain_ids: Some((fid, did)),
            color_context: None,
        });
        let obj = ComObject::new(Self::new(dc, factory_iface, None));
        obj.inner.borrow_mut().target = Some(bm);
        obj
    }

    /// A copy of the bound target's pixels `(premul BGRA, stride, width, height)`, for
    /// `ID2D1Bitmap::CopyFromRenderTarget`. `None` if no target is bound.
    pub(crate) fn snapshot(&self) -> Option<(Vec<u8>, usize, u32, u32)> {
        let ctx = self.inner.borrow();
        let b = ctx.dc.target_bitmap()?;
        Some((b.data.clone(), b.stride, b.desc.width, b.desc.height))
    }

    /// Build a render-target context: bind `buf` into the core and keep `bm` (which carries the
    /// present destination) as the bound target. Used by the Hwnd/DC/DXGI render-target builders.
    #[cfg(windows)] // only the Windows-only presenting render targets build a context this way.
    pub(crate) fn with_target(mut dc: d2d::DeviceContext, bm: ComObject<BitmapObj>, buf: PixelBuffer, factory: ID2D1Factory) -> ComObject<Self> {
        dc.set_target_bitmap(buf);
        let obj = ComObject::new(Self::new(dc, factory, None));
        obj.inner.borrow_mut().target = Some(bm);
        obj
    }
}

/// Copy the finished target pixels into the bound `IWICBitmap` (`CreateWicBitmapRenderTarget`), if any.
/// The core renders premultiplied BGRA8, top-down, row pitch `buf.stride`; both D2D-supported WIC
/// target formats — 32bppPBGRA (premultiplied) and 32bppBGR (ignore) — store BGRA bytes in that same
/// layout, so this is a stride-aware byte copy into the WIC bitmap's locked buffer.
fn wic_writeback(ctx: &Ctx) -> WResult<()> {
    let Some(wic) = ctx.wic.as_ref() else { return Ok(()) };
    let Some(buf) = ctx.dc.target_bitmap() else { return Ok(()) };
    let (w, h) = (buf.desc.width as usize, buf.desc.height as usize);
    let row = w * 4;
    unsafe {
        let lock = wic.Lock(core::ptr::null(), windows::Win32::Graphics::Imaging::WICBitmapLockWrite.0 as u32)?;
        let dst_stride = lock.GetStride()? as usize;
        let (mut size, mut data) = (0u32, core::ptr::null_mut::<u8>());
        lock.GetDataPointer(&mut size, &mut data)?;
        if !data.is_null() {
            for y in 0..h {
                let src = &buf.data[y * buf.stride..y * buf.stride + row];
                core::slice::from_raw_parts_mut(data.add(y * dst_stride), row).copy_from_slice(src);
            }
        }
    }
    Ok(())
}

/// Copy the finished target pixels to the bound bitmap's present destination, if it has one.
/// Off Windows there are no present destinations (no HWND/DXGI/DC targets), so this is a no-op.
#[cfg(not(windows))]
fn present_bound(_ctx: &Ctx) -> WResult<()> {
    Ok(())
}
#[cfg(windows)]
fn present_bound(ctx: &Ctx) -> WResult<()> {
    let Some(bm) = ctx.target.as_ref() else { return Ok(()) };
    let present = bm.present.borrow();
    let (Some(p), Some(buf)) = (present.as_ref(), ctx.dc.target_bitmap()) else { return Ok(()) };
    p.flush(buf)
}

/// The params a context draws text with until the app sets its own (or after it sets NULL): DirectWrite's system
/// default rendering params, as Direct2D takes them from its DirectWrite factory.
fn system_text_params() -> RenderingParams {
    dwrite::default_rendering_params().map(|p| read_params(&p)).unwrap_or_default()
}

/// Read a `d2d::text::RenderingParams` from an `IDWriteRenderingParams` (one of our dwrite objects):
/// the five base scalars, plus grayscale contrast / grid-fit / the finer rendering mode from the 1/2/3
/// revisions when the object exposes them.
fn read_params(p: &IDWriteRenderingParams) -> RenderingParams {
    unsafe {
        let mode = p.cast::<IDWriteRenderingParams3>().map(|p3| p3.GetRenderingMode1().0 as u32).unwrap_or(p.GetRenderingMode().0 as u32);
        RenderingParams {
            rasterizer: d2d::text::hinting::Rasterizer::FontCache,
            dpi: [96.0, 96.0],
            gamma: p.GetGamma(),
            enhanced_contrast: p.GetEnhancedContrast(),
            grayscale_contrast: p.cast::<IDWriteRenderingParams1>().map(|p1| p1.GetGrayscaleEnhancedContrast()).unwrap_or(1.0),
            cleartype_level: p.GetClearTypeLevel(),
            pixel_geometry: p.GetPixelGeometry().0 as u32,
            rendering_mode: mode,
            grid_fit: p.cast::<IDWriteRenderingParams2>().map(|p2| p2.GetGridFitMode().0 as u32).unwrap_or(0),
            // Primitive blend isn't part of IDWriteRenderingParams; it's applied per-draw from the
            // context's SetPrimitiveBlend state (like the core path), so default to source-over here.
            blend: d2d::pixels::Blend::SourceOver,
            no_snap: false,
            // Set per draw by the device context, like the blend.
            raster_clip: None,
            clip_px: None,
        }
    }
}

/// Read a font face's raw file bytes + face index through the **public** `IDWriteFontFace`
/// interface — the fallback for a *foreign* face (created by the system DirectWrite, not our drop-in,
/// so it does not implement the private `ITextInterop`). `GetFiles` → `GetReferenceKey`/`GetLoader` →
/// `CreateStreamFromKey` → `ReadFileFragment`, exactly how DWrite itself resolves a face's bytes. The
/// first file is the font (every TTF/OTF/collection face references one file). Returns owned bytes.
fn foreign_face_bytes(face: &IDWriteFontFace) -> Option<(Vec<u8>, u32)> {
    unsafe {
        let mut n = 0u32;
        face.GetFiles(&mut n, None).ok()?;
        if n == 0 {
            return None;
        }
        let mut files: Vec<Option<IDWriteFontFile>> = vec![None; n as usize];
        face.GetFiles(&mut n, Some(files.as_mut_ptr())).ok()?;
        let file = files.into_iter().flatten().next()?;
        let mut key: *mut core::ffi::c_void = core::ptr::null_mut();
        let mut key_len = 0u32;
        file.GetReferenceKey(&mut key, &mut key_len).ok()?;
        let loader = file.GetLoader().ok()?;
        let stream = loader.CreateStreamFromKey(key as *const core::ffi::c_void, key_len).ok()?;
        let size = stream.GetFileSize().ok()?;
        let mut frag: *mut core::ffi::c_void = core::ptr::null_mut();
        let mut ctx: *mut core::ffi::c_void = core::ptr::null_mut();
        stream.ReadFileFragment(&mut frag, 0, size, &mut ctx).ok()?;
        let bytes = core::slice::from_raw_parts(frag as *const u8, size as usize).to_vec();
        stream.ReleaseFileFragment(ctx);
        Some((bytes, face.GetIndex()))
    }
}

/// Font bytes + face index for a glyph run's face: our fast path (`ITextInterop`, no copy) first, then
/// the public-interface fallback for a foreign face. Returns owned bytes in both cases (the caller
/// renders from them within the call).
fn interop_rasterizer<T: Interface>(object: &T) -> d2d::text::hinting::Rasterizer {
    let mut kind = 1;
    if let Ok(it) = object.cast::<ITextInterop>() {
        let _ = unsafe { it.FontRasterizer(&mut kind) };
    }
    if kind == 0 { d2d::text::hinting::Rasterizer::DirectWrite } else { d2d::text::hinting::Rasterizer::FontCache }
}

fn face_rasterizer(face: &IDWriteFontFace) -> d2d::text::hinting::Rasterizer {
    use d2d::text::hinting::Rasterizer;
    if face.cast::<ITextInterop>().is_ok() { return interop_rasterizer(face); }
    let path = (|| unsafe {
        let mut count = 0;
        face.GetFiles(&mut count, None).ok()?;
        if count == 0 { return None; }
        let mut files = vec![None; count as usize];
        face.GetFiles(&mut count, Some(files.as_mut_ptr())).ok()?;
        let file = files.into_iter().flatten().next()?;
        let mut key = core::ptr::null_mut();
        let mut len = 0;
        file.GetReferenceKey(&mut key, &mut len).ok()?;
        let loader: IDWriteLocalFontFileLoader = file.GetLoader().ok()?.cast().ok()?;
        let mut path = vec![0u16; loader.GetFilePathLengthFromKey(key, len).ok()? as usize + 1];
        loader.GetFilePathFromKey(key, len, &mut path).ok()?;
        Some(String::from_utf16_lossy(&path[..path.len() - 1]))
    })();
    path.map_or(Rasterizer::DirectWrite, |p| dwrite::font_file_rasterizer(&p))
}

fn run_face_bytes(face: &IDWriteFontFace) -> Option<(Vec<u8>, u32)> {
    if let Ok(it) = face.cast::<ITextInterop>() {
        let (mut data, mut len, mut idx) = (core::ptr::null(), 0usize, 0u32);
        if unsafe { it.FontBlob(&mut data, &mut len, &mut idx) }.is_ok() && !data.is_null() {
            return Some((unsafe { core::slice::from_raw_parts(data, len) }.to_vec(), idx));
        }
    }
    foreign_face_bytes(face)
}

/// `DrawText` inputs from a *foreign* `IDWriteTextFormat` (no `ITextInterop`): the format's public
/// getters give the layout fields, and the font bytes come from the family resolved through the
/// format's own font collection (`FindFamilyName` → `GetFontFamily` → `GetFirstMatchingFont` →
/// `CreateFontFace`), the way DWrite's own `DrawText` resolves the format's font.
fn foreign_format(fmt: &IDWriteTextFormat) -> Option<(Vec<u8>, u32, f32, u32, u32, i32, d2d::text::hinting::Rasterizer)> {
    unsafe {
        let em = fmt.GetFontSize();
        let align = fmt.GetTextAlignment().0 as u32;
        let wrap = fmt.GetWordWrapping().0 as u32;
        // Right-to-left and bottom-to-top (which lays out as right-to-left along its columns).
        let rtl = (fmt.GetReadingDirection().0 & 1) as i32;
        let collection = fmt.GetFontCollection().ok()?;
        let mut name = vec![0u16; fmt.GetFontFamilyNameLength() as usize + 1];
        fmt.GetFontFamilyName(&mut name).ok()?;
        let (mut index, mut exists) = (0u32, windows::core::BOOL(0));
        collection.FindFamilyName(windows_core::PCWSTR(name.as_ptr()), &mut index, &mut exists).ok()?;
        if !exists.as_bool() {
            return None;
        }
        let family = collection.GetFontFamily(index).ok()?;
        let font = family.GetFirstMatchingFont(fmt.GetFontWeight(), fmt.GetFontStretch(), fmt.GetFontStyle()).ok()?;
        let face = font.CreateFontFace().ok()?;
        let (bytes, idx) = foreign_face_bytes(&face)?;
        Some((bytes, idx, em, align, wrap, rtl, face_rasterizer(&face)))
    }
}

/// Our dwrite's system font collection (via the public DWrite factory), for resolving a fallback
/// candidate family to its bytes. `None` if the factory/collection can't be created.
fn system_collection() -> Option<IDWriteFontCollection> {
    unsafe {
        let factory: IDWriteFactory = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED).ok()?;
        let mut coll = None;
        factory.GetSystemFontCollection(&mut coll, false).ok()?;
        coll
    }
}

/// A family name → `(bytes, face index, simulations)` through the collection (FindFamilyName → the face
/// matching the format's weight/stretch/style → CreateFontFace → GetFiles) — the fallback loader
/// `draw_text_fallback` drives, the runtime equivalent of the suite's on-disk `load_fallback`.
/// Font fallback's face of a family for `ch`: its first matching font when that maps `ch`, else the family's
/// best match among the faces that do, in the matcher's order (style, then the nearest weight, the heavier
/// on a tie), an upright face standing in for a slanted request as its simulated oblique.
fn family_bytes(coll: &IDWriteFontCollection, name: &str, weight: DWRITE_FONT_WEIGHT, stretch: DWRITE_FONT_STRETCH, style: DWRITE_FONT_STYLE, ch: char) -> Option<(Vec<u8>, u32, u32)> {
    unsafe {
        let w: Vec<u16> = name.encode_utf16().chain(core::iter::once(0)).collect();
        let (mut idx, mut exists) = (0u32, windows::core::BOOL(0));
        coll.FindFamilyName(windows_core::PCWSTR(w.as_ptr()), &mut idx, &mut exists).ok()?;
        if !exists.as_bool() {
            return None;
        }
        let family = coll.GetFontFamily(idx).ok()?;
        let font = family.GetFirstMatchingFont(weight, stretch, style).ok()?;
        let (font, sims) = if font.HasCharacter(ch as u32).is_ok_and(|b| b.as_bool()) {
            let sims = font.GetSimulations().0 as u32;
            (font, sims)
        } else {
            let list = family.GetMatchingFonts(weight, stretch, style).ok()?;
            let fonts: Vec<IDWriteFont> = (0..list.GetFontCount()).filter_map(|i| list.GetFont(i).ok()).collect();
            let style_rank = |have: i32| match style.0 {
                2 => [2, 1, 0][have.clamp(0, 2) as usize],
                1 => [2, 0, 1][have.clamp(0, 2) as usize],
                _ => have.clamp(0, 2),
            };
            // A slanted request takes italic and oblique alike before weight (as the matcher does).
            let slant_rank = |have: i32| if style.0 == 0 { have.clamp(0, 2) } else if have == 0 { 2 } else { 0 };
            // An upright face ranks as its simulated oblique under a slanted request.
            let shown = |f: &IDWriteFont| if style.0 != 0 && f.GetStyle().0 == 0 { 1 } else { f.GetStyle().0 };
            let best = fonts
                .into_iter()
                .filter(|f| f.HasCharacter(ch as u32).is_ok_and(|b| b.as_bool()))
                .min_by_key(|f| (slant_rank(shown(f)), (f.GetWeight().0 - weight.0).unsigned_abs(), u32::MAX - f.GetWeight().0 as u32, style_rank(shown(f))))?;
            let sims = best.GetSimulations().0 as u32 | if shown(&best) == 1 && best.GetStyle().0 == 0 { 2 } else { 0 };
            (best, sims)
        };
        let face = font.CreateFontFace().ok()?;
        foreign_face_bytes(&face).map(|(b, i)| (b, i, sims))
    }
}

/// The text format's locale name (`fallback::candidates` is locale-aware — Han unification), via the
/// public `IDWriteTextFormat` getter so it works for our formats and foreign ones alike.
fn format_locale(fmt: &IDWriteTextFormat) -> String {
    unsafe {
        let n = fmt.GetLocaleNameLength();
        if n == 0 {
            return String::new();
        }
        let mut buf = vec![0u16; n as usize + 1];
        if fmt.GetLocaleName(&mut buf).is_err() {
            return String::new();
        }
        String::from_utf16_lossy(&buf[..n as usize])
    }
}

fn brush_props(p: *const D2D1_BRUSH_PROPERTIES) -> (f32, Option<[f32; 6]>) {
    if p.is_null() {
        (1.0, None)
    } else {
        let p = unsafe { &*p };
        (p.opacity, Some(mat(&p.transform)))
    }
}

/// The authored brush transform the facade echoes back (identity when no `D2D1_BRUSH_PROPERTIES`).
/// The core brush of an image brush: it realizes its image (bitmap, command list or effect output) over `source` and samples
/// that as a bitmap brush (`DeviceContext::create_image_brush`). Shared by `CreateImageBrush` and a brush rebuilt after a setter.
pub(crate) fn image_pixel_brush(dc: &mut d2d::DeviceContext, img: &ID2D1Image, source: &D2D_RECT_F, extend: [u32; 2], interp: u32, opacity: f32, transform: Option<[f32; 6]>) -> WResult<d2d::PixelBrush> {
    let source = rectf(source);
    if let Ok(l) = img.cast_object_ref::<CommandListObj>() {
        return Ok(dc.create_image_brush(d2d::BrushImage::CommandList(&l.list.clone()), source, extend, interp, opacity, transform));
    }
    match super::image::image_target_of(img) {
        Some(DrawTarget::Bitmap(b)) => Ok(dc.create_image_brush(d2d::BrushImage::Bitmap(&b), source, extend, interp, opacity, transform)),
        Some(DrawTarget::Effect(e, _)) => Ok(dc.create_image_brush(d2d::BrushImage::Effect(&e), source, extend, interp, opacity, transform)),
        None => Err(windows_core::Error::from(E_NOTIMPL)),
    }
}

fn brush_mx(p: *const D2D1_BRUSH_PROPERTIES) -> Matrix3x2 {
    if p.is_null() { Matrix3x2 { M11: 1.0, M12: 0.0, M21: 0.0, M22: 1.0, M31: 0.0, M32: 0.0 } } else { unsafe { &*p }.transform }
}

/// Read a `D2D1_GRADIENT_MESH_PATCH` into the core patch: row-major `point[row*4+col]`, the four
/// corner colours (`color00,03,30,33`) and per-edge modes `[left, top, right, bottom]`.
fn mesh_patch(p: &D2D1_GRADIENT_MESH_PATCH) -> MeshPatch {
    MeshPatch {
        points: [
            vec2(&p.point00), vec2(&p.point01), vec2(&p.point02), vec2(&p.point03),
            vec2(&p.point10), vec2(&p.point11), vec2(&p.point12), vec2(&p.point13),
            vec2(&p.point20), vec2(&p.point21), vec2(&p.point22), vec2(&p.point23),
            vec2(&p.point30), vec2(&p.point31), vec2(&p.point32), vec2(&p.point33),
        ],
        colors: [col(&p.color00), col(&p.color03), col(&p.color30), col(&p.color33)],
        extend: [p.leftEdgeMode.0 as u32, p.topEdgeMode.0 as u32, p.rightEdgeMode.0 as u32, p.bottomEdgeMode.0 as u32],
    }
}

/// The intrinsic DIP bounds of an `ID2D1Image` for `GetImage{Local,World}Bounds`: a bitmap's pixel
/// rect scaled to DIPs, an effect's finite output bounds, or the empty rect for an infinite image
/// (Flood/Border/Tile output, a command list) — as native returns `(0,0,0,0)` for an infinite image.
fn image_bounds_dip(img: Ref<ID2D1Image>) -> WResult<[f32; 4]> {
    match image_target(img) {
        Some(DrawTarget::Bitmap(b)) => Ok([0.0, 0.0, b.width as f32 * 96.0 / b.dpi.0, b.height as f32 * 96.0 / b.dpi.1]),
        Some(DrawTarget::Effect(e, _)) => Ok(e.intrinsic_bounds().unwrap_or([0.0; 4])),
        None => Ok([0.0; 4]),
    }
}

impl ContextObj {
    /// Run a drawing closure that drives the core renderer, catching any Rust panic and latching it as
    /// a deferred draw error (surfaced at `EndDraw`) instead of letting it unwind across the
    /// `extern "system"` COM vtable shim — which, since the shim is `nounwind`, aborts the host process
    /// (G5: a Paint.NET GPU plugin whose effect graph has a missing input took paint.rs down). Native
    /// defers such a failure to `EndDraw` rather than crashing; a drop-in must too. The documented
    /// trigger is a malformed effect graph (a missing effect input, directly or through an unrealised
    /// command-list input), for which native returns E_INVALIDARG; a genuinely unexpected panic also
    /// surfaces here rather than aborting. (Exact HRESULT per malformed-graph shape is VM-gated.)
    fn guard_draw(&self, f: impl FnOnce()) {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
            Ok(()) => {}
            Err(payload) => {
                // Surface the caught panic. A caught panic here means a real bug (this draw would CRASH a
                // panic=abort drop-in — d2d-ref, wasm, a browser — where catch_unwind is a no-op). Fuzz
                // gates match "panicked", but a suppressing panic hook (conformance/wic_fuzz install one)
                // mutes the default print, so echo the message ourselves in test/debug builds so it is
                // never silent. Release stays quiet and just latches the error.
                #[cfg(any(test, debug_assertions))]
                {
                    let msg = payload
                        .downcast_ref::<&str>()
                        .copied()
                        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                        .unwrap_or("<non-string panic payload>");
                    eprintln!("d2d guard_draw: a draw panicked (would crash a panic=abort build): {msg}");
                }
                self.inner.borrow_mut().dc.latch_error(windows_core::HRESULT(0x8007_0057u32 as i32));
            }
        }
    }

    /// Fill a core geometry with whichever brush kind `brush` is (solid or a gradient/bitmap/image
    /// brush), mirroring the core's `fill_rectangle*` (rects) / `fill_geometry*` (other) split.
    fn fill(&self, geom: Geometry, brush: Ref<ID2D1Brush>, rect_fastpath: bool) {
        if let Some(b) = brush.as_ref() {
            self.fill_brush(geom, b, rect_fastpath);
        } else {
            self.latch_invalid();
        }
    }

    /// Fill `geom` with a resolved `ID2D1Brush` (solid or gradient/bitmap/image). Shared by `fill`
    /// and `FillMesh` (which fills each triangle through here). Dispatches exactly as the conformance
    /// core driver's `Op::Fill`/`Op::Primitive` do, so the FFI pixels match the core (and oracle):
    /// * a **solid** brush on a rectangle takes the axis-aligned `fill_rectangle` fast path only for
    ///   `FillRectangle` (`rect_fastpath`); `FillGeometry(RectangleGeometry)` tessellates through
    ///   `fill_geometry` as the core does.
    /// * a **gradient/bitmap/image** brush on a rectangle always takes `fill_rectangle_brush` — the
    ///   core uses it for both `FillRectangle` and `FillGeometry(RectangleGeometry)` (09-linear-gamma);
    ///   the fast path and the general fill agree there, and it records the same command.
    fn fill_brush(&self, geom: Geometry, b: &ID2D1Brush, rect_fastpath: bool) {
        self.guard_draw(|| {
            let mut ctx = self.inner.borrow_mut();
            if let Ok(s) = b.cast_object_ref::<SolidBrushObj>() {
                // The brush's *current* state reaches the fill (and the recorded command), as native's does.
                let brush = s.live(&ctx.dc);
                match geom {
                    Geometry::Rect(r) if rect_fastpath => ctx.dc.fill_rectangle(r, &brush),
                    g => ctx.dc.fill_geometry(&g, &brush),
                }
            } else if let Ok(p) = b.cast_object_ref::<PixelBrushObj>() {
                let brush = p.live(&mut ctx.dc);
                match geom {
                    Geometry::Rect(r) => ctx.dc.fill_rectangle_brush(r, &brush),
                    g => ctx.dc.fill_geometry_brush(&g, &brush),
                }
            }
        });
    }

    /// `FillOpacityMask` (shared by the RenderTarget form, which carries `content`, and the
    /// DeviceContext form, which omits it — both composite the mask's alpha under the brush). Any
    /// brush: solid or gradient/bitmap/image.
    fn fill_opacity_mask(&self, mask: Ref<ID2D1Bitmap>, brush: Ref<ID2D1Brush>, content: u32, dest: *const D2D_RECT_F, source: *const D2D_RECT_F) {
        let Some(bm) = mask.as_ref().and_then(|b| b.cast_object_ref::<BitmapObj>().ok()) else { return self.latch_invalid() };
        let src_ref = bm.src.borrow();
        let Some(src) = src_ref.as_ref() else { return self.latch_invalid() };
        let Some(b) = brush.as_ref() else { return self.latch_invalid() };
        let dest = (!dest.is_null()).then(|| rectf(unsafe { &*dest }));
        let source = (!source.is_null()).then(|| rectf(unsafe { &*source }));
        self.guard_draw(|| {
            let mut ctx = self.inner.borrow_mut();
            if let Ok(s) = b.cast_object_ref::<SolidBrushObj>() {
                let brush = s.live(&ctx.dc);
                ctx.dc.fill_opacity_mask(src, &brush, content, dest, source);
            } else if let Ok(p) = b.cast_object_ref::<PixelBrushObj>() {
                let brush = p.live(&mut ctx.dc);
                ctx.dc.fill_opacity_mask_brush(src, &brush, content, dest, source);
            }
        });
    }

    /// Stroke a core geometry with any brush (solid or gradient/bitmap/image).
    fn draw(&self, geom: Geometry, brush: Ref<ID2D1Brush>, width: f32, style: StrokeStyle) {
        let Some(b) = brush.as_ref() else { return self.latch_invalid() };
        self.guard_draw(|| {
            let mut ctx = self.inner.borrow_mut();
            if let Ok(s) = b.cast_object_ref::<SolidBrushObj>() {
                let brush = s.live(&ctx.dc);
                ctx.dc.draw_geometry(&geom, &brush, width, &style);
            } else if let Ok(p) = b.cast_object_ref::<PixelBrushObj>() {
                let brush = p.live(&mut ctx.dc);
                ctx.dc.draw_geometry_brush(&geom, &brush, width, &style);
            }
        });
    }

    /// `PushLayer` (both the `D2D1_LAYER_PARAMETERS`/`…1` forms share these fields): the geometric
    /// mask and opacity brush are recovered from their interface pointers and forwarded. Bit 1 of
    /// `LAYER_OPTIONS` (INITIALIZE_FOR_CLEARTYPE) means INITIALIZE_FROM_BACKGROUND, as in `LAYER_OPTIONS1`.
    #[allow(clippy::too_many_arguments)]
    fn push_layer(&self, bounds: D2D_RECT_F, mask: &Option<ID2D1Geometry>, mask_aa: D2D1_ANTIALIAS_MODE, t: &Matrix3x2, opacity: f32, brush: &Option<ID2D1Brush>, options: u32) {
        self.guard_draw(|| {
            let mut ctx = self.inner.borrow_mut();
            let mask_obj = mask.as_ref().and_then(|g| g.cast_object_ref::<GeometryObj>().ok());
            let brush_obj = brush.as_ref().and_then(|b| b.cast_object_ref::<PixelBrushObj>().ok());
            let solid_obj = brush.as_ref().and_then(|b| b.cast_object_ref::<SolidBrushObj>().ok());
            // In d2d1's order the core latches WRONG_STATE for a not-in-draw push first; only once in draw
            // do we reject a mask geometry or opacity brush from another factory (WRONG_FACTORY). The layer
            // still pushes (the foreign mask dropped) so the layer stack stays balanced — the latched error
            // discards the frame anyway.
            let in_draw = ctx.dc.is_in_draw();
            let foreign = |f: &Option<ID2D1Factory>| f.as_ref().is_some_and(|f| f.as_raw() != self.factory.as_raw());
            let mask_f = mask_obj.as_ref().is_some_and(|o| foreign(&o.owning_factory()));
            let brush_f = brush_obj.as_ref().is_some_and(|o| foreign(&o.factory))
                || solid_obj.as_ref().is_some_and(|o| foreign(&o.factory));
            if in_draw && (mask_f || brush_f) {
                ctx.dc.latch_error(windows_core::HRESULT(0x8899_0012u32 as i32));
            }
            let mask_geom = if in_draw && mask_f { None } else { mask_obj.map(|o| o.geometry()) };
            let solid = solid_obj.map(|s| s.live(&ctx.dc).as_pixel_brush());
            let pixel = brush_obj.map(|o| o.live(&mut ctx.dc));
            ctx.dc.push_layer(rectf(&bounds), mask_geom.as_ref(), mask_aa.0 as u32, Some([t.M11, t.M12, t.M21, t.M22, t.M31, t.M32]), opacity, pixel.as_ref().or(solid.as_ref()), options);
        });
    }

    /// `DrawText`: pull the text-format's font bytes + layout fields out of our dwrite object (via
    /// [`ITextInterop`]) and render through the core's `draw_text` (layout → glyph runs → blend).
    fn draw_text(&self, s: &windows_core::PCWSTR, len: u32, fmt: Ref<IDWriteTextFormat>, rect: *const D2D_RECT_F, brush: Ref<ID2D1Brush>, options: D2D1_DRAW_TEXT_OPTIONS, measuring: DWRITE_MEASURING_MODE) {
        let Some(fmt) = fmt.as_ref() else { return self.latch_invalid() };
        let Some(b) = brush.as_ref() else { return self.latch_invalid() };
        // Fast path: our text format implements `ITextInterop` (borrowed blob). Fallback: a foreign
        // `IDWriteTextFormat` (system dwrite) read through its public getters + font collection.
        let (font_bytes, idx, em, align, wrap, rtl, sims, rasterizer) = if let Ok(it) = fmt.cast::<ITextInterop>() {
            let (mut data, mut dlen, mut fidx) = (core::ptr::null(), 0usize, 0u32);
            let (mut em, mut align, mut wrap, mut rtl, mut sims) = (0f32, 0u32, 0u32, 0i32, 0u32);
            unsafe {
                let _ = it.FontBlob(&mut data, &mut dlen, &mut fidx);
                let _ = it.FormatParams(&mut em, &mut align, &mut wrap, &mut rtl, &mut sims);
            }
            if data.is_null() {
                return self.latch_invalid();
            }
            (unsafe { core::slice::from_raw_parts(data, dlen) }.to_vec(), fidx, em, align, wrap, rtl, sims, interop_rasterizer(fmt))
        } else {
            match foreign_format(fmt) {
                Some((b, i, em, a, w, r, rasterizer)) => (b, i, em, a, w, r, 0u32, rasterizer),
                None => return self.latch_invalid(),
            }
        };
        // Native latches E_INVALIDARG (deferred to EndDraw with the op's tags) when the DEVICE em
        // reaches 32768 = 2^15 — the layout-extent limit — but ONLY under a GDI measuring mode; NATURAL
        // renders the glyph clipped at any size (see `draw_text_em_gate`, which folds the world
        // transform, not just dpi: fuzz 15125/15058 size+1e6-scale, VM probe natural-vs-GDI).
        // A NaN layout box fails native's internal CreateTextLayout first, silently (the core draws nothing).
        let r = if rect.is_null() { [0.0; 4] } else { rectf(unsafe { &*rect }) };
        let gate = if d2d::text::draw_text_rect_rejected(r) { None } else { self.inner.borrow().dc.draw_text_em_gate(em, measuring.0 as u32, len > 0) };
        if let Some(refused) = gate {
            if refused {
                self.latch_invalid();
            }
            return;
        }
        let bytes = font_bytes.as_slice();
        let text = String::from_utf16_lossy(unsafe { core::slice::from_raw_parts(s.0, len as usize) });
        // A NaN/negative layout box is handled SILENTLY inside the core draw (draw nothing, S_OK): native's
        // DrawText swallows its failed internal CreateTextLayout without latching (r12 17323/17356 EndDraw
        // hr 0). Only the app's own CreateTextLayout returns E_INVALIDARG (dwrite-ffi make_layout). See
        // `draw_text_rect_rejected`.
        // Paragraph alignment is read from the public getter (present on both our format and a foreign
        // one), so DrawText honours SetParagraphAlignment the way native's internal layout does.
        let para = unsafe { fmt.GetParagraphAlignment() }.0 as u8;
        let flow = unsafe { fmt.GetFlowDirection() }.0 as u8;
        let (mut lm, mut ls, mut lb) = (DWRITE_LINE_SPACING_METHOD(0), 0.0f32, 0.0f32);
        let _ = unsafe { fmt.GetLineSpacing(&mut lm, &mut ls, &mut lb) };
        let tf = TextFormat { em, base_rtl: rtl != 0, align: align as u8, wrap: wrap as u8, para, flow, line_spacing: (lm.0 as u8, ls, lb) };
        // Native's DrawText resolves system font fallback (CJK/Thai/Arabic → a covering family); drive
        // the core's fallback path with our system collection as the candidate loader, so a complex
        // script renders its glyphs, not tofu. `locale` steers Han unification.
        let locale = format_locale(fmt);
        let coll = system_collection();
        let (fw, fst, fs) = unsafe { (fmt.GetFontWeight(), fmt.GetFontStretch(), fmt.GetFontStyle()) };
        // The format's own family, for a character its matched face lacks, resolves in the format's collection.
        let own = unsafe {
            let mut name = vec![0u16; fmt.GetFontFamilyNameLength() as usize + 1];
            fmt.GetFontFamilyName(&mut name).ok().map(|_| String::from_utf16_lossy(&name[..name.len() - 1]))
        };
        let own_coll = unsafe { fmt.GetFontCollection() }.ok().or_else(|| coll.clone());
        let load = |name: &str, _: usize, ch: char| match name {
            d2d::text::fallback::OWN_FAMILY => own_coll.as_ref().zip(own.as_deref()).and_then(|(c, n)| family_bytes(c, n, fw, fst, fs, ch)),
            _ => coll.as_ref().and_then(|c| family_bytes(c, name, fw, fst, fs, ch)),
        };
        let m = measuring.0 as u32;
        // Variable-font instancing for DrawText: the format's weight/stretch/style (plus any explicit
        // IDWriteTextFormat3 axis values) map to the font's axis values the same way native's
        // `ConvertWeightStretchStyleToFontAxisValues` does, then normalize to the gvar coords the core
        // instances the outline with. A static font (no `fvar`) or the default instance → empty → base
        // outline, so this is inert for non-variable text. (opsz is not auto-applied — AUTOMATIC_FONT_
        // AXES defaults to NONE; wire `size` here when an optical-size scene needs it.)
        // `packed` (the resolved axis values in DWRITE user space) feeds the shaper so a variable-font
        // instance lays out with native's instanced advances; `coords` (normalized) feeds the raster.
        let mut packed: Vec<(u32, f32)> = Vec::new();
        let coords = {
            let mut explicit: Vec<DWRITE_FONT_AXIS_VALUE> = Vec::new();
            if let Ok(tf3) = fmt.cast::<IDWriteTextFormat3>() {
                let k = unsafe { tf3.GetFontAxisValueCount() } as usize;
                if k > 0 {
                    explicit = vec![DWRITE_FONT_AXIS_VALUE::default(); k];
                    let _ = unsafe { tf3.GetFontAxisValues(&mut explicit) };
                }
            }
            let (w, st, sy) = unsafe { (fmt.GetFontWeight(), fmt.GetFontStretch(), fmt.GetFontStyle()) };
            let user = dwrite::wss_to_axis_values(&explicit, w, st, sy, 0.0);
            packed = user.iter().map(|a| (a.axisTag.0, a.value)).collect();
            d2d::text::variations::normalized_from_dwrite(bytes, idx, &packed)
        };
        // The shaper only instances when the font is actually variable (the normalize above returns
        // empty for a static face); keep the user axes only then, so static text shapes unchanged.
        let shaper_vars: &[(u32, f32)] = if coords.is_empty() { &[] } else { &packed };
        self.guard_draw(|| {
            let mut ctx = self.inner.borrow_mut();
            let mut params = ctx.text_params;
            params.rasterizer = rasterizer;
            params.no_snap = options.0 & 1 != 0; // D2D1_DRAW_TEXT_OPTIONS_NO_SNAP
            // D2D1_DRAW_TEXT_OPTIONS_CLIP: an axis-aligned clip of the layout box around the draw, at the
            // current antialias mode (native DrawTextLayoutHelper).
            let clip = options.0 & 2 != 0 && !d2d::text::draw_text_rect_rejected(r);
            let clip = clip.then(|| {
                let (o, w, h) = d2d::text::normalize_layout_rect(r);
                ctx.dc.push_text_clip([o[0], o[1], o[0] + w, o[1] + h])
            });
            if let Ok(sb) = b.cast_object_ref::<SolidBrushObj>() {
                let live = sb.live(&ctx.dc);
                ctx.dc.draw_text_fallback(&text, bytes, idx, &locale, &tf, m, r, &live, &params, options.0 & 4 != 0, sims, &coords, shaper_vars, &[], None, &[], &[], &[], &[], load);
            } else if let Ok(pb) = b.cast_object_ref::<PixelBrushObj>() {
                let live = pb.live(&mut ctx.dc);
                ctx.dc.draw_text_fallback_brush(&text, bytes, idx, &locale, &tf, m, r, &live, &params, options.0 & 4 != 0, sims, &coords, shaper_vars, &[], None, &[], &[], &[], &[], load);
            }
            if let Some(layer) = clip {
                ctx.dc.pop_text_clip(layer);
            }
        });
    }

    /// `DrawGlyphRun`: read the positioned run (ids/advances/offsets direct), pull the font bytes from
    /// its font face (our dwrite object, via [`ITextInterop`]) and render through `draw_glyph_run`.
    fn draw_glyph_run(&self, origin: &Vector2, run: *const DWRITE_GLYPH_RUN, brush: Ref<ID2D1Brush>, measuring: DWRITE_MEASURING_MODE, palette: Option<u32>) {
        if run.is_null() { return self.latch_invalid(); }
        let run = unsafe { &*run };
        let Some(face) = run.fontFace.as_ref() else { return self.latch_invalid() };
        let Some(b) = brush.as_ref() else { return self.latch_invalid() };
        // Like DrawText (not DrawTextLayout, which native renders at any device em — its em limit is
        // enforced in CreateGdiCompatibleTextLayout at creation), DrawGlyphRun latches E_INVALIDARG
        // (deferred to EndDraw) at device em ≥ 32768 = 2^15 under a GDI measuring mode; NATURAL renders
        // clipped at any size. A glyph run emitted by a DrawTextLayout (LayoutRenderer re-entry) is exempt.
        let gate = if self.in_layout_draw.get() { None } else { self.inner.borrow().dc.draw_text_em_gate(run.fontEmSize, measuring.0 as u32, run.glyphCount > 0) };
        if let Some(refused) = gate {
            if refused {
                self.latch_invalid();
            }
            return;
        }
        // Our text objects (fast path) or any foreign `IDWriteFontFace` (public `GetFiles`).
        let Some((font_bytes, idx)) = run_face_bytes(face) else { return self.latch_invalid() };
        let sims = unsafe { face.GetSimulations() }.0 as u32;
        let bytes = font_bytes.as_slice();
        let n = run.glyphCount as usize;
        if n == 0 { return; }
        if run.glyphIndices.is_null() { return self.latch_invalid(); }
        let ids = unsafe { core::slice::from_raw_parts(run.glyphIndices, n) };
        let default_advances;
        let adv = if run.glyphAdvances.is_null() {
            let mut metrics = vec![DWRITE_GLYPH_METRICS::default(); n];
            let mut font_metrics = DWRITE_FONT_METRICS::default();
            unsafe {
                face.GetMetrics(&mut font_metrics);
                if face.GetDesignGlyphMetrics(ids.as_ptr(), n as u32, metrics.as_mut_ptr(), run.isSideways.as_bool()).is_err() {
                    return self.latch_invalid();
                }
            }
            if font_metrics.designUnitsPerEm == 0 { return self.latch_invalid(); }
            default_advances = metrics.iter().map(|m| m.advanceWidth as f32 * run.fontEmSize / font_metrics.designUnitsPerEm as f32).collect::<Vec<_>>();
            &default_advances
        } else { unsafe { core::slice::from_raw_parts(run.glyphAdvances, n) } };
        let off = if run.glyphOffsets.is_null() { &[] } else { unsafe { core::slice::from_raw_parts(run.glyphOffsets, n) } };
        let glyphs: Vec<Glyph> =
            (0..n).map(|i| Glyph { id: ids[i], advance: adv[i], offset: off.get(i).map(|o| [o.advanceOffset, o.ascenderOffset]).unwrap_or([0.0, 0.0]) }).collect();
        // Variable-font instancing: the face's axis values (a named instance, or a FontResource-
        // created instance) become the run's normalized variation coords, so the core renders the
        // gvar-interpolated outline. Default/static faces report default values → empty → no change.
        let axis_vals: Vec<(u32, f32)> = match face.cast::<IDWriteFontFace5>() {
            Ok(f5) => {
                let m = unsafe { f5.GetFontAxisValueCount() };
                let mut v = vec![DWRITE_FONT_AXIS_VALUE::default(); m as usize];
                if m > 0 {
                    let _ = unsafe { f5.GetFontAxisValues(&mut v) };
                }
                v.iter().map(|a| (a.axisTag.0, a.value)).collect()
            }
            Err(_) => Vec::new(),
        };
        let coords = d2d::text::variations::normalized_from_dwrite(bytes, idx, &axis_vals);
        let gr = GlyphRun {
            font_data: bytes,
            face_index: idx,
            em_size: run.fontEmSize,
            glyphs: &glyphs,
            baseline_origin: [origin.X, origin.Y],
            measuring_mode: measuring.0 as u32,
            is_sideways: run.isSideways.as_bool(),
            bidi_level: run.bidiLevel,
            // DrawGlyphRun draws at the exact origin — no baseline pixel-snapping (a DrawText/layout
            // feature). Native places the run on the sub-pixel grid directly.
            no_snap: true,
            simulations: sims,
            coords: &coords,
        };
        self.guard_draw(|| {
            let mut ctx = self.inner.borrow_mut();
            let mut params = ctx.text_params;
            params.rasterizer = face_rasterizer(face);
            // A colour palette means `DrawPaintGlyphRun` (COLRv1 paint tree); its own palette colours apply,
            // not the brush, so the colour path stays solid-only. A plain run takes any brush.
            if let Ok(sb) = b.cast_object_ref::<SolidBrushObj>() {
                let brush = sb.live(&ctx.dc);
                match palette {
                    Some(p) => { ctx.dc.draw_color_glyph_run(&gr, &brush, p, &params); }
                    None => { ctx.dc.draw_glyph_run(&gr, &brush, &params); }
                }
            } else if let Ok(pb) = b.cast_object_ref::<PixelBrushObj>() {
                if palette.is_none() {
                    let brush = pb.live(&mut ctx.dc);
                    ctx.dc.draw_glyph_run_brush(&gr, &brush, &params);
                }
            }
        });
    }

    /// `DrawTextLayout`: render a (possibly foreign) `IDWriteTextLayout` the way native does — drive
    /// its `Draw` with an internal `IDWriteTextRenderer` that draws each glyph run (with its per-range
    /// `SetDrawingEffect` brush, else the default), underlines/strikethroughs as rects, and inline
    /// objects via their own `Draw`. Works for our layouts and foreign ones alike.
    fn draw_text_layout(&self, rt: ID2D1RenderTarget, origin: &Vector2, layout: Ref<IDWriteTextLayout>, brush: Ref<ID2D1Brush>, options: D2D1_DRAW_TEXT_OPTIONS) {
        let (Some(layout), Some(brush)) = (layout.as_ref(), brush.as_ref()) else { return self.latch_invalid() };
        // D2D1_DRAW_TEXT_OPTIONS_CLIP: clip to the layout box (origin + its max width/height), as native
        // DrawTextLayoutHelper does from the layout's metrics.
        let clip = (options.0 & 2 != 0).then(|| {
            let (mw, mh) = unsafe { (layout.GetMaxWidth(), layout.GetMaxHeight()) };
            self.inner.borrow_mut().dc.push_text_clip([origin.X, origin.Y, origin.X + mw, origin.Y + mh])
        });
        let (t, dpi) = {
            let c = self.inner.borrow();
            (c.dc.get_transform(), c.dc.get_dpi())
        };
        // Device transform (DPI-scale ∘ world), mirroring core's `device_transform()`: each column of
        // the world matrix scaled by the axis DPI factor. Used only for baseline snapping; the matrix
        // handed to dwrite's `Draw` stays the world transform (it reads DPI via GetPixelsPerDip).
        let (sx, sy) = (dpi.0 as f64 / 96.0, dpi.1 as f64 / 96.0);
        let dev_dt = [t[0] as f64 * sx, t[1] as f64 * sy, t[2] as f64 * sx, t[3] as f64 * sy, t[4] as f64 * sx, t[5] as f64 * sy];
        let renderer: IDWriteTextRenderer = LayoutRenderer {
            rt,
            brush: brush.clone(),
            transform: DWRITE_MATRIX { m11: t[0], m12: t[1], m21: t[2], m22: t[3], dx: t[4], dy: t[5] },
            dev_dt,
            ppd: dpi.0 / 96.0,
            no_snap: options.0 & 1 != 0,
        }
        .into();
        self.in_layout_draw.set(true);
        let _ = unsafe { layout.Draw(None, &renderer, origin.X, origin.Y) };
        self.in_layout_draw.set(false);
        if let Some(layer) = clip {
            self.inner.borrow_mut().dc.pop_text_clip(layer);
        }
    }

    /// `DrawBitmap`: resolve the bitmap's pixel source and composite through the core. Shared by the
    /// ID2D1RenderTarget 1.0 overload (`BITMAP_INTERPOLATION_MODE`) and the DeviceContext overload
    /// (`INTERPOLATION_MODE` + perspective); both pass `interp` as the raw mode value (0=nearest,
    /// 1=linear, matching `filter_of`). A null dest/source defaults to the whole bitmap (DIP = texel
    /// count, as native's 1.0 overload).
    /// Latch `hr` as a deferred draw error (surfaced at EndDraw with the current tags), for a draw
    /// that cannot resolve a bound resource. Native defers such errors to EndDraw; a silent no-op +
    /// S_OK would hide an internal failure (e.g. a bitmap whose pixels never loaded under memory
    /// pressure) and leave wrong pixels — a fingerprint. E_INVALIDARG is D2D's invalid-resource code.
    fn latch_invalid(&self) {
        self.inner.borrow_mut().dc.latch_error(windows_core::HRESULT(0x8007_0057u32 as i32));
    }

    /// Latch D2DERR_WRONG_FACTORY (deferred to EndDraw) and return false when `rf` (a bound resource's
    /// owning factory) is not this context's factory — native rejects a resource from another factory
    /// in a draw. (A same-factory, different-device resource is D2DERR_WRONG_RESOURCE_DOMAIN, which the
    /// core brush path already latches; the FFI image/geometry objects carry only their factory here.)
    fn check_factory(&self, rf: &Option<ID2D1Factory>) -> bool {
        if let Some(rf) = rf {
            if rf.as_raw() != self.factory.as_raw() {
                self.inner.borrow_mut().dc.latch_error(windows_core::HRESULT(0x8899_0012u32 as i32)); // WRONG_FACTORY
                return false;
            }
        }
        true
    }

    /// Return D2DERR_WRONG_FACTORY directly (the `Create*` call's own HRESULT, not a deferred draw
    /// error) when `rf` is not this context's factory — the resource-identity check a `Create*` that
    /// takes another resource makes before using it, as native does. `check_factory` is the draw-call
    /// (latch-deferred) sibling.
    fn require_factory(&self, rf: &Option<ID2D1Factory>) -> WResult<()> {
        if rf.as_ref().is_some_and(|rf| rf.as_raw() != self.factory.as_raw()) {
            return Err(windows_core::Error::from(windows_core::HRESULT(0x8899_0012u32 as i32)));
        }
        Ok(())
    }

    /// The stroke style for a draw, or `None` (having latched WRONG_FACTORY) when it is from another
    /// factory — the caller then bails. A missing or non-ours style is the default, as `stroke_of` gives.
    fn checked_stroke(&self, s: Ref<ID2D1StrokeStyle>) -> Option<StrokeStyle> {
        match s.as_ref().and_then(|s| s.cast_object_ref::<StrokeStyleObj>().ok()) {
            Some(o) => self.check_factory(&o.factory).then(|| o.style.clone()),
            None => Some(StrokeStyle::default()),
        }
    }

    fn draw_bitmap(&self, bitmap: Ref<ID2D1Bitmap>, dest: *const D2D_RECT_F, opacity: f32, interp: u32, source: *const D2D_RECT_F) {
        let Some(bm) = bitmap.as_ref() else { return self.latch_invalid() };
        let Ok(obj) = bm.cast_object_ref::<BitmapObj>() else { return self.latch_invalid() };
        // D2D1_INTERPOLATION_MODE is 0..=5 (NEAREST..HIGH_QUALITY_CUBIC); native defers E_INVALIDARG
        // for an out-of-range mode to EndDraw with the op's tags.
        if interp > 5 {
            return self.latch_invalid();
        }
        // Cross-domain: a bitmap from another factory → WRONG_FACTORY, from another device (same
        // factory) → WRONG_RESOURCE_DOMAIN (core `check_domain`, as the brush path). `domain_ids` None
        // (a 1.0 present bitmap) falls back to the factory-only check.
        match obj.domain_ids {
            Some((f, d)) => {
                if !self.inner.borrow_mut().dc.check_draw_domain(f, d) {
                    return;
                }
            }
            None => {
                if !self.check_factory(&obj.factory) {
                    return;
                }
            }
        }
        // The source pixels: an uploaded image, else the bitmap's rendered target buffer (a compatible
        // render target's GetBitmap hands back a TARGET-only bitmap whose pixels live in `buf`).
        let Some(src) = obj.readable_src() else { return self.latch_invalid() };
        let full = [0.0, 0.0, src.width as f32, src.height as f32];
        let dest = if dest.is_null() { full } else { rectf(unsafe { &*dest }) };
        let source = if source.is_null() { full } else { rectf(unsafe { &*source }) };
        self.guard_draw(|| self.inner.borrow_mut().dc.draw_bitmap(&src, dest, source, opacity, interp));
    }

    /// `DrawImage`: a closed command list replays its recorded primitives; a bare `Flood` uses the
    /// composite-aware `draw_flood`; another effect output runs the effect graph; anything else blits
    /// a bitmap.
    fn draw_image(&self, image: Ref<ID2D1Image>, offset: [f32; 2], rect: Option<[f32; 4]>, interp: i32, composite: i32) {
        // D2D1_COMPOSITE_MODE is 0..=12 (SOURCE_OVER..MASK_INVERT). Native rejects any other value with a
        // latched E_INVALIDARG for DrawImage of ANY image type, deferred to the next HRESULT call / EndDraw.
        // The command-list core path already checks this; validate up front so bitmap and effect targets are
        // covered too (r11 16045/16337/16392/16615/16814/16844/16924, all composite 13 on a bitmap/effect).
        if !(0..=12).contains(&composite) {
            self.inner.borrow_mut().dc.latch_error(windows_core::HRESULT(0x8007_0057u32 as i32));
            return;
        }
        // Cross-domain bitmap image: another factory → WRONG_FACTORY, another device → WRONG_RESOURCE_DOMAIN
        // (as DrawBitmap). (Command lists / effects carry their own domain through the core graph.)
        if let Some(b) = image.as_ref().and_then(|i| i.cast_object_ref::<BitmapObj>().ok()) {
            match b.domain_ids {
                Some((f, d)) => {
                    if !self.inner.borrow_mut().dc.check_draw_domain(f, d) {
                        return;
                    }
                }
                None => {
                    if !self.check_factory(&b.factory) {
                        return;
                    }
                }
            }
        }
        if let Some(list) = image.as_ref().and_then(|i| i.cast_object_ref::<CommandListObj>().ok()) {
            self.inner.borrow_mut().dc.draw_image_list(&list.list, offset, rect, interp as u32, composite as u32);
            return;
        }
        // A ColorManagement output native can't realise (SIMPLE profile): record the draw's tags so
        // EndDraw reports E_INVALIDARG latched there, and draw nothing (cms2-04/05).
        if image.as_ref().and_then(|i| i.cast_object_ref::<CmFailObj>().ok()).is_some() {
            let mut ctx = self.inner.borrow_mut();
            let tags = ctx.dc.get_tags();
            ctx.cm_fail = Some(tags);
            return;
        }
        // The effect-graph pull runs core code that can panic on a malformed graph (a missing input, G5); guard
        // so that aborts the draw into an EndDraw error, never the host process.
        self.guard_draw(move || {
            match image_target(image) {
                // A bare Flood takes the BGRA8 composite path; on a float target it goes through the effect graph's
                // f32 path like any other image (draw_flood writes UNORM8 bytes).
                Some(DrawTarget::Effect(_, Some(flood))) if !self.inner.borrow().dc.target_bitmap().is_some_and(|t| d2d::float::is_highp(t.desc.format)) => {
                    self.inner.borrow_mut().dc.draw_flood(flood, offset, rect, interp as u32, composite as u32)
                }
                Some(DrawTarget::Effect(e, _)) => {
                    e.mark_drawn(); // a HistogramOutput readback becomes valid once the effect is drawn
                    self.inner.borrow_mut().dc.draw_image_effect_rect(&e, offset, rect, interp as u32, composite as u32);
                }
                Some(DrawTarget::Bitmap(b)) => self.inner.borrow_mut().dc.draw_image_bitmap(&b, offset, rect, interp as u32, composite as u32),
                None => {}
            }
        });
    }
}

impl ID2D1Resource_Impl for ContextObj_Impl {
    fn GetFactory(&self, f: windows_core::OutRef<ID2D1Factory>) {
        let _ = f.write(Some(self.factory.clone()));
    }
}

impl ID2D1RenderTarget_Impl for ContextObj_Impl {
    // The 1.0 bitmap/brush creators forward to the 1.1 (`ID2D1DeviceContext`) overloads: a 1.0 bitmap
    // is a 1.1 bitmap with NONE options and no colour context; a 1.0 bitmap brush's interpolation enum
    // (0=NEAREST, 1=LINEAR) is the low end of the 1.1 enum. `ID2D1Bitmap1`/`ID2D1BitmapBrush1` carry the
    // 1.0 interfaces, so the result casts straight back. Common 1.0 drop-in paths.
    fn CreateBitmap(&self, s: &D2D_SIZE_U, d: *const core::ffi::c_void, p: u32, b: *const D2D1_BITMAP_PROPERTIES) -> WResult<ID2D1Bitmap> {
        let props1 = (unsafe { b.as_ref() }).map(|b| D2D1_BITMAP_PROPERTIES1 {
            pixelFormat: b.pixelFormat, dpiX: b.dpiX, dpiY: b.dpiY,
            bitmapOptions: D2D1_BITMAP_OPTIONS(0), colorContext: core::mem::ManuallyDrop::new(None),
        });
        let pp = props1.as_ref().map_or(core::ptr::null(), |p| p as *const _);
        ID2D1DeviceContext_Impl::CreateBitmap(self, s, d, p, pp)?.cast()
    }
    fn CreateBitmapFromWicBitmap(&self, w: Ref<windows::Win32::Graphics::Imaging::IWICBitmapSource>, b: *const D2D1_BITMAP_PROPERTIES) -> WResult<ID2D1Bitmap> {
        let props1 = (unsafe { b.as_ref() }).map(|b| D2D1_BITMAP_PROPERTIES1 {
            pixelFormat: b.pixelFormat, dpiX: b.dpiX, dpiY: b.dpiY,
            bitmapOptions: D2D1_BITMAP_OPTIONS(0), colorContext: core::mem::ManuallyDrop::new(None),
        });
        let pp = props1.as_ref().map_or(core::ptr::null(), |p| p as *const _);
        ID2D1DeviceContext_Impl::CreateBitmapFromWicBitmap(self, w, pp)?.cast()
    }
    fn CreateSharedBitmap(&self, _r: *const windows_core::GUID, d: *mut core::ffi::c_void, b: *const D2D1_BITMAP_PROPERTIES, o: windows_core::OutRef<ID2D1Bitmap>) -> WResult<()> {
        // Native ID2D1RenderTarget::CreateSharedBitmap shares an existing resource identified by QI
        // (not by trusting `riid`): an IWICBitmapLock, another ID2D1Bitmap, or (Windows) an IDXGISurface.
        // A bare IWICBitmap/IWICBitmapSource is NOT a supported resource — native rejects it with
        // E_INVALIDARG (VM-verified on all RT types), so any resource that is not one of the three
        // falls through to E_INVALIDARG. ponytail: a WIC lock is snapshotted, not live memory-shared.
        let unk = unsafe { windows_core::IUnknown::from_raw_borrowed(&d) }.ok_or(windows_core::Error::from(super::E_INVALIDARG))?;
        let dpi = if b.is_null() || unsafe { &*b }.dpiX == 0.0 { (96.0, 96.0) } else { (unsafe { &*b }.dpiX, unsafe { &*b }.dpiY) };
        let bitmap = if let Ok(lock) = unk.cast::<windows::Win32::Graphics::Imaging::IWICBitmapLock>() {
            super::image::bitmap_from_wic_lock(&lock)?
        } else if let Ok(src) = unk.cast_object_ref::<BitmapObj>() {
            src.src_clone().ok_or(windows_core::Error::from(super::E_INVALIDARG))?
        } else {
            return Err(super::E_INVALIDARG.into()); // IWICBitmap / unsupported resource (native: E_INVALIDARG)
        };
        let size = D2D_SIZE_U { width: bitmap.width, height: bitmap.height };
        let domain_ids = Some(self.inner.borrow().dc.domain_ids());
        let obj: ID2D1Bitmap1 = BitmapObj {
            pixels: size, dpi, buf: RefCell::new(None), src: RefCell::new(Some(bitmap)), options: D2D1_BITMAP_OPTIONS(0),
            present: RefCell::new(None), factory: Some(self.factory_iface()), domain_ids, color_context: None,
        }
        .into();
        let _ = o.write(obj.cast().ok());
        Ok(())
    }
    fn CreateBitmapBrush(&self, b: Ref<ID2D1Bitmap>, bp: *const D2D1_BITMAP_BRUSH_PROPERTIES, p: *const D2D1_BRUSH_PROPERTIES) -> WResult<ID2D1BitmapBrush> {
        let bp1 = (unsafe { bp.as_ref() }).map(|bp| D2D1_BITMAP_BRUSH_PROPERTIES1 {
            extendModeX: bp.extendModeX, extendModeY: bp.extendModeY,
            interpolationMode: D2D1_INTERPOLATION_MODE(bp.interpolationMode.0),
        });
        let pbp = bp1.as_ref().map_or(core::ptr::null(), |p| p as *const _);
        ID2D1DeviceContext_Impl::CreateBitmapBrush(self, b, pbp, p)?.cast()
    }
    fn CreateSolidColorBrush(&self, color: *const D2D1_COLOR_F, brushprops: *const D2D1_BRUSH_PROPERTIES) -> WResult<ID2D1SolidColorBrush> {
        let opacity = (!brushprops.is_null()).then(|| unsafe { &*brushprops }.opacity);
        let brush = self.inner.borrow().dc.create_solid_color_brush(col(unsafe { &*color }), opacity);
        Ok(SolidBrushObj::new(brush, unsafe { *color }, opacity.unwrap_or(1.0), brush_mx(brushprops)).with_factory(self.factory_iface()).into())
    }
    fn CreateGradientStopCollection(&self, stops: *const D2D1_GRADIENT_STOP, count: u32, gamma: D2D1_GAMMA, extend: D2D1_EXTEND_MODE) -> WResult<ID2D1GradientStopCollection> {
        // StopsObj implements ID2D1GradientStopCollection1; hand back the base interface it derives.
        let obj: ID2D1GradientStopCollection1 = unsafe { StopsObj::read(stops, count, gamma.0 as u32, extend.0 as u32) }.with_factory(self.factory_iface()).into();
        obj.cast()
    }
    fn CreateLinearGradientBrush(&self, props: *const D2D1_LINEAR_GRADIENT_BRUSH_PROPERTIES, brushprops: *const D2D1_BRUSH_PROPERTIES, stops: Ref<ID2D1GradientStopCollection>) -> WResult<ID2D1LinearGradientBrush> {
        let p = unsafe { &*props };
        let sc = stops.as_ref().ok_or(windows_core::Error::from(E_NOTIMPL))?;
        let so = sc.cast_object_ref::<StopsObj>()?;
        let (opacity, transform) = brush_props(brushprops);
        let brush = self.inner.borrow().dc.create_linear_gradient_brush(
            vec2(&p.startPoint),
            vec2(&p.endPoint),
            &so.stops,
            so.gamma,
            so.extend,
            opacity,
            transform,
        );
        let params = BrushParams::Linear { start: p.startPoint, end: p.endPoint, stops: sc.clone() };
        let b: ID2D1LinearGradientBrush = PixelBrushObj::new(brush, opacity, brush_mx(brushprops), params).with_factory(self.factory_iface()).into();
        #[cfg(all(windows, target_env = "gnu"))]
        super::resources::gnu_point_abi::patch_linear(&b);
        Ok(b)
    }
    fn CreateRadialGradientBrush(&self, props: *const D2D1_RADIAL_GRADIENT_BRUSH_PROPERTIES, brushprops: *const D2D1_BRUSH_PROPERTIES, stops: Ref<ID2D1GradientStopCollection>) -> WResult<ID2D1RadialGradientBrush> {
        let p = unsafe { &*props };
        let sc = stops.as_ref().ok_or(windows_core::Error::from(E_NOTIMPL))?;
        let so = sc.cast_object_ref::<StopsObj>()?;
        let (opacity, transform) = brush_props(brushprops);
        let brush = self.inner.borrow().dc.create_radial_gradient_brush(
            vec2(&p.center),
            vec2(&p.gradientOriginOffset),
            [p.radiusX, p.radiusY],
            &so.stops,
            so.gamma,
            so.extend,
            opacity,
            transform,
        );
        let params = BrushParams::Radial { center: p.center, offset: p.gradientOriginOffset, radius: [p.radiusX, p.radiusY], stops: sc.clone() };
        let b: ID2D1RadialGradientBrush = PixelBrushObj::new(brush, opacity, brush_mx(brushprops), params).with_factory(self.factory_iface()).into();
        #[cfg(all(windows, target_env = "gnu"))]
        super::resources::gnu_point_abi::patch_radial(&b);
        Ok(b)
    }
    fn CreateCompatibleRenderTarget(&self, desired_size: *const D2D_SIZE_F, desired_pixel_size: *const D2D_SIZE_U, _fmt: *const D2D1_PIXEL_FORMAT, _opts: D2D1_COMPATIBLE_RENDER_TARGET_OPTIONS) -> WResult<ID2D1BitmapRenderTarget> {
        // Size resolution (native): desiredPixelSize wins; else desiredSize (DIPs) × dpi; else the
        // parent's own pixel size. DPI and format are inherited from the parent (we always render
        // premultiplied BGRA8, native's default compatible-target format).
        let ctx = self.inner.borrow();
        let dpi = ctx.dc.get_dpi().0;
        let parent = ctx.dc.target_bitmap();
        let (pw, ph) = parent.map(|b| (b.desc.width, b.desc.height)).unwrap_or((0, 0));
        let (w, h) = if let Some(ps) = unsafe { desired_pixel_size.as_ref() } {
            (ps.width, ps.height)
        } else if let Some(ds) = unsafe { desired_size.as_ref() } {
            ((ds.width * dpi / 96.0).round() as u32, (ds.height * dpi / 96.0).round() as u32)
        } else {
            (pw, ph)
        };
        let sibling = ctx.dc.sibling();
        drop(ctx);
        ContextObj::with_compatible_target(sibling, self.factory_iface(), w, h, dpi).cast()
    }
    fn CreateLayer(&self, s: *const D2D_SIZE_F) -> WResult<ID2D1Layer> {
        // A null size means "size at PushLayer" — native stores (0,0); the backing surface is lazy.
        let size = if s.is_null() { D2D_SIZE_F { width: 0.0, height: 0.0 } } else { unsafe { *s } };
        Ok(super::resources::LayerObj { size, factory: Some(self.factory_iface()) }.into())
    }
    fn CreateMesh(&self) -> WResult<ID2D1Mesh> {
        Ok(MeshObj::new(self.factory_iface()).into())
    }
    fn DrawLine(&self, p0: &Vector2, p1: &Vector2, b: Ref<ID2D1Brush>, w: f32, s: Ref<ID2D1StrokeStyle>) {
        let sobj = s.as_ref().and_then(|s| s.cast_object_ref::<StrokeStyleObj>().ok());
        // A stroke style from another factory is WRONG_FACTORY.
        if !self.check_factory(&sobj.as_ref().and_then(|o| o.factory.clone())) {
            return;
        }
        let style = sobj.map(|o| o.style.clone());
        let Some(brush) = b.as_ref() else { return self.latch_invalid() };
        self.guard_draw(|| {
            let mut ctx = self.inner.borrow_mut();
            if let Ok(sb) = brush.cast_object_ref::<SolidBrushObj>() {
                let live = sb.live(&ctx.dc);
                ctx.dc.draw_line(vec2(p0), vec2(p1), &live, w, style.as_ref());
            } else if let Ok(pb) = brush.cast_object_ref::<PixelBrushObj>() {
                let live = pb.live(&mut ctx.dc);
                ctx.dc.draw_line_brush(vec2(p0), vec2(p1), &live, w, style.as_ref());
            }
        });
    }
    fn DrawRectangle(&self, rect: *const D2D_RECT_F, brush: Ref<ID2D1Brush>, width: f32, style: Ref<ID2D1StrokeStyle>) {
        let rect = rectf(unsafe { &*rect });
        let Some(style) = self.checked_stroke(style) else { return };
        let Some(b) = brush.as_ref() else { return self.latch_invalid() };
        self.guard_draw(|| {
            let mut ctx = self.inner.borrow_mut();
            if let Ok(s) = b.cast_object_ref::<SolidBrushObj>() {
                let live = s.live(&ctx.dc);
                ctx.dc.draw_rectangle(rect, &live, width, &style);
            } else if let Ok(p) = b.cast_object_ref::<PixelBrushObj>() {
                let live = p.live(&mut ctx.dc);
                ctx.dc.draw_rectangle_brush(rect, &live, width, &style);
            }
        });
    }
    fn FillRectangle(&self, rect: *const D2D_RECT_F, brush: Ref<ID2D1Brush>) {
        self.fill(Geometry::Rect(rectf(unsafe { &*rect })), brush, true);
    }
    fn DrawRoundedRectangle(&self, rr: *const D2D1_ROUNDED_RECT, brush: Ref<ID2D1Brush>, width: f32, style: Ref<ID2D1StrokeStyle>) {
        let r = unsafe { &*rr };
        let Some(style) = self.checked_stroke(style) else { return };
        self.draw(Geometry::Rounded { rect: rectf(&r.rect), radius: [r.radiusX, r.radiusY] }, brush, width, style);
    }
    fn FillRoundedRectangle(&self, rr: *const D2D1_ROUNDED_RECT, brush: Ref<ID2D1Brush>) {
        let r = unsafe { &*rr };
        self.fill(Geometry::Rounded { rect: rectf(&r.rect), radius: [r.radiusX, r.radiusY] }, brush, true);
    }
    fn DrawEllipse(&self, ellipse: *const D2D1_ELLIPSE, brush: Ref<ID2D1Brush>, width: f32, style: Ref<ID2D1StrokeStyle>) {
        let e = unsafe { &*ellipse };
        let Some(style) = self.checked_stroke(style) else { return };
        self.draw(Geometry::Ellipse { center: [e.point.X, e.point.Y], radius: [e.radiusX, e.radiusY] }, brush, width, style);
    }
    fn FillEllipse(&self, ellipse: *const D2D1_ELLIPSE, brush: Ref<ID2D1Brush>) {
        let e = unsafe { &*ellipse };
        self.fill(Geometry::Ellipse { center: [e.point.X, e.point.Y], radius: [e.radiusX, e.radiusY] }, brush, true);
    }
    fn DrawGeometry(&self, geometry: Ref<ID2D1Geometry>, brush: Ref<ID2D1Brush>, width: f32, style: Ref<ID2D1StrokeStyle>) {
        let Some(obj) = geometry.as_ref().and_then(|g| g.cast_object_ref::<GeometryObj>().ok()) else { return self.latch_invalid() };
        if !self.check_factory(&obj.owning_factory()) {
            return;
        }
        let Some(style) = self.checked_stroke(style) else { return };
        self.draw(obj.geometry(), brush, width, style);
    }
    fn FillGeometry(&self, geometry: Ref<ID2D1Geometry>, brush: Ref<ID2D1Brush>, _opacity: Ref<ID2D1Brush>) {
        let Some(obj) = geometry.as_ref().and_then(|g| g.cast_object_ref::<GeometryObj>().ok()) else { return self.latch_invalid() };
        if !self.check_factory(&obj.owning_factory()) {
            return;
        }
        self.fill(obj.geometry(), brush, false);
    }
    fn FillMesh(&self, m: Ref<ID2D1Mesh>, b: Ref<ID2D1Brush>) {
        let Some(mesh) = m.as_ref().and_then(|m| m.cast_object_ref::<MeshObj>().ok()) else { return self.latch_invalid() };
        // Native requires aliased antialias mode: in per-primitive FillMesh latches D2DERR_WRONG_STATE
        // (the sticky error, surfaced at the next HRESULT call and at EndDraw) and draws nothing.
        if !self.inner.borrow_mut().dc.aliased_draw_check() {
            return;
        }
        // Native rejects a mesh from another factory (WRONG_FACTORY, after the state/aliased check).
        if !self.check_factory(&mesh.factory) {
            return;
        }
        let Some(brush) = b.as_ref() else { return self.latch_invalid() };
        // Each triangle is filled on its own (so overlapping triangles double-cover), as native does.
        // Native's FillMesh rasterizes each triangle like a D3D triangle (1/256 snap + top-left), not
        // the geometry scan converter — both solid and gradient/bitmap/image go through that rule for
        // byte-exact edges (the scan converter placed the triangle edge a pixel differently).
        let tris = mesh.tris.borrow().clone();
        self.guard_draw(|| {
            let mut ctx = self.inner.borrow_mut();
            if let Ok(s) = brush.cast_object_ref::<SolidBrushObj>() {
                let live = s.live(&ctx.dc);
                for t in &tris {
                    ctx.dc.fill_mesh_solid(t, &live);
                }
            } else if let Ok(p) = brush.cast_object_ref::<PixelBrushObj>() {
                let live = p.live(&mut ctx.dc);
                for t in &tris {
                    ctx.dc.fill_mesh_brush(t, &live);
                }
            }
        });
    }
    fn FillOpacityMask(&self, mask: Ref<ID2D1Bitmap>, brush: Ref<ID2D1Brush>, content: D2D1_OPACITY_MASK_CONTENT, dest: *const D2D_RECT_F, source: *const D2D_RECT_F) {
        self.fill_opacity_mask(mask, brush, content.0 as u32, dest, source);
    }
    fn DrawBitmap(&self, b: Ref<ID2D1Bitmap>, d: *const D2D_RECT_F, o: f32, i: D2D1_BITMAP_INTERPOLATION_MODE, s: *const D2D_RECT_F) {
        self.draw_bitmap(b, d, o, i.0 as u32, s);
    }
    fn DrawText(&self, s: &windows_core::PCWSTR, l: u32, f: Ref<IDWriteTextFormat>, r: *const D2D_RECT_F, b: Ref<ID2D1Brush>, o: D2D1_DRAW_TEXT_OPTIONS, m: DWRITE_MEASURING_MODE) {
        self.draw_text(s, l, f, r, b, o, m);
    }
    fn DrawTextLayout(&self, o: &Vector2, l: Ref<IDWriteTextLayout>, b: Ref<ID2D1Brush>, opt: D2D1_DRAW_TEXT_OPTIONS) {
        if let Ok(rt) = self.to_interface::<ID2D1DeviceContext7>().cast::<ID2D1RenderTarget>() {
            self.draw_text_layout(rt, o, l, b, opt);
        }
    }
    fn DrawGlyphRun(&self, o: &Vector2, g: *const DWRITE_GLYPH_RUN, b: Ref<ID2D1Brush>, m: DWRITE_MEASURING_MODE) {
        self.draw_glyph_run(o, g, b, m, None);
    }
    fn SetTransform(&self, transform: *const Matrix3x2) {
        self.inner.borrow_mut().dc.set_transform(mat(unsafe { &*transform }));
    }
    fn GetTransform(&self, transform: *mut Matrix3x2) {
        let m = self.inner.borrow().dc.get_transform();
        unsafe { *transform = Matrix3x2 { M11: m[0], M12: m[1], M21: m[2], M22: m[3], M31: m[4], M32: m[5] } };
    }
    fn SetAntialiasMode(&self, mode: D2D1_ANTIALIAS_MODE) {
        self.inner.borrow_mut().dc.set_antialias_mode(mode.0 as u32);
    }
    fn GetAntialiasMode(&self) -> D2D1_ANTIALIAS_MODE {
        D2D1_ANTIALIAS_MODE(self.inner.borrow().dc.get_antialias_mode() as i32)
    }
    fn SetTextAntialiasMode(&self, mode: D2D1_TEXT_ANTIALIAS_MODE) {
        self.inner.borrow_mut().dc.set_text_antialias_mode(mode.0 as u32);
    }
    fn GetTextAntialiasMode(&self) -> D2D1_TEXT_ANTIALIAS_MODE {
        D2D1_TEXT_ANTIALIAS_MODE(self.inner.borrow().dc.get_text_antialias_mode() as i32)
    }
    fn SetTextRenderingParams(&self, p: Ref<IDWriteRenderingParams>) {
        let mut c = self.inner.borrow_mut();
        c.dc.note_text_rendering_params(p.as_ref().is_some());
        c.text_params = p.as_ref().map(read_params).unwrap_or_else(system_text_params);
        c.text_params_obj = p.as_ref().cloned();
    }
    fn GetTextRenderingParams(&self, p: windows_core::OutRef<IDWriteRenderingParams>) {
        // Hand back exactly the object last set (NULL only when none was ever set), as native does.
        let _ = p.write(self.inner.borrow().text_params_obj.clone());
    }
    fn SetTags(&self, tag1: u64, tag2: u64) {
        self.inner.borrow_mut().dc.set_tags(tag1, tag2);
    }
    fn GetTags(&self, tag1: *mut u64, tag2: *mut u64) {
        let (t1, t2) = self.inner.borrow().dc.get_tags();
        unsafe {
            if !tag1.is_null() {
                *tag1 = t1;
            }
            if !tag2.is_null() {
                *tag2 = t2;
            }
        }
    }
    fn PushLayer(&self, l: *const D2D1_LAYER_PARAMETERS, _layer: Ref<ID2D1Layer>) {
        let p = unsafe { &*l };
        self.push_layer(p.contentBounds, &p.geometricMask, p.maskAntialiasMode, &p.maskTransform, p.opacity, &p.opacityBrush, p.layerOptions.0 as u32);
    }
    fn PopLayer(&self) {
        self.inner.borrow_mut().dc.pop_layer();
    }
    fn Flush(&self, tag1: *mut u64, tag2: *mut u64) -> WResult<()> {
        let (hr, t1, t2) = self.inner.borrow_mut().dc.flush();
        unsafe {
            if !tag1.is_null() {
                *tag1 = t1;
            }
            if !tag2.is_null() {
                *tag2 = t2;
            }
        }
        hr.ok()
    }
    fn SaveDrawingState(&self, block: Ref<ID2D1DrawingStateBlock>) {
        let Some(b) = block.as_ref() else { return };
        let Ok(sb) = b.cast_object_ref::<StateBlockObj>() else { return };
        self.inner.borrow().dc.save_drawing_state(&mut sb.state.borrow_mut());
    }
    fn RestoreDrawingState(&self, block: Ref<ID2D1DrawingStateBlock>) {
        let Some(b) = block.as_ref() else { return };
        let Ok(sb) = b.cast_object_ref::<StateBlockObj>() else { return };
        let state = *sb.state.borrow();
        self.inner.borrow_mut().dc.restore_drawing_state(&state);
    }
    fn PushAxisAlignedClip(&self, r: *const D2D_RECT_F, antialias: D2D1_ANTIALIAS_MODE) {
        self.inner.borrow_mut().dc.push_axis_aligned_clip(rectf(unsafe { &*r }), antialias.0 as u32);
    }
    fn PopAxisAlignedClip(&self) {
        self.inner.borrow_mut().dc.pop_axis_aligned_clip();
    }
    fn Clear(&self, color: *const D2D1_COLOR_F) {
        let c = (!color.is_null()).then(|| col(unsafe { &*color }));
        self.inner.borrow_mut().dc.clear(c);
    }
    fn BeginDraw(&self) {
        let mut ctx = self.inner.borrow_mut();
        ctx.cm_fail = None;
        // CopyFromMemory/CopyFromBitmap may have updated the target between drawing scopes.
        if let Some(buf) = ctx.target.as_ref().and_then(|bm|bm.buffer_clone()) {
            ctx.dc.set_target_bitmap(buf);
        }
        ctx.dc.begin_draw();
    }
    fn EndDraw(&self, tag1: *mut u64, tag2: *mut u64) -> WResult<()> {
        let mut ctx = self.inner.borrow_mut();
        let (mut hr, mut t1, mut t2) = ctx.dc.end_draw();
        // A SIMPLE-profile ColorManagement draw fails realisation in native: EndDraw returns
        // E_INVALIDARG latched at that draw's tags (cms2-04/05). The core draw path can't model it, so
        // override when the core latched no earlier error.
        if hr.is_ok() {
            if let Some((ct1, ct2)) = ctx.cm_fail {
                hr = windows_core::HRESULT(0x8007_0057u32 as i32);
                (t1, t2) = (ct1, ct2);
            }
        }
        unsafe {
            if !tag1.is_null() {
                *tag1 = t1;
            }
            if !tag2.is_null() {
                *tag2 = t2;
            }
        }
        hr.ok()?;
        if let (Some(bm), Some(buf)) = (ctx.target.as_ref(), ctx.dc.target_bitmap()) {
            bm.set_buffer(buf.clone());
        }
        // Copy the finished frame to its destination: an IWICBitmap (WIC render target, portable) and/or
        // the window / DXGI surface (Windows presenting targets).
        wic_writeback(&ctx)?;
        present_bound(&ctx)
    }
    fn GetPixelFormat(&self) -> D2D1_PIXEL_FORMAT {
        D2D1_PIXEL_FORMAT { format: DXGI_FORMAT_B8G8R8A8_UNORM, alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED }
    }
    fn SetDpi(&self, dpix: f32, dpiy: f32) {
        self.inner.borrow_mut().dc.set_dpi(dpix, dpiy);
    }
    fn GetDpi(&self, dpix: *mut f32, dpiy: *mut f32) {
        let (x, y) = self.inner.borrow().dc.get_dpi();
        unsafe {
            *dpix = x;
            *dpiy = y;
        }
    }
    fn GetSize(&self) -> D2D_SIZE_F {
        let ctx = self.inner.borrow();
        match ctx.dc.target_bitmap() {
            Some(b) => D2D_SIZE_F { width: b.desc.width as f32 * 96.0 / ctx.dc.get_dpi().0, height: b.desc.height as f32 * 96.0 / ctx.dc.get_dpi().1 },
            None => D2D_SIZE_F::default(),
        }
    }
    fn GetPixelSize(&self) -> D2D_SIZE_U {
        match self.inner.borrow().dc.target_bitmap() {
            Some(b) => D2D_SIZE_U { width: b.desc.width, height: b.desc.height },
            None => D2D_SIZE_U::default(),
        }
    }
    fn GetMaximumBitmapSize(&self) -> u32 {
        16384
    }
    fn IsSupported(&self, _p: *const D2D1_RENDER_TARGET_PROPERTIES) -> BOOL {
        true.into()
    }
}

impl ID2D1DeviceContext_Impl for ContextObj_Impl {
    fn CreateBitmap(&self, size: &D2D_SIZE_U, sourcedata: *const core::ffi::c_void, pitch: u32, props: *const D2D1_BITMAP_PROPERTIES1) -> WResult<ID2D1Bitmap1> {
        let (w, h) = (size.width, size.height);
        let (fmt, alpha, requested_dpi, options) = if props.is_null() {
            (d2d::abi::DXGI_FORMAT_B8G8R8A8_UNORM, d2d::abi::D2D1_ALPHA_MODE_PREMULTIPLIED, (0.0, 0.0), D2D1_BITMAP_OPTIONS_NONE)
        } else {
            let p = unsafe { &*props };
            (d2d::abi::DXGI_FORMAT(p.pixelFormat.format.0), d2d::abi::D2D1_ALPHA_MODE(p.pixelFormat.alphaMode.0), (p.dpiX, p.dpiY), p.bitmapOptions)
        };
        let (dpi, _) = d2d::bitmap::validate_create_bitmap(fmt, alpha, d2d::abi::D2D1_BITMAP_OPTIONS(options.0), requested_dpi.0, requested_dpi.1, self.inner.borrow().dc.get_dpi()).map_err(windows_core::Error::from)?;
        let bpp = d2d::bitmap::bytes_per_pixel(fmt).ok_or(windows_core::Error::from(d2d::bitmap::WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT))?;
        let row = (w as usize).checked_mul(bpp).ok_or(windows_core::Error::from(super::E_INVALIDARG))?;
        let len = row.checked_mul(h as usize).filter(|&n| n <= isize::MAX as usize).ok_or(windows_core::Error::from(super::E_INVALIDARG))?;
        if !sourcedata.is_null() && (pitch as usize) < row {
            return Err(super::E_INVALIDARG.into());
        }
        let mut data = Vec::new();
        data.try_reserve_exact(len).map_err(|_| windows_core::Error::from(windows_core::HRESULT(0x8007000Eu32 as i32)))?;
        data.resize(len, 0);
        if !sourcedata.is_null() {
            self.inner.borrow().dc.note_texture_upload();
            for y in 0..h as usize {
                let p = unsafe { (sourcedata as *const u8).add(y * pitch as usize) };
                data[y * row..(y + 1) * row].copy_from_slice(unsafe { core::slice::from_raw_parts(p, row) });
            }
        }
        // Storage is verbatim DXGI texels. Alpha interpretation belongs to sampling, not upload.
        // Empty drawable bitmaps need the same source storage as initialized ones: CopyFromMemory
        // fills that storage, and DrawBitmap/brushes consume it. Only targets/readback use buffers.
        let (buf, src) = if options.contains(D2D1_BITMAP_OPTIONS_TARGET) || options.contains(D2D1_BITMAP_OPTIONS_CPU_READ) {
            (Some(d2d::contracts::PixelBuffer {
                desc: d2d::contracts::ImageDesc { width: w, height: h, format: fmt, alpha_mode: alpha, dpi_x: dpi.0, dpi_y: dpi.1 },
                stride: row, data,
            }), None)
        } else {
            (None, Some(Bitmap { width: w, height: h, premul_bgra: data, format: fmt, alpha, dpi }))
        };
        let options = if props.is_null() { D2D1_BITMAP_OPTIONS(0) } else { unsafe { &*props }.bitmapOptions };
        let color_context = if props.is_null() { None } else { unsafe { (*props).colorContext.as_ref().cloned() } };
        let domain_ids = Some(self.inner.borrow().dc.domain_ids());
        Ok(BitmapObj { pixels: *size, dpi, buf: RefCell::new(buf), src: RefCell::new(src), options, present: RefCell::new(None), factory: Some(self.factory_iface()), domain_ids, color_context }.into())
    }
    fn CreateBitmapFromWicBitmap(&self, w: Ref<windows::Win32::Graphics::Imaging::IWICBitmapSource>, props: *const D2D1_BITMAP_PROPERTIES1) -> WResult<ID2D1Bitmap1> {
        // The WIC source is premultiplied BGRA8 (`bitmap_from_wic`); a non-BGRA target format would
        // need conversion, which no passing scene exercises.
        let src = w.as_ref().ok_or(windows_core::Error::from(E_NOTIMPL))?;
        let bitmap = bitmap_from_wic(src)?;
        self.inner.borrow().dc.note_texture_upload();
        let size = D2D_SIZE_U { width: bitmap.width, height: bitmap.height };
        let dpi = if props.is_null() || unsafe { &*props }.dpiX == 0.0 { (96.0, 96.0) } else { (unsafe { &*props }.dpiX, unsafe { &*props }.dpiY) };
        let options = if props.is_null() { D2D1_BITMAP_OPTIONS(0) } else { unsafe { &*props }.bitmapOptions };
        let color_context = if props.is_null() { None } else { unsafe { (*props).colorContext.as_ref().cloned() } };
        let domain_ids = Some(self.inner.borrow().dc.domain_ids());
        Ok(BitmapObj { pixels: size, dpi, buf: RefCell::new(None), src: RefCell::new(Some(bitmap)), options, present: RefCell::new(None), factory: Some(self.factory_iface()), domain_ids, color_context }.into())
    }
    fn CreateColorContext(&self, space: D2D1_COLOR_SPACE, profile: *const u8, size: u32) -> WResult<ID2D1ColorContext> {
        // CUSTOM takes the raw ICC bytes; SRGB/SCRGB are named contexts.
        let ctx = if space.0 == 0 {
            let bytes = if profile.is_null() { &[][..] } else { unsafe { core::slice::from_raw_parts(profile, size as usize) } };
            ColorContext::from_profile_bytes(bytes)
        } else {
            ColorContext::from_color_space(d2d::abi::D2D1_COLOR_SPACE(space.0))
        };
        let cc: ID2D1ColorContext1 = ColorContextObj { ctx, factory: Some(self.factory_iface()) }.into();
        cc.cast()
    }
    fn CreateColorContextFromFilename(&self, filename: &windows_core::PCWSTR) -> WResult<ID2D1ColorContext> {
        // Native reads the ICC file and parses it; the core's `from_profile_bytes` (oracle-verified via
        // the CreateColorContext/CUSTOM path) does the parse, so this is read-file → that parse. A read
        // failure maps to its HRESULT_FROM_WIN32 (file-not-found etc.), as native surfaces.
        let path = unsafe { filename.to_string() }.map_err(|_| windows_core::Error::from(E_NOTIMPL))?;
        let bytes = std::fs::read(&path)
            .map_err(|e| windows_core::Error::from(windows_core::HRESULT(0x8007_0000u32 as i32 | (e.raw_os_error().unwrap_or(2) & 0xffff))))?;
        let cc: ID2D1ColorContext1 = ColorContextObj { ctx: ColorContext::from_profile_bytes(&bytes), factory: Some(self.factory_iface()) }.into();
        cc.cast()
    }
    fn CreateColorContextFromWicColorContext(&self, c: Ref<windows::Win32::Graphics::Imaging::IWICColorContext>) -> WResult<ID2D1ColorContext> {
        let ctx = super::image::color_context_from_wic(c.as_ref().ok_or(windows_core::Error::from(windows_core::HRESULT(0x8007_0057u32 as i32)))?)?;
        let cc: ID2D1ColorContext1 = ColorContextObj { ctx, factory: Some(self.factory_iface()) }.into();
        cc.cast()
    }
    fn CreateBitmapFromDxgiSurface(&self, s: Ref<windows::Win32::Graphics::Dxgi::IDXGISurface>, props: *const D2D1_BITMAP_PROPERTIES1) -> WResult<ID2D1Bitmap1> {
        // A DXGI surface only exists on Windows; elsewhere apps use CreateBitmap + CopyFromMemory.
        #[cfg(not(windows))]
        return {
            let _ = (s, props);
            Err(E_NOTIMPL.into())
        };
        #[cfg(windows)]
        {
            let dpi = if props.is_null() || unsafe { &*props }.dpiX == 0.0 { (96.0, 96.0) } else { (unsafe { &*props }.dpiX, unsafe { &*props }.dpiY) };
            // CANNOT_DRAW ⇒ a target that cannot be a drawable source, so skip the readback into `src`.
            let readable = props.is_null() || !unsafe { &*props }.bitmapOptions.contains(D2D1_BITMAP_OPTIONS_CANNOT_DRAW);
            Ok(super::present::bitmap_from_dxgi(s, dpi, readable, self.factory_iface())?.into())
        }
    }
    fn CreateEffect(&self, e: *const windows_core::GUID) -> WResult<ID2D1Effect> {
        let clsid = unsafe { *e };
        // ColorManagement now has a core graph node (Kind::ColorManagement): it evaluates the colour
        // transform in f32 through the graph, so a cms feeding another effect keeps the context
        // precision (asifcolorspace) instead of a pre-quantised 8bpc bitmap. Its colour contexts are
        // bound through `EffectObj::SetValue` (COLOR_CONTEXT properties 0/2) like any other effect.
        let effect = self.inner.borrow().dc.create_effect(clsid).map_err(windows_core::Error::from)?;
        Ok(EffectObj::new(effect, clsid).into())
    }
    fn CreateGradientStopCollection(&self, s: *const D2D1_GRADIENT_STOP, sc: u32, pre: D2D1_COLOR_SPACE, post: D2D1_COLOR_SPACE, bp: D2D1_BUFFER_PRECISION, e: D2D1_EXTEND_MODE, ci: D2D1_COLOR_INTERPOLATION_MODE) -> WResult<ID2D1GradientStopCollection1> {
        // Direct2D's CreateGradientStopCollection (…1): rejects an empty list, a CUSTOM (0)
        // interpolation space, an out-of-range colour space / extend / interpolation mode, and an
        // unknown buffer precision — E_INVALIDARG — then stores all fields verbatim.
        if sc == 0 || pre.0 == 0 || post.0 == 0 || pre.0 > 2 || post.0 > 2 || bp.0 < 1 || bp.0 > 5 || e.0 < 0 || e.0 > 2 || ci.0 < 0 || ci.0 > 1 {
            return Err(super::E_INVALIDARG.into());
        }
        Ok(unsafe { StopsObj::read1(s, sc, pre.0 as u32, post.0 as u32, bp.0 as u32, e.0 as u32, ci.0 as u32) }.with_factory(self.factory_iface()).into())
    }
    fn CreateImageBrush(&self, image: Ref<ID2D1Image>, imageprops: *const D2D1_IMAGE_BRUSH_PROPERTIES, brushprops: *const D2D1_BRUSH_PROPERTIES) -> WResult<ID2D1ImageBrush> {
        let img = image.as_ref().ok_or(windows_core::Error::from(E_NOTIMPL))?.clone();
        let ip = unsafe { &*imageprops };
        let (opacity, transform) = brush_props(brushprops);
        let (extend, interp) = ([ip.extendModeX.0 as u32, ip.extendModeY.0 as u32], ip.interpolationMode.0 as u32);
        let brush = image_pixel_brush(&mut self.inner.borrow_mut().dc, &img, &ip.sourceRectangle, extend, interp, opacity, transform)?;
        let params = BrushParams::Image {
            extend: [ip.extendModeX, ip.extendModeY],
            interp: ip.interpolationMode,
            source: ip.sourceRectangle,
            image: RefCell::new(Some(img)),
        };
        Ok(PixelBrushObj::new(brush, opacity, brush_mx(brushprops), params).with_factory(self.factory_iface()).into())
    }
    fn CreateBitmapBrush(&self, bitmap: Ref<ID2D1Bitmap>, bp: *const D2D1_BITMAP_BRUSH_PROPERTIES1, brushprops: *const D2D1_BRUSH_PROPERTIES) -> WResult<ID2D1BitmapBrush1> {
        let bmp = bitmap.as_ref().ok_or(windows_core::Error::from(E_NOTIMPL))?;
        let obj = bmp.cast_object_ref::<BitmapObj>()?;
        let src_ref = obj.src.borrow();
        let src = src_ref.as_ref().ok_or(windows_core::Error::from(E_NOTIMPL))?;
        let p = unsafe { &*bp };
        let (opacity, transform) = brush_props(brushprops);
        let brush = self.inner.borrow().dc.create_bitmap_brush(
            src,
            [p.extendModeX.0 as u32, p.extendModeY.0 as u32],
            p.interpolationMode.0 as u32,
            opacity,
            transform,
        );
        let params = BrushParams::Bitmap { extend: [p.extendModeX, p.extendModeY], interp: p.interpolationMode, bitmap: RefCell::new(Some(bmp.clone())) };
        Ok(PixelBrushObj::new(brush, opacity, brush_mx(brushprops), params).with_factory(self.factory_iface()).into())
    }
    fn CreateCommandList(&self) -> WResult<ID2D1CommandList> {
        // A draw error latched before this call (e.g. WRONG_STATE from FillMesh/DrawSpriteBatch under
        // per-primitive AA, or E_INVALIDARG from a bad DrawImage) is NOT surfaced at CreateCommandList:
        // native creates the list, lets the caller record its ops, and returns the error from
        // ID2D1CommandList::Close (VM probe 2026-11-07, seeds 15026/16026: native runs the list's nested
        // ops so the tags reach the last nested value and Close reports the error). Capture it and defer
        // to Close so recording proceeds and the tags match — not an abort here that empties the list and
        // leaves the tag at the op level (r11 15026/15030/15082/15095 stay fixed; ~62 tag cases align).
        let err = self.inner.borrow().dc.current_error();
        let list = self.inner.borrow().dc.create_command_list();
        Ok(CommandListObj { list, factory: Some(self.factory_iface()), pending_err: err.is_err().then_some(err) }.into())
    }
    fn IsDxgiFormatSupported(&self, _f: DXGI_FORMAT) -> BOOL {
        true.into()
    }
    fn IsBufferPrecisionSupported(&self, _p: D2D1_BUFFER_PRECISION) -> BOOL {
        true.into()
    }
    fn GetImageLocalBounds(&self, i: Ref<ID2D1Image>) -> WResult<D2D_RECT_F> {
        let dip = image_bounds_dip(i)?;
        let b = self.inner.borrow().dc.image_local_bounds(dip);
        Ok(D2D_RECT_F { left: b[0], top: b[1], right: b[2], bottom: b[3] })
    }
    fn GetImageWorldBounds(&self, i: Ref<ID2D1Image>) -> WResult<D2D_RECT_F> {
        let dip = image_bounds_dip(i)?;
        let b = self.inner.borrow().dc.image_world_bounds(dip);
        Ok(D2D_RECT_F { left: b[0], top: b[1], right: b[2], bottom: b[3] })
    }
    fn GetGlyphRunWorldBounds(&self, o: &Vector2, g: *const DWRITE_GLYPH_RUN, m: DWRITE_MEASURING_MODE) -> WResult<D2D_RECT_F> {
        let run = unsafe { &*g };
        let it = run.fontFace.as_ref().and_then(|f| f.cast::<ITextInterop>().ok()).ok_or(windows_core::Error::from(E_NOTIMPL))?;
        let sims = run.fontFace.as_ref().map(|f| unsafe { f.GetSimulations() }.0 as u32).unwrap_or(0);
        let (mut data, mut dlen, mut idx) = (core::ptr::null(), 0usize, 0u32);
        unsafe { let _ = it.FontBlob(&mut data, &mut dlen, &mut idx); }
        if data.is_null() {
            return Err(E_NOTIMPL.into());
        }
        let bytes = unsafe { core::slice::from_raw_parts(data, dlen) };
        let n = run.glyphCount as usize;
        let ids = unsafe { core::slice::from_raw_parts(run.glyphIndices, n) };
        let adv = unsafe { core::slice::from_raw_parts(run.glyphAdvances, n) };
        let off = unsafe { core::slice::from_raw_parts(run.glyphOffsets, n) };
        let glyphs: Vec<Glyph> =
            (0..n).map(|i| Glyph { id: ids[i], advance: adv[i], offset: [off[i].advanceOffset, off[i].ascenderOffset] }).collect();
        let gr = GlyphRun {
            font_data: bytes,
            face_index: idx,
            em_size: run.fontEmSize,
            glyphs: &glyphs,
            baseline_origin: [o.X, o.Y],
            measuring_mode: m.0 as u32,
            is_sideways: run.isSideways.as_bool(),
            bidi_level: run.bidiLevel,
            no_snap: false,
            simulations: sims,
            coords: &[],
        };
        let ctx = self.inner.borrow();
        let mut params = ctx.text_params;
        if let Some(face) = run.fontFace.as_ref() { params.rasterizer = face_rasterizer(face); }
        let b = ctx.dc.glyph_run_world_bounds(&gr, &params);
        Ok(D2D_RECT_F { left: b[0], top: b[1], right: b[2], bottom: b[3] })
    }
    fn GetDevice(&self, d: windows_core::OutRef<ID2D1Device>) {
        // The device that created this context (how apps spin up sibling contexts). `None` only for
        // the 1.0 render targets, which native does not expose a device on.
        let _ = d.write(self.device.clone());
    }
    fn SetTarget(&self, image: Ref<ID2D1Image>) {
        let mut ctx = self.inner.borrow_mut();
        // A command-list target diverts subsequent draws into the list; the bound bitmap stays put.
        if let Some(list) = image.as_ref().and_then(|i| i.cast_object_ref::<CommandListObj>().ok()) {
            ctx.dc.record_into(&list.list);
            return;
        }
        // Any other target: stop recording, hand the current buffer back to its bitmap, then bind.
        ctx.dc.stop_recording();
        if let Some(old) = ctx.target.take() {
            if let Some(buf) = ctx.dc.take_target() {
                old.set_buffer(buf);
            }
        }
        if let Some(img) = image.as_ref() {
            if let Ok(bm) = img.cast_object::<BitmapObj>() {
                if let Some(buf) = bm.buffer_clone() {
                    ctx.dc.set_target_bitmap(buf);
                }
                ctx.target = Some(bm);
            }
        }
    }
    fn GetTarget(&self, image: windows_core::OutRef<ID2D1Image>) {
        if let Some(bm) = self.inner.borrow().target.as_ref() {
            let _ = image.write(bm.cast::<ID2D1Image>().ok());
        }
    }
    fn SetRenderingControls(&self, c: *const D2D1_RENDERING_CONTROLS) {
        // Precision caps each effect intermediate; the tile size is the tiles a layer pop walks.
        // An invalid precision or a zero tile dimension drops the whole call, as Direct2D does.
        if !c.is_null() {
            let controls = unsafe { *c };
            let (p, t) = (controls.bufferPrecision.0 as u32, [controls.tileSize.width, controls.tileSize.height]);
            if p <= 5 && t[0] != 0 && t[1] != 0 {
                self.rendering_controls.set(controls);
            }
            self.inner.borrow_mut().dc.set_rendering_controls(p, t);
        }
    }
    fn GetRenderingControls(&self, c: *mut D2D1_RENDERING_CONTROLS) {
        if !c.is_null() {
            unsafe { *c = self.rendering_controls.get() };
        }
    }
    fn SetPrimitiveBlend(&self, blend: D2D1_PRIMITIVE_BLEND) {
        self.inner.borrow_mut().dc.set_primitive_blend(blend.0 as u32);
    }
    fn GetPrimitiveBlend(&self) -> D2D1_PRIMITIVE_BLEND {
        D2D1_PRIMITIVE_BLEND(self.inner.borrow().dc.get_primitive_blend() as i32)
    }
    fn SetUnitMode(&self, unit: D2D1_UNIT_MODE) {
        self.inner.borrow_mut().dc.set_unit_mode(unit.0 as u32);
    }
    fn GetUnitMode(&self) -> D2D1_UNIT_MODE {
        D2D1_UNIT_MODE(self.inner.borrow().dc.get_unit_mode() as i32)
    }
    fn DrawGlyphRun(&self, o: &Vector2, g: *const DWRITE_GLYPH_RUN, _d: *const DWRITE_GLYPH_RUN_DESCRIPTION, b: Ref<ID2D1Brush>, m: DWRITE_MEASURING_MODE) {
        self.draw_glyph_run(o, g, b, m, None);
    }
    fn DrawImage(&self, image: Ref<ID2D1Image>, targetoffset: *const Vector2, rect: *const D2D_RECT_F, interp: D2D1_INTERPOLATION_MODE, composite: D2D1_COMPOSITE_MODE) {
        let offset = if targetoffset.is_null() { [0.0, 0.0] } else { vec2(unsafe { &*targetoffset }) };
        let rect = (!rect.is_null()).then(|| rectf(unsafe { &*rect }));
        self.draw_image(image, offset, rect, interp.0, composite.0);
    }
    fn DrawGdiMetafile(&self, g: Ref<ID2D1GdiMetafile>, o: *const Vector2) {
        // DrawGdiMetafile(metafile, targetOffset): draw at its natural bounds size, offset in DIPs.
        let Some(m) = g.as_ref().and_then(|g| g.cast_object_ref::<MetafileObj>().ok()) else { return self.latch_invalid() };
        let off = if o.is_null() { [0.0, 0.0] } else { vec2(unsafe { &*o }) };
        let b = m.mf.bounds();
        let dest = [off[0] + b[0], off[1] + b[1], off[0] + b[2], off[1] + b[3]];
        self.inner.borrow_mut().dc.draw_gdi_metafile(&m.mf, dest, None);
    }
    fn DrawBitmap(&self, bitmap: Ref<ID2D1Bitmap>, dest: *const D2D_RECT_F, opacity: f32, interp: D2D1_INTERPOLATION_MODE, source: *const D2D_RECT_F, _persp: *const windows_numerics::Matrix4x4) {
        self.draw_bitmap(bitmap, dest, opacity, interp.0 as u32, source);
    }
    fn PushLayer(&self, l: *const D2D1_LAYER_PARAMETERS1, _layer: Ref<ID2D1Layer>) {
        let p = unsafe { &*l };
        self.push_layer(p.contentBounds, &p.geometricMask, p.maskAntialiasMode, &p.maskTransform, p.opacity, &p.opacityBrush, p.layerOptions.0 as u32);
    }
    fn InvalidateEffectInputRectangle(&self, e: Ref<ID2D1Effect>, _i: u32, _r: *const D2D_RECT_F) -> WResult<()> {
        // The CPU core keeps no render cache, so there is nothing to invalidate incrementally — the
        // effect's whole output stays "invalid" until it is realized (see `GetEffectInvalidRectangles`).
        e.as_ref().and_then(|e| e.cast_object_ref::<EffectObj>().ok()).ok_or(windows_core::Error::from(E_NOTIMPL))?;
        Ok(())
    }
    fn GetEffectInvalidRectangleCount(&self, e: Ref<ID2D1Effect>) -> WResult<u32> {
        e.as_ref().and_then(|e| e.cast_object_ref::<EffectObj>().ok()).ok_or(windows_core::Error::from(E_NOTIMPL))?;
        Ok(1) // a never-realized effect has its entire output invalid (native returns 1 infinite rect)
    }
    fn GetEffectInvalidRectangles(&self, e: Ref<ID2D1Effect>, r: *mut D2D_RECT_F, c: u32) -> WResult<()> {
        e.as_ref().and_then(|e| e.cast_object_ref::<EffectObj>().ok()).ok_or(windows_core::Error::from(E_NOTIMPL))?;
        // The whole (infinite) output is invalid — the CPU core has no render cache to track sub-regions.
        if c >= 1 {
            unsafe { *r = D2D_RECT_F { left: f32::NEG_INFINITY, top: f32::NEG_INFINITY, right: f32::INFINITY, bottom: f32::INFINITY } };
        }
        Ok(())
    }
    fn GetEffectRequiredInputRectangles(&self, e: Ref<ID2D1Effect>, r: *const D2D_RECT_F, d: *const D2D1_EFFECT_INPUT_DESCRIPTION, ri: *mut D2D_RECT_F, c: u32) -> WResult<()> {
        let obj = e.as_ref().and_then(|e| e.cast_object_ref::<EffectObj>().ok()).ok_or(windows_core::Error::from(E_NOTIMPL))?;
        // The render output rect (null ⇒ the effect's full output), and the requested inputs.
        let render = if r.is_null() { obj.effect.intrinsic_bounds().unwrap_or([0.0; 4]) } else { rectf(unsafe { &*r }) };
        let reqs: Vec<(d2d::effects::Effect, u32)> = (0..c as isize)
            .map(|k| {
                let desc = unsafe { &*d.offset(k) };
                let eff = desc.effect.as_ref().and_then(|e| e.cast_object_ref::<EffectObj>().ok()).map(|o| o.effect.clone());
                // An unrecoverable effect gets an impossible slot so it matches nothing (empty rect).
                eff.map(|ef| (ef, desc.inputIndex)).unwrap_or((obj.effect.clone(), u32::MAX))
            })
            .collect();
        let dpi = self.inner.borrow().dc.get_dpi();
        let rects = obj.effect.required_input_rectangles(render, &reqs, dpi);
        for (i, b) in rects.iter().take(c as usize).enumerate() {
            unsafe { *ri.add(i) = D2D_RECT_F { left: b[0], top: b[1], right: b[2], bottom: b[3] } };
        }
        Ok(())
    }
    fn FillOpacityMask(&self, m: Ref<ID2D1Bitmap>, b: Ref<ID2D1Brush>, d: *const D2D_RECT_F, s: *const D2D_RECT_F) {
        // The DeviceContext form omits `content`; native treats the mask as alpha coverage (GRAPHICS).
        self.fill_opacity_mask(m, b, 0, d, s);
    }
}

/// Read an `IStream` to end into a byte vector (used for the SVG XML the harness streams in, and the
/// effect-registration XML for `RegisterEffectFromStream`).
pub(crate) fn read_stream(s: &IStream) -> Vec<u8> {
    let mut out = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let mut read = 0u32;
        let hr = unsafe { s.Read(chunk.as_mut_ptr().cast(), chunk.len() as u32, Some(&mut read)) };
        if read == 0 || hr.is_err() {
            break;
        }
        out.extend_from_slice(&chunk[..read as usize]);
    }
    out
}

/// The geometry an empty realization holds: no figures, so it draws nothing.
fn empty_geometry() -> d2d::geometry::Geometry {
    d2d::geometry::Geometry::Path { fill: d2d::geometry::FillMode::Alternate, figures: Vec::new() }
}

// Geometry realizations (DC1) are not implemented — no passing scene uses them.
impl ID2D1DeviceContext1_Impl for ContextObj_Impl {
    fn CreateFilledGeometryRealization(&self, g: Ref<ID2D1Geometry>, t: f32) -> WResult<ID2D1GeometryRealization> {
        // A geometry from another factory is WRONG_FACTORY, checked before the geometry is used.
        let gobj = g.as_ref().and_then(|g| g.cast_object_ref::<GeometryObj>().ok()).ok_or(windows_core::Error::from(E_NOTIMPL))?;
        self.require_factory(&gobj.owning_factory())?;
        let geom = gobj.geometry();
        if d2d::geometry::realization::realization_clips_empty(&geom, None) {
            return Ok(RealizationObj { geom: empty_geometry(), stroke: None, tolerance: t, factory: Some(self.factory_iface()) }.into());
        }
        if geom.realize_bad_number(t) {
            return Err(windows_core::Error::from(windows_core::HRESULT(0x8899_0011u32 as i32)));
        }
        Ok(RealizationObj { geom: geom.realized(), stroke: None, tolerance: t, factory: Some(self.factory_iface()) }.into())
    }
    fn CreateStrokedGeometryRealization(&self, g: Ref<ID2D1Geometry>, t: f32, w: f32, s: Ref<ID2D1StrokeStyle>) -> WResult<ID2D1GeometryRealization> {
        // A geometry from another factory is WRONG_FACTORY, checked before the geometry is used.
        let gobj = g.as_ref().and_then(|g| g.cast_object_ref::<GeometryObj>().ok()).ok_or(windows_core::Error::from(E_NOTIMPL))?;
        self.require_factory(&gobj.owning_factory())?;
        let geom = gobj.geometry();
        // A stroke style from another factory is WRONG_FACTORY, checked before the style is used.
        let st_obj = s.as_ref().and_then(|s| s.cast_object_ref::<StrokeStyleObj>().ok());
        if let Some(o) = st_obj.as_ref() {
            self.require_factory(&o.factory)?;
        }
        let st = st_obj.as_ref().map(|o| o.style.clone()).unwrap_or_default();
        // A NaN width bakes an empty realization before the geometry is looked at: S_OK even for a geometry
        // whose width-1 realization is BAD_NUMBER (r17 22090: the same rounded rect at width 0.99 is BAD).
        if w.is_nan() {
            return Ok(RealizationObj { geom, stroke: Some((0.0, st)), tolerance: t, factory: Some(self.factory_iface()) }.into());
        }
        if d2d::geometry::realization::realization_clips_empty(&geom, Some((w, &st))) {
            return Ok(RealizationObj { geom: empty_geometry(), stroke: Some((0.0, st)), tolerance: t, factory: Some(self.factory_iface()) }.into());
        }
        if geom.stroked_realize_bad_number(w, &st, t) {
            return Err(windows_core::Error::from(windows_core::HRESULT(0x8899_0011u32 as i32)));
        }
        // A pen that voids at create time (DIP space, this tolerance) bakes an empty realization, kept as
        // a zero width that strokes nothing (`d2d::geometry::stroked_realization_voids`).
        let w = if d2d::geometry::stroked_realization_voids(&geom, w, &st, t) { 0.0 } else { w };
        Ok(RealizationObj { geom: geom.realized(), stroke: Some((w, st)), tolerance: t, factory: Some(self.factory_iface()) }.into())
    }
    fn DrawGeometryRealization(&self, g: Ref<ID2D1GeometryRealization>, b: Ref<ID2D1Brush>) {
        let Some(r) = g.as_ref().and_then(|g| g.cast_object_ref::<RealizationObj>().ok()) else { return self.latch_invalid() };
        let Some(b) = b.as_ref() else { return self.latch_invalid() };
        // A realization from another factory is WRONG_FACTORY.
        if !self.check_factory(&r.factory) {
            return;
        }
        // A filled realization bakes native's feather mesh; a stroked one re-rasters. The core's
        // `draw_geometry_realization` does both — not the ordinary fill/stroke AA path. A gradient/
        // bitmap/image brush can't ride the baked flat-colour mesh, so it re-rasters (native's
        // intermediate path) via `draw_geometry_realization_brush`.
        let mut ctx = self.inner.borrow_mut();
        if let Ok(sb) = b.cast_object_ref::<SolidBrushObj>() {
            let live = sb.live(&ctx.dc);
            ctx.dc.draw_geometry_realization(&r.geom, r.stroke.as_ref(), r.tolerance, &live);
        } else if let Ok(pb) = b.cast_object_ref::<PixelBrushObj>() {
            let live = pb.live(&mut ctx.dc);
            ctx.dc.draw_geometry_realization_brush(&r.geom, r.stroke.as_ref(), r.tolerance, &live);
        }
    }
}

impl ID2D1DeviceContext2_Impl for ContextObj_Impl {
    fn CreateImageSourceFromWic(&self, wic: Ref<windows::Win32::Graphics::Imaging::IWICBitmapSource>, _l: D2D1_IMAGE_SOURCE_LOADING_OPTIONS, _a: D2D1_ALPHA_MODE) -> WResult<ID2D1ImageSourceFromWic> {
        let src = wic.as_ref().ok_or(windows_core::Error::from(E_NOTIMPL))?;
        let bitmap = bitmap_from_wic(src)?;
        let source = self.inner.borrow().dc.create_image_source_from_wic(bitmap);
        Ok(ImageSrcObj { source, factory: Some(self.factory_iface()), wic_source: wic.as_ref().cloned(), image_source: None, props: None }.into())
    }
    fn CreateTransformedImageSource(&self, imagesource: Ref<ID2D1ImageSource>, props: *const D2D1_TRANSFORMED_IMAGE_SOURCE_PROPERTIES) -> WResult<ID2D1TransformedImageSource> {
        let src = imagesource.as_ref().ok_or(windows_core::Error::from(E_NOTIMPL))?.cast_object_ref::<ImageSrcObj>()?;
        let p = unsafe { &*props };
        let source = self.inner.borrow().dc.create_transformed_image_source(&src.source, p.orientation.0 as u32, [p.scaleX, p.scaleY], p.interpolationMode.0 as u32);
        Ok(ImageSrcObj { source, factory: Some(self.factory_iface()), wic_source: None, image_source: imagesource.as_ref().cloned(), props: Some(*p) }.into())
    }
    fn CreateInk(&self, s: *const D2D1_INK_POINT) -> WResult<ID2D1Ink> {
        let p = unsafe { &*s };
        let ink = self.inner.borrow().dc.create_ink([p.x, p.y, p.radius]);
        Ok(InkObj { ink: RefCell::new(ink), factory: Some(self.factory_iface()) }.into())
    }
    fn CreateInkStyle(&self, p: *const D2D1_INK_STYLE_PROPERTIES) -> WResult<ID2D1InkStyle> {
        let (shape, transform) = if p.is_null() {
            (0, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        } else {
            let props = unsafe { &*p };
            (props.nibShape.0 as u32, mat(&props.nibTransform))
        };
        let style = self.inner.borrow().dc.create_ink_style(shape, transform);
        Ok(InkStyleObj { style: RefCell::new(style), factory: Some(self.factory_iface()) }.into())
    }
    fn CreateGradientMesh(&self, patches: *const D2D1_GRADIENT_MESH_PATCH, count: u32) -> WResult<ID2D1GradientMesh> {
        let patches = (0..count as isize).map(|i| mesh_patch(unsafe { &*patches.offset(i) })).collect();
        let mesh = self.inner.borrow().dc.create_gradient_mesh(patches);
        Ok(GradientMeshObj { mesh, factory: Some(self.factory_iface()) }.into())
    }
    fn CreateLookupTable3D(&self, p: D2D1_BUFFER_PRECISION, e: *const u32, d: *const u8, _dc: u32, s: *const u32) -> WResult<ID2D1LookupTable3D> {
        // Decode the native strided buffer (R fastest, RGBA) into the core's flat `[f32; 4]` entries,
        // honouring the row/slice strides and the table's `precision`. Each channel is `bpc` bytes;
        // native holds the LUT at this precision and interpolates the decoded values, so the decode must
        // match (VM probe, lut-probe): UNORM8 = byte·257/65536 (WARP's `unorm_to_f32`, NOT byte/255 — the
        // 0.9999847 factor compounds through the trilinear), UNORM16 = u16 LE/65535 (lossless vs float),
        // else raw 32BPC_FLOAT (16 B/entry).
        let ext = unsafe { core::slice::from_raw_parts(e, 3) };
        let strides = unsafe { core::slice::from_raw_parts(s, 2) };
        let n = [ext[0] as usize, ext[1] as usize, ext[2] as usize];
        let (bpc, decode): (usize, fn(*const u8) -> f32) = match p.0 {
            1 => (1, |q| unsafe { *q } as f32 * (257.0 / 65536.0)),
            3 => (2, |q| u16::from_le_bytes(unsafe { [*q, *q.add(1)] }) as f32 / 65535.0),
            _ => (4, |q| f32::from_le_bytes(unsafe { [*q, *q.add(1), *q.add(2), *q.add(3)] })),
        };
        let mut data = Vec::with_capacity(n[0] * n[1] * n[2]);
        for b in 0..n[2] {
            for g in 0..n[1] {
                let row = unsafe { d.add(b * strides[1] as usize + g * strides[0] as usize) };
                for r in 0..n[0] {
                    let px = unsafe { row.add(r * bpc * 4) };
                    data.push([decode(px), decode(unsafe { px.add(bpc) }), decode(unsafe { px.add(2 * bpc) }), decode(unsafe { px.add(3 * bpc) })]);
                }
            }
        }
        let lut = self.inner.borrow().dc.create_lookup_table3d([ext[0], ext[1], ext[2]], data, p.0 as u32).map_err(windows_core::Error::from)?;
        Ok(LutObj { lut, factory: Some(self.factory_iface()) }.into())
    }
    fn CreateImageSourceFromDxgi(&self, s: *const Option<windows::Win32::Graphics::Dxgi::IDXGISurface>, sc: u32, _cs: windows::Win32::Graphics::Dxgi::Common::DXGI_COLOR_SPACE_TYPE, _o: D2D1_IMAGE_SOURCE_FROM_DXGI_OPTIONS) -> WResult<ID2D1ImageSource> {
        // Only LOW_QUALITY_PRIMARY_CONVERSION (bit 0) is a legal option; native rejects the rest with
        // E_INVALIDARG (Direct2D rejects when `(opts & 0xFFFFFFFE) != 0`).
        if _o.0 & !1 != 0 {
            return Err(windows_core::HRESULT(0x8007_0057u32 as i32).into());
        }
        // A DXGI surface only exists on Windows; elsewhere there is nothing to read.
        #[cfg(not(windows))]
        return {
            let _ = (s, sc, _cs);
            Err(E_NOTIMPL.into())
        };
        // Read the first (BGRA) plane's pixels off the surface into a core bitmap and wrap it as an
        // image source, the same object `CreateImageSourceFromWic` produces. Multi-plane (YUV) sources
        // (sc > 1) are not reproduced by the CPU core.
        #[cfg(windows)]
        {
            let surface = (sc >= 1).then(|| unsafe { &*s }).and_then(|o| o.as_ref()).ok_or(windows_core::Error::from(E_NOTIMPL))?;
            let bitmap = super::present::read_dxgi(surface)?;
            let source = self.inner.borrow().dc.create_image_source_from_wic(bitmap);
            let obj: ID2D1ImageSourceFromWic = ImageSrcObj { source, factory: Some(self.factory_iface()), wic_source: None, image_source: None, props: None }.into();
            obj.cast()
        }
    }
    fn GetGradientMeshWorldBounds(&self, m: Ref<ID2D1GradientMesh>) -> WResult<D2D_RECT_F> {
        let obj = m.as_ref().ok_or(windows_core::Error::from(super::E_INVALIDARG))?.cast_object_ref::<GradientMeshObj>()?;
        let [l, t, r, b] = self.inner.borrow().dc.gradient_mesh_world_bounds(&obj.mesh);
        Ok(D2D_RECT_F { left: l, top: t, right: r, bottom: b })
    }
    fn DrawInk(&self, i: Ref<ID2D1Ink>, b: Ref<ID2D1Brush>, s: Ref<ID2D1InkStyle>) {
        let Some(ink) = i.as_ref().and_then(|i| i.cast_object_ref::<InkObj>().ok()) else { return self.latch_invalid() };
        let Some(b) = b.as_ref() else { return self.latch_invalid() };
        // An ink from another factory is WRONG_FACTORY.
        if !self.check_factory(&ink.factory) {
            return;
        }
        let so = s.as_ref().and_then(|s| s.cast_object_ref::<InkStyleObj>().ok());
        // An ink style from another factory is WRONG_FACTORY.
        if !self.check_factory(&so.as_ref().and_then(|o| o.factory.clone())) {
            return;
        }
        let style = so.as_ref().map(|o| *o.style.borrow()); // InkStyle is Copy; snapshot out of the cell
        let mut ctx = self.inner.borrow_mut();
        // A gradient/bitmap/image brush can't use the ISM fast path's per-colour gamma LUT, so it
        // takes the AsGeometry path (`draw_ink_brush`) — the same outline fill with a per-pixel source.
        if let Ok(sb) = b.cast_object_ref::<SolidBrushObj>() {
            let live = sb.live(&ctx.dc);
            ctx.dc.draw_ink(&ink.ink.borrow(), &live, style.as_ref());
        } else if let Ok(pb) = b.cast_object_ref::<PixelBrushObj>() {
            let live = pb.live(&mut ctx.dc);
            ctx.dc.draw_ink_brush(&ink.ink.borrow(), &live, style.as_ref());
        }
    }
    fn DrawGradientMesh(&self, mesh: Ref<ID2D1GradientMesh>) {
        if let Some(m) = mesh.as_ref().and_then(|m| m.cast_object_ref::<GradientMeshObj>().ok()) {
            // A gradient mesh from another factory is WRONG_FACTORY.
            if !self.check_factory(&m.factory) {
                return;
            }
            self.inner.borrow_mut().dc.draw_gradient_mesh(&m.mesh);
        }
    }
    fn DrawGdiMetafile(&self, g: Ref<ID2D1GdiMetafile>, d: *const D2D_RECT_F, s: *const D2D_RECT_F) {
        // DeviceContext2 overload: explicit destination (null ⇒ natural bounds) and source sub-rect.
        let Some(m) = g.as_ref().and_then(|g| g.cast_object_ref::<MetafileObj>().ok()) else { return self.latch_invalid() };
        let dest = if d.is_null() { m.mf.bounds() } else { rectf(unsafe { &*d }) };
        let src = (!s.is_null()).then(|| rectf(unsafe { &*s }));
        self.inner.borrow_mut().dc.draw_gdi_metafile(&m.mf, dest, src);
    }
}

impl ID2D1DeviceContext3_Impl for ContextObj_Impl {
    fn CreateSpriteBatch(&self) -> WResult<ID2D1SpriteBatch> {
        Ok(SpriteBatchObj { sprites: RefCell::new(Vec::new()), factory: Some(self.factory_iface()) }.into())
    }
    fn DrawSpriteBatch(&self, batch: Ref<ID2D1SpriteBatch>, _start: u32, _count: u32, bitmap: Ref<ID2D1Bitmap>, interp: D2D1_BITMAP_INTERPOLATION_MODE, _opts: D2D1_SPRITE_OPTIONS) {
        // Native requires aliased antialias mode: DrawSpriteBatch under per-primitive latches
        // D2DERR_WRONG_STATE and draws nothing, matching Direct2D's DrawSpriteBatch.
        if !self.inner.borrow_mut().dc.aliased_draw_check() {
            return;
        }
        let Some(sb) = batch.as_ref().and_then(|b| b.cast_object_ref::<SpriteBatchObj>().ok()) else { return self.latch_invalid() };
        // A sprite batch or its bitmap from another factory is WRONG_FACTORY (another device, same
        // factory → WRONG_RESOURCE_DOMAIN, as DrawBitmap), after the state/aliased check above.
        if !self.check_factory(&sb.factory) {
            return;
        }
        let Some(bmp) = bitmap.as_ref().and_then(|b| b.cast_object_ref::<BitmapObj>().ok()) else { return self.latch_invalid() };
        match bmp.domain_ids {
            Some((f, d)) => {
                if !self.inner.borrow_mut().dc.check_draw_domain(f, d) {
                    return;
                }
            }
            None => {
                if !self.check_factory(&bmp.factory) {
                    return;
                }
            }
        }
        let Some(src) = bmp.src_clone() else { return self.latch_invalid() };
        let sprites = sb.sprites.borrow().clone();
        let mut ctx = self.inner.borrow_mut();
        let core = ctx.dc.create_sprite_batch(sprites);
        ctx.dc.draw_sprite_batch(&core, &src, interp.0 as u32, 0);
    }
}

// Text / color-glyph methods (DC4) are not implemented — no passing scene reaches them.
impl ID2D1DeviceContext4_Impl for ContextObj_Impl {
    fn CreateSvgGlyphStyle(&self) -> WResult<ID2D1SvgGlyphStyle> {
        Ok(super::resources::SvgGlyphStyleObj::new(self.factory_iface()).into())
    }
    fn DrawText(&self, s: &windows_core::PCWSTR, l: u32, f: Ref<IDWriteTextFormat>, r: *const D2D_RECT_F, b: Ref<ID2D1Brush>, _g: Ref<ID2D1SvgGlyphStyle>, _c: u32, o: D2D1_DRAW_TEXT_OPTIONS, m: DWRITE_MEASURING_MODE) {
        // The DC4 overload adds an SVG glyph style (for OT-SVG fonts); for a normal run it draws the
        // same text as the base DrawText, so share the helper (SVG-in-glyph styling is a follow-up).
        self.draw_text(s, l, f, r, b, o, m);
    }
    fn DrawTextLayout(&self, o: &Vector2, l: Ref<IDWriteTextLayout>, b: Ref<ID2D1Brush>, _g: Ref<ID2D1SvgGlyphStyle>, _c: u32, opt: D2D1_DRAW_TEXT_OPTIONS) {
        if let Ok(rt) = self.to_interface::<ID2D1DeviceContext7>().cast::<ID2D1RenderTarget>() {
            self.draw_text_layout(rt, o, l, b, opt);
        }
    }
    // `DrawColorBitmapGlyphRun`: the CBDT/sbix (and raw PBGRA) sub-run `TranslateColorGlyphRun` splits
    // out — a strike bitmap per glyph, selected and placed by `d2d::text::color::bitmap` (matching
    // Direct2D's `GetColorBitmapGlyphImage`). `snap` (origin-snap) is not acted on yet; no scene exercises it.
    fn DrawColorBitmapGlyphRun(&self, f: DWRITE_GLYPH_IMAGE_FORMATS, o: &Vector2, g: *const DWRITE_GLYPH_RUN, m: DWRITE_MEASURING_MODE, _s: D2D1_COLOR_BITMAP_GLYPH_SNAP_OPTION) {
        let run = unsafe { &*g };
        if run.fontFace.is_none() {
            return self.latch_invalid();
        }
        if run.fontEmSize * self.inner.borrow().dc.get_dpi().1 / 96.0 >= 32768.0 {
            return self.latch_invalid();
        }
        let Some((font_bytes, idx)) = run.fontFace.as_ref().and_then(run_face_bytes) else { return self.latch_invalid() };
        let bytes = font_bytes.as_slice();
        let n = run.glyphCount as usize;
        let ids = unsafe { core::slice::from_raw_parts(run.glyphIndices, n) };
        let adv = unsafe { core::slice::from_raw_parts(run.glyphAdvances, n) };
        let off = unsafe { core::slice::from_raw_parts(run.glyphOffsets, n) };
        let glyphs: Vec<Glyph> =
            (0..n).map(|i| Glyph { id: ids[i], advance: adv[i], offset: [off[i].advanceOffset, off[i].ascenderOffset] }).collect();
        let gr = GlyphRun {
            font_data: bytes,
            face_index: idx,
            em_size: run.fontEmSize,
            glyphs: &glyphs,
            baseline_origin: [o.X, o.Y],
            measuring_mode: m.0 as u32,
            is_sideways: run.isSideways.as_bool(),
            bidi_level: run.bidiLevel,
            no_snap: true,
            simulations: 0,
            coords: &[],
        };
        self.guard_draw(|| {
            let mut ctx = self.inner.borrow_mut();
            let params = ctx.text_params;
            ctx.dc.draw_color_bitmap_glyph_run(&gr, f.0 as u32, &params);
        });
    }
    // `DrawSvgGlyphRun`: the OT-SVG sub-run `TranslateColorGlyphRun` splits out — each glyph's `SVG `
    // document rendered through our SVG renderer under the native glyph transform (`d2d::text`:
    // `draw_svg_glyph_run`). `b` is the context foreground (`context-fill`); the SVG glyph `style`
    // (per-paint overrides) is not acted on yet — no scene exercises it.
    fn DrawSvgGlyphRun(&self, o: &Vector2, g: *const DWRITE_GLYPH_RUN, b: Ref<ID2D1Brush>, _st: Ref<ID2D1SvgGlyphStyle>, _c: u32, m: DWRITE_MEASURING_MODE) {
        let run = unsafe { &*g };
        if run.fontFace.is_none() {
            return self.latch_invalid();
        }
        if run.fontEmSize * self.inner.borrow().dc.get_dpi().1 / 96.0 >= 32768.0 {
            return self.latch_invalid();
        }
        let Some((font_bytes, idx)) = run.fontFace.as_ref().and_then(run_face_bytes) else { return self.latch_invalid() };
        let bytes = font_bytes.as_slice();
        let n = run.glyphCount as usize;
        let ids = unsafe { core::slice::from_raw_parts(run.glyphIndices, n) };
        let adv = unsafe { core::slice::from_raw_parts(run.glyphAdvances, n) };
        let off = unsafe { core::slice::from_raw_parts(run.glyphOffsets, n) };
        let glyphs: Vec<Glyph> =
            (0..n).map(|i| Glyph { id: ids[i], advance: adv[i], offset: [off[i].advanceOffset, off[i].ascenderOffset] }).collect();
        // Context foreground for `context-fill`: the solid brush's straight colour × its opacity.
        let color = b
            .as_ref()
            .and_then(|br| br.cast_object_ref::<SolidBrushObj>().ok())
            .map(|s| {
                let (c, o) = s.current();
                [c.r, c.g, c.b, c.a * o]
            })
            .unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let gr = GlyphRun {
            font_data: bytes,
            face_index: idx,
            em_size: run.fontEmSize,
            glyphs: &glyphs,
            baseline_origin: [o.X, o.Y],
            measuring_mode: m.0 as u32,
            is_sideways: run.isSideways.as_bool(),
            bidi_level: run.bidiLevel,
            no_snap: true,
            simulations: 0,
            coords: &[],
        };
        self.guard_draw(|| {
            let mut ctx = self.inner.borrow_mut();
            let params = ctx.text_params;
            ctx.dc.draw_svg_glyph_run(&gr, color, &params);
        });
    }
    fn GetColorBitmapGlyphImage(&self, f: DWRITE_GLYPH_IMAGE_FORMATS, o: &Vector2, ff: Ref<IDWriteFontFace>, em: f32, gi: u16, sw: BOOL, w: *const Matrix3x2, dx: f32, dy: f32, gt: *mut Matrix3x2, img: windows_core::OutRef<ID2D1Image>) -> WResult<()> {
        // Direct2D's GetColorBitmapGlyphImage: only the colour-bitmap formats
        // (PNG 16 / JPEG 32 / TIFF 64 / PBGRA 128) are valid here (SVG goes through GetSvgGlyphImage).
        let built = (|| -> WResult<(ID2D1Image, [f32; 6])> {
            if !matches!(f.0, 16 | 32 | 64 | 128) {
                return Err(super::E_INVALIDARG.into());
            }
            let face = ff.as_ref().ok_or(windows_core::Error::from(super::E_INVALIDARG))?;
            let (font_bytes, idx) = run_face_bytes(face).ok_or(windows_core::Error::from(super::E_INVALIDARG))?;
            let sims = unsafe { face.GetSimulations() }.0 as u16;
            // A NULL world transform is the identity (native reads the identity constant).
            let world = if w.is_null() {
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            } else {
                let m = unsafe { &*w };
                [m.M11, m.M12, m.M21, m.M22, m.M31, m.M32]
            };
            let (bitmap, t) = d2d::text::color::bitmap::color_bitmap_glyph_image(font_bytes.as_slice(), idx, gi, em, sims, world, (dx, dy), [o.X, o.Y], sw.as_bool())
                .ok_or(windows_core::Error::from(super::E_INVALIDARG))?;
            let size = D2D_SIZE_U { width: bitmap.width, height: bitmap.height };
            let domain_ids = Some(self.inner.borrow().dc.domain_ids());
            let obj: ID2D1Bitmap1 = BitmapObj {
                pixels: size, dpi: (96.0, 96.0), buf: RefCell::new(None), src: RefCell::new(Some(bitmap)), options: D2D1_BITMAP_OPTIONS(0),
                present: RefCell::new(None), factory: Some(self.factory_iface()), domain_ids, color_context: None,
            }
            .into();
            Ok((obj.cast()?, t))
        })();
        match built {
            Ok((image, t)) => {
                if !gt.is_null() {
                    unsafe { *gt = Matrix3x2 { M11: t[0], M12: t[1], M21: t[2], M22: t[3], M31: t[4], M32: t[5] } };
                }
                let _ = img.write(Some(image));
                Ok(())
            }
            Err(e) => {
                let _ = img.write(None);
                Err(e)
            }
        }
    }
    fn GetSvgGlyphImage(&self, o: &Vector2, ff: Ref<IDWriteFontFace>, em: f32, gi: u16, sw: BOOL, w: *const Matrix3x2, b: Ref<ID2D1Brush>, st: Ref<ID2D1SvgGlyphStyle>, c: u32, gt: *mut Matrix3x2, img: windows_core::OutRef<ID2D1CommandList>) -> WResult<()> {
        // Direct2D's GetSvgGlyphImage records the OT-SVG glyph into a command
        // list at a canonical em-units scale and returns a glyph→world transform that scales it. Our
        // command-list replay is translate-only (device-resolved primitives), so we instead bake the
        // placed glyph into the list at the baseline origin and return the caller's world transform; a
        // direct SetTransform(gt)+DrawImage(list) reproduces DrawSvgGlyphRun byte-for-byte. ponytail:
        // re-scaling the returned list past its baked placement is not supported (the command-list
        // model stores device-resolved primitives), and the returned transform is our baked-list
        // convention, not native's canonical-list scale.
        let _ = (st, c, sw);
        let built = (|| -> WResult<(ID2D1CommandList, [f32; 6])> {
            let face = ff.as_ref().ok_or(windows_core::Error::from(super::E_INVALIDARG))?;
            let (font_bytes, idx) = run_face_bytes(face).ok_or(windows_core::Error::from(super::E_INVALIDARG))?;
            let bytes = font_bytes.as_slice();
            let world = if w.is_null() {
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            } else {
                let m = unsafe { &*w };
                [m.M11, m.M12, m.M21, m.M22, m.M31, m.M32]
            };
            // Context foreground for `context-fill` (as DrawSvgGlyphRun).
            let color = b
                .as_ref()
                .and_then(|br| br.cast_object_ref::<SolidBrushObj>().ok())
                .map(|s| {
                    let (cc, o) = s.current();
                    [cc.r, cc.g, cc.b, cc.a * o]
                })
                .unwrap_or([0.0, 0.0, 0.0, 1.0]);
            let glyphs = [Glyph { id: gi, advance: 0.0, offset: [0.0, 0.0] }];
            let gr = GlyphRun {
                font_data: bytes,
                face_index: idx,
                em_size: em,
                glyphs: &glyphs,
                baseline_origin: [o.X, o.Y],
                measuring_mode: 0,
                is_sideways: false,
                bidi_level: 0,
                no_snap: true,
                simulations: 0,
                coords: &[],
            };
            // Record the placed SVG glyph into a fresh command list (identity transform, so the list is
            // in DIP space at the baseline origin), then hand it back to draw under `world`.
            let mut ctx = self.inner.borrow_mut();
            let list = ctx.dc.create_command_list();
            let saved = ctx.dc.get_transform();
            ctx.dc.set_transform([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
            ctx.dc.record_into(&list);
            let params = ctx.text_params;
            let ok = ctx.dc.draw_svg_glyph_run(&gr, color, &params);
            ctx.dc.stop_recording();
            ctx.dc.set_transform(saved);
            if !ok {
                return Err(super::E_INVALIDARG.into());
            }
            list.close();
            Ok((CommandListObj { list, factory: Some(self.factory_iface()), pending_err: None }.into(), world))
        })();
        match built {
            Ok((list, t)) => {
                if !gt.is_null() {
                    unsafe { *gt = Matrix3x2 { M11: t[0], M12: t[1], M21: t[2], M22: t[3], M31: t[4], M32: t[5] } };
                }
                let _ = img.write(Some(list));
                Ok(())
            }
            Err(e) => {
                let _ = img.write(None);
                Err(e)
            }
        }
    }
}

impl ID2D1DeviceContext5_Impl for ContextObj_Impl {
    fn CreateSvgDocument(&self, xml: Ref<IStream>, viewport: &D2D_SIZE_F) -> WResult<ID2D1SvgDocument> {
        let bytes = xml.as_ref().map(read_stream).unwrap_or_default();
        let xml = String::from_utf8_lossy(&bytes);
        let doc = d2d::svg::Document::parse(&xml, [viewport.width, viewport.height])
            .map_err(|c| windows_core::Error::from(windows_core::HRESULT(c)))?;
        Ok(SvgDocObj::new(doc, Some(self.factory_iface())).into())
    }
    fn DrawSvgDocument(&self, doc: Ref<ID2D1SvgDocument>) {
        let Some(d) = doc.as_ref().and_then(|d| d.cast_object_ref::<SvgDocObj>().ok()) else { return self.latch_invalid() };
        d.doc.borrow().render(&mut self.inner.borrow_mut().dc);
    }
    fn CreateColorContextFromDxgiColorSpace(&self, c: windows::Win32::Graphics::Dxgi::Common::DXGI_COLOR_SPACE_TYPE) -> WResult<ID2D1ColorContext1> {
        Ok(ColorContextObj { ctx: ColorContext::from_dxgi_color_space(d2d::abi::DXGI_COLOR_SPACE_TYPE(c.0)), factory: Some(self.factory_iface()) }.into())
    }
    fn CreateColorContextFromSimpleColorProfile(&self, p: *const D2D1_SIMPLE_COLOR_PROFILE) -> WResult<ID2D1ColorContext1> {
        let s = unsafe { &*p };
        let ctx = ColorContext::from_simple_color_profile(SimpleProfile {
            red: [s.redPrimary.X, s.redPrimary.Y],
            green: [s.greenPrimary.X, s.greenPrimary.Y],
            blue: [s.bluePrimary.X, s.bluePrimary.Y],
            white: [s.whitePointXZ.X, s.whitePointXZ.Y],
            gamma: s.gamma.0 as u32,
        });
        Ok(ColorContextObj { ctx, factory: Some(self.factory_iface()) }.into())
    }
}

impl ID2D1DeviceContext6_Impl for ContextObj_Impl {
    fn BlendImage(&self, image: Ref<ID2D1Image>, mode: D2D1_BLEND_MODE, targetoffset: *const Vector2, rect: *const D2D_RECT_F, interp: D2D1_INTERPOLATION_MODE) {
        let Some(DrawTarget::Bitmap(bm)) = image_target(image) else { return };
        let offset = if targetoffset.is_null() { [0.0, 0.0] } else { vec2(unsafe { &*targetoffset }) };
        let rect = (!rect.is_null()).then(|| rectf(unsafe { &*rect }));
        self.inner.borrow_mut().dc.blend_image(&bm, mode.0 as u32, offset, rect, interp.0 as u32);
    }
}

// Colour (COLRv1 paint) glyph runs: the core's `draw_color_glyph_run` paints the paint tree.
impl ID2D1DeviceContext7_Impl for ContextObj_Impl {
    fn GetPaintFeatureLevel(&self) -> DWRITE_PAINT_FEATURE_LEVEL {
        DWRITE_PAINT_FEATURE_LEVEL_COLR_V1
    }
    fn DrawPaintGlyphRun(&self, o: &Vector2, g: *const DWRITE_GLYPH_RUN, b: Ref<ID2D1Brush>, palette: u32, m: DWRITE_MEASURING_MODE) {
        self.draw_glyph_run(o, g, b, m, Some(palette));
    }
    fn DrawGlyphRunWithColorSupport(&self, o: &Vector2, g: *const DWRITE_GLYPH_RUN, _d: *const DWRITE_GLYPH_RUN_DESCRIPTION, b: Ref<ID2D1Brush>, _st: Ref<ID2D1SvgGlyphStyle>, palette: u32, m: DWRITE_MEASURING_MODE, _snap: D2D1_COLOR_BITMAP_GLYPH_SNAP_OPTION) {
        self.draw_glyph_run(o, g, b, m, Some(palette));
    }
}

// --- presenting render targets: Hwnd / DC / GDI-interop extras over the shared ID2D1RenderTarget ---
// Windows-only: HWND windows, GDI device contexts and the DC render target do not exist elsewhere, so
// off Windows the object simply does not expose these interfaces (QI returns E_NOINTERFACE).

#[cfg(windows)]
impl ID2D1HwndRenderTarget_Impl for ContextObj_Impl {
    fn CheckWindowState(&self) -> D2D1_WINDOW_STATE {
        // NONE. Native's OCCLUDED comes from its internal swap chain's present (DXGI_STATUS_OCCLUDED),
        // which a real compositor raises; captured on the VM, native returns NONE for shown, minimized
        // AND hidden windows (occlusion doesn't register without an interactive desktop), so NONE is
        // the headless-verifiable match.
        D2D1_WINDOW_STATE(0)
    }
    fn Resize(&self, pixelsize: *const D2D_SIZE_U) -> WResult<()> {
        // Reallocate the target at the new pixel size; GetPixelSize/GetSize then read it (GetSize in
        // DIPs = pixels·96/dpi). Native accepts a 0×0 resize with S_OK (verified) — bitmap_target(0,0)
        // is an empty buffer, so GetPixelSize becomes 0×0, matching.
        let sz = unsafe { &*pixelsize };
        let mut ctx = self.inner.borrow_mut();
        let dpi = ctx.dc.get_dpi().0;
        ctx.dc.set_target_bitmap(d2d::bitmap_target(sz.width, sz.height, dpi));
        Ok(())
    }
    fn GetHwnd(&self) -> HWND {
        self.inner
            .borrow()
            .target
            .as_ref()
            .and_then(|bm| match &*bm.present.borrow() {
                Some(Present::Gdi { hwnd: Some(h), .. }) => Some(*h),
                _ => None,
            })
            .unwrap_or_default()
    }
}

impl ID2D1BitmapRenderTarget_Impl for ContextObj_Impl {
    /// `ID2D1BitmapRenderTarget::GetBitmap` — the offscreen target this compatible render target draws
    /// into. After the compatible target's `EndDraw` its rendered pixels live on this `BitmapObj`'s
    /// buffer (published by `EndDraw`), so the parent can `DrawBitmap` it. The same object is returned
    /// on every call, as native's backing bitmap is.
    fn GetBitmap(&self) -> WResult<ID2D1Bitmap> {
        self.inner.borrow().target.as_ref().ok_or(windows_core::Error::from(E_NOTIMPL))?.cast()
    }
}

#[cfg(windows)]
impl ID2D1DCRenderTarget_Impl for ContextObj_Impl {
    fn BindDC(&self, hdc: HDC, psubrect: *const RECT) -> WResult<()> {
        let r = unsafe { &*psubrect };
        let (w, h) = ((r.right - r.left).max(0) as u32, (r.bottom - r.top).max(0) as u32);
        let mut ctx = self.inner.borrow_mut();
        let dpi = ctx.dc.get_dpi().0;
        ctx.dc.set_target_bitmap(d2d::bitmap_target(w, h, dpi));
        if let Some(bm) = ctx.target.as_ref() {
            *bm.present.borrow_mut() = Some(Present::Gdi { hwnd: None, hdc: Some(hdc), origin: (r.left, r.top) });
        }
        Ok(())
    }
}

#[cfg(windows)]
impl ID2D1GdiInteropRenderTarget_Impl for ContextObj_Impl {
    /// Hand the caller a DIB-backed memory DC mirroring the target (COPY) or cleared (CLEAR), for
    /// interleaved GDI drawing; `ReleaseDC` copies it back.
    fn GetDC(&self, mode: D2D1_DC_INITIALIZE_MODE) -> WResult<HDC> {
        let ctx = self.inner.borrow();
        let buf = ctx.dc.target_bitmap().ok_or(windows_core::Error::from(E_NOTIMPL))?;
        let bmi = super::present::bitmapinfo(buf.desc.width, buf.desc.height);
        let mut bits: *mut c_void = core::ptr::null_mut();
        let bmp = unsafe { CreateDIBSection(None, &bmi, DIB_RGB_COLORS, &mut bits, None, 0)? };
        if mode.0 == 0 {
            // COPY: seed the DIB with the current target pixels.
            unsafe { core::ptr::copy_nonoverlapping(buf.data.as_ptr(), bits.cast(), buf.data.len()) };
        }
        let dc = unsafe { CreateCompatibleDC(None) };
        let old = unsafe { SelectObject(dc, bmp.into()) };
        drop(ctx);
        *self.gdi.borrow_mut() = Some(GdiScratch { dc, bmp, old, bits: bits.cast() });
        Ok(dc)
    }
    fn ReleaseDC(&self, _update: *const RECT) -> WResult<()> {
        let Some(g) = self.gdi.borrow_mut().take() else { return Ok(()) };
        let mut ctx = self.inner.borrow_mut();
        if let Some(mut buf) = ctx.dc.take_target() {
            let n = buf.data.len();
            // Copy the DIB back verbatim — GDI leaves the alpha byte it wrote (FillRect zeroes it), and
            // native's ReleaseDC preserves exactly that (VM-verified: a GDI-filled texel reads back with
            // alpha 0, not forced opaque).
            unsafe { core::ptr::copy_nonoverlapping(g.bits, buf.data.as_mut_ptr(), n) };
            ctx.dc.set_target_bitmap(buf);
        }
        unsafe {
            SelectObject(g.dc, g.old);
            let _ = DeleteDC(g.dc);
            let _ = DeleteObject(g.bmp.into());
        }
        Ok(())
    }
}
