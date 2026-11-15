//! `GetGdiInterop` → `IDWriteGdiInterop`: the LOGFONT ⇄ font conversions and a GDI bitmap render
//! target. `CreateBitmapRenderTarget` makes a top-down 32-bit DIB behind a memory DC;
//! `IDWriteBitmapRenderTarget::DrawGlyphRun` rasterises a run into it through the shared
//! `d2d::text` core, so the bitmap carries the core's glyph coverage.
//!
//! `DrawGlyphRun` rasterises the run with native's own GDI DIB pipeline, ported fresh in
//! [`gdi_raster`] (`text::gdi_raster`) — a distinct rasteriser/blend from the Direct2D text path
//! (`text::render_glyph_run`/`text::blend`), reproduced from DirectWrite:
//!   * The target **defaults to ClearType** (the bitmap render target's constructor
//!     zeroes its AA field; `DWRITE_TEXT_ANTIALIAS_MODE_CLEARTYPE = 0`). With the pinned *flat* pixel
//!     geometry the three colour sub-pixels share coverage, so ClearType output is R=G=B — which earlier
//!     looked like grayscale but is `DrawClearType` (6×1), not `DrawGrayscale`.
//!   * ClearType (`gdi_raster::draw_glyph_run_gdi`, grayscale=false): 6×1 coverage, the GDI gamma +
//!     enhanced-contrast tables (`GammaTable`/`EnhancedContrastTable`), gamma-space blend, dst alpha
//!     preserved. Grayscale (an app `SetTextAntialiasMode(GRAYSCALE)`): 8×1 popcount coverage +
//!     `ComputeGrayscaleAlphaByBitCount` + straight-alpha `GrayscaleAlphaBlend`.
//!
//! The alpha curves + blends are byte-exact against native (verified via `dwrite-probe`'s
//! `GetBitmapData` on the VM). The residual is a ≤1-sub-sample coverage difference at hinted stem
//! edges — the TrueType hinting interpreter's sub-pixel stem placement (`text::hinting`, upstream),
//! the same floor that leaves scene-17 / text-01 off by a few edge pixels.

use super::fonts::{SystemFonts, font_iface};
use super::{E_INVALIDARG, ITextInterop};
use core::cell::RefCell;
use core::slice;
use d2d::text::{Glyph as TGlyph, GlyphRun as TRun, RenderingParams, gdi_raster, render_glyph_run};
use d2d_types::abi::{D2D1_ALPHA_MODE_IGNORE, DXGI_FORMAT_B8G8R8A8_UNORM};
use d2d_types::contracts::{ImageDesc, PixelBuffer};
use skrifa::attribute::{Attributes, Style};
use skrifa::string::StringId;
use skrifa::{FontRef, MetadataProvider};
use std::sync::Arc;
use windows::Win32::Foundation::{COLORREF, RECT};
use windows::Win32::Graphics::Gdi::*;
use windows::Win32::Graphics::DirectWrite::*;
use windows_core::{BOOL, Interface, Ref, Result as WResult, implement};

#[implement(IDWriteGdiInterop)]
pub(crate) struct GdiInteropObj {
    pub(crate) fonts: Arc<SystemFonts>,
}

/// Write `family`/`weight`/`italic` into a `LOGFONTW` (the fields GDI font selection keys on; the
/// rest are the neutral defaults native leaves — DEFAULT_CHARSET, zero sizes).
fn fill_logfont(out: *mut LOGFONTW, family: &str, weight: u16, italic: bool) {
    let lf = unsafe { &mut *out };
    *lf = LOGFONTW { lfWeight: weight as i32, lfItalic: italic as u8, lfCharSet: DEFAULT_CHARSET, ..Default::default() };
    for (d, u) in lf.lfFaceName.iter_mut().zip(family.encode_utf16().take(31)) {
        *d = u;
    }
}

/// The family name + weight + italic parsed from a face's bytes (via our `ITextInterop`).
fn face_fields(face: &IDWriteFontFace) -> Option<(String, u16, bool)> {
    let interop: ITextInterop = face.cast().ok()?;
    let (mut data, mut len, mut index) = (core::ptr::null(), 0usize, 0u32);
    if unsafe { interop.FontBlob(&mut data, &mut len, &mut index) }.is_err() || data.is_null() {
        return None;
    }
    let font = FontRef::from_index(unsafe { slice::from_raw_parts(data, len) }, index).ok()?;
    let fam = font.localized_strings(StringId::FAMILY_NAME).english_or_first()?.to_string();
    let attrs = Attributes::new(&font);
    Some((fam, attrs.weight.value() as u16, attrs.style != Style::Normal))
}

impl GdiInteropObj {
    fn face_from_logfont(&self, lf: &LOGFONTW) -> WResult<IDWriteFontFace> {
        let end = lf.lfFaceName.iter().position(|&c| c == 0).unwrap_or(32);
        let family = String::from_utf16_lossy(&lf.lfFaceName[..end]);
        let entry = self.fonts.find_match(&family, lf.lfWeight as u16, lf.lfItalic != 0).ok_or(E_INVALIDARG)?;
        // Build the same FaceObj the collection yields.
        let font = font_iface(self.fonts.clone(), entry)?;
        unsafe { font.CreateFontFace() }
    }
}

impl IDWriteGdiInterop_Impl for GdiInteropObj_Impl {
    fn CreateFontFromLOGFONT(&self, logfont: *const LOGFONTW) -> WResult<IDWriteFont> {
        let lf = unsafe { &*logfont };
        let end = lf.lfFaceName.iter().position(|&c| c == 0).unwrap_or(32);
        let family = String::from_utf16_lossy(&lf.lfFaceName[..end]);
        let entry = self.fonts.find_match(&family, lf.lfWeight as u16, lf.lfItalic != 0).ok_or(E_INVALIDARG)?;
        font_iface(self.fonts.clone(), entry)
    }
    fn ConvertFontToLOGFONT(&self, font: Ref<IDWriteFont>, logfont: *mut LOGFONTW, issystemfont: *mut BOOL) -> WResult<()> {
        let f = font.as_ref().ok_or(E_INVALIDARG)?;
        let weight = unsafe { f.GetWeight() }.0 as u16;
        let italic = unsafe { f.GetStyle() } != DWRITE_FONT_STYLE_NORMAL;
        let family = {
            let names = unsafe { f.GetFontFamily()?.GetFamilyNames()? };
            let mut b = [0u16; 32];
            let _ = unsafe { names.GetString(0, &mut b) };
            let end = b.iter().position(|&c| c == 0).unwrap_or(32);
            String::from_utf16_lossy(&b[..end])
        };
        fill_logfont(logfont, &family, weight, italic);
        unsafe { *issystemfont = true.into() };
        Ok(())
    }
    fn ConvertFontFaceToLOGFONT(&self, font: Ref<IDWriteFontFace>, logfont: *mut LOGFONTW) -> WResult<()> {
        let (family, weight, italic) = face_fields(font.as_ref().ok_or(E_INVALIDARG)?).ok_or(E_INVALIDARG)?;
        fill_logfont(logfont, &family, weight, italic);
        Ok(())
    }
    fn CreateFontFaceFromHdc(&self, hdc: HDC) -> WResult<IDWriteFontFace> {
        let hfont = HFONT(unsafe { GetCurrentObject(hdc, OBJ_FONT) }.0);
        let mut lf = LOGFONTW::default();
        if unsafe { GetObjectW(hfont.into(), core::mem::size_of::<LOGFONTW>() as i32, Some(&mut lf as *mut _ as *mut _)) } == 0 {
            return Err(E_INVALIDARG.into());
        }
        self.face_from_logfont(&lf)
    }
    fn CreateBitmapRenderTarget(&self, _hdc: HDC, width: u32, height: u32) -> WResult<IDWriteBitmapRenderTarget> {
        BitmapRenderTargetObj::new(width, height)
    }
}

// --- IDWriteBitmapRenderTarget (a DIB behind a memory DC) --------------------------------------

struct Dib {
    dc: HDC,
    bitmap: HGDIOBJ,
    old: HGDIOBJ,
    bits: *mut u8,
    width: u32,
    height: u32,
}

impl Dib {
    fn new(width: u32, height: u32) -> Option<Dib> {
        let (w, h) = (width.max(1), height.max(1));
        unsafe {
            let dc = CreateCompatibleDC(None);
            if dc.is_invalid() {
                return None;
            }
            // Top-down 32-bit BGRA (negative height), so row 0 is the top like our PixelBuffer.
            let bi = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER { biSize: core::mem::size_of::<BITMAPINFOHEADER>() as u32, biWidth: w as i32, biHeight: -(h as i32), biPlanes: 1, biBitCount: 32, biCompression: BI_RGB.0, ..Default::default() },
                ..Default::default()
            };
            let mut bits: *mut core::ffi::c_void = core::ptr::null_mut();
            let Ok(bitmap) = CreateDIBSection(Some(dc), &bi, DIB_RGB_COLORS, &mut bits, None, 0) else {
                let _ = DeleteDC(dc);
                return None;
            };
            let old = SelectObject(dc, bitmap.into());
            Some(Dib { dc, bitmap: bitmap.into(), old, bits: bits as *mut u8, width: w, height: h })
        }
    }
}

impl Drop for Dib {
    fn drop(&mut self) {
        unsafe {
            SelectObject(self.dc, self.old);
            let _ = DeleteObject(self.bitmap);
            let _ = DeleteDC(self.dc);
        }
    }
}

#[implement(IDWriteBitmapRenderTarget2)]
pub(crate) struct BitmapRenderTargetObj {
    dib: RefCell<Dib>,
    ppd: RefCell<f32>,
    transform: RefCell<[f32; 6]>,
    aa: core::cell::Cell<i32>,
}

impl BitmapRenderTargetObj {
    fn new(width: u32, height: u32) -> WResult<IDWriteBitmapRenderTarget> {
        let dib = Dib::new(width, height).ok_or(E_INVALIDARG)?;
        // Native's bitmap render target defaults to ClearType (its ctor zeroes the AA field).
        let rt: IDWriteBitmapRenderTarget2 = BitmapRenderTargetObj { dib: RefCell::new(dib), ppd: RefCell::new(1.0), transform: RefCell::new([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]), aa: core::cell::Cell::new(DWRITE_TEXT_ANTIALIAS_MODE_CLEARTYPE.0) }.into();
        Ok(rt.cast()?)
    }
}

impl IDWriteBitmapRenderTarget_Impl for BitmapRenderTargetObj_Impl {
    fn DrawGlyphRun(&self, bx: f32, by: f32, measuringmode: DWRITE_MEASURING_MODE, glyphrun: *const DWRITE_GLYPH_RUN, _params: Ref<IDWriteRenderingParams>, textcolor: COLORREF, blackboxrect: *mut RECT) -> WResult<()> {
        if !blackboxrect.is_null() {
            unsafe { *blackboxrect = RECT::default() };
        }
        let run = unsafe { glyphrun.as_ref() }.ok_or(E_INVALIDARG)?;
        let face = run.fontFace.as_ref().ok_or(E_INVALIDARG)?;
        let interop: ITextInterop = face.cast().map_err(|_| windows_core::Error::from(E_INVALIDARG))?;
        let (mut data, mut len, mut index) = (core::ptr::null(), 0usize, 0u32);
        if unsafe { interop.FontBlob(&mut data, &mut len, &mut index) }.is_err() || data.is_null() {
            return Err(E_INVALIDARG.into());
        }
        let bytes = unsafe { slice::from_raw_parts(data, len) }.to_vec();
        let n = run.glyphCount as usize;
        let ids = unsafe { slice::from_raw_parts(run.glyphIndices, n) };
        let adv: &[f32] = if run.glyphAdvances.is_null() { &[] } else { unsafe { slice::from_raw_parts(run.glyphAdvances, n) } };
        let off: &[DWRITE_GLYPH_OFFSET] = if run.glyphOffsets.is_null() { &[] } else { unsafe { slice::from_raw_parts(run.glyphOffsets, n) } };
        let glyphs: Vec<TGlyph> = (0..n).map(|i| TGlyph { id: ids[i], advance: adv.get(i).copied().unwrap_or(0.0), offset: off.get(i).map(|o| [o.advanceOffset, o.ascenderOffset]).unwrap_or([0.0, 0.0]) }).collect();

        let ppd = *self.ppd.borrow();
        let t = *self.transform.borrow();
        let dt = [t[0] as f64 * ppd as f64, t[1] as f64 * ppd as f64, t[2] as f64 * ppd as f64, t[3] as f64 * ppd as f64, t[4] as f64, t[5] as f64];
        let trun = TRun { font_data: &bytes, face_index: index, em_size: run.fontEmSize, glyphs: &glyphs, baseline_origin: [bx, by], measuring_mode: measuringmode.0 as u32, is_sideways: run.isSideways.as_bool(), bidi_level: run.bidiLevel, no_snap: false, simulations: 0, coords: &[] };
        let c = textcolor.0; // text COLORREF (0x00BBGGRR)

        // The render params (gamma/contrast/level) the caller passed. `grayscale_contrast` is the
        // scalar native's DrawGrayscale keys on (IDWriteRenderingParams1::GetGrayscaleEnhancedContrast).
        let params = match _params.as_ref() {
            Some(p) => RenderingParams {
                gamma: unsafe { p.GetGamma() },
                enhanced_contrast: unsafe { p.GetEnhancedContrast() },
                grayscale_contrast: p.cast::<IDWriteRenderingParams1>().map(|p1| unsafe { p1.GetGrayscaleEnhancedContrast() }).unwrap_or(1.0),
                cleartype_level: unsafe { p.GetClearTypeLevel() },
                pixel_geometry: unsafe { p.GetPixelGeometry() }.0 as u32,
                rendering_mode: unsafe { p.GetRenderingMode() }.0 as u32,
                ..RenderingParams::default()
            },
            None => RenderingParams::default(),
        };

        let dib = self.dib.borrow();
        let (w, h) = (dib.width, dib.height);
        let grayscale = self.aa.get() == DWRITE_TEXT_ANTIALIAS_MODE_GRAYSCALE.0;
        // Native's GDI DIB pipeline composites straight onto the DIB: ClearType (6×1) by default or
        // grayscale (8×1) when the target's AA mode is grayscale. Outline/aliased fall back below.
        {
            let px = unsafe { slice::from_raw_parts_mut(dib.bits as *mut u32, (w * h) as usize) };
            if gdi_raster::draw_glyph_run_gdi(px, w, h, &dt, &trun, c, grayscale, &params) {
                return Ok(());
            }
        }
        // Fallback (outline, aliased): render over a copy of the DIB through the D2D text
        // core, then write it back. IGNORE alpha ⇒ the core resolves DEFAULT to ClearType.
        let size = (w * h * 4) as usize;
        let color = [(c & 0xFF) as f32 / 255.0, ((c >> 8) & 0xFF) as f32 / 255.0, ((c >> 16) & 0xFF) as f32 / 255.0, 1.0];
        let mut buf = PixelBuffer {
            desc: ImageDesc { width: w, height: h, format: DXGI_FORMAT_B8G8R8A8_UNORM, alpha_mode: D2D1_ALPHA_MODE_IGNORE, dpi_x: 96.0, dpi_y: 96.0 },
            stride: (w * 4) as usize,
            data: unsafe { slice::from_raw_parts(dib.bits, size) }.to_vec(),
        };
        let aa = if grayscale { 2 } else { 1 };
        render_glyph_run(&mut buf, &dt, &trun, color, aa, &params);
        unsafe { core::ptr::copy_nonoverlapping(buf.data.as_ptr(), dib.bits, size) };
        Ok(())
    }
    fn GetMemoryDC(&self) -> HDC {
        self.dib.borrow().dc
    }
    fn GetPixelsPerDip(&self) -> f32 {
        *self.ppd.borrow()
    }
    fn SetPixelsPerDip(&self, pixelsperdip: f32) -> WResult<()> {
        *self.ppd.borrow_mut() = pixelsperdip;
        Ok(())
    }
    fn GetCurrentTransform(&self, transform: *mut DWRITE_MATRIX) -> WResult<()> {
        let t = *self.transform.borrow();
        unsafe { *transform = DWRITE_MATRIX { m11: t[0], m12: t[1], m21: t[2], m22: t[3], dx: t[4], dy: t[5] } };
        Ok(())
    }
    fn SetCurrentTransform(&self, transform: *const DWRITE_MATRIX) -> WResult<()> {
        *self.transform.borrow_mut() = match unsafe { transform.as_ref() } {
            Some(m) => [m.m11, m.m12, m.m21, m.m22, m.dx, m.dy],
            None => [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        };
        Ok(())
    }
    fn GetSize(&self) -> WResult<windows::Win32::Foundation::SIZE> {
        let d = self.dib.borrow();
        Ok(windows::Win32::Foundation::SIZE { cx: d.width as i32, cy: d.height as i32 })
    }
    fn Resize(&self, width: u32, height: u32) -> WResult<()> {
        let d = self.dib.borrow();
        if d.width == width && d.height == height {
            return Ok(());
        }
        drop(d);
        *self.dib.borrow_mut() = Dib::new(width, height).ok_or(E_INVALIDARG)?;
        Ok(())
    }
}

impl IDWriteBitmapRenderTarget1_Impl for BitmapRenderTargetObj_Impl {
    fn GetTextAntialiasMode(&self) -> DWRITE_TEXT_ANTIALIAS_MODE {
        DWRITE_TEXT_ANTIALIAS_MODE(self.aa.get())
    }
    fn SetTextAntialiasMode(&self, antialiasmode: DWRITE_TEXT_ANTIALIAS_MODE) -> WResult<()> {
        if !(0..=1).contains(&antialiasmode.0) {
            return Err(E_INVALIDARG.into());
        }
        self.aa.set(antialiasmode.0);
        Ok(())
    }
}

impl IDWriteBitmapRenderTarget2_Impl for BitmapRenderTargetObj_Impl {
    fn GetBitmapData(&self) -> WResult<DWRITE_BITMAP_DATA_BGRA32> {
        let d = self.dib.borrow();
        Ok(DWRITE_BITMAP_DATA_BGRA32 { width: d.width, height: d.height, pixels: d.bits as *mut u32 })
    }
}
