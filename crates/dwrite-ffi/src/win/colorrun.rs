//! `IDWriteFactory2::TranslateColorGlyphRun` + its `IDWriteColorGlyphRunEnumerator`. Decomposes a
//! COLRv0 colour glyph run into coloured sub-runs: each colour layer of each base glyph becomes a
//! one-glyph `DWRITE_COLOR_GLYPH_RUN` at the base glyph's position, with the layer's CPAL colour and
//! palette index (`0xFFFF` = the run's foreground colour, for a base glyph that has no colour record).
//! A run with no colour glyph at all reports `DWRITE_E_NOCOLOR`, as native does.
//!
//! The COLRv0 layer decomposition lives in `d2d::text::color::colrv0_layers` (additive, shared);
//! this module is just the COM enumerator + the CPAL colour lookup. COLRv1 (paint-tree / gradient /
//! bitmap / SVG) glyphs fall through to the base monochrome run — the `DWRITE_COLOR_GLYPH_RUN1`
//! paint-tree path is `IDWriteFactory4::TranslateColorGlyphRun`, not this one.

use super::{E_INVALIDARG, ITextInterop};
use core::cell::Cell;
use core::mem::ManuallyDrop;
use core::slice;
use d2d::text::color::colrv0_layers;
use skrifa::{FontRef, GlyphId, MetadataProvider};
use windows::Win32::Graphics::DirectWrite::*;
use windows_core::{BOOL, HRESULT, Interface, Result as WResult, implement};

const DWRITE_E_NOCOLOR: HRESULT = HRESULT(0x8898_500Cu32 as i32);
const PALETTE_FOREGROUND: u16 = 0xFFFF;

/// One decomposed sub-run: its owned 1-element glyph arrays plus the colour/palette/origin. The
/// `DWRITE_COLOR_GLYPH_RUN` the enumerator hands out points into these (kept stable by the `Box`).
struct SubRun {
    glyph: [u16; 1],
    advance: [f32; 1],
    offset: [DWRITE_GLYPH_OFFSET; 1],
    ox: f32,
    oy: f32,
    color: DWRITE_COLOR_F,
    palette: u16,
}

#[implement(IDWriteColorGlyphRunEnumerator1)]
pub(crate) struct EnumObj {
    face: IDWriteFontFace,
    em_size: f32,
    sideways: BOOL,
    bidi: u32,
    // The decomposed sub-runs are outline glyphs filled with the layer colour, so every sub-run
    // reports the font's outline format and the measuring mode the run was translated with (the two
    // fields `DWRITE_COLOR_GLYPH_RUN1` adds over the base). ponytail: VM-pending — the base run is
    // byte-exact, the two extra fields follow native's COLRv0-outline convention but are unverified.
    img_format: DWRITE_GLYPH_IMAGE_FORMATS,
    measuring: DWRITE_MEASURING_MODE,
    runs: Vec<Box<SubRun>>,
    cur: Cell<i32>,
    // The current sub-run's `DWRITE_COLOR_GLYPH_RUN1`, rebuilt on each `GetCurrentRun` and kept alive
    // until the next call (its `fontFace` is a counted ref we drop when it is replaced/dropped). The
    // base `IDWriteColorGlyphRunEnumerator::GetCurrentRun` returns `&run1.Base` (same address).
    out: Cell<*mut DWRITE_COLOR_GLYPH_RUN1>,
}

impl Drop for EnumObj {
    fn drop(&mut self) {
        self.clear_out();
    }
}

impl EnumObj {
    fn clear_out(&self) {
        let p = self.out.replace(core::ptr::null_mut());
        if !p.is_null() {
            let mut b = unsafe { Box::from_raw(p) };
            unsafe { ManuallyDrop::drop(&mut b.Base.glyphRun.fontFace) }; // release the counted face ref
        }
    }

    /// Build the current sub-run's `DWRITE_COLOR_GLYPH_RUN1`, store it (freeing any previous), and
    /// return the pointer. Both `GetCurrentRun` overloads share it; the base one re-casts to `&Base`.
    fn build_run1(&self, i: usize) -> *mut DWRITE_COLOR_GLYPH_RUN1 {
        self.clear_out();
        let sr = &self.runs[i];
        let cgr = Box::new(DWRITE_COLOR_GLYPH_RUN1 {
            Base: DWRITE_COLOR_GLYPH_RUN {
                glyphRun: DWRITE_GLYPH_RUN {
                    fontFace: ManuallyDrop::new(Some(self.face.clone())),
                    fontEmSize: self.em_size,
                    glyphCount: 1,
                    glyphIndices: sr.glyph.as_ptr(),
                    glyphAdvances: sr.advance.as_ptr(),
                    glyphOffsets: sr.offset.as_ptr(),
                    isSideways: self.sideways,
                    bidiLevel: self.bidi,
                },
                glyphRunDescription: core::ptr::null_mut(),
                baselineOriginX: sr.ox,
                baselineOriginY: sr.oy,
                runColor: sr.color,
                paletteIndex: sr.palette,
            },
            glyphImageFormat: self.img_format,
            measuringMode: self.measuring,
        });
        let p = Box::into_raw(cgr);
        self.out.set(p);
        p
    }
}

/// `TranslateColorGlyphRun`: build the enumerator, or `DWRITE_E_NOCOLOR` if no glyph has colour.
/// `measuring` is the run's `DWRITE_MEASURING_MODE` (carried on the `DWRITE_COLOR_GLYPH_RUN1`).
pub(crate) fn create(bx: f32, by: f32, run: *const DWRITE_GLYPH_RUN, palette_index: u32, measuring: DWRITE_MEASURING_MODE) -> WResult<IDWriteColorGlyphRunEnumerator> {
    if run.is_null() {
        return Err(E_INVALIDARG.into());
    }
    let run = unsafe { &*run };
    let Some(face) = run.fontFace.as_ref() else { return Err(E_INVALIDARG.into()) };
    let interop: ITextInterop = face.cast().map_err(|_| windows_core::Error::from(E_INVALIDARG))?;
    let (mut data, mut len, mut index) = (core::ptr::null(), 0usize, 0u32);
    if unsafe { interop.FontBlob(&mut data, &mut len, &mut index) }.is_err() || data.is_null() {
        return Err(E_INVALIDARG.into());
    }
    let bytes = unsafe { slice::from_raw_parts(data, len) };
    let Ok(font) = FontRef::from_index(bytes, index) else { return Err(E_INVALIDARG.into()) };

    let n = run.glyphCount as usize;
    let ids = unsafe { slice::from_raw_parts(run.glyphIndices, n) };
    let adv: &[f32] = if run.glyphAdvances.is_null() { &[] } else { unsafe { slice::from_raw_parts(run.glyphAdvances, n) } };
    let off: &[DWRITE_GLYPH_OFFSET] = if run.glyphOffsets.is_null() { &[] } else { unsafe { slice::from_raw_parts(run.glyphOffsets, n) } };

    // CPAL palette (requested index, clamped to 0 as native falls back); colours are straight RGBA.
    let palettes = font.color_palettes();
    let palette = palettes.get(palette_index as u16).or_else(|| palettes.get(0));

    let mut runs: Vec<Box<SubRun>> = Vec::new();
    let mut any_color = false;
    let mut pen = bx;
    for i in 0..n {
        let a = adv.get(i).copied().unwrap_or(0.0);
        let o = off.get(i).copied().unwrap_or(DWRITE_GLYPH_OFFSET { advanceOffset: 0.0, ascenderOffset: 0.0 });
        match colrv0_layers(&font, GlyphId::new(ids[i] as u32)) {
            Some(layers) => {
                any_color = true;
                for layer in layers {
                    let color = cpal(palette.as_ref(), layer.palette_index);
                    runs.push(Box::new(SubRun { glyph: [layer.glyph], advance: [a], offset: [o], ox: pen, oy: by, color, palette: layer.palette_index }));
                }
            }
            // No colour record: the base glyph is drawn monochrome in the run's foreground colour.
            None => runs.push(Box::new(SubRun { glyph: [ids[i]], advance: [a], offset: [o], ox: pen, oy: by, color: DWRITE_COLOR_F { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }, palette: PALETTE_FOREGROUND })),
        }
        // A right-to-left run's pen walks left from its right edge (each sub-run keeps the run's level).
        pen += if run.bidiLevel & 1 == 1 { -a } else { a };
    }
    if !any_color {
        return Err(DWRITE_E_NOCOLOR.into());
    }
    // COLRv0 layers are outline glyphs; report the font's outline format (CFF else TrueType).
    let img_format = if font.table_data(skrifa::Tag::new(b"CFF ")).is_some() { DWRITE_GLYPH_IMAGE_FORMATS_CFF } else { DWRITE_GLYPH_IMAGE_FORMATS_TRUETYPE };
    let e: IDWriteColorGlyphRunEnumerator1 = EnumObj {
        face: face.clone(),
        em_size: run.fontEmSize,
        sideways: run.isSideways,
        bidi: run.bidiLevel,
        img_format,
        measuring,
        runs,
        cur: Cell::new(-1),
        out: Cell::new(core::ptr::null_mut()),
    }
    .into();
    Ok(e.cast()?)
}

/// CPAL entry `index` as straight RGBA (0..1), or the foreground sentinel → opaque black placeholder
/// (the caller uses its own brush when `paletteIndex == 0xFFFF`).
fn cpal(palette: Option<&skrifa::color::ColorPalette>, index: u16) -> DWRITE_COLOR_F {
    if index == PALETTE_FOREGROUND {
        return DWRITE_COLOR_F { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    }
    let c = palette.and_then(|p| p.colors().get(index as usize).copied());
    match c {
        Some(c) => DWRITE_COLOR_F { r: c.red as f32 / 255.0, g: c.green as f32 / 255.0, b: c.blue as f32 / 255.0, a: c.alpha as f32 / 255.0 },
        None => DWRITE_COLOR_F { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
    }
}

impl IDWriteColorGlyphRunEnumerator_Impl for EnumObj_Impl {
    fn MoveNext(&self) -> WResult<BOOL> {
        self.clear_out();
        let next = self.cur.get() + 1;
        self.cur.set(next);
        Ok(((next as usize) < self.runs.len()).into())
    }
    fn GetCurrentRun(&self) -> WResult<*mut DWRITE_COLOR_GLYPH_RUN> {
        let i = self.cur.get();
        if i < 0 || (i as usize) >= self.runs.len() {
            return Err(E_INVALIDARG.into());
        }
        // `Base` is the first field of `DWRITE_COLOR_GLYPH_RUN1`, so the RUN1 pointer IS the base one.
        Ok(self.build_run1(i as usize) as *mut DWRITE_COLOR_GLYPH_RUN)
    }
}

impl IDWriteColorGlyphRunEnumerator1_Impl for EnumObj_Impl {
    fn GetCurrentRun(&self) -> WResult<*mut DWRITE_COLOR_GLYPH_RUN1> {
        let i = self.cur.get();
        if i < 0 || (i as usize) >= self.runs.len() {
            return Err(E_INVALIDARG.into());
        }
        Ok(self.build_run1(i as usize))
    }
}
