//! `IDWriteTextFormat` and `IDWriteTextLayout` over `d2d::text::layout`. The format carries the
//! resolved font + alignment fields (and hands them to our `d2d1.dll` via [`ITextInterop`] for
//! `DrawText`); the layout computes the whole `d2d::text::layout::Layout` up front and forwards
//! every metrics/hit-test/`Draw` query to it.

use super::fonts::{FaceObj, FontSource, SystemFonts};
use super::{E_NOTIMPL, ITextInterop, ITextInterop_Impl};
use core::ffi::c_void;
use core::mem::ManuallyDrop;
use core::slice;
use d2d::text::layout::{self, Run};
use d2d::text::shaping;
use skrifa::instance::{LocationRef, Size};
use skrifa::{FontRef, GlyphId, MetadataProvider};
use std::cell::{Cell, RefCell};
use std::sync::Arc;
use windows::Win32::Graphics::DirectWrite::*;
use windows_core::{BOOL, HRESULT, Interface, PCWSTR, PWSTR, Ref, Result as WResult, implement};

/// `HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER)` — the probe return the `Get*Metrics(count)` APIs
/// use so the caller can size its buffer.
const E_NOT_SUFFICIENT: HRESULT = HRESULT(0x8007_007Au32 as i32);

// --- IDWriteTextFormat -------------------------------------------------------------------------

#[implement(IDWriteTextFormat3, ITextInterop)]
pub(crate) struct FormatObj {
    source: FontSource,
    bytes: Arc<Vec<u8>>,
    index: u32,
    family: String,
    em: f32,
    /// The resolved face's `DWRITE_FONT_SIMULATIONS` (e.g. bold-sim for Impact at weight 700), so
    /// layout applies native's bold advance embolden.
    sims: u16,
    fonts: Arc<SystemFonts>,
    // The font selection requested at creation — returned verbatim by the weight/style/stretch/locale
    // getters (native keeps the request, not the resolved face's attributes).
    weight: u16,  // DWRITE_FONT_WEIGHT
    style: u32,   // DWRITE_FONT_STYLE
    stretch: u16, // DWRITE_FONT_STRETCH
    locale: String,
    align: Cell<u32>,   // DWRITE_TEXT_ALIGNMENT, 1:1 with layout::TextFormat.align
    para: Cell<u32>,    // DWRITE_PARAGRAPH_ALIGNMENT
    wrap: Cell<u32>,    // DWRITE_WORD_WRAPPING
    reading: Cell<u32>, // DWRITE_READING_DIRECTION (base_rtl() derives the layout's RTL flag)
    flow: Cell<u32>,    // DWRITE_FLOW_DIRECTION
    tab: Cell<f32>,     // incremental tab stop
    // Line spacing: (method, height, baseline, leadingBefore, fontLineGapUsage). The base
    // IDWriteTextFormat::SetLineSpacing writes the first three; IDWriteTextFormat2 the full set.
    line: Cell<(i32, f32, f32, f32, i32)>,
    trimming: Cell<(u32, u16, u16)>, // granularity, delimiter, delimiterCount
    trim_sign: RefCell<Option<IDWriteInlineObject>>,
    // IDWriteTextFormat1/2/3 state — round-trips through the matching Get*/Set* pairs.
    vgo: Cell<i32>,             // DWRITE_VERTICAL_GLYPH_ORIENTATION (default DEFAULT=0)
    last_line_wrap: Cell<bool>, // native default TRUE
    optical: Cell<i32>,         // DWRITE_OPTICAL_ALIGNMENT (default NONE=0)
    fallback: RefCell<Option<IDWriteFontFallback>>,
    auto_axes: Cell<i32>, // DWRITE_AUTOMATIC_FONT_AXES (default NONE=0)
    font_axes: RefCell<Vec<DWRITE_FONT_AXIS_VALUE>>,
}

impl FormatObj {
    /// Right-to-left, or bottom-to-top (which lays out as right-to-left along its columns).
    fn base_rtl(&self) -> bool {
        self.reading.get() & 1 == 1
    }
}

/// `CreateTextFormat`: resolve the family + weight/style to a font and store the format fields. The
/// requested weight/style/stretch/locale are kept for the getters; the default incremental tab stop
/// is 4× the em size (DirectWrite's default).
pub(crate) fn make_format(fonts: &Arc<SystemFonts>, family: &str, weight: u16, style: u32, stretch: u16, em: f32, locale: &str) -> WResult<IDWriteTextFormat> {
    // A family the collection lacks still makes a format: native lays its text out through the system
    // fallback, which gives Segoe UI to Latin and the scripts Segoe UI covers (VM r25 "NoSuchFont" runs), and
    // the per-character fallback takes the rest. The format keeps the requested name.
    let stand_in;
    let (base, keep_name) = if fonts.resolve_sims(family, weight, style != 0).is_some() {
        (family, false)
    } else {
        stand_in = fonts.stand_in().ok_or(E_NOTIMPL)?;
        (stand_in.as_str(), true)
    };
    let (bytes, index, canon, sims) = fonts.resolve_sims(base, weight, style != 0).ok_or(E_NOTIMPL)?;
    let canon = if keep_name { family.to_string() } else { canon };
    let fmt: IDWriteTextFormat3 = FormatObj {
        source: fonts.resolve_source(base, weight, style != 0).ok_or(E_NOTIMPL)?,
        bytes,
        index,
        family: canon,
        em,
        sims,
        fonts: fonts.clone(),
        weight,
        style,
        stretch,
        locale: locale.to_string(),
        align: Cell::new(0),
        para: Cell::new(0),
        wrap: Cell::new(0),
        reading: Cell::new(0),
        flow: Cell::new(0),
        tab: Cell::new(em * 4.0),
        line: Cell::new((0, 0.0, 0.0, 0.0, 0)),
        trimming: Cell::new((0, 0, 0)),
        trim_sign: RefCell::new(None),
        vgo: Cell::new(0),
        last_line_wrap: Cell::new(true),
        optical: Cell::new(0),
        fallback: RefCell::new(None),
        auto_axes: Cell::new(0),
        font_axes: RefCell::new(Vec::new()),
    }
    .into();
    Ok(fmt.cast()?)
}

impl ITextInterop_Impl for FormatObj_Impl {
    unsafe fn FontRasterizer(&self, rasterizer: *mut u32) -> HRESULT {
        if rasterizer.is_null() { return super::E_INVALIDARG; }
        unsafe { *rasterizer = (self.source.rasterizer() == d2d::text::hinting::Rasterizer::FontCache) as u32; }
        HRESULT(0)
    }
    unsafe fn FontBlob(&self, data: *mut *const u8, len: *mut usize, index: *mut u32) -> HRESULT {
        unsafe {
            *data = self.bytes.as_ptr();
            *len = self.bytes.len();
            *index = self.index;
        }
        HRESULT(0)
    }
    unsafe fn FormatParams(&self, em: *mut f32, align: *mut u32, wrap: *mut u32, base_rtl: *mut i32, sims: *mut u32) -> HRESULT {
        unsafe {
            *em = self.em;
            *align = self.align.get();
            *wrap = self.wrap.get();
            *base_rtl = self.base_rtl() as i32;
            *sims = self.sims as u32;
        }
        HRESULT(0)
    }
}

impl IDWriteTextFormat_Impl for FormatObj_Impl {
    fn SetTextAlignment(&self, a: DWRITE_TEXT_ALIGNMENT) -> WResult<()> {
        self.align.set(a.0 as u32);
        Ok(())
    }
    fn SetParagraphAlignment(&self, p: DWRITE_PARAGRAPH_ALIGNMENT) -> WResult<()> {
        self.para.set(p.0 as u32);
        Ok(())
    }
    fn SetWordWrapping(&self, w: DWRITE_WORD_WRAPPING) -> WResult<()> {
        self.wrap.set(w.0 as u32);
        Ok(())
    }
    fn SetReadingDirection(&self, r: DWRITE_READING_DIRECTION) -> WResult<()> {
        self.reading.set(r.0 as u32);
        Ok(())
    }
    fn SetFlowDirection(&self, f: DWRITE_FLOW_DIRECTION) -> WResult<()> {
        self.flow.set(f.0 as u32);
        Ok(())
    }
    fn SetIncrementalTabStop(&self, t: f32) -> WResult<()> {
        self.tab.set(t);
        Ok(())
    }
    fn SetTrimming(&self, o: *const DWRITE_TRIMMING, s: Ref<IDWriteInlineObject>) -> WResult<()> {
        if let Some(t) = unsafe { o.as_ref() } {
            self.trimming.set((t.granularity.0 as u32, t.delimiter as u16, t.delimiterCount as u16));
        }
        *self.trim_sign.borrow_mut() = s.as_ref().cloned();
        Ok(())
    }
    fn SetLineSpacing(&self, m: DWRITE_LINE_SPACING_METHOD, s: f32, b: f32) -> WResult<()> {
        let (_, _, _, lead, gap) = self.line.get();
        self.line.set((m.0, s, b, lead, gap));
        Ok(())
    }
    fn GetTextAlignment(&self) -> DWRITE_TEXT_ALIGNMENT {
        DWRITE_TEXT_ALIGNMENT(self.align.get() as i32)
    }
    fn GetParagraphAlignment(&self) -> DWRITE_PARAGRAPH_ALIGNMENT {
        DWRITE_PARAGRAPH_ALIGNMENT(self.para.get() as i32)
    }
    fn GetWordWrapping(&self) -> DWRITE_WORD_WRAPPING {
        DWRITE_WORD_WRAPPING(self.wrap.get() as i32)
    }
    fn GetReadingDirection(&self) -> DWRITE_READING_DIRECTION {
        DWRITE_READING_DIRECTION(self.reading.get() as i32)
    }
    fn GetFlowDirection(&self) -> DWRITE_FLOW_DIRECTION {
        DWRITE_FLOW_DIRECTION(self.flow.get() as i32)
    }
    fn GetIncrementalTabStop(&self) -> f32 {
        self.tab.get()
    }
    fn GetTrimming(&self, o: *mut DWRITE_TRIMMING, s: windows_core::OutRef<IDWriteInlineObject>) -> WResult<()> {
        let (g, d, dc) = self.trimming.get();
        unsafe { *o = DWRITE_TRIMMING { granularity: DWRITE_TRIMMING_GRANULARITY(g as i32), delimiter: d as u32, delimiterCount: dc as u32 } };
        s.write(self.trim_sign.borrow().clone())?;
        Ok(())
    }
    fn GetLineSpacing(&self, m: *mut DWRITE_LINE_SPACING_METHOD, s: *mut f32, b: *mut f32) -> WResult<()> {
        let (lm, ls, lb, _, _) = self.line.get();
        unsafe {
            *m = DWRITE_LINE_SPACING_METHOD(lm);
            *s = ls;
            *b = lb;
        }
        Ok(())
    }
    fn GetFontCollection(&self) -> WResult<IDWriteFontCollection> {
        Ok(super::fonts::collection(self.fonts.clone()))
    }
    fn GetFontFamilyNameLength(&self) -> u32 {
        self.family.encode_utf16().count() as u32
    }
    fn GetFontFamilyName(&self, fontfamilyname: PWSTR, namesize: u32) -> WResult<()> {
        super::write_wide(&self.family, fontfamilyname, namesize)
    }
    fn GetFontWeight(&self) -> DWRITE_FONT_WEIGHT {
        DWRITE_FONT_WEIGHT(self.weight as i32)
    }
    fn GetFontStyle(&self) -> DWRITE_FONT_STYLE {
        DWRITE_FONT_STYLE(self.style as i32)
    }
    fn GetFontStretch(&self) -> DWRITE_FONT_STRETCH {
        DWRITE_FONT_STRETCH(self.stretch as i32)
    }
    fn GetFontSize(&self) -> f32 {
        self.em
    }
    fn GetLocaleNameLength(&self) -> u32 {
        self.locale.encode_utf16().count() as u32
    }
    fn GetLocaleName(&self, localename: PWSTR, namesize: u32) -> WResult<()> {
        super::write_wide(&self.locale, localename, namesize)
    }
}

impl IDWriteTextFormat1_Impl for FormatObj_Impl {
    fn SetVerticalGlyphOrientation(&self, o: DWRITE_VERTICAL_GLYPH_ORIENTATION) -> WResult<()> {
        self.vgo.set(o.0);
        Ok(())
    }
    fn GetVerticalGlyphOrientation(&self) -> DWRITE_VERTICAL_GLYPH_ORIENTATION {
        DWRITE_VERTICAL_GLYPH_ORIENTATION(self.vgo.get())
    }
    fn SetLastLineWrapping(&self, enabled: BOOL) -> WResult<()> {
        self.last_line_wrap.set(enabled.as_bool());
        Ok(())
    }
    fn GetLastLineWrapping(&self) -> BOOL {
        self.last_line_wrap.get().into()
    }
    fn SetOpticalAlignment(&self, a: DWRITE_OPTICAL_ALIGNMENT) -> WResult<()> {
        self.optical.set(a.0);
        Ok(())
    }
    fn GetOpticalAlignment(&self) -> DWRITE_OPTICAL_ALIGNMENT {
        DWRITE_OPTICAL_ALIGNMENT(self.optical.get())
    }
    fn SetFontFallback(&self, f: Ref<IDWriteFontFallback>) -> WResult<()> {
        *self.fallback.borrow_mut() = f.as_ref().cloned();
        Ok(())
    }
    fn GetFontFallback(&self) -> WResult<IDWriteFontFallback> {
        // The custom fallback set via SetFontFallback, else the system fallback. ponytail: native
        // returns NULL (S_OK) when none was set; a typed `Result<IDWriteFontFallback>` cannot express
        // a null-success, so an unset format reports the system fallback (the fallback DrawText uses).
        match self.fallback.borrow().clone() {
            Some(f) => Ok(f),
            None => Ok(super::fonts::FallbackObj { fonts: self.fonts.clone(), mappings: Vec::new() }.into()),
        }
    }
}

impl IDWriteTextFormat2_Impl for FormatObj_Impl {
    fn SetLineSpacing(&self, o: *const DWRITE_LINE_SPACING) -> WResult<()> {
        let o = unsafe { o.as_ref() }.ok_or(super::E_INVALIDARG)?;
        self.line.set((o.method.0, o.height, o.baseline, o.leadingBefore, o.fontLineGapUsage.0));
        Ok(())
    }
    fn GetLineSpacing(&self, o: *mut DWRITE_LINE_SPACING) -> WResult<()> {
        let (m, h, b, lead, gap) = self.line.get();
        unsafe {
            *o = DWRITE_LINE_SPACING {
                method: DWRITE_LINE_SPACING_METHOD(m),
                height: h,
                baseline: b,
                leadingBefore: lead,
                fontLineGapUsage: DWRITE_FONT_LINE_GAP_USAGE(gap),
            };
        }
        Ok(())
    }
}

impl IDWriteTextFormat3_Impl for FormatObj_Impl {
    fn SetFontAxisValues(&self, values: *const DWRITE_FONT_AXIS_VALUE, count: u32) -> WResult<()> {
        *self.font_axes.borrow_mut() = read_axis_values(values, count);
        Ok(())
    }
    fn GetFontAxisValueCount(&self) -> u32 {
        // The axes set via SetFontAxisValues. ponytail: a legacy CreateTextFormat font (no explicit
        // axes) synthesizes wght/wdth/ital/opsz axes in native; that synthesis is VM-pending, so an
        // unset format reports none here (round-trips exactly once SetFontAxisValues is called).
        self.font_axes.borrow().len() as u32
    }
    fn GetFontAxisValues(&self, out: *mut DWRITE_FONT_AXIS_VALUE, count: u32) -> WResult<()> {
        write_axis_values(&self.font_axes.borrow(), out, count)
    }
    fn GetAutomaticFontAxes(&self) -> DWRITE_AUTOMATIC_FONT_AXES {
        DWRITE_AUTOMATIC_FONT_AXES(self.auto_axes.get())
    }
    fn SetAutomaticFontAxes(&self, axes: DWRITE_AUTOMATIC_FONT_AXES) -> WResult<()> {
        self.auto_axes.set(axes.0);
        Ok(())
    }
}

/// Copy a caller `DWRITE_FONT_AXIS_VALUE[count]` into an owned Vec (null/0 → empty).
fn read_axis_values(values: *const DWRITE_FONT_AXIS_VALUE, count: u32) -> Vec<DWRITE_FONT_AXIS_VALUE> {
    if values.is_null() || count == 0 {
        return Vec::new();
    }
    unsafe { slice::from_raw_parts(values, count as usize) }.to_vec()
}

/// Write stored axis values into a caller buffer; `E_NOT_SUFFICIENT` if the buffer is too small (the
/// `IDWrite*::GetFontAxisValues(count)` contract — the caller sizes with `GetFontAxisValueCount`).
pub(crate) fn write_axis_values(axes: &[DWRITE_FONT_AXIS_VALUE], out: *mut DWRITE_FONT_AXIS_VALUE, count: u32) -> WResult<()> {
    if (count as usize) < axes.len() {
        return Err(E_NOT_SUFFICIENT.into());
    }
    if !out.is_null() && !axes.is_empty() {
        unsafe { slice::from_raw_parts_mut(out, axes.len()) }.copy_from_slice(axes);
    }
    Ok(())
}

// --- IDWriteTextLayout -------------------------------------------------------------------------

/// One per-range `SetFont*` override, stored in call order; flattened last-wins per position for Draw
/// (faces + per-range em) and the `Get*` getters (effective value + its uniform range at a position).
#[derive(Clone)]
enum FmtAttr {
    Family(String),
    Weight(u16),
    Style(u32),
    Stretch(u16),
    Size(f32),
}

#[implement(IDWriteTextLayout4)]
pub(crate) struct LayoutObj {
    // The laid-out result, re-computed when trimming changes. `font`/`tf`/`text`/`max_w`/`max_h` are
    // kept so `SetTrimming` can re-lay the string.
    layout: RefCell<layout::Layout>,
    font: shaping::Font,
    // The layout's format (alignment/wrap/para/reading-RTL/em). Mutable: the inherited IDWriteTextFormat
    // setters (SetTextAlignment/ParagraphAlignment/WordWrapping/ReadingDirection) change it in place and
    // re-lay the string, exactly as making a new layout from that format would (Paint.NET sets the
    // alignment on the layout AFTER CreateTextLayout, then draws through IDWriteTextLayout::Draw).
    tf: Cell<layout::TextFormat>,
    // Raw DWRITE reading/flow directions + incremental tab stop, for the Get* round-trip (flow/tab do
    // not reach layout_ex yet; reading's RTL flag is folded into `tf.base_rtl`).
    reading: Cell<u32>,
    flow: Cell<u32>,
    tab: Cell<f32>,
    text: String,
    max_w: f32,
    max_h: f32,
    trimming: Cell<Option<layout::Trimming>>,
    text_u16_len: u32,
    source: FontSource,
    bytes: Arc<Vec<u8>>,
    index: u32,
    family: String,
    em: f32,
    fonts: Arc<SystemFonts>,
    measuring: u32,
    /// `pixelsPerDip` from `CreateGdiCompatibleTextLayout` (1.0 for the natural `CreateTextLayout`),
    /// threaded into the layout so GDI modes use the right device ppem for VDMX line metrics + hinted
    /// advances. A GDI-compatible layout that ignored it measured NATURAL (wrong line box / carets).
    ppd: f32,
    /// Bold-sim advance delta (design units) for the base face, 0 for a real face — native's bold
    /// embolden, applied to the layout advances.
    bold_delta_du: i32,
    // The base format fields, so the inherited IDWriteTextFormat getters and the per-range
    // IDWriteTextLayout::Get* getters report the layout's single (whole-range) format.
    weight: u16,
    style: u32,
    stretch: u16,
    locale: String,
    // Per-range character-format overrides (SetFontFamilyName/Weight/Style/Stretch/Size) and
    // underline/strikethrough, each stored as (startPosition, length, value) in call order. Flattened
    // last-wins per position: for Draw (faces + per-range em via the core `FontSpan`s) and for the
    // Get* getters (effective value at a position + the uniform range around it).
    fmt_ranges: RefCell<Vec<(u32, u32, FmtAttr)>>,
    ul_ranges: RefCell<Vec<(u32, u32, bool)>>,
    stk_ranges: RefCell<Vec<(u32, u32, bool)>>,
    drawing_effect: RefCell<Option<windows_core::IUnknown>>,
    inline_object: RefCell<Option<IDWriteInlineObject>>,
    typography: RefCell<Option<IDWriteTypography>>,
    // IDWriteTextLayout1-4 layout-wide settings (stored whole-layout, like the fields above; native
    // keys pair-kerning/char-spacing/axis-values per range — thread ranges if a scene reads a
    // partial range back differently). Defaults match native (last-line-wrapping on, the rest off).
    pair_kerning: Cell<bool>,
    char_spacing: Cell<(f32, f32, f32)>,
    // The (start, length) ranges SetPairKerning / SetCharacterSpacing were last given. Native echoes
    // the set range back from the getter verbatim — Set([0, UINT32_MAX)) reads back UINT32_MAX, while
    // Set([0, text-len)) reads back text-len (VM-verified) — so store and return it rather than a fixed
    // value. Default [0, UINT32_MAX) (an unset uniform layout, like the format-inherited getters).
    pair_kerning_range: Cell<(u32, u32)>,
    char_spacing_range: Cell<(u32, u32)>,
    vertical_glyph_orientation: Cell<i32>,
    last_line_wrapping: Cell<bool>,
    optical_alignment: Cell<i32>,
    font_fallback: RefCell<Option<IDWriteFontFallback>>,
    line_spacing: Cell<DWRITE_LINE_SPACING>,
    layout_font_axes: RefCell<Vec<DWRITE_FONT_AXIS_VALUE>>,
    automatic_font_axes: Cell<i32>,
}

/// `CreateTextLayout`: lay the string out with the format's font + fields, keeping the whole result.
pub(crate) fn make_layout(fmt: Ref<IDWriteTextFormat>, text: &str, max_w: f32, max_h: f32, measuring: u32, ppd: f32) -> WResult<IDWriteTextLayout> {
    // Native rejects a max width/height that is not `>= 0` (negative, −inf, NaN); ±0 and +inf lay out
    // (12 VM probes, C API fuzz effects seed 3).
    if [max_w, max_h].iter().any(|v| v.is_nan() || *v < 0.0) {
        return Err(super::E_INVALIDARG.into());
    }
    let fo = fmt.as_ref().ok_or(E_NOTIMPL)?.cast_object_ref::<FormatObj>().map_err(|_| windows_core::Error::from(E_NOTIMPL))?;
    // A reading direction that is not perpendicular to the flow direction fails the layout with
    // DWRITE_E_FLOWDIRECTIONCONFLICTS (VM: every horizontal/horizontal and vertical/vertical pair).
    if (fo.reading.get() < 2) != (fo.flow.get() < 2) {
        return Err(windows_core::HRESULT(0x8898_500Bu32 as i32).into());
    }
    let font = shaping::Font::from_bytes((*fo.bytes).clone(), fo.index).ok_or(E_NOTIMPL)?;
    let (lm, ls, lb, lead, gap) = fo.line.get();
    let tf = layout::TextFormat { em: fo.em, base_rtl: fo.base_rtl(), align: fo.align.get() as u8, wrap: fo.wrap.get() as u8, para: fo.para.get() as u8, flow: fo.flow.get() as u8, line_spacing: (lm as u8, ls, lb) };
    // Native CreateTextLayout rejects a negative maxHeight with E_INVALIDARG; a negative maxWidth it
    // tolerates (clamped to 0 = everything overflows). Match that asymmetry.
    if max_h < 0.0 {
        return Err(super::E_INVALIDARG.into());
    }
    let (max_w, max_h) = (max_w.max(0.0), max_h.max(0.0));
    // Bold-sim faces (e.g. Impact at weight 700) carry a design-unit advance embolden into layout.
    let bold_delta_du = if fo.sims & DWRITE_FONT_SIMULATIONS_BOLD.0 as u16 != 0 { (font.upm as f32 / 50.0).round() as i32 } else { 0 };
    let laid = layout::layout_ex(&font, &tf, text, max_w, max_h, &layout::LayoutOpts { measuring, pixels_per_dip: ppd, base_bold_delta_du: bold_delta_du, ..Default::default() });
    let layout: IDWriteTextLayout4 = LayoutObj {
        layout: RefCell::new(laid),
        font,
        tf: Cell::new(tf),
        reading: Cell::new(fo.reading.get()),
        flow: Cell::new(fo.flow.get()),
        tab: Cell::new(fo.tab.get()),
        text: text.to_string(),
        max_w,
        max_h,
        trimming: Cell::new(None),
        text_u16_len: text.encode_utf16().count() as u32,
        source: fo.source.clone(),
        bytes: fo.bytes.clone(),
        index: fo.index,
        family: fo.family.clone(),
        em: fo.em,
        fonts: fo.fonts.clone(),
        measuring,
        ppd,
        bold_delta_du,
        weight: fo.weight,
        style: fo.style,
        stretch: fo.stretch,
        locale: fo.locale.clone(),
        fmt_ranges: RefCell::new(Vec::new()),
        ul_ranges: RefCell::new(Vec::new()),
        stk_ranges: RefCell::new(Vec::new()),
        drawing_effect: RefCell::new(None),
        inline_object: RefCell::new(None),
        typography: RefCell::new(None),
        pair_kerning: Cell::new(false),
        char_spacing: Cell::new((0.0, 0.0, 0.0)),
        pair_kerning_range: Cell::new((0, u32::MAX)),
        char_spacing_range: Cell::new((0, u32::MAX)),
        vertical_glyph_orientation: Cell::new(0),
        last_line_wrapping: Cell::new(true),
        optical_alignment: Cell::new(0),
        font_fallback: RefCell::new(None),
        line_spacing: Cell::new(DWRITE_LINE_SPACING { method: DWRITE_LINE_SPACING_METHOD(lm), height: ls, baseline: lb, leadingBefore: lead, fontLineGapUsage: DWRITE_FONT_LINE_GAP_USAGE(gap) }),
        layout_font_axes: RefCell::new(Vec::new()),
        automatic_font_axes: Cell::new(0),
    }
    .into();
    Ok(layout.cast()?)
}

impl IDWriteTextLayout1_Impl for LayoutObj_Impl {
    fn SetPairKerning(&self, ispairkerningenabled: BOOL, textrange: &DWRITE_TEXT_RANGE) -> WResult<()> {
        self.pair_kerning.set(ispairkerningenabled.as_bool());
        self.pair_kerning_range.set((textrange.startPosition, textrange.length));
        Ok(())
    }
    fn GetPairKerning(&self, _currentposition: u32, ispairkerningenabled: *mut BOOL, textrange: *mut DWRITE_TEXT_RANGE) -> WResult<()> {
        unsafe { *ispairkerningenabled = self.pair_kerning.get().into() };
        self.put_stored_range(textrange, self.pair_kerning_range.get());
        Ok(())
    }
    fn SetCharacterSpacing(&self, leadingspacing: f32, trailingspacing: f32, minimumadvancewidth: f32, textrange: &DWRITE_TEXT_RANGE) -> WResult<()> {
        self.char_spacing.set((leadingspacing, trailingspacing, minimumadvancewidth));
        self.char_spacing_range.set((textrange.startPosition, textrange.length));
        Ok(())
    }
    fn GetCharacterSpacing(&self, _currentposition: u32, leadingspacing: *mut f32, trailingspacing: *mut f32, minimumadvancewidth: *mut f32, textrange: *mut DWRITE_TEXT_RANGE) -> WResult<()> {
        let (l, t, m) = self.char_spacing.get();
        unsafe {
            *leadingspacing = l;
            *trailingspacing = t;
            *minimumadvancewidth = m;
        }
        self.put_stored_range(textrange, self.char_spacing_range.get());
        Ok(())
    }
}

impl IDWriteTextLayout2_Impl for LayoutObj_Impl {
    fn GetMetrics(&self, textmetrics: *mut DWRITE_TEXT_METRICS1) -> WResult<()> {
        let lay = self.layout.borrow();
        let m = &lay.metrics;
        unsafe {
            *textmetrics = DWRITE_TEXT_METRICS1 {
                Base: DWRITE_TEXT_METRICS {
                    left: m.left,
                    top: m.top,
                    width: m.width,
                    widthIncludingTrailingWhitespace: m.width_trailing,
                    height: m.height,
                    layoutWidth: m.layout_width,
                    layoutHeight: m.layout_height,
                    maxBidiReorderingDepth: m.bidi_depth,
                    lineCount: m.line_count,
                },
                // ponytail: our layout keeps no trailing-whitespace line, so this equals `height`.
                heightIncludingTrailingWhitespace: m.height,
            };
        }
        Ok(())
    }
    fn SetVerticalGlyphOrientation(&self, glyphorientation: DWRITE_VERTICAL_GLYPH_ORIENTATION) -> WResult<()> {
        self.vertical_glyph_orientation.set(glyphorientation.0);
        Ok(())
    }
    fn GetVerticalGlyphOrientation(&self) -> DWRITE_VERTICAL_GLYPH_ORIENTATION {
        DWRITE_VERTICAL_GLYPH_ORIENTATION(self.vertical_glyph_orientation.get())
    }
    fn SetLastLineWrapping(&self, islastlinewrappingenabled: BOOL) -> WResult<()> {
        self.last_line_wrapping.set(islastlinewrappingenabled.as_bool());
        Ok(())
    }
    fn GetLastLineWrapping(&self) -> BOOL {
        self.last_line_wrapping.get().into()
    }
    fn SetOpticalAlignment(&self, opticalalignment: DWRITE_OPTICAL_ALIGNMENT) -> WResult<()> {
        self.optical_alignment.set(opticalalignment.0);
        Ok(())
    }
    fn GetOpticalAlignment(&self) -> DWRITE_OPTICAL_ALIGNMENT {
        DWRITE_OPTICAL_ALIGNMENT(self.optical_alignment.get())
    }
    fn SetFontFallback(&self, fontfallback: Ref<IDWriteFontFallback>) -> WResult<()> {
        *self.font_fallback.borrow_mut() = fontfallback.as_ref().cloned();
        Ok(())
    }
    fn GetFontFallback(&self) -> WResult<IDWriteFontFallback> {
        // The explicitly set fallback, else the default system fallback (as native hands back).
        match self.font_fallback.borrow().clone() {
            Some(f) => Ok(f),
            None => Ok(super::fonts::FallbackObj { fonts: self.fonts.clone(), mappings: Vec::new() }.into()),
        }
    }
}

impl IDWriteTextLayout3_Impl for LayoutObj_Impl {
    fn InvalidateLayout(&self) -> WResult<()> {
        self.relayout();
        Ok(())
    }
    fn SetLineSpacing(&self, linespacingoptions: *const DWRITE_LINE_SPACING) -> WResult<()> {
        if !linespacingoptions.is_null() {
            let o = unsafe { *linespacingoptions };
            self.line_spacing.set(o);
            self.tf.set(layout::TextFormat { line_spacing: (o.method.0 as u8, o.height, o.baseline), ..self.tf.get() });
            self.relayout();
        }
        Ok(())
    }
    fn GetLineSpacing(&self, linespacingoptions: *mut DWRITE_LINE_SPACING) -> WResult<()> {
        if !linespacingoptions.is_null() {
            unsafe { *linespacingoptions = self.line_spacing.get() };
        }
        Ok(())
    }
    fn GetLineMetrics(&self, linemetrics: *mut DWRITE_LINE_METRICS1, maxlinecount: u32, actuallinecount: *mut u32) -> WResult<()> {
        let lay = self.layout.borrow();
        let lines = &lay.lines;
        unsafe { *actuallinecount = lines.len() as u32 };
        if linemetrics.is_null() || (maxlinecount as usize) < lines.len() {
            return Err(E_NOT_SUFFICIENT.into());
        }
        let dst = unsafe { slice::from_raw_parts_mut(linemetrics, lines.len()) };
        for (d, l) in dst.iter_mut().zip(lines) {
            *d = DWRITE_LINE_METRICS1 {
                Base: DWRITE_LINE_METRICS {
                    length: l.length,
                    trailingWhitespaceLength: l.trailing_ws,
                    newlineLength: l.newline,
                    height: l.height,
                    baseline: l.baseline,
                    isTrimmed: l.is_trimmed.into(),
                },
                // ponytail: our layout has no per-line leading-before/after spacing.
                leadingBefore: 0.0,
                leadingAfter: 0.0,
            };
        }
        Ok(())
    }
}

impl IDWriteTextLayout4_Impl for LayoutObj_Impl {
    fn SetFontAxisValues(&self, fontaxisvalues: *const DWRITE_FONT_AXIS_VALUE, fontaxisvaluecount: u32, _textrange: &DWRITE_TEXT_RANGE) -> WResult<()> {
        *self.layout_font_axes.borrow_mut() = read_axis_values(fontaxisvalues, fontaxisvaluecount);
        Ok(())
    }
    fn GetFontAxisValueCount(&self, _currentposition: u32) -> u32 {
        self.layout_font_axes.borrow().len() as u32
    }
    fn GetFontAxisValues(&self, _currentposition: u32, fontaxisvalues: *mut DWRITE_FONT_AXIS_VALUE, fontaxisvaluecount: u32, textrange: *mut DWRITE_TEXT_RANGE) -> WResult<()> {
        self.put_range(textrange);
        write_axis_values(&self.layout_font_axes.borrow(), fontaxisvalues, fontaxisvaluecount)
    }
    fn GetAutomaticFontAxes(&self) -> DWRITE_AUTOMATIC_FONT_AXES {
        DWRITE_AUTOMATIC_FONT_AXES(self.automatic_font_axes.get())
    }
    fn SetAutomaticFontAxes(&self, automaticfontaxes: DWRITE_AUTOMATIC_FONT_AXES) -> WResult<()> {
        self.automatic_font_axes.set(automaticfontaxes.0);
        Ok(())
    }
}

fn hit_metrics(h: &layout::HitMetrics, dx: f32, dy: f32) -> DWRITE_HIT_TEST_METRICS {
    DWRITE_HIT_TEST_METRICS {
        textPosition: h.pos,
        length: h.length,
        left: h.left + dx,
        top: h.top + dy,
        width: h.width,
        height: h.height,
        bidiLevel: h.bidi as u32,
        isText: h.is_text.into(),
        isTrimmed: h.is_trimmed.into(),
    }
}

impl LayoutObj {
    /// Write the uniform-format range to a per-range getter's out-pointer. The layout is
    /// single-format (no per-range overrides), so native reports the maximal range `[0, UINT32_MAX)`
    /// — the format attribute spans every position, not just the text length. No-op on a null pointer.
    fn put_range(&self, r: *mut DWRITE_TEXT_RANGE) {
        if !r.is_null() {
            unsafe { *r = DWRITE_TEXT_RANGE { startPosition: 0, length: u32::MAX } };
        }
    }
    /// The range an explicitly-ranged per-range getter reports: native echoes the range the matching
    /// Set* was last given, verbatim (VM-verified), so return the stored (start, length).
    fn put_stored_range(&self, r: *mut DWRITE_TEXT_RANGE, range: (u32, u32)) {
        if !r.is_null() {
            unsafe { *r = DWRITE_TEXT_RANGE { startPosition: range.0, length: range.1 } };
        }
    }

    /// Re-lay the string with the current trimming (`SetTrimming`); other opts stay default.
    fn relayout(&self) {
        let opts = layout::LayoutOpts { trimming: self.trimming.get(), measuring: self.measuring, pixels_per_dip: self.ppd, base_bold_delta_du: self.bold_delta_du, ..Default::default() };
        *self.layout.borrow_mut() = layout::layout_ex(&self.font, &self.tf.get(), &self.text, self.max_w, self.max_h, &opts);
    }

    /// UTF-16 code-unit position → UTF-8 byte offset into `text` (the layout keys faces by byte range).
    fn utf16_to_byte(&self, pos: u32) -> usize {
        if pos == 0 {
            return 0;
        }
        let mut cu = 0u32;
        for (b, ch) in self.text.char_indices() {
            if cu >= pos {
                return b;
            }
            cu += ch.len_utf16() as u32;
        }
        self.text.len()
    }

    /// Effective `(family, weight, style, stretch, size?)` at a UTF-16 position: the base format, then
    /// every covering per-range `SetFont*` in call order (later wins). `size` is `None` ⇒ the base em.
    fn eff_fmt(&self, pos: u32) -> (String, u16, u32, u16, Option<f32>) {
        let mut k = (self.family.clone(), self.weight, self.style, self.stretch, None);
        for (p, l, a) in self.fmt_ranges.borrow().iter() {
            if pos >= *p && pos < p.saturating_add(*l) {
                match a {
                    FmtAttr::Family(f) => k.0 = f.clone(),
                    FmtAttr::Weight(w) => k.1 = *w,
                    FmtAttr::Style(s) => k.2 = *s,
                    FmtAttr::Stretch(s) => k.3 = *s,
                    FmtAttr::Size(s) => k.4 = Some(*s),
                }
            }
        }
        k
    }

    /// Effective decoration bool at a position (base off, then each covering `SetUnderline`/
    /// `SetStrikethrough` in call order — a later `false` clears an earlier `true`).
    fn eff_bool(&self, ranges: &[(u32, u32, bool)], pos: u32) -> bool {
        let mut v = false;
        for (p, l, b) in ranges {
            if pos >= *p && pos < p.saturating_add(*l) {
                v = *b;
            }
        }
        v
    }

    /// The uniform run `[start, length)` around `pos` where `proj` is constant — what native's `Get*`
    /// getters report as the range the returned value spans.
    fn uniform_range<T: PartialEq>(&self, pos: u32, proj: impl Fn(u32) -> T) -> (u32, u32) {
        let n = self.text_u16_len;
        if n == 0 {
            return (0, u32::MAX);
        }
        let v = proj(pos.min(n - 1));
        let mut s = pos.min(n);
        while s > 0 && proj(s - 1) == v {
            s -= 1;
        }
        let mut e = pos.min(n);
        while e < n && proj(e) == v {
            e += 1;
        }
        (s, e - s)
    }

    /// Flatten the per-range overrides for Draw: contiguous positions whose effective face differs from
    /// the base become a `FontSpan` (byte range + loaded face + per-range em), and the underline /
    /// strikethrough overrides become byte ranges. A size-only run keeps the base family/weight/style/
    /// stretch (so it resolves the base face) and carries its em. Mirrors the core fallback-box flatten.
    #[allow(clippy::type_complexity)]
    fn build_spans(&self) -> (Vec<(usize, usize, Arc<Vec<u8>>, u32, u16, String, shaping::Font, Option<f32>)>, Vec<(usize, usize)>, Vec<(usize, usize)>) {
        let n = self.text_u16_len;
        let base = (self.family.clone(), self.weight, self.style, self.stretch, None::<f32>);
        let mut keys = vec![base.clone(); n as usize];
        for (p, l, a) in self.fmt_ranges.borrow().iter() {
            let end = p.saturating_add(*l).min(n) as usize;
            for k in keys.iter_mut().take(end).skip((*p).min(n) as usize) {
                match a {
                    FmtAttr::Family(f) => k.0 = f.clone(),
                    FmtAttr::Weight(w) => k.1 = *w,
                    FmtAttr::Style(s) => k.2 = *s,
                    FmtAttr::Stretch(s) => k.3 = *s,
                    FmtAttr::Size(s) => k.4 = Some(*s),
                }
            }
        }
        let mut runs = Vec::new();
        let mut p = 0u32;
        while p < n {
            let mut q = p + 1;
            while q < n && keys[q as usize] == keys[p as usize] {
                q += 1;
            }
            let k = &keys[p as usize];
            if *k != base {
                let (bs, be) = (self.utf16_to_byte(p), self.utf16_to_byte(q));
                if k.0 == base.0 && k.1 == base.1 && k.2 == base.2 && k.3 == base.3 {
                    // Size-only run: the base face at a per-range em (no re-resolve, so it works even
                    // with an empty collection — the em is all that differs).
                    if let Some(font) = shaping::Font::from_bytes((*self.bytes).clone(), self.index) {
                        runs.push((bs, be, self.bytes.clone(), self.index, 0, self.family.clone(), font, k.4));
                    }
                } else if let Some((bytes, idx, fam, sims)) = self.fonts.resolve_full(&k.0, k.1, k.2 as u8, k.3).or_else(|| {
                    // A family the collection lacks takes the stand-in (the system fallback's Segoe UI), as the base does.
                    self.fonts.resolve_full(&k.0, 400, 0, 5).is_none().then(|| self.fonts.stand_in().and_then(|f| self.fonts.resolve_full(&f, k.1, k.2 as u8, k.3))).flatten()
                }) {
                    if let Some(font) = shaping::Font::from_bytes((*bytes).clone(), idx) {
                        runs.push((bs, be, bytes, idx, sims, fam, font, k.4));
                    }
                }
            }
            p = q;
        }
        let bool_ranges = |rs: &[(u32, u32, bool)]| -> Vec<(usize, usize)> {
            let mut flags = vec![false; n as usize];
            for (p, l, b) in rs {
                let end = p.saturating_add(*l).min(n) as usize;
                for f in flags.iter_mut().take(end).skip((*p).min(n) as usize) {
                    *f = *b;
                }
            }
            let mut out = Vec::new();
            let mut i = 0usize;
            while i < flags.len() {
                if flags[i] {
                    let mut j = i + 1;
                    while j < flags.len() && flags[j] {
                        j += 1;
                    }
                    out.push((self.utf16_to_byte(i as u32), self.utf16_to_byte(j as u32)));
                    i = j;
                } else {
                    i += 1;
                }
            }
            out
        };
        let ul = bool_ranges(&self.ul_ranges.borrow());
        let stk = bool_ranges(&self.stk_ranges.borrow());
        (runs, ul, stk)
    }

    /// `GetOverhangMetrics`: how far the drawn ink overhangs each edge of the layout box
    /// (`maxWidth`×`maxHeight`), positive = outside. Unions every glyph's ink box (skrifa glyph
    /// bounds, design units scaled by em/upm, placed at its pen in layout space) and subtracts the
    /// box. Single-font (our layout's one face); a fallback layout would need its per-run faces.
    fn overhang(&self) -> DWRITE_OVERHANG_METRICS {
        let zero = DWRITE_OVERHANG_METRICS { left: 0.0, top: 0.0, right: 0.0, bottom: 0.0 };
        let Ok(font) = FontRef::from_index(&self.bytes, self.index) else { return zero };
        let upm = font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f32;
        let gm = font.glyph_metrics(Size::unscaled(), LocationRef::default());
        let scale = self.em / upm;
        let (mut l, mut t, mut r, mut b) = (f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        let lay = self.layout.borrow();
        for run in lay.runs.iter().filter_map(Run::glyph) {
            let mut pen = run.x;
            for i in 0..run.glyphs.len() {
                if let Some(bb) = gm.bounds(GlyphId::new(run.glyphs[i] as u32)) {
                    let (gx, gy) = (pen + run.offsets[i].0, run.y - run.offsets[i].1);
                    l = l.min(gx + bb.x_min * scale);
                    r = r.max(gx + bb.x_max * scale);
                    t = t.min(gy - bb.y_max * scale); // design y-up → device y-down
                    b = b.max(gy - bb.y_min * scale);
                }
                pen += run.advances[i];
            }
        }
        if !l.is_finite() {
            return zero;
        }
        let (lw, lh) = (lay.metrics.layout_width, lay.metrics.layout_height);
        DWRITE_OVERHANG_METRICS { left: -l, top: -t, right: r - lw, bottom: b - lh }
    }

    /// Build `DWRITE_GLYPH_RUN` + description for one laid-out glyph run and hand it to the renderer
    /// callback (the path `IDWriteTextLayout::Draw` drives; mirrors native's per-run emission). The run's
    /// face (`bytes`/`index`/`sims`/`family`) is its per-range face (the base face for a single-format
    /// layout), and the em is `g.em` (the base em, or a per-range `SetFontSize`).
    #[allow(clippy::too_many_arguments)]
    fn draw_run(&self, renderer: &IDWriteTextRenderer, ctx: *const c_void, g: &layout::GlyphRun, ox: f32, oy: f32, bytes: &Arc<Vec<u8>>, index: u32, sims: u16, family: &str) -> WResult<()> {
        let offsets: Vec<DWRITE_GLYPH_OFFSET> = g.offsets.iter().map(|&(a, asc)| DWRITE_GLYPH_OFFSET { advanceOffset: a, ascenderOffset: asc }).collect();
        // Native passes glyphOffsets = NULL when a run has no per-glyph offsets (the common case — a
        // run with no GPOS mark/cursive positioning), rather than an all-zero array. A text renderer
        // sees NULL, not zeros; match that so the emitted run is byte-identical (r11 query_metric).
        let offsets_ptr = if offsets.iter().all(|o| o.advanceOffset == 0.0 && o.ascenderOffset == 0.0) {
            core::ptr::null()
        } else {
            offsets.as_ptr()
        };
        let text_u16: Vec<u16> = g.text.encode_utf16().collect();
        let face3: IDWriteFontFace7 = FaceObj { bytes: bytes.clone(), index, family: family.to_string(), source: self.source.clone(), sims, axis_values: Vec::new() }.into();
        let face: IDWriteFontFace = face3.cast()?;
        let run = DWRITE_GLYPH_RUN {
            fontFace: ManuallyDrop::new(Some(face)),
            fontEmSize: g.em,
            // A tab's run carries its advance in the layout but reaches the renderer with no glyphs, as native's does.
            glyphCount: if g.text == "\t" { 0 } else { g.glyphs.len() as u32 },
            glyphIndices: g.glyphs.as_ptr(),
            glyphAdvances: g.advances.as_ptr(),
            glyphOffsets: offsets_ptr,
            isSideways: false.into(),
            bidiLevel: g.level as u32,
        };
        let desc = DWRITE_GLYPH_RUN_DESCRIPTION {
            localeName: PCWSTR::null(),
            string: PCWSTR(text_u16.as_ptr()),
            stringLength: text_u16.len() as u32,
            clusterMap: g.clusters.as_ptr(),
            textPosition: g.text_pos,
        };
        let r = unsafe { renderer.DrawGlyphRun(Some(ctx), ox + g.x, oy + g.y, DWRITE_MEASURING_MODE(self.measuring as i32), &run, &desc, None) };
        let _ = ManuallyDrop::into_inner(run.fontFace);
        r
    }

    /// Emit every glyph run — and underline/strikethrough rect — to the renderer. With per-range
    /// `SetFont*`/underline/strikethrough set, the string is re-laid-out through the core's per-range
    /// model (`FontSpan`s + decoration byte ranges) so each run carries its own face and em and the
    /// decorations their geometry; otherwise the cached single-format layout is used.
    fn emit_runs(&self, r: &IDWriteTextRenderer, ctx: *const c_void, ox: f32, oy: f32) -> WResult<()> {
        if self.fmt_ranges.borrow().is_empty() && self.ul_ranges.borrow().is_empty() && self.stk_ranges.borrow().is_empty() {
            let lay = self.layout.borrow();
            for run in &lay.runs {
                if let Run::Glyph(g) = run {
                    self.draw_run(r, ctx, g, ox, oy, &self.bytes, self.index, 0, &self.family)?;
                }
            }
            return Ok(());
        }
        let (spans, ul, stk) = self.build_spans();
        let font_spans: Vec<layout::FontSpan> = spans.iter().map(|(s, e, _, _, sims, _, f, em)| layout::FontSpan { start: *s, end: *e, font: f, em: *em, bold: sims & 1 != 0 }).collect();
        let opts = layout::LayoutOpts { fonts: &font_spans, underline: &ul, strikethrough: &stk, trimming: self.trimming.get(), measuring: self.measuring, pixels_per_dip: self.ppd, base_bold_delta_du: self.bold_delta_du, ..Default::default() };
        let laid = layout::layout_ex(&self.font, &self.tf.get(), &self.text, self.max_w, self.max_h, &opts);
        for run in &laid.runs {
            if let Run::Glyph(g) = run {
                let byte = self.utf16_to_byte(g.text_pos);
                match spans.iter().find(|(s, e, ..)| byte >= *s && byte < *e) {
                    Some((_, _, bytes, idx, sims, fam, _, _)) => self.draw_run(r, ctx, g, ox, oy, bytes, *idx, *sims, fam)?,
                    None => self.draw_run(r, ctx, g, ox, oy, &self.bytes, self.index, 0, &self.family)?,
                }
            }
        }
        // Underline / strikethrough rects, one per decorated run (native emits these after the glyph
        // runs via `DrawUnderline`/`DrawStrikethrough`). The renderer draws `[x, y+offset, x+width,
        // y+offset+thickness]`, so pass the run baseline `y = dec.y - dec.offset` and the font offset.
        let mode = DWRITE_MEASURING_MODE(self.measuring as i32);
        for d in &laid.decorations {
            let (bx, by) = (ox + d.x, oy + d.y - d.offset);
            if d.strike {
                let s = DWRITE_STRIKETHROUGH { width: d.width, thickness: d.thickness, offset: d.offset, readingDirection: DWRITE_READING_DIRECTION_LEFT_TO_RIGHT, flowDirection: DWRITE_FLOW_DIRECTION_TOP_TO_BOTTOM, localeName: PCWSTR::null(), measuringMode: mode };
                unsafe { r.DrawStrikethrough(Some(ctx), bx, by, &s, None) }?;
            } else {
                let u = DWRITE_UNDERLINE { width: d.width, thickness: d.thickness, offset: d.offset, runHeight: 0.0, readingDirection: DWRITE_READING_DIRECTION_LEFT_TO_RIGHT, flowDirection: DWRITE_FLOW_DIRECTION_TOP_TO_BOTTOM, localeName: PCWSTR::null(), measuringMode: mode };
                unsafe { r.DrawUnderline(Some(ctx), bx, by, &u, None) }?;
            }
        }
        Ok(())
    }
}

impl IDWriteTextLayout_Impl for LayoutObj_Impl {
    fn Draw(&self, clientdrawingcontext: *const c_void, renderer: Ref<IDWriteTextRenderer>, originx: f32, originy: f32) -> WResult<()> {
        let Some(r) = renderer.as_ref() else { return Ok(()) };
        self.emit_runs(r, clientdrawingcontext, originx, originy)
    }
    fn GetMetrics(&self, textmetrics: *mut DWRITE_TEXT_METRICS) -> WResult<()> {
        let lay = self.layout.borrow();
        let m = &lay.metrics;
        unsafe {
            *textmetrics = DWRITE_TEXT_METRICS {
                left: m.left,
                top: m.top,
                width: m.width,
                widthIncludingTrailingWhitespace: m.width_trailing,
                height: m.height,
                layoutWidth: m.layout_width,
                layoutHeight: m.layout_height,
                maxBidiReorderingDepth: m.bidi_depth,
                lineCount: m.line_count,
            };
        }
        Ok(())
    }
    fn GetLineMetrics(&self, linemetrics: *mut DWRITE_LINE_METRICS, maxlinecount: u32, actuallinecount: *mut u32) -> WResult<()> {
        let lay = self.layout.borrow();
        let lines = &lay.lines;
        unsafe { *actuallinecount = lines.len() as u32 };
        if linemetrics.is_null() || (maxlinecount as usize) < lines.len() {
            return Err(E_NOT_SUFFICIENT.into());
        }
        let dst = unsafe { slice::from_raw_parts_mut(linemetrics, lines.len()) };
        for (d, l) in dst.iter_mut().zip(lines) {
            *d = DWRITE_LINE_METRICS {
                length: l.length,
                trailingWhitespaceLength: l.trailing_ws,
                newlineLength: l.newline,
                height: l.height,
                baseline: l.baseline,
                isTrimmed: l.is_trimmed.into(),
            };
        }
        Ok(())
    }
    fn GetClusterMetrics(&self, clustermetrics: *mut DWRITE_CLUSTER_METRICS, maxclustercount: u32, actualclustercount: *mut u32) -> WResult<()> {
        let lay = self.layout.borrow();
        let clusters = &lay.clusters;
        unsafe { *actualclustercount = clusters.len() as u32 };
        if clustermetrics.is_null() || (maxclustercount as usize) < clusters.len() {
            return Err(E_NOT_SUFFICIENT.into());
        }
        let dst = unsafe { slice::from_raw_parts_mut(clustermetrics, clusters.len()) };
        for (d, c) in dst.iter_mut().zip(clusters) {
            *d = DWRITE_CLUSTER_METRICS { width: c.width, length: c.length, _bitfield: c.bits };
        }
        Ok(())
    }
    fn HitTestPoint(&self, pointx: f32, pointy: f32, istrailinghit: *mut BOOL, isinside: *mut BOOL, hittestmetrics: *mut DWRITE_HIT_TEST_METRICS) -> WResult<()> {
        let h = self.layout.borrow().hit_test_point(pointx, pointy);
        unsafe {
            *istrailinghit = h.is_trailing.into();
            *isinside = h.is_inside.into();
            *hittestmetrics = hit_metrics(&h, 0.0, 0.0);
        }
        Ok(())
    }
    fn HitTestTextPosition(&self, textposition: u32, _istrailinghit: BOOL, pointx: *mut f32, pointy: *mut f32, hittestmetrics: *mut DWRITE_HIT_TEST_METRICS) -> WResult<()> {
        let c = self.layout.borrow().hit_test_text_position(textposition);
        unsafe {
            *pointx = c.x;
            *pointy = c.y;
            *hittestmetrics = DWRITE_HIT_TEST_METRICS {
                textPosition: textposition,
                length: 0,
                left: c.left,
                top: c.top,
                width: c.width,
                height: c.height,
                bidiLevel: c.bidi as u32,
                // A caret in untrimmed text is always "text" for every position 0..=len, INCLUDING an
                // empty layout (native HitTestTextPosition(0) on "" returns isText=TRUE — r11 10 seeds).
                // A trimmed layout would need the layout's trimmed-span flag (no such case yet).
                isText: true.into(),
                isTrimmed: false.into(),
            };
        }
        Ok(())
    }
    fn HitTestTextRange(&self, textposition: u32, textlength: u32, originx: f32, originy: f32, hittestmetrics: *mut DWRITE_HIT_TEST_METRICS, maxhittestmetricscount: u32, actualhittestmetricscount: *mut u32) -> WResult<()> {
        let boxes = self.layout.borrow().hit_test_text_range(textposition, textlength);
        unsafe { *actualhittestmetricscount = boxes.len() as u32 };
        if hittestmetrics.is_null() || (maxhittestmetricscount as usize) < boxes.len() {
            return Err(E_NOT_SUFFICIENT.into());
        }
        let dst = unsafe { slice::from_raw_parts_mut(hittestmetrics, boxes.len()) };
        for (d, b) in dst.iter_mut().zip(&boxes) {
            *d = hit_metrics(b, originx, originy);
        }
        Ok(())
    }
    // --- the rest: not exercised by the text corpus -------------------------------------------
    fn SetMaxWidth(&self, _maxwidth: f32) -> WResult<()> {
        Ok(())
    }
    fn SetMaxHeight(&self, _maxheight: f32) -> WResult<()> {
        Ok(())
    }
    fn SetFontCollection(&self, _c: Ref<IDWriteFontCollection>, _r: &DWRITE_TEXT_RANGE) -> WResult<()> {
        Ok(())
    }
    fn SetFontFamilyName(&self, n: &PCWSTR, r: &DWRITE_TEXT_RANGE) -> WResult<()> {
        let name = unsafe { n.to_string() }.unwrap_or_default();
        self.fmt_ranges.borrow_mut().push((r.startPosition, r.length, FmtAttr::Family(name)));
        Ok(())
    }
    fn SetFontWeight(&self, w: DWRITE_FONT_WEIGHT, r: &DWRITE_TEXT_RANGE) -> WResult<()> {
        self.fmt_ranges.borrow_mut().push((r.startPosition, r.length, FmtAttr::Weight(w.0 as u16)));
        Ok(())
    }
    fn SetFontStyle(&self, s: DWRITE_FONT_STYLE, r: &DWRITE_TEXT_RANGE) -> WResult<()> {
        self.fmt_ranges.borrow_mut().push((r.startPosition, r.length, FmtAttr::Style(s.0 as u32)));
        Ok(())
    }
    fn SetFontStretch(&self, s: DWRITE_FONT_STRETCH, r: &DWRITE_TEXT_RANGE) -> WResult<()> {
        self.fmt_ranges.borrow_mut().push((r.startPosition, r.length, FmtAttr::Stretch(s.0 as u16)));
        Ok(())
    }
    fn SetFontSize(&self, s: f32, r: &DWRITE_TEXT_RANGE) -> WResult<()> {
        // native's `SetFontSize` validates exactly two things: it rejects a
        // non-positive size (`s <= 0.0`, so NaN/+inf/a huge finite size all pass — `NaN <= 0` is false)
        // and a `startPosition + length` that overflows u32; otherwise it applies the size and returns
        // S_OK. It does NOT check finiteness or a device-em limit — the GDI 2^15 em cap is a draw-time
        // `GetGdiCompatibleGlyphMetrics` check, not a SetFontSize one, and a later overlapping call can
        // overwrite a huge range before any glyph is measured (fuzz r21 26772). Match it exactly.
        if s <= 0.0 {
            return Err(super::E_INVALIDARG.into());
        }
        if r.startPosition.checked_add(r.length).is_none() {
            return Err(super::E_INVALIDARG.into());
        }
        self.fmt_ranges.borrow_mut().push((r.startPosition, r.length, FmtAttr::Size(s)));
        Ok(())
    }
    fn SetUnderline(&self, u: BOOL, r: &DWRITE_TEXT_RANGE) -> WResult<()> {
        self.ul_ranges.borrow_mut().push((r.startPosition, r.length, u.as_bool()));
        Ok(())
    }
    fn SetStrikethrough(&self, s: BOOL, r: &DWRITE_TEXT_RANGE) -> WResult<()> {
        self.stk_ranges.borrow_mut().push((r.startPosition, r.length, s.as_bool()));
        Ok(())
    }
    fn SetDrawingEffect(&self, e: Ref<windows_core::IUnknown>, _r: &DWRITE_TEXT_RANGE) -> WResult<()> {
        *self.drawing_effect.borrow_mut() = e.as_ref().cloned();
        Ok(())
    }
    fn SetInlineObject(&self, o: Ref<IDWriteInlineObject>, _r: &DWRITE_TEXT_RANGE) -> WResult<()> {
        *self.inline_object.borrow_mut() = o.as_ref().cloned();
        Ok(())
    }
    fn SetTypography(&self, t: Ref<IDWriteTypography>, _r: &DWRITE_TEXT_RANGE) -> WResult<()> {
        *self.typography.borrow_mut() = t.as_ref().cloned();
        Ok(())
    }
    fn SetLocaleName(&self, _n: &PCWSTR, _r: &DWRITE_TEXT_RANGE) -> WResult<()> {
        Ok(())
    }
    fn GetMaxWidth(&self) -> f32 {
        self.max_w
    }
    fn GetMaxHeight(&self) -> f32 {
        self.max_h
    }
    fn GetFontCollection(&self, _p: u32, c: windows_core::OutRef<IDWriteFontCollection>, r: *mut DWRITE_TEXT_RANGE) -> WResult<()> {
        self.put_range(r);
        c.write(Some(super::fonts::collection(self.fonts.clone())))
    }
    fn GetFontFamilyNameLength(&self, p: u32, n: *mut u32, r: *mut DWRITE_TEXT_RANGE) -> WResult<()> {
        unsafe { *n = self.eff_fmt(p).0.encode_utf16().count() as u32 };
        self.put_stored_range(r, self.uniform_range(p, |q| self.eff_fmt(q).0));
        Ok(())
    }
    fn GetFontFamilyName(&self, p: u32, n: PWSTR, s: u32, r: *mut DWRITE_TEXT_RANGE) -> WResult<()> {
        let fam = self.eff_fmt(p).0;
        self.put_stored_range(r, self.uniform_range(p, |q| self.eff_fmt(q).0));
        super::write_wide(&fam, n, s)
    }
    fn GetFontWeight(&self, p: u32, w: *mut DWRITE_FONT_WEIGHT, r: *mut DWRITE_TEXT_RANGE) -> WResult<()> {
        unsafe { *w = DWRITE_FONT_WEIGHT(self.eff_fmt(p).1 as i32) };
        self.put_stored_range(r, self.uniform_range(p, |q| self.eff_fmt(q).1));
        Ok(())
    }
    fn GetFontStyle(&self, p: u32, s: *mut DWRITE_FONT_STYLE, r: *mut DWRITE_TEXT_RANGE) -> WResult<()> {
        unsafe { *s = DWRITE_FONT_STYLE(self.eff_fmt(p).2 as i32) };
        self.put_stored_range(r, self.uniform_range(p, |q| self.eff_fmt(q).2));
        Ok(())
    }
    fn GetFontStretch(&self, p: u32, s: *mut DWRITE_FONT_STRETCH, r: *mut DWRITE_TEXT_RANGE) -> WResult<()> {
        unsafe { *s = DWRITE_FONT_STRETCH(self.eff_fmt(p).3 as i32) };
        self.put_stored_range(r, self.uniform_range(p, |q| self.eff_fmt(q).3));
        Ok(())
    }
    fn GetFontSize(&self, p: u32, s: *mut f32, r: *mut DWRITE_TEXT_RANGE) -> WResult<()> {
        unsafe { *s = self.eff_fmt(p).4.unwrap_or(self.em) };
        self.put_stored_range(r, self.uniform_range(p, |q| self.eff_fmt(q).4.map(f32::to_bits)));
        Ok(())
    }
    fn GetUnderline(&self, p: u32, u: *mut BOOL, r: *mut DWRITE_TEXT_RANGE) -> WResult<()> {
        let ul = self.ul_ranges.borrow();
        unsafe { *u = self.eff_bool(&ul, p).into() };
        self.put_stored_range(r, self.uniform_range(p, |q| self.eff_bool(&ul, q)));
        Ok(())
    }
    fn GetStrikethrough(&self, p: u32, s: *mut BOOL, r: *mut DWRITE_TEXT_RANGE) -> WResult<()> {
        let stk = self.stk_ranges.borrow();
        unsafe { *s = self.eff_bool(&stk, p).into() };
        self.put_stored_range(r, self.uniform_range(p, |q| self.eff_bool(&stk, q)));
        Ok(())
    }
    fn GetDrawingEffect(&self, _p: u32, e: windows_core::OutRef<windows_core::IUnknown>, r: *mut DWRITE_TEXT_RANGE) -> WResult<()> {
        self.put_range(r);
        e.write(self.drawing_effect.borrow().clone())
    }
    fn GetInlineObject(&self, _p: u32, o: windows_core::OutRef<IDWriteInlineObject>, r: *mut DWRITE_TEXT_RANGE) -> WResult<()> {
        self.put_range(r);
        o.write(self.inline_object.borrow().clone())
    }
    fn GetTypography(&self, _p: u32, t: windows_core::OutRef<IDWriteTypography>, r: *mut DWRITE_TEXT_RANGE) -> WResult<()> {
        self.put_range(r);
        t.write(self.typography.borrow().clone())
    }
    fn GetLocaleNameLength(&self, _p: u32, n: *mut u32, r: *mut DWRITE_TEXT_RANGE) -> WResult<()> {
        unsafe { *n = self.locale.encode_utf16().count() as u32 };
        self.put_range(r);
        Ok(())
    }
    fn GetLocaleName(&self, _p: u32, n: PWSTR, s: u32, r: *mut DWRITE_TEXT_RANGE) -> WResult<()> {
        self.put_range(r);
        super::write_wide(&self.locale, n, s)
    }
    fn GetOverhangMetrics(&self) -> WResult<DWRITE_OVERHANG_METRICS> {
        Ok(self.overhang())
    }
    fn DetermineMinWidth(&self) -> WResult<f32> {
        Ok(self.layout.borrow().min_width())
    }
}

// The layout's inherited IDWriteTextFormat setters change the LAYOUT's format and re-lay the string,
// exactly as creating a new layout from a format with those values would — native honours a
// SetTextAlignment/SetParagraphAlignment/… called on the layout after CreateTextLayout (Paint.NET does
// this, then draws through IDWriteTextLayout::Draw). The getters report the layout's current values.
impl IDWriteTextFormat_Impl for LayoutObj_Impl {
    fn SetTextAlignment(&self, a: DWRITE_TEXT_ALIGNMENT) -> WResult<()> {
        let mut t = self.tf.get();
        t.align = a.0 as u8;
        self.tf.set(t);
        self.relayout();
        Ok(())
    }
    fn SetParagraphAlignment(&self, p: DWRITE_PARAGRAPH_ALIGNMENT) -> WResult<()> {
        let mut t = self.tf.get();
        t.para = p.0 as u8;
        self.tf.set(t);
        self.relayout();
        Ok(())
    }
    fn SetWordWrapping(&self, w: DWRITE_WORD_WRAPPING) -> WResult<()> {
        let mut t = self.tf.get();
        t.wrap = w.0 as u8;
        self.tf.set(t);
        self.relayout();
        Ok(())
    }
    fn SetReadingDirection(&self, r: DWRITE_READING_DIRECTION) -> WResult<()> {
        self.reading.set(r.0 as u32);
        let mut t = self.tf.get();
        t.base_rtl = r.0 & 1 == 1;
        self.tf.set(t);
        self.relayout();
        Ok(())
    }
    fn SetFlowDirection(&self, f: DWRITE_FLOW_DIRECTION) -> WResult<()> {
        self.flow.set(f.0 as u32);
        Ok(())
    }
    fn SetIncrementalTabStop(&self, t: f32) -> WResult<()> {
        self.tab.set(t);
        Ok(())
    }
    fn SetTrimming(&self, o: *const DWRITE_TRIMMING, s: Ref<IDWriteInlineObject>) -> WResult<()> {
        let (upright_sign, upright_bold) = super::fonts::upright_sign(&self.fonts, &self.locale, self.weight, self.style != 0);
        self.trimming.set(unsafe { o.as_ref() }.map(|t| layout::Trimming {
            granularity: t.granularity.0 as u8,
            delimiter: t.delimiter,
            delimiter_count: t.delimiterCount,
            ellipsis: s.as_ref().is_some(),
            // ponytail: the layout's own format stands in for the sign's (the usual case).
            upright_sign,
            upright_bold,
            sign: None,
        }));
        self.relayout();
        Ok(())
    }
    fn SetLineSpacing(&self, m: DWRITE_LINE_SPACING_METHOD, s: f32, b: f32) -> WResult<()> {
        let prev = self.line_spacing.get();
        self.line_spacing.set(DWRITE_LINE_SPACING { method: m, height: s, baseline: b, ..prev });
        self.tf.set(layout::TextFormat { line_spacing: (m.0 as u8, s, b), ..self.tf.get() });
        self.relayout();
        Ok(())
    }
    fn GetTextAlignment(&self) -> DWRITE_TEXT_ALIGNMENT {
        DWRITE_TEXT_ALIGNMENT(self.tf.get().align as i32)
    }
    fn GetParagraphAlignment(&self) -> DWRITE_PARAGRAPH_ALIGNMENT {
        DWRITE_PARAGRAPH_ALIGNMENT(self.tf.get().para as i32)
    }
    fn GetWordWrapping(&self) -> DWRITE_WORD_WRAPPING {
        DWRITE_WORD_WRAPPING(self.tf.get().wrap as i32)
    }
    fn GetReadingDirection(&self) -> DWRITE_READING_DIRECTION {
        DWRITE_READING_DIRECTION(self.reading.get() as i32)
    }
    fn GetFlowDirection(&self) -> DWRITE_FLOW_DIRECTION {
        DWRITE_FLOW_DIRECTION(self.flow.get() as i32)
    }
    fn GetIncrementalTabStop(&self) -> f32 {
        self.tab.get()
    }
    fn GetTrimming(&self, o: *mut DWRITE_TRIMMING, _s: windows_core::OutRef<IDWriteInlineObject>) -> WResult<()> {
        if let Some(t) = self.trimming.get() {
            unsafe { *o = DWRITE_TRIMMING { granularity: DWRITE_TRIMMING_GRANULARITY(t.granularity as i32), delimiter: t.delimiter, delimiterCount: t.delimiter_count } };
        }
        Ok(())
    }
    fn GetLineSpacing(&self, m: *mut DWRITE_LINE_SPACING_METHOD, s: *mut f32, b: *mut f32) -> WResult<()> {
        let ls = self.line_spacing.get();
        unsafe { *m = ls.method; *s = ls.height; *b = ls.baseline; }
        Ok(())
    }
    fn GetFontCollection(&self) -> WResult<IDWriteFontCollection> {
        Ok(super::fonts::collection(self.fonts.clone()))
    }
    fn GetFontFamilyNameLength(&self) -> u32 {
        self.family.encode_utf16().count() as u32
    }
    fn GetFontFamilyName(&self, n: PWSTR, s: u32) -> WResult<()> {
        super::write_wide(&self.family, n, s)
    }
    fn GetFontWeight(&self) -> DWRITE_FONT_WEIGHT {
        DWRITE_FONT_WEIGHT(self.weight as i32)
    }
    fn GetFontStyle(&self) -> DWRITE_FONT_STYLE {
        DWRITE_FONT_STYLE(self.style as i32)
    }
    fn GetFontStretch(&self) -> DWRITE_FONT_STRETCH {
        DWRITE_FONT_STRETCH(self.stretch as i32)
    }
    fn GetFontSize(&self) -> f32 {
        self.em
    }
    fn GetLocaleNameLength(&self) -> u32 {
        self.locale.encode_utf16().count() as u32
    }
    fn GetLocaleName(&self, n: PWSTR, s: u32) -> WResult<()> {
        super::write_wide(&self.locale, n, s)
    }
}

// --- IDWriteInlineObject: the ellipsis trimming sign (CreateEllipsisTrimmingSign) -------------

#[implement(IDWriteInlineObject)]
pub(crate) struct EllipsisObj {
    source: FontSource,
    bytes: Arc<Vec<u8>>,
    index: u32,
    family: String,
    em: f32,
    sims: u16,
}

/// `CreateEllipsisTrimmingSign(textFormat)`: a "…" inline object in the format's font/size, the sign
/// a layout draws when a line is trimmed (`SetTrimming`). It keeps the format face's simulations: a bold one
/// widens the sign's glyph and both draw on the simulated face (VM: Impact weight 700 measures and draws
/// (1133 + 41)·em/upem with the bold simulation, its italic draws oblique at the regular width).
pub(crate) fn make_ellipsis(fmt: Ref<IDWriteTextFormat>) -> WResult<IDWriteInlineObject> {
    let fo = fmt.as_ref().ok_or(E_NOTIMPL)?.cast_object_ref::<FormatObj>().map_err(|_| windows_core::Error::from(E_NOTIMPL))?;
    Ok(EllipsisObj { source: fo.source.clone(), bytes: fo.bytes.clone(), index: fo.index, family: fo.family.clone(), em: fo.em, sims: fo.sims }.into())
}

impl EllipsisObj {
    fn font(&self) -> Option<shaping::Font> {
        shaping::Font::from_bytes((*self.bytes).clone(), self.index)
    }

    /// The bold simulation's design-unit widening of the sign's glyph `id`: round(upem/50) on an inked glyph.
    fn bold_du(&self, font: &shaping::Font, id: u16) -> i32 {
        if self.sims & DWRITE_FONT_SIMULATIONS_BOLD.0 as u16 != 0 && font.has_outline(id) { (font.upm as f32 / 50.0).round() as i32 } else { 0 }
    }
}

impl IDWriteInlineObject_Impl for EllipsisObj_Impl {
    fn Draw(&self, ctx: *const c_void, renderer: Ref<IDWriteTextRenderer>, originx: f32, originy: f32, _issideways: BOOL, _isrtl: BOOL, _effect: Ref<windows_core::IUnknown>) -> WResult<()> {
        let (Some(r), Some(font)) = (renderer.as_ref(), self.font()) else { return Ok(()) };
        let glyphs = shaping::shape(&font, "\u{2026}", false, None, None);
        let scale = self.em / font.upm as f32;
        let ids: Vec<u16> = glyphs.iter().map(|g| g.id).collect();
        let advances: Vec<f32> = glyphs.iter().enumerate().map(|(i, g)| (g.advance + if i == 0 { self.bold_du(&font, g.id) } else { 0 }) as f32 * scale).collect();
        let offsets: Vec<DWRITE_GLYPH_OFFSET> = glyphs.iter().map(|g| DWRITE_GLYPH_OFFSET { advanceOffset: g.dx as f32 * scale, ascenderOffset: g.dy as f32 * scale }).collect();
        let clusters = vec![0u16; 1];
        let text_u16: Vec<u16> = "\u{2026}".encode_utf16().collect();
        let face3: IDWriteFontFace7 = FaceObj { bytes: self.bytes.clone(), index: self.index, family: self.family.clone(), source: self.source.clone(), sims: self.sims, axis_values: Vec::new() }.into();
        let face: IDWriteFontFace = face3.cast()?;
        let run = DWRITE_GLYPH_RUN {
            fontFace: ManuallyDrop::new(Some(face)),
            fontEmSize: self.em,
            glyphCount: ids.len() as u32,
            glyphIndices: ids.as_ptr(),
            glyphAdvances: advances.as_ptr(),
            glyphOffsets: offsets.as_ptr(),
            isSideways: false.into(),
            bidiLevel: 0,
        };
        let desc = DWRITE_GLYPH_RUN_DESCRIPTION { localeName: PCWSTR::null(), string: PCWSTR(text_u16.as_ptr()), stringLength: text_u16.len() as u32, clusterMap: clusters.as_ptr(), textPosition: 0 };
        let res = unsafe { r.DrawGlyphRun(Some(ctx), originx, originy, DWRITE_MEASURING_MODE(0), &run, &desc, None) };
        let _ = ManuallyDrop::into_inner(run.fontFace);
        res
    }
    fn GetMetrics(&self) -> WResult<DWRITE_INLINE_OBJECT_METRICS> {
        // The sign's width is the font's horizontal-ellipsis advance; DirectWrite reports height and
        // baseline 0 for it (the line it sits on supplies the vertical metrics).
        let width = self
            .font()
            .map(|f| {
                let bold = shaping::shape(&f, "\u{2026}", false, None, None).first().map_or(0, |g| self.bold_du(&f, g.id));
                layout::ellipsis_width_bold(&f, &layout::TextFormat { em: self.em, base_rtl: false, align: 0, wrap: 0, para: 0, flow: 0, line_spacing: (0, 0.0, 0.0) }, bold)
            })
            .unwrap_or(0.0);
        Ok(DWRITE_INLINE_OBJECT_METRICS { width, height: 0.0, baseline: 0.0, supportsSideways: false.into() })
    }
    fn GetOverhangMetrics(&self) -> WResult<DWRITE_OVERHANG_METRICS> {
        Ok(DWRITE_OVERHANG_METRICS { left: 0.0, top: 0.0, right: 0.0, bottom: 0.0 })
    }
    fn GetBreakConditions(&self, before: *mut DWRITE_BREAK_CONDITION, after: *mut DWRITE_BREAK_CONDITION) -> WResult<()> {
        unsafe {
            *before = DWRITE_BREAK_CONDITION_NEUTRAL;
            *after = DWRITE_BREAK_CONDITION_NEUTRAL;
        }
        Ok(())
    }
}

#[cfg(test)]
mod layout_tests {
    //! `IDWriteTextLayout1-4` state round-trips. Self-contained: builds a `LayoutObj` directly over
    //! the committed `colrtest.ttf` (a real glyf/cmap font that shapes), so no system font corpus.
    use super::*;

    #[test]
    fn missing_family_without_segoe_ui_takes_the_first_family() {
        // A collection without Segoe UI (a machine without Microsoft's fonts): a format naming a missing
        // family still succeeds, laid out with the collection's first family, keeping the requested name.
        let bytes = Arc::new(std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../d2d/src/text/color/testdata/colrtest.ttf")).unwrap());
        let fonts = SystemFonts::from_sources(vec![FontSource::Bytes(bytes.clone())]);
        let fmt = make_format(&fonts, "NoSuchFont", 400, 0, 5, 16.0, "en-us").expect("a missing family still makes a format");
        let fo = fmt.cast_object_ref::<FormatObj>().unwrap();
        assert_eq!(fo.family, "NoSuchFont");
        assert!(Arc::ptr_eq(&fo.bytes, &bytes) || *fo.bytes == *bytes);
    }

    fn layout4() -> IDWriteTextLayout4 {
        let bytes = Arc::new(std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../d2d/src/text/color/testdata/colrtest.ttf")).unwrap());
        let font = shaping::Font::from_bytes((*bytes).clone(), 0).unwrap();
        let tf = layout::TextFormat { em: 16.0, base_rtl: false, align: 0, wrap: 0, para: 0, flow: 0, line_spacing: (0, 0.0, 0.0) };
        let laid = layout::layout_ex(&font, &tf, "Ab", 1000.0, 1000.0, &layout::LayoutOpts::default());
        LayoutObj {
            layout: RefCell::new(laid),
            font,
            tf: Cell::new(tf),
            reading: Cell::new(0),
            flow: Cell::new(0),
            tab: Cell::new(16.0 * 4.0),
            text: "Ab".into(),
            max_w: 1000.0,
            max_h: 1000.0,
            trimming: Cell::new(None),
            text_u16_len: 2,
            source: FontSource::Bytes(bytes.clone()),
            bytes,
            index: 0,
            family: "T".into(),
            em: 16.0,
            fonts: SystemFonts::from_sources(vec![]),
            measuring: 0,
            ppd: 1.0,
            bold_delta_du: 0,
            weight: 400,
            style: 0,
            stretch: 5,
            locale: "en-us".into(),
            fmt_ranges: RefCell::new(Vec::new()),
            ul_ranges: RefCell::new(Vec::new()),
            stk_ranges: RefCell::new(Vec::new()),
            drawing_effect: RefCell::new(None),
            inline_object: RefCell::new(None),
            typography: RefCell::new(None),
            pair_kerning: Cell::new(false),
            char_spacing: Cell::new((0.0, 0.0, 0.0)),
            pair_kerning_range: Cell::new((0, u32::MAX)),
            char_spacing_range: Cell::new((0, u32::MAX)),
            vertical_glyph_orientation: Cell::new(0),
            last_line_wrapping: Cell::new(true),
            optical_alignment: Cell::new(0),
            font_fallback: RefCell::new(None),
            line_spacing: Cell::new(DWRITE_LINE_SPACING::default()),
            layout_font_axes: RefCell::new(Vec::new()),
            automatic_font_axes: Cell::new(0),
        }
        .into()
    }

    #[test]
    fn textlayout_leaf_roundtrips() {
        let l = layout4();
        let range = DWRITE_TEXT_RANGE { startPosition: 0, length: u32::MAX };
        unsafe {
            assert!(l.GetLastLineWrapping().as_bool(), "native default last-line-wrapping is on");

            l.SetPairKerning(true, range).unwrap();
            let mut pk = BOOL(0);
            let mut r = DWRITE_TEXT_RANGE::default();
            l.GetPairKerning(0, &mut pk, Some(&mut r)).unwrap();
            assert!(pk.as_bool());
            assert_eq!(r.length, u32::MAX, "uniform format spans [0, UINT32_MAX)");

            l.SetCharacterSpacing(1.0, 2.0, 3.0, range).unwrap();
            let (mut a, mut b, mut c) = (0.0f32, 0.0f32, 0.0f32);
            l.GetCharacterSpacing(0, &mut a, &mut b, &mut c, Some(&mut r)).unwrap();
            assert_eq!((a, b, c), (1.0, 2.0, 3.0));

            l.SetVerticalGlyphOrientation(DWRITE_VERTICAL_GLYPH_ORIENTATION_STACKED).unwrap();
            assert_eq!(l.GetVerticalGlyphOrientation(), DWRITE_VERTICAL_GLYPH_ORIENTATION_STACKED);

            l.SetLastLineWrapping(false).unwrap();
            assert!(!l.GetLastLineWrapping().as_bool());

            l.SetOpticalAlignment(DWRITE_OPTICAL_ALIGNMENT_NO_SIDE_BEARINGS).unwrap();
            assert_eq!(l.GetOpticalAlignment(), DWRITE_OPTICAL_ALIGNMENT_NO_SIDE_BEARINGS);

            assert!(l.GetFontFallback().is_ok(), "default fallback is the system fallback, not null");

            let ls = DWRITE_LINE_SPACING { method: DWRITE_LINE_SPACING_METHOD_UNIFORM, height: 20.0, baseline: 16.0, leadingBefore: 0.0, fontLineGapUsage: DWRITE_FONT_LINE_GAP_USAGE_DEFAULT };
            l.SetLineSpacing(&ls).unwrap();
            let mut got = DWRITE_LINE_SPACING::default();
            l.GetLineSpacing(&mut got).unwrap();
            assert_eq!(got.height, 20.0);

            let axes = [DWRITE_FONT_AXIS_VALUE { axisTag: DWRITE_FONT_AXIS_TAG_WEIGHT, value: 650.0 }];
            l.SetFontAxisValues(&axes, range).unwrap();
            assert_eq!(l.GetFontAxisValueCount(0), 1);
            let mut out = [DWRITE_FONT_AXIS_VALUE::default(); 1];
            l.GetFontAxisValues(0, &mut out, Some(&mut r)).unwrap();
            assert_eq!(out[0].value, 650.0);

            l.SetAutomaticFontAxes(DWRITE_AUTOMATIC_FONT_AXES_OPTICAL_SIZE).unwrap();
            assert_eq!(l.GetAutomaticFontAxes(), DWRITE_AUTOMATIC_FONT_AXES_OPTICAL_SIZE);

            let l2: IDWriteTextLayout2 = l.cast().unwrap();
            let mut tm1 = DWRITE_TEXT_METRICS1::default();
            l2.GetMetrics(&mut tm1).unwrap();
            assert_eq!(tm1.heightIncludingTrailingWhitespace, tm1.Base.height);
        }
    }

    /// Records the em of every glyph run a layout emits + the underline count, to probe that per-range
    /// sizes and the underline reach Draw.
    #[implement(IDWriteTextRenderer)]
    struct CapRenderer {
        ems: std::rc::Rc<RefCell<Vec<f32>>>,
        underlines: std::rc::Rc<std::cell::Cell<u32>>,
    }
    impl IDWritePixelSnapping_Impl for CapRenderer_Impl {
        fn IsPixelSnappingDisabled(&self, _c: *const c_void) -> WResult<BOOL> {
            Ok(true.into())
        }
        fn GetCurrentTransform(&self, _c: *const c_void, t: *mut DWRITE_MATRIX) -> WResult<()> {
            unsafe { *t = DWRITE_MATRIX { m11: 1.0, m12: 0.0, m21: 0.0, m22: 1.0, dx: 0.0, dy: 0.0 } };
            Ok(())
        }
        fn GetPixelsPerDip(&self, _c: *const c_void) -> WResult<f32> {
            Ok(1.0)
        }
    }
    impl IDWriteTextRenderer_Impl for CapRenderer_Impl {
        fn DrawGlyphRun(&self, _c: *const c_void, _x: f32, _y: f32, _m: DWRITE_MEASURING_MODE, run: *const DWRITE_GLYPH_RUN, _d: *const DWRITE_GLYPH_RUN_DESCRIPTION, _e: Ref<windows_core::IUnknown>) -> WResult<()> {
            self.ems.borrow_mut().push(unsafe { (*run).fontEmSize });
            Ok(())
        }
        fn DrawUnderline(&self, _c: *const c_void, _x: f32, _y: f32, _u: *const DWRITE_UNDERLINE, _e: Ref<windows_core::IUnknown>) -> WResult<()> {
            self.underlines.set(self.underlines.get() + 1);
            Ok(())
        }
        fn DrawStrikethrough(&self, _c: *const c_void, _x: f32, _y: f32, _s: *const DWRITE_STRIKETHROUGH, _e: Ref<windows_core::IUnknown>) -> WResult<()> {
            Ok(())
        }
        fn DrawInlineObject(&self, _c: *const c_void, _x: f32, _y: f32, _o: Ref<IDWriteInlineObject>, _sw: BOOL, _rtl: BOOL, _e: Ref<windows_core::IUnknown>) -> WResult<()> {
            Ok(())
        }
    }

    /// Per-range `SetFont*`/`SetUnderline` feed the core per-range model: the `Get*(position)` getters
    /// read back the effective value + its uniform range, and `Draw` emits each range's run at its own
    /// em. (The layout is over the no-corpus `colrtest.ttf`, so this exercises per-range SIZE + the
    /// getters + the per-run emit; a per-range FACE change needs the system collection — the gate's
    /// `D2D_FONTS` FFI probe covers that.)
    #[test]
    fn per_range_size_getters_and_render() {
        let l = layout4(); // text "Ab", base em 16, weight 400
        unsafe {
            // SetFontSize(32) on [1,1) ('b'); the getter reads base before it and the override at it.
            l.SetFontSize(32.0, DWRITE_TEXT_RANGE { startPosition: 1, length: 1 }).unwrap();
            let (mut sz, mut r) = (0.0f32, DWRITE_TEXT_RANGE::default());
            l.GetFontSize(0, &mut sz, Some(&mut r)).unwrap();
            assert_eq!((sz, r.startPosition, r.length), (16.0, 0, 1));
            l.GetFontSize(1, &mut sz, Some(&mut r)).unwrap();
            assert_eq!((sz, r.startPosition, r.length), (32.0, 1, 1));

            // SetFontWeight(BOLD) on [0,1); bold at 0, normal at 1.
            l.SetFontWeight(DWRITE_FONT_WEIGHT_BOLD, DWRITE_TEXT_RANGE { startPosition: 0, length: 1 }).unwrap();
            let (mut w, mut wr) = (DWRITE_FONT_WEIGHT_NORMAL, DWRITE_TEXT_RANGE::default());
            l.GetFontWeight(0, &mut w, Some(&mut wr)).unwrap();
            assert_eq!((w, wr.startPosition, wr.length), (DWRITE_FONT_WEIGHT_BOLD, 0, 1));
            l.GetFontWeight(1, &mut w, Some(&mut wr)).unwrap();
            assert_eq!(w, DWRITE_FONT_WEIGHT_NORMAL);

            // SetUnderline(true) on [1,1); off at 0, on at 1.
            l.SetUnderline(true, DWRITE_TEXT_RANGE { startPosition: 1, length: 1 }).unwrap();
            let (mut u, mut ur) = (BOOL(0), DWRITE_TEXT_RANGE::default());
            l.GetUnderline(0, &mut u, Some(&mut ur)).unwrap();
            assert!(!u.as_bool());
            l.GetUnderline(1, &mut u, Some(&mut ur)).unwrap();
            assert!(u.as_bool());
            assert_eq!((ur.startPosition, ur.length), (1, 1));

            // Draw emits the 'b' run at em 32, the 'A' run at the base 16, and an underline for 'b'.
            let ems = std::rc::Rc::new(RefCell::new(Vec::new()));
            let uls = std::rc::Rc::new(std::cell::Cell::new(0u32));
            let cap: IDWriteTextRenderer = CapRenderer { ems: ems.clone(), underlines: uls.clone() }.into();
            l.Draw(None, &cap, 0.0, 0.0).unwrap();
            let got = ems.borrow();
            assert!(got.contains(&32.0), "per-range em 32 not emitted: {got:?}");
            assert!(got.contains(&16.0), "base em 16 not emitted: {got:?}");
            assert!(uls.get() >= 1, "underline on 'b' not emitted to the renderer");
        }
    }

    /// The inherited IDWriteTextFormat setters on the LAYOUT re-lay the string (Paint.NET sets alignment
    /// after CreateTextLayout, then draws): SetTextAlignment/ParagraphAlignment move the ink, and every
    /// Set*/Get* round-trips. "Ab" in a 1000-px box, so centre/trailing shift the metrics' left far.
    #[test]
    fn layout_format_setters_relayout_and_roundtrip() {
        let l = layout4();
        unsafe {
            let bl: IDWriteTextLayout = l.cast().unwrap();
            let mut lead = DWRITE_TEXT_METRICS::default();
            bl.GetMetrics(&mut lead).unwrap(); // default LEADING: left ~0
            l.SetTextAlignment(DWRITE_TEXT_ALIGNMENT_CENTER).unwrap();
            assert_eq!(l.GetTextAlignment(), DWRITE_TEXT_ALIGNMENT_CENTER);
            let mut cen = DWRITE_TEXT_METRICS::default();
            bl.GetMetrics(&mut cen).unwrap();
            assert!(cen.left > lead.left + 100.0, "centre must shift left far right: {} -> {}", lead.left, cen.left);
            l.SetTextAlignment(DWRITE_TEXT_ALIGNMENT_TRAILING).unwrap();
            let mut trail = DWRITE_TEXT_METRICS::default();
            bl.GetMetrics(&mut trail).unwrap();
            assert!(trail.left > cen.left, "trailing left ({}) must exceed centre ({})", trail.left, cen.left);
            // Every other inherited setter round-trips through its getter.
            l.SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_CENTER).unwrap();
            assert_eq!(l.GetParagraphAlignment(), DWRITE_PARAGRAPH_ALIGNMENT_CENTER);
            l.SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP).unwrap();
            assert_eq!(l.GetWordWrapping(), DWRITE_WORD_WRAPPING_NO_WRAP);
            l.SetReadingDirection(DWRITE_READING_DIRECTION_RIGHT_TO_LEFT).unwrap();
            assert_eq!(l.GetReadingDirection(), DWRITE_READING_DIRECTION_RIGHT_TO_LEFT);
            l.SetFlowDirection(DWRITE_FLOW_DIRECTION_BOTTOM_TO_TOP).unwrap();
            assert_eq!(l.GetFlowDirection(), DWRITE_FLOW_DIRECTION_BOTTOM_TO_TOP);
            l.SetIncrementalTabStop(123.0).unwrap();
            assert_eq!(l.GetIncrementalTabStop(), 123.0);
            let base: IDWriteTextFormat = l.cast().unwrap();
            base.SetLineSpacing(DWRITE_LINE_SPACING_METHOD_UNIFORM, 20.0, 16.0).unwrap();
            let (mut m, mut h, mut b) = (DWRITE_LINE_SPACING_METHOD_DEFAULT, 0.0f32, 0.0f32);
            base.GetLineSpacing(&mut m, &mut h, &mut b).unwrap();
            assert_eq!((m, h, b), (DWRITE_LINE_SPACING_METHOD_UNIFORM, 20.0, 16.0));
        }
    }
}
