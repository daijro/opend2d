//! `IDWriteTextAnalyzer` over the `d2d::text::shaping` OpenType shaper (rustybuzz). `GetGlyphs`
//! shapes a run to glyph ids + a cluster map; `GetGlyphPlacements` measures it (advances/offsets in
//! DIPs). The shaper is the same one `d2d::text::layout` is verified against per run, so an app that
//! drives the analyzer directly gets the same glyph ids/advances native produces.
//!
//! `AnalyzeScript`/`Bidi`/`NumberSubstitution`/`LineBreakpoints` report a single LTR run with simple
//! break opportunities — enough to feed `GetGlyphs`. ponytail: full UAX-9 bidi, per-run script
//! itemization and the UAX-14 line-break table aren't reproduced (the shaper auto-detects script and
//! the corpus is single-script LTR); extend when a complex-analysis scene needs them.

use super::{E_INVALIDARG, ITextInterop};
use core::slice;
use d2d::text::shaping::{self, Font};
use windows::Win32::Graphics::DirectWrite::*;
use windows_core::{BOOL, HRESULT, Interface, PCWSTR, Ref, Result as WResult, implement};

/// `HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER)` — `GetGlyphs` returns it when the glyph buffer is
/// too small so the caller can retry with a bigger one.
const E_NOT_SUFFICIENT: HRESULT = HRESULT(0x8007_007Au32 as i32);

#[implement(IDWriteTextAnalyzer)]
pub(crate) struct AnalyzerObj;

pub(crate) fn analyzer() -> IDWriteTextAnalyzer {
    AnalyzerObj.into()
}

/// The shaping font behind a face (our faces carry their bytes via [`ITextInterop`]).
fn font_of(face: Ref<IDWriteFontFace>) -> Option<Font> {
    let it: ITextInterop = face.as_ref()?.cast().ok()?;
    let (mut data, mut len, mut index) = (core::ptr::null(), 0usize, 0u32);
    if unsafe { it.FontBlob(&mut data, &mut len, &mut index) }.is_err() || data.is_null() {
        return None;
    }
    Font::from_bytes(unsafe { slice::from_raw_parts(data, len) }.to_vec(), index)
}

/// Byte-offset (into the UTF-8 shaping string) → UTF-16 unit index, length `s.len()+1`.
fn byte_to_u16(s: &str) -> Vec<u16> {
    let mut map = vec![0u16; s.len() + 1];
    let mut u = 0u16;
    for (b, ch) in s.char_indices() {
        for m in map.iter_mut().take(b + ch.len_utf8()).skip(b) {
            *m = u;
        }
        u += ch.len_utf16() as u16;
    }
    if let Some(last) = map.last_mut() {
        *last = u;
    }
    map
}

impl IDWriteTextAnalyzer_Impl for AnalyzerObj_Impl {
    fn AnalyzeScript(&self, source: Ref<IDWriteTextAnalysisSource>, pos: u32, len: u32, sink: Ref<IDWriteTextAnalysisSink>) -> WResult<()> {
        // One run, default (shaper-detected) script — our GetGlyphs auto-detects from the text.
        if let (Some(sink), _src) = (sink.as_ref(), &source) {
            let sa = DWRITE_SCRIPT_ANALYSIS { script: 0, shapes: DWRITE_SCRIPT_SHAPES_DEFAULT };
            unsafe { sink.SetScriptAnalysis(pos, len, &sa)? };
        }
        Ok(())
    }
    fn AnalyzeBidi(&self, _source: Ref<IDWriteTextAnalysisSource>, pos: u32, len: u32, sink: Ref<IDWriteTextAnalysisSink>) -> WResult<()> {
        if let Some(sink) = sink.as_ref() {
            unsafe { sink.SetBidiLevel(pos, len, 0, 0)? }; // single LTR run
        }
        Ok(())
    }
    fn AnalyzeNumberSubstitution(&self, source: Ref<IDWriteTextAnalysisSource>, pos: u32, len: u32, sink: Ref<IDWriteTextAnalysisSink>) -> WResult<()> {
        // Native invokes the sink only for runs the source actually assigns a substitution object; a
        // run with no substitution (GetNumberSubstitution → None) produces no sink call (VM-verified).
        let (Some(src), Some(sink)) = (source.as_ref(), sink.as_ref()) else { return Ok(()) };
        let (mut sublen, mut ns) = (0u32, None);
        unsafe { src.GetNumberSubstitution(pos, &mut sublen, &mut ns)? };
        if ns.is_some() {
            let run = len.min(sublen.max(1));
            unsafe { sink.SetNumberSubstitution(pos, run, ns.as_ref())? };
        }
        Ok(())
    }
    fn AnalyzeLineBreakpoints(&self, source: Ref<IDWriteTextAnalysisSource>, pos: u32, len: u32, sink: Ref<IDWriteTextAnalysisSink>) -> WResult<()> {
        // Per-character break conditions (DWRITE_BREAK_CONDITION: CAN_BREAK=1, MAY_NOT_BREAK=2), packed
        // breakConditionBefore:2 | breakConditionAfter:2 | isWhitespace:1 | isSoftHyphen:1. Verified vs
        // native: before = CAN_BREAK after a whitespace char else MAY_NOT_BREAK; after = CAN_BREAK on a
        // whitespace char or the last position else MAY_NOT_BREAK. ponytail: only ASCII space is treated
        // as whitespace (the full UAX-14 class table isn't reproduced).
        let (Some(src), Some(sink)) = (source.as_ref(), sink.as_ref()) else { return Ok(()) };
        let (mut tp, mut tn) = (core::ptr::null_mut::<u16>(), 0u32);
        unsafe { src.GetTextAtPosition(pos, &mut tp, &mut tn)? };
        if tp.is_null() {
            return Ok(());
        }
        let n = (tn as usize).min(len as usize);
        let text = unsafe { slice::from_raw_parts(tp, n) };
        let bp: Vec<DWRITE_LINE_BREAKPOINT> = (0..n)
            .map(|i| {
                let is_ws = text[i] == 0x20;
                let prev_ws = i > 0 && text[i - 1] == 0x20;
                let before: u8 = if prev_ws { 1 } else { 2 };
                let after: u8 = if is_ws || i == n - 1 { 1 } else { 2 };
                let mut b = DWRITE_LINE_BREAKPOINT::default();
                b._bitfield = before | (after << 2) | ((is_ws as u8) << 4);
                b
            })
            .collect();
        unsafe { sink.SetLineBreakpoints(pos, &bp)? };
        Ok(())
    }
    fn GetGlyphs(&self, text: &PCWSTR, textlength: u32, fontface: Ref<IDWriteFontFace>, _issideways: BOOL, isrtl: BOOL, _script: *const DWRITE_SCRIPT_ANALYSIS, _locale: &PCWSTR, _numsub: Ref<IDWriteNumberSubstitution>, _features: *const *const DWRITE_TYPOGRAPHIC_FEATURES, _featurerangelengths: *const u32, _featureranges: u32, maxglyphcount: u32, clustermap: *mut u16, _textprops: *mut DWRITE_SHAPING_TEXT_PROPERTIES, glyphindices: *mut u16, _glyphprops: *mut DWRITE_SHAPING_GLYPH_PROPERTIES, actualglyphcount: *mut u32) -> WResult<()> {
        let font = font_of(fontface).ok_or(E_INVALIDARG)?;
        let text16 = unsafe { slice::from_raw_parts(text.0, textlength as usize) };
        let s = String::from_utf16_lossy(text16);
        let glyphs = shaping::shape(&font, &s, isrtl.as_bool(), None, None);
        unsafe { *actualglyphcount = glyphs.len() as u32 };
        if glyphs.len() > maxglyphcount as usize {
            return Err(E_NOT_SUFFICIENT.into());
        }
        let ids = unsafe { slice::from_raw_parts_mut(glyphindices, glyphs.len()) };
        for (d, g) in ids.iter_mut().zip(&glyphs) {
            *d = g.id;
        }
        // Cluster map (one entry per UTF-16 unit → first glyph of that cluster). Clusters are UTF-8
        // byte offsets from the shaper; map them to UTF-16 units. LTR fill; RTL mirrors the order.
        let b2u = byte_to_u16(&s);
        let cmap = unsafe { slice::from_raw_parts_mut(clustermap, textlength as usize) };
        cmap.fill(0);
        for (gi, g) in glyphs.iter().enumerate() {
            let u = *b2u.get(g.cluster as usize).unwrap_or(&0) as usize;
            let gi = if isrtl.as_bool() { glyphs.len() - 1 - gi } else { gi } as u16;
            for c in cmap.iter_mut().skip(u) {
                *c = gi;
            }
        }
        Ok(())
    }
    fn GetGlyphPlacements(&self, text: &PCWSTR, _clustermap: *const u16, _textprops: *mut DWRITE_SHAPING_TEXT_PROPERTIES, textlength: u32, _glyphindices: *const u16, _glyphprops: *const DWRITE_SHAPING_GLYPH_PROPERTIES, glyphcount: u32, fontface: Ref<IDWriteFontFace>, emsize: f32, _issideways: BOOL, isrtl: BOOL, _script: *const DWRITE_SCRIPT_ANALYSIS, _locale: &PCWSTR, _features: *const *const DWRITE_TYPOGRAPHIC_FEATURES, _featurerangelengths: *const u32, _featureranges: u32, glyphadvances: *mut f32, glyphoffsets: *mut DWRITE_GLYPH_OFFSET) -> WResult<()> {
        place(fontface, text, textlength, glyphcount, emsize, isrtl, glyphadvances, glyphoffsets, None)
    }
    fn GetGdiCompatibleGlyphPlacements(&self, text: &PCWSTR, _clustermap: *const u16, _textprops: *const DWRITE_SHAPING_TEXT_PROPERTIES, textlength: u32, _glyphindices: *const u16, _glyphprops: *const DWRITE_SHAPING_GLYPH_PROPERTIES, glyphcount: u32, fontface: Ref<IDWriteFontFace>, emsize: f32, pixelsperdip: f32, _transform: *const DWRITE_MATRIX, usegdinatural: BOOL, _issideways: BOOL, isrtl: BOOL, _script: *const DWRITE_SCRIPT_ANALYSIS, _locale: &PCWSTR, _features: *const *const DWRITE_TYPOGRAPHIC_FEATURES, _featurerangelengths: *const u32, _featureranges: u32, glyphadvances: *mut f32, glyphoffsets: *mut DWRITE_GLYPH_OFFSET) -> WResult<()> {
        // classic = !useGdiNatural (GDI_CLASSIC rounds the kern to whole pixels; GDI_NATURAL doesn't).
        place(fontface, text, textlength, glyphcount, emsize, isrtl, glyphadvances, glyphoffsets, Some((pixelsperdip, !usegdinatural.as_bool())))
    }
}

/// Shared placement: re-shape the run and emit advances/offsets in DIPs. `gdi = Some((ppd, classic))`
/// gives the GDI-compatible advance — the TrueType-hinted whole-pixel glyph advance at the device ppem
/// plus the shaper's kern (`d2d::text::gdi::GdiMeasure`, the same path layout's GDI modes use), not a
/// naive round of the design advance.
#[allow(clippy::too_many_arguments)]
fn place(fontface: Ref<IDWriteFontFace>, text: &PCWSTR, textlength: u32, glyphcount: u32, emsize: f32, isrtl: BOOL, advances: *mut f32, offsets: *mut DWRITE_GLYPH_OFFSET, gdi: Option<(f32, bool)>) -> WResult<()> {
    let font = font_of(fontface).ok_or(E_INVALIDARG)?;
    let text16 = unsafe { slice::from_raw_parts(text.0, textlength as usize) };
    let s = String::from_utf16_lossy(text16);
    let glyphs = shaping::shape(&font, &s, isrtl.as_bool(), None, None);
    let n = (glyphcount as usize).min(glyphs.len());
    let upm = font.upm as f32;
    // GDI-compatible advances come from the hinted-advance engine (falls back to a design-advance round
    // only when the font lacks the TrueType scaler tables).
    let gm = gdi.and_then(|(ppd, classic)| d2d::text::gdi::GdiMeasure::new(&font, emsize, ppd, classic, None).map(|m| (m, ppd)));
    let adv = unsafe { slice::from_raw_parts_mut(advances, glyphcount as usize) };
    let off = unsafe { slice::from_raw_parts_mut(offsets, glyphcount as usize) };
    for i in 0..n {
        let a = glyphs[i].advance as f32 * emsize / upm;
        adv[i] = match (&gm, gdi) {
            // GetGlyphPlacements has no simulation context here, so no synthetic-bold advance (unchanged).
            (Some((m, _)), _) => m.advance(glyphs[i].id, glyphs[i].advance, emsize / upm, false),
            (None, Some((ppd, _))) => (a * ppd).round() / ppd,
            (None, None) => a,
        };
        off[i] = DWRITE_GLYPH_OFFSET {
            advanceOffset: glyphs[i].dx as f32 * emsize / upm,
            ascenderOffset: glyphs[i].dy as f32 * emsize / upm,
        };
    }
    Ok(())
}
