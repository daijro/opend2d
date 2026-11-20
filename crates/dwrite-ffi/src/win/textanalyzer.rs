//! `IDWriteTextAnalyzer` over the `d2d::text::shaping` OpenType shaper (rustybuzz). `GetGlyphs`
//! shapes a run to glyph ids + a cluster map; `GetGlyphPlacements` measures it (advances/offsets in
//! DIPs). The shaper is the same one `d2d::text::layout` is verified against per run, so an app that
//! drives the analyzer directly gets the same glyph ids/advances native produces.
//!
//! `AnalyzeLineBreakpoints` is DirectWrite's breaker, the one `d2d::text::layout` wraps by. `AnalyzeScript`/`Bidi`/
//! `NumberSubstitution` report a single LTR run — enough to feed `GetGlyphs`. ponytail: full UAX-9 bidi and per-run
//! script itemization aren't reproduced (the shaper auto-detects script and the corpus is single-script LTR); extend
//! when a complex-analysis scene needs them.

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
        // DirectWrite's breaker over the range (`d2d::text::layout::line_breakpoints`), told the unit the source has before
        // the range and, for the one after, the last unit of the text the source reports at the range's end, which is
        // the one native reads. A range past the u32 limit fails as native's does; an empty one reports nothing.
        let (Some(src), Some(sink)) = (source.as_ref(), sink.as_ref()) else { return Ok(()) };
        if len == 0 {
            return Ok(());
        }
        let end = pos.checked_add(len).ok_or(E_INVALIDARG)?;
        let text_at = |p: u32| -> WResult<&[u16]> {
            let (mut tp, mut tn) = (core::ptr::null_mut::<u16>(), 0u32);
            unsafe { src.GetTextAtPosition(p, &mut tp, &mut tn)? };
            Ok(if tp.is_null() { &[] } else { unsafe { slice::from_raw_parts(tp, tn as usize) } })
        };
        let mut units: Vec<u16> = Vec::with_capacity(len as usize);
        while units.len() < len as usize {
            let chunk = text_at(pos + units.len() as u32)?;
            if chunk.is_empty() {
                break;
            }
            units.extend_from_slice(&chunk[..chunk.len().min(len as usize - units.len())]);
        }
        let before = if pos > 0 {
            let (mut tp, mut tn) = (core::ptr::null_mut::<u16>(), 0u32);
            unsafe { src.GetTextBeforePosition(pos, &mut tp, &mut tn)? };
            (!tp.is_null() && tn > 0).then(|| unsafe { *tp.add(tn as usize - 1) })
        } else {
            None
        };
        let after = text_at(end)?.last().copied();
        let bp: Vec<DWRITE_LINE_BREAKPOINT> = d2d::text::layout::line_breakpoints(&units, before, after)
            .into_iter()
            .map(|b| {
                let mut x = DWRITE_LINE_BREAKPOINT::default();
                x._bitfield = b;
                x
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[implement(IDWriteTextAnalysisSource)]
    struct Source(Vec<u16>);
    impl IDWriteTextAnalysisSource_Impl for Source_Impl {
        fn GetTextAtPosition(&self, pos: u32, s: *mut *mut u16, len: *mut u32) -> WResult<()> {
            let rest = self.0.get(pos as usize..).filter(|r| !r.is_empty());
            unsafe {
                *s = rest.map_or(core::ptr::null_mut(), |r| r.as_ptr() as *mut u16);
                *len = rest.map_or(0, |r| r.len() as u32);
            }
            Ok(())
        }
        fn GetTextBeforePosition(&self, pos: u32, s: *mut *mut u16, len: *mut u32) -> WResult<()> {
            unsafe {
                *s = if pos > 0 { self.0.as_ptr() as *mut u16 } else { core::ptr::null_mut() };
                *len = pos;
            }
            Ok(())
        }
        fn GetParagraphReadingDirection(&self) -> DWRITE_READING_DIRECTION {
            DWRITE_READING_DIRECTION_LEFT_TO_RIGHT
        }
        fn GetLocaleName(&self, _pos: u32, len: *mut u32, name: *mut *mut u16) -> WResult<()> {
            unsafe {
                *len = 0;
                *name = core::ptr::null_mut();
            }
            Ok(())
        }
        fn GetNumberSubstitution(&self, _pos: u32, len: *mut u32, ns: windows_core::OutRef<'_, IDWriteNumberSubstitution>) -> WResult<()> {
            unsafe { *len = 0 };
            let _ = ns.write(None);
            Ok(())
        }
    }

    #[implement(IDWriteTextAnalysisSink)]
    struct Sink(RefCell<Vec<u8>>);
    impl IDWriteTextAnalysisSink_Impl for Sink_Impl {
        fn SetScriptAnalysis(&self, _pos: u32, _len: u32, _sa: *const DWRITE_SCRIPT_ANALYSIS) -> WResult<()> {
            Ok(())
        }
        fn SetLineBreakpoints(&self, _pos: u32, len: u32, bp: *const DWRITE_LINE_BREAKPOINT) -> WResult<()> {
            self.0.borrow_mut().extend(unsafe { slice::from_raw_parts(bp, len as usize) }.iter().map(|b| b._bitfield));
            Ok(())
        }
        fn SetBidiLevel(&self, _pos: u32, _len: u32, _explicit: u8, _resolved: u8) -> WResult<()> {
            Ok(())
        }
        fn SetNumberSubstitution(&self, _pos: u32, _len: u32, _ns: Ref<IDWriteNumberSubstitution>) -> WResult<()> {
            Ok(())
        }
    }

    /// VM `AnalyzeLineBreakpoints` over whole texts and sub-ranges, whose neighbours the source supplies: the unit
    /// before the range, and for the one after, the last unit of the text the source reports at the range's end.
    #[test]
    fn line_breakpoints_match_native() {
        let path: Vec<u16> = "C:\\Users\\Public\\Documents\\Projects\\2026\\Quarterly Report (final).xlsx".encode_utf16().collect();
        let whole = vec![
            10u8, 6, 9, 10, 10, 10, 10, 6, 9, 10, 10, 10, 10, 10, 6, 9, 10, 10, 10, 10, 10, 10, 10, 10, 6, 9, 10, 10, 10, 10, 10, 10, 10, 6, 9,
            10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 22, 9, 10, 10, 10, 10, 10, 22, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 6,
        ];
        for (text, pos, len, want) in [
            (path.clone(), 0, path.len(), whole),
            (path.clone(), 7, 3, vec![4, 9, 2]),
            ("(555) 012-3456".encode_utf16().collect(), 5, 2, vec![20, 1]),
            ("hyphen-ated co-op/ed A/B and/or".encode_utf16().collect(), 11, 3, vec![20, 9, 2]),
            (vec![0xD800, 0x78, 0xDC00, 0x79, 0xD800, 0xDC00, 0x7A], 3, 2, vec![8, 2]),
        ] {
            let src: IDWriteTextAnalysisSource = Source(text).into();
            let sink: IDWriteTextAnalysisSink = Sink(RefCell::new(Vec::new())).into();
            unsafe { analyzer().AnalyzeLineBreakpoints(&src, pos as u32, len as u32, &sink) }.unwrap();
            assert_eq!(*sink.cast_object_ref::<Sink>().unwrap().0.borrow(), want, "{pos}+{len}");
        }
    }
}
