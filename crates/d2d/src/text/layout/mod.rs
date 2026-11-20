//! Layout: IDWriteTextFormat/IDWriteTextLayout behaviour — itemization (bidi + script),
//! shaping each run via [`super::shaping`], hard line breaks, bidi reordering, run
//! positioning, and the metrics DirectWrite exposes (DWRITE_TEXT_METRICS, per-line and
//! per-cluster metrics). Verified against native by the harness `text_layout` query op
//! (see `tests/text_layout`).
//!
//! Scope so far: natural measuring, single font/format, bidi, hard line breaks and soft
//! word-wrapping, alignment, and caret/point/range hit testing — all byte-exact vs native. GDI
//! measuring, trimming, inline objects, typography and font fallback are listed at the end.

mod justify;
mod linebreak;

pub use linebreak::line_breakpoints;

use super::shaping::{self, Font, Glyph};
use unicode_bidi::{BidiClass, BidiInfo, Level};
use linebreak::linebreaks;
use unicode_linebreak::BreakOpportunity;
use unicode_script::{Script, UnicodeScript};

/// One `IDWriteTypography` feature applied over a UTF-16 range (`SetTypography`), threaded to the
/// shaper. A range boundary forces a glyph-run split, exactly as DirectWrite emits.
#[derive(Clone, Copy, Debug)]
pub struct Feature {
    pub start: u32,
    pub end: u32,
    pub tag: [u8; 4],
    pub value: u32,
}

/// `DWRITE_TRIMMING` + whether an ellipsis sign is installed (`CreateEllipsisTrimmingSign`). A line
/// whose visible content overflows the layout width is trimmed per this.
#[derive(Clone, Copy, Debug, Default)]
pub struct Trimming {
    /// `DWRITE_TRIMMING_GRANULARITY`: 1 character, 2 word (0 = none, no trimming).
    pub granularity: u8,
    /// Delimiter char code for path trimming (0 = none): the preserved tail begins at the
    /// `delimiter_count`-th delimiter counted back from the end, with the sign at the front.
    pub delimiter: u32,
    pub delimiter_count: u32,
    pub ellipsis: bool,
    /// A vertical layout's upright sign extent per em ([`upright_sign_per_em`] of the face font fallback gives
    /// the vertical ellipsis), or 0 when unresolved: the sign then spans the format face's ellipsis.
    pub upright_sign: f32,
    /// The part of `upright_sign` per em that a bold-simulated sign face adds (0 otherwise). It widens the sign's box
    /// but not the glyph's own advance, so a bottom-to-top column, which draws its sign from the foot of the box,
    /// lifts the glyph off the box top by it (VM r25 30351).
    pub upright_bold: f32,
    /// The sign's ellipsis glyph and advance per em when font fallback gives the sign its own face (the format
    /// family maps no ellipsis), or `None` for the format face's.
    pub sign: Option<(u16, f32)>,
}

/// A fixed-size inline object set over a UTF-16 range (`IDWriteTextLayout::SetInlineObject`). Its
/// glyphs are replaced by one cluster of `width`, and it contributes `baseline`/`height-baseline`
/// to the line's ascent/descent.
#[derive(Clone, Copy, Debug)]
pub struct InlineObject {
    pub start: u32,
    pub end: u32,
    pub width: f32,
    pub height: f32,
    pub baseline: f32,
    pub sideways: bool,
}

/// A byte range `[start,end)` of the layout text shaped with a specific fallback `font`, as
/// resolved by [`super::fallback::map_characters`]. Spans for the base family are omitted (a byte
/// in no span uses the base `font` argument), so a single-font layout passes an empty slice and is
/// byte-identical to before fallback existed.
#[derive(Clone, Copy)]
pub struct FontSpan<'a> {
    pub start: usize,
    pub end: usize,
    pub font: &'a Font,
    /// Per-range font size override (`SetFontSize`) in DIPs, or `None` to use the format's base em.
    /// A size-only range still gets a span (with the base `font`) so this em reaches the run.
    pub em: Option<f32>,
    /// The span's face is bold-simulated: an inked glyph widens like the base face's (`base_bold_delta_du`).
    pub bold: bool,
}

/// Per-call layout inputs beyond the format: typography feature ranges, trimming, inline objects,
/// resolved font-fallback spans, and the measuring mode. All default so `layout` stays the simple
/// entry point (`measuring` 0 = natural, byte-identical to before GDI measuring existed).
#[derive(Default)]
pub struct LayoutOpts<'a> {
    pub features: &'a [Feature],
    pub trimming: Option<Trimming>,
    pub inline: &'a [InlineObject],
    pub fonts: &'a [FontSpan<'a>],
    /// Per-range underline / strikethrough as BYTE ranges `[start,end)` (`SetUnderline`/
    /// `SetStrikethrough`). A boundary forces a run split so each run is wholly decorated or not.
    pub underline: &'a [(usize, usize)],
    pub strikethrough: &'a [(usize, usize)],
    /// `DWRITE_MEASURING_MODE`: 0 natural, 1 GDI classic, 2 GDI natural. The GDI modes measure with
    /// whole-pixel hinted advances and VDMX line metrics ([`super::gdi`]).
    pub measuring: u32,
    /// `dpi/96` for the GDI modes' integer ppem (`emSize · pixelsPerDip`); 0 is treated as 1.0.
    pub pixels_per_dip: f32,
    /// Bold-simulation advance delta for the BASE face, in design units (`round(upem/50)`, native's
    /// bold embolden), or 0 for a real face. Applied per inked base glyph: added to the natural
    /// advance, or as +1 device pixel in a GDI mode. Default 0 ⇒ non-bold-sim layouts are unchanged.
    pub base_bold_delta_du: i32,
    /// DrawText's text transform (`[m11, m12, m21, m22]`, native `GetTextTransform`): DrawText hands its
    /// GDI-compatible layout the world transform. dwrite then measures at `ppd · |(m21, m22)|`
    /// (`GetGdiCompatibleMetrics`' y-axis scale), and `GetGlyphAdvances` returns DESIGN advances when the
    /// rasterizer transform has no zero component (rotated/sheared). `None` = identity (app layouts).
    pub transform: Option<[f32; 4]>,
}

/// The base font plus the fallback spans: resolves the font for any byte of the layout text.
struct Fonts<'a> {
    base: &'a Font,
    spans: &'a [FontSpan<'a>],
    /// GDI-compatible measurer for the base face, present iff a GDI measuring mode is active. It
    /// overrides the base font's advances and line metrics; fallback faces keep the natural path
    /// (no GDI scene mixes fallback with GDI measuring).
    gdi: Option<super::gdi::GdiMeasure<'a>>,
    /// The GDI measurer of each span (per-range format or fallback face), at its em.
    span_gdi: Vec<Option<super::gdi::GdiMeasure<'a>>>,
    /// Bold-sim advance delta (design units) for the base face; 0 for a real face.
    bold_delta_du: i32,
    /// GDI measuring under a rotated/sheared DrawText transform: advances stay natural (design), only
    /// the line metrics and snapping are GDI-compatible ([`LayoutOpts::transform`]).
    design_advances: bool,
    /// The DrawText transform's y-scale `s` (1 = none), see `layout_ex` / `vmetrics`.
    metric_scale: f32,
    /// A vertical layout (flow across the page): line metrics centre each font's em box, see `vmetrics`.
    vertical: bool,
}

impl<'a> Fonts<'a> {
    /// The font a byte shapes with: its fallback span, else the base family.
    fn at(&self, byte: usize) -> &'a Font {
        self.spans.iter().find(|s| byte >= s.start && byte < s.end).map(|s| s.font).unwrap_or(self.base)
    }

    /// A byte's per-range font size (`SetFontSize`) override, or `None` for the format's base em.
    fn em_at(&self, byte: usize) -> Option<f32> {
        self.spans.iter().find(|s| byte >= s.start && byte < s.end).and_then(|s| s.em)
    }

    /// The GDI measurer for `font` at `byte`: the format span's own, else the base face's.
    fn gdi_at(&self, font: &Font, byte: usize) -> Option<&super::gdi::GdiMeasure<'a>> {
        match self.spans.iter().position(|s| byte >= s.start && byte < s.end) {
            Some(i) => self.span_gdi.get(i).and_then(|g| g.as_ref()),
            None if std::ptr::eq(font, self.base) => self.gdi.as_ref(),
            None => None,
        }
    }

    /// A glyph's DIP advance: the GDI advance (hinted glyph advance + kern) for the base face under a
    /// GDI measuring mode, else the natural `shaped · scale`. `shaped_advance` is the shaper's
    /// (kerned) advance in design units.
    fn advance(&self, font: &Font, byte: usize, glyph_id: u16, shaped_advance: i32, scale: f32) -> f32 {
        // Bold simulation emboldens only glyphs that HAVE an outline (an empty glyph like the space keeps its
        // advance): the base face's, or a bold-simulated span's (VM r25 30546: a Leelawadee UI weight-700 range,
        // regular only in the cold collection, widens its inked glyphs by round(upem/50) as the base would).
        let delta_du = self.bold_delta(font, byte);
        // A zero-advance glyph (a mark, or an emoji sequence's joiner glyph) keeps its zero advance (r25 30086).
        // Under a GDI measuring mode every advancing glyph gains the bold pixel, the space too (r25: 70 of 75
        // bold-simulated GDI spaces are one pixel wider; 30844: a Segoe UI Emoji space 13 px, not 12).
        let bold = delta_du != 0 && shaped_advance != 0;
        match self.gdi_at(font, byte) {
            Some(g) if !self.design_advances => g.advance(glyph_id, shaped_advance, scale, bold),
            _ => (shaped_advance + if bold && font.has_outline(glyph_id) { delta_du } else { 0 }) as f32 * scale,
        }
    }

    /// The bold simulation's design-unit widening of an inked glyph at `byte`: round(upem/50) for a bold-simulated
    /// base face or span, else 0.
    fn bold_delta(&self, font: &Font, byte: usize) -> i32 {
        match self.spans.iter().find(|s| byte >= s.start && byte < s.end).map(|s| s.bold) {
            Some(true) => (font.upm as f32 / 50.0).round() as i32,
            Some(false) => 0,
            None if std::ptr::eq(font, self.base) => self.bold_delta_du,
            None => 0,
        }
    }

    /// The span around `[s,e)` DirectWrite shapes as one: the neighbouring bytes in the same face and bidi
    /// level at the same em (a decoration or style change alone does not split it; a size change does).
    fn shaping_span(&self, font: &Font, text: &str, s: usize, e: usize, levels: &[Level]) -> (usize, usize) {
        // Same face, bidi level and em: a size change ends the context (r25 30247: 'بال' at em 0.001 leaves its
        // lam unjoined before a 3.46-em 'ع'), a decoration or style range resolving to the same face does not
        // (30173, 30837).
        let same = |b: usize| std::ptr::eq(self.at(b), font) && levels.get(b) == levels.get(s) && self.em_at(b) == self.em_at(s);
        let mut a = s;
        while let Some((p, _)) = text[..a].char_indices().next_back() {
            if !same(p) || text[p..].starts_with(is_newline) {
                break;
            }
            a = p;
        }
        let mut z = e;
        while let Some(ch) = text[z..].chars().next() {
            if !same(z) || is_newline(ch) {
                break;
            }
            z += ch.len_utf8();
        }
        (a, z)
    }

    /// Whether the run starting at byte `b` stands upright in a vertical layout (runs split where it changes).
    fn upright_at(&self, text: &str, b: usize) -> bool {
        self.vertical && upright_at(text, b)
    }

    /// [`Self::advance`] along the run's direction: an upright run's vertical advance (design units in
    /// `shaped_advance`) is the scaled value, or under a GDI mode that value rounded to whole device pixels.
    fn advance_dir(&self, font: &Font, byte: usize, glyph_id: u16, shaped_advance: i32, scale: f32, upright: bool) -> f32 {
        if !upright {
            return self.advance(font, byte, glyph_id, shaped_advance, scale);
        }
        // Bold simulation lengthens an upright advance as it widens a horizontal one: one device pixel under a GDI
        // mode (VM r25 30046, 30743: Segoe UI Emoji at weights 700 and 900 spans round(1922·ppem/2048) + 1 px).
        let delta_du = self.bold_delta(font, byte);
        let bold = delta_du != 0 && shaped_advance != 0;
        match self.gdi_at(font, byte) {
            Some(g) if !self.design_advances => g.vadvance(shaped_advance, if bold { 1.0 } else { 0.0 }),
            _ => (shaped_advance + if bold && font.has_outline(glyph_id) { delta_du } else { 0 }) as f32 * scale,
        }
    }

    /// The font's DIP line `(ascent, descent)`. A vertical layout centres the font's em box on the
    /// central baseline: the ascent drops by [`Self::vhalf`] and the descent grows by it, so a line's
    /// baseline is the highest centred ascent and its height adds the deepest centred descent (VM r25:
    /// Consolas GDI 37/18, Segoe UI with Comic Sans 13/6; natural lines of one font sit at half height).
    fn vmetrics(&self, font: &Font, byte: usize, em: f32, upright: bool) -> (f32, f32) {
        let (asc, desc) = self.vmetrics_h(font, byte, em);
        if !self.vertical {
            return (asc, desc);
        }
        // A GDI line box comes from the face's GDI-compatible ascent and descent in design units. An upright run
        // centres them, the ascent taking the odd unit; a rotated run drops its horizontal baseline by the
        // face's central-baseline offset. Each edge then rounds to whole device pixels, half up (native's
        // line-spacing refresh; VM: Segoe UI Emoji upright at device em 40.19 is 27 + 26 px from 2191 + 510
        // units, Segoe UI 'a' rotated 26 + 28 px from 2191 + 561 less 848).
        if let Some(g) = self.gdi_at(font, byte).filter(|_| self.metric_scale == 1.0) {
            let p = g.ppd();
            let (a_du, d_du) = Self::gdi_compat_du(font, em, p, 1.0);
            let (a, d, delta) = if upright {
                let t = a_du + d_du;
                ((t + 1) >> 1, t >> 1, 0)
            } else {
                let (fa, fd, _) = Self::line_design(font);
                (a_du, d_du, (fa - fd).div_euclid(2))
            };
            let rhu = |v: f32| {
                let n = v.round_ties_even();
                if n - v == -0.5 { n + 1.0 } else { n }
            };
            let snap = |du: i32| rhu((du as f32 * em) / font.upm as f32 * p) * (1.0 / p);
            return (snap(a - delta), snap(delta + d));
        }
        // A natural line box splits the face's line gap between its ascent (the lower half) and descent, and works
        // in design units as the GDI box does: an upright run centres ascent plus descent, a rotated run drops by
        // the central offset, half the ascent less the descent rounded down (r25 30936: Arial's 1854 + 33 − 710 =
        // 1177 units above the central baseline; 30222: Palatino's 2150 − 768 sets the line's baseline).
        if self.gdi_at(font, byte).is_none() {
            let (fa, fd, gap) = Self::line_design(font);
            let (a, d, delta) = if upright {
                let t = fa + fd;
                ((t + 1) >> 1, t >> 1, 0)
            } else {
                (fa, fd, (fa - fd).div_euclid(2))
            };
            let k = |du: i32| (du as f32 * em) / font.upm as f32;
            return (k(a + (gap >> 1) - delta), k(delta + d + ((gap + 1) >> 1)));
        }
        let half = self.vhalf(font, byte, em);
        (asc - half, desc + half)
    }
    /// How far a vertical run's horizontal baseline sits from the central baseline: half the font's
    /// ascent less its descent, rounded to whole device pixels from the design metrics under a GDI mode.
    fn vhalf(&self, font: &Font, byte: usize, em: f32) -> f32 {
        match self.gdi_at(font, byte) {
            // Under GDI the run's own baseline is its line-spacing ascent rounded to pixels below the line top, the
            // central baseline that ascent less the central offset rounded on its own: the offset is the
            // difference of the two roundings (VM r25 30743: Segoe UI at device em 136.45 sits 146 − 89 = 57 px
            // from the central baseline, where 848 units round to 56).
            Some(g) if self.metric_scale == 1.0 => {
                let p = g.ppd();
                let (a_du, _) = Self::gdi_compat_du(font, em, p, 1.0);
                let (fa, fd, _) = Self::line_design(font);
                let delta = (fa - fd).div_euclid(2);
                let rhu = |v: f32| {
                    let n = v.round_ties_even();
                    if n - v == -0.5 { n + 1.0 } else { n }
                };
                let snap = |du: i32| rhu((du as f32 * em) / font.upm as f32 * p);
                (snap(a_du) - snap(a_du - delta)) * (1.0 / p)
            }
            Some(g) => {
                let (asc, desc, _) = Self::line_design(font);
                ((asc - desc) as f32 / 2.0 * em * g.ppd() / font.upm as f32).round() / g.ppd()
            }
            // Natural: half the ascent less the descent, the line gap left out (r25 30936: Arial's rotated runs sit
            // (1854 − 434) / 2 = 710 units from the central baseline).
            None => Self::vhalf_design(font, em),
        }
    }
    /// The central offset in design units scaled to the em, unrounded whatever the measuring mode.
    fn vhalf_design(font: &Font, em: f32) -> f32 {
        let (asc, desc, _) = Self::line_design(font);
        (asc - desc).div_euclid(2) as f32 * (em / font.upm as f32)
    }
    /// The font's horizontal DIP line `(ascent, descent)`: GDI VDMX metrics for the base face under a GDI
    /// measuring mode (falling back to natural when the font has no VDMX entry), else natural.
    fn vmetrics_h(&self, font: &Font, byte: usize, em: f32) -> (f32, f32) {
        if let Some(g) = self.gdi_at(font, byte) {
            {
                if self.metric_scale != 1.0 {
                    return Self::vmetrics_gdi_scaled(font, em, g.ppd(), self.metric_scale);
                }
                if let Some(m) = g.vmetrics() {
                    return m;
                }
                // No VDMX record: native's GDI-compatible metrics scale the design
                // ascent/descent/lineGap to the FLOAT device em (em·ppd) — NOT the rounded integer ppem,
                // which is only the VDMX lookup key — rounding each to whole device pixels, then back to
                // DIPs (Segoe UI em·ppd 5.742: descent round(514·5.742/2048)=1, where ppem 6 gives
                // round(1.506)=2). The baseline is the rounded ascent (fuzz 5182 Times Bold ppem 7: baseline
                // round(1825·7.197/2048) = 6).
                // The line gap stays out of the GDI line box (VM r25 30082 / 30208: Bahnschrift 28 + 7 px at
                // device em 35.4 without its 7 px typographic gap; 30796: Arial at ppem 737, past VDMX, is
                // 667 + 156 px without its 24 px gap).
                let (asc, desc, _) = Self::line_design(font);
                let k = em * g.ppd() / font.upm as f32;
                // Back to DIPs times the f32 reciprocal of ppd, not divided by it (VM: Tahoma's 13 px ascent at 120 dpi is
                // 10.400001 DIP, 13·0.8, where 13 / 1.25 is 10.3999996 and a baseline on a half-pixel tie snaps a pixel high).
                let px = |du: i32| (du as f32 * k).round();
                let r = 1.0 / g.ppd();
                return (px(asc) * r, px(desc) * r);
            }
        }
        Self::vmetrics_natural(font, em)
    }
    /// DirectWrite line vertical metrics `(ascent, descent)` of one font at `em` DIPs, reproducing
    /// DirectWrite's default font metrics: with the OS/2 `USE_TYPO_METRICS` flag set, the
    /// sTypo* values; otherwise `usWinAscent`/`usWinDescent` with the line gap folded in as
    /// `max(0, hheaLineHeight − usWinLineHeight)`. (ttf_parser's own `ascender()` returns hhea when
    /// non-zero and never the Win values, which places e.g. Segoe UI Emoji ~6 px too high.) Ascent
    /// includes the line gap (it sits above the baseline). For the usual case where `usWin` equals
    /// hhea asc/desc this is identical to hhea + its gap.
    /// ponytail: the table-presence condition native uses to prefer hhea over usWin for some fonts
    /// is not decoded — for every tested font usWin == hhea (so both agree) except the colour font,
    /// which native puts on the usWin path; revisit if a scene needs a font where they differ.
    fn vmetrics_natural(font: &Font, em: f32) -> (f32, f32) {
        let (asc, desc, gap) = Self::line_design(font);
        let scale = em / font.upm as f32;
        // Natural: the line gap sits above the baseline (ascent includes it), unlike the GDI fallback.
        ((asc + gap) as f32 * scale, desc as f32 * scale)
    }
    /// GDI line box under a scaled DrawText transform (y-scale `s`, layout pixels-per-DIP `p = ppd·s`),
    /// as native's layout refreshes its line-spacing metrics: `GetGdiCompatibleMetrics(em, p,
    /// transform)` — whose ppem folds the transform scale a second time, `em·p·s` — gives VDMX (a record
    /// below 255 ppem) or the design ascent/descent rounded at that float em, in design units; each is then
    /// scaled to DIPs and rounded half-up on the `p` grid, and the line gap is dropped (default spacing).
    /// Fits Arial/Times/Segoe UI @13 at ×0.7, ×1.5 and ×2 (dwrite-dump line metrics).
    fn vmetrics_gdi_scaled(font: &Font, em: f32, p: f32, s: f32) -> (f32, f32) {
        let rhu = |v: f32| {
            let n = v.round_ties_even();
            if n - v == -0.5 { n + 1.0 } else { n }
        };
        let upm = font.upm as f32;
        let (a_du, d_du) = Self::gdi_compat_du(font, em, p, s);
        let snap = |du: f32| rhu(du * em / upm * p) * (1.0 / p);
        (snap(a_du as f32), snap(d_du as f32))
    }
    /// `GetGdiCompatibleMetrics`' ascent and descent in design units at `em·p·s` device pixels: the VDMX record
    /// (below 255 ppem) or the design ascent/descent rounded to whole pixels at that float em, mapped back.
    fn gdi_compat_du(font: &Font, em: f32, p: f32, s: f32) -> (i32, i32) {
        let rhu = |v: f32| {
            let n = v.round_ties_even();
            if n - v == -0.5 { n + 1.0 } else { n }
        };
        let upm = font.upm as f32;
        let ppem_f = em * p * s;
        let du_px = upm / ppem_f;
        let (asc, desc, _) = Self::line_design(font);
        let vdmx = (ppem_f > 0.0 && ppem_f < 255.0)
            .then(|| font.tt().raw_face().table(rustybuzz::ttf_parser::Tag::from_bytes(b"VDMX")))
            .flatten()
            .and_then(|b| super::gdi::vdmx_lookup(b, rhu(ppem_f) as i32));
        let (a, d) = match vdmx {
            Some((ymax, ymin)) => (rhu(ymax as f32 * du_px), rhu(-(ymin as f32) * du_px)),
            None => (rhu(rhu(asc as f32 / du_px) * du_px), rhu(rhu(desc as f32 / du_px) * du_px)),
        };
        (a as i32, d as i32)
    }
    /// Design-unit line box `(ascent, descent, line-gap)` — all positive. The natural line box folds the
    /// gap into the ascent (× em/upm); the GDI VDMX-miss fallback rounds ascent/descent/gap separately
    /// (× em·ppd/upm) and keeps the gap out of the baseline. See [`vmetrics_natural`] for the OS/2
    /// `USE_TYPO_METRICS` vs `usWin` table selection.
    fn line_design(font: &Font) -> (i32, i32, i32) {
        let tt = font.tt();
        let hhea = tt.tables().hhea;
        let (asc, desc, gap): (i32, i32, i32) = match tt.tables().os2 {
            Some(os2) if os2.use_typographic_metrics() => (
                os2.typographic_ascender() as i32,
                os2.typographic_descender() as i32,
                os2.typographic_line_gap() as i32,
            ),
            Some(os2) => {
                let (wa, wd) = (os2.windows_ascender() as i32, os2.windows_descender() as i32); // wd ≤ 0
                let hhea_lh = hhea.ascender as i32 - hhea.descender as i32 + hhea.line_gap as i32;
                let gap = (hhea_lh - (wa - wd)).max(0);
                (wa, wd, gap)
            }
            None => (hhea.ascender as i32, hhea.descender as i32, hhea.line_gap as i32),
        };
        (asc, -desc, gap)
    }
    /// Max `(ascent, descent)` over the fonts used on byte range `[a,b)` — the mixed-font line
    /// metrics DirectWrite reports (the tallest run on the line wins). An inline object's range draws no
    /// glyphs, so its font does not count: a line holding only objects is sized by them alone (VM r19 24453:
    /// height 11.925 = the object's, not Segoe UI's descent). An empty range uses the base.
    fn line_vmetrics(&self, text: &str, em: f32, a: usize, b: usize, objs: &[ObjSpan]) -> (f32, f32) {
        let mut out: Option<(f32, f32)> = None;
        for (i, _) in text[a..b].char_indices() {
            if objs.iter().any(|o| o.s <= a + i && a + i < o.e) {
                continue;
            }
            // Per-range SetFontSize: a bigger sub-range makes the line taller, so each position's
            // metrics use its own em (the tallest run wins), not the base em.
            let (va, vd) = self.vmetrics(self.at(a + i), a + i, self.em_at(a + i).unwrap_or(em), self.upright_at(text, a + i));
            out = Some(out.map_or((va, vd), |(pa, pd)| (pa.max(va), pd.max(vd))));
        }
        out.unwrap_or_else(|| if objs.iter().any(|o| o.s < b && o.e > a) { (0.0, 0.0) } else { self.vmetrics(self.base, usize::MAX, em, false) })
    }
}

/// An inline object resolved to a byte range in the layout text.
#[derive(Clone, Copy)]
struct ObjSpan {
    s: usize,
    e: usize,
    obj: InlineObject,
}

/// The subset of IDWriteTextFormat state that affects the layout results we reproduce.
#[derive(Clone, Copy, Debug)]
pub struct TextFormat {
    /// Font size (em) in DIPs.
    pub em: f32,
    /// Reading direction: `true` = right-to-left paragraph base direction.
    pub base_rtl: bool,
    /// `DWRITE_TEXT_ALIGNMENT`: 0 leading, 1 trailing, 2 center, 3 justified (see `justify.rs`).
    pub align: u8,
    /// `DWRITE_WORD_WRAPPING`: 0 wrap, 1 no-wrap (one line per paragraph). Only these two matter
    /// to the current scenes; other modes fall back to wrapping.
    pub wrap: u8,
    /// `DWRITE_PARAGRAPH_ALIGNMENT`: 0 near (lines at the top of the layout box), 1 far (bottom),
    /// 2 center. Shifts every line's baseline vertically by a constant within the `max_height` box.
    pub para: u8,
    /// `DWRITE_FLOW_DIRECTION`: 0 top-to-bottom. Bottom-to-top (1) — and left-to-right (2), which
    /// DirectWrite's vertical alignment treats alike — stacks the lines upward from the box bottom and
    /// swaps near/far paragraph alignment.
    pub flow: u8,
    /// `SetLineSpacing` (method, lineSpacing, baseline). DEFAULT (0) keeps each line's own box; UNIFORM
    /// (1) gives every line that height and baseline; PROPORTIONAL (2) scales the line's own height and
    /// baseline by them.
    pub line_spacing: (u8, f32, f32),
}

impl TextFormat {
    /// A line box `(height, baseline)` after the format's line spacing.
    fn spaced(&self, h: f32, bl: f32) -> (f32, f32) {
        match self.line_spacing {
            (1, sh, sb) => (sh, sb),
            (2, sh, sb) => (h * sh, bl * sb),
            _ => (h, bl),
        }
    }
}

/// One glyph run as DirectWrite hands it to `IDWriteTextRenderer::DrawGlyphRun`: glyphs in
/// logical order, advances/offsets in DIPs, and the baseline origin (`x`,`y`). For an RTL
/// run `x` is the right edge (the pen start), matching native.
#[derive(Clone, Debug)]
pub struct GlyphRun {
    pub x: f32,
    /// The pen this run was placed at in its line before alignment: its left edge. A right-to-left run's `x` is its
    /// right end, and the line's alignment offset joins `left` before the run's width does, as DirectWrite adds them,
    /// so that end rounds as native's (VM: a bold Segoe UI right-to-left line at 168 DPI whose first run ends at
    /// 240.00002, not 240, moving a glyph on a half-pixel tie).
    pub left: f32,
    pub y: f32,
    /// The run box's top edge (baseline − `font_asc`), carried as its own value rather than
    /// recomputed from `y` in the caret. A vertical paragraph-alignment shift (`voffset`) moves it
    /// with `y`; reconstructing it as `y − font_asc` after a large shift loses a ULP to cancellation
    /// (the caret top of a centre/far-aligned line). Equals `y − font_asc` before any shift.
    pub top: f32,
    pub level: u8,
    pub glyphs: Vec<u16>,
    pub advances: Vec<f32>,
    pub offsets: Vec<(f32, f32)>,
    /// UTF-16 cluster map: text-unit → glyph index within this run.
    pub clusters: Vec<u16>,
    /// UTF-16 position of this run's first character in the whole string.
    pub text_pos: u32,
    pub text: String,
    /// This run's font vertical metrics in DIPs — ascent (baseline from the line top of a line of
    /// only this font) and full line height. A caret on this run uses them, so a run in a fallback
    /// face keeps that face's own box even on a line sized by a taller run. Equal to the layout's
    /// `baseline`/`line_height` for a single-font layout.
    pub font_asc: f32,
    pub font_height: f32,
    /// This run's em in DIPs (the base format em, or a per-range `SetFontSize`) — the size the
    /// renderer rasterizes its glyphs at. Equals the layout's base em for a single-size layout.
    pub em: f32,
    /// A vertical layout's run orientation: 0 horizontal; 1 rotated (drawn turned 90° clockwise about
    /// its origin, DWRITE_GLYPH_ORIENTATION_ANGLE_90_DEGREES); 2 upright (a sideways run, each glyph
    /// hanging from its vertical origin on the column centre).
    pub orient: u8,
    /// A vertical layout's offset from the central baseline to this run font's horizontal baseline (an upright
    /// run's in unrounded design units, which only its underline uses).
    pub vhalf: f32,
}

/// An inline object placed by the layout (`IDWriteTextRenderer::DrawInlineObject`): a trimming
/// ellipsis sign or a `SetInlineObject` placeholder. `x`,`y` is the top-left origin DirectWrite
/// passes to the callback (`y` = line baseline − object baseline).
#[derive(Clone, Copy, Debug)]
pub struct InlinePlacement {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub baseline: f32,
    pub rtl: bool,
    pub sideways: bool,
    /// A trimming sign's ellipsis glyph (in the format's face) and em: the sign draws it on the line's
    /// baseline across its width. `None` for an application's inline object, which the app draws.
    pub sign: Option<(u16, f32)>,
}

/// One entry in the layout's draw-order output: either a glyph run or an inline object, matching the
/// interleaving of `DrawGlyphRun`/`DrawInlineObject` callbacks a text renderer receives.
#[derive(Clone, Debug)]
pub enum Run {
    Glyph(GlyphRun),
    Inline(InlinePlacement),
}

/// An underline or strikethrough rectangle (`IDWriteTextRenderer::DrawUnderline`/`DrawStrikethrough`):
/// a filled device-space rect of the brush, one per decorated run. `y` is its top (device y-down);
/// the line's offset below (underline) or through (strikethrough) the baseline and its thickness come
/// from the run's font metrics scaled by the run's em.
#[derive(Clone, Copy, Debug)]
pub struct Decoration {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub thickness: f32,
    /// The line's offset from the run baseline (`y == baseline + offset`), so the renderer path can
    /// recover `IDWriteTextRenderer::DrawUnderline`'s baseline origin + `DWRITE_UNDERLINE.offset`.
    pub offset: f32,
    /// True for a strikethrough (`DrawStrikethrough`), false for an underline (`DrawUnderline`).
    pub strike: bool,
    /// The run baseline before pixel snapping (`y == snap(base) + offset` for an underline).
    pub base: f32,
    /// The underline offset before pixel snapping.
    pub raw_offset: f32,
    /// 1 in a vertical layout: drawn turned 90° clockwise about (`x`, `y`) like its run, the line
    /// `offset` below that origin in the turned frame; 0 otherwise.
    pub orient: u8,
}

/// Snap an underline's offset below the baseline to a whole device pixel so the line sits on the pixel
/// grid, the way DirectWrite grid-fits an underline. Round `ppd·offset` to the nearest pixel (a tie
/// going up), then force ±1 device pixel for a non-zero offset that would otherwise round to 0 so
/// the line never vanishes. Only the underline offset snaps this way — not its thickness, and not the
/// strikethrough. An offset beyond the range where a float still holds integers exactly (`2^23`) is
/// left unsnapped.
pub(crate) fn snap_underline_offset(offset: f32, ppd: f32) -> f32 {
    if ppd == 0.0 {
        return offset;
    }
    let scaled = ppd * offset;
    let mut snapped = if scaled.abs() >= 8388608.0 { scaled } else { round_half_up(scaled) };
    if snapped == 0.0 && scaled != 0.0 {
        snapped = if scaled >= 0.0 { 1.0 } else { -1.0 };
    }
    snapped / ppd
}

/// Round a device coordinate to the nearest pixel (a tie going up), the way DirectWrite grid-fits the
/// underline rect's baseline origin — while the glyph run itself
/// keeps its fractional baseline, so the underline lands on a whole pixel and the text does not.
///
/// Layout space equals the device-space snap only for an integer origin and an untranslated transform; the
/// d2d fill re-snaps `Decoration::base` in device space (`fill_decorations`). The dwrite-ffi renderer path
/// still takes this layout-space value.
fn snap_pixel(v: f32, ppd: f32) -> f32 {
    if ppd == 0.0 {
        return v;
    }
    let scaled = ppd * v;
    let snapped = if scaled.abs() >= 8388608.0 { scaled } else { round_half_up(scaled) };
    snapped / ppd
}

/// Nearest integer, a tie going up: the pixel snapper's `cvtss2si` (round to nearest even) with its
/// exact-half bump.
pub(crate) fn round_half_up(v: f32) -> f32 {
    let n = v.round_ties_even();
    if n - v == -0.5 { n + 1.0 } else { n }
}

impl Run {
    fn shift_x(&mut self, dx: f32) {
        match self {
            Run::Glyph(g) => {
                g.x += dx;
                g.left += dx;
            }
            Run::Inline(o) => o.x += dx,
        }
    }
    fn shift_y(&mut self, dy: f32) {
        match self {
            Run::Glyph(g) => {
                g.y += dy;
                g.top += dy;
            }
            Run::Inline(o) => o.y += dy,
        }
    }
    /// The glyph run, if this entry is one.
    pub fn glyph(&self) -> Option<&GlyphRun> {
        match self {
            Run::Glyph(g) => Some(g),
            Run::Inline(_) => None,
        }
    }
}

impl GlyphRun {
    /// This run's clusters as `(start_u16, width)`, each width the sum of its glyphs' advances.
    /// Used to record *justified* cluster widths, which DWrite's GetClusterMetrics reports (not the
    /// natural widths `cluster_metrics` re-shapes). Cluster = maximal text-unit span sharing a first
    /// glyph, exactly as [`Layout::hit_clusters`] splits.
    fn cluster_widths(&self) -> Vec<(u32, f32)> {
        let cl = &self.clusters;
        let mut out = Vec::new();
        let mut u = 0;
        while u < cl.len() {
            let g = cl[u] as usize;
            let mut u2 = u + 1;
            while u2 < cl.len() && cl[u2] as usize == g {
                u2 += 1;
            }
            let g_next = if u2 < cl.len() { cl[u2] as usize } else { self.advances.len() };
            out.push((self.text_pos + u as u32, self.advances[g..g_next].iter().sum()));
            u = u2;
        }
        out
    }
}

/// DWRITE_LINE_METRICS (lengths are UTF-16 code-unit counts).
#[derive(Clone, Copy, Debug)]
pub struct LineMetrics {
    pub length: u32,
    pub trailing_ws: u32,
    pub newline: u32,
    pub height: f32,
    pub baseline: f32,
    pub is_trimmed: bool,
}

/// DWRITE_CLUSTER_METRICS: `bits` packs the flag bitfield (bit0 canWrapLineAfter,
/// bit1 isWhitespace, bit2 isNewline, bit3 isSoftHyphen, bit4 isRightToLeft).
#[derive(Clone, Copy, Debug)]
pub struct ClusterMetric {
    pub width: f32,
    pub length: u16,
    pub bits: u16,
}

/// DWRITE_TEXT_METRICS.
#[derive(Clone, Copy, Debug)]
pub struct TextMetrics {
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub width_trailing: f32,
    pub height: f32,
    pub layout_width: f32,
    pub layout_height: f32,
    pub bidi_depth: u32,
    pub line_count: u32,
}

/// The full result of laying out a string, mirroring what an `IDWriteTextLayout` reports.
pub struct Layout {
    /// Draw-order output: glyph runs and inline objects (per line, then visual order; a hard
    /// newline ends a line with a zero-glyph run, as DirectWrite emits).
    pub runs: Vec<Run>,
    /// Underline / strikethrough rectangles to fill with the text brush, in draw order (all after the
    /// glyph runs, as DirectWrite issues `DrawUnderline`/`DrawStrikethrough` after `DrawGlyphRun`).
    pub decorations: Vec<Decoration>,
    pub lines: Vec<LineMetrics>,
    pub clusters: Vec<ClusterMetric>,
    pub metrics: TextMetrics,
    /// Baseline-from-line-top and line height, both constant for a single font.
    pub baseline: f32,
    pub line_height: f32,
    total_u16: u32,
    /// Trimmed-away UTF-16 ranges `[start,end)` with the caret DirectWrite returns for interior
    /// positions (ellipsis leading edge) and for the end-of-text position (its trailing edge, width
    /// 0). Empty when nothing is trimmed.
    trimmed_spans: Vec<(u32, u32, Caret, Caret)>,
    /// Inline-object UTF-16 ranges `[start,end)` and the placed object box, so a hit test on an
    /// object position returns the object's own box.
    objects: Vec<(u32, u32, InlinePlacement)>,
    /// The end-of-text caret when the layout ends in a trailing empty line (empty text, or text that
    /// ends in a hard line break): DirectWrite puts that caret on the fresh empty line below, which
    /// carries no glyph run. `None` when the last content line owns the end caret.
    empty_line_caret: Option<Caret>,
    /// A vertical layout's box width, across which its columns stack (device x = width − column-frame y); `None` for
    /// a horizontal layout. Its trimming signs keep the column frame and draw through it.
    column_width: Option<f32>,
    /// How far a vertical layout's upright trimming sign draws below its box top ([`Trimming::upright_bold`]).
    sign_lift: f32,
}

/// Result of `IDWriteTextLayout::HitTestTextPosition`: the caret point (`x`,`y`) and the
/// DWRITE_HIT_TEST_METRICS box of the character at that position.
#[derive(Clone, Copy, Debug)]
pub struct Caret {
    pub x: f32,
    pub y: f32,
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
    pub bidi: u8,
}

/// `DWRITE_HIT_TEST_METRICS`: a text-range or hit-test box plus its flags. `is_trailing`/
/// `is_inside` are only set by [`Layout::hit_test_point`].
#[derive(Clone, Copy, Debug)]
pub struct HitMetrics {
    pub pos: u32,
    pub length: u32,
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
    pub bidi: u8,
    pub is_text: bool,
    pub is_trimmed: bool,
    pub is_trailing: bool,
    pub is_inside: bool,
}

/// One cluster placed on a line (visual geometry), the atom both hit-test queries work over.
struct HitCluster {
    line: usize,
    top: f32,
    height: f32,
    pos: u32,
    length: u32,
    left: f32,
    width: f32,
    bidi: u8,
}

impl Layout {
    /// The glyph runs (skipping inline objects) in draw order.
    pub(crate) fn glyph_runs(&self) -> impl Iterator<Item = &GlyphRun> {
        self.runs.iter().filter_map(Run::glyph)
    }

    /// What a draw paints, in order: the glyph runs and each trimming sign as a run of its ellipsis glyph on the
    /// line's baseline (VM r25 30734: Impact's dots under a fully trimmed line). A sign's run has no text position
    /// (`u32::MAX`), so it draws with the sign's face. A vertical layout's sign stands upright in its column, from the
    /// start of its box along it (VM r25 30076: Leelawadee UI read bottom to top shows the vertical ellipsis turned with
    /// it; seven faces, both reading directions). The glyph runs draw in text order, as DirectWrite's layout walks its
    /// runs, not their visual order, so where runs overlap a pixel the later one in the text blends last (VM r25 30298: a
    /// bold '3' reaching into the Hebrew run before it); the signs, with no text position, come after them.
    pub(crate) fn drawn_runs(&self) -> impl Iterator<Item = std::borrow::Cow<'_, GlyphRun>> {
        let mut runs: Vec<&Run> = self.runs.iter().collect();
        runs.sort_by_key(|r| match r {
            Run::Glyph(g) => g.text_pos,
            Run::Inline(_) => u32::MAX,
        });
        runs.into_iter().filter_map(|r| match r {
            Run::Glyph(g) => Some(std::borrow::Cow::Borrowed(g)),
            Run::Inline(p) => p.sign.filter(|_| !p.sideways || self.column_width.is_some()).map(|(id, em)| {
                let (x, y, orient) = match self.column_width.filter(|_| p.sideways) {
                    // Its place across the columns rounds to a whole unit in the column frame (VM r25 30141: Georgia in a
                    // 37.49-DIP box; ponytail: probed at 96 DPI only, where a unit is a pixel).
                    Some(w) => (w - (p.y + p.baseline).round(), p.x + self.sign_lift, 2),
                    None => (p.x, p.y + p.baseline, 0),
                };
                std::borrow::Cow::Owned(GlyphRun {
                    x,
                    left: x,
                    y,
                    top: if orient == 2 { y } else { p.y },
                    level: 0,
                    glyphs: vec![id],
                    advances: vec![p.width],
                    offsets: vec![(0.0, 0.0)],
                    clusters: Vec::new(),
                    text_pos: u32::MAX,
                    text: String::new(),
                    font_asc: 0.0,
                    font_height: 0.0,
                    em,
                    orient,
                    vhalf: 0.0,
                })
            }),
        })
    }

    /// `IDWriteTextLayout::DetermineMinWidth`: the smallest layout width that needs no word break —
    /// the widest unbreakable cluster segment, trailing whitespace excluded (the same per-segment
    /// visible width [`wrap_lines`] fits). A segment ends after a `canWrapLineAfter` cluster (bit 0);
    /// whitespace clusters (bit 1) don't extend the visible width.
    pub fn min_width(&self) -> f32 {
        let (mut min_w, mut seg, mut vis) = (0.0f32, 0.0f32, 0.0f32);
        for c in &self.clusters {
            seg += c.width;
            if c.bits & 2 == 0 {
                vis = seg; // non-whitespace extends the visible width to here
            }
            if c.bits & 1 != 0 {
                min_w = min_w.max(vis);
                seg = 0.0;
                vis = 0.0;
            }
        }
        min_w.max(vis)
    }

    /// The top (from the layout origin) of each line, honouring per-line heights.
    fn line_tops(&self) -> Vec<f32> {
        let mut tops = Vec::with_capacity(self.lines.len());
        let mut acc = 0.0;
        for l in &self.lines {
            tops.push(acc);
            acc += l.height;
        }
        tops
    }

    /// Every glyph cluster placed on its line, in logical (text) order, with visual x extents —
    /// the shared basis for `HitTestPoint` and `HitTestTextRange`.
    fn hit_clusters(&self) -> Vec<HitCluster> {
        let tops = self.line_tops();
        let mut out = Vec::new();
        for r in self.glyph_runs() {
            if r.glyphs.is_empty() {
                continue; // newline / zero-glyph run: contributes no box
            }
            // Line index: the line whose baseline matches this run's baseline origin.
            let line = (0..self.lines.len())
                .find(|&i| (tops[i] + self.lines[i].baseline - r.y).abs() < 0.5)
                .unwrap_or(0);
            let (top, height) = (tops[line], self.lines[line].height);
            let rtl = r.level & 1 == 1;
            let cl = &r.clusters;
            let mut u = 0usize;
            while u < cl.len() {
                let g = cl[u] as usize;
                let mut u2 = u + 1;
                while u2 < cl.len() && cl[u2] as usize == g {
                    u2 += 1;
                }
                let g_next = if u2 < cl.len() { cl[u2] as usize } else { r.advances.len() };
                let width: f32 = r.advances[g..g_next].iter().sum();
                let prefix: f32 = r.advances[..g].iter().sum();
                let left = if rtl { r.x - prefix - width } else { r.x + prefix };
                out.push(HitCluster {
                    line,
                    top,
                    height,
                    pos: r.text_pos + u as u32,
                    length: (u2 - u) as u32,
                    left,
                    width,
                    bidi: r.level,
                });
                u = u2;
            }
        }
        out
    }

    /// `IDWriteTextLayout::HitTestPoint`: the cluster a device point falls in, whether the point is
    /// past that cluster's leading half (trailing hit) and whether it lies inside the text.
    pub fn hit_test_point(&self, x: f32, y: f32) -> HitMetrics {
        let clusters = self.hit_clusters();
        let tops = self.line_tops();
        let total_h: f32 = self.lines.iter().map(|l| l.height).sum();
        // Pick the line: clamp the point vertically onto the laid-out lines.
        let line = (0..self.lines.len())
            .find(|&i| y >= tops[i] && y < tops[i] + self.lines[i].height)
            .unwrap_or(if y < 0.0 { 0 } else { self.lines.len().saturating_sub(1) });
        let mut on: Vec<&HitCluster> = clusters.iter().filter(|c| c.line == line).collect();
        on.sort_by(|a, b| a.left.total_cmp(&b.left)); // visual order
        let inside_y = y >= 0.0 && y < total_h;
        let (line_left, line_right) = match (on.first(), on.last()) {
            (Some(f), Some(l)) => (f.left, l.left + l.width),
            _ => (0.0, 0.0),
        };
        let metrics = |c: &HitCluster, is_trailing: bool, is_inside: bool| HitMetrics {
            pos: c.pos,
            length: c.length,
            left: c.left,
            top: c.top,
            width: c.width,
            height: c.height,
            bidi: c.bidi,
            is_text: true,
            is_trimmed: false,
            is_trailing,
            is_inside,
        };
        if on.is_empty() {
            return HitMetrics { pos: 0, length: 0, left: 0.0, top: tops.get(line).copied().unwrap_or(0.0), width: 0.0, height: self.line_height, bidi: 0, is_text: true, is_trimmed: false, is_trailing: false, is_inside: false };
        }
        if x < line_left {
            let c = on[0];
            return metrics(c, c.bidi & 1 == 1, false);
        }
        if x >= line_right {
            let c = on[on.len() - 1];
            return metrics(c, c.bidi & 1 == 0, false);
        }
        let c = on.iter().find(|c| x >= c.left && x < c.left + c.width).copied().unwrap_or(on[on.len() - 1]);
        let mid = c.left + c.width * 0.5;
        let trailing = if c.bidi & 1 == 1 { x < mid } else { x >= mid };
        metrics(c, trailing, inside_y)
    }

    /// `IDWriteTextLayout::HitTestTextRange`: one box per maximal same-line, same-direction stretch
    /// of the text range `[pos, pos+len)`, matching how DirectWrite segments a range for selection.
    pub fn hit_test_text_range(&self, pos: u32, len: u32) -> Vec<HitMetrics> {
        let end = pos + len;
        let mut covered: Vec<HitCluster> = self.hit_clusters().into_iter().filter(|c| c.pos < end && c.pos + c.length > pos).collect();
        covered.sort_by_key(|c| c.pos);
        let mut out: Vec<HitMetrics> = Vec::new();
        for c in covered {
            // Extend the current box when the next cluster shares its line and direction and is
            // the immediate logical successor; otherwise start a new box.
            match out.last_mut() {
                Some(b) if b.top == c.top && b.bidi == c.bidi && b.pos + b.length == c.pos => {
                    let right = (b.left + b.width).max(c.left + c.width);
                    b.left = b.left.min(c.left);
                    b.width = right - b.left;
                    b.length += c.length;
                }
                _ => out.push(HitMetrics {
                    pos: c.pos,
                    length: c.length,
                    left: c.left,
                    top: c.top,
                    width: c.width,
                    height: c.height,
                    bidi: c.bidi,
                    is_text: true,
                    is_trimmed: false,
                    is_trailing: false,
                    is_inside: false,
                }),
            }
        }
        out
    }

    /// Leading-edge hit test of UTF-16 position `pos` (clamped to the text length), mirroring
    /// `HitTestTextPosition(pos, isTrailingHit=false)`. The caret point `x` is the leading edge
    /// (right side for an RTL run); `width` is the character's advance.
    pub fn hit_test_text_position(&self, pos: u32) -> Caret {
        let p = pos.min(self.total_u16);
        // A position inside an inline object returns the object's own box.
        if let Some((_, _, o)) = self.objects.iter().find(|(s, e, _)| p >= *s && p < *e) {
            return Caret { x: o.x, y: o.y, left: o.x, top: o.y, width: o.width, height: o.height, bidi: o.rtl as u8 };
        }
        // A trimmed-away position maps to its line's ellipsis sign: the end-of-text position to the
        // sign's trailing edge, interior positions to its leading edge.
        if let Some((_, _, lead, trail)) = self.trimmed_spans.iter().find(|(s, e, ..)| p >= *s && p < *e) {
            return if p == self.total_u16 { *trail } else { *lead };
        }
        // End-of-text on a trailing empty line (empty text, or text ending in a hard break): the caret
        // sits on that fresh line, which owns no glyph run, so answer it before the run search.
        if p == self.total_u16 {
            if let Some(c) = self.empty_line_caret {
                return c;
            }
        }
        // The run that owns position `p` (half-open text range), else the run ending at `p`
        // (end-of-text / end-of-line caret).
        let within = self.glyph_runs().find(|r| r.text_pos <= p && p < r.text_pos + r.clusters.len() as u32);
        // Trimmed-away positions belong to no kept run; DirectWrite collapses their caret to the end
        // of the visible content, which is the last glyph run's trailing edge (`within` stays None).
        let Some(r) = within
            .or_else(|| self.glyph_runs().find(|r| r.text_pos + r.clusters.len() as u32 == p))
            .or_else(|| self.glyph_runs().last())
        else {
            // No glyph run at all (a degenerate run-less layout): fall back to the empty-line caret,
            // else a zero-width caret at the layout origin — never panic where native returns a caret.
            return self.empty_line_caret.unwrap_or(Caret { x: 0.0, y: 0.0, left: 0.0, top: 0.0, width: 0.0, height: self.line_height, bidi: 0 });
        };
        let rtl = r.level & 1 == 1;
        let (g, prefix) = match within {
            Some(_) => {
                let g = r.clusters[(p - r.text_pos) as usize] as usize;
                (Some(g), r.advances[..g].iter().sum::<f32>())
            }
            None => (None, r.advances.iter().sum::<f32>()), // trailing edge of the run
        };
        let x = if rtl { r.x - prefix } else { r.x + prefix };
        let width = g.and_then(|g| r.advances.get(g)).copied().unwrap_or(0.0);
        // A text caret uses the run font's own box (`font_asc`/`font_height`) sitting on the line's
        // baseline — so on a line raised by a tall inline object or a taller fallback run the caret
        // keeps this run's font height (single-font: these equal `baseline`/`line_height`).
        let top = r.top;
        Caret {
            x,
            y: top,
            left: if rtl { x - width } else { x },
            top,
            width,
            height: r.font_height,
            bidi: r.level,
        }
    }
}

/// UTF-16 code-unit offset of a UTF-8 byte offset in `text`.
fn u16_pos(text: &str, byte: usize) -> u32 {
    text[..byte].encode_utf16().count() as u32
}

/// Byte offset of a UTF-16 code-unit offset in `text` (inverse of [`u16_pos`]).
fn u16_to_byte(text: &str, pos: u32) -> usize {
    let mut u = 0u32;
    for (b, ch) in text.char_indices() {
        if u >= pos {
            return b;
        }
        u += ch.len_utf16() as u32;
    }
    text.len()
}

/// The rustybuzz features active over byte range `[s,e)` — those whose UTF-16 span covers it. Runs
/// are split at feature boundaries (see [`itemize`]) so a sub-run is wholly inside or outside each.
///
/// Once the layout carries any typography, DirectWrite stops auto-applying its on-by-default discretionary
/// features (kern, liga, clig, calt) to every run (verified: text2c-04 ops 2/3 leave even uncovered runs
/// unkerned, op4 re-enables with an explicit kern=1; VM: Calibri's 'ffi', 'fl' and 'fi' ligatures go under an
/// unrelated 'onum' range and come back with liga=1; r25 30049). So when features exist, those a run does not
/// set are turned off. Those defaults belong to the shaping library's generic engine; the Arabic, Indic,
/// Khmer and Myanmar engines shape and place glyphs from their own feature lists and keep them (VM: Segoe UI
/// Arabic kerns the same under an unrelated 'onum' range, Latin, Cyrillic, Greek and Hebrew do not; r25 30231).
fn active_features(feats: &[Feature], text: &str, s: usize, e: usize) -> Vec<rustybuzz::Feature> {
    use rustybuzz::ttf_parser::Tag;
    let (s16, e16) = (u16_pos(text, s), u16_pos(text, e));
    let mut v: Vec<rustybuzz::Feature> = feats
        .iter()
        .filter(|f| f.start <= s16 && e16 <= f.end)
        .map(|f| rustybuzz::Feature::new(Tag::from_bytes(&f.tag), f.value, ..))
        .collect();
    let kern = Tag::from_bytes(b"kern");
    let own_placement = text[s..e].chars().map(|c| c.script()).find(|sc| !matches!(sc, Script::Common | Script::Inherited)).is_some_and(|sc| {
        use Script::*;
        matches!(sc, Arabic | Syriac | Devanagari | Bengali | Gurmukhi | Gujarati | Oriya | Tamil | Telugu | Kannada | Malayalam | Khmer | Myanmar)
    });
    if !feats.is_empty() && !own_placement {
        for tag in [kern, Tag::from_bytes(b"liga"), Tag::from_bytes(b"clig"), Tag::from_bytes(b"calt")] {
            if !v.iter().any(|f| f.tag == tag) {
                v.push(rustybuzz::Feature::new(tag, 0, ..));
            }
        }
    }
    v
}

/// Shape one run with explicit features. Mirrors [`shaping::shape`] (whose feature list is fixed to
/// the shaper defaults); duplicated here because that owned-by-another-module API takes no features.
/// Shape `text[s..e]` with `ctx` = the surrounding span of the same face and bidi level as shaping context:
/// DirectWrite shapes such a span as one, so a run split off by a format change still joins its neighbours
/// (VM r25 30260: a lone 'م' before more Arabic takes its initial form), while a neighbour in another face
/// does not (r25 30189: 'ح' before a Times New Roman 'ب' stays final).
/// `upright`: a vertical layout's upright run shapes top to bottom (the `vert` forms), each glyph advancing by
/// its vertical advance and hanging from its vertical origin with no offset of its own.
fn shape_run(font: &Font, text: &str, s: usize, e: usize, ctx: (usize, usize), rtl: bool, feats: &[rustybuzz::Feature], upright: bool) -> Vec<Glyph> {
    let mut buf = rustybuzz::UnicodeBuffer::new();
    // DirectWrite's Thai/Lao engine prepares the text over the whole shaping span: a mark with nothing to sit on
    // shapes onto a dotted circle of its own (VM: the tone mark of 'ศะ้' after its sara a), and a few marks move.
    let span = &text[ctx.0..ctx.1];
    let thai_lao = span.chars().map(|c| unicode_script::UnicodeScript::script(&c)).find(|sc| !matches!(sc, unicode_script::Script::Common | unicode_script::Script::Inherited));
    match thai_lao {
        Some(unicode_script::Script::Thai | unicode_script::Script::Lao) => {
            for (ch, b) in shaping::thai_lao_prepare(span) {
                let b = ctx.0 + b;
                if (s..e).contains(&b) {
                    buf.add(ch, (b - s) as u32);
                }
            }
        }
        _ => buf.push_str(&text[s..e]),
    }
    buf.set_pre_context(&text[ctx.0..s]);
    buf.set_post_context(&text[e..ctx.1]);
    // DirectWrite shows a variation selector on its own with the font's glyph and advance (a lone VS16 the
    // face lacks: its .notdef box at full advance, VM r25 30422), where HarfBuzz would hide it; any other
    // default-ignorable character alone in its cluster is hidden as HarfBuzz does, below.
    buf.set_flags(rustybuzz::BufferFlags::PRESERVE_DEFAULT_IGNORABLES);
    buf.set_direction(match (upright, rtl) {
        (true, _) => rustybuzz::Direction::TopToBottom,
        (false, true) => rustybuzz::Direction::RightToLeft,
        (false, false) => rustybuzz::Direction::LeftToRight,
    });
    let out = rustybuzz::shape(&font.hb(), feats, buf);
    let (infos, pos) = (out.glyph_infos(), out.glyph_positions());
    // A top-to-bottom (upright) run comes back in logical order whatever its level; only a right-to-left one is
    // visual (r25 30497: an upright emoji ZWJ run of a bottom-to-top column keeps its logical glyph order).
    let visual_rtl = rtl && !upright;
    let vadvance = if upright { font.default_vadvance() } else { None };
    let mut glyphs: Vec<Glyph> = infos
        .iter()
        .zip(pos)
        .map(|(i, p)| match upright {
            // ponytail: GPOS vertical placement (vpal/vkrn offsets) is dropped; add it when a scene needs it.
            true => Glyph { id: i.glyph_id as u16, cluster: i.cluster, advance: vadvance.filter(|_| p.y_advance != 0).unwrap_or(-p.y_advance), dx: 0, dy: 0 },
            false => Glyph { id: i.glyph_id as u16, cluster: i.cluster, advance: p.x_advance, dx: p.x_offset, dy: p.y_offset },
        })
        .collect();
    // A default ignorable other than a variation selector, alone in its cluster, shows as the space glyph at
    // zero advance (VM r25 30086: a lone ZWJ in Consolas is glyph 3, not the font's own ZWJ glyph).
    let space = font.tt().glyph_index(' ').map_or(0, |g| g.0);
    for k in 0..glyphs.len() {
        let c = glyphs[k].cluster as usize;
        let Some(ch) = text[s + c..e].chars().next() else { continue };
        let alone = glyphs.iter().filter(|g| g.cluster as usize == c).count() == 1
            && glyphs.iter().all(|g| g.cluster as usize <= c || g.cluster as usize >= c + ch.len_utf8())
            && (c + ch.len_utf8() == e - s || glyphs.iter().any(|g| g.cluster as usize == c + ch.len_utf8()));
        // A hidden ignorable the font did not ligate is hidden inside a cluster too (r25 30330: a ZWJ ending its run
        // after 👩 shows as Segoe UI Emoji's space glyph, not its own ZWJ glyph).
        let next = glyphs.iter().map(|g| g.cluster as usize).filter(|&x| x > c).min().unwrap_or(e - s);
        let own = text[s + c..s + next].chars().any(|h| hidden_ignorable(h) && font.tt().glyph_index(h).map_or(0, |g| g.0) == glyphs[k].id);
        if (alone && (hidden_ignorable(ch) || ch == '\t')) || (!alone && own) {
            glyphs[k] = Glyph { id: space, cluster: glyphs[k].cluster, advance: 0, dx: 0, dy: 0 };
        }
    }
    // A ZWJ starting the run is a cluster of its own. DirectWrite clusters a run's text as Unicode's extended
    // grapheme clusters, where a ZWJ joins the pictograph after it only when a pictograph comes before it (UAX #29
    // GB11); HarfBuzz joins a pictograph to any ZWJ before it (r25 30497: a range boundary inside 👨 starts the run
    // "ZWJ 👩 ZWJ 👧", which DirectWrite clusters [0, 1, 1, 1, 1, 1]). Only when the ZWJ kept a glyph of its own.
    if text[s..e].starts_with('\u{200D}') && glyphs.iter().filter(|g| g.cluster == 0).count() > 1 {
        let zwj = font.tt().glyph_index('\u{200D}').map_or(0, |g| g.0);
        // A right-to-left run comes back in visual order, its logically first glyph last.
        let lead = if visual_rtl { glyphs.len() - 1 } else { 0 };
        if glyphs[lead].cluster == 0 && (glyphs[lead].id == space || glyphs[lead].id == zwj) {
            for (k, g) in glyphs.iter_mut().enumerate() {
                if g.cluster == 0 && k != lead {
                    g.cluster = '\u{200D}'.len_utf8() as u32;
                }
            }
        }
    }
    // HarfBuzz joins a regional-indicator pair into one cluster; DirectWrite keeps one cluster per indicator
    // when the font draws them apart (r25 30703: a too-narrow box breaks 🇺🇸 between its indicators).
    let ri = |ch: Option<char>| ch.is_some_and(|c| matches!(c as u32, 0x1F1E6..=0x1F1FF));
    for k in 0..glyphs.len() {
        let c = glyphs[k].cluster as usize;
        let mut it = text[s + c..e].chars();
        if k > 0 && glyphs[k - 1].cluster as usize == c && ri(it.next()) && ri(it.next()) && glyphs.iter().filter(|g| g.cluster as usize == c).count() == 2 {
            // HarfBuzz returns a right-to-left run in visual order: there the first of the two is the second indicator.
            glyphs[if visual_rtl { k - 1 } else { k }].cluster = (c + 4) as u32;
        }
    }
    // An upright cluster stacks in one vertical advance, carried by its last advancing glyph (VM r25 30743, 30046:
    // the family ZWJ sequence's glyphs [1285, 1466, 1273] advance [0, em, 0] down the column).
    if upright {
        for k in 0..glyphs.len() {
            if glyphs[k].advance != 0 && glyphs[k + 1..].iter().any(|g| g.cluster == glyphs[k].cluster && g.advance != 0) {
                glyphs[k].advance = 0;
            }
        }
    }
    if visual_rtl {
        glyphs.reverse();
    }
    // Callers read clusters in logical order (each cluster ends where the next starts). A featured RTL run can
    // come back already in logical order, which the reversal then flips (r25 30028: Hebrew under `smcp` gave
    // clusters [2, 0], a reversed byte range), so restore cluster order; glyphs within a cluster keep theirs.
    glyphs.sort_by_key(|g| g.cluster);
    glyphs
}

/// A default-ignorable character DirectWrite hides (shown as the space glyph at zero advance): every one but
/// the variation selectors.
fn hidden_ignorable(ch: char) -> bool {
    matches!(ch as u32, 0x00AD | 0x034F | 0x061C | 0x115F..=0x1160 | 0x17B4..=0x17B5 | 0x180B..=0x180F | 0x200B..=0x200F | 0x202A..=0x202E
        | 0x2060..=0x206F | 0x3164 | 0xFEFF | 0xFFA0 | 0xFFF0..=0xFFF8 | 0x1BCA0..=0x1BCA3 | 0x1D173..=0x1D17A | 0xE0000..=0xE00FF)
}

/// Whether the character at byte `b` continues the cluster before it: a combining mark, a joiner or
/// selector, an emoji modifier, tag or keycap, or the character after a ZWJ or a virama. A regional indicator
/// does not continue its partner: native splits a flag pair at a family boundary (r25 30858).
fn continues_cluster(text: &str, b: usize) -> bool {
    let (Some(ch), Some(prev)) = (text.get(b..).and_then(|t| t.chars().next()), text.get(..b).and_then(|t| t.chars().next_back())) else { return false };
    // A consonant after a virama joins its conjunct (r25 30247: a range starting on the त of स्ते).
    prev == '\u{200D}'
        || matches!(prev as u32, 0x094D | 0x09CD | 0x0A4D | 0x0ACD | 0x0B4D | 0x0BCD | 0x0C4D | 0x0CCD | 0x0D4D | 0x0DCA)
        || matches!(ch as u32,
            0x0300..=0x036F | 0x0483..=0x0489 | 0x0591..=0x05BD | 0x05BF | 0x05C1..=0x05C2 | 0x05C4..=0x05C5 | 0x05C7
            | 0x0610..=0x061A | 0x064B..=0x065F | 0x0670 | 0x06D6..=0x06DC | 0x06DF..=0x06E4 | 0x06E7..=0x06E8 | 0x06EA..=0x06ED
            | 0x0900..=0x0903 | 0x093A..=0x093C | 0x093E..=0x094F | 0x0951..=0x0957 | 0x0962..=0x0963
            | 0x0E31 | 0x0E34..=0x0E3A | 0x0E47..=0x0E4E
            | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x200C..=0x200D | 0x20D0..=0x20FF | 0xFE00..=0xFE0F | 0xFE20..=0xFE2F
            | 0x1F3FB..=0x1F3FF | 0xE0020..=0xE007F | 0xE0100..=0xE01EF)
}

/// Whitespace a line can end in as trailing whitespace; a tab is content (VM r25 30843: a line ending in a
/// tab reports no trailing whitespace).
fn is_trailing_space(ch: char) -> bool {
    ch.is_whitespace() && ch != '\t'
}

/// The first tab stop past `pen`, stops every `step` from the line start.
fn next_tab_stop(pen: f32, step: f32) -> f32 {
    if !(step > 0.0) {
        return pen;
    }
    ((pen / step).floor() + 1.0) * step
}

/// Is `ch` counted as a line break (paragraph separator) by DirectWrite line metrics?
fn is_newline(ch: char) -> bool {
    matches!(ch, '\n' | '\r' | '\u{0085}' | '\u{2028}' | '\u{2029}' | '\u{000B}' | '\u{000C}')
}

/// Itemize `[s,e)` into runs of constant bidi level and script; Common/Inherited extend
/// the current script (DirectWrite and HarfBuzz both do this). `breaks` are extra forced split
/// offsets (typography feature-range boundaries), so each run is wholly inside or outside each.
fn itemize(text: &str, s: usize, e: usize, levels: &[Level], breaks: &[usize]) -> Vec<(usize, usize, u8)> {
    let mut runs = Vec::new();
    let (mut rs, mut rl, mut rsc) = (s, levels[s].number(), None::<Script>);
    // A neutral character takes the script of the text before it, as DirectWrite's script analysis runs over the
    // whole text ahead of bidi: a space after Latin that bidi puts at a right-to-left level is a Latin run of its
    // own, not the start of the Hebrew run after it (r25 30493: 'abc אבג 123' in a right-to-left paragraph).
    let known = |c: char| Some(c.script()).filter(|&c| c != Script::Common && c != Script::Inherited);
    let mut prev = text[..s].chars().rev().find_map(known);
    for (b, ch) in text[s..e].char_indices().map(|(i, c)| (s + i, c)) {
        let lvl = levels[b].number();
        let sc_eff = known(ch).or(prev);
        prev = sc_eff;
        let split = b > rs && (lvl != rl || breaks.contains(&b) || matches!((rsc, sc_eff), (Some(a), Some(c)) if a != c));
        if split {
            runs.push((rs, b, rl));
            rs = b;
            rl = lvl;
            rsc = sc_eff;
        } else if rsc.is_none() {
            rsc = sc_eff;
        }
    }
    runs.push((rs, e, rl));
    runs
}

/// Each glyph's (advance, ascender) offset in DIPs: the shaper's placement, under a GDI mode in whole shaping pixels.
#[allow(clippy::too_many_arguments)]
fn glyph_offsets(fonts: &Fonts, font: &Font, s: usize, glyphs: &[Glyph], advances: &[f32], scale: f32, rtl: bool, up: bool, gdi: Option<&super::gdi::GdiMeasure>, gdi_rtl: Option<&super::gdi::GdiMeasure>) -> Vec<(f32, f32)> {
    let bold = fonts.bold_delta(font, s) != 0;
    let (mut pen_du, mut pen) = (0i32, 0f32);
    // The last advancing glyph, the base a mark hangs on: (id, shaper's placement, design advance, design pen, pen,
    // reported offset, own advance).
    let mut base: Option<(u16, i32, i32, i32, f32, f32, f32)> = None;
    let tt = font.tt();
    glyphs
        .iter()
        .zip(advances)
        .map(|(g, &a)| {
            // Under a GDI mode a zero-advance glyph on a mark anchor of the advancing glyph before it rounds its anchors
            // and placement separately ([`super::gdi::GdiMeasure::mark_offset`]). The shaper's placement is the base's
            // anchor less the mark's, plus the base's own placement and the mark's remainder, less the design advances
            // between them in visual order (a right-to-left mark sits left of its base, its pen past the base's advance).
            let mark = gdi.or(gdi_rtl).zip(base).filter(|_| g.advance == 0).and_then(|(m, (b, b_dx, b_adv, b_pen_du, b_pen, b_x, b_own))| {
                let (ba, ma) = font.mark_anchors(b, g.id)?;
                let between = if rtl { b_adv + b_pen_du - pen_du } else { pen_du - b_pen_du };
                m.mark_offset((ba, ma), g.dx - (ba - ma) - b_dx + between, rtl, if rtl { b_x + b_own } else { b_x }, pen - b_pen)
            });
            // Under design advances a left-to-right mark keeps to its base, so the base's bold widening comes off its
            // placement (r25 30086, 30902: Segoe UI Emoji's family girl 41 design units, round(upem/50), further left). A
            // right-to-left mark's offset already counts it through the base's own advance.
            let widen = base
                .filter(|&(b, ..)| bold && gdi.is_none() && !rtl && !up && g.advance == 0 && font.has_outline(b) && font.mark_anchors(b, g.id).is_some())
                .map_or(0, |_| fonts.bold_delta(font, s));
            let x = gdi.filter(|_| g.dx != 0).and_then(|m| m.offset(pen_du + g.dx, pen)).unwrap_or((g.dx - widen) as f32 * scale);
            // A right-to-left glyph's offset is DirectWrite's: the renderer ends the glyph's own advance at the pen
            // and points the offset left (`text::rtl_glyphs`), so the shaper's visual offset becomes
            // `advance − own − offset` (a kern moves the advance alone; VM: native reports no offset for it).
            // Under a GDI mode the shaper's offset rounds to shaping pixels on its own; a zero-advance glyph instead
            // lands on its shaping position in whole pixels from the run's right edge, less the whole-pixel pen (the
            // left-to-right rule mirrored). r25 census: every non-bold right-to-left GDI glyph.
            // ponytail: a bold-simulated run's zero-advance glyphs off a mark anchor keep the unsnapped offset; native
            // lands them a pixel or two from this rule, the bold pixels' share of the pen being open.
            let (x, own) = if rtl && !up {
                let own = fonts.advance(font, s, g.id, tt.glyph_hor_advance(rustybuzz::ttf_parser::GlyphId(g.id)).map_or(0, i32::from), scale);
                let rest = a - own;
                let x = match gdi_rtl.and_then(|m| m.offset(0, 0.0).map(|_| m)) {
                    Some(_) if g.dx != 0 && g.advance == 0 && bold => rest - x,
                    Some(m) if g.dx != 0 && g.advance == 0 => rest + m.offset(pen_du - g.dx, pen).unwrap_or(0.0),
                    Some(m) if g.dx != 0 => rest + m.offset(-g.dx, 0.0).unwrap_or(0.0),
                    Some(_) => rest,
                    None => rest - x,
                };
                (x, own)
            } else {
                (x, 0.0)
            };
            let x = mark.unwrap_or(x);
            if g.advance != 0 {
                base = Some((g.id, g.dx, g.advance, pen_du, pen, x, own));
            }
            (pen_du, pen) = (pen_du + g.advance, pen + a);
            (x, gdi.or(gdi_rtl).and_then(|m| m.offset_y(g.dy)).unwrap_or(g.dy as f32 * scale))
        })
        .collect()
}

/// An upright cluster's glyph offsets from the run's horizontal shaping (`hg`, its advances `ha` and offsets `ho`): across
/// the column `x − w/2 + own/2` for the glyph's horizontal position `x`, its cluster's horizontal advance `w` and its own
/// advance, in design units halved whole, or whole pixels under a GDI mode; along it, back over the vertical advances of
/// the cluster's glyphs before it.
#[allow(clippy::too_many_arguments)]
fn upright_offsets(fonts: &Fonts, font: &Font, s: usize, hg: &[Glyph], ha: &[f32], ho: &[(f32, f32)], vadv: &[f32], scale: f32, gdi: Option<&super::gdi::GdiMeasure>, out: &mut [(f32, f32)]) {
    let delta = fonts.bold_delta(font, s);
    let tt = font.tt();
    let own_du = |g: &Glyph| tt.glyph_hor_advance(rustybuzz::ttf_parser::GlyphId(g.id)).map_or(0, i32::from);
    // The design-unit width a natural layout gives a glyph: the bold widening on an inked advancing glyph.
    let widened = |id: u16, adv: i32| adv + if delta != 0 && adv != 0 && font.has_outline(id) { delta } else { 0 };
    let mut i = 0;
    while i < hg.len() {
        let end = (i..hg.len()).find(|&k| hg[k].cluster != hg[i].cluster).unwrap_or(hg.len());
        let (mut x, mut x_du, mut along) = (0.0f32, 0i32, 0.0f32);
        let w: f32 = ha[i..end].iter().sum();
        let w_du: i32 = hg[i..end].iter().map(|g| widened(g.id, g.advance)).sum();
        for k in i..end {
            let own = own_du(&hg[k]);
            let across = match gdi.and_then(|m| m.whole(0.0).map(|_| m)) {
                Some(m) => m.whole(x + ho[k].0 - w / 2.0 + fonts.advance(font, s, hg[k].id, own, scale) / 2.0).unwrap_or(0.0),
                None => (x_du + (ho[k].0 / scale).round() as i32 - w_du / 2 + widened(hg[k].id, own) / 2) as f32 * scale,
            };
            out[k] = (-along, across);
            x += ha[k];
            x_du += widened(hg[k].id, hg[k].advance);
            along += vadv[k];
        }
        i = end;
    }
}


/// Build a shaped glyph run (DIP advances/offsets, UTF-16 cluster map) from a logical run.
/// `cut`: the run ends where the line (or its visible part) is cut, not at a format or script change. Such a
/// run keeps the kerning its last cluster has with the next one: DirectWrite shapes the paragraph before it
/// breaks it into lines (r25 30625: an 'A' ending its line keeps its pair adjustment against the 'V' starting
/// the next), so the run is shaped on through that next cluster, whose glyphs are then dropped.
/// Shape `[s,e)` of the shaping span `ctx`. With `extend` (the run ends where shaping does not stop: a line wrap, a
/// decoration, a style resolving to the same face, a tab), shape on through the next character and the rest of its
/// cluster when the span continues there at the same em and script, keeping only `[s,e)`'s glyphs, so its last
/// glyph kerns against what follows.
#[allow(clippy::too_many_arguments)]
fn shape_cut(fonts: &Fonts, font: &Font, text: &str, s: usize, e: usize, ctx: (usize, usize), rtl: bool, feats: &[rustybuzz::Feature], up: bool, extend: bool) -> Vec<Glyph> {
    let script = |b: usize| text[b..].chars().next().map(|c| c.script()).filter(|&c| c != Script::Common && c != Script::Inherited);
    let own = text[s..e].chars().rev().map(|c| c.script()).find(|&c| c != Script::Common && c != Script::Inherited);
    let next = (extend && e < ctx.1 && fonts.em_at(e) == fonts.em_at(s) && (own.is_none() || script(e).is_none() || script(e) == own)).then(|| {
        let mut z = e + text[e..].chars().next().map_or(0, char::len_utf8);
        while z < ctx.1 && continues_cluster(text, z) {
            z += text[z..].chars().next().map_or(0, char::len_utf8);
        }
        z
    });
    match next {
        Some(z) => shape_run(font, text, s, z, ctx, rtl, feats, up).into_iter().filter(|g| (g.cluster as usize) < e - s).collect(),
        None => shape_run(font, text, s, e, ctx, rtl, feats, up),
    }
}

fn build_run(fonts: &Fonts, fmt: &TextFormat, text: &str, s: usize, e: usize, level: u8, levels: &[Level], feats: &[Feature], cut: bool, line: (usize, usize)) -> ShapedRun {
    let font = fonts.at(s);
    // Per-range SetFontSize: this run's em is the span's override (if any), else the base format em.
    // One itemized run never straddles a format boundary (they are break points), so `em_at(s)` is the
    // em of the whole run.
    let em = fonts.em_at(s).unwrap_or(fmt.em);
    let scale = em / font.upm as f32;
    let rtl = level & 1 == 1;
    let up = fonts.upright_at(text, s);
    let para_ctx = fonts.shaping_span(font, text, s, e, levels);
    // Under a typography range DirectWrite shapes each line's piece of a run alone, no context across the line
    // break (VM: 'مرحبا' wrapped a letter a line joins its letters across the breaks, but with any feature set,
    // even kern on, shows an isolated meem; r25 30477), yet aligns the line on the width it had shaped as part
    // of the paragraph (r25 30009: 'AV' broken under an explicit kern range draws unkerned, centred kerned).
    let (s16, e16) = (u16_pos(text, s), u16_pos(text, e));
    let typography = feats.iter().any(|f| f.start <= s16 && e16 <= f.end);
    let ctx = if typography { (para_ctx.0.max(line.0), para_ctx.1.min(line.1)) } else { para_ctx };
    let shape = |ctx: (usize, usize)| shape_cut(fonts, font, text, s, e, ctx, rtl, &active_features(feats, text, s, e), up, cut);
    let glyphs = shape(ctx);
    let width_of = |gs: &[Glyph]| -> f32 { gs.iter().map(|g| fonts.advance_dir(font, s, g.id, g.advance, scale, up)).sum() };
    let align_delta = if ctx != para_ctx { width_of(&shape(para_ctx)) - width_of(&glyphs) } else { 0.0 };
    let base_u16 = u16_pos(text, s);
    let len_u16 = u16_pos(text, e) - base_u16;

    // UTF-16 cluster map: text-unit → index of the first glyph of its cluster. HarfBuzz
    // clusters are byte offsets relative to the run slice, so add the run start `s`.
    let mut cmap = vec![0u16; len_u16 as usize];
    let mut seen: Option<usize> = None;
    for (gi, g) in glyphs.iter().enumerate() {
        let abs = s + g.cluster as usize;
        let pos = (u16_pos(text, abs) - base_u16) as usize;
        if seen != Some(abs) {
            for c in cmap.iter_mut().skip(pos) {
                *c = gi as u16;
            }
            seen = Some(abs);
        }
    }

    let advances: Vec<f32> = glyphs.iter().map(|g| fonts.advance_dir(font, s, g.id, g.advance, scale, up)).collect();
    // ponytail: left-to-right horizontal runs only; a right-to-left or upright run keeps the design offset.
    let gdi = if fonts.design_advances || up || rtl { None } else { fonts.gdi_at(font, s) };
    let gdi_rtl = if fonts.design_advances || up || !rtl { None } else { fonts.gdi_at(font, s) };
    let mut offsets = glyph_offsets(fonts, font, s, &glyphs, &advances, scale, rtl, up, gdi, gdi_rtl);
    // An upright cluster's glyphs sit where the run's horizontal shaping puts them, the cluster centred on the column
    // (VM r25 30046, 30497, 30743: Segoe UI Emoji's family upright): across the column at the glyph's horizontal
    // position less half the cluster's horizontal advance plus half its own advance, whole pixels under a GDI mode;
    // along it, back at the cluster's start. A run whose horizontal shaping picks other glyphs keeps its centred glyphs.
    // The rule holds at either bidi level (VM r25 30497: an odd-level upright run of a bottom-to-top column).
    // ponytail: the oblique simulation's share of the along offset (the sideways shear of the across one, VM r25 30046)
    // is left to the renderer's shear.
    if up {
        let hg = shape_cut(fonts, font, text, s, e, ctx, false, &active_features(feats, text, s, e), false, cut);
        if hg.len() == glyphs.len() && hg.iter().zip(&glyphs).all(|(h, g)| h.id == g.id && h.cluster == g.cluster) {
            let gdi_h = if fonts.design_advances { None } else { fonts.gdi_at(font, s) };
            let ha: Vec<f32> = hg.iter().map(|g| fonts.advance(font, s, g.id, g.advance, scale)).collect();
            let ho = glyph_offsets(fonts, font, s, &hg, &ha, scale, false, false, gdi_h, None);
            upright_offsets(fonts, font, s, &hg, &ha, &ho, &advances, scale, gdi_h, &mut offsets);
        }
    }
    let width: f32 = advances.iter().sum();
    let (asc, desc) = fonts.vmetrics(font, s, em, up);
    // An upright run's offset only places its underline, which takes the unrounded design value (VM r25 30488).
    let vhalf = match (fonts.vertical, up) {
        (false, _) => 0.0,
        (true, true) => Fonts::vhalf_design(font, em),
        (true, false) => fonts.vhalf(font, s, em),
    };
    let tabs: Vec<usize> = glyphs.iter().enumerate().filter(|(_, g)| text[s + g.cluster as usize..].starts_with('\t')).map(|(i, _)| i).collect();
    ShapedRun {
        level,
        width,
        align_delta,
        tabs,
        offsets,
        glyphs,
        scale,
        em,
        advances,
        text_pos: base_u16,
        text: text[s..e].to_string(),
        cmap,
        font_asc: asc,
        font_height: asc + desc,
        vhalf,
    }
}

struct ShapedRun {
    level: u8,
    width: f32,
    /// The paragraph-shaped width less `width` for a run a line break made DirectWrite reshape (see `build_run`):
    /// alignment measures the line with it.
    align_delta: f32,
    /// The glyphs standing for a tab, whose advance runs to the next tab stop where the run is laid.
    tabs: Vec<usize>,
    scale: f32,
    /// This run's em in DIPs (base format em, or a per-range `SetFontSize`), for the render em_size.
    em: f32,
    glyphs: Vec<Glyph>,
    /// DIP advance per glyph — GDI whole-pixel under a GDI measuring mode, else natural `design·scale`.
    advances: Vec<f32>,
    /// DIP offset per glyph (see [`super::gdi::GdiMeasure::offset`]).
    offsets: Vec<(f32, f32)>,
    text_pos: u32,
    text: String,
    cmap: Vec<u16>,
    font_asc: f32,
    font_height: f32,
    vhalf: f32,
}

impl ShapedRun {
    /// Emit the DWRITE glyph run at line-x `pen` (left edge), baseline `y`. Returns the run
    /// and the pen advanced past it.
    fn emit(&self, pen: f32, y: f32) -> GlyphRun {
        let rtl = self.level & 1 == 1;
        GlyphRun {
            x: if rtl { pen + self.width } else { pen },
            left: pen,
            y,
            top: y - self.font_asc,
            level: self.level,
            glyphs: self.glyphs.iter().map(|g| g.id).collect(),
            advances: self.advances.clone(),
            offsets: self.offsets.clone(),
            clusters: self.cmap.clone(),
            text_pos: self.text_pos,
            text: self.text.clone(),
            font_asc: self.font_asc,
            font_height: self.font_height,
            em: self.em,
            orient: 0,
            vhalf: self.vhalf,
        }
    }
}

/// One cluster of the content, for wrapping and cluster metrics (logical order).
struct Cl {
    bs: usize,
    be: usize,
    width: f32,
    level: u8,
}

/// Shape `[s,e)` and return its clusters in logical order (grouping glyphs that share a
/// HarfBuzz cluster), with DIP widths. Byte offsets are absolute into `text`.
fn content_clusters(fonts: &Fonts, fmt: &TextFormat, text: &str, s: usize, e: usize, levels: &[Level], feats: &[Feature], breaks: &[usize], hard: &[usize], objs: &[ObjSpan]) -> Vec<Cl> {
    let mut out = Vec::new();
    for &(rs, re, l) in itemize(text, s, e, levels, breaks).iter() {
        // An inline object is one cluster of the object's width (its glyphs are replaced).
        if let Some(o) = objs.iter().find(|o| o.s <= rs && re <= o.e) {
            if rs == o.s {
                out.push(Cl { bs: o.s, be: o.e, width: o.obj.width, level: l });
            }
            continue;
        }
        let font = fonts.at(rs);
        let scale = fonts.em_at(rs).unwrap_or(fmt.em) / font.upm as f32;
        let up = fonts.upright_at(text, rs);
        let glyphs = shape_cut(fonts, font, text, rs, re, fonts.shaping_span(font, text, rs, re, levels), l & 1 == 1, &active_features(feats, text, rs, re), up, !hard.contains(&re));
        let mut i = 0;
        while i < glyphs.len() {
            let c = glyphs[i].cluster;
            let (mut j, mut width) = (i, 0.0f32);
            while j < glyphs.len() && glyphs[j].cluster == c {
                width += fonts.advance_dir(font, rs, glyphs[j].id, glyphs[j].advance, scale, up);
                j += 1;
            }
            let be = if j < glyphs.len() { rs + glyphs[j].cluster as usize } else { re };
            out.push(Cl { bs: rs + c as usize, be, width, level: l });
            i = j;
        }
    }
    out
}

/// Byte offset of `[a,b)`'s trailing whitespace (where it starts); `b` if none.
fn trailing_ws_start(text: &str, a: usize, b: usize) -> usize {
    let mut t = b;
    while t > a {
        let prev = text[..t].char_indices().next_back().unwrap().0;
        if is_trailing_space(text[prev..t].chars().next().unwrap()) {
            t = prev;
        } else {
            break;
        }
    }
    t
}

/// Strip trailing line-break chars from a paragraph `[ps,pe)`, returning the content end.
fn content_end_of(text: &str, ps: usize, pe: usize) -> usize {
    let mut e = pe;
    while e > ps && is_newline(text[..e].chars().next_back().unwrap()) {
        e = text[..e].char_indices().next_back().unwrap().0;
    }
    e
}

/// Number of maximal same-level (direction) runs in `[a,b)` — DirectWrite's per-line reordering
/// count, i.e. the number of HitTestTextRange boxes a whole-line range yields. 0 if empty.
fn level_runs(levels: &[Level], a: usize, b: usize) -> u32 {
    let mut count = 0u32;
    let mut prev: Option<u8> = None;
    for i in a..b {
        let l = levels[i].number();
        if prev != Some(l) {
            count += 1;
            prev = Some(l);
        }
    }
    count
}

/// Greedy word-wrap of content clusters into line byte-ranges `[a,b)` (UAX#14 opportunities,
/// trailing whitespace excluded from the fit width). Single run or multi-run; one line when
/// it all fits. A word (segment) wider than the line is emergency-broken at cluster boundaries,
/// as `DWRITE_WORD_WRAPPING_WRAP` does (it breaks an over-long word, not overflows it).
/// `emergency`: a word wider than the line may be split at cluster boundaries (DWrite's emergency
/// break); `any_cluster`: every cluster boundary is a break opportunity (CHARACTER wrapping).
/// `over(w)`: a line `w` DIPs wide overflows the box.
fn wrap_lines(text: &str, cls: &[Cl], brk: &std::collections::HashMap<usize, BreakOpportunity>, over: &dyn Fn(f32) -> bool, emergency: bool, any_cluster: bool, tab_step: f32) -> Vec<(usize, usize)> {
    // Widths from a pen position: a tab runs to the next tab stop.
    let width_at = |ci: usize, pen: f32| if &text[cls[ci].bs..cls[ci].be] == "\t" { next_tab_stop(pen, tab_step) - pen } else { cls[ci].width };
    // Summed from zero as the cluster widths are (r11 16155: 'fox' after a 7.2-DIP 'The quick brown ' fits an
    // 8.8-DIP box, where adding the widths onto the pen and taking the pen off again overflowed it in f32).
    let full_at = |s: usize, e: usize, pen: f32| (s..e).fold(0.0f32, |w, ci| w + width_at(ci, pen + w));
    // For fitting, a line's trailing whitespace hangs, a tab included (r25 30050: 'b\t' stays one line past
    // the width and breaks after its tab), though a tab is content in the line's metrics.
    let visible_at = |s: usize, e: usize, pen: f32| {
        let mut last = s;
        for ci in s..e {
            if !text[cls[ci].bs..cls[ci].be].chars().next().unwrap().is_whitespace() {
                last = ci + 1;
            }
        }
        full_at(s, last, pen)
    };
    let visible = |s: usize, e: usize| visible_at(s, e, 0.0);
    // Segment boundaries: a cluster whose end is a break opportunity (or the last cluster). A segment
    // whose own width exceeds the line is greedily split at cluster boundaries (emergency break):
    // fit clusters while the running width stays ≤ max_width, break when the next would exceed.
    let mut segs: Vec<(usize, usize)> = Vec::new();
    // The pieces of a word split by an emergency break each start a line of their own (below).
    let mut split_piece: Vec<bool> = Vec::new();
    let mut seg_start = 0;
    let mut push_seg = |a: usize, b: usize, segs: &mut Vec<(usize, usize)>, split_piece: &mut Vec<bool>| {
        if !emergency || !over(visible(a, b)) {
            segs.push((a, b));
            split_piece.push(false);
            return;
        }
        // Break before a cluster that would overflow the piece, a zero-width one too (VM r25 30937: a
        // fallback-font combining mark after an overflowing 'e' starts the next line); the trailing
        // whitespace stays with the last piece.
        let (mut sub, mut cum) = (a, 0.0f32);
        for ci in a..b {
            let w = cls[ci].width;
            let space = text[cls[ci].bs..cls[ci].be].chars().next().unwrap().is_whitespace();
            if ci > sub && !space && over(cum + w) {
                segs.push((sub, ci));
                split_piece.push(true);
                (sub, cum) = (ci, 0.0);
            }
            cum += w;
        }
        segs.push((sub, b));
        split_piece.push(true);
    };
    for ci in 0..cls.len() {
        if any_cluster || brk.contains_key(&cls[ci].be) || ci + 1 == cls.len() {
            push_seg(seg_start, ci + 1, &mut segs, &mut split_piece);
            seg_start = ci + 1;
        }
    }

    let mut lines = Vec::new();
    let mut line_seg = 0;
    let mut cur_full = 0.0f32;
    for k in 0..segs.len() {
        let (ss, se) = segs[k];
        // A whitespace-only segment (visible width 0) never starts a new line: DWrite keeps it as the
        // current line's trailing whitespace, not a line of its own. Without this, a degenerate box
        // narrower than one glyph (so `cur_full` already exceeds `max_width`) puts an inter-word space
        // on its own line and pushes the next word down a line (browser r5 5082).
        // A word too wide for any line moves to a line of its own before it breaks inside, each piece on its own
        // line (VM r25 30617: ' lead' leaves its leading space alone on the first line, then breaks 'lead'
        // letter by letter).
        // A zero-width cluster that is not blank does move down: under character wrapping a combining mark of its own
        // cluster starts the next line after an overflowing base (VM r25 30483: 'a' and its fallback-face diaeresis,
        // 'n' and a tilde split off by a format range, each mark on a line of its own).
        let blank = (ss..se).all(|ci| text[cls[ci].bs..cls[ci].be].chars().all(|c| c.is_whitespace() || c == '\u{200B}'));
        if k > line_seg && (split_piece[k] || (!blank && over(cur_full + visible_at(ss, se, cur_full)))) {
            lines.push((cls[segs[line_seg].0].bs, cls[segs[k - 1].1 - 1].be));
            line_seg = k;
            cur_full = 0.0;
        }
        cur_full += full_at(ss, se, cur_full);
    }
    lines.push((cls[segs[line_seg].0].bs, cls[segs.last().unwrap().1 - 1].be));
    lines
}

/// Itemize `[a,b)`, shape each run, reorder visually, emit at pen `pen0` on baseline `y`.
/// Returns the pen after the range.
fn emit_range(fonts: &Fonts, fmt: &TextFormat, text: &str, a: usize, b: usize, line: (usize, usize), align_delta: &mut f32, levels: &[Level], feats: &[Feature], breaks: &[usize], hard: &[usize], objs: &[ObjSpan], pen0: f32, y: f32, out: &mut Vec<Run>, obj_recs: &mut Vec<(u32, u32, InlinePlacement)>) -> f32 {
    if b <= a {
        return pen0;
    }
    // Items in logical order: shaped text runs and inline objects (whose glyphs are replaced).
    enum Emit {
        Text(ShapedRun),
        Obj(ObjSpan),
    }
    let mut items: Vec<(u8, Emit)> = Vec::new();
    for &(s, e, l) in itemize(text, a, b, levels, breaks).iter() {
        match objs.iter().find(|o| o.s <= s && e <= o.e) {
            Some(o) if s == o.s => items.push((l, Emit::Obj(*o))),
            Some(_) => {} // object interior already emitted at its start
            None => items.push((l, Emit::Text(build_run(fonts, fmt, text, s, e, l, levels, feats, !hard.contains(&e), line)))),
        }
    }
    let order = BidiInfo::reorder_visual(&items.iter().map(|(l, _)| Level::new(*l).unwrap()).collect::<Vec<_>>());
    // A right-to-left paragraph counts its tab stops from its right edge, the line start (r25 30038): resolve
    // the tabs walking from the right, the glyphs of a left-to-right run last to first.
    let mut rtl_tabs: Vec<Vec<f32>> = vec![Vec::new(); items.len()];
    if fmt.base_rtl {
        let mut q = 0.0f32;
        for &vi in order.iter().rev() {
            match &items[vi].1 {
                Emit::Text(r) => {
                    let mut a = r.advances.clone();
                    let from_right: Vec<usize> = if r.level & 1 == 1 { (0..a.len()).collect() } else { (0..a.len()).rev().collect() };
                    for i in from_right {
                        if r.tabs.contains(&i) {
                            a[i] = next_tab_stop(q, 4.0 * fmt.em) - q;
                        }
                        q += a[i];
                    }
                    rtl_tabs[vi] = a;
                }
                Emit::Obj(o) => q += o.obj.width,
            }
        }
    }
    *align_delta += items.iter().map(|(_, it)| if let Emit::Text(r) = it { r.align_delta } else { 0.0 }).sum::<f32>();
    let mut pen = pen0;
    for &vi in &order {
        let (l, item) = &items[vi];
        match item {
            Emit::Text(r) if !r.tabs.is_empty() && fmt.base_rtl => {
                let w: f32 = rtl_tabs[vi].iter().sum();
                let mut g = r.emit(pen, y);
                if r.level & 1 == 1 {
                    g.x = pen + w;
                }
                g.advances = rtl_tabs[vi].clone();
                pen += w;
                out.push(Run::Glyph(g));
            }
            Emit::Text(r) if !r.tabs.is_empty() && r.level & 1 == 0 => {
                // A tab advances to the next tab stop from the line start, every 4 em (DirectWrite's default
                // incremental tab stop; VM r25 30843: 'tab\t' at em 3.92 ends at 15.69).
                let mut g = r.emit(pen, y);
                let mut p = pen;
                for (i, a) in g.advances.iter_mut().enumerate() {
                    if r.tabs.contains(&i) {
                        *a = next_tab_stop(p, 4.0 * fmt.em) - p;
                    }
                    p += *a;
                }
                pen = p;
                out.push(Run::Glyph(g));
            }
            Emit::Text(r) => {
                out.push(Run::Glyph(r.emit(pen, y)));
                pen += r.width;
            }
            Emit::Obj(o) => {
                let pl = InlinePlacement {
                    x: pen,
                    y: y - o.obj.baseline,
                    width: o.obj.width,
                    height: o.obj.height,
                    baseline: o.obj.baseline,
                    rtl: l & 1 == 1,
                    sideways: o.obj.sideways,
                    sign: None,
                };
                out.push(Run::Inline(pl));
                obj_recs.push((o.obj.start, o.obj.end, pl));
                pen += o.obj.width;
            }
        }
    }
    pen
}

/// Effective horizontal alignment `TextLayout::AlignHorizontally` resolves `(textAlignment,
/// readingDirection)` to: 0 = left, 1 = center, 2 = right (justified resolves to leading, i.e.
/// left for LTR reading, since the spread is applied to advances elsewhere).
fn effective_align(align: u8, base_rtl: bool) -> u8 {
    let rtl = base_rtl as u8; // reading-direction bit (layout flags & 1)
    match align {
        2 => 1,                       // center
        1 => 2 * ((!rtl) & 1),        // trailing: flips the reading-direction bit
        _ => 2 * (rtl & 1),           // leading (0) or justified (3) base offset
    }
}

/// Per-line left shift for effective alignment `align` and a line whose width (excluding trailing
/// whitespace) is `width`, matching `AlignHorizontally`'s float order (no optical/indent terms).
fn align_offset(align: u8, layout_width: f32, width: f32) -> f32 {
    match align {
        1 => (layout_width - width) * 0.5,
        2 => layout_width - width,
        _ => 0.0,
    }
}

/// Snap a DIP coordinate onto the device-pixel grid the way a GDI-compatible layout does: a
/// +0.25-biased floor, `floor(v·ppd + 0.25) / ppd`. This is the exact op horizontal and vertical
/// alignment apply to the alignment offset under a
/// GDI measuring mode (each coordinate: `t = v·ppd + 0.25`; floor it when `|t| < 2^23`; then `v = t/ppd`). It is NOT a
/// +0.5 round and NOT a plain floor — the +0.25 bias rounds up only when the device-pixel fraction is
/// ≥ 0.75, which is what matches native where round and floor disagree (fuzz 9303 trailing @dpi120
/// −15.844→−16, 15281 trailing @dpi120 110.924→110, 15003 centre @dpi120 −41.554→−42, 9487 centre
/// @dpi192 −26.897→−27). The 2^23 guard is a no-op for `f32::floor`, which already leaves integers be.
fn snap_gdi(v: f32, ppd: f32) -> f32 {
    (v * ppd + 0.25).floor() / ppd
}

/// Per-line alignment offset in DIPs. Natural measuring keeps the sub-pixel DIP offset; GDI measuring
/// snaps it onto the device-pixel grid ([`snap_gdi`]).
fn aligned_offset(align: u8, max_width: f32, vis: f32, gdi: bool, ppd: f32) -> f32 {
    let off = align_offset(align, max_width, vis);
    if gdi {
        snap_gdi(off, ppd)
    } else {
        off
    }
}

/// Width of the ellipsis sign (`CreateEllipsisTrimmingSign`) — the advance of the font's horizontal
/// ellipsis glyph. DirectWrite reports height/baseline 0 for it, so only the width matters.
pub fn ellipsis_width(font: &Font, fmt: &TextFormat) -> f32 {
    ellipsis_width_bold(font, fmt, 0)
}

/// [`ellipsis_width`] for a face whose bold simulation widens the sign's glyph by `bold` design units, added to
/// the design advance before it scales, as every bold-simulated advance is.
pub fn ellipsis_width_bold(font: &Font, fmt: &TextFormat, bold: i32) -> f32 {
    let scale = fmt.em / font.upm as f32;
    shaping::shape(font, "\u{2026}", false, None, None).iter().enumerate().map(|(i, g)| (g.advance + if i == 0 { bold } else { 0 }) as f32 * scale).sum()
}

/// A vertical layout's trimming sign stands upright: the vertical ellipsis U+22EE, which the layout's faces lack,
/// so it comes from the face font fallback gives it (Segoe UI Symbol). Its extent along the column per em is that
/// glyph's vertical advance (for a face without vmtx its typo ascent plus descent), widened by round(upem/50)
/// when the face is bold-simulated (`sims`), as a bold-simulated advance is. VM r25: all 22 vertical signs, in
/// eleven format fonts, span Segoe UI Symbol's 1922 of 2048 units, or 1963 at weight 600 and over.
/// Returns the extent and its bold part, both per em.
pub fn upright_sign_per_em(bytes: Vec<u8>, index: u32, sims: u32) -> Option<(f32, f32)> {
    let font = Font::from_bytes(bytes, index)?;
    let text = "\u{22EE}";
    let adv = shape_run(&font, text, 0, text.len(), (0, text.len()), false, &[], true).first()?.advance;
    let bold = if sims & 1 != 0 { (font.upm as f32 / 50.0).round() } else { 0.0 };
    Some(((adv as f32 + bold) / font.upm as f32, bold / font.upm as f32))
}

/// An ellipsis sign placed at `x` on baseline `y`.
fn ellipsis_place(x: f32, y: f32, w: f32, sign: Option<(u16, f32)>) -> InlinePlacement {
    InlinePlacement { x, y, width: w, height: 0.0, baseline: 0.0, rtl: false, sideways: false, sign }
}

/// Where a trimmed line keeps content: `[keep_start,keep_end)` preserved (before trailing-whitespace
/// stripping), `ellipsis_front` putting the sign before the content (path trimming) vs after.
struct TrimPlan {
    keep_start: usize,
    keep_end: usize,
    ellipsis_front: bool,
    /// A path trim's head: `[line start, head_end)` kept before the sign (before trailing-whitespace stripping), the
    /// line start when it keeps none.
    head_end: usize,
}

/// Decide what a trimmed line keeps. Character/word granularity keep the longest fitting leading run
/// (word backs up to the last break opportunity); a delimiter preserves the tail from the Nth
/// delimiter counted from the end (path ellipsis), the sign going in front. A delimiter count of 0, or a
/// line with fewer delimiters than the count, trims as if there were no delimiter (VM r25 30685: '0123456789'
/// with a space delimiter keeps only its '0').
fn plan_trim(text: &str, line_cls: &[&Cl], a: usize, line_end: usize, max_width: f32, tr: &Trimming, ellipsis_w: f32, brk: &std::collections::HashMap<usize, BreakOpportunity>) -> TrimPlan {
    if tr.delimiter != 0 && tr.delimiter_count > 0 {
        let dch = char::from_u32(tr.delimiter).unwrap_or('\0');
        let hits: Vec<usize> = text[a..line_end].match_indices(dch).map(|(i, _)| a + i).collect();
        let n = tr.delimiter_count as usize;
        if hits.len() >= n {
            // A preserved tail wider than the box keeps only its end that fits (r25 30500: ' two' in a 43-DIP box
            // keeps its 'o').
            let budget = max_width - if tr.ellipsis { ellipsis_w } else { 0.0 };
            let tail: Vec<&&Cl> = line_cls.iter().filter(|c| c.bs >= hits[hits.len() - n]).collect();
            let mut width: f32 = tail.iter().map(|c| c.width).sum();
            let mut keep_start = hits[hits.len() - n];
            for c in &tail {
                if width <= budget {
                    break;
                }
                width -= c.width;
                keep_start = c.be;
            }
            // The line's start fills what the tail and the sign leave, cut as an end trim at the granularity (VM: Segoe
            // UI 'The quick brown fox jumps over the lazy dog again and again' in a 90-DIP box keeps 'The quic', the sign
            // and ' again' by character, 'The', the sign and ' again' by word, 'The', the sign and ' and again' with two
            // delimiters).
            let head_cls: Vec<&Cl> = line_cls.iter().filter(|c| c.be <= keep_start).copied().collect();
            // A head with no cluster that fits keeps none (the end trim's one-cluster floor does not apply: a 90-DIP box
            // over Arial 16 'C:/Users/name/documents/report.txt' path-trimmed at '/' is the sign and '/report.txt').
            // A tail cut down to fit leaves no head (VM r25 30500: ' two' in a 43-DIP box keeps only its 'o').
            let head_budget = budget - width;
            let whole_tail = keep_start == hits[hits.len() - n];
            let head_end = if whole_tail && head_budget > 0.0 && !head_cls.is_empty() { trim_walk(text, &head_cls, a, head_budget, tr.granularity, brk) } else { a };
            let head_w: f32 = head_cls.iter().filter(|c| c.be <= head_end).map(|c| c.width).sum();
            let head_end = if head_w > head_budget { a } else { head_end };
            return TrimPlan { keep_start, keep_end: line_end, ellipsis_front: true, head_end };
        }
    }
    // dwrite `TrimRange`: from the line's width including trailing whitespace (the clusters summed in
    // order, f32), walk back from the end subtracting each cluster until the rest fits the budget at a
    // usable cut — any cluster for character granularity; for word granularity a line-break
    // opportunity after the cluster, or a whitespace cluster after the cut — and the cluster before the
    // cut is not whitespace (trailing spaces go with the cut). Once a usable cut was seen, later fits
    // need not be usable. A cut never lands at the very end. When the remaining width is exactly 0 the
    // walk keeps one more cluster.
    let budget = (max_width - if tr.ellipsis { ellipsis_w } else { 0.0 }).max(0.0);
    let keep_end = trim_walk(text, line_cls, a, budget, tr.granularity, brk);
    TrimPlan { keep_start: a, keep_end, ellipsis_front: false, head_end: a }
}

/// The end of what an end trim keeps of `line_cls` (from `a`) within `budget`, at `granularity` (see [`plan_trim`]).
fn trim_walk(text: &str, line_cls: &[&Cl], a: usize, budget: f32, granularity: u8, brk: &std::collections::HashMap<usize, BreakOpportunity>) -> usize {
    let n = line_cls.len();
    let ws_first = |c: &Cl| text[c.bs..c.be].chars().next().is_some_and(char::is_whitespace);
    let ws_last = |c: &Cl| text[c.bs..c.be].chars().next_back().is_some_and(char::is_whitespace);
    let mut width = 0.0f32;
    for c in line_cls {
        width += c.width;
    }
    let mut k = n;
    let mut found = false;
    let mut before_ws = false;
    while k > 0 {
        if budget >= width {
            let ok = granularity != 2 || brk.contains_key(&line_cls[k - 1].be) || (k < n && ws_first(line_cls[k]));
            if ok {
                found = true;
                before_ws = ws_last(line_cls[k - 1]);
            }
            if (ok || found) && k < n && !before_ws {
                break;
            }
        }
        k -= 1;
        width -= line_cls[k].width;
    }
    if width == 0.0 && k < n {
        k += 1;
    }
    if k > 0 { line_cls[k - 1].be } else { a }
}

/// Lay out with default options (no typography features/inline objects/trimming).
pub fn layout(font: &Font, fmt: &TextFormat, text: &str, max_width: f32, max_height: f32) -> Layout {
    layout_ex(font, fmt, text, max_width, max_height, &LayoutOpts::default())
}

pub fn layout_ex(font: &Font, fmt: &TextFormat, text: &str, max_width: f32, max_height: f32, opts: &LayoutOpts) -> Layout {
    // A flow across the page (left-to-right 2, right-to-left 3) means a vertical reading direction: the
    // only pairing CreateTextLayout accepts.
    if fmt.flow >= 2 {
        return layout_vertical(font, fmt, text, max_width, max_height, opts);
    }
    layout_core(font, fmt, text, max_width, max_height, opts, &[])
}

/// [`layout_ex`] for a horizontal reading direction; `extra_breaks` are byte offsets where runs must split.
fn layout_core(font: &Font, fmt: &TextFormat, text: &str, max_width: f32, max_height: f32, opts: &LayoutOpts, extra_breaks: &[usize]) -> Layout {
    // Fallback resolver: per-byte font (base family, or a resolved fallback span). A GDI measuring
    // mode (1 classic / 2 natural) builds the base face's GDI measurer (whole-pixel advances + VDMX
    // line metrics); natural (0) leaves it None and is byte-identical to before GDI measuring existed.
    let base_ppd = if opts.pixels_per_dip > 0.0 { opts.pixels_per_dip } else { 1.0 };
    // ponytail: the y-axis scale stands in for the full rasterizer transform, so an axis-aligned
    // non-uniform or quarter-turn DrawText transform measures at the y scale only.
    // DrawText's world transform (native `GetTextTransform`) reaches its GDI-compatible layout
    // (probed with CreateGdiCompatibleTextLayout and native's layout dump, Arial/Times/Segoe UI @13):
    // * no zero component (rotated/sheared): DESIGN advances (native's advance query skips the hinter);
    // * y-scale `s = |(m21, m22)| != 1`: the layout runs on the device grid `ppd·s` (axis-aligned advances
    //   hinted at that ppem, ×2 'i' 6 px → 3 DIP) and its line box is the line-spacing refresh over
    //   `GetGdiCompatibleMetrics(em, ppd·s, transform)` — see `Fonts::vmetrics`.
    let (ppd, design_advances, metric_scale) = match opts.transform {
        Some(m) if matches!(opts.measuring, 1 | 2) => {
            let s = (m[2] * m[2] + m[3] * m[3]).sqrt();
            let scaled = s.is_finite() && s > 0.0 && (s - 1.0).abs() > 1e-6;
            (if scaled { base_ppd * s } else { base_ppd }, !m.iter().any(|&v| super::gdi::fixed_zero(v)), if scaled { s } else { 1.0 })
        }
        _ => (base_ppd, false, 1.0),
    };
    let gdi = matches!(opts.measuring, 1 | 2)
        .then(|| super::gdi::GdiMeasure::new(font, fmt.em, base_ppd, opts.measuring == 1, opts.transform))
        .flatten();
    // Under a GDI measuring mode every span (a per-range format or a fallback face) is measured like the
    // base face: whole-pixel advances and line metrics at its own em (VM, Yu Gothic UI fallback under
    // GDI classic: advances 38/32 and a 51/41 line box at em 37.75).
    let span_gdi = opts.fonts.iter()
        .map(|sp| matches!(opts.measuring, 1 | 2).then(|| super::gdi::GdiMeasure::new(sp.font, sp.em.unwrap_or(fmt.em), base_ppd, opts.measuring == 1, opts.transform)).flatten())
        .collect();
    let fonts = Fonts { base: font, spans: opts.fonts, gdi, span_gdi, bold_delta_du: opts.base_bold_delta_du, design_advances, metric_scale, vertical: fmt.flow >= 2 };
    // Base-font line box (DirectWrite default spacing, gap above the baseline): used for the empty
    // layout, the newline run and the caret default. Per-line metrics below take the tallest run.
    let (base_asc, base_desc) = fonts.vmetrics(font, usize::MAX, fmt.em, false);
    let baseline = base_asc;
    let line_height = base_asc + base_desc;

    // Feature-range, inline-object and font-fallback boundaries force run splits (byte offsets).
    let feats = opts.features;
    let obj_spans: Vec<ObjSpan> = opts.inline.iter().map(|o| ObjSpan { s: u16_to_byte(text, o.start), e: u16_to_byte(text, o.end), obj: *o }).collect();
    let mut breaks: Vec<usize> = feats.iter().flat_map(|f| [u16_to_byte(text, f.start), u16_to_byte(text, f.end)]).collect();
    breaks.extend(obj_spans.iter().flat_map(|o| [o.s, o.e]));
    // The splits shaping does not cross: a run ending at any other split (a decoration, a style resolving to the
    // same face, a tab) still kerns against what follows, as DirectWrite shapes its span as one (r25 30475: 'V'
    // kerns against the struck-through 'A' after it).
    let mut hard = breaks.clone();
    hard.extend_from_slice(extra_breaks);
    // A format or decoration boundary splits a cluster only where the face or the size changes; otherwise the
    // cluster stays in the run it starts (VM r25 30422: a strikethrough range ending between 👍 and its skin-tone
    // modifier leaves 👍🏽 one ligature run; 30139: 'ते' stays whole across a family range both sides of which
    // fall back to Nirmala UI). A size change does split one (r25 30036: a size range starting at a virama).
    let same_face = |b: usize| text.get(..b).and_then(|t| t.char_indices().next_back()).is_some_and(|(p, _)| std::ptr::eq(fonts.at(p), fonts.at(b)) && fonts.em_at(p) == fonts.em_at(b));
    // Such a boundary moves to its cluster's end instead, where the run splits (VM r25 30625: a family range
    // starting on the vowel sign of 'दु' leaves the cluster whole and starts its run at 'नि').
    let at_cluster_end = |b: usize| {
        let mut p = b;
        if continues_cluster(text, p) && same_face(p) {
            while p < text.len() && continues_cluster(text, p) {
                p += text[p..].chars().next().map_or(1, char::len_utf8);
            }
        }
        p
    };
    breaks.extend(opts.fonts.iter().flat_map(|s| [s.start, s.end]).map(at_cluster_end));
    breaks.extend(opts.underline.iter().chain(opts.strikethrough).flat_map(|&(s, e)| [s, e]).map(at_cluster_end));
    breaks.extend_from_slice(extra_breaks);
    // A tab is a run of its own (DirectWrite draws it as a run of no glyphs, with its own underline or
    // strikethrough: r25 30156 strikes 'b' and the tab after it as two segments).
    breaks.extend(text.match_indices('\t').flat_map(|(i, _)| [i, i + 1]));
    breaks.sort_unstable();
    breaks.dedup();

    let base = if fmt.base_rtl { Level::rtl() } else { Level::ltr() };
    let bidi = BidiInfo::new(text, Some(base));
    // UAX #9 L1: a segment or paragraph separator, and the whitespace before it, take the paragraph level (r25
    // 30038: the tab of 'tab\there' in a right-to-left paragraph sits between its two left-to-right words).
    let mut levels = bidi.levels.clone();
    for (i, ch) in text.char_indices() {
        if matches!(bidi.original_classes[i], BidiClass::S | BidiClass::B) {
            let ws = text[..i].char_indices().rev().take_while(|&(p, _)| matches!(bidi.original_classes[p], BidiClass::WS | BidiClass::FSI | BidiClass::LRI | BidiClass::RLI | BidiClass::PDI)).last().map_or(i, |(p, _)| p);
            levels[ws..i + ch.len_utf8()].fill(base);
        }
    }

    // Break opportunities keyed by the byte *after* the break. A Thai run breaks between its words, which DirectWrite
    // finds with its dictionary breaker before the pair table runs.
    let thai_words = super::thai::word_breaks(text);
    let mut brk: std::collections::HashMap<usize, BreakOpportunity> = linebreaks(text, &thai_words).collect();
    // `linebreaks` always reports a Mandatory break at end-of-text; keep it for canWrapAfter.
    brk.entry(text.len()).or_insert(BreakOpportunity::Mandatory);

    let align = effective_align(fmt.align, fmt.base_rtl);
    let no_wrap = fmt.wrap == 1; // DWRITE_WORD_WRAPPING_NO_WRAP
    let trim = opts.trimming.filter(|t| t.granularity != 0);
    let ellipsis_w = match trim.filter(|t| t.ellipsis) {
        None => 0.0,
        Some(t) if fonts.vertical && t.upright_sign > 0.0 => t.upright_sign * fmt.em,
        Some(Trimming { sign: Some((_, w)), .. }) => w * fmt.em,
        // A bold-simulated face widens the sign's ellipsis as it does any inked glyph (VM r25 30062: Lucida Console
        // at weight 900 spans its ellipsis plus round(upem/50) units).
        Some(_) => ellipsis_width_bold(font, fmt, fonts.bold_delta_du),
    };
    let ellipsis_glyph = trim.and_then(|t| t.sign).map(|(g, _)| (g, fmt.em)).or_else(|| shaping::shape(font, "\u{2026}", false, None, None).first().map(|g| (g.id, fmt.em)));

    let mut runs_out: Vec<Run> = Vec::new();
    let mut lines: Vec<LineMetrics> = Vec::new();
    // Justified cluster widths by UTF-16 start position (empty unless a line justifies).
    let mut justified: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
    let mut trimmed_spans: Vec<(u32, u32, Caret, Caret)> = Vec::new();
    let mut obj_recs: Vec<(u32, u32, InlinePlacement)> = Vec::new();
    let mut cum_top = 0.0f32; // running top of the current line (variable line heights)
    // A bottom-to-top flow (and left-to-right, which dwrite `AlignVertically` groups with it) stacks the
    // lines upward: line k's top is `−(lines above) − its height`, and the whole block later moves down
    // by the total height (with the paragraph offset), so the first line ends at the box bottom.
    let upward = (fmt.flow.wrapping_add(1) & 2) != 0;
    // Glyph runs per line, in order (an underline groups runs within one line).
    let mut line_glyph_runs: Vec<usize> = Vec::new();
    let mut max_w = 0.0f32;
    let mut min_off = f32::INFINITY; // metrics.left = min over lines of the alignment offset
    let mut max_right = 0.0f32; // max over lines of (offset + width incl. trailing whitespace)
    let mut bidi_depth = 1u32;

    // Each paragraph (split at mandatory breaks by the bidi algorithm) wraps into one or
    // more visual lines; a trailing newline is kept as a zero-glyph run on the last line. The lines
    // and their boxes are found for every paragraph first, so trimming can look at the next line.
    let mut para_lines = Vec::with_capacity(bidi.paragraphs.len());
    let mut line_box: Vec<(f32, f32, bool)> = Vec::new(); // (height, baseline, ends in a newline)
    for para in &bidi.paragraphs {
        let (ps, pe) = (para.range.start, para.range.end);
        let content_end = content_end_of(text, ps, pe);
        let cls = content_clusters(&fonts, fmt, text, ps, content_end, &levels, feats, &breaks, &hard, &obj_spans);
        let line_ranges = if cls.is_empty() {
            vec![(ps, ps)]
        } else if no_wrap {
            vec![(ps, content_end)]
        } else {
            // GDI measuring fits in whole device pixels: the line's DIP width and the box each scale by ppd and round
            // to the nearest pixel (a tie going up), and the line overflows when its pixels exceed the box's (dwrite's
            // line breaker). Natural compares DIPs. So 'Ag' 44 px wide fits a 43.92 box (fuzz 9716), and a line of
            // whole-pixel advances filling the box exactly fits though its f32 DIP sum lands an ulp past it (VM 120 and
            // 144 dpi, Segoe UI, Tahoma, Arial, Calibri, Verdana, Yu Gothic UI, Malgun Gothic: the last word wrapped
            // early; vertical Malgun Gothic CJK filling 165 px of a 110-DIP column sums to 110.0000076).
            let gdi = matches!(opts.measuring, 1 | 2);
            let px = |v: f32| { let s = v * ppd; if s.abs() < 8388608.0 { round_half_up(s) } else { s } };
            let over = |w: f32| if gdi { px(w) > px(max_width) } else { w > max_width };
            // DWRITE_WORD_WRAPPING: WRAP (0) and EMERGENCY_BREAK (2) split an over-wide word at cluster
            // boundaries — WRAP only when no trimming is set (dwrite `BreakLines`: a trimmed layout trims
            // the overflowing word instead); WHOLE_WORD (3) never splits; CHARACTER (4) breaks anywhere.
            let emergency = match fmt.wrap { 0 => trim.is_none(), 3 => false, _ => true };
            wrap_lines(text, &cls, &brk, &over, emergency, fmt.wrap == 4, 4.0 * fmt.em)
        };
        for (li, &(a, b)) in line_ranges.iter().enumerate() {
            // Line ascent/descent: the max over the run fonts on the line (mixed-font fallback),
            // raised by any inline object on the line. A paragraph's last line counts its newline too (r25 30500:
            // a '\n' that starts a 45-DIP range makes a 3.47-DIP 'line one' 54.78 tall).
            let nl = li + 1 == line_ranges.len() && content_end < pe;
            let (mut asc, mut desc) = fonts.line_vmetrics(text, fmt.em, a, if nl { pe } else { b }, &obj_spans);
            for o in &obj_spans {
                if o.s < b && o.e > a {
                    asc = asc.max(o.obj.baseline);
                    desc = desc.max(o.obj.height - o.obj.baseline);
                }
            }
            let (h, bl) = fmt.spaced(asc + desc, asc);
            line_box.push((h, bl, li + 1 == line_ranges.len() && content_end < pe));
        }
        para_lines.push((cls, line_ranges));
    }
    // A layout that is empty or ends in a hard line break ends with one more, empty line (see below).
    let extra_line = bidi.paragraphs.last().is_none_or(|p| content_end_of(text, p.range.start, p.range.end) < p.range.end);
    // dwrite `TrimLines`: with trimming on, walk the lines summing heights; when the next line would
    // pass the box bottom, every later line is cut (no height, nothing drawn) and this one is trimmed
    // like an over-wide line — unless a path delimiter is set or the line ends in a hard break.
    let n_lines = line_box.len() + extra_line as usize;
    let box_h = |i: usize| line_box.get(i).map_or(fmt.spaced(base_asc + base_desc, base_asc).0, |b| b.0);
    let (mut cut_trim, mut hidden) = (vec![false; n_lines], vec![false; n_lines]);
    if let Some(tr) = trim {
        let (mut keep, mut cum) = (true, 0.0f32);
        for i in 0..n_lines {
            if !keep {
                hidden[i] = true;
                continue;
            }
            cum += box_h(i);
            if i + 1 < n_lines && cum + box_h(i + 1) > max_height {
                keep = false;
                cut_trim[i] = tr.delimiter == 0 && !line_box.get(i).is_some_and(|b| b.2);
            }
        }
    }

    let mut gline = 0usize;
    for (para, (cls, line_ranges)) in bidi.paragraphs.iter().zip(&para_lines) {
        let (ps, pe) = (para.range.start, para.range.end);
        let content_end = content_end_of(text, ps, pe);
        let newline_u16 = u16_pos(text, pe) - u16_pos(text, content_end);
        for (li, &(a, b)) in line_ranges.iter().enumerate() {
            let is_last = li + 1 == line_ranges.len();
            let g = gline;
            gline += 1;
            let b_eff = if is_last { pe } else { b };
            if hidden[g] {
                // A cut line keeps its text length but has no height and draws nothing.
                let tw_byte = trailing_ws_start(text, a, b_eff);
                lines.push(LineMetrics {
                    length: u16_pos(text, b_eff) - u16_pos(text, a),
                    trailing_ws: u16_pos(text, b_eff) - u16_pos(text, tw_byte),
                    newline: if is_last { newline_u16 } else { 0 },
                    height: 0.0,
                    baseline: 0.0,
                    is_trimmed: true,
                });
                line_glyph_runs.push(0);
                continue;
            }
            let (line_h, line_bl, _) = line_box[g];
            let y = if upward { -cum_top - line_h } else { cum_top } + line_bl;

            // DirectWrite emits a wrapped line's trailing whitespace as its own glyph run,
            // positioned past the visible content and excluded from the line's width. Runs are
            // first emitted at line-x 0, then shifted by the alignment offset below.
            let vis_byte = trailing_ws_start(text, a, b);
            let line_start = runs_out.len();
            let obj_start = obj_recs.len();

            // Does this line need trimming? (visible content overflows the layout width).
            // A tab counts at its tab-stop width, as the line breaker measured it (r25 30811: 'tab\there' overflows
            // its 101-DIP box only with the tab's 70 DIPs, and trims to 'tab').
            let mut tab_pen = 0.0f32;
            let line_own: Vec<Cl> = cls
                .iter()
                .filter(|c| c.bs >= a && c.bs < b)
                .map(|c| {
                    let width = if &text[c.bs..c.be] == "\t" { next_tab_stop(tab_pen, 4.0 * fmt.em) - tab_pen } else { c.width };
                    tab_pen += width;
                    Cl { width, ..*c }
                })
                .collect();
            let line_cls: Vec<&Cl> = line_own.iter().collect();
            let line_visible_w: f32 = line_cls.iter().filter(|c| c.bs < vis_byte).map(|c| c.width).sum();
            let trimmed = trim.is_some() && (line_visible_w > max_width || cut_trim[g]);

            // (start_u16, end_u16, caret-x before alignment, caret width) of the trimmed-away run,
            // whose caret DirectWrite maps onto the ellipsis sign (width 0 with no sign).
            let mut trim_rec: Option<(u32, u32, f32, f32)> = None;
            // Where the line's trailing-whitespace runs start in `runs_out` (none on a trimmed line).
            let mut ws_start = usize::MAX;
            let mut vis_delta = 0.0f32;
            // A trimmed line with no visible text has nothing to trim: native's TrimRange stops at once, so the line
            // keeps its whitespace and gets no sign, only the trimmed flag (r25 30723: a lone ' ' over cut lines).
            let (visible_w, after_ws) = if let Some(tr) = trim.filter(|_| trimmed && vis_byte > a) {
                // Trim the line: keep a fitting prefix (or path tail) and an ellipsis sign, drop the
                // rest. DirectWrite always draws the kept glyph run first, then the sign (even when
                // the sign is visually in front). The line's own metrics stay the full line.
                let plan = plan_trim(text, &line_cls, a, b, max_width, &tr, ellipsis_w, &brk);
                // What the trim keeps draws without its trailing whitespace, unless whitespace is all it keeps: a
                // walk that runs out at exactly zero keeps one cluster, a leading space too (VM r25 30371: ' lead'
                // in a box narrower than its sign keeps and draws the space).
                let keep_vis = Some(trailing_ws_start(text, plan.keep_start, plan.keep_end)).filter(|&v| v > plan.keep_start).unwrap_or(plan.keep_end);
                // The sign stands for what was cut, trailing whitespace included (r25 30885: '→ ' gives up its space
                // to the sign). A line that keeps all its text shows one only when lines below it were cut and the
                // sign fits after it (VM r18 23231: a lone overflowing inline object reports no sign; r25 30759: a
                // lone 'A' over cut lines, with no room for the sign, has none; 30998 fits one).
                let kept_all = plan.keep_start == a && plan.keep_end >= b;
                let sign = tr.ellipsis && (!kept_all || (cut_trim[g] && line_visible_w + ellipsis_w <= max_width));
                let sign_w = if sign { ellipsis_w } else { 0.0 };
                // A path trim's head draws first, without its trailing whitespace, the sign after it (VM layout queries: the
                // head's cut-off space is trimmed, its caret on the sign).
                let head_vis = Some(trailing_ws_start(text, a, plan.head_end)).filter(|&v| v > a).unwrap_or(plan.head_end);
                let head_x = if plan.head_end > a {
                    emit_range(&fonts, fmt, text, a, head_vis, (a, b), &mut 0.0, &levels, feats, &breaks, &hard, &obj_spans, 0.0, y, &mut runs_out, &mut obj_recs)
                } else {
                    0.0
                };
                let content_x = if plan.ellipsis_front { head_x + sign_w } else { 0.0 };
                let first = runs_out.len();
                let content_end_x = emit_range(&fonts, fmt, text, plan.keep_start, keep_vis, (a, b), &mut 0.0, &levels, feats, &breaks, &hard, &obj_spans, content_x, y, &mut runs_out, &mut obj_recs);
                // The sign follows the kept text's last character in its direction: right of a left-to-right run,
                // left of a right-to-left one, reported right-to-left there (VM r25 30453: after 'אבג' it sits
                // between 'abc' and the Hebrew; 30161: after Latin in a right-to-left paragraph, on the right).
                // What lies visually beyond it moves over. Path trimming puts it in front.
                let last = text[plan.keep_start..keep_vis].char_indices().next_back().map(|(i, _)| plan.keep_start + i);
                let at = last.map(|l| u16_pos(text, l));
                let run = at.and_then(|p| runs_out[first..].iter().filter_map(Run::glyph).find(|g| g.text_pos <= p && p < g.text_pos + g.clusters.len() as u32));
                let rtl = run.is_some_and(|g| g.level & 1 == 1);
                let sign_x = match run {
                    _ if plan.ellipsis_front => head_x,
                    Some(g) if rtl => g.x - g.advances.iter().sum::<f32>(),
                    Some(g) => g.x + g.advances.iter().sum::<f32>(),
                    None => content_end_x,
                };
                if sign && !plan.ellipsis_front {
                    for r in &mut runs_out[first..] {
                        let left = match r {
                            Run::Glyph(g) if g.level & 1 == 1 => g.x - g.advances.iter().sum::<f32>(),
                            Run::Glyph(g) => g.x,
                            Run::Inline(p) => p.x,
                        };
                        if left >= sign_x - 1e-4 {
                            r.shift_x(sign_w);
                        }
                    }
                }
                if sign {
                    runs_out.push(Run::Inline(InlinePlacement { rtl, ..ellipsis_place(sign_x, y, ellipsis_w, ellipsis_glyph) }));
                }
                trim_rec = Some(if plan.ellipsis_front {
                    (u16_pos(text, if plan.head_end > a { head_vis } else { a }), u16_pos(text, plan.keep_start), head_x, sign_w)
                } else {
                    (u16_pos(text, keep_vis), u16_pos(text, text.len()) + 1, sign_x, sign_w)
                });
                let total_w = if plan.ellipsis_front { content_end_x } else { content_end_x + sign_w };
                (total_w, total_w)
            } else {
                let visible_w = emit_range(&fonts, fmt, text, a, vis_byte, (a, b), &mut vis_delta, &levels, feats, &breaks, &hard, &obj_spans, 0.0, y, &mut runs_out, &mut obj_recs);
                ws_start = runs_out.len();
                let after_ws = emit_range(&fonts, fmt, text, vis_byte, b, (a, b), &mut 0.0, &levels, feats, &breaks, &hard, &obj_spans, visible_w, y, &mut runs_out, &mut obj_recs);
                (visible_w, after_ws)
            };

            if is_last && content_end < pe {
                runs_out.push(Run::Glyph(GlyphRun {
                    x: after_ws,
                    left: after_ws,
                    y,
                    top: y - baseline,
                    level: base.number(),
                    glyphs: Vec::new(),
                    advances: Vec::new(),
                    offsets: Vec::new(),
                    clusters: vec![0; newline_u16 as usize],
                    text_pos: u16_pos(text, content_end),
                    text: text[content_end..pe].to_string(),
                    font_asc: baseline,
                    font_height: line_height,
                    vhalf: 0.0,
                    em: fmt.em,
                    orient: 0,
                }));
            }

            // Justification (align 3, non-last line): expand the visible glyphs to fill the
            // layout width, repositioning the trailing-whitespace run; returns the justified
            // widths for the metrics below. Other alignments / last line: a no-op returning None.
            // DirectWrite justifies no trimmed line, and no paragraph's last line unless the format does not
            // wrap (NO_WRAP justifies every line, the last too).
            let skip = trimmed || (is_last && fmt.wrap != 1);
            let visible_runs = if ws_start == usize::MAX { runs_out.len() } else { ws_start } - line_start;
            let space_em = |pos: u32| {
                let f = fonts.at(u16_to_byte(text, pos));
                let tt = f.tt();
                tt.glyph_index(' ').and_then(|g| tt.glyph_hor_advance(g)).unwrap_or(0) as f32 / f.upm as f32
            };
            let kashida = |pos: u32| {
                let f = fonts.at(u16_to_byte(text, pos));
                let tt = f.tt();
                let g = tt.glyph_index('\u{640}')?;
                let adv = tt.glyph_hor_advance(g).filter(|&a| a > 0)?;
                Some((g.0, adv as f32 / f.upm as f32))
            };
            let just = justify::justify_line(&mut runs_out[line_start..], visible_runs, fmt.align, skip, max_width, matches!(opts.measuring, 1 | 2).then_some(ppd), &space_em, &kashida);
            if just.is_some() {
                // GetClusterMetrics reports the justified widths, so record them (keyed by UTF-16
                // start) to override the natural widths `cluster_metrics` computes below.
                for r in runs_out[line_start..].iter() {
                    if let Run::Glyph(g) = r {
                        justified.extend(g.cluster_widths());
                    }
                }
            }
            let (visible_w, after_ws) = just.unwrap_or((visible_w, after_ws));
            // A right-to-left paragraph puts the line's trailing whitespace at its visual end, the left, at the
            // paragraph's level (UAX #9 L1): it ends where the content begins (VM r25 30091: the space after
            // 'Y' is a level-1 run whose origin is Y's left edge, and TEXT_METRICS.left reaches its left edge).
            let ws_left = if fmt.base_rtl && ws_start < runs_out.len() {
                let mut pen = 0.0f32;
                for r in &mut runs_out[ws_start..] {
                    if let Run::Glyph(g) = r {
                        g.level = 1;
                        g.x = pen;
                        pen -= g.advances.iter().sum::<f32>();
                    }
                }
                after_ws - visible_w
            } else {
                // A left-to-right paragraph's trailing whitespace takes level 0 the same way, laid out left to right
                // after the content in text order (VM r25 30163: the space between two Hebrew words that ends a
                // line is a level-0 run at the end of the line's last letter).
                if ws_start < runs_out.len() {
                    let mut ws: Vec<usize> = (ws_start..runs_out.len()).filter(|&i| matches!(runs_out[i], Run::Glyph(_))).collect();
                    ws.sort_by_key(|&i| runs_out[i].glyph().map_or(0, |g| g.text_pos));
                    let mut pen = visible_w;
                    for i in ws {
                        if let Run::Glyph(g) = &mut runs_out[i] {
                            g.level = 0;
                            g.x = pen;
                            g.left = pen;
                            pen += g.advances.iter().sum::<f32>();
                        }
                    }
                }
                0.0
            };

            // Alignment: translate the whole line (visible runs, trailing-whitespace run and the
            // newline run) by the per-line offset (see `aligned_offset`: natural keeps the sub-pixel
            // DIP offset, GDI computes it on the whole device-pixel grid).
            let offset = aligned_offset(align, max_width, visible_w + vis_delta, matches!(opts.measuring, 1 | 2), ppd);
            if offset != 0.0 {
                for r in &mut runs_out[line_start..] {
                    // A right-to-left run still at its pen ends at the shifted pen plus its width (`GlyphRun::left`).
                    let end = match r {
                        Run::Glyph(g) if g.level & 1 == 1 => Some(g.advances.iter().sum::<f32>()).filter(|&w| g.x == g.left + w),
                        _ => None,
                    };
                    r.shift_x(offset);
                    if let (Some(w), Run::Glyph(g)) = (end, r) {
                        g.x = g.left + w;
                    }
                }
                for (_, _, o) in &mut obj_recs[obj_start..] {
                    o.x += offset;
                }
            }

            // Record the trimmed-away range's carets (on the ellipsis sign) after alignment: interior
            // positions sit at its leading edge, the end-of-text position at its trailing edge.
            if let Some((s, e, cx, cw)) = trim_rec {
                let top = y - line_bl;
                let x = cx + offset;
                let lead = Caret { x, y: top, left: x, top, width: cw, height: line_h, bidi: 0 };
                let trail = Caret { x: x + cw, left: x + cw, width: 0.0, ..lead };
                trimmed_spans.push((s, e, lead, trail));
            }

            // Line metrics (UTF-16 lengths); the last line carries the newline chars.
            let tw_byte = trailing_ws_start(text, a, b_eff);
            lines.push(LineMetrics {
                length: u16_pos(text, b_eff) - u16_pos(text, a),
                trailing_ws: u16_pos(text, b_eff) - u16_pos(text, tw_byte),
                newline: if is_last { newline_u16 } else { 0 },
                height: line_h,
                baseline: line_bl,
                is_trimmed: trimmed,
            });
            line_glyph_runs.push(runs_out[line_start..].iter().filter(|r| matches!(r, Run::Glyph(_))).count());
            cum_top += line_h;
            max_w = max_w.max(visible_w);
            min_off = min_off.min(offset - ws_left);
            max_right = max_right.max(offset + if ws_left > 0.0 { visible_w } else { after_ws });
            bidi_depth = bidi_depth.max(level_runs(&levels, a, vis_byte));
        }
    }

    // DirectWrite ends the layout with one more line when the text is empty or ends in a hard line
    // break: an empty string is a single empty line, and a trailing newline pushes the caret onto a
    // fresh empty line below. That final line carries no glyph run (invisible to hit_clusters and the
    // renderer), only its own line box (the base per-line height), the alignment offset and the
    // end-of-text caret. A mid-text empty line already comes from its paragraph's zero-glyph newline
    // run, so this fires only for the trailing case (`lines` empty, or the last line ended a para).
    let empty_line_caret = if extra_line && hidden[n_lines - 1] {
        lines.push(LineMetrics { length: 0, trailing_ws: 0, newline: 0, height: 0.0, baseline: 0.0, is_trimmed: true });
        None
    } else if extra_line {
        let (line_h, line_bl) = fmt.spaced(base_asc + base_desc, base_asc);
        let top = if upward { -cum_top - line_h } else { cum_top };
        // GDI measuring snaps the render origin onto the device grid; the empty line's metric box still
        // spans the *ideal* (un-snapped) origin as well. Native reports TEXT_METRICS.left as the lower of
        // the two and widthIncludingTrailingWhitespace as the higher, so an empty centre/trailing line
        // under GDI carries the snap residual as its trailing-whitespace width (r11: trailing 16129/16605,
        // centre 16222/16932, exact). Natural measuring has ideal == offset, so this is a no-op there.
        let ideal = align_offset(align, max_width, 0.0);
        let offset = if matches!(opts.measuring, 1 | 2) { snap_gdi(ideal, ppd) } else { ideal };
        lines.push(LineMetrics { length: 0, trailing_ws: 0, newline: 0, height: line_h, baseline: line_bl, is_trimmed: false });
        cum_top += line_h;
        min_off = min_off.min(offset).min(ideal);
        max_right = max_right.max(offset).max(ideal);
        Some(Caret { x: offset, y: top, left: offset, top, width: 0.0, height: line_h, bidi: base.number() })
    } else {
        None
    };

    // Vertical paragraph alignment (`DWRITE_PARAGRAPH_ALIGNMENT`): NEAR (0) keeps lines at the box
    // top (offset 0). FAR (1) bottoms the block of lines in the `max_height` box, CENTER (2) centers
    // it; both shift every line's baseline by one constant (negative when the content is taller than
    // the box, so a tall line overflows above — DirectWrite does not clip). `cum_top` is the total
    // height of all lines.
    // An upward flow stacks the lines from the box bottom, so NEAR and FAR swap.
    let para = match (fmt.para, upward) {
        (1, false) | (0, true) => 1,
        (2, _) => 2,
        _ => 0,
    };
    let voffset = match para {
        1 => max_height - cum_top,
        2 => (max_height - cum_top) / 2.0,
        _ => 0.0,
    };
    // Under a GDI measuring mode the paragraph-alignment shift snaps to the device grid with the same
    // +0.25-biased floor the horizontal offset uses (dwrite `AlignVertically`: the FAR/CENTER offset
    // goes through `floor(v·ppd + 0.25)/ppd`). A plain floor was one device row low at high dpi.
    let voffset = if matches!(opts.measuring, 1 | 2) { snap_gdi(voffset, ppd) } else { voffset };
    // One shift for every line: the offset, plus the total height for an upward stack.
    let shift = if upward { voffset + cum_top } else { voffset };
    if shift != 0.0 {
        for r in &mut runs_out {
            r.shift_y(shift);
        }
        for (_, _, o) in &mut obj_recs {
            o.y += shift;
        }
    }
    // The trailing empty line shifts with the rest (it has no run to carry the offset).
    let empty_line_caret = empty_line_caret.map(|c| Caret { y: c.y + shift, top: c.top + shift, ..c });

    // DirectWrite draws every glyph run first (all lines), then every inline object, so a text
    // renderer sees DrawGlyphRun calls before DrawInlineObject. Stable-partition to match.
    let runs_out: Vec<Run> = {
        let (glyphs, inlines): (Vec<Run>, Vec<Run>) = runs_out.into_iter().partition(|r| matches!(r, Run::Glyph(_)));
        glyphs.into_iter().chain(inlines).collect()
    };

    let clusters = cluster_metrics(&fonts, fmt, text, &bidi, &levels, &brk, feats, &breaks, &hard, &obj_spans, &justified);

    // `min_off` is the min line-alignment offset; it is non-finite only when a line's width overflows
    // to inf (an infinite em, or NO_WRAP glyphs whose advances sum past f32::MAX), giving a trailing /
    // centre offset of `max_width - inf = -inf`. Native emits that non-finite value in TEXT_METRICS.left
    // (VM-captured 2026-11-07: trailing/centre left = non-finite, not max_width and not 0), so pass it
    // through — every finite layout is unchanged (`min_off` is always set from ≥1 line's offset).
    let left = min_off;
    let metrics = TextMetrics {
        left,
        top: voffset,
        width: max_w,
        width_trailing: max_right - left,
        height: cum_top,
        layout_width: max_width,
        layout_height: max_height,
        bidi_depth,
        line_count: lines.len() as u32,
    };
    // Underline / strikethrough rectangles (filled after the glyph runs): one per decorated run — the
    // runs split at every decoration boundary, so each is wholly under a range or not. The line's
    // offset from the baseline and its thickness come from the run's font metrics scaled by the run's
    // em (a bigger per-range size thickens and lowers its underline). `position` is y-up (underline
    // below the baseline, strikethrough above), so the device top is `baseline − position·scale`.
    let mut decorations: Vec<Decoration> = Vec::new();
    if !opts.underline.is_empty() || !opts.strikethrough.is_empty() {
        // dwrite `DrawUnderlines`: consecutive underlined runs on a line draw as one underline — its
        // offset and thickness the width-weighted means of the runs' (each from its own font and em),
        // on the group's lowest baseline. The baseline snaps to the pixel grid and the offset below it
        // snaps independently; the thickness does not. Strikethrough snaps neither (prtl-strike is
        // byte-exact unsnapped).
        let glyph_runs: Vec<&GlyphRun> = runs_out.iter().filter_map(Run::glyph).collect();
        let span = |g: &GlyphRun| {
            let bs = u16_to_byte(text, g.text_pos);
            (bs, u16_to_byte(text, g.text_pos + g.text.encode_utf16().count() as u32))
        };
        // A line's trailing whitespace is never decorated, whatever its range says (r23-r27: no underline or
        // strikethrough over any of 144 decorated trailing-space runs, in either direction).
        let mut trailing = vec![false; glyph_runs.len()];
        let mut first = 0;
        for n in &line_glyph_runs {
            let last = (first..first + n).rev().find(|&i| glyph_runs[i].text.chars().any(|c| !c.is_whitespace()));
            for t in &mut trailing[last.map_or(first, |l| l + 1)..first + n] {
                *t = true;
            }
            first += n;
        }
        let is_trailing = |g: &GlyphRun| glyph_runs.iter().position(|r| std::ptr::eq(*r, g)).is_some_and(|i| trailing[i]);
        let underlined = |g: &GlyphRun| {
            let (bs, be) = span(g);
            !g.glyphs.is_empty() && !is_trailing(g) && opts.underline.iter().any(|&(s, e)| s <= bs && be <= e)
        };
        let mut first = 0;
        for n in &line_glyph_runs {
            let line = &glyph_runs[first..first + n];
            first += n;
            let mut k = 0;
            while k < line.len() {
                if !underlined(line[k]) {
                    k += 1;
                    continue;
                }
                let end = (k..line.len()).find(|&j| !underlined(line[j])).unwrap_or(line.len());
                let (mut off, mut thick, mut total_w, mut base) = (0.0f32, 0.0f32, 0.0f32, f32::MIN);
                for g in &line[k..end] {
                    let font = fonts.at(span(g).0);
                    let (pos, th) = font.underline();
                    let width: f32 = g.advances.iter().sum();
                    off -= pos as f32 * g.em / font.upm as f32 * width;
                    thick += th as f32 * g.em / font.upm as f32 * width;
                    total_w += width;
                    base = base.max(g.y);
                }
                if total_w > 0.0 {
                    let offset = snap_underline_offset(off / total_w, opts.pixels_per_dip);
                    let y = snap_pixel(base, opts.pixels_per_dip) + offset;
                    // One DrawUnderline for the whole group, its width the runs' total (VM r21 26167: a single
                    // 10.44-wide underline over a 4.95 and a 5.49 run), so their seam pixel blends once.
                    let left = line[k..end].iter().map(|g| { let width: f32 = g.advances.iter().sum(); if g.level & 1 == 1 { g.x - width } else { g.x } }).fold(f32::INFINITY, f32::min);
                    decorations.push(Decoration { x: left, y, width: total_w, thickness: thick / total_w, offset, strike: false, base, raw_offset: off / total_w, orient: 0 });
                }
                k = end;
            }
        }
        for g in &glyph_runs {
            if g.glyphs.is_empty() || is_trailing(g) {
                continue;
            }
            let (bs, be) = span(g);
            let font = fonts.at(bs);
            let scale = g.em / font.upm as f32;
            let width: f32 = g.advances.iter().sum();
            let x = if g.level & 1 == 1 { g.x - width } else { g.x };
            if opts.strikethrough.iter().any(|&(s, e)| s <= bs && be <= e) {
                let (pos, thick) = font.strikeout();
                let soff = -(pos as f32) * scale;
                decorations.push(Decoration { x, y: g.y + soff, width, thickness: thick as f32 * scale, offset: soff, strike: true, base: g.y, raw_offset: soff, orient: 0 });
            }
        }
    }
    Layout {
        runs: runs_out,
        decorations,
        lines,
        clusters,
        metrics,
        baseline,
        line_height,
        total_u16: u16_pos(text, text.len()),
        trimmed_spans,
        objects: obj_recs,
        empty_line_caret,
        column_width: None,
        sign_lift: 0.0,
    }
}

/// Cluster metrics over the whole string in logical order.
fn cluster_metrics(
    fonts: &Fonts,
    fmt: &TextFormat,
    text: &str,
    bidi: &BidiInfo,
    levels: &[Level],
    brk: &std::collections::HashMap<usize, BreakOpportunity>,
    feats: &[Feature],
    breaks: &[usize],
    hard: &[usize],
    objs: &[ObjSpan],
    justified: &std::collections::HashMap<u32, f32>,
) -> Vec<ClusterMetric> {
    let mut out = Vec::new();
    for para in &bidi.paragraphs {
        let (ps, pe) = (para.range.start, para.range.end);
        let content_end = content_end_of(text, ps, pe);
        for c in content_clusters(fonts, fmt, text, ps, content_end, levels, feats, breaks, hard, objs) {
            let mut bits = 0u16;
            if brk.contains_key(&c.be) {
                bits |= 1; // canWrapLineAfter
            }
            if text[c.bs..c.be].chars().next().unwrap().is_whitespace() {
                bits |= 2;
            }
            if c.level & 1 == 1 {
                bits |= 16;
            }
            let width = justified.get(&u16_pos(text, c.bs)).copied().unwrap_or(c.width);
            out.push(ClusterMetric { width, length: (u16_pos(text, c.be) - u16_pos(text, c.bs)) as u16, bits });
        }
        // The trailing newline is its own cluster: zero width, isNewline, canWrapAfter.
        if content_end < pe {
            out.push(ClusterMetric {
                width: 0.0,
                length: (u16_pos(text, pe) - u16_pos(text, content_end)) as u16,
                bits: 0b111, // canWrapLineAfter | isWhitespace | isNewline
            });
        }
    }
    out
}

// Done: leading/trailing/centre alignment (`AlignHorizontally`, natural measuring).
//
// Not yet implemented. The harness `text_layout` op (frozen for this task) cannot yet express
// the inputs/outputs for the starred items, so they need a scene-format change before they can
// be captured and verified; the rest are blocked on another track:
//  - justification (align 3): byte-exact spread needs DirectWrite's per-glyph
//    DWRITE_JUSTIFICATION_OPPORTUNITY model (`TextAnalyzer::GetJustificationOpportunities`) plus
//    the priority/residual distribution in `JustifyGlyphAdvances`/`GetJustifiedGlyphAdvanceAdjustments`.
//    rustybuzz exposes HarfBuzz glyph props, not DWrite justification opportunities, so this is a
//    separate reimplementation. Lays out leading until then.
//  - GDI-compatible measuring (whole-pixel advances + baseline snapping, and the pixel-grid snap
//    of the alignment offset in `AlignHorizontally`). BLOCKED on hinted GDI advances: e.g. Arial
//    'f' at 21px is 6px under GDI_CLASSIC but 5px rounded from the natural advance, so it is not a
//    rounding of the natural metric — it needs TrueType-hinted GdiCompatibleGlyphMetrics, which
//    rustybuzz/ttf_parser do not compute. Source: dwrite GetGdiCompatibleGlyphMetrics /
//    the TrueType rasteriser; coordinate with the hinting-interpreter agent.
//  - System font fallback. BLOCKED on fonts, not algorithm. The harness records the resolved family
//    per run (text2c-05-fallback: "Hi 日本語" at Arial → run0 "Hi " Arial, run1 "日本語" Yu Gothic UI),
//    so fallback IS captured and testable. But reproducing run1 byte-for-byte needs the VM's system
//    fallback collection (`IDWriteFactory::GetSystemFontFallback` → `MapCharacters`) AND the actual
//    fallback faces (Yu Gothic UI, …), which are neither installed here (ref/windows/Fonts holds only
//    Arial) nor redistributable into the repo/tests. Itemizing by cmap coverage would reproduce the
//    run split and the Arial run, but not run1's glyphs/advances/family. By design:
//    cross-platform text fallback stays functional coverage until a portable, licensed collection
//    with the VM's fallback order exists. Stop here; the capture is the reference for that day.

#[cfg(test)]
mod tests {
    use super::*;

    // Regression: a layout whose line width overflows to inf (here an infinite em) reports a
    // NON-FINITE `metrics.left` under trailing/centre alignment — the alignment offset is
    // `max_width - inf = -inf`. Native emits that non-finite value in TEXT_METRICS.left
    // (VM-captured 2026-11-07), not 0 and not max_width. Leading alignment keeps a finite left.
    #[test]
    fn inf_em_alignment_left_is_non_finite() {
        let font = Font::from_bytes(include_bytes!("../color/testdata/colrtest.ttf").to_vec(), 0)
            .expect("load test font");
        let mk = |align| TextFormat { em: f32::INFINITY, base_rtl: false, align, wrap: 1, para: 0, flow: 0, line_spacing: (0, 0.0, 0.0) };
        for align in [1u8, 2] {
            let left = layout(&font, &mk(align), "Ag", 200.0, 200.0).metrics.left;
            assert!(!left.is_finite(), "align {align}: expected non-finite left, got {left}");
        }
        assert!(layout(&font, &mk(0), "Ag", 200.0, 200.0).metrics.left.is_finite());
    }

    /// VM r25 (Segoe UI Emoji's family sequence man, woman, girl under GDI measuring): DirectWrite reports whole-pixel
    /// offsets. The man's placement and the woman's kern round on their own. The zero-advance girl hangs on the woman by
    /// a mark anchor plus a placement, which round separately, and she steps back over the woman's advance, the bold
    /// pixel included. Right to left, the offsets point left. 30643: right to left, GDI natural. 30720: left to right.
    /// 30703: bold-simulated, right to left. 30153: bold-simulated, GDI classic.
    #[test]
    fn a_gdi_run_reports_whole_pixel_offsets() {
        let Some(bytes) = d2d_reference::fonts_dir().and_then(|d| std::fs::read(d.join("seguiemj.ttf")).ok()) else { return };
        let font = Font::from_bytes(bytes, 0).expect("load Segoe UI Emoji");
        for (em, measuring, rtl, bold, want) in [
            (37.814377, 2, true, 0, [2.0, 1.0, 10.0]),
            (35.255348, 2, false, 0, [7.0, 0.0, -28.0]),
            (79.57112, 2, true, 41, [5.0, 3.0, 19.0]),
            (41.58361, 1, false, 41, [9.0, 0.0, -36.0]),
        ] {
            let fmt = TextFormat { em, base_rtl: rtl, align: 0, wrap: 1, para: 0, flow: 0, line_spacing: (0, 0.0, 0.0) };
            let opts = LayoutOpts { features: &[], trimming: None, inline: &[], fonts: &[], underline: &[], strikethrough: &[], measuring, pixels_per_dip: 1.0, base_bold_delta_du: bold, transform: None };
            let lay = layout_ex(&font, &fmt, "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}", 200.0, 200.0, &opts);
            let run = lay.runs.iter().filter_map(Run::glyph).find(|g| g.glyphs.len() == 3).expect("one family run");
            assert_eq!(run.glyphs, [1285, 1466, 1273]);
            assert_eq!(run.offsets.iter().map(|o| o.0).collect::<Vec<_>>(), want, "em {em} measuring {measuring} rtl {rtl} bold {bold}");
        }
    }

    /// VM (GDI classic at 1.25 pixels per DIP): a line whose whole-pixel advances fill the box exactly keeps its last
    /// word, though the f32 sum of their DIP widths lands past the box. Segoe UI 9 'The quick brown fox ' is 105 px in an
    /// 84-DIP (105-px) box; Arial 12 '... jumps over ' is 174 px in 139.2 DIPs.
    #[test]
    fn a_gdi_line_that_fills_the_box_to_the_pixel_fits() {
        let Some(dir) = d2d_reference::fonts_dir() else { return };
        for (file, em, box_w, want) in [("segoeui.ttf", 9.0, 84.0, vec![20, 20, 3]), ("arial.ttf", 12.0, 139.2, vec![26, 17])] {
            let Ok(bytes) = std::fs::read(dir.join(file)) else { return };
            let font = Font::from_bytes(bytes, 0).expect("load font");
            let fmt = TextFormat { em, base_rtl: false, align: 0, wrap: 0, para: 0, flow: 0, line_spacing: (0, 0.0, 0.0) };
            let opts = LayoutOpts { features: &[], trimming: None, inline: &[], fonts: &[], underline: &[], strikethrough: &[], measuring: 1, pixels_per_dip: 1.25, base_bold_delta_du: 0, transform: None };
            let lay = layout_ex(&font, &fmt, "The quick brown fox jumps over the lazy dog", box_w, 90.0, &opts);
            assert_eq!(lay.lines.iter().map(|l| l.length).collect::<Vec<_>>(), want, "{file}");
        }
    }

    /// VM r25 30743 (Segoe UI Emoji's family sequence upright in a top-to-bottom column, GDI natural, em 15.77 at 1.5
    /// pixels per DIP, bold-simulated): the cluster takes one vertical advance; its glyphs sit across the column where
    /// the horizontal shaping puts them, the cluster centred, and the girl back at the cluster's start along it.
    #[test]
    fn an_upright_cluster_sits_where_its_horizontal_shaping_puts_it() {
        let Some(bytes) = d2d_reference::fonts_dir().and_then(|d| std::fs::read(d.join("seguiemj.ttf")).ok()) else { return };
        let font = Font::from_bytes(bytes, 0).expect("load Segoe UI Emoji");
        let fmt = TextFormat { em: 15.768087, base_rtl: false, align: 0, wrap: 1, para: 0, flow: 2, line_spacing: (0, 0.0, 0.0) };
        let opts = LayoutOpts { features: &[], trimming: None, inline: &[], fonts: &[], underline: &[], strikethrough: &[], measuring: 2, pixels_per_dip: 1.5, base_bold_delta_du: 41, transform: None };
        let lay = layout_ex(&font, &fmt, "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}", 200.0, 200.0, &opts);
        let run = lay.runs.iter().filter_map(Run::glyph).find(|g| g.glyphs.len() == 3).expect("one family run");
        assert_eq!(run.glyphs, [1285, 1466, 1273]);
        assert_eq!(run.advances, [0.0, 16.0, 0.0]);
        let offsets: Vec<(f32, f32)> = run.offsets.iter().map(|&(a, b)| (a + 0.0, b)).collect();
        assert_eq!(offsets, [(0.0, -4.0), (0.0, 6.0), (-16.0, -8.0 / 3.0)]);
    }

    /// VM r25 30497 (a range boundary inside 👨 of the family sequence): the next run, "ZWJ 👩 ZWJ 👧", clusters its
    /// leading ZWJ alone, as Unicode's extended grapheme clusters of the run's own text do; the rest is one cluster.
    #[test]
    fn a_zwj_starting_its_run_is_a_cluster_of_its_own() {
        let Some(bytes) = d2d_reference::fonts_dir().and_then(|d| std::fs::read(d.join("seguiemj.ttf")).ok()) else { return };
        let font = Font::from_bytes(bytes, 0).expect("load Segoe UI Emoji");
        let text = "\u{200D}\u{1F469}\u{200D}\u{1F467}";
        for upright in [false, true] {
            let glyphs = shape_run(&font, text, 0, text.len(), (0, text.len()), false, &[], upright);
            let got: Vec<(u16, u32)> = glyphs.iter().map(|g| (g.id, g.cluster)).collect();
            assert_eq!(got, [(3, 0), (1466, 3), (1273, 3)], "upright {upright}");
        }
    }

    // Regression: an EMPTY layout under a GDI measuring mode with centre/trailing alignment reports the
    // device-snap residual as its widthIncludingTrailingWhitespace. The render origin is grid-snapped,
    // but the metric box also spans the un-snapped ideal origin — TEXT_METRICS.left is the lower of the
    // two, width_trailing the higher. Values VM-captured 2026-11-07 (r11 16129/16222/16605/16932).
    #[test]
    fn empty_gdi_aligned_trailing_width_is_snap_residual() {
        let font = Font::from_bytes(include_bytes!("../color/testdata/colrtest.ttf").to_vec(), 0)
            .expect("load test font");
        // (dwrite_align, measuring, max_width, ppd, expect_left, expect_width_trailing)
        let cases = [
            (1u8, 1u32, 25.293354034423828f32, 2.0f32, 25.0f32, 0.2933540344238281f32),
            (2, 1, 94.4090576171875, 2.0, 47.0, 0.20452880859375),
            (1, 2, 1.1327617168426514, 2.0, 1.0, 0.13276171684265137),
            (2, 2, 14.349650382995605, 1.25, 7.174825191497803, 0.025174617767333984),
        ];
        for (align, measuring, mw, ppd, el, ewt) in cases {
            let fmt = TextFormat { em: 16.0, base_rtl: false, align, wrap: 0, para: 0, flow: 0, line_spacing: (0, 0.0, 0.0) };
            let opts = LayoutOpts { features: &[], trimming: None, inline: &[], fonts: &[], underline: &[], strikethrough: &[], measuring, pixels_per_dip: ppd, base_bold_delta_du: 0, transform: None };
            let m = layout_ex(&font, &fmt, "", mw, 200.0, &opts).metrics;
            assert_eq!(m.left, el, "align {align} meas {measuring}: left");
            assert_eq!(m.width_trailing, ewt, "align {align} meas {measuring}: width_trailing");
        }
    }

    // Regression: under a FAR/CENTER vertical paragraph alignment the whole line block shifts down by a
    // large voffset. The first line's caret top must equal that voffset exactly (line top 0 + voffset),
    // i.e. max_height − metrics.height. Reconstructing it as baseline − font_asc AFTER the shift loses a
    // ULP to cancellation (r11 16945/16956). The run carries its own `top`, so the shift is exact.
    #[test]
    fn vertical_align_caret_top_no_cancellation() {
        let font = Font::from_bytes(include_bytes!("../color/testdata/colrtest.ttf").to_vec(), 0)
            .expect("load test font");
        let fmt = TextFormat { em: 44.0, base_rtl: false, align: 0, wrap: 1, para: 1, flow: 0, line_spacing: (0, 0.0, 0.0) }; // far
        let opts = LayoutOpts { features: &[], trimming: None, inline: &[], fonts: &[], underline: &[], strikethrough: &[], measuring: 0, pixels_per_dip: 1.0, base_bold_delta_du: 0, transform: None };
        let max_height = 300.0;
        let lay = layout_ex(&font, &fmt, "x", 400.0, max_height, &opts);
        let top = lay.hit_test_text_position(0).top;
        assert_eq!(top, max_height - lay.metrics.height, "far-aligned caret top must be voffset exactly");
    }
}

/// Whether `ch` stands upright in vertical text (Unicode Vertical_Orientation U or Tu; UAX #50): CJK,
/// kana, Hangul, fullwidth forms, emoji and the symbols that keep their orientation. Every other
/// character is rotated with its run.
fn upright(ch: char) -> bool {
    matches!(ch as u32,
        0xA7 | 0xA9 | 0xAE | 0xB1 | 0xBC..=0xBE | 0xD7 | 0xF7 | 0x2EA..=0x2EB
        | 0x1100..=0x11FF | 0x1401..=0x167F | 0x18B0..=0x18FF
        | 0x2016 | 0x2020..=0x2021 | 0x2030..=0x2031 | 0x203B..=0x203C | 0x2042 | 0x2047..=0x2049 | 0x2051
        | 0x20DD..=0x20E0 | 0x20E2..=0x20E4 | 0x2100..=0x2101 | 0x2103..=0x2109 | 0x210F | 0x2113..=0x2114
        | 0x2116..=0x2117 | 0x211E..=0x2123 | 0x2125 | 0x2127 | 0x2129 | 0x212E | 0x2135..=0x213F
        | 0x2145..=0x214A | 0x214C..=0x214D | 0x214F..=0x2189 | 0x218C..=0x218F | 0x221E | 0x2234..=0x2235
        | 0x2300..=0x2307 | 0x230C..=0x231F | 0x2322..=0x2328 | 0x232B..=0x237B | 0x237D..=0x239A
        | 0x23BE..=0x23CD | 0x23CF | 0x23D1..=0x23DB | 0x23E2..=0x2422 | 0x2424..=0x24FF | 0x25A0..=0x2619
        | 0x2620..=0x2767 | 0x2776..=0x2793 | 0x2B12..=0x2B2F | 0x2B50..=0x2B59 | 0x2BB8..=0x2BFF
        | 0x2E80..=0xA4CF | 0xA960..=0xA97F | 0xAC00..=0xD7FF | 0xE000..=0xFAFF | 0xFE10..=0xFE1F
        | 0xFE30..=0xFE6F | 0xFF01..=0xFF60 | 0xFFE0..=0xFFE7 | 0x1F000..=0x1FAFF | 0x20000..=0x3FFFD)
}

/// Whether the character at byte `b` stands upright: a character continuing its cluster stands as its base does,
/// also where a format boundary cuts it into a run of its own (r25 30046: a ZWJ of the family emoji stays upright).
fn upright_at(text: &str, mut b: usize) -> bool {
    while continues_cluster(text, b) {
        b = text[..b].char_indices().next_back().map_or(0, |(i, _)| i);
    }
    text[b..].chars().next().is_some_and(upright)
}

/// A vertical reading direction lays out as a horizontal layout in the column frame: each line is a
/// column along `max_height` (top-to-bottom reads as left-to-right there, bottom-to-top as right-to-left,
/// which the format's `base_rtl` already carries), and the lines stack across `max_width`, left to right
/// (flow 2) or right to left (flow 3). Runs split where characters change between upright and rotated.
/// Each run keeps its position along the column as `y`; across, a rotated run sits at the column centre
/// less half its ascent-descent difference (its em box centred on the column), an upright one on the
/// centre. VM (r25 vertical queries): Segoe UI and Calibri offsets match their win ascent/descent.
/// ponytail: underline/strikethrough are not placed vertically yet (dropped), and the line, caret and
/// metrics queries keep the column frame.
fn layout_vertical(font: &Font, fmt: &TextFormat, text: &str, max_width: f32, max_height: f32, opts: &LayoutOpts) -> Layout {
    let mut cuts = Vec::new();
    let mut prev = None;
    for (b, ch) in text.char_indices() {
        // A character continuing its cluster takes its base's orientation (r25 30333: the VS16 of '❤️' stays
        // in the upright emoji's run).
        if continues_cluster(text, b) {
            continue;
        }
        let u = upright(ch);
        if prev.is_some_and(|p| p != u) {
            cuts.push(b);
        }
        prev = Some(u);
    }
    // The columns lay out as horizontal lines in a frame turned 90° clockwise: line length = the box
    // height, the lines stack across the box width from its right edge (frame y = width − device x).
    // Left-to-right columns (flow 2) stack upward in that frame, the first at the device left, with near
    // and far paragraph alignment swapped — `layout_core`'s upward flow (VM r25 30160: NEAR left 0.57 =
    // width − floor(width − total)).
    let mut lay = layout_core(font, fmt, text, max_height, max_width, opts, &cuts);
    // A decoration starts at its first run's origin and turns with it (VM r25: every vertical underline and
    // strikethrough reports orientation 90° at its run's x and y). An upright run's strikethrough is centred
    // on the central baseline; its underline hangs from the run's own baseline, as a rotated run's does (VM r25
    // 30488: an upright emoji's underline at its central line less Segoe UI Emoji's 848-unit offset). Neither
    // snaps to the pixel grid here.
    let frame_runs: Vec<(f32, f32, f32, f32, bool)> = lay
        .runs
        .iter()
        .filter_map(Run::glyph)
        .map(|g| {
            let w: f32 = g.advances.iter().sum();
            let left = if g.level & 1 == 1 { g.x - w } else { g.x };
            (left, left + w, g.y, g.vhalf, g.text.chars().next().is_some_and(upright))
        })
        .collect();
    for d in &mut lay.decorations {
        let run = frame_runs.iter().find(|r| (r.2 - d.base).abs() < 1e-3 && (r.0 - d.x).abs() < 1e-3);
        let (vhalf, up) = run.map_or((0.0, false), |r| (r.3, r.4));
        let centred = up && d.strike;
        d.offset = if centred { -d.thickness / 2.0 } else { d.raw_offset };
        (d.x, d.y) = (max_width - d.base - if centred { 0.0 } else { vhalf }, d.x);
        d.base = f32::NAN;
        d.orient = 1;
    }
    lay.column_width = Some(max_width);
    // A bottom-to-top column draws its sign upward from the foot of the box, with the glyph's own advance, so a bold
    // sign's widening lands above it: the dots stay where the regular sign's are (VM r25 30351, bold and regular).
    lay.sign_lift = match opts.trimming {
        Some(t) if fmt.base_rtl && t.ellipsis && t.upright_sign > 0.0 => t.upright_bold * fmt.em,
        _ => 0.0,
    };
    for r in &mut lay.runs {
        // A trimming sign stands upright down the column (VM r25 30351 reports it sideways).
        if let Run::Inline(p) = r {
            p.sideways |= p.sign.is_some();
        }
        let Run::Glyph(g) = r else { continue };
        let up = upright_at(text, u16_to_byte(text, g.text_pos));
        // `g.y` is the central baseline in the frame. An upright run hangs on it; a rotated run's own
        // baseline sits `vhalf` further left.
        // An upright run keeps its unrounded place across the columns in every measuring mode; the renderer's pixel
        // snapper rounds the column once origin and place are summed (VM sweeps: Segoe UI Emoji upright at em 40 under
        // natural measuring reports 26.6015625 and 53.3984375 and steps where the origin brings that sum to a half pixel).
        let x = max_width - g.y - if up { 0.0 } else { g.vhalf };
        (g.x, g.y, g.top) = (x, g.x, g.x);
        g.orient = if up { 2 } else { 1 };
    }
    lay
}

