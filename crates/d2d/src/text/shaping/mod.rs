//! Shaping: an itemized run (one font, script, direction) → positioned glyphs.
//!
//! DirectWrite shapes each run through the OpenType shaper in `TextShaping.dll`
//! (`IDWriteTextAnalyzer::GetGlyphs`/`GetGlyphPlacements`). rustybuzz is a line-for-line
//! port of HarfBuzz, which implements the same OpenType shaping model; we rely on it and
//! verify glyph ids/advances/offsets against native per run (`tests/text_layout`).
//!
//! Output is in font **design units**; layout scales by `emSize/upm` the way DirectWrite
//! does for natural measuring (ideal, fractional advances).

use rustybuzz::ttf_parser;

/// A loaded TrueType/OpenType face. Owns the bytes so a borrowed `rustybuzz::Face` or
/// `ttf_parser::Face` can be built per use (both borrow the data).
pub struct Font {
    data: Vec<u8>,
    index: u32,
    /// Units per em (design-space scale; Arial = 2048).
    pub upm: u16,
    /// Variable-font instance axes `(tag, user value)`, empty for the default/static instance. When
    /// set, the shaper applies the instance's advances/offsets (HVAR, else `gvar` phantoms) so a
    /// variable-font instance lays out with native's instanced metrics, not the default master's.
    vars: Vec<ttf_parser::Variation>,
}

/// Validate a TrueType Collection header against the data length before handing it to ttf_parser,
/// which would otherwise trip a debug-only bounds assertion on a crafted collection. Returns `true`
/// for a non-collection (a bare sfnt's directory is based at offset 0 and cannot reach u32::MAX, so
/// it is left to ttf_parser). For a `ttcf`, requires the collection offset table (`12 + numFonts*4`),
/// the selected font's offset, and that font's sfnt header + table directory (`numTables*16`) to all
/// lie within `data`. All sfnt fields are big-endian. Arithmetic is in u64 (every term ≤ ~2^34).
fn ttc_header_ok(data: &[u8], index: u32) -> bool {
    if data.len() < 12 || data[0..4] != *b"ttcf" {
        return true;
    }
    let be32 = |o: usize| u32::from_be_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]) as u64;
    let len = data.len() as u64;
    let num_fonts = be32(8);
    // The offset table (numFonts × Offset32 at byte 12) must fit, and `index` must select a font.
    if 12 + num_fonts * 4 > len || index as u64 >= num_fonts {
        return false;
    }
    let font_off = be32(12 + index as usize * 4);
    // The selected font's sfnt header (12 bytes) + table directory (numTables × 16-byte records) must fit.
    if font_off + 12 > len {
        return false;
    }
    let off = font_off as usize;
    let num_tables = u16::from_be_bytes([data[off + 4], data[off + 5]]) as u64;
    font_off + 12 + num_tables * 16 <= len
}

/// DirectWrite kerns from GPOS alone when a face has it: the legacy `kern` table goes unused even for a script
/// whose GPOS holds no kerning, where HarfBuzz falls back to the table (r25 30940: Malgun Gothic's GPOS has no
/// kern feature; 30153: Leelawadee UI's kerns only Thai; 30762: Segoe UI Emoji's 'latn' has none). Hide the
/// table from the shaper in this face's own copy by renaming its directory tag ('kern' → 'kerm' keeps the
/// directory sorted). The bytes were validated by `ttc_header_ok` and ttf_parser.
fn hide_legacy_kern(data: &mut [u8], index: u32) {
    let be32 = |d: &[u8], o: usize| d.get(o..o + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize);
    let dir = if data.starts_with(b"ttcf") { be32(data, 12 + index as usize * 4) } else { Some(0) };
    let Some(dir) = dir else { return };
    let Some(n) = data.get(dir + 4..dir + 6).map(|b| u16::from_be_bytes([b[0], b[1]]) as usize) else { return };
    let records = (0..n).map(|i| dir + 12 + i * 16).filter(|&r| r + 4 <= data.len());
    let (mut gpos, mut kern) = (false, None);
    for r in records {
        match &data[r..r + 4] {
            b"GPOS" => gpos = true,
            b"kern" => kern = Some(r),
            _ => {}
        }
    }
    if let (true, Some(r)) = (gpos, kern) {
        data[r + 3] = b'm';
    }
}

impl Font {
    pub fn from_bytes(data: Vec<u8>, index: u32) -> Option<Self> {
        Self::from_bytes_var(data, index, &[])
    }

    /// As [`from_bytes`](Self::from_bytes), instanced at the variable-font axes `vars` (`(packed tag,
    /// user value)`, the DWRITE little-endian tag). Empty = the default instance (identical to
    /// `from_bytes`). A static font ignores the axes.
    pub fn from_bytes_var(data: Vec<u8>, index: u32, vars: &[(u32, f32)]) -> Option<Self> {
        // Reject a malformed TrueType Collection up front: ttf_parser reads `numFonts` then
        // `read_array32::<Offset32>(numFonts)`, and seeks each font's table directory at a file-supplied
        // offset — a huge numFonts or offset makes those reads exceed u32::MAX and trip a debug_assert
        // inside ttf_parser (parser.rs `read_bytes`). The catch_unwind below is only a partial firewall
        // (a panic=abort build cannot catch it), so validate the TTC header against `data.len()` first.
        if !ttc_header_ok(&data, index) {
            return None;
        }
        // Second line of defence: ttf_parser's other debug-only assertions compile out in release (the
        // bounds-checked read then returns None, rejecting the font); catch the debug-build panic so a
        // malformed font is rejected uniformly rather than unwinding across the COM ABI.
        let upm = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ttf_parser::Face::parse(&data, index).ok().map(|f| f.units_per_em())
        }))
        .ok()
        .flatten()?;
        let vars = vars.iter().map(|&(t, v)| ttf_parser::Variation { axis: ttf_parser::Tag(u32::from_le_bytes(t.to_be_bytes())), value: v }).collect();
        let mut data = data;
        hide_legacy_kern(&mut data, index);
        Some(Font { data, index, upm, vars })
    }

    pub fn hb(&self) -> rustybuzz::Face<'_> {
        let mut f = rustybuzz::Face::from_slice(&self.data, self.index).expect("already validated in from_bytes");
        if !self.vars.is_empty() {
            let vs: Vec<rustybuzz::Variation> = self.vars.iter().map(|v| rustybuzz::Variation { tag: v.axis, value: v.value }).collect();
            f.set_variations(&vs);
        }
        f
    }

    /// The vertical advance DirectWrite gives every glyph of a face without a `vmtx` table: the OS/2 typo
    /// ascender less descender, where HarfBuzz takes the line ascender and descender (r25: a vertical layout's
    /// trimming sign from Segoe UI Symbol spans typo 1491 + 431, not hhea 2210 + 514). `None` when the face has
    /// `vmtx`.
    pub fn default_vadvance(&self) -> Option<i32> {
        let tt = self.tt();
        if tt.tables().vmtx.is_some() {
            return None;
        }
        Some(tt.typographic_ascender()? as i32 - tt.typographic_descender()? as i32)
    }

    /// The x of the base's and the mark's anchors (design units) when the face's GPOS attaches `mark` to `base` by a
    /// mark-to-base lookup; `None` when no such lookup covers the pair. Each lookup that takes the mark (its mark
    /// filtering set permitting) replaces the attachment, so the last one decides.
    pub fn mark_anchors(&self, base: u16, mark: u16) -> Option<(i32, i32)> {
        use ttf_parser::gpos::PositioningSubtable;
        let tt = self.tt();
        let (gpos, gdef) = (tt.tables().gpos?, tt.tables().gdef);
        let (base, mark) = (ttf_parser::GlyphId(base), ttf_parser::GlyphId(mark));
        gpos.lookups
            .into_iter()
            .filter(|l| l.mark_filtering_set.is_none_or(|set| gdef.is_some_and(|g| g.is_mark_glyph(mark, Some(set)))))
            .filter_map(|l| {
                l.subtables.into_iter::<PositioningSubtable>().find_map(|s| match s {
                    PositioningSubtable::MarkToBase(t) => {
                        let (class, m) = t.marks.get(t.mark_coverage.get(mark)?)?;
                        Some((t.anchors.get(t.base_coverage.get(base)?, class)?.x as i32, m.x as i32))
                    }
                    _ => None,
                })
            })
            .last()
    }

    /// This face's variable-font instance as normalized coords; empty for a static face or the default instance.
    pub fn coords(&self) -> Vec<skrifa::raw::types::F2Dot14> {
        let axes: Vec<(u32, f32)> = self.vars.iter().map(|v| (u32::from_le_bytes(v.axis.to_bytes()), v.value)).collect();
        if axes.is_empty() { Vec::new() } else { super::variations::normalized_from_dwrite(&self.data, self.index, &axes) }
    }

    /// The face index within the font file (non-zero only for a collection).
    pub fn index(&self) -> u32 {
        self.index
    }

    pub fn tt(&self) -> ttf_parser::Face<'_> {
        let mut f = ttf_parser::Face::parse(&self.data, self.index).expect("already validated in from_bytes");
        for v in &self.vars {
            f.set_variation(v.axis, v.value);
        }
        f
    }

    /// `post` underline `(position, thickness)` in design units; `position` is y-up (negative = below
    /// the baseline). DirectWrite draws the underline from these metrics; a font without the table
    /// falls back to a conventional 10%/5%-of-em line.
    pub fn underline(&self) -> (i32, i32) {
        self.tt().underline_metrics().map_or_else(
            || (-(self.upm as i32) / 10, self.upm as i32 / 20),
            |m| (m.position as i32, m.thickness as i32))
    }

    /// OS/2 strikeout `(position, thickness)` in design units; `position` is y-up (positive = above
    /// the baseline, through the text).
    pub fn strikeout(&self) -> (i32, i32) {
        self.tt().strikeout_metrics().map_or_else(
            || (self.upm as i32 / 4, self.upm as i32 / 20),
            |m| (m.position as i32, m.thickness as i32))
    }

    /// Whether `glyph_id` has an outline (a non-empty glyph). Bold simulation emboldens — and so bumps
    /// the advance of — only inked glyphs; an empty glyph (the space) keeps its advance.
    pub fn has_outline(&self, glyph_id: u16) -> bool {
        self.tt().glyph_bounding_box(ttf_parser::GlyphId(glyph_id)).is_some()
    }
}

/// One shaped glyph, in font design units. `cluster` is the byte/UTF-16 index back into
/// the run's source text (HarfBuzz cluster), matching DirectWrite's cluster map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glyph {
    pub id: u16,
    pub cluster: u32,
    pub advance: i32,
    pub dx: i32,
    pub dy: i32,
}

/// Shape one run. `rtl` sets the base direction; `script`/`lang` come from itemization
/// (None lets HarfBuzz guess, used only for isolated tests). Features are left to the
/// shaper's defaults, which enable the same ccmp/liga/clig/calt/kern/mark set DirectWrite
/// turns on for a horizontal run.
pub fn shape(
    font: &Font,
    text: &str,
    rtl: bool,
    script: Option<rustybuzz::Script>,
    lang: Option<rustybuzz::Language>,
) -> Vec<Glyph> {
    let mut buf = rustybuzz::UnicodeBuffer::new();
    buf.push_str(text);
    buf.set_direction(if rtl { rustybuzz::Direction::RightToLeft } else { rustybuzz::Direction::LeftToRight });
    if let Some(s) = script {
        buf.set_script(s);
    }
    if let Some(l) = lang {
        buf.set_language(l);
    }
    let out = rustybuzz::shape(&font.hb(), &[], buf);
    let infos = out.glyph_infos();
    let pos = out.glyph_positions();
    let mut glyphs: Vec<Glyph> = infos
        .iter()
        .zip(pos)
        .map(|(i, p)| Glyph {
            id: i.glyph_id as u16,
            cluster: i.cluster,
            advance: p.x_advance,
            dx: p.x_offset,
            dy: p.y_offset,
        })
        .collect();
    // HarfBuzz emits an RTL run in visual (left-to-right on screen) order; DirectWrite
    // stores glyph runs in logical (text) order with per-glyph advances — reverse to match.
    if rtl {
        glyphs.reverse();
    }
    glyphs
}

/// DirectWrite's Thai/Lao shaping classes: 0 other, 1 a leading or following vowel, 2 a consonant (or a
/// character that may carry marks: '-', the dashes, ×, NBSP, •, ◌, ◻◼◽◾), 3 a below vowel, 4 phinthu, 5 an
/// above vowel or tone mark, 6 sara am, 7 ZWNJ/ZWJ. From TextShaping.dll's class tables.
fn thai_lao_class(ch: char) -> u8 {
    const THAI: [u8; 93] = [
        0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 5, 1, 6, 5, 5, 5, 5, 3, 3, 4, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    const LAO: [u8; 96] = [
        0, 2, 2, 0, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0,
        2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 5, 1, 6, 5, 5, 5, 5, 3, 3, 4, 5, 3, 2, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 5,
        5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2,
    ];
    let c = ch as u32;
    match c {
        0x0E00..=0x0E5C => THAI[(c - 0x0E00) as usize],
        0x0E80..=0x0EDF => LAO[(c - 0x0E80) as usize],
        0x200C | 0x200D => 7,
        0x2D | 0x2012..=0x2015 | 0xD7 | 0xA0 | 0x2022 | 0x25CC | 0x25FB..=0x25FE => 2,
        _ => 0,
    }
}

/// The text DirectWrite's Thai/Lao shaping engine hands on to shaping, as (character, cluster byte offset)
/// pairs. The engine walks the characters by class and
/// - inserts a dotted circle before a mark with nothing to sit on, which the mark then forms its own cluster on:
///   an above vowel or tone mark not on a consonant (after ะ, า, ำ, a space, another dotted circle, or at the
///   start), a below vowel not on a consonant or after an above mark, a phinthu after nothing or after another
///   phinthu or an above mark, and a second sara am on one base;
/// - moves a below vowel typed after a phinthu in front of it;
/// - splits a sara am into its nikhahit, right after the base it sits on, and a sara aa in its place. HarfBuzz's
///   Thai shaper splits it the same way except where a joiner stands between, so only then is it split here,
///   the base through the sara aa one cluster as HarfBuzz would make them.
pub(crate) fn thai_lao_prepare(text: &str) -> Vec<(char, usize)> {
    let mut out: Vec<(char, usize)> = Vec::with_capacity(text.len());
    let Some(first) = text.chars().next() else { return out };
    // `state`: the class of the last base, 6 after a sara am, 8 after an inserted dotted circle.
    let mut state = thai_lao_class(first);
    let mut cur = state;
    // The current base (the last base, below vowel, phinthu or misplaced mark) and whether it took a sara am.
    let mut base = 0usize;
    let mut base_am = false;
    for (i, ch) in text.char_indices() {
        let prev = cur;
        cur = thai_lao_class(ch);
        let mut swap = false;
        let invalid = match cur {
            3 => {
                let bad = state != 2 || prev == 5;
                if !bad && prev == 4 {
                    cur = 4;
                    swap = true;
                }
                base_am = false;
                bad
            }
            4 => {
                base_am = false;
                state & 0xF7 == 0 || prev.wrapping_sub(4) <= 1
            }
            5 => state != 2,
            6 => {
                let bad = state != 2 || base_am;
                state = 6;
                base_am = true;
                bad
            }
            7 => false,
            _ => {
                state = cur;
                base_am = false;
                false
            }
        };
        if invalid {
            // The mark becomes the base its successors check against.
            out.push(('\u{25CC}', i));
            state = 8;
            base_am = cur == 6;
        }
        let joiner_since_base = out.get(base + 1..).is_some_and(|t| t.iter().any(|&(c, _)| thai_lao_class(c) == 7));
        if swap && out.last().is_some_and(|&(c, _)| thai_lao_class(c) == 4) {
            let (phinthu, at) = out.pop().unwrap();
            out.push((ch, at));
            out.push((phinthu, at));
        } else if cur == 6 && !invalid && joiner_since_base {
            let lao = ch as u32 >= 0x0E80;
            let at = out[base].1;
            out.insert(base + 1, (if lao { '\u{0ECD}' } else { '\u{0E4D}' }, at));
            for e in &mut out[base..] {
                e.1 = at;
            }
            out.push((if lao { '\u{0EB2}' } else { '\u{0E32}' }, at));
        } else {
            out.push((ch, i));
        }
        if invalid || !matches!(cur, 5..=7) {
            base = out.len() - 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn thai_marks_without_a_consonant_get_dotted_circles() {
        // ะ and า take no tone mark, nor does the text start; a consonant does (VM probes: 'ไน้เศะ้น' shapes its
        // second ้ on a dotted circle, its first on น).
        let at = |t: &str| super::thai_lao_prepare(t).iter().filter(|e| e.0 == '\u{25CC}').map(|&(_, b)| t[..b].chars().count()).collect::<Vec<_>>();
        assert_eq!(at("ไน้เศะ้น"), vec![6]);
        assert_eq!(at("ไหะ๋แฎี่"), vec![3]);
        assert_eq!(at("้ก"), vec![0]);
        assert_eq!(at("กำ"), Vec::<usize>::new());
        let text = |t: &str| super::thai_lao_prepare(t).iter().map(|e| e.0).collect::<String>();
        // A below vowel typed after a phinthu goes in front of it.
        assert_eq!(text("กฺุ"), "กฺุ");
        // A sara am's nikhahit goes right after its base, a joiner between or not.
        assert_eq!(text("ก\u{200D}่ำ"), "ก\u{0E4D}\u{200D}่\u{0E32}");
        assert_eq!(text("ก่ำ"), "ก่ำ");
    }

    use super::*;

    #[test]
    fn legacy_kern_hidden_only_beside_gpos() {
        // A bare sfnt directory: version, numTables, then 16-byte records (tag first).
        let dir = |tags: &[&[u8; 4]]| {
            let mut d = vec![0, 1, 0, 0, 0, tags.len() as u8, 0, 0, 0, 0, 0, 0];
            for t in tags {
                d.extend_from_slice(*t);
                d.extend_from_slice(&[0; 12]);
            }
            d
        };
        let mut with_gpos = dir(&[b"GPOS", b"head", b"kern"]);
        hide_legacy_kern(&mut with_gpos, 0);
        assert_eq!(&with_gpos[12 + 32..12 + 36], b"kerm");
        let mut without = dir(&[b"head", b"kern"]);
        hide_legacy_kern(&mut without, 0);
        assert_eq!(&without[12 + 16..12 + 20], b"kern");
    }

    // Regression (wic_fuzz font loader): a malformed TrueType Collection ('ttcf') makes ttf_parser
    // read past u32::MAX and trip a debug-only assertion; from_bytes must reject it (None), never
    // panic (a debug-build panic would unwind across the COM ABI; a panic=abort build cannot even be
    // caught). The ttc_header_ok guard rejects it before ttf_parser is reached.
    #[test]
    fn malformed_ttcf_is_rejected_not_panic() {
        // The 21 gate failures' class: a huge numFonts (the fuzz inputs had ~1.6e9) makes
        // read_array32 read 12 + numFonts*4 bytes, exceeding u32::MAX.
        let mut big = b"ttcf".to_vec();
        big.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // version
        big.extend_from_slice(&0x6200_d58eu32.to_be_bytes()); // numFonts ~1.6e9 (as in font_1314)
        assert!(!ttc_header_ok(&big, 0));
        assert!(Font::from_bytes(big, 0).is_none());
        // A per-font offset near u32::MAX trips the selected font's directory read; also rejected.
        let mut off = b"ttcf".to_vec();
        off.extend_from_slice(&0x0001_0000u32.to_be_bytes());
        off.extend_from_slice(&1u32.to_be_bytes()); // numFonts = 1
        off.extend_from_slice(&0xFFFF_FFF0u32.to_be_bytes()); // font 0 offset near u32::MAX
        assert!(!ttc_header_ok(&off, 0));
        assert!(Font::from_bytes(off, 0).is_none());
        // A non-collection is left to ttf_parser — the guard must not reject it (no over-rejection).
        assert!(ttc_header_ok(b"\x00\x01\x00\x00bare-sfnt-ish bytes........", 0));
    }
}
