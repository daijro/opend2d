//! Justified alignment (`DWRITE_TEXT_ALIGNMENT_JUSTIFIED`), byte-exact against native
//! DirectWrite. Unlike classic "stretch the spaces" justification, DirectWrite expands
//! *every* glyph: spaces get a word-spacing opportunity and each letter gets inter-character
//! expansion, applied as a mix of larger advances and per-glyph leading offsets.
//!
//! Reproduces DirectWrite's justification opportunities (the per-glyph
//! `DWRITE_JUSTIFICATION_OPPORTUNITY`s), `GetJustifiedGlyphAdvanceAdjustments` (the priority +
//! residual deficit distribution) and `ApplyGlyphAdvanceAdjustments` (advance += adjustment,
//! leading edge → glyph offset). Scope: Latin, natural measuring — see end of file.
//!
//! The opportunity rules for the Latin + space case (em = font size in DIPs):
//!  * space (`DWRITE_SCRIPT_JUSTIFY_BLANK`): `expansionMaximum = em/2`, priority 2, trailing
//!    edge only, residual-expandable.
//!  * letter/punctuation (`CHARACTER`/`NONE`): `expansionMaximum = em/8`, priority 3, both
//!    edges, no residual.
//!
//! `GetMaximumAdjustment` returns `expansionMaximum × (number of enabled edges)`.

use super::{GlyphRun, Run};

/// One glyph's justification opportunity (Latin subset of `DWRITE_JUSTIFICATION_OPPORTUNITY`).
struct Opp {
    run: usize,
    glyph: usize,
    /// The UTF-16 position of the glyph's cluster: its place in the line's logical order.
    unit: u32,
    /// `expansionMaximum`, the per-edge cap.
    per_edge: f32,
    /// `compressionMaximum`, the per-edge cap when the line is too long.
    compress: f32,
    lead: bool,
    trail: bool,
    priority: u8,
    residual: bool,
}

impl Opp {
    /// `GetMaximumAdjustment(opp, expand=true)`: the cap summed over the enabled edges.
    fn max_adjust(&self) -> f32 {
        self.per_edge * (self.lead as u32 + self.trail as u32) as f32
    }
}

/// DirectWrite's script properties flags by ISO 15924 code, from the reference dwrite.dll's script table:
/// 1 restrict caret to clusters, 2 uses word dividers, 4 discrete writing, 8 block writing, 16 distributed
/// within cluster, 32 connected writing, 64 cursive writing.
static SCRIPT_FLAGS: &[([u8; 4], u8)] = &[(*b"Zzzz", 0), (*b"Zyyy", 6), (*b"Zinh", 1), (*b"Arab", 98), (*b"Armn", 6), (*b"Avst", 6), (*b"Bali", 1), (*b"Bamu", 6), (*b"Batk", 6), (*b"Beng", 35), (*b"Bopo", 12), (*b"Brah", 6), (*b"Brai", 6), (*b"Bugi", 6), (*b"Buhd", 4), (*b"Cans", 6), (*b"Cari", 4), (*b"Cham", 7), (*b"Cher", 6), (*b"Copt", 6), (*b"Xsux", 12), (*b"Cprt", 4), (*b"Cyrl", 6), (*b"Dsrt", 6), (*b"Deva", 35), (*b"Egyp", 13), (*b"Ethi", 6), (*b"Geor", 6), (*b"Glag", 6), (*b"Goth", 6), (*b"Grek", 6), (*b"Gujr", 7), (*b"Guru", 35), (*b"Hani", 12), (*b"Hang", 15), (*b"Hano", 4), (*b"Hebr", 7), (*b"Hira", 12), (*b"Armi", 6), (*b"Phli", 6), (*b"Prti", 6), (*b"Java", 5), (*b"Kthi", 35), (*b"Knda", 7), (*b"Kana", 12), (*b"Kali", 6), (*b"Khar", 5), (*b"Khmr", 21), (*b"Laoo", 21), (*b"Latn", 6), (*b"Lepc", 7), (*b"Limb", 7), (*b"Linb", 12), (*b"Lisu", 6), (*b"Lyci", 6), (*b"Lydi", 6), (*b"Mlym", 7), (*b"Mand", 98), (*b"Mtei", 7), (*b"Mong", 98), (*b"Mymr", 7), (*b"Talu", 7), (*b"Nkoo", 98), (*b"Ogam", 34), (*b"Olck", 6), (*b"Ital", 6), (*b"Xpeo", 14), (*b"Sarb", 6), (*b"Orkh", 6), (*b"Orya", 7), (*b"Osma", 6), (*b"Phag", 98), (*b"Phnx", 4), (*b"Rjng", 5), (*b"Runr", 4), (*b"Samr", 6), (*b"Saur", 7), (*b"Shaw", 6), (*b"Sinh", 7), (*b"Sund", 7), (*b"Sylo", 35), (*b"Syrc", 98), (*b"Tglg", 6), (*b"Tagb", 6), (*b"Tale", 6), (*b"Lana", 5), (*b"Tavt", 21), (*b"Taml", 7), (*b"Telu", 7), (*b"Thaa", 7), (*b"Thai", 21), (*b"Tibt", 7), (*b"Tfng", 7), (*b"Ugar", 12), (*b"Vaii", 6), (*b"Yiii", 12), (*b"Cakm", 7), (*b"Merc", 6), (*b"Mero", 14), (*b"Plrd", 5), (*b"Shrd", 7), (*b"Sora", 7), (*b"Takr", 7), (*b"Bass", 6), (*b"Aghb", 6), (*b"Dupl", 98), (*b"Elba", 6), (*b"Gran", 7), (*b"Khoj", 7), (*b"Sind", 7), (*b"Lina", 12), (*b"Mahj", 6), (*b"Mani", 98), (*b"Mend", 6), (*b"Modi", 35), (*b"Mroo", 6), (*b"Nbat", 6), (*b"Narb", 6), (*b"Perm", 6), (*b"Hmng", 7), (*b"Palm", 6), (*b"Pauc", 6), (*b"Phlp", 98), (*b"Sidd", 33), (*b"Tirh", 35), (*b"Wara", 6), (*b"Adlm", 34), (*b"Ahom", 6), (*b"Hluw", 14), (*b"Bhks", 35), (*b"Hatr", 6), (*b"Marc", 3), (*b"Mult", 3), (*b"Newa", 35), (*b"Hung", 6), (*b"Osge", 6), (*b"Sgnw", 6), (*b"Tang", 12), (*b"Gonm", 3), (*b"Nshu", 12), (*b"Soyo", 1), (*b"Zanb", 1), (*b"Dogr", 35), (*b"Gong", 35), (*b"Rohg", 98), (*b"Maka", 3), (*b"Medf", 6), (*b"Sogo", 2), (*b"Sogd", 98), (*b"Elym", 4), (*b"Hmnp", 3), (*b"Nand", 35), (*b"Wcho", 3), (*b"Chrs", 98), (*b"Diak", 3), (*b"Kits", 13), (*b"Yezi", 6), (*b"Cpmn", 12), (*b"Kawi", 5), (*b"Nagm", 6), (*b"Ougr", 98), (*b"Tnsa", 6), (*b"Toto", 6), (*b"Vith", 6)];

/// The script properties flags of `ch`'s own script: native looks a letter up by its character alone, so a
/// Common character (a digit, an emoji) is Common's and an Inherited one (a variation selector) Inherited's,
/// whatever its run (r25 30422: a VS16 cluster on its own keeps its width on a compressed line).
// ponytail: native also files two character classes (property values 13 and 23 of one UCD property) under
// Common whatever their script; find which property that is when a scene puts such a character on a
// justified line.
fn script_flags(ch: char) -> u8 {
    use unicode_script::UnicodeScript;
    let code = ch.script().short_name().as_bytes();
    SCRIPT_FLAGS.iter().find(|(iso, _)| iso.as_slice() == code).map_or(6, |&(_, f)| f)
}

/// Which glyphs of a visible run are blanks (space opportunities). Maps UTF-16 units to glyphs
/// via the cluster map; assumes BMP text (1 unit per char), which holds for the scenes here.
fn blank_glyphs(run: &GlyphRun) -> Vec<bool> {
    glyph_chars(run).iter().map(|c| c.is_some_and(char::is_whitespace)).collect()
}

/// The UTF-16 text position of the cluster holding glyph `g`: the first unit whose cluster starts at or before it.
fn cluster_unit(run: &GlyphRun, g: usize) -> u32 {
    let start = run.clusters.iter().map(|&c| c as usize).filter(|&c| c <= g).max().unwrap_or(0);
    run.text_pos + run.clusters.iter().position(|&c| c as usize == start).unwrap_or(0) as u32
}

/// Each glyph's character: the first character of the cluster it starts (`None` for a later glyph of a
/// cluster). Maps UTF-16 units to characters assuming BMP text, as `blank_glyphs` always has.
fn glyph_chars(run: &GlyphRun) -> Vec<Option<char>> {
    let chars: Vec<char> = run.text.chars().collect();
    let mut out = vec![None; run.glyphs.len()];
    let mut seen = vec![false; run.glyphs.len()];
    for (u, &g) in run.clusters.iter().enumerate() {
        let g = g as usize;
        if g < out.len() && !seen[g] {
            seen[g] = true;
            out[g] = chars.get(u).copied();
        }
    }
    out
}

/// Expand a wrapped line's visible glyphs so the content fills `layout_width`, reproducing
/// DirectWrite justified alignment. `runs` is the line's runs in draw order (visible content
/// runs followed by the trailing-whitespace run). Does nothing unless `align == 3` and not `skip`
/// (the caller's trimmed or non-justified last line). Returns the new `(visible_width, right_edge_including_trailing)`
/// when it justified, so the caller can keep its metrics, else `None`.
/// `space_em(text_pos)` is the design advance of the space glyph, in ems, of the face a run at `text_pos` uses.
pub fn justify_line(runs: &mut [Run], visible_runs: usize, align: u8, skip: bool, layout_width: f32, gdi_ppd: Option<f32>, space_em: &dyn Fn(u32) -> f32) -> Option<(f32, f32)> {
    if align != 3 || skip {
        return None;
    }
    let visible_glyph_runs = runs[..visible_runs.min(runs.len())].iter().filter(|r| matches!(r, Run::Glyph(_))).count();
    // ponytail: inline objects are left out of justification (no scene puts one on a justified
    // line); read native's handling when one does.
    let mut runs: Vec<&mut GlyphRun> = runs
        .iter_mut()
        .filter_map(|r| match r {
            Run::Glyph(g) => Some(g),
            Run::Inline(_) => None,
        })
        .collect();
    // The line's visible runs are justified; its trailing whitespace is not (DirectWrite leaves it at natural width,
    // past the fill). A leading space is justified with the rest, also where a right-to-left paragraph draws it at
    // the line's visual end (VM: ' lead' in a right-to-left paragraph compresses its space by half).
    if !runs[..visible_glyph_runs].iter().any(|r| r.text.chars().any(|c| !c.is_whitespace())) {
        return None;
    }
    let visible: Vec<usize> = (0..visible_glyph_runs).collect();

    // Build the opportunity list over visible glyphs in draw order.
    let mut opps: Vec<Opp> = Vec::new();
    // Only the text after the line's last tab is justified (the stretch before it is left alone; a line
    // ending in a tab justifies nothing).
    let after_tab = runs
        .iter()
        .filter_map(|r| r.text.encode_utf16().collect::<Vec<u16>>().iter().rposition(|&u| u == 0x09).map(|i| r.text_pos + i as u32 + 1))
        .max()
        .unwrap_or(0);
    for &ri in &visible {
        let blank = blank_glyphs(&runs[ri]);
        // Each run's opportunities scale with its own em (a per-range font size).
        let em = runs[ri].em;
        let run = &runs[ri];
        let chars = glyph_chars(run);
        for (g, &is_blank) in blank.iter().enumerate() {
            // Only a cluster's first glyph takes an opportunity: the rest of the cluster (a matra, a conjunct's
            // second glyph, an emoji ZWJ family's later glyphs) keeps its advance (r25 30513, 30627, 30858).
            let Some(u) = run.clusters.iter().position(|&c| c as usize == g) else { continue };
            let unit = run.text_pos + u as u32;
            if unit < after_tab {
                continue;
            }
            // A letter expands only in a discrete-writing script: by em/4 per edge for block writing (or
            // distributed within its cluster without word dividers), else em/8; a connected script's letter
            // (Devanagari, Arabic) does not (VM r25 30139: 'नमस' keeps its advances on a justified line).
            let flags = chars[g].map_or(6, script_flags);
            // A block-writing letter (or one distributed in its cluster without word dividers) also takes what
            // the maxima leave unfilled, as a space does (r25 30132: '日本語のテキスト' at em 0.70 fills its
            // 32-DIP line, 1.794 per letter edge).
            let block = flags & 8 != 0 || (flags & 16 != 0 && flags & 2 == 0);
            let letter = if flags & 4 == 0 {
                0.0
            } else if block {
                em * 0.25
            } else {
                em * 0.125
            };
            // Compression: a blank by half the face's design space advance up to em/4 (under GDI measuring too,
            // not the space's rounded width: r25 30966 compresses a Georgia space of 11 px by 5.068 DIPs, which
            // carries into the next letters), a discrete-script letter by em/32.
            opps.push(if is_blank {
                let half = em * space_em(run.text_pos) * 0.5;
                Opp { run: ri, glyph: g, unit, per_edge: em * 0.5, compress: half.min(em * 0.25), lead: false, trail: true, priority: 2, residual: true }
            } else {
                let compress = if flags & 4 == 0 { 0.0 } else { em * 0.03125 };
                // A block letter shares the spaces' priority (r25 30193: '한국어 텍스' grows 1, 2, 2 / 1 / 2, 1 px,
                // the deficit split over letters and space in proportion to their maxima).
                let block = block && flags & 4 != 0;
                Opp { run: ri, glyph: g, unit, per_edge: letter, compress, lead: true, trail: true, priority: if block { 2 } else { 3 }, residual: block }
            });
        }
    }
    if opps.is_empty() {
        return None;
    }
    // Native keeps the opportunities in logical order, so the pixel rounding below carries its error from one
    // character to the next in text order, not draw order (r25 30831: a left-to-right '12 3' inside a
    // right-to-left run grows 1, 1, 0 px, not 1, 0, 1).
    opps.sort_by_key(|o| o.unit);
    // The outer margins of the line get no adjustment. Native anchors the line's visual ends: from the visually
    // first run's first glyph it clears the leading edge (trailing in a right-to-left run) of that glyph and of
    // every glyph logically before it (after, right-to-left); from the visually last run's last glyph, the
    // trailing edge (leading) of it and every glyph logically after it (before). A single-direction line loses
    // just its two end edges; a mixed one far more (VM: 'ab אב' in a right-to-left paragraph compresses
    // a by 2, b, א and ב by 1 and its space not at all).
    let ends = |last: bool| {
        let order: Vec<usize> = if last { visible.iter().rev().copied().collect() } else { visible.clone() };
        order.into_iter().find_map(|ri| {
            let r = &runs[ri];
            let rtl = r.level & 1 == 1;
            let nonzero = |&g: &usize| r.advances[g] != 0.0 && cluster_unit(r, g) >= after_tab;
            // The visually first glyph is a left-to-right run's first, a right-to-left run's last.
            let g = if rtl != last { (0..r.glyphs.len()).rev().find(nonzero) } else { (0..r.glyphs.len()).find(nonzero) }?;
            Some((rtl, cluster_unit(r, g)))
        })
    };
    if let Some((rtl, u)) = ends(false) {
        for o in opps.iter_mut() {
            if !rtl && o.unit <= u {
                o.lead = false;
            } else if rtl && o.unit >= u {
                o.trail = false;
            }
        }
    }
    if let Some((rtl, u)) = ends(true) {
        for o in opps.iter_mut() {
            if rtl && o.unit <= u {
                o.lead = false;
            } else if !rtl && o.unit >= u {
                o.trail = false;
            }
        }
    }

    let natural: f32 = visible.iter().map(|&ri| runs[ri].advances.iter().sum::<f32>()).sum();
    let mut deficit = layout_width - natural;
    if deficit.abs() <= (natural / opps.len() as f32) * f32::EPSILON {
        return None;
    }
    let mut adjust = vec![0.0f32; opps.len()];
    if deficit < 0.0 {
        // A line too long compresses: every opportunity shares one compression priority, each giving up to
        // its compression maximum in proportion, with no residual (VM r25 30591: Courier New letters, 27.33
        // DIP each, give up em/32 per edge on a NO_WRAP justified line far wider than its box).
        let max_c: Vec<f32> = opps.iter().map(|o| o.compress * (o.lead as u32 + o.trail as u32) as f32).collect();
        let total: f32 = max_c.iter().sum();
        if total <= 0.0 {
            return None;
        }
        let scale = (-deficit / total).min(1.0);
        for i in 0..opps.len() {
            adjust[i] = -max_c[i] * scale;
        }
    }

    // Distribute the deficit by priority (highest priority = lowest number first), each
    // opportunity taking up to its max; a lower priority absorbs what remains. All Latin maxes
    // are finite, so `GetJustifiedGlyphAdvanceAdjustments`'s infinite-max branch is unused.
    let max_adj: Vec<f32> = opps.iter().map(Opp::max_adjust).collect();
    let max_priority = (0..opps.len()).filter(|&i| max_adj[i] > 0.0).map(|i| opps[i].priority).max().unwrap_or(0);

    for p in 1..=max_priority {
        if deficit <= f32::EPSILON {
            break;
        }
        let total: f32 = (0..opps.len()).filter(|&i| opps[i].priority == p && max_adj[i] > 0.0).map(|i| max_adj[i]).sum();
        if total <= 0.0 {
            continue;
        }
        let use_full = deficit > total;
        let scale = deficit / total;
        for i in 0..opps.len() {
            if opps[i].priority == p && max_adj[i] > 0.0 {
                let a = if use_full { max_adj[i] } else { max_adj[i] * scale };
                deficit -= a;
                adjust[i] += a;
            }
        }
    }
    // Residual: anything still unfilled spreads equally per residual-capable edge (only spaces).
    if deficit > f32::EPSILON {
        let edges: u32 = opps.iter().filter(|o| o.residual).map(|o| o.lead as u32 + o.trail as u32).sum();
        if edges > 0 {
            let per = deficit / edges as f32;
            for (i, o) in opps.iter().enumerate() {
                if o.residual {
                    adjust[i] += per * (o.lead as u32 + o.trail as u32) as f32;
                }
            }
        }
    }

    // Apply: the advance grows by the whole adjustment; a leading edge also offsets the glyph
    // right by its leading share (half for a two-edge opportunity, all of it otherwise). Under a GDI
    // measuring mode (`gdi_ppd`) each adjustment snaps to whole device pixels (half up) with the rounding
    // error carried into the next, and a two-edge leading half snaps too (DirectWrite's
    // ApplyGlyphAdvanceAdjustments).
    let snap = |v: f32, ppd: f32| {
        let x = v * ppd;
        if x.abs() >= 8388608.0 {
            return v;
        }
        let r = x.round_ties_even();
        (if r - x == -0.5 { r + 1.0 } else { r }) / ppd
    };
    let mut carry = 0.0f32;
    for i in 0..opps.len() {
        let mut a = adjust[i];
        if a == 0.0 {
            continue;
        }
        if let Some(ppd) = gdi_ppd {
            let q = snap(a + carry, ppd);
            carry += a - q;
            a = q;
        }
        let o = &opps[i];
        let run = &mut runs[o.run];
        run.advances[o.glyph] += a;
        if o.lead {
            let lead = match (o.trail, gdi_ppd) {
                (true, Some(ppd)) => snap(a * 0.5, ppd),
                (true, None) => a * 0.5,
                (false, _) => a,
            };
            run.offsets[o.glyph].0 += lead;
        }
    }

    // Lay the runs out again from the line start in draw order: the expanded content, then the trailing runs.
    let new_vis: f32 = visible.iter().map(|&ri| runs[ri].advances.iter().sum::<f32>()).sum();
    let mut pen = 0.0f32;
    for run in runs.iter_mut() {
        let w: f32 = run.advances.iter().sum();
        run.x = if run.level & 1 == 1 { pen + w } else { pen };
        pen += w;
    }
    Some((new_vis, pen))
}

// Scope / not yet reproduced (no scene exercises these, so left for when one does):
//  - Arabic kashida (JUSTIFY_ARABIC_*): native inserts kashida glyphs with a FLT_MAX, priority-1
//    opportunity; this handles only the finite Latin space/character opportunities.
//  - compression (content wider than the layout): native's negative-deficit path.
//  - GDI measuring: `ApplyGlyphAdvanceAdjustments` rounds each adjustment to whole pixels; only
//    natural measuring (no rounding) is reproduced.
//  - multi-glyph clusters and bidi/RTL visual reordering within a justified line (Latin scenes
//    here are 1:1 glyph/char, single LTR run).
