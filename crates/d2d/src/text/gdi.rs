//! GDI-compatible text measuring (`DWRITE_MEASURING_MODE` GDI_CLASSIC / GDI_NATURAL).
//!
//! Native's GDI-compatible layout (`DWriteFactory::CreateGdiCompatibleTextLayout`) measures with
//! GDI metrics instead of the ideal (natural) ones the default layout uses:
//!
//! * **Advances** are the TrueType-hinted whole-pixel glyph advance at the integer ppem (what GDI's
//!   `hdmx`/`LTSH` tables cache), reproduced here by the shared `hinting` interpreter, *plus* the
//!   shaper's GPOS advance adjustment (kerning). Both GDI modes shape in whole device pixels: the
//!   glyph's compatible advance (design units) scaled to pixels and rounded, plus each GPOS
//!   adjustment rounded to pixels at the integer ppem, the sum converted back to DIPs
//!   (`GetGdiCompatibleGlyphPlacements`).
//! * **Line ascent/descent** come from the font's `VDMX` table at that ppem, reproducing
//!   DirectWrite's GDI-compatible metrics: a binary search over the
//!   per-ppem records gives `yMax`/`yMin` in pixels, used in place of the `usWinAscent`/`usWinDescent`
//!   scaling the natural path applies. (The `lineGap` field is pixel-truncated and is 0 at these
//!   sizes, so the GDI baseline is `rect.top + yMax`.) A font with no `VDMX` entry falls back to the
//!   natural metrics — no GDI scene needs that path.
//!
//! Only the base face is measured in GDI mode (the GDI scenes are single-font). A GDI layout with
//! font fallback would need per-face VDMX/hinting; unexercised, so left to the natural path.

use super::hinting::{self, Config, Mode};
use super::shaping::Font;
use skrifa::raw::tables::hmtx::Hmtx;
use skrifa::raw::{types::Tag, FontRef, TableProvider};
use skrifa::GlyphId;

/// GDI-compatible measurer for one face at one em, `ppd` and layout transform: native's glyph metric
/// setup (see [`Metric`]), the `hmtx` table (to isolate the shaper's kern) and the resolved VDMX line metrics.
pub struct GdiMeasure<'a> {
    metric: Metric<'a>,
    hmtx: Hmtx<'a>,
    /// A variable face's instance (normalized coords; empty for a static face): its instanced design
    /// advance is what the shaper's GPOS adjustment is measured against.
    font: FontRef<'a>,
    coords: Vec<skrifa::raw::types::F2Dot14>,
    /// The layout's device grid, `ppd` times the transform's y scale `|(m21, m22)|` when that is not 1.
    ppd: f32,
    /// The line box's device ppem: `round(em·ppd)` along the transform's y axis.
    y_ppem: i32,
    upem: i32,
    /// The shaping units: device pixels per design unit at the actual (fractional) em,
    /// `em·ppd / upem`, and the integer ppem GPOS values are scaled at, `round(upem · px_per_du)`.
    /// So a fractional em (36.5 ⇒ device ppem 37) scales the 37-ppem advance back down
    /// (38 px ⇒ 1051 du ⇒ 37.48 ⇒ 37 px).
    px_per_du: f32,
    otl_ppem: i32,
    /// DIPs per shaping unit, `em / (px_per_du · upem)` (≈ 1/ppd).
    dip_per_px: f32,
    /// VDMX ascent/descent in DIPs, or `None` when the font has no VDMX entry (use natural metrics).
    vmetrics: Option<(f32, f32)>,
    /// Native's text layout rounds the glyph advance to the pixel grid only when it finds the layout
    /// transform a uniform axis scale or quarter turn; any other
    /// transform (a shear, a non-uniform scale) keeps the design-unit GDI advance as is (VM, Arial 20
    /// 'Hamburg' under [1,0,.5,1]: 1331 du = 12.998 DIP).
    fit: bool,
}

impl<'a> GdiMeasure<'a> {
    /// Prepare GDI measuring for `font` at `em` DIPs and `ppd` (`dpi/96`). `classic` selects
    /// GDI_CLASSIC vs GDI_NATURAL; `xf` is the layout transform `[m11, m12, m21, m22]` (see [`gdi_config`]).
    /// `None` only if the font lacks the TrueType scaler tables.
    pub fn new(font: &'a Font, em: f32, ppd: f32, classic: bool, xf: Option<[f32; 4]>) -> Option<Self> {
        // The whole font file, straight from ttf_parser's parsed record (no copy), at the font's own face
        // index (a fallback face can sit in a collection: Yu Gothic UI is face 1 of YuGothM.ttc).
        let face = font.tt();
        let data = face.raw_face().data;
        let fr = FontRef::from_index(data, font.index()).ok()?;
        // A transform that scales y by `s != 1` runs the layout on the device grid `ppd·s`.
        let s = xf.map_or(1.0, |m| (m[2] * m[2] + m[3] * m[3]).sqrt());
        let grid_ppd = if s.is_finite() && s > 0.0 && (s - 1.0).abs() > 1e-6 { ppd * s } else { ppd };
        // The line box measures at the per-axis y scale (native's scaler takes max(|m21|, |m22|)), smaller under
        // a shear ([1,0,.5,1]: 20 px, not round(20·1.118) = 22). A generic matrix keeps design advances.
        let len = |m: [f32; 4]| (m[2] * m[2] + m[3] * m[3]).sqrt();
        let y = xf.filter(|&m| m.iter().any(|&v| fixed_zero(v)) && len(m) > 0.0).map_or(1.0, |m| m[2].abs().max(m[3].abs()) / len(m));
        let y_ppem = (em as f64 * grid_ppd as f64 * y as f64).round() as i32;
        // A variable face measures at its instance (r25 30082: Bahnschrift Bold Condensed 'A', 906 units, is
        // 15 px at ppem 35, not the default instance's 23 px less a 7 px adjustment).
        let coords = font.coords();
        let metric = Metric::new(fr.clone(), &coords, em, ppd, classic, xf)?;
        let vmetrics = vdmx_vmetrics(&fr, y_ppem, grid_ppd);
        let hmtx = fr.hmtx().ok()?;
        let upem = fr.head().ok()?.units_per_em() as i32;
        let px_per_du = em * grid_ppd / upem as f32;
        let otl_ppem = round_half_up(upem as f32 * px_per_du);
        let dip_per_px = em / (px_per_du * upem as f32);
        let fit = xf.is_none_or(transform_scale_factor);
        Some(Self { metric, hmtx, font: fr, coords, ppd: grid_ppd, y_ppem, upem, px_per_du, otl_ppem, dip_per_px, vmetrics, fit })
    }

    /// GDI advance of a shaped glyph in DIPs, in shaping units: the glyph's compatible advance
    /// ([`Metric::advance_du`]) scaled to pixels at the actual em and
    /// rounded, plus the shaper's GPOS advance adjustment (`shaped_advance_du − hmtx advance`) rounded
    /// to pixels at the integer ppem, converted back to DIPs. `bold` requests the synthetic-bold +1 device px
    /// (GetGdiCompatibleGlyphMetrics'), which native's GDI metric adds to every glyph with a nonzero advance, the
    /// space included (VM: Impact w600 GDI space advance 473 vs 394 du), unlike the design metric, which widens
    /// only inked glyphs; a zero-advance glyph (a combining mark) is left alone.
    pub fn advance(&self, glyph_id: u16, shaped_advance_du: i32, _scale: f32, bold: bool) -> f32 {
        // A device em that rounds to zero lays every glyph out at zero, the bold pixel too (r25 30247: Malgun
        // Gothic's bold-simulated space at em 0.001 measured 1048 DIP through the 1/ppem conversion).
        if self.metric.device_em == 0.0 {
            return 0.0;
        }
        let du = self.metric.advance_du(glyph_id, bold);
        let raw_du = if self.coords.is_empty() {
            self.hmtx.advance(GlyphId::new(glyph_id as u32)).unwrap_or(0) as i32
        } else {
            use skrifa::MetadataProvider;
            let gm = self.font.glyph_metrics(skrifa::instance::Size::unscaled(), skrifa::instance::LocationRef::new(&self.coords));
            gm.advance_width(GlyphId::new(glyph_id as u32)).unwrap_or(0.0).round() as i32
        };
        if !self.fit {
            return (du + shaped_advance_du - raw_du) as f32 * self.px_per_du * self.dip_per_px;
        }
        // ponytail: the shaper reports one summed adjustment, so two GPOS values on one glyph round
        // once together where native rounds each; split per lookup if a font stacks them.
        let px = round_half_up(du as f32 * self.px_per_du) + otl_design_to_px(shaped_advance_du - raw_du, self.otl_ppem, self.upem);
        px as f32 * self.dip_per_px
    }

    /// GDI vertical advance in DIPs of a design-unit advance (an upright glyph in vertical text). As
    /// [`Self::advance`]: whole pixels at the device ppem, back to design units, then whole pixels at the
    /// actual em (VM r25: Yu Gothic UI 2048/2048 at em 3.82 → 4 px). `bold_px` is the bold simulation's pixel.
    pub fn vadvance(&self, advance_du: i32, bold_px: f32) -> f32 {
        let device_em = self.metric.device_em;
        if device_em <= 0.0 {
            return 0.0;
        }
        let upem = self.upem as f64;
        let du = (((advance_du as f64 * device_em / upem).round() + bold_px as f64) * upem / device_em).round();
        round_half_up(du as f32 * self.px_per_du) as f32 * self.dip_per_px
    }

    /// A positioned glyph's x offset in DIPs. A GDI layout shapes in whole device pixels, so the glyph lands at its
    /// design pen plus placement (`pos_du`) scaled at the OTL ppem, measured from the run's whole-pixel pen
    /// (VM r25 30153: Segoe UI Emoji's family sequence places its first glyph at 9 px, its 425-unit placement at
    /// ppem 42). `None` when the layout transform keeps design-unit advances.
    pub fn offset(&self, pos_du: i32, pen: f32) -> Option<f32> {
        self.fit.then(|| (otl_design_to_px(pos_du, self.otl_ppem, self.upem) - round_half_up(pen / self.dip_per_px)) as f32 * self.dip_per_px)
    }

    /// A DIP length in whole pixels, half up. `None` when the layout transform keeps design-unit advances.
    pub fn whole(&self, dip: f32) -> Option<f32> {
        self.fit.then(|| round_half_up(dip / self.dip_per_px) as f32 * self.dip_per_px)
    }

    /// A positioned glyph's y offset in DIPs: the shaper's placement in whole pixels at the OTL ppem (VM r25: Nirmala
    /// UI's u-matra under GDI measuring sits whole pixels below its consonant at 1, 1.25, 1.5 and 2 pixels per DIP).
    /// `None` when the layout transform keeps design-unit advances.
    pub fn offset_y(&self, dy_du: i32) -> Option<f32> {
        self.fit.then(|| otl_design_to_px(dy_du, self.otl_ppem, self.upem) as f32 * self.dip_per_px)
    }

    /// A mark's x offset in DIPs when it hangs on a base glyph by a GPOS mark anchor. Shaping in whole pixels, the
    /// base's anchor, the mark's anchor and the mark's remaining placement (`rest_du`) each round on their own at the
    /// OTL ppem; the mark adds the base's own offset (`base`) and steps back over the advances between the two
    /// (`span`). VM r25 30153: Segoe UI Emoji's family sequence hangs its last glyph on the second at anchor 504 with
    /// a −1009 placement, 10 − 21 − 25 = −36 px at ppem 42 where one rounding of the design position gives −37.
    /// A right-to-left offset points left (`rtl`), its `base` the base's offset plus own advance (VM r25 30703:
    /// 43 + 3 − 46 − (20 − 39) = 19 px at ppem 80, bold-simulated). `None` when the layout keeps design advances.
    pub fn mark_offset(&self, (base_anchor, mark_anchor): (i32, i32), rest_du: i32, rtl: bool, base: f32, span: f32) -> Option<f32> {
        let px = |v| otl_design_to_px(v, self.otl_ppem, self.upem);
        let whole = |v: f32| round_half_up(v / self.dip_per_px);
        let attach = px(base_anchor) - px(mark_anchor) + px(rest_du);
        self.fit.then(|| (if rtl { -attach } else { attach } + whole(base) - whole(span)) as f32 * self.dip_per_px)
    }

    /// GDI line `(ascent, descent)` in DIPs, or `None` when the font has no VDMX entry.
    pub fn vmetrics(&self) -> Option<(f32, f32)> {
        self.vmetrics
    }

    /// Device ppem (`round(em·ppd)`) — the grid the GDI line box rounds to when VDMX misses.
    pub fn ppem(&self) -> i32 {
        self.y_ppem
    }

    /// `ppd` (`dpi/96`), to convert device-pixel GDI metrics back to DIPs.
    pub fn ppd(&self) -> f32 {
        self.ppd
    }
}

/// Native's GDI-compatible glyph metric setup (`GetGdiCompatibleGlyphMetrics`) for one face at `em·ppd`
/// under a transform: the 16.16 rasterizer transform, the hinting instance at the scaler's ppem, and the
/// float inverse the hinted advance is mapped back to em units through.
struct Metric<'a> {
    inst: hinting::Instance<'a>,
    font: FontRef<'a>,
    upem: i32,
    /// The hinting ppem (0 lays every glyph out at zero).
    device_em: f64,
    /// |t00| of the 16.16 rasterizer transform: the compatible-width path's scale.
    t00: i32,
    /// The inverse transform's entry along the advance (display → em units), as native's f32.
    inv: f32,
    /// The transform is axis-aligned: the display advance rounds to whole pixels first.
    axis: bool,
    /// The advance runs leftward on the display (t00 < 0, a mirror): its whole-pixel round, `floor(v + 0.5)` on the
    /// negative display value, takes a half-pixel advance down where the upright one goes up.
    mirror: bool,
    /// The compatible-width ppem: GDI classic under an axis-aligned uniform scale to a ppem of at most 255
    /// on a static face, where the scaler takes `hdmx` / linear widths ahead of the hinted advance.
    compat: Option<u8>,
    classic: bool,
}

impl<'a> Metric<'a> {
    fn new(font: FontRef<'a>, coords: &[skrifa::raw::types::F2Dot14], em: f32, ppd: f32, classic: bool, xf: Option<[f32; 4]>) -> Option<Self> {
        let (t, ppem) = metric_transform(em, ppd, xf);
        // The scaler hints at the integer ppem of a uniform scale, else at the x axis' scale (VM: Courier New
        // and Arial at em 29.3 under [1,0,0,2] and [1,0,k,1] advance by their ppem-29 hinted widths).
        let device_em = match (ppem, xf) {
            (Some(p), _) => p as f64,
            (None, Some(m)) => ((ppd * em) as f64 * m[0].abs().max(m[1].abs()) as f64).round(),
            (None, None) => unreachable!(),
        };
        // A non-uniform scale hints each axis at its own whole-pixel scale (y: max(|m21|, |m22|)), the glyph
        // flagged stretched.
        let em_y = match (ppem, xf) {
            (None, Some(m)) => Some(((ppd * em) as f64 * m[2].abs().max(m[3].abs()) as f64).round()).filter(|&y| y != device_em),
            _ => None,
        };
        let config = Config { em_y, is_stretched: em_y.is_some(), ..gdi_config(classic, xf) };
        let inst = hinting::Instance::new_instanced(&font, device_em, config, coords)?;
        let upem = font.head().ok()?.units_per_em() as i32;
        // The float inverse of the 16.16 transform: `(1/det)·t11`, `−t01·(1/det)`, `−t10·(1/det)`.
        let f = |v: i32| v as f32 * (1.0 / 65536.0);
        let (a, b, c, d) = (f(t[0]), f(t[1]), f(t[2]), f(t[3]));
        let det = d * a - b * c;
        let (inv0, inv1, inv2) = if det != 0.0 { let r = 1.0 / det; (r * d, -b * r, -c * r) } else { (0.0, 0.0, 0.0) };
        // The advance runs along display x, or display y under a quarter turn (t00 = 0).
        let inv = if t[0] == 0 { inv1 } else { inv0 }.abs();
        let axis = inv1 == 0.0 && inv2 == 0.0;
        let compat = ppem.filter(|&p| classic && axis && (1..=255).contains(&p) && font.table_data(Tag::new(b"fvar")).is_none()).map(|p| p as u8);
        Some(Self { inst, font, upem, device_em, t00: t[0].wrapping_abs(), inv, axis, mirror: t[0] < 0, compat, classic })
    }

    /// The GDI-compatible advance of `gid` in design units: the em fraction (16.16) of the scaler's width
    /// scaled by upem and rounded — the compatible width when the scaler has one, else the hinted advance
    /// mapped back through the float inverse transform (rounded to whole pixels first when axis-aligned) and
    /// rounded to 16.16 (so 33 px at ppem 59 is 1146 units, not 1145.49 → 1145). `bold` adds the bold
    /// simulation's pixel to a glyph with a design advance, even one its hinting takes to zero pixels (VM r25
    /// 30622: Impact's bold-simulated space at ppem 2 advances 1 px); a zero-advance mark keeps zero.
    fn advance_du(&self, gid: u16, bold: bool) -> i32 {
        if self.device_em == 0.0 {
            return 0;
        }
        let bold = bold && self.font.hmtx().ok().and_then(|h| h.advance(GlyphId::new(gid as u32))).unwrap_or(0) != 0;
        let em = match self.compat_width(gid) {
            Some(w) => {
                let w = if bold { w + 65536 } else { w };
                fixdiv((w + 0x8000) & !0xFFFF, self.t00)
            }
            None => {
                let x = self.inst.hint(gid).map_or(0.0, |h| h.advance_px());
                match self.overscaled_wrap(gid, x + if bold { 1.0 } else { 0.0 }) {
                    Some(wrapped) => float_to_fixed(wrapped * self.inv),
                    None => {
                        // VM layout dump, Tahoma at em 13 GDI natural under [-1,0,0,1]: 'y', hinted to 6.5 px, advances
                        // 6 px mirrored and 7 upright.
                        let x = match (self.axis, self.mirror) {
                            (true, false) => (x + 0.5).floor(),
                            (true, true) => -(0.5 - x).floor(),
                            (false, _) => x,
                        };
                        float_to_fixed((x + if bold { 1.0 } else { 0.0 }) * self.inv)
                    }
                }
            }
        };
        ((em as i64 * self.upem as i64 + 0x8000) >> 16) as i32
    }

    /// GDI classic measures an advance in the scaler's six-times x-overscaled space, where the advance's 16.16
    /// form (the 26.6 advance shifted left by 10) wraps past 32 bits from 32768/6 px; divided back by six, the
    /// wrapped value is the advance native reports, unless the glyph's own six-times extent overflows the
    /// bitmap bounds first, which keeps the plain advance (dwrite reading; VM probes, Comic Sans 'M', 'N', 'A',
    /// 'W' and 'o' at ppem 3750 to 12460: 236 of 237 exact, r25 30198 'M' at ppem 6250: −5405 px, not 5518).
    /// The wrapped advance in pixels, or `None` when it does not wrap.
    /// ponytail: one probe ('o' at ppem 10421.2) reports as ppem 10422 would; GDI natural is unprobed.
    fn overscaled_wrap(&self, gid: u16, advance_px: f32) -> Option<f32> {
        if !self.classic || !self.axis {
            return None;
        }
        let a26 = (advance_px as f64 * 384.0).round() as i64;
        let v = (a26 << 10) as i32;
        if v >= 0 {
            return None;
        }
        let x_max = skrifa::MetadataProvider::glyph_metrics(&self.font, skrifa::instance::Size::unscaled(), skrifa::instance::LocationRef::default())
            .bounds(GlyphId::new(gid as u32))
            .map_or(0.0, |b| b.x_max as f64);
        if x_max * self.device_em * 6.0 / self.upem as f64 >= 32768.0 {
            return None;
        }
        Some(((v as i64 + 3) / 6) as f32 / 65536.0)
    }

    /// The scaler's compatible width of `gid` (16.16 px) at the compatible-width ppem: the `hdmx` record's
    /// width, else — unless the head flags say instructions may depend on the size or alter the advance and
    /// `LTSH` does not vouch for linear scaling at this ppem — the linear width `(|t00| / upem)·advance`
    /// (VM: Consolas, whose head flags clear both bits, advances by its rounded linear width at every ppem).
    fn compat_width(&self, gid: u16) -> Option<i32> {
        let ppem = self.compat?;
        let g = gid as usize;
        if let Some(rec) = hdmx_record(&self.font, ppem) {
            return rec.get(2 + g).map(|&w| (w as i32) << 16);
        }
        let flags = self.font.head().ok()?.flags().bits();
        if flags & 0x14 != 0 {
            let ltsh = self.font.table_data(Tag::new(b"LTSH"))?;
            if ppem < *ltsh.as_bytes().get(4 + g)? {
                return None;
            }
        }
        let aw = self.font.hmtx().ok()?.advance(GlyphId::new(gid as u32)).unwrap_or(0) as i32;
        Some((self.t00 / self.upem).wrapping_mul(aw))
    }
}

/// The `hdmx` device record for `ppem` (pixelSize, maxWidth, then one width per glyph), as native's
/// lookup walks the size-sorted records of a version-0 table.
fn hdmx_record<'b>(font: &FontRef<'b>, ppem: u8) -> Option<&'b [u8]> {
    let b = font.table_data(Tag::new(b"hdmx"))?.as_bytes();
    if b.len() < 8 || u16::from_be_bytes([b[0], b[1]]) != 0 {
        return None;
    }
    let n = u16::from_be_bytes([b[2], b[3]]) as usize;
    let size = u32::from_be_bytes([b[4], b[5], b[6], b[7]]) as usize;
    (0..n).map(|i| 8 + i * size).take_while(|&o| b.get(o).is_some_and(|&p| p <= ppem)).find(|&o| b[o] == ppem).and_then(|o| b.get(o..o + size))
}

/// Native's 16.16 rasterizer transform for GDI-compatible metrics (`[t00, t01, t10, t11]`, y flipped) at
/// `em·ppd` under `xf` (identity when `None`): a uniform scale or quarter turn snaps to its integer ppem
/// (rounded half up, also returned), anything else keeps the fractional scale.
fn metric_transform(em: f32, ppd: f32, xf: Option<[f32; 4]>) -> ([i32; 4], Option<i32>) {
    let half_up = |v: f32| { let n = v.round_ties_even(); (if n - v == -0.5 { n + 1.0 } else { n }) as i32 };
    let e = ppd * em;
    let Some(m) = xf else {
        let p = half_up(e);
        return ([p << 16, 0, 0, p << 16], Some(p));
    };
    const TOL: f32 = 1.0 / 65536.0;
    let mut v = [m[0] * e, m[1] * e, m[2] * e, m[3] * e];
    let zero = |x: f32| x.abs() < TOL;
    let pair = if !zero(v[1]) { (zero(v[0]) && zero(v[3])).then_some((v[1], v[2])) } else { zero(v[2]).then_some((v[0], v[3])) };
    let mut ppem = None;
    if let Some((a, _)) = pair.filter(|&(a, b)| zero(a.abs() - b.abs())) {
        let a = a.abs();
        let p = half_up(a);
        ppem = Some(p);
        if !zero(a) {
            let s = p as f32 / a;
            v = v.map(|x| x * s);
        }
    }
    ([float_to_fixed(v[0]), float_to_fixed(-v[2]), float_to_fixed(-v[1]), float_to_fixed(v[3])], ppem)
}

/// Native's float → 16.16 conversion: `(trunc(v·2^32) + 0x8000) >> 16`, saturated.
fn float_to_fixed(v: f32) -> i32 {
    ((((v * 65536.0) * 65536.0) as i64).saturating_add(0x8000) >> 16).clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// 16.16 divide, rounded half away from zero.
fn fixdiv(a: i32, b: i32) -> i32 {
    if b == 0 {
        return i32::MAX;
    }
    let (a, b) = ((a as i64) << 16, b as i64);
    let h = b / 2;
    let q = if (a < 0) == (b < 0) { (a + h) / b } else { (a - h) / b };
    q.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// Native's layout scale test: `[m11, m12, m21, m22]` is a uniform axis scale or a uniform quarter turn,
/// within 2^-16 (`IsZeroWithTolerance`).
/// A transform component the scaler sees as zero: its 16.16 fixed-point TransMatrix rounds it to 0 (a float
/// quarter turn's cos 90° = −4.37e−8 is zero there; r25 30223, 30477, 30944 lay out on whole pixels).
pub(crate) fn fixed_zero(v: f32) -> bool {
    (v * 65536.0).round() == 0.0
}

fn transform_scale_factor(m: [f32; 4]) -> bool {
    const T: f32 = 1.0 / 65536.0;
    let z = |v: f32| v.abs() < T;
    let (a, b) = if !z(m[1]) { (m[1], m[2]) } else { (m[0], m[3]) };
    let off = if !z(m[1]) { z(m[0]) && z(m[3]) } else { z(m[2]) };
    off && z(a.abs() - b.abs())
}

/// Round to the nearest integer, a tie going up (the shaping client's float → shaping-unit rounding).
pub fn round_half_up(v: f32) -> i32 {
    let n = v.round_ties_even();
    (if n - v == -0.5 { n + 1.0 } else { n }) as i32
}

/// A GPOS design-unit value in whole pixels at `ppem`: `value·ppem / upem` rounded, a tie going up.
fn otl_design_to_px(value: i32, ppem: i32, upem: i32) -> i32 {
    if upem == 0 {
        return value;
    }
    let half = if value >= 0 { upem >> 1 } else { 1 - (upem >> 1) };
    (half + value * ppem) / upem
}

/// The hinting config for GDI measuring: whole-pixel glyph advances (metric pass), no render phantom.
/// The GDI metric pass under the transform `xf` (`[m11, m12, m21, m22]`), as native's transform setup
/// derives its pass flags from the transform matrix: compatible widths need t01 = m12 = 0, so a tilted baseline gives
/// GDI-classic GDI-natural's single pass; and m11 = 0 toggles vertical LCD, which hints the glyph's x on whole
/// pixels (VM, Arial 20 'Hamburg' under [0,1,-1,0]: [14,11,16,11,11,7,11], one pass with flags 0x85).
fn gdi_config(classic: bool, xf: Option<[f32; 4]>) -> Config {
    let classic = classic && xf.is_none_or(|m| fixed_zero(m[1]));
    Config {
        rasterizer: hinting::Rasterizer::FontCache,
        mode: if classic { Mode::GdiClassic } else { Mode::GdiNatural },
        symmetric: false,
        bgr: false,
        is_rotated: false,
        is_stretched: false,
        axes_swapped: xf.is_some_and(|m| fixed_zero(m[0])),
        // Advance measurement reads the phantom points, not composite component offsets, so the component
        // offset's rounding regime does not reach the output here.
        rotation_kind: 0,
        em_y: None,
        version: 42,
        gdi_render: false,
        cleartype: false,
        // GDI-natural's metric pass starts the advance phantom on the 1/16-px subpixel grid (whole when the
        // glyph program is inhibited) and rounds the hinted advance to whole pixels afterwards
        // (`Instance::hint`): a design advance whose 26.6 fraction reaches 30/64 rounds up. GDI-classic
        // keeps its whole-pixel phantom.
        subpixel_advance: !classic,
        grid_fit: true, // GDI-compatible advances always grid-fit
        bold_fu: 0,
    }
}

/// GDI-compatible glyph advance in **design units** (`IDWriteFontFace::GetGdiCompatibleGlyphMetrics`
/// advanceWidth / `IDWriteFontFace1::GetGdiCompatibleGlyphAdvances`) at `em`, `ppd` and transform `xf`
/// ([`Metric::advance_du`]); a transform with no zero component keeps the design advance. `classic` is
/// `!useGdiNatural`. `bold` requests the bold simulation: one device pixel on every glyph with a nonzero advance,
/// the space included (VM: the GDI advance is exactly reg+1 px at every em 16..96 for Impact, its space 473 vs
/// 394 du), or round(upem/50) design units on an inked glyph where the design advance stands. `None` when the
/// font lacks the scaler tables.
pub fn gdi_compat_advance_du(fr: &FontRef, gid: u16, em: f32, ppd: f32, classic: bool, xf: Option<[f32; 4]>, upem: u16, bold: bool) -> Option<i32> {
    let (t, _) = metric_transform(em, ppd, xf);
    if t.iter().all(|&v| v != 0) {
        // Design metrics, which bold simulation widens by upem/50 (VM: Impact bold-simulated under a 30° turn).
        use skrifa::MetadataProvider;
        let inked = bold && fr.glyph_metrics(skrifa::instance::Size::unscaled(), skrifa::instance::LocationRef::default()).bounds(GlyphId::new(gid as u32)).is_some();
        let bold = if inked { (upem as f32 / 50.0).round() as i32 } else { 0 };
        return Some(fr.hmtx().ok()?.advance(GlyphId::new(gid as u32)).unwrap_or(0) as i32 + bold);
    }
    Some(Metric::new(fr.clone(), &[], em, ppd, classic, xf)?.advance_du(gid, bold))
}

/// GDI-compatible design metrics for `IDWriteFontFace::GetGdiCompatibleMetrics`: the VDMX
/// `(yMax, yMin)` pixel record at `ppem` scaled back to design units (both positive). `None` when the
/// font has no VDMX record at `ppem`. ponytail: round-to-nearest matches the VM's only pinned size
/// (ppem 16, where it is exact); confirm the rounding against a non-dividing ppem if a scene needs one.
pub fn gdi_compat_vmetrics(fr: &FontRef, dev: f32, upem: u16) -> Option<(u16, u16)> {
    if !(dev > 0.0 && dev < 255.0) {
        return None;
    }
    let (ymax, ymin) = vdmx_lookup(fr.table_data(Tag::new(b"VDMX"))?.as_bytes(), round_half_up(dev))?;
    // Back to design units on the float device em, not the integer ppem that keys the record (VM: Segoe UI at
    // device em 40.19 reports 2191 / 561 from 43 / 11 px).
    let du = |px: i32| round_half_up(px as f32 * (upem as f32 / dev)) as u16;
    Some((du(ymax as i32), du(-(ymin as i32))))
}


/// VDMX line `(ascent, descent)` in DIPs at `ppem`: `yMax` / `-yMin` from the per-ppem record,
/// divided back to DIPs. `None` when the font has no VDMX table or no record at this ppem.
///
/// ponytail: the ratio group is chosen as the first that carries `ppem` — correct for the 1:1
/// (square-pixel) aspect every GDI scene uses; a non-square DPI/transform would need the aspect
/// match DirectWrite's GDI-compatible metrics do.
fn vdmx_vmetrics(fr: &FontRef, ppem: i32, ppd: f32) -> Option<(f32, f32)> {
    let (ymax, ymin) = vdmx_lookup(fr.table_data(Tag::new(b"VDMX"))?.as_bytes(), ppem)?;
    let r = 1.0 / ppd;
    Some((ymax as f32 * r, -(ymin as f32) * r))
}

/// Find the `(yMax, yMin)` pixel record for `ppem` in a `VDMX` table.
pub(crate) fn vdmx_lookup(b: &[u8], ppem: i32) -> Option<(i16, i16)> {
    let rd16 = |o: usize| -> Option<u16> { Some(u16::from_be_bytes([*b.get(o)?, *b.get(o + 1)?])) };
    let num_ratios = rd16(4)? as usize; // header: version, numRecs, numRatios
    let offs = 6 + num_ratios * 4; // the group offsets follow the 4-byte ratio records
    // The first ratio record for a square pixel (`yStartRatio <= xRatio <= yEndRatio`, or the 0:0 default)
    // names the group, and only a Windows ANSI record (bCharSet 1) counts: native lays Tahoma, whose one
    // record has bCharSet 0, out on its design metrics (VM r25 30441: 24 + 5 px at ppem 24, not VDMX 26 + 6).
    let i = (0..num_ratios).find(|&i| {
        let r = |k: usize| b.get(6 + 4 * i + k).copied();
        let (Some(cs), Some(x), Some(y0), Some(y1)) = (r(0), r(1), r(2), r(3)) else { return false };
        cs == 1 && ((x == 0 && y0 == 0 && y1 == 0) || (y0 <= x && x <= y1))
    })?;
    vdmx_group(b, rd16(offs + 2 * i)? as usize, ppem)
}

/// Scan one VDMX group (`recs`, startSize, endSize, then `recs` × {yPelHeight, yMax, yMin}) for
/// the record whose pel height equals `ppem`.
fn vdmx_group(b: &[u8], go: usize, ppem: i32) -> Option<(i16, i16)> {
    let recs = u16::from_be_bytes([*b.get(go)?, *b.get(go + 1)?]) as usize;
    (0..recs).find_map(|r| {
        let q = go + 4 + r * 6;
        let y = u16::from_be_bytes([*b.get(q)?, *b.get(q + 1)?]) as i32;
        (y == ppem).then(|| (i16::from_be_bytes([b[q + 2], b[q + 3]]), i16::from_be_bytes([b[q + 4], b[q + 5]])))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-group VDMX table with two records (ppem 13 → (13,-3), 21 → (19,-5), mirroring Arial)
    /// exercises the header skip, group offset and per-record scan.
    #[test]
    fn vdmx_parse() {
        #[rustfmt::skip]
        let v: Vec<u8> = vec![
            0,0, 0,1, 0,1,          // version, numRecs, numRatios
            1,1,1,1,                // one ratio record
            0,12,                   // offset to the group (header 6 + ratio 4 + offset 2 = 12)
            0,2, 8,30,              // group: recs=2, startSize, endSize
            0,13, 0,13, 255,253,    // ppem 13 → yMax 13, yMin -3
            0,21, 0,19, 255,251,    // ppem 21 → yMax 19, yMin -5
        ];
        assert_eq!(vdmx_lookup(&v, 13), Some((13, -3)));
        assert_eq!(vdmx_lookup(&v, 21), Some((19, -5)));
        assert_eq!(vdmx_lookup(&v, 9), None);
    }

    /// The metric transform snaps a uniform scale to its ppem and keeps a shear fractional; the hinted
    /// advance maps back through the f32 inverse and 16.16 em units (VM: 33 px at ppem 59 is 1146 units).
    #[test]
    fn metric_transform_and_em_rounding() {
        let (t, p) = metric_transform(29.3, 1.0, Some([2.0, 0.0, 0.0, 2.0]));
        assert_eq!((t, p), ([59 << 16, 0, 0, 59 << 16], Some(59)));
        let (t, p) = metric_transform(29.3, 1.0, Some([1.0, 0.0, -0.723, 1.0]));
        assert_eq!((t[0], t[2], p), (float_to_fixed(29.3), 0, None));
        let inv = (1.0 / (59.0f32 * 59.0)) * 59.0;
        assert_eq!((float_to_fixed(33.0 * inv) as i64 * 2048 + 0x8000) >> 16, 1146);
        assert_eq!(fixdiv(5 << 16, 10 << 16), 0x8000);
    }

}

