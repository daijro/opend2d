//! Phase 2D.1 glyph-run rendering: the pixels of one positioned `DWRITE_GLYPH_RUN`, exactly as
//! DirectWrite's rasterizer and Direct2D's `DrawGlyphRun` produce them. Layout, shaping, fallback
//! and metrics (the `DrawText`/`DrawTextLayout` front end) are phase 2D.2 (`text::layout` /
//! `text::shaping`, another owner); this module is the back end they call once glyphs are placed.
//!
//! Entry point: [`render_glyph_run`]. It consumes a [`GlyphRun`] (font bytes + positioned glyph ids)
//! and paints into a premultiplied BGRA8 [`PixelBuffer`] under the device transform.
//!
//! ## Rendering-mode selection (R2) — reproduced from native
//!
//! DirectWrite's recommended rendering mode. With
//! `DEFAULT` params and no explicit mode, the device em-pixel size decides outline vs analysis:
//! the outline cutoff is **100 px for an antialiased threshold** (grayscale/ClearType) and **350 px
//! for an aliased threshold** — not the ~100 px the text spec's R6 guessed. Below the cutoff the
//! glyph is grid-fitted and scan-converted by DirectWrite (NATURAL / NATURAL_SYMMETRIC / GDI / …);
//! at or above it the glyph is filled as geometry (R6, `OUTLINE`).
//!
//! ## Rendering-mode dispatch (what [`render_analysis`] wires)
//!
//! `render_analysis` resolves the text AA mode for the target, produces the coverage that mode
//! needs and composites it with the matching blend. Every `D2D1_TEXT_ANTIALIAS_MODE` × measuring
//! mode × grid-fit combination is wired:
//!
//! * **Grayscale (R1/R3):** 4× coverage (`raster::grayscale`/`coverage`) → `blend::blend_grayscale`.
//! * **ClearType (R4):** 3×1 coverage (`raster::cleartype`/`coverage`) → `blend::blend_cleartype`
//!   (dual-source). Byte-exact end-to-end on the grid-fit-disabled isolation scenes
//!   (`textmodes-01` black, `textmodes-03` coloured foreground).
//! * **Aliased (R5):** 1× binary coverage (`raster::aliased`/`coverage`) → `blend_grayscale` with the
//!   aliased branch. Byte-exact on the grid-fit-disabled isolation scene (`textmodes-02`): bi-level
//!   adds native's X-on-centre topology (a vertical edge on a pixel-centre column covers it), which the
//!   oversampled paths don't run (see `raster::resolve_aliased`).
//! * **GDI measuring (CLEARTYPE_GDI_CLASSIC/_NATURAL):** the `hinting` interpreter runs the
//!   GDI-compatible grid-fit for the measuring mode; coverage/blend as above.
//! * **OUTLINE mode (R6):** the glyph outlines are filled through the geometry realization feather
//!   mesh (`fill_outline` → `geometry::realization::fill_feather`) — the same AA tessellator native's
//!   `TextOutlineRealizer::CacheGeometry` sweeps them with — straight coverage, source-over.
//!
//! A **rotated/skewed** transform disables grid-fit (`GetRecommendedRenderingModeInternal` returns
//! `gridFitMode = DISABLED` off-axis) and renders the unhinted outline transformed into the rotated
//! device frame; ClearType keeps its device-horizontal RGB subpixel structure (the 6×/5× oversample is
//! on the device axes, so the colour fringes follow the physical stripes — `text-07`, correcting the
//! spec's old grayscale-fallback rule). Only a font without a TrueType scaler still returns `false`.
//! `text-07` matches native's shapes and fringes but is not byte-exact (≤26 LSB on edges): rotation
//! turns every glyph edge diagonal, exposing the scan converter's ≤1-sub-sample diagonal-DDA precision
//! floor (the curve-tangent residual the grid-fit-disabled fixtures show) on every edge at once.
//! The **grid-fit-enabled** scenes (`text-*`, `16`/`17`) run end-to-end but are not byte-exact: they
//! diverge at the TrueType hinting interpreter's sub-pixel stem placement (`hinting`, upstream of this
//! module), the same residual grayscale `text-01` shows.
//!
//! Font tables and glyph outlines are parsed with `skrifa`; DirectWrite's hinting and scan
//! conversion are **not** taken from it (its output is not DirectWrite's — architecture rule).

pub mod raster; // 2D-2: DirectWrite scan converter + GlyphRunAnalysis alpha textures
pub mod blend;
pub mod gdi_raster; // GDI bitmap render target rasteriser (8×1 grayscale, GDI alpha curve + straight-alpha blend)
/// DirectWrite's TrueType hinting interpreter (phase 2D: the analysis-mode back end).
/// Produces grid-fitted glyph outlines for the grayscale/ClearType/aliased scan converter.
pub mod hinting;

use crate::geometry::{Figure, FillMode, Geometry, Segment};
use d2d_types::contracts::PixelBuffer;
use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{DrawSettings, OutlinePen};
use skrifa::raw::TableProvider;
use skrifa::{FontRef, GlyphId, MetadataProvider};

/// One positioned glyph in a run. `advance` and `offset` are DIPs in the run's reading direction,
/// already measured by layout (GDI advance rounding, if any, is applied upstream).
#[derive(Clone, Copy, Debug)]
pub struct Glyph {
    pub id: u16,
    pub advance: f32,
    pub offset: [f32; 2],
}

/// A `DWRITE_GLYPH_RUN` ready to rasterize: the font, the em size and the positioned glyphs, plus
/// the baseline origin (DIPs, before the device transform) and the run attributes that steer
/// rendering-mode selection and pixel snapping.
#[derive(Clone, Copy, Debug)]
pub struct GlyphRun<'a> {
    pub font_data: &'a [u8],
    pub face_index: u32,
    /// Em size in DIPs (the `fontEmSize` the app passed `DrawGlyphRun`).
    pub em_size: f32,
    pub glyphs: &'a [Glyph],
    /// Pen position of the first glyph on the baseline, in DIPs (pre-transform).
    pub baseline_origin: [f32; 2],
    /// `DWRITE_MEASURING_MODE`: 0 natural, 1 GDI classic, 2 GDI natural.
    pub measuring_mode: u32,
    pub is_sideways: bool,
    pub bidi_level: u32,
    /// `D2D1_DRAW_TEXT_OPTIONS_NO_SNAP`: skip baseline pixel snapping.
    pub no_snap: bool,
    /// `DWRITE_FONT_SIMULATIONS` the run's font face carries (1 bold, 2 oblique, 3 both) — native
    /// shears the glyph for oblique and emboldens it for bold; 0 for a real face.
    pub simulations: u32,
    /// Normalized variation coordinates (`fvar` axes → F2Dot14) when the run's face is an instance of
    /// a variable font; empty for a static face or the default instance. The glyph outlines are
    /// `gvar`-interpolated at these coordinates. [`crate::text::variations`].
    pub coords: &'a [skrifa::raw::types::F2Dot14],
}

/// Each glyph's own advance in DIPs, the face's advance at the run's em under transform `xf`: design units
/// scaled by em/upem; under a GDI measuring mode the GDI-compatible advance, in whole pixels when `xf` is an
/// axis-aligned uniform scale (DirectWrite's metrics converter). Bold simulation widens an advancing glyph: an inked
/// one by round(upem/50) design units, any one by a pixel under GDI (VM: a zero-advance mark keeps 0).
pub fn own_advances(run: &GlyphRun, xf: &[f64; 6]) -> Vec<f32> {
    let Ok(font) = FontRef::from_index(run.font_data, run.face_index) else { return vec![0.0; run.glyphs.len()] };
    let upem = font.head().map_or(2048, |h| h.units_per_em());
    let gm = font.glyph_metrics(Size::unscaled(), LocationRef::new(run.coords));
    let bold = run.simulations & 1 != 0;
    let gdi = matches!(run.measuring_mode, 1 | 2);
    let client = client_units(run, xf, upem);
    let xf = [xf[0] as f32, xf[1] as f32, xf[2] as f32, xf[3] as f32];
    run.glyphs
        .iter()
        .map(|g| {
            let gid = GlyphId::new(g.id as u32);
            let design = gm.advance_width(gid).unwrap_or(0.0).round() as i32;
            let bold = bold && design != 0;
            let du = gdi
                .then(|| gdi::gdi_compat_advance_du(&font, g.id, run.em_size, 1.0, run.measuring_mode == 1, Some(xf), upem, bold))
                .flatten()
                .unwrap_or_else(|| design + if bold && gm.bounds(gid).is_some() { (upem as f32 / 50.0).round() as i32 } else { 0 });
            client(du)
        })
        .collect()
}

/// DirectWrite's metrics converter: design units to DIPs at the run's em under transform `xf`. Under a GDI measuring
/// mode on an axis-aligned uniform scale or a quarter turn each value rounds to whole pixels, half up.
pub(crate) fn client_units(run: &GlyphRun, xf: &[f64; 6], upem: u16) -> impl Fn(i32) -> f32 {
    let gdi = matches!(run.measuring_mode, 1 | 2);
    let xf = [xf[0] as f32, xf[1] as f32, xf[2] as f32, xf[3] as f32];
    let zero = |v: f32| v.abs() < 1.0 / 65536.0;
    let pair = if zero(xf[1]) { zero(xf[2]).then_some((xf[0], xf[3])) } else { (zero(xf[0]) && zero(xf[3])).then_some((xf[1], xf[2])) };
    let px_per_du = match pair.filter(|&(a, b)| gdi && zero(a.abs() - b.abs())) {
        Some((a, _)) => run.em_size * a.abs() / upem as f32,
        None => 1.0,
    };
    let dip_per = run.em_size / (px_per_du * upem as f32);
    move |du| {
        let v = du as f32 * px_per_du;
        (if px_per_du != 1.0 && v.abs() < 8388608.0 { gdi::round_half_up(v) as f32 } else { v }) * dip_per
    }
}

/// A private [`GlyphRun::simulations`] bit: the run's oblique simulation is already in the transform, as the vertical
/// shear of an upright glyph in a vertical run (the scaler's sideways oblique).
pub(crate) const SIM_SIDEWAYS_OBLIQUE: u32 = 0x100;

/// The vertical origin, in design units, that GDI-natural measuring reports for a sideways glyph at `em` pixels: the
/// scaler places the vertical-origin phantom point on a whole pixel at the integer ppem; under the oblique simulation
/// the sideways slant then lowers it by the slant of the phantom's x, half the advance (`advance >> 1` design units in
/// 26.6), and the point maps back to design units through the unslanted scale (dwrite reading; VM probes,
/// GetGdiCompatibleGlyphMetrics of Yu Gothic UI Bold at em 70 to 117, upright and oblique: 60 of 60 exact).
fn gdi_natural_vertical_origin(vorg: i32, advance: i32, em: f32, upem: i32, oblique: bool) -> i32 {
    let ppem = (em as f64 + 0.5).floor().max(1.0);
    let upem = upem.max(1) as f64;
    let mut y = (vorg as f64 * ppem / upem + 0.5).floor() as i64 * 64;
    if oblique {
        let x = ((advance >> 1) as f64 * ppem * 64.0 / upem + 0.5).floor() as i64;
        y -= (x * 22272 + 0x8000) >> 16;
    }
    (y as f64 * upem / (64.0 * ppem) + 0.5).floor() as i32
}

/// The identity transform: an outline run's glyph origins are measured under it ([`rtl_glyphs`]).
pub const IDENTITY: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// Whether the scaler hints at the fractional device em. DirectWrite rounds the em to whole pixels only for a mode
/// that forces grid-fitting (aliased, GDI) or a face that forces integer ppems, and then only under a uniform
/// scale, a flip or a quarter turn of the world transform (the oblique simulation comes later); the scaler itself
/// rounds the em of a face whose head flags force integer ppems. VM: Segoe UI Emoji, which does not, hints at em
/// 19.44 under a vertical shear in every mode, Arial at 19.
fn fractional_em(dt: &[f64; 6], run: &GlyphRun, font: &FontRef, mode: u32) -> bool {
    if font.head().is_ok_and(|h| h.flags().bits() & 8 != 0) {
        return false;
    }
    // An upright glyph's sideways oblique, drawn as a vertical shear of the transform, is the scaler's, not the world's.
    let k = if run.simulations & SIM_SIDEWAYS_OBLIQUE != 0 { raster::OBLIQUE_SHEAR } else { 0.0 };
    let dt = [dt[0] - k * dt[2], dt[1] - k * dt[3], dt[2], dt[3]];
    let em = run.em_size as f64;
    let zero = |v: f64| (v * em).abs() < 1.0 / 65536.0;
    let pair = if zero(dt[1]) { zero(dt[2]).then_some((dt[0], dt[3])) } else { (zero(dt[0]) && zero(dt[3])).then_some((dt[1], dt[2])) };
    let uniform = pair.is_some_and(|(a, b)| zero(a.abs() - b.abs()));
    !(uniform && matches!(mode, MODE_ALIASED | MODE_GDI_CLASSIC | MODE_GDI_NATURAL))
}

/// A right-to-left run's glyphs as a left-to-right pen loop draws them: in the run's own order from the same pen origin
/// (the run's right end), each advance negated so the pen moves left. Native places a right-to-left glyph with its own
/// advance ([`own_advances`] under `xf`: the device transform for a rasterized or colour run, none for an outline run,
/// whose glyph origins Direct2D takes at one pixel per DIP) ending at the pen and its advance offset pointing left
/// (`pen − own − advanceOffset`), then moves the pen left by the run's advance: an offset of `−own − advanceOffset` and
/// an advance of `−advance`, which the pen loop takes one f32 step at a time as native does. Summing the advances to a
/// left edge and adding them back rounds differently and flips half-pixel ties under GDI measuring (VM: Segoe UI 14
/// Hebrew at 175% from a whole-DIP origin, scattered glyphs a pixel off). A kerned glyph so keeps its place against
/// its right neighbour with no offset, as DirectWrite reports it.
pub fn rtl_glyphs(run: &GlyphRun, xf: &[f64; 6]) -> Vec<Glyph> {
    let own = own_advances(run, xf);
    run.glyphs.iter().zip(own).map(|(g, a)| Glyph { advance: -g.advance, offset: [-a - g.offset[0], g.offset[1]], ..*g }).collect()
}

/// The run as drawn left to right: a right-to-left run through [`rtl_glyphs`] (its glyphs held in `store`), any other
/// as is.
pub fn drawn_ltr<'a>(run: &GlyphRun<'a>, xf: &[f64; 6], store: &'a mut Vec<Glyph>) -> GlyphRun<'a> {
    if run.bidi_level & 1 == 0 {
        return *run;
    }
    *store = rtl_glyphs(run, xf);
    let store: &'a Vec<Glyph> = store;
    GlyphRun { glyphs: store, bidi_level: 0, ..*run }
}

/// The three scalars + geometry the `IDWriteRenderingParams` the context carries resolve to, plus
/// the explicit `DWRITE_RENDERING_MODE` (0 = `DEFAULT`, let the recommendation decide).
#[derive(Clone, Copy, Debug)]
pub struct RenderingParams {
    /// System fonts use FontCache arithmetic; app-provided fonts use in-process DirectWrite.
    pub rasterizer: hinting::Rasterizer,
    /// Render-target DPI, independent of its world transform; used by native's mode-7 recommendation.
    pub dpi: [f32; 2],
    pub gamma: f32,
    pub enhanced_contrast: f32,
    pub grayscale_contrast: f32,
    pub cleartype_level: f32,
    /// `DWRITE_PIXEL_GEOMETRY`: 0 flat, 1 RGB, 2 BGR.
    pub pixel_geometry: u32,
    /// `DWRITE_RENDERING_MODE`.
    pub rendering_mode: u32,
    /// `DWRITE_GRID_FIT_MODE`: 0 default (grid-fit via the TrueType hinting interpreter), 1 disabled
    /// (unhinted outline — the scan converter's verified input), 2 enabled. Only disabled vs
    /// not-disabled is acted on; both grid-fitting modes run the hinter.
    pub grid_fit: u32,
    /// `D2D1_PRIMITIVE_BLEND` resolved to the pixel composite mode. Text honours it exactly as fills
    /// do: SOURCE_OVER is the normal path, COPY replaces the destination, MIN/ADD/MAX go per-channel
    /// (browser `fillText` under `globalCompositeOperation` copy/lighter/etc.).
    pub blend: crate::pixels::Blend,
    /// `D2D1_DRAW_TEXT_OPTIONS_NO_SNAP` of the DrawText/DrawTextLayout being drawn: its text renderer
    /// disables pixel snapping, so the layout's runs keep their sub-pixel baselines.
    pub no_snap: bool,
    /// The device pixel rect the draw's edge list is clipped to (the target within the pushed clips and
    /// layer bounds), set by the device context for the draw. An outline fill is clipped to it as a
    /// geometry fill is, and under COPY a curve wholly off it stays its chord.
    pub raster_clip: Option<[i32; 4]>,
    /// The pushed clips' pixel rect by the pixel-centre rule (an antialiased clip's partly covered pixels left
    /// out), which a bitmap brush's text intermediate is drawn within.
    pub clip_px: Option<[i32; 4]>,
}

impl Default for RenderingParams {
    /// The harness-pinned set: gamma 2.2, enhanced/grayscale contrast 1.0,
    /// ClearType level 1.0, flat pixel geometry, `DEFAULT` rendering mode, `DEFAULT` grid fit.
    fn default() -> Self {
        RenderingParams {
            rasterizer: hinting::Rasterizer::FontCache,
            dpi: [96.0, 96.0],
            gamma: 2.2,
            enhanced_contrast: 1.0,
            grayscale_contrast: 1.0,
            cleartype_level: 1.0,
            pixel_geometry: 0,
            rendering_mode: 0,
            grid_fit: 0,
            blend: crate::pixels::Blend::SourceOver,
            no_snap: false,
            raster_clip: None,
            clip_px: None,
        }
    }
}

// DWRITE_RENDERING_MODE1 values. 4 = NATURAL and 5 = NATURAL_SYMMETRIC are the two smooth
// sub-pixel modes; the recommendation picks between them from the font's `gasp` SYMMETRIC_SMOOTHING
// flag at the device ppem (see `gasp_symmetric`).
const MODE_DEFAULT: u32 = 0;
const MODE_ALIASED: u32 = 1;
const MODE_GDI_CLASSIC: u32 = 2;
const MODE_GDI_NATURAL: u32 = 3;
const MODE_NATURAL: u32 = 4;
const MODE_NATURAL_SYMMETRIC: u32 = 5;
const MODE_OUTLINE: u32 = 6;
const MODE_NATURAL_SYMMETRIC_DOWNSAMPLED: u32 = 7;

/// `D2D1_TEXT_ANTIALIAS_MODE`.
pub const AA_ALIASED: u32 = 3;
pub const AA_CLEARTYPE: u32 = 1;
pub const AA_GRAYSCALE: u32 = 2;

/// Slant tangent of `DWRITE_FONT_SIMULATIONS_OBLIQUE`: [`raster::OBLIQUE_SHEAR`], the TrueType scaler's
/// shear. Overridable via `D2D_OBLIQUE_SKEW` while calibrating.
const OBLIQUE_SKEW: f64 = raster::OBLIQUE_SHEAR;

/// `GetRecommendedRenderingMode` (R2) for the axis-aligned case every text scene uses. Reproduces
/// the native outline cutoff (100 px antialiased / 350 px aliased) and maps the measuring mode to a
/// `DWRITE_RENDERING_MODE1`. The NATURAL-vs-NATURAL_SYMMETRIC split (the font's `gasp` table) only
/// affects the grid-fitted coverage (not implemented yet), so below the cutoff this returns the
/// symmetric mode as the representative analysis mode.
pub fn recommended_rendering_mode(run: &GlyphRun, dt: &[f64; 6], text_aa_mode: u32, params: &RenderingParams) -> u32 {
    if params.rendering_mode != MODE_DEFAULT {
        return params.rendering_mode;
    }
    // Device em-pixel size = em_size · the transform's max axis scale (dpi/96 for an untransformed
    // target); rotation/skew only occurs in analysis-mode scenes, so the axis-max is exact here.
    let scale = (dt[0].abs()).max(dt[3].abs()) as f32;
    let ppem = run.em_size * scale;
    let threshold = if text_aa_mode == AA_ALIASED { 350.0 } else { 100.0 };
    // The outline cutoff measures the em by the matrix max factor (its larger singular value), which
    // is the axis max only on the axes: VM RECMODE at 45°, em 120 → OUTLINE, em 90 → NATURAL_SYMMETRIC
    // (r8 9209: a rotated Arial at max factor ≈10 is outline text, not a bitmap).
    let mf = color::bitmap::max_factor([dt[0] as f32, dt[1] as f32, dt[2] as f32, dt[3] as f32]);
    if run.em_size * mf >= threshold {
        return MODE_OUTLINE;
    }
    // A transform that is not axis-aligned skips the gasp lookup: native picks NATURAL_SYMMETRIC
    // (DOWNSAMPLED at ≥172.8 DPI) for it, a quarter turn included (VM probe-vct: Courier New, Verdana and
    // Arial 'Wlg' 17.3 px under [0,1,-1,0] render as NATURAL_SYMMETRIC, Courier's GRIDFIT-only gasp too).
    let zero = |v: f64| v.abs() < 1.0 / 65536.0;
    let off_axis = !(zero(dt[1]) && zero(dt[2]));
    let ds = params.dpi[0].min(params.dpi[1]) * 100.0 >= 17280.0;
    match run.measuring_mode {
        1 => MODE_GDI_CLASSIC,
        2 => MODE_GDI_NATURAL,
        _ if text_aa_mode == AA_ALIASED => MODE_ALIASED,
        _ if off_axis && ds => MODE_NATURAL_SYMMETRIC_DOWNSAMPLED,
        _ if off_axis => MODE_NATURAL_SYMMETRIC,
        // NATURAL vs NATURAL_SYMMETRIC comes from the font's `gasp` DOGRAY flag at the rounded device
        // ppem (native's recommended-rendering-mode smoothing table): a range that enables
        // grayscale AA (DOGRAY) renders NATURAL_SYMMETRIC (the 4×4 grid), a range that does not renders
        // NATURAL (the horizontal-only 8×1 grid). Courier New (9-36 px) and Palatino Linotype (9-17 px)
        // have GRIDFIT-only ranges (no DOGRAY) → NATURAL/8×1; Palatino ≥18 px re-enables DOGRAY → 4×4.
        // Native GetRecommendedRenderingModeInternal selects 2x2 grayscale at >=172.8 DPI.
        _ if gasp_symmetric(run.font_data, run.face_index, ppem) && ds => MODE_NATURAL_SYMMETRIC_DOWNSAMPLED,
        _ if gasp_symmetric(run.font_data, run.face_index, ppem) => MODE_NATURAL_SYMMETRIC,
        _ => MODE_NATURAL,
    }
}

/// The device ppem `GetRecommendedRenderingMode` looks a gasp range up at: rounded to nearest, a tie
/// going up (`cvtss2si` plus the exact-half bump), read unsigned — so a negative, NaN or too-large
/// value is 0xFFFF.
fn gasp_ppem(ppem: f32) -> u16 {
    let r = ppem.round_ties_even();
    let r = if r - ppem == -0.5 { r + 1.0 } else { r };
    if (0.0..=65535.0).contains(&r) { r as u16 } else { 0xFFFF }
}

/// Whether native renders NATURAL_SYMMETRIC (mode 5, the 4×4 grid) rather than NATURAL (mode 4, 8×1)
/// at `ppem` device pixels. Captured from `GetRecommendedRenderingMode`: with a version-1 `gasp`, symmetric
/// when the range covering the rounded ppem sets SYMMETRIC_SMOOTHING (the first range whose `rangeMaxPPEM` ≥
/// the ppem, [`gasp_ppem`], else the last); with no such table, symmetric when the rounded ppem is **≥ 21**.
/// (Palatino Linotype and Courier New, version-0 gasp, switch 4→5 at ppem 21.)
/// **No `gasp` table → NATURAL (4)** below ppem 21 — a gasp-less font has no SYMMETRIC_SMOOTHING
/// range, so GetRecommendedRenderingMode returns NATURAL
/// (VM RECMODE: Lucida Console, no gasp, @16/@20 → mode 4, @24 → mode 5 via the ≥21 rule).
fn gasp_symmetric(font_data: &[u8], face_index: u32, ppem: f32) -> bool {
    const GASP_SYMMETRIC_SMOOTHING: u16 = 0x0008;
    const SYMMETRIC_PPEM: u16 = 21;
    let ppem = gasp_ppem(ppem);
    // A version-1 gasp with ranges builds the smoothing toggle array, and the range covering the ppem decides, the last
    // range reaching every larger ppem (VM: Calibri, whose range to 21 lacks SYMMETRIC_SMOOTHING, renders NATURAL at
    // ppem 21, NATURAL_SYMMETRIC from 22). Without one (no gasp, a version-0 gasp) the ppem decides: symmetric from 21.
    let ranges = FontRef::from_index(font_data, face_index).ok().and_then(|font| {
        let d = font.table_data(skrifa::raw::types::Tag::new(b"gasp"))?;
        let d = d.as_bytes();
        let n = if d.len() >= 4 && u16::from_be_bytes([d[0], d[1]]) != 0 { u16::from_be_bytes([d[2], d[3]]) as usize } else { 0 };
        let v: Vec<(u16, u16)> = (0..n).map(|i| 4 + i * 4).take_while(|&o| o + 4 <= d.len()).map(|o| (u16::from_be_bytes([d[o], d[o + 1]]), u16::from_be_bytes([d[o + 2], d[o + 3]]))).collect();
        (!v.is_empty()).then_some(v)
    });
    let Some(ranges) = ranges else { return ppem >= SYMMETRIC_PPEM };
    let flags = ranges.iter().find(|r| ppem <= r.0).or(ranges.last()).map_or(0, |r| r.1);
    flags & GASP_SYMMETRIC_SMOOTHING != 0
}

/// Whether native grid-fits (runs the glyph hinting program) for the SMOOTH modes NATURAL / NATURAL_
/// SYMMETRIC at `ppem` device pixels. DirectWrite's recommended rendering mode
/// drives its `DWRITE_GRID_FIT_MODE` output from a gasp toggle array built **from the SYMMETRIC_GRIDFIT
/// bit (0x04)**, gated and toggled by that bit — so grid-fit is enabled iff the gasp range covering
/// the rounded ppem sets 0x04.
/// ppem is rounded as for [`gasp_symmetric`], from at least 0, and a ppem of 0 reads as 1. A version-0
/// gasp builds no symmetric arrays (so the smoothing-table pointer is null ⇒ grid-fit ENABLED), and no/short `gasp` likewise
/// ⇒ true (the conservative default that leaves gasp-less fonts hinting as before). The GDI and aliased
/// modes force grid-fit on (the per-mode 0x80000 bit); NATURAL[4], NATURAL_SYMMETRIC[5]
/// and NATURAL_SYMMETRIC_DOWNSAMPLED[7] all consult this gate via the 0x100000 bit (DOWNSAMPLED is
/// byte-identical to NAT_SYM; only OUTLINE[6]=0xFC000000 has neither, and it never reaches the hinter).
/// Verified: Consolas 'A' ppem 10 (gasp 0x0A, no 0x04) renders UNHINTED (contour != our hinted), ppem 11
/// (0x07, has 0x04) renders HINTED (byte-exact).
fn gasp_gridfit(font_data: &[u8], face_index: u32, ppem: f32) -> bool {
    const GASP_SYMMETRIC_GRIDFIT: u16 = 0x0004;
    let ppem = gasp_ppem(ppem.max(0.0)).max(1);
    let Ok(font) = FontRef::from_index(font_data, face_index) else { return true };
    let Some(data) = font.table_data(skrifa::raw::types::Tag::new(b"gasp")) else { return true };
    let d = data.as_bytes();
    if d.len() < 4 || u16::from_be_bytes([d[0], d[1]]) == 0 {
        return true; // no table / version-0 gasp: no SYMMETRIC_GRIDFIT array ⇒ native grid-fits
    }
    let n = u16::from_be_bytes([d[2], d[3]]) as usize;
    // Past the last range, that range's setting holds (native's toggle array ends at 0xFFFF).
    let mut last = None;
    for i in 0..n {
        let o = 4 + i * 4;
        if o + 4 > d.len() {
            break;
        }
        let flags = u16::from_be_bytes([d[o + 2], d[o + 3]]);
        if ppem <= u16::from_be_bytes([d[o], d[o + 1]]) {
            return flags & GASP_SYMMETRIC_GRIDFIT != 0;
        }
        last = Some(flags);
    }
    last.is_none_or(|f| f & GASP_SYMMETRIC_GRIDFIT != 0)
}

/// Whether the glyphs of `run` are hinted at device ppem `ppem`: `fit` is DWRITE_GRID_FIT_MODE, DISABLED (1) leaves
/// every glyph unhinted, ENABLED (2) overrides the gasp gate, and the GDI and aliased modes always hint.
fn grid_fitted(run: &GlyphRun, mode: u32, aa: u32, fit: u32, ppem: f32) -> bool {
    fit != 1 && (fit == 2 || matches!(mode, MODE_GDI_CLASSIC | MODE_GDI_NATURAL) || aa == AA_ALIASED || gasp_gridfit(run.font_data, run.face_index, ppem))
}

/// The rounded y ppem the scaler's bitmap bold sees ([`raster::BoldPpem`]), 0 off a bitmap bold. An unhinted glyph's
/// scaler runs in font units, so it sees the units per em there, which always takes the plain dilation (VM: r25 30268,
/// Tahoma bold oblique with grid fit disabled).
fn bitmap_bold_ppem(run: &GlyphRun, mode: u32, aa: u32, fit: u32, dt: &[f64; 6], bold: bool) -> i32 {
    let ppem = run.em_size * dt[0].abs().max(dt[3].abs()) as f32;
    if !bold || !grid_fitted(run, mode, aa, fit, ppem) { 0 } else { (run.em_size as f64 * dt[3] + 0.5).floor() as i32 }
}

/// Whether the font's `gasp` range at `ppem` asks for real antialiasing — `DOGRAY` (0x02) OR
/// `SYMMETRIC_GRIDFIT` (0x04). A range with NEITHER is a pure-`GRIDFIT` (bi-level) range (0x01): the font
/// is meant for black-and-white at that size, so when forced to antialias at a NATURAL_SYMMETRIC ppem,
/// native fills the glyph as OUTLINE geometry (the AA tessellator feather of the grid-fit-hinted outline,
/// continuous AA) rather than the 4×4 grayscale scan. This separates Courier New / Palatino Linotype
/// (v0 `gasp`, GRIDFIT-only at mid ppem ⇒ feather) from Impact / Times / Georgia (DOGRAY or SYMMETRIC_GRIDFIT
/// ⇒ scan — e.g. Impact ≤21 px is 0x05 = GRIDFIT+SYMMETRIC_GRIDFIT, still a scan). No `gasp` / short
/// table → true (⇒ scan), the conservative fallback that leaves gasp-less fonts on the verified path.
fn gasp_has_dogray(font_data: &[u8], face_index: u32, ppem: f32) -> bool {
    const GASP_DOGRAY: u16 = 0x0006; // DOGRAY | SYMMETRIC_GRIDFIT
    let t = ppem as i32;
    let ppem = (if (t as f32 - ppem) == -0.5 { t + 1 } else { t }).clamp(0, 0xFFFF) as u16;
    let Ok(font) = FontRef::from_index(font_data, face_index) else { return true };
    let Some(data) = font.table_data(skrifa::raw::types::Tag::new(b"gasp")) else { return true };
    let d = data.as_bytes();
    if d.len() < 4 {
        return true;
    }
    let n = u16::from_be_bytes([d[2], d[3]]) as usize;
    for i in 0..n {
        let o = 4 + i * 4;
        if o + 4 > d.len() {
            break;
        }
        let max_ppem = u16::from_be_bytes([d[o], d[o + 1]]);
        if ppem <= max_ppem {
            return u16::from_be_bytes([d[o + 2], d[o + 3]]) & GASP_DOGRAY != 0;
        }
    }
    true
}

/// DirectWrite's thin-font families, matched by exact family name: their glyph analyses add 0.5 to the text
/// contrast after the light-on-dark ramp, in every blend (grayscale, ClearType and GDI ClearType, whose
/// contrast is otherwise 0). VM: Courier New 'W' reports contrast 1.5 for GDI, NATURAL and NATURAL_SYMMETRIC.
fn thin_font_boost(font_data: &[u8], face_index: u32) -> f32 {
    use skrifa::MetadataProvider;
    const THIN: [&str; 6] = ["Courier New", "Fixed Miriam Transparent", "Miriam Fixed", "Rod", "Rod Transparent", "Simplified Arabic Fixed"];
    let family = FontRef::from_index(font_data, face_index)
        .ok()
        .and_then(|f| f.localized_strings(skrifa::string::StringId::FAMILY_NAME).english_or_first().map(|s| s.to_string()));
    if family.is_some_and(|f| THIN.contains(&f.as_str())) { 0.5 } else { 0.0 }
}

/// Whether DirectWrite widens each span of the run's cached glyph bitmaps one sub-sample to the right before the merge
/// (its glyph-bitmap cache packs every filled span one sample longer): a thin-font face ([`thin_font_boost`]) of weight
/// at most 500 with no bold simulation, on an antialiased (multi-column) grid. So Courier New and Courier New Italic
/// widen, their Bold faces and Italic under simulated bold do not (VM: cold Courier 'W' analyses; r25 30556's
/// grid-fit-disabled sideways Italic 'x', whose native scan equals ours and whose merged grid is one sample wider).
fn merge_widen(run: &GlyphRun) -> bool {
    run.simulations & 1 == 0
        && thin_font_boost(run.font_data, run.face_index) != 0.0
        && FontRef::from_index(run.font_data, run.face_index).ok().and_then(|f| f.os2().ok()).is_some_and(|o| o.us_weight_class() <= 500)
}

/// Paint one positioned glyph run into premultiplied BGRA8 `target` under `device_transform`
/// (world ∘ DPI, the same `[m11,m12,m21,m22,dx,dy]` the context builds). `color` is the foreground
/// brush's straight RGBA; `text_aa_mode` is `D2D1_TEXT_ANTIALIAS_MODE`. Returns `true` if the run
/// was painted, `false` if it needs a rendering path not implemented yet (the analysis/hinting
/// pipeline) so callers/tests can tell a gap from a success.
pub fn render_glyph_run(
    target: &mut PixelBuffer,
    device_transform: &[f64; 6],
    run: &GlyphRun,
    color: [f32; 4],
    text_aa_mode: u32,
    params: &RenderingParams,
) -> bool {
    render_glyph_run_fg(target, device_transform, run, TextFg::Solid(color), text_aa_mode, params)
}

/// Glyph-run foreground source: a flat solid colour (the per-run LUT path) or a gradient/bitmap/image
/// brush sampled per pixel in the blend. The glyph *coverage* is identical either way.
#[derive(Clone, Copy)]
pub enum TextFg<'a> {
    Solid([f32; 4]),
    Brush(&'a crate::brush::Brush),
}

/// Run a byte-only text renderer against `target`: directly on a BGRA8 target; on a float target into a transparent
/// BGRA8 scratch that is then composited source-over in f32 (the outline fill, bitmap glyphs).
/// ponytail: the scratch drops a primitive blend other than SOURCE_OVER on a float target.
pub(crate) fn via_bgra8(target: &mut PixelBuffer, f: impl FnOnce(&mut PixelBuffer) -> bool) -> bool {
    if !crate::float::is_highp(target.desc.format) {
        return f(target);
    }
    let (w, h) = (target.desc.width, target.desc.height);
    let mut tmp = PixelBuffer {
        desc: d2d_types::contracts::ImageDesc { format: d2d_types::abi::DXGI_FORMAT_B8G8R8A8_UNORM, ..target.desc },
        stride: w as usize * 4,
        data: vec![0u8; w as usize * h as usize * 4],
    };
    let ok = f(&mut tmp);
    let mut ft = crate::float::FloatTarget::new(target);
    for (i, px) in tmp.data.chunks_exact(4).enumerate() {
        if px != [0, 0, 0, 0] {
            let src = [px[2], px[1], px[0], px[3]].map(|v| v as f32 / 255.0);
            ft.blend_px(i % w as usize, i / w as usize, src, 64, crate::pixels::Blend::SourceOver);
        }
    }
    ok
}

/// The device rect of `buf`'s nonzero-alpha pixels.
/// The device box of `run`'s glyph design boxes at its em, mapped by `dt`: the run's ink box as geometry, which a singular
/// transform collapses to its extent rather than to nothing. `None` for a run with no inked glyph or a non-finite box.
/// ponytail: a sideways run's boxes are taken unturned.
fn run_ink_box(run: &GlyphRun, dt: &[f64; 6]) -> Option<[f32; 4]> {
    let font = FontRef::from_index(run.font_data, run.face_index).ok()?;
    let gm = font.glyph_metrics(Size::new(run.em_size), LocationRef::new(run.coords));
    let mut store = Vec::new();
    let run = drawn_ltr(run, dt, &mut store);
    let (mut pen, mut out): (f32, Option<[f64; 4]>) = (0.0, None);
    for g in run.glyphs {
        if let Some(b) = gm.bounds(GlyphId::new(g.id as u32)) {
            let (x, y) = (run.baseline_origin[0] + pen + g.offset[0], run.baseline_origin[1] - g.offset[1]);
            for (px, py) in [(x + b.x_min, y - b.y_max), (x + b.x_max, y - b.y_max), (x + b.x_min, y - b.y_min), (x + b.x_max, y - b.y_min)] {
                let (px, py) = (px as f64, py as f64);
                let d = [dt[0] * px + dt[2] * py + dt[4], dt[1] * px + dt[3] * py + dt[5]];
                out = Some(out.map_or([d[0], d[1], d[0], d[1]], |o| [o[0].min(d[0]), o[1].min(d[1]), o[2].max(d[0]), o[3].max(d[1])]));
            }
        }
        pen += g.advance;
    }
    // A non-finite origin or transform places the run nowhere (r19 24857: a NaN origin draws nothing and fails nothing).
    out.filter(|o| o.iter().all(|v| v.is_finite())).map(|o| o.map(|v| v as f32))
}

fn ink_bounds(buf: &PixelBuffer) -> Option<[i32; 4]> {
    let (w, h) = (buf.desc.width as usize, buf.desc.height as usize);
    let mut r = [i32::MAX, i32::MAX, i32::MIN, i32::MIN];
    for y in 0..h {
        for x in 0..w {
            // ClearType glyph coverage writes the RGB subpixel channels with a zero alpha, so detect ink
            // on ANY channel, not alpha alone — otherwise an image-brush text run never finds its ink
            // rect and never re-realizes per draw (it fell back to the creation realization).
            if buf.data[y * buf.stride + x * 4..y * buf.stride + x * 4 + 4].iter().any(|&b| b != 0) {
                r = [r[0].min(x as i32), r[1].min(y as i32), r[2].max(x as i32 + 1), r[3].max(y as i32 + 1)];
            }
        }
    }
    (r[0] < r[2]).then_some(r)
}

/// `ID2D1DeviceContext4::DrawGlyphRun` (and `DrawText`/`DrawTextLayout`) with a gradient/bitmap/image
/// brush: the same coverage + mode resolution as [`render_glyph_run`], the brush sampled per pixel in
/// the blend ([`blend::blend_grayscale_brush`]/[`blend::blend_cleartype_brush`]).
pub fn render_glyph_run_brush(
    target: &mut PixelBuffer,
    device_transform: &[f64; 6],
    run: &GlyphRun,
    brush: &crate::brush::Brush,
    text_aa_mode: u32,
    params: &RenderingParams,
) -> bool {
    render_glyph_run_fg(target, device_transform, run, TextFg::Brush(brush), text_aa_mode, params)
}

fn render_glyph_run_fg(
    target: &mut PixelBuffer,
    device_transform: &[f64; 6],
    run: &GlyphRun,
    fg: TextFg,
    text_aa_mode: u32,
    params: &RenderingParams,
) -> bool {
    if run.glyphs.is_empty() {
        return true;
    }
    let Ok(font) = FontRef::from_index(run.font_data, run.face_index) else { return false };
    // A component that is zero in 16.16 is zero for every text decision: native draws a float quarter turn
    // (cos 90° = −4.37e-8) exactly as the exact one, grid-fit included (VM probe, byte-identical).
    let device_transform = &raster::scaler_matrix(device_transform);

    // RTL run: glyphs in logical order from the run's right edge. Every render path below advances the pen by each
    // glyph's advance, so the run draws with its advances negated, the pen stepping left as native's does ([`rtl_glyphs`]).
    let mut run = *run;
    // Degenerate layout: a device baseline out of i32 range. Native lays each glyph on the raster
    // oversample grid as `glyph.left + cvttss2si(oversample · running_x)`, walking `running_x += advance`
    // in f32 (d2d1 InitGlyphPositions). `cvttss2si` returns INT_MIN for ANY out-of-range value — ±inf,
    // NaN, or a finite-huge pen such as a normalized layout rect's -3.4e38 edge — unlike Rust's saturating
    // `as i32`. The run bitmap is composited bbox-relative at a clip-clamped destination, and those INT_MIN
    // cells drag to the clip origin — so a run whose device PEN X goes out of range draws an overlapping
    // glyph stack at x≈0 (VM: a -inf-left DrawText stacks at bbox [1,0,22,32] dest [0,0]; r13 18669's
    // -3.4e38 right-edge rect does the same). The per-side decision comes from `normalize_layout_rect`'s
    // origin: `origin.x` out of range (left OR right) collapses in x and DRAWS; `origin.y` out of range
    // (top OR bottom) sends the device Y off-canvas, which native renders as nothing. Axis-aligned only; no
    // out-of-range scene rotates. The finite path is untouched (byte-exact). The overlapping-stack coverage
    // keeps a ~21px (grayscale) / ~42px (ClearType) left-edge residual from native's un-ported
    // ProcessSubRects stack merge; extent and draw/blank are exact.
    let collapsed: Vec<Glyph>;
    let dt = device_transform;
    let mut collapse: Option<(Vec<i64>, f64)> = None;
    if dt[1] == 0.0 && dt[2] == 0.0 && dt[0] != 0.0 && dt[3] != 0.0 {
        // Device Y non-finite (±inf/NaN) poisons the y arithmetic → native draws nothing (per-side y rule,
        // r12 17348/17615 bottom/top = -inf). A finite-huge y (1e9 or the 3.4e38 sentinel) needs no special
        // case: the glyph rasterizes at that device row and the raster clip drops it off-canvas = nothing.
        if !(dt[3] * run.baseline_origin[1] as f64 + dt[5]).is_finite() {
            return true;
        }
        // Native's glyph pen is a 64-bit cvttss2si(oversample · running_x): it wraps to INT_MIN only when the
        // oversampled pen leaves i64 range (±inf, NaN, or the ~3.4e38 D2D infinity sentinel) — a MERELY-large
        // finite pen (e.g. a -1e9 rect edge, device 8e9 at ox=8) stays a valid 64-bit int OFF-canvas and
        // draws nothing (r11 16458, r12 17451/17654, recap-g 9072). Cheap pre-check (|device x| < 2^63 /
        // max-oversample) skips the mode/oversample resolve for finite normal runs and keeps them
        // allocation-free; i64 (not i32) is the whole point — the i32 threshold wrongly collapsed the -1e9s.
        let safe = 9_223_372_036_854_775_808.0 / 8.0;
        let mut running = dt[0] * run.baseline_origin[0] as f64 + dt[4];
        let mut maybe = !(running.abs() < safe);
        for g in run.glyphs.iter() {
            running += dt[0] * g.advance as f64;
            maybe |= !(running.abs() < safe);
        }
        if maybe {
            let mode = recommended_rendering_mode(&run, dt, text_aa_mode, params);
            let aa = resolve_text_aa(text_aa_mode, target);
            // Mode's x-oversample (native `dword_180373538[mode]`): aliased 1, ClearType 6,
            // NATURAL_SYMMETRIC 4, other grayscale (NATURAL/GDI) 8 — the grid the pen is cvtt'd on.
            let ox: f64 = if aa == AA_ALIASED || mode == MODE_ALIASED {
                1.0
            } else if aa == AA_CLEARTYPE {
                6.0
            } else if mode == MODE_NATURAL_SYMMETRIC {
                4.0
            } else {
                8.0
            };
            let cvtt = |v: f64| -> i64 {
                if v >= -9_223_372_036_854_775_808.0 && v < 9_223_372_036_854_775_808.0 { v as i64 } else { i64::MIN }
            };
            let mut running = dt[0] * run.baseline_origin[0] as f64 + dt[4];
            let mut pens: Vec<i64> = Vec::with_capacity(run.glyphs.len());
            for g in run.glyphs.iter() {
                pens.push(cvtt(ox * running));
                running += dt[0] * g.advance as f64;
            }
            // A merely-large pen whose 64-bit int is valid (off-canvas) is NOT INT_MIN — only collapse when
            // native wraps, i.e. when some glyph's oversampled pen leaves i64 range (inf/NaN/3.4e38 sentinel).
            if pens.iter().any(|&p| p == i64::MIN) {
                collapse = Some((pens, ox));
            }
        }
    }
    if let Some((pens, ox)) = collapse {
        if !matches!(run.measuring_mode, 1 | 2) {
            // Only the GDI measuring modes wrap the out-of-range (INT_MIN) pen back on-canvas to the clip
            // origin (GDI-compatible glyph-metrics path). The non-GDI NATURAL positioner leaves
            // `glyph.left + INT_MIN` ≈ −2.1e9 off-canvas, so native draws NOTHING for a NATURAL-measured
            // out-of-range run (r12 17240 Times 'Ag' and 17434's op#6 are NATURAL -inf → blank, VM-confirmed).
            return true;
        }
        // A near-zero device em (e.g. a 1e-20 `size`) produces no ink: native draws nothing even when the
        // 3.4e38-sentinel origin collapses the run on-canvas, because the glyphs round to nothing at that
        // em. Without this the collapsed sub-run rasterizes a spurious speck (r16 21932: size 1e-20 with a
        // rect.left = −3.4e38). The finite-pen path already renders this off-canvas, so this only bites the
        // sentinel collapse.
        if (run.em_size as f64 * dt[0].abs().max(dt[3].abs())).round() == 0.0 {
            return true;
        }
        let min_os = *pens.iter().min().unwrap();
        if matches!(params.blend, crate::pixels::Blend::SourceOver) && run.measuring_mode == 0 {
            // SOURCE_OVER: native composites each glyph's quad in run order (one aliased-rectangle
            // fast-path fill per glyph slot into its
            // PRIVATE FillSubrect coverage buffer), so an overlapping collapsed stack alpha-composites
            // glyph-by-glyph. Render each collapsed glyph as its own 1-glyph run at its bbox-relative
            // device x; a finite sub-run takes the ordinary path and uses the suite-exact single-glyph
            // blend. (A winding union of all contours in one scan over/under-covers the shared AA edges.)
            // EXCEPTION — GDI_CLASSIC (measuring 1): native fills the whole collapsed run through ONE
            // FillSubrect (native's fill-subrect dump of the cold Times 'Wx1!' @NaN-origin: a single call
            // whose coverage buffer is the merged overlapping glyphs, not per-glyph), so fall through to
            // the single-coverage merge below. GDI_NATURAL (2) / NATURAL (0) / ClearType keep per-glyph.
            let mut ok = true;
            for (i, g) in run.glyphs.iter().enumerate() {
                let dev_x = pens[i].wrapping_sub(min_os) as f64 / ox;
                let one = [Glyph { advance: 0.0, ..*g }];
                let mut sub = run;
                sub.baseline_origin = [((dev_x - dt[4]) / dt[0]) as f32, run.baseline_origin[1]];
                sub.bidi_level = 0;
                sub.glyphs = &one;
                ok &= render_glyph_run_fg(target, dt, &sub, fg, text_aa_mode, params);
            }
            return ok;
        }
        // Under a replacing/non-additive SetPrimitiveBlend (COPY/MIN/ADD/MAX) a per-glyph composite would
        // replace/clip the overlap to the last glyph and lose coverage; native instead merges the stack's
        // coverage into ONE glyph-run bitmap and applies the primitive blend ONCE. Reproduce that with the
        // single-scan union: collapse the advances (bbox-relative) and fall through to the ordinary one-run
        // path (16162: CT 'fjord' under COPY on a premultiplied target).
        let pen_dev: Vec<f64> = pens.iter().map(|&p| p.wrapping_sub(min_os) as f64 / ox).collect();
        run.baseline_origin[0] = ((pen_dev[0] - dt[4]) / dt[0]) as f32;
        collapsed = (0..run.glyphs.len())
            .map(|i| {
                let adv = if i + 1 < pen_dev.len() { ((pen_dev[i + 1] - pen_dev[i]) / dt[0]) as f32 } else { 0.0 };
                Glyph { advance: adv, ..run.glyphs[i] }
            })
            .collect();
        run.glyphs = &collapsed;
    }
    let mode = recommended_rendering_mode(&run, device_transform, text_aa_mode, params);
    let mut rtl = Vec::new();
    let run = &drawn_ltr(&run, if mode == MODE_OUTLINE { &IDENTITY } else { device_transform }, &mut rtl);
    if mode == MODE_OUTLINE {
        return via_bgra8(target, |t| {
            fill_outline(t, device_transform, run, &font, &fg, params.blend, text_aa_mode == AA_ALIASED, params.raster_clip);
            true
        });
    }
    // Smooth grayscale (NATURAL_SYMMETRIC 4×4 or NATURAL 8×1) for a font whose `gasp` has no DOGRAY at
    // this ppem (Courier New / Palatino Linotype / Lucida Console …): native runs the DWrite
    // `GlyphRunAnalysis` merge (an RLE sub-column blit) + box resolve + per-font enhanced-contrast over
    // the grid-fit-hinted fill — see [`fill_hinted_geometry`]. The plain scan (`render_analysis`, EC 1.0)
    // is the fallback for everyone else, so DOGRAY fonts (Impact, Times, Georgia) stay byte-exact.
    // It is a grid-fit path, so a rotated/skewed transform or explicit DISABLED grid-fit (both unhinted in
    // native) stays on `render_analysis`'s unhinted outline — r20 25314: Arial @16 ppem (no DOGRAY) under a
    // singular [.5 .5 .5 .5] collapses to native's diagonal hairlines, not upright hinted glyphs.
    // Grayscale only: ClearType keeps its subpixel path whatever the gasp DOGRAY bit says (r4 4485: explicit
    // ClearType Times Bold @9 on a premultiplied target — native's per-channel CT, dest alpha kept).
    if (mode == MODE_NATURAL_SYMMETRIC || mode == MODE_NATURAL)
        && resolve_text_aa(text_aa_mode, target) == AA_GRAYSCALE
        && device_transform[1] == 0.0
        && device_transform[2] == 0.0
        && params.grid_fit != 1
        && !gasp_has_dogray(run.font_data, run.face_index, run.em_size * (device_transform[0].abs().max(device_transform[3].abs())) as f32)
        && fill_hinted_geometry(target, device_transform, run, &font, &fg, params, mode)
    {
        return true;
    }
    render_analysis(target, device_transform, run, &font, fg, text_aa_mode, params, mode)
}


/// Render a grid-fit-hinted glyph through native's no-DOGRAY grayscale path (see [`gasp_has_dogray`]):
/// scan the hinted contours to the overscaled grid (NATURAL_SYMMETRIC → 4×4, NATURAL → 8×1), box-resolve
/// to coverage (`round(255·popcount/N)`), and composite with the text blend. DirectWrite's run merge is an
/// RLE sub-column blit at `glyph.x_subcol − overscaleX·rect.left` (no phase bias in either grayscale mode).
/// [`merge_widen`] fonts merge an OR-dilate-left of our scan; thin-font families ([`thin_font_boost`])
/// add +0.5 contrast. Verified byte-exact against native's captured merged grid + oracle (Courier "0"@27
/// 4×4, 'A'@16 8×1; Palatino "0"@16 8×1 plain 0/88). False if no hinted outline.
fn fill_hinted_geometry(target: &mut PixelBuffer, dt: &[f64; 6], run: &GlyphRun, font: &FontRef, fg: &TextFg, params: &RenderingParams, mode: u32) -> bool {
    // Dropout control is PER GLYPH (native calls the scan converter once per glyph with that glyph's
    // resolved SCANTYPE, not the `prep`-level default): a glyph program that clears the smart bit the
    // prep set — e.g. Palatino '6'/'R' resolve to mode 4 (exact, no smart) though prep leaves 5 — must
    // scan-convert without the smart stub test, or the vertical dropout wrongly skips its bowl-tangent
    // recoveries (textx-glyph-386/391 under-cover). Collect the per-glyph modes and scan each glyph
    // under its own, matching native's per-glyph merge; the single returned `drop` is the prep default.
    let mut glyph_modes = Vec::new();
    let Some((contours, _drop)) = hinted_contours_with_modes(dt, run, font, mode, AA_GRAYSCALE, params.rasterizer, params.grid_fit, Some(&mut glyph_modes), None) else {
        return false;
    };
    // NATURAL_SYMMETRIC vertically anti-aliases (4×4); NATURAL is horizontal-only (8×1).
    let (sx, sy) = if mode == MODE_NATURAL_SYMMETRIC { (4, 4) } else { (8, 1) };
    let boost = merge_widen(run);
    // NATURAL synthetic bold emboldens the scanned bits here as on the plain scan ([`natural_bitmap_bold`]).
    let bold = if natural_bitmap_bold(run, mode, dt) { (1, 0) } else { (0, 0) };
    let _bold_ppem = raster::BoldPpem::set(bitmap_bold_ppem(run, mode, AA_GRAYSCALE, params.grid_fit, dt, bold == (1, 0)));
    let tex = raster::coverage_glyphs(&contours, &glyph_modes, raster::Kind::Grayscale, sx, sy, bold, boost);
    if tex.width() <= 0 || tex.height() <= 0 {
        return true; // nothing covered (e.g. a space)
    }
    let alpha = blend::AlphaTexture { bytes: &tex.data, bounds: [tex.left, tex.top, tex.right, tex.bottom] };
    let thin = thin_font_boost(run.font_data, run.face_index);
    match fg {
        TextFg::Solid(color) => {
            let mut bp = blend::BlendParams::resolve(*color, params.gamma, params.grayscale_contrast, params.cleartype_level);
            (bp.contrast, bp.enhanced_contrast) = (bp.contrast + thin, bp.enhanced_contrast + thin);
            blend::blend_grayscale(target, alpha, *color, &bp, AA_GRAYSCALE, params.blend);
        }
        TextFg::Brush(b) => {
            let gamma_row = blend::BlendParams::resolve([0.0, 0.0, 0.0, 1.0], params.gamma, 0.0, params.cleartype_level).gamma_row;
            blend::blend_grayscale_brush(target, alpha, b, gamma_row, params.grayscale_contrast + thin, AA_GRAYSCALE, params.blend, params.clip_px);
        }
    }
    true
}

/// Whether a grayscale run of this rendering mode scan-converts on the 8×1 grid (NATURAL and the two
/// GDI modes) rather than NATURAL_SYMMETRIC's 4×4 — the grayscale rows of DirectWrite's
/// `dword_1801D4B20[8·aa+mode]` oversample table (0x18 ⇒ 8×1, 0x44 ⇒ 4×4).
/// Whether this mode renders with **no vertical antialiasing** — grayscale scans 8×1, ClearType 6×1
/// (one device-row sample), the horizontal-only path native uses for the GDI measuring modes and plain
/// NATURAL. Only NATURAL_SYMMETRIC adds vertical AA (grayscale 4×4, ClearType 6×5).
fn horizontal_only(mode: u32) -> bool {
    matches!(mode, MODE_GDI_CLASSIC | MODE_GDI_NATURAL | MODE_NATURAL)
}

/// `D2D1_TEXT_ANTIALIAS_MODE` resolved for this target: `DEFAULT` → GRAYSCALE on a target that
/// carries alpha (our premultiplied BGRA), CLEARTYPE on an opaque (alpha-ignore) target (R2 / the
/// DEFAULT-resolution open question).
pub(crate) fn resolve_text_aa(mode: u32, target: &PixelBuffer) -> u32 {
    use d2d_types::abi::D2D1_ALPHA_MODE_IGNORE;
    if mode != 0 {
        mode
    } else if target.desc.alpha_mode == D2D1_ALPHA_MODE_IGNORE {
        AA_CLEARTYPE
    } else {
        AA_GRAYSCALE
    }
}

/// Analysis-mode glyph rendering (R1/R3/R4/R5): resolve the text AA mode for the target, produce the
/// coverage the mode needs (grayscale / ClearType 3×1 / aliased 1×1) from either the unhinted outline
/// (`grid_fit == DISABLED`, the scan converter's verified input) or the glyph grid-fitted through
/// DirectWrite's TrueType interpreter (`hinting`), then composite with the matching text blend
/// (`blend`): grayscale/aliased source-over or ClearType dual-source. Returns false only for a path
/// still missing a sub-piece — ClearType under a rotated transform (the rotated-subpixel raster is
/// not reproduced; native renders it, see text-07) or a font without a TrueType scaler.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
/// Upright GDI-classic synthetic bold: the render pass's x-overscale widens by the bold at every size
/// ([`hinting::Config::bold_fu`]). A run whose baseline stays level counts, flipped upside down or sheared along it
/// (VM: Leelawadee UI and Trebuchet under a y-flip, Impact and Trebuchet under a one-axis shear; r25 30206, 30866),
/// or mirrored in x, alone or with y in a half turn, which then embolds the outline mirrored (VM: Malgun Gothic and
/// Trebuchet bold under an x-flip, grayscale and ClearType, every contour point the mirrored overscale widening plus the
/// one-pixel embold; Tahoma and Trebuchet MS bold-simulated at 13 px under a half turn, grayscale and ClearType); a
/// quarter turn keeps the other modes' bold.
fn gdi_classic_upright_bold(run: &GlyphRun, mode: u32, dt: &[f64; 6]) -> bool {
    let upright = dt[1] == 0.0 && dt[0] != 0.0 && dt[3] != 0.0;
    run.simulations & 1 != 0 && mode == MODE_GDI_CLASSIC && upright
}

/// Upright GDI-classic synthetic bold at a one-pixel boldX (device ppem up to 50) emboldens the scanned bits rather
/// than the outline; above that native embolds the hinted outline, grayscale and ClearType alike (VM contour
/// captures: Trebuchet, Tahoma and Georgia at ppem 13 to 38 keep the regular outline; Leelawadee UI, Comic Sans and
/// Trebuchet at ppem 53 to 58 embolden it). A flipped or sheared run embolds its outline at one pixel too (VM:
/// Trebuchet and Impact under a y-flip or a shear, 96 and 71 pixels off to exact with grid fit off).
/// Synthetic bold ClearType off a level baseline (a quarter turn, a y-shear) embolds the hinted outline in every
/// rendering mode, as grayscale does, rather than dilating the scanned bits (VM: Trebuchet 'x' turned in GDI classic,
/// ppem 58: 269 pixels to exact; Lucida Console turned in GDI natural at ppem 43, where a one-pixel boldX leaves some
/// pixels lighter than the regular glyph's, so no dilation: r25 30269 and 30622; Segoe UI Emoji 'Ag' under a 0.042
/// y-shear, 78 px to 2: r25 30277).
fn tilted_ct_bold(run: &GlyphRun, dt: &[f64; 6], aa: u32) -> bool {
    run.simulations & 1 != 0 && aa == AA_CLEARTYPE && dt[1] != 0.0
}

/// A hinted aliased run's synthetic bold at a boldX of two pixels or more embolds the hinted outline, right and up by
/// boldX and boldY before the oblique slants it, as the other modes do; at one pixel it dilates the 1-bit bitmap (VM fill
/// contours of r25 30288: Leelawadee UI bold at ppem 66, aliased, every point outset 0 to 2 px in x and 0 to 1 px in y,
/// and under the oblique by that much sheared).
fn aliased_outline_bold(run: &GlyphRun, aa: u32, device_em: f64) -> bool {
    run.simulations & 1 != 0 && aa == AA_ALIASED && raster::bold_amount(device_em.round() as i32).0 >= 2
}

/// A two-scale aliased GDI-classic synthetic bold embolds the hinted outline (one pixel in the glyph's
/// pre-stretch space), as a flip or a shear does, rather than the post-scan 1-bit dilate (one pixel in
/// the stretched device grid): the two agree only under a uniform scale (r25 fz-5029 Verdana w700 upright
/// under [1.3,1], 9 px to exact). The CT sibling is [`natural_ct_outline_bold`].
fn alias_ts_outline(run: &GlyphRun, aa: u32, mode: u32, dt: &[f64; 6]) -> bool {
    run.simulations & 1 != 0 && aa == AA_ALIASED && mode == MODE_GDI_CLASSIC && two_scale(run, dt)
}

fn gdi_classic_bitmap_bold(run: &GlyphRun, mode: u32, dt: &[f64; 6]) -> bool {
    let ppem = (run.em_size as f64 * dt[..4].iter().fold(0f64, |m, v| m.max(v.abs()))).round() as i32;
    gdi_classic_upright_bold(run, mode, dt) && dt[2] == 0.0 && dt[0] > 0.0 && dt[3] > 0.0 && raster::bold_amount(ppem).0 == 1
}

/// NATURAL and GDI_NATURAL synthetic bold at a one-pixel boldX on an upright run leaves the outline as it is and
/// emboldens the overscaled bitmap, grayscale and ClearType (VM: Palatino and Trebuchet bold contours equal the regular
/// ones at ppem 10 and 32), as GDI classic does ([`gdi_classic_bitmap_bold`]).
fn natural_bitmap_bold(run: &GlyphRun, mode: u32, dt: &[f64; 6]) -> bool {
    let ppem = (run.em_size as f64 * dt[..4].iter().fold(0f64, |m, v| m.max(v.abs()))).round() as i32;
    // Under a non-uniform x-stretch the one-pixel bitmap dilate (1 px in the stretched device grid) is
    // not the one-pixel outline embolden (1 px in the glyph's pre-stretch space), which native applies
    // here — so the two-scale CT bold takes the outline path ([`natural_ct_outline_bold`]), not this.
    run.simulations & 1 != 0
        && matches!(mode, MODE_NATURAL | MODE_GDI_NATURAL)
        && dt[1] == 0.0
        && dt[2] == 0.0
        && dt[0] > 0.0
        && dt[3] > 0.0
        && !two_scale(run, dt)
        && raster::bold_amount(ppem).0 == 1
}

/// A non-uniform axis-aligned x-stretch (GDI two-scale): x and y device scales differ, no shear.
fn two_scale(run: &GlyphRun, dt: &[f64; 6]) -> bool {
    dt[1] == 0.0 && dt[2] == 0.0 && dt[0] > 0.0 && dt[3] > 0.0
        && ((dt[0].abs() - dt[3].abs()) * run.em_size as f64).abs() >= 1.0 / 65536.0
}

/// ClearType synthetic bold on a level baseline (flips and x-shears included; VM: Verdana and Trebuchet under both flips
/// and a shear embold the hinted outline 1 px, mirrored with an x-flip) that embolds the hinted outline as grayscale
/// does, with no sub-pixel bit dilation:
/// at any boldX in NATURAL_SYMMETRIC and its downsampled mode, above a one-pixel boldX in NATURAL and GDI_NATURAL (VM:
/// Palatino and Trebuchet bold contours widen 1 px at ppem 10 and 32 in the symmetric modes, 2 px at ppem 58 in all).
/// Hinted glyphs only: an unhinted one keeps the sub-pixel dilation.
/// ponytail: native embolds an unhinted glyph's design outline instead (r25 30773 under ClearType, Trebuchet at ppem 1.2
/// with gasp hinting off), as `overscale_bold` does for grayscale; port it when a ladder pins the ClearType case.
fn natural_ct_outline_bold(run: &GlyphRun, mode: u32, dt: &[f64; 6], aa: u32, fit: u32) -> bool {
    let ppem = (run.em_size as f64 * dt[..4].iter().fold(0f64, |m, v| m.max(v.abs()))).round() as i32;
    let symmetric = matches!(mode, MODE_NATURAL_SYMMETRIC | MODE_NATURAL_SYMMETRIC_DOWNSAMPLED);
    // A non-uniform x-stretch embolds the hinted outline (pre-stretch), as a flip does below: native's
    // one-pixel bitmap dilate is in the stretched device grid, but the bold widens in the glyph's own
    // space, so a one-pixel boldX needs the outline embolden here (NATURAL/GDI_NATURAL CT, grid-fitted;
    // r25 fz-5015 Tahoma w700 oblique under [1.3,1], 55 px to exact). Uniform scales keep the bitmap bold.
    if two_scale(run, dt)
        && aa == AA_CLEARTYPE
        && matches!(mode, MODE_NATURAL | MODE_GDI_NATURAL)
        && grid_fitted(run, mode, aa, fit, run.em_size * dt[0].abs().max(dt[3].abs()) as f32)
    {
        return true;
    }
    run.simulations & 1 != 0
        && aa == AA_CLEARTYPE
        && (symmetric || matches!(mode, MODE_NATURAL | MODE_GDI_NATURAL))
        && dt[1] == 0.0
        && dt[0] != 0.0
        && dt[3] != 0.0
        // GDI natural and NATURAL under a flip (a mirror or a half turn) embolden the outline at a one-pixel boldX too,
        // mirrored, where upright they dilate the scanned bits (VM: Malgun Gothic and Trebuchet bold, GDI natural ClearType
        // under both flips and a half turn: native contours outset one pixel, mirrored; Tahoma and Verdana bold-simulated
        // at 13 px, NATURAL ClearType under an x-flip).
        && (symmetric || raster::bold_amount(ppem).0 >= 2 || (matches!(mode, MODE_GDI_NATURAL | MODE_NATURAL) && (dt[0] < 0.0 || dt[3] < 0.0)))
        && grid_fitted(run, mode, aa, fit, run.em_size * dt[0].abs().max(dt[3].abs()) as f32)
}

fn render_analysis(
    target: &mut PixelBuffer,
    dt: &[f64; 6],
    run: &GlyphRun,
    font: &FontRef,
    fg: TextFg,
    text_aa_mode: u32,
    params: &RenderingParams,
    mode: u32,
) -> bool {
    let aa = resolve_text_aa(text_aa_mode, target);
    // Under a non-SOURCE_OVER primitive blend ClearType's dual-source merger has no per-subpixel dest
    // to lerp against, so native composites the text with the GRAYSCALE-AA coverage, not the RGB
    // subpixel coverage — verified CT-under-COPY == grayscale-under-COPY (19/20 px, maxd 26; w951
    // 14951a probe: `ctc951-k-op` vs `gs951-k-copy-aa2`), while CT-vs-grayscale under OVER differ by
    // up to 211. So route the whole ClearType path (raster + blend) through grayscale here; the
    // coverage then composites in [`blend::blend_grayscale`]'s non-over arm.
    let aa = if aa == AA_CLEARTYPE && !matches!(params.blend, crate::pixels::Blend::SourceOver) {
        AA_GRAYSCALE
    } else {
        aa
    };
    // An explicit DWRITE_RENDERING_MODE_ALIASED produces 1-bit coverage whatever the text AA mode is
    // (a DEFAULT/grayscale/ClearType text AA mode with an aliased rendering mode set on the params).
    // The oversample (sx=sy=1) and smooth-mode gates already special-case MODE_ALIASED; route the
    // coverage kind, blend, and simulated-bold through the aliased path too so the glyph is bi-level.
    let aa = if mode == MODE_ALIASED { AA_ALIASED } else { aa };
    let kind = match aa {
        AA_CLEARTYPE => raster::Kind::ClearType,
        AA_ALIASED => raster::Kind::Aliased,
        _ => raster::Kind::Grayscale,
    };
    // A generic (rotated/skewed) transform disables grid-fit by default: the recommended rendering mode
    // returns `gridFitMode = DISABLED` for it. The glyph is rasterized from the unhinted outline transformed
    // into the rotated device frame, and ClearType keeps its device-horizontal RGB subpixel structure there
    // (text-07) — the 6× / 5× oversample is on the device axes, so the colour fringes follow the physical
    // stripes. An explicit ENABLED overrides the recommendation: the glyph is hinted at the scaler's axis
    // scales and mapped through the residual like a quarter turn (VM, r25 30215 and Arial/Segoe UI/Times/
    // Georgia/Verdana/Tahoma at 20-330° and skews, every AA mode, natural/GDI measuring: exact).
    // ponytail: an exact 45°/135° diagonal (|m11| = |m12|) still leaves ±46 on a few edges; probe the scaler's
    // tie handling if a scene needs it.
    let rotated = hint_scale(dt).is_none() && params.grid_fit != 2;
    // Explicit DISABLED under a uniform axis-aligned scale, on a font that forces integer ppems (head.flags bit 3,
    // which turns on native's ppem rounding), takes the same unscaled scaler path as a gasp-off glyph at the
    // rounded ppem: native draws the two identically (VM em 4.45 NAT_SYM, disabled vs default: same contours and
    // pixels). So it shares the instance's font-unit mapping and the font's dropout control; any other DISABLED
    // run keeps its fractional em on the unhinted outline.
    // ponytail: verified for the antialiased smooth modes only; the GDI modes and aliased text keep the unhinted
    // outline (r23 28231, GDI_NATURAL disabled, matches there and not here) until probed.
    let integer_ppem = font.head().is_ok_and(|h| h.flags().bits() & 8 != 0);
    let uniform_axis = dt[1] == 0.0 && dt[2] == 0.0 && ((dt[0].abs() - dt[3].abs()) * run.em_size as f64).abs() < 1.0 / 65536.0;
    let smooth = matches!(mode, MODE_NATURAL | MODE_NATURAL_SYMMETRIC | MODE_NATURAL_SYMMETRIC_DOWNSAMPLED) && aa != AA_ALIASED;
    // A glyph the font's gasp leaves unhinted at this ppem draws as grid fit disabled does: off the axes at its exact em
    // and, bold, with the design embold (VM: Comic Sans 'abo' at ppem 3 and 6.4, gasp without grid fitting, in
    // NATURAL and both symmetric modes, grayscale and ClearType, upright and under a y-shear: the contours equal the
    // disabled ones, a 6.4 em scaling unrounded under the shear and the bold outset by strength(upm)).
    let disabled = !grid_fitted(run, mode, aa, params.grid_fit, run.em_size * { let (sx, sy) = axis_scales(dt); sx.max(sy) } as f32);
    let off_axis_disabled = disabled && !(uniform_axis && integer_ppem && smooth);
    // DWRITE_FONT_SIMULATIONS_BOLD, grayscale NATURAL_SYMMETRIC, on the grid-fit-disabled-but-drawn-hinted
    // path (DISABLED grid-fit under a uniform axis-aligned integer-ppem scale, which native draws via its
    // UNSCALED/design scaler): native embolds the UNHINTED DESIGN outline with the constant design
    // strength(upm) via the per-point embolden, then scales to device and scans (the overscale-bold path), rather than
    // hinting then embolding the device outline. Captured byte-exact on the VM (the per-point embolden on the Malgun
    // 'x': its input is the unhinted glyf design outline, its amount is strength(upm), not bold_amount).
    // Non-bold stays on the hinted path (byte-exact); the device-hinted (grid-fit-on, e.g. Palatino @ 58),
    // aliased, and rotated/disabled bitmap-embold sim-bold surfaces are unchanged.
    // ClearType takes it too, the downsampled symmetric mode does, and NATURAL does above a one-pixel boldX; at one
    // pixel NATURAL keeps the outline and takes the plain bitmap dilation (VM: Trebuchet, Georgia and Malgun bold
    // contours with grid fit disabled grow by strength(upm) at ppem 4.3 to 58 in both symmetric modes and at 58 in
    // NATURAL, grayscale and ClearType alike). A face that does not force integer ppems does the same at its exact
    // em (VM: Segoe UI Emoji U+1F642 bold contours at ppem 12 to 57).
    let overscale_bold = run.simulations & 1 != 0
        && aa != AA_ALIASED
        && (matches!(mode, MODE_NATURAL_SYMMETRIC | MODE_NATURAL_SYMMETRIC_DOWNSAMPLED)
            || (matches!(mode, MODE_NATURAL | MODE_GDI_NATURAL) && raster::bold_amount((run.em_size as f64 * dt[..4].iter().fold(0f64, |m, v| m.max(v.abs()))).round() as i32).0 >= 2))
        && disabled
        && uniform_axis;
    // Aliased synthetic bold on the same grid-fit-disabled design-scaler path: native embolds the UNHINTED
    // DESIGN outline with the constant strength(upm) (VM contour capture: the w700 outline is outset from
    // w400), then scans aliased — it does NOT post-pass dilate the 1-bit bitmap. The old `embolden_aliased`
    // OR-dilate over-covered at large em (Impact em114 'c' +859 vs native +208) because a 1-bit pixel dilate
    // adds whole pixels while a ~1px outline widen only flips the threshold-crossing ones. Same design
    // strength and gate as the grayscale overscale-bold above; rotated/non-uniform aliased keeps the
    // post-pass dilate (the design-scaler path doesn't apply there — and rotated bold is a separate lane).
    // A FORCE_INTEGER_PPEM face's aliased synthetic bold is a per-point outline embolden (strength(upm))
    // only ABOVE a device ppem of 50; at ppem <= 50 native renders through its bitmap path and dilates the
    // 1-bit coverage instead. The rasterizer sets up its per-point embolden only when the integer device
    // ppem exceeds 50; at or below it the per-point outset is skipped and only the advance phantom carries
    // the +1px bold (VM probes of the embolden input: Impact 'A' em 48 dilates, em 56 embolds; Lucida
    // Console r25 30952 op6 em 12 dilates — both byte-exact). Below the threshold the bold falls to the
    // aliased OR-dilate, the same path a face that does not force integer ppems takes.
    // A face that does not force integer ppems takes the same design embold once its boldX reaches two pixels (VM fill
    // contours of r25 30384: Segoe UI Emoji bold at ppem 54.9 with grid fit off, every point outset by up to 70 and 69
    // in 26.6, strength(2048) at that ppem).
    let aliased_overscale = run.simulations & 1 != 0
        && aa == AA_ALIASED
        && params.grid_fit == 1
        && uniform_axis
        && raster::bold_amount((run.em_size as f64 * dt[0].abs().max(dt[3].abs())).round() as i32).0 >= 2;
    // strength(upm) is threaded into `unhinted_glyph_coverage` (the off-axis/disabled branch) for these
    // runs; rotated/non-uniform aliased keeps the post-pass OR-dilate (`aliased_post_dilate`).
    // Upright GDI-classic bold with grid fit disabled embolds the design outline at strength(upm) above a one-pixel
    // boldX, as its hinted path embolds the hinted outline; at one pixel it keeps the bitmap dilate (VM: Leelawadee
    // UI at ppem 58.4, ClearType and grayscale, upright and oblique; Trebuchet, Tahoma and Georgia at 12.7 to 37.5).
    let gdi_classic_design_bold = params.grid_fit == 1
        && aa != AA_ALIASED
        && gdi_classic_upright_bold(run, mode, dt)
        && !gdi_classic_bitmap_bold(run, mode, dt);
    let off_axis_bold = (aliased_overscale || gdi_classic_design_bold)
        .then(|| outline_embold::strength(font.head().map(|h| h.units_per_em() as i32).unwrap_or(2048)));
    let aliased_post_dilate = !aliased_overscale;
    let mut downsampled = Vec::new();
    // The GDI measuring modes' ClearType is GDI-compatible: drawn at level 1 whatever the params say (VM, sc33).
    let _flipped = raster::Flipped::set(dt[0] * dt[3] - dt[1] * dt[2] < 0.0);
    let _ct_level = raster::CtLevel::set(if matches!(mode, MODE_GDI_CLASSIC | MODE_GDI_NATURAL) { 1.0 } else { params.cleartype_level });
    let tex = if overscale_bold {
        // Overscale-bold: embolden the unhinted design outline with strength(upm), scale to device + scan.
        let upm = font.head().map(|h| h.units_per_em() as i32).unwrap_or(2048);
        let mut snapped = *run;
        snapped.baseline_origin = snap_baseline(run.baseline_origin, dt, run.no_snap);
        let group = |boxes: &[Option<[i32; 4]>], slack: i32| cache_groups(run, font, &fg, params, target, slack, boxes.len(), || boxes.to_vec());
        let mut texs = unhinted_glyph_coverage_grouped(dt, &snapped, font, aa, mode, params.rasterizer, matches!(fg, TextFg::Brush(_)), Some(outline_embold::strength(upm)), Some(&group));
        let first = if texs.is_empty() { raster::AlphaTexture::empty() } else { texs.remove(0) };
        downsampled = texs;
        first
    } else if off_axis_disabled || rotated {
        // DISABLED: unhinted outline (the scan converter's verified input). DrawGlyphRun still snaps
        // the baseline to the pixel grid, so snap before handing the run to the scan converter.
        // The glyph cache groups the run as it does a hinted one (VM r25 30900: Segoe UI's joined Arabic under a transpose
        // composites glyph by glyph where the joins overlap).
        let mut snapped = *run;
        snapped.baseline_origin = snap_baseline(run.baseline_origin, dt, run.no_snap);
        let group = |boxes: &[Option<[i32; 4]>], slack: i32| cache_groups(run, font, &fg, params, target, slack, boxes.len(), || boxes.to_vec());
        let mut texs = unhinted_glyph_coverage_grouped(dt, &snapped, font, aa, mode, params.rasterizer, matches!(fg, TextFg::Brush(_)), off_axis_bold, Some(&group));
        let first = if texs.is_empty() { raster::AlphaTexture::empty() } else { texs.remove(0) };
        downsampled = texs;
        first
    } else {
        // DWRITE_FONT_SIMULATIONS_BOLD: native emboldens the rasterized glyph bitmap (not the outline),
        // per AA mode. Grayscale emboldens the device-res count buffer in the scan (the grayscale embolden,
        // threaded here); aliased is a post-pass below; ClearType is not yet ported. Amount from the
        // device ppem (`raster::bold_amount`).
        // Grayscale synthetic bold is emboldened on the hinted OUTLINE (`embolden_hinted` in
        // `hinted_contours_with_modes`), so the resolved-coverage OR-dilate is suppressed for grayscale;
        // ClearType and aliased still dilate the resolved grid.
        // The bitmap bold turns with a flipped or quarter-turned glyph ([`raster::bold_dir`]).
        // GDI-classic synthetic bold at a one-pixel boldX, ClearType and grayscale, dilates the scanned bits one
        // pixel right and none up ([`gdi_classic_bitmap_bold`]).
        // NATURAL and GDI_NATURAL do the same ([`natural_bitmap_bold`]). At a small size the dilation is the
        // scaler's run darkening ([`raster::BoldPpem`]).
        let bitmap_bold = (gdi_classic_bitmap_bold(run, mode, dt) || natural_bitmap_bold(run, mode, dt)) && aa != AA_ALIASED;
        let _bold_ppem = raster::BoldPpem::set(bitmap_bold_ppem(run, mode, aa, params.grid_fit, dt, bitmap_bold));
        let bold = if bitmap_bold {
            raster::bold_dir(dt, (1, 0))
        } else if run.simulations & 1 != 0
            && aa != AA_GRAYSCALE
            && !(aa == AA_CLEARTYPE && gdi_classic_upright_bold(run, mode, dt))
            && !tilted_ct_bold(run, dt, aa)
            && !natural_ct_outline_bold(run, mode, dt, aa, params.grid_fit)
        {
            raster::bold_dir(dt, raster::bold_amount((run.em_size as f64 * dt[..4].iter().fold(0f64, |m, v| m.max(v.abs()))).round() as i32))
        } else {
            (0, 0)
        };
        let mut glyph_modes = Vec::new();
        let mut glyph_phases = Vec::new();
        let hinted = hinted_contours_with_modes(dt, run, font, mode, aa, params.rasterizer, params.grid_fit, Some(&mut glyph_modes), Some(&mut glyph_phases));
        match hinted {
            Some((c, _)) if !glyph_modes.is_empty() => {
                let ds = aa == AA_GRAYSCALE && mode == MODE_NATURAL_SYMMETRIC_DOWNSAMPLED;
                let (sx, sy) = match kind {
                    raster::Kind::Aliased => (1, 1),
                    raster::Kind::ClearType | raster::Kind::ClearTypeH => (6, if horizontal_only(mode) { 1 } else { 5 }),
                    _ if horizontal_only(mode) => (8, 1),
                    _ => (4, 4),
                };
                let groups = cache_groups(run, font, &fg, params, target, (sy > 1) as i32, glyph_modes.len(), || {
                    let boxes = raster::glyph_os_boxes(&c, &glyph_modes, sx, sy, matches!(kind, raster::Kind::ClearType | raster::Kind::ClearTypeH));
                    // A thin face's cached bitmap packs every filled span one sample longer ([`merge_widen`]), so its box
                    // reaches one sample further right and meets a neighbour one sample away (VM: Courier New 'ος' at 2.77 px,
                    // NATURAL 8×1, draws as one texture while its neighbours two samples apart do not).
                    let boxes: Vec<_> = if sx > 1 && merge_widen(run) { boxes.into_iter().map(|b| b.map(|[l, t, r, b]| [l, t, r + 1, b])).collect() } else { boxes };
                    // A bitmap-bold glyph's bitmap is its dilated one, so it meets its neighbours by that.
                    if bitmap_bold && bold != (0, 0) {
                        let (gx, gy) = (bold.0 * sx, bold.1 * sy);
                        boxes.into_iter().map(|b| b.map(|[l, t, r, b]| [l + gx.min(0), t - gy.max(0), r + gx.max(0), b - gy.min(0)])).collect()
                    } else {
                        boxes
                    }
                });
                let widen = sx > 1 && merge_widen(run);
                let mut texs: Vec<_> = groups.iter().map(|g| {
                    let (mut gc, mut gm, mut gp) = (Vec::new(), Vec::new(), Vec::new());
                    for &i in g {
                        let (r, m) = &glyph_modes[i];
                        let s = gc.len();
                        gc.extend_from_slice(&c[r.clone()]);
                        gm.push((s..gc.len(), *m));
                        gp.push(glyph_phases.get(i).copied().unwrap_or(false));
                    }
                    if ds { raster::coverage_downsampled(&gc, &gm, &gp, true, widen) } else { raster::coverage_glyphs(&gc, &gm, kind, sx, sy, bold, widen) }
                }).collect();
                downsampled = texs.split_off(1);
                texs.pop().unwrap_or_else(raster::AlphaTexture::empty)
            }
            // Grayscale NATURAL (and the GDI modes) scan-convert on an 8×1 grid, not the 4×4
            // NATURAL_SYMMETRIC uses (`dword_1801D4B20[8·aa+mode]`: 0x18 ⇒ 8×1, 0x44 ⇒ 4×4); same
            // hinted geometry, different oversample. ClearType/aliased keep `coverage_outline`.
            Some((c, drop)) if matches!(kind, raster::Kind::Grayscale) && horizontal_only(mode) => {
                raster::coverage_box(&c, drop, 8, 1, bold, false)
            }
            // ClearType with no vertical AA (GDI modes, plain NATURAL): 6×1, not the 6×5 of
            // NATURAL_SYMMETRIC — native's horizontal-only ClearType (see `raster::Kind::ClearTypeH`).
            Some((c, drop)) if matches!(kind, raster::Kind::ClearType) && horizontal_only(mode) => {
                raster::coverage_outline(&c, drop, raster::Kind::ClearTypeH, bold)
            }
            Some((c, drop)) => raster::coverage_outline(&c, drop, kind, bold),
            None => return false,
        }
    };
    // Aliased bold: OR-dilate the final 1-bit texture (the aliased embolden); the grayscale embolden already
    // ran inside the scan above, and ClearType is not yet ported.
    // The unhinted surfaces embolden inside the scan (`unhinted_glyph_coverage`, the design outline for a
    // grid-fit-disabled uniform run); only a hinted aliased run is emboldened here.
    let hinted = !overscale_bold && !(off_axis_disabled || rotated);
    let device_em_max = run.em_size as f64 * dt[..4].iter().fold(0f64, |m, v| m.max(v.abs()));
    let (tex, downsampled) = if run.simulations & 1 != 0 && aa == AA_ALIASED && hinted && aliased_post_dilate && !aliased_outline_bold(run, aa, device_em_max) && !alias_ts_outline(run, aa, mode, dt) {
        let ppem = device_em_max.round() as i32;
        let (bx, by) = raster::bold_dir(dt, raster::bold_amount(ppem));
        (raster::embolden_aliased(&tex, bx, by), downsampled.iter().map(|t| raster::embolden_aliased(t, bx, by)).collect())
    } else {
        (tex, downsampled)
    };
    let thin = thin_font_boost(run.font_data, run.face_index);
    for tex in std::iter::once(tex).chain(downsampled) {
        if tex.width() <= 0 || tex.height() <= 0 {
            continue; // nothing covered (e.g. a space)
        }
        let alpha = blend::AlphaTexture { bytes: &tex.data, bounds: [tex.left, tex.top, tex.right, tex.bottom] };
        // Native dwrite!GetAlphaBlendParamsWithLightOnDarkAdjustment selects the system
        // compatible gamma (1.2 on the oracle) and zero contrast for GDI ClearType.
        let compatible = aa == AA_CLEARTYPE && matches!(mode, MODE_GDI_CLASSIC | MODE_GDI_NATURAL);
        let gamma = if compatible { 1.2 } else { params.gamma };
        // Grayscale text takes the params' grayscale enhanced contrast, ClearType the enhanced contrast (VM: the two set
        // apart, atlasplan sc32 scene 4).
        let contrast = if compatible { 0.0 } else if aa == AA_CLEARTYPE { params.enhanced_contrast } else { params.grayscale_contrast };
        match fg {
            TextFg::Solid(color) => {
                let mut bp = blend::BlendParams::resolve(color, gamma, contrast, params.cleartype_level);
                (bp.contrast, bp.enhanced_contrast) = (bp.contrast + thin, bp.enhanced_contrast + thin);
                if aa == AA_CLEARTYPE {
                    blend::blend_cleartype(target, alpha, color, &bp, params.blend);
                } else {
                    blend::blend_grayscale(target, alpha, color, &bp, aa, params.blend);
                }
            }
            TextFg::Brush(b) => {
                // `gamma_row` is colour-independent; the contrast/intensity terms are per-pixel (in the blend).
                let gamma_row = blend::BlendParams::resolve([0.0, 0.0, 0.0, 1.0], gamma, contrast, params.cleartype_level).gamma_row;
                if aa == AA_CLEARTYPE {
                    blend::blend_cleartype_brush(target, alpha, b, gamma_row, contrast + thin, params.blend, params.clip_px);
                } else {
                    blend::blend_grayscale_brush(target, alpha, b, gamma_row, contrast + thin, aa, params.blend, params.clip_px);
                }
            }
        }
    }
    true
}

/// DirectWrite's merge ranges: without a MERG table the whole run is one range (the font-face flag 0x11 the
/// builder defaults to). With one, consecutive glyphs chain while the class pair's entry says merge or group
/// (LTR 0x01/0x02, RTL 0x10/0x20), a subordinate second glyph (0x04/0x40) keeping the chain's class, and only a
/// chain that merged at least once becomes a range.
fn merge_ranges(font: &FontRef, run: &GlyphRun) -> Vec<std::ops::Range<usize>> {
    let n = run.glyphs.len();
    let Some(t) = font.table_data(skrifa::raw::types::Tag::new(b"MERG")).map(|d| d.as_bytes().to_vec()) else { return vec![0..n] };
    let rd = |o: usize| t.get(o..o + 2).map_or(0, |b| u16::from_be_bytes([b[0], b[1]]) as usize);
    let (count, data, ndefs, defs) = (rd(2), rd(4), rd(6), rd(8));
    let class = |g: u16| -> usize {
        let g = g as usize;
        for k in 0..ndefs {
            let o = rd(defs + 2 * k);
            if rd(o) == 1 {
                let (start, len) = (rd(o + 2), rd(o + 4));
                if g >= start && g < start + len {
                    return rd(o + 6 + 2 * (g - start));
                }
            } else {
                for r in 0..rd(o + 2) {
                    let e = o + 4 + 6 * r;
                    if g >= rd(e) && g <= rd(e + 2) {
                        return rd(e + 4);
                    }
                }
            }
        }
        0
    };
    let rtl = run.bidi_level & 1 != 0;
    let (link, merge, sub) = if rtl { (0x30, 0x10, 0x40) } else { (0x03, 0x01, 0x04) };
    let mut out = Vec::new();
    let (mut start, mut merged) = (0, false);
    let mut prev = run.glyphs.first().map_or(0, |g| class(g.id));
    for i in 1..n {
        let cur = class(run.glyphs[i].id);
        let e = if prev < count && cur < count { t.get(data + prev * count + cur).copied().unwrap_or(0) } else { 0 };
        if e & link != 0 {
            merged |= e & merge != 0;
            if e & sub == 0 {
                prev = cur;
            }
        } else {
            if merged {
                out.push(start..i);
                merged = false;
            }
            start = i;
            prev = cur;
        }
    }
    if merged {
        out.push(start..n);
    }
    out
}

/// d2d1's merge rects: glyphs whose oversampled bitmap rects meet (x inclusive, y with `slack` rows)
/// join one merge rect; every other glyph stands alone.
/// d2d1's glyph cache draws each glyph as its own blended quad, except the glyphs it merges because their oversampled
/// bitmap rects meet, which are scanned together (DirectWrite ORs their bits into one run bitmap). The cache needs a
/// solid brush, SOURCE_OVER, and a target format whose table entry has flag 0x2000 (A8, R8, R8G8, R8G8B8A8, B8G8R8A8,
/// R10G10B10A2 UNORM); otherwise the whole run merges. An opaque brush asks DWrite for merge ranges (the font's MERG
/// table, [`merge_ranges`]); a translucent one makes the whole run one range (native's aliasing-mode pass, alpha ≠ 1).
/// `n` glyph entries with their oversampled `boxes` (computed only when the cache is on); `slack` is one when y is
/// oversampled. Returns the single glyphs in glyph order, then the merged groups in first-glyph order, the order d2d1
/// draws them: the cached quads first, the merged bitmaps after, so where a lone glyph and a merged one share a pixel
/// the merged one blends last (VM r25 30121: 'abc' bold under a y-flip, a lone 'c' beside a merged 'ab').
/// ponytail: the entries are one per run glyph whenever all of them scan; otherwise the whole run is one range.
#[allow(clippy::too_many_arguments)]
fn cache_groups(run: &GlyphRun, font: &FontRef, fg: &TextFg, params: &RenderingParams, target: &PixelBuffer, slack: i32, n: usize, boxes: impl FnOnce() -> Vec<Option<[i32; 4]>>) -> Vec<Vec<usize>> {
    let cached = matches!(fg, TextFg::Solid(_))
        && matches!(params.blend, crate::pixels::Blend::SourceOver)
        && matches!(target.desc.format.0, 24 | 28 | 49 | 61 | 65 | 87);
    if !cached {
        return vec![(0..n).collect()];
    }
    let boxes = boxes();
    let opaque = matches!(fg, TextFg::Solid(c) if c[3] == 1.0);
    let ranges = if opaque && n == run.glyphs.len() { merge_ranges(font, run) } else { vec![0..n] };
    let mut groups = Vec::new();
    let mut next = 0;
    for r in ranges {
        groups.extend((next..r.start).map(|i| vec![i]));
        groups.extend(merge_groups(&boxes[r.clone()], slack).into_iter().map(|g| g.into_iter().map(|i| i + r.start).collect::<Vec<_>>()));
        next = r.end;
    }
    groups.extend((next..n).map(|i| vec![i]));
    groups.sort_by_key(|g| (g.len() > 1, g[0]));
    groups
}

fn merge_groups(boxes: &[Option<[i32; 4]>], slack: i32) -> Vec<Vec<usize>> {
    let mut groups: Vec<(Vec<usize>, Option<[i32; 4]>)> = Vec::new();
    for (i, &b) in boxes.iter().enumerate() {
        let (mut ids, mut rect) = (vec![i], b);
        if let Some(b) = b {
            let meets = |r: &[i32; 4]| r[0] <= b[2] && b[0] <= r[2] && r[1] < b[3] + slack && b[1] < r[3] + slack;
            groups.retain(|(g, r)| {
                let hit = r.is_some_and(|r| meets(&r));
                if hit {
                    ids.extend(g);
                    rect = rect.zip(*r).map(|(a, r)| [a[0].min(r[0]), a[1].min(r[1]), a[2].max(r[2]), a[3].max(r[3])]);
                }
                !hit
            });
        }
        ids.sort_unstable();
        groups.push((ids, rect));
    }
    groups.sort_by_key(|(g, _)| g[0]);
    groups.into_iter().map(|(g, _)| g).collect()
}

/// d2d1's ComputeMergeRects over float device rects: within each merge range, a sweep in glyph order joins a glyph to
/// what it meets (inclusive) into merge rects, returned as glyph index ranges; `skip` glyphs (bounds cut to nothing)
/// take no part. Ported as native runs it, quirks included: the "previous glyph" starts at index 0 rather than at
/// the range's first glyph, a glyph meeting the run's extent only through that previous glyph pairs with it or
/// extends the last merge rect, and a glyph reaching back past it folds every glyph and merge rect from the first
/// one it meets into one.
fn merge_rects(rects: &[[f32; 4]], skip: &[bool], ranges: &[std::ops::Range<usize>]) -> Vec<std::ops::Range<usize>> {
    type Out = Vec<([f32; 4], std::ops::Range<usize>)>;
    let meets = |a: &[f32; 4], b: &[f32; 4]| b[2] >= a[0] && a[2] >= b[0] && b[3] >= a[1] && a[3] >= b[1];
    let union = |a: [f32; 4], b: &[f32; 4]| [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])];
    let next_start = |out: &Out, k: usize| out.get(k).map_or(usize::MAX, |m| m.1.start);
    // One step of the backward walk: the next glyph, or past the merge rect it has reached.
    let step = |i: &mut usize, k: &mut usize, ms: &mut usize, out: &Out| {
        if *i < *ms {
            *i += 1;
        } else {
            *k += 1;
            *i = out[*k - 1].1.end;
            *ms = next_start(out, *k);
        }
    };
    let mut out: Out = Vec::new();
    for r in ranges {
        if r.end < 2 {
            continue;
        }
        let mut first = r.start;
        while first + 2 <= r.end && skip[first] {
            first += 1;
        }
        if first + 2 > r.end {
            continue;
        }
        let range_out = out.len();
        let (mut min_l, mut max_r) = (rects[first][0], rects[first][2]);
        let (mut prev_l, mut prev_r) = (-f32::MAX, -f32::MAX);
        let mut last = 0usize;
        for g in first + 1..r.end {
            if skip[g] {
                continue;
            }
            let b = rects[g];
            if b[0] > max_r {
                (prev_r, max_r, prev_l) = (max_r, b[2], min_l);
                last = g;
                continue;
            }
            if min_l > b[2] {
                (prev_l, prev_r, min_l) = (min_l, max_r, b[0]);
                last = g;
                continue;
            }
            if b[0] > prev_r || prev_l > b[2] {
                let own = out.len() > range_out;
                match out.last_mut() {
                    Some((mr, mg)) if own && mg.end > last => {
                        if meets(&b, mr) {
                            mg.end = g + 1;
                            *mr = union(*mr, &b);
                        }
                    }
                    _ => {
                        let p = rects[last];
                        if meets(&b, &p) {
                            out.push((union(p, &b), last..g + 1));
                        }
                    }
                }
            } else {
                let (mut i, mut k) = (r.start, range_out);
                let mut ms = next_start(&out, k);
                while i < g && i < ms && skip[i] {
                    step(&mut i, &mut k, &mut ms, &out);
                }
                let mut hit = None;
                while i < g {
                    let o = if i < ms { rects[i] } else { out[k].0 };
                    if meets(&b, &o) {
                        hit = Some(if i < ms { i } else { out[k].1.start });
                        break;
                    }
                    loop {
                        step(&mut i, &mut k, &mut ms, &out);
                        if i >= g || i >= ms || !skip[i] {
                            break;
                        }
                    }
                }
                if let Some(start) = hit {
                    let keep = k;
                    let mut rect = b;
                    while i < g {
                        rect = union(rect, &if i < ms { rects[i] } else { out[k].0 });
                        loop {
                            step(&mut i, &mut k, &mut ms, &out);
                            if i >= g || i >= ms || !skip[i] {
                                break;
                            }
                        }
                    }
                    out.truncate(keep);
                    out.push((rect, start..g + 1));
                }
            }
            (prev_l, min_l, prev_r, max_r) = (min_l, min_l.min(b[0]), max_r, max_r.max(b[2]));
            last = g;
        }
    }
    out.into_iter().map(|(_, g)| g).collect()
}

/// Grid-fit every glyph of the run through the TrueType hinting interpreter and lay the hinted
/// outlines out in device space for the scan converter. The hinter bakes the device em scale, so
/// only the pen translation is applied here: the baseline y is snapped to the pixel grid (R2), the
/// pen x keeps its sub-pixel phase (subpixel modes) or snaps to the pixel grid (aliased /
/// GDI-classic), and the hinter's y-up F26Dot6 points are flipped to device y-down. `aa` and `mode`
/// pick the hinting mode: aliased → full classic grid-fit, the GDI measuring modes → the
/// GDI-compatible grid-fit, ClearType/grayscale → the smooth (y-only) grid-fit.
#[cfg_attr(not(test), allow(dead_code))] // the unit tests and GDI-raster diagnostics
pub(crate) fn hinted_contours(dt: &[f64; 6], run: &GlyphRun, font: &FontRef, mode: u32, aa: u32) -> Option<(Vec<(Vec<[f64; 2]>, Vec<bool>)>, u16)> {
    hinted_contours_using(dt, run, font, mode, aa, hinting::Rasterizer::FontCache, false)
}

pub(crate) fn hinted_contours_using(dt: &[f64; 6], run: &GlyphRun, font: &FontRef, mode: u32, aa: u32, rasterizer: hinting::Rasterizer, fit_on: bool) -> Option<(Vec<(Vec<[f64; 2]>, Vec<bool>)>, u16)> {
    hinted_contours_with_modes(dt, run, font, mode, aa, rasterizer, if fit_on { 2 } else { 0 }, None, None)
}

/// The per-axis scales `(x, y)` native hints a run at, or `None` when it does not hint by default (an
/// explicit grid-fit ENABLED still hints a generic matrix at [`axis_scales`]; the callers check it).
/// GetRecommendedRenderingMode disables grid-fit only for a GENERIC matrix (every 2×2 component nonzero; VM
/// RECMODE: a 30° rotation, [2.56,.34,−.74,1.73]); a quarter turn or a one-axis shear stays grid-fitted. The
/// scaler reduces the matrix to per-axis scales (x = max(|m11|,|m12|),
/// y = max(|m21|,|m22|)), hints there (MPPEM the larger, the CVT stretched between them) and maps the
/// hinted outline through the residual.
pub(crate) fn hint_scale(dt: &[f64; 6]) -> Option<(f64, f64)> {
    (!dt[..4].iter().all(|&v| v != 0.0)).then(|| axis_scales(dt))
}

/// The TrueType scaler's per-axis scales for any matrix, generic ones included: x = max(|m11|,|m12|),
/// y = max(|m21|,|m22|); its ppem is the rounded y scale.
fn axis_scales(dt: &[f64; 6]) -> (f64, f64) {
    (dt[0].abs().max(dt[1].abs()), dt[2].abs().max(dt[3].abs()))
}

/// Native's GETINFO rotate/stretch flags for a hinted matrix (the scaler's rotate/stretch state), on the rows
/// normalized by the hint scale: orthogonal rows are stretched iff their lengths differ and rotated iff
/// m11 ≠ 0 and either off-diagonal is; non-orthogonal rows are rotated and stretched when both m11 and
/// m12 are nonzero, else stretched unless the first row is unit and the determinant is 1.
fn rotate_stretch(dt: &[f64; 6], s: f64) -> (bool, bool) {
    // VM: both one-axis shears ([1,0,.5,1] and [1,.5,0,1]) hint exactly like the identity, so read the rows
    // with the zero off-diagonal second (either orientation), which makes a unit-row, det-1 shear neither.
    let (a, b, c, d) = if dt[1] == 0.0 { (dt[0] / s, dt[1] / s, dt[2] / s, dt[3] / s) } else { (dt[0] / s, dt[2] / s, dt[1] / s, dt[3] / s) };
    let near = |x: f64, y: f64| (x - y).abs() < 1.0 / 65536.0;
    if near(a * c + b * d, 0.0) {
        return (a != 0.0 && (b != 0.0 || c != 0.0), !near(a * a + b * b, c * c + d * d));
    }
    if a != 0.0 && b != 0.0 {
        return (true, true);
    }
    (false, !(near(a * a + b * b, 1.0) && near(a * d - b * c, 1.0)))
}

/// The SCANCTRL dropout mode for the grid-fit-disabled scan path (`render_analysis`'s unhinted/collapsed
/// branch). Native still runs the font's `prep` for a grid-fit-disabled glyph and its per-glyph
/// dropout-control check evaluates the resolved SCANCTRL word at the device ppem, so the fill uses the
/// font's dropout, not the scaler's [`raster::DEFAULT_DROPOUT`]. For a font whose SCANCTRL enables
/// dropout by ppem threshold (Arial, Times), the gate (`Retained::dropout_on`) is thus ppem-driven: a
/// full-size rotated run stays quiet because its device ppem exceeds the threshold (Arial @19 px > 17),
/// while a det~0 collapsed fold fires because the collapse drops the device ppem below it (r20 24013
/// @≈4 px ≤ 44). The query instance hints upright (`is_rotated: false`) to match native's `prep` — see
/// the config below.
/// Scan-convert an UNHINTED (grid-fit DISABLED) glyph run into its coverage texture, honouring the
/// rendering `mode` (`DWRITE_RENDERING_MODE1`, == the internal `MODE_*` numbering) and `aa`
/// (`D2D1_TEXT_ANTIALIAS_MODE`). This is the single implementation shared by `render_analysis` (the
/// D2D DrawGlyphRun path) and the `IDWriteGlyphRunAnalysis` COM shim (`dwrite-ffi`): the mode selects
/// the oversample grid (GDI/NATURAL are horizontal-only 6×1 / 8×1, NATURAL_SYMMETRIC is 6×5 / 4×4), and
/// the dropout control comes from the font's grid-fit-disabled scan-control exactly as the blend path
/// computes it. `is_brush` only affects the NATURAL_SYMMETRIC_DOWNSAMPLED grayscale downsample. The run
/// baseline is taken as given (callers snap it first when their API snaps).
pub fn unhinted_glyph_coverage(dt: &[f64; 6], run: &GlyphRun, font: &FontRef, aa: u32, mode: u32, rasterizer: hinting::Rasterizer, is_brush: bool, design_bold: Option<(i32, i32)>) -> raster::AlphaTexture {
    unhinted_glyph_coverage_grouped(dt, run, font, aa, mode, rasterizer, is_brush, design_bold, None).pop().unwrap_or_else(raster::AlphaTexture::empty)
}

/// [`unhinted_glyph_coverage`] split into the glyph cache's groups ([`raster::unhinted_grouped`]).
#[allow(clippy::too_many_arguments)]
fn unhinted_glyph_coverage_grouped(dt: &[f64; 6], run: &GlyphRun, font: &FontRef, aa: u32, mode: u32, rasterizer: hinting::Rasterizer, is_brush: bool, design_bold: Option<(i32, i32)>, group: Option<raster::GroupFn>) -> Vec<raster::AlphaTexture> {
    let dt = &raster::scaler_matrix(dt);
    // Off the axes (a rotation, a quarter turn or a one-axis shear) the bold goes into the design outline at
    // strength(upm), not the bitmap (VM: bold runs under rotations, quarter turns and shears exact with it, at
    // every mode).
    let design_bold = design_bold.or_else(|| {
        (run.simulations & 1 != 0 && (dt[1] != 0.0 || dt[2] != 0.0)).then(|| outline_embold::strength(font.head().map_or(2048, |h| h.units_per_em() as i32)))
    });
    let kind = match aa {
        AA_CLEARTYPE => raster::Kind::ClearType,
        AA_ALIASED => raster::Kind::Aliased,
        _ => raster::Kind::Grayscale,
    };
    let dropout = disabled_dropout_mode(dt, run, font, mode, aa, rasterizer);
    raster::unhinted_grouped(
        dt,
        run,
        font,
        kind,
        horizontal_only(mode),
        matches!(mode, MODE_GDI_CLASSIC | MODE_GDI_NATURAL),
        (mode == MODE_NATURAL_SYMMETRIC_DOWNSAMPLED).then_some(is_brush),
        dropout,
        design_bold,
        merge_widen(run),
        group,
    )
}

fn disabled_dropout_mode(dt: &[f64; 6], run: &GlyphRun, font: &FontRef, mode: u32, aa: u32, rasterizer: hinting::Rasterizer) -> u16 {
    // The prep that decides the dropout runs at the scaler's axis scales even for a generic matrix (VM: Arial
    // under an 80° turn, 48 px, no dropout pass; max(|m11|,|m22|) read it as 9 px, under the threshold).
    let (sx, sy) = axis_scales(dt);
    let scale = sx.max(sy);
    let (em_x, em_y) = ((run.em_size as f64 * sx).round(), (run.em_size as f64 * sy).round());
    if em_x.min(em_y) < 1.0 {
        return raster::DEFAULT_DROPOUT;
    }
    let gdi_natural = mode == MODE_GDI_NATURAL || (mode == MODE_GDI_CLASSIC && dt[1] != 0.0);
    let hmode = match (mode, aa) {
        (MODE_GDI_NATURAL, AA_ALIASED) | (MODE_GDI_CLASSIC, AA_ALIASED) => hinting::Mode::GdiClassic,
        (MODE_GDI_CLASSIC | MODE_GDI_NATURAL, _) if gdi_natural => hinting::Mode::GdiNatural,
        (MODE_GDI_CLASSIC, _) => hinting::Mode::GdiClassic,
        (_, AA_ALIASED) => hinting::Mode::Aliased,
        (_, AA_CLEARTYPE) => hinting::Mode::ClearType,
        _ => hinting::Mode::Grayscale,
    };
    let cfg = hinting::Config {
        rasterizer,
        mode: hmode,
        symmetric: matches!(mode, MODE_NATURAL_SYMMETRIC | MODE_NATURAL_SYMMETRIC_DOWNSAMPLED),
        grid_fit: false,
        bgr: false,
        // Native hints a grid-fit-disabled glyph in its upright design frame (the device transform is
        // applied to the outline, not the hinting grid), so its `prep`'s rotation and stretch environment
        // queries both report false regardless of the device transform — DirectWrite's scaler always
        // configures the interpreter that way, confirmed on the VM for a 15° Arial run and a rank-1
        // collapsed Times run. Passing the device flags sent `prep` down a different branch (Arial
        // SCANCTRL 0x1ff "dropout always on" vs the real 0x111, threshold 17), over-enabling dropout on
        // every rotated/skewed/stretched run.
        is_rotated: false,
        is_stretched: false,
        axes_swapped: dt[0] == 0.0,
        // Grid-fit is off on this probe, so composite offsets take the font-unit path (no grid round) and
        // this never reaches the output; it only gates the dropout-mode read.
        rotation_kind: 0,
        em_y: (em_y != em_x).then_some(em_y),
        version: 42,
        gdi_render: aa != AA_ALIASED && matches!(mode, MODE_GDI_CLASSIC | MODE_GDI_NATURAL),
        cleartype: aa == AA_CLEARTYPE,
        subpixel_advance: gdi_natural && aa != AA_ALIASED,
        bold_fu: 0,
    };
    match hinting::Instance::new_instanced(font, em_x, cfg, run.coords) {
        Some(inst) => {
            let (enabled, scan_type) = inst.scan_control();
            raster::dropout_mode(enabled, scan_type)
        }
        None => raster::DEFAULT_DROPOUT,
    }
}

/// Diagnostic (text scan task), behind the off-by-default `inject-contours` feature so no prod build
/// carries it: parse the flat contour-injection file named by D2D_INJECT_CONTOURS. One glyph per line;
/// contours separated by `|`; points space-separated as `x,y,o` (fsc device units: CT x = device×384,
/// gray x = device×512, y = device×64 y-up, o = on-curve 1/0). Returns glyph→contour→(x,y,on); the
/// caller places each at OUR pen origin, so native's hinted SHAPE runs through our raster at our
/// positions — splitting a hinting residual from a raster one. `None` when the var is unset.
#[cfg(feature = "inject-contours")]
fn inject_contours() -> Option<Vec<Vec<Vec<(f64, f64, bool)>>>> {
    let s = std::fs::read_to_string(std::env::var_os("D2D_INJECT_CONTOURS")?).ok()?;
    Some(
        s.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|line| {
                line.split('|')
                    .map(|c| {
                        c.split_whitespace()
                            .filter_map(|p| {
                                let mut it = p.split(',');
                                let x: f64 = it.next()?.parse().ok()?;
                                let y: f64 = it.next()?.parse().ok()?;
                                Some((x, y, it.next() == Some("1")))
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect(),
    )
}

/// `fit_on`: the params ask for grid-fitting explicitly (DWRITE_GRID_FIT_MODE_ENABLED), which DirectWrite
/// passes through without consulting the gasp table.
#[allow(clippy::too_many_arguments)]
fn hinted_contours_with_modes(dt: &[f64; 6], run: &GlyphRun, font: &FontRef, mode: u32, aa: u32, rasterizer: hinting::Rasterizer, fit: u32, mut glyph_modes: Option<&mut Vec<(std::ops::Range<usize>,u16)>>, mut glyph_phases: Option<&mut Vec<bool>>) -> Option<(Vec<(Vec<[f64; 2]>, Vec<bool>)>, u16)> {
    // The oblique simulation's shear joins the matrix before the scaler reduces it to axis scales (the glyph's
    // y axis gains skew times its x axis), so under a world shear it changes the hinting scales (VM: Verdana
    // under a vertical shear hints at ppem 17, not 14).
    let hm = if run.simulations & 2 != 0 { [dt[0], dt[1], dt[2] - OBLIQUE_SKEW * dt[0], dt[3] - OBLIQUE_SKEW * dt[1], dt[4], dt[5]] } else { *dt };
    let (sx, sy) = axis_scales(&hm);
    let scale = sx.max(sy);
    // TrueType hinting runs at an integer ppem: native rounds the device em before scaling/grid-fit
    // (DirectWrite's recommended rendering mode: `(int)em` then `+1` when the
    // fraction is exactly 0.5 — i.e. round-half-up, since the device em is always n or n.5 for the
    // integer DPI scales). Hinting the fractional 19.5 px of the 144-DPI scene instead of 20 shifted
    // the whole glyph (text-09); rounding snaps it to native. A no-op at 96 DPI (em already integer). A face that
    // does not force integer ppems keeps its fractional em where DirectWrite does not round it ([`fractional_em`]).
    let device_em = (run.em_size as f64 * scale).round();
    let (em_x, em_y) = ((run.em_size as f64 * sx).round(), (run.em_size as f64 * sy).round());
    // An em that rounds to 0 device ppem draws nothing. Grid-fitted, it fails dwrite's scaler
    // (native's scaler fails a zero integer scale, so the rasterizer yields no bitmap;
    // r14 19531: ClearType under a 1e-6 world transform). With grid-fit disabled the fill runs on an empty
    // rect (VM: Segoe UI 'T'/'TM' NATURAL at em 0.2/0.45, an empty fill rect, no ink).
    if em_x.min(em_y) < 1.0 {
        return Some((Vec::new(), 2));
    }
    // In the smooth modes the rounding is the font's own request: head flags bit 3 ("force ppem to integer
    // values"). A font without it (Segoe UI Emoji) scales at the exact device em there, hinted or not (VM:
    // 'lHiWg' at device em 5.6, gray and ClearType, 96 and 192 DPI: Segoe UI Emoji matches only unrounded,
    // Segoe UI and Consolas only rounded). The GDI modes and aliased text keep the whole ppem for such a face
    // under a uniform scale (r25 30118, 30329, 30476, 30891: Segoe UI Emoji under GDI-natural or aliased), and
    // only there ([`fractional_em`]).
    let (em_x, em_y) = if fractional_em(dt, run, font, if aa == AA_ALIASED { MODE_ALIASED } else { mode }) { (run.em_size as f64 * sx, run.em_size as f64 * sy) } else { (em_x, em_y) };
    // Native's transform setup keeps compatible widths only while the glyph's x axis has no device-y part
    // (TransMatrix t01 = m12 = 0). Without them antialiased GDI-classic carries GDI-natural's pass flags: one
    // pass, no metric pass (VM, Arial 'H' under [1,.5,0,1] or [0,1,-1,0]: a single pass with flags 0x81 / 0x85).
    let gdi_natural = mode == MODE_GDI_NATURAL || (mode == MODE_GDI_CLASSIC && dt[1] != 0.0);
    let hmode = match (mode, aa) {
        (MODE_GDI_NATURAL, AA_ALIASED) => hinting::Mode::GdiClassic,
        (MODE_GDI_CLASSIC, AA_ALIASED) => hinting::Mode::GdiClassic,
        (MODE_GDI_CLASSIC | MODE_GDI_NATURAL, _) if gdi_natural => hinting::Mode::GdiNatural,
        (MODE_GDI_CLASSIC, _) => hinting::Mode::GdiClassic,
        (_, AA_ALIASED) => hinting::Mode::Aliased,
        (_, AA_CLEARTYPE) => hinting::Mode::ClearType,
        _ => hinting::Mode::Grayscale,
    };
    let symmetric = matches!(mode, MODE_NATURAL_SYMMETRIC | MODE_NATURAL_SYMMETRIC_DOWNSAMPLED);
    // Native disables grid-fit (scales the unhinted outline) for the SMOOTH modes when the font's gasp
    // lacks SYMMETRIC_GRIDFIT at this ppem (`GetRecommendedRenderingModeInternal` per-mode `dword_1801D4B20`,
    // indexed by DWRITE_RENDERING_MODE1: GDI_CLASSIC[2]/GDI_NATURAL[3]/ALIASED[1] force grid-fit on via
    // 0x80000; NATURAL[4]/NAT_SYM[5]/NAT_SYM_DOWNSAMPLED[7] all consult the gasp 0x04 toggle via 0x100000
    // (DS entry 0x0A110056 == NAT_SYM 0x0A110056); only OUTLINE[6]=0xFC000000 has neither, and it never
    // reaches here. Matches native rounding of the device ppem.
    // `fit` is DWRITE_GRID_FIT_MODE: DISABLED (1) leaves every glyph unhinted, ENABLED (2) overrides the gasp gate.
    let grid_fit = grid_fitted(run, mode, aa, fit, run.em_size * scale as f32);
    let cfg = hinting::Config {
        rasterizer,
        mode: hmode,
        symmetric,
        grid_fit,
        bgr: false,
        is_rotated: rotate_stretch(&hm, scale).0,
        is_stretched: if hm[1] == 0.0 && hm[2] == 0.0 { (hm[0].abs() - hm[3].abs()).abs() > 1e-6 } else { rotate_stretch(&hm, scale).1 },
        axes_swapped: dt[0] == 0.0,
        // How the glyph→device matrix rotates, read as native's scaler does: 0 axis-aligned, 1 a pure
        // quarter turn (an off-diagonal but both diagonals collapsed), 2 a general rotation or skew (an
        // off-diagonal AND a diagonal both live). A synthetic oblique is an outline shear the scaler removes
        // to hint upright, not a world rotation: the normal oblique never enters `dt`, and the sideways
        // oblique (a scaler vertical shear already in `dt`) is stripped here with the same `k` as the
        // em/metrics setup. So the off-diagonal (is it turned at all) is read from that oblique-free world
        // transform — a plain oblique at no turn stays 0 (Arial Black oblique) — while the diagonal (pure turn
        // vs general) is read from the scaler frame `hm`, where a quarter turn composed with an oblique gains a
        // diagonal (r25 30514 30900). An upright glyph's sideways oblique is that turn and slant (below).
        rotation_kind: {
            let k = if run.simulations & SIM_SIDEWAYS_OBLIQUE != 0 { raster::OBLIQUE_SHEAR } else { 0.0 };
            let off = dt[1] - k * dt[3] != 0.0 || dt[2] != 0.0;
            let diag = hm[0] != 0.0 || hm[3] != 0.0;
            // An upright glyph's sideways oblique is a turn and a slant to the scaler, a general matrix: both axes of a
            // composite offset round on the subpixel bank (VM: Segoe UI Emoji's woman oblique upright at em 78, its
            // (0, 36) component 88/64 px up where upright rounds it to 64; native scan contours, 299 of 315 points the
            // per-point slant, the component's 16 another 24/64).
            if run.simulations & SIM_SIDEWAYS_OBLIQUE != 0 { 2 } else if off { if diag { 2 } else { 1 } } else if run.simulations & 2 != 0 { 3 } else { 0 }
        },
        em_y: (em_y != em_x).then_some(em_y),
        version: 42,
        // GDI-classic antialiased rendering uses native's *render* pass (subpixel-x hinting + the in-IUP
        // x-overscale squeeze), byte-exact; layout keeps the metric pass
        // for advances (text/gdi.rs). The render GETINFO drops `grayscale_cleartype` for a ClearType (RGB)
        // target — native's CT outline differs from its grayscale-CT one (`cleartype` below). Aliased GDI-classic stays whole-pixel full grid-fit. GDI-natural always
        // renders through its render pass (GDI-classic vertical grid-fit + natural x-subpixel stems).
        gdi_render: aa != AA_ALIASED && matches!(mode, MODE_GDI_CLASSIC | MODE_GDI_NATURAL),
        cleartype: aa == AA_CLEARTYPE,
        // GDI-natural subpixel advance phantom, antialiased only (aliased keeps whole-pixel).
        subpixel_advance: gdi_natural && aa != AA_ALIASED,
        // GDI-classic synthetic bold widens the render pass's x-overscale by the bold strength (the instance
        // applies it to the antialiased render pass only).
        bold_fu: if gdi_classic_upright_bold(run, mode, dt) && hmode == hinting::Mode::GdiClassic {
            outline_embold::strength(font.head().map(|h| h.units_per_em() as i32).unwrap_or(2048)).1
        } else {
            0
        },
    };
    // Carry the run's variation coords so a grid-fit variable-font instance hints its gvar outline.
    let mut inst = hinting::Instance::new_instanced(font, em_x, cfg, run.coords)?;
    // On the axes, an unhinted oblique glyph is sheared by the instance (native's per-font-unit transform entry);
    // a hinted one by the post-hint skew below.
    let upright = dt[1] == 0.0 && dt[2] == 0.0 && dt[0] > 0.0 && dt[3] > 0.0;
    // A near-diagonal transform shifts the hinted outline 1/64 px right before the scan: the scaler flags a
    // matrix that is not a positive square when either row of its em-scaled 16.16 form has two magnitudes
    // within 33 of each other, and its bitmap-size pass then adds one 26.6 unit to every x (dwrite reading;
    // VM: aliased text at 45, 135, 225 and 315 degrees and at 45 ± 0.001 exact, 44.99 unflagged).
    // ponytail: grayscale/ClearType at an exact diagonal keep a smaller edge residual (Arial 12: 24 px of
    // 66), likely scan-converter ties on the exactly diagonal hinted edges; capture native contours there.
    let diagonal_shift = {
        // In f64: a huge or non-finite em-scaled entry is never near (an i64 cast would saturate and overflow abs).
        let f = |v: f64| (v * run.em_size as f64 * 65536.0).round();
        let near = |a: f64, b: f64| (f(a).abs() - f(b).abs()).abs() <= 33.0;
        let positive_square = hm[1] == 0.0 && hm[2] == 0.0 && hm[0] == hm[3] && hm[0] > 0.0;
        i64::from(!positive_square && (near(hm[0], hm[1]) || near(hm[2], hm[3])))
    };
    // Flipped on the axes too: native folds the oblique into the transform before the flip, so the unhinted outline
    // still shears in font units and the flip follows exactly (VM: Segoe UI Bold's alef oblique under a y-flip, three
    // points 1/64 px right when sheared after scaling).
    let axis_aligned = dt[1] == 0.0 && dt[2] == 0.0 && dt[0] != 0.0 && dt[3] != 0.0;
    let unhinted_oblique = run.simulations & 2 != 0 && axis_aligned && !inst.grid_fit();
    if unhinted_oblique {
        inst.set_oblique();
    }
    // Aliased and the GDI measuring modes (GDI-classic, GDI-natural) position the glyph origin on the
    // whole-pixel grid — GDI-compatible placement rounds x half-up; the non-GDI subpixel modes keep the
    // phase. (GDI-natural still hints natural x-subpixel *stems* inside the glyph via `hmode`; only the
    // origin snaps.) Dropping GDI-natural here fixes the ~48-seed r12 8×1 coverage class: we were drifting
    // the whole glyph by the sub-pixel pen while native holds it on whole pixels (fillsubrect-cap proof).
    let subpixel = !matches!(hmode, hinting::Mode::Aliased | hinting::Mode::GdiClassic | hinting::Mode::GdiNatural);
    let origin = snap_baseline(run.baseline_origin, dt, run.no_snap);
    let mut out: Vec<(Vec<[f64; 2]>, Vec<bool>)> = Vec::new();
    // Diagnostic (text scan task), behind the off-by-default `inject-contours` feature so it is absent
    // from prod builds: D2D_INJECT_CONTOURS replaces each glyph's hinted output contours with native's
    // captured fsc shape, placed at OUR pen origin. XDIV/YDIV give the fsc device units (CT x=/384 6×,
    // gray x=/512 8×, y=/64); DX/DY are sub-pixel origin probes. inject_map keys native's record by
    // first-occurrence of the glyph id (native's glyph cache) so composites/repeats stay aligned.
    #[cfg(feature = "inject-contours")]
    let inject = inject_contours();
    #[cfg(feature = "inject-contours")]
    let (xdiv, ydiv, idx_sh, idy_sh) = (
        std::env::var("D2D_INJECT_XDIV").ok().and_then(|s| s.parse().ok()).unwrap_or(384.0f64),
        std::env::var("D2D_INJECT_YDIV").ok().and_then(|s| s.parse().ok()).unwrap_or(64.0f64),
        std::env::var("D2D_INJECT_DX").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0f64),
        std::env::var("D2D_INJECT_DY").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0f64),
    );
    #[cfg(feature = "inject-contours")]
    let mut inject_map: std::collections::HashMap<u16, usize> = std::collections::HashMap::new();
    #[cfg(feature = "inject-contours")]
    let mut inject_next = 0usize;
    #[cfg(feature = "inject-contours")]
    if std::env::var_os("D2D_INJECT_DBG").is_some() { eprintln!("HCM aa={} mode={} inject={:?} nglyphs={}", aa, mode, inject.as_ref().map(|v| v.len()), run.glyphs.len()); }
    // DirectWrite places each glyph from f32 origins: the pen sums the advances in DIPs, a uniform-scale
    // transform folds its translation into the origin (`b + d/m11`) and scales the placement, and any other
    // transform maps each origin as `(y·m21 + x·m11) + dx`. The origin then lands on round-half-up sub-samples.
    // An f64 sum broke placement ties the other way (VM: a quarter turn whose device y is 7.3·5 = 36.5
    // sub-rows lands on 7.4 natively; f64 gave 7.2).
    let f = dt.map(|v| v as f32);
    let tol = 1.0 / 65536.0;
    let uniform = f[1].abs() < tol && f[2].abs() < tol && (f[3] - f[0]).abs() < tol;
    let scale = if uniform { f[0] } else { 1.0 };
    let (mut pen_x, origin_y) = if uniform { (origin[0] + f[4] / f[0], origin[1] + f[5] / f[0]) } else { (origin[0], origin[1]) };
    let round_half_up = |v: f32| { let r = v.round_ties_even(); if r - v == -0.5 { r + 1.0 } else { r } };
    for g in run.glyphs {
        let gx = pen_x + g.offset[0];
        let gy = origin_y - g.offset[1];
        let (pen_dev_x, pen_dev_y) = if uniform { (gx, gy) } else { ((gy * f[2] + gx * f[0]) + f[4], (gx * f[1] + gy * f[3]) + f[5]) };
        let place = |v: f32, n: f64| round_half_up(n as f32 * scale * v) as f64 / n;
        // Sub-pixel x/y quantize to the mode's oversample grid: ClearType positions on 1/6-px
        // sub-columns (the 6× grid), grayscale NATURAL_SYMMETRIC on 1/4 px (the 4× grid). The
        // vertical phase is the mode's vertical oversample: NATURAL_SYMMETRIC is 4× grayscale / 5×
        // ClearType, every other smooth mode is a single device row (whole-pixel y). DrawText snaps the
        // baseline to a whole device pixel upstream ([`snap_baseline`]), so this quantize is a no-op
        // there; it only bites DrawGlyphRun, which native places on the sub-pixel grid directly — a
        // baseline of 57.6 lands at 57.5 (2/4 sub-rows, verified byte-exact against native's gray.bin
        // for Impact 'R'@48 NATURAL_SYMMETRIC), not the whole-pixel 58 the old `round()` gave.
        // Captured decisively (Arial 'g'@20): native's baseline 0.25 px render is byte-identical to
        // its 2/6-px render — native snaps 0.25 → 2/6, so a 1/4 grid (which keeps 0.25 = 1.5 sub-col,
        // a half-sub-column tie) diverges at the quarter phases while 0 and 0.5 stay exact.
        // Grayscale NATURAL (non-GDI measuring) oversamples 8× horizontally, so native positions the pen
        // x on the 1/8-px sub-column grid (not whole-pixel) — a fractional pen (DrawText into a rect with a
        // non-integer left, e.g. 13.09 → dev 16.36) lands at 16.375, shifting the strokes by up to a
        // sub-sample vs our old whole-pixel round (w951 14951b: 'x' Arial 9 @120dpi, 24 px / maxd 90 →
        // byte-exact). A no-op where the pen is already integer, so the integer-pen scenes are unaffected.
        // The GDI *measuring* modes (GDI_CLASSIC, GDI_NATURAL) are different: GDI-compatible placement is
        // WHOLE-PIXEL x — native rounds the glyph origin half-up (`subpixel` false below for both), so a
        // sub-pixel pen collapses into whole-pixel bands. Captured byte-exact for Segoe 'd'@17.66
        // GDI_NATURAL across 8 sub-pixel pens: native renders are identical within each half-pixel band
        // (pen 2.0–2.375 → origin 2, 2.5–2.875 → origin 3), and our whole-pixel render matches each band
        // (see memory `vm-cold-capture-method`, fillsubrect-cap/). GDI_NATURAL still hints its stems with
        // natural x-subpixel inside the glyph; only the origin snaps whole-pixel.
        let phases = if aa == AA_CLEARTYPE { 6.0 } else if mode == MODE_NATURAL_SYMMETRIC_DOWNSAMPLED { 2.0 } else if symmetric { 4.0 } else if horizontal_only(mode) { 8.0 } else { 1.0 };
        let vphases = if symmetric { if aa == AA_CLEARTYPE { 5.0 } else if mode == MODE_NATURAL_SYMMETRIC_DOWNSAMPLED { 2.0 } else { 4.0 } } else { 1.0 };
        let px = place(pen_dev_x, if subpixel { phases } else { 1.0 });
        let py = place(pen_dev_y, if subpixel { vphases } else { 1.0 });
        // DWRITE_FONT_SIMULATIONS_OBLIQUE: native shears the hinted glyph right by its height above the
        // baseline (a device-space skew applied after hinting). `OBLIQUE_SKEW` is the slant tangent.
        let skew = if run.simulations & 2 != 0 && !unhinted_oblique {
            std::env::var("D2D_OBLIQUE_SKEW").ok().and_then(|s| s.parse().ok()).unwrap_or(OBLIQUE_SKEW)
        } else {
            0.0
        };
        if let Some(mut h) = inst.hint(g.id) {
            // Upright GDI classic embolds ClearType's outline here too, except at a one-pixel boldX, where it emboldens
            // the scanned bits instead (`render_analysis`); its hinter widens the x-overscale at every size
            // (`hinting::Config::bold_fu`).
            let ct_outline = (aa == AA_CLEARTYPE && gdi_classic_upright_bold(run, mode, dt))
                || tilted_ct_bold(run, dt, aa)
                || natural_ct_outline_bold(run, mode, dt, aa, fit);
            // An aliased run embolds its outline too once boldX reaches two pixels; at one pixel it dilates the scanned
            // bits ([`aliased_outline_bold`]).
            if run.simulations & 1 != 0 && (aa == AA_GRAYSCALE || ct_outline || aliased_outline_bold(run, aa, device_em) || alias_ts_outline(run, aa, mode, dt)) && (!gdi_classic_bitmap_bold(run, mode, dt) || alias_ts_outline(run, aa, mode, dt)) && !natural_bitmap_bold(run, mode, dt) {
                let (bx, by) = raster::bold_amount(device_em as i32);
                // boldX==1 is native's small-amount axis-aligned branch (horizontal_one_pixel); boldX>=2
                // is the general edge-offset-and-intersect embolden, both on the hinted outline.
                if bx == 1 {
                    embolden::horizontal_one_pixel(&mut h);
                } else if bx >= 2 {
                    embolden::embolden_hinted(&mut h, bx, by);
                }
            }
            let contour_start = out.len();
            #[cfg(feature = "inject-contours")]
            if let Some(recs) = &inject {
                // Look up this glyph's native record by first-occurrence of its id (native's glyph cache).
                let rec_idx = *inject_map.entry(g.id).or_insert_with(|| { let i = inject_next; inject_next += 1; i });
                if let Some(contours) = recs.get(rec_idx) {
                    for c in contours {
                        if c.len() < 3 { continue; } // skip degenerate records (e.g. the 1-point space)
                        let pts: Vec<[f64; 2]> = c.iter().map(|&(x, y, _)| [px + idx_sh + x / xdiv, py + idy_sh - y / ydiv]).collect();
                        let onc: Vec<bool> = c.iter().map(|&(_, _, o)| o).collect();
                        out.push((pts, onc));
                    }
                }
                if let Some(modes) = glyph_modes.as_mut() { modes.push((contour_start..out.len(), raster::dropout_mode(h.scan_control.0, h.scan_control.1))); }
                if let Some(phases) = glyph_phases.as_mut() { phases.push(((px * 2.0).round() as i64 ^ (py * 2.0).round() as i64) & 1 != 0); }
                if std::env::var_os("D2D_INJECT_DBG").is_some() { eprintln!("INJECT glyph gid={} rec={} nout={} px={:.2} py={:.2}", g.id, inject_map.get(&g.id).copied().unwrap_or(usize::MAX), out.len() - contour_start, px, py); }
                pen_x += g.advance;
                continue;
            }
            // Hand the scan converter the hinted *quadratics* (device points + on-curve flags), not a
            // pre-flattened polyline: `raster::coverage_outline` decomposes curves with the native
            // split_quad/calc_spline path, the same one the unhinted grayscale/cleartype use. Flattening
            // here (the old `contours()`) replaced that with a polyline + calc_line and lost native curve
            // precision — the dominant grayscale-coverage residual on curved glyphs.
            let mut s = 0usize;
            for &e in &h.contour_ends {
                let e = e as usize;
                if e >= s && e < h.points.len() {
                    // Off the axes, or flipped on them, the hinted outline (em-pixel units) maps through the
                    // residual L/scale (VM: a quarter turn, a one-axis shear and a half turn hint then map).
                    // The oblique simulation's shear is part of that residual, so it maps the same way.
                    let pts = if skew == 0.0 && upright {
                        h.points[s..=e].iter().map(|p| [px + p.x as f64 / 64.0, py - p.y as f64 / 64.0]).collect()
                    } else {
                        // Native's post-hint transform: the residual matrix in 16.16 (the oblique skew folded in,
                        // as native's transform setup folds it into the transform matrix), each product
                        // rounded half away from zero to 26.6.
                        // Each TransMatrix row divides by its own axis scale.
                        let q = |v: f64, s: f64| (v / s * 65536.0).round() as i64;
                        let c = [q(dt[0], sx), q(skew * dt[0] - dt[2], sy), q(dt[1], sx), q(skew * dt[1] - dt[3], sy)];
                        let mul = |a: i32, f: i64| { let v = a as i64 * f; (v + (v >> 63) + 0x8000) >> 16 };
                        h.points[s..=e].iter().map(|p| {
                            [px + (mul(p.x, c[0]) + mul(p.y, c[1]) + diagonal_shift) as f64 / 64.0, py + (mul(p.x, c[2]) + mul(p.y, c[3])) as f64 / 64.0]
                        }).collect()
                    };
                    out.push((pts, h.on_curve[s..=e].to_vec()));
                }
                s = e + 1;
            }
            if let Some(phases) = glyph_phases.as_mut() {
                // Only the parity (bit 0) of the summed ×2 device position is used, and bit 0 of a sum
                // equals the XOR of the operands' bit 0, so `^` is identical to `+` here — and avoids the
                // i64 overflow when a huge/unrepresentable glyph position saturates both casts to i64::MAX
                // (fuzz 2186: MAX + MAX panicked on the debug overflow check).
                phases.push(((px * 2.0).round() as i64 ^ (py * 2.0).round() as i64) & 1 != 0);
            }
            if let Some(modes) = glyph_modes.as_mut() {
                modes.push((contour_start..out.len(), raster::dropout_mode(h.scan_control.0, h.scan_control.1)));
            }
            // D2D_OURPTS_GO: dump this glyph's hinted device points ×64 (y-up, like native's
            // scan-converter contour list) so a scene render can be diffed glyph-for-glyph against the
            // d2d_contour_hook capture. The conformance runner toggles this per scene (gated by
            // D2D_OURPTS=<scene>); diagnostic only, inert without the env var.
            if std::env::var_os("D2D_OURPTS_GO").is_some() {
                let xs: Vec<i64> = h.points.iter().map(|p| ((px + p.x as f64 / 64.0) * 64.0).round() as i64).collect();
                let ys: Vec<i64> = h.points.iter().map(|p| ((py - p.y as f64 / 64.0) * -64.0).round() as i64).collect();
                eprintln!("OURPTS gid={} nc={} x={:?} y={:?} scan={:?}", g.id, h.contour_ends.len(), xs, ys, h.scan_control);
            }
        }
        pen_x += g.advance;
    }
    // The scan converter's dropout control uses the font's resolved SCANCTRL/SCANTYPE (from `prep`).
    let (enabled, scan_type) = inst.scan_control();
    Some((out, raster::dropout_mode(enabled, scan_type)))
}

/// Normalize a `DrawText` layout rect to `(origin, width, height)`. Native normalizes an inverted
/// rect (right<left or bottom<top) before laying out — the origin is the per-axis minimum and the box
/// extent is the absolute difference — the same inverted-rect normalization geometry/bitmap apply.
/// Fuzzer scenes pass degenerate rects (`text-aliased-01-fjord`: right≈0 < left=3.88, so native draws
/// from x≈0, not x=3.88). A normal rect is unchanged (`min`==left, `abs`==right-left).
pub fn normalize_layout_rect(rect: [f32; 4]) -> ([f32; 2], f32, f32) {
    ([rect[0].min(rect[2]), rect[1].min(rect[3])], (rect[2] - rect[0]).abs(), (rect[3] - rect[1]).abs())
}

/// Native's `DrawText` builds a transient layout via `IDWriteFactory::CreateTextLayout`, which gates on
/// `maxWidth >= 0.0 && maxHeight >= 0.0` (dwrite `CreateTextLayout@DWriteFactory`) — so a rect whose
/// normalized box width or height is **NaN** (a non-finite rect component, e.g. r12 17620/17323 op's
/// `top=NaN`, 17356 op's `right=NaN`) makes the internal CreateTextLayout fail; `+inf` passes (inf ≥ 0).
/// The box dims are [`normalize_layout_rect`]'s w/h (always ≥ 0 for a finite rect, so only a NaN
/// component trips this — an inverted rect has a positive width, which native accepts).
///
/// DrawText **swallows** that failure and draws nothing, returning **S_OK** (r12 17323/17356 EndDraw
/// hr 0, tag 0 — native does NOT latch). So the DrawText caller draws nothing WITHOUT latching. Only
/// the app's own `IDWriteFactory::CreateTextLayout` returns E_INVALIDARG (dwrite-ffi `make_layout`).
pub fn draw_text_rect_rejected(rect: [f32; 4]) -> bool {
    let (_, w, h) = normalize_layout_rect(rect);
    !(w >= 0.0) || !(h >= 0.0)
}

/// R6 (OUTLINE mode, device em ≥ 350 px): fill the run's glyph outlines through the geometry
/// realization feather mesh — the same antialiased tessellator native's outline text uses, not the
/// 8×8 geometry AA. Native `d2d1!TextOutlineRealizer::CacheGeometry` sweeps the outline with an
/// the AA tessellator — the identical
/// AA tessellator the filled geometry realization reproduces byte-exact ([`crate::geometry::realization`])
/// — and composites the feather through WARP's realization VS/PS. Routing outline glyphs through
/// [`crate::geometry::realization::fill_feather`] reuses that path; edges where the fill enters a pixel
/// (left/top) are byte-exact. Native caches the glyph mesh with **fixed-point** vertices
/// (`TextRealizationVertexFixed`, bbox shifted to a local (1,1) origin) where the realization uses
/// float `CD3DRealizationVertex`; that quantization leaves a ≤11-LSB residual on trailing edges
/// (right/bottom), to be closed by a fixed-point feather variant in `geometry::realization` (its owner).
#[allow(clippy::too_many_arguments)]
/// The hinter for the glyph outlines DirectWrite's GetGlyphRunOutline returns, and so d2d1's large-glyph outline fill:
/// each glyph's program run at ppem = upem, one pixel per design unit, under the classic (bi-level) GETINFO, its 26.6
/// result kept; a bold simulation then emboldens the hinted points (VM sb39-sb42: Calibri 'O', 'B' and 'g' plain, every
/// point exact, and Calibri 'O' bold, the embolden's input hinted and its output the outline). `None` for a variable
/// instance or a face without TrueType hinting tables, which keep the design outline.
pub fn outline_hinter<'a>(font: &FontRef<'a>, coords: &[skrifa::raw::types::F2Dot14]) -> Option<hinting::Instance<'a>> {
    if !coords.is_empty() {
        return None;
    }
    let upem = font.head().ok()?.units_per_em();
    hinting::Instance::new(font, upem as f64, hinting::Config { mode: hinting::Mode::Aliased, symmetric: false, ..hinting::Config::default() })
}

fn fill_outline(target: &mut PixelBuffer, dt: &[f64; 6], run: &GlyphRun, font: &FontRef, fg: &TextFg, blend: crate::pixels::Blend, aliased: bool, raster_clip: Option<[i32; 4]>) {
    let upem = font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f32;
    let scale = run.em_size / upem;
    let outlines = font.outline_glyphs();

    // Baseline pixel snapping (R2): snap the baseline origin to the device pixel grid perpendicular
    // to the baseline, expressed as a DIP shift so the DIP→device flatten stays exact.
    let origin = snap_baseline(run.baseline_origin, dt, run.no_snap);

    // DrawGlyphRun's merge ranges (ComputeMergeRanges): one glyph never merges, any brush but an opaque
    // solid ranges the whole run, an aliased run ranges nothing, and otherwise the font's MERG table
    // decides. Within a range, glyphs whose device bounds meet (ComputeMergeRects) are one shape group
    // filled through the ordinary geometry fill after the single glyphs (RenderMergedGlyphs), never cached.
    let n = run.glyphs.len();
    let opaque = matches!(fg, TextFg::Solid(c) if c[3] == 1.0);
    let ranges = if n <= 1 {
        Vec::new()
    } else if !opaque {
        vec![0..n]
    } else if aliased {
        Vec::new()
    } else {
        merge_ranges(font, run)
    };
    let aa = if aliased { crate::pixels::Aa::Aliased } else { crate::pixels::Aa::PerPrimitive };
    let dm: crate::geometry::Matrix = [dt[0] as f32, dt[1] as f32, dt[2] as f32, dt[3] as f32, dt[4] as f32, dt[5] as f32];
    // The fill is clipped to the draw's clip rect, as a geometry fill is; under COPY its edge list leaves a curve
    // wholly off that rect as its chord.
    let clip = raster_clip.filter(|_| !aliased && blend == crate::pixels::Blend::Copy);
    let texels = match fg { TextFg::Brush(b) => b.as_text_texels(), TextFg::Solid(_) => None };
    let fill = |target: &mut PixelBuffer, figs: &[Vec<[f32; 2]>]| match fg {
        TextFg::Solid(color) => crate::pixels::Target::new(target).with_clip(raster_clip).fill_path(
            figs, crate::pixels::FillMode::Winding, crate::pixels::premultiply(*color), aa, blend, true,
        ),
        TextFg::Brush(b) => {
            // The brush's intermediate covers the fill's whole-pixel bounds cut to the target and the draw's clip (a
            // pushed layer's pixel rect among them), its planes set up on that quad (VM probes of r25 30608: the
            // layer's right edge moves the CUBIC phase in its last column, its left and top edges do not).
            let quad = texels.as_ref().and_then(|t| {
                let bb = figs.iter().flatten().fold(None, |b: Option<[f32; 4]>, p| Some(b.map_or([p[0], p[1], p[0], p[1]], |b| [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])])))?;
                let (tw, th) = (target.desc.width as f32, target.desc.height as f32);
                let lim = raster_clip.map_or([0.0, 0.0, tw, th], |c| [c[0].max(0) as f32, c[1].max(0) as f32, (c[2] as f32).min(tw), (c[3] as f32).min(th)]);
                Some(t.with_text_quad([bb[0].floor(), bb[1].floor(), bb[2].ceil(), bb[3].ceil()], lim))
            });
            crate::brush::fill_geometry_cov(target, figs, true, aa, quad.as_ref().or(texels.as_ref()).unwrap_or(b), blend, false, true, raster_clip, true)
        }
    };
    // The outline GetGlyphRunOutline returns, hinted at ppem = upem ([`outline_hinter`]); each glyph's points are
    // read once though its outline is drawn twice (the run's figures, then the cached realization).
    let hinter = outline_hinter(font, run.coords);
    let points: std::cell::RefCell<std::collections::HashMap<u16, Option<outline_embold::RawGlyph>>> = Default::default();
    let draw = |id: u16, scale: f32, pen: [f32; 2]| {
        let mut figures: Vec<Figure> = Vec::new();
        let gid = GlyphId::new(id as u32);
        let pts = points.borrow_mut().entry(id).or_insert_with(|| hinter.as_ref().and_then(|h| raster::outline_points_fu(h, gid))).clone();
        let mut builder = OutlineBuilder { figures: &mut figures, scale, pen, cur: None, last: pen };
        // The simulations apply to that outline (r25 30767): a bold one emboldens it in design space, an oblique
        // one then slants it.
        let bold = run.simulations & 1 != 0;
        let oblique = run.simulations & 2 != 0 && !run.is_sideways;
        let raw = pts.or_else(|| if bold && run.coords.is_empty() { raster::glyf_points_fu(font, gid) } else { None });
        match (raw, outlines.get(gid)) {
            (Some(r), _) => outline_embold::draw_points(&r, bold.then(|| outline_embold::strength(upem as i32)), oblique, &mut builder),
            (None, Some(glyph)) => {
                let k = if oblique { raster::OBLIQUE_SHEAR as f32 } else { 0.0 };
                let mut sheared = outline_embold::Shear { inner: &mut builder, k };
                if bold {
                    let _ = outline_embold::draw_bold(&glyph, upem as i32, None, &mut sheared);
                } else {
                    let _ = glyph.draw(DrawSettings::unhinted(Size::unscaled(), LocationRef::new(run.coords)), &mut sheared);
                }
            }
            (None, None) => {}
        }
        builder.flush();
        figures
    };

    // Each glyph's device-space fill polygons and its pen, or None when it lies wholly off the target.
    // Native bounds a glyph run's rasterization to the target/clip rect — d2d1 `GlyphRunRenderer` tiles it
    // into sub-rects within the render target (`ProcessSubRects`) and never works outside it. Our outline
    // fill works in DEVICE space, so an extreme world transform would otherwise drive it over an
    // astronomical span. Fast path: a glyph whose device box is wholly off the target draws nothing (fuzz
    // 7978: a 1e6 scale puts the run at device ~9e7). The control-point box is a superset of the glyph,
    // so skipping only when it is wholly outside never drops one that touches the target.
    let mut pen_x = origin[0];
    let glyphs: Vec<Option<(Vec<Vec<[f32; 2]>>, [f32; 2])>> = run
        .glyphs
        .iter()
        .map(|g| {
            let glyph_origin = [pen_x + g.offset[0], origin[1] - g.offset[1]];
            pen_x += g.advance;
            let figures = draw(g.id, scale, glyph_origin);
            if glyph_outside_target(&figures, dt, target.desc.width as f64, target.desc.height as f64) {
                return None;
            }
            // Each glyph is its own transformed shape: the outline at the run's em size about the glyph's own
            // origin, under the transform's linear part translated to the origin's device position.
            let (x, y) = (glyph_origin[0], glyph_origin[1]);
            let gm = [dm[0], dm[1], dm[2], dm[3], (x * dm[0] + y * dm[2]) + dm[4], (x * dm[1] + y * dm[3]) + dm[5]];
            let local = Geometry::Path { fill: FillMode::Winding, figures: draw(g.id, scale, [0.0, 0.0]) };
            let geom = Geometry::Transformed { matrix: gm, child: Box::new(local) };
            Some((geom.flatten_fill_on(&crate::geometry::IDENTITY, clip).into_iter().filter(|p| p.filled).map(|p| p.pts).collect(), glyph_origin))
        })
        .collect();
    // Each glyph's device bounds intersected with the target; an empty one is skipped by the merge sweep.
    let (tw, th) = (target.desc.width as f32, target.desc.height as f32);
    let bounds: Vec<[f32; 4]> = glyphs
        .iter()
        .map(|g| {
            let b = g.as_ref().and_then(|g| {
                g.0.iter().flatten().fold(None, |b: Option<[f32; 4]>, p| Some(b.map_or([p[0], p[1], p[0], p[1]], |b| [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])])))
            });
            b.map_or([0.0; 4], |b| [b[0].max(0.0), b[1].max(0.0), b[2].min(tw), b[3].min(th)])
        })
        .collect();
    let skip: Vec<bool> = bounds.iter().map(|b| !(b[0] < b[2] && b[1] < b[3])).collect();
    let groups = merge_rects(&bounds, &skip, &ranges);
    let mut merged = vec![false; n];
    groups.iter().flat_map(|g| g.clone()).for_each(|i| merged[i] = true);
    // Native sizes the cache bucket by `MILMatrix3x2::GetMaxFactor(transform)·emSize`: the larger singular
    // value, so a shear counts (r12 17381's m12 = 200 at 192 dpi; max(|m11|,|m22|) alone saw 5.8).
    let device_em = run.em_size * crate::extras::print::compute_maximum_scale_factor(dt.map(|v| v as f32));
    // Only an opaque solid SOURCE_OVER run is cached as a realization mesh: DrawGlyphRun takes the uncached
    // path unless the brush is solid and opaque and the primitive blend is SOURCE_OVER. The uncached path fills
    // each glyph shape under the run transform as an ordinary geometry fill. ANY glyph past native's
    // text-rasterization limit (device em ≥ 2^15, where native stops building a realization mesh and falls back to
    // the plain outline fill) goes through that fill too (the 28.4 edge list and the geometry fill's −0.5
    // pixel-centre convention). That path strip-clips the figure to ±524287 and its scan converter clamps to the
    // target, so it is bounded. The feathered realization is NOT: its tessellator sweep degenerates on the huge,
    // precision-lossy contour of a glyph that size (fuzz 7978's 1e6-scaled run hangs there). An aliased run is
    // cached like an antialiased one, its mesh drawn with no feather.
    let cached = opaque && blend == crate::pixels::Blend::SourceOver && device_em < 32768.0;
    for (i, g) in glyphs.iter().enumerate() {
        let Some((figs, glyph_origin)) = g else { continue };
        if merged[i] {
            continue;
        }
        if !cached {
            fill(target, figs);
            continue;
        }
        // The cached feather mesh. Native caches the glyph at em = bucket in its own space and
        // draws it by the transform's linear part scaled by em/bucket, translated to the glyph origin's
        // device position.
        let bucket = crate::geometry::realization::colr_outline_bucket(device_em);
        let geom = Geometry::Path { fill: FillMode::Winding, figures: draw(run.glyphs[i].id, bucket / upem, [0.0, 0.0]) };
        let m = dt.map(|v| v as f32);
        let k = run.em_size / bucket;
        let lin = [m[0] * k, m[1] * k, m[2] * k, m[3] * k];
        let (x, y) = (glyph_origin[0], glyph_origin[1]);
        let o = [(y * m[2] + x * m[0]) + m[4], (y * m[3] + x * m[1]) + m[5]];
        if let TextFg::Solid(color) = fg {
            crate::geometry::realization::fill_feather_fixed(target, &geom, *color, lin, o, blend, if aliased { 0.0 } else { crate::geometry::realization::AA_WIDTH });
        }
    }
    for g in &groups {
        let figs: Vec<Vec<[f32; 2]>> = g.clone().filter_map(|i| glyphs[i].as_ref()).flat_map(|(f, _)| f.iter().cloned()).collect();
        fill(target, &figs);
    }
}

/// True when a glyph's outline, mapped to device space by `dt`, lies entirely outside the `w`×`h`
/// device target (plus a small AA margin) — so it draws nothing and must be skipped before the
/// device-space outline fill (see the call site in [`fill_outline`]). The box is taken over the raw
/// control points, a conservative superset of the glyph, so this never drops a glyph that would touch
/// the target; a non-finite or empty box falls through to the existing fill (the hang this guards is a
/// finite-but-astronomical span from an extreme transform, fuzz 7978).
fn glyph_outside_target(figures: &[Figure], dt: &[f64; 6], w: f64, h: f64) -> bool {
    let (mut minx, mut miny, mut maxx, mut maxy) =
        (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut acc = |p: [f32; 2]| {
        let (x, y) = (p[0] as f64, p[1] as f64);
        let (dx, dy) = (dt[0] * x + dt[2] * y + dt[4], dt[1] * x + dt[3] * y + dt[5]);
        minx = minx.min(dx);
        miny = miny.min(dy);
        maxx = maxx.max(dx);
        maxy = maxy.max(dy);
    };
    for f in figures {
        acc(f.start);
        for s in &f.segments {
            match *s {
                Segment::Line(p) => acc(p),
                Segment::Quadratic(c, p) => {
                    acc(c);
                    acc(p);
                }
                Segment::Cubic(a, b, p) => {
                    acc(a);
                    acc(b);
                    acc(p);
                }
                Segment::Arc { to, .. } => acc(to),
            }
        }
    }
    drop(acc);
    const MARGIN: f64 = 2.0;
    minx.is_finite()
        && maxx.is_finite()
        && miny.is_finite()
        && maxy.is_finite()
        && (maxx < -MARGIN || minx > w + MARGIN || maxy < -MARGIN || miny > h + MARGIN)
}

/// Snap `origin` (DIPs) to the device pixel grid along the baseline-perpendicular axis, returning a
/// DIP origin. Axis-aligned transforms snap the y device coordinate (`floor(y+0.5)`, DirectWrite's
/// round) and leave x sub-pixel; a rotated transform is left unsnapped.
pub fn snap_baseline(origin: [f32; 2], dt: &[f64; 6], no_snap: bool) -> [f32; 2] {
    if no_snap {
        return origin;
    }
    // A horizontal baseline (the run's x axis maps onto device x: m12 = 0) snaps its device y; a vertical
    // one (a quarter turn: m11 = 0) snaps its device x. Any other direction is left unsnapped.
    // The device coordinate is the f32 product and sum, as native maps the point: at 120 dpi a baseline 13.2 DIP down
    // is 16.5 px in f32 and snaps to 17, where f64 left it 16.4999998 and snapped to 16 (VM: Arial, Tahoma and Calibri
    // DrawText lines at a 2.0 DIP rect top, one pixel high).
    // The way back is the f32 product with the scale's f32 reciprocal, not a division (VM: a Segoe UI line whose
    // baseline snaps to 121 px at dpi 144 draws at 121·(1/1.5) = 80.66667175 DIP, 121.0000076 px, where 121 / 1.5
    // gives 80.666664 and 121.0; fuzz 7909's outline glyphs sit there).
    let snap = |m: f32, d: f32| {
        let snapped = ((m * origin[1] + d) as f64 + 0.5).floor() as f32;
        [origin[0], (snapped - d) * (1.0 / m)]
    };
    if dt[1] == 0.0 && dt[3] != 0.0 {
        return snap(dt[3] as f32, dt[5] as f32);
    }
    if dt[0] == 0.0 && dt[2] != 0.0 {
        return snap(dt[2] as f32, dt[4] as f32);
    }
    origin
}

/// Collects a glyph's `skrifa` outline (design units) into [`Figure`]s already mapped to DIP space:
/// `x = pen.x + dx·scale`, `y = pen.y − dy·scale` (TrueType y-up flipped to device y-down).
struct OutlineBuilder<'a> {
    figures: &'a mut Vec<Figure>,
    scale: f32,
    pen: [f32; 2],
    cur: Option<Figure>,
    last: [f32; 2],
}

impl OutlineBuilder<'_> {
    fn pt(&self, x: f32, y: f32) -> [f32; 2] {
        [self.pen[0] + x * self.scale, self.pen[1] - y * self.scale]
    }
    fn flush(&mut self) {
        if let Some(f) = self.cur.take() {
            self.figures.push(f);
        }
    }
}

impl OutlinePen for OutlineBuilder<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        self.flush();
        self.last = self.pt(x, y);
        self.cur = Some(Figure { start: self.last, filled: true, closed: true, segments: Vec::new(), seg_flags: Vec::new() });
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.pt(x, y);
        if let Some(f) = &mut self.cur {
            f.segments.push(Segment::Line(p));
            self.last = p;
        }
    }
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        let (c, p) = (self.pt(cx, cy), self.pt(x, y));
        if let Some(f) = &mut self.cur {
            // DirectWrite raises a glyph's quadratic to a cubic itself, after the glyph position is added,
            // with the weights 1/3 and 1 − 1/3 each rounded in f32 (not a path geometry's elevation).
            let third = 1.0f32 / 3.0;
            let raise = |e: [f32; 2]| [c[0] * (1.0 - third) + e[0] * third, c[1] * (1.0 - third) + e[1] * third];
            f.segments.push(Segment::Cubic(raise(self.last), raise(p), p));
            self.last = p;
        }
    }
    fn curve_to(&mut self, c0x: f32, c0y: f32, c1x: f32, c1y: f32, x: f32, y: f32) {
        let (c0, c1, p) = (self.pt(c0x, c0y), self.pt(c1x, c1y), self.pt(x, y));
        if let Some(f) = &mut self.cur {
            f.segments.push(Segment::Cubic(c0, c1, p));
            self.last = p;
        }
    }
    fn close(&mut self) {}
}

// Phase 2D-2: DirectWrite layout and shaping (their own owner).
pub mod fallback; // system font fallback (IDWriteFontFallback::MapCharacters)
pub mod thai; // Thai word breaking (DirectWrite's dictionary breaker)
/// `ID2D1DeviceContext::GetGlyphRunWorldBounds` (ffirects, this agent): the glyph-run ink bounds.
pub mod bounds;
mod embolden;
pub mod outline_embold;
pub mod gdi; // GDI-compatible measuring (advances + VDMX line metrics) for layout's GDI modes
pub mod layout;
pub mod shaping;
pub mod color;
pub mod variations; // variable-font gvar instancing (glyph-point interpolation at a location)

/// Straight foreground RGBA of a solid brush (its colour with the opacity multiplier folded into
/// alpha), the way `DrawText`/`DrawGlyphRun` take a solid `foregroundBrush`.
fn brush_rgba(b: &crate::resource::SolidColorBrush) -> [f32; 4] {
    [b.color[0], b.color[1], b.color[2], b.color[3] * b.opacity]
}

/// A UTF-16 text range `[start,end)` of a laid-out string resolved to a specific font face — the
/// fallback face `text::fallback` chose for that range. [`DeviceContext::draw_layout`] renders each
/// glyph run with the face whose range contains the run's start position (`GlyphRun::text_pos`),
/// falling back to the base `font_data`/`face_index` for any run no span covers. The host supplies
/// these from its own font loader (d2d has no system font collection of its own); an empty slice
/// means a single-font layout, byte-identical to before fallback existed.
#[derive(Clone, Copy)]
pub struct FaceSpan<'a> {
    pub start: u32,
    pub end: u32,
    pub data: &'a [u8],
    pub index: u32,
    /// The face's DWRITE_FONT_SIMULATIONS: a format range's or fallback family's matched face can be
    /// simulated (an italic request on a family without one gets a simulated oblique).
    pub sims: u32,
    /// A format range's variable-font instance (normalized coords); empty for a static face.
    pub coords: &'a [skrifa::raw::types::F2Dot14],
}

/// Phase 2D-3: Direct2D's text surface — `DrawGlyphRun`, `DrawText` and `DrawTextLayout` — wired to
/// the glyph-run renderer above. Native funnels `DrawText` → an internal `IDWriteTextLayout` →
/// per-run `IDWriteTextRenderer::DrawGlyphRun` → `CTextRenderer::DrawGlyphRun` → glyph-run analysis
/// → text blend; this mirrors that path with `text::layout` placing the glyphs and
/// [`render_glyph_run`] painting each run.
impl crate::context::DeviceContext {
    /// `ID2D1RenderTarget::DrawGlyphRun`: paint one positioned run with a solid foreground brush.
    /// Returns `true` when the run was painted and `false` when it needs a path not implemented yet
    /// (the caller treats `false` as "unsupported", not "drew wrong"). A no-op while recording a
    /// command list (text recording is not wired).
    /// Record a glyph run into the open command list (its owned glyphs + font bytes, the device
    /// transform and text antialias state baked at record time), so a `DrawGlyphRun`/`DrawText` issued
    /// while recording replays by re-rasterizing the run against the target — text can't flatten
    /// (gamma blend reads the destination). [G6]
    fn record_glyph_run(&mut self, run: &GlyphRun, fg: crate::command::TextFg, palette: Option<u32>, params: &RenderingParams) {
        let (state, dpi) = self.rec_env();
        let cmd = crate::command::RecCmd::GlyphRun {
            state,
            dpi,
            font: std::rc::Rc::from(run.font_data),
            face_index: run.face_index,
            em_size: run.em_size,
            glyphs: std::rc::Rc::from(run.glyphs),
            baseline_origin: run.baseline_origin,
            measuring_mode: run.measuring_mode,
            is_sideways: run.is_sideways,
            bidi_level: run.bidi_level,
            no_snap: run.no_snap,
            simulations: run.simulations,
            text_aa: self.get_text_antialias_mode(),
            params: *params,
            fg,
            palette,
        };
        self.record(cmd);
    }

    /// What DrawGlyphRun checks after its brush: a negative em draws nothing, and the text rendering
    /// params' mode must suit the text antialias mode or the draw latches E_INVALIDARG — ClearType
    /// rejects ALIASED and OUTLINE, grayscale rejects ALIASED, ALIASED rejects the GDI and natural modes.
    /// The mode is the params' legacy `GetRenderingMode`, which reports NATURAL_SYMMETRIC_DOWNSAMPLED as
    /// NATURAL_SYMMETRIC. True when the run must not draw. A recording runs this check too.
    fn glyph_run_rejected(&mut self, run: &GlyphRun, params: &RenderingParams) -> bool {
        if run.em_size < 0.0 {
            return true;
        }
        let mode = if params.rendering_mode == MODE_NATURAL_SYMMETRIC_DOWNSAMPLED { MODE_NATURAL_SYMMETRIC } else { params.rendering_mode };
        let bad = match self.get_text_antialias_mode() {
            AA_CLEARTYPE => mode == MODE_ALIASED || mode == MODE_OUTLINE,
            AA_GRAYSCALE => mode == MODE_ALIASED,
            AA_ALIASED => (MODE_GDI_CLASSIC..=MODE_NATURAL_SYMMETRIC).contains(&mode),
            _ => false,
        };
        if bad {
            self.process_new_error(crate::state::E_INVALIDARG);
        }
        bad
    }

    /// Whether the em matrix (em, turned a quarter for a sideways run, times the device transform) is
    /// axis-aligned or a quarter turn, uniformly scaled, and 2^15 or more: the size DirectWrite's glyph
    /// bitmap path refuses. Zero is within 2^-16, as DirectWrite tests it. A non-finite em is never uniform.
    fn em_matrix_over_bitmap_limit(&self, e: f32, sideways: bool) -> bool {
        let m = self.device_transform().map(|v| v as f32);
        let ([a, b], [c, d]) = if sideways { ([0.0, -e], [e, 0.0]) } else { ([e, 0.0], [0.0, e]) };
        let (m11, m12, m21, m22) = (b * m[2] + a * m[0], b * m[3] + a * m[1], d * m[2] + c * m[0], d * m[3] + c * m[1]);
        let zero = |v: f32| v < 1.0 / 65536.0 && v > -1.0 / 65536.0;
        let (x, y) = if zero(m12) && zero(m21) {
            (m11, m22)
        } else if zero(m11) && zero(m22) {
            (m12, m21)
        } else {
            return false;
        };
        zero(x.abs() - y.abs()) && x.abs() >= 32768.0
    }

    /// A run drawn in an explicit bitmap mode (not DEFAULT or OUTLINE) fails, latching E_INVALIDARG, when
    /// its em matrix (em, turned a quarter for a sideways run, times the device transform) is axis-aligned
    /// or a quarter turn, uniformly scaled, and 2^15 or more: DirectWrite's glyph bitmap cache rejects that
    /// size when the run is rendered, so a recording never meets it. DEFAULT resolves such an em to
    /// OUTLINE, and an empty run never reaches the cache. Zero is within 2^-16, as DirectWrite tests it.
    pub(crate) fn glyph_run_em_rejected(&mut self, run: &GlyphRun, params: &RenderingParams) -> bool {
        if matches!(params.rendering_mode, MODE_DEFAULT | MODE_OUTLINE) || run.glyphs.is_empty() {
            return false;
        }
        // No clip culls it first, as a list plays or drawn directly (VM: r24 29278's list run alone under a clip off the
        // target latches at the DrawImage; r23 28090 and r24 29376 too).
        let bad = self.em_matrix_over_bitmap_limit(run.em_size, run.is_sideways);
        if bad {
            self.process_new_error(crate::state::E_INVALIDARG);
        }
        bad
    }

    pub fn draw_glyph_run(&mut self, run: &GlyphRun, brush: &crate::resource::SolidColorBrush, params: &RenderingParams) -> bool {
        let _a = self.glyph_api(run, 24);
        if !self.ready_to_draw() || !self.check_domain(brush.domain) || self.glyph_run_rejected(run, params) {
            return true;
        }
        let (dpi_x,dpi_y)=self.get_dpi();
        let params=&RenderingParams { dpi: [dpi_x,dpi_y], blend: crate::context::blend_of(self.get_primitive_blend()), raster_clip: self.raster_clip(), clip_px: self.centre_clip_px(), ..*params };
        if self.recording.is_some() {
            self.record_glyph_run(run, crate::command::TextFg::Solid(brush_rgba(brush)), None, params);
            return true;
        }
        if self.glyph_run_em_rejected(run, params) {
            return true;
        }
        let dt = self.device_transform();
        let aa = self.text_aa_for_draw();
        let color = brush_rgba(brush);
        match &mut self.target {
            Some(t) => render_glyph_run(t, &dt, run, color, aa, params),
            None => true,
        }
    }

    /// `ID2D1DeviceContext4::DrawGlyphRun` colour path (`ENABLE_COLOR_FONT`): expand the run into
    /// DirectWrite's coloured sub-runs (COLR/CPAL layers, then bitmap glyphs) and paint each. Base
    /// glyphs with no colour data fall back to the monochrome [`draw_glyph_run`].
    pub fn draw_color_glyph_run(&mut self, run: &GlyphRun, brush: &crate::resource::SolidColorBrush, palette: u32, params: &RenderingParams) -> bool {
        let _a = self.api(0, 0, None); // ponytail: a colour glyph run's command and primitives are not counted
        self.draw_color_glyph_run_at(run, None, brush, palette, params)
    }

    /// [`draw_color_glyph_run`] whose monochrome glyphs are placed from `mono` (DrawText's own run) when given.
    fn draw_color_glyph_run_at(&mut self, run: &GlyphRun, mono: Option<&GlyphRun>, brush: &crate::resource::SolidColorBrush, palette: u32, params: &RenderingParams) -> bool {
        if !self.ready_to_draw() || !self.check_domain(brush.domain) {
            return true;
        }
        let (dpi_x, dpi_y) = self.get_dpi();
        let params = &RenderingParams { dpi: [dpi_x, dpi_y], blend: crate::context::blend_of(self.get_primitive_blend()), ..*params };
        if self.recording.is_some() {
            self.record_glyph_run(run, crate::command::TextFg::Solid(brush_rgba(brush)), Some(palette), params);
            return true;
        }
        let dt = self.device_transform();
        let aa = self.text_aa_for_draw();
        let color = brush_rgba(brush);
        match &mut self.target {
            Some(t) => color::render_color_glyph_run(t, &dt, run, TextFg::Solid(color), palette, aa, params, mono),
            None => true,
        }
    }

    /// [`draw_color_glyph_run_at`] for DrawText with a gradient/bitmap/image brush: the colour glyphs paint their
    /// palette, the monochrome ones the brush (r25 30790: an emoji under a bitmap brush keeps its colours).
    fn draw_color_glyph_run_brush(&mut self, run: &GlyphRun, mono: Option<&GlyphRun>, brush: &crate::PixelBrush, params: &RenderingParams) -> bool {
        // ponytail: recorded into a command list, or under an image brush, the run draws monochrome; add when one needs colour.
        if self.recording.is_some() || brush.image.is_some() {
            return self.draw_glyph_run_brush(mono.unwrap_or(run), brush, params);
        }
        if !self.ready_to_draw() || self.a8_brush_rejected(brush) || !self.check_domain(brush.domain) {
            return true;
        }
        let grad_rr = self.realize_brush_for_draw(brush);
        let brush = grad_rr.as_ref().unwrap_or(brush);
        let (dpi_x, dpi_y) = self.get_dpi();
        let params = &RenderingParams { dpi: [dpi_x, dpi_y], blend: crate::context::blend_of(self.get_primitive_blend()), raster_clip: self.raster_clip(), clip_px: self.centre_clip_px(), ..*params };
        let dt = self.device_transform();
        let aa = self.text_aa_for_draw();
        match &mut self.target {
            Some(t) => color::render_color_glyph_run(t, &dt, run, TextFg::Brush(&brush.brush), 0, aa, params, mono),
            None => true,
        }
    }

    /// `ID2D1DeviceContext4::DrawColorBitmapGlyphRun`: paint a run of CBDT/sbix (or raw PBGRA) bitmap
    /// strikes. `formats` is the `DWRITE_GLYPH_IMAGE_FORMATS` the caller split the run to. See
    /// [`color::bitmap::render_bitmap_glyph_run`]. (Command-list recording is handled by the caller's
    /// glyph-run record path for the monochrome/colour runs; a bitmap run draws directly.)
    pub fn draw_color_bitmap_glyph_run(&mut self, run: &GlyphRun, formats: u32, params: &RenderingParams) -> bool {
        let _a = self.api(0, 0, None); // ponytail: not counted
        if !self.ready_to_draw() {
            return true;
        }
        let params = &RenderingParams { blend: crate::context::blend_of(self.get_primitive_blend()), ..*params };
        // Recording, a command list keeps the run and draws it where the list is drawn.
        if self.recording.is_some() {
            self.record_glyph_run(run, crate::command::TextFg::ColorBitmap(formats), None, params);
            return true;
        }
        let dt = self.device_transform();
        match &mut self.target {
            Some(t) => color::bitmap::render_bitmap_glyph_run(t, &dt, run, formats, params),
            None => true,
        }
    }

    /// `ID2D1DeviceContext4::DrawSvgGlyphRun`: paint a run of OT-SVG glyphs. Each glyph's SVG document
    /// (the `SVG ` table, gzip-inflated if needed) renders through our SVG renderer under the glyph
    /// transform native uses: the SVG coordinate system has its origin at the glyph origin with the
    /// y-axis pointing down (SVG convention), scaled by `emSize / unitsPerEm`. `color` is the context
    /// foreground (`context-fill`/`context-stroke`); explicit paints ignore it.
    pub fn draw_svg_glyph_run(&mut self, run: &GlyphRun, color: [f32; 4], params: &RenderingParams) -> bool {
        let _a = self.api(0, 0, None); // ponytail: not counted
        if !self.ready_to_draw() {
            return true;
        }
        // Recording, a command list keeps the run whole and renders its documents where the list is drawn.
        if self.recording.is_some() {
            self.record_glyph_run(run, crate::command::TextFg::Svg(color), None, params);
            return true;
        }
        let Ok(font) = FontRef::from_index(run.font_data, run.face_index) else { return false };
        let Ok(head) = font.head() else { return false };
        let upem = head.units_per_em() as f32;
        let Ok(svg) = font.svg() else { return false };
        // Native renders each glyph's document into a command list and draws that as an image with SOURCE_OVER, so MIN,
        // ADD and MAX do not reach the shapes: they draw source-over, among themselves (r25 30172: MAX over overlapping
        // rects) and onto the target (r25 30850: an opaque red rect stays red under all three; VM probes of every
        // primitive blend, which match the source-over draw exactly). COPY draws each shape (r25 30251).
        let blend = self.get_primitive_blend();
        let ignore_blend = matches!(blend, 2..=4);
        if ignore_blend {
            self.set_primitive_blend(0);
        }
        let saved = self.get_transform();
        let s = run.em_size / upem;
        // The glyph's command list replays with the world reapplied as (T·S)·S⁻¹ in f32 (S the dpi scale, inverted
        // through 1/(sx·sy)). At 120 and 240 dpi that round trip leaves T a few ulps off, enough to flip an edge that
        // sits on a 1/256 rounding tie (r25 30624; VM probes of tie edges at 120, 144 and 240 dpi).
        // ponytail: the document's own transforms compose after the round trip, not inside it as native's per-shape
        // ((B·T)·S)·S⁻¹ does; route the document through a recorded list if a transformed glyph needs that ulp.
        let (sx, sy) = if self.state.unit == 0 { (self.get_dpi().0 / 96.0, self.get_dpi().1 / 96.0) } else { (1.0, 1.0) };
        let inv = 1.0 / (sx * sy);
        let (qx, qy) = (sy * inv, sx * inv);
        let replay = |t: [f32; 6]| [t[0] * sx * qx, t[1] * sy * qy, t[2] * sx * qx, t[3] * sy * qy, t[4] * sx * qx, t[5] * sy * qy];
        let (mut pen_x, pen_y) = (run.baseline_origin[0], run.baseline_origin[1]);
        for g in run.glyphs {
            let ox = pen_x + g.offset[0];
            let oy = pen_y - g.offset[1];
            pen_x += g.advance;
            let Some(raw) = svg.glyph_data(GlyphId::from(g.id)) else { continue };
            // ponytail: a gzip-compressed OT-SVG document (0x1f 0x8b) is skipped — the crate has no
            // public inflate yet and the common OT-SVG fonts (Twemoji, the test font) ship plain SVG;
            // wire a gzip inflater when a compressed-SVG oracle appears.
            if raw.starts_with(&[0x1f, 0x8b]) {
                continue;
            }
            let Ok(xml) = core::str::from_utf8(raw) else { continue };
            let Ok(doc) = crate::svg::Document::parse(xml, [upem, upem]) else { continue };
            // SVG coords (origin at glyph origin, y-down) → DIP: scale `s`, translate to the pen, then
            // the current world transform. The device DPI is applied by the draws the renderer emits.
            self.state.transform = replay(crate::geometry::mat_mul(&[s, 0.0, 0.0, s, ox, oy], &saved));
            doc.render(self);
        }
        self.state.transform = saved;
        if ignore_blend {
            self.set_primitive_blend(blend);
        }
        true
    }

    /// `ID2D1RenderTarget::DrawText`: lay `text` out in `fmt` inside `rect` (DIPs) and draw it with
    /// a solid brush. `measuring` is `DWRITE_MEASURING_MODE`; `enable_color` takes the colour-font
    /// path. Returns `true` when every run was painted, `false` if any needed an unimplemented path.
    #[allow(clippy::too_many_arguments)]
    /// DrawText's em gate (the FFI's DrawGlyphRun shares it): `None` draws; `Some(true)` draws nothing and
    /// latches E_INVALIDARG (native defers it to EndDraw with the op's tags, also while recording: r23
    /// 28811); `Some(false)` draws nothing without an error. Only the GDI measuring modes (1/2) gate.
    /// - A NaN in the WORLD transform fails GetGdiCompatibleGlyphMetrics even with no glyphs (16913 dx=nan,
    ///   16524 dy=nan). An inf does not (7671: singular, dy=-inf, empty → S_OK). The world matrix, not
    ///   `device_transform()`: a singular world times an inf translation makes a spurious 0·∞ NaN.
    /// - The 2^15 limit is per laid-out glyph, so empty text never trips it (16015 op9).
    /// - It refuses exactly where DrawGlyphRun's bitmap path does ([`Self::em_matrix_over_bitmap_limit`]):
    ///   an axis-aligned or quarter-turn, uniformly scaled em matrix of 2^15 or more (15125/15058, 16015
    ///   op18, 28811: ×1e6). VM, Times em 20 GDI_NATURAL: ×1e6 and ×2000 refuse, ×1000 passes; 1e6×2000,
    ///   2000×1e6, 1e6×1 and 1×1e6 all pass. A sheared or rotated em passes (28092 op16, 21649 [1e9, 0.57,
    ///   -0, 0.36]), and a non-finite em is never uniform (17208 size inf, 19346 3.4e38 at 192 dpi).
    /// We draw nothing for any huge or non-finite em (the unbounded-raster backstop). A non-positive/NaN size
    /// is the caller's per-op CreateTextFormat reject. DrawTextLayout renders at any device em; its limit
    /// is `gdi_compat_layout_em_rejected`, at layout creation.
    pub fn draw_text_em_gate(&self, em: f32, measuring: u32, has_glyphs: bool) -> Option<bool> {
        if !matches!(measuring, 1 | 2) {
            return None;
        }
        if self.state.transform.iter().any(|v| v.is_nan()) {
            return Some(true);
        }
        if !has_glyphs {
            return None;
        }
        let m = self.device_transform();
        let scale = (m[0] * m[0] + m[1] * m[1]).sqrt().max((m[2] * m[2] + m[3] * m[3]).sqrt()) as f32;
        let device_em = em * scale;
        if !(em > 0.0) || !device_em.is_finite() {
            return Some(false);
        }
        if device_em.abs() < 32768.0 {
            return None;
        }
        Some(self.em_matrix_over_bitmap_limit(em, false))
    }

    /// D2D1_DRAW_TEXT_OPTIONS_CLIP (native `DrawTextLayoutHelper`): clip a text draw to its layout box
    /// `rect` (DIPs). An axis-aligned or quarter-turn world transform pushes an axis-aligned clip at the
    /// current antialias mode. Any other transform pushes the text clip layer (`PushTextClipLayer`): content
    /// bounds = the box, no mask, mask AA = the antialias mode, opacity 1, INITIALIZE_FROM_BACKGROUND (plus
    /// IGNORE_ALPHA on an alpha-ignore target), pushed under a COPY primitive blend that is then restored.
    /// Returns the kind pushed for [`Self::pop_text_clip`] (true = layer).
    pub fn push_text_clip(&mut self, rect: [f32; 4]) -> bool {
        let m = self.get_transform();
        let aa = self.get_antialias_mode();
        if (m[1] == 0.0 && m[2] == 0.0) || (m[0] == 0.0 && m[3] == 0.0) {
            self.push_axis_aligned_clip(rect, aa);
            return false;
        }
        let ignore_alpha = self.target.as_ref().is_some_and(|t| t.desc.alpha_mode == d2d_types::abi::D2D1_ALPHA_MODE_IGNORE);
        let blend = self.get_primitive_blend();
        self.set_primitive_blend(1); // D2D1_PRIMITIVE_BLEND_COPY
        self.push_layer(rect, None, aa, None, 1.0, None, 1 | if ignore_alpha { 2 } else { 0 });
        self.set_primitive_blend(blend);
        true
    }

    /// Pop what [`Self::push_text_clip`] pushed (native `DrawTextPopper`).
    pub fn pop_text_clip(&mut self, layer: bool) {
        if layer {
            self.pop_layer();
        } else {
            self.pop_axis_aligned_clip();
        }
    }

    /// The em the em gate measures for a layout: the largest per-range format em when format ems size every
/// byte of the text, else the base em. A DrawTextLayout whose base size is inf but whose ranges size every
/// character draws at the range ems (r19 24453); the gate is per laid-out glyph.
fn laid_out_em(base: f32, len: usize, formats: &[FormatFace]) -> f32 {
    let mut covered = vec![false; len];
    let mut em = None::<f32>;
    for &(start, end, _, _, fem, _, _) in formats {
        if let Some(v) = fem {
            em = Some(em.map_or(v, |m: f32| m.max(v)));
            covered[start.min(len)..end.min(len)].iter_mut().for_each(|c| *c = true);
        }
    }
    match em {
        Some(v) if covered.iter().all(|&c| c) => v,
        _ => base,
    }
}

/// `CreateGdiCompatibleTextLayout` rejects (E_INVALIDARG) a device em — `em · pixelsPerDip ·
    /// transformScale` — that reaches 2^15 = 32768, the same GetGdiCompatibleGlyphMetrics limit, but
    /// enforced at layout creation rather than per draw. Plain `CreateTextLayout` (natural) never rejects.
    /// Exact edge VM-pinned 2026-11-08 (r11 16180): dpi 96 em 32767.998 S_OK / 32768 E_INVALIDARG; dpi 192
    /// (ppd 2) the edge halves to em 16384 — so it is the device em, not the font em. `transform_scale` is
    /// the max column magnitude of the layout transform (1.0 for the identity transform d2d passes). The
    /// reject is only for a FINITE device em ≥ 2^15: when `em·ppd·scale` overflows to non-finite the layout
    /// takes native's degenerate path and succeeds (r12 rerun 7713: em 3.4e38 × ppd 2 overflows → S_OK).
    pub fn gdi_compat_layout_em_rejected(em: f32, ppd: f32, transform_scale: f32) -> bool {
        let device_em = em * ppd * transform_scale;
        device_em.is_finite() && device_em >= 32768.0
    }

    pub fn draw_text(
        &mut self,
        text: &str,
        font_data: &[u8],
        face_index: u32,
        fmt: &layout::TextFormat,
        measuring: u32,
        rect: [f32; 4],
        brush: &crate::resource::SolidColorBrush,
        params: &RenderingParams,
        enable_color: bool,
    ) -> bool {
        // A NaN/negative layout box (17620/17323 op top=NaN) draws nothing: native's DrawText swallows
        // its failed internal CreateTextLayout and returns S_OK (r12 17323/17356 EndDraw hr 0) — so do
        // NOT latch here, and do it before the em gate (r12 17218: a NaN box under a NaN transform is
        // S_OK). Only the app's own CreateTextLayout returns E_INVALIDARG (make_layout). See
        // `draw_text_rect_rejected`.
        if draw_text_rect_rejected(rect) {
            return true;
        }
        if let Some(refused) = self.draw_text_em_gate(fmt.em, measuring, !text.is_empty()) {
            if refused {
                self.process_new_error(crate::state::E_INVALIDARG);
            }
            return true;
        }
        let Some(font) = shaping::Font::from_bytes(font_data.to_vec(), face_index) else { return false };
        let (origin, w, h) = normalize_layout_rect(rect);
        // A GDI measuring mode (1/2) must feed the layout its GDI-compatible advances + VDMX line
        // metrics, so a DrawText through here matches the fallback-resolving layout path the suite
        // drives (`layout_ex` with the measuring mode) — not plain natural `layout`.
        let opts = layout::LayoutOpts { measuring, pixels_per_dip: if self.get_unit_mode() == 0 { self.get_dpi().0 / 96.0 } else { 1.0 }, transform: Some(self.text_transform()), ..Default::default() };
        let lay = layout::layout_ex(&font, fmt, text, w, h, &opts);
        // Single-font: no fallback faces. Callers that resolve system font fallback lay out with
        // `layout::layout_ex` and drive `draw_layout` directly with the resolved faces.
        self.draw_layout(&lay, origin, font_data, face_index, measuring, brush, params, enable_color, 0, &[], &[])
    }

    /// `ID2D1RenderTarget::DrawTextLayout`: draw an already-computed [`layout::Layout`] at `origin`
    /// (DIP top-left), run by run. Each run's baseline origin is `origin + run.(x,y)`. `faces` are
    /// the system-font-fallback faces layout resolved per text range ([`FaceSpan`]): a run shaped
    /// with a fallback face renders with that face's bytes (its glyph ids are in that face's glyph
    /// space), not the base `font_data`. An empty slice draws every run with the base face.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_layout(
        &mut self,
        lay: &layout::Layout,
        origin: [f32; 2],
        font_data: &[u8],
        face_index: u32,
        measuring: u32,
        brush: &crate::resource::SolidColorBrush,
        params: &RenderingParams,
        enable_color: bool,
        sims: u32,
        faces: &[FaceSpan],
        coords: &[skrifa::raw::types::F2Dot14],
    ) -> bool {
        let mut ok = true;
        // An app's inline objects are not drawn (DrawInlineObject is the app's callback); a trimming sign is.
        for r in lay.drawn_runs() {
            let r = &*r;
            if r.glyphs.is_empty() {
                continue;
            }
            let glyphs: Vec<Glyph> = (0..r.glyphs.len())
                .map(|i| Glyph { id: r.glyphs[i], advance: r.advances[i], offset: [r.offsets[i].0, r.offsets[i].1] })
                .collect();
            // The face this run shaped with: the format/fallback span covering its start, else the base,
            // each with its own bold/oblique simulation.
            // A trimming sign (no text position) takes the sign's own face, the span marked `start == u32::MAX`.
            let fallback = faces.iter().find(|f| if r.text_pos == u32::MAX { f.start == u32::MAX } else { r.text_pos >= f.start && r.text_pos < f.end });
            let (font_data, face_index) = fallback.map_or((font_data, face_index), |f| (f.data, f.index));
            let gr = GlyphRun {
                font_data,
                face_index,
                // Per-range SetFontSize: this run carries its own em (from the format span), so the
                // renderer rasterizes at that size. Equal to the base `em_size` for an unsized run.
                em_size: r.em,
                glyphs: &glyphs,
                baseline_origin: [origin[0] + r.x, origin[1] + r.y],
                // A trimming sign draws its ellipsis natural-measured whatever the layout's mode (VM r19 24261: at
                // a −inf origin a GDI run collapses onto the clip origin, the sign draws nothing).
                measuring_mode: if r.text_pos == u32::MAX { 0 } else { measuring },
                is_sideways: false,
                bidi_level: r.level as u32,
                no_snap: params.no_snap,
                simulations: fallback.map_or(sims, |f| f.sims),
                coords: fallback.map_or(coords, |f| f.coords),
            };
            let gr = match r.orient {
                _ if params.no_snap => gr,
                0 => GlyphRun { baseline_origin: self.layout_snapped(gr.baseline_origin), no_snap: true, ..gr },
                1 => GlyphRun { baseline_origin: self.layout_snapped_vertical(gr.baseline_origin), no_snap: true, ..gr },
                _ => gr,
            };
            // DrawText with colour fonts still draws each run through DrawGlyphRun and meets its checks (VM r25 30968:
            // ClearType under an OUTLINE mode latches E_INVALIDARG); DrawPaintGlyphRun does not (r25 30307 30458 30875).
            ok &= self.draw_oriented(r, &gr, params, |dc, gr| if enable_color && dc.glyph_run_rejected(gr, params) {
                true
            } else if enable_color {
                // Its monochrome glyphs draw from the run as laid out.
                let snapped = GlyphRun { baseline_origin: dc.color_text_baseline(gr, params), no_snap: false, ..*gr };
                dc.draw_color_glyph_run_at(&snapped, Some(gr), brush, 0, params)
            } else {
                dc.draw_glyph_run(gr, brush, params)
            });
        }
        // Underline / strikethrough rectangles, filled with the text brush after the glyph runs
        // (DrawUnderline/DrawStrikethrough follow DrawGlyphRun). DIP rect = origin + the decoration's
        // layout rect; `fill_rectangle` applies the context transform, as the glyph runs do.
        self.fill_decorations(lay, origin, measuring, params, |dc, r| dc.fill_rectangle(r, brush));
        ok
    }

    /// DirectWrite's layout pixel snapping (`IDWritePixelSnapping`) of a horizontal run's baseline: the device grid of
    /// the transform's linear part scaled by pixels-per-DIP plus its translation in DIPs, unscaled. Under a translated
    /// transform at a DPI other than 96 the run then draws off the whole pixel by (pixelsPerDip - 1)·translation (VM,
    /// 144 dpi rect-top sweeps: upright under dy 10.3 and flipped under dy 183.75 switch rows where this rounds; r25 30413).
    fn layout_snapped(&self, o: [f32; 2]) -> [f32; 2] {
        let (d, w) = (self.device_transform(), self.get_transform());
        snap_baseline(o, &[d[0], d[1], d[2], d[3], w[4] as f64, w[5] as f64], false)
    }

    /// The same snapping of a run's origin in a vertical layout, a rotated run's or an upright column's centre line:
    /// only its layout x (the column's cross axis) snaps, through the transform's x scale (m12 = 0) or, under a quarter
    /// turn, its m12, times pixels-per-DIP, with the translation along that axis again unscaled; the place along the
    /// column stays the renderer's to round to its oversample grid (dwrite reading; VM sb33, a vertical Arabic run at
    /// 120 dpi under identity and x-flipped worlds swept in 0.1 DIP translations: 18 of 18 placements; r25 30900).
    fn layout_snapped_vertical(&self, o: [f32; 2]) -> [f32; 2] {
        let w = self.get_transform();
        let (dx, dy) = if self.get_unit_mode() == 1 { (96.0, 96.0) } else { self.get_dpi() };
        // GetPixelsPerDip: dpiY times f32(1/96); GetCurrentTransform scales its x column and dx by dpiX/dpiY.
        let (r, ppd) = (dx / dy, dy * (1.0 / 96.0));
        let (s, t) = if w[1] == 0.0 && w[3] != 0.0 {
            (w[0] * r * ppd, w[4] * r)
        } else if w[0] == 0.0 && w[2] != 0.0 {
            (w[1] * ppd, w[5])
        } else {
            return o;
        };
        if s == 0.0 {
            return o;
        }
        let mut v = s * o[0] + t;
        if v.abs() < 8388608.0 {
            v = layout::round_half_up(v);
        }
        [(v - t) * (1.0 / s), o[1]]
    }

    /// Draw a layout run by its vertical orientation ([`layout::GlyphRun::orient`]). A horizontal run draws
    /// as is. A rotated run draws turned 90° clockwise about its origin (native's renderer applies the
    /// glyph-orientation transform for DWRITE_GLYPH_ORIENTATION_ANGLE_90_DEGREES). An upright run's sideways
    /// glyphs, turned back by that same rotation, stand upright: each draws on its own, centred on the column
    /// and hanging from its vertical origin, the pen moving down the column.
    /// The baseline DrawText's text renderer hands a colour glyph run: pixel-snapped (r25 30167, 30193), except under
    /// NO_SNAP in the natural modes, where the run keeps its sub-pixel baseline for the colour placement's quarter grid
    /// (VM probe-fam: a family emoji under NO_SNAP at natural measuring lands at y .25 from a baseline near 94.22, 948 px
    /// off snapped; GDI-measured and aliased runs still snap, r25 30233_5).
    fn color_text_baseline(&self, gr: &GlyphRun, params: &RenderingParams) -> [f32; 2] {
        let natural = gr.measuring_mode == 0 && self.get_text_antialias_mode() != AA_ALIASED && !matches!(params.rendering_mode, 1..=3);
        if params.no_snap && natural {
            gr.baseline_origin
        } else {
            snap_baseline(gr.baseline_origin, &self.device_transform(), false)
        }
    }

    fn draw_oriented(&mut self, r: &layout::GlyphRun, gr: &GlyphRun, params: &RenderingParams, mut draw: impl FnMut(&mut Self, &GlyphRun) -> bool) -> bool {
        match r.orient {
            1 => {
                let saved = self.get_transform();
                let p = gr.baseline_origin;
                self.state.transform = crate::geometry::mat_mul(&[0.0, 1.0, -1.0, 0.0, p[0] + p[1], p[1] - p[0]], &saved);
                let ok = draw(self, gr);
                self.state.transform = saved;
                ok
            }
            2 => {
                let Ok(font) = FontRef::from_index(gr.font_data, gr.face_index) else { return false };
                let upem = font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f32;
                let scale = gr.em_size / upem;
                let gm = font.glyph_metrics(Size::unscaled(), LocationRef::default());
                let vmtx = font.vmtx().ok();
                // A face without vmtx hangs its glyphs from its typo ascent, the top of the typo ascent-plus-descent its
                // vertical advance spans (VM r25 30076: Segoe UI Symbol's vertical ellipsis sign).
                let ascent = match (&vmtx, font.os2()) {
                    (None, Ok(os2)) => os2.s_typo_ascender() as f32,
                    _ => font.hhea().map_or(upem, |h| h.ascender().to_i16() as f32),
                };
                let mut places: Vec<(u16, [f32; 2])> = Vec::new();
                let mut pen = gr.baseline_origin[1];
                let btt = gr.bidi_level & 1 != 0;
                // A GDI-natural run in the outline path takes its glyph origins at one pixel per DIP with no transform,
                // as the outline renderer measures them, its column snapped to whole device pixels and its glyphs' y as
                // laid out (VM probes, Yu Gothic UI Bold upright at em 95.3 and 144 dpi: the glyph sits on the column
                // rounded half up, centred on half its advance rounded to whole DIPs, for kana and kanji of four widths).
                // A natural run in the outline path snaps its column too (VM probes, the same face at em 70 to 117,
                // upright and oblique).
                // The natural bitmap path snaps its column too, after the layout rounds it in the column frame (VM r25 30497:
                // Segoe UI Emoji upright in a bottom-to-top column; 30141; vertical trimming signs in seven faces).
                // GDI natural's bitmap path snaps it as well (VM r25 30046: Segoe UI Emoji upright at em 78 in a GDI natural
                // column, regular and bold, a pixel left of native unsnapped), and so does GDI classic's (VM sweeps of the
                // column's phase through the origin and the layout width, both reading and flow directions: the glyph steps
                // where origin plus line offset crosses a half pixel), but not under the oblique simulation (r25 30248, Tahoma
                // fallback kanji at 192 DPI, and 30333, Segoe UI Emoji bottom to top, both italic: exact unsnapped).
                let outline_path =
                    recommended_rendering_mode(gr, &self.device_transform(), self.get_text_antialias_mode(), params) == MODE_OUTLINE;
                let outline = matches!(gr.measuring_mode, 1 | 2) && outline_path;
                let snap = !gr.no_snap && (outline_path || gr.measuring_mode != 1 || gr.simulations & 2 == 0);
                let mut cx = if snap { self.layout_snapped_vertical(gr.baseline_origin)[0] } else { gr.baseline_origin[0] };
                // The renderer draws the run turned about its baseline origin (ox, oy), DirectWrite's orientation transform
                // putting the turn's x offset at ox + oy in f32, so the column comes back from the device mapping as
                // (ox + oy) - oy; a GDI mode's bitmap path floors that to its pixel, so a snapped column the f32 sum leaves a
                // hair short of its whole pixel draws a pixel left (VM probes of r25 30936: Segoe UI Emoji upright in a GDI
                // classic column, the column at 119 to 185 with oy 137.16 a pixel left, 118, 375 and 631 not, nor 119 with
                // oy 117.16).
                // ponytail: an axis-aligned unmirrored device transform only; a turned or mirrored one maps the error elsewhere.
                let dt = self.device_transform();
                let oy = gr.baseline_origin[1];
                if snap && matches!(gr.measuring_mode, 1 | 2) && !outline_path && dt[1] == 0.0 && dt[2] == 0.0 && dt[0] > 0.0 && (cx + oy) - oy < cx {
                    cx -= (1.0 / dt[0]) as f32;
                }
                for g in gr.glyphs {
                    let gid = GlyphId::new(g.id as u32);
                    let half = gm.advance_width(gid).unwrap_or(0.0) / 2.0;
                    // Vertical origin: the glyph top plus its top side bearing (vmtx), else the ascent.
                    let vorg = match (&vmtx, gm.bounds(gid)) {
                        (Some(v), Some(b)) => b.y_max + v.side_bearing(gid).unwrap_or(0) as f32,
                        _ => ascent,
                    };
                    // Each glyph is placed here, so it draws as a left-to-right run of one.
                    let one = [Glyph { id: g.id, advance: 0.0, offset: [0.0, 0.0] }];
                    // A bottom-to-top run (odd bidi level) starts at the column's foot: each glyph's em box
                    // spans the advance above the pen.
                    let top = if btt { pen - g.advance } else { pen };
                    // DirectWrite measures a sideways glyph's vertical origin through its metrics converter, so a GDI
                    // mode rounds it to whole pixels (VM r25 30352: Segoe UI Emoji upright in a GDI-classic column).
                    let (vorg_du, adv_du) = (vorg.round(), gm.advance_width(gid).unwrap_or(0.0).round());
                    let oblique = gr.simulations & 2 != 0;
                    let vorg = if outline {
                        let du = gdi_natural_vertical_origin(vorg_du as i32, adv_du as i32, gr.em_size, upem as i32, oblique);
                        client_units(&GlyphRun { glyphs: &one, ..*gr }, &IDENTITY, upem as u16)(du)
                    } else if matches!(gr.measuring_mode, 1 | 2) {
                        // The bitmap path measures it as the scaler does at the device ppem.
                        let (run1, dt) = (GlyphRun { glyphs: &one, ..*gr }, self.device_transform());
                        let ppem = gr.em_size * (dt[2] * dt[2] + dt[3] * dt[3]).sqrt() as f32;
                        let du = gdi_natural_vertical_origin(vorg_du as i32, adv_du as i32, ppem, upem as i32, oblique);
                        client_units(&run1, &dt, upem as u16)(du)
                    } else {
                        vorg * scale
                    };
                    // A GDI mode centres the glyph on its half advance in whole pixels too, on the snapped column (VM r25 30488
                    // and 30743, GDI natural: 210 and 91 px off unrounded; 30046, GDI natural; VM probes of Segoe UI Emoji's woman
                    // upright at em 78, GDI classic: a pixel right of the design half's place).
                    // Under the oblique simulation's slant, which leaves the transform unaligned, a GDI mode centres on the
                    // unhinted width on the 1/16-pixel grid at the integer ppem, a whole pixel wider under the bold simulation,
                    // as a GDI-compatible advance (VM probes, GetGdiCompatibleGlyphMetrics sideways of Yu Gothic UI Bold at em 70
                    // to 117: 25 and 25 of 25; em 103.9's 1449 units centre on 37 DIPs, not the design 36.5; VM DDI capture of
                    // r25 30046's flag bold and oblique upright in a GDI natural column: the mesh 0.5 DIP left of the oblique
                    // one's; VM probes of Segoe UI Emoji's man and woman joined, bold and oblique at em 78, bitmap path).
                    let slanted_du = || {
                        let ppem = (gr.em_size as f64 + 0.5).floor().max(1.0);
                        let bold = if gr.simulations & 1 != 0 && adv_du != 0.0 { 16.0 } else { 0.0 };
                        let sixteenths = (adv_du as f64 * ppem * 16.0 / upem as f64 + 0.5).floor() + bold;
                        (sixteenths / 16.0 * upem as f64 / ppem + 0.5).floor() as i32
                    };
                    let half = if outline {
                        // The centring half is that of the sideways GDI-natural advance: the hinted whole-pixel width
                        // upright, the slanted width above under the oblique simulation.
                        let du = if oblique {
                            slanted_du()
                        } else {
                            gdi::gdi_compat_advance_du(&font, g.id, gr.em_size, 1.0, false, None, upem as u16, gr.simulations & 1 != 0)
                                .unwrap_or(adv_du as i32)
                        };
                        client_units(&GlyphRun { glyphs: &one, ..*gr }, &IDENTITY, upem as u16)(du) / 2.0
                    } else if matches!(gr.measuring_mode, 1 | 2) {
                        // The bitmap path centres on half the sideways advance width DirectWrite's glyph-run analysis takes
                        // from its sideways glyph metrics: in a GDI mode the GDI-compatible advance, whole pixels at the
                        // rasterizer's ppem (em times the device scale, rounded) and a pixel wider bold, read back in
                        // design units and converted to client units at the exact em, rounding to whole device pixels
                        // again (dwrite reading; VM probes, Segoe UI Emoji's woman upright under GDI natural at 144 DPI,
                        // em 12 to 29.6: 49 of 50 placements; r25 30743); on the slanted width under the oblique.
                        let (run1, dt) = (GlyphRun { glyphs: &one, ..*gr }, self.device_transform());
                        let du = if oblique {
                            slanted_du()
                        } else {
                            let ppd = (dt[2] * dt[2] + dt[3] * dt[3]).sqrt() as f32;
                            let bold = gr.simulations & 1 != 0;
                            gdi::gdi_compat_advance_du(&font, g.id, gr.em_size, ppd, gr.measuring_mode == 1, None, upem as u16, bold)
                                .unwrap_or(adv_du as i32 + if bold { (upem / 50.0).round() as i32 } else { 0 })
                        };
                        client_units(&run1, &dt, upem as u16)(du) / 2.0
                    } else {
                        half * scale
                    };
                    // The glyph's offset runs along the column with the pen (down a top-to-bottom column, up a bottom-to-top
                    // one) and across it to the right in either (VM r25 30497: Segoe UI Emoji's family in a bottom-to-top
                    // column).
                    // Under the oblique simulation DirectWrite slants each sideways glyph about its own place: its across offset
                    // gains the slant's share along the column, k·across, whole pixels under a GDI mode, and the glyph shears
                    // about its own centre, not the column's (VM probes of r25 30046: Segoe UI Emoji's man and woman joined, GDI
                    // natural, report advance offsets -5 and 8 for ascender offsets -16 and 25, and 0 upright).
                    // ponytail: the layout's own offsets stay unslanted; a custom renderer would see them so.
                    let slant = if gr.simulations & 2 != 0 && g.offset[1] != 0.0 {
                        let a = raster::OBLIQUE_SHEAR as f32 * g.offset[1];
                        if matches!(gr.measuring_mode, 1 | 2) { a.round() } else { a }
                    } else {
                        0.0
                    };
                    let (along, across) = if btt { (-(g.offset[0] + slant), g.offset[1]) } else { (g.offset[0] + slant, g.offset[1]) };
                    let mut o = [cx - half + across, top + vorg + along];
                    if gr.simulations & 2 != 0 {
                        // The oblique simulation slants a sideways glyph about its own left origin, and DWrite lowers the
                        // vertical origin to match: design metrics take (22272·(advance/2)) >> 16 design units off it, and
                        // the GDI-natural outline path measures it as the scaler does ([`gdi_natural_vertical_origin`]).
                        // Drawn here as a slant about the glyph's centre (the column's, shifted by its across offset), the glyph's origin is that vertical origin plus
                        // the slant of its half advance (dwrite reading; VM probes, Yu Gothic UI Bold oblique upright at em
                        // 70 to 117, natural and GDI natural).
                        // ponytail: GDI classic and the GDI bitmap path keep the unadjusted origin (unprobed).
                        let k = raster::OBLIQUE_SHEAR as f32;
                        let vorg = match gr.measuring_mode {
                            0 => Some((vorg_du - ((22272 * (adv_du as i32 >> 1)) >> 16) as f32) * scale),
                            1 | 2 => Some(vorg),
                            _ => None,
                        };
                        if let Some(v) = vorg {
                            o[1] = top + v + k * half + along;
                        }
                        // Upright in a vertical run the glyph is sideways to DWrite, whose oblique simulation then shears
                        // the glyph's y by its x (the scaler's sideways slant: m00 −= k·m01, m10 −= k·m11) about its
                        // vertical origin on the column's centre line, not its x by its y. The run draws under one shear
                        // about x = 0, so each glyph's pivot comes off its y.
                        o[1] -= k * (cx + across);
                    }
                    places.push((g.id, o));
                    pen = if btt { pen - g.advance } else { pen + g.advance };
                }
                // The column's glyphs draw as one run, each at its own place (no advance, the place as its offset), so the
                // renderer merges their coverage as a sideways run's (VM probes, Segoe UI Emoji's man and woman joined, upright
                // at em 78: drawn one by one, their overlap blended twice).
                let Some(&(_, o0)) = places.first() else { return true };
                let glyphs: Vec<Glyph> = places.iter().map(|&(id, o)| Glyph { id, advance: 0.0, offset: [o[0] - o0[0], o0[1] - o[1]] }).collect();
                // Only the column snaps: an upright glyph's place along it is the renderer's to quantize, as a sideways
                // run's is natively — continuous in the outline path, the oversample grid in the bitmap path, under
                // natural measuring (VM probes, Yu Gothic UI Bold upright at em 20 and 117, the origin swept in 0.05 DIP
                // steps: native's top edge moves with it where ours stepped whole pixels). GDI natural's bitmap path
                // keeps the whole-pixel step (same probes).
                let no_snap = gr.no_snap || outline || gr.measuring_mode == 0;
                let k = if gr.simulations & 2 != 0 { raster::OBLIQUE_SHEAR as f32 } else { 0.0 };
                let _turned = crate::geometry::realization::SidewaysCache::set(outline_path.then_some(k));
                let run = GlyphRun { glyphs: &glyphs, baseline_origin: o0, bidi_level: 0, no_snap, ..*gr };
                if gr.simulations & 2 != 0 {
                    let saved = self.get_transform();
                    self.state.transform = crate::geometry::mat_mul(&[1.0, k, 0.0, 1.0, 0.0, 0.0], &saved);
                    let ok = draw(self, &GlyphRun { simulations: (gr.simulations & !2) | SIM_SIDEWAYS_OBLIQUE, ..run });
                    self.state.transform = saved;
                    ok
                } else {
                    draw(self, &run)
                }
            }
            _ => draw(self, gr),
        }
    }

    /// D2D's text renderer fills each underline / strikethrough under its own antialias mode: ALIASED,
    /// and at least one device pixel thick, when the text is aliased (text antialias mode or rendering
    /// params) or measured in a GDI mode; per-primitive otherwise. The context's mode comes back after.
    fn fill_decorations(&mut self, lay: &layout::Layout, origin: [f32; 2], measuring: u32, params: &RenderingParams, mut fill: impl FnMut(&mut Self, [f32; 4])) {
        if lay.decorations.is_empty() {
            return;
        }
        let aliased = self.get_text_antialias_mode() == AA_ALIASED || params.rendering_mode == MODE_ALIASED || measuring != 0;
        // One device pixel in DIPs: |row 1| / |det| of the world transform scaled by dpi/96 (96 under
        // the pixel unit mode).
        let (dx, dy) = if self.get_unit_mode() == 1 { (96.0, 96.0) } else { self.get_dpi() };
        let (sx, sy) = (dx / 96.0, dy / 96.0);
        let m = self.get_transform();
        let (a, b) = (m[0] * sx, m[1] * sy);
        let device_px = (b * b + a * a).sqrt() / ((m[3] * sy) * a - (m[2] * sx) * b).abs();
        let saved = self.get_antialias_mode();
        self.set_antialias_mode(if aliased { 1 } else { 0 });
        for d in &lay.decorations {
            let thickness = if aliased { device_px.max(d.thickness) } else { d.thickness };
            let x = origin[0] + d.x;
            // DWrite's layout pixel snapper (TextLayout::Draw): the renderer's text transform is the world transform
            // with its x column and dx scaled by dpiX/dpiY ([`DeviceContext::layout_snapped_vertical`]) and its
            // translation otherwise in DIPs. A horizontal layout snaps the underline's baseline y and its offset through
            // m22·ppd and dy when m12 is 0, or through m21·ppd and dx when m11 is 0; a vertical one snaps the baseline x
            // through m11·ppd and dx or m12·ppd and dy. Any other transform snaps nothing, the offset included, and a
            // strikethrough is never snapped (VM r21 26167, r19 24453 offset 0.744 to 1 in pixel units at 192 dpi; r25
            // 30489: under a 0.78 y-shear the aliased band fits base + raw offset).
            let snapper = |vertical: bool| -> Option<(f32, f32)> {
                let (k, ppd) = (dx / dy, dy * (1.0 / 96.0));
                let t = [m[0] * k, m[1], m[2] * k, m[3], m[4] * k, m[5]];
                let (s, o) = match (t[1] == 0.0 && t[3] != 0.0, t[0] == 0.0 && t[2] != 0.0, vertical) {
                    (true, _, false) => (t[3] * ppd, t[5]),
                    (true, _, true) => (t[0] * ppd, t[4]),
                    (false, true, false) => (t[2] * ppd, t[4]),
                    (false, true, true) => (t[1] * ppd, t[5]),
                    _ => return None,
                };
                (s != 0.0).then_some((s, o))
            };
            let snap = |v: f32, (s, o): (f32, f32)| (layout::round_half_up(s * v + o) - o) * (1.0 / s);
            // The offset rounds at the same scale to at least one whole unit, times the stored reciprocal.
            let snap_off = |v: f32, (s, _): (f32, f32)| {
                let r = layout::round_half_up(s * v);
                (if r == 0.0 && s * v != 0.0 { (s * v).signum() } else { r }) * (1.0 / s)
            };
            let y = match (d.strike || !d.base.is_finite(), snapper(false)) {
                (true, _) => origin[1] + d.y,
                (false, Some(sn)) => snap(origin[1] + d.base, sn) + snap_off(d.raw_offset, sn),
                (false, None) => origin[1] + d.base + d.raw_offset,
            };
            if d.orient == 1 {
                // Turned with its vertical run: the rect below the origin in the turned frame, its baseline (device x
                // here) and offset snapped by the vertical snapper (r25 30936).
                let (bx, off) = match snapper(true) {
                    Some(sn) if !d.strike && x.is_finite() => (snap(x, sn), snap_off(d.offset, sn)),
                    _ => (x, d.offset),
                };
                let p = [bx, origin[1] + d.y];
                self.state.transform = crate::geometry::mat_mul(&[0.0, 1.0, -1.0, 0.0, p[0] + p[1], p[1] - p[0]], &m);
                fill(self, [p[0], p[1] + off, p[0] + d.width, p[1] + off + thickness]);
                self.state.transform = m;
                continue;
            }
            fill(self, [x, y, x + d.width, y + thickness]);
        }
        self.set_antialias_mode(saved);
    }

    /// [`draw_glyph_run`] with a gradient/bitmap/image brush: the glyph coverage is painted with the
    /// brush as a per-pixel foreground ([`render_glyph_run_brush`]).
    pub fn draw_glyph_run_brush(&mut self, run: &GlyphRun, brush: &crate::PixelBrush, params: &RenderingParams) -> bool {
        let _a = self.glyph_api(run, brush.batch);
        // Native rejects an A8 bitmap brush first (r24 29873), then validates the brush's domain, then the
        // rendering params (EnsureTextRenderingParams), and only then voids a singular bitmap/image brush, so a
        // params conflict still latches under one (r23 28092/28726).
        if !self.ready_to_draw() || self.a8_brush_rejected(brush) || !self.check_domain(brush.domain) || self.glyph_run_rejected(run, params) {
            return true;
        }
        // The brush is placed under the device transform of the draw, as for a fill.
        let grad_rr = self.realize_brush_for_draw(brush);
        let brush = grad_rr.as_ref().unwrap_or(brush);
        if brush.brush.singular() {
            // A HIGH_QUALITY_CUBIC image brush over a bitmap, or a crop of one, that the draw places singular fails it where
            // the run's ink box holds a pixel centre inside the clip, unless its matrix is rank one and clamps both ways (VM
            // probes, DrawText under rank-one and zero matrices, every extend pair, collapsed and tiny transforms; r24 29837,
            // a crop).
            // ponytail: an image brush only; a bitmap brush's singular HIGH_QUALITY_CUBIC text draw is unprobed.
            use crate::brush::{Brush, BrushOrigin};
            use crate::sampling::Extend;
            if let Brush::Bitmap { transform: m, interpolation: 5, extend, origin: BrushOrigin::BitmapImage | BrushOrigin::StagedImage, .. } = &brush.brush {
                let fails = brush.image.is_none() && (m[..4] == [0.0; 4] || *extend != [Extend::Clamp, Extend::Clamp]);
                if fails && self.recording.is_none() && run_ink_box(run, &self.device_transform()).is_some_and(|b| self.draw_pixels(b).is_some()) {
                    self.process_new_error(crate::state::INTERMEDIATE_TOO_LARGE);
                }
            }
            return true;
        }
        let (dpi_x,dpi_y)=self.get_dpi();
        let params=&RenderingParams { dpi: [dpi_x,dpi_y], blend: crate::context::blend_of(self.get_primitive_blend()), raster_clip: self.raster_clip(), clip_px: self.centre_clip_px(), ..*params };
        if self.recording.is_some() {
            self.record_glyph_run(run, crate::command::TextFg::Brush(brush.brush.clone()), None, params);
            return true;
        }
        if self.glyph_run_em_rejected(run, params) {
            return true;
        }
        let dt = self.device_transform();
        let aa = self.text_aa_for_draw();
        // An effect image brush is realized over the run's ink bounds (native's glyph-run redirection
        // rect, imgbrush-02): render the coverage once to find them.
        let per_draw = match (brush.image.is_some() || brush.crop.is_some(), &self.target) {
            (true, Some(t)) => {
                let mut scratch = crate::context::bitmap_target(t.desc.width, t.desc.height, t.desc.dpi_x);
                render_glyph_run_brush(&mut scratch, &dt, run, &crate::brush::Brush::Solid([255.0; 4]), aa, params);
                ink_bounds(&scratch).and_then(|d| self.image_brush_for_draw(brush, d, true, true))
            }
            _ => None,
        };
        // With the ink rect now found (see `ink_bounds`), an image brush realizes per draw as a fill
        // does, so `image_brush_for_draw`'s `None` for an image-brush run means the placement native
        // cannot size (a degenerate/huge transform) — void, as FillRectangle/FillGeometry (VM-probed:
        // a 3.4e38-matrix effect image brush voids DrawGlyphRun). An empty run (no ink) voids too, which
        // paints nothing either way.
        if brush.image.is_some() && per_draw.is_none() {
            return true;
        }
        let brush = per_draw.as_ref().unwrap_or(brush);
        match &mut self.target {
            Some(t) => render_glyph_run_brush(t, &dt, run, &brush.brush, aa, params),
            None => true,
        }
    }

    /// [`draw_layout`] with a gradient/bitmap/image brush. Under ENABLE_COLOR_FONT a colour glyph paints its
    /// own palette colours and only the monochrome glyphs take the brush. Shares the run layout with `draw_layout`.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_layout_brush(
        &mut self,
        lay: &layout::Layout,
        origin: [f32; 2],
        font_data: &[u8],
        face_index: u32,
        measuring: u32,
        brush: &crate::PixelBrush,
        params: &RenderingParams,
        enable_color: bool,
        sims: u32,
        faces: &[FaceSpan],
        coords: &[skrifa::raw::types::F2Dot14],
    ) -> bool {
        // A brush is placed once, under the caller's transform: a vertical run turns the transform it draws under, not the
        // brush, bitmap (r25 30090) and gradient (r25 30748, 30923) alike, so its runs get the placed brush without its
        // definition.
        let placed = brush.def.is_some()
            .then(|| crate::PixelBrush { def: None, ..self.realize_brush_for_draw(brush).unwrap_or_else(|| brush.clone()) });
        let brush = placed.as_ref().unwrap_or(brush);
        let mut ok = true;
        for r in lay.drawn_runs() {
            let r = &*r;
            if r.glyphs.is_empty() {
                continue;
            }
            let glyphs: Vec<Glyph> = (0..r.glyphs.len())
                .map(|i| Glyph { id: r.glyphs[i], advance: r.advances[i], offset: [r.offsets[i].0, r.offsets[i].1] })
                .collect();
            // A trimming sign (no text position) takes the sign's own face, the span marked `start == u32::MAX`.
            let fallback = faces.iter().find(|f| if r.text_pos == u32::MAX { f.start == u32::MAX } else { r.text_pos >= f.start && r.text_pos < f.end });
            let (font_data, face_index) = fallback.map_or((font_data, face_index), |f| (f.data, f.index));
            let gr = GlyphRun {
                font_data,
                face_index,
                // Per-range SetFontSize: this run carries its own em (from the format span), so the
                // renderer rasterizes at that size. Equal to the base `em_size` for an unsized run.
                em_size: r.em,
                glyphs: &glyphs,
                baseline_origin: [origin[0] + r.x, origin[1] + r.y],
                // A trimming sign draws its ellipsis natural-measured whatever the layout's mode (VM r19 24261: at
                // a −inf origin a GDI run collapses onto the clip origin, the sign draws nothing).
                measuring_mode: if r.text_pos == u32::MAX { 0 } else { measuring },
                is_sideways: false,
                bidi_level: r.level as u32,
                no_snap: params.no_snap,
                simulations: fallback.map_or(sims, |f| f.sims),
                coords: fallback.map_or(coords, |f| f.coords),
            };
            let gr = match r.orient {
                _ if params.no_snap => gr,
                0 => GlyphRun { baseline_origin: self.layout_snapped(gr.baseline_origin), no_snap: true, ..gr },
                1 => GlyphRun { baseline_origin: self.layout_snapped_vertical(gr.baseline_origin), no_snap: true, ..gr },
                _ => gr,
            };
            ok &= self.draw_oriented(r, &gr, params, |dc, gr| if enable_color && dc.glyph_run_rejected(gr, params) {
                true
            } else if enable_color {
                let snapped = GlyphRun { baseline_origin: dc.color_text_baseline(gr, params), no_snap: false, ..*gr };
                dc.draw_color_glyph_run_brush(&snapped, Some(gr), brush, params)
            } else {
                dc.draw_glyph_run_brush(gr, brush, params)
            });
        }
        self.fill_decorations(lay, origin, measuring, params, |dc, r| dc.fill_rectangle_brush(r, brush));
        ok
    }

    /// [`draw_text`] with a gradient/bitmap/image brush.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_text_brush(
        &mut self,
        text: &str,
        font_data: &[u8],
        face_index: u32,
        fmt: &layout::TextFormat,
        measuring: u32,
        rect: [f32; 4],
        brush: &crate::PixelBrush,
        params: &RenderingParams,
    ) -> bool {
        let Some(font) = shaping::Font::from_bytes(font_data.to_vec(), face_index) else { return false };
        let opts = layout::LayoutOpts { measuring, pixels_per_dip: if self.get_unit_mode() == 0 { self.get_dpi().0 / 96.0 } else { 1.0 }, transform: Some(self.text_transform()), ..Default::default() };
        let lay = layout::layout_ex(&font, fmt, text, (rect[2] - rect[0]).max(0.0), (rect[3] - rect[1]).max(0.0), &opts);
        self.draw_layout_brush(&lay, [rect[0], rect[1]], font_data, face_index, measuring, brush, params, false, 0, &[], &[])
    }

    /// `DrawText` with system font fallback (what native's DrawText does, and what a drop-in through
    /// the C API must do): characters the base face can't cover are mapped to a fallback family the
    /// same way DirectWrite's `IDWriteFontFallback::MapCharacters` does ([`fallback::map_characters`]),
    /// and each resolved run renders with that face. `load_family(family, byte, ch)` turns a candidate
    /// family name into its bytes + face index (the caller resolves it through the font collection — the
    /// system one for the FFI, on-disk for the suite); [`fallback::OWN_FAMILY`] asks for the format's own
    /// family at `byte` in the format's collection. Without fallback a CJK/Thai/Arabic run renders tofu, so this is
    /// the path the real `DrawText` uses; the single-font [`draw_text`] stays for runs with no fallback.
    #[allow(clippy::too_many_arguments)]
    /// Native `GetTextTransform`: the world transform's 2×2 with the x column scaled by the DPI aspect
    /// (`dpiX/dpiY`, 1 in unit mode PIXELS), as DrawText hands it to its GDI-compatible layout.
    fn text_transform(&self) -> [f32; 4] {
        let t = self.get_transform();
        let (dx, dy) = self.get_dpi();
        let k = if self.get_unit_mode() == 1 || dy == 0.0 { 1.0 } else { dx / dy };
        [t[0] * k, t[1], t[2] * k, t[3]]
    }

    pub fn draw_text_fallback(
        &mut self,
        text: &str,
        base_bytes: &[u8],
        base_index: u32,
        locale: &str,
        fmt: &layout::TextFormat,
        measuring: u32,
        rect: [f32; 4],
        brush: &crate::resource::SolidColorBrush,
        params: &RenderingParams,
        enable_color: bool,
        sims: u32,
        coords: &[skrifa::raw::types::F2Dot14],
        shaper_vars: &[(u32, f32)],
        // IDWriteTextLayout per-call layout inputs (empty for DrawText): typography feature ranges,
        // trimming, and inline-object placeholders, threaded into `layout_ex` so DrawTextLayout exercises
        // the core's trimming/feature/inline implementation rather than the format-only path.
        features: &[layout::Feature],
        trimming: Option<layout::Trimming>,
        inline: &[layout::InlineObject],
        formats: &[FormatFace],
        // Per-range underline / strikethrough as BYTE ranges (empty for DrawText and the FFI).
        underline: &[(usize, usize)],
        strikethrough: &[(usize, usize)],
        load_family: impl Fn(&str, usize, char) -> Option<(Vec<u8>, u32, u32)>,
    ) -> bool {
        // DrawText's rect is both the layout box and the draw origin; native flips an inverted rect, so
        // normalize to the top-left origin + absolute extents, then lay out the box and draw at it.
        let (origin, w, h) = normalize_layout_rect(rect);
        self.draw_text_fallback_box(text, base_bytes, base_index, locale, fmt, measuring, origin, w, h, brush, params, enable_color, sims, coords, shaper_vars, features, trimming, inline, formats, underline, strikethrough, None, load_family)
    }

    /// Lay out `text` in a `max_width`×`max_height` box and draw it at `origin` — the box size kept
    /// independent of the origin (unlike the rect form, where a non-finite origin cancels the box extent
    /// in f32: `(-inf) + w - (-inf) = NaN`). This matches native's `CreateTextLayout(width,height)` +
    /// `DrawTextLayout(origin)`: a `DrawTextLayout` with a non-finite origin (fuzz: inf/NaN/3.4e38) still
    /// lays out with its finite box, and the glyph-run path collapses the off-canvas run to the clip
    /// origin (`render_glyph_run_fg`). Rejection is on the box alone (NaN/negative), never the origin —
    /// native swallows its failed internal `CreateTextLayout` and returns S_OK, so do NOT latch here.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_text_fallback_box(
        &mut self,
        text: &str,
        base_bytes: &[u8],
        base_index: u32,
        locale: &str,
        fmt: &layout::TextFormat,
        measuring: u32,
        origin: [f32; 2],
        max_width: f32,
        max_height: f32,
        brush: &crate::resource::SolidColorBrush,
        params: &RenderingParams,
        enable_color: bool,
        sims: u32,
        coords: &[skrifa::raw::types::F2Dot14],
        shaper_vars: &[(u32, f32)],
        features: &[layout::Feature],
        trimming: Option<layout::Trimming>,
        inline: &[layout::InlineObject],
        formats: &[FormatFace],
        // Per-range underline / strikethrough as BYTE ranges (empty for DrawText and the FFI).
        underline: &[(usize, usize)],
        strikethrough: &[(usize, usize)],
        // The layout's own pixelsPerDip (the `CreateGdiCompatibleTextLayout` argument a DrawTextLayout's
        // layout was made with), or `None` for DrawText's internal layout: the context's DPI under its unit
        // mode (PIXELS has the em already in device pixels, ppd = 1).
        pixels_per_dip: Option<f32>,
        load_family: impl Fn(&str, usize, char) -> Option<(Vec<u8>, u32, u32)>,
    ) -> bool {
        // The failed internal CreateTextLayout of a NaN box draws nothing silently, ahead of the em gate.
        if !(max_width >= 0.0) || !(max_height >= 0.0) {
            return true;
        }
        if let Some(refused) = self.draw_text_em_gate(Self::laid_out_em(fmt.em, text.len(), formats), measuring, !text.is_empty()) {
            if refused && pixels_per_dip.is_none() {
                self.process_new_error(crate::state::E_INVALIDARG);
            }
            return true;
        }
        let Some(base_font) = shaping::Font::from_bytes_var(base_bytes.to_vec(), base_index, shaper_vars) else { return false };
        // The per-range format faces (a face whose bytes fail to parse is dropped: the base/fallback face then
        // applies there), then the fallback resolved against them, as `assemble_spans` orders the spans.
        let fmt_fonts: Vec<FmtFont> = formats.iter()
            .filter_map(|&(s, e, b, i, em, fs, axes)| {
                let coords = variations::normalized_from_dwrite(b, i, axes);
                let axes = if coords.is_empty() { &[] } else { axes };
                shaping::Font::from_bytes_var(b.to_vec(), i, axes).map(|sf| (s, e, b, i, em, fs, sf, coords))
            })
            .collect();
        let trimming = upright_sign(trimming, fmt, locale, text.len(), &load_family);
        let sign_face = ellipsis_face(trimming, fmt, locale, text.len(), &load_family);
        let trimming = trimming.map(|t| layout::Trimming { sign: sign_face.as_ref().and_then(|f| f.1), ..t });
        let (fam, spans) = resolve_fallback(text, locale, &base_font, &fmt_fonts, &[underline, strikethrough].concat(), load_family);
        let (font_spans, mut face_spans) = assemble_spans(text, &spans, &fam, &fmt_fonts);
        face_spans.extend(sign_face.as_ref().map(|((b, i, sims), _)| FaceSpan { start: u32::MAX, end: u32::MAX, data: b, index: *i, sims: *sims, coords: &[] }));
        // A bold-sim base face widens inked advances by round(upem/50) design units (native's embolden).
        let bold_delta = if sims & 1 != 0 { (base_font.upm as f32 / 50.0).round() as i32 } else { 0 };
        // DrawTextLayout threads the IDWriteTextLayout per-call inputs (trimming/features/inline) into
        // layout_ex; DrawText passes them empty. Box size stays max_width/max_height (main's origin split).
        let opts = layout::LayoutOpts { fonts: &font_spans, features, trimming, inline, underline, strikethrough, measuring, pixels_per_dip: pixels_per_dip.unwrap_or(if self.get_unit_mode() == 0 { self.get_dpi().0 / 96.0 } else { 1.0 }), base_bold_delta_du: bold_delta, transform: pixels_per_dip.is_none().then(|| self.text_transform()) };
        let lay = layout::layout_ex(&base_font, fmt, text, max_width, max_height, &opts);
        self.draw_layout(&lay, origin, base_bytes, base_index, measuring, brush, params, enable_color, sims, &face_spans, coords)
    }

    /// [`draw_text_fallback`] with a gradient/bitmap/image brush (see [`draw_layout_brush`]).
    #[allow(clippy::too_many_arguments)]
    pub fn draw_text_fallback_brush(
        &mut self,
        text: &str,
        base_bytes: &[u8],
        base_index: u32,
        locale: &str,
        fmt: &layout::TextFormat,
        measuring: u32,
        rect: [f32; 4],
        brush: &crate::PixelBrush,
        params: &RenderingParams,
        enable_color: bool,
        sims: u32,
        coords: &[skrifa::raw::types::F2Dot14],
        shaper_vars: &[(u32, f32)],
        features: &[layout::Feature],
        trimming: Option<layout::Trimming>,
        inline: &[layout::InlineObject],
        formats: &[FormatFace],
        // Per-range underline / strikethrough as BYTE ranges (empty for DrawText and the FFI).
        underline: &[(usize, usize)],
        strikethrough: &[(usize, usize)],
        load_family: impl Fn(&str, usize, char) -> Option<(Vec<u8>, u32, u32)>,
    ) -> bool {
        let (origin, w, h) = normalize_layout_rect(rect);
        self.draw_text_fallback_box_brush(text, base_bytes, base_index, locale, fmt, measuring, origin, w, h, brush, params, enable_color, sims, coords, shaper_vars, features, trimming, inline, formats, underline, strikethrough, None, load_family)
    }

    /// `draw_text_fallback_box` for a gradient/bitmap/image brush. See `draw_text_fallback_box`.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_text_fallback_box_brush(
        &mut self,
        text: &str,
        base_bytes: &[u8],
        base_index: u32,
        locale: &str,
        fmt: &layout::TextFormat,
        measuring: u32,
        origin: [f32; 2],
        max_width: f32,
        max_height: f32,
        brush: &crate::PixelBrush,
        params: &RenderingParams,
        enable_color: bool,
        sims: u32,
        coords: &[skrifa::raw::types::F2Dot14],
        shaper_vars: &[(u32, f32)],
        features: &[layout::Feature],
        trimming: Option<layout::Trimming>,
        inline: &[layout::InlineObject],
        // Per-range character-format FACES: `(byte_start, byte_end, face_bytes, face_index)` runs the
        // caller resolved from SetFontWeight/Style/Stretch/FontFamilyName over a sub-range (empty for
        // DrawText and plain DrawTextLayout). Each takes precedence over fallback and the base family on
        // its bytes, and carries a per-range SIZE as the span's `em` (`FontSpan::em`).
        formats: &[FormatFace],
        // Per-range underline / strikethrough as BYTE ranges (empty for DrawText and the FFI).
        underline: &[(usize, usize)],
        strikethrough: &[(usize, usize)],
        // The layout's own pixelsPerDip (the `CreateGdiCompatibleTextLayout` argument a DrawTextLayout's
        // layout was made with), or `None` for DrawText's internal layout: the context's DPI under its unit
        // mode (PIXELS has the em already in device pixels, ppd = 1).
        pixels_per_dip: Option<f32>,
        load_family: impl Fn(&str, usize, char) -> Option<(Vec<u8>, u32, u32)>,
    ) -> bool {
        // The failed internal CreateTextLayout of a NaN box draws nothing silently, ahead of the em gate.
        if !(max_width >= 0.0) || !(max_height >= 0.0) {
            return true;
        }
        if let Some(refused) = self.draw_text_em_gate(Self::laid_out_em(fmt.em, text.len(), formats), measuring, !text.is_empty()) {
            if refused && pixels_per_dip.is_none() {
                self.process_new_error(crate::state::E_INVALIDARG);
            }
            return true;
        }
        let Some(base_font) = shaping::Font::from_bytes_var(base_bytes.to_vec(), base_index, shaper_vars) else { return false };
        // The per-range format faces (a face whose bytes fail to parse is dropped: the base/fallback face then
        // applies there), then the fallback resolved against them, as `assemble_spans` orders the spans.
        let fmt_fonts: Vec<FmtFont> = formats.iter()
            .filter_map(|&(s, e, b, i, em, fs, axes)| {
                let coords = variations::normalized_from_dwrite(b, i, axes);
                let axes = if coords.is_empty() { &[] } else { axes };
                shaping::Font::from_bytes_var(b.to_vec(), i, axes).map(|sf| (s, e, b, i, em, fs, sf, coords))
            })
            .collect();
        let trimming = upright_sign(trimming, fmt, locale, text.len(), &load_family);
        let sign_face = ellipsis_face(trimming, fmt, locale, text.len(), &load_family);
        let trimming = trimming.map(|t| layout::Trimming { sign: sign_face.as_ref().and_then(|f| f.1), ..t });
        let (fam, spans) = resolve_fallback(text, locale, &base_font, &fmt_fonts, &[underline, strikethrough].concat(), load_family);
        let (font_spans, mut face_spans) = assemble_spans(text, &spans, &fam, &fmt_fonts);
        face_spans.extend(sign_face.as_ref().map(|((b, i, sims), _)| FaceSpan { start: u32::MAX, end: u32::MAX, data: b, index: *i, sims: *sims, coords: &[] }));
        let bold_delta = if sims & 1 != 0 { (base_font.upm as f32 / 50.0).round() as i32 } else { 0 };
        let opts = layout::LayoutOpts { fonts: &font_spans, features, trimming, inline, underline, strikethrough, measuring, pixels_per_dip: pixels_per_dip.unwrap_or(if self.get_unit_mode() == 0 { self.get_dpi().0 / 96.0 } else { 1.0 }), base_bold_delta_du: bold_delta, transform: pixels_per_dip.is_none().then(|| self.text_transform()) };
        let lay = layout::layout_ex(&base_font, fmt, text, max_width, max_height, &opts);
        self.draw_layout_brush(&lay, origin, base_bytes, base_index, measuring, brush, params, enable_color, sims, &face_spans, coords)
    }
}

/// One resolved fallback face: bytes + face index and its `DWRITE_FONT_SIMULATIONS` (rendering), and a shaper
/// [`shaping::Font`] (layout). The owned bytes/font back the borrowing [`layout::FontSpan`] / [`FaceSpan`] a draw
/// builds, so this must outlive the layout + draw.
type FamEntry = (Vec<u8>, u32, u32, shaping::Font);

/// Map the text to fallback spans exactly as DirectWrite's `MapCharacters` does ([`fallback::map_characters`]),
/// for each character its own face (format range, else base) lacks.
/// `load_family(family, byte, ch)` resolves a family to its face matching the format's weight, style and
/// stretch at `byte` best among those that map `ch`, as MapCharacters' font does (r25 30922: Hangul in a
/// weight-700 range takes Malgun Gothic Bold); [`fallback::OWN_FAMILY`] names the family of the format at
/// `byte` in the layout's own collection. The spans returned are those taking a fallback face:
/// `(start, end, index into the faces)`.
fn resolve_fallback(text: &str, locale: &str, base_font: &shaping::Font, fmt_fonts: &[FmtFont], decorated: &[(usize, usize)], load_family: impl Fn(&str, usize, char) -> Option<(Vec<u8>, u32, u32)>) -> (Vec<FamEntry>, Vec<(usize, usize, usize)>) {
    let covers = |sf: &shaping::Font, ch: char| sf.tt().glyph_index(ch).is_some();
    // A character falls back when the font it would otherwise take lacks it: its format range's face, else
    // the base. Native falls back inside a family range too (r25 30146: an emoji in a Lucida Console range).
    let own = |b: usize| fmt_fonts.iter().find(|f| f.0 <= b && b < f.1).map_or(base_font, |f| &f.6);
    // Native maps each stretch of uniform format on its own: every range boundary splits it, an underline or
    // strikethrough range's too (VM r25 30422: a VS16 opening an underline range leaves the emoji run before
    // it for the base face's .notdef).
    let mut bounds: Vec<usize> = fmt_fonts.iter().flat_map(|f| [f.0, f.1]).chain(decorated.iter().flat_map(|&(s, e)| [s, e])).collect();
    bounds.sort_unstable();
    bounds.dedup();
    let range_of = |b: usize| bounds.partition_point(|&x| x <= b);
    // A format family the collection lacks lays its text out in Segoe UI, the stand-in native's system fallback
    // gives Latin, but native maps each character through the fallback list: one a family listed ahead of Segoe UI
    // for it covers goes to that family (VM r25 30178: Arial Black in ja-jp gives '→' and '™' to Yu Gothic UI and
    // its space to Segoe UI). Such a family has no face of its own for a character its stand-in maps; asked once
    // per format range.
    let missing: Vec<std::cell::Cell<Option<bool>>> = vec![std::cell::Cell::new(None); bounds.len() + 1];
    let own_covers = |b: usize, ch: char| {
        if !covers(own(b), ch) {
            return false;
        }
        let m = &missing[range_of(b)];
        let lacks = m.get().unwrap_or_else(|| {
            let v = load_family(fallback::OWN_FAMILY, b, ch).is_none();
            m.set(Some(v));
            v
        });
        !lacks || fallback::candidates(ch as u32, locale).into_iter().take_while(|&f| f != "Segoe UI").all(|f| load_family(f, b, ch).is_none())
    };
    // Each character takes the best face of a family that maps it, as MapCharacters' font does, so one family
    // can give neighbouring characters different faces (VM: in Consolas weight 300 italic, 'ا' takes Segoe UI
    // Light, which has Arabic, and the 'Ⴀ' beside it Segoe UI Light Italic, in either order). Resolved per
    // family, format range and character; ranges and characters resolving to one face share it.
    let fam: std::cell::RefCell<Vec<FamEntry>> = Default::default();
    let faces: std::cell::RefCell<std::collections::HashMap<(String, usize, char), Option<usize>>> = Default::default();
    let face_of = |f: &str, b: usize, ch: char| -> Option<usize> {
        let key = (f.to_string(), range_of(b), ch);
        if let Some(&r) = faces.borrow().get(&key) {
            return r;
        }
        let r = load_family(f, b, ch).and_then(|(bytes, idx, sims)| {
            let mut fam = fam.borrow_mut();
            if let Some(i) = fam.iter().position(|e| e.1 == idx && e.2 == sims && e.0 == bytes) {
                return Some(i);
            }
            let sf = shaping::Font::from_bytes(bytes.clone(), idx)?;
            fam.push((bytes, idx, sims, sf));
            Some(fam.len() - 1)
        });
        faces.borrow_mut().insert(key, r);
        r
    };
    let spans = fallback::map_characters_at(text, locale, own_covers, range_of, |f, b, ch| face_of(f, b, ch).is_some());
    // A family's span splits where its characters' faces differ; a joining character (ZWJ, a variation selector,
    // a skin-tone modifier, an emoji tag, the keycap) stays in the face before it, in the same format range, when
    // that face maps it.
    let mut out: Vec<(usize, usize, usize)> = Vec::new();
    for s in spans {
        let Some(f) = s.family else { continue };
        for (o, ch) in text[s.start..s.end].char_indices() {
            let b = s.start + o;
            let joins = matches!(ch as u32, 0x200D | 0xFE00..=0xFE0F | 0x1F3FB..=0x1F3FF | 0xE0020..=0xE007F | 0x20E3);
            let prev = out.last().filter(|p| p.1 == b && range_of(p.1 - 1) == range_of(b)).map(|p| p.2).filter(|&i| joins && covers(&fam.borrow()[i].3, ch));
            let Some(i) = prev.or_else(|| face_of(f, b, ch)) else { continue };
            let end = b + ch.len_utf8();
            match out.last_mut() {
                Some(p) if p.1 == b && p.2 == i => p.1 = end,
                _ => out.push((b, end, i)),
            }
        }
    }
    (fam.into_inner(), out)
}

/// A vertical layout's trimming with its upright sign resolved ([`layout::upright_sign_per_em`]): the vertical
/// ellipsis's face is the first fallback family for U+22EE that maps it, matched to the base format the sign was
/// made from (`end`: the byte past the text, where no range applies; VM r25 30866: a weight-700 range on the
/// first digits leaves Impact's sign at weight 400, unemboldened).
fn upright_sign(trimming: Option<layout::Trimming>, fmt: &layout::TextFormat, locale: &str, end: usize, load_family: &impl Fn(&str, usize, char) -> Option<(Vec<u8>, u32, u32)>) -> Option<layout::Trimming> {
    let t = trimming?;
    if !t.ellipsis || fmt.flow < 2 {
        return Some(t);
    }
    let face = fallback::candidates(0x22EE, locale).into_iter().find_map(|f| load_family(f, end, '\u{22EE}'));
    let (upright_sign, upright_bold) = face.and_then(|(b, i, sims)| layout::upright_sign_per_em(b, i, sims)).unwrap_or((0.0, 0.0));
    Some(layout::Trimming { upright_sign, upright_bold, ..t })
}

/// The trimming sign's own face when the format family has none mapping the ellipsis (a family the collection
/// lacks): native's sign is a layout of U+2026 in the format, so font fallback gives it the first family of the
/// locale's list that maps it (VM r25 30854: Arial Black in ja-jp signs with Yu Gothic UI Bold's 1666-unit
/// ellipsis, not the Segoe UI that lays out its Latin). A vertical format's sign is the vertical ellipsis U+22EE,
/// always from fallback ([`upright_sign`]). Returns the face with its ellipsis glyph and advance per em, widened by
/// round(upem/50) when bold-simulated.
fn ellipsis_face(trimming: Option<layout::Trimming>, fmt: &layout::TextFormat, locale: &str, end: usize, load_family: &impl Fn(&str, usize, char) -> Option<(Vec<u8>, u32, u32)>) -> Option<((Vec<u8>, u32, u32), Option<(u16, f32)>)> {
    trimming.filter(|t| t.ellipsis)?;
    let ch = if fmt.flow >= 2 { '\u{22EE}' } else { '\u{2026}' };
    if ch == '\u{2026}' && load_family(fallback::OWN_FAMILY, end, ch).is_some() {
        return None;
    }
    let (bytes, index, sims) = fallback::candidates(ch as u32, locale).into_iter().find_map(|f| load_family(f, end, ch))?;
    let sign = shaping::Font::from_bytes(bytes.clone(), index).and_then(|font| {
        let g = *shaping::shape(&font, ch.encode_utf8(&mut [0; 4]), false, None, None).first()?;
        let bold = if sims & 1 != 0 { (font.upm as f32 / 50.0).round() } else { 0.0 };
        Some((g.id, (g.advance as f32 + bold) / font.upm as f32))
    });
    Some(((bytes, index, sims), sign))
}

/// A per-range format face: byte range, bytes, face index, the range's em, its simulations, its shaper font,
/// and its variable-font instance (normalized coords; empty for a static face or the default instance).
type FmtFont<'a> = (usize, usize, &'a [u8], u32, Option<f32>, u32, shaping::Font, Vec<skrifa::raw::types::F2Dot14>);

/// A per-range format face as a caller passes it: byte range, bytes, face index, the range's em (`None` = the
/// base em), its simulations, and its variable-font instance as user axis values (DWRITE-packed tags; empty for
/// a static face).
pub type FormatFace<'a> = (usize, usize, &'a [u8], u32, Option<f32>, u32, &'a [(u32, f32)]);

/// The layout's font spans and the draw's face spans, in precedence order (first covering span wins): each
/// fallback span, cut at the format ranges it crosses and taking their em, then the per-range format faces;
/// the base face covers the rest.
fn assemble_spans<'a>(text: &str, spans: &[(usize, usize, usize)], fam: &'a [FamEntry], fmt_fonts: &'a [FmtFont<'a>]) -> (Vec<layout::FontSpan<'a>>, Vec<FaceSpan<'a>>) {
    let (mut fonts, mut faces) = (Vec::new(), Vec::new());
    for &(start, end, i) in spans {
        let (bytes, idx, sims, sf) = &fam[i];
        let mut cuts = vec![start, end];
        cuts.extend(fmt_fonts.iter().flat_map(|f| [f.0, f.1]).filter(|&c| c > start && c < end));
        cuts.sort_unstable();
        cuts.dedup();
        for w in cuts.windows(2) {
            let em = fmt_fonts.iter().find(|f| f.0 <= w[0] && w[0] < f.1).and_then(|f| f.4);
            fonts.push(layout::FontSpan { start: w[0], end: w[1], font: sf, em, bold: sims & 1 != 0 });
            faces.push(FaceSpan { start: u16_pos(text, w[0]), end: u16_pos(text, w[1]), data: bytes.as_slice(), index: *idx, sims: *sims, coords: &[] });
        }
    }
    for (s, e, b, i, em, fs, sf, coords) in fmt_fonts {
        fonts.push(layout::FontSpan { start: *s, end: *e, font: sf, em: *em, bold: fs & 1 != 0 });
        faces.push(FaceSpan { start: u16_pos(text, *s), end: u16_pos(text, *e), data: b, index: *i, sims: *fs, coords });
    }
    (fonts, faces)
}

/// UTF-16 code-unit offset of a byte offset in `text` (DirectWrite ranges are UTF-16).
fn u16_pos(text: &str, byte: usize) -> u32 {
    text[..byte].encode_utf16().count() as u32
}

#[cfg(test)]
mod vertical_origin_tests {
    use super::gdi_natural_vertical_origin as vo;

    /// Native GetGdiCompatibleGlyphMetrics (sideways, GDI natural) of Yu Gothic UI Bold, vertical origin 1802.
    #[test]
    fn gdi_natural_vertical_origin_matches_native() {
        assert_eq!(vo(1802, 1448, 70.3, 2048, false), 1814);
        assert_eq!(vo(1802, 1448, 103.9, 2048, false), 1812);
        assert_eq!(vo(1802, 1448, 70.3, 2048, true), 1568);
        assert_eq!(vo(1802, 1575, 70.3, 2048, true), 1547);
        assert_eq!(vo(1802, 1448, 81.7, 2048, true), 1552);
        assert_eq!(vo(1802, 2048, 117.2, 2048, true), 1455);
    }
}

#[cfg(test)]
mod merge_rect_tests {
    use super::merge_rects;

    #[test]
    fn merge_rects_sweep() {
        let r = |l: f32, r: f32| [l, 0.0, r, 10.0];
        // Two glyphs that meet merge; a disjoint third stays single.
        assert_eq!(merge_rects(&[r(0.0, 10.0), r(10.0, 20.0), r(30.0, 40.0)], &[false; 3], &[0..3]), vec![0..2]);
        // With glyph 0 cut away, the sweep's previous glyph is still index 0, so glyphs 1 and 2 meeting each other
        // do not merge (r25 30258).
        assert_eq!(merge_rects(&[[0.0; 4], r(0.0, 50.0), r(42.0, 90.0)], &[true, false, false], &[0..3]), vec![]);
        // A glyph reaching back past its predecessor folds everything from the first one it meets.
        assert_eq!(merge_rects(&[r(0.0, 30.0), r(40.0, 50.0), r(5.0, 45.0)], &[false; 3], &[0..3]), vec![0..3]);
    }
}

#[cfg(test)]
mod ctbits_harness {
    use super::*;
    // Isolated single-glyph raw 6× CT grid harness (ZTMP-gated): renders one glyph through the real
    // draw path (`hinted_contours` → `coverage_outline(ClearType)`) so `resolve_cleartype`'s D2D_CTBITS
    // dump prints our pre-resolve 6×5 scan grid for one clean glyph, to diff against native PRERESOLVE.
    #[test]
    fn ztmp_ct_glyph() {
        if std::env::var("ZTMP").is_err() { return; }
        let spec = std::env::var("ZTMP").unwrap(); // "file:char:em:measuring" e.g. arial.ttf:H:13:1
        let p: Vec<&str> = spec.split(':').collect();
        let Some(bytes) = d2d_reference::fonts_dir().and_then(|d| std::fs::read(d.join(p[0])).ok()) else {
            eprintln!("skip: {} not under D2D_FONTS / ref/windows/Fonts", p[0]);
            return;
        };
        let font = FontRef::from_index(&bytes, 0).unwrap();
        let ch: char = p[1].chars().next().unwrap();
        let em: f32 = p[2].parse().unwrap();
        let measuring: u32 = p[3].parse().unwrap();
        let gid = font.charmap().map(ch).unwrap().to_u32() as u16;
        let glyphs = [Glyph { id: gid, advance: em, offset: [0.0, 0.0] }];
        // ZTMP_BL="x,y" overrides the baseline origin so a specific sub-pixel phase can be tested.
        let bl: [f32; 2] = std::env::var("ZTMP_BL").ok()
            .map(|s| { let v: Vec<f32> = s.split(',').map(|x| x.parse().unwrap()).collect(); [v[0], v[1]] })
            .unwrap_or([0.0, 0.0]);
        let run = GlyphRun {
            font_data: &bytes, face_index: 0, em_size: em, glyphs: &glyphs,
            baseline_origin: bl, measuring_mode: measuring, is_sideways: false,
            bidi_level: 0, no_snap: false, simulations: 0,
            coords: &[],
        };
        // ZTMP_DT="m11,m12,m21,m22" hints under a transform (quarter turns, shears).
        let dt = std::env::var("ZTMP_DT").ok()
            .map(|s| { let v: Vec<f64> = s.split(',').map(|x| x.parse().unwrap()).collect(); [v[0], v[1], v[2], v[3], 0.0, 0.0] })
            .unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        // ZTMP_RMODE overrides the rendering mode (4 = NATURAL/8×1, 5 = NATURAL_SYMMETRIC/4×4) so a
        // no-gasp or <21px glyph can be traced at the mode native actually renders (recommended_rendering_mode);
        // ZTMP_AA overrides the AA class (default ClearType) so grayscale movers can be captured too.
        let mode = std::env::var("ZTMP_RMODE").ok().and_then(|s| s.parse().ok())
            .unwrap_or(match measuring { 1 => MODE_GDI_CLASSIC, 2 => MODE_GDI_NATURAL, _ => MODE_NATURAL_SYMMETRIC });
        let mov_aa = std::env::var("ZTMP_AA").ok().and_then(|s| s.parse().ok()).unwrap_or(AA_CLEARTYPE);
        if std::env::var_os("ZTMP_STG").is_some() {
            let device_em = em as f64; // harness dt is identity (96 DPI)
            let cfg = hinting::Config { rasterizer: hinting::Rasterizer::FontCache, mode: hinting::Mode::GdiClassic, symmetric: false, bgr: false, is_rotated: false, is_stretched: false, axes_swapped: false, rotation_kind: 0, em_y: None, version: 42, gdi_render: true, cleartype: true, subpixel_advance: false, grid_fit: true, bold_fu: 0 };
            let inst = hinting::Instance::new(&font, device_em, cfg).unwrap();
            let s = inst.dbg_storage();
            let at = |i: usize| s.get(i).copied().unwrap_or(-9);
            eprintln!("STG storage[2]={} [8]={} [18]={} [19]={} [55..65]={:?}", at(2), at(8), at(18), at(19), (55..65).map(at).collect::<Vec<_>>());
            eprintln!("  native CT: [2]=0 [8]=0 [18]=1 [19]=1 [55..65]=[0,0,0,0,0,0,0,0,0,94]");
            return;
        }
        if let Ok(np) = std::env::var("ZTMP_MOV") {
            hinting::enable_trace();
            let _ = hinted_contours(&dt, &run, &font, mode, mov_aa).unwrap();
            let ours = hinting::take_trace();
            if let Ok(pp) = std::env::var("ZTMP_PT") {
                let p: usize = pp.parse().unwrap();
                eprintln!("PTTRACE pt{p} across {} records:", ours.len());
                for (i, r) in ours.iter().enumerate() {
                    let op_name = |op: u8| -> &'static str { match op { 0x2E|0x2F=>"MDAP",0x30|0x31=>"IUP",0x32|0x33=>"SHP",0x34|0x35=>"SHC",0x38=>"SHPIX",0x39=>"IP",0x3A|0x3B=>"MSIRP",0x3C=>"ALIGNRP",0x3E|0x3F=>"MIAP",0xC0..=0xDF=>"MDRP",0xE0..=0xFF=>"MIRP",_=>"?" } };
                    if p < r.x.len() { eprintln!("  #{i} {:<7} npts={} ip={} pt{p}=({},{})", op_name(r.op), r.x.len(), r.ip, r.x[p], r.y[p]); }
                }
            }
            let nat: serde_json::Value = serde_json::from_slice(&std::fs::read(np).unwrap()).unwrap();
            let ncont = ours.first().map(|r| r.x.len()).unwrap_or(0).saturating_sub(4);
            // Native's point-mover dump logs after RS (and other non-point-moving handlers) too, which
            // our `is_mover` trace excludes — so filter native to the point-moving ops before aligning.
            let is_pm = |op: &str| matches!(op, "MIAP"|"IP"|"MIRP"|"MDRP"|"SHPIX"|"MSIRP"|"ALIGNRP"|"MDAP"|"IUP"|"SHP"|"SHC");
            // GDI-classic runs the glyph program TWICE (metric pass g448=0, then render pass g448=0x3);
            // ZTMP_G448 filters native to the render pass so it aligns with our single render trace.
            let g448f: Option<i64> = std::env::var("ZTMP_G448").ok().and_then(|s| s.parse().ok());
            let natg: Vec<&serde_json::Value> = nat.as_array().unwrap().iter()
                .filter(|r| r["x"].as_array().unwrap().len() == ncont && is_pm(r["op"].as_str().unwrap_or(""))
                    && g448f.is_none_or(|g| r["g"][1].as_i64() == Some(g))).collect();
            let op_name = |op: u8| -> &'static str { match op { 0x2E|0x2F=>"MDAP",0x30|0x31=>"IUP",0x32|0x33=>"SHP",0x34|0x35=>"SHC",0x38=>"SHPIX",0x39=>"IP",0x3A|0x3B=>"MSIRP",0x3C=>"ALIGNRP",0x3E|0x3F=>"MIAP",0xC0..=0xDF=>"MDRP",0xE0..=0xFF=>"MIRP",_=>"?" } };
            eprintln!("MOV ncont={ncont} ours={} native_pm={}", ours.len(), natg.len());
            eprintln!("  ours ops:   {:?}", ours.iter().map(|o| op_name(o.op)).collect::<Vec<_>>());
            eprintln!("  native ops: {:?}", natg.iter().map(|n| n["op"].as_str().unwrap_or("?")).collect::<Vec<_>>());
            let col = |n: &serde_json::Value, k: &str| -> Vec<i32> { n[k].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect() };
            for (i, (o, n)) in ours.iter().zip(natg.iter()).enumerate() {
                let (nx, ny) = (col(n, "x"), col(n, "y"));
                let nop = n["op"].as_str().unwrap_or("?");
                if op_name(o.op) != nop || o.x[..ncont] != nx[..] || o.y[..ncont] != ny[..] {
                    let xd: Vec<String> = o.x[..ncont].iter().zip(&nx).enumerate().filter(|(_,(a,b))| a!=b).map(|(k,(a,b))| format!("x{k}:{}/{}",a,b)).collect();
                    let yd: Vec<String> = o.y[..ncont].iter().zip(&ny).enumerate().filter(|(_,(a,b))| a!=b).map(|(k,(a,b))| format!("y{k}:{}/{}",a,b)).collect();
                    eprintln!("MOV DIVERGE #{i} ourop={} natop={nop} ip={} rp{:?} xdiff: {} ydiff: {}", op_name(o.op), n.get("ip").map(|v|v.to_string()).unwrap_or_default(), n["rp"], xd.join(" "), yd.join(" "));
                    return;
                }
            }
            eprintln!("MOV exact over {} movers", ours.len().min(natg.len()));
            // Also compare the FINAL composite contour (all points, not just the ncont component) to
            // native's last composite record — catches accent/assembly divergences the component scan misses.
            let ncomp = ours.last().map(|r| r.x.len()).unwrap_or(0).saturating_sub(4);
            if let Some(oc) = ours.last() {
                if let Some(nc) = nat.as_array().unwrap().iter().rev().find(|r| r["x"].as_array().unwrap().len() == ncomp) {
                    let (nx, ny) = (col(nc, "x"), col(nc, "y"));
                    let d: Vec<String> = (0..ncomp).filter(|&k| oc.x[k] != nx[k] || oc.y[k] != ny[k])
                        .map(|k| format!("p{k}:({},{})/({},{})", oc.x[k], oc.y[k], nx[k], ny[k])).collect();
                    eprintln!("MOVCOMP ncomp={ncomp} {}", if d.is_empty() { "EXACT".into() } else { format!("DIVERGE {}", d.join(" ")) });
                }
            }
            return;
        }
        let (c, drop) = hinted_contours(&dt, &run, &font, mode, AA_CLEARTYPE).unwrap();
        let ck = if horizontal_only(mode) { raster::Kind::ClearTypeH } else { raster::Kind::ClearType };
        let tex = raster::coverage_outline(&c, drop, ck, (0, 0));
        eprintln!("ZTMP gid={gid} ours bounds=[{},{},{},{}]", tex.left, tex.top, tex.right, tex.bottom);
        let (w, h) = (tex.width() as usize, tex.height() as usize);
        if let Ok(np) = std::env::var("ZTMP_NAT") {
            // native CT alpha: 3 bytes/px, row-major from top; bounds in ZTMP_NB "l,t,r,b".
            let nb: Vec<i32> = std::env::var("ZTMP_NB").unwrap().split(',').map(|s| s.parse().unwrap()).collect();
            let ndata = std::fs::read(np).unwrap();
            let (nl, nt, nr, _nbm) = (nb[0], nb[1], nb[2], nb[3]);
            let nw = (nr - nl) as usize;
            eprintln!("native bounds={nb:?} ({} px)", ndata.len() / 3);
            for y in 0..h as i32 {
                for x in 0..w as i32 {
                    let (dx, dy) = (tex.left + x, tex.top + y);
                    let oi = (y as usize * w + x as usize) * 3;
                    let o = [tex.data[oi], tex.data[oi + 1], tex.data[oi + 2]];
                    let (nx, ny) = (dx - nl, dy - nt);
                    let n = if nx >= 0 && ny >= 0 && (nx as usize) < nw {
                        let ni = (ny as usize * nw + nx as usize) * 3;
                        if ni + 2 < ndata.len() { [ndata[ni], ndata[ni + 1], ndata[ni + 2]] } else { [0, 0, 0] }
                    } else { [0, 0, 0] };
                    if o != n {
                        eprintln!("DIFF ({dx},{dy}) ours[{},{},{}] native[{},{},{}]", o[0], o[1], o[2], n[0], n[1], n[2]);
                    }
                }
            }
            // Full mid-stem row (ours vs native), all columns, to read the per-column transform.
            let midy = (nt + _nbm) / 2;
            for (tag, dat, bl, bt, bw2) in [("OURS", &tex.data, tex.left, tex.top, w), ("NATV", &ndata, nl, nt, nw)] {
                let ry = (midy - bt) as usize;
                let row: Vec<String> = (0..bw2).map(|cx| { let i = (ry * bw2 + cx) * 3; format!("[{},{},{}]", dat[i], dat[i+1], dat[i+2]) }).collect();
                eprintln!("ROW {tag} y={midy} {}", row.join(" "));
            }
        }
    }

    /// Grayscale single-glyph harness (ZTMP_GRAY="file:char:em:rmode", rmode 4=NATURAL/8×1,
    /// 5=NATURAL_SYMMETRIC/4×4). Dumps our hinted points (for the native scan-converter contour-list compare) and,
    /// under ZTMP_NAT (native .gray.bin, 1 byte/px, top-down) + ZTMP_NB "l,t,r,b", diffs our
    /// grayscale coverage cell-by-cell — the raster half of the diagonal-MDRP landing.
    #[test]
    fn ztmp_gray_glyph() {
        let Ok(spec) = std::env::var("ZTMP_GRAY") else { return };
        let p: Vec<&str> = spec.split(':').collect();
        let Some(bytes) = d2d_reference::fonts_dir().and_then(|d| std::fs::read(d.join(p[0])).ok()) else {
            eprintln!("skip: {} not under D2D_FONTS / ref/windows/Fonts", p[0]);
            return;
        };
        let font = FontRef::from_index(&bytes, 0).unwrap();
        let ch: char = p[1].chars().next().unwrap();
        let em: f32 = p[2].parse().unwrap();
        let rmode: u32 = p[3].parse().unwrap();
        let gid = font.charmap().map(ch).unwrap().to_u32() as u16;
        let glyphs = [Glyph { id: gid, advance: em, offset: [0.0, 0.0] }];
        let bl: [f32; 2] = std::env::var("ZTMP_BL").ok()
            .map(|s| { let v: Vec<f32> = s.split(',').map(|x| x.parse().unwrap()).collect(); [v[0], v[1]] })
            .unwrap_or([0.0, 0.0]);
        let run = GlyphRun {
            font_data: &bytes, face_index: 0, em_size: em, glyphs: &glyphs,
            baseline_origin: bl, measuring_mode: 0, is_sideways: false,
            bidi_level: 0, no_snap: false, simulations: 0,
            coords: &[],
        };
        let dt = std::env::var("ZTMP_DT").ok()
            .map(|s| { let v: Vec<f64> = s.split(',').map(|x| x.parse().unwrap()).collect(); [v[0], v[1], v[2], v[3], 0.0, 0.0] })
            .unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let (c, drop) = hinted_contours(&dt, &run, &font, rmode, AA_GRAYSCALE).unwrap();
        // Our hinted points (device px ×64, y-up like native's scan-converter contour list), for the contour compare.
        if std::env::var_os("ZTMP_PTS").is_some() {
            for (pts, _) in &c {
                let xs: Vec<i64> = pts.iter().map(|p| (p[0] * 64.0).round() as i64).collect();
                let ys: Vec<i64> = pts.iter().map(|p| (-p[1] * 64.0).round() as i64).collect();
                eprintln!("PTS x={xs:?}");
                eprintln!("PTS y={ys:?}");
            }
        }
        let tex = if horizontal_only(rmode) { raster::coverage_box(&c, drop, 8, 1, (0, 0), false) } else { raster::coverage_outline(&c, drop, raster::Kind::Grayscale, (0, 0)) };
        eprintln!("ZGRAY gid={gid} ours bounds=[{},{},{},{}] grid={}", tex.left, tex.top, tex.right, tex.bottom, if horizontal_only(rmode) { "8x1" } else { "4x4" });
        let (w, h) = (tex.width() as usize, tex.height() as usize);
        if let Ok(np) = std::env::var("ZTMP_NAT") {
            let nb: Vec<i32> = std::env::var("ZTMP_NB").unwrap().split(',').map(|s| s.parse().unwrap()).collect();
            let nd = std::fs::read(np).unwrap();
            let (nl, nt, nr) = (nb[0], nb[1], nb[2]);
            let nw = (nr - nl) as usize;
            eprintln!("native bounds={nb:?} ({} px)", nd.len());
            for y in 0..h as i32 {
                for x in 0..w as i32 {
                    let (dx, dy) = (tex.left + x, tex.top + y);
                    let o = tex.data[y as usize * w + x as usize];
                    let (nx, ny) = (dx - nl, dy - nt);
                    let n = if nx >= 0 && ny >= 0 && (nx as usize) < nw { nd.get(ny as usize * nw + nx as usize).copied().unwrap_or(0) } else { 0 };
                    if o != n { eprintln!("DIFF ({dx},{dy}) ours={o} native={n}"); }
                }
            }
        }
    }
}

#[cfg(test)]
mod huge_transform_no_hang {
    use super::*;

    // Regression for fuzz 7978: a glyph run under an extreme world transform must not hang. 7978 draws
    // text under a 1e6 world scale, which puts the device glyph box at ~1e8 — wholly off the target.
    // Native clips the run to the target (d2d1 `GlyphRunRenderer` sub-rects) and draws nothing; our
    // outline fill flattened/tessellated the full device span → OOM/hang. The device-box guard in
    // `fill_outline` skips a wholly-off-target glyph. WITHOUT the guard this test never returns (the
    // fill spins over a ~1e8-pixel span), so a `cargo test` timeout catches the regression.
    #[test]
    fn offscreen_huge_scale_glyph_draws_nothing_without_hanging() {
        let bytes = include_bytes!("color/testdata/colrtest.ttf").to_vec();
        let font = FontRef::from_index(&bytes, 0).unwrap();
        let id = [1.0f64, 0.0, 0.0, 1.0, 0.0, 0.0];
        let fg = TextFg::Solid([0.0, 0.0, 0.0, 1.0]);

        // Find an outline glyph that actually draws at a normal scale, so the huge case below genuinely
        // exercises the fill (not a no-op on an empty glyph).
        let mut gid = 0u16;
        for g in 1u16..11 {
            let glyphs = [Glyph { id: g, advance: 1000.0, offset: [0.0, 0.0] }];
            let run = GlyphRun {
                font_data: &bytes, face_index: 0, em_size: 48.0, glyphs: &glyphs,
                baseline_origin: [8.0, 48.0], measuring_mode: 0, is_sideways: false,
                bidi_level: 0, no_snap: false, simulations: 0, coords: &[],
            };
            let mut t = crate::context::bitmap_target(64, 64, 96.0);
            fill_outline(&mut t, &id, &run, &font, &fg, crate::pixels::Blend::SourceOver, true, None);
            if t.data.iter().any(|&b| b != 0) {
                gid = g;
                break;
            }
        }
        assert!(gid != 0, "colrtest.ttf must have a drawable outline glyph");

        // The hang case: a 1e6 world scale. Baseline (10,30) DIP → device (1e7, 3e7), so the whole glyph
        // box is off the 64×64 target. Must return (no hang) and draw nothing — matching native's clip.
        let glyphs = [Glyph { id: gid, advance: 1000.0, offset: [0.0, 0.0] }];
        let run = GlyphRun {
            font_data: &bytes, face_index: 0, em_size: 48.0, glyphs: &glyphs,
            baseline_origin: [10.0, 30.0], measuring_mode: 0, is_sideways: false,
            bidi_level: 0, no_snap: false, simulations: 0, coords: &[],
        };
        let huge = [1_000_000.0f64, 0.0, 0.0, 1_000_000.0, 0.0, 0.0];
        let mut target = crate::context::bitmap_target(64, 64, 96.0);
        fill_outline(&mut target, &huge, &run, &font, &fg, crate::pixels::Blend::SourceOver, true, None);
        assert!(target.data.iter().all(|&b| b == 0), "a wholly-off-target glyph draws nothing");
    }

    // The covering case of fuzz 7978: a complex glyph at a ~2e6 device scale, positioned so its outline
    // COVERS the 256×32 target (the wholly-off skip does not fire). Without the oversized→geometry-fill
    // route, the feather realization's AA tessellator sweep degenerates on the ~1e8-magnitude contour
    // and spins forever; with it, the geometry fill strip-clips + buffer-clamps and returns. Gated on
    // D2D_FONTS (ref Consolas) so it runs in conformance but not in a fontless unit run; under a cargo
    // test timeout, a feather regression here hangs the suite. Uses the real 7978 culprit glyph id 137.
    #[test]
    fn covering_huge_glyph_no_hang() {
        let Ok(dir) = std::env::var("D2D_FONTS") else { return };
        let Ok(bytes) = std::fs::read(format!("{dir}/consola.ttf")) else { return };
        let font = FontRef::from_index(&bytes, 0).unwrap();
        let glyphs = [Glyph { id: 137, advance: 24.5, offset: [0.0, 0.0] }];
        let run = GlyphRun {
            font_data: &bytes, face_index: 0, em_size: 40.91920471191406, glyphs: &glyphs,
            baseline_origin: [46.06309127807617, 11.913410186767578], measuring_mode: 0,
            is_sideways: false, bidi_level: 0, no_snap: false, simulations: 0, coords: &[],
        };
        // Device transform that brings glyph 137's ~1e8 device box onto the 256×32 target (world
        // [1e6,0,0,1e6,-57.6e6,-91.5e6] × dpi 192).
        let dt = [2_000_000.0f64, 0.0, 0.0, 2_000_000.0, -115_200_000.0, -183_000_000.0];
        let mut target = crate::context::bitmap_target(256, 32, 192.0);
        fill_outline(&mut target, &dt, &run, &font, &TextFg::Solid([0.0, 0.0, 0.0, 1.0]), crate::pixels::Blend::SourceOver, false, None);
        // Reaching this line is the whole point: the fill returned instead of spinning. (Glyph 137's
        // device box overlaps the target — Wine-confirmed to hang here without the fix — so this
        // exercises the covering fill path, not the wholly-off skip.) The buffer stays intact.
        assert_eq!(target.data.len(), 256 * 32 * 4);
    }

    // The device-box guard itself: a glyph at normal DIP coords under a 1e6 scale is wholly off a small
    // target (⇒ skip), identity keeps it on (⇒ draw), and an empty/degenerate box never skips.
    #[test]
    fn glyph_outside_target_guard() {
        // A box at DIP (50,50)-(60,60): under 1e6 it maps to device (5e7..6e7), wholly off a 64×64
        // target; under identity it stays within 64×64 (so it must NOT skip).
        let figs = vec![Figure {
            start: [50.0, 50.0], filled: true, closed: true,
            segments: vec![Segment::Line([60.0, 50.0]), Segment::Line([60.0, 60.0]), Segment::Line([50.0, 60.0])],
            seg_flags: Vec::new(),
        }];
        let id = [1.0f64, 0.0, 0.0, 1.0, 0.0, 0.0];
        let huge = [1_000_000.0f64, 0.0, 0.0, 1_000_000.0, 0.0, 0.0];
        assert!(glyph_outside_target(&figs, &huge, 64.0, 64.0), "1e6 scale ⇒ wholly off a 64×64 target");
        assert!(!glyph_outside_target(&figs, &id, 64.0, 64.0), "identity ⇒ on the target");
        assert!(!glyph_outside_target(&[], &huge, 64.0, 64.0), "empty box never skips");
    }
}

#[cfg(test)]
mod em_reject_boundary {
    use crate::context::DeviceContext;
    use crate::resource::Domain;

    fn dc() -> DeviceContext {
        let mut c = DeviceContext::new(Domain { factory_id: 0, device_id: 0 });
        c.set_transform([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]); // identity world (dpi defaults to 96 ⇒ device == world)
        c
    }

    // DrawText / DrawGlyphRun: a DEVICE em reaching 2^15 under a GDI measuring mode rejects; the edge is
    // exactly 32768.0 (the f32 just below passes). NATURAL never rejects; empty text never rejects even at
    // a huge em (16015 op9). VM-pinned 2026-11-08 (16015/16913/16524).
    #[test]
    fn drawtext_device_em_edge_is_exactly_32768() {
        let c = dc();
        let below = f32::from_bits(32768f32.to_bits() - 1); // 32767.998046875
        for meas in [1u32, 2] {
            assert_eq!(c.draw_text_em_gate(below, meas, true), None, "device em {below} meas {meas} passes");
            assert_eq!(c.draw_text_em_gate(32768.0, meas, true), Some(true), "device em 32768 meas {meas} rejects");
            assert_eq!(c.draw_text_em_gate(32768.0, meas, false), None, "empty never rejects");
        }
        assert_eq!(c.draw_text_em_gate(1e9, 0, true), None, "NATURAL never rejects");
        assert_eq!(c.draw_text_em_gate(1e9, 1, false), None, "empty under GDI never rejects (16015 op9)");
        assert_eq!(c.draw_text_em_gate(f32::INFINITY, 1, true), Some(false), "a non-finite device em succeeds, drawing nothing (17208)");
        let mut c = dc();
        c.set_transform([3.5, 4.6, -1.6, 1.25, 0.0, 0.0]);
        assert_eq!(c.draw_text_em_gate(8000.0, 2, true), Some(false), "a sheared em passes (28092 op16)");
        c.set_transform([1e6, 0.0, 0.0, 1e6, 0.0, 0.0]);
        assert_eq!(c.draw_text_em_gate(18.0, 2, true), Some(true), "an axis-aligned 1e6 scale refuses (28811)");
        c.set_transform([1e6, 0.0, 0.0, 2000.0, 0.0, 0.0]);
        assert_eq!(c.draw_text_em_gate(20.0, 2, true), Some(false), "a non-uniform em passes (VM: 1e6×2000)");
    }

    // A NaN transform cell fails the GDI metric path for a DrawText even with no glyphs (16913 dx=nan,
    // 16524 dy=nan); an INF cell does NOT (7671 dy=-inf, empty → S_OK). NATURAL is unaffected.
    #[test]
    fn gdi_nan_transform_rejects_but_inf_does_not() {
        let mut c = dc();
        c.set_transform([1.0, 0.0, 0.0, 1.0, f32::NAN, 0.0]);
        assert_eq!(c.draw_text_em_gate(16.0, 1, false), Some(true), "NaN translation rejects a GDI DrawText even empty");
        assert_eq!(c.draw_text_em_gate(16.0, 2, true), Some(true));
        assert_eq!(c.draw_text_em_gate(16.0, 0, true), None, "NATURAL unaffected by a NaN transform");
        // A singular transform with an inf translation (7671): empty text ⇒ S_OK, not rejected.
        c.set_transform([0.0, 0.0, 0.0, 0.0, 23.36, f32::NEG_INFINITY]);
        assert_eq!(c.draw_text_em_gate(36.0, 1, false), None, "inf translation + empty text ⇒ S_OK");
        assert_eq!(c.draw_text_em_gate(36.0, 2, false), None, "inf translation + empty text ⇒ S_OK (gdi-natural)");
    }

    // CreateGdiCompatibleTextLayout: a device em (em · ppd · transformScale) reaching 2^15 rejects at
    // creation; the edge is exactly 32768.0 and scales with ppd (dpi 192 ⇒ ppd 2 ⇒ edge em 16384) and with
    // the transform. VM-pinned 2026-11-08 (16180).
    #[test]
    fn gdi_compat_layout_em_edge() {
        let below = f32::from_bits(32768f32.to_bits() - 1);
        assert!(!DeviceContext::gdi_compat_layout_em_rejected(below, 1.0, 1.0), "em {below} ppd1 passes");
        assert!(DeviceContext::gdi_compat_layout_em_rejected(32768.0, 1.0, 1.0), "em 32768 ppd1 rejects");
        assert!(!DeviceContext::gdi_compat_layout_em_rejected(16383.0, 2.0, 1.0), "em 16383 ppd2 passes");
        assert!(DeviceContext::gdi_compat_layout_em_rejected(16384.0, 2.0, 1.0), "em 16384 ppd2 rejects (device 32768)");
        assert!(DeviceContext::gdi_compat_layout_em_rejected(16384.0, 1.0, 2.0), "transform scale folds in");
        // A FINITE huge device em still rejects; one that OVERFLOWS to non-finite does NOT (r12 7713).
        assert!(DeviceContext::gdi_compat_layout_em_rejected(f32::MAX, 1.0, 1.0), "finite huge device em rejects");
        assert!(!DeviceContext::gdi_compat_layout_em_rejected(f32::MAX, 2.0, 1.0), "device em overflow ⇒ S_OK");
        assert!(!DeviceContext::gdi_compat_layout_em_rejected(f32::INFINITY, 1.0, 1.0), "inf em ⇒ S_OK");
    }

    // DrawText's transient-layout reject (native `CreateTextLayout` gate maxWidth>=0 && maxHeight>=0):
    // a NaN box rejects; +inf and an inverted (positive-normalized) finite rect pass.
    #[test]
    fn drawtext_rect_nan_rejects_inf_and_inverted_pass() {
        use crate::text::draw_text_rect_rejected;
        assert!(draw_text_rect_rejected([f32::NAN, 0.0, 10.0, 10.0]), "NaN left ⇒ NaN width ⇒ reject (17620 op8)");
        assert!(draw_text_rect_rejected([0.0, 0.0, 10.0, f32::NAN]), "NaN bottom ⇒ reject");
        assert!(!draw_text_rect_rejected([10.0, 0.0, 5.0, 10.0]), "inverted-x finite rect ⇒ positive width ⇒ accept");
        assert!(!draw_text_rect_rejected([f32::NEG_INFINITY, 0.0, 3.4e38, 10.0]), "−inf..3.4e38 ⇒ inf width ⇒ accept");
        assert!(!draw_text_rect_rejected([0.0, 0.0, 10.0, 10.0]), "normal rect ⇒ accept");
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod fallback_faces {
    use super::{fallback, resolve_fallback, shaping};

    /// The VM fonts (`D2D_FONTS`, else the main checkout's gitignored ref/windows/Fonts); `None` skips.
    fn fonts_dir() -> Option<std::path::PathBuf> {
        if let Some(p) = std::env::var_os("D2D_FONTS") {
            return Some(p.into());
        }
        let out = std::process::Command::new("git").args(["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "--path-format=absolute", "--git-common-dir"]).output().ok()?;
        let dir = std::path::PathBuf::from(String::from_utf8(out.stdout).ok()?.trim()).parent()?.join("ref/windows/Fonts");
        dir.join("consola.ttf").is_file().then_some(dir)
    }

    /// VM probe (Consolas weight 300 italic, en-US, em 40, cluster widths): in either order, 'ا' takes Segoe UI
    /// Light (7.5195313; the Light Italic has no Arabic) and 'Ⴀ' Segoe UI Light Italic (29.492188), each the
    /// family's best face that maps it.
    #[test]
    fn each_character_takes_the_best_face_that_maps_it() {
        let Some(dir) = fonts_dir() else { return };
        let base = shaping::Font::from_bytes(std::fs::read(dir.join("consola.ttf")).unwrap(), 0).unwrap();
        let load = |f: &str, _: usize, ch: char| {
            let family = if f == fallback::OWN_FAMILY { "Consolas" } else { f };
            let (path, face, sims) = dwrite::resolve_face_sims_covering(&dir, family, 300, 2, 5, ch)?;
            Some((std::fs::read(path).ok()?, face, sims as u32))
        };
        for text in ["\u{0627}\u{10A0}", "\u{10A0}\u{0627}"] {
            let (fam, spans) = resolve_fallback(text, "en-US", &base, &[], &[], load);
            let width = |ch: char| {
                let b = text.find(ch).unwrap();
                let &(_, _, i) = spans.iter().find(|s| s.0 <= b && b < s.1).unwrap();
                let tt = fam[i].3.tt();
                tt.glyph_index(ch).and_then(|g| tt.glyph_hor_advance(g)).unwrap() as f32 * 40.0 / tt.units_per_em() as f32
            };
            assert_eq!((width('\u{0627}'), width('\u{10A0}')), (7.519_531_3, 29.492_188), "{text:?}");
        }
    }
}

#[cfg(test)]
mod rtl_origins {
    use super::*;

    /// VM probe (IDWriteFactory4::ComputeGlyphOrigins, Segoe UI Emoji at em 66.0843, a right-to-left run of advances
    /// 88.5, 86.5, 0, 21.6 and offsets 0, −2, 3, 1.5 from x 0): each glyph's own advance ends at the pen and its offset
    /// points left; a GDI mode measures the own advance in whole pixels of the transform's scale, and bold widens
    /// every advancing glyph but not the zero-advance one.
    #[test]
    fn right_to_left_glyphs_end_their_own_advance_at_the_pen() {
        let Some(bytes) = d2d_reference::fonts_dir().and_then(|d| std::fs::read(d.join("seguiemj.ttf")).ok()) else { return };
        let glyphs: Vec<Glyph> = [(2337, 88.5, 0.0), (1218, 86.5, -2.0), (1273, 0.0, 3.0), (1285, 21.6, 1.5)]
            .map(|(id, advance, o)| Glyph { id, advance, offset: [o, 0.0] })
            .into();
        let run = |measuring_mode, simulations| GlyphRun {
            font_data: &bytes, face_index: 0, em_size: 66.0843276977539, glyphs: &glyphs, baseline_origin: [0.0, 0.0],
            measuring_mode, is_sideways: false, bidi_level: 1, no_snap: true, simulations, coords: &[],
        };
        let origins = |r: &GlyphRun, xf: [f64; 6]| {
            let mut pen = r.baseline_origin[0];
            rtl_glyphs(r, &xf).iter().map(|g| (pen + g.offset[0], pen += g.advance).0).collect::<Vec<f32>>()
        };
        let x2 = [2.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let flip = [-2.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let one = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        for (m, sims, xf, want) in [
            (0, 0, x2, [-90.73688, -177.23688, -178.0, -203.63716]),
            (2, 0, x2, [-90.5, -177.0, -178.0, -203.5]),
            (2, 0, flip, [-90.5, -177.0, -178.0, -203.5]),
            (2, 0, one, [-91.0, -177.5, -178.0, -203.5]),
            (0, 1, x2, [-92.05986, -178.55986, -178.0, -204.96014]),
            (2, 1, x2, [-91.0, -177.5, -178.0, -204.0]),
        ] {
            let got = origins(&run(m, sims), xf);
            assert!(got.iter().zip(want).all(|(g, w)| (g - w).abs() < 2e-4), "mode {m} sims {sims} {xf:?}: {got:?}");
        }
    }
}

#[cfg(test)]
mod fractional_em_tests {
    use super::*;

    /// VM (CreateGlyphRunAnalysis, '6' at em 19.44): Segoe UI Emoji, whose head flags do not force integer ppems,
    /// hints at the fractional em under a vertical shear in GDI classic, GDI natural and natural symmetric, and at
    /// 19 untransformed in GDI classic; Arial, which forces integer ppems, hints at 19 under the shear too.
    #[test]
    fn a_face_without_integer_ppems_keeps_its_em_off_a_uniform_scale() {
        let dir = d2d_reference::fonts_dir();
        let (Some(emoji), Some(arial)) = (dir.as_ref().and_then(|d| std::fs::read(d.join("seguiemj.ttf")).ok()), dir.as_ref().and_then(|d| std::fs::read(d.join("arial.ttf")).ok())) else { return };
        let glyphs = [Glyph { id: 25, advance: 10.0, offset: [0.0, 0.0] }];
        let e = GlyphRun {
            font_data: &emoji, face_index: 0, em_size: 19.442612, glyphs: &glyphs, baseline_origin: [0.0, 0.0], measuring_mode: 1,
            is_sideways: false, bidi_level: 0, no_snap: false, simulations: 0, coords: &[],
        };
        let a = GlyphRun { font_data: &arial, ..e };
        let (fe, fa) = (FontRef::new(&emoji).unwrap(), FontRef::new(&arial).unwrap());
        let shear = [1.0, -0.22949564456939697, 0.0, 1.0, 0.0, 0.0];
        let flip = [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert!(fractional_em(&shear, &e, &fe, MODE_GDI_CLASSIC));
        assert!(fractional_em(&shear, &e, &fe, MODE_GDI_NATURAL));
        assert!(fractional_em(&IDENTITY, &e, &fe, MODE_NATURAL_SYMMETRIC));
        assert!(!fractional_em(&IDENTITY, &e, &fe, MODE_GDI_CLASSIC));
        assert!(!fractional_em(&flip, &e, &fe, MODE_GDI_NATURAL));
        assert!(!fractional_em(&shear, &a, &fa, MODE_GDI_CLASSIC));
        // The sideways oblique of an upright glyph is not a world shear.
        let k = raster::OBLIQUE_SHEAR;
        assert!(!fractional_em(&[1.0, k, 0.0, 1.0, 0.0, 0.0], &GlyphRun { simulations: SIM_SIDEWAYS_OBLIQUE, ..e }, &fe, MODE_GDI_CLASSIC));
    }
}

#[cfg(test)]
mod integer_ppem_flag {
    use super::*;

    /// A face rounds the device em to a whole ppem only when its head flags ask for it (bit 3). Segoe UI
    /// and Segoe UI Emoji share the 'l' outline (1516 units high, upem 2048); at em 5.6 the flagged face
    /// draws it at ppem 6 (284/64 px), the unflagged one at the exact em (265/64 px).
    #[test]
    fn unflagged_face_keeps_the_exact_em() {
        for (file, top) in [("segoeui.ttf", 284.0), ("seguiemj.ttf", 265.0)] {
            let Some(bytes) = d2d_reference::fonts_dir().and_then(|d| std::fs::read(d.join(file)).ok()) else {
                eprintln!("skip: {file} not under D2D_FONTS / ref/windows/Fonts");
                return;
            };
            let font = FontRef::from_index(&bytes, 0).unwrap();
            let gid = font.charmap().map('l').unwrap().to_u32() as u16;
            let glyphs = [Glyph { id: gid, advance: 0.0, offset: [0.0, 0.0] }];
            let run = GlyphRun {
                font_data: &bytes, face_index: 0, em_size: 5.6, glyphs: &glyphs,
                baseline_origin: [0.0, 0.0], measuring_mode: 0, is_sideways: false,
                bidi_level: 0, no_snap: false, simulations: 0, coords: &[],
            };
            let (c, _) = hinted_contours(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &run, &font, MODE_NATURAL_SYMMETRIC, AA_GRAYSCALE).unwrap();
            let min_y = c.iter().flat_map(|(p, _)| p.iter().map(|q| q[1])).fold(f64::INFINITY, f64::min);
            assert_eq!(-min_y * 64.0, top, "{file}");
        }
    }
}
