//! `ID2D1DeviceContext::GetGlyphRunWorldBounds` — the world-space ink bounds of a positioned glyph
//! run, matching Direct2D's.
//!
//! Native builds the device transform (world ∘ DPI), runs the run through its glyph-run analyzer
//! (the same analyzer the ClearType/grayscale/aliased raster uses) and returns
//! its glyph-run bounds — the integer device-pixel rectangle the glyph coverage occupies
//! (the **ink box**, floor/ceil to pixels, unioned across the aliased and antialiased textures), not
//! the advance/layout box. It then divides the DPI back out, yielding world-space DIPs (the world
//! transform's scale/rotation stays, only the DPI scale is removed). Native always measures through
//! the analyzer, even at the em sizes where `DrawGlyphRun` would instead fill outlines, so this does
//! too — it is independent of the `fill_outline` render path.
//!
//! The texture bounds come straight from the shared scan converter (`text::raster`): its
//! `AlphaTexture` rectangle is already the floored/ceiled device bbox of the (grid-fitted) outline,
//! so reusing it keeps the bounds exact wherever the raster coverage is (the grid-fit-disabled
//! isolation scenes), with the hinting residual the rest of `text::` carries elsewhere.

use super::raster::{self, AlphaTexture};
use super::{hinted_contours_using, recommended_rendering_mode, resolve_text_aa, snap_baseline, GlyphRun, RenderingParams};
use crate::context::DeviceContext;
use skrifa::FontRef;

/// `D2D1_TEXT_ANTIALIAS_MODE` → the raster coverage kind (same mapping as `render_analysis`).
fn kind_of(aa: u32) -> raster::Kind {
    match aa {
        1 => raster::Kind::ClearType,
        3 => raster::Kind::Aliased,
        _ => raster::Kind::Grayscale,
    }
}

/// The glyph-run coverage texture the analyzer would produce (device pixels), picking the hinted or
/// unhinted contour source exactly as [`super::render_analysis`] does.
fn analyzer_texture(dt: &[f64; 6], run: &GlyphRun, font: &FontRef, params: &RenderingParams, mode: u32, aa: u32) -> AlphaTexture {
    let dt = &raster::scaler_matrix(dt);
    let kind = kind_of(aa);
    let rotated = super::hint_scale(dt).is_none();
    if params.grid_fit == 1 || rotated {
        let mut snapped = *run;
        snapped.baseline_origin = snap_baseline(run.baseline_origin, dt, run.no_snap);
        raster::unhinted(dt, &snapped, font, kind, super::horizontal_only(mode), matches!(mode, super::MODE_GDI_CLASSIC | super::MODE_GDI_NATURAL), raster::DEFAULT_DROPOUT)
    } else {
        match hinted_contours_using(dt, run, font, mode, aa, params.rasterizer, params.grid_fit == 2) {
            Some((c, drop)) => raster::coverage_outline(&c, drop, kind, (0, 0)),
            None => AlphaTexture { left: 0, top: 0, right: 0, bottom: 0, data: Vec::new() },
        }
    }
}

impl DeviceContext {
    /// `ID2D1DeviceContext::GetGlyphRunWorldBounds`: the world-space (DIP) ink bounds of `run`. An
    /// empty run — or a run that covers nothing (e.g. only spaces) — returns the native empty-rect
    /// sentinel `(inf, inf, -inf, -inf)` collapsed to a zero rect at the baseline origin, matching
    /// native's `right <= left` early-out (it leaves the caller's rect as the analyzer's empty box).
    pub fn glyph_run_world_bounds(&self, run: &GlyphRun, params: &RenderingParams) -> [f32; 4] {
        let empty = [0.0_f32; 4];
        if run.glyphs.is_empty() {
            return empty;
        }
        let Ok(font) = FontRef::from_index(run.font_data, run.face_index) else { return empty };
        let dt = self.device_transform();
        let text_aa = self.get_text_antialias_mode();
        let aa = match self.target_bitmap() {
            Some(t) => resolve_text_aa(text_aa, t),
            None => if text_aa != 0 { text_aa } else { 2 },
        };
        let mode = recommended_rendering_mode(run, &dt, text_aa, params);
        let mut rtl = Vec::new();
        let run = &super::drawn_ltr(run, if mode == super::MODE_OUTLINE { &super::IDENTITY } else { &dt }, &mut rtl);
        let tex = analyzer_texture(&dt, run, &font, params, mode, aa);
        if tex.right <= tex.left || tex.bottom <= tex.top {
            return empty;
        }
        // The rasterizer allocates the glyph bitmap one cell wider/taller on the right/bottom when a
        // max coordinate lands on a cell boundary, so a curve-tangent dropout sample can be recovered
        // into it (see raster::device_bounds). GetAlphaTextureBounds reports the INK extent, not the
        // allocation, so drop trailing all-zero right columns and bottom rows (bpp-aware). An empty
        // extension trims back to the tight bound (ffirects-01-glyph-bounds); a tangent that actually
        // recovered ink keeps the wider bound.
        let (w, h) = (tex.width(), tex.height());
        let bpp = (tex.data.len() / (w as usize * h as usize)).max(1);
        let px_nonzero = |col: i32, row: i32| {
            let o = (row as usize * w as usize + col as usize) * bpp;
            tex.data.get(o..o + bpp).is_some_and(|p| p.iter().any(|&b| b != 0))
        };
        let mut right = tex.right;
        while right - 1 > tex.left && !(0..h).any(|r| px_nonzero(right - 1 - tex.left, r)) {
            right -= 1;
        }
        let mut bottom = tex.bottom;
        while bottom - 1 > tex.top && !(0..w).any(|c| px_nonzero(c, bottom - 1 - tex.top)) {
            bottom -= 1;
        }
        let tex = AlphaTexture { right, bottom, ..tex };
        // Device pixels → world DIPs: divide the DPI back out (world transform stays). Unit mode
        // PIXELS keeps device = world, so the factor is 1.
        let (dx, dy) = self.get_dpi();
        let (sx, sy) = if self.get_unit_mode() == 0 { (96.0 / dx, 96.0 / dy) } else { (1.0, 1.0) };
        [tex.left as f32 * sx, tex.top as f32 * sy, tex.right as f32 * sx, tex.bottom as f32 * sy]
    }
}
