//! `ID2D1Ink` / `ID2D1InkStyle` + `DrawInk`.
//!
//! Direct2D dispatches `DrawInk` three ways: an opaque **fast path** draws an instanced nib through
//! the GPU vertex stage, and the non-opaque / odd-blend **as-geometry** fallback builds an ink
//! outline shape and fills it through the ordinary coverage rasterizer. This module reproduces the
//! as-geometry path (fully CPU-deterministic, reusing the proven rasterizer) and the opaque fast
//! path, byte-for-byte against Direct2D.
//!
//! As-geometry:
//! 1. Flatten each cubic-Bézier segment of the centre curve with the float Bézier flattener
//!    (tolerance `6·GetInkFlatteningTolerance`, round `0.5`/square `0.25`), carrying a per-vertex
//!    radius sampled from the cubic `(1−t)³r0 + 3(1−t)²t·r1 + 3(1−t)t²·r2 + t³r3`.
//! 2. Sweep the nib: a nib outline (round → circle, κ = `0x3f0d6289`; square → box) at every flattened
//!    vertex, plus a tangent trapezoid between consecutive vertices.
//! 3. Fill the union **nonzero** (it is the stroke), premultiply, and composite with the current
//!    primitive blend (`crate::pixels::fill_path`), byte-exact against the oracle.
//!
//! Opaque fast path (`v2-ink`): per-primitive-AA, opaque, SourceOver round ink. Reproduced in
//! [`swept_nib_mask`] as a feathered swept nib — an `n`-gon dot at each flattened centre-line point (`n`
//! from a level-of-detail table), plus a tangent-trapezoid body between points — MAX-accumulated into
//! a coverage mask, then the brush composited through it.
//!   * dot ([`feather_disc`]): two coincident **unit** `(cos, sin)` rings (outer cov 0, inner cov 1,
//!     at angle `(k·6.2831855)/N` via `cosf`/`sinf`) + a centre (cov 1); a feather triangle strip
//!     between the rings, then a centre fan over the inner ring. Each ring vertex sits at the nib
//!     silhouette `c + E·(u·r)` and feathers ±`FEATHER` device px along the radial normal
//!     `normalize(E·u)`. Arc count `N` from the level-of-detail table (`8 → 16` here).
//!   * body ([`body_vertex`]): instanced per segment, reading the two consecutive points' position and
//!     radius; it places a taper silhouette `S` (`cos = |Δr|/len`, `sin = √(1−cos²)`, the same tangent
//!     trapezoid `round_quad` computes) plus a `±0.5·n̂` feather along the unit device normal, per-vertex
//!     coverage `min(2·|A|, 1)·param1` with `|A| = radius` — core (param1 = 1) at `S−0.5·n̂` cov 1,
//!     feather (param1 = 0) at `S+0.5·n̂` cov 0: a 1.0-px symmetric band. Plain f32 (no FMA), with the
//!     GPU's `sqrt`/`rsq`.
//!   * composite ([`composite_mask`]): the brush colour is quantised **round-half-up**
//!     (`floor(c·255+0.5)`, the shader rule, *not* float-store round-to-nearest-even), then composited
//!     through a per-brush **gamma-corrected coverage LUT** ([`ink_gamma_lut`]), not a linear
//!     source-over. The LUT row is `LUT[i] = round_half_up(premul·val)` with
//!     `val = base[i] + ((brushLuma·slope[i])>>8)`, `base`/`slope` the gamma table and
//!     `brushLuma = round_half_up(255·(¼r+½g+¼b))`. This makes the solid interior byte-exact (`v2-ink`
//!     interior `[179,51,26,255]`, the `0.7·255 = 178.5` tie rounding to 179).
//!
//! The swept-nib mask is rendered per [`nib_mask_tiles`] tile with the tile's top-left at the atlas origin, and
//! the tile is the ink's bounds WITHOUT the feather, so the outer half-pixel past its top/left lies
//! outside the atlas viewport and is frustum-clipped before plane setup — a +1-A8 bias along the
//! straight body's top feather and the start cap's left rim that [`swept_nib_mask`]/[`atlas_tri`] reproduce.
//! The rim pixels further depend on the coverage raster's **fused** FMA plane eval
//! ([`crate::warp::fill_triangle_cov`]). `v2-ink` is byte-exact; the 8 aliased as-geometry scenes
//! (`ink2-*`, `inkfast-*`) take the separate fill path.

use crate::context::DeviceContext;
use crate::geometry::{flatten_fill_prims_scale, Matrix, Prim};
use crate::pixels::{Aa, Blend, FillMode, Target};
use crate::resource::SolidColorBrush;
use d2d_types::contracts::PixelBuffer;

/// One `D2D1_INK_POINT`: position and radius (DIPs).
pub type InkPoint = [f32; 3];

/// A `D2D1_INK_BEZIER_SEGMENT`: two control ink-points and the segment end.
#[derive(Clone, Copy)]
pub struct InkBezierSegment {
    pub p1: InkPoint,
    pub p2: InkPoint,
    pub end: InkPoint,
}

/// `ID2D1Ink`: a start ink-point and a list of cubic-Bézier ink segments.
pub struct Ink {
    pub start: InkPoint,
    pub segments: Vec<InkBezierSegment>,
}

/// `ID2D1InkStyle`: nib shape (`D2D1_INK_NIB_SHAPE` ROUND=0/SQUARE=1) and a 3×2 nib transform.
#[derive(Clone, Copy)]
pub struct InkStyle {
    pub nib_shape: u32,
    pub nib_transform: Matrix,
}

impl Default for InkStyle {
    fn default() -> Self {
        InkStyle { nib_shape: 0, nib_transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0] }
    }
}

impl Ink {
    /// `AddSegments`.
    pub fn add_segments(&mut self, segs: &[InkBezierSegment]) {
        self.segments.extend_from_slice(segs);
    }
    /// `RemoveSegmentsAtEnd`: drop `count` segments off the tail. Native rejects `count` greater than
    /// the segment count with `E_INVALIDARG` and leaves the ink unchanged (verified on the VM — it does
    /// not saturate), so guard here where both the COM shim and the renderer route through.
    pub fn remove_segments_at_end(&mut self, count: u32) -> Result<(), i32> {
        let n = self.segments.len();
        if count as usize > n {
            return Err(-2147024809); // E_INVALIDARG (0x80070057)
        }
        self.segments.truncate(n - count as usize);
        Ok(())
    }
    /// `StreamAsGeometry`: the swept-nib outline as device-space closed polygons (circles/boxes +
    /// connectors) whose nonzero union is the stroke — the same geometry `DrawInk`'s AsGeometry path
    /// fills, flattened the way native fills it (28.4 `CMILBezier`). `transform` is world∘DPI.
    pub fn stream_as_geometry(&self, transform: &Matrix, style: &InkStyle) -> Vec<Vec<[f32; 2]>> {
        // No render target here — return the full geometry, so disable the far-off-target cull.
        outline_polys(self, transform, style, 16.0, usize::MAX, usize::MAX)
    }

    /// `GetBounds`: the stroked-nib bounds under `transform` (world∘DPI already composed). Native's
    /// The segment flattener unions, per cubic segment, the four control points' device-space
    /// bounding box inflated by `max|radius|·GetNibExtents·factorHelper` — the loose cull box, not a
    /// tight swept bound.
    pub fn get_bounds(&self, transform: &Matrix, style: &InkStyle) -> [f32; 4] {
        let extent = nib_extents(style.nib_shape) * factor_helper(nib_matrix(transform, style.nib_transform));
        let mut b = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
        let mut p0 = self.start;
        for s in &self.segments {
            let pts = [p0, s.p1, s.p2, s.end];
            let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
            let mut max_r = 0.0f32;
            for p in pts {
                let d = apply(transform, [p[0], p[1]]);
                lo = [lo[0].min(d[0]), lo[1].min(d[1])];
                hi = [hi[0].max(d[0]), hi[1].max(d[1])];
                max_r = max_r.max(p[2].abs());
            }
            let inf = extent * max_r;
            b = [b[0].min(lo[0] - inf), b[1].min(lo[1] - inf), b[2].max(hi[0] + inf), b[3].max(hi[1] + inf)];
            p0 = s.end;
        }
        b
    }
}

impl DeviceContext {
    /// `CreateInk`: a new ink seeded with its start point.
    pub fn create_ink(&self, start: InkPoint) -> Ink {
        Ink { start, segments: Vec::new() }
    }

    /// `CreateInkStyle`.
    pub fn create_ink_style(&self, nib_shape: u32, nib_transform: Matrix) -> InkStyle {
        InkStyle { nib_shape, nib_transform }
    }

    /// `DrawInk` — **AsGeometry path**. Builds the swept-nib outline
    /// and fills it nonzero through the coverage rasterizer under the current antialias mode and
    /// primitive blend. This matches native for the non-opaque / odd-blend fallback; the opaque
    /// fast path (instanced WARP VS) is not reproduced — see the module note.
    pub fn draw_ink(&mut self, ink: &Ink, brush: &SolidColorBrush, style: Option<&InkStyle>) {
        if !self.is_in_draw() {
            return;
        }
        // A NaN POSITION is decided per-geometry by `ink_voids` below — a NaN CONTROL strokes the
        // collapsed segment (native's segment flattener emits {P0,P3}; VM dump, r21 26516 want
        // [80,52,39,130]), a NaN KNOT with no finite neighbour to close on voids — NOT a blanket void
        // here. A NaN RADIUS stays per-segment (`resolve_nan_radius`): a leading NaN-radius run drops and
        // the finite remainder strokes (r14 19962), but a NaN radius after a finite segment voids (16091).
        // Recording, a command list keeps the ink and strokes it where the list is drawn.
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            let segments = ink.segments.clone().into();
            self.record(crate::command::RecCmd::Ink { state, dpi, start: ink.start, segments, color: brush.color, opacity: brush.opacity, style: style.copied() });
            return;
        }
        let default = InkStyle::default();
        let style = style.unwrap_or(&default);
        let m = device_matrix(self);
        // Native fills the ink outline shape as a geometry unless the tiled paths take it: a
        // per-primitive ink under SOURCE_OVER, MIN or MAX goes to the tiled coverage-mask pass, and an opaque
        // solid one the fast path accepts under those blends draws direct.
        let tiled_blend = matches!(self.get_primitive_blend(), 0 | 2 | 4);
        let opaque = brush.opacity == 1.0 && brush.color[3].clamp(0.0, 1.0) == 1.0;
        if !(self.get_antialias_mode() != 1 && tiled_blend) && !(tiled_blend && opaque && can_fast_path(&m, style)) {
            let blend = blend_of(self.get_primitive_blend());
            if self.get_antialias_mode() == 1 {
                // The aliased fill is the general fill tessellator's scanner triangles, rasterized as they are.
                let clip = self.reach_rect();
                if ink_misses_clip(ink, &m, style, clip) {
                    return;
                }
                let Some(shape) = as_geometry_shape(ink, &m, style, Some(clip), 16.0, true) else { return };
                let solid = crate::brush::Brush::Solid(crate::brush::premul_f(brush.color, brush.opacity));
                let Some(mut buf) = self.take_target() else { return };
                for tri in crate::geometry::tessellate::tessellate(&shape, &crate::geometry::IDENTITY, 0.25) {
                    crate::brush::fill_triangle_brush(&mut buf, tri, &solid, blend);
                }
                self.set_target_bitmap(buf);
                return;
            }
            let Some(figures) = self.ink_outline_polys(ink, &m, style, 16.0) else { return };
            let Some(mut buf) = self.take_target() else { return };
            Target::new(&mut buf).fill_path(&figures, FillMode::Winding, brush.premul(), Aa::PerPrimitive, blend, true);
            self.set_target_bitmap(buf);
            return;
        }
        // A per-primitive ink under SOURCE_OVER, MIN or MAX goes through the tiled coverage-mask pass: drawn from the
        // ink's own point stream (`swept_nib_mask`) when the instanced fast path accepts the nib (on WARP a square
        // nib qualifies too), else from the ink outline shape (`outline_mask`).
        if self.get_antialias_mode() != 1 && tiled_blend {
            if ink_misses_clip(ink, &m, style, self.reach_rect()) {
                return;
            }
            let Some(mut buf) = self.take_target() else { return };
            let (w, h) = (buf.desc.width as usize, buf.desc.height as usize);
            let (mask, tiles) = if can_fast_path(&m, style) { swept_nib_mask(ink, &m, style, w, h) } else { outline_mask(ink, &m, style, w, h) };
            let solid = crate::brush::Brush::Solid(crate::brush::premul_f(brush.color, brush.opacity));
            if self.get_primitive_blend() == 2 {
                let lut = ink_min_lut([brush.color[0], brush.color[1], brush.color[2]], brush.color[3].clamp(0.0, 1.0) * brush.opacity);
                composite_mask_min(&mut buf, &mask, &lut);
            } else if self.get_primitive_blend() == 4 {
                composite_mask_brush(&mut buf, &mask, &solid, &tiles, 4);
            } else if opaque {
                composite_mask(&mut buf, &mask, &ink_gamma_lut([brush.color[0], brush.color[1], brush.color[2]]));
            } else {
                composite_mask_brush(&mut buf, &mask, &solid, &tiles, 0);
            }
            self.set_target_bitmap(buf);
            return;
        }
        let (tw, th) = self.target_bitmap().map_or((0, 0), |b| (b.desc.width as usize, b.desc.height as usize));
        let sub;
        // An aliased ink with an inf-DEVICE-position knot (a finite ink position that overflows under the
        // transform, finite radius) floods off its finite swept-nib prefix — native drops the inf knot's
        // quads rather than voiding. Bypass the NaN-radius/void machinery for it and render the full
        // outline with the non-finite figures dropped (r15 20064).
        let floods_inf = inf_pos_knot_floods(ink, &m, self.get_antialias_mode() == 1);
        let ink = match resolve_nan_radius(ink) {
            // A mid-stroke NaN radius voids unless a pre-NaN segment floods the target: then native
            // strokes that flood prefix (pre_nan_radius_floods, r15 20859).
            NanResolve::Void => match pre_nan_radius_floods(ink, &m, style, tw, th, self.get_antialias_mode() == 1) {
                Some(cut) => {
                    sub = Ink { start: ink.start, segments: ink.segments[..cut].to_vec() };
                    &sub
                }
                None if floods_inf => ink,
                None => return,
            },
            NanResolve::Keep => ink,
            NanResolve::Sub(s) => {
                sub = s;
                &sub
            }
        };
        if !floods_inf
            && (ink_voids(ink, &m, style.nib_shape) || nib_voids(&m, style) || ink_radius_voids(ink, &m, style, tw, th, self.get_antialias_mode() == 1) || ink_extent_voids(ink, &m, style) || ink_radius_threshold_voids(ink, &m, style))
        {
            return;
        }
        // Only the aliased direct route (an opaque solid ink the fast path accepts) is left here.
        let aa = Aa::Aliased;
        let blend_mode = self.get_primitive_blend();
        let (tw, th) = self.target_bitmap().map_or((0, 0), |b| (b.desc.width as usize, b.desc.height as usize));
        let figures = outline_polys(ink, &m, style, 16.0, tw, th);
        let src = brush.premul();
        let blend = blend_of(blend_mode);
        let Some(mut buf) = self.take_target() else {
            return;
        };
        Target::new(&mut buf).fill_path(&figures, FillMode::Winding, src, aa, blend, true);
        self.set_target_bitmap(buf);
    }

    /// `DrawInk` with a gradient/bitmap/image brush. The ink mask-path gate looks only
    /// at the nib geometry, not the brush, so a per-primitive opaque SourceOver round-nib ink takes it
    /// *with any brush*: native builds the swept-nib coverage mask and composites the brush through it
    /// with the general-purpose ink blend's per-pixel gamma correction (`composite_mask_brush`).
    /// A non-opaque brush (the LUT assumes α≡1), aliased, odd blend or square nib falls to `AsGeometry`
    /// (`brush::fill_geometry` over the swept outline), exactly the solid non-fast-path with a source.
    /// `AsGeometry` keeps the shared aliased ink-outline scan-converter tie.
    pub fn draw_ink_brush(&mut self, ink: &Ink, brush: &crate::PixelBrush, style: Option<&InkStyle>) {
        if !self.is_in_draw() || !self.check_brush(brush) {
            return;
        }
        // NaN POSITION is decided by `ink_voids` below (same as `draw_ink`); NaN RADIUS stays
        // per-segment (leading-drop vs mid-void, see `draw_ink`).
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            let segments = ink.segments.clone().into();
            self.record(crate::command::RecCmd::InkBrush { state, dpi, start: ink.start, segments, brush: brush.clone(), style: style.copied() });
            return;
        }
        let default = InkStyle::default();
        // A gradient or bitmap brush is placed under the device transform current at the draw, as every fill
        // realizes it (a world transform set after the brush was made must apply).
        let placed = self.realize_brush_for_fill(brush);
        let brush = placed.as_ref().unwrap_or(brush);
        let style = style.unwrap_or(&default);
        let m = device_matrix(self);
        // A brush ink the tiled coverage-mask pass does not take (aliased, or a blend other than SOURCE_OVER,
        // MIN or MAX) fills the ink outline shape; a non-solid brush never draws direct. An image brush, or a
        // bitmap brush sampled other than NEAREST/LINEAR, is redirected instead: the ink is drawn as a command
        // list image through the brush's effect graph (`BrushRequiresRedirection`, `RenderRedirectedCommand`).
        // ponytail: a redirected ink keeps the earlier void rules; port the command list's ink bounds
        // (`GetBoundsInternal`, NaN when CBounds' Bézier solve meets one) to drop them.
        let redirected = matches!(&brush.brush, crate::brush::Brush::Bitmap { origin, interpolation, .. } if *origin != crate::brush::BrushOrigin::PlainBitmap || *interpolation >= 2);
        if !redirected && !(self.get_antialias_mode() != 1 && matches!(self.get_primitive_blend(), 0 | 2 | 4)) {
            let aa = if self.get_antialias_mode() == 1 { Aa::Aliased } else { Aa::PerPrimitive };
            let blend = blend_of(self.get_primitive_blend());
            if aa == Aa::Aliased && blend == Blend::SourceOver && matches!(brush.brush, crate::brush::Brush::Linear { .. } | crate::brush::Brush::Radial { .. }) {
                let clip = self.reach_rect();
                if ink_misses_clip(ink, &m, style, clip) {
                    return;
                }
                let Some(shape) = as_geometry_shape(ink, &m, style, Some(clip), 64.0, true) else { return };
                let Some(mut buf) = self.take_target() else { return };
                for tri in crate::geometry::tessellate::tessellate(&shape, &crate::geometry::IDENTITY, 0.25) {
                    crate::brush::fill_triangle_brush(&mut buf, tri, &brush.brush, blend);
                }
                self.set_target_bitmap(buf);
                return;
            }
            let Some(figures) = self.ink_outline_polys(ink, &m, style, if aa == Aa::Aliased { 64.0 } else { 16.0 }) else { return };
            let Some(mut buf) = self.take_target() else { return };
            crate::brush::fill_geometry(&mut buf, &figures, true, aa, &brush.brush, blend);
            self.set_target_bitmap(buf);
            return;
        }
        if !redirected && self.get_antialias_mode() != 1 && matches!(self.get_primitive_blend(), 0 | 2 | 4) {
            if ink_misses_clip(ink, &m, style, self.reach_rect()) {
                return;
            }
            let Some(mut buf) = self.take_target() else { return };
            let (w, h) = (buf.desc.width as usize, buf.desc.height as usize);
            let (mask, tiles) = if can_fast_path(&m, style) { swept_nib_mask(ink, &m, style, w, h) } else { outline_mask(ink, &m, style, w, h) };
            composite_mask_brush(&mut buf, &mask, &brush.brush, &tiles, self.get_primitive_blend());
            self.set_target_bitmap(buf);
            return;
        }
        let (tw, th) = self.target_bitmap().map_or((0, 0), |b| (b.desc.width as usize, b.desc.height as usize));
        let sub;
        // An aliased ink with an inf-DEVICE-position knot (a finite ink position that overflows under the
        // transform, finite radius) floods off its finite swept-nib prefix — native drops the inf knot's
        // quads rather than voiding. Bypass the NaN-radius/void machinery for it and render the full
        // outline with the non-finite figures dropped (r15 20064).
        let floods_inf = inf_pos_knot_floods(ink, &m, self.get_antialias_mode() == 1);
        let ink = match resolve_nan_radius(ink) {
            // A mid-stroke NaN radius voids unless a pre-NaN segment floods the target: then native
            // strokes that flood prefix (pre_nan_radius_floods, r15 20859).
            NanResolve::Void => match pre_nan_radius_floods(ink, &m, style, tw, th, self.get_antialias_mode() == 1) {
                Some(cut) => {
                    sub = Ink { start: ink.start, segments: ink.segments[..cut].to_vec() };
                    &sub
                }
                None if floods_inf => ink,
                None => return,
            },
            NanResolve::Keep => ink,
            NanResolve::Sub(s) => {
                sub = s;
                &sub
            }
        };
        if !floods_inf
            && (ink_voids(ink, &m, style.nib_shape) || nib_voids(&m, style) || ink_radius_voids(ink, &m, style, tw, th, self.get_antialias_mode() == 1) || ink_extent_voids(ink, &m, style) || ink_radius_threshold_voids(ink, &m, style))
        {
            return;
        }
        let aa = if self.get_antialias_mode() == 1 { Aa::Aliased } else { Aa::PerPrimitive };
        let blend_mode = self.get_primitive_blend();
        // PARKED: native's ink mask-path gate is GEOMETRY-only (ignores opacity) — 16863's 0.68-opacity
        // brush takes the instanced fast path. Dropping is_opaque gains 16863 (6144→119) but needs the WARP triangle
        // raster (paint.rs) for 16561's emission + a byte-exact composite for the 1405/16863 ≤1-LSB floors.
        let fast = aa == Aa::PerPrimitive
            && blend_mode == 0
            && style.nib_shape != 1
            && can_fast_path(&m, style);
        let Some(mut buf) = self.take_target() else {
            return;
        };
        if fast {
            let (w, h) = (buf.desc.width as usize, buf.desc.height as usize);
            let (mask, tiles) = swept_nib_mask(ink, &m, style, w, h);
            composite_mask_brush(&mut buf, &mask, &brush.brush, &tiles, 0);
        } else {
            // Aliased brush ink is the as-geometry path → the WARP fill tessellator
            // (the WARP tessellator), which flattens the nib-circle beziers on a 26.6 (×64) grid, not the
            // 28.4 (×16) of the CPU edge list. Per-primitive keeps 28.4.
            let fill_scale = if aa == Aa::Aliased { 64.0 } else { 16.0 };
            let (tw, th) = (buf.desc.width as usize, buf.desc.height as usize);
            let figures = outline_polys(ink, &m, style, fill_scale, tw, th);
            let blend = blend_of(blend_mode);
            // A per-primitive (non-aliased) SourceOver brush ink that ISN'T the round positive-det fast
            // path (a square nib, or a reflected/degenerate nib that failed `can_fast_path`) still takes
            // native's ink mask blend: the ink mask-path gate is geometry-blind, so it composites the brush
            // through the swept-nib coverage mask with the general-purpose ink blend's PER-PIXEL gamma — exactly
            // the `fast` path above, but with the scan-converted coverage the non-round nib needs. The
            // AsGeometry `fill_geometry` below is a linear composite with a GLOBAL brush sample (wrong
            // colour, and a transparent global sample voids pixels native fills per-pixel). Build the same
            // 64-level swept-nib coverage the solid non-fast path builds (`outline_polys` → `rasterize_edges`)
            // and run it through `composite_mask_brush`'s float gamma band (byte-exact on
            // inkbrush-03-perprim-linear). Gated to a gradient brush (Linear/Radial) like the aliased-union
            // branch: a bitmap/image brush keeps `fill_geometry` (per-pixel image sampling through this mask
            // is far slower and not what native does for those).
            if aa == Aa::PerPrimitive
                && blend == Blend::SourceOver
                && matches!(brush.brush, crate::brush::Brush::Linear { .. } | crate::brush::Brush::Radial { .. })
            {
                let mut mask = vec![0f32; tw * th];
                // See the solid path: a non-finite start-cap device position voids the AsGeometry mask;
                // otherwise drop the NaN nibs and let a finite body flood.
                let s0 = apply(&m, [ink.start[0], ink.start[1]]);
                if s0[0].is_finite() && s0[1].is_finite() {
                    let finite_figs = drop_nonfinite_figures(figures.clone());
                    if let Some(clipped) = crate::pixels::clip_huge(&finite_figs) {
                        crate::pixels::rasterize_edges(&crate::pixels::figure_edges(&clipped), FillMode::Winding, aa, tw, th, |x, y, cov| {
                            mask[y * tw + x] = cov as f32 / 64.0;
                        });
                    }
                }
                composite_mask_brush(&mut buf, &mask, &brush.brush, &nib_mask_tiles(ink, &m, style, tw, th), 0);
            }
            // Aliased gradient ink: native fills the swept-nib union (circles+rects) through the
            // the intersecting scan sweep and fits the gradient texcoord plane PER rasterized triangle
            // (= our mesh/`fill_triangle_brush` path). Tessellate that union (`tessellate_ink_union`,
            // which pre-splits the crossings with native's exact arithmetic) and composite per triangle;
            // a non-gradient (bitmap/image) ink keeps the scan-converter + global sample.
            else if aa == Aa::Aliased
                && blend == Blend::SourceOver
                && matches!(brush.brush, crate::brush::Brush::Linear { .. } | crate::brush::Brush::Radial { .. })
            {
                // For the inf-position-knot flood, native's AsGeometry sink emits one BeginFigure/EndFigure
                // per swept-nib quad and never emits a quad for the inf knot's non-finite quad, so it is
                // DROPPED per-quad and the finite prefix floods uncancelled (r15 20064). Everywhere else a
                // non-finite vertex voids the whole ink, as the scan-converted (solid) path and native do
                // (fuzz r11 16404 inf radius draws nothing; 16539's 1.5M non-finite vertices hung the union;
                // r15 20793 / 23927's ink10 raw-inf CONTROL voids, not a per-quad drop-and-fill).
                let tris = if floods_inf {
                    crate::geometry::fill_tess::tessellate_ink_union(&drop_nonfinite_figures(figures))
                } else if figures.iter().flatten().all(|p| p[0].is_finite() && p[1].is_finite()) {
                    crate::geometry::fill_tess::tessellate_ink_union(&figures)
                } else {
                    Vec::new()
                };
                for tri in tris {
                    crate::brush::fill_triangle_brush(&mut buf, tri, &brush.brush, blend);
                }
            } else {
                crate::brush::fill_geometry(&mut buf, &figures, true, aa, &brush.brush, blend);
            }
        }
        self.set_target_bitmap(buf);
    }
}

impl DeviceContext {
    /// The ink outline shape's fill polygons over the current clip (the as-geometry fill), or `None` when
    /// nothing draws: the ink's grown bounds miss the clip, or a figure carries a NaN.
    fn ink_outline_polys(&self, ink: &Ink, m: &Matrix, style: &InkStyle, fill_scale: f32) -> Option<Vec<Vec<[f32; 2]>>> {
        let clip = self.reach_rect();
        if ink_misses_clip(ink, m, style, clip) {
            return None;
        }
        as_geometry_polys(ink, m, style, Some(clip), fill_scale, self.get_antialias_mode() == 1)
    }
}

// --- opaque fast path -----------------------------------

/// Device-space AA feather half-width: the swept-nib silhouette carries a coverage band from cov 1 at
/// `silhouette − FEATHER` to cov 0 at `silhouette + FEATHER` (symmetric, centred on the silhouette).
/// `0.5` is the **exact** value the instanced nib VS uses: the VS displaces the
/// core vertex by `−0.5·n̂` and the feather vertex by `+0.5·n̂` along the unit device normal, cov
/// `min(2·|A|,1)` (a clean 1-px linear band), so the
/// mask coverage is right; the former ≤13-LSB residual was the composite (now [`ink_gamma_lut`]), not
/// this constant — see the module `fast path` note.
const FEATHER: f32 = 0.5;

/// The fast-path test: round-nib fast path needs `det(nib∘device) ≥
/// factorHelper²·0.05` (and the matrix invertible). The WARP device flags (`+210` set, `+188`=0 for
/// per-primitive→mask path) always hold for the software target, so only the determinant gate is tested.
fn can_fast_path(m: &Matrix, style: &InkStyle) -> bool {
    let e = nib_matrix(m, style.nib_transform);
    let det = e[0] * e[3] - e[1] * e[2];
    let f = factor_helper(e);
    invert2x2(e).is_some() && det >= f * f * 0.05
}

/// Dot level-of-detail: the canonical circular dot's arc count for a stroke whose maximum nib radius
/// is `r`, from `Inking::GetInkTessellationData`'s table the level-of-detail table (round nib).
fn dot_arc_count(r: f32) -> usize {
    const TABLE: [(usize, f32); 9] = [
        (6, 1.8), (10, 5.1), (16, 13.0), (24, 29.2), (34, 58.6), (46, 107.2), (60, 182.4), (76, 292.6), (94, 447.6),
    ];
    for &(n, thresh) in &TABLE {
        if r <= thresh {
            return n;
        }
    }
    94
}

/// MAX-accumulate one faceted dot into `mask` as the instanced dot VS places it: an outer ring (`cov 0`),
/// an inner ring and a centre vertex, over `n` unit directions `u = (cos, sin)` of `(k·6.2831855)/n`. Each
/// vertex sits at `c + E·(u·r)` plus a feather offset along the unit normal `normalise(E⁻ᵀ·u)`. Its
/// half-width `A` is that offset projected onto the normal: from half a pixel up the band runs ±0.5 and
/// the core has coverage 1; below it the inner ring pulls in by `A` with coverage `4A²`, and the outer
/// ring sits `1 − A` out. Both offsets scale by the draw's feather `scale`. The band is the triangle strip
/// `outer[i], inner[i], outer[i+1], …`, then a fan from the centre vertex (direction `(1, 0)`) over the
/// inner ring. `e` = nib∘device 2×2 and `einv` its inverse, `c` = centre, `r` = radius.
#[allow(clippy::too_many_arguments)]
fn feather_disc(e: &[f32; 4], einv: &[f32; 4], c: [f32; 2], r: f32, n: usize, scale: f32, mask: &mut [f32], w: usize, h: usize) {
    if n < 3 {
        return;
    }
    // One direction's (outer vertex, inner vertex, inner coverage).
    let vertex = |u: [f32; 2]| {
        let o = apply2(e, [u[0] * r, u[1] * r]);
        let nraw = [u[0] * einv[0] + u[1] * einv[1], u[0] * einv[2] + u[1] * einv[3]];
        let ninv = 1.0 / (nraw[0] * nraw[0] + nraw[1] * nraw[1]).sqrt();
        let nn = [ninv * nraw[0], ninv * nraw[1]];
        let a = o[0] * nn[0] + o[1] * nn[1];
        let (cov, halfw) = if a < 0.5 { ((a * a) * 4.0, a) } else { (1.0, 0.5) };
        let inner = 1.0 - halfw;
        let at = |param1: f32| {
            let f = (param1 * (-inner - halfw) + inner) * scale;
            [(f * nn[0] + o[0]) + c[0], (f * nn[1] + o[1]) + c[1]]
        };
        (at(0.0), at(1.0), cov)
    };
    let ring: Vec<_> = (0..n)
        .map(|k| {
            let t = (k as f32 * 6.283_185_5) / n as f32;
            vertex([crate::crt::system_crt::cosf(t), crate::crt::system_crt::sinf(t)])
        })
        .collect();
    let centre_cov = vertex([1.0, 0.0]).2;
    for i in 0..n {
        let j = (i + 1) % n;
        let ((oi, ii, ci), (oj, ij, cj)) = (ring[i], ring[j]);
        atlas_tri(mask, w, h, [oi, ii, oj], [0.0, ci, 0.0]);
        atlas_tri(mask, w, h, [oj, ii, ij], [0.0, ci, cj]);
    }
    for i in 0..n {
        let j = (i + 1) % n;
        atlas_tri(mask, w, h, [c, ring[i].1, ring[j].1], [centre_cov, ring[i].2, ring[j].2]);
    }
}

/// MAX-accumulate the swept-body connector between two consecutive centre-line points into `mask`,
/// reproducing the instanced body vertex shader (precompiled VS entry 3, as emitted by the
/// `CreateVertexShader` hook — see the module note). Native draws the 8 cube-corner vertices
/// — `(PARAMETER0, PARAMETER1, PARAMETER2)` = (segment end 0/1, core/feather 1/0,
/// side 0/1) — as a triangle strip over the circle quad-index table `[0..8)`, instanced per segment, each
/// placed by the body VS. `p0`/`p1` are device-space `[x, y, radius]`.
fn body_strip(e: &[f32; 4], p0: InkPoint, p1: InkPoint, scale: f32, mask: &mut [f32], w: usize, h: usize) {
    // The canonical 8-vertex strip: the 8 cube corners in strip order 0..8.
    const CUBE: [[f32; 3]; 8] = [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
    ];
    // The body is the convex hull of two congruent nib ellipses `{p_i + E·(u·r_i)}` (nib∘device E,
    // ink radius r_i a NIB-space half-width). Map the device centre-line into nib space with `E⁻¹` so
    // the tangent taper `cos = |Δr|/len` uses the nib-space segment (the two nibs are unit circles
    // there); `body_vertex` then applies E to place each silhouette endpoint and measures the feather in
    // device space (see its note). This is geometrically exact for a rotated/skewed elliptical nib
    // (istyle4, 7463/18966), where the old scalar `√|det E|` radius approximation — a circle of the
    // right AREA but wrong SHAPE — over-covered. `can_fast_path` already verified E is invertible; for
    // identity E (the common fast path) every map is the identity, so this is byte-for-byte the old
    // device-space body (16863, inkbody, the device-radius probes stay exact).
    let einv = match invert2x2(*e) {
        Some(v) => v,
        None => return,
    };
    let q0 = apply2(&einv, [p0[0], p0[1]]);
    let q1 = apply2(&einv, [p1[0], p1[1]]);
    let (p0, p1) = ([q0[0], q0[1], p0[2]], [q1[0], q1[1], p1[2]]);
    let mut pos = [[0.0f32; 2]; 8];
    let mut cov = [0.0f32; 8];
    for (k, c) in CUBE.iter().enumerate() {
        match body_vertex(e, &einv, p0, p1, *c, scale) {
            Some((pt, v)) => (pos[k], cov[k]) = (pt, v), // body_vertex returns the device vertex
            None => return, // degenerate |Δradius| >= len: native collapses the body (the discs cover it)
        }
    }
    // 8-vertex triangle strip → 6 triangles, each odd one assembled `(vₙ₊₁, vₙ, vₙ₊₂)` as D3D does (the
    // winding the viewport clip walks); fill_triangle_cov adapts to either winding.
    for i in 0..6 {
        let k = if i % 2 == 0 { [i, i + 1, i + 2] } else { [i + 1, i, i + 2] };
        atlas_tri(mask, w, h, k.map(|k| pos[k]), k.map(|k| cov[k]));
    }
}

/// One body-VS vertex. `p0`/`p1` are consecutive **nib-space** `[x, y, radius]` (`body_strip` maps the
/// device centre-line in with `E⁻¹`); `e` is the nib∘device 2×2 and `einv` its inverse. `c = (param0,
/// param1, param2)` is the cube corner (segment end / core-feather / side). Returns the **device** vertex
/// and coverage, or `None` for the degenerate branch the shader sends to the origin.
///
/// The common outer tangent of the two nib circles touches both at one nib-space direction `ndir`
/// (`cos = |Δr|/len`, `sin = √(1−cos²)`). The shader interpolates the centre and radius along the
/// segment in nib space and maps the silhouette point `centre + radius·ndir` through `E`. Its feather
/// normal is `E⁻ᵀ·ndir`, normalised, and the half-width `|A|` is the device offset `E·(radius·ndir)`
/// projected onto it; a half-width under 0.5 px narrows the feather to `|A|` and scales the coverage
/// by `2|A|`. The offset scales by the draw's feather `scale`. Plain f32, each multiply-add as a multiply
/// then an add.
fn body_vertex(e: &[f32; 4], einv: &[f32; 4], p0: InkPoint, p1: InkPoint, c: [f32; 3], scale: f32) -> Option<([f32; 2], f32)> {
    let (param0, param1, param2) = (c[0], c[1], c[2]);
    let d = [p0[0] - p1[0], p0[1] - p1[1]]; // nib-space P0 − P1
    let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
    let dr = -p0[2] + p1[2]; // R1 − R0 (native add order)
    let absdr = dr.abs();
    let sign = if dr < 0.0 { -1.0 } else { 1.0 };
    if absdr >= len {
        return None;
    }
    let radius = param0 * dr + p0[2];
    let cos = absdr / len;
    let sin = ((-cos) * cos + 1.0).max(0.0).sqrt();
    let r6 = [d[0] / len, d[1] / len]; // (P0 − P1)/len
    let centre = [param0 * (-p0[0] + p1[0]) + p0[0], param0 * (-p0[1] + p1[1]) + p0[1]];
    let cs = cos * sign;
    let ss = sin * (param2 * -2.0 + 1.0);
    // nib-space outer-tangent direction (unit), shared by both endpoints.
    let ndir = [cs * r6[0] + (-r6[1]) * ss, cs * r6[1] + r6[0] * ss];
    let off = [radius * ndir[0], radius * ndir[1]];
    let sil = apply2(e, [off[0] + centre[0], off[1] + centre[1]]);
    let nraw = [ndir[0] * einv[0] + ndir[1] * einv[1], ndir[0] * einv[2] + ndir[1] * einv[3]];
    let ninv = 1.0 / (nraw[0] * nraw[0] + nraw[1] * nraw[1]).sqrt();
    let n = [ninv * nraw[0], ninv * nraw[1]];
    let off_dev = apply2(e, off);
    let abs_a = (off_dev[0] * n[0] + off_dev[1] * n[1]).abs();
    let (cov, halfw) = if abs_a < 0.5 { (abs_a * 2.0, abs_a) } else { (1.0, 0.5) };
    let inner = 1.0 - halfw;
    let om = (param1 * (-inner - halfw) + inner) * scale; // feather offset, scaled by the draw's feather
    Some(([om * n[0] + sil[0], om * n[1] + sil[1]], cov * param1))
}

/// Native-rounded float→UNORM8: WARP's `cvtss2si` (round-ties-even) with the `−0.5`-tie bump that
/// `d2d1`'s table code applies on every store, making it round-**half-up**.
fn round_hu(x: f32) -> i32 {
    let r = x.round_ties_even();
    let mut i = r as i32;
    if r - x == -0.5 {
        i += 1;
    }
    i
}

/// The 256-entry ink coverage coverage LUT the fast-path blend samples: a **gamma-corrected** coverage→premultiplied-colour table, not a linear
/// source-over. Per entry `i` (the A8 mask value): a gamma table (the gamma-table builder
/// mode 4 — `base[i] + ((brushLuma·slope[i])>>8)`) maps coverage to `val`, then the premultiplied
/// opaque brush is `round_half_up(channel·val)`. `brushLuma = round_half_up(255·(¼r+½g+¼b))`. The four
/// the gamma target-ratio constants and the two scale factors are the native values;
/// reproduced op-for-op in f32 (byte-exact against the oracle). `color` is the straight RGB (the
/// fast path is opaque, so alpha ≡ 1).
fn ink_gamma_lut(color: [f32; 3]) -> [[u8; 4]; 256] {
    let (base, slope) = ink_gamma_bands();
    let [r, g, b] = color;
    let luma = round_hu((((g * 0.5) + (r * 0.25)) + (b * 0.25)) * 255.0);
    let mut lut = [[0u8; 4]; 256];
    for (i, e) in lut.iter_mut().enumerate() {
        let lut_val = (base[i] + ((luma * slope[i]) >> 8)) as f32;
        *e = [
            round_hu(b * lut_val).clamp(0, 255) as u8,
            round_hu(g * lut_val).clamp(0, 255) as u8,
            round_hu(r * lut_val).clamp(0, 255) as u8,
            round_hu(lut_val).clamp(0, 255) as u8,
        ];
    }
    lut
}

/// The coverage→`(base, slope)` gamma band tables, the colour-**independent** `t`-only part of the
/// ink-mask gamma correction (the gamma-table builder mode 4). Per pixel the corrected
/// coverage is `val = base[cov] + ((brushLuma · slope[cov]) >> 8)`, baked into the solid
/// [`ink_gamma_lut`] (one colour).
fn ink_gamma_bands() -> ([i32; 256], [i32; 256]) {
    const RATIOS: [f32; 4] = [0.018_475_0, -0.099_075_0, 0.204_175_0, -0.057_175_0];
    let (g0, g1, g2, g3) = (RATIOS[0] * 4.031_434_1, RATIOS[1] * 4.015_686_5, RATIOS[2] * 4.031_434_1, RATIOS[3] * 4.015_686_5);
    let (mut base, mut slope) = ([0i32; 256], [0i32; 256]);
    for i in 0..256 {
        let t = i as f32 * 0.003_921_568_9;
        let band = (1.0 - t) * t;
        base[i] = round_hu((((t * g1 + g3) * band) + t) * 255.0) as u8 as i32;
        slope[i] = round_hu(((t * g0 + g2) * band) * 255.0) as u8 as i32;
    }
    (base, slope)
}

/// The fast-path blend: composite the opaque ink through the coverage `mask` by indexing the gamma LUT
/// with each pixel's quantised mask value (round-to-nearest-even A8) and source-over-ing the
/// premultiplied LUT colour onto the `B8G8R8A8` target in WARP's integer UNORM8 (`out = src +
/// dst·(255−src_a)`).
fn composite_mask(buf: &mut PixelBuffer, mask: &[f32], lut: &[[u8; 4]; 256]) {
    let (w, h) = (buf.desc.width as usize, buf.desc.height as usize);
    if let Ok(p) = std::env::var("D2D_INK_MASK_F32") {
        let bytes: Vec<u8> = mask[..w * h].iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = std::fs::write(&p, &bytes);
    }
    let um = |v: i32, m: i32| -> i32 {
        let t = v * m + 128;
        (t + (t >> 8)) >> 8
    };
    for y in 0..h {
        for x in 0..w {
            let cov = mask[y * w + x];
            if cov <= 0.0 {
                continue;
            }
            let c8 = (cov * 255.0).round_ties_even().clamp(0.0, 255.0) as usize;
            let src = lut[c8];
            let ia = 255 - src[3] as i32;
            let i = y * buf.stride + x * 4;
            for k in 0..4 {
                let dst = buf.data[i + k] as i32;
                buf.data[i + k] = (src[k] as i32 + um(dst, ia)).clamp(0, 255) as u8;
            }
        }
    }
}

/// The MIN ink lookup table (the per-colour ink table built for a MIN key): for each coverage byte
/// `v`, white blended toward the straight brush colour by `a·v/255` in the 1.4-power space, stored ×255
/// rounded half up, alpha 255.
fn ink_min_lut(c: [f32; 3], a: f32) -> [[u8; 4]; 256] {
    let pw = crate::crt::system_crt::powf;
    let k = [pw(c[0], 1.4) * a, pw(c[1], 1.4) * a, pw(c[2], 1.4) * a];
    let byte = |x: f32| round_hu(pw(x, 0.714_285_73) * 255.0) as u8;
    core::array::from_fn(|v| {
        let s = v as f32 / 255.0;
        let inv = 1.0 - a * s;
        [byte(k[2] * s + inv), byte(k[1] * s + inv), byte(k[0] * s + inv), 255]
    })
}

/// The instanced MIN composite: each covered pixel takes the per-channel minimum of the target and the
/// MIN lookup table's entry for its coverage byte.
fn composite_mask_min(buf: &mut PixelBuffer, mask: &[f32], lut: &[[u8; 4]; 256]) {
    let (w, h) = (buf.desc.width as usize, buf.desc.height as usize);
    for y in 0..h {
        for x in 0..w {
            let cov = mask[y * w + x];
            if cov <= 0.0 {
                continue;
            }
            let src = lut[(cov * 255.0).round_ties_even().clamp(0.0, 255.0) as usize];
            let i = y * buf.stride + x * 4;
            for k in 0..4 {
                buf.data[i + k] = buf.data[i + k].min(src[k]);
            }
        }
    }
}

/// the general-purpose ink blend: a gradient/bitmap/image brush composited through the coverage `mask`.
/// Unlike the solid [`composite_mask`] (a CPU-built byte LUT), the gamma correction runs in its pixel
/// kernel in f32 (byte-exact on 3 colours × every mask level): each byte `c`
/// unpacks as `c·(1/256) + c·(1/65280)`, the straight brush colour gives `L = (r·¼ + b·¼) + g·½`, the
/// A8 mask `a` is corrected to `a' = ((P·a + Q)·((1−a)·a))·4 + a` with `P = L·g0 + g1`, `Q = L·g2 + g3`
/// (`g` = the gamma target ratios), the premultiplied brush is scaled by `a'`, clamped and
/// stored ×255 to nearest, then source-over'd as `s + ((d·ia + (d·ia >> 8) + 128) >> 8)`.
///
/// Each tile is an aliased `CRectTessellator` fill, (TL, TR, BL) and (TR, BR, BL), so a gradient
/// samples its texcoord plane fitted per triangle at BL, the provoking vertex (traced quad of
/// inkbrush-03-perprim-linear: SetupTriangleInternal2 gets (BL, TL, TR), (BL, TR, BR)), as the
/// aliased geometry fill does, its lerp kept fractional.
///
/// `blend` is the primitive blend. Under MAX the kernel runs without the gamma correction (the brush
/// times the sampled coverage, for a solid brush too). Under MIN (a non-solid brush) it is the min-blend
/// kernel: the straight colour raised to 1.4, scaled by the coverage, blended from white by the brush
/// alpha and taken back through the 1/1.4 power, alpha 1. Both are combined into the target by the
/// output merger (max or min) and stored to nearest, ties to even.
fn composite_mask_brush(buf: &mut PixelBuffer, mask: &[f32], brush: &crate::brush::Brush, tiles: &[[f32; 4]], blend: u32) {
    use crate::brush::Brush;
    // The brush opacity is the shader's last multiply, after the kernel, not part of the sampled texel.
    let mut unit = brush.clone();
    let opacity = match &mut unit {
        Brush::Linear { opacity: o, .. } | Brush::Radial { opacity: o, .. } | Brush::Bitmap { opacity: o, .. } => std::mem::replace(o, 1.0),
        _ => 1.0,
    };
    let brush = &unit;
    const G: [f32; 4] = [0.018_475, -0.099_075, 0.204_175, -0.057_175];
    // `gamma_table`'s two scale factors (text/blend): a SOLID ink's coverage indexes the per-colour gamma
    // LUT (the fast-path ink blend), whose base/delta coefficients are the `G` row scaled by these, not ·4.
    const GAMMA_C0: f32 = 4.031_434_1;
    const GAMMA_C1: f32 = 4.015_686_5;
    let (w, h) = (buf.desc.width as usize, buf.desc.height as usize);
    let unorm = |c: f32| c * 0.003_906_25 + c * f32::from_bits(0x3780_8081);
    // A solid ink is one constant colour, so native composites it through its per-colour gamma LUT, not the
    // general-purpose per-pixel gamma kernel the gradient/bitmap brushes use (see the branch below).
    let solid = matches!(brush, Brush::Solid(_));
    let (a1, a2) = (G[0] * GAMMA_C0, G[2] * GAMMA_C0);
    let (b1, b2) = (G[1] * GAMMA_C1, G[3] * GAMMA_C1);
    let tris = tiles.iter().flat_map(|&tile| {
        let [l, t, r, b] = tile;
        [(tile, [[l, t], [r, t], [l, b]]), (tile, [[r, t], [r, b], [l, b]])]
    });
    let byte = |x: usize, y: usize| (mask[y * w + x] * 255.0 + 0.5).floor().clamp(0.0, 255.0) as i32;
    for ([l, t, r, b], pos) in tris {
        let (p, area) = crate::warp::snap_tri(pos);
        let sample = crate::brush::triangle_brush_sampler(pos, brush, true, [0.0, 0.0, w as f32, h as f32]);
        // The kernel reads the A8 tile bilinearly through the triangle's own texcoord plane (the tile at the
        // atlas origin, 1/512 per texel), fitted like any interpolated attribute; that plane can land a hair
        // off the texel centre and blend in a neighbour. Texels clamp to the atlas; past the tile it is clear.
        let (tw, th) = ((r - l) as i32, (b - t) as i32);
        let ord = [2usize, 0, 1];
        let (sx, sy) = (ord.map(|k| p[k][0]), ord.map(|k| p[k][1]));
        const INV: f32 = 1.0 / 512.0;
        let pu = crate::warp::plane::setup_triangle_attr(sx, sy, ord.map(|k| (pos[k][0] * INV + pos[k][1] * -0.0) + -l * INV));
        let pv = crate::warp::plane::setup_triangle_attr(sx, sy, ord.map(|k| (pos[k][0] * -0.0 + pos[k][1] * INV) + -t * INV));
        let texel = |i: i32, j: i32| {
            let (i, j) = (i.clamp(0, 511), j.clamp(0, 511));
            if i < tw && j < th { byte(l as usize + i as usize, t as usize + j as usize) } else { 0 }
        };
        let tile_sample = |x: usize, y: usize| -> f32 {
            let (xc, yc) = (x as f32 + 0.5, y as f32 + 0.5);
            let q = |pl: [f32; 3]| (((xc * pl[0] + (yc * pl[1] + pl[2])) * 512.0) * 256.0 - 128.0).floor() as i32;
            let (qu, qv) = (q(pu), q(pv));
            let (tx, fx, ty, fy) = (qu >> 8, qu & 0xff, qv >> 8, qv & 0xff);
            let sum = texel(tx, ty) * (256 - fx) * (256 - fy) + texel(tx + 1, ty) * fx * (256 - fy)
                + texel(tx, ty + 1) * (256 - fx) * fy + texel(tx + 1, ty + 1) * fx * fy;
            (sum >> 9) as f32 / 128.0
        };
        crate::warp::cover_tri(p, area, w as i64, h as i64, |x, y| {
            // A source-over solid brush's per-colour LUT indexes the coverage byte itself; the per-pixel
            // kernels read the tile through the sampler.
            let c8 = if solid && blend == 0 { byte(x, y) as f32 } else { tile_sample(x, y) };
            // The tile is composited whole; only under MIN does an uncovered pixel change (to white times
            // the brush opacity).
            if c8 <= 0.0 && blend != 2 {
                return;
            }
            // The gradient/bitmap kernel's filtered sample is 8.7 fixed point (the Q8 lerp `>> 1`); a SOLID
            // brush has no sampler lerp, so native's per-colour LUT keeps the premultiplied colour at full
            // precision (the 8.7 floor would nudge a channel sitting on a round boundary 1 LSB low).
            let s = if solid { sample(x, y).map(|v| unorm(v)) } else { sample(x, y).map(|v| unorm((v * 128.0).floor() / 128.0)) };
            let st = if s[3] > 0.0 { [s[0] / s[3], s[1] / s[3], s[2] / s[3]] } else { [s[0], s[1], s[2]] };
            let l = (st[2] * 0.25 + st[0] * 0.25) + st[1] * 0.5;
            // A solid ink is one constant colour, so native composites it through its per-colour gamma LUT
            // (the fast-path ink blend), not the general per-pixel kernel: the coverage byte `c8` reads a
            // round-half-up `base`/`delta` pair from `gamma_table` row 4 (the `G` ratios scaled by
            // `GAMMA_C0`/`GAMMA_C1`, not the kernel's flat ·4), the straight luminance byte `ef` modulates
            // the delta via `weight = base + ((ef·delta)>>8)`, and the premultiplied source is scaled by
            // that integer `weight`. This is byte-exact where the flat-·4 float band was ±1 (r22 27147,
            // r23 28602/28233). A gradient/bitmap brush has a per-pixel colour (no colour LUT), so it keeps
            // the float band (`else`), byte-exact on inkbrush-03-perprim-linear / ism-m-lin.
            let prod = if blend == 4 {
                // The MAX pixel kernel is just the sampled A8 coverage times the colour.
                let a = unorm(c8);
                s.map(|v| ((a * v) * opacity).clamp(0.0, 1.0) * 255.0)
            } else if blend == 2 {
                let pw = |x: f32, y: f32| crate::warpmath::exp2(crate::warpmath::log2(x) * y);
                let (m, a) = (unorm(c8), s[3]);
                let white = -m * a + 1.0;
                let c = st.map(|v| (pw((pw(v, 1.4) * m) * a + white, 1.0 / 1.4) * opacity) * 255.0);
                [c[0], c[1], c[2], 255.0]
            } else if solid {
                let c = c8 * 0.003_921_568_9;
                let band = (1.0 - c) * c;
                let base = (((c * b1 + b2) * band + c) * 255.0 + 0.5).floor() as i32;
                let delta = (((c * a1 + a2) * band) * 255.0 + 0.5).floor() as i32;
                let ef = (l * 255.0 + 0.5).floor().clamp(0.0, 255.0) as i32;
                let weight = (base as i64 + ((ef as i64 * delta as i64) >> 8)) as f32;
                s.map(|v| (v * weight).clamp(0.0, 255.0))
            } else {
                let (pp, q) = (l * G[0] + G[1], l * G[2] + G[3]);
                let a = unorm(c8);
                let adj = ((pp * a + q) * ((1.0 - a) * a)) * 4.0 + a;
                s.map(|v| ((v * adj) * opacity).clamp(0.0, 1.0) * 255.0)
            };
            // A FULLY-OPAQUE composite (premul source alpha 255) is a direct store of the quantized
            // gradient sample, so its UNORM8 rounds ties-to-even like the gradient FILL path
            // (`fill_geometry_cov`); a semi-transparent / gamma-corrected source rounds half-up (the
            // table-store bias the ink-mask LUT uses — fuzz 16863). Splitting on the source alpha keeps the
            // opaque-gradient inks (inkbrush-03-perprim-linear, ism-m-lin: 0.7·255 = 178.5 → 178) and the
            // semi-transparent gamma path both byte-exact.
            // MIN and MAX have no source-over table: the merged float is stored directly, ties to even.
            let opaque = blend != 0 || prod[3].round_ties_even() >= 255.0;
            let src = prod.map(|v| if opaque { v.round_ties_even() } else { (v + 0.5).floor() } as i32);
            let i = y * buf.stride + x * 4;
            for k in 0..4 {
                let v = src[k].clamp(0, 255) as u8;
                if blend != 0 {
                    buf.data[i + k] = if blend == 2 { buf.data[i + k].min(v) } else { buf.data[i + k].max(v) };
                    continue;
                }
                let t = buf.data[i + k] as i32 * (255 - src[3]);
                buf.data[i + k] = (src[k] + ((t + (t >> 8) + 128) >> 8)).clamp(0, 255) as u8;
            }
        });
    }
}

/// The device rects the instanced fast path blends the mask through: Direct2D takes
/// The ink-metrics pass takes the ink-space bounds of every point (control points included) and its max
/// |radius|, `Transform2DBounds` them, inflates by `GetNibExtents·(max_r·factorHelper)`, intersects the
/// target and rounds out and rounds out; the instanced fast path then walks that in square
/// tiles of `max(⌊2·extent⌋, 128)` px, at most 512.
fn nib_mask_tiles(ink: &Ink, m: &Matrix, style: &InkStyle, w: usize, h: usize) -> Vec<[f32; 4]> {
    let (bounds, max_r) = ink_metrics(ink);
    let d = transform_2d_bounds(m, bounds);
    let ext = nib_extents(style.nib_shape) * (max_r * factor_helper(nib_matrix(m, style.nib_transform)));
    let (l, t, r, b) = (d[0] - ext, d[1] - ext, d[2] + ext, d[3] + ext);
    let (l, t, r, b) = (l.max(0.0), t.max(0.0), r.min(w as f32), b.min(h as f32));
    if !(l < r && t < b) {
        return Vec::new();
    }
    let (l, t, r, b) = (l.floor() as i32, t.floor() as i32, r.ceil() as i32, b.ceil() as i32);
    let size = ((ext + ext) as i32).clamp(128, 512);
    let mut tiles = Vec::new();
    for y in (t..b).step_by(size as usize) {
        for x in (l..r).step_by(size as usize) {
            tiles.push([x, y, (x + size).min(r), (y + size).min(b)].map(|v| v as f32));
        }
    }
    tiles
}

/// Build the instanced fast path's feather coverage mask for `ink`, composited afterwards with the brush
/// ([`composite_mask`], [`composite_mask_min`], [`composite_mask_brush`]).
/// `InkStageManager::BeginDraw` renders each [`nib_mask_tiles`] tile on its own, with the tile's top-left
/// at the 512×512 atlas origin, and only the tile's pixels are blended; so the feather past a tile's
/// top or left edge is frustum-clipped by the atlas viewport ([`atlas_tri`]). Also returns the tiles whose
/// stream added any vertices: native composites only the tiles whose mask primitive received vertices.
fn swept_nib_mask(ink: &Ink, m: &Matrix, style: &InkStyle, w: usize, h: usize) -> (Vec<f32>, Vec<[f32; 4]>) {
    let e = nib_matrix(m, style.nib_transform);
    let factor = factor_helper(e);
    let tol = ink_flattening_tolerance(style.nib_shape);
    let tiles = nib_mask_tiles(ink, m, style, w, h);
    // Per tile, the instanced tessellator streams the ink in that tile's atlas space (the tile offset folded
    // into the device transform), culled against the tile only when the ink's grown bounds spill past it.
    let (bounds, max_r) = ink_metrics(ink);
    let d = transform_2d_bounds(m, bounds);
    let ext = nib_extents(style.nib_shape) * (max_r * factor);
    let grown = [d[0] - ext, d[1] - ext, d[2] + ext, d[3] + ext];
    let streams: Vec<Vec<(InkPoint, bool, bool)>> = tiles
        .iter()
        .map(|&[l, t, r, b]| {
            let spills = l > grown[0] || t > grown[1] || grown[2] > r || grown[3] > b;
            let mt = [m[0], m[1], m[2], m[3], m[4] - l, m[5] - t];
            instanced_points(ink, &mt, factor, spills.then_some([0.0, 0.0, r - l, b - t]), tol, style.nib_shape)
        })
        .collect();
    // The dot level of detail and the feather half-width both key off the draw's
    // nib reach, DrawInk's `nibExtents·(max|r|·factorHelper(E))` over the ink's control points (the
    // metrics, not the culled point stream). One ring + one feather width serve every dot of the stroke
    // (native's DrawIndexedInstanced uses a single ring for all instances). The feather saturates at
    // `FEATHER` once the nib reaches one pixel and scales down below it (sub-pixel nibs collapse).
    let maxdev = ext;
    let n = dot_arc_count(maxdev);
    let fw = maxdev.min(FEATHER);
    let nan = |p: &InkPoint| p.iter().any(|v| v.is_nan());
    // A square nib draws its dots and bodies through the square-nib shaders (feather width `2·fw`).
    let square = style.nib_shape == 1;
    let ei = invert2x2(e).unwrap_or([0.0; 4]);
    let mut mask = vec![0.0f32; w * h];
    for (&[l, t, r, b], pts) in tiles.iter().zip(&streams) {
        let (tw, th) = ((r - l) as usize, (b - t) as usize);
        let mut tile = vec![0.0f32; tw * th];
        // A dot or body with a NaN position or radius has NaN vertices, which the rasterizer drops.
        for (i, &(p, dot, connect)) in pts.iter().enumerate() {
            let c = [p[0], p[1]];
            if dot && !nan(&p) && !bbox_misses(c, c, ink_reach(&e, p[2]), tw, th) {
                if square {
                    atlas_strip(&mut tile, tw, th, &square_dot_verts(&e, &ei, c, p[2], 2.0 * fw), &SQUARE_DOT_STRIP);
                } else {
                    feather_disc(&e, &ei, c, p[2], n, 2.0 * fw, &mut tile, tw, th);
                }
            }
            let Some(&(q, _, _)) = pts.get(i + 1).filter(|_| connect) else { continue };
            if nan(&p) || nan(&q) {
                continue;
            }
            let reach = ink_reach(&e, p[2].abs().max(q[2].abs()));
            let (lo, hi) = ([p[0].min(q[0]), p[1].min(q[1])], [p[0].max(q[0]), p[1].max(q[1])]);
            if !bbox_misses(lo, hi, reach, tw, th) {
                if square {
                    atlas_strip(&mut tile, tw, th, &square_body_verts(&e, &ei, p, q, 2.0 * fw), &SQUARE_BODY_STRIP);
                } else {
                    body_strip(&e, p, q, 2.0 * fw, &mut tile, tw, th);
                }
            }
        }
        for y in 0..th {
            let row = (t as usize + y) * w + l as usize;
            mask[row..row + tw].copy_from_slice(&tile[y * tw..(y + 1) * tw]);
        }
    }
    let drawn = tiles.iter().zip(&streams).filter(|(_, p)| !p.is_empty()).map(|(t, _)| *t).collect();
    (mask, drawn)
}

/// The tiled coverage mask for a nib the instanced fast path refuses (singular, reflected or too
/// thin): per [`nib_mask_tiles`] tile, native builds the ink outline shape in the tile's
/// atlas space (the tile offset folded into the device transform), culled against the tile only when the
/// ink's grown bounds spill past it, and fills it with the hardware rasterizer clipped to the tile: the
/// same edge walk as a COPY geometry fill (pixel-centre shift, trapezoids on clean bands).
/// Also returns the tiles the rasterizer emitted anything into, the only ones composited.
fn outline_mask(ink: &Ink, m: &Matrix, style: &InkStyle, w: usize, h: usize) -> (Vec<f32>, Vec<[f32; 4]>) {
    let (bounds, max_r) = ink_metrics(ink);
    let d = transform_2d_bounds(m, bounds);
    let ext = nib_extents(style.nib_shape) * (max_r * factor_helper(nib_matrix(m, style.nib_transform)));
    let grown = [d[0] - ext, d[1] - ext, d[2] + ext, d[3] + ext];
    let mut mask = vec![0.0f32; w * h];
    let mut drawn = Vec::new();
    for [l, t, r, b] in nib_mask_tiles(ink, m, style, w, h) {
        let spills = l > grown[0] || t > grown[1] || grown[2] > r || grown[3] > b;
        let mt = [m[0], m[1], m[2], m[3], m[4] - l, m[5] - t];
        let Some(polys) = as_geometry_polys(ink, &mt, style, spills.then_some([0.0, 0.0, r - l, b - t]), 16.0, false) else { continue };
        let slices: Vec<&[[f32; 2]]> = polys.iter().map(|f| f.as_slice()).collect();
        let (lu, tu) = (l as usize, t as usize);
        let mut any = false;
        crate::pixels::copy_tess::fill_copy(&slices, false, (r as usize) - lu, (b as usize) - tu, true, |x, y, cov, _| {
            mask[(tu + y) * w + lu + x] = cov.c8() as f32 / 255.0;
            any = true;
        });
        if any {
            drawn.push([l, t, r, b]);
        }
    }
    (mask, drawn)
}

/// The vertex buffer the instanced tessellator fills with the ink's segment points: every
/// point carries a dot flag and a connect-to-next flag. A batch that is not the ink's first and whose first
/// point does not equal the last buffered point exactly (position and radius, so never a NaN) starts a new
/// run: its first point gets no dot. One that does equal it joins: that point is dropped and the previous
/// one connects. A batch's last point connects to nothing until a later batch joins it.
fn instanced_points(ink: &Ink, m: &Matrix, factor: f32, clip: Option<[f32; 4]>, tol: f32, nib_shape: u32) -> Vec<(InkPoint, bool, bool)> {
    let mut out: Vec<(InkPoint, bool, bool)> = Vec::new();
    send_ink_segments(ink, m, factor, clip, tol, nib_shape, &mut |pts, first| {
        let mut pts = pts;
        let mut dot = first;
        if !first {
            if let Some(last) = out.last_mut().filter(|l| l.0 == pts[0]) {
                last.2 = true;
                pts = &pts[1..];
                dot = true;
            }
        }
        let n = pts.len();
        out.extend(pts.iter().enumerate().map(|(i, &p)| (p, i > 0 || dot, i + 1 < n)));
    });
    out
}

// --- square-nib instanced shaders ---------------------------------------------------------------

/// The square dot's mesh: (corner x 0/1, corner y 0/1, coverage vertex 0/1, edge normal x, y), drawn as
/// a triangle strip over [`SQUARE_DOT_STRIP`]: an inner box at full coverage ringed by an octagon at zero.
const SQUARE_DOT_MESH: [[f32; 5]; 12] = [
    [0.0, 0.0, 0.0, 0.0, -1.0], [1.0, 0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 1.0],
];
const SQUARE_DOT_STRIP: [usize; 23] = [2, 3, 0, 4, 1, 5, 5, 2, 6, 3, 7, 4, 8, 5, 9, 9, 6, 6, 7, 10, 8, 11, 9];
/// The square body's mesh: (segment end 0/1, coverage vertex 0/1, side 0/1, cap 0/1), a triangle strip
/// over [`SQUARE_BODY_STRIP`].
const SQUARE_BODY_MESH: [[f32; 4]; 12] = [
    [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0],
];
const SQUARE_BODY_STRIP: [usize; 20] = [0, 1, 2, 3, 4, 5, 6, 7, 7, 2, 2, 8, 4, 9, 9, 3, 3, 10, 5, 11];

/// A square nib of radius `r` shrunk by the half feather `h` along each axis, measured through the inverse
/// nib∘device matrix's column lengths: `(coverage, half-width x, half-width y)`. An axis that shrinks past
/// zero collapses to zero, and its coverage scales down (x by `r/(h·n0)`, y by `2·(r−h·n1)/(h·n1)+1`).
fn square_shrink(r: f32, h: f32, ei: &[f32; 4]) -> (f32, f32, f32) {
    let n0 = (ei[0] * ei[0] + ei[2] * ei[2]).sqrt();
    let n1 = (ei[1] * ei[1] + ei[3] * ei[3]).sqrt();
    let (hn0, hn1) = (h * n0, h * n1);
    let (rx, ry) = ((-h) * n0 + r, (-h) * n1 + r);
    let (mut cov, ix) = if rx < 0.0 { (rx / hn0 + 1.0, 0.0) } else { (1.0, rx) };
    let iy = if ry < 0.0 {
        cov *= (ry + ry) / hn1 + 1.0;
        0.0
    } else {
        ry
    };
    (cov, ix, iy)
}

/// `E·(ox, oy) + c`, the order the ink vertex shaders evaluate it.
fn nib_offset(e: &[f32; 4], ox: f32, oy: f32, c: [f32; 2]) -> [f32; 2] {
    [(ox * e[0] + oy * e[2]) + c[0], (ox * e[1] + oy * e[3]) + c[1]]
}

/// The square-nib dot vertex shader over [`SQUARE_DOT_MESH`]: each vertex's atlas position and coverage.
/// The coverage vertices sit on the shrunk box; the others push the box corner out by the feather width
/// `w` along the edge normal mapped through `E⁻¹` and normalised.
fn square_dot_verts(e: &[f32; 4], ei: &[f32; 4], c: [f32; 2], r: f32, w: f32) -> [([f32; 2], f32); 12] {
    let (cov, ix, iy) = square_shrink(r, 0.5 * w, ei);
    SQUARE_DOT_MESH.map(|m| {
        let mut p = nib_offset(e, ix * (m[0] * 2.0 - 1.0), iy * (m[1] * 2.0 - 1.0), c);
        if m[2] != 1.0 {
            let n = [m[3] * ei[0] + m[4] * ei[1], m[3] * ei[2] + m[4] * ei[3]];
            let rs = 1.0 / (n[0] * n[0] + n[1] * n[1]).sqrt();
            p = [w * (rs * n[0]) + p[0], w * (rs * n[1]) + p[1]];
        }
        (p, cov * m[2])
    })
}

/// Which box corner (x sign, y sign) a square body takes on `side`: the extents of the two boxes in nib
/// space (positions through `E⁻¹`) decide it, as the corner table of the geometry route does.
fn square_body_corner(ei: &[f32; 4], a: InkPoint, b: InkPoint, side: f32) -> (f32, f32) {
    let ext = |p: InkPoint| {
        let (nx, ny) = (p[1] * ei[2] + p[0] * ei[0], p[1] * ei[3] + p[0] * ei[1]);
        [-p[2] + nx, p[2] + nx, -p[2] + ny, p[2] + ny]
    };
    let (ea, eb) = (ext(a), ext(b));
    let q: [f32; 4] = core::array::from_fn(|k| {
        let c = eb[k] >= ea[k];
        side * (if c { -1.0 } else { 1.0 }) + if c { 1.0 } else { 0.0 }
    });
    let r = q.map(|v| 1.0 - v);
    let z1 = (r[2] * r[1]) * r[3] + ((r[0] * (1.0 - r[2])) * (1.0 - r[3]) + r[1] * r[0]);
    let z2 = (q[0] * q[3]) * q[1] + (((q[2] * r[0]) * r[1]) + q[3] * q[2]);
    let sign = |z: f32| if z > 0.0 { -1.0 } else if z < 0.0 { 3.0 } else { 1.0 };
    (sign(z2), sign(z1))
}

/// The square-nib body vertex shader over [`SQUARE_BODY_MESH`] between points `a` and `b`: the side
/// vertices run along the two chosen box corners, feathered ∓`w/2` along the edge's normal (narrowed with
/// the coverage where the corners pass within half a pixel of the centre line); the cap vertices are each
/// end's shrunk box corner.
fn square_body_verts(e: &[f32; 4], ei: &[f32; 4], a: InkPoint, b: InkPoint, w: f32) -> [([f32; 2], f32); 12] {
    SQUARE_BODY_MESH.map(|m| {
        let (t, cov_flag, side) = (m[0], m[1], m[2]);
        let (sx, sy) = square_body_corner(ei, a, b, side);
        if m[3] == 1.0 {
            let rad = t * (b[2] - a[2]) + a[2];
            let cen = [t * (b[0] - a[0]) + a[0], t * (b[1] - a[1]) + a[1]];
            let (cov, ix, iy) = square_shrink(rad, 0.5 * w, ei);
            return (nib_offset(e, ix * sx, iy * sy, cen), cov * cov_flag);
        }
        let oa = nib_offset(e, sx * a[2], sy * a[2], [0.0, 0.0]);
        let ob = nib_offset(e, sx * b[2], sy * b[2], [0.0, 0.0]);
        let ac = [oa[0] + a[0], oa[1] + a[1]];
        let bc = [ob[0] + b[0], ob[1] + b[1]];
        let d = [bc[0] - ac[0], bc[1] - ac[1]];
        let perp = [d[1], -d[0]];
        let rs = 1.0 / (perp[0] * perp[0] + perp[1] * perp[1]).sqrt();
        let mut n = [rs * perp[0], rs * perp[1]];
        if side == 0.0 {
            n = [-n[0], -n[1]];
        }
        let od = [t * (ob[0] - oa[0]) + oa[0], t * (ob[1] - oa[1]) + oa[1]];
        let proj = (od[0] * n[0] + od[1] * n[1]).abs();
        let (cov, tt) = if proj < 0.5 { (proj * 2.0, proj) } else { (1.0, 0.5) };
        let base = [t * d[0] + ac[0], t * d[1] + ac[1]];
        let k0 = 1.0 - tt;
        let k = (cov_flag * (-k0 - tt) + k0) * w;
        ([k * n[0] + base[0], k * n[1] + base[1]], cov * cov_flag)
    })
}

/// Rasterize a vertex shader's triangle strip into the atlas tile (degenerate triangles draw nothing).
fn atlas_strip(mask: &mut [f32], w: usize, h: usize, verts: &[([f32; 2], f32)], strip: &[usize]) {
    for k in 0..strip.len().saturating_sub(2) {
        let (i, j, l) = if k % 2 == 0 { (strip[k], strip[k + 1], strip[k + 2]) } else { (strip[k + 1], strip[k], strip[k + 2]) };
        if i == j || j == l || i == l {
            continue;
        }
        atlas_tri(mask, w, h, [verts[i].0, verts[j].0, verts[l].0], [verts[i].1, verts[j].1, verts[l].1]);
    }
}

/// One mask triangle in tile-local atlas coordinates: WARP frustum-clips a triangle with a vertex
/// outside the 512×512 atlas viewport before its plane setup, interpolating the coverage onto the
/// clipped fan (inkbody-01/02: the feather past the tile's top and left).
fn atlas_tri(mask: &mut [f32], w: usize, h: usize, pos: [[f32; 2]; 3], cov: [f32; 3]) {
    const ATLAS: [f32; 4] = [0.0, 0.0, 512.0, 512.0];
    if pos.iter().any(|p| p[0] < 0.0 || p[0] > 512.0 || p[1] < 0.0 || p[1] > 512.0) {
        let tri: [[f32; 3]; 3] = core::array::from_fn(|k| [pos[k][0], pos[k][1], cov[k]]);
        for s in crate::warp::viewport_clip(tri, ATLAS) {
            crate::warp::fill_triangle_cov(mask, w, h, s.map(|v| [v[0], v[1]]), s.map(|v| v[2]));
        }
    } else {
        crate::warp::fill_triangle_cov(mask, w, h, pos, cov);
    }
}

/// Device half-reach of a disc/body of radius `r` under nib∘device `e`: `|r|·‖E‖₁` bounds `E·(u·r)`
/// for `|u|≤1` in each axis, plus the feather band. The conservative radius both ink paths use to skip
/// far-off-target work: a figure whose bbox (centre ± reach) misses the target draws nothing — the
/// fill clamps rows/cols to [0,w]×[0,h], and a closed figure entirely outside the target adds zero net
/// winding to any in-target pixel — so skipping it keeps on-target output byte-identical. This bounds
/// the per-point work for a centre-line that enters from far off-target (fuzz 1258, an ink scaled to
/// device x≈12800), with no allocation scaling with the off-target extent.
fn ink_reach(e: &[f32; 4], r: f32) -> f32 {
    r.abs() * (e[0].abs() + e[1].abs() + e[2].abs() + e[3].abs()) + FEATHER + 1.0
}

/// True when the bbox `[lo, hi]` grown by `reach` does not intersect the target `[0,w]×[0,h]`.
/// `usize::MAX` for either dimension means "unbounded" (geometry extraction) — never a miss.
fn bbox_misses(lo: [f32; 2], hi: [f32; 2], reach: f32, w: usize, h: usize) -> bool {
    if w == usize::MAX || h == usize::MAX {
        return false;
    }
    hi[0] + reach < 0.0 || lo[0] - reach > w as f32 || hi[1] + reach < 0.0 || lo[1] - reach > h as f32
}

// --- outline construction --------------------------------------------------------------------

/// Drop the swept figures that carry a non-finite vertex (a device-overflowing control's quads go NaN),
/// keeping the finite ones. The PER-PRIMITIVE mask path uses this before `clip_huge` so a finite flooding
/// body still clips and strokes, instead of the whole ink voiding when `clip_huge` bails on the first NaN
/// vertex (r21 26737: seg2's 3.4e38 controls → NaN quads drop, seg0's finite body floods off the start
/// cap to its 1e9 knot). Native's culling sink drops those nibs the same way. The ALIASED AsGeometry path
/// deliberately does NOT use this — it keeps the NaN so its finite-check voids the figure, as native's
/// aliased sink poisons it. A finite ink keeps every figure, so this is inert for it.
fn drop_nonfinite_figures(figures: Vec<Vec<[f32; 2]>>) -> Vec<Vec<[f32; 2]>> {
    figures.into_iter().filter(|f| f.iter().all(|v| v[0].is_finite() && v[1].is_finite())).collect()
}

/// Device-space fill polygons for the swept nib, flattened via the 28.4 fill flattener. Round nib →
/// a circle (`AddEllipse`) at each vertex + an outer-tangent trapezoid (`ConstructRoundQuadVertices`)
/// between consecutive vertices; square nib → a box at each vertex + a connector whose two corners
/// per box come from `LookupSquareQuadCorners`. Their nonzero union is the stroke.
fn outline_polys(ink: &Ink, m: &Matrix, style: &InkStyle, fill_scale: f32, w: usize, h: usize) -> Vec<Vec<[f32; 2]>> {
    let e = nib_matrix(m, style.nib_transform); // nib∘device 2×2 (ellipse/box offsets)
    let einv = invert2x2(e);
    // Robustness (fuzz 841): a non-finite / enormous nib radius makes every swept nib span far past the
    // target, so each nib flattens to thousands of vertices and the union tessellation chokes (one seed
    // reached 4.2M verts → multi-hour hang). A nib whose device reach already blankets the whole target
    // is visually identical to one clamped to that reach, and off-target nibs are then culled by
    // `bbox_misses`; clamp each point's radius so `ink_reach` can't exceed ~2× the target extent (plus
    // a margin). Valid ink radii are far smaller than this, so it is inert for real strokes. A NaN
    // radius (degenerate input) collapses to 0 rather than poisoning the fold.
    // A rank-1 (singular, non-invertible) nib∘device maps the unit disc to a line segment, so the swept
    // shape is a BAND, not discs + connectors: native's degenerate sink sweeps each centre-line edge into
    // a quad whose two sides are the vertices offset by ±radius·off along the degenerate offset vector.
    // (A rank-0 / all-zero nib was already voided by `nib_voids`.) The band dispatches BEFORE the radius
    // clamp below: native culls a band edge whose `centre ± radius·off` overflows f32 to ±inf (the quad is
    // non-finite and dropped), so an off-target huge-radius segment draws nothing — the clamp would turn
    // it into a finite target-sized quad that spuriously fills.
    if einv.is_none() {
        let raw = flatten_centerline(ink, m, style.nib_shape);
        // Native's segment flattener coarsens a NaN-POSITION-control segment to its finite {P0,P3} knots
        // (AIS n=2) and emits ONE band quad over the full chord (r19 24319). A NaN position makes every
        // interior flattened point NaN, so the unfiltered band is EMPTY; dropping the non-finite points
        // makes the finite knots consecutive so the window bridges them (the bridged quad == native's
        // a quad emit). ONLY for a NaN POSITION: an inf/huge control (non-NaN) also drives flattened points
        // non-finite, but native does NOT coarsen those — it drops the overflowing quads, exactly what the
        // unfiltered `degenerate_band` already does and renders byte-exact (r16 21570 huge pos + NaN
        // RADIUS; r21 26496 huge pos/radius — both must stay on the original path). Gating on a NaN control
        // position confines the bridge to the 24319 class.
        if ink_has_nan_pos(ink) {
            let pts: Vec<InkPoint> = raw
                .iter()
                .copied()
                .filter(|p| p[0].is_finite() && p[1].is_finite() && p[2].is_finite())
                .collect();
            let mut polys = degenerate_band(&pts, &e, style.nib_shape, w, h, fill_scale);
            // Coverage truncation: native emits the full {P0,P3} band quad but RASTERISES only the finite
            // flattened REMAINDER — a whole-column vertical cut at `floor(P2.deviceX)+4`, dropping the
            // NaN-poisoned prefix (VM-derived: op2-* P2-sweep + 24319, the band quad stays identical while
            // the cut tracks P2). Clip only when the filter actually dropped a poisoned prefix.
            if pts.len() < raw.len() {
                if let Some(xcut) = degenerate_bridge_xcut(ink, m) {
                    polys = polys
                        .into_iter()
                        .map(|p| clip_poly_xmax(&p, xcut))
                        .filter(|p| p.len() >= 3)
                        .collect();
                }
            }
            return polys;
        }
        return degenerate_band(&raw, &e, style.nib_shape, w, h, fill_scale);
    }
    let l1e = e[0].abs() + e[1].abs() + e[2].abs() + e[3].abs();
    let r_cap = (l1e > 1e-6).then(|| (2.0 * (w as f32 + h as f32) + 64.0) / l1e);
    let clamp_r = |r: f32| match r_cap {
        Some(rc) => if r.is_nan() { 0.0 } else { r.clamp(-rc, rc) },
        None => r,
    };
    // A control radius whose device silhouette `|radius|·factorHelper(E)·nibExtents` overflows f32 drops
    // its WHOLE segment's swept body, not just the nibs at that control. Native forward-differences the
    // per-point radius across a segment; an overflowing control radius pushes the second-difference
    // `6·(r0−2r1+r2)` past f32, so EVERY interior point's radius becomes ±inf and its silhouette vertex
    // `centre + E·(radius·dir)` is non-finite — the sink commits none of them, keeping only the segments
    // whose control radii all stay finite. The start cap (the ink's start point, its own leading disc) and
    // a shared knot that is a finite segment's start/end cap survive. This is the per-segment analogue of
    // the start-cap `ink_extent_voids` (fuzz recap9 12740: seg0.p2 carries a 3.4e38 control RADIUS → only
    // the start cap draws, a tiny ink, instead of voiding the stroke OR flooding the target). A FINITE
    // control radius (even 1e9) keeps its segment and is clamped below, so it still floods as native does.
    let nib_ext = nib_extents(style.nib_shape);
    let fh = factor_helper(e);
    let sil_overflow = |r: f32| !(r.abs() * fh * nib_ext).is_finite();
    let round = style.nib_shape != 1;
    let mut polys: Vec<Vec<[f32; 2]>> = Vec::new();
    if ink_has_nan_pos(ink) {
        // A NaN control/endpoint POSITION does NOT void: native flattens the segment (its per-segment bbox
        // cull's fminf/fmaxf ignore NaN) and lays a disc at every flattened point plus a connector quad
        // between CONSECUTIVE points. A non-finite point's disc and BOTH its connectors (to the previous and
        // the next point) carry its NaN and are dropped by the rasteriser — so native does NOT bridge across
        // the gap: the finite points on either side are their own discs, unconnected across the NaN. Reset
        // `last` at a skip so a connector never spans a dropped point (fuzz 920002: a narrow all-positive band
        // with a NaN segment — a spanning bridge over-drew it 294 px vs native's 92). Runs break only at a
        // finite→device OVERFLOW (16863); a raw ±inf control has voided upstream (`ink_voids`). Non-NaN inks
        // take the flat path below, byte-identical to the historical ink.
        // A point whose RADIUS is non-finite is dropped like a non-finite position: an overflowing control
        // radius (3.4e38) forward-differences the interior radii to ±inf, so native's sink drops those
        // silhouette batches — the flat path's `sil_overflow` analogue for the NaN-position run-bridge path.
        // A finite-huge radius (1e9) stays finite here and still floods via `clamp_r` below, as native does
        // (r19 24186: seg0 has a NaN position, a 1e9 knot that floods, and a 3.4e38 end that overflows and
        // must drop — clamping its inf points to `r_cap` over-drew the flood's winding).
        let placeable = |p: &InkPoint| p[0].is_finite() && p[1].is_finite() && p[2].is_finite();
        for run in &flatten_centerline_runs(ink, m, style.nib_shape, false) {
            let mut last: Option<(InkPoint, [[f32; 2]; 4])> = None;
            for &raw in run.iter() {
                if !placeable(&raw) {
                    last = None; // a dropped point breaks the connector chain — native joins only neighbours
                    continue;
                }
                let c = [raw[0], raw[1]];
                let r = clamp_r(raw[2]);
                let cb = if round { [[0.0; 2]; 4] } else { box_corners(c, r, &e) };
                if !bbox_misses(c, c, ink_reach(&e, r), w, h) {
                    if round {
                        polys.push(flatten_fill_prims_scale(ellipse_start(c, r, &e), &ellipse_prims(c, r, &e), fill_scale, None));
                    } else {
                        polys.push(cb.to_vec());
                    }
                }
                if let (Some((prev, pb)), Some(einv)) = (last, einv) {
                    let reach = ink_reach(&e, r.abs().max(prev[2].abs()));
                    let (lo, hi) = ([c[0].min(prev[0]), c[1].min(prev[1])], [c[0].max(prev[0]), c[1].max(prev[1])]);
                    if !bbox_misses(lo, hi, reach, w, h) {
                        let (p0n, p1n) = (apply2(&einv, [prev[0], prev[1]]), apply2(&einv, c));
                        if round {
                            if let Some(q) = round_quad(p0n, prev[2], p1n, r) {
                                polys.push(q.iter().map(|&v| apply2(&e, v)).collect());
                            }
                        } else if let Some((a, b)) = square_corners(p0n, prev[2], p1n, r) {
                            polys.push(vec![pb[a], pb[b], cb[b], cb[a]]);
                        }
                    }
                }
                last = Some(([c[0], c[1], r], cb));
            }
        }
    } else {
        // Flat flatten (one list), the historical path for inks with no NaN position. A finite control that
        // overflows to a non-finite device coord flattens to ±inf/NaN vertices that `bbox_misses` culls and
        // whose connectors are skipped, so it drops out without a bridge.
        // Flatten per segment (mirroring `flatten_centerline`) so a whole overflowing segment's body can be
        // dropped: `keep[i]` is false for every interior point of a segment whose control radii overflow the
        // silhouette (see `sil_overflow` above). The start cap (pt0) is always kept; a segment's end knot
        // (which is also the next segment's start cap) is kept when EITHER adjacent segment stays finite.
        let tol = ink_flattening_tolerance(style.nib_shape);
        let tp = |p: InkPoint| -> InkPoint { let d = apply(m, [p[0], p[1]]); [d[0], d[1], p[2]] };
        let start = tp(ink.start);
        let mut pts = vec![[start[0], start[1], start[2].abs()]];
        let mut keep = vec![true];
        let seg_overflow = |r0: f32, s: &InkBezierSegment| {
            sil_overflow(r0) || sil_overflow(s.p1[2]) || sil_overflow(s.p2[2]) || sil_overflow(s.end[2])
        };
        let mut p0 = start;
        let mut r0 = ink.start[2];
        for (k, s) in ink.segments.iter().enumerate() {
            let ovf = seg_overflow(r0, s);
            let before = pts.len();
            flatten_ink_segment(p0, tp(s.p1), tp(s.p2), tp(s.end), tol, &mut pts);
            keep.resize(pts.len(), !ovf);
            // The last appended point is this segment's end knot = the next segment's start cap: keep it
            // when the next segment is finite even if this one overflowed.
            if pts.len() > before {
                if let Some(next) = ink.segments.get(k + 1) {
                    if !seg_overflow(s.end[2], next) {
                        *keep.last_mut().unwrap() = true;
                    }
                }
            }
            p0 = tp(s.end);
            r0 = s.end[2];
        }
        for p in &mut pts {
            p[2] = clamp_r(p[2]);
        }
        let boxes: Vec<[[f32; 2]; 4]> = if round {
            Vec::new()
        } else {
            pts.iter().map(|p| box_corners([p[0], p[1]], p[2], &e)).collect()
        };
        for (i, &p) in pts.iter().enumerate() {
            if !keep[i] {
                continue; // dropped overflow nib: no disc/box, and no connector touches it
            }
            let c = [p[0], p[1]];
            let r = p[2];
            if !bbox_misses(c, c, ink_reach(&e, r), w, h) {
                if round {
                    polys.push(flatten_fill_prims_scale(ellipse_start(c, r, &e), &ellipse_prims(c, r, &e), fill_scale, None));
                } else {
                    polys.push(boxes[i].to_vec());
                }
            }
            if i == 0 || !keep[i - 1] {
                continue; // first knot, or a dropped predecessor — no connector to a non-survivor
            }
            let Some(einv) = einv else { continue };
            let prev = pts[i - 1];
            let reach = ink_reach(&e, r.abs().max(prev[2].abs()));
            let (lo, hi) = ([c[0].min(prev[0]), c[1].min(prev[1])], [c[0].max(prev[0]), c[1].max(prev[1])]);
            if bbox_misses(lo, hi, reach, w, h) {
                continue;
            }
            let (p0n, p1n) = (apply2(&einv, [prev[0], prev[1]]), apply2(&einv, c));
            if round {
                if let Some(q) = round_quad(p0n, prev[2], p1n, r) {
                    polys.push(q.iter().map(|&v| apply2(&e, v)).collect());
                }
            } else if let Some((a, b)) = square_corners(p0n, prev[2], p1n, r) {
                polys.push(vec![boxes[i - 1][a], boxes[i - 1][b], boxes[i][b], boxes[i][a]]);
            }
        }
    }
    // The WARP fill tessellator snaps every vertex it receives (quad- and line-emit corners too, not just
    // flattened beziers) to its 26.6 grid with round-to-nearest. Circle verts are already gridded by
    // `flatten_fill_prims_scale`; snap the straight round_quad/box corners the same way so the whole
    // outline lives on the grid native fills.
    if fill_scale > 16.0 {
        let snap = |v: f32| ((fill_scale * v - fill_scale * 0.5).round() + fill_scale * 0.5) / fill_scale;
        for p in &mut polys {
            for v in p.iter_mut() {
                *v = [snap(v[0]), snap(v[1])];
            }
        }
    }
    polys
}

/// The rank-1 (singular) nib's offset vector: the nib maps the unit disc to a line segment, so the swept
/// shape is a band `centre-line ± radius·off`. `off` is the dominant column `d` of `E` (column 0, or
/// column 1 when column 0 ≈ 0) normalized to `d̂`, projected back onto both columns — `(d̂·col0, d̂·col1)`
/// — a square nib (`shape ≠ 0`) scaling it by `|d̂.x| + |d̂.y|`. Columns are `col0 = (e[0], e[2])`,
/// `col1 = (e[1], e[3])`. Mirrors native's `GetDegenerateOffsetVector`.
fn degenerate_offset_vector(e: &[f32; 4], nib_shape: u32) -> [f32; 2] {
    let (dx, dy) = if e[0].abs() < 0.000_001 && e[2].abs() < 0.000_001 {
        (e[1], e[3])
    } else {
        (e[0], e[2])
    };
    let len = ((dx * dx) + (dy * dy)).sqrt();
    if len < 0.000_001 {
        return [0.0, 0.0];
    }
    let (hx, hy) = (dx / len, dy / len);
    let off = [(hx * e[0]) + (hy * e[2]), (hx * e[1]) + (hy * e[3])];
    if nib_shape != 0 {
        let s = hx.abs() + hy.abs();
        [off[0] * s, off[1] * s]
    } else {
        off
    }
}

/// Build the swept-nib band for a rank-1 (singular) nib: each consecutive pair of flattened centre-line
/// vertices `A`, `B` becomes a closed quad with sides `A ± rA·off` and `B ± rB·off`, wound by the sign of
/// `(B.y − A.y)·off.x − (B.x − A.x)·off.y` so the nonzero union is the stroke. A quad whose union bbox
/// misses the target, or carries a non-finite vertex, is dropped (native's culling sink). Mirrors
/// the degenerate-ink geometry adapter.
fn degenerate_band(pts: &[InkPoint], e: &[f32; 4], nib_shape: u32, w: usize, h: usize, fill_scale: f32) -> Vec<Vec<[f32; 2]>> {
    let off = degenerate_offset_vector(e, nib_shape);
    let edge = |p: InkPoint, sgn: f32| [p[0] + sgn * p[2] * off[0], p[1] + sgn * p[2] * off[1]];
    let snap = |v: f32| ((fill_scale * v - fill_scale * 0.5).round() + fill_scale * 0.5) / fill_scale;
    let mut polys: Vec<Vec<[f32; 2]>> = Vec::new();
    for win in pts.windows(2) {
        let (a, b) = (win[0], win[1]);
        let reach = ink_reach(e, a[2].abs().max(b[2].abs()));
        let (lo, hi) = ([a[0].min(b[0]), a[1].min(b[1])], [a[0].max(b[0]), a[1].max(b[1])]);
        if bbox_misses(lo, hi, reach, w, h) {
            continue;
        }
        let (ap, am, bp, bm) = (edge(a, 1.0), edge(a, -1.0), edge(b, 1.0), edge(b, -1.0));
        let s = (b[1] - a[1]) * off[0] - (b[0] - a[0]) * off[1];
        let quad = if s <= 0.0 { [am, ap, bp, bm] } else { [ap, am, bm, bp] };
        if quad.iter().all(|p| p[0].is_finite() && p[1].is_finite()) {
            polys.push(if fill_scale > 16.0 {
                quad.iter().map(|&v| [snap(v[0]), snap(v[1])]).collect()
            } else {
                quad.to_vec()
            });
        }
    }
    polys
}

/// Device x at which native truncates a bridged (NaN-control) degenerate band: the device x of the
/// first finite control POSITION after the NaN, plus [`BRIDGE_CUT_SLACK`]. Native rasterises the band
/// only over the finite flattened remainder (end knot → this control); the NaN-poisoned prefix drops.
/// VM-derived (r19 24319 + op2-* P2-sweep: the right boundary is a vertical line tracking this x).
fn degenerate_bridge_xcut(ink: &Ink, m: &Matrix) -> Option<f32> {
    let ctrls = core::iter::once(ink.start)
        .chain(ink.segments.iter().flat_map(|s| [s.p1, s.p2, s.end]));
    let mut seen_nan = false;
    for p in ctrls {
        let finite = p[0].is_finite() && p[1].is_finite();
        if seen_nan && finite {
            let d = apply(m, [p[0], p[1]]);
            // Native keeps WHOLE device columns up to `floor(P2.x) + BRIDGE_CUT_COLS`, then clips — a
            // per-column cut, not a sub-pixel slice (a kept column shows its FULL natural band coverage,
            // op2-2 col 54 = 255). The `+4` is the band's own device x-extent past the control, a VM-
            // derived constant (independent of the control radius across the op2-* P2-sweep and 24319).
            // Clip at the right edge of the last kept column (an integer device x) so kept columns are
            // untouched and only columns beyond it drop.
            return d[0].is_finite().then_some(d[0].floor() + BRIDGE_CUT_COLS + 1.0);
        }
        seen_nan |= p[0].is_nan() || p[1].is_nan();
    }
    None
}

/// Device columns of band kept past the truncation control for a bridged degenerate band (VM-derived:
/// op2-* P2-sweep + r19 24319; `last kept col = floor(P2.deviceX) + 4`).
const BRIDGE_CUT_COLS: f32 = 4.0;

/// Clip a polygon to the half-plane `x ≤ xmax` (Sutherland–Hodgman, single edge). Used to truncate a
/// bridged degenerate band at native's vertical coverage cut.
fn clip_poly_xmax(poly: &[[f32; 2]], xmax: f32) -> Vec<[f32; 2]> {
    let n = poly.len();
    if n < 2 {
        return Vec::new();
    }
    let mut out: Vec<[f32; 2]> = Vec::with_capacity(n + 2);
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        let (ain, bin) = (a[0] <= xmax, b[0] <= xmax);
        if ain {
            out.push(a);
        }
        if ain != bin {
            let t = (xmax - a[0]) / (b[0] - a[0]);
            out.push([xmax, a[1] + t * (b[1] - a[1])]);
        }
    }
    out
}

/// A NaN in any control POSITION (x/y): native's ink sink rejects the whole DrawInk (the point can't be
/// placed), so it voids regardless of where the NaN sits. A NaN RADIUS is handled per-segment by
/// [`resolve_nan_radius`] instead (it can be a droppable leading cap).
fn ink_has_nan_pos(ink: &Ink) -> bool {
    core::iter::once(ink.start)
        .chain(ink.segments.iter().flat_map(|s| [s.p1, s.p2, s.end]))
        .any(|p| p[0].is_nan() || p[1].is_nan())
}

enum NanResolve {
    /// No NaN radius — stroke the ink unchanged.
    Keep,
    /// A NaN radius poisons the connected mid-stroke outline — void the whole DrawInk.
    Void,
    /// A leading NaN-radius run was dropped; stroke this finite remainder.
    Sub(Ink),
}

/// Resolve a NaN in the ink's RADII (positions go through [`ink_has_nan_pos`] → whole void). Native's
/// per-segment sweep builds each flattened point's radius as a cubic Bernstein of its four control radii
/// (the segment flatten), so a NaN radius makes that segment's interior points NaN. Native DROPS a
/// LEADING run of NaN-radius geometry — the start cap (if its radius is NaN) plus any first segments
/// whose radius is NaN — and strokes the finite remainder (fuzz r14 19962: a NaN at seg0 draws
/// seg1/seg2). A NaN radius that first appears AFTER a finite segment voids the whole ink: it poisons
/// the connected outline mid-stroke and the shared scanner rejects the lot (fuzz r11 16091: a NaN at
/// seg1). Dropping the whole stroke, or a NaN at the first kept disc, also voids.
fn resolve_nan_radius(ink: &Ink) -> NanResolve {
    let seg_nan = |s: &InkBezierSegment| s.p1[2].is_nan() || s.p2[2].is_nan() || s.end[2].is_nan();
    // A non-finite control POSITION (an `inf`; NaN positions voided upstream) in a segment makes its HFD
    // flatten compute `inf − inf = NaN` between finite neighbours, poisoning the outline — so such a
    // segment is NOT a clean radius-only drop.
    let bad_pos = |s: &InkBezierSegment| {
        [s.p1, s.p2, s.end].iter().any(|p| !(p[0].is_finite() && p[1].is_finite()))
    };
    let start_nan = ink.start[2].is_nan();
    if !start_nan && !ink.segments.iter().any(seg_nan) {
        return NanResolve::Keep;
    }
    // The leading NaN-radius run: a NaN start cap attaches to seg0, then drop leading NaN segments — but
    // only while they are POSITION-finite. A dropped segment with a non-finite position voids instead
    // (fuzz r18 23099: a NaN radius at seg0 that also has a `-inf` control under a 1e6 nib → void, not
    // the leading-drop that fuzz 19962's position-finite seg0 gets).
    let mut k = usize::from(start_nan);
    while k < ink.segments.len() && seg_nan(&ink.segments[k]) {
        if bad_pos(&ink.segments[k]) {
            return NanResolve::Void;
        }
        k += 1;
    }
    // A NaN radius after the leading run (preceded by a finite segment) is a mid-stroke poison → void;
    // dropping every segment leaves nothing to stroke → void.
    if k == 0 || k >= ink.segments.len() || ink.segments[k..].iter().any(seg_nan) {
        return NanResolve::Void;
    }
    // The dropped leading run (the NaN-radius start cap and the segments consumed before the first kept
    // knot) must be POSITION-finite to drop cleanly: a non-finite position there — an inf start cap, or an
    // inf control in the bridge segment whose end becomes the new start — poisons the join to the kept
    // remainder, so native voids the whole ink rather than stroking the remainder (18749 istyle11: an inf
    // start position with an inf seg0 control). A position-finite leading run still drops (19962). This is
    // the same position guard the radius-NaN loop applies above, extended to the bridge the loop skips.
    let dropped_pos_bad = !(ink.start[0].is_finite() && ink.start[1].is_finite())
        || ink.segments[..k]
            .iter()
            .any(|s| [s.p1, s.p2, s.end].iter().any(|p| !(p[0].is_finite() && p[1].is_finite())));
    if dropped_pos_bad {
        return NanResolve::Void;
    }
    // Stroke from the last dropped segment's end (the first kept disc); a NaN there can't start it.
    let start = ink.segments[k - 1].end;
    if start[2].is_nan() {
        return NanResolve::Void;
    }
    NanResolve::Sub(Ink { start, segments: ink.segments[k..].to_vec() })
}

fn flatten_centerline(ink: &Ink, m: &Matrix, nib_shape: u32) -> Vec<InkPoint> {
    let tol = ink_flattening_tolerance(nib_shape);
    // Native's segment flattener builds each flattened point's radius as |cubic Bernstein of the SIGNED
    // control radii| — the magnitude is taken on the RESULT, not the controls — so a negative control
    // mid-curve can pull the interpolated magnitude below the abs-at-control value (23229's p2 = −40). Pass
    // the signed radius through; `radius_cubic` takes the `|·|` on its Bernstein result. The start point's
    // radius is |Bernstein(0)| = |r0|. (A constant negative radius can't tell the two apart — |−2| at every
    // control equals |Bernstein(−2…−2)| — which is why the old constant-radius probe read as abs-at-control.)
    let tp = |p: InkPoint| -> InkPoint {
        let d = apply(m, [p[0], p[1]]);
        [d[0], d[1], p[2]]
    };
    let start = tp(ink.start);
    let mut out = vec![[start[0], start[1], start[2].abs()]];
    let mut p0 = start;
    for s in &ink.segments {
        let (p1, p2, p3) = (tp(s.p1), tp(s.p2), tp(s.end));
        flatten_ink_segment(p0, p1, p2, p3, tol, &mut out);
        p0 = p3;
    }
    out
}

/// PARKED (is_opaque-drop WIP): per-segment RUNS for the ink fast path. Native flattens per segment
/// (the per-segment flattener); a segment whose FINITE raw control point
/// overflows to a NON-FINITE DEVICE coord (op6/16350: −3.4e38×1.25=−inf) emits nothing — break the run
/// there. A raw ±inf stays (r9 812 clips+draws); a raw NaN voids the whole ink (`ink_has_nan`). AsGeometry
/// keeps the flat `flatten_centerline`. NOTE: our `flatten_ink_segment` matches the Bézier flattener
/// (crates/d2d/src/geometry/flatten.rs) bit-for-bit, so the flatten is already exact; the open residuals
/// (16561 emission over-reach, 812 raw-inf clip, 1405/16863 ≤1-LSB composite floors) are NOT the flatten.
fn flatten_centerline_runs(ink: &Ink, m: &Matrix, nib_shape: u32, break_at_nan: bool) -> Vec<Vec<InkPoint>> {
    let tol = ink_flattening_tolerance(nib_shape);
    // Signed control radius, like `flatten_centerline`: the per-segment flatten's `radius_cubic` takes the
    // `|·|` on its Bernstein RESULT (native's segment flattener), so the controls stay signed; a standalone
    // point pushed on its own (a run's first knot) carries `|radius|` since its radius is |Bernstein(0)|.
    let tp = |p: InkPoint| -> InkPoint {
        let d = apply(m, [p[0], p[1]]);
        [d[0], d[1], p[2]]
    };
    let knot = |p: InkPoint| -> InkPoint { [p[0], p[1], p[2].abs()] };
    let overflow = |raw: InkPoint, dev: &InkPoint| {
        raw[0].is_finite() && raw[1].is_finite() && (!dev[0].is_finite() || !dev[1].is_finite())
    };
    // The only UNCONDITIONAL break is a finite control that OVERFLOWS to a non-finite DEVICE coord: native
    // run-breaks there and strokes the finite remainder (16863). A raw NaN control is handled two ways
    // depending on the caller:
    //  - `break_at_nan` (the `ink_voids` test): also break at a raw NaN, so the only NaN LEFT inside a run
    //    comes from a raw ±inf's inf−inf, which voids the ink (16561/812) — a NaN POSITION itself never
    //    voids.
    //  - otherwise (the sweepers): keep the NaN segment IN the run so its finite endpoints bridge across
    //    the poisoned span (d2d-ref ink-only: a NaN control strokes start-cap↔remainder). A raw ±inf then
    //    stays in the run too (it has already voided through `ink_voids`).
    let raw_nan = |p: InkPoint| p[0].is_nan() || p[1].is_nan();
    let raw_inf = |p: InkPoint| p[0].is_infinite() || p[1].is_infinite();
    let mut runs: Vec<Vec<InkPoint>> = Vec::new();
    let mut cur: Vec<InkPoint> = Vec::new();
    let start = tp(ink.start);
    let mut p0raw = ink.start;
    let mut p0 = start;
    if !(break_at_nan && raw_nan(ink.start)) && !overflow(ink.start, &start) {
        cur.push(knot(start));
    }
    for s in &ink.segments {
        let (p1, p2, p3) = (tp(s.p1), tp(s.p2), tp(s.end));
        let has_nan = raw_nan(p0raw) || raw_nan(s.p1) || raw_nan(s.p2) || raw_nan(s.end);
        let has_inf = raw_inf(p0raw) || raw_inf(s.p1) || raw_inf(s.p2) || raw_inf(s.end);
        // Break at a raw NaN only for `ink_voids`, and only when the segment has no raw ±inf — an inf must
        // reach the flatten so its inf−inf = NaN stays in the run and voids (16561). The sweepers
        // (`break_at_nan` false) keep every NaN segment in its run and bridge across it.
        let poisoned = (break_at_nan && has_nan && !has_inf)
            || overflow(p0raw, &p0) || overflow(s.p1, &p1) || overflow(s.p2, &p2) || overflow(s.end, &p3);
        if !poisoned {
            if cur.is_empty() {
                cur.push(knot(p0));
            }
            flatten_ink_segment(p0, p1, p2, p3, tol, &mut cur);
        } else if !cur.is_empty() {
            runs.push(core::mem::take(&mut cur));
        }
        p0raw = s.end;
        p0 = p3;
    }
    if !cur.is_empty() {
        runs.push(cur);
    }
    runs
}

/// Native voids a whole DrawInk when a *kept* centre-curve segment cannot be tessellated to finite
/// device vertices: a raw NaN coord/radius (r9 812's ink8 has a NaN radius), or a flatten that
/// *produces* a NaN. The latter is a raw ±inf control point between finite neighbours (16561 seg0
/// `p2.x=inf`, r9 812 ink6 `p1.x=inf`): the forward-difference HFD computes opposite infinities whose
/// sum is NaN, so the swept outline carries NaN and the shared scanner rejects the lot (both render as
/// the bare clear). This is the ink analogue of `Geometry::workspace_bad_number`'s whole-figure
/// BAD_NUMBER void. Checking the RUNS — not the flat `flatten_centerline` — is what separates it from a
/// *clean* finite→±inf device overflow (16863/16350 start −3.4e38×dpi): that overflow is run-broken, so
/// the finite remainder (16863 seg1) still draws and never reaches the NaN check here.
/// A nib whose nib∘device 2×2 is the ZERO matrix (a point nib, e.g. a `[0,0,0,0]` nib_transform) draws
/// nothing: every nib offset `E·u` is 0, so the swept outline collapses to the zero-width centre-line
/// (zero area), which the as-geometry geometry sink fills to nothing (fuzz r15 20793 /
/// r14 19962 — our union tessellator instead made a spurious whole-target blob from the degenerate
/// point figures). This is rank-0 only: a rank-1 (line) nib like 18235's `[-0.391;4]` sweeps to a
/// non-zero-area ribbon and still draws, so gate on all-zero, not `det == 0`.
fn nib_voids(m: &Matrix, style: &InkStyle) -> bool {
    nib_matrix(m, style.nib_transform) == [0.0; 4]
}

/// The finite remainder of a per-primitive inf-radius ink (its overflowing nibs dropped) can FLOOD the
/// target through GEOMETRY alone, not just a huge start cap: r20 25049's `seg2.p1.x = -1280` control
/// sweeps the small-nib outline so its winding covers the whole 128×96 target (byte-exact once rendered).
/// Native commits such a flood; it voids only a remainder that does NOT cover the target. Render the
/// remainder's coverage the same way the per-primitive mask path does (`outline_polys` drops the
/// silhouette-overflowing and NaN nibs) and report whether every target pixel is covered. This is the
/// geometry analogue of the start-cap silhouette flood, and it leaves a partial remainder (24260's ~17px
/// cap, 20408/22213's poisoned stumps) voiding as before.
fn inf_radius_remainder_floods(ink: &Ink, m: &Matrix, style: &InkStyle, w: usize, h: usize) -> bool {
    if w == 0 || h == 0 {
        return false;
    }
    let figures = drop_nonfinite_figures(outline_polys(ink, m, style, 16.0, w, h));
    let Some(clipped) = crate::pixels::clip_huge(&figures) else {
        return false;
    };
    let mut seen = vec![false; w * h];
    let mut covered = 0usize;
    crate::pixels::rasterize_edges(&crate::pixels::figure_edges(&clipped), FillMode::Winding, Aa::PerPrimitive, w, h, |x, y, _cov| {
        let idx = y * w + x;
        if !seen[idx] {
            seen[idx] = true;
            covered += 1;
        }
    });
    covered == w * h
}

/// A control/knot RADIUS of ±inf makes that nib infinitely wide; whether the DrawInk voids depends on
/// the render path, which native's segment-list and quad-emit dumps pin exactly:
///  - ALIASED → the AsGeometry sink builds box/connector quads, and an inf radius (or a device-overflow
///    position) makes those quads non-finite (NaN); the NaN vertices poison the whole figure so the
///    scanner rejects it → VOID (r19 24751 square: one finite box then 79 NaN quads; r20 25434 round:
///    all-NaN round connectors; r15 20793).
///  - PER-PRIMITIVE → the nib sweep drops the inf-radius nib (its silhouette overflows) and commits the
///    rest, so the ink STROKES when a finite prefix FLOODS the target (r19 24700/24237/24275: an
///    inf-radius end under a huge nib floods off the finite start cap — byte-exact). It voids only when
///    the outline can't commit: no flooding prefix (24260's ~17px cap), a device-non-finite position
///    (20408 end.y=3.4e38 → device inf; a raw NaN/inf), or a finite-huge position whose flatten
///    second-difference `6·(p0−2p1+p2)` overflows f32 (22213 seg3 3.4e38: 12·3.4e38 = inf → NaN flatten).
///    A 1e9 position stays finite under ×12 and strokes (24700). A NaN RADIUS is resolved per-segment by
///    `resolve_nan_radius`; a non-finite START-CAP silhouette voids via `ink_extent_voids`.
fn ink_radius_voids(ink: &Ink, m: &Matrix, style: &InkStyle, w: usize, h: usize, aliased: bool) -> bool {
    let has_inf_radius = core::iter::once(ink.start)
        .chain(ink.segments.iter().flat_map(|s| [s.p1, s.p2, s.end]))
        .any(|p| p[2].is_infinite());
    if !has_inf_radius {
        return false;
    }
    if aliased {
        return true; // AsGeometry NaN quads poison the figure
    }
    // Per-primitive: native drops the overflowing nibs and COMMITS the finite remainder. If that
    // remainder's geometry floods the whole target (r20 25049's -1280 control winding), native commits
    // it — even when the start cap alone does not flood. Render the remainder and override the void when
    // it covers the target. A poisoned KNOT (start or a segment end whose DEVICE position is non-finite or
    // overflows f32, r15 20408 end.y=3.4e38→inf) can't close the outline, so native voids it regardless of
    // what the remainder would cover — the flood override is gated to finite-knot inks (25049's overflow is
    // a CONTROL, which only drops its own segment). A partial remainder (24260's ~17px cap) falls through.
    let arith_limit = f32::MAX / 12.0;
    let knot_poison = core::iter::once(ink.start)
        .chain(ink.segments.iter().map(|s| s.end))
        .any(|k| {
            let d = apply(m, [k[0], k[1]]);
            !(d[0].is_finite() && d[1].is_finite()) || d[0].abs() >= arith_limit || d[1].abs() >= arith_limit
        });
    if !knot_poison && inf_radius_remainder_floods(ink, m, style, w, h) {
        return false;
    }
    // Void unless a finite prefix floods the target (the start cap's swept silhouette
    // covering it).
    let sil = ink.start[2].abs() * factor_helper(nib_matrix(m, style.nib_transform)) * nib_extents(style.nib_shape);
    let c0 = apply(m, [ink.start[0], ink.start[1]]);
    let floods = sil.is_finite()
        && c0[0] - sil <= 0.0
        && c0[0] + sil >= w as f32
        && c0[1] - sil <= 0.0
        && c0[1] + sil >= h as f32;
    if !floods {
        return true;
    }
    // Even with a flood, a device-non-finite position, or a finite-huge position whose flatten
    // second-difference `6·(…−2·p…)` overflows f32 (|device| ≥ f32::MAX/12), poisons the stepped stream
    // beyond the droppable inf-radius nibs → void.
    let arith_limit = f32::MAX / 12.0;
    core::iter::once(ink.start)
        .chain(ink.segments.iter().flat_map(|s| [s.p1, s.p2, s.end]))
        .any(|p| {
            let d = apply(m, [p[0], p[1]]);
            !(d[0].is_finite() && d[1].is_finite()) || d[0].abs() >= arith_limit || d[1].abs() >= arith_limit
        })
}

/// A DrawInk voids when the swept nib's FIRST disc — the start cap — has a non-finite device silhouette.
/// The nib sweep emits nib geometry in path order; a silhouette vertex `centre + E·(radius·dir)` whose
/// `|radius|·FactorHelper` overflows f32 is ±inf, and when that happens at the START point nothing valid
/// is ever emitted, so native's tessellation drops the whole primitive and renders nothing (fuzz r14
/// 19622: a start radius −3.4e38 under a FactorHelper 1.73 device matrix → inf start cap; we had clamped
/// the huge nib to the target and washed it). A huge radius at a LATER control point does NOT void here:
/// the finite leading discs are already committed (fuzz recap 12207: a start radius 400 under a 1e6 nib
/// fills the target before the 3.4e38 at `seg2.p2` is reached, so native draws it). `|·|` keeps −inf
/// infinite; a NaN start radius is handled upstream by [`ink_voids`].
fn ink_extent_voids(ink: &Ink, m: &Matrix, style: &InkStyle) -> bool {
    let fh = factor_helper(nib_matrix(m, style.nib_transform));
    let ne = nib_extents(style.nib_shape);
    // The start cap always closes the stroke, so a silhouette that overflows f32 voids any ink.
    if !(ink.start[2].abs() * fh * ne).is_finite() {
        return true;
    }
    // A NaN-POSITION ink collapses each NaN-control segment to its finite {P0,P3} knots (native
    // the segment flattener emits only the knots). A stroked knot whose DEVICE silhouette reaches DrawInk's
    // 26.6 edge-list limit (2^25) can't close the outline and voids (24186 end radius 3.4e38 ≫ 2^25) —
    // where a finite-control segment's interpolated radii stay bounded and flood/stroke (26516's 1e6 nib ×
    // r2.43 ≈ 2.4e6 stays under 2^25 and strokes).
    if ink_has_nan_pos(ink) {
        let limit = (1u32 << 25) as f32;
        // A NaN-collapsed outline can't close when a stroked knot can't be represented in DrawInk's 26.6
        // edge list (2^25) — EITHER because its swept-nib RADIUS silhouette reaches the limit (24186 end
        // radius 3.4e38), OR because its DEVICE POSITION itself does (r22 27143 seg1.end.y = 1e9 under a
        // round nib: native voids this NaN-position ink where the removed blanket NaN-pos void used to;
        // 26516's small knots stay under 2^25 and stroke). The position check is gated to NaN-position
        // inks, so a non-NaN ink that floods off a prior huge nib (26737 seg0.end.x = 1e9) keeps its path.
        let knot_over = |c: InkPoint| {
            let sil = c[2].abs() * fh * ne;
            if sil.is_finite() && sil >= limit {
                return true;
            }
            let d = apply(m, [c[0], c[1]]);
            let pos = d[0].abs().max(d[1].abs());
            pos.is_finite() && pos >= limit
        };
        if core::iter::once(ink.start).chain(ink.segments.iter().map(|s| s.end)).any(knot_over) {
            return true;
        }
    }
    false
}

/// A NaN-POSITION control collapses its segment to the finite knots, but native's segment flattener also
/// inserts a point wherever the `|Bernstein radius|` crosses the radius threshold `rthr = 0.5/scale`
/// (`EstimatePointAtRadiusThreshold`) — and that inserted point sits at the poisoned (NaN) position, so a
/// NaN-control segment whose radius crosses `rthr` emits a NaN vertex and the whole fill voids (r15 20473
/// radii 0.069↔1.42 straddle 0.4; r17 22561 0.086↔0.83 straddle 0.5), while one whose radius stays on one
/// side strokes the clean {P0,P3} (r21 26516's 1e6 nib pushes `rthr` to 5e-7, uncrossable).
///
/// The crossing test is native's own, run at the ACTUAL flattened points, not a dense resample: a NaN
/// control's `TryDoubleTheStep` never blocks (its `tol < |2nd-diff|` guards are all false on NaN), so it
/// coarsens the subdivision all the way back to ONE step and the segment flattener emits exactly {P0,P3}
/// (VM the segment-list builder: 26516 n=2, 20473/22561 n=3 with the inserted NaN). So native compares only the
/// two knot radii `|r0|`,`|r3|` (= `|Bernstein(0)|`,`|Bernstein(1)|`) against `rthr`: `(|r3|-rthr)·(|r0|
/// -rthr) < 0`. A mid-segment dip the control radii cause is never sampled, so it must not void here.
/// `scale` is the nib∘device spectral norm (`factor_helper`).
fn ink_radius_threshold_voids(ink: &Ink, m: &Matrix, style: &InkStyle) -> bool {
    let fh = factor_helper(nib_matrix(m, style.nib_transform));
    if !(fh > 0.0) {
        return false; // a zero/degenerate nib has no silhouette; nib_voids handles it
    }
    let rthr = 0.5 / fh;
    let ctrl_nan = |s: &InkBezierSegment| s.p1[0].is_nan() || s.p1[1].is_nan() || s.p2[0].is_nan() || s.p2[1].is_nan();
    let mut p0 = ink.start;
    for s in &ink.segments {
        if ctrl_nan(s) {
            // Bernstein at t=0 and t=1 reduce to |r0| and |r3|, via radius_cubic for native's op order.
            let r = [p0[2], s.p1[2], s.p2[2], s.end[2]];
            if (radius_cubic(1.0, r) - rthr) * (radius_cubic(0.0, r) - rthr) < 0.0 {
                return true;
            }
        }
        p0 = s.end;
    }
    false
}

fn ink_voids(ink: &Ink, m: &Matrix, nib_shape: u32) -> bool {
    // The nib is swept to a CLOSED outline that joins consecutive flattened knots, so where a non-finite
    // knot sits decides whether the outline can still close with finite vertices:
    //  - an INTERIOR non-finite knot is bracketed by the finite knots on either side; the flatten bridges
    //    the span and the outline stays finite, so it strokes (a NaN control, 21708 seg0.p2 / 23365 seg3;
    //    a poisoned MIDDLE segment between two finite ones, 16559 s1);
    //  - a TERMINAL non-finite knot — the start cap (`ink.start`) or the last end cap — has no finite
    //    neighbour to bridge from, so the closing vertex is non-finite and the whole fill is rejected
    //    (19055/21292, poisoned LAST segment).
    // So a NaN POSITION strokes only when BOTH terminals can close; failing either voids:
    //  1. at least one STROKEABLE segment exists — its start knot `p0` (`ink.start` for the first segment,
    //     the previous segment's end otherwise) and its `end` finite, and its CONTROLS `p1`/`p2` carry no
    //     raw ±inf — so there is a finite outline to sweep at all. A NaN in a CONTROL is NOT disqualifying:
    //     `flatten_ink_segment` degenerates it to the finite endpoint disc, so the segment still strokes
    //     (r21 26516's lone NaN-control-finite-knot segment draws that disc, native `want` [80,52,39,130]) —
    //     the same treatment `last_ok` below already gives a bridged NaN control. An inf control poisons the
    //     flatten to NaN and a non-finite KNOT can't seed a disc, so both still fail.
    //  2. the LAST segment closes on finite ON-CURVE knots — its start knot and its end (a NaN in its
    //     CONTROLS is bridged by the flatten) — so the end cap is finite (21292/19816/22597, whose last
    //     segment starts from or ends on a NaN knot, fail this; 18966/23365 pass).
    if ink_has_nan_pos(ink) {
        let mut p0 = ink.start;
        let has_clean = ink.segments.iter().any(|s| {
            let ok = [p0, s.end].iter().all(|p| p[0].is_finite() && p[1].is_finite())
                && [s.p1[0], s.p1[1], s.p2[0], s.p2[1]].iter().all(|v| !v.is_infinite());
            p0 = s.end;
            ok
        });
        let last_ok = match ink.segments.last() {
            Some(last) => {
                let n = ink.segments.len();
                let p0l = if n >= 2 { ink.segments[n - 2].end } else { ink.start };
                // Terminals finite, and the last segment's CONTROLS carry no raw ±inf: a NaN control there
                // is bridged (18966 seg3.p1) but an inf poisons the closing stroke and voids (19055 seg2.p2
                // = −inf). Mid-segment inf controls are bridged by the clean segments after them.
                [p0l, last.end].iter().all(|p| p[0].is_finite() && p[1].is_finite())
                    && [last.p1[0], last.p1[1], last.p2[0], last.p2[1]].iter().all(|v| !v.is_infinite())
            }
            None => false,
        };
        if !has_clean || !last_ok {
            return true;
        }
    }
    // A NaN RADIUS left after `resolve_nan_radius` voids (it can't seed a disc). A NaN CONTROL POSITION
    // does NOT void here — the run-bridge sweep strokes across it. What still voids is a run that *contains*
    // a NaN after that break: a raw ±inf CONTROL between finite neighbours, whose HFD flatten computes
    // inf−inf = NaN and poisons the whole swept outline (16561 seg0 p2.x=inf, r9 812).
    let pts = core::iter::once(ink.start).chain(ink.segments.iter().flat_map(|s| [s.p1, s.p2, s.end]));
    if pts.map(|p| p[2]).any(f32::is_nan) {
        return true;
    }
    // A run still carries a NaN after the position/radius checks above. If every POSITION is finite the NaN
    // is an INTERNAL float forward-difference overflow (a ~3.4e38 control: `6·(p1−2p2+p3)=±inf`, then
    // `inf−inf=NaN`); native does NOT void. It flattens PER SEGMENT (the segment flattener,
    // `CBezierFlattener<float>`) and its sink (its segment sink) emits the start cap in
    // the FIRST segment's first point batch, dropping only the NaN-carrying batches and never committing
    // past the first poisoned segment — so the ink strokes iff its FIRST segment flattens entirely finite
    // (r21 26737: seg0 finite, seg2's 3.4e38→NaN, floods off the start cap; r22 27260 seg0 overflows,
    // voids; r21 26270 whose first kept segment overflows, voids). A raw ±inf POSITION still voids — its
    // connected outline can't close (16561; r22 27461's inf knot, which `ink_has_nan_pos` skips).
    let has_nan =
        flatten_centerline_runs(ink, m, nib_shape, true).iter().flatten().any(|p| p.iter().any(|v| v.is_nan()));
    if !has_nan {
        return false;
    }
    let inf_pos = core::iter::once(ink.start)
        .chain(ink.segments.iter().flat_map(|s| [s.p1, s.p2, s.end]))
        .any(|p| p[0].is_infinite() || p[1].is_infinite());
    inf_pos || !first_segment_finite(ink, m, nib_shape)
}

fn first_segment_finite(ink: &Ink, m: &Matrix, nib_shape: u32) -> bool {
    let Some(s) = ink.segments.first() else {
        return false;
    };
    let tol = ink_flattening_tolerance(nib_shape);
    let tp = |p: InkPoint| -> InkPoint {
        let d = apply(m, [p[0], p[1]]);
        [d[0], d[1], p[2]]
    };
    let p0 = tp(ink.start);
    let (p1, p2, p3) = (tp(s.p1), tp(s.p2), tp(s.end));
    let mut seg = Vec::new();
    flatten_ink_segment(p0, p1, p2, p3, tol, &mut seg);
    p0[0].is_finite() && p0[1].is_finite() && seg.iter().all(|q| q[0].is_finite() && q[1].is_finite())
}

fn pre_nan_radius_floods(ink: &Ink, m: &Matrix, style: &InkStyle, w: usize, h: usize, aliased: bool) -> Option<usize> {
    // Aliased DrawInk takes the AsGeometry sink, where a mid-stroke NaN radius makes the swept box/quad
    // vertices NaN, poisoning the whole figure -> the scanner rejects it -> VOID (r18 23927 ink12: an
    // aliased NaN-radius ink with a 1e9/3.4e38 position that WOULD flood under the per-primitive ink mask path
    // is voided here instead). The flood rescue is the per-primitive mask-path behaviour only — native commits
    // the flood there because the mask path drops the NaN nibs (all the validation seeds 20859/18861/22939/
    // 24260/19962 are per-primitive). So no flood rescue when aliased.
    if aliased {
        return None;
    }
    // A mid-stroke NaN radius voids the connected ink outline (16091). It survives and strokes only when a
    // SEGMENT before the first NaN-radius segment floods the target: its control radius is FINITE, does not
    // overflow the f32 silhouette (that DROPS the segment, it does not flood), and reaches the clamp cap
    // |r| >= r_cap; AND the first segment flattens entirely finite, so the flood is real and on-target
    // (r15 20859: seg0.p1 = 1e9 floods). A LEADING NaN segment (no finite segment before it, r13 18812),
    // a merely-large START CAP that is not a segment (r15 20168), an overflowing control that drops
    // (r17 22939), or a first segment that does not flatten finite all void as native does.
    let _ = (w, h);
    let seg_nan = |s: &InkBezierSegment| s.p1[2].is_nan() || s.p2[2].is_nan() || s.end[2].is_nan();
    let first_nan = ink.segments.iter().position(seg_nan)?;
    if first_nan == 0 || !first_segment_finite(ink, m, style.nib_shape) {
        return None;
    }
    let e = nib_matrix(m, style.nib_transform);
    let nib_ext = nib_extents(style.nib_shape);
    let fh = factor_helper(e);
    let sil_overflow = |r: f32| !(r.abs() * fh * nib_ext).is_finite();
    // 26.6 device-coordinate overflow: a swept silhouette vertex whose device magnitude passes 2^25 wraps
    // DrawInk's WARP 26.6 edge list into a huge flood that native's sink commits BEFORE the NaN poisons the
    // outline (r15 20859: seg0.p1 = 1e9 reaches ~1e9 device; r13 18861: a ~4e6 transform pushes seg0's
    // device positions past 2^25). A moderate nib that stays inside 2^25 is connected to the NaN and voids
    // (r19 24260: a 400 radius reaches only ~1440 device → native leaves the whole target empty). An
    // overflowing control (sil_overflow) DROPS its segment, it does not flood.
    let limit = (1u32 << 25) as f32;
    let floods = core::iter::once(ink.start)
        .chain(ink.segments[..first_nan].iter().flat_map(|s| [s.p1, s.p2, s.end]))
        .any(|c| {
            let r = c[2];
            if !r.is_finite() || sil_overflow(r) {
                return false;
            }
            let d = apply(m, [c[0], c[1]]);
            let extent = d[0].abs().max(d[1].abs()) + r.abs() * fh * nib_ext;
            extent.is_finite() && extent >= limit
        });
    floods.then_some(first_nan)
}

/// Native's aliased AsGeometry sink DROPS the swept-nib box/connector quads at a knot whose DEVICE
/// position is non-finite because a FINITE ink position overflowed f32 under the device transform
/// (3.4e38 × 1.25 dpi → inf): the quad emit is never called for a non-finite quad, so those quads cannot
/// cancel the fill. The finite swept-nib prefix then floods UNCANCELLED — where a merely finite-huge
/// knot (no overflow) instead emits a NaN quad that poisons/clips the fill (r15 20064 dpi120: seg2.end
/// 3.4e38 overflows to inf → the seg0/seg1 −40 nib floods 931; 23927 dpi96: 3.4e38 stays finite →
/// NaN quad → voids). Gated to aliased, a finite start cap (an inf START can't anchor the outline —
/// the start-cap void handles it) and a FINITE overflowing radius (a non-finite radius poisons the
/// quad itself → void). The render drops the non-finite figures and tessellates the finite union, so
/// an ink whose finite prefix covers nothing still voids naturally.
fn inf_pos_knot_floods(ink: &Ink, m: &Matrix, aliased: bool) -> bool {
    if !aliased {
        return false;
    }
    // Only a raw-finite-POSITION ink takes this overflow flood: a NaN or raw ±inf position anywhere routes
    // through native's NaN-bridge / void handling instead, where the overflow knot does NOT expose a prefix
    // flood (fuzz recap9 1083 ink11: a NaN seg0.p1 and raw-inf seg3 controls alongside a 3.4e38→inf knot —
    // our fix spuriously flooded it off that knot; native renders it through the NaN-position path).
    let raw_bad = core::iter::once(ink.start)
        .chain(ink.segments.iter().flat_map(|s| [s.p1, s.p2, s.end]))
        .any(|p| !p[0].is_finite() || !p[1].is_finite());
    if raw_bad {
        return false;
    }
    ink.segments.iter().map(|s| s.end).any(|k| {
        let d = apply(m, [k[0], k[1]]);
        k[2].is_finite() && !(d[0].is_finite() && d[1].is_finite())
    })
}

/// Forward-difference flatten of one cubic (positions excluding `p0`, including the endpoint `p3`),
/// each carrying `radius = |cubic(t)|`, with no radius-threshold points (see [`flatten_ink_batches`]).
fn flatten_ink_segment(p0: InkPoint, p1: InkPoint, p2: InkPoint, p3: InkPoint, tol: f32, out: &mut Vec<InkPoint>) {
    // ponytail: the instanced-path callers keep the flattener's earlier NaN handling (a NaN-ignoring max,
    // `<=` tests), which their void rules were measured against, until that path is ported.
    flatten_ink_batches(false, false, p0, [p1, p2, p3], tol, f32::NAN, &mut |b, _| out.extend_from_slice(&b[1..]));
}

/// The per-segment ink flattener: the float Bézier flattener over one segment's device positions, emitting batches
/// of at most nine steps (plus the end point once the steps run out), each batch led by the previous
/// batch's last point. A point's radius is `|cubic(t)|` of the signed control radii. Where the radius
/// crosses `thr` (`0.5/factorHelper`) between two steps, up to three points per batch are inserted at the
/// interpolated crossing ([`radius_threshold_point`]); a NaN `thr` inserts none.
#[allow(clippy::too_many_arguments)]
fn flatten_ink_batches(native: bool, first: bool, p0: InkPoint, c: [InkPoint; 3], tol: f32, thr: f32, emit: &mut dyn FnMut(&[InkPoint], bool)) {
    let r = [p0[2], c[0][2], c[1][2], c[2][2]];
    let xy = |p: InkPoint| [p[0], p[1]];
    let mut f = InkFlat::new(native, tol, xy(p0), xy(c[0]), xy(c[1]), xy(c[2]));
    let mut start = [p0[0], p0[1], p0[2].abs()];
    let (mut prev_t, mut prev_d) = (0.0f32, start[2] - thr);
    let mut first = first;
    let mut steps: Vec<([f32; 2], f32)> = Vec::with_capacity(10);
    let mut out: Vec<InkPoint> = Vec::with_capacity(14);
    loop {
        steps.clear();
        let mut more = true;
        if f.csteps > 1 {
            while steps.len() < 9 {
                f.step();
                steps.push((f.cur, f.t));
                if f.within(f.e1) || f.stepsize as f64 <= 0.001 {
                    while f.try_double() {}
                } else {
                    f.halve();
                }
                if f.csteps <= 1 {
                    if steps.len() < 9 {
                        steps.push((f.b3, 1.0));
                        more = false;
                    }
                    break;
                }
            }
        } else {
            steps.push((f.b3, 1.0));
            more = false;
        }
        out.clear();
        out.push(start);
        let mut inserted = 0;
        for &(q, t) in &steps {
            let rr = radius_cubic(t, r);
            let d = rr - thr;
            if d * prev_d < 0.0 && inserted < 3 {
                out.push(radius_threshold_point(prev_d, d, prev_t, t, p0, &c));
                inserted += 1;
            }
            out.push([q[0], q[1], rr]);
            (prev_t, prev_d) = (t, d);
        }
        emit(&out, first);
        first = false;
        start = out[out.len() - 1];
        if !more {
            return;
        }
    }
}

/// The radius-threshold point: the segment point at the parameter where the radius, linear between
/// the two steps' threshold distances `d0` (at `t0`) and `d1` (at `t1`), meets the threshold.
fn radius_threshold_point(d0: f32, d1: f32, t0: f32, t1: f32, p0: InkPoint, c: &[InkPoint; 3]) -> InkPoint {
    let t = t0 - ((t1 - t0) * d0) / (d1 - d0);
    let (mt, t3) = (1.0 - t, (t * t) * t);
    let m3 = mt * 3.0;
    let (b1, b2, b0) = ((m3 * mt) * t, (m3 * t) * t, (mt * mt) * mt);
    [
        (((b0 * p0[0]) + (b1 * c[0][0])) + (b2 * c[1][0])) + (t3 * c[2][0]),
        (((b1 * c[0][1]) + (b0 * p0[1])) + (b2 * c[1][1])) + (t3 * c[2][1]),
        ((((b1 * c[0][2]) + (b0 * p0[2])) + (b2 * c[1][2])) + (t3 * c[2][2])).abs(),
    ]
}

/// `radius(t) = (1−t)³r0 + 3(1−t)²t·r1 + 3(1−t)t²·r2 + t³r3`, native op order, `|·|`.
fn radius_cubic(t: f32, r: [f32; 4]) -> f32 {
    let mt = 1.0 - t;
    (((mt * mt) * mt) * r[0]
        + ((mt * (mt * 3.0)) * t) * r[1]
        + (((mt * 3.0) * t) * t) * r[2]
        + ((t * t) * t) * r[3])
        .abs()
}

// The float Bézier flattener (adaptive forward-differencing), the arithmetic of
// `crate::geometry::flatten::Flattener`, plus a parameter `t` accumulated per step for the radius.
struct InkFlat {
    b3: [f32; 2],
    cur: [f32; 2],
    e0: [f32; 2],
    e1: [f32; 2],
    e2: [f32; 2],
    err_bound: f32,
    fuzz: f32,
    csteps: i32,
    stepsize: f32,
    t: f32,
    native: bool,
}

/// SSE `maxss`/`minss`: the first operand when it compares greater (less), else the second, so a NaN on
/// either side gives the second.
fn maxss(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

fn minss(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

impl InkFlat {
    fn new(native: bool, tol: f32, p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], p3: [f32; 2]) -> Self {
        let err_bound = tol.max(0.0) * 6.0;
        let mut f = InkFlat {
            b3: p3,
            cur: p0,
            e0: [p3[0] - p0[0], p3[1] - p0[1]],
            e1: [6.0 * (p1[0] - 2.0 * p2[0] + p3[0]), 6.0 * (p1[1] - 2.0 * p2[1] + p3[1])],
            e2: [6.0 * (p0[0] - 2.0 * p1[0] + p2[0]), 6.0 * (p0[1] - 2.0 * p1[1] + p2[1])],
            err_bound,
            fuzz: err_bound * 0.25,
            csteps: 1,
            stepsize: 1.0,
            t: 0.0,
            native,
        };
        while !(f.within(f.e1) && f.within(f.e2)) && f.stepsize as f64 > 0.001 {
            f.halve();
        }
        f
    }

    /// The step-size test: the larger component magnitude (`maxss`, so a NaN in `y` reads as NaN) does not
    /// exceed the error bound. A NaN passes.
    fn within(&self, e: [f32; 2]) -> bool {
        if self.native { !(maxss(e[0].abs(), e[1].abs()) > self.err_bound) } else { e[0].abs().max(e[1].abs()) <= self.err_bound }
    }

    fn halve(&mut self) {
        let ne1 = [(self.e1[0] + self.e2[0]) * 0.125, (self.e1[1] + self.e2[1]) * 0.125];
        self.e0 = [(self.e0[0] - ne1[0]) * 0.5, (self.e0[1] - ne1[1]) * 0.5];
        self.e2 = [self.e2[0] * 0.25, self.e2[1] * 0.25];
        self.e1 = ne1;
        self.csteps *= 2;
        self.stepsize = ((self.stepsize as f64) * 0.5) as f32;
    }

    fn step(&mut self) {
        self.cur = [self.cur[0] + self.e0[0], self.cur[1] + self.e0[1]];
        let o1 = self.e1;
        self.e0 = [self.e0[0] + self.e1[0], self.e0[1] + self.e1[1]];
        self.e1 = [2.0 * o1[0] - self.e2[0], 2.0 * o1[1] - self.e2[1]];
        self.e2 = o1;
        self.csteps -= 1;
        self.t += self.stepsize;
    }

    /// Double the step when the step count is even and every second difference stays within the fuzz;
    /// a NaN difference never doubles.
    fn try_double(&mut self) -> bool {
        if self.csteps & 1 != 0 {
            return false;
        }
        let d = [2.0 * self.e1[0] - self.e2[0], 2.0 * self.e1[1] - self.e2[1]];
        if [self.e2[0], self.e2[1], d[0], d[1]].iter().any(|v| if self.native { !(self.fuzz >= v.abs()) } else { self.fuzz < v.abs() }) {
            return false;
        }
        self.e0 = [2.0 * self.e0[0] + self.e1[0], 2.0 * self.e0[1] + self.e1[1]];
        self.e2 = [self.e2[0] * 4.0, self.e2[1] * 4.0];
        self.e1 = [d[0] * 4.0, d[1] * 4.0];
        self.stepsize *= 2.0;
        self.csteps /= 2;
        true
    }
}

// --- nib outlines ----------------------------------------------------------------------------

const KAPPA: f32 = 0.552_284_8; // circle cubic control factor, native `__real@3f0d6289`

/// `AddEllipse`: the 13 control points of a 4-arc circle of radius `r`, offsets shaped by the 2×2
/// `e` and centred at `c`. Returns the figure start (`P0`).
fn ellipse_start(c: [f32; 2], r: f32, e: &[f32; 4]) -> [f32; 2] {
    off(c, e, [-r, 0.0])
}

fn ellipse_prims(c: [f32; 2], r: f32, e: &[f32; 4]) -> Vec<Prim> {
    let k = r * KAPPA;
    // Native point order (circle starting at (−r,0), through (0,−r),(r,0),(0,r)).
    let pat: [[f32; 2]; 13] = [
        [-r, 0.0], [-r, -k], [-k, -r], [0.0, -r], [k, -r], [r, -k], [r, 0.0], [r, k], [k, r], [0.0, r],
        [-k, r], [-r, k], [-r, 0.0],
    ];
    let p: Vec<[f32; 2]> = pat.iter().map(|&o| off(c, e, o)).collect();
    vec![
        Prim::Cubic(p[0], p[1], p[2], p[3]),
        Prim::Cubic(p[3], p[4], p[5], p[6]),
        Prim::Cubic(p[6], p[7], p[8], p[9]),
        Prim::Cubic(p[9], p[10], p[11], p[12]),
    ]
}

/// A square nib's 4 box corners, order 0=(−r,−r) 1=(r,−r) 2=(r,r) 3=(−r,r), shaped by `e` at `c`
/// (`SquareInkToGeometrySinkAdapter`'s per-point box).
fn box_corners(c: [f32; 2], r: f32, e: &[f32; 4]) -> [[f32; 2]; 4] {
    [off(c, e, [-r, -r]), off(c, e, [r, -r]), off(c, e, [r, r]), off(c, e, [-r, r])]
}

/// `LookupSquareQuadCorners`: the two box-corner indices for the connector between boxes `(p0,r0)`
/// and `(p1,r1)` (axis-aligned in the nib-normalised space), from the native 16-entry table; `None`
/// (`0xFFFF`) when the boxes need no connector. The connector is `[prevBox[a], prevBox[b],
/// curBox[b], curBox[a]]`.
fn square_corners(p0: [f32; 2], r0: f32, p1: [f32; 2], r1: f32) -> Option<(usize, usize)> {
    // native the corner-index table: 16 (cornerA, cornerB) pairs indexed by the 4-bit extent code.
    const TABLE: [(u16, u16); 16] = [
        (1, 3), (1, 0), (2, 3), (2, 0), (0, 3), (0xFFFF, 0xFFFF), (0xFFFF, 0xFFFF), (2, 1),
        (1, 2), (0xFFFF, 0xFFFF), (0xFFFF, 0xFFFF), (3, 0), (0, 2), (3, 2), (0, 1), (3, 1),
    ];
    let bit3 = if p1[1] + r1 <= p0[1] + r0 { 0 } else { 8 };
    let bit1 = if p1[0] + r1 <= p0[0] + r0 { 0 } else { 2 };
    let bit0 = if p1[0] - r1 > p0[0] - r0 { 1 } else { 0 };
    let bit2 = if p1[1] - r1 <= p0[1] - r0 { 0 } else { 4 };
    let (a, b) = TABLE[(bit3 + bit1 + bit0 + bit2) as usize];
    if a == 0xFFFF {
        None
    } else {
        Some((a as usize, b as usize))
    }
}

/// `ConstructRoundQuadVertices`: the outer tangent trapezoid between circle `(p0,r0)` and `(p1,r1)`
/// in the (nib-normalised) space the caller works in; `None` when one circle contains the other.
fn round_quad(p0: [f32; 2], r0: f32, p1: [f32; 2], r1: f32) -> Option<[[f32; 2]; 4]> {
    let (dx, dy) = (p1[0] - p0[0], p1[1] - p0[1]);
    let len = (dy * dy + dx * dx).sqrt();
    let mut s = 1.0f32;
    let mut dr = r1 - r0;
    if dr < 0.0 {
        s = -1.0;
        dr = -dr;
    }
    if dr >= len {
        return None;
    }
    let cos = dr / len;
    let mut sin = 1.0 - cos * cos;
    if sin > 0.0 {
        sin = sin.sqrt();
    }
    let (ux, uy) = (dx / len, dy / len);
    let a = (uy * s) * cos;
    let b = (ux * s) * cos;
    let (y0, y1) = (p0[1] - a * r0, p1[1] - a * r1);
    let (x0, x1) = (p0[0] - b * r0, p1[0] - b * r1);
    let nx = ux * sin;
    let ny = (-uy) * sin;
    Some([
        [ny * r0 + x0, nx * r0 + y0],
        [x0 - ny * r0, y0 - nx * r0],
        [x1 - ny * r1, y1 - nx * r1],
        [x1 + ny * r1, y1 + nx * r1],
    ])
}

// --- as-geometry outline shape ---------------------------------------------------------------

/// One figure of the ink outline shape: a nib ellipse (four closed cubics about a device centre with a
/// radius) or a closed four-point quad.
enum NibFig {
    Ellipse([f32; 2], f32),
    Quad([[f32; 2]; 4]),
}

/// The ink's metrics: the bounds of its start and control points, and its largest |radius|. Each bound
/// takes a point unless the point is strictly beyond it the other way, so a NaN coordinate lasts only
/// until the next point replaces it.
fn ink_metrics(ink: &Ink) -> ([f32; 4], f32) {
    let s = ink.start;
    let (mut b, mut r) = ([s[0], s[1], s[0], s[1]], s[2].abs());
    let lo = |c: f32, v: f32| if v > c { c } else { v };
    let hi = |c: f32, v: f32| if c > v { c } else { v };
    for g in &ink.segments {
        r = maxss(r, maxss(g.p1[2].abs(), maxss(g.p2[2].abs(), g.end[2].abs())));
        for p in [g.p1, g.p2, g.end] {
            b = [lo(b[0], p[0]), lo(b[1], p[1]), hi(b[2], p[0]), hi(b[3], p[1])];
        }
    }
    (b, r)
}

/// The rect's device bounds: an ordered rect's four corners under `m`; a NaN corner coordinate (an infinite
/// edge times a zero matrix term) makes both bounds on its axis NaN.
fn transform_2d_bounds(m: &Matrix, b: [f32; 4]) -> [f32; 4] {
    if b[0] > b[2] || b[1] > b[3] {
        return b;
    }
    let c = [[b[0], b[1]], [b[2], b[1]], [b[0], b[3]], [b[2], b[3]]].map(|p| [(m[2] * p[1] + m[0] * p[0]) + m[4], (m[3] * p[1] + m[1] * p[0]) + m[5]]);
    let axis = |k: usize| {
        if c.iter().any(|p| p[k].is_nan()) {
            return (f32::NAN, f32::NAN);
        }
        c.iter().fold((c[0][k], c[0][k]), |(lo, hi), p| (minss(p[k], lo), maxss(p[k], hi)))
    };
    let ((l, r), (t, b)) = (axis(0), axis(1));
    [l, t, r, b]
}

/// `DrawInk`'s early exit: nothing draws unless the ink's device bounds, grown by the nib reach
/// `nibExtents·(max|r|·factorHelper)`, overlap the clip. A NaN in the grown bounds fails the overlap.
fn ink_misses_clip(ink: &Ink, m: &Matrix, style: &InkStyle, clip: [f32; 4]) -> bool {
    let (b, r) = ink_metrics(ink);
    let ext = nib_extents(style.nib_shape) * (r * factor_helper(nib_matrix(m, style.nib_transform)));
    let d = transform_2d_bounds(m, b);
    if d[0] > d[2] || d[1] > d[3] {
        return true;
    }
    let (l, t, r, b) = (d[0] - ext, d[1] - ext, d[2] + ext, d[3] + ext);
    clip[0] >= clip[2] || clip[1] >= clip[3] || l >= r || t >= b || !(clip[2] > l) || !(r > clip[0]) || !(clip[3] > t) || !(b > clip[1])
}

/// The ink's segment stream with the clip rect: each segment's device control points are bounded
/// (`minss`/`maxss` seeded by its start, so a NaN control drops out of the bounds and a NaN start keeps
/// them NaN), grown by its largest |radius| times `nibExtents·factor`. A segment whose grown bounds miss
/// the clip, or are NaN, is skipped; one not inside it flattens through [`cull_pairs`] against the clip
/// grown the same way. An ink with no segments sends its start twice, the first at radius 0.
#[allow(clippy::too_many_arguments)]
fn send_ink_segments(ink: &Ink, m: &Matrix, factor: f32, clip: Option<[f32; 4]>, tol: f32, nib_shape: u32, sink: &mut dyn FnMut(&[InkPoint], bool)) {
    let tp = |p: InkPoint| -> InkPoint { let d = apply(m, [p[0], p[1]]); [d[0], d[1], p[2]] };
    let mut p0 = tp(ink.start);
    if ink.segments.is_empty() {
        sink(&[[p0[0], p0[1], 0.0], [p0[0], p0[1], p0[2].abs()]], true);
        return;
    }
    let ext = nib_extents(nib_shape) * factor;
    let thr = 0.5 / factor;
    for (i, g) in ink.segments.iter().enumerate() {
        let c = [tp(g.p1), tp(g.p2), tp(g.end)];
        let (mut b, mut mr) = ([p0[0], p0[1], p0[0], p0[1]], p0[2].abs());
        for q in &c {
            b = [minss(q[0], b[0]), minss(q[1], b[1]), maxss(q[0], b[2]), maxss(q[1], b[3])];
            mr = maxss(q[2].abs(), mr);
        }
        let infl = mr * ext;
        let (l, t, r, bt) = (b[0] - infl, b[1] - infl, b[2] + infl, b[3] + infl);
        let Some([cl, ct, cr, cb]) = clip else {
            flatten_ink_batches(true, i == 0, p0, c, tol, thr, sink);
            p0 = c[2];
            continue;
        };
        let drawn = !(l >= r) && !(t >= bt) && !(cl >= cr) && !(ct >= cb) && r > cl && cr > l && bt > ct && cb > t;
        if drawn {
            if !(l >= cl) || !(t >= ct) || !(cr >= r) || !(cb >= bt) {
                let rect = [cl - infl, ct - infl, cr + infl, cb + infl];
                flatten_ink_batches(true, i == 0, p0, c, tol, thr, &mut |pts, first| cull_pairs(pts, first, rect, sink));
            } else {
                flatten_ink_batches(true, i == 0, p0, c, tol, thr, sink);
            }
        }
        p0 = c[2];
    }
}

/// The segment culling: keeps each consecutive pair of a batch whose bounds overlap `rect` (every
/// comparison ordered, so a pair with a NaN coordinate is culled), forwarding each run of kept pairs as a
/// batch of its own; only a run starting at the batch's first point keeps the batch's `first` flag.
fn cull_pairs(pts: &[InkPoint], first: bool, [l, t, r, b]: [f32; 4], sink: &mut dyn FnMut(&[InkPoint], bool)) {
    let mut run: Option<usize> = None;
    for i in 1..pts.len() {
        let (p, q) = (pts[i - 1], pts[i]);
        let (lo_x, hi_x) = if p[0] > q[0] { (q[0], p[0]) } else { (p[0], q[0]) };
        let (lo_y, hi_y) = if p[1] > q[1] { (q[1], p[1]) } else { (p[1], q[1]) };
        if hi_x >= l && r >= lo_x && hi_y >= t && b >= lo_y {
            run.get_or_insert(i - 1);
        } else if let Some(s) = run.take() {
            sink(&pts[s..i], s == 0 && first);
        }
    }
    if let Some(s) = run {
        sink(&pts[s..], s == 0 && first);
    }
}

/// The ink outline shape's figures, as its nib adapter receives each batch: a round nib lays an ellipse
/// at every point after the first (and the first when `first`) and the tangent quad to its predecessor; a
/// square nib a box per point (the first only when `first`) and the connector the box pair needs; a
/// singular nib (`E` not invertible) only the band quad of each pair.
fn as_geometry_figures(ink: &Ink, m: &Matrix, style: &InkStyle, clip: Option<[f32; 4]>) -> (Vec<NibFig>, [f32; 4]) {
    let e = nib_matrix(m, style.nib_transform);
    let factor = factor_helper(e);
    // Native's 2×2 inverse also fails when `1/det` is not finite.
    let einv = invert2x2(e).filter(|_| (1.0 / (e[0] * e[3] - e[1] * e[2])).is_finite());
    let off = degenerate_offset_vector(&e, style.nib_shape);
    let round = style.nib_shape == 0;
    let mut figs = Vec::new();
    let mut adapter = |pts: &[InkPoint], first: bool| {
        let xy = |p: InkPoint| [p[0], p[1]];
        let Some(einv) = einv else {
            let side = |p: InkPoint, s: f32| [p[0] + s * (p[2] * off[0]), p[1] + s * (p[2] * off[1])];
            for w in pts.windows(2) {
                let (a, b) = (w[0], w[1]);
                let (ap, am, bp, bm) = (side(a, 1.0), side(a, -1.0), side(b, 1.0), side(b, -1.0));
                let s = (b[1] - a[1]) * off[0] - (b[0] - a[0]) * off[1];
                figs.push(NibFig::Quad(if s <= 0.0 { [am, ap, bp, bm] } else { [ap, am, bm, bp] }));
            }
            return;
        };
        let norm = |p: InkPoint| apply2(&einv, xy(p));
        if round {
            if first {
                figs.push(NibFig::Ellipse(xy(pts[0]), pts[0][2]));
            }
            for w in pts.windows(2) {
                let (a, b) = (w[0], w[1]);
                figs.push(NibFig::Ellipse(xy(b), b[2]));
                if let Some(q) = round_quad(norm(a), a[2], norm(b), b[2]) {
                    figs.push(NibFig::Quad(q.map(|v| apply2(&e, v))));
                }
            }
        } else {
            let mut pb = box_corners(xy(pts[0]), pts[0][2], &e);
            if first {
                figs.push(NibFig::Quad(pb));
            }
            for w in pts.windows(2) {
                let (a, b) = (w[0], w[1]);
                let cb = box_corners(xy(b), b[2], &e);
                figs.push(NibFig::Quad(cb));
                if let Some((i, j)) = square_corners(norm(a), a[2], norm(b), b[2]) {
                    figs.push(NibFig::Quad([pb[i], pb[j], cb[j], cb[i]]));
                }
                pb = cb;
            }
        }
    };
    send_ink_segments(ink, m, factor, clip, ink_flattening_tolerance(style.nib_shape), style.nib_shape, &mut adapter);
    (figs, e)
}

/// Whether the fill bounds calculator flags NaN on one axis of a cubic (`p[0]` the current point): a NaN
/// control or end; otherwise, unless the coordinates are monotone (both sign tests ordered and
/// non-negative), the extremum solve in double flags a NaN discriminant `b²−ac`, a NaN root, or a NaN
/// value at a positive root. A NaN test product (inf − inf) takes the solve.
fn cubic_axis_bounds_nan(p: [f32; 4]) -> bool {
    if p[1..].iter().any(|v| v.is_nan()) {
        return true;
    }
    let (a, b, c) = (p[1] - p[0], p[2] - p[1], p[3] - p[2]);
    if (p[3] - p[1]) * a >= 0.0 && (p[2] - p[0]) * c >= 0.0 {
        return false;
    }
    let e = b.abs() * 0.000_001;
    if e > a.abs() && e > c.abs() {
        return false;
    }
    let (ad, bd, cd) = (a as f64, b as f64, c as f64);
    let x = bd * bd - ad * cd;
    if x.is_nan() {
        return true;
    }
    if !(x > 0.0) {
        return false;
    }
    let (s, by_a) = (x.sqrt(), a.abs() > c.abs());
    let den = if by_a { ad } else { cd };
    let roots = [((-bd - s) / den) as f32, ((s + -bd) / den) as f32];
    if roots.iter().any(|r| r.is_nan()) {
        return true;
    }
    roots.into_iter().filter(|&r| r > 0.0).any(|r| {
        let t = if by_a { 1.0 / (r + 1.0) } else { r / (r + 1.0) };
        let mt = 1.0 - t;
        let v = ((((mt * p[0]) * (mt * mt)) + (((p[1] * 3.0) * t) * (mt * mt))) + (((p[2] * 3.0) * (t * t)) * mt)) + ((t * p[3]) * (t * t));
        v.is_nan()
    })
}

/// The ink outline shape as one winding path of closed figures (nib ellipses as their four cubics, quads
/// snapped to the `fill_scale` grid past 28.4), or `None` when a figure point (an ellipse's control points
/// included) is NaN: the shape's fill bounds then come out NaN and the fill draws nothing. A shape reaching
/// past the ±524287 safe box is strip-clipped to it first, curves as curves. Aliased, the
/// fill tessellator's scanner range-checks every figure start and line point, so a clip crossing that
/// overflowed to infinity fails the whole fill; the antialiased edge list takes it at the far corner.
fn as_geometry_shape(ink: &Ink, m: &Matrix, style: &InkStyle, clip: Option<[f32; 4]>, fill_scale: f32, aliased: bool) -> Option<crate::geometry::Geometry> {
    use crate::geometry::{Figure, Geometry, Segment};
    use crate::pixels::Seg;
    let (figs, e) = as_geometry_figures(ink, m, style, clip);
    let snap = |v: f32| ((fill_scale * v - fill_scale * 0.5).round() + fill_scale * 0.5) / fill_scale;
    let mut nan_bounds = false;
    let shapes: Vec<([f32; 2], Vec<Seg>)> = figs
        .into_iter()
        .map(|f| match f {
            NibFig::Ellipse(c, r) => {
                let prims = ellipse_prims(c, r, &e);
                if prims.iter().any(|p| matches!(*p, Prim::Cubic(a, b, c, d) if (0..2).any(|k| cubic_axis_bounds_nan([a[k], b[k], c[k], d[k]])))) {
                    nan_bounds = true;
                }
                (ellipse_start(c, r, &e), prims.iter().map(|p| match *p { Prim::Cubic(_, a, b, c) => Seg::Cubic(a, b, c), Prim::Line(_, q) => Seg::Line(q) }).collect())
            }
            NibFig::Quad(q) => {
                let q = if fill_scale > 16.0 { q.map(|v| [snap(v[0]), snap(v[1])]) } else { q };
                (q[0], vec![Seg::Line(q[1]), Seg::Line(q[2]), Seg::Line(q[3])])
            }
        })
        .collect();
    let pts = |(st, segs): &([f32; 2], Vec<Seg>)| std::iter::once(*st).chain(segs.iter().flat_map(|g| match *g { Seg::Line(q) => vec![q], Seg::Cubic(a, b, c) => vec![a, b, c] })).collect::<Vec<_>>();
    if nan_bounds || shapes.iter().flat_map(pts).any(|p| p[0].is_nan() || p[1].is_nan()) {
        return None;
    }
    let huge = shapes.iter().flat_map(pts).any(|p| !(p[0].abs() <= 524287.0 && p[1].abs() <= 524287.0));
    let far = |p: [f32; 2]| if p.iter().any(|v| v.is_infinite()) && !p.iter().any(|v| v.is_nan()) { crate::pixels::FAR } else { p };
    let shapes: Vec<([f32; 2], Vec<Seg>)> = if huge { shapes.into_iter().filter_map(|(st, segs)| crate::pixels::clip_figure(st, &segs)).collect() } else { shapes };
    if aliased && shapes.iter().flat_map(pts).any(|p| !(p[0].is_finite() && p[1].is_finite())) {
        return None;
    }
    let build = |shapes: Vec<([f32; 2], Vec<Seg>)>| {
        let figures = shapes
            .into_iter()
            .map(|(st, segs)| if huge { (far(st), segs) } else { (st, segs) })
            .map(|(start, segs)| Figure {
                start,
                filled: true,
                closed: true,
                segments: segs.iter().map(|g| match *g { Seg::Line(q) => Segment::Line(if huge { far(q) } else { q }), Seg::Cubic(a, b, c) => Segment::Cubic(a, b, c) }).collect(),
                seg_flags: Vec::new(),
            })
            .collect();
        Geometry::Path { fill: crate::geometry::FillMode::Winding, figures }
    };
    // The aliased fill tessellator strip-clips a shape whose bounds leave the clip rect against that rect
    // before its scanner sees it, as the safe-box clip does (curves crossing it flattened at 0.25).
    if let (true, Some(c)) = (aliased, clip) {
        let b = build(shapes.clone()).get_bounds(&crate::geometry::IDENTITY);
        if !(b[0] >= b[2] || b[1] >= b[3]) && (b[0] < c[0] || b[1] < c[1] || c[2] < b[2] || c[3] < b[3]) {
            return Some(build(shapes.into_iter().filter_map(|(st, segs)| crate::pixels::clip_figure_rect(st, &segs, c)).collect()));
        }
    }
    Some(build(shapes))
}

/// [`as_geometry_shape`] flattened as a fill at `fill_scale`.
fn as_geometry_polys(ink: &Ink, m: &Matrix, style: &InkStyle, clip: Option<[f32; 4]>, fill_scale: f32, aliased: bool) -> Option<Vec<Vec<[f32; 2]>>> {
    let crate::geometry::Geometry::Path { figures, .. } = as_geometry_shape(ink, m, style, clip, fill_scale, aliased)? else { unreachable!() };
    use crate::geometry::Segment;
    Some(
        figures
            .iter()
            .map(|f| {
                let mut cur = f.start;
                let prims: Vec<Prim> = f
                    .segments
                    .iter()
                    .map(|g| {
                        let (p, end) = match *g {
                            Segment::Cubic(a, b, c) => (Prim::Cubic(cur, a, b, c), c),
                            Segment::Line(q) | Segment::Quadratic(_, q) | Segment::Arc { to: q, .. } => (Prim::Line(cur, q), q),
                        };
                        cur = end;
                        p
                    })
                    .collect();
                flatten_fill_prims_scale(f.start, &prims, fill_scale, None)
            })
            .collect(),
    )
}

// --- matrices --------------------------------------------------------------------------------

/// `GetNibExtents`: round → `1.0`, square → `√2`.
fn nib_extents(nib_shape: u32) -> f32 {
    if nib_shape == 1 {
        std::f32::consts::SQRT_2
    } else {
        1.0
    }
}

/// `GetInkFlatteningTolerance`: round → `0.5`, square → `0.25`.
fn ink_flattening_tolerance(nib_shape: u32) -> f32 {
    if nib_shape != 0 {
        0.25
    } else {
        0.5
    }
}

/// `GetFactorHelper(M, 1.0)`: the larger singular value of the 2×2 (max |diag| when off-diagonal 0).
fn factor_helper(e: [f32; 4]) -> f32 {
    let [m11, m12, m21, m22] = e;
    if m12 == 0.0 && m21 == 0.0 {
        return m11.abs().max(m22.abs());
    }
    let a = m11 * m11 + m12 * m12;
    let c = m21 * m21 + m22 * m22;
    let b = m21 * m11 + m22 * m12;
    let disc = ((b * 4.0) * b) + (a - c) * (a - c);
    let root = disc.max(0.0).sqrt();
    (((root) + (c + a)) * 0.5).max(0.0).sqrt()
}

/// nib∘device 2×2: the device transform's 2×2 composed with the ink style's nib transform, as
/// Direct2D builds it (`new = dev2x2` combined with the nib 2×2).
fn nib_matrix(m: &Matrix, nib: Matrix) -> [f32; 4] {
    // dev2x2 = [m[0],m[1],m[2],m[3]] (m11,m12,m21,m22); nib = [n11,n12,n21,n22,..].
    let (d11, d12, d21, d22) = (m[0], m[1], m[2], m[3]);
    let (n11, n12, n21, n22) = (nib[0], nib[1], nib[2], nib[3]);
    [
        d11 * n11 + d21 * n12,
        d12 * n11 + d22 * n12,
        d11 * n21 + d21 * n22,
        d12 * n21 + d22 * n22,
    ]
}

fn invert2x2(e: [f32; 4]) -> Option<[f32; 4]> {
    let [a, b, c, d] = e;
    let det = a * d - b * c;
    if det == 0.0 {
        return None;
    }
    let inv = 1.0 / det;
    Some([d * inv, -b * inv, -c * inv, a * inv])
}

/// Apply a 2×2 `e` to offset `o` and add centre `c`: `c + e·o` (native `AddEllipse` offset form).
fn off(c: [f32; 2], e: &[f32; 4], o: [f32; 2]) -> [f32; 2] {
    [e[0] * o[0] + e[2] * o[1] + c[0], e[1] * o[0] + e[3] * o[1] + c[1]]
}

fn apply2(e: &[f32; 4], p: [f32; 2]) -> [f32; 2] {
    [e[0] * p[0] + e[2] * p[1], e[1] * p[0] + e[3] * p[1]]
}

/// Apply a 3×2 device matrix to a point.
fn apply(m: &Matrix, p: [f32; 2]) -> [f32; 2] {
    [m[0] * p[0] + m[2] * p[1] + m[4], m[1] * p[0] + m[3] * p[1] + m[5]]
}

fn blend_of(mode: u32) -> Blend {
    match mode {
        1 => Blend::Copy,
        2 => Blend::Min,
        3 => Blend::Add,
        4 => Blend::Max,
        _ => Blend::SourceOver,
    }
}

/// deviceTransform = DpiScale ∘ world from the context's public state (as `mesh.rs`).
fn device_matrix(dc: &DeviceContext) -> Matrix {
    let w = dc.get_transform();
    let (dx, dy) = dc.get_dpi();
    let s: Matrix = if dc.get_unit_mode() == 0 {
        [dx / 96.0, 0.0, 0.0, dy / 96.0, 0.0, 0.0]
    } else {
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    };
    [
        s[0] * w[0] + s[2] * w[1],
        s[1] * w[0] + s[3] * w[1],
        s[0] * w[2] + s[2] * w[3],
        s[1] * w[2] + s[3] * w[3],
        s[0] * w[4] + s[2] * w[5] + s[4],
        s[1] * w[4] + s[3] * w[5] + s[5],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bitmap_target, Factory};

    const CURVE: [InkBezierSegment; 1] =
        [InkBezierSegment { p1: [40.0, 10.0, 5.0], p2: [80.0, 90.0, 8.0], end: [110.0, 40.0, 2.0] }];

    // A rank-1 nib `E = [a,a,a,a]`: dominant column `(a,a)`, `d̂ = (1/√2, 1/√2)`, so the round offset is
    // `(d̂·col0, d̂·col1) = (a√2, a√2)`; a square nib scales by `|d̂.x|+|d̂.y| = √2`, giving `(2a, 2a)`.
    #[test]
    fn degenerate_offset_vector_rank1() {
        let a = 1.7625_f32;
        let round = degenerate_offset_vector(&[a, a, a, a], 0);
        assert!((round[0] - a * 2.0_f32.sqrt()).abs() < 1e-3 && (round[1] - round[0]).abs() < 1e-6, "{round:?}");
        let square = degenerate_offset_vector(&[a, a, a, a], 1);
        assert!((square[0] - 2.0 * a).abs() < 1e-3 && (square[1] - square[0]).abs() < 1e-6, "{square:?}");
        // A dead column 0 falls back to column 1.
        let col1 = degenerate_offset_vector(&[0.0, 3.0, 0.0, 4.0], 0);
        assert!(col1[0].abs() < 1e-4 && (col1[1] - 5.0).abs() < 1e-3, "{col1:?}");
    }

    // A singular (rank-1) nib draws a non-empty band where our old path drew ~nothing.
    // A -3.4e38 control overflows the flattener's y differences to NaN while x stays finite (native trace of
    // r10 15088: 13.6759 then 20.4962 after the start), and the culling sink, whose comparisons are all
    // ordered, drops every pair carrying that NaN.
    #[test]
    fn nan_pairs_are_culled() {
        let mut out = Vec::new();
        let c = [[92.719_12, 1e-20, 2.456_151], [112.832_3, -3.4e38, 1.237_44], [1e9, 58.113_377, 0.722_877_4]];
        flatten_ink_batches(true, true, [12.443_213, 36.834_35, 2.934_126], c, 0.5, 5e5, &mut |b, _| out.extend_from_slice(&b[1..]));
        assert!(out[0][1].is_nan() && (out[0][0] - 13.6759).abs() < 1e-3 && (out[1][0] - 20.4962).abs() < 1e-3);
        assert_eq!(out.last().map(|p| [p[0], p[1]]), Some([1e9, 58.113_377]));
        let pts = [[10.0, 10.0, 1.0], [12.0, f32::NAN, 1.0], [14.0, 20.0, 1.0], [16.0, 21.0, 1.0]];
        let mut runs = Vec::new();
        cull_pairs(&pts, true, [0.0, 0.0, 100.0, 100.0], &mut |b, f| runs.push((b.len(), f)));
        assert_eq!(runs, vec![(2, false)]);
    }

    // The square-nib dot and body vertex shaders against the native shader run vertex by vertex on the
    // captured mesh (identity nib, unit feather): a 4-radius dot, a 0.3-radius dot that collapses to its
    // centre with coverage 0.12, and a body between two points at full and at sub-pixel radii.
    #[test]
    fn square_nib_vertex_shaders() {
        let id = [1.0, 0.0, 0.0, 1.0];
        let close = |got: [([f32; 2], f32); 12], want: &[[f32; 3]]| {
            for (k, (g, w)) in got.iter().zip(want).enumerate() {
                assert!((g.0[0] - w[0]).abs() < 1e-4 && (g.0[1] - w[1]).abs() < 1e-4 && (g.1 - w[2]).abs() < 1e-6, "vertex {k}: {g:?} vs {w:?}");
            }
        };
        close(square_dot_verts(&id, &id, [6.0, 6.0], 4.0, 1.0), &[
            [2.5, 1.5, 0.0], [9.5, 1.5, 0.0], [1.5, 2.5, 0.0], [2.5, 2.5, 1.0], [9.5, 2.5, 1.0], [10.5, 2.5, 0.0],
            [1.5, 9.5, 0.0], [2.5, 9.5, 1.0], [9.5, 9.5, 1.0], [10.5, 9.5, 0.0], [2.5, 10.5, 0.0], [9.5, 10.5, 0.0],
        ]);
        close(square_dot_verts(&id, &id, [6.0, 6.0], 0.3, 1.0), &[
            [6.0, 5.0, 0.0], [6.0, 5.0, 0.0], [5.0, 6.0, 0.0], [6.0, 6.0, 0.120_000_035], [6.0, 6.0, 0.120_000_035], [7.0, 6.0, 0.0],
            [5.0, 6.0, 0.0], [6.0, 6.0, 0.120_000_035], [6.0, 6.0, 0.120_000_035], [7.0, 6.0, 0.0], [6.0, 7.0, 0.0], [6.0, 7.0, 0.0],
        ]);
        let (a, b) = ([8.0, 18.0, 4.0], [9.932_373, 16.516_846, 3.989_257_8]);
        close(square_body_verts(&id, &id, a, b, 1.0), &[
            [12.3069, 22.3947, 0.0], [14.2285, 20.9008, 0.0], [11.6931, 21.6053, 1.0], [13.6147, 20.1114, 1.0],
            [4.3020, 14.3985, 1.0], [6.2451, 12.9261, 1.0], [3.6980, 13.6015, 0.0], [5.6411, 12.1291, 0.0],
            [11.5, 21.5, 1.0], [4.5, 14.5, 1.0], [13.4216, 20.0061, 1.0], [6.4431, 13.0276, 1.0],
        ].map(|v| v));
        close(square_body_verts(&id, &id, [8.0, 18.0, 0.3], [b[0], b[1], 0.2], 1.0), &[
            [8.677_139, 18.736_511, 0.0], [10.601_715, 17.260_086, 0.0], [8.023_361, 17.979_813, 0.846_278_3], [9.947_952, 16.503_387, 0.564_185_5],
            [7.934_509, 18.044_571, 0.833_602_8], [9.888_702, 16.546_555, 0.555_735_2], [7.371_872, 17.217_865, 0.0], [9.326_08, 15.719_849, 0.0],
            [8.0, 18.0, 0.120_000_035], [8.0, 18.0, 0.120_000_035], [9.932_373, 16.516_846, -0.080_000_01], [9.932_373, 16.516_846, -0.080_000_01],
        ]);
    }

    #[test]
    fn singular_nib_draws_a_band() {
        let px = render(1, [0.0, 0.0, 0.0, 1.0], 0, [-1.5, -1.5, -1.5, -1.5, 0.0, 0.0], [20.0, 60.0, 3.0], &CURVE);
        let lit = px.chunks_exact(4).filter(|p| p[3] != 0).count();
        assert!(lit > 50, "singular-nib band should stroke, got {lit} lit px");
    }

    fn render(aa: u32, color: [f32; 4], shape: u32, nib: Matrix, start: InkPoint, segs: &[InkBezierSegment]) -> Vec<u8> {
        let factory = Factory::new(false);
        let mut dc = factory.create_device().create_device_context();
        dc.set_dpi(96.0, 96.0);
        dc.set_target_bitmap(bitmap_target(128, 96, 96.0));
        dc.begin_draw();
        dc.clear(Some([1.0, 1.0, 1.0, 1.0]));
        dc.set_antialias_mode(aa);
        let brush = dc.create_solid_color_brush(color, None);
        let style = dc.create_ink_style(shape, nib);
        let mut ink = dc.create_ink(start);
        ink.add_segments(segs);
        dc.draw_ink(&ink, &brush, Some(&style));
        let _ = dc.end_draw();
        dc.take_target().unwrap().data
    }

    // Robustness (fuzz 841): a non-finite / enormous nib radius with extreme control points must not
    // blow the swept-nib outline up to millions of vertices — that hung the union tessellator for
    // hours. The radius clamp in `outline_polys` bounds the vertex count (and culls off-target nibs).
    #[test]
    fn huge_radius_ink_outline_is_bounded() {
        let ink = Ink {
            start: [19.0, 0.0, f32::INFINITY],
            segments: vec![
                InkBezierSegment { p1: [90.0, 1e9, 2.6], p2: [-1e9, 40.0, 1.3], end: [93.0, 62.0, 0.46] },
                InkBezierSegment { p1: [69.0, 0.5, 2.15], p2: [88.0, 20.0, 3.1], end: [0.5, 32.0, 2.2] },
            ],
        };
        let m = [2.0, 0.0, 0.0, 2.0, 0.0, 0.0]; // dpi-192 device
        for shape in [0u32, 1] {
            let style = InkStyle { nib_shape: shape, nib_transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0] };
            let polys = outline_polys(&ink, &m, &style, 64.0, 128, 128);
            let verts: usize = polys.iter().map(|p| p.len()).sum();
            assert!(verts < 200_000, "nib {shape}: outline must stay bounded, got {verts} verts");
        }
    }

    // Render inkbody-01 (white brush, transparent clear) and dump to $D2D_INK_OUT — the output is then a
    // monotonic LUT_white[coverage], directly comparable to native's same-setup render to localize the rim +1.
    #[test]
    fn ink_white_dump() {
        let Some(out) = std::env::var_os("D2D_INK_OUT") else { return };
        let factory = Factory::new(false);
        let mut dc = factory.create_device().create_device_context();
        dc.set_dpi(96.0, 96.0);
        dc.set_target_bitmap(bitmap_target(128, 96, 96.0));
        dc.begin_draw();
        dc.clear(Some([0.0, 0.0, 0.0, 0.0]));
        dc.set_antialias_mode(0);
        let brush = dc.create_solid_color_brush([1.0, 1.0, 1.0, 1.0], None);
        let style = dc.create_ink_style(0, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mut ink = dc.create_ink([16.0, 40.0, 6.0]);
        ink.add_segments(&[InkBezierSegment { p1: [48.0, 45.333, 6.0], p2: [80.0, 50.667, 6.0], end: [112.0, 56.0, 6.0] }]);
        dc.draw_ink(&ink, &brush, Some(&style));
        let _ = dc.end_draw();
        std::fs::write(out, dc.take_target().unwrap().data).unwrap();
    }

    // Count white pixels not reachable from the border over the white region = interior holes. The
    // swept nib's nonzero union must be gap-free (a hole would mean a missing connector/disc).
    fn holes(px: &[u8]) -> usize {
        const W: usize = 128;
        const H: usize = 96;
        let white = |i: usize| px[i * 4] == 255 && px[i * 4 + 1] == 255 && px[i * 4 + 2] == 255;
        let mut seen = vec![false; W * H];
        let mut stack: Vec<usize> = (0..W * H).filter(|&i| (i % W == 0 || i % W == W - 1 || i / W == 0 || i / W == H - 1) && white(i)).collect();
        for &i in &stack {
            seen[i] = true;
        }
        while let Some(i) = stack.pop() {
            let (x, y) = (i % W, i / W);
            for (nx, ny) in [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)] {
                if nx < W && ny < H {
                    let j = ny * W + nx;
                    if !seen[j] && white(j) {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        (0..W * H).filter(|&i| white(i) && !seen[i]).count()
    }

    // The opaque fast path: deep-interior pixels are byte-exact against the `v2-ink`
    // WARP oracle — cov 1 ⇒ gamma LUT entry 255 ⇒ the round-half-up premultiplied brush
    // `[179,51,26,255]`, confirming the disc+body union and the fast-path gamma-LUT composite.
    // (The feather *rim* is a ≤1-LSB near-miss — [`feather_disc`]'s n-gon vs native's dot VS; the scene
    // stays an honest FAIL, not in `passing.txt`. See the module note.)
    #[test]
    fn v2_ink_interior_byte_exact() {
        let Some(suite) = d2d_reference::suite_dir() else { return };
        let px = render(0, [0.1, 0.2, 0.7, 1.0], 0, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [20.0, 70.0, 3.0], &CURVE);
        let oracle = std::fs::read(suite.join("oracle/win11-26100/v2-ink.bgra")).unwrap();
        // Deep-interior sample points (well inside the stroke, away from the feathered rim).
        for &(x, y) in &[(61usize, 51usize), (61, 48)] {
            let i = (y * 128 + x) * 4;
            assert_eq!(px[i..i + 4], oracle[i..i + 4], "interior ({x},{y}) differs from the v2-ink oracle");
        }
    }

    #[test]
    fn ink2_01_byte_exact() {
        let Some(suite) = d2d_reference::suite_dir() else { return };
        let px = render(1, [0.1, 0.2, 0.7, 0.6], 0, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [20.0, 70.0, 3.0], &CURVE);
        let oracle = std::fs::read(suite.join("oracle/win11-26100/ink2-01-round-alias.bgra")).unwrap();
        let diffs = px.chunks_exact(4).zip(oracle.chunks_exact(4)).filter(|(a, b)| a != b).count();
        assert_eq!(diffs, 0, "{diffs} px differ from the WARP AsGeometry oracle");
    }

    #[test]
    fn swept_outline_is_gap_free() {
        for (shape, nib) in [(0u32, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]), (1, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]), (0, [1.6, 0.0, 0.0, 0.6, 0.0, 0.0])] {
            let px = render(1, [0.1, 0.2, 0.7, 0.6], shape, nib, [20.0, 70.0, 3.0], &CURVE);
            let filled = px.chunks_exact(4).filter(|p| !(p[0] == 255 && p[1] == 255 && p[2] == 255)).count();
            assert!(filled > 500, "shape {shape}: stroke barely drawn ({filled} px)");
            assert_eq!(holes(&px), 0, "shape {shape}: swept outline has interior holes");
        }
    }

    #[test]
    fn get_bounds_contains_stroke() {
        // Every non-white pixel of the drawn stroke must lie inside GetBounds.
        let style = InkStyle::default();
        let ink = Ink { start: [20.0, 70.0, 3.0], segments: CURVE.to_vec() };
        let b = ink.get_bounds(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &style);
        let px = render(1, [0.1, 0.2, 0.7, 0.6], 0, style.nib_transform, [20.0, 70.0, 3.0], &CURVE);
        for (i, p) in px.chunks_exact(4).enumerate() {
            if !(p[0] == 255 && p[1] == 255 && p[2] == 255) {
                let (x, y) = ((i % 128) as f32, (i / 128) as f32);
                assert!(x + 1.0 >= b[0] && x <= b[2] + 1.0 && y + 1.0 >= b[1] && y <= b[3] + 1.0, "pixel ({x},{y}) outside bounds {b:?}");
            }
        }
    }
}
