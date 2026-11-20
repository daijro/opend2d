//! Colour-font glyph runs (`ENABLE_COLOR_FONT`). `IDWriteFactory4::TranslateColorGlyphRun` reports
//! a run's colour formats in priority order — COLR paint tree, COLR layers, PNG/JPEG/TIFF bitmaps,
//! OT-SVG — and Direct2D dispatches each. The reference font (Segoe
//! UI Emoji on win11-26100, every `textx-emoji-*` scene) is **COLR v1**: a paint graph of
//! solid/linear/radial-gradient fills under transforms and glyph clips, which native renders with
//! `ID2D1DeviceContext7::DrawPaintGlyphRun` → `PaintTreeRenderer` (it maps each `DWRITE_PAINT_ELEMENT`
//! onto Direct2D brushes, geometry and layers: `RenderGlyphNode`→`RenderLeafGlyph` fills a glyph
//! outline with the paint's brush; `GetLinear/RadialGradientBrush` build the gradient brushes;
//! `RenderCompositeNode` composites subtrees). It has no bitmap or SVG tables, so those formats never
//! fire for this corpus.
//!
//! This reproduces that mapping, driving the identical paint-graph traversal with `skrifa`'s
//! `ColorPainter` (spec-equivalent to DirectWrite's `IDWritePaintReader`) and reusing the d2d core's
//! WARP-exact gradient brushes (`crate::brush`) and the shared geometry rasterizer
//! (`brush::fill_geometry`) — no second rasterizer. Each paint-tree leaf is a glyph outline filled
//! with a solid/linear/radial brush, composited source-over, exactly as `RenderLeafGlyph` does.
//!
//! DrawText's paint-tree bitmap is rendered at device resolution. Ordinary leaves use DirectWrite
//! coverage and the gamma-1.8 text blend. A leaf whose transformed em exceeds 100 pixels switches
//! to outlines: solid brushes use the feather realization, gradients use the geometry scan. Both
//! outline paths blend linearly. The leaf transform, not just the emoji's base size, decides this.
//! Native emoji captures are identical with warm and cold font identities; the remaining coverage
//! differences are not a font-cache dilation (measured via VM probes).
//!
//! `PaintComposite` (every mode) and nested glyph clips are reproduced via an offscreen group stack
//! ([`Frame`]) mirroring native's command-list intermediates: `RenderCompositeNode` merges the
//! backdrop/source offscreens with the Composite (Porter-Duff) / Blend effect, which reuse the
//! byte-exact [`crate::layer::composite`] table and [`crate::effects::builtin::blend`] shader through
//! the native DWRITE→D2D mode maps ([`dwrite_to_composite`]/[`dwrite_to_blend`]); a clip glyph masks
//! its subtree's offscreen by the outline coverage (`RenderGlyphNode`). COLRv1 two-circle radials
//! (`r0 ≠ 0`, OR the focal point outside the outer circle — **common in Segoe UI Emoji**, not just
//! `colorfont2-04`) use native's internal conical-gradient shader, reproduced as
//! [`fill_conical`] (the exact two-circle solve [`conical_t`] feeding the core gradient ramp). Sweep
//! gradients (`colorfont2-09` only) use [`Painter::fill_sweep`]; its exact `atan2` lives in WARP's
//! native's sweep-gradient color-source shader (a VM-capture target), so it too sits at the sampler floor.
//!
//! The run baseline (historically ~6.5 px high from `hhea` ascender + gap) is now correct: `text::layout`
//! uses `OS/2.usWinAscent` for a font without `USE_TYPO_METRICS`, so the emoji are vertically aligned.

use super::hinting::{Config as HConfig, Instance as HInstance, Mode as HMode};
use super::raster::{self, AlphaTexture};
use super::{render_glyph_run_fg, snap_baseline, GlyphRun, RenderingParams, TextFg};
use std::cell::RefCell;
use std::collections::HashMap;
use crate::bitmap::unorm_mul;
use crate::brush::Brush as PBrush;
use crate::effects::builtin::blend::blend;
use crate::geometry::{Figure, FillMode, Geometry, Segment};
use crate::pixels::{Aa, Blend, Target};
use d2d_types::contracts::{ImageDesc, PixelBuffer};
use skrifa::color::{Brush as CBrush, ColorGlyphCollection, ColorGlyphFormat, ColorPainter, ColorStop, CompositeMode, Extend, Transform};
use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{DrawSettings, OutlinePen};
use skrifa::raw::tables::gasp::GaspRangeBehavior;
use skrifa::raw::tables::glyf::Glyph;
use skrifa::raw::types::BoundingBox;
use skrifa::raw::TableProvider;
use skrifa::{FontRef, GlyphId, MetadataProvider};

pub mod bitmap;
mod paint;

const PALETTE_FOREGROUND: u16 = 0xFFFF; // CPAL "use the text foreground colour" sentinel.

/// 3×2 affine `[xx, yx, xy, yy, dx, dy]` (same convention as `crate::brush`): `apply(m,p)` maps a
/// point, `mul(a,b)` composes so the result applies `b` first then `a` (`a ∘ b`).
type Mat = [f32; 6];
fn mat_of(t: &Transform) -> Mat {
    [t.xx, t.yx, t.xy, t.yy, t.dx, t.dy]
}
/// A design-unit (y-up) transform as DirectWrite hands a COLR `PaintTransform` to d2d1: in em units, y down
/// (`[xx, −yx, −xy, yy, dx·k, −dy·k]`, `k` = 1/upem).
fn em_of(m: &Mat, k: f32) -> Mat {
    [m[0], -m[1], -m[2], m[3], m[4] * k, -m[5] * k]
}
fn mul(a: &Mat, b: &Mat) -> Mat {
    [
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
        a[0] * b[4] + a[2] * b[5] + a[4],
        a[1] * b[4] + a[3] * b[5] + a[5],
    ]
}
fn apply(m: &Mat, x: f32, y: f32) -> [f32; 2] {
    [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]
}
fn invert(m: &Mat) -> Mat {
    let det = m[0] * m[3] - m[1] * m[2];
    let i = 1.0 / det;
    [m[3] * i, -m[1] * i, -m[2] * i, m[0] * i, (m[2] * m[5] - m[3] * m[4]) * i, (m[1] * m[4] - m[0] * m[5]) * i]
}

/// Two-circle (COLRv1 `r0 ≠ 0`) radial parameter `t` at brush-space point `p`: the largest `t` whose
/// interpolated circle — centre `lerp(c0,c1,t)`, radius `lerp(r0,r1,t)` — passes through `p` with a
/// non-negative radius, or `None` where the gradient is undefined (no real/valid root). Solves
/// `a·t² − 2B·t + c = 0` with exact f32 mul/add/sqrt/div (no FMA), the arithmetic native's
/// The conical-gradient brush shader (general, and the `a==0` variant) runs
/// per pixel. `a = |c1−c0|² − (r1−r0)²`, `B = (p−c0)·(c1−c0) + r0·(r1−r0)`, `c = |p−c0|² − r0²`.
/// Native's conical/two-circle **realization placement** — the first step of the realization chain
/// that produces the colorizer leaf atlas frame. Maps the two circles through the
/// brush→device matrix `b2d` into four device points and the device footprint:
///   P0 = c0,  P1 = c1,  P2 = c0 + b2d_linear·(r0,0),  P3 = c1 + b2d_linear·(r1,0)
/// (the radius offsets are the design-unit radii rotated/scaled by `b2d`'s 2×2 — anisotropic for a
/// stretched transform), and `footprint = max(|P1 P3|, |P0 P2|, |P0 P1|)` = max(r1, r0, |c1−c0|) in
/// device units. Verified against cf04's captured values; the downstream frame builder
/// (native's conical-gradient color source, which normalizes these into the VS `cb1` v1/v2 with the (W−1)/W
/// scale + atlas sub-pixel) is wired in a following commit. INERT until then.
#[allow(dead_code)]
fn conical_placement(c0: [f32; 2], r0: f32, c1: [f32; 2], r1: f32, b2d: &Mat) -> ([[f32; 2]; 4], f32) {
    let lin = |v: [f32; 2]| [b2d[0] * v[0] + b2d[2] * v[1], b2d[1] * v[0] + b2d[3] * v[1]]; // 2×2 only
    let p0 = apply(b2d, c0[0], c0[1]);
    let p1 = apply(b2d, c1[0], c1[1]);
    let (dr0, dr1) = (lin([r0, 0.0]), lin([r1, 0.0]));
    let p2 = [p0[0] + dr0[0], p0[1] + dr0[1]];
    let p3 = [p1[0] + dr1[0], p1[1] + dr1[1]];
    let dist = |a: [f32; 2], b: [f32; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    let footprint = dist(p1, p3).max(dist(p0, p2)).max(dist(p0, p1));
    ([p0, p1, p2, p3], footprint)
}

/// Native's COLR conical colorizer frame — the VS `cb1` the two-circle PS consumes — reconstructed from
/// geometry (the realization chain, over
/// GetRealizationPlacementData/Size/CalculateTextureMapping). Returns `(scale, offset, v2)` for the
/// per-pixel frame `v1 = scale·(device − c0_device) + offset` (isotropic `scale`; `offset` in v1 units)
/// and the normalized circle params `v2 = (_, r0, dr, |cd|)/upem` the solve consumes
/// (`A=v2.w²−v2.z²`, `B=v1.x·v2.w+v2.z·v2.y`, `C=|v1|²−v2.y²`). The scale carries the `(W−1)/W` texcoord
/// factor with `W = radial_width(max(r0,r1,|cd|))` — the DESIGN-unit footprint's gradient-texture width
/// (cf04: 320 → 512 → 511/512) — over the device scale. The offset places c0 at the atlas tile (the
/// leaf ink-bounds min minus a 2 px border, per axis) plus a `+0.5/W` half-texel ALONG the gradient axis.
/// Verified bit-exact against cf04's captured cb1. INERT until wired into `fill_conical` (next commit).
#[allow(dead_code, clippy::too_many_arguments)]
fn conical_frame(c0: [f32; 2], r0: f32, c1: [f32; 2], r1: f32, b2d: &Mat, upem: f32, ink_min: [f32; 2]) -> (f32, [f32; 2], [f32; 4]) {
    let cd = [c1[0] - c0[0], c1[1] - c0[1]];
    let axis = (cd[0] * cd[0] + cd[1] * cd[1]).sqrt();
    let design_fp = r0.max(r1).max(axis);
    let w = crate::brush::radial_width(design_fp, 0) as f32;
    let b2d_scale = (b2d[0] * b2d[0] + b2d[1] * b2d[1]).sqrt();
    let scale = (1.0 / (upem * b2d_scale)) * (w - 1.0) / w;
    let c0d = apply(b2d, c0[0], c0[1]);
    let tile = [ink_min[0] - 2.0, ink_min[1] - 2.0]; // atlas = ink bounds + 2 px border each side
    let (hx, hy) = if axis > 0.0 { (0.5 / w * cd[0] / axis, 0.5 / w * cd[1] / axis) } else { (0.0, 0.0) };
    let offset = [scale * (tile[0] - c0d[0]) + hx, scale * (tile[1] - c0d[1]) + hy];
    (scale, offset, [0.5, r0 / upem, (r1 - r0) / upem, axis / upem])
}

/// One raw `glyf` contour (design points + on-curve flags) → device-space (points, on-curve) for
/// `coverage_outline`, inserting the TrueType implied on-curve midpoint between consecutive off-curve
/// points IN DESIGN SPACE (before `dev` transforms each), so off/on alternate as the scan converter
/// expects — the hinted path's byte-exact midpoint rule, not `skrifa`'s post-scale float one.
fn expand_glyf_contour(raw: &[(f64, f64, bool)], dev: &impl Fn(f64, f64) -> [f64; 2]) -> (Vec<[f64; 2]>, Vec<bool>) {
    let n = raw.len();
    let (mut dp, mut don) = (Vec::new(), Vec::new());
    if n == 0 {
        return (dp, don);
    }
    // Start the walk at an on-curve point; if the contour is all off-curve, synthesise the start as the
    // midpoint of the last and first points (the standard TrueType rule).
    let mut seq: Vec<(f64, f64, bool)> = Vec::with_capacity(n + 1);
    match raw.iter().position(|p| p.2) {
        Some(f) => (0..n).for_each(|k| seq.push(raw[(f + k) % n])),
        None => {
            seq.push(((raw[n - 1].0 + raw[0].0) * 0.5, (raw[n - 1].1 + raw[0].1) * 0.5, true));
            seq.extend_from_slice(raw);
        }
    }
    let mut prev_off: Option<(f64, f64)> = None;
    for (i, &(x, y, on)) in seq.iter().enumerate() {
        if i == 0 || on {
            dp.push(dev(x, y));
            don.push(true);
            prev_off = None;
        } else {
            if let Some((cx, cy)) = prev_off {
                dp.push(dev((cx + x) * 0.5, (cy + y) * 0.5));
                don.push(true);
            }
            dp.push(dev(x, y));
            don.push(false);
            prev_off = Some((x, y));
        }
    }
    (dp, don)
}

/// The device ink-bounds min of flattened/quadratic leaf contours: on-curve endpoints plus each
/// quadratic's per-axis turning point `t*=(p0−c)/(p0−2c+p1)` when `t*∈(0,1)` — the visual extent, not
/// the raw off-curve control points (which sit outside the ink and would mis-phase the atlas tile).
/// Convert one device-space contour (points + on-curve flags, as the scan converter takes) into a
/// filled closed `Figure` with line/quadratic segments, for the COLR_FEATHER geometry feather. `skrifa`
/// emits an explicit on-curve end per quadratic, so an off-curve point is always followed by an
/// on-curve one; a run of on-curve points is a polyline. `None` if the contour has no on-curve point.
fn contour_to_figure(pts: &[[f64; 2]], on: &[bool]) -> Option<Figure> {
    let n = pts.len();
    let start_i = (0..n).find(|&i| on.get(i).copied().unwrap_or(true))?;
    let p = |i: usize| [pts[i % n][0] as f32, pts[i % n][1] as f32];
    let ison = |i: usize| on.get(i % n).copied().unwrap_or(true);
    let mut segs = Vec::new();
    let (mut i, mut count) = (start_i, 0usize);
    while count < n {
        let j = i + 1;
        if ison(j) {
            segs.push(Segment::Line(p(j)));
            i = j;
            count += 1;
        } else {
            // off-curve control at j, on-curve end at j+1 (skrifa's explicit midpoint).
            segs.push(Segment::Quadratic(p(j), p(j + 1)));
            i = j + 1;
            count += 2;
        }
    }
    Some(Figure { start: p(start_i), filled: true, closed: true, segments: segs, seg_flags: Vec::new() })
}

/// Native's rectangular COLR group mask samples an 8×8 grid at pixel-centred eighths.
/// Edges exactly on a sample exclude the right/bottom sample (round-half-down). Keep the mask
/// fractional until it modulates the completed group's colour; an intermediate alpha byte loses
/// the native half-coverage tie (253 × 1/2 → 126).
fn box_cov(rect: Option<[f32; 4]>, x: i32, y: i32) -> f32 {
    let Some([l, t, r, b]) = rect else { return 1.0 };
    let edge = |v: f32, p: i32| ((v - p as f32) * 8.0 - 0.5).ceil().clamp(0.0, 8.0);
    let cx = (edge(r, x) - edge(l, x)).max(0.0);
    let cy = (edge(b, y) - edge(t, y)).max(0.0);
    cx * cy / 64.0
}

fn ink_min_of(contours: &[(Vec<[f64; 2]>, Vec<bool>)]) -> Option<[f32; 2]> {
    let mut mn = [f64::INFINITY; 2];
    for (pts, on) in contours {
        let n = pts.len();
        for (i, &p) in pts.iter().enumerate() {
            if on.get(i).copied().unwrap_or(true) {
                mn[0] = mn[0].min(p[0]);
                mn[1] = mn[1].min(p[1]);
            } else if n > 0 {
                let prev = pts[(i + n - 1) % n];
                let next = pts[(i + 1) % n];
                let p0 = if on.get((i + n - 1) % n).copied().unwrap_or(true) { prev } else { [(prev[0] + p[0]) * 0.5, (prev[1] + p[1]) * 0.5] };
                let p2 = if on.get((i + 1) % n).copied().unwrap_or(true) { next } else { [(next[0] + p[0]) * 0.5, (next[1] + p[1]) * 0.5] };
                for a in 0..2 {
                    let den = p0[a] - 2.0 * p[a] + p2[a];
                    if den.abs() > 1e-9 {
                        let ts = (p0[a] - p[a]) / den;
                        if ts > 0.0 && ts < 1.0 {
                            mn[a] = mn[a].min((1.0 - ts) * (1.0 - ts) * p0[a] + 2.0 * (1.0 - ts) * ts * p[a] + ts * ts * p2[a]);
                        }
                    }
                }
            }
        }
    }
    mn[0].is_finite().then_some([mn[0] as f32, mn[1] as f32])
}

fn conical_t(p: [f32; 2], c0: [f32; 2], r0: f32, c1: [f32; 2], r1: f32) -> Option<f32> {
    let cd = [c1[0] - c0[0], c1[1] - c0[1]];
    let dr = r1 - r0;
    let a = cd[0] * cd[0] + cd[1] * cd[1] - dr * dr;
    let pd = [p[0] - c0[0], p[1] - c0[1]];
    let bb = pd[0] * cd[0] + pd[1] * cd[1] + r0 * dr;
    let cc = pd[0] * pd[0] + pd[1] * pd[1] - r0 * r0;
    if a == 0.0 {
        // Degenerate pencil (native op 23251): one linear root.
        if bb == 0.0 {
            return None;
        }
        let t = cc / (2.0 * bb);
        return (r0 + dr * t >= 0.0).then_some(t);
    }
    let disc = bb * bb - a * cc;
    if disc < 0.0 {
        return None;
    }
    let root = disc.sqrt();
    let mut best: Option<f32> = None;
    for t in [(bb + root) / a, (bb - root) / a] {
        if r0 + dr * t >= 0.0 && best.is_none_or(|b| t > b) {
            best = Some(t);
        }
    }
    best
}

/// skrifa `Extend` → `crate::brush` extend int (0 clamp, 1 wrap, 2 mirror).
fn extend_int(e: Extend) -> u32 {
    match e {
        Extend::Repeat => 1,
        Extend::Reflect => 2,
        _ => 0,
    }
}

/// `DWRITE_COLOR_COMPOSITE_MODE` (== skrifa `CompositeMode as u8`) → `D2D1_COMPOSITE_MODE` for the
/// Porter-Duff modes 1,3..=12, exactly as native `DWriteCompositeModeToD2DCompositeMode` maps them
/// DEST (2) is handled by the caller — native swaps the effect inputs and copies the
/// backdrop, so the result is just the backdrop.
fn dwrite_to_composite(m: u8) -> u32 {
    match m {
        1 => 10,  // SRC → SOURCE_COPY
        3 => 0,   // SRC_OVER
        4 => 1,   // DEST_OVER
        5 => 2,   // SRC_IN
        6 => 3,   // DEST_IN
        7 => 4,   // SRC_OUT
        8 => 5,   // DEST_OUT
        9 => 6,   // SRC_ATOP
        10 => 7,  // DEST_ATOP
        11 => 8,  // XOR
        _ => 9,   // 12 PLUS
    }
}

/// `DWRITE_COLOR_COMPOSITE_MODE` → `D2D1_BLEND_MODE` for the blend modes 13..=27, as native
/// The composite-mode → blend-mode mapping; the values index
/// `effects::builtin::blend::blend`.
fn dwrite_to_blend(m: u8) -> u32 {
    match m {
        13 => 1,   // SCREEN
        14 => 11,  // OVERLAY
        15 => 2,   // DARKEN
        16 => 3,   // LIGHTEN
        17 => 9,   // COLOR_DODGE
        18 => 5,   // COLOR_BURN
        19 => 13,  // HARD_LIGHT
        20 => 12,  // SOFT_LIGHT
        21 => 18,  // DIFFERENCE
        22 => 19,  // EXCLUSION
        23 => 0,   // MULTIPLY
        24 => 20,  // HSL_HUE
        25 => 21,  // HSL_SATURATION
        26 => 22,  // HSL_COLOR
        _ => 23,   // 27 HSL_LUMINOSITY
    }
}

/// One pixel of the composite native runs for `PaintComposite`: premultiplied BGRA source `s` composited over backdrop `d` with a
/// `DWRITE_COLOR_COMPOSITE_MODE`. Porter-Duff (1..=12) goes through the integer `CompositeEffect` blend
/// unit (`crate::layer::composite`); blend (13..=27) through the `CBlend` shader
/// (`effects::builtin::blend`), which works in straight RGBA, so the BGRA channels are swapped around
/// it. CLEAR (0) draws nothing.
fn composite_pixel(mode: u8, s: [u8; 4], d: [u8; 4]) -> [u8; 4] {
    match mode {
        0 => [0; 4], // CLEAR
        2 => d,      // DEST: keep the backdrop
        1 | 3..=12 => crate::layer::composite(s.map(|v| v as i32), d, dwrite_to_composite(mode)).unwrap_or(d),
        _ => {
            let rgba = |p: [u8; 4]| [p[2] as f32 / 255.0, p[1] as f32 / 255.0, p[0] as f32 / 255.0, p[3] as f32 / 255.0];
            let o = blend(dwrite_to_blend(mode), rgba(d), rgba(s), [0.0, 0.0]);
            let q = |x: f32| (x * 255.0).round_ties_even().clamp(0.0, 255.0) as u8;
            [q(o[2]), q(o[1]), q(o[0]), q(o[3])]
        }
    }
}

/// Composite the premultiplied offscreen `src` onto `dst` per pixel with [`composite_pixel`] — the
/// backdrop/source command-list intermediates native hands the composite effect.
// A composite group's buffer is drawn back onto the device the way native draws a layer back. An axis-aligned
// draw-back (a sub-pixel shift, or a scale per axis) runs DrawBitmap's 16.16 fixed-point texcoord walk with the 8-bit
// row-then-column lerp over the layer cut to the group's content, anchored at its edge (VM jit-dumps of the
// draw-back; probes: shifted upright groups 4-30 px -> 0-2, anisotropic groups 6-48 -> 0).

/// The group's content: its non-transparent texels' bounds `[l, t, r, b]` and those texels as their own texture.
fn group_layer(buf: &PixelBuffer) -> Option<([usize; 4], Vec<u8>)> {
    let (w, h, st) = (buf.desc.width as usize, buf.desc.height as usize, buf.stride);
    let (mut l, mut t, mut r, mut b) = (w, h, 0, 0);
    for y in 0..h {
        for x in 0..w {
            if buf.data[y * st + x * 4 + 3] != 0 {
                (l, t, r, b) = (l.min(x), t.min(y), r.max(x + 1), b.max(y + 1));
            }
        }
    }
    (r > l && b > t).then(|| ([l, t, r, b], (t..b).flat_map(|y| buf.data[y * st + l * 4..y * st + r * 4].iter().copied()).collect()))
}

/// `r` (`[l, t, r, b]`) mapped through `m`: the bounds of its four corners.
fn map_bounds(m: &Mat, r: [f32; 4]) -> [f32; 4] {
    let c = [apply(m, r[0], r[1]), apply(m, r[2], r[1]), apply(m, r[0], r[3]), apply(m, r[2], r[3])];
    let lo = [0, 1].map(|k| c.iter().map(|p| p[k]).fold(f32::INFINITY, f32::min));
    let hi = [0, 1].map(|k| c.iter().map(|p| p[k]).fold(f32::NEG_INFINITY, f32::max));
    [lo[0], lo[1], hi[0], hi[1]]
}

/// A turned group drawn back as native draws it. Direct2D fills the group as an anisotropic image brush, so its
/// layer is the brush's need rect, and the draw walks the brush's device rect:
/// - the device rect is the group's leaf bounds, rounded out, grown by half a texel and mapped to the device, then
///   cut to the target;
/// - the layer is that rect mapped back from its pixel centres and grown by a texel;
/// - the layer's first texel is that rect's corner in a 512-texel atlas. It holds the group's texels inside the leaf
///   bounds, plus a one-texel border on each side the rect cuts when both sides are under 256.
///
/// The draw samples the layer through the alpha-blit float plane. VM captures of the layer and allocation rects on
/// five turned and sheared groups match this exactly, as do the texel planes traced from the draw.
fn turned_back(buf: &PixelBuffer, to_buf: &Mat, leaves: [f32; 4], desc: ImageDesc) -> PixelBuffer {
    let (ow, oh) = (desc.width as i32, desc.height as i32);
    let mut out = PixelBuffer { desc, stride: ow as usize * 4, data: vec![0u8; (ow * oh) as usize * 4] };
    if !(leaves[0] < leaves[2] && leaves[1] < leaves[3]) {
        return out;
    }
    let img = [leaves[0].floor() as i32, leaves[1].floor() as i32, leaves[2].ceil() as i32, leaves[3].ceil() as i32];
    let round_out = |r: [f32; 4], e: f32| {
        [(r[0] - e).round_ties_even() as i32, (r[1] - e).round_ties_even() as i32, (r[2] + e - 0.5).ceil() as i32, (r[3] + e - 0.5).ceil() as i32]
    };
    let grown = [img[0] as f32 - 0.5, img[1] as f32 - 0.5, img[2] as f32 + 0.5, img[3] as f32 + 0.5];
    let d = round_out(map_bounds(&invert(to_buf), grown), 0.0);
    let dev = [d[0].max(0), d[1].max(0), d[2].min(ow), d[3].min(oh)];
    if dev[2] <= dev[0] || dev[3] <= dev[1] {
        return out;
    }
    let centres = [dev[0] as f32 + 0.5, dev[1] as f32 + 0.5, dev[2] as f32 - 0.5, dev[3] as f32 - 0.5];
    let outer = round_out(map_bounds(to_buf, centres), 1.0);
    let content = [outer[0].max(img[0]), outer[1].max(img[1]), outer[2].min(img[2]), outer[3].min(img[3])];
    let small = content[2] - content[0] < 256 && content[3] - content[1] < 256;
    let border = |cut: bool| i32::from(small && cut);
    let valid = [
        content[0] - border(content[0] > outer[0]),
        content[1] - border(content[1] > outer[1]),
        content[2] + border(content[2] < outer[2]),
        content[3] + border(content[3] < outer[3]),
    ];
    let (tw, th) = ((outer[2] - outer[0]).max(0) as usize, (outer[3] - outer[1]).max(0) as usize);
    let mut data = vec![0u8; tw * th * 4];
    let (bw, bh) = (buf.desc.width as i32, buf.desc.height as i32);
    for y in valid[1].max(0).max(outer[1])..valid[3].min(bh).min(outer[3]) {
        let (x0, x1) = (valid[0].max(0).max(outer[0]), valid[2].min(bw).min(outer[2]));
        if x1 <= x0 {
            break;
        }
        let src = y as usize * buf.stride + x0 as usize * 4;
        let dst = ((y - outer[1]) as usize * tw + (x0 - outer[0]) as usize) * 4;
        data[dst..dst + (x1 - x0) as usize * 4].copy_from_slice(&buf.data[src..src + (x1 - x0) as usize * 4]);
    }
    let tex = crate::sampling::Texture { data: &data, width: tw as u32, height: th as u32 };
    let m = [to_buf[0], to_buf[1], to_buf[2], to_buf[3], to_buf[4] - outer[0] as f32, to_buf[5] - outer[1] as f32];
    // ponytail: a layer past 512 texels takes the next power of two (unprobed; d2d1's atlas refuses it).
    let size = (tw.max(th) as u32).max(512).next_power_of_two();
    let ex = [crate::sampling::Extend::Transparent; 2];
    crate::sampling::alphablt_float_iter(&m, [size, size], dev.map(|v| v as usize), oh as u32, |x, y, qx, qy| {
        let i = (y * ow as usize + x) * 4;
        out.data[i..i + 4].copy_from_slice(&tex.sample_alphablt_q8(ex, qx, qy, false).map(|v| v.clamp(0, 255) as u8));
    });
    out
}

/// The group drawn back onto a `desc`-sized grid through the axis-aligned `to_buf` (device point → `buf`
/// coordinates); outside the group is transparent.
fn draw_back(buf: &PixelBuffer, to_buf: &Mat, desc: ImageDesc) -> PixelBuffer {
    let (ow, oh) = (desc.width as usize, desc.height as usize);
    let mut out = PixelBuffer { desc, stride: ow * 4, data: vec![0u8; ow * oh * 4] };
    let Some(([l, t, r, b], data)) = group_layer(buf) else { return out };
    let tex = crate::sampling::Texture { data: &data, width: (r - l) as u32, height: (b - t) as u32 };
    let ex = [crate::sampling::Extend::Transparent; 2];
    let m = [to_buf[0], to_buf[4] - l as f32, to_buf[3], to_buf[5] - t as f32];
    let (w, h) = (tex.width as f32, tex.height as f32);
    let (bx, sx) = crate::sampling::axis_fixed(0.0, w, -m[1] / m[0], (w - m[1]) / m[0]);
    let (by, sy) = crate::sampling::axis_fixed(0.0, h, -m[3] / m[2], (h - m[3]) / m[2]);
    for y in 0..oh {
        let uy = by.wrapping_add((y as i32).wrapping_mul(sy));
        for x in 0..ow {
            let v = tex.sample_fixed(ex, bx.wrapping_add((x as i32).wrapping_mul(sx)), uy);
            let i = (y * ow + x) * 4;
            out.data[i..i + 4].copy_from_slice(&v.map(|v| (v * 255.0).round_ties_even().clamp(0.0, 255.0) as u8));
        }
    }
    out
}

/// `buf` moved by the sub-pixel `f` (each in [0, 1)): out(x) ≈ (1−f)·in(x) + f·in(x−1) per axis.
fn bilinear_shift(buf: &PixelBuffer, f: [f32; 2]) -> PixelBuffer {
    draw_back(buf, &[1.0, 0.0, 0.0, 1.0, -f[0], -f[1]], buf.desc)
}

fn composite_buf(src: &PixelBuffer, dst: &mut PixelBuffer, mode: u8) {
    let (w, h, st) = (dst.desc.width as usize, dst.desc.height as usize, dst.stride);
    for y in 0..h {
        for x in 0..w {
            let i = y * st + x * 4;
            let s = [src.data[i], src.data[i + 1], src.data[i + 2], src.data[i + 3]];
            let d = [dst.data[i], dst.data[i + 1], dst.data[i + 2], dst.data[i + 3]];
            dst.data[i..i + 4].copy_from_slice(&composite_pixel(mode, s, d));
        }
    }
}

/// Merge a clip layer down: the subtree's premultiplied `src` modulated by the clip glyph's per-pixel
/// coverage `mask` (`w·h` alpha), source-over `dst` — the same integer composite [`fill_geometry`]
/// uses, so a clipped leaf is byte-identical to an unclipped one wherever the mask is 255.
/// WARP's gradient-mask UNORM8 product `v·m/255`: the high-byte correction is taken before the
/// +128 rounding bias (unlike [`unorm_mul`], which folds the bias in first).
fn warp_unorm_mul(v: i32, m: i32) -> i32 {
    let p = v * m;
    (p + (p >> 8) + 128) >> 8
}

fn clip_down(src: &PixelBuffer, dst: &mut PixelBuffer, mask: &[u8]) {
    let (w, h, st) = (dst.desc.width as usize, dst.desc.height as usize, dst.stride);
    for y in 0..h {
        for x in 0..w {
            let m = mask[y * w + x] as i32;
            if m == 0 {
                continue;
            }
            let i = y * st + x * 4;
            let sc: [i32; 4] = core::array::from_fn(|k| unorm_mul(src.data[i + k] as i32, m));
            let ia = 255 - sc[3];
            let out: [u8; 4] = core::array::from_fn(|k| (sc[k] + unorm_mul(dst.data[i + k] as i32, ia)).clamp(0, 255) as u8);
            dst.data[i..i + 4].copy_from_slice(&out);
        }
    }
}

/// Draw the `scale`×`scale` supersampled premultiplied paint-tree bitmap `src` back onto `dst`,
/// minified `1/scale` and composited source-over — native's `DrawingContext::GetPaintTreeGlyphImage`
/// returns the supersampled bitmap (at a `scale`× DPI) and an image→device transform, and the caller
/// draws it with `DrawImage(interpolationMode = LINEAR, SOURCE_OVER)` (the
/// `GetPaintTreeGlyphImage` → `SetTransform` → `DrawImage(…, 1, 0)` lambda). For an aligned power-of-two
/// minify WARP's `DrawImage` LINEAR is a **single flat `scale²` box average, round-half-up**
/// (`(sum + scale²/2) / scale²`) — *not* a two-level 2×2 mip (whose intermediate rounding inflates
/// e.g. `colorfont2-03` (26,17) to 202 vs native 201) nor a 2-tap bilinear. Pinned by a WARP
/// `minify-probe` capture (VM capture): a known 64×64 bitmap `DrawImage`-minified ×1/4 reads back the
/// exact box average, with .5 ties rounding **up** (avg 4.5→5, 8.5→9). Then source-over onto `dst` with
/// the same integer premultiplied composite the leaves use (`crate::bitmap::unorm_mul`).
fn downsample_over(src: PixelBuffer, dst: &mut PixelBuffer, scale: u32, blend: Blend) {
    let (sw, sc, n) = (src.desc.width as usize, scale as usize, (scale * scale) as i32);
    let (dw, dh, dstride) = (dst.desc.width as usize, dst.desc.height as usize, dst.stride);
    for y in 0..dh {
        for x in 0..dw {
            let mut sum = [0i32; 4];
            for j in 0..sc {
                for ii in 0..sc {
                    let o = ((y * sc + j) * sw + x * sc + ii) * 4;
                    for c in 0..4 {
                        sum[c] += src.data[o + c] as i32;
                    }
                }
            }
            let avg = sum.map(|s| (s + n / 2) / n);
            if crate::float::is_highp(dst.desc.format) {
                // A float target takes the byte paint-tree image source-over in f32 (VM: a COLR glyph on an R32 float
                // target is within 1/255 of the BGRA8 one, sc29 scenes 23/24).
                if avg[3] != 0 || avg[..3].iter().any(|&v| v != 0) {
                    let src = [avg[2], avg[1], avg[0], avg[3]].map(|v| v as f32 / 255.0);
                    crate::float::FloatTarget::new(dst).blend_px(x, y, src, 64, Blend::SourceOver);
                }
                continue;
            }
            let i = y * dstride + x * 4;
            // Under ADD or MAX the painted image composites under the primitive blend, as an SVG glyph's does.
            if matches!(blend, Blend::Add | Blend::Max) {
                if avg != [0; 4] {
                    let d = [dst.data[i], dst.data[i + 1], dst.data[i + 2], dst.data[i + 3]];
                    let out = crate::pixels::composite(d, avg.map(|v| v.clamp(0, 255) as u8), 64, blend);
                    dst.data[i..i + 4].copy_from_slice(&out);
                }
                continue;
            }
            let ia = 255 - avg[3];
            let out: [u8; 4] = core::array::from_fn(|c| (avg[c] + unorm_mul(dst.data[i + c] as i32, ia)).clamp(0, 255) as u8);
            dst.data[i..i + 4].copy_from_slice(&out);
        }
    }
}

/// Snap the run's device origin to the `ss`× sub-pixel grid in x and the `ss_y`× grid in y (the
/// paint-tree image's render origin is `round(deviceOrigin · ss) / ss`, and native bakes the residual
/// into the render matrix — here it is folded straight into the base transform). `o` is the DIP origin;
/// returns the snapped DIP origin. A rotated origin snaps in device space and is mapped back through the
/// transform; a quarter turn is left unsnapped.
fn snap_origin(o: [f32; 2], dt: &[f64; 6], ss: u32, ss_y: u32) -> [f32; 2] {
    // The device origin is formed and rounded to the grid in f32, half up (VM: a DrawText run from x 2.3 DIP at 120 dpi
    // lands at device 2.875 exactly in f32 and snaps to 3.0, where the f64 product 2.87499994 snapped to 2.75; a run at
    // device x 54.625 snaps to 54.75, not to the even 54.5).
    let f = dt.map(|v| v as f32);
    let snap = |d: f32, s: u32| (d * s as f32 + 0.5).floor() as f64 / s as f64;
    // A rotated run snaps its device origin on the same grids, x and y each on its own, and the paint image is drawn
    // from there (VM, a rotated two-rect composite: device origins 115.416, 87.257 and 138.499, 104.708 put the
    // leaves at 115.5, 87.25 and 138.5, 104.75). Here a grid of 1 is whole pixels.
    let det = dt[0] * dt[3] - dt[1] * dt[2];
    if dt[2] != 0.0 && dt[0] != 0.0 && dt[3] != 0.0 && det != 0.0 {
        let x = snap(f[0] * o[0] + f[2] * o[1] + f[4], ss) - dt[4];
        let y = snap(f[1] * o[0] + f[3] * o[1] + f[5], ss_y) - dt[5];
        return [((x * dt[3] - y * dt[2]) / det) as f32, ((y * dt[0] - x * dt[1]) / det) as f32];
    }
    // Snapped whenever device x follows the origin's x alone (no m21), as under a vertical skew: x first, then
    // y on its own grid from the snapped x (r25 30565, m12 0.7: native's paint image puts the origin on the
    // quarter grid in both axes).
    if ss <= 1 || dt[0] == 0.0 || dt[3] == 0.0 || dt[2] != 0.0 {
        return o;
    }
    let x = ((snap(f[0] * o[0] + f[4], ss) - dt[4]) / dt[0]) as f32;
    let k = if dt[1] == 0.0 { f[5] } else { f[1] * x + f[5] };
    [x, ((snap(f[3] * o[1] + k, ss_y) - k as f64) / dt[3]) as f32]
}

/// Paint a colour glyph run. Returns `true` when every glyph was painted, `false` when a glyph needs
/// a path not reproduced (sweep gradient, composite/blend layer, or a non-colour glyph whose
/// monochrome rendering is itself unimplemented) so the caller reports it as unsupported. The
/// monochrome glyphs draw with `fg`, placed from `mono` when given.
#[allow(clippy::too_many_arguments)]
pub fn render_color_glyph_run(
    target: &mut PixelBuffer,
    dt: &[f64; 6],
    run: &GlyphRun,
    fg: TextFg,
    palette_index: u32,
    aa: u32,
    params: &RenderingParams,
    mono: Option<&GlyphRun>,
) -> bool {
    // A DrawText run's monochrome glyphs draw from its own baseline, with its snapping (`mono`), as the plain run
    // would; a direct colour run's from the colour origin.
    let mono_shift = mono.map(|m| ([m.baseline_origin[0] - run.baseline_origin[0], m.baseline_origin[1] - run.baseline_origin[1]], m.no_snap));
    // A right-to-left run draws as its left-to-right equivalent from its left edge, as the monochrome path does
    // (r25 30373: RTL Arabic with ENABLE_COLOR_FONT); Direct2D places colour glyphs by DirectWrite's glyph origins
    // under the device transform, whatever the size.
    let mut rtl = Vec::new();
    let run = &super::drawn_ltr(run, dt, &mut rtl);
    let Ok(font) = FontRef::from_index(run.font_data, run.face_index) else { return false };
    let colr = ColorGlyphCollection::new(&font);
    // CPAL palette (index clamped to 0 when out of range, as native falls back); absent ⇒ empty.
    let palettes = font.color_palettes();
    let palette = palettes.get(palette_index as u16).or_else(|| palettes.get(0));
    // Native uses a reciprocal multiply, not a divide.
    let colors: Vec<[f32; 4]> = palette
        .map(|p| p.colors().iter().map(|c| [c.red, c.green, c.blue, c.alpha].map(|v| v as f32 * (1.0 / 255.0))).collect())
        .unwrap_or_default();

    let upem = font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f32;
    let s = run.em_size / upem;
    // The colour glyphs ignore the aliased text mode and render as under grayscale (VM: Segoe UI Emoji at em 60,
    // aliased byte-identical to grayscale for every weight and style); the run's monochrome glyphs stay aliased.
    let mono_aa = aa;
    let aa_mode = Aa::PerPrimitive;

    // Native renders the COLRv1 paint tree into an intermediate bitmap supersampled by the
    // rendering-mode's oversample (`dpiScale·8` for
    // NATURAL, `·4` for NATURAL_SYMMETRIC, `·1` for aliased/GDI). The mode is
    // DirectWrite's recommended rendering mode: for the natural antialiased path a device em
    // of **≤ 20 px → NATURAL (×8)**, **> 20 px → NATURAL_SYMMETRIC (×4)** (mode picked on round(deviceEm) > 20),
    // so small emoji get the finer ×8 grid. The supersample keeps each paint's gradient footprint ≥1
    // device texel, so the many tiny localized highlight/shadow gradients (radii a fraction of a device
    // pixel at em 12) realize with real spatial variation instead of collapsing to a flat wash; the
    // `1/ss` mip downsample then produces the glyph's edge/interior AA.
    // The device em is the em under the transform's larger singular value, so a turned or sheared run measures its
    // true size (VM probe-mf: runs turned 30-45 degrees at em 62-90 and 144-240 dpi, past 128 px only by this
    // measure, are 208-837 px off on the paint-image path and 0-14 drawn straight; r25 30460_10, a shear).
    let device_em = run.em_size * bitmap::max_factor([dt[0] as f32, dt[1] as f32, dt[2] as f32, dt[3] as f32]);
    // Native only takes the supersampled paint-tree-image path when
    // `device_em·MaxFactor·dpiScale ≤ 128` on both axes; above that it draws the leaves straight to the
    // target at device resolution (no intermediate, no minify). `> 128` is that gate.
    // Below it: NATURAL ×8 for ≤ 20 px, NATURAL_SYMMETRIC ×4 above.
    // [DEVICE-RES TEST — colorfont2 needs the device-res single sample like the ordinary gradient fill.]
    let snap_ss: u32 = if device_em > 128.0 {
        1
    } else if device_em.round() > 20.0 {
        4
    } else {
        8
    };
    // The paint-tree image is placed on its own sub-pixel grid (`snap_origin`): x on a quarter-pixel grid, y on
    // a quarter-pixel grid at either supersample, unless the rendering params set a non-symmetric mode. The
    // baseline then does not first snap to a whole device pixel (VM, COLR 'C' at em 50: baseline y 67.25 and
    // 67.75 render alike, 67.4993 as 67.5, and none as 67). With NATURAL or a GDI mode set in the params, the
    // baseline y snaps to a whole pixel as a monochrome run's does.
    let quarter_y = snap_ss > 1 && matches!(params.rendering_mode, 0 | 5 | 7);
    // The rendering mode DirectWrite settles on decides too (VM probe-colr, a COLR 'C' at baselines x 22.0..22.9):
    // an OUTLINE run is placed where its baseline falls, x and y, and a GDI mode snaps x to a whole pixel (r25 30249,
    // under GDI_NATURAL params a glyph run left behind). The colour run's OUTLINE cut-off measures the em under the
    // world transform alone, without the DPI scale (r25 30523: em 74 at 144 dpi keeps the quarter grid; 30240: em 105
    // at 144 dpi is outline).
    // With the default rendering mode a GDI-measured run settles on the matching GDI mode, as the monochrome path's
    // recommended mode does (r25 30371_4: Segoe UI Emoji under GDI_CLASSIC measuring, 191 px -> 0).
    let mode = if params.rendering_mode != super::MODE_DEFAULT {
        params.rendering_mode
    } else if run.measuring_mode == 1 {
        super::MODE_GDI_CLASSIC
    } else if run.measuring_mode == 2 {
        super::MODE_GDI_NATURAL
    } else {
        let k = params.dpi[0].max(params.dpi[1]) / 96.0;
        let world_em = run.em_size * bitmap::max_factor([dt[0] as f32, dt[1] as f32, dt[2] as f32, dt[3] as f32]) / k;
        // An aliased run below the outline size is placed as the GDI modes place it, on whole pixels in x (VM: Segoe UI
        // Emoji DrawText at em 20 to 60, aliased: 530-1785 px off placed as grayscale, 0-9 this way; r25 30233_5).
        if world_em >= 100.0 { super::MODE_OUTLINE } else if aa == super::AA_ALIASED { super::MODE_GDI_CLASSIC } else { super::MODE_DEFAULT }
    };
    // So is a run past 128 device px, whose leaves draw straight to the target (r25 30278, 30138).
    let unsnapped = mode == super::MODE_OUTLINE || device_em > 128.0;
    // Each colour glyph's paint image is placed at its own origin snapped to the quarter grid in x as in y, at
    // either supersample (VM probe: four glyphs at em 8.07-50.4 from x 3.0 and 3.53, and r25 30691, where a second
    // glyph whose pen sits at x 41.148 draws at 41.25), so every glyph pen goes through `place`.
    let snapped = !(unsnapped && !run.no_snap);
    let pre = if !snapped || quarter_y { run.baseline_origin } else { snap_baseline(run.baseline_origin, dt, run.no_snap) };
    let gdi_mode = matches!(mode, super::MODE_GDI_CLASSIC | super::MODE_GDI_NATURAL) && snap_ss > 1 && !run.no_snap;
    let gdi = gdi_mode && dt[1] == 0.0 && dt[2] == 0.0 && dt[0] != 0.0;
    // Rotated, a GDI mode puts the origin on a whole device pixel in x and in y (r25 30179_8, an aliased rotated emoji:
    // 105 px off on the quarter grid, 1 on whole pixels).
    let whole = gdi_mode && dt[2] != 0.0;
    let place = |o: [f32; 2]| {
        if !snapped {
            return o;
        }
        let o = if gdi { [(((dt[0] * o[0] as f64 + dt[4]).round() - dt[4]) / dt[0]) as f32, o[1]] } else { o };
        snap_origin(o, dt, if gdi || whole { 1 } else { snap_ss.min(4) }, if whole { 1 } else if quarter_y && !run.no_snap { 4 } else { snap_ss })
    };
    let origin = place(pre);
    let (tw, th) = (target.desc.width, target.desc.height);
    // Leaves render at device resolution (ss = 1): > 128 fills each leaf's hinted outline through the
    // the AA tessellator feather (`Painter::feather`), ≤ 128 scans it with the font-scaler grayscale converter —
    // both already produce the device-pixel coverage (no supersampled intermediate / minify confound).
    // An unsnapped run's composite groups are drawn back bilinearly (`Painter::push_layer`), which reads one pixel
    // above and left of each: paint with that margin and crop it at the end.
    let composites = run.glyphs.iter().any(|g| {
        colr.get(GlyphId::new(g.id as u32)).is_some_and(|cg| {
            let mut p = HasComposite(false);
            let _ = cg.paint(LocationRef::default(), &mut p);
            p.0
        })
    });
    // Axis-aligned, flipped or quarter-turned only: a rotated run's groups are not redrawn this way (r25 30019).
    let tiny = |a: f64, b: f64| a.abs() <= 1e-6 * b.abs();
    let square = (tiny(dt[1], dt[0]) && tiny(dt[2], dt[3])) || (tiny(dt[0], dt[1]) && tiny(dt[3], dt[2]));
    let regroup = unsnapped && composites && square;
    // Under any other world transform keeping orientation (a rotation, a shear, a scale) a group renders in world
    // space at the DPI scale and is drawn back through the world transform bilinearly instead (`Painter::push_layer`;
    // VM probes: Xor and Multiply groups under [2,0,0,0.8], [0.8,0,0,1.6], 2x turned 30 degrees and 0.7x turned 20
    // at 144 dpi all match to +-1 this way, 812-1623 px off drawn straight; r25 30126_10, a shear at 144 dpi).
    let world = [dt[0] / (params.dpi[0] / 96.0) as f64, dt[1] / (params.dpi[0] / 96.0) as f64, dt[2] / (params.dpi[1] / 96.0) as f64, dt[3] / (params.dpi[1] / 96.0) as f64];
    let unit = (world[0].abs() + world[1].abs() - 1.0).abs() < 1e-6 && (world[2].abs() + world[3].abs() - 1.0).abs() < 1e-6;
    let turn = (unsnapped && composites && !(square && unit) && dt[0] * dt[3] - dt[1] * dt[2] > 0.0 && world.iter().all(|v| v.is_finite()))
        .then(|| [world[0] as f32, world[1] as f32, world[2] as f32, world[3] as f32, 0.0, 0.0]);
    let regroup = regroup && turn.is_none();
    // The paint-tree image renders into its own bitmap, 2 px past each glyph's whole-pixel paint bounds, so a leaf at
    // the target's edge is not clipped there (VM: r25 30620's 'dbf' at em 99 under 120 dpi, a feathered leaf crossing
    // the target's top row). Paint with a margin that holds every glyph's bitmap and crop it at the end.
    let bitmap_margin = || {
        let dt0: Mat = [dt[0] as f32, dt[1] as f32, dt[2] as f32, dt[3] as f32, dt[4] as f32, dt[5] as f32];
        let oblique = if run.simulations & 2 != 0 && run.simulations & 1 == 0 { super::raster::OBLIQUE_SHEAR as f32 } else { 0.0 };
        let mut x = pre[0];
        let mut need = 0.0f32;
        for g in run.glyphs {
            let pen = place([x + g.offset[0], pre[1] - g.offset[1]]);
            x += g.advance;
            let Some(cg) = colr.get(GlyphId::new(g.id as u32)) else { continue };
            let mut b = PaintBounds { stack: vec![mul(&dt0, &[s, 0.0, oblique * s, -s, pen[0], pen[1]])], font: &font, acc: None };
            let _ = cg.paint(LocationRef::default(), &mut b);
            // Only a bitmap reaching the target matters; the rest of it is cropped away.
            let Some(r) = b.acc else { continue };
            let (l, t, rr, bb) = (r[0].floor() - 2.0, r[1].floor() - 2.0, r[2].ceil() + 2.0, r[3].ceil() + 2.0);
            if rr <= 0.0 || bb <= 0.0 || l >= tw as f32 || t >= th as f32 {
                continue;
            }
            need = need.max(-l).max(-t).max(rr - tw as f32).max(bb - th as f32);
        }
        if need.is_finite() { need.clamp(0.0, 4096.0) as u32 } else { 0 }
    };
    let margin: u32 = if regroup { 1 } else if !unsnapped { bitmap_margin() } else { 0 };
    let mf = margin as f32;
    let dt_ss: Mat = [dt[0] as f32, dt[1] as f32, dt[2] as f32, dt[3] as f32, dt[4] as f32 + mf, dt[5] as f32 + mf];
    let (bw, bh) = (tw + 2 * margin, th + 2 * margin);
    let mut buf = PixelBuffer {
        desc: ImageDesc { width: bw, height: bh, format: d2d_types::abi::DXGI_FORMAT_B8G8R8A8_UNORM, ..target.desc },
        stride: (bw * 4) as usize,
        data: vec![0u8; bw as usize * bh as usize * 4],
    };

    let mut pen_x = origin[0];
    let mut raw_x = pre[0];
    let mut ok = true;
    // Consecutive glyphs with no colour data draw as one monochrome run starting at pen x `mono_x`.
    let (mut mono_glyphs, mut mono_x) = (Vec::new(), 0.0f32);
    let flush = |target: &mut PixelBuffer, glyphs: &mut Vec<super::Glyph>, x: f32| -> bool {
        if glyphs.is_empty() {
            return true;
        }
        let sub = match mono_shift {
            Some((d, no_snap)) => GlyphRun { glyphs, baseline_origin: [run.baseline_origin[0] + d[0] + x - origin[0], run.baseline_origin[1] + d[1]], no_snap, ..*run },
            None => GlyphRun { glyphs, baseline_origin: [x, origin[1]], ..*run },
        };
        // A singular brush voids the monochrome glyphs, as it voids a glyph run; the colour glyphs still paint.
        let drawn = matches!(fg, TextFg::Brush(b) if b.singular()) || render_glyph_run_fg(target, dt, &sub, fg, mono_aa, params);
        glyphs.clear();
        drawn
    };
    for g in run.glyphs {
        let gid = GlyphId::new(g.id as u32);
        let raw_pen = [pen_x + g.offset[0], origin[1] - g.offset[1]];
        let pen = place([raw_x + g.offset[0], pre[1] - g.offset[1]]);
        pen_x += g.advance;
        raw_x += g.advance;
        match colr.get(gid) {
            Some(cg) => {
                ok &= flush(target, &mut mono_glyphs, mono_x);
                // font design units (y-up) → DIP → supersampled device; the −s flips to device y-down. The oblique
                // simulation alone slants the whole paint in design units, x += k·y; with bold it slants only the
                // leaves (`Painter::shear`).
                let oblique = if run.simulations & 2 != 0 { super::raster::OBLIQUE_SHEAR as f32 } else { 0.0 };
                let (k, shear) = if run.simulations & 1 != 0 { (0.0, oblique) } else { (oblique, 0.0) };
                let m0 = mul(&dt_ss, &[s, 0.0, k * s, -s, pen[0], pen[1]]);
                // Only the paint-tree image (device em ≤ 128) renders into a bitmap; above it the leaves draw straight
                // to the target (r25 30161, em 181 px under a rotation: re-framed it gains a pixel).
                let paint_origin = if device_em > 128.0 { [0.0; 2] } else {
                    let mut b = PaintBounds { stack: vec![m0], font: &font, acc: None };
                    let _ = cg.paint(LocationRef::default(), &mut b);
                    // DWrite derives these bounds on whole device pixels (the leaf boxes rounded out), and the bitmap
                    // starts 2 px before them (VM paint-bitmap dump of r25 30523: 94×72 at (29, 13) for leaf bounds
                    // 31.65..120.8 × 15.44..82.5).
                    b.acc.map_or([0.0; 2], |r| [r[0].floor() - 2.0, r[1].floor() - 2.0])
                };
                let mut p = Painter {
                    target: &mut buf,
                    font: &font,
                    colors: &colors,
                    // ponytail: a foreground (0xFFFF) paint under a non-solid text brush paints black; no system
                    // colour font has one (Segoe UI Emoji: none). Sample the brush if a font needs it.
                    fg: match fg { TextFg::Solid(c) => c, TextFg::Brush(_) => [0.0, 0.0, 0.0, 1.0] },
                    stack: vec![m0],
                    frames: Vec::new(),
                    aa: aa_mode,
                    unsupported: false,
                    // device_em > 128 ⇒ native draws each leaf straight to the target at device
                    // resolution in OUTLINE mode: fill the hinted outline through the AA tessellator
                    // feather. ≤ 128 takes the supersampled fsc-scan paint-tree-image path instead.
                    feather: !matches!(aa_mode, Aa::Aliased) && device_em > 128.0, run_feather: None,
                    clip: Vec::new(),
                    hints: RefCell::new(HashMap::new()),
                    leaf_idx: 0,
                    pending_linear: None,
                    bold: run.simulations & 1 != 0,
                    origin_dev: [m0[4], m0[5]],
                    regroup,
                    turn,
                    canvas: None, cell: None, next_cell: None, next_src: None, next_pd: false, vp_origins: None,
                    paint_origin,
                    // d2d1 records the paint tree under the em size times the world 2x2 at the DPI scale, translated
                    // only by the origin's sub-pixel phase: the origin goes through the world, snaps to its grids, and
                    // its whole device pixels are split off and drawn back with the image.
                    // Only the paint-tree image records this way; an unsnapped run (OUTLINE, or past 128 px) is recorded as a
                    // command list in em units and replayed under the world ([`Painter::replay_em`]). An oblique-only face
                    // slants the em matrix itself (em, 0, −k·em, em), as d2d1 multiplies in the simulation's shear.
                    em_ops: Vec::new(), pending_em: None, pending_brush_em: None, pending_radial: None,
                    // Past the paint-tree size (or in OUTLINE mode) d2d1 records the paint under the em matrix alone and draws
                    // the list as an image under the glyph origin's translation times the world: each recorded fill replays
                    // as ((C·T)·S)·S⁻¹ in f32, the device transform that times the dpi scale S, and a brush realizes through
                    // that ([`Painter::em_world`]).
                    replay_em: (unsnapped && turn.is_none()).then(|| {
                        let (sx, sy) = ((params.dpi[0] / 96.0) as f32, (params.dpi[1] / 96.0) as f32);
                        let w = [(dt[0] / sx as f64) as f32, (dt[1] / sy as f64) as f32, (dt[2] / sx as f64) as f32, (dt[3] / sy as f64) as f32, (dt[4] / sx as f64) as f32, (dt[5] / sy as f64) as f32];
                        let em = run.em_size;
                        let o = [(raw_x - g.advance + g.offset[0]) + 0.0, (pre[1] - g.offset[1]) + 0.0];
                        let t_img = [w[0], w[1], w[2], w[3], (o[0] * w[0] + o[1] * w[2]) + w[4], (o[0] * w[1] + o[1] * w[3]) + w[5]];
                        ([em, 0.0, -k * em, em, 0.0, 0.0], t_img, [sx, sy], mf)
                    }),
                    em_base: (!unsnapped).then(|| {
                        let (sx, sy) = ((params.dpi[0] / 96.0) as f32, (params.dpi[1] / 96.0) as f32);
                        let w = [(dt[0] / sx as f64) as f32, (dt[1] / sy as f64) as f32, (dt[2] / sx as f64) as f32, (dt[3] / sy as f64) as f32, (dt[4] / sx as f64) as f32, (dt[5] / sy as f64) as f32];
                        let em = run.em_size;
                        let raw = [raw_x - g.advance + g.offset[0], pre[1] - g.offset[1]];
                        let (ox, oy) = ((w[2] * raw[1] + w[0] * raw[0]) + w[4], (w[3] * raw[1] + w[1] * raw[0]) + w[5]);
                        let rhu = |v: f32| { let r = v.round_ties_even(); if r - v == -0.5 { r + 1.0 } else { r } };
                        let (gx, gy) = (if gdi || whole { 1 } else { snap_ss.min(4) }, if whole { 1 } else if quarter_y && !run.no_snap { 4 } else { snap_ss });
                        let (ux, uy) = (sx * gx as f32, sy * gy as f32);
                        let (ox, oy) = if snapped { (rhu(ox * ux) / ux, rhu(oy * uy) / uy) } else { (ox, oy) };
                        let (ix, iy) = ((ox * sx).floor(), (oy * sy).floor());
                        let (px, py) = ((ox - ix / sx) * sx, (oy - iy / sy) * sy);
                        let g2 = -k * em;
                        ([(em * w[0] + 0.0 * w[2]) * sx, (w[1] * em + 0.0 * w[3]) * sy, (g2 * w[0] + em * w[2]) * sx, (em * w[3] + g2 * w[1]) * sy, px, py], [ix + mf, iy + mf])
                    }),
                    raw_origin_dev: {
                        // The glyph origin before the paint-tree image's sub-pixel snap.
                        let p = [raw_pen[0] + run.baseline_origin[0] - origin[0], raw_pen[1] + run.baseline_origin[1] - origin[1]];
                        [(dt[0] * p[0] as f64 + dt[2] * p[1] as f64 + dt[4]) as f32 + mf, (dt[1] * p[0] as f64 + dt[3] * p[1] as f64 + dt[5]) as f32 + mf]
                    },
                    shear,
                };
                if (!paint::draw(&font, gid, &mut p) && cg.paint(LocationRef::default(), &mut p).is_err()) || p.unsupported {
                    ok = false;
                }
                place_back(&mut buf, target, margin, params.blend);
            }
            // No colour data: draw the base outline monochrome on the target, as native falls back
            // (a plain glyph run, not the paint-tree image path — so no supersample).
            None => {
                if mono_glyphs.is_empty() {
                    mono_x = pen_x - g.advance;
                }
                mono_glyphs.push(*g);
            }
        }
    }
    ok &= flush(target, &mut mono_glyphs, mono_x);
    ok
}

/// A turned group's device-to-buffer map for an origin `org`: the inverse turn `ri`, then the buffer origin `lo`.
fn turn_to_buf(ri: &Mat, lo: [f32; 2], org: [f32; 2]) -> Mat {
    let t = apply(ri, -org[0], -org[1]);
    [ri[0], ri[1], ri[2], ri[3], t[0] - lo[0], t[1] - lo[1]]
}

/// Composite one colour glyph's painted buffer onto the target and clear it for the next glyph: native draws each
/// glyph's paint-tree image (or its leaves) onto the target in turn, so overlapping glyphs are not pre-blended.
fn place_back(buf: &mut PixelBuffer, target: &mut PixelBuffer, margin: u32, blend: Blend) {
    // Diagnostic: dump our device-res paint-tree buffer (premultiplied BGRA, same shape as native's
    // D2D_DUMP_COLRBITMAP) before the source-over place-back, so it can be diffed against native's
    // paint-tree bitmap to classify an emoji residual as coverage (alpha) vs colour vs composite.
    #[cfg(not(target_arch = "wasm32"))]
    if let Ok(path) = std::env::var("D2D_DUMP_OURPAINTTREE") {
        let _ = std::fs::write(&path, &buf.data);
        eprintln!("OURPAINTTREE {}x{} -> {path}", buf.desc.width, buf.desc.height);
    }
    let (tw, th) = (target.desc.width, target.desc.height);
    let len = buf.data.len();
    let painted = if margin == 0 { std::mem::replace(&mut buf.data, vec![0u8; len]) } else {
        let mut inner = vec![0u8; tw as usize * th as usize * 4];
        for y in 0..th as usize {
            let src = (y + margin as usize) * buf.stride + margin as usize * 4;
            inner[y * tw as usize * 4..(y + 1) * tw as usize * 4].copy_from_slice(&buf.data[src..src + tw as usize * 4]);
        }
        buf.data.fill(0);
        inner
    };
    let src = PixelBuffer { desc: ImageDesc { width: tw, height: th, ..buf.desc }, stride: (tw * 4) as usize, data: painted };
    downsample_over(src, target, 1, blend);
}

/// An open offscreen group on the paint-graph render stack, mirroring native's command-list
/// intermediates: `PaintComposite` renders its backdrop and source into separate `Layer`s and the
/// composite effect merges them (`RenderCompositeNode`); a glyph `Clip` renders a subtree into an
/// offscreen that is masked by the clip outline when popped (`RenderGlyphNode`). A `ClipBox`
/// similarly masks the completed group, so overlapping leaves share one fractional edge.
enum Frame {
    ClipBox { buf: PixelBuffer, rect: [f32; 4], quad: Option<[[f32; 2]; 4]> },
    /// `shift`: set on a glyph's outermost composite in an unsnapped run, the sub-pixel part of the glyph origin
    /// the group was rendered without; the group is drawn back bilinearly at that offset.
    /// `turn`: set instead on a turned run's outermost composite: the device→buffer map the upright group is
    /// sampled back through, and its leaves' bounds in the buffer so far.
    Layer { buf: PixelBuffer, mode: u8, shift: Option<[f32; 2]>, turn: Option<(Mat, [f32; 4])> },
    Clip { buf: PixelBuffer, mask: Vec<u8> },
}

/// Maps COLRv1 paint-graph callbacks onto the d2d core, mirroring `PaintTreeRenderer`. `stack` is the
/// transform stack (current = top) mapping font design units to device pixels; `frames` is the
/// offscreen group stack (empty ⇒ fills go straight to `target`).
struct Painter<'a> {
    target: &'a mut PixelBuffer,
    font: &'a FontRef<'a>,
    colors: &'a [[f32; 4]],
    fg: [f32; 4],
    stack: Vec<Mat>,
    frames: Vec<Frame>,
    aa: Aa,
    unsupported: bool,
    /// Leaves are filled as geometry through the AA tessellator feather (OUTLINE mode) rather than the
    /// fsc grayscale scan — native's choice at `device_em > 128` (see construction).
    feather: bool,
    /// The run's `feather` while a turned group, which decides by its own buffer size, is open.
    run_feather: Option<bool>,
    /// Stack of device-space clip rectangles (`[l,t,r,b]`), each the running intersection of all enclosing
    /// COLRv1 `ClipBox`es. The fractional coverage masks the completed group when its frame pops;
    /// applying it separately to overlapping leaves makes the clipped edge too opaque.
    clip: Vec<[f32; 4]>,
    /// TrueType hinting interpreters, one per distinct leaf device-em (the paint scale varies by
    /// subtree), reused across the glyph's leaves. See [`Painter::hinted_leaf`].
    hints: RefCell<HashMap<i64, HInstance<'a>>>,
    /// Diagnostic prefix-probe counter: index of the next leaf. `COLR_LEAF_LIMIT=k` skips leaves ≥ k.
    leaf_idx: u32,
    /// Raw COLR linear-gradient geometry `(p0, p1, p2, lo, hi)` (design units) for the leaf about to be
    /// filled, set by `paint::Node::draw`. `brush()` projects the p2 rotation from it as d2d1 does;
    /// `None` ⇒ the skrifa fallback's pre-resolved points.
    pending_linear: Option<([f32; 2], [f32; 2], [f32; 2], f32, f32)>,
    /// DWRITE_FONT_SIMULATIONS_BOLD: each leaf is a glyph run of the simulated face, so it is emboldened.
    bold: bool,
    /// The run is unsnapped (OUTLINE mode, or past 128 device px) and axis-aligned: its composite groups are drawn
    /// back bilinearly ([`Painter::push_layer`]).
    regroup: bool,
    /// The run's world transform (its device matrix without the DPI scale) when its composite groups are drawn in world
    /// space and mapped back (`push_layer`).
    turn: Option<Mat>,
    /// The size of the buffers a turned group renders into, while one is open (else the target's).
    canvas: Option<ImageDesc>,
    /// The outermost composite's atlas cell (`[l, t, r, b]` in buffer pixels) while its groups paint, and the bounds the
    /// paint tree hands over just before it opens one.
    cell: Option<[f32; 4]>,
    next_cell: Option<[f32; 4]>,
    /// A Porter-Duff composite's source bounds and kind, handed over with `next_cell`; and the backdrop's and source's
    /// viewport origins once its cells are set.
    next_src: Option<[f32; 4]>,
    next_pd: bool,
    vp_origins: Option<([f32; 2], [f32; 2])>,
    /// The glyph origin in device pixels, as placed (`origin_dev`) and before the sub-pixel snap (`raw_origin_dev`).
    origin_dev: [f32; 2],
    raw_origin_dev: [f32; 2],
    /// The paint-tree bitmap's origin when the glyph has no `ClipBox` ([`Painter::bitmap_origin`]).
    paint_origin: [f32; 2],
    /// The paint-tree transform as d2d1 builds it, in em units (y down) onto the glyph's phase frame, and that frame's
    /// whole-pixel device origin ([`Painter::brush`]).
    em_base: Option<(Mat, [f32; 2])>,
    /// An unsnapped run's recording and replay ([`Painter::em_world`]): the em matrix, the image transform (glyph origin
    /// times the world), the dpi scale and the buffer margin.
    replay_em: Option<(Mat, Mat, [f32; 2], f32)>,
    /// Each open paint transform in em units (y down) as DirectWrite hands it to d2d1, one per `stack` push.
    em_ops: Vec<Mat>,
    /// Our reader's em-unit matrix for the next `push_transform`, and for the next fill's brush transform.
    pending_em: Option<Mat>,
    pending_brush_em: Option<Mat>,
    /// Our reader's raw COLR radial (`c0`, `c1`, `r1` in design units, the stop domain's `hi`) for the next fill.
    pending_radial: Option<([f32; 2], [f32; 2], f32, f32)>,
    /// The oblique slant of each leaf's outline (x += shear·y in design units) under bold plus oblique. An
    /// oblique-only run slants its whole paint instead; under bold plus oblique the leaves, as glyph runs of
    /// the simulated face, are emboldened then slanted, while the gradients and the clip box stay upright
    /// (VM probe: a moon and a square emoji clipped at their upright clip box's right edge).
    shear: f32,
}

impl Painter<'_> {
    fn cur(&self) -> Mat {
        *self.stack.last().unwrap()
    }

    /// The current transform for a leaf's own outline: under bold plus oblique it is sheared in design units
    /// (`shear`), the paint transform is not. See [`Painter::shear`].
    /// A turned group's buffer: the device-to-buffer map for an origin, the buffer's origin and far corner in the upright
    /// frame, and the upright bounds of the target's preimage (what the draw-back reads).
    fn turn_frame(&self, r: Mat) -> (Mat, [f32; 2], [f32; 2], [f32; 4]) {
        let ri = invert(&r);
        let raw = self.raw_origin_dev;
        let (w, h) = (self.target.desc.width as f32, self.target.desc.height as f32);
        let up = |p: [f32; 2]| apply(&ri, p[0] - raw[0], p[1] - raw[1]);
        let corners = [up([0.0, 0.0]), up([w, 0.0]), up([0.0, h]), up([w, h])];
        let min = [0, 1].map(|k| corners.iter().map(|c| c[k]).fold(f32::INFINITY, f32::min));
        let max = [0, 1].map(|k| corners.iter().map(|c| c[k]).fold(f32::NEG_INFINITY, f32::max));
        let lo = min.map(|v| v.floor() - 2.0);
        let hi = max.map(|v| v.ceil() + 2.0);
        (ri, lo, hi, [min[0], min[1], max[0], max[1]])
    }

    /// A Porter-Duff composite's viewport origins: the backdrop draws straight into the output cell, which keeps the
    /// pixel before its bounds on an axis drawn back at a fraction (`ext`); the source renders into a texture of its own
    /// over its bounds cut to that cell, and is blended on after.
    fn pd_origins(&mut self, bounds: Option<[f32; 4]>, lo: [f32; 2], ext: [f32; 2]) -> Option<([f32; 2], [f32; 2])> {
        let (pd, src) = (std::mem::take(&mut self.next_pd), self.next_src.take());
        let (b, s, c) = (bounds?, src?, self.cell?);
        let back = [(b[0].floor() - ext[0]).max(lo[0]), (b[1].floor() - ext[1]).max(lo[1])];
        pd.then(|| (back, [s[0].floor().max(back[0]), s[1].floor().max(back[1])])).filter(|_| c[0] < c[2])
    }

    /// The matrix the next outermost group renders its contents with: the current one, with the glyph origin moved to its
    /// whole pixel on the regroup path or mapped into the upright buffer on the turn path ([`Self::push_layer`]).
    pub(super) fn group_frame(&self) -> Mat {
        let m = self.cur();
        if self.frames.iter().any(|f| matches!(f, Frame::Layer { .. })) {
            return m;
        }
        if self.regroup {
            let (o, r) = (self.origin_dev, self.raw_origin_dev);
            return [m[0], m[1], m[2], m[3], m[4] + (r[0].floor() - o[0]), m[5] + (r[1].floor() - o[1])];
        }
        if let Some(r) = self.turn {
            let (ri, lo, ..) = self.turn_frame(r);
            return mul(&turn_to_buf(&ri, lo, self.origin_dev), &m);
        }
        m
    }

    fn leaf_cur(&self) -> Mat {
        let m = self.cur();
        [m[0], m[1], m[2] + self.shear * m[0], m[3] + self.shear * m[1], m[4], m[5]]
    }

    // DirectWrite's antialiased outline threshold is 100 px at the leaf's largest singular scale,
    // including nested paint transforms. This is separate from the whole-run bitmap-size gate.
    fn leaf_outline(&self) -> bool {
        let upem = self.font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f32;
        !matches!(self.aa, Aa::Aliased)
            && !self.feather
            && crate::extras::print::compute_maximum_scale_factor(self.cur()) * upem > 100.0
    }

    /// The buffer fills currently land in: the top offscreen group, or `target` when none is open.
    fn active_buf(&mut self) -> &mut PixelBuffer {
        if let Some(Frame::Layer { buf, .. } | Frame::Clip { buf, .. } | Frame::ClipBox { buf, .. }) = self.frames.last_mut() {
            return buf;
        }
        &mut *self.target
    }

    /// The buffer size fills work in: the target's, or an open turned group's.
    fn canvas(&self) -> ImageDesc {
        self.canvas.unwrap_or(self.target.desc)
    }

    /// A transparent offscreen the size of the canvas (a new group's backing bitmap).
    fn new_buf(&self) -> PixelBuffer {
        let desc = self.canvas();
        PixelBuffer { desc, stride: desc.width as usize * 4, data: vec![0u8; desc.width as usize * desc.height as usize * 4] }
    }

    /// `glyph_id`'s outline coverage (one alpha byte per pixel, `w·h`) under the current transform —
    /// the clip mask a glyph `Clip` is intersected with when it pops.
    fn clip_mask(&self, glyph_id: GlyphId) -> Vec<u8> {
        let (w, h) = (self.canvas().width as usize, self.canvas().height as usize);
        let Some(polys) = self.outline_polys(glyph_id) else { return vec![0u8; w * h] };
        let mut cov = self.new_buf();
        // A glyph outline reaches the general rasterizer as a shape, never the rect special case's 16.8 corners, though
        // its contour may be a rectangle (r25 30063: a rect leaf's left edge 0.0003 px past an eighth-pixel sample).
        Target::new(&mut cov).fill_path_in(&polys, crate::pixels::FillMode::Winding, [255, 255, 255, 255], self.aa, Blend::SourceOver, true, false);
        let st = cov.stride;
        (0..h).flat_map(|y| (0..w).map(move |x| (y, x))).map(|(y, x)| cov.data[y * st + x * 4 + 3]).collect()
    }

    /// CPAL/foreground colour × stop alpha → straight RGBA (0..1).
    fn color(&self, palette_index: u16, alpha: f32) -> [f32; 4] {
        let c = if palette_index == PALETTE_FOREGROUND {
            self.fg
        } else {
            *self.colors.get(palette_index as usize).unwrap_or(&self.fg)
        };
        [c[0], c[1], c[2], c[3] * alpha]
    }

    fn stops(&self, cs: &[ColorStop]) -> Vec<(f32, [f32; 4])> {
        cs.iter().map(|s| (s.offset, self.color(s.palette_index, s.alpha))).collect()
    }

    /// Build the d2d brush for a COLR paint. `b2d` maps the brush's coordinate space to device. Sets
    /// `unsupported` (returning `None`) for paints the core cannot represent; a degenerate gradient
    /// also returns `None` (nothing to fill), matching native's skip.
    /// The device origin of the bitmap native renders the paint tree into: its bounds less 2, rounded half up. The
    /// bounds are the `ClipBox`, else the paint's own bounds (`paint_origin`, paint-tree images only). Gradient leaves are realized in that
    /// bitmap's frame, so their placement keeps its small coordinates (r25 30070, 30531, 30819: radial ±1 at 120 and
    /// 144 dpi with the brush built in absolute device space).
    fn bitmap_origin(&self) -> [f32; 2] {
        self.clip.first().map_or(self.paint_origin, |r| [(r[0]-1.5).floor(), (r[1]-1.5).floor()])
    }

    /// d2d1's paint transform from em units onto the frame whose origin is `tile`, with the open transform nodes and
    /// the fill's brush transform (`brush_em`) multiplied in as `Matrix3x2F::SetProduct` rounds them: on the paint-tree
    /// image ([`Painter::em_base`]), or for an unsnapped run as its recorded list replays ([`Painter::replay_em`]).
    fn em_world(&self, tile: [f32; 2], brush_em: Option<Mat>) -> Option<Mat> {
        if let Some((g, t, [sx, sy], mf)) = self.replay_em {
            // ponytail: a composite drawn back by a sub-pixel shift or a turn keeps the device-space brush; its group
            // replays through an intermediate this does not frame.
            if self.frames.iter().any(|f| matches!(f, Frame::Layer { shift: Some(_), .. } | Frame::Layer { turn: Some(_), .. })) {
                return None;
            }
            let c = self.em_ops.iter().fold(g, |w, n| mul(&w, n));
            let bt = mul(&t, &c);
            let inv = 1.0 / (sx * sy);
            let (qx, qy) = (sy * inv, sx * inv);
            let user = [bt[0] * sx * qx, bt[1] * sy * qy, bt[2] * sx * qx, bt[3] * sy * qy, bt[4] * sx * qx, bt[5] * sy * qy];
            let dev = [user[0] * sx, user[1] * sy, user[2] * sx, user[3] * sy, (user[4] * sx + mf) - tile[0], (user[5] * sy + mf) - tile[1]];
            return Some(brush_em.map_or(dev, |b| mul(&dev, &b)));
        }
        self.em_base.map(|(b, i)| {
            let w = self.em_ops.iter().fold([b[0], b[1], b[2], b[3], b[4] - (tile[0] - i[0]), b[5] - (tile[1] - i[1])], |w, t| mul(&w, t));
            brush_em.map_or(w, |t| mul(&w, &t))
        })
    }

    /// `brush_em`: the fill's own brush transform in em units, if it has one.
    fn brush(&mut self, cb: &CBrush, b2d: &Mat, brush_em: Option<Mat>) -> Option<PBrush> {
        let tile = self.bitmap_origin();
        let mut local = *b2d;
        local[4] -= tile[0]; local[5] -= tile[1];
        // A brush realizes through the paint transform d2d1 builds (em units onto the phase frame), less the bitmap's
        // origin in that frame, all in f32: the absolute device mapping less the tile rounds a placement point an ulp
        // apart (VM paint-bitmap dump of r25 30523: a radial pixel at a ramp tie, exact only this way). Each transform
        // node multiplies in ahead of the frame (T × current), and a brush transform ahead of that (B × world), as
        // `Matrix3x2F::SetProduct` rounds them.
        let em_local = self.em_world(tile, brush_em);
        let upem = self.font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f32;
        let unit = 1.0 / upem;
        let em = em_local.is_some();
        let local = em_local.unwrap_or([local[0]*upem, local[1]*upem, -local[2]*upem, -local[3]*upem, local[4], local[5]]);
        let b2d = &local;
        let pending_radial = self.pending_radial.take();
        // Gradient interpolation gamma: native builds the stop collection with D2D1_GAMMA_2_2 (=0),
        // i.e. interpolate in sRGB (read from ReadDWriteGradientStopsAndRange).
        const GAMMA: u32 = 0;
        match cb {
            CBrush::Solid { palette_index, alpha } => Some(PBrush::solid(self.color(*palette_index, *alpha), 1.0)),
            CBrush::LinearGradient { p0, p1, color_stops, extend } => {
                let stops = self.stops(color_stops);
                match self.pending_linear.take() {
                    // Our reader's raw COLR points, projected as d2d1 `PaintTreeRenderer::
                    // GetLinearGradientBrush` does in paint space (f32): p1 onto the (p0, p2 − p0) line →
                    // foot, the gradient spanning foot→p1 over the stop domain. `PBrush::linear`'s
                    // realization then re-projects the end in device space, which keeps the tilt a
                    // non-uniform leaf transform gives (textx-emoji-134..136).
                    // DirectWrite hands d2d1 the points in em units (x·k, y·−k), so the projection runs there.
                    Some((rp0, rp1, rp2, lo, hi)) if em => {
                        let to_em = |p: [f32; 2]| [p[0] * unit, p[1] * -unit];
                        let (rp0, rp1, rp2) = (to_em(rp0), to_em(rp1), to_em(rp2));
                        let v = [rp2[0] - rp0[0], rp2[1] - rp0[1]];
                        let vv = v[1] * v[1] + v[0] * v[0];
                        if vv.abs() < 1e-8 { return None; }
                        let pr = ((rp1[0] - rp0[0]) * v[0] + (rp1[1] - rp0[1]) * v[1]) / vv;
                        let foot = [v[0] * pr + rp0[0], v[1] * pr + rp0[1]];
                        let d = [rp1[0] - foot[0], rp1[1] - foot[1]];
                        let (s, e) = ([d[0] * lo + foot[0], d[1] * lo + foot[1]], [d[0] * hi + foot[0], d[1] * hi + foot[1]]);
                        Some(PBrush::linear(&stops, GAMMA, extend_int(*extend), 1.0, s, e, b2d))
                    }
                    Some((rp0, rp1, rp2, lo, hi)) => {
                        let v = [rp2[0] - rp0[0], rp2[1] - rp0[1]];
                        let vv = v[1] * v[1] + v[0] * v[0];
                        if vv.abs() < 1e-8 { return None; }
                        let pr = ((rp1[0] - rp0[0]) * v[0] + (rp1[1] - rp0[1]) * v[1]) / vv;
                        let foot = [v[0] * pr + rp0[0], v[1] * pr + rp0[1]];
                        let d = [rp1[0] - foot[0], rp1[1] - foot[1]];
                        let (s, e) = ([d[0] * lo + foot[0], d[1] * lo + foot[1]], [d[0] * hi + foot[0], d[1] * hi + foot[1]]);
                        Some(PBrush::linear(&stops, GAMMA, extend_int(*extend), 1.0, [s[0] * unit, -s[1] * unit], [e[0] * unit, -e[1] * unit], b2d))
                    }
                    // skrifa fallback: points are already resolved; realize against `b2d` as before.
                    None => Some(PBrush::linear(&stops, GAMMA, extend_int(*extend), 1.0, [p0.x*unit, -p0.y*unit], [p1.x*unit, -p1.y*unit], b2d)),
                }
            }
            CBrush::RadialGradient { c0, c1, r1, color_stops, extend, .. } => {
                if *r1 <= 0.0 {
                    return None; // outer radius ≤ 0: nothing drawn (native returns early too).
                }
                // Only native's `GetRadialGradientBrush` case A reaches here (r0 == 0 and the focal
                // point c0 inside the outer circle): the D2D radial gradient brush with outer circle
                // (c1, r1) and focal point c0. Two-circle radials are routed to `fill_conical` by
                // `fill_glyph` before `brush` is called.
                let stops = self.stops(color_stops);
                // d2d1 places the brush from DirectWrite's em-unit circle (x·k, y·−k): centre c0 + (c1 − c0)·hi,
                // radius r1·hi and the origin offset c0 − centre.
                if let (true, Some((a, b, r, hi))) = (em, pending_radial) {
                    let (c0, c1) = ([a[0] * unit, a[1] * -unit], [b[0] * unit, b[1] * -unit]);
                    let centre = [c0[0] + (c1[0] - c0[0]) * hi, c0[1] + (c1[1] - c0[1]) * hi];
                    let radius = r * unit * hi;
                    return Some(PBrush::radial(&stops, GAMMA, extend_int(*extend), 1.0, centre, [radius, radius], [c0[0] - centre[0], c0[1] - centre[1]], b2d));
                }
                let offset = [(c0.x - c1.x)*unit, -(c0.y - c1.y)*unit];
                Some(PBrush::radial(&stops, GAMMA, extend_int(*extend), 1.0, [c1.x*unit, -c1.y*unit], [*r1*unit, *r1*unit], offset, b2d))
            }
            // Sweep gradients are routed to `fill_sweep` by `fill_glyph`, never reaching `brush`.
            CBrush::SweepGradient { .. } => None,
        }
    }

    /// `gid`'s outline flattened to device-space fill polygons under the current transform (`cur`),
    /// or `None` if the font has no such glyph. Shared by the solid/gradient leaf and the conical.
    fn outline_polys(&self, gid: GlyphId) -> Option<Vec<Vec<[f32; 2]>>> {
        let glyph = self.font.outline_glyphs().get(gid)?;
        let mut figures = Vec::new();
        let mut pen = RawPen { figures: &mut figures, cur: None };
        glyph.draw(DrawSettings::unhinted(Size::unscaled(), LocationRef::default()), &mut pen).ok()?;
        pen.flush();
        let geom = Geometry::Path { fill: FillMode::Winding, figures };
        Some(geom.flatten_fill(&self.leaf_cur()).into_iter().filter(|p| p.filled).map(|p| p.pts).collect())
    }

    /// `gid`'s outline as device-space contours (points + on-curve flags) under the current transform
    /// (`cur`) — the input `raster::coverage_outline` takes. Mirrors native `RenderLeafGlyph`, which
    /// draws each leaf as a one-glyph `DrawGlyphRun(em=1, NATURAL)`: the leaf coverage is DirectWrite's
    /// analysis-grayscale scan conversion of this outline, not a geometry 8×8 fill. `skrifa` emits an
    /// explicit on-curve end per quadratic, so off/on alternate exactly as the scan converter expects.
    fn outline_contours(&self, gid: GlyphId) -> Vec<(Vec<[f64; 2]>, Vec<bool>)> {
        let Some(glyph) = self.font.outline_glyphs().get(gid) else { return Vec::new() };
        let mut pen = ContourPen { m: self.leaf_cur(), contours: Vec::new(), pts: Vec::new(), on: Vec::new() };
        let _ = glyph.draw(DrawSettings::unhinted(Size::unscaled(), LocationRef::default()), &mut pen);
        pen.flush();
        pen.contours
    }

    /// `gid`'s leaf coverage via native's analysis scan converter ([`raster::coverage_outline`]) at the
    /// current (supersampled) transform. Native draws each leaf as `DrawGlyphRun(em=1)` with the paint
    /// transform folded into the world matrix: the leaf's
    /// device em is the transform scale (≈ `ss·em_size`, always > 20 ⇒ NATURAL_SYMMETRIC, grid-fit
    /// enabled). So grid-fit the leaf through the TrueType hinting interpreter ([`hinted_leaf`]) exactly
    /// as the regular text path does, feeding the scan converter the raw glyf points (native fixmul-
    /// scaled) instead of `skrifa`'s float-inserted implied midpoints — verified byte-exact to native's
    /// scan-converter contour output (the leaf `glyph32415` @ em96 oracle). A rotated/stretched paint
    /// transform (off the scalar-ppem hinter's model) falls back to the unhinted outline, as native's
    /// off-axis rule disables grid-fit there anyway.
    /// Leaf coverage plus the leaf outline's device ink-bounds min (`None` for the unhinted fallback).
    /// The COLR conical colorizer evaluates the gradient at the paint-tree atlas's pixel centres; the
    /// atlas tile origin is the ink-bounds min minus an integer border, so the per-pixel sample grid is
    /// phased by `frac(ink_min)`. The min is the FLATTENED extent (each quadratic's on-curve endpoints
    /// plus its per-axis turning point `t* = (p0−c)/(p0−2c+p1)` when `t*∈(0,1)`), NOT the raw control
    /// points — an off-curve control sits outside the visual ink and would mis-phase the grid.
    /// An opaque solid outline leaf: the unhinted outline filled through the cached outline realization
    /// ([`crate::geometry::realization::fill_colr_outline`]) in the leaf's premultiplied BGRA colour `color` into a
    /// transparent buffer, `None` when it inks nothing. Nonzero winding across the contours handles overlapping
    /// figures. The mesh carries the colour, so each texel is the colour times its float coverage rounded once (VM
    /// probes: a grey rect leaf's edge holds 95 at alpha 188, where the colour times the 8-bit coverage gives 94).
    fn feather_layer(&self, gid: GlyphId, color: [f32; 4]) -> Option<PixelBuffer> {
        // A solid outline leaf, whether enlarged past 100 px or in a run past 128 px, is drawn from the cached
        // outline realization: the outline baked at its size bucket, its AA directions built in glyph space
        // and mapped by the leaf transform (device-space normals give different coverage under anisotropic
        // paint transforms). VM probe: an upright rect leaf at em 140 from y 100.3 matches only this way; the
        // device-space feather sat 0.06 px high.
        let upem = self.font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f32;
        let m = self.leaf_cur();
        let device_em = upem * m[0].abs().max(m[1].abs()).max(m[2].abs()).max(m[3].abs());
        let bake = crate::geometry::realization::colr_outline_bucket(device_em) / upem;
        let glyph = self.font.outline_glyphs().get(gid)?;
        let mut pen = ContourPen {
            m: [bake, 0.0, 0.0, -bake, 0.0, 0.0],
            contours: Vec::new(), pts: Vec::new(), on: Vec::new(),
        };
        // The cached outline is GetGlyphRunOutline's, hinted at ppem = upem ([`super::outline_hinter`], kept under key -1).
        let hinted = {
            let mut cache = self.hints.borrow_mut();
            if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(-1) {
                if let Some(h) = super::outline_hinter(self.font, &[]) {
                    e.insert(h);
                }
            }
            cache.get(&-1).and_then(|h| super::raster::outline_points_fu(h, gid))
        };
        match &hinted {
            Some(p) => super::outline_embold::draw_points(p, None, false, &mut pen),
            None => {
                let _ = glyph.draw(DrawSettings::unhinted(Size::unscaled(), LocationRef::default()), &mut pen);
            }
        }
        pen.flush();
        let figures: Vec<Figure> = pen.contours.iter().filter(|(pts, _)| pts.len() >= 3).filter_map(|(pts, on)| contour_to_figure(pts, on)).collect();
        if figures.is_empty() {
            return None;
        }
        let geom = Geometry::Path { fill: FillMode::Winding, figures };
        let mut buf = self.new_buf();
        let transform = [(m[0]/bake) as f64, (m[1]/bake) as f64, (-m[2]/bake) as f64, (-m[3]/bake) as f64, m[4] as f64, m[5] as f64];
        let rgba = [color[2], color[1], color[0], color[3]].map(|v| v.round_ties_even().clamp(0.0, 255.0) / 255.0);
        // In a composite's cell the leaf is drawn through the atlas viewport, which begins at the cell for the backdrop
        // (the first cell of the shelf) and one cell to its left for the source beside it: WARP clips a leaf crossing
        // the viewport's edge there and fans the rest (VM DDI of geo/c126b: 512-texel atlas, cells 107 px apart; the
        // native vertex shader's output clipped so reproduces the atlas exactly).
        let depth = self.frames.iter().filter(|f| matches!(f, Frame::Layer { .. })).count();
        let vp = match self.vp_origins {
            Some((back, src)) => match depth { 1 => Some(back), 2 => Some(src), _ => None }.map(|o| [o[0], o[1], o[0] + 512.0, o[1] + 512.0]),
            None => self.cell.and_then(|c| match depth { 1 => Some(c[0]), 2 => Some(2.0 * c[0] - c[2]), _ => None }.map(|x| [x, c[1], x + 512.0, c[1] + 512.0])),
        };
        crate::geometry::realization::fill_colr_outline(&mut buf, &geom, &transform, rgba, vp);
        buf.data.chunks_exact(4).any(|p| p[3] > 0).then_some(buf)
    }

    /// Merge a premultiplied leaf `layer` (the active buffer's size) source-over onto the active buffer with the
    /// solid leaf composite [`Self::fill_coverage`] uses.
    fn merge_leaf(&mut self, layer: &PixelBuffer) {
        let buf = self.active_buf();
        let (w, h) = (buf.desc.width as usize, buf.desc.height as usize);
        for y in 0..h {
            for x in 0..w {
                let (i, j) = (y * buf.stride + x * 4, y * layer.stride + x * 4);
                let s = &layer.data[j..j + 4];
                if s[3] == 0 {
                    continue;
                }
                let ia = 255 - s[3] as i32;
                for k in 0..4 {
                    buf.data[i + k] = (s[k] as i32 + unorm_mul(buf.data[i + k] as i32, ia)).clamp(0, 255) as u8;
                }
            }
        }
    }

    fn leaf_coverage(&self, gid: GlyphId) -> (AlphaTexture, Option<[f32; 2]>) {
        // D2D_LEAFLIST: per-leaf (gid, device em, canvas bbox) dump — triage for picking the leaf that
        // owns a divergent pixel (30066). Inert without the env var.
        let r = self.leaf_coverage_inner(gid);
        if std::env::var_os("D2D_LEAFLIST").is_some() {
            let m = self.leaf_cur();
            let c = self.cur();
            let upem = self.font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f32;
            let em = (m[0] * m[0] + m[1] * m[1]).sqrt() * upem;
            eprintln!("LEAFLIST gid={} em={:.4} bbox=[{},{},{},{}] shear={} cur={:?} leafcur={:?}",
                gid.to_u32(), em, r.0.left, r.0.top, r.0.right, r.0.bottom, self.shear, c, m);
        }
        r
    }
    fn leaf_coverage_inner(&self, gid: GlyphId) -> (AlphaTexture, Option<[f32; 2]>) {
        // device_em > 128 (`self.feather`): native fills each leaf's outline as GEOMETRY. A solid leaf is cached as
        // the tessellator's feathered fill ([`Self::feather_layer`], verified byte-exact to native's captured leaf edge
        // curve at em150), NOT the glyph scan converter's grayscale scan; a gradient leaf is not cached and takes the
        // plain 8×8 fill, as outline text with a non-solid brush does (r25 30138: a sweep leaf's left edge 6/8 covered
        // where the feather gives 0.78). The composite then runs LINEAR (no gamma) — see `fill_coverage`. At ≤ 128
        // the glyph scan below is native's choice (byte-exact at em50).
        if self.leaf_outline() || self.feather {
            // Non-solid outline leaves take the ordinary geometry fill (8×8, no text gamma).
            // Native's gradient strip in emoji-134 has alpha 64/96 on its two rows, whereas
            // substituting a solid brush gives the feather coverage 116/202.
            return (AlphaTexture {
                left: 0, top: 0,
                right: self.canvas().width as i32, bottom: self.canvas().height as i32,
                data: self.clip_mask(gid),
            }, None);
        }
        let kind = if matches!(self.aa, Aa::Aliased) { raster::Kind::Aliased } else { raster::Kind::Grayscale };
        // COLR-leaf grid-fit decision (native `GetPaintTreeGlyphImage` → `GetPaintTreeImage` leaf
        // rasterization, per-leaf at the leaf's own device ppem). Native grid-fits (hints) a leaf iff the
        // font's `gasp` table turns grid-fitting on at that ppem; a sub-element a `PaintScale` drops below
        // the gasp grid-fit floor (seguiemj 🎉/🎊 confetti gid 4938 at em 7–8, gasp ≤8 ppem = no gridfit)
        // renders from the UNHINTED outline instead, as does a rotated leaf (our hinter can't grid-fit
        // off-axis, and native leaves a rotated origin unsnapped). Hinting every smooth leaf regardless
        // regressed ~24 all-base-em emoji whose native edges ARE grid-fit vs the tiny ones that are not;
        // keying on gasp matches native's own decision and generalizes to any COLR font. The remaining
        // unhinted scan-converter exactness (and our hinter's drift at 9–11 ppem, where gasp DOES grid-
        // fit but our grid-fit is inexact) is the hinting agent's.
        // The em-1 leaf matrix is the scaler's device px per em (textx-emoji-315/316/317/324).
        let m = self.leaf_cur();
        let nudge = raster::max45_nudge([m[0], m[1], m[2], m[3]].map(f64::from));
        if !matches!(self.aa, Aa::Aliased) {
            let m = self.leaf_cur();
            // DWrite grid-fits an off-axis leaf unless all four em-1 matrix entries are nonzero
            // (dwrite GetRecommendedRenderingModeInternal, IsZeroWithTolerance at 1/65536): a quarter
            // turn or a single-axis skew still snaps when gasp enables grid-fit (the scaler then
            // pre-scales by the per-axis magnitude, textx-emoji-049), a general rotation stays unhinted.
            let upem = self.font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f64;
            let device_em = (m[0] as f64 * m[0] as f64 + m[1] as f64 * m[1] as f64).sqrt() * upem;
            // Dropout mode for the unhinted leaf branches, from the font's prep scan control (not a
            // hardcoded 2): seguiemj's prep enables dropout at every ppem, so a collapsed sub-pixel
            // overlay recovers via simple dropout (see [`leaf_dropout`] / `raster::extrema_collapsed`).
            let unhinted_drop = self.leaf_dropout(device_em);
            let nonzero = |v: f32| (v * upem as f32).abs() >= 1.0 / 65536.0;
            let rotated = nonzero(m[1]) || nonzero(m[2]);
            if m[..4].iter().all(|&v| nonzero(v)) {
                return (raster::coverage_colr(&self.unhinted_leaf(gid), unhinted_drop, kind, nudge), None);
            }
            // Consult `gasp` at the leaf's integer device ppem; the COLR leaf path is NATURAL_SYMMETRIC,
            // so the SYMMETRIC_GRIDFIT bit decides. seguiemj: ≤8 ppem → 0x0a (no gridfit), ≥9 → 0x0f
            // (gridfit). Absent/unmatched gasp ⇒ grid-fit (DWrite's default), so non-gasp fonts hint.
            let ppem = device_em.round_ties_even() as i64;
            let gridfit = self
                .font
                .gasp()
                .ok()
                .and_then(|g| {
                    g.gasp_ranges()
                        .iter()
                        .find(|r| ppem <= r.range_max_ppem() as i64)
                        .map(|r| r.range_gasp_behavior().contains(GaspRangeBehavior::GASP_SYMMETRIC_GRIDFIT))
                })
                .unwrap_or(true);
            if rotated {
                let contours = if gridfit { self.fixed_leaf(gid, true).unwrap_or_else(|| self.raw_snapped_leaf(gid)) } else { self.unhinted_leaf(gid) };
                return (raster::coverage_colr(&contours, unhinted_drop, kind, nudge), None);
            }
            if gridfit {
                if let Some((contours, drop)) = self.hinted_leaf(gid) {
                    let contours = self.fixed_leaf(gid, true).unwrap_or(contours);
                    return (raster::coverage_colr(&contours, drop, kind, nudge), ink_min_of(&contours));
                }
            }
            let contours = self.unhinted_leaf(gid);
            return (raster::coverage_colr(&contours, unhinted_drop, kind, nudge), ink_min_of(&contours));
        }
        match self.hinted_leaf(gid) {
            Some((contours, drop)) => (raster::coverage_colr(&contours, drop, kind, nudge), ink_min_of(&contours)),
            None => (raster::coverage_colr(&self.outline_contours(gid), 2, kind, nudge), None),
        }
    }

    /// Unhinted TrueType points stay in design-unit F26Dot6 until the fixed post-transform.
    /// Reproduces DirectWrite's glyph scaling setup, verified byte-identical.
    fn unhinted_leaf(&self, gid: GlyphId) -> Vec<(Vec<[f64; 2]>, Vec<bool>)> {
        self.fixed_leaf(gid, false).unwrap_or_else(|| self.outline_contours(gid))
    }

    fn fixed_leaf(&self, gid: GlyphId, gridfit: bool) -> Option<Vec<(Vec<[f64; 2]>, Vec<bool>)>> {
        use super::hinting::{fixdiv, fixmul};
        let m = self.leaf_cur();
        let upem = self.font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as i32;
        let raw = self.font.glyf().ok().zip(self.font.loca(None).ok());
        let simple = raw.as_ref().and_then(|(gf, lc)| lc.get_glyf(gid, gf).ok().flatten()).and_then(|gl| match gl {
            Glyph::Simple(s) => Some(s),
            _ => None,
        });
        let glyph = simple?;
        if gridfit && !glyph.instructions().is_empty() { return None; }
        // NewTransform receives the em-1 run's 16.16 world matrix. mth_IntelMul divides its
        // components by unitsPerEm first, then rounds EACH product before adding the two axes.
        let world: [i32; 4] = core::array::from_fn(|i| (m[i] * upem as f32 * 65536.0).round() as i32);
        let scales = [world[0].abs().max(world[1].abs()), world[2].abs().max(world[3].abs())];
        let divisors = if gridfit { scales } else { [upem << 16; 2] };
        let matrix: [i32; 4] = core::array::from_fn(|i| fixdiv(world[i], divisors[i / 2]));
        // native's glyph-run analyzer slow-path position init: cvtss2si plus the
        // negative-half correction is round-half-up, on both axes even under rotation.
        let origin = [m[4], m[5]].map(|v| ((v * 4.0 + 0.5).floor() * 0.25) as f64);
        let mut pts: Vec<(i32, i32, bool)> = glyph.points().map(|p| {
            let (x, y) = if gridfit {
                (scale_leaf_point(p.x as i32, scales[0], upem), scale_leaf_point(p.y as i32, scales[1], upem))
            } else { (p.x as i32 * 64, p.y as i32 * 64) };
            (x, y, p.on_curve)
        }).collect();
        // The bold simulation emboldens a leaf in its own frame before the transform, as it does a grayscale
        // glyph (VM: Segoe UI Emoji at weight 700). A grid-fit leaf takes the scaler's DEVICE embolden by
        // `bold_amount(device ppem)` of the grid-fit points. A grid-fit-OFF leaf (native disables grid-fit for
        // an off-axis paint transform) takes the DESIGN-strength embolden `strength(upem)` of the design points at
        // any device boldX, as a symmetric-mode unhinted glyph does (VM reye capture of 30066's rotated right eye:
        // +1.45 px x +1.33 px y matches (41, 40) font units x em/upem, not device (2, 1); r25 30790's leaf at a
        // one-pixel device boldX takes strength (41, 40) in dwrite's embold too).
        if self.bold && !matches!(self.aa, Aa::Aliased) {
            let mut h = super::hinting::HintedOutline {
                points: pts.iter().map(|&(x, y, _)| super::hinting::Point { x, y }).collect(),
                on_curve: pts.iter().map(|p| p.2).collect(),
                contour_ends: glyph.end_pts_of_contours().iter().map(|e| e.get()).collect(),
                ..Default::default()
            };
            if gridfit {
                let (bx, by) = raster::bold_amount(scales[1] >> 16);
                if bx == 1 {
                    super::embolden::horizontal_one_pixel(&mut h);
                } else if bx >= 2 {
                    super::embolden::embolden_hinted(&mut h, bx, by);
                }
            } else {
                let (sx, sy) = super::outline_embold::strength(upem);
                super::embolden::embolden_hinted(&mut h, sx, sy);
            }
            for (q, p) in pts.iter_mut().zip(&h.points) {
                (q.0, q.1) = (p.x, p.y);
            }
        }
        let points: Vec<_> = pts.into_iter().map(|(x, y, on)| {
            ([origin[0] + (fixmul(x, matrix[0]) + fixmul(y, matrix[2])) as f64 / 64.0,
              origin[1] + (fixmul(x, matrix[1]) + fixmul(y, matrix[3])) as f64 / 64.0], on)
        }).collect();
        let mut contours = Vec::new();
        let mut start = 0;
        for end in glyph.end_pts_of_contours() {
            let end = end.get() as usize;
            if end >= start && end < points.len() {
                contours.push(points[start..=end].iter().copied().unzip());
            }
            start = end + 1;
        }
        Some(contours)
    }

    /// The UNHINTED leaf outline shifted to the 1/ss-snapped device origin — native's COLR-leaf
    /// placement (origin-snapped, not grid-fit). The snap is the residual `snap(pen) − pen` applied
    /// rigidly to `outline_contours`'s device points (`pen = cur·origin`, ss=4 smooth / 1 aliased).
    fn raw_snapped_leaf(&self, gid: GlyphId) -> Vec<(Vec<[f64; 2]>, Vec<bool>)> {
        let m = self.leaf_cur();
        let pen = apply(&m, 0.0, 0.0);
        let ss = if matches!(self.aa, Aa::Aliased) { 1.0 } else { 4.0 };
        let snap = |v: f32| ((v as f64) * ss).round_ties_even() / ss;
        let (dx, dy) = (snap(pen[0]) - pen[0] as f64, snap(pen[1]) - pen[1] as f64);
        let dev = |x: f64, y: f64| {
            let p = apply(&m, x as f32, y as f32);
            [p[0] as f64 + dx, p[1] as f64 + dy]
        };
        // Match the hinted path's RAW-glyf quadratics: `skrifa`'s outline pen inserts the implied
        // on-curve midpoint between two off-curve points in FLOAT after scaling (±1 sub-sample off on
        // curved stems — `glyph_run_arcs` note); native inserts it in design units. So walk the raw
        // `glyf` points and insert midpoints in design space, then transform. Composite/CFF leaves (no
        // simple glyf) keep `skrifa`'s pen via `outline_contours`.
        let raw = self.font.glyf().ok().zip(self.font.loca(None).ok());
        let simple = raw.as_ref().and_then(|(gf, lc)| lc.get_glyf(gid, gf).ok().flatten()).and_then(|gl| match gl {
            Glyph::Simple(s) => Some(s),
            _ => None,
        });
        let Some(s) = simple else {
            let mut c = self.outline_contours(gid);
            for (pts, _) in &mut c {
                for p in pts {
                    p[0] += dx;
                    p[1] += dy;
                }
            }
            return c;
        };
        let pts: Vec<(f64, f64, bool)> = s.points().map(|p| (p.x as f64, p.y as f64, p.on_curve)).collect();
        let mut out: Vec<(Vec<[f64; 2]>, Vec<bool>)> = Vec::new();
        let mut start = 0usize;
        for e in s.end_pts_of_contours() {
            let e = e.get() as usize;
            if e >= start && e < pts.len() {
                out.push(expand_glyf_contour(&pts[start..=e], &dev));
            }
            start = e + 1;
        }
        out
    }

    /// Grid-fit `gid` through the hinting interpreter at the leaf's device em (the scale of the current
    /// transform × upem) and lay the hinted contours out in supersampled-device space for the scan
    /// converter, returning `(contours, dropout_mode)`. Only the axis-aligned ~uniform case maps to the
    /// scalar-ppem hinter; a rotated or stretched `cur` (an off-axis paint transform) returns `None` so
    /// the caller uses the unhinted outline (native disables grid-fit off-axis). The device em is cached
    /// across leaves of the same scale. `None` also when the font lacks the TrueType scaler tables.
    fn hinted_leaf(&self, gid: GlyphId) -> Option<(Vec<(Vec<[f64; 2]>, Vec<bool>)>, u16)> {
        let id16 = u16::try_from(gid.to_u32()).ok()?;
        let m = self.leaf_cur();
        let sx = (m[0] * m[0] + m[1] * m[1]).sqrt();
        let sy = (m[2] * m[2] + m[3] * m[3]).sqrt();
        // Rotated (off-diagonal) or anisotropic ⇒ outside the scalar-ppem hinter: fall back (unhinted).
        if m[1].abs() > 1e-4 || m[2].abs() > 1e-4 || (sx - sy).abs() > 1e-3 * sx.max(sy) {
            return None;
        }
        let upem = self.font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f64;
        // Scale at the EXACT device em, NOT rounded to integer ppem. seguiemj's COLR leaves carry no
        // TrueType instructions, so grid-fitting only scales the outline; native rasterizes each leaf at
        // its exact em, and rounding it (e.g. a PaintScaled face at em 24.78 → 25) scaled the leaf ~1 %
        // too big and spilled its edge by a pixel (162 😋 face: worst 142→). The integer ppem only
        // matters for the gasp decision (`leaf_coverage`) and instructed hinting, which this font lacks.
        let device_em = sx as f64 * upem;
        if device_em < 1.0 {
            return None;
        }
        // Leaves are always NATURAL_SYMMETRIC (device em ≈ ss·em ≫ 20); aliased runs hint classic.
        let cfg = HConfig {
            mode: if matches!(self.aa, Aa::Aliased) { HMode::Aliased } else { HMode::Grayscale },
            symmetric: true,
            ..Default::default()
        };
        // Instance::new rounds em to F26Dot6; the cache must distinguish the same scales.
        let key = (device_em * 64.0).round() as i64;
        let mut cache = self.hints.borrow_mut();
        if !cache.contains_key(&key) {
            cache.insert(key, HInstance::new(self.font, device_em, cfg)?);
        }
        let inst = cache.get(&key)?;
        let mut h = inst.hint(id16)?;
        // The bold simulation emboldens the hinted leaf as it does a grayscale glyph (VM: Segoe UI Emoji at weight 700).
        if self.bold && !matches!(self.aa, Aa::Aliased) {
            let (bx, by) = raster::bold_amount(device_em as i32);
            if bx == 1 {
                super::embolden::horizontal_one_pixel(&mut h);
            } else if bx >= 2 {
                super::embolden::embolden_hinted(&mut h, bx, by);
            }
        }
        // Pen = the glyph's device origin; DrawGlyphRun snaps baseline y to the grid and x to the
        // symmetric 4-phase sub-pixel grid (the run origin is pre-snapped to the ss grid, so for an
        // identity paint transform this is a no-op). xMin==lsb for every leaf ⇒ the hinter's lsb shift
        // is 0, so the hinted points sit at their absolute font position relative to the pen, as native.
        // The hinter works in the glyph's upright frame; native applies the world transform to the
        // hinted outline, so re-apply `cur`'s per-axis direction here — a mirrored/flipped paint
        // transform (negative `cur[0]`/`cur[3]`, e.g. a reused-and-reflected eye) flips the offset sign.
        let pen = apply(&m, 0.0, 0.0);
        let sgx = (m[0] as f64).signum();
        let sgy = (m[3] as f64).signum();
        // Native snaps the leaf's device origin to a 1/ss grid in
        // BOTH axes, ss keyed to the run's DWRITE_RENDERING_MODE1: GDI_*/ALIASED (1,2,3) → 1 (whole
        // pixel); NATURAL (4) → 8 in x, 1 in y; NATURAL_SYMMETRIC/DEFAULT (0,5,7) → 4 in both; OUTLINE
        // (6) or device em > 128 → no snap. The snap is cvtss2si (round half-to-even), residual folded
        // into the render transform. Segoe UI Emoji's `gasp` sets SYMMETRIC_SMOOTHING at every ppem
        // (≤8: 0x0a, 9+: 0x0f), so every leaf is NATURAL_SYMMETRIC ⇒ ss=4 in x and y at all sizes;
        // aliased = whole pixel. (Verified byte-for-byte: ss=4 is native's placement; the old
        // whole-pixel-y round spread the box8-resolved near-horizontal edges top-fuller/bottom-thinner.)
        let ss = if matches!(self.aa, Aa::Aliased) { 1.0 } else { 4.0 };
        let snap = |v: f32| ((v as f64) * ss).round_ties_even() / ss;
        let (px, py) = (snap(pen[0]), snap(pen[1]));
        let mut out: Vec<(Vec<[f64; 2]>, Vec<bool>)> = Vec::new();
        let mut s = 0usize;
        for &e in &h.contour_ends {
            let e = e as usize;
            if e >= s && e < h.points.len() {
                let pts = h.points[s..=e].iter().map(|p| [px + sgx * p.x as f64 / 64.0, py + sgy * p.y as f64 / 64.0]).collect();
                out.push((pts, h.on_curve[s..=e].to_vec()));
            }
            s = e + 1;
        }
        let (enabled, scan_type) = inst.scan_control();
        Some((out, raster::dropout_mode(enabled, scan_type)))
    }

    /// The scan-converter dropout mode for a leaf rendered from an UNHINTED outline (no-gridfit, rotated
    /// or anisotropic leaves that bypass `hinted_leaf`). Native still runs the font's `prep` and hands
    /// the fsc fill that `SCANCTRL`/`SCANTYPE` state at every leaf — seguiemj's prep enables dropout at
    /// all ppem (VM: gid 16022 at em 6/8/14 → a4=1, fillbm mode 0 after the collapse mask) — so the old
    /// hardcoded `2` (no dropout, from the disproven "seguiemj has no prep" assumption) dropped the
    /// sub-pixel overlay slivers. Build the prep-only instance at the leaf's device em and read its
    /// resolved scan control; fonts whose prep leaves dropout off (or that lack a prep) still get `2`.
    fn leaf_dropout(&self, device_em: f64) -> u16 {
        if device_em < 1.0 {
            return 2;
        }
        let cfg = HConfig {
            mode: if matches!(self.aa, Aa::Aliased) { HMode::Aliased } else { HMode::Grayscale },
            symmetric: true,
            ..Default::default()
        };
        let key = (device_em * 64.0).round() as i64;
        let mut cache = self.hints.borrow_mut();
        if !cache.contains_key(&key) {
            match HInstance::new(self.font, device_em, cfg) {
                Some(i) => { cache.insert(key, i); }
                None => return 2,
            }
        }
        let (enabled, scan_type) = cache.get(&key).map(|i| i.scan_control()).unwrap_or((false, 0));
        raster::dropout_mode(enabled, scan_type)
    }

    /// Composite `cov` (device-pixel grayscale coverage, `[left,top,right,bottom]`) × the premultiplied
    /// per-pixel BGRA source `src(x+0.5,y+0.5)` source-over into the active buffer — the shared body of
    /// every leaf fill (`RenderLeafGlyph`: glyph coverage × paint brush). `src` returns a premultiplied
    /// BGRA source (float 0..255), or `None` where a gradient is undefined. Gradient text keeps
    /// the filtered colour in float through gamma and coverage, then stores it as UNORM8.
    fn fill_coverage(&mut self, cov: &AlphaTexture, solid: bool, src: impl Fn(f32, f32) -> Option<[f32; 4]>) {
        let dbg = std::env::var("COLR_DBG_PX").ok().and_then(|s| {
            let mut p = s.split(',');
            Some((p.next()?.trim().parse::<i32>().ok()?, p.next()?.trim().parse::<i32>().ok()?))
        });
        // Outline coverage blends linearly. Scanned solid text uses the byte LUT, while a
        // gradient glyph uses its fixed-contrast mask and float gamma shader.
        let linear = self.feather || self.leaf_outline();
        let buf = self.active_buf();
        let (w, h, stride) = (buf.desc.width as i32, buf.desc.height as i32, buf.stride);
        let cw = cov.width();
        for y in cov.top.max(0)..cov.bottom.min(h) {
            for x in cov.left.max(0)..cov.right.min(w) {
                let cov8 = cov.data[((y - cov.top) * cw + (x - cov.left)) as usize] as i32;
                if cov8 == 0 {
                    continue;
                }
                let Some(s) = src(x as f32 + 0.5, y as f32 + 0.5) else { continue };
                let s = if linear { s.map(|v| v.round_ties_even().clamp(0.0, 255.0)) } else { s };
                let i = y as usize * stride + x as usize * 4;
                // Native renders each leaf as grayscale text onto the paint-tree bitmap, so the coverage
                // is gamma-blended (gamma 1.8, the fresh internal DC's default) not composited linearly;
                // the weight is `premul_fg · corrected(cov, straight_fg)` (`super::blend`). `s` is premul
                // BGRA — unpremultiply for the correction's luminance/intensity; skip fully-transparent.
                let a = s[3];
                if a == 0.0 {
                    continue;
                }
                let fg = [s[2] / a, s[1] / a, s[0] / a, a / 255.0];
                let corrected = if linear { cov8 } else { super::blend::leaf_gamma_coverage(cov8 as u8, fg) };
                if dbg == Some((x, y)) {
                    eprintln!("  LEAF@({x},{y}): cov8={cov8} src_premul_BGRA={s:?} corrected={corrected}");
                }
                let sc: [i32; 4] = if !linear && !solid {
                    // The WARP UNORM-fixed→float conversion,
                    // followed by the gradient-mask pixel shader and float→UNORM<8>.
                    let sampled = s.map(|v| v * (1.0 / 256.0) + v * (1.0 / 65280.0));
                    let fg = [sampled[2] / sampled[3], sampled[1] / sampled[3], sampled[0] / sampled[3], sampled[3]];
                    let weight = super::blend::leaf_brush_coverage(cov8 as u8, fg);
                    core::array::from_fn(|k| (sampled[k] * weight * 255.0).round_ties_even().clamp(0.0, 255.0) as i32)
                } else if !linear {
                    // The text lookup-table build: premultiplied
                    // float colour times corrected coverage, then round-half-up to the LUT byte.
                    core::array::from_fn(|k| ((s[k] * (1.0 / 255.0) * corrected as f32) + 0.5).floor().clamp(0.0, 255.0) as i32)
                } else {
                    // A non-solid outline leaf is an ordinary AA geometry fill: the fill kernel weights the
                    // byte colour by the coverage byte as `(t + (t >> 8) + 128) >> 8`, as the brush fringe does.
                    core::array::from_fn(|k| warp_unorm_mul(s[k].round_ties_even().clamp(0.0, 255.0) as i32, corrected))
                };
                if dbg == Some((x, y)) { eprintln!("  LEAF_SOURCE={sc:?}"); }
                let ia = 255 - sc[3];
                let out: [u8; 4] = core::array::from_fn(|k| {
                    let d = buf.data[i + k] as i32;
                    // The WARP gradient-mask shader (emoji-108).
                    // The high-byte correction precedes the +128 rounding bias. An outline-mode
                    // gradient leaf composites the same way (textx-emoji-051/053).
                    let scaled = if !solid { warp_unorm_mul(d, ia) } else { unorm_mul(d, ia) };
                    (sc[k] + scaled).clamp(0, 255) as u8
                });
                buf.data[i..i + 4].copy_from_slice(&out);
            }
        }
    }

    /// Fill `gid`'s leaf coverage with `brush` (built under `b2d`), source-over — native `RenderLeafGlyph`
    /// (a one-glyph `DrawGlyphRun` with the paint brush as foreground).
    fn fill_outline(&mut self, gid: GlyphId, brush: &PBrush) {
        // Only an opaque solid outline leaf is drawn from the cached feather mesh. A translucent one fills its
        // shape through the ordinary geometry fill, as a monochrome outline glyph does (VM: a 0.6-alpha rect
        // leaf at em 116 and 140 lands on 1/64 coverage steps where the opaque one feathers).
        if let PBrush::Solid(c) = brush {
            if (self.feather || self.leaf_outline()) && c[3] < 255.0 {
                if let Some(polys) = self.outline_polys(gid) {
                    let src = c.map(|v| v.round_ties_even().clamp(0.0, 255.0) as u8);
                    let aa = self.aa;
                    Target::new(self.active_buf()).fill_path_in(&polys, crate::pixels::FillMode::Winding, src, aa, Blend::SourceOver, true, false);
                }
                return;
            }
            if self.feather || self.leaf_outline() {
                if let Some(layer) = self.feather_layer(gid, *c) {
                    self.merge_leaf(&layer);
                }
                return;
            }
        }
        let (cov, _) = self.leaf_coverage(gid);
        // The text mask shader retains filtered colour in float. Outline leaves use the
        // geometry shader's truncating WARP gradient sampler (native's radial pixel shader).
        if std::env::var_os("COLR_DBG_PX").is_some() {
            if let PBrush::Radial { to_unit, focal, .. } = brush { eprintln!("RADIAL_MAP tile={:?} m={to_unit:?} focal={focal:?}", self.bitmap_origin()); }
        }
        let solid = matches!(brush, PBrush::Solid(_));
        let text = !solid && !self.feather && !self.leaf_outline();
        let tile = self.bitmap_origin();
        self.fill_coverage(&cov, solid, |x, y| Some(if text { brush.source_for_text(x-tile[0], y-tile[1]) } else { brush.source_at(x-tile[0], y-tile[1], true) }));
    }

    /// Fill leaf coverage `cov` with the radial-realized 1-D ramp `gb` ([`PBrush::ramp`]) sampled at the
    /// per-pixel parameter `t_of(p)` (`p` in brush space via `to_brush`) — the shared body of the conical
    /// and sweep fills. `sample_ramp_f(t)` samples at `u = t·u1` through the full-float `sample_f`, exactly
    /// as native's radial sampler does (the ramp is CalcTextureSize-radial-realized, not linear `view_width`).
    /// `None` from `t_of` leaves the pixel untouched.
    /// `wide` picks the ramp texcoord scale: the conical colorizer samples at `u = t·W` (native's radial
    /// scale, [`PBrush::sample_ramp_f_w`]); the sweep colorizer at `u = t·u1` ([`PBrush::sample_ramp_f`]).
    fn fill_gradient(&mut self, cov: &AlphaTexture, to_brush: &Mat, gb: &PBrush, wide: bool, t_of: impl Fn([f32; 2]) -> Option<f32>) {
        let dbg = std::env::var("COLR_DBG_PX").ok().and_then(|s| {
            let mut p = s.split(',');
            Some((p.next()?.trim().parse::<i32>().ok()?, p.next()?.trim().parse::<i32>().ok()?))
        });
        self.fill_coverage(cov, false, |x, y| {
            let t = t_of(apply(to_brush, x, y))?;
            if dbg == Some((x as i32, y as i32)) {
                eprintln!("  GRADIENT@({},{}) brush=({:.2},{:.2}) t={t:.4}", x as i32, y as i32, apply(to_brush, x, y)[0], apply(to_brush, x, y)[1]);
            }
            let s = if wide { gb.sample_ramp_f_w(t) } else { gb.sample_ramp_f(t) };
            Some(s.map(|v| (v * 255.0).round_ties_even().clamp(0.0, 255.0)))
        });
    }

    /// Fill `gid`'s outline with a COLRv1 two-circle radial (native's conical-gradient shader): `t`
    /// per pixel is the two-circle solve [`conical_t`], fed through the shared ramp [`Self::fill_gradient`].
    /// `W` is the device radius of the outer circle, sized like the core radial brush.
    #[allow(clippy::too_many_arguments)]
    fn fill_conical(&mut self, gid: GlyphId, c0: [f32; 2], r0: f32, c1: [f32; 2], r1: f32, stops: &[ColorStop], extend: Extend, b2d: &Mat) {
        // Native realizes the gradient ramp as an 11-level mip pool (1024..1, `EnsureRealizedForBrushType`)
        // and the conical-gradient shader binds the BASE level (MostDetailedMip=0, the 1024-wide ramp) —
        // confirmed by the SRV-bind capture (colorfont2-04 conical → level 0, colorfont2-09 sweep → 0;
        // only the plain radial brush picks a footprint level, which `Brush::Radial` already does). The
        // two-circle `t` varies too sharply (near the focal) for a narrow mip, so the conical uses full
        // resolution. Footprint ≥ 1024 → `radial_width` = 1024 (the base level).
        let gb = PBrush::ramp(&self.stops(stops), 0, extend_int(extend), 1024.0);
        let to_brush = invert(b2d);
        if let Ok(s) = std::env::var("COLR_DBG_PX") {
            let mut p = s.split(',');
            if let (Some(dx), Some(dy)) = (p.next().and_then(|v| v.trim().parse::<f32>().ok()), p.next().and_then(|v| v.trim().parse::<f32>().ok())) {
                let pb = apply(&to_brush, dx + 0.5, dy + 0.5);
                let dist = ((pb[0] - c0[0]).powi(2) + (pb[1] - c0[1]).powi(2)).sqrt();
                eprintln!("  CONICAL c0={c0:?} c1={c1:?} r0={r0} r1={r1} b2d={b2d:?}\n    p_brush@({dx},{dy})={pb:?} dist(p,c0)={dist:.1} t={:?}", conical_t(pb, c0, r0, c1, r1));
            }
        }
        let (cov, ink_min) = self.leaf_coverage(gid);
        // Native realizes
        // the two-circle gradient into a texture sized by the DESIGN-unit footprint and evaluates it at
        // the leaf atlas's PIXEL CENTRES. For a NON-concentric (focal) gradient the colorizer frame
        // carries, verified byte-exact on cf04 (844/15 → 384/1):
        //   • `K = (W−1)/W`, `W = radial_width(max(r0,r1,|c1−c0|))` in DESIGN units (cf04: 320→512),
        //     applied via `conical_t` (ratio-invariant) at `p' = c0 + (p−c0)·K`;
        //   • atlas-centre sampling: floor-snap each device sample to the atlas grid phased by
        //     `frac(ink_min)` — `floor(d−f)+0.5+f` (bounded ±0.5);
        //   • a `+0.5/W` half-texel ALONG the gradient axis (`0.5·upem/W · cd/|cd|`).
        // A CONCENTRIC two-circle (c0==c1, axis=0 — common in COLR emoji, an annular radial) has no axis
        // for the half-texel and takes a different (radial) realization, so the focal correction above is
        // wrong for it; leave those at the plain two-circle solve (unchanged — never worse).
        let cd = [c1[0] - c0[0], c1[1] - c0[1]];
        let axis = (cd[0] * cd[0] + cd[1] * cd[1]).sqrt();
        let upem = self.font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f32;
        let tile = ink_min.unwrap_or([cov.left as f32, cov.top as f32]).map(|v| (v - 2.0).round_ties_even());
        let world = [b2d[0]*upem, b2d[1]*upem, -b2d[2]*upem, -b2d[3]*upem, b2d[4]-tile[0], b2d[5]-tile[1]];
        let mut mapping = invert(&world);
        let unit = 1.0 / upem;
        mapping[4] -= c0[0]*unit; mapping[5] += c0[1]*unit;
        let angle = crate::crt::system_crt::atan2f(-cd[1], cd[0]);
        let (sin, cos) = (crate::crt::system_crt::sinf(-angle), crate::crt::system_crt::cosf(-angle));
        mapping = mul(&[cos, sin, -sin, cos, 0.0, 0.0], &mapping);
        let (span, left) = if matches!(extend, Extend::Pad) { (1022.0, 1.0) } else { (1024.0, 0.0) };
        mapping = mapping.map(|v| v * span);
        mapping[4] += left;
        mapping = crate::warp::plane::warp_fit_affine(&mapping.map(|v| v / 1024.0));
        let radius = r0 * unit;
        let dr = r1 * unit - radius;
        let dx = c0[0]*unit - c1[0]*unit;
        let dy = c0[1]*unit - c1[1]*unit;
        let distance = (dx*dx + dy*dy).sqrt();

        self.fill_coverage(&cov, false, |x, y| {
            let p = [mapping[4] + (x-tile[0])*mapping[0] + (y-tile[1])*mapping[2], mapping[5] + (x-tile[0])*mapping[1] + (y-tile[1])*mapping[3]];
            let a = distance * distance - dr * dr;
            let b = p[0] * distance + dr * radius;
            let c = p[0]*p[0] + p[1]*p[1] - radius * radius;
            let disc = b*b - c*a;
            if disc < 0.0 { return None; }
            let sign = if a >= 0.0 || dr <= 0.0 { 1.0 } else { -1.0 };
            let t = (sign * disc.sqrt() + b) / a;
            if dr*t + radius < 0.0 { return None; }
            Some(gb.sample_ramp_text(t))
        });
    }

    /// Fill `gid`'s outline with a COLRv1 sweep gradient — native's internal sweep-gradient color source.
    /// `t` per pixel is the clockwise angle about the sweep centre `c0` (skrifa/COLR convention; the
    /// brush space is y-up, so the standard counter-clockwise `atan2` is negated), normalised into the
    /// `[start,end]` degree span, fed through the shared ramp [`Self::fill_gradient`] (`W` = the device
    /// radius of the footprint, so the ramp is sampled at device granularity, like the conical's).
    ///
    /// ponytail: byte-parity is blocked on two upstream things, so this is correct-by-construction, not
    /// yet exact — (1) the shared gradient-sampler ±1-2 LSB floor (`brush/mod.rs`, the radial agent's
    /// fix, not on main) and (2) WARP's own sweep `atan2`, which lives in the sweep-gradient color source
    /// pixel shader (a VM-capture target) rather than `warpmath`; `f32::atan2` stands in until that lands.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn fill_sweep(&mut self, gid: GlyphId, c0: [f32; 2], start_deg: f32, end_deg: f32, stops: &[ColorStop], extend: Extend, b2d: &Mat, brush_em: Option<Mat>) {
        let span = end_deg - start_deg;
        if span == 0.0 {
            return;
        }
        // Like the conical, native's sweep-gradient color source binds the BASE mip level (1024-wide ramp;
        // SRV-bind capture: colorfont2-09 sweep → MostDetailedMip=0) — the angle varies too sharply for a
        // narrow mip. Footprint ≥ 1024 → `radial_width` = 1024 (the base level).
        let gb = PBrush::ramp(&self.stops(stops), 0, extend_int(extend), 1024.0);
        let upem = self.font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f32;
        let (cov, ink_min) = self.leaf_coverage(gid);
        let tile = ink_min.unwrap_or([cov.left as f32, cov.top as f32]).map(|v| (v - 2.0).round_ties_even());
        let mut local = *b2d;
        local[4] -= tile[0]; local[5] -= tile[1];
        // On the paint-tree image the colour source is placed through d2d1's em-unit paint transform, from
        // DirectWrite's em-unit centre (x·k, y·−k), as the radial and linear brushes are.
        let mapping = match self.em_world(tile, brush_em) {
            Some(w) => sweep_mapping_world([c0[0] * (1.0 / upem), c0[1] * -(1.0 / upem)], &w, extend),
            None => sweep_mapping(c0, &local, upem, extend),
        };
        // An outline leaf (past 100 px) takes the sweep colour source's byte truncated, not rounded, as WARP's
        // geometry gradient shaders store it (r25 30138, 30853, 30908: the rounded sample sat a level high on
        // ramp steps; radial and linear leaves already sample truncated through the brush).
        let trunc = self.feather || self.leaf_outline();
        self.fill_coverage(&cov, false, |x, y| {
            let p = [mapping[4] + (x-tile[0])*mapping[0] + (y-tile[1])*mapping[2], mapping[5] + (x-tile[0])*mapping[1] + (y-tile[1])*mapping[3]];
            let s = gb.sample_ramp_text(sweep_parameter(p, start_deg, end_deg));
            Some(if trunc { s.map(f32::floor) } else { s })
        });
    }
}

impl ColorPainter for Painter<'_> {
    fn push_transform(&mut self, transform: Transform) {
        let m = mul(&self.cur(), &mat_of(&transform));
        self.stack.push(m);
        let k = 1.0 / self.font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f32;
        let em = self.pending_em.take().unwrap_or_else(|| em_of(&mat_of(&transform), k));
        self.em_ops.push(em);
    }
    fn pop_transform(&mut self) {
        self.stack.pop();
        self.em_ops.pop();
    }

    // Native clips the completed subtree once. Isolating leaves preserves this mask, but adding
    // those separately clipped images does not reproduce the full glyph (emoji 060/061).
    // Transform and intersect the box here, then apply its coverage when the frame pops.
    fn push_clip_box(&mut self, clip_box: BoundingBox<f32>) {
        let m = self.cur();
        let corners = [
            apply(&m, clip_box.x_min, clip_box.y_min),
            apply(&m, clip_box.x_max, clip_box.y_min),
            apply(&m, clip_box.x_max, clip_box.y_max),
            apply(&m, clip_box.x_min, clip_box.y_max),
        ];
        let (mut l, mut t, mut r, mut b) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in corners {
            l = l.min(p[0]);
            t = t.min(p[1]);
            r = r.max(p[0]);
            b = b.max(p[1]);
        }
        let rect = match self.clip.last() {
            Some(&[cl, ct, cr, cb]) => [l.max(cl), t.max(ct), r.min(cr), b.min(cb)],
            None => [l, t, r, b],
        };
        self.clip.push(rect);
        // Off the axes and the quarter turns the box is a turned quad, and the group is masked by that quad's
        // antialiased fill (VM probe-rg: a Segoe UI Emoji face turned 30 or 237 degrees loses rim coverage in
        // proportion to the quad's pixel coverage; 25-31 px off through the bounding rectangle, 0-2 this way).
        let off_axis = |a: f32, b: f32| a.abs() > 1e-6 && b.abs() > 1e-6;
        let quad = (off_axis(m[0], m[1]) || off_axis(m[2], m[3])).then_some(corners);
        self.frames.push(Frame::ClipBox { buf: self.new_buf(), rect, quad });
    }
    // A `PaintGlyph` wrapping a non-brush subtree (a nested clip): render the subtree into an offscreen
    // and intersect it with this glyph's coverage when the clip pops (native `RenderGlyphNode`).
    fn push_clip_glyph(&mut self, glyph_id: GlyphId) {
        let (buf, mask) = (self.new_buf(), self.clip_mask(glyph_id));
        self.frames.push(Frame::Clip { buf, mask });
    }
    fn pop_clip(&mut self) {
        match self.frames.pop() {
            Some(Frame::Clip { buf, mask }) => clip_down(&buf, self.active_buf(), &mask),
            Some(Frame::ClipBox { buf, rect, quad }) => {
                self.clip.pop();
                let quad_mask = quad.map(|q| {
                    let mut m = self.new_buf();
                    Target::new(&mut m).fill_path(&[q.to_vec()], crate::pixels::FillMode::Winding, [255; 4], Aa::PerPrimitive, Blend::SourceOver, true);
                    m
                });
                let dst = self.active_buf();
                for y in 0..dst.desc.height as i32 {
                    for x in 0..dst.desc.width as i32 {
                        let i = y as usize * dst.stride + x as usize * 4;
                        // PopLayerRectOp masks the group through the rectangle AA byte (round-half-down
                        // coverage) in WARP's gradient-mask kernel, whose UNORM8 products take the
                        // high-byte correction before the +128 bias (textx-emoji-023/054/187/271).
                        let mask = match &quad_mask {
                            Some(m) => m.data[i + 3] as i32,
                            None => (box_cov(Some(rect), x, y) * 255.0 - 0.5).ceil() as i32,
                        };
                        let sc: [i32; 4] = core::array::from_fn(|k| warp_unorm_mul(buf.data[i + k] as i32, mask));
                        let ia = 255 - sc[3];
                        for k in 0..4 {
                            dst.data[i + k] = (sc[k] + warp_unorm_mul(dst.data[i + k] as i32, ia)).clamp(0, 255) as u8;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn fill_glyph(&mut self, glyph_id: GlyphId, brush_transform: Option<Transform>, brush: CBrush<'_>) {
        let this_leaf = self.leaf_idx;
        self.leaf_idx += 1;
        let pending_brush_em = self.pending_brush_em.take();
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Ok(only) = std::env::var("COLR_LEAF_ONLY") {
                if only.parse::<u32>().ok() != Some(this_leaf) {
                    return;
                }
            }
            if std::env::var_os("COLR_LEAF_SOLID").is_some() {
                self.fill_outline(glyph_id, &PBrush::solid([0.0, 0.0, 0.0, 1.0], 1.0));
                return;
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Ok(lim) = std::env::var("COLR_LEAF_LIMIT") {
            if lim.parse::<u32>().is_ok_and(|k| this_leaf >= k) {
                return;
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        if std::env::var_os("COLR_DUMP_TREE").is_some() {
            let m = self.cur();
            let stopsfmt = |cs: &[ColorStop]| cs.iter().map(|s| { let c = self.color(s.palette_index, s.alpha); format!("{:.3}:rgba({:.0},{:.0},{:.0},{:.2})", s.offset, c[0]*255.0, c[1]*255.0, c[2]*255.0, c[3]) }).collect::<Vec<_>>().join(" ");
            let kind = match &brush {
                CBrush::Solid { palette_index, alpha } => format!("Solid pal={palette_index} a={alpha}"),
                CBrush::LinearGradient { p0, p1, color_stops, extend } => format!("Linear p0=({:.0},{:.0}) p1=({:.0},{:.0}) ext={extend:?} stops=[{}]", p0.x, p0.y, p1.x, p1.y, stopsfmt(color_stops)),
                CBrush::RadialGradient { c0, r0, c1, r1, extend, color_stops } => format!("Radial c0={c0:?} r0={r0} c1={c1:?} r1={r1} ext={extend:?} stops=[{}]", stopsfmt(color_stops)),
                CBrush::SweepGradient { extend, color_stops, .. } => format!("Sweep ext={extend:?} stops=[{}]", stopsfmt(color_stops)),
            };
            let bb = self.outline_polys(glyph_id).map(|ps| {
                let (mut l, mut t, mut r, mut b) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
                for p in &ps { for v in p { l = l.min(v[0]); t = t.min(v[1]); r = r.max(v[0]); b = b.max(v[1]); } }
                [l, t, r, b]
            });
            eprintln!("LEAF gid={} cur=[{:.8},{:.8},{:.8},{:.8},{:.6},{:.6}] bbox={bb:?} brush={kind}", glyph_id.to_u32(), m[0], m[1], m[2], m[3], m[4], m[5]);
        }
        // A turned group's layer is cut to its leaves' glyph boxes (`turned_back`). A radial with no outer radius
        // draws nothing and adds none.
        // ponytail: the box is the outline's design bounds; an emboldened leaf's widening is left out (unprobed).
        if !matches!(&brush, CBrush::RadialGradient { r1, .. } if *r1 <= 0.0) {
            let bb = self.font.glyph_metrics(Size::unscaled(), LocationRef::default()).bounds(glyph_id);
            let r = bb.map(|b| map_bounds(&self.leaf_cur(), [b.x_min, b.y_min, b.x_max, b.y_max]));
            let acc = self.frames.iter_mut().find_map(|f| match f {
                Frame::Layer { turn: Some((_, acc)), .. } => Some(acc),
                _ => None,
            });
            if let (Some(r), Some(acc)) = (r, acc) {
                *acc = [acc[0].min(r[0]), acc[1].min(r[1]), acc[2].max(r[2]), acc[3].max(r[3])];
            }
        }
        let b2d = match brush_transform {
            Some(t) => mul(&self.cur(), &mat_of(&t)),
            None => self.cur(),
        };
        // A radial that is NOT native's D2D-radial-brush case — `r0 ≠ 0`, or the focal point outside
        // the outer circle — is drawn by native's internal conical-gradient two-circle shader
        // Native maps it as a radial-gradient brush. Route it to `fill_conical`.
        if let CBrush::RadialGradient { c0, r0, c1, r1, color_stops, extend } = &brush {
            if *r1 <= 0.0 {
                return; // native returns early; nothing drawn.
            }
            let d2 = (c0.x - c1.x) * (c0.x - c1.x) + (c0.y - c1.y) * (c0.y - c1.y);
            if *r0 != 0.0 || *r1 * *r1 <= d2 {
                self.fill_conical(glyph_id, [c0.x, c0.y], *r0, [c1.x, c1.y], *r1, color_stops, *extend, &b2d);
                return;
            }
        }
        let k = 1.0 / self.font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f32;
        let brush_em = brush_transform.map(|t| pending_brush_em.unwrap_or_else(|| em_of(&mat_of(&t), k)));
        if let CBrush::SweepGradient { c0, start_angle, end_angle, color_stops, extend } = &brush {
            self.fill_sweep(glyph_id, [c0.x, c0.y], *start_angle, *end_angle, color_stops, *extend, &b2d, brush_em);
            return;
        }
        if let Some(b) = self.brush(&brush, &b2d, brush_em) {
            self.fill_outline(glyph_id, &b);
        }
    }

    // A bare `fill` (a gradient/solid directly under a clip, e.g. `PaintGlyph(clip, gradient)`) would
    // fill the whole active clip offscreen with the brush, masked when the clip pops. It never occurs
    // in the corpus, and the clip scenes use a glyph-leaf child (routed through `fill_glyph`), so it
    // stays unsupported. ponytail: fill the active buffer with a full-extent rect brush if a font needs it.
    fn fill(&mut self, _brush: CBrush<'_>) {
        self.unsupported = true;
    }

    // COLRv1 `PaintComposite`: skrifa brackets the backdrop and source in layers and composites the
    // source with `composite_mode` (`skrifa` traversal.rs: push SrcOver, paint backdrop, push `mode`,
    // paint source, pop `mode`, pop SrcOver). Native renders each to a command-list offscreen and the
    // Composite (Porter-Duff) / Blend effect merges them (`RenderCompositeNode`). Each `push_layer`
    // opens an offscreen; `pop_layer_with_mode` composites it onto the group below with that mode,
    // reusing the byte-exact `crate::layer::composite` table and `effects::builtin::blend` shader.
    fn push_layer(&mut self, composite_mode: CompositeMode) {
        let buf = self.new_buf();
        let next_cell = self.next_cell.take();
        // An unsnapped run (OUTLINE, or past 128 device px) renders a composite group into an intermediate placed at
        // the glyph origin's whole pixel and draws it back bilinearly at the sub-pixel remainder (VM: composites at
        // x+0.25/0.5/0.75, y+0.3 match a bilinear shift of the whole-pixel render to ±0.002, across 96-192 dpi).
        let outermost = !self.frames.iter().any(|f| matches!(f, Frame::Layer { .. }));
        let shift = (outermost && self.regroup).then(|| {
            let r = self.raw_origin_dev;
            [r[0] - r[0].floor(), r[1] - r[1].floor()]
        });
        if let Some(f) = shift {
            // Render the group with the glyph origin on its whole pixel.
            let (m, o, r) = (self.cur(), self.origin_dev, self.raw_origin_dev);
            let d = [r[0] - f[0] - o[0], r[1] - f[1] - o[1]];
            self.stack.push([m[0], m[1], m[2], m[3], m[4] + d[0], m[5] + d[1]]);
            // Its backdrop and source each render into an atlas cell over the composite's whole-pixel bounds cut to
            // what the draw-back reads: the target (the buffer less its 1 px margin), and the row or column before it
            // on an axis drawn back at a fraction.
            let (w, h) = (self.target.desc.width as f32, self.target.desc.height as f32);
            let lo = f.map(|v| if v > 0.0 { 0.0 } else { 1.0 });
            self.cell = next_cell.map(|b| [b[0].floor().max(lo[0]), b[1].floor().max(lo[1]), b[2].ceil().min(w - 1.0), b[3].ceil().min(h - 1.0)])
                .filter(|c| c[0] < c[2] && c[1] < c[3]);
            let ext = f.map(|v| if v > 0.0 { 1.0 } else { 0.0 });
            self.vp_origins = self.pd_origins(next_cell, lo, ext);
        }
        // Turned, the group renders upright with the glyph origin on a whole pixel of its own buffer, which is then
        // drawn back through the rest of the matrix bilinearly (VM probe: Xor and Multiply groups at em 116 turned 30°
        // match an upright render turned about its whole-pixel origin, against 908 px drawn straight).
        let (buf, turn) = match self.turn.filter(|_| outermost) {
            Some(r) => {
                let (ri, lo, hi, span) = self.turn_frame(r);
                let to_buf = |org: [f32; 2]| turn_to_buf(&ri, lo, org);
                let (o, raw) = (self.origin_dev, self.raw_origin_dev);
                let m = self.cur();
                self.stack.push(mul(&to_buf(o), &m));
                // Its backdrop and source render into atlas cells as on the regroup path, over the composite's bounds in
                // this upright buffer cut to the target's preimage there, the region the draw-back reads (VM DDI of r25
                // 30126's group under a shear: 125x64 cells; geo/c126u: 19 of 20 sheared, turned and scaled probes exact).
                let cut = [span[0] - lo[0], span[1] - lo[1], span[2] - lo[0], span[3] - lo[1]];
                self.cell = next_cell.map(|b| [b[0].floor().max(cut[0].floor()), b[1].floor().max(cut[1].floor()), b[2].ceil().min(cut[2].ceil()), b[3].ceil().min(cut[3].ceil())])
                    .filter(|c| c[0] < c[2] && c[1] < c[3]);
                self.vp_origins = self.pd_origins(next_cell, [cut[0].floor(), cut[1].floor()], [1.0, 1.0]);
                // The leaves then take the path their size in this buffer calls for, not their device size: a group
                // under [2.64, 0.97] at em 60 is drawn at 60 px and scanned, though the run reaches 158 px on the
                // device (VM probe-an2, anisotropic groups at em 53-91 that pass 128 px on the device: 420-969 px off
                // through the outline feather, 3-98 scanned; groups at em 110 and 140 are unchanged).
                let upem = self.font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f32;
                let em = crate::extras::print::compute_maximum_scale_factor(self.cur()) * upem;
                self.run_feather = Some(self.feather);
                self.feather = self.feather && em > 128.0;
                self.canvas = Some(ImageDesc { width: (hi[0] - lo[0]) as u32, height: (hi[1] - lo[1]) as u32, ..self.target.desc });
                (self.new_buf(), Some((to_buf(raw), [f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY])))
            }
            None => (buf, None),
        };
        self.frames.push(Frame::Layer { buf, mode: composite_mode as u8, shift, turn });
    }
    fn pop_layer_with_mode(&mut self, _composite_mode: CompositeMode) {
        if let Some(Frame::Layer { buf, mode, shift, turn }) = self.frames.pop() {
            if !self.frames.iter().any(|f| matches!(f, Frame::Layer { .. })) {
                self.cell = None;
                self.vp_origins = None;
            }
            match (shift, turn) {
                (Some(f), _) => {
                    self.stack.pop();
                    composite_buf(&bilinear_shift(&buf, f), self.active_buf(), mode);
                }
                (None, Some((to_buf, leaves))) => {
                    self.stack.pop();
                    if let Some(f) = self.run_feather.take() {
                        self.feather = f;
                    }
                    self.canvas = None;
                    let dst = self.active_buf();
                    let axis = to_buf[1] == 0.0 && to_buf[2] == 0.0 && to_buf[0] > 0.0 && to_buf[3] > 0.0;
                    let back = if axis { draw_back(&buf, &to_buf, dst.desc) } else { turned_back(&buf, &to_buf, leaves, dst.desc) };
                    composite_buf(&back, dst, mode);
                }
                (None, None) => composite_buf(&buf, self.active_buf(), mode),
            }
        }
    }
}

// The sweep-gradient realization placement + texture mapping.
fn sweep_mapping(center: [f32; 2], b2d: &Mat, upem: f32, extend: Extend) -> Mat {
    let world = [b2d[0] * upem, b2d[1] * upem, -b2d[2] * upem, -b2d[3] * upem, b2d[4], b2d[5]];
    sweep_mapping_world([center[0] / upem, -center[1] / upem], &world, extend)
}

/// [`sweep_mapping`] for an em-unit `center` under the em-unit → device `world`.
fn sweep_mapping_world(center: [f32; 2], world: &Mat, extend: Extend) -> Mat {
    let world = *world;
    let c = apply(&world, center[0], center[1]);
    let xe = apply(&world, center[0] + 1024.0, center[1]);
    let ye = apply(&world, center[0], center[1] + 1024.0);
    let span = if matches!(extend, Extend::Pad) { 1023.0 } else { 1024.0 };
    let reciprocal = 1.0 / (span * span);
    let rect = [
        (span * xe[0] - span * c[0]) * reciprocal,
        (span * xe[1] - span * c[1]) * reciprocal,
        (span * ye[0] - span * c[0]) * reciprocal,
        (span * ye[1] - span * c[1]) * reciprocal,
        (span * span * c[0]) * reciprocal,
        (span * span * c[1]) * reciprocal,
    ];
    crate::warp::plane::warp_fit_affine(&invert(&rect).map(|v| v * (1.0 / 1024.0)))
}

// The sweep-gradient text pixel shader, instructions 12–48.
fn sweep_parameter(p: [f32; 2], start: f32, end: f32) -> f32 {
    let [x, y] = p;
    let ratio = (1.0 / x.abs().max(y.abs())) * x.abs().min(y.abs());
    let square = ratio * ratio;
    let mut poly = square * 0.0208351 - 0.085133;
    poly = square * poly + 0.180141;
    poly = square * poly - 0.3302995;
    poly = square * poly + 0.999866;
    let mut angle = ratio * poly;
    if x.abs() < y.abs() { angle += angle * -2.0 + f32::from_bits(0x3fc90fdb); }
    if x < -x { angle += f32::from_bits(0xc0490fdb); }
    if x.min(y) < -x.min(y) && x.max(y) >= -x.max(y) { angle = -angle; }
    let degrees = if angle < 0.0 { angle * 57.29578 + 360.0 } else { angle * 57.29578 };
    let clockwise = 360.0 - degrees;
    let span = (end - start).abs();
    if span == 0.0 { return if start < clockwise { 1.0 } else { 0.0 }; }
    let t = (clockwise - start.min(end)) / span;
    if end < start { 1.0 - t } else { t }
}

// DirectWrite's glyph scaling computation (FRound/SRound/FixRound).
fn scale_leaf_point(value: i32, em: i32, upem: i32) -> i32 {
    let denom = upem << 16;
    let shift = (em | denom).trailing_zeros().saturating_sub(1);
    let (mut numerator, mut divisor) = (em >> shift, denom >> shift);
    if numerator >= 0x2000000 { divisor >>= 6; } else { numerator <<= 6; }
    if numerator > 0x7fff {
        super::hinting::fixmul(value, super::hinting::fixdiv(numerator, divisor))
    } else if divisor > 0 && (divisor & (divisor - 1)) == 0 {
        ((numerator as i64 * value as i64 + (divisor >> 1) as i64) >> divisor.trailing_zeros()) as i32
    } else {
        let product = numerator as i64 * value as i64;
        let rounded = (product.abs() + (divisor >> 1) as i64) / divisor as i64;
        if value < 0 { -rounded as i32 } else { rounded as i32 }
    }
}

/// Collects a glyph's `skrifa` outline into [`Figure`]s in raw design units (y-up); the painter's
/// transform does the scale, y-flip and placement.
struct RawPen<'a> {
    figures: &'a mut Vec<Figure>,
    cur: Option<Figure>,
}
impl RawPen<'_> {
    fn flush(&mut self) {
        if let Some(f) = self.cur.take() {
            self.figures.push(f);
        }
    }
}
impl OutlinePen for RawPen<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        self.flush();
        self.cur = Some(Figure { start: [x, y], filled: true, closed: true, segments: Vec::new(), seg_flags: Vec::new() });
    }
    fn line_to(&mut self, x: f32, y: f32) {
        if let Some(f) = &mut self.cur {
            f.segments.push(Segment::Line([x, y]));
        }
    }
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        if let Some(f) = &mut self.cur {
            f.segments.push(Segment::Quadratic([cx, cy], [x, y]));
        }
    }
    fn curve_to(&mut self, c0x: f32, c0y: f32, c1x: f32, c1y: f32, x: f32, y: f32) {
        if let Some(f) = &mut self.cur {
            f.segments.push(Segment::Cubic([c0x, c0y], [c1x, c1y], [x, y]));
        }
    }
    fn close(&mut self) {}
}

/// Collects a glyph's `skrifa` outline into device-space contours (points + on-curve flags) under the
/// transform `m` — the (points, on_curve) pairs [`raster::coverage_outline`] scan-converts. `skrifa`
/// emits an explicit on-curve end per quadratic, so off/on alternate as the TrueType walk expects.
struct ContourPen {
    m: Mat,
    contours: Vec<(Vec<[f64; 2]>, Vec<bool>)>,
    pts: Vec<[f64; 2]>,
    on: Vec<bool>,
}
impl ContourPen {
    fn xf(&self, x: f32, y: f32) -> [f64; 2] {
        let p = apply(&self.m, x, y);
        [p[0] as f64, p[1] as f64]
    }
    fn push(&mut self, x: f32, y: f32, on: bool) {
        let p = self.xf(x, y);
        self.pts.push(p);
        self.on.push(on);
    }
    fn flush(&mut self) {
        if !self.pts.is_empty() {
            self.contours.push((std::mem::take(&mut self.pts), std::mem::take(&mut self.on)));
        }
    }
}
impl OutlinePen for ContourPen {
    fn move_to(&mut self, x: f32, y: f32) {
        self.flush();
        self.push(x, y, true);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.push(x, y, true);
    }
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        self.push(cx, cy, false);
        self.push(x, y, true);
    }
    fn curve_to(&mut self, c0x: f32, c0y: f32, c1x: f32, c1y: f32, x: f32, y: f32) {
        // ponytail: TrueType glyf (emoji / colrtest) has no cubics; split into two quads via the 3/4
        // control rule so a CFF COLR base would still scan-convert rather than panic.
        let last = *self.pts.last().unwrap_or(&[0.0, 0.0]);
        let (a, b, e) = (self.xf(c0x, c0y), self.xf(c1x, c1y), self.xf(x, y));
        let q0 = [(last[0] + 3.0 * a[0]) / 4.0, (last[1] + 3.0 * a[1]) / 4.0];
        let q1 = [(e[0] + 3.0 * b[0]) / 4.0, (e[1] + 3.0 * b[1]) / 4.0];
        let mid = [(q0[0] + q1[0]) / 2.0, (q0[1] + q1[1]) / 2.0];
        for (p, o) in [(q0, false), (mid, true), (q1, false), (e, true)] {
            self.pts.push(p);
            self.on.push(o);
        }
    }
    fn close(&mut self) {}
}

// --- COLRv0 run decomposition (IDWriteFactory2::TranslateColorGlyphRun) ------------------------

/// One COLRv0 layer: a glyph painted in a CPAL palette colour (index `0xFFFF` = the run's foreground
/// colour). The enumerator in `dwrite-ffi` turns each into a `DWRITE_COLOR_GLYPH_RUN` sub-run.
pub struct ColorLayer {
    pub glyph: u16,
    pub palette_index: u16,
}

/// Decompose `gid`'s COLRv0 colour layers (glyph + CPAL palette index, in paint order), the way
/// `TranslateColorGlyphRun` expands a colour glyph. `None` if the glyph has no colour record or uses
/// COLRv1 features (transforms, gradients, clips, compositing) a flat colour run can't express —
/// those are the `DWRITE_COLOR_GLYPH_RUN1`/paint-tree path, not this one. Additive: it only reads
/// the font's COLR/CPAL tables and shares nothing with the render path above.
pub fn colrv0_layers(font: &FontRef, gid: GlyphId) -> Option<Vec<ColorLayer>> {
    // The COLRv0 record specifically — `get` prefers COLRv1 when a glyph has both (modern emoji do),
    // but TranslateColorGlyphRun (F2) expands the flat v0 layers.
    let cg = ColorGlyphCollection::new(font).get_with_format(gid, ColorGlyphFormat::ColrV0)?;
    let mut cap = LayerCapture { layers: Vec::new(), v1: false };
    cg.paint(LocationRef::default(), &mut cap).ok()?;
    (!cap.v1 && !cap.layers.is_empty()).then_some(cap.layers)
}

/// Captures a COLRv0 glyph's flat solid layers via skrifa's paint walk; any COLRv1-only callback
/// (transform / clip / gradient fill / compositing layer) sets `v1`, disqualifying the glyph.
struct LayerCapture {
    layers: Vec<ColorLayer>,
    v1: bool,
}

impl ColorPainter for LayerCapture {
    fn push_transform(&mut self, _t: Transform) {
        self.v1 = true;
    }
    fn pop_transform(&mut self) {}
    fn push_clip_glyph(&mut self, _g: GlyphId) {
        self.v1 = true;
    }
    fn push_clip_box(&mut self, _b: BoundingBox<f32>) {
        self.v1 = true;
    }
    fn pop_clip(&mut self) {}
    fn fill(&mut self, _brush: CBrush) {
        self.v1 = true;
    }
    fn fill_glyph(&mut self, glyph_id: GlyphId, brush_transform: Option<Transform>, brush: CBrush) {
        match brush {
            CBrush::Solid { palette_index, .. } if brush_transform.is_none() => self.layers.push(ColorLayer { glyph: glyph_id.to_u32() as u16, palette_index }),
            _ => self.v1 = true,
        }
    }
    fn push_layer(&mut self, _c: CompositeMode) {
        self.v1 = true;
    }
    fn pop_layer(&mut self) {}
}

/// Whether a colour glyph's paint graph composites (calls `push_layer`).
struct HasComposite(bool);

/// The device bounds of a paint tree without a `ClipBox`: its leaf glyph boxes under their transforms.
// ponytail: the glyph boxes, not each leaf's clipped outline; the bitmap origin they give only re-frames gradient
// placement, so a box a pixel off changes nothing measured.
struct PaintBounds<'a> {
    stack: Vec<Mat>,
    font: &'a FontRef<'a>,
    acc: Option<[f32; 4]>,
}

impl ColorPainter for PaintBounds<'_> {
    fn push_transform(&mut self, t: Transform) {
        let m = mul(self.stack.last().unwrap(), &mat_of(&t));
        self.stack.push(m);
    }
    fn pop_transform(&mut self) {
        self.stack.pop();
    }
    fn push_clip_glyph(&mut self, g: GlyphId) {
        let Some(b) = self.font.glyph_metrics(Size::unscaled(), LocationRef::default()).bounds(g) else { return };
        let r = map_bounds(self.stack.last().unwrap(), [b.x_min, b.y_min, b.x_max, b.y_max]);
        self.acc = Some(self.acc.map_or(r, |a| [a[0].min(r[0]), a[1].min(r[1]), a[2].max(r[2]), a[3].max(r[3])]));
    }
    fn push_clip_box(&mut self, _b: BoundingBox<f32>) {}
    fn pop_clip(&mut self) {}
    fn fill(&mut self, _brush: CBrush) {}
    fn push_layer(&mut self, _c: CompositeMode) {}
    fn pop_layer(&mut self) {}
}

impl ColorPainter for HasComposite {
    fn push_transform(&mut self, _t: Transform) {}
    fn pop_transform(&mut self) {}
    fn push_clip_glyph(&mut self, _g: GlyphId) {}
    fn push_clip_box(&mut self, _b: BoundingBox<f32>) {}
    fn pop_clip(&mut self) {}
    fn fill(&mut self, _brush: CBrush) {}
    fn push_layer(&mut self, _c: CompositeMode) {
        self.0 = true;
    }
    fn pop_layer(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_shift_spreads_a_pixel_right_and_down() {
        let mut buf = crate::context::bitmap_target(4, 4, 96.0);
        buf.data.fill(0);
        let i = buf.stride + 4;
        buf.data[i..i + 4].copy_from_slice(&[200, 100, 40, 200]);
        let out = bilinear_shift(&buf, [0.25, 0.5]);
        let px = |x: usize, y: usize| out.data[y * out.stride + x * 4 + 3];
        // Alpha 200 split 0.75/0.25 across x and 0.5/0.5 across y.
        assert_eq!([px(1, 1), px(2, 1), px(1, 2), px(2, 2)], [75, 25, 75, 25]);
        assert_eq!(px(0, 0), 0);
    }

    #[test]
    fn rotated_origin_snaps_in_device_space() {
        // The rotated two-rect probe at 120 dpi: baseline (-20.3, 20.6) lands at device (115.416, 87.257).
        let r = [-0.544639035015027f64, -0.838670567945424, 0.838670567945424, -0.544639035015027];
        let dt = [r[0] * 1.25, r[1] * 1.25, r[2] * 1.25, r[3] * 1.25, 80.0, 80.0];
        let dev = |o: [f32; 2]| [dt[0] * o[0] as f64 + dt[2] * o[1] as f64 + dt[4], dt[1] * o[0] as f64 + dt[3] * o[1] as f64 + dt[5]];
        let quarter = dev(snap_origin([-20.3, 20.6], &dt, 4, 4));
        assert!((quarter[0] - 115.5).abs() < 1e-4 && (quarter[1] - 87.25).abs() < 1e-4, "{quarter:?}");
        let whole = dev(snap_origin([-20.3, 20.6], &dt, 1, 1));
        assert!((whole[0] - 115.0).abs() < 1e-4 && (whole[1] - 87.0).abs() < 1e-4, "{whole:?}");
    }

    // Requires the same optional system-font fixtures as conformance. Captured through DrawText
    // with D2D_LEAF_ONLY=21, both with the native gradient and D2D_LEAF_SOLID=1. No font is embedded.
    #[test]
    fn enlarged_leaf_brush_coverage() {
        let Some(dir) = std::env::var_os("D2D_FONTS") else { return };
        let bytes = std::fs::read(std::path::Path::new(&dir).join("seguiemj.ttf")).unwrap();
        let font = FontRef::new(&bytes).unwrap();
        let mut target = crate::bitmap_target(48, 48, 96.0);
        let m = [0.0, 1.7940673828125 / 2048.0, -127.058349609375 / 2048.0, 0.0, 95.12593841552734, 9.556124687194824];
        let mut painter = Painter {
            target: &mut target, font: &font, colors: &[], fg: [0.0, 0.0, 0.0, 1.0],
            stack: vec![m], frames: Vec::new(), aa: Aa::PerPrimitive, unsupported: false,
            feather: false, run_feather: None, clip: Vec::new(), hints: RefCell::new(HashMap::new()), leaf_idx: 0, pending_linear: None, bold: false, shear: 0.0, origin_dev: [0.0; 2], raw_origin_dev: [0.0; 2], paint_origin: [0.0; 2], replay_em: None, em_base: None, em_ops: Vec::new(), pending_em: None, pending_brush_em: None, pending_radial: None, regroup: false, turn: None, canvas: None, cell: None, next_cell: None, next_src: None, next_pd: false, vp_origins: None,
        };
        assert!(painter.leaf_outline());
        let glyph = GlyphId::new(18372);
        let (cov, _) = painter.leaf_coverage(glyph);
        let at = |c: &AlphaTexture, x: i32, y: i32| c.data[((y - c.top) * c.width() + x - c.left) as usize];
        assert_eq!([at(&cov, 9, 10), at(&cov, 9, 11)], [64, 96]);
        let feather = painter.feather_layer(glyph, [0.0, 0.0, 0.0, 255.0]).unwrap();
        assert_eq!([(10 * 48 + 9) * 4 + 3, (11 * 48 + 9) * 4 + 3].map(|i| feather.data[i]), [116, 202]);
        painter.fill_outline(glyph, &PBrush::solid([0.0, 0.0, 0.0, 1.0], 1.0));
        assert_eq!(painter.target.data[(11 * 48 + 9) * 4 + 3], 202);
        painter.target.data.fill(0);
        // A translucent solid takes the geometry fill's coverage; the feather's would give 153·202/255 = 121.
        painter.fill_outline(glyph, &PBrush::solid([0.0, 0.0, 0.0, 0.6], 1.0));
        assert_eq!(painter.target.data[(11 * 48 + 9) * 4 + 3], 58);
        painter.target.data.fill(0);
        let black = [0.0, 0.0, 0.0, 1.0];
        let gradient = PBrush::linear(&[(0.0, black), (1.0, black)], 0, 0, 1.0,
            [0.0, 0.0], [48.0, 0.0], &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        painter.fill_outline(glyph, &gradient);
        assert_eq!(painter.target.data[(11 * 48 + 9) * 4 + 3], 96);
        // Rotation alone must not force outlines when the largest scale remains below the threshold.
        painter.stack[0][2] *= 0.5;
        assert!(!painter.leaf_outline());
    }

    #[test]
    fn gradient_mask_retains_filtered_colour_until_store() {
        let Some(dir) = std::env::var_os("D2D_FONTS") else { return };
        let bytes = std::fs::read(std::path::Path::new(&dir).join("seguiemj.ttf")).unwrap();
        let font = FontRef::new(&bytes).unwrap();
        let mut target = crate::bitmap_target(36, 36, 96.0);
        let m = [6.0 / 2048.0, 0.0, 0.0, -6.0 / 2048.0, 4.0, 8.0];
        let mut painter = Painter {
            target: &mut target, font: &font, colors: &[], fg: [0.0, 0.0, 0.0, 1.0],
            stack: vec![m], frames: Vec::new(), aa: Aa::PerPrimitive, unsupported: false,
            feather: false, run_feather: None, clip: Vec::new(), hints: RefCell::new(HashMap::new()), leaf_idx: 0, pending_linear: None, bold: false, shear: 0.0, origin_dev: [0.0; 2], raw_origin_dev: [0.0; 2], paint_origin: [0.0; 2], replay_em: None, em_base: None, em_ops: Vec::new(), pending_em: None, pending_brush_em: None, pending_radial: None, regroup: false, turn: None, canvas: None, cell: None, next_cell: None, next_src: None, next_pd: false, vp_origins: None,
        };
        let gray = [186.0 / 255.0, 182.0 / 255.0, 194.0 / 255.0, 1.0];
        let stops = [(2133.0 / 16384.0, gray), (4523.0 / 16384.0, [1.0; 4]),
            (9899.0 / 16384.0, [1.0; 4]), (12885.0 / 16384.0, gray)];
        let brush = PBrush::linear(&stops, 0, 0, 1.0, [1808.0, 449.0], [1143.0, 707.0], &m);
        painter.fill_outline(GlyphId::new(16015), &brush);
        // Native emoji-084 leaf 46, isolated on transparent black. Rounding the brush first gives G=104.
        assert_eq!(&painter.target.data[(6 * 36 + 8) * 4..][..4], &[107, 105, 105, 121]);
        // The same native capture's quarter-turned eyes: grid-fit outer eye, unhinted inner eye.
        let black = PBrush::solid([0.0, 0.0, 0.0, 1.0], 1.0);
        painter.target.data.fill(0);
        painter.stack[0] = [0.0, 11.18701171875 / 2048.0, -11.18701171875 / 2048.0, 0.0,
            11.570033073425293, 0.02948904037475586];
        painter.fill_outline(GlyphId::new(15965), &black);
        let alpha = |p: &Painter<'_>, x: usize, y: usize| p.target.data[(y * 36 + x) * 4 + 3];
        assert_eq!([alpha(&painter, 6, 5), alpha(&painter, 7, 5), alpha(&painter, 6, 6), alpha(&painter, 7, 6)], [103, 88, 47, 47]);
        painter.target.data.fill(0);
        painter.stack[0] = [0.0, 2.79656982421875 / 2048.0, -2.79656982421875 / 2048.0, 0.0,
            7.926517963409424, 4.473737716674805];
        painter.fill_outline(GlyphId::new(15965), &black);
        assert_eq!([alpha(&painter, 6, 5), alpha(&painter, 6, 6)], [19, 0]);
    }

    #[test]
    fn fractional_gridfit_scale_matches_native() {
        // Native emoji-162 leaf 0: NewTransform matrix magnitude 1621944 / 65536,
        // gasp grid-fit enabled, mth_IntelMul inputs captured before reflection.
        let design = [1336, 1581, 1948, 2193, 2315, 1092, 725, 480, 358, -236, -84, 185, 540, 742, 945, 1300, 1569, 1721];
        let native = [1033, 1223, 1507, 1696, 1790, 845, 561, 371, 277, -183, -65, 143, 418, 574, 731, 1005, 1213, 1331];
        for (point, expected) in design.into_iter().zip(native) {
            assert_eq!(scale_leaf_point(point, 1621944, 2048), expected);
        }
    }

    #[test]
    fn rotated_leaf_matches_native_fixed_contour() {
        let Some(dir) = std::env::var_os("D2D_FONTS") else { return };
        let bytes = std::fs::read(std::path::Path::new(&dir).join("seguiemj.ttf")).unwrap();
        let font = FontRef::new(&bytes).unwrap();
        let mut target = crate::bitmap_target(48, 48, 96.0);
        let painter = Painter {
            target: &mut target, font: &font, colors: &[], fg: [0.0, 0.0, 0.0, 1.0],
            stack: vec![[11.9520263671875 / 2048.0, -3.202880859375 / 2048.0,
                -3.202880859375 / 2048.0, -11.9520263671875 / 2048.0, 5.600680351257324, 17.190303802490234]],
            frames: Vec::new(), aa: Aa::PerPrimitive, unsupported: false,
            feather: false, run_feather: None, clip: Vec::new(), hints: RefCell::new(HashMap::new()), leaf_idx: 0, pending_linear: None, bold: false, shear: 0.0, origin_dev: [0.0; 2], raw_origin_dev: [0.0; 2], paint_origin: [0.0; 2], replay_em: None, em_base: None, em_ops: Vec::new(), pending_em: None, pending_brush_em: None, pending_radial: None, regroup: false, turn: None, canvas: None, cell: None, next_cell: None, next_src: None, next_pd: false, vp_origins: None,
        };
        let contours = painter.unhinted_leaf(GlyphId::new(4070));
        // Native emoji-326 leaf 0, mth_IntelMul output before 4× overscale and device placement.
        let x = [522,614,735,800,810,790,770,689,571,419,327,236,114,50,40,60,80,161,278,431];
        let y = [45,69,163,287,432,508,584,703,779,799,775,751,657,533,389,313,237,117,41,21];
        assert_eq!(contours[0].0.len(), x.len());
        for (i, p) in contours[0].0.iter().enumerate() {
            assert_eq!(*p, [5.5 + x[i] as f64 / 64.0, 17.25 - y[i] as f64 / 64.0]);
        }
    }

    // Native emoji-060 with opaque-black leaves and only the COLR ClipBox xMax varied. The
    // unclipped group alpha is 253; these captures distinguish 8×8 from analytic and 4×4 clipping.
    #[test]
    fn clip_box_uses_eighth_pixel_samples() {
        for (bound, alpha) in [(2048, 0), (2060, 32), (2070, 32), (2080, 32), (2091, 63),
            (2100, 63), (2105, 95), (2112, 95), (2120, 95), (2130, 126), (2150, 158)] {
            let right = 4.0 + bound as f32 * 12.0 / 2048.0;
            let c = box_cov(Some([0.0, 0.0, right, 48.0]), 16, 7);
            assert_eq!((253.0 * c).round_ties_even() as u8, alpha, "bound={bound}");
        }
    }

    #[test]
    fn hinted_leaf_cache_does_not_depend_on_previous_size() {
        let Some(dir) = std::env::var_os("D2D_FONTS") else { return };
        let bytes = std::fs::read(std::path::Path::new(&dir).join("seguiemj.ttf")).unwrap();
        let font = FontRef::new(&bytes).unwrap();
        let mut target = crate::bitmap_target(48, 48, 96.0);
        let mut painter = Painter {
            target: &mut target, font: &font, colors: &[], fg: [0.0, 0.0, 0.0, 1.0],
            stack: vec![[24.75 / 2048.0, 0.0, 0.0, -24.75 / 2048.0, 4.0, 28.0]],
            frames: Vec::new(), aa: Aa::PerPrimitive, unsupported: false,
            feather: false, run_feather: None, clip: Vec::new(), hints: RefCell::new(HashMap::new()), leaf_idx: 0, pending_linear: None, bold: false, shear: 0.0, origin_dev: [0.0; 2], raw_origin_dev: [0.0; 2], paint_origin: [0.0; 2], replay_em: None, em_base: None, em_ops: Vec::new(), pending_em: None, pending_brush_em: None, pending_radial: None, regroup: false, turn: None, canvas: None, cell: None, next_cell: None, next_src: None, next_pd: false, vp_origins: None,
        };
        let glyph = GlyphId::new(4070);
        let first = painter.hinted_leaf(glyph).unwrap();
        painter.stack[0][0] = 24.76 / 2048.0;
        painter.stack[0][3] = -24.76 / 2048.0;
        let shared = painter.hinted_leaf(glyph).unwrap();
        painter.hints.borrow_mut().clear();
        let fresh = painter.hinted_leaf(glyph).unwrap();
        assert_ne!(first, fresh);
        assert_eq!(shared, fresh);
    }

    // `mul` must compose affines as `a ∘ b` (apply b then a) in the `[xx,yx,xy,yy,dx,dy]` convention,
    // or every paint-tree transform (and the font→device base) mis-places its glyph.
    #[test]
    fn mul_composes_apply_b_then_a() {
        let a: Mat = [2.0, 0.0, 0.0, 3.0, 10.0, 20.0]; // scale (2,3) then translate (10,20)
        let b: Mat = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // translate (1,1)
        let m = mul(&a, &b);
        // apply(m, p) must equal a(b(p)): b(0,0)=(1,1); a(1,1)=(2·1+10, 3·1+20)=(12,23).
        let p = [m[0] * 0.0 + m[2] * 0.0 + m[4], m[1] * 0.0 + m[3] * 0.0 + m[5]];
        assert_eq!(p, [12.0, 23.0]);
        // A font→device base with the y-flip: design (0, upem) is the top, must map above the baseline.
        let s = 32.0 / 2048.0;
        let m0 = mul(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[s, 0.0, 0.0, -s, 4.0, 36.5]);
        let top = [m0[0] * 0.0 + m0[2] * 2048.0 + m0[4], m0[1] * 0.0 + m0[3] * 2048.0 + m0[5]];
        assert!(top[1] < 36.5, "design top must land above the baseline, got {top:?}");
    }

    #[test]
    fn extend_mapping() {
        assert_eq!(extend_int(Extend::Pad), 0);
        assert_eq!(extend_int(Extend::Repeat), 1);
        assert_eq!(extend_int(Extend::Reflect), 2);
    }

    // The native composite-mode maps (d2d1 `DWriteCompositeModeTo{Composite,Blend}Mode`) and the
    // special CLEAR/DEST cases must stay pinned, or `PaintComposite` picks the wrong effect/formula.
    #[test]
    fn composite_mode_maps() {
        // Porter-Duff dwrite → D2D1_COMPOSITE_MODE (sample the asymmetric ones).
        assert_eq!(dwrite_to_composite(1), 10); // SRC → SOURCE_COPY
        assert_eq!(dwrite_to_composite(4), 1); // DEST_OVER
        assert_eq!(dwrite_to_composite(5), 2); // SRC_IN
        assert_eq!(dwrite_to_composite(12), 9); // PLUS
        // Blend dwrite → D2D1_BLEND_MODE (indexes effects::builtin::blend::blend).
        assert_eq!(dwrite_to_blend(13), 1); // SCREEN
        assert_eq!(dwrite_to_blend(19), 13); // HARD_LIGHT
        assert_eq!(dwrite_to_blend(23), 0); // MULTIPLY
        assert_eq!(dwrite_to_blend(27), 23); // HSL_LUMINOSITY

        let s = [100, 50, 20, 200]; // premultiplied BGRA source / backdrop
        let d = [80, 60, 30, 100];
        assert_eq!(composite_pixel(0, s, d), [0; 4]); // CLEAR draws nothing
        assert_eq!(composite_pixel(2, s, d), d); // DEST keeps the backdrop
        assert_eq!(composite_pixel(3, [10, 20, 30, 255], d), [10, 20, 30, 255]); // opaque SRC_OVER = source
        assert_eq!(composite_pixel(12, s, d), [180, 110, 50, 255]); // PLUS: summed, round-once, saturating alpha
    }

    // The two-circle solve must give t=0 on the start circle, t=1 on the end circle, and pick the
    // larger non-negative-radius root inside the pencil. Uses the colrtest `colD` geometry (contained
    // circles: |c1−c0|=200 < r1−r0=240, so every interior point has one valid root).
    #[test]
    fn conical_t_endpoints() {
        let (c0, r0, c1, r1) = ([360.0, 400.0], 80.0, [560.0, 400.0], 320.0);
        // On circle0's boundary (c0 + r0·x̂) → t = 0.
        assert_eq!(conical_t([440.0, 400.0], c0, r0, c1, r1), Some(0.0));
        // On circle1's boundary (c1 + r1·x̂) → t = 1.
        let t1 = conical_t([880.0, 400.0], c0, r0, c1, r1).unwrap();
        assert!((t1 - 1.0).abs() < 1e-5, "end-circle t = {t1}");
        // A point between the circles has t ∈ (0,1) with a non-negative interpolated radius.
        let tm = conical_t([560.0, 400.0], c0, r0, c1, r1).unwrap();
        assert!(tm > 0.0 && tm < 1.0 && r0 + (r1 - r0) * tm >= 0.0, "mid t = {tm}");
    }

    // The conical realization placement reproduces native's captured device points + footprint for
    // colorfont2-04 (c0=(360,400) r0=80 c1=(560,400) r1=320, paint transform b2d=[.05,0,0,-.05,12,52]):
    // decoded from the conical-gradient realization placement + size.
    #[test]
    fn conical_placement_cf04() {
        let b2d = [0.05f32, 0.0, 0.0, -0.05, 12.0, 52.0];
        let (p, fp) = conical_placement([360.0, 400.0], 80.0, [560.0, 400.0], 320.0, &b2d);
        assert_eq!(p[0], [30.0, 32.0]); // P0 = c0 device
        assert_eq!(p[1], [40.0, 32.0]); // P1 = c1 device
        assert_eq!(p[2], [34.0, 32.0]); // P2 = c0 + r0·x̂ device (r0 device = 4)
        assert_eq!(p[3], [56.0, 32.0]); // P3 = c1 + r1·x̂ device (r1 device = 16)
        assert_eq!(fp, 16.0); // footprint = max(r1_dev, r0_dev, |cd|_dev) = max(16, 4, 10)
    }

    // The conical colorizer frame reproduces cf04's captured cb1 bit-exact (scale 0.019960936,
    // offsets -0.2984375/-0.33933595, v2=(_,0.08,0.24,0.2)). upem=1000, leaf ink-bounds min=(17,17),
    // design footprint = max(80,320,200) = 320 → W = radial_width(320) = 512 → (W-1)/W = 511/512.
    #[test]
    fn conical_frame_cf04() {
        let b2d = [0.05f32, 0.0, 0.0, -0.05, 12.0, 52.0];
        let (scale, offset, v2) = conical_frame([360.0, 400.0], 80.0, [560.0, 400.0], 320.0, &b2d, 1000.0, [17.0, 17.0]);
        assert!((scale - 0.019960936).abs() < 1e-8, "scale {scale}");
        assert!((offset[0] - (-0.2984375)).abs() < 1e-6, "offset.x {}", offset[0]);
        assert!((offset[1] - (-0.33933595)).abs() < 1e-6, "offset.y {}", offset[1]);
        assert!((v2[1] - 0.08).abs() < 1e-6 && (v2[2] - 0.24).abs() < 1e-6 && (v2[3] - 0.2).abs() < 1e-6, "v2 {v2:?}");
    }
}
