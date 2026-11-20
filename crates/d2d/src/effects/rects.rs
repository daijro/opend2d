//! Effect-graph rectangle mapping — the backward (`MapOutputRectToInputRects`) and invalidation
//! (`MapInvalidRect`) passes behind `ID2D1DeviceContext::GetEffectRequiredInputRectangles`,
//! `GetEffectInvalidRectangle{Count,s}` and the effect side of `GetImage{Local,World}Bounds`.
//!
//! The forward pass (`MapInputRectsToOutputRect`) is already [`super::Effect::bounds`]. This adds its
//! two inverses, reusing that machinery:
//!
//! * **Required input**: native converts the output rect DIPs→device px, propagates it backward through
//!   the transform graph (each node's `MapOutputRectToInputRects`) and reads the resolved rect at each
//!   requested input, device px→DIPs. For a pointwise node the input rect is the output rect; for a
//!   crop it is clipped to the crop rect; for a reach filter (blur/convolve/…) it is grown by the
//!   node's own forward halo (`bounds − inputBounds`), which keeps blur/offset/border/etc. exact
//!   without re-deriving each kernel radius.
//! * **Invalid output** (`MapInvalidRect`): an invalidated input sub-rect mapped *forward* to the
//!   output region it dirties — the same halo, the other direction.
//!
//! Rects are device pixels; at 96 DPI that is DIPs, so a scene passing integer DIP rects round-trips
//! exactly (native's `ConvertRectFromDips`/`ToDips` scale by dpi/96 and round). ponytail: a single
//! coalesced invalid rect per effect and the queried effect's own invalidations only — a separable
//! blur's multiple output rects and transitive child-effect invalidation are not reproduced (no scene
//! reaches them); integer-DIP round-trip is the tested path.

use super::{input_bounds, Effect, Input, Kind, Prop, Rect, INF};
use crate::context::DeviceContext;
use std::rc::Rc;

impl Effect {
    /// `ID2D1Properties::GetValue`: the current value stored for property `index` (ffirects agent), or
    /// `None` if the effect has none (unset built-in slots have no core default table).
    pub fn stored_value(&self, index: u32) -> Option<Prop> {
        self.0.borrow().props.get(&index).cloned()
    }
}

/// DIP rect `[l,t,r,b]` → device-pixel [`Rect`] (native `ConvertRectFromDips`: `round(coord·dpi/96)`).
fn from_dips(r: [f32; 4], dpi: (f32, f32)) -> Rect {
    let q = |v: f32, d: f32| (v * d / 96.0).round_ties_even() as i32;
    Rect { l: q(r[0], dpi.0), t: q(r[1], dpi.1), r: q(r[2], dpi.0), b: q(r[3], dpi.1) }
}

/// Device-pixel [`Rect`] → DIP rect (native `ConvertRectToDips`: `coord·96/dpi`).
fn to_dips(r: Rect, dpi: (f32, f32)) -> [f32; 4] {
    let s = |v: i32, d: f32| v as f32 * 96.0 / d;
    [s(r.l, dpi.0), s(r.t, dpi.1), s(r.r, dpi.0), s(r.b, dpi.1)]
}

impl Effect {
    /// Same `Rc` as `other` (effect identity, to match a `D2D1_EFFECT_INPUT_DESCRIPTION`'s effect).
    fn is(&self, other: &Effect) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }

    /// This node's intrinsic output bounds as DIPs, or `None` for an infinite output (Flood/Border/
    /// Tile/generators) — `GetImageLocalBounds` returns the empty rect for an infinite image.
    pub fn intrinsic_bounds(&self) -> Option<[f32; 4]> {
        let b = self.bounds();
        if b.l == INF.l || b.r == INF.r || b.t == INF.t || b.b == INF.b {
            None
        } else {
            Some([b.l as f32, b.t as f32, b.r as f32, b.b as f32])
        }
    }

    /// Per-side forward halo of a reach filter: how far its output extends past input 0 on each side
    /// (`bounds − inputBounds`, ≥0). The backward map grows the output rect by this, the invalid map too.
    fn halo(&self) -> (i32, i32, i32, i32) {
        let ib = input_bounds(&self.0.borrow().inputs[0]);
        let ob = self.bounds();
        ((ib.l - ob.l).max(0), (ib.t - ob.t).max(0), (ob.r - ib.r).max(0), (ob.b - ib.b).max(0))
    }

    /// Inverse affine of `out` (map output corners through the inverse matrix, bbox, grow the 0.5-texel
    /// linear reach) — the backward of [`Effect::affine_output_bounds`].
    fn affine_input_bounds(&self, out: Rect) -> Rect {
        let m = self.affine_matrix();
        let det = m[0] * m[3] - m[1] * m[2];
        if det == 0.0 {
            return out;
        }
        let inv = |x: f32, y: f32| {
            [(m[3] * (x - m[4]) - m[2] * (y - m[5])) / det, (-m[1] * (x - m[4]) + m[0] * (y - m[5])) / det]
        };
        let k = if self.affine_sampling().0 { 0.5 } else { 0.0 };
        let c = [inv(out.l as f32, out.t as f32), inv(out.r as f32, out.t as f32), inv(out.l as f32, out.b as f32), inv(out.r as f32, out.b as f32)];
        let minx = c.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let maxx = c.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
        let miny = c.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        let maxy = c.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);
        Rect { l: (minx - k).floor() as i32, t: (miny - k).floor() as i32, r: (maxx + k).ceil() as i32, b: (maxy + k).ceil() as i32 }
    }

    /// `MapOutputRectToInputRects` for input `slot`: the rect of that input required to render `out`.
    fn map_output_to_input(&self, slot: usize, out: Rect) -> Rect {
        use Kind::*;
        let kind = self.0.borrow().kind;
        let ib = input_bounds(&self.0.borrow().inputs[slot]);
        match kind {
            // Pointwise / per-input 1:1 (colour, blend/composite, opacity, lighting, LUT): the input
            // rect is the output rect, clipped to what the input actually holds.
            Color(_) | ColorManagement | OpacityMetadata | YCbCr | HighlightsShadows | Opacity | ColorMatrix | LookupTable3D | Composite | Blend | Arithmetic | CrossFade
            | PointDiffuse | DistantDiffuse | SpotDiffuse | PointSpecular | DistantSpecular | SpotSpecular => out.intersect(ib),
            Crop => out.intersect(self.crop_rect()).intersect(ib),
            Border => ib,                       // samples the whole (finite) source to extend it
            Tile => self.crop_rect().intersect(ib), // reads the tile window, independent of output
            Affine2D | Straighten => self.affine_input_bounds(out).intersect(ib),
            GaussianBlur | DirectionalBlur | Shadow | ConvolveMatrix | Morphology | Sharpen | EdgeDetection | Emboss | DisplacementMap | Posterize => {
                let (hl, ht, hr, hb) = self.halo();
                Rect { l: out.l - hl, t: out.t - ht, r: out.r + hr, b: out.b + hb }.intersect(ib)
            }
            // Generators / resamplers with no simple analytic inverse: pass the output rect through.
            // Histogram reads its whole input but its image output is empty, so this is unreached.
            Custom | Atlas | Scale | Dpi | Transform3D | Perspective3D | Turbulence | Flood | Histogram | BitmapSource => out,
        }
    }

    /// Propagate the root output rect backward through the graph, recording the resolved rect for each
    /// requested `(effect, inputIndex)` (matched by effect identity + slot).
    fn propagate_required(&self, out: Rect, reqs: &[(Effect, u32)], results: &mut [Option<Rect>]) {
        let inputs = self.0.borrow().inputs.clone();
        for (slot, inp) in inputs.iter().enumerate() {
            let req = self.map_output_to_input(slot, out);
            for (i, (e, idx)) in reqs.iter().enumerate() {
                if *idx as usize == slot && self.is(e) {
                    results[i] = Some(req);
                }
            }
            if let Input::Effect(child) = inp {
                child.propagate_required(req, reqs, results);
            }
        }
    }

    /// `GetEffectRequiredInputRectangles`: for each requested input, the rect of it needed to render
    /// `out_rect` (DIPs). Order matches `reqs`. A request naming an unreached input returns the empty
    /// rect (native leaves it zero).
    pub fn required_input_rectangles(&self, out_rect: [f32; 4], reqs: &[(Effect, u32)], dpi: (f32, f32)) -> Vec<[f32; 4]> {
        let mut results = vec![None; reqs.len()];
        self.propagate_required(from_dips(out_rect, dpi), reqs, &mut results);
        results.into_iter().map(|r| to_dips(r.unwrap_or(Rect { l: 0, t: 0, r: 0, b: 0 }), dpi)).collect()
    }

}

impl DeviceContext {
    /// `GetImageLocalBounds`: the image's bounds in DIPs under an identity world transform (DPI only).
    /// `bounds_dip` is the image's intrinsic DIP bounds (bitmap `[0,0,w,h]` scaled to DIPs, or an
    /// effect's `intrinsic_bounds`). Device px floor/ceil then back to DIPs, matching native.
    pub fn image_local_bounds(&self, bounds_dip: [f32; 4]) -> [f32; 4] {
        let dpi = self.get_dpi();
        let (sx, sy) = if self.get_unit_mode() == 0 { (dpi.0 / 96.0, dpi.1 / 96.0) } else { (1.0, 1.0) };
        map_bounds(bounds_dip, &[sx, 0.0, 0.0, sy, 0.0, 0.0], (sx, sy))
    }

    /// `GetImageWorldBounds`: as above but through the context's world transform (world ∘ DPI).
    pub fn image_world_bounds(&self, bounds_dip: [f32; 4]) -> [f32; 4] {
        let dt = self.device_transform();
        let m = [dt[0] as f32, dt[1] as f32, dt[2] as f32, dt[3] as f32, dt[4] as f32, dt[5] as f32];
        let dpi = self.get_dpi();
        let (sx, sy) = if self.get_unit_mode() == 0 { (dpi.0 / 96.0, dpi.1 / 96.0) } else { (1.0, 1.0) };
        map_bounds(bounds_dip, &m, (sx, sy))
    }
}

/// Map a DIP rect through device matrix `m` to a device-pixel bbox (floor/ceil), then divide the
/// device scale `(sx,sy)` back out to DIPs.
fn map_bounds(r: [f32; 4], m: &[f32; 6], scale: (f32, f32)) -> [f32; 4] {
    let p = |x: f32, y: f32| [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]];
    let c = [p(r[0], r[1]), p(r[2], r[1]), p(r[0], r[3]), p(r[2], r[3])];
    let minx = c.iter().map(|q| q[0]).fold(f32::INFINITY, f32::min);
    let maxx = c.iter().map(|q| q[0]).fold(f32::NEG_INFINITY, f32::max);
    let miny = c.iter().map(|q| q[1]).fold(f32::INFINITY, f32::min);
    let maxy = c.iter().map(|q| q[1]).fold(f32::NEG_INFINITY, f32::max);
    let (l, t, rr, b) = (minx.floor(), miny.floor(), maxx.ceil(), maxy.ceil());
    [l / scale.0, t / scale.1, rr / scale.0, b / scale.1]
}
