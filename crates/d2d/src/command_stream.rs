//! `ID2D1CommandList::Stream` (ffirects agent): replay a closed command list's recorded primitives
//! to a caller-provided sink. The FFI forwards each call to the caller's `ID2D1CommandSink`.
//!
//! The command list records **device-independent** primitives + the record-time transform (see
//! [`crate::command`]). This streams each primitive resolved to device space under its recorded
//! transform, with a freshly built solid brush — so the streamed pixels equal a direct
//! [`DeviceContext::draw_image`] of the list (the record model's replay≡direct guarantee). The method
//! stream still differs from native's original-call stream (native replays the app's un-baked
//! `SetTransform`/`FillRectangle(brush)` calls); the FFI `ID2D1CommandSink` forwarder re-expresses the
//! brush/glyph/effect primitives. ponytail: a geometric-mask/opacity-brush layer streams with its
//! bounds+opacity only.

use crate::command::{CommandList, RecCmd};
use crate::resource::SolidColorBrush;
use crate::state::DrawState;

/// The device transform (`world ∘ DPI`) a recorded command resolves under — DPI scale in DIP unit
/// mode, identity in PIXELS mode.
fn device_of(state: &DrawState, dpi: (f32, f32)) -> [f32; 6] {
    let w = state.transform;
    let (sx, sy) = if state.unit == 0 { (dpi.0 / 96.0, dpi.1 / 96.0) } else { (1.0, 1.0) };
    // world then diag(sx, sy): x-column × sx, y-column × sy.
    [w[0] * sx, w[1] * sy, w[2] * sx, w[3] * sy, w[4] * sx, w[5] * sy]
}

fn ap(m: &[f32; 6], x: f32, y: f32) -> [f32; 2] {
    [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]
}

/// The sink `CommandList::stream` drives — the device-space primitive subset a command list records.
/// The FFI implements it forwarding to the caller's `ID2D1CommandSink` (identity transform, synthesised
/// brushes/geometry). `aa`/`blend` are `D2D1_ANTIALIAS_MODE`/`D2D1_PRIMITIVE_BLEND`.
pub trait StreamSink {
    fn begin(&mut self);
    /// A recorded `Clear` (straight RGBA). Default no-op (a vector-only sink ignores it); the FFI
    /// `ID2D1CommandSink` forwarder overrides it to stream the caller `ID2D1CommandSink::Clear`.
    fn clear(&mut self, _color: [f32; 4]) {}
    fn fill_rect(&mut self, rect: [f32; 4], brush: SolidColorBrush, aa: u32, blend: u32);
    fn fill_polys(&mut self, figures: &[Vec<[f32; 2]>], winding: bool, brush: SolidColorBrush, aa: u32, blend: u32);
    fn push_layer(&mut self, bounds: [f32; 4], opacity: f32, init_from_backdrop: bool);
    fn pop_layer(&mut self);
    /// A recorded axis-aligned clip, its device-space rect and antialias mode. Default no-op.
    fn push_clip(&mut self, _rect: [f32; 4], _antialias: u32) {}
    fn pop_clip(&mut self) {}
    /// A recorded `DrawImage` of a (realized) image bitmap at device top-left `p`, with its optional
    /// image-rectangle `source`, interpolation and composite mode. Default no-op: a sink that only
    /// reconstructs vector primitives ignores it; the FFI `ID2D1CommandSink` forwarder overrides it to
    /// stream the caller `ID2D1CommandSink::DrawImage`. [G12]
    fn draw_image(&mut self, _bitmap: &crate::bitmap::Bitmap, _p: [f32; 2], _source: Option<[f32; 4]>, _interpolation: u32, _composite: u32) {}
    /// A recorded gradient/bitmap/image brush fill: the device-space fill `figures` plus the brush's
    /// device-independent `def` and the device matrix `m` (to map the brush's own coordinates into
    /// device space). Default no-op — only the vector (print) sink reconstructs the brush; raster and
    /// the FFI forwarder handle brush fills on their own paths. [print]
    fn fill_gradient(&mut self, _figures: &[Vec<[f32; 2]>], _winding: bool, _def: &crate::brush::BrushDef, _m: [f32; 6]) {}
    /// A recorded stroke: the un-widened device-space path `orig`, the pre-`widened` outline, the
    /// device-space `width`, and the dash pattern. The default fills the widened outline (what a raster
    /// sink needs); the vector (print) sink keeps the original path and expresses the stroke through
    /// attributes. [print]
    fn stroke_geom(&mut self, _orig: &[Vec<[f32; 2]>], widened: &[Vec<[f32; 2]>], _width: f32, _dashes: &[f32], brush: SolidColorBrush, aa: u32, blend: u32) {
        self.fill_polys(widened, true, brush, aa, blend);
    }
    /// A recorded solid-colour glyph run: the full font bytes, the glyph ids, each glyph's XPS advance
    /// width (hundredths of the em), the device em size, the device-space baseline origin, and the
    /// straight-sRGB colour. Default no-op — only the vector (print) sink emits `<Glyphs>` + a font
    /// part. Brush-foreground and colour (COLR) runs are not routed here. [print]
    fn glyph_run(&mut self, _font: &[u8], _ids: &[u16], _advance_widths: &[f32], _em_size: f32, _origin: [f32; 2], _color: [f32; 4]) {}
    fn end(&mut self);
}

impl CommandList {
    /// Replay every recorded primitive, in order, to `sink` (between `begin`/`end`). A straight solid
    /// colour is recovered from the recorded premultiplied-RGBA source for each fill.
    pub fn stream(&self, sink: &mut dyn StreamSink) {
        sink.begin();
        let inner = self.0.borrow();
        let dom = inner.domain;
        for cmd in &inner.cmds {
            match cmd {
                RecCmd::Clear { color, .. } => sink.clear(*color),
                RecCmd::FillRect { state, dpi, rect, color, opacity } => {
                    let m = device_of(state, *dpi);
                    let b = SolidColorBrush { domain: dom, color: *color, opacity: *opacity };
                    let a = ap(&m, rect[0], rect[1]);
                    let c = ap(&m, rect[2], rect[3]);
                    if m[1] == 0.0 && m[2] == 0.0 {
                        sink.fill_rect([a[0].min(c[0]), a[1].min(c[1]), a[0].max(c[0]), a[1].max(c[1])], b, state.aa, state.blend);
                    } else {
                        let (bb, dd) = (ap(&m, rect[2], rect[1]), ap(&m, rect[0], rect[3]));
                        sink.fill_polys(&[vec![a, bb, c, dd]], true, b, state.aa, state.blend);
                    }
                }
                RecCmd::FillPolys { state, dpi, figures, winding, color, opacity } => {
                    let m = device_of(state, *dpi);
                    let b = SolidColorBrush { domain: dom, color: *color, opacity: *opacity };
                    let dev: Vec<Vec<[f32; 2]>> = figures.iter().map(|f| f.iter().map(|p| ap(&m, p[0], p[1])).collect()).collect();
                    sink.fill_polys(&dev, *winding, b, state.aa, state.blend);
                }
                RecCmd::FillGeom { state, dpi, geom, color, opacity } => {
                    let m = device_of(state, *dpi);
                    let b = SolidColorBrush { domain: dom, color: *color, opacity: *opacity };
                    let figs: Vec<Vec<[f32; 2]>> = geom.flatten_fill(&m).into_iter().filter(|p| p.filled).map(|p| p.pts).collect();
                    let winding = geom.fill_mode() == crate::geometry::FillMode::Winding;
                    sink.fill_polys(&figs, winding, b, state.aa, state.blend);
                }
                RecCmd::StrokeGeom { state, dpi, geom, color, opacity, width, style } => {
                    let m = device_of(state, *dpi);
                    let b = SolidColorBrush { domain: dom, color: *color, opacity: *opacity };
                    let ds = [dpi.0 / 96.0, dpi.1 / 96.0];
                    let widened = geom.widen_scaled(*width, style, &m, ds, crate::geometry::DEFAULT_TOLERANCE, 16.0, None, None);
                    // The un-widened device-space path + device thickness + dashes, so a vector sink can
                    // emit a stroked `<Path>` (the raster default just fills `widened`).
                    let orig: Vec<Vec<[f32; 2]>> = geom.flatten(&m, crate::geometry::DEFAULT_TOLERANCE).into_iter().map(|p| p.pts).collect();
                    sink.stroke_geom(&orig, &widened, *width * ds[0], &style.dashes, b, state.aa, state.blend);
                }
                RecCmd::Line { state, dpi, p0, p1, width, style, color, opacity } => {
                    // Streamed as the line's stroke.
                    let geom = crate::context::line_geometry(*p0, *p1);
                    let style = style.clone().unwrap_or_default();
                    let m = device_of(state, *dpi);
                    let b = SolidColorBrush { domain: dom, color: *color, opacity: *opacity };
                    let ds = [dpi.0 / 96.0, dpi.1 / 96.0];
                    let widened = geom.widen_scaled(*width, &style, &m, ds, crate::geometry::DEFAULT_TOLERANCE, 16.0, None, None);
                    let orig: Vec<Vec<[f32; 2]>> = geom.flatten(&m, crate::geometry::DEFAULT_TOLERANCE).into_iter().map(|p| p.pts).collect();
                    sink.stroke_geom(&orig, &widened, *width * ds[0], &style.dashes, b, state.aa, state.blend);
                }
                RecCmd::PushLayer { state, dpi, bounds, opacity, layer_options, .. } => {
                    let m = device_of(state, *dpi);
                    let a = ap(&m, bounds[0], bounds[1]);
                    let c = ap(&m, bounds[2], bounds[3]);
                    sink.push_layer([a[0].min(c[0]), a[1].min(c[1]), a[0].max(c[0]), a[1].max(c[1])], *opacity, layer_options & 1 != 0);
                }
                RecCmd::PopLayer => sink.pop_layer(),
                RecCmd::PushClip { state, dpi, rect, antialias } => {
                    let m = device_of(state, *dpi);
                    let a = ap(&m, rect[0], rect[1]);
                    let c = ap(&m, rect[2], rect[3]);
                    sink.push_clip([a[0].min(c[0]), a[1].min(c[1]), a[0].max(c[0]), a[1].max(c[1])], *antialias);
                }
                RecCmd::PopClip => sink.pop_clip(),
                RecCmd::DrawImage { state, dpi, bitmap, offset, source, interpolation, composite } => {
                    let m = device_of(state, *dpi);
                    sink.draw_image(bitmap, ap(&m, offset[0], offset[1]), *source, *interpolation, *composite);
                }
                // Gradient/bitmap/image brush fills carry a re-realizable `def`; the vector (print) sink
                // reconstructs the brush from it. A brush with no def (image/COLR ramp) is skipped.
                RecCmd::FillRectBrush { state, dpi, rect, brush } => {
                    if let Some(def) = &brush.def {
                        let m = device_of(state, *dpi);
                        let (a, b, c, d) = (ap(&m, rect[0], rect[1]), ap(&m, rect[2], rect[1]), ap(&m, rect[2], rect[3]), ap(&m, rect[0], rect[3]));
                        sink.fill_gradient(&[vec![a, b, c, d]], true, def, m);
                    }
                }
                RecCmd::FillGeomBrush { state, dpi, geom, brush } => {
                    if let Some(def) = &brush.def {
                        let m = device_of(state, *dpi);
                        let figs: Vec<Vec<[f32; 2]>> = geom.flatten_fill(&m).into_iter().filter(|p| p.filled).map(|p| p.pts).collect();
                        let winding = geom.fill_mode() == crate::geometry::FillMode::Winding;
                        sink.fill_gradient(&figs, winding, def, m);
                    }
                }
                // A solid-colour, non-COLR glyph run → the vector (print) sink emits `<Glyphs>` + a font
                // part. Brush foreground / colour fonts are a follow-up (left to the FFI forwarder).
                RecCmd::GlyphRun { state, dpi, font, em_size, glyphs, baseline_origin, fg, palette, .. } => {
                    if let (None, crate::command::TextFg::Solid(c)) = (palette, fg) {
                        let m = device_of(state, *dpi);
                        let scale = (m[0] * m[0] + m[1] * m[1]).sqrt();
                        let origin = ap(&m, baseline_origin[0], baseline_origin[1]);
                        let ids: Vec<u16> = glyphs.iter().map(|g| g.id).collect();
                        let advw: Vec<f32> = glyphs.iter().map(|g| g.advance / em_size * 100.0).collect();
                        sink.glyph_run(font, &ids, &advw, em_size * scale, origin, *c);
                    }
                }
                // The FFI `ID2D1CommandSink` forwarder re-expresses these (brush strokes, nested
                // lists, effect draws); not re-expressed as a solid-colour sink primitive.
                RecCmd::StrokeGeomBrush { .. }
                | RecCmd::LineBrush { .. }
                | RecCmd::DrawImageList { .. }
                | RecCmd::DrawEffect { .. }
                | RecCmd::BlitFloat { .. }
                | RecCmd::SpriteBatch { .. }
                | RecCmd::Mesh { .. }
                | RecCmd::MeshBrush { .. }
                | RecCmd::DrawBitmap { .. }
                | RecCmd::Ink { .. }
                | RecCmd::InkBrush { .. }
                | RecCmd::GradientMesh { .. }
                | RecCmd::OpacityMask { .. }
                | RecCmd::OpacityMaskBrush { .. }
                | RecCmd::Flood { .. }
                | RecCmd::Effect { .. } => {}
            }
        }
        sink.end();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::Domain;

    #[derive(Default)]
    struct Rec {
        ops: Vec<String>,
    }
    impl StreamSink for Rec {
        fn begin(&mut self) {
            self.ops.push("begin".into());
        }
        fn fill_rect(&mut self, rect: [f32; 4], _b: SolidColorBrush, _aa: u32, _blend: u32) {
            self.ops.push(format!("rect {:?}", rect));
        }
        fn fill_polys(&mut self, figs: &[Vec<[f32; 2]>], _w: bool, _b: SolidColorBrush, _aa: u32, _blend: u32) {
            self.ops.push(format!("polys {}", figs.len()));
        }
        fn push_layer(&mut self, _b: [f32; 4], _o: f32, _i: bool) {
            self.ops.push("push".into());
        }
        fn pop_layer(&mut self) {
            self.ops.push("pop".into());
        }
        fn end(&mut self) {
            self.ops.push("end".into());
        }
    }

    #[test]
    fn streams_recorded_primitives_in_order() {
        let list = CommandList::new(Domain { factory_id: 0, device_id: 0 });
        list.0.borrow_mut().cmds.push(RecCmd::FillRect {
            state: DrawState::default(),
            dpi: (96.0, 96.0),
            rect: [1.0, 2.0, 3.0, 4.0],
            color: [0.1, 0.2, 0.3, 1.0],
            opacity: 1.0,
        });
        list.close();
        let mut rec = Rec::default();
        list.stream(&mut rec);
        assert_eq!(rec.ops, vec!["begin", "rect [1.0, 2.0, 3.0, 4.0]", "end"]);
    }
}
