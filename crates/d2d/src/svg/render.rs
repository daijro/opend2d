//! The SVG document render: visit the DOM and emit ordinary Direct2D calls, so every pixel comes
//! from the geometry/brush code. Native (its shape-element push etc., from the native build) builds a
//! rectangle/rounded/ellipse/path geometry and `FillGeometry`s it, then `DrawGeometry`s the stroke;
//! this reproduces that call sequence for solid paint.
//!
//! Scope that is byte-exact today: solid fills and strokes of every shape, groups, `use`, nested
//! viewports and `viewBox` transforms. Opacity layers, `clip-path`, gradient paint, `image`, and the
//! root/nested overflow clip need DeviceContext layer/clip/gradient-geometry support that other
//! phases own; the DOM for them is parsed but they are not emitted (see the final report).

// `!(x > 0.0)` is deliberate throughout: like native's `!(w > 0f)` checks it also rejects NaN,
// which `x <= 0.0` would not.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use super::attr::{PaintType, PathCmd, Units, Value};
use super::{Document, Element};
use crate::geometry::stroke::{Cap, Join};
use crate::geometry::{mat_mul, Figure, FillMode, Geometry, Matrix, Segment, StrokeStyle};
use crate::DeviceContext;

const MAX_REF_DEPTH: i32 = 16;

#[derive(Clone)]
struct Style {
    fill: Paint,
    stroke: Paint,
    fill_opacity: f32,
    stroke_opacity: f32,
    fill_rule: FillMode,
    stroke_width: f32,
    line_cap: i32,
    line_join: i32,
    miter_limit: f32,
    dashes: Vec<(f32, Units)>,
    dash_offset: f32,
    color: [f32; 4],
    visible: bool,
}

#[derive(Clone)]
struct Paint {
    ty: PaintType,
    color: [f32; 4],
    id: String,
}

impl Style {
    fn default() -> Style {
        Style {
            fill: Paint { ty: PaintType::Color, color: [0.0, 0.0, 0.0, 1.0], id: String::new() },
            stroke: Paint { ty: PaintType::None, color: [0.0; 4], id: String::new() },
            fill_opacity: 1.0,
            stroke_opacity: 1.0,
            fill_rule: FillMode::Winding,
            stroke_width: 1.0,
            line_cap: 0,
            line_join: 3,
            miter_limit: 4.0,
            dashes: Vec::new(),
            dash_offset: 0.0,
            color: [0.0, 0.0, 0.0, 1.0],
            visible: true,
        }
    }
}

impl Document {
    /// `DrawSvgDocument` → the SVG document render.
    pub fn render(&self, dc: &mut DeviceContext) {
        let Some(root) = self.root() else { return };
        let saved = dc.get_transform();
        draw_viewport(dc, self, root, &Style::default(), self.viewport[0], self.viewport[1]);
        dc.set_transform(saved);
    }
}

fn get_float(el: &Element, name: &str, default: f32) -> f32 {
    match el.attr(name) {
        Some(Value::Float(v)) => *v,
        // A length read as a float: only user-unit values resolve here; percentages need a basis the
        // render subset does not thread through, so they fall back to the default.
        Some(Value::Length(v, Units::Number)) => *v,
        _ => default,
    }
}

fn get_enum(el: &Element, name: &str, default: i32) -> i32 {
    match el.attr(name) {
        Some(Value::Enum(v)) => *v,
        _ => default,
    }
}

fn get_matrix(el: &Element, name: &str) -> Option<Matrix> {
    match el.attr(name) {
        Some(Value::Matrix(m)) => Some(*m),
        _ => None,
    }
}

/// The `xlink:href`/paint id stripped of a leading `#` (`SvgAttributeStringType.ID`).
fn href_id<'a>(el: &'a Element, name: &str) -> Option<&'a str> {
    match el.attr(name) {
        Some(Value::Str(s)) => Some(s.strip_prefix('#').unwrap_or(s)),
        _ => None,
    }
}

fn apply_style(el: &Element, style: &mut Style) {
    if let Some(Value::Paint { ty, color, id }) = el.attr("fill") {
        style.fill = Paint { ty: *ty, color: *color, id: id.clone() };
    }
    if let Some(Value::Paint { ty, color, id }) = el.attr("stroke") {
        style.stroke = Paint { ty: *ty, color: *color, id: id.clone() };
    }
    style.fill_opacity = get_float(el, "fill-opacity", style.fill_opacity);
    style.stroke_opacity = get_float(el, "stroke-opacity", style.stroke_opacity);
    style.fill_rule = if get_enum(el, "fill-rule", fill_mode_int(style.fill_rule)) == 1 {
        FillMode::Winding
    } else {
        FillMode::Alternate
    };
    style.stroke_width = get_float(el, "stroke-width", style.stroke_width);
    style.line_cap = get_enum(el, "stroke-linecap", style.line_cap);
    style.line_join = get_enum(el, "stroke-linejoin", style.line_join);
    style.miter_limit = get_float(el, "stroke-miterlimit", style.miter_limit);
    style.dash_offset = get_float(el, "stroke-dashoffset", style.dash_offset);
    if let Some(Value::Dashes(d)) = el.attr("stroke-dasharray") {
        style.dashes = d.clone();
    }
    if let Some(Value::Color(c)) = el.attr("color") {
        style.color = *c;
    }
    style.visible = get_enum(el, "visibility", if style.visible { 0 } else { 1 }) == 0;
}

fn fill_mode_int(m: FillMode) -> i32 {
    match m {
        FillMode::Winding => 1,
        FillMode::Alternate => 0,
    }
}

fn draw_element(dc: &mut DeviceContext, doc: &Document, el: &Element, parent: &Style, depth: i32) {
    if el.is_text_content() {
        return;
    }
    match el.tag() {
        "defs" | "desc" | "stop" | "linearGradient" | "radialGradient" | "title" | "clipPath" => return,
        _ => {}
    }
    if get_enum(el, "display", 0) == 1 {
        return;
    }
    let mut style = parent.clone();
    apply_style(el, &mut style);
    // ponytail: opacity<1 and clip-path need a real DeviceContext layer/mask (not yet wired); the
    // content is drawn without them rather than silently dropped. Those scenes are not claimed exact.
    let saved = dc.get_transform();
    if let Some(m) = get_matrix(el, "transform") {
        dc.set_transform(mat_mul(&m, &saved));
    }
    match el.tag() {
        "svg" => draw_nested_svg(dc, doc, el, &style, depth),
        "g" => draw_children(dc, doc, el, &style, depth),
        "use" => draw_use(dc, doc, el, &style, depth),
        "image" => {} // image decode/draw not in the byte-exact subset
        _ => draw_shape(dc, doc, el, &style),
    }
    dc.set_transform(saved);
}

fn draw_children(dc: &mut DeviceContext, doc: &Document, el: &Element, style: &Style, depth: i32) {
    for child in el.children() {
        draw_element(dc, doc, child, style, depth);
    }
}

fn draw_use(dc: &mut DeviceContext, doc: &Document, el: &Element, style: &Style, depth: i32) {
    if depth >= MAX_REF_DEPTH {
        return;
    }
    let Some(id) = href_id(el, "xlink:href") else { return };
    let Some(target) = doc.find_by_id(id) else { return };
    let target = target.clone();
    let dx = get_float(el, "x", 0.0);
    let dy = get_float(el, "y", 0.0);
    let saved = dc.get_transform();
    dc.set_transform(mat_mul(&[1.0, 0.0, 0.0, 1.0, dx, dy], &saved));
    draw_element(dc, doc, &target, style, depth + 1);
    dc.set_transform(saved);
}

fn draw_viewport(dc: &mut DeviceContext, doc: &Document, el: &Element, parent: &Style, dw: f32, dh: f32) {
    if get_enum(el, "display", 0) == 1 {
        return;
    }
    let mut style = parent.clone();
    apply_style(el, &mut style);
    let w = get_float(el, "width", dw);
    let h = get_float(el, "height", dh);
    if !(w > 0.0) || !(h > 0.0) {
        return;
    }
    let saved = dc.get_transform();
    if let Some(Value::ViewBox(vb)) = el.attr("viewBox") {
        if !(vb[2] > 0.0) || !(vb[3] > 0.0) {
            return;
        }
        let t = view_box_transform(vb, w, h, el);
        dc.set_transform(mat_mul(&t, &saved));
    }
    draw_children(dc, doc, el, &style, 0);
    dc.set_transform(saved);
}

fn draw_nested_svg(dc: &mut DeviceContext, doc: &Document, el: &Element, style: &Style, depth: i32) {
    let (dw, dh) = surrounding_basis(doc, el);
    let x = get_float(el, "x", 0.0);
    let y = get_float(el, "y", 0.0);
    let w = get_float(el, "width", dw);
    let h = get_float(el, "height", dh);
    if !(w > 0.0) || !(h > 0.0) {
        return;
    }
    let saved = dc.get_transform();
    if x != 0.0 || y != 0.0 {
        dc.set_transform(mat_mul(&[1.0, 0.0, 0.0, 1.0, x, y], &saved));
    }
    // ponytail: nested `overflow:hidden` viewport clip omitted (no rect-clip in DeviceContext yet).
    if let Some(Value::ViewBox(vb)) = el.attr("viewBox") {
        if vb[2] > 0.0 && vb[3] > 0.0 {
            let t = view_box_transform(vb, w, h, el);
            let cur = dc.get_transform();
            dc.set_transform(mat_mul(&t, &cur));
        }
    }
    draw_children(dc, doc, el, style, depth);
    dc.set_transform(saved);
}

fn surrounding_basis(doc: &Document, _el: &Element) -> (f32, f32) {
    // The viewBox search walks ancestors; the owned tree does not keep parent pointers, so fall back
    // to the document viewport (correct unless a nested <svg> inherits an ancestor <svg> viewBox).
    (doc.viewport[0], doc.viewport[1])
}

/// `ComputeViewBoxTransform` (`CalcViewportTransform`): scale the viewBox into `w`×`h` honoring
/// `preserveAspectRatio`.
fn view_box_transform(vb: &[f32; 4], w: f32, h: f32, el: &Element) -> Matrix {
    let (align, slice) = match el.attr("preserveAspectRatio") {
        Some(Value::Aspect { align, slice, .. }) => (*align, *slice),
        _ => (5, false), // xMidYMid meet
    };
    let mut sx = w / vb[2];
    let mut sy = h / vb[3];
    if align != 0 {
        let s = if slice { sx.max(sy) } else { sx.min(sy) };
        sx = s;
        sy = s;
    }
    let (ax, ay) = align_fractions(align);
    let ox = -vb[0] * sx + ax * (w - vb[2] * sx);
    let oy = -vb[1] * sy + ay * (h - vb[3] * sy);
    [sx, 0.0, 0.0, sy, ox, oy]
}

fn align_fractions(align: i32) -> (f32, f32) {
    match align {
        0 | 1 => (0.0, 0.0),
        2 => (0.5, 0.0),
        3 => (1.0, 0.0),
        4 => (0.0, 0.5),
        5 => (0.5, 0.5),
        6 => (1.0, 0.5),
        7 => (0.0, 1.0),
        8 => (0.5, 1.0),
        9 => (1.0, 1.0),
        _ => (0.5, 0.5),
    }
}

fn draw_shape(dc: &mut DeviceContext, doc: &Document, el: &Element, style: &Style) {
    if !style.visible {
        return;
    }
    let fillable = el.tag() != "line";
    let Some(geom) = build_shape(el, style.fill_rule) else { return };

    if fillable && style.fill.ty != PaintType::None {
        if let Some(brush) = solid_for(dc, doc, &style.fill, style.fill_opacity.clamp(0.0, 1.0), style) {
            dc.fill_geometry(&geom, &brush);
        }
    }
    if style.stroke.ty == PaintType::None || !(style.stroke_width > 0.0) {
        return;
    }
    if let Some(brush) = solid_for(dc, doc, &style.stroke, style.stroke_opacity.clamp(0.0, 1.0), style) {
        let ss = stroke_style(style);
        dc.draw_geometry(&geom, &brush, style.stroke_width, &ss);
    }
}

/// Resolve a paint to a solid brush. Gradient (`url(#…)` → gradient element) paint is not emitted
/// yet; a funcIRI that resolves to no gradient uses native's documented fallback color.
fn solid_for(
    dc: &DeviceContext,
    doc: &Document,
    paint: &Paint,
    opacity: f32,
    style: &Style,
) -> Option<crate::SolidColorBrush> {
    let color = match paint.ty {
        PaintType::Color => paint.color,
        PaintType::CurrentColor => style.color,
        PaintType::Uri | PaintType::UriNone | PaintType::UriColor | PaintType::UriCurrentColor => {
            if let Some(target) = doc.find_by_id(&paint.id) {
                if matches!(target.tag(), "linearGradient" | "radialGradient") {
                    return None; // gradient paint: not in the byte-exact subset
                }
            }
            match paint.ty {
                PaintType::UriColor => paint.color,
                PaintType::UriCurrentColor => style.color,
                PaintType::Uri => [0.0, 0.0, 0.0, 1.0],
                _ => return None,
            }
        }
        PaintType::None => return None,
    };
    Some(dc.create_solid_color_brush(color, Some(opacity)))
}

fn stroke_style(style: &Style) -> StrokeStyle {
    let cap = match style.line_cap {
        1 => Cap::Square,
        2 => Cap::Round,
        _ => Cap::Flat,
    };
    let join = match style.line_join {
        1 => Join::Bevel,
        2 => Join::Round,
        _ => Join::MiterOrBevel,
    };
    // Dash lengths and offset are expressed in stroke-width units, as D2D1 expects.
    let mut dashes: Vec<f32> = Vec::new();
    let total: f32 = style.dashes.iter().map(|(v, _)| v.max(0.0)).sum();
    if !style.dashes.is_empty() && total > 0.0 && style.stroke_width > 0.0 {
        dashes = style.dashes.iter().map(|(v, _)| v.max(0.0) / style.stroke_width).collect();
        // SVG: an odd number of dash values repeats the list to an even count (the stored
        // attribute keeps the original odd array — GetAttributeValue round-trips [4,2,1] — so this
        // duplication is render-only). Native's SVG stroker does this; D2D1's plain StrokeStyle does
        // not, and a lone element renders nothing.
        if dashes.len() % 2 == 1 {
            dashes.extend_from_within(..);
        }
    }
    let dash_offset = if dashes.is_empty() { 0.0 } else { style.dash_offset / style.stroke_width };
    StrokeStyle { start_cap: cap, end_cap: cap, dash_cap: cap, join, miter_limit: style.miter_limit.max(1.0), dash_offset, dashes, ..Default::default() }
}

fn build_shape(el: &Element, fill_rule: FillMode) -> Option<Geometry> {
    match el.tag() {
        "rect" => {
            let w = get_float(el, "width", 0.0);
            let h = get_float(el, "height", 0.0);
            if !(w > 0.0) || !(h > 0.0) {
                return None;
            }
            let x = get_float(el, "x", 0.0);
            let y = get_float(el, "y", 0.0);
            let rx_spec = matches!(el.attr("rx"), Some(Value::Length(v, _)) if *v >= 0.0);
            let ry_spec = matches!(el.attr("ry"), Some(Value::Length(v, _)) if *v >= 0.0);
            if rx_spec || ry_spec {
                let mut rx = get_float(el, "rx", 0.0);
                let mut ry = get_float(el, "ry", 0.0);
                if !rx_spec {
                    rx = ry;
                }
                if !ry_spec {
                    ry = rx;
                }
                rx = rx.min(w / 2.0);
                ry = ry.min(h / 2.0);
                if rx > 0.0 && ry > 0.0 {
                    return Some(Geometry::Rounded { rect: [x, y, x + w, y + h], radius: [rx, ry] });
                }
            }
            Some(Geometry::Rect([x, y, x + w, y + h]))
        }
        "circle" => {
            let r = get_float(el, "r", 0.0);
            if !(r > 0.0) {
                return None;
            }
            Some(Geometry::Ellipse { center: [get_float(el, "cx", 0.0), get_float(el, "cy", 0.0)], radius: [r, r] })
        }
        "ellipse" => {
            let rx = get_float(el, "rx", 0.0);
            let ry = get_float(el, "ry", 0.0);
            if !(rx > 0.0) || !(ry > 0.0) {
                return None;
            }
            Some(Geometry::Ellipse { center: [get_float(el, "cx", 0.0), get_float(el, "cy", 0.0)], radius: [rx, ry] })
        }
        "line" => {
            let a = [get_float(el, "x1", 0.0), get_float(el, "y1", 0.0)];
            let b = [get_float(el, "x2", 0.0), get_float(el, "y2", 0.0)];
            Some(Geometry::Path {
                fill: FillMode::Winding,
                figures: vec![Figure { start: a, filled: false, closed: false, segments: vec![Segment::Line(b)], seg_flags: Vec::new() }],
            })
        }
        "path" => match el.attr("d") {
            Some(Value::Path { cmds, data }) => Some(path_to_geometry(cmds, data, fill_rule)),
            _ => None,
        },
        "polygon" | "polyline" => {
            let Some(Value::Points(pts)) = el.attr("points") else { return None };
            if pts.len() < 2 {
                return None;
            }
            let segments = pts[1..].iter().map(|p| Segment::Line(*p)).collect();
            Some(Geometry::Path {
                fill: fill_rule,
                figures: vec![Figure { start: pts[0], filled: true, closed: el.tag() == "polygon", segments, seg_flags: Vec::new() }],
            })
        }
        _ => None,
    }
}

/// Lower an SVG `d` command/data stream to a path geometry, resolving relative/smooth/arc commands
/// to the absolute line/cubic/quadratic/arc segments a geometry sink receives.
fn path_to_geometry(cmds: &[PathCmd], data: &[f32], fill: FillMode) -> Geometry {
    let mut figures: Vec<Figure> = Vec::new();
    let mut cur: Option<Figure> = None;
    let (mut px, mut py) = (0.0f32, 0.0f32); // current point
    let (mut sx, mut sy) = (0.0f32, 0.0f32); // subpath start
    let mut prev_cubic: Option<[f32; 2]> = None; // last cubic control (for S)
    let mut prev_quad: Option<[f32; 2]> = None; // last quad control (for T)
    let mut n = 0usize;

    macro_rules! flush {
        () => {
            if let Some(f) = cur.take() {
                figures.push(f);
            }
        };
    }

    for &c in cmds {
        let k = c.arg_count();
        if n + k > data.len() {
            break;
        }
        let a = &data[n..n + k];
        n += k;
        let rel = matches!(
            c,
            PathCmd::MoveRel
                | PathCmd::LineRel
                | PathCmd::CubicRel
                | PathCmd::CubicSmoothRel
                | PathCmd::QuadRel
                | PathCmd::QuadSmoothRel
                | PathCmd::ArcRel
                | PathCmd::HorizRel
                | PathCmd::VertRel
        );
        let (ox, oy) = if rel { (px, py) } else { (0.0, 0.0) };
        let mut is_cubic = false;
        let mut is_quad = false;
        match c {
            PathCmd::MoveAbs | PathCmd::MoveRel => {
                flush!();
                px = ox + a[0];
                py = oy + a[1];
                sx = px;
                sy = py;
                cur = Some(Figure { start: [px, py], filled: true, closed: false, segments: Vec::new(), seg_flags: Vec::new() });
            }
            PathCmd::LineAbs | PathCmd::LineRel => {
                px = ox + a[0];
                py = oy + a[1];
                push_seg(&mut cur, Segment::Line([px, py]));
            }
            PathCmd::HorizAbs | PathCmd::HorizRel => {
                px = ox + a[0];
                push_seg(&mut cur, Segment::Line([px, py]));
            }
            PathCmd::VertAbs | PathCmd::VertRel => {
                py = oy + a[0];
                push_seg(&mut cur, Segment::Line([px, py]));
            }
            PathCmd::CubicAbs | PathCmd::CubicRel => {
                let c1 = [ox + a[0], oy + a[1]];
                let c2 = [ox + a[2], oy + a[3]];
                px = ox + a[4];
                py = oy + a[5];
                push_seg(&mut cur, Segment::Cubic(c1, c2, [px, py]));
                prev_cubic = Some(c2);
                is_cubic = true;
            }
            PathCmd::CubicSmoothAbs | PathCmd::CubicSmoothRel => {
                let c1 = reflect(prev_cubic, px, py);
                let c2 = [ox + a[0], oy + a[1]];
                px = ox + a[2];
                py = oy + a[3];
                push_seg(&mut cur, Segment::Cubic(c1, c2, [px, py]));
                prev_cubic = Some(c2);
                is_cubic = true;
            }
            PathCmd::QuadAbs | PathCmd::QuadRel => {
                let cc = [ox + a[0], oy + a[1]];
                px = ox + a[2];
                py = oy + a[3];
                push_seg(&mut cur, Segment::Quadratic(cc, [px, py]));
                prev_quad = Some(cc);
                is_quad = true;
            }
            PathCmd::QuadSmoothAbs | PathCmd::QuadSmoothRel => {
                let cc = reflect(prev_quad, px, py);
                px = ox + a[0];
                py = oy + a[1];
                push_seg(&mut cur, Segment::Quadratic(cc, [px, py]));
                prev_quad = Some(cc);
                is_quad = true;
            }
            PathCmd::ArcAbs | PathCmd::ArcRel => {
                px = ox + a[5];
                py = oy + a[6];
                push_seg(
                    &mut cur,
                    Segment::Arc {
                        to: [px, py],
                        radius: [a[0], a[1]],
                        rotation: a[2],
                        clockwise: a[4] == 1.0,
                        large: a[3] == 1.0,
                    },
                );
            }
            PathCmd::ClosePath => {
                if let Some(f) = &mut cur {
                    f.closed = true;
                }
                flush!();
                px = sx;
                py = sy;
            }
        }
        if !is_cubic {
            prev_cubic = None;
        }
        if !is_quad {
            prev_quad = None;
        }
    }
    flush!();
    Geometry::Path { fill, figures }
}

fn push_seg(cur: &mut Option<Figure>, seg: Segment) {
    if let Some(f) = cur {
        f.segments.push(seg);
    }
}

/// Reflect the previous control point about the current point (SVG smooth-curve rule).
fn reflect(prev: Option<[f32; 2]>, px: f32, py: f32) -> [f32; 2] {
    match prev {
        Some([cx, cy]) => [2.0 * px - cx, 2.0 * py - cy],
        None => [px, py],
    }
}
