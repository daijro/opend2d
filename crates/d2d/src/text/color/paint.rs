//! Static COLR paint decoding. Keep the Pad stop domain before brush realization:
//! shortening it to the first/last stop changes the finite gradient ramp.
use super::*;
use skrifa::raw::tables::colr::{Colr, ColorLine, Paint};
use skrifa::raw::types::Point;

type Pt = Point<f32>;

enum Ink {
    Solid(u16, f32),
    // Raw COLR points p0, p1, p2 (design units) + stop domain lo/hi. The p2 rotation projection is
    // deferred to `Painter::brush` (see `Painter::pending_linear`), which follows d2d1's order.
    Linear(Pt, Pt, Pt, f32, f32, Vec<ColorStop>, Extend),
    // The circles over the stop domain, then the raw COLR c0, c1, r1 and the domain's hi (`Painter::pending_radial`).
    Radial(Pt, f32, Pt, f32, Vec<ColorStop>, Extend, ([f32; 2], [f32; 2], f32, f32)),
    Sweep(Pt, f32, f32, Vec<ColorStop>, Extend),
}
impl Ink {
    fn brush(&self) -> CBrush<'_> {
        match self {
            Self::Solid(palette_index, alpha) => CBrush::Solid { palette_index: *palette_index, alpha: *alpha },
            // p0/p1 here are the raw endpoints; the device-space projection in `brush()` uses
            // `pending_linear` (set by `Node::draw`), so these only serve the skrifa fallback.
            Self::Linear(p0, p1, _p2, _lo, _hi, color_stops, extend) => CBrush::LinearGradient { p0: *p0, p1: *p1, color_stops, extend: *extend },
            Self::Radial(c0, r0, c1, r1, color_stops, extend, _) => CBrush::RadialGradient { c0: *c0, r0: *r0, c1: *c1, r1: *r1, color_stops, extend: *extend },
            Self::Sweep(c0, start_angle, end_angle, color_stops, extend) => CBrush::SweepGradient { c0: *c0, start_angle: *start_angle, end_angle: *end_angle, color_stops, extend: *extend },
        }
    }
}

enum Node {
    Empty,
    Ink(Ink),
    Group(Vec<Node>),
    // The design-unit matrix and the em-unit one DirectWrite builds for d2d1 (`Painter::em_ops`).
    Transform(Mat, Mat, Box<Node>),
    Glyph(GlyphId, Box<Node>),
    Clip(BoundingBox<f32>, Box<Node>),
    Composite(CompositeMode, Box<Node>, Box<Node>),
}
impl Node {
    /// The ink under a run of transforms, with their product in design units and in em units. d2d1 folds each
    /// transform ahead of the brush transform below it (`T × B`), so the em product runs outermost first.
    fn leaf(&self) -> Option<(&Ink, Option<(Mat, Mat)>)> {
        match self {
            Self::Ink(ink) => Some((ink, None)),
            Self::Transform(m, em, child) => {
                let (ink, next) = child.leaf()?;
                Some((ink, Some(next.map_or((*m, *em), |(n, ne)| (mul(m, &n), mul(&ne, em))))))
            }
            _ => None,
        }
    }
    fn draw(&self, painter: &mut super::Painter) {
        let transform = |m: &Mat| Transform { xx: m[0], yx: m[1], xy: m[2], yy: m[3], dx: m[4], dy: m[5] };
        // The raw COLR linear points for the device-space p2 projection in `Painter::brush`, or `None`
        // for non-linear inks so a stale value never leaks to the next leaf.
        let pend = |ink: &Ink| match ink {
            Ink::Linear(p0, p1, p2, lo, hi, _, _) => Some(([p0.x, p0.y], [p1.x, p1.y], [p2.x, p2.y], *lo, *hi)),
            _ => None,
        };
        let pend_radial = |ink: &Ink| match ink { Ink::Radial(.., raw) => Some(*raw), _ => None };
        match self {
            Self::Empty => {}
            Self::Ink(ink) => { painter.pending_linear = pend(ink); painter.pending_radial = pend_radial(ink); painter.fill(ink.brush()); }
            Self::Group(nodes) => { for node in nodes { node.draw(painter); } }
            Self::Transform(m, em, child) => {
                painter.pending_em = Some(*em);
                painter.push_transform(transform(m)); child.draw(painter); painter.pop_transform();
            }
            Self::Glyph(gid, child) => {
                if let Some((ink, m)) = child.leaf() {
                    painter.pending_linear = pend(ink);
                    painter.pending_radial = pend_radial(ink);
                    painter.pending_brush_em = m.map(|(_, em)| em);
                    painter.fill_glyph(*gid, m.as_ref().map(|(m, _)| transform(m)), ink.brush());
                } else if !matches!(**child, Self::Empty) {
                    painter.push_clip_glyph(*gid); child.draw(painter); painter.pop_clip();
                }
            }
            Self::Clip(rect, child) => {
                painter.push_clip_box(*rect); child.draw(painter); painter.pop_clip();
            }
            Self::Composite(mode, backdrop, source) => {
                // A blend mode merges the groups through the Blend effect, whose inputs render into atlas cells over the
                // composite's bounds; a Porter-Duff mode's Composite effect draws the backdrop into the output cell and
                // blends on the source from a texture of its own ([`super::Painter::push_layer`]).
                let pd = (*mode as u8) < CompositeMode::Screen as u8;
                let frame = painter.group_frame();
                painter.next_cell = self.bounds(&frame, painter.font);
                painter.next_src = pd.then(|| source.bounds(&frame, painter.font)).flatten();
                painter.next_pd = pd;
                painter.push_layer(CompositeMode::SrcOver); backdrop.draw(painter);
                painter.push_layer(*mode); source.draw(painter);
                painter.pop_layer_with_mode(*mode); painter.pop_layer_with_mode(CompositeMode::SrcOver);
            }
        }
    }
}

impl Node {
    /// The device bounds of the glyphs this node paints under `m`, their outline boxes: what d2d1 bounds a
    /// composite's command list by.
    fn bounds(&self, m: &Mat, font: &FontRef<'_>) -> Option<[f32; 4]> {
        let union = |a: Option<[f32; 4]>, b: Option<[f32; 4]>| match (a, b) {
            (Some(a), Some(b)) => Some([a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]),
            (a, b) => a.or(b),
        };
        match self {
            Self::Empty | Self::Ink(_) => None,
            Self::Group(nodes) => nodes.iter().fold(None, |acc, n| union(acc, n.bounds(m, font))),
            Self::Transform(t, _, child) => child.bounds(&mul(m, t), font),
            Self::Clip(_, child) => child.bounds(m, font),
            Self::Glyph(gid, _) => {
                let b = font.glyph_metrics(Size::unscaled(), LocationRef::default()).bounds(*gid)?;
                Some(map_bounds(m, [b.x_min, b.y_min, b.x_max, b.y_max]))
            }
            Self::Composite(_, backdrop, source) => union(backdrop.bounds(m, font), source.bounds(m, font)),
        }
    }
}

fn line(line: ColorLine<'_>) -> Option<(Vec<ColorStop>, Extend, f32, f32)> {
    let extend = line.extend();
    let mut stops: Vec<_> = line.color_stops().iter().map(|s| ColorStop {
        offset: s.stop_offset().to_f32(), palette_index: s.palette_index(), alpha: s.alpha().to_f32(),
    }).collect();
    stops.sort_by(|a, b| a.offset.total_cmp(&b.offset));
    let (mut lo, mut hi) = (stops.first()?.offset, stops.last()?.offset);
    if extend == Extend::Pad { lo = lo.min(0.0); hi = hi.max(1.0); }
    if hi == lo { return None; }
    for stop in &mut stops { stop.offset = (stop.offset - lo) / (hi - lo); }
    Some((stops, extend, lo, hi))
}
fn lerp(a: Pt, b: Pt, t: f32) -> Pt { Pt::new(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t) }
fn scale(x: f32, y: f32, cx: f32, cy: f32) -> Mat { [x, 0.0, 0.0, y, cx - x * cx, cy - y * cy] }
fn rotate(angle: f32, cx: f32, cy: f32) -> Mat {
    let a = angle * std::f32::consts::PI;
    let (s, c) = (crate::crt::sinf(a), crate::crt::cosf(a));
    [c, s, -s, c, s * cy + (1.0 - c) * cx, -s * cx + (1.0 - c) * cy]
}
// The em-unit (y down) matrices DirectWrite's COLR reader builds for d2d1, `k` = 1/upem: centres go to em units
// first, (cx·k, cy·−k); a scale about a centre keeps it with (1 − s)·c, a rotation or skew with c − M·c.
fn scale_em(x: f32, y: f32, cx: f32, cy: f32, k: f32) -> Mat {
    [x, 0.0, 0.0, y, (1.0 - x) * (cx * k), (1.0 - y) * (cy * -k)]
}
fn about_em(m: [f32; 4], cx: f32, cy: f32, k: f32) -> Mat {
    let (cx, cy) = (cx * k, cy * -k);
    [m[0], m[1], m[2], m[3], cx - (cy * m[2] + cx * m[0]), cy - (cy * m[3] + cx * m[1])]
}
fn rotate_em(angle: f32, cx: f32, cy: f32, k: f32) -> Mat {
    let a = angle * std::f32::consts::PI;
    let (s, c) = (crate::crt::sinf(a), crate::crt::cosf(a));
    about_em([c, -s, s, c], cx, cy, k)
}
fn skew_em(x: f32, y: f32, cx: f32, cy: f32, k: f32) -> Mat {
    let tan = crate::crt::tanf;
    about_em([1.0, tan(-(y * std::f32::consts::PI)), tan(x * std::f32::consts::PI), 1.0], cx, cy, k)
}
fn skew(x: f32, y: f32, cx: f32, cy: f32) -> Mat {
    // DirectWrite's paint reader takes each shear with tanf of the angle (half-turns times π), not sin/cos.
    let tan = |v: f32| crate::crt::tanf(v * std::f32::consts::PI);
    let (x, y) = (tan(x), tan(y));
    [1.0, y, -x, 1.0, x * cy, -y * cx]
}

struct Reader<'a> { colr: Colr<'a>, remaining: u32, k: f32 }
impl<'a> Reader<'a> {
    fn glyph(&mut self, gid: GlyphId, depth: u32) -> Option<Node> {
        let (paint, _) = self.colr.v1_base_glyph(gid)?;
        let node = self.node(paint, depth)?;
        Some(if let Some(rect) = self.colr.v1_clip_box(gid) {
            if rect.format() != 1 { return None; }
            Node::Clip(BoundingBox {
                x_min: rect.x_min().to_i16() as f32, y_min: rect.y_min().to_i16() as f32,
                x_max: rect.x_max().to_i16() as f32, y_max: rect.y_max().to_i16() as f32,
            }, Box::new(node))
        } else { node })
    }
    fn node(&mut self, paint: Paint<'a>, depth: u32) -> Option<Node> {
        if depth >= 64 { return None; }
        self.remaining = self.remaining.checked_sub(1)?;
        let k = self.k;
        macro_rules! xf {
            ($p:expr, $m:expr, $em:expr) => { Node::Transform($m, $em, Box::new(self.node($p.paint().ok()?, depth + 1)?)) };
        }
        Some(match paint {
            Paint::Solid(p) => Node::Ink(Ink::Solid(p.palette_index(), p.alpha().to_f32())),
            Paint::LinearGradient(p) => {
                let Some((stops, extend, lo, hi)) = line(p.color_line().ok()?) else { return Some(Node::Empty) };
                let a = Pt::new(p.x0().to_i16() as f32, p.y0().to_i16() as f32);
                let b = Pt::new(p.x1().to_i16() as f32, p.y1().to_i16() as f32);
                let v = Pt::new(p.x2().to_i16() as f32 - a.x, p.y2().to_i16() as f32 - a.y);
                let len = v.x * v.x + v.y * v.y;
                if len == 0.0 || (b.x - a.x) * v.y == (b.y - a.y) * v.x { return None; }
                // Degenerate check only (collinearity is affine-invariant). The foot projection and
                // lo/hi lerp are deferred to `Painter::brush`, whose realization re-projects in device
                // space (the COLRv1 p2 rotation is not preserved by a non-uniform leaf transform).
                Node::Ink(Ink::Linear(a, b, Pt::new(p.x2().to_i16() as f32, p.y2().to_i16() as f32), lo, hi, stops, extend))
            }
            Paint::RadialGradient(p) => {
                let Some((stops, extend, lo, hi)) = line(p.color_line().ok()?) else { return Some(Node::Empty) };
                let a = Pt::new(p.x0().to_i16() as f32, p.y0().to_i16() as f32);
                let b = Pt::new(p.x1().to_i16() as f32, p.y1().to_i16() as f32);
                let r = p.radius0().to_u16() as f32;
                let dr = p.radius1().to_u16() as f32 - r;
                let raw = ([a.x, a.y], [b.x, b.y], p.radius1().to_u16() as f32, hi);
                Node::Ink(Ink::Radial(lerp(a, b, lo), r + dr * lo, lerp(a, b, hi), r + dr * hi, stops, extend, raw))
            }
            Paint::SweepGradient(p) => {
                let Some((stops, extend, lo, hi)) = line(p.color_line().ok()?) else { return Some(Node::Empty) };
                let a = p.start_angle().to_f32() * 180.0 + 180.0;
                let span = (p.end_angle().to_f32() - p.start_angle().to_f32()) * 180.0;
                // Native GetSweepGradientBrush (d2d1 0x2cb084) preserves stop order. Reversing
                // the ramp changes the side carrying its clamp-border texel.
                let (start, end) = (a + span * lo, a + span * hi);
                Node::Ink(Ink::Sweep(Pt::new(p.center_x().to_i16() as f32, p.center_y().to_i16() as f32), start, end, stops, extend))
            }
            Paint::ColrLayers(p) => {
                let mut nodes = Vec::new();
                for i in p.first_layer_index()..p.first_layer_index().checked_add(p.num_layers() as u32)? {
                    nodes.push(self.node(self.colr.v1_layer(i as usize)?.0, depth + 1)?);
                }
                Node::Group(nodes)
            }
            Paint::Glyph(p) => Node::Glyph(p.glyph_id().into(), Box::new(self.node(p.paint().ok()?, depth + 1)?)),
            Paint::ColrGlyph(p) => self.glyph(p.glyph_id().into(), depth + 1)?,
            Paint::Composite(p) => Node::Composite(p.composite_mode(),
                Box::new(self.node(p.backdrop_paint().ok()?, depth + 1)?), Box::new(self.node(p.source_paint().ok()?, depth + 1)?)),
            Paint::Transform(p) => {
                let m = p.transform().ok()?;
                let m = [m.xx().to_f32(), m.yx().to_f32(), m.xy().to_f32(), m.yy().to_f32(), m.dx().to_f32(), m.dy().to_f32()];
                xf!(p, m, super::em_of(&m, k))
            }
            Paint::Translate(p) => {
                let (dx, dy) = (p.dx().to_i16() as f32, p.dy().to_i16() as f32);
                xf!(p, [1.0, 0.0, 0.0, 1.0, dx, dy], [1.0, 0.0, 0.0, 1.0, dx * k, dy * -k])
            }
            Paint::Scale(p) => {
                let (x, y) = (p.scale_x().to_f32(), p.scale_y().to_f32());
                xf!(p, scale(x, y, 0.0, 0.0), [x, 0.0, 0.0, y, 0.0, 0.0])
            }
            Paint::ScaleAroundCenter(p) => {
                let (x, y, cx, cy) = (p.scale_x().to_f32(), p.scale_y().to_f32(), p.center_x().to_i16() as f32, p.center_y().to_i16() as f32);
                xf!(p, scale(x, y, cx, cy), scale_em(x, y, cx, cy, k))
            }
            Paint::ScaleUniform(p) => {
                let s = p.scale().to_f32();
                xf!(p, scale(s, s, 0.0, 0.0), [s, 0.0, 0.0, s, 0.0, 0.0])
            }
            Paint::ScaleUniformAroundCenter(p) => {
                let (s, cx, cy) = (p.scale().to_f32(), p.center_x().to_i16() as f32, p.center_y().to_i16() as f32);
                xf!(p, scale(s, s, cx, cy), scale_em(s, s, cx, cy, k))
            }
            Paint::Rotate(p) => xf!(p, rotate(p.angle().to_f32(), 0.0, 0.0), rotate_em(p.angle().to_f32(), 0.0, 0.0, k)),
            Paint::RotateAroundCenter(p) => {
                let (a, cx, cy) = (p.angle().to_f32(), p.center_x().to_i16() as f32, p.center_y().to_i16() as f32);
                xf!(p, rotate(a, cx, cy), rotate_em(a, cx, cy, k))
            }
            Paint::Skew(p) => {
                let (x, y) = (p.x_skew_angle().to_f32(), p.y_skew_angle().to_f32());
                xf!(p, skew(x, y, 0.0, 0.0), skew_em(x, y, 0.0, 0.0, k))
            }
            Paint::SkewAroundCenter(p) => {
                let (x, y, cx, cy) = (p.x_skew_angle().to_f32(), p.y_skew_angle().to_f32(), p.center_x().to_i16() as f32, p.center_y().to_i16() as f32);
                xf!(p, skew(x, y, cx, cy), skew_em(x, y, cx, cy, k))
            }
            // Leave variation resolution with skrifa; no partial draw escapes a failed parse.
            _ => return None,
        })
    }
}

pub(super) fn draw(font: &FontRef<'_>, gid: GlyphId, painter: &mut super::Painter) -> bool {
    let Ok(colr) = font.colr() else { return false };
    let k = 1.0 / font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f32;
    let Some(root) = (Reader { colr, remaining: 16384, k }).glyph(gid, 0) else { return false };
    root.draw(painter);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use skrifa::raw::{FontData, FontRead};

    fn color_line(extend: u8) -> Vec<u8> {
        let mut data = vec![extend, 0, 4];
        for offset in [2133i16, 4523, 9899, 12885] {
            data.extend_from_slice(&offset.to_be_bytes());
            data.extend_from_slice(&7u16.to_be_bytes());
            data.extend_from_slice(&16384i16.to_be_bytes());
        }
        data
    }

    #[test]
    fn pad_preserves_domain_and_repeat_uses_stop_range() {
        for extend in [0, 1] {
            let data = color_line(extend);
            let (stops, _, lo, hi) = line(ColorLine::read(FontData::new(&data)).unwrap()).unwrap();
            if extend == 0 {
                assert_eq!((lo, hi), (0.0, 1.0));
                assert_eq!(stops[0].offset, 2133.0 / 16384.0);
                assert_eq!(stops[3].offset, 12885.0 / 16384.0);
            } else {
                assert_eq!((lo, hi), (2133.0 / 16384.0, 12885.0 / 16384.0));
                assert_eq!((stops[0].offset, stops[3].offset), (0.0, 1.0));
            }
        }
        let mut data = color_line(0);
        data[3..5].copy_from_slice(&(-8192i16).to_be_bytes());
        data[21..23].copy_from_slice(&24576i16.to_be_bytes());
        let (stops, _, lo, hi) = line(ColorLine::read(FontData::new(&data)).unwrap()).unwrap();
        assert_eq!((lo, hi), (-0.5, 1.5));
        assert_eq!((stops[0].offset, stops[3].offset), (0.0, 1.0));
    }

    #[test]
    fn cyclic_colr_glyph_is_bounded() {
        let mut data = vec![0; 34];
        data[..2].copy_from_slice(&1u16.to_be_bytes());
        data[14..18].copy_from_slice(&34u32.to_be_bytes());
        data.extend_from_slice(&1u32.to_be_bytes());
        data.extend_from_slice(&1u16.to_be_bytes());
        data.extend_from_slice(&10u32.to_be_bytes());
        data.extend_from_slice(&[11, 0, 1]); // PaintColrGlyph referencing itself.
        let colr = Colr::read(FontData::new(&data)).unwrap();
        assert!((Reader { colr, remaining: 16384, k: 1.0 / 1000.0 }).glyph(GlyphId::new(1), 0).is_none());
    }
}
