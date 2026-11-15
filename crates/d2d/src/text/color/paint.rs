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
    Radial(Pt, f32, Pt, f32, Vec<ColorStop>, Extend),
    Sweep(Pt, f32, f32, Vec<ColorStop>, Extend),
}
impl Ink {
    fn brush(&self) -> CBrush<'_> {
        match self {
            Self::Solid(palette_index, alpha) => CBrush::Solid { palette_index: *palette_index, alpha: *alpha },
            // p0/p1 here are the raw endpoints; the device-space projection in `brush()` uses
            // `pending_linear` (set by `Node::draw`), so these only serve the skrifa fallback.
            Self::Linear(p0, p1, _p2, _lo, _hi, color_stops, extend) => CBrush::LinearGradient { p0: *p0, p1: *p1, color_stops, extend: *extend },
            Self::Radial(c0, r0, c1, r1, color_stops, extend) => CBrush::RadialGradient { c0: *c0, r0: *r0, c1: *c1, r1: *r1, color_stops, extend: *extend },
            Self::Sweep(c0, start_angle, end_angle, color_stops, extend) => CBrush::SweepGradient { c0: *c0, start_angle: *start_angle, end_angle: *end_angle, color_stops, extend: *extend },
        }
    }
}

enum Node {
    Empty,
    Ink(Ink),
    Group(Vec<Node>),
    Transform(Mat, Box<Node>),
    Glyph(GlyphId, Box<Node>),
    Clip(BoundingBox<f32>, Box<Node>),
    Composite(CompositeMode, Box<Node>, Box<Node>),
}
impl Node {
    fn leaf(&self) -> Option<(&Ink, Option<Mat>)> {
        match self {
            Self::Ink(ink) => Some((ink, None)),
            Self::Transform(m, child) => {
                let (ink, next) = child.leaf()?;
                Some((ink, Some(next.map_or(*m, |n| mul(m, &n)))))
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
        match self {
            Self::Empty => {}
            Self::Ink(ink) => { painter.pending_linear = pend(ink); painter.fill(ink.brush()); }
            Self::Group(nodes) => { for node in nodes { node.draw(painter); } }
            Self::Transform(m, child) => {
                painter.push_transform(transform(m)); child.draw(painter); painter.pop_transform();
            }
            Self::Glyph(gid, child) => {
                if let Some((ink, m)) = child.leaf() {
                    painter.pending_linear = pend(ink);
                    painter.fill_glyph(*gid, m.as_ref().map(transform), ink.brush());
                } else if !matches!(**child, Self::Empty) {
                    painter.push_clip_glyph(*gid); child.draw(painter); painter.pop_clip();
                }
            }
            Self::Clip(rect, child) => {
                painter.push_clip_box(*rect); child.draw(painter); painter.pop_clip();
            }
            Self::Composite(mode, backdrop, source) => {
                painter.push_layer(CompositeMode::SrcOver); backdrop.draw(painter);
                painter.push_layer(*mode); source.draw(painter);
                painter.pop_layer_with_mode(*mode); painter.pop_layer_with_mode(CompositeMode::SrcOver);
            }
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
fn skew(x: f32, y: f32, cx: f32, cy: f32) -> Mat {
    // DirectWrite's paint reader takes each shear with tanf of the angle (half-turns times π), not sin/cos.
    let tan = |v: f32| crate::crt::tanf(v * std::f32::consts::PI);
    let (x, y) = (tan(x), tan(y));
    [1.0, y, -x, 1.0, x * cy, -y * cx]
}

struct Reader<'a> { colr: Colr<'a>, remaining: u32 }
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
        macro_rules! xf {
            ($p:expr, $m:expr) => { Node::Transform($m, Box::new(self.node($p.paint().ok()?, depth + 1)?)) };
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
                Node::Ink(Ink::Radial(lerp(a, b, lo), r + dr * lo, lerp(a, b, hi), r + dr * hi, stops, extend))
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
                xf!(p, [m.xx().to_f32(), m.yx().to_f32(), m.xy().to_f32(), m.yy().to_f32(), m.dx().to_f32(), m.dy().to_f32()])
            }
            Paint::Translate(p) => xf!(p, [1.0, 0.0, 0.0, 1.0, p.dx().to_i16() as f32, p.dy().to_i16() as f32]),
            Paint::Scale(p) => xf!(p, scale(p.scale_x().to_f32(), p.scale_y().to_f32(), 0.0, 0.0)),
            Paint::ScaleAroundCenter(p) => xf!(p, scale(p.scale_x().to_f32(), p.scale_y().to_f32(), p.center_x().to_i16() as f32, p.center_y().to_i16() as f32)),
            Paint::ScaleUniform(p) => xf!(p, scale(p.scale().to_f32(), p.scale().to_f32(), 0.0, 0.0)),
            Paint::ScaleUniformAroundCenter(p) => xf!(p, scale(p.scale().to_f32(), p.scale().to_f32(), p.center_x().to_i16() as f32, p.center_y().to_i16() as f32)),
            Paint::Rotate(p) => xf!(p, rotate(p.angle().to_f32(), 0.0, 0.0)),
            Paint::RotateAroundCenter(p) => xf!(p, rotate(p.angle().to_f32(), p.center_x().to_i16() as f32, p.center_y().to_i16() as f32)),
            Paint::Skew(p) => xf!(p, skew(p.x_skew_angle().to_f32(), p.y_skew_angle().to_f32(), 0.0, 0.0)),
            Paint::SkewAroundCenter(p) => xf!(p, skew(p.x_skew_angle().to_f32(), p.y_skew_angle().to_f32(), p.center_x().to_i16() as f32, p.center_y().to_i16() as f32)),
            // Leave variation resolution with skrifa; no partial draw escapes a failed parse.
            _ => return None,
        })
    }
}

pub(super) fn draw(font: &FontRef<'_>, gid: GlyphId, painter: &mut super::Painter) -> bool {
    let Ok(colr) = font.colr() else { return false };
    let Some(root) = (Reader { colr, remaining: 16384 }).glyph(gid, 0) else { return false };
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
        assert!((Reader { colr, remaining: 16384 }).glyph(GlyphId::new(1), 0).is_none());
    }
}
