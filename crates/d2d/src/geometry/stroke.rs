//! Stroke widening: native `d2d1.dll`'s `CWidener`/`CSimplePen` (the MIL stroker — MIT-licensed
//! `dotnet/wpf` `WpfGfx/core/geometry/strokefigure.{cpp,h}`, `cpen.cpp`, confirmed against
//! d2d1's `CShapeBase::WidenToSink`). It offsets the skeleton polyline by the
//! pen (a circle of radius w/2 in device space), building two "rails" into a single closed outline
//! figure — caps and round joins emitted as bezier arcs — which is then filled under nonzero winding
//! by the same `CMILBezier` 28.4 cracker as ordinary fills (`super::flatten_fill_prims`).
//!
//! Scope: the polyline case (straight skeleton segments), which is what the geometry stroke scenes
//! exercise. The widener runs in `f64` (`GpPointR`) exactly as native; outline vertices are cast to
//! `f32` for the fill flattener, matching native's `CShape` (f32) figures.
//! ponytail: curve-skeleton widening (`CCubicSegment::Widen`) and the full `CDasher` synchronization
//! are not reproduced — dashed strokes use a plain arc-length split (good enough is not byte-exact);
//! curved strokes pre-flatten the skeleton. Both are noted where they bite.

use super::{flatten_fill_prims, Matrix, Point, Prim};

/// `D2D1_CAP_STYLE`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Cap {
    Flat,
    Square,
    Round,
    Triangle,
}
/// `D2D1_LINE_JOIN`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Join {
    Miter,
    Bevel,
    Round,
    MiterOrBevel,
}

/// `D2D1_STROKE_TRANSFORM_TYPE` (`ID2D1StrokeStyle1`): what the pen width is measured in.
/// `Normal` scales with the full device transform; `Fixed` ignores the world transform but keeps
/// the DPI (width in DIPs); `Hairline` forces one device pixel.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TransformType {
    Normal,
    Fixed,
    Hairline,
}

/// An immutable stroke style (`ID2D1StrokeStyle`/`ID2D1StrokeStyle1`). `dashes` are in stroke-width
/// units.
#[derive(Clone)]
pub struct StrokeStyle {
    pub start_cap: Cap,
    pub end_cap: Cap,
    pub dash_cap: Cap,
    pub join: Join,
    pub miter_limit: f32,
    pub dash_offset: f32,
    pub dashes: Vec<f32>,
    pub transform_type: TransformType,
}

impl Default for StrokeStyle {
    /// D2D's default: flat caps, miter join, miter limit 10, no dashes, normal transform.
    fn default() -> Self {
        StrokeStyle {
            start_cap: Cap::Flat,
            end_cap: Cap::Flat,
            dash_cap: Cap::Flat,
            join: Join::Miter,
            miter_limit: 10.0,
            dash_offset: 0.0,
            dashes: Vec::new(),
            transform_type: TransformType::Normal,
        }
    }
}

impl StrokeStyle {
    /// The dash lengths the pen dashes with: each length's absolute value, or `None` (a solid stroke)
    /// when their f32 sum is not positive. Native builds this when it sets up the pen; the style keeps
    /// the dashes as given (r15 20168: `[-1e9, -1e-20, 5.72]` dashes as `[1e9, 1e-20, 5.72]`).
    pub(crate) fn pen_dashes(&self) -> Option<Vec<f32>> {
        let d: Vec<f32> = self.dashes.iter().map(|v| v.abs()).collect();
        let sum = d.iter().fold(0.0f32, |s, &v| s + v);
        if sum <= 0.0 {
            None
        } else {
            Some(d)
        }
    }
}

// --- f64 vector helpers (GpPointR) ----------------------------------------------------------
type V = [f64; 2];
fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1]]
}
fn add(a: V, b: V) -> V {
    [a[0] + b[0], a[1] + b[1]]
}
fn scale(a: V, s: f64) -> V {
    [a[0] * s, a[1] * s]
}
fn neg(a: V) -> V {
    [-a[0], -a[1]]
}
fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn det(a: V, b: V) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
/// `GpPointR::TurnRight`: (x,y) → (-y, x).
fn turn_right(a: V) -> V {
    [-a[1], a[0]]
}
fn f32p(a: V) -> Point {
    [a[0] as f32, a[1] as f32]
}
/// Row-major linear transform (native's "point·matrix" convention): `[v·m0+v·m2, v·m1+v·m3]`.
fn lin(m: &[f64; 4], v: V) -> V {
    [v[0] * m[0] + v[1] * m[2], v[0] * m[1] + v[1] * m[3]]
}

const FUZZ: f64 = 1e-6;
const SQ_LENGTH_FUZZ: f64 = 1e-4;
const ARC_AS_BEZIER: f64 = 0.5522847498307933984;
const TOLERANCE: f64 = 0.25;
const IDENT4: [f64; 4] = [1.0, 0.0, 0.0, 1.0];

// `GetBezierDistance(dot, radius)` — control-point distance (fraction of radius) for a bezier
// approximating the arc between two radius vectors whose dot product is `rdot`.
fn get_bezier_distance(rdot: f64, radius: f64) -> f64 {
    let rad_sq = radius * radius;
    let ra = 0.5 * (rad_sq + rdot);
    if ra < 0.0 {
        return 0.0;
    }
    let denom_sq = rad_sq - ra;
    if denom_sq <= 0.0 {
        return 0.0;
    }
    let denom = denom_sq.sqrt();
    let num = (4.0 / 3.0) * (radius - ra.sqrt());
    if num <= denom * FUZZ {
        0.0
    } else {
        num / denom
    }
}

// Larger singular value of the 2×2 pen matrix (native's pen-setup inline formula).
fn max_singular(m: &[f64; 4]) -> f64 {
    let n0 = m[0] * m[0] + m[1] * m[1];
    let n1 = m[3] * m[3] + m[2] * m[2];
    let cross = m[0] * m[2] + m[3] * m[1];
    let s = (4.0 * cross * cross + (n0 - n1) * (n0 - n1)).sqrt();
    (0.5 * (s + (n1 + n0))).sqrt()
}

/// Max singular value (operator 2-norm) of a 3×2 transform's linear part — the worst-case point
/// magnification, used to scale the local curve flattener so device chords meet tolerance.
pub(super) fn max_singular_m(m: &super::Matrix) -> f64 {
    max_singular(&[m[0] as f64, m[1] as f64, m[2] as f64, m[3] as f64])
}

/// The pen shape after `CPen`'s per-primitive reset: the width/2 circle mapped by the
/// device matrix. A uniform result (rotation + uniform scale) collapses to a **circular** pen
/// (radius = column norm, matrix = identity); a non-uniform one keeps the 2×2 matrix + inverse as an
/// **elliptical** pen (radius 1), so stroke width varies with direction. `base` is the 2×2 the width
/// is measured in: the full device matrix (`Normal`), DPI only (`Fixed`), or identity with the width
/// forced to one device pixel (`Hairline`).
pub(super) struct PenShape {
    elliptical: bool,
    m: [f64; 4],   // M_pen (identity when circular)
    inv: [f64; 4], // inverse (valid when elliptical)
    radius: f64,
    refine: f64,
    /// A geometry realization's flattening tolerance: its rails reach the realization sweep as curves,
    /// flattened on the sweep's grid at that tolerance, not through the 28.4 fill cracker.
    rail_tol: Option<f32>,
    /// The pixel rect a COPY stroke's edge list is clipped to, which skips subdividing a curve off it
    /// (see `flatten_fill_prims_scale`).
    target: Option<[i32; 4]>,
    /// The dash pattern's unit: the pen width in pen space (before `base`).
    pub(super) dash_width: f64,
    /// The quadratic form measuring a device vector's pen-space length (the inverse of `base`); NaN
    /// when `base` does not invert.
    pub(super) dash_q: [f64; 3],
    /// `ID2D1Geometry::Widen`: collects each output contour as the widener emits it (lines and beziers,
    /// unflattened, unclipped) for the sink, beside the flattened pieces.
    pub(super) raw: Option<RawSink>,
    /// `StrokeContainsPoint`: the hit-test sink the pen streams its pieces to.
    pub(super) hit: Option<HitShared>,
    /// The device direction of the base's x axis: a figure whose every segment is too short to widen
    /// is drawn as a dot facing it.
    pub(super) dot_dir: V,
    /// The widener's tolerance: a line shorter than `tol·SQ_LENGTH_FUZZ` (squared) is skipped as
    /// degenerate, as a curve under `tol²·SQ_LENGTH_FUZZ` is.
    pub(super) tol: f64,
    /// A draw's viewable region `[l, t, r, b]`: the target, inflated by the pen's extent (see
    /// [`PenShape::with_viewable`]). The dasher starts no dash outside it; `Widen` has none.
    pub(super) viewable: Option<[f64; 4]>,
}

/// The contours a [`PenShape`] in Widen mode collects, as the pen's f64 rails.
pub(super) type RawSink = std::rc::Rc<std::cell::RefCell<Vec<RawRail>>>;

/// One collected contour: its f64 points and, per segment, whether it is a bezier (3 points) or a
/// line (1 point).
pub(crate) struct RawRail {
    pub(crate) pts: Vec<V>,
    pub(crate) bez: Vec<bool>,
}

impl RawRail {
    /// The contour as the sink receives it, in f32.
    pub(super) fn figure(&self) -> super::Figure {
        Rail { pts: self.pts.clone(), bez: self.bez.clone() }.figure()
    }
}

impl PenShape {
    /// How far the pen reaches from the centre line in device space (its radius under the pen matrix).
    pub(super) fn reach(&self) -> f64 {
        if self.elliptical { self.radius * (self.m[0].abs() + self.m[1].abs() + self.m[2].abs() + self.m[3].abs()) } else { self.radius }
    }

    /// Whether the pen can normalize the direction `dir` (in its pre-image for the elliptical pen):
    /// `StartFigure`/`DoCorner` fail with BAD_NUMBER on a zero or non-finite length.
    pub(super) fn normalizes(&self, dir: [f64; 2]) -> bool {
        let t = if self.elliptical { lin(&self.inv, dir) } else { dir };
        let l = (t[0] * t[0] + t[1] * t[1]).sqrt();
        l > 0.0 && l.is_finite()
    }

    /// This pen with its rails flattened for a geometry realization at `tol` (see `rail_tol`).
    pub(super) fn with_rail_tol(self, tol: Option<f32>) -> PenShape {
        PenShape { rail_tol: tol, ..self }
    }

    /// This pen filling into a COPY draw whose edge list is clipped to that pixel rect (see `target`).
    pub(super) fn with_target(self, target: Option<[i32; 4]>) -> PenShape {
        PenShape { target, ..self }
    }

    /// This pen drawing into a target of `size` pixels under the render transform `m`: its viewable
    /// region is the target inflated by the pen's extent — half the width, times the miter limit·√2
    /// for a mitered join or √2 for a square cap — mapped through `m` as the vector `(pad, pad)`, each
    /// side by the magnitude of that vector's component, in f32. A stroke not measured in world units
    /// keeps none.
    pub(super) fn with_viewable(self, size: Option<[usize; 2]>, width: f32, style: &StrokeStyle, m: &super::Matrix) -> PenShape {
        let viewable = size.filter(|_| style.transform_type == TransformType::Normal).map(|[w, h]| {
            let inflate = match style.join {
                Join::Miter | Join::MiterOrBevel => (if style.miter_limit >= 1.0 { style.miter_limit } else { 1.0 } as f64 * std::f64::consts::SQRT_2) as f32,
                _ if [style.start_cap, style.end_cap, style.dash_cap].contains(&Cap::Square) => std::f32::consts::SQRT_2,
                _ => 1.0,
            };
            let pad = width.abs() / 2.0 * inflate;
            let (px, py) = ((pad * m[0] + pad * m[2]).abs(), (pad * m[1] + pad * m[3]).abs());
            [(0.0 - px) as f64, (0.0 - py) as f64, (w as f32 + px) as f64, (h as f32 + py) as f64]
        });
        PenShape { viewable, ..self }
    }

    /// Whether a draw widens the cubic `p0..p3` (device space) as a single step to its end: native's
    /// widener does so for a curve whose bounds — the current point and the three control points, in
    /// f32 — lie wholly outside the viewable region, taking the end tangent there. NaN bounds are not
    /// outside it.
    pub(super) fn skips_curve(&self, p0: V, p1: V, p2: V, p3: V) -> bool {
        let Some([l, t, r, b]) = self.viewable else { return false };
        let ps = [p0, p1, p2, p3].map(|p| [p[0] as f32, p[1] as f32]);
        let lo = |k: usize| ps.iter().fold(f32::NAN, |m, p| p[k].min(m));
        let hi = |k: usize| ps.iter().fold(f32::NAN, |m, p| p[k].max(m));
        let (x0, x1, y0, y1) = (lo(0), hi(0), lo(1), hi(1));
        x1 >= x0 && y1 >= y0 && (x1 < l as f32 || (r as f32) < x0 || y1 < t as f32 || (b as f32) < y0)
    }

    /// This pen collecting its contours for `ID2D1Geometry::Widen` (see `raw`).
    pub(super) fn with_raw(self, raw: Option<RawSink>) -> PenShape {
        PenShape { raw, ..self }
    }

    /// This pen streaming its pieces to a stroke hit test (see `hit`).
    pub(super) fn with_hit(self, hit: Option<HitShared>) -> PenShape {
        PenShape { hit, ..self }
    }

    pub(super) fn build(width: f32, device: &Matrix, dpi_scale: [f32; 2], ttype: TransformType, tol: f64) -> Option<PenShape> {
        let w = if ttype == TransformType::Hairline { 1.0f32 } else { width };
        let r = (w * 0.5).abs();
        let b: [f32; 4] = match ttype {
            TransformType::Fixed => [dpi_scale[0], 0.0, 0.0, dpi_scale[1]],
            TransformType::Hairline => [1.0, 0.0, 0.0, 1.0],
            TransformType::Normal => [device[0], device[1], device[2], device[3]],
        };
        let dash_width = (r + r) as f64;
        let dash_q = inverse_quadratic(b);
        let dot_dir = [b[2] as f64 * 0.0 + b[0] as f64, b[3] as f64 * 0.0 + b[1] as f64];
        // M_pen = P·base with P = [r, 0, 0, r]: each entry the f32 sum of two f32 products, as native
        // forms it (an infinite width meets a zero base entry as NaN), stored as doubles.
        let mf = [b[2] * 0.0 + b[0] * r, b[1] * r + b[3] * 0.0, b[2] * r + b[0] * 0.0, b[1] * 0.0 + b[3] * r];
        let mut m = mf.map(|x| x as f64);
        let degen = (tol * 0.004) * (tol * 0.004);
        let uniform = m[0] == m[3] && m[1] == -m[2];
        let sqnorm = if uniform { m[0] * m[0] + m[1] * m[1] } else { m[0] * m[0] + m[1] * m[1] + m[2] * m[2] + m[3] * m[3] };
        // Unordered compares: a NaN pen is declined like one under the length fuzz.
        if !(sqnorm >= degen) {
            return None;
        }
        let maxsv = max_singular_f32(mf) as f64;
        // `ComputeRefinementThreshold`: radius²·(2·(1−tol/maxSV)² − 1), floored at −2·radius².
        let refine = |radius: f64| {
            let f = if tol > maxsv { -2.0 } else { let u = 1.0 - tol / maxsv; 2.0 * u * u - 1.0 };
            radius * radius * f
        };
        if uniform {
            let radius = sqnorm.sqrt();
            return Some(PenShape { elliptical: false, m: IDENT4, inv: IDENT4, radius, refine: refine(radius), rail_tol: None, target: None, dash_width, dash_q, raw: None, hit: None, dot_dir, viewable: None, tol });
        }
        let mut det = m[0] * m[3] - m[1] * m[2];
        if !(det.abs() >= degen) {
            return None;
        }
        if det < 0.0 {
            m[0] = -m[0]; // native negates M00/M01 and det for a reflecting matrix
            m[1] = -m[1];
            det = -det;
        }
        let id = 1.0 / det;
        let inv = [m[3] * id, -m[1] * id, -m[2] * id, m[0] * id];
        Some(PenShape { elliptical: true, m, inv, radius: 1.0, refine: refine(1.0), rail_tol: None, target: None, dash_width, dash_q, raw: None, hit: None, dot_dir, viewable: None, tol })
    }
}

/// The pen setup's larger singular value of the f32 pen matrix, in f32: the closed form, or the
/// larger diagonal magnitude when the matrix is diagonal.
fn max_singular_f32([a, b, c, d]: [f32; 4]) -> f32 {
    if b != 0.0 || c != 0.0 {
        let (n0, n1, x) = (b * b + a * a, c * c + d * d, b * d + c * a);
        let s = ((x * 4.0) * x + (n0 - n1) * (n0 - n1)).sqrt();
        ((s + (n1 + n0)) * 0.5).sqrt()
    } else if d.abs() < a.abs() {
        a.abs()
    } else {
        d.abs()
    }
}

/// `StrokeContainsPoint`'s widening sink. Native does not hit-test the widened outline: every piece
/// the pen streams (a quad from both rails' last points to their new ones, a join's or a cap's fan off
/// the other rail) is tested on its own, its winding count starting from zero, and the point is inside
/// when any piece winds round it or it comes within the tolerance of any piece's edge. Points are taken
/// relative to the query point in f64; a NaN one fails the query with BAD_NUMBER at the end of its piece.
pub(crate) struct HitSink {
    d: V, // the query point negated in f32
    tol: f64,
    prev: V,
    count: i32,
    on_edge: bool,
    nan: bool,
    hit: bool,
    pub(super) bad: bool,
    cur: [V; 2],
    saved: [V; 2],
}

/// A [`HitSink`] shared with the pens a query builds.
pub(super) type HitShared = std::rc::Rc<std::cell::RefCell<HitSink>>;

impl HitSink {
    pub(super) fn new(pt: Point, tol: f32) -> HitShared {
        let z = [0.0; 2];
        std::rc::Rc::new(std::cell::RefCell::new(HitSink { d: [(-pt[0]) as f64, (-pt[1]) as f64], tol: tol as f64, prev: z, count: 0, on_edge: false, nan: false, hit: false, bad: false, cur: [z; 2], saved: [z; 2] }))
    }

    /// Whether the point is inside the stroke.
    pub(super) fn contains(&self) -> bool {
        self.hit || self.on_edge
    }

    /// A point relative to the query point, through the helper's identity matrix (so an infinite
    /// coordinate makes the other one NaN).
    fn rel(&self, p: V) -> V {
        [(0.0 * p[1] + p[0]) + self.d[0], (p[1] + 0.0 * p[0]) + self.d[1]]
    }

    fn start_at(&mut self, p: V) {
        self.nan |= p[0].is_nan() || p[1].is_nan();
        self.prev = self.rel(p);
    }

    fn step(&mut self, q: V) {
        super::hit_step(self.tol * self.tol, &mut self.on_edge, &mut self.count, self.prev, q);
        self.nan |= q[0].is_nan() || q[1].is_nan();
        self.prev = q;
    }

    fn line(&mut self, p: V) {
        let q = self.rel(p);
        self.step(q);
    }

    fn open(&mut self, side: usize) {
        self.count = 0;
        self.start_at(self.cur[1 - side]);
        self.line(self.cur[side]);
    }

    fn close(&mut self, side: usize) -> bool {
        self.line(self.cur[1 - side]);
        if self.nan {
            self.bad = true;
            return false;
        }
        self.hit |= self.count != 0;
        true
    }

    fn quad_to(&mut self, l: V, r: V) {
        if self.bad {
            return;
        }
        self.open(LEFT);
        self.line(l);
        self.line(r);
        if self.close(LEFT) {
            self.cur = [l, r];
        }
    }

    fn polyline(&mut self, side: usize, pts: &[V]) {
        if self.bad {
            return;
        }
        self.open(side);
        for &p in pts {
            self.line(p);
        }
        if self.close(side) {
            if let Some(&p) = pts.last() {
                self.cur[side] = p;
            }
        }
    }

    fn curve(&mut self, side: usize, c1: V, c2: V, e: V) {
        if self.bad {
            return;
        }
        self.open(side);
        let mut pts = Vec::new();
        flatten_cubic_tan(self.prev, self.rel(c1), self.rel(c2), self.rel(e), &mut pts, self.tol.max(0.0) * 6.0);
        for (q, _) in pts {
            self.step(q);
        }
        if self.close(side) {
            self.cur[side] = e;
        }
    }

    fn switch_sides(&mut self) {
        if self.bad {
            return;
        }
        self.count = 0;
        self.start_at(self.cur[LEFT]);
        self.line(self.cur[RIGHT]);
        self.cur.swap(LEFT, RIGHT);
    }

    /// The widening sink's caps at terminal `t` (0 start, 1 end), from the pen's rails and its pen
    /// vector `v`.
    fn cap(&mut self, cap: Cap, t: usize, center: V, rails: [V; 2], v: V) {
        match cap {
            Cap::Flat => self.polyline(LEFT, &[rails[t]]),
            Cap::Square => {
                let v = if t == 1 { neg(v) } else { v };
                self.polyline(LEFT, &[sub(rails[1 - t], v), sub(rails[t], v), rails[t]]);
            }
            Cap::Triangle => self.polyline(LEFT, &[if t != 0 { add(center, v) } else { sub(center, v) }, rails[t]]),
            Cap::Round => {
                let v = if t == 0 { neg(v) } else { v };
                let mid = add(center, v);
                let across = scale(sub(rails[t], center), ARC_AS_BEZIER);
                let along = scale(v, ARC_AS_BEZIER);
                self.curve(LEFT, add(along, rails[1 - t]), sub(mid, across), mid);
                self.curve(LEFT, add(mid, across), add(along, rails[t]), rails[t]);
            }
        }
    }
}

/// `|v·B⁻¹|²` as a quadratic form `[xx, xy, yy]` in a device vector `v` (the 2×2 inverse in `f64`);
/// all NaN when `B` is singular.
fn inverse_quadratic(b: [f32; 4]) -> [f64; 3] {
    let [a, b, c, d] = b.map(|x| x as f64);
    let det = a * d - b * c;
    let id = 1.0 / det;
    if det == 0.0 || !id.is_finite() {
        return [f64::NAN; 3];
    }
    let i = [id * d, -b * id, -c * id, id * a];
    let xy = i[3] * i[1] + i[2] * i[0];
    [i[0] * i[0] + i[1] * i[1], xy + xy, i[2] * i[2] + i[3] * i[3]]
}

// --- a rail: a polyline of line/bezier segments (MIL m_rgPoints + m_rgTypes) ----------------
#[derive(Default)]
struct Rail {
    pts: Vec<V>,
    bez: Vec<bool>, // per-segment: false=line (1 pt), true=bezier (3 pts)
}
impl Rail {
    fn start(p: V) -> Rail {
        Rail { pts: vec![p], bez: Vec::new() }
    }
    fn line_to(&mut self, p: V) {
        self.pts.push(p);
        self.bez.push(false);
    }
    fn bezier_to(&mut self, c1: V, c2: V, p: V) {
        self.pts.push(c1);
        self.pts.push(c2);
        self.pts.push(p);
        self.bez.push(true);
    }
    fn set_end(&mut self, p: V) {
        *self.pts.last_mut().unwrap() = p;
    }
    /// The rail as a sink figure: its start, then its lines and beziers, each point cast to f32.
    fn figure(&self) -> super::Figure {
        let mut segments = Vec::with_capacity(self.bez.len());
        let mut i = 1;
        for &b in &self.bez {
            if b {
                segments.push(super::Segment::Cubic(f32p(self.pts[i]), f32p(self.pts[i + 1]), f32p(self.pts[i + 2])));
                i += 3;
            } else {
                segments.push(super::Segment::Line(f32p(self.pts[i])));
                i += 1;
            }
        }
        super::Figure { start: f32p(self.pts[0]), filled: true, closed: false, segments, seg_flags: Vec::new() }
    }
    /// The rail traversed backward.
    fn reversed(&self) -> Rail {
        let mut r = Rail::start(*self.pts.last().unwrap());
        r.reverse_join(self);
        r
    }
    // Append `other` reversed (skipping its last point, assumed to equal our end) — MIL ReverseJoin.
    fn reverse_join(&mut self, other: &Rail) {
        if other.pts.len() < 2 || other.bez.is_empty() {
            return;
        }
        for i in (0..other.pts.len() - 1).rev() {
            self.pts.push(other.pts[i]);
        }
        for i in (0..other.bez.len()).rev() {
            self.bez.push(other.bez[i]);
        }
    }
    // Flatten the rail (closed) through the native fill cracker and return a device polygon — or, for a
    // geometry realization (`rail_tol`), on the realization sweep's grid at its tolerance.
    fn flatten_closed(&self, rail_tol: Option<f32>, target: Option<[i32; 4]>) -> Vec<Point> {
        let mut prims: Vec<Prim> = Vec::new();
        let mut i = 0usize;
        for &is_bez in &self.bez {
            if is_bez {
                prims.push(Prim::Cubic([0.0; 2], f32p(self.pts[i + 1]), f32p(self.pts[i + 2]), f32p(self.pts[i + 3])));
                i += 3;
            } else {
                prims.push(Prim::Line([0.0; 2], f32p(self.pts[i + 1])));
                i += 1;
            }
        }
        // Native strip-clips a widened stroke shape past |524287| against its [-524287, 524287] rect
        // (CShapeBase::ClipWithRect) before the 28.4 flattener, exactly like a huge fill — else the
        // flattener's ±2^23 clamp distorts the rail slopes. Route a huge rail through the same bezier
        // strip clip; a finite rail keeps the plain fill cracker.
        if let Some(tol) = rail_tol {
            let mut poly = vec![f32p(self.pts[0])];
            for pr in &prims {
                match *pr {
                    Prim::Line(_, e) => poly.push(e),
                    Prim::Cubic(_, c1, c2, e) => {
                        let p0 = *poly.last().unwrap();
                        super::realization::flatten_cubic(p0, c1, c2, e, &mut poly, tol as f64);
                    }
                }
            }
            return poly;
        }
        let huge = self.pts.iter().any(|p| p[0].abs() > 524_287.0 || p[1].abs() > 524_287.0);
        super::fill_figure_pts(f32p(self.pts[0]), &prims, 16.0, huge, target).unwrap_or_default()
    }
}

// --- the simple pen (circular, device space) ------------------------------------------------
struct Pen {
    r: f64,
    r_sq: f64,
    nominal_miter_limit: f64,
    miter_limit: f64,
    refine: f64,
    // The elliptical (matrix) pen (`byte160==0`): tangents map through `inv` to the pre-image before
    // normalizing, offsets back through `m` to device space. Circular pens leave both at identity.
    elliptical: bool,
    m: [f64; 4],
    inv: [f64; 4],
    join: Join,
    vec_rad: V,
    vec_offset: V,
    cur: [V; 2], // RAIL_LEFT=0, RAIL_RIGHT=1
    rail: [Rail; 2],
    prev_pt: V,  // the last accepted skeleton point (chord origin for `round_to`)
    prev_dir: V, // the last accepted direction (curve tangent / line / chord)
    // Native's pen returns `0x88990011` (BAD_NUMBER) and aborts the WHOLE `DrawGeometry` when a tangent
    // can't be normalized: `CPen::StartFigure`/`UpdateOffset`/`DoCorner` fail the `len>0 && _finite(len)`
    // check (a zero/`inf`/`NaN` direction), and `AcceptCurvePoint` fails it for a PURE-inf curve tangent
    // (a `NaN` curve tangent instead takes the near-zero reuse branch). VM trap, fuzz 17866: its
    // inf-arc-radius cubics give an `inf` first tangent → StartFigure errors → 9 stack captures → the
    // stroke draws nothing. This latches that abort so the widener voids the whole stroke.
    errored: bool,
    rail_tol: Option<f32>,
    target: Option<[i32; 4]>,
    // A closed figure ends as ONE contour (outer rail, flat cap, inner rail reversed), as native's
    // widening sink adds it; cleared for a realized rectangle, which native widens as two rectangles.
    bridge: bool,
    raw: Option<RawSink>,
    // A plain (undashed) stroke collecting for `ID2D1Geometry::Widen`: native's widening sink takes a
    // figure's start cap as its own figure after the stroke, whose start stays flat. Dash caps stay
    // inline, so the dasher's pens leave this off.
    split_start_cap: bool,
    start_cap_fig: Option<Rail>,
    hit: Option<HitShared>,
    // A plain open figure's start cap, which the widener draws after the figure as a figure of its own:
    // its point, direction and cap.
    deferred_cap: Option<(V, V, Cap)>,
}

const LEFT: usize = 0;
const RIGHT: usize = 1;

impl Pen {
    fn new(pen: &PenShape, style: &StrokeStyle) -> Pen {
        let r = pen.radius;
        let nominal = (style.miter_limit as f64).max(1.0);
        Pen {
            r,
            r_sq: r * r,
            nominal_miter_limit: nominal,
            miter_limit: nominal * r,
            refine: pen.refine,
            elliptical: pen.elliptical,
            m: pen.m,
            inv: pen.inv,
            join: style.join,
            vec_rad: [0.0; 2],
            vec_offset: [0.0; 2],
            cur: [[0.0; 2]; 2],
            rail: [Rail::default(), Rail::default()],
            prev_pt: [0.0; 2],
            prev_dir: [0.0; 2],
            errored: false,
            rail_tol: pen.rail_tol,
            target: pen.target,
            bridge: true,
            raw: pen.raw.clone(),
            split_start_cap: false,
            start_cap_fig: None,
            hit: pen.hit.clone(),
            deferred_cap: None,
        }
    }

    /// The hit-test sink, when this pen streams to one.
    fn hit(&self) -> Option<std::cell::RefMut<'_, HitSink>> {
        self.hit.as_ref().map(|h| h.borrow_mut())
    }

    /// The hit-test sink's cap at terminal `t` from the pen's current rails.
    fn hit_cap(&self, t: usize, center: V, cap: Cap) {
        if let Some(mut h) = self.hit() {
            h.cap(cap, t, center, self.cur, self.pen_vec(self.vec_rad));
        }
    }

    /// Record an output contour for `ID2D1Geometry::Widen` when collecting (see [`PenShape::raw`]).
    fn emit(&self, rail: &Rail) {
        if let Some(raw) = &self.raw {
            raw.borrow_mut().push(RawRail { pts: rail.pts.clone(), bez: rail.bez.clone() });
        }
    }
    // Radius vector (`CPen::UpdateOffset`): the tangent mapped to the pen's pre-image (`inv·dir` for
    // the elliptical pen, else `dir`), normalized with `sqrt(x²+y²)` (`_mm_sqrt_pd`, NOT hypot) and
    // scaled to the pen radius (1 for the elliptical pen — the scale lives in the matrix).
    fn radius_vec(&mut self, dir: V) -> V {
        let t = if self.elliptical { lin(&self.inv, dir) } else { dir };
        let l = (t[0] * t[0] + t[1] * t[1]).sqrt();
        if l > 0.0 && l.is_finite() {
            scale(t, self.r / l)
        } else {
            // Native `CPen::UpdateOffset`/`StartFigure`/`DoCorner` return `0x88990011` here (a zero,
            // `inf`, or `NaN` direction fails `len>0 && _finite(len)`), aborting the whole stroke.
            self.errored = true;
            self.vec_rad
        }
    }
    // Device-space offset from a pre-image radius vector: turn right, then (elliptical) map through
    // the pen matrix back to device space.
    fn offset_of(&self, vec_rad: V) -> V {
        self.pen_vec(turn_right(vec_rad))
    }
    // `GetPenVector`: a pre-image vector in device space (through the pen matrix for the elliptical pen).
    fn pen_vec(&self, v: V) -> V {
        if self.elliptical { lin(&self.m, v) } else { v }
    }
    // A curve point's radius vector (`ComputeRadiusVector` as `AcceptCurvePoint` inlines it): the
    // direction mapped to the pen's pre-image, scaled to the radius; `None` (BAD_NUMBER) when that
    // cannot be normalized.
    fn curve_rad(&self, dir: V, sq: f64) -> Option<V> {
        if !self.elliptical {
            return Some(scale(dir, self.r / sq.sqrt()));
        }
        let t = lin(&self.inv, dir);
        let l = (t[0] * t[0] + t[1] * t[1]).sqrt();
        (l > 0.0 && l.is_finite()).then(|| scale(t, self.r / l))
    }
    fn update_offset(&mut self, dir: V) {
        self.vec_rad = self.radius_vec(dir);
        self.vec_offset = self.offset_of(self.vec_rad);
    }

    fn start_figure(&mut self, pt: V, dir: V, cap: Cap) {
        self.update_offset(dir);
        self.prev_pt = pt;
        self.prev_dir = dir;
        self.cur[LEFT] = sub(pt, self.vec_offset);
        self.cur[RIGHT] = add(pt, self.vec_offset);
        // StartWith: both rails start at the right offset point.
        self.rail[LEFT] = Rail::start(self.cur[RIGHT]);
        self.rail[RIGHT] = Rail::start(self.cur[RIGHT]);
        if let Some(mut h) = self.hit() {
            h.cur = [self.cur[RIGHT]; 2];
        }
        if self.split_start_cap && self.hit.is_some() {
            // The widener starts a plain figure flat and draws its start cap after it.
            self.hit_cap(RAIL_START, pt, Cap::Flat);
            if cap != Cap::Flat {
                self.deferred_cap = Some((pt, dir, cap));
            }
            let hit = self.hit.take();
            self.base_cap(RAIL_START, pt, cap);
            self.hit = hit;
            return;
        }
        if self.split_start_cap && self.raw.is_some() && cap != Cap::Flat {
            // The cap on scratch rails as its own figure: right offset, the cap, back to the right offset
            // (r23 28061, widen probes: a round cap streams `bezier bezier line`, a square one four lines).
            let (rails, cur) = (std::mem::take(&mut self.rail), self.cur);
            self.rail = [Rail::start(cur[RIGHT]), Rail::start(cur[RIGHT])];
            self.base_cap(RAIL_START, pt, cap);
            let scratch = std::mem::take(&mut self.rail[LEFT]);
            let mut fig = Rail::start(cur[RIGHT]);
            if scratch.pts[0] != cur[RIGHT] {
                fig.line_to(scratch.pts[0]);
            }
            fig.pts.extend_from_slice(&scratch.pts[1..]);
            fig.bez.extend_from_slice(&scratch.bez);
            fig.line_to(cur[RIGHT]);
            self.start_cap_fig = Some(fig);
            (self.rail, self.cur) = (rails, cur);
            self.cap_flat(RAIL_START);
            return;
        }
        self.base_cap(RAIL_START, pt, cap);
    }

    // Closed figures have no caps: the stroke region is the annulus between the outer and inner
    // rails, each starting at its own offset point; `finish_two` joins them into native's contour.
    fn start_closed(&mut self, pt: V, dir: V) {
        self.update_offset(dir);
        self.prev_pt = pt;
        self.prev_dir = dir;
        self.cur[LEFT] = sub(pt, self.vec_offset);
        self.cur[RIGHT] = add(pt, self.vec_offset);
        self.rail[LEFT] = Rail::start(self.cur[LEFT]);
        self.rail[RIGHT] = Rail::start(self.cur[RIGHT]);
        // The widener starts a closed figure flat: both rails at the right offset point, the left one
        // across to its own.
        if let Some(mut h) = self.hit() {
            h.cur = [self.cur[RIGHT]; 2];
        }
        self.hit_cap(RAIL_START, pt, Cap::Flat);
    }

    // A closed figure's two rails. Normally they are nested loops wound the same way, so nonzero winding
    // fills the band when the inner is vertex-reversed (two contours, outer + reversed inner). But when
    // the pen is wider than the shape's local half-width the inner offset INVERTS (the thin-ellipse
    // 8451): the inner rail crosses to the other side, so its signed area flips relative to the outer,
    // and two opposite loops cancel to empty. Native's `CShapeWideningSink::AddFigure` always emits ONE
    // self-crossing contour (flat end cap + `ReverseJoin`), which winds +2 in the inverted centre and
    // fills. We emit native's contour whenever `bridge` is set: a realization's region sweep keeps the
    // crossings where the rails meet the cap (r21 26529), and a COPY fill's edge walk sees its point
    // order (it clips 32 points at a time). Without the bridge (a realized rectangle), the two contours
    // become one on a detected inversion (same-signed rail areas ⇒ not nested); `flatten_closed`
    // reflects any inner inversion, so the device areas are the test.
    fn finish_two(&mut self) -> Vec<Vec<Point>> {
        self.hit_cap(RAIL_END, self.prev_pt, Cap::Flat);
        let left = if self.rail[LEFT].bez.is_empty() { Vec::new() } else { self.rail[LEFT].flatten_closed(self.rail_tol, self.target) };
        let right = if self.rail[RIGHT].bez.is_empty() { Vec::new() } else { self.rail[RIGHT].flatten_closed(self.rail_tol, self.target) };
        // Signed area (shoelace, device space) of a closed polyline; sign = winding direction.
        let area = |p: &[Point]| -> f64 {
            let n = p.len();
            if n < 3 {
                return 0.0;
            }
            (0..n).map(|i| {
                let a = p[i];
                let b = p[(i + 1) % n];
                a[0] as f64 * b[1] as f64 - b[0] as f64 * a[1] as f64
            }).sum::<f64>()
                * 0.5
        };
        let (al, ar) = (area(&left), area(&right));
        // Nested (normal): outer and reversed-inner wind oppositely ⇒ the raw loops wind the SAME way, so
        // sign(al)==sign(ar). Inverted inner ⇒ its loop flipped ⇒ signs DIFFER (or inner collapsed).
        let inverted = al != 0.0 && ar != 0.0 && (al > 0.0) != (ar > 0.0) && !left.is_empty() && !right.is_empty();
        if inverted || self.bridge {
            self.rail[LEFT].line_to(self.cur[RIGHT]); // MIL EndStrokeClosed flat cap: rails share a last point
            let r = std::mem::take(&mut self.rail[RIGHT]);
            let l = std::mem::take(&mut self.rail[LEFT]);
            // Native starts both rails at the right rail's start point, so the contour opens with the line
            // from it to the left rail's start: right start, left rail, flat cap, right rail reversed.
            let mut fig = Rail::start(r.pts[0]);
            fig.line_to(l.pts[0]);
            fig.pts.extend_from_slice(&l.pts[1..]);
            fig.bez.extend_from_slice(&l.bez);
            fig.reverse_join(&r); // one self-crossing contour (LEFT ++ RIGHT reversed)
            self.emit(&fig);
            let fig = fig.flatten_closed(self.rail_tol, self.target);
            return if fig.len() >= 2 { vec![fig] } else { Vec::new() };
        }
        let mut out = Vec::new();
        if left.len() >= 2 {
            self.emit(&self.rail[LEFT]);
            out.push(left);
        }
        if right.len() >= 2 {
            self.emit(&self.rail[RIGHT].reversed());
            let mut right = right;
            right.reverse();
            out.push(right);
        }
        out
    }

    // Widen a line segment to `point` (`AcceptLinePoint` → `QuadTo`): line both rails to `point` ∓
    // the current offset. Also used as native `AdvanceToPoint` (same body, offset already set). Native
    // checks nothing here: a non-finite point goes into the rails as is, and only a direction the pen
    // cannot normalize fails the stroke (the rectangle widener can hand it a NaN corner along a finite
    // direction).
    fn accept_line(&mut self, point: V) {
        self.cur[LEFT] = sub(point, self.vec_offset);
        self.cur[RIGHT] = add(point, self.vec_offset);
        self.rail[LEFT].line_to(self.cur[LEFT]);
        self.rail[RIGHT].line_to(self.cur[RIGHT]);
        if let Some(mut h) = self.hit() {
            h.quad_to(self.cur[LEFT], self.cur[RIGHT]);
        }
        self.prev_pt = point;
    }

    // Widen to a flattened curve point (`CPen::AcceptCurvePoint`). The radius follows the curve's
    // **analytic tangent**, normalized with `sqrt` (not hypot). When the new radius stays close to
    // the previous one (`dot ≥ refine`) AND both rails step the same way (`dot(ΔR,ΔL) > 0`), the two
    // rails extend directly (`QuadTo`). Otherwise — an inner-rail backtrack or a cusp — native rounds
    // the corner to the chord via `RoundTo`, and on the curve's last point (`is_last`) rounds again
    // from the chord to the true endpoint tangent. `is_last` is `CCubicSegment::Widen`'s last-point flag.
    fn accept_curve_point(&mut self, point: V, tan: V, is_last: bool) {
        let tsq = tan[0] * tan[0] + tan[1] * tan[1];
        // A near-zero OR NaN tangent reuses the previous radius (native's unordered compare takes that
        // branch); an infinite one cannot be normalized and is the BAD_NUMBER that voids the whole stroke
        // (`ID2D1Geometry::Widen` returns 0x88990011). A NaN point errors below, on its chord.
        if tsq.is_infinite() {
            self.errored = true;
            return;
        }
        let new_rad = if tsq >= 1.0e-12 {
            match self.curve_rad(tan, tsq) {
                Some(r) => r,
                None => {
                    self.errored = true;
                    return;
                }
            }
        } else {
            self.vec_rad
        };
        if dot(new_rad, self.vec_rad) >= self.refine {
            let off = self.offset_of(new_rad);
            let rail_l = sub(point, off);
            let rail_r = add(point, off);
            if dot(sub(rail_r, self.cur[RIGHT]), sub(rail_l, self.cur[LEFT])) > 0.0 {
                self.vec_rad = new_rad;
                self.vec_offset = off;
                self.cur[LEFT] = rail_l;
                self.cur[RIGHT] = rail_r;
                self.rail[LEFT].line_to(rail_l);
                self.rail[RIGHT].line_to(rail_r);
                if let Some(mut h) = self.hit() {
                    h.quad_to(rail_l, rail_r);
                }
                self.prev_pt = point;
                self.prev_dir = tan;
                return;
            }
        }
        // RoundTo path: round from the previous direction to the chord, advance along the chord,
        // then (last point only) round from the chord to the true endpoint tangent.
        let chord = sub(point, self.prev_pt);
        let csq = chord[0] * chord[0] + chord[1] * chord[1];
        // A NaN or infinite chord (a non-finite flattened point) cannot be normalized: BAD_NUMBER.
        if !csq.is_finite() {
            self.errored = true;
            return;
        }
        if csq > 1.0e-12 {
            let Some(chord_rad) = self.curve_rad(chord, csq) else {
                self.errored = true;
                return;
            };
            self.round_to(chord_rad, self.prev_pt, self.prev_dir, chord);
            self.accept_line(point); // AdvanceToPoint: both rails along the chord offset
            if is_last {
                self.round_to(new_rad, point, chord, tan);
                self.prev_dir = tan;
            } else {
                self.prev_dir = chord;
            }
        }
        self.prev_pt = point;
    }

    // `CPen::RoundTo`: round the corner at `center` turning from `prev_dir` to `new_dir`, with the
    // new radius `new_rad`. Streamed through `CShapeWideningSink` (the no-op `SyncWithCenter`/
    // `Begin/EndJoin` slots dropped): round the convex/outer rail, inner-corner the concave rail, and
    // emit the inner wedge as a sub-figure detour on the LEFT rail (`BeginSubFigure`/`EndSubFigure`).
    fn round_to(&mut self, new_rad: V, center: V, prev_dir: V, new_dir: V) {
        let prev_rad = self.vec_rad;
        let off = self.offset_of(new_rad);
        let new_rail = [sub(center, off), add(center, off)]; // [LEFT, RIGHT]
        let cross = det(prev_dir, new_dir);
        let outer = if cross <= 0.0 { RIGHT } else { LEFT };
        let inner = 1 - outer;
        let inner_prev = self.cur[inner];
        self.round_corner_ex(center, self.cur[outer], new_rail[outer], prev_rad, new_rad, outer, outer);
        self.cur[inner] = new_rail[inner];
        self.rail[inner].line_to(center);
        self.rail[inner].line_to(new_rail[inner]);
        if let Some(mut h) = self.hit() {
            h.cur[inner] = new_rail[inner];
        }
        // Inner wedge sub-figure (always on the LEFT rail), with a second rounding whose radii are
        // negated for a left turn and swapped for a right turn (native's sign-flip handling).
        let (p39, p41, prad, nrad) = if cross > 0.0 {
            (inner_prev, new_rail[inner], neg(prev_rad), neg(new_rad))
        } else {
            (new_rail[inner], inner_prev, new_rad, prev_rad)
        };
        let saved = self.cur[LEFT]; // BeginSubFigure saves the LEFT rail's current end (= new_rail[LEFT])
        self.rail[LEFT].line_to(p39);
        if let Some(mut h) = self.hit() {
            h.saved = h.cur;
            h.cur = [p39; 2];
        }
        self.round_corner_ex(center, p39, p41, prad, nrad, LEFT, LEFT);
        self.rail[LEFT].line_to(p41);
        self.rail[LEFT].line_to(center);
        self.rail[LEFT].line_to(p39);
        self.rail[LEFT].line_to(saved); // EndSubFigure closes the detour
        if let Some(mut h) = self.hit() {
            h.polyline(LEFT, &[p41, center, p39]);
            h.cur = h.saved;
        }
        self.cur[LEFT] = saved; // the sub-figure's rail-prev change is reverted
        self.vec_rad = new_rad;
        self.vec_offset = off;
        self.prev_pt = center;
    }

    fn base_cap(&mut self, which: usize, center: V, cap: Cap) {
        if self.hit.is_some() {
            self.hit_cap(which, center, cap);
            let hit = self.hit.take();
            self.base_cap(which, center, cap);
            self.hit = hit;
            return;
        }
        let side = which; // TERMINAL2SIDE: start→LEFT(0), end→RIGHT(1)
        match cap {
            Cap::Flat => self.cap_flat(side),
            Cap::Square => self.square_cap(which, center),
            Cap::Triangle => {
                let mut p = self.pen_vec(self.vec_rad); // GetPenVector(vecRad)
                p = if which == RAIL_START { sub(center, p) } else { add(center, p) };
                // CapTriangle: left rail: apex, then base-end (the `side` offset).
                self.rail[LEFT].line_to(p);
                self.rail[LEFT].line_to(self.cur[side]);
            }
            Cap::Round => self.round_cap(which, center),
        }
    }

    // CapFlat: left rail current ← opposite offset, then line to `side` offset.
    fn cap_flat(&mut self, side: usize) {
        self.rail[LEFT].set_end(self.cur[1 - side]);
        self.rail[LEFT].line_to(self.cur[side]);
    }

    fn square_cap(&mut self, which: usize, _center: V) {
        let v = self.pen_vec(self.vec_rad); // GetPenVector(vecRad)
        if which == RAIL_START {
            // From the right offset, where both rails start: back by V, across, and forward to the left
            // offset, all on the left rail.
            self.rail[LEFT].line_to(sub(self.cur[RIGHT], v));
            self.rail[LEFT].line_to(sub(self.cur[LEFT], v));
            self.rail[LEFT].line_to(self.cur[LEFT]);
        } else {
            // Extend the segment forward by V, then flat cap.
            let end = add(self.rail_spine_end(), v);
            self.accept_line(end);
            self.cap_flat(RAIL_END);
        }
    }
    // The current spine endpoint = midpoint of the two rail currents.
    fn rail_spine_end(&self) -> V {
        scale(add(self.cur[LEFT], self.cur[RIGHT]), 0.5)
    }

    fn round_cap(&mut self, which: usize, center: V) {
        let side = which;
        let pt_end = self.cur[side];
        let pt_start = self.cur[1 - side];
        let vec_across = scale(sub(pt_end, center), ARC_AS_BEZIER);
        let mut vec_along = self.pen_vec(self.vec_rad); // GetPenVector
        if which == RAIL_START {
            vec_along = neg(vec_along);
        }
        let pt_mid = add(center, vec_along);
        let vec_along = scale(vec_along, ARC_AS_BEZIER);
        // BezierCap → two beziers on the LEFT rail.
        self.rail[LEFT].bezier_to(add(pt_start, vec_along), sub(pt_mid, vec_across), pt_mid);
        self.rail[LEFT].bezier_to(add(pt_mid, vec_across), add(pt_end, vec_along), pt_end);
    }

    // The inner side of a corner goes through the centre to the next segment's offset point, which
    // becomes that side's current point (so a closing flat cap starts from it).
    fn do_inner_corner(&mut self, side: usize, center: V, next: [V; 2]) {
        self.cur[side] = next[side];
        if let Some(mut h) = self.hit() {
            h.cur[side] = next[side];
        }
        self.rail[side].line_to(center);
        self.rail[side].line_to(next[side]);
    }

    /// The widener's corner between two segments: after a skipped (too short) segment, a non-round join
    /// is a miter limited to 1.
    fn corner(&mut self, center: V, vec_in: V, vec_out: V, after_skip: bool) {
        if !after_skip || self.join == Join::Round {
            return self.do_corner_join(center, vec_in, vec_out, false);
        }
        let saved = (self.join, self.nominal_miter_limit, self.miter_limit);
        (self.join, self.nominal_miter_limit, self.miter_limit) = (Join::Miter, 1.0, self.r);
        self.do_corner_join(center, vec_in, vec_out, false);
        (self.join, self.nominal_miter_limit, self.miter_limit) = saved;
    }

    /// The widener's corner: forced round, else after a skipped segment a miter limited to 1.
    fn corner_flags(&mut self, center: V, vec_in: V, vec_out: V, force_round: bool, after_skip: bool) {
        if force_round {
            self.do_corner_join(center, vec_in, vec_out, true);
        } else {
            self.corner(center, vec_in, vec_out, after_skip);
        }
    }

    fn do_corner(&mut self, center: V, vec_in: V, vec_out: V) {
        self.do_corner_join(center, vec_in, vec_out, false);
    }

    /// `do_corner`, with `force_round` overriding the pen's join to `Round` for this corner only — the
    /// per-segment `D2D1_PATH_SEGMENT_FORCE_ROUND_LINE_JOIN` flag.
    fn do_corner_join(&mut self, center: V, vec_in: V, vec_out: V, force_round: bool) {
        let vec_rad = self.radius_vec(vec_out);
        let vec_offset = self.offset_of(vec_rad);
        let next = [sub(center, vec_offset), add(center, vec_offset)];

        let rdet = det(vec_in, vec_out);
        let rdot_dir = dot(vec_in, vec_out);
        let mut f180 = false;
        let side;
        if rdet.abs() <= rdot_dir.abs() * SQ_LENGTH_FUZZ {
            if rdot_dir > 0.0 {
                // A flat join does nothing: the pen keeps its current points, radius and offset.
                return;
            }
            f180 = true;
            side = RIGHT;
        } else if rdet > 0.0 {
            side = LEFT;
        } else {
            side = RIGHT;
        }

        let join = if force_round { Join::Round } else { self.join };
        match join {
            Join::Round => {
                self.do_inner_corner(1 - side, center, next);
                self.round_corner_ex(center, self.cur[side], next[side], self.vec_rad, vec_rad, side, side);
            }
            Join::Bevel => {
                if f180 {
                    self.switch_sides();
                } else {
                    self.do_inner_corner(1 - side, center, next);
                    self.bevel_corner(side, next[side]);
                }
            }
            Join::Miter | Join::MiterOrBevel => {
                if f180 {
                    if self.join == Join::Miter {
                        self.do_180_miter();
                    } else {
                        self.switch_sides();
                    }
                } else {
                    self.do_inner_corner(1 - side, center, next);
                    let mut rdot = 0.0;
                    if let Some(miter) = self.miter_point(vec_rad, rdet, self.cur[side], vec_in, next[side], vec_out, &mut rdot) {
                        self.miter_to(side, miter, next[side]);
                    } else if self.join == Join::MiterOrBevel {
                        self.bevel_corner(side, next[side]);
                    } else {
                        self.limited_miter(self.cur[side], next[side], rdot, vec_rad, side);
                    }
                }
            }
        }
        self.vec_rad = vec_rad;
        self.vec_offset = vec_offset;
        self.prev_pt = center;
        self.prev_dir = vec_out;
    }

    fn bevel_corner(&mut self, side: usize, next: V) {
        self.cur[side] = next;
        if let Some(mut h) = self.hit() {
            h.polyline(side, &[next]);
        }
        self.rail[side].line_to(next);
    }

    // Native always emits two points for a miter (`PolylineWedge(side, 2, [miter, nextRail])`): the
    // miter apex then the next segment's rail start — never just the apex, even for open figures.
    fn miter_to(&mut self, side: usize, miter: V, next_start: V) {
        if let Some(mut h) = self.hit() {
            h.polyline(side, &[miter, next_start]);
        }
        self.rail[side].line_to(miter);
        self.rail[side].line_to(next_start);
        self.cur[side] = next_start;
    }

    #[allow(clippy::too_many_arguments)]
    fn miter_point(&self, vec_rad: V, rdet: f64, pt_in: V, vec_in: V, next: V, vec_out: V, rdot: &mut f64) -> Option<V> {
        *rdot = -dot(vec_rad, self.vec_rad);
        let pt = sub(next, pt_in);
        let rin_num = det(pt, vec_out);
        let rout_num = det(pt, vec_in);
        let done = if rdet < 0.0 {
            rin_num < 0.0 && rout_num > 0.0 && rdet < rin_num * FUZZ
        } else {
            rin_num > 0.0 && rout_num < 0.0 && rdet > rin_num * FUZZ
        };
        if !done {
            if *rdot < 0.0 {
                return Some(next); // smooth join, gloss over
            }
            return None;
        }
        let rin = rin_num / rdet;
        let miter = add(pt_in, scale(vec_in, rin));
        // Miter-limit test, reproducing native `GetMiterPoint`'s exact f64 operation order
        // (`(r²·ml·ml − (r²+r²))·r² >= rdot·(r²·ml·ml)`, with `r²·ml·ml` grouped as `(r²·ml)·ml`) —
        // not a pre-squared `(ml·r)²`, which rounds differently at the LSB.
        let m = (self.r_sq * self.nominal_miter_limit) * self.nominal_miter_limit;
        if (m - (self.r_sq + self.r_sq)) * self.r_sq >= *rdot * m {
            Some(miter)
        } else {
            None
        }
    }

    fn limited_miter(&mut self, _pt_in: V, next: V, rdot: f64, vec_rad_next: V, side: usize) {
        let mut rdenom = (self.r_sq + rdot) / 2.0;
        if rdenom > 0.0 {
            rdenom = rdenom.sqrt();
            let mut rratio = (self.r_sq - rdot) / 2.0;
            rratio = if rratio < 0.0 { 0.0 } else { rratio.sqrt() };
            rratio = self.miter_limit - rratio;
            if rratio < 0.0 {
                rratio = 0.0;
            }
            if rdenom > rratio * FUZZ {
                rratio /= rdenom;
                let v = self.pen_vec(self.vec_rad); // GetPenVector(m_vecRad)
                let w = self.pen_vec(vec_rad_next); // GetPenVector(vecRadNext)
                let p0 = add(self.cur[side], scale(v, rratio));
                let p1 = sub(next, scale(w, rratio));
                self.cur[side] = next;
                self.rail[side].line_to(p0);
                self.rail[side].line_to(p1);
                self.rail[side].line_to(next);
                if let Some(mut h) = self.hit() {
                    h.polyline(side, &[p0, p1, next]);
                }
            }
        }
    }

    fn do_180_miter(&mut self) {
        let vec = scale(self.pen_vec(self.vec_rad), self.nominal_miter_limit);
        if let Some(hit) = self.hit.take() {
            // Out to the miter limit on both rails, across, and back on the other rail's point.
            let [l, r] = self.cur;
            let mut h = hit.borrow_mut();
            h.quad_to(add(l, vec), add(r, vec));
            h.switch_sides();
            h.quad_to(r, l);
            drop(h);
            self.cur[LEFT] = add(self.cur[LEFT], vec);
            self.cur[RIGHT] = add(self.cur[RIGHT], vec);
            self.rail[LEFT].set_end(self.cur[LEFT]);
            self.rail[RIGHT].set_end(self.cur[RIGHT]);
            self.switch_sides();
            self.hit = Some(hit);
            return;
        }
        // Both rails run out from their segment ends to the miter limit, cross over, and come back to
        // the other rail's point; the pen's current points swap sides. `Widen` streams exactly this,
        // and a drawn stroke fills it (DrawGeometry = Fill(Widen) on the VM, the segment ends kept).
        let [l, r] = self.cur;
        self.rail[LEFT].line_to(add(l, vec));
        self.rail[RIGHT].line_to(add(r, vec));
        self.rail[LEFT].line_to(add(r, vec));
        self.rail[RIGHT].line_to(add(l, vec));
        self.rail[LEFT].line_to(r);
        self.rail[RIGHT].line_to(l);
        self.cur = [r, l];
    }

    fn switch_sides(&mut self) {
        if let Some(mut h) = self.hit() {
            h.switch_sides();
        }
        let left = self.cur[LEFT];
        self.rail[LEFT].line_to(self.cur[RIGHT]);
        self.rail[RIGHT].line_to(left);
    }

    // `CPen::RoundCorner`: round from `prev_rad` to `new_rad` on `rail[side]`. For the ordinary join
    // case callers pass `prev_rad = self.vec_rad` and `sweep = side`; `round_to`'s inner wedge needs
    // an explicit previous radius and sweep. `sweep` only decides the mid-arc
    // orientation in the >90° (two-bezier) case.
    #[allow(clippy::too_many_arguments)]
    fn round_corner_ex(&mut self, center: V, pt_in: V, pt_next: V, prev_rad: V, new_rad: V, side: usize, sweep: usize) {
        let r = dot(new_rad, prev_rad);
        if r > self.refine {
            // Flat turn — bevel.
            self.rail[side].line_to(pt_next);
            if let Some(mut h) = self.hit() {
                h.polyline(side, &[pt_next]);
            }
        } else if r >= 0.0 {
            let d = get_bezier_distance(r, self.r);
            let b1 = add(pt_in, scale(self.pen_vec(prev_rad), d));
            let b2 = sub(pt_next, scale(self.pen_vec(new_rad), d));
            self.rail[side].bezier_to(b1, b2, pt_next);
            if let Some(mut h) = self.hit() {
                h.curve(side, b1, b2, pt_next);
            }
        } else {
            // Two beziers through the arc midpoint (complex-arithmetic sqrt).
            let c2_real = new_rad[0] * prev_rad[0] - new_rad[1] * prev_rad[1];
            let c2_imag = new_rad[0] * prev_rad[1] + new_rad[1] * prev_rad[0];
            let l = self.r * self.r;
            let c_real = (0.5 * (l + c2_real)).abs().sqrt();
            let c_imag = (if c2_imag > 0.0 { 1.0 } else { -1.0 }) * (0.5 * (l - c2_real)).abs().sqrt();
            let mut vec_mid = [c_real, c_imag];
            let mut direction = turn_right(new_rad);
            if sweep == LEFT {
                direction = neg(direction);
            }
            if dot(vec_mid, direction) < 0.0 {
                vec_mid = neg(vec_mid);
            }
            let rad_dot_mid = dot(new_rad, vec_mid).abs();
            let d = get_bezier_distance(rad_dot_mid, self.r);
            let mut pt_mid = turn_right(vec_mid);
            if sweep == LEFT {
                pt_mid = neg(pt_mid);
            }
            pt_mid = add(self.pen_vec(pt_mid), center);
            let bez = add(pt_in, scale(self.pen_vec(prev_rad), d));
            let vec_bez_at_mid = scale(self.pen_vec(vec_mid), d);
            self.rail[side].bezier_to(bez, sub(pt_mid, vec_bez_at_mid), pt_mid);
            let bez2 = sub(pt_next, scale(self.pen_vec(new_rad), d));
            self.rail[side].bezier_to(add(pt_mid, vec_bez_at_mid), bez2, pt_next);
            if let Some(mut h) = self.hit() {
                h.curve(side, bez, sub(pt_mid, vec_bez_at_mid), pt_mid);
                h.curve(side, add(pt_mid, vec_bez_at_mid), bez2, pt_next);
            }
        }
        self.cur[side] = pt_next;
    }

    // Join the rails into a closed outline and flatten it to a device polygon.
    fn finish(&mut self) -> Vec<Point> {
        if let Some((pt, dir, cap)) = self.deferred_cap.take() {
            // The deferred start cap, a figure of its own: started at the figure's start with its cap,
            // closed flat.
            let hit = self.hit.take();
            let state = (self.cur, self.vec_rad, self.vec_offset, self.prev_pt);
            self.update_offset(dir);
            self.cur = [sub(pt, self.vec_offset), add(pt, self.vec_offset)];
            self.hit = hit;
            if let Some(mut h) = self.hit() {
                h.cur = [self.cur[RIGHT]; 2];
            }
            self.hit_cap(RAIL_START, pt, cap);
            self.hit_cap(RAIL_END, pt, Cap::Flat);
            (self.cur, self.vec_rad, self.vec_offset, self.prev_pt) = state;
        }
        let right = std::mem::take(&mut self.rail[RIGHT]);
        self.rail[LEFT].reverse_join(&right);
        if self.rail[LEFT].bez.is_empty() {
            return Vec::new();
        }
        self.emit(&self.rail[LEFT]);
        if let Some(cap) = self.start_cap_fig.take() {
            self.emit(&cap);
        }
        self.rail[LEFT].flatten_closed(self.rail_tol, self.target)
    }
}

const RAIL_START: usize = 0;
const RAIL_END: usize = 1;

/// Widen a flattened skeleton polyline into device-space fill polygons (nonzero winding). `pts` is
/// the device-space polyline; `closed` adds the wrap-around join. `width` is the device pen width.
/// Dashed figures are routed through `super::dash` (native `CDasher`) before reaching here.
pub(super) fn widen(pts: &[Point], closed: bool, pen: &PenShape, style: &StrokeStyle) -> Vec<Vec<Point>> {
    widen_one(&to_v(pts), closed, pen, style, style.start_cap, style.end_cap)
}

/// [`widen`] with `force_pts` — device vertices where the join is forced round
/// (`D2D1_PATH_SEGMENT_FORCE_ROUND_LINE_JOIN`), regardless of the stroke style.
pub(super) fn widen_flags(pts: &[V], closed: bool, pen: &PenShape, style: &StrokeStyle, force_pts: &[V], errored: &mut bool, run: &mut RunEnd) -> Vec<Vec<Point>> {
    // The widener skips a line under the length fuzz and keeps its current point, so a figure is all
    // skipped when every vertex lies within the fuzz of its start.
    let fuzz = pen.tol * SQ_LENGTH_FUZZ;
    if pts.len() >= 2 && pts[1..].iter().all(|&q| !(dot(sub(q, pts[0]), sub(q, pts[0])) >= fuzz)) {
        return collapsed_dot(pts[0], closed, pen, style, errored);
    }
    widen_one_flags(pts, closed, pen, style, style.start_cap, style.end_cap, force_pts, errored, true, run)
}

/// A figure whose every segment the widener skipped: it ends the figure unstarted, and the pen draws a
/// dot at the start facing the base's x axis with the figure's caps, round ones when it is closed
/// (failing with BAD_NUMBER when that axis does not normalize).
pub(super) fn collapsed_dot(start: V, closed: bool, pen: &PenShape, style: &StrokeStyle, errored: &mut bool) -> Vec<Vec<Point>> {
    let (start_cap, end_cap) = if closed { (Cap::Round, Cap::Round) } else { (style.start_cap, style.end_cap) };
    let dot = cap_figure(start, pen.dot_dir, pen, style, start_cap, end_cap, errored);
    if dot.len() >= 2 { vec![dot] } else { Vec::new() }
}

/// How a stroked run of a figure split at unstroked segments starts and ends (`super::RunCaps`), and
/// the run's first direction, which the wideners report back.
#[derive(Clone, Default)]
pub(super) struct RunEnd {
    /// The start cap is drawn inline (a run after a gap), not after the figure.
    pub(super) inline_start: bool,
    /// The run closes its figure: it turns the corner at its end onto this direction (forced round
    /// when set) and ends flat.
    pub(super) close: Option<(V, bool)>,
    pub(super) first_dir: Option<V>,
    /// Zero-length lines the pen draws at the end before its end cap (a dash the dasher extends to a
    /// vertex and then ends there).
    pub(super) end_repeats: usize,
    /// Vertices whose corner comes after a segment the widener skipped (a miter limited to 1).
    pub(super) skip_pts: Vec<V>,
}

/// True when `p` coincides (within the segment fuzz) with a forced-round-join vertex.
fn is_force_pt(p: V, pts: &[V]) -> bool {
    let fuzz = TOLERANCE * SQ_LENGTH_FUZZ;
    pts.iter().any(|&q| {
        let d = sub(p, q);
        dot(d, d) < fuzz
    })
}

/// Widen a zero-length dash (native `CSimplePen` StartFigure+AcceptLinePoint+EndStrokeOpen at one
/// point): round cap → full disc, square → square, triangle → two triangles, flat → nothing. `dir`
/// orients the caps. Returns a single outline (empty for a flat-capped dot). Used by `super::dash`.
/// A figure that is only caps: the pen starts at `pt` along `dir` with `start_cap` and ends the stroke
/// there with `end_cap`, on the rails it started with (no segment between).
pub(super) fn cap_figure(pt: [f64; 2], dir: [f64; 2], pen: &PenShape, style: &StrokeStyle, start_cap: Cap, end_cap: Cap, errored: &mut bool) -> Vec<Point> {
    let mut p = Pen::new(pen, style);
    p.start_figure(pt, dir, start_cap);
    p.base_cap(RAIL_END, pt, end_cap);
    *errored |= p.errored;
    p.finish()
}

pub(super) fn dash_dot(pt: [f64; 2], dir: [f64; 2], pen: &PenShape, style: &StrokeStyle, start_cap: Cap, end_cap: Cap, errored: &mut bool) -> Vec<Point> {
    let mut p = Pen::new(pen, style);
    p.start_figure(pt, dir, start_cap);
    p.accept_line(pt);
    p.base_cap(RAIL_END, pt, end_cap);
    *errored |= p.errored;
    p.finish()
}

/// Widen one dash lying along a flattened curve skeleton: drive the pen with `AcceptCurvePoint`
/// per probed point (native `CDasher` → `CPen::AcceptCurvePoint`), smoothly offsetting along each
/// point's analytic tangent — not widening the chord polyline with a join at every vertex. `pts` is
/// `(point, tangent, is_last)` in order, `pts[0]` the dash start (its tangent orients the start
/// cap). A single point degenerates to a dash dot. ponytail: a curve dash that spans an edge corner
/// is not handled (no scene dashes across a curve↔edge junction); add `do_corner` if one appears.
pub(super) fn widen_curve_dash(pts: &[([f64; 2], super::dash::DashPt)], start_cap: Cap, end_cap: Cap, pen: &PenShape, style: &StrokeStyle, force_pts: &[V], errored: &mut bool) -> Vec<Point> {
    if pts.len() < 2 {
        return dash_dot(pts[0].0, pts[0].1.tan, pen, style, start_cap, end_cap, errored);
    }
    let mut p = Pen::new(pen, style);
    p.start_figure(pts[0].0, pts[0].1.tan, start_cap);
    // A dash that opens on a zero-length edge reaches the corner before any other point: the pen
    // turns it at the start vertex.
    if let Some((c, vin, vout)) = pts[0].1.corner {
        p.corner_flags(c, vin, vout, is_force_pt(c, force_pts), pts[0].1.skip);
    }
    for &(pt, d) in &pts[1..] {
        if !d.accept {
        } else if d.line {
            p.accept_line(pt);
        } else {
            p.accept_curve_point(pt, d.tan, d.last);
        }
        if let Some((c, vin, vout)) = d.corner {
            // r23 28061: a dash through the cusp of a zero-width ellipse turns with a round join.
            p.corner_flags(c, vin, vout, is_force_pt(c, force_pts), d.skip);
        }
    }
    p.base_cap(RAIL_END, pts.last().unwrap().0, end_cap);
    *errored |= p.errored;
    p.finish()
}

/// `CBezierFlattener<double>` (`CCubicSegment::Widen`): the flattened points of a cubic for curve
/// widening, each with its analytic tangent, the last flagged `is_last` — the stream native feeds
/// the dasher/pen. Excludes `p0`; the returned start tangent is `GetFirstTangent` (`first_tangent_f64`).
pub(super) fn flatten_curve_tan(p0: [f64; 2], p1: [f64; 2], p2: [f64; 2], p3: [f64; 2], curve_err: f64) -> (V, Vec<(V, V)>) {
    let mut out = Vec::new();
    flatten_cubic_tan(p0, p1, p2, p3, &mut out, curve_err);
    (first_tangent_f64(p0, p1, p2, p3, cubic_fuzz(curve_err / 6.0)), out)
}

fn to_v(pts: &[Point]) -> Vec<V> {
    pts.iter().map(|p| [p[0] as f64, p[1] as f64]).collect()
}

// Widen a single open/closed polyline (no dashes). Open figures yield one outline (rails joined by
// caps); closed figures yield the outer and inner contours separately (filled under nonzero winding).
// `pub(super)` so `super::dash` widens each dash sub-stroke through the same pen.
pub(super) fn widen_one(pts: &[V], closed: bool, pen: &PenShape, style: &StrokeStyle, start_cap: Cap, end_cap: Cap) -> Vec<Vec<Point>> {
    widen_one_flags(pts, closed, pen, style, start_cap, end_cap, &[], &mut false, false, &mut RunEnd::default())
}

/// [`widen_one`] with `force_pts` — device vertices where the join is forced round. `errored` is set
/// when the pen hit a BAD_NUMBER (a non-finite/zero tangent native rejects): the caller voids the
/// whole stroke ([`super::Geometry::widen_inner`]).
#[allow(clippy::too_many_arguments)]
/// `split_cap`: a plain stroke's figure (not a dash), whose start cap Widen streams on its own.
#[allow(clippy::too_many_arguments)]
pub(super) fn widen_one_flags(pts: &[V], closed: bool, pen: &PenShape, style: &StrokeStyle, start_cap: Cap, end_cap: Cap, force_pts: &[V], errored: &mut bool, split_cap: bool, run: &mut RunEnd) -> Vec<Vec<Point>> {
    // Drop consecutive duplicate / near-zero segments (GetFirstTangent fuzz).
    let fuzz = pen.tol * SQ_LENGTH_FUZZ;
    let mut p: Vec<V> = Vec::with_capacity(pts.len());
    // Per kept point, whether a dropped (too short) segment follows it: the corner there is then a
    // miter limited to 1.
    let mut skip_after: Vec<bool> = Vec::with_capacity(pts.len());
    for &q in pts {
        if p.last().is_none_or(|&l| dot(sub(q, l), sub(q, l)) >= fuzz) {
            p.push(q);
            skip_after.push(false);
        } else if let Some(s) = skip_after.last_mut() {
            *s = true;
        }
    }
    // A closing point at the start closes the loop. One merely near it ends the last segment, and the
    // closing edge from it, too short, is skipped: the closing corner turns at the start after a skip.
    let mut close_skip = false;
    let mut short_close = false;
    if closed && p.len() > 1 && dot(sub(*p.last().unwrap(), p[0]), sub(*p.last().unwrap(), p[0])) < fuzz {
        if *p.last().unwrap() != p[0] && p.len() > 2 {
            short_close = true;
        } else {
            close_skip = skip_after.pop().unwrap_or(false);
            p.pop();
            skip_after.truncate(p.len());
        }
    }
    // ponytail: the rule is the widener's; a dash's polyline keeps its joins (the dasher's corners are not ported).
    if !split_cap {
        skip_after.iter_mut().for_each(|s| *s = false);
        close_skip = false;
        short_close = false;
    }
    if p.len() < 2 {
        return Vec::new(); // degenerate single point (round-dot cap) — not exercised
    }
    let n = p.len();
    let dir = |a: V, b: V| sub(b, a);
    let mut s = Pen::new(pen, style);
    s.split_start_cap = split_cap && !run.inline_start;
    run.first_dir = Some(dir(p[0], p[1]));

    if closed && short_close {
        s.start_closed(p[0], dir(p[0], p[1]));
        s.accept_line(p[1]);
        let mut vec_in = dir(p[0], p[1]);
        for k in 1..n - 1 {
            let d = dir(p[k], p[k + 1]);
            s.corner_flags(p[k], vec_in, d, is_force_pt(p[k], force_pts), skip_after[k] || is_force_pt(p[k], &run.skip_pts));
            s.accept_line(p[k + 1]);
            vec_in = d;
        }
        s.corner_flags(p[0], vec_in, dir(p[0], p[1]), is_force_pt(p[0], force_pts), true);
        *errored |= s.errored;
        return s.finish_two();
    }
    if closed {
        s.start_closed(p[0], dir(p[0], p[1]));
        s.accept_line(p[1 % n]);
        let mut vec_in = dir(p[0], p[1]);
        for k in 1..n {
            let a = p[k];
            let b = p[(k + 1) % n];
            let d = dir(a, b);
            s.corner_flags(a, vec_in, d, is_force_pt(a, force_pts), skip_after[k]);
            s.accept_line(b);
            vec_in = d;
        }
        // The closing corner reads the widener's skip flags: one set by a skipped segment and cleared
        // by the next widened one (so a skip before a widened closing edge has already turned the
        // corner at the last vertex), and one set by a skip before the figure's first segment.
        s.corner_flags(p[0], vec_in, dir(p[0], p[1]), is_force_pt(p[0], force_pts), close_skip || skip_after[0]);
        *errored |= s.errored;
        return s.finish_two();
    }
    s.start_figure(p[0], dir(p[0], p[1]), start_cap);
    s.accept_line(p[1]);
    let mut vec_in = dir(p[0], p[1]);
    for k in 1..n - 1 {
        let d = dir(p[k], p[k + 1]);
        s.corner_flags(p[k], vec_in, d, is_force_pt(p[k], force_pts), skip_after[k] || is_force_pt(p[k], &run.skip_pts));
        s.accept_line(p[k + 1]);
        vec_in = d;
    }
    for _ in 0..run.end_repeats {
        s.accept_line(p[n - 1]);
    }
    if let Some((d, round)) = run.close {
        s.corner_flags(p[n - 1], vec_in, d, round, skip_after[n - 1]);
        s.base_cap(RAIL_END, p[n - 1], Cap::Flat);
    } else {
        s.base_cap(RAIL_END, p[n - 1], end_cap);
    }
    *errored |= s.errored;
    let poly = s.finish();
    if poly.len() >= 2 { vec![poly] } else { Vec::new() }
}

/// Native's fast axis-aligned-rectangle widen: the `DrawRectangle`/`DrawGeometry`
/// fast path for an axis-aligned rectangle stroke under a **non-uniform** device matrix. Instead of
/// the device-space elliptical pen it widens the rect into **local-space** frame rects — the four
/// corners as local squares plus the four edges (or one rect when the pen is wider than the rect) —
/// which are then transformed and filled through the ordinary 8×8 coverage path. The square corners
/// become sheared parallelograms under the transform, so this differs from the pen's device-space
/// miter at the LSB; reproducing it makes a sheared thin rect stroke (`08-affine-dpi`) byte-exact.
/// The gate mirrors native's fast-rectangle-widen eligibility test: a
/// solid (non-dashed, the caller already excludes dashes) miter / miter-or-bevel join with
/// `miterLimit ≥ √2` and a NORMAL transform type. Returns the local frame rects (local space).
fn fast_widen_rects(rect: [f32; 4], width: f32) -> Vec<[f32; 4]> {
    let [l, t, r, b] = rect;
    let h = width.abs() * 0.5;
    // Pen wider than the rect in either axis ⇒ a single filled rect (native's "wider than extent" branch).
    if width.abs() > r - l || width.abs() > b - t {
        return vec![[l - h, t - h, r + h, b + h]];
    }
    vec![
        [l - h, t - h, l + h, t + h], // top-left corner
        [l + h, t - h, r - h, t + h], // top edge
        [r - h, t - h, r + h, t + h], // top-right corner
        [l - h, t + h, l + h, b - h], // left edge
        [r - h, t + h, r + h, b - h], // right edge
        [l - h, b - h, l + h, b + h], // bottom-left corner
        [l + h, b - h, r - h, b + h], // bottom edge
        [r - h, b - h, r + h, b + h], // bottom-right corner
    ]
}

/// Native's rectangle-to-sink widen: widen an axis-aligned rectangle (`rect`, local
/// space, pen `width` in local units) under device matrix `m`. A non-uniform matrix with a plain
/// mitered stroke takes native's fast axis-aligned-rectangle frame (above); otherwise the
/// device-space pen is used — for a uniform matrix the pen is circular and the mitered annulus equals
/// the frame, matching the rectangle-fill tessellator path the uniform rect scenes already pass. Pen
/// path: native feeds the pen the device corners (via `TransformPoints`) and the edge directions
/// computed as `edge_length · matrix_row` — NOT corner differences, whose f32 rounding would differ.
/// Non-dashed; the dashed/curve paths keep the general widener.
pub(super) fn widen_rect(rect: [f32; 4], width: f32, m: &Matrix, pen: &PenShape, style: &StrokeStyle, tol: f32) -> Vec<Vec<Point>> {
    let uniform = m[0] == m[3] && m[1] == -m[2];
    let join_ok = matches!(style.join, Join::Miter | Join::MiterOrBevel);
    if !uniform && join_ok && style.miter_limit >= std::f32::consts::SQRT_2 && style.transform_type == TransformType::Normal {
        // Transform each local frame rect's corners by the f32 device matrix (native maps the rect
        // vertices in f32 here, unlike the pen path's f64 corners) and fill under nonzero winding.
        let ap = |x: f32, y: f32| [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]];
        return fast_widen_rects(rect, width)
            .into_iter()
            .map(|[l, t, r, b]| vec![ap(l, t), ap(r, t), ap(r, b), ap(l, b)])
            .collect();
    }
    widen_rect_pen(rect, m, pen, style, tol)
}

/// The rectangle widener proper: native `TransformPoints` maps the rect corners into **double**
/// `MilPoint2D` (not the f32 path the fill uses), so the thin-stroke rails do not inherit the corners'
/// f32 rounding; the edge directions are the transformed columns, a degenerate one replaced
/// ([`rect_widener_frame`]).
pub(super) fn widen_rect_pen(rect: [f32; 4], m: &Matrix, pen: &PenShape, style: &StrokeStyle, tol: f32) -> Vec<Vec<Point>> {
    let mut s = rect_pen(rect, Some(m), pen, style, tol);
    // A rectangle's realization widens into two separate offset rectangles, not the bridged contour.
    s.bridge = pen.rail_tol.is_none();
    s.finish_two()
}

/// The rectangle widener's pen: the closed walk of [`rect_widener_frame`], up to its closing corner.
fn rect_pen(rect: [f32; 4], m: Option<&Matrix>, pen: &PenShape, style: &StrokeStyle, tol: f32) -> Pen {
    let (corners, dirs) = rect_widener_frame(rect, m, tol);
    let mut s = Pen::new(pen, style);
    s.start_closed(corners[0], dirs[0]);
    s.accept_line(corners[1]);
    for k in 1..4 {
        s.do_corner(corners[k], dirs[k - 1], dirs[k]);
        s.accept_line(corners[(k + 1) % 4]);
    }
    s.do_corner(corners[0], dirs[3], dirs[0]);
    s
}

/// Widen a figure given as device-space primitives (lines + cubics), reproducing native's
/// curve-skeleton widening: a cubic is flattened by the **double-precision** `CBezierFlattener`
/// and each point is offset perpendicular to its analytic tangent (`CPen::AcceptCurvePoint`), not
/// pre-flattened to a chord polyline. `prims` includes the closing edge for closed figures (the
/// last prim ends at `start`); `closed` adds the wrap-around join. Pen is circular (radius `r`).
pub(super) fn widen_prims(start: Point, prims: &[Prim], closed: bool, pen: &PenShape, style: &StrokeStyle) -> Vec<Vec<Point>> {
    widen_prims_flags(start, prims, None, closed, pen, style, &[], TOLERANCE * 6.0, &mut false, &mut RunEnd::default())
}

/// `widen_prims` with `force_pts` — device vertices (`D2D1_PATH_SEGMENT_FORCE_ROUND_LINE_JOIN`) where
/// the join is forced round regardless of the stroke style. A prim whose endpoint is one of these
/// points carries a round join after it.
#[allow(clippy::too_many_arguments)]
pub(super) fn widen_prims_flags(start: Point, prims: &[Prim], ends64: Option<&super::LineEnds64>, closed: bool, pen: &PenShape, style: &StrokeStyle, force_pts: &[V], curve_err: f64, errored: &mut bool, run: &mut RunEnd) -> Vec<Vec<Point>> {
    // Drop degenerate (zero-length) line segments — a max-radius rounded rect has zero-length
    // straight edges; native's figure walk skips them (GetFirstTangent fuzz).
    let fuzz = pen.tol * SQ_LENGTH_FUZZ;
    let mut segs: Vec<(V, V, V, bool, [V; 4], bool)> = Vec::new(); // (start, start_tan, end_tan, is_cubic, pts, force_round_after)
    // Each segment starts at the walk's current point: a skipped line does not advance it, so the next
    // segment starts where the last kept one ended (r23 28462: an ellipse's 1-ULP apex line).
    // Line ends (and the start) as the widener maps them, in f64, when the figure's local form is known.
    let start = ends64.map_or(v(start), |e| e.start);
    let line_end = |i: usize, b: Point| ends64.and_then(|e| e.lines.get(i).copied().flatten()).unwrap_or(v(b));
    let mut cur = start;
    // A skipped segment makes the next corner a miter limited to 1: per kept segment, whether one
    // follows it, and whether one leads the figure.
    let (mut skip_after, mut lead_skip): (Vec<bool>, bool) = (Vec::new(), false);
    let mut skipped = |skip_after: &mut Vec<bool>| match skip_after.last_mut() {
        Some(s) => *s = true,
        None => lead_skip = true,
    };
    for (i, pr) in prims.iter().enumerate() {
        let end = match *pr {
            Prim::Line(_, b) => line_end(i, b),
            Prim::Cubic(_, _, _, p3) => v(p3),
        };
        let fr = is_force_pt(end, force_pts);
        match *pr {
            Prim::Line(_, b) => {
                let (a, b) = (cur, line_end(i, b));
                let d = sub(b, a);
                if dot(d, d) >= fuzz {
                    segs.push((a, d, d, false, [a, b, b, b], fr));
                    skip_after.push(false);
                    cur = b;
                } else {
                    skipped(&mut skip_after);
                }
            }
            Prim::Cubic(_, p1, p2, p3) => {
                let p = [cur, v(p1), v(p2), v(p3)];
                let fuzz = cubic_fuzz(curve_err / 6.0);
                let st = first_tangent_f64(p[0], p[1], p[2], p[3], fuzz);
                // A curve with no first direction (a zero-radius corner) is skipped like a short line,
                // leaving the current point.
                if fuzz >= dot(st, st) {
                    skipped(&mut skip_after);
                    continue;
                }
                cur = p[3];
                let et = last_tangent_f64(p[0], p[1], p[2], p[3], fuzz);
                segs.push((p[0], st, et, true, p, fr));
                skip_after.push(false);
            }
        }
    }
    if segs.is_empty() && !prims.is_empty() {
        return collapsed_dot(start, closed, pen, style, errored);
    }
    let mut sp = Pen::new(pen, style);
    sp.split_start_cap = !run.inline_start;
    run.first_dir = Some(segs[0].1);
    let n = segs.len();
    if closed {
        sp.start_closed(start, segs[0].1);
    } else {
        sp.start_figure(start, segs[0].1, style.start_cap);
    }
    for i in 0..n {
        let (s, st, et, is_cubic, pts, _fr) = segs[i];
        if i > 0 {
            // The join after prim i-1 is round when that prim carried FORCE_ROUND_LINE_JOIN.
            sp.corner_flags(s, segs[i - 1].2, st, segs[i - 1].5, skip_after[i - 1]);
        }
        if is_cubic && pen.skips_curve(pts[0], pts[1], pts[2], pts[3]) {
            sp.accept_curve_point(pts[3], et, true);
        } else if is_cubic {
            let mut pts_tan = Vec::new();
            flatten_cubic_tan(pts[0], pts[1], pts[2], pts[3], &mut pts_tan, curve_err);
            let last = pts_tan.len() - 1;
            for (j, (p, t)) in pts_tan.into_iter().enumerate() {
                // The last flattened point carries the segment's authoritative end tangent
                // (`GetLastTangent`, = the tangent the following join uses), not `flatten_cubic_tan`'s
                // own fallback: the two use different degeneracy gates and disagree for an eccentric
                // curve whose P3−P2 is tiny but non-zero (rounded-rect corner with ry ≪ pen, fuzz
                // 5192). The mismatch left the outer rail short of the true corner end, so the next
                // (flat) join kept the stale point and the rail sheared diagonally across the shape.
                let t = if j == last { et } else { t };
                sp.accept_curve_point(p, t, j == last);
            }
        } else {
            sp.accept_line(pts[1]);
        }
    }
    if closed {
        // The closing join follows the last prim; honour its FORCE_ROUND_LINE_JOIN flag.
        sp.corner_flags(start, segs[n - 1].2, segs[0].1, segs[n - 1].5, skip_after[n - 1] || lead_skip);
        *errored |= sp.errored;
        return sp.finish_two();
    }
    let (_, _, et, is_cubic, pts, _) = segs[n - 1];
    let end = if is_cubic { pts[3] } else { pts[1] };
    if let Some((d, round)) = run.close {
        sp.corner_flags(end, et, d, round, skip_after[n - 1]);
        sp.base_cap(RAIL_END, end, Cap::Flat);
    } else {
        sp.base_cap(RAIL_END, end, style.end_cap);
    }
    *errored |= sp.errored;
    let poly = sp.finish();
    if poly.len() >= 2 { vec![poly] } else { Vec::new() }
}

fn v(p: Point) -> V {
    [p[0] as f64, p[1] as f64]
}

/// The rectangle widener's walk of the ordered rect `[l, t, r, b]`: its four corners (mapped through
/// `m` in f64) and the direction it hands the pen along each edge. With no transform those are the
/// raw width and height, so a zero or non-finite side reaches the pen as is. Under `m` they are the
/// transformed width and height columns, and a column under the length fuzz `min(tol²·1e-4, 1)` (a
/// NaN one included) takes the other's perpendicular, or both the axes when both are.
pub(super) fn rect_widener_frame([l, t, r, b]: [f32; 4], m: Option<&Matrix>, tol: f32) -> ([V; 4], [V; 4]) {
    let (w, h) = ((r - l) as f64, (b - t) as f64);
    let (corners, xe, ye) = match m {
        None => ([[l, t], [r, t], [r, b], [l, b]].map(v), [w, 0.0], [0.0, h]),
        Some(m) => {
            let m = m.map(|x| x as f64);
            let tp = |x: f32, y: f32| { let (x, y) = (x as f64, y as f64); [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]] };
            let mut xe = [w * m[0] + m[2] * 0.0, w * m[1] + m[3] * 0.0];
            let mut ye = [h * m[2] + m[0] * 0.0, h * m[3] + m[1] * 0.0];
            let fuzz = (tol * tol * 0.000_099_999_997f32).min(1.0) as f64;
            // An unordered compare counts a NaN column as degenerate.
            let degenerate = |d: V| !(dot(d, d) >= fuzz);
            match (degenerate(xe), degenerate(ye)) {
                (false, false) => {}
                (false, true) => ye = [-xe[1], xe[0]],
                (true, false) => xe = [ye[1], -ye[0]],
                (true, true) => (xe, ye) = ([1.0, 0.0], [0.0, 1.0]),
            }
            ([tp(l, t), tp(r, t), tp(r, b), tp(l, b)], xe, ye)
        }
    };
    (corners, [xe, ye, neg(xe), neg(ye)])
}

/// The rectangle widener as a query runs it ([`rect_pen`], ended flat); whether its pen fails with
/// BAD_NUMBER.
pub(super) fn rect_walk(rect: [f32; 4], m: Option<&Matrix>, pen: &PenShape, style: &StrokeStyle, tol: f32) -> bool {
    let mut s = rect_pen(rect, m, pen, style, tol);
    s.finish_two();
    s.errored
}

/// Native's widener streaming figures to the pen as a query runs it (`CWidener::DoSegment` and its
/// figure end), returning whether the pen failed with `D2DERR_BAD_NUMBER`. `figs` are `(start, prims
/// with the closing edge, closed)` before `m`, the query transform (`None` for identity, which native
/// leaves off). A line's end is mapped in f64 and a curve's points in f32, both from the f64 current
/// point. A segment shorter than the length fuzz is skipped and leaves the current point; a NaN
/// direction is kept (the compare is unordered), and a curve's first direction falls past a NaN leg to
/// its next control point. The corner after a skipped segment is a miter limited to 1 unless round.
/// An open figure ends in its end cap, then its start cap as a figure of its own; a closed one turns
/// the corner at its start and ends flat. A figure whose every segment is skipped is a dot facing the
/// base's x axis, with round caps when closed.
pub(super) fn pen_walk(figs: &[(Point, Vec<Prim>, bool)], m: Option<&Matrix>, pen: &PenShape, style: &StrokeStyle, tol: f64) -> bool {
    let line = |p: Point| -> V {
        let Some(m) = m else { return v(p) };
        let (x, y) = (p[0] as f64, p[1] as f64);
        [m[2] as f64 * y + m[0] as f64 * x + m[4] as f64, m[3] as f64 * y + m[1] as f64 * x + m[5] as f64]
    };
    let curve = |p: Point| -> V {
        let Some(m) = m else { return v(p) };
        v([(m[2] * p[1] + m[0] * p[0]) + m[4], (m[3] * p[1] + m[1] * p[0]) + m[5]])
    };
    let (line_fuzz, curve_fuzz) = (tol * SQ_LENGTH_FUZZ, cubic_fuzz(tol));
    for (start, prims, closed) in figs {
        let mut s = Pen::new(pen, style);
        s.split_start_cap = true;
        let start = line(*start);
        let mut cur = start;
        let (mut prev, mut first_dir): (Option<V>, V) = (None, [0.0; 2]);
        let (mut skipped, mut lead_skip) = (false, false);
        for pr in prims {
            let seg = match *pr {
                Prim::Line(_, e) => {
                    let p = line(e);
                    let t = sub(p, cur);
                    (!(line_fuzz > dot(t, t))).then_some((t, t, p, None))
                }
                Prim::Cubic(_, c1, c2, e) => {
                    let (p1, p2, p3) = (curve(c1), curve(c2), curve(e));
                    let t = first_tangent_f64(cur, p1, p2, p3, curve_fuzz);
                    (!(curve_fuzz >= dot(t, t))).then(|| (t, last_tangent_f64(cur, p1, p2, p3, curve_fuzz), p3, Some((p1, p2))))
                }
            };
            let Some((first, last, end, controls)) = seg else {
                skipped = true;
                lead_skip |= prev.is_none();
                continue;
            };
            match prev {
                None if *closed => s.start_closed(cur, first),
                None => s.start_figure(cur, first, style.start_cap),
                Some(vin) => s.corner(cur, vin, first, skipped),
            }
            if prev.is_none() {
                first_dir = first;
            }
            skipped = false;
            match controls {
                None => s.accept_line(end),
                Some((p1, p2)) => {
                    let mut pts = Vec::new();
                    flatten_cubic_tan(cur, p1, p2, end, &mut pts, tol * 6.0);
                    let n = pts.len();
                    for (j, (p, t)) in pts.into_iter().enumerate() {
                        s.accept_curve_point(p, if j + 1 == n { last } else { t }, j + 1 == n);
                    }
                }
            }
            if s.errored {
                return true;
            }
            prev = Some(last);
            cur = end;
        }
        match prev {
            Some(vin) if *closed => {
                s.corner(start, vin, first_dir, skipped || lead_skip);
                s.finish_two();
            }
            Some(_) => {
                s.base_cap(RAIL_END, cur, style.end_cap);
                s.finish();
            }
            None if !prims.is_empty() => {
                let caps = if *closed { (Cap::Round, Cap::Round) } else { (style.start_cap, style.end_cap) };
                let mut d = Pen::new(pen, style);
                d.start_figure(start, pen.dot_dir, caps.0);
                d.base_cap(RAIL_END, start, caps.1);
                s.errored |= d.errored;
            }
            None => {}
        }
        if s.errored {
            return true;
        }
    }
    false
}

/// `GetFirstTangent` (double): the cubic's initial tangent `P1−P0`, falling back to `P2−P0` /
/// `P3−P0` when below the squared fuzz.
/// The widener's squared-length fuzz for a cubic's tangents at flattening tolerance `tol`
/// (`curve_err / 6`): `tol²·SQ_LENGTH_FUZZ`. A line segment's fuzz is `tol·SQ_LENGTH_FUZZ`.
pub(super) fn cubic_fuzz(tol: f64) -> f64 {
    tol * tol * SQ_LENGTH_FUZZ
}

/// `GetLastTangent`: the cubic's end direction P3−P2, falling back to P3−P1 / P3−P0 when its squared
/// length is within an eighth of the cubic `fuzz`. A thin ellipse's apex leg `P3−P2 ≈ [0.002, 0]` is
/// well above it and stays horizontal (fuzz 22144); a collapsed rounded-rect corner's coincident P2≡P3
/// (5192) and a 1e-20 x-radius corner's leg (r22 27044) fall back to the corner's real direction.
/// A NaN leg is kept, as native's unordered compare keeps it (the first tangent instead falls back).
fn last_tangent_f64(p0: V, p1: V, p2: V, p3: V, fuzz: f64) -> V {
    let gate = fuzz * 0.125;
    for c in [p2, p1] {
        let t = sub(p3, c);
        if !(dot(t, t) <= gate) {
            return t;
        }
    }
    sub(p3, p0)
}

fn first_tangent_f64(p0: V, p1: V, p2: V, p3: V, fuzz: f64) -> V {
    for c in [p1, p2] {
        let t = sub(c, p0);
        if dot(t, t) > fuzz {
            return t;
        }
    }
    sub(p3, p0)
}

// Native `CBezierFlattener<double,GpPointR>` (the widening flattener): adaptive forward-differencing
// in f64, appending (point, analytic-tangent) per output point EXCLUDING p0, INCLUDING p3. Same
// algorithm as the f32 query flattener (`super::flatten`) but double precision, matching
// `CCubicSegment::Widen`. Tangent = `6·e0 − e1 − 2·e2` (interior) / `GetLastTangent` (endpoint).
/// `err_bound` = the flattener's per-step error ceiling (default [`TOLERANCE`]·6, on the device-space
/// skeleton).
pub(super) fn flatten_cubic_tan(p0: V, p1: V, p2: V, p3: V, out: &mut Vec<(V, V)>, err_bound: f64) {
    let fuzz = err_bound * 0.25;
    let mut cur = p0;
    let mut e0 = sub(p3, p0);
    let mut e1 = [6.0 * (p1[0] - 2.0 * p2[0] + p3[0]), 6.0 * (p1[1] - 2.0 * p2[1] + p3[1])];
    let mut e2 = [6.0 * (p0[0] - 2.0 * p1[0] + p2[0]), 6.0 * (p0[1] - 2.0 * p1[1] + p2[1])];
    let mut csteps = 1i32;
    let mut stepsize = 1.0f64;
    // `maxsd`: the second operand when either is NaN (not f64::max, which drops the NaN), so a NaN y
    // difference makes the pair NaN, and native's unordered compares then take it as flat.
    let ma = |p: V| { let (a, b) = (p[0].abs(), p[1].abs()); if a > b { a } else { b } };
    let halve = |e0: &mut V, e1: &mut V, e2: &mut V, cs: &mut i32, ss: &mut f64| {
        let ne1 = [(e1[0] + e2[0]) * 0.125, (e1[1] + e2[1]) * 0.125];
        *e0 = [(e0[0] - ne1[0]) * 0.5, (e0[1] - ne1[1]) * 0.5];
        *e2 = [e2[0] * 0.25, e2[1] * 0.25];
        *e1 = ne1;
        *cs *= 2;
        *ss *= 0.5;
    };
    while (ma(e2) > err_bound || ma(e1) > err_bound) && stepsize > 0.001 {
        halve(&mut e0, &mut e1, &mut e2, &mut csteps, &mut stepsize);
    }
    while csteps > 1 {
        cur = [cur[0] + e0[0], cur[1] + e0[1]];
        let o1 = e1;
        e0 = [e0[0] + e1[0], e0[1] + e1[1]];
        e1 = [2.0 * o1[0] - e2[0], 2.0 * o1[1] - e2[1]];
        e2 = o1;
        csteps -= 1;
        out.push((cur, [(e0[0] * 6.0 - e1[0]) - (e2[0] + e2[0]), (e0[1] * 6.0 - e1[1]) - (e2[1] + e2[1])]));
        if !(ma(e1) > err_bound) || stepsize <= 0.001 {
            while csteps & 1 == 0 {
                let d = [2.0 * e1[0] - e2[0], 2.0 * e1[1] - e2[1]];
                if !super::flatten::step_doubles(e2, d, fuzz) {
                    break;
                }
                e0 = [2.0 * e0[0] + e1[0], 2.0 * e0[1] + e1[1]];
                e2 = [e2[0] * 4.0, e2[1] * 4.0];
                e1 = [d[0] * 4.0, d[1] * 4.0];
                stepsize *= 2.0;
                csteps /= 2;
            }
        } else {
            halve(&mut e0, &mut e1, &mut e2, &mut csteps, &mut stepsize);
        }
    }
    // Endpoint tangent = GetLastTangent, gated by the flattener's squared-length fuzz (tol²·1e-4, tol =
    // err_bound/6), not its quarter tolerance.
    out.push((p3, last_tangent_f64(p0, p1, p2, p3, cubic_fuzz(err_bound / 6.0))));
}

#[cfg(test)]
mod tests {
    use super::*;

    // The tangent gates fall back only for a leg within the cubic fuzz: a 1e-20 corner leg takes the
    // corner's direction, a thin ellipse's 0.002 apex leg keeps its own.
    #[test]
    fn cubic_tangent_gates() {
        let fuzz = cubic_fuzz(TOLERANCE);
        let corner = [[16.3, -0.18], [16.3 + 5.5e-21, -0.18], [16.3, 13.2], [16.3, 29.8]];
        assert_eq!(first_tangent_f64(corner[0], corner[1], corner[2], corner[3], fuzz)[0].abs(), 0.0);
        let down = [[16.3, 29.8], [16.3, 16.4], [16.3 - 5.5e-21, -0.18], [16.3 - 1e-20, -0.18]];
        assert!(last_tangent_f64(down[0], down[1], down[2], down[3], fuzz)[1] < 0.0);
        let apex = [[0.0, 0.0], [0.0, 1.0], [-0.002, 2.0], [0.0, 2.0]];
        assert_eq!(last_tangent_f64(apex[0], apex[1], apex[2], apex[3], fuzz), [0.002, 0.0]);
    }

    // A cubic with an infinite control widens through native's own arithmetic: under an identity
    // transform it flattens to NaN points and voids the stroke (r19 24419); under a scale, `0·inf`
    // turns the control's y NaN, the curve tests flat and widens as a line to its end (r13 18852).
    #[test]
    fn inf_control_cubic_depends_on_the_transform() {
        use crate::geometry::{Figure, FillMode, Geometry, Segment};
        let fig = Figure {
            start: [6.98, 13.3],
            filled: true,
            closed: false,
            segments: vec![Segment::Cubic([f32::INFINITY, 24.31], [f32::INFINITY, 35.61], [0.0, 46.62]), Segment::Line([39.62, 76.16])],
            seg_flags: Vec::new(),
        };
        let g = Geometry::Path { fill: FillMode::Winding, figures: vec![fig] };
        let st = StrokeStyle::default();
        assert!(g.widen(9.42, &st, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [1.0, 1.0], 0.25).is_empty());
        assert!(!g.widen(9.42, &st, &[1.25, 0.0, 0.0, 1.25, 0.0, 0.0], [1.25, 1.25], 0.25).is_empty());
    }

    // A realization widens a closed figure into native's ONE contour: outer rail, flat cap to the inner
    // rail's current point (the closing corner set it to the first segment's offset), inner rail
    // reversed — point for point as native's Widen of the closed path square.
    #[test]
    fn closed_realization_widen_is_one_bridged_contour() {
        use crate::geometry::{Figure, FillMode, Geometry, Segment};
        let fig = Figure {
            start: [12.0, 10.0],
            filled: true,
            closed: true,
            segments: vec![Segment::Line([84.0, 10.0]), Segment::Line([84.0, 54.0]), Segment::Line([12.0, 54.0])],
            seg_flags: Vec::new(),
        };
        let g = Geometry::Path { fill: FillMode::Winding, figures: vec![fig] };
        let got = g.widen_realization(3.0, &StrokeStyle::default(), 0.25);
        assert_eq!(got.len(), 1);
        // Native's figure starts at the inner rail's start and ends with the line back to it.
        let native: [[f32; 2]; 27] = [
            [12.0, 11.5], [12.0, 8.5], [84.0, 8.5], [85.5, 8.5], [85.5, 10.0], [85.5, 54.0], [85.5, 55.5], [84.0, 55.5],
            [12.0, 55.5], [10.5, 55.5], [10.5, 54.0], [10.5, 10.0], [10.5, 8.5], [12.0, 8.5], [12.0, 11.5], [12.0, 10.0],
            [13.5, 10.0], [13.5, 54.0], [12.0, 54.0], [12.0, 52.5], [84.0, 52.5], [84.0, 54.0], [82.5, 54.0], [82.5, 10.0],
            [84.0, 10.0], [84.0, 11.5], [12.0, 11.5],
        ];
        assert_eq!(got[0], native);
    }

    // The pen dashes with each length's magnitude; lengths summing to zero stroke solid.
    #[test]
    fn pen_dashes_take_magnitudes() {
        let st = |d: &[f32]| StrokeStyle { dashes: d.to_vec(), ..StrokeStyle::default() };
        assert_eq!(st(&[-1e9, -1e-20, 5.75]).pen_dashes(), Some(vec![1e9, 1e-20, 5.75]));
        assert_eq!(st(&[0.0, -0.0]).pen_dashes(), None);
        assert_eq!(st(&[]).pen_dashes(), None);
    }

    // A thick horizontal segment with flat caps widens to its ±half-width rectangle.
    #[test]
    fn flat_rectangle() {
        let s = StrokeStyle::default();
        let pen = PenShape::build(12.0, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [1.0, 1.0], TransformType::Normal, 0.25).unwrap();
        let polys = widen(&[[10.0, 20.0], [90.0, 20.0]], false, &pen, &s);
        assert_eq!(polys.len(), 1);
        let p = &polys[0];
        let miny = p.iter().map(|q| q[1]).fold(f32::INFINITY, f32::min);
        let maxy = p.iter().map(|q| q[1]).fold(f32::NEG_INFINITY, f32::max);
        assert!((miny - 14.0).abs() < 1e-3 && (maxy - 26.0).abs() < 1e-3, "y {miny}..{maxy}");
    }
}
