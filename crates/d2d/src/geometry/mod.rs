//! Phase 1C geometry: path/rect/rounded/ellipse/group/transformed geometries, the native curve
//! flattener, stroke styles, and the geometry queries (`ComputeArea`, `ComputeLength`,
//! `ComputePointAtLength`, `GetBounds`, `FillContainsPoint`, …) reproducing Direct2D on
//! WARP.
//!
//! Geometry produces **flattened device-space polylines** that feed the 1B polygon rasterizer
//! (`crate::pixels::fill_path`), and scalar/bool query results checked against the WARP oracle via
//! the harness v2 `query` op. Everything runs in `f32` in native operation order (MXCSR `0x1F80`).

mod dash;
pub mod combine;
pub(crate) mod flatten;
mod milbezier;
pub mod realization;
pub mod stroke;
pub mod fill_tess;
pub mod tessellate;
pub mod aatess;

pub use flatten::Point;
pub use stroke::StrokeStyle;

/// Row-major 3×2 affine `[m11, m12, m21, m22, dx, dy]`; a point maps as
/// `(x·m11 + y·m21 + dx, x·m12 + y·m22 + dy)`.
pub type Matrix = [f32; 6];

pub const IDENTITY: Matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
/// `D2D1_DEFAULT_FLATTENING_TOLERANCE`.
pub const DEFAULT_TOLERANCE: f32 = 0.25;
/// Native ellipse/rounded-rect corner control offset (the figure builder).
const ROUND_CTRL: f64 = 0.4477152501692065;

/// `D2D1_FILL_MODE`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FillMode {
    Alternate,
    Winding,
}

#[inline]
pub fn transform_point(m: &Matrix, [x, y]: Point) -> Point {
    [x * m[0] + y * m[2] + m[4], x * m[1] + y * m[3] + m[5]]
}

/// The rect a rectangle geometry's shape holds: each axis ordered low-to-high, swapped only when
/// `lo > hi` (a NaN never swaps). Every shape operation walks it; only `GetRect` keeps the raw one.
pub(crate) fn ordered_rect([l, t, r, b]: [f32; 4]) -> [f32; 4] {
    let [[l, r], [t, b]] = [[l, r], [t, b]].map(|[lo, hi]| if lo > hi { [hi, lo] } else { [lo, hi] });
    [l, t, r, b]
}

/// Compose: apply `a` then `b` (`a ∘ b`, point maps through `a` first).
pub fn mat_mul(a: &Matrix, b: &Matrix) -> Matrix {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    ]
}

/// Tolerance normalization: `≤0 → 0.25`, else `max(tol, 1e-6)`.
/// The scanner's fixed-point workspace (set from a shape's tight bounds, then mapping each flattened
/// point): whether a shape with tight bounds `b` and flattened points `pts` is rejected with
/// `0x88990011` (`D2DERR_BAD_NUMBER`). Inverted (empty) bounds take a fixed unit box; a NaN bound
/// rejects; the fixed scale `2^20` (centre 0) holds unless a bound corner would overflow it past `2^48`
/// (`(left-1)·2^20·1.5 ≤ -2^48 || (right+1)·2^20·1.5 ≥ 2^48`, and top/bottom), in which case a
/// non-finite extent rejects and a finite one takes the dynamic per-axis scale `2^48/(extent·1.5+2)`
/// with the midpoint centre snapped to the scale grid. Each point maps to
/// `round_half_away((p-centre)·scale)`, 64-bit truncated (INT64_MIN for ±inf / NaN / ≥ 2^63), and
/// rejects when `|i| > 2^48` — a near-degenerate curve whose analytic bounds stay small while its
/// flattening overshoots to a huge or non-finite point.
pub(crate) fn workspace_rejects(b: [f32; 4], pts: impl IntoIterator<Item = [f32; 2]>) -> bool {
    const P20: f64 = 1_048_576.0; // 2^20
    const P48: f64 = 281_474_976_710_656.0; // 2^48
    let [mut bl, mut bt, mut br, mut bb] = b.map(|v| v as f64);
    if bl > br || bt > bb {
        (bl, bt, br, bb) = (0.0, 0.0, 1.0, 1.0);
    }
    if [bl, bt, br, bb].iter().any(|v| v.is_nan()) {
        return true;
    }
    let overflow = (bl - 1.0) * P20 * 1.5 <= -P48
        || (bt - 1.0) * P20 * 1.5 <= -P48
        || (br + 1.0) * P20 * 1.5 >= P48
        || (bb + 1.0) * P20 * 1.5 >= P48;
    let (cx, cy, sx, sy) = if !overflow {
        (0.0, 0.0, P20, P20)
    } else {
        if !(br - bl).is_finite() || !(bb - bt).is_finite() {
            return true; // the dynamic scale's extents must be finite
        }
        let scl = |ext: f64| {
            let n = ext * 1.5 + 2.0;
            if n <= 3.131513062514021e-294 { 1.0 } else { P48 / n }
        };
        let snap = |c: f64, s: f64| if s == 0.0 { c } else { (s * c + 0.5).floor() / s };
        let (sx, sy) = (scl(br - bl), scl(bb - bt));
        (snap((bl + br) * 0.5, sx), snap((bt + bb) * 0.5, sy), sx, sy)
    };
    let out = |v: f64, c: f64, s: f64| -> bool {
        let d = (v - c) * s;
        let r = if d < 0.0 { d - 0.5 } else { d + 0.5 }; // round half away from zero
        // 64-bit truncate toward zero; INT64_MIN for ±inf / NaN / magnitude ≥ 2^63.
        let i: i64 = if r >= -9.223372036854776e18 && r < 9.223372036854776e18 { r as i64 } else { i64::MIN };
        (i.wrapping_add(P48 as i64) as u64) > ((P48 as u64) << 1) // |i| > 2^48
    };
    pts.into_iter().any(|q| out(q[0] as f64, cx, sx) || out(q[1] as f64, cy, sy))
}

pub fn normalize_tolerance(tol: f32) -> f32 {
    if tol <= 0.0 { DEFAULT_TOLERANCE } else { tol.max(1e-6) }
}

/// Whether `CreateStrokedGeometryRealization` bakes an empty realization: it widens at create time in
/// DIP space (identity transform) with the realization's flattening `tolerance`, and the pen voids when
/// its radius² falls under `(0.004·tol)²` ([`stroke::PenShape`]). r13 18223's 0.76 DIP stroke draws up to
/// tolerance 70 and nothing from 100 on (VM ladder; the void is at 0.38/0.004 = 95). A shape that widens
/// itself ([`realization::stroke_special_case`]: a rectangle, or a rounded rect or ellipse with equal
/// radii, stroked solid with a normal pen inside the safe box) never builds the pen, so it never voids.
pub fn stroked_realization_voids(geom: &Geometry, width: f32, style: &StrokeStyle, tolerance: f32) -> bool {
    width != 0.0
        && realization::stroke_special_case(geom, width, style).is_none()
        && stroke::PenShape::build(width, &IDENTITY, [1.0, 1.0], style.transform_type, normalize_tolerance(tolerance) as f64).is_none()
}

/// A path segment, as supplied to a geometry sink (quadratics and arcs are raised to cubics when
/// building primitives, exactly as native does).
#[derive(Clone, Copy, Debug)]
pub enum Segment {
    Line(Point),
    Cubic(Point, Point, Point),
    Quadratic(Point, Point),
    Arc { to: Point, radius: Point, rotation: f32, clockwise: bool, large: bool },
}

#[derive(Clone, Debug, Default)]
pub struct Figure {
    pub start: Point,
    pub filled: bool,
    pub closed: bool,
    pub segments: Vec<Segment>,
    /// `D2D1_PATH_SEGMENT` flags per segment (aligned with `segments`; empty/short ⇒ NONE), from
    /// `ID2D1GeometrySink::SetSegmentFlags`. Only `FORCE_UNSTROKED` (1) and `FORCE_ROUND_LINE_JOIN` (2)
    /// exist; they affect STROKING only (fill ignores them). See [`Geometry::stroke_figures`]. One
    /// entry past the segments, when present, holds the flags in force at `EndFigure`, which the
    /// closing edge takes; without it the closing edge takes the last segment's.
    pub seg_flags: Vec<u32>,
}

impl Figure {
    /// A NaN in the shape d2d1 builds from this figure: a NaN point, or an arc with a non-finite end or parameter,
    /// whose conversion is NaN.
    pub(crate) fn raw_has_nan(&self) -> bool {
        let nan = |p: &Point| p[0].is_nan() || p[1].is_nan();
        let fin = |p: &Point| p[0].is_finite() && p[1].is_finite();
        let mut cur = self.start;
        nan(&cur) || self.segments.iter().any(|seg| {
            let bad = match seg {
                Segment::Line(p) => nan(p),
                Segment::Quadratic(c, p) => nan(c) || nan(p),
                Segment::Cubic(a, b, p) => nan(a) || nan(b) || nan(p),
                Segment::Arc { to, radius, rotation, .. } => !fin(&cur) || !fin(to) || !fin(radius) || !rotation.is_finite(),
            };
            cur = match seg { Segment::Line(p) | Segment::Quadratic(_, p) | Segment::Cubic(_, _, p) => *p, Segment::Arc { to, .. } => *to };
            bad
        })
    }

    /// The flags a closed figure's closing edge (and its corner at the start) takes.
    fn close_flags(&self) -> u32 {
        let n = self.segments.len();
        self.seg_flags.get(n).or_else(|| self.seg_flags.get(n.wrapping_sub(1))).copied().unwrap_or(0)
    }

    /// Whether arc segment `i` (from the point before it) uses its rotation: native's sink converts an
    /// arc as it arrives, and an arc with coincident ends or a too-small radius becomes nothing or a
    /// straight line before the rotation is read (r22 27656: a zero-radius arc with an infinite rotation
    /// fills as a line).
    fn arc_uses_rotation(&self, i: usize) -> bool {
        let Some(&Segment::Arc { to, radius, .. }) = self.segments.get(i) else { return false };
        let cur = match i.checked_sub(1).map(|j| &self.segments[j]) {
            None => self.start,
            Some(Segment::Line(q) | Segment::Quadratic(_, q) | Segment::Cubic(_, _, q)) => *q,
            Some(Segment::Arc { to, .. }) => *to,
        };
        arc_degenerate(cur, to, radius).is_none()
    }

    /// Whether every segment point (ends and controls) is the start itself: a figure the widener skips
    /// whole.
    fn stays_at_start(&self) -> bool {
        let s = self.start;
        self.segments.iter().all(|seg| match seg {
            Segment::Line(p) => *p == s,
            Segment::Quadratic(c, p) => *c == s && *p == s,
            Segment::Cubic(a, b, p) => *a == s && *b == s && *p == s,
            Segment::Arc { to, .. } => *to == s,
        })
    }

    /// A figure with any non-finite coordinate (a point, or an arc radius/rotation): native's figure
    /// sink rejects it, so the whole figure is dropped from fill/stroke/bounds (VM-pinned: a filled
    /// path whose cubic has a nan control point draws nothing — the clear shows through). We otherwise
    /// flatten it and over-draw.
    pub fn has_nonfinite(&self) -> bool {
        let p = |q: Point| !(q[0].is_finite() && q[1].is_finite());
        p(self.start)
            || self.segments.iter().enumerate().any(|(i, s)| match *s {
                Segment::Line(a) => p(a),
                Segment::Cubic(a, b, c) => p(a) || p(b) || p(c),
                Segment::Quadratic(a, b) => p(a) || p(b),
                Segment::Arc { to, radius, rotation, .. } => p(to) || p(radius) || (!rotation.is_finite() && self.arc_uses_rotation(i)),
            })
    }

    /// A non-finite POSITION only (segment endpoint / control point), ignoring a non-finite arc
    /// radius or rotation. Native's sink rejects a figure whose POINTS are non-finite (voiding the
    /// whole geometry), but a non-finite arc radius/rotation only DEGENERATES that arc (it still
    /// realizes/fills the figure — recap 7463: a `[-inf,…]` arc radius strokes a degenerate arc, so
    /// the whole-geometry void must not fire on it). Used for the whole-geometry void; the per-figure
    /// flatten filters keep `has_nonfinite` (they drop the single bad figure, matching native there).
    pub fn has_nonfinite_pos(&self) -> bool {
        let p = |q: Point| !(q[0].is_finite() && q[1].is_finite());
        p(self.start)
            || self.segments.iter().any(|s| match *s {
                Segment::Line(a) => p(a),
                Segment::Cubic(a, b, c) => p(a) || p(b) || p(c),
                Segment::Quadratic(a, b) => p(a) || p(b),
                Segment::Arc { to, .. } => p(to),
            })
    }

    /// A non-finite coordinate in any SEGMENT (endpoint, cubic/quad control, or arc radius/rotation),
    /// ignoring the figure START. This is native's `ID2D1Geometry::Widen` BAD_NUMBER condition for a
    /// direct stroke (Widen-oracle confirmed: 17436 op7 `line to [_,inf]`, 9304 inf control + nan
    /// endpoint, 17277 nan endpoint, 17866 inf arc radius, 17877 inf controls all return 0x88990011),
    /// unlike a non-finite START, which native drops so the figure still strokes (16387 inf-y start,
    /// 17627/17998 NaN start). Our widener CLIPS a non-finite endpoint to the viewport before the pen
    /// sees it (native's pen sees the raw point and errors), so the whole-stroke void is decided here on
    /// the raw figure, not during widening.
    pub fn has_nonfinite_nonstart(&self) -> bool {
        let p = |q: Point| !(q[0].is_finite() && q[1].is_finite());
        // A non-finite ENDPOINT is the BAD_NUMBER (9304 nan `to`, 17436 line to `inf`). A non-finite
        // cubic/quad CONTROL is left to the widener's own arithmetic, as native: under an identity
        // transform an inf control flattens to NaN points or an inf end tangent and errors (24419), under
        // a scaling one `0·inf` makes its other coordinate NaN, the curve tests flat and widens as a line
        // to its endpoint (22884, 18852). A finite-huge control can still overflow the 28.4 edge list
        // in device space and void later (20085/21166).
        self.segments.iter().any(|s| match *s {
            Segment::Line(a) => p(a),
            Segment::Cubic(_, _, c) => p(c),
            Segment::Quadratic(_, b) => p(b),
            Segment::Arc { to, radius, rotation, .. } => p(to) || p(radius) || !rotation.is_finite(),
        })
    }

    /// The larger finite `|device coordinate|` of this figure's START under `m`, or `0` when the start
    /// is non-finite. The stroke path compares this to the fixed-point edge-overflow cutoff: native
    /// opens the widened contour at the start vertex, so a finite `±1e9` start overflows the 28.4 edge
    /// list and native draws nothing (fuzz 17927), whereas a huge coordinate later in the figure — a
    /// `-1e9` arc endpoint (fuzz 7408) or a `NaN` segment endpoint (fuzz 11119) — is widened and
    /// rendered per-segment. A `NaN`/`inf` start is not an overflow (native's bounds drop it), so it
    /// is skipped here and the figure is still widened (fuzz 17627/17998 stroke a `NaN` start).
    fn start_device_maxabs(&self, m: &Matrix) -> f32 {
        self.start_device_maxabs_via(&self.start, m)
    }

    /// [`Self::start_device_maxabs`] with the start as the transformed shapes around the figure moved it.
    fn start_device_maxabs_via(&self, moved: &Point, m: &Matrix) -> f32 {
        // Gate on the LOCAL start's finiteness: a non-finite LOCAL start is not the overflow (native
        // drops it and still widens — 16387 strokes an inf-y start, 17627/17998 a NaN start), but a
        // FINITE-huge local start that overflows f32 under the device transform IS (17436's `-3.4e38`
        // start → `-inf`), so take the device magnitude with an overflow-to-inf counting as huge.
        if !(self.start[0].is_finite() && self.start[1].is_finite()) {
            return 0.0;
        }
        let d = transform_point(m, *moved);
        let f = |v: f32| if v.is_nan() { 0.0 } else { v.abs() };
        f(d[0]).max(f(d[1]))
    }
}

/// A retained geometry. Rect/rounded/ellipse lower to figures on demand; groups and transformed
/// geometries compose their children.
#[derive(Clone, Debug)]
pub enum Geometry {
    Rect([f32; 4]),
    Rounded { rect: [f32; 4], radius: Point },
    Ellipse { center: Point, radius: Point },
    Path { fill: FillMode, figures: Vec<Figure> },
    Group { fill: FillMode, children: Vec<Geometry> },
    Transformed { matrix: Matrix, child: Box<Geometry> },
}

/// A cubic/line primitive after quadratics/arcs are raised and the transform is applied — the input
/// to the flattener (the fill-path primitive sink).
#[derive(Clone, Copy)]
pub(crate) enum Prim {
    Line(Point, Point),
    Cubic(Point, Point, Point, Point),
}

/// One figure's flattened device-space polyline.
#[derive(Clone, Debug)]
pub struct Polyline {
    pub pts: Vec<Point>,
    pub closed: bool,
    pub filled: bool,
}

// --- lowering shapes to primitives ----------------------------------------------------------

fn quadratic_to_cubic(start: Point, c: Point, end: Point) -> (Point, Point) {
    // Quadratic→cubic elevation: c1 = start·⅓ + ctrl·⅔, c2 = end·⅓ + ctrl·⅔, with
    // these exact f32 constants (not `1−t`), so the raised control points are bit-identical.
    let (a, b) = (0.33333334f32, 0.66666669f32);
    (
        [start[0] * a + c[0] * b, start[1] * a + c[1] * b],
        [end[0] * a + c[0] * b, end[1] * a + c[1] * b],
    )
}

// Native `ArcToBezier`: the exact arc→cubic conversion the figure sink runs on
// the CPU. All-f32 geometry through the CRT's `sinf`/`cosf`/`atan2f` ([`crate::crt`]); the Bézier
// tangent factor alone is f64 (native uses a `double` sqrt, not `tanf`). The arc becomes 1–4 equal
// sub-arcs — the count is chosen from the sweep quadrant and the large-arc flag, not `ceil(θ/90°)` —
// and each sub-arc one cubic. `clockwise` is D2D's sweep direction (CLOCKWISE), `large` the arc size
// (LARGE). Inputs are in the figure's coordinate space (device space here, which equals that space
// under the identity/axis transforms the captured arc scenes use).
/// What an arc converts to before its rotation is read: nothing for coincident ends, a straight line
/// to the end for a too-small radius, else `None` (a real arc).
fn arc_degenerate(start: Point, to: Point, radius: Point) -> Option<Vec<[Point; 4]>> {
    let hdx = (to[0] - start[0]) * 0.5;
    let hdy = (to[1] - start[1]) * 0.5;
    let d2 = hdy * hdy + hdx * hdx;
    if d2 < 1.0e-12 {
        return Some(Vec::new());
    }
    let thresh = d2 * 1.0e-12;
    (thresh >= radius[0] * radius[0] || thresh >= radius[1] * radius[1]).then(|| vec![[start, to, to, to]])
}

fn arc_to_cubics(start: Point, to: Point, radius: Point, rotation: f32, clockwise: bool, large: bool) -> Vec<[Point; 4]> {
    use crate::crt::system_crt::{atan2f, cosf, sinf};
    if let Some(d) = arc_degenerate(start, to, radius) {
        return d;
    }
    let (sx, sy) = (start[0], start[1]);
    let (ex, ey) = (to[0], to[1]);
    let hdx = (ex - sx) * 0.5;
    let hdy = (ey - sy) * 0.5;
    let (mut rx, mut ry) = (radius[0], radius[1]);
    rx = rx.abs();
    ry = ry.abs();
    // rotate the half-chord into the ellipse's axis frame (native: cos/sin of −rotation°)
    // The threshold test runs in f64; an unordered (NaN) rotation is rotated by, as native's test
    // sends it: the arc comes out NaN.
    let (cos_r, sin_r) = if !((rotation as f64).abs() < 1.0e-6) {
        let a = -rotation * 0.017453292;
        (cosf(a), sinf(a))
    } else {
        (1.0f32, 0.0f32)
    };
    let hx = cos_r * hdx - sin_r * hdy;
    let hy = cos_r * hdy + sin_r * hdx;
    let mut nx = hx / rx;
    let mut ny = hy / ry;
    let lambda = ny * ny + nx * nx;
    // centre offset in the normalized frame (ox pairs with x, oy with y), and whether the radii were
    // grown to reach the endpoints (centre then sits at the chord midpoint).
    let (ox, oy, center_at_mid) = if lambda > 1.0 {
        let s = lambda.sqrt();
        rx *= s;
        ry *= s;
        nx /= s;
        ny /= s;
        (0.0f32, 0.0f32, true)
    } else {
        let t = ((1.0 - lambda) / lambda).sqrt();
        let (q0, q1) = if large == clockwise { (-t, t) } else { (t, -t) };
        (q1 * ny, q0 * nx, false)
    };
    // back-rotation of a normalized ellipse point (p, q) to device space
    let (m0, m1, m2, m3) = (cos_r * rx, sin_r * ry, -sin_r * rx, cos_r * ry);
    let mut midx = (sx + ex) * 0.5;
    let mut midy = (sy + ey) * 0.5;
    if !center_at_mid {
        midx += m1 * oy + m0 * ox;
        midy += m3 * oy + m2 * ox;
    }
    let map = |p: f32, q: f32| [m0 * p + m1 * q + midx, m2 * p + m3 * q + midy];
    // start / end points relative to the centre, normalized
    let (svx, svy) = (-nx - ox, -ny - oy);
    let (evx, evy) = (nx - ox, ny - oy);
    let dot = evy * svy + evx * svx;
    let cross = evy * svx - evx * svy;
    // equal sub-arc count {1,2,3,4} from the sweep quadrant + large flag, and the per-sub-arc cos/sin;
    // a NaN dot takes the obtuse branch
    let (nseg, cos_s, sin_s) = if !(dot >= 0.0) {
        let n = if large { 3 } else { 2 };
        let seg = adjust_sweep(atan2f(cross, dot), clockwise) / n as f32;
        (n, cosf(seg), sinf(seg))
    } else if large {
        let seg = adjust_sweep(atan2f(cross, dot), clockwise) / 4.0;
        (4, cosf(seg), sinf(seg))
    } else {
        (1, dot, cross)
    };
    // Bézier tangent factor k = 4/3·tan(seg/4), via f64 sqrt in native's exact order; each guard
    // skips only on an ordered compare, so a NaN angle gives a NaN factor.
    let kappa = {
        let c = (cos_s as f64 + 1.0) * 0.5;
        let mut k = 0.0f64;
        if !(0.0 > c) {
            let omc = 1.0 - c;
            if !(0.0 >= omc) {
                let (s_half, c_half) = (omc.sqrt(), c.sqrt());
                let num = (1.0 - c_half) * (4.0 / 3.0);
                if !(s_half * 1.0e-6 >= num) {
                    k = num / s_half;
                }
            }
        }
        k as f32
    };
    let kap = if clockwise { kappa } else { -kappa };
    let nkap = -kap;

    // Walk the sub-arcs: the start vector rotates by (cos_s, sin_s) each step; emit each cubic's two
    // control points and its on-ellipse end. Native fills a flat [c1, c2, end] triple per sub-arc.
    let mut pts: Vec<Point> = Vec::with_capacity(3 * nseg as usize);
    let (mut cx, mut cy) = (svx, svy);
    let (mut c1n, mut c1p) = (kap * svx, nkap * svy);
    for _ in 1..nseg {
        let (px, py) = (cx, cy);
        let nxx = px * cos_s - py * sin_s;
        let nxy = px * sin_s + py * cos_s;
        pts.push(map(px + c1p, py + c1n));
        let c2y = nxy - nxx * kap;
        pts.push(map(nxx - nxy * nkap, c2y));
        pts.push(map(nxx, nxy));
        c1n = nxx * kap;
        c1p = nxy * nkap;
        cx = nxx;
        cy = nxy;
    }
    // final sub-arc — it ends exactly at (ex, ey)
    pts.push(map(cx + c1p, cy + c1n));
    pts.push(map(evx + kap * evy, evy - evx * kap));
    pts.push([ex, ey]);

    let mut out = Vec::with_capacity(nseg as usize);
    let mut p0 = start;
    for i in 0..nseg as usize {
        out.push([p0, pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]]);
        p0 = pts[3 * i + 2];
    }
    out
}

/// Pull an `atan2f` sweep angle into D2D's sweep direction (native: CLOCKWISE → non-negative).
fn adjust_sweep(ang: f32, clockwise: bool) -> f32 {
    if clockwise {
        if ang < 0.0 { ang + 6.2831855 } else { ang }
    } else if ang > 0.0 {
        ang - 6.2831855
    } else {
        ang
    }
}

// The 16-point native rounded-rectangle / ellipse figure. `radius` is clamped per axis to the
// half-extent, and corner controls are offset from the rect corners by `r·ROUND_CTRL` — the exact
// float expression native uses (center-relative kappa would be 1 ULP off).
fn rounded_rect_prims(rect: [f32; 4], radius: Point) -> (Point, Vec<Prim>) {
    // A NaN radius makes the corner control points NaN; native's radius clamp (a plain min/max) lets
    // NaN through, so the figure is degenerate and nothing rasterizes — fill AND stroke draw nothing
    // (fuzz 11883: a [19.2, NaN]-radius rounded rect strokes nothing natively; we were sanitizing the
    // NaN to 0 via `max(0).min(h)` and drawing a sharp-cornered rect). Emit an empty figure.
    if radius[0].is_nan() || radius[1].is_nan() {
        return ([rect[0], rect[1]], vec![]);
    }
    rounded_rect_figure(rect, radius)
}

/// The rounded-rectangle / ellipse figure the shape streams: its 16 points as four corner cubics
/// joined by lines, NaN radii included.
fn rounded_rect_figure(rect: [f32; 4], radius: Point) -> (Point, Vec<Prim>) {
    let (pts, _) = rounded_rect_points(rect, radius);
    let prims = vec![
        Prim::Cubic(pts[0], pts[1], pts[2], pts[3]),
        Prim::Line(pts[3], pts[4]),
        Prim::Cubic(pts[4], pts[5], pts[6], pts[7]),
        Prim::Line(pts[7], pts[8]),
        Prim::Cubic(pts[8], pts[9], pts[10], pts[11]),
        Prim::Line(pts[11], pts[12]),
        Prim::Cubic(pts[12], pts[13], pts[14], pts[15]),
        Prim::Line(pts[15], pts[0]),
    ];
    (pts[0], prims)
}

/// The 16 points of the native rounded-rectangle / ellipse figure and the radii it stores (each the
/// radius' magnitude, cut to the half-extent when larger; a NaN radius passes through).
fn rounded_rect_points(rect: [f32; 4], radius: Point) -> ([Point; 16], [f32; 2]) {
    // Native normalizes an inverted rect (left>right or top>bottom) before laying out the figure, so
    // the per-axis radius clamp uses the true half-extent rather than a negative span (fuzz 20036:
    // a stroked rounded-rect with left>right otherwise built mirrored corner cubics).
    let [l, t, r, b] = ordered_rect(rect); // a NaN side never swaps
    // Each radius is its magnitude clamped to the half-extent: a NEGATIVE radius rounds like its
    // positive (r10 15019: native −2.18 == +2.18, rr019-* probes), and a radius over the half-extent
    // saturates — a bare `.min(half)` kept the negative (fuzz 924: a −305 y-radius ballooned the
    // corner cubics far outside the rect and over-filled). Native cuts with `if r > half`, so a NaN
    // half-extent keeps the radius and a NaN radius stays NaN.
    let cut = |r: f32, half: f32| if r > half { half } else { r };
    let rx = cut(radius[0].abs(), (r - l) * 0.5);
    let ry = cut(radius[1].abs(), (b - t) * 0.5);
    // The control offset is the f32 radius times the double constant, rounded once to f32 (r23
    // 28164/28812: an f32 constant puts a corner control an ULP off).
    let cx = (rx as f64 * ROUND_CTRL) as f32;
    let cy = (ry as f64 * ROUND_CTRL) as f32;
    let p = |x: f32, y: f32| [x, y];
    // Points P0..P15 as `InitBufferWithRoundedRectanglePoints` lays them out.
    let pts = [
        p(l, t + ry),   // 0 LEFT
        p(l, t + cy),   // 1
        p(l + cx, t),   // 2
        p(l + rx, t),   // 3 TOP
        p(r - rx, t),   // 4 TOP
        p(r - cx, t),   // 5
        p(r, t + cy),   // 6
        p(r, t + ry),   // 7 RIGHT
        p(r, b - ry),   // 8 RIGHT
        p(r, b - cy),   // 9
        p(r - cx, b),   // 10
        p(r - rx, b),   // 11 BOTTOM
        p(l + rx, b),   // 12 BOTTOM
        p(l + cx, b),   // 13
        p(l, b - cy),   // 14
        p(l, b - ry),   // 15 LEFT
    ];
    (pts, [rx, ry])
}

/// One segment for [`hit_core`]: a line end, or a cubic's two controls and end, in f64.
enum HitSeg {
    Line([f64; 2]),
    Cubic([f64; 2], [f64; 2], [f64; 2]),
}

/// Native's hit tester over filled figures `(start, prims, _)` streamed under `m`: each point is
/// taken relative to `pt` in f64 ([`hit_core`]).
fn hit_test(figs: &[(Point, Vec<Prim>, bool)], pt: Point, m: &Matrix, tol: f64, winding: bool) -> bool {
    let (dx, dy) = ((m[4] - pt[0]) as f64, (m[5] - pt[1]) as f64);
    let map = |p: Point| { let (x, y) = (p[0] as f64, p[1] as f64); [m[2] as f64 * y + m[0] as f64 * x + dx, m[3] as f64 * y + m[1] as f64 * x + dy] };
    let figs: Vec<([f64; 2], Vec<HitSeg>)> = figs
        .iter()
        .map(|(s, prims, _)| (map(*s), prims.iter().map(|p| match *p {
            Prim::Line(_, e) => HitSeg::Line(map(e)),
            Prim::Cubic(_, c1, c2, e) => HitSeg::Cubic(map(c1), map(c2), map(e)),
        }).collect()))
        .collect();
    hit_core(&figs, tol, winding)
}

/// The hit tester's test of figures whose points are relative to the query point: a point or segment
/// strictly within `tol` of it counts as on the outline, a hit; otherwise the figures' winding count
/// decides, by parity unless `winding`. A figure closes back to its start; a curve flattens through the
/// f64 widening flattener at `tol`.
fn hit_core(figs: &[([f64; 2], Vec<HitSeg>)], tol: f64, winding: bool) -> bool {
    let tol_sq = tol * tol;
    let (mut on_edge, mut count) = (false, 0i32);
    let mut step = |prev: &mut [f64; 2], p: [f64; 2]| {
        hit_step(tol_sq, &mut on_edge, &mut count, *prev, p);
        *prev = p;
    };
    for (first, segs) in figs {
        let mut prev = *first;
        for seg in segs {
            match *seg {
                HitSeg::Line(e) => step(&mut prev, e),
                HitSeg::Cubic(c1, c2, e) => {
                    let mut pts = Vec::new();
                    stroke::flatten_cubic_tan(prev, c1, c2, e, &mut pts, tol.max(0.0) * 6.0);
                    for (q, _) in pts {
                        step(&mut prev, q);
                    }
                }
            }
        }
        step(&mut prev, *first);
    }
    on_edge || if winding { count != 0 } else { count & 1 != 0 }
}

/// One edge of the hit tester, `prev` to `p`, both relative to the query point: once a point or edge
/// comes strictly within the tolerance (`tol_sq` squared) the query point is on the outline for good;
/// the edge's crossing of the positive x axis adds to the winding `count`.
fn hit_step(tol_sq: f64, on_edge: &mut bool, count: &mut i32, prev: [f64; 2], p: [f64; 2]) {
    if !*on_edge {
        if tol_sq > p[0] * p[0] + p[1] * p[1] {
            *on_edge = true;
        } else {
            let (sx, sy) = (p[0] - prev[0], p[1] - prev[1]);
            let len_sq = sy * sy + sx * sx;
            let t = -(sy * prev[1] + sx * prev[0]);
            if t >= 0.0 && len_sq >= t {
                let (cx, cy) = (sx * t + prev[0] * len_sq, sy * t + prev[1] * len_sq);
                *on_edge = len_sq * tol_sq * len_sq > cy * cy + cx * cx;
            }
        }
    }
    if prev[1] > 0.0 {
        if p[1] <= 0.0 && p[1] * prev[0] - prev[1] * p[0] >= 0.0 {
            *count -= 1;
        }
    } else if p[1] > 0.0 && prev[1] * p[0] - p[1] * prev[0] >= 0.0 {
        *count += 1;
    }
}

/// The rectangle `[l, t, r, b]` an ellipse is built on (`center ± radius`).
fn ellipse_rect(center: Point, radius: Point) -> [f32; 4] {
    [center[0] - radius[0], center[1] - radius[1], center[0] + radius[0], center[1] + radius[1]]
}

// Build a figure's primitives (transform applied to every control point, then quadratics/arcs
// raised), plus the closing line for closed figures (native `BuildPrimitives` includeClosing).
fn figure_prims(fig: &Figure, m: &Matrix) -> (Point, Vec<Prim>) {
    figure_prims_by(fig, |p| map_point(m, p), |p| map_point(m, p))
}

/// [`figure_prims`] with the start mapped by `ts` and every other local point by `tp`, as a
/// transforming sink maps `BeginFigure` and the segments it is sent apart.
fn figure_prims_by(fig: &Figure, ts: impl Fn(Point) -> Point, tp: impl Fn(Point) -> Point) -> (Point, Vec<Prim>) {
    let start = ts(fig.start);
    let mut cur = start;
    let mut cur_local = fig.start; // untransformed current point, for the arc's ellipse math
    let mut prims = Vec::new();
    for seg in &fig.segments {
        match *seg {
            Segment::Line(to) => {
                let e = tp(to);
                prims.push(Prim::Line(cur, e));
                cur = e;
                cur_local = to;
            }
            Segment::Cubic(c1, c2, to) => {
                let e = tp(to);
                prims.push(Prim::Cubic(cur, tp(c1), tp(c2), e));
                cur = e;
                cur_local = to;
            }
            Segment::Quadratic(c, to) => {
                // The path's shape builder raises a quadratic to a cubic as it is added, in the
                // figure's own space; the transform then maps the cubic's controls.
                let e = tp(to);
                let (c1, c2) = quadratic_to_cubic(cur_local, c, to);
                prims.push(Prim::Cubic(cur, tp(c1), tp(c2), e));
                cur = e;
                cur_local = to;
            }
            Segment::Arc { to, radius, rotation, clockwise, large } => {
                // Convert the arc to cubics in the figure's LOCAL (pre-transform) space — native's
                // ArcToBezier runs on the stored local arc — then affine-map every control point.
                // An affine transform of a cubic is the cubic of the mapped control points, so this
                // applies a rotation/shear/non-uniform scale to the arc exactly; transforming only the
                // endpoints while keeping the local radii (the old path) built the wrong ellipse under
                // any non-axis matrix (fuzz r5 5289 op#12 under a strong shear).
                for c in arc_to_cubics(cur_local, to, radius, rotation, clockwise, large) {
                    prims.push(Prim::Cubic(tp(c[0]), tp(c[1]), tp(c[2]), tp(c[3])));
                }
                cur = tp(to);
                cur_local = to;
            }
        }
    }
    if fig.closed && cur != start {
        prims.push(Prim::Line(cur, start));
    }
    (start, prims)
}

/// Like [`figure_prims`], plus the device positions where a round join is forced: the *start* vertex
/// of each segment carrying `D2D1_PATH_SEGMENT_FORCE_ROUND_LINE_JOIN` (bit 2) — i.e. the join that
/// *precedes* the flagged segment. Verified against native (`segflags-02-roundjoin`): flagging the
/// first segment is a no-op (its start is the start cap, not a join), and flagging a later segment
/// rounds the join at its start. Position-keyed, so it survives curve flattening and the
/// degenerate-segment dedup both wideners do.
fn figure_prims_flags(fig: &Figure, m: &Matrix) -> (Point, Vec<Prim>, Vec<Point>) {
    let (start, prims) = figure_prims(fig, m);
    let mut round_pts = Vec::new();
    let mut cur = fig.start;
    for (si, seg) in fig.segments.iter().enumerate() {
        if si > 0 && fig.seg_flags.get(si).copied().unwrap_or(0) & 2 != 0 {
            round_pts.push(transform_point(m, cur));
        }
        cur = match *seg {
            Segment::Line(to) | Segment::Cubic(_, _, to) | Segment::Quadratic(_, to) | Segment::Arc { to, .. } => to,
        };
    }
    // A closed figure's closing edge and its corner at the start take the flags in force at its end.
    if fig.closed && fig.close_flags() & 2 != 0 {
        if cur != fig.start {
            round_pts.push(transform_point(m, cur));
        }
        round_pts.push(start);
    }
    (start, prims, round_pts)
}

/// The endpoint of a path segment (the pen position after it), in figure space.
fn segment_end(seg: &Segment) -> Point {
    match *seg {
        Segment::Line(to) | Segment::Cubic(_, _, to) | Segment::Quadratic(_, to) | Segment::Arc { to, .. } => to,
    }
}


/// Split a figure carrying `FORCE_UNSTROKED` (`D2D1_PATH_SEGMENT` bit 1) segments into open stroked
/// sub-figures: a flagged segment is a gap (not stroked), so the stroke ends with its cap before it and
/// resumes at the segment's endpoint. Unflagged segments accumulate into the current run. Fill ignores
/// the flags, so this is only used on the widen (stroke) path. [SetSegmentFlags]
/// How native's widener caps a stroked run of a figure split at its unstroked segments. A run after a
/// gap starts with the dash cap, drawn inline, and a run before a gap ends with it. The figure's first
/// run, when no gap leads, starts flat and takes its start cap as a figure of its own after the figure;
/// the last run, when no gap trails, ends with the end cap. A closed figure strokes its closing edge
/// with its last segment's flags. When that edge and the first segment are stroked, the last run turns
/// the corner at the start onto the first run's direction and ends flat, and the first run starts flat
/// with no cap. When a gap closes the figure instead, the first run's deferred cap is the dash cap.
#[derive(Clone, Copy)]
pub(crate) struct RunCaps {
    pub(crate) start: RunCap,
    pub(crate) end: RunCap,
    /// The start cap is drawn after the figure, as a figure of its own.
    pub(crate) deferred: bool,
    /// The run closes the figure with the corner onto the first run (forced round when `close_round`).
    pub(crate) closes: bool,
    pub(crate) close_round: bool,
    /// The figure's first and last runs.
    pub(crate) first: bool,
    pub(crate) last: bool,
}

/// Which of the stroke style's caps a run end takes.
#[derive(Clone, Copy)]
pub(crate) enum RunCap {
    Start,
    End,
    Dash,
    Flat,
}

impl RunCaps {
    /// `style` with this run's start and end caps.
    pub(crate) fn style(&self, style: &StrokeStyle) -> StrokeStyle {
        let cap = |c: RunCap| match c {
            RunCap::Start => style.start_cap,
            RunCap::End => style.end_cap,
            RunCap::Dash => style.dash_cap,
            RunCap::Flat => stroke::Cap::Flat,
        };
        StrokeStyle { start_cap: cap(self.start), end_cap: cap(self.end), ..style.clone() }
    }
}

/// A figure's stroked runs between its unstroked segments (bit 1 of the segment flags), each an open
/// figure keeping its segments' flags, with the caps native's widener gives it ([`RunCaps`]).
fn split_unstroked(fig: &Figure) -> Vec<(Figure, RunCaps)> {
    let flag = |i: usize| fig.seg_flags.get(i).copied().unwrap_or(0);
    let mut segs: Vec<(Segment, u32)> = fig.segments.iter().enumerate().map(|(i, s)| (*s, flag(i))).collect();
    let last_flag = fig.close_flags();
    // The closing edge, when the figure does not end at its start, takes the last segment's flags.
    let end = segs.last().map_or(fig.start, |s| segment_end(&s.0));
    if fig.closed && end != fig.start {
        segs.push((Segment::Line(fig.start), last_flag));
    }
    let lead = segs.first().is_some_and(|s| s.1 & 1 != 0);
    let active = segs.last().is_some_and(|s| s.1 & 1 == 0);
    let mut runs: Vec<Figure> = Vec::new();
    let mut cur_start = fig.start;
    let (mut cur, mut flags): (Vec<Segment>, Vec<u32>) = (Vec::new(), Vec::new());
    let sub = |start, segments, seg_flags| Figure { start, filled: fig.filled, closed: false, segments, seg_flags };
    for (seg, f) in segs {
        if f & 1 != 0 {
            if !cur.is_empty() {
                runs.push(sub(cur_start, std::mem::take(&mut cur), std::mem::take(&mut flags)));
            }
            cur_start = segment_end(&seg); // skip the unstroked segment; resume at its endpoint
        } else {
            // A stroked segment keeps its flags: a forced round join still rounds the corner before it.
            cur.push(seg);
            flags.push(f);
        }
    }
    if !cur.is_empty() {
        runs.push(sub(cur_start, cur, flags));
    }
    let k = runs.len();
    runs.into_iter()
        .enumerate()
        .map(|(i, r)| {
            let (first, last) = (i == 0, i + 1 == k);
            let (start, deferred) = match (first && !lead, fig.closed) {
                (false, _) => (RunCap::Dash, false),
                (true, false) => (RunCap::Start, true),
                (true, true) if active => (RunCap::Flat, false),
                (true, true) => (RunCap::Dash, true),
            };
            let closes = last && fig.closed && active && !lead;
            let end = match (last && active, fig.closed) {
                (true, false) => RunCap::End,
                (true, true) if !lead => RunCap::Flat,
                _ => RunCap::Dash,
            };
            (r, RunCaps { start, end, deferred, closes, close_round: last_flag & 2 != 0, first, last })
        })
        .collect()
}

/// A figure's start and line ends as the widener maps them, from its local `(start, prims)` under `m`:
/// each in f64 (`(y·m21 + x·m11) + dx`; an identity transform leaves the point untouched), so a
/// coordinate the f32 map would round or overflow keeps its double value. A curve's points map in f32
/// with the device prims, so its entry is `None`. `force` holds the forced-round join vertices at the
/// same positions: a line's end in f64, a curve's in f32.
pub(crate) struct LineEnds64 {
    pub(crate) start: [f64; 2],
    pub(crate) lines: Vec<Option<[f64; 2]>>,
    pub(crate) force: Vec<[f64; 2]>,
    /// The figure is a stroked run of a figure split at its unstroked segments ([`split_unstroked`]).
    pub(crate) run: Option<RunCaps>,
}
fn map_f64(m: &Matrix, p: Point) -> [f64; 2] {
    if *m == IDENTITY {
        return [p[0] as f64, p[1] as f64];
    }
    let (x, y) = (p[0] as f64, p[1] as f64);
    [m[2] as f64 * y + m[0] as f64 * x + m[4] as f64, m[3] as f64 * y + m[1] as f64 * x + m[5] as f64]
}
fn line_ends_f64(start: Point, local: &[Prim], m: &Matrix, force: Vec<[f64; 2]>) -> LineEnds64 {
    LineEnds64 { start: map_f64(m, start), lines: local.iter().map(|p| match *p { Prim::Line(_, e) => Some(map_f64(m, e)), Prim::Cubic(..) => None }).collect(), force, run: None }
}
/// Every prim end of a closed shape as a forced-round vertex (a rounded rect's or ellipse's corners).
fn shape_force_f64(local: &[Prim], m: &Matrix) -> Vec<[f64; 2]> {
    let f = |p: Point| { let q = map_point(m, p); [q[0] as f64, q[1] as f64] };
    local.iter().map(|p| match *p { Prim::Line(_, e) => map_f64(m, e), Prim::Cubic(_, _, _, e) => f(e) }).collect()
}
/// [`figure_prims_flags`]' forced-round vertices with each one after a line mapped in f64.
fn path_force_f64(fig: &Figure, m: &Matrix) -> Vec<[f64; 2]> {
    let mut out = Vec::new();
    let (mut cur, mut after_line) = (fig.start, true);
    for (si, seg) in fig.segments.iter().enumerate() {
        if si > 0 && fig.seg_flags.get(si).copied().unwrap_or(0) & 2 != 0 {
            out.push(if after_line { map_f64(m, cur) } else { let q = transform_point(m, cur); [q[0] as f64, q[1] as f64] });
        }
        after_line = matches!(seg, Segment::Line(_));
        cur = segment_end(seg);
    }
    // A closed figure's closing edge and its corner at the start take the flags in force at its end.
    if fig.closed && fig.close_flags() & 2 != 0 {
        if cur != fig.start {
            out.push(if after_line { map_f64(m, cur) } else { let q = transform_point(m, cur); [q[0] as f64, q[1] as f64] });
        }
        out.push(map_f64(m, fig.start));
    }
    out
}

fn flatten_prims(start: Point, prims: &[Prim], tol: f32) -> Vec<Point> {
    let mut out = vec![start];
    for pr in prims {
        match *pr {
            Prim::Line(_, e) => out.push(e),
            Prim::Cubic(p0, p1, p2, p3) => flatten::flatten_cubic(p0, p1, p2, p3, tol, &mut out),
        }
    }
    out
}

// Device coord → 28.4 fixed point, matching `pixels::Edge` (and native
// the 28.4 point transform is `ceil(16·v − 8)`, clamped to ±2^23 sub-units.
fn to_28_4(v: f32) -> i32 {
    (16.0 * v - 8.0).ceil().clamp(-8388608.0, 8388608.0) as i32
}
// 28.4 → device coord such that `to_28_4` recovers it exactly.
fn from_28_4(p: [i32; 2]) -> Point {
    [(p[0] as f32 + 8.0) / 16.0, (p[1] as f32 + 8.0) / 16.0]
}

/// Snap a device point to the WARP fill tessellator's sub-pixel grid (round-to-nearest of the
/// half-pixel-biased `scale·v`). Native snaps every vertex the aliased fill tessellator receives.
pub(crate) fn snap_grid(v: Point, scale: f32) -> Point {
    let s = |c: f32| ((scale * c - scale * 0.5).round() + scale * 0.5) / scale;
    [s(v[0]), s(v[1])]
}

// Flatten device-space prims the way native **fills** do (the fill path flattens them on 28.4 coordinates), rather than the float query flattener: cubics go
// through the MIL integer cracker; line endpoints and the carried curve endpoint pass through as
// their 28.4 value. The result is the exact polyline native's scan converter receives. Shared with
// the stroker, which flattens its widened outline (with round cap/join beziers) the same way.
pub(crate) fn flatten_fill_prims(start: Point, prims: &[Prim]) -> Vec<Point> {
    flatten_fill_prims_scale(start, prims, 16.0, None)
}

/// As [`flatten_fill_prims`] but at an explicit sub-pixel `scale` (16 = 28.4, Direct2D.s CPU rasterizer
/// edge list; 64 = 26.6, WARP.s fill tessellator,
/// which `DrawInk`'s aliased brush fill routes through). The MIL integer cracker's subdivision count
/// scales with the grid, so a finer grid tracks the curve more closely, as the DDI capture shows.
/// `target` is the pixel rect a COPY fill's 28.4 edge list is clipped to (the target within the draw's
/// clips): that sink does not subdivide a curve whose loose bounds miss it, it steps straight to the
/// curve's end.
pub(crate) fn flatten_fill_prims_scale(start: Point, prims: &[Prim], scale: f32, target: Option<[i32; 4]>) -> Vec<Point> {
    // 28.4 rounds pixel centres with ceil; the finer
    // WARP-tessellator grid (26.6) rounds to nearest, which keeps a symmetric nib circle symmetric (the
    // ceil bias shifted one side 1/64 and dropped inkbrush's (24,33)). Verified against the DDI capture.
    // No ±2²³ clamp on an i32 control point: the edge setup clamps the flattened points, not the
    // cracker's input (w_bm bz-bulge2, a −1e6 control point under an in-range curve).
    // ponytail: past i32 (unclipped stroke outlines; native clips those shapes first) keeps the old
    // ±2²³ clamp rather than cvttss2si's INT_MIN, until the stroker clips its outline.
    let c1 = |v: f32| {
        let t = scale * v - scale * 0.5;
        let t = if scale > 16.0 { t.round() } else { t.ceil() };
        if (-2147483648.0..2147483648.0).contains(&t) { t as i32 } else { t.clamp(-8388608.0, 8388608.0) as i32 }
    };
    let back = |p: [i32; 2]| [(p[0] as f32 + scale * 0.5) / scale, (p[1] as f32 + scale * 0.5) / scale];
    let tol = (scale / 16.0).round().max(1.0) as i32; // 16→1 (28.4), 64→4 (26.6)
    let clip = target.filter(|_| scale == 16.0).map(|r| r.map(|v| v << 4));
    let mut out = vec![start];
    let mut cur = [c1(start[0]), c1(start[1])];
    let mut pts28 = Vec::new();
    for pr in prims {
        match *pr {
            Prim::Line(_, e) => {
                out.push(e);
                cur = [c1(e[0]), c1(e[1])];
            }
            Prim::Cubic(_, p1, p2, p3) => {
                pts28.clear();
                let c = |p: Point| [c1(p[0]), c1(p[1])];
                milbezier::flatten_cubic_scaled(cur, c(p1), c(p2), c(p3), tol, clip, &mut pts28);
                for &p in &pts28 {
                    out.push(back(p));
                }
                if let Some(&last) = pts28.last() {
                    cur = last;
                }
            }
        }
    }
    out
}

/// One device-space fill figure flattened ([`flatten_fill_prims_scale`]), through native's strip clip
/// first ([`crate::pixels::clip_figure`]) when its shape `clip`s ([`Geometry::clips_huge`]):
/// The safe-box clip clips a curve before the fill flattener sees it. `None` drops a figure the strips
/// never touch. An infinite crossing becomes [`crate::pixels::FAR`]; a NaN one stays, and voids the
/// whole fill downstream.
pub(crate) fn fill_figure_pts(start: Point, prims: &[Prim], fill_scale: f32, clip: bool, target: Option<[i32; 4]>) -> Option<Vec<Point>> {
    use crate::pixels::Seg;
    if !clip {
        return Some(flatten_fill_prims_scale(start, prims, fill_scale, target));
    }
    let segs: Vec<Seg> = prims.iter().map(|pr| match *pr { Prim::Line(_, e) => Seg::Line(e), Prim::Cubic(_, a, b, c) => Seg::Cubic(a, b, c) }).collect();
    let (s, segs) = crate::pixels::clip_figure(start, &segs)?;
    let far = |p: Point| if p.iter().any(|v| v.is_infinite()) && !p.iter().any(|v| v.is_nan()) { crate::pixels::FAR } else { p };
    let mut prev = far(s);
    let prims: Vec<Prim> = segs
        .iter()
        .map(|g| {
            let pr = match *g { Seg::Line(e) => Prim::Line(prev, far(e)), Seg::Cubic(a, b, c) => Prim::Cubic(prev, a, b, c) };
            prev = match pr { Prim::Line(_, e) | Prim::Cubic(_, _, _, e) => e };
            pr
        })
        .collect();
    Some(flatten_fill_prims_scale(far(s), &prims, fill_scale, target))
}

impl Geometry {
    /// The filled region under `m`, flattened as a fill and cut to the axis-aligned `rect` of the space
    /// `m` maps into: a masked layer's clip geometry, the mask strip-clipped to the content bounds.
    pub(crate) fn clip_to_rect(&self, m: &Matrix, rect: [f32; 4]) -> Geometry {
        use crate::pixels::Seg;
        // The cut takes the mask's lines and curves as they are: the strip clipper flattens only a curve that
        // crosses the rect, and passes the others through for the fill to flatten (r27 32360, 32923).
        let prims: Vec<(Point, Vec<Prim>)> = match self {
            Geometry::Path { figures, .. } => figures.iter().filter(|f| f.filled && !f.has_nonfinite()).map(|f| figure_prims(f, m)).collect(),
            _ => self.prims_per_figure(m).into_iter().map(|(s, p, _, _)| (s, p)).collect(),
        };
        let figures = prims
            .into_iter()
            .filter_map(|(start, prims)| {
                let segs: Vec<Seg> = prims.iter().map(|pr| match *pr { Prim::Line(_, e) => Seg::Line(e), Prim::Cubic(_, a, b, c) => Seg::Cubic(a, b, c) }).collect();
                let (start, segs) = crate::pixels::clip_figure_rect(start, &segs, rect)?;
                let segments = segs.iter().map(|g| match *g { Seg::Line(e) => Segment::Line(e), Seg::Cubic(a, b, c) => Segment::Cubic(a, b, c) }).collect();
                Some(Figure { start, filled: true, closed: true, segments, seg_flags: Vec::new() })
            })
            .collect();
        Geometry::Path { fill: self.fill_mode(), figures }
    }

    /// A path figure with an arc segment whose radius has a negative component: native's geometry
    /// sink rejects it with `E_INVALIDARG` and never creates the geometry (`ID2D1GeometrySink::Close`
    /// fails). VM-probed decisively — a negative arc radius fails, a non-finite *point* does not
    /// (fuzz seeds 20003/20041 reject; 20015, with positive radii and inf points, is accepted).
    /// Rect/rounded/ellipse radii are a separate validation and not covered here.
    pub fn has_invalid_arc_radius(&self) -> bool {
        let Geometry::Path { figures, .. } = self else { return false };
        figures
            .iter()
            .flat_map(|f| &f.segments)
            .any(|s| matches!(s, Segment::Arc { radius, .. } if radius[0] < 0.0 || radius[1] < 0.0))
    }

    /// Flatten to device-space polylines under `transform` (world/DPI already composed) at
    /// `tolerance` (normalized here, as every native compute path does).
    pub fn flatten(&self, transform: &Matrix, tolerance: f32) -> Vec<Polyline> {
        let tol = normalize_tolerance(tolerance);
        let mut out = Vec::new();
        self.flatten_into(transform, tol, &mut out);
        out
    }

    /// Whether the scanner rejects this shape under the device transform `m` with `0x88990011`
    /// (`D2DERR_BAD_NUMBER`): [`workspace_rejects`] over its analytic tight bounds and the points the
    /// scanner converts ([`Self::scan_points`]; VM-trapped: arc 7443's control points overflow the fixed
    /// workspace). Shared by the geometry
    /// realization, the stroke widener and the scanned geometry queries.
    pub fn workspace_bad_number(&self, m: &Matrix) -> bool {
        // The ANALYTIC device-space bounds, which stay small for a near-degenerate curve whose
        // flattening overshoots.
        workspace_rejects(self.get_bounds(m), self.scan_points(m))
    }

    /// The points the scanner converts into its workspace: each figure's start, line ends and curve
    /// control points (it flattens curves on the grid afterwards). A curve far out at x = 3.4e38 whose
    /// points all snap to the centre scans fine (r23 28433), where an f32 flattening there overflows.
    fn scan_points(&self, m: &Matrix) -> Vec<Point> {
        let mut pts = Vec::new();
        for (start, prims, ..) in self.prims_per_figure(m) {
            pts.push(start);
            for p in prims {
                match p {
                    Prim::Line(_, b) => pts.push(b),
                    Prim::Cubic(_, c1, c2, e) => pts.extend([c1, c2, e]),
                }
            }
        }
        pts
    }

    /// The points the scanned queries must hold in their workspace: each figure's start and segment
    /// ends. The workspace range-checks starts and line ends; a curve goes to the scanner unchecked and
    /// is flattened there, where only its on-curve points (within its bounds unless non-finite) count,
    /// so its control points do not.
    fn scan_line_points(&self, m: &Matrix) -> Vec<Point> {
        let mut pts = Vec::new();
        for (start, prims, ..) in self.prims_per_figure(m) {
            pts.push(start);
            pts.extend(prims.iter().map(|p| match *p {
                Prim::Line(_, b) | Prim::Cubic(_, _, _, b) => b,
            }));
        }
        pts
    }

    /// Whether `Tessellate` under `m` fails with `D2DERR_BAD_NUMBER`. The tessellator's workspace is set
    /// from the shape's local bounds, hollow figures included, while the points stream through `m`.
    /// Empty or NaN bounds tessellate nothing; otherwise a non-finite point reaching the sink, or the
    /// workspace rejecting a point ([`combine::workspace::Workspace::tessellation_rejects`]), fails it.
    pub fn tessellate_bad_number(&self, m: &Matrix) -> bool {
        let b = self.bounds_of(&IDENTITY, true);
        if !(b[2] >= b[0] && b[3] >= b[1]) {
            return false;
        }
        self.any_nonfinite_fill_void() || combine::workspace::Workspace::tessellation_rejects(b, self.scan_line_points(m))
    }

    /// Whether scanning this shape fails with `D2DERR_BAD_NUMBER`: a non-finite point reaches the
    /// scanner's sink ([`Self::any_nonfinite_fill_void`]: any in a filled figure, an infinite on-curve
    /// point in a hollow one), or its workspace rejects ([`Self::workspace_bad_number`]).
    fn scan_bad_number(&self, m: &Matrix) -> bool {
        self.any_nonfinite_fill_void() || self.workspace_bad_number(m)
    }

    /// Whether `ComputeArea` fails with `D2DERR_BAD_NUMBER`: a rectangle's area is closed-form (never
    /// fails, NaN included); any other shape is scanned.
    pub fn area_bad_number(&self, m: &Matrix) -> bool {
        // The scan's workspace comes from the fill bounds under `m`, hollow figures included.
        self.as_axis_aligned_rect().is_none() && (self.any_nonfinite_fill_void() || workspace_rejects(self.bounds_of(m, true), self.scan_line_points(m)))
    }

    /// Whether `Outline` fails with `D2DERR_BAD_NUMBER`: the shape is scanned in the workspace of its
    /// fill bounds under `m` with hollow figures included.
    pub fn outline_bad_number(&self, m: &Matrix) -> bool {
        self.any_nonfinite_fill_void() || workspace_rejects(self.bounds_of(m, true), self.scan_line_points(m))
    }

    /// Whether `CombineWithGeometry` fails with `D2DERR_BAD_NUMBER`: both shapes are scanned in one
    /// workspace over the union of their tight bounds (an empty shape contributes none).
    pub fn combine_bad_number(&self, other: &Geometry, m: &Matrix, m_other: &Matrix) -> bool {
        if self.any_nonfinite_fill_void() || other.any_nonfinite_fill_void() {
            return true;
        }
        // Each shape's fill bounds under its matrix, hollow figures included.
        let u = scan_union(self.bounds_of(m, true), other.bounds_of(m_other, true));
        workspace_rejects(u, self.scan_line_points(m).into_iter().chain(other.scan_line_points(m_other)))
    }

    /// Whether `FillContainsPoint(pt, m)` fails with `D2DERR_BAD_NUMBER`. Native streams the filled
    /// figures to its hit tester, which takes each point through `m` with the query point subtracted, in
    /// double, and fails when a coordinate comes out NaN: a NaN point, or an infinite one meeting a zero
    /// matrix entry (`0·∞`, so any infinite point under an axis-aligned `m`) or an infinite translation
    /// of the opposite sign. Only points on the outline count: an infinite curve control does not fail
    /// (native probe).
    pub fn fill_contains_bad_number(&self, pt: Point, m: &Matrix, tolerance: f32) -> bool {
        let (dx, dy) = ((m[4] - pt[0]) as f64, (m[5] - pt[1]) as f64);
        let rel = |p: Point| {
            let (x, y) = (p[0] as f64, p[1] as f64);
            [m[0] as f64 * x + m[2] as f64 * y + dx, m[1] as f64 * x + m[3] as f64 * y + dy]
        };
        let nan = |q: [f64; 2]| q[0].is_nan() || q[1].is_nan();
        // A curve is flattened in those relative coordinates at the normalized tolerance and each
        // flattened point tested: an infinite coordinate there turns its differences NaN.
        let err = normalize_tolerance(tolerance).max(0.0) as f64 * 6.0;
        let mut figs = Vec::new();
        self.filled_figures(&mut figs);
        for (start, prims) in figs {
            let mut cur = rel(start);
            if nan(cur) {
                return true;
            }
            for pr in prims {
                match pr {
                    Prim::Line(_, b) => cur = rel(b),
                    Prim::Cubic(_, c1, c2, e) => {
                        let mut pts = Vec::new();
                        stroke::flatten_cubic_tan(cur, rel(c1), rel(c2), rel(e), &mut pts, err);
                        if pts.iter().any(|&(q, _)| nan(q)) {
                            return true;
                        }
                        cur = rel(e);
                    }
                }
                if nan(cur) {
                    return true;
                }
            }
        }
        false
    }

    /// The filled figures as the shape streams them, untransformed (quadratics and arcs raised to
    /// cubics; a rounded rectangle's or ellipse's NaN radius kept).
    fn filled_figures(&self, out: &mut Vec<(Point, Vec<Prim>)>) {
        match self {
            Geometry::Rect([l, t, r, b]) => {
                let c = [[*l, *t], [*r, *t], [*r, *b], [*l, *b]];
                out.push((c[0], (0..3).map(|i| Prim::Line(c[i], c[i + 1])).collect()));
            }
            Geometry::Rounded { rect, radius } => out.push(rounded_rect_figure(*rect, *radius)),
            Geometry::Ellipse { center, radius } => out.push(rounded_rect_figure(ellipse_rect(*center, *radius), *radius)),
            Geometry::Path { figures, .. } => {
                for fig in figures.iter().filter(|f| f.filled) {
                    out.push(figure_prims(fig, &IDENTITY));
                }
            }
            Geometry::Group { children, .. } => children.iter().for_each(|c| c.filled_figures(out)),
            Geometry::Transformed { matrix, child } => {
                let mut inner = Vec::new();
                child.filled_figures(&mut inner);
                out.extend(inner.into_iter().map(|(s, pr)| (transform_point(matrix, s), transform_prims(&pr, matrix))));
            }
        }
    }

    /// Whether `CompareWithGeometry` fails with `D2DERR_BAD_NUMBER`: an empty shape, or bounds that do
    /// not overlap, settle the relation without a scan; otherwise both shapes are scanned in one workspace
    /// over the union of their bounds. A NaN point turns the bounds NaN, which cannot prove the shapes
    /// disjoint, so the scan runs and rejects it; an infinite one can still leave them apart.
    pub fn compare_bad_number(&self, other: &Geometry, m_other: &Matrix) -> bool {
        // Each shape's fill bounds, hollow figures included: shapes found apart (or empty) report
        // disjoint before either is scanned.
        let (a, b) = (self.bounds_of(&IDENTITY, true), other.bounds_of(m_other, true));
        // Neither shape's bounds may hold a NaN, whether or not they settle the relation.
        if a.iter().chain(&b).any(|v| v.is_nan()) {
            return true;
        }
        let empty = |r: [f32; 4]| r[0] > r[2] || r[1] > r[3];
        if empty(a) || empty(b) || a[2] < b[0] || b[2] < a[0] || a[3] < b[1] || b[3] < a[1] {
            return false;
        }
        if self.any_nonfinite_fill_void() || other.any_nonfinite_fill_void() {
            return true;
        }
        workspace_rejects(scan_union(a, b), self.scan_line_points(&IDENTITY).into_iter().chain(other.scan_line_points(m_other)))
    }

    /// Whether `CreateFilled`/`CreateStrokedGeometryRealization` rejects this geometry with `0x88990011`
    /// (`D2DERR_BAD_NUMBER`). A geometry realization bakes in DIP space at the IDENTITY transform —
    /// Direct2D.s geometry realization ignores the context transform (a context dy=−3.4e38
    /// at 192 dpi must NOT false-reject, r13 18223), so the reject is on the geometry's own coordinates.
    /// Either the per-point scanner reject ([`Geometry::workspace_bad_number`] — a non-finite / out-of-range
    /// flattened point, incl. the near-degenerate arc), or the GetExtents non-finite extent from the f32
    /// strip-clip overflow past the ±524287 safe box (`clips_huge` + `clip_figure`'s raw non-finite crossing
    /// — the 3.4e38 case whose unclipped points are still finite). Reads the raw crossing clip_figure
    /// returns; the fill/stroke `FAR` mapping that callers apply is untouched.
    /// A filled realization's bounds skip hollow figures, as native's fill bounds do, so only the filled ones
    /// can leave it empty; every figure, hollow ones included, then goes through the safe-box clip and the scanner
    /// (r15 20390: a NaN arc in a hollow figure beside a finite filled one fails the create).
    pub fn realize_bad_number(&self, tolerance: f32) -> bool {
        let filled = realization::filled_only(self);
        let fill = filled.as_ref().unwrap_or(self);
        if fill.realize_empty() {
            return false;
        }
        // A hollow figure reaches the scanner as it is, unclipped, in the workspace the filled figures' bounds set
        // (clipped to the safe box when they pass it). The scanner range-checks each figure's start and line ends
        // and flattens curves unchecked, so a NaN, infinite or huge line point there fails, a curve's does not
        // (VM probes p390: a hollow line to NaN, inf or -3.4e38 fails beside a finite fill; a NaN cubic or arc does not).
        let mut hollow = Vec::new();
        self.hollow_line_points(&IDENTITY, &mut hollow);
        let b = if fill.clips_huge(&IDENTITY) { fill.control_clipped().bounds_of(&IDENTITY, false) } else { fill.bounds_of(&IDENTITY, false) };
        (!hollow.is_empty() && workspace_rejects(b, hollow)) || self.realize_scan_bad_number(tolerance)
    }

    /// This shape with each quadratic or cubic whose control is NaN and whose end is not replaced by a line to its end
    /// (see [`Self::stroked_realize_bad_number`]).
    fn nan_controls_as_lines(&self) -> Geometry {
        let nan = |p: &Point| p[0].is_nan() || p[1].is_nan();
        match self {
            Geometry::Path { fill, figures } => Geometry::Path {
                fill: *fill,
                figures: figures
                    .iter()
                    .map(|f| Figure {
                        segments: f
                            .segments
                            .iter()
                            .map(|s| match *s {
                                Segment::Cubic(a, b, to) if (nan(&a) || nan(&b)) && !nan(&to) => Segment::Line(to),
                                Segment::Quadratic(a, to) if nan(&a) && !nan(&to) => Segment::Line(to),
                                other => other,
                            })
                            .collect(),
                        ..f.clone()
                    })
                    .collect(),
            },
            Geometry::Group { fill, children } => Geometry::Group { fill: *fill, children: children.iter().map(|c| c.nan_controls_as_lines()).collect() },
            Geometry::Transformed { matrix, child } => Geometry::Transformed { matrix: *matrix, child: Box::new(child.nan_controls_as_lines()) },
            other => other.clone(),
        }
    }

    /// Each hollow path figure's start and line ends under `m` (see [`Self::realize_bad_number`]).
    fn hollow_line_points(&self, m: &Matrix, out: &mut Vec<Point>) {
        match self {
            Geometry::Path { figures, .. } => {
                for f in figures.iter().filter(|f| !f.filled) {
                    out.push(transform_point(m, f.start));
                    out.extend(f.segments.iter().filter_map(|g| match g {
                        Segment::Line(p) => Some(transform_point(m, *p)),
                        _ => None,
                    }));
                }
            }
            Geometry::Group { children, .. } => children.iter().for_each(|c| c.hollow_line_points(m, out)),
            Geometry::Transformed { matrix, child } => child.hollow_line_points(&mat_mul(matrix, m), out),
            _ => {}
        }
    }

    /// [`Self::realize_bad_number`] past the empty test: the safe-box clip and the scan.
    fn realize_scan_bad_number(&self, _tolerance: f32) -> bool {
        use crate::pixels::Seg;
        let m = &IDENTITY;
        // Native clips the shape to the safe device box (the safe-device-box clip) BEFORE
        // the scanner's bad-number check, which clamps curve CONTROLS: a finite-huge control then bulges
        // only to the clamped control, so the raw float flatten no longer overflows to `inf` (which
        // tripped a false workspace reject) and the strip-clip crossing stays finite (fuzz r17 22736: a
        // `-3.4e38` quad control realizes S_OK). A huge ON-CURVE point is not a control, so it is left
        // raw and still overflows the crossing → the real BAD_NUMBER. When the shape does not clip
        // (`!clips_huge`), the scanner runs on it as-is (the near-degenerate flatten-overshoot case).
        let clipped;
        let shape: &Geometry = if self.clips_huge(m) {
            clipped = self.control_clipped();
            &clipped
        } else {
            self
        };
        if shape.workspace_bad_number(m) {
            return true;
        }
        if shape.clips_huge(m) {
            let fin = |q: [f32; 2]| q[0].is_finite() && q[1].is_finite();
            // The strip clipper takes the shape's curves as curves, flattening only one the cut crosses: a corner
            // arc wholly past the box drops out unflattened, where an f32 flatten of it at -3.4e38 overflows to NaN
            // (r11 16792: a rounded rect with a -3.4e38 left edge realizes S_OK).
            for (start, prims, ..) in shape.prims_per_figure(m) {
                if prims.is_empty() {
                    continue;
                }
                let segs: Vec<Seg> = prims.iter().map(|p| match *p { Prim::Line(_, b) => Seg::Line(b), Prim::Cubic(_, a, b, c) => Seg::Cubic(a, b, c) }).collect();
                match crate::pixels::clip_figure(start, &segs) {
                    None => return true, // a NaN crossing voids the figure ⇒ non-finite extent
                    Some((s, segs)) => {
                        if !fin(s) || segs.iter().any(|g| !fin(match *g { Seg::Line(q) | Seg::Cubic(_, _, q) => q })) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// The geometry a realization bakes: an empty one where [`Self::realize_empty`] holds (the empty
    /// S_OK realization draws nothing: fuzz r17 22393, a -inf arc rotation), else this geometry.
    pub fn realized(&self) -> Geometry {
        if self.realize_empty() { Geometry::Path { fill: FillMode::Alternate, figures: Vec::new() } } else { self.clone() }
    }

    /// Direct2D's geometry realization bakes an EMPTY realization, S_OK, when the
    /// shape's bounds compare inverted (a NaN anywhere, or no points: the empty-bounds sentinel) or
    /// degenerate. the safe-device-box clip skips a degenerate box and the check after it
    /// rejects it: an inf-centre ellipse or an all-±inf figure lies on a line at infinity
    /// (realize-nonfinite-* probes; fuzz r11 16092). A ±inf rect or rounded-rect edge keeps a proper box
    /// and is clipped to the safe bounds instead ([`Self::clips_huge`]; r11 16190, 16461).
    /// ponytail: and a shape whose ±inf sits only in rect/rounded-rect/ellipse parameters clips to a
    /// finite primitive (S_OK: fuzz r11 16166, 16190, 16461), where our flatten+strip clip of it would
    /// overflow; a ±inf PATH point keeps the workspace/widen tests (r11 16683 is BAD). Port
    /// the safe-device-box clip to take both from one clip.
    pub fn realize_empty(&self) -> bool {
        let b = self.get_bounds(&IDENTITY);
        self.raw_shape_has_nan() || !(b[2] > b[0] && b[3] > b[1]) || (self.any_nonfinite() && !self.path_nonfinite())
    }

    /// A non-finite position inside a Path figure (not a rect/rounded-rect/ellipse parameter).
    fn path_nonfinite(&self) -> bool {
        match self {
            Geometry::Rect(_) | Geometry::Rounded { .. } | Geometry::Ellipse { .. } => false,
            Geometry::Path { figures, .. } => figures.iter().any(|f| f.has_nonfinite_pos()),
            Geometry::Group { children, .. } => children.iter().any(|c| c.path_nonfinite()),
            Geometry::Transformed { child, .. } => child.path_nonfinite(),
        }
    }

    fn flatten_into(&self, m: &Matrix, tol: f32, out: &mut Vec<Polyline>) {
        match self {
            Geometry::Rect(rect) => {
                let [l, t, r, b] = ordered_rect(*rect);
                let tp = |p| transform_point(m, p);
                out.push(Polyline {
                    pts: vec![tp([l, t]), tp([r, t]), tp([r, b]), tp([l, b])],
                    closed: true,
                    filled: true,
                });
            }
            Geometry::Rounded { rect, radius } => {
                let (start, prims) = rounded_rect_prims(*rect, *radius);
                let prims = transform_prims(&prims, m);
                out.push(Polyline { pts: flatten_prims(map_point(m, start), &prims, tol), closed: true, filled: true });
            }
            Geometry::Ellipse { center, radius } => {
                let rect = [center[0] - radius[0], center[1] - radius[1], center[0] + radius[0], center[1] + radius[1]];
                let (start, prims) = rounded_rect_prims(rect, *radius);
                let prims = transform_prims(&prims, m);
                out.push(Polyline { pts: flatten_prims(map_point(m, start), &prims, tol), closed: true, filled: true });
            }
            Geometry::Path { figures, .. } => {
                for fig in figures.iter().filter(|f| !f.has_nonfinite()) {
                    let (start, prims) = figure_prims(fig, m);
                    out.push(Polyline { pts: flatten_prims(start, &prims, tol), closed: fig.closed, filled: fig.filled });
                }
            }
            Geometry::Group { children, .. } => {
                for c in children {
                    c.flatten_into(m, tol, out);
                }
            }
            Geometry::Transformed { matrix, child } => {
                child.flatten_into(&mat_mul(matrix, m), tol, out);
            }
        }
    }

    /// This geometry with every cubic/quad CONTROL point clamped to the safe device box (±524287), as
    /// native's realize clips curve controls before the scanner. ON-CURVE points (figure
    /// start, segment endpoints, arc radii) are left raw, so a huge endpoint still overflows the
    /// strip-clip crossing and voids (the real BAD_NUMBER). Used only by [`realize_bad_number`]; a
    /// finite-huge control (fuzz r17 22736: `-3.4e38` quad control) is then bounded → the flatten no
    /// longer overflows to `inf` and the realization is S_OK.
    fn control_clipped(&self) -> Geometry {
        let c = |q: Point| [q[0].clamp(-524_287.0, 524_287.0), q[1].clamp(-524_287.0, 524_287.0)];
        match self {
            Geometry::Path { fill, figures } => Geometry::Path {
                fill: *fill,
                figures: figures
                    .iter()
                    .map(|f| Figure {
                        start: f.start,
                        filled: f.filled,
                        closed: f.closed,
                        seg_flags: f.seg_flags.clone(),
                        segments: f
                            .segments
                            .iter()
                            .map(|s| match *s {
                                Segment::Cubic(a, b, to) => Segment::Cubic(c(a), c(b), to),
                                Segment::Quadratic(a, to) => Segment::Quadratic(c(a), to),
                                // A huge/non-finite arc RADIUS makes `arc_to_cubics` emit huge control
                                // points whose float flatten overflows; the arc is near-straight at that
                                // radius and native's box-clip keeps only its finite near-endpoint part,
                                // so degenerate it to a line to its endpoint for the finiteness check
                                // (fuzz r17 22736 fig1: a `3.4e38` arc radius).
                                Segment::Arc { to, radius, .. } if !(radius[0].abs() <= 524_287.0 && radius[1].abs() <= 524_287.0) => Segment::Line(to),
                                other => other,
                            })
                            .collect(),
                    })
                    .collect(),
            },
            Geometry::Group { fill, children } => Geometry::Group { fill: *fill, children: children.iter().map(|ch| ch.control_clipped()).collect() },
            Geometry::Transformed { matrix, child } => Geometry::Transformed { matrix: *matrix, child: Box::new(child.control_clipped()) },
            other => other.clone(),
        }
    }

    /// Flatten to device-space polylines the way native **fills** do: through d2d1's 28.4 edge sink
    /// (the shared edge-list adapter, used by the CPU and WARP geometry rasterizers), whose
    /// beziers go through the MIL integer cracker ([`flatten_fill_prims`]).
    pub fn flatten_fill(&self, transform: &Matrix) -> Vec<Polyline> {
        self.flatten_fill_on(transform, None)
    }

    /// [`flatten_fill`] for a COPY fill whose edge list is clipped to the pixel rect `target` (see
    /// [`flatten_fill_prims_scale`]).
    pub fn flatten_fill_on(&self, transform: &Matrix, target: Option<[i32; 4]>) -> Vec<Polyline> {
        let mut out = Vec::new();
        self.flatten_fill_into(transform, target, &mut out);
        out
    }

    fn flatten_fill_into(&self, m: &Matrix, target: Option<[i32; 4]>, out: &mut Vec<Polyline>) {
        match self {
            Geometry::Rect(rect) => {
                let [l, t, r, b] = ordered_rect(*rect);
                // The rect streams into the strip clip through the device transform's sink: its first
                // corner by the full product, the others by the matrix's form, so an infinite edge under a
                // scale stays infinite rather than meeting a zero entry.
                let (ts, tp) = (|p| map_point(m, p), |p| sink_point(m, p));
                let pts = vec![ts([l, t]), tp([r, t]), tp([r, b]), tp([l, b])];
                // Past the safe box a rect takes the same strip clip as any shape (r20 25198: a left of
                // -3.4e38 crosses to the far top-left and fills the slope-1 wedge native draws).
                if self.clips_huge(m) {
                    let prims: Vec<Prim> = (0..4).map(|i| Prim::Line(pts[i], pts[(i + 1) % 4])).collect();
                    out.extend(fill_figure_pts(pts[0], &prims, 16.0, true, target).map(|pts| Polyline { pts, closed: true, filled: true }));
                } else {
                    out.push(Polyline { pts, closed: true, filled: true });
                }
            }
            Geometry::Rounded { rect, radius } => {
                let (start, prims) = rounded_rect_prims(*rect, *radius);
                let prims = transform_prims(&prims, m);
                let clip = self.clips_huge(m);
                out.extend(fill_figure_pts(map_point(m, start), &prims, 16.0, clip, target).map(|pts| Polyline { pts, closed: true, filled: true }));
            }
            Geometry::Ellipse { center, radius } => {
                // Direct2D.s float bezier flattener builds the transformed ellipse arc's
                // forward-difference second-difference vectors and its error test is the component max
                // `fmaxf(|6·Δ²x|, |6·Δ²y|)`, not a length. For the ellipse the dominant second-difference
                // magnitude is 6·kappa·(device semi-axis) (kappa = the 90°-arc tangent factor 0.5522847):
                // an SSH probe ladder pinned the flip at FLT_MAX/(6·kappa) = 1.0269e38 EXACTLY (1.02e38
                // draws, 1.027e38 nothing). Each device semi-axis component is a product that overflows to
                // ±inf at a near-FLT_MAX radius, so native's forward difference goes non-finite and it
                // draws NOTHING. Reproduce with native's own arithmetic: the component max — NOT a hypot
                // (which never overflows) and NOT sqrt(x²+y²) (which would overflow ~1.8e19, far too
                // early). Our rounded-rect control points give a 2.686·R second difference, not native's
                // 6·kappa·R arc, so scale the magnitude directly. Sign-independent: |radius| drives it, so
                // a small negative radius still draws when finite.
                const KAPPA: f32 = 0.5522847498307933;
                let ax = (radius[0] * m[0]).abs().max((radius[0] * m[1]).abs());
                let ay = (radius[1] * m[2]).abs().max((radius[1] * m[3]).abs());
                if !(6.0 * KAPPA * ax).is_finite() || !(6.0 * KAPPA * ay).is_finite() {
                    return;
                }
                let rect = [center[0] - radius[0], center[1] - radius[1], center[0] + radius[0], center[1] + radius[1]];
                let (start, prims) = rounded_rect_prims(rect, *radius);
                let prims = transform_prims(&prims, m);
                let clip = self.clips_huge(m);
                out.extend(fill_figure_pts(map_point(m, start), &prims, 16.0, clip, target).map(|pts| Polyline { pts, closed: true, filled: true }));
            }
            Geometry::Path { figures, .. } => {
                let clip = self.clips_huge(m);
                for fig in figures.iter().filter(|f| !f.has_nonfinite()) {
                    let (start, prims) = figure_prims(fig, m);
                    out.extend(fill_figure_pts(start, &prims, 16.0, clip, target).map(|pts| Polyline { pts, closed: fig.closed, filled: fig.filled }));
                }
            }
            Geometry::Group { children, .. } => {
                for c in children {
                    c.flatten_fill_into(m, target, out);
                }
            }
            Geometry::Transformed { matrix, child } => {
                child.flatten_fill_into(&mat_mul(matrix, m), target, out);
            }
        }
    }

    /// The fill rule governing this geometry's filled region.
    pub fn fill_mode(&self) -> FillMode {
        match self {
            Geometry::Path { fill, .. } | Geometry::Group { fill, .. } => *fill,
            Geometry::Transformed { child, .. } => child.fill_mode(),
            _ => FillMode::Winding, // rect/rounded/ellipse are simple, single-contour shapes
        }
    }
}

/// Every join of a rounded rectangle or ellipse figure is round: native sends those shapes with
/// `D2D1_PATH_SEGMENT_FORCE_ROUND_LINE_JOIN` on all their segments, which only shows where a join
/// actually turns (a zero radius collapsing the shape into cusps, r20 25700).
/// A rounded rectangle's (or ellipse's) figure as the widener receives it, local and under `m`: the
/// shape streams four corner beziers and the three sides between them, then closes the figure, so its
/// fourth side is a closing edge, which the widener adds only when the last corner ends away from the
/// start. An ellipse's sides have no length: its figure closes on its last corner (with every corner
/// skipped as degenerate, the skipped corners' sides draw a diamond whose fourth edge native never draws).
fn shape_stroke_prims(mut local: Vec<Prim>, m: &Matrix) -> (Vec<Prim>, Vec<Prim>) {
    let mut pr = transform_prims(&local, m);
    if let Some(&Prim::Line(a, b)) = pr.last() {
        if a == b {
            local.pop();
            pr.pop();
        }
    }
    (local, pr)
}

fn prim_ends(prims: &[Prim]) -> Vec<Point> {
    prims.iter().map(|p| match *p { Prim::Line(_, b) | Prim::Cubic(_, _, _, b) => b }).collect()
}

/// A shape point under the transform as native's sink chain maps it: an identity transform leaves the
/// point untouched rather than multiplying through it, which only a −0 or non-finite coordinate can
/// tell apart (`−0 + 0` is +0, `0·inf` is NaN).
fn map_point(m: &Matrix, p: Point) -> Point {
    if *m == IDENTITY { p } else { transform_point(m, p) }
}

/// A shape's point through the transforming sinks of the transformed shapes it sits in (innermost
/// first).
fn pre_point(pre: &[Matrix], p: Point) -> Point {
    pre.iter().fold(p, |q, m| sink_point(m, q))
}

fn pre_prims(pre: &[Matrix], prims: &[Prim]) -> Vec<Prim> {
    if pre.is_empty() {
        return prims.to_vec();
    }
    let tp = |p| pre_point(pre, p);
    prims
        .iter()
        .map(|p| match *p {
            Prim::Line(a, b) => Prim::Line(tp(a), tp(b)),
            Prim::Cubic(a, b, c, d) => Prim::Cubic(tp(a), tp(b), tp(c), tp(d)),
        })
        .collect()
}

/// A path figure as it streams out of the transformed shapes it sits in: its arcs and quadratics
/// become the cubics the shape builder made of them in its own space, every point goes through the
/// sinks, and each segment's flags carry over to the curves it became.
fn pre_figure(pre: &[Matrix], fig: &Figure) -> Figure {
    if pre.is_empty() {
        return fig.clone();
    }
    let (mut segments, mut seg_flags) = (Vec::new(), Vec::new());
    let mut cur = fig.start;
    for (si, seg) in fig.segments.iter().enumerate() {
        let one = Figure { start: cur, filled: fig.filled, closed: false, segments: vec![*seg], seg_flags: Vec::new() };
        let f = fig.seg_flags.get(si).copied();
        for pr in figure_prims(&one, &IDENTITY).1 {
            segments.push(match pr {
                Prim::Line(_, e) => Segment::Line(pre_point(pre, e)),
                Prim::Cubic(_, c1, c2, e) => Segment::Cubic(pre_point(pre, c1), pre_point(pre, c2), pre_point(pre, e)),
            });
            if let Some(f) = f {
                seg_flags.push(f);
            }
        }
        cur = segment_end(seg);
    }
    if let Some(&f) = fig.seg_flags.get(fig.segments.len()) {
        seg_flags.push(f);
    }
    Figure { start: pre_point(pre, fig.start), filled: fig.filled, closed: fig.closed, segments, seg_flags }
}

/// A point through the transforming sink a shape streams into under a matrix, which picks its
/// routine by the matrix's form: an identity passes the point through, a translation adds, and a scale
/// and translation (no rotation or shear) multiplies each axis by its own entry only, so an infinite
/// coordinate never meets the other axis's zero entry; anything else takes the full product.
pub(crate) fn sink_point(m: &Matrix, [x, y]: Point) -> Point {
    if m[1] != 0.0 || m[2] != 0.0 {
        transform_point(m, [x, y])
    } else if m[0] == 1.0 && m[3] == 1.0 {
        if m[4] == 0.0 && m[5] == 0.0 { [x, y] } else { [x + m[4], y + m[5]] }
    } else {
        [x * m[0] + m[4], y * m[3] + m[5]]
    }
}

pub(crate) fn transform_prims(prims: &[Prim], m: &Matrix) -> Vec<Prim> {
    let tp = |p| map_point(m, p);
    prims
        .iter()
        .map(|p| match *p {
            Prim::Line(a, b) => Prim::Line(tp(a), tp(b)),
            Prim::Cubic(a, b, c, d) => Prim::Cubic(tp(a), tp(b), tp(c), tp(d)),
        })
        .collect()
}

// --- queries ---------------------------------------------------------------------------------

impl Geometry {
    /// `ComputeArea`: the axis-aligned rectangle fast path (`|Δx·Δy|·|det|`); otherwise the area of
    /// the filled region under the fill rule, as the region scan resolves it ([`combine::area`]).
    pub fn compute_area(&self, transform: &Matrix, tolerance: f32) -> f32 {
        // A shape the widener takes as a rectangle has a closed-form area: the rectangle's times the
        // transform's determinant.
        if let Some([l, t, r, b]) = self.as_axis_aligned_rect() {
            let base = ((r - l) * (b - t)).abs();
            let det = (transform[0] * transform[3] - transform[2] * transform[1]).abs();
            return base * det;
        }
        combine::area(self, transform, tolerance)
    }

    /// The rectangle this shape is, if any: a rectangle's ordered rect, or, for a transformed shape
    /// whose matrix keeps the axes aligned, its child's rectangle mapped through the matrix (an
    /// inverted one passes through; a NaN corner coordinate turns its axis NaN).
    fn as_axis_aligned_rect(&self) -> Option<[f32; 4]> {
        match self {
            Geometry::Rect(rect) => Some(ordered_rect(*rect)),
            Geometry::Transformed { matrix: m, child } if (m[1] == 0.0 && m[2] == 0.0) || (m[0] == 0.0 && m[3] == 0.0) => {
                child.as_axis_aligned_rect().map(|r| if r[0] > r[2] || r[1] > r[3] { r } else { map_bounds(r, m) })
            }
            _ => None,
        }
    }

    /// `ComputeLength`: f32 running sum of `sqrt(dy²+dx²)` over the flattened polylines, adding the
    /// closing edge for closed figures (the closing-edge pass).
    /// `ComputeLength`: native's length walk flattens each curve at the RAW tolerance (no
    /// normalization: zero, a negative or NaN flattens as finely as the flattener steps), and skips a
    /// curve whose first tangent fails under the length fuzz (a huge tolerance drops every curve), leaving
    /// the current point at its end. Its flattener takes a NaN second difference as flat
    /// ([`flatten::flatten_cubic_sink`]).
    pub fn compute_length(&self, transform: &Matrix, tolerance: f32) -> f32 {
        let mut total = 0.0f32;
        let mut pts = Vec::new();
        let mut figs = Vec::new();
        self.length_figures(transform, &mut figs);
        for (start, prims) in figs {
            let mut cur = start;
            for pr in &prims {
                match *pr {
                    Prim::Line(_, to) => {
                        let (dx, dy) = (to[0] - cur[0], to[1] - cur[1]);
                        total = (dy * dy + dx * dx).sqrt() + total;
                        cur = to;
                    }
                    Prim::Cubic(_, p1, p2, p3) => {
                        if !flatten::first_tangent_ok(cur, p1, p2, p3, tolerance) {
                            cur = p3;
                            continue;
                        }
                        pts.clear();
                        flatten::flatten_cubic_sink(cur, p1, p2, p3, tolerance, &mut pts);
                        for &q in &pts {
                            let (dx, dy) = (q[0] - cur[0], q[1] - cur[1]);
                            total = (dy * dy + dx * dx).sqrt() + total;
                            cur = q;
                        }
                    }
                }
            }
        }
        total
    }

    /// `GetBounds`: tight fill bounds using true cubic extrema (not the control hull). An empty
    /// fill region (no filled figure, or an empty X/Y range) yields Direct2D.s fill-bounds
    /// sentinel `{+∞, +∞, FLT_MAX, FLT_MAX}`, not an all-`-∞` inverted rect.
    /// The shape-generator clip test: any corner of the shape's own tight bounds,
    /// transformed, past |524287| sends it through the safe-box clip. A curve whose control
    /// polygon alone pokes out is not clipped.
    pub(crate) fn clips_huge(&self, m: &Matrix) -> bool {
        let [l, t, r, b] = self.get_bounds(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        [[l, t], [r, t], [l, b], [r, b]].iter().flat_map(|&p| transform_point(m, p)).any(|v| v.is_nan() || v.abs() > 524287.0)
    }

    pub fn get_bounds(&self, transform: &Matrix) -> [f32; 4] {
        self.bounds_of(transform, false)
    }

    /// [`Self::get_bounds`], with the hollow figures when `hollow` (the bounds a stroke query falls
    /// back to when its pen is too small to widen).
    pub(crate) fn bounds_of(&self, transform: &Matrix, hollow: bool) -> [f32; 4] {
        self.fill_bounds((*transform != IDENTITY).then_some(transform), hollow)
    }

    /// Native's per-shape fill bounds; `m` is `None` when no matrix is given (an identity one handed
    /// down by a transformed shape still maps).
    fn fill_bounds(&self, m: Option<&Matrix>, hollow: bool) -> [f32; 4] {
        match (self, m) {
            // A rectangle holds its ordered rect, mapped corner by corner under a matrix.
            (Geometry::Rect(rect), Some(m)) => map_bounds(ordered_rect(*rect), m),
            (Geometry::Rect(rect), None) => ordered_rect(*rect),
            // A transformed shape hands its own matrix, composed with the caller's, to its child.
            (Geometry::Transformed { matrix, child }, _) => child.fill_bounds(Some(&m.map_or(*matrix, |a| mat_mul(matrix, a))), hollow),
            // A rounded rectangle or ellipse under no matrix, or one keeping the axes aligned, bounds
            // its 16 figure points (all NaN when any coordinate is NaN) and maps that box through the
            // matrix.
            (Geometry::Rounded { .. } | Geometry::Ellipse { .. }, _) if m.is_none_or(|m| (m[1] == 0.0 && m[2] == 0.0) || (m[0] == 0.0 && m[3] == 0.0)) => {
                let (rect, radius) = match self {
                    Geometry::Rounded { rect, radius } => (*rect, *radius),
                    Geometry::Ellipse { center, radius } => (ellipse_rect(*center, *radius), *radius),
                    _ => unreachable!(),
                };
                let (p, _) = rounded_rect_points(rect, radius);
                if p.iter().flatten().any(|v| v.is_nan()) {
                    return [f32::NAN; 4];
                }
                let (mut lo, mut hi) = (p[0], p[0]);
                for q in &p[1..] {
                    lo = [lo[0].min(q[0]), lo[1].min(q[1])];
                    hi = [hi[0].max(q[0]), hi[1].max(q[1])];
                }
                let b = [lo[0], lo[1], hi[0], hi[1]];
                match m {
                    Some(m) if !(b[0] > b[2] || b[1] > b[3]) => map_bounds(b, m),
                    _ => b,
                }
            }
            // A path under a matrix without rotation or shear maps its local bounds (unless hollow
            // figures it must skip force the stream); a rotated or sheared one streams its points
            // through it.
            (Geometry::Path { figures, .. }, Some(m)) if m[1] == 0.0 && m[2] == 0.0 && (hollow || figures.iter().all(|f| f.filled)) => {
                let b = self.stream_bounds(&IDENTITY, hollow);
                if b[0] > b[2] || b[1] > b[3] { b } else { map_bounds(b, m) }
            }
            _ => self.stream_bounds(m.unwrap_or(&IDENTITY), hollow),
        }
    }

    /// The bounds calculator over this shape's stream under `transform` (see [`Self::get_bounds`]).
    fn stream_bounds(&self, transform: &Matrix, hollow: bool) -> [f32; 4] {
        // A path streams its fill figures into native's bounds sink, which starts at ±FLT_MAX, keeps
        // a figure's infinite points and turns the whole bounds NaN at any NaN coordinate (r23 28114,
        // 28277, 28495).
        // A rounded rectangle or ellipse streamed (under a rotation or shear) feeds the same sink.
        let path = matches!(self, Geometry::Path { .. } | Geometry::Rounded { .. } | Geometry::Ellipse { .. });
        let mut lo = [if path { f32::MAX } else { f32::INFINITY }; 2];
        let mut hi = [if path { -f32::MAX } else { f32::NEG_INFINITY }; 2];
        if self.accumulate_bounds(transform, hollow, &mut lo, &mut hi) {
            return [f32::NAN; 4];
        }
        if lo[0] > hi[0] || lo[1] > hi[1] {
            return [f32::INFINITY, f32::INFINITY, f32::MAX, f32::MAX];
        }
        [lo[0], lo[1], hi[0], hi[1]]
    }

    /// Widen `lo`/`hi` by this shape's bounds; true when a path point is NaN (see [`Self::get_bounds`]).
    fn accumulate_bounds(&self, m: &Matrix, hollow: bool, lo: &mut [f32; 2], hi: &mut [f32; 2]) -> bool {
        let prims: Vec<Prim> = match self {
            Geometry::Group { children, .. } => {
                return children.iter().fold(false, |nan, c| c.accumulate_bounds(m, hollow, lo, hi) | nan);
            }
            Geometry::Transformed { matrix, child } => {
                return child.accumulate_bounds(&mat_mul(matrix, m), hollow, lo, hi);
            }
            Geometry::Rect([l, t, r, b]) => {
                for p in [[*l, *t], [*r, *t], [*r, *b], [*l, *b]] {
                    include(lo, hi, transform_point(m, p));
                }
                return false;
            }
            Geometry::Rounded { .. } | Geometry::Ellipse { .. } => {
                let (rect, radius) = match self {
                    Geometry::Rounded { rect, radius } => (*rect, *radius),
                    Geometry::Ellipse { center, radius } => (ellipse_rect(*center, *radius), *radius),
                    _ => unreachable!(),
                };
                let prims = transform_prims(&rounded_rect_figure(rect, radius).1, m);
                // Streamed into the bounds sink like a path figure: a NaN coordinate turns it NaN.
                let nan = |p: &Point| p[0].is_nan() || p[1].is_nan();
                if prims.iter().any(|pr| match pr {
                    Prim::Line(a, b) => nan(a) || nan(b),
                    Prim::Cubic(p0, p1, p2, p3) => [p0, p1, p2, p3].into_iter().any(nan),
                }) {
                    return true;
                }
                prims
            }
            Geometry::Path { figures, .. } => {
                // Fill bounds: a hollow figure is not streamed to the fill sink, so it contributes
                // nothing (native `GetFillBounds`) unless asked for; a path of only hollow figures is
                // then empty.
                let mut all = Vec::new();
                let nan = |p: &Point| p[0].is_nan() || p[1].is_nan();
                for fig in figures.iter().filter(|f| f.filled || hollow) {
                    let (start, prims) = figure_prims(fig, m);
                    // `BeginFigure` counts the start point: a figure that is a lone point has bounds.
                    if nan(&start) {
                        return true;
                    }
                    include(lo, hi, start);
                    all.extend(prims);
                }
                if all.iter().any(|pr| match pr {
                    Prim::Line(a, b) => nan(a) || nan(b),
                    Prim::Cubic(p0, p1, p2, p3) => [p0, p1, p2, p3].into_iter().any(nan),
                }) {
                    return true;
                }
                all
            }
        };
        let (mut nan, path) = (false, matches!(self, Geometry::Path { .. } | Geometry::Rounded { .. } | Geometry::Ellipse { .. }));
        for pr in &prims {
            match *pr {
                Prim::Line(a, b) => {
                    include(lo, hi, a);
                    include(lo, hi, b);
                }
                Prim::Cubic(p0, p1, p2, p3) => {
                    include(lo, hi, p0);
                    include(lo, hi, p3);
                    nan |= cubic_axis_extrema(p0[0], p1[0], p2[0], p3[0], path, &mut lo[0], &mut hi[0]);
                    nan |= cubic_axis_extrema(p0[1], p1[1], p2[1], p3[1], path, &mut lo[1], &mut hi[1]);
                }
            }
        }
        nan
    }

    /// `FillContainsPoint`: analytic for rect/ellipse/rounded under an axis-aligned transform;
    /// otherwise a winding/parity ray cast over the flattened polylines under the fill rule.
    pub fn fill_contains(&self, pt: Point, transform: &Matrix, tolerance: f32) -> bool {
        // Native's hit tester over the shape's own stream (hollow figures ignored), the point taken
        // relative to the query in f64: a point within the normalised tolerance of the outline is
        // inside; otherwise the winding count decides by the fill mode.
        let mut figs = Vec::new();
        self.fill_hit_figures(&mut figs);
        hit_test(&figs, pt, transform, normalize_tolerance(tolerance) as f64, self.fill_mode() == FillMode::Winding)
    }

    /// The figures FillContainsPoint's hit tester takes: the shape's stream in its own space (a
    /// transformed geometry through its matrix), without the hollow figures the tester ignores.
    fn fill_hit_figures(&self, out: &mut Vec<(Point, Vec<Prim>, bool)>) {
        match self {
            Geometry::Path { figures, .. } => {
                for fig in figures.iter().filter(|f| f.filled) {
                    let (s, pr) = figure_prims(fig, &IDENTITY);
                    out.push((s, pr, fig.closed));
                }
            }
            Geometry::Group { children, .. } => children.iter().for_each(|c| c.fill_hit_figures(out)),
            Geometry::Transformed { matrix, child } => {
                let mut inner = Vec::new();
                child.fill_hit_figures(&mut inner);
                out.extend(inner.into_iter().map(|(s, pr, c)| (transform_point(matrix, s), pre_prims(&[*matrix], &pr), c)));
            }
            _ => self.sink_figures(false, out),
        }
    }


    /// True if ANY part of the geometry has a non-finite coordinate. A geometry REALIZATION
    /// (`CreateFilled`/`CreateStrokedGeometryRealization`) of such a geometry is rejected as a whole by
    /// native — the realization is empty and `DrawGeometryRealization` draws nothing — unlike a direct
    /// `FillGeometry`/`DrawGeometry`, which drops only the offending figures and keeps the finite ones.
    /// Fuzz r11 16021: g1's figure 1 has a NaN cubic control point, so native voids the whole stroked
    /// realization even though figure 2 is finite; per-figure filtering would stroke figure 2 (a 1e9
    /// width covering the target). Matches the fill guards' position-only rule (a non-finite rect/ellipse
    /// RADIUS is clamped+drawn, not rejected — see `prims_per_figure`).
    pub(crate) fn any_nonfinite(&self) -> bool {
        match self {
            Geometry::Rect(r) => !r.iter().all(|v| v.is_finite()),
            Geometry::Rounded { rect, .. } => !rect.iter().all(|v| v.is_finite()),
            Geometry::Ellipse { center, .. } => !center.iter().all(|v| v.is_finite()),
            Geometry::Path { figures, .. } => figures.iter().any(|f| f.has_nonfinite_pos()),
            Geometry::Group { children, .. } => children.iter().any(|c| c.any_nonfinite()),
            Geometry::Transformed { child, .. } => child.any_nonfinite(),
        }
    }

    /// [`Self::any_nonfinite_fill_void`] as a FILL applies it: a rectangle voids up front only on a NaN.
    /// It streams its corners into the safe-box strip clip, where an infinite corner is just outside: a
    /// rect wholly past one side drops out, and one crossing the box voids on the NaN crossing it makes.
    /// ponytail: a path figure with an infinite point still voids up front; route it through the strip
    /// clip too if a path that stays wholly past one side shows up drawing natively.
    pub(crate) fn fill_nonfinite_void(&self) -> bool {
        match self {
            Geometry::Rect(r) => r.iter().any(|v| v.is_nan()),
            Geometry::Group { children, .. } => children.iter().any(|c| c.fill_nonfinite_void()),
            Geometry::Transformed { child, .. } => child.fill_nonfinite_void(),
            _ => self.any_nonfinite_fill_void(),
        }
    }

    /// The whole-geometry non-finite void for `fill_geometry`/`fill_geometry_brush`. A FILLED figure
    /// voids on ANY non-finite coordinate ([`Figure::has_nonfinite`], control included: fuzz r17 22268's
    /// filled figure with a non-finite control, 16347's filled `to`). A HOLLOW (`filled:false`) figure
    /// never voids: the fill sink drops its points and segments at its hollow start, whatever they hold.
    /// `clip_figure` is kept, so the surviving figures' huge finite coordinates clip/flatten exactly as
    /// before.
    pub(crate) fn any_nonfinite_fill_void(&self) -> bool {
        match self {
            Geometry::Rect(r) => !r.iter().all(|v| v.is_finite()),
            Geometry::Rounded { rect, .. } => !rect.iter().all(|v| v.is_finite()),
            Geometry::Ellipse { center, .. } => !center.iter().all(|v| v.is_finite()),
            Geometry::Path { figures, .. } => figures.iter().any(|f| f.filled && f.has_nonfinite()),
            Geometry::Group { children, .. } => children.iter().any(|c| c.any_nonfinite_fill_void()),
            Geometry::Transformed { child, .. } => child.any_nonfinite_fill_void(),
        }
    }

    /// Whether the shape's own SIGNED loose bounds are inverted (min > max on an axis), which makes the
    /// realization's `Initialize` return an empty S_OK rather than reach the widen. An Ellipse with a
    /// NEGATIVE radius gives `center ± r` with min > max (fuzz 18671: radius `[-3.49, …]` + a huge width
    /// still strokes S_OK, unlike a plain rect). Rects/rounded-rects are normalized before the widen, so
    /// an inverted rect is NOT inverted here; Paths carry no signed radius.
    fn signed_bounds_inverted(&self) -> bool {
        match self {
            Geometry::Ellipse { radius, .. } => radius[0] < 0.0 || radius[1] < 0.0,
            Geometry::Transformed { child, .. } => child.signed_bounds_inverted(),
            Geometry::Group { children, .. } => children.iter().any(|c| c.signed_bounds_inverted()),
            _ => false,
        }
    }

    /// Whether STROKING this geometry overflows native's 28.4 fixed-point edge list at a figure START,
    /// voiding the WHOLE `DrawGeometry` (not just that figure). `limit` is the [`Geometry::widen_inner`]
    /// cutoff, already net of the device half-width (so a huge WIDTH drives it to 0 → any start
    /// overflows). A finite `±1e9`/`±3.4e38` start opens the widened contour out of i32 range, so native
    /// bails the whole call (fuzz 17707: a 1e9 start voids all three figures; 3643/17200/17436/17927).
    /// The caller gates this on [`Self::any_nonfinite`] so an ordinary finite huge-coordinate stroke is
    /// left to cover (fuzz 15265); a non-overflow non-finite figure is widened per-figure (7408/17307).
    /// A figure every segment of which stays at its start widens no segment: the widener ends it as a
    /// dot, which reaches the edge list only if it has area (see [`stroke::collapsed_dot`]), so its start
    /// alone does not void (r26 31490: an empty figure at x = 1e9 beside drawn ones).
    fn stroke_overflows(&self, m: &Matrix, limit: f32) -> bool {
        self.stroke_overflows_by(m, limit, true, true)
    }

    /// [`Self::stroke_overflows`] counting only the path figure starts (`paths`) and/or the non-finite
    /// rect, rounded rect and ellipse shapes (`shapes`).
    fn stroke_overflows_by(&self, m: &Matrix, limit: f32, paths: bool, shapes: bool) -> bool {
        match self {
            Geometry::Path { figures, .. } => paths && figures.iter().any(|f| !f.stays_at_start() && f.start_device_maxabs(m) >= limit),
            Geometry::Group { children, .. } => children.iter().any(|c| c.stroke_overflows_by(m, limit, paths, shapes)),
            Geometry::Transformed { matrix, child } => child.stroke_overflows_by(&mat_mul(matrix, m), limit, paths, shapes),
            // A non-finite rect/rounded/ellipse widens to a NaN outline that native's render drops (its
            // Widen returns S_OK with a NaN figure, then the edge build voids) — fuzz 17436's nan/-inf
            // rect stroke draws nothing. The figure wideners never reach a per-segment pen error for
            // these fast-path shapes, so void them here.
            Geometry::Rect(r) | Geometry::Rounded { rect: r, .. } => shapes && !r.iter().all(|v| v.is_finite()),
            Geometry::Ellipse { center, radius } => shapes && (!center.iter().all(|v| v.is_finite()) || !radius.iter().all(|v| v.is_finite())),
        }
    }

    /// Whether STROKING voids the whole `DrawGeometry` because a Path figure has a non-finite SEGMENT
    /// point (endpoint/control/arc) — native's `Widen` BAD_NUMBER, decided on the raw figure before our
    /// widener clips it (see [`Figure::has_nonfinite_nonstart`]). A non-finite START is NOT included, so
    /// it still strokes. Rect/rounded/ellipse are covered by [`Self::stroke_overflows`].
    fn stroke_nonfinite_voids(&self) -> bool {
        match self {
            Geometry::Path { figures, .. } => figures.iter().any(|f| f.has_nonfinite_nonstart()),
            Geometry::Group { children, .. } => children.iter().any(|c| c.stroke_nonfinite_voids()),
            Geometry::Transformed { child, .. } => child.stroke_nonfinite_voids(),
            _ => false,
        }
    }

    /// `Widen`/`DrawGeometry`: the stroked region as fill pieces in device space, to union under
    /// nonzero winding. For `STROKE_TRANSFORM_NORMAL` the pen scales with the transform, so the
    /// device pen width is `width · maxScaleFactor(transform)`; the path is flattened in device
    /// space at `tolerance`. Caps/joins/dashes come from `stroke`. See its module docs for what is
    /// byte-exact (axis-aligned rectangles) versus geometrically-faithful (native `CWidener`).
    /// Whether STROKING this geometry voids the whole call with `D2DERR_BAD_NUMBER` (0x88990011) — the
    /// condition native's `ID2D1Geometry::Widen` reports and that a **stroked geometry realization**
    /// (`CreateStrokedGeometryRealization`) must return instead of baking an empty mesh (fuzz 16683).
    /// True when a Path figure has a non-finite SEGMENT point (endpoint/control/arc —
    /// [`Figure::has_nonfinite_nonstart`]), or a non-finite figure START / rect / ellipse whose widened
    /// contour overflows the 28.4 edge list ([`Self::stroke_overflows`]). A non-finite START alone, or a
    /// finite-huge coordinate, still strokes. `widen_inner` returns an empty piece set in exactly these
    /// cases; callers that need the HRESULT (realizations) check this predicate.
    pub fn widen_bad_number(&self, width: f32, transform: &Matrix, dpi_scale: [f32; 2]) -> bool {
        width != 0.0 && (self.stroke_nonfinite_voids() || self.widen_overflow_void(width, false, transform, dpi_scale))
    }

    /// Whether a widening query (`Widen`, `GetWidenedBounds`, or `StrokeContainsPoint` at `hit`) under
    /// `m` fails with `D2DERR_BAD_NUMBER`.
    ///
    /// A rectangle, or a rounded rectangle / ellipse with equal stored radii, stroked solid with a
    /// normal-transform pen is widened analytically: nothing reaches the pen, so
    /// only `StrokeContainsPoint` can fail, when its hit tester meets a NaN on the widened outline
    /// ([`Self::fill_contains_bad_number`] over the outer rectangle). Any other shape streams its
    /// figures through the widener ([`Self::widen_walk`]), which fails on a direction the pen cannot
    /// normalize, and `StrokeContainsPoint`'s sink on a NaN point. The edge-list overflows of
    /// [`Self::widen_bad_number`] belong to filling the widened outline, which a query never does.
    pub fn widen_query_bad_number(&self, width: f32, style: &StrokeStyle, m: &Matrix, tolerance: f32, hit: Option<Point>) -> bool {
        if let Some(p) = hit {
            return self.stroke_hit(p, width, style, m, tolerance).is_none();
        }
        if style.dashes.is_empty() && style.transform_type == stroke::TransformType::Normal && self.analytic_widen_outer(width).is_some() {
            return false;
        }
        let tol = normalize_tolerance(tolerance) as f64;
        // A pen under the length fuzz widens nothing.
        stroke::PenShape::build(width, m, [1.0, 1.0], style.transform_type, tol).is_some_and(|pen| self.widen_walk(&pen, style, m, tol))
    }

    /// `StrokeContainsPoint`: whether `pt` is inside the stroke, or `None` when the query fails with
    /// `D2DERR_BAD_NUMBER`. The analytic widen ([`Self::widen_query_bad_number`]) streams its rings to
    /// the fill hit tester at the raw tolerance, with no pen; any other shape streams through the
    /// widener into native's hit-test sink ([`stroke::HitSink`]), where a pen under the length fuzz
    /// contains nothing.
    fn stroke_hit(&self, pt: Point, width: f32, style: &StrokeStyle, m: &Matrix, tolerance: f32) -> Option<bool> {
        if style.dashes.is_empty() && style.transform_type == stroke::TransformType::Normal {
            if let (Some(outer), Some(rings)) = (self.analytic_widen_outer(width), self.analytic_widen_rings(width, style)) {
                if Geometry::Rect(outer).fill_contains_bad_number(pt, m, tolerance) {
                    return None;
                }
                let figs: Vec<(Point, Vec<Prim>, bool)> = rings.into_iter().map(|(s, p)| (s, p, true)).collect();
                return Some(hit_test(&figs, pt, m, tolerance as f64, false));
            }
        }
        let tol = normalize_tolerance(tolerance);
        let Some(pen) = stroke::PenShape::build(width, m, [1.0, 1.0], style.transform_type, tol as f64) else { return Some(false) };
        let sink = stroke::HitSink::new(pt, tol);
        let errored = self.widen_walk(&pen.with_hit(Some(sink.clone())), style, m, tol as f64);
        let sink = sink.borrow();
        (!errored && !sink.bad).then(|| sink.contains())
    }

    /// The widener streaming this shape to `pen` under `m`, as a query runs it; whether it fails with
    /// `D2DERR_BAD_NUMBER`. A dash pattern puts the dasher between the widener and the pen, which then
    /// sees only the dashes it draws; an axis-aligned rectangle ([`Self::as_axis_aligned_rect`]: a
    /// transformed one by its bounds) takes the rectangle widener under `m`; any other shape the general
    /// walk ([`stroke::pen_walk`]).
    fn widen_walk(&self, pen: &stroke::PenShape, style: &StrokeStyle, m: &Matrix, tol: f64) -> bool {
        let xf = |m: &Matrix| (*m != IDENTITY).then_some(*m);
        let mut figs = Vec::new();
        self.sink_figures(true, &mut figs);
        if style.pen_dashes().is_some() {
            if let Some(rect) = self.as_axis_aligned_rect() {
                return dash::rect_dashed_rejects(rect, xf(m).as_ref(), pen, style, tol as f32);
            }
            let mut err = false;
            for (start, prims, closed) in figs {
                let (start, prims) = if *m == IDENTITY { (start, prims) } else { (transform_point(m, start), transform_prims(&prims, m)) };
                dash::widen_dashed_flags(start, &prims, closed, pen, style, tol as f32, &[], &mut err);
                if err {
                    return true;
                }
            }
            return false;
        }
        if let Some(rect) = self.as_axis_aligned_rect() {
            return stroke::rect_walk(rect, xf(m).as_ref(), pen, style, tol as f32);
        }
        stroke::pen_walk(&figs, xf(m).as_ref(), pen, style, tol)
    }

    /// The outer rectangle of native's analytic widen ([`Self::widen_query_bad_number`]), or `None`
    /// when the shape does not take it: the rectangle, or the rounded shape's bounds over its 16 figure
    /// points (all NaN when a coordinate is NaN), pushed out by half the width.
    fn analytic_widen_outer(&self, width: f32) -> Option<[f32; 4]> {
        let rounded = |rect: [f32; 4], radius: Point| {
            let (pts, [rx, ry]) = rounded_rect_points(rect, radius);
            if rx != ry {
                return None;
            }
            if pts.iter().flatten().any(|c| c.is_nan()) {
                return Some([f32::NAN; 4]);
            }
            let mut b = [pts[0][0], pts[0][1], pts[0][0], pts[0][1]];
            for p in &pts[1..] {
                b = [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])];
            }
            Some(b)
        };
        let [l, t, r, b] = match self {
            Geometry::Rect(r) => *r,
            Geometry::Rounded { rect, radius } => rounded(*rect, *radius)?,
            Geometry::Ellipse { center, radius } => rounded(ellipse_rect(*center, *radius), *radius)?,
            _ => return None,
        };
        let h = width.abs() * 0.5;
        Some([l - h, t - h, r + h, b + h])
    }

    /// Native's analytic widen at `width`, as `(start, prims without the closing edge)` rings, or
    /// `None` when the shape does not take it (see [`Self::widen_query_bad_number`]). A rectangle widens
    /// to its rect pushed out by half the width and, when wider and taller than the pen, pulled in by
    /// it. A rounded shape with equal stored radii `r` widens from the bounds of its 16 figure points
    /// (all NaN when a coordinate is): an outer rounded rect of radius `h + r` and, when wider and
    /// taller than the pen, an inner one of radius `r − h`, a plain rect once `r ≤ h`.
    fn analytic_widen_rings(&self, width: f32, style: &StrokeStyle) -> Option<Vec<(Point, Vec<Prim>)>> {
        let w = width.abs();
        let h = w * 0.5;
        let rect_ring = |[l, t, r, b]: [f32; 4]| ([l, t], vec![Prim::Line([l, t], [r, t]), Prim::Line([r, t], [r, b]), Prim::Line([r, b], [l, b])]);
        let rounded_ring = |rect: [f32; 4], radius: f32| {
            let (p, _) = rounded_rect_points(rect, [radius, radius]);
            (p[0], vec![
                Prim::Cubic(p[0], p[1], p[2], p[3]),
                Prim::Line(p[3], p[4]),
                Prim::Cubic(p[4], p[5], p[6], p[7]),
                Prim::Line(p[7], p[8]),
                Prim::Cubic(p[8], p[9], p[10], p[11]),
                Prim::Line(p[11], p[12]),
                Prim::Cubic(p[12], p[13], p[14], p[15]),
            ])
        };
        let (bounds, radius) = match self {
            Geometry::Rect(rect) => (ordered_rect(*rect), None),
            Geometry::Rounded { rect, radius } => (*rect, Some(*radius)),
            Geometry::Ellipse { center, radius } => (ellipse_rect(*center, *radius), Some(*radius)),
            _ => return None,
        };
        let Some(radius) = radius else {
            let [l, t, r, b] = bounds;
            // The outer ring follows the join: a round join rounds it by the half width; otherwise the
            // 90° bevel offset of the join (a bevel, a miter-or-bevel under a √2 limit, or a miter
            // whose limit falls under 2/√2) cuts its corners.
            let outer = [l - h, t - h, h + r, h + b];
            let sqrt2 = std::f32::consts::SQRT_2;
            let f = match style.join {
                stroke::Join::Round => 0.0,
                stroke::Join::Bevel => 1.0,
                stroke::Join::MiterOrBevel => if style.miter_limit <= sqrt2 { 1.0 } else { 0.0 },
                stroke::Join::Miter => (2.0 - style.miter_limit * sqrt2).max(0.0).min(1.0),
            };
            let o = (f * 0.5) * w;
            let [ol, ot, or, ob] = outer;
            let mut rings = vec![if style.join == stroke::Join::Round {
                rounded_ring(outer, h)
            } else if o > 0.0 {
                let p = |x, y| [x, y];
                let pts = [p(ol + o, ot), p(or - o, ot), p(or, ot + o), p(or, ob - o), p(or - o, ob), p(ol + o, ob), p(ol, ob - o)];
                let start = p(ol, ot + o);
                let mut prims = Vec::new();
                let mut cur = start;
                for q in pts {
                    prims.push(Prim::Line(cur, q));
                    cur = q;
                }
                (start, prims)
            } else {
                rect_ring(outer)
            }];
            if w < r - l && w < b - t {
                rings.push(rect_ring([l + h, t + h, r - h, b - h]));
            }
            return Some(rings);
        };
        let (pts, [rx, ry]) = rounded_rect_points(bounds, radius);
        if rx != ry {
            return None;
        }
        let [l, t, r, b] = if pts.iter().flatten().any(|c| c.is_nan()) {
            [f32::NAN; 4]
        } else {
            pts[1..].iter().fold([pts[0][0], pts[0][1], pts[0][0], pts[0][1]], |b, p| [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])])
        };
        let mut rings = vec![rounded_ring([l - h, t - h, h + r, h + b], h + rx)];
        if r - l > w && b - t > w {
            let inner = [l + h, t + h, r - h, b - h];
            rings.push(if rx <= h { rect_ring(inner) } else { rounded_ring(inner, rx - h) });
        }
        Some(rings)
    }

    /// The figures the shape streams to a sink, untransformed and with nothing non-finite dropped:
    /// `(start, prims with the closing edge, closed)`. A rectangle is its ordered rect's four edges, a
    /// rounded shape its native 16-point figure (a NaN radius kept), a path its figures; `stroke` splits
    /// each figure at its `FORCE_UNSTROKED` segments, which the widener skips as gaps.
    /// The segment flags (D2D1_PATH_SEGMENT bits) each primitive of each figure [`Self::sink_figures`]
    /// streams (unsplit) carries: a path segment's own, for every cubic of an arc and for the closing
    /// edge the last one's; a rounded rectangle or ellipse forces round joins on all.
    pub(crate) fn figures_prim_flags(&self, out: &mut Vec<Vec<u32>>) {
        match self {
            Geometry::Path { figures, .. } => {
                for fig in figures {
                    let mut flags = Vec::new();
                    let mut cur = fig.start;
                    for (i, seg) in fig.segments.iter().enumerate() {
                        let f = fig.seg_flags.get(i).copied().unwrap_or(0) & 3;
                        let n = match *seg {
                            Segment::Arc { to, radius, rotation, clockwise, large } => arc_to_cubics(cur, to, radius, rotation, clockwise, large).len(),
                            _ => 1,
                        };
                        flags.extend(std::iter::repeat_n(f, n));
                        cur = segment_end(seg);
                    }
                    if fig.closed && cur != fig.start {
                        flags.push(flags.last().copied().unwrap_or(0));
                    }
                    out.push(flags);
                }
            }
            Geometry::Group { children, .. } => children.iter().for_each(|c| c.figures_prim_flags(out)),
            Geometry::Transformed { child, .. } => child.figures_prim_flags(out),
            Geometry::Rounded { .. } | Geometry::Ellipse { .. } => out.push(vec![2; 8]),
            Geometry::Rect(_) => out.push(vec![0; 4]),
        }
    }

    /// Whether each figure [`Self::sink_figures`] streams (unsplit) is filled, in the same order.
    pub(crate) fn figures_filled(&self, out: &mut Vec<bool>) {
        match self {
            Geometry::Path { figures, .. } => out.extend(figures.iter().map(|f| f.filled)),
            Geometry::Group { children, .. } => children.iter().for_each(|c| c.figures_filled(out)),
            Geometry::Transformed { child, .. } => child.figures_filled(out),
            _ => out.push(true),
        }
    }

    /// Per sink figure (in [`Self::sink_figures`] order), whether its last primitive is the closing
    /// edge the walk appends rather than one of the shape's own segments: always for the built-in
    /// shapes; for a closed path figure only when its last segment does not already end at its start.
    pub(crate) fn figures_close_appended(&self, out: &mut Vec<bool>) {
        match self {
            Geometry::Path { figures, .. } => out.extend(figures.iter().map(|f| {
                let end = match f.segments.last() {
                    None => f.start,
                    Some(Segment::Line(q) | Segment::Quadratic(_, q) | Segment::Cubic(_, _, q) | Segment::Arc { to: q, .. }) => *q,
                };
                f.closed && end != f.start
            })),
            Geometry::Group { children, .. } => children.iter().for_each(|c| c.figures_close_appended(out)),
            Geometry::Transformed { child, .. } => child.figures_close_appended(out),
            _ => out.push(true),
        }
    }

    pub(crate) fn sink_figures(&self, stroke: bool, out: &mut Vec<(Point, Vec<Prim>, bool)>) {
        let rounded = |rect: [f32; 4], radius: Point| {
            let (p, _) = rounded_rect_points(rect, radius);
            let prims = vec![
                Prim::Cubic(p[0], p[1], p[2], p[3]),
                Prim::Line(p[3], p[4]),
                Prim::Cubic(p[4], p[5], p[6], p[7]),
                Prim::Line(p[7], p[8]),
                Prim::Cubic(p[8], p[9], p[10], p[11]),
                Prim::Line(p[11], p[12]),
                Prim::Cubic(p[12], p[13], p[14], p[15]),
                Prim::Line(p[15], p[0]),
            ];
            (p[0], prims, true)
        };
        match self {
            Geometry::Rect(rect) => {
                let [l, t, r, b] = ordered_rect(*rect);
                let (a, bb, c, d) = ([l, t], [r, t], [r, b], [l, b]);
                out.push((a, vec![Prim::Line(a, bb), Prim::Line(bb, c), Prim::Line(c, d), Prim::Line(d, a)], true));
            }
            Geometry::Rounded { rect, radius } => out.push(rounded(*rect, *radius)),
            Geometry::Ellipse { center, radius } => out.push(rounded(ellipse_rect(*center, *radius), *radius)),
            Geometry::Path { figures, .. } => {
                for fig in figures {
                    if stroke && fig.seg_flags.iter().any(|&f| f & 1 != 0) {
                        for (sub, _) in split_unstroked(fig) {
                            let (s, pr) = figure_prims(&sub, &IDENTITY);
                            out.push((s, pr, false));
                        }
                    } else {
                        let (s, pr) = figure_prims(fig, &IDENTITY);
                        out.push((s, pr, fig.closed));
                    }
                }
            }
            Geometry::Group { children, .. } => children.iter().for_each(|c| c.sink_figures(stroke, out)),
            // A transformed shape streams its child through its own transforming sink: the full product
            // for a figure's start, the matrix's own form for its segments.
            Geometry::Transformed { matrix, child } => {
                let mut inner = Vec::new();
                child.sink_figures(stroke, &mut inner);
                out.extend(inner.into_iter().map(|(s, pr, c)| (transform_point(matrix, s), pre_prims(&[*matrix], &pr), c)));
            }
        }
    }

    /// [`Self::widen_bad_number`] without its non-finite segment point void: the edge-list overflows.
    /// A `hairline` pen is one device pixel whatever the width (the pen setup never reads it), so
    /// its width cannot overflow (r23 28792: a 3.4e38-wide hairline draws).
    fn widen_overflow_void(&self, width: f32, hairline: bool, transform: &Matrix, dpi_scale: [f32; 2]) -> bool {
        if width == 0.0 {
            return false;
        }
        let device_limit = (134_217_728.0 - self.widen_half_device(width, hairline, transform, dpi_scale)).max(0.0);
        self.widen_overflow_rejects(width, hairline) || (self.any_nonfinite() && self.stroke_overflows(transform, device_limit))
    }

    /// The device half-width the edge-list cutoff is net of: a hairline's half pixel, else the half width
    /// under the largest scale.
    fn widen_half_device(&self, width: f32, hairline: bool, transform: &Matrix, dpi_scale: [f32; 2]) -> f32 {
        let scale = stroke::max_singular_m(transform).max(dpi_scale[0] as f64).max(dpi_scale[1] as f64).max(1.0) as f32;
        if hairline { 0.5 } else { width.abs() / 2.0 * scale }
    }

    /// [`Self::widen_overflow_void`] without the figure-start overflow: a pen half-width whose 28.4
    /// conversion overflows.
    fn widen_overflow_rejects(&self, width: f32, hairline: bool) -> bool {
        if width == 0.0 {
            return false;
        }
        let width = if hairline { 1.0 } else { width };
        // A FINITE pen half-width so large that its 28.4 conversion (`·16`) overflows f32 to ±inf widens
        // to a non-finite edge list → BAD_NUMBER with no non-finite coordinate (fuzz 18171: a finite rect
        // stroked at width 3.4e38 → half·16 = inf). Which primitives this rejects is shape-dependent, and
        // keyed on the LOCAL half-width, not the transform-scaled device one — native widens then clips to
        // the safe device box, so a normal width under a huge transform (or a 1e9 width) strokes
        // (17025/17100); only a width whose own conversion overflows f32 voids. Three guards:
        //  • `width.is_finite()` — a ±inf/NaN width is handled by native and still strokes S_OK (fuzz
        //    20616/18005: a `-inf` stroke_width), so it must NOT void here.
        //  • `!any_nonfinite()` leaves a non-finite GEOMETRY to the figure-start overflow.
        //  • `!signed_bounds_inverted()` — an ellipse with a negative radius has inverted signed loose
        //    bounds, so the realization is empty S_OK even at a huge width (fuzz 18671).
        // The straight-edged RECTANGLE primitive overflows at EITHER sign (fuzz 18171 at +3.4e38, 26642 at
        // -3.4e38), and so does any outline that needs a join or a curve: a path or group with a polyline,
        // a closed figure, an arc or a Bézier fails at either sign. A lone open straight segment (two caps,
        // no join) and the analytic ellipse / rounded-rect primitives never overflow (all probed at
        // ±3.4e38: S_OK; fuzz 20640's -3.4e38 ellipse).
        let overflow = !(width / 2.0 * 16.0).is_finite() && !self.any_nonfinite() && !self.signed_bounds_inverted();
        width.is_finite() && overflow && self.widen_overflow_fails()
    }

    /// Whether widening this shape at a pen half-width whose fixed-point conversion overflows fails (see
    /// [`Self::widen_bad_number`]): every outline with a join or a curve does, at either sign of the
    /// width; a lone open straight segment and the analytic ellipse / rounded rect do not.
    fn widen_overflow_fails(&self) -> bool {
        match self {
            Geometry::Rect(_) => true,
            Geometry::Ellipse { .. } | Geometry::Rounded { .. } => false,
            Geometry::Path { figures, .. } => {
                figures.iter().any(|f| !f.segments.is_empty() && !(!f.closed && matches!(f.segments[..], [Segment::Line(_)])))
            }
            Geometry::Group { children, .. } => children.iter().any(Self::widen_overflow_fails),
            Geometry::Transformed { child, .. } => child.widen_overflow_fails(),
        }
    }

    /// Whether a stroked geometry realization's create fails with `D2DERR_BAD_NUMBER` (fuzz 16683).
    /// Direct2D.s geometry realization first takes the shape's loose bounds with
    /// no transform and returns S_OK with an EMPTY realization when they compare inverted — which a NaN
    /// anywhere in the raw shape makes them; only then does it clip to the safe device box
    /// (when a corner passes |524287|) and widen, where a pen
    /// BAD_NUMBER fails the create. So a NaN shape (an explicit NaN, or an arc with a non-finite end or
    /// parameter, whose conversion is NaN: fuzz 16021, 16313, 11119) is empty, while ±inf / huge points
    /// alone reach the widen ([`Self::widen_bad_number`], in DIP space). The widen first resets its pen at
    /// the realization's `tolerance`; a pen that resets empty ([`stroked_realization_voids`]) widens
    /// nothing, so nothing fails (r26 31098: a 3.4e38-wide stroke at tolerance inf is S_OK). Otherwise
    /// it widens the shape as the safe-box clip left it ([`realization::stroke_safe_clipped`]: a figure
    /// that starts past the box gains a run along it, and so a join), and the scan of that outline fails
    /// on a crossing the clip took past the f32 range ([`Self::realize_bad_number`], r26 31389).
    pub fn stroked_realize_bad_number(&self, width: f32, style: &StrokeStyle, tolerance: f32) -> bool {
        // An empty realization never widens; `realization::realization_clips_empty` decides that first.
        if stroked_realization_voids(self, width, style, tolerance) {
            return false;
        }
        // A curve with a NaN control tests flat in the widener and widens as a line to its end, so its end meets the
        // same checks a line's does (VM p753: a hollow cubic with a NaN control fails at a -3.4e38 end, passes at -100).
        if self.nan_controls_as_lines().realize_scan_bad_number(tolerance) {
            return true;
        }
        // A shape inside the box comes out of the clip as it went in, its forced round joins kept.
        // ponytail: a shape with a non-finite point keeps the unclipped widen, as the realization's own
        // widen does (r11 16683: native's clip carries the -inf on into a failing widen).
        let clipped = if self.clips_huge(&IDENTITY) && !self.any_nonfinite() { realization::stroke_safe_clipped(self, width, style) } else { None };
        clipped.as_ref().unwrap_or(self).widen_bad_number(width, &IDENTITY, [1.0, 1.0])
    }

    /// A NaN in the shape d2d1 builds from this geometry with no transform applied (see
    /// [`Self::stroked_realize_bad_number`]).
    pub(crate) fn raw_shape_has_nan(&self) -> bool {
        let nan = |p: &Point| p[0].is_nan() || p[1].is_nan();
        match self {
            Geometry::Rect(r) => r.iter().any(|v| v.is_nan()),
            Geometry::Rounded { rect, radius } => rect.iter().chain(radius.iter()).any(|v| v.is_nan()),
            Geometry::Ellipse { center, radius } => nan(center) || nan(radius),
            Geometry::Path { figures, .. } => figures.iter().any(Figure::raw_has_nan),
            Geometry::Group { children, .. } => children.iter().any(|c| c.raw_shape_has_nan()),
            Geometry::Transformed { matrix, child } => matrix.iter().any(|v| v.is_nan()) || child.raw_shape_has_nan(),
        }
    }

    pub fn widen(&self, width: f32, style: &StrokeStyle, transform: &Matrix, dpi_scale: [f32; 2], tolerance: f32) -> Vec<Vec<Point>> {
        self.widen_scaled(width, style, transform, dpi_scale, tolerance, 16.0, None, None)
    }

    /// As [`widen`] but at an explicit fill grid (`fill_scale`): a stroke widens to a fill, so an
    /// ALIASED stroke's pieces go through the same WARP 26.6 tessellator gate as an aliased fill —
    /// every piece vertex snapped round-to-nearest to 1/64 ([`snap_grid`]). 16 keeps the 28.4 path.
    /// `target` is the pixel rect a COPY stroke's edge list is clipped to (see [`flatten_fill_prims_scale`]);
    /// `viewable` the pixel size of the target a draw strokes into, whose dashes start only near it.
    #[allow(clippy::too_many_arguments)]
    pub fn widen_scaled(&self, width: f32, style: &StrokeStyle, transform: &Matrix, dpi_scale: [f32; 2], tolerance: f32, fill_scale: f32, target: Option<[i32; 4]>, viewable: Option<[usize; 2]>) -> Vec<Vec<Point>> {
        let mut pieces = match self.rounded_stroke_rings(width, style, transform, target) {
            Some(rings) => rings,
            None => self.widen_inner(width, style, transform, dpi_scale, tolerance, None, target, viewable, None),
        };
        if fill_scale > 16.0 {
            for piece in &mut pieces {
                for v in piece.iter_mut() {
                    *v = snap_grid(*v, fill_scale);
                }
            }
        }
        pieces
    }

    /// A rounded rect or ellipse stroke the target draws through the shape's own widen
    /// ([`Self::special_case_widen`]): the device rings filled as any path, and so voided by a non-finite
    /// point. The stroke fills nonzero, so the inner ring's pieces run reversed: nested rings of
    /// opposite turn cover the band the rings' alternate fill does.
    fn rounded_stroke_rings(&self, width: f32, style: &StrokeStyle, transform: &Matrix, target: Option<[i32; 4]>) -> Option<Vec<Vec<Point>>> {
        if !matches!(self, Geometry::Rounded { .. } | Geometry::Ellipse { .. }) {
            return None;
        }
        let Geometry::Path { figures, .. } = self.special_case_widen(width, style, transform)? else { return None };
        let mut pieces = Vec::new();
        for (i, figure) in figures.into_iter().enumerate() {
            let ring = Geometry::Path { fill: FillMode::Alternate, figures: vec![figure] };
            if ring.any_nonfinite_fill_void() {
                return Some(Vec::new());
            }
            for p in ring.flatten_fill_on(&IDENTITY, target).into_iter().filter(|p| p.filled) {
                let mut pts = p.pts;
                if i > 0 {
                    pts.reverse();
                }
                pieces.push(pts);
            }
        }
        Some(pieces)
    }

    /// A stroked geometry realization's create-time widen: DIP space (identity, 96 dpi) at the
    /// realization's flattening `tolerance`, with the rails' curves flattened on the realization sweep's
    /// grid at that tolerance rather than by the 28.4 fill cracker.
    pub(crate) fn widen_realization(&self, width: f32, style: &StrokeStyle, tolerance: f32) -> Vec<Vec<Point>> {
        // The widener runs at the normalized tolerance; its curved rails reach the realization sweep, which
        // flattens them at the raw one, no finer than 3e-12 of the outline's larger extent (see
        // `realization::scan_tolerance`).
        let [l, t, r, b] = self.get_bounds(&IDENTITY);
        let (w, h) = (r as f64 - l as f64, b as f64 - t as f64);
        let ext = if w > h { w } else { h } + width.abs() as f64;
        let floor = if ext.is_finite() { ext } else { 1.0 } * 3e-12;
        let rail = if tolerance as f64 > floor { tolerance as f64 } else { floor };
        self.widen_inner(width, style, &IDENTITY, [1.0, 1.0], normalize_tolerance(tolerance), Some(rail as f32), None, None, None)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn widen_inner(&self, width: f32, style: &StrokeStyle, transform: &Matrix, dpi_scale: [f32; 2], tolerance: f32, rail_tol: Option<f32>, target: Option<[i32; 4]>, viewable: Option<[usize; 2]>, raw: Option<stroke::RawSink>) -> Vec<Vec<Point>> {
        let tol = normalize_tolerance(tolerance);
        // Native strokes with |width|: the pen setup stores the half-width as
        // (width/2).abs() (see `PenShape::build`), so a negative width — even -1e9 — widens the
        // same as its magnitude (fuzz 15265: a huge negative width covers the whole target, not
        // nothing). Only an exact zero width draws no stroke, and not for a hairline pen, which never
        // reads the width.
        if width == 0.0 && style.transform_type != stroke::TransformType::Hairline {
            return Vec::new();
        }
        let width = width.abs();
        // Fixed-point edge-overflow cutoff for a NON-FINITE figure (see `prims_per_figure_keep`):
        // native widens a figure with `NaN`/`inf` coords and renders its finite rails, but only while
        // the widened contour's opening vertex stays inside the 28.4 edge list's i32 range. A finite
        // `±1e9` START, or a half-width that alone overflows, makes native draw nothing (fuzz 3643 /
        // 17200: `±1e9` width; 17927: a `-1e9` start). A huge coordinate LATER in the figure does not:
        // a `-1e9` arc endpoint (fuzz 7408) or `NaN`/`±1e9` segment endpoint (fuzz 11119), and a `±1e9`
        // control or arc radius, are widened per-segment and rendered byte-exact (the six kept
        // non-finite strokes). `scale` over-estimates the post-widen magnification so the cutoff is
        // conservative; the device half-width is subtracted so the pen's own reach can't overflow.
        const FIXED_PT_LIMIT: f32 = 134_217_728.0; // 2^27: the 28.4 edge list's i32 range / 16
        let scale = stroke::max_singular_m(transform).max(dpi_scale[0] as f64).max(dpi_scale[1] as f64).max(1.0) as f32;
        let hairline = style.transform_type == stroke::TransformType::Hairline;
        let device_hw = if hairline { 0.5 } else { width / 2.0 * scale };
        // `ID2D1Geometry::Widen` (`raw`) hands the pen's contours to a sink: no 28.4 edge list, so
        // none of the edge-list cutoffs below apply (r23 28808: an ellipse at x = −3.4e38 widens).
        let device_limit = if raw.is_some() { f32::INFINITY } else { (FIXED_PT_LIMIT - device_hw).max(0.0) };
        // Whole-geometry BAD_NUMBER void (non-finite segment point, or a non-finite start/rect that
        // overflows the 28.4 edge list) — draws nothing for the WHOLE DrawGeometry, not per-figure.
        // See [`Self::widen_bad_number`]; `device_limit` above is also the per-figure cutoff below.
        // A dashed stroke never hands a non-finite point to the pen: the dasher drops a NaN-length
        // segment and starts no dash at a non-finite point, so only the overflow voids apply to it
        // (recap 673: a dashed figure with infinite controls draws its finite dashes).
        let void = if raw.is_some() {
            style.pen_dashes().is_none() && self.stroke_nonfinite_voids()
        } else {
            self.widen_overflow_rejects(width, hairline)
                || (self.any_nonfinite() && self.stroke_overflows_by(transform, device_limit, false, true))
                || (style.pen_dashes().is_none() && self.stroke_nonfinite_voids())
        };
        if void {
            return Vec::new();
        }
        // A non-finite vertex in the widened outline leaves its bounds non-finite, so the fill skips the
        // safe-box clip, and a figure start past the 28.4 range then overflows the edge list: the whole
        // draw voids. A non-finite figure that widens to finite vertices keeps the clip (r26 31342).
        let start_overflow = raw.is_none() && self.any_nonfinite() && self.stroke_overflows_by(transform, device_limit, true, false);
        // The pen = the width/2 circle mapped by the device matrix (the pen setup);
        // uniform transforms collapse it to a circle, non-uniform ones give a direction-varying
        // ellipse. `transform_type` selects what the width is measured in (world / DPI / pixels).
        let pen = match stroke::PenShape::build(width, transform, dpi_scale, style.transform_type, tol as f64) {
            Some(p) => p.with_rail_tol(rail_tol).with_target(target).with_viewable(viewable, width, style, transform).with_raw(raw.clone()),
            None => return Vec::new(),
        };
        let dashed = style.pen_dashes().is_some();
        // Native routes an axis-aligned rectangle through its own rect widener: a non-uniform matrix
        // uses native's fast axis-aligned-rectangle frame (local-space frame), a uniform one the pen
        // (the rect widener); both round differently from the general widener's corner
        // differences. `width` is the local pen width the frame is built from.
        if !dashed {
            if let Geometry::Rect(rect) = self {
                return stroke::widen_rect(ordered_rect(*rect), width, transform, &pen, style, tol);
            }
            // A transformed shape that is an axis-aligned rectangle (see [`Self::as_axis_aligned_rect`])
            // takes the rectangle widener on its bounds under `transform` alone.
            if let Some(rect) = self.as_axis_aligned_rect() {
                return stroke::widen_rect_pen(rect, transform, &pen, style, tol);
            }
        }
        // Native dashes a rect through its own rect widener, which walks the shape's ORDERED rect
        // (l,t)→(r,t)→(r,b)→(l,b) along the widener's frame. The general dasher on the literal corners
        // of an INVERTED rect (left>right or top>bottom) starts the perimeter at the wrong corner and
        // shifts the dash phase — rectdash_inv diverged maxd 239 vs native, and ordering t/b made our
        // render byte-exact to native's inverted capture (op7 / r11 16594). A NaN side never swaps, so
        // a NaN rect keeps its finite edge's dashes.
        if dashed {
            if let Some(rect) = self.as_axis_aligned_rect() {
                return dash::rect_dashed(rect, Some(transform), &pen, style, tol, &mut false);
            }
        }
        // A non-uniform transform gives the matrix (elliptical) pen: native widens the device-space
        // figure with it, so the rails follow device orientation and the curves flatten in device space.
        let mut pieces = Vec::new();
        let mut err = false;
        // A figure split at unstroked segments: its first run's direction (the closing corner turns
        // onto it) and its deferred start-cap figure, which `Widen` streams after the figure's last run.
        let (mut first_dir, mut pending_cap) = (None, None);
        for ((start, prims, closed, round_pts), lines) in self.widen_figures_f64(transform, device_limit) {
            // FORCE_ROUND_LINE_JOIN vertices (device space), as f64 for the wideners, at the positions the
            // f64 line ends give them.
            let force_pts: Vec<[f64; 2]> = match &lines {
                Some(e) => e.force.clone(),
                None => round_pts.iter().map(|p| [p[0] as f64, p[1] as f64]).collect(),
            };
            let run = lines.as_ref().and_then(|e| e.run).filter(|_| !dashed);
            let run_style = run.map(|r| r.style(style));
            let st = run_style.as_ref().unwrap_or(style);
            let mut end = stroke::RunEnd {
                inline_start: run.is_some_and(|r| !r.deferred),
                close: run.filter(|r| r.closes).and_then(|r| first_dir.map(|d| (d, r.close_round))),
                first_dir: None,
                ..Default::default()
            };
            let raw_len = raw.as_ref().map_or(0, |r| r.borrow().len());
            if dashed {
                // The dasher: split the figure into dashes, each widened as its own open
                // sub-stroke with the dash cap. See `dash`; forced-round vertices keep their round joins.
                pieces.extend(dash::widen_dashed(start, &prims, lines.as_ref(), closed, &pen, style, &force_pts, tol as f64 * 6.0));
            } else if prims.iter().any(|p| matches!(p, Prim::Cubic(..))) {
                // Native widens the curve skeleton directly (double-precision flattening + offset along
                // the analytic tangent), not a pre-flattened chord polyline. Prims are already in device
                // space here, so flatten at the device tolerance (`tolerance`·6, the flattener's scale; a
                // stroked realization passes its own, r14 19242's 0.4997 → 4 chords per corner).
                pieces.extend(stroke::widen_prims_flags(start, &prims, lines.as_ref(), closed, &pen, st, &force_pts, tol as f64 * 6.0, &mut err, &mut end));
            } else {
                // All-line figure: the byte-exact polyline widener (round joins are vertex-keyed), on the
                // path's vertices mapped in f64 as native's widener maps a line end.
                let poly: Vec<[f64; 2]> = match lines {
                    Some(e) if e.lines.iter().all(Option::is_some) => std::iter::once(e.start).chain(e.lines.into_iter().flatten()).collect(),
                    _ => flatten_prims(start, &prims, tol).iter().map(|p| [p[0] as f64, p[1] as f64]).collect(),
                };
                pieces.extend(stroke::widen_flags(&poly, closed, &pen, st, &force_pts, &mut err, &mut end));
            }
            if let Some(r) = run {
                if r.first {
                    first_dir = end.first_dir;
                    // The run's own outline and then its start cap: hold the cap until the figure ends.
                    if let Some(raw) = &raw {
                        let mut raw = raw.borrow_mut();
                        if r.deferred && raw.len() >= raw_len + 2 {
                            pending_cap = raw.pop();
                        }
                    }
                }
                if r.last {
                    if let (Some(raw), Some(cap)) = (&raw, pending_cap.take()) {
                        raw.borrow_mut().push(cap);
                    }
                    first_dir = None;
                }
            }
        }
        // A non-finite/zero tangent native can't normalize (`CPen` BAD_NUMBER, 0x88990011) aborts the
        // whole DrawGeometry — draw nothing (fuzz 17866: an inf arc radius gives an inf first tangent,
        // the pen.s figure-start errors; our `radius_vec`/`accept_curve_point` set `errored`).
        if err {
            if let Some(raw) = &raw {
                raw.borrow_mut().clear();
            }
            return Vec::new();
        }
        if start_overflow && pieces.iter().flatten().flatten().any(|v| !v.is_finite()) {
            return Vec::new();
        }
        pieces
    }

    /// The shape's own widen, when it has one: a rect, or a rounded rect / ellipse with equal stored
    /// radii, stroked solid with a normal pen, widens analytically into closed rings in its own space
    /// ([`Self::analytic_widen_rings`]), mapped by `transform`, under alternate fill. `Widen` streams
    /// these rings, and a stroke draws them as the filled shape.
    fn special_case_widen(&self, width: f32, style: &StrokeStyle, transform: &Matrix) -> Option<Geometry> {
        if !style.dashes.is_empty() || style.transform_type != stroke::TransformType::Normal {
            return None;
        }
        let figures = self
            .analytic_widen_rings(width, style)?
            .into_iter()
            .map(|(start, prims)| Figure {
                start: map_point(transform, start),
                filled: true,
                closed: true,
                segments: transform_prims(&prims, transform).into_iter().map(|p| match p {
                    Prim::Line(_, e) => Segment::Line(e),
                    Prim::Cubic(_, c1, c2, e) => Segment::Cubic(c1, c2, e),
                }).collect(),
                seg_flags: Vec::new(),
            })
            .collect();
        Some(Geometry::Path { fill: FillMode::Alternate, figures })
    }

    /// `ID2D1Geometry::Widen`'s streamed form (as native hands it to the sink), built from the widened
    /// contours: an axis-aligned rectangle streams the clean outer/inner rectangles (alternate fill,
    /// closed, CW from the top-left corner); every other shape streams winding, *open* figures that
    /// trace back to the start with an explicit closing line. Byte-exact vs native for rects and for
    /// flat-cap / miter-or-bevel-join strokes (lines and curves). Two known core-widener differences
    /// remain: native keeps round joins/caps as bezier segments (this flattens to lines), and native
    /// splits a square cap into its own figure (this keeps one). Those need a richer widener contour.
    pub fn widen_figures(&self, width: f32, style: &StrokeStyle, transform: &Matrix, dpi_scale: [f32; 2], tolerance: f32) -> Geometry {
        if let Some(rings) = self.special_case_widen(width, style, transform) {
            return rings;
        }
        // Every other shape streams the widener's own contours: its rails' lines and beziers, unflattened.
        let raw: stroke::RawSink = Default::default();
        self.widen_inner(width, style, transform, dpi_scale, tolerance, None, None, None, Some(raw.clone()));
        let figures = raw.take().iter().map(stroke::RawRail::figure).collect();
        Geometry::Path { fill: FillMode::Winding, figures }
    }

    /// `StrokeContainsPoint` ([`Self::stroke_hit`]); a query that fails contains nothing.
    pub fn stroke_contains(&self, pt: Point, width: f32, style: &StrokeStyle, transform: &Matrix, tolerance: f32) -> bool {
        self.stroke_hit(pt, width, style, transform, tolerance).unwrap_or(false)
    }


    /// `GetWidenedBounds`: native runs the widener into a bounds sink, so these are the bounds of the
    /// stream `Widen` gives ([`Self::widen_figures`]: each point to f32, a NaN one making them all NaN,
    /// none at all `[inf, inf, MAX, MAX]`), or, when the pen is too small to widen, the shape's bounds
    /// with its hollow figures. The analytic widen has no pen to decline.
    pub fn widened_bounds(&self, width: f32, style: &StrokeStyle, transform: &Matrix, tolerance: f32) -> [f32; 4] {
        // A rectangle whose corners land past the ±524287 box our widener clips to: native's bounds are the stroke's own
        // box, the ordered rect grown by half the pen (VM probe-wb: a 3.4e38-wide rect, and a rounded rect with negative
        // radii, at widths 0.56, 2 and 7.5; in-range ones already match the widened outline).
        let rect = match self {
            Geometry::Rect(r) => Some(r),
            // A rounded rectangle with no positive radius is the rectangle; a rounded one keeps native's other bounds
            // (r26 31535: radius 7.55, a 3.4e38-tall rect, left edge 9.545 with no growth).
            Geometry::Rounded { rect, radius } if radius[0] <= 0.0 && radius[1] <= 0.0 => Some(rect),
            _ => None,
        };
        if let (Some(r), [a, 0.0, 0.0, d, ..]) = (rect, *transform) {
            let [l, t, rr, b] = ordered_rect(*r);
            let (p, q) = (transform_point(transform, [l, t]), transform_point(transform, [rr, b]));
            let huge = p.iter().chain(&q).any(|v| v.abs() > 524287.0);
            if huge && p.iter().chain(&q).all(|v| v.is_finite()) && style.dashes.is_empty() && style.transform_type == stroke::TransformType::Normal {
                let (hx, hy) = (width.abs() * 0.5 * a.abs(), width.abs() * 0.5 * d.abs());
                return [p[0].min(q[0]) - hx, p[1].min(q[1]) - hy, p[0].max(q[0]) + hx, p[1].max(q[1]) + hy];
            }
        }
        let tol = normalize_tolerance(tolerance) as f64;
        let analytic = style.dashes.is_empty() && style.transform_type == stroke::TransformType::Normal && self.analytic_widen_rings(width, style).is_some();
        if !analytic && stroke::PenShape::build(width, transform, [1.0, 1.0], style.transform_type, tol).is_none() {
            return self.bounds_of(transform, true);
        }
        self.widen_figures(width, style, transform, [1.0, 1.0], tolerance).get_bounds(&IDENTITY)
    }


    /// `ComputePointAtLength` (the length walk): the point and
    /// unit tangent at arc-length `len`. Returns `None` for empty geometry. A thin view of
    /// [`Geometry::point_and_segment_at_length`] starting at segment 0.
    pub fn point_at_length(&self, len: f32, transform: &Matrix, tolerance: f32) -> Option<(Point, Point)> {
        self.point_and_segment_at_length(len, 0, transform, tolerance).map(|(p, t, ..)| (p, t))
    }

    /// `ComputePointAndSegmentAtLength` (`ID2D1PathGeometry1`): the point + unit tangent at arc-length
    /// `len` measured from segment `start_segment`, plus `(end_segment, end_figure, length_to_end)`.
    /// `end_segment` is the **global** index (across figures, each figure's closing edge counted) of
    /// the segment the point lies on; `end_figure` its figure; `length_to_end` the arc length before
    /// that segment, measured from `start_segment`. When `len` runs past the end, the three are the
    /// total segment count, figure count and total length (native's end-of-geometry result). Point and
    /// tangent use the exact `point_at_length` arithmetic (so that query stays oracle-verified); the
    /// indices and `length_to_end` were measured against Direct2D.
    pub fn point_and_segment_at_length(&self, len: f32, start_segment: u32, transform: &Matrix, tolerance: f32) -> Option<(Point, Point, u32, u32, f32)> {
        // The length walk takes the raw tolerance (see [`Self::compute_length`]) and a length of at
        // least zero (`maxss(len, 0)`: a negative or NaN length is the start).
        let tol = tolerance;
        let len = if len > 0.0 { len } else { 0.0 };
        let mut figs = Vec::new();
        self.length_figures(transform, &mut figs);
        // Native's segment index counts one per streamed segment plus a closed figure's implicit close
        // (even when zero-length, so it has no prim) — `combine::segment_count` per figure. A segment
        // is one prim for a line/cubic/quadratic (an arc fans out to several prims, the known arc gap).
        let seg_counts: Vec<u32> = match self {
            Geometry::Path { figures, .. } => figures.iter().map(|f| (f.segments.len() + usize::from(f.closed)) as u32).collect(),
            _ => Vec::new(),
        };
        let mut acc = 0.0f32;
        let mut seg_base = 0u32; // global segment index at the start of the current figure
        // Native's length walk: the current point and the last direction start as NaN. A line moves the
        // direction only when its end differs from the current point (a NaN end always does); a measured
        // curve sets it to its last tangent; a curve without a first tangent is skipped but still moves
        // the current point to its end. Nothing guards a zero or NaN segment length. Running out of
        // geometry reports the current point and the normalised last direction.
        let mut lastdir: Point = [f32::NAN; 2];
        let mut cur: Point = [f32::NAN; 2];
        let mut pts = Vec::new();
        let mut tans = Vec::new();
        for (fig_idx, (start, prims)) in figs.iter().enumerate() {
            let fig = fig_idx as u32;
            cur = *start;
            for (pi, pr) in prims.iter().enumerate() {
                let seg = seg_base + pi as u32; // native segment index of this prim
                let measuring = seg >= start_segment; // skip (position only) segments before start_segment
                let seg_start = acc; // arc length before this segment (what native reports on a hit)
                match *pr {
                    Prim::Line(_, to) => {
                        if measuring {
                            let d = sub(to, cur);
                            if to[0] != cur[0] || to[1] != cur[1] {
                                lastdir = d;
                            }
                            let dist = (d[1] * d[1] + d[0] * d[0]).sqrt();
                            let reached = dist + acc;
                            if reached > len {
                                let frac = 1.0 - (reached - len) / dist;
                                return Some(([frac * d[0] + cur[0], frac * d[1] + cur[1]], [d[0] / dist, d[1] / dist], seg, fig, seg_start));
                            }
                            acc = dist + acc;
                        }
                        cur = to;
                    }
                    Prim::Cubic(_, p1, p2, p3) => {
                        if measuring && flatten::first_tangent_ok(cur, p1, p2, p3, tol) {
                            pts.clear();
                            tans.clear();
                            flatten::flatten_cubic_t(cur, p1, p2, p3, tol, &mut pts, &mut tans);
                            let seed = flatten::first_tangent(cur, p1, p2, p3, tol);
                            if let Some(&lt) = tans.last() {
                                lastdir = lt;
                            }
                            for i in 0..pts.len() {
                                let d = sub(pts[i], cur);
                                let dist = (d[1] * d[1] + d[0] * d[0]).sqrt();
                                let reached = dist + acc;
                                if reached > len {
                                    let frac = 1.0 - (reached - len) / dist;
                                    let prev = if i == 0 { seed } else { tans[i - 1] };
                                    let tx = frac * tans[i][0] + (1.0 - frac) * prev[0];
                                    let ty = frac * tans[i][1] + (1.0 - frac) * prev[1];
                                    let tn = (ty * ty + tx * tx).sqrt();
                                    return Some(([frac * d[0] + cur[0], frac * d[1] + cur[1]], [tx / tn, ty / tn], seg, fig, seg_start));
                                }
                                acc = dist + acc;
                                cur = pts[i];
                            }
                        }
                        cur = p3;
                    }
                }
            }
            seg_base += seg_counts.get(fig_idx).copied().unwrap_or(prims.len() as u32);
        }
        // `len` ran past the end: the current point and the last direction, with the totals.
        let n = (lastdir[1] * lastdir[1] + lastdir[0] * lastdir[0]).sqrt();
        let (point, tangent) = (cur, [lastdir[0] / n, lastdir[1] / n]);
        Some((point, tangent, seg_base, figs.len() as u32, acc))
    }

    // Device-space `(start, prims incl. closing edge, closed)` per figure — the skeleton the
    // length/point-at-length walk follows and the stroker widens (`figure_prims` for paths, the
    // native 16-point construction for ellipse/rounded, 4 lines for a rect). Groups/transformed
    // children compose. Every figure is included (filled/hollow alike).
    /// Every figure as the length walk (ComputeLength, ComputePointAtLength) receives it: the shape's
    /// own stream, nothing dropped. A non-finite rect, rounded rect, ellipse or path figure streams its
    /// points as they are, and segment flags do not split a figure (they only gate stroking).
    /// The figures the length walk measures, through the transforming sink of `m`: a figure's start by
    /// the full product (`BeginFigure`), its other points by the matrix's own form ([`sink_point`]). The
    /// world sink is left out under an identity; a transformed shape always wraps its own.
    fn length_figures(&self, m: &Matrix, out: &mut Vec<(Point, Vec<Prim>)>) {
        self.length_figures_wrap(m, false, out)
    }

    fn length_figures_wrap(&self, m: &Matrix, wrap: bool, out: &mut Vec<(Point, Vec<Prim>)>) {
        let ts = |p| if wrap { transform_point(m, p) } else { map_point(m, p) };
        let tp = |p| sink_point(m, p);
        let map = |s: Point, pr: Vec<Prim>| (ts(s), pr.iter().map(|p| match *p {
            Prim::Line(a, b) => Prim::Line(tp(a), tp(b)),
            Prim::Cubic(a, b, c, d) => Prim::Cubic(tp(a), tp(b), tp(c), tp(d)),
        }).collect());
        match self {
            Geometry::Rect(rect) => {
                let [l, t, r, b] = ordered_rect(*rect);
                let (a, bb, c, d) = (ts([l, t]), tp([r, t]), tp([r, b]), tp([l, b]));
                out.push((a, vec![Prim::Line(a, bb), Prim::Line(bb, c), Prim::Line(c, d), Prim::Line(d, a)]));
            }
            Geometry::Rounded { rect, radius } => {
                let (s, pr) = rounded_rect_prims(*rect, *radius);
                out.push(map(s, pr));
            }
            Geometry::Ellipse { center, radius } => {
                let rect = [center[0] - radius[0], center[1] - radius[1], center[0] + radius[0], center[1] + radius[1]];
                let (s, pr) = rounded_rect_prims(rect, *radius);
                out.push(map(s, pr));
            }
            Geometry::Path { figures, .. } => out.extend(figures.iter().map(|f| figure_prims_by(f, ts, tp))),
            Geometry::Group { children, .. } => children.iter().for_each(|c| c.length_figures_wrap(m, wrap, out)),
            // A transformed shape maps its child's points through its own sink, then the caller's: two
            // roundings per point, not one composed matrix.
            Geometry::Transformed { matrix, child } => {
                let mut inner = Vec::new();
                child.length_figures_wrap(matrix, true, &mut inner);
                out.extend(inner.into_iter().map(|(s, pr)| map(s, pr)));
            }
        }
    }

    fn prims_per_figure(&self, m: &Matrix) -> Vec<(Point, Vec<Prim>, bool, Vec<Point>)> {
        let mut out = Vec::new();
        self.prims_per_figure_into(m, None, &mut out, &mut None, &[]);
        out
    }
    /// Like [`Self::prims_per_figure`] but for STROKING: it keeps a non-finite path figure instead of
    /// dropping it, unless the figure's finite START device coordinate reaches `limit` — the fixed-point
    /// edge-overflow cutoff ([`Geometry::widen_inner`], [`Figure::start_device_maxabs`]). FILL rejects any non-finite figure (fuzz 189),
    /// but native widens a non-finite figure per-segment and renders its finite rails whenever the
    /// edges stay representable: our widener is a line-for-line port and reproduces the result (fuzz
    /// 7408 + the six non-finite strokes it renders byte-exact). `limit` is expressed in `m`-space, so
    /// callers pre-divide the device cutoff by the post-widen scale for the local-then-transform path.
    fn prims_per_figure_keep(&self, m: &Matrix, limit: f32) -> Vec<(Point, Vec<Prim>, bool, Vec<Point>)> {
        let mut out = Vec::new();
        self.prims_per_figure_into(m, Some(limit), &mut out, &mut None, &[]);
        out
    }
    /// Each figure as the fill path walks it — whole, since segment flags only gate stroking — with
    /// whether it is closed and filled: the input of the region scanner (Outline, Combine, area). A
    /// transformed shape streams its child through its own transforming sink before `m` maps it.
    pub(crate) fn fill_figures(&self, m: &Matrix, out: &mut Vec<(Point, Vec<Prim>, bool, bool)>) {
        self.fill_figures_pre(m, &[], out)
    }

    /// [`Self::fill_figures`] under `pre`, the transformed shapes' matrices, innermost first.
    fn fill_figures_pre(&self, m: &Matrix, pre: &[Matrix], out: &mut Vec<(Point, Vec<Prim>, bool, bool)>) {
        match self {
            Geometry::Path { figures, .. } => {
                for fig in figures.iter().filter(|f| !f.has_nonfinite()) {
                    let (start, prims) = if pre.is_empty() {
                        figure_prims(fig, m)
                    } else {
                        let (start, prims) = figure_prims(fig, &IDENTITY);
                        (map_point(m, pre_point(pre, start)), transform_prims(&pre_prims(pre, &prims), m))
                    };
                    out.push((start, prims, fig.closed, fig.filled));
                }
            }
            Geometry::Group { children, .. } => children.iter().for_each(|c| c.fill_figures_pre(m, pre, out)),
            Geometry::Transformed { matrix, child } => {
                let pre: Vec<Matrix> = std::iter::once(*matrix).chain(pre.iter().copied()).collect();
                child.fill_figures_pre(m, &pre, out)
            }
            _ => {
                let mut figs = Vec::new();
                self.prims_per_figure_into(m, None, &mut figs, &mut None, pre);
                // A rectangle streams three sides and lets the closed figure's end close it.
                if let Geometry::Rect(_) = self {
                    figs.iter_mut().for_each(|f| _ = f.1.pop());
                }
                out.extend(figs.into_iter().map(|(s, p, c, _)| (s, p, c, true)));
            }
        }
    }
    /// [`Self::prims_per_figure_into`]'s output, plus for each entry an all-line path figure's vertices
    /// in device space as the widener takes them: each line end mapped in f64 (an identity transform
    /// leaves it untouched), so a coordinate the f32 map overflows stays finite.
    fn widen_figures_f64(&self, m: &Matrix, limit: f32) -> Vec<((Point, Vec<Prim>, bool, Vec<Point>), Option<LineEnds64>)> {
        let (mut out, mut lines) = (Vec::new(), Some(Vec::new()));
        self.prims_per_figure_into(m, Some(limit), &mut out, &mut lines, &[]);
        out.into_iter().zip(lines.unwrap_or_default()).collect()
    }
    /// `pre`: the matrices of the transformed shapes this shape sits in, innermost first; each maps
    /// the shape's points through its own transforming sink ([`sink_point`]) before `m`.
    fn prims_per_figure_into(&self, m: &Matrix, stroke_limit: Option<f32>, out: &mut Vec<(Point, Vec<Prim>, bool, Vec<Point>)>, lines: &mut Option<Vec<Option<LineEnds64>>>, pre: &[Matrix]) {
        match self {
            // Native draws nothing for a rect/rounded/ellipse with a non-finite POSITION coordinate
            // (its rect bounds or ellipse center) — same sink rule as a non-finite path figure
            // ([`Figure::has_nonfinite`]); fuzz 189 = a NaN rect edge. A non-finite RADIUS is NOT
            // rejected (native clamps it and draws, fuzz 155 = inf rounded-rect radius), so the guards
            // check only the position, not the radius.
            Geometry::Rect(r) if !r.iter().all(|v| v.is_finite()) => {}
            Geometry::Rect(rect) => {
                let [l, t, r, b] = ordered_rect(*rect);
                let local = [[l, t], [r, t], [r, b], [l, b]].map(|p| pre_point(pre, p));
                let tp = |p| transform_point(m, p);
                let (a, bb, c, d) = (tp(local[0]), tp(local[1]), tp(local[2]), tp(local[3]));
                out.push((a, vec![Prim::Line(a, bb), Prim::Line(bb, c), Prim::Line(c, d), Prim::Line(d, a)], true, Vec::new()));
                // The widener maps each line end it is streamed through the device matrix in f64.
                if let Some(lines) = lines {
                    let prims: Vec<Prim> = (0..4).map(|i| Prim::Line(local[i], local[(i + 1) % 4])).collect();
                    lines.push(Some(line_ends_f64(local[0], &prims, m, Vec::new())));
                }
            }
            // `Widen`'s sink (an infinite `stroke_limit`) has no edge list: it keeps the shape and the
            // dasher lays the dashes of its finite edges.
            Geometry::Rounded { rect, .. } if !rect.iter().all(|v| v.is_finite()) && stroke_limit != Some(f32::INFINITY) => {}
            Geometry::Rounded { rect, radius } => {
                let (s, local) = rounded_rect_prims(*rect, *radius);
                let (s, local) = (pre_point(pre, s), pre_prims(pre, &local));
                let (local, pr) = shape_stroke_prims(local, m);
                let round = prim_ends(&pr);
                out.push((map_point(m, s), pr, true, round));
                if let Some(l) = lines {
                    l.push(Some(line_ends_f64(s, &local, m, shape_force_f64(&local, m))));
                }
            }
            Geometry::Ellipse { center, .. } if !center.iter().all(|v| v.is_finite()) && stroke_limit != Some(f32::INFINITY) => {}
            Geometry::Ellipse { center, radius } => {
                let rect = [center[0] - radius[0], center[1] - radius[1], center[0] + radius[0], center[1] + radius[1]];
                let (s, local) = rounded_rect_prims(rect, *radius);
                let (s, local) = (pre_point(pre, s), pre_prims(pre, &local));
                let (local, pr) = shape_stroke_prims(local, m);
                let round = prim_ends(&pr);
                out.push((map_point(m, s), pr, true, round));
                if let Some(l) = lines {
                    l.push(Some(line_ends_f64(s, &local, m, shape_force_f64(&local, m))));
                }
            }
            Geometry::Path { figures, .. } => {
                // The shape's own figures decide what it drops; the transformed shapes it sits in
                // only move its points.
                for (_, fig) in figures.iter().map(|f| (f, pre_figure(pre, f))).filter(|(f, p)| match stroke_limit {
                    None => !f.has_nonfinite(),
                    Some(limit) => !f.has_nonfinite() || f.start_device_maxabs_via(&p.start, m) < limit,
                }) {
                    let fig = &fig;
                    // A FORCE_UNSTROKED segment is a gap: split the figure into open stroked sub-runs
                    // at each flagged segment (native D2D1_PATH_SEGMENT_FORCE_UNSTROKED). A
                    // FORCE_ROUND_LINE_JOIN (bit 2) segment forces the join after it to be round
                    // (`figure_prims_flags`). Unflagged figures keep the single closed/open entry.
                    // A cubic/quad with an `inf` control degenerates to a line to its endpoint (native's
                    // stroke render, fuzz 22884); done here on the LOCAL figure so a device-only overflow
                    // still strokes as a curve and voids.
                    if fig.seg_flags.iter().any(|&f| f & 1 != 0) {
                        for (sub, caps) in split_unstroked(fig) {
                            let (s, pr, fr) = figure_prims_flags(&sub, m);
                            out.push((s, pr, false, fr));
                            if let Some(l) = lines {
                                let (ls, local) = figure_prims(&sub, &IDENTITY);
                                l.push(Some(LineEnds64 { run: Some(caps), ..line_ends_f64(ls, &local, m, path_force_f64(&sub, m)) }));
                            }
                        }
                    } else {
                        let (s, pr, fr) = figure_prims_flags(fig, m);
                        out.push((s, pr, fig.closed, fr));
                        if let Some(l) = lines {
                            let (ls, local) = figure_prims(fig, &IDENTITY);
                            l.push(Some(line_ends_f64(ls, &local, m, path_force_f64(fig, m))));
                        }
                    }
                }
            }
            Geometry::Group { children, .. } => {
                for c in children {
                    c.prims_per_figure_into(m, stroke_limit, out, lines, pre);
                }
            }
            // A transformed shape streams its child through its own transforming sink (f32, by the
            // matrix's form); the caller then maps those points.
            Geometry::Transformed { matrix, child } => {
                let pre: Vec<Matrix> = std::iter::once(*matrix).chain(pre.iter().copied()).collect();
                child.prims_per_figure_into(m, stroke_limit, out, lines, &pre)
            }
        }
        // Shapes other than paths carry no f64 line vertices.
        if let Some(l) = lines {
            l.resize_with(out.len(), || None);
        }
    }
}

/// Two shapes' scan bounds as one workspace: the second's when the first is empty, the first's when
/// the second is, else each side widened by the other's on an ordered compare (a NaN side stays).
pub(crate) fn scan_union(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let empty = |r: [f32; 4]| r[0] > r[2] || r[1] > r[3];
    if empty(a) {
        return b;
    }
    let mut u = a;
    if !empty(b) {
        if u[0] > b[0] {
            u[0] = b[0];
        }
        if u[1] > b[1] {
            u[1] = b[1];
        }
        if b[2] > u[2] {
            u[2] = b[2];
        }
        if b[3] > u[3] {
            u[3] = b[3];
        }
    }
    u
}

/// Bounds `[l, t, r, b]` mapped corner by corner through `m` in f32; an axis turns NaN when any
/// corner's coordinate on it is NaN (an infinite side times a zero matrix entry), else spans their
/// min and max.
fn map_bounds([l, t, r, b]: [f32; 4], m: &Matrix) -> [f32; 4] {
    let c = [[l, t], [r, t], [l, b], [r, b]].map(|[x, y]| [(x * m[0] + y * m[2]) + m[4], (y * m[3] + x * m[1]) + m[5]]);
    let span = |i: usize| match c.iter().any(|p| p[i].is_nan()) {
        true => [f32::NAN; 2],
        false => c.iter().fold([f32::INFINITY, f32::NEG_INFINITY], |[lo, hi], p| [lo.min(p[i]), hi.max(p[i])]),
    };
    let ([x0, x1], [y0, y1]) = (span(0), span(1));
    [x0, y0, x1, y1]
}

fn include(lo: &mut [f32; 2], hi: &mut [f32; 2], p: Point) {
    lo[0] = lo[0].min(p[0]);
    lo[1] = lo[1].min(p[1]);
    hi[0] = hi[0].max(p[0]);
    hi[1] = hi[1].max(p[1]);
}

fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1]]
}


// Managed `AccumulateCubicAxisExtrema`: refine one axis' bounds with the cubic's interior extrema.
/// Widen `lo`/`hi` by a cubic's interior extremum on one axis, as native's bounds sink does: only a
/// non-monotone axis is solved; the derivative's roots come from the double quadratic in `r = t/(1−t)`
/// (or `(1−t)/t` when the start difference dominates), and each root's point is the f32 Bernstein sum
/// in native's order, lowering the minimum or else raising the maximum. The end points are the
/// caller's (the current point and `p3`).
fn cubic_axis_extrema(p0: f32, p1: f32, p2: f32, p3: f32, path: bool, lo: &mut f32, hi: &mut f32) -> bool {
    // A path's bounds calculator skips on ordered compares only, so a NaN difference goes on to the
    // solver; the compact shapes' bounds skip it.
    let (d0, d2) = (p1 - p0, p3 - p2);
    let (g0, g1) = ((p3 - p1) * d0, (p2 - p0) * d2);
    if if path { g0 >= 0.0 && g1 >= 0.0 } else { !(g0 < 0.0 || g1 < 0.0) } {
        return false;
    }
    let d1 = p2 - p1;
    let tiny = d1.abs() * 0.000_001;
    if if path { tiny > d0.abs() && tiny > d2.abs() } else { !(tiny <= d0.abs() || tiny <= d2.abs()) } {
        return false;
    }
    let (a, b, c) = (d0 as f64, d1 as f64, d2 as f64);
    let by_start = d0.abs() > d2.abs();
    let (disc, den) = if by_start { (b * b - c * a, a) } else { (b * b - a * c, c) };
    // A path's bounds calculator marks the bounds NaN at a NaN discriminant, root or extremum, as at a
    // NaN point (an infinite control gives `∞ − ∞`).
    let mut nan = disc.is_nan();
    let mut roots = [0.0f32; 2];
    let mut n = 0;
    if disc > 0.0 {
        let s = disc.sqrt();
        let r0 = ((-b - s) / den) as f32;
        roots[0] = r0;
        n = usize::from(r0 > 0.0);
        let r1 = ((s + -b) / den) as f32;
        roots[n] = r1;
        n += usize::from(r1 > 0.0);
        nan |= r0.is_nan() || r1.is_nan();
    }
    let (p1x3, p2x3) = (p1 * 3.0, p2 * 3.0);
    for &r in &roots[..n] {
        let t = if by_start { 1.0 / (r + 1.0) } else { r / (r + 1.0) };
        let u = 1.0 - t;
        let v = (((u * p0) * (u * u) + (p1x3 * t) * (u * u)) + (p2x3 * (t * t)) * u) + (t * p3) * (t * t);
        if *lo > v {
            *lo = v;
        } else if v > *hi {
            *hi = v;
        }
        nan |= v.is_nan();
    }
    path && nan
}

#[cfg(test)]
mod point_and_segment_tests {
    use super::*;

    // Under an identity transform a shape's points pass through untouched: a −0 keeps its sign and an
    // infinite coordinate does not turn its partner NaN (`0·inf`), as native's sink chain leaves them.
    #[test]
    fn identity_transform_keeps_points_bitwise() {
        let prims = transform_prims(&[Prim::Line([f32::INFINITY, -0.0], [1.0, 2.0])], &IDENTITY);
        let Prim::Line(a, _) = prims[0] else { panic!() };
        assert_eq!(a[0], f32::INFINITY);
        assert_eq!(a[1].to_bits(), (-0.0f32).to_bits());
        let figs = Geometry::Rounded { rect: [1.0, -0.0, 9.0, 7.0], radius: [2.0, 2.0] }.prims_per_figure(&IDENTITY);
        let ys: Vec<f32> = figs[0].1.iter().flat_map(|p| match *p { Prim::Line(a, b) => vec![a[1], b[1]], Prim::Cubic(a, b, c, d) => vec![a[1], b[1], c[1], d[1]] }).collect();
        assert!(ys.iter().any(|y| y.to_bits() == (-0.0f32).to_bits()), "the top edge keeps its -0");
        // A real transform still multiplies through.
        assert_eq!(map_point(&[2.0, 0.0, 0.0, 2.0, 1.0, 1.0], [1.0, 1.0]), [3.0, 3.0]);
    }

    // A rect fill past the safe box is strip-clipped like any shape: a left edge at -3.4e38 crosses
    // to the far top-left point, and the in-range corners keep their own coordinates.
    #[test]
    fn huge_rect_fill_is_strip_clipped() {
        let g = Geometry::Rect([82.171_9, 47.755_33, -3.4e38, 32.758_91]);
        let polys = g.flatten_fill(&IDENTITY);
        assert_eq!(polys.len(), 1);
        let p = &polys[0].pts;
        assert!(p.contains(&crate::pixels::FAR) && p.contains(&[82.171_9, 47.755_33]), "{p:?}");
        assert!(p.iter().all(|q| q[0].abs() <= 524_288.0 && q[1].abs() <= 524_288.0));
    }

    // Widening queries fail BAD_NUMBER as native's widener does (VM probes 907100-907169, 26100):
    // a path with any non-finite point, start included, and a rounded shape whose radii differ, give
    // the pen a direction it cannot normalize; under a 1e6 scale `0·inf` turns an inf control's leg
    // NaN, the curve tests flat and widens. An analytic widen (a rectangle, or equal radii) fails only
    // the hit test, on a NaN outline.
    #[test]
    fn widen_queries_fail_as_native() {
        let st = StrokeStyle::default();
        let bad = |g: &Geometry, m: &Matrix, hit: Option<Point>| g.widen_query_bad_number(3.0, &st, m, DEFAULT_TOLERANCE, hit);
        let path = |start: Point, seg: Segment| Geometry::Path {
            fill: FillMode::Alternate,
            figures: vec![Figure { start, filled: true, closed: true, segments: vec![seg], seg_flags: Vec::new() }],
        };
        let inf_ctrl = path([5.0, 5.0], Segment::Quadratic([f32::INFINITY, 10.0], [30.0, 30.0]));
        let big = [1e6, 0.0, 0.0, 1e6, 3.4e38, 660_696.1];
        assert!(bad(&inf_ctrl, &IDENTITY, None) && bad(&inf_ctrl, &IDENTITY, Some([20.0, 10.0])));
        assert!(!bad(&inf_ctrl, &big, None));
        assert!(bad(&path([f32::INFINITY, 5.0], Segment::Line([30.0, 5.0])), &big, None));
        assert!(!bad(&path([5.0, 5.0], Segment::Line([3.4e38, 5.0])), &big, None));
        let nan_rect = Geometry::Rect([5.0, 5.0, f32::NAN, 30.0]);
        assert!(!bad(&nan_rect, &IDENTITY, None) && bad(&nan_rect, &IDENTITY, Some([20.0, 10.0])));
        let ellipse = Geometry::Ellipse { center: [20.0, 20.0], radius: [f32::INFINITY, 5.0] };
        assert!(bad(&ellipse, &IDENTITY, None));
        let nan_radius = Geometry::Rounded { rect: [5.0, 5.0, 30.0, 30.0], radius: [3.0, f32::NAN] };
        let nan_side = Geometry::Rounded { rect: [5.0, 5.0, 30.0, f32::NAN], radius: [3.0, 3.0] };
        assert!(bad(&nan_radius, &IDENTITY, None));
        assert!(!bad(&nan_side, &IDENTITY, None) && bad(&nan_side, &IDENTITY, Some([20.0, 10.0])));
        assert!(!bad(&Geometry::Rect([5.0, 5.0, 30.0, 30.0]), &IDENTITY, Some([20.0, 10.0])));
    }

    // A dashed widening query fails where native's dasher hands the pen a direction it cannot
    // normalize (VM probes, 26100): a figure opening on a dash starts the pen along its first segment,
    // and a dash running into a corner turns it there, so a NaN end fails when the corner falls on a
    // dash and not when it falls in a gap. Under a transform the rectangle widener gives a NaN side
    // its finite perpendicular direction.
    #[test]
    fn dashed_widen_queries_fail_as_native() {
        let on = StrokeStyle { dashes: vec![2.0, 2.0], ..Default::default() };
        let off = StrokeStyle { dashes: vec![2.0, 2.0], dash_offset: 2.5, ..Default::default() };
        let bad = |g: &Geometry, st: &StrokeStyle, m: &Matrix| g.widen_query_bad_number(3.0, st, m, DEFAULT_TOLERANCE, None);
        let nan_end = Geometry::Path {
            fill: FillMode::Alternate,
            figures: vec![Figure { start: [5.0, 5.0], filled: true, closed: true, segments: vec![Segment::Line([30.0, 5.0]), Segment::Line([30.0, f32::NAN])], seg_flags: Vec::new() }],
        };
        assert!(bad(&nan_end, &on, &IDENTITY) && !bad(&nan_end, &off, &IDENTITY));
        let nan_rect = Geometry::Rect([5.0, 5.0, f32::NAN, 30.0]);
        let big = [1e6, 0.0, 0.0, 1e6, 3.4e38, 660_696.1];
        assert!(bad(&nan_rect, &on, &IDENTITY) && !bad(&nan_rect, &on, &big) && !bad(&nan_rect, &off, &IDENTITY));
        assert!(!bad(&Geometry::Rounded { rect: [5.0, 5.0, 30.0, f32::NAN], radius: [3.0, 3.0] }, &on, &big));
    }

    // A rect a FIXED or HAIRLINE pen strokes skips the analytic widen for the rectangle widener (VM
    // probes, 26100): with no transform its zero, NaN or infinite side reaches the pen and fails; under
    // a transform a degenerate column (NaN included) takes the other's perpendicular and passes.
    #[test]
    fn rect_widener_queries_fail_as_native() {
        let fixed = StrokeStyle { transform_type: stroke::TransformType::Fixed, ..Default::default() };
        let bad = |r: [f32; 4], m: &Matrix| Geometry::Rect(r).widen_query_bad_number(3.0, &fixed, m, DEFAULT_TOLERANCE, None);
        let big = [1e6, 0.0, 0.0, 1e6, 3.4e38, 660_696.1];
        for r in [[5.0, 5.0, f32::NAN, 30.0], [5.0, 5.0, f32::INFINITY, 30.0], [5.0, 5.0, 5.0, 30.0], [5.0, f32::NAN, 30.0, 30.0]] {
            assert!(bad(r, &IDENTITY) && !bad(r, &big), "{r:?}");
        }
        assert!(!bad([5.0, 5.0, 30.0, 30.0], &IDENTITY));
    }

    // A pen half-width whose 28.4 conversion overflows f32 (|width|·8 ≳ f32::MAX) fails a STROKED
    // realization create with D2DERR_BAD_NUMBER only for the straight-edged rectangle primitive, at
    // either sign; the analytic ellipse and rounded-rect strokes stay S_OK at any width. (Native probes
    // on 26100.8875: rect ±3.4e38 → BAD_NUMBER, ellipse/rounded ±3.4e38 → S_OK, all at width -1e9 → S_OK.)
    #[test]
    fn stroked_realize_overflow_is_rectangle_only() {
        let fmax = 3.3999999521443642e38f32;
        let rect = Geometry::Rect([7.230661, -0.7130976, 36.283192, 22.818274]);
        let ell = Geometry::Ellipse { center: [22.369406, 7.544503], radius: [53.883808, 3200.0] };
        let rnd = Geometry::Rounded { rect: [7.230661, -0.7130976, 36.283192, 22.818274], radius: [5.0, 5.0] };
        // Rect: BAD_NUMBER at an overflowing width of EITHER sign (fuzz 18171 and 26642).
        assert!(rect.stroked_realize_bad_number(-fmax, &StrokeStyle::default(), DEFAULT_TOLERANCE));
        assert!(rect.stroked_realize_bad_number(fmax, &StrokeStyle::default(), DEFAULT_TOLERANCE));
        // Rect: a finite, non-overflowing huge width still strokes (no BAD_NUMBER).
        assert!(!rect.stroked_realize_bad_number(-1e9, &StrokeStyle::default(), DEFAULT_TOLERANCE));
        assert!(!rect.stroked_realize_bad_number(1e9, &StrokeStyle::default(), DEFAULT_TOLERANCE));
        // Ellipse / rounded-rect: never BAD_NUMBER from width overflow, either sign (fuzz 20640).
        for w in [-fmax, fmax, -1e9, 1e9] {
            assert!(!ell.stroked_realize_bad_number(w, &StrokeStyle::default(), DEFAULT_TOLERANCE), "ellipse w={w:e}");
            assert!(!rnd.stroked_realize_bad_number(w, &StrokeStyle::default(), DEFAULT_TOLERANCE), "rounded w={w:e}");
        }
        // A path (native probes at ±3.4e38, sign-independent): a polyline, a closed figure or a curve
        // overflows; a lone open straight segment does not, nor does a group of one.
        let fig = |closed: bool, segs: Vec<Segment>| Geometry::Path {
            fill: FillMode::Alternate,
            figures: vec![Figure { start: [7.0, 1.0], filled: true, closed, segments: segs, seg_flags: Vec::new() }],
        };
        let line = |to: Point| Segment::Line(to);
        let cubic = Segment::Cubic([9.0, 0.0], [10.0, 9.0], [12.0, 3.0]);
        let bad = [
            fig(true, vec![line([30.0, 4.0]), line([12.0, 20.0])]),
            fig(false, vec![line([30.0, 4.0]), line([12.0, 20.0])]),
            fig(true, vec![line([30.0, 4.0])]),
            fig(false, vec![cubic]),
        ];
        let ok = [fig(false, vec![line([30.0, 4.0])])];
        for w in [-fmax, fmax] {
            for g in &bad {
                assert!(g.stroked_realize_bad_number(w, &StrokeStyle::default(), DEFAULT_TOLERANCE), "{g:?} w={w:e}");
            }
            for g in &ok {
                assert!(!g.stroked_realize_bad_number(w, &StrokeStyle::default(), DEFAULT_TOLERANCE), "{g:?} w={w:e}");
                let grp = Geometry::Group { fill: FillMode::Alternate, children: vec![g.clone()] };
                assert!(!grp.stroked_realize_bad_number(w, &StrokeStyle::default(), DEFAULT_TOLERANCE));
            }
        }
    }

    // An arc's rotation only matters once the conversion reads it: a zero radius is a straight line
    // whatever the rotation, while a real arc with a NaN rotation comes out NaN. A hollow figure's
    // infinite start does not void a fill.
    #[test]
    fn arc_rotation_and_hollow_fill_void() {
        let arc = |radius: Point, rotation: f32| Segment::Arc { to: [36.6, 60.6], radius, rotation, clockwise: true, large: true };
        let fig = |segments: Vec<Segment>, filled: bool| Figure { start: [15.0, 29.3], filled, closed: true, segments, seg_flags: Vec::new() };
        assert!(!fig(vec![arc([-0.0, 0.0], f32::INFINITY)], true).has_nonfinite());
        assert!(fig(vec![arc([20.0, 20.0], f32::INFINITY)], true).has_nonfinite());
        let nan_arc = arc_to_cubics([15.0, 29.3], [36.6, 60.6], [30.0, 30.0], f32::NAN, true, true);
        assert!(nan_arc.iter().flatten().flatten().any(|v| v.is_nan()));
        let hollow_inf_start = Figure { start: [f32::INFINITY, 9.0], ..fig(vec![Segment::Line([52.0, 0.0])], false) };
        let filled_tri = fig(vec![Segment::Line([40.0, 6.0]), Segment::Line([20.0, 34.0])], true);
        assert!(!Geometry::Path { fill: FillMode::Alternate, figures: vec![hollow_inf_start, filled_tri] }.any_nonfinite_fill_void());
    }

    // Native rejects a path arc with a negative radius component (E_INVALIDARG); positive radii and
    // non-arc geometries are fine. (fuzz 20003/20041 vs 20015.)
    #[test]
    fn negative_arc_radius_is_invalid() {
        let arc = |r: Point| Geometry::Path {
            fill: FillMode::Alternate,
            figures: vec![Figure {
                start: [0.0, 0.0],
                filled: true,
                closed: true,
                segments: vec![Segment::Arc { to: [10.0, 10.0], radius: r, rotation: 0.0, clockwise: true, large: false }],
                seg_flags: Vec::new(),
            }],
        };
        assert!(!arc([20.0, 10.0]).has_invalid_arc_radius());
        assert!(arc([-20.0, 10.0]).has_invalid_arc_radius()); // negative x
        assert!(arc([20.0, -1.0]).has_invalid_arc_radius()); // negative y
        assert!(!Geometry::Ellipse { center: [0.0, 0.0], radius: [-5.0, -5.0] }.has_invalid_arc_radius()); // not a path arc
    }

    // The 1c2-01 'g' figure: start (20,60), cubic → (110,60), quadratic → (20,60), closed.
    fn g() -> Geometry {
        Geometry::Path {
            fill: FillMode::Alternate,
            figures: vec![Figure {
                start: [20.0, 60.0],
                filled: true,
                closed: true,
                segments: vec![Segment::Cubic([40.0, 12.0], [92.0, 12.0], [110.0, 60.0]), Segment::Quadratic([65.0, 92.0], [20.0, 60.0])],
                seg_flags: Vec::new(),
            }],
        }
    }

    // Expected (end_segment, end_figure, length_to_end) measured against Direct2D
    // (10.0.26100.8875) ID2D1PathGeometry1::ComputePointAndSegmentAtLength at tol 0.25, identity
    // transform — a capture probe. Point/tangent reuse the oracle-verified
    // point_at_length (scene 1c2-01), spot-checked here; the new fields are what this asserts.
    #[test]
    fn matches_native() {
        let g = g();
        for &(len, eseg, efig, elen) in &[(120.0f32, 0u32, 0u32, 0.0f32), (160.0, 1, 0, 122.459846), (1000.0, 3, 1, 219.529526)] {
            let (_, _, seg, fig, l) = g.point_and_segment_at_length(len, 0, &IDENTITY, 0.25).unwrap();
            assert_eq!((seg, fig), (eseg, efig), "len {len}");
            assert!((l - elen).abs() < 0.01, "len {len}: length_to_end {l} != {elen}");
        }
        let (p, t, ..) = g.point_and_segment_at_length(160.0, 0, &IDENTITY, 0.25).unwrap();
        assert!((p[0] - 75.942078).abs() < 0.01 && (p[1] - 75.041054).abs() < 0.01, "point {p:?}");
        assert!((t[0] + 0.985378).abs() < 0.001 && (t[1] - 0.170384).abs() < 0.001, "tangent {t:?}");
        // start_segment skips earlier segments; length_to_end is measured from there.
        let (_, _, seg, fig, l) = g.point_and_segment_at_length(5.0, 1, &IDENTITY, 0.25).unwrap();
        assert_eq!((seg, fig), (1, 0));
        assert!(l.abs() < 0.01, "start_segment length_to_end {l}");
    }

    // FillEllipse near-FLT_MAX overflow flip, pinned byte-exact to native d2d1 10.0.26100.8875 by an
    // SSH probe ladder (elllad: center [48,48], other semi-axis 25, 96×96 dpi96, solid fill). Native's
    // The float bezier flattener forms 6·Δ² of the transformed ellipse arc and tests the
    // component max; that magnitude is 6·kappa·R, so at |R| ≥ FLT_MAX/(6·kappa) = 1.0269e38 it overflows
    // to ±inf, the forward difference goes non-finite, and native draws NOTHING. The exact flip sits
    // between 1.026e38 (draws) and 1.027e38 (voids), both signs and both axes — all 32 ladder points
    // byte-exact. flatten_fill drops the closed figure iff it voids, so an empty polyline set is the
    // "draws nothing" signal; this guards the exact flip (and that it is NOT a hypot length nor
    // sqrt(x²+y²), which would flip at ~1.03e38-via-length or ~1.8e19 respectively).
    #[test]
    fn ellipse_near_fltmax_overflow_flip() {
        let draws = |radius: [f32; 2]| {
            !Geometry::Ellipse { center: [48.0, 48.0], radius }.flatten_fill(&IDENTITY).is_empty()
        };
        for &r in &[1.0e38f32, 1.02e38, 1.026e38] {
            for &s in &[1.0f32, -1.0] {
                assert!(draws([r * s, 25.0]), "x-axis {r}*{s} should draw");
                assert!(draws([25.0, r * s]), "y-axis {r}*{s} should draw");
            }
        }
        for &r in &[1.027e38f32, 1.1e38, 1.2e38, 1.27e38, 1.3e38] {
            for &s in &[1.0f32, -1.0] {
                assert!(!draws([r * s, 25.0]), "x-axis {r}*{s} should void");
                assert!(!draws([25.0, r * s]), "y-axis {r}*{s} should void");
            }
        }
    }
}
