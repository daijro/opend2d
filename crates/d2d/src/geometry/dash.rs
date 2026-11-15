//! Dashing — the MIL stroker's dasher (MIT-licensed `dotnet/wpf`
//! `WpfGfx/core/geometry/strokefigure.{cpp,h}`), verified byte-for-byte against Direct2D. The dasher
//! sits between the figure walk and the
//! pen (widener → dasher → pen): it buffers one edge of segments at a time, lays the
//! dash pattern along it (in `f64`, in dash-space ↔ edge-space), and emits every "on" dash as an
//! independent **open** sub-stroke — start cap, interior joins, end cap — which `stroke::widen_one`
//! widens exactly as Direct2D's pen would. A zero-length dash becomes a cap "dot"
//! (`stroke::dash_dot`: round→disc, square→square, flat→nothing).
//!
//! The pattern persists across corners: a dash that does not end at a corner spans it with the
//! figure's join; one that ends at a corner is cut with a dash cap. A closed figure that starts
//! mid-dash abuts its first and last dash with flat caps at the start vertex (`first_cap_pending`),
//! reproduced here by holding the first dash and either merging it with the spanning last dash
//! (one outline through the start vertex) or, when the figure closes on a gap, emitting it flat plus
//! the zero-length start-vertex dash native appends.
//!
//! The figure arrives in **device space**, but the pattern is laid in **pen space**: dash lengths are
//! multiples of the untransformed pen width, and each segment's length is measured through the
//! inverse of the pen transform (a quadratic form), so dashes stretch with the transform exactly as
//! Direct2D's do, non-uniform ones included. Curves are dashed along
//! their flattened skeleton with chord widening — the same curve-widening precision class as the
//! non-dashed path, not byte-exact.

use super::stroke::{self, Cap, PenShape, StrokeStyle};
use super::{Point, Prim};

type V = [f64; 2];

fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1]]
}
fn len(a: V) -> f64 {
    (a[0] * a[0] + a[1] * a[1]).sqrt() // Direct2D uses sqrt(x²+y²), not hypot
}

const FUZZ: f64 = 1e-6;
const SQ_LENGTH_FUZZ: f64 = 1e-4;
const TOLERANCE: f64 = 0.25;
// Dash-end / segment-end coincidence epsilon (Direct2D literal `1e-6`): only a dash
// boundary within this of the final segment end is snapped to it. A larger value would absorb the
// sub-tolerance gap between a near-miss corner and a dash boundary, shifting every later dash.
const COINCIDENCE: f64 = 1e-6;

/// The safe device box half-size: a dash piece wholly outside it is clipped away before filling.
const SAFE_BOX: f64 = 524287.0;

// Single-precision fmod with sign handling, clamped to [0, m].
fn gp_mod_f(a: f32, m: f32) -> f32 {
    let v = if a >= 0.0 {
        if m <= a {
            a - ((a / m) as i32 as f32) * m
        } else {
            a
        }
    } else {
        let neg = -a;
        let rem = if m <= neg { neg - ((neg / m) as i32 as f32) * m } else { neg };
        m - rem
    };
    v.max(0.0).min(m)
}

// --- the dash sequence -----------------------------------------------------------------------
// Dash-space is the position along the (shifted) dash array; edge-space is the arc length along
// the current edge. The cumulative-ends array holds count+1 ends, shifted so 0 is the dash-offset
// start; dash k spans dashes[k-1]..dashes[k] and is "on" iff k is odd.
struct DashSeq {
    dashes: Vec<f64>,
    len: f64,
    start_dash: usize,
    cur: usize,
    loc: f64,   // current location (dash space)
    edge0: f64, // edge-space start recorded at the last new-edge prepare
    iter: u32,  // iterations wrapped since then
    scale: f64, // dash-space -> edge-space stretch: a period under 0.1 device px is stretched to it
    inv_scale: f64,
}

impl DashSeq {
    // Build the shifted cumulative array; find the dash the offset lands in.
    fn new(style: &StrokeStyle, width: f64) -> Option<DashSeq> {
        let pen = style.pen_dashes()?;
        let n0 = pen.len();
        // Direct2D requires an EVEN dash count (an odd count is E_INVALIDARG), but an
        // odd dash array still draws dashed: D2D's stroke style repeats an odd array once so the on/off
        // alternation is consistent (a single `[a]` becomes `[a, a]` = equal on/off), then the dasher
        // gets the even count. Mirror that here (15299's `[5.105]`), rather than drawing nothing.
        let src: Vec<f32> = if n0 % 2 == 0 { pen.clone() } else { pen.iter().chain(&pen).copied().collect() };
        let n = src.len();
        let w = width;
        let mut dashes = vec![0.0f64; n + 1];
        for i in 0..n {
            dashes[i + 1] = src[i] as f64 * w + dashes[i];
        }
        let total = dashes[n];
        if total.is_nan() || total <= 0.0 {
            return None;
        }
        // Wrap the offset into [0, total) — single-precision GpModF, exactly as native.
        let off_raw = style.dash_offset as f64 * w;
        let mut off = off_raw;
        if off < 0.0 || total <= off {
            off = gp_mod_f(off_raw as f32, total as f32) as f64;
            if off < 0.0 || total <= off {
                off = 0.0;
            }
        }
        let mut start_dash = 1usize;
        while start_dash < n && dashes[start_dash] < off {
            start_dash += 1;
        }
        for d in &mut dashes {
            *d -= off;
        }
        Some(DashSeq { dashes, len: total, start_dash, cur: start_dash, loc: 0.0, edge0: 0.0, iter: 0, scale: 1.0, inv_scale: 1.0 })
    }

    fn is_on_dash(&self) -> bool {
        self.cur & 1 == 1
    }
    fn dash_to_edge(&self, d: f64) -> f64 {
        let x = if self.iter == 0 { d - self.edge0 } else { self.iter as f64 * self.len + d - self.edge0 };
        x * self.scale
    }
    fn edge_to_dash(&self, e: f64) -> f64 {
        let x = e * self.inv_scale;
        (if self.iter == 0 { x } else { x - self.iter as f64 * self.len }) + self.edge0
    }
    /// Entering a segment whose device length per edge unit is `ratio`: a period shorter than a tenth of
    /// a device pixel is stretched to one (the 0.1 is the single-precision literal).
    fn update_with_new_segment(&mut self, ratio: f64) {
        let period = ratio * self.len;
        let min = 0.1f32 as f64;
        if period >= min {
            (self.scale, self.inv_scale) = (1.0, 1.0);
        } else {
            self.scale = min / period;
            self.inv_scale = 1.0 / self.scale;
        }
    }
    fn next_endpoint(&self) -> f64 {
        self.dash_to_edge(self.dashes[self.cur])
    }
    /// What remains of the current dash or gap, in dash space (`GetStep`).
    fn step(&self) -> f64 {
        self.dashes[self.cur] - self.loc
    }
    /// The length of the dash after the current gap (`GetLengthOfNextDash`).
    fn next_dash_len(&self) -> f64 {
        let i = if self.cur == self.dashes.len() - 1 { 0 } else { self.cur };
        self.dashes[i + 1] - self.dashes[i]
    }
    fn prepare_for_new_edge(&mut self) {
        self.iter = 0;
        self.edge0 = self.loc;
    }
    fn reset(&mut self) {
        self.cur = self.start_dash;
        self.loc = 0.0;
    }
    fn advance_to(&mut self, edge_loc: f64) {
        self.loc = self.edge_to_dash(edge_loc);
    }
    fn increment(&mut self) {
        self.loc = self.dashes[self.cur];
        self.cur += 1;
        if self.cur >= self.dashes.len() {
            self.cur = 1;
            self.iter += 1;
            self.loc = self.dashes[0];
        }
    }
}

// --- one edge's segment buffer ---------------------------------------------------------------
// Entry 0 is the edge start (location 0); each later entry is a segment end with its cumulative
// edge-space (pen-space) location, chord direction per edge unit (for point interpolation and
// straight widening) and analytic tangent (for curve-point offset). `is_line`/`is_last` are the
// per-segment flags: a line segment widens straight (`AdvanceToPoint`), a curve segment smoothly
// along the interpolated tangent (`AcceptCurvePoint`). Near-zero segments are dropped (FUZZ gate).
#[derive(Default)]
struct Segs {
    pts: Vec<V>,
    loc: Vec<f64>,
    dir: Vec<V>,         // device chord per edge unit of the segment ending at this entry
    tan: Vec<V>,         // analytic tangent: = chord dir for a line, the curve tangent else
    is_line: Vec<bool>,  // straight segment ⇒ AdvanceToPoint, else AcceptCurvePoint
    is_last: Vec<bool>,  // marks the cubic's final flattened point
    ratio: Vec<f64>,     // device length per edge unit of the segment ending here
    q: [f64; 3],         // the edge-length quadratic form (pen space: the inverse of the pen transform)
    cur: usize,          // m_uCurrentSegment
}

impl Segs {
    fn push_entry(&mut self, pt: V, loc: f64, dir: V, tan: V, is_line: bool, is_last: bool, ratio: f64) {
        self.ratio.push(ratio);
        self.pts.push(pt);
        self.loc.push(loc);
        self.dir.push(dir);
        self.tan.push(tan);
        self.is_line.push(is_line);
        self.is_last.push(is_last);
    }
    fn start_with(&mut self, pt: V, dir: V) {
        self.reset();
        self.push_entry(pt, 0.0, dir, dir, true, false, 0.0);
    }
    // Add a segment: `tangent` None ⇒ a line segment (tangent = chord direction); Some ⇒ a curve
    // segment carrying the analytic tangent. Edge locations are pen-space lengths (the quadratic form
    // `q`), so the chord direction is the device chord per edge unit.
    fn add(&mut self, pt: V, tangent: Option<V>, is_last: bool) {
        let [dx, dy] = sub(pt, *self.pts.last().unwrap());
        let q = self.q;
        let l = (dx * q[1] * dy + dx * q[0] * dx + dy * q[2] * dy).sqrt();
        if l >= FUZZ {
            let r = 1.0 / l;
            let chord = [r * dx, r * dy];
            let loc = l + self.loc.last().unwrap();
            let ratio = (dy * dy + dx * dx).sqrt() / l;
            self.push_entry(pt, loc, chord, tangent.unwrap_or(chord), tangent.is_none(), is_last, ratio);
        }
    }
    fn reset(&mut self) {
        self.pts.clear();
        self.loc.clear();
        self.dir.clear();
        self.tan.clear();
        self.is_line.clear();
        self.is_last.clear();
        self.ratio.clear();
        self.cur = 1;
    }
    fn is_empty(&self) -> bool {
        self.pts.len() < 2
    }
    fn is_last(&self) -> bool {
        self.pts.len() - 1 == self.cur
    }
    fn get_length(&self) -> f64 {
        *self.loc.last().unwrap()
    }
    fn current_end(&self) -> f64 {
        self.loc[self.cur]
    }
    fn cur_is_curve(&self) -> bool {
        !self.is_line[self.cur]
    }
    fn cur_is_last(&self) -> bool {
        self.is_last[self.cur]
    }
    fn increment(&mut self) -> bool {
        self.cur += 1;
        self.cur >= self.pts.len()
    }
    /// The edge location the pattern may jump to from `d` on the current segment: where its chord next
    /// enters the box `[-reach, reach]²` (or the segment end when it never does after `d`); `None`
    /// when `d` is already inside.
    fn box_entry_after(&self, d: f64, reach: f64) -> Option<f64> {
        let (a, b) = (self.pts[self.cur - 1], self.pts[self.cur]);
        let (la, lb) = (self.loc[self.cur - 1], self.loc[self.cur]);
        if !(lb > la) || !(d >= la) {
            return None;
        }
        // Liang-Barsky clip of the chord against the box, in chord parameter t.
        let (mut t0, mut t1) = (0.0f64, 1.0f64);
        for k in 0..2 {
            let dk = b[k] - a[k];
            for (p, q) in [(-dk, a[k] + reach), (dk, reach - a[k])] {
                if p == 0.0 {
                    if q < 0.0 {
                        t0 = 1.0;
                        t1 = 0.0;
                    }
                } else {
                    let r = q / p;
                    if p < 0.0 { t0 = t0.max(r) } else { t1 = t1.min(r) }
                }
            }
        }
        let to_loc = |t: f64| la + t * (lb - la);
        if !(t0 <= t1) || d > to_loc(t1) {
            return Some(lb);
        }
        (d < to_loc(t0)).then(|| to_loc(t0))
    }
    // (point, tangent) at an edge-space location on the current segment (`ProbeAt`). The point is
    // interpolated along the chord direction; the tangent defaults to the segment tangent and, for a
    // curve segment, is interpolated between the previous and current segment tangents.
    fn probe_at(&self, mut rloc: f64, at_seg_end: bool) -> (V, V) {
        let cur = self.cur;
        if at_seg_end || rloc > self.loc[cur] {
            return (self.pts[cur], self.tan[cur]);
        }
        rloc -= self.loc[cur - 1];
        if rloc < 0.0 {
            rloc = 0.0;
        }
        let base = self.pts[cur - 1];
        let dir = self.dir[cur];
        let tan = if self.is_line[cur] {
            self.tan[cur]
        } else {
            let span = self.loc[cur] - self.loc[cur - 1];
            let f = rloc / span;
            let g = 1.0 - f;
            [g * self.tan[cur - 1][0] + f * self.tan[cur][0], g * self.tan[cur - 1][1] + f * self.tan[cur][1]]
        };
        ([base[0] + dir[0] * rloc, base[1] + dir[1] * rloc], tan)
    }
}

/// One point of a dash in progress, as the dasher hands it to the pen: its tangent (a curve point's
/// analytic tangent, else the direction), whether it ends its Bézier, whether it lies on a line segment
/// (`AcceptLinePoint`) rather than a curve (`AcceptCurvePoint`), and the corner the pen draws there.
#[derive(Clone, Copy)]
pub(super) struct DashPt {
    pub(super) tan: V,
    pub(super) last: bool,
    pub(super) line: bool,
    /// The corner the pen turns after this point: its vertex, incoming and outgoing directions. The
    /// vertex is the edge's own end, which differs from the point when the dasher dropped the edge (a
    /// NaN-length edge ends at a NaN vertex).
    pub(super) corner: Option<(V, V, V)>,
    /// That corner comes after a segment the widener skipped: a miter limited to 1.
    pub(super) skip: bool,
    /// False for an entry that only turns a further corner (the dasher drops the zero- or NaN-length
    /// edge between two corners, so the pen gets no point for it).
    pub(super) accept: bool,
}

// --- the dasher ------------------------------------------------------------------------------
struct Dasher<'a> {
    seq: DashSeq,
    segs: Segs,
    style: &'a StrokeStyle,
    pen: &'a PenShape,
    dash_cap: Cap,
    pen_down: bool,
    /// The dash being walked started outside the viewable region and is not drawn (`m_fIgnoreDash`).
    ignore: bool,
    first_cap_pending: bool,
    holding_first: bool,
    started: bool,
    cur: Vec<V>,              // the dash in progress (device polyline)
    cur_tan: Vec<DashPt>,     // per-point pen call data (aligned with `cur`), replayed for curve dashes
    cur_is_curve: bool,       // any point came from a curve segment ⇒ widen smoothly, not as chords
    cur_start_cap: Cap,
    cur_start_dir: V,
    held_first: Option<(Vec<V>, Cap, V, Vec<V>)>, // (polyline, dash-cap end, start dir, skip corners) when first_cap_pending
    // `Widen`'s open figure: its start cap, drawn as its own figure after the last dash.
    deferred_start_cap: Option<(V, V, Cap)>,
    pieces: Vec<Vec<Point>>,
    errored: bool, // the pen met a direction it cannot normalize (BAD_NUMBER)
    // Corners whose join is forced round: a rounded rect's or ellipse's every vertex (native's shape
    // forces its pen's joins round), a path's FORCE_ROUND_LINE_JOIN vertices.
    force_pts: &'a [V],
}

impl<'a> Dasher<'a> {
    fn new(seq: DashSeq, style: &'a StrokeStyle, pen: &'a PenShape, force_pts: &'a [V]) -> Dasher<'a> {
        Dasher {
            force_pts,
            seq,
            segs: Segs { q: pen.dash_q, ..Segs::default() },
            style,
            pen,
            dash_cap: style.dash_cap,
            pen_down: false,
            ignore: false,
            first_cap_pending: false,
            holding_first: false,
            started: false,
            cur: Vec::new(),
            cur_tan: Vec::new(),
            cur_is_curve: false,
            cur_start_cap: Cap::Flat,
            cur_start_dir: [0.0, 0.0],
            held_first: None,
            deferred_start_cap: None,
            pieces: Vec::new(),
            errored: false,
        }
    }

    // --- pen operations, captured as dash sub-strokes ---------------------------------------
    fn begin_dash(&mut self, pt: V, dir: V, cap: Cap) {
        self.cur.clear();
        self.cur.push(pt);
        self.cur_tan.clear();
        self.cur_tan.push(DashPt { tan: dir, last: false, line: true, corner: None, skip: false, accept: true });
        self.cur_is_curve = false;
        self.cur_start_cap = cap;
        self.cur_start_dir = dir;
        self.started = true;
    }
    // Straight/line dash: widen the chord polyline with joins (the pen advancing point to point).
    /// The vertices of the current dash's corners that come after a skipped segment.
    fn cur_skip(&self) -> Vec<V> {
        self.cur_tan.iter().filter(|d| d.skip).filter_map(|d| d.corner.map(|c| c.0)).collect()
    }
    /// `skip`: the dash's corners after a skipped segment (a miter limited to 1).
    fn emit_line(&mut self, pts: &[V], start_cap: Cap, end_cap: Cap, dir: V, skip: &[V]) {
        // >= 2 distinct points → an open stroke (interior joins); else a cap dot.
        let distinct = pts.len() >= 2 && pts[1..].iter().any(|&p| len(sub(p, pts[0])) >= TOLERANCE * SQ_LENGTH_FUZZ);
        if distinct {
            let dev: Vec<V> = pts.to_vec();
            // `Widen` streams every pen call: a dash extended to a vertex and ended there repeats it.
            let end_repeats = if self.pen.raw.is_some() { pts.windows(2).rev().take_while(|w| w[0] == w[1]).count() } else { 0 };
            let mut run = stroke::RunEnd { end_repeats, skip_pts: skip.to_vec(), ..Default::default() };
            self.pieces.extend(stroke::widen_one_flags(&dev, false, self.pen, self.style, start_cap, end_cap, self.force_pts, &mut self.errored, false, &mut run));
        } else {
            // A dash started where the figure ends is only its caps; one started and ended at a point
            // draws the zero-length line between.
            let dot = if pts.len() == 1 {
                stroke::cap_figure(pts[0], dir, self.pen, self.style, start_cap, end_cap, &mut self.errored)
            } else {
                stroke::dash_dot(pts[0], dir, self.pen, self.style, start_cap, end_cap, &mut self.errored)
            };
            if dot.len() >= 2 {
                self.pieces.push(dot);
            }
        }
    }
    fn emit_dash(&mut self, end_cap: Cap) {
        // `Widen` streams a line dash that turns a corner as the pen draws it, so a corner reached through
        // a zero-length edge keeps the edge's direction and its own join.
        let turns = self.pen.raw.is_some() && !self.holding_first && self.cur_tan.iter().any(|d| d.corner.is_some());
        if self.cur_is_curve || turns {
            // Curve dash: drive the pen's AcceptCurvePoint per point (smooth tangent offset). A curve
            // FIRST dash is emitted in place with its flat start, as native emits every dash, so the
            // hold ends here rather than catching the next dash (r9 14350: a line dash lost its cap).
            self.holding_first = false;
            let pts = std::mem::take(&mut self.cur);
            let tans = std::mem::take(&mut self.cur_tan);
            self.cur_is_curve = false;
            let seq: Vec<(V, DashPt)> = pts.iter().copied().zip(tans).collect();
            let poly = stroke::widen_curve_dash(&seq, self.cur_start_cap, end_cap, self.pen, self.style, self.force_pts, &mut self.errored);
            if poly.len() >= 2 {
                self.pieces.push(poly);
            }
            return;
        }
        let cur = std::mem::take(&mut self.cur);
        let skip = self.cur_skip();
        self.cur_tan.clear();
        if self.holding_first {
            self.held_first = Some((cur, end_cap, self.cur_start_dir, skip));
            self.holding_first = false;
            return;
        }
        self.emit_line(&cur, self.cur_start_cap, end_cap, self.cur_start_dir, &skip);
    }

    // --- figure walk entry points (the pen-interface overrides) -----------------------------
    fn start_figure(&mut self, pt: V, dir: V, closed: bool, cap: Cap) {
        self.seq.reset();
        self.pen_down = self.seq.is_on_dash();
        self.first_cap_pending = false;
        if self.pen_down {
            // A figure that opens on a dash starts the pen along its first segment.
            self.errored |= !self.pen.normalizes(dir);
            let mut c = cap;
            if closed {
                // The first dash starts flat; the last dash abuts it at the start vertex. Filling
                // holds the first dash to merge the two; `Widen` streams it as its own figure.
                self.first_cap_pending = true;
                self.holding_first = self.pen.raw.is_none();
                c = Cap::Flat;
            } else if self.pen.raw.is_some() {
                // `Widen` starts an open figure's first dash flat and, after its last dash, draws the
                // start cap as a figure of its own.
                if cap != Cap::Flat {
                    self.deferred_start_cap = Some((pt, dir, cap));
                }
                c = Cap::Flat;
            } else {
                // Open figure: native's dasher realizes the figure START cap as a separate
                // zero-length cap-shape DOT at the start vertex — `dash_dot(startCap, startCap)`,
                // the full shape — then starts the first dash's body with a FLAT cap, unioned under
                // nonzero winding. (A non-dashed stroke keeps the inline start cap: this path is
                // dasher-only.) Verified byte-exact on dashrect-08 (square) and the round/[2,2]-dash
                // captures: the dot's forward half overlaps the body, adding the interior edges that
                // give native's +4 coverage at the cap/rail junction. A Flat start cap yields an
                // empty dot (no change).
                // A Flat start cap makes no dot at all (Widen streams no figure for it).
                if cap != Cap::Flat {
                    let dot = stroke::dash_dot(pt, dir, self.pen, self.style, cap, cap, &mut self.errored);
                    if dot.len() >= 2 {
                        self.pieces.push(dot);
                    }
                }
                c = Cap::Flat;
            }
            self.begin_dash(pt, dir, c);
        }
        self.segs.start_with(pt, dir);
    }
    // Feed one figure-walk point into the edge buffer: `tangent` None ⇒ a line point
    // (`AcceptLinePoint`), Some ⇒ a curve point carrying its analytic tangent (`AcceptCurvePoint`).
    fn accept(&mut self, pt: V, tangent: Option<V>, is_last: bool) {
        self.segs.add(pt, tangent, is_last);
    }
    fn do_corner(&mut self, center: V, dir_in: V, dir_out: V, after_skip: bool) {
        self.flush(false);
        // A dash running through the corner: the pen draws the corner there (a curve dash replays it),
        // failing on an outgoing direction it cannot normalize.
        if self.pen_down {
            self.errored |= !self.pen.normalizes(dir_out);
            match self.cur_tan.last_mut() {
                // A second corner with no point between (a zero-length edge, which the dasher drops):
                // the pen turns again at the same point.
                Some(last) if last.corner.is_some() => {
                    if let Some(&p) = self.cur.last() {
                        self.cur.push(p);
                        self.cur_tan.push(DashPt { tan: dir_in, last: false, line: true, corner: Some((center, dir_in, dir_out)), skip: after_skip, accept: false });
                    }
                }
                Some(last) => {
                    last.corner = Some((center, dir_in, dir_out));
                    last.skip = after_skip;
                }
                None => {}
            }
        }
        // pen.DoCorner is implicit: the corner vertex is already in `cur` (pushed when the edge end
        // was reached), so widen_one applies the join there.
        self.segs.start_with(center, dir_out);
    }
    fn end_stroke_open(&mut self, pt_end: V, vec_end: V, end_cap: Cap, start_cap: Cap) {
        self.flush(true);
        if self.pen_down {
            self.emit_dash(end_cap);
        } else if !self.started && self.seq.is_on_dash() {
            // Never laid a dash but the whole figure is within one — a single point: the pen's start
            // and end caps, no segment between.
            let dot = stroke::cap_figure(pt_end, vec_end, self.pen, self.style, start_cap, end_cap, &mut self.errored);
            if dot.len() >= 2 {
                self.pieces.push(dot);
            }
        }
        if let Some((pt, dir, cap)) = self.deferred_start_cap.take() {
            // The start cap restarts the dasher at the figure's start, ending there with the dash cap.
            let fig = stroke::cap_figure(pt, dir, self.pen, self.style, cap, self.dash_cap, &mut self.errored);
            if fig.len() >= 2 {
                self.pieces.push(fig);
            }
        }
    }
    fn end_stroke_closed(&mut self, pt_end: V, vec_end: V) {
        self.flush(!self.first_cap_pending);
        if self.pen_down {
            if self.first_cap_pending && self.pen.raw.is_some() {
                // `Widen` streams the last dash as the pen draws it: turning the corner at the start
                // vertex onto the first edge, ending flat there.
                self.cur_is_curve = true;
                self.emit_dash(Cap::Flat);
            } else if self.first_cap_pending {
                self.merge_and_emit();
            } else {
                self.emit_dash(self.dash_cap); // started on a gap — plain dash cap
            }
        } else {
            if self.first_cap_pending && self.dash_cap != Cap::Flat {
                // No last dash: the pen starts a figure at the start vertex with the dash cap and ends
                // it closed (flat), with no segment between.
                let dot = stroke::cap_figure(pt_end, vec_end, self.pen, self.style, self.dash_cap, Cap::Flat, &mut self.errored);
                if dot.len() >= 2 {
                    self.pieces.push(dot);
                }
            }
            self.emit_held_first();
        }
    }
    // The spanning last dash and the held first dash are one continuous stroke through the start
    // vertex: concatenate them (the shared vertex becomes an interior join) and cap both free ends
    // with the dash cap — equivalent to native's two abutting flat caps at the vertex.
    fn merge_and_emit(&mut self) {
        // A first dash that never ended is the pen's stroke over the whole figure: the pen closes it,
        // turning the corner at the start vertex onto the first edge and ending flat there.
        if self.holding_first {
            self.cur_is_curve = true;
        }
        if self.cur_is_curve {
            // A curve last dash goes through the pen like any curve dash and abuts the first dash with
            // a flat end at the start vertex (the pen's closed ending); the first stays its own dash.
            self.emit_dash(Cap::Flat);
            self.emit_held_first();
            return;
        }
        let mut combined = std::mem::take(&mut self.cur);
        let mut skip = self.cur_skip();
        let dir = self.cur_start_dir;
        let Some((first, _end, _d, first_skip)) = self.held_first.take() else {
            // The first dash was a curve, already emitted: the last abuts it with a flat end.
            self.emit_line(&combined, self.cur_start_cap, Cap::Flat, dir, &skip);
            return;
        };
        combined.extend_from_slice(&first[1.min(first.len())..]);
        skip.extend(first_skip);
        self.emit_line(&combined, self.dash_cap, self.dash_cap, dir, &skip);
    }
    fn emit_held_first(&mut self) {
        if let Some((first, end_cap, dir, skip)) = self.held_first.take() {
            self.emit_line(&first, Cap::Flat, end_cap, dir, &skip);
        }
    }

    // --- dash/gap bookkeeping (`StartANewDash`/`ExtendCurrentDash`/`TerminateCurrentDash`) ----
    /// `world_len`: the dash's length on the device (its dash-space length times the segment's scale).
    fn start_a_new_dash(&mut self, rloc: f64, at_vertex: bool, world_len: f64) {
        let (pt, dir) = self.segs.probe_at(rloc, at_vertex);
        // A dash starting at a non-finite point fails Direct2D's viewable test and is not started
        // (the pattern runs on; its end does nothing).
        if !(pt[0].is_finite() && pt[1].is_finite()) {
            return;
        }
        // A draw starts no dash whose start lies outside the viewable region grown by the dash's own
        // length: the pen stays up through it, and the pattern runs on.
        if let Some([l, t, r, b]) = self.pen.viewable {
            if !(pt[0] >= l - world_len && pt[0] <= r + world_len && pt[1] >= t - world_len && pt[1] <= b + world_len) {
                self.ignore = true;
                self.started = true;
                return;
            }
        }
        self.ignore = false;
        self.pen_down = true;
        // The pen starts the dash along its direction (`StartFigure`), failing on one it cannot
        // normalize.
        self.errored |= !self.pen.normalizes(dir);
        self.begin_dash(pt, dir, self.dash_cap);
    }
    fn extend_current_dash(&mut self, rloc: f64, at_vertex: bool) {
        let (pt, tan) = self.segs.probe_at(rloc, at_vertex);
        self.cur_is_curve |= self.segs.cur_is_curve();
        self.cur.push(pt);
        self.cur_tan.push(DashPt { tan, last: self.segs.cur_is_last(), line: !self.segs.cur_is_curve(), corner: None, skip: false, accept: true });
    }
    fn terminate_current_dash(&mut self, rloc: f64, at_vertex: bool) {
        if std::mem::take(&mut self.ignore) {
            return;
        }
        let (pt, tan) = self.segs.probe_at(rloc, at_vertex);
        self.cur_is_curve |= self.segs.cur_is_curve();
        self.cur.push(pt);
        // TerminateCurrentDash passes is_last=1 to AcceptCurvePoint.
        self.cur_tan.push(DashPt { tan, last: true, line: !self.segs.cur_is_curve(), corner: None, skip: false, accept: true });
        self.emit_dash(self.dash_cap);
        self.pen_down = false;
    }

    // Lay the dash pattern along the buffered edge, then empty it (flush).
    fn flush(&mut self, last_edge: bool) {
        if self.segs.is_empty() {
            return;
        }
        let mut is_on = self.seq.is_on_dash();
        if is_on != self.pen_down {
            // A corner fell exactly on a dash↔gap boundary: resync.
            if self.pen_down {
                self.terminate_current_dash(0.0, false);
            } else {
                self.start_a_new_dash(0.0, false, self.seq.step() * self.segs.ratio[self.segs.cur]);
            }
        }
        self.seq.update_with_new_segment(self.segs.ratio[self.segs.cur]);
        self.seq.prepare_for_new_edge();
        // DoS guard: a figure point with an astronomically huge (but finite, so not dropped by the
        // non-finite filter) coordinate makes one edge ~1e30 device units long, so the dash walk steps
        // a dash period at a time for ~forever (fuzz 1770/3765; native itself does not render these
        // quickly). Any renderable target is a few thousand px, so >2^18 dash steps on one edge means
        // sub-pixel/off-screen dashes that cannot affect on-screen pixels — stop walking.
        let mut steps = 0usize;
        loop {
            steps += 1;
            if steps > (1 << 18) {
                break;
            }
            // Between dashes on a stretch of the current segment that lies far outside the safe device
            // box, step the pattern on by whole periods: the dash positions come from the period count,
            // as Direct2D's, so the dashes after the jump are the ones its full walk lays, and no skipped
            // dash could reach the box (a 1e9-radius arc's far end returns on screen, recap 673).
            if !self.pen_down {
                let reach = SAFE_BOX + self.pen.reach() * (self.style.miter_limit.max(2.0) as f64) + 1.0;
                let d = self.seq.next_endpoint();
                if let Some(target) = self.segs.box_entry_after(d, reach) {
                    let k = ((target - d) / (self.seq.len * self.seq.scale)).floor() - 1.0;
                    if k >= 1.0 && k < (u32::MAX - self.seq.iter) as f64 {
                        self.seq.iter += k as u32;
                    }
                }
            }
            let dash_end = self.seq.next_endpoint();
            let seg_end = self.segs.current_end();
            is_on = self.seq.is_on_dash();
            if self.segs.is_last() && (dash_end - seg_end).abs() < COINCIDENCE {
                if self.pen_down {
                    self.extend_current_dash(seg_end, true);
                }
                self.do_dash_or_gap_end_at_edge_end(last_edge, is_on);
                break;
            } else if dash_end > seg_end {
                if self.pen_down {
                    self.extend_current_dash(seg_end, true);
                }
                let done = self.segs.increment();
                self.seq.advance_to(seg_end);
                if done {
                    break;
                }
                self.seq.update_with_new_segment(self.segs.ratio[self.segs.cur]);
            } else {
                if self.pen_down {
                    if is_on {
                        self.terminate_current_dash(dash_end, false);
                    }
                } else if !is_on {
                    self.start_a_new_dash(dash_end, false, self.seq.next_dash_len() * self.segs.ratio[self.segs.cur]);
                }
                self.seq.increment();
            }
        }
        self.segs.reset();
    }

    // A dash/gap end that coincides with the edge end (`DoDashOrGapEndAtEdgeEnd`).
    fn do_dash_or_gap_end_at_edge_end(&mut self, last_edge: bool, is_on: bool) {
        if last_edge {
            if !is_on {
                // End of a gap at the figure end: start a zero-length dash so the figure caps it.
                self.start_a_new_dash(self.segs.get_length(), true, 0.0);
            }
        } else {
            if is_on {
                self.terminate_current_dash(self.segs.get_length(), true);
            }
            self.seq.increment();
        }
    }
}

// One edge of the figure: its start, the points to feed as `(point, tangent, is_last)` — tangent
// `None` for a straight segment (`AcceptLinePoint`), `Some` for a flattened curve point carrying its
// analytic tangent (`AcceptCurvePoint`) — and the raw start/end directions.
struct Edge {
    start: V,
    pts: Vec<(V, Option<V>, bool)>,
    first_dir: V,
    end_dir: V,
    /// A segment too short to widen came before this edge: the corner onto it is a miter limited
    /// to 1 (the widener's skip flag, which the dasher passes to the pen).
    after_skip: bool,
}

fn v(p: Point) -> V {
    [p[0] as f64, p[1] as f64]
}

// Build the edge list from the device-space prims. A line is one edge (one line point); a cubic is
// one edge whose points are its double-precision flattened chords (the cubic widener), each
// carrying the analytic tangent so the dasher widens them smoothly, the last flagged `is_last`.
// Near-zero line edges are dropped (native skips degenerate segments), but a NaN one is kept, as
// native's unordered compare keeps it. Directions are raw (non-unit).
/// The figure's edges for the dasher. With `ends64` (the figure's f64 start and line ends, as the
/// widener maps them) each segment starts at the walk's current point, which a dropped segment leaves.
fn build_edges(prims: &[Prim], ends64: Option<&super::LineEnds64>, curve_err: f64) -> (Vec<Edge>, bool) {
    let mut edges = Vec::new();
    let fuzz = curve_err / 6.0 * SQ_LENGTH_FUZZ;
    let mut cur = ends64.map(|e| e.start);
    // Whether a segment was skipped since the last kept edge.
    let mut skipped = false;
    for (i, pr) in prims.iter().enumerate() {
        match *pr {
            Prim::Line(a, b) => {
                let b = ends64.and_then(|e| e.lines.get(i).copied().flatten()).unwrap_or(v(b));
                let a = cur.unwrap_or(v(a));
                let d = sub(b, a);
                if !(fuzz > len(d) * len(d)) {
                    edges.push(Edge { start: a, pts: vec![(b, None, false)], first_dir: d, end_dir: d, after_skip: skipped });
                    skipped = false;
                    cur = cur.map(|_| b);
                } else {
                    skipped = true;
                }
            }
            Prim::Cubic(p0, p1, p2, p3) => {
                let p0 = cur.unwrap_or(v(p0));
                let (first, flat) = stroke::flatten_curve_tan(p0, v(p1), v(p2), v(p3), curve_err);
                // The widener skips a curve with no first direction under its fuzz, as a short line.
                if flat.is_empty() || stroke::cubic_fuzz(curve_err / 6.0) >= first[0] * first[0] + first[1] * first[1] {
                    skipped = true;
                    continue;
                }
                cur = cur.map(|_| v(p3));
                let end = flat.last().unwrap().1;
                let n = flat.len();
                let pts = flat.iter().enumerate().map(|(i, &(p, t))| (p, Some(t), i == n - 1)).collect();
                edges.push(Edge { start: p0, pts, first_dir: first, end_dir: end, after_skip: skipped });
                skipped = false;
            }
        }
    }
    (edges, skipped)
}

/// Dash one figure (device-space prims, already transformed) and return the stroked fill pieces
/// (unioned under nonzero winding by the caller). `width` is the device pen width.
/// `curve_err` is the curve flattener's error bound (the widen tolerance times six).
#[allow(clippy::too_many_arguments)]
pub(super) fn widen_dashed(start: Point, prims: &[Prim], ends64: Option<&super::LineEnds64>, closed: bool, pen: &PenShape, style: &StrokeStyle, force_pts: &[V], curve_err: f64) -> Vec<Vec<Point>> {
    let (edges, trailing_skip) = build_edges(prims, ends64, curve_err);
    if edges.is_empty() && !prims.is_empty() {
        return collapsed_dashed(ends64.map_or(v(start), |e| e.start), closed, pen, style, &mut false);
    }
    dash_edges(&edges, trailing_skip, closed, pen, style, force_pts, &mut false)
}

/// A figure whose every segment the widener skipped ends unstarted at the dasher, which passes it to the
/// pen ([`stroke::collapsed_dot`]) when the pattern stands on a dash.
/// ponytail: the pattern is taken at its start; native keeps where the previous figure left it, so a
/// collapsed figure after a dashed one can differ. Carry the dasher across figures if one shows up.
fn collapsed_dashed(start: V, closed: bool, pen: &PenShape, style: &StrokeStyle, errored: &mut bool) -> Vec<Vec<Point>> {
    match DashSeq::new(style, pen.dash_width) {
        Some(seq) if seq.is_on_dash() && pen.dash_q.iter().all(|q| !q.is_nan()) => stroke::collapsed_dot(start, closed, pen, style, errored),
        _ => Vec::new(),
    }
}

/// [`widen_dashed`], setting `errored` when the pen met a dash direction it cannot normalize
/// (native's BAD_NUMBER).
#[allow(clippy::too_many_arguments)]
pub(super) fn widen_dashed_flags(start: Point, prims: &[Prim], closed: bool, pen: &PenShape, style: &StrokeStyle, tol: f32, force_pts: &[V], errored: &mut bool) -> Vec<Vec<Point>> {
    let (edges, trailing_skip) = build_edges(prims, None, tol as f64 * 6.0);
    if edges.is_empty() && !prims.is_empty() {
        return collapsed_dashed(v(start), closed, pen, style, errored);
    }
    dash_edges(&edges, trailing_skip, closed, pen, style, force_pts, errored)
}

/// The rectangle widener's dashed walk of the ordered rect, along the widener's corners and edge
/// directions ([`stroke::rect_widener_frame`]); `errored` is set where the pen fails (BAD_NUMBER).
pub(super) fn rect_dashed(rect: [f32; 4], m: Option<&super::Matrix>, pen: &PenShape, style: &StrokeStyle, tol: f32, errored: &mut bool) -> Vec<Vec<Point>> {
    let (c, dirs) = stroke::rect_widener_frame(rect, m, tol);
    let edges: Vec<Edge> = (0..4).map(|i| Edge { start: c[i], pts: vec![(c[(i + 1) % 4], None, false)], first_dir: dirs[i], end_dir: dirs[i], after_skip: false }).collect();
    dash_edges(&edges, false, true, pen, style, &[], errored)
}

/// Whether [`rect_dashed`] fails with BAD_NUMBER.
pub(super) fn rect_dashed_rejects(rect: [f32; 4], m: Option<&super::Matrix>, pen: &PenShape, style: &StrokeStyle, tol: f32) -> bool {
    let mut err = false;
    rect_dashed(rect, m, pen, style, tol, &mut err);
    err
}

/// Dash a figure's edges: the dasher's walk, start to end (closing at the start when `closed`).
/// `trailing_skip`: a segment too short to widen came after the last edge (the closing corner then
/// turns after a skip).
fn dash_edges(edges: &[Edge], trailing_skip: bool, closed: bool, pen: &PenShape, style: &StrokeStyle, force_pts: &[V], errored: &mut bool) -> Vec<Vec<Point>> {
    let seq = match DashSeq::new(style, pen.dash_width) {
        Some(s) if pen.dash_q.iter().all(|q| !q.is_nan()) => s,
        _ => return Vec::new(), // E_INVALIDARG dash array / singular pen transform → Direct2D draws nothing
    };
    if edges.is_empty() {
        return Vec::new();
    }
    let mut d = Dasher::new(seq, style, pen, force_pts);
    d.start_figure(edges[0].start, edges[0].first_dir, closed, style.start_cap);
    for &(p, t, last) in &edges[0].pts {
        d.accept(p, t, last);
    }
    for (prev, e) in edges.iter().zip(&edges[1..]) {
        d.do_corner(e.start, prev.end_dir, e.first_dir, e.after_skip);
        for &(p, t, last) in &e.pts {
            d.accept(p, t, last);
        }
    }
    let last = edges.last().unwrap();
    if closed {
        // The closing corner turns after a skip since the last edge, or one before the first.
        d.do_corner(edges[0].start, last.end_dir, edges[0].first_dir, trailing_skip || edges[0].after_skip);
        d.end_stroke_closed(edges[0].start, edges[0].first_dir);
    } else {
        let end_pt = last.pts.last().unwrap().0;
        d.end_stroke_open(end_pt, last.end_dir, style.end_cap, style.start_cap);
    }
    *errored |= d.errored;
    d.pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    // GpModF matches C fmod-with-sign for the common positive case and clamps to [0, m].
    #[test]
    fn gpmodf_basic() {
        assert!((gp_mod_f(12.5, 25.0) - 12.5).abs() < 1e-6);
        assert!((gp_mod_f(27.5, 25.0) - 2.5).abs() < 1e-5);
        assert!((gp_mod_f(-5.0, 20.0) - 15.0).abs() < 1e-5);
        assert_eq!(gp_mod_f(5.0, 0.0), 0.0); // degenerate, clamped
    }

    // A zero-length "on" dash with a round dash cap emits a disc (a non-empty outline); a flat one
    // emits nothing. This is the 07-dashes-caps dot that a plain arc-length split dropped.
    #[test]
    fn zero_dash_dot() {
        let round = StrokeStyle { dash_cap: Cap::Round, dashes: vec![2.0, 1.0, 0.0, 1.0], ..Default::default() };
        let pen = PenShape::build(6.0, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [1.0, 1.0], stroke::TransformType::Normal, 0.25).unwrap();
        let polys = widen_dashed([0.0, 0.0], &[Prim::Line([0.0, 10.0], [100.0, 10.0])], None, false, &pen, &round, &[], 1.5);
        assert!(polys.iter().any(|p| p.len() > 8), "expected a round dot outline among {} pieces", polys.len());
    }
}
