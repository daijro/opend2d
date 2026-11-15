// Ported from dotnet/wpf (MIT, Copyright (c) .NET Foundation and Contributors):
// src/Microsoft.DotNet.Wpf/src/WpfGfx/core/geometry/Boolean.{cpp,h}. See THIRD-PARTY-NOTICES.

//! Stage 3 — reconstruction. At each junction the module classifies the new heads (which side of the
//! filled region each one bounds, or whether it is redundant) and strings the chains into figures:
//! *pre-figures* are chains linked end to end, started where two heads meet, joined where tails meet,
//! and emitted when a pre-figure's ends come together. Fill chains bound the region; the chains of
//! hollow figures are *stroke* chains, kept as open figures where they are not covered by the fill.
//! A figure is emitted in grid coordinates; curve fragments on consecutive edges coalesce back into the
//! original curve, trimmed to the run.
//!
//! Direct2D's reconstructor follows WPF's `COutline` for fill chains and adds stroke chains, single
//! points and an explicit output direction for them.

use super::curveflat::{trim, Bez};
use super::scanner::Pt;
use super::sweep::{Module, ScanFailed, Scanner, BOOL_REDUNDANT, CANCELLED, COINCIDENT, DROPPED, MERGED, OUT_REVERSED, SELF_REDUNDANT, SHAPE, SIDE_RIGHT, STROKE};

/// Chain flag: a two-shape operation flipped the chain's side.
const BOOL_FLIP: u16 = 0x1000;

type R<T> = Result<T, ScanFailed>;

#[derive(Clone, Debug, PartialEq)]
pub enum OutSeg {
    Line(Pt),
    /// Two control points and the end point.
    Bezier(Pt, Pt, Pt),
}

/// An output figure in grid coordinates. A single point has no segments.
#[derive(Clone, Debug, PartialEq)]
pub struct OutFig {
    pub start: Pt,
    pub filled: bool,
    pub closed: bool,
    pub segs: Vec<OutSeg>,
}

/// Chains linked into one piece of a figure.
#[derive(Clone, Copy, Debug, Default)]
struct PreFig {
    first: Option<usize>,
    last: Option<usize>,
    /// Started from one stroke head: one end is open.
    open: bool,
}

/// The curve run being coalesced: the curve, its parameter range, its direction, and the last vertex.
#[derive(Clone, Copy)]
struct Run {
    id: u32,
    lo: f64,
    hi: f64,
    reversed: bool,
    last: usize,
}

/// `CReconstructor`: strings classified chains into figures.
pub struct Recon<'a> {
    catalog: &'a [Bez],
    figs: Vec<PreFig>,
    pub out: Vec<OutFig>,
    /// Fill chains only: stroke chains and points are not reconstructed.
    pub fill_only: bool,
    // Figure under construction.
    cur: Pt,
    downward: bool,
    seg_reversed: bool,
    run: Option<Run>,
}

impl<'a> Recon<'a> {
    pub fn new(catalog: &'a [Bez]) -> Self {
        Recon { catalog, figs: vec![], out: vec![], fill_only: false, cur: [0.0; 2], downward: true, seg_reversed: false, run: None }
    }

    // ---- pre-figures ----

    fn owner(&self, s: &Scanner, c: usize) -> R<usize> {
        s.c[c].owner.ok_or(ScanFailed)
    }

    /// `StartPreFigure`: `first` and `last` (or `first` alone, leaving an open end) begin a pre-figure.
    fn start_pre_figure(&mut self, s: &mut Scanner, first: usize, last: Option<usize>) {
        let id = self.figs.len();
        self.figs.push(PreFig { first: Some(first), last: Some(last.unwrap_or(first)), open: last.is_none() });
        s.c[first].owner = Some(id);
        if let Some(l) = last {
            s.c[l].owner = Some(id);
        }
        s.c[first].next = last;
    }

    /// `Assume`: the trailer's chains follow the leader's.
    fn assume(&mut self, s: &mut Scanner, leader: usize, trailer: usize) {
        let (lt, tf, tl) = (self.figs[leader].last.unwrap(), self.figs[trailer].first.unwrap(), self.figs[trailer].last.unwrap());
        s.c[lt].next = Some(tf);
        self.figs[leader].last = Some(tl);
        s.c[tl].owner = Some(leader);
        if tf != tl {
            s.c[tf].owner = None;
        }
        self.figs[trailer].first = None;
        self.figs[trailer].last = None;
    }

    /// `CPreFigure::Reverse`.
    fn reverse(&mut self, s: &mut Scanner, f: usize) {
        let mut prev = None;
        let mut x = self.figs[f].first;
        while let Some(c) = x {
            let n = s.c[c].next;
            s.c[c].next = prev;
            s.c[c].flags ^= OUT_REVERSED;
            prev = Some(c);
            x = n;
        }
        let fig = &mut self.figs[f];
        std::mem::swap(&mut fig.first, &mut fig.last);
    }

    /// `Open`: the second open end reached emits the pre-figure as an open figure.
    fn open(&mut self, s: &mut Scanner, f: usize) -> R<()> {
        if self.figs[f].open {
            self.add_to_shape(s, f, false)
        } else {
            self.figs[f].open = true;
            Ok(())
        }
    }

    /// `AppendFillTails`: two tails meet — close the figure if they share it, else join the figures.
    fn append_fill_tails(&mut self, s: &mut Scanner, a: usize, b: usize) -> R<()> {
        let (fa, fb) = (self.owner(s, a)?, self.owner(s, b)?);
        if fa == fb {
            self.add_to_shape(s, fa, true)
        } else {
            self.assume(s, fa, fb);
            Ok(())
        }
    }

    /// `AppendFillHeadPairs`: start a pre-figure for each (left, right) pair of heads; true if one is left.
    fn append_head_pairs(&mut self, s: &mut Scanner, mut i: usize, last: usize) -> R<bool> {
        let mut odd = true;
        while i != last {
            let n = self.pair_right(s, i)?;
            self.start_pre_figure(s, n, Some(i));
            if n == last {
                return Ok(false);
            }
            i = s.relevant(s.c[n].right, true).ok_or(ScanFailed)?;
            odd = true;
        }
        Ok(odd)
    }

    /// `AppendFillTailPairs`: join each (left, right) pair of tails; true if one is left.
    fn append_tail_pairs(&mut self, s: &mut Scanner, mut i: usize, last: usize) -> R<bool> {
        let mut odd = true;
        while i != last {
            let n = self.pair_right(s, i)?;
            self.append_fill_tails(s, i, n)?;
            if n == last {
                return Ok(false);
            }
            i = s.relevant(s.c[n].right, true).ok_or(ScanFailed)?;
            odd = true;
        }
        Ok(odd)
    }

    /// A left chain `i`'s relevant right neighbour, which must be a right chain.
    fn pair_right(&self, s: &Scanner, i: usize) -> R<usize> {
        if s.side_right(i) {
            return Err(ScanFailed);
        }
        match s.relevant(s.c[i].right, true) {
            Some(n) if s.side_right(n) => Ok(n),
            _ => Err(ScanFailed),
        }
    }

    /// `Append` (reversed): head `h` continues tail `t`'s pre-figure at its end.
    fn link_after(&mut self, s: &mut Scanner, t: usize, h: usize) -> R<()> {
        let f = self.owner(s, t)?;
        s.c[t].next = Some(h);
        s.c[h].owner = Some(f);
        self.figs[f].last = Some(h);
        Ok(())
    }

    /// `Append`: head `h` precedes tail `t` in its pre-figure.
    fn link_before(&mut self, s: &mut Scanner, h: usize, t: usize) -> R<()> {
        let f = self.owner(s, t)?;
        s.c[h].next = Some(t);
        s.c[h].owner = Some(f);
        self.figs[f].first = Some(h);
        Ok(())
    }

    /// `ProcessFillChains`.
    fn process_fill_chains(&mut self, s: &mut Scanner) -> R<()> {
        let j0 = s.j.clone();
        let rel_r = |s: &Scanner, x: Option<usize>| s.relevant(x, true);
        let rel_l = |s: &Scanner, x: Option<usize>| s.relevant(x, false);
        let (i, j, k, m) = (rel_r(s, j0.lh), rel_l(s, j0.rh), rel_r(s, j0.lt), rel_l(s, j0.rt));
        let Some(i) = i else {
            // Tails only.
            let Some(k) = k else { return Ok(()) };
            let m = m.ok_or(ScanFailed)?;
            if s.side_right(k) {
                // The region continues on the left: the inner tails pair up, the outer two join.
                let n = rel_r(s, s.c[k].right);
                if n.is_some() && n != Some(m) {
                    let l2 = rel_l(s, s.c[m].left).ok_or(ScanFailed)?;
                    if self.append_tail_pairs(s, n.unwrap(), l2)? {
                        return Err(ScanFailed);
                    }
                }
                return self.append_fill_tails(s, m, k);
            }
            return if self.append_tail_pairs(s, k, m)? { Err(ScanFailed) } else { Ok(()) };
        };
        let j = j.ok_or(ScanFailed)?;
        if !s.side_right(i) {
            // Void on the left: heads and tails pair up; an odd head continues the odd tail.
            let odd_h = self.append_head_pairs(s, i, j)?;
            let odd_t = match k {
                Some(k) => self.append_tail_pairs(s, k, m.ok_or(ScanFailed)?)?,
                None => false,
            };
            if odd_h != odd_t {
                return Err(ScanFailed);
            }
            if odd_h {
                self.link_after(s, m.unwrap(), j)?;
            }
            return Ok(());
        }
        let Some(k) = k else {
            // Filled on the left, heads only: the outer two start a pre-figure, the inner ones pair up.
            if i == j {
                return Err(ScanFailed);
            }
            self.start_pre_figure(s, i, Some(j));
            let n = rel_r(s, s.c[i].right);
            if n.is_none() || n == Some(j) {
                return Ok(());
            }
            let l2 = rel_l(s, s.c[j].left).ok_or(ScanFailed)?;
            return if self.append_head_pairs(s, n.unwrap(), l2)? { Err(ScanFailed) } else { Ok(()) };
        };
        // Filled on the left with tails: the leftmost head continues the leftmost tail's pre-figure.
        let m = m.ok_or(ScanFailed)?;
        self.link_before(s, i, k)?;
        let (h, hj) = if i == j { (None, None) } else { (Some(rel_r(s, s.c[i].right).ok_or(ScanFailed)?), Some(j)) };
        let (t, tm) = if k == m { (None, None) } else { (Some(rel_r(s, s.c[k].right).ok_or(ScanFailed)?), Some(m)) };
        if h.is_none() && t.is_none() {
            return Ok(());
        }
        let odd_h = match h {
            Some(h) => self.append_head_pairs(s, h, hj.unwrap())?,
            None => false,
        };
        let odd_t = match t {
            Some(t) => self.append_tail_pairs(s, t, tm.unwrap())?,
            None => false,
        };
        if odd_h != odd_t {
            return Err(ScanFailed);
        }
        if odd_h {
            self.link_after(s, m, j)?;
        }
        Ok(())
    }

    /// `AppendStrokeTails`: two stroke tails meet.
    fn append_stroke_tails(&mut self, s: &mut Scanner, a: usize, b: usize) -> R<()> {
        let (mut fa, mut fb) = (self.owner(s, a)?, self.owner(s, b)?);
        if fa == fb {
            return self.add_to_shape(s, fa, false);
        }
        match (s.reverse_on_output(a), s.reverse_on_output(b)) {
            (true, true) => self.reverse(s, fa),
            (true, false) => std::mem::swap(&mut fa, &mut fb),
            (false, false) => self.reverse(s, fb),
            (false, true) => {}
        }
        self.assume(s, fa, fb);
        if self.figs[fb].open {
            self.open(s, fa)?;
        }
        Ok(())
    }

    /// `ProcessAJunction`.
    fn process_a_junction(&mut self, s: &mut Scanner) -> R<()> {
        self.process_fill_chains(s)?;
        if self.fill_only {
            return Ok(());
        }
        let live_stroke = |f: u16| f & STROKE != 0 && f & DROPPED == 0;
        let next_stroke = |s: &Scanner, x: Option<usize>| s.go_while(x, true, |f| !live_stroke(f));
        // Stroke tails pair up left to right; so do stroke heads, each pair starting a pre-figure.
        let mut i = next_stroke(s, s.j.lt);
        while let Some(a) = i {
            let Some(b) = next_stroke(s, s.c[a].right) else { break };
            self.append_stroke_tails(s, a, b)?;
            i = next_stroke(s, s.c[b].right);
        }
        let mut j = next_stroke(s, s.j.lh);
        while let Some(a) = j {
            let Some(b) = next_stroke(s, s.c[a].right) else { break };
            s.c[b].flags ^= OUT_REVERSED;
            self.start_pre_figure(s, b, Some(a));
            j = next_stroke(s, s.c[b].right);
        }
        // An odd tail runs on into an odd head; an odd tail alone ends its pre-figure; an odd head
        // alone starts one.
        match (i, j) {
            (Some(t), Some(h)) => {
                let f = self.owner(s, t)?;
                s.c[h].owner = Some(f);
                if s.reverse_on_output(t) {
                    s.c[h].flags ^= OUT_REVERSED;
                    s.c[h].next = Some(t);
                    self.figs[f].first = Some(h);
                } else {
                    s.c[t].next = Some(h);
                    self.figs[f].last = Some(h);
                }
            }
            (Some(t), None) => {
                let f = self.owner(s, t)?;
                self.open(s, f)?;
            }
            (None, Some(h)) => self.start_pre_figure(s, h, None),
            (None, None) => {}
        }
        if !s.j.no_isolated {
            let p = s.v[s.j.rep].pt;
            self.out.push(OutFig { start: p, filled: false, closed: false, segs: vec![] });
        }
        Ok(())
    }

    // ---- figure output ----

    /// `CReconstructor::AddToShape`: emit the pre-figure's chains as one figure.
    fn add_to_shape(&mut self, s: &mut Scanner, f: usize, filled: bool) -> R<()> {
        let first = self.figs[f].first.ok_or(ScanFailed)?;
        let up = |s: &Scanner, c: usize| if s.is_point(c) || s.c[c].flags & STROKE != 0 { s.c[c].flags & OUT_REVERSED != 0 } else { s.side_right(c) };
        let start = if up(s, first) { s.c[first].tail } else { s.c[first].head };
        let start_pt = s.v[start].pt;
        self.out.push(OutFig { start: start_pt, filled, closed: false, segs: vec![] });
        self.cur = start_pt;
        let mut x = Some(first);
        while let Some(c) = x {
            let reversed = s.c[c].flags & super::scanner::REVERSED != 0;
            if up(s, c) {
                self.downward = false;
                self.seg_reversed = !reversed;
                let mut v = s.v[s.c[c].tail].prev;
                while let Some(p) = v {
                    self.add_vertex(s, p);
                    v = s.v[p].prev;
                }
            } else {
                self.downward = true;
                self.seg_reversed = reversed;
                let mut v = s.v[s.c[c].head].next;
                while let Some(p) = v {
                    self.add_vertex(s, p);
                    v = s.v[p].next;
                }
            }
            s.c[c].owner = None;
            x = s.c[c].next;
        }
        self.flush_curve(s);
        let fig = self.out.last_mut().unwrap();
        fig.closed = start_pt == self.cur;
        self.figs[f] = PreFig::default();
        Ok(())
    }

    /// `AddVertexWithCurves`.
    fn add_vertex(&mut self, s: &Scanner, v: usize) {
        let e = if self.downward { v } else { s.v[v].next.unwrap() };
        let Some((id, t0, t1)) = s.v[e].curve else {
            self.flush_curve(s);
            self.cur = s.v[v].pt;
            self.out.last_mut().unwrap().segs.push(OutSeg::Line(self.cur));
            return;
        };
        let extended = match &mut self.run {
            Some(r) if r.id == id => {
                if self.seg_reversed && r.lo == t1 {
                    r.lo = t0;
                    true
                } else if !self.seg_reversed && r.hi == t0 {
                    r.hi = t1;
                    true
                } else {
                    false
                }
            }
            _ => false,
        };
        if !extended {
            self.flush_curve(s);
            self.run = Some(Run { id, lo: t0, hi: t1, reversed: self.seg_reversed, last: v });
        }
        let r = self.run.as_mut().unwrap();
        r.last = v;
        self.cur = s.v[v].pt;
    }

    /// `FlushCurve`: the run's curve trimmed to its range, ending at the run's last vertex.
    fn flush_curve(&mut self, s: &Scanner) {
        let Some(r) = self.run.take() else { return };
        if r.hi - r.lo < 0.000001 {
            return;
        }
        let b = trim(self.catalog[r.id as usize], r.lo, r.hi);
        let (mut c1, mut c2) = ([b[2], b[3]], [b[4], b[5]]);
        if r.reversed {
            std::mem::swap(&mut c1, &mut c2);
        }
        self.out.last_mut().unwrap().segs.push(OutSeg::Bezier(c1, c2, s.v[r.last].pt));
    }
}

/// The `COutline` classifier in front of the reconstructor.
pub struct Outline<'a> {
    pub r: Recon<'a>,
}

impl<'a> Outline<'a> {
    pub fn new(catalog: &'a [Bez]) -> Self {
        Outline { r: Recon::new(catalog) }
    }
}

impl Module for Outline<'_> {
    /// `COutline::ProcessTheJunction`: classify the heads (hollow chains are stroke chains), then
    /// reconstruct.
    fn junction(&mut self, s: &mut Scanner) -> R<()> {
        let mut tail = s.go_while(s.j.lt, true, |f| f & 0x40 != 0);
        let mut left = s.go_while(s.j.left, false, |f| f & 0x40 != 0);
        let mut inside = left.is_some_and(|l| !s.side_right(l));
        let first_inside = inside;
        // Classify one fill head against the tail it continues or the chain on its left.
        let fill = |s: &mut Scanner, x: usize, tail: &mut Option<usize>, left: &mut Option<usize>| {
            match tail.take() {
                Some(t) => s.continue_from(x, t),
                None => s.classify(x, *left),
            }
            *left = Some(x);
            s.c[x].flags & SELF_REDUNDANT == 0
        };
        let mut h = s.j.lh;
        while let Some(x0) = h {
            let mut x = x0;
            if s.c[x].flags & COINCIDENT != 0 {
                // A bundle of chains along one edge: at most one fill and one stroke chain survive.
                let (mut pfill, mut pstroke) = (None, None);
                if s.c[x].flags & STROKE != 0 {
                    if inside {
                        s.c[x].flags |= SELF_REDUNDANT;
                    } else {
                        pstroke = Some(x);
                    }
                } else if fill(s, x, &mut tail, &mut left) {
                    pfill = Some(x);
                }
                loop {
                    x = s.c[x].right.unwrap();
                    if s.c[x].flags & STROKE != 0 {
                        if inside {
                            s.c[x].flags |= SELF_REDUNDANT;
                        } else if pstroke.is_some() {
                            s.c[x].flags |= BOOL_REDUNDANT;
                        } else {
                            pstroke = Some(x);
                        }
                    } else if fill(s, x, &mut tail, &mut left) {
                        if let Some(p) = pfill.take() {
                            if pstroke.is_some() || inside {
                                s.c[x].flags |= CANCELLED;
                                s.c[p].flags |= CANCELLED;
                            } else {
                                // Two fill chains cancel; their edge survives as a stroke.
                                pstroke = Some(p);
                                s.c[x].flags |= CANCELLED;
                                s.c[p].flags |= MERGED;
                            }
                        } else {
                            pfill = Some(x);
                        }
                    }
                    if s.c[x].flags & COINCIDENT == 0 {
                        break;
                    }
                }
                if let Some(p) = pstroke {
                    if pfill.is_some() || inside {
                        s.c[p].flags |= CANCELLED;
                    }
                }
                if pfill.is_some() {
                    inside = !inside;
                }
            } else if s.c[x].flags & STROKE != 0 {
                if inside {
                    s.c[x].flags |= BOOL_REDUNDANT;
                }
            } else if fill(s, x, &mut tail, &mut left) {
                inside = !inside;
            }
            h = s.c[x].right;
        }
        s.j.no_isolated = first_inside || s.j.points[0] == 0 || s.j.lh.is_some() || s.j.lt.is_some();
        self.r.process_a_junction(s)
    }
}


/// `C2ShapesProcessor`'s state at a junction: per shape, the tail a head continues, the chain on the
/// left, and whether the point left of the current head is inside the shape.
#[derive(Default)]
struct Shapes {
    tail: [Option<usize>; 2],
    left: [Option<usize>; 2],
    inside: [bool; 2],
}

/// The classifier hooks of a two-shape operation (`C2ShapesProcessor`'s virtual methods).
trait TwoShapes {
    fn shapes(&mut self) -> &mut Shapes;
    /// A live fill chain alone on its edge.
    fn single_fill(&mut self, s: &mut Scanner, x: usize);
    /// A stroke chain alone on its edge, outside its own shape.
    fn single_stroke(&mut self, s: &mut Scanner, x: usize);
    /// A fill chain of each shape along one edge.
    fn fill_pair(&mut self, s: &mut Scanner, a: usize, b: usize);
    /// A fill chain and a stroke chain of the other shape along one edge.
    fn mixed_pair(&mut self, s: &mut Scanner, fill: usize, stroke: usize);
    /// A stroke chain of each shape along one edge.
    fn stroke_pair(&mut self, s: &mut Scanner, a: usize, b: usize);
    /// The junction's single points (`ClassifyTheJunction`), when no live chain covered them.
    fn classify_points(&mut self, s: &mut Scanner);
}

/// Classify a fill head in its own shape: continue that shape's tail, else follow its left chain.
fn own(sh: &mut Shapes, s: &mut Scanner, x: usize, k: usize) {
    match sh.tail[k].take() {
        Some(t) => s.continue_from(x, t),
        None => s.classify(x, sh.left[k]),
    }
    sh.left[k] = Some(x);
}

/// `C2ShapesProcessor::Classify`.
fn classify_two(t: &mut impl TwoShapes, s: &mut Scanner) {
    let solid = |k: u16| move |f: u16| (f & SHAPE) != k || f & 0x40 != 0;
    let sh = t.shapes();
    sh.tail = [s.go_while(s.j.lt, true, solid(0)), s.go_while(s.j.lt, true, solid(1))];
    sh.left = [s.go_while(s.j.left, false, solid(0)), s.go_while(s.j.left, false, solid(1))];
    sh.inside = sh.left.map(|l| l.is_some_and(|l| s.c[l].flags & SIDE_RIGHT == 0));
    let mut h = s.j.lh;
    while let Some(x) = h {
        let f = s.c[x].flags;
        let k = (f & SHAPE) as usize;
        if f & COINCIDENT != 0 {
            h = bundle(t, s, x);
            continue;
        }
        if f & STROKE != 0 {
            if t.shapes().inside[k] {
                s.c[x].flags |= SELF_REDUNDANT;
            } else {
                t.single_stroke(s, x);
            }
        } else {
            own(t.shapes(), s, x, k);
            if s.c[x].flags & SELF_REDUNDANT == 0 {
                t.single_fill(s, x);
                let sh = t.shapes();
                sh.inside[k] = !sh.inside[k];
            }
        }
        h = s.c[x].right;
    }
    if !s.j.no_isolated {
        t.classify_points(s);
    }
}

/// `ClassifyCoincidentBundle`: the chains along one edge, both shapes, resolve to at most one fill and
/// one stroke chain per shape before the pair classifiers see them; returns the chain after them.
fn bundle(t: &mut impl TwoShapes, s: &mut Scanner, mut x: usize) -> Option<usize> {
    let (mut fills, mut strokes): ([Option<usize>; 2], [Option<usize>; 2]) = ([None; 2], [None; 2]);
    loop {
        let k = (s.c[x].flags & SHAPE) as usize;
        let inside = t.shapes().inside[k];
        if s.c[x].flags & STROKE != 0 {
            if inside {
                s.c[x].flags |= SELF_REDUNDANT;
            } else if strokes[k].is_some() {
                s.c[x].flags |= BOOL_REDUNDANT;
            } else {
                strokes[k] = Some(x);
            }
        } else {
            own(t.shapes(), s, x, k);
            if s.c[x].flags & SELF_REDUNDANT == 0 {
                if let Some(p) = fills[k].take() {
                    s.c[x].flags |= CANCELLED;
                    if strokes[k].is_some() || inside {
                        s.c[p].flags |= CANCELLED;
                    } else {
                        s.c[p].flags |= MERGED;
                        strokes[k] = Some(p);
                    }
                } else {
                    fills[k] = Some(x);
                }
            }
        }
        if s.c[x].flags & COINCIDENT == 0 {
            break;
        }
        x = s.c[x].right.unwrap();
    }
    for k in 0..2 {
        if let (Some(_), Some(st)) = (fills[k], strokes[k]) {
            s.c[st].flags |= CANCELLED;
            strokes[k] = None;
        }
    }
    match (fills, strokes) {
        ([Some(a), Some(b)], _) => {
            t.fill_pair(s, a, b);
            let sh = t.shapes();
            sh.inside = [!sh.inside[0], !sh.inside[1]];
        }
        ([Some(a), None], [_, Some(b)]) => {
            t.mixed_pair(s, a, b);
            t.shapes().inside[0] ^= true;
        }
        ([Some(a), None], _) => {
            t.single_fill(s, a);
            t.shapes().inside[0] ^= true;
        }
        ([None, b], [Some(a), sb]) => match (b, sb) {
            (None, Some(sb)) => t.stroke_pair(s, a, sb),
            (None, None) => t.single_stroke(s, a),
            (Some(b), _) => {
                t.mixed_pair(s, b, a);
                t.shapes().inside[1] ^= true;
            }
        },
        ([None, Some(b)], _) => {
            t.single_fill(s, b);
            t.shapes().inside[1] ^= true;
        }
        ([None, None], [None, Some(b)]) => t.single_stroke(s, b),
        _ => {}
    }
    s.c[x].right
}

/// `CBoolean`: classifies the heads of two shapes for a combine mode (0 union, 1 intersect, 2 xor,
/// 3 exclude), then reconstructs.
pub struct Boolean<'a> {
    pub r: Recon<'a>,
    mode: u32,
    sh: Shapes,
    /// Per shape, whether a non-redundant chain of it was met at the junction.
    seen: [bool; 2],
}

impl<'a> Boolean<'a> {
    pub fn new(catalog: &'a [Bez], mode: u32) -> Self {
        Boolean { r: Recon::new(catalog), mode, sh: Shapes::default(), seen: [false; 2] }
    }

    fn live(s: &mut Scanner, x: usize) {
        if s.c[x].flags & DROPPED == 0 {
            s.j.no_isolated = true;
        }
    }
}

impl TwoShapes for Boolean<'_> {
    fn shapes(&mut self) -> &mut Shapes {
        &mut self.sh
    }

    /// `CBoolean::ClassifySingleFillChain`: union keeps what is outside the other shape, intersect
    /// what is inside it; xor flips what is inside it; exclude keeps the first shape outside the second
    /// and the second inside the first, flipped.
    fn single_fill(&mut self, s: &mut Scanner, x: usize) {
        let k = (s.c[x].flags & SHAPE) as usize;
        let other = self.sh.inside[1 - k];
        self.seen[k] = true;
        let stroke = s.c[x].flags & STROKE != 0;
        let (redundant, flip) = match self.mode {
            0 => (other, false),
            1 => (!other, false),
            2 => (other && stroke, other && !stroke),
            3 if k == 1 => (!self.sh.inside[0], self.sh.inside[0]),
            3 => (self.sh.inside[1], false),
            _ => (false, false),
        };
        if redundant {
            s.c[x].flags |= BOOL_REDUNDANT;
        }
        if flip {
            s.c[x].flags ^= BOOL_FLIP;
        }
        Self::live(s, x);
    }

    /// `CBoolean::ClassifySingleStrokeChain`.
    fn single_stroke(&mut self, s: &mut Scanner, x: usize) {
        let k = (s.c[x].flags & SHAPE) as usize;
        let other = self.sh.inside[1 - k];
        self.seen[k] = true;
        let redundant = match self.mode {
            0 | 2 => other,
            1 => !other,
            3 => k == 1 || self.sh.inside[1],
            _ => false,
        };
        if redundant {
            s.c[x].flags |= BOOL_REDUNDANT;
        }
        Self::live(s, x);
    }

    /// `CBoolean::ClassifyCoincidentFillPair`.
    fn fill_pair(&mut self, s: &mut Scanner, a: usize, b: usize) {
        let differ = self.sh.inside[0] != self.sh.inside[1];
        let cancel = match self.mode {
            0 => differ,
            1 => {
                if differ {
                    // The shapes touch along the edge without overlapping: it survives as a stroke.
                    s.c[a].flags |= 0x4000;
                }
                false
            }
            2 => true,
            3 => !differ,
            _ => {
                s.j.no_isolated = true;
                return;
            }
        };
        if cancel {
            s.c[a].flags |= CANCELLED;
            s.c[b].flags |= CANCELLED;
        } else {
            s.c[b].flags |= BOOL_REDUNDANT;
        }
        s.j.no_isolated = true;
    }

    /// `CBoolean::ClassifyCoincidentMixedPair`.
    fn mixed_pair(&mut self, s: &mut Scanner, fill: usize, stroke: usize) {
        match self.mode {
            1 => s.c[fill].flags |= BOOL_REDUNDANT,
            0 | 2 => s.c[stroke].flags |= BOOL_REDUNDANT,
            3 => {
                s.c[stroke].flags |= BOOL_REDUNDANT;
                if s.c[fill].flags & SHAPE != 0 {
                    s.c[fill].flags |= BOOL_REDUNDANT;
                }
            }
            _ => {}
        }
        s.j.no_isolated = true;
    }

    /// `CBoolean::ClassifyCoincidentStrokePair`.
    fn stroke_pair(&mut self, s: &mut Scanner, a: usize, b: usize) {
        if self.mode <= 1 {
            s.c[b].flags |= BOOL_REDUNDANT;
        } else {
            s.c[a].flags |= CANCELLED;
            s.c[b].flags |= CANCELLED;
        }
        s.j.no_isolated = true;
    }

    /// `CBoolean::ClassifyTheJunction`: whether the junction's single points survive the operation.
    fn classify_points(&mut self, s: &mut Scanner) {
        let points = s.j.points;
        if self.mode != 1 && points == [0, 0] {
            s.j.no_isolated = true;
            return;
        }
        let mut t = s.j.lt;
        while let Some(c) = t {
            let f = s.c[c].flags;
            if f & SELF_REDUNDANT == 0 {
                self.seen[(f & SHAPE) as usize] = true;
            }
            if f & DROPPED == 0 || self.mode == 0 {
                s.j.no_isolated = true;
                return;
            }
            t = s.c[c].right;
        }
        let (in0, in1) = (!self.seen[0] && self.sh.inside[0], !self.seen[1] && self.sh.inside[1]);
        let a = points[0] != 0 || self.seen[0] || in0;
        let b = points[1] != 0 || self.seen[1] || in1;
        let drop = match self.mode {
            0 => (!a && !b) || in0 || in1,
            1 => !a || !b || (in0 && in1),
            2 => a == b || in0 != in1,
            3 => !a || b || (in0 && !in1),
            _ => false,
        };
        if drop {
            s.j.no_isolated = true;
        }
    }
}

impl Module for Boolean<'_> {
    /// `CBoolean::ProcessTheJunction`: classify, then reconstruct.
    fn junction(&mut self, s: &mut Scanner) -> R<()> {
        self.seen = [false; 2];
        classify_two(self, s);
        self.r.process_a_junction(s)
    }
}

/// `CRelation`: how two shapes meet, as the dimensions (−1 none, 0 point, 1 curve, 2 region) of the
/// intersections of their interiors, boundaries and exteriors (`m[a][b]`, rows shape 0, columns
/// shape 1, in that order), gathered junction by junction.
pub struct Relation {
    sh: Shapes,
    /// Each shape's dimension.
    pub dim: [i32; 2],
    pub m: [[i32; 3]; 3],
    /// Per shape at the junction: a fill chain met; stroke chains met; a fill or stroke chain met
    /// inside or outside the other shape.
    fill: [bool; 2],
    strokes: [u32; 2],
    fill_in: [bool; 2],
    fill_out: [bool; 2],
    stroke_in: [bool; 2],
    stroke_out: [bool; 2],
}

impl Default for Relation {
    fn default() -> Self {
        let mut m = [[-1; 3]; 3];
        m[2][2] = 2;
        Relation { sh: Shapes::default(), dim: [-1; 2], m, fill: [false; 2], strokes: [0; 2], fill_in: [false; 2], fill_out: [false; 2], stroke_in: [false; 2], stroke_out: [false; 2] }
    }
}

impl Relation {
    /// `CRelationResults::UpdateIntersectionDimension`.
    fn raise(&mut self, i: usize, j: usize, d: i32) {
        if d > self.m[i][j] {
            self.m[i][j] = d;
        }
    }

    fn live(s: &mut Scanner, x: usize) {
        if s.c[x].flags & DROPPED == 0 {
            s.j.points[(s.c[x].flags & SHAPE) as usize] = 0;
        }
    }

    /// `CRelation::EndGeometry`: a shape of higher dimension also meets the other's exterior.
    pub fn finish(&mut self) {
        let [a, b] = self.dim;
        if a > b {
            self.raise(0, 2, a);
        } else if b > a {
            self.raise(2, 0, b);
        }
    }

    /// `GetRelation`: 1 disjoint, 2 is_contained, 3 contains, 4 overlap.
    pub fn relation(&self) -> u32 {
        let m = &self.m;
        if m[0][0] < 0 {
            if m[0][1] >= 0 || m[1][0] >= 0 || m[1][1] >= 0 { 4 } else { 1 }
        } else if m[0][2] < 0 {
            2
        } else if m[2][0] >= 0 {
            4
        } else {
            3
        }
    }
}

impl TwoShapes for Relation {
    fn shapes(&mut self) -> &mut Shapes {
        &mut self.sh
    }

    fn single_fill(&mut self, s: &mut Scanner, x: usize) {
        let k = (s.c[x].flags & SHAPE) as usize;
        self.fill[k] = true;
        if self.sh.inside[1 - k] {
            self.fill_in[k] = true;
        } else {
            self.fill_out[k] = true;
        }
        Self::live(s, x);
    }

    fn single_stroke(&mut self, s: &mut Scanner, x: usize) {
        let k = (s.c[x].flags & SHAPE) as usize;
        self.strokes[k] += 1;
        if self.sh.inside[1 - k] {
            self.stroke_in[k] = true;
        } else {
            self.stroke_out[k] = true;
        }
        Self::live(s, x);
    }

    fn fill_pair(&mut self, s: &mut Scanner, a: usize, b: usize) {
        self.m[1][1] = 1;
        if s.side_right(a) != s.side_right(b) {
            self.m[0][2] = 2;
            self.m[2][0] = 2;
        } else {
            self.m[0][0] = 2;
        }
        s.j.points = [0, 0];
    }

    fn mixed_pair(&mut self, s: &mut Scanner, fill: usize, stroke: usize) {
        self.fill[(s.c[fill].flags & SHAPE) as usize] = true;
        let k = (s.c[stroke].flags & SHAPE) as usize;
        if k == 1 {
            self.raise(1, 0, 1);
        } else {
            self.raise(0, 1, 1);
        }
        self.strokes[k] += 1;
        s.j.points = [0, 0];
    }

    fn stroke_pair(&mut self, s: &mut Scanner, _a: usize, _b: usize) {
        self.raise(0, 0, 1);
        self.strokes[0] += 1;
        self.strokes[1] += 1;
        s.j.points = [0, 0];
    }

    /// `CRelation::ClassifyTheJunction`: the live tails count as the shapes' chains at the point.
    fn classify_points(&mut self, s: &mut Scanner) {
        let mut t = s.go_while(s.j.lt, true, |f| f & DROPPED != 0);
        while let Some(c) = t {
            let k = (s.c[c].flags & SHAPE) as usize;
            if s.c[c].flags & STROKE != 0 {
                self.strokes[k] += 1;
            } else {
                self.fill[k] = true;
            }
            t = s.go_while(s.c[c].right, true, |f| f & DROPPED != 0);
        }
    }
}

impl Module for Relation {
    /// `CRelation::ProcessTheJunction`: classify, then record where the junction point lies in each
    /// shape (interior, boundary, exterior, at a dimension) and what the chains through it show.
    fn junction(&mut self, s: &mut Scanner) -> R<()> {
        self.fill = [false; 2];
        self.strokes = [0; 2];
        (self.fill_in, self.fill_out, self.stroke_in, self.stroke_out) = ([false; 2], [false; 2], [false; 2], [false; 2]);
        classify_two(self, s);
        let (mut pt, mut ins, mut outs) = ([[-1, -1, 2]; 2], [false; 2], [false; 2]);
        for k in 0..2 {
            let p = &mut pt[k];
            let (mut on, mut interior) = (-1, -1);
            if self.fill[k] {
                self.dim[k] = self.dim[k].max(2);
                *p = [-1, 1, -1];
                on = 1;
            } else if self.sh.inside[k] {
                *p = [2, -1, -1];
                interior = 2;
                ins[k] = true;
            } else {
                outs[k] = true;
            }
            let n = self.strokes[k];
            if n % 2 == 1 {
                self.dim[k] = self.dim[k].max(1);
                if on < 0 {
                    p[1] = 0;
                }
                p[2] = -1;
            } else if n > 0 {
                self.dim[k] = self.dim[k].max(1);
                if interior < 1 {
                    p[0] = 1;
                    interior = 1;
                }
                p[2] = -1;
            }
            if s.j.points[k] != 0 && !s.j.no_isolated {
                self.dim[k] = self.dim[k].max(0);
                if interior < 0 {
                    p[0] = 0;
                }
                p[2] = -1;
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                let (a, b) = (pt[0][i], pt[1][j]);
                self.raise(i, j, if a == 1 && b == 1 { 0 } else { a.min(b) });
            }
        }
        if ins[0] {
            if ins[1] {
                self.raise(0, 0, 2);
            } else if outs[1] {
                self.raise(0, 2, 2);
            }
        } else if outs[0] && ins[1] {
            self.raise(2, 0, 2);
        }
        if self.fill_in[0] {
            self.raise(0, 0, 2);
            self.raise(1, 0, 1);
            self.raise(2, 0, 2);
        }
        if self.fill_in[1] {
            self.raise(0, 0, 2);
            self.raise(0, 1, 1);
            self.raise(0, 2, 2);
        }
        if self.fill_out[0] {
            self.raise(0, 2, 2);
            self.raise(1, 2, 1);
        }
        if self.fill_out[1] {
            self.raise(2, 0, 2);
            self.raise(2, 1, 1);
        }
        if self.stroke_in[0] || self.stroke_in[1] {
            self.raise(0, 0, 1);
        }
        if self.stroke_out[0] {
            self.raise(0, 2, 1);
        }
        if self.stroke_out[1] {
            self.raise(2, 0, 1);
        }
        Ok(())
    }
}

/// The scan of the AA tessellator, a geometry realization's tessellator: heads classified by the fill
/// rule alone (coincident pairs cancel, so no edge survives as a stroke) and fill chains only
/// reconstructed, each pre-figure becoming one ring.
pub struct Realize<'a> {
    pub r: Recon<'a>,
}

impl<'a> Realize<'a> {
    pub fn new(catalog: &'a [Bez]) -> Self {
        let mut r = Recon::new(catalog);
        r.fill_only = true;
        Realize { r }
    }
}

impl Module for Realize<'_> {
    fn junction(&mut self, s: &mut Scanner) -> R<()> {
        s.classify_fill_only();
        self.r.process_a_junction(s)
    }
}

/// `CArea`: twice the filled area in grid units, as the sum of the boundary edges' cross products
/// (a right side's edges count negatively), accumulated in the order the sweep passes them.
#[derive(Default)]
pub struct Area {
    pub sum: f64,
}

impl Area {
    /// `UpdateWithEdge`: the edge from `v` to the next vertex of chain `c`.
    fn edge(&mut self, s: &Scanner, c: usize, v: usize) {
        let (p, q) = (s.v[v].pt, s.v[s.v[v].next.unwrap()].pt);
        let cross = q[1] * p[0] - q[0] * p[1];
        if s.side_right(c) {
            self.sum -= cross;
        } else {
            self.sum += cross;
        }
    }
}

impl Module for Area {
    /// The junction's live tails contribute their last edges.
    fn junction(&mut self, s: &mut Scanner) -> R<()> {
        s.classify_fill_only();
        for c in s.span(s.j.lt, s.j.rt) {
            if s.c[c].flags & DROPPED == 0 {
                let mut v = s.c[c].cursor;
                if v == s.c[c].tail {
                    v = s.v[v].prev.unwrap();
                }
                self.edge(s, c, v);
            }
        }
        Ok(())
    }

    /// A chain's cursor moved: the edge it just passed contributes.
    fn current_vertex(&mut self, s: &mut Scanner, c: usize) -> R<()> {
        let v = s.v[s.c[c].cursor].prev.unwrap();
        self.edge(s, c, v);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::scanner::ChainList;
    use super::*;

    fn outline(figs: &[(&[Pt], bool)], winding: bool) -> Vec<OutFig> {
        let mut l = ChainList::default();
        for (f, filled) in figs {
            l.begin_figure(f[0], *filled);
            for &p in &f[1..] {
                l.add(p);
            }
            l.end_figure(true);
        }
        let mut s = Scanner::new(l.chains(), [winding; 2]);
        let mut o = Outline::new(&[]);
        s.scan(&mut o).unwrap();
        o.r.out
    }

    fn lines(start: Pt, pts: &[Pt]) -> OutFig {
        OutFig { start, filled: true, closed: true, segs: pts.iter().map(|&p| OutSeg::Line(p)).collect() }
    }

    // A square comes back as one closed figure. It is emitted where its two chains end, from the
    // pre-figure's first chain (the right side, traversed up from its tail), then down the left one.
    #[test]
    fn square_outline() {
        let o = outline(&[(&[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], true)], false);
        assert_eq!(o, vec![lines([0.0, 0.0], &[[10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]])]);
    }

    // Two overlapping squares (one figure each, nonzero) merge into one outline with no inner edges.
    #[test]
    fn overlapping_squares_merge() {
        let a: &[Pt] = &[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let b: &[Pt] = &[[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]];
        let o = outline(&[(a, true), (b, true)], true);
        assert_eq!(o.len(), 1);
        let f = &o[0];
        assert!(f.closed && f.filled);
        assert_eq!(f.segs.len(), 8);
    }

    // A bowtie under alternate fill becomes two triangles meeting at the crossing.
    #[test]
    fn bowtie_outline_is_two_triangles() {
        let o = outline(&[(&[[0.0, 0.0], [10.0, 10.0], [10.0, 0.0], [0.0, 10.0]], true)], false);
        assert_eq!(o.len(), 2);
        assert!(o.iter().all(|f| f.closed && f.segs.len() == 3));
    }
}
