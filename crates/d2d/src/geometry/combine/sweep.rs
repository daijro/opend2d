// Ported from dotnet/wpf (MIT, Copyright (c) .NET Foundation and Contributors):
// src/Microsoft.DotNet.Wpf/src/WpfGfx/core/geometry/scanner.{cpp,h} and
// src/Microsoft.DotNet.Wpf/src/WpfGfx/common/shared/heap.h. See THIRD-PARTY-NOTICES.

//! Stage 2 — the sweep. The monotone chains of stage 1 are swept from the largest (y, x) down. Active
//! chains are kept left to right; the next vertex processed is the higher of the top inactive chain's
//! head and the top *candidate* (the active chain whose current edge ends highest). Chains are split
//! where they cross, where a vertex lies on another chain, and where two chains run along each other,
//! so that every chain meets others only at its ends. Wherever chains meet the scanner forms a
//! *junction* — the tails ending and the heads starting at one point — and hands it to the module
//! (Outline, Combine, ...), which classifies the heads and builds its output.
//!
//! Direct2D's build of the sweep follows WPF's, and also: counts single-point chains at a junction
//! instead of activating them; re-sorts a split chain in place in the candidate heap rather than
//! removing and re-inserting it, and skips the neighbour tests (and the re-insertion) for an edge
//! whose x range meets neither neighbour's; moves every chain end at a junction onto the junction's
//! representative point; links the vertices of the junction's relevant tails and heads into a ring;
//! and leaves classification to the module.

use super::intersect::{approx_point, param_cmp, side_of_intersection, side_of_point, yx_sort_intersection_and_point, yx_sort_intersection_pair, Kind, Loc, Seg};
use super::scanner::{Chain, Cv, Pt, REVERSED as REVERSED_FLAG};
use std::cmp::Ordering;

/// Chain flag: shares its first edge with the active chain on its right.
pub const COINCIDENT: u16 = 0x20;
/// Chain flag: classified as a right side.
pub const SIDE_RIGHT: u16 = 0x100;
/// Chain flag: redundant in its own shape.
pub const SELF_REDUNDANT: u16 = 0x200;
/// Chain flag: cancelled against a coincident chain.
pub const CANCELLED: u16 = 0x400;
/// Chain flag: the operand index in a two-shape operation.
pub const SHAPE: u16 = 0x1;
/// Chain flag: redundant for a two-shape operation.
pub const BOOL_REDUNDANT: u16 = 0x2000;
/// Redundant either way.
pub const REDUNDANT: u16 = SELF_REDUNDANT | BOOL_REDUNDANT;
/// Chain flag: the chain coincides with another and stands for both.
pub const MERGED: u16 = 0x800;
/// Chain flag: flips the module's output direction of a stroke or point chain.
pub const OUT_REVERSED: u16 = 0x8000;
/// Chain flags of a stroke chain (one that bounds no area).
pub const STROKE: u16 = 0x4840;
/// Redundant or cancelled.
pub const DROPPED: u16 = 0x2600;
/// Flags a chain split off another keeps: reversed, operand, hollow.
const INHERITED: u16 = 0x51;
/// Chains a junction's vertex rings (and fill reconstruction) skip.
pub const UNLINKED: u16 = 0x6E40;

/// The scanner gave up on an inconsistent configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScanFailed;
pub(crate) type R<T> = Result<T, ScanFailed>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VKind {
    /// An input point: the base or tip of its segment.
    Endpoint,
    /// The crossing of two segments, known exactly only through them.
    Isect,
    /// A crossing at a grid point (another segment's endpoint).
    Exact,
}

/// An intersection as seen from one of the two segments: whether that segment was the record's AB,
/// the other segment's base vertex, and the record.
#[derive(Clone, Copy, Debug)]
pub struct EdgeIsect {
    pub on_ab: bool,
    pub cross: usize,
    pub rec: usize,
}

#[derive(Clone, Debug)]
pub struct Vertex {
    /// Approximate coordinates (exact for exact kinds, until a junction moves them onto its point).
    pub pt: Pt,
    /// Exact grid coordinates (exact kinds only).
    pub exact: [i64; 2],
    pub kind: VKind,
    pub next: Option<usize>,
    pub prev: Option<usize>,
    /// An endpoint's segment tip; any other vertex's segment base.
    pub seg: Option<usize>,
    pub ei: Option<EdgeIsect>,
    /// The curve fragment of the edge above this vertex.
    pub curve: Option<Cv>,
    pub smooth: bool,
    /// The junction ring through this chain end (itself when unlinked).
    pub link: usize,
}

#[derive(Clone, Debug)]
pub struct SChain {
    pub head: usize,
    /// The base of the current edge.
    pub cursor: usize,
    pub tail: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub flags: u16,
    pub winding: u16,
    /// 1-based index in the candidate heap, 0 when out of it.
    cand: usize,
    /// The module's figure under construction this chain belongs to, and the chain after it there.
    pub owner: Option<usize>,
    pub next: Option<usize>,
}

/// The chains meeting at one point: tails ending and heads starting there, left to right, and the
/// active chains just outside them.
#[derive(Clone, Debug, Default)]
pub struct Junction {
    pub rep: usize,
    pub lt: Option<usize>,
    pub rt: Option<usize>,
    pub lh: Option<usize>,
    pub rh: Option<usize>,
    pub left: Option<usize>,
    pub right: Option<usize>,
    /// Single-point chains at the junction, per operand.
    pub points: [u32; 2],
    /// Set by the module: the junction's single points are covered or joined, so none is emitted.
    pub no_isolated: bool,
}

/// The task the sweep serves.
pub trait Module {
    /// Process the current junction (`ProcessTheJunction`).
    fn junction(&mut self, s: &mut Scanner) -> R<()>;
    /// The cursor of a non-redundant chain moved to the next vertex (`ProcessCurrentVertex`).
    fn current_vertex(&mut self, _s: &mut Scanner, _c: usize) -> R<()> {
        Ok(())
    }
}

/// An intersection on an edge: the edge's [`EdgeIsect`], where it falls on the edge, and its grid point
/// when it lands on a segment's tip.
#[derive(Clone, Copy, Debug)]
struct Res {
    ei: EdgeIsect,
    loc: Option<Loc>,
    exact: Option<[i64; 2]>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Heap {
    Master,
    Cand,
}

fn cmp_pts(a: [i64; 2], b: [i64; 2]) -> Ordering {
    a[1].cmp(&b[1]).then(a[0].cmp(&b[0]))
}

pub struct Scanner {
    pub v: Vec<Vertex>,
    pub c: Vec<SChain>,
    pub recs: Vec<Seg>,
    /// The inactive chains, a max-heap on heads (1-based).
    master: Vec<usize>,
    /// The active chains, a max-heap on their current edges' tips (1-based).
    cand: Vec<usize>,
    pub leftmost: Option<usize>,
    pub j: Junction,
    /// Each operand's fill rule: nonzero (true) or alternate.
    pub winding: [bool; 2],
    /// Recycle what the sweep has passed (see [`Self::recycle_retired`]): set by a module that reads no chain
    /// after it retires dropped.
    pub recycle: bool,
    pool: Pool,
}

/// The slots the sweep has finished with, for reuse, and each intersection record's vertex count. WPF's scanner
/// draws its vertices, chains and intersections from chunked pools and frees an intersection that splits nothing;
/// recycling a retired dropped chain's vertices as well keeps a sweep whose crossings pile up (r11 16792's
/// self-overlapping stroke outline: 4.7 million of them) to the band it is in.
#[derive(Default)]
struct Pool {
    free_v: Vec<usize>,
    free_c: Vec<usize>,
    free_r: Vec<usize>,
    /// Per record: the intersection vertices that reference it, and whether it is in use.
    refs: Vec<u32>,
    live: Vec<bool>,
    /// Records made in the current step, freed after it when nothing took them.
    fresh: Vec<usize>,
    /// Chains retired at the current junction.
    retired: Vec<usize>,
}

impl Scanner {
    /// Loads the chains (stage 1's output, in insertion order) for sweeping.
    pub fn new(chains: &[Chain], winding: [bool; 2]) -> Scanner {
        let mut s = Scanner { v: vec![], c: vec![], recs: vec![], master: vec![0], cand: vec![0], leftmost: None, j: Junction::default(), winding, recycle: false, pool: Pool::default() };
        for ch in chains {
            let first = s.v.len();
            for (i, &p) in ch.pts.iter().enumerate() {
                let id = s.v.len();
                let curve = if i > 0 { ch.edges[i - 1] } else { None };
                s.v.push(Vertex { pt: p, exact: [p[0] as i64, p[1] as i64], kind: VKind::Endpoint, next: None, prev: None, seg: None, ei: None, curve, smooth: false, link: id });
                if i > 0 {
                    s.v[id - 1].next = Some(id);
                    s.v[id - 1].seg = Some(id);
                    s.v[id].prev = Some(id - 1);
                }
            }
            let id = s.c.len();
            s.c.push(SChain { head: first, cursor: first, tail: s.v.len() - 1, left: None, right: None, flags: ch.flags, winding: 0, cand: 0, owner: None, next: None });
            s.master_insert(id);
        }
        s
    }

    /// `Scan`: sweep everything.
    pub fn scan(&mut self, m: &mut dyn Module) -> R<()> {
        while self.move_on(m)? {}
        Ok(())
    }

    // ---- vertices ----

    fn exact_kind(&self, v: usize) -> bool {
        self.v[v].kind != VKind::Isect
    }

    pub fn seg_base(&self, v: usize) -> usize {
        if self.v[v].kind == VKind::Endpoint { v } else { self.v[v].seg.unwrap() }
    }

    fn seg_tip(&self, v: usize) -> Option<usize> {
        self.v[self.seg_base(v)].seg
    }

    /// The segment from base vertex `b` to its tip, `[bx, by, tx, ty]`.
    fn seg_coords(&self, b: usize) -> [i64; 4] {
        let (p, t) = (self.v[b].exact, self.v[self.v[b].seg.unwrap()].exact);
        [p[0], p[1], t[0], t[1]]
    }

    /// `CVertex::CompareWith`: (y, then x) order, exact.
    pub fn cmp(&self, a: usize, b: usize) -> Ordering {
        let (va, vb) = (&self.v[a], &self.v[b]);
        match (self.exact_kind(a), self.exact_kind(b)) {
            (true, true) => cmp_pts(va.exact, vb.exact),
            (true, false) => yx_sort_intersection_and_point(&self.recs[vb.ei.unwrap().rec], va.exact).reverse(),
            (false, true) => yx_sort_intersection_and_point(&self.recs[va.ei.unwrap().rec], vb.exact),
            (false, false) => {
                let (ra, rb) = (va.ei.unwrap().rec, vb.ei.unwrap().rec);
                if ra == rb { Ordering::Equal } else { yx_sort_intersection_pair(&self.recs[ra], &self.recs[rb]) }
            }
        }
    }

    /// `CVertex::LocateVertex`: the side of the edge at `e` vertex `v` lies on — 1 right, 0 on it, −1 left.
    fn locate(&self, e: usize, v: usize) -> i32 {
        let base = self.seg_base(e);
        let ab = self.seg_coords(base);
        if self.exact_kind(v) {
            side_of_point(self.v[v].exact, ab)
        } else {
            let ei = self.v[v].ei.unwrap();
            if ei.cross == base { 0 } else { side_of_intersection(&self.recs[ei.rec], ab) }
        }
    }

    fn alloc(&mut self, mut vx: Vertex) -> usize {
        if vx.kind == VKind::Isect {
            self.pool.refs[vx.ei.unwrap().rec] += 1;
        }
        if let Some(id) = self.pool.free_v.pop() {
            vx.link = id;
            self.v[id] = vx;
            return id;
        }
        let id = self.v.len();
        vx.link = id;
        self.v.push(vx);
        id
    }

    /// A new intersection record's slot.
    fn alloc_rec(&mut self, s: Seg) -> usize {
        let id = match self.pool.free_r.pop() {
            Some(id) => {
                self.recs[id] = s;
                self.pool.refs[id] = 0;
                id
            }
            None => {
                self.recs.push(s);
                self.pool.refs.push(0);
                self.pool.live.push(false);
                self.recs.len() - 1
            }
        };
        self.pool.live[id] = true;
        if self.recycle {
            self.pool.fresh.push(id);
        }
        id
    }

    fn free_rec(&mut self, r: usize) {
        if self.pool.live[r] {
            self.pool.live[r] = false;
            self.pool.free_r.push(r);
        }
    }

    /// Frees the chains retired at the junction just flushed that are dropped and no module figure holds: their
    /// vertices other than segment endpoints (which other chains' vertices reference as their segment's base), the
    /// records only those referenced, and the chain.
    fn recycle_retired(&mut self) {
        let retired = std::mem::take(&mut self.pool.retired);
        for &c in &retired {
            let ch = &self.c[c];
            if ch.flags & DROPPED == 0 || ch.flags & MERGED != 0 || ch.owner.is_some() || ch.next.is_some() {
                continue;
            }
            let (head, tail) = (ch.head, ch.tail);
            let mut x = Some(head);
            while let Some(v) = x {
                x = if v == tail { None } else { self.v[v].next };
                match self.v[v].kind {
                    VKind::Endpoint => continue,
                    VKind::Isect => {
                        let r = self.v[v].ei.unwrap().rec;
                        self.pool.refs[r] -= 1;
                        if self.pool.refs[r] == 0 {
                            self.free_rec(r);
                        }
                    }
                    VKind::Exact => {}
                }
                self.pool.free_v.push(v);
            }
            self.pool.free_c.push(c);
        }
        self.pool.retired = retired;
        self.pool.retired.clear();
    }

    /// Frees the step's new records that no vertex took.
    fn recycle_fresh(&mut self) {
        let fresh = std::mem::take(&mut self.pool.fresh);
        for &r in &fresh {
            if self.pool.refs[r] == 0 {
                self.free_rec(r);
            }
        }
        self.pool.fresh = fresh;
        self.pool.fresh.clear();
    }

    fn copy_vertex(&mut self, v: usize) -> usize {
        let mut n = self.v[v].clone();
        n.next = None;
        n.prev = None;
        n.curve = None;
        self.alloc(n)
    }

    fn exact_vertex(&mut self, p: [i64; 2]) -> usize {
        let pt = [p[0] as f64, p[1] as f64];
        self.alloc(Vertex { pt, exact: p, kind: VKind::Exact, next: None, prev: None, seg: None, ei: None, curve: None, smooth: false, link: 0 })
    }

    /// `CVertex::LinkEdgeTo`: link `v` to `next` and repoint the segment links below.
    fn link_edge_to(&mut self, v: usize, next: Option<usize>) {
        self.v[v].next = next;
        if let Some(n) = next {
            self.v[n].prev = Some(v);
        }
        let base = if self.v[v].kind == VKind::Endpoint { Some(v) } else { self.v[v].seg };
        let mut p = next;
        while let Some(x) = p {
            if self.v[x].kind == VKind::Endpoint {
                break;
            }
            self.v[x].seg = base;
            p = self.v[x].next;
        }
        if let (Some(x), Some(b)) = (p, base) {
            self.v[b].seg = Some(x);
        }
    }

    // ---- intersections ----

    /// `CIntersectionResult::IntersectSegments`: a transverse crossing of the half-open segments (base
    /// excluded) based at `ab` and `cd`, seen from `ab`, with its location on each.
    fn intersect_segments(&mut self, ab: usize, cd: usize) -> Option<(Res, Loc, Loc)> {
        let (p, q) = (self.seg_coords(ab), self.seg_coords(cd));
        let s = Seg::new(p, q);
        if s.kind != Kind::Transverse || s.loc_ab == Some(Loc::AtFirst) || s.loc_cd == Some(Loc::AtFirst) {
            return None;
        }
        let (la, lc) = (s.loc_ab.unwrap(), s.loc_cd.unwrap());
        let exact = if la == Loc::AtLast {
            Some([p[2], p[3]])
        } else if lc == Loc::AtLast {
            Some([q[2], q[3]])
        } else {
            None
        };
        let rec = self.alloc_rec(s);
        Some((Res { ei: EdgeIsect { on_ab: true, cross: cd, rec }, loc: None, exact }, la, lc))
    }

    /// Orders an intersection result against vertex `x` on the same segment.
    fn cmp_res(&self, r: &Res, x: usize) -> Ordering {
        if self.exact_kind(x) {
            match r.exact {
                Some(e) => cmp_pts(e, self.v[x].exact),
                None => yx_sort_intersection_and_point(&self.recs[r.ei.rec], self.v[x].exact),
            }
        } else {
            let xe = self.v[x].ei.unwrap();
            match r.exact {
                Some(e) => yx_sort_intersection_and_point(&self.recs[xe.rec], e).reverse(),
                // Parameters grow down the segment, against the (y, x) order.
                None => param_cmp(&self.recs[r.ei.rec], r.ei.on_ab, &self.recs[xe.rec], xe.on_ab).reverse(),
            }
        }
    }

    /// `CVertex::QueryAndSetEdgeIntersection`: whether the intersection, at `loc` on its segment, lies on
    /// the half-open edge at `e` (base excluded), and where.
    fn query_and_set(&self, e: usize, loc: Loc, r: &mut Res) -> bool {
        let tip = self.v[e].next.unwrap();
        let below_base = if self.v[e].kind == VKind::Endpoint {
            if self.v[tip].kind == VKind::Endpoint {
                r.loc = Some(loc);
                return true;
            }
            Ordering::Less
        } else {
            self.cmp_res(r, e)
        };
        if below_base != Ordering::Less {
            return false;
        }
        if self.v[tip].kind == VKind::Endpoint && loc == Loc::AtLast {
            r.loc = Some(Loc::AtLast);
            return true;
        }
        match self.cmp_res(r, tip) {
            Ordering::Less => false,
            Ordering::Equal => {
                r.loc = Some(Loc::AtLast);
                true
            }
            Ordering::Greater => {
                r.loc = Some(Loc::Open);
                true
            }
        }
    }

    /// `CVertex::Intersect`: the crossing of the edges at `a` and `b`, seen from each.
    fn intersect(&mut self, a: usize, b: usize) -> R<Option<(Res, Res)>> {
        if self.seg_tip(a).is_none() || self.seg_tip(b).is_none() {
            return Err(ScanFailed);
        }
        let (sa, sb) = (self.seg_base(a), self.seg_base(b));
        let Some((mut ra, la, lc)) = self.intersect_segments(sa, sb) else { return Ok(None) };
        let mut rb = Res { ei: EdgeIsect { on_ab: false, cross: sa, rec: ra.ei.rec }, ..ra };
        if !self.query_and_set(a, la, &mut ra) || !self.query_and_set(b, lc, &mut rb) {
            return Ok(None);
        }
        Ok(Some((ra, rb)))
    }

    // ---- chain surgery ----

    fn tip(&self, c: usize) -> usize {
        self.v[self.c[c].cursor].next.unwrap()
    }

    fn is_tail_isect(&self, c: usize, r: &Res) -> bool {
        r.loc == Some(Loc::AtLast) && self.tip(c) == self.c[c].tail
    }

    /// `CChain::SplitAtVertex`: cut the chain at `pv`; the part below becomes a new chain.
    fn split_at_vertex(&mut self, c: usize, pv: usize) -> Option<usize> {
        if pv == self.c[c].head || self.v[pv].next.is_none() {
            return None;
        }
        let cp = self.copy_vertex(pv);
        let ch = SChain { head: cp, cursor: cp, tail: self.c[c].tail, left: None, right: None, flags: self.c[c].flags & INHERITED, winding: 0, cand: 0, owner: None, next: None };
        let id = match self.pool.free_c.pop() {
            Some(id) => {
                self.c[id] = ch;
                id
            }
            None => {
                self.c.push(ch);
                self.c.len() - 1
            }
        };
        let nx = self.v[pv].next;
        self.link_edge_to(cp, nx);
        self.v[cp].smooth = false;
        self.v[pv].next = None;
        self.c[c].tail = pv;
        Some(id)
    }

    /// `CChain::SplitAtIntersection`: split at an intersection on the current edge.
    fn split_at_intersection(&mut self, c: usize, r: &Res) -> Option<usize> {
        let cur = self.c[c].cursor;
        let tip = self.v[cur].next?;
        if r.loc == Some(Loc::AtLast) {
            return if tip != self.c[c].tail { self.split_at_vertex(c, tip) } else { None };
        }
        let nv = match r.exact {
            Some(e) => self.exact_vertex(e),
            None => {
                let b = self.seg_base(cur);
                let q = self.seg_coords(b);
                let t = self.recs[r.ei.rec].parameter(r.ei.on_ab);
                let pt = approx_point([q[0] as f64, q[1] as f64], [q[2] as f64, q[3] as f64], t);
                self.alloc(Vertex { pt, exact: [0, 0], kind: VKind::Isect, next: None, prev: None, seg: None, ei: Some(r.ei), curve: None, smooth: false, link: 0 })
            }
        };
        // The edge is subdivided, so its curve fragment no longer fits it.
        self.v[tip].curve = None;
        self.link_edge_to(nv, Some(tip));
        self.link_edge_to(cur, Some(nv));
        self.split_at_vertex(c, nv)
    }

    /// `CChain::SplitAtExactPoint`: split at a grid point on the current edge.
    fn split_at_exact_point(&mut self, c: usize, p: [i64; 2]) -> Option<usize> {
        let cur = self.c[c].cursor;
        let tip = self.v[cur].next?;
        let nv = self.exact_vertex(p);
        let at = if self.cmp(nv, tip) != Ordering::Equal {
            self.v[tip].curve = None;
            self.link_edge_to(nv, Some(tip));
            self.link_edge_to(cur, Some(nv));
            nv
        } else if tip != self.c[c].tail {
            tip
        } else {
            return None;
        };
        self.split_at_vertex(c, at)
    }

    /// `CChain::SplitAtIncidentVertex`: split where vertex `pv` of another chain lies on the current edge.
    fn split_at_incident_vertex(&mut self, c: usize, pv: usize) -> R<Option<usize>> {
        if self.exact_kind(pv) {
            let p = self.v[pv].exact;
            return Ok(self.split_at_exact_point(c, p));
        }
        // Cross the current edge with one of the segments through `pv` that is not collinear with it.
        let cur = self.c[c].cursor;
        let base = self.seg_base(cur);
        let cross = self.v[pv].ei.unwrap().cross;
        let own = self.seg_base(pv);
        for other in [cross, own] {
            if let Some((mut r, la, _)) = self.intersect_segments(base, other) {
                if !self.query_and_set(cur, la, &mut r) {
                    return Err(ScanFailed);
                }
                return Ok(self.split_at_intersection(c, &r));
            }
        }
        Err(ScanFailed)
    }

    // ---- heaps ----

    fn heap(&self, h: Heap) -> &Vec<usize> {
        if h == Heap::Master { &self.master } else { &self.cand }
    }

    fn key(&self, h: Heap, c: usize) -> usize {
        if h == Heap::Master { self.c[c].head } else { self.tip(c) }
    }

    fn heap_gt(&self, h: Heap, i: usize, j: usize) -> bool {
        let hv = self.heap(h);
        self.cmp(self.key(h, hv[i]), self.key(h, hv[j])) == Ordering::Greater
    }

    fn heap_swap(&mut self, h: Heap, i: usize, j: usize) {
        if h == Heap::Cand {
            let (a, b) = (self.cand[i], self.cand[j]);
            self.c[a].cand = j;
            self.c[b].cand = i;
            self.cand.swap(i, j);
        } else {
            self.master.swap(i, j);
        }
    }

    fn bubble_up(&mut self, h: Heap, mut i: usize) -> usize {
        while i > 1 && self.heap_gt(h, i, i / 2) {
            self.heap_swap(h, i, i / 2);
            i /= 2;
        }
        i
    }

    fn push_down(&mut self, h: Heap, mut i: usize) {
        let end = self.heap(h).len();
        while i < end {
            let lg = 2 * i < end && self.heap_gt(h, 2 * i, i);
            let rg = 2 * i + 1 < end && self.heap_gt(h, 2 * i + 1, i);
            let n = match (lg, rg) {
                (false, false) => end,
                (true, false) => 2 * i,
                (false, true) => 2 * i + 1,
                (true, true) => if self.heap_gt(h, 2 * i, 2 * i + 1) { 2 * i } else { 2 * i + 1 },
            };
            if n < end {
                self.heap_swap(h, n, i);
            }
            i = n;
        }
    }

    fn heap_remove(&mut self, h: Heap, i: usize) {
        let last = self.heap(h).len() - 1;
        if h == Heap::Cand {
            let c = self.cand[i];
            self.c[c].cand = 0;
        }
        if i < last {
            let moved = self.heap(h)[last];
            if h == Heap::Cand {
                self.cand[i] = moved;
                self.c[moved].cand = i;
                self.cand.pop();
            } else {
                self.master[i] = moved;
                self.master.pop();
            }
            if self.bubble_up(h, i) == i {
                self.push_down(h, i);
            }
        } else if h == Heap::Cand {
            self.cand.pop();
        } else {
            self.master.pop();
        }
    }

    fn heap_insert(&mut self, h: Heap, c: usize) {
        if h == Heap::Cand {
            self.cand.push(c);
            self.c[c].cand = self.cand.len() - 1;
        } else {
            self.master.push(c);
        }
        let n = self.heap(h).len() - 1;
        self.bubble_up(h, n);
    }

    /// `CChainList::Insert`: queue a chain for activation, its cursor back at its head.
    fn master_insert(&mut self, c: usize) {
        self.heap_insert(Heap::Master, c);
        let ch = &mut self.c[c];
        ch.cursor = ch.head;
        ch.left = None;
        ch.right = None;
    }

    /// `CHeap::Update`: re-sort a candidate whose current edge changed.
    fn cand_update(&mut self, c: usize) {
        let i = self.c[c].cand;
        if self.bubble_up(Heap::Cand, i) == i {
            self.push_down(Heap::Cand, i);
        }
    }

    fn master_top(&self) -> Option<usize> {
        self.master.get(1).copied()
    }

    fn cand_top(&self) -> Option<usize> {
        self.cand.get(1).copied()
    }

    // ---- active list ----

    fn link(&mut self, l: Option<usize>, r: Option<usize>) {
        if let Some(l) = l {
            self.c[l].right = r;
        }
        if let Some(r) = r {
            self.c[r].left = l;
        }
    }

    /// `CActiveList::Locate`: the active chains around a new chain's head, and whether it lies on the
    /// right one.
    fn locate_active(&self, c: usize) -> (Option<usize>, Option<usize>, bool) {
        let head = self.c[c].head;
        let (mut left, mut right) = (None, self.leftmost);
        while let Some(r) = right {
            let side = self.locate(self.c[r].cursor, head);
            if side != 1 {
                return (left, right, side == 0);
            }
            left = right;
            right = self.c[r].right;
        }
        (left, None, false)
    }

    fn active_remove(&mut self, first: usize, last: usize) {
        let (p, n) = (self.c[first].left, self.c[last].right);
        match p {
            None => self.leftmost = n,
            Some(p) => self.c[p].right = n,
        }
        if let Some(n) = n {
            self.c[n].left = p;
        }
        self.c[first].left = None;
        self.c[last].right = None;
    }

    // ---- scanner-level splits ----

    fn split_chain_at_intersection(&mut self, c: usize, r: &Res) {
        if let Some(p) = self.split_at_intersection(c, r) {
            self.master_insert(p);
        }
    }

    fn split_chain_at_current_edge_tip(&mut self, c: usize) {
        let tip = self.tip(c);
        if let Some(p) = self.split_at_vertex(c, tip) {
            self.master_insert(p);
        }
    }

    /// `SplitChainAtCoincidentEdge`.
    fn split_chain_at_coincident_edge(&mut self, c: usize, v: usize) -> R<()> {
        if let Some(p) = self.split_at_incident_vertex(c, v)? {
            self.master_insert(p);
        }
        Ok(())
    }

    /// `SplitNeighbor`: split `c` (not a candidate) and its candidate neighbour where their current edges
    /// cross; true if the neighbour was split.
    fn split_neighbor(&mut self, c: usize, nb: Option<usize>) -> R<bool> {
        let Some(nb) = nb else { return Ok(false) };
        let Some((rc, rn)) = self.intersect(self.c[c].cursor, self.c[nb].cursor)? else { return Ok(false) };
        if !self.is_tail_isect(c, &rc) {
            self.split_chain_at_intersection(c, &rc);
        }
        if self.is_tail_isect(nb, &rn) {
            return Ok(false);
        }
        self.split_chain_at_intersection(nb, &rn);
        self.cand_update(nb);
        Ok(true)
    }

    fn split_coincident_left_of(&mut self, c: usize) -> R<()> {
        let mut k = self.c[c].left;
        while let Some(x) = k {
            if self.c[x].flags & COINCIDENT == 0 {
                break;
            }
            self.split_chain_at_coincident_edge(x, self.c[c].tail)?;
            self.cand_update(x);
            k = self.c[x].left;
        }
        Ok(())
    }

    fn split_coincident_right_of(&mut self, c: usize) -> R<()> {
        let mut i = c;
        while self.c[i].flags & COINCIDENT != 0 {
            i = self.c[i].right.unwrap();
            self.split_chain_at_coincident_edge(i, self.c[c].tail)?;
            self.cand_update(i);
        }
        Ok(())
    }

    /// `SplitPairAtIntersection`: two candidates that become neighbours.
    fn split_pair_at_intersection(&mut self, l: Option<usize>, r: Option<usize>) -> R<()> {
        let (Some(l), Some(r)) = (l, r) else { return Ok(()) };
        let Some((rl, rr)) = self.intersect(self.c[l].cursor, self.c[r].cursor)? else { return Ok(()) };
        if !self.is_tail_isect(l, &rl) {
            self.split_chain_at_intersection(l, &rl);
            self.cand_update(l);
            self.split_coincident_left_of(l)?;
        }
        if !self.is_tail_isect(r, &rr) {
            self.split_chain_at_intersection(r, &rr);
            self.cand_update(r);
            self.split_coincident_right_of(r)?;
        }
        Ok(())
    }

    /// `SplitAtCoincidentIntersection`: two head chains running along each other are cut where the
    /// shorter first edge ends.
    fn split_at_coincident_intersection(&mut self, c: usize) -> R<()> {
        let Some(r) = self.c[c].right else { return Ok(()) };
        let ab = self.seg_coords(self.seg_base(self.c[c].cursor));
        let rt = self.seg_tip(self.c[r].cursor).unwrap();
        if side_of_point(self.v[rt].exact, ab) != 0 {
            return Ok(());
        }
        match self.cmp(self.tip(c), self.tip(r)) {
            Ordering::Greater => {
                self.split_chain_at_current_edge_tip(c);
                self.split_chain_at_coincident_edge(r, self.tip(c))?;
            }
            Ordering::Less => {
                self.split_chain_at_current_edge_tip(r);
                let mut l = c;
                loop {
                    self.split_chain_at_coincident_edge(l, self.c[r].tail)?;
                    match self.c[l].left {
                        Some(x) if self.c[x].flags & COINCIDENT != 0 => l = x,
                        _ => break,
                    }
                }
            }
            Ordering::Equal => {
                self.split_chain_at_current_edge_tip(c);
                self.split_chain_at_current_edge_tip(r);
            }
        }
        self.c[c].flags |= COINCIDENT;
        Ok(())
    }

    // ---- junctions ----

    /// Adds a chain starting at the junction: a single point is only counted.
    fn take_head(&mut self, c: usize) {
        if self.c[c].head == self.c[c].tail {
            self.j.points[(self.c[c].flags & SHAPE) as usize] += 1;
        } else if self.j.lh.is_none() {
            self.j.lh = Some(c);
            self.j.rh = Some(c);
        } else {
            self.insert_head(c);
        }
    }

    /// `CJunction::InsertHead`: place a head among the others by its first edge's direction.
    fn insert_head(&mut self, c: usize) {
        let Some(lh) = self.j.lh else {
            self.j.lh = Some(c);
            self.j.rh = Some(c);
            return;
        };
        let key = self.tip(c);
        let (mut prev, mut cur) = (None, Some(lh));
        while prev != self.j.rh && self.locate(self.c[cur.unwrap()].cursor, key) == 1 {
            prev = cur;
            cur = self.c[cur.unwrap()].right;
        }
        self.c[c].left = prev;
        self.c[c].right = cur;
        self.link(prev, Some(c));
        self.link(Some(c), cur);
        if cur == Some(lh) {
            self.j.lh = Some(c);
        } else if prev == self.j.rh {
            self.j.rh = Some(c);
        }
    }

    /// Grabs the inactive chains starting at the junction point.
    fn grab_heads(&mut self) {
        while let Some(t) = self.master_top() {
            if self.cmp(self.j.rep, self.c[t].head) != Ordering::Equal {
                break;
            }
            self.heap_remove(Heap::Master, 1);
            self.take_head(t);
        }
    }

    /// `Activate`: start the junction at a new chain's head.
    fn activate(&mut self, c: usize, m: &mut dyn Module) -> R<()> {
        let (left, right, incident) = self.locate_active(c);
        self.j.rep = self.c[c].head;
        self.j.left = left;
        self.j.right = right;
        self.take_head(c);
        if incident {
            // The point lies on active chains: they end here, and their rest starts here.
            self.j.lt = right;
            let mut r = right;
            loop {
                self.j.rt = r;
                let Some(rc) = r else {
                    self.j.right = None;
                    break;
                };
                let piece = self.split_at_incident_vertex(rc, self.j.rep)?;
                self.cand_update(rc);
                self.insert_head(piece.ok_or(ScanFailed)?);
                if self.c[rc].flags & COINCIDENT == 0 {
                    self.j.right = self.c[rc].right;
                    break;
                }
                r = self.c[rc].right;
            }
        }
        self.grab_heads();
        self.flush(m)
    }

    /// The junction at a chain's tail (`CJunction::ProcessAtTail`).
    fn process_at_tail(&mut self, c: usize, m: &mut dyn Module) -> R<()> {
        self.j.lt = Some(c);
        self.j.rt = Some(c);
        self.j.rep = self.c[c].tail;
        let mut l = self.c[c].left;
        while let Some(x) = l {
            if self.cmp(self.c[x].tail, self.j.rep) != Ordering::Equal {
                break;
            }
            self.j.lt = Some(x);
            l = self.c[x].left;
        }
        self.j.left = l;
        let mut r = self.c[c].right;
        while let Some(x) = r {
            if self.cmp(self.c[x].tail, self.j.rep) != Ordering::Equal {
                break;
            }
            self.j.rt = Some(x);
            r = self.c[x].right;
        }
        self.j.right = r;
        self.grab_heads();
        self.flush(m)
    }

    /// Walks from `from` along `right` (or left) to the first chain not skipped by the rings.
    pub fn relevant(&self, from: Option<usize>, rightward: bool) -> Option<usize> {
        self.go_while(from, rightward, |f| f & UNLINKED != 0)
    }

    /// `GoRightWhileRedundant`/`GoLeftWhileRedundant`: the first chain from `from` that `skip` rejects not.
    pub fn go_while(&self, from: Option<usize>, rightward: bool, skip: impl Fn(u16) -> bool) -> Option<usize> {
        let mut x = from;
        while let Some(c) = x {
            if !skip(self.c[c].flags) {
                return Some(c);
            }
            x = if rightward { self.c[c].right } else { self.c[c].left };
        }
        None
    }

    // ---- classification (chain methods the modules call) ----

    /// The chain's side after a two-shape operation's flip.
    pub fn side_right(&self, c: usize) -> bool {
        let f = self.c[c].flags;
        (f & SIDE_RIGHT != 0) != (f & 0x1000 != 0)
    }

    pub fn is_point(&self, c: usize) -> bool {
        self.c[c].head == self.c[c].tail
    }

    /// `CChain::ReverseOnOutput`.
    pub fn reverse_on_output(&self, c: usize) -> bool {
        if self.is_point(c) || self.c[c].flags & STROKE != 0 { self.c[c].flags & OUT_REVERSED != 0 } else { self.side_right(c) }
    }

    /// `CChain::Classify`: side and redundancy from the chain on the left (fill rule dependent).
    pub fn classify(&mut self, c: usize, left: Option<usize>) {
        let f = self.c[c].flags;
        if self.winding[(f & SHAPE) as usize] {
            let lw = left.map_or(0, |l| self.c[l].winding);
            if f & STROKE != 0 && lw != 0 {
                self.c[c].flags |= SELF_REDUNDANT;
                return;
            }
            let w = if f & REVERSED_FLAG != 0 { lw.wrapping_sub(1) } else { lw.wrapping_add(1) };
            self.c[c].winding = w;
            if lw != 0 {
                self.c[c].flags |= if w != 0 { SELF_REDUNDANT } else { SIDE_RIGHT };
            } else if w == 0 {
                self.c[c].flags |= SELF_REDUNDANT;
            }
        } else if left.is_some_and(|l| self.c[l].flags & SIDE_RIGHT == 0) {
            self.c[c].flags |= if f & 0x40 != 0 { SELF_REDUNDANT } else { SIDE_RIGHT };
        }
    }

    /// `ClassifyFillOnly`: classify the junction's heads by the fill rule alone (the first continues
    /// the leftmost tail, the rest follow their left neighbour), then cancel each pair of adjacent
    /// live single-edge heads that end at the same point.
    pub fn classify_fill_only(&mut self) {
        let Some(lh) = self.j.lh else { return };
        let (mut x, mut left) = (Some(lh), self.j.left);
        if let Some(t) = self.j.lt {
            self.continue_from(lh, t);
            left = Some(lh);
            x = self.c[lh].right;
        }
        while let Some(c) = x {
            self.classify(c, left);
            left = Some(c);
            x = self.c[c].right;
        }
        let redundant = |f: u16| f & REDUNDANT != 0;
        let single = |s: &Scanner, c: usize| s.v[s.c[c].head].next == Some(s.c[c].tail);
        let mut a = self.go_while(Some(lh), true, redundant);
        while let Some(p) = a {
            let Some(q) = self.go_while(self.c[p].right, true, redundant) else { break };
            if single(self, p) && single(self, q) && self.cmp(self.c[p].tail, self.c[q].tail) == Ordering::Equal {
                self.c[p].flags |= CANCELLED;
                self.c[q].flags |= CANCELLED;
                a = self.go_while(self.c[q].right, true, redundant);
            } else {
                a = Some(q);
            }
        }
    }

    /// `CChain::Continue`: side and redundancy continuing the tail `above` at the junction.
    pub fn continue_from(&mut self, c: usize, above: usize) {
        let (f, af, aw) = (self.c[c].flags, self.c[above].flags, self.c[above].winding);
        if !self.winding[(f & SHAPE) as usize] {
            self.c[c].flags |= af & 0x300;
            return;
        }
        if (af & REVERSED_FLAG != 0) == (f & REVERSED_FLAG != 0) {
            self.c[c].winding = aw;
            self.c[c].flags |= af & 0x300;
            return;
        }
        let w = if f & REVERSED_FLAG != 0 { aw.wrapping_sub(2) } else { aw.wrapping_add(2) };
        self.c[c].winding = w;
        match af & 0x300 {
            SIDE_RIGHT => self.c[c].flags |= SELF_REDUNDANT,
            SELF_REDUNDANT => self.c[c].flags |= if w == 0 { SIDE_RIGHT } else { SELF_REDUNDANT },
            _ => {}
        }
    }

    /// Links the end vertices (`end`) of the relevant chains from `first` to `last` into a ring.
    fn ring(&mut self, first: Option<usize>, last: Option<usize>, end: fn(&SChain) -> usize) {
        let (Some(k), Some(m)) = (first, last) else { return };
        let mut x = k;
        while x != m {
            let n = self.relevant(self.c[x].right, true).unwrap();
            let (a, b) = (end(&self.c[x]), end(&self.c[n]));
            self.v[a].link = b;
            x = n;
        }
        let (a, b) = (end(&self.c[m]), end(&self.c[k]));
        self.v[a].link = b;
    }

    /// Chains from `l` to `r` along `right`.
    pub fn span(&self, l: Option<usize>, r: Option<usize>) -> Vec<usize> {
        let mut out = vec![];
        let mut x = l;
        while let Some(c) = x {
            out.push(c);
            if Some(c) == r {
                break;
            }
            x = self.c[c].right;
        }
        out
    }

    /// `CJunction::Flush`.
    fn flush(&mut self, m: &mut dyn Module) -> R<()> {
        // Retire the tails.
        if let (Some(lt), Some(rt)) = (self.j.lt, self.j.rt) {
            if self.recycle {
                self.pool.retired = self.span(Some(lt), Some(rt));
            }
            self.active_remove(lt, rt);
            for c in self.span(Some(lt), Some(rt)) {
                let i = self.c[c].cand;
                self.heap_remove(Heap::Cand, i);
            }
        }
        let (tk, tm) = (self.relevant(self.j.lt, true), self.relevant(self.j.rt, false));
        self.ring(tk, tm, |c| c.tail);
        // Split the heads against their neighbours and each other, or the neighbours against each other.
        if let (Some(lh), Some(rh)) = (self.j.lh, self.j.rh) {
            let (left, right) = (self.j.left, self.j.right);
            if self.split_neighbor(lh, left)? {
                self.split_coincident_left_of(left.unwrap())?;
            }
            if self.split_neighbor(rh, right)? {
                self.split_coincident_right_of(right.unwrap())?;
            }
            let mut x = Some(lh);
            while let Some(c) = x {
                if c == rh {
                    break;
                }
                self.split_at_coincident_intersection(c)?;
                x = self.c[c].right;
            }
        } else {
            self.split_pair_at_intersection(self.j.left, self.j.right)?;
        }
        // Every chain end at the junction takes the representative point.
        let p = self.v[self.j.rep].pt;
        for c in self.span(self.j.lh, self.j.rh) {
            let h = self.c[c].head;
            self.v[h].pt = p;
        }
        for c in self.span(self.j.lt, self.j.rt) {
            let t = self.c[c].tail;
            self.v[t].pt = p;
        }
        m.junction(self)?;
        // Ring the relevant heads, through the relevant tails when there are any.
        let (nn, i1) = (self.relevant(self.j.lt, true), self.relevant(self.j.rt, false));
        let (i2, i3) = (self.relevant(self.j.lh, true), self.relevant(self.j.rh, false));
        if let (Some(i2), Some(i3)) = (i2, i3) {
            let mut x = i2;
            while x != i3 {
                let n = self.relevant(self.c[x].right, true).unwrap();
                let (a, b) = (self.c[x].head, self.c[n].head);
                self.v[a].link = b;
                x = n;
            }
            let mut to = self.c[i2].head;
            if let (Some(nn), Some(i1)) = (nn, i1) {
                let t = self.c[i1].tail;
                self.v[t].link = to;
                to = self.c[nn].tail;
            }
            let h = self.c[i3].head;
            self.v[h].link = to;
        }
        // Activate the heads.
        if let (Some(lh), Some(rh)) = (self.j.lh, self.j.rh) {
            let (prev, next) = (self.j.left, self.j.right);
            self.link(prev, Some(lh));
            self.link(Some(rh), next);
            if prev.is_none() {
                self.leftmost = Some(lh);
            }
            for c in self.span(Some(lh), Some(rh)) {
                self.heap_insert(Heap::Cand, c);
            }
        }
        self.j = Junction::default();
        if self.recycle {
            self.recycle_retired();
        }
        Ok(())
    }

    // ---- the sweep ----

    /// `ProcessCandidate`: move the top candidate down its chain and process the vertex reached.
    fn process_candidate(&mut self, c: usize, m: &mut dyn Module) -> R<()> {
        let tip = self.tip(c);
        if tip == self.c[c].tail {
            return self.process_at_tail(c, m);
        }
        self.c[c].cursor = tip;
        let (left, right) = (self.c[c].left, self.c[c].right);
        let apart = |s: &Scanner, nb: Option<usize>| nb.map_or(true, |nb| !s.x_overlap(c, nb));
        let (l_apart, r_apart) = (apart(self, left), apart(self, right));
        if l_apart && r_apart {
            self.cand_update(c);
        } else {
            self.heap_remove(Heap::Cand, 1);
            if !l_apart && self.split_neighbor(c, left)? {
                self.split_coincident_left_of(left.unwrap())?;
            }
            if !r_apart && self.split_neighbor(c, right)? {
                self.split_coincident_right_of(right.unwrap())?;
            }
            self.heap_insert(Heap::Cand, c);
        }
        if self.c[c].flags & REDUNDANT == 0 {
            m.current_vertex(self, c)?;
        }
        Ok(())
    }

    /// Whether two active chains' current segments overlap in x.
    fn x_overlap(&self, a: usize, b: usize) -> bool {
        let range = |c: usize| {
            let q = self.seg_coords(self.seg_base(self.c[c].cursor));
            (q[0].min(q[2]), q[0].max(q[2]))
        };
        let ((a0, a1), (b0, b1)) = (range(a), range(b));
        b0 <= a1 && a0 <= b1
    }

    /// `MoveOn`: process the next highest vertex; false when done.
    fn move_on(&mut self, m: &mut dyn Module) -> R<bool> {
        match (self.master_top(), self.cand_top()) {
            (None, None) => return Ok(false),
            (None, Some(c)) => self.process_candidate(c, m)?,
            (Some(i), None) => {
                self.heap_remove(Heap::Master, 1);
                self.activate(i, m)?;
            }
            (Some(i), Some(c)) => {
                let o = self.cmp(self.tip(c), self.c[i].head);
                if o == Ordering::Greater || (o == Ordering::Equal && self.tip(c) == self.c[c].tail) {
                    self.process_candidate(c, m)?;
                } else {
                    self.heap_remove(Heap::Master, 1);
                    self.activate(i, m)?;
                }
            }
        }
        if self.recycle {
            self.recycle_fresh();
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::super::scanner::ChainList;
    use super::*;

    /// Records each junction: its point, tails and heads (as lists of chain points), and isolated points.
    #[derive(Default)]
    struct Rec(Vec<(Pt, Vec<Vec<Pt>>, Vec<Vec<Pt>>, [u32; 2])>);

    fn pts(s: &Scanner, c: usize) -> Vec<Pt> {
        let mut out = vec![];
        let mut v = Some(s.c[c].head);
        while let Some(x) = v {
            out.push(s.v[x].pt);
            v = s.v[x].next;
        }
        out
    }

    impl Module for Rec {
        fn junction(&mut self, s: &mut Scanner) -> R<()> {
            let tails = s.span(s.j.lt, s.j.rt).into_iter().map(|c| pts(s, c)).collect();
            let heads = s.span(s.j.lh, s.j.rh).into_iter().map(|c| pts(s, c)).collect();
            self.0.push((s.v[s.j.rep].pt, tails, heads, s.j.points));
            Ok(())
        }
    }

    fn sweep(figs: &[&[Pt]]) -> Vec<(Pt, Vec<Vec<Pt>>, Vec<Vec<Pt>>, [u32; 2])> {
        let mut l = ChainList::default();
        for f in figs {
            l.begin_figure(f[0], true);
            for &p in &f[1..] {
                l.add(p);
            }
            l.end_figure(true);
        }
        let mut s = Scanner::new(l.chains(), [false; 2]);
        let mut r = Rec::default();
        s.scan(&mut r).unwrap();
        r.0
    }

    // A square: both chains start at the bottom-right corner (the bottom edge's chain on the left) and
    // end at the top-left one.
    #[test]
    fn square_has_two_junctions() {
        let j = sweep(&[&[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]]);
        let bottom = vec![[10.0, 10.0], [0.0, 10.0], [0.0, 0.0]];
        let right = vec![[10.0, 10.0], [10.0, 0.0], [0.0, 0.0]];
        assert_eq!(j, vec![
            ([10.0, 10.0], vec![], vec![bottom.clone(), right.clone()], [0, 0]),
            ([0.0, 0.0], vec![bottom, right], vec![], [0, 0]),
        ]);
    }

    // A bowtie crosses itself at (5, 5): the chains are split there, so that junction has two tails and
    // two heads, the heads crossed over.
    #[test]
    fn bowtie_splits_at_the_crossing() {
        let j = sweep(&[&[[0.0, 0.0], [10.0, 10.0], [10.0, 0.0], [0.0, 10.0]]]);
        let mid = j.iter().find(|x| x.0 == [5.0, 5.0]).expect("crossing junction");
        assert_eq!((mid.1.len(), mid.2.len()), (2, 2));
        assert_eq!(mid.2, vec![vec![[5.0, 5.0], [0.0, 0.0]], vec![[5.0, 5.0], [10.0, 0.0]]]);
        assert_eq!(j.len(), 5);
    }

    // A triangle whose top vertex lies on a square's right side: activating it splits that side there.
    #[test]
    fn vertex_on_a_chain_splits_it() {
        let j = sweep(&[&[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], &[[10.0, 5.0], [20.0, 0.0], [12.0, 0.0]]]);
        let at = j.iter().find(|x| x.0 == [10.0, 5.0]).unwrap();
        assert_eq!(at.1, vec![vec![[10.0, 10.0], [10.0, 5.0]]]);
        assert_eq!(at.2, vec![
            vec![[10.0, 5.0], [10.0, 0.0], [0.0, 0.0]],
            vec![[10.0, 5.0], [12.0, 0.0]],
            vec![[10.0, 5.0], [20.0, 0.0], [12.0, 0.0]],
        ]);
    }

    // Two squares sharing the side x = 10: the two chains along it start together and are cut to the
    // shared edge, the left one marked coincident with its right neighbour.
    #[test]
    fn shared_side_is_coincident() {
        let mut l = ChainList::default();
        for f in [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], [[10.0, 0.0], [20.0, 0.0], [20.0, 10.0], [10.0, 10.0]]] {
            l.begin_figure(f[0], true);
            for &p in &f[1..] {
                l.add(p);
            }
            l.end_figure(true);
        }
        let mut s = Scanner::new(l.chains(), [false; 2]);
        struct Flags(Vec<(Pt, Vec<(Vec<Pt>, bool)>)>);
        impl Module for Flags {
            fn junction(&mut self, s: &mut Scanner) -> R<()> {
                let heads = s.span(s.j.lh, s.j.rh).into_iter().map(|c| (pts(s, c), s.c[c].flags & COINCIDENT != 0)).collect();
                self.0.push((s.v[s.j.rep].pt, heads));
                Ok(())
            }
        }
        let mut f = Flags(vec![]);
        s.scan(&mut f).unwrap();
        let at = f.0.iter().find(|x| x.0 == [10.0, 10.0]).unwrap();
        assert_eq!(at.1, vec![
            (vec![[10.0, 10.0], [0.0, 10.0], [0.0, 0.0]], false),
            (vec![[10.0, 10.0], [10.0, 0.0]], true),
            (vec![[10.0, 10.0], [10.0, 0.0]], false),
        ]);
    }

    // A figure that is a single point is counted at its junction, not swept.
    #[test]
    fn point_figure_is_counted() {
        let j = sweep(&[&[[3.0, 3.0]]]);
        assert_eq!(j, vec![([3.0, 3.0], vec![], vec![], [1, 0])]);
    }
}
