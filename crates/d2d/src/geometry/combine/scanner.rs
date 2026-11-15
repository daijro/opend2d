// Ported from dotnet/wpf (MIT, Copyright (c) .NET Foundation and Contributors):
// src/Microsoft.DotNet.Wpf/src/WpfGfx/core/geometry/scanner.{cpp,h}. See THIRD-PARTY-NOTICES.

//! Direct2D's region scanner, ported stage by stage (the MIT-licensed WPF `CScanner` it descends from is
//! the structural guide; each stage is checked against d2d1's own build, which differs in places).
//!
//! Stage 1 — monotone chains. Every figure the scanner is sent is cut into chains that are monotone in
//! (y, then x): a chain is a run of vertices whose successive points keep descending, or keep ascending
//! (a *reversed* chain). A turning point closes the current chain and starts the next with a copy of the
//! turning vertex. The sweep runs from the largest (y, x) down: the master heap is a max-heap on each
//! chain's head, its largest vertex, so a chain's points are kept largest-first. Unlike WPF, d2d1 never
//! merges a figure's last chain into its first at the seam, and inserts even a figure's only chain. A
//! hollow figure's chains carry [`HOLLOW`].

/// A grid point: the workspace's integer coordinates, held as doubles as the scanner does.
pub type Pt = [f64; 2];

/// Chain flag: the chain's points were sent ascending, against the (descending) sweep.
pub const REVERSED: u16 = 0x10;
/// Chain flag: the chain belongs to a hollow figure (it bounds no area; Outline keeps it as a stroke).
pub const HOLLOW: u16 = 0x40;
/// Chain flags the next chain of a figure inherits as [`HOLLOW`].
const STROKE_LIKE: u16 = 0x4840;

/// A curve fragment on an edge: the curve's id and its parameters at the edge's two ends, in the order
/// the curve runs.
pub type Cv = (u32, f64, f64);

/// One monotone chain: its vertices in sweep order (head, the largest, first), the curve fragment of
/// each edge between consecutive vertices, and its flags.
#[derive(Debug, Clone, PartialEq)]
pub struct Chain {
    pub pts: Vec<Pt>,
    pub edges: Vec<Option<Cv>>,
    pub flags: u16,
}

impl Chain {
    pub fn head(&self) -> Pt {
        self.pts[0]
    }
}

/// `a` < `b` in (y, then x).
fn ascending(a: Pt, b: Pt) -> bool {
    a[1] < b[1] || (a[1] == b[1] && a[0] < b[0])
}

/// Builds chains from figures as the scanner's sink receives them.
#[derive(Default)]
pub struct ChainList {
    /// The chain being built, in sending order (reversed later for a reversed chain).
    cur: Vec<Pt>,
    cur_edges: Vec<Option<Cv>>,
    cur_flags: u16,
    first: Pt,
    last: Pt,
    filled: bool,
    /// Finished chains in insertion order, and the master heap over them (1-based, as d2d1's).
    chains: Vec<Chain>,
    heap: Vec<usize>,
    /// Flags every new chain starts with (the operand index of a two-shape operation).
    pub base: u16,
    /// Hollow figures are not scanned (ComputeArea); Outline and Combine keep them as stroke chains.
    pub skip_hollow: bool,
}

impl ChainList {
    /// `BeginFigure`: a new chain starting at `pt`; a hollow figure's chains carry [`HOLLOW`].
    pub fn begin_figure(&mut self, pt: Pt, filled: bool) {
        self.first = pt;
        self.last = pt;
        self.filled = filled;
        self.cur = vec![pt];
        self.cur_edges.clear();
        self.cur_flags = self.base | if filled { 0 } else { HOLLOW };
    }

    /// `AddVertices`, one point at a time: a point that keeps the chain's trend extends it; a turning
    /// point (a repeat counts as descending) closes the chain into the heap and starts the next one from
    /// a copy of the last vertex; a repeat of the current point is then dropped.
    pub fn add(&mut self, pt: Pt) {
        self.add_edge(pt, None);
    }

    /// [`Self::add`] for a point reached along a curve fragment `cv` (the edge from the last point).
    pub fn add_edge(&mut self, pt: Pt, cv: Option<Cv>) {
        // The trend test comes first: a repeated point is not ascending, so it closes an ascending
        // chain (a fully rounded rectangle's zero-length side does) before it is dropped.
        let asc = ascending(self.last, pt);
        if self.cur.len() >= 2 && asc != (self.cur_flags & REVERSED != 0) {
            let last = *self.cur.last().unwrap();
            let inherit = self.base | if self.cur_flags & STROKE_LIKE != 0 { HOLLOW } else { 0 };
            self.push_current();
            self.cur = vec![last];
            self.cur_edges.clear();
            self.cur_flags = inherit;
        }
        if pt == self.last {
            return;
        }
        if self.cur.len() == 1 {
            self.cur_flags = if asc { self.cur_flags | REVERSED } else { self.cur_flags & !REVERSED };
        }
        self.cur.push(pt);
        self.cur_edges.push(cv);
        self.last = pt;
    }

    /// `EndFigure`: a filled or closed figure returns to its first point; the last chain enters the heap.
    pub fn end_figure(&mut self, closed: bool) {
        if (self.filled || closed) && self.last != self.first {
            let first = self.first;
            self.add(first);
        }
        self.push_current();
    }

    fn push_current(&mut self) {
        let mut pts = std::mem::take(&mut self.cur);
        let mut edges = std::mem::take(&mut self.cur_edges);
        if self.cur_flags & REVERSED != 0 {
            // A fragment keeps the curve's own parameter order (`SetCurveInfo`); only its edge moves.
            pts.reverse();
            edges.reverse();
        }
        self.chains.push(Chain { pts, edges, flags: self.cur_flags });
        // Sift up on the head (a max-heap on (y, then x)); ties keep the earlier chain above.
        self.heap.push(self.chains.len() - 1);
        let mut i = self.heap.len();
        while i > 1 {
            let (c, p) = (self.chains[self.heap[i - 1]].head(), self.chains[self.heap[i / 2 - 1]].head());
            if !ascending(p, c) {
                break;
            }
            self.heap.swap(i - 1, i / 2 - 1);
            i /= 2;
        }
    }

    /// The chains in insertion order.
    pub fn chains(&self) -> &[Chain] {
        &self.chains
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(figs: &[(&[Pt], bool)]) -> Vec<Chain> {
        let mut l = ChainList::default();
        for (pts, filled) in figs {
            l.begin_figure(pts[0], *filled);
            for &p in &pts[1..] {
                l.add(p);
            }
            l.end_figure(true);
        }
        l.chains().to_vec()
    }

    // A square sent clockwise from its top-left: the top and right run ascending (one reversed chain,
    // kept largest-first), the bottom and left back down (descending) — two chains, both headed at the
    // bottom-right corner.
    #[test]
    fn square_is_two_chains() {
        let c = build(&[(&[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], true)]);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0], Chain { pts: vec![[10.0, 10.0], [10.0, 0.0], [0.0, 0.0]], edges: vec![None; 2], flags: REVERSED });
        assert_eq!(c[1], Chain { pts: vec![[10.0, 10.0], [0.0, 10.0], [0.0, 0.0]], edges: vec![None; 2], flags: 0 });
    }

    // A figure that starts mid-side keeps its seam: d2d1 does not merge its last chain into its first.
    #[test]
    fn seam_is_not_merged() {
        let c = build(&[(&[[10.0, 5.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0], [10.0, 0.0]], true)]);
        assert_eq!(c.len(), 3);
        assert_eq!(c.iter().map(|c| c.flags & REVERSED).collect::<Vec<_>>(), [REVERSED, 0, REVERSED]);
        assert_eq!(c[2].pts, vec![[10.0, 5.0], [10.0, 0.0], [0.0, 0.0]]);
    }

    // A hollow figure's chains are stroke chains; a repeated point adds nothing.
    #[test]
    fn hollow_and_repeats() {
        let c = build(&[(&[[0.0, 0.0], [0.0, 0.0], [5.0, 5.0]], false)]);
        // Closed, the return segment is a second chain, inheriting HOLLOW; open, there is just one.
        assert_eq!(c, vec![
            Chain { pts: vec![[5.0, 5.0], [0.0, 0.0]], edges: vec![None], flags: HOLLOW | REVERSED },
            Chain { pts: vec![[5.0, 5.0], [0.0, 0.0]], edges: vec![None], flags: HOLLOW },
        ]);
        let mut l = ChainList::default();
        l.begin_figure([0.0, 0.0], false);
        l.add([5.0, 5.0]);
        l.end_figure(false);
        assert_eq!(l.chains(), &[Chain { pts: vec![[5.0, 5.0], [0.0, 0.0]], edges: vec![None], flags: HOLLOW | REVERSED }]);
    }
}
