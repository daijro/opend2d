//! Native `d2d1.dll` fill tessellator — the `CScanner` sweep + `CTessellator` band triangulation
//! that `FillGeometry` and aliased `DrawInk` feed to WARP (the explicit triangle VB WARP rasterises,
//! with the gradient texcoord as a per-vertex attribute fitted per triangle). Ported 1:1 from the MIL
//! algorithm (MIT-licensed `dotnet/wpf` `WpfGfx/core/geometry/scanner.cpp`/`tessellate.cpp`):
//! `CTessellator::ProcessTheJunction`/`CreateBands`/`SplitTheBand`/`MergeTheBands`/`ProcessAllTails`/
//! `Connect`/`ProcessAsLeft`/`ProcessAsRight`, driven by the simple-polygon reduction of
//! `CScanner::Scan`/`MoveOn` (vertices in sweep order; a junction per extremum). This is the
//! no-intersection path (one simple, non-self-intersecting contour); self-intersecting / multi-contour
//! fills additionally need the robust `CScanner` intersection split (`ExactArithmetic`/
//! `LineSegmentIntersection`), ported separately.
//!
//! Sweep key (`CScanner::CVertex::ComparePoints`): larger Y first, ties larger X. Ear turn
//! (`IsLeftTurn`): `Determinant(B−A, C−B) > 0`. Emitted triangles drop the zero-area ones native's
//! sink drops and carry the **provoking (last) vertex** native's VB puts last, oriented positive-area
//! (native's VB is all CCW). Validated byte-exact (99/99 incl. provoking vertex) against native's
//! captured `inkbrush-02-linear` triangle stream (from the WARP draw dump).

use super::flatten::Point;
use super::tessellate::Tri;

#[inline]
fn higher(a: Point, b: Point) -> bool {
    // a is processed before b (sweep: larger Y first, ties larger X)
    (a[1], a[0]) > (b[1], b[0])
}
#[inline]
fn det(a: Point, b: Point, c: Point) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
}
#[inline]
fn is_left_turn(a: Point, b: Point, c: Point) -> bool {
    // Determinant(B-A, C-B) > 0
    (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) > 0.0
}

/// A ceiling vertex-ref node (doubly linked), arena-indexed. Nodes are never freed (correctness only).
#[derive(Clone, Copy)]
struct Vr {
    p: Point,
    l: Option<usize>,
    r: Option<usize>,
}

struct Chain {
    v: Vec<Point>,      // head (highest) .. tail (lowest)
    right_side: bool,   // band side (set positionally at the head junction)
    ceil: Option<usize>, // this chain's ceiling end (leftmost if left chain, rightmost if right)
}

struct Tess {
    vr: Vec<Vr>,
    chains: Vec<Chain>,
    active: Vec<usize>, // chain indices, left -> right
    tris: Vec<Tri>,
}

impl Tess {
    fn new_vr(&mut self, p: Point) -> usize {
        self.vr.push(Vr { p, l: None, r: None });
        self.vr.len() - 1
    }
    fn link(&mut self, a: usize, b: Option<usize>) {
        self.vr[a].r = b;
        if let Some(b) = b {
            self.vr[b].l = Some(a);
        }
    }
    fn emit(&mut self, a: Point, b: Point, c: Point) {
        // drop zero-area; orient positive-area keeping provoking vertex `c` last (native's VB order)
        let ar = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
        if ar == 0.0 {
            return;
        }
        if ar < 0.0 {
            self.tris.push([b, a, c]);
        } else {
            self.tris.push([a, b, c]);
        }
    }

    // ProcessAsRight: append `pnext` as the rightmost ceiling vertex, cut ears leftward.
    fn process_as_right(&mut self, ch: usize, pnext: usize) {
        let pt = self.vr[pnext].p;
        let pleft = self.chains[ch].ceil.unwrap();
        self.link(pleft, Some(pnext));
        self.chains[ch].ceil = Some(pnext);
        let mut pl = self.vr[pnext].l;
        while let Some(pli) = pl {
            let pll = self.vr[pli].l;
            match pll {
                Some(plli) if !is_left_turn(self.vr[plli].p, self.vr[pli].p, pt) => {
                    self.emit(pt, self.vr[pli].p, self.vr[plli].p);
                    self.link(plli, Some(pnext));
                    pl = Some(plli);
                }
                _ => break,
            }
        }
    }
    // ProcessAsLeft: prepend `pnext` as the leftmost ceiling vertex, cut ears rightward.
    fn process_as_left(&mut self, ch: usize, pnext: usize) {
        let pt = self.vr[pnext].p;
        let pright = self.chains[ch].ceil.unwrap();
        self.link(pnext, Some(pright));
        self.chains[ch].ceil = Some(pnext);
        let mut pr = self.vr[pnext].r;
        while let Some(pri) = pr {
            let prr = self.vr[pri].r;
            match prr {
                Some(prri) if !is_left_turn(pt, self.vr[pri].p, self.vr[prri].p) => {
                    self.emit(pt, self.vr[pri].p, self.vr[prri].p);
                    self.link(pnext, Some(prri));
                    pr = Some(prri);
                }
                _ => break,
            }
        }
    }
    fn process_vertex(&mut self, ch: usize, p: Point) {
        if self.chains[ch].ceil.is_none() {
            return;
        }
        let vr = self.new_vr(p);
        if self.chains[ch].right_side {
            self.process_as_right(ch, vr);
        } else {
            self.process_as_left(ch, vr);
        }
    }
}

/// Which side of chain `ch` (at `pt`'s sweep height) `pt` lies on: `R` if to the right (larger x).
/// Edge taken is the one straddling `pt.y` (the chain is monotone descending in y).
fn locate(v: &[Point], pt: Point) -> char {
    let y = pt[1];
    let (mut a, mut b) = (v[0], if v.len() > 1 { v[1] } else { v[0] });
    for k in 0..v.len().saturating_sub(1) {
        let (y0, y1) = (v[k][1], v[k + 1][1]);
        if (y0 >= y && y >= y1) || (y1 >= y && y >= y0) {
            a = v[k];
            b = v[k + 1];
            break;
        }
    }
    let d = det(a, b, pt); // edge head->tail (decreasing y); pt to the right (larger x) => det>0
    if d > 0.0 {
        'R'
    } else if d < 0.0 {
        'L'
    } else {
        'I'
    }
}

/// Tessellate one simple (non-self-intersecting) closed polygon `loop_pts` into native's triangle
/// stream (positive-area, provoking vertex last), as `d2d1`'s fill tessellator emits it to WARP.
pub fn tessellate_simple(loop_pts: &[Point]) -> Vec<Tri> {
    let n = loop_pts.len();
    if n < 3 {
        return Vec::new();
    }
    let lp = loop_pts;
    let prev = |i: usize| lp[(i + n - 1) % n];
    let next = |i: usize| lp[(i + 1) % n];
    let is_max = |i: usize| higher(lp[i], prev(i)) && higher(lp[i], next(i));
    let is_min = |i: usize| higher(prev(i), lp[i]) && higher(next(i), lp[i]);

    // Orientation (device, y-down) for the chain reversed flag (unused once sides are positional).
    // Build chains = monotone runs between sweep extrema.
    let i0 = (0..n).find(|&i| is_max(i) || is_min(i));
    let i0 = match i0 {
        Some(i) => i,
        None => return Vec::new(),
    };
    let mut runs: Vec<Vec<usize>> = Vec::new();
    let mut i = i0;
    let mut run = vec![i];
    for _ in 0..n {
        let j = (i + 1) % n;
        run.push(j);
        if is_max(j) || is_min(j) {
            runs.push(std::mem::take(&mut run));
            run = vec![j];
        }
        i = j;
    }
    let mut t = Tess { vr: Vec::new(), chains: Vec::new(), active: Vec::new(), tris: Vec::new() };
    // chain index per (head point, tail point); map points to chains for the driver.
    for r in &runs {
        let (a, b) = (r[0], *r.last().unwrap());
        let fwd = higher(lp[a], lp[b]);
        let idxs: Vec<usize> = if fwd { r.clone() } else { r.iter().rev().cloned().collect() };
        let v: Vec<Point> = idxs.iter().map(|&k| lp[k]).collect();
        t.chains.push(Chain { v, right_side: false, ceil: None });
    }
    let key = |p: Point| (p[0].to_bits(), p[1].to_bits());
    use std::collections::HashMap;
    let mut heads_at: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    let mut tails_at: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    let mut interior_owner: HashMap<(u32, u32), usize> = HashMap::new();
    for (ci, ch) in t.chains.iter().enumerate() {
        heads_at.entry(key(ch.v[0])).or_default().push(ci);
        tails_at.entry(key(*ch.v.last().unwrap())).or_default().push(ci);
        for k in 1..ch.v.len() - 1 {
            interior_owner.insert(key(ch.v[k]), ci);
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    // Total order (NaN-safe): `partial_cmp().unwrap()` panics on a non-finite coordinate (fuzz 1178
    // class). `total_cmp` is a deterministic total order that matches the finite (y,x)-descending sweep
    // order and sorts NaN to a consistent end instead of crashing.
    order.sort_by(|&a, &b| lp[b][1].total_cmp(&lp[a][1]).then_with(|| lp[b][0].total_cmp(&lp[a][0])));

    for &vi in &order {
        let pt = lp[vi];
        if is_max(vi) {
            let mut hs = heads_at.get(&key(pt)).cloned().unwrap_or_default();
            // Robustness (fuzz 1178): a valid local-max vertex has exactly two chains starting here; a
            // non-finite / coincident-vertex degeneracy yields a different count, and the paired side
            // assignment (`hs[0]`/`hs[1]`) would panic. Skip such a vertex — on valid input `hs.len()`
            // is always 2, so this is inert. (The matching min-vertex guard drops the orphaned tails.)
            if hs.len() != 2 {
                continue;
            }
            // InsertHead order: h0 is left of h1 if h1's 2nd vertex is right of h0's first edge.
            if det(pt, t.chains[hs[0]].v[1], t.chains[hs[1]].v[1]) < 0.0 {
                hs.swap(0, 1);
            }
            // Locate position + left/right neighbors in the active list.
            let mut jleft: Option<usize> = None;
            let mut jright: Option<usize> = None;
            let mut ins = 0usize;
            for (k, &c) in t.active.iter().enumerate() {
                if locate(&t.chains[c].v, pt) != 'R' {
                    jright = Some(c);
                    break;
                }
                jleft = Some(c);
                ins = k + 1;
            }
            // Positional sides: inside a band (jleft is a Left chain) => SPLIT; else START.
            let inside = jleft.map(|c| !t.chains[c].right_side).unwrap_or(false);
            if inside {
                t.chains[hs[0]].right_side = true;
                t.chains[hs[1]].right_side = false;
            } else {
                t.chains[hs[0]].right_side = false;
                t.chains[hs[1]].right_side = true;
            }
            t.process_the_junction(&hs, &[], jleft, jright, pt);
            // ActivateBatch: insert heads into the active list between jleft and jright.
            for (off, &h) in hs.iter().enumerate() {
                t.active.insert(ins + off, h);
            }
        } else if is_min(vi) {
            let mut ts = tails_at.get(&key(pt)).cloned().unwrap_or_default();
            ts.sort_by_key(|&c| t.active.iter().position(|&x| x == c).unwrap_or(usize::MAX));
            // Robustness (fuzz 1178): a non-finite / degenerate coordinate can make a vertex classify as
            // a local min with no chain tail registered at its point (or tails not in the active list),
            // breaking the sweep invariant. Skip the junction instead of unwrapping, but still drop the
            // stray tails so the sweep terminates. On valid input every tail is active, so this is inert.
            let (li, ri) = match (
                ts.first().and_then(|&c| t.active.iter().position(|&x| x == c)),
                ts.last().and_then(|&c| t.active.iter().position(|&x| x == c)),
            ) {
                (Some(li), Some(ri)) => (li, ri),
                _ => {
                    t.active.retain(|&c| !ts.contains(&c));
                    continue;
                }
            };
            let jleft = if li > 0 { Some(t.active[li - 1]) } else { None };
            let jright = if ri + 1 < t.active.len() { Some(t.active[ri + 1]) } else { None };
            t.process_the_junction(&[], &ts, jleft, jright, pt);
            t.active.retain(|&c| !ts.contains(&c));
        } else if let Some(&ci) = interior_owner.get(&key(pt)) {
            t.process_vertex(ci, pt);
        }
    }
    t.tris
}

/// Tessellate the nonzero-winding UNION of overlapping closed figures (e.g. `DrawInk`'s nib circles
/// + offset rects) into native's triangle stream. Native fills the overlapping figures through the
/// `CScanner` intersecting sweep, which splits edges at their crossings — the crossing vertex
/// `I = base + λ·(tip − base)`, `λ = det(AC,DC)/det(AB,DC)` (scanner.cpp `EvalIntersectApproxCoordinates`,
/// exact f64 determinants on the 26.6-grid integer workspace). Here we pre-split every pairwise
/// crossing with that exact arithmetic, extract the nonzero-winding boundary (one simple contour for
/// a swept nib), and hand it to [`tessellate_simple`]. Validated byte-exact (f32): reproduces native's
/// captured inkbrush-02 101-vertex outline and its 99 triangles (order + provoking vertex).
pub fn tessellate_ink_union(figures: &[Vec<Point>]) -> Vec<Tri> {
    const G: f64 = 64.0; // 26.6 workspace scale
    // Orient each figure CCW (area2 > 0) in the ×64 integer workspace, dedup consecutive + closing.
    let mut polys: Vec<Vec<(f64, f64)>> = Vec::new();
    for f in figures {
        let mut q: Vec<(f64, f64)> = Vec::new();
        for v in f {
            let p = ((v[0] as f64 * G).round(), (v[1] as f64 * G).round());
            if q.last() != Some(&p) {
                q.push(p);
            }
        }
        if q.len() > 1 && q.first() == q.last() {
            q.pop();
        }
        if q.len() < 3 {
            continue;
        }
        let area2: f64 = (0..q.len()).map(|i| { let a = q[i]; let b = q[(i + 1) % q.len()]; a.0 * b.1 - b.0 * a.1 }).sum();
        if area2 < 0.0 {
            q.reverse();
        }
        polys.push(q);
    }
    // All directed edges.
    let mut edges: Vec<((f64, f64), (f64, f64))> = Vec::new();
    for poly in &polys {
        for i in 0..poly.len() {
            edges.push((poly[i], poly[(i + 1) % poly.len()]));
        }
    }
    let det = |ax: f64, ay: f64, bx: f64, by: f64| ax * by - ay * bx;
    // transverse crossing of AB × CD on the OPEN segments → λ along AB, else None.
    let cross_lambda = |a: (f64, f64), b: (f64, f64), c: (f64, f64), d: (f64, f64)| -> Option<f64> {
        let (xab, yab) = (b.0 - a.0, b.1 - a.1);
        let (xdc, ydc) = (c.0 - d.0, c.1 - d.1);
        let (xac, yac) = (c.0 - a.0, c.1 - a.1);
        let abdc = det(xab, yab, xdc, ydc);
        if abdc == 0.0 {
            return None;
        }
        let lam = det(xac, yac, xdc, ydc) / abdc;
        let mu = det(xab, yab, xac, yac) / abdc;
        if lam > 0.0 && lam < 1.0 && mu > 0.0 && mu < 1.0 {
            Some(lam)
        } else {
            None
        }
    };
    // Pre-split each edge at its crossings (I = base + λ·(tip−base), f64 workspace).
    let mut segs: Vec<((f64, f64), (f64, f64))> = Vec::new();
    for i in 0..edges.len() {
        let (a, b) = edges[i];
        let mut cuts: Vec<(f64, (f64, f64))> = vec![(0.0, a), (1.0, b)];
        for j in 0..edges.len() {
            if i == j {
                continue;
            }
            let (c, d) = edges[j];
            if let Some(lam) = cross_lambda(a, b, c, d) {
                cuts.push((lam, (a.0 + lam * (b.0 - a.0), a.1 + lam * (b.1 - a.1))));
            }
        }
        cuts.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap());
        for w in cuts.windows(2) {
            if w[0].1 != w[1].1 {
                segs.push((w[0].1, w[1].1));
            }
        }
    }
    // Winding number of a workspace point wrt all polys (ray cast).
    let winding = |px: f64, py: f64| -> i32 {
        let mut w = 0;
        for poly in &polys {
            let nn = poly.len();
            for i in 0..nn {
                let (x0, y0) = poly[i];
                let (x1, y1) = poly[(i + 1) % nn];
                if (y0 <= py) != (y1 <= py) {
                    let t = (py - y0) / (y1 - y0);
                    if x0 + t * (x1 - x0) > px {
                        w += if y1 > y0 { 1 } else { -1 };
                    }
                }
            }
        }
        w
    };
    // Boundary segs: the midpoint's two sides straddle the fill (one winding 0, the other nonzero).
    let dev = |p: (f64, f64)| [(p.0 / G) as f32, (p.1 / G) as f32];
    let mut nxt: std::collections::HashMap<(u32, u32), [f32; 2]> = std::collections::HashMap::new();
    // Start points of the boundary segments, in discovery order; `bstarts[0]` is the single-loop start.
    let mut bstarts: Vec<[f32; 2]> = Vec::new();
    let mut count = 0usize;
    for (a, b) in &segs {
        let (mx, my) = ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5);
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let l = (dx * dx + dy * dy).sqrt();
        if l == 0.0 {
            continue;
        }
        let (nx, ny) = (-dy / l * 0.01, dx / l * 0.01);
        if (winding(mx + nx, my + ny) != 0) != (winding(mx - nx, my - ny) != 0) {
            let (da, db) = (dev(*a), dev(*b));
            nxt.insert((da[0].to_bits(), da[1].to_bits()), db);
            bstarts.push(da);
            count += 1;
        }
    }
    if count < 3 {
        return Vec::new();
    }
    // Trace a directed boundary cycle starting at `s`, following each vertex's single successor until
    // it closes back on `s` (or the guard trips). `visited` is shared so cycle enumeration never
    // re-traces a vertex; a chain that revisits a non-start vertex before closing is a broken trace and
    // yields None (skip it) rather than a bogus partial loop.
    let trace_from = |s: [f32; 2], visited: &mut std::collections::HashSet<(u32, u32)>| -> Option<Vec<[f32; 2]>> {
        let mut loop_pts = vec![s];
        visited.insert((s[0].to_bits(), s[1].to_bits()));
        let mut cur = *nxt.get(&(s[0].to_bits(), s[1].to_bits()))?;
        let mut guard = 0;
        while cur != s && guard < count * 2 {
            let ck = (cur[0].to_bits(), cur[1].to_bits());
            if !visited.insert(ck) {
                return None;
            }
            loop_pts.push(cur);
            cur = *nxt.get(&ck)?;
            guard += 1;
        }
        Some(loop_pts)
    };
    let mut visited = std::collections::HashSet::new();
    if nxt.len() == count {
        // No two boundary segments start at the same device point, so `nxt` is a clean partition of the
        // winding boundary into disjoint directed cycles. Native fills this union in one intersecting
        // sweep. Trace EVERY cycle in boundary-discovery order — outer contours AND holes (a hole traces
        // opposite-wound, negative area) — and fill the whole set nonzero-winding through the
        // AA tessellator sweep (`aatess::tessellate_contours`, the same sweep that leaves glyph and
        // realization holes empty). It subtracts the opposite-wound inner contour, so a holed swept nib
        // leaves native's exact empty hole — replacing 22755's outer-only trace that double-filled it,
        // and still drawing a nib that breaks into several disjoint pieces. An ordinary single-contour
        // ink has one outer cycle, whose byte-exact (incl. provoking-vertex) stream is unchanged.
        let mut loops: Vec<Vec<[f64; 2]>> = Vec::new();
        for &s in &bstarts {
            if visited.contains(&(s[0].to_bits(), s[1].to_bits())) {
                continue;
            }
            if let Some(lp) = trace_from(s, &mut visited) {
                loops.push(lp.iter().map(|p| [p[0] as f64, p[1] as f64]).collect());
            }
        }
        let refs: Vec<&[[f64; 2]]> = loops.iter().map(|l| l.as_slice()).collect();
        crate::geometry::aatess::tessellate_contours(&refs)
            .into_iter()
            .map(|t| [[t[0][0] as f32, t[0][1] as f32], [t[1][0] as f32, t[1][1] as f32], [t[2][0] as f32, t[2][1] as f32]])
            .collect()
    } else {
        // A device-point collision (a huge/degenerate union quantized onto the f32 device grid) makes
        // the cycle partition unreliable: trace the single directed boundary from the first boundary
        // segment, following successors until it closes (or the guard trips), and tessellate whatever it
        // collects. This matches native's clipped result for that huge/degenerate case.
        let start = bstarts[0];
        let mut loop_pts = vec![start];
        let mut cur = match nxt.get(&(start[0].to_bits(), start[1].to_bits())) {
            Some(&n) => n,
            None => return Vec::new(),
        };
        let mut guard = 0;
        while cur != start && guard < count * 2 {
            loop_pts.push(cur);
            cur = match nxt.get(&(cur[0].to_bits(), cur[1].to_bits())) {
                Some(&n) => n,
                None => return Vec::new(),
            };
            guard += 1;
        }
        tessellate_simple(&loop_pts)
    }
}

impl Tess {
    fn get_ceiling(&self, ch: usize) -> usize {
        self.chains[ch].ceil.unwrap()
    }
    fn set_ceiling(&mut self, ch: usize, vr: usize) {
        self.chains[ch].ceil = Some(vr);
    }
    fn create_bands(&mut self, from: usize, to: usize, head_pt_src: &[usize]) {
        // pair consecutive (left,right) heads into new bands; ceiling = single head vertex.
        // `from`/`to` are chain indices linked via the junction head order (`head_pt_src`).
        let order = head_pt_src;
        let pos = |c: usize| order.iter().position(|&x| x == c).unwrap();
        let mut i = pos(from);
        let tp = pos(to);
        loop {
            let l = order[i];
            let r = order[i + 1];
            let hv = self.new_vr(self.chains[l].v[0]);
            self.set_ceiling(l, hv);
            self.set_ceiling(r, hv);
            if i + 1 == tp {
                break;
            }
            i += 2;
        }
    }

    fn connect(&mut self, ch: usize, ceiling_vr: usize) {
        let head = self.chains[ch].v[0];
        self.set_ceiling(ch, ceiling_vr);
        let hv = self.new_vr(head);
        if self.chains[ch].right_side {
            self.process_as_right(ch, hv);
        } else {
            self.process_as_left(ch, hv);
        }
    }

    fn lowest_ceiling_from(&self, left_chain: usize) -> usize {
        let mut pc = self.get_ceiling(left_chain);
        while let Some(r) = self.vr[pc].r {
            if higher(self.vr[pc].p, self.vr[r].p) {
                pc = r; // r is Lower than pc (smaller y)
            } else {
                break;
            }
        }
        pc
    }

    fn split_the_band(&mut self, p_lh: usize, p_rh: usize, jleft: usize, jright: Option<usize>) {
        let p_left_ceiling = self.lowest_ceiling_from(jleft);
        // Split the ceiling: right piece = copy of p_left_ceiling, assuming its right links.
        let rp = self.vr[p_left_ceiling].p;
        let right_first = self.new_vr(rp);
        let old_r = self.vr[p_left_ceiling].r;
        self.vr[right_first].r = old_r;
        if let Some(r) = old_r {
            self.vr[r].l = Some(right_first);
        }
        self.vr[right_first].l = None;
        self.vr[p_left_ceiling].r = None;
        if self.vr[right_first].r.is_none() {
            if let Some(jr) = jright {
                self.set_ceiling(jr, right_first);
            }
        }
        self.connect(p_lh, p_left_ceiling);
        self.connect(p_rh, right_first);
    }

    fn merge_the_bands(&mut self, p_lt: usize, p_rt: usize, jleft: Option<usize>) {
        let p_left_ceiling = self.get_ceiling(p_lt);
        let p_right_ceiling = self.get_ceiling(p_rt);
        if let Some(ll) = self.vr[p_left_ceiling].l {
            self.link(ll, Some(p_right_ceiling));
        } else if let Some(jl) = jleft {
            self.set_ceiling(jl, p_right_ceiling);
            self.vr[p_right_ceiling].l = None;
        }
    }

    fn process_all_tails(&mut self, tails: &[usize], pt: Point) {
        for &tc in tails {
            let vr = self.new_vr(pt);
            if self.chains[tc].right_side {
                self.process_as_right(tc, vr);
            } else {
                self.process_as_left(tc, vr);
            }
        }
    }

    fn process_the_junction(
        &mut self,
        heads: &[usize],
        tails: &[usize],
        jleft: Option<usize>,
        jright: Option<usize>,
        pt: Point,
    ) {
        let p_lh = heads.first().copied();
        let p_rh = heads.last().copied();
        let p_lt = tails.first().copied();
        let p_rt = tails.last().copied();
        if p_lh.is_none() && p_lt.is_none() {
            return;
        }
        if !tails.is_empty() {
            self.process_all_tails(tails, pt);
        }
        if let (Some(lh), Some(rh)) = (p_lh, p_rh) {
            let (mut from, mut to): (Option<usize>, Option<usize>) = (None, None);
            if !self.chains[lh].right_side {
                from = Some(lh);
                to = Some(rh);
            } else if let Some(lt) = p_lt {
                self.chains[lh].ceil = self.chains[lt].ceil; // AssumeTask
                self.chains[lt].ceil = None;
                if lh != rh {
                    from = self.head_right_of(lh, heads);
                    to = Some(rh);
                }
            } else {
                let jl = jleft.expect("split needs a left band chain");
                self.split_the_band(lh, rh, jl, jright);
                from = self.head_right_of(lh, heads);
                if from != Some(rh) {
                    to = self.head_left_of(rh, heads);
                }
            }
            if let (Some(f), Some(tt)) = (from, to) {
                self.create_bands(f, tt, heads);
            }
        } else if let Some(lt) = p_lt {
            if self.chains[lt].right_side {
                self.merge_the_bands(lt, p_rt.unwrap(), jleft);
            }
        }
    }
    fn head_right_of(&self, ch: usize, heads: &[usize]) -> Option<usize> {
        let i = heads.iter().position(|&x| x == ch)?;
        heads.get(i + 1).copied()
    }
    fn head_left_of(&self, ch: usize, heads: &[usize]) -> Option<usize> {
        let i = heads.iter().position(|&x| x == ch)?;
        if i == 0 {
            None
        } else {
            heads.get(i - 1).copied()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Native's captured inkbrush-02-linear fill outline (simple polygon, 26.6 grid) and the exact
    // 99-triangle VB stream d2d1 submitted to WARP (captured from the WARP draw dump). The port must
    // reproduce the stream byte-exact incl. the provoking (last) vertex.
    const OUTLINE: &[Point] = &[
    [12.0, 44.0],
    [10.453125, 43.6875],
    [9.171875, 42.828125],
    [8.3125, 41.546875],
    [8.0, 40.0],
    [8.3125, 38.453125],
    [8.723116874694824, 37.8409309387207],
    [8.65625, 37.796875],
    [11.609375, 33.3125],
    [11.672414779663086, 33.354034423828125],
    [12.03718376159668, 32.816829681396484],
    [12.015625, 32.796875],
    [14.890625, 29.703125],
    [14.912185668945312, 29.72307586669922],
    [15.0, 29.59375],
    [15.592700958251953, 29.191299438476562],
    [15.546875, 29.125],
    [18.34375, 27.203125],
    [18.384424209594727, 27.262311935424805],
    [19.046875, 26.8125],
    [19.310997009277344, 26.759675979614258],
    [19.296875, 26.71875],
    [22.0, 25.78125],
    [22.015050888061523, 25.82461166381836],
    [23.065616607666016, 25.612375259399414],
    [23.0625, 25.5625],
    [25.703125, 25.40625],
    [25.70621681213379, 25.455631256103516],
    [25.953125, 25.40625],
    [26.56593894958496, 25.53005027770996],
    [26.578125, 25.453125],
    [29.140625, 25.859375],
    [29.129064559936523, 25.933271408081055],
    [29.79476547241211, 26.0677547454834],
    [29.8125, 26.015625],
    [34.73114776611328, 27.696290969848633],
    [39.04585647583008, 29.049924850463867],
    [40.364871978759766, 29.146652221679688],
    [41.44889831542969, 28.919593811035156],
    [42.66579055786133, 28.199392318725586],
    [44.29107666015625, 26.462017059326172],
    [46.18865203857422, 23.338926315307617],
    [48.328125, 18.40625],
    [48.34037399291992, 18.41156768798828],
    [49.171875, 17.171875],
    [50.453125, 16.3125],
    [52.0, 16.0],
    [53.546875, 16.3125],
    [54.828125, 17.171875],
    [55.6875, 18.453125],
    [56.0, 20.0],
    [55.6875, 21.546875],
    [55.65962600708008, 21.58843231201172],
    [55.671875, 21.59375],
    [53.43927001953125, 26.741140365600586],
    [53.4375, 26.75],
    [53.431766510009766, 26.758441925048828],
    [53.421875, 26.78125],
    [53.41755294799805, 26.779375076293945],
    [53.11205291748047, 27.229293823242188],
    [53.171875, 27.265625],
    [50.90625, 30.984375],
    [50.852996826171875, 30.95183753967285],
    [50.40932083129883, 31.613317489624023],
    [50.421875, 31.625],
    [50.34991455078125, 31.70189094543457],
    [50.328125, 31.734375],
    [50.30498504638672, 31.749895095825195],
    [48.140625, 34.0625],
    [48.12165832519531, 34.04475402832031],
    [48.046875, 34.15625],
    [47.232261657714844, 34.709381103515625],
    [47.265625, 34.765625],
    [44.984375, 36.125],
    [44.94805145263672, 36.06399917602539],
    [44.484375, 36.375],
    [43.75067138671875, 36.523223876953125],
    [43.765625, 36.59375],
    [41.453125, 37.09375],
    [41.43536376953125, 37.009803771972656],
    [40.625, 37.171875],
    [40.34780502319336, 37.115875244140625],
    [40.34375, 37.171875],
    [38.0, 37.0],
    [38.00312423706055, 36.95687484741211],
    [37.10939407348633, 36.77812957763672],
    [37.09375, 36.828125],
    [32.296875, 35.3125],
    [32.3133544921875, 35.26027297973633],
    [32.21699523925781, 35.240806579589844],
    [32.203125, 35.28125],
    [27.511110305786133, 33.68773651123047],
    [25.794422149658203, 33.41722869873047],
    [24.090240478515625, 33.52814865112305],
    [22.43360710144043, 34.099403381347656],
    [20.430788040161133, 35.47212219238281],
    [18.12554931640625, 37.97890853881836],
    [15.34375, 42.203125],
    [15.276883125305176, 42.1590690612793],
    [14.828125, 42.828125],
    [13.546875, 43.6875],
];
    const NATIVE: &[[Point;3]] = &[
    [[12.0,44.0],[10.453125,43.6875],[13.546875,43.6875]],
    [[14.828125,42.828125],[13.546875,43.6875],[10.453125,43.6875]],
    [[10.453125,43.6875],[9.171875,42.828125],[14.828125,42.828125]],
    [[15.276883125305176,42.1590690612793],[14.828125,42.828125],[9.171875,42.828125]],
    [[9.171875,42.828125],[8.3125,41.546875],[15.276883125305176,42.1590690612793]],
    [[8.3125,41.546875],[8.0,40.0],[15.276883125305176,42.1590690612793]],
    [[8.0,40.0],[8.3125,38.453125],[15.276883125305176,42.1590690612793]],
    [[18.12554931640625,37.97890853881836],[15.34375,42.203125],[15.276883125305176,42.1590690612793]],
    [[18.12554931640625,37.97890853881836],[15.276883125305176,42.1590690612793],[8.3125,38.453125]],
    [[8.3125,38.453125],[8.723116874694824,37.8409309387207],[18.12554931640625,37.97890853881836]],
    [[8.723116874694824,37.8409309387207],[8.65625,37.796875],[18.12554931640625,37.97890853881836]],
    [[41.43536376953125,37.009803771972656],[40.625,37.171875],[40.34780502319336,37.115875244140625]],
    [[40.34375,37.171875],[38.0,37.0],[40.34780502319336,37.115875244140625]],
    [[40.34780502319336,37.115875244140625],[38.0,37.0],[41.43536376953125,37.009803771972656]],
    [[38.0,37.0],[38.00312423706055,36.95687484741211],[41.43536376953125,37.009803771972656]],
    [[38.00312423706055,36.95687484741211],[37.10939407348633,36.77812957763672],[41.43536376953125,37.009803771972656]],
    [[43.765625,36.59375],[41.453125,37.09375],[41.43536376953125,37.009803771972656]],
    [[43.765625,36.59375],[41.43536376953125,37.009803771972656],[37.10939407348633,36.77812957763672]],
    [[43.75067138671875,36.523223876953125],[43.765625,36.59375],[37.10939407348633,36.77812957763672]],
    [[44.484375,36.375],[43.75067138671875,36.523223876953125],[37.10939407348633,36.77812957763672]],
    [[44.94805145263672,36.06399917602539],[44.484375,36.375],[37.10939407348633,36.77812957763672]],
    [[20.430788040161133,35.47212219238281],[18.12554931640625,37.97890853881836],[8.65625,37.796875]],
    [[37.09375,36.828125],[32.296875,35.3125],[37.10939407348633,36.77812957763672]],
    [[37.10939407348633,36.77812957763672],[32.296875,35.3125],[44.94805145263672,36.06399917602539]],
    [[32.296875,35.3125],[32.3133544921875,35.26027297973633],[44.94805145263672,36.06399917602539]],
    [[32.3133544921875,35.26027297973633],[32.21699523925781,35.240806579589844],[44.94805145263672,36.06399917602539]],
    [[47.265625,34.765625],[44.984375,36.125],[44.94805145263672,36.06399917602539]],
    [[47.265625,34.765625],[44.94805145263672,36.06399917602539],[32.21699523925781,35.240806579589844]],
    [[47.232261657714844,34.709381103515625],[47.265625,34.765625],[32.21699523925781,35.240806579589844]],
    [[48.046875,34.15625],[47.232261657714844,34.709381103515625],[32.21699523925781,35.240806579589844]],
    [[22.43360710144043,34.099403381347656],[20.430788040161133,35.47212219238281],[8.65625,37.796875]],
    [[48.12165832519531,34.04475402832031],[48.046875,34.15625],[32.21699523925781,35.240806579589844]],
    [[32.203125,35.28125],[27.511110305786133,33.68773651123047],[32.21699523925781,35.240806579589844]],
    [[32.21699523925781,35.240806579589844],[27.511110305786133,33.68773651123047],[48.12165832519531,34.04475402832031]],
    [[24.090240478515625,33.52814865112305],[22.43360710144043,34.099403381347656],[8.65625,37.796875]],
    [[27.511110305786133,33.68773651123047],[25.794422149658203,33.41722869873047],[48.12165832519531,34.04475402832031]],
    [[11.672414779663086,33.354034423828125],[25.794422149658203,33.41722869873047],[24.090240478515625,33.52814865112305]],
    [[11.672414779663086,33.354034423828125],[24.090240478515625,33.52814865112305],[8.65625,37.796875]],
    [[8.65625,37.796875],[11.609375,33.3125],[11.672414779663086,33.354034423828125]],
    [[11.672414779663086,33.354034423828125],[12.03718376159668,32.816829681396484],[25.794422149658203,33.41722869873047]],
    [[25.794422149658203,33.41722869873047],[12.03718376159668,32.816829681396484],[48.12165832519531,34.04475402832031]],
    [[12.03718376159668,32.816829681396484],[12.015625,32.796875],[48.12165832519531,34.04475402832031]],
    [[50.30498504638672,31.749895095825195],[48.140625,34.0625],[48.12165832519531,34.04475402832031]],
    [[50.30498504638672,31.749895095825195],[48.12165832519531,34.04475402832031],[12.015625,32.796875]],
    [[50.328125,31.734375],[50.30498504638672,31.749895095825195],[12.015625,32.796875]],
    [[50.34991455078125,31.70189094543457],[50.328125,31.734375],[12.015625,32.796875]],
    [[50.421875,31.625],[50.34991455078125,31.70189094543457],[12.015625,32.796875]],
    [[50.40932083129883,31.613317489624023],[50.421875,31.625],[12.015625,32.796875]],
    [[50.852996826171875,30.95183753967285],[50.40932083129883,31.613317489624023],[12.015625,32.796875]],
    [[14.912185668945312,29.72307586669922],[50.852996826171875,30.95183753967285],[12.015625,32.796875]],
    [[12.015625,32.796875],[14.890625,29.703125],[14.912185668945312,29.72307586669922]],
    [[14.912185668945312,29.72307586669922],[15.0,29.59375],[50.852996826171875,30.95183753967285]],
    [[15.0,29.59375],[15.592700958251953,29.191299438476562],[50.852996826171875,30.95183753967285]],
    [[15.592700958251953,29.191299438476562],[40.364871978759766,29.146652221679688],[50.852996826171875,30.95183753967285]],
    [[15.592700958251953,29.191299438476562],[15.546875,29.125],[40.364871978759766,29.146652221679688]],
    [[39.04585647583008,29.049924850463867],[40.364871978759766,29.146652221679688],[15.546875,29.125]],
    [[40.364871978759766,29.146652221679688],[41.44889831542969,28.919593811035156],[50.852996826171875,30.95183753967285]],
    [[41.44889831542969,28.919593811035156],[42.66579055786133,28.199392318725586],[50.852996826171875,30.95183753967285]],
    [[34.73114776611328,27.696290969848633],[39.04585647583008,29.049924850463867],[15.546875,29.125]],
    [[53.171875,27.265625],[50.90625,30.984375],[50.852996826171875,30.95183753967285]],
    [[53.171875,27.265625],[50.852996826171875,30.95183753967285],[42.66579055786133,28.199392318725586]],
    [[18.384424209594727,27.262311935424805],[34.73114776611328,27.696290969848633],[15.546875,29.125]],
    [[53.11205291748047,27.229293823242188],[53.171875,27.265625],[42.66579055786133,28.199392318725586]],
    [[15.546875,29.125],[18.34375,27.203125],[18.384424209594727,27.262311935424805]],
    [[18.384424209594727,27.262311935424805],[19.046875,26.8125],[34.73114776611328,27.696290969848633]],
    [[53.41755294799805,26.779375076293945],[53.11205291748047,27.229293823242188],[42.66579055786133,28.199392318725586]],
    [[19.046875,26.8125],[19.310997009277344,26.759675979614258],[34.73114776611328,27.696290969848633]],
    [[53.431766510009766,26.758441925048828],[53.421875,26.78125],[53.41755294799805,26.779375076293945]],
    [[53.431766510009766,26.758441925048828],[53.41755294799805,26.779375076293945],[42.66579055786133,28.199392318725586]],
    [[53.4375,26.75],[53.431766510009766,26.758441925048828],[42.66579055786133,28.199392318725586]],
    [[53.43927001953125,26.741140365600586],[53.4375,26.75],[42.66579055786133,28.199392318725586]],
    [[19.310997009277344,26.759675979614258],[19.296875,26.71875],[34.73114776611328,27.696290969848633]],
    [[42.66579055786133,28.199392318725586],[44.29107666015625,26.462017059326172],[53.43927001953125,26.741140365600586]],
    [[19.296875,26.71875],[29.79476547241211,26.0677547454834],[34.73114776611328,27.696290969848633]],
    [[29.79476547241211,26.0677547454834],[29.8125,26.015625],[34.73114776611328,27.696290969848633]],
    [[29.129064559936523,25.933271408081055],[29.79476547241211,26.0677547454834],[19.296875,26.71875]],
    [[29.140625,25.859375],[29.129064559936523,25.933271408081055],[19.296875,26.71875]],
    [[22.015050888061523,25.82461166381836],[29.140625,25.859375],[19.296875,26.71875]],
    [[19.296875,26.71875],[22.0,25.78125],[22.015050888061523,25.82461166381836]],
    [[22.015050888061523,25.82461166381836],[23.065616607666016,25.612375259399414],[29.140625,25.859375]],
    [[23.065616607666016,25.612375259399414],[23.0625,25.5625],[29.140625,25.859375]],
    [[23.0625,25.5625],[26.56593894958496,25.53005027770996],[29.140625,25.859375]],
    [[25.70621681213379,25.455631256103516],[26.56593894958496,25.53005027770996],[23.0625,25.5625]],
    [[26.56593894958496,25.53005027770996],[26.578125,25.453125],[29.140625,25.859375]],
    [[25.70621681213379,25.455631256103516],[25.953125,25.40625],[26.56593894958496,25.53005027770996]],
    [[23.0625,25.5625],[25.703125,25.40625],[25.70621681213379,25.455631256103516]],
    [[44.29107666015625,26.462017059326172],[46.18865203857422,23.338926315307617],[53.43927001953125,26.741140365600586]],
    [[55.671875,21.59375],[53.43927001953125,26.741140365600586],[46.18865203857422,23.338926315307617]],
    [[55.65962600708008,21.58843231201172],[55.671875,21.59375],[46.18865203857422,23.338926315307617]],
    [[55.6875,21.546875],[55.65962600708008,21.58843231201172],[46.18865203857422,23.338926315307617]],
    [[56.0,20.0],[55.6875,21.546875],[46.18865203857422,23.338926315307617]],
    [[55.6875,18.453125],[56.0,20.0],[46.18865203857422,23.338926315307617]],
    [[48.34037399291992,18.41156768798828],[55.6875,18.453125],[46.18865203857422,23.338926315307617]],
    [[46.18865203857422,23.338926315307617],[48.328125,18.40625],[48.34037399291992,18.41156768798828]],
    [[54.828125,17.171875],[55.6875,18.453125],[48.34037399291992,18.41156768798828]],
    [[48.34037399291992,18.41156768798828],[49.171875,17.171875],[54.828125,17.171875]],
    [[53.546875,16.3125],[54.828125,17.171875],[49.171875,17.171875]],
    [[49.171875,17.171875],[50.453125,16.3125],[53.546875,16.3125]],
    [[50.453125,16.3125],[52.0,16.0],[53.546875,16.3125]],
];
    #[test]
    fn reproduces_native_ink_triangles() {
        let got = tessellate_simple(OUTLINE);
        assert_eq!(got.len(), NATIVE.len(), "triangle count");
        for (i, (g, n)) in got.iter().zip(NATIVE.iter()).enumerate() {
            assert_eq!(g, n, "triangle {i} ours={g:?} native={n:?}");
        }
    }

    // Robustness (fuzz 1178): a non-finite coordinate breaks the sweep's vertex classification so a
    // min-vertex can have no registered tails; the driver must not panic (unwrap on empty/absent).
    #[test]
    fn nonfinite_coords_do_not_panic() {
        let inf = f32::INFINITY;
        let n = f32::NAN;
        let cases: &[&[Point]] = &[
            &[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [inf, 5.0]],
            &[[0.0, 0.0], [10.0, 0.0], [5.0, inf]],
            &[[inf, inf], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            &[[0.0, 0.0], [inf, 0.0], [10.0, 10.0], [0.0, inf]],
            &[[1e30, 1e30], [-1e30, 0.0], [0.0, -1e30]],
            &[[0.0, 0.0], [10.0, 0.0], [5.0, n], [10.0, 10.0], [0.0, 10.0]],
            // coincident / self-touching / collinear degeneracies
            &[[5.0, 0.0], [10.0, 10.0], [0.0, 10.0], [5.0, 0.0], [5.0, 0.0]],
            &[[0.0, 0.0], [10.0, 10.0], [0.0, 0.0], [10.0, 0.0], [0.0, 10.0]],
            &[[0.0, 5.0], [10.0, 5.0], [5.0, 0.0], [5.0, 10.0]], // figure-8 cross
            &[[0.0, 0.0], [10.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            &[[5.0, inf], [10.0, 0.0], [0.0, 0.0], [5.0, inf], [5.0, 10.0]],
            &[[inf, 0.0], [10.0, 0.0], [10.0, 10.0], [inf, 0.0]],
        ];
        for (i, c) in cases.iter().enumerate() {
            let _ = tessellate_simple(c); // must return, not panic
            let _ = tessellate_ink_union(&[c.to_vec()]);
            eprintln!("case {i}: ok");
        }
    }
}
