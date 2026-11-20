//! Native single-sweep AA tessellation of a simple polygon — the `CScanner` +
//! no-intersection tessellator path (scanner.cpp / tessellate.cpp), ported
//! line-for-line from the raster agent's `tess_validated_99of99.py` (validated against native's 99
//! inkbrush-02 triangles). Unlike [`super::tessellate`] (the public `Tessellate`, monotone only),
//! this does the full band split/merge at junctions, so fringe and interior come out as ONE
//! tessellation with no seam — the fix for large-glyph outline holes where the old two-pass feather
//! (`realization::winding_interior` + `feather_figure`) left pixels on the shared boundary uncovered.
//!
//! Sweep key: larger y first, ties larger x (`CScanner::CVertex::IsHigherThan`). Ceiling vertex refs
//! (`CVertexRef`, task data) are a doubly-linked list, held here in an arena (`vrs`) with `Option`
//! index links; each chain's active-list / head links mirror the Python `chain.L/.R` in `clinks`.
//! Emits positive-area triangles with the provoking vertex last, dropping zero-area.
//!
//! Robustness: the port is exact for a clean simple polygon (the validated case). Adversarial glyph
//! outlines (degenerate/self-touching/non-finite) can drive the sweep into states the algorithm does
//! not expect; rather than panic (the harness builds `panic = "abort"`), every sweep-state access that
//! could be absent sets `failed` and bails, and `tessellate_loop` returns an empty Vec. The caller
//! then falls back to the robust `winding_interior`. Non-anomalous (clean) input never trips a guard,
//! so the 99/99 self-test is unaffected.

type P = [f64; 2];

#[inline]
fn gt(a: P, b: P) -> bool {
    (a[1], a[0]) > (b[1], b[0])
}
#[inline]
fn det(a: P, b: P, c: P) -> f64 {
    (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
}
/// The corner-cutting stop test: the ceiling is not carved at a left turn OR a straight run (`det ≥ 0`),
/// so collinear ceiling vertices (a horizontal edge's interior points) stay in the ceiling and each
/// gets its own triangle from the next vertex below, as the AA tessellator does.
#[inline]
fn is_left_turn(a: P, b: P, c: P) -> bool {
    det(a, b, c) >= 0.0
}

#[derive(Clone, Copy)]
struct Vr {
    p: P,
    l: Option<usize>,
    r: Option<usize>,
}

struct Chain {
    v: Vec<P>,
    cur: usize,
    right_side: bool,
    ceil: Option<usize>, // VR arena index
}
impl Chain {
    fn head(&self) -> P {
        self.v[0]
    }
    /// `LocateVertex`: which side of this (monotone-descending) chain `pt` is on, at pt.y.
    fn locate(&self, pt: P) -> char {
        let y = pt[1];
        let (mut a, mut b) = (self.v[0], if self.v.len() > 1 { self.v[1] } else { self.v[0] });
        for k in 0..self.v.len() - 1 {
            let (y0, y1) = (self.v[k][1], self.v[k + 1][1]);
            if (y0 >= y && y >= y1) || (y1 >= y && y >= y0) {
                a = self.v[k];
                b = self.v[k + 1];
                break;
            }
        }
        match det(a, b, pt) {
            d if d > 0.0 => 'R',
            d if d < 0.0 => 'L',
            _ => 'I',
        }
    }
}

#[derive(Default)]
struct Junction {
    heads: Vec<usize>,
    tails: Vec<usize>,
    left: Option<usize>,
    right: Option<usize>,
    point: P,
}

struct Tess {
    vrs: Vec<Vr>,
    chains: Vec<Chain>,
    clinks: Vec<(Option<usize>, Option<usize>)>, // per-chain (L, R)
    active: Vec<usize>,
    tris: Vec<[P; 3]>,
    jct: Junction,
    failed: bool,
}

/// `let x = guard!(self, option);` — unwrap `option`, or mark the sweep failed and `return`.
macro_rules! guard {
    ($self:ident, $opt:expr) => {
        match $opt {
            Some(x) => x,
            None => {
                $self.failed = true;
                return;
            }
        }
    };
}

impl Tess {
    fn new_vr(&mut self, p: P) -> usize {
        self.vrs.push(Vr { p, l: None, r: None });
        self.vrs.len() - 1
    }
    fn vlink(&mut self, a: Option<usize>, b: Option<usize>) {
        if let Some(a) = a {
            self.vrs[a].r = b;
        }
        if let Some(b) = b {
            self.vrs[b].l = a;
        }
    }
    fn cl(&self, ch: usize) -> Option<usize> {
        self.clinks[ch].0
    }
    fn cr(&self, ch: usize) -> Option<usize> {
        self.clinks[ch].1
    }
    fn emit(&mut self, a: P, b: P, c: P) {
        let ar = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
        if ar == 0.0 {
            return;
        }
        let (a, b) = if ar < 0.0 { (b, a) } else { (a, b) };
        self.tris.push([a, b, c]);
    }

    fn process_as_right(&mut self, ch: usize, pnext: usize) {
        if self.failed {
            return;
        }
        let pt = self.vrs[pnext].p;
        let pleft = guard!(self, self.chains[ch].ceil);
        self.vlink(Some(pleft), Some(pnext));
        self.chains[ch].ceil = Some(pnext);
        let mut pl = self.vrs[pnext].l;
        while let Some(pli) = pl {
            let pll = self.vrs[pli].l;
            match pll {
                None => break,
                Some(plli) if is_left_turn(self.vrs[plli].p, self.vrs[pli].p, pt) => break,
                Some(plli) => {
                    let (pnp, plp, pllp) = (self.vrs[pnext].p, self.vrs[pli].p, self.vrs[plli].p);
                    self.emit(pnp, plp, pllp);
                    self.vlink(Some(plli), Some(pnext));
                    pl = Some(plli);
                }
            }
        }
    }
    fn process_as_left(&mut self, ch: usize, pnext: usize) {
        if self.failed {
            return;
        }
        let pt = self.vrs[pnext].p;
        let pright = guard!(self, self.chains[ch].ceil);
        self.vlink(Some(pnext), Some(pright));
        self.chains[ch].ceil = Some(pnext);
        let mut pr = self.vrs[pnext].r;
        while let Some(pri) = pr {
            let prr = self.vrs[pri].r;
            match prr {
                None => break,
                Some(prri) if is_left_turn(pt, self.vrs[pri].p, self.vrs[prri].p) => break,
                Some(prri) => {
                    let (pnp, prp, prrp) = (self.vrs[pnext].p, self.vrs[pri].p, self.vrs[prri].p);
                    self.emit(pnp, prp, prrp);
                    self.vlink(Some(pnext), Some(prri));
                    pr = Some(prri);
                }
            }
        }
    }

    fn process_current_vertex(&mut self, ch: usize) {
        if self.failed || self.chains[ch].ceil.is_none() {
            return;
        }
        let p = self.chains[ch].v[self.chains[ch].cur];
        let pvr = self.new_vr(p);
        if self.chains[ch].right_side {
            self.process_as_right(ch, pvr);
        } else {
            self.process_as_left(ch, pvr);
        }
    }

    /// `CreateBands`: pair the junction's heads from `first` (a left chain) to `last` into bands; a head
    /// left over once the pairs run out takes over the junction's rightmost tail.
    fn create_bands(&mut self, first: usize, last: usize) {
        let mut left = first;
        let mut odd = true;
        while left != last {
            if self.chains[left].right_side {
                self.failed = true;
                return;
            }
            let right = guard!(self, self.cr(left));
            if !self.chains[right].right_side {
                self.failed = true;
                return;
            }
            let pvr = self.new_vr(self.chains[left].head());
            self.chains[left].ceil = Some(pvr);
            self.chains[right].ceil = Some(pvr);
            if right == last {
                odd = false;
                break;
            }
            left = guard!(self, self.cr(right));
        }
        if odd {
            let Some(&tail) = self.jct.tails.last() else {
                self.failed = true;
                return;
            };
            self.chains[left].ceil = self.chains[tail].ceil;
            self.chains[tail].ceil = None;
        }
    }

    fn process_all_tails(&mut self, pt: P) {
        let tails = self.jct.tails.clone();
        for t in tails {
            if self.failed {
                return;
            }
            let pvr = self.new_vr(pt);
            if self.chains[t].right_side {
                self.process_as_right(t, pvr);
            } else {
                self.process_as_left(t, pvr);
            }
        }
    }

    /// Walk right from the band's ceiling while the next ref is lower (larger y) — the lowest ceiling.
    fn lowest_ceiling_from(&self, left_chain: usize) -> Option<usize> {
        let mut pc = self.chains[left_chain].ceil?;
        while let Some(r) = self.vrs[pc].r {
            if gt(self.vrs[pc].p, self.vrs[r].p) {
                pc = r;
            } else {
                break;
            }
        }
        Some(pc)
    }

    fn connect(&mut self, ch: usize, ceiling_vr: usize) {
        if self.failed {
            return;
        }
        let phead = self.new_vr(self.chains[ch].head());
        self.chains[ch].ceil = Some(ceiling_vr);
        if self.chains[ch].right_side {
            self.process_as_right(ch, phead);
        } else {
            self.process_as_left(ch, phead);
        }
    }

    fn split_the_band(&mut self, p_lh: usize, p_rh: usize) {
        let p_left = guard!(self, self.jct.left);
        let p_left_ceiling = guard!(self, self.lowest_ceiling_from(p_left));
        // right piece = copy of pLeftCeiling + its right links
        let p_right_ceiling = self.new_vr(self.vrs[p_left_ceiling].p);
        let old_r = self.vrs[p_left_ceiling].r;
        self.vrs[p_right_ceiling].r = old_r;
        if let Some(or) = old_r {
            self.vrs[or].l = Some(p_right_ceiling);
        }
        self.vrs[p_right_ceiling].l = None;
        self.vrs[p_left_ceiling].r = None;
        if self.vrs[p_right_ceiling].r.is_none() {
            let right = guard!(self, self.jct.right);
            self.chains[right].ceil = Some(p_right_ceiling);
        }
        self.connect(p_lh, p_left_ceiling);
        self.connect(p_rh, p_right_ceiling);
    }

    fn merge_the_bands(&mut self, p_lt: usize, p_rt: usize) {
        let p_left_ceiling = guard!(self, self.chains[p_lt].ceil);
        let p_right_ceiling = guard!(self, self.chains[p_rt].ceil);
        if let Some(lc_l) = self.vrs[p_left_ceiling].l {
            self.vlink(Some(lc_l), Some(p_right_ceiling));
        } else if let Some(p) = self.jct.left {
            self.chains[p].ceil = Some(p_right_ceiling);
            self.vrs[p_right_ceiling].l = None;
        }
    }

    fn process_the_junction(&mut self) {
        if self.failed {
            return;
        }
        let p_lh = self.jct.heads.first().copied();
        let p_rh = self.jct.heads.last().copied();
        let p_lt = self.jct.tails.first().copied();
        let p_rt = self.jct.tails.last().copied();
        if p_lh.is_none() && p_lt.is_none() {
            return;
        }
        let pt = self.jct.point;
        if p_lt.is_some() {
            self.process_all_tails(pt);
        }
        if let Some(p_lh) = p_lh {
            let p_rh = guard!(self, p_rh);
            let (mut p_from, mut p_to): (Option<usize>, Option<usize>) = (None, None);
            if !self.chains[p_lh].right_side {
                p_from = Some(p_lh);
                p_to = Some(p_rh);
            } else if let Some(p_lt) = p_lt {
                // AssumeTask: the head takes the tail's ceiling
                self.chains[p_lh].ceil = self.chains[p_lt].ceil;
                self.chains[p_lt].ceil = None;
                if p_lh != p_rh {
                    p_from = self.cr(p_lh);
                    p_to = Some(p_rh);
                }
            } else {
                self.split_the_band(p_lh, p_rh);
                p_from = self.cr(p_lh);
                if p_from != Some(p_rh) {
                    p_to = self.cl(p_rh);
                }
            }
            if let (Some(f), Some(t)) = (p_from, p_to) {
                self.create_bands(f, t);
            }
        } else if self.chains[guard!(self, p_lt)].right_side {
            self.merge_the_bands(guard!(self, p_lt), guard!(self, p_rt));
        }
    }
}

/// Positional head classification (equiv. to the winding `Classify` for nonzero fill): inside a band
/// (`jct.left` is a Left chain) ⇒ SPLIT (leftmost head is a Right chain); else START.
fn classify_heads(t: &mut Tess) {
    if t.jct.heads.is_empty() {
        t.failed = true;
        return;
    }
    let inside = t.jct.left.is_some_and(|l| !t.chains[l].right_side);
    let (h0, hlast) = (t.jct.heads[0], *t.jct.heads.last().unwrap());
    if inside {
        t.chains[h0].right_side = true;
        t.chains[hlast].right_side = false;
    } else {
        t.chains[h0].right_side = false;
        t.chains[hlast].right_side = true;
    }
}

fn relink(t: &mut Tess) {
    for k in 0..t.active.len() {
        let c = t.active[k];
        t.clinks[c].0 = if k > 0 { Some(t.active[k - 1]) } else { None };
        t.clinks[c].1 = if k + 1 < t.active.len() { Some(t.active[k + 1]) } else { None };
    }
}

/// Sweep order comparator: descending y, ties descending x; `total_cmp` so non-finite coords sort
/// deterministically instead of panicking (the finite pre-check already rejects them).
fn sweep_cmp(a: P, b: P) -> std::cmp::Ordering {
    b[1].total_cmp(&a[1]).then(b[0].total_cmp(&a[0]))
}

/// Tessellate one simple closed polygon `loop_` (no self-intersections) into AA tessellator triangles.
/// Returns an empty Vec on a non-finite or anomalous input (the caller falls back to `winding_interior`).
pub fn tessellate_loop(loop_: &[P]) -> Vec<[P; 3]> {
    tessellate_contours(&[loop_])
}

/// Tessellate one or more closed contours through the AA tessellator sweep. Multiple contours are
/// filled nonzero-winding (a counter is an opposite-wound inner contour, left empty), so glyphs with
/// holes (`O`, `A`, `B`, …) get the same single-sweep interior. Each contour's chains take that
/// contour's own winding for their fill side; the band split/merge at junctions subtracts the counter.
/// Empty on non-finite or anomalous input (the caller falls back to `winding_interior`).
pub fn tessellate_contours(contours: &[&[P]]) -> Vec<[P; 3]> {
    tessellate(contours, false)
}

/// [`tessellate_contours`] for a resolved region's contours (non-overlapping, every one wound with the
/// region on its right, a hole against its outer): a chain's side follows from its direction alone, as
/// the scanner's winding gives native's chains, so a hole's chains bound the fill instead of claiming
/// the hole's inside.
pub fn tessellate_region(contours: &[&[P]]) -> Vec<[P; 3]> {
    tessellate(contours, true)
}

fn tessellate(contours: &[&[P]], region: bool) -> Vec<[P; 3]> {
    let mut v: Vec<P> = Vec::new();
    let mut cof: Vec<usize> = vec![0]; // per-contour start offsets into `v`
    for c in contours {
        v.extend_from_slice(c);
        cof.push(v.len());
    }
    let n = v.len();
    if n < 3 || !v.iter().all(|p| p[0].is_finite() && p[1].is_finite()) {
        return Vec::new();
    }
    let mut cid = vec![0usize; n]; // contour id per global vertex index
    for c in 0..contours.len() {
        for gi in cof[c]..cof[c + 1] {
            cid[gi] = c;
        }
    }
    // Contour-local previous/next (wrap within the vertex's own contour).
    let prev_gi = |gi: usize| { let c = cid[gi]; let l = cof[c + 1] - cof[c]; cof[c] + (gi - cof[c] + l - 1) % l };
    let next_gi = |gi: usize| { let c = cid[gi]; let l = cof[c + 1] - cof[c]; cof[c] + (gi - cof[c] + 1) % l };
    let is_max = |gi: usize| gt(v[gi], v[prev_gi(gi)]) && gt(v[gi], v[next_gi(gi)]);
    let is_min = |gi: usize| gt(v[prev_gi(gi)], v[gi]) && gt(v[next_gi(gi)], v[gi]);

    let mut t = Tess {
        vrs: Vec::new(),
        chains: Vec::new(),
        clinks: Vec::new(),
        active: Vec::new(),
        tris: Vec::new(),
        jct: Junction::default(),
        failed: false,
    };
    // A point two contour vertices share (loops the region scan resolved touching at a crossing, or a
    // contour revisiting a vertex) is a junction of its own: every chain through it ends there and the
    // next starts there, so each contour's chain is carried through it.
    // `+ 0.0` keys -0 as 0, as `==` does.
    let key = |p: P| [(p[0] + 0.0).to_bits(), (p[1] + 0.0).to_bits()];
    let mut occ: std::collections::HashMap<[u64; 2], usize> = std::collections::HashMap::new();
    for p in &v {
        *occ.entry(key(*p)).or_default() += 1;
    }
    let shared = |gi: usize| occ.get(&key(v[gi])).is_some_and(|&n| n > 1);
    let split = |gi: usize| is_max(gi) || is_min(gi) || shared(gi);
    let signed = |c: usize| -> f64 {
        let l = cof[c + 1] - cof[c];
        (0..l).map(|li| { let a = v[cof[c] + li]; let b = v[cof[c] + (li + 1) % l]; a[0] * b[1] - b[0] * a[1] }).sum()
    };
    let region_ccw = (0..contours.len()).map(signed).sum::<f64>() > 0.0;
    // Build monotone chains per contour (split at sweep extrema and shared points), head-high..tail-low.
    for c in 0..contours.len() {
        let l = cof[c + 1] - cof[c];
        if l < 3 {
            continue;
        }
        let gi = |li: usize| cof[c] + li;
        let i0 = (0..l).find(|&li| split(gi(li))).unwrap_or(0);
        let mut seq: Vec<Vec<usize>> = Vec::new();
        let mut run = vec![gi(i0)];
        let mut li = i0;
        for _ in 0..l {
            let lj = (li + 1) % l;
            run.push(gi(lj));
            if split(gi(lj)) {
                seq.push(std::mem::replace(&mut run, vec![gi(lj)]));
            }
            li = lj;
        }
        let ccw = if region { region_ccw } else { signed(c) > 0.0 };
        for run in &seq {
            let (a, b) = (run[0], *run.last().unwrap());
            let fwd = gt(v[a], v[b]); // run[0] is the head (higher)
            let idxs: Vec<usize> = if fwd { run.clone() } else { run.iter().rev().copied().collect() };
            let pts: Vec<P> = idxs.iter().map(|&k| v[k]).collect();
            let rev = !fwd; // figure direction goes UP in sweep
            let right_side = if ccw { rev } else { !rev };
            t.chains.push(Chain { v: pts, cur: 0, right_side, ceil: None });
            t.clinks.push((None, None));
        }
    }

    // Driver: process vertices in sweep order.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| sweep_cmp(v[a], v[b]));
    let eqp = |a: P, b: P| a[0] == b[0] && a[1] == b[1];
    // The chains each point heads, ends, or passes through (a chain's points never change), in chain
    // order, so a vertex finds its chains without visiting them all.
    let mut heads: std::collections::HashMap<[u64; 2], Vec<usize>> = std::collections::HashMap::new();
    let mut tails: std::collections::HashMap<[u64; 2], Vec<usize>> = std::collections::HashMap::new();
    let mut inner: std::collections::HashMap<[u64; 2], (usize, usize)> = std::collections::HashMap::new();
    for (c, ch) in t.chains.iter().enumerate() {
        heads.entry(key(ch.head())).or_default().push(c);
        if let Some(&q) = ch.v.last() {
            tails.entry(key(q)).or_default().push(c);
        }
        for k in 1..ch.v.len().saturating_sub(1) {
            inner.entry(key(ch.v[k])).or_insert_with(|| (c, ch.v.iter().position(|&q| eqp(q, ch.v[k])).unwrap_or(k)));
        }
    }
    // Heads left to right: by the direction of each chain's first segment.
    let order_heads = |t: &Tess, pt: P, hs: &mut Vec<usize>| {
        hs.sort_by(|&a, &b| {
            let (ca, cb) = (&t.chains[a].v, &t.chains[b].v);
            if ca.len() > 1 && cb.len() > 1 && det(pt, ca[1], cb[1]) < 0.0 { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Less }
        });
    };
    let mut gi_at = 0;
    while gi_at < order.len() {
        if t.failed {
            return Vec::new();
        }
        let vi = order[gi_at];
        let pt = v[vi];
        let mut end = gi_at + 1;
        while end < order.len() && eqp(v[order[end]], pt) {
            end += 1;
        }
        let group = end - gi_at;
        gi_at = end;
        if group == 1 && !is_max(vi) && !is_min(vi) {
            if let Some(&(ch, pos)) = inner.get(&key(pt)) {
                t.chains[ch].cur = pos;
                t.process_current_vertex(ch);
            }
            continue;
        }
        let mut hs: Vec<usize> = heads.get(&key(pt)).cloned().unwrap_or_default();
        let mut ts: Vec<usize> = tails.get(&key(pt)).cloned().unwrap_or_default();
        order_heads(&t, pt, &mut hs);
        if ts.is_empty() {
            for k in 0..hs.len() {
                t.clinks[hs[k]].0 = if k > 0 { Some(hs[k - 1]) } else { None };
                t.clinks[hs[k]].1 = if k + 1 < hs.len() { Some(hs[k + 1]) } else { None };
            }
            let (mut pos_l, mut pos_r): (Option<usize>, Option<usize>) = (None, None);
            for &c in &t.active {
                if t.chains[c].locate(pt) != 'R' {
                    pos_r = Some(c);
                    break;
                }
                pos_l = Some(c);
            }
            t.jct = Junction { heads: hs.clone(), tails: Vec::new(), left: pos_l, right: pos_r, point: pt };
            classify_heads(&mut t);
            t.process_the_junction();
            let idx = match pos_l {
                Some(l) => match t.active.iter().position(|&c| c == l) {
                    Some(p) => p + 1,
                    None => return Vec::new(),
                },
                None => 0,
            };
            for (off, &h) in hs.iter().enumerate() {
                t.active.insert(idx + off, h);
            }
            relink(&mut t);
        } else {
            // every tail must be live in the active list; otherwise the sweep state is anomalous (a
            // degenerate tiny outline: r25 30923, Leelawadee UI Semilight at em 0.81).
            if ts.iter().any(|&c| !t.active.contains(&c)) {
                return Vec::new();
            }
            ts.sort_by_key(|&c| t.active.iter().position(|&x| x == c).unwrap());
            let li = t.active.iter().position(|&c| c == ts[0]).unwrap();
            let ri = t.active.iter().position(|&c| c == *ts.last().unwrap()).unwrap();
            let left = if li > 0 { Some(t.active[li - 1]) } else { None };
            let right = if ri + 1 < t.active.len() { Some(t.active[ri + 1]) } else { None };
            // A junction that ends chains and starts others continues them: the heads keep the sides
            // their contours' windings gave them.
            for k in 0..hs.len() {
                t.clinks[hs[k]].0 = if k > 0 { Some(hs[k - 1]) } else { left };
                t.clinks[hs[k]].1 = if k + 1 < hs.len() { Some(hs[k + 1]) } else { right };
            }
            t.jct = Junction { heads: hs.clone(), tails: ts.clone(), left, right, point: pt };
            t.process_the_junction();
            t.active.retain(|c| !ts.contains(c));
            for (off, &h) in hs.iter().enumerate() {
                t.active.insert(li + off, h);
            }
            relink(&mut t);
        }
    }
    if t.failed {
        return Vec::new();
    }
    t.tris
}

#[cfg(test)]
mod tests {
    use super::*;
    include!("aatess_testdata.rs");

    // Canonical rotation of a triangle's vertex-set, for order-independent comparison.
    fn canon(t: &[P; 3]) -> [[u64; 2]; 3] {
        let mut v: Vec<[u64; 2]> = t.iter().map(|p| [p[0].to_bits(), p[1].to_bits()]).collect();
        v.sort();
        [v[0], v[1], v[2]]
    }

    #[test]
    fn ib2_matches_native_99() {
        let loop_: Vec<P> = IB2_LOOP.iter().map(|&p| [p[0], p[1]]).collect();
        let got = tessellate_loop(&loop_);
        use std::collections::HashMap;
        let mut want: HashMap<[[u64; 2]; 3], i32> = HashMap::new();
        for t in IB2_TRIS.iter() {
            *want.entry(canon(t)).or_default() += 1;
        }
        let mut have: HashMap<[[u64; 2]; 3], i32> = HashMap::new();
        for t in &got {
            *have.entry(canon(t)).or_default() += 1;
        }
        let matching: i32 = want.iter().map(|(k, &c)| c.min(*have.get(k).copied().get_or_insert(0))).sum();
        eprintln!("emitted {} native {} matching(vset) {}", got.len(), IB2_TRIS.len(), matching);
        assert_eq!(got.len(), IB2_TRIS.len(), "triangle count");
        assert_eq!(matching, IB2_TRIS.len() as i32, "all native triangles reproduced");
    }
}
