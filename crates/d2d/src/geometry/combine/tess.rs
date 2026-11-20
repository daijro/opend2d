//! `CTessellator`, the scanner module behind `ID2D1Geometry::Tessellate`: the region the fill rule leaves
//! is cut into bands between left and right chains, and each band is triangulated as the sweep reaches
//! its vertices. A band's ceiling is a doubly linked list of vertex refs; a new vertex on a band side is
//! attached to the ceiling and cuts ears off it while the turn is convex, emitting one triangle per ear.
//! Ported from d2d1's tessellator sweep (the MIT-licensed `dotnet/wpf` tessellator is its
//! skeleton, see THIRD-PARTY-NOTICES); `fill_tess` keeps the simple-polygon driver for WARP's fills.

use super::scanner::Pt;
use super::sweep::{Module, ScanFailed, Scanner, DROPPED, R};

/// A ceiling vertex: the scanner vertex it stands for and its neighbours on the ceiling.
#[derive(Clone, Copy)]
struct VRef {
    v: usize,
    l: Option<usize>,
    r: Option<usize>,
}

/// The tessellator module: ceilings per chain and the triangles emitted, in grid coordinates.
#[derive(Default)]
pub struct Tessellator {
    vr: Vec<VRef>,
    ceiling: Vec<Option<usize>>,
    pub tris: Vec<[Pt; 3]>,
}

impl Tessellator {
    fn new_ref(&mut self, v: usize) -> usize {
        self.vr.push(VRef { v, l: None, r: None });
        self.vr.len() - 1
    }

    fn ceil(&self, c: usize) -> Option<usize> {
        self.ceiling.get(c).copied().flatten()
    }

    fn set_ceil(&mut self, c: usize, v: Option<usize>) {
        if self.ceiling.len() <= c {
            self.ceiling.resize(c + 1, None);
        }
        self.ceiling[c] = v;
    }

    /// `ProcessCurrentVertex` / `Connect` / the tails: attach the new ceiling vertex `p` to chain `c`'s
    /// ceiling on the chain's side, then cut ears while the turn at the ceiling is convex. A right-side
    /// chain appends `p` on the ceiling's right and walks left; a left side prepends and walks right.
    fn attach(&mut self, s: &Scanner, c: usize, p: usize) -> R<()> {
        let ceil = self.ceil(c).ok_or(ScanFailed)?;
        let pt = |t: &Self, x: usize| s.v[t.vr[x].v].pt;
        let pp = pt(self, p);
        if s.side_right(c) {
            self.vr[ceil].r = Some(p);
            self.vr[p].l = Some(ceil);
            self.set_ceil(c, Some(p));
            let mut a = ceil;
            while let Some(b) = self.vr[a].l {
                let (pa, pb) = (pt(self, a), pt(self, b));
                if (pp[1] - pa[1]) * (pa[0] - pb[0]) - (pa[1] - pb[1]) * (pp[0] - pa[0]) >= 0.0 {
                    break;
                }
                self.tris.push([pp, pa, pb]);
                self.vr[b].r = Some(p);
                self.vr[p].l = Some(b);
                a = b;
            }
        } else {
            self.vr[p].r = Some(ceil);
            self.vr[ceil].l = Some(p);
            self.set_ceil(c, Some(p));
            let mut a = ceil;
            while let Some(b) = self.vr[a].r {
                let (pa, pb) = (pt(self, a), pt(self, b));
                if (pb[1] - pa[1]) * (pa[0] - pp[0]) - (pb[0] - pa[0]) * (pa[1] - pp[1]) >= 0.0 {
                    break;
                }
                self.tris.push([pa, pp, pb]);
                self.vr[p].r = Some(b);
                self.vr[b].l = Some(p);
                a = b;
            }
        }
        Ok(())
    }

    /// `Connect`: chain `c` takes `ceil` as its ceiling and attaches its head vertex.
    fn connect(&mut self, s: &Scanner, c: usize, ceil: usize) -> R<()> {
        let p = self.new_ref(s.c[c].head);
        self.set_ceil(c, Some(ceil));
        self.attach(s, c, p)
    }

    fn right_of(s: &Scanner, c: usize) -> Option<usize> {
        s.go_while(s.c[c].right, true, |f| f & DROPPED != 0)
    }

    /// `CreateBands`: the heads from `from` to `to` pair up, a left-side head with the right-side head
    /// after it, sharing a new ceiling at the left head's vertex; an odd last head takes over the
    /// rightmost tail's ceiling.
    fn create_bands(&mut self, s: &Scanner, mut from: usize, to: usize) -> R<()> {
        loop {
            if from == to {
                let rt = s.go_while(s.j.rt, false, |f| f & DROPPED != 0).ok_or(ScanFailed)?;
                let c = self.ceil(rt);
                self.set_ceil(from, c);
                self.set_ceil(rt, None);
                return Ok(());
            }
            if s.side_right(from) {
                return Err(ScanFailed);
            }
            let r = Self::right_of(s, from).ok_or(ScanFailed)?;
            if !s.side_right(r) {
                return Err(ScanFailed);
            }
            let v = self.new_ref(s.c[from].head);
            self.set_ceil(from, Some(v));
            self.set_ceil(r, Some(v));
            if r == to {
                return Ok(());
            }
            from = Self::right_of(s, r).ok_or(ScanFailed)?;
        }
    }

    /// `SplitTheBand`: heads starting inside the band left of the junction split its ceiling at its
    /// lowest vertex; the left head connects to the left part, the right head to a copy of that vertex
    /// heading the right part, and the heads between pair into new bands.
    fn split_the_band(&mut self, s: &Scanner, lh: usize, rh: usize) -> R<()> {
        if lh == rh {
            return Err(ScanFailed);
        }
        let left = s.go_while(s.j.left, false, |f| f & DROPPED != 0).ok_or(ScanFailed)?;
        let mut low = self.ceil(left).ok_or(ScanFailed)?;
        while let Some(r) = self.vr[low].r {
            let (a, b) = (s.v[self.vr[low].v].pt, s.v[self.vr[r].v].pt);
            if a[1] <= b[1] && (b[1] != a[1] || a[0] <= b[0]) {
                break;
            }
            low = r;
        }
        let copy = self.new_ref(self.vr[low].v);
        let after = self.vr[low].r;
        self.vr[copy].r = after;
        if let Some(a) = after {
            self.vr[a].l = Some(copy);
        }
        self.vr[low].r = None;
        if after.is_none() {
            let right = s.go_while(s.j.right, true, |f| f & DROPPED != 0).ok_or(ScanFailed)?;
            self.set_ceil(right, Some(copy));
        }
        self.connect(s, lh, low)?;
        self.connect(s, rh, copy)?;
        let next = Self::right_of(s, lh).ok_or(ScanFailed)?;
        if next != rh {
            let last = s.go_while(s.c[rh].left, false, |f| f & DROPPED != 0).ok_or(ScanFailed)?;
            self.create_bands(s, next, last)?;
        }
        Ok(())
    }
}

impl Module for Tessellator {
    fn junction(&mut self, s: &mut Scanner) -> R<()> {
        s.classify_fill_only();
        let skip = |f: u16| f & DROPPED != 0;
        let lh = s.go_while(s.j.lh, true, skip);
        let rh = s.go_while(s.j.rh, false, skip);
        let lt = s.go_while(s.j.lt, true, skip);
        let rt = s.go_while(s.j.rt, false, skip);
        if lh.is_none() && lt.is_none() {
            return Ok(());
        }
        // `ProcessAllTails`: every tail's last vertex closes onto its band's ceiling.
        if let Some(mut t) = lt {
            let rt = rt.ok_or(ScanFailed)?;
            loop {
                let p = self.new_ref(s.c[t].tail);
                self.attach(s, t, p)?;
                if t == rt {
                    break;
                }
                match Self::right_of(s, t) {
                    Some(n) => t = n,
                    None => break,
                }
            }
        }
        if let Some(lh) = lh {
            let rh = rh.ok_or(ScanFailed)?;
            if !s.side_right(lh) {
                return self.create_bands(s, lh, rh);
            }
            let next = if let Some(lt) = lt {
                // The leftmost head continues the leftmost tail's band.
                let c = self.ceil(lt);
                self.set_ceil(lh, c);
                self.set_ceil(lt, None);
                if lh == rh {
                    return Ok(());
                }
                Self::right_of(s, lh)
            } else {
                self.split_the_band(s, lh, rh)?;
                return Ok(());
            };
            if let Some(n) = next {
                return self.create_bands(s, n, rh);
            }
            return Ok(());
        }
        // `MergeTheBands`: only tails, the two bands either side of them join.
        let (lt, rt) = (lt.ok_or(ScanFailed)?, rt.ok_or(ScanFailed)?);
        if !s.side_right(lt) {
            return Ok(());
        }
        if lt == rt {
            return Err(ScanFailed);
        }
        let (lc, rc) = (self.ceil(lt).ok_or(ScanFailed)?, self.ceil(rt).ok_or(ScanFailed)?);
        if let Some(ll) = self.vr[lc].l {
            self.vr[ll].r = Some(rc);
            self.vr[rc].l = Some(ll);
        } else {
            let left = s.go_while(s.j.left, false, skip).ok_or(ScanFailed)?;
            self.set_ceil(left, Some(rc));
            self.vr[rc].l = None;
        }
        Ok(())
    }

    fn current_vertex(&mut self, s: &mut Scanner, c: usize) -> R<()> {
        let p = self.new_ref(s.c[c].cursor);
        self.attach(s, c, p)
    }
}
