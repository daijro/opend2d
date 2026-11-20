// Ported from dotnet/wpf (MIT, Copyright (c) .NET Foundation and Contributors):
// src/Microsoft.DotNet.Wpf/src/WpfGfx/core/geometry/LineSegmentIntersection.{cpp,h}. See THIRD-PARTY-NOTICES.

//! The scanner's robust segment intersection, on integer workspace coordinates (|v| < 2^49). For
//! segments AB and CD it classifies the intersection (empty, transverse, overlapping), locates a
//! transverse one on each segment (at the first point, inside, at the last point), and orders
//! intersection points against grid points and against each other, all exactly. A transverse
//! intersection is I = A + λ·AB = C + μ·CD with λ = det(AC, DC) / det(AB, DC) and
//! μ = det(AB, AC) / det(AB, DC); the determinants fit i128, the point comparisons use [`Big`].

use super::exact::Big;
use std::cmp::Ordering;

/// What two segments share.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Empty,
    /// Parallel and overlapping.
    NonTransverse,
    /// Crossing or touching at one point.
    Transverse,
}

/// Where a transverse intersection lies on one of the two segments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loc {
    AtFirst,
    Open,
    AtLast,
}

fn det(ax: i128, ay: i128, bx: i128, by: i128) -> i128 {
    ax * by - ay * bx
}

/// The intersection of AB = `[ax, ay, bx, by]` with CD.
#[derive(Clone, Copy, Debug)]
pub struct Seg {
    pub kind: Kind,
    pub loc_ab: Option<Loc>,
    pub loc_cd: Option<Loc>,
    a: [i128; 2],
    ab: [i128; 2],
    dc: [i128; 2],
    ac: [i128; 2],
    /// det(AB, DC), the common denominator; det(AC, DC), λ's numerator; det(AB, AC), μ's numerator.
    d_abdc: i128,
    d_acdc: i128,
    d_abac: i128,
    /// A transverse intersection's double coordinates and their error bounds ([`Seg::approx`]), made once.
    apx: [f64; 2],
    err: [f64; 2],
    /// Coordinates and determinants small enough that the exact rational comparison ([`Seg::coord_i128`]
    /// and its cross-product) stays within `i128`/256 bits without a `Big`: the common fixed-grid case.
    small: bool,
}

impl Seg {
    /// `PairwiseIntersect`.
    pub fn new(ab: [i64; 4], cd: [i64; 4]) -> Seg {
        let [a0, a1, b0, b1] = ab.map(i128::from);
        let [c0, c1, d0, d1] = cd.map(i128::from);
        let mut s = Seg {
            kind: Kind::Empty,
            loc_ab: None,
            loc_cd: None,
            a: [a0, a1],
            ab: [b0 - a0, b1 - a1],
            dc: [c0 - d0, c1 - d1],
            ac: [c0 - a0, c1 - a1],
            d_abdc: 0,
            d_acdc: 0,
            d_abac: 0,
            apx: [0.0; 2],
            err: [0.0; 2],
            small: false,
        };
        // Disjoint bounding boxes.
        if a0.min(b0) > c0.max(d0) || a0.max(b0) < c0.min(d0) || a1.min(b1) > c1.max(d1) || a1.max(b1) < c1.min(d1) {
            return s;
        }
        let (xab, yab) = (b0 - a0, b1 - a1);
        let (xdc, ydc) = (c0 - d0, c1 - d1);
        let (xac, yac) = (c0 - a0, c1 - a1);
        s.d_abdc = det(xab, yab, xdc, ydc);
        s.d_abac = det(xab, yab, xac, yac);
        if s.d_abdc == 0 {
            // Parallel: overlapping when collinear and C or D falls on AB, or CD spans it.
            if s.d_abac == 0 {
                let rc = region_on_ab(xab, yab, xac, yac);
                let rd = region_on_ab(xab, yab, d0 - a0, d1 - a1);
                let before = |r: Region| r == Region::BeforeFirst;
                let after = |r: Region| r == Region::AfterLast;
                if !((before(rc) && before(rd)) || (after(rc) && after(rd))) {
                    s.kind = Kind::NonTransverse;
                }
            }
            return s;
        }
        s.d_acdc = det(xac, yac, xdc, ydc);
        let sg = s.d_abdc.signum();
        // λ = d_acdc / d_abdc must lie in [0, 1], μ = d_abac / d_abdc too.
        if s.d_acdc.signum() == -sg || s.d_abac.signum() == -sg {
            return s;
        }
        let lam_vs_one = (s.d_acdc.abs()).cmp(&s.d_abdc.abs());
        let mu_vs_one = (s.d_abac.abs()).cmp(&s.d_abdc.abs());
        if lam_vs_one == Ordering::Greater || mu_vs_one == Ordering::Greater {
            return s;
        }
        s.kind = Kind::Transverse;
        let loc = |num: i128, vs_one: Ordering| {
            if num == 0 {
                Loc::AtFirst
            } else if vs_one == Ordering::Less {
                Loc::Open
            } else {
                Loc::AtLast
            }
        };
        s.loc_ab = Some(loc(s.d_acdc, lam_vs_one));
        s.loc_cd = Some(loc(s.d_abac, mu_vs_one));
        let q = s.d_acdc as f64 / s.d_abdc as f64;
        for k in 0..2 {
            let (a, ab) = (s.a[k] as f64, s.ab[k] as f64);
            s.apx[k] = a + q * ab;
            s.err[k] = (a.abs() + 6.0 * ab.abs()) * f64::EPSILON;
        }
        // The exact rational comparison stays in i128/256 bits (no Big) when |a|,|ab| < 2^41 and the
        // determinants < 2^84: then `coord`'s numerator a·d+n·ab < 2^126 (fits i128) and two of those
        // cross-multiplied < 2^210 (fits 256 bits). The fixed 2^20 workspace (realization/region) keeps
        // coordinates under 2^40, so this holds there; the fit workspace's wider coordinates take the
        // Big path. Bounds validated here so [`Seg::coord_i128`] can multiply unchecked.
        const C: u128 = 1 << 41;
        const D: u128 = 1 << 84;
        s.small = s.a[0].unsigned_abs() < C && s.a[1].unsigned_abs() < C && s.ab[0].unsigned_abs() < C && s.ab[1].unsigned_abs() < C && s.d_abdc.unsigned_abs() < D && s.d_acdc.unsigned_abs() < D;
        s
    }

    /// The span on axis `k` that holds a transverse intersection: the overlap of AB's and CD's extents
    /// (`GetTransverseIntersectionYSpan`, and its x twin). Exact on the grid.
    fn span(&self, k: usize) -> (i128, i128) {
        let (a, b) = (self.a[k], self.a[k] + self.ab[k]);
        let c = self.a[k] + self.ac[k];
        let d = c - self.dc[k];
        (a.min(b).max(c.min(d)), a.max(b).min(c.max(d)))
    }

    /// The transverse intersection's coordinate `k` as a double and a bound on its error: `a + (λn/λd)·ab` with
    /// λ in [0, 1] and `a`, `ab` exact grid integers (|·| < 2^53). The quotient carries three roundings (the two
    /// conversions and the division), the product and the sum one each, so the error is at most
    /// 2^-53·(|a| + 5.1·|ab|); the bound doubles that (`ComputeIntersectionPointYCoordinateInterval`).
    fn approx(&self, k: usize) -> (f64, f64) {
        (self.apx[k], self.err[k])
    }

    /// The intersection's coordinate `k` as a rational `num / den` with `den > 0`.
    fn coord(&self, k: usize) -> (Big, Big) {
        let (n, d) = if self.d_abdc < 0 { (-self.d_acdc, -self.d_abdc) } else { (self.d_acdc, self.d_abdc) };
        let (d, n) = (Big::from_i128(d), Big::from_i128(n));
        (Big::from_i128(self.a[k]).mul(d).add(n.mul(Big::from_i128(self.ab[k]))), d)
    }

    /// [`Seg::coord`]'s `num / den` (den > 0) in native `i128`, multiplied UNCHECKED — only valid, and
    /// only called, when `self.small` (bounds validated in `new`: `num = a·d + n·ab < 2^126`). The same
    /// integers the `Big` form holds, so cross-multiplying two of these is the identical exact compare
    /// without building a `Big` per numerator (the hot path: `yx_sort_intersection_pair` is ~50% of P1).
    fn coord_i128(&self, k: usize) -> (i128, i128) {
        debug_assert!(self.small);
        let (n, d) = if self.d_abdc < 0 { (-self.d_acdc, -self.d_abdc) } else { (self.d_acdc, self.d_abdc) };
        (self.a[k] * d + n * self.ab[k], d)
    }

    /// The parameter along AB (`ab`) or CD as the scanner's double (`ParameterAlongAB`/`CD`): 0 or 1 at
    /// an endpoint, else the double determinants' quotient clamped to [0, 1].
    pub fn parameter(&self, ab: bool) -> f64 {
        match if ab { self.loc_ab } else { self.loc_cd } {
            Some(Loc::AtFirst) => 0.0,
            Some(Loc::AtLast) => 1.0,
            _ => {
                let (xab, yab, xdc, ydc, xac, yac) = self.diffs();
                let num = if ab { det_f64(xac, yac, xdc, ydc) } else { det_f64(xab, yab, xac, yac) };
                (num / det_f64(xab, yab, xdc, ydc)).min(1.0).max(0.0)
            }
        }
    }

    /// The exact parameter along AB (`ab`) or CD as `num / den` with `den > 0`.
    fn param_frac(&self, ab: bool) -> (i128, i128) {
        let num = if ab { self.d_acdc } else { self.d_abac };
        if self.d_abdc < 0 { (-num, -self.d_abdc) } else { (num, self.d_abdc) }
    }

    fn diffs(&self) -> (f64, f64, f64, f64, f64, f64) {
        let (xab, yab) = (self.ab[0] as f64, self.ab[1] as f64);
        let (xdc, ydc) = (self.dc[0] as f64, self.dc[1] as f64);
        let (xac, yac) = (self.ac[0] as f64, self.ac[1] as f64);
        (xab, yab, xdc, ydc, xac, yac)
    }
}

/// A determinant `a·d − b·c` of exact-integer doubles as the scanner keeps it: the rounded products'
/// difference, or the exact difference when the products round equal past 2^53.
pub fn det_f64(a: f64, b: f64, c: f64, d: f64) -> f64 {
    let (p, q) = (a * d, b * c);
    if p == q && p.abs() > 9.007_199_254_740_992e15 {
        (a as i128 * d as i128 - b as i128 * c as i128) as f64
    } else {
        p - q
    }
}

/// The approximate point at parameter `t` along the segment from `base` to `tip`: `(tip − base)·t + base`.
pub fn approx_point(base: [f64; 2], tip: [f64; 2], t: f64) -> [f64; 2] {
    [(tip[0] - base[0]) * t + base[0], (tip[1] - base[1]) * t + base[1]]
}

/// Where a point M on AB's line lies relative to the segment (`ComputeRegionWhenPointPIsOnAB`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Region {
    BeforeFirst,
    First,
    Open,
    Last,
    AfterLast,
}

/// `AM = (xam, yam)` against `AB = (xab, yab)`, M collinear with AB.
pub fn region_on_ab(xab: i128, yab: i128, xam: i128, yam: i128) -> Region {
    let dot = xab * xam + yab * yam;
    let len = xab * xab + yab * yab;
    match (dot.cmp(&0), dot.cmp(&len)) {
        (Ordering::Less, _) => Region::BeforeFirst,
        (Ordering::Equal, _) => Region::First,
        (_, Ordering::Less) => Region::Open,
        (_, Ordering::Equal) => Region::Last,
        _ => Region::AfterLast,
    }
}

/// The side of point C relative to the oriented line AB: 1 left, 0 on it, −1 right (`LocatePointRelativeToLine`).
pub fn side_of_point(c: [i64; 2], ab: [i64; 4]) -> i32 {
    let [a0, a1, b0, b1] = ab.map(i128::from);
    det(b0 - a0, b1 - a1, c[0] as i128 - a0, c[1] as i128 - a1).signum() as i32
}

/// The side of a transverse intersection relative to the oriented line EF (`LocateTransverseIntersectionRelativeToLine`).
pub fn side_of_intersection(s: &Seg, ef: [i64; 4]) -> i32 {
    let [e0, e1, f0, f1] = ef.map(i128::from);
    let (x, d) = s.coord(0);
    let (y, _) = s.coord(1);
    // det(EF, I − E)·d = (f−e)x·(y − e1·d) − (f−e)y·(x − e0·d).
    let (fx, fy) = (Big::from_i128(f0 - e0), Big::from_i128(f1 - e1));
    let v = fx.mul(y.sub(Big::from_i128(e1).mul(d))).sub(fy.mul(x.sub(Big::from_i128(e0).mul(d))));
    v.signum()
}

/// Orders a transverse intersection against grid point E by (y, then x) (`YXSortTransverseIntersectionAndPoint`).
pub fn yx_sort_intersection_and_point(s: &Seg, e: [i64; 2]) -> Ordering {
    for k in [1, 0] {
        // Exact shortcuts first, as WPF's sort takes them: the intersection lies within both segments' extents, so a
        // coordinate outside them orders at once, and a double interval that misses it does too. Only a near tie
        // needs the big-integer products.
        let (lo, hi) = s.span(k);
        let v = e[k] as i128;
        if hi < v {
            return Ordering::Less;
        }
        if lo > v {
            return Ordering::Greater;
        }
        if lo == hi {
            continue; // the intersection's coordinate is the span's, `v`
        }
        let (x, err) = s.approx(k);
        let p = e[k] as f64;
        if x + err < p {
            return Ordering::Less;
        }
        if x - err > p {
            return Ordering::Greater;
        }
        // n/d vs e[k] (den > 0) == n vs e[k]·d. `small` builds n in i128 (e[k]·d can exceed i128, so cross
        // through Big either way).
        let o = if s.small {
            let (n, d) = s.coord_i128(k);
            Big::from_i128(n).cmp(&Big::from_i128(e[k] as i128).mul(Big::from_i128(d)))
        } else {
            let (n, d) = s.coord(k);
            n.cmp(&Big::from_i128(e[k] as i128).mul(d))
        };
        if o != Ordering::Equal {
            return o;
        }
    }
    Ordering::Equal
}

/// Orders two transverse intersections by (y, then x) (`YXSortTransverseIntersectionPair`).
pub fn yx_sort_intersection_pair(s: &Seg, t: &Seg) -> Ordering {
    for k in [1, 0] {
        // Disjoint spans, equal degenerate spans, or disjoint double intervals decide exactly before any big-integer
        // product (`YXSortTransverseIntersectionPair`).
        let ((lo1, hi1), (lo2, hi2)) = (s.span(k), t.span(k));
        if hi1 < lo2 {
            return Ordering::Less;
        }
        if lo1 > hi2 {
            return Ordering::Greater;
        }
        if lo1 == hi1 && lo2 == hi2 {
            continue; // both coordinates are their spans', here equal
        }
        let ((x1, e1), (x2, e2)) = (s.approx(k), t.approx(k));
        if x1 + e1 < x2 - e2 {
            return Ordering::Less;
        }
        if x1 - e1 > x2 + e2 {
            return Ordering::Greater;
        }
        // n1/d1 vs n2/d2 (den > 0) == n1·d2 vs n2·d1. When both Segs are `small`, build the numerators in
        // native i128 (no per-numerator Big) and cross-multiply through the Big; else the all-Big form.
        let o = if s.small && t.small {
            let ((n1, d1), (n2, d2)) = (s.coord_i128(k), t.coord_i128(k));
            Big::from_i128(n1).mul(Big::from_i128(d2)).cmp(&Big::from_i128(n2).mul(Big::from_i128(d1)))
        } else {
            let ((n1, d1), (n2, d2)) = (s.coord(k), t.coord(k));
            n1.mul(d2).cmp(&n2.mul(d1))
        };
        if o != Ordering::Equal {
            return o;
        }
    }
    Ordering::Equal
}

/// Orders two transverse intersections on a common segment by their parameters along it, the segment
/// being AB (`s_ab`) or CD of each (`SortTransverseIntersectionsAlongCommonLineSegment`).
pub fn param_cmp(s: &Seg, s_ab: bool, t: &Seg, t_ab: bool) -> Ordering {
    let ((n1, d1), (n2, d2)) = (s.param_frac(s_ab), t.param_frac(t_ab));
    // Float filter (the two yx_sort comparators have one; param_cmp lacked it, so every parameter sort
    // formed two Bigs). The parameters are n/d in [0, 1] with d > 0 and n >= 0, so λ1 vs λ2 == n1·d2 vs
    // n2·d1; compute both cross products as f64 and decide when their rounding intervals are disjoint,
    // falling to the exact cross-multiply only on a near tie. Each i128→f64 conversion and the multiply
    // round once (<= 2^-53 relative), so the product's error is under 3.1·2^-53 of its magnitude;
    // 4·2^-53 (= 2·f64::EPSILON) is a safe over-estimate, so a disjoint pair orders exactly. Ports WPF
    // LambdaABSortTransverseIntersectionPair's double-product filter (LineSegmentIntersection.cpp:2214),
    // which likewise compares the products and is exact only on a tie.
    let (p1, p2) = (n1 as f64 * d2 as f64, n2 as f64 * d1 as f64);
    let (e1, e2) = (p1.abs() * (2.0 * f64::EPSILON), p2.abs() * (2.0 * f64::EPSILON));
    if p1 + e1 < p2 - e2 {
        return Ordering::Less;
    }
    if p1 - e1 > p2 + e2 {
        return Ordering::Greater;
    }
    Big::from_i128(n1).mul(Big::from_i128(d2)).cmp(&Big::from_i128(n2).mul(Big::from_i128(d1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The span and double-interval shortcuts order exactly as the big-integer products alone do, on crossings that
    /// are far apart, clustered in a sliver, on axis-parallel segments, and near the 2^48 workspace edge.
    #[test]
    fn sort_shortcuts_match_exact_order() {
        let exact_pair = |s: &Seg, t: &Seg| {
            for k in [1, 0] {
                let ((n1, d1), (n2, d2)) = (s.coord(k), t.coord(k));
                let o = n1.mul(d2).cmp(&n2.mul(d1));
                if o != Ordering::Equal {
                    return o;
                }
            }
            Ordering::Equal
        };
        let exact_point = |s: &Seg, e: [i64; 2]| {
            for k in [1, 0] {
                let (n, d) = s.coord(k);
                let o = n.cmp(&Big::from_i128(e[k] as i128).mul(d));
                if o != Ordering::Equal {
                    return o;
                }
            }
            Ordering::Equal
        };
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut rnd = |m: i64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % (2 * m as u64 + 1)) as i64 - m
        };
        let mut segs = Vec::new();
        for round in 0..40_000 {
            let m = [8, 1000, 1 << 30, (1 << 47) - 1][round % 4];
            let mut q = [0i64; 8];
            for v in q.iter_mut() {
                *v = rnd(m);
            }
            if round % 7 == 0 {
                q[3] = q[1]; // a horizontal segment
            }
            let s = Seg::new([q[0], q[1], q[2], q[3]], [q[4], q[5], q[6], q[7]]);
            if s.kind == Kind::Transverse {
                segs.push(s);
            }
        }
        assert!(segs.len() > 2000);
        for (i, s) in segs.iter().enumerate() {
            let t = &segs[(i * 7 + 3) % segs.len()];
            assert_eq!(yx_sort_intersection_pair(s, t), exact_pair(s, t));
            assert_eq!(yx_sort_intersection_pair(s, s), Ordering::Equal);
            let e = [s.a[0] as i64, s.a[1] as i64];
            assert_eq!(yx_sort_intersection_and_point(s, e), exact_point(s, e));
            let e = [rnd(8), rnd(8)];
            assert_eq!(yx_sort_intersection_and_point(s, e), exact_point(s, e));
        }
    }

    #[test]
    fn classifies_and_orders() {
        // An X: crossing at (5, 5) inside both.
        let x = Seg::new([0, 0, 10, 10], [0, 10, 10, 0]);
        assert_eq!(x.kind, Kind::Transverse);
        assert_eq!((x.loc_ab, x.loc_cd), (Some(Loc::Open), Some(Loc::Open)));
        assert_eq!(yx_sort_intersection_and_point(&x, [5, 5]), Ordering::Equal);
        assert_eq!(yx_sort_intersection_and_point(&x, [6, 5]), Ordering::Less);
        assert_eq!(yx_sort_intersection_and_point(&x, [0, 6]), Ordering::Less);
        assert_eq!(x.parameter(true), 0.5);
        assert_eq!(approx_point([0.0, 0.0], [10.0, 10.0], x.parameter(true)), [5.0, 5.0]);
        // Products past 2^53 that round equal fall back to the exact difference.
        let big = (1i64 << 40) as f64;
        assert_eq!(det_f64(big + 1.0, big, big, big - 1.0), -1.0); // (2^80 − 1) − 2^80
        // A T: CD starts on AB.
        let t = Seg::new([0, 0, 10, 0], [4, 0, 4, 7]);
        assert_eq!((t.kind, t.loc_ab, t.loc_cd), (Kind::Transverse, Some(Loc::Open), Some(Loc::AtFirst)));
        // Shared endpoint.
        let v = Seg::new([0, 0, 10, 0], [10, 0, 10, 9]);
        assert_eq!((v.loc_ab, v.loc_cd), (Some(Loc::AtLast), Some(Loc::AtFirst)));
        // Collinear overlap, collinear apart, parallel apart, crossing lines but disjoint segments.
        assert_eq!(Seg::new([0, 0, 10, 0], [5, 0, 15, 0]).kind, Kind::NonTransverse);
        assert_eq!(Seg::new([0, 0, 10, 0], [11, 0, 15, 0]).kind, Kind::Empty);
        assert_eq!(Seg::new([0, 0, 10, 0], [0, 1, 10, 1]).kind, Kind::Empty);
        assert_eq!(Seg::new([0, 0, 4, 4], [10, 0, 6, 4]).kind, Kind::Empty);
        // Sides, and two crossings of one segment ordered along it and in (y, x).
        assert_eq!(side_of_point([5, 9], [0, 0, 10, 0]), 1);
        assert_eq!(side_of_intersection(&x, [0, 0, 10, 0]), 1);
        let near = Seg::new([0, 0, 100, 0], [10, -5, 10, 5]);
        let far = Seg::new([0, 0, 100, 0], [70, -5, 70, 5]);
        assert_eq!(param_cmp(&near, true, &far, true), Ordering::Less);
        assert_eq!(param_cmp(&near, false, &far, false), Ordering::Equal); // both halfway up their CD
        assert_eq!(near.parameter(false), 0.5);
        assert_eq!(yx_sort_intersection_pair(&near, &far), Ordering::Less);
        // Far coordinates near the 2^48 workspace edge still order exactly.
        let big = 1i64 << 47;
        let a = Seg::new([-big, -big, big, big - 1], [-big, big, big, -big]);
        let b = Seg::new([-big, -big, big, big], [-big, big, big, -big]);
        assert_eq!(yx_sort_intersection_pair(&a, &b), Ordering::Less);
    }
}
