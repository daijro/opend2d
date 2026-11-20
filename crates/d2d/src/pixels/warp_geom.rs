//! WARP's aliased geometry rasterizer, which follows WPF's open-source MIL aarasterizer (dotnet/wpf
//! `aarasterizer.cpp`, its aliased filler). Direct2D hands an aliased `FillGeometry`/`DrawGeometry`
//! outline to WARP as 28.4 points; WARP builds one fixed-point DDA edge per non-horizontal segment
//! (the edge setup), steps
//! the active edges one scanline at a time, re-sorts them by x, and fills `[x0, x1)` between winding
//! transitions (the aarasterizer's winding and alternate span fills). Scanline `k` samples
//! `y = 16k`, pixel `x` samples `x = 16x`, both in the 28.4 space.

/// One edge as the edge setup builds it (as in WPF's aarasterizer): pixel x, `floor(dx/dy)` per scanline, the
/// error term with its up/down steps, the scanline range `[start, end)` and the winding direction.
#[derive(Clone, Copy, Debug)]
struct Edge {
    x: i32,
    dx: i32,
    error: i32,
    error_up: i32,
    error_down: i32,
    start: i32,
    end: i32,
    dir: i32,
}

/// The edge setup for one point run (aliased: no ×8 subsampling, so the `(v+8)<<0−8` bias is a
/// no-op), clipped to scanlines `[0, h)`.
fn init_edges(pts: &[[i32; 2]], h: i32, out: &mut Vec<Edge>) {
    let clamp = |v: i32| v.clamp(-0x80_0000, 0x80_0000);
    for s in pts.windows(2) {
        let ([xa, ya], [xb, yb]) = ([clamp(s[0][0]), clamp(s[0][1])], [clamp(s[1][0]), clamp(s[1][1])]);
        let (x_top, y_top, y_bot, dx, dir) = if yb < ya { (xb, yb, ya, xa - xb, -1) } else { (xa, ya, yb, xb - xa, 1) };
        let (start, end) = ((y_top + 15) >> 4, (y_bot + 15) >> 4);
        if end <= start || end <= 0 || start >= h {
            continue;
        }
        let dy = y_bot - y_top;
        let (q, r) = (dx.div_euclid(dy), dx.rem_euclid(dy));
        // Step x from y_top to the first sample scanline in 1/16 rows (floor semantics, error from −1).
        let (mut x, mut error) = (x_top, -1i32);
        let frac = y_top & 15;
        if frac != 0 {
            for _ in 0..16 - frac {
                let t = error + r;
                if t < 0 {
                    error = t;
                    x += q;
                } else {
                    error = t - dy;
                    x += q + 1;
                }
            }
        }
        // Round up to whole pixels: pixel x is right of the edge iff 16·x >= the stepped crossing.
        if x & 15 != 0 {
            error -= dy * (16 - (x & 15));
            x += 15;
        }
        let mut e = Edge { x: x >> 4, dx: q, error: error >> 4, error_up: r, error_down: dy, start, end, dir };
        if e.start < 0 {
            // Clip-top advance: the exact crossing `rows` scanlines further down.
            let v = (e.error + dy) as i64 + dx as i64 * (0 - e.start) as i64;
            let (qq, rr) = (v.div_euclid(dy as i64), v.rem_euclid(dy as i64));
            e.x += qq as i32;
            e.error = rr as i32 - dy;
            e.start = 0;
        }
        out.push(e);
    }
}

/// Rasterize figures of 28.4 points (each closed back to its first point) on a `w×h` target, calling
/// `span(y, x0, x1)` for every filled pixel run, clipped to the target.
pub(crate) fn fill_aliased(figures: &[Vec<[i32; 2]>], winding: bool, w: usize, h: usize, mut span: impl FnMut(usize, usize, usize)) {
    let mut edges = Vec::new();
    for f in figures {
        if f.len() < 2 {
            continue;
        }
        let mut pts = f.clone();
        if pts.first() != pts.last() {
            pts.push(pts[0]);
        }
        init_edges(&pts, h as i32, &mut edges);
    }
    if edges.is_empty() {
        return;
    }
    // Inactive edges in start order (native QuickSort/insertion-sorts by (start, x)).
    edges.sort_by_key(|e| (e.start, e.x));
    let mut next = 0;
    let mut active: Vec<Edge> = Vec::new();
    let mut y = edges[0].start;
    while y < h as i32 && (next < edges.len() || !active.is_empty()) {
        // Insert edges starting here before any active edge at the same x.
        while next < edges.len() && edges[next].start == y {
            let e = edges[next];
            let at = active.iter().position(|a| a.x >= e.x).unwrap_or(active.len());
            active.insert(at, e);
            next += 1;
        }
        if active.is_empty() {
            y = edges[next].start;
            continue;
        }
        if y >= 0 {
            emit(&active, winding, y as usize, w, &mut span);
        }
        // Advance to the next scanline: drop finished edges, step the rest, keep x order (stable).
        y += 1;
        active.retain(|e| e.end > y);
        for e in &mut active {
            e.x += e.dx;
            e.error += e.error_up;
            if e.error >= 0 {
                e.error -= e.error_down;
                e.x += 1;
            }
        }
        active.sort_by_key(|e| e.x);
    }
}

/// The aarasterizer's winding/alternate span fill for one scanline: a span runs from the edge
/// that opens it to the edge that closes it, extended while the next span starts where it ended.
fn emit(active: &[Edge], winding: bool, y: usize, w: usize, span: &mut impl FnMut(usize, usize, usize)) {
    let close = |i: usize| -> Option<usize> {
        if winding {
            let mut acc = active[i].dir;
            let mut j = i + 1;
            while j < active.len() {
                acc += active[j].dir;
                if acc == 0 {
                    return Some(j);
                }
                j += 1;
            }
            None
        } else {
            (i + 1 < active.len()).then_some(i + 1)
        }
    };
    let mut i = 0;
    while i < active.len() {
        let Some(mut j) = close(i) else { break };
        let x0 = active[i].x;
        if x0 != active[j].x {
            while j + 1 < active.len() && active[j + 1].x == active[j].x {
                match close(j + 1) {
                    Some(k) => j = k,
                    None => break,
                }
            }
            let (a, b) = (x0.clamp(0, w as i32) as usize, active[j].x.clamp(0, w as i32) as usize);
            if a < b {
                span(y, a, b);
            }
        }
        i = j + 1;
    }
}

#[cfg(test)]
mod tests {
    use super::fill_aliased;

    /// The sampling rule: scanline k samples y = 16k, pixel x samples x = 16x, an edge crossing exactly
    /// at a sample includes it on its right (a 2×2-pixel square from (16,16) to (48,48) covers pixels
    /// 1..3 × 1..3, half-open), and a bow-tie's two lobes fill under both rules.
    #[test]
    fn square_and_bowtie() {
        let mut px = Vec::new();
        fill_aliased(&[vec![[16, 16], [48, 16], [48, 48], [16, 48]]], true, 8, 8, |y, a, b| px.extend((a..b).map(|x| (x, y))));
        assert_eq!(px, vec![(1, 1), (2, 1), (1, 2), (2, 2)]);
        let mut n = 0;
        fill_aliased(&[vec![[0, 0], [128, 128], [128, 0], [0, 128]]], true, 8, 8, |_, a, b| n += b - a);
        let mut m = 0;
        fill_aliased(&[vec![[0, 0], [128, 128], [128, 0], [0, 128]]], false, 8, 8, |_, a, b| m += b - a);
        assert_eq!(n, m);
        assert!(n > 0);
    }
}
