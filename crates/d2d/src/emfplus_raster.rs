//! GDI+'s aliased polygon scan converter, which `DrawGdiMetafile` uses for EMF+ playback. A
//! flattened polygon in ×16 sub-pixel space is turned into one fixed-point DDA per edge, then walked
//! scanline by scanline and filled even-odd.
//!
//! Both stages are byte-exact against GDI+ on the ellipse oracle (`FillEllipse 12,8,38,30` →
//! `mfc_nat.bgra`): `init_edges` reproduces all 28 edges and `rasterize` reproduces all 30 filled
//! rows (see the `ellipse_byte_exact` test). The flatten (the ellipse's four kappa-Béziers through the
//! forward-difference flattener) and the device half-pixel transform that positions the polygon are
//! wired in by the caller.

/// One scan edge, as GDI+'s edge setup builds it: current x (pixel), integer x-step per scanline, Bresenham error + increment, |dy|, and the
/// `[y0, y1)` active scanline range. Winding is kept for the winding-fill variant (unused here).
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub x: i32,
    pub dx: i32,
    pub err: i32,
    pub errn: i32,
    pub dy: i32,
    pub y0: i32,
    pub y1: i32,
    pub w: i32,
}

/// GDI+'s edge setup for the aliased fill (SmoothingMode None): points stay in ×16
/// sub-pixel space (no ×8 re-scale), the alignment loop snaps each edge's x to its first integer
/// scanline, and the stored x is `x16 >> 4` (pixel) with the sub-pixel carried in `err`. `poly` is
/// the flattened polygon in ×16 units; the figure is closed implicitly.
pub fn init_edges(poly: &[(i32, i32)]) -> Vec<Edge> {
    let n = poly.len();
    let mut edges = Vec::new();
    for i in 0..n {
        let (p0x, p0y) = poly[i];
        let (p1x, p1y) = poly[(i + 1) % n];
        let (mut x16, abs_dy, mut dx16, top_y, top_ceil, bot_ceil, winding); // x, |dy|, dx, startY, ceilTop, ceilBot, winding
        if p1y - p0y < 0 {
            // edge goes up: walk it top→bottom from p1
            abs_dy = p0y - p1y;
            winding = -1;
            dx16 = p0x - p1x;
            x16 = p1x;
            top_y = p1y;
            top_ceil = p1y + 15;
            bot_ceil = p0y + 15;
        } else {
            abs_dy = p1y - p0y;
            winding = 1;
            dx16 = p1x - p0x;
            x16 = p0x;
            top_y = p0y;
            top_ceil = p0y + 15;
            bot_ceil = p1y + 15;
        }
        let y_start = top_ceil >> 4; // yStart (ceil of top)
        let y_end = bot_ceil >> 4; // yEnd   (ceil of bottom)
        if y_end <= y_start {
            continue; // horizontal / sub-scanline: contributes nothing
        }
        // The integer x-step and the error increment: signed-divide dx by |dy|, rounding toward -inf for dx<0.
        let (x_step, err_inc);
        if dx16 < 0 {
            let neg_dx = -dx16;
            if neg_dx < abs_dy {
                x_step = -1;
                err_inc = abs_dy + dx16;
            } else {
                let rem = neg_dx % abs_dy;
                if rem > 0 {
                    x_step = -(neg_dx / abs_dy) - 1;
                    err_inc = abs_dy - rem;
                } else {
                    x_step = -(neg_dx / abs_dy);
                    err_inc = rem;
                }
            }
        } else if dx16 < abs_dy {
            x_step = 0;
            err_inc = dx16;
        } else {
            err_inc = dx16 % abs_dy;
            x_step = dx16 / abs_dy;
        }
        // Align x from the top vertex down to the first integer scanline (16 - fractional sub-rows).
        let mut err = -1; // error accumulator
        let frac = top_y & 0xF;
        if frac != 0 {
            for _ in 0..(16 - frac) {
                let no_carry = (err_inc + err) < 0;
                err += err_inc;
                let stepped = x_step + x16;
                let err_after = err;
                if !no_carry {
                    err -= abs_dy;
                }
                x16 = if err_after < 0 { stepped } else { stepped + 1 };
            }
        }
        if x16 & 0xF != 0 {
            err -= abs_dy * (16 - (x16 & 0xF));
            x16 += 15;
        }
        edges.push(Edge {
            x: x16 >> 4,
            dx: x_step,
            err: err >> 4,
            errn: err_inc,
            dy: abs_dy,
            y0: y_start,
            y1: y_end,
            w: winding,
        });
    }
    edges
}

/// Walk the edges scanline by scanline (Bresenham advance per pixel-row:
/// `x += dx; err += errn; if err >= 0 { x += 1; err -= dy }`) and fill even-odd
/// spans `[xLeft, xRight)`. Returns, per scanline `y`, the
/// filled pixel columns clamped to `[0, w)`.
pub fn rasterize(edges: &[Edge], w: i32, h: i32) -> Vec<Vec<i32>> {
    let mut cross: Vec<Vec<i32>> = vec![Vec::new(); h.max(0) as usize];
    for e in edges {
        let mut x = e.x;
        let mut err = e.err;
        for y in e.y0..e.y1 {
            if (0..h).contains(&y) {
                cross[y as usize].push(x);
            }
            x += e.dx;
            err += e.errn;
            if err >= 0 {
                x += 1;
                err -= e.dy;
            }
        }
    }
    cross
        .into_iter()
        .map(|mut xs| {
            xs.sort_unstable();
            let mut px = Vec::new();
            let mut k = 0;
            while k + 1 < xs.len() {
                for x in xs[k]..xs[k + 1] {
                    if (0..w).contains(&x) {
                        px.push(x);
                    }
                }
                k += 2;
            }
            px
        })
        .collect()
}

/// Flatten one cubic bezier (4 control points in ×16 sub-pixel space) by GDI+'s 32-bit
/// forward-difference subdivision (fixed-point i32 wrapping arithmetic; `>>` is arithmetic).
/// Appends the flattened points to `out`. Verified byte-exact against GDI+.
fn flatten_bezier(p: &[(i32, i32); 4], out: &mut Vec<(i32, i32)>) {
    let mut t = [0i32; 13];
    t[0] = 1;
    let xs = [p[0].0, p[1].0, p[2].0, p[3].0];
    let ys = [p[0].1, p[1].1, p[2].1, p[3].1];
    t[9] = *xs.iter().min().unwrap() - 16;
    t[10] = *ys.iter().min().unwrap() - 16;
    t[11] = *xs.iter().max().unwrap() + 16;
    t[12] = *ys.iter().max().unwrap() + 16;
    let (xm, ym) = (t[9], t[10]);
    let x0 = p[0].0 - xm;
    let x1 = p[1].0 - xm;
    let x2 = p[2].0 - xm;
    let x3 = p[3].0 - xm;
    let y0 = p[0].1 - ym;
    let y1 = p[1].1 - ym;
    let y2 = p[2].1 - ym;
    let y3 = p[3].1 - ym;
    // Coords are bounded (<0x4000 after the -min bias) for metafile playback, so the 64-bit flattener is never needed.
    t[1] = x0.wrapping_shl(10);
    t[2] = (x3 - x0).wrapping_shl(10);
    t[3] = 6144i32.wrapping_mul(x3 + x1 - 2 * x2);
    t[4] = 6144i32.wrapping_mul(x2 + x0 - 2 * x1);
    t[5] = y0.wrapping_shl(10);
    t[6] = (y3 - y0).wrapping_shl(10);
    t[7] = 6144i32.wrapping_mul(y3 + y1 - 2 * y2);
    t[8] = 6144i32.wrapping_mul(y2 + y0 - 2 * y1);
    let mut div_shift: u32 = 0;
    loop {
        let limit = 24576i32.wrapping_shl(div_shift);
        if t[3].abs().max(t[4].abs()) <= limit && t[7].abs().max(t[8].abs()) <= limit {
            break;
        }
        div_shift += 2;
        let x_avg = t[3].wrapping_add(t[4]) >> 1;
        t[3] = x_avg;
        let y_avg = t[8].wrapping_add(t[7]) >> 1;
        t[7] = y_avg;
        t[2] = t[2].wrapping_sub(x_avg >> div_shift) >> 1;
        t[6] = t[6].wrapping_sub(y_avg >> div_shift) >> 1;
        t[0] = t[0].wrapping_mul(2);
    }
    t[1] = t[1].wrapping_mul(8);
    t[2] = t[2].wrapping_mul(8);
    let sh = div_shift as i32 - 3;
    let shl = (3 - div_shift as i32).max(0) as u32;
    if sh < 0 {
        t[4] = t[4].wrapping_shl(shl);
        t[3] = t[3].wrapping_shl(shl);
    } else {
        t[4] >>= sh as u32;
        t[3] >>= sh as u32;
    }
    t[5] = t[5].wrapping_mul(8);
    t[6] = t[6].wrapping_mul(8);
    if sh < 0 {
        t[8] = t[8].wrapping_shl(shl);
        t[7] = t[7].wrapping_shl(shl);
    } else {
        t[8] >>= sh as u32;
        t[7] >>= sh as u32;
    }
    let (xdd, xd) = (t[3], t[2]);
    t[1] = t[1].wrapping_add(xd);
    t[2] = xdd.wrapping_add(xd);
    let xddd = xdd - t[4];
    t[4] = xdd;
    t[3] = xdd.wrapping_add(xddd);
    let (ydd, yd) = (t[7], t[6]);
    t[5] = t[5].wrapping_add(yd);
    t[6] = ydd.wrapping_add(yd);
    let yddd = ydd - t[8];
    t[8] = ydd;
    t[7] = ydd.wrapping_add(yddd);
    t[0] -= 1;
    let mut guard = 1_000_000i64;
    loop {
        out.push((t[9] + (t[1].wrapping_add(4096) >> 13), t[10] + (t[5].wrapping_add(4096) >> 13)));
        let mut steps = t[0];
        if steps == 0 {
            break;
        }
        let (mut xdd_hi, mut xdd_lo, mut ydd_hi, mut ydd_lo) = (t[3], t[4], t[7], t[8]);
        if t[3].abs().max(t[4].abs()).max(t[7].abs().max(t[8].abs())) > 196608 {
            xdd_hi = xdd_lo.wrapping_add(xdd_hi) >> 3;
            t[3] = xdd_hi;
            ydd_hi = ydd_lo.wrapping_add(ydd_hi) >> 3;
            t[2] = t[2].wrapping_sub(xdd_hi) >> 1;
            xdd_lo >>= 2;
            t[4] = xdd_lo;
            ydd_lo >>= 2;
            t[6] = t[6].wrapping_sub(ydd_hi) >> 1;
            steps = steps.wrapping_mul(2);
            t[7] = ydd_hi;
            t[8] = ydd_lo;
            t[0] = steps;
        }
        while steps & 1 == 0 {
            if xdd_lo.abs().max((2i32.wrapping_mul(xdd_hi).wrapping_sub(xdd_lo)).abs()) > 49152 {
                break;
            }
            if ydd_lo.abs().max((2i32.wrapping_mul(ydd_hi).wrapping_sub(ydd_lo)).abs()) > 49152 {
                break;
            }
            let xd_prev = t[2];
            xdd_lo = xdd_lo.wrapping_mul(4);
            t[4] = xdd_lo;
            ydd_lo = ydd_lo.wrapping_mul(4);
            steps >>= 1;
            t[2] = xdd_hi.wrapping_add(2i32.wrapping_mul(xd_prev));
            xdd_hi = 8i32.wrapping_mul(xdd_hi).wrapping_sub(xdd_lo);
            t[3] = xdd_hi;
            let yd_prev = t[6];
            t[8] = ydd_lo;
            t[6] = ydd_hi.wrapping_add(2i32.wrapping_mul(yd_prev));
            ydd_hi = 8i32.wrapping_mul(ydd_hi).wrapping_sub(ydd_lo);
            t[7] = ydd_hi;
            t[0] = steps;
        }
        t[0] = steps - 1;
        let (step_xdd, step_xd) = (t[3], t[2]);
        t[1] = t[1].wrapping_add(step_xd);
        t[2] = step_xdd.wrapping_add(step_xd);
        let step_xddd = step_xdd - t[4];
        t[4] = step_xdd;
        t[3] = step_xdd.wrapping_add(step_xddd);
        let (step_ydd, step_yd) = (t[7], t[6]);
        t[5] = t[5].wrapping_add(step_yd);
        t[6] = step_ydd.wrapping_add(step_yd);
        t[7] = 2i32.wrapping_mul(step_ydd).wrapping_sub(t[8]);
        t[8] = step_ydd;
        guard -= 1;
        if guard == 0 {
            break;
        }
    }
}

/// Flatten an ellipse (four kappa-beziers, as GDI+ builds it) to a closed ×16 polygon under the
/// device transform `[sx, sy, tx, ty]` (diagonal scale + ×16-unit translate), via the
/// fixed-point vertex snap (`(floor((s·c+t)·16+0.5)+15)>>4`, float32) + `flatten_bezier`.
/// For natural-size metafile playback `sx = sy = 16` and `(tx, ty)` is the measured device
/// half-pixel (see `device_halfpixel`).
/// GDI+'s fixed-point vertex snap through a diagonal device matrix (scale `s` + ×16-unit
/// translate `t`): `(floor((s·c + t)·16 + 0.5) + 15) >> 4`, in float32. Lines and bezier control
/// points alike are snapped this way before flattening / edge building.
pub fn snap_coord(c: f32, s: f32, t: f32) -> i32 {
    (((s * c + t) * 16.0 + 0.5).floor() as i32 + 15) >> 4
}

pub fn flatten_ellipse(rect: [f32; 4], sx: f32, sy: f32, tx: f32, ty: f32) -> Vec<(i32, i32)> {
    let (ex, ey, ew, eh) = (rect[0], rect[1], rect[2], rect[3]);
    let (cx, cy, rx, ry) = (ew * 0.5 + ex, eh * 0.5 + ey, ew * 0.5, eh * 0.5);
    let k: f32 = (2.0f32.sqrt() - 1.0) * 4.0 / 3.0;
    let unit = [
        (1.0, 0.0), (1.0, k), (k, 1.0), (0.0, 1.0), (-k, 1.0), (-1.0, k), (-1.0, 0.0),
        (-1.0, -k), (-k, -1.0), (0.0, -1.0), (k, -1.0), (1.0, -k), (1.0, 0.0),
    ];
    let p13: Vec<(i32, i32)> = unit
        .iter()
        .map(|&(ux, uy)| (snap_coord(ux * rx + cx, sx, tx), snap_coord(uy * ry + cy, sy, ty)))
        .collect();
    let mut poly = Vec::new();
    for q in 0..4 {
        flatten_bezier(&[p13[q * 3], p13[q * 3 + 1], p13[q * 3 + 2], p13[q * 3 + 3]], &mut poly);
    }
    let mut dd: Vec<(i32, i32)> = Vec::new();
    for p in poly {
        if dd.last() != Some(&p) {
            dd.push(p);
        }
    }
    if dd.len() > 1 && dd.first() == dd.last() {
        dd.pop();
    }
    dd
}

/// PixelOffsetMode `Half` (−0.5px) in ×16 sub-pixel units — the clean half-pixel GDI+ applies at the
/// aliased raster. Combined with the per-axis logical→device scale `16·96/dpi` (the metafile's
/// EMF+ logical 96-dpi normalized to its recorded device dpi), this is the full device transform for
/// natural-size metafile playback: `flatten_ellipse(rect, 16·96/dpiX, 16·96/dpiY, HALF_PIXEL_X16 +
/// 16·destX, HALF_PIXEL_X16 + 16·destY)`. Byte-exact vs native `DrawGdiMetafile` on 10 diverse
/// ellipses (verified per-dpi, not a fitted constant; the 96/dpi scale is what places the extreme
/// edges of large ellipses correctly).
pub const HALF_PIXEL_X16: f32 = -8.0;

/// Per-axis logical→device scale (×16 sub-pixel) for natural-size playback of a metafile recorded at
/// `dpi`: `16·96/dpi` (EMF+ logical units are 96-dpi; GDI+ normalizes to the device). This is the
/// scale GDI+'s metafile transform reduces to for a 1:1 draw.
pub fn device_scale(dpi: (f32, f32)) -> (f32, f32) {
    (16.0 * 96.0 / dpi.0, 16.0 * 96.0 / dpi.1)
}

/// Walk the edges scanline by scanline and fill by the **nonzero-winding** rule
/// (GDI+'s winding filler): sort each scanline's crossings by x, accumulate the per-edge
/// winding (±1), and fill `[x_k, x_{k+1})` wherever the running winding is nonzero. Strokes fill this
/// way (a self-overlapping widened outline — e.g. a back-tracking polyline — must not cancel).
pub fn rasterize_winding(edges: &[Edge], w: i32, h: i32) -> Vec<Vec<i32>> {
    let mut cross: Vec<Vec<(i32, i32)>> = vec![Vec::new(); h.max(0) as usize];
    for e in edges {
        let mut x = e.x;
        let mut err = e.err;
        for y in e.y0..e.y1 {
            if (0..h).contains(&y) {
                cross[y as usize].push((x, e.w));
            }
            x += e.dx;
            err += e.errn;
            if err >= 0 {
                x += 1;
                err -= e.dy;
            }
        }
    }
    cross
        .into_iter()
        .map(|mut xs| {
            xs.sort_unstable_by_key(|&(x, _)| x);
            let mut px = Vec::new();
            let mut wind = 0;
            for k in 0..xs.len().saturating_sub(1) {
                wind += xs[k].1;
                if wind != 0 {
                    for x in xs[k].0..xs[k + 1].0 {
                        if (0..w).contains(&x) {
                            px.push(x);
                        }
                    }
                }
            }
            px
        })
        .collect()
}

/// Widen an open polyline into its stroke outline — GDI+'s pen widener for the common case: solid
/// pen, **miter** join, **flat** cap (the System.Drawing `DrawLines` default). Each segment is offset
/// ±w/2 along its right-perpendicular; the outer corner takes the miter intersection (1 point), the
/// inner corner bevels (2 points); flat caps close the two sides directly. Verified against
/// `GdipWidenPath` on seg/V/collinear. Zero-length segments are dropped (matching GDI+). Fill the
/// result with `rasterize_winding`. Points are pre-transformed to device space by the caller.
/// Widen an open polyline into its GDI+ pen outline. `join` 0=miter/1=bevel (round is a follow-up),
/// flat caps. Device-space float vertices in the order `GdipWidenPath` emits (outer forward, inner
/// reverse), to be snapped and filled by nonzero winding. Byte-exact vs native for miter and bevel.
/// ponytail: square/round caps are follow-ups — native's playback cap widening differs from
/// `GdipWidenPath` (the public API) and needs a closer study of playback stroking to pin byte-exact.
pub fn widen_stroke(pts: &[(f64, f64)], w: f64, join: u8) -> Vec<(f64, f64)> {
    let hw = w * 0.5;
    // per-segment (normal.x, normal.y, unitdir.x, unitdir.y); drop zero-length segments
    let mut segs: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut p: Vec<(f64, f64)> = vec![pts[0]];
    for i in 0..pts.len().saturating_sub(1) {
        let (dx, dy) = (pts[i + 1].0 - pts[i].0, pts[i + 1].1 - pts[i].1);
        let l = (dx * dx + dy * dy).sqrt();
        if l == 0.0 {
            continue;
        }
        segs.push((dy / l * hw, -dx / l * hw, dx / l, dy / l));
        p.push(pts[i + 1]);
    }
    let n = p.len();
    if n < 2 {
        return Vec::new();
    }
    let off = |side: f64, i: usize, vi: usize| (p[vi].0 + side * segs[i].0, p[vi].1 + side * segs[i].1);
    let isect = |p0: (f64, f64), d0: (f64, f64), p1: (f64, f64), d1: (f64, f64)| -> Option<(f64, f64)> {
        let den = d0.0 * d1.1 - d0.1 * d1.0;
        if den.abs() < 1e-12 {
            return None;
        }
        let t = ((p1.0 - p0.0) * d1.1 - (p1.1 - p0.1) * d1.0) / den;
        Some((p0.0 + t * d0.0, p0.1 + t * d0.1))
    };
    // One join at vertex `vi`: outer turn gets a miter tip (join 0) or the two offset points (bevel,
    // join 1); an inner turn, or a degenerate miter, is always the two offset points. ponytail: round
    // joins (join 2) are a follow-up; treated as bevel here.
    let join_pts = |side: f64, vi: usize| -> Vec<(f64, f64)> {
        let a = off(side, vi - 1, vi);
        let b = off(side, vi, vi);
        let (da, db) = ((segs[vi - 1].2, segs[vi - 1].3), (segs[vi].2, segs[vi].3));
        if join == 0 && (da.0 * db.1 - da.1 * db.0) * side > 0.0 {
            if let Some(ip) = isect(a, da, b, db) {
                return vec![ip];
            }
        }
        vec![a, b] // bevel / inner / degenerate miter
    };
    let mut out = vec![off(1.0, 0, 0)]; // top-start
    for vi in 1..n - 1 {
        out.extend(join_pts(1.0, vi));
    }
    out.push(off(1.0, n - 2, n - 1)); // top-end
    out.push(off(-1.0, n - 2, n - 1)); // bottom-end (flat end cap closes the two sides directly)
    for vi in (1..n - 1).rev() {
        // right side runs end→start, so its join points come in reverse of the forward [prev,next]
        out.extend(join_pts(-1.0, vi).into_iter().rev());
    }
    out.push(off(-1.0, 0, 0)); // bottom-start (flat start cap closes back to out[0])
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Native oracle (`or_A.bgra`, FillEllipse 12,8,38,30): per filled row the inclusive pixel run.
    const ORACLE: &[(i32, i32, i32)] = &[
        (8, 26, 35), (9, 23, 38), (10, 21, 41), (11, 19, 42), (12, 18, 44), (13, 17, 45),
        (14, 15, 46), (15, 15, 46), (16, 14, 47), (17, 13, 48), (18, 13, 48), (19, 13, 48),
        (20, 12, 49), (21, 12, 49), (22, 12, 49), (23, 12, 49), (24, 12, 49), (25, 12, 49),
        (26, 13, 48), (27, 13, 48), (28, 13, 48), (29, 14, 47), (30, 15, 46), (31, 15, 46),
        (32, 16, 45), (33, 17, 44), (34, 19, 42), (35, 21, 41), (36, 23, 38), (37, 26, 35),
    ];

    fn approx(got: &[(f64, f64)], want: &[(f64, f64)]) {
        assert_eq!(got.len(), want.len(), "vertex count: {got:?}");
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g.0 - w.0).abs() < 0.01 && (g.1 - w.1).abs() < 0.01, "{g:?} vs {w:?}");
        }
    }

    // Oracles below are GdipWidenPath outlines of a width-6 pen.
    #[test]
    fn widen_miter_matches_gdiplus() {
        // V polyline (8,30)-(32,8)-(56,30), miter pen: outer corner = miter tip, inner = bevel, flat caps.
        let got = widen_stroke(&[(8.0, 30.0), (32.0, 8.0), (56.0, 30.0)], 6.0, 0);
        approx(&got, &[
            (5.973, 27.789), (32.0, 3.93), (58.027, 27.789),
            (53.973, 32.211), (29.973, 10.211), (34.027, 10.211), (10.027, 32.211),
        ]);
    }

    #[test]
    fn widen_bevel_matches_gdiplus() {
        // Same V, bevel join: the outer miter tip becomes the two offset points (a,b).
        let got = widen_stroke(&[(8.0, 30.0), (32.0, 8.0), (56.0, 30.0)], 6.0, 1);
        approx(&got, &[
            (5.973, 27.789), (29.973, 5.789), (34.027, 5.789), (58.027, 27.789),
            (53.973, 32.211), (29.973, 10.211), (34.027, 10.211), (10.027, 32.211),
        ]);
    }

    #[test]
    fn ellipse_byte_exact() {
        // Full pipeline in Rust with the DERIVED device transform: scale 16·96/dpi per axis +
        // PixelOffsetMode Half, dpi from the reference header (szlDevice 1024×768 / szlMM 271×203).
        let (dpix, dpiy) = ((1024.0 / 271.0) * 25.4_f32, (768.0 / 203.0) * 25.4_f32);
        let (sx, sy) = device_scale((dpix, dpiy));
        let poly = flatten_ellipse([12.0, 8.0, 38.0, 30.0], sx, sy, HALF_PIXEL_X16, HALF_PIXEL_X16);
        let rows = rasterize(&init_edges(&poly), 64, 48);
        for y in 0..48usize {
            let want: Vec<i32> = ORACLE
                .iter()
                .find(|&&(ry, ..)| ry as usize == y)
                .map(|&(_, l, r)| (l..=r).collect())
                .unwrap_or_default();
            assert_eq!(rows[y], want, "row {y} mismatch");
        }
    }
}
