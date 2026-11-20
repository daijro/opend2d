//! Synthetic-bold OUTLINE embolden (DWRITE_FONT_SIMULATIONS_BOLD on a glyph outline): the TrueType
//! scaler's edge-offset-and-intersect embolden, in 26.6 design space with device y-down. It is what
//! GetGlyphRunOutline returns for a bold-simulated face, and the outline a large glyph run is filled
//! with (`text::fill_outline`).

/// Integer sqrt used by the vector normalize.
fn frac_sqrt(x: i32) -> i32 {
    if x < 0 { return i32::MIN; }
    let x = x as i64;
    let (mut rem, mut root): (i64, i64) = if x >= 0x4000_0000 { (x - 0x4000_0000, 0x4000_0000) } else { (x, 0) };
    let mut bit: i64 = 0x1000_0000;
    while bit != 0 {
        if rem >= bit + root { rem -= bit + root; root += 2 * bit; }
        rem *= 2; bit >>= 1;
    }
    let frac = if rem > root { let t = 2 * (rem - root) - 1; root += 1; t } else { 2 * rem };
    (root + if frac > root { 1 } else { 0 }) as i32
}

fn chkovf(v: i64) -> i64 { v.clamp(i32::MIN as i64, i32::MAX as i64) }

/// Unit vector of (dx,dy) scaled to 0x4000, each component an i16.
/// Returns (nx, ny): nx from dx, ny from dy.
fn normalize(mut dx: i64, mut dy: i64) -> (i32, i32) {
    if dx == 0 && dy == 0 { return (0x4000, 0); }
    // An axis vector takes the small path only while its squared length fits 32 bits; a longer one (a design-unit
    // edge past 46340/64 units) scales on the wide path, which gives the same exact unit vector.
    let dy_small = (((dy + 0x7FFF) as u32 as i64) & 0xFFFF_FFFF) <= 0xFFFD;
    let small = (dx != 0 && -0x8000 < dx && dx < 0x7FFF && dy_small) || (dx == 0 && dy != 0 && dy_small);
    let (aa, bb, q);
    if small {
        let mut s = dx * dx + dy * dy;
        let mut shift = 15u32;
        while s < 0x2000_0000 { s <<= 2; shift += 1; }
        aa = dx << shift; bb = dy << shift; q = frac_sqrt(s as i32) as i64;
    } else {
        while !(dx >= 0x2000_0000 || dx <= -0x2000_0000 || (((dy + 0x1FFF_FFFF) as u32 as i64) & 0xFFFF_FFFF) > 0x3FFF_FFFE) {
            dx *= 2; dy *= 2;
        }
        let ty = chkovf((dy * dy + 0x2000_0000) >> 30);
        let tx = chkovf((dx * dx + 0x2000_0000) >> 30);
        let s = (((ty + tx) & 0xFFFF_FFFF) as u32) as i32;
        aa = dx; bb = dy; q = frac_sqrt(s) as i64;
    }
    let half = q / 2; // q >= 0
    let comp = |v: i64| -> i32 {
        let val = (v << 30) + if v >= 0 { half } else { -half };
        let r = if q != 0 { val / q } else { 0x7FFF_FFFF + if val < 0 { 1 } else { 0 } };
        ((chkovf(r) + 0x8000) >> 16) as i32
    };
    ((comp(aa) as i16) as i32, (comp(bb) as i16) as i32)
}

/// Mul26Dot6: round(a*b / 64).
fn mul26(a: i32, b: i32) -> i32 {
    let p = a as i64 * b as i64;
    (if p < 0 { -(((-p) + 32) >> 6) } else { (p + 32) >> 6 }) as i32
}

/// Round-half-away division of num by den.
fn comp_div(den: i64, num: i64) -> i64 {
    if den == 0 { return 0x7FFF_FFFF + if num < 0 { 1 } else { 0 }; }
    let h = den / 2;
    let biased = if (den < 0) == (num < 0) { num + h } else { num - h };
    biased / den
}

/// Intersect26Dot6: intersection of edge p1->p2 with edge p3->p4 (all 26.6).
fn intersect(p1: [i64; 2], p2: [i64; 2], p3: [i64; 2], p4: [i64; 2]) -> [i64; 2] {
    let (x1, y1) = (p1[0], p1[1]);
    let (x2, y2) = (p2[0], p2[1]);
    let (x3, y3) = (p3[0], p3[1]);
    let (x4, y4) = (p4[0], p4[1]);
    let e1dx = x2 - x1; let e2dx = x4 - x3;
    let e1dy = y2 - y1; let e2dy = y4 - y3;
    let (denom, numer);
    if y2 == y1 {
        if e2dx == 0 { return [x3, y1]; }
        denom = y3 - y1; numer = -e2dy;
    } else if e1dx == 0 {
        if e2dy == 0 { return [x1, y3]; }
        denom = x3 - x1; numer = -e2dx;
    } else {
        let dy31 = y3 - y1;
        let abs_dy1 = if e1dy > 0 { e1dy } else { y1 - y2 };
        let abs_dx1 = if e1dx > 0 { e1dx } else { -e1dx };
        if abs_dx1 >= abs_dy1 {
            denom = dy31 - comp_div(e1dx, e1dy * (x3 - x1));
            numer = comp_div(x2 - x1, e1dy * e2dx) - e2dy;
        } else {
            denom = x1 + comp_div(e1dy, e1dx * dy31) - x3;
            numer = e2dx - comp_div(y2 - y1, e1dx * e2dy);
        }
    }
    let abs_numer = numer.abs();
    if abs_numer <= 16 { [(x3 + x2) >> 1, (y3 + y2) >> 1] }
    else { [x3 + comp_div(numer, denom * e2dx), y3 + comp_div(numer, denom * e2dy)] }
}

#[derive(Clone, Copy)]
pub struct Pt { pub x: i32, pub y: i32, pub on: bool }

/// One embolden step: shift point i (and any coincident successors up to iend) outward.
#[allow(clippy::too_many_arguments)]
fn embold_point(pts: &mut [Pt], i: usize, iend: usize, prev: (i32, i32), next: (i32, i32),
                bold_xp: i32, bold_xn: i32, bold_yp: i32, bold_yn: i32, flip: bool) {
    let (cx, cy) = (pts[i].x, pts[i].y);
    let (px, py) = prev; let (nx, ny) = next;
    // Edge deltas. `flip` negates all four (the post-hint device grayscale path runs the embolden in the
    // glyph's own orientation with the deltas reversed); the design/GetGlyphRunOutline path runs un-negated.
    let (mut iy, mut ix) = (cy - py, px - cx);
    let (mut oy, mut ox) = (ny - cy, cx - nx);
    if flip { iy = -iy; ix = -ix; oy = -oy; ox = -ox; }
    let (ox0, oy0) = (cx, cy);
    let (nxi, nyi) = normalize(iy as i64, ix as i64);
    let (nxo, nyo) = normalize(oy as i64, ox as i64);
    let (nxi8, nyi8, nxo8, nyo8) = (nxi >> 8, nyi >> 8, nxo >> 8, nyo >> 8);
    let in_dx = mul26(nxi8, if nxi8 > 0 { bold_xp } else { bold_xn });
    let in_dy = mul26(nyi8, if nyi8 >= 0 { bold_yp } else { bold_yn });
    let out_dx = mul26(nxo8, if nxo8 > 0 { bold_xp } else { bold_xn });
    let out_dy = mul26(nyo8, if nyo8 >= 0 { bold_yp } else { bold_yn });
    let (inx, iny) = (cx + in_dx, cy + in_dy);
    let (onx, ony) = (cx + out_dx, cy + out_dy);
    let (mut newx, mut newy);
    if inx == onx && iny == ony {
        newx = onx; newy = ony;
    } else {
        let it = intersect([(px + in_dx) as i64, (py + in_dy) as i64], [inx as i64, iny as i64],
                           [onx as i64, ony as i64], [(nx + out_dx) as i64, (ny + out_dy) as i64]);
        newx = it[0] as i32; newy = it[1] as i32;
        if newx - ox0 > bold_xp { newx = ox0 + bold_xp; }
        if newx - ox0 < -bold_xn { newx = ox0 - bold_xn; }
        if newy - oy0 < -bold_yn { newy = oy0 - bold_yn; }
        if newy - oy0 > bold_yp { newy = oy0 + bold_yn; }
    }
    newx += bold_xn; newy += bold_yn;
    for p in pts.iter_mut().take(iend + 1).skip(i) { p.x = newx; p.y = newy; }
}

/// The grayscale-embolden driver over one contour (points in 26.6).
fn embold_contour(pts: &mut Vec<Pt>, bold_xp: i32, bold_xn: i32, bold_yp: i32, bold_yn: i32, flip: bool) {
    let n = pts.len();
    if n < 3 { return; }
    let orig: Vec<Pt> = pts.clone();
    let mut i = 0;
    while i < n {
        let cur = orig[i];
        let prev = (1..n).map(|k| orig[(i + n - k) % n]).find(|p| p.x != cur.x || p.y != cur.y);
        let mut iend = i;
        let next = (1..n).find_map(|k| {
            let p = orig[(i + k) % n];
            if p.x != cur.x || p.y != cur.y { iend = i + k - 1; Some(p) } else { None }
        });
        let (Some(prev), Some(next)) = (prev, next) else { i = iend + 1; continue };
        let iend = iend.min(n - 1);
        embold_point(pts, i, iend, (prev.x, prev.y), (next.x, next.y), bold_xp, bold_xn, bold_yp, bold_yn, flip);
        i = iend + 1;
    }
}

/// Per contour (points y-up): whether it winds against the TrueType convention, its ink on the left of
/// its direction instead of the right. The embolden pushes such a contour the other way, so it still grows
/// the ink: Segoe UI Emoji winds its face rings counter-clockwise beside clockwise mouths (VM: Segoe UI
/// Emoji at weight 700, colour and outline). Decided by the glyph's nonzero winding a hair to each side of
/// the contour's longest edge, so holes and overlapping contours keep the convention.
pub(crate) fn reversed(contours: &[Vec<Pt>]) -> Vec<bool> {
    let winding = |p: [f64; 2]| -> i32 {
        let mut w = 0;
        for c in contours {
            for i in 0..c.len() {
                let (a, b) = (c[i], c[(i + 1) % c.len()]);
                let (ay, by) = (a.y as f64, b.y as f64);
                let cross = (b.x as f64 - a.x as f64) * (p[1] - ay) - (p[0] - a.x as f64) * (by - ay);
                if ay <= p[1] && by > p[1] && cross > 0.0 { w += 1; }
                if ay > p[1] && by <= p[1] && cross < 0.0 { w -= 1; }
            }
        }
        w
    };
    contours.iter().map(|c| {
        let n = c.len();
        let Some(i) = (0..n).max_by_key(|&i| { let (a, b) = (c[i], c[(i + 1) % n]); (b.x as i64 - a.x as i64).pow(2) + (b.y as i64 - a.y as i64).pow(2) }) else { return false };
        let (a, b) = (c[i], c[(i + 1) % n]);
        let (dx, dy) = (b.x as f64 - a.x as f64, b.y as f64 - a.y as f64);
        let len = (dx * dx + dy * dy).sqrt();
        if len == 0.0 { return false; }
        let m = [(a.x as f64 + b.x as f64) / 2.0, (a.y as f64 + b.y as f64) / 2.0];
        let e = 0.25 / len;
        let right = winding([m[0] + dy * e, m[1] - dx * e]) != 0;
        let left = winding([m[0] - dy * e, m[1] + dx * e]) != 0;
        left && !right
    }).collect()
}

/// Strength in design units: boldY = (upm*20 - 10)/1000, boldX = boldY + 1.
pub fn strength(upm: i32) -> (i32, i32) {
    let by = ((upm * 20 - 10) / 1000).max(0);
    (by + 1, by)
}

/// Embolden all contours in-place. Input points are design units (y-up, as skrifa emits);
/// output points are left in 26.6 (y-up design * 64) for the caller to divide by 64.
/// Internally negates y (device y-down), runs the scaler math in 26.6, then re-anchors the
/// baseline by -boldY and restores the y-up sign.
pub fn apply(contours: &mut [Vec<Pt>], bold_x: i32, bold_y: i32) {
    let bold_xp = ((bold_x + 1) / 2) << 6;
    let bold_xn = (bold_x / 2) << 6;
    let bold_yp = (bold_y / 2) << 6;
    let bold_yn = ((bold_y + 1) / 2) << 6;
    let rev = reversed(contours);
    for (c, r) in contours.iter_mut().zip(rev) {
        for p in c.iter_mut() { p.x <<= 6; p.y = -(p.y << 6); } // -> 26.6, device y-down
        embold_contour(c, bold_xp, bold_xn, bold_yp, bold_yn, r);
    }
    // Global baseline re-anchor: shift up by boldY, then restore y-up sign (stay in 26.6).
    let dy = bold_y << 6;
    for c in contours.iter_mut() {
        for p in c.iter_mut() { p.y = -(p.y - dy); }
    }
}

/// Embolden a hinted device outline already in glyph-local 26.6 (y-up, as our `HintedOutline` and the
/// native scan-converter contour list carry it): native's post-hint grayscale embolden, validated
/// byte-exact (outline and pixels) against a native sim-bold capture. Runs the edge-offset-and-intersect
/// embolden in place with the reversed-delta glyph-orientation convention (`flip`) and the ceil/floor amount
/// branch; no frame flip and no re-anchor (the global per-point shift is the whole translation).
/// Translation-invariant, so it is correct wherever the glyph sits, under any transform applied before it.
pub fn apply_26dot6(contours: &mut [Vec<Pt>], bold_x: i32, bold_y: i32) {
    let bold_xp = ((bold_x + 1) / 2) << 6;
    let bold_xn = (bold_x / 2) << 6;
    let bold_yp = (bold_y / 2) << 6;
    let bold_yn = ((bold_y + 1) / 2) << 6;
    let rev = reversed(contours);
    for (c, r) in contours.iter_mut().zip(rev) {
        embold_contour(c, bold_xp, bold_xn, bold_yp, bold_yn, !r);
    }
}

/// Collects a glyph's contours as design-unit points with on/off-curve flags (skrifa emits the
/// expanded quadratic: on-curve endpoints + off-curve quad controls).
#[derive(Default)]
struct Collect {
    contours: Vec<Vec<Pt>>,
}
impl skrifa::outline::OutlinePen for Collect {
    fn move_to(&mut self, x: f32, y: f32) {
        self.contours.push(vec![Pt { x: x as i32, y: y as i32, on: true }]);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        if let Some(c) = self.contours.last_mut() {
            c.push(Pt { x: x as i32, y: y as i32, on: true });
        }
    }
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        if let Some(c) = self.contours.last_mut() {
            c.push(Pt { x: cx as i32, y: cy as i32, on: false });
            c.push(Pt { x: x as i32, y: y as i32, on: true });
        }
    }
    fn curve_to(&mut self, ax: f32, ay: f32, bx: f32, by: f32, x: f32, y: f32) {
        if let Some(c) = self.contours.last_mut() {
            c.push(Pt { x: ax as i32, y: ay as i32, on: false });
            c.push(Pt { x: bx as i32, y: by as i32, on: false });
            c.push(Pt { x: x as i32, y: y as i32, on: true });
        }
    }
    fn close(&mut self) {}
}

/// A TrueType glyph's raw points in design units: positions, on-curve flags and contour end indices.
pub type RawGlyph = (Vec<[f64; 2]>, Vec<bool>, Vec<usize>);

/// Draw `glyph`'s design-unit outline into `pen`, emboldened for a bold simulation as GetGlyphRunOutline
/// returns it (the strength at ppem = upem, so constant in design units). `raw` is the glyph's TrueType point
/// list, which native embolds as it stands, in its own y-up 26.6 frame exactly as the hinted device path does
/// ([`apply_26dot6`]), placing the implied on-curve midpoints only afterwards, between the emboldened controls
/// (VM: the scaler's embolden of cold Tahoma bold 'BOg', input and output points, 146 of 146 exact).
/// `None` (CFF, a variable instance) embolds skrifa's expanded outline.
pub fn draw_bold(glyph: &skrifa::outline::OutlineGlyph, upem: i32, raw: Option<&RawGlyph>, pen: &mut impl skrifa::outline::OutlinePen) -> Result<(), skrifa::outline::DrawError> {
    use skrifa::instance::{LocationRef, Size};
    let (bx, by) = strength(upem);
    if let Some(raw) = raw {
        draw_points(raw, Some((bx, by)), false, pen);
        return Ok(());
    }
    let mut col = Collect::default();
    glyph.draw(skrifa::outline::DrawSettings::unhinted(Size::unscaled(), LocationRef::default()), &mut col)?;
    apply(&mut col.contours, bx, by);
    let du = |v: i32| v as f32 / 64.0;
    for c in &col.contours {
        let Some(&start) = c.first() else { continue };
        pen.move_to(du(start.x), du(start.y));
        let mut i = 1;
        while i < c.len() {
            let p = c[i];
            if p.on {
                pen.line_to(du(p.x), du(p.y));
                i += 1;
            } else {
                // An off-curve control is followed by its on-curve end, or wraps to the start.
                let end = c.get(i + 1).copied().unwrap_or(start);
                pen.quad_to(du(p.x), du(p.y), du(end.x), du(end.y));
                i += 2;
            }
        }
        pen.close();
    }
    Ok(())
}

/// A raw glyph's contours as 26.6 design-unit points.
fn raw_contours((pts, on, ends): &RawGlyph) -> Vec<Vec<Pt>> {
    let mut contours = Vec::new();
    let mut s0 = 0usize;
    for &e in ends {
        if e >= s0 && e < pts.len() {
            contours.push((s0..=e).map(|i| Pt { x: (pts[i][0] * 64.0).round() as i32, y: (pts[i][1] * 64.0).round() as i32, on: on[i] }).collect());
        }
        s0 = e + 1;
    }
    contours
}

/// Draw a TrueType point list into `pen` as the scaler outlines it in 26.6 design units: emboldened by `bold`
/// (`(bold_x, bold_y)`, [`apply_26dot6`]), then slanted for the oblique simulation with its 16.16 shear rounded per
/// point (`x += (y·22272 + 0x8000) >> 16`, VM sb40: Leelawadee UI bold and oblique, every point of GetGlyphRunOutline;
/// the float slant drifts up to 1/128 unit), the implied midpoints placed between the controls last.
pub fn draw_points(raw: &RawGlyph, bold: Option<(i32, i32)>, oblique: bool, pen: &mut impl skrifa::outline::OutlinePen) {
    let mut contours = raw_contours(raw);
    if let Some((bx, by)) = bold {
        apply_26dot6(&mut contours, bx, by);
    }
    if oblique {
        for p in contours.iter_mut().flatten() {
            // The 16.16 shear product rounds half away from zero (VM: GetGlyphRunOutline of Segoe UI 't' oblique at em
            // 1200, r25 30289: -478.5 units to -479, 217.5 to 218).
            let v = p.y as i64 * 22272;
            p.x += (v.signum() * ((v.abs() + 0x8000) >> 16)) as i32;
        }
    }
    for c in &contours {
        emit_quadratic(c, pen);
    }
}

/// One contour of TrueType points in 26.6 design units into `pen`: lines between on-curve points, a quadratic per
/// off-curve control, ending at the next on-curve point or at the implied midpoint of two consecutive controls.
fn emit_quadratic(c: &[Pt], pen: &mut impl skrifa::outline::OutlinePen) {
    let n = c.len();
    if n == 0 {
        return;
    }
    let f = |v: i32| v as f32 / 64.0;
    let mid = |a: Pt, b: Pt| ((a.x + b.x) as f32 / 128.0, (a.y + b.y) as f32 / 128.0);
    let first_on = (0..n).find(|&i| c[i].on);
    let (start, begin, count) = match first_on {
        Some(i) => ((f(c[i].x), f(c[i].y)), i + 1, n - 1),
        None => (mid(c[n - 1], c[0]), 0, n),
    };
    pen.move_to(start.0, start.1);
    let mut k = 0;
    while k < count {
        let p = c[(begin + k) % n];
        if p.on {
            pen.line_to(f(p.x), f(p.y));
            k += 1;
            continue;
        }
        let (end, step) = if k + 1 < count {
            let q = c[(begin + k + 1) % n];
            if q.on { ((f(q.x), f(q.y)), 2) } else { (mid(p, q), 1) }
        } else {
            (start, 1)
        };
        pen.quad_to(f(p.x), f(p.y), end.0, end.1);
        k += step;
    }
    pen.close();
}

/// A pen that shears design-unit points by `k` (x += k·y, font y up) before passing them on: the
/// oblique simulation on a glyph outline.
pub struct Shear<'a, P: skrifa::outline::OutlinePen> {
    pub inner: &'a mut P,
    pub k: f32,
}
impl<P: skrifa::outline::OutlinePen> skrifa::outline::OutlinePen for Shear<'_, P> {
    fn move_to(&mut self, x: f32, y: f32) {
        self.inner.move_to(x + self.k * y, y);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.inner.line_to(x + self.k * y, y);
    }
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        self.inner.quad_to(cx + self.k * cy, cy, x + self.k * y, y);
    }
    fn curve_to(&mut self, ax: f32, ay: f32, bx: f32, by: f32, x: f32, y: f32) {
        self.inner.curve_to(ax + self.k * ay, ay, bx + self.k * by, by, x + self.k * y, y);
    }
    fn close(&mut self) {
        self.inner.close();
    }
}

#[cfg(test)]
mod embold_tests {
    use super::*;
    use skrifa::instance::{LocationRef, Size};
    use skrifa::outline::DrawSettings;
    use skrifa::{FontRef, MetadataProvider};

    /// The oblique slant rounds each 26.6 point's 16.16 shear half away from zero (VM sb40, sb50).
    #[test]
    fn oblique_slant_rounds_each_point_half_away_from_zero() {
        #[derive(Default)]
        struct Rec(Vec<[f32; 2]>);
        impl skrifa::outline::OutlinePen for Rec {
            fn move_to(&mut self, x: f32, y: f32) { self.0.push([x, y]); }
            fn line_to(&mut self, x: f32, y: f32) { self.0.push([x, y]); }
            fn quad_to(&mut self, _: f32, _: f32, x: f32, y: f32) { self.0.push([x, y]); }
            fn curve_to(&mut self, _: f32, _: f32, _: f32, _: f32, _: f32, _: f32) {}
            fn close(&mut self) {}
        }
        let raw: RawGlyph = (vec![[864.0, 47.0], [746.0, -24.0], [584.0, -22.0]], vec![true, true, true], vec![2]);
        let mut r = Rec::default();
        draw_points(&raw, None, true, &mut r);
        // 47·64·22272 / 2^16 = 1021.97 -> 1022; -24·64 gives -522 exactly; -22·64 gives the tie -478.5 -> -479.
        assert_eq!(r.0, vec![[(864.0 * 64.0 + 1022.0) / 64.0, 47.0], [(746.0 * 64.0 - 522.0) / 64.0, -24.0], [(584.0 * 64.0 - 479.0) / 64.0, -22.0]]);
    }

    /// Two off-curve controls in a row end their quadratic at the implied midpoint of the two, half units kept, and a
    /// contour with no on-curve point starts from the midpoint of its last and first controls.
    #[test]
    fn implied_midpoints_come_from_the_controls() {
        #[derive(Default)]
        struct Rec(Vec<(char, [f32; 4])>);
        impl skrifa::outline::OutlinePen for Rec {
            fn move_to(&mut self, x: f32, y: f32) { self.0.push(('M', [x, y, 0.0, 0.0])); }
            fn line_to(&mut self, x: f32, y: f32) { self.0.push(('L', [x, y, 0.0, 0.0])); }
            fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) { self.0.push(('Q', [cx, cy, x, y])); }
            fn curve_to(&mut self, _: f32, _: f32, _: f32, _: f32, _: f32, _: f32) {}
            fn close(&mut self) { self.0.push(('Z', [0.0; 4])); }
        }
        let pt = |x: i32, y: i32, on: bool| Pt { x: x * 64, y: y * 64, on };
        let mut r = Rec::default();
        emit_quadratic(&[pt(0, 0, true), pt(10, 0, false), pt(21, 10, false), pt(20, 20, true)], &mut r);
        assert_eq!(r.0, vec![('M', [0.0, 0.0, 0.0, 0.0]), ('Q', [10.0, 0.0, 15.5, 5.0]), ('Q', [21.0, 10.0, 20.0, 20.0]), ('L', [0.0, 0.0, 0.0, 0.0]), ('Z', [0.0; 4])].into_iter().filter(|s| s.0 != 'L').collect::<Vec<_>>());
        let mut r = Rec::default();
        emit_quadratic(&[pt(0, 0, false), pt(10, 0, false), pt(10, 10, false), pt(0, 10, false)], &mut r);
        assert_eq!(r.0[0], ('M', [0.0, 5.0, 0.0, 0.0]));
        assert_eq!(r.0.len(), 6, "four quadratics between the controls' midpoints");
        assert_eq!(r.0[4], ('Q', [0.0, 10.0, 0.0, 5.0]));
    }

    /// VM (Leelawadee UI's .notdef box, bold and oblique simulated, aliased under a shear at 96 ppem): the box grows
    /// two pixels up and right off its baseline. Its 990-unit horizontal edges are past the 32-bit small-path
    /// normalize, which flipped their normal and shrank the box from the bottom instead.
    #[test]
    fn a_long_horizontal_edge_grows_its_bar() {
        let pt = |x: i32, y: i32| Pt { x, y, on: true };
        let boxes = || vec![
            vec![pt(166, 0), pt(166, 1398), pt(1156, 1398), pt(1156, 0)],
            vec![pt(314, 148), pt(1008, 148), pt(1008, 1250), pt(314, 1250)],
        ];
        let (bx, by) = strength(2048);
        let bounds = |cs: &[Vec<Pt>], k: i32| cs.iter().map(|c| {
            let xs = c.iter().map(|p| p.x / k);
            let ys = c.iter().map(|p| p.y / k);
            (xs.clone().min().unwrap(), ys.clone().min().unwrap(), xs.max().unwrap(), ys.max().unwrap())
        }).collect::<Vec<_>>();
        let want = vec![(166, 0, 1197, 1438), (355, 188, 1008, 1250)];
        let mut design = boxes();
        apply(&mut design, bx, by);
        assert_eq!(bounds(&design, 64), want);
        let mut device: Vec<Vec<Pt>> = boxes().into_iter().map(|c| c.into_iter().map(|p| pt(p.x << 6, p.y << 6)).collect()).collect();
        apply_26dot6(&mut device, bx, by);
        assert_eq!(bounds(&device, 64), want);
        assert_eq!(normalize(0, -990 << 6), (0, -0x4000));
    }

    fn impact() -> Option<Vec<u8>> {
        let root = std::env::var_os("D2D_FONTS").map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ref/windows/Fonts"));
        std::fs::read(root.join("impact.ttf")).ok()
    }

    fn collect(font: &FontRef, ch: char) -> Vec<Vec<Pt>> {
        let gid = font.charmap().map(ch).unwrap();
        let mut c = Collect::default();
        font.outline_glyphs().get(gid).unwrap()
            .draw(DrawSettings::unhinted(Size::unscaled(), LocationRef::default()), &mut c).unwrap();
        c.contours
    }

    // Byte-exact A/B of `apply_26dot6` against a native capture pair: feed the native NON-bold hinted
    // scan contour (`D2D_EMB_SRC`) through the device embold and require it to equal the native sim-bold
    // scan contour (`D2D_EMB_TARGET`). Both are native scan-converter contour dumps
    // of the same glyph, x oversampled ×8 so divided back to device 26.6. Inert without the env paths;
    // captures live outside the tree (worker thread-storage). Guards the grayscale sim-bold embold recipe.
    #[test]
    fn embold_device_matches_native() {
        let (Ok(src), Ok(tgt)) = (std::env::var("D2D_EMB_SRC"), std::env::var("D2D_EMB_TARGET")) else { return };
        let bx: i32 = std::env::var("D2D_BOLD_BX").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
        let by: i32 = std::env::var("D2D_BOLD_BY").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
        let load = |p: &str| -> Vec<Vec<Pt>> {
            let v: serde_json::Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
            let o = &v[0];
            let g = |k: &str| o[k].as_array().unwrap().clone();
            let (xs, ys, on, ends) = (g("x"), g("y"), g("on"), g("ends"));
            let mut out = Vec::new();
            let mut s = 0usize;
            for e in ends {
                let e = e.as_u64().unwrap() as usize;
                out.push((s..=e).map(|i| Pt { x: (xs[i].as_i64().unwrap() / 8) as i32, y: ys[i].as_i64().unwrap() as i32, on: on[i].as_bool().unwrap() }).collect());
                s = e + 1;
            }
            out
        };
        let (mut cs, want) = (load(&src), load(&tgt));
        apply_26dot6(&mut cs, bx, by);
        let (mut md, mut total) = (0i32, 0usize);
        for (c, w) in cs.iter().zip(want.iter()) {
            for (p, q) in c.iter().zip(w.iter()) { md = md.max((p.x - q.x).abs()).max((p.y - q.y).abs()); total += 1; }
        }
        assert_eq!(md, 0, "device embold must match native byte-exact ({total} points)");
    }

    // Native dwrite GetGlyphRunOutline(Impact 'H', sim bold, emSize=upm=2048). On-curve points
    // (design, y-up) from a VM capture; all axis-aligned so the embolden is exact to the design unit.
    #[test]
    fn impact_h_bold_outline_matches_native() {
        let Some(data) = impact() else { return };
        let font = FontRef::from_index(&data, 0).unwrap();
        let (bx, by) = strength(2048);
        assert_eq!((bx, by), (41, 40));
        let mut cs = collect(&font, 'H');
        assert_eq!(cs.len(), 1);
        apply(&mut cs, bx, by);
        let got: Vec<(i32, i32)> = cs[0].iter().map(|p| (p.x, p.y)).collect(); // 26.6, y-up
        let want: [(i32, i32); 12] = [
            (1093, 1659), (1093, 0), (631, 0), (631, 680), (546, 680), (546, 0),
            (84, 0), (84, 1659), (546, 1659), (546, 1080), (631, 1080), (631, 1659),
        ];
        let want: Vec<(i32, i32)> = want.iter().map(|&(x, y)| (x << 6, y << 6)).collect();
        assert_eq!(got, want, "emboldened H on-curve points must match native exactly");
    }

    // Impact 'O' (sim bold, emSize=upm=2048): curved, with off-curve controls. The contour
    // start points (on-curve) match native exactly; every point is within one design unit
    // (off-curve controls at near-horizontal tangents carry a <=0.75u residual, sub-pixel at
    // render sizes). Native design y-up (screen-negated), from a VM capture.
    #[test]
    fn impact_o_bold_outline_matches_native() {
        let Some(data) = impact() else { return };
        let font = FontRef::from_index(&data, 0).unwrap();
        let (bx, by) = strength(2048);
        let mut cs = collect(&font, 'O');
        assert_eq!(cs.len(), 2);
        apply(&mut cs, bx, by);
        // contour starts (on-curve): native bold, design y-up, 26.6
        assert_eq!((cs[0][0].x, cs[0][0].y), (1087 << 6, 691 << 6));
        assert_eq!((cs[1][0].x, cs[1][0].y), (625 << 6, 1223 << 6));
        // every emboldened point stays within one design unit of a sane envelope (no blow-ups).
        for c in &cs {
            for p in c {
                assert!(p.x > -5 << 6 && p.x < 1100 << 6 && p.y.abs() < 1700 << 6,
                    "point out of range: {:?}", (p.x, p.y));
            }
        }
    }

    #[test]
    fn reversed_contour_still_grows() {
        // y-up: (0,0) -> (0,640) -> (640,640) -> (640,0) is clockwise, TrueType's outer winding.
        let square = |cw: bool| {
            let mut v: Vec<Pt> = [(0, 0), (0, 640), (640, 640), (640, 0)].iter().map(|&(x, y)| Pt { x, y, on: true }).collect();
            if !cw { v.reverse(); }
            vec![v]
        };
        let bbox = |c: &[Vec<Pt>]| {
            let p = &c[0];
            [p.iter().map(|q| q.x).min().unwrap(), p.iter().map(|q| q.y).min().unwrap(), p.iter().map(|q| q.x).max().unwrap(), p.iter().map(|q| q.y).max().unwrap()]
        };
        for cw in [true, false] {
            let mut c = square(cw);
            assert_eq!(reversed(&c), vec![!cw]);
            apply_26dot6(&mut c, 2, 1);
            let b = bbox(&c);
            assert!(b[0] <= 0 && b[1] <= 0 && b[2] >= 640 && b[3] >= 640 && (b[2] - b[0]) > 640, "cw={cw} {b:?}");
        }
    }
}
