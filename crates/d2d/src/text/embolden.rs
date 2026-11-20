//! Native glyph emboldening: outline embolden + 26.6 intersection.
use super::hinting::HintedOutline;

fn comp_div(den: i64, num: i64) -> i64 {
    if den == 0 { return if num < 0 { i32::MIN as i64 } else { i32::MAX as i64 }; }
    (num + if (den < 0) == (num < 0) { den / 2 } else { -den / 2 }) / den
}

fn intersect(a: [i64; 2], b: [i64; 2], c: [i64; 2], d: [i64; 2]) -> [i64; 2] {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let (ex, ey) = (d[0] - c[0], d[1] - c[1]);
    let (num, den) = if dy == 0 {
        if ex == 0 { return [c[0], a[1]]; }
        (c[1] - a[1], -ey)
    } else if dx == 0 {
        if ey == 0 { return [a[0], c[1]]; }
        (c[0] - a[0], -ex)
    } else if dx.abs() >= dy.abs() {
        (c[1] - a[1] - comp_div(dx, dy * (c[0] - a[0])), comp_div(dx, dy * ex) - ey)
    } else {
        (a[0] + comp_div(dy, dx * (c[1] - a[1])) - c[0], ex - comp_div(dy, dx * ey))
    };
    if den.abs() <= 16 { [(c[0] + b[0]) >> 1, (c[1] + b[1]) >> 1] }
    else { [c[0] + comp_div(den, num * ex), c[1] + comp_div(den, num * ey)] }
}

/// Native's one-pixel horizontal outline embolden (the special width-one branch).
pub(super) fn horizontal_one_pixel(outline: &mut HintedOutline) {
    // A contour wound against its place in the glyph has its right side on the rising edges.
    let rev = {
        use super::outline_embold::Pt;
        let mut cs: Vec<Vec<Pt>> = Vec::new();
        let mut start = 0usize;
        for &end in &outline.contour_ends {
            let end = end as usize;
            if end >= outline.points.len() || end < start { break; }
            cs.push((start..=end).map(|i| Pt { x: outline.points[i].x, y: outline.points[i].y, on: outline.on_curve[i] }).collect());
            start = end + 1;
        }
        super::outline_embold::reversed(&cs)
    };
    let mut start = 0;
    for (k, &end) in outline.contour_ends.iter().enumerate() {
        let r = rev.get(k).copied().unwrap_or(false);
        let end = end as usize;
        if end >= outline.points.len() || end < start { return; }
        let original: Vec<[i64; 2]> = outline.points[start..=end].iter().map(|p| [p.x as i64, p.y as i64]).collect();
        let n = original.len();
        if n >= 3 {
            for i in 0..n {
                let cur = original[i];
                let Some(prev) = (1..n).map(|k| original[(i + n - k) % n]).find(|p| *p != cur) else { continue };
                let Some(next) = (1..n).map(|k| original[(i + k) % n]).find(|p| *p != cur) else { continue };
                let left = if (prev[1] > cur[1]) != r && prev[1] != cur[1] { 64 } else { 0 };
                let right = if (cur[1] > next[1]) != r && cur[1] != next[1] { 64 } else { 0 };
                // Only the point moves: each edge pivots on its unmoved far end, so a joint where one edge
                // widens lands short of the parallel offset (VM: Verdana bold 'W' at ppem 13, the inner
                // apexes move 25/64 px, not 32).
                let x = if left == right { cur[0] + left } else {
                    intersect(prev, [cur[0] + left, cur[1]], [cur[0] + right, cur[1]], next)[0].clamp(cur[0], cur[0] + 64)
                };
                outline.points[start + i].x = x as i32;
            }
        }
        start = end + 1;
    }
}

/// Post-hint grayscale synthetic-bold embolden of the hinted device outline (the grayscale embolden on the
/// glyph-local 26.6 outline): the byte-exact device embold, validated against a native capture. Applied
/// in glyph space before any device transform, so it is correct under rotation/shear/flip too.
pub(super) fn embolden_hinted(outline: &mut HintedOutline, bold_x: i32, bold_y: i32) {
    use super::outline_embold::Pt;
    let mut contours: Vec<Vec<Pt>> = Vec::new();
    let mut start = 0usize;
    for &end in &outline.contour_ends {
        let end = end as usize;
        if end >= outline.points.len() || end < start {
            return;
        }
        contours.push(
            (start..=end).map(|i| Pt { x: outline.points[i].x, y: outline.points[i].y, on: outline.on_curve[i] }).collect(),
        );
        start = end + 1;
    }
    super::outline_embold::apply_26dot6(&mut contours, bold_x, bold_y);
    let mut i = 0usize;
    for c in &contours {
        for p in c {
            outline.points[i].x = p.x;
            outline.points[i].y = p.y;
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn impact_outline_matches_native_bold_points() {
        let root=std::env::var_os("D2D_FONTS").map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ref/windows/Fonts"));
        let Ok(data)=std::fs::read(root.join("impact.ttf")) else { return };
        let font=skrifa::FontRef::from_index(&data,0).unwrap();
        let inst=crate::text::hinting::Instance::new(&font,48.0,Default::default()).unwrap();
        let mut outline=inst.hint(53).unwrap();
        horizontal_one_pixel(&mut outline);
        // Native gridfit glyph-fill input, Impact R at48px, synthetic bold, divided by its4x overscale.
        let expected=[(128,2432),(574,2432),(1085,2432),(1401,2363),(1600,2077),(1600,1764),
            (1600,1479),(1459,1283),(1252,1263),(1440,1217),(1569,1061),(1600,931),
            (1600,638),(1600,0),(948,0),(948,802),(948,996),(888,1088),(824,1088),
            (824,0),(128,0),(824,2048),(824,1472),(864,1472),(948,1534),(948,1706),
            (948,1848),(948,1971),(867,2048)];
        assert_eq!(outline.points.iter().map(|p|(p.x,p.y)).collect::<Vec<_>>(),expected);
    }
}
