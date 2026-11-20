//! Native dashing (`CDasher`) pixel diagnostics. The byte-exact dash scenes (dash-01..04,
//! geometry-06) are guarded by the shared `oracle_conformance` suite via `conformance/passing.txt`;
//! this target instead guards the two scenes that ride the pre-existing stroke-widening gaps —
//! `07-dashes-caps` (a diagonal chevron, the same diagonal join/cap gap as `geometry-02/03`) and
//! `dash-05-curve` (the curve-widening gap, shared with `05-curves-arcs`) — against *regression*:
//! the dash split itself is correct, so the residual stays small and bounded. On failure it prints
//! a diff map so the location is obvious.

use d2d::geometry::stroke::{Cap, Join};
use d2d::geometry::{Figure, FillMode, Geometry, Segment, StrokeStyle};
use d2d::{bitmap_target, Factory};
use std::path::PathBuf;


fn oracle_bgra(name: &str) -> Vec<u8> {
    let p = d2d_reference::suite_dir().unwrap().join("oracle/win11-26100").join(format!("{name}.bgra"));
    std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// Render one dashed open/closed path and return BGRA. Mirrors the conformance driver for the
/// single `draw(geometry, brush, width, style)` the dash scenes use.
fn render(w: u32, h: u32, dpi: f32, bg: [f32; 4], ink: [f32; 4], geom: &Geometry, width: f32, style: &StrokeStyle) -> Vec<u8> {
    let factory = Factory::new(false);
    let device = factory.create_device();
    let mut dc = device.create_device_context();
    dc.set_dpi(dpi, dpi);
    dc.set_target_bitmap(bitmap_target(w, h, dpi));
    dc.begin_draw();
    dc.clear(Some([0.0; 4])); // implicit transparent, as the harness drives native
    dc.clear(Some(bg));
    let ink_b = dc.create_solid_color_brush(ink, None);
    dc.draw_geometry(geom, &ink_b, width, style);
    let _ = dc.end_draw();
    dc.take_target().unwrap().data
}

fn diff(name: &str, w: u32, got: &[u8], want: &[u8]) -> usize {
    let h = got.len() / 4 / w as usize;
    let mut n = 0;
    let mut worst = (0i32, 0u32, 0u32);
    let mut map = vec![vec![b' '; w as usize]; h];
    for (i, (a, b)) in got.chunks_exact(4).zip(want.chunks_exact(4)).enumerate() {
        let d = (0..4).map(|c| (a[c] as i32 - b[c] as i32).abs()).max().unwrap();
        if d > 0 {
            n += 1;
            let (x, y) = (i as u32 % w, i as u32 / w);
            map[y as usize][x as usize] = if d <= 2 { b'.' } else if d <= 6 { b'o' } else { b'#' };
            if d > worst.0 {
                worst = (d, x, y);
            }
        }
    }
    println!("{name}: {n} differing pixels, worst Δ{} at ({},{})", worst.0, worst.1, worst.2);
    for row in &map {
        if row.iter().any(|&c| c != b' ') {
            println!("|{}|", String::from_utf8_lossy(row));
        }
    }
    n
}

fn path(fill: FillMode, start: [f32; 2], closed: bool, segs: Vec<[f32; 2]>) -> Geometry {
    Geometry::Path {
        fill,
        figures: vec![Figure { start, filled: true, closed, segments: segs.into_iter().map(Segment::Line).collect(), seg_flags: Vec::new() }],
    }
}

// dash-05-curve: a dashed cubic — exercises curve splitting. Not byte-exact (the curve-widening
// precision gap, shared with the non-dashed path); this maps how far off it is.
#[test]
fn dash_curve_05() {
    if d2d_reference::suite_dir().is_none() { return; }
    let geom = Geometry::Path {
        fill: FillMode::Alternate,
        figures: vec![Figure {
            start: [12.0, 78.0],
            filled: true,
            closed: false,
            segments: vec![Segment::Cubic([20.0, 6.0], [108.0, 6.0], [116.0, 78.0])],
            seg_flags: Vec::new(),
        }],
    };
    let style = StrokeStyle {
        start_cap: Cap::Round,
        end_cap: Cap::Round,
        dash_cap: Cap::Round,
        join: Join::Round,
        miter_limit: 10.0,
        dash_offset: 0.0,
        dashes: vec![3.0, 2.0],
        ..Default::default()
    };
    let got = render(128, 96, 96.0, [1.0, 1.0, 1.0, 1.0], [0.12, 0.35, 0.8, 1.0], &geom, 4.0, &style);
    let n = diff("dash-05-curve", 128, &got, &oracle_bgra("dash-05-curve"));
    assert!(n <= 220, "curve-dash residual grew to {n}px — the split regressed, not just the AA gap");
}

// Scene 07-dashes-caps: open chevron, round/square caps, round dash cap, zero-length dash, offset.
#[test]
fn dashes_caps_07() {
    if d2d_reference::suite_dir().is_none() { return; }
    let geom = path(FillMode::Alternate, [12.0, 80.0], false, vec![[30.0, 15.0], [66.0, 72.0], [114.0, 15.0]]);
    let style = StrokeStyle {
        start_cap: Cap::Round,
        end_cap: Cap::Square,
        dash_cap: Cap::Round,
        join: Join::Round,
        miter_limit: 4.0,
        dash_offset: 0.75,
        dashes: vec![2.0, 1.0, 0.0, 1.0],
        ..Default::default()
    };
    let got = render(128, 96, 96.0, [1.0, 1.0, 1.0, 1.0], [0.12, 0.35, 0.8, 1.0], &geom, 6.0, &style);
    let n = diff("07-dashes-caps", 128, &got, &oracle_bgra("07-dashes-caps"));
    assert!(n <= 300, "diagonal-dash residual grew to {n}px — the split regressed, not just the AA gap");
}
