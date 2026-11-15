//! Phase 1C conformance: run `d2d::geometry` queries against the WARP oracle's recorded
//! `query` results (`conformance/oracle/win11-26100/<scene>.json` → `queries`). The harness v2
//! `query` op exercises `ComputeArea`, `ComputeLength`, `GetBounds` and `FillContainsPoint` with an
//! identity transform and the default tolerance (0.25), so the tests compare those four.
//!
//! Length, bounds and fill-containment are asserted **bit-exact** (f32); area is asserted within
//! 1 ULP because native accumulates it with a fixed-point scanline scanner (`CScannerModuleD`)
//! rather than the exact polygon shoelace — see memory `geometry-native-flattener`.

use d2d::geometry::{FillMode, Figure, Geometry, Segment, StrokeStyle, DEFAULT_TOLERANCE, IDENTITY};
use d2d::pixels::{premultiply, Aa, Blend, FillMode as PFill, Target};
use d2d_reference::{read_scene, Op, Scene, Shape, Figure as SFigure, Segment as SSeg};
use d2d_types::abi::{D2D1_ALPHA_MODE_PREMULTIPLIED, DXGI_FORMAT_B8G8R8A8_UNORM};
use d2d_types::contracts::{ImageDesc, PixelBuffer};
use std::collections::HashMap;
use std::path::PathBuf;

fn oracle_bgra(name: &str) -> Vec<u8> {
    let p = d2d_reference::suite_dir().unwrap().join("oracle/win11-26100").join(format!("{name}.bgra"));
    std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}


fn scene_path(name: &str) -> PathBuf {
    d2d_reference::suite_dir().unwrap().join("scenes").join(format!("{name}.json"))
}
fn oracle_queries(name: &str) -> Vec<serde_json::Value> {
    let p = d2d_reference::suite_dir().unwrap().join("oracle/win11-26100").join(format!("{name}.json"));
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))).unwrap();
    v["queries"].as_array().cloned().unwrap_or_default()
}

fn seg(s: &SSeg) -> Segment {
    match *s {
        SSeg::Line { to } => Segment::Line(to),
        SSeg::Cubic { p1, p2, to } => Segment::Cubic(p1, p2, to),
        SSeg::Quadratic { control, to } => Segment::Quadratic(control, to),
        SSeg::Arc { to, radius, rotation, sweep, size } => {
            Segment::Arc { to, radius, rotation, clockwise: sweep == 1, large: size == 1 }
        }
    }
}
fn figure(f: &SFigure) -> Figure {
    Figure { start: f.start, filled: f.filled, closed: f.closed, segments: f.segments.iter().map(seg).collect(), seg_flags: Vec::new() }
}
fn geometry(shape: &Shape) -> Geometry {
    match shape {
        Shape::Rect { rect } => Geometry::Rect(*rect),
        Shape::Rounded { rect, radius } => Geometry::Rounded { rect: *rect, radius: *radius },
        Shape::Ellipse { center, radius } => Geometry::Ellipse { center: *center, radius: *radius },
        Shape::Path { fill, figures } => Geometry::Path {
            fill: if *fill == 1 { FillMode::Winding } else { FillMode::Alternate },
            figures: figures.iter().map(figure).collect(),
        },
    }
}

fn ulps_apart(a: f32, b: f32) -> i64 {
    (a.to_bits() as i64 - b.to_bits() as i64).abs()
}

// Compare one query result against the oracle. Returns a human-readable mismatch, or None if OK.
fn check(g: &Geometry, kind: &str, point: Option<[f32; 2]>, want: &serde_json::Value) -> Option<String> {
    match kind {
        "length" => {
            let got = g.compute_length(&IDENTITY, DEFAULT_TOLERANCE);
            let want = want.as_f64().unwrap() as f32;
            (got.to_bits() != want.to_bits()).then(|| format!("length got {got} ({:#010x}) want {want} ({:#010x})", got.to_bits(), want.to_bits()))
        }
        "area" => {
            let got = g.compute_area(&IDENTITY, DEFAULT_TOLERANCE);
            let want = want.as_f64().unwrap() as f32;
            let u = ulps_apart(got, want);
            (u > 1).then(|| format!("area got {got} want {want} ({u} ULP)"))
        }
        "bounds" => {
            let got = g.get_bounds(&IDENTITY);
            let want: Vec<f32> = want.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
            (0..4).find_map(|i| {
                (got[i].to_bits() != want[i].to_bits())
                    .then(|| format!("bounds[{i}] got {} want {}", got[i], want[i]))
            })
        }
        "contains_point" => {
            let got = g.fill_contains(point.expect("contains_point needs a point"), &IDENTITY, DEFAULT_TOLERANCE);
            let want = want.as_bool().unwrap();
            (got != want).then(|| format!("contains got {got} want {want}"))
        }
        other => panic!("unknown query kind {other:?}"),
    }
}

fn run_scene(name: &str) {
    let scene: Scene = read_scene(&scene_path(name)).unwrap();
    let want = oracle_queries(name);
    let by_op: HashMap<u64, &serde_json::Value> =
        want.iter().map(|q| (q["op"].as_u64().unwrap(), q)).collect();

    let mut geoms: HashMap<&str, Geometry> = HashMap::new();
    let mut failures = Vec::new();
    for (i, op) in scene.ops.iter().enumerate() {
        match op {
            Op::Geometry { id, shape } => {
                geoms.insert(id, geometry(shape));
            }
            Op::Query { geometry: id, kind, point, .. } => {
                let g = geoms.get(id.as_str()).unwrap_or_else(|| panic!("{name}: query on undefined {id}"));
                let w = by_op.get(&(i as u64)).unwrap_or_else(|| panic!("{name}: no oracle query at op {i}"));
                if let Some(msg) = check(g, kind, *point, &w["result"]) {
                    failures.push(format!("{name} op {i} ({kind} on {id}): {msg}"));
                }
            }
            Op::QueryGeom { geometry: id, kind, length, .. } if kind == "point_at_length" => {
                let g = geoms.get(id.as_str()).unwrap_or_else(|| panic!("{name}: query on undefined {id}"));
                let w = by_op.get(&(i as u64)).unwrap_or_else(|| panic!("{name}: no oracle query at op {i}"));
                let want: Vec<f32> = w["result"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
                let (p, t) = g.point_at_length(length.unwrap(), &IDENTITY, DEFAULT_TOLERANCE).unwrap();
                let got = [p[0], p[1], t[0], t[1]];
                if let Some(k) = (0..4).find(|&k| got[k].to_bits() != want[k].to_bits()) {
                    failures.push(format!("{name} op {i} (point_at_length len {:?} on {id}): [{}] got {} want {}", length, k, got[k], want[k]));
                }
            }
            _ => {}
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// Flatten `g` at device tolerance 0.25 (identity device transform: dpi 96, no world transform) and
// fill it through the 1B rasterizer, exactly as a DeviceContext FillGeometry would.
fn fill_into(t: &mut Target, g: &Geometry, color: [f32; 4]) {
    let src = premultiply(color);
    let polys = g.flatten_fill(&IDENTITY);
    let fill = if g.fill_mode() == FillMode::Winding { PFill::Winding } else { PFill::Alternate };
    let mut edges: Vec<[f32; 4]> = Vec::new();
    for poly in &polys {
        if !poly.filled || poly.pts.len() < 2 {
            continue;
        }
        let p = &poly.pts;
        for i in 0..p.len() {
            let a = p[i];
            let b = p[(i + 1) % p.len()];
            edges.push([a[0], a[1], b[0], b[1]]);
        }
    }
    t.fill_edges(&edges, fill, src, Aa::PerPrimitive, Blend::SourceOver);
}

// Render a fill-only scene (Clear, Solid, Geometry, Primitive/Fill without width) through the
// native flattener + the 1B rasterizer, returning the count of pixels differing from the oracle.
fn fill_scene_diffs(name: &str) -> u32 {
    let scene = read_scene(&scene_path(name)).unwrap();
    let [w, h] = scene.size;
    let mut buf = PixelBuffer {
        desc: ImageDesc {
            width: w,
            height: h,
            format: DXGI_FORMAT_B8G8R8A8_UNORM,
            alpha_mode: D2D1_ALPHA_MODE_PREMULTIPLIED,
            dpi_x: scene.dpi,
            dpi_y: scene.dpi,
        },
        stride: w as usize * 4,
        data: vec![0; w as usize * h as usize * 4],
    };
    let mut color = [0.0f32; 4];
    let mut geoms: HashMap<&str, Geometry> = HashMap::new();
    let mut t = Target::new(&mut buf);
    for op in &scene.ops {
        match op {
            Op::Clear { color: c } => t.clear(premultiply(*c)),
            Op::Solid { id, color: c, opacity: None, transform: None } => {
                color = *c;
                let _ = id;
            }
            Op::Geometry { id, shape } => {
                geoms.insert(id, geometry(shape));
            }
            Op::Primitive { shape, width: None, .. } => fill_into(&mut t, &geometry(shape), color),
            Op::Fill { geometry: id, .. } => fill_into(&mut t, &geoms[id.as_str()], color),
            _ => panic!("{name}: unexpected op for the pure-fill proof"),
        }
    }
    let want = oracle_bgra(name);
    assert_eq!(buf.data.len(), want.len());
    (0..buf.data.len()).step_by(4).filter(|&i| buf.data[i..i + 4] != want[i..i + 4]).count() as u32
}

/// A generic curved path (cubic + quadratic, winding) filled through the native MIL `CMILBezier`
/// 28.4 cracker (`flatten_fill`) and the 1B rasterizer is byte-identical to the WARP oracle. The
/// earlier residual was the float query flattener feeding the wrong polyline; fills use the integer
/// cracker, which is what native's scan converter sees.
#[test]
fn oracle_path_fill_pixels() {
    if d2d_reference::suite_dir().is_none() { return; }
    assert_eq!(fill_scene_diffs("1c-05-cubic-path-fill"), 0);
}

/// raster-07's three horizontal strokes (widths 1.0, 0.5, 0.25) widen to axis-aligned rectangles,
/// which the 1B rasterizer fills byte-exactly — the stroke-widening + rasterizer proof. Rows ≥ 64
/// (the diagonal stroke) carry the sloped-edge coverage residual and are excluded.
#[test]
fn oracle_axis_stroke_pixels() {
    if d2d_reference::suite_dir().is_none() { return; }
    let name = "raster-07-thin-hairlines";
    let scene = read_scene(&scene_path(name)).unwrap();
    let [w, h] = scene.size;
    let mut buf = PixelBuffer {
        desc: ImageDesc {
            width: w,
            height: h,
            format: DXGI_FORMAT_B8G8R8A8_UNORM,
            alpha_mode: D2D1_ALPHA_MODE_PREMULTIPLIED,
            dpi_x: scene.dpi,
            dpi_y: scene.dpi,
        },
        stride: w as usize * 4,
        data: vec![0; w as usize * h as usize * 4],
    };
    let mut color = [0.0f32; 4];
    let mut geoms: HashMap<&str, Geometry> = HashMap::new();
    let mut t = Target::new(&mut buf);
    for op in &scene.ops {
        match op {
            Op::Clear { color: c } => t.clear(premultiply(*c)),
            Op::Solid { color: c, .. } => color = *c,
            Op::Geometry { id, shape } => {
                geoms.insert(id, geometry(shape));
            }
            Op::Draw { geometry: id, width, .. } => {
                let pieces = geoms[id.as_str()].widen(*width, &StrokeStyle::default(), &IDENTITY, [1.0, 1.0], DEFAULT_TOLERANCE);
                let edges: Vec<[f32; 4]> = pieces
                    .iter()
                    .flat_map(|p| (0..p.len()).map(move |i| {
                        let a = p[i];
                        let b = p[(i + 1) % p.len()];
                        [a[0], a[1], b[0], b[1]]
                    }))
                    .collect();
                t.fill_edges(&edges, PFill::Winding, premultiply(color), Aa::PerPrimitive, Blend::SourceOver);
            }
            _ => {}
        }
    }
    let want = oracle_bgra(name);
    let band = 64 * w as usize * 4; // rows 0..64 hold only the three horizontal strokes
    for i in (0..band).step_by(4) {
        if buf.data[i..i + 4] != want[i..i + 4] {
            let p = (i / 4) as u32;
            panic!("{name}: first diff at ({},{}) got {:?} want {:?}", p % w, p / w, &buf.data[i..i + 4], &want[i..i + 4]);
        }
    }
}

#[test]
fn oracle_queries_match() {
    if d2d_reference::suite_dir().is_none() { return; }
    // v2-query (ellipse) is 0B's; the `1c-*` scenes are added and captured on the VM by this task:
    // rect (fast-path area), rounded rect, a cubic path (true-extrema bounds), and an even-odd
    // two-figure path with a hole. Together they cover all four query kinds across every shape.
    for name in [
        "v2-query",
        "1c-01-rect-queries",
        "1c-02-rounded-queries",
        "1c-03-path-cubic-queries",
        "1c-04-path-alternate-hole",
        "1c2-01-point-at-length",
    ] {
        run_scene(name);
    }
}
