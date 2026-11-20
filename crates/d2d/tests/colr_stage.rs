//! Black-box bisection of the COLRv1 emoji color path with synthetic single-stage fonts: render each
//! tiny test font (solid alpha, sub-pixel edges, layered source-over, gradient) through our
//! `draw_color_glyph_run` and diff against native d2d-ref (`DrawPaintGlyphRun`) captures, so every stage
//! becomes a fixed input with a known final pixel. Font bytes + native BGRA live under `D2D_COLR_DIR`; every
//! test skips when it is unset.

use d2d::text::{Glyph as TGlyph, GlyphRun as TGlyphRun, RenderingParams};
use d2d::{bitmap_target, Factory};
use skrifa::instance::{LocationRef, Size};
use skrifa::{FontRef, GlyphId, MetadataProvider};

/// The synthetic-font + native-capture dir (`D2D_COLR_DIR`), shown with `{DIR}`. Unset, it names a path
/// that does not exist, so every read fails and its test skips.
const DIR: ColrDir = ColrDir;
struct ColrDir;
impl std::fmt::Display for ColrDir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&std::env::var("D2D_COLR_DIR").unwrap_or_else(|_| "<D2D_COLR_DIR unset>".into()))
    }
}

fn resolve(font: &[u8], text: &str, em: f32) -> Vec<TGlyph> {
    let fr = FontRef::from_index(font, 0).unwrap();
    let upem = fr.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f32;
    let cmap = fr.charmap();
    let gm = fr.glyph_metrics(Size::unscaled(), LocationRef::default());
    text.chars()
        .map(|ch| {
            let id = cmap.map(ch).map(|g| g.to_u32() as u16).unwrap_or(0);
            let adv = gm.advance_width(GlyphId::new(id as u32)).unwrap_or(0.0) * em / upem;
            TGlyph { id, advance: adv, offset: [0.0, 0.0] }
        })
        .collect()
}

/// Render `text` from `font` as a colour glyph run at `baseline`, em `em`, into a `w×h` white target.
fn render_ours(font: &[u8], text: &str, em: f32, baseline: [f32; 2], color: [f32; 4], palette: u32, w: u32, h: u32) -> Vec<u8> {
    let factory = Factory::new(false);
    let device = factory.create_device();
    let mut dc = device.create_device_context();
    dc.set_dpi(96.0, 96.0);
    dc.set_target_bitmap(bitmap_target(w, h, 96.0));
    dc.begin_draw();
    dc.clear(Some([1.0, 1.0, 1.0, 1.0]));
    let glyphs = resolve(font, text, em);
    let brush = dc.create_solid_color_brush(color, Some(1.0));
    dc.set_text_antialias_mode(2);
    let run = TGlyphRun {
        font_data: font,
        face_index: 0,
        em_size: em,
        glyphs: &glyphs,
        baseline_origin: baseline,
        measuring_mode: 0,
        is_sideways: false,
        bidi_level: 0,
        no_snap: false,
        simulations: 0,
        coords: &[],
    };
    dc.draw_color_glyph_run(&run, &brush, palette, &RenderingParams::default());
    dc.end_draw();
    dc.take_target().unwrap().data
}

#[test]
fn colr_e150_no_ss() {
    // em 150 (device>128) ⇒ native takes the NO-supersample path (draws the leaf grayscale at device
    // res directly, no minify). Compare our leaf coverage to native's without the minify confound.
    let Ok(font) = std::fs::read(format!("{DIR}/t1_solid_a1.ttf")) else { return };
    let w = 200usize;
    let row = 80usize;
    for (tag, bx) in [("e150_x00", 12.0), ("e150_x50", 12.5), ("e150_x125", 12.125f32)] {
        let Ok(native) = std::fs::read(format!("{DIR}/{tag}_native.bgra")) else { continue };
        let ours = render_ours(&font, "A", 150.0, [bx, 152.0], [0.0, 0.0, 0.0, 1.0], 0, 200, 200);
        let cov = |d: &[u8], x: usize| 255i32 - d[(row * w + x) * 4 + 1] as i32;
        // find left edge from native
        let s = (8..40).find(|&x| cov(&native, x) > 0).unwrap_or(12);
        let o: Vec<i32> = (s.saturating_sub(1)..s + 3).map(|x| cov(&ours, x)).collect();
        let n: Vec<i32> = (s.saturating_sub(1)..s + 3).map(|x| cov(&native, x)).collect();
        eprintln!("{tag} bx={bx} edge@{s} ours={o:?} native={n:?}");
    }
}

#[test]
fn colr_row_profile() {
    // Full horizontal scanline through the square interior: ours vs native coverage (255−G) per column,
    // to see square extent (left/right edge), width, and edge shape. x00 baseline (bx=12.0).
    let Ok(font) = std::fs::read(format!("{DIR}/t1_solid_a1.ttf")) else { return };
    let Ok(native) = std::fs::read(format!("{DIR}/x00_native.bgra")) else { return };
    let ours = render_ours(&font, "A", 50.0, [12.0, 52.0], [0.0, 0.0, 0.0, 1.0], 0, 64, 64);
    let w = 64usize;
    let row = 26usize;
    let cov = |d: &[u8], x: usize| 255i32 - d[(row * w + x) * 4 + 1] as i32;
    let o: Vec<i32> = (8..40).map(|x| cov(&ours, x)).collect();
    let n: Vec<i32> = (8..40).map(|x| cov(&native, x)).collect();
    eprintln!("row {row} cols 8..40");
    eprintln!("  ours  ={o:?}");
    eprintln!("  native={n:?}");
    // also a vertical column through the interior to see top/bottom edges
    let col = 20usize;
    let covc = |d: &[u8], y: usize| 255i32 - d[(y * w + col) * 4 + 1] as i32;
    let oc: Vec<i32> = (8..40).map(|y| covc(&ours, y)).collect();
    let nc: Vec<i32> = (8..40).map(|y| covc(&native, y)).collect();
    eprintln!("col {col} rows 8..40");
    eprintln!("  ours  ={oc:?}");
    eprintln!("  native={nc:?}");
}

/// Single-leaf probe for emoji-162's face circle (gid 4070), solid fill, at 162's exact device
/// transform (baked into the probe font's PaintTransform). Renders ours, dumps the ALPHA (=coverage,
/// black-on-white → 255−G) around the right edge (x≈32, where 162 under-covers) and the full ink bbox,
/// and writes ours.bgra for the native-VM diff. Native is rendered from probe162.json via d2d-ref.
#[test]
fn probe162_face_circle() {
    // COLR_PROBE_DIR holds the probe font/native pairs (skips when unset).
    let dir = std::env::var("COLR_PROBE_DIR").unwrap_or_else(|_| "<COLR_PROBE_DIR unset>".into());
    // COLR_PROBE=<basename> selects which probe font/native pair under `dir` (default probe162).
    let base = std::env::var("COLR_PROBE").unwrap_or_else(|_| "probe162".into());
    let Ok(font) = std::fs::read(format!("{dir}/{base}.ttf")) else {
        eprintln!("{base}.ttf missing — run the probe builder");
        return;
    };
    let bx: f32 = std::env::var("COLR_PROBE_BX").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let by: f32 = std::env::var("COLR_PROBE_BY").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let ours = render_ours(&font, "A", 24.0, [bx, by], [1.0, 0.0, 0.0, 1.0], 0, 72, 72);
    std::fs::write(format!("{dir}/{base}_ours.bgra"), &ours).ok();
    let w = 72usize;
    let cov = |d: &[u8], x: usize, y: usize| 255i32 - d[(y * w + x) * 4 + 1] as i32; // 255−G = red coverage
    // ink bbox
    let (mut l, mut t, mut r, mut b) = (w as i32, w as i32, 0i32, 0i32);
    for y in 0..w { for x in 0..w { if cov(&ours, x, y) > 0 { l = l.min(x as i32); t = t.min(y as i32); r = r.max(x as i32 + 1); b = b.max(y as i32 + 1); } } }
    eprintln!("ours ink bbox = [{l},{t},{r},{b}] (expect ~[8,7,33,32])");
    for y in 17..22 {
        let rowv: Vec<i32> = (29..34).map(|x| cov(&ours, x, y)).collect();
        eprintln!("ours row {y} cols 29..34 = {rowv:?}");
    }
    // If native is present, diff.
    if let Ok(nat) = std::fs::read(format!("{dir}/{base}_native.bgra")) {
        let (mut bytes, mut maxd, mut mx, mut my) = (0u64, 0i32, 0usize, 0usize);
        for i in 0..ours.len().min(nat.len()) {
            let d = (ours[i] as i32 - nat[i] as i32).abs();
            if d != 0 { bytes += 1; if d > maxd { maxd = d; mx = (i / 4) % w; my = (i / 4) / w; } }
        }
        eprintln!("DIFF vs native: bytes={bytes} maxd={maxd} @({mx},{my})");
        for y in 17..22 {
            let o: Vec<i32> = (29..34).map(|x| cov(&ours, x, y)).collect();
            let n: Vec<i32> = (29..34).map(|x| cov(&nat, x, y)).collect();
            eprintln!("  y{y} cols29..34 ours={o:?} native={n:?}");
        }
    } else {
        eprintln!("(native.bgra absent — render probe162.json on the VM, pull to {dir}/native.bgra, re-run)");
    }
}

/// Render the square via the PLAIN (non-color) grayscale text path in red, to check whether native's
/// COLR leaf coverage equals plain DrawGlyphRun (i.e. the text raster + gamma blend) — the hinting
/// agent said plain text is byte-exact to native. Native COLR leaf left-edge = [255,196,140,83].
#[test]
fn colr_plain_vs_leaf() {
    let Ok(font) = std::fs::read(format!("{DIR}/t1_solid_a1.ttf")) else { return };
    let w = 64usize;
    let row = 26usize;
    for (tag, bx) in [("x00", 12.0), ("x25", 12.25), ("x50", 12.5), ("x75", 12.75f32)] {
        let factory = Factory::new(false);
        let device = factory.create_device();
        let mut dc = device.create_device_context();
        dc.set_dpi(96.0, 96.0);
        dc.set_target_bitmap(bitmap_target(64, 64, 96.0));
        dc.begin_draw();
        dc.clear(Some([1.0, 1.0, 1.0, 1.0]));
        let glyphs = resolve(&font, "A", 50.0);
        let brush = dc.create_solid_color_brush([1.0, 0.0, 0.0, 1.0], Some(1.0));
        dc.set_text_antialias_mode(2);
        let run = TGlyphRun {
            font_data: &font, face_index: 0, em_size: 50.0, glyphs: &glyphs,
            baseline_origin: [bx, 52.0], measuring_mode: 0, is_sideways: false, bidi_level: 0, no_snap: false, simulations: 0, coords: &[],
        };
        dc.draw_glyph_run(&run, &brush, &RenderingParams::default());
        dc.end_draw();
        let ours = dc.take_target().unwrap().data;
        let cov = |d: &[u8], x: usize| 255i32 - d[(row * w + x) * 4 + 1] as i32;
        let o: Vec<i32> = (12..16).map(|x| cov(&ours, x)).collect();
        eprintln!("plain {tag} bx={bx} left-edge cols 12.. ours={o:?}  (native COLR leaf=[255,196,140,83] by bx)");
    }
}

/// Dump the seguiemj paint tree for a char (default ☕ U+2615, emoji-000) — transforms, clips, layers,
/// and each leaf's gradient (type, extend, stops, angles) — to find structural t bugs (transform
/// nesting / extend / sweep angles / space), not the ±1 sampler. Set COLR_DUMP_CHAR to another char.
#[test]
fn colr_paint_tree_dump() {
    use skrifa::color::{Brush as CB, ColorGlyphCollection, ColorPainter, CompositeMode, Transform};
    use skrifa::raw::types::BoundingBox;
    let Some(font) = d2d_reference::fonts_dir().and_then(|d| std::fs::read(d.join("seguiemj.ttf")).ok()) else {
        eprintln!("skip: seguiemj absent");
        return;
    };
    let ch = std::env::var("COLR_DUMP_CHAR").ok().and_then(|s| s.chars().next()).unwrap_or('☕');
    let fr = FontRef::from_index(&font, 0).unwrap();
    let gid = fr.charmap().map(ch).unwrap();
    let colr = ColorGlyphCollection::new(&fr);
    struct Rec {
        depth: usize,
    }
    impl Rec {
        fn ind(&self) -> String {
            "  ".repeat(self.depth)
        }
    }
    impl ColorPainter for Rec {
        fn push_transform(&mut self, t: Transform) {
            eprintln!("{}push_transform {:?}", self.ind(), [t.xx, t.yx, t.xy, t.yy, t.dx, t.dy]);
            self.depth += 1;
        }
        fn pop_transform(&mut self) {
            self.depth = self.depth.saturating_sub(1);
            eprintln!("{}pop_transform", self.ind());
        }
        fn push_clip_box(&mut self, b: BoundingBox<f32>) {
            eprintln!("{}push_clip_box {:?}", self.ind(), (b.x_min, b.y_min, b.x_max, b.y_max));
        }
        fn push_clip_glyph(&mut self, g: GlyphId) {
            eprintln!("{}push_clip_glyph {}", self.ind(), g.to_u32());
        }
        fn pop_clip(&mut self) {
            eprintln!("{}pop_clip", self.ind());
        }
        fn fill(&mut self, b: CB<'_>) {
            eprintln!("{}fill {}", self.ind(), brush_str(&b));
        }
        fn fill_glyph(&mut self, g: GlyphId, bt: Option<Transform>, b: CB<'_>) {
            let t = bt.map(|t| [t.xx, t.yx, t.xy, t.yy, t.dx, t.dy]);
            eprintln!("{}fill_glyph gid={} brush_tf={:?} {}", self.ind(), g.to_u32(), t, brush_str(&b));
        }
        fn push_layer(&mut self, m: CompositeMode) {
            eprintln!("{}push_layer {:?}", self.ind(), m);
            self.depth += 1;
        }
        fn pop_layer_with_mode(&mut self, m: CompositeMode) {
            self.depth = self.depth.saturating_sub(1);
            eprintln!("{}pop_layer {:?}", self.ind(), m);
        }
    }
    fn brush_str(b: &CB<'_>) -> String {
        match b {
            CB::Solid { palette_index, alpha } => format!("SOLID pal={palette_index} a={alpha}"),
            CB::LinearGradient { p0, p1, extend, color_stops } => {
                let s: Vec<_> = color_stops.iter().map(|c| (c.offset, c.palette_index, c.alpha)).collect();
                format!("LINEAR p0={p0:?} p1={p1:?} extend={extend:?} stops={s:?}")
            }
            CB::RadialGradient { c0, r0, c1, r1, extend, color_stops } => {
                let s: Vec<_> = color_stops.iter().map(|c| (c.offset, c.palette_index, c.alpha)).collect();
                format!("RADIAL c0={c0:?} r0={r0} c1={c1:?} r1={r1} extend={extend:?} stops={s:?}")
            }
            CB::SweepGradient { c0, start_angle, end_angle, extend, color_stops } => {
                let s: Vec<_> = color_stops.iter().map(|c| (c.offset, c.palette_index, c.alpha)).collect();
                format!("SWEEP c0={c0:?} start={start_angle} end={end_angle} extend={extend:?} stops={s:?}")
            }
        }
    }
    eprintln!("=== paint tree for {ch:?} gid={} ===", gid.to_u32());
    let mut rec = Rec { depth: 0 };
    colr.get(gid).unwrap().paint(skrifa::instance::LocationRef::default(), &mut rec).unwrap();
}

#[test]
fn colr_e150_fine() {
    // Our leaf coverage (col 12, row 80) across the em150 bx sweep, to pair with native and extract g.
    let Ok(font) = std::fs::read(format!("{DIR}/t1_solid_a1.ttf")) else { return };
    let w = 200usize;
    let mut out = Vec::new();
    for i in 0..=32 {
        let bx = 12.0 + i as f32 / 32.0;
        let ours = render_ours(&font, "A", 150.0, [bx, 152.0], [0.0, 0.0, 0.0, 1.0], 0, 200, 200);
        let cov = 255i32 - ours[(80 * w + 12) * 4 + 1] as i32;
        out.push(cov);
    }
    eprintln!("e150_fine ours col12: {out:?}");
}

#[test]
fn colr_edge_sweep() {
    // Solid opaque square swept by sub-pixel x; read the left-edge column coverage (1 − G/255, since
    // red-over-white gives G=B=255·(1−cov)) for ours vs native to pin the partial-edge downsample.
    let Ok(font) = std::fs::read(format!("{DIR}/t1_solid_a1.ttf")) else {
        eprintln!("skip colr_edge_sweep: font absent");
        return;
    };
    let w = 64usize;
    let row = 26usize; // a scanline through the square interior (vertical coverage = full)
    for (tag, bx) in [("x00", 12.0), ("x25", 12.25), ("x50", 12.5), ("x75", 12.75), ("x10", 12.1f32), ("x30", 12.3)] {
        let native = std::fs::read(format!("{DIR}/{tag}_native.bgra")).unwrap_or_default();
        let ours = render_ours(&font, "A", 50.0, [bx, 52.0], [0.0, 0.0, 0.0, 1.0], 0, 64, 64);
        let covn = |d: &[u8], x: usize| 255i32 - d[(row * w + x) * 4 + 1] as i32;
        let start = 12usize;
        let o: Vec<i32> = (start..start + 4).map(|x| covn(&ours, x)).collect();
        if native.is_empty() {
            eprintln!("{tag} bx={bx} left-edge cols {start}.. ours={o:?} native=(none)");
            continue;
        }
        let n: Vec<i32> = (start..start + 4).map(|x| covn(&native, x)).collect();
        eprintln!("{tag} bx={bx} left-edge cols {start}.. ours={o:?} native={n:?}");
    }
}

#[test]
fn colr_stage_vs_native() {
    // (font file, native bgra) pairs; skip cleanly if the scratch captures are absent.
    let cases: &[(&str, &str)] = &[("t1_solid_a1.ttf", "t1_native.bgra"), ("t2_solid_a05.ttf", "t2_native.bgra")];
    for (ttf, nat) in cases {
        let Ok(font) = std::fs::read(format!("{DIR}/{ttf}")) else {
            eprintln!("skip colr_stage: {ttf} absent");
            return;
        };
        let native = std::fs::read(format!("{DIR}/{nat}")).unwrap();
        let ours = render_ours(&font, "A", 50.0, [12.0, 52.0], [0.0, 0.0, 0.0, 1.0], 0, 64, 64);
        assert_eq!(ours.len(), native.len(), "{ttf}: size");
        let w = 64usize;
        let (mut nd, mut maxd) = (0u32, 0i32);
        let mut first = None;
        for (i, (a, b)) in ours.as_chunks::<4>().0.iter().zip(native.as_chunks::<4>().0).enumerate() {
            let d = (0..4).map(|k| (a[k] as i32 - b[k] as i32).abs()).max().unwrap();
            if d != 0 {
                nd += 1;
                if d > maxd {
                    maxd = d;
                }
                if first.is_none() {
                    first = Some((i % w, i / w, *a, *b));
                }
            }
        }
        eprintln!("{ttf}: {nd} px diff, maxΔ={maxd} first={first:?}");
    }
}
