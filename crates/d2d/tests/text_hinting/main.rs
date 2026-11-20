//! Checks for the DirectWrite TrueType hinting interpreter. Font-gated: the fonts are read from
//! `ref/windows/Fonts` (gitignored, never committed) and each test skips cleanly when absent, as in
//! CI. The GDI-advance oracle is `fixtures/gdi_advances.json`, native
//! `GetGdiCompatibleGlyphMetrics` captured on the reference VM.
//!
//! 1. `hinting_runs_and_matches_independent_interpreter` — aliased/mono grid-fitting matches skrifa's
//!    FreeType-faithful interpreter (bounding boxes), and every Arial glyph program runs to finite
//!    output.
//! 2. `use_my_metrics_shares_base_advance` — a USE_MY_METRICS composite shares its base glyph's hinted
//!    advance (the 'á'=18→17 fix), in every mode, no oracle needed.
//! 3. `subpixel_matches_v40_reference` — the smooth (grayscale) modes grid-fit y only; the hinted
//!    outline matches skrifa's independent v40 "smooth" interpreter on vertical-stem glyphs.
//! 4. `gdi_advances_all_fonts` — GDI-classic (≥97%) and GDI-natural (≥95%) whole-pixel advances
//!    match native for the 14 core fonts + Bahnschrift; the residual is DirectWrite's
//!    compatible-width proportional-spacing model (not reproduced).

use d2d::text::hinting::{Config, Instance, Mode};
use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{DrawSettings, Engine, HintingInstance, HintingOptions, OutlinePen, Target};
use skrifa::{FontRef, MetadataProvider};

/// The native font dir (`D2D_FONTS`, else `ref/windows/Fonts` — `d2d_reference::fonts_dir`), shown with
/// `{FONT_DIR}`. Without either it names a path that does not exist, so every read fails and its test skips.
const FONT_DIR: FontDir = FontDir;
struct FontDir;
impl std::fmt::Display for FontDir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        static DIR: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            d2d_reference::fonts_dir().map_or_else(|| "<no D2D_FONTS or ref/windows/Fonts>".into(), |d| d.display().to_string())
        });
        f.write_str(&DIR)
    }
}
const SAMPLE: &str = "HnOoRgABCMWd863.áé"; // stems, round glyphs, digits, punctuation, composites

fn load_arial() -> Option<Vec<u8>> {
    std::fs::read(format!("{FONT_DIR}/arial.ttf")).ok()
}

#[test]
fn glyph_scan_control_includes_component_modes() {
    let Some(bytes) = load_arial() else { return; };
    let font = FontRef::new(&bytes).unwrap();
    let instance = Instance::new(&font, 21.0, Config::default()).unwrap();
    // Native 16-text-grayscale: the two Arabic outlines use simple dropout modes 0 and 1.
    assert_eq!(instance.hint(913).unwrap().scan_control, (true, 0));
    assert_eq!(instance.hint(941).unwrap().scan_control, (true, 1));
}

#[test]
fn gdi_classic_f_origin_matches_native() {
    let Some(bytes) = load_arial() else { return; };
    let font = FontRef::new(&bytes).unwrap();
    let instance = Instance::new(&font, 13.0, Config { mode: Mode::GdiClassic, gdi_render: true, cleartype: true, symmetric: false, ..Default::default() }).unwrap();
    let glyph = instance.hint(font.charmap().map('f').unwrap().to_u32() as u16).unwrap();
    // Native dwrite.dll 10.0.26100.8875: metric advance 192/64, phase factor 54471.
    assert_eq!((glyph.points[0].x, glyph.points[0].y), (57, 0));
}

#[test]
fn georgia_composite_accent_offset_matches_native() {
    let Ok(bytes) = std::fs::read(format!("{FONT_DIR}/georgia.ttf")) else { return; };
    let font = FontRef::new(&bytes).unwrap();
    let instance = Instance::new(&font, 16.0, Config { symmetric: false, ..Default::default() }).unwrap();
    let glyph = instance.hint(font.charmap().map('é').unwrap().to_u32() as u16).unwrap();
    // Native cold contour: the acute component shifts by 36/64 px, not 64/64 px.
    assert_eq!((glyph.points[43].x, glyph.points[43].y), (384, 733));
}

#[test]
fn segoe_composite_tilde_matches_native() {
    let Ok(bytes) = std::fs::read(format!("{FONT_DIR}/segoeui.ttf")) else { return; };
    let font = FontRef::new(&bytes).unwrap();
    let instance = Instance::new(&font, 20.0, Config { symmetric: false, ..Default::default() }).unwrap();
    let glyph = instance.hint(font.charmap().map('ñ').unwrap().to_u32() as u16).unwrap();
    // Native dwrite.dll cold capture: parent WCVTF(-4) scales to -2, then MSIRP shifts 16/64 px.
    assert_eq!((glyph.points[29].x, glyph.points[29].y), (184, 768));
}

#[test]
fn georgia_italic_temporary_phantom_vectors_match_native() {
    let Ok(bytes) = std::fs::read(format!("{FONT_DIR}/georgiai.ttf")) else { return; };
    let font = FontRef::new(&bytes).unwrap();
    let instance = Instance::new(&font, 48.0, Config::default()).unwrap();
    d2d::text::hinting::enable_trace();
    let glyph = instance.hint(font.charmap().map('R').unwrap().to_u32() as u16).unwrap();
    let trace = d2d::text::hinting::take_trace();
    // Native dwrite.dll 10.0.26100.8875 cold mover trace: CALL58 temporarily positions
    // the advance phantom at (512,2112), then restores it to the original advance.
    assert!(trace.iter().any(|r| r.op == 0xe0 && r.x[68] == 512 && r.y[68] == 2112));
    assert_eq!(glyph.advance, 2156);
    // The interpolated lower stem previously landed at (628,304).
    assert_eq!((glyph.points[7].x, glyph.points[7].y), (630, 303));
}

#[test]
fn georgia_italic_diagonal_stems_match_native() {
    let Ok(bytes) = std::fs::read(format!("{FONT_DIR}/georgiai.ttf")) else { return; };
    let font = FontRef::new(&bytes).unwrap();
    let instance = Instance::new(&font, 48.0, Config::default()).unwrap();
    // Native dwrite.dll 10.0.26100.8875 cold scan-converter glyph-fill contours, natural symmetric grayscale.
    for (ch, expected) in [
        ('R', [(0, 2056, 128), (7, 630, 303), (25, 312, 304), (26, 659, 1798), (36, 441, 2112), (58, 1628, 1670)]),
        ('n', [(0, 1724, 256), (7, 1098, 213), (25, 426, 1159), (26, 426, 1201), (36, 507, 1536), (58, 1696, 234)]),
    ] {
        let glyph = instance.hint(font.charmap().map(ch).unwrap().to_u32() as u16).unwrap();
        for (index, x, y) in expected {
            assert_eq!((glyph.points[index].x, glyph.points[index].y), (x, y), "{ch} point {index}");
        }
    }
}

/// Bounding box of every coordinate a pen receives (pixels).
#[derive(Default)]
struct BBox {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    any: bool,
}
impl BBox {
    fn add(&mut self, x: f32, y: f32) {
        if !self.any {
            self.min_x = x;
            self.max_x = x;
            self.min_y = y;
            self.max_y = y;
            self.any = true;
        } else {
            self.min_x = self.min_x.min(x);
            self.max_x = self.max_x.max(x);
            self.min_y = self.min_y.min(y);
            self.max_y = self.max_y.max(y);
        }
    }
}
impl OutlinePen for BBox {
    fn move_to(&mut self, x: f32, y: f32) {
        self.add(x, y);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.add(x, y);
    }
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        self.add(cx, cy);
        self.add(x, y);
    }
    fn curve_to(&mut self, c0x: f32, c0y: f32, c1x: f32, c1y: f32, x: f32, y: f32) {
        self.add(c0x, c0y);
        self.add(c1x, c1y);
        self.add(x, y);
    }
    fn close(&mut self) {}
}

#[test]
fn hinting_runs_and_matches_independent_interpreter() {
    let Some(bytes) = load_arial() else {
        eprintln!("skipping: {FONT_DIR}/arial.ttf not present");
        return;
    };
    let font = FontRef::new(&bytes).expect("parse arial");
    let outlines = font.outline_glyphs();
    let charmap = font.charmap();

    let ppems = [8.0_f32, 10.0, 11.0, 12.0, 16.0, 20.0, 24.0, 32.0];
    let mut hinted_glyphs = 0usize;
    let mut compared = 0usize;
    let mut matched = 0usize;
    let mut first_diff: Option<String> = None;

    for &ppem in &ppems {
        // Our aliased instance, and skrifa's mono interpreter: both are the classic
        // bi-level path with no backward-compatibility and no ClearType flags.
        // Match skrifa's reported scaler version (40) so the cross-check isolates our
        // interpreter's arithmetic/semantics from Arial's version-dependent branches
        // (production uses the real DirectWrite version, 42).
        let mine = Instance::new(
            &font,
            ppem as f64,
            Config { mode: Mode::Aliased, symmetric: false, version: 40, ..Default::default() },
        )
        .expect("build instance");
        let skrifa_inst = HintingInstance::new(
            &outlines,
            Size::new(ppem),
            LocationRef::default(),
            HintingOptions { engine: Engine::Interpreter, target: Target::Mono },
        )
        .expect("build skrifa instance");

        for ch in SAMPLE.chars() {
            let Some(gid) = charmap.map(ch) else { continue };
            // 1. Our hinter must run to completion with finite output.
            let out = mine.hint(gid.to_u32() as u16).expect("hint glyph");
            for p in &out.points {
                assert!(p.x.abs() < 1 << 24 && p.y.abs() < 1 << 24, "finite point {ch}@{ppem}");
            }
            assert!(out.advance >= 0, "advance {ch}@{ppem}");
            if !out.points.is_empty() {
                hinted_glyphs += 1;
            }

            // 2. Compare hinted bounding boxes and the hinted advance width.
            let glyph = outlines.get(gid).expect("outline");
            let mut sk = BBox::default();
            let metrics = glyph
                .draw(DrawSettings::hinted(&skrifa_inst, false), &mut sk)
                .expect("skrifa draw");
            // The hinted advance (the GDI advance width) is checked exactly for simple
            // glyphs. Composite advance involves USE_MY_METRICS and no-instruction phantom
            // nuances that are deliberately simplified here (the outline still matches).
            let composite = matches!(ch, 'á' | 'é');
            if let (Some(adv), false) = (metrics.advance_width, composite) {
                assert!(
                    (out.advance_px() - adv).abs() <= 0.02,
                    "advance '{ch}' @ {ppem}px: mine={} skrifa={adv}",
                    out.advance_px()
                );
            }
            if !sk.any || out.points.is_empty() {
                continue;
            }
            let mut mine_bb = BBox::default();
            for p in &out.points {
                mine_bb.add(p.x as f32 / 64.0, p.y as f32 / 64.0);
            }
            compared += 1;
            if ch == 'R' && ppem == 24.0 {
                // Native dwrite.dll 10.0.26100.8875 cold aliased contour: xmax1013, versus
                // skrifa's1011. Pin the native result rather than expanding the general tolerance.
                assert_eq!((mine_bb.min_x, mine_bb.min_y, mine_bb.max_x, mine_bb.max_y), (2.0, 0.0, 1013.0 / 64.0, 17.0));
                matched += 1;
                continue;
            }
            let tol = 0.03; // ~1 F26Dot6 unit of rounding slop
            let ok = (mine_bb.min_x - sk.min_x).abs() <= tol
                && (mine_bb.max_x - sk.max_x).abs() <= tol
                && (mine_bb.min_y - sk.min_y).abs() <= tol
                && (mine_bb.max_y - sk.max_y).abs() <= tol;
            if ok {
                matched += 1;
            } else if first_diff.is_none() {
                first_diff = Some(format!(
                    "'{ch}' @ {ppem}px: mine=[{:.3},{:.3},{:.3},{:.3}] skrifa=[{:.3},{:.3},{:.3},{:.3}]",
                    mine_bb.min_x, mine_bb.min_y, mine_bb.max_x, mine_bb.max_y,
                    sk.min_x, sk.min_y, sk.max_x, sk.max_y
                ));
            }
        }
    }

    assert!(hinted_glyphs > 0, "no glyphs hinted");
    eprintln!("hinting cross-check: {matched}/{compared} bbox matches; glyphs hinted: {hinted_glyphs}");
    if let Some(d) = &first_diff {
        eprintln!("first bbox difference: {d}");
    }
    // Compare the independent interpreter within the existing rounding tolerance, with the
    // directly captured native exception above checked exactly.
    assert_eq!(
        matched, compared,
        "aliased hinting diverged from the reference interpreter; first: {first_diff:?}"
    );
}

// ---------------------------------------------------------------------------------------------
// Part 2: GDI-compatible glyph metrics, every textx font, and the smooth (subpixel) coverage path.
// Oracle: native IDWriteFontFace::GetGdiCompatibleGlyphMetrics + IDWriteGlyphRunAnalysis, captured
// on the reference VM into `fixtures/gdi_advances.json` (committed). Fonts are read from
// `ref/windows/Fonts` (gitignored); the tests skip when they are absent, as in CI.

const FONTS: &[(&str, &str)] = &[
    ("Arial", "arial.ttf"), ("Calibri", "calibri.ttf"), ("Cambria", "cambria.ttc"),
    ("Comic Sans MS", "comic.ttf"), ("Consolas", "consola.ttf"), ("Courier New", "cour.ttf"),
    ("Georgia", "georgia.ttf"), ("Lucida Console", "lucon.ttf"), ("Palatino Linotype", "pala.ttf"),
    ("Segoe UI", "segoeui.ttf"), ("Tahoma", "tahoma.ttf"), ("Times New Roman", "times.ttf"),
    ("Trebuchet MS", "trebuc.ttf"), ("Verdana", "verdana.ttf"), ("Bahnschrift", "bahnschrift.ttf"),
];

fn font_bytes(file: &str) -> Option<Vec<u8>> {
    std::fs::read(format!("{FONT_DIR}/{file}")).ok()
}

/// Items 2/3/4: GDI-classic and GDI-natural whole-pixel advances match native
/// `GetGdiCompatibleGlyphMetrics` for every textx font, the composite USE_MY_METRICS advances are
/// exact, and every glyph of every font hints to finite output. The remaining ~classic/~natural
/// residual is DirectWrite's compatible-width proportional-spacing model (not reproduced).
#[test]
fn gdi_advances_all_fonts() {
    let Some(suite) = d2d_reference::suite_dir() else { return };
    let fixture = suite.join("fixtures/text_hinting/fixtures/gdi_advances.json");
    let oracle: serde_json::Value = serde_json::from_slice(&std::fs::read(fixture).unwrap()).unwrap();
    let fmap: std::collections::HashMap<&str, &str> = FONTS.iter().copied().collect();
    if font_bytes("arial.ttf").is_none() {
        eprintln!("skipping gdi_advances_all_fonts: {FONT_DIR} not present");
        return;
    }
    let (mut cok, mut ctot, mut nok) = (0u32, 0u32, 0u32);
    let mut cfirst = String::new();
    for fo in oracle["fonts"].as_array().unwrap() {
        let name = fo["font"].as_str().unwrap();
        let upem = fo["upem"].as_f64().unwrap();
        let Some(bytes) = font_bytes(fmap[name]) else { continue };
        let font = FontRef::from_index(&bytes, 0).expect("font");
        for row in fo["rows"].as_array().unwrap() {
            let r = row.as_array().unwrap();
            let ppem = r[0].as_f64().unwrap();
            let cl = Instance::new(&font, ppem, Config { mode: Mode::GdiClassic, symmetric: false, version: 42, ..Default::default() }).unwrap();
            let na = Instance::new(&font, ppem, Config { mode: Mode::GdiNatural, symmetric: false, version: 42, subpixel_advance: true, ..Default::default() }).unwrap();
            for g in r[1].as_array().unwrap() {
                let g = g.as_array().unwrap();
                let id = g[0].as_u64().unwrap() as u16;
                let ch = g[1].as_str().unwrap_or("?");
                let cexp = g[3].as_f64().unwrap() * ppem / upem;
                let nexp = g[4].as_f64().unwrap() * ppem / upem;
                let ch_adv = cl.hint(id).expect("hint classic");
                let nh_adv = na.hint(id).expect("hint natural");
                assert!(ch_adv.advance >= 0 && ch_adv.points.iter().all(|p| p.x.abs() < 1 << 24 && p.y.abs() < 1 << 24), "finite {name} id{id}@{ppem}");
                // The GDI-natural metric advance stays sub-pixel; an unrotated layout rounds it (as native does for an axis-aligned transform).
                let (cm, nm) = (ch_adv.advance_px() as f64, (nh_adv.advance_px() as f64 + 0.5).floor());
                ctot += 1;
                let cmatch = (cm - cexp).abs() <= 0.01;
                if cmatch { cok += 1; } else if cfirst.is_empty() { cfirst = format!("{name} '{ch}' id{id}@{ppem}: mine={cm} native={cexp:.3}"); }
                if (nm - nexp).abs() <= 0.01 { nok += 1; }
                let _ = ch;
            }
        }
    }
    let crate_ = cok as f64 / ctot as f64;
    let nrate = nok as f64 / ctot as f64;
    eprintln!("GDI-classic advance match: {cok}/{ctot} ({:.1}%); GDI-natural {nok}/{ctot} ({:.1}%); first classic diff: {cfirst}", crate_ * 100.0, nrate * 100.0);
    // GDI-classic carries a whole-pixel ("compatible") advance (the advance phantom grid-fit, rounded).
    // GDI-natural's metric pass runs the glyph program with g448=0x1 (subpixel_hinting only, no
    // compatible_width) from an advance phantom on the 1/16-px grid, then the unrotated consumer rounds the
    // hinted advance to whole pixels (`gdi::display_px`, native's axis-aligned rounding) — so a 26.6 fraction from 30/64 rounds up
    // (a whole-pixel phantom while hinting is inhibited).
    // Both match native for ≥99% of the 15 fonts' glyphs; a drop below these is a hinting regression.
    assert!(crate_ >= 0.97, "GDI-classic advance match regressed to {:.1}% (first: {cfirst})", crate_ * 100.0);
    assert!(nrate >= 0.999, "GDI-natural advance match regressed to {:.1}%", nrate * 100.0);
}

/// Item 1: the smooth (grayscale natural-symmetric) interpreter grid-fits **y to the whole-pixel
/// grid**, matching skrifa's independent v40 "smooth" interpreter exactly on vertical-stem glyphs
/// (y extent < 1/64 px). **x is not cross-checked against v40**: v40 leaves x unhinted, but
/// DirectWrite (part 4) runs x-direction MDRP/MIRP on the 1/16-px subpixel grid, so a stem's width
/// is CVT-controlled and genuinely differs from v40's raw scaled width (Arial 'I'@11px 1.031→1.219,
/// 'L'@11px 4.906→4.875). The x model is validated against the native oracle by the `textx-glyph`
/// conformance scenes, not against v40.
#[test]
fn subpixel_matches_v40_reference() {
    use skrifa::outline::{SmoothMode};
    let Some(bytes) = font_bytes("arial.ttf") else {
        eprintln!("skipping subpixel_matches_v40_reference: {FONT_DIR} not present");
        return;
    };
    let fonts = [("arial.ttf", "HILEFTNKMBDR"), ("verdana.ttf", "HILEFTNKMBDR"), ("tahoma.ttf", "HILEFTNKMBDR")];
    let _ = bytes;
    let mut compared = 0;
    for (file, sample) in fonts {
        let Some(bytes) = font_bytes(file) else { continue };
        let font = FontRef::from_index(&bytes, 0).unwrap();
        let outlines = font.outline_glyphs();
        let charmap = font.charmap();
        for &ppem in &[11.0_f32, 12.0, 13.0, 16.0, 20.0] {
            let mine = Instance::new(&font, ppem as f64, Config { mode: Mode::Grayscale, symmetric: true, version: 42, ..Default::default() }).unwrap();
            let sk = HintingInstance::new(&outlines, Size::new(ppem), LocationRef::default(),
                HintingOptions { engine: Engine::Interpreter, target: Target::Smooth { mode: SmoothMode::Normal, symmetric_rendering: true, preserve_linear_metrics: true } }).unwrap();
            for ch in sample.chars() {
                let Some(gid) = charmap.map(ch) else { continue };
                let h = mine.hint(gid.to_u32() as u16).unwrap();
                if h.points.is_empty() { continue; }
                let mut mb = BBox::default();
                for p in &h.points { mb.add(p.x as f32 / 64.0, p.y as f32 / 64.0); }
                let mut sb = BBox::default();
                outlines.get(gid).unwrap().draw(DrawSettings::hinted(&sk, false), &mut SkBox(&mut sb)).unwrap();
                // skrifa y-up; mine y-up too. Only the y extent is a v40 cross-check (whole-pixel
                // grid-fit); x is CVT-controlled on the subpixel grid and validated by conformance.
                let tol = 0.03;
                let (mh, sh) = (mb.max_y - mb.min_y, sb.max_y - sb.min_y);
                assert!((mh - sh).abs() <= tol,
                    "subpixel y diverged from v40 reference: {file} '{ch}'@{ppem} mine h{mh:.3} skrifa h{sh:.3}");
                compared += 1;
            }
        }
    }
    assert!(compared > 0, "no glyphs compared");
    eprintln!("subpixel vs v40 reference: {compared} vertical-stem glyphs agree");
}

/// A pen that feeds skrifa outline coordinates into a [`BBox`] (skrifa y is up, matching our hinter).
struct SkBox<'a>(&'a mut BBox);
impl OutlinePen for SkBox<'_> {
    fn move_to(&mut self, x: f32, y: f32) { self.0.add(x, y); }
    fn line_to(&mut self, x: f32, y: f32) { self.0.add(x, y); }
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) { self.0.add(cx, cy); self.0.add(x, y); }
    fn curve_to(&mut self, ax: f32, ay: f32, bx: f32, by: f32, x: f32, y: f32) { self.0.add(ax, ay); self.0.add(bx, by); self.0.add(x, y); }
    fn close(&mut self) {}
}

/// Item 2: a composite with USE_MY_METRICS takes its advance from the flagged component's *hinted*
/// phantom points, so an accented letter shares its base letter's hinted advance exactly. Before the
/// fix, Arial 'á' hinted to 18 px while 'a' (and native) gave 17. This invariant needs no oracle.
/// The hinted outline matches native **byte-for-byte**, point by point. The oracle is the
/// hinted contour list DirectWrite hands its scan converter, captured on the reference VM — Arial,
/// natural-symmetric, grid-fit enabled. Those coords are the hinted F26Dot6 points scaled by the
/// scan oversample (×4 for the 4×4 grayscale grid, ×1 for aliased). The grayscale cases include the
/// two that part 5 could not place (`o16`, `a13`): the fix was snapping the phantom points' x to the
/// 1/16-px subpixel grid (`hinting::Instance::execute`), since native MIRPs stems off those phantoms.
/// Fonts are gitignored; skips when absent, as in CI.
#[test]
fn hinted_points_match_native() {
    let Some(suite) = d2d_reference::suite_dir() else { return };
    use skrifa::MetadataProvider;
    // (fixture, char, em px, mode, scan oversample, explicit gid — 0 = resolve `char` via cmap).
    // The `ar*` cases are the 16-text Arabic shaped forms (no cmap entry) that fall back to Arial at
    // 21 px NATURAL_SYMMETRIC: our hinting is byte-exact to
    // native's scan-converter contour list, proving the 16-text residual is the warm/cold rule / shaping, not hinting.
    let cases = [
        ("o16", 'o', 16.0f64, Mode::Grayscale, 4, 0u16), ("a13", 'a', 13.0, Mode::Grayscale, 4, 0),
        ("H20", 'H', 20.0, Mode::Grayscale, 4, 0), ("o20", 'o', 20.0, Mode::Grayscale, 4, 0),
        ("g20", 'g', 20.0, Mode::Grayscale, 4, 0), ("e20", 'e', 20.0, Mode::Grayscale, 4, 0),
        ("R40", 'R', 40.0, Mode::Grayscale, 4, 0), ("Q20", 'Q', 20.0, Mode::Grayscale, 4, 0),
        ("Hal20", 'H', 20.0, Mode::Aliased, 1, 0), ("nal20", 'n', 20.0, Mode::Aliased, 1, 0),
        ("ar995", '\0', 21.0, Mode::Grayscale, 4, 995), ("ar941", '\0', 21.0, Mode::Grayscale, 4, 941),
        ("ar931", '\0', 21.0, Mode::Grayscale, 4, 931), ("ar913", '\0', 21.0, Mode::Grayscale, 4, 913),
        ("ar910", '\0', 21.0, Mode::Grayscale, 4, 910),
    ];
    let Some(bytes) = font_bytes("arial.ttf") else {
        eprintln!("skipping hinted_points_match_native: {FONT_DIR} not present");
        return;
    };
    let font = FontRef::from_index(&bytes, 0).unwrap();
    let cm = font.charmap();
    for (name, ch, em, mode, scale, gid) in cases {
        let path = suite.join("fixtures/text_hinting/fixtures/contours");
        let nat: serde_json::Value = serde_json::from_slice(&std::fs::read(path.join(format!("{name}.json"))).unwrap()).unwrap();
        let nat = &nat[0];
        let (nx, ny, non) = (nat["x"].as_array().unwrap(), nat["y"].as_array().unwrap(), nat["on"].as_array().unwrap());
        let id = if gid != 0 { gid } else { cm.map(ch).unwrap().to_u32() as u16 };
        let inst = Instance::new(&font, em, Config { mode, symmetric: true, version: 42, ..Default::default() }).unwrap();
        let h = inst.hint(id).unwrap();
        assert_eq!(h.points.len(), nx.len(), "{name}: point count mine={} native={}", h.points.len(), nx.len());
        for (i, p) in h.points.iter().enumerate() {
            assert_eq!(p.x * scale, nx[i].as_i64().unwrap() as i32, "{name}: point {i} x");
            assert_eq!(p.y * scale, ny[i].as_i64().unwrap() as i32, "{name}: point {i} y");
            assert_eq!(h.on_curve[i], non[i].as_bool().unwrap(), "{name}: point {i} on-curve");
        }
    }
}

/// Investigation scaffold (inert unless `D2D_MOVERS_DIR` is set): diff our glyph-program mover
/// stream against native point-mover captures listed in `<dir>/manifest.json`
/// (`[{tag,file,char,em,symmetric}]`), reporting the first x/y divergence with native's op label.
#[test]
fn movers_scan() {
    let Ok(dir) = std::env::var("D2D_MOVERS_DIR") else { return };
    let man: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{dir}/manifest.json")).unwrap()).unwrap();
    for m in man.as_array().unwrap() {
        let tag = m["tag"].as_str().unwrap();
        let file = m["file"].as_str().unwrap();
        let ch = m["char"].as_str().unwrap().chars().next().unwrap();
        let em = m["em"].as_f64().unwrap();
        let symmetric = m["symmetric"].as_bool().unwrap();
        let Some(bytes) = font_bytes(file) else { continue };
        let Ok(raw) = std::fs::read(format!("{dir}/{tag}.movers.json")) else { continue };
        let font = FontRef::from_index(&bytes, 0).unwrap();
        let id = font.charmap().map(ch).unwrap().to_u32() as u16;
        let nat: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        let inst = Instance::new(&font, em, Config { mode: Mode::Grayscale, symmetric, version: 42, ..Default::default() }).unwrap();
        d2d::text::hinting::enable_trace();
        let _ = inst.hint(id).unwrap();
        let ours = d2d::text::hinting::take_trace();
        let ncont = ours.first().map(|r| r.x.len()).unwrap_or(0).saturating_sub(4);
        let natg: Vec<&serde_json::Value> = nat.as_array().unwrap().iter().filter(|r| r["x"].as_array().unwrap().len() == ncont).collect();
        if ours.len() != natg.len() { eprintln!("[{tag}] COUNT ours={} native={}", ours.len(), natg.len()); continue; }
        let col = |n: &serde_json::Value, k: &str| -> Vec<i32> { n[k].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect() };
        let mut found = false;
        for (i, (o, n)) in ours.iter().zip(natg.iter()).enumerate() {
            let (nx, ny) = (col(n, "x"), col(n, "y"));
            if o.x[..ncont] != nx[..] || o.y[..ncont] != ny[..] {
                let xd: Vec<String> = o.x[..ncont].iter().zip(&nx).enumerate().filter(|(_, (a, b))| a != b).map(|(k, (a, b))| format!("x{k}:{a}/{b}")).collect();
                let yd: Vec<String> = o.y[..ncont].iter().zip(&ny).enumerate().filter(|(_, (a, b))| a != b).map(|(k, (a, b))| format!("y{k}:{a}/{b}")).collect();
                eprintln!("[{tag}] DIVERGE #{i} {} rp{:?} x[{}]{} y[{}]{}", n["op"].as_str().unwrap(), n["rp"],
                    xd.len(), xd.iter().take(5).cloned().collect::<Vec<_>>().join(" "),
                    yd.len(), yd.iter().take(5).cloned().collect::<Vec<_>>().join(" "));
                found = true; break;
            }
        }
        if !found { eprintln!("[{tag}] exact ({} recs)", ours.len()); }
    }
}

/// The glyph program's per-handler point moves (MDAP/MDRP/MIRP/IP/IUP) match native DirectWrite
/// byte-for-byte in both x and y. The oracle is native's per-move point trace, captured on the
/// reference VM: after each move it records the glyph zone's
/// coordinates and rp0-2. Two fixtures lock the two subpixel regimes — Courier New 'e' @16 px
/// (NATURAL, the 8×1 grid, x not grid-snapped; several points land on a .5 scale tie, so it also
/// pins the half-toward-+∞ outline scaling) and Palatino '6' @27 px (NATURAL_SYMMETRIC, the 4×4 grid,
/// x on the 1/16-px bank). Guards `math::scale` and the subpixel round bank against regression;
/// `fixtures/movers/*.json` are the raw captures.
#[test]
fn movers_match_native() {
    // (fixture, file, char, em, symmetric)
    let cases = [
        ("cour_e16", "cour.ttf", 'e', 16.0, false),
        ("pala_6_27", "pala.ttf", '6', 27.0, true),
    ];
    let Some(_) = font_bytes("cour.ttf") else {
        eprintln!("skipping movers_match_native: {FONT_DIR} not present");
        return;
    };
    let Some(suite) = d2d_reference::suite_dir() else { return };
    let dir = suite.join("fixtures/text_hinting/fixtures/movers");
    for (name, file, ch, em, symmetric) in cases {
        let bytes = font_bytes(file).unwrap();
        let font = FontRef::from_index(&bytes, 0).unwrap();
        let id = font.charmap().map(ch).unwrap().to_u32() as u16;
        let nat: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join(format!("{name}.json"))).unwrap()).unwrap();
        let inst = Instance::new(&font, em, Config { mode: Mode::Grayscale, symmetric, version: 42, ..Default::default() }).unwrap();
        d2d::text::hinting::enable_trace();
        let _ = inst.hint(id).unwrap();
        let ours = d2d::text::hinting::take_trace();
        let ncont = ours.first().map(|r| r.x.len()).unwrap_or(0).saturating_sub(4);
        // Native records the glyph element only (phantoms excluded); match by contour-point count.
        let natg: Vec<&serde_json::Value> =
            nat.as_array().unwrap().iter().filter(|r| r["x"].as_array().unwrap().len() == ncont).collect();
        assert_eq!(ours.len(), natg.len(), "{name}: mover count mine={} native={}", ours.len(), natg.len());
        let col = |n: &serde_json::Value, k: &str| -> Vec<i32> { n[k].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect() };
        for (i, (o, n)) in ours.iter().zip(natg.iter()).enumerate() {
            assert_eq!(o.x[..ncont], col(n, "x")[..], "{name}: record {i} x diverged from native");
            assert_eq!(o.y[..ncont], col(n, "y")[..], "{name}: record {i} y diverged from native");
        }
    }
}

/// The GDI-classic **render pass** (`Config::gdi_render`) is byte-exact to
/// native's rendered hinting. Native hints a GDI-classic glyph twice — a metric pass (the whole-pixel
/// compatible advance) and a render pass (`g448=0x83`, the rendered outline). The render pass does
/// x-subpixel hinting under compatible width, skips the fpgm fn6 curve-grid-fit block via the interpreter's
/// RS[] read-store returning a hardcoded `storage[8]→0` neutralisation, and squeezes x by the overscale inside
/// the IUP point-interpolation (the phase-control post-pass). The fixture is a native point-mover trace of Arial 'u' @13 px in that
/// pass; every point-mover (x and y) matches ours byte-for-byte. Guards the whole render path (RS[8]
/// neutralisation + in-IUP phase control + overscale) against regression.
#[test]
fn gdi_render_movers_match_native() {
    let Some(suite) = d2d_reference::suite_dir() else { return };
    let Some(bytes) = font_bytes("arial.ttf") else {
        eprintln!("skipping gdi_render_movers_match_native: {FONT_DIR} not present");
        return;
    };
    use skrifa::MetadataProvider;
    let font = FontRef::from_index(&bytes, 0).unwrap();
    let id = font.charmap().map('u').unwrap().to_u32() as u16;
    let nat: serde_json::Value = serde_json::from_slice(
        &std::fs::read(suite.join("fixtures/text_hinting/fixtures/movers/u_gdirender13.json")).unwrap(),
    )
    .unwrap();
    let inst = Instance::new(&font, 13.0, Config { mode: Mode::GdiClassic, gdi_render: true, symmetric: false, version: 42, ..Default::default() }).unwrap();
    d2d::text::hinting::enable_trace();
    let _ = inst.hint(id).unwrap();
    let ours = d2d::text::hinting::take_trace();
    let ncont = ours.first().map(|r| r.x.len()).unwrap_or(0).saturating_sub(4);
    let natg = nat.as_array().unwrap();
    assert_eq!(ours.len(), natg.len(), "render mover count mine={} native={}", ours.len(), natg.len());
    let col = |n: &serde_json::Value, k: &str| -> Vec<i32> { n[k].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect() };
    for (i, (o, n)) in ours.iter().zip(natg).enumerate() {
        assert_eq!(o.x[..ncont], col(n, "x")[..], "render record {i} x diverged from native");
        assert_eq!(o.y[..ncont], col(n, "y")[..], "render record {i} y diverged from native");
    }
}

#[test]
fn use_my_metrics_shares_base_advance() {
    use skrifa::MetadataProvider;
    let Some(bytes) = font_bytes("arial.ttf") else {
        eprintln!("skipping use_my_metrics_shares_base_advance: {FONT_DIR} not present");
        return;
    };
    let font = FontRef::from_index(&bytes, 0).unwrap();
    let cm = font.charmap();
    // a-/e-based accented letters are composites with USE_MY_METRICS on the base in Arial; they must
    // share the base's hinted advance in every mode (this is exactly the 'á'=18→17 bug this fixed).
    for (acc, base) in [('á', 'a'), ('é', 'e'), ('à', 'a'), ('â', 'a')] {
        let (Some(ag), Some(bg)) = (cm.map(acc), cm.map(base)) else { continue };
        for ppem in [12.0_f64, 16.0, 20.0, 32.0] {
            for mode in [Mode::GdiClassic, Mode::Aliased, Mode::Grayscale] {
                let inst = Instance::new(&font, ppem, Config { mode, symmetric: true, version: 42, ..Default::default() }).unwrap();
                let a = inst.hint(ag.to_u32() as u16).unwrap().advance;
                let b = inst.hint(bg.to_u32() as u16).unwrap().advance;
                assert_eq!(a, b, "USE_MY_METRICS: '{acc}' advance {a} != base '{base}' advance {b} ({mode:?} @ {ppem}px)");
            }
        }
    }
}

/// `HintedOutline::contours()` flattens curves using the on-curve flags: a round glyph ('o') gains
/// vertices over its control-point count, a rectilinear one ('H') does not. (The raster coverage path
/// must consume these, not the raw points.)
#[test]
fn contours_flatten_curves() {
    use skrifa::MetadataProvider;
    let Some(bytes) = font_bytes("arial.ttf") else { return };
    let font = FontRef::from_index(&bytes, 0).unwrap();
    let cm = font.charmap();
    let inst = Instance::new(&font, 32.0, Config { mode: Mode::Grayscale, symmetric: true, version: 42, ..Default::default() }).unwrap();
    let o = inst.hint(cm.map('o').unwrap().to_u32() as u16).unwrap();
    let ocs = o.contours();
    assert!(!ocs.is_empty() && ocs.iter().map(|c| c.len()).sum::<usize>() > o.points.len(), "'o' curves should flatten to extra vertices");
    let h = inst.hint(cm.map('H').unwrap().to_u32() as u16).unwrap();
    let hcs = h.contours();
    assert_eq!(hcs.iter().map(|c| c.len()).sum::<usize>(), h.points.len(), "'H' is rectilinear: no added vertices");
}

/// GDI-classic x-overscale subsystem: (a) the per-point "link memory" the
/// x-direction MIRP/MSIRP/MDRP/ALIGNRP/IP movers build, and (c) the phase-control post-pass that
/// squeezes x, both against a native phase-control capture of Arial 13 px "Hamburgefons 1234"
/// (`fixtures/phase/ham13.json`: pre-pass `x`, link records, `x_post`, `overscale`).
///
/// * **Link memory** — hint each glyph using the captured GDI-classic grayscale render mode;
///   every `p1`/`p2`/`child`/broken-flag equals native's. The render mode is essential: the metric
///   pass does not build phase-control links.
/// * **Post-pass** — feed native's pre-pass `x` + link records through `execute_phase_control` with
///   native's `overscale`; the result equals native's `x_post` for all 16 glyphs (isolates the
///   post-pass math from the hinting that produced `x`).
#[test]
fn gdi_overscale_link_and_phase_match_native() {
    use d2d::text::hinting::{enable_capture, execute_phase_control, take_capture, LinkRec};
    let Some(suite) = d2d_reference::suite_dir() else { return };
    let Some(bytes) = font_bytes("arial.ttf") else {
        eprintln!("skipping gdi_overscale_link_and_phase_match_native: {FONT_DIR} not present");
        return;
    };
    let path = suite.join("fixtures/text_hinting/fixtures/phase/ham13.json");
    let nat: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let font = FontRef::from_index(&bytes, 0).unwrap();
    let cm = font.charmap();
    // "Hamburgefons 1234" with the (outline-less) space dropped — one record per filled glyph.
    let chars = ['H', 'a', 'm', 'b', 'u', 'r', 'g', 'e', 'f', 'o', 'n', 's', '1', '2', '3', '4'];
    assert_eq!(nat.len(), chars.len());
    let inst = Instance::new(&font, 13.0, Config { mode: Mode::GdiClassic, symmetric: false, version: 42, gdi_render: true, ..Default::default() }).unwrap();
    let col = |r: &serde_json::Value, k: &str| -> Vec<i32> {
        r[k].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect()
    };
    for (rec, ch) in nat.iter().zip(chars) {
        let id = cm.map(ch).unwrap().to_u32() as u16;
        enable_capture();
        let _ = inst.hint(id).unwrap();
        let links_mine = take_capture().expect("capture enabled");
        let nlinks = rec["links"].as_array().unwrap();

        // (a) link memory matches native, point by point (except the known upstream divergences).
        assert_eq!(links_mine.len(), nlinks.len(), "'{ch}': point count");
        for (i, (l, n)) in links_mine.iter().zip(nlinks).enumerate() {
            let mine = (l.p1, l.p2, l.child, l.flags & 1);
            let native = (
                n["p1"].as_i64().unwrap() as i32,
                n["p2"].as_i64().unwrap() as i32,
                n["child"].as_i64().unwrap() as i32,
                n["flags"].as_u64().unwrap() as u16 & 1,
            );
            assert_eq!(mine, native, "'{ch}' pt{i} (p1,p2,child,broken)");
        }

        // (c) post-pass: native pre-pass x + native links → native x_post.
        let mut x = col(rec, "x");
        let org_x = col(rec, "orig_x");
        let mut links: Vec<LinkRec> = nlinks
            .iter()
            .map(|n| LinkRec {
                p1: n["p1"].as_i64().unwrap() as i32,
                p2: n["p2"].as_i64().unwrap() as i32,
                child: n["child"].as_i64().unwrap() as i32,
                flags: n["flags"].as_u64().unwrap() as u16,
            })
            .collect();
        let overscale = rec["overscale"].as_i64().unwrap() as i32;
        let last = *col(rec, "ends").last().unwrap();
        execute_phase_control(&mut x, &org_x, &mut links, overscale, last);
        assert_eq!(x, col(rec, "x_post"), "'{ch}': phase post-pass x");
    }
}

/// Part 1: the GDI-classic x-overscale factor `g+464` that `Instance::hint`
/// exposes — `|FixDiv(hinted compatible advance, fixmul(design advance, scale))|` — equals native's
/// per-glyph value (captured in `ham13.json`) for all 16 "Hamburgefons 1234" glyphs, including the
/// narrow 'f' (3 px compatible advance for a 3.6 px design advance) and the wide 's'/'m'. The
/// existing GDI-classic hinter already produces the right compatible advance, so no separate
/// proportional-redistribution model is needed for these.
/// GDI-classic synthetic bold leaves the outline unemboldened: only the render pass's x-overscale widens, to the
/// compatible advance plus the bold pixel over the advance scaled with the bold strength added in font units.
/// Native's scanned contours (x in eighths of a pixel, as the grayscale scan takes them; y in 1/64 px), regular
/// and bold, for Trebuchet 'x' at ppem 29 and Tahoma 't' at ppem 13 (whose emboldened advance rounds up a unit).
#[test]
fn gdi_classic_bold_widens_only_the_overscale() {
    use skrifa::MetadataProvider;
    let cases: [(&str, char, f64, [&[i32]; 2], &[i32]); 2] = [
        (
            "trebuc.ttf",
            'x',
            29.0,
            [
                &[5904, 3832, 1728, 128, 3032, 352, 1920, 3832, 5840, 7280, 4632, 7568],
                &[6088, 3952, 1752, 152, 3136, 384, 1952, 3912, 6016, 7456, 4728, 7752],
            ],
            &[0, 357, 0, 0, 496, 960, 960, 641, 960, 960, 504, 0],
        ),
        (
            "tahoma.ttf",
            't',
            13.0,
            [
                &[1424, 944, 416, 416, 416, 0, 0, 416, 416, 1072, 1072, 2064, 2064, 1072, 1072, 1072, 1088, 1144, 1192, 1416, 1600, 1728, 1976, 2032, 2064, 2064, 1904, 1568],
                &[1696, 1160, 552, 552, 552, 136, 136, 552, 552, 1208, 1208, 2432, 2432, 1208, 1208, 1208, 1232, 1296, 1360, 1632, 1856, 2024, 2328, 2392, 2432, 2432, 2248, 1864],
            ],
            &[0, 0, 68, 142, 384, 384, 448, 448, 576, 576, 448, 448, 384, 384, 176, 143, 107, 91, 78, 64, 64, 64, 64, 64, 64, 10, 5, 0],
        ),
    ];
    for (file, ch, ppem, [regular, bold], y) in cases {
        let Some(bytes) = font_bytes(file) else {
            eprintln!("skipping gdi_classic_bold_widens_only_the_overscale: {FONT_DIR} has no {file}");
            return;
        };
        let font = FontRef::from_index(&bytes, 0).unwrap();
        let id = font.charmap().map(ch).unwrap().to_u32() as u16;
        for (bold_fu, x) in [(0, regular), (40, bold)] {
            let config = Config { mode: Mode::GdiClassic, gdi_render: true, symmetric: false, version: 42, bold_fu, ..Default::default() };
            let h = Instance::new(&font, ppem, config).unwrap().hint(id).unwrap();
            let ours: Vec<[i32; 2]> = h.points.iter().map(|p| [p.x - h.points[0].x, p.y - h.points[0].y]).collect();
            let native: Vec<[i32; 2]> = x.iter().zip(y).map(|(&px, &py)| [(px - x[0]) / 8, py - y[0]]).collect();
            assert_eq!(ours, native, "{file} '{ch}' ppem {ppem} bold strength {bold_fu}");
        }
    }
}

#[test]
fn gdi_overscale_factor_matches_native() {
    let Some(suite) = d2d_reference::suite_dir() else { return };
    let Some(bytes) = font_bytes("arial.ttf") else {
        eprintln!("skipping gdi_overscale_factor_matches_native: {FONT_DIR} not present");
        return;
    };
    let path = suite.join("fixtures/text_hinting/fixtures/phase/ham13.json");
    let nat: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let font = FontRef::from_index(&bytes, 0).unwrap();
    let cm = font.charmap();
    let chars = ['H', 'a', 'm', 'b', 'u', 'r', 'g', 'e', 'f', 'o', 'n', 's', '1', '2', '3', '4'];
    let inst = Instance::new(&font, 13.0, Config { mode: Mode::GdiClassic, symmetric: false, version: 42, ..Default::default() }).unwrap();
    for (rec, ch) in nat.iter().zip(chars) {
        let id = cm.map(ch).unwrap().to_u32() as u16;
        let h = inst.hint(id).unwrap();
        assert_eq!(h.overscale, rec["overscale"].as_i64().unwrap() as i32, "'{ch}' overscale (adv {})", h.advance);
    }
}

/// A storage flag reads as 0 under sub-pixel positioning only inside the font-tool functions DirectWrite
/// recognises. Impact's function 66 is not one, so its diagonal caps keep their advance phantom (VM:
/// GetGdiCompatibleGlyphAdvances, GDI natural, ppem 26).
#[test]
fn impact_diagonal_caps_keep_their_gdi_natural_advance() {
    let Some(bytes) = font_bytes("impact.ttf") else { return };
    let font = FontRef::from_index(&bytes, 0).expect("font");
    let config = Config { mode: Mode::GdiNatural, symmetric: false, version: 42, subpixel_advance: true, ..Default::default() };
    let inst = Instance::new(&font, 26.0, config).unwrap();
    let cm = font.charmap();
    for (ch, native) in [('A', 13.0), ('V', 14.0), ('W', 21.0), ('Y', 12.0), ('o', 13.0)] {
        let px = inst.hint(cm.map(ch).unwrap().to_u32() as u16).unwrap().advance_px();
        assert_eq!((px + 0.5).floor(), native, "'{ch}' advance {px}");
    }
}

/// GDI classic measures an advance in the scaler's six-times x-overscaled space, whose 16.16 advance wraps
/// from 32768/6 px; Comic Sans 'M' at ppem 6250 (em 5000 at 120 dpi) reports the wrapped width (VM:
/// -4324 DIPs, a whole -5405 px), 'W' beside it keeps its own (its six-times extent overflows first).
#[test]
fn comic_sans_gdi_classic_advance_wraps_at_huge_ppem() {
    let Some(bytes) = font_bytes("comic.ttf") else { return };
    let font = FontRef::from_index(&bytes, 0).expect("font");
    let cm = font.charmap();
    let du = |ch: char| d2d::text::gdi::gdi_compat_advance_du(&font, cm.map(ch).unwrap().to_u32() as u16, 5000.0, 1.25, true, None, 2048, false).unwrap();
    // DIPs = du * em / upem, rounded to whole pixels at 1.25 px per DIP.
    let px = |du: i32| (du as f64 * 6250.0 / 2048.0).round() as i32;
    assert_eq!(px(du('M')), -5405);
    assert_eq!(px(du('W')), 6497);
}
