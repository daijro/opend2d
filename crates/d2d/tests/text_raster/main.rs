//! Phase 2D-2 scan-converter tests: `d2d::text::raster::grayscale` alpha textures compared against
//! native `IDWriteGlyphRunAnalysis::CreateAlphaTexture` captures (on the reference VM,
//! `DWRITE_GRID_FIT_MODE_DISABLED` so the scan converter's input is the unhinted outline). Needs the
//! gitignored `ref/windows/Fonts`; skips with a message when absent.

use d2d::text::raster::{self, AlphaTexture};
use d2d::text::{Glyph, GlyphRun};
use skrifa::FontRef;
use std::path::PathBuf;

const ROOT: &str = env!("CARGO_MANIFEST_DIR"); // crates/d2d
// Captured fixtures live in the conformance suite (the conformance/suite submodule).
fn fix() -> PathBuf {
    d2d_reference::suite_dir().unwrap().join("fixtures/text_raster/fixtures")
}

// Captured fixtures (Arial, natural-symmetric, grid-fit disabled). The ported scan
// converter reproduces native bounds byte-exactly on every one (including `a13`, which the earlier
// approximation could not), so all are in the strict set.
const FIXTURES: &[&str] = &["o20", "H20", "e20", "g20", "o16", "R40", "a13"];
// Both grayscale and ClearType fixtures are byte-exact. ClearType must form implied midpoints and
// scan crossings in the native y-up frame before flipping the final rows; e20/g20 witness this.
// o16 also witnesses positioning on an integral oversampled pixel at its fractional baseline.
const GRAY_EXACT: &[&str] = &["o20", "H20", "e20", "g20", "o16", "R40", "a13"];
const CT_EXACT: &[&str] = FIXTURES;

#[derive(serde::Deserialize)]
struct GlyphSpec {
    id: u16,
    advance: f32,
}
#[derive(serde::Deserialize)]
struct Bounds {
    bounds: [i32; 4],
}
#[derive(serde::Deserialize)]
struct Fixture {
    font: String,
    em_size: f32,
    transform: [f64; 6],
    baseline: [f32; 2],
    glyphs: Vec<GlyphSpec>,
    gray: Bounds,
    cleartype: Bounds,
}

fn font_bytes(file: &str) -> Option<Vec<u8>> {
    let dir = std::env::var_os("D2D_FONTS").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(ROOT).join("../../ref/windows/Fonts"));
    std::fs::read(dir.join(file)).ok()
}

fn load(name: &str) -> Option<(Fixture, Vec<u8>)> {
    let fx: Fixture = serde_json::from_slice(&std::fs::read(fix().join(format!("{name}.json"))).ok()?).unwrap();
    let font = font_bytes(&fx.font)?;
    Some((fx, font))
}

fn glyphs_of(fx: &Fixture) -> Vec<Glyph> {
    fx.glyphs.iter().map(|g| Glyph { id: g.id, advance: g.advance, offset: [0.0, 0.0] }).collect()
}

fn run_of<'a>(fx: &Fixture, glyphs: &'a [Glyph]) -> GlyphRun<'a> {
    GlyphRun {
        font_data: &[],
        face_index: 0,
        em_size: fx.em_size,
        glyphs,
        baseline_origin: fx.baseline,
        measuring_mode: 0,
        is_sideways: false,
        bidi_level: 0,
        no_snap: true,
        simulations: 0,
        coords: &[],
    }
}

/// Render the grayscale texture for a fixture.
fn render(name: &str) -> Option<(AlphaTexture, [i32; 4], Vec<u8>)> {
    let (fx, font) = load(name)?;
    let font = FontRef::new(&font).unwrap();
    let glyphs = glyphs_of(&fx);
    let tex = raster::grayscale(&fx.transform, &run_of(&fx, &glyphs), &font, raster::DEFAULT_DROPOUT);
    let want = std::fs::read(fix().join(format!("{name}.gray.bin"))).unwrap();
    Some((tex, fx.gray.bounds, want))
}

/// Render the ClearType (3 bytes/pixel) texture for a fixture.
fn render_ct(name: &str) -> Option<(AlphaTexture, [i32; 4], Vec<u8>)> {
    let (fx, font) = load(name)?;
    let font = FontRef::new(&font).unwrap();
    let glyphs = glyphs_of(&fx);
    let tex = raster::cleartype(&fx.transform, &run_of(&fx, &glyphs), &font, raster::DEFAULT_DROPOUT);
    let want = std::fs::read(fix().join(format!("{name}.ct.bin"))).unwrap();
    Some((tex, fx.cleartype.bounds, want))
}

/// The ported scan converter reproduces native `CreateAlphaTexture` (`ALIASED_1x1`) byte-exactly in
/// **bounds** on every fixture, byte-for-byte everywhere on the exact cases (`H`, `e20`, `a13`), and
/// within one 4×4 sample (`|Δ|≤16`) otherwise. The residual is dropout control, which this module does
/// not run (see `raster`'s "Residual" section); `R40`'s one top-right corner reaches `|Δ|=63`.
#[test]
fn grayscale_matches_native() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    if render("o20").is_none() {
        eprintln!("skip grayscale_matches_native: ref/windows/Fonts/arial.ttf absent");
        return;
    }
    for &name in FIXTURES {
        let (tex, bounds, want) = render(name).unwrap();
        let got = [tex.left, tex.top, tex.right, tex.bottom];
        assert_eq!(got, bounds, "{name}: grayscale bounds mismatch");
        assert_eq!(tex.data.len(), want.len(), "{name}: size mismatch");
        let exact = tex.data.iter().zip(&want).filter(|(a, b)| a == b).count();
        let max_delta = tex.data.iter().zip(&want).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap_or(0);
        let pct = 100.0 * exact as f64 / want.len() as f64;
        eprintln!("gray {name:4}: {exact}/{} exact ({pct:.1}%), max |Δ|={max_delta}", want.len());
        if GRAY_EXACT.contains(&name) {
            assert_eq!(exact, want.len(), "{name}: expected byte-exact grayscale, got {pct:.1}%");
        }
        assert!(max_delta <= 64, "{name}: grayscale coverage off by too much (max |Δ|={max_delta})");
        assert!(pct >= 96.0, "{name}: grayscale only {pct:.1}% byte-exact");
    }
}

/// ClearType 3×1 (`CreateAlphaTexture3x1`): exact bounds on every fixture, byte-exact on `H`/`a13`,
/// within a filter step elsewhere (the same dropout-control residual as grayscale, amplified by the
/// three overlapping colour boxes). Pins the model: 6× horizontal + three offset width-6 boxes, 5×
/// vertical `{4,9,10,9,4}`, `round(255·s/216)` — both filter tables derived from the algorithm, never copied.
#[test]
fn cleartype_matches_native() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    if render_ct("o20").is_none() {
        eprintln!("skip cleartype_matches_native: ref/windows/Fonts/arial.ttf absent");
        return;
    }
    for &name in FIXTURES {
        let (tex, bounds, want) = render_ct(name).unwrap();
        let got = [tex.left, tex.top, tex.right, tex.bottom];
        assert_eq!(got, bounds, "{name}: cleartype bounds mismatch");
        assert_eq!(tex.data.len(), want.len(), "{name}: size mismatch");
        let exact = tex.data.iter().zip(&want).filter(|(a, b)| a == b).count();
        let max_delta = tex.data.iter().zip(&want).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap_or(0);
        let pct = 100.0 * exact as f64 / want.len() as f64;
        eprintln!("ct   {name:4}: {exact}/{} exact ({pct:.1}%), max |Δ|={max_delta}", want.len());
        if CT_EXACT.contains(&name) {
            assert_eq!(exact, want.len(), "{name}: expected byte-exact cleartype, got {pct:.1}%");
        }
        assert!(max_delta <= 24, "{name}: cleartype coverage off by too much (max |Δ|={max_delta})");
        assert!(pct >= 92.0, "{name}: cleartype only {pct:.1}% byte-exact");
    }
}

/// Aliased (bi-level, ×1) reproduces native's X-on-centre topology: a vertical stem edge landing
/// exactly on a pixel-centre column (device x.5) covers that column, so Arial `H` at 20px keeps both
/// 2px stems (native `textmodes-02`). Without it the left stem's right edge (device 9.5, dropped by
/// the per-row exit rounding) would leave a 1px stem — the residual this path fixed.
#[test]
fn aliased_x_on_center() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    let Some(font) = font_bytes("arial.ttf") else {
        eprintln!("skip aliased_x_on_center: ref/windows/Fonts/arial.ttf absent");
        return;
    };
    let font = FontRef::new(&font).unwrap();
    use skrifa::MetadataProvider;
    let gid = font.charmap().map('H').unwrap().to_u32() as u16;
    let glyphs = [Glyph { id: gid, advance: 0.0, offset: [0.0, 0.0] }];
    let run = GlyphRun {
        font_data: &[], face_index: 0, em_size: 20.0, glyphs: &glyphs,
        baseline_origin: [6.0, 24.0], measuring_mode: 0, is_sideways: false, bidi_level: 0, no_snap: false, simulations: 0, coords: &[],
    };
    let tex = raster::aliased(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &run, &font, raster::DEFAULT_DROPOUT);
    assert_eq!([tex.left, tex.top, tex.right, tex.bottom], [8, 10, 19, 24], "H aliased bounds");
    let on = |x: i32, y: i32| tex.data[((y - tex.top) * tex.width() + (x - tex.left)) as usize] > 0;
    // Left stem = cols 8,9 (9 is the X-on-centre column); right stem = cols 17,18 (whole-pixel edges).
    for y in [10, 11, 20, 23] {
        assert!(on(8, y) && on(9, y), "left stem both columns at y={y}");
        assert!(on(17, y) && on(18, y), "right stem both columns at y={y}");
        assert!(!on(10, y), "left stem exactly 2px (col 10 empty) at y={y}");
    }
}

/// Grid-fit a fixture's first glyph through the hinting interpreter in `mode` and lay its hinted
/// **quadratics** out in device space, as `text::render_glyph_run`'s hinted path does (identity-
/// transform fixtures: y snapped to the pixel grid, x on the 4-phase natural-symmetric grid). Returns
/// the device-point contours (parallel points + on-curve flags) and the font's resolved dropout mode,
/// ready for [`raster::coverage_outline`].
fn hinted_contours(name: &str, mode: d2d::text::hinting::Mode) -> (Vec<(Vec<[f64; 2]>, Vec<bool>)>, u16) {
    use d2d::text::hinting::{Config, Instance};
    let (fx, font) = load(name).unwrap();
    let font = FontRef::new(&font).unwrap();
    let g = &fx.glyphs[0];
    let (bx, by) = (fx.baseline[0], fx.baseline[1]);
    let fbx = bx.floor();
    let px = fbx + ((bx - fbx) * 4.0).round() / 4.0;
    let py = by.round();
    let cfg = Config { rasterizer: Default::default(), mode, symmetric: true, bgr: false, is_rotated: false, is_stretched: false, axes_swapped: false, em_y: None, version: 42, gdi_render: false, cleartype: false, subpixel_advance: false, grid_fit: true };
    let inst = Instance::new(&font, fx.em_size as f64, cfg).unwrap();
    let drop = { let (en, st) = inst.scan_control(); raster::dropout_mode(en, st) };
    let h = inst.hint(g.id).unwrap();
    let mut contours: Vec<(Vec<[f64; 2]>, Vec<bool>)> = Vec::new();
    let mut s = 0usize;
    for &e in &h.contour_ends {
        let e = e as usize;
        if e >= s && e < h.points.len() {
            let pts = h.points[s..=e].iter().map(|p| [px as f64 + p.x as f64 / 64.0, py as f64 - p.y as f64 / 64.0]).collect();
            contours.push((pts, h.on_curve[s..=e].to_vec()));
        }
        s = e + 1;
    }
    (contours, drop)
}

/// The **hinted** glyph path: grid-fit each fixture glyph through the TrueType interpreter
/// (`hinting`), hand the hinted **quadratics** to [`raster::coverage_outline`], and compare to the
/// native `IDWriteGlyphRunAnalysis` hinted capture (`grid_fit` ENABLED). Keeping curves as quadratics
/// so they reach the native `split_quad`/`calc_spline` path — plus computing implied on-curve midpoints
/// in oversampled integer space (`(a+b+1)>>1`, native's scan-converter glyph fill) — makes `o`/`Q`/`H`/`g`/`e`/`R`
/// byte-exact. `o16`/`a13` stay ≤2 samples off: with the fill proven faithful, that residual is the
/// hinting interpreter's outline placement, upstream of this module (see the `raster` "Residual" note).
#[test]
fn coverage_outline_matches_native_hinted() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    use d2d::text::hinting::Mode;
    use d2d::text::raster::Kind;
    if load("o20").is_none() {
        eprintln!("skip coverage_outline_matches_native_hinted: fonts absent");
        return;
    }
    // Hinted device-pixel bounds from the native captures (grid_fit enabled, natural-symmetric).
    let hbounds: &[(&str, [i32; 4])] = &[
        ("o20", [1, 7, 12, 18]), ("g20", [1, 4, 11, 19]), ("e20", [1, 7, 12, 18]),
        ("R40", [5, 7, 31, 36]), ("o16", [1, 5, 9, 14]), ("a13", [1, 4, 7, 11]),
        ("H20", [2, 3, 14, 18]), ("Q20", [1, 3, 16, 19]),
    ];
    let exact: &[&str] = &["H20", "o20", "Q20", "g20", "e20", "R40", "o16", "a13"];
    for &(name, hb) in hbounds {
        let (contours, drop) = hinted_contours(name, Mode::Grayscale);
        let tex = raster::coverage_outline(&contours, drop, Kind::Grayscale, (0, 0));
        let want = std::fs::read(fix().join(format!("{name}.hinted.gray.bin"))).unwrap();
        assert_eq!([tex.left, tex.top, tex.right, tex.bottom], hb, "{name}: hinted bounds mismatch");
        assert_eq!(tex.data.len(), want.len(), "{name}: size mismatch");
        let ex = tex.data.iter().zip(&want).filter(|(a, b)| a == b).count();
        let md = tex.data.iter().zip(&want).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap_or(0);
        let pct = 100.0 * ex as f64 / want.len() as f64;
        eprintln!("hinted {name:4}: {ex}/{} exact ({pct:.1}%), max |Δ|={md}", want.len());
        if exact.contains(&name) {
            assert_eq!(ex, want.len(), "{name}: expected byte-exact hinted coverage, got {pct:.1}%");
        }
        assert!(md <= 32, "{name}: hinted coverage off by too much (max |Δ|={md})");
        assert!(pct >= 93.0, "{name}: hinted coverage only {pct:.1}% byte-exact");
    }
}

/// The **hinted ClearType** path (`coverage_outline` with `Kind::ClearType`) — the path grid-fit CT
/// scenes (`text-02/05/06`) use — against native `CreateAlphaTexture(CLEARTYPE_3x1)` captured grid-fit
/// **enabled** (`<name>.hinted.ct.bin`). Straight/simple glyphs (`H`, `a`) are byte-exact: the filter
/// and 6×5 scan are faithful there. Curved glyphs keep a residual (feeding native's own CT contour list
/// into this scan diverges too) — the curve DDA at the finer 6×5 grid plus the hinting interpreter's
/// CT-mode placement (upstream), not the filter. Pinning it needs native's 1-bit 6×5 pre-resolve grid.
#[test]
fn hinted_cleartype_matches_native() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    use d2d::text::hinting::Mode;
    use d2d::text::raster::Kind;
    if load("o20").is_none() {
        eprintln!("skip hinted_cleartype_matches_native: fonts absent");
        return;
    }
    let hbounds: &[(&str, [i32; 4])] = &[
        ("e20", [1, 7, 12, 18]), ("o20", [1, 7, 12, 18]), ("g20", [1, 4, 12, 19]),
        ("o16", [1, 5, 10, 14]), ("a13", [1, 4, 8, 11]), ("H20", [2, 3, 15, 18]),
    ];
    let exact: &[&str] = &["H20", "a13"]; // straight / simple: filter + 6×5 scan byte-exact
    for &(name, hb) in hbounds {
        let (contours, drop) = hinted_contours(name, Mode::ClearType);
        let tex = raster::coverage_outline(&contours, drop, Kind::ClearType, (0, 0));
        let want = std::fs::read(fix().join(format!("{name}.hinted.ct.bin"))).unwrap();
        assert_eq!([tex.left, tex.top, tex.right, tex.bottom], hb, "{name}: hinted CT bounds mismatch");
        assert_eq!(tex.data.len(), want.len(), "{name}: size mismatch");
        let ex = tex.data.iter().zip(&want).filter(|(a, b)| a == b).count();
        let md = tex.data.iter().zip(&want).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap_or(0);
        let pct = 100.0 * ex as f64 / want.len() as f64;
        eprintln!("hinted-ct {name:4}: {ex}/{} exact ({pct:.1}%), max |Δ|={md}", want.len());
        if exact.contains(&name) {
            assert_eq!(ex, want.len(), "{name}: expected byte-exact hinted ClearType, got {pct:.1}%");
        }
    }
}
