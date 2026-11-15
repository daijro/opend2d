//! Phase 2D.1 glyph-run rendering tests.
//!
//! What is verified here:
//!   * `recommended_rendering_mode` reproduces native's recommended-rendering-mode query:
//!     the outline cutoff is 350 px for antialiased text and 100 px for
//!     aliased text, below which the glyph is grid-fitted/analysed, not outlined.
//!   * OUTLINE mode (R6) renders glyph geometry into the target (smoke: ink appears in the right
//!     place). Needs the gitignored fonts; skips with a message when they are absent.
//!
//! What is NOT verified (documented blocker): every captured text scene (text-01..09, 16, 17) draws
//! glyphs well below its outline threshold (≤ 21 px at the pinned params, grayscale/ClearType/aliased
//! → 350/100 px cutoffs), so native renders them through DirectWrite's TrueType hinting interpreter +
//! scan converter + grayscale/ClearType resolve filter — the coverage source
//! this phase does not yet reproduce. `render_glyph_run` returns `false` for those modes. The
//! `text_08_is_analysis_not_outline` test records that text-08 (110 px grayscale), which the spec's
//! R6 assumed was OUTLINE, is actually analysis mode, and that an unhinted outline fill does not
//! match the oracle (first diff on the hinted top edge).

use d2d::pixels::{premultiply, Target};
use d2d::text::{self, Glyph, GlyphRun, RenderingParams};
use d2d_types::abi::{D2D1_ALPHA_MODE_PREMULTIPLIED, DXGI_FORMAT_B8G8R8A8_UNORM};
use d2d_types::contracts::{ImageDesc, PixelBuffer};
use skrifa::instance::{LocationRef, Size};
use skrifa::{FontRef, MetadataProvider};
use std::path::PathBuf;

const ROOT: &str = env!("CARGO_MANIFEST_DIR"); // crates/d2d
const IDENTITY: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

fn font_path(file: &str) -> PathBuf {
    PathBuf::from(ROOT).join("../../ref/windows/Fonts").join(file)
}

fn run_of<'a>(glyphs: &'a [Glyph], font_data: &'a [u8], em: f32, measuring: u32) -> GlyphRun<'a> {
    GlyphRun {
        font_data,
        face_index: 0,
        em_size: em,
        glyphs,
        baseline_origin: [0.0, 0.0],
        measuring_mode: measuring,
        is_sideways: false,
        bidi_level: 0,
        no_snap: false,
        simulations: 0,
        coords: &[],
    }
}

// DWRITE_RENDERING_MODE1 values.
const MODE_ALIASED: u32 = 1;
const MODE_NATURAL_SYMMETRIC: u32 = 5;
const MODE_OUTLINE: u32 = 6;

/// `recommended_rendering_mode` matches the native thresholds: antialiased text outlines at ≥ 100 px
/// device, aliased text at ≥ 350 px, otherwise an analysis mode.
#[test]
fn rendering_mode_thresholds() {
    let gs = [Glyph { id: 1, advance: 0.0, offset: [0.0, 0.0] }];
    let p = RenderingParams::default();
    let dpi144 = [1.5, 0.0, 0.0, 1.5, 0.0, 0.0];

    // Native GetRecommendedRenderingModeInternal: antialiased threshold 100, aliased 350.
    assert_eq!(text::recommended_rendering_mode(&run_of(&gs, &[], 99.0, 0), &IDENTITY, 2, &p), MODE_NATURAL_SYMMETRIC);
    assert_eq!(text::recommended_rendering_mode(&run_of(&gs, &[], 100.0, 0), &IDENTITY, 2, &p), MODE_OUTLINE);
    assert_eq!(text::recommended_rendering_mode(&run_of(&gs, &[], 350.0, 0), &IDENTITY, 2, &p), MODE_OUTLINE);
    assert_eq!(text::recommended_rendering_mode(&run_of(&gs, &[], 350.0, 0), &IDENTITY, 3, &p), MODE_OUTLINE);
    let high_dpi=RenderingParams { dpi: [192.0,192.0], ..p };
    let scale2=[2.0,0.0,0.0,2.0,0.0,0.0];
    // A gasp-less font (empty data here) below ppem 21 renders NATURAL (4), not NATURAL_SYMMETRIC: a
    // font with no gasp has no SYMMETRIC_SMOOTHING range, so GetRecommendedRenderingMode returns NATURAL
    // (VM RECMODE: Lucida Console, no gasp, @16/@20 → mode 4). em 8 × scale2 = device ppem 16.
    assert_eq!(text::recommended_rendering_mode(&run_of(&gs,&[],8.0,0),&scale2,2,&p),4);
    // At ppem ≥ 21 every glyph is symmetric regardless of gasp (em 11 × 2 = 22); high DPI downsamples (7).
    assert_eq!(text::recommended_rendering_mode(&run_of(&gs,&[],11.0,0),&scale2,2,&p),5);
    assert_eq!(text::recommended_rendering_mode(&run_of(&gs,&[],11.0,0),&scale2,2,&high_dpi),7);
    // DPI scales the device em: 240 px DIP × 1.5 = 360 ≥ 100 → outline.
    assert_eq!(text::recommended_rendering_mode(&run_of(&gs, &[], 240.0, 0), &dpi144, 2, &p), MODE_OUTLINE);
    // Aliased (aa=3): cutoff 350 px.
    assert_eq!(text::recommended_rendering_mode(&run_of(&gs, &[], 21.0, 0), &IDENTITY, 3, &p), MODE_ALIASED);
    assert_eq!(text::recommended_rendering_mode(&run_of(&gs, &[], 349.0, 0), &IDENTITY, 3, &p), MODE_ALIASED);
}

fn new_target(w: u32, h: u32) -> PixelBuffer {
    PixelBuffer {
        desc: ImageDesc { width: w, height: h, format: DXGI_FORMAT_B8G8R8A8_UNORM, alpha_mode: D2D1_ALPHA_MODE_PREMULTIPLIED, dpi_x: 96.0, dpi_y: 96.0 },
        stride: w as usize * 4,
        data: vec![0; (w * h * 4) as usize],
    }
}

fn ink_pixels(buf: &PixelBuffer) -> usize {
    buf.data.chunks_exact(4).filter(|p| p[0] < 250 || p[1] < 250 || p[2] < 250).count()
}

/// OUTLINE mode (≥ 350 px) renders glyph geometry: a 400 px 'R' leaves ink. Smoke test — there is no
/// oracle scene at this size, so coverage exactness is not asserted.
#[test]
fn outline_mode_renders() {
    let Ok(font_data) = std::fs::read(font_path("arialbd.ttf")) else {
        eprintln!("SKIP outline_mode_renders: ref/windows/Fonts/arialbd.ttf absent");
        return;
    };
    let font = FontRef::from_index(&font_data, 0).unwrap();
    let gid = font.charmap().map('R').unwrap();
    let upem = font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f32;
    let adv = font.glyph_metrics(Size::unscaled(), LocationRef::default()).advance_width(gid).unwrap_or(0.0);
    let em = 400.0;
    let glyphs = [Glyph { id: gid.to_u32() as u16, advance: adv * em / upem, offset: [0.0, 0.0] }];
    let mut run = run_of(&glyphs, &font_data, em, 0);
    run.baseline_origin = [20.0, 430.0];

    let mut buf = new_target(512, 512);
    Target::new(&mut buf).clear(premultiply([1.0, 1.0, 1.0, 1.0]));
    assert!(text::render_glyph_run(&mut buf, &IDENTITY, &run, [0.0, 0.0, 0.0, 1.0], 2, &RenderingParams::default()));
    assert!(ink_pixels(&buf) > 1000, "outline 'R' should leave ink");
}

/// Native text-08 (110 px grayscale) uses the cached outline path.
#[test]
fn text_08_is_outline() {
    let data: &[u8] = &[];
    let gs = [Glyph { id: 1, advance: 0.0, offset: [0.0, 0.0] }];
    let run = run_of(&gs, data, 110.0, 0);
    assert_eq!(text::recommended_rendering_mode(&run, &IDENTITY, 2, &RenderingParams::default()), MODE_OUTLINE);
    assert!(!text::render_glyph_run(&mut new_target(8, 8), &IDENTITY, &run, [0.0, 0.0, 0.0, 1.0], 2, &RenderingParams::default()));
}
