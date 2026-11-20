// Local glyph-coverage oracle: run `unhinted_glyph_coverage` (grid-fit-disabled, the
// IDWriteGlyphRunAnalysis surface) for one glyph of a given font natively — no VM, no Wine — and with
// `D2D_TEXOUT=<path>` write the raw alpha texture, to diff byte-for-byte against a native dwrite.dll
// `dwrite-dump` capture (crates/dwrite-ffi/analysis.py fixtures). Prints the texture's bounds so the
// caller can align them. This is how the composite-accent residual was isolated and verified.
//
// Usage: cargo run -p d2d --example arcprobe -- <font.ttf> <char> [em] [baseline_x] [baseline_y] [aa] [mode]
//   aa:   1 ClearType (default), 2 grayscale, 3 aliased        mode: DWRITE_RENDERING_MODE1 (3 = GDI_NATURAL)
//   sims: DWRITE_FONT_SIMULATIONS (0 none, 1 bold) — drives the synthetic-bold embolden.
//   e.g.  D2D_TEXOUT=/tmp/e.ct.bin cargo run -p d2d --example arcprobe -- verdana.ttf é 34.435 5 30 1 3 1
use skrifa::{FontRef, MetadataProvider};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = &a[1];
    let ch: char = a[2].chars().next().unwrap();
    let em: f32 = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(34.43518829345703);
    let bx: f32 = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(5.0);
    let by: f32 = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(30.0);
    let aa: u32 = a.get(6).map(|s| s.parse().unwrap()).unwrap_or(d2d::text::AA_CLEARTYPE);
    let mode: u32 = a.get(7).map(|s| s.parse().unwrap()).unwrap_or(3); // GDI_NATURAL
    let sims: u32 = a.get(8).map(|s| s.parse().unwrap()).unwrap_or(0);
    let data = std::fs::read(path).unwrap();
    let font = FontRef::from_index(&data, 0).unwrap();
    let gid = font.charmap().map(ch).unwrap();
    eprintln!("glyph '{ch}' gid={} em={em} baseline=[{bx},{by}]", gid.to_u32());
    let glyphs = [d2d::text::Glyph { id: gid.to_u32() as u16, advance: 0.0, offset: [0.0, 0.0] }];
    let run = d2d::text::GlyphRun {
        font_data: &data,
        face_index: 0,
        em_size: em,
        glyphs: &glyphs,
        baseline_origin: [bx, by],
        measuring_mode: 2,
        is_sideways: false,
        bidi_level: 0,
        no_snap: false,
        simulations: sims,
        coords: &[],
    };
    let dt = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let tex = d2d::text::unhinted_glyph_coverage(
        &dt, &run, &font, aa, mode,
        d2d::text::hinting::Rasterizer::DirectWrite, false, None,
    );
    eprintln!("tex {}x{} top={} left={}", tex.width(), tex.height(), tex.top, tex.left);
    if let Ok(out) = std::env::var("D2D_TEXOUT") {
        std::fs::write(&out, &tex.data).unwrap();
        eprintln!("wrote {} bytes to {out}", tex.data.len());
    }
}
