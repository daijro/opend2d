// Run-level sim-bold coverage oracle: reproduce a native `dwrite-dump` capture (its `out.json`) through
// `unhinted_glyph_coverage` (the grid-fit-disabled analysis surface) — no VM, no Wine — and write the raw
// alpha texture so it can be diffed byte-for-byte against the capture's `.ct.bin` / `.gray.bin`. Drives
// the synthetic-bold port (`sims` comes from the capture, so bold runs embolden through `unhinted_ds`).
//
// Usage: cargo run -p d2d --example simboldprobe -- <font.ttf> <out.json> <aa> <out.bin>
//   aa: 1 ClearType, 2 grayscale, 3 aliased.  Prints bounds+dims (align vs the capture's out.json bounds).
use serde_json::Value;
use skrifa::FontRef;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (font_path, json_path, aa_s, out_path) = (&a[1], &a[2], &a[3], &a[4]);
    let aa: u32 = aa_s.parse().unwrap();
    let data = std::fs::read(font_path).unwrap();
    let font = FontRef::from_index(&data, 0).unwrap();
    let txt = std::fs::read_to_string(json_path).unwrap();
    let j: Value = serde_json::from_str(txt.trim_start_matches('\u{feff}')).unwrap();
    let em = j["em_size"].as_f64().unwrap() as f32;
    let base = &j["baseline"];
    let bx = base[0].as_f64().unwrap() as f32;
    let by = base[1].as_f64().unwrap() as f32;
    let measuring = j["measuring"].as_u64().unwrap() as u32;
    let mode = j["rendering_mode"].as_u64().unwrap() as u32;
    let sims = j["sims"].as_u64().unwrap_or(0) as u32;
    let glyphs: Vec<d2d::text::Glyph> = j["glyphs"].as_array().unwrap().iter().map(|g| d2d::text::Glyph {
        id: g["id"].as_u64().unwrap() as u16,
        advance: g["advance"].as_f64().unwrap() as f32,
        offset: [0.0, 0.0],
    }).collect();
    eprintln!("em={em} base=[{bx},{by}] measuring={measuring} mode={mode} sims={sims} glyphs={}", glyphs.len());
    let run = d2d::text::GlyphRun {
        font_data: &data,
        face_index: 0,
        em_size: em,
        glyphs: &glyphs,
        baseline_origin: [bx, by],
        measuring_mode: measuring,
        is_sideways: false,
        bidi_level: 0,
        no_snap: false,
        simulations: sims,
        coords: &[],
    };
    let dt = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let tex = d2d::text::unhinted_glyph_coverage(&dt, &run, &font, aa, mode, d2d::text::hinting::Rasterizer::DirectWrite, false, None);
    let bpp = if aa == d2d::text::AA_CLEARTYPE { 3 } else { 1 };
    eprintln!("tex bounds=[{},{},{},{}] {}x{} bpp={} bytes={}", tex.left, tex.top, tex.right, tex.bottom, tex.width(), tex.height(), bpp, tex.data.len());
    std::fs::write(out_path, &tex.data).unwrap();
}
