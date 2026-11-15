//! Phase 2D-2 text-layout conformance: drive the Rust shaping/layout engine with the
//! same inputs the harness `text_layout` query op fed native DirectWrite, and compare
//! glyph ids, advances, offsets, line breaks and metrics byte-for-byte against the oracle
//! captured on the reference VM (`conformance/oracle/win11-26100/text2-*.json`).
//!
//! Fonts are the VM's own files (gitignored, shared at `ref/windows/Fonts`); when they
//! are not present the font-dependent cases skip rather than fail.

mod justify;

use d2d::text::fallback;
use d2d::text::layout::{self, FontSpan, TextFormat};
use d2d::text::shaping::{self, Font};
use serde_json::Value;
use std::path::PathBuf;

const ROOT: &str = env!("CARGO_MANIFEST_DIR"); // crates/d2d

/// DWrite family name → (file under `ref/windows/Fonts`, face index) for the fonts the fallback
/// scenes resolve to. Yu Gothic UI Regular is face 1 of `YuGothM`; `YuGothR` contains UI Semilight,
/// which shares its advances and vertical metrics but has thinner outlines.
fn family_file(family: &str) -> Option<(&'static str, u32)> {
    Some(match family {
        "Arial" => ("arial.ttf", 0),
        "Verdana" => ("verdana.ttf", 0),
        "Consolas" => ("consola.ttf", 0),
        "Tahoma" => ("tahoma.ttf", 0),
        "Yu Gothic UI" => ("YuGothM.ttc", 1),
        "Malgun Gothic" => ("malgun.ttf", 0),
        "Nirmala UI" => ("Nirmala.ttc", 0),
        "Leelawadee UI" => ("LeelawUI.ttf", 0),
        "Segoe UI" => ("segoeui.ttf", 0),
        "Segoe UI Emoji" => ("seguiemj.ttf", 0),
        _ => return None,
    })
}

fn load_family(family: &str) -> Option<Font> {
    let (file, idx) = family_file(family)?;
    Font::from_bytes(std::fs::read(fonts_dir()?.join(file)).ok()?, idx)
}

/// Whether a font's cmap has a glyph for `ch` (DirectWrite's fallback coverage test).
fn covers(font: &Font, ch: char) -> bool {
    font.tt().glyph_index(ch).is_some()
}

/// `ref/windows/Fonts` under the *main* checkout (ref is gitignored, absent from worktrees),
/// located via the shared git common dir so this works from any worktree.
fn fonts_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("D2D_FONTS") {
        return Some(PathBuf::from(p));
    }
    let out = std::process::Command::new("git")
        .args(["-C", ROOT, "rev-parse", "--path-format=absolute", "--git-common-dir"])
        .output()
        .ok()?;
    let gitdir = PathBuf::from(String::from_utf8(out.stdout).ok()?.trim());
    let dir = gitdir.parent()?.join("ref/windows/Fonts");
    dir.is_dir().then_some(dir)
}

fn load(name: &str) -> Option<Font> {
    d2d_reference::suite_dir()?; // the layout scenes/oracles live in the suite; skip the test when it is absent
    let path = fonts_dir()?.join(name);
    Font::from_bytes(std::fs::read(path).ok()?, 0)
}

fn oracle(scene: &str) -> Value {
    let p = d2d_reference::suite_dir().unwrap().join("oracle/win11-26100").join(format!("{scene}.json"));
    serde_json::from_reader(std::fs::File::open(p).unwrap()).unwrap()
}

/// The `result` of the N-th `text_layout` query op in a captured scene.
fn layout_query(scene: &Value, op: u64) -> &Value {
    scene["queries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|q| q["op"].as_u64() == Some(op) && q["kind"] == "text_layout")
        .map(|q| &q["result"])
        .unwrap()
}

fn f32s(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}
fn u16s(v: &Value) -> Vec<u16> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u16).collect()
}

/// Scale a design-unit advance to DIPs the way DirectWrite natural measuring does.
fn to_dip(units: i32, em: f32, upm: u16) -> f32 {
    units as f32 * em / upm as f32
}

/// The input params of the N-th op in a committed scene file (`harness/scenes`).
fn scene_op(scene: &str, op: usize) -> Value {
    let p = d2d_reference::suite_dir().unwrap().join("scenes").join(format!("{scene}.json"));
    let s: Value = serde_json::from_reader(std::fs::File::open(p).unwrap()).unwrap();
    s["ops"][op].clone()
}

/// Equal as f32, treating +0.0 and -0.0 as equal (DirectWrite reports a signed-zero left edge).
fn feq(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || a == b
}

fn run_layout(scene: &str, op: usize, font: &Font) -> layout::Layout {
    let o = scene_op(scene, op);
    let fmt = TextFormat {
        em: o["size"].as_f64().unwrap() as f32,
        base_rtl: o["reading"].as_u64().unwrap_or(0) == 1,
        align: o["align"].as_u64().unwrap_or(0) as u8,
        wrap: o["wrap"].as_u64().unwrap_or(0) as u8,
        para: o["para"].as_u64().unwrap_or(0) as u8,
        flow: o["flow"].as_u64().unwrap_or(0) as u8,
        line_spacing: (0, 0.0, 0.0),
    };
    let trimming = o["trimming"].as_object().map(|t| layout::Trimming {
        granularity: t["granularity"].as_u64().unwrap() as u8,
        delimiter: t.get("delimiter").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        delimiter_count: t.get("delimiter_count").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        ellipsis: t.get("ellipsis").and_then(|v| v.as_bool()).unwrap_or(false),
        upright_sign: 0.0,
        sign: None,
    });
    let feats: Vec<layout::Feature> = o["features"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|f| {
            let pos = f["position"].as_u64().unwrap() as u32;
            let t = f["tag"].as_str().unwrap().as_bytes();
            layout::Feature {
                start: pos,
                end: pos + f["length"].as_u64().unwrap() as u32,
                tag: [t[0], t[1], t[2], t[3]],
                value: f["value"].as_u64().unwrap() as u32,
            }
        })
        .collect();
    let inline: Vec<layout::InlineObject> = o["inline_objects"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|io| {
            let pos = io["position"].as_u64().unwrap() as u32;
            layout::InlineObject {
                start: pos,
                end: pos + io["length"].as_u64().unwrap() as u32,
                width: io["size"][0].as_f64().unwrap() as f32,
                height: io["size"][1].as_f64().unwrap() as f32,
                baseline: io["baseline"].as_f64().unwrap() as f32,
                sideways: io.get("supports_sideways").and_then(|v| v.as_bool()).unwrap_or(false),
            }
        })
        .collect();
    // Resolve system font fallback: load every candidate family on disk once, map the text to
    // per-run families (base family = the scene's `font`), then build the font spans layout needs.
    let text = o["text"].as_str().unwrap();
    let locale = o["locale"].as_str().unwrap_or("en-US");
    let mut cache: std::collections::HashMap<&'static str, Font> = std::collections::HashMap::new();
    for ch in text.chars() {
        for fam in fallback::candidates(ch as u32, locale) {
            if !cache.contains_key(fam) {
                if let Some(f) = load_family(fam) {
                    cache.insert(fam, f);
                }
            }
        }
    }
    let spans = fallback::map_characters(text, locale, |ch| covers(font, ch), |fam, ch| cache.get(fam).is_some_and(|f| covers(f, ch)));
    let font_spans: Vec<FontSpan> = spans
        .iter()
        .filter_map(|s| s.family.and_then(|f| cache.get(f)).map(|f| FontSpan { start: s.start, end: s.end, font: f, em: None, bold: false }))
        .collect();

    layout::layout_ex(
        font,
        &fmt,
        text,
        o["max_width"].as_f64().unwrap() as f32,
        o["max_height"].as_f64().unwrap() as f32,
        &layout::LayoutOpts { features: &feats, trimming, inline: &inline, fonts: &font_spans, ..Default::default() },
    )
}

/// Assert a computed layout equals the native capture for op `op` of a scene.
fn assert_layout(scene: &str, op: usize, font: &Font) {
    let got = run_layout(scene, op, font);
    let oracle = oracle(scene);
    let want = layout_query(&oracle, op as u64);

    // Text metrics.
    let m = &want["metrics"];
    let gm = &got.metrics;
    assert!(feq(gm.left, m["left"].as_f64().unwrap() as f32), "{scene} op{op}: left {} vs {}", gm.left, m["left"]);
    assert!(feq(gm.width, m["width"].as_f64().unwrap() as f32), "{scene} op{op}: width {} vs {}", gm.width, m["width"]);
    assert!(feq(gm.width_trailing, m["width_trailing"].as_f64().unwrap() as f32), "{scene} op{op}: width_trailing");
    assert!(feq(gm.height, m["height"].as_f64().unwrap() as f32), "{scene} op{op}: height {} vs {}", gm.height, m["height"]);
    assert!(feq(gm.layout_width, m["layout_width"].as_f64().unwrap() as f32), "{scene} op{op}: layout_width");
    assert!(feq(gm.layout_height, m["layout_height"].as_f64().unwrap() as f32), "{scene} op{op}: layout_height");
    assert_eq!(gm.bidi_depth, m["bidi_depth"].as_u64().unwrap() as u32, "{scene} op{op}: bidi_depth");
    assert_eq!(gm.line_count, m["line_count"].as_u64().unwrap() as u32, "{scene} op{op}: line_count");

    // Line metrics.
    let wl = want["lines"].as_array().unwrap();
    assert_eq!(got.lines.len(), wl.len(), "{scene} op{op}: line count");
    for (i, (g, w)) in got.lines.iter().zip(wl).enumerate() {
        assert_eq!(g.length, w["length"].as_u64().unwrap() as u32, "{scene} op{op} line{i}: length");
        assert_eq!(g.trailing_ws, w["trailing_ws"].as_u64().unwrap() as u32, "{scene} op{op} line{i}: trailing_ws");
        assert_eq!(g.newline, w["newline"].as_u64().unwrap() as u32, "{scene} op{op} line{i}: newline");
        assert!(feq(g.height, w["height"].as_f64().unwrap() as f32), "{scene} op{op} line{i}: height");
        assert!(feq(g.baseline, w["baseline"].as_f64().unwrap() as f32), "{scene} op{op} line{i}: baseline");
        assert_eq!(g.is_trimmed, w["trimmed"].as_bool().unwrap(), "{scene} op{op} line{i}: trimmed");
    }

    // Cluster metrics.
    let wc = want["clusters"].as_array().unwrap();
    assert_eq!(got.clusters.len(), wc.len(), "{scene} op{op}: cluster count");
    for (i, (g, w)) in got.clusters.iter().zip(wc).enumerate() {
        assert!(feq(g.width, w["width"].as_f64().unwrap() as f32), "{scene} op{op} cluster{i}: width {} vs {}", g.width, w["width"]);
        assert_eq!(g.length, w["length"].as_u64().unwrap() as u16, "{scene} op{op} cluster{i}: length");
        assert_eq!(g.bits, w["bits"].as_u64().unwrap() as u16, "{scene} op{op} cluster{i}: bits {} vs {}", g.bits, w["bits"]);
    }

    // Draw-order runs: glyph runs and inline objects interleaved.
    let wr = want["runs"].as_array().unwrap();
    assert_eq!(got.runs.len(), wr.len(), "{scene} op{op}: run count");
    for (i, (run, w)) in got.runs.iter().zip(wr).enumerate() {
        match run {
            layout::Run::Glyph(g) => {
                assert!(w["inline"].is_null(), "{scene} op{op} run{i}: expected inline, got glyph");
                assert!(feq(g.x, w["x"].as_f64().unwrap() as f32), "{scene} op{op} run{i}: x {} vs {}", g.x, w["x"]);
                assert!(feq(g.y, w["y"].as_f64().unwrap() as f32), "{scene} op{op} run{i}: y {} vs {}", g.y, w["y"]);
                assert_eq!(g.level as u64, w["bidi"].as_u64().unwrap(), "{scene} op{op} run{i}: level");
                assert_eq!(g.glyphs, u16s(&w["glyphs"]), "{scene} op{op} run{i}: glyphs");
                assert_eq!(
                    g.advances.iter().map(|a| a.to_bits()).collect::<Vec<_>>(),
                    f32s(&w["advances"]).iter().map(|a| a.to_bits()).collect::<Vec<_>>(),
                    "{scene} op{op} run{i}: advances",
                );
                assert_eq!(g.clusters, u16s(&w["clusters"]), "{scene} op{op} run{i}: clusters");
                assert_eq!(g.text_pos as u64, w["text_pos"].as_u64().unwrap(), "{scene} op{op} run{i}: text_pos");
                assert_eq!(g.text, w["text"].as_str().unwrap(), "{scene} op{op} run{i}: text");
            }
            layout::Run::Inline(o) => {
                let w = &w["inline"];
                assert!(!w.is_null(), "{scene} op{op} run{i}: expected glyph, got inline");
                let f = |k: &str| w[k].as_f64().unwrap() as f32;
                assert!(feq(o.x, f("x")), "{scene} op{op} run{i}: inline x {} vs {}", o.x, w["x"]);
                assert!(feq(o.y, f("y")), "{scene} op{op} run{i}: inline y {} vs {}", o.y, w["y"]);
                assert!(feq(o.width, f("width")), "{scene} op{op} run{i}: inline width {} vs {}", o.width, w["width"]);
                assert!(feq(o.height, f("height")), "{scene} op{op} run{i}: inline height {} vs {}", o.height, w["height"]);
                assert!(feq(o.baseline, f("baseline")), "{scene} op{op} run{i}: inline baseline {} vs {}", o.baseline, w["baseline"]);
                assert_eq!(o.rtl, w["rtl"].as_bool().unwrap(), "{scene} op{op} run{i}: inline rtl");
                assert_eq!(o.sideways, w["sideways"].as_bool().unwrap(), "{scene} op{op} run{i}: inline sideways");
            }
        }
    }

    // Caret hit testing (leading edge of every text position).
    for w in want["carets"].as_array().unwrap() {
        let pos = w["pos"].as_u64().unwrap() as u32;
        let c = got.hit_test_text_position(pos);
        let f = |k: &str| w[k].as_f64().unwrap() as f32;
        assert!(feq(c.x, f("x")), "{scene} op{op} caret{pos}: x {} vs {}", c.x, w["x"]);
        assert!(feq(c.y, f("y")), "{scene} op{op} caret{pos}: y {} vs {}", c.y, w["y"]);
        assert!(feq(c.left, f("left")), "{scene} op{op} caret{pos}: left {} vs {}", c.left, w["left"]);
        assert!(feq(c.top, f("top")), "{scene} op{op} caret{pos}: top");
        assert!(feq(c.width, f("width")), "{scene} op{op} caret{pos}: width {} vs {}", c.width, w["width"]);
        assert!(feq(c.height, f("height")), "{scene} op{op} caret{pos}: height");
        assert_eq!(c.bidi as u64, w["bidi"].as_u64().unwrap(), "{scene} op{op} caret{pos}: bidi");
    }
}

#[test]
fn layout_single_line_latin() {
    let Some(arial) = load("arial.ttf") else {
        eprintln!("SKIP layout_single_line_latin: arial.ttf not found");
        return;
    };
    assert_layout("text2-01-layout", 0, &arial);
}

#[test]
fn layout_bidi_two_paragraphs() {
    let Some(arial) = load("arial.ttf") else {
        eprintln!("SKIP layout_bidi_two_paragraphs: arial.ttf not found");
        return;
    };
    assert_layout("text2-01-layout", 2, &arial);
}

#[test]
fn layout_word_wrap() {
    let Some(arial) = load("arial.ttf") else {
        eprintln!("SKIP layout_word_wrap: arial.ttf not found");
        return;
    };
    // op 1: 43-char sentence wrapped to maxWidth 90 → 4 lines, each wrapped line's trailing
    // space its own glyph run excluded from the line width.
    assert_layout("text2-01-layout", 1, &arial);
}

/// Alignment (`AlignHorizontally`): centre/trailing shift each line by a per-line offset. The
/// text2b-01-align scene also captures a justified op (4); justification is not implemented yet
/// (see layout/mod.rs), so it is intentionally not asserted here.
#[test]
fn layout_align_center_single() {
    let Some(arial) = load("arial.ttf") else {
        eprintln!("SKIP layout_align_center_single: arial.ttf not found");
        return;
    };
    assert_layout("text2b-01-align", 0, &arial); // centre, single line
}

#[test]
fn layout_align_trailing_single() {
    let Some(arial) = load("arial.ttf") else {
        eprintln!("SKIP layout_align_trailing_single: arial.ttf not found");
        return;
    };
    assert_layout("text2b-01-align", 1, &arial); // trailing, single line
}

#[test]
fn layout_align_center_wrapped() {
    let Some(arial) = load("arial.ttf") else {
        eprintln!("SKIP layout_align_center_wrapped: arial.ttf not found");
        return;
    };
    assert_layout("text2b-01-align", 2, &arial); // centre, 4 wrapped lines
}

#[test]
fn layout_align_trailing_wrapped() {
    let Some(arial) = load("arial.ttf") else {
        eprintln!("SKIP layout_align_trailing_wrapped: arial.ttf not found");
        return;
    };
    assert_layout("text2b-01-align", 3, &arial); // trailing, 4 wrapped lines
}

/// Compare one computed `HitMetrics` box with an oracle `hit_points`/`hit_ranges` entry.
fn assert_hit(scene: &str, op: usize, tag: &str, g: &layout::HitMetrics, w: &Value, point: bool) {
    let f = |k: &str| w[k].as_f64().unwrap() as f32;
    assert_eq!(g.pos as u64, w["pos"].as_u64().unwrap(), "{scene} op{op} {tag}: pos");
    assert_eq!(g.length as u64, w["length"].as_u64().unwrap(), "{scene} op{op} {tag}: length");
    assert!(feq(g.left, f("left")), "{scene} op{op} {tag}: left {} vs {}", g.left, w["left"]);
    assert!(feq(g.top, f("top")), "{scene} op{op} {tag}: top {} vs {}", g.top, w["top"]);
    assert!(feq(g.width, f("width")), "{scene} op{op} {tag}: width {} vs {}", g.width, w["width"]);
    assert!(feq(g.height, f("height")), "{scene} op{op} {tag}: height {} vs {}", g.height, w["height"]);
    assert_eq!(g.bidi as u64, w["bidi"].as_u64().unwrap(), "{scene} op{op} {tag}: bidi");
    assert_eq!(g.is_text, w["is_text"].as_bool().unwrap(), "{scene} op{op} {tag}: is_text");
    assert_eq!(g.is_trimmed, w["is_trimmed"].as_bool().unwrap(), "{scene} op{op} {tag}: is_trimmed");
    if point {
        assert_eq!(g.is_trailing, w["is_trailing"].as_bool().unwrap(), "{scene} op{op} {tag}: is_trailing");
        assert_eq!(g.is_inside, w["is_inside"].as_bool().unwrap(), "{scene} op{op} {tag}: is_inside");
    }
}

/// HitTestPoint and HitTestTextRange against the native capture for one op.
fn assert_hit_tests(scene: &str, op: usize, font: &Font) {
    let got = run_layout(scene, op, font);
    let oracle = oracle(scene);
    let want = layout_query(&oracle, op as u64);
    let o = scene_op(scene, op);
    for (i, p) in o["hit_test_points"].as_array().into_iter().flatten().enumerate() {
        let (x, y) = (p[0].as_f64().unwrap() as f32, p[1].as_f64().unwrap() as f32);
        assert_hit(scene, op, &format!("pt{i}"), &got.hit_test_point(x, y), &want["hit_points"][i], true);
    }
    for (i, rg) in o["hit_test_ranges"].as_array().into_iter().flatten().enumerate() {
        let (pos, len) = (rg[0].as_u64().unwrap() as u32, rg[1].as_u64().unwrap() as u32);
        let got_boxes = got.hit_test_text_range(pos, len);
        let want_boxes = want["hit_ranges"][i].as_array().unwrap();
        assert_eq!(got_boxes.len(), want_boxes.len(), "{scene} op{op} range{i}: box count");
        for (j, (g, w)) in got_boxes.iter().zip(want_boxes).enumerate() {
            assert_hit(scene, op, &format!("range{i}.{j}"), g, w, false);
        }
    }
}

#[test]
fn layout_hit_test_wrapped() {
    let Some(arial) = load("arial.ttf") else {
        eprintln!("SKIP layout_hit_test_wrapped: arial.ttf not found");
        return;
    };
    assert_layout("text2c-01-hittest", 0, &arial);
    assert_hit_tests("text2c-01-hittest", 0, &arial);
}

#[test]
fn layout_hit_test_bidi() {
    let Some(arial) = load("arial.ttf") else {
        eprintln!("SKIP layout_hit_test_bidi: arial.ttf not found");
        return;
    };
    assert_layout("text2c-01-hittest", 1, &arial);
    assert_hit_tests("text2c-01-hittest", 1, &arial);
}

/// Inline objects: a fixed-size object replaces its range as one cluster, draws via DrawInlineObject
/// at the computed origin, and raises the line's ascent/descent only when taller than the font.
#[test]
fn layout_inline_objects() {
    let Some(arial) = load("arial.ttf") else {
        eprintln!("SKIP layout_inline_objects: arial.ttf not found");
        return;
    };
    assert_layout("text2c-03-inline", 0, &arial); // object mid-line, shorter than font
    assert_layout("text2c-03-inline", 1, &arial); // object taller than font: line grows
    assert_layout("text2c-03-inline", 2, &arial); // object between words, wraps
}

/// Trimming a single no-wrap line: character/word granularity, with and without an ellipsis sign,
/// and path (delimiter) trimming with the sign at the front.
#[test]
fn layout_trimming() {
    let Some(arial) = load("arial.ttf") else {
        eprintln!("SKIP layout_trimming: arial.ttf not found");
        return;
    };
    assert_layout("text2c-02-trimming", 0, &arial); // character, no sign
    assert_layout("text2c-02-trimming", 1, &arial); // character + ellipsis
    assert_layout("text2c-02-trimming", 2, &arial); // word + ellipsis
    assert_layout("text2c-02-trimming", 3, &arial); // path (delimiter '/') + ellipsis
}

/// Typography feature ranges reach the shaper: `kern=0` removes the AV/VA kern, and a feature range
/// that covers only part of the text splits the glyph run exactly where DirectWrite splits it.
#[test]
fn layout_typography_kern() {
    let Some(arial) = load("arial.ttf") else {
        eprintln!("SKIP layout_typography_kern: arial.ttf not found");
        return;
    };
    assert_layout("text2c-04-typography", 0, &arial); // default (kern on)
    assert_layout("text2c-04-typography", 1, &arial); // kern off over the whole range
    assert_layout("text2c-04-typography", 2, &arial); // kern off over [0,2): run splits, both unkerned
    assert_layout("text2c-04-typography", 3, &arial); // liga off over [0,2): any typography drops default kern
    assert_layout("text2c-04-typography", 4, &arial); // kern off [0,2) + kern on [2,5): explicit re-enable
}

/// System font fallback: "Hi 日本語" at Arial/en-US splits into an Arial run ("Hi ") and a
/// Yu Gothic UI run ("日本語"), with mixed-font line metrics (the CJK run is taller) and per-run
/// caret heights — all byte-exact vs the native capture, which also records the resolved family.
#[test]
fn layout_fallback_japanese() {
    let Some(arial) = load("arial.ttf") else {
        eprintln!("SKIP layout_fallback_japanese: arial.ttf not found");
        return;
    };
    if load_family("Yu Gothic UI").is_none() {
        eprintln!("SKIP layout_fallback_japanese: Yu Gothic UI (YuGothM.ttc) not found");
        return;
    }
    assert_layout("text2c-05-fallback", 0, &arial);
    assert_resolved_families("text2c-05-fallback", 0, &arial);
}

/// Drive every op of a fallback scene: assert the full layout (runs/glyphs/advances/metrics/
/// clusters/carets) and the resolved family per run, loading each op's base family from the scene.
/// Skips cleanly when the oracle capture or a required font is absent.
fn run_fallback_scene(scene: &str, ops: &[usize]) {
    let Some(suite) = d2d_reference::suite_dir() else { return };
    let oracle_json = suite.join("oracle/win11-26100").join(format!("{scene}.json"));
    if !oracle_json.exists() {
        eprintln!("SKIP {scene}: oracle capture not present");
        return;
    }
    for &op in ops {
        let base_family = scene_op(scene, op)["font"].as_str().unwrap().to_string();
        let Some(base) = load_family(&base_family) else {
            eprintln!("SKIP {scene} op{op}: base family {base_family:?} not found");
            continue;
        };
        assert_layout(scene, op, &base);
        assert_resolved_families(scene, op, &base);
    }
}

/// Korean (Hangul), Devanagari, Thai and emoji all fall back from Arial to their system faces.
#[test]
fn layout_fallback_scripts() {
    run_fallback_scene("fallback-01-scripts", &[0, 1, 2, 3]);
}

/// Arabic and Hebrew fall back from Verdana (which lacks them) to Segoe UI, with RTL shaping.
#[test]
fn layout_fallback_bidi() {
    run_fallback_scene("fallback-02-bidi", &[0, 1]);
}

/// The same Han text resolves to Yu Gothic UI under en-US but Malgun Gothic under ko-KR — the
/// locale effect on the fallback candidate order.
#[test]
fn layout_fallback_locale() {
    run_fallback_scene("fallback-03-locale", &[0, 1]);
}

/// Check the family each run resolved to matches the oracle's recorded `font` per run.
fn assert_resolved_families(scene: &str, op: usize, font: &Font) {
    let o = scene_op(scene, op);
    let text = o["text"].as_str().unwrap();
    let base_family = o["font"].as_str().unwrap();
    let locale = o["locale"].as_str().unwrap_or("en-US");
    let mut cache: std::collections::HashMap<&'static str, Font> = std::collections::HashMap::new();
    for ch in text.chars() {
        for fam in fallback::candidates(ch as u32, locale) {
            if !cache.contains_key(fam) {
                if let Some(f) = load_family(fam) {
                    cache.insert(fam, f);
                }
            }
        }
    }
    let spans = fallback::map_characters(text, locale, |ch| covers(font, ch), |fam, ch| cache.get(fam).is_some_and(|f| covers(f, ch)));
    // Expected family per run from the oracle, in run order.
    let want: Vec<String> = oracle(scene)["queries"]
        .as_array().unwrap().iter().find(|q| q["op"].as_u64() == Some(op as u64)).unwrap()["result"]["runs"]
        .as_array().unwrap().iter().filter(|r| r["glyphs"].is_array()).map(|r| r["font"].as_str().unwrap().to_string()).collect();
    let got: Vec<String> = spans.iter().map(|s| s.family.unwrap_or(base_family).to_string()).collect();
    assert_eq!(got, want, "{scene} op{op}: resolved families");
}

#[test]
fn shaping_matches_native_latin() {
    let Some(arial) = load("arial.ttf") else {
        eprintln!("SKIP shaping_matches_native_latin: arial.ttf not found");
        return;
    };
    let scene = oracle("text2-01-layout");

    // op 0: "AV ffi Åé", natural. One LTR Latin run; pure shaping (glyph ids + advances,
    // including the AV kern pair and the absence of an f-ligature in Arial).
    let run = &layout_query(&scene, 0)["runs"][0];
    let text = run["text"].as_str().unwrap();
    let em = run["em"].as_f64().unwrap() as f32;
    let want_ids = u16s(&run["glyphs"]);
    let want_adv = f32s(&run["advances"]);

    let glyphs = shaping::shape(&arial, text, false, Some(rustybuzz::script::LATIN), None);
    let got_ids: Vec<u16> = glyphs.iter().map(|g| g.id).collect();
    let got_adv: Vec<f32> = glyphs.iter().map(|g| to_dip(g.advance, em, arial.upm)).collect();

    assert_eq!(got_ids, want_ids, "glyph ids differ");
    assert_eq!(
        got_adv.iter().map(|a| a.to_bits()).collect::<Vec<_>>(),
        want_adv.iter().map(|a| a.to_bits()).collect::<Vec<_>>(),
        "advances differ: got {got_adv:?} want {want_adv:?}"
    );
}

#[test]
fn shaping_matches_native_arabic() {
    let Some(arial) = load("arial.ttf") else {
        eprintln!("SKIP shaping_matches_native_arabic: arial.ttf not found");
        return;
    };
    let scene = oracle("text2-01-layout");

    // op 2, run 2: the Arabic "مرحبا" (RTL, bidi level 1) — joining forms via GSUB and
    // RTL glyph order. Tests the Arabic shaper against native TextShaping.
    let run = &layout_query(&scene, 2)["runs"][2];
    let text = run["text"].as_str().unwrap();
    let em = run["em"].as_f64().unwrap() as f32;
    let want_ids = u16s(&run["glyphs"]);
    let want_adv = f32s(&run["advances"]);

    let glyphs = shaping::shape(&arial, text, true, Some(rustybuzz::script::ARABIC), None);
    let got_ids: Vec<u16> = glyphs.iter().map(|g| g.id).collect();
    let got_adv: Vec<f32> = glyphs.iter().map(|g| to_dip(g.advance, em, arial.upm)).collect();

    assert_eq!(got_ids, want_ids, "arabic glyph ids differ");
    assert_eq!(
        got_adv.iter().map(|a| a.to_bits()).collect::<Vec<_>>(),
        want_adv.iter().map(|a| a.to_bits()).collect::<Vec<_>>(),
        "arabic advances differ: got {got_adv:?} want {want_adv:?}"
    );
}

// Native DWriteTextLayout::Draw callbacks, Windows 11 26100, captured 2026-11-04.
#[test]
fn gdi_natural_rounds_positioned_advances_to_device_pixels() {
    let Some(font) = load_family("Arial") else { return };
    for (text, em, expected) in [
        ("AV ffi Åé", 21.0, vec![12.0, 14.0, 6.0, 6.0, 6.0, 5.0, 6.0, 14.0, 12.0]),
        ("Hamburgefons 1234", 13.0, vec![9.0, 7.0, 11.0, 8.0, 7.0, 4.0, 8.0, 7.0, 5.0, 8.0, 7.0, 7.0, 4.0, 7.0, 7.0, 7.0, 7.0]),
    ] {
        let format = TextFormat { em, base_rtl: false, align: 0, wrap: 0, para: 0, flow: 0, line_spacing: (0, 0.0, 0.0) };
        let opts = layout::LayoutOpts { measuring: 2, pixels_per_dip: 1.0, ..Default::default() };
        let output = layout::layout_ex(&font, &format, text, 1000.0, 1000.0, &opts);
        let advances: Vec<f32> = output.runs.iter().filter_map(|r| r.glyph()).flat_map(|r| r.advances.iter().copied()).collect();
        assert_eq!(advances, expected, "{text}");
    }
}

// Empty text and trailing-newline layouts. DirectWrite emits one empty line for an empty string and a
// trailing empty line after a hard break, and the end-of-text caret sits on that empty line. Byte-exact
// vs native (Segoe UI 7.767pt, centre, GDI_NATURAL, box 74.37; Windows 11 26100, captured 2026-11-07).
// Regression for the HitTestTextPosition-on-empty-text panic (differential-fuzz seeds 15017, 15028,
// 15065, 15074, 15084) and for GDI-compatible layouts measuring the natural line box (10.33) instead of
// the VDMX box (11.0).
#[test]
fn layout_empty_and_trailing_newline() {
    let Some(font) = load_family("Segoe UI") else { return };
    let fmt = TextFormat { em: 7.767032, base_rtl: false, align: 2, wrap: 0, para: 0, flow: 0, line_spacing: (0, 0.0, 0.0) };
    let opts = layout::LayoutOpts { measuring: 2, pixels_per_dip: 1.0, ..Default::default() };
    let lay = |text: &str| layout::layout_ex(&font, &fmt, text, 74.37126, 24.0, &opts);

    // Empty string: exactly one empty line with the GDI (VDMX) box 11/9 — not natural 10.33 — and a
    // centred end caret at 37, height 11. No panic (this was the crash).
    let o = lay("");
    assert_eq!(o.metrics.line_count, 1);
    assert_eq!((o.lines[0].height, o.lines[0].baseline, o.lines[0].length), (11.0, 9.0, 0));
    assert_eq!((o.metrics.left, o.metrics.height), (37.0, 11.0));
    let c = o.hit_test_text_position(0);
    assert_eq!((c.x, c.height), (37.0, 11.0));
    assert_eq!(o.hit_test_text_position(99).x, 37.0); // clamped past-the-end still resolves

    // Trailing newline "x\n": two lines (the "x" line + a trailing empty line); the end caret (pos 2)
    // is on the empty line, centred at 37 — not on the "x" line's newline run (y would differ).
    let o = lay("x\n");
    assert_eq!(o.metrics.line_count, 2);
    assert_eq!((o.lines[1].height, o.lines[1].length, o.lines[1].newline), (11.0, 0, 0));
    assert_eq!((o.hit_test_text_position(2).x, o.hit_test_text_position(2).height), (37.0, 11.0));

    // Bare "\n" is two empty lines; "x\n\ny" is three (x, empty, y).
    assert_eq!(lay("\n").metrics.line_count, 2);
    assert_eq!(lay("x\n\ny").metrics.line_count, 3);

    // An internal newline with no trailing break stays two lines, both the GDI box 11 (the measuring
    // fix: GDI-compatible layouts used to fall through to the natural 10.33 box).
    let o = lay("ab\ncd");
    assert_eq!(o.metrics.line_count, 2);
    assert!(o.lines.iter().all(|l| l.height == 11.0));
    assert_eq!(o.metrics.left, 32.0);
}
