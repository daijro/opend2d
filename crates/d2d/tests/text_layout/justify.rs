//! Justified alignment (`DWRITE_TEXT_ALIGNMENT_JUSTIFIED`, align 3) byte-exact against native
//! DirectWrite. Reuses the `text_layout` query oracle and helpers from the sibling `main.rs`.
//!
//! Justification's outputs are the glyph-run advances *and per-glyph offsets* plus the text/line
//! metrics, cluster widths and carets — all compared bit-for-bit here. DWrite's GetClusterMetrics
//! reports *justified* cluster widths (confirmed against ops 0/3/4 here and text2b-01-align op4);
//! `layout::cluster_metrics` reproduces that by overriding natural widths on justified lines.
//! (`assert_layout` additionally checks glyph ids/clusters/text but not per-glyph offsets, which is
//! why justification has its own asserter.)

use d2d::text::shaping::Font;

/// f32 equal treating +0.0/-0.0 as equal, like `main.rs`.
fn feq(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || a == b
}

/// Compare a justified layout against the native `text_layout` capture: text metrics, line
/// metrics, every glyph run (advances and offsets bit-exact), and leading-edge carets.
fn assert_justified(scene: &str, op: usize, font: &Font) {
    let got = crate::run_layout(scene, op, font);
    let oracle = crate::oracle(scene);
    let want = crate::layout_query(&oracle, op as u64);

    let m = &want["metrics"];
    let gm = &got.metrics;
    assert!(feq(gm.left, m["left"].as_f64().unwrap() as f32), "{scene} op{op}: left {} vs {}", gm.left, m["left"]);
    assert!(feq(gm.width, m["width"].as_f64().unwrap() as f32), "{scene} op{op}: width {} vs {}", gm.width, m["width"]);
    assert!(feq(gm.width_trailing, m["width_trailing"].as_f64().unwrap() as f32), "{scene} op{op}: width_trailing {} vs {}", gm.width_trailing, m["width_trailing"]);
    assert!(feq(gm.height, m["height"].as_f64().unwrap() as f32), "{scene} op{op}: height");
    assert_eq!(gm.line_count, m["line_count"].as_u64().unwrap() as u32, "{scene} op{op}: line_count");

    let wl = want["lines"].as_array().unwrap();
    assert_eq!(got.lines.len(), wl.len(), "{scene} op{op}: line count");
    for (i, (g, w)) in got.lines.iter().zip(wl).enumerate() {
        assert_eq!(g.length, w["length"].as_u64().unwrap() as u32, "{scene} op{op} line{i}: length");
        assert_eq!(g.trailing_ws, w["trailing_ws"].as_u64().unwrap() as u32, "{scene} op{op} line{i}: trailing_ws");
        assert!(feq(g.baseline, w["baseline"].as_f64().unwrap() as f32), "{scene} op{op} line{i}: baseline");
    }

    // Cluster metrics: DWrite reports justified widths (the whole point of this check).
    let wc = want["clusters"].as_array().unwrap();
    assert_eq!(got.clusters.len(), wc.len(), "{scene} op{op}: cluster count");
    for (i, (g, w)) in got.clusters.iter().zip(wc).enumerate() {
        assert!(feq(g.width, w["width"].as_f64().unwrap() as f32), "{scene} op{op} cluster{i}: width {} vs {}", g.width, w["width"]);
    }

    let wr = want["runs"].as_array().unwrap();
    assert_eq!(got.runs.len(), wr.len(), "{scene} op{op}: run count");
    for (i, (r, w)) in got.runs.iter().zip(wr).enumerate() {
        let d2d::text::layout::Run::Glyph(g) = r else { panic!("{scene} op{op} run{i}: unexpected inline object") };
        assert!(feq(g.x, w["x"].as_f64().unwrap() as f32), "{scene} op{op} run{i}: x {} vs {}", g.x, w["x"]);
        assert_eq!(g.glyphs, w["glyphs"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u16).collect::<Vec<_>>(), "{scene} op{op} run{i}: glyphs");
        // Advances and offsets bit-exact: the justification expansion lives here.
        let wa: Vec<u32> = w["advances"].as_array().unwrap().iter().map(|x| (x.as_f64().unwrap() as f32).to_bits()).collect();
        assert_eq!(g.advances.iter().map(|a| a.to_bits()).collect::<Vec<_>>(), wa, "{scene} op{op} run{i}: advances got {:?} want {:?}", g.advances, w["advances"]);
        let wo: Vec<(u32, u32)> = w["offsets"].as_array().unwrap().iter().map(|o| ((o[0].as_f64().unwrap() as f32).to_bits(), (o[1].as_f64().unwrap() as f32).to_bits())).collect();
        assert_eq!(g.offsets.iter().map(|o| (o.0.to_bits(), o.1.to_bits())).collect::<Vec<_>>(), wo, "{scene} op{op} run{i}: offsets got {:?} want {:?}", g.offsets, w["offsets"]);
    }

    for w in want["carets"].as_array().unwrap() {
        let pos = w["pos"].as_u64().unwrap() as u32;
        let c = got.hit_test_text_position(pos);
        assert!(feq(c.x, w["x"].as_f64().unwrap() as f32), "{scene} op{op} caret{pos}: x {} vs {}", c.x, w["x"]);
        assert!(feq(c.width, w["width"].as_f64().unwrap() as f32), "{scene} op{op} caret{pos}: width {} vs {}", c.width, w["width"]);
    }
}

/// `text2b-01-align` op 4: "The quick brown fox jumps over the lazy dog", size 16, wrapped to
/// width 90 into four lines. The three non-last lines are justified (each interior space expands
/// to em/2 and every letter edge takes its share of the residual); the last line stays leading.
#[test]
fn justify_wrapped_latin() {
    let Some(arial) = crate::load("arial.ttf") else {
        eprintln!("SKIP justify_wrapped_latin: arial.ttf not found");
        return;
    };
    assert_justified("text2b-01-align", 4, &arial);
}

/// `justify-01-latin`: the same sentence justified at two widths (ops 0/1 — different per-line
/// deficits), a sentence with punctuation (op 2 — commas/semicolons expand as characters), a
/// single-letter-word line (op 3 — small deficit absorbed entirely by the spaces, so letters
/// stay natural), and a single short line (op 4 — the last line is not justified).
#[test]
fn justify_wide() {
    let Some(arial) = crate::load("arial.ttf") else {
        eprintln!("SKIP justify_wide: arial.ttf not found");
        return;
    };
    assert_justified("justify-01-latin", 0, &arial);
}

#[test]
fn justify_narrow() {
    let Some(arial) = crate::load("arial.ttf") else {
        eprintln!("SKIP justify_narrow: arial.ttf not found");
        return;
    };
    assert_justified("justify-01-latin", 1, &arial);
}

#[test]
fn justify_punctuation() {
    let Some(arial) = crate::load("arial.ttf") else {
        eprintln!("SKIP justify_punctuation: arial.ttf not found");
        return;
    };
    assert_justified("justify-01-latin", 2, &arial);
}

#[test]
fn justify_spaces_absorb_deficit() {
    let Some(arial) = crate::load("arial.ttf") else {
        eprintln!("SKIP justify_spaces_absorb_deficit: arial.ttf not found");
        return;
    };
    assert_justified("justify-01-latin", 3, &arial);
}

#[test]
fn justify_last_line_not_justified() {
    let Some(arial) = crate::load("arial.ttf") else {
        eprintln!("SKIP justify_last_line_not_justified: arial.ttf not found");
        return;
    };
    assert_justified("justify-01-latin", 4, &arial);
}
