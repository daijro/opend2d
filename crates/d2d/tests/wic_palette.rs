//! WIC median-cut palette conformance: `d2d::wic::quantize::median_cut` (the engine behind
//! `IWICPalette::InitializeFromBitmap`) must match real WIC byte-for-byte. Oracle
//! captured from native WIC on the reference system; the full text
//! (type/flags/colours) is additionally checked end-to-end through the COM ABI by
//! `crates/wic-ffi/decisive.py`. Sources are authored (deterministic gradients / LCG noise).

use std::path::PathBuf;

use d2d::wic::convert::{convert, Format};
use d2d::wic::quantize::median_cut;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    name: String,
    src: String,
    fmt: String,
    width: u32,
    height: u32,
    count: u32,
    addtrans: u32,
}

fn dir() -> PathBuf {
    d2d_reference::suite_dir().unwrap().join("fixtures/wic_palette")
}

/// The `0xAARRGGBB` palette the native capture holds (`colors <hex,hex,...>` line).
fn oracle_colors(txt: &str) -> Vec<u32> {
    let line = txt.lines().find_map(|l| l.strip_prefix("colors ")).unwrap_or("");
    line.split(',').filter(|s| !s.is_empty()).map(|h| u32::from_str_radix(h, 16).unwrap()).collect()
}

#[test]
fn median_cut_matches_wic() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    let d = dir();
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(d.join("cases.json")).unwrap()).unwrap();
    assert!(!cases.is_empty());
    let mut fails = Vec::new();
    for c in &cases {
        let src = std::fs::read(d.join(format!("{}.src.bin", c.src))).unwrap();
        let sf = Format::from_guid(&c.fmt).unwrap();
        let bgr = convert(sf, Format::Bgr24, &src, c.width, c.height, &[]).unwrap();
        let mut got = median_cut(&bgr, (c.count - c.addtrans) as usize);
        if c.addtrans != 0 {
            got.push(0);
        }
        let want = oracle_colors(&std::fs::read_to_string(d.join("oracle").join(format!("{}.txt", c.name))).unwrap());
        if got != want {
            let at = got.iter().zip(&want).position(|(a, b)| a != b);
            fails.push(format!("{}: differ at {at:?} (len {} vs {})", c.name, got.len(), want.len()));
        }
    }
    assert!(fails.is_empty(), "WIC palette mismatches:\n  {}", fails.join("\n  "));
}
