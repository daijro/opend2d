//! WIC decode conformance: every corpus image decoded by `d2d::wic` must match real
//! native WIC byte-for-byte — native frame format, size, stride, palette and raw pixels.
//! The oracle was captured from native WIC on the reference system:
//! `cases.json` holds each frame's format/size/stride/palette, `captures/<name>.f0.bin` its pixels.

use std::collections::BTreeMap;
use std::path::PathBuf;

use d2d::wic::decode_png;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    format: String,
    width: u32,
    height: u32,
    bpp: u32,
    stride: usize,
    palette: String,
}

fn dir() -> PathBuf {
    d2d_reference::suite_dir().unwrap().join("fixtures/wic")
}

#[test]
fn png_decode_matches_wic() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    let cases: BTreeMap<String, Case> =
        serde_json::from_slice(&std::fs::read(dir().join("cases.json")).unwrap()).unwrap();
    assert!(!cases.is_empty());

    let mut failures = Vec::new();
    for (name, case) in &cases {
        let input = std::fs::read(dir().join("corpus").join(format!("{name}.png"))).unwrap();
        let expected = std::fs::read(dir().join("captures").join(format!("{name}.f0.bin"))).unwrap();
        let img = match decode_png(&input) {
            Ok(img) => img,
            Err(e) => {
                failures.push(format!("{name}: decode error {e:?}"));
                continue;
            }
        };
        let palette = img
            .palette
            .iter()
            .map(|c| format!("{c:08x}"))
            .collect::<Vec<_>>()
            .join(",");
        let got_fmt = img.format.guid();
        if got_fmt != case.format {
            failures.push(format!("{name}: format {got_fmt} != {}", case.format));
        } else if (img.width, img.height) != (case.width, case.height) {
            failures.push(format!(
                "{name}: size {}x{} != {}x{}",
                img.width, img.height, case.width, case.height
            ));
        } else if img.stride != case.stride || img.format.bits_per_pixel() != case.bpp {
            failures.push(format!(
                "{name}: stride {}/bpp {} != {}/{}",
                img.stride,
                img.format.bits_per_pixel(),
                case.stride,
                case.bpp
            ));
        } else if palette != case.palette {
            failures.push(format!("{name}: palette [{palette}] != [{}]", case.palette));
        } else if img.pixels != expected {
            let at = img
                .pixels
                .iter()
                .zip(&expected)
                .position(|(a, b)| a != b)
                .unwrap_or(img.pixels.len().min(expected.len()));
            failures.push(format!(
                "{name}: pixels differ at byte {at} (len {} vs {})",
                img.pixels.len(),
                expected.len()
            ));
        }
    }
    assert!(failures.is_empty(), "WIC decode mismatches:\n  {}", failures.join("\n  "));
}
