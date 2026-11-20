//! WIC JPEG decode conformance: every corpus JPEG decoded by `d2d::wic::decode_jpeg` must match
//! real WIC byte-for-byte — native frame format, size, stride and raw pixels.
//! Oracle captured from native WIC on the reference system (see `capture.py`);
//! `cases.json` holds each frame's format/size/stride, `captures/<name>.f0.bin` its pixels.
//! The corpus is produced from a synthetic pattern by `gen_corpus.py` (cjpeg / libjpeg-turbo).

use std::collections::BTreeMap;
use std::path::PathBuf;

use d2d::wic::{decode_jpeg, decode_jpeg_scaled, encode_jpeg, jpeg_closest_scale, Subsampling, WicFormat};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    format: String,
    width: u32,
    height: u32,
    bpp: u32,
    stride: usize,
}

fn dir() -> PathBuf {
    d2d_reference::suite_dir().unwrap().join("fixtures/wic_jpeg")
}

#[test]
fn jpeg_decode_matches_wic() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    let cases: BTreeMap<String, Case> =
        serde_json::from_slice(&std::fs::read(dir().join("cases.json")).unwrap()).unwrap();
    assert!(!cases.is_empty());

    let mut failures = Vec::new();
    for (name, case) in &cases {
        let input = std::fs::read(dir().join("corpus").join(format!("{name}.jpg"))).unwrap();
        let expected = std::fs::read(dir().join("captures").join(format!("{name}.f0.bin"))).unwrap();
        let img = match decode_jpeg(&input) {
            Ok(img) => img,
            Err(e) => {
                failures.push(format!("{name}: decode error {e:?}"));
                continue;
            }
        };
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
        } else if img.pixels != expected {
            let at = img
                .pixels
                .iter()
                .zip(&expected)
                .position(|(a, b)| a != b)
                .unwrap_or(img.pixels.len().min(expected.len()));
            failures.push(format!(
                "{name}: pixels differ at byte {at} ({} vs {}), len {} vs {}",
                img.pixels.get(at).copied().unwrap_or(0),
                expected.get(at).copied().unwrap_or(0),
                img.pixels.len(),
                expected.len()
            ));
        }
    }
    assert!(failures.is_empty(), "WIC JPEG decode mismatches:\n  {}", failures.join("\n  "));
}

/// Native reduced-size decode (`IWICBitmapSourceTransform`) at 1/2, 1/4 and 1/8 must match real WIC
/// byte-for-byte — format, size, stride and pixels. Oracle captured by `scale_capture.py`.
#[test]
fn jpeg_scaled_decode_matches_wic() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    #[derive(Deserialize)]
    struct S {
        format: String,
        width: u32,
        height: u32,
        bpp: u32,
        stride: usize,
    }
    let cases: BTreeMap<String, BTreeMap<String, S>> =
        serde_json::from_slice(&std::fs::read(dir().join("scaled.json")).unwrap()).unwrap();
    assert!(!cases.is_empty());

    let mut failures = Vec::new();
    for (name, denoms) in &cases {
        let input = std::fs::read(dir().join("corpus").join(format!("{name}.jpg"))).unwrap();
        for (d, case) in denoms {
            let denom: usize = d.parse().unwrap();
            let want =
                std::fs::read(dir().join("captures/scaled").join(format!("{name}.{d}.bin"))).unwrap();
            let img = match decode_jpeg_scaled(&input, denom) {
                Ok(img) => img,
                Err(e) => {
                    failures.push(format!("{name} 1/{d}: decode error 0x{e:08x}"));
                    continue;
                }
            };
            if img.format.guid() != case.format {
                failures.push(format!("{name} 1/{d}: format {} != {}", img.format.guid(), case.format));
            } else if (img.width, img.height) != (case.width, case.height) {
                failures.push(format!(
                    "{name} 1/{d}: {}x{} != {}x{}",
                    img.width, img.height, case.width, case.height
                ));
            } else if img.stride != case.stride || img.format.bits_per_pixel() != case.bpp {
                failures.push(format!("{name} 1/{d}: stride/bpp mismatch"));
            } else if img.pixels != want {
                let at = img.pixels.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(0);
                failures.push(format!(
                    "{name} 1/{d}: pixels differ at byte {at} ({} vs {}), len {} vs {}",
                    img.pixels.get(at).copied().unwrap_or(0),
                    want.get(at).copied().unwrap_or(0),
                    img.pixels.len(),
                    want.len()
                ));
            }
        }
    }
    assert!(failures.is_empty(), "WIC JPEG scaled-decode mismatches:\n  {}", failures.join("\n  "));
}

/// `jpeg_closest_scale` reproduces WIC's `GetClosestScaleDenominator` + output-size snapping (read
/// from native WIC): pick the largest denominator in {1,2,4,8} whose ⌈dim/denom⌉ covers the
/// request, output = ⌈dim/denom⌉.
#[test]
fn jpeg_closest_scale_selection() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    let c = jpeg_closest_scale;
    assert_eq!(c(24, 16, 24, 16), (1, (24, 16)));
    assert_eq!(c(24, 16, 12, 8), (2, (12, 8)));
    assert_eq!(c(24, 16, 6, 4), (4, (6, 4)));
    assert_eq!(c(24, 16, 3, 2), (8, (3, 2)));
    // Between scales snaps *down* in denominator (toward the larger output that still covers it).
    assert_eq!(c(24, 16, 13, 9), (1, (24, 16)));
    assert_eq!(c(24, 16, 11, 7), (2, (12, 8)));
    assert_eq!(c(24, 16, 5, 3), (4, (6, 4)));
    // A request of 1 on an axis forces maximum downscale on it (denom = 8).
    assert_eq!(c(24, 16, 1, 1), (8, (3, 2)));
}

/// `encode_jpeg` must reproduce WIC's `IWICBitmapEncoder` default output (quality 90, 4:2:0 for
/// colour, baseline, JFIF) byte-for-byte. Oracle `encode/<name>.jpg` captured by `encode_capture.py`
/// (real WIC) from the raw inputs in `encode/inputs/`. Non-default quality and
/// subsampling are verified against libjpeg-turbo's cjpeg (WIC's own encoder) in gen_encode notes.
#[test]
fn jpeg_encode_matches_wic() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    #[derive(Deserialize)]
    struct EncCase {
        width: u32,
        height: u32,
        gray: bool,
    }
    let cases: BTreeMap<String, EncCase> =
        serde_json::from_slice(&std::fs::read(dir().join("encode/cases.json")).unwrap()).unwrap();
    assert!(!cases.is_empty());

    let mut failures = Vec::new();
    for (name, case) in &cases {
        let pixels = std::fs::read(dir().join("encode/inputs").join(format!("{name}.bin"))).unwrap();
        let want = std::fs::read(dir().join("encode").join(format!("{name}.jpg"))).unwrap();
        let fmt = if case.gray { WicFormat::Gray8 } else { WicFormat::Bgr24 };
        // WIC default: quality 90, 4:2:0; CreateBitmapFromMemory bitmaps carry 0 DPI → APP0 density 0.
        let got = encode_jpeg(fmt, case.width, case.height, &pixels, 90, Subsampling::S420, (0, 0));
        if got != want {
            let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
            failures.push(format!("{name}: differ at byte {at}, len {} vs {}", got.len(), want.len()));
        }
    }
    assert!(failures.is_empty(), "WIC JPEG encode mismatches:\n  {}", failures.join("\n  "));
}

/// Malformed/truncated files must be rejected with WIC's exact `WINCODEC_ERR_*` HRESULT (or decode,
/// for the cases libjpeg tolerates). Oracle `errors.json` captured with `errors_capture.py` on the
/// reference VM: a case with `"error"` is rejected with that HRESULT; one with frame fields decodes.
#[test]
fn jpeg_error_hresults() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    #[derive(Deserialize)]
    struct ErrCase {
        #[serde(default)]
        error: Option<String>,
    }
    let cases: BTreeMap<String, ErrCase> =
        serde_json::from_slice(&std::fs::read(dir().join("errors.json")).unwrap()).unwrap();
    assert!(!cases.is_empty());

    let mut failures = Vec::new();
    for (name, case) in &cases {
        let input = std::fs::read(dir().join("errors").join(format!("{name}.jpg"))).unwrap();
        match (decode_jpeg(&input), &case.error) {
            (Err(got), Some(want)) => {
                let want = u32::from_str_radix(want.trim_start_matches("0x"), 16).unwrap();
                if got != want {
                    failures.push(format!("{name}: HRESULT 0x{got:08x} != {want:#010x}"));
                }
            }
            (Ok(_), Some(want)) => failures.push(format!("{name}: decoded but WIC rejects it ({want})")),
            (Err(got), None) => failures.push(format!("{name}: rejected 0x{got:08x} but WIC decodes it")),
            (Ok(_), None) => {}
        }
    }
    assert!(failures.is_empty(), "WIC JPEG error-HRESULT mismatches:\n  {}", failures.join("\n  "));
}

/// Malformed input must never panic: truncations and single-byte corruptions of every corpus file
/// may only ever return `Ok`/`Err`, exercising the parser, entropy reader and IDCT bounds.
#[test]
fn jpeg_malformed_no_panic() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    for entry in std::fs::read_dir(dir().join("corpus")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("jpg") {
            continue;
        }
        let good = std::fs::read(&path).unwrap();
        // Every truncation.
        for n in 0..good.len() {
            let _ = decode_jpeg(&good[..n]);
        }
        // Single-byte corruptions at a stride of positions (keeps the test quick).
        for i in (0..good.len()).step_by(7) {
            let mut bad = good.clone();
            bad[i] ^= 0xFF;
            let _ = decode_jpeg(&bad);
        }
    }
}
