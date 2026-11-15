//! WIC DDS decode conformance: every corpus `.dds` decoded by `d2d::wic::decode_dds` must match real
//! native WIC byte-for-byte — per-frame native format, size, stride and raw pixels — and the
//! files WIC rejects must error here too. Oracle captured from native WIC on the
//! reference system: `cases.json` holds each file's frames (or the failing WIC HRESULT),
//! `captures/<name>.f<N>.bin` the pixels.

use std::collections::BTreeMap;
use std::path::PathBuf;

use d2d::wic::decode_dds;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    frames: Vec<Frame>,
}

#[derive(Deserialize)]
struct Frame {
    frame: u32,
    format: String,
    width: u32,
    height: u32,
    bpp: u32,
    stride: usize,
}

fn dir() -> PathBuf {
    d2d_reference::suite_dir().unwrap().join("fixtures/dds")
}

#[test]
fn dds_decode_matches_wic() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    let cases: BTreeMap<String, Case> =
        serde_json::from_slice(&std::fs::read(dir().join("cases.json")).unwrap()).unwrap();
    assert!(!cases.is_empty());

    let mut failures = Vec::new();
    for (name, case) in &cases {
        let input = std::fs::read(dir().join("corpus").join(format!("{name}.dds"))).unwrap();
        let got = decode_dds(&input);

        if case.error.is_some() {
            // WIC rejects this file; we must too.
            if got.is_ok() {
                failures.push(format!("{name}: decoded but WIC rejected it ({})", case.error.as_ref().unwrap()));
            }
            continue;
        }

        let frames = match got {
            Ok(f) => f,
            Err(e) => {
                failures.push(format!("{name}: decode error {e:?}, expected {} frame(s)", case.frames.len()));
                continue;
            }
        };
        if frames.len() != case.frames.len() {
            failures.push(format!("{name}: {} frames != {}", frames.len(), case.frames.len()));
            continue;
        }
        for (f, exp) in frames.iter().zip(&case.frames) {
            let i = exp.frame;
            let guid = f.format.guid();
            if guid != exp.format {
                failures.push(format!("{name}.f{i}: format {guid} != {}", exp.format));
            } else if (f.width, f.height) != (exp.width, exp.height) {
                failures.push(format!("{name}.f{i}: {}x{} != {}x{}", f.width, f.height, exp.width, exp.height));
            } else if f.stride != exp.stride || f.format.bits_per_pixel() != exp.bpp {
                failures.push(format!(
                    "{name}.f{i}: stride {}/bpp {} != {}/{}",
                    f.stride, f.format.bits_per_pixel(), exp.stride, exp.bpp
                ));
            } else {
                let want = std::fs::read(dir().join("captures").join(format!("{name}.f{i}.bin"))).unwrap();
                if f.pixels != want {
                    let at = f.pixels.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(f.pixels.len());
                    failures.push(format!(
                        "{name}.f{i}: pixels differ at byte {at} (len {} vs {})",
                        f.pixels.len(),
                        want.len()
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "WIC DDS decode mismatches:\n  {}", failures.join("\n  "));
}
