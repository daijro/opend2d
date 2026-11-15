//! WIC JPEG-XR decode conformance: every corpus `.jxr` decoded by `d2d::wic::decode_jxr` must match
//! real WIC byte-for-byte — native frame format, size, bpp, stride and raw pixels. The
//! oracle was captured from native WIC on the reference system: `cases.json` holds each
//! frame's format/size/bpp/stride, `captures/<name>.f0.bin` its pixels. The corpus covers every sample
//! type (8/16-bit unsigned, 16/32-bit fixed point, 16-bit half, 32-bit float), grey/RGB/BGR, planar
//! alpha (straight and premultiplied), and several encoder modes (lossy, lossless, tiled, interleaved
//! alpha) — all decode identically.

use std::collections::BTreeMap;
use std::path::PathBuf;

use d2d::wic::{decode_jxr, encode_jxr, JxrEncodeOptions};
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
    d2d_reference::suite_dir().unwrap().join("fixtures/wic/jxr")
}

#[test]
fn jxr_decode_matches_wic() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    let cases: BTreeMap<String, Case> =
        serde_json::from_slice(&std::fs::read(dir().join("cases.json")).unwrap()).unwrap();
    assert!(!cases.is_empty());

    let mut failures = Vec::new();
    for (name, case) in &cases {
        let input = std::fs::read(dir().join("corpus").join(format!("{name}.jxr"))).unwrap();
        let expected = std::fs::read(dir().join("captures").join(format!("{name}.f0.bin"))).unwrap();
        let img = match decode_jxr(&input) {
            Ok(img) => img,
            Err(e) => {
                failures.push(format!("{name}: decode error {e:?}"));
                continue;
            }
        };
        if img.guid() != case.format {
            failures.push(format!("{name}: format {} != {}", img.guid(), case.format));
        } else if (img.width, img.height) != (case.width, case.height) {
            failures.push(format!(
                "{name}: size {}x{} != {}x{}",
                img.width, img.height, case.width, case.height
            ));
        } else if img.bpp != case.bpp || img.stride != case.stride {
            failures.push(format!(
                "{name}: bpp {}/stride {} != {}/{}",
                img.bpp, img.stride, case.bpp, case.stride
            ));
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
    assert!(failures.is_empty(), "WIC JPEG-XR decode mismatches:\n  {}", failures.join("\n  "));
}

/// Malformed input must return an error, never panic or read out of bounds. (Real WIC rejects garbage
/// and empty input with `WINCODEC_ERR_COMPONENTNOTFOUND` 0x88982F50 and finds no usable frame in a
/// truncated/empty-directory file; we map all of these to the coarse `WicError`, matching how the PNG
/// decoder treats malformed input — exact `WINCODEC_ERR_*` parity is separate work.)
#[test]
fn jxr_malformed_is_rejected() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    let good = std::fs::read(dir().join("corpus/bgra32.jxr")).unwrap();
    let cases: &[&[u8]] = &[
        &[],                       // empty
        b"not an image at all!!",  // bad magic
        &good[..8],                // header only, no directory
        &good[..100],              // valid header, truncated codestream
        &[0x49, 0x49, 0xBC, 0x01, 0x08, 0, 0, 0, 0, 0, 0, 0], // II/BC magic, empty directory
    ];
    for (i, bytes) in cases.iter().enumerate() {
        assert!(decode_jxr(bytes).is_err(), "malformed case {i} should be rejected");
    }
}

/// Encoder conformance (the JPEG-XR *encoder* task). `encode/cases.json` + `encode/jxrenc-*.jxr` are
/// real WIC encodings — captured from native WIC on the reference VM — of the
/// `captures/<source>` pixel buffers across a matrix of WIC encoder options (default, lossless,
/// quality levels, overlap, spatial order, tiling, subsampling, interleaved alpha). `encode_jxr` must
/// reproduce each byte-for-byte.
#[derive(Deserialize)]
struct EncodeCase {
    source: String,
    format: String,
    width: u32,
    height: u32,
    stride: usize,
    file: String,
    options: BTreeMap<String, serde_json::Value>,
}

/// `6fddc324-4e03-4bfe-b185-3d77768dc90c` → the 16 raw little-endian GUID bytes (data1/2/3 LE, data4
/// as-is) that the container and `encode_jxr` use.
fn guid_bytes(s: &str) -> [u8; 16] {
    let hex: Vec<u8> = s.bytes().filter(|b| *b != b'-').collect();
    let byte = |i: usize| u8::from_str_radix(std::str::from_utf8(&hex[i * 2..i * 2 + 2]).unwrap(), 16).unwrap();
    let mut g = [0u8; 16];
    for (dst, src) in [0, 1, 2, 3].iter().zip([3, 2, 1, 0]) {
        g[*dst] = byte(src); // data1 LE
    }
    g[4] = byte(5);
    g[5] = byte(4); // data2 LE
    g[6] = byte(7);
    g[7] = byte(6); // data3 LE
    for i in 8..16 {
        g[i] = byte(i); // data4 big-endian
    }
    g
}

fn options_of(map: &BTreeMap<String, serde_json::Value>) -> JxrEncodeOptions {
    let mut o = JxrEncodeOptions::default();
    for (k, v) in map {
        match k.as_str() {
            "ImageQuality" => o.image_quality = v.as_f64().unwrap() as f32,
            "Lossless" => o.lossless = v.as_bool().unwrap(),
            "HorizontalTileSlices" => o.h_tile_slices = v.as_u64().unwrap() as u16,
            "VerticalTileSlices" => o.v_tile_slices = v.as_u64().unwrap() as u16,
            // Accepted but no-ops without UseCodecOptions, which no committed oracle sets.
            "UseCodecOptions" | "Overlap" | "Subsampling" | "FrequencyOrder" | "InterleavedAlpha" => {}
            other => panic!("unhandled option {other}"),
        }
    }
    o
}

#[test]
fn jxr_encode_matches_wic() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    let enc = dir().join("encode");
    let cases: BTreeMap<String, EncodeCase> =
        serde_json::from_slice(&std::fs::read(enc.join("cases.json")).unwrap()).unwrap();
    assert!(!cases.is_empty());

    let mut failures = Vec::new();
    for (name, case) in &cases {
        let pixels = std::fs::read(dir().join("captures").join(&case.source)).unwrap();
        let expected = std::fs::read(enc.join(&case.file)).unwrap();
        let got = match encode_jxr(
            &guid_bytes(&case.format),
            case.width,
            case.height,
            &pixels,
            case.stride,
            &options_of(&case.options),
            (96.0, 96.0), // these oracles were captured at 96 DPI
        ) {
            Ok(b) => b,
            Err(e) => {
                failures.push(format!("{name}: encode error {e:?}"));
                continue;
            }
        };
        if got != expected {
            let at = got.iter().zip(&expected).position(|(a, b)| a != b).unwrap_or(got.len().min(expected.len()));
            failures.push(format!("{name}: differ at byte {at} (len {} vs {})", got.len(), expected.len()));
        }
    }
    assert!(failures.is_empty(), "WIC JPEG-XR encode mismatches:\n  {}", failures.join("\n  "));
}
