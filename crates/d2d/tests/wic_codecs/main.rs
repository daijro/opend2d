//! WIC BMP/GIF/ICO/TIFF decode conformance: every corpus image decoded by `d2d::wic` must match
//! real WIC byte-for-byte — native frame format, size, stride, palette and raw
//! pixels. The oracle is captured with `capture.py` (runs `wic-dump` on the reference VM against
//! `gen.py`'s corpus): `cases.json` holds each frame's format/size/stride/palette, and
//! `captures/<name>.f<N>.bin` its pixels.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use d2d::wic::{
    decode_bmp, decode_gif, decode_ico, decode_tiff, encode_bmp, encode_gif, encode_tiff, Decoded,
    WicError,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Frame {
    format: String,
    width: u32,
    height: u32,
    bpp: u32,
    stride: usize,
    palette: String,
}
#[derive(Deserialize)]
struct Case {
    file: String,
    #[serde(default)]
    frames: Vec<Frame>,
}

fn dir() -> PathBuf {
    d2d_reference::suite_dir().unwrap().join("fixtures/wic_codecs")
}

fn decode(file: &str, bytes: &[u8]) -> Result<Vec<Decoded>, String> {
    match Path::new(file).extension().and_then(|e| e.to_str()) {
        Some("bmp") => decode_bmp(bytes).map(|d| vec![d]).map_err(|e| format!("{e:?}")),
        Some("ico") | Some("cur") => decode_ico(bytes).map_err(|e| format!("{e:?}")),
        Some("gif") => decode_gif(bytes).map_err(|e| format!("{e:?}")),
        Some("tif") | Some("tiff") => decode_tiff(bytes).map_err(|e| format!("{e:?}")),
        other => Err(format!("no decoder for extension {other:?}")),
    }
}

fn palette_hex(pal: &[u32]) -> String {
    pal.iter().map(|c| format!("{c:08x}")).collect::<Vec<_>>().join(",")
}

#[test]
fn wic_codecs_match() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    let cases: BTreeMap<String, Case> =
        serde_json::from_slice(&std::fs::read(dir().join("cases.json")).unwrap()).unwrap();
    assert!(!cases.is_empty());

    let mut failures = Vec::new();
    for (name, case) in &cases {
        let bytes = std::fs::read(dir().join("corpus").join(&case.file)).unwrap();
        let frames = match decode(&case.file, &bytes) {
            Ok(f) => f,
            Err(e) => {
                failures.push(format!("{name}: decode error {e}"));
                continue;
            }
        };
        if frames.len() != case.frames.len() {
            failures.push(format!(
                "{name}: {} frames decoded != {} expected",
                frames.len(),
                case.frames.len()
            ));
            continue;
        }
        for (i, (got, exp)) in frames.iter().zip(&case.frames).enumerate() {
            let expected =
                std::fs::read(dir().join("captures").join(format!("{name}.f{i}.bin"))).unwrap();
            let pal = palette_hex(&got.palette);
            if got.format != exp.format {
                failures.push(format!("{name}.f{i}: format {} != {}", got.format, exp.format));
            } else if (got.width, got.height) != (exp.width, exp.height) {
                failures.push(format!(
                    "{name}.f{i}: size {}x{} != {}x{}",
                    got.width, got.height, exp.width, exp.height
                ));
            } else if got.stride != exp.stride || got.bpp() != exp.bpp {
                failures.push(format!(
                    "{name}.f{i}: stride {}/bpp {} != {}/{}",
                    got.stride,
                    got.bpp(),
                    exp.stride,
                    exp.bpp
                ));
            } else if pal != exp.palette {
                failures.push(format!("{name}.f{i}: palette [{pal}] != [{}]", exp.palette));
            } else if got.pixels != expected {
                let at = got
                    .pixels
                    .iter()
                    .zip(&expected)
                    .position(|(a, b)| a != b)
                    .unwrap_or(got.pixels.len().min(expected.len()));
                failures.push(format!(
                    "{name}.f{i}: pixels differ at byte {at} (len {} vs {})",
                    got.pixels.len(),
                    expected.len()
                ));
            }
        }
    }
    assert!(failures.is_empty(), "WIC codec mismatches:\n  {}", failures.join("\n  "));
}

#[derive(Deserialize)]
struct EncCase {
    suffix: String,
    w: u32,
    h: u32,
    stride: usize,
    palette: Vec<String>,
}

/// `encode_bmp` must produce the real BMP encoder's exact bytes (oracle in `captures_enc/`).
#[test]
fn encode_bmp_matches_wic() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    let cases: BTreeMap<String, EncCase> =
        serde_json::from_slice(&std::fs::read(dir().join("enc_cases.json")).unwrap()).unwrap();
    let mut failures = Vec::new();
    for (name, c) in &cases {
        let pixels = std::fs::read(dir().join("enc_inputs").join(format!("{name}.bin"))).unwrap();
        let expected = std::fs::read(dir().join("captures_enc").join(format!("{name}.bmp"))).unwrap();
        let img = Decoded {
            width: c.w,
            height: c.h,
            format: format!("6fddc324-4e03-4bfe-b185-3d77768dc9{}", c.suffix),
            stride: c.stride,
            pixels,
            palette: c.palette.iter().map(|s| u32::from_str_radix(s, 16).unwrap()).collect(),
        };
        // The oracle encodes a CreateBitmapFromMemory source with no SetResolution -> 0 DPM.
        match encode_bmp(&img, 0.0) {
            Ok(got) if got == expected => {}
            Ok(got) => {
                let at = got.iter().zip(&expected).position(|(a, b)| a != b).unwrap_or(got.len().min(expected.len()));
                failures.push(format!("{name}: bytes differ at {at} (len {} vs {})", got.len(), expected.len()));
            }
            Err(e) => failures.push(format!("{name}: encode error {e:?}")),
        }
    }
    assert!(failures.is_empty(), "BMP encode mismatches:\n  {}", failures.join("\n  "));
}

#[derive(Deserialize)]
struct GifEncCase {
    w: u32,
    h: u32,
    pal: Vec<String>,
}

/// `encode_gif` must produce the real GIF encoder's exact bytes (oracle in `gif_captures/`), across
/// palette sizes, LZW dictionary growth / sub-block splitting and multi-batch (large) images.
#[test]
fn encode_gif_matches_wic() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    let cases: BTreeMap<String, GifEncCase> =
        serde_json::from_slice(&std::fs::read(dir().join("gif_enc_cases.json")).unwrap()).unwrap();
    assert!(!cases.is_empty());
    let mut failures = Vec::new();
    for (name, c) in &cases {
        let pixels = std::fs::read(dir().join("gif_enc_inputs").join(format!("{name}.bin"))).unwrap();
        let expected = std::fs::read(dir().join("gif_captures").join(format!("{name}.gif"))).unwrap();
        let img = Decoded {
            width: c.w,
            height: c.h,
            format: "6fddc324-4e03-4bfe-b185-3d77768dc904".into(), // 8bppIndexed
            stride: c.w as usize,
            pixels,
            palette: c.pal.iter().map(|s| u32::from_str_radix(s, 16).unwrap()).collect(),
        };
        match encode_gif(&img) {
            Ok(got) if got == expected => {}
            Ok(got) => {
                let at = got.iter().zip(&expected).position(|(a, b)| a != b).unwrap_or(got.len().min(expected.len()));
                failures.push(format!("{name}: bytes differ at {at} (len {} vs {})", got.len(), expected.len()));
            }
            Err(e) => failures.push(format!("{name}: encode error {e:?}")),
        }
    }
    assert!(failures.is_empty(), "GIF encode mismatches:\n  {}", failures.join("\n  "));
}

#[derive(Deserialize)]
struct TifEncCase {
    suffix: String,
    w: u32,
    h: u32,
    stride: usize,
    pal: Vec<String>,
}

/// `encode_tiff` must produce the real TIFF encoder's exact bytes (oracle in `tif_captures/`):
/// LZW + horizontal predictor, 96 dpi, across gray / RGB / RGBA / palette and single/multi-strip.
#[test]
fn encode_tiff_matches_wic() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    let cases: BTreeMap<String, TifEncCase> =
        serde_json::from_slice(&std::fs::read(dir().join("tif_enc_cases.json")).unwrap()).unwrap();
    assert!(!cases.is_empty());
    let mut failures = Vec::new();
    for (name, c) in &cases {
        let pixels = std::fs::read(dir().join("tif_enc_inputs").join(format!("{name}.bin"))).unwrap();
        let expected = std::fs::read(dir().join("tif_captures").join(format!("{name}.tif"))).unwrap();
        let img = Decoded {
            width: c.w,
            height: c.h,
            format: format!("6fddc324-4e03-4bfe-b185-3d77768dc9{}", c.suffix),
            stride: c.stride,
            pixels,
            palette: c.pal.iter().map(|s| u32::from_str_radix(s, 16).unwrap()).collect(),
        };
        match encode_tiff(&img) {
            Ok(got) if got == expected => {}
            Ok(got) => {
                let at = got.iter().zip(&expected).position(|(a, b)| a != b).unwrap_or(got.len().min(expected.len()));
                failures.push(format!("{name}: bytes differ at {at} (len {} vs {})", got.len(), expected.len()));
            }
            Err(e) => failures.push(format!("{name}: encode error {e:?}")),
        }
    }
    assert!(failures.is_empty(), "TIFF encode mismatches:\n  {}", failures.join("\n  "));
}

/// Malformed-input rejection: each decoder must return the exact `WINCODEC_ERR_*` HRESULT the real
/// codec returns, at the first failing call. The byte patterns and their HRESULTs were captured from
/// native WIC (decoder created by CLSID, `Initialize` → `GetFrame` → `CopyPixels`,
/// run under Wine). STREAMREAD(0x88982F72) is a too-short header read; the content codes differ per
/// codec: BMP UNKNOWNIMAGEFORMAT(0x88982F07), GIF BADHEADER(0x88982F61), ICO/TIFF BADIMAGE(0x88982F60).
#[test]
fn malformed_input_hresults_match_wic() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    // A minimal but valid 2×2 24bpp BMP header (14-byte file header + 40-byte BITMAPINFOHEADER)
    // with no pixel data, and the same with an invalid 7bpp depth.
    fn bmp_header(bpp: u16) -> Vec<u8> {
        let mut v = b"BM".to_vec();
        v.extend_from_slice(&70u32.to_le_bytes()); // bfSize
        v.extend_from_slice(&0u32.to_le_bytes()); // reserved
        v.extend_from_slice(&54u32.to_le_bytes()); // bfOffBits
        v.extend_from_slice(&40u32.to_le_bytes()); // biSize
        v.extend_from_slice(&2i32.to_le_bytes()); // width
        v.extend_from_slice(&2i32.to_le_bytes()); // height
        v.extend_from_slice(&1u16.to_le_bytes()); // planes
        v.extend_from_slice(&bpp.to_le_bytes()); // bpp
        v.extend_from_slice(&[0u8; 24]); // comp..biClrImportant
        v
    }
    let garbage = b"not an image at all, really".to_vec();
    type Dec = fn(&[u8]) -> Result<(), WicError>;
    let bmp: Dec = |b| decode_bmp(b).map(|_| ());
    let gif: Dec = |b| decode_gif(b).map(|_| ());
    let ico: Dec = |b| decode_ico(b).map(|_| ());
    let tif: Dec = |b| decode_tiff(b).map(|_| ());

    let cases: &[(&str, Dec, Vec<u8>, u32)] = &[
        ("bmp_empty", bmp, vec![], 0x8898_2F72),
        ("bmp_magic", bmp, b"BM".to_vec(), 0x8898_2F72),
        ("bmp_trunchdr", bmp, [b"BM".as_ref(), &[0u8; 10]].concat(), 0x8898_2F72),
        ("bmp_garbage", bmp, garbage.clone(), 0x8898_2F07),
        ("bmp_badbpp", bmp, bmp_header(7), 0x8898_2F07),
        ("bmp_nopixels", bmp, bmp_header(24), 0x8898_2F60),
        ("gif_sig", gif, b"GIF89a".to_vec(), 0x8898_2F72),
        ("gif_truncscreen", gif, [b"GIF89a".as_ref(), &[2, 0]].concat(), 0x8898_2F72),
        ("gif_empty", gif, vec![], 0x8898_2F72),
        ("gif_garbage", gif, garbage.clone(), 0x8898_2F61),
        ("ico_magic", ico, vec![0, 0, 1, 0, 1, 0], 0x8898_2F60),
        ("ico_empty", ico, vec![], 0x8898_2F72),
        ("ico_garbage", ico, garbage.clone(), 0x8898_2F60),
        ("tif_empty", tif, vec![], 0x8898_2F60),
        ("tif_magic", tif, b"II\x2a\x00".to_vec(), 0x8898_2F60),
        ("tif_badifdoff", tif, b"II\x2a\x00\xff\xff\x00\x00".to_vec(), 0x8898_2F72),
        ("tif_garbage", tif, garbage, 0x8898_2F60),
    ];
    let mut failures = Vec::new();
    for (name, dec, bytes, want) in cases {
        match dec(bytes) {
            Err(e) if e.hresult() == *want => {}
            Err(e) => failures.push(format!("{name}: got 0x{:08x} want 0x{want:08x}", e.hresult())),
            Ok(()) => failures.push(format!("{name}: decoded ok, expected error 0x{want:08x}")),
        }
    }
    assert!(failures.is_empty(), "malformed-input HRESULT mismatches:\n  {}", failures.join("\n  "));
}

#[derive(Deserialize)]
struct PngEncCase {
    suffix: String,
    w: u32,
    h: u32,
    stride: usize,
    #[serde(default)]
    palette: Vec<String>,
}

/// `encode_png` must produce the real PNG encoder's exact bytes (GDI+ `SPNGWRITE` + Chromium zlib):
/// IHDR/sRGB/gAMA/IDAT/IEND, the adaptive filter heuristic and byte-identical deflate, per format.
#[test]
fn encode_png_matches_wic() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    use d2d::wic::encode_png;
    let dir = dir();
    let cases: BTreeMap<String, PngEncCase> =
        serde_json::from_slice(&std::fs::read(dir.join("png_enc_cases.json")).unwrap()).unwrap();
    assert!(!cases.is_empty());
    let mut failures = Vec::new();
    for (name, c) in &cases {
        let pixels = std::fs::read(dir.join("png_enc_inputs").join(format!("{name}.bin"))).unwrap();
        let expected = std::fs::read(dir.join("png_captures").join(format!("{name}.png"))).unwrap();
        let img = Decoded {
            width: c.w,
            height: c.h,
            format: format!("6fddc324-4e03-4bfe-b185-3d77768dc9{}", c.suffix),
            stride: c.stride,
            pixels,
            palette: c.palette.iter().map(|s| u32::from_str_radix(s, 16).unwrap()).collect(),
        };
        match encode_png(&img) {
            Ok(got) if got == expected => {}
            Ok(got) => {
                let at = got.iter().zip(&expected).position(|(a, b)| a != b).unwrap_or(got.len().min(expected.len()));
                failures.push(format!("{name}: bytes differ at {at} (len {} vs {})", got.len(), expected.len()));
            }
            Err(e) => failures.push(format!("{name}: encode error {e:?}")),
        }
    }
    assert!(failures.is_empty(), "PNG encode mismatches:\n  {}", failures.join("\n  "));
}
