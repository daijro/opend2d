//! Direct2D text-blend conformance: feed native DirectWrite alpha textures into `d2d::text::blend`
//! and require the result to equal native `ID2D1DeviceContext::DrawGlyphRun` on WARP, byte for byte.
//!
//! Fixtures in `fixtures/*.json` are captured from Direct2D on the
//! reference VM (Windows 11 LTSC 26100): each holds the `IDWriteGlyphRunAnalysis` alpha texture
//! (grayscale 1bpp or ClearType 3bpp R,G,B) + bounds, the `GetAlphaBlendParams` scalars, the run's
//! colour/AA/background, and the premultiplied-BGRA8 `DrawGlyphRun` output over that background.
//! This verifies the blend independently of DirectWrite's (unimplemented) hinting rasterizer.

use d2d::abi::{D2D1_ALPHA_MODE_PREMULTIPLIED, DXGI_FORMAT_B8G8R8A8_UNORM};
use d2d::contracts::{ImageDesc, PixelBuffer};
use d2d::pixels::premultiply;
use d2d::text::blend::{blend_cleartype, blend_grayscale, AlphaTexture, BlendParams};
use serde_json::Value;

/// Decode standard base64 (the fixtures carry no line breaks or padding-free variants).
fn b64_decode(s: &str) -> Vec<u8> {
    let val = |c: u8| match c {
        b'A'..=b'Z' => c - b'A',
        b'a'..=b'z' => c - b'a' + 26,
        b'0'..=b'9' => c - b'0' + 52,
        b'+' => 62,
        b'/' => 63,
        _ => 64,
    };
    let mut out = Vec::new();
    let (mut acc, mut bits) = (0u32, 0u32);
    for &c in s.as_bytes() {
        let v = val(c);
        if v == 64 {
            continue; // '=' padding or whitespace
        }
        acc = acc << 6 | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    out
}

fn f32s(v: &Value) -> [f32; 4] {
    let a = v.as_array().unwrap();
    std::array::from_fn(|i| a[i].as_f64().unwrap() as f32)
}

fn target_cleared(size: [u32; 2], bg: [f32; 4]) -> PixelBuffer {
    let px = premultiply(bg);
    PixelBuffer {
        desc: ImageDesc {
            width: size[0],
            height: size[1],
            format: DXGI_FORMAT_B8G8R8A8_UNORM,
            alpha_mode: D2D1_ALPHA_MODE_PREMULTIPLIED,
            dpi_x: 96.0,
            dpi_y: 96.0,
        },
        stride: size[0] as usize * 4,
        data: px.iter().copied().cycle().take(size[0] as usize * size[1] as usize * 4).collect(),
    }
}

fn check(name: &str) {
    // Fixtures live in the conformance suite (the conformance/suite submodule); absent ⇒ skip.
    let Some(suite) = d2d_reference::suite_dir() else { return };
    let path = suite.join(format!("fixtures/text_blend/fixtures/tblend-{name}.json"));
    let j: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let fg = f32s(&j["fg"]);
    let bg = f32s(&j["bg"]);
    let size = [j["target"][0].as_u64().unwrap() as u32, j["target"][1].as_u64().unwrap() as u32];
    let b = &j["bounds"];
    let bounds = [0, 1, 2, 3].map(|i| b[i].as_i64().unwrap() as i32);
    let alpha = b64_decode(j["alpha_b64"].as_str().unwrap());
    let expected = b64_decode(j["out_b64"].as_str().unwrap());
    let (gamma, contrast, level) = (
        j["blend"]["gamma"].as_f64().unwrap() as f32,
        j["blend"]["contrast"].as_f64().unwrap() as f32,
        j["blend"]["level"].as_f64().unwrap() as f32,
    );
    let bp = BlendParams::resolve(fg, gamma, contrast, level);
    let tex = AlphaTexture { bytes: &alpha, bounds };

    let mut target = target_cleared(size, bg);
    // These fixtures are plain DrawGlyphRun (source-over); no SetPrimitiveBlend is exercised here.
    match j["d2d_aa"].as_u64().unwrap() {
        1 => blend_cleartype(&mut target, tex, fg, &bp, d2d::pixels::Blend::SourceOver),
        aa => blend_grayscale(&mut target, tex, fg, &bp, aa as u32, d2d::pixels::Blend::SourceOver),
    }

    let diffs = target.data.iter().zip(&expected).filter(|(a, b)| a != b).count();
    let maxd = target.data.iter().zip(&expected).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap_or(0);
    assert_eq!(
        target.data, expected,
        "text blend {name}: {diffs}/{} bytes differ (max abs {maxd}) from native DrawGlyphRun",
        expected.len()
    );
}

#[test]
fn grayscale_black() {
    check("gray-black");
}

#[test]
fn grayscale_red() {
    check("gray-red");
}

#[test]
fn grayscale_cyan() {
    check("gray-cyan");
}

#[test]
fn grayscale_bold_natural_symmetric() {
    check("gray-big");
}

#[test]
fn aliased_black() {
    check("aliased-black");
}

#[test]
fn cleartype_black() {
    check("ct-black");
}

#[test]
fn cleartype_light_on_dark() {
    check("ct-lightondark");
}
