//! WIC pixel-processing conformance: `d2d::wic` convert / scale / flip-rotate / clip must match real
//! native WIC byte-for-byte. Oracle captured from native WIC (the same native DLL as the
//! reference VM, run locally under Wine); fixtures live beside this file.

use std::path::PathBuf;

use d2d::wic::convert::{convert, Format};
use d2d::wic::fliprotate::{clip, flip_rotate};
use serde::Deserialize;

fn dir() -> PathBuf {
    d2d_reference::suite_dir().unwrap().join("fixtures/wic_process")
}

#[derive(Deserialize)]
struct ConvCase {
    src: String,
    dst: String,
    src_fmt: String,
    dst_fmt: String,
    width: u32,
    height: u32,
    #[serde(default)]
    palette: Vec<String>,
}

#[test]
fn convert_matches_wic() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    let d = dir().join("convert");
    let cases: Vec<ConvCase> =
        serde_json::from_slice(&std::fs::read(d.join("cases.json")).unwrap()).unwrap();
    assert!(!cases.is_empty());
    let mut fails = Vec::new();
    for c in &cases {
        let src = std::fs::read(d.join(format!("{}.src.bin", c.src))).unwrap();
        let expected = std::fs::read(d.join(format!("{}-{}.out.bin", c.src, c.dst))).unwrap();
        let (sf, df) = (
            Format::from_guid(&c.src_fmt).unwrap(),
            Format::from_guid(&c.dst_fmt).unwrap(),
        );
        let palette: Vec<u32> =
            c.palette.iter().map(|h| u32::from_str_radix(h, 16).unwrap()).collect();
        match convert(sf, df, &src, c.width, c.height, &palette) {
            Ok(got) if got == expected => {}
            Ok(got) => {
                let at = got.iter().zip(&expected).position(|(a, b)| a != b);
                fails.push(format!(
                    "{}->{}: differ at {at:?} (len {} vs {})",
                    c.src,
                    c.dst,
                    got.len(),
                    expected.len()
                ));
            }
            Err(e) => fails.push(format!("{}->{}: error {e:?}", c.src, c.dst)),
        }
    }
    assert!(fails.is_empty(), "WIC convert mismatches:\n  {}", fails.join("\n  "));
}

#[derive(Deserialize)]
struct FrCase {
    src: String,
    width: u32,
    height: u32,
    bpp: u32,
    options: u32,
}

#[test]
fn flip_rotate_matches_wic() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    let d = dir().join("fliprotate");
    let cases: Vec<FrCase> =
        serde_json::from_slice(&std::fs::read(d.join("cases.json")).unwrap()).unwrap();
    assert!(!cases.is_empty());
    let mut fails = Vec::new();
    for c in &cases {
        let src = std::fs::read(d.join(format!("{}.src.bin", c.src))).unwrap();
        let expected =
            std::fs::read(d.join(format!("{}-{}.out.bin", c.src, c.options))).unwrap();
        let (got, _, _) = flip_rotate(&src, c.width, c.height, c.bpp, c.options).unwrap();
        if got != expected {
            let at = got.iter().zip(&expected).position(|(a, b)| a != b);
            fails.push(format!("{} opt {}: differ at {at:?}", c.src, c.options));
        }
    }
    assert!(fails.is_empty(), "WIC flip/rotate mismatches:\n  {}", fails.join("\n  "));
}

#[derive(Deserialize)]
struct ClipCase {
    src: String,
    width: u32,
    height: u32,
    bpp: u32,
    x: u32,
    y: u32,
    cw: u32,
    ch: u32,
}

#[test]
fn clip_matches_wic() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    let d = dir().join("clip");
    let cases: Vec<ClipCase> =
        serde_json::from_slice(&std::fs::read(d.join("cases.json")).unwrap()).unwrap();
    assert!(!cases.is_empty());
    let mut fails = Vec::new();
    for c in &cases {
        let src = std::fs::read(d.join(format!("{}.src.bin", c.src))).unwrap();
        let name = format!("{}-{}_{}_{}_{}", c.src, c.x, c.y, c.cw, c.ch);
        let expected = std::fs::read(d.join(format!("{name}.out.bin"))).unwrap();
        let got = clip(&src, c.width, c.height, c.bpp, c.x, c.y, c.cw, c.ch).unwrap();
        if got != expected {
            let at = got.iter().zip(&expected).position(|(a, b)| a != b);
            fails.push(format!("{name}: differ at {at:?}"));
        }
    }
    assert!(fails.is_empty(), "WIC clip mismatches:\n  {}", fails.join("\n  "));
}

#[derive(Deserialize)]
struct ScaleCase {
    src: String,
    srcw: u32,
    srch: u32,
    bpp: u32,
    dstw: u32,
    dsth: u32,
    mode: i32,
    #[serde(default)]
    straight_alpha: bool,
}

#[test]
fn scale_matches_wic() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    use d2d::wic::scale::{scale, Interp};
    let d = dir().join("scale");
    let cases: Vec<ScaleCase> =
        serde_json::from_slice(&std::fs::read(d.join("cases.json")).unwrap()).unwrap();
    assert!(!cases.is_empty());
    let mut fails = Vec::new();
    for c in &cases {
        let src = std::fs::read(d.join(format!("{}.src.bin", c.src))).unwrap();
        let name = format!("{}-m{}-{}x{}", c.src, c.mode, c.dstw, c.dsth);
        let expected = std::fs::read(d.join(format!("{name}.out.bin"))).unwrap();
        let got = scale(
            &src, c.srcw, c.srch, c.bpp, c.dstw, c.dsth,
            Interp::from_mode(c.mode).unwrap(), c.straight_alpha,
        );
        match got {
            Ok(g) if g == expected => {}
            Ok(g) => {
                let at = g.iter().zip(&expected).position(|(a, b)| a != b);
                fails.push(format!("{name}: differ at {at:?} (len {} vs {})", g.len(), expected.len()));
            }
            Err(e) => fails.push(format!("{name}: error {e:?}")),
        }
    }
    assert!(fails.is_empty(), "WIC scale mismatches:\n  {}", fails.join("\n  "));
}

#[derive(Deserialize)]
struct InterlaceCase {
    name: String,
    file: String,
    format: String,
    width: u32,
    height: u32,
    stride: usize,
    palette: String,
}

/// Adam7-interlaced PNGs decode byte-for-byte like the real codec (same pixels as a non-interlaced
/// frame — the `png` crate de-interlaces; here we confirm it against native WIC on interlaced input).
#[test]
fn interlaced_png_decode_matches_wic() {
    if d2d_reference::suite_dir().is_none() {
        return;
    }
    use d2d::wic::decode_png;
    let d = dir().join("interlace");
    let cases: Vec<InterlaceCase> =
        serde_json::from_slice(&std::fs::read(d.join("cases.json")).unwrap()).unwrap();
    assert!(!cases.is_empty());
    let mut fails = Vec::new();
    for c in &cases {
        let input = std::fs::read(d.join(&c.file)).unwrap();
        let expected = std::fs::read(d.join(format!("{}.out.bin", c.name))).unwrap();
        let img = decode_png(&input).unwrap();
        let palette =
            img.palette.iter().map(|p| format!("{p:08x}")).collect::<Vec<_>>().join(",");
        if img.format.guid() != c.format {
            fails.push(format!("{}: format {} != {}", c.name, img.format.guid(), c.format));
        } else if (img.width, img.height, img.stride) != (c.width, c.height, c.stride) {
            fails.push(format!("{}: size/stride mismatch", c.name));
        } else if palette != c.palette {
            fails.push(format!("{}: palette [{palette}] != [{}]", c.name, c.palette));
        } else if img.pixels != expected {
            let at = img.pixels.iter().zip(&expected).position(|(a, b)| a != b);
            fails.push(format!("{}: pixels differ at {at:?}", c.name));
        }
    }
    assert!(fails.is_empty(), "interlaced PNG decode mismatches:\n  {}", fails.join("\n  "));
}
