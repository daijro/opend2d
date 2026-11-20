//! `IWICBitmapDecoderInfo` identity fields (the codec registry), driven in-process and diffed against
//! the native `wic_com/oracle/ci_*.txt` top section (everything before the per-frame `f0.*` lines,
//! which need a real decode). Isolates whether the component-info rewrite matches native.

use windows::Win32::Graphics::Imaging::*;
use windows_core::GUID;

// The off-Windows CoTaskMemAlloc/PropVariantClear symbols are defined for real in `meta.rs`.

fn getstr(f: impl Fn(&mut [u16], *mut u32) -> windows_core::Result<()>) -> String {
    let mut n = 0u32;
    let _ = f(&mut [], &mut n);
    if n == 0 {
        return String::new();
    }
    let mut buf = vec![0u16; n as usize];
    if f(&mut buf, &mut n).is_err() {
        return String::new();
    }
    String::from_utf16_lossy(&buf[..n.saturating_sub(1) as usize])
}

fn guids(g: &GUID) -> String {
    super::guid_to_str(g)
}

/// The decoder-info top section, in the exact `wic-dump compinfo` order/format.
fn info_text(di: &IWICBitmapDecoderInfo) -> String {
    use std::fmt::Write;
    let mut o = String::new();
    unsafe {
        writeln!(o, "clsid {}", guids(&di.GetCLSID().unwrap())).ok();
        writeln!(o, "componenttype {}", di.GetComponentType().unwrap().0).ok();
        writeln!(o, "friendlyname {}", getstr(|b, p| di.GetFriendlyName(b, p))).ok();
        writeln!(o, "author {}", getstr(|b, p| di.GetAuthor(b, p))).ok();
        writeln!(o, "vendor {}", guids(&di.GetVendorGUID().unwrap())).ok();
        writeln!(o, "version {}", getstr(|b, p| di.GetVersion(b, p))).ok();
        writeln!(o, "specversion {}", getstr(|b, p| di.GetSpecVersion(b, p))).ok();
        writeln!(o, "infocontainer {}", guids(&di.GetContainerFormat().unwrap())).ok();
        writeln!(o, "mimetypes {}", getstr(|b, p| di.GetMimeTypes(b, p))).ok();
        writeln!(o, "fileextensions {}", getstr(|b, p| di.GetFileExtensions(b, p))).ok();
        writeln!(o, "colormgmt {}", getstr(|b, p| di.GetColorManagementVersion(b, p))).ok();
        writeln!(o, "animation {}", di.DoesSupportAnimation().map(|b| b.as_bool()).unwrap_or(false) as u8).ok();
        writeln!(o, "chromakey {}", di.DoesSupportChromakey().map(|b| b.as_bool()).unwrap_or(false) as u8).ok();
        writeln!(o, "lossless {}", di.DoesSupportLossless().map(|b| b.as_bool()).unwrap_or(false) as u8).ok();
        writeln!(o, "multiframe {}", di.DoesSupportMultiframe().map(|b| b.as_bool()).unwrap_or(false) as u8).ok();
    }
    o
}

#[test]
fn png_decoder_info_matches_oracle_top() {
    let png = GUID::from_u128(0x1b7cfaf4_713f_473c_bbcd_6137425faeaf);
    let di = super::components::decoder_info(&png).unwrap();
    let got = info_text(&di);

    // Read the suite oracle at run time (not include_str!) so the public tree, which ships without
    // the private conformance/suite submodule, still compiles and the test skips when it is absent.
    let oracle_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../conformance/suite/fixtures/wic_com/oracle/ci_png.txt");
    let Ok(oracle) = std::fs::read_to_string(oracle_path) else {
        eprintln!("suite not checked out, skipped");
        return;
    };
    // Compare the decoder-info fields line-by-line (skip `container` — emitted from the decoder, not
    // the info object — and the per-frame `f*`/`pixelformats` lines which need a real decode).
    for line in oracle.lines() {
        let key = line.split(' ').next().unwrap_or("");
        if key == "container" || key == "pixelformats" || key.starts_with('f') && key.contains('.') {
            continue;
        }
        assert!(got.lines().any(|g| g == line), "decoder-info missing/mismatched line:\n  want: {line}\n  got:\n{got}");
    }
}
