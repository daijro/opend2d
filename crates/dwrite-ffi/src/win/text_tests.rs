//! `IDWriteTextFormat1`/`2`/`3` state: documented defaults + set→get round-trips, driven through the
//! real COM vtable (`DWriteCreateFactory` → `CreateTextFormat` → QI the format up). Gated on
//! `D2D_FONTS` so "Arial" resolves through the system collection, like `font_discovery_tests`.

use windows::Win32::Graphics::DirectWrite::*;
use windows_core::{Interface, PCWSTR};

fn arial_format3() -> Option<IDWriteTextFormat3> {
    let fonts = std::env::var_os("D2D_FONTS")?;
    if !std::path::Path::new(&fonts).join("arial.ttf").exists() {
        return None;
    }
    unsafe {
        let mut raw = std::ptr::null_mut();
        super::DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, &IDWriteFactory::IID, &mut raw).ok().unwrap();
        let factory = IDWriteFactory::from_raw(raw);
        let family: Vec<u16> = "Arial\0".encode_utf16().collect();
        let locale: Vec<u16> = "en-us\0".encode_utf16().collect();
        let fmt = factory
            .CreateTextFormat(PCWSTR(family.as_ptr()), None, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 16.0, PCWSTR(locale.as_ptr()))
            .unwrap();
        // The format must QI up to IDWriteTextFormat3 (version parity with native dwrite).
        Some(fmt.cast().unwrap())
    }
}

#[test]
fn text_format3_defaults_and_roundtrips() {
    let Some(f3) = arial_format3() else { return };
    unsafe {
        // --- documented defaults (fresh format) ---
        assert_eq!(f3.GetVerticalGlyphOrientation(), DWRITE_VERTICAL_GLYPH_ORIENTATION_DEFAULT);
        assert!(f3.GetLastLineWrapping().as_bool());
        assert_eq!(f3.GetOpticalAlignment(), DWRITE_OPTICAL_ALIGNMENT_NONE);
        assert_eq!(f3.GetAutomaticFontAxes(), DWRITE_AUTOMATIC_FONT_AXES_NONE);
        assert_eq!(f3.GetFontAxisValueCount(), 0);

        // --- IDWriteTextFormat1 round-trips ---
        f3.SetVerticalGlyphOrientation(DWRITE_VERTICAL_GLYPH_ORIENTATION_STACKED).unwrap();
        assert_eq!(f3.GetVerticalGlyphOrientation(), DWRITE_VERTICAL_GLYPH_ORIENTATION_STACKED);
        f3.SetLastLineWrapping(false).unwrap();
        assert!(!f3.GetLastLineWrapping().as_bool());
        f3.SetOpticalAlignment(DWRITE_OPTICAL_ALIGNMENT_NO_SIDE_BEARINGS).unwrap();
        assert_eq!(f3.GetOpticalAlignment(), DWRITE_OPTICAL_ALIGNMENT_NO_SIDE_BEARINGS);
        // GetFontFallback returns a non-null fallback once one is set.
        let fb = f3.GetFontFallback().unwrap();
        f3.SetFontFallback(&fb).unwrap();
        assert!(f3.GetFontFallback().is_ok());

        // --- IDWriteTextFormat2 line spacing (full struct) ---
        let ls = DWRITE_LINE_SPACING {
            method: DWRITE_LINE_SPACING_METHOD_UNIFORM,
            height: 20.0,
            baseline: 16.0,
            leadingBefore: 0.25,
            fontLineGapUsage: DWRITE_FONT_LINE_GAP_USAGE_ENABLED,
        };
        f3.SetLineSpacing(&ls).unwrap();
        let mut got = DWRITE_LINE_SPACING::default();
        f3.GetLineSpacing(&mut got).unwrap();
        assert_eq!(got.method, ls.method);
        assert_eq!(got.height, 20.0);
        assert_eq!(got.baseline, 16.0);
        assert_eq!(got.leadingBefore, 0.25);
        assert_eq!(got.fontLineGapUsage, ls.fontLineGapUsage);
        // The base (3-arg) getter reads the same unified method/height/baseline.
        let base: IDWriteTextFormat = f3.cast().unwrap();
        let (mut m, mut h, mut b) = (DWRITE_LINE_SPACING_METHOD_DEFAULT, 0.0f32, 0.0f32);
        base.GetLineSpacing(&mut m, &mut h, &mut b).unwrap();
        assert_eq!(m, DWRITE_LINE_SPACING_METHOD_UNIFORM);
        assert_eq!(h, 20.0);
        assert_eq!(b, 16.0);

        // --- IDWriteTextFormat3 font axis values ---
        f3.SetAutomaticFontAxes(DWRITE_AUTOMATIC_FONT_AXES_OPTICAL_SIZE).unwrap();
        assert_eq!(f3.GetAutomaticFontAxes(), DWRITE_AUTOMATIC_FONT_AXES_OPTICAL_SIZE);
        let axes = [
            DWRITE_FONT_AXIS_VALUE { axisTag: DWRITE_FONT_AXIS_TAG_WEIGHT, value: 650.0 },
            DWRITE_FONT_AXIS_VALUE { axisTag: DWRITE_FONT_AXIS_TAG_WIDTH, value: 90.0 },
        ];
        f3.SetFontAxisValues(&axes).unwrap();
        assert_eq!(f3.GetFontAxisValueCount(), 2);
        // Too-small buffer → the HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER) probe contract.
        let mut one = [DWRITE_FONT_AXIS_VALUE::default(); 1];
        assert!(f3.GetFontAxisValues(&mut one).is_err());
        let mut out = [DWRITE_FONT_AXIS_VALUE::default(); 2];
        f3.GetFontAxisValues(&mut out).unwrap();
        assert_eq!(out[0].axisTag, DWRITE_FONT_AXIS_TAG_WEIGHT);
        assert_eq!(out[0].value, 650.0);
        assert_eq!(out[1].axisTag, DWRITE_FONT_AXIS_TAG_WIDTH);
        assert_eq!(out[1].value, 90.0);
    }
}
