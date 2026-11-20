//! `IDWriteFontFace5`/`IDWriteFontResource` variable-font surface. Self-contained: a minimal sfnt
//! carrying only an `fvar` table drives the deterministic axis readback (count/tags/defaults/ranges/
//! attributes), so these run without the system font corpus. The static-font `HasVariations == false`
//! path uses an sfnt with no `fvar` table. The static-font synthesized axis *values* are VM-pending
//! (see `FaceObj::face_axis_values`) and are intentionally not asserted here.

use super::fonts::{FaceObj, FontSource};
use std::sync::Arc;
use windows::Win32::Graphics::DirectWrite::*;
use windows_core::Interface;

/// 16.16 fixed from an f32, as `fvar` stores axis coordinates.
fn fixed(v: f32) -> i32 {
    (v * 65536.0).round() as i32
}

/// A minimal single-face sfnt whose only table is `fvar`, carrying `axes` as
/// `(tag, min, default, max)`. Enough for `FontRef::from_index` + `font.fvar()`.
fn fvar_font(axes: &[(&[u8; 4], f32, f32, f32)]) -> Vec<u8> {
    let mut fvar = Vec::new();
    fvar.extend_from_slice(&1u16.to_be_bytes()); // majorVersion
    fvar.extend_from_slice(&0u16.to_be_bytes()); // minorVersion
    fvar.extend_from_slice(&16u16.to_be_bytes()); // axesArrayOffset (right after the 16-byte header)
    fvar.extend_from_slice(&2u16.to_be_bytes()); // reserved (countSizePairs)
    fvar.extend_from_slice(&(axes.len() as u16).to_be_bytes()); // axisCount
    fvar.extend_from_slice(&20u16.to_be_bytes()); // axisSize
    fvar.extend_from_slice(&0u16.to_be_bytes()); // instanceCount
    fvar.extend_from_slice(&(axes.len() as u16 * 4 + 4).to_be_bytes()); // instanceSize
    for (tag, min, def, max) in axes {
        fvar.extend_from_slice(*tag);
        fvar.extend_from_slice(&fixed(*min).to_be_bytes());
        fvar.extend_from_slice(&fixed(*def).to_be_bytes());
        fvar.extend_from_slice(&fixed(*max).to_be_bytes());
        fvar.extend_from_slice(&0u16.to_be_bytes()); // flags
        fvar.extend_from_slice(&256u16.to_be_bytes()); // axisNameID
    }
    sfnt(&[(b"fvar", &fvar)])
}

/// Wrap tables into a one-font sfnt (TrueType). `searchRange`/`entrySelector`/`rangeShift` are left
/// zero — `read-fonts` does not validate them.
fn sfnt(tables: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
    let n = tables.len();
    let mut out = Vec::new();
    out.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // sfntVersion
    out.extend_from_slice(&(n as u16).to_be_bytes());
    out.extend_from_slice(&0u16.to_be_bytes()); // searchRange
    out.extend_from_slice(&0u16.to_be_bytes()); // entrySelector
    out.extend_from_slice(&0u16.to_be_bytes()); // rangeShift
    let mut offset = 12 + n * 16;
    for (tag, data) in tables {
        out.extend_from_slice(*tag);
        out.extend_from_slice(&0u32.to_be_bytes()); // checksum (unchecked)
        out.extend_from_slice(&(offset as u32).to_be_bytes());
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        offset += data.len();
    }
    for (_, data) in tables {
        out.extend_from_slice(data);
    }
    out
}

fn face(bytes: Vec<u8>) -> IDWriteFontFace5 {
    let bytes = Arc::new(bytes);
    let face: IDWriteFontFace7 =
        FaceObj { bytes: bytes.clone(), index: 0, family: "T".into(), source: FontSource::Bytes(bytes), sims: 0, axis_values: Vec::new() }.into();
    face.cast().unwrap()
}

#[test]
fn fontface5_reports_fvar_axes() {
    let f = face(fvar_font(&[(b"wght", 100.0, 400.0, 900.0), (b"wdth", 50.0, 100.0, 200.0)]));
    unsafe {
        assert!(f.HasVariations().as_bool(), "a font with fvar has variations");
        assert_eq!(f.GetFontAxisValueCount(), 2);
        let mut vals = [DWRITE_FONT_AXIS_VALUE::default(); 2];
        f.GetFontAxisValues(&mut vals).unwrap();
        assert_eq!(vals[0].axisTag, DWRITE_FONT_AXIS_TAG_WEIGHT);
        assert_eq!(vals[0].value, 400.0, "wght defaults to its fvar default");
        assert_eq!(vals[1].axisTag, DWRITE_FONT_AXIS_TAG_WIDTH);
        assert_eq!(vals[1].value, 100.0);
    }
}

#[test]
fn fontface5_static_synthesizes_four_axes() {
    // A font with no fvar table: HasVariations is false and GetFontAxisValues synthesizes the four
    // standard axes wght/wdth/ital/slnt (VM-verified order/shape vs native). With no OS/2 or post,
    // the defaults are weight 400, width 100%, upright (ital 0, slnt 0).
    let f = face(sfnt(&[(b"CFF ", b"stub")]));
    unsafe {
        assert!(!f.HasVariations().as_bool(), "a font without fvar has no variations");
        assert_eq!(f.GetFontAxisValueCount(), 4);
        let mut v = [DWRITE_FONT_AXIS_VALUE::default(); 4];
        f.GetFontAxisValues(&mut v).unwrap();
        assert_eq!(v[0], DWRITE_FONT_AXIS_VALUE { axisTag: DWRITE_FONT_AXIS_TAG_WEIGHT, value: 400.0 });
        assert_eq!(v[1], DWRITE_FONT_AXIS_VALUE { axisTag: DWRITE_FONT_AXIS_TAG_WIDTH, value: 100.0 });
        assert_eq!(v[2], DWRITE_FONT_AXIS_VALUE { axisTag: DWRITE_FONT_AXIS_TAG_ITALIC, value: 0.0 });
        assert_eq!(v[3], DWRITE_FONT_AXIS_VALUE { axisTag: DWRITE_FONT_AXIS_TAG_SLANT, value: 0.0 });
    }
}

#[test]
fn fontresource_exposes_axis_model() {
    let f = face(fvar_font(&[(b"wght", 100.0, 400.0, 900.0), (b"slnt", 0.0, 0.0, 0.0)]));
    unsafe {
        let r = f.GetFontResource().unwrap();
        assert!(r.HasVariations().as_bool());
        // Native's FontResource reports the fvar axes plus the registered axes absent from fvar,
        // fixed at metric-derived defaults, in canonical order opsz/wght/wdth/ital/slnt — VM-verified
        // on Segoe UI Variable (fvar opsz,wght → [opsz, wght, wdth 100, ital 0, slnt 0]). This font's
        // fvar is wght + slnt, so the model is [wght 100-900, wdth 100, ital 0, slnt 0].
        assert_eq!(r.GetFontAxisCount(), 4);
        assert_eq!(r.GetFontFaceIndex(), 0);

        let mut defs = [DWRITE_FONT_AXIS_VALUE::default(); 4];
        r.GetDefaultFontAxisValues(&mut defs).unwrap();
        assert_eq!(defs.map(|d| (d.axisTag, d.value)), [
            (DWRITE_FONT_AXIS_TAG_WEIGHT, 400.0),
            (DWRITE_FONT_AXIS_TAG_WIDTH, 100.0),
            (DWRITE_FONT_AXIS_TAG_ITALIC, 0.0),
            (DWRITE_FONT_AXIS_TAG_SLANT, 0.0),
        ]);

        let mut ranges = [DWRITE_FONT_AXIS_RANGE::default(); 4];
        r.GetFontAxisRanges(&mut ranges).unwrap();
        assert_eq!((ranges[0].minValue, ranges[0].maxValue), (100.0, 900.0)); // wght from fvar
        assert_eq!((ranges[1].minValue, ranges[1].maxValue), (100.0, 100.0)); // wdth synthesized, fixed
        assert_eq!((ranges[2].minValue, ranges[2].maxValue), (0.0, 0.0)); // ital synthesized, fixed
        assert_eq!((ranges[3].minValue, ranges[3].maxValue), (0.0, 0.0)); // slnt from fvar, pinned

        // Only wght varies (min != max) → VARIABLE; the rest are pinned → not VARIABLE.
        assert_eq!(r.GetFontAxisAttributes(0), DWRITE_FONT_AXIS_ATTRIBUTES_VARIABLE);
        assert_eq!(r.GetFontAxisAttributes(1), DWRITE_FONT_AXIS_ATTRIBUTES_NONE);
        assert_eq!(r.GetFontAxisAttributes(3), DWRITE_FONT_AXIS_ATTRIBUTES_NONE);
    }
}

#[test]
fn fontresource_creates_axis_valued_reference() {
    let f = face(fvar_font(&[(b"wght", 100.0, 400.0, 900.0), (b"wdth", 50.0, 100.0, 200.0)]));
    unsafe {
        let r = f.GetFontResource().unwrap();
        // The reference resolves to the face's own axis values (core does not instance coordinates).
        let faceref = r.CreateFontFaceReference(DWRITE_FONT_SIMULATIONS_NONE, &[]).unwrap();
        assert_eq!(faceref.GetFontAxisValueCount(), 2);
        let mut vals = [DWRITE_FONT_AXIS_VALUE::default(); 2];
        faceref.GetFontAxisValues(&mut vals).unwrap();
        assert_eq!(vals[0].axisTag, DWRITE_FONT_AXIS_TAG_WEIGHT);
        assert_eq!(vals[0].value, 400.0);
        assert_eq!(vals[1].axisTag, DWRITE_FONT_AXIS_TAG_WIDTH);
        // CreateFontFace off the reference round-trips back to a FontFace5 with the same axes.
        let f5 = faceref.CreateFontFace().unwrap();
        assert_eq!(f5.GetFontAxisValueCount(), 2);
    }
}

#[test]
fn fontset_leaf_surface() {
    // A two-member set: a variable face (fvar) + a static one. QI the base set up to FontSet4.
    let var = Arc::new(fvar_font(&[(b"wght", 100.0, 400.0, 900.0), (b"wdth", 50.0, 100.0, 200.0)]));
    let stat = Arc::new(sfnt(&[(b"CFF ", b"stub")]));
    let set = super::fontset::font_set(vec![
        (FontSource::Bytes(var.clone()), 0),
        (FontSource::Bytes(stat.clone()), 0),
    ]);
    unsafe {
        let s4: IDWriteFontSet4 = set.cast().unwrap();
        let s1: IDWriteFontSet1 = set.cast().unwrap();
        assert_eq!(s4.GetFontCount(), 2);

        // FontSet1 member accessors on the variable member.
        assert_eq!(s1.GetFontLocality(0), DWRITE_LOCALITY_LOCAL);
        let r = s1.CreateFontResource(0).unwrap();
        // fvar wght + wdth, plus the synthesized ital + slnt (native's FontResource axis model).
        assert_eq!(r.GetFontAxisCount(), 4);
        let f5 = s1.CreateFontFace(0).unwrap();
        assert!(f5.HasVariations().as_bool());
        let _fr: IDWriteFontFaceReference1 = s1.GetFontFaceReference(0).unwrap();

        // FontSet1::GetFontAxisRanges reports the member's raw fvar variation axes (wght + wdth),
        // not the FontResource design-axis model — so 2, unchanged by the resource synthesis.
        let mut ranges = [DWRITE_FONT_AXIS_RANGE::default(); 2];
        let mut n = 0u32;
        s1.GetFontAxisRanges(0, &mut ranges, &mut n).unwrap();
        assert_eq!(n, 2);

        // GetFilteredFonts by explicit index selects just that member.
        let only_var = s1.GetFilteredFonts(&[0]).unwrap();
        assert_eq!(only_var.GetFontCount(), 1);

        // FontSet3 source type: a memory member is UNKNOWN.
        let s3: IDWriteFontSet3 = set.cast().unwrap();
        assert_eq!(s3.GetFontSourceType(0), DWRITE_FONT_SOURCE_TYPE_UNKNOWN);

        // FontSet4 weight/stretch/style → axis values: native always emits wght, wdth, ital AND slnt
        // (ITALIC → ital 1 / slnt −12), VM-verified.
        let mut out = [DWRITE_FONT_AXIS_VALUE::default(); 5];
        let count = s4.ConvertWeightStretchStyleToFontAxisValues(
            None,
            DWRITE_FONT_WEIGHT(700),
            DWRITE_FONT_STRETCH(5),
            DWRITE_FONT_STYLE_ITALIC,
            0.0,
            &mut out,
        );
        assert_eq!(count, 4);
        assert_eq!(out[0], DWRITE_FONT_AXIS_VALUE { axisTag: DWRITE_FONT_AXIS_TAG_WEIGHT, value: 700.0 });
        assert_eq!(out[1], DWRITE_FONT_AXIS_VALUE { axisTag: DWRITE_FONT_AXIS_TAG_WIDTH, value: 100.0 });
        assert_eq!(out[2], DWRITE_FONT_AXIS_VALUE { axisTag: DWRITE_FONT_AXIS_TAG_ITALIC, value: 1.0 });
        assert_eq!(out[3], DWRITE_FONT_AXIS_VALUE { axisTag: DWRITE_FONT_AXIS_TAG_SLANT, value: -12.0 });
    }
}

#[test]
fn fontface7_paint_feature_level_from_colr() {
    // No COLR table → no paint feature level for the COLR formats.
    let f = face(fvar_font(&[(b"wght", 100.0, 400.0, 900.0)]));
    unsafe {
        let f7: IDWriteFontFace7 = f.cast().unwrap();
        assert_eq!(
            f7.GetPaintFeatureLevel(DWRITE_GLYPH_IMAGE_FORMATS_COLR_PAINT_TREE),
            DWRITE_PAINT_FEATURE_LEVEL_NONE
        );
    }

    // A COLR v1 table → COLR_V1 for the paint-tree format, and none for a non-COLR format.
    let colr = {
        let mut t = Vec::new();
        t.extend_from_slice(&1u16.to_be_bytes()); // version 1
        t.extend_from_slice(&0u16.to_be_bytes()); // numBaseGlyphRecords
        t.extend_from_slice(&[0u8; 10]); // remaining v0 header (offsets/count), unused here
        t
    };
    let f = face(sfnt(&[(b"COLR", &colr)]));
    unsafe {
        let f7: IDWriteFontFace7 = f.cast().unwrap();
        assert_eq!(
            f7.GetPaintFeatureLevel(DWRITE_GLYPH_IMAGE_FORMATS_COLR_PAINT_TREE),
            DWRITE_PAINT_FEATURE_LEVEL_COLR_V1
        );
        assert_eq!(
            f7.GetPaintFeatureLevel(DWRITE_GLYPH_IMAGE_FORMATS_TRUETYPE),
            DWRITE_PAINT_FEATURE_LEVEL_NONE
        );
    }
}
