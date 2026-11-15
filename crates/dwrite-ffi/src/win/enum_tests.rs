//! `IDWriteFontList1` and `IDWriteColorGlyphRunEnumerator1` — QI parity with native (an app gets the
//! base interface then QIs it up) plus the methods each newer leaf adds. Gated on `D2D_FONTS`.

use super::fonts::{FaceObj, FontSource};
use core::mem::ManuallyDrop;
use std::sync::Arc;
use windows::Win32::Graphics::DirectWrite::*;
use windows_core::{BOOL, Interface, PCWSTR};

#[test]
fn font_list1_enumerates_matching_fonts() {
    let Some(fonts) = std::env::var_os("D2D_FONTS") else { return };
    if !std::path::Path::new(&fonts).join("arial.ttf").exists() {
        return;
    }
    unsafe {
        let mut raw = std::ptr::null_mut();
        super::DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, &IDWriteFactory::IID, &mut raw).ok().unwrap();
        let factory = IDWriteFactory::from_raw(raw);
        let mut coll = None;
        factory.GetSystemFontCollection(&mut coll, false).unwrap();
        let coll = coll.unwrap();
        let name: Vec<u16> = "Arial\0".encode_utf16().collect();
        let (mut idx, mut exists) = (0u32, BOOL(0));
        coll.FindFamilyName(PCWSTR(name.as_ptr()), &mut idx, &mut exists).unwrap();
        if !exists.as_bool() {
            return;
        }
        let family = coll.GetFontFamily(idx).unwrap();
        let list = family.GetMatchingFonts(DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STRETCH_NORMAL, DWRITE_FONT_STYLE_NORMAL).unwrap();
        // The GetMatchingFonts result QIs up to IDWriteFontList1 like native dwrite.
        let list1: IDWriteFontList1 = list.cast().unwrap();
        assert!(list1.GetFontCount() >= 1);
        assert_eq!(list1.GetFontLocality(0), DWRITE_LOCALITY_LOCAL);
        // FontList1::GetFont returns the most-derived IDWriteFont3.
        let font3: IDWriteFont3 = list1.GetFont(0).unwrap();
        assert_eq!(font3.GetLocality(), DWRITE_LOCALITY_LOCAL);
        // The reference round-trips back to a usable face.
        let faceref = list1.GetFontFaceReference(0).unwrap();
        faceref.CreateFontFace().unwrap();
        assert!(list1.GetFont(9999).is_err(), "out-of-range index → E_INVALIDARG");
    }
}

#[test]
fn color_enumerator1_reports_run1_fields() {
    let Some(fonts) = std::env::var_os("D2D_FONTS") else { return };
    let Ok(bytes) = std::fs::read(std::path::Path::new(&fonts).join("seguiemj.ttf")) else { return };
    let bytes = Arc::new(bytes);
    unsafe {
        let face4: IDWriteFontFace7 = FaceObj { bytes: bytes.clone(), index: 0, family: "T".into(), source: FontSource::Bytes(bytes), sims: 0, axis_values: Vec::new() }.into();
        let face: IDWriteFontFace = face4.cast().unwrap();
        let mut gid = [0u16; 1];
        face.GetGlyphIndices([0x1F600u32].as_ptr(), 1, gid.as_mut_ptr()).unwrap(); // 😀
        let adv = [0.0f32; 1];
        let run = DWRITE_GLYPH_RUN {
            fontFace: ManuallyDrop::new(Some(face.clone())),
            fontEmSize: 32.0,
            glyphCount: 1,
            glyphIndices: gid.as_ptr(),
            glyphAdvances: adv.as_ptr(),
            glyphOffsets: std::ptr::null(),
            isSideways: BOOL(0),
            bidiLevel: 0,
        };
        let f3: IDWriteFactory7 = super::factory::FactoryObj::new().into();
        let f2: IDWriteFactory2 = f3.cast().unwrap();
        // seguiemj carries COLRv0-compatible base records; a non-colour glyph would be E_NOCOLOR.
        let Ok(en) = f2.TranslateColorGlyphRun(0.0, 0.0, &run, None, DWRITE_MEASURING_MODE_NATURAL, None, 0) else { return };
        // The enumerator QIs up to IDWriteColorGlyphRunEnumerator1 like native.
        let en1: IDWriteColorGlyphRunEnumerator1 = en.cast().unwrap();
        assert!(en1.MoveNext().unwrap().as_bool());
        let r1 = &*en1.GetCurrentRun().unwrap();
        assert_eq!(r1.measuringMode, DWRITE_MEASURING_MODE_NATURAL);
        assert_eq!(r1.glyphImageFormat, DWRITE_GLYPH_IMAGE_FORMATS_TRUETYPE, "seguiemj is a glyf (not CFF) outline font");
        assert_eq!(r1.Base.glyphRun.glyphCount, 1);
        // The base enumerator's GetCurrentRun returns the same address re-cast to &Base.
        let en_base: IDWriteColorGlyphRunEnumerator = en1.cast().unwrap();
        assert_eq!(en_base.GetCurrentRun().unwrap() as usize, en1.GetCurrentRun().unwrap() as usize);
    }
}
