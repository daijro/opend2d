//! `IDWriteFontFace4` glyph-image-format detection over real Windows fonts (gated on `D2D_FONTS`):
//! Arial is a pure TrueType outline face; Segoe UI Emoji (Win11) is COLR v1 with v0-compatible base
//! records on top of glyf outlines.

use super::fonts::{FaceObj, FontSource};
use std::sync::Arc;
use windows::Win32::Graphics::DirectWrite::*;
use windows_core::Interface;

fn face(file: &str) -> Option<IDWriteFontFace7> {
    let fonts = std::env::var_os("D2D_FONTS")?;
    let bytes = Arc::new(std::fs::read(std::path::Path::new(&fonts).join(file)).ok()?);
    Some(FaceObj { bytes: bytes.clone(), index: 0, family: "T".into(), source: FontSource::Bytes(bytes), sims: 0, axis_values: Vec::new() }.into())
}

#[test]
fn glyph_image_formats_outline_vs_colr() {
    const TTF: DWRITE_GLYPH_IMAGE_FORMATS = DWRITE_GLYPH_IMAGE_FORMATS_TRUETYPE;
    const COLR: DWRITE_GLYPH_IMAGE_FORMATS = DWRITE_GLYPH_IMAGE_FORMATS_COLR;
    const PAINT: DWRITE_GLYPH_IMAGE_FORMATS = DWRITE_GLYPH_IMAGE_FORMATS_COLR_PAINT_TREE;
    const SVG: DWRITE_GLYPH_IMAGE_FORMATS = DWRITE_GLYPH_IMAGE_FORMATS_SVG;

    if let Some(arial) = face("arial.ttf") {
        assert_eq!(unsafe { arial.GetGlyphImageFormats2() }, TTF, "arial is pure TrueType outline");
    }
    let Some(emoji) = face("seguiemj.ttf") else { return };
    unsafe {
        // glyf outlines + COLR v1 paint tree + v0-compatible base records.
        assert_eq!(emoji.GetGlyphImageFormats2(), TTF | COLR | PAINT);
        assert!(!emoji.GetGlyphImageFormats2().contains(SVG), "seguiemj has no SVG table");

        // Per-glyph `GetGlyphImageFormats(gid, ppemFirst, ppemLast)` reports only the formats in which the
        // glyph is STORED as image data — an SVG document or a colour-bitmap strike covering it — NOT the
        // constructed outline/COLR formats (those are the font-wide `GetGlyphImageFormats2` bits asserted
        // above). seguiemj carries no SVG table and no colour-bitmap strikes, so every glyph reports NONE
        // here — the COLR emoji and the plain ASCII glyph alike. (Native-verified; matches the per-glyph
        // impl corrected in 9769fbe8.)
        const NONE: DWRITE_GLYPH_IMAGE_FORMATS = DWRITE_GLYPH_IMAGE_FORMATS_NONE;
        let base: IDWriteFontFace = emoji.cast().unwrap();
        let cps = [0x1F600u32, 'A' as u32]; // 😀 and 'A'
        let mut gids = [0u16; 2];
        base.GetGlyphIndices(cps.as_ptr(), 2, gids.as_mut_ptr()).unwrap();

        assert_eq!(emoji.GetGlyphImageFormats(gids[0], 0, 1000).unwrap(), NONE, "COLR emoji glyph stores no image data");
        assert_eq!(emoji.GetGlyphImageFormats(gids[1], 0, 1000).unwrap(), NONE, "ASCII glyph stores no image data");

        // A format the glyph lacks → E_INVALIDARG (this font has no PNG/SVG image data).
        let mut data = DWRITE_GLYPH_IMAGE_DATA::default();
        let mut ctx = std::ptr::null_mut();
        assert!(emoji.GetGlyphImageData(gids[1], 32, DWRITE_GLYPH_IMAGE_FORMATS_PNG, &mut data, &mut ctx).is_err());
    }
}
