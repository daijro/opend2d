//! The DirectWrite factory (`IDWriteFactory`…3) and rendering-params object. The factory builds the
//! system font collection, text formats/layouts and custom rendering params; everything else stubs.

use super::{E_NOTIMPL, analysis, colorrun, fontfile, fonts, fonts::SystemFonts, fontset, misc, text, textanalyzer, wide_to_string};
#[cfg(windows)] // GDI interop is Windows-only (see win/mod.rs).
use super::gdi;
use std::cell::RefCell;
use std::sync::Arc;
use windows::Win32::Graphics::DirectWrite::*;
use windows_core::{BOOL, Interface, PCWSTR, Ref, Result as WResult, implement};

#[implement(IDWriteFactory7)]
pub(crate) struct FactoryObj {
    fonts: RefCell<Option<Arc<SystemFonts>>>,
}

impl FactoryObj {
    pub(crate) fn new() -> Self {
        Self { fonts: RefCell::new(None) }
    }
    pub(crate) fn fonts(&self) -> Arc<SystemFonts> {
        self.fonts.borrow_mut().get_or_insert_with(SystemFonts::scan).clone()
    }
    /// The system fonts grouped under the requested family model: the cached weight-stretch-style
    /// scan by default, or a fresh typographic scan (requested rarely, not cached).
    pub(crate) fn system_fonts(&self, model: DWRITE_FONT_FAMILY_MODEL) -> Arc<SystemFonts> {
        if model == DWRITE_FONT_FAMILY_MODEL_TYPOGRAPHIC {
            SystemFonts::scan_model(true)
        } else {
            self.fonts()
        }
    }
}

impl IDWriteFactory_Impl for FactoryObj_Impl {
    fn GetSystemFontCollection(&self, fontcollection: windows_core::OutRef<IDWriteFontCollection>, _checkforupdates: BOOL) -> WResult<()> {
        let c = fonts::collection(self.fonts());
        fontcollection.write(Some(c))
    }
    fn CreateCustomFontCollection(&self, collectionloader: Ref<IDWriteFontCollectionLoader>, collectionkey: *const core::ffi::c_void, collectionkeysize: u32) -> WResult<IDWriteFontCollection> {
        let loader = collectionloader.as_ref().ok_or(E_NOTIMPL)?;
        // Enumerate the loader's files through a factory handle (ours), read each file's bytes now,
        // then group every face into a collection — the same grouping as the system scan.
        let factory: IDWriteFactory = { let f: IDWriteFactory7 = FactoryObj::new().into(); f.cast()? };
        let enumerator = unsafe { loader.CreateEnumeratorFromKey(&factory, collectionkey, collectionkeysize)? };
        let mut sources = Vec::new();
        while unsafe { enumerator.MoveNext()? }.as_bool() {
            let file = unsafe { enumerator.GetCurrentFontFile()? };
            let bytes = fontfile::read_font_file(&file)?;
            sources.push(file.cast_object_ref::<fontfile::FontFileObj>().map(|f| f.as_source())
                .unwrap_or_else(|_| fonts::FontSource::Bytes(bytes)));
        }
        Ok(fonts::collection(SystemFonts::from_sources(sources)))
    }
    fn RegisterFontCollectionLoader(&self, _l: Ref<IDWriteFontCollectionLoader>) -> WResult<()> {
        Ok(())
    }
    fn UnregisterFontCollectionLoader(&self, _l: Ref<IDWriteFontCollectionLoader>) -> WResult<()> {
        Ok(())
    }
    fn CreateFontFileReference(&self, filepath: &PCWSTR, _lastwritetime: *const windows::Win32::Foundation::FILETIME) -> WResult<IDWriteFontFile> {
        fontfile::file_from_path(&wide_to_string(filepath.0, None))
    }
    fn CreateCustomFontFileReference(&self, k: *const core::ffi::c_void, ks: u32, l: Ref<IDWriteFontFileLoader>) -> WResult<IDWriteFontFile> {
        let loader = l.as_ref().ok_or(E_NOTIMPL)?;
        fontfile::file_from_loader(unsafe { core::slice::from_raw_parts(k as *const u8, ks as usize) }, loader)
    }
    fn CreateFontFace(&self, _t: DWRITE_FONT_FACE_TYPE, n: u32, f: *const Option<IDWriteFontFile>, i: u32, _s: DWRITE_FONT_SIMULATIONS) -> WResult<IDWriteFontFace> {
        fontfile::make_font_face(n, f, i)
    }
    fn CreateRenderingParams(&self) -> WResult<IDWriteRenderingParams> {
        default_rendering_params()
    }
    fn CreateMonitorRenderingParams(&self, _m: windows::Win32::Graphics::Gdi::HMONITOR) -> WResult<IDWriteRenderingParams> {
        // Per-monitor gamma/geometry aren't modeled; the system default matches native's default
        // monitor params on the reference VM (gamma 1.8, enhanced 0.5, RGB, default mode).
        default_rendering_params()
    }
    fn CreateCustomRenderingParams(&self, gamma: f32, enhancedcontrast: f32, cleartypelevel: f32, pixelgeometry: DWRITE_PIXEL_GEOMETRY, renderingmode: DWRITE_RENDERING_MODE) -> WResult<IDWriteRenderingParams> {
        let p: IDWriteRenderingParams3 = ParamsObj::new(gamma, enhancedcontrast, 1.0, cleartypelevel, pixelgeometry.0, renderingmode.0, 0).into();
        Ok(p.cast()?)
    }
    fn RegisterFontFileLoader(&self, _l: Ref<IDWriteFontFileLoader>) -> WResult<()> {
        Ok(())
    }
    fn UnregisterFontFileLoader(&self, _l: Ref<IDWriteFontFileLoader>) -> WResult<()> {
        Ok(())
    }
    fn CreateTextFormat(&self, fontfamilyname: &PCWSTR, fontcollection: Ref<IDWriteFontCollection>, fontweight: DWRITE_FONT_WEIGHT, fontstyle: DWRITE_FONT_STYLE, fontstretch: DWRITE_FONT_STRETCH, fontsize: f32, localename: &PCWSTR) -> WResult<IDWriteTextFormat> {
        // Native rejects a non-positive or NaN em size (VM-verified: 0, <0, NaN → E_INVALIDARG;
        // any positive value, including +inf, is accepted). `!(fontsize > 0.0)` captures exactly that.
        if !(fontsize > 0.0) {
            return Err(windows_core::Error::from(windows_core::HRESULT(0x8007_0057u32 as i32)));
        }
        let family = wide_to_string(fontfamilyname.0, None);
        let locale = wide_to_string(localename.0, None);
        let fonts = match fontcollection.as_ref() {
            Some(collection) => collection.cast_object_ref::<fonts::CollectionObj>()?.fonts.clone(),
            None => self.fonts(),
        };
        text::make_format(&fonts, &family, fontweight.0 as u16, fontstyle.0 as u32, fontstretch.0 as u16, fontsize, &locale)
    }
    fn CreateTypography(&self) -> WResult<IDWriteTypography> {
        Ok(misc::typography())
    }
    fn GetGdiInterop(&self) -> WResult<IDWriteGdiInterop> {
        // GDI interop (LOGFONT ⇄ font, GDI bitmap render target) is Windows-only.
        #[cfg(not(windows))]
        return Err(E_NOTIMPL.into());
        #[cfg(windows)]
        Ok(gdi::GdiInteropObj { fonts: self.fonts() }.into())
    }
    fn CreateTextLayout(&self, string: &PCWSTR, stringlength: u32, textformat: Ref<IDWriteTextFormat>, maxwidth: f32, maxheight: f32) -> WResult<IDWriteTextLayout> {
        let s = wide_to_string(string.0, Some(stringlength as usize));
        text::make_layout(textformat, &s, maxwidth, maxheight, 0, 1.0)
    }
    fn CreateGdiCompatibleTextLayout(&self, string: &PCWSTR, stringlength: u32, textformat: Ref<IDWriteTextFormat>, layoutwidth: f32, layoutheight: f32, pixelsperdip: f32, transform: *const DWRITE_MATRIX, usegdinatural: BOOL) -> WResult<IDWriteTextLayout> {
        let s = wide_to_string(string.0, Some(stringlength as usize));
        // Native rejects a GDI-compatible layout whose DEVICE em (em · pixelsPerDip · transformScale)
        // reaches 2^15 = 32768 — GetGdiCompatibleGlyphMetrics' limit, enforced here at creation (plain
        // CreateTextLayout never rejects). VM-pinned 2026-11-08 (r11 16180): dpi 96 edge em 32767.998
        // S_OK / 32768 E_INVALIDARG; dpi 192 (ppd 2) edge em 16384 — i.e. the device em, not the font em.
        if let Some(fmt) = textformat.as_ref() {
            let sc = unsafe { transform.as_ref() }
                .map_or(1.0, |t| (t.m11 as f64).hypot(t.m12 as f64).max((t.m21 as f64).hypot(t.m22 as f64)) as f32);
            if d2d::DeviceContext::gdi_compat_layout_em_rejected(unsafe { fmt.GetFontSize() }, pixelsperdip, sc) {
                return Err(windows_core::Error::from(windows_core::HRESULT(0x8007_0057u32 as i32)));
            }
        }
        // `pixelsperdip` feeds the GDI device ppem (VDMX line metrics + hinted advances); ignoring it
        // measured NATURAL. ponytail: the layout itself still assumes `transform` scale 1 (identity in
        // every GDI scene so far); thread its max-singular into make_layout if a sheared one appears.
        text::make_layout(textformat, &s, layoutwidth, layoutheight, if usegdinatural.as_bool() { 2 } else { 1 }, pixelsperdip)
    }
    fn CreateEllipsisTrimmingSign(&self, textformat: Ref<IDWriteTextFormat>) -> WResult<IDWriteInlineObject> {
        text::make_ellipsis(textformat)
    }
    fn CreateTextAnalyzer(&self) -> WResult<IDWriteTextAnalyzer> {
        Ok(textanalyzer::analyzer())
    }
    fn CreateNumberSubstitution(&self, _m: DWRITE_NUMBER_SUBSTITUTION_METHOD, _l: &PCWSTR, _i: BOOL) -> WResult<IDWriteNumberSubstitution> {
        // The interface is opaque (no methods); the analyzer treats digits as-is, so the config is
        // not acted on. ponytail: carry method/locale here if a number-substitution scene needs it.
        Ok(misc::number_substitution())
    }
    fn CreateGlyphRunAnalysis(&self, g: *const DWRITE_GLYPH_RUN, p: f32, t: *const DWRITE_MATRIX, r: DWRITE_RENDERING_MODE, m: DWRITE_MEASURING_MODE, bx: f32, by: f32) -> WResult<IDWriteGlyphRunAnalysis> {
        // The pre-F2 overload: no grid-fit/antialias args (grid-fit default; ClearType unless the
        // rendering mode is aliased, which `create` keys off directly) and a separate pixels-per-dip
        // folded into the transform.
        analysis::create(g, matrix_scaled(t, p), r.0, m.0, 0, DWRITE_TEXT_ANTIALIAS_MODE_CLEARTYPE.0, [bx, by])
    }
}

/// The system default `IDWriteRenderingParams`: the values native `CreateRenderingParams` returns on
/// the reference VM — gamma 1.8, enhanced contrast 0.5, grayscale contrast 1.0, ClearType level 1.0,
/// RGB pixel geometry, `DEFAULT` rendering mode. Hardcoded (not read from the host registry) so the
/// drop-in matches native regardless of the environment's ClearType settings.
/// The system's default text rendering params (`CreateRenderingParams`): what a device context draws text with until
/// the app sets its own.
pub fn default_rendering_params() -> WResult<IDWriteRenderingParams> {
    let p: IDWriteRenderingParams3 = ParamsObj::new(1.8, 0.5, 1.0, 1.0, DWRITE_PIXEL_GEOMETRY_RGB.0, DWRITE_RENDERING_MODE_DEFAULT.0, DWRITE_GRID_FIT_MODE_DEFAULT.0).into();
    Ok(p.cast()?)
}

/// `DWRITE_MATRIX` (or identity) scaled by `ppd` (pixels-per-dip) into the full DIP→device transform.
fn matrix_scaled(t: *const DWRITE_MATRIX, ppd: f32) -> [f64; 6] {
    let m = unsafe { t.as_ref() }.copied().unwrap_or(DWRITE_MATRIX { m11: 1.0, m12: 0.0, m21: 0.0, m22: 1.0, dx: 0.0, dy: 0.0 });
    let s = ppd as f64;
    [m.m11 as f64 * s, m.m12 as f64 * s, m.m21 as f64 * s, m.m22 as f64 * s, m.dx as f64, m.dy as f64]
}

impl IDWriteFactory1_Impl for FactoryObj_Impl {
    fn GetEudcFontCollection(&self, _c: windows_core::OutRef<IDWriteFontCollection>, _u: BOOL) -> WResult<()> {
        Ok(())
    }
    fn CreateCustomRenderingParams(&self, gamma: f32, enhancedcontrast: f32, enhancedcontrastgrayscale: f32, cleartypelevel: f32, pixelgeometry: DWRITE_PIXEL_GEOMETRY, renderingmode: DWRITE_RENDERING_MODE) -> WResult<IDWriteRenderingParams1> {
        let p: IDWriteRenderingParams3 = ParamsObj::new(gamma, enhancedcontrast, enhancedcontrastgrayscale, cleartypelevel, pixelgeometry.0, renderingmode.0, 0).into();
        Ok(p.cast()?)
    }
}

impl IDWriteFactory2_Impl for FactoryObj_Impl {
    fn GetSystemFontFallback(&self) -> WResult<IDWriteFontFallback> {
        Ok(fonts::FallbackObj { fonts: self.fonts(), mappings: Vec::new() }.into())
    }
    fn CreateFontFallbackBuilder(&self) -> WResult<IDWriteFontFallbackBuilder> {
        Ok(fonts::fallback_builder(self.fonts()))
    }
    fn TranslateColorGlyphRun(&self, bx: f32, by: f32, g: *const DWRITE_GLYPH_RUN, _d: *const DWRITE_GLYPH_RUN_DESCRIPTION, m: DWRITE_MEASURING_MODE, _w: *const DWRITE_MATRIX, c: u32) -> WResult<IDWriteColorGlyphRunEnumerator> {
        colorrun::create(bx, by, g, c, m)
    }
    fn CreateCustomRenderingParams(&self, gamma: f32, enhancedcontrast: f32, grayscaleenhancedcontrast: f32, cleartypelevel: f32, pixelgeometry: DWRITE_PIXEL_GEOMETRY, renderingmode: DWRITE_RENDERING_MODE, gridfitmode: DWRITE_GRID_FIT_MODE) -> WResult<IDWriteRenderingParams2> {
        let p: IDWriteRenderingParams3 = ParamsObj::new(gamma, enhancedcontrast, grayscaleenhancedcontrast, cleartypelevel, pixelgeometry.0, renderingmode.0, gridfitmode.0).into();
        Ok(p.cast()?)
    }
    fn CreateGlyphRunAnalysis(&self, g: *const DWRITE_GLYPH_RUN, t: *const DWRITE_MATRIX, r: DWRITE_RENDERING_MODE, m: DWRITE_MEASURING_MODE, gf: DWRITE_GRID_FIT_MODE, aa: DWRITE_TEXT_ANTIALIAS_MODE, bx: f32, by: f32) -> WResult<IDWriteGlyphRunAnalysis> {
        analysis::create(g, matrix_scaled(t, 1.0), r.0, m.0, gf.0, aa.0, [bx, by])
    }
}

impl IDWriteFactory3_Impl for FactoryObj_Impl {
    fn CreateGlyphRunAnalysis(&self, g: *const DWRITE_GLYPH_RUN, t: *const DWRITE_MATRIX, r: DWRITE_RENDERING_MODE1, m: DWRITE_MEASURING_MODE, gf: DWRITE_GRID_FIT_MODE, aa: DWRITE_TEXT_ANTIALIAS_MODE, bx: f32, by: f32) -> WResult<IDWriteGlyphRunAnalysis> {
        analysis::create(g, matrix_scaled(t, 1.0), r.0, m.0, gf.0, aa.0, [bx, by])
    }
    fn CreateCustomRenderingParams(&self, gamma: f32, enhancedcontrast: f32, grayscaleenhancedcontrast: f32, cleartypelevel: f32, pixelgeometry: DWRITE_PIXEL_GEOMETRY, renderingmode: DWRITE_RENDERING_MODE1, gridfitmode: DWRITE_GRID_FIT_MODE) -> WResult<IDWriteRenderingParams3> {
        Ok(ParamsObj::new(gamma, enhancedcontrast, grayscaleenhancedcontrast, cleartypelevel, pixelgeometry.0, renderingmode.0, gridfitmode.0).into())
    }
    fn CreateFontFaceReference(&self, f: Ref<IDWriteFontFile>, i: u32, s: DWRITE_FONT_SIMULATIONS) -> WResult<IDWriteFontFaceReference> {
        fontfile::faceref_from_file(f.as_ref().ok_or(E_NOTIMPL)?, i, s)
    }
    fn CreateFontFaceReference2(&self, filepath: &PCWSTR, _l: *const windows::Win32::Foundation::FILETIME, i: u32, s: DWRITE_FONT_SIMULATIONS) -> WResult<IDWriteFontFaceReference> {
        fontfile::faceref_from_path(filepath, i, s)
    }
    fn GetSystemFontSet(&self) -> WResult<IDWriteFontSet> {
        Ok(fontset::font_set(self.fonts().face_refs()))
    }
    fn CreateFontSetBuilder(&self) -> WResult<IDWriteFontSetBuilder> {
        Ok(fontset::builder())
    }
    fn CreateFontCollectionFromFontSet(&self, s: Ref<IDWriteFontSet>) -> WResult<IDWriteFontCollection1> {
        // Group the set's members into families, exactly like the system/custom collections.
        let faces = fontset::set_faces(s.as_ref().ok_or(E_NOTIMPL)?).ok_or(E_NOTIMPL)?;
        Ok(fonts::collection1(SystemFonts::from_faces(faces)))
    }
    fn GetSystemFontCollection(&self, _d: BOOL, c: windows_core::OutRef<IDWriteFontCollection1>, _u: BOOL) -> WResult<()> {
        c.write(Some(fonts::collection1(self.fonts())))
    }
    fn GetFontDownloadQueue(&self) -> WResult<IDWriteFontDownloadQueue> {
        Ok(misc::download_queue())
    }
}

impl IDWriteFactory4_Impl for FactoryObj_Impl {
    fn TranslateColorGlyphRun(&self, baselineorigin: &windows_numerics::Vector2, glyphrun: *const DWRITE_GLYPH_RUN, _glyphrundescription: *const DWRITE_GLYPH_RUN_DESCRIPTION, _desiredglyphimageformats: DWRITE_GLYPH_IMAGE_FORMATS, measuringmode: DWRITE_MEASURING_MODE, _worldanddpitransform: *const DWRITE_MATRIX, colorpaletteindex: u32) -> WResult<IDWriteColorGlyphRunEnumerator1> {
        Ok(colorrun::create(baselineorigin.X, baselineorigin.Y, glyphrun, colorpaletteindex, measuringmode)?.cast()?)
    }
    // Off-Windows (the shim binding, where the FFI conformance runs) the glyphOrigins out-array is
    // passed through and filled for every glyph. On Windows, windows-rs 0.61.3 models the array as a
    // single returned Vector2 and its #[implement] vtable thunk writes only glyphOrigins[0], so the
    // full array cannot be produced through it there (same for the shim's upstream bindgen); the impl
    // returns origin[0]. See `glyph_origins`.
    #[cfg(not(windows))]
    fn ComputeGlyphOrigins(&self, glyphrun: *const DWRITE_GLYPH_RUN, baselineorigin: &windows_numerics::Vector2, glyphorigins: *mut windows_numerics::Vector2) -> WResult<()> {
        write_glyph_origins(glyphrun, *baselineorigin, 0, IDENTITY, glyphorigins);
        Ok(())
    }
    #[cfg(windows)]
    fn ComputeGlyphOrigins(&self, glyphrun: *const DWRITE_GLYPH_RUN, baselineorigin: &windows_numerics::Vector2) -> WResult<windows_numerics::Vector2> {
        Ok(glyph_origins(glyphrun, *baselineorigin, 0, IDENTITY).first().copied().unwrap_or(*baselineorigin))
    }
    #[cfg(not(windows))]
    fn ComputeGlyphOrigins2(&self, glyphrun: *const DWRITE_GLYPH_RUN, measuringmode: DWRITE_MEASURING_MODE, baselineorigin: &windows_numerics::Vector2, worldanddpitransform: *const DWRITE_MATRIX, glyphorigins: *mut windows_numerics::Vector2) -> WResult<()> {
        write_glyph_origins(glyphrun, *baselineorigin, measuringmode.0 as u32, matrix_or_identity(worldanddpitransform), glyphorigins);
        Ok(())
    }
    #[cfg(windows)]
    fn ComputeGlyphOrigins2(&self, glyphrun: *const DWRITE_GLYPH_RUN, measuringmode: DWRITE_MEASURING_MODE, baselineorigin: &windows_numerics::Vector2, worldanddpitransform: *const DWRITE_MATRIX) -> WResult<windows_numerics::Vector2> {
        let xf = matrix_or_identity(worldanddpitransform);
        Ok(glyph_origins(glyphrun, *baselineorigin, measuringmode.0 as u32, xf).first().copied().unwrap_or(*baselineorigin))
    }
}

impl IDWriteFactory5_Impl for FactoryObj_Impl {
    fn CreateFontSetBuilder(&self) -> WResult<IDWriteFontSetBuilder1> {
        Ok(fontset::builder().cast()?)
    }
    fn CreateInMemoryFontFileLoader(&self) -> WResult<IDWriteInMemoryFontFileLoader> {
        Ok(fontfile::in_memory_loader())
    }
    fn CreateHttpFontFileLoader(&self, _referrerurl: &PCWSTR, _extraheaders: &PCWSTR) -> WResult<IDWriteRemoteFontFileLoader> {
        Ok(fontfile::remote_loader())
    }
    fn AnalyzeContainerType(&self, filedata: *const core::ffi::c_void, filedatasize: u32) -> DWRITE_CONTAINER_TYPE {
        if filedata.is_null() || filedatasize == 0 {
            return DWRITE_CONTAINER_TYPE_UNKNOWN;
        }
        fontfile::analyze_container(unsafe { core::slice::from_raw_parts(filedata as *const u8, filedatasize as usize) })
    }
    fn UnpackFontFile(&self, containertype: DWRITE_CONTAINER_TYPE, filedata: *const core::ffi::c_void, filedatasize: u32) -> WResult<IDWriteFontFileStream> {
        if filedata.is_null() {
            return Err(E_NOTIMPL.into());
        }
        fontfile::unpack(containertype, unsafe { core::slice::from_raw_parts(filedata as *const u8, filedatasize as usize) })
    }
}

impl IDWriteFactory6_Impl for FactoryObj_Impl {
    fn CreateFontFaceReference(&self, fontfile: Ref<IDWriteFontFile>, faceindex: u32, fontsimulations: DWRITE_FONT_SIMULATIONS, _fontaxisvalues: *const DWRITE_FONT_AXIS_VALUE, _fontaxisvaluecount: u32) -> WResult<IDWriteFontFaceReference1> {
        // ponytail: axis values are not instanced; the reference resolves to the face's own values.
        Ok(fontfile::faceref_from_file(fontfile.as_ref().ok_or(E_NOTIMPL)?, faceindex, fontsimulations)?.cast()?)
    }
    fn CreateFontResource(&self, fontfile: Ref<IDWriteFontFile>, faceindex: u32) -> WResult<IDWriteFontResource> {
        let file = fontfile.as_ref().ok_or(E_NOTIMPL)?;
        let bytes = fontfile::read_font_file(file)?;
        let source = file.cast_object_ref::<fontfile::FontFileObj>().map(|f| f.as_source()).unwrap_or_else(|_| fonts::FontSource::Bytes(bytes.clone()));
        Ok(fonts::FontResourceObj { bytes, index: faceindex, source }.into())
    }
    fn GetSystemFontSet(&self, _includedownloadablefonts: BOOL) -> WResult<IDWriteFontSet1> {
        Ok(fontset::font_set(self.fonts().face_refs()).cast()?)
    }
    fn GetSystemFontCollection(&self, _includedownloadablefonts: BOOL, fontfamilymodel: DWRITE_FONT_FAMILY_MODEL) -> WResult<IDWriteFontCollection2> {
        Ok(fonts::collection1(self.system_fonts(fontfamilymodel)).cast()?)
    }
    fn CreateFontCollectionFromFontSet(&self, fontset: Ref<IDWriteFontSet>, fontfamilymodel: DWRITE_FONT_FAMILY_MODEL) -> WResult<IDWriteFontCollection2> {
        let faces = fontset::set_faces(fontset.as_ref().ok_or(E_NOTIMPL)?).ok_or(E_NOTIMPL)?;
        let typographic = fontfamilymodel == DWRITE_FONT_FAMILY_MODEL_TYPOGRAPHIC;
        Ok(fonts::collection1(SystemFonts::from_faces_model(faces, typographic)).cast()?)
    }
    fn CreateFontSetBuilder(&self) -> WResult<IDWriteFontSetBuilder2> {
        Ok(fontset::builder().cast()?)
    }
    fn CreateTextFormat(&self, fontfamilyname: &PCWSTR, fontcollection: Ref<IDWriteFontCollection>, fontaxisvalues: *const DWRITE_FONT_AXIS_VALUE, fontaxisvaluecount: u32, fontsize: f32, localename: &PCWSTR) -> WResult<IDWriteTextFormat3> {
        if !(fontsize > 0.0) {
            return Err(windows_core::Error::from(windows_core::HRESULT(0x8007_0057u32 as i32)));
        }
        let family = wide_to_string(fontfamilyname.0, None);
        let locale = wide_to_string(localename.0, None);
        let fonts = match fontcollection.as_ref() {
            Some(collection) => collection.cast_object_ref::<fonts::CollectionObj>()?.fonts.clone(),
            None => self.fonts(),
        };
        // Weight/stretch/style default to normal; the axis values carry the variation request.
        let fmt3: IDWriteTextFormat3 = text::make_format(&fonts, &family, 400, 0, 5, fontsize, &locale)?.cast()?;
        if fontaxisvaluecount > 0 && !fontaxisvalues.is_null() {
            unsafe { fmt3.SetFontAxisValues(core::slice::from_raw_parts(fontaxisvalues, fontaxisvaluecount as usize))? };
        }
        Ok(fmt3)
    }
}

impl IDWriteFactory7_Impl for FactoryObj_Impl {
    fn GetSystemFontSet(&self, _includedownloadablefonts: BOOL) -> WResult<IDWriteFontSet2> {
        Ok(fontset::font_set(self.fonts().face_refs()).cast()?)
    }
    fn GetSystemFontCollection(&self, _includedownloadablefonts: BOOL, _fontfamilymodel: DWRITE_FONT_FAMILY_MODEL) -> WResult<IDWriteFontCollection3> {
        Ok(fonts::collection1(self.fonts()).cast()?)
    }
}

/// Every glyph origin of a run (`IDWriteFactory4::ComputeGlyphOrigins`): the pen walks the baseline
/// by each glyph's advance and each origin is the pen plus the glyph's offset. LTR advances after
/// placing; RTL (odd `bidiLevel`) ends each glyph's own advance at the pen, points the offset left and
/// then advances (`d2d::text::own_advances` in `measuring` mode under `xf`, the advance too when the run
/// has none); a sideways run advances along y. `DWRITE_GLYPH_OFFSET.ascenderOffset` points toward the
/// ascender (−y for horizontal).
fn glyph_origins(run: *const DWRITE_GLYPH_RUN, baseline: windows_numerics::Vector2, measuring: u32, xf: [f64; 6]) -> Vec<windows_numerics::Vector2> {
    use windows_numerics::Vector2;
    if run.is_null() {
        return Vec::new();
    }
    let run = unsafe { &*run };
    let n = run.glyphCount as usize;
    let adv = (!run.glyphAdvances.is_null()).then(|| unsafe { core::slice::from_raw_parts(run.glyphAdvances, n) });
    let off = (!run.glyphOffsets.is_null()).then(|| unsafe { core::slice::from_raw_parts(run.glyphOffsets, n) });
    let rtl = run.bidiLevel & 1 == 1;
    let sideways = run.isSideways.as_bool();
    let own = (!sideways && (rtl || adv.is_none())).then(|| own_advances(run, measuring, &xf)).flatten();
    let (mut px, mut py) = (baseline.X, baseline.Y);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let o = own.as_ref().map_or(0.0, |v| v[i]);
        let a = adv.map_or(o, |s| s[i]);
        let (ao, co) = off.map(|s| (s[i].advanceOffset, s[i].ascenderOffset)).unwrap_or((0.0, 0.0));
        let origin = if sideways {
            if rtl {
                py -= a;
            }
            let g = Vector2 { X: px - co, Y: py + ao };
            if !rtl {
                py += a;
            }
            g
        } else if rtl {
            let g = Vector2 { X: px - o - ao, Y: py - co };
            px -= a;
            g
        } else {
            let g = Vector2 { X: px + ao, Y: py - co };
            px += a;
            g
        };
        out.push(origin);
    }
    out
}

const IDENTITY: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// A caller's `DWRITE_MATRIX` as a transform, identity when absent.
fn matrix_or_identity(m: *const DWRITE_MATRIX) -> [f64; 6] {
    unsafe { m.as_ref() }.map_or(IDENTITY, |m| [m.m11 as f64, m.m12 as f64, m.m21 as f64, m.m22 as f64, m.dx as f64, m.dy as f64])
}

/// The run face's own glyph advances ([`d2d::text::own_advances`]); `None` for a face without our font bytes.
fn own_advances(run: &DWRITE_GLYPH_RUN, measuring: u32, xf: &[f64; 6]) -> Option<Vec<f32>> {
    use super::ITextInterop;
    let face = run.fontFace.as_ref()?;
    let interop: ITextInterop = face.cast().ok()?;
    let (mut data, mut len, mut index) = (core::ptr::null(), 0usize, 0u32);
    if unsafe { interop.FontBlob(&mut data, &mut len, &mut index) }.is_err() || data.is_null() || run.glyphIndices.is_null() {
        return None;
    }
    let ids = unsafe { core::slice::from_raw_parts(run.glyphIndices, run.glyphCount as usize) };
    let glyphs: Vec<d2d::text::Glyph> = ids.iter().map(|&id| d2d::text::Glyph { id, advance: 0.0, offset: [0.0, 0.0] }).collect();
    let core = d2d::text::GlyphRun {
        font_data: unsafe { core::slice::from_raw_parts(data, len) },
        face_index: index,
        em_size: run.fontEmSize,
        glyphs: &glyphs,
        baseline_origin: [0.0, 0.0],
        measuring_mode: measuring,
        is_sideways: false,
        bidi_level: run.bidiLevel,
        no_snap: true,
        simulations: unsafe { face.GetSimulations() }.0 as u32,
        coords: &[],
    };
    Some(d2d::text::own_advances(&core, xf))
}

/// Fill a caller `*mut Vector2` buffer (sized to the run's `glyphCount`) with [`glyph_origins`].
#[cfg(not(windows))]
fn write_glyph_origins(run: *const DWRITE_GLYPH_RUN, baseline: windows_numerics::Vector2, measuring: u32, xf: [f64; 6], out: *mut windows_numerics::Vector2) {
    if out.is_null() {
        return;
    }
    let origins = glyph_origins(run, baseline, measuring, xf);
    if !origins.is_empty() {
        unsafe { core::slice::from_raw_parts_mut(out, origins.len()) }.copy_from_slice(&origins);
    }
}

// --- IDWriteRenderingParams (up to 3) ----------------------------------------------------------

#[implement(IDWriteRenderingParams3)]
pub(crate) struct ParamsObj {
    gamma: f32,
    enhanced: f32,
    grayscale: f32,
    cleartype: f32,
    pixel_geom: i32,
    mode: i32,
    gridfit: i32,
}

impl ParamsObj {
    fn new(gamma: f32, enhanced: f32, grayscale: f32, cleartype: f32, pixel_geom: i32, mode: i32, gridfit: i32) -> Self {
        Self { gamma, enhanced, grayscale, cleartype, pixel_geom, mode, gridfit }
    }
}

impl IDWriteRenderingParams_Impl for ParamsObj_Impl {
    fn GetGamma(&self) -> f32 {
        self.gamma
    }
    fn GetEnhancedContrast(&self) -> f32 {
        self.enhanced
    }
    fn GetClearTypeLevel(&self) -> f32 {
        self.cleartype
    }
    fn GetPixelGeometry(&self) -> DWRITE_PIXEL_GEOMETRY {
        DWRITE_PIXEL_GEOMETRY(self.pixel_geom)
    }
    fn GetRenderingMode(&self) -> DWRITE_RENDERING_MODE {
        // The legacy enum has no NATURAL_SYMMETRIC_DOWNSAMPLED (7); it reads as NATURAL_SYMMETRIC.
        DWRITE_RENDERING_MODE(if self.mode == 7 { 5 } else { self.mode })
    }
}

impl IDWriteRenderingParams1_Impl for ParamsObj_Impl {
    fn GetGrayscaleEnhancedContrast(&self) -> f32 {
        self.grayscale
    }
}

impl IDWriteRenderingParams2_Impl for ParamsObj_Impl {
    fn GetGridFitMode(&self) -> DWRITE_GRID_FIT_MODE {
        DWRITE_GRID_FIT_MODE(self.gridfit)
    }
}

impl IDWriteRenderingParams3_Impl for ParamsObj_Impl {
    fn GetRenderingMode1(&self) -> DWRITE_RENDERING_MODE1 {
        DWRITE_RENDERING_MODE1(self.mode)
    }
}

#[cfg(test)]
mod factory_tests {
    //! `IDWriteFactory4-7` QI chain + the container/loader/glyph-origin methods. Self-contained: no
    //! system font corpus (the system collection may be empty here), no Wine/VM.
    use super::*;
    use windows_core::Interface;

    fn factory7() -> IDWriteFactory7 {
        FactoryObj::new().into()
    }

    /// A WOFF 1.0 container wrapping one uncompressed table (compLength == origLength).
    fn woff_one_table(flavor: u32, tag: &[u8; 4], table: &[u8]) -> Vec<u8> {
        let mut w = Vec::new();
        w.extend_from_slice(b"wOFF");
        w.extend_from_slice(&flavor.to_be_bytes());
        w.extend_from_slice(&0u32.to_be_bytes()); // length (unchecked)
        w.extend_from_slice(&1u16.to_be_bytes()); // numTables
        w.extend_from_slice(&0u16.to_be_bytes()); // reserved
        w.extend_from_slice(&0u32.to_be_bytes()); // totalSfntSize
        w.extend_from_slice(&[0u8; 24]); // majorVer+minorVer + meta/priv offsets+lengths (44-byte header total)
        let table_off = 44 + 20u32;
        w.extend_from_slice(tag);
        w.extend_from_slice(&table_off.to_be_bytes());
        w.extend_from_slice(&(table.len() as u32).to_be_bytes()); // compLength
        w.extend_from_slice(&(table.len() as u32).to_be_bytes()); // origLength
        w.extend_from_slice(&0u32.to_be_bytes()); // origChecksum
        w.extend_from_slice(table);
        w
    }

    #[test]
    fn factory_qi_chain_to_7() {
        let base: IDWriteFactory = {
            let f: IDWriteFactory7 = FactoryObj::new().into();
            f.cast().unwrap()
        };
        // An app gets the base factory then QIs up — every leaf must resolve.
        let _f4: IDWriteFactory4 = base.cast().unwrap();
        let _f5: IDWriteFactory5 = base.cast().unwrap();
        let _f6: IDWriteFactory6 = base.cast().unwrap();
        let _f7: IDWriteFactory7 = base.cast().unwrap();
    }

    #[test]
    fn analyze_container_type() {
        let f = factory7();
        let woff = b"wOFF\0\0\0\0";
        let woff2 = b"wOF2\0\0\0\0";
        let sfnt = [0x00u8, 0x01, 0x00, 0x00, 0, 0, 0, 0];
        unsafe {
            assert_eq!(f.AnalyzeContainerType(woff.as_ptr() as *const _, woff.len() as u32), DWRITE_CONTAINER_TYPE_WOFF);
            assert_eq!(f.AnalyzeContainerType(woff2.as_ptr() as *const _, woff2.len() as u32), DWRITE_CONTAINER_TYPE_WOFF2);
            assert_eq!(f.AnalyzeContainerType(sfnt.as_ptr() as *const _, sfnt.len() as u32), DWRITE_CONTAINER_TYPE_UNKNOWN);
        }
    }

    #[test]
    fn unpack_woff_to_sfnt() {
        let f = factory7();
        let table = b"CFF-table-bytes!"; // 16 bytes, already 4-aligned
        let woff = woff_one_table(0x4F54544F, b"CFF ", table); // flavor 'OTTO'
        unsafe {
            let stream = f.UnpackFontFile(DWRITE_CONTAINER_TYPE_WOFF, woff.as_ptr() as *const _, woff.len() as u32).unwrap();
            let size = stream.GetFileSize().unwrap() as usize;
            let (mut frag, mut ctx) = (core::ptr::null_mut(), core::ptr::null_mut());
            stream.ReadFileFragment(&mut frag, 0, size as u64, &mut ctx).unwrap();
            let out = core::slice::from_raw_parts(frag as *const u8, size);
            assert_eq!(&out[0..4], &0x4F54544Fu32.to_be_bytes(), "sfnt flavor preserved");
            assert_eq!(u16::from_be_bytes([out[4], out[5]]), 1, "one table");
            // Table data lands after the 12-byte offset table + one 16-byte directory entry.
            assert_eq!(&out[28..28 + table.len()], table, "table bytes recovered");
            stream.ReleaseFileFragment(ctx);
            // WOFF2 is not decoded — a real error, never a stub.
            assert!(f.UnpackFontFile(DWRITE_CONTAINER_TYPE_WOFF2, woff.as_ptr() as *const _, woff.len() as u32).is_err());
        }
    }

    #[test]
    fn in_memory_loader_roundtrips() {
        let f = factory7();
        let bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../d2d/src/text/color/testdata/colrtest.ttf")).unwrap();
        unsafe {
            let loader = f.CreateInMemoryFontFileLoader().unwrap();
            assert_eq!(loader.GetFileCount(), 0);
            let file = loader
                .CreateInMemoryFontFileReference(None::<&IDWriteFactory>, bytes.as_ptr() as *const _, bytes.len() as u32, None::<&windows_core::IUnknown>)
                .unwrap();
            assert_eq!(loader.GetFileCount(), 1);
            let face = f.CreateFontFace(DWRITE_FONT_FACE_TYPE_TRUETYPE, &[Some(file)], 0, DWRITE_FONT_SIMULATIONS_NONE).unwrap();
            assert!(face.GetGlyphCount() > 0, "the in-memory reference reads back as a usable face");
        }
    }

    // Off-Windows (the shim binding) the full per-glyph origin array is filled; see the cfg note on
    // the impl for the on-Windows windows-rs limitation.
    #[cfg(not(windows))]
    #[test]
    fn compute_glyph_origins_fills_all() {
        let f = factory7();
        let baseline = windows_numerics::Vector2 { X: 10.0, Y: 20.0 };
        let ids = [1u16, 2, 3];
        let advances = [10.0f32, 20.0, 30.0];
        let run = DWRITE_GLYPH_RUN {
            fontFace: core::mem::ManuallyDrop::new(None),
            fontEmSize: 16.0,
            glyphCount: 3,
            glyphIndices: ids.as_ptr(),
            glyphAdvances: advances.as_ptr(),
            glyphOffsets: core::ptr::null(),
            isSideways: BOOL(0),
            bidiLevel: 0,
        };
        let mut origins = [windows_numerics::Vector2 { X: 0.0, Y: 0.0 }; 3];
        unsafe {
            f.ComputeGlyphOrigins(&run, baseline, &mut origins).unwrap();
        }
        // LTR: origin[i] = baseline.x + sum(advances[0..i]); every y stays on the baseline.
        assert_eq!(origins.map(|o| o.X), [10.0, 20.0, 40.0]);
        assert!(origins.iter().all(|o| o.Y == 20.0));
        let _ = core::mem::ManuallyDrop::into_inner(run.fontFace);
    }

    #[test]
    fn system_collection_family_model() {
        let f = factory7();
        unsafe {
            let c2 = f.GetSystemFontCollection(false, DWRITE_FONT_FAMILY_MODEL_WEIGHT_STRETCH_STYLE).unwrap();
            assert_eq!(c2.GetFontFamilyModel(), DWRITE_FONT_FAMILY_MODEL_WEIGHT_STRETCH_STYLE);
            let _c3: IDWriteFontCollection3 = c2.cast().unwrap();
        }
    }
}
