//! Probe for the dwrite.dll COM surface that `dwrite-dump` does not cover: it drives whatever
//! `dwrite.dll` the loader binds (native on the VM, or our drop-in under Wine with `dwrite=n`) and
//! prints one JSON object. `probe.py` runs it both ways and diffs, so every method is verified
//! byte-for-byte against native.
//!
//!   dwrite-probe <font-file> <text>
//!
//! Covers: CreateFontFileReference + Analyze, CreateFontFace (file) + GetGlyphIndices, the system
//! collection's GetMatchingFonts, and CreateTextAnalyzer's GetGlyphs / GetGlyphPlacements.

#[cfg(not(windows))]
fn main() {
    eprintln!("dwrite-probe requires Windows; run under Wine or on the VM");
    std::process::exit(2);
}

#[cfg(windows)]
fn main() -> windows_core::Result<()> {
    use windows::Win32::Graphics::DirectWrite::*;
    use windows::core::{BOOL, HSTRING, Interface};

    let args: Vec<String> = std::env::args().collect();
    // `shape` mode: full glyph shaping of one family for the layout agent (Arabic joining etc.).
    //   dwrite-probe shape <family> <text> [locale]
    if args.get(1).map(String::as_str) == Some("shape") {
        if args.len() < 4 {
            eprintln!("usage: dwrite-probe shape <family> <text> [locale]");
            std::process::exit(2);
        }
        let locale = args.get(4).cloned().unwrap_or_else(|| "en-us".into());
        return shape_mode(&args[2], &args[3], &locale);
    }
    if args.len() != 3 {
        eprintln!("usage: dwrite-probe <font-file> <text>  |  dwrite-probe shape <family> <text> [locale]");
        std::process::exit(2);
    }
    let path = HSTRING::from(&args[1]);
    let text: Vec<u16> = args[2].encode_utf16().collect();

    unsafe {
        let factory: IDWriteFactory = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED)?;

        // --- CreateFontFileReference + Analyze -------------------------------------------------
        let file = factory.CreateFontFileReference(&path, None)?;
        let (mut supported, mut filetype, mut facetype, mut numfaces) = (BOOL(0), DWRITE_FONT_FILE_TYPE(0), DWRITE_FONT_FACE_TYPE(0), 0u32);
        file.Analyze(&mut supported, &mut filetype, Some(&mut facetype), &mut numfaces)?;

        // --- CreateFontFace from the file, then cmap glyph ids --------------------------------
        let files = [Some(file.clone())];
        let face = factory.CreateFontFace(DWRITE_FONT_FACE_TYPE_TRUETYPE, &files, 0, DWRITE_FONT_SIMULATIONS_NONE)?;
        let codepoints: Vec<u32> = args[2].chars().map(|c| c as u32).collect();
        let mut ids = vec![0u16; codepoints.len()];
        face.GetGlyphIndices(codepoints.as_ptr(), codepoints.len() as u32, ids.as_mut_ptr())?;

        // --- GetMatchingFonts on the system collection ----------------------------------------
        let mut collection: Option<IDWriteFontCollection> = None;
        factory.GetSystemFontCollection(&mut collection, false)?;
        let collection = collection.unwrap();
        let (mut fidx, mut fexists) = (0u32, BOOL(0));
        collection.FindFamilyName(&HSTRING::from("Arial"), &mut fidx, &mut fexists)?;
        let match_count = if fexists.as_bool() {
            let family = collection.GetFontFamily(fidx)?;
            family.GetMatchingFonts(DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STRETCH_NORMAL, DWRITE_FONT_STYLE_NORMAL)?.GetFontCount()
        } else {
            0
        };

        // --- CreateTextAnalyzer: GetGlyphs + GetGlyphPlacements --------------------------------
        let analyzer = factory.CreateTextAnalyzer()?;
        let (ana_glyphs, ana_advances) = analyze_run(&analyzer, &face, &text);
        let (gdi_advances, line_breaks, numsub_calls) = analyze_extra(&analyzer, &face, &text);

        // --- CreateRenderingParams: the system defaults ---------------------------------------
        let rp = factory.CreateRenderingParams().ok();
        let (rg, rec, rgec, rctl, rpg, rrm) = match &rp {
            Some(p) => {
                let gec = p.cast::<IDWriteRenderingParams1>().map(|p1| p1.GetGrayscaleEnhancedContrast()).unwrap_or(-1.0);
                (p.GetGamma(), p.GetEnhancedContrast(), gec, p.GetClearTypeLevel(), p.GetPixelGeometry().0, p.GetRenderingMode().0)
            }
            None => (-1.0, -1.0, -1.0, -1.0, -1, -1),
        };

        // --- Font fallback: system + a built mapping ------------------------------------------
        let factory2: IDWriteFactory2 = factory.cast()?;
        let sys_fb = factory2.GetSystemFontFallback()?;
        let (sys_len, sys_fam) = fallback_map(&sys_fb, &text, Some(&collection));
        let builder = factory2.CreateFontFallbackBuilder()?;
        let ranges = [DWRITE_UNICODE_RANGE { first: 0, last: 0x10FFFF }];
        let fams = [HSTRING::from("Arial")];
        let famptrs = [fams[0].as_ptr()];
        builder.AddMapping(&ranges, &famptrs, None, &HSTRING::from(""), &HSTRING::from(""), 1.0)?;
        let built_fb = builder.CreateFontFallback()?;
        let (built_len, built_fam) = fallback_map(&built_fb, &text, Some(&collection));
        // IDWriteFontFallbackBuilder::AddMappings: merge the built fallback's mappings into a fresh
        // builder; the merged fallback maps identically.
        let builder2 = factory2.CreateFontFallbackBuilder()?;
        builder2.AddMappings(&built_fb)?;
        let merged_fb = builder2.CreateFontFallback()?;
        let (merged_len, merged_fam) = fallback_map(&merged_fb, &text, Some(&collection));

        // --- Font download queue + a built font set's properties ------------------------------
        let factory3: IDWriteFactory3 = factory.cast()?;
        let dq = factory3.GetFontDownloadQueue()?;
        let (dq_empty, dq_gen) = (dq.IsEmpty().as_bool(), dq.GetGenerationCount());
        // AddListener/RemoveListener: register a no-op listener, then remove by its token.
        let listener: IDWriteFontDownloadListener = ProbeDownloadListener.into();
        let dq_token = dq.AddListener(&listener).unwrap_or(0xFFFF_FFFF);
        let dq_removed = dq.RemoveListener(dq_token).is_ok();
        let setb = factory3.CreateFontSetBuilder()?;
        let faceref = factory3.CreateFontFaceReference2(&path, None, 0, DWRITE_FONT_SIMULATIONS_NONE)?;
        setb.AddFontFaceReference2(&faceref)?;
        let set = setb.CreateFontSet()?;
        let fs_count = set.GetFontCount();
        let fams = set.GetPropertyValues(DWRITE_FONT_PROPERTY_ID_FAMILY_NAME)?;
        let (fs_fam_count, fs_fam0) = (fams.GetCount(), {
            let mut b = [0u16; 64];
            let _ = fams.GetString(0, &mut b);
            String::from_utf16_lossy(&b).trim_end_matches('\0').to_string()
        });
        // CreateFontCollectionFromFontSet → the IDWriteFontCollection1/FontFamily1/Font3 cascade.
        let coll1 = factory3.CreateFontCollectionFromFontSet(&set)?;
        let coll_count = coll1.GetFontFamilyCount();
        let fam1 = IDWriteFontCollection1::GetFontFamily(&coll1, 0)?;
        let font3 = fam1.GetFont(0)?;
        let coll_fam = {
            let names = font3.GetFontFamily()?.GetFamilyNames()?;
            let mut b = [0u16; 64];
            let _ = names.GetString(0, &mut b);
            String::from_utf16_lossy(&b).trim_end_matches('\0').to_string()
        };

        // --- TranslateColorGlyphRun: a COLRv0 emoji (U+1F600) → coloured sub-runs -------------
        let color_run = translate_color(&collection, &factory.cast()?);

        // --- GetGdiInterop: ConvertFontFaceToLOGFONT + a bitmap render target -----------------
        let gdi = factory.GetGdiInterop()?;
        let mut lf = windows::Win32::Graphics::Gdi::LOGFONTW::default();
        gdi.ConvertFontFaceToLOGFONT(&face, &mut lf)?;
        let lf_name = {
            let end = lf.lfFaceName.iter().position(|&c| c == 0).unwrap_or(32);
            String::from_utf16_lossy(&lf.lfFaceName[..end])
        };
        let brt = gdi.CreateBitmapRenderTarget(None, 48, 24)?;
        let sz = brt.GetSize()?;
        let ppd = brt.GetPixelsPerDip();
        // The BRT's default text AA mode before any SetTextAntialiasMode (0=ClearType, 1=grayscale).
        let brt_aa = brt.cast::<IDWriteBitmapRenderTarget1>().map(|b| b.GetTextAntialiasMode().0).unwrap_or(-1);
        // DrawGlyphRun 'H' (white, em 16) into the DIB, read it via GetBitmapData, and emit the full
        // signature: brt_sig ([nz,sum,first6]), brt_px (every nonzero pixel), brt_hist (distinct grey
        // levels + counts) and brt_params (the params actually passed to DrawGlyphRun).
        let (brt_sig, brt_px, brt_hist, brt_params) = {
            use core::mem::ManuallyDrop;
            use windows::Win32::Foundation::COLORREF;
            let hid = [ids[0]];
            let hadv = [11.554688f32];
            let hoff = [DWRITE_GLYPH_OFFSET::default()];
            let hrun = DWRITE_GLYPH_RUN { fontFace: ManuallyDrop::new(Some(face.clone())), fontEmSize: 16.0, glyphCount: 1, glyphIndices: hid.as_ptr(), glyphAdvances: hadv.as_ptr(), glyphOffsets: hoff.as_ptr(), isSideways: BOOL(0), bidiLevel: 0 };
            let rp = factory.CreateCustomRenderingParams(2.2, 0.0, 1.0, DWRITE_PIXEL_GEOMETRY_FLAT, DWRITE_RENDERING_MODE_NATURAL)?;
            let r = brt.DrawGlyphRun(2.0, 14.0, DWRITE_MEASURING_MODE_NATURAL, &hrun, &rp, COLORREF(0x00FFFFFF), None);
            let _ = ManuallyDrop::into_inner(hrun.fontFace);
            r?;
            let rt2: IDWriteBitmapRenderTarget2 = brt.cast()?;
            let bd = rt2.GetBitmapData()?;
            let w = bd.width as usize;
            let px = core::slice::from_raw_parts(bd.pixels, (bd.width * bd.height) as usize);
            let nz = px.iter().filter(|&&p| p & 0x00FFFFFF != 0).count();
            let sum: u64 = px.iter().flat_map(|p| p.to_le_bytes()).map(|b| b as u64).sum();
            let first: Vec<String> = px.iter().filter(|&&p| p & 0xFFFFFF != 0).take(6).map(|p| format!("\"{:06x}\"", p & 0xFFFFFF)).collect();
            let sig = format!("[{nz},{sum},[{}]]", first.join(","));
            let pxs: Vec<String> = px.iter().enumerate().filter(|&(_, &p)| p & 0xFFFFFF != 0).map(|(i, &p)| format!("[{},{},\"{:06x}\"]", i % w, i / w, p & 0xFFFFFF)).collect();
            // Distinct low-byte grey levels (R=G=B for grayscale) with their pixel counts, sorted.
            let mut greys: Vec<u32> = px.iter().filter(|&&p| p & 0xFFFFFF != 0).map(|&p| p & 0xFF).collect();
            greys.sort_unstable();
            let mut hist: Vec<(u32, usize)> = Vec::new();
            for g in &greys {
                match hist.last_mut() {
                    Some(e) if e.0 == *g => e.1 += 1,
                    _ => hist.push((*g, 1)),
                }
            }
            let hists: Vec<String> = hist.iter().map(|(g, c)| format!("[{g},{c}]")).collect();
            let params = format!("[{:.6},{:.6},{:.6},{},{}]", rp.GetGamma(), rp.GetEnhancedContrast(), rp.GetClearTypeLevel(), rp.GetPixelGeometry().0, rp.GetRenderingMode().0);
            (sig, format!("[{}]", pxs.join(",")), format!("[{}]", hists.join(",")), params)
        };
        // Same 'H' with EXPLICIT grayscale AA (SetTextAntialiasMode GRAYSCALE) into a fresh DIB — native's
        // 8×1 grayscale curve, distinct from the default (ClearType). Emit its distinct-grey histogram.
        let brt_gray_hist = {
            use core::mem::ManuallyDrop;
            use windows::Win32::Foundation::COLORREF;
            let g = gdi.CreateBitmapRenderTarget(None, 48, 24)?;
            g.cast::<IDWriteBitmapRenderTarget1>()?.SetTextAntialiasMode(DWRITE_TEXT_ANTIALIAS_MODE_GRAYSCALE)?;
            let hid = [ids[0]];
            let hadv = [11.554688f32];
            let hoff = [DWRITE_GLYPH_OFFSET::default()];
            let hrun = DWRITE_GLYPH_RUN { fontFace: ManuallyDrop::new(Some(face.clone())), fontEmSize: 16.0, glyphCount: 1, glyphIndices: hid.as_ptr(), glyphAdvances: hadv.as_ptr(), glyphOffsets: hoff.as_ptr(), isSideways: BOOL(0), bidiLevel: 0 };
            let rp = factory.CreateCustomRenderingParams(2.2, 0.0, 1.0, DWRITE_PIXEL_GEOMETRY_FLAT, DWRITE_RENDERING_MODE_NATURAL)?;
            let r = g.DrawGlyphRun(2.0, 14.0, DWRITE_MEASURING_MODE_NATURAL, &hrun, &rp, COLORREF(0x00FFFFFF), None);
            let _ = ManuallyDrop::into_inner(hrun.fontFace);
            r?;
            let bd = g.cast::<IDWriteBitmapRenderTarget2>()?.GetBitmapData()?;
            let px = core::slice::from_raw_parts(bd.pixels, (bd.width * bd.height) as usize);
            let mut greys: Vec<u32> = px.iter().filter(|&&p| p & 0xFFFFFF != 0).map(|&p| p & 0xFF).collect();
            greys.sort_unstable();
            let mut hist: Vec<(u32, usize)> = Vec::new();
            for gy in &greys {
                match hist.last_mut() {
                    Some(e) if e.0 == *gy => e.1 += 1,
                    _ => hist.push((*gy, 1)),
                }
            }
            format!("[{}]", hist.iter().map(|(gy, c)| format!("[{gy},{c}]")).collect::<Vec<_>>().join(","))
        };

        // --- CreateEllipsisTrimmingSign + a trimmed layout -----------------------------------
        let tfmt = factory.CreateTextFormat(&HSTRING::from("Arial"), None, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 20.0, &HSTRING::from("en-us"))?;
        tfmt.SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP)?;
        let sign = factory.CreateEllipsisTrimmingSign(&tfmt)?;
        let em = sign.GetMetrics()?;
        let tlw: Vec<u16> = "Hello World Wide".encode_utf16().collect();
        let tl = factory.CreateTextLayout(&tlw, &tfmt, 60.0, 1000.0)?;
        let trimming = DWRITE_TRIMMING { granularity: DWRITE_TRIMMING_GRANULARITY_CHARACTER, delimiter: 0, delimiterCount: 0 };
        tl.SetTrimming(&trimming, &sign)?;
        let mut ttm = DWRITE_TEXT_METRICS::default();
        tl.GetMetrics(&mut ttm)?;
        let mut lm = [DWRITE_LINE_METRICS::default(); 1];
        let mut lc = 0u32;
        let _ = tl.GetLineMetrics(Some(&mut lm), &mut lc);

        // --- CreateTextLayout: DetermineMinWidth + GetMetrics ---------------------------------
        let lw: Vec<u16> = "Hello World Wide".encode_utf16().collect();
        let format = factory.CreateTextFormat(&HSTRING::from("Arial"), None, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 20.0, &HSTRING::from("en-us"))?;
        let layout = factory.CreateTextLayout(&lw, &format, 1000.0, 1000.0)?;
        let min_width = layout.DetermineMinWidth()?;
        let mut tm = DWRITE_TEXT_METRICS::default();
        layout.GetMetrics(&mut tm)?;
        let oh = layout.GetOverhangMetrics()?;

        // --- IDWriteFontFace5 axis values (static-font synthesis oracle) ----------------------
        let axis_s = {
            let f5: IDWriteFontFace5 = face.cast()?;
            let n = f5.GetFontAxisValueCount();
            let mut av = vec![DWRITE_FONT_AXIS_VALUE::default(); n as usize];
            if n > 0 {
                f5.GetFontAxisValues(&mut av)?;
            }
            let items: Vec<String> = av.iter().map(|a| format!("[{},{:.6}]", a.axisTag.0, a.value)).collect();
            format!("{{\"has_variations\":{},\"count\":{},\"values\":[{}]}}", f5.HasVariations().as_bool(), n, items.join(","))
        };

        let ids_s = ids.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(",");
        let ag_s = ana_glyphs.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(",");
        let aa_s = ana_advances.iter().map(|a| format!("{a:.6}")).collect::<Vec<_>>().join(",");
        let gdi_s = gdi_advances.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(",");
        let lb_s = line_breaks.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        println!(
            "{{\"analyze\":{{\"supported\":{},\"file_type\":{},\"face_type\":{},\"num_faces\":{}}},\"cmap_ids\":[{}],\"match_count\":{},\"analyzer_glyphs\":[{}],\"analyzer_advances\":[{}],\"rparams\":{{\"gamma\":{:.6},\"enhanced\":{:.6},\"grayscale\":{:.6},\"cleartype\":{:.6},\"pixel_geometry\":{},\"rendering_mode\":{}}},\"min_width\":{:.6},\"metrics_width\":{:.6},\"fallback_sys\":[{},\"{}\"],\"fallback_built\":[{},\"{}\"],\"overhang\":[{:.6},{:.6},{:.6},{:.6}],\"dq\":[{},{}],\"fontset\":[{},{},\"{}\"],\"coll_from_set\":[{},\"{}\"],\"color_run\":{},\"logfont\":[\"{}\",{},{},{}],\"brt\":[{},{},{:.4}],\"ellipsis\":[{:.4},{:.4},{:.4}],\"trim\":[{},{},{},{:.4}],\"brt_sig\":{},\"brt_params\":{},\"brt_aa\":{},\"brt_hist\":{},\"brt_gray_hist\":{},\"brt_px\":{},\"axis\":{},\"gdi_advances\":[{}],\"line_breaks\":[{}],\"numsub_calls\":{},\"dq_listener\":[{},{}],\"fallback_merged\":[{},\"{}\"]}}",
            supported.as_bool(), filetype.0, facetype.0, numfaces, ids_s, match_count, ag_s, aa_s, rg, rec, rgec, rctl, rpg, rrm, min_width, tm.width, sys_len, sys_fam, built_len, built_fam, oh.left, oh.top, oh.right, oh.bottom, dq_empty, dq_gen, fs_count, fs_fam_count, fs_fam0, coll_count, coll_fam, color_run, lf_name, lf.lfWeight, lf.lfItalic, lf.lfCharSet.0, sz.cx, sz.cy, ppd, em.width, em.height, em.baseline, lc, lm[0].length, lm[0].isTrimmed.as_bool(), ttm.width, brt_sig, brt_params, brt_aa, brt_hist, brt_gray_hist, brt_px, axis_s, gdi_s, lb_s, numsub_calls, dq_token, dq_removed, merged_len, merged_fam
        );
    }
    Ok(())
}

/// `shape` mode: full glyph shaping of one system family, the way DirectWrite's layout does it —
/// AnalyzeScript + AnalyzeBidi, then GetGlyphs with that `DWRITE_SCRIPT_ANALYSIS` + `isRightToLeft`
/// (from the resolved bidi level) + locale + default features, then GetGlyphPlacements. Prints one
/// JSON object with the glyph ids, advances (em 16), offsets, cluster map and the shaping text/glyph
/// property bitfields — the native target the layout agent needs (e.g. Arabic joining for "مرحبا").
#[cfg(windows)]
fn shape_mode(family: &str, text_str: &str, locale: &str) -> windows_core::Result<()> {
    use windows::Win32::Graphics::DirectWrite::*;
    use windows::core::{BOOL, HSTRING, PCWSTR};
    let text: Vec<u16> = text_str.encode_utf16().collect();
    unsafe {
        let factory: IDWriteFactory = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED)?;
        let mut collection: Option<IDWriteFontCollection> = None;
        factory.GetSystemFontCollection(&mut collection, false)?;
        let collection = collection.unwrap();
        let (mut idx, mut exists) = (0u32, BOOL(0));
        collection.FindFamilyName(&HSTRING::from(family), &mut idx, &mut exists)?;
        if !exists.as_bool() {
            eprintln!("family not found: {family}");
            std::process::exit(3);
        }
        let face = collection.GetFontFamily(idx)?.GetFirstMatchingFont(DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STRETCH_NORMAL, DWRITE_FONT_STYLE_NORMAL)?.CreateFontFace()?;

        let analyzer = factory.CreateTextAnalyzer()?;
        let src: IDWriteTextAnalysisSource = ProbeSource { text: text.clone() }.into();
        let sink: IDWriteTextAnalysisSink = ProbeSink::default().into();
        analyzer.AnalyzeScript(&src, 0, text.len() as u32, &sink)?;
        analyzer.AnalyzeBidi(&src, 0, text.len() as u32, &sink)?;
        let script = ProbeSink::take(&sink);
        let rtl = ProbeSink::bidi(&sink) & 1 == 1;
        let loc = HSTRING::from(locale);

        let maxg = text.len() * 3 + 16;
        let mut clusters = vec![0u16; text.len()];
        let mut tprops = vec![DWRITE_SHAPING_TEXT_PROPERTIES::default(); text.len()];
        let mut glyphs = vec![0u16; maxg];
        let mut gprops = vec![DWRITE_SHAPING_GLYPH_PROPERTIES::default(); maxg];
        let mut actual = 0u32;
        analyzer.GetGlyphs(PCWSTR(text.as_ptr()), text.len() as u32, &face, false, rtl, &script, &loc, None, None, None, 0, maxg as u32, clusters.as_mut_ptr(), tprops.as_mut_ptr(), glyphs.as_mut_ptr(), gprops.as_mut_ptr(), &mut actual)?;
        glyphs.truncate(actual as usize);
        gprops.truncate(actual as usize);
        let mut advances = vec![0f32; actual as usize];
        let mut offsets = vec![DWRITE_GLYPH_OFFSET::default(); actual as usize];
        analyzer.GetGlyphPlacements(PCWSTR(text.as_ptr()), clusters.as_ptr(), tprops.as_mut_ptr(), text.len() as u32, glyphs.as_ptr(), gprops.as_ptr(), actual, &face, 16.0, false, rtl, &script, &loc, None, None, 0, advances.as_mut_ptr(), offsets.as_mut_ptr())?;

        let join = |v: Vec<String>| v.join(",");
        println!(
            "{{\"script\":{},\"shapes\":{},\"rtl\":{},\"glyphs\":[{}],\"advances\":[{}],\"offsets\":[{}],\"clusters\":[{}],\"text_props\":[{}],\"glyph_props\":[{}]}}",
            script.script,
            script.shapes.0,
            rtl,
            join(glyphs.iter().map(|x| x.to_string()).collect()),
            join(advances.iter().map(|x| format!("{x:.4}")).collect()),
            join(offsets.iter().map(|o| format!("[{:.4},{:.4}]", o.advanceOffset, o.ascenderOffset)).collect()),
            join(clusters.iter().map(|x| x.to_string()).collect()),
            join(tprops.iter().map(|p| p._bitfield.to_string()).collect()),
            join(gprops.iter().map(|p| p._bitfield.to_string()).collect()),
        );
    }
    Ok(())
}

/// `IDWriteFactory2::TranslateColorGlyphRun` on the COLRv0 emoji U+1F600 from "Segoe UI Emoji":
/// returns a JSON array `[subRunCount, [[glyph, paletteIndex, r,g,b,a], ...]]`. `[]` if the family
/// is unavailable or the call fails.
#[cfg(windows)]
fn translate_color(collection: &windows::Win32::Graphics::DirectWrite::IDWriteFontCollection, factory2: &windows::Win32::Graphics::DirectWrite::IDWriteFactory2) -> String {
    use core::mem::ManuallyDrop;
    use windows::Win32::Graphics::DirectWrite::*;
    use windows::core::{BOOL, HSTRING};
    let run = || -> windows_core::Result<String> {
        let (mut idx, mut exists) = (0u32, BOOL(0));
        unsafe { collection.FindFamilyName(&HSTRING::from("Segoe UI Emoji"), &mut idx, &mut exists)? };
        if !exists.as_bool() {
            return Ok("[]".into());
        }
        let face = unsafe { collection.GetFontFamily(idx)?.GetFirstMatchingFont(DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STRETCH_NORMAL, DWRITE_FONT_STYLE_NORMAL)?.CreateFontFace()? };
        let cp = [0x1F600u32];
        let mut ids = [0u16; 1];
        unsafe { face.GetGlyphIndices(cp.as_ptr(), 1, ids.as_mut_ptr())? };
        let advances = [0.0f32];
        let offsets = [DWRITE_GLYPH_OFFSET::default()];
        let gr = DWRITE_GLYPH_RUN {
            fontFace: ManuallyDrop::new(Some(face.clone())),
            fontEmSize: 32.0,
            glyphCount: 1,
            glyphIndices: ids.as_ptr(),
            glyphAdvances: advances.as_ptr(),
            glyphOffsets: offsets.as_ptr(),
            isSideways: BOOL(0),
            bidiLevel: 0,
        };
        let en = unsafe { factory2.TranslateColorGlyphRun(0.0, 0.0, &gr, None, DWRITE_MEASURING_MODE_NATURAL, None, 0) };
        let _ = ManuallyDrop::into_inner(gr.fontFace);
        let en = en?;
        let mut subs: Vec<String> = Vec::new();
        while unsafe { en.MoveNext()? }.as_bool() {
            let r = unsafe { &*en.GetCurrentRun()? };
            let g = unsafe { *r.glyphRun.glyphIndices };
            let c = r.runColor;
            subs.push(format!("[{},{},{:.4},{:.4},{:.4},{:.4}]", g, r.paletteIndex, c.r, c.g, c.b, c.a));
        }
        Ok(format!("[{},[{}]]", subs.len(), subs.join(",")))
    };
    run().unwrap_or_else(|_| "[]".into())
}

/// `IDWriteFontFallback::MapCharacters` over the whole text with base family "Arial": the first
/// run's mapped length and the mapped font's family name (empty if unmapped).
#[cfg(windows)]
fn fallback_map(fb: &windows::Win32::Graphics::DirectWrite::IDWriteFontFallback, text: &[u16], base_collection: Option<&windows::Win32::Graphics::DirectWrite::IDWriteFontCollection>) -> (u32, String) {
    use windows::Win32::Graphics::DirectWrite::*;
    use windows::core::HSTRING;
    let src: IDWriteTextAnalysisSource = ProbeSource { text: text.to_vec() }.into();
    let (mut len, mut font) = (0u32, None);
    let mut scale = 0.0f32;
    if unsafe { fb.MapCharacters(&src, 0, text.len() as u32, base_collection, &HSTRING::from("Arial"), DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, &mut len, &mut font, &mut scale) }.is_err() {
        return (0, String::new());
    }
    let fam = font
        .and_then(|f| unsafe { f.GetFontFamily() }.ok())
        .and_then(|fam| unsafe { fam.GetFamilyNames() }.ok())
        .map(|names| {
            let mut buf = [0u16; 64];
            let _ = unsafe { names.GetString(0, &mut buf) };
            String::from_utf16_lossy(&buf).trim_end_matches('\0').to_string()
        })
        .unwrap_or_default();
    (len, fam)
}

/// Shape `text` with the analyzer (GetGlyphs) and measure it (GetGlyphPlacements, 16 px em, natural),
/// returning the glyph ids and advances. Empty vecs if the analyzer or a call is unavailable.
#[cfg(windows)]
unsafe fn analyze_run(analyzer: &windows::Win32::Graphics::DirectWrite::IDWriteTextAnalyzer, face: &windows::Win32::Graphics::DirectWrite::IDWriteFontFace, text: &[u16]) -> (Vec<u16>, Vec<f32>) {
    use windows::Win32::Graphics::DirectWrite::*;
    use windows::core::PCWSTR;
    // A text-analysis source that hands back the whole string LTR, en-us. Implemented in-process.
    let src: IDWriteTextAnalysisSource = ProbeSource { text: text.to_vec() }.into();
    // Script analysis via a sink that captures the (first) script analysis.
    let sink: IDWriteTextAnalysisSink = ProbeSink::default().into();
    if unsafe { analyzer.AnalyzeScript(&src, 0, text.len() as u32, &sink) }.is_err() {
        return (Vec::new(), Vec::new());
    }
    let script = ProbeSink::take(&sink);
    let maxg = text.len() * 3 + 16;
    let mut cluster = vec![0u16; text.len()];
    let mut text_props = vec![DWRITE_SHAPING_TEXT_PROPERTIES::default(); text.len()];
    let mut glyphs = vec![0u16; maxg];
    let mut glyph_props = vec![DWRITE_SHAPING_GLYPH_PROPERTIES::default(); maxg];
    let mut actual = 0u32;
    let locale = windows::core::HSTRING::from("en-us");
    if unsafe {
        analyzer.GetGlyphs(PCWSTR(text.as_ptr()), text.len() as u32, face, false, false, &script, &locale, None, None, None, 0, maxg as u32, cluster.as_mut_ptr(), text_props.as_mut_ptr(), glyphs.as_mut_ptr(), glyph_props.as_mut_ptr(), &mut actual)
    }
    .is_err()
    {
        return (Vec::new(), Vec::new());
    }
    glyphs.truncate(actual as usize);
    let mut advances = vec![0f32; actual as usize];
    let mut offsets = vec![DWRITE_GLYPH_OFFSET::default(); actual as usize];
    let _ = unsafe {
        analyzer.GetGlyphPlacements(
            PCWSTR(text.as_ptr()), cluster.as_ptr(), text_props.as_mut_ptr(), text.len() as u32, glyphs.as_ptr(), glyph_props.as_ptr(), actual, face, 16.0, false, false, &script, &locale, None, None, 0, advances.as_mut_ptr(), offsets.as_mut_ptr(),
        )
    };
    (glyphs, advances)
}

/// IDWriteTextAnalyzer::GetGdiCompatibleGlyphPlacements (ppd 1.0, non-GDI-natural, 16 px em),
/// AnalyzeLineBreakpoints and AnalyzeNumberSubstitution for the run. Returns the GDI advances, the
/// per-position line-break bitfields, and how many times the number-substitution sink fired.
#[cfg(windows)]
unsafe fn analyze_extra(analyzer: &windows::Win32::Graphics::DirectWrite::IDWriteTextAnalyzer, face: &windows::Win32::Graphics::DirectWrite::IDWriteFontFace, text: &[u16]) -> (Vec<f32>, Vec<u8>, u32) {
    use windows::Win32::Graphics::DirectWrite::*;
    use windows::core::PCWSTR;
    let src: IDWriteTextAnalysisSource = ProbeSource { text: text.to_vec() }.into();
    let sink: IDWriteTextAnalysisSink = ProbeSink::default().into();
    if unsafe { analyzer.AnalyzeScript(&src, 0, text.len() as u32, &sink) }.is_err() {
        return (Vec::new(), Vec::new(), 0);
    }
    let script = ProbeSink::take(&sink);
    let maxg = text.len() * 3 + 16;
    let mut cluster = vec![0u16; text.len()];
    let mut text_props = vec![DWRITE_SHAPING_TEXT_PROPERTIES::default(); text.len()];
    let mut glyphs = vec![0u16; maxg];
    let mut glyph_props = vec![DWRITE_SHAPING_GLYPH_PROPERTIES::default(); maxg];
    let mut actual = 0u32;
    let locale = windows::core::HSTRING::from("en-us");
    if unsafe { analyzer.GetGlyphs(PCWSTR(text.as_ptr()), text.len() as u32, face, false, false, &script, &locale, None, None, None, 0, maxg as u32, cluster.as_mut_ptr(), text_props.as_mut_ptr(), glyphs.as_mut_ptr(), glyph_props.as_mut_ptr(), &mut actual) }.is_err() {
        return (Vec::new(), Vec::new(), 0);
    }
    let mut gdi_adv = vec![0f32; actual as usize];
    let mut gdi_off = vec![DWRITE_GLYPH_OFFSET::default(); actual as usize];
    let _ = unsafe {
        analyzer.GetGdiCompatibleGlyphPlacements(
            PCWSTR(text.as_ptr()), cluster.as_ptr(), text_props.as_mut_ptr(), text.len() as u32, glyphs.as_ptr(), glyph_props.as_ptr(), actual, face, 16.0, 1.0, None, false, false, false, &script, &locale, None, None, 0, gdi_adv.as_mut_ptr(), gdi_off.as_mut_ptr(),
        )
    };
    // AnalyzeLineBreakpoints + AnalyzeNumberSubstitution write into a fresh sink.
    let sink2: IDWriteTextAnalysisSink = ProbeSink::default().into();
    let _ = unsafe { analyzer.AnalyzeLineBreakpoints(&src, 0, text.len() as u32, &sink2) };
    let breaks = ProbeSink::linebreaks(&sink2);
    let _ = unsafe { analyzer.AnalyzeNumberSubstitution(&src, 0, text.len() as u32, &sink2) };
    let numsub = ProbeSink::numsub(&sink2);
    (gdi_adv, breaks, numsub)
}

#[cfg(windows)]
#[windows::core::implement(windows::Win32::Graphics::DirectWrite::IDWriteFontDownloadListener)]
struct ProbeDownloadListener;

#[cfg(windows)]
#[allow(non_snake_case)]
impl windows::Win32::Graphics::DirectWrite::IDWriteFontDownloadListener_Impl for ProbeDownloadListener_Impl {
    fn DownloadCompleted(&self, _q: windows_core::Ref<windows::Win32::Graphics::DirectWrite::IDWriteFontDownloadQueue>, _ctx: windows_core::Ref<windows_core::IUnknown>, _result: windows_core::HRESULT) {}
}

#[cfg(windows)]
#[windows::core::implement(windows::Win32::Graphics::DirectWrite::IDWriteTextAnalysisSource)]
struct ProbeSource {
    text: Vec<u16>,
}

#[cfg(windows)]
#[allow(non_snake_case)]
impl windows::Win32::Graphics::DirectWrite::IDWriteTextAnalysisSource_Impl for ProbeSource_Impl {
    fn GetTextAtPosition(&self, pos: u32, s: *mut *mut u16, len: *mut u32) -> windows_core::Result<()> {
        unsafe {
            if (pos as usize) < self.text.len() {
                *s = self.text.as_ptr().add(pos as usize) as *mut u16;
                *len = self.text.len() as u32 - pos;
            } else {
                *s = core::ptr::null_mut();
                *len = 0;
            }
        }
        Ok(())
    }
    fn GetTextBeforePosition(&self, _pos: u32, s: *mut *mut u16, len: *mut u32) -> windows_core::Result<()> {
        unsafe {
            *s = core::ptr::null_mut();
            *len = 0;
        }
        Ok(())
    }
    fn GetParagraphReadingDirection(&self) -> windows::Win32::Graphics::DirectWrite::DWRITE_READING_DIRECTION {
        windows::Win32::Graphics::DirectWrite::DWRITE_READING_DIRECTION_LEFT_TO_RIGHT
    }
    fn GetLocaleName(&self, _pos: u32, len: *mut u32, name: *mut *mut u16) -> windows_core::Result<()> {
        unsafe {
            *len = self.text.len() as u32;
            *name = core::ptr::null_mut();
        }
        Ok(())
    }
    fn GetNumberSubstitution(&self, _pos: u32, len: *mut u32, ns: windows_core::OutRef<windows::Win32::Graphics::DirectWrite::IDWriteNumberSubstitution>) -> windows_core::Result<()> {
        unsafe { *len = self.text.len() as u32 };
        let _ = ns.write(None);
        Ok(())
    }
}

#[cfg(windows)]
#[derive(Default)]
#[windows::core::implement(windows::Win32::Graphics::DirectWrite::IDWriteTextAnalysisSink)]
struct ProbeSink {
    script: std::cell::Cell<windows::Win32::Graphics::DirectWrite::DWRITE_SCRIPT_ANALYSIS>,
    bidi: std::cell::Cell<u8>,
    linebreaks: std::cell::RefCell<Vec<u8>>,
    numsub: std::cell::Cell<u32>,
}

#[cfg(windows)]
impl ProbeSink {
    fn take(sink: &windows::Win32::Graphics::DirectWrite::IDWriteTextAnalysisSink) -> windows::Win32::Graphics::DirectWrite::DWRITE_SCRIPT_ANALYSIS {
        use windows::core::Interface;
        sink.cast_object_ref::<ProbeSink>().map(|s| s.script.get()).unwrap_or_default()
    }
    fn bidi(sink: &windows::Win32::Graphics::DirectWrite::IDWriteTextAnalysisSink) -> u8 {
        use windows::core::Interface;
        sink.cast_object_ref::<ProbeSink>().map(|s| s.bidi.get()).unwrap_or(0)
    }
    fn linebreaks(sink: &windows::Win32::Graphics::DirectWrite::IDWriteTextAnalysisSink) -> Vec<u8> {
        use windows::core::Interface;
        sink.cast_object_ref::<ProbeSink>().map(|s| s.linebreaks.borrow().clone()).unwrap_or_default()
    }
    fn numsub(sink: &windows::Win32::Graphics::DirectWrite::IDWriteTextAnalysisSink) -> u32 {
        use windows::core::Interface;
        sink.cast_object_ref::<ProbeSink>().map(|s| s.numsub.get()).unwrap_or(0)
    }
}

#[cfg(windows)]
#[allow(non_snake_case)]
impl windows::Win32::Graphics::DirectWrite::IDWriteTextAnalysisSink_Impl for ProbeSink_Impl {
    fn SetScriptAnalysis(&self, _pos: u32, _len: u32, sa: *const windows::Win32::Graphics::DirectWrite::DWRITE_SCRIPT_ANALYSIS) -> windows_core::Result<()> {
        unsafe { self.script.set(*sa) };
        Ok(())
    }
    fn SetLineBreakpoints(&self, _pos: u32, len: u32, bp: *const windows::Win32::Graphics::DirectWrite::DWRITE_LINE_BREAKPOINT) -> windows_core::Result<()> {
        let mut out = self.linebreaks.borrow_mut();
        for i in 0..len as usize {
            out.push(unsafe { (*bp.add(i))._bitfield });
        }
        Ok(())
    }
    fn SetBidiLevel(&self, _pos: u32, _len: u32, _explicit: u8, resolved: u8) -> windows_core::Result<()> {
        self.bidi.set(resolved);
        Ok(())
    }
    fn SetNumberSubstitution(&self, _pos: u32, _len: u32, _ns: windows_core::Ref<windows::Win32::Graphics::DirectWrite::IDWriteNumberSubstitution>) -> windows_core::Result<()> {
        self.numsub.set(self.numsub.get() + 1);
        Ok(())
    }
}
