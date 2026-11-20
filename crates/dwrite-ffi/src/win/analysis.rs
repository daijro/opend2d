//! `IDWriteGlyphRunAnalysis` over `d2d::text::raster` — the coverage source Direct2D's glyph draw
//! asks DirectWrite for. `CreateGlyphRunAnalysis` builds one; the object answers `GetAlphaTextureBounds`
//! (the device-pixel rect), `CreateAlphaTexture` (the coverage bytes — 1/px grayscale or 3/px B,G,R
//! ClearType) and `GetAlphaBlendParams` (the gamma/contrast/level the text shader uses).
//!
//! The textures and bounds come straight from `d2d::text::raster::{grayscale,cleartype,aliased}`,
//! which is ported from DirectWrite's scan converter and verified byte-for-byte against
//! native `dwrite-dump` captures (grid-fit disabled). Validation (`E_INVALIDARG` ranges, the
//! texture-type-vs-analysis-mode rule) and the `GetAlphaBlendParams` branch formula match
//! DirectWrite's glyph-run analysis (texture bounds + light-on-dark alpha-blend params).

use super::{E_INVALIDARG, ITextInterop};
use core::slice;
use d2d::text::raster::{self, AlphaTexture};
use d2d::text::{Glyph, GlyphRun};
use skrifa::FontRef;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::DirectWrite::*;
use windows_core::{Interface, Ref, Result as WResult, implement};

/// Which blend/coverage class the analysis is: picks the `raster` kind, the texture type bit
/// (ClearType = 3×1) and the `GetAlphaBlendParams` branch.
#[derive(Clone, Copy, PartialEq)]
enum Branch {
    Grayscale,
    ClearType,
    Aliased,
}

#[implement(IDWriteGlyphRunAnalysis)]
pub(crate) struct AnalysisObj {
    tex: AlphaTexture,
    branch: Branch,
}

/// `CreateGlyphRunAnalysis`: validate (matching native's `DWriteFactory::CreateGlyphRunAnalysis`
/// `E_INVALIDARG` ranges), resolve the font bytes from the run's face via [`ITextInterop`], build the
/// `d2d::text` glyph run and scan-convert it into the coverage texture for the analysis's mode.
///
/// `dt` is the full DIP→device transform (pixels-per-dip folded in by the caller); `baseline` is the
/// run origin in DIPs. `rendering_mode` is `DWRITE_RENDERING_MODE1`, `antialias`
/// `DWRITE_TEXT_ANTIALIAS_MODE` (0 ClearType, 1 grayscale).
#[allow(clippy::too_many_arguments)]
pub(crate) fn create(
    run: *const DWRITE_GLYPH_RUN,
    dt: [f64; 6],
    rendering_mode: i32,
    measuring: i32,
    grid_fit: i32,
    antialias: i32,
    baseline: [f32; 2],
) -> WResult<IDWriteGlyphRunAnalysis> {
    if run.is_null() {
        return Err(E_INVALIDARG.into());
    }
    let run = unsafe { &*run };
    let Some(face) = run.fontFace.as_ref() else { return Err(E_INVALIDARG.into()) };
    if run.glyphIndices.is_null()
        || run.fontEmSize < 0.0
        || !dt.iter().all(|v| v.is_finite()) // native `IsValidTransform` rejects a non-finite matrix
        || !(0..=7).contains(&rendering_mode)
        || !(0..=2).contains(&measuring)
        || !(0..=2).contains(&grid_fit)
        || !(0..=1).contains(&antialias)
    {
        return Err(E_INVALIDARG.into());
    }

    // Font bytes via our private interop interface (the face a D2D/dwrite caller holds is our
    // `FaceObj`); a foreign face has no `d2d::text` outline source, so it is unsupported.
    let interop: ITextInterop = face.cast().map_err(|_| windows_core::Error::from(E_INVALIDARG))?;
    let (mut data, mut len, mut index) = (core::ptr::null(), 0usize, 0u32);
    if unsafe { interop.FontBlob(&mut data, &mut len, &mut index) }.is_err() || data.is_null() {
        return Err(E_INVALIDARG.into());
    }
    let bytes = unsafe { slice::from_raw_parts(data, len) };
    let Ok(font) = FontRef::from_index(bytes, index) else { return Err(E_INVALIDARG.into()) };

    let n = run.glyphCount as usize;
    let ids = unsafe { slice::from_raw_parts(run.glyphIndices, n) };
    let advances: &[f32] =
        if run.glyphAdvances.is_null() { &[] } else { unsafe { slice::from_raw_parts(run.glyphAdvances, n) } };
    let offsets: &[DWRITE_GLYPH_OFFSET] =
        if run.glyphOffsets.is_null() { &[] } else { unsafe { slice::from_raw_parts(run.glyphOffsets, n) } };
    let glyphs: Vec<Glyph> = (0..n)
        .map(|i| Glyph {
            id: ids[i],
            advance: advances.get(i).copied().unwrap_or(0.0),
            offset: offsets.get(i).map(|o| [o.advanceOffset, o.ascenderOffset]).unwrap_or([0.0, 0.0]),
        })
        .collect();

    // Rendering mode ALIASED(1) is the 1-sample binary texture regardless of AA mode; otherwise the
    // AA mode picks grayscale (1×1) vs ClearType (3×1). `no_snap`: `CreateGlyphRunAnalysis` takes the
    // baseline as given (the caller already snapped/phased it — the sub-pixel phase is deliberate).
    let branch = if rendering_mode == DWRITE_RENDERING_MODE1_ALIASED.0 {
        Branch::Aliased
    } else if antialias == DWRITE_TEXT_ANTIALIAS_MODE_CLEARTYPE.0 {
        Branch::ClearType
    } else {
        Branch::Grayscale
    };
    let trun = GlyphRun {
        font_data: bytes,
        face_index: index,
        em_size: run.fontEmSize,
        glyphs: &glyphs,
        baseline_origin: baseline,
        measuring_mode: measuring as u32,
        is_sideways: run.isSideways.as_bool(),
        bidi_level: run.bidiLevel,
        no_snap: true,
        simulations: 0,
        coords: &[],
    };
    // A right-to-left run draws as its left-to-right equivalent, each glyph's own advance (the face's, bold
    // simulation included) ending at the pen.
    let mut rtl = Vec::new();
    let sims = unsafe { face.GetSimulations() }.0 as u32;
    let trun = GlyphRun { simulations: 0, ..d2d::text::drawn_ltr(&GlyphRun { simulations: sims, ..trun }, &dt, &mut rtl) };
    // The unhinted scan-converter path (byte-exact for grid-fit DISABLED, the verified oracle), shared
    // with the D2D DrawGlyphRun path via `d2d::text::unhinted_glyph_coverage` — ONE rasterization, so
    // the rendering mode selects the same oversample grid both paths use (GDI/NATURAL are horizontal-only
    // 6×1 / 8×1; NATURAL_SYMMETRIC is 6×5 / 4×4), fixing the extra curve-apex row the old mode-agnostic
    // `raster::cleartype` (always 6×5) produced for GDI/NATURAL curved glyphs. Grid-fit ENABLED/DEFAULT
    // would run the TrueType hinting interpreter first; that hinted-analysis path is not yet in the COM
    // shim. The analysis is the in-process DirectWrite rasterizer (not the FontCache service).
    let aa = match branch {
        Branch::ClearType => d2d::text::AA_CLEARTYPE,
        Branch::Aliased => d2d::text::AA_ALIASED,
        Branch::Grayscale => d2d::text::AA_GRAYSCALE,
    };
    let tex = d2d::text::unhinted_glyph_coverage(
        &dt,
        &trun,
        &font,
        aa,
        rendering_mode as u32,
        d2d::text::hinting::Rasterizer::DirectWrite,
        false,
        None,
    );
    Ok(AnalysisObj { tex, branch }.into())
}

impl IDWriteGlyphRunAnalysis_Impl for AnalysisObj_Impl {
    fn GetAlphaTextureBounds(&self, texturetype: DWRITE_TEXTURE_TYPE) -> WResult<RECT> {
        // Native: `type > 1` ⇒ E_INVALIDARG; a valid type that is not the analysis's own type ⇒
        // empty rect + S_OK (`a2 == (HIWORD(renderingModeAttributes) & 1)`).
        if !(0..=1).contains(&texturetype.0) {
            return Err(E_INVALIDARG.into());
        }
        let is_ct = self.branch == Branch::ClearType;
        if (texturetype == DWRITE_TEXTURE_CLEARTYPE_3x1) != is_ct {
            return Ok(RECT::default());
        }
        let t = &self.tex;
        Ok(RECT { left: t.left, top: t.top, right: t.right, bottom: t.bottom })
    }

    fn CreateAlphaTexture(&self, texturetype: DWRITE_TEXTURE_TYPE, texturebounds: *const RECT, alphavalues: *mut u8, buffersize: u32) -> WResult<()> {
        // Native: null or empty rect ⇒ E_INVALIDARG; type ∉ {0,1} ⇒ E_INVALIDARG.
        if texturebounds.is_null() {
            return Err(E_INVALIDARG.into());
        }
        let r = unsafe { *texturebounds };
        if r.right <= r.left || r.bottom <= r.top || !(0..=1).contains(&texturetype.0) {
            return Err(E_INVALIDARG.into());
        }
        let want_ct = texturetype == DWRITE_TEXTURE_CLEARTYPE_3x1;
        let bpp = if want_ct { 3 } else { 1 };
        let (w, h) = ((r.right - r.left) as usize, (r.bottom - r.top) as usize);
        let out = unsafe { slice::from_raw_parts_mut(alphavalues, (buffersize as usize).min(w * h * bpp)) };
        out.fill(0);
        // A requested type that is not the analysis's own mode leaves zeros (its bounds were already
        // reported empty); otherwise window-copy the overlap with the analysis's coverage texture.
        if want_ct != (self.branch == Branch::ClearType) {
            return Ok(());
        }
        let t = &self.tex;
        let tw = t.width() as usize;
        for y in r.top.max(t.top)..r.bottom.min(t.bottom) {
            for x in r.left.max(t.left)..r.right.min(t.right) {
                let si = ((y - t.top) as usize * tw + (x - t.left) as usize) * bpp;
                let di = ((y - r.top) as usize * w + (x - r.left) as usize) * bpp;
                if di + bpp <= out.len() && si + bpp <= t.data.len() {
                    out[di..di + bpp].copy_from_slice(&t.data[si..si + bpp]);
                }
            }
        }
        Ok(())
    }

    fn GetAlphaBlendParams(&self, renderingparams: Ref<IDWriteRenderingParams>, blendgamma: *mut f32, blendenhancedcontrast: *mut f32, blendcleartypelevel: *mut f32) -> WResult<()> {
        let p = renderingparams.as_ref().ok_or(E_INVALIDARG)?;
        // `GetAlphaBlendParams` = `…WithLightOnDarkAdjustment` with colour (0,0,0): the luminance ramp
        // is 1.0 and the per-analysis offset/suppress bytes are 0 for a plain run, so the three
        // scalars are the params' values selected by the analysis's branch:
        // grayscale → (gamma, grayscaleEnhancedContrast, 0); ClearType → (gamma, enhancedContrast,
        // clearTypeLevel); aliased → (gamma, 0, 1).
        let gamma = unsafe { p.GetGamma() };
        let (contrast, level) = match self.branch {
            Branch::Grayscale => {
                let g = p.cast::<IDWriteRenderingParams1>().map(|p1| unsafe { p1.GetGrayscaleEnhancedContrast() }).unwrap_or_else(|_| unsafe { p.GetEnhancedContrast() });
                (g, 0.0)
            }
            Branch::ClearType => (unsafe { p.GetEnhancedContrast() }, unsafe { p.GetClearTypeLevel() }),
            Branch::Aliased => (0.0, 1.0),
        };
        unsafe {
            *blendgamma = gamma;
            *blendenhancedcontrast = contrast;
            *blendcleartypelevel = level;
        }
        Ok(())
    }
}
