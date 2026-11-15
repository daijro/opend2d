//! The pixel-processing sources: `IWICFormatConverter`, `IWICBitmapScaler`,
//! `IWICBitmapFlipRotator`, `IWICBitmapClipper`. Each `Initialize` reads its input source through
//! the public ABI, runs the matching `d2d::wic` routine, and holds the result as an
//! `IWICBitmapSource`.

use super::{Source, bpp_of, e, guid_to_str, impl_source, read_source, wicerr};
use d2d::wic::color_transform::icm;
use d2d::wic::convert::{Format, convert};
use d2d::wic::fliprotate::{clip, flip_rotate};
use d2d::wic::scale::{Interp, scale, scale_gray16_to_grayfloat};
use std::cell::RefCell;
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Imaging::*;
use windows_core::{GUID, Ref, Result as WResult, implement};

#[implement(IWICFormatConverter)]
#[derive(Default)]
pub(crate) struct ConverterObj {
    src: RefCell<Option<Source>>,
}
impl_source!(ConverterObj_Impl);

impl IWICFormatConverter_Impl for ConverterObj_Impl {
    fn Initialize(&self, source: Ref<IWICBitmapSource>, dstformat: *const GUID, _dither: WICBitmapDitherType, _pal: Ref<IWICPalette>, _alpha: f64, _translate: WICBitmapPaletteType) -> WResult<()> {
        let s = read_source(source.as_ref().ok_or(e(E_POINTER))?)?;
        let dstg = unsafe { *dstformat };
        let sf = Format::from_guid(&guid_to_str(&s.format)).ok_or(e(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT))?;
        let df = Format::from_guid(&guid_to_str(&dstg)).ok_or(e(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT))?;
        let pixels = convert(sf, df, &s.pixels, s.width, s.height, &s.palette).map_err(wicerr)?;
        let stride = ((s.width * bpp_of(&dstg) + 7) / 8) as usize;
        *self.src.borrow_mut() = Some(Source { format: dstg, width: s.width, height: s.height, stride, pixels, palette: Vec::new(), dpi: s.dpi });
        Ok(())
    }
    fn CanConvert(&self, _src: *const GUID, _dst: *const GUID) -> WResult<windows_core::BOOL> {
        Ok(true.into())
    }
}

/// The canonical format native's scaler converts a source to before scaling, or `None` when the
/// source scales in place. The whole-byte BGR/BGRA/gray formats (BGR24/RGB24/BGR32/BGRA32/PBGRA32/
/// Gray8) always scale in place. The straight/premultiplied 32bpp RGBA formats (RGBA32/PRGBA32) are
/// scaled in place ONLY by the HighQualityCubic resampler; every lower mode (NearestNeighbor/Linear/
/// Cubic/Fant) converts them to 32bppBGRA first. The sub-byte and higher-than-8bpc formats
/// (BGR555/BGR565/Gray16/RGBA64) convert to 32bppPBGRA — native scales those at their own depth, but
/// our scaler only does 8/24/32bpp, so this is a lossy fallback until the higher-depth paths land.
fn scaler_canonical(fmt: &GUID, mode: Interp) -> Option<GUID> {
    const BGRA32: GUID = GUID::from_u128(0x6fddc324_4e03_4bfe_b185_3d77768dc90f);
    const PBGRA32: GUID = GUID::from_u128(0x6fddc324_4e03_4bfe_b185_3d77768dc910);
    if fmt.data1 == 0xf5c7ad2d || fmt.data1 == 0x3cc4a650 {
        return (mode != Interp::HighQualityCubic).then_some(BGRA32); // RGBA32/PRGBA32
    }
    if fmt.data1 == 0x6fddc324 {
        return match fmt.data4[7] {
            0x09 | 0x0a | 0x0b | 0x16 => Some(PBGRA32), // BGR555/BGR565/Gray16/RGBA64 (depth we can't scale yet)
            _ => None,
        };
    }
    None
}

#[implement(IWICBitmapScaler)]
#[derive(Default)]
pub(crate) struct ScalerObj {
    src: RefCell<Option<Source>>,
}
impl_source!(ScalerObj_Impl);

impl IWICBitmapScaler_Impl for ScalerObj_Impl {
    fn Initialize(&self, source: Ref<IWICBitmapSource>, w: u32, h: u32, mode: WICBitmapInterpolationMode) -> WResult<()> {
        let s = read_source(source.as_ref().ok_or(e(E_POINTER))?)?;
        let interp = Interp::from_mode(mode.0).ok_or(e(E_INVALIDARG))?;
        // Gray16 under the Fant resampler is a special case: native routes the source through an
        // IWICFormatConverter to 32bppGrayFloat (sRGB decode + luma) and resamples in float, emitting
        // 32bppGrayFloat. Every other resampler converts Gray16 to 32bppPBGRA (the generic path
        // below), matching native's per-mode source-format selection.
        if s.format.data1 == 0x6fddc324 && s.format.data4[7] == 0x0b && interp == Interp::Fant {
            const GRAYFLOAT: GUID = GUID::from_u128(0x6fddc324_4e03_4bfe_b185_3d77768dc911);
            let pixels = scale_gray16_to_grayfloat(&s.pixels, s.width, s.height, w, h);
            let stride = (w * 4) as usize;
            *self.src.borrow_mut() =
                Some(Source { format: GRAYFLOAT, width: w, height: h, stride, pixels, palette: Vec::new(), dpi: s.dpi });
            return Ok(());
        }
        // Native's BitmapScaler scales a fixed set of formats in place (BGR24/BGR32/BGRA32/Gray8/
        // PBGRA32/RGB24) but CONVERTS the rest to a canonical scalable format first, and reports the
        // scaled output in that format (VM-derived from native-r1: 16bpp BGR555/565 + Gray16 + 64bpp
        // RGBA64 → 32bppPBGRA; straight/premul 32bpp RGBA32/PRGBA32 → 32bppBGRA). Mirror that: convert
        // to the canonical format, then scale there.
        let (fmt, pre) = match scaler_canonical(&s.format, interp) {
            Some(c) => {
                let sf = Format::from_guid(&guid_to_str(&s.format)).ok_or(e(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT))?;
                let df = Format::from_guid(&guid_to_str(&c)).ok_or(e(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT))?;
                (c, convert(sf, df, &s.pixels, s.width, s.height, &s.palette).map_err(wicerr)?)
            }
            None => (s.format, s.pixels.clone()),
        };
        let bpp = bpp_of(&fmt);
        // Native premultiplies a straight-alpha 32bpp source (BGRA32 or RGBA32) before a blending
        // scale; the premultiplied formats (PBGRA32/PRGBA32) and opaque formats need no round trip.
        let straight = fmt.data1 == 0xf5c7ad2d || (fmt.data1 == 0x6fddc324 && fmt.data4[7] == 0x0f);
        let pixels = scale(&pre, s.width, s.height, bpp, w, h, interp, straight).map_err(wicerr)?;
        let stride = ((w * bpp + 7) / 8) as usize;
        *self.src.borrow_mut() = Some(Source { format: fmt, width: w, height: h, stride, pixels, palette: Vec::new(), dpi: s.dpi });
        Ok(())
    }
}

#[implement(IWICBitmapFlipRotator)]
#[derive(Default)]
pub(crate) struct FlipRotatorObj {
    src: RefCell<Option<Source>>,
}
impl_source!(FlipRotatorObj_Impl);

impl IWICBitmapFlipRotator_Impl for FlipRotatorObj_Impl {
    fn Initialize(&self, source: Ref<IWICBitmapSource>, options: WICBitmapTransformOptions) -> WResult<()> {
        let s = read_source(source.as_ref().ok_or(e(E_POINTER))?)?;
        let bpp = bpp_of(&s.format);
        let (pixels, ow, oh) = flip_rotate(&s.pixels, s.width, s.height, bpp, options.0 as u32).map_err(wicerr)?;
        let stride = ((ow * bpp + 7) / 8) as usize;
        *self.src.borrow_mut() = Some(Source { format: s.format, width: ow, height: oh, stride, pixels, palette: s.palette, dpi: s.dpi });
        Ok(())
    }
}

#[implement(IWICBitmapClipper)]
#[derive(Default)]
pub(crate) struct ClipperObj {
    src: RefCell<Option<Source>>,
}
impl_source!(ClipperObj_Impl);

impl IWICBitmapClipper_Impl for ClipperObj_Impl {
    fn Initialize(&self, source: Ref<IWICBitmapSource>, prc: *const WICRect) -> WResult<()> {
        let s = read_source(source.as_ref().ok_or(e(E_POINTER))?)?;
        let bpp = bpp_of(&s.format);
        let r = unsafe { &*prc };
        let (cw, ch) = (r.Width as u32, r.Height as u32);
        let pixels = clip(&s.pixels, s.width, s.height, bpp, r.X as u32, r.Y as u32, cw, ch).map_err(wicerr)?;
        let stride = ((cw * bpp + 7) / 8) as usize;
        *self.src.borrow_mut() = Some(Source { format: s.format, width: cw, height: ch, stride, pixels, palette: s.palette, dpi: s.dpi });
        Ok(())
    }
}

/// `IWICColorTransform`: recolour a bitmap between two ICC-profile colour contexts through the
/// legacy "Win " CMM (`d2d::wic::color_transform::icm`), with alpha passed through. The destination
/// pixel format must match the source's bit depth (the CMM recolours in place, it does not convert
/// container formats here); `128bppRGBAFloat` is rejected exactly as native
/// (`WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT`).
#[implement(IWICColorTransform)]
#[derive(Default)]
pub(crate) struct ColorTransformObj {
    src: RefCell<Option<Source>>,
}
impl_source!(ColorTransformObj_Impl);

/// An `IWICColorContext`'s ICC profile bytes (two-call size/fill pattern); empty for a non-profile
/// (EXIF) context, which then fails `IccProfile::parse`.
fn profile_bytes(ctx: &IWICColorContext) -> WResult<Vec<u8>> {
    let mut n = 0u32;
    unsafe { ctx.GetProfileBytes(&mut [], &mut n)? };
    let mut v = vec![0u8; n as usize];
    if n > 0 {
        unsafe { ctx.GetProfileBytes(&mut v, &mut n)? };
    }
    Ok(v)
}

/// Pack a transformed RGB pixel stream into the destination WIC format (`data4[7]`): 24bppRGB=0x0d,
/// 32bppBGRA=0x0f (alpha 0xFF), 64bppRGBA=0x16 (alpha 0xFFFF). `rgb8`/`rgb16` yield one pixel each.
fn pack_rgb(dst: &GUID, n: usize, mut rgb8: impl FnMut(usize) -> [u8; 3], mut rgb16: impl FnMut(usize) -> [u16; 3]) -> WResult<Vec<u8>> {
    if dst.data1 != 0x6fddc324 {
        return Err(e(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT));
    }
    let mut out = Vec::new();
    match dst.data4[7] {
        0x0d => (0..n).for_each(|i| out.extend_from_slice(&rgb8(i))),
        0x0f => (0..n).for_each(|i| { let o = rgb8(i); out.extend_from_slice(&[o[2], o[1], o[0], 0xFF]); }),
        0x16 => (0..n).for_each(|i| { let o = rgb16(i); for c in o { out.extend_from_slice(&c.to_le_bytes()); } out.extend_from_slice(&[0xFF, 0xFF]); }),
        _ => return Err(e(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT)),
    }
    Ok(out)
}

impl IWICColorTransform_Impl for ColorTransformObj_Impl {
    fn Initialize(&self, source: Ref<IWICBitmapSource>, picontextsource: Ref<IWICColorContext>, picontextdest: Ref<IWICColorContext>, pixelfmtdest: *const GUID) -> WResult<()> {
        let s = read_source(source.as_ref().ok_or(e(E_POINTER))?)?;
        let dst = unsafe { *pixelfmtdest };
        let src_bytes = profile_bytes(picontextsource.as_ref().ok_or(e(E_POINTER))?)?;
        let dst_bytes = profile_bytes(picontextdest.as_ref().ok_or(e(E_POINTER))?)?;
        // The destination is always an RGB matrix/shaper profile (the CMM maps *to* a display space).
        let dp = icm::IccProfile::parse(&dst_bytes);

        // 3→3 (RGB/CLUT/mAB source) — recolour in place; dest must match the source bit depth (a
        // container conversion we don't do here), and a zero/float dest depth is unsupported.
        if let Some(sp) = icm::IccProfile::parse(&src_bytes) {
            if bpp_of(&dst) == 0 || bpp_of(&dst) != bpp_of(&s.format) {
                return Err(e(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT));
            }
            let dp = dp.ok_or(e(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT))?;
            let combi = icm::Combi::new(&sp, &dp);
            let mut px = s.pixels.clone();
            // channel order by WIC format id (data4[7]): 32bppBGRA=0x0f, 24bppRGB=0x0d, 64bppRGBA=0x16.
            match (s.format.data1, s.format.data4[7]) {
                (0x6fddc324, 0x0f) => {
                    for p in px.chunks_exact_mut(4) {
                        let o = combi.apply([p[2], p[1], p[0]]);
                        (p[0], p[1], p[2]) = (o[2], o[1], o[0]);
                    }
                }
                (0x6fddc324, 0x0d) => {
                    for p in px.chunks_exact_mut(3) {
                        p.copy_from_slice(&combi.apply([p[0], p[1], p[2]]));
                    }
                }
                (0x6fddc324, 0x16) => {
                    for p in px.chunks_exact_mut(8) {
                        let rd = |i: usize| u16::from_le_bytes([p[i], p[i + 1]]);
                        let o = combi.apply16([rd(0), rd(2), rd(4)]);
                        p[0..2].copy_from_slice(&o[0].to_le_bytes());
                        p[2..4].copy_from_slice(&o[1].to_le_bytes());
                        p[4..6].copy_from_slice(&o[2].to_le_bytes());
                    }
                }
                _ => return Err(e(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT)),
            }
            *self.src.borrow_mut() = Some(Source { format: dst, pixels: px, ..s });
            return Ok(());
        }

        // CMYK 4→3: a 'prtr'/'CMYK' source (32bppCMYK, data4[7]=0x1c) → an RGB dest format.
        if let Some(cmyk) = icm::cmyk_a2b(&src_bytes) {
            if s.format.data1 != 0x6fddc324 || s.format.data4[7] != 0x1c || bpp_of(&dst) == 0 {
                return Err(e(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT));
            }
            let dp = dp.ok_or(e(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT))?;
            let combi = icm::Combi4::new(&cmyk, &dp);
            let cm: Vec<[u8; 4]> = s.pixels.chunks_exact(4).map(|p| [p[0], p[1], p[2], p[3]]).collect();
            let n = cm.len();
            let px = pack_rgb(&dst, n, |i| combi.apply(cm[i]), |i| combi.apply16(cm[i]))?;
            let stride = ((s.width * bpp_of(&dst) + 7) / 8) as usize;
            *self.src.borrow_mut() = Some(Source { format: dst, width: s.width, height: s.height, stride, pixels: px, palette: vec![], dpi: s.dpi });
            return Ok(());
        }
        Err(e(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT))
    }
}
