//! The imaging factory and its simplest products: palette, pixel-format component info, an in-memory
//! bitmap and a file-backed stream.

use super::{Source, bpp_of, e, guid_to_str, impl_source, read_source, wicerr};
use d2d::wic::convert::{Format, convert};
use d2d::wic::quantize::median_cut;
use std::cell::{Cell, RefCell};
use std::fs::File;
use std::io::{Read as _, Seek as _, SeekFrom};
use super::stream::StreamObj;
use std::rc::Rc;
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Imaging::*;
use windows::Win32::System::Com::*;
use windows_core::{GUID, HRESULT, Interface, Ref, Result as WResult, implement};

#[implement(IWICImagingFactory)]
pub(crate) struct FactoryObj;

impl FactoryObj {
    pub(crate) fn new() -> Self {
        FactoryObj
    }
}

impl IWICImagingFactory_Impl for FactoryObj_Impl {
    fn CreateDecoderFromFilename(
        &self,
        wzfilename: &windows_core::PCWSTR,
        _vendor: *const GUID,
        _access: GENERIC_ACCESS_RIGHTS,
        _opts: WICDecodeOptions,
    ) -> WResult<IWICBitmapDecoder> {
        let path = unsafe { wzfilename.to_string() }.map_err(|_| e(E_INVALIDARG))?;
        let bytes = std::fs::read(&path).map_err(|_| e(WINCODEC_ERR_COMPONENTNOTFOUND))?;
        super::decode::DecoderObj::from_bytes(bytes)
    }
    fn CreateDecoderFromStream(&self, stream: Ref<IStream>, _vendor: *const GUID, _opts: WICDecodeOptions) -> WResult<IWICBitmapDecoder> {
        super::decode::DecoderObj::from_stream(stream.as_ref().ok_or(e(E_POINTER))?)
    }
    fn CreateDecoderFromFileHandle(&self, h: usize, _v: *const GUID, _o: WICDecodeOptions) -> WResult<IWICBitmapDecoder> {
        // A raw Win32 file HANDLE only has meaning on Windows (callers on other platforms use
        // `CreateDecoderFromStream`/`FromFilename`).
        #[cfg(not(windows))]
        {
            let _ = h;
            return Err(e(WINCODEC_ERR_WIN32ERROR));
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::FromRawHandle;
            // Borrow the caller's file HANDLE (don't take ownership — WIC must not close it): read
            // from the start, then `forget` so the handle stays the caller's.
            let mut file = unsafe { File::from_raw_handle(h as *mut _) };
            let read = (|| -> std::io::Result<Vec<u8>> {
                file.seek(SeekFrom::Start(0))?;
                let mut b = Vec::new();
                file.read_to_end(&mut b)?;
                Ok(b)
            })();
            std::mem::forget(file);
            super::decode::DecoderObj::from_bytes(read.map_err(|_| e(WINCODEC_ERR_WIN32ERROR))?)
        }
    }
    fn CreateComponentInfo(&self, clsid: *const GUID) -> WResult<IWICComponentInfo> {
        super::components::create_component_info(clsid)
    }
    fn CreateDecoder(&self, container: *const GUID, vendor: *const GUID) -> WResult<IWICBitmapDecoder> {
        super::components::create_decoder(container,vendor)
    }
    fn CreateEncoder(&self, container: *const GUID, _vendor: *const GUID) -> WResult<IWICBitmapEncoder> {
        super::encode::EncoderObj::for_container(unsafe { *container })
    }
    fn CreatePalette(&self) -> WResult<IWICPalette> {
        Ok(PaletteObj::default().into())
    }
    fn CreateFormatConverter(&self) -> WResult<IWICFormatConverter> {
        Ok(super::process::ConverterObj::default().into())
    }
    fn CreateBitmapScaler(&self) -> WResult<IWICBitmapScaler> {
        Ok(super::process::ScalerObj::default().into())
    }
    fn CreateBitmapClipper(&self) -> WResult<IWICBitmapClipper> {
        Ok(super::process::ClipperObj::default().into())
    }
    fn CreateBitmapFlipRotator(&self) -> WResult<IWICBitmapFlipRotator> {
        Ok(super::process::FlipRotatorObj::default().into())
    }
    fn CreateStream(&self) -> WResult<IWICStream> {
        Ok(StreamObj::default().into())
    }
    fn CreateColorContext(&self) -> WResult<IWICColorContext> {
        Ok(ColorContextObj::default().into())
    }
    fn CreateColorTransformer(&self) -> WResult<IWICColorTransform> {
        Ok(super::process::ColorTransformObj::default().into())
    }
    fn CreateBitmap(&self, w: u32, h: u32, fmt: *const GUID, _o: WICBitmapCreateCacheOption) -> WResult<IWICBitmap> {
        let format = unsafe { *fmt };
        let bpp = bpp_of(&format);
        if bpp == 0 {
            return Err(e(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT));
        }
        if w == 0 || h == 0 { return Err(e(E_INVALIDARG)); }
        // WIC allocates a TIGHT stride (ceil(w*bpp/8)), not DWORD-aligned: a
        // full-bitmap lock hands back this buffer directly, so a flat lock-write round-trips to the
        // packed pixels byte-exact (verified: bitmaplock rgb24/gray8 identity through the WIC ABI).
        let stride = w.checked_mul(bpp).and_then(|n| n.checked_add(7)).ok_or(e(HRESULT(0x80070216u32 as i32)))? / 8;
        let (_, size) = super::buffer_size(w,h,bpp,stride,usize::MAX)?;
        let mut pixels = Vec::new();
        pixels.try_reserve_exact(size).map_err(|_|e(E_OUTOFMEMORY))?;
        pixels.resize(size,0);
        let stride = stride as usize;
        Ok(BitmapObj::new(Source { format, width: w, height: h, stride, pixels, palette: Vec::new(), dpi: (0.0, 0.0) }).into())
    }
    fn CreateBitmapFromSource(&self, src: Ref<IWICBitmapSource>, _o: WICBitmapCreateCacheOption) -> WResult<IWICBitmap> {
        Ok(BitmapObj::new(super::read_source(src.as_ref().ok_or(e(E_POINTER))?)?).into())
    }
    fn CreateBitmapFromSourceRect(&self, s: Ref<IWICBitmapSource>, x: u32, y: u32, w: u32, h: u32) -> WResult<IWICBitmap> {
        // Same crop as CreateBitmapClipper (the oracle-verified `clip`), materialised into a bitmap.
        let src = super::read_source(s.as_ref().ok_or(e(E_POINTER))?)?;
        let bpp = bpp_of(&src.format);
        let pixels = d2d::wic::fliprotate::clip(&src.pixels, src.width, src.height, bpp, x, y, w, h).map_err(super::wicerr)?;
        let stride = ((w * bpp + 7) / 8) as usize;
        Ok(BitmapObj::new(Source { format: src.format, width: w, height: h, stride, pixels, palette: src.palette, dpi: src.dpi }).into())
    }
    fn CreateBitmapFromMemory(&self, w: u32, h: u32, fmt: *const GUID, cbstride: u32, size: u32, buf: *const u8) -> WResult<IWICBitmap> {
        if fmt.is_null() || buf.is_null() || w == 0 || h == 0 { return Err(e(E_INVALIDARG)); }
        let format = unsafe { *fmt };
        let bpp = bpp_of(&format);
        if bpp == 0 { return Err(e(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT)); }
        let (_, needed) = super::buffer_size(w,h,bpp,cbstride,size as usize)?;
        // WIC's memory-bitmap init preserves the supplied stride and omits final-row padding.
        let src = unsafe { core::slice::from_raw_parts(buf,needed) };
        let mut pixels = Vec::new();
        pixels.try_reserve_exact(needed).map_err(|_|e(E_OUTOFMEMORY))?;
        pixels.extend_from_slice(src);
        Ok(BitmapObj::new(Source { format, width:w, height:h, stride:cbstride as usize, pixels, palette:Vec::new(), dpi:(0.0,0.0) }).into())
    }

    fn CreateBitmapFromHBITMAP(&self, hbm: windows::Win32::Graphics::Gdi::HBITMAP, _p: windows::Win32::Graphics::Gdi::HPALETTE, option: WICBitmapAlphaChannelOption) -> WResult<IWICBitmap> {
        // GDI `HBITMAP`s only exist on Windows.
        #[cfg(not(windows))]
        return {
            let _ = (hbm, option);
            Err(e(E_NOTIMPL))
        };
        #[cfg(windows)]
        {
        use windows::Win32::Graphics::Gdi::*;
        let mut bm = BITMAP::default();
        if unsafe { GetObjectW(hbm.into(), core::mem::size_of::<BITMAP>() as i32, Some(&mut bm as *mut _ as *mut core::ffi::c_void)) } == 0 || bm.bmWidth <= 0 || bm.bmHeight == 0 {
            return Err(e(E_INVALIDARG));
        }
        let (w, h) = (bm.bmWidth as u32, bm.bmHeight.unsigned_abs());
        // WIC's init-from-HBITMAP: 32bpp maps by alpha option, 24bpp → BGR;
        // the alpha option only renames the format (no premultiply), so the bytes are the DIB bits.
        let suffix: u8 = match (bm.bmBitsPixel, option.0) {
            (32, 0) => 0x0f, // WICBitmapUseAlpha → 32bppBGRA
            (32, 1) => 0x10, // WICBitmapUsePremultipliedAlpha → 32bppPBGRA
            (32, 2) => 0x0e, // WICBitmapIgnoreAlpha → 32bppBGR
            (32, _) => return Err(e(E_INVALIDARG)),
            (24, _) => 0x0c, // 24bppBGR
            _ => return Err(e(E_NOTIMPL)), // indexed / 16bpp sources not yet reproduced
        };
        let (bgra, _) = dib_bits(hbm, w, h, 32);
        let (pixels, stride) = if suffix == 0x0c {
            (bgra.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect(), (w * 3) as usize)
        } else {
            (bgra, (w * 4) as usize)
        };
        let format = GUID::from_u128(0x6fddc324_4e03_4bfe_b185_3d77768dc900u128 | suffix as u128);
        Ok(BitmapObj::new(Source { format, width: w, height: h, stride, pixels, palette: Vec::new(), dpi: (96.0, 96.0) }).into())
        }
    }
    fn CreateBitmapFromHICON(&self, hicon: windows::Win32::UI::WindowsAndMessaging::HICON) -> WResult<IWICBitmap> {
        // GDI icons only exist on Windows.
        #[cfg(not(windows))]
        return {
            let _ = hicon;
            Err(e(E_NOTIMPL))
        };
        #[cfg(windows)]
        {
        use windows::Win32::Graphics::Gdi::*;
        use windows::Win32::UI::WindowsAndMessaging::{GetIconInfo, ICONINFO};
        let mut ii = ICONINFO::default();
        unsafe { GetIconInfo(hicon, &mut ii) }.map_err(|_| e(E_INVALIDARG))?;
        // Colour-mask (monochrome) icons use HrMaskMonochromeIcon — not reproduced here.
        if ii.hbmColor.is_invalid() {
            unsafe {
                let _ = DeleteObject(ii.hbmMask.into());
            }
            return Err(e(E_NOTIMPL));
        }
        let mut bm = BITMAP::default();
        unsafe { GetObjectW(ii.hbmColor.into(), core::mem::size_of::<BITMAP>() as i32, Some(&mut bm as *mut _ as *mut core::ffi::c_void)) };
        let (w, h) = (bm.bmWidth as u32, bm.bmHeight.unsigned_abs());
        let (mut bgra, _) = dib_bits(ii.hbmColor, w, h, 32);
        // Use the colour bitmap's own alpha if any; otherwise apply the AND mask (set pixel →
        // alpha 0xFF where the mask bit is clear, fully transparent where set), as native does.
        let has_alpha = bm.bmBitsPixel == 32 && bgra.chunks_exact(4).any(|p| p[3] != 0);
        if !has_alpha {
            let (mask, ms) = dib_bits(ii.hbmMask, w, h, 8);
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let p = (y * w as usize + x) * 4;
                    if mask.get(y * ms + x).copied().unwrap_or(0) != 0 {
                        bgra[p..p + 4].fill(0);
                    } else {
                        bgra[p + 3] = 0xFF;
                    }
                }
            }
        }
        unsafe {
            let _ = DeleteObject(ii.hbmColor.into());
            let _ = DeleteObject(ii.hbmMask.into());
        }
        let format = GUID::from_u128(0x6fddc324_4e03_4bfe_b185_3d77768dc90fu128); // 32bppBGRA
        Ok(BitmapObj::new(Source { format, width: w, height: h, stride: (w * 4) as usize, pixels: bgra, palette: Vec::new(), dpi: (96.0, 96.0) }).into())
        }
    }
    fn CreateComponentEnumerator(&self, types: u32, _o: u32) -> WResult<IEnumUnknown> {
        Ok(super::components::create_enumerator(types))
    }
    fn CreateFastMetadataEncoderFromDecoder(&self, d: Ref<IWICBitmapDecoder>) -> WResult<IWICFastMetadataEncoder> {
        use windows_core::Interface;
        // Native QIs the decoder for IWICMetadataBlockReader; only the GIF decoder implements it at
        // the container level, so every other decoder gets WINCODEC_ERR_UNSUPPORTEDOPERATION.
        let d = d.as_ref().ok_or(e(E_POINTER))?;
        // Decoder-scope fast encoding is GIF-only (native's block-reader QI); seed the decoder-level
        // reader (logscrdesc) and retain the writable stream for the in-place Commit.
        let obj = d.cast_object_ref::<super::decode::DecoderObj>().map_err(|_| e(WINCODEC_ERR_UNSUPPORTEDOPERATION))?;
        let container = obj.container.get();
        super::meta::reject_fast_encoder(container)?;
        let raw = obj.raw.borrow().clone();
        let reader = super::meta::decoder_query_reader(container, raw.clone()).ok();
        Ok(super::meta::fast_encoder(container, reader, obj.stream.borrow().clone(), raw, super::meta::FastScope::Decoder))
    }
    fn CreateFastMetadataEncoderFromFrameDecode(&self, f: Ref<IWICBitmapFrameDecode>) -> WResult<IWICFastMetadataEncoder> {
        use windows_core::Interface;
        // Frame-level block readers exist for every container (native accepts FromFrameDecode for
        // PNG/JPEG/TIFF/GIF — verified on the VM); non-GIF commits serialise nothing yet (a no-op,
        // matching native when no block is dirty).
        let f = f.as_ref().ok_or(e(E_POINTER))?;
        let obj = f.cast_object_ref::<super::decode::FrameObj>().map_err(|_| e(WINCODEC_ERR_UNSUPPORTEDOPERATION))?;
        let reader = super::meta::frame_query_reader(obj.container, obj.raw.clone(), obj.index).ok();
        Ok(super::meta::fast_encoder(obj.container, reader, obj.stream.clone(), obj.raw.clone(), super::meta::FastScope::Frame(obj.index)))
    }
    fn CreateQueryWriter(&self, m: *const GUID, _v: *const GUID) -> WResult<IWICMetadataQueryWriter> {
        Ok(super::meta::query_writer_empty(unsafe { *m }))
    }
    fn CreateQueryWriterFromReader(&self, r: Ref<IWICMetadataQueryReader>, _v: *const GUID) -> WResult<IWICMetadataQueryWriter> {
        super::meta::query_writer_from_reader(r.as_ref().ok_or(e(E_POINTER))?)
    }
}

/// `GetDIBits` a GDI bitmap into a top-down DIB at `bitcount` bpp; returns the bytes and row stride
/// (DWORD-aligned, as GDI produces). Used by CreateBitmapFromHBITMAP/HICON. Windows-only (GDI).
#[cfg(windows)]
fn dib_bits(hbm: windows::Win32::Graphics::Gdi::HBITMAP, w: u32, h: u32, bitcount: u16) -> (Vec<u8>, usize) {
    use windows::Win32::Graphics::Gdi::*;
    let stride = (((w * bitcount as u32 + 31) / 32) * 4) as usize;
    let mut bmi = BITMAPINFO::default();
    bmi.bmiHeader.biSize = core::mem::size_of::<BITMAPINFOHEADER>() as u32;
    bmi.bmiHeader.biWidth = w as i32;
    bmi.bmiHeader.biHeight = -(h as i32); // top-down
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = bitcount;
    bmi.bmiHeader.biCompression = 0; // BI_RGB
    let mut px = vec![0u8; stride * h as usize];
    unsafe {
        let hdc = GetDC(None);
        GetDIBits(hdc, hbm, 0, h, Some(px.as_mut_ptr() as *mut core::ffi::c_void), &mut bmi, DIB_RGB_COLORS);
        ReleaseDC(None, hdc);
    }
    (px, stride)
}

// --- colour context -----------------------------------------------------------------------------

/// `IWICColorContext`: holds an ICC profile (`InitializeFromMemory`/`FromFilename`) or an EXIF
/// colour-space id (`InitializeFromExifColorSpace`). Profile bytes are stored/returned verbatim, as
/// native does; the actual colour transform (`CreateColorTransformer`) is separate.
#[implement(IWICColorContext)]
#[derive(Default)]
pub(crate) struct ColorContextObj {
    kind: Cell<i32>, // WICColorContextType: 0 uninitialised, 1 profile, 2 EXIF colour space
    profile: RefCell<Vec<u8>>,
    exif: Cell<u32>,
}

impl IWICColorContext_Impl for ColorContextObj_Impl {
    fn InitializeFromFilename(&self, wzfilename: &windows_core::PCWSTR) -> WResult<()> {
        let path = unsafe { wzfilename.to_string() }.map_err(|_| e(E_INVALIDARG))?;
        *self.profile.borrow_mut() = std::fs::read(&path).map_err(|_| e(WINCODEC_ERR_WIN32ERROR))?;
        self.kind.set(1);
        Ok(())
    }
    fn InitializeFromMemory(&self, pb: *const u8, cb: u32) -> WResult<()> {
        *self.profile.borrow_mut() = unsafe { core::slice::from_raw_parts(pb, cb as usize) }.to_vec();
        self.kind.set(1);
        Ok(())
    }
    fn InitializeFromExifColorSpace(&self, value: u32) -> WResult<()> {
        self.exif.set(value);
        self.kind.set(2);
        Ok(())
    }
    fn GetType(&self) -> WResult<WICColorContextType> {
        Ok(WICColorContextType(self.kind.get()))
    }
    fn GetProfileBytes(&self, cb: u32, pb: *mut u8, actual: *mut u32) -> WResult<()> {
        let p = self.profile.borrow();
        if !actual.is_null() {
            unsafe { *actual = p.len() as u32 };
        }
        if cb == 0 {
            return Ok(());
        }
        if (cb as usize) < p.len() || pb.is_null() {
            return Err(e(E_INVALIDARG));
        }
        unsafe { core::ptr::copy_nonoverlapping(p.as_ptr(), pb, p.len()) };
        Ok(())
    }
    fn GetExifColorSpace(&self) -> WResult<u32> {
        Ok(self.exif.get())
    }
}

// --- palette ------------------------------------------------------------------------------------

/// The 16 standard VGA/HTML colours (`0xAARRGGBB`) WIC appends to
/// the fixed halftone palettes (skipping any already in the colour cube).
const VGA16: [u32; 16] = [
    0xFF000000, 0xFFFFFFFF, 0xFFC0C0C0, 0xFF808080, 0xFF800000, 0xFFFF0000, 0xFF008000, 0xFF00FF00,
    0xFF000080, 0xFF0000FF, 0xFF808000, 0xFFFFFF00, 0xFF800080, 0xFFFF00FF, 0xFF008080, 0xFF00FFFF,
];

/// The RGB cube dimensions per `WICBitmapPaletteTypeFixedHalftone*` (from `rgPaletteAttributes`).
fn halftone_dims(ty: i32) -> (u32, u32, u32) {
    match ty {
        3 => (2, 2, 2),
        4 => (3, 3, 3),
        5 => (4, 4, 4),
        6 => (5, 5, 5),
        7 => (6, 6, 6),
        8 => (6, 7, 6),
        9 => (8, 8, 4),
        _ => (0, 0, 0),
    }
}

/// The exact colour table for a `WICBitmapPaletteTypeFixed*`, computed as native's
/// WIC's predefined-palette builder does: grey ramps, or an RGB cube (`round(255·i/(d-1))`) with the
/// 16 VGA colours unique-appended for Halftone8..216, plus an optional trailing transparent entry.
fn predefined_palette(ty: i32, addtrans: bool) -> Option<Vec<u32>> {
    let grey = |v: u32| 0xFF00_0000u32 | (v << 16) | (v << 8) | v;
    let mut c: Vec<u32> = match ty {
        2 => vec![0xFF00_0000, 0xFFFF_FFFF],   // FixedBW
        10 => (0..4).map(|i| grey(i * 85)).collect(), // FixedGray4
        11 => (0..16).map(|i| grey(i * 17)).collect(), // FixedGray16
        12 => (0..256).map(grey).collect(),    // FixedGray256
        3..=9 => {
            let (d1, d2, d3) = halftone_dims(ty);
            let step = |x: u32, d: u32| ((255.0f32 / (d - 1) as f32) * x as f32 + 0.5) as u32;
            let mut v = Vec::new();
            for i in 0..d1 {
                for j in 0..d2 {
                    for k in 0..d3 {
                        v.push(0xFF00_0000 | (step(i, d1) << 16) | (step(j, d2) << 8) | step(k, d3));
                    }
                }
            }
            if matches!(ty, 3..=7) {
                for &col in &VGA16 {
                    if v.len() < 256 && !v.contains(&col) {
                        v.push(col);
                    }
                }
            }
            v
        }
        _ => return None,
    };
    if addtrans && matches!(ty, 3..=9) {
        if c.len() >= 256 {
            c.pop();
        }
        c.push(0);
    }
    Some(c)
}

#[implement(IWICPalette)]
#[derive(Default)]
pub(crate) struct PaletteObj {
    pub(crate) colors: RefCell<Vec<u32>>,
    /// The palette type reported by `GetType` (`WICBitmapPaletteTypeCustom` = 0 until initialised).
    ty: Cell<i32>,
}

impl IWICPalette_Impl for PaletteObj_Impl {
    fn InitializePredefined(&self, t: WICBitmapPaletteType, a: windows_core::BOOL) -> WResult<()> {
        let colors = predefined_palette(t.0, a.as_bool()).ok_or(e(E_INVALIDARG))?;
        *self.colors.borrow_mut() = colors;
        self.ty.set(t.0);
        Ok(())
    }
    fn InitializeCustom(&self, colors: *const u32, count: u32) -> WResult<()> {
        *self.colors.borrow_mut() = unsafe { core::slice::from_raw_parts(colors, count as usize) }.to_vec();
        self.ty.set(WICBitmapPaletteTypeCustom.0);
        Ok(())
    }
    fn InitializeFromBitmap(&self, s: Ref<IWICBitmapSource>, count: u32, addtrans: windows_core::BOOL) -> WResult<()> {
        // WIC's palette-from-bitmap: the colour count must be 2..=256; with the
        // transparent-entry option the median-cut target drops by one and a transparent entry is
        // appended afterwards. The source is converted to 24bpp BGR (the scan pipeline's format 12)
        // and every pixel fed to the median-cut quantiser.
        if !(2..=256).contains(&count) {
            return Err(e(E_INVALIDARG));
        }
        let src = read_source(s.as_ref().ok_or(e(E_POINTER))?)?;
        let sf = Format::from_guid(&guid_to_str(&src.format)).ok_or(e(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT))?;
        let bgr = convert(sf, Format::Bgr24, &src.pixels, src.width, src.height, &src.palette).map_err(wicerr)?;
        let add = addtrans.as_bool();
        let mut colors = median_cut(&bgr, (if add { count - 1 } else { count }) as usize);
        if add {
            colors.push(0);
        }
        *self.colors.borrow_mut() = colors;
        self.ty.set(WICBitmapPaletteTypeCustom.0);
        Ok(())
    }
    fn InitializeFromPalette(&self, pal: Ref<IWICPalette>) -> WResult<()> {
        let pal = pal.as_ref().ok_or(e(E_POINTER))?;
        let n = unsafe { pal.GetColorCount()? };
        let mut colors = vec![0u32; n as usize];
        let mut got = 0u32;
        unsafe { pal.GetColors(&mut colors, &mut got)? };
        colors.truncate(got as usize);
        *self.colors.borrow_mut() = colors;
        self.ty.set(unsafe { pal.GetType() }.map(|t| t.0).unwrap_or(WICBitmapPaletteTypeCustom.0));
        Ok(())
    }
    fn GetType(&self) -> WResult<WICBitmapPaletteType> {
        Ok(WICBitmapPaletteType(self.ty.get()))
    }
    fn GetColorCount(&self) -> WResult<u32> {
        Ok(self.colors.borrow().len() as u32)
    }
    fn GetColors(&self, count: u32, pcolors: *mut u32, actual: *mut u32) -> WResult<()> {
        let colors = self.colors.borrow();
        let n = (count as usize).min(colors.len());
        unsafe {
            core::ptr::copy_nonoverlapping(colors.as_ptr(), pcolors, n);
            *actual = n as u32;
        }
        Ok(())
    }
    fn IsBlackWhite(&self) -> WResult<windows_core::BOOL> {
        let c = self.colors.borrow();
        Ok((!c.is_empty() && c.iter().all(|&x| x == 0xFF00_0000 || x == 0xFFFF_FFFF)).into())
    }
    fn IsGrayscale(&self) -> WResult<windows_core::BOOL> {
        let c = self.colors.borrow();
        Ok((!c.is_empty() && c.iter().all(|&x| x >> 24 == 0xFF && (x >> 16) & 0xFF == (x >> 8) & 0xFF && (x >> 8) & 0xFF == x & 0xFF)).into())
    }
    fn HasAlpha(&self) -> WResult<windows_core::BOOL> {
        Ok(self.colors.borrow().iter().any(|c| c >> 24 != 0xff).into())
    }
}

// --- in-memory bitmap ---------------------------------------------------------------------------

#[implement(IWICBitmap)]
pub(crate) struct BitmapObj {
    // `Rc<RefCell>` so an `IWICBitmapLock` can keep the pixel buffer alive and hand out a stable
    // raw pointer into it (the WIC lock contract is direct in-place read/write access).
    pub(crate) src: Rc<RefCell<Option<Source>>>,
    locks: Rc<Cell<i32>>,
}

impl BitmapObj {
    pub(crate) fn new(src: Source) -> Self {
        BitmapObj { src: Rc::new(RefCell::new(Some(src))), locks: Rc::new(Cell::new(0)) }
    }
}

impl_source!(BitmapObj_Impl, this, if this.locks.get() < 0 { Err(e(WINCODEC_ERR_ALREADYLOCKED)) } else { Ok(()) });

impl IWICBitmap_Impl for BitmapObj_Impl {
    fn Lock(&self, rc: *const WICRect, flags: u32) -> WResult<IWICBitmapLock> {
        if flags & 3 == 0 { return Err(e(E_INVALIDARG)); }
        let write = flags & 2 != 0;
        let count = self.locks.get();
        if count < 0 || (write && count != 0) { return Err(e(WINCODEC_ERR_ALREADYLOCKED)); }
        let b = self.src.borrow();
        let s = b.as_ref().ok_or(e(WINCODEC_ERR_NOTINITIALIZED))?;
        let (x,y,w,h) = super::pixel_rect(rc,s.width,s.height)?;
        let bpp = bpp_of(&s.format) as usize;
        let temporary = if x as usize * bpp & 7 != 0 {
            // Native WIC stages an unaligned bitmap lock's packed pixels in a DWORD-strided buffer.
            let stride = (w as usize * bpp + 31) / 32 * 4;
            let mut pixels = vec![0u8;stride*h as usize];
            for row in 0..h as usize {
                super::copy_bits(&s.pixels[(y as usize+row)*s.stride..],x as usize*bpp,&mut pixels[row*stride..],0,w as usize*bpp);
            }
            Some((pixels,stride))
        } else { None };
        self.locks.set(if write { -1 } else { count+1 });
        Ok(BitmapLockObj { src:self.src.clone(), locks:self.locks.clone(), write, temporary, x,y,w,h }.into())
    }
    fn SetPalette(&self, pal: Ref<IWICPalette>) -> WResult<()> {
        let pal = pal.as_ref().ok_or(e(E_POINTER))?;
        let n = unsafe { pal.GetColorCount()? };
        let mut colors = vec![0u32; n as usize];
        let mut got = 0u32;
        unsafe { pal.GetColors(&mut colors, &mut got)? };
        colors.truncate(got as usize);
        if let Some(s) = self.src.borrow_mut().as_mut() {
            s.palette = colors;
        }
        Ok(())
    }
    fn SetResolution(&self, dpix: f64, dpiy: f64) -> WResult<()> {
        if let Some(s) = self.src.borrow_mut().as_mut() {
            s.dpi = (dpix, dpiy);
        }
        Ok(())
    }
}

/// `IWICBitmapLock`: direct access to a (sub-)rectangle of a bitmap's pixels. Holds the bitmap's
/// `Rc` so the buffer outlives the lock; `GetDataPointer` returns a raw pointer into it.
#[implement(IWICBitmapLock)]
pub(crate) struct BitmapLockObj {
    src: Rc<RefCell<Option<Source>>>,
    locks: Rc<Cell<i32>>,
    write: bool,
    temporary: Option<(Vec<u8>,usize)>,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

impl Drop for BitmapLockObj {
    fn drop(&mut self) {
        if self.write {
            if let Some((pixels,stride)) = &self.temporary {
                if let Some(s) = self.src.borrow_mut().as_mut() {
                    let bpp = bpp_of(&s.format) as usize;
                    for row in 0..self.h as usize {
                        super::copy_bits(&pixels[row*stride..],0,&mut s.pixels[(self.y as usize+row)*s.stride..],self.x as usize*bpp,self.w as usize*bpp);
                    }
                }
            }
            self.locks.set(0);
        } else { self.locks.set(self.locks.get()-1); }
    }
}

impl IWICBitmapLock_Impl for BitmapLockObj_Impl {
    fn GetSize(&self, pw: *mut u32, ph: *mut u32) -> WResult<()> {
        unsafe {
            *pw = self.w;
            *ph = self.h;
        }
        Ok(())
    }
    fn GetStride(&self) -> WResult<u32> {
        Ok(self.temporary.as_ref().map_or(self.src.borrow().as_ref().ok_or(e(WINCODEC_ERR_NOTINITIALIZED))?.stride,|(_,stride)|*stride) as u32)
    }
    fn GetPixelFormat(&self) -> WResult<GUID> {
        Ok(self.src.borrow().as_ref().ok_or(e(WINCODEC_ERR_NOTINITIALIZED))?.format)
    }
    fn GetDataPointer(&self, pcb: *mut u32, ppb: *mut *mut u8) -> WResult<()> {
        if pcb.is_null() || ppb.is_null() { return Err(e(E_INVALIDARG)); }
        if let Some((pixels,_)) = &self.temporary {
            unsafe { *pcb=pixels.len() as u32; *ppb=pixels.as_ptr() as *mut u8; }
            return Ok(());
        }
        let b = self.src.borrow();
        let s = b.as_ref().ok_or(e(WINCODEC_ERR_NOTINITIALIZED))?;
        let bpp = bpp_of(&s.format) as usize;
        let off = self.y as usize * s.stride + self.x as usize * bpp / 8;
        let row = (self.w as usize * bpp + 7) / 8;
        let size = s.stride * self.h.saturating_sub(1) as usize + row;
        unsafe {
            *ppb = s.pixels.as_ptr().add(off) as *mut u8;
            *pcb = size as u32;
        }
        Ok(())
    }
}

/// Read an entire `IStream` into a `Vec` (seek to the start first). Used by the stream-fed decoders.
pub(crate) fn read_stream(stream: &IStream) -> WResult<Vec<u8>> {
    unsafe {
        let mut size = 0u64;
        stream.Seek(0, STREAM_SEEK_END, Some(&mut size))?;
        stream.Seek(0, STREAM_SEEK_SET, None)?;
        let mut buf = vec![0u8; size as usize];
        let mut read = 0u32;
        stream.Read(buf.as_mut_ptr() as *mut _, size as u32, Some(&mut read)).ok()?;
        buf.truncate(read as usize);
        Ok(buf)
    }
}
