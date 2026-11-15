//! `IWICBitmapEncoder` + `IWICBitmapFrameEncode` for every container our core encodes byte-exact:
//! JPEG, PNG, TIFF, GIF and BMP. The frame accumulates size / pixel format / palette / pixels, then
//! `Commit` dispatches to the matching `d2d::wic::encode_*` (native-default options) and writes the
//! file to the stream. JPEG additionally serialises its metadata query writer (EXIF/COM).

use super::{bpp_of, e, guid_to_str, read_source, wicerr};
use d2d::wic::convert::{Format, convert};
use d2d::wic::{Decoded, JxrEncodeOptions, Subsampling, WicFormat, encode_bmp, encode_gif_meta, encode_jpeg, encode_jxr, encode_png_meta, encode_tiff_meta, xmp_packet};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Imaging::*;
use windows::Win32::System::Com::{IErrorLog, IStream};
use windows::Win32::System::Com::StructuredStorage::{IPropertyBag2, IPropertyBag2_Impl, PROPBAG2};
use windows::Win32::System::Variant::VARIANT;
use windows_core::{GUID, HRESULT, IUnknown, Interface, PCWSTR, Ref, Result as WResult, implement};

const JPEG_CONTAINER: &str = "19e4a5aa-5662-4fc5-a0c0-1758028e1057";
const PNG_CONTAINER: &str = "1b7cfaf4-713f-473c-bbcd-6137425faeaf";
const TIFF_CONTAINER: &str = "163bcc30-e2e9-4f0b-961d-a3e9fdb788a3";
const GIF_CONTAINER: &str = "1f8a5601-7d4d-4cbd-9c82-1bc8d4eeb9a5";
const BMP_CONTAINER: &str = "0af1d87e-fcfe-4188-bdeb-a7906471cbe3";
const WMP_CONTAINER: &str = "57a37caa-367a-4540-916b-f183c5093a4b"; // JPEG-XR / HD Photo
const DDS_CONTAINER: &str = "9967cb95-2e85-4ac8-8ca2-83d7ccd425c9";

/// The containers whose `encode_*` core we wire (verified byte-exact against native with its default
/// options). Others fall through to `WINCODEC_ERR_COMPONENTNOTFOUND`, as native's `CreateEncoder`
/// does for a codec without an encoder (ICO is decode-only).
fn supported(container: &GUID) -> bool {
    matches!(
        guid_to_str(container).as_str(),
        JPEG_CONTAINER | PNG_CONTAINER | TIFF_CONTAINER | GIF_CONTAINER | BMP_CONTAINER | WMP_CONTAINER | DDS_CONTAINER
    )
}

/// The 16 raw little-endian GUID bytes (data1/2/3 little-endian, data4 as-is) — the pixel-format key
/// `d2d::wic::encode_jxr` expects.
fn guid_le_bytes(g: &GUID) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[0..4].copy_from_slice(&g.data1.to_le_bytes());
    b[4..6].copy_from_slice(&g.data2.to_le_bytes());
    b[6..8].copy_from_slice(&g.data3.to_le_bytes());
    b[8..16].copy_from_slice(&g.data4);
    b
}

/// A scalar `VARIANT` payload in one of the four VARTYPEs the JPEG-XR encoder options use, read from
/// the raw layout (`vt` at offset 0, union at offset 8): `VT_R4`=4, `VT_BOOL`=11, `VT_UI1`=17,
/// `VT_UI2`=18 — the exact types native validates.
enum Scalar {
    F32(f32),
    Bool(bool),
    U8(u8),
    U16(u16),
}

unsafe fn variant_scalar(v: *const VARIANT) -> Option<Scalar> {
    #[repr(C)]
    union Val {
        b: u8,
        i: i16,
        u: u16,
        f: f32,
        _q: u64,
    }
    #[repr(C)]
    struct Raw {
        vt: u16,
        _r: [u16; 3],
        val: Val,
    }
    unsafe {
        let r = &*(v as *const Raw);
        Some(match r.vt {
            4 => Scalar::F32(r.val.f),
            11 => Scalar::Bool(r.val.i != 0),
            17 => Scalar::U8(r.val.b),
            18 => Scalar::U16(r.val.u),
            _ => return None,
        })
    }
}

/// The JPEG-XR frame's encoder-options property bag. `IWICBitmapEncoder::CreateNewFrame` hands one to
/// the caller, who writes options into it and passes it to `IWICBitmapFrameEncode::Initialize`; each
/// `Write` parses the option into the shared [`JxrEncodeOptions`] the frame reads at `Commit`.
#[implement(IPropertyBag2)]
pub(crate) struct JxrOptionBag {
    opts: Rc<RefCell<JxrEncodeOptions>>,
}

impl IPropertyBag2_Impl for JxrOptionBag_Impl {
    fn Write(&self, c: u32, pbag: *const PROPBAG2, pvar: *const VARIANT) -> WResult<()> {
        let mut o = self.opts.borrow_mut();
        for i in 0..c as usize {
            let pb = unsafe { &*pbag.add(i) };
            let name = unsafe { pb.pstrName.to_string() }.unwrap_or_default();
            let Some(s) = (unsafe { variant_scalar(pvar.add(i)) }) else { continue };
            match (name.as_str(), s) {
                ("ImageQuality", Scalar::F32(f)) => o.image_quality = f,
                ("Lossless", Scalar::Bool(b)) => o.lossless = b,
                ("HorizontalTileSlices", Scalar::U16(u)) => o.h_tile_slices = u,
                ("VerticalTileSlices", Scalar::U16(u)) => o.v_tile_slices = u,
                // Overlap / Subsampling / FrequencyOrder / InterleavedAlpha / UseCodecOptions (and
                // AlphaQuality, …) are accepted but no-ops at the default the committed oracle uses.
                _ => {}
            }
        }
        Ok(())
    }
    fn Read(&self, _c: u32, _pbag: *const PROPBAG2, _err: Ref<IErrorLog>, _pvar: *mut VARIANT, _phr: *mut HRESULT) -> WResult<()> {
        Err(e(E_NOTIMPL))
    }
    fn CountProperties(&self) -> WResult<u32> {
        Ok(0)
    }
    fn GetPropertyInfo(&self, _i: u32, _c: u32, _pbag: *mut PROPBAG2, _pc: *mut u32) -> WResult<()> {
        Err(e(E_NOTIMPL))
    }
    fn LoadObject(&self, _name: &PCWSTR, _hint: u32, _obj: Ref<IUnknown>, _err: Ref<IErrorLog>) -> WResult<()> {
        Err(e(E_NOTIMPL))
    }
}

/// Splice metadata segments into a JPEG right after the APP0 JFIF marker — where native inserts
/// APP1 `Exif` / COM (before the first DQT). With no APP0, they go straight after SOI.
fn insert_after_app0(base: Vec<u8>, segs: &[u8]) -> Vec<u8> {
    if segs.is_empty() {
        return base;
    }
    let pos = if base.len() >= 6 && base[2] == 0xFF && base[3] == 0xE0 {
        4 + u16::from_be_bytes([base[4], base[5]]) as usize
    } else {
        2
    };
    let mut out = Vec::with_capacity(base.len() + segs.len());
    out.extend_from_slice(&base[..pos]);
    out.extend_from_slice(segs);
    out.extend_from_slice(&base[pos..]);
    out
}

/// Splice ancillary chunks into a PNG right before the first IDAT — where native inserts tEXt/tIME
/// (after IHDR/sRGB/gAMA).
fn insert_png_chunks(png: Vec<u8>, chunks: &[u8]) -> Vec<u8> {
    if chunks.is_empty() {
        return png;
    }
    let mut o = 8;
    while o + 8 <= png.len() {
        let ln = u32::from_be_bytes([png[o], png[o + 1], png[o + 2], png[o + 3]]) as usize;
        if &png[o + 4..o + 8] == b"IDAT" {
            break;
        }
        o += 12 + ln;
    }
    let mut out = Vec::with_capacity(png.len() + chunks.len());
    out.extend_from_slice(&png[..o]);
    out.extend_from_slice(chunks);
    out.extend_from_slice(&png[o..]);
    out
}

/// The WIC pixel format our JPEG encoder accepts (24bppBGR → colour, 8bppGray → grayscale).

#[implement(IWICBitmapEncoder)]
pub(crate) struct EncoderObj {
    stream: RefCell<Option<IStream>>,
    container: GUID,
}

impl EncoderObj {
    pub(crate) fn for_container(container: GUID) -> WResult<IWICBitmapEncoder> {
        if supported(&container) {
            Ok(EncoderObj { stream: RefCell::new(None), container }.into())
        } else {
            // A container with no registered encoder (e.g. ICO — Windows ships only an ICO *decoder*):
            // WIC's encoder factory returns
            // `WINCODEC_ERR_COMPONENTNOTFOUND` (0x88982f50, -2003292336) and propagates it — NOT
            // UNSUPPORTEDOPERATION (0x88982f81), which is what we returned before.
            Err(e(WINCODEC_ERR_COMPONENTNOTFOUND))
        }
    }
}

impl IWICBitmapEncoder_Impl for EncoderObj_Impl {
    fn Initialize(&self, stream: Ref<IStream>, _cache: WICBitmapEncoderCacheOption) -> WResult<()> {
        *self.stream.borrow_mut() = Some(stream.as_ref().ok_or(e(E_POINTER))?.clone());
        Ok(())
    }
    fn GetContainerFormat(&self) -> WResult<GUID> {
        Ok(self.container)
    }
    fn GetEncoderInfo(&self) -> WResult<IWICBitmapEncoderInfo> {
        super::components::encoder_info(&self.container)
    }
    fn SetColorContexts(&self, _c: u32, _p: *const Option<IWICColorContext>) -> WResult<()> {
        Ok(())
    }
    fn SetPalette(&self, _p: Ref<IWICPalette>) -> WResult<()> {
        Ok(())
    }
    fn SetThumbnail(&self, _t: Ref<IWICBitmapSource>) -> WResult<()> {
        Ok(())
    }
    fn SetPreview(&self, _p: Ref<IWICBitmapSource>) -> WResult<()> {
        Ok(())
    }
    fn CreateNewFrame(&self, frame: windows_core::OutRef<IWICBitmapFrameEncode>, options: windows_core::OutRef<IPropertyBag2>) -> WResult<()> {
        let stream = self.stream.borrow().clone().ok_or(e(WINCODEC_ERR_NOTINITIALIZED))?;
        // JPEG-XR takes encoder options through the property bag; give the frame and the bag a shared
        // options cell so the caller's writes reach `Commit`. Other containers have no options.
        let opts = (guid_to_str(&self.container) == WMP_CONTAINER)
            .then(|| Rc::new(RefCell::new(JxrEncodeOptions::default())));
        frame.write(Some(FrameEncodeObj::new(stream, self.container, opts.clone()).into()))?;
        match &opts {
            Some(o) => options.write(Some(JxrOptionBag { opts: o.clone() }.into()))?,
            None => options.write(None)?,
        }
        Ok(())
    }
    fn Commit(&self) -> WResult<()> {
        Ok(())
    }
    fn GetMetadataQueryWriter(&self) -> WResult<IWICMetadataQueryWriter> {
        Ok(super::meta::query_writer_empty(self.container))
    }
}

#[implement(IWICBitmapFrameEncode)]
pub(crate) struct FrameEncodeObj {
    stream: IStream,
    container: GUID,
    size: Cell<(u32, u32)>,
    format: Cell<GUID>,
    dpi: Cell<(f64, f64)>,
    pixels: RefCell<Vec<u8>>,
    palette: RefCell<Vec<u32>>,
    /// Persistent query writer: edits made through `GetMetadataQueryWriter` accumulate here and are
    /// serialised into the output on `Commit` (JPEG only so far).
    meta: IWICMetadataQueryWriter,
    /// JPEG-XR encoder options, shared with the property bag from `CreateNewFrame` (`None` otherwise).
    jxr_opts: Option<Rc<RefCell<JxrEncodeOptions>>>,
}

impl FrameEncodeObj {
    fn new(stream: IStream, container: GUID, jxr_opts: Option<Rc<RefCell<JxrEncodeOptions>>>) -> Self {
        FrameEncodeObj { stream, container, size: Cell::new((0, 0)), format: Cell::new(GUID::zeroed()), dpi: Cell::new((0.0, 0.0)), pixels: RefCell::new(Vec::new()), palette: RefCell::new(Vec::new()), meta: super::meta::query_writer_empty(container), jxr_opts }
    }
}

impl IWICBitmapFrameEncode_Impl for FrameEncodeObj_Impl {
    fn Initialize(&self, _options: Ref<IPropertyBag2>) -> WResult<()> {
        Ok(())
    }
    fn SetSize(&self, w: u32, h: u32) -> WResult<()> {
        self.size.set((w, h));
        Ok(())
    }
    fn SetResolution(&self, dpix: f64, dpiy: f64) -> WResult<()> {
        self.dpi.set((dpix, dpiy));
        Ok(())
    }
    fn SetPixelFormat(&self, ppixelformat: *mut GUID) -> WResult<()> {
        // Accept the requested format if we encode it; otherwise leave it for the caller to see.
        self.format.set(unsafe { *ppixelformat });
        Ok(())
    }
    fn SetColorContexts(&self, _c: u32, _p: *const Option<IWICColorContext>) -> WResult<()> {
        Ok(())
    }
    fn SetPalette(&self, pal: Ref<IWICPalette>) -> WResult<()> {
        let pal = pal.as_ref().ok_or(e(E_POINTER))?;
        let n = unsafe { pal.GetColorCount()? };
        let mut colors = vec![0u32; n as usize];
        let mut got = 0u32;
        unsafe { pal.GetColors(&mut colors, &mut got)? };
        colors.truncate(got as usize);
        *self.palette.borrow_mut() = colors;
        Ok(())
    }
    fn SetThumbnail(&self, _t: Ref<IWICBitmapSource>) -> WResult<()> {
        Ok(())
    }
    fn WritePixels(&self, _line_count: u32, _stride: u32, _buffer_size: u32, pixels: *const u8) -> WResult<()> {
        let (w, h) = self.size.get();
        let bpp = bpp_of(&self.format.get());
        let tight = ((w * bpp + 7) / 8) as usize;
        let len = tight * h as usize;
        *self.pixels.borrow_mut() = unsafe { core::slice::from_raw_parts(pixels, len) }.to_vec();
        Ok(())
    }
    fn WriteSource(&self, source: Ref<IWICBitmapSource>, _rc: *const WICRect) -> WResult<()> {
        let s = read_source(source.as_ref().ok_or(e(E_POINTER))?)?;
        self.format.set(s.format);
        if self.size.get() == (0, 0) {
            self.size.set((s.width, s.height));
        }
        self.dpi.set(s.dpi);
        if !s.palette.is_empty() {
            *self.palette.borrow_mut() = s.palette;
        }
        *self.pixels.borrow_mut() = s.pixels;
        Ok(())
    }
    fn Commit(&self) -> WResult<()> {
        let (w, h) = self.size.get();
        let fmtguid = self.format.get();
        let bytes = if guid_to_str(&self.container) == WMP_CONTAINER {
            let opts = self.jxr_opts.as_ref().ok_or(e(WINCODEC_ERR_NOTINITIALIZED))?.borrow();
            let stride = ((w * bpp_of(&fmtguid) + 7) / 8) as usize;
            let (dx, dy) = self.dpi.get();
            encode_jxr(&guid_le_bytes(&fmtguid), w, h, &self.pixels.borrow(), stride, &opts, (dx as f32, dy as f32)).map_err(wicerr)?
        } else if guid_to_str(&self.container) == JPEG_CONTAINER {
            // Native's JPEG `WriteSource` converts the source to the encoder's input format via an
            // IWICFormatConverter — 8bppGray for a gray source, else 24bppBGR — then entropy-codes
            // (byte-exact). Replicate: pick the target, convert (a no-op for a source already in it),
            // encode. This accepts any convertible source (e.g. BGR555/BGR565/RGBA64 → 24bppBGR).
            let src = Format::from_guid(&guid_to_str(&fmtguid)).ok_or(e(WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT))?;
            let gray = fmtguid.data1 == 0x6fddc324 && matches!(fmtguid.data4[7], 0x05 | 0x06 | 0x07 | 0x08 | 0x0b);
            let (cf, wf) = if gray { (Format::Gray8, WicFormat::Gray8) } else { (Format::Bgr24, WicFormat::Bgr24) };
            let px = convert(src, cf, &self.pixels.borrow(), w, h, &self.palette.borrow()).map_err(wicerr)?;
            let (dx, dy) = self.dpi.get();
            let base = encode_jpeg(wf, w, h, &px, 90, Subsampling::S420, (dx as u16, dy as u16));
            insert_after_app0(base, &super::meta::jpeg_metadata_segments(&self.meta.cast()?))
        } else {
            let bpp = bpp_of(&fmtguid);
            let img = Decoded {
                width: w,
                height: h,
                format: guid_to_str(&fmtguid),
                stride: ((w * bpp + 7) / 8) as usize,
                pixels: self.pixels.borrow().clone(),
                palette: self.palette.borrow().clone(),
            };
            let base = match guid_to_str(&self.container).as_str() {
                PNG_CONTAINER => {
                    let xmp = xmp_packet(&super::meta::xmp_props(&self.meta.cast()?));
                    encode_png_meta(&img, xmp.as_deref())
                }
                TIFF_CONTAINER => encode_tiff_meta(&img, &super::meta::tiff_ifd_fields(&self.meta.cast()?)),
                GIF_CONTAINER => encode_gif_meta(&img, &super::meta::gif_meta(&self.meta.cast()?)),
                BMP_CONTAINER => encode_bmp(&img, self.dpi.get().0 as f32),
                DDS_CONTAINER => d2d::wic::dds_encode::encode_dds_bc3(&img),
                _ => return Err(e(WINCODEC_ERR_UNSUPPORTEDOPERATION)),
            }
            .map_err(wicerr)?;
            if guid_to_str(&self.container) == PNG_CONTAINER {
                insert_png_chunks(base, &super::meta::png_metadata_chunks(&self.meta.cast()?))
            } else {
                base
            }
        };
        let mut written = 0u32;
        unsafe { self.stream.Write(bytes.as_ptr() as *const _, bytes.len() as u32, Some(&mut written)) }.ok()?;
        Ok(())
    }
    fn GetMetadataQueryWriter(&self) -> WResult<IWICMetadataQueryWriter> {
        Ok(self.meta.clone())
    }
}
