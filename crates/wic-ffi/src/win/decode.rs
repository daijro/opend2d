//! `IWICBitmapDecoder` + `IWICBitmapFrameDecode`: sniff the container signature, decode with the
//! matching `d2d::wic` decoder, and expose each frame as an `IWICBitmapSource`. JPEG frames also
//! expose `IWICBitmapSourceTransform` for reduced-size (scaled) decoding.

use super::factory::read_stream;
use super::{Source, e, ecode, guid_from_str, impl_source, wicerr};
use d2d::wic::{
    DecodedImage, Decoded, DdsFrame, decode_bmp, decode_dds, decode_gif, decode_ico, decode_jpeg,
    decode_jpeg_scaled, decode_png, decode_tiff, jpeg_closest_scale,
};
#[cfg(feature = "jxr")]
use d2d::wic::decode_jxr;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Imaging::*;
use windows::Win32::System::Com::{IStream, STREAM_SEEK_CUR, STREAM_SEEK_SET};
use windows_core::{GUID, Interface, Ref, Result as WResult, implement};

/// Container-format GUIDs (what `GetContainerFormat` reports), by sniffed signature.
fn container_guid(b: &[u8]) -> GUID {
    // JPEG-XR (WMP) sniffing is gated on the `jxr` feature; without it a JPEG-XR stream is an
    // unrecognised container, so the decoder factory reports codec-not-installed.
    #[cfg(feature = "jxr")]
    if b.starts_with(b"II\xbc\x00") || b.starts_with(b"II\xbc\x01") {
        return GUID::from_u128(0x57a37caa_367a_4540_916b_f183c5093a4b);
    }
    let u = if b.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]) {
        0x1b7cfaf4_713f_473c_bbcd_6137425faeaf
    } else if b.starts_with(&[0xff, 0xd8]) {
        0x19e4a5aa_5662_4fc5_a0c0_1758028e1057
    } else if b.starts_with(b"BM") {
        0x0af1d87e_fcfe_4188_bdeb_a7906471cbe3
    } else if b.starts_with(b"GIF8") {
        0x1f8a5601_7d4d_4cbd_9c82_1bc8d4eeb9a5
    } else if b.starts_with(&[0, 0, 1, 0]) || b.starts_with(&[0, 0, 2, 0]) {
        0xa3a860c4_338f_4c17_919a_fba4b5628f21
    } else if b.starts_with(b"II\x2a\x00") || b.starts_with(b"MM\x00\x2a") {
        0x163bcc30_e2e9_4f0b_961d_a3e9fdb788a3
    } else if b.starts_with(b"DDS ") {
        0x9967cb95_2e85_4ac8_8ca2_83d7ccd425c9
    } else {
        0
    };
    GUID::from_u128(u)
}

/// One decoded frame plus, for JPEG, the raw codestream the source-transform re-decodes at scale.
#[derive(Clone)]
struct FrameState {
    src: Source,
    jpeg: Option<Vec<u8>>,
    full: (u32, u32),
}

impl From<DecodedImage> for Source {
    fn from(i: DecodedImage) -> Source {
        Source { format: guid_from_str(&i.format.guid()), width: i.width, height: i.height, stride: i.stride, pixels: i.pixels, palette: i.palette, dpi: (0.0, 0.0) }
    }
}
impl From<Decoded> for Source {
    fn from(i: Decoded) -> Source {
        Source { format: guid_from_str(&i.format), width: i.width, height: i.height, stride: i.stride, pixels: i.pixels, palette: i.palette, dpi: (0.0, 0.0) }
    }
}
impl From<DdsFrame> for Source {
    fn from(i: DdsFrame) -> Source {
        Source { format: guid_from_str(&i.format.guid()), width: i.width, height: i.height, stride: i.stride, pixels: i.pixels, palette: Vec::new(), dpi: (0.0, 0.0) }
    }
}

/// Sniff the container and decode every frame — or the exact `WINCODEC_ERR_*` the native codec
/// reports (an unknown signature is `WINCODEC_ERR_COMPONENTNOTFOUND`).
fn decode_all(b: &[u8]) -> WResult<Vec<FrameState>> {
    decode_selected(container_guid(b), b)
}

fn decode_selected(container: GUID, b: &[u8]) -> WResult<Vec<FrameState>> {
    let one = |s: Source, jpeg: Option<Vec<u8>>| vec![FrameState { full: (s.width, s.height), src: s, jpeg }];
    let plain = |v: Vec<Source>| v.into_iter().map(|s| FrameState { full: (s.width, s.height), src: s, jpeg: None }).collect::<Vec<_>>();

    if container == GUID_ContainerFormatPng {
        Ok(one(decode_png(b).map_err(wicerr)?.into(), None))
    } else if container == GUID_ContainerFormatJpeg {
        Ok(one(decode_jpeg(b).map_err(ecode)?.into(), Some(b.to_vec())))
    } else if container == GUID_ContainerFormatBmp {
        Ok(one(decode_bmp(b).map_err(wicerr)?.into(), None))
    } else if container == GUID_ContainerFormatGif {
        Ok(plain(decode_gif(b).map_err(wicerr)?.into_iter().map(Source::from).collect()))
    } else if container == GUID_ContainerFormatIco {
        Ok(plain(decode_ico(b).map_err(wicerr)?.into_iter().map(Source::from).collect()))
    } else if container == GUID_ContainerFormatTiff {
        Ok(plain(decode_tiff(b).map_err(wicerr)?.into_iter().map(Source::from).collect()))
    } else if container == GUID_ContainerFormatDds {
        Ok(plain(decode_dds(b).map_err(wicerr)?.into_iter().map(Source::from).collect()))
    } else {
        // JPEG-XR (WMP) decode is gated on the `jxr` feature; without it the container is
        // unrecognised and reports codec-not-installed.
        #[cfg(feature = "jxr")]
        if container == GUID_ContainerFormatWmp {
            let image = decode_jxr(b).map_err(wicerr)?;
            let format = guid_from_str(&image.guid());
            return Ok(one(Source { format, width: image.width, height: image.height, stride: image.stride, pixels: image.pixels, palette: Vec::new(), dpi: (0.0, 0.0) }, None));
        }
        Err(e(WINCODEC_ERR_COMPONENTNOTFOUND))
    }
}

#[implement(IWICBitmapDecoder)]
pub(crate) struct DecoderObj {
    frames: RefCell<Vec<FrameState>>,
    attempted: Cell<bool>,
    pub(crate) container: Cell<GUID>,
    /// The raw container bytes, shared with frames for the metadata query reader.
    pub(crate) raw: RefCell<Rc<Vec<u8>>>,
    /// The writable stream the decoder was initialised from (file/handle decodes have none): the
    /// fast metadata encoder's in-place `Commit` writes the edited container back into it.
    pub(crate) stream: RefCell<Option<IStream>>,
}

impl DecoderObj {
    /// An uninitialized decoder (the `CoCreateInstance(CLSID_WICPngDecoder)` path); `Initialize`
    /// reads its stream.
    pub(crate) fn new_empty() -> Self { Self::for_container(GUID_ContainerFormatPng) }
    pub(crate) fn for_container(container: GUID) -> Self {
        DecoderObj { frames: RefCell::new(Vec::new()), attempted: Cell::new(false), container: Cell::new(container), raw: RefCell::new(Rc::new(Vec::new())), stream: RefCell::new(None) }
    }
    fn initialize_stream(&self, stream: &IStream) -> WResult<()> {
        if self.attempted.get() { return Err(e(WINCODEC_ERR_WRONGSTATE)); }
        let container = self.container.get();
        // WIC binds its stream before header validation; a failed Initialize still binds it.
        if container != GUID_ContainerFormatWmp { self.attempted.set(true); }
        let mut pos = 0;
        unsafe { stream.Seek(0, STREAM_SEEK_CUR, Some(&mut pos))?; }
        let mut bytes = read_stream(stream)?;
        // The JPEG codestream starts at the current stream position; other built-in decoders seek.
        if container == GUID_ContainerFormatJpeg && pos != 0 {
            bytes = bytes.get(pos as usize..).unwrap_or(&[]).to_vec();
        }
        *self.stream.borrow_mut() = Some(stream.clone());
        let frames = decode_selected(container, &bytes)?;
        self.attempted.set(true);
        *self.frames.borrow_mut() = frames;
        *self.raw.borrow_mut() = Rc::new(bytes);
        Ok(())
    }
    pub(crate) fn from_bytes(bytes: Vec<u8>) -> WResult<IWICBitmapDecoder> {
        let frames = decode_all(&bytes)?;
        Ok(DecoderObj { attempted: Cell::new(true), container: Cell::new(container_guid(&bytes)), frames: RefCell::new(frames), raw: RefCell::new(Rc::new(bytes)), stream: RefCell::new(None) }.into())
    }
    /// Decode a stream, retaining it (writable) for in-place metadata serialization.
    pub(crate) fn from_stream(stream: &IStream) -> WResult<IWICBitmapDecoder> {
        let bytes = read_stream(stream)?;
        let frames = decode_all(&bytes)?;
        Ok(DecoderObj { attempted: Cell::new(true), container: Cell::new(container_guid(&bytes)), frames: RefCell::new(frames), raw: RefCell::new(Rc::new(bytes)), stream: RefCell::new(Some(stream.clone())) }.into())
    }
}

impl IWICBitmapDecoder_Impl for DecoderObj_Impl {
    fn QueryCapability(&self, stream: Ref<IStream>) -> WResult<u32> {
        let stream = stream.as_ref().ok_or(e(E_INVALIDARG))?;
        self.initialize_stream(stream)?;
        // WIC's QueryCapability: combine header flags
        // and metadata availability across frames after binding and checking the stream.
        let container = self.container.get();
        let flags = if container == GUID_ContainerFormatBmp { 2 }
            else if container == GUID_ContainerFormatGif || container == GUID_ContainerFormatTiff { 14 }
            else if container == GUID_ContainerFormatDds { 4 }
            else { 10 };
        Ok(flags)
    }
    fn Initialize(&self, stream: Ref<IStream>, _opts: WICDecodeOptions) -> WResult<()> {
        self.initialize_stream(stream.as_ref().ok_or(e(E_INVALIDARG))?)
    }
    fn GetContainerFormat(&self) -> WResult<GUID> {
        Ok(self.container.get())
    }
    fn GetDecoderInfo(&self) -> WResult<IWICBitmapDecoderInfo> {
        super::components::decoder_info(&self.container.get())
    }
    fn CopyPalette(&self, _p: Ref<IWICPalette>) -> WResult<()> {
        Err(e(WINCODEC_ERR_PALETTEUNAVAILABLE))
    }
    fn GetMetadataQueryReader(&self) -> WResult<IWICMetadataQueryReader> {
        super::meta::decoder_query_reader(self.container.get(), self.raw.borrow().clone())
    }
    fn GetPreview(&self) -> WResult<IWICBitmapSource> {
        // WIC's GetPreview always rejects (no built-in codec supplies a preview).
        Err(e(WINCODEC_ERR_UNSUPPORTEDOPERATION))
    }
    fn GetColorContexts(&self, _c: u32, _p: *mut Option<IWICColorContext>, actual: *mut u32) -> WResult<()> {
        unsafe { *actual = 0 };
        Ok(())
    }
    fn GetThumbnail(&self) -> WResult<IWICBitmapSource> {
        Err(e(WINCODEC_ERR_CODECNOTHUMBNAIL))
    }
    fn GetFrameCount(&self) -> WResult<u32> {
        let container = self.container.get();
        if !self.attempted.get() && (container == GUID_ContainerFormatDds || container == GUID_ContainerFormatWmp) {
            return Err(e(WINCODEC_ERR_WRONGSTATE));
        }
        if !self.attempted.get() && container == GUID_ContainerFormatGif { return Err(e(WINCODEC_ERR_FRAMEMISSING)); }
        if matches!(container, GUID_ContainerFormatBmp | GUID_ContainerFormatJpeg | GUID_ContainerFormatPng) { return Ok(1); }
        Ok(self.frames.borrow().len() as u32)
    }
    fn GetFrame(&self, index: u32) -> WResult<IWICBitmapFrameDecode> {
        let container = self.container.get();
        if !self.attempted.get() && (container == GUID_ContainerFormatDds || container == GUID_ContainerFormatWmp) { return Err(e(WINCODEC_ERR_WRONGSTATE)); }
        if index == 0 && self.frames.borrow().is_empty() && matches!(container, GUID_ContainerFormatBmp | GUID_ContainerFormatJpeg | GUID_ContainerFormatPng) {
            return Ok(make_frame(FrameObj { src: RefCell::new(None), jpeg: None, full: (0, 0), container, raw: self.raw.borrow().clone(), index, stream: self.stream.borrow().clone() }));
        }
        let fs = self.frames.borrow().get(index as usize).cloned().ok_or(e(WINCODEC_ERR_FRAMEMISSING))?;
        Ok(make_frame(FrameObj::new(fs, container, self.raw.borrow().clone(), index, self.stream.borrow().clone())))
    }

}

// --- frame --------------------------------------------------------------------------------------

#[implement(IWICBitmapFrameDecode)]
pub(crate) struct FrameObj {
    src: RefCell<Option<Source>>,
    jpeg: Option<Vec<u8>>,
    full: (u32, u32),
    pub(crate) container: GUID,
    pub(crate) raw: Rc<Vec<u8>>,
    pub(crate) index: u32,
    /// The decoder's writable stream (see [`DecoderObj::stream`]); the frame fast encoder's
    /// `Commit` writes the edited container back into it.
    pub(crate) stream: Option<IStream>,
}

impl FrameObj {
    fn new(fs: FrameState, container: GUID, raw: Rc<Vec<u8>>, index: u32, stream: Option<IStream>) -> Self {
        FrameObj { src: RefCell::new(Some(fs.src)), jpeg: fs.jpeg, full: fs.full, container, raw, index, stream }
    }
}

#[implement(IWICBitmapFrameDecode, IWICBitmapSourceTransform)]
struct TransformFrameObj { inner: FrameObj }
impl core::ops::Deref for TransformFrameObj {
    type Target = FrameObj;
    fn deref(&self) -> &FrameObj { &self.inner }
}
fn make_frame(frame: FrameObj) -> IWICBitmapFrameDecode {
    if matches!(frame.container, GUID_ContainerFormatPng | GUID_ContainerFormatJpeg | GUID_ContainerFormatWmp) {
        TransformFrameObj { inner: frame }.into()
    } else { frame.into() }
}
pub(crate) fn frame_metadata(frame: &IWICBitmapFrameDecode) -> WResult<(GUID, Rc<Vec<u8>>, u32, Option<IStream>)> {
    if let Ok(obj) = frame.cast_object_ref::<FrameObj>() { return Ok((obj.container, obj.raw.clone(), obj.index, obj.stream.clone())); }
    if let Ok(obj) = frame.cast_object_ref::<TransformFrameObj>() { return Ok((obj.container, obj.raw.clone(), obj.index, obj.stream.clone())); }
    Err(e(WINCODEC_ERR_UNSUPPORTEDOPERATION))
}

impl_source!(FrameObj_Impl);
impl_source!(TransformFrameObj_Impl);

macro_rules! frame_decode_impl { ($ty:ty) => {

impl IWICBitmapFrameDecode_Impl for $ty {
    fn GetMetadataQueryReader(&self) -> WResult<IWICMetadataQueryReader> {
        super::meta::frame_query_reader(self.container, self.raw.clone(), self.index)
    }
    fn GetColorContexts(&self, ccount: u32, pp: *mut Option<IWICColorContext>, actual: *mut u32) -> WResult<()> {
        // Native initialises the caller-supplied contexts from the frame's embedded ICC profile /
        // EXIF colour space; the count is reported regardless of whether buffers were passed.
        let contexts = super::color::frame_color_contexts(self.container.to_u128(), &self.raw, self.index);
        if !actual.is_null() {
            unsafe { *actual = contexts.len() as u32 };
        }
        if pp.is_null() || ccount == 0 {
            return Ok(());
        }
        let slots = unsafe { core::slice::from_raw_parts(pp, ccount as usize) };
        for (i, cd) in contexts.iter().enumerate().take(ccount as usize) {
            let ctx = slots[i].as_ref().ok_or(e(E_INVALIDARG))?;
            match cd {
                super::color::ColorData::Profile(b) => unsafe { ctx.InitializeFromMemory(b) }?,
                super::color::ColorData::Exif(v) => unsafe { ctx.InitializeFromExifColorSpace(*v) }?,
            }
        }
        Ok(())
    }
    fn GetThumbnail(&self) -> WResult<IWICBitmapSource> {
        Err(e(WINCODEC_ERR_CODECNOTHUMBNAIL))
    }
}

}; }
frame_decode_impl!(FrameObj_Impl);
frame_decode_impl!(TransformFrameObj_Impl);

impl IWICBitmapSourceTransform_Impl for TransformFrameObj_Impl {
    fn CopyPixels(&self, _prc: *const WICRect, w: u32, h: u32, _fmt: *const GUID, _t: WICBitmapTransformOptions, stride: u32, size: u32, buf: *mut u8) -> WResult<()> {
        let jpeg = self.jpeg.as_ref().ok_or(e(WINCODEC_ERR_UNSUPPORTEDOPERATION))?;
        let (denom, _) = jpeg_closest_scale(self.full.0, self.full.1, w, h);
        let img: Source = decode_jpeg_scaled(jpeg, denom as usize).map_err(ecode)?.into();
        let slice = unsafe { core::slice::from_raw_parts_mut(buf, size as usize) };
        img.copy_pixels(core::ptr::null(), stride, slice)
    }
    fn GetClosestSize(&self, pw: *mut u32, ph: *mut u32) -> WResult<()> {
        let (_, (ow, oh)) = jpeg_closest_scale(self.full.0, self.full.1, unsafe { *pw }, unsafe { *ph });
        unsafe {
            *pw = ow;
            *ph = oh;
        }
        Ok(())
    }
    fn GetClosestPixelFormat(&self, _fmt: *mut GUID) -> WResult<()> {
        Ok(())
    }
    fn DoesSupportTransform(&self, t: WICBitmapTransformOptions) -> WResult<windows_core::BOOL> {
        Ok((t == WICBitmapTransformRotate0).into())
    }
}
