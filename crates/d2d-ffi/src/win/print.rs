//! `ID2D1PrintControl` — a print job that serialises each page's command list into an XPS package
//! written to the caller's `IPrintDocumentPackageTarget`.
//!
//! Direct2D validates the target offers an MSXPS or
//! OpenXPS package type, obtains the `IXpsOMObjectFactory`/`IXpsOMPackageWriter` from the target, and
//! drives the XPS Object Model — emitting **vector** XPS (Path/Glyphs). We reproduce the same package
//! structure using the same OS XPS OM obtained from the target: each page's command list is replayed
//! through a `StreamSink` (`VectorSink`) that serialises its primitives into a root `Canvas` of XPS
//! `Path`s, byte-for-byte identical to native's package (verified vs native, cov-12-print).
//!
//! ponytail: solid fills (fill_rect/fill_polys) only — glyph runs → XPS `Glyphs` + a font part, and
//! gradient/image brushes, are follow-ups when a scene exercises them.

use super::image::CommandListObj;
use std::cell::RefCell;
use windows::Win32::Graphics::Direct2D::Common::D2D_SIZE_F;
use windows::Win32::Graphics::Direct2D::*;
use windows::Win32::Graphics::Imaging::{IWICImagingFactory, GUID_ContainerFormatPng, GUID_WICPixelFormat32bppBGRA, GUID_WICPixelFormat32bppPBGRA, WICBitmapEncoderNoCache};
use windows::Win32::Storage::Xps::Printing::{IPrintDocumentPackageTarget, ID_DOCUMENTPACKAGETARGET_MSXPS, ID_DOCUMENTPACKAGETARGET_OPENXPS};
use windows::Win32::Foundation::HGLOBAL;
use windows::Win32::Storage::Xps::*;
use windows::Win32::System::Com::StructuredStorage::CreateStreamOnHGlobal;
use windows::Win32::System::Com::{CoCreateGuid, CoTaskMemFree, STREAM_SEEK_SET};
use windows_core::{implement, Interface, Ref, Result as WResult, BOOL, PCWSTR};

const E_INVALIDARG: windows_core::HRESULT = windows_core::HRESULT(0x8007_0057u32 as i32);
const D2DERR_PRINT_FORMAT_NOT_SUPPORTED: windows_core::HRESULT = windows_core::HRESULT(0x8899_002Cu32 as i32);
const D2DERR_PRINT_JOB_CLOSED: windows_core::HRESULT = windows_core::HRESULT(0x8899_002Bu32 as i32);

/// Open XPS writer state; dropped to `None` by `Close`, so a later `AddPage`/`Close` is rejected.
struct PrintInner {
    factory: IXpsOMObjectFactory,
    writer: IXpsOMPackageWriter,
    page_no: u32,
    /// `D2D1_PRINT_CONTROL_PROPERTIES.fontSubset`: NONE embeds the full font (what we do, byte-exact
    /// vs native); DEFAULT/EACHPAGE subset it (pending a font subsetter — see add_glyphs).
    font_subset: D2D1_PRINT_FONT_SUBSET_MODE,
    /// The caller's WIC factory, used to encode a bitmap-brush's texels to the PNG image part (native
    /// routes image serialization through WIC too, so a system WIC gives a matching PNG).
    wic: Option<IWICImagingFactory>,
}

#[implement(ID2D1PrintControl)]
pub(crate) struct PrintControlObj {
    inner: RefCell<Option<PrintInner>>,
}

/// `ID2D1Device::CreatePrintControl`: validate the target's package type and open an XPS package
/// writer on it (the WIC factory and print-control properties are not needed — pages are encoded as
/// PNG by the core and the default font-subset behaviour is XPS-OM's).
pub(crate) fn create_print_control(wic: Ref<IWICImagingFactory>, target: Ref<IPrintDocumentPackageTarget>, props: *const D2D1_PRINT_CONTROL_PROPERTIES) -> WResult<ID2D1PrintControl> {
    let wic = wic.as_ref().cloned();
    // fontSubset steers how glyph-run fonts embed (DEFAULT when no properties were passed).
    let font_subset = if props.is_null() { D2D1_PRINT_FONT_SUBSET_MODE_DEFAULT } else { unsafe { (*props).fontSubset } };
    let target = target.as_ref().ok_or(windows_core::Error::from(E_INVALIDARG))?;
    // The target must offer an MSXPS or OpenXPS package type, else D2DERR_PRINT_FORMAT_NOT_SUPPORTED.
    let (mut count, mut types) = (0u32, core::ptr::null_mut());
    unsafe { target.GetPackageTargetTypes(&mut count, &mut types)? };
    let kind = unsafe { core::slice::from_raw_parts(types, count as usize) }
        .iter()
        .find(|g| **g == ID_DOCUMENTPACKAGETARGET_OPENXPS || **g == ID_DOCUMENTPACKAGETARGET_MSXPS)
        .copied();
    unsafe { CoTaskMemFree(Some(types.cast())) };
    let kind = kind.ok_or(windows_core::Error::from(D2DERR_PRINT_FORMAT_NOT_SUPPORTED))?;

    let xps: IXpsDocumentPackageTarget = unsafe { target.GetPackageTarget(&kind)? };
    let factory = unsafe { xps.GetXpsOMFactory()? };
    let fdseq = unsafe { factory.CreatePartUri(PCWSTR(wide("/FixedDocumentSequence.fdseq").as_ptr()))? };
    // Native requests a discard-control part (progressive-download hint); passing it adds the
    // DiscardControl.xml part + its relationship + content-type, matching native's package.
    let discard = unsafe { factory.CreatePartUri(PCWSTR(wide("/DiscardControl.xml").as_ptr()))? };
    let writer = unsafe { xps.GetXpsOMPackageWriter(&fdseq, &discard)? };
    // A page must live inside a document: start the fixed document before any AddPage (else
    // XPS_E_MISSING_DOCUMENT). Native's print control does this internally.
    let fdoc = unsafe { factory.CreatePartUri(PCWSTR(wide("/Documents/1/FixedDocument.fdoc").as_ptr()))? };
    unsafe {
        writer.StartNewDocument(
            &fdoc,
            None::<&IXpsOMPrintTicketResource>,
            None::<&IXpsOMDocumentStructureResource>,
            None::<&IXpsOMSignatureBlockResourceCollection>,
            None::<&IXpsOMPartUriCollection>,
        )?
    };
    Ok(PrintControlObj { inner: RefCell::new(Some(PrintInner { factory, writer, page_no: 0, font_subset, wic })) }.into())
}

impl ID2D1PrintControl_Impl for PrintControlObj_Impl {
    fn AddPage(&self, commandlist: Ref<ID2D1CommandList>, pagesize: &D2D_SIZE_F, _ticket: Ref<windows::Win32::System::Com::IStream>, tag1: *mut u64, tag2: *mut u64) -> WResult<()> {
        if !tag1.is_null() {
            unsafe { *tag1 = 0 };
        }
        if !tag2.is_null() {
            unsafe { *tag2 = 0 };
        }
        let mut guard = self.inner.borrow_mut();
        let inner = guard.as_mut().ok_or(windows_core::Error::from(D2DERR_PRINT_JOB_CLOSED))?;
        let list = commandlist
            .as_ref()
            .and_then(|c| c.cast_object_ref::<CommandListObj>().ok())
            .ok_or(windows_core::Error::from(E_INVALIDARG))?;
        inner.page_no += 1;
        // Serialise the page as native does: a ROOT Canvas of vector primitives (Path/Glyphs) built
        // through the same OS XPS OM, so the package parts match. The command list replays its recorded
        // device-independent primitives through the vector sink.
        unsafe { add_vector_page(inner, &list.list, *pagesize) }
    }

    fn Close(&self) -> WResult<()> {
        let inner = self.inner.borrow_mut().take().ok_or(windows_core::Error::from(D2DERR_PRINT_JOB_CLOSED))?;
        unsafe { inner.writer.Close() }
    }
}

/// Build a `.fpage` as a ROOT `Canvas` of vector primitives (the command list's recorded fills,
/// resolved to device space by `CommandList::stream`) via the OS XPS OM, and hand it to the writer —
/// native's default vector serialisation (not the raster fallback). Matched part-by-part vs native.
unsafe fn add_vector_page(inner: &PrintInner, list: &d2d::CommandList, size: D2D_SIZE_F) -> WResult<()> {
    let n = inner.page_no;
    let f = &inner.factory;
    unsafe {
        let dims = XPS_SIZE { width: size.width, height: size.height };
        let page_uri = f.CreatePartUri(PCWSTR(wide(&format!("/Documents/1/Pages/{n}.fpage")).as_ptr()))?;
        let page = f.CreatePage(&dims, windows_core::w!("und"), &page_uri)?;
        let canvas = f.CreateCanvas()?;
        // Native names the root canvas via the accessibility short description (→ AutomationProperties.Name="ROOT").
        canvas.SetAccessibilityShortDescription(windows_core::w!("ROOT"))?;
        let mut sink = VectorSink { factory: f, visuals: canvas.GetVisuals()?, stack: Vec::new(), font_subset: inner.font_subset, wic: &inner.wic, err: None };
        list.stream(&mut sink);
        if let Some(e) = sink.err {
            return Err(e);
        }
        page.GetVisuals()?.Append(&canvas)?;
        inner.writer.AddPage(
            &page,
            &dims,
            None::<&IXpsOMPartUriCollection>,
            None::<&IXpsOMStoryFragmentsResource>,
            None::<&IXpsOMPrintTicketResource>,
            None::<&IXpsOMImageResource>,
        )
    }
}

/// A `StreamSink` that turns each recorded primitive into an XPS-OM `Path` on the ROOT canvas. A solid
/// fill → a closed `Path` with the device-space figures and an sRGB solid brush. ponytail: fills only
/// so far (glyph runs / images are follow-ups); the first error is latched and surfaced by the caller.
struct VectorSink<'a> {
    factory: &'a IXpsOMObjectFactory,
    visuals: IXpsOMVisualCollection,
    /// Parent visual collections suspended by `push_layer` (a layer nests a child `Canvas`).
    stack: Vec<IXpsOMVisualCollection>,
    /// The print control's font-subset mode, steering how a glyph run's font embeds.
    font_subset: D2D1_PRINT_FONT_SUBSET_MODE,
    /// The caller's WIC factory, for encoding a bitmap-brush's texels to the PNG image part.
    wic: &'a Option<IWICImagingFactory>,
    err: Option<windows_core::Error>,
}

impl VectorSink<'_> {
    /// An XPS geometry over `figures` (device-space polylines), each figure stroked-by-default so it
    /// serialises to the abbreviated `Data="M…L…"` form native uses. `fill` closes and fills each
    /// figure (`…Z`, for a fill); a stroke passes `false` for an open, unfilled path.
    fn build_geom(&self, figures: &[Vec<[f32; 2]>], fill: bool) -> WResult<IXpsOMGeometry> {
        unsafe {
            let geom = self.factory.CreateGeometry()?;
            let figs = geom.GetFigures()?;
            for poly in figures {
                if poly.is_empty() {
                    continue;
                }
                let figure = self.factory.CreateGeometryFigure(&XPS_POINT { x: poly[0][0], y: poly[0][1] })?;
                let segtypes = vec![XPS_SEGMENT_TYPE_LINE; poly.len() - 1];
                let segdata: Vec<f32> = poly[1..].iter().flat_map(|p| [p[0], p[1]]).collect();
                let segstrokes = vec![BOOL(1); poly.len() - 1];
                figure.SetSegments((poly.len() - 1) as u32, segdata.len() as u32, segtypes.as_ptr(), segdata.as_ptr(), segstrokes.as_ptr())?;
                figure.SetIsClosed(fill)?;
                // Leave IsFilled at its default (true) for a stroke so the figure keeps the abbreviated
                // `Data="M…L…"` form — an explicit SetIsFilled(false) forces the expanded PathGeometry.
                if fill {
                    figure.SetIsFilled(true)?;
                }
                figs.Append(&figure)?;
            }
            Ok(geom)
        }
    }
    /// Append a `Path` over `geom` filled by `brush`, with the explicit identity RenderTransform native
    /// emits on every primitive.
    fn add_path(&mut self, geom: &IXpsOMGeometry, brush: &IXpsOMBrush) -> WResult<()> {
        unsafe {
            let path = self.factory.CreatePath()?;
            path.SetGeometryLocal(geom)?;
            path.SetFillBrushLocal(brush)?;
            let xform = self.factory.CreateMatrixTransform(&XPS_MATRIX { m11: 1.0, m12: 0.0, m21: 0.0, m22: 1.0, m31: 0.0, m32: 0.0 })?;
            path.SetTransformLocal(&xform)?;
            self.visuals.Append(&path)?;
        }
        Ok(())
    }
    /// Append a closed `Path` filling `figures` (device-space polylines) with `brush`.
    fn add_fill(&mut self, figures: &[Vec<[f32; 2]>], brush: &d2d::resource::SolidColorBrush) -> WResult<()> {
        unsafe {
            let geom = self.build_geom(figures, true)?;
            let fill = self.factory.CreateSolidColorBrush(&xps_srgb(brush.color()), None)?;
            // Native keeps the brush opacity SEPARATE from the colour (a `<SolidColorBrush Opacity=…>`
            // element) rather than baking it into alpha; setting it only when ≠1 keeps the opaque
            // fill serialising to the inline `Fill="#rrggbb"` attribute.
            if brush.opacity() != 1.0 {
                fill.SetOpacity(brush.opacity())?;
            }
            self.add_path(&geom, &fill.cast()?)
        }
    }
    /// Append a `Path` over `figures` filled by a gradient brush reconstructed from `def` (its own
    /// coordinates mapped into device space by `m`). Non-gradient defs are skipped (follow-up).
    fn add_gradient(&mut self, figures: &[Vec<[f32; 2]>], def: &d2d::brush::BrushDef, m: [f32; 6]) -> WResult<()> {
        use d2d::brush::BrushDef;
        unsafe {
            // XPS GradientStops: one per (offset, straight sRGB colour).
            let mk_stops = |stops: &[(f32, [f32; 4])]| -> WResult<Vec<IXpsOMGradientStop>> {
                stops.iter().map(|(off, c)| self.factory.CreateGradientStop(&xps_srgb(*c), None, *off)).collect()
            };
            let brush: IXpsOMBrush = match def {
                BrushDef::Linear { stops, start, end, .. } => {
                    let s = mk_stops(stops)?;
                    if s.len() < 2 {
                        return Ok(());
                    }
                    let b = self.factory.CreateLinearGradientBrush(&s[0], &s[1], &ap_m(m, *start), &ap_m(m, *end))?;
                    let coll = b.GetGradientStops()?;
                    for st in &s[2..] {
                        coll.Append(st)?;
                    }
                    b.cast()?
                }
                BrushDef::Radial { stops, center, radii, offset, .. } => {
                    let s = mk_stops(stops)?;
                    if s.len() < 2 {
                        return Ok(());
                    }
                    let origin = [center[0] + offset[0], center[1] + offset[1]];
                    // Device-scale the radii (identity/dpi-1 at 96dpi); off-axis radii skew is a follow-up.
                    let (sx, sy) = ((m[0] * m[0] + m[1] * m[1]).sqrt(), (m[2] * m[2] + m[3] * m[3]).sqrt());
                    let radii_sz = XPS_SIZE { width: radii[0] * sx, height: radii[1] * sy };
                    let b = self.factory.CreateRadialGradientBrush(&s[0], &s[1], &ap_m(m, *center), &ap_m(m, origin), &radii_sz)?;
                    let coll = b.GetGradientStops()?;
                    for st in &s[2..] {
                        coll.Append(st)?;
                    }
                    b.cast()?
                }
                // WRAP/MIRROR bitmap brushes tile directly to an XPS `<ImageBrush>` with a TileMode; any
                // CLAMP axis (CLAMP/CLAMP or mixed CLAMP+WRAP/MIRROR) builds native's nested clamp-emulation.
                BrushDef::Bitmap { texels, size, extend, xform, .. } => {
                    if extend[0] == 0 || extend[1] == 0 {
                        self.clamp_bitmap_brush(texels, *size, *extend, figures_bbox(figures))?
                    } else {
                        self.bitmap_image_brush(texels, *size, *extend, m, *xform)?
                    }
                }
            };
            let geom = self.build_geom(figures, true)?;
            self.add_path(&geom, &brush)
        }
    }
    /// WIC-encode a bitmap brush's premultiplied-BGRA `texels` to a PNG image part (as native's
    /// ImageConvertor does) and return the XPS image resource. The part lives at the package root
    /// `/Resources/Images/{GUID}.png` under a fresh random GUID, and the frame is tagged 96 DPI so the
    /// encoder writes the `pHYs` chunk (3779 ppm) native emits.
    fn encode_png_image_resource(&self, texels: &[u8], size: [u32; 2]) -> WResult<IXpsOMImageResource> {
        unsafe {
            let wic = self.wic.as_ref().ok_or(windows_core::Error::from(E_INVALIDARG))?;
            let [w, h] = size;
            let bmp = wic.CreateBitmapFromMemory(w, h, &GUID_WICPixelFormat32bppPBGRA, w * 4, texels)?;
            let istream = CreateStreamOnHGlobal(HGLOBAL(core::ptr::null_mut()), true)?;
            let encoder = wic.CreateEncoder(&GUID_ContainerFormatPng, core::ptr::null())?;
            encoder.Initialize(&istream, WICBitmapEncoderNoCache)?;
            let (mut frame, mut props) = (None, None);
            encoder.CreateNewFrame(&mut frame, &mut props)?;
            let frame = frame.ok_or(windows_core::Error::from(E_INVALIDARG))?;
            frame.Initialize(props.as_ref())?;
            frame.SetSize(w, h)?;
            frame.SetResolution(96.0, 96.0)?;
            frame.SetPixelFormat(&mut GUID_WICPixelFormat32bppBGRA.clone())?;
            frame.WriteSource(&bmp, core::ptr::null())?;
            frame.Commit()?;
            encoder.Commit()?;
            istream.Seek(0, STREAM_SEEK_SET, None)?;
            let guid = CoCreateGuid()?;
            let part = format!("/Resources/Images/{}.png", fmt_guid(&guid));
            let uri = self.factory.CreatePartUri(PCWSTR(wide(&part).as_ptr()))?;
            Ok(self.factory.CreateImageResource(&istream, XPS_IMAGE_TYPE_PNG, &uri)?)
        }
    }
    /// A tiling XPS `ImageBrush` from a bitmap brush's texels: Viewbox=Viewport=(0,0,w,h) with a
    /// TileMode from the extend modes (WRAP→Tile, MIRROR→flip) — native maps the pixel box, not the fill.
    fn bitmap_image_brush(&self, texels: &[u8], size: [u32; 2], extend: [u32; 2], _m: [f32; 6], _xform: [f32; 6]) -> WResult<IXpsOMBrush> {
        unsafe {
            let [w, h] = size;
            let imgres = self.encode_png_image_resource(texels, size)?;
            let vb = XPS_RECT { x: 0.0, y: 0.0, width: w as f32, height: h as f32 };
            let brush = self.factory.CreateImageBrush(&imgres, &vb, &vb)?;
            // MIRROR (D2D1_EXTEND_MODE 2) flips per axis; WRAP (1) tiles. CLAMP routes elsewhere.
            let tile = match (extend[0] == 2, extend[1] == 2) {
                (false, false) => XPS_TILE_MODE_TILE,
                (true, false) => XPS_TILE_MODE_FLIPX,
                (false, true) => XPS_TILE_MODE_FLIPY,
                (true, true) => XPS_TILE_MODE_FLIPXY,
            };
            brush.SetTileMode(tile)?;
            Ok(brush.cast()?)
        }
    }
    /// A closed, filled XPS rectangle geometry in the expanded `PathGeometry` form native's
    /// `CreateXpsRectangleGeometry` emits: StartPoint=(x,y), CCW corners, `IsStroked="false"`.
    fn rect_geom(&self, x: f32, y: f32, w: f32, h: f32) -> WResult<IXpsOMGeometry> {
        unsafe {
            let geom = self.factory.CreateGeometry()?;
            let figs = geom.GetFigures()?;
            let figure = self.factory.CreateGeometryFigure(&XPS_POINT { x, y })?;
            let segtypes = [XPS_SEGMENT_TYPE_LINE; 3];
            let segdata = [x, y + h, x + w, y + h, x + w, y];
            let segstrokes = [BOOL(0); 3];
            figure.SetSegments(3, 6, segtypes.as_ptr(), segdata.as_ptr(), segstrokes.as_ptr())?;
            figure.SetIsClosed(true)?;
            figs.Append(&figure)?;
            Ok(geom)
        }
    }
    /// A CLAMP-extend bitmap brush, serialised the way native's `dxc::EmulateClampExtendMode` does:
    /// an outer `VisualBrush` (Viewbox=Viewport=fill bounds) over a `Canvas` whose resource dictionary
    /// holds a `MainVisual` path (the image at its pixel box), tiled over a 3×3 grid of clamp cells.
    /// `bbox` is the fill bounds [L,T,R,B] in brush space (identity brush transform — the common case).
    fn clamp_bitmap_brush(&self, texels: &[u8], size: [u32; 2], extend: [u32; 2], bbox: [f32; 4]) -> WResult<IXpsOMBrush> {
        unsafe {
            let (wf, hf) = (size[0] as f32, size[1] as f32);
            let [l, t, r, b] = bbox;
            let imgres = self.encode_png_image_resource(texels, size)?;
            // Outer VisualBrush: maps the fill bounds onto themselves (identity), carrying an identity
            // brush transform. TileMode NONE is the default and stays unserialised.
            let fbox = XPS_RECT { x: l, y: t, width: r - l, height: b - t };
            let outer = self.factory.CreateVisualBrush(&fbox, &fbox)?;
            let id = self.factory.CreateMatrixTransform(&XPS_MATRIX { m11: 1.0, m12: 0.0, m21: 0.0, m22: 1.0, m31: 0.0, m32: 0.0 })?;
            outer.SetTransformLocal(&id)?;
            // The MainVisual resource: a path over the pixel box filled by the untiled image. The OM call
            // order follows native's EmulateClampExtendMode exactly — in particular the dictionary is
            // attached to the canvas BEFORE the entry is appended, and each cell brush's visual lookup is
            // set only AFTER the cell path is in the canvas's visual tree, so "MainVisual" resolves.
            let canvas = self.factory.CreateCanvas()?;
            let dict = self.factory.CreateDictionary()?;
            let main_path = self.factory.CreatePath()?;
            let main_geom = self.rect_geom(0.0, 0.0, wf, hf)?;
            main_path.SetGeometryLocal(&main_geom)?;
            let img_vb = XPS_RECT { x: 0.0, y: 0.0, width: wf, height: hf };
            let img_brush = self.factory.CreateImageBrush(&imgres, &img_vb, &img_vb)?;
            img_brush.SetTileMode(XPS_TILE_MODE_NONE)?;
            main_path.SetFillBrushLocal(&img_brush)?;
            canvas.SetDictionaryLocal(&dict)?;
            dict.Append(windows_core::w!("MainVisual"), &main_path)?;
            let visuals = canvas.GetVisuals()?;
            // The 3×3 clamp cells: each cell paints a sub-window of MainVisual, tiled, over its clipped
            // fill-space rect. Cells narrowed to the 0.12 overlap sliver (or clipped away) are dropped.
            let tile = match (extend[0], extend[1]) {
                (0, 2) => XPS_TILE_MODE_FLIPY,
                (2, 0) => XPS_TILE_MODE_FLIPX,
                _ => XPS_TILE_MODE_TILE,
            };
            for (rect, view) in clamp_cells(wf, hf, [l, t, r, b], extend) {
                let [cx, cy, cw, ch] = rect;
                if 0.12 >= cw || 0.12 >= ch {
                    continue;
                }
                let cell_path = self.factory.CreatePath()?;
                let cell_geom = self.rect_geom(cx, cy, cw, ch)?;
                cell_path.SetGeometryLocal(&cell_geom)?;
                let [vx, vy, vw, vh] = view;
                let vrect = XPS_RECT { x: vx, y: vy, width: vw, height: vh };
                let vbrush = self.factory.CreateVisualBrush(&vrect, &vrect)?;
                vbrush.SetTileMode(tile)?;
                cell_path.SetFillBrushLocal(&vbrush)?;
                visuals.Append(&cell_path)?;
                vbrush.SetVisualLookup(windows_core::w!("MainVisual"))?;
            }
            // Outer brush last: TileMode NONE, the canvas as its visual, then the brush transform.
            outer.SetTileMode(XPS_TILE_MODE_NONE)?;
            outer.SetVisualLocal(&canvas)?;
            Ok(outer.cast()?)
        }
    }
    /// Open a layer: a child `Canvas` named LAYER with the layer opacity and a rectangular clip
    /// (the content bounds). Subsequent fills append into it until `pop_layer`.
    fn add_layer(&mut self, bounds: [f32; 4], opacity: f32) -> WResult<()> {
        unsafe {
            let canvas = self.factory.CreateCanvas()?;
            canvas.SetAccessibilityShortDescription(windows_core::w!("LAYER"))?;
            if opacity != 1.0 {
                canvas.SetOpacity(opacity)?;
            }
            // Clip = the content-bounds rectangle. Native lists the closing point explicitly
            // (M L,T L R,T R,B L,B L,T Z), so the figure walks all four corners back to the start.
            let [l, t, r, b] = bounds;
            let geom = self.factory.CreateGeometry()?;
            let figure = self.factory.CreateGeometryFigure(&XPS_POINT { x: l, y: t })?;
            let segtypes = [XPS_SEGMENT_TYPE_LINE; 4];
            let segdata = [r, t, r, b, l, b, l, t];
            let segstrokes = [BOOL(1); 4];
            figure.SetSegments(4, 8, segtypes.as_ptr(), segdata.as_ptr(), segstrokes.as_ptr())?;
            figure.SetIsClosed(true)?;
            figure.SetIsFilled(true)?;
            geom.GetFigures()?.Append(&figure)?;
            canvas.SetClipGeometryLocal(&geom)?;
            self.visuals.Append(&canvas)?;
            self.stack.push(self.visuals.clone());
            self.visuals = canvas.GetVisuals()?;
        }
        Ok(())
    }
    /// Append a stroked (not filled) `Path` over `figures` with `width` thickness, the solid `brush`,
    /// and `dashes` (D2D thickness-multiple pairs → XPS `StrokeDashArray`). Native keeps the original
    /// path (not the widened outline) and expresses the stroke through attributes.
    fn add_stroke(&mut self, figures: &[Vec<[f32; 2]>], width: f32, dashes: &[f32], brush: &d2d::resource::SolidColorBrush) -> WResult<()> {
        unsafe {
            let geom = self.build_geom(figures, false)?;
            let path = self.factory.CreatePath()?;
            path.SetGeometryLocal(&geom)?;
            path.SetStrokeThickness(width)?;
            let sb = self.factory.CreateSolidColorBrush(&xps_srgb(brush.color()), None)?;
            if brush.opacity() != 1.0 {
                sb.SetOpacity(brush.opacity())?;
            }
            let sbb: IXpsOMBrush = sb.cast()?;
            path.SetStrokeBrushLocal(&sbb)?;
            if !dashes.is_empty() {
                let dc = path.GetStrokeDashes()?;
                for pair in dashes.chunks(2) {
                    dc.Append(&XPS_DASH { length: pair[0], gap: *pair.get(1).unwrap_or(&0.0) })?;
                }
            }
            let xform = self.factory.CreateMatrixTransform(&XPS_MATRIX { m11: 1.0, m12: 0.0, m21: 0.0, m22: 1.0, m31: 0.0, m32: 0.0 })?;
            path.SetTransformLocal(&xform)?;
            self.visuals.Append(&path)?;
        }
        Ok(())
    }
    /// Append a `<Glyphs>` run of `font` (embedded as an obfuscated `.odttf` part under a fresh random
    /// v4 GUID, as native does) at `origin`, em `em_size`, colour `color`, with `ids`/`advance_widths`.
    fn add_glyphs(&mut self, font: &[u8], ids: &[u16], advance_widths: &[f32], em_size: f32, origin: [f32; 2], color: [f32; 4]) -> WResult<()> {
        unsafe {
            // The obfuscated font part is named by a fresh random (v4) GUID; the OM keys the
            // obfuscation off it. CoCreateGuid is native's own source, so ours is a v4 GUID too.
            let guid = CoCreateGuid()?;
            let part = format!("/Resources/Fonts/{}.odttf", fmt_guid(&guid));
            let uri = self.factory.CreatePartUri(PCWSTR(wide(&part).as_ptr()))?;
            // NONE embeds the full font (byte-exact vs native). DEFAULT/EACHPAGE subset it, which we
            // can't do yet — embed the full font meanwhile (renders identically; byte-size differs).
            // ponytail: subset here once the DWrite-equivalent subsetter is ported (owner-scheduled).
            let embed = font;
            let _ = self.font_subset; // honoured once a subsetter exists; NONE is already correct
            // Font bytes → a growable in-memory IStream for the resource to read.
            let stream = CreateStreamOnHGlobal(HGLOBAL(core::ptr::null_mut()), true)?;
            stream.Write(embed.as_ptr().cast(), embed.len() as u32, None).ok()?;
            stream.Seek(0, STREAM_SEEK_SET, None)?;
            let fontres = self.factory.CreateFontResource(&stream, XPS_FONT_EMBEDDING_OBFUSCATED, &uri, false)?;
            let glyphs = self.factory.CreateGlyphs(&fontres)?;
            glyphs.SetFontRenderingEmSize(em_size)?;
            glyphs.SetOrigin(&XPS_POINT { x: origin[0], y: origin[1] })?;
            let fill = self.factory.CreateSolidColorBrush(&xps_srgb(color), None)?;
            let fb: IXpsOMBrush = fill.cast()?;
            glyphs.SetFillBrushLocal(&fb)?;
            let indices: Vec<XPS_GLYPH_INDEX> = ids
                .iter()
                .zip(advance_widths)
                .map(|(&id, &aw)| XPS_GLYPH_INDEX { index: id as i32, advanceWidth: aw, horizontalOffset: 0.0, verticalOffset: 0.0 })
                .collect();
            let editor = glyphs.GetGlyphsEditor()?;
            editor.SetGlyphIndices(indices.len() as u32, indices.as_ptr())?;
            editor.ApplyEdits()?;
            let xform = self.factory.CreateMatrixTransform(&XPS_MATRIX { m11: 1.0, m12: 0.0, m21: 0.0, m22: 1.0, m31: 0.0, m32: 0.0 })?;
            glyphs.SetTransformLocal(&xform)?;
            self.visuals.Append(&glyphs)?;
        }
        Ok(())
    }
    fn latch(&mut self, r: WResult<()>) {
        if self.err.is_none() {
            if let Err(e) = r {
                self.err = Some(e);
            }
        }
    }
}

impl d2d::command_stream::StreamSink for VectorSink<'_> {
    fn begin(&mut self) {}
    fn end(&mut self) {}
    fn fill_rect(&mut self, rect: [f32; 4], brush: d2d::resource::SolidColorBrush, _aa: u32, _blend: u32) {
        let poly = vec![[rect[0], rect[1]], [rect[2], rect[1]], [rect[2], rect[3]], [rect[0], rect[3]]];
        let r = self.add_fill(&[poly], &brush);
        self.latch(r);
    }
    fn fill_polys(&mut self, figures: &[Vec<[f32; 2]>], _winding: bool, brush: d2d::resource::SolidColorBrush, _aa: u32, _blend: u32) {
        let r = self.add_fill(figures, &brush);
        self.latch(r);
    }
    fn fill_gradient(&mut self, figures: &[Vec<[f32; 2]>], _winding: bool, def: &d2d::brush::BrushDef, m: [f32; 6]) {
        let r = self.add_gradient(figures, def, m);
        self.latch(r);
    }
    fn stroke_geom(&mut self, orig: &[Vec<[f32; 2]>], _widened: &[Vec<[f32; 2]>], width: f32, dashes: &[f32], brush: d2d::resource::SolidColorBrush, _aa: u32, _blend: u32) {
        let r = self.add_stroke(orig, width, dashes, &brush);
        self.latch(r);
    }
    fn glyph_run(&mut self, font: &[u8], ids: &[u16], advance_widths: &[f32], em_size: f32, origin: [f32; 2], color: [f32; 4]) {
        let r = self.add_glyphs(font, ids, advance_widths, em_size, origin, color);
        self.latch(r);
    }
    fn push_layer(&mut self, bounds: [f32; 4], opacity: f32, _init: bool) {
        let r = self.add_layer(bounds, opacity);
        self.latch(r);
    }
    fn pop_layer(&mut self) {
        if let Some(parent) = self.stack.pop() {
            self.visuals = parent;
        }
    }
}

/// A straight (non-premultiplied) sRGB `XPS_COLOR` from an `[r,g,b,a]` in 0..1.
fn xps_srgb(c: [f32; 4]) -> XPS_COLOR {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    XPS_COLOR {
        colorType: XPS_COLOR_TYPE_SRGB,
        value: XPS_COLOR_0 { sRGB: XPS_COLOR_0_0 { alpha: q(c[3]), red: q(c[0]), green: q(c[1]), blue: q(c[2]) } },
    }
}

/// Map `(x,y)` through the device matrix `m` (`[a,b,c,d,e,f]`) to an `XPS_POINT`.
fn ap_m(m: [f32; 6], p: [f32; 2]) -> XPS_POINT {
    XPS_POINT { x: m[0] * p[0] + m[2] * p[1] + m[4], y: m[1] * p[0] + m[3] * p[1] + m[5] }
}

/// The axis-aligned bounds [L,T,R,B] of a set of device-space polyline figures.
fn figures_bbox(figures: &[Vec<[f32; 2]>]) -> [f32; 4] {
    let (mut l, mut t, mut r, mut b) = (f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for poly in figures {
        for p in poly {
            l = l.min(p[0]);
            t = t.min(p[1]);
            r = r.max(p[0]);
            b = b.max(p[1]);
        }
    }
    [l, t, r, b]
}

/// The 3×3 grid of (fill-space rect, pixel-space view) cells for a CLAMP bitmap brush, ported from
/// native's `dxc::BuildExtendModeCellRects` + `BuildExtendModeCellViews`. `w`×`h` is the bitmap pixel
/// box; `[l,t,r,b]` the fill bounds in brush space; `extend` the two D2D extend modes (0 CLAMP, else
/// WRAP/MIRROR — exactly one axis is non-CLAMP in the mixed case, both are CLAMP otherwise). Rects are
/// pre-clipped to the fill; the caller drops any cell narrowed to the 0.12 overlap sliver. A non-CLAMP
/// axis uses the full `(0, size)` window on every cell (its TileMode does the wrap/mirror).
fn clamp_cells(w: f32, h: f32, bounds: [f32; 4], extend: [u32; 2]) -> [([f32; 4], [f32; 4]); 9] {
    let [l, t, r, b] = bounds;
    // Pre-clip cell rects (x,y,w,h), 0.12 overlap inset. Order: C,R,BR,B,BL,L,TL,T,TR.
    let rects = [
        [0.0, 0.0, w, h],
        [w - 0.12, 0.0, r - (w - 0.12), h],
        [w - 0.12, h - 0.12, r - (w - 0.12), b - (h - 0.12)],
        [0.0, h - 0.12, w, b - (h - 0.12)],
        [l, h - 0.12, 0.12 - l, b - (h - 0.12)],
        [l, 0.0, 0.12 - l, h],
        [l, t, 0.12 - l, 0.12 - t],
        [0.0, t, w, 0.12 - t],
        [w - 0.12, t, r - (w - 0.12), 0.12 - t],
    ];
    // Per-axis view windows into MainVisual. A CLAMP axis uses the clamp sub-windows (insets 0.1/0.2/0.3);
    // a non-CLAMP axis uses (0, size) on every cell. Order matches `rects` (C,R,BR,B,BL,L,TL,T,TR).
    let xw = if extend[0] == 0 {
        [[0.0, w], [w - 0.3, 0.2], [w - 0.3, 0.2], [0.1, w - 0.1], [0.1, 0.2], [0.1, 0.2], [0.2, 0.2], [0.1, w - 0.1], [w - 0.3, 0.2]]
    } else {
        [[0.0, w]; 9]
    };
    let yh = if extend[1] == 0 {
        [[0.0, h], [0.1, h - 0.1], [h - 0.3, 0.2], [h - 0.3, 0.2], [h - 0.3, 0.2], [0.1, h - 0.1], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2]]
    } else {
        [[0.0, h]; 9]
    };
    let mut out = [([0.0; 4], [0.0; 4]); 9];
    for i in 0..9 {
        let [x, y, cw, ch] = rects[i];
        // Clip the cell rect to the fill bounds (native's final pass in BuildExtendModeCellRects).
        let nx = l.max(x);
        let ny = t.max(y);
        out[i] = ([nx, ny, r.min(x + cw) - nx, b.min(y + ch) - ny], [xw[i][0], yh[i][0], xw[i][1], yh[i][1]]);
    }
    out
}

/// A `GUID` as the uppercase hyphenated form native uses for an obfuscated-font part name
/// (`XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX`, no braces).
fn fmt_guid(g: &windows_core::GUID) -> String {
    let d = g.data4;
    format!(
        "{:08X}-{:04X}-{:04X}-{:02X}{:02X}-{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}",
        g.data1, g.data2, g.data3, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]
    )
}

/// A null-terminated UTF-16 buffer for a `PCWSTR` argument (kept alive by the caller's binding).
fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}
