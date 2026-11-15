//! The factory and device: top of the object graph. `FactoryObj` builds geometries, stroke styles,
//! drawing-state blocks and devices; `DeviceObj` builds device contexts.

use super::E_NOTIMPL;
use super::context::ContextObj;
use super::geometry::GeometryObj;
use super::image::EffectObj;
use super::resources::{StateBlockObj, StrokeStyleObj};
use d2d::geometry::stroke::{Cap, Join, TransformType};
use d2d::geometry::{Geometry, StrokeStyle};
use std::cell::RefCell;
use windows::Win32::Graphics::Direct2D::Common::*;
use windows::Win32::Graphics::Direct2D::*;
use windows::Win32::Graphics::DirectWrite::IDWriteRenderingParams;
use windows::Win32::Graphics::Dxgi::IDXGIDevice;
use windows_core::{IUnknownImpl, Interface, Ref, Result as WResult, implement};

fn cap_of(c: D2D1_CAP_STYLE) -> Cap {
    match c.0 {
        1 => Cap::Square,
        2 => Cap::Round,
        3 => Cap::Triangle,
        _ => Cap::Flat,
    }
}
fn join_of(j: D2D1_LINE_JOIN) -> Join {
    match j.0 {
        1 => Join::Bevel,
        2 => Join::Round,
        3 => Join::MiterOrBevel,
        _ => Join::Miter,
    }
}

/// Read a `CreateStrokeStyle`/`CreateStrokeStyle1` dash array into a `Vec` (null/empty → none).
/// The effective dash array for a stroke style: a custom array when the style is CUSTOM (or a
/// pattern was supplied), else D2D's predefined pattern for the dash style (in stroke-width units,
/// matching native's `GetDashes`). A predefined style otherwise renders solid — a real fingerprint.
fn dashes_of(style: D2D1_DASH_STYLE, dashes: *const f32, count: u32) -> Vec<f32> {
    if !dashes.is_null() && count != 0 {
        return unsafe { core::slice::from_raw_parts(dashes, count as usize) }.to_vec();
    }
    match style.0 {
        1 => vec![2.0, 2.0],                     // DASH
        2 => vec![0.0, 2.0],                     // DOT
        3 => vec![2.0, 2.0, 0.0, 2.0],           // DASH_DOT
        4 => vec![2.0, 2.0, 0.0, 2.0, 0.0, 2.0], // DASH_DOT_DOT
        _ => Vec::new(),                         // SOLID / CUSTOM-with-no-array
    }
}

#[implement(ID2D1Factory1, ID2D1Multithread)]
pub(crate) struct FactoryObj {
    factory: d2d::Factory,
    /// `ID2D1Multithread`'s critical section: the owning thread and its re-entry depth.
    section: (std::sync::Mutex<(Option<std::thread::ThreadId>, u32)>, std::sync::Condvar),
}
impl FactoryObj {
    pub(crate) fn new(multithreaded: bool) -> Self {
        Self { factory: d2d::Factory::new(multithreaded), section: Default::default() }
    }
}

/// `ID2D1Multithread`, which every native factory exposes; ComputeSharp.D2D1's resource texture
/// manager reaches it through an effect's factory before it creates its texture. A MULTI_THREADED
/// factory has a re-entrant critical section; a single-threaded one has none, so Enter/Leave do nothing.
impl ID2D1Multithread_Impl for FactoryObj_Impl {
    fn GetMultithreadProtected(&self) -> windows_core::BOOL {
        self.factory.is_multithreaded().into()
    }
    fn Enter(&self) {
        if !self.factory.is_multithreaded() {
            return;
        }
        let me = std::thread::current().id();
        let (lock, cv) = &self.section;
        let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
        while held.0.is_some_and(|owner| owner != me) {
            held = cv.wait(held).unwrap_or_else(|e| e.into_inner());
        }
        *held = (Some(me), held.1 + 1);
    }
    fn Leave(&self) {
        if !self.factory.is_multithreaded() {
            return;
        }
        let (lock, cv) = &self.section;
        let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
        if held.0 == Some(std::thread::current().id()) {
            held.1 -= 1;
            if held.1 == 0 {
                held.0 = None;
                cv.notify_one();
            }
        }
    }
}

thread_local! {
    /// The factory an effect object reports from `ID2D1Image::GetFactory` (see `EffectObj`).
    static STANDIN: ID2D1Factory = ID2D1Factory1::from(FactoryObj::new(false)).into();
}

/// A stand-in single-threaded factory, for an object that must report one but does not keep its own.
pub(crate) fn standin_factory() -> ID2D1Factory {
    STANDIN.with(Clone::clone)
}

impl FactoryObj_Impl {
    /// A counted `ID2D1Factory` reference to this factory, to hand to the resources it creates so
    /// their `GetFactory` returns the creating factory (native resources hold a strong factory ref).
    pub(crate) fn factory_iface(&self) -> ID2D1Factory {
        self.to_interface::<ID2D1Factory1>().cast::<ID2D1Factory>().unwrap()
    }
}

impl ID2D1Factory_Impl for FactoryObj_Impl {
    fn ReloadSystemMetrics(&self) -> WResult<()> {
        Ok(())
    }
    fn GetDesktopDpi(&self, dpix: *mut f32, dpiy: *mut f32) {
        unsafe {
            *dpix = 96.0;
            *dpiy = 96.0;
        }
    }
    fn CreateRectangleGeometry(&self, rectangle: *const D2D_RECT_F) -> WResult<ID2D1RectangleGeometry> {
        let r = unsafe { &*rectangle };
        Ok(GeometryObj::shape(Geometry::Rect([r.left, r.top, r.right, r.bottom])).with_factory(self.factory_iface()).into())
    }
    fn CreateRoundedRectangleGeometry(&self, rr: *const D2D1_ROUNDED_RECT) -> WResult<ID2D1RoundedRectangleGeometry> {
        let r = unsafe { &*rr };
        Ok(GeometryObj::shape(Geometry::Rounded {
            rect: [r.rect.left, r.rect.top, r.rect.right, r.rect.bottom],
            radius: [r.radiusX, r.radiusY],
        })
        .with_factory(self.factory_iface())
        .into())
    }
    fn CreateEllipseGeometry(&self, ellipse: *const D2D1_ELLIPSE) -> WResult<ID2D1EllipseGeometry> {
        let e = unsafe { &*ellipse };
        Ok(GeometryObj::shape(Geometry::Ellipse { center: [e.point.X, e.point.Y], radius: [e.radiusX, e.radiusY] }).with_factory(self.factory_iface()).into())
    }
    fn CreateGeometryGroup(&self, fillmode: D2D1_FILL_MODE, geometries: *const Option<ID2D1Geometry>, count: u32) -> WResult<ID2D1GeometryGroup> {
        let fill = if fillmode.0 == 1 { d2d::geometry::FillMode::Winding } else { d2d::geometry::FillMode::Alternate };
        let children: Vec<ID2D1Geometry> =
            (0..count as isize).filter_map(|i| unsafe { &*geometries.offset(i) }.clone()).collect();
        Ok(GeometryObj::group(fill, children).with_factory(self.factory_iface()).into())
    }
    fn CreateTransformedGeometry(&self, g: Ref<ID2D1Geometry>, transform: *const windows_numerics::Matrix3x2) -> WResult<ID2D1TransformedGeometry> {
        let source = g.as_ref().ok_or(windows_core::Error::from(E_NOTIMPL))?.clone();
        Ok(GeometryObj::transformed(super::mat_opt(transform), source).with_factory(self.factory_iface()).into())
    }
    fn CreatePathGeometry(&self) -> WResult<ID2D1PathGeometry> {
        Ok(GeometryObj::path().with_factory(self.factory_iface()).into())
    }
    fn CreateStrokeStyle(&self, props: *const D2D1_STROKE_STYLE_PROPERTIES, dashes: *const f32, count: u32) -> WResult<ID2D1StrokeStyle> {
        let p = unsafe { &*props };
        let obj: ID2D1StrokeStyle1 = StrokeStyleObj {
            style: StrokeStyle {
                start_cap: cap_of(p.startCap),
                end_cap: cap_of(p.endCap),
                dash_cap: cap_of(p.dashCap),
                join: join_of(p.lineJoin),
                miter_limit: p.miterLimit,
                dash_offset: p.dashOffset,
                dashes: dashes_of(p.dashStyle, dashes, count),
                transform_type: TransformType::Normal,
            },
            dash_style: p.dashStyle,
            factory: Some(self.factory_iface()),
        }
        .into();
        obj.cast()
    }
    fn CreateDrawingStateBlock(&self, d: *const D2D1_DRAWING_STATE_DESCRIPTION, p: Ref<IDWriteRenderingParams>) -> WResult<ID2D1DrawingStateBlock> {
        let mut state = self.factory.create_drawing_state_block();
        if !d.is_null() {
            // Native seeds the block from the passed description (NULL = default state).
            let v = unsafe { &*d };
            let m = v.transform;
            state.set_description([m.M11, m.M12, m.M21, m.M22, m.M31, m.M32], v.antialiasMode.0 as u32, v.textAntialiasMode.0 as u32, (v.tag1, v.tag2));
        }
        let obj: ID2D1DrawingStateBlock1 = StateBlockObj { state: RefCell::new(state), factory: Some(self.factory_iface()), text_params: RefCell::new(p.as_ref().cloned()) }.into();
        obj.cast()
    }
    fn CreateWicBitmapRenderTarget(&self, t: Ref<windows::Win32::Graphics::Imaging::IWICBitmap>, p: *const D2D1_RENDER_TARGET_PROPERTIES) -> WResult<ID2D1RenderTarget> {
        use windows::Win32::Graphics::Imaging::{GUID_WICPixelFormat32bppBGR, GUID_WICPixelFormat32bppPBGRA, IWICBitmapSource};
        let wic = t.as_ref().ok_or(windows_core::Error::from(windows_core::HRESULT(0x8007_0057u32 as i32)))?.clone();
        // D2D WIC render targets support 32bppPBGRA (premultiplied) and 32bppBGR (ignore) only; any
        // other WIC format gets D2DERR_UNSUPPORTED_PIXEL_FORMAT (= WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT),
        // as native returns. GetSize/GetPixelFormat live on the IWICBitmapSource base.
        let src: IWICBitmapSource = wic.cast()?;
        let fmt = unsafe { src.GetPixelFormat()? };
        if fmt != GUID_WICPixelFormat32bppPBGRA && fmt != GUID_WICPixelFormat32bppBGR {
            return Err(windows_core::HRESULT(0x8898_2F80u32 as i32).into());
        }
        let (mut w, mut h) = (0u32, 0u32);
        unsafe { src.GetSize(&mut w, &mut h)? };
        ContextObj::with_wic_target(&self.factory, self.factory_iface(), wic, w, h, super::rt_dpi(p)).cast()
    }
    fn CreateHwndRenderTarget(&self, r: *const D2D1_RENDER_TARGET_PROPERTIES, h: *const D2D1_HWND_RENDER_TARGET_PROPERTIES) -> WResult<ID2D1HwndRenderTarget> {
        #[cfg(not(windows))]
        return {
            let _ = (r, h);
            Err(E_NOTIMPL.into())
        };
        #[cfg(windows)]
        super::present::hwnd_render_target(&self.factory, self.factory_iface(), r, h)
    }
    fn CreateDxgiSurfaceRenderTarget(&self, s: Ref<windows::Win32::Graphics::Dxgi::IDXGISurface>, p: *const D2D1_RENDER_TARGET_PROPERTIES) -> WResult<ID2D1RenderTarget> {
        #[cfg(not(windows))]
        return {
            let _ = (s, p);
            Err(E_NOTIMPL.into())
        };
        #[cfg(windows)]
        super::present::dxgi_render_target(&self.factory, self.factory_iface(), s, p)
    }
    fn CreateDCRenderTarget(&self, p: *const D2D1_RENDER_TARGET_PROPERTIES) -> WResult<ID2D1DCRenderTarget> {
        #[cfg(not(windows))]
        return {
            let _ = p;
            Err(E_NOTIMPL.into())
        };
        #[cfg(windows)]
        super::present::dc_render_target(&self.factory, self.factory_iface(), p)
    }
}

impl ID2D1Factory1_Impl for FactoryObj_Impl {
    fn CreateDevice(&self, _dxgi: Ref<IDXGIDevice>) -> WResult<ID2D1Device> {
        Ok(DeviceObj::new(self.factory.create_device(), self.factory_iface()).into())
    }
    fn CreateStrokeStyle(&self, props: *const D2D1_STROKE_STYLE_PROPERTIES1, dashes: *const f32, count: u32) -> WResult<ID2D1StrokeStyle1> {
        let p = unsafe { &*props };
        let transform_type = match p.transformType.0 {
            1 => TransformType::Fixed,
            2 => TransformType::Hairline,
            _ => TransformType::Normal,
        };
        Ok(StrokeStyleObj {
            style: StrokeStyle {
                start_cap: cap_of(p.startCap),
                end_cap: cap_of(p.endCap),
                dash_cap: cap_of(p.dashCap),
                join: join_of(p.lineJoin),
                miter_limit: p.miterLimit,
                dash_offset: p.dashOffset,
                dashes: dashes_of(p.dashStyle, dashes, count),
                transform_type,
            },
            dash_style: p.dashStyle,
            factory: Some(self.factory_iface()),
        }
        .into())
    }
    fn CreatePathGeometry(&self) -> WResult<ID2D1PathGeometry1> {
        // Same path geometry as the base factory; native exposes it as ID2D1PathGeometry1, which
        // adds ComputePointAndSegmentAtLength (implemented on GeometryObj).
        Ok(GeometryObj::path().with_factory(self.factory_iface()).into())
    }
    fn CreateDrawingStateBlock(&self, d: *const D2D1_DRAWING_STATE_DESCRIPTION1, p: Ref<IDWriteRenderingParams>) -> WResult<ID2D1DrawingStateBlock1> {
        let mut state = self.factory.create_drawing_state_block();
        if !d.is_null() {
            let v = unsafe { &*d };
            let m = v.transform;
            state.set_description1([m.M11, m.M12, m.M21, m.M22, m.M31, m.M32], v.antialiasMode.0 as u32, v.textAntialiasMode.0 as u32, (v.tag1, v.tag2), v.primitiveBlend.0 as u32, v.unitMode.0 as u32);
        }
        Ok(StateBlockObj { state: RefCell::new(state), factory: Some(self.factory_iface()), text_params: RefCell::new(p.as_ref().cloned()) }.into())
    }
    fn CreateGdiMetafile(&self, s: Ref<windows::Win32::System::Com::IStream>) -> WResult<ID2D1GdiMetafile> {
        let bytes = s.as_ref().map(super::context::read_stream).unwrap_or_default();
        let mf = d2d::metafile::GdiMetafile::parse(bytes).ok_or(windows_core::Error::from(windows_core::HRESULT(0x8007_0057u32 as i32)))?;
        let obj: ID2D1GdiMetafile1 = super::metafile::MetafileObj { mf, factory: Some(self.factory_iface()) }.into();
        obj.cast()
    }
    fn RegisterEffectFromStream(&self, c: *const windows_core::GUID, x: Ref<windows::Win32::System::Com::IStream>, b: *const D2D1_PROPERTY_BINDING, bc: u32, f: PD2D1_EFFECT_FACTORY) -> WResult<()> {
        let xml = x.as_ref().map(super::context::read_stream).unwrap_or_default();
        super::author::register_from_stream(&self.factory, unsafe { *c }, &xml, f, b, bc)
    }
    fn RegisterEffectFromString(&self, c: *const windows_core::GUID, x: &windows_core::PCWSTR, b: *const D2D1_PROPERTY_BINDING, bc: u32, f: PD2D1_EFFECT_FACTORY) -> WResult<()> {
        let xml = unsafe { x.to_string() }.map_err(|_| windows_core::Error::from(super::E_NOTIMPL))?;
        super::author::register_from_string(&self.factory, unsafe { *c }, &xml, f, b, bc)
    }
    fn UnregisterEffect(&self, c: *const windows_core::GUID) -> WResult<()> {
        self.factory.unregister_effect(unsafe { *c }).map_err(windows_core::Error::from)
    }
    fn GetRegisteredEffects(&self, effects: *mut windows_core::GUID, effects_count: u32, returned: *mut u32, registered: *mut u32) -> WResult<()> {
        // Matches Direct2D's GetRegisteredEffects enumeration:
        // write up to `effects_count` CLSIDs (the system effects
        // followed by this factory's custom registrations), set `*returned` = written and
        // `*registered` = total, and return D2DERR_INSUFFICIENT_BUFFER if the buffer is too small. A
        // request to write (count>0) with no `effects` or no `returned` pointer is E_INVALIDARG.
        if !returned.is_null() {
            unsafe { *returned = 0 };
        }
        if !registered.is_null() {
            unsafe { *registered = 0 };
        }
        if effects_count > 0 && (effects.is_null() || returned.is_null()) {
            if !effects.is_null() {
                unsafe { *effects = windows_core::GUID::from_u128(0) }; // native writes GUID_NULL first
            }
            return Err(super::E_INVALIDARG.into());
        }
        let list = self.factory.registered_effect_clsids();
        let total = list.len() as u32;
        if !registered.is_null() {
            unsafe { *registered = total };
        }
        if !effects.is_null() {
            let n = effects_count.min(total);
            for (i, clsid) in list.iter().take(n as usize).enumerate() {
                unsafe { *effects.add(i) = *clsid };
            }
            if !returned.is_null() {
                unsafe { *returned = n };
            }
            if total > effects_count {
                return Err(windows_core::HRESULT(0x8899_003Au32 as i32).into()); // D2DERR_INSUFFICIENT_BUFFER
            }
        }
        Ok(())
    }
    fn GetEffectProperties(&self, e: *const windows_core::GUID) -> WResult<ID2D1Properties> {
        // Direct2D's GetEffectProperties returns a properties object reflecting the
        // effect's *registration* schema + default property values — which can differ from a created
        // instance (Flood's schema Color is {0,0,0,0}, a created Flood's is {0,0,0,1}). Build a transient
        // instance for the schema (names/types/count) but mark it schema-only so GetValue reports the
        // registration default.
        let clsid = unsafe { *e };
        let effect = self.factory.create_device().create_device_context().create_effect(clsid).map_err(windows_core::Error::from)?;
        let obj: ID2D1Effect = EffectObj::new(effect, clsid).into_schema().into();
        obj.cast()
    }
}

impl ID2D1Resource_Impl for FactoryObj_Impl {
    fn GetFactory(&self, f: windows_core::OutRef<ID2D1Factory>) {
        // A factory is its own factory. `to_interface` hands back a counted reference to this same
        // COM object (no stored self-cycle), matching native where GetFactory returns the factory.
        let _ = f.write(self.to_interface::<ID2D1Factory1>().cast::<ID2D1Factory>().ok());
    }
}

#[implement(ID2D1Device)]
pub(crate) struct DeviceObj {
    device: d2d::Device,
    factory: ID2D1Factory,
    /// `SetMaximumTextureMemory` value, round-tripped by the getter. Native's default is 128 MiB
    /// (a hint only — nothing enforces it here, but the value must read back).
    max_texture_memory: std::cell::Cell<u64>,
}
impl DeviceObj {
    pub(crate) fn new(device: d2d::Device, factory: ID2D1Factory) -> Self {
        Self { device, factory, max_texture_memory: std::cell::Cell::new(134_217_728) }
    }
}
impl ID2D1Resource_Impl for DeviceObj_Impl {
    fn GetFactory(&self, f: windows_core::OutRef<ID2D1Factory>) {
        let _ = f.write(Some(self.factory.clone()));
    }
}
impl ID2D1Device_Impl for DeviceObj_Impl {
    fn CreateDeviceContext(&self, _options: D2D1_DEVICE_CONTEXT_OPTIONS) -> WResult<ID2D1DeviceContext> {
        let dc: ID2D1DeviceContext7 = ContextObj::new(self.device.create_device_context(), self.factory.clone(), Some(self.to_interface::<ID2D1Device>())).into();
        dc.cast()
    }
    fn CreatePrintControl(&self, w: Ref<windows::Win32::Graphics::Imaging::IWICImagingFactory>, d: Ref<windows::Win32::Storage::Xps::Printing::IPrintDocumentPackageTarget>, p: *const D2D1_PRINT_CONTROL_PROPERTIES) -> WResult<ID2D1PrintControl> {
        // The print control serialises pages into an XPS-OM package — Windows XPS only.
        #[cfg(not(windows))]
        return {
            let _ = (w, d, p);
            Err(E_NOTIMPL.into())
        };
        #[cfg(windows)]
        super::print::create_print_control(w, d, p)
    }
    fn SetMaximumTextureMemory(&self, m: u64) {
        self.max_texture_memory.set(m);
    }
    fn GetMaximumTextureMemory(&self) -> u64 {
        self.max_texture_memory.get()
    }
    fn ClearResources(&self, _m: u32) {}
}
