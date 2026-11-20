//! Brushes, gradient-stop collections, stroke styles, drawing-state blocks and bitmaps. Each wraps
//! the matching `d2d` handle; the context recovers them with `cast_object_ref`/`cast_object`.

use super::{E_NOTIMPL, col, mat, vec2};

/// Presentation state for surface/HWND/DC-backed bitmaps. Those targets only exist on Windows
/// (`win::present`); off Windows the `BitmapObj::present` field is therefore always `None`.
#[cfg(windows)]
pub(crate) use super::present::Present;
#[cfg(not(windows))]
pub(crate) enum Present {}

/// `E_INVALIDARG` — the getters return it for an out-of-range segment/index (as native validates).
const E_INVALIDARG: windows_core::HRESULT = windows_core::HRESULT(0x8007_0057u32 as i32);
use d2d::bitmap::Bitmap;
use d2d::contracts::PixelBuffer;
use d2d::extras::ink::{Ink, InkBezierSegment, InkStyle};
use d2d::geometry::{Geometry, StrokeStyle};
use d2d::{DrawingStateBlock, PixelBrush, SolidColorBrush};
use std::cell::{Cell, RefCell};
use windows::Win32::Graphics::Direct2D::Common::*;
use windows::Win32::Graphics::Direct2D::*;
use windows::Win32::Graphics::DirectWrite::IDWriteRenderingParams;
use windows_core::{Interface, Ref, Result as WResult, implement};
use windows_numerics::{Matrix3x2, Vector2};

/// The identity brush transform native returns when none was set.
const IDENTITY_M: Matrix3x2 = Matrix3x2 { M11: 1.0, M12: 0.0, M21: 0.0, M22: 1.0, M31: 0.0, M32: 0.0 };

// --- solid brush -------------------------------------------------------------------------------

/// `brush` is the device-realised brush the rasteriser draws with; `color`/`opacity`/`transform` are
/// the authored values the getters echo back (native returns what was last set, even unrealised —
/// a `Set*` after creation updates these, not the already-realised `brush`).
#[implement(ID2D1SolidColorBrush)]
pub(crate) struct SolidBrushObj {
    pub(crate) brush: SolidColorBrush,
    color: Cell<D2D1_COLOR_F>,
    opacity: Cell<f32>,
    transform: Cell<Matrix3x2>,
    pub(crate) factory: Option<ID2D1Factory>,
}
impl SolidBrushObj {
    pub(crate) fn new(brush: SolidColorBrush, color: D2D1_COLOR_F, opacity: f32, transform: Matrix3x2) -> Self {
        Self { brush, color: Cell::new(color), opacity: Cell::new(opacity), transform: Cell::new(transform), factory: None }
    }
    /// Record the creating factory (chained before `.into()`), for `GetFactory`.
    pub(crate) fn with_factory(mut self, factory: ID2D1Factory) -> Self {
        self.factory = Some(factory);
        self
    }
    /// Wrap a core brush whose authored colour/opacity are its own, at identity transform (the
    /// command-stream replay's synthesised brushes).
    pub(crate) fn from_core(brush: SolidColorBrush) -> Self {
        let c = brush.color();
        Self::new(brush, D2D1_COLOR_F { r: c[0], g: c[1], b: c[2], a: c[3] }, brush.opacity(), IDENTITY_M)
    }
    /// The brush's *current* colour and opacity (post `SetColor`/`SetOpacity`), so a fill records the
    /// live state rather than the creation-time `brush` snapshot — native honours post-creation mutation.
    pub(crate) fn current(&self) -> (D2D1_COLOR_F, f32) {
        (self.color.get(), self.opacity.get())
    }
    /// The core brush a draw uses: rebuilt from the current colour and opacity, so every draw honours a post-creation
    /// `SetColor`/`SetOpacity` as native's does (it realizes brushes per draw).
    pub(crate) fn live(&self, dc: &d2d::DeviceContext) -> SolidColorBrush {
        let (c, o) = self.current();
        dc.create_solid_color_brush(col(&c), Some(o))
    }
}
impl ID2D1Resource_Impl for SolidBrushObj_Impl {
    fn GetFactory(&self, f: windows_core::OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
impl ID2D1Brush_Impl for SolidBrushObj_Impl {
    fn SetOpacity(&self, o: f32) {
        self.opacity.set(o);
    }
    fn SetTransform(&self, t: *const Matrix3x2) {
        if !t.is_null() {
            self.transform.set(unsafe { *t });
        }
    }
    fn GetOpacity(&self) -> f32 {
        self.opacity.get()
    }
    fn GetTransform(&self, t: *mut Matrix3x2) {
        if !t.is_null() {
            unsafe { *t = self.transform.get() };
        }
    }
}
impl ID2D1SolidColorBrush_Impl for SolidBrushObj_Impl {
    fn SetColor(&self, c: *const D2D1_COLOR_F) {
        if !c.is_null() {
            self.color.set(unsafe { *c });
        }
    }
    fn GetColor(&self) -> D2D1_COLOR_F {
        self.color.get()
    }
}

// --- gradient / bitmap / image brush (one type, three interfaces) ------------------------------

/// The authored parameters of whichever brush kind a `PixelBrushObj` was created as. The core
/// realises the brush and drops these, so the facade keeps them and returns them verbatim — exactly
/// what native's getters do, whatever the realisation. `Set*` mutates them in place.
pub(crate) enum BrushParams {
    Linear { start: Vector2, end: Vector2, stops: ID2D1GradientStopCollection },
    Radial { center: Vector2, offset: Vector2, radius: [f32; 2], stops: ID2D1GradientStopCollection },
    // A bitmap/image brush's interpolation is the full `D2D1_INTERPOLATION_MODE`; the legacy
    // `ID2D1BitmapBrush::GetInterpolationMode` reports NEAREST for mode 0, LINEAR otherwise.
    Bitmap { extend: [D2D1_EXTEND_MODE; 2], interp: D2D1_INTERPOLATION_MODE, bitmap: RefCell<Option<ID2D1Bitmap>> },
    Image { extend: [D2D1_EXTEND_MODE; 2], interp: D2D1_INTERPOLATION_MODE, source: D2D_RECT_F, image: RefCell<Option<ID2D1Image>> },
}

/// One object answers every brush interface, so the bitmap- and image-brush extend and interpolation setters and getters act on
/// whichever kind it was created as (an app holding an image brush through a bitmap-brush pointer, which native refuses).
/// ponytail: QueryInterface succeeds for the other kinds' interfaces where native returns E_NOINTERFACE.
#[implement(ID2D1LinearGradientBrush, ID2D1RadialGradientBrush, ID2D1ImageBrush, ID2D1BitmapBrush1)]
pub(crate) struct PixelBrushObj {
    /// The core brush as last realized; a setter marks it `dirty` and the next draw rebuilds it ([`Self::live`]).
    brush: RefCell<PixelBrush>,
    dirty: Cell<bool>,
    opacity: Cell<f32>,
    transform: Cell<Matrix3x2>,
    params: RefCell<BrushParams>,
    pub(crate) factory: Option<ID2D1Factory>,
}
impl PixelBrushObj {
    pub(crate) fn new(brush: PixelBrush, opacity: f32, transform: Matrix3x2, params: BrushParams) -> Self {
        Self { brush: RefCell::new(brush), dirty: Cell::new(false), opacity: Cell::new(opacity), transform: Cell::new(transform), params: RefCell::new(params), factory: None }
    }
    /// The core brush a draw uses: after any setter (opacity, transform, extend modes, interpolation, points, radii, source
    /// rectangle, bitmap or image), rebuilt from the brush's current state as native realizes brushes per draw (VM: a bitmap
    /// brush given its extend modes, interpolation, opacity and transform after creation draws as one created with them).
    pub(crate) fn live(&self, dc: &mut d2d::DeviceContext) -> PixelBrush {
        if self.dirty.replace(false) {
            if let Some(b) = self.rebuild(dc) {
                *self.brush.borrow_mut() = b;
            }
        }
        self.brush.borrow().clone()
    }
    fn rebuild(&self, dc: &mut d2d::DeviceContext) -> Option<PixelBrush> {
        let opacity = self.opacity.get();
        let m = mat(&self.transform.get());
        let transform = (m != mat(&IDENTITY_M)).then_some(m);
        match &*self.params.borrow() {
            BrushParams::Linear { start, end, stops } => {
                let so = stops.cast_object_ref::<StopsObj>().ok()?;
                Some(dc.create_linear_gradient_brush(vec2(start), vec2(end), &so.stops, so.gamma, so.extend, opacity, transform))
            }
            BrushParams::Radial { center, offset, radius, stops } => {
                let so = stops.cast_object_ref::<StopsObj>().ok()?;
                Some(dc.create_radial_gradient_brush(vec2(center), vec2(offset), *radius, &so.stops, so.gamma, so.extend, opacity, transform))
            }
            BrushParams::Bitmap { extend, interp, bitmap } => {
                let bmp = bitmap.borrow().clone()?;
                let obj = bmp.cast_object_ref::<BitmapObj>().ok()?;
                let src = obj.src.borrow();
                Some(dc.create_bitmap_brush(src.as_ref()?, [extend[0].0 as u32, extend[1].0 as u32], interp.0 as u32, opacity, transform))
            }
            BrushParams::Image { extend, interp, source, image } => {
                let img = image.borrow().clone()?;
                super::context::image_pixel_brush(dc, &img, source, [extend[0].0 as u32, extend[1].0 as u32], interp.0 as u32, opacity, transform).ok()
            }
        }
    }
    /// Record the creating factory (chained before `.into()`), for `GetFactory`.
    pub(crate) fn with_factory(mut self, factory: ID2D1Factory) -> Self {
        self.factory = Some(factory);
        self
    }
}
impl ID2D1Resource_Impl for PixelBrushObj_Impl {
    fn GetFactory(&self, f: windows_core::OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
impl ID2D1Brush_Impl for PixelBrushObj_Impl {
    fn SetOpacity(&self, o: f32) {
        self.dirty.set(true);
        self.opacity.set(o);
    }
    fn SetTransform(&self, t: *const Matrix3x2) {
        self.dirty.set(true);
        if !t.is_null() {
            self.transform.set(unsafe { *t });
        }
    }
    fn GetOpacity(&self) -> f32 {
        self.opacity.get()
    }
    fn GetTransform(&self, t: *mut Matrix3x2) {
        if !t.is_null() {
            unsafe { *t = self.transform.get() };
        }
    }
}
impl ID2D1LinearGradientBrush_Impl for PixelBrushObj_Impl {
    fn SetStartPoint(&self, p: &Vector2) {
        self.dirty.set(true);
        if let BrushParams::Linear { start, .. } = &mut *self.params.borrow_mut() {
            *start = *p;
        }
    }
    fn SetEndPoint(&self, p: &Vector2) {
        self.dirty.set(true);
        if let BrushParams::Linear { end, .. } = &mut *self.params.borrow_mut() {
            *end = *p;
        }
    }
    fn GetStartPoint(&self) -> Vector2 {
        match &*self.params.borrow() {
            BrushParams::Linear { start, .. } => *start,
            _ => Vector2::default(),
        }
    }
    fn GetEndPoint(&self) -> Vector2 {
        match &*self.params.borrow() {
            BrushParams::Linear { end, .. } => *end,
            _ => Vector2::default(),
        }
    }
    fn GetGradientStopCollection(&self, c: windows_core::OutRef<ID2D1GradientStopCollection>) {
        if let BrushParams::Linear { stops, .. } = &*self.params.borrow() {
            let _ = c.write(Some(stops.clone()));
        }
    }
}
impl ID2D1RadialGradientBrush_Impl for PixelBrushObj_Impl {
    fn SetCenter(&self, p: &Vector2) {
        self.dirty.set(true);
        if let BrushParams::Radial { center, .. } = &mut *self.params.borrow_mut() {
            *center = *p;
        }
    }
    fn SetGradientOriginOffset(&self, p: &Vector2) {
        self.dirty.set(true);
        if let BrushParams::Radial { offset, .. } = &mut *self.params.borrow_mut() {
            *offset = *p;
        }
    }
    fn SetRadiusX(&self, r: f32) {
        self.dirty.set(true);
        if let BrushParams::Radial { radius, .. } = &mut *self.params.borrow_mut() {
            radius[0] = r;
        }
    }
    fn SetRadiusY(&self, r: f32) {
        self.dirty.set(true);
        if let BrushParams::Radial { radius, .. } = &mut *self.params.borrow_mut() {
            radius[1] = r;
        }
    }
    fn GetCenter(&self) -> Vector2 {
        match &*self.params.borrow() {
            BrushParams::Radial { center, .. } => *center,
            _ => Vector2::default(),
        }
    }
    fn GetGradientOriginOffset(&self) -> Vector2 {
        match &*self.params.borrow() {
            BrushParams::Radial { offset, .. } => *offset,
            _ => Vector2::default(),
        }
    }
    fn GetRadiusX(&self) -> f32 {
        match &*self.params.borrow() {
            BrushParams::Radial { radius, .. } => radius[0],
            _ => 0.0,
        }
    }
    fn GetRadiusY(&self) -> f32 {
        match &*self.params.borrow() {
            BrushParams::Radial { radius, .. } => radius[1],
            _ => 0.0,
        }
    }
    fn GetGradientStopCollection(&self, c: windows_core::OutRef<ID2D1GradientStopCollection>) {
        if let BrushParams::Radial { stops, .. } = &*self.params.borrow() {
            let _ = c.write(Some(stops.clone()));
        }
    }
}
impl ID2D1ImageBrush_Impl for PixelBrushObj_Impl {
    fn SetImage(&self, i: Ref<ID2D1Image>) {
        self.dirty.set(true);
        if let BrushParams::Image { image, .. } = &*self.params.borrow() {
            *image.borrow_mut() = i.as_ref().map(|i| i.clone());
        }
    }
    fn SetExtendModeX(&self, m: D2D1_EXTEND_MODE) {
        self.dirty.set(true);
        if let BrushParams::Image { extend, .. } | BrushParams::Bitmap { extend, .. } = &mut *self.params.borrow_mut() {
            extend[0] = m;
        }
    }
    fn SetExtendModeY(&self, m: D2D1_EXTEND_MODE) {
        self.dirty.set(true);
        if let BrushParams::Image { extend, .. } | BrushParams::Bitmap { extend, .. } = &mut *self.params.borrow_mut() {
            extend[1] = m;
        }
    }
    fn SetInterpolationMode(&self, m: D2D1_INTERPOLATION_MODE) {
        self.dirty.set(true);
        if let BrushParams::Image { interp, .. } | BrushParams::Bitmap { interp, .. } = &mut *self.params.borrow_mut() {
            *interp = m;
        }
    }
    fn SetSourceRectangle(&self, r: *const D2D_RECT_F) {
        self.dirty.set(true);
        if let (false, BrushParams::Image { source, .. }) = (r.is_null(), &mut *self.params.borrow_mut()) {
            *source = unsafe { *r };
        }
    }
    fn GetImage(&self, i: windows_core::OutRef<ID2D1Image>) {
        if let BrushParams::Image { image, .. } = &*self.params.borrow() {
            let _ = i.write(image.borrow().clone());
        }
    }
    fn GetExtendModeX(&self) -> D2D1_EXTEND_MODE {
        match &*self.params.borrow() {
            BrushParams::Image { extend, .. } | BrushParams::Bitmap { extend, .. } => extend[0],
            _ => D2D1_EXTEND_MODE(0),
        }
    }
    fn GetExtendModeY(&self) -> D2D1_EXTEND_MODE {
        match &*self.params.borrow() {
            BrushParams::Image { extend, .. } | BrushParams::Bitmap { extend, .. } => extend[1],
            _ => D2D1_EXTEND_MODE(0),
        }
    }
    fn GetInterpolationMode(&self) -> D2D1_INTERPOLATION_MODE {
        match &*self.params.borrow() {
            BrushParams::Image { interp, .. } | BrushParams::Bitmap { interp, .. } => *interp,
            _ => D2D1_INTERPOLATION_MODE(0),
        }
    }
    fn GetSourceRectangle(&self, r: *mut D2D_RECT_F) {
        if let (false, BrushParams::Image { source, .. }) = (r.is_null(), &*self.params.borrow()) {
            unsafe { *r = *source };
        }
    }
}
impl ID2D1BitmapBrush_Impl for PixelBrushObj_Impl {
    fn SetExtendModeX(&self, m: D2D1_EXTEND_MODE) {
        self.dirty.set(true);
        if let BrushParams::Bitmap { extend, .. } | BrushParams::Image { extend, .. } = &mut *self.params.borrow_mut() {
            extend[0] = m;
        }
    }
    fn SetExtendModeY(&self, m: D2D1_EXTEND_MODE) {
        self.dirty.set(true);
        if let BrushParams::Bitmap { extend, .. } | BrushParams::Image { extend, .. } = &mut *self.params.borrow_mut() {
            extend[1] = m;
        }
    }
    fn SetInterpolationMode(&self, m: D2D1_BITMAP_INTERPOLATION_MODE) {
        self.dirty.set(true);
        // Legacy setter: NEAREST/LINEAR map onto the full mode.
        if let BrushParams::Bitmap { interp, .. } | BrushParams::Image { interp, .. } = &mut *self.params.borrow_mut() {
            *interp = D2D1_INTERPOLATION_MODE(m.0);
        }
    }
    fn SetBitmap(&self, b: Ref<ID2D1Bitmap>) {
        self.dirty.set(true);
        if let BrushParams::Bitmap { bitmap, .. } = &*self.params.borrow() {
            *bitmap.borrow_mut() = b.as_ref().map(|b| b.clone());
        }
    }
    fn GetExtendModeX(&self) -> D2D1_EXTEND_MODE {
        match &*self.params.borrow() {
            BrushParams::Bitmap { extend, .. } | BrushParams::Image { extend, .. } => extend[0],
            _ => D2D1_EXTEND_MODE(0),
        }
    }
    fn GetExtendModeY(&self) -> D2D1_EXTEND_MODE {
        match &*self.params.borrow() {
            BrushParams::Bitmap { extend, .. } | BrushParams::Image { extend, .. } => extend[1],
            _ => D2D1_EXTEND_MODE(0),
        }
    }
    fn GetInterpolationMode(&self) -> D2D1_BITMAP_INTERPOLATION_MODE {
        match &*self.params.borrow() {
            BrushParams::Bitmap { interp, .. } | BrushParams::Image { interp, .. } => {
                D2D1_BITMAP_INTERPOLATION_MODE((*interp != D2D1_INTERPOLATION_MODE_NEAREST_NEIGHBOR) as i32)
            }
            _ => D2D1_BITMAP_INTERPOLATION_MODE(0),
        }
    }
    fn GetBitmap(&self, b: windows_core::OutRef<ID2D1Bitmap>) {
        if let BrushParams::Bitmap { bitmap, .. } = &*self.params.borrow() {
            let _ = b.write(bitmap.borrow().clone());
        }
    }
}
impl ID2D1BitmapBrush1_Impl for PixelBrushObj_Impl {
    fn SetInterpolationMode1(&self, m: D2D1_INTERPOLATION_MODE) {
        self.dirty.set(true);
        if let BrushParams::Bitmap { interp, .. } | BrushParams::Image { interp, .. } = &mut *self.params.borrow_mut() {
            *interp = m;
        }
    }
    fn GetInterpolationMode1(&self) -> D2D1_INTERPOLATION_MODE {
        match &*self.params.borrow() {
            BrushParams::Bitmap { interp, .. } | BrushParams::Image { interp, .. } => *interp,
            _ => D2D1_INTERPOLATION_MODE(0),
        }
    }
}

/// x86_64-pc-windows-gnu only: the real `windows` crate emits the four D2D1_POINT_2F getters as a
/// by-value `-> Vector2` COM return, and rustc-gnu mis-lowers that *definition* so the value never
/// reaches the caller (the harness's `-> Vector2` *call* is lowered to the member ABI correctly — it
/// reads the right point from native MSVC d2d1). We can't touch the real crate, so after a brush is
/// created we overwrite just those four vtable slots with explicit member-ABI thunks
/// (`fn(this, *mut Vector2) -> *mut Vector2` — a hidden result pointer as the second argument,
/// returned in RAX), which rustc lowers unambiguously. Local to the linear/radial brush, nothing else
/// changes. Off Windows the shim's `fix-udt-return-abi.py` already emits the out-param form, and
/// windows-msvc is correct as-is, so this is gnu-only.
#[cfg(all(windows, target_env = "gnu"))]
pub(crate) mod gnu_point_abi {
    use super::{BrushParams, PixelBrushObj};
    use core::ffi::c_void;
    use std::sync::OnceLock;
    use windows::Win32::Graphics::Direct2D::{ID2D1LinearGradientBrush, ID2D1RadialGradientBrush};
    use windows_core::Interface;
    use windows_numerics::Vector2;

    /// Recover the brush impl from any of its interface pointers (QI via `cast_object_ref`) and write
    /// the picked authored point through the caller's result pointer.
    unsafe fn write_point(this: *mut c_void, r: *mut Vector2, pick: fn(&BrushParams) -> Vector2) {
        if let Some(u) = windows_core::IUnknown::from_raw_borrowed(&this) {
            if let Ok(o) = u.cast_object_ref::<PixelBrushObj>() {
                unsafe { *r = pick(&o.params.borrow()) };
            }
        }
    }
    unsafe extern "system" fn lin_start(this: *mut c_void, r: *mut Vector2) -> *mut Vector2 {
        unsafe { write_point(this, r, |p| if let BrushParams::Linear { start, .. } = p { *start } else { Vector2::default() }) };
        r
    }
    unsafe extern "system" fn lin_end(this: *mut c_void, r: *mut Vector2) -> *mut Vector2 {
        unsafe { write_point(this, r, |p| if let BrushParams::Linear { end, .. } = p { *end } else { Vector2::default() }) };
        r
    }
    unsafe extern "system" fn rad_center(this: *mut c_void, r: *mut Vector2) -> *mut Vector2 {
        unsafe { write_point(this, r, |p| if let BrushParams::Radial { center, .. } = p { *center } else { Vector2::default() }) };
        r
    }
    unsafe extern "system" fn rad_offset(this: *mut c_void, r: *mut Vector2) -> *mut Vector2 {
        unsafe { write_point(this, r, |p| if let BrushParams::Radial { offset, .. } = p { *offset } else { Vector2::default() }) };
        r
    }

    /// Repoint `raw`'s vtable at a leaked copy with `slots` replaced. Every brush of a kind shares one
    /// original (static) vtable, so the patched copy is built once and cached.
    unsafe fn install(raw: *mut c_void, n: usize, slots: [(usize, *const c_void); 2], cache: &OnceLock<usize>) {
        let vtp = raw as *mut *const c_void; // &(object's vtable pointer)
        let orig = unsafe { *vtp } as *const *const c_void;
        let patched = *cache.get_or_init(|| {
            let mut v: Vec<*const c_void> = (0..n).map(|i| unsafe { *orig.add(i) }).collect();
            for (s, f) in slots {
                v[s] = f;
            }
            Box::leak(v.into_boxed_slice()).as_ptr() as usize
        });
        unsafe { *vtp = patched as *const c_void };
    }
    pub(crate) fn patch_linear(b: &ID2D1LinearGradientBrush) {
        static C: OnceLock<usize> = OnceLock::new();
        unsafe { install(b.as_raw(), 13, [(10, lin_start as *const c_void), (11, lin_end as *const c_void)], &C) };
    }
    pub(crate) fn patch_radial(b: &ID2D1RadialGradientBrush) {
        static C: OnceLock<usize> = OnceLock::new();
        unsafe { install(b.as_raw(), 17, [(12, rad_center as *const c_void), (13, rad_offset as *const c_void)], &C) };
    }
}

// --- gradient stop collection ------------------------------------------------------------------

#[implement(ID2D1GradientStopCollection1)]
pub(crate) struct StopsObj {
    pub(crate) stops: Vec<(f32, [f32; 4])>,
    pub(crate) gamma: u32,
    pub(crate) extend: u32,
    /// The `...DESCRIPTION1` fields the `ID2D1GradientStopCollection1` getters report. The base
    /// `CreateGradientStopCollection(gamma, extend)` maps to these exactly as native's
    /// Direct2D's `CreateGradientStopCollection` (the base overload) does: pre =
    /// post = SRGB, precision = 8BPC_UNORM, mode = STRAIGHT, regardless of gamma (gamma only drives the
    /// internal interpolation space, not these public getters). The `...1` overload stores them verbatim.
    pub(crate) pre_space: u32,
    pub(crate) post_space: u32,
    pub(crate) precision: u32,
    pub(crate) interp_mode: u32,
    factory: Option<ID2D1Factory>,
}
impl StopsObj {
    /// Read `count` D2D stops into the core's `(position, color)` list.
    pub(crate) unsafe fn read(stops: *const D2D1_GRADIENT_STOP, count: u32, gamma: u32, extend: u32) -> Self {
        let stops = (0..count as isize)
            .map(|i| {
                let s = unsafe { &*stops.offset(i) };
                (s.position, col(&s.color))
            })
            .collect();
        // Base-overload `...1` defaults, per Direct2D's CreateGradientStopCollection.
        Self { stops, gamma, extend, pre_space: 1, post_space: 1, precision: 1, interp_mode: 0, factory: None }
    }
    /// Read `count` stops with the full `...DESCRIPTION1` state (the `CreateGradientStopCollection1`
    /// overload): pre/post colour space, buffer precision and colour-interpolation mode round-trip
    /// verbatim; `gamma` is derived back from `pre_space` (SCRGB ⇒ 1.0, else 2.2) for the inherited
    /// `GetColorInterpolationGamma`.
    pub(crate) unsafe fn read1(stops: *const D2D1_GRADIENT_STOP, count: u32, pre_space: u32, post_space: u32, precision: u32, extend: u32, interp_mode: u32) -> Self {
        let stops = (0..count as isize)
            .map(|i| {
                let s = unsafe { &*stops.offset(i) };
                (s.position, col(&s.color))
            })
            .collect();
        let gamma = if pre_space == 2 { 1 } else { 0 }; // SCRGB ⇒ GAMMA_1_0, else GAMMA_2_2
        Self { stops, gamma, extend, pre_space, post_space, precision, interp_mode, factory: None }
    }
    /// Record the creating factory (chained before `.into()`), for `GetFactory`.
    pub(crate) fn with_factory(mut self, factory: ID2D1Factory) -> Self {
        self.factory = Some(factory);
        self
    }
}
impl ID2D1Resource_Impl for StopsObj_Impl {
    fn GetFactory(&self, f: windows_core::OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
impl ID2D1GradientStopCollection_Impl for StopsObj_Impl {
    fn GetGradientStopCount(&self) -> u32 {
        self.stops.len() as u32
    }
    fn GetGradientStops(&self, s: *mut D2D1_GRADIENT_STOP, c: u32) {
        // Copy out the first `c` stops (native fills exactly min(c, count), leaving any excess slots
        // untouched).
        for (i, (pos, col)) in self.stops.iter().take(c as usize).enumerate() {
            unsafe {
                *s.add(i) = D2D1_GRADIENT_STOP { position: *pos, color: D2D1_COLOR_F { r: col[0], g: col[1], b: col[2], a: col[3] } };
            }
        }
    }
    fn GetColorInterpolationGamma(&self) -> D2D1_GAMMA {
        D2D1_GAMMA(self.gamma as i32)
    }
    fn GetExtendMode(&self) -> D2D1_EXTEND_MODE {
        D2D1_EXTEND_MODE(self.extend as i32)
    }
}
impl ID2D1GradientStopCollection1_Impl for StopsObj_Impl {
    fn GetGradientStops1(&self, s: *mut D2D1_GRADIENT_STOP, c: u32) {
        // The …1 stops are the same (position, colour) pairs; native stores one list.
        self.GetGradientStops(s, c);
    }
    fn GetPreInterpolationSpace(&self) -> D2D1_COLOR_SPACE {
        D2D1_COLOR_SPACE(self.pre_space as i32)
    }
    fn GetPostInterpolationSpace(&self) -> D2D1_COLOR_SPACE {
        D2D1_COLOR_SPACE(self.post_space as i32)
    }
    fn GetBufferPrecision(&self) -> D2D1_BUFFER_PRECISION {
        D2D1_BUFFER_PRECISION(self.precision as i32)
    }
    fn GetColorInterpolationMode(&self) -> D2D1_COLOR_INTERPOLATION_MODE {
        D2D1_COLOR_INTERPOLATION_MODE(self.interp_mode as i32)
    }
}

// --- layer (ID2D1Layer: a reusable offscreen a PushLayer composites into) ----------------------

/// `ID2D1Layer` is an opaque reusable-offscreen handle: native allocates the backing surface lazily
/// at PushLayer, so the object only carries the size hint `CreateLayer` was given (0 = size-at-push).
#[implement(ID2D1Layer)]
pub(crate) struct LayerObj {
    pub(crate) size: windows::Win32::Graphics::Direct2D::Common::D2D_SIZE_F,
    pub(crate) factory: Option<ID2D1Factory>,
}
impl ID2D1Resource_Impl for LayerObj_Impl {
    fn GetFactory(&self, f: windows_core::OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
impl ID2D1Layer_Impl for LayerObj_Impl {
    fn GetSize(&self) -> windows::Win32::Graphics::Direct2D::Common::D2D_SIZE_F {
        self.size
    }
}

// --- SVG glyph style (ID2D1SvgGlyphStyle: the fill/stroke an OT-SVG context-fill resolves to) ----

/// `ID2D1SvgGlyphStyle` holds the fill + stroke a `DrawSvgGlyphRun` uses for `context-fill`/
/// `context-stroke`. A plain value holder: set/get round-trip, no rendering state. Default (native):
/// no fill/stroke brush, stroke width 1.0, no dashes.
#[implement(ID2D1SvgGlyphStyle)]
pub(crate) struct SvgGlyphStyleObj {
    fill: RefCell<Option<ID2D1Brush>>,
    stroke: RefCell<Option<ID2D1Brush>>,
    stroke_width: std::cell::Cell<f32>,
    dashes: RefCell<Vec<f32>>,
    dash_offset: std::cell::Cell<f32>,
    pub(crate) factory: Option<ID2D1Factory>,
}
impl SvgGlyphStyleObj {
    pub(crate) fn new(factory: ID2D1Factory) -> Self {
        Self { fill: RefCell::new(None), stroke: RefCell::new(None), stroke_width: std::cell::Cell::new(1.0), dashes: RefCell::new(Vec::new()), dash_offset: std::cell::Cell::new(0.0), factory: Some(factory) }
    }
}
impl ID2D1Resource_Impl for SvgGlyphStyleObj_Impl {
    fn GetFactory(&self, f: windows_core::OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
impl ID2D1SvgGlyphStyle_Impl for SvgGlyphStyleObj_Impl {
    fn SetFill(&self, brush: Ref<ID2D1Brush>) -> WResult<()> {
        *self.fill.borrow_mut() = brush.as_ref().cloned();
        Ok(())
    }
    fn GetFill(&self, brush: windows_core::OutRef<ID2D1Brush>) {
        let _ = brush.write(self.fill.borrow().clone());
    }
    fn SetStroke(&self, brush: Ref<ID2D1Brush>, strokewidth: f32, dashes: *const f32, dashescount: u32, dashoffset: f32) -> WResult<()> {
        *self.stroke.borrow_mut() = brush.as_ref().cloned();
        self.stroke_width.set(strokewidth);
        self.dash_offset.set(dashoffset);
        *self.dashes.borrow_mut() = if dashes.is_null() { Vec::new() } else { unsafe { core::slice::from_raw_parts(dashes, dashescount as usize) }.to_vec() };
        Ok(())
    }
    fn GetStrokeDashesCount(&self) -> u32 {
        self.dashes.borrow().len() as u32
    }
    fn GetStroke(&self, brush: windows_core::OutRef<ID2D1Brush>, strokewidth: *mut f32, dashes: *mut f32, dashescount: u32, dashoffset: *mut f32) {
        let _ = brush.write(self.stroke.borrow().clone());
        unsafe {
            if !strokewidth.is_null() {
                *strokewidth = self.stroke_width.get();
            }
            if !dashoffset.is_null() {
                *dashoffset = self.dash_offset.get();
            }
            let d = self.dashes.borrow();
            for (i, v) in d.iter().take(dashescount as usize).enumerate() {
                *dashes.add(i) = *v;
            }
        }
    }
}

// --- stroke style ------------------------------------------------------------------------------

#[implement(ID2D1StrokeStyle1)]
pub(crate) struct StrokeStyleObj {
    pub(crate) style: d2d::geometry::StrokeStyle,
    /// The dash style the caller created with (DASH/DOT/… or CUSTOM), kept verbatim for GetDashStyle —
    /// `style.dashes` holds the *effective* array (predefined patterns expanded), which can't recover it.
    pub(crate) dash_style: D2D1_DASH_STYLE,
    pub(crate) factory: Option<ID2D1Factory>,
}
impl ID2D1Resource_Impl for StrokeStyleObj_Impl {
    fn GetFactory(&self, f: windows_core::OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
impl ID2D1StrokeStyle_Impl for StrokeStyleObj_Impl {
    fn GetStartCap(&self) -> D2D1_CAP_STYLE {
        D2D1_CAP_STYLE(cap_of(self.style.start_cap))
    }
    fn GetEndCap(&self) -> D2D1_CAP_STYLE {
        D2D1_CAP_STYLE(cap_of(self.style.end_cap))
    }
    fn GetDashCap(&self) -> D2D1_CAP_STYLE {
        D2D1_CAP_STYLE(cap_of(self.style.dash_cap))
    }
    fn GetMiterLimit(&self) -> f32 {
        self.style.miter_limit
    }
    fn GetLineJoin(&self) -> D2D1_LINE_JOIN {
        use d2d::geometry::stroke::Join;
        D2D1_LINE_JOIN(match self.style.join {
            Join::Miter => 0,
            Join::Bevel => 1,
            Join::Round => 2,
            Join::MiterOrBevel => 3,
        })
    }
    fn GetDashOffset(&self) -> f32 {
        self.style.dash_offset
    }
    fn GetDashStyle(&self) -> D2D1_DASH_STYLE {
        self.dash_style
    }
    fn GetDashesCount(&self) -> u32 {
        self.style.dashes.len() as u32
    }
    fn GetDashes(&self, d: *mut f32, c: u32) {
        for (i, v) in self.style.dashes.iter().take(c as usize).enumerate() {
            unsafe { *d.add(i) = *v };
        }
    }
}

/// Core [`Cap`](d2d::geometry::stroke::Cap) → `D2D1_CAP_STYLE` value.
fn cap_of(c: d2d::geometry::stroke::Cap) -> i32 {
    use d2d::geometry::stroke::Cap;
    match c {
        Cap::Flat => 0,
        Cap::Square => 1,
        Cap::Round => 2,
        Cap::Triangle => 3,
    }
}
impl ID2D1StrokeStyle1_Impl for StrokeStyleObj_Impl {
    fn GetStrokeTransformType(&self) -> D2D1_STROKE_TRANSFORM_TYPE {
        use d2d::geometry::stroke::TransformType;
        D2D1_STROKE_TRANSFORM_TYPE(match self.style.transform_type {
            TransformType::Fixed => 1,
            TransformType::Hairline => 2,
            TransformType::Normal => 0,
        })
    }
}

// --- drawing state block -----------------------------------------------------------------------

#[implement(ID2D1DrawingStateBlock1)]
pub(crate) struct StateBlockObj {
    pub(crate) state: RefCell<DrawingStateBlock>,
    pub(crate) factory: Option<ID2D1Factory>,
    /// The text rendering params the block holds (SetTextRenderingParams / GetTextRenderingParams).
    /// `None` is native's default (GetTextRenderingParams returns NULL).
    pub(crate) text_params: RefCell<Option<IDWriteRenderingParams>>,
}
impl ID2D1Resource_Impl for StateBlockObj_Impl {
    fn GetFactory(&self, f: windows_core::OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
impl ID2D1DrawingStateBlock_Impl for StateBlockObj_Impl {
    fn GetDescription(&self, d: *mut D2D1_DRAWING_STATE_DESCRIPTION) {
        // The block's saved state (filled by SaveDrawingState or SetDescription), as native returns it.
        if !d.is_null() {
            let (t, aa, text_aa, tags) = self.state.borrow().description();
            unsafe {
                *d = D2D1_DRAWING_STATE_DESCRIPTION {
                    antialiasMode: D2D1_ANTIALIAS_MODE(aa as i32),
                    textAntialiasMode: D2D1_TEXT_ANTIALIAS_MODE(text_aa as i32),
                    tag1: tags.0,
                    tag2: tags.1,
                    transform: Matrix3x2 { M11: t[0], M12: t[1], M21: t[2], M22: t[3], M31: t[4], M32: t[5] },
                };
            }
        }
    }
    fn SetDescription(&self, d: *const D2D1_DRAWING_STATE_DESCRIPTION) {
        if d.is_null() {
            return;
        }
        let d = unsafe { &*d };
        let m = d.transform;
        self.state.borrow_mut().set_description(
            [m.M11, m.M12, m.M21, m.M22, m.M31, m.M32],
            d.antialiasMode.0 as u32,
            d.textAntialiasMode.0 as u32,
            (d.tag1, d.tag2),
        );
    }
    fn SetTextRenderingParams(&self, p: Ref<IDWriteRenderingParams>) {
        *self.text_params.borrow_mut() = p.as_ref().cloned();
    }
    fn GetTextRenderingParams(&self, p: windows_core::OutRef<IDWriteRenderingParams>) {
        // The block's stored params (NULL = native's default state).
        let _ = p.write(self.text_params.borrow().clone());
    }
}
impl ID2D1DrawingStateBlock1_Impl for StateBlockObj_Impl {
    fn GetDescription(&self, d: *mut D2D1_DRAWING_STATE_DESCRIPTION1) {
        // The …1 description adds primitiveBlend + unitMode to the base four, as native returns them.
        if !d.is_null() {
            let (t, aa, text_aa, tags, blend, unit) = self.state.borrow().description1();
            unsafe {
                *d = D2D1_DRAWING_STATE_DESCRIPTION1 {
                    antialiasMode: D2D1_ANTIALIAS_MODE(aa as i32),
                    textAntialiasMode: D2D1_TEXT_ANTIALIAS_MODE(text_aa as i32),
                    tag1: tags.0,
                    tag2: tags.1,
                    transform: Matrix3x2 { M11: t[0], M12: t[1], M21: t[2], M22: t[3], M31: t[4], M32: t[5] },
                    primitiveBlend: D2D1_PRIMITIVE_BLEND(blend as i32),
                    unitMode: D2D1_UNIT_MODE(unit as i32),
                };
            }
        }
    }
    fn SetDescription(&self, d: *const D2D1_DRAWING_STATE_DESCRIPTION1) {
        if d.is_null() {
            return;
        }
        let d = unsafe { &*d };
        let m = d.transform;
        self.state.borrow_mut().set_description1(
            [m.M11, m.M12, m.M21, m.M22, m.M31, m.M32],
            d.antialiasMode.0 as u32,
            d.textAntialiasMode.0 as u32,
            (d.tag1, d.tag2),
            d.primitiveBlend.0 as u32,
            d.unitMode.0 as u32,
        );
    }
}

// --- bitmap ------------------------------------------------------------------------------------

/// A bitmap: a render-target/staging surface owns a `PixelBuffer`; an inline source image owns a
/// `Bitmap`. The context takes/returns `buf` as the target binds/unbinds. `present`, when set, names
/// where the pixels go at `EndDraw` (a window/DC/DXGI surface — see [`super::present`]).
#[implement(ID2D1Bitmap1)]
pub(crate) struct BitmapObj {
    pub(crate) pixels: D2D_SIZE_U,
    pub(crate) dpi: (f32, f32),
    pub(crate) buf: RefCell<Option<PixelBuffer>>,
    /// The source-image pixels (an inline/WIC image, or a surface written by `CopyFrom*`). Interior-
    /// mutable so `CopyFromMemory`/`CopyFromBitmap`/`CopyFromRenderTarget` can fill it in place.
    pub(crate) src: RefCell<Option<Bitmap>>,
    /// The `D2D1_BITMAP_OPTIONS` the bitmap was created with — `GetOptions` returns these verbatim.
    pub(crate) options: D2D1_BITMAP_OPTIONS,
    // Read only on Windows (`GetSurface`, the Hwnd/DC present path); off Windows always `None`.
    #[cfg_attr(not(windows), allow(dead_code))]
    pub(crate) present: RefCell<Option<Present>>,
    pub(crate) factory: Option<ID2D1Factory>,
    /// The `(factory_id, device_id)` of the device context that created this bitmap, so a draw of it on
    /// another context latches WRONG_FACTORY / WRONG_RESOURCE_DOMAIN like native (the brush path carries
    /// `Domain` in core; bitmaps are a plain core buffer, so the FFI records it). `None` = don't check
    /// (a 1.0 render-target / present bitmap, which native does not device-scope).
    pub(crate) domain_ids: Option<(u64, u64)>,
    /// The color context the bitmap was created with (`GetColorContext`), if any.
    pub(crate) color_context: Option<ID2D1ColorContext>,
}
impl BitmapObj {
    /// Retain the last completed pixels while the context draws into its working buffer.
    /// EndDraw publishes the completed buffer back here, so CopyFromBitmap and format getters
    /// keep working without requiring the caller to unbind its target.
    pub(crate) fn buffer_clone(&self) -> Option<PixelBuffer> {
        self.buf.borrow().clone()
    }
    pub(crate) fn set_buffer(&self, buf: PixelBuffer) {
        *self.buf.borrow_mut() = Some(buf);
    }
    /// A clone of the source image, if this bitmap carries one.
    pub(crate) fn src_clone(&self) -> Option<Bitmap> {
        self.src.borrow().clone()
    }
    /// The premultiplied-BGRA pixels a `CopyFromBitmap` reads out of this bitmap: the uploaded `src`
    /// if present, else the rendered target `buf` tight-packed. The buf fallback is what lets an
    /// offscreen app read a target back the standard way — render to a TARGET bitmap, CopyFromBitmap
    /// it into a CPU-read staging bitmap, Map that — exactly as the harness capture does. Only the
    /// texel bytes are retained in their original format, including float render targets.
    pub(crate) fn readable_src(&self) -> Option<Bitmap> {
        if let Some(s) = self.src.borrow().as_ref() {
            return Some(s.clone());
        }
        let b = self.buf.borrow();
        let buf = b.as_ref()?;
        let (w, h) = (buf.desc.width, buf.desc.height);
        let row = w as usize * d2d::bitmap::bytes_per_pixel(buf.desc.format)?;
        let mut premul = Vec::with_capacity(row * h as usize);
        for y in 0..h as usize {
            premul.extend_from_slice(&buf.data[y * buf.stride..y * buf.stride + row]);
        }
        Some(Bitmap { width: w, height: h, premul_bgra: premul, format: buf.desc.format, alpha: buf.desc.alpha_mode, dpi: (buf.desc.dpi_x, buf.desc.dpi_y) })
    }
    /// Report the stored DXGI format without changing its alpha interpretation.
    fn pixel_format(&self) -> D2D1_PIXEL_FORMAT {
        let pair = self.src.borrow().as_ref().map(|s| (s.format.0, s.alpha.0))
            .or_else(|| self.buf.borrow().as_ref().map(|b| (b.desc.format.0, b.desc.alpha_mode.0)))
            .unwrap_or((87, 1));
        D2D1_PIXEL_FORMAT { format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(pair.0), alphaMode: D2D1_ALPHA_MODE(pair.1) }
    }
    fn write_src(&self, dx: u32, dy: u32, w: u32, rows: &[u8], src_stride: usize) {
        let bpp = d2d::bitmap::bytes_per_pixel(d2d::abi::DXGI_FORMAT(self.pixel_format().format.0)).unwrap_or(4);
        let write = |data: &mut [u8], stride: usize| {
            let n = rows.len() / src_stride.max(1);
            for y in 0..n.min(self.pixels.height.saturating_sub(dy) as usize) {
                let count = w.min(self.pixels.width.saturating_sub(dx)) as usize * bpp;
                let to = (dy as usize + y) * stride + dx as usize * bpp;
                let from = y * src_stride;
                if let (Some(dst), Some(src)) = (data.get_mut(to..to + count), rows.get(from..from + count)) {
                    dst.copy_from_slice(src);
                }
            }
        };
        if let Some(b) = self.buf.borrow_mut().as_mut() {
            write(&mut b.data, b.stride);
        }
        if let Some(s) = self.src.borrow_mut().as_mut() {
            write(&mut s.premul_bgra, s.width as usize * bpp);
        }
    }

}
impl ID2D1Resource_Impl for BitmapObj_Impl {
    fn GetFactory(&self, f: windows_core::OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
impl ID2D1Image_Impl for BitmapObj_Impl {}
impl ID2D1Bitmap_Impl for BitmapObj_Impl {
    fn GetSize(&self) -> D2D_SIZE_F {
        D2D_SIZE_F { width: self.pixels.width as f32 * 96.0 / self.dpi.0, height: self.pixels.height as f32 * 96.0 / self.dpi.1 }
    }
    fn GetPixelSize(&self) -> D2D_SIZE_U {
        self.pixels
    }
    fn GetPixelFormat(&self) -> D2D1_PIXEL_FORMAT {
        self.pixel_format()
    }
    fn GetDpi(&self, dpix: *mut f32, dpiy: *mut f32) {
        unsafe {
            *dpix = self.dpi.0;
            *dpiy = self.dpi.1;
        }
    }
    fn CopyFromBitmap(&self, dst: *const D2D_POINT_2U, bitmap: Ref<ID2D1Bitmap>, src: *const D2D_RECT_U) -> WResult<()> {
        let obj = bitmap.as_ref().and_then(|b| b.cast_object_ref::<BitmapObj>().ok()).ok_or(windows_core::Error::from(E_NOTIMPL))?;
        let sb = obj.readable_src().ok_or(windows_core::Error::from(E_NOTIMPL))?;
        let (dx, dy) = dst_point(dst);
        if self.pixel_format().format.0 != sb.format.0 { return Err(E_INVALIDARG.into()); }
        let bpp = d2d::bitmap::bytes_per_pixel(sb.format).ok_or(windows_core::Error::from(E_INVALIDARG))?;
        let (rows, w, stride) = extract_rect(&sb.premul_bgra, sb.width, sb.height, src, bpp);
        self.write_src(dx, dy, w, &rows, stride);
        Ok(())
    }
    fn CopyFromRenderTarget(&self, d: *const D2D_POINT_2U, rt: Ref<ID2D1RenderTarget>, s: *const D2D_RECT_U) -> WResult<()> {
        let ctx = rt.as_ref().and_then(|r| r.cast_object_ref::<super::context::ContextObj>().ok()).ok_or(windows_core::Error::from(E_NOTIMPL))?;
        let (data, _stride, w, h) = ctx.snapshot().ok_or(windows_core::Error::from(E_NOTIMPL))?;
        let (dx, dy) = dst_point(d);
        let (rows, cw, stride) = extract_rect(&data, w, h, s, _stride / (w as usize).max(1));
        self.write_src(dx, dy, cw, &rows, stride);
        Ok(())
    }
    fn CopyFromMemory(&self, r: *const D2D_RECT_U, d: *const core::ffi::c_void, pitch: u32) -> WResult<()> {
        if d.is_null() {
            return Err(E_INVALIDARG.into());
        }
        let (dx, dy, w, h) = if r.is_null() {
            (0, 0, self.pixels.width, self.pixels.height)
        } else {
            let rc = unsafe { &*r };
            (rc.left, rc.top, rc.right.saturating_sub(rc.left), rc.bottom.saturating_sub(rc.top))
        };
        let bpp = d2d::bitmap::bytes_per_pixel(d2d::abi::DXGI_FORMAT(self.pixel_format().format.0)).ok_or(windows_core::Error::from(E_INVALIDARG))?;
        if dx > self.pixels.width || dy > self.pixels.height || w > self.pixels.width - dx || h > self.pixels.height - dy || (pitch as usize) < w as usize * bpp {
            return Err(E_INVALIDARG.into());
        }
        for y in 0..h {
            let row = unsafe { core::slice::from_raw_parts((d as *const u8).add(y as usize * pitch as usize), w as usize * bpp) };
            self.write_src(dx, dy + y, w, row, w as usize * bpp);
        }
        Ok(())
    }
}

/// A `D2D_POINT_2U*` destination offset (null ⇒ origin).
fn dst_point(p: *const D2D_POINT_2U) -> (u32, u32) {
    if p.is_null() { (0, 0) } else { let p = unsafe { &*p }; (p.x, p.y) }
}

/// Pull a `D2D_RECT_U` sub-rect (null ⇒ the whole image) out of a `w·h` premultiplied-BGRA buffer into
/// a tightly-packed row block; returns `(rows, width_px, stride_bytes)`.
fn extract_rect(px: &[u8], w: u32, h: u32, rect: *const D2D_RECT_U, bpp: usize) -> (Vec<u8>, u32, usize) {
    let (l, t, right, bottom) = if rect.is_null() {
        (0, 0, w, h)
    } else {
        let r = unsafe { &*rect };
        (r.left.min(w), r.top.min(h), r.right.min(w), r.bottom.min(h))
    };
    let (cw, ch) = (right.saturating_sub(l), bottom.saturating_sub(t));
    let stride = cw as usize * bpp;
    let mut out = vec![0u8; stride * ch as usize];
    for ry in 0..ch {
        let so = ((t + ry) as usize * w as usize + l as usize) * bpp;
        let to = ry as usize * stride;
        out[to..to + stride].copy_from_slice(&px[so..so + stride]);
    }
    (out, cw, stride)
}
// --- ink + ink style (ID2D1Ink / ID2D1InkStyle) -----------------------------------------------

/// A nib point `D2D1_INK_POINT { x, y, radius }` → the core `InkPoint` `[x, y, radius]`.
fn ink_pt(p: &D2D1_INK_POINT) -> [f32; 3] {
    [p.x, p.y, p.radius]
}

#[implement(ID2D1Ink)]
pub(crate) struct InkObj {
    pub(crate) ink: RefCell<Ink>,
    pub(crate) factory: Option<ID2D1Factory>,
}
impl ID2D1Resource_Impl for InkObj_Impl {
    fn GetFactory(&self, f: windows_core::OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
impl ID2D1Ink_Impl for InkObj_Impl {
    fn SetStartPoint(&self, p: *const D2D1_INK_POINT) {
        self.ink.borrow_mut().start = ink_pt(unsafe { &*p });
    }
    fn GetStartPoint(&self) -> D2D1_INK_POINT {
        let s = self.ink.borrow().start;
        D2D1_INK_POINT { x: s[0], y: s[1], radius: s[2] }
    }
    fn AddSegments(&self, segments: *const D2D1_INK_BEZIER_SEGMENT, count: u32) -> WResult<()> {
        let segs: Vec<InkBezierSegment> = (0..count as isize)
            .map(|i| {
                let s = unsafe { &*segments.offset(i) };
                InkBezierSegment { p1: ink_pt(&s.point1), p2: ink_pt(&s.point2), end: ink_pt(&s.point3) }
            })
            .collect();
        self.ink.borrow_mut().add_segments(&segs);
        Ok(())
    }
    fn RemoveSegmentsAtEnd(&self, count: u32) -> WResult<()> {
        self.ink.borrow_mut().remove_segments_at_end(count).map_err(|c| windows_core::Error::from(windows_core::HRESULT(c)))
    }
    fn SetSegments(&self, start: u32, seg: *const D2D1_INK_BEZIER_SEGMENT, count: u32) -> WResult<()> {
        let mut ink = self.ink.borrow_mut();
        if start as usize + count as usize > ink.segments.len() {
            return Err(E_INVALIDARG.into());
        }
        for i in 0..count as usize {
            let s = unsafe { &*seg.add(i) };
            ink.segments[start as usize + i] = InkBezierSegment { p1: ink_pt(&s.point1), p2: ink_pt(&s.point2), end: ink_pt(&s.point3) };
        }
        Ok(())
    }
    fn SetSegmentAtEnd(&self, seg: *const D2D1_INK_BEZIER_SEGMENT) -> WResult<()> {
        let mut ink = self.ink.borrow_mut();
        let last = ink.segments.len().checked_sub(1).ok_or(windows_core::Error::from(E_INVALIDARG))?;
        let s = unsafe { &*seg };
        ink.segments[last] = InkBezierSegment { p1: ink_pt(&s.point1), p2: ink_pt(&s.point2), end: ink_pt(&s.point3) };
        Ok(())
    }
    fn GetSegmentCount(&self) -> u32 {
        self.ink.borrow().segments.len() as u32
    }
    fn GetSegments(&self, start: u32, seg: *mut D2D1_INK_BEZIER_SEGMENT, count: u32) -> WResult<()> {
        let ink = self.ink.borrow();
        if start as usize + count as usize > ink.segments.len() {
            return Err(E_INVALIDARG.into());
        }
        for i in 0..count as usize {
            let s = &ink.segments[start as usize + i];
            unsafe { *seg.add(i) = D2D1_INK_BEZIER_SEGMENT { point1: d2d_ink_pt(s.p1), point2: d2d_ink_pt(s.p2), point3: d2d_ink_pt(s.end) } };
        }
        Ok(())
    }
    fn StreamAsGeometry(&self, s: Ref<ID2D1InkStyle>, t: *const windows_numerics::Matrix3x2, _tol: f32, sink: Ref<ID2D1SimplifiedGeometrySink>) -> WResult<()> {
        let sink = sink.as_ref().ok_or(windows_core::Error::from(E_NOTIMPL))?;
        let style = ink_style_of(s);
        let figures = self.ink.borrow().stream_as_geometry(&super::mat_opt(t), &style);
        // Emit the swept-nib outline as filled, closed, nonzero-wound figures (how `DrawInk` fills it).
        unsafe {
            sink.SetFillMode(D2D1_FILL_MODE_WINDING);
            for fig in &figures {
                if fig.is_empty() {
                    continue;
                }
                sink.BeginFigure(Vector2 { X: fig[0][0], Y: fig[0][1] }, D2D1_FIGURE_BEGIN_FILLED);
                let pts: Vec<Vector2> = fig[1..].iter().map(|p| Vector2 { X: p[0], Y: p[1] }).collect();
                sink.AddLines(&pts);
                sink.EndFigure(D2D1_FIGURE_END_CLOSED);
            }
        }
        Ok(())
    }
    fn GetBounds(&self, s: Ref<ID2D1InkStyle>, t: *const windows_numerics::Matrix3x2) -> WResult<D2D_RECT_F> {
        let style = ink_style_of(s);
        let b = self.ink.borrow().get_bounds(&super::mat_opt(t), &style);
        Ok(D2D_RECT_F { left: b[0], top: b[1], right: b[2], bottom: b[3] })
    }
}

/// A core ink-point `[x,y,radius]` → `D2D1_INK_POINT`.
fn d2d_ink_pt(p: [f32; 3]) -> D2D1_INK_POINT {
    D2D1_INK_POINT { x: p[0], y: p[1], radius: p[2] }
}

/// The `ID2D1InkStyle`'s core style, or the default (ROUND nib, identity) when none is given.
fn ink_style_of(s: Ref<ID2D1InkStyle>) -> InkStyle {
    s.as_ref().and_then(|s| s.cast_object_ref::<InkStyleObj>().ok()).map(|o| *o.style.borrow()).unwrap_or_default()
}

#[implement(ID2D1InkStyle)]
pub(crate) struct InkStyleObj {
    pub(crate) style: RefCell<InkStyle>,
    pub(crate) factory: Option<ID2D1Factory>,
}
impl ID2D1Resource_Impl for InkStyleObj_Impl {
    fn GetFactory(&self, f: windows_core::OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
impl ID2D1InkStyle_Impl for InkStyleObj_Impl {
    fn SetNibTransform(&self, t: *const windows_numerics::Matrix3x2) {
        if !t.is_null() {
            let m = unsafe { &*t };
            self.style.borrow_mut().nib_transform = [m.M11, m.M12, m.M21, m.M22, m.M31, m.M32];
        }
    }
    fn GetNibTransform(&self, t: *mut windows_numerics::Matrix3x2) {
        let m = self.style.borrow().nib_transform;
        unsafe { *t = windows_numerics::Matrix3x2 { M11: m[0], M12: m[1], M21: m[2], M22: m[3], M31: m[4], M32: m[5] } };
    }
    fn SetNibShape(&self, s: D2D1_INK_NIB_SHAPE) {
        self.style.borrow_mut().nib_shape = s.0 as u32;
    }
    fn GetNibShape(&self) -> D2D1_INK_NIB_SHAPE {
        D2D1_INK_NIB_SHAPE(self.style.borrow().nib_shape as i32)
    }
}

// --- geometry realization (ID2D1GeometryRealization: baked geometry, drawn by DrawGeometryRealization) --

/// A filled or stroked geometry realization. The core has no bake step; a realization draws the same
/// pixels as filling / stroking the geometry, so we keep the geometry (+ stroke width/style) and
/// replay it at `DrawGeometryRealization`.
#[implement(ID2D1GeometryRealization)]
pub(crate) struct RealizationObj {
    pub(crate) geom: Geometry,
    pub(crate) stroke: Option<(f32, StrokeStyle)>,
    /// The flattening tolerance the realization was created with (its widen and tessellation use it).
    pub(crate) tolerance: f32,
    pub(crate) factory: Option<ID2D1Factory>,
}
impl ID2D1Resource_Impl for RealizationObj_Impl {
    fn GetFactory(&self, f: windows_core::OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
impl ID2D1GeometryRealization_Impl for RealizationObj_Impl {}

// --- bitmap ------------------------------------------------------------------------------------

impl ID2D1Bitmap1_Impl for BitmapObj_Impl {
    fn GetColorContext(&self, c: windows_core::OutRef<ID2D1ColorContext>) {
        // The color context the bitmap was created with (NULL if none), as native returns.
        let _ = c.write(self.color_context.clone());
    }
    fn GetOptions(&self) -> D2D1_BITMAP_OPTIONS {
        self.options
    }
    fn GetSurface(&self) -> WResult<windows::Win32::Graphics::Dxgi::IDXGISurface> {
        // The surface this bitmap was created from (CreateBitmapFromDxgiSurface) lives in its present
        // destination; its D3D texture QIs back to that same surface. A non-surface-backed bitmap has
        // none — native's D2DBitmap::GetSurface returns D2DERR_INVALID_CALL (0x8899000A) for that.
        // Off Windows there are no surface-backed bitmaps (no DXGI), so always the latter.
        #[cfg(windows)]
        {
            use windows_core::Interface;
            match &*self.present.borrow() {
                Some(Present::Dxgi { tex, .. }) => return tex.cast(),
                _ => {}
            }
        }
        Err(windows_core::HRESULT(0x8899_000Au32 as i32).into())
    }
    fn Map(&self, _o: D2D1_MAP_OPTIONS) -> WResult<D2D1_MAPPED_RECT> {
        // A CPU-readable bitmap exposes its pixels; prefer the source surface (an uploaded/`CopyFrom*`
        // image, stride = width·4), else the bound target staging buffer. The pointer stays valid while
        // the `BitmapObj` lives (neither Vec is reallocated between Map/Unmap).
        if let Some(s) = self.src.borrow().as_ref() {
            return Ok(D2D1_MAPPED_RECT { pitch: s.width * d2d::bitmap::bytes_per_pixel(s.format).unwrap_or(4) as u32, bits: s.premul_bgra.as_ptr() as *mut u8 });
        }
        let b = self.buf.borrow();
        let buf = b.as_ref().ok_or_else(|| windows_core::Error::from(E_NOTIMPL))?;
        Ok(D2D1_MAPPED_RECT { pitch: buf.stride as u32, bits: buf.data.as_ptr() as *mut u8 })
    }
    fn Unmap(&self) -> WResult<()> {
        Ok(())
    }
}
