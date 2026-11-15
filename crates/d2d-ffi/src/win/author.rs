//! Application-authored custom effects (`ID2D1Factory1::RegisterEffectFromString`/`Stream`): the app
//! supplies a `PD2D1_EFFECT_FACTORY` that builds its own `ID2D1EffectImpl` + transform nodes, and D2D
//! (us) hands `Initialize` an `ID2D1EffectContext` and `ID2D1TransformGraph` to build the graph. These
//! bridge the COM callbacks onto the `d2d` core's effect graph (effects/custom.rs): the context
//! factories return COM transform nodes carrying a core `Transform`, the graph methods wire them into
//! the core `TransformGraph`, and a custom pixel-shader draw transform (the app's own
//! `ID2D1DrawTransform`) runs through our DXBC executor.
//!
//! Wired: the full authoring surface over the core's CPU graph — `CreateEffect`, the
//! offset/border/blend/bounds-adjustment/from-effect transform-node factories, resource textures
//! (`Create/FindResourceTexture` + `ID2D1DrawInfo::SetResourceTexture`), `CreateColorContext`, feature
//! queries, and the graph builders (`AddNode`/`RemoveNode`/`ConnectNode`/`ConnectToEffectInput`/
//! `SetOutputNode`, plus `SetSingleTransformNode`/`SetPassthroughGraph`), the app's own pixel
//! (`ID2D1DrawTransform`), compute (`ID2D1ComputeTransform` + `ID2D1ComputeInfo` + `LoadComputeShader`,
//! dispatched through the DXBC executor into a UAV image) and vertex (`SetVertexProcessing` +
//! `LoadVertexShader` + `Create/FindVertexBuffer`, the VS deforming a mesh the PS then shades) draw
//! transforms, `CreateColorContext`/`FromFilename`/`FromWicColorContext` (including the Exif Adobe-RGB
//! profile native synthesises), and mutable resources (`ID2D1VertexBuffer::Map`/`Unmap`,
//! `ID2D1ResourceTexture::Update`). The whole effect-authoring surface is wired — no live method
//! returns `E_NOTIMPL`.

use super::image::EffectObj;
use core::ffi::c_void;
use d2d::effects::custom::{VbElement, VertexBuffer};
use d2d::effects::{Effect, EffectContext, EffectImpl, NodeId, Props, ResourceTexture, Transform, TransformGraph};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use windows::Win32::Foundation::{POINT, RECT};
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL, D3D_FEATURE_LEVEL_11_0};
use windows::Win32::Graphics::Direct2D::*;
use windows_core::{implement, ComObjectInner, Error, GUID, HRESULT, IUnknown, Interface, OutRef, Ref, Result as WResult};

const E_FAIL: HRESULT = HRESULT(0x8000_4005u32 as i32);
const E_INVALIDARG: HRESULT = HRESULT(0x8007_0057u32 as i32);
/// `GetMaximumSupportedFeatureLevel` with no requested level ≤ the device's (native returns this raw).
const NO_FEATURE_LEVEL: HRESULT = HRESULT(0x8899_0026u32 as i32);
/// Our stand-in WARP device's maximum feature level (the conformance oracle runs WARP at 11_0).
const DEVICE_MAX_LEVEL: D3D_FEATURE_LEVEL = D3D_FEATURE_LEVEL_11_0;

/// `ID2D1Factory1::RegisterEffectFromString`: register a custom effect keyed by `clsid` whose author
/// object D2D builds by calling `factory`. The `d2d` core parses the XML schema; [`BridgeEffectImpl`]
/// drives the app's COM `Initialize` each time the effect is created.
pub(crate) fn register_from_string(core: &d2d::Factory, clsid: GUID, xml: &str, factory: PD2D1_EFFECT_FACTORY, bindings: *const D2D1_PROPERTY_BINDING, binding_count: u32) -> WResult<()> {
    let bindings = Rc::new(capture_bindings(bindings, binding_count));
    core.register_effect_from_string(clsid, xml, Rc::new(move || Box::new(BridgeEffectImpl::new(factory, bindings.clone()))))
        .map_err(Error::from)
}

/// `ID2D1Factory1::RegisterEffectFromStream`: same, from the UTF-8 XML the stream holds.
pub(crate) fn register_from_stream(core: &d2d::Factory, clsid: GUID, xml: &[u8], factory: PD2D1_EFFECT_FACTORY, bindings: *const D2D1_PROPERTY_BINDING, binding_count: u32) -> WResult<()> {
    let bindings = Rc::new(capture_bindings(bindings, binding_count));
    core.register_effect_from_stream(clsid, xml, Rc::new(move || Box::new(BridgeEffectImpl::new(factory, bindings.clone()))))
        .map_err(Error::from)
}

/// One captured `D2D1_PROPERTY_BINDING`: the property name and its set-function. ComputeSharp.D2D1
/// binds its `Buffer` property to a setter that stores the constant-buffer bytes in the app effect,
/// which the effect's `PrepareForRender` then pushes to the pixel shader. [G2]
struct Binding {
    name: String,
    set: PD2D1_PROPERTY_SET_FUNCTION,
}

/// Capture the `D2D1_PROPERTY_BINDING` array `RegisterEffectFromString` is given (name + set-function),
/// so `SetValue` on a bound property can reach the app's own setter. [G2]
fn capture_bindings(b: *const D2D1_PROPERTY_BINDING, count: u32) -> Vec<Binding> {
    if b.is_null() {
        return Vec::new();
    }
    (0..count as usize)
        .map(|i| {
            let bind = unsafe { &*b.add(i) };
            let name = unsafe { bind.propertyName.to_string() }.unwrap_or_default();
            Binding { name, set: bind.setFunction }
        })
        .collect()
}

/// Core `EffectImpl` that defers to the application's COM `ID2D1EffectImpl`.
struct BridgeEffectImpl {
    factory: PD2D1_EFFECT_FACTORY,
    bindings: Rc<Vec<Binding>>,
    /// The app's effect object, built once (native builds it at CreateEffect) and reused on rebuild,
    /// so a value set through a binding — or a constant buffer pushed in `PrepareForRender` — survives
    /// a graph rebuild from a property change. [G2]
    app: RefCell<Option<ID2D1EffectImpl>>,
    /// The app's pixel-shader draw transforms and their core nodes' input-reach cells, re-asked before
    /// each draw (see [`input_reach`]).
    probes: Rc<Probes>,
}

/// An app draw transform, its core node's input-reach cell, and its input count.
type Probes = RefCell<Vec<(ID2D1Transform, Rc<RefCell<Vec<[i32; 4]>>>, u32)>>;

impl BridgeEffectImpl {
    fn new(factory: PD2D1_EFFECT_FACTORY, bindings: Rc<Vec<Binding>>) -> Self {
        BridgeEffectImpl { factory, bindings, app: RefCell::new(None), probes: Rc::default() }
    }
    /// The retained app effect, building it (via the registered factory) on first use.
    fn app(&self) -> Result<ID2D1EffectImpl, HRESULT> {
        if let Some(app) = self.app.borrow().clone() {
            return Ok(app);
        }
        let f = self.factory.ok_or(E_FAIL)?;
        let mut unk: Option<IUnknown> = None;
        unsafe { f(OutRef::from(&mut unk)) }.ok().map_err(|e| e.code())?;
        let app: ID2D1EffectImpl = unk.ok_or(E_FAIL)?.cast().map_err(|e| e.code())?;
        *self.app.borrow_mut() = Some(app.clone());
        Ok(app)
    }
}

impl EffectImpl for BridgeEffectImpl {
    fn initialize(&self, _props: &Props, ctx: &EffectContext, graph: &mut TransformGraph) -> Result<(), HRESULT> {
        let app = self.app()?;
        // Wrap the core context/graph in COM objects for the callbacks. The context is a clone (valid for
        // as long as the app keeps it); the graph's raw pointers are valid only for this synchronous call,
        // the `&mut graph` dormant while the reentrant callbacks mutate it through the pointer.
        let ectx: ID2D1EffectContext = EffectContextObj { ctx: ctx.clone() }.into();
        self.probes.borrow_mut().clear();
        let graph_com: ID2D1TransformGraph = TransformGraphObj { ctx: ctx as *const _, graph: graph as *mut _, nodes: RefCell::new(Vec::new()), probes: self.probes.clone() }.into();
        unsafe { app.Initialize(&ectx, &graph_com) }.map_err(|e| e.code())
    }

    fn prepare_for_render(&self) {
        // Native calls PrepareForRender before every draw: the app re-pushes its pixel-shader constant
        // buffer through SetPixelShaderConstantBuffer onto the DrawInfo it holds, which shares its b0
        // cell with the core ShaderNode — so the shader sees the current constants this draw. [G2]
        if let Some(app) = self.app.borrow().clone() {
            let _ = unsafe { app.PrepareForRender(D2D1_CHANGE_TYPE_NONE) };
        }
        // Then its draw transforms' input rects, which may depend on the constants just pushed.
        for (t, reach, n) in self.probes.borrow().iter() {
            *reach.borrow_mut() = input_reach(t, *n);
        }
    }

    fn set_bound_value(&self, name: &str, data: &[u8]) {
        let Some(binding) = self.bindings.iter().find(|b| b.name == name) else { return };
        let Some(setfn) = binding.set else { return };
        let Some(app) = self.app.borrow().clone() else { return };
        // The set-function takes the effect as a borrowed `Ref<IUnknown>` (repr(transparent) over the
        // COM pointer), so invoke it through a raw-pointer signature — ABI-identical, no `Ref` ctor. [G2]
        let setfn: unsafe extern "system" fn(*mut c_void, *const u8, u32) -> HRESULT = unsafe { core::mem::transmute(setfn) };
        let _ = unsafe { setfn(app.as_raw(), data.as_ptr(), data.len() as u32) };
    }
}

/// Build a core `Transform` from the app's own shader node — an `ID2D1ComputeTransform` (compute pass)
/// or else an `ID2D1DrawTransform` (pixel pass).
fn app_shader_transform(ctx: &EffectContext, node: &ID2D1TransformNode, probes: &Probes) -> WResult<Transform> {
    if let Ok(compute) = node.cast::<ID2D1ComputeTransform>() {
        return compute_transform(ctx, &compute, node);
    }
    draw_transform(ctx, node, probes)
}

/// How far each input rect the transform requests reaches past its output rect, `[l, t, r, b]` px
/// (`MapOutputRectToInputRects` of a unit output rect, saturating): 0 for the 1:1 identity, `i32::MAX`
/// for an infinite request (ComputeSharp.D2D1 asks that for a complex input). Native maps forward
/// before it maps back, so `MapInputRectsToOutputRect` runs first: a ComputeSharp transform mapper
/// takes its constants there. Empty (identity) if the transform refuses.
fn input_reach(t: &ID2D1Transform, n: u32) -> Vec<[i32; 4]> {
    let inf = vec![RECT { left: i32::MIN, top: i32::MIN, right: i32::MAX, bottom: i32::MAX }; n as usize];
    let (mut out, mut opaque) = (RECT::default(), RECT::default());
    if n > 0 && unsafe { t.MapInputRectsToOutputRect(inf.as_ptr(), inf.as_ptr(), n, &mut out, &mut opaque) }.is_err() {
        return Vec::new();
    }
    let unit = RECT { left: 0, top: 0, right: 1, bottom: 1 };
    let mut req = vec![RECT::default(); n as usize];
    if unsafe { t.MapOutputRectToInputRects(&unit, &mut req) }.is_err() {
        return Vec::new();
    }
    let past = |v: i64| v.clamp(0, i32::MAX as i64) as i32;
    req.iter().map(|r| [past(-(r.left as i64)), past(-(r.top as i64)), past(r.right as i64 - 1), past(r.bottom as i64 - 1)]).collect()
}

/// `ID2D1DrawTransform`: hand it an `ID2D1DrawInfo`, collect its pixel shader + `b0` + bound resource
/// textures, then realise the pixel pass.
fn draw_transform(ctx: &EffectContext, node: &ID2D1TransformNode, probes: &Probes) -> WResult<Transform> {
    let input_count = unsafe { node.GetInputCount() };
    let draw: ID2D1DrawTransform = node.cast()?;
    let info = DrawInfoObj::default().into_object();
    let iface: ID2D1DrawInfo = info.to_interface();
    unsafe { draw.SetDrawInfo(&iface)? };
    let ps_id = info.shader_id.get().ok_or(Error::from(E_FAIL))?;
    let textures = info.textures.borrow().clone();
    // A vertex shader was bound (SetVertexProcessing): a mesh draw. Else the 1:1 pixel pass.
    if let Some(vs_id) = info.vs_id.get() {
        let vb = info.vb.borrow().clone().ok_or(Error::from(E_FAIL))?;
        let vs_cbuffer = info.vs_cbuffer.borrow().clone();
        let ps_cbuffer = info.cbuffer.borrow().clone();
        // Viewport = the transform's output rect (it maps the VS clip space onto it).
        let inf = [RECT { left: i32::MIN, top: i32::MIN, right: i32::MAX, bottom: i32::MAX }; 8];
        let (mut out, mut oo) = (RECT::default(), RECT::default());
        unsafe { node.cast::<ID2D1Transform>()?.MapInputRectsToOutputRect(inf.as_ptr(), inf.as_ptr(), input_count, &mut out, &mut oo)? };
        return ctx
            .vertex_shader_transform(vs_id, ps_id, vs_cbuffer, ps_cbuffer, vb, input_count, textures, [out.left, out.top, out.right, out.bottom])
            .map_err(Error::from);
    }
    // Bind the SAME b0 cell to the core node so a later PrepareForRender updates this draw's constants;
    // likewise the resource textures (bound late by ComputeSharp) and the input-reach cell, re-asked of
    // the transform before each draw.
    let reach: Rc<RefCell<Vec<[i32; 4]>>> = Rc::default();
    if let Ok(transform) = node.cast::<ID2D1Transform>() {
        *reach.borrow_mut() = input_reach(&transform, input_count);
        probes.borrow_mut().push((transform, reach.clone(), input_count));
    }
    ctx.pixel_shader_transform(ps_id, info.cbuffer.clone(), input_count, info.textures.clone(), reach, info.filter.get()).map_err(Error::from)
}

/// `ID2D1ComputeTransform`: hand it an `ID2D1ComputeInfo`, collect the compute shader + `b0` + resource
/// textures, ask it to size the dispatch (`CalculateThreadgroups`), then realise the compute pass.
fn compute_transform(ctx: &EffectContext, compute: &ID2D1ComputeTransform, node: &ID2D1TransformNode) -> WResult<Transform> {
    let input_count = unsafe { node.GetInputCount() };
    let info = ComputeInfoObj::default().into_object();
    let iface: ID2D1ComputeInfo = info.to_interface();
    unsafe { compute.SetComputeInfo(&iface)? };
    let id = info.shader_id.get().ok_or(Error::from(E_FAIL))?;
    let cbuffer = info.cbuffer.borrow().clone();
    let textures = info.textures.borrow().clone();
    // Thread-group counts captured once (fixed-size / generator compute; see `ComputeNode`). The rect
    // is a placeholder — a per-draw dispatch sized to the live output rect is the push-model follow-on.
    let (mut gx, mut gy, mut gz) = (0u32, 0u32, 0u32);
    let rect = RECT { left: 0, top: 0, right: 0, bottom: 0 };
    unsafe { compute.CalculateThreadgroups(&rect, &mut gx, &mut gy, &mut gz)? };
    ctx.compute_shader_transform(id, cbuffer, input_count, textures, [gx, gy, gz], [0, 0]).map_err(Error::from)
}

// --- ID2D1EffectContext ------------------------------------------------------------------------

#[implement(ID2D1EffectContext)]
struct EffectContextObj {
    /// Owned (its stores are shared `Rc`s): an app may keep the interface past `Initialize` and call it
    /// later — ComputeSharp.D2D1 creates a resource texture through it when one is set.
    ctx: EffectContext,
}
impl EffectContextObj_Impl {
    fn ctx(&self) -> &EffectContext {
        &self.ctx
    }
}
#[allow(non_snake_case)]
impl ID2D1EffectContext_Impl for EffectContextObj_Impl {
    fn GetDpi(&self, dpix: *mut f32, dpiy: *mut f32) {
        let (x, y) = self.ctx().get_dpi();
        unsafe {
            *dpix = x;
            *dpiy = y;
        }
    }
    fn LoadPixelShader(&self, shaderid: *const GUID, buffer: *const u8, count: u32) -> WResult<()> {
        let dxbc = unsafe { core::slice::from_raw_parts(buffer, count as usize) };
        self.ctx().load_pixel_shader(unsafe { *shaderid }, dxbc).map_err(Error::from)
    }
    fn LoadVertexShader(&self, shaderid: *const GUID, buffer: *const u8, count: u32) -> WResult<()> {
        let dxbc = unsafe { core::slice::from_raw_parts(buffer, count as usize) };
        self.ctx().load_vertex_shader(unsafe { *shaderid }, dxbc).map_err(Error::from)
    }
    fn LoadComputeShader(&self, shaderid: *const GUID, buffer: *const u8, count: u32) -> WResult<()> {
        let dxbc = unsafe { core::slice::from_raw_parts(buffer, count as usize) };
        self.ctx().load_compute_shader(unsafe { *shaderid }, dxbc).map_err(Error::from)
    }
    fn IsShaderLoaded(&self, shaderid: *const GUID) -> windows_core::BOOL {
        self.ctx().is_shader_loaded(unsafe { *shaderid }).into()
    }
    fn IsBufferPrecisionSupported(&self, _p: D2D1_BUFFER_PRECISION) -> windows_core::BOOL {
        true.into()
    }
    fn CreateEffect(&self, effectid: *const GUID) -> WResult<ID2D1Effect> {
        let clsid = unsafe { *effectid };
        let effect = self.ctx().create_effect(clsid).map_err(Error::from)?;
        Ok(EffectObj::new(effect, clsid).into())
    }
    /// The highest requested feature level our device (WARP 11_0) supports; none ≤ it → native's raw
    /// `0x88990026`.
    fn GetMaximumSupportedFeatureLevel(&self, levels: *const D3D_FEATURE_LEVEL, count: u32) -> WResult<D3D_FEATURE_LEVEL> {
        if count == 0 || levels.is_null() {
            return Err(NO_FEATURE_LEVEL.into());
        }
        let req = unsafe { core::slice::from_raw_parts(levels, count as usize) };
        req.iter()
            .copied()
            .filter(|l| l.0 <= DEVICE_MAX_LEVEL.0)
            .max_by_key(|l| l.0)
            .ok_or(NO_FEATURE_LEVEL.into())
    }
    fn CreateTransformNodeFromEffect(&self, effect: Ref<ID2D1Effect>) -> WResult<ID2D1TransformNode> {
        let effect = effect.as_ref().ok_or(Error::from(E_INVALIDARG))?.cast_object_ref::<EffectObj>().map_err(|_| Error::from(E_INVALIDARG))?.effect.clone();
        Ok(EffectNodeObj { ctx: &self.ctx as *const _, effect }.into())
    }
    fn CreateBlendTransform(&self, numinputs: u32, desc: *const D2D1_BLEND_DESCRIPTION) -> WResult<ID2D1BlendTransform> {
        if numinputs == 0 || desc.is_null() {
            return Err(E_INVALIDARG.into());
        }
        Ok(BlendNodeObj { ctx: &self.ctx as *const _, inputs: numinputs, desc: RefCell::new(unsafe { *desc }) }.into())
    }
    fn CreateBorderTransform(&self, x: D2D1_EXTEND_MODE, y: D2D1_EXTEND_MODE) -> WResult<ID2D1BorderTransform> {
        if !(0..=2).contains(&x.0) || !(0..=2).contains(&y.0) {
            return Err(E_INVALIDARG.into());
        }
        Ok(BorderNodeObj { ctx: &self.ctx as *const _, x: Cell::new(x), y: Cell::new(y) }.into())
    }
    fn CreateOffsetTransform(&self, offset: &POINT) -> WResult<ID2D1OffsetTransform> {
        Ok(OffsetNodeObj { ctx: &self.ctx as *const _, offset: Cell::new(*offset) }.into())
    }
    fn CreateBoundsAdjustmentTransform(&self, bounds: *const RECT) -> WResult<ID2D1BoundsAdjustmentTransform> {
        if bounds.is_null() {
            return Err(E_INVALIDARG.into());
        }
        Ok(BoundsNodeObj { ctx: &self.ctx as *const _, rect: Cell::new(unsafe { *bounds }) }.into())
    }
    fn CreateResourceTexture(&self, resourceid: *const GUID, props: *const D2D1_RESOURCE_TEXTURE_PROPERTIES, data: *const u8, _strides: *const u32, datasize: u32) -> WResult<ID2D1ResourceTexture> {
        let p = unsafe { props.as_ref() }.ok_or(Error::from(E_INVALIDARG))?;
        if p.extents.is_null() {
            return Err(E_INVALIDARG.into());
        }
        let id = if resourceid.is_null() { None } else { Some(unsafe { *resourceid }) };
        // Read `dimensions` extents/extend-modes (capped so a garbage count can't over-read; the core
        // rejects `dimensions ∉ {1,2,3}`). ponytail: assumes tightly-packed data — row strides ignored.
        let n = p.dimensions.min(4) as usize;
        let extents: Vec<u32> = (0..n).map(|i| unsafe { *p.extents.add(i) }).collect();
        let extend: Vec<u32> = if p.extendModes.is_null() { Vec::new() } else { (0..n).map(|i| unsafe { *p.extendModes.add(i) }.0 as u32).collect() };
        let bytes = if data.is_null() { &[][..] } else { unsafe { core::slice::from_raw_parts(data, datasize as usize) } };
        let tex = self
            .ctx()
            .create_resource_texture_keyed(id, p.dimensions, &extents, p.bufferPrecision.0 as u32, p.channelDepth.0 as u32, p.filter.0 as u32, &extend, bytes)
            .map_err(Error::from)?;
        Ok(ResourceTextureObj { tex }.into())
    }
    fn FindResourceTexture(&self, resourceid: *const GUID) -> WResult<ID2D1ResourceTexture> {
        let tex = self.ctx().find_resource_texture(unsafe { *resourceid }).map_err(Error::from)?;
        Ok(ResourceTextureObj { tex }.into())
    }
    fn CreateVertexBuffer(&self, properties: *const D2D1_VERTEX_BUFFER_PROPERTIES, resourceid: *const GUID, custom: *const D2D1_CUSTOM_VERTEX_BUFFER_PROPERTIES) -> WResult<ID2D1VertexBuffer> {
        let p = unsafe { properties.as_ref() }.ok_or(Error::from(E_INVALIDARG))?;
        let cp = unsafe { custom.as_ref() }.ok_or(Error::from(E_INVALIDARG))?;
        if cp.inputElements.is_null() {
            return Err(E_INVALIDARG.into());
        }
        let id = if resourceid.is_null() { None } else { Some(unsafe { *resourceid }) };
        let data = if p.data.is_null() { &[][..] } else { unsafe { core::slice::from_raw_parts(p.data, p.byteWidth as usize) } };
        let els = unsafe { core::slice::from_raw_parts(cp.inputElements, cp.elementCount as usize) };
        let elements: Vec<VbElement> = els
            .iter()
            .map(|e| VbElement { semantic: unsafe { e.semanticName.to_string() }.unwrap_or_default(), index: e.semanticIndex, offset: e.alignedByteOffset as usize, components: format_components(e.format) })
            .collect();
        let vb = self.ctx().create_vertex_buffer(id, data, cp.stride, elements).map_err(Error::from)?;
        Ok(VertexBufferObj { vb }.into())
    }
    fn FindVertexBuffer(&self, resourceid: *const GUID) -> WResult<ID2D1VertexBuffer> {
        let vb = self.ctx().find_vertex_buffer(unsafe { *resourceid }).map_err(Error::from)?;
        Ok(VertexBufferObj { vb }.into())
    }
    fn CreateColorContext(&self, space: D2D1_COLOR_SPACE, profile: *const u8, size: u32) -> WResult<ID2D1ColorContext> {
        // CUSTOM (space 0) carries raw ICC bytes; SRGB/SCRGB are named — same as the device context.
        let ctx = if space.0 == 0 {
            let bytes = if profile.is_null() { &[][..] } else { unsafe { core::slice::from_raw_parts(profile, size as usize) } };
            d2d::color_context::ColorContext::from_profile_bytes(bytes)
        } else {
            d2d::color_context::ColorContext::from_color_space(d2d::abi::D2D1_COLOR_SPACE(space.0))
        };
        let cc: ID2D1ColorContext1 = super::image::ColorContextObj { ctx, factory: None }.into();
        cc.cast()
    }
    fn CreateColorContextFromFilename(&self, filename: &windows_core::PCWSTR) -> WResult<ID2D1ColorContext> {
        // Read the ICC file and parse it through the core (as the device context does): read → the
        // CUSTOM `from_profile_bytes` parse; a read failure maps to its HRESULT_FROM_WIN32.
        let path = unsafe { filename.to_string() }.map_err(|_| Error::from(E_INVALIDARG))?;
        let bytes = std::fs::read(&path).map_err(|e| Error::from(HRESULT(0x8007_0000u32 as i32 | (e.raw_os_error().unwrap_or(2) & 0xffff))))?;
        let cc: ID2D1ColorContext1 = super::image::ColorContextObj { ctx: d2d::color_context::ColorContext::from_profile_bytes(&bytes), factory: None }.into();
        cc.cast()
    }
    fn CreateColorContextFromWicColorContext(&self, c: Ref<windows::Win32::Graphics::Imaging::IWICColorContext>) -> WResult<ID2D1ColorContext> {
        let ctx = super::image::color_context_from_wic(c.as_ref().ok_or(Error::from(E_INVALIDARG))?)?;
        let cc: ID2D1ColorContext1 = super::image::ColorContextObj { ctx, factory: None }.into();
        cc.cast()
    }
    /// `CheckFeatureSupport`: WARP supports doubles and compute-shader 4x; both feature-data structs
    /// are one `BOOL`. A bad feature or size → `E_INVALIDARG` (native `CDImageEffectContext`).
    fn CheckFeatureSupport(&self, feature: D2D1_FEATURE, data: *mut c_void, size: u32) -> WResult<()> {
        if size != 4 || data.is_null() || !(feature == D2D1_FEATURE_DOUBLES || feature == D2D1_FEATURE_D3D10_X_HARDWARE_OPTIONS) {
            return Err(E_INVALIDARG.into());
        }
        unsafe { *(data as *mut i32) = 1 };
        Ok(())
    }
}

// --- transform-node COM objects (the EffectContext factories hand these to the graph) ----------
//
// Each holds the core-side parameters + the context pointer (valid for the synchronous Initialize)
// and builds its core `Transform` in `transform()`, which `AddNode` calls once.

#[implement(ID2D1OffsetTransform)]
struct OffsetNodeObj {
    ctx: *const EffectContext,
    offset: Cell<POINT>,
}
impl OffsetNodeObj_Impl {
    fn transform(&self) -> WResult<Transform> {
        let p = self.offset.get();
        Ok(unsafe { &*self.ctx }.create_offset_transform([p.x, p.y]))
    }
}
#[allow(non_snake_case)]
impl ID2D1TransformNode_Impl for OffsetNodeObj_Impl {
    fn GetInputCount(&self) -> u32 {
        1
    }
}
#[allow(non_snake_case)]
impl ID2D1OffsetTransform_Impl for OffsetNodeObj_Impl {
    fn SetOffset(&self, offset: &POINT) {
        self.offset.set(*offset);
    }
    fn GetOffset(&self) -> POINT {
        self.offset.get()
    }
}

#[implement(ID2D1BorderTransform)]
struct BorderNodeObj {
    ctx: *const EffectContext,
    x: Cell<D2D1_EXTEND_MODE>,
    y: Cell<D2D1_EXTEND_MODE>,
}
impl BorderNodeObj_Impl {
    fn transform(&self) -> WResult<Transform> {
        Ok(unsafe { &*self.ctx }.create_border_transform(self.x.get().0 as u32, self.y.get().0 as u32))
    }
}
#[allow(non_snake_case)]
impl ID2D1TransformNode_Impl for BorderNodeObj_Impl {
    fn GetInputCount(&self) -> u32 {
        1
    }
}
#[allow(non_snake_case)]
impl ID2D1ConcreteTransform_Impl for BorderNodeObj_Impl {
    fn SetOutputBuffer(&self, _p: D2D1_BUFFER_PRECISION, _c: D2D1_CHANNEL_DEPTH) -> WResult<()> {
        Ok(())
    }
    fn SetCached(&self, _c: windows_core::BOOL) {}
}
#[allow(non_snake_case)]
impl ID2D1BorderTransform_Impl for BorderNodeObj_Impl {
    fn SetExtendModeX(&self, m: D2D1_EXTEND_MODE) {
        self.x.set(m);
    }
    fn SetExtendModeY(&self, m: D2D1_EXTEND_MODE) {
        self.y.set(m);
    }
    fn GetExtendModeX(&self) -> D2D1_EXTEND_MODE {
        self.x.get()
    }
    fn GetExtendModeY(&self) -> D2D1_EXTEND_MODE {
        self.y.get()
    }
}

#[implement(ID2D1BoundsAdjustmentTransform)]
struct BoundsNodeObj {
    ctx: *const EffectContext,
    rect: Cell<RECT>,
}
impl BoundsNodeObj_Impl {
    fn transform(&self) -> WResult<Transform> {
        let r = self.rect.get();
        Ok(unsafe { &*self.ctx }.create_bounds_adjustment_transform([r.left, r.top, r.right, r.bottom]))
    }
}
#[allow(non_snake_case)]
impl ID2D1TransformNode_Impl for BoundsNodeObj_Impl {
    fn GetInputCount(&self) -> u32 {
        1
    }
}
#[allow(non_snake_case)]
impl ID2D1BoundsAdjustmentTransform_Impl for BoundsNodeObj_Impl {
    fn SetOutputBounds(&self, bounds: *const RECT) {
        self.rect.set(unsafe { *bounds });
    }
    fn GetOutputBounds(&self, bounds: *mut RECT) {
        unsafe { *bounds = self.rect.get() };
    }
}

#[implement(ID2D1BlendTransform)]
struct BlendNodeObj {
    ctx: *const EffectContext,
    inputs: u32,
    desc: RefCell<D2D1_BLEND_DESCRIPTION>,
}
impl BlendNodeObj_Impl {
    fn transform(&self) -> WResult<Transform> {
        let d = self.desc.borrow();
        unsafe { &*self.ctx }
            .create_blend_desc_transform(
                self.inputs,
                d.sourceBlend.0 as u32,
                d.destinationBlend.0 as u32,
                d.blendOperation.0 as u32,
                d.sourceBlendAlpha.0 as u32,
                d.destinationBlendAlpha.0 as u32,
                d.blendOperationAlpha.0 as u32,
                d.blendFactor,
            )
            .map_err(Error::from)
    }
}
#[allow(non_snake_case)]
impl ID2D1TransformNode_Impl for BlendNodeObj_Impl {
    fn GetInputCount(&self) -> u32 {
        self.inputs
    }
}
#[allow(non_snake_case)]
impl ID2D1ConcreteTransform_Impl for BlendNodeObj_Impl {
    fn SetOutputBuffer(&self, _p: D2D1_BUFFER_PRECISION, _c: D2D1_CHANNEL_DEPTH) -> WResult<()> {
        Ok(())
    }
    fn SetCached(&self, _c: windows_core::BOOL) {}
}
#[allow(non_snake_case)]
impl ID2D1BlendTransform_Impl for BlendNodeObj_Impl {
    fn SetDescription(&self, description: *const D2D1_BLEND_DESCRIPTION) {
        *self.desc.borrow_mut() = unsafe { *description };
    }
    fn GetDescription(&self, description: *mut D2D1_BLEND_DESCRIPTION) {
        unsafe { *description = *self.desc.borrow() };
    }
}

/// `CreateTransformNodeFromEffect`: a pre-wired sub-effect as a 0-input graph leaf.
#[implement(ID2D1TransformNode)]
struct EffectNodeObj {
    ctx: *const EffectContext,
    effect: Effect,
}
impl EffectNodeObj_Impl {
    fn transform(&self) -> WResult<Transform> {
        Ok(unsafe { &*self.ctx }.create_transform_node_from_effect(self.effect.clone()))
    }
}
#[allow(non_snake_case)]
impl ID2D1TransformNode_Impl for EffectNodeObj_Impl {
    fn GetInputCount(&self) -> u32 {
        0
    }
}

// --- ID2D1ResourceTexture ----------------------------------------------------------------------

#[implement(ID2D1ResourceTexture)]
struct ResourceTextureObj {
    tex: Rc<ResourceTexture>,
}
#[allow(non_snake_case)]
impl ID2D1ResourceTexture_Impl for ResourceTextureObj_Impl {
    fn Update(&self, min: *const u32, max: *const u32, _strides: *const u32, dims: u32, data: *const u8, count: u32) -> WResult<()> {
        // ponytail: tightly-packed source (row strides ignored) — the common full/sub-region update.
        let d = dims as usize;
        let (minv, maxv) = (unsafe { core::slice::from_raw_parts(min, d) }, unsafe { core::slice::from_raw_parts(max, d) });
        let src = if data.is_null() { &[][..] } else { unsafe { core::slice::from_raw_parts(data, count as usize) } };
        self.tex.update(minv, maxv, dims, src).map_err(Error::from)
    }
}

// --- ID2D1VertexBuffer -------------------------------------------------------------------------

/// f32-component count of a vertex-element `DXGI_FORMAT` (R32G32B32A32/B32/G32/R32_FLOAT → 4/3/2/1).
fn format_components(f: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT) -> usize {
    match f.0 {
        2 => 4,
        6 => 3,
        16 => 2,
        41 => 1,
        _ => 4,
    }
}

#[implement(ID2D1VertexBuffer)]
struct VertexBufferObj {
    vb: Rc<VertexBuffer>,
}
#[allow(non_snake_case)]
impl ID2D1VertexBuffer_Impl for VertexBufferObj_Impl {
    fn Map(&self, data: *mut *mut u8, buffersize: u32) -> WResult<()> {
        let (ptr, len) = self.vb.map();
        if buffersize as usize > len {
            return Err(E_INVALIDARG.into());
        }
        unsafe { *data = ptr };
        Ok(())
    }
    fn Unmap(&self) -> WResult<()> {
        self.vb.unmap().map_err(Error::from)
    }
}

// --- ID2D1TransformGraph -----------------------------------------------------------------------

#[implement(ID2D1TransformGraph)]
struct TransformGraphObj {
    ctx: *const EffectContext,
    graph: *mut TransformGraph,
    /// COM node identity (its `IUnknown`) → core `NodeId`, so `ConnectNode`/`SetOutputNode` resolve
    /// the nodes `AddNode` added (ours or the app's own draw transform).
    nodes: RefCell<Vec<(IUnknown, NodeId)>>,
    probes: Rc<Probes>,
}
impl TransformGraphObj_Impl {
    fn node_id(&self, node: &ID2D1TransformNode) -> Option<NodeId> {
        let raw = node.cast::<IUnknown>().ok()?.as_raw();
        self.nodes.borrow().iter().find(|(u, _)| u.as_raw() == raw).map(|(_, id)| *id)
    }
}
#[allow(non_snake_case)]
impl ID2D1TransformGraph_Impl for TransformGraphObj_Impl {
    fn GetInputCount(&self) -> u32 {
        unsafe { &*self.graph }.get_input_count()
    }
    /// The single shader draw transform: build it from the app's `ID2D1DrawTransform` and make it the
    /// graph output wired input i → node input i.
    fn SetSingleTransformNode(&self, node: Ref<ID2D1TransformNode>) -> WResult<()> {
        let node = node.as_ref().ok_or(Error::from(E_INVALIDARG))?;
        let t = app_shader_transform(unsafe { &*self.ctx }, node, &self.probes)?;
        unsafe { &mut *self.graph }.set_single_transform_node(t).map_err(Error::from)
    }
    /// `AddNode`: build the node's core `Transform` (one of our factory nodes, else the app's own draw
    /// transform), add it, and record its identity → `NodeId`.
    fn AddNode(&self, node: Ref<ID2D1TransformNode>) -> WResult<()> {
        let node = node.as_ref().ok_or(Error::from(E_INVALIDARG))?;
        let ctx = unsafe { &*self.ctx };
        let t = if let Ok(o) = node.cast_object_ref::<OffsetNodeObj>() {
            o.transform()?
        } else if let Ok(o) = node.cast_object_ref::<BorderNodeObj>() {
            o.transform()?
        } else if let Ok(o) = node.cast_object_ref::<BoundsNodeObj>() {
            o.transform()?
        } else if let Ok(o) = node.cast_object_ref::<BlendNodeObj>() {
            o.transform()?
        } else if let Ok(o) = node.cast_object_ref::<EffectNodeObj>() {
            o.transform()?
        } else {
            app_shader_transform(ctx, node, &self.probes)?
        };
        let id = unsafe { &mut *self.graph }.add_node(t);
        self.nodes.borrow_mut().push((node.cast::<IUnknown>()?, id));
        Ok(())
    }
    fn RemoveNode(&self, node: Ref<ID2D1TransformNode>) -> WResult<()> {
        let node = node.as_ref().ok_or(Error::from(E_INVALIDARG))?;
        let id = self.node_id(node).ok_or(Error::from(E_INVALIDARG))?;
        unsafe { &mut *self.graph }.remove_node(id).map_err(Error::from)
    }
    fn SetOutputNode(&self, node: Ref<ID2D1TransformNode>) -> WResult<()> {
        let node = node.as_ref().ok_or(Error::from(E_INVALIDARG))?;
        let id = self.node_id(node).ok_or(Error::from(E_INVALIDARG))?;
        unsafe { &mut *self.graph }.set_output_node(id).map_err(Error::from)
    }
    fn ConnectNode(&self, from: Ref<ID2D1TransformNode>, to: Ref<ID2D1TransformNode>, index: u32) -> WResult<()> {
        let from = self.node_id(from.as_ref().ok_or(Error::from(E_INVALIDARG))?).ok_or(Error::from(E_INVALIDARG))?;
        let to = self.node_id(to.as_ref().ok_or(Error::from(E_INVALIDARG))?).ok_or(Error::from(E_INVALIDARG))?;
        unsafe { &mut *self.graph }.connect_node(from, to, index).map_err(Error::from)
    }
    fn ConnectToEffectInput(&self, input: u32, node: Ref<ID2D1TransformNode>, node_input: u32) -> WResult<()> {
        let id = self.node_id(node.as_ref().ok_or(Error::from(E_INVALIDARG))?).ok_or(Error::from(E_INVALIDARG))?;
        unsafe { &mut *self.graph }.connect_to_effect_input(input, id, node_input).map_err(Error::from)
    }
    fn Clear(&self) {
        self.nodes.borrow_mut().clear();
        unsafe { &mut *self.graph }.clear();
    }
    fn SetPassthroughGraph(&self, i: u32) -> WResult<()> {
        unsafe { &mut *self.graph }.set_passthrough_graph(i).map_err(Error::from)
    }
}

// --- ID2D1DrawInfo (collects the draw transform's pixel shader, constant buffer, textures) ------

#[implement(ID2D1DrawInfo)]
#[derive(Default)]
struct DrawInfoObj {
    shader_id: Cell<Option<GUID>>,
    /// The pixel-shader `b0` constants, in a SHARED cell bound to the core `ShaderNode` so the app's
    /// `PrepareForRender` (`SetPixelShaderConstantBuffer`) updates the constants this draw uses, without
    /// rebuilding the graph (ComputeSharp sets its buffer here, not in `SetDrawInfo`). [G2]
    cbuffer: Rc<RefCell<Vec<u8>>>,
    /// `(register, texture)` for each `SetResourceTexture`, sampled by the shader at its computed coord: a
    /// SHARED cell bound to the core `ShaderNode` like `cbuffer`, so a texture the app binds after
    /// `SetDrawInfo` (ComputeSharp's resource texture manager, when its property is set) reaches the shader.
    textures: Rc<RefCell<Vec<(u32, Rc<ResourceTexture>)>>>,
    /// `SetVertexProcessing`: the bound vertex shader GUID, vertex buffer, and its `b0` constants.
    vs_id: Cell<Option<GUID>>,
    vb: RefCell<Option<Rc<VertexBuffer>>>,
    vs_cbuffer: RefCell<Vec<u8>>,
    /// The input sampler's `D2D1_FILTER` from `SetInputDescription` (input 0); `0` =
    /// `MIN_MAG_MIP_POINT`, the effect default (`Default`). Selects point vs bilinear when a shader
    /// samples a fractional computed position (see `ShaderNode::mag_linear`).
    filter: Cell<u32>,
}
#[allow(non_snake_case)]
impl ID2D1DrawInfo_Impl for DrawInfoObj_Impl {
    fn SetPixelShader(&self, shaderid: *const GUID, _options: D2D1_PIXEL_OPTIONS) -> WResult<()> {
        self.shader_id.set(Some(unsafe { *shaderid }));
        Ok(())
    }
    fn SetPixelShaderConstantBuffer(&self, buffer: *const u8, count: u32) -> WResult<()> {
        *self.cbuffer.borrow_mut() = unsafe { core::slice::from_raw_parts(buffer, count as usize) }.to_vec();
        Ok(())
    }
    fn SetResourceTexture(&self, index: u32, texture: Ref<ID2D1ResourceTexture>) -> WResult<()> {
        let tex = texture.as_ref().ok_or(Error::from(E_INVALIDARG))?.cast_object_ref::<ResourceTextureObj>().map_err(|_| Error::from(E_INVALIDARG))?.tex.clone();
        let mut textures = self.textures.borrow_mut();
        textures.retain(|(i, _)| *i != index); // a rebind replaces the register's texture
        textures.push((index, tex));
        Ok(())
    }
    fn SetVertexShaderConstantBuffer(&self, buffer: *const u8, count: u32) -> WResult<()> {
        *self.vs_cbuffer.borrow_mut() = unsafe { core::slice::from_raw_parts(buffer, count as usize) }.to_vec();
        Ok(())
    }
    fn SetVertexProcessing(&self, vb: Ref<ID2D1VertexBuffer>, _o: D2D1_VERTEX_OPTIONS, _b: *const D2D1_BLEND_DESCRIPTION, _r: *const D2D1_VERTEX_RANGE, vs: *const GUID) -> WResult<()> {
        if !vs.is_null() {
            self.vs_id.set(Some(unsafe { *vs }));
        }
        if let Some(v) = vb.as_ref() {
            let obj = v.cast_object_ref::<VertexBufferObj>().map_err(|_| Error::from(E_INVALIDARG))?;
            *self.vb.borrow_mut() = Some(obj.vb.clone());
        }
        Ok(())
    }
}
#[allow(non_snake_case)]
impl ID2D1RenderInfo_Impl for DrawInfoObj_Impl {
    // The input sampler's `D2D1_FILTER`: point vs bilinear when the shader samples a fractional
    // computed position (a texel-aligned tap is identical either way). The core node reads it to match
    // the app's chosen filter; input 0's filter stands for the effect (the scenes set one per effect).
    fn SetInputDescription(&self, i: u32, d: &D2D1_INPUT_DESCRIPTION) -> WResult<()> {
        if i == 0 {
            self.filter.set(d.filter.0 as u32);
        }
        Ok(())
    }
    fn SetOutputBuffer(&self, _p: D2D1_BUFFER_PRECISION, _c: D2D1_CHANNEL_DEPTH) -> WResult<()> {
        Ok(())
    }
    fn SetCached(&self, _c: windows_core::BOOL) {}
    fn SetInstructionCountHint(&self, _c: u32) {}
}

// --- ID2D1ComputeInfo (collects the compute transform's shader, constant buffer, textures) --------

#[implement(ID2D1ComputeInfo)]
#[derive(Default)]
struct ComputeInfoObj {
    shader_id: Cell<Option<GUID>>,
    cbuffer: RefCell<Vec<u8>>,
    textures: RefCell<Vec<(u32, Rc<ResourceTexture>)>>,
}
#[allow(non_snake_case)]
impl ID2D1ComputeInfo_Impl for ComputeInfoObj_Impl {
    fn SetComputeShader(&self, shaderid: *const GUID) -> WResult<()> {
        self.shader_id.set(Some(unsafe { *shaderid }));
        Ok(())
    }
    fn SetComputeShaderConstantBuffer(&self, buffer: *const u8, count: u32) -> WResult<()> {
        *self.cbuffer.borrow_mut() = unsafe { core::slice::from_raw_parts(buffer, count as usize) }.to_vec();
        Ok(())
    }
    fn SetResourceTexture(&self, index: u32, texture: Ref<ID2D1ResourceTexture>) -> WResult<()> {
        let tex = texture.as_ref().ok_or(Error::from(E_INVALIDARG))?.cast_object_ref::<ResourceTextureObj>().map_err(|_| Error::from(E_INVALIDARG))?.tex.clone();
        self.textures.borrow_mut().push((index, tex));
        Ok(())
    }
}
#[allow(non_snake_case)]
impl ID2D1RenderInfo_Impl for ComputeInfoObj_Impl {
    fn SetInputDescription(&self, _i: u32, _d: &D2D1_INPUT_DESCRIPTION) -> WResult<()> {
        Ok(())
    }
    fn SetOutputBuffer(&self, _p: D2D1_BUFFER_PRECISION, _c: D2D1_CHANNEL_DEPTH) -> WResult<()> {
        Ok(())
    }
    fn SetCached(&self, _c: windows_core::BOOL) {}
    fn SetInstructionCountHint(&self, _c: u32) {}
}

// --- verification (run under Wine: `CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER=wine cargo test ...`) -
//
// Drives the full multi-node authoring surface through the COM objects (the app effect's `Initialize`
// calls `CreateBorderTransform`/`CreateBlendTransform` + `AddNode`/`ConnectNode`/`ConnectToEffectInput`
// /`SetOutputNode`) and renders through the core, comparing byte-for-byte to the identical graph built
// directly on the core API (the custom-fx-* path the WARP oracle already verifies). Also asserts the
// invalid-call HRESULTs. fxshader-01 separately proves the author path byte-exact vs the WARP oracle
// through our d2d1.dll under Wine.
#[cfg(test)]
mod tests {
    use super::super::E_NOTIMPL;
    use super::*;
    use d2d::bitmap::Bitmap;
    use d2d::effects::{Input, NOTFOUND};
    use d2d::{bitmap_target, Factory};
    use std::cell::RefCell;

    const XML2: &str = r#"<?xml version='1.0'?><Effect><Property name='DisplayName' type='string' value='G'/><Inputs><Input name='a'/><Input name='b'/></Inputs></Effect>"#;

    /// Premultiplied source-over: src·ONE + dst·INV_SRC_ALPHA, ADD.
    const SRC_OVER: D2D1_BLEND_DESCRIPTION = D2D1_BLEND_DESCRIPTION {
        sourceBlend: D2D1_BLEND_ONE,
        destinationBlend: D2D1_BLEND_INV_SRC_ALPHA,
        blendOperation: D2D1_BLEND_OPERATION_ADD,
        sourceBlendAlpha: D2D1_BLEND_ONE,
        destinationBlendAlpha: D2D1_BLEND_INV_SRC_ALPHA,
        blendOperationAlpha: D2D1_BLEND_OPERATION_ADD,
        blendFactor: [0.0; 4],
    };

    thread_local! {
        static INVALID: RefCell<Vec<(&'static str, i32)>> = const { RefCell::new(Vec::new()) };
    }

    /// App effect: `input0 → Border(wrap) → Blend(_, input1) → output`, built over the COM surface.
    #[implement(ID2D1EffectImpl)]
    struct GraphEffect;
    #[allow(non_snake_case)]
    impl ID2D1EffectImpl_Impl for GraphEffect_Impl {
        fn Initialize(&self, ctx: Ref<ID2D1EffectContext>, graph: Ref<ID2D1TransformGraph>) -> WResult<()> {
            let ctx = ctx.as_ref().unwrap();
            let graph = graph.as_ref().unwrap();
            let border: ID2D1TransformNode = unsafe { ctx.CreateBorderTransform(D2D1_EXTEND_MODE_WRAP, D2D1_EXTEND_MODE_WRAP)? }.cast()?;
            let blend: ID2D1TransformNode = unsafe { ctx.CreateBlendTransform(2, &SRC_OVER)? }.cast()?;
            unsafe {
                graph.AddNode(&border)?;
                graph.AddNode(&blend)?;
                graph.ConnectToEffectInput(0, &border, 0)?;
                graph.ConnectNode(&border, &blend, 0)?;
                graph.ConnectToEffectInput(1, &blend, 1)?;
                graph.SetOutputNode(&blend)?;
            }
            // Record the invalid-call HRESULTs for the test to assert (0 == S_OK).
            let rec = |k, r: WResult<()>| INVALID.with(|v| v.borrow_mut().push((k, r.err().map_or(0, |e| e.code().0))));
            rec("blend0", unsafe { ctx.CreateBlendTransform(0, &SRC_OVER) }.map(|_| ()));
            rec("badborder", unsafe { ctx.CreateBorderTransform(D2D1_EXTEND_MODE(9), D2D1_EXTEND_MODE_WRAP) }.map(|_| ()));
            rec("findrt", unsafe { ctx.FindResourceTexture(&GUID::from_u128(0xdead)) }.map(|_| ()));
            rec("vs", unsafe { ctx.LoadVertexShader(&GUID::from_u128(1), &[0u8; 4]) }); // unparseable DXBC
            let orphan: ID2D1TransformNode = unsafe { ctx.CreateOffsetTransform(POINT { x: 0, y: 0 })? }.cast()?;
            rec("connect_unknown", unsafe { graph.ConnectNode(&orphan, &blend, 0) });
            INVALID.with(|v| v.borrow_mut().push(("maxfl", unsafe { ctx.GetMaximumSupportedFeatureLevel(&[]) }.err().map_or(0, |e| e.code().0))));
            Ok(())
        }
        fn PrepareForRender(&self, _c: D2D1_CHANGE_TYPE) -> WResult<()> {
            Ok(())
        }
        fn SetGraph(&self, _g: Ref<ID2D1TransformGraph>) -> WResult<()> {
            Err(E_NOTIMPL.into())
        }
    }

    unsafe extern "system" fn graph_factory(out: OutRef<IUnknown>) -> HRESULT {
        let e: ID2D1EffectImpl = GraphEffect.into();
        match e.cast::<IUnknown>() {
            Ok(u) => {
                let _ = out.write(Some(u));
                HRESULT(0)
            }
            Err(e) => e.code(),
        }
    }

    /// The identical graph built directly on the core API — the byte-for-byte reference.
    struct CoreGraph;
    impl EffectImpl for CoreGraph {
        fn initialize(&self, _p: &Props, ctx: &EffectContext, g: &mut TransformGraph) -> Result<(), HRESULT> {
            let border = g.add_node(ctx.create_border_transform(1, 1));
            let blend = g.add_node(ctx.create_blend_desc_transform(2, 2, 6, 1, 2, 6, 1, [0.0; 4])?);
            g.connect_to_effect_input(0, border, 0)?;
            g.connect_node(border, blend, 0)?;
            g.connect_to_effect_input(1, blend, 1)?;
            g.set_output_node(blend)
        }
    }

    fn render(factory: &Factory, clsid: GUID) -> Vec<u8> {
        let mut dc = factory.create_device().create_device_context();
        dc.set_dpi(96.0, 96.0);
        dc.set_target_bitmap(bitmap_target(4, 4, 96.0));
        dc.begin_draw();
        dc.clear(Some([0.0; 4]));
        let eff = dc.create_effect(clsid).unwrap();
        eff.set_input(0, Input::Bitmap(Rc::new(Bitmap::from_straight_rgba(2, 2, &[255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255]))));
        eff.set_input(1, Input::Bitmap(Rc::new(Bitmap::from_straight_rgba(4, 4, &[0, 0, 0, 128].repeat(16)))));
        dc.draw_image_effect(&eff, [0.0, 0.0], 0, 0);
        let _ = dc.end_draw();
        dc.take_target().unwrap().data
    }

    // --- compute effect ---------------------------------------------------------------------

    const CS_GUID: GUID = GUID::from_u128(0xc0_f0_0001);
    // A suite-relative shader/oracle fixture, read at runtime (the tests below guard on the suite's
    // presence and skip when absent, so a public clone without the submodule still builds).
    fn sb(rel: &str) -> Vec<u8> {
        std::fs::read(d2d_reference::suite_dir().unwrap().join(rel)).unwrap()
    }
    const CS_XML: &str = r#"<?xml version='1.0'?><Effect><Property name='DisplayName' type='string' value='C'/><Inputs></Inputs></Effect>"#;

    /// The app's 0-input compute transform (shader2-09: an 8×8 unorm generator, dispatch 2×2×1).
    #[implement(ID2D1ComputeTransform)]
    struct ComputeXform;
    #[allow(non_snake_case)]
    impl ID2D1TransformNode_Impl for ComputeXform_Impl {
        fn GetInputCount(&self) -> u32 {
            0
        }
    }
    #[allow(non_snake_case)]
    impl ID2D1Transform_Impl for ComputeXform_Impl {
        fn MapOutputRectToInputRects(&self, _o: *const RECT, _i: *mut RECT, _c: u32) -> WResult<()> {
            Ok(())
        }
        fn MapInputRectsToOutputRect(&self, _i: *const RECT, _op: *const RECT, _c: u32, output: *mut RECT, output_opaque: *mut RECT) -> WResult<()> {
            unsafe {
                *output = RECT { left: 0, top: 0, right: 8, bottom: 8 };
                *output_opaque = RECT { left: 0, top: 0, right: 8, bottom: 8 };
            }
            Ok(())
        }
        fn MapInvalidRect(&self, _i: u32, invalid: &RECT) -> WResult<RECT> {
            Ok(*invalid)
        }
    }
    #[allow(non_snake_case)]
    impl ID2D1ComputeTransform_Impl for ComputeXform_Impl {
        fn SetComputeInfo(&self, info: Ref<ID2D1ComputeInfo>) -> WResult<()> {
            unsafe { info.as_ref().unwrap().SetComputeShader(&CS_GUID) }
        }
        fn CalculateThreadgroups(&self, _rect: *const RECT, dx: *mut u32, dy: *mut u32, dz: *mut u32) -> WResult<()> {
            unsafe {
                (*dx, *dy, *dz) = (2, 2, 1);
            }
            Ok(())
        }
    }

    #[implement(ID2D1EffectImpl)]
    struct ComputeEffect;
    #[allow(non_snake_case)]
    impl ID2D1EffectImpl_Impl for ComputeEffect_Impl {
        fn Initialize(&self, ctx: Ref<ID2D1EffectContext>, graph: Ref<ID2D1TransformGraph>) -> WResult<()> {
            unsafe { ctx.as_ref().unwrap().LoadComputeShader(&CS_GUID, &sb("scenes/shader2-09-compute-uav.dxbc"))? };
            let xform: ID2D1ComputeTransform = ComputeXform.into();
            let node: ID2D1TransformNode = xform.cast()?;
            unsafe { graph.as_ref().unwrap().SetSingleTransformNode(&node) }
        }
        fn PrepareForRender(&self, _c: D2D1_CHANGE_TYPE) -> WResult<()> {
            Ok(())
        }
        fn SetGraph(&self, _g: Ref<ID2D1TransformGraph>) -> WResult<()> {
            Err(E_NOTIMPL.into())
        }
    }

    unsafe extern "system" fn compute_factory(out: OutRef<IUnknown>) -> HRESULT {
        let e: ID2D1EffectImpl = ComputeEffect.into();
        match e.cast::<IUnknown>() {
            Ok(u) => {
                let _ = out.write(Some(u));
                HRESULT(0)
            }
            Err(e) => e.code(),
        }
    }

    #[test]
    fn author_compute_effect_matches_warp() {
        let Some(_suite) = d2d_reference::suite_dir() else { return };
        let cs_bin = sb("oracle/win11-26100/shader2-09-compute-uav.bin");
        let f = Factory::new(false);
        let clsid = GUID::from_u128(0xc0_f0_0002);
        register_from_string(&f, clsid, CS_XML, Some(compute_factory), core::ptr::null(), 0).unwrap();
        let mut dc = f.create_device().create_device_context();
        dc.set_dpi(96.0, 96.0);
        dc.set_target_bitmap(bitmap_target(8, 8, 96.0));
        dc.begin_draw();
        dc.clear(Some([0.0; 4]));
        let eff = dc.create_effect(clsid).unwrap();
        dc.draw_image_effect(&eff, [0.0, 0.0], 0, 0);
        let _ = dc.end_draw();
        let bgra = dc.take_target().unwrap().data;
        // Target is premultiplied BGRA8; the UAV capture is straight RGBA8 (opaque). BGRA = swap(RGBA).
        for p in 0..64 {
            assert_eq!(bgra[p * 4], cs_bin[p * 4 + 2], "px {p} B");
            assert_eq!(bgra[p * 4 + 1], cs_bin[p * 4 + 1], "px {p} G");
            assert_eq!(bgra[p * 4 + 2], cs_bin[p * 4], "px {p} R");
            assert_eq!(bgra[p * 4 + 3], cs_bin[p * 4 + 3], "px {p} A");
        }
    }

    // --- bound constant-buffer (ComputeSharp-style) effect -----------------------------------
    // A pixel-shader effect whose b0 is NOT set at SetDrawInfo but pushed in PrepareForRender from a
    // value delivered through a D2D1_PROPERTY_BINDING set-function, exactly as ComputeSharp.D2D1 does.
    // Verifies the whole G2 bridge: capture the binding; SetValue -> the app's set-function stores the
    // bytes; PrepareForRender re-pushes them through SetPixelShaderConstantBuffer into the cell the core
    // ShaderNode reads. The output must equal the SAME shader with the constant buffer BAKED at
    // Initialize (the existing path already byte-exact to the WARP oracle) — so it is byte-exact too.
    const BOUND_GUID: GUID = GUID::from_u128(0xb0_cf_0001);
    const BOUND_XML: &str = r#"<?xml version='1.0'?><Effect><Property name='DisplayName' type='string' value='B'/><Inputs><Input name='Source'/></Inputs><Property name='Buffer' type='blob'/></Effect>"#;
    thread_local! {
        static BOUND_CBUFFER: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
        static BOUND_DXBC: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    }
    /// The `Buffer` binding set-function: store the constant-buffer bytes (ComputeSharp keeps them in
    /// the effect; this test keeps them in a thread-local, one effect per test).
    unsafe extern "system" fn set_buffer(_effect: windows_core::Ref<'_, IUnknown>, data: *const u8, size: u32) -> HRESULT {
        BOUND_CBUFFER.with(|c| *c.borrow_mut() = unsafe { core::slice::from_raw_parts(data, size as usize) }.to_vec());
        HRESULT(0)
    }

    #[implement(ID2D1DrawTransform)]
    struct BoundXform {
        draw_info: Rc<RefCell<Option<ID2D1DrawInfo>>>,
    }
    #[allow(non_snake_case)]
    impl ID2D1TransformNode_Impl for BoundXform_Impl {
        fn GetInputCount(&self) -> u32 {
            1
        }
    }
    #[allow(non_snake_case)]
    impl ID2D1Transform_Impl for BoundXform_Impl {
        fn MapOutputRectToInputRects(&self, output: *const RECT, inputs: *mut RECT, count: u32) -> WResult<()> {
            for i in 0..count as usize {
                unsafe { *inputs.add(i) = *output };
            }
            Ok(())
        }
        fn MapInputRectsToOutputRect(&self, inputs: *const RECT, _o: *const RECT, count: u32, output: *mut RECT, oo: *mut RECT) -> WResult<()> {
            unsafe {
                *output = if count == 0 { RECT { left: i32::MIN, top: i32::MIN, right: i32::MAX, bottom: i32::MAX } } else { *inputs };
                *oo = RECT { left: 0, top: 0, right: 0, bottom: 0 };
            }
            Ok(())
        }
        fn MapInvalidRect(&self, _i: u32, invalid: &RECT) -> WResult<RECT> {
            Ok(*invalid)
        }
    }
    #[allow(non_snake_case)]
    impl ID2D1DrawTransform_Impl for BoundXform_Impl {
        fn SetDrawInfo(&self, info: Ref<ID2D1DrawInfo>) -> WResult<()> {
            let info = info.as_ref().unwrap();
            unsafe { info.SetPixelShader(&BOUND_GUID, D2D1_PIXEL_OPTIONS_NONE)? };
            // Retain the DrawInfo (no constant buffer here) so PrepareForRender can push b0 onto it.
            *self.draw_info.borrow_mut() = Some(info.clone());
            Ok(())
        }
    }

    #[implement(ID2D1EffectImpl)]
    struct BoundEffect {
        draw_info: Rc<RefCell<Option<ID2D1DrawInfo>>>,
    }
    #[allow(non_snake_case)]
    impl ID2D1EffectImpl_Impl for BoundEffect_Impl {
        fn Initialize(&self, ctx: Ref<ID2D1EffectContext>, graph: Ref<ID2D1TransformGraph>) -> WResult<()> {
            unsafe { ctx.as_ref().unwrap().LoadPixelShader(&BOUND_GUID, &BOUND_DXBC.with(|d| d.borrow().clone()))? };
            let xform: ID2D1DrawTransform = BoundXform { draw_info: self.draw_info.clone() }.into();
            let node: ID2D1TransformNode = xform.cast()?;
            unsafe { graph.as_ref().unwrap().SetSingleTransformNode(&node) }
        }
        fn PrepareForRender(&self, _c: D2D1_CHANGE_TYPE) -> WResult<()> {
            if let Some(di) = self.draw_info.borrow().as_ref() {
                BOUND_CBUFFER.with(|c| unsafe { di.SetPixelShaderConstantBuffer(&c.borrow()) }).ok();
            }
            Ok(())
        }
        fn SetGraph(&self, _g: Ref<ID2D1TransformGraph>) -> WResult<()> {
            Err(E_NOTIMPL.into())
        }
    }
    unsafe extern "system" fn bound_factory(out: OutRef<IUnknown>) -> HRESULT {
        let e: ID2D1EffectImpl = BoundEffect { draw_info: Rc::new(RefCell::new(None)) }.into();
        match e.cast::<IUnknown>() {
            Ok(u) => {
                let _ = out.write(Some(u));
                HRESULT(0)
            }
            Err(e) => e.code(),
        }
    }

    #[test]
    fn author_bound_cbuffer_matches_baked() {
        let Some(_suite) = d2d_reference::suite_dir() else { return };
        let dxbc = sb("scenes/fxshader-01-scale-bias.dxbc");
        let cbuffer: Vec<u8> = [0.5f32, 0.1, -0.2, 0.0].iter().flat_map(|f| f.to_le_bytes()).collect();
        let input = || Input::Bitmap(Rc::new(Bitmap::from_straight_rgba(2, 2, &[255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 128, 128, 128, 255])));
        let render = |clsid: GUID, bound: bool| -> Vec<u8> {
            let f = Factory::new(false);
            if bound {
                BOUND_DXBC.with(|d| *d.borrow_mut() = dxbc.clone());
                // "Buffer\0" as UTF-16 (the shim has no `w!` macro).
                const BUFFER_W: &[u16] = &[b'B' as u16, b'u' as u16, b'f' as u16, b'f' as u16, b'e' as u16, b'r' as u16, 0];
                let binding = [D2D1_PROPERTY_BINDING { propertyName: windows_core::PCWSTR(BUFFER_W.as_ptr()), setFunction: Some(set_buffer), getFunction: None }];
                register_from_string(&f, clsid, BOUND_XML, Some(bound_factory), binding.as_ptr(), 1).unwrap();
            } else {
                let (dx, cb) = (dxbc.clone(), cbuffer.clone());
                f.register_effect_from_string(clsid, BOUND_XML, Rc::new(move || Box::new(d2d::effects::PixelShaderEffect { shader_id: BOUND_GUID, dxbc: dx.clone(), cbuffer: cb.clone(), input_count: 1, precision: None, input_inflate: None, filter: 0 }))).unwrap();
            }
            let mut dc = f.create_device().create_device_context();
            dc.set_dpi(96.0, 96.0);
            dc.set_target_bitmap(bitmap_target(2, 2, 96.0));
            dc.begin_draw();
            dc.clear(Some([0.0; 4]));
            let eff = dc.create_effect(clsid).unwrap();
            eff.set_input(0, input());
            if bound {
                eff.set_bound_value("Buffer", &cbuffer);
            }
            dc.draw_image_effect(&eff, [0.0, 0.0], 0, 0);
            let _ = dc.end_draw();
            dc.take_target().unwrap().data
        };
        let baked = render(GUID::from_u128(0xb0_cf_0010), false);
        let bound = render(GUID::from_u128(0xb0_cf_0011), true);
        assert_eq!(bound, baked, "a constant buffer delivered via binding + PrepareForRender must render identically to a baked one");
    }

    // --- a resource texture bound after SetDrawInfo (ComputeSharp.D2D1's resource texture manager) ---

    const LATE_GUID: GUID = GUID::from_u128(0x1a7e_0001);
    thread_local! {
        static LATE_DXBC: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
        /// 0: no texture, 1: bound in SetDrawInfo, 2: bound in PrepareForRender.
        static LATE_MODE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    }
    /// shader3-05-restex's 64-entry RGBA float LUT, created through the effect context `ctx`.
    fn late_lut(ctx: &ID2D1EffectContext) -> ID2D1ResourceTexture {
        let data: Vec<u8> = (0..64).flat_map(|i| [i as f32 / 63.0, 1.0 - i as f32 / 63.0, (i % 8) as f32 / 7.0, 1.0]).flat_map(f32::to_le_bytes).collect();
        let (extent, extend) = (64u32, D2D1_EXTEND_MODE_CLAMP);
        let props = D2D1_RESOURCE_TEXTURE_PROPERTIES {
            extents: &extent,
            dimensions: 1,
            bufferPrecision: D2D1_BUFFER_PRECISION_32BPC_FLOAT,
            channelDepth: D2D1_CHANNEL_DEPTH_4,
            filter: D2D1_FILTER_MIN_MAG_MIP_POINT,
            extendModes: &extend,
        };
        unsafe { ctx.CreateResourceTexture(None, &props, Some(&data), None) }.unwrap()
    }

    #[implement(ID2D1DrawTransform)]
    struct LateXform {
        draw_info: Rc<RefCell<Option<ID2D1DrawInfo>>>,
        ctx: ID2D1EffectContext,
    }
    #[allow(non_snake_case)]
    impl ID2D1TransformNode_Impl for LateXform_Impl {
        fn GetInputCount(&self) -> u32 {
            0
        }
    }
    #[allow(non_snake_case)]
    impl ID2D1Transform_Impl for LateXform_Impl {
        fn MapOutputRectToInputRects(&self, _o: *const RECT, _i: *mut RECT, _c: u32) -> WResult<()> {
            Ok(())
        }
        fn MapInputRectsToOutputRect(&self, _i: *const RECT, _o: *const RECT, _c: u32, output: *mut RECT, oo: *mut RECT) -> WResult<()> {
            unsafe {
                *output = RECT { left: i32::MIN, top: i32::MIN, right: i32::MAX, bottom: i32::MAX };
                *oo = RECT::default();
            }
            Ok(())
        }
        fn MapInvalidRect(&self, _i: u32, invalid: &RECT) -> WResult<RECT> {
            Ok(*invalid)
        }
    }
    #[allow(non_snake_case)]
    impl ID2D1DrawTransform_Impl for LateXform_Impl {
        fn SetDrawInfo(&self, info: Ref<ID2D1DrawInfo>) -> WResult<()> {
            let info = info.as_ref().unwrap();
            unsafe { info.SetPixelShader(&LATE_GUID, D2D1_PIXEL_OPTIONS_NONE)? };
            if LATE_MODE.with(|m| m.get()) == 1 {
                unsafe { info.SetResourceTexture(1, &late_lut(&self.ctx))? };
            }
            *self.draw_info.borrow_mut() = Some(info.clone());
            Ok(())
        }
    }

    #[implement(ID2D1EffectImpl)]
    struct LateEffect {
        draw_info: Rc<RefCell<Option<ID2D1DrawInfo>>>,
        /// The effect context kept past Initialize, as ComputeSharp keeps it.
        ctx: RefCell<Option<ID2D1EffectContext>>,
    }
    #[allow(non_snake_case)]
    impl ID2D1EffectImpl_Impl for LateEffect_Impl {
        fn Initialize(&self, ctx: Ref<ID2D1EffectContext>, graph: Ref<ID2D1TransformGraph>) -> WResult<()> {
            let ctx = ctx.as_ref().unwrap();
            unsafe { ctx.LoadPixelShader(&LATE_GUID, &LATE_DXBC.with(|d| d.borrow().clone()))? };
            *self.ctx.borrow_mut() = Some(ctx.clone());
            let xform: ID2D1DrawTransform = LateXform { draw_info: self.draw_info.clone(), ctx: ctx.clone() }.into();
            unsafe { graph.as_ref().unwrap().SetSingleTransformNode(&xform.cast::<ID2D1TransformNode>()?) }
        }
        fn PrepareForRender(&self, _c: D2D1_CHANGE_TYPE) -> WResult<()> {
            if LATE_MODE.with(|m| m.get()) == 2 {
                if let (Some(info), Some(ctx)) = (self.draw_info.borrow().as_ref(), self.ctx.borrow().as_ref()) {
                    unsafe { info.SetResourceTexture(1, &late_lut(ctx))? };
                }
            }
            Ok(())
        }
        fn SetGraph(&self, _g: Ref<ID2D1TransformGraph>) -> WResult<()> {
            Err(E_NOTIMPL.into())
        }
    }
    unsafe extern "system" fn late_factory(out: OutRef<IUnknown>) -> HRESULT {
        let e: ID2D1EffectImpl = LateEffect { draw_info: Rc::new(RefCell::new(None)), ctx: RefCell::new(None) }.into();
        match e.cast::<IUnknown>() {
            Ok(u) => {
                let _ = out.write(Some(u));
                HRESULT(0)
            }
            Err(e) => e.code(),
        }
    }

    /// A resource texture the app binds in PrepareForRender, created through the effect context it kept
    /// from Initialize (ComputeSharp's manager, set as a property after the graph is built), reaches the
    /// shader exactly as one bound in SetDrawInfo — Paint.NET's clouds read their permutation this way.
    #[test]
    fn author_late_resource_texture_matches_early() {
        let Some(_suite) = d2d_reference::suite_dir() else { return };
        LATE_DXBC.with(|d| *d.borrow_mut() = sb("scenes/shader3-05-restex.dxbc"));
        let render = |mode: u8, clsid: GUID| -> Vec<u8> {
            LATE_MODE.with(|m| m.set(mode));
            let f = Factory::new(false);
            register_from_string(&f, clsid, CS_XML, Some(late_factory), core::ptr::null(), 0).unwrap();
            let mut dc = f.create_device().create_device_context();
            dc.set_dpi(96.0, 96.0);
            dc.set_target_bitmap(bitmap_target(64, 1, 96.0));
            dc.begin_draw();
            dc.clear(Some([0.0; 4]));
            let eff = dc.create_effect(clsid).unwrap();
            dc.draw_image_effect(&eff, [0.0, 0.0], 0, 0);
            let _ = dc.end_draw();
            dc.take_target().unwrap().data
        };
        let none = render(0, GUID::from_u128(0x1a7e_0010));
        let early = render(1, GUID::from_u128(0x1a7e_0011));
        let late = render(2, GUID::from_u128(0x1a7e_0012));
        assert_ne!(early, none, "the LUT must reach the shader");
        assert_eq!(late, early, "a texture bound after SetDrawInfo must render as one bound in it");
    }

    /// Native's factories expose `ID2D1Multithread` and an effect is an `ID2D1Image` with a factory:
    /// ComputeSharp's resource texture manager goes through both before it creates its texture.
    #[test]
    fn factory_is_multithread_and_effect_is_image() {
        let f: ID2D1Factory1 = super::super::factory::FactoryObj::new(true).into();
        assert!(unsafe { f.cast::<ID2D1Multithread>().unwrap().GetMultithreadProtected() }.as_bool());
        let core = Factory::new(false);
        let dc = core.create_device().create_device_context();
        let flood = dc.create_effect(d2d::abi::CLSID_D2D1Flood).unwrap();
        let effect: ID2D1Effect = super::super::image::EffectObj::new(flood, d2d::abi::CLSID_D2D1Flood).into();
        let factory = unsafe { effect.cast::<ID2D1Image>().unwrap().GetFactory() }.unwrap();
        assert!(!unsafe { factory.cast::<ID2D1Multithread>().unwrap().GetMultithreadProtected() }.as_bool());
    }

    // --- vertex-processing effect ------------------------------------------------------------

    const VS_GUID: GUID = GUID::from_u128(0x7e5_0001);
    const PS_GUID: GUID = GUID::from_u128(0x7e5_0002);
    const VSFX_XML: &str = r#"<?xml version='1.0'?><Effect><Property name='DisplayName' type='string' value='V'/><Inputs><Input name='a'/></Inputs></Effect>"#;

    /// The app's draw transform: a quad mesh, the vsfx-deform VS (scale 1), the PS sampling input 0.
    #[implement(ID2D1DrawTransform)]
    struct VertexXform {
        vb: ID2D1VertexBuffer,
    }
    #[allow(non_snake_case)]
    impl ID2D1TransformNode_Impl for VertexXform_Impl {
        fn GetInputCount(&self) -> u32 {
            1
        }
    }
    #[allow(non_snake_case)]
    impl ID2D1Transform_Impl for VertexXform_Impl {
        fn MapOutputRectToInputRects(&self, o: *const RECT, i: *mut RECT, c: u32) -> WResult<()> {
            for k in 0..c as usize {
                unsafe { *i.add(k) = *o };
            }
            Ok(())
        }
        fn MapInputRectsToOutputRect(&self, _i: *const RECT, _op: *const RECT, _c: u32, output: *mut RECT, oo: *mut RECT) -> WResult<()> {
            unsafe {
                *output = RECT { left: 0, top: 0, right: 32, bottom: 32 };
                *oo = RECT::default();
            }
            Ok(())
        }
        fn MapInvalidRect(&self, _i: u32, invalid: &RECT) -> WResult<RECT> {
            Ok(*invalid)
        }
    }
    #[allow(non_snake_case)]
    impl ID2D1DrawTransform_Impl for VertexXform_Impl {
        fn SetDrawInfo(&self, info: Ref<ID2D1DrawInfo>) -> WResult<()> {
            // The VS reads the scene→clip system cbuffer D2D binds at b0, so no user cbuffer is set.
            let info = info.as_ref().unwrap();
            unsafe {
                info.SetVertexProcessing(&self.vb, D2D1_VERTEX_OPTIONS_NONE, None, None, Some(&VS_GUID))?;
                info.SetPixelShader(&PS_GUID, D2D1_PIXEL_OPTIONS_NONE)?;
                info.SetInputDescription(0, D2D1_INPUT_DESCRIPTION { filter: D2D1_FILTER_MIN_MAG_MIP_POINT, levelOfDetailCount: 0 })?;
            }
            Ok(())
        }
    }

    #[implement(ID2D1EffectImpl)]
    struct VertexEffect;
    #[allow(non_snake_case)]
    impl ID2D1EffectImpl_Impl for VertexEffect_Impl {
        fn Initialize(&self, ctx: Ref<ID2D1EffectContext>, graph: Ref<ID2D1TransformGraph>) -> WResult<()> {
            use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32_FLOAT;
            let ctx = ctx.as_ref().unwrap();
            // Bound for the fn body so the vertex-shader bytes outlive the CreateVertexBuffer call below
            // (`shaderBufferWithInputSignature` holds a raw pointer into them).
            let (vs_dxbc, ps_dxbc) = (sb("scenes/vsfx-deform.vs.dxbc"), sb("scenes/vsfx-deform.ps.dxbc"));
            unsafe {
                ctx.LoadVertexShader(&VS_GUID, &vs_dxbc)?;
                ctx.LoadPixelShader(&PS_GUID, &ps_dxbc)?;
            }
            // The ffiauthor-01-vertex-deform mesh: a quad (half 0.8, scene space) rotated 20°, uv 0..1.
            let (co, si, h) = (20.0f64.to_radians().cos(), 20.0f64.to_radians().sin(), 0.8f64);
            let rot = |x: f64, y: f64| ((x * co - y * si) as f32, (x * si + y * co) as f32);
            let (tl, tr, bl, br) = (rot(-h, h), rot(h, h), rot(-h, -h), rot(h, -h));
            let v = |p: (f32, f32), u: f32, w: f32| [p.0, p.1, u, w].iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
            let (a, b, c, d) = (v(tl, 0.0, 0.0), v(tr, 1.0, 0.0), v(bl, 0.0, 1.0), v(br, 1.0, 1.0));
            let data: Vec<u8> = [&a, &b, &c, &c, &b, &d].into_iter().flatten().copied().collect();
            let elems = [
                D2D1_INPUT_ELEMENT_DESC { semanticName: windows_core::s!("POSITION"), semanticIndex: 0, format: DXGI_FORMAT_R32G32_FLOAT, inputSlot: 0, alignedByteOffset: 0 },
                D2D1_INPUT_ELEMENT_DESC { semanticName: windows_core::s!("TEXCOORD"), semanticIndex: 0, format: DXGI_FORMAT_R32G32_FLOAT, inputSlot: 0, alignedByteOffset: 8 },
            ];
            let props = D2D1_VERTEX_BUFFER_PROPERTIES { inputCount: 1, usage: D2D1_VERTEX_USAGE_STATIC, data: data.as_ptr(), byteWidth: data.len() as u32 };
            let custom = D2D1_CUSTOM_VERTEX_BUFFER_PROPERTIES { shaderBufferWithInputSignature: vs_dxbc.as_ptr(), shaderBufferSize: vs_dxbc.len() as u32, inputElements: elems.as_ptr(), elementCount: 2, stride: 16 };
            let vb = unsafe { ctx.CreateVertexBuffer(&props, None, Some(&custom))? };
            let xform: ID2D1DrawTransform = VertexXform { vb }.into();
            let node: ID2D1TransformNode = xform.cast()?;
            unsafe { graph.as_ref().unwrap().SetSingleTransformNode(&node) }
        }
        fn PrepareForRender(&self, _c: D2D1_CHANGE_TYPE) -> WResult<()> {
            Ok(())
        }
        fn SetGraph(&self, _g: Ref<ID2D1TransformGraph>) -> WResult<()> {
            Err(E_NOTIMPL.into())
        }
    }

    unsafe extern "system" fn vertex_factory(out: OutRef<IUnknown>) -> HRESULT {
        let e: ID2D1EffectImpl = VertexEffect.into();
        match e.cast::<IUnknown>() {
            Ok(u) => {
                let _ = out.write(Some(u));
                HRESULT(0)
            }
            Err(e) => e.code(),
        }
    }

    #[test]
    fn author_vertex_effect_matches_warp() {
        let Some(_suite) = d2d_reference::suite_dir() else { return };
        let f = Factory::new(false);
        let clsid = GUID::from_u128(0x7e5_0003);
        register_from_string(&f, clsid, VSFX_XML, Some(vertex_factory), core::ptr::null(), 0).unwrap();
        let mut dc = f.create_device().create_device_context();
        dc.set_dpi(96.0, 96.0);
        dc.set_target_bitmap(bitmap_target(32, 32, 96.0));
        dc.begin_draw();
        dc.clear(Some([0.0; 4]));
        let eff = dc.create_effect(clsid).unwrap();
        let rgba: Vec<u8> = (0..32u32 * 32).flat_map(|i| [((i % 32) * 8 & 255) as u8, ((i / 32) * 8 & 255) as u8, 128, 255]).collect();
        eff.set_input(0, Input::Bitmap(Rc::new(Bitmap::from_straight_rgba(32, 32, &rgba))));
        dc.draw_image_effect(&eff, [0.0, 0.0], 1, 0);
        let _ = dc.end_draw();
        // The deformed custom-VS effect, built through the full COM surface, is byte-exact vs native WARP.
        assert_eq!(dc.take_target().unwrap().data, sb("oracle/win11-26100/ffiauthor-01-vertex-deform.bgra"), "vertex effect must match the WARP oracle");
    }

    // --- CreateColorContextFromWicColorContext -----------------------------------------------

    /// A stand-in `IWICColorContext` returning a chosen type/Exif-space/profile.
    #[implement(windows::Win32::Graphics::Imaging::IWICColorContext)]
    struct MockWic {
        ty: i32,
        exif: u32,
        profile: Vec<u8>,
    }
    #[allow(non_snake_case)]
    impl windows::Win32::Graphics::Imaging::IWICColorContext_Impl for MockWic_Impl {
        fn InitializeFromFilename(&self, _: &windows_core::PCWSTR) -> WResult<()> {
            Ok(())
        }
        fn InitializeFromMemory(&self, _: *const u8, _: u32) -> WResult<()> {
            Ok(())
        }
        fn InitializeFromExifColorSpace(&self, _: u32) -> WResult<()> {
            Ok(())
        }
        fn GetType(&self) -> WResult<windows::Win32::Graphics::Imaging::WICColorContextType> {
            Ok(windows::Win32::Graphics::Imaging::WICColorContextType(self.ty))
        }
        fn GetProfileBytes(&self, cb: u32, buf: *mut u8, actual: *mut u32) -> WResult<()> {
            unsafe {
                *actual = self.profile.len() as u32;
                if cb > 0 {
                    core::ptr::copy_nonoverlapping(self.profile.as_ptr(), buf, (cb as usize).min(self.profile.len()));
                }
            }
            Ok(())
        }
        fn GetExifColorSpace(&self) -> WResult<u32> {
            Ok(self.exif)
        }
    }

    #[test]
    fn color_context_from_wic_maps_native() {
        use d2d::color_context::Space;
        use windows::Win32::Graphics::Imaging::IWICColorContext;
        let from = |ty, exif, profile: &[u8]| {
            let wic: IWICColorContext = MockWic { ty, exif, profile: profile.to_vec() }.into();
            super::super::image::color_context_from_wic(&wic)
        };
        // Profile → ICC bytes (Custom); Exif ≠ 2 → sRGB; Exif == 2 → E_NOTIMPL; Uninitialized → E_INVALIDARG.
        let p = from(1, 0, &[1, 2, 3, 4]).unwrap();
        assert_eq!(p.get_profile(), &[1, 2, 3, 4]);
        assert!(matches!(p.space(), Space::Custom));
        assert!(matches!(from(2, 1, &[]).unwrap().space(), Space::Srgb));
        // Exif Adobe RGB (2) → the synthesised 480-byte Adobe RGB ICC (ported from Direct2D).
        let adobe = from(2, 2, &[]).unwrap();
        let prof = adobe.get_profile();
        assert_eq!(prof.len(), 480);
        assert_eq!(&prof[0..4], &[0, 0, 1, 0xe0]); // ICC size field = 480
        assert_eq!(&prof[316..320], &[0, 0, 0x9c, 0x18]); // rXYZ.X Adobe RGB colorant
        assert_eq!(&prof[272..274], &[0x02, 0x33]); // rTRC gamma ≈ 2.199
        assert_eq!(&prof[396..401], b"opRGB");
        assert_eq!(from(0, 0, &[]).err().map(|e| e.code()), Some(E_INVALIDARG));
    }

    #[test]
    fn author_multi_node_graph_matches_core() {
        let ffi = Factory::new(false);
        let cl_ffi = GUID::from_u128(0xa17_0001);
        register_from_string(&ffi, cl_ffi, XML2, Some(graph_factory), core::ptr::null(), 0).unwrap();
        let ffi_px = render(&ffi, cl_ffi);

        let core = Factory::new(false);
        let cl_core = GUID::from_u128(0xa17_0002);
        core.register_effect_from_string(cl_core, XML2, Rc::new(|| Box::new(CoreGraph))).unwrap();
        let core_px = render(&core, cl_core);

        assert_eq!(ffi_px, core_px, "FFI-authored multi-node graph must match the direct core build");

        let inv = INVALID.with(|v| v.borrow().clone());
        let hr = |k| inv.iter().find(|(l, _)| *l == k).map(|(_, h)| *h);
        assert_eq!(hr("blend0"), Some(E_INVALIDARG.0), "CreateBlendTransform(0)");
        assert_eq!(hr("badborder"), Some(E_INVALIDARG.0), "CreateBorderTransform(invalid mode)");
        assert_eq!(hr("findrt"), Some(NOTFOUND.0), "FindResourceTexture(absent)");
        assert_eq!(hr("vs"), Some(E_FAIL.0), "LoadVertexShader(unparseable DXBC)");
        assert_eq!(hr("connect_unknown"), Some(E_INVALIDARG.0), "ConnectNode(un-added node)");
        assert_eq!(hr("maxfl"), Some(NO_FEATURE_LEVEL.0), "GetMaximumSupportedFeatureLevel(count 0)");
    }
}
