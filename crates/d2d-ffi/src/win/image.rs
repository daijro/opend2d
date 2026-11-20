//! Effects (2B/2C), effect / image-source output images, SVG documents (2E) and sprite batches
//! (extras). Each wraps the matching `d2d` core handle; the device context recovers them from an
//! `ID2D1Image`/`ID2D1Effect` with `cast_object_ref` (see [`image_target`]/[`effect_input`]).

use super::resources::{BitmapObj, SolidBrushObj};
use super::geometry::GeometryObj;
use super::{E_INVALIDARG, E_NOTIMPL};
use d2d::abi::{CLSID_D2D1BitmapSource, CLSID_D2D1ColorManagement, CLSID_D2D1Flood, CLSID_D2D1Histogram};
use d2d::bitmap::Bitmap;
use d2d::command_stream::StreamSink;
use d2d::geometry::{Figure, FillMode, Geometry, Segment};
use d2d::resource::SolidColorBrush;
use d2d::color_context::ColorContext;
use d2d::effects::{Effect, Input, Lut3d, Prop, PropKind};
use windows_core::GUID;
use d2d::extras::image_source::ImageSource;
use d2d::extras::mesh::GradientMesh;
use d2d::extras::Sprite;
use d2d::CommandList;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use windows::Win32::Graphics::Direct2D::Common::*;
use windows::Win32::Graphics::Direct2D::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_COLOR_SPACE_TYPE;
use windows::Win32::Graphics::Imaging::IWICBitmapSource;
use windows_core::{ComObject, IUnknown, Interface, OutRef, Ref, Result as WResult, implement};

// --- effect -----------------------------------------------------------------------------------

#[implement(ID2D1Effect, ID2D1Image)]
pub(crate) struct EffectObj {
    pub(crate) effect: Effect,
    clsid: GUID,
    input_count: RefCell<u32>,
    /// BGRA `draw_flood` colour, captured when this is a bare `Flood` (see `GetOutput`).
    flood: RefCell<Option<[i32; 4]>>,
    /// The input images as the app set them, so `GetInput` hands back exactly what was bound. Shared
    /// (`Rc`) with the output image so the draw path can walk nested effects' live command-list inputs.
    inputs: Rc<RefCell<Vec<Option<ID2D1Image>>>>,
    /// ColorManagement pending colour contexts + alpha mode (props 0,2,4): when both contexts are set
    /// the core transform is built and pushed (`set_cms`), so the cms evaluates in f32 through the
    /// graph (kept at the context precision for its downstream consumer), not pre-quantised to 8bpc.
    cms_src: RefCell<Option<ComObject<ColorContextObj>>>,
    cms_dst: RefCell<Option<ComObject<ColorContextObj>>>,
    cms_alpha: Cell<u32>,
    cms_intents: Cell<[u32; 2]>,
    cms_quality: Cell<u32>,
    /// A SIMPLE-profile context pair native fails to realise: `GetOutput` hands back [`CmFailObj`].
    cms_fail: Cell<bool>,
    /// `ID2D1Factory1::GetEffectProperties` returns a schema-only Properties view: `GetValue` reports
    /// the effect's *registration* default, which can differ from a created instance's value (Flood's
    /// schema Color is {0,0,0,0}, a created Flood's is {0,0,0,1}). Set on the transient GetEffectProperties object.
    schema: Cell<bool>,
}
impl EffectObj {
    pub(crate) fn new(effect: Effect, clsid: GUID) -> Self {
        let initial_inputs = effect.initial_input_count();
        Self {
            effect,
            clsid,
            input_count: RefCell::new(initial_inputs),
            flood: RefCell::new(None),
            inputs: Rc::new(RefCell::new(Vec::new())),
            cms_src: RefCell::new(None),
            cms_dst: RefCell::new(None),
            cms_alpha: Cell::new(1),
            cms_intents: Cell::new([0, 0]),
            cms_quality: Cell::new(1),
            cms_fail: Cell::new(false),
            schema: Cell::new(false),
        }
    }
    /// A schema-only Properties view for `GetEffectProperties` (see the `schema` field).
    pub(crate) fn into_schema(self) -> Self {
        self.schema.set(true);
        self
    }
    /// Remember the input bound at `index` (so `GetInput` is faithful).
    fn store_input(&self, index: u32, input: Option<ID2D1Image>) {
        let mut v = self.inputs.borrow_mut();
        if v.len() <= index as usize {
            v.resize(index as usize + 1, None);
        }
        v[index as usize] = input;
    }
    /// (Re)build the ColorManagement transform once both contexts are set: a SIMPLE-profile pair native
    /// fails to realise (`cms_fail` → `GetOutput` yields [`CmFailObj`]); otherwise push the core
    /// transform so the effect evaluates it in f32 through the graph.
    fn rebuild_cms(&self) {
        let (s, d) = (self.cms_src.borrow(), self.cms_dst.borrow());
        let default = d2d::color_context::ColorContext::from_color_space(d2d::abi::D2D1_COLOR_SPACE_SRGB);
        let source = s.as_ref().map_or(&default, |v|&v.ctx);
        let dest = d.as_ref().map_or(&default, |v|&v.ctx);
        let transform = d2d::color_context::ColorTransform::with_options(source, dest,
            self.cms_alpha.get() == 2, self.cms_intents.get(), self.cms_quality.get());
        self.cms_fail.set(transform.is_none());
        self.effect.set_cms(transform);
    }
}

/// Decode a `SetValue` blob (little-endian, exactly as the harness serialises it) into a core
/// `Prop`, keyed by the declared property type. Mirrors `conformance.rs::eff_prop`.
fn decode_prop(ty: D2D1_PROPERTY_TYPE, data: *const u8, size: u32) -> Option<Prop> {
    let b = if data.is_null() { &[][..] } else { unsafe { core::slice::from_raw_parts(data, size as usize) } };
    let u = || b.get(..4).map(|s| u32::from_le_bytes(s.try_into().unwrap()));
    let arr = |k: usize| -> Option<Vec<f32>> {
        (b.len() >= k * 4).then(|| (0..k).map(|i| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())).collect())
    };
    Some(match ty {
        D2D1_PROPERTY_TYPE_FLOAT => Prop::F32(f32::from_bits(u()?)),
        D2D1_PROPERTY_TYPE_UINT32 | D2D1_PROPERTY_TYPE_ENUM => Prop::U32(u()?),
        D2D1_PROPERTY_TYPE_INT32 => Prop::I32(u()? as i32),
        D2D1_PROPERTY_TYPE_BOOL => Prop::Bool(u()? != 0),
        D2D1_PROPERTY_TYPE_VECTOR2 => Prop::V2(arr(2)?.try_into().unwrap()),
        D2D1_PROPERTY_TYPE_VECTOR3 => Prop::V3(arr(3)?.try_into().unwrap()),
        D2D1_PROPERTY_TYPE_VECTOR4 => Prop::V4(arr(4)?.try_into().unwrap()),
        D2D1_PROPERTY_TYPE_MATRIX_3X2 => Prop::M32(arr(6)?.try_into().unwrap()),
        D2D1_PROPERTY_TYPE_MATRIX_4X4 => Prop::M44(arr(16)?.try_into().unwrap()),
        D2D1_PROPERTY_TYPE_MATRIX_5X4 => Prop::M54(arr(20)?.try_into().unwrap()),
        // Float-array blob (e.g. the ConvolveMatrix kernel), serialised as little-endian f32s.
        D2D1_PROPERTY_TYPE_BLOB => Prop::Floats((0..b.len() / 4).map(|i| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())).collect()),
        _ => return None,
    })
}

impl ID2D1Properties_Impl for EffectObj_Impl {
    fn SetValue(&self, index: u32, r#type: D2D1_PROPERTY_TYPE, data: *const u8, datasize: u32) -> WResult<()> {
        // ColorManagement: properties 0/2 are the source/destination colour contexts (COLOR_CONTEXT),
        // 4 is the alpha mode. Collect them and (re)build the core transform; the effect then evaluates
        // in f32 through the graph (kept at the context precision for a downstream effect), not a
        // pre-quantised 8bpc bitmap.
        // System properties (D2D1_PROPERTY_PRECISION, …) take the generic path: every effect carries them, and
        // Paint.NET's BitmapImage sets the precision on its ColorManagement node.
        if self.clsid == CLSID_D2D1ColorManagement && index & 0x8000_0000 == 0 {
            match index {
                0 | 2 => {
                    if r#type != D2D1_PROPERTY_TYPE_COLOR_CONTEXT || data.is_null() || datasize as usize != core::mem::size_of::<usize>() {
                        return Err(E_INVALIDARG.into());
                    }
                    let ctx = color_context_of(data);
                    if index == 0 { *self.cms_src.borrow_mut() = ctx; } else { *self.cms_dst.borrow_mut() = ctx; }
                }
                1 | 3 | 4 | 5 => {
                    if data.is_null() || datasize != 4 { return Err(E_INVALIDARG.into()); }
                    let value = unsafe { core::ptr::read_unaligned(data.cast::<u32>()) };
                    match index {
                        1 | 3 if value <= 3 => {
                            let mut intents = self.cms_intents.get();
                            intents[(index / 2) as usize] = value;
                            self.cms_intents.set(intents);
                        }
                        4 if value == 1 || value == 2 => self.cms_alpha.set(value),
                        5 if value <= 2 => self.cms_quality.set(value),
                        _ => return Err(E_INVALIDARG.into()),
                    }
                }
                _ => return Err(E_INVALIDARG.into()),
            }
            self.rebuild_cms();
            return Ok(());
        }
        // The LookupTable3D effect's LUT property is an IUnknown → recover the `ID2D1LookupTable3D`
        // and bind its core `Lut3d` (the only built-in with an IUnknown property).
        if r#type == D2D1_PROPERTY_TYPE_IUNKNOWN {
            let raw = unsafe { *(data as *const *mut core::ffi::c_void) };
            // The D2D1BitmapSource effect's WicBitmapSource property (index 0, an IUnknown): native
            // reads ANY IWICBitmapSource through its public vtable (GetSize/CopyPixels), including a
            // foreign (non-opend2d) source from the system/Wine-builtin WIC. Convert it to a
            // bitmap and bridge it as the core effect's input 0 — the bitmap-source effect treats its source as
            // input 0 (conformance.rs's BitmapSource path) — so the effect renders instead of staying
            // empty when the source is not one of our own WIC objects.
            if self.clsid == CLSID_D2D1BitmapSource && index == 0 {
                let wic: IWICBitmapSource =
                    unsafe { IUnknown::from_raw_borrowed(&raw) }.ok_or(windows_core::Error::from(E_INVALIDARG))?.cast()?;
                self.effect.set_input(0, Input::Bitmap(Rc::new(bitmap_from_wic(&wic)?)));
                let mut c = self.input_count.borrow_mut();
                *c = (*c).max(1);
                return Ok(());
            }
            let lut = unsafe { windows_core::IUnknown::from_raw_borrowed(&raw) }.and_then(|u| u.cast_object_ref::<LutObj>().ok());
            if let Some(lut) = lut {
                self.effect.set_lookup_table(lut.lut.clone());
                return Ok(());
            }
            // A custom effect's IUnknown property — ComputeSharp.D2D1's `TransformMapper` and
            // `ResourceTextureManager0..15` (G14) — goes to its binding setter as native passes it: the
            // interface pointer, which the effect impl keeps. A resource texture manager then binds its
            // texture to the draw info (SetResourceTexture), so a shader's lookup table (Paint.NET's
            // clouds permutation) reaches the executor. A non-lut IUnknown on a built-in (which never
            // declares one but LookupTable3D) stays E_NOTIMPL.
            if self.effect.custom_property_count().is_some() {
                if let Some(name) = self.effect.custom_property_name(index) {
                    let bytes = unsafe { core::slice::from_raw_parts(data, datasize as usize) };
                    self.effect.set_bound_value(&name, bytes);
                }
                return Ok(());
            }
            return Err(windows_core::Error::from(E_NOTIMPL));
        }
        let prop = decode_prop(r#type, data, datasize).ok_or(windows_core::Error::from(E_NOTIMPL))?;
        // A bare Flood drawn directly keeps 2A's composite-aware `draw_flood` path (layers-10); grab
        // its colour as BGRA, exactly as `conformance.rs` does.
        if index == 0 && self.clsid == CLSID_D2D1Flood {
            if let Prop::V4(c) = &prop {
                let q = |v: f32| (v * 255.0).round_ties_even().clamp(0.0, 255.0) as i32;
                *self.flood.borrow_mut() = Some([q(c[2]), q(c[1]), q(c[0]), q(c[3])]);
            }
        }
        // A custom effect: deliver the raw value to the property's binding setter (ComputeSharp's
        // `Buffer`) — a no-op without a binding — so the app's `PrepareForRender` sees it; then store
        // the value (for `GetValue`) and rebuild as before. [G2]
        if self.effect.custom_property_count().is_some() {
            if let Some(name) = self.effect.custom_property_name(index) {
                let bytes = if data.is_null() { &[][..] } else { unsafe { core::slice::from_raw_parts(data, datasize as usize) } };
                self.effect.set_bound_value(&name, bytes);
            }
        }
        self.effect.set_value(index, prop).map_err(windows_core::Error::from)
    }
    fn GetPropertyCount(&self) -> u32 {
        if let Some(n) = self.effect.custom_property_count() {
            return n;
        }
        builtin_props(&self.clsid).len() as u32
    }
    fn GetPropertyName(&self, i: u32, n: windows_core::PWSTR, c: u32) -> WResult<()> {
        let name = if self.effect.custom_property_count().is_some() {
            self.effect.custom_property_name(i).ok_or(windows_core::Error::from(E_NOTIMPL))?
        } else {
            builtin_props(&self.clsid).get(i as usize).ok_or(windows_core::Error::from(E_NOTIMPL))?.0.to_owned()
        };
        // Write the name as NUL-terminated UTF-16, clamped to the caller's buffer `c`.
        let utf16: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
        for (k, w) in utf16.iter().take(c as usize).enumerate() {
            unsafe { *n.0.add(k) = *w };
        }
        Ok(())
    }
    fn GetPropertyNameLength(&self, i: u32) -> u32 {
        if self.effect.custom_property_count().is_some() {
            return self.effect.custom_property_name(i).map(|n| n.encode_utf16().count() as u32).unwrap_or(0);
        }
        builtin_props(&self.clsid).get(i as usize).map(|(name, _, _)| name.encode_utf16().count() as u32).unwrap_or(0)
    }
    fn GetType(&self, i: u32) -> D2D1_PROPERTY_TYPE {
        if self.effect.custom_property_count().is_some() {
            return self.effect.custom_property_kind(i).map(prop_type_of).unwrap_or(D2D1_PROPERTY_TYPE(0));
        }
        builtin_props(&self.clsid).get(i as usize).map(|&(_, ty, _)| ty).unwrap_or(D2D1_PROPERTY_TYPE(0))
    }
    fn GetPropertyIndex(&self, n: &windows_core::PCWSTR) -> u32 {
        let name = unsafe { n.to_string() }.unwrap_or_default();
        if self.effect.custom_property_count().is_some() {
            return self.effect.custom_property_index(&name).unwrap_or(u32::MAX);
        }
        builtin_props(&self.clsid).iter().position(|(nm, _, _)| *nm == name).map(|i| i as u32).unwrap_or(u32::MAX)
    }
    fn SetValueByName(&self, n: &windows_core::PCWSTR, t: D2D1_PROPERTY_TYPE, d: *const u8, s: u32) -> WResult<()> {
        // Direct2D resolves the name to an index and delegates to SetValue;
        // an unknown name is E_INVALIDARG.
        let idx = self.GetPropertyIndex(n);
        if idx == u32::MAX {
            return Err(E_INVALIDARG.into());
        }
        self.SetValue(idx, t, d, s)
    }
    fn GetValueByName(&self, n: &windows_core::PCWSTR, t: D2D1_PROPERTY_TYPE, d: *mut u8, s: u32) -> WResult<()> {
        // Direct2D's GetValueByName: a zero size is
        // E_INVALIDARG, else resolve the name to an index and delegate to GetValue.
        if s == 0 {
            return Err(E_INVALIDARG.into());
        }
        let idx = self.GetPropertyIndex(n);
        if idx == u32::MAX {
            return Err(E_INVALIDARG.into());
        }
        self.GetValue(idx, t, d, s)
    }
    fn GetValue(&self, i: u32, _t: D2D1_PROPERTY_TYPE, d: *mut u8, s: u32) -> WResult<()> {
        if self.clsid == CLSID_D2D1ColorManagement && matches!(i, 1 | 3 | 4 | 5) {
            if d.is_null() || s != 4 { return Err(E_INVALIDARG.into()); }
            let value = match i { 1 | 3 => self.cms_intents.get()[(i/2) as usize], 4 => self.cms_alpha.get(), _ => self.cms_quality.get() };
            unsafe { core::ptr::write_unaligned(d.cast::<u32>(), value) };
            return Ok(());
        }
        // The Histogram effect's HistogramOutput (index 2): a read-only output the compute pass fills. Native
        // returns E_INVALIDARG until the effect has been drawn + flushed, and again after any
        // property/input change; only then does it hand back the FLOAT[NumBins] blob.
        if self.clsid == CLSID_D2D1Histogram && i == 2 {
            let Some(hist) = self.effect.histogram_cached() else {
                return Err(E_INVALIDARG.into());
            };
            let bytes: Vec<u8> = hist.iter().flat_map(|f| f.to_le_bytes()).collect();
            let n = (s as usize).min(bytes.len());
            unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), d, n) };
            return Ok(());
        }
        // A schema-only view (GetEffectProperties) reports the registration default, which can differ
        // from a created instance's value (Flood schema Color {0,0,0,0} vs instance {0,0,0,1}).
        if self.schema.get() {
            let prop = schema_default(&self.clsid, i).ok_or(windows_core::Error::from(E_NOTIMPL))?;
            let bytes = encode_prop(&prop);
            let n = (s as usize).min(bytes.len());
            unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), d, n) };
            return Ok(());
        }
        // Native hands back the property's DEFAULT when nothing was set (e.g. Flood's Color reads
        // {0,0,0,1} before any SetValue), not E_NOTIMPL.
        let prop = self.effect.stored_value(i).or_else(|| builtin_default(&self.clsid, i)).ok_or(windows_core::Error::from(E_NOTIMPL))?;
        let bytes = encode_prop(&prop);
        let n = (s as usize).min(bytes.len());
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), d, n) };
        Ok(())
    }
    fn GetValueSize(&self, i: u32) -> u32 {
        if self.clsid == CLSID_D2D1ColorManagement && matches!(i, 1 | 3 | 4 | 5) { return 4; }
        // HistogramOutput is a FLOAT[NumBins] blob (NumBins = property 0, default 256).
        if self.clsid == CLSID_D2D1Histogram && i == 2 {
            let num_bins = match self.effect.stored_value(0) {
                Some(Prop::U32(n)) => n.max(1),
                _ => 256,
            };
            return num_bins * 4;
        }
        if let Some(kind) = self.effect.custom_property_kind(i) {
            // A custom author property: fixed types report their declared size; a blob reports the
            // stored value's byte length (what GetValue would hand back), 0 until set. [G2]
            return match kind {
                PropKind::Blob => self.effect.stored_value(i).map(|p| encode_prop(&p).len() as u32).unwrap_or(0),
                k => prop_fixed_size(k),
            };
        }
        // The property's declared byte size (what native reports whether or not a value is set).
        builtin_props(&self.clsid).get(i as usize).map(|&(_, _, size)| size).unwrap_or(0)
    }
    fn GetSubProperties(&self, i: u32) -> WResult<ID2D1Properties> {
        // Direct2D's GetSubProperties: an out-of-range index is D2DERR_INVALID_PROPERTY
        // (0x88990029). Every valid author property has a system sub-property object exposing
        // DISPLAYNAME/ISREADONLY/DEFAULT (verified vs native on Flood's Color); return that.
        let count = self.effect.custom_property_count().unwrap_or_else(|| builtin_props(&self.clsid).len() as u32);
        if i >= count {
            return Err(windows_core::HRESULT(D2DERR_INVALID_PROPERTY).into());
        }
        let (name, ty, default) = if self.effect.custom_property_count().is_some() {
            let name = self.effect.custom_property_name(i).unwrap_or_default();
            let ty = self.effect.custom_property_kind(i).map(prop_type_of).unwrap_or(D2D1_PROPERTY_TYPE(0));
            (name, ty, self.effect.stored_value(i))
        } else {
            let (name, ty, _) = builtin_props(&self.clsid)[i as usize];
            (name.to_owned(), ty, builtin_default(&self.clsid, i))
        };
        let obj: ID2D1Properties = SubPropsObj {
            name: name.encode_utf16().chain(std::iter::once(0)).collect(),
            readonly: false,
            ty,
            default,
        }.into();
        Ok(obj)
    }
}

// D2DERR_INVALID_PROPERTY / D2DERR_NO_SUBPROPERTIES (D2D error facet 0x899).
const D2DERR_INVALID_PROPERTY: i32 = 0x8899_0029u32 as i32;
const D2DERR_NO_SUBPROPERTIES: i32 = 0x8899_002Au32 as i32;
// D2D1_SUBPROPERTY system indices native exposes on every property's sub-property object.
const SUBPROP_DISPLAYNAME: u32 = 0x8000_0000;
const SUBPROP_ISREADONLY: u32 = 0x8000_0001;
const SUBPROP_DEFAULT: u32 = 0x8000_0004;

/// A built-in author property's DEFAULT value (what `GetValue`/`GetSubProperties(DEFAULT)` hand back
/// before any `SetValue`). Mirrors the core's render-time defaults (`effects/mod.rs`). Only exposed
/// built-ins need a row.
fn builtin_default(clsid: &GUID, i: u32) -> Option<Prop> {
    match (clsid, i) {
        (c, 0) if *c == CLSID_D2D1Flood => Some(Prop::V4([0.0, 0.0, 0.0, 1.0])),
        _ => None,
    }
}

/// A built-in property's *registration-schema* DEFAULT (`ID2D1Factory1::GetEffectProperties`), which
/// can differ from the created-instance default: native's Flood Color schema default is {0,0,0,0}
/// (VM-verified), while a created Flood reads {0,0,0,1} ([`builtin_default`]).
fn schema_default(clsid: &GUID, i: u32) -> Option<Prop> {
    match (clsid, i) {
        (c, 0) if *c == CLSID_D2D1Flood => Some(Prop::V4([0.0, 0.0, 0.0, 0.0])),
        _ => None,
    }
}

/// The effect-specific (non-system) property schema a built-in exposes through `ID2D1Properties`:
/// `(name, D2D1_PROPERTY_TYPE, byte size)`, in index order. The core built-ins carry no name/type
/// table, so the facade supplies it. Only effects a scene reads are listed (ponytail: add a row when
/// a scene needs it); an unlisted effect reports no properties, exactly as its core has none.
fn builtin_props(clsid: &GUID) -> &'static [(&'static str, D2D1_PROPERTY_TYPE, u32)] {
    if *clsid == CLSID_D2D1Flood {
        &[("Color", D2D1_PROPERTY_TYPE_VECTOR4, 16)]
    } else {
        &[]
    }
}

/// A property's system sub-property object (`ID2D1Properties::GetSubProperties`). Native exposes
/// DISPLAYNAME (string), ISREADONLY (bool) and DEFAULT (the property's type); MIN/MAX/FIELDS/INDEX
/// return D2DERR_INVALID_PROPERTY. Verified byte-exact vs native on Flood's Color.
#[implement(ID2D1Properties)]
struct SubPropsObj {
    name: Vec<u16>, // NUL-terminated UTF-16 display name
    readonly: bool,
    ty: D2D1_PROPERTY_TYPE,
    default: Option<Prop>,
}
impl SubPropsObj {
    fn bytes(&self, index: u32) -> Option<Vec<u8>> {
        match index {
            SUBPROP_DISPLAYNAME => Some(self.name.iter().flat_map(|w| w.to_le_bytes()).collect()),
            SUBPROP_ISREADONLY => Some((self.readonly as u32).to_le_bytes().to_vec()),
            SUBPROP_DEFAULT => self.default.as_ref().map(encode_prop),
            _ => None,
        }
    }
}
impl ID2D1Properties_Impl for SubPropsObj_Impl {
    fn GetPropertyCount(&self) -> u32 { 0 }
    fn GetPropertyName(&self, _i: u32, _n: windows_core::PWSTR, _c: u32) -> WResult<()> { Err(E_INVALIDARG.into()) }
    fn GetPropertyNameLength(&self, _i: u32) -> u32 { 0 }
    fn GetType(&self, i: u32) -> D2D1_PROPERTY_TYPE {
        match i {
            SUBPROP_DISPLAYNAME => D2D1_PROPERTY_TYPE_STRING,
            SUBPROP_ISREADONLY => D2D1_PROPERTY_TYPE_BOOL,
            SUBPROP_DEFAULT => self.ty,
            _ => D2D1_PROPERTY_TYPE(0),
        }
    }
    fn GetPropertyIndex(&self, _n: &windows_core::PCWSTR) -> u32 { u32::MAX }
    fn SetValue(&self, _i: u32, _t: D2D1_PROPERTY_TYPE, _d: *const u8, _s: u32) -> WResult<()> { Err(E_INVALIDARG.into()) }
    fn SetValueByName(&self, _n: &windows_core::PCWSTR, _t: D2D1_PROPERTY_TYPE, _d: *const u8, _s: u32) -> WResult<()> { Err(E_INVALIDARG.into()) }
    fn GetValue(&self, i: u32, _t: D2D1_PROPERTY_TYPE, d: *mut u8, s: u32) -> WResult<()> {
        let bytes = self.bytes(i).ok_or(windows_core::Error::from(windows_core::HRESULT(D2DERR_INVALID_PROPERTY)))?;
        let n = (s as usize).min(bytes.len());
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), d, n) };
        Ok(())
    }
    fn GetValueByName(&self, _n: &windows_core::PCWSTR, _t: D2D1_PROPERTY_TYPE, _d: *mut u8, _s: u32) -> WResult<()> { Err(E_INVALIDARG.into()) }
    fn GetValueSize(&self, i: u32) -> u32 { self.bytes(i).map(|b| b.len() as u32).unwrap_or(0) }
    fn GetSubProperties(&self, _i: u32) -> WResult<ID2D1Properties> { Err(windows_core::HRESULT(D2DERR_NO_SUBPROPERTIES).into()) }
}

/// A registered custom effect's author-property schema (from the core [`Effect`]) drives the facade's
/// `ID2D1Properties`, the dynamic counterpart to [`builtin_props`]: `GetPropertyCount/Name/Index/Type`
/// answer from the schema so a caller (e.g. ComputeSharp resolving its `Buffer`) can address the
/// property at all. `None` for a built-in (which keeps the static table). [G2]
fn prop_type_of(k: PropKind) -> D2D1_PROPERTY_TYPE {
    match k {
        PropKind::F32 => D2D1_PROPERTY_TYPE_FLOAT,
        PropKind::U32 => D2D1_PROPERTY_TYPE_UINT32,
        PropKind::I32 => D2D1_PROPERTY_TYPE_INT32,
        PropKind::Bool => D2D1_PROPERTY_TYPE_BOOL,
        PropKind::V2 => D2D1_PROPERTY_TYPE_VECTOR2,
        PropKind::V3 => D2D1_PROPERTY_TYPE_VECTOR3,
        PropKind::V4 => D2D1_PROPERTY_TYPE_VECTOR4,
        PropKind::M32 => D2D1_PROPERTY_TYPE_MATRIX_3X2,
        PropKind::M44 => D2D1_PROPERTY_TYPE_MATRIX_4X4,
        PropKind::M54 => D2D1_PROPERTY_TYPE_MATRIX_5X4,
        PropKind::Blob => D2D1_PROPERTY_TYPE_BLOB,
        PropKind::IUnknown => D2D1_PROPERTY_TYPE_IUNKNOWN,
    }
}
/// A fixed-type property's declared byte size (a blob's size is the stored value's length instead).
fn prop_fixed_size(k: PropKind) -> u32 {
    match k {
        PropKind::F32 | PropKind::U32 | PropKind::I32 | PropKind::Bool => 4,
        PropKind::V2 => 8,
        PropKind::V3 => 12,
        PropKind::V4 => 16,
        PropKind::M32 => 24,
        PropKind::M44 => 64,
        PropKind::M54 => 80,
        PropKind::Blob => 0,
        PropKind::IUnknown => core::mem::size_of::<usize>() as u32, // an interface pointer
    }
}

/// Serialise a `Prop` back to its little-endian `GetValue` bytes (the inverse of [`decode_prop`]).
fn encode_prop(p: &Prop) -> Vec<u8> {
    let floats = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
    match p {
        Prop::F32(v) => v.to_le_bytes().to_vec(),
        Prop::U32(v) => v.to_le_bytes().to_vec(),
        Prop::I32(v) => v.to_le_bytes().to_vec(),
        Prop::Bool(v) => (*v as u32).to_le_bytes().to_vec(),
        Prop::V2(v) => floats(v),
        Prop::V3(v) => floats(v),
        Prop::V4(v) => floats(v),
        Prop::M32(v) => floats(v),
        Prop::M44(v) => floats(v),
        Prop::M54(v) => floats(v),
        Prop::Floats(v) => floats(v),
    }
}
/// An effect is also an `ID2D1Image` (its output), as native's: ComputeSharp.D2D1 queries a Flood for it
/// to reach the factory. It reports a stand-in factory, not its creator.
impl ID2D1Resource_Impl for EffectObj_Impl {
    fn GetFactory(&self, f: OutRef<ID2D1Factory>) {
        let _ = f.write(Some(super::factory::standin_factory()));
    }
}
impl ID2D1Image_Impl for EffectObj_Impl {}

impl EffectObj {
    /// The colour a bare Flood (no inputs) carries forward for the composite-aware blit.
    fn output_flood(&self) -> Option<[i32; 4]> {
        (*self.input_count.borrow() == 0).then(|| *self.flood.borrow()).flatten()
    }
}

impl ID2D1Effect_Impl for EffectObj_Impl {
    fn SetInputCount(&self, count: u32) -> WResult<()> {
        *self.input_count.borrow_mut() = count;
        Ok(())
    }
    fn SetInput(&self, index: u32, input: Ref<ID2D1Image>, _invalidate: windows_core::BOOL) {
        self.store_input(index, input.as_ref().cloned());
        // A command-list image has no edge: each draw renders it over the rect that draw needs, on the drawing
        // context (`DeviceContext::bake_lists`).
        if let Some(list) = input.as_ref().and_then(|i| i.cast_object_ref::<CommandListObj>().ok()) {
            self.effect.set_input(index, d2d::effects::Input::list(list.list.clone()));
            let mut c = self.input_count.borrow_mut();
            *c = (*c).max(index + 1);
            return;
        }
        // A nested effect is bound live: its own command-list inputs, even ones set after this bind, are part of
        // the graph every draw renders (G3/G16).
        if let Some(inp) = effect_input(input) {
            self.effect.set_input(index, inp);
            let mut c = self.input_count.borrow_mut();
            *c = (*c).max(index + 1);
        }
    }
    fn GetInput(&self, index: u32, input: OutRef<ID2D1Image>) {
        let _ = input.write(self.inputs.borrow().get(index as usize).and_then(|i| i.clone()));
    }
    fn GetInputCount(&self) -> u32 {
        *self.input_count.borrow()
    }
    fn GetOutput(&self, output: OutRef<ID2D1Image>) {
        // Only a bare Flood (no inputs) carries its colour forward for the composite-aware blit.
        // A ColorManagement with a SIMPLE-profile context pair: native fails realisation; drawing the
        // output latches E_INVALIDARG and writes no pixels (cms2-04/05).
        if self.cms_fail.get() {
            let img: ID2D1Image = CmFailObj.into();
            let _ = output.write(Some(img));
            return;
        }
        let flood = self.output_flood();
        let img: ID2D1Image = ImageObj { effect: self.effect.clone(), flood, inputs: self.inputs.clone(), factory: None }.into();
        let _ = output.write(Some(img));
    }
}

// --- effect-output image (`GetOutput`) ---------------------------------------------------------

#[implement(ID2D1Image)]
pub(crate) struct ImageObj {
    pub(crate) effect: Effect,
    pub(crate) flood: Option<[i32; 4]>,
    /// The drawn effect's bound inputs (shared, live). Walked at draw time to realise command-list
    /// inputs on *nested* effects too (G3), picking up lists set after the nested effect was bound (G16).
    pub(crate) inputs: Rc<RefCell<Vec<Option<ID2D1Image>>>>,
    // ponytail: effect-output image; carries the factory once EffectObj does, else None (null).
    pub(crate) factory: Option<ID2D1Factory>,
}
impl ID2D1Resource_Impl for ImageObj_Impl {
    fn GetFactory(&self, f: OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
impl ID2D1Image_Impl for ImageObj_Impl {}

/// The output of a ColorManagement effect whose context pair native cannot realise (a SIMPLE-profile
/// context): drawing it latches E_INVALIDARG at EndDraw and writes no pixels (cms2-04/05). The device
/// context recognises this image in its draw path.
#[implement(ID2D1Image)]
pub(crate) struct CmFailObj;
impl ID2D1Resource_Impl for CmFailObj_Impl {
    // Synthetic failure-placeholder image (never user-created), so it has no factory to report.
    fn GetFactory(&self, f: OutRef<ID2D1Factory>) {
        let _ = f.write(None);
    }
}
impl ID2D1Image_Impl for CmFailObj_Impl {}

/// What an `ID2D1Image` should draw as — a bitmap blit, an effect-graph evaluation, or (for a bare
/// `Flood`) the composite-aware `draw_flood` keyed by its BGRA colour.
pub(crate) enum DrawTarget {
    Bitmap(Bitmap),
    Effect(Effect, Option<[i32; 4]>),
}
pub(crate) fn image_target(img: Ref<ID2D1Image>) -> Option<DrawTarget> {
    let img = img.as_ref()?;
    if let Ok(o) = img.cast_object_ref::<BitmapObj>() {
        // A bitmap used as a draw source or effect input: its CPU source if any, else the rendered
        // pixels of a render-target bitmap (D2D1_BITMAP_OPTIONS_TARGET, no CPU source) — native accepts
        // a target bitmap as an effect input / DrawImage source; `src_clone` alone dropped it (G4).
        return o.readable_src().map(DrawTarget::Bitmap);
    }
    if let Ok(o) = img.cast_object_ref::<ImageSrcObj>() {
        return Some(DrawTarget::Bitmap(o.source.bitmap.clone()));
    }
    if let Ok(o) = img.cast_object_ref::<ImageObj>() {
        return Some(DrawTarget::Effect(o.effect.clone(), o.flood));
    }
    if let Ok(o) = img.cast_object_ref::<EffectObj>() {
        return Some(DrawTarget::Effect(o.effect.clone(), o.output_flood())); // its output, as native's
    }
    None
}
fn effect_input(img: Ref<ID2D1Image>) -> Option<Input> {
    match image_target(img)? {
        DrawTarget::Bitmap(b) => Some(Input::Bitmap(Rc::new(b))),
        DrawTarget::Effect(e, _) => Some(Input::Effect(e)),
    }
}

// --- WIC / transformed image source (one type, two sibling interfaces) -------------------------

#[implement(ID2D1ImageSourceFromWic, ID2D1TransformedImageSource)]
pub(crate) struct ImageSrcObj {
    pub(crate) source: ImageSource,
    pub(crate) factory: Option<ID2D1Factory>,
    /// The WIC source this was created from (`ID2D1ImageSourceFromWic::GetSource`), if any.
    pub(crate) wic_source: Option<IWICBitmapSource>,
    /// The image source a transformed source wraps (`ID2D1TransformedImageSource::GetSource`), if any.
    pub(crate) image_source: Option<ID2D1ImageSource>,
    /// The transform properties a transformed source was created with (`GetProperties`).
    pub(crate) props: Option<D2D1_TRANSFORMED_IMAGE_SOURCE_PROPERTIES>,
}
/// Read the already-premultiplied BGRA of a WIC source into a core `Bitmap` (the harness feeds a raw
/// `32bppPBGRA` buffer, so `CopyPixels` returns exactly those bytes).
pub(crate) fn bitmap_from_wic(src: &IWICBitmapSource) -> WResult<Bitmap> {
    let (mut w, mut h) = (0u32, 0u32);
    unsafe { src.GetSize(&mut w, &mut h)? };
    let mut premul = vec![0u8; (w * h * 4) as usize];
    unsafe { src.CopyPixels(core::ptr::null(), w * 4, &mut premul)? };
    Ok(Bitmap {
        width: w,
        height: h,
        premul_bgra: premul,
        format: d2d::abi::DXGI_FORMAT_B8G8R8A8_UNORM,
        alpha: d2d::abi::D2D1_ALPHA_MODE_PREMULTIPLIED,
        dpi: (96.0, 96.0),
    })
}

/// Copy an `IWICBitmapLock`'s locked pixels (honouring its stride) into a core `Bitmap`. This is the
/// only WIC resource `ID2D1RenderTarget::CreateSharedBitmap` accepts (alongside `ID2D1Bitmap` and, on
/// Windows, `IDXGISurface`) — a bare `IWICBitmap`/`IWICBitmapSource` is rejected (E_INVALIDARG).
pub(crate) fn bitmap_from_wic_lock(lock: &windows::Win32::Graphics::Imaging::IWICBitmapLock) -> WResult<Bitmap> {
    let (mut w, mut h) = (0u32, 0u32);
    unsafe { lock.GetSize(&mut w, &mut h)? };
    let stride = unsafe { lock.GetStride()? } as usize;
    let (mut size, mut ptr) = (0u32, core::ptr::null_mut::<u8>());
    unsafe { lock.GetDataPointer(&mut size, &mut ptr)? };
    let data = unsafe { core::slice::from_raw_parts(ptr, size as usize) };
    let row = (w * 4) as usize;
    let mut premul = vec![0u8; row * h as usize];
    for y in 0..h as usize {
        premul[y * row..(y + 1) * row].copy_from_slice(&data[y * stride..y * stride + row]);
    }
    Ok(Bitmap {
        width: w,
        height: h,
        premul_bgra: premul,
        format: d2d::abi::DXGI_FORMAT_B8G8R8A8_UNORM,
        alpha: d2d::abi::D2D1_ALPHA_MODE_PREMULTIPLIED,
        dpi: (96.0, 96.0),
    })
}
impl ID2D1Resource_Impl for ImageSrcObj_Impl {
    fn GetFactory(&self, f: OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
impl ID2D1Image_Impl for ImageSrcObj_Impl {}
impl ID2D1ImageSource_Impl for ImageSrcObj_Impl {
    fn OfferResources(&self) -> WResult<()> {
        Ok(())
    }
    fn TryReclaimResources(&self) -> WResult<windows_core::BOOL> {
        // `resourcesDiscarded`: the CPU core never discards an offered source, so nothing was lost —
        // native returns FALSE here (VM-verified on d2d1 10.0.26100.8875).
        Ok(false.into())
    }
}
impl ID2D1ImageSourceFromWic_Impl for ImageSrcObj_Impl {
    // A WIC-backed image source does not support explicit caching; native returns
    // D2DERR_UNSUPPORTED_OPERATION (VM-verified on d2d1 10.0.26100.8875), not S_OK.
    fn EnsureCached(&self, _r: *const D2D_RECT_U) -> WResult<()> {
        Err(windows_core::Error::from(windows_core::HRESULT(0x8899_0003u32 as i32)))
    }
    fn TrimCache(&self, _r: *const D2D_RECT_U) -> WResult<()> {
        Err(windows_core::Error::from(windows_core::HRESULT(0x8899_0003u32 as i32)))
    }
    // The WIC source this image source was created from (NULL if not WIC-backed), as native returns.
    fn GetSource(&self, s: OutRef<IWICBitmapSource>) {
        let _ = s.write(self.wic_source.clone());
    }
}
impl ID2D1TransformedImageSource_Impl for ImageSrcObj_Impl {
    // The image source this transformed source wraps, as native returns.
    fn GetSource(&self, s: OutRef<ID2D1ImageSource>) {
        let _ = s.write(self.image_source.clone());
    }
    fn GetProperties(&self, p: *mut D2D1_TRANSFORMED_IMAGE_SOURCE_PROPERTIES) {
        // The properties it was created with, else a valid identity default (never leave unset).
        if !p.is_null() {
            unsafe {
                *p = self.props.unwrap_or(D2D1_TRANSFORMED_IMAGE_SOURCE_PROPERTIES {
                    orientation: D2D1_ORIENTATION_DEFAULT,
                    scaleX: 1.0,
                    scaleY: 1.0,
                    interpolationMode: D2D1_INTERPOLATION_MODE_LINEAR,
                    options: D2D1_TRANSFORMED_IMAGE_SOURCE_OPTIONS_NONE,
                });
            }
        }
    }
}

// --- sprite batch ------------------------------------------------------------------------------

#[implement(ID2D1SpriteBatch)]
pub(crate) struct SpriteBatchObj {
    pub(crate) sprites: RefCell<Vec<Sprite>>,
    pub(crate) factory: Option<ID2D1Factory>,
}
impl ID2D1Resource_Impl for SpriteBatchObj_Impl {
    fn GetFactory(&self, f: OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
/// Read `count` sprites from the parallel strided arrays `AddSprites`/`SetSprites` pass.
#[allow(clippy::too_many_arguments)]
unsafe fn read_sprites(
    count: u32,
    dest: *const D2D_RECT_F,
    src: *const D2D_RECT_U,
    col: *const D2D1_COLOR_F,
    xf: *const windows_numerics::Matrix3x2,
    dstride: u32,
    sstride: u32,
    cstride: u32,
    tstride: u32,
) -> Vec<Sprite> {
    let at = |base: *const u8, stride: u32, i: u32| unsafe { base.add((stride as usize).max(1).wrapping_mul(i as usize)) };
    (0..count)
        .map(|i| {
            let d = unsafe { &*(at(dest.cast(), dstride, i) as *const D2D_RECT_F) };
            let source = (!src.is_null()).then(|| {
                let r = unsafe { &*(at(src.cast(), sstride, i) as *const D2D_RECT_U) };
                [r.left as f32, r.top as f32, r.right as f32, r.bottom as f32]
            });
            let color = (!col.is_null()).then(|| {
                let c = unsafe { &*(at(col.cast(), cstride, i) as *const D2D1_COLOR_F) };
                [c.r, c.g, c.b, c.a]
            });
            let transform = (!xf.is_null()).then(|| {
                let m = unsafe { &*(at(xf.cast(), tstride, i) as *const windows_numerics::Matrix3x2) };
                [m.M11, m.M12, m.M21, m.M22, m.M31, m.M32]
            });
            Sprite { dest: [d.left, d.top, d.right, d.bottom], source, color, transform }
        })
        .collect()
}
impl ID2D1SpriteBatch_Impl for SpriteBatchObj_Impl {
    fn AddSprites(&self, count: u32, dest: *const D2D_RECT_F, src: *const D2D_RECT_U, col: *const D2D1_COLOR_F, xf: *const windows_numerics::Matrix3x2, ds: u32, ss: u32, cs: u32, ts: u32) -> WResult<()> {
        self.sprites.borrow_mut().extend(unsafe { read_sprites(count, dest, src, col, xf, ds, ss, cs, ts) });
        Ok(())
    }
    fn SetSprites(&self, start: u32, count: u32, dest: *const D2D_RECT_F, src: *const D2D_RECT_U, col: *const D2D1_COLOR_F, xf: *const windows_numerics::Matrix3x2, ds: u32, ss: u32, cs: u32, ts: u32) -> WResult<()> {
        // Native updates `count` sprites in place starting at `start`; the rest of the batch is
        // untouched. (The sprites being updated must already be in the batch.)
        let mut v = self.sprites.borrow_mut();
        if start as usize + count as usize > v.len() {
            return Err(E_INVALIDARG.into());
        }
        for (i, s) in unsafe { read_sprites(count, dest, src, col, xf, ds, ss, cs, ts) }.into_iter().enumerate() {
            v[start as usize + i] = s;
        }
        Ok(())
    }
    fn GetSprites(&self, start: u32, count: u32, d: *mut D2D_RECT_F, sr: *mut D2D_RECT_U, co: *mut D2D1_COLOR_F, t: *mut windows_numerics::Matrix3x2) -> WResult<()> {
        let sprites = self.sprites.borrow();
        if start as usize + count as usize > sprites.len() {
            return Err(windows_core::Error::from(E_NOTIMPL));
        }
        // Tightly-packed output arrays (any may be null). A sprite with no transform reports identity.
        for i in 0..count as usize {
            let s = &sprites[start as usize + i];
            if !d.is_null() {
                unsafe { *d.add(i) = D2D_RECT_F { left: s.dest[0], top: s.dest[1], right: s.dest[2], bottom: s.dest[3] } };
            }
            if !sr.is_null() {
                let r = s.source.unwrap_or([0.0; 4]);
                unsafe { *sr.add(i) = D2D_RECT_U { left: r[0] as u32, top: r[1] as u32, right: r[2] as u32, bottom: r[3] as u32 } };
            }
            if !co.is_null() {
                let c = s.color.unwrap_or([1.0, 1.0, 1.0, 1.0]);
                unsafe { *co.add(i) = D2D1_COLOR_F { r: c[0], g: c[1], b: c[2], a: c[3] } };
            }
            if !t.is_null() {
                let m = s.transform.unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
                unsafe { *t.add(i) = windows_numerics::Matrix3x2 { M11: m[0], M12: m[1], M21: m[2], M22: m[3], M31: m[4], M32: m[5] } };
            }
        }
        Ok(())
    }
    fn GetSpriteCount(&self) -> u32 {
        self.sprites.borrow().len() as u32
    }
    fn Clear(&self) {
        self.sprites.borrow_mut().clear();
    }
}

// --- command list (an ID2D1Image you record into and replay) -----------------------------------

#[implement(ID2D1CommandList)]
pub(crate) struct CommandListObj {
    pub(crate) list: CommandList,
    pub(crate) factory: Option<ID2D1Factory>,
    /// A draw error that was already latched on the creating DC when this list was created. Native does
    /// not surface such an error at CreateCommandList (it records the list, then reports it from Close);
    /// we capture it here and return it from `Close` so the recording runs and the tags match native.
    pub(crate) pending_err: Option<windows_core::HRESULT>,
}
impl ID2D1Resource_Impl for CommandListObj_Impl {
    fn GetFactory(&self, f: OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
impl ID2D1Image_Impl for CommandListObj_Impl {}
impl ID2D1CommandList_Impl for CommandListObj_Impl {
    fn Stream(&self, sink: Ref<ID2D1CommandSink>) -> WResult<()> {
        let sink = sink.as_ref().ok_or(windows_core::Error::from(E_NOTIMPL))?;
        let mut driver = SinkDriver { sink };
        self.list.stream(&mut driver);
        Ok(())
    }
    fn Close(&self) -> WResult<()> {
        self.list.close();
        // Native reports from Close (not from CreateCommandList) a draw error latched while this list
        // was the target, after the list has recorded its ops: one latched before creation
        // (`pending_err`), or one a draw latched while recording (the list's `rec_error`, API-observable
        // — a nested DrawImage with an out-of-range composite mode returns E_INVALIDARG here).
        if let Some(e) = self.pending_err.or_else(|| self.list.rec_error()) {
            return Err(e.into());
        }
        // A clip or layer pushed while recording and never popped fails the Close (clclip probes).
        if self.list.unbalanced() {
            return Err(windows_core::HRESULT(0x8899_0016u32 as i32).into());
        }
        Ok(())
    }
}

/// Drives a caller's `ID2D1CommandSink` from the core's device-resolved replay (see
/// `d2d::command_stream`): identity transform, a synthesised solid brush per fill, the device-space
/// geometry. Errors from the sink are ignored (the stream is best-effort, as native's is).
struct SinkDriver<'a> {
    sink: &'a ID2D1CommandSink,
}
impl SinkDriver<'_> {
    fn solid(&self, brush: SolidColorBrush) -> Option<ID2D1Brush> {
        let scb: ID2D1SolidColorBrush = SolidBrushObj::from_core(brush).into();
        scb.cast().ok()
    }
}
impl StreamSink for SinkDriver<'_> {
    fn begin(&mut self) {
        unsafe { let _ = self.sink.BeginDraw(); }
    }
    fn clear(&mut self, color: [f32; 4]) {
        let c = D2D1_COLOR_F { r: color[0], g: color[1], b: color[2], a: color[3] };
        unsafe { let _ = self.sink.Clear(Some(&c)); }
    }
    fn fill_rect(&mut self, rect: [f32; 4], brush: SolidColorBrush, aa: u32, blend: u32) {
        let Some(b) = self.solid(brush) else { return };
        let r = D2D_RECT_F { left: rect[0], top: rect[1], right: rect[2], bottom: rect[3] };
        unsafe {
            let _ = self.sink.SetAntialiasMode(D2D1_ANTIALIAS_MODE(aa as i32));
            let _ = self.sink.SetPrimitiveBlend(D2D1_PRIMITIVE_BLEND(blend as i32));
            let _ = self.sink.SetTransform(&IDENTITY_3X2);
            let _ = self.sink.FillRectangle(&r, &b);
        }
    }
    fn fill_polys(&mut self, figures: &[Vec<[f32; 2]>], winding: bool, brush: SolidColorBrush, aa: u32, blend: u32) {
        let Some(b) = self.solid(brush) else { return };
        let figs: Vec<Figure> = figures
            .iter()
            .filter(|f| !f.is_empty())
            .map(|f| Figure {
                start: f[0],
                filled: true,
                closed: true,
                segments: f[1..].iter().map(|&p| Segment::Line(p)).collect(),
                seg_flags: Vec::new(),
            })
            .collect();
        let geom = Geometry::Path { fill: if winding { FillMode::Winding } else { FillMode::Alternate }, figures: figs };
        let pg: ID2D1PathGeometry = GeometryObj::shape(geom).into();
        let Ok(g) = pg.cast::<ID2D1Geometry>() else { return };
        unsafe {
            let _ = self.sink.SetAntialiasMode(D2D1_ANTIALIAS_MODE(aa as i32));
            let _ = self.sink.SetPrimitiveBlend(D2D1_PRIMITIVE_BLEND(blend as i32));
            let _ = self.sink.SetTransform(&IDENTITY_3X2);
            let _ = self.sink.FillGeometry(&g, &b, None);
        }
    }
    fn push_layer(&mut self, bounds: [f32; 4], opacity: f32, _init: bool) {
        let p = D2D1_LAYER_PARAMETERS1 {
            contentBounds: D2D_RECT_F { left: bounds[0], top: bounds[1], right: bounds[2], bottom: bounds[3] },
            opacity,
            ..Default::default()
        };
        unsafe { let _ = self.sink.PushLayer(&p, None); }
    }
    fn pop_layer(&mut self) {
        unsafe { let _ = self.sink.PopLayer(); }
    }
    fn push_clip(&mut self, rect: [f32; 4], antialias: u32) {
        let r = D2D_RECT_F { left: rect[0], top: rect[1], right: rect[2], bottom: rect[3] };
        unsafe { let _ = self.sink.PushAxisAlignedClip(&r, D2D1_ANTIALIAS_MODE(antialias as i32)); }
    }
    fn pop_clip(&mut self) {
        unsafe { let _ = self.sink.PopAxisAlignedClip(); }
    }
    fn end(&mut self) {
        unsafe { let _ = self.sink.EndDraw(); }
    }
}

/// Identity 3×2, for the device-space (baked-transform) replay.
const IDENTITY_3X2: windows_numerics::Matrix3x2 = windows_numerics::Matrix3x2 { M11: 1.0, M12: 0.0, M21: 0.0, M22: 1.0, M31: 0.0, M32: 0.0 };

// --- gradient mesh -----------------------------------------------------------------------------

#[implement(ID2D1GradientMesh)]
pub(crate) struct GradientMeshObj {
    pub(crate) mesh: GradientMesh,
    pub(crate) factory: Option<ID2D1Factory>,
}
impl ID2D1Resource_Impl for GradientMeshObj_Impl {
    fn GetFactory(&self, f: OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
impl ID2D1GradientMesh_Impl for GradientMeshObj_Impl {
    fn GetPatchCount(&self) -> u32 {
        self.mesh.patches.len() as u32
    }
    fn GetPatches(&self, start: u32, p: *mut D2D1_GRADIENT_MESH_PATCH, count: u32) -> WResult<()> {
        let patches = &self.mesh.patches;
        if start as usize + count as usize > patches.len() {
            return Err(windows_core::Error::from(E_NOTIMPL));
        }
        for i in 0..count as usize {
            unsafe { *p.add(i) = d2d_patch(&patches[start as usize + i]) };
        }
        Ok(())
    }
}

/// A core `MeshPatch` → `D2D1_GRADIENT_MESH_PATCH` (the inverse of `context::mesh_patch`): the 16
/// row-major control points, the four stored corner colours (`color00,03,30,33`) and the per-edge
/// modes `[left, top, right, bottom]`.
fn d2d_patch(m: &d2d::extras::mesh::MeshPatch) -> D2D1_GRADIENT_MESH_PATCH {
    let v = |p: [f32; 2]| windows_numerics::Vector2 { X: p[0], Y: p[1] };
    let c = |x: [f32; 4]| D2D1_COLOR_F { r: x[0], g: x[1], b: x[2], a: x[3] };
    let pt = &m.points;
    D2D1_GRADIENT_MESH_PATCH {
        point00: v(pt[0]), point01: v(pt[1]), point02: v(pt[2]), point03: v(pt[3]),
        point10: v(pt[4]), point11: v(pt[5]), point12: v(pt[6]), point13: v(pt[7]),
        point20: v(pt[8]), point21: v(pt[9]), point22: v(pt[10]), point23: v(pt[11]),
        point30: v(pt[12]), point31: v(pt[13]), point32: v(pt[14]), point33: v(pt[15]),
        color00: c(m.colors[0]), color03: c(m.colors[1]), color30: c(m.colors[2]), color33: c(m.colors[3]),
        leftEdgeMode: D2D1_PATCH_EDGE_MODE(m.extend[0] as i32),
        topEdgeMode: D2D1_PATCH_EDGE_MODE(m.extend[1] as i32),
        rightEdgeMode: D2D1_PATCH_EDGE_MODE(m.extend[2] as i32),
        bottomEdgeMode: D2D1_PATCH_EDGE_MODE(m.extend[3] as i32),
    }
}

// --- colour context (`ID2D1ColorContext`/`ID2D1ColorContext1`) --------------------------------

/// Build a core colour context from an `IWICColorContext` (`CreateColorContextFromWicColorContext` on
/// both the device and effect contexts), mirroring Direct2D's own conversion:
/// a Profile context reads its ICC bytes; an Exif context maps any value other than 2 to sRGB; an
/// Uninitialized context → `E_INVALIDARG`. Exif value 2 (Adobe RGB) — which native synthesises as a
/// 480-byte ICC profile — is `E_NOTIMPL` (that profile is not reproduced here).
pub(crate) fn color_context_from_wic(wic: &windows::Win32::Graphics::Imaging::IWICColorContext) -> windows_core::Result<ColorContext> {
    match unsafe { wic.GetType()? }.0 {
        1 => {
            // WICColorContextProfile: read the ICC bytes (query the size, then fill).
            let mut size = 0u32;
            unsafe { wic.GetProfileBytes(&mut [], &mut size)? };
            let mut buf = vec![0u8; size as usize];
            unsafe { wic.GetProfileBytes(&mut buf, &mut size)? };
            Ok(ColorContext::from_profile_bytes(&buf))
        }
        2 if unsafe { wic.GetExifColorSpace()? } == 2 => Ok(ColorContext::from_profile_bytes(&adobe_rgb_icc())), // Exif Adobe RGB
        2 => Ok(ColorContext::from_color_space(d2d::abi::D2D1_COLOR_SPACE(1))),        // Exif → sRGB
        _ => Err(windows_core::Error::from(windows_core::HRESULT(0x8007_0057u32 as i32))), // Uninitialized → E_INVALIDARG
    }
}

/// Build the 480-byte Adobe RGB ICC profile D2D synthesises for an Exif ColorSpace=2 context — a
/// fresh port of Direct2D's Adobe RGB profile builder, read field-by-field: a v2 `mntr`/`RGB `/`XYZ `
/// display profile, D50 PCS illuminant, D65 media white, three
/// `curv` TRCs at gamma 0x0233 (≈2.2), the D50-adapted Adobe RGB colorants, and `opRGB` text tags.
fn adobe_rgb_icc() -> Vec<u8> {
    let mut p = vec![0u8; 480];
    let be32 = |p: &mut [u8], o: usize, v: u32| p[o..o + 4].copy_from_slice(&v.to_be_bytes());
    // Header.
    be32(&mut p, 0, 480); // profile size
    be32(&mut p, 8, 0x0200_0000); // version 2.0
    p[12..24].copy_from_slice(b"mntrRGB XYZ ");
    be32(&mut p, 24, 0x07d0_0001); // created 2000-01
    be32(&mut p, 28, 0x0001_0000); // -01 00:00
    p[36..44].copy_from_slice(b"acspMSFT");
    be32(&mut p, 68, 0x0000_f6d6); // PCS illuminant = D50 (0.9642, 1.0, 0.8249)
    be32(&mut p, 72, 0x0001_0000);
    be32(&mut p, 76, 0x0000_d32d);
    // Tag table (sig, offset, size).
    be32(&mut p, 128, 9);
    let tags: [(&[u8; 4], u32, u32); 9] = [(b"wtpt", 240, 20), (b"rTRC", 260, 14), (b"gTRC", 276, 14), (b"bTRC", 292, 14), (b"rXYZ", 308, 20), (b"gXYZ", 328, 20), (b"bXYZ", 348, 20), (b"cprt", 368, 9), (b"desc", 384, 96)];
    for (i, (sig, off, sz)) in tags.iter().enumerate() {
        let e = 132 + i * 12;
        p[e..e + 4].copy_from_slice(*sig);
        be32(&mut p, e + 4, *off);
        be32(&mut p, e + 8, *sz);
    }
    // wtpt: D65 media white.
    p[240..244].copy_from_slice(b"XYZ ");
    for (i, v) in [0x0000_f351u32, 0x0001_0000, 0x0001_16cc].iter().enumerate() {
        be32(&mut p, 248 + i * 4, *v);
    }
    // r/g/b TRC: a one-entry `curv` = gamma 0x0233 (u8Fixed8 ≈ 2.199).
    for off in [260usize, 276, 292] {
        p[off..off + 4].copy_from_slice(b"curv");
        be32(&mut p, off + 8, 1);
        p[off + 12..off + 14].copy_from_slice(&0x0233u16.to_be_bytes());
    }
    // r/g/b colorants: D50-adapted Adobe RGB, s15Fixed16.
    for (tag, xyz) in [(308usize, [0x9c18u32, 0x4fa5, 0x04fc]), (328, [0x348d, 0xa02c, 0x0f95]), (348, [0x2631, 0x102f, 0xbe9c])] {
        p[tag..tag + 4].copy_from_slice(b"XYZ ");
        for (i, v) in xyz.iter().enumerate() {
            be32(&mut p, tag + 8 + i * 4, *v);
        }
    }
    // cprt (text) + desc ("opRGB").
    p[368..372].copy_from_slice(b"text");
    p[384..388].copy_from_slice(b"desc");
    be32(&mut p, 392, 6);
    p[396..401].copy_from_slice(b"opRGB");
    p
}

#[implement(ID2D1ColorContext1)]
pub(crate) struct ColorContextObj {
    pub(crate) ctx: ColorContext,
    pub(crate) factory: Option<ID2D1Factory>,
}
impl ID2D1Resource_Impl for ColorContextObj_Impl {
    fn GetFactory(&self, f: OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
impl ID2D1ColorContext_Impl for ColorContextObj_Impl {
    fn GetColorSpace(&self) -> D2D1_COLOR_SPACE {
        use d2d::color_context::Space;
        D2D1_COLOR_SPACE(match self.ctx.space() {
            Space::ScRgb => 2,
            Space::Srgb => 1,
            Space::Custom => 0,
        })
    }
    fn GetProfileSize(&self) -> u32 {
        self.ctx.get_profile().len() as u32
    }
    fn GetProfile(&self, profile: *mut u8, size: u32) -> WResult<()> {
        let p = self.ctx.get_profile();
        let n = (size as usize).min(p.len());
        unsafe { core::ptr::copy_nonoverlapping(p.as_ptr(), profile, n) };
        Ok(())
    }
}
impl ID2D1ColorContext1_Impl for ColorContextObj_Impl {
    fn GetColorContextType(&self) -> D2D1_COLOR_CONTEXT_TYPE {
        D2D1_COLOR_CONTEXT_TYPE(self.ctx.color_context_type().0)
    }
    fn GetDXGIColorSpace(&self) -> DXGI_COLOR_SPACE_TYPE {
        DXGI_COLOR_SPACE_TYPE(self.ctx.dxgi_color_space().0)
    }
    fn GetSimpleColorProfile(&self, p: *mut D2D1_SIMPLE_COLOR_PROFILE) -> WResult<()> {
        if let Some(s) = self.ctx.simple_color_profile() {
            unsafe {
                *p = D2D1_SIMPLE_COLOR_PROFILE {
                    redPrimary: windows_numerics::Vector2 { X: s.red[0], Y: s.red[1] },
                    greenPrimary: windows_numerics::Vector2 { X: s.green[0], Y: s.green[1] },
                    bluePrimary: windows_numerics::Vector2 { X: s.blue[0], Y: s.blue[1] },
                    whitePointXZ: windows_numerics::Vector2 { X: s.white[0], Y: s.white[1] },
                    gamma: D2D1_GAMMA1(s.gamma as i32),
                }
            };
        }
        Ok(())
    }
}

// --- ColorManagement colour-context property helper ------------------------------------------

/// Recover the `ColorContextObj` behind a `D2D1_PROPERTY_TYPE_COLOR_CONTEXT` property (the data is the
/// interface pointer's bytes, as the harness passes it). Used by `EffectObj`'s ColorManagement branch.
fn color_context_of(data: *const u8) -> Option<ComObject<ColorContextObj>> {
    let raw = unsafe { *(data as *const *mut core::ffi::c_void) };
    unsafe { IUnknown::from_raw_borrowed(&raw) }.and_then(|u| u.cast_object::<ColorContextObj>().ok())
}

// --- 3D lookup table (`CreateLookupTable3D`; an opaque resource bound to a LookupTable3D effect) --

#[implement(ID2D1LookupTable3D)]
pub(crate) struct LutObj {
    pub(crate) lut: Rc<Lut3d>,
    pub(crate) factory: Option<ID2D1Factory>,
}
impl ID2D1Resource_Impl for LutObj_Impl {
    fn GetFactory(&self, f: OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
impl ID2D1LookupTable3D_Impl for LutObj_Impl {}
