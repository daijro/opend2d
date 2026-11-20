//! The SVG DOM COM surface (`ID2D1SvgDocument`, `ID2D1SvgElement`, `ID2D1SvgAttribute` and the four
//! value objects `ID2D1SvgPaint`/`ID2D1SvgPathData`/`ID2D1SvgPointCollection`/
//! `ID2D1SvgStrokeDashArray`), forwarding to `d2d::svg`. Every element/attribute object shares the
//! one `Rc<RefCell<Document>>`; an element names its node by the document's stable id, so a handle
//! survives re-parenting exactly as a live COM element does. The DOM semantics (valid children,
//! type-mismatch/`E_INVALIDARG`, not-found) live in the core so this and the conformance replay
//! forward to one source of truth.

use super::context::read_stream;
use super::E_NOTIMPL;
use d2d::svg::attr::{Kind, PathCmd};
use d2d::svg::{Document, PaintType, Units, Value, D2DERR_INVALID_CALL, E_INVALIDARG};
use std::cell::RefCell;
use std::rc::Rc;
use windows::Win32::Graphics::Direct2D::Common::{D2D1_COLOR_F, D2D1_FILL_MODE, D2D_SIZE_F};
use windows::Win32::Graphics::Direct2D::*;
use windows::Win32::System::Com::IStream;
use windows_core::{implement, BOOL, Interface, OutRef, Ref, Result as WResult, GUID, HRESULT, PCWSTR, PWSTR};
use windows_numerics::Vector2;

/// `D2DERR_INSUFFICIENT_BUFFER` = `HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER)` — a too-small
/// caller buffer in the string/array getters.
const D2DERR_INSUFFICIENT_BUFFER: i32 = -2147024774i32; // 0x8007007A

fn err(code: i32) -> windows_core::Error {
    HRESULT(code).into()
}

// --- small string helpers ---------------------------------------------------------------------

fn pcwstr(s: &PCWSTR) -> String {
    if s.is_null() {
        String::new()
    } else {
        unsafe { s.to_string() }.unwrap_or_default()
    }
}
/// Read exactly `n` UTF-16 units (the `SetTextValue(value, valueCount)` convention).
fn pcwstr_n(s: &PCWSTR, n: u32) -> String {
    if s.is_null() {
        return String::new();
    }
    let units = unsafe { core::slice::from_raw_parts(s.0, n as usize) };
    String::from_utf16_lossy(units)
}
fn utf16_len(s: &str) -> u32 {
    s.encode_utf16().count() as u32
}
/// Write `s` (plus a null) into a caller buffer of `cap` UTF-16 units.
fn write_pwstr(dst: PWSTR, cap: u32, s: &str) -> WResult<()> {
    let utf16: Vec<u16> = s.encode_utf16().collect();
    if (utf16.len() + 1) as u32 > cap {
        return Err(err(D2DERR_INSUFFICIENT_BUFFER));
    }
    unsafe {
        for (i, &u) in utf16.iter().enumerate() {
            dst.0.add(i).write(u);
        }
        dst.0.add(utf16.len()).write(0);
    }
    Ok(())
}

// --- POD attribute (D2D1_SVG_ATTRIBUTE_POD_TYPE) ⇄ Value --------------------------------------

/// The declared [`Kind`] and byte size for a POD attribute type, or `None` if the enum is unknown.
fn pod_kind(t: D2D1_SVG_ATTRIBUTE_POD_TYPE) -> Option<(Kind, usize)> {
    Some(match t.0 {
        0 => (Kind::Float, 4),
        1 => (Kind::Color, 16),
        2 => (Kind::FillRule, 4),
        3 => (Kind::Display, 4),
        4 => (Kind::Overflow, 4),
        5 => (Kind::LineCap, 4),
        6 => (Kind::LineJoin, 4),
        7 => (Kind::Visibility, 4),
        8 => (Kind::Matrix, 24),
        9 => (Kind::UnitType, 4),
        10 => (Kind::ExtendMode, 4),
        11 => (Kind::AspectRatio, 12),
        12 => (Kind::ViewBox, 16),
        13 => (Kind::Length, 8),
        _ => return None,
    })
}
unsafe fn rf(p: *const core::ffi::c_void, off: usize) -> f32 {
    unsafe { (p.cast::<u8>().add(off) as *const f32).read_unaligned() }
}
unsafe fn ri(p: *const core::ffi::c_void, off: usize) -> i32 {
    unsafe { (p.cast::<u8>().add(off) as *const i32).read_unaligned() }
}
unsafe fn wf(p: *mut core::ffi::c_void, off: usize, v: f32) {
    unsafe { (p.cast::<u8>().add(off) as *mut f32).write_unaligned(v) };
}
unsafe fn wi(p: *mut core::ffi::c_void, off: usize, v: i32) {
    unsafe { (p.cast::<u8>().add(off) as *mut i32).write_unaligned(v) };
}
unsafe fn pod_to_value(kind: Kind, p: *const core::ffi::c_void) -> Value {
    let rf = |o| unsafe { rf(p, o) };
    let ri = |o| unsafe { ri(p, o) };
    match kind {
        Kind::Float => Value::Float(rf(0)),
        Kind::Color => Value::Color([rf(0), rf(4), rf(8), rf(12)]),
        Kind::Matrix => Value::Matrix([rf(0), rf(4), rf(8), rf(12), rf(16), rf(20)]),
        Kind::ViewBox => Value::ViewBox([rf(0), rf(4), rf(8), rf(12)]),
        Kind::Length => Value::Length(rf(0), if ri(4) == 1 { Units::Percentage } else { Units::Number }),
        Kind::AspectRatio => Value::Aspect { defer: ri(0) != 0, align: ri(4), slice: ri(8) == 1 },
        _ => Value::Enum(ri(0)), // FillRule/Display/Overflow/LineCap/LineJoin/Visibility/UnitType/ExtendMode
    }
}
unsafe fn value_to_pod(v: &Value, p: *mut core::ffi::c_void) {
    let wf = |o, x| unsafe { wf(p, o, x) };
    let wi = |o, x| unsafe { wi(p, o, x) };
    match v {
        Value::Float(x) => wf(0, *x),
        Value::Color(c) => (0..4).for_each(|i| wf(i * 4, c[i])),
        Value::Matrix(m) => (0..6).for_each(|i| wf(i * 4, m[i])),
        Value::ViewBox(b) => (0..4).for_each(|i| wf(i * 4, b[i])),
        Value::Length(x, u) => {
            wf(0, *x);
            wi(4, if *u == Units::Percentage { 1 } else { 0 });
        }
        Value::Aspect { defer, align, slice } => {
            wi(0, *defer as i32);
            wi(4, *align);
            wi(8, *slice as i32);
        }
        Value::Enum(e) => wi(0, *e),
        _ => {}
    }
}

fn paint_type(t: D2D1_SVG_PAINT_TYPE) -> PaintType {
    match t.0 {
        1 => PaintType::Color,
        2 => PaintType::CurrentColor,
        3 => PaintType::Uri,
        4 => PaintType::UriNone,
        5 => PaintType::UriColor,
        6 => PaintType::UriCurrentColor,
        _ => PaintType::None,
    }
}
fn path_cmd(c: D2D1_SVG_PATH_COMMAND) -> PathCmd {
    // PathCmd's discriminants are the D2D1_SVG_PATH_COMMAND values; clamp unknown to ClosePath.
    [
        PathCmd::ClosePath, PathCmd::MoveAbs, PathCmd::MoveRel, PathCmd::LineAbs, PathCmd::LineRel,
        PathCmd::CubicAbs, PathCmd::CubicRel, PathCmd::QuadAbs, PathCmd::QuadRel, PathCmd::ArcAbs,
        PathCmd::ArcRel, PathCmd::HorizAbs, PathCmd::HorizRel, PathCmd::VertAbs, PathCmd::VertRel,
        PathCmd::CubicSmoothAbs, PathCmd::CubicSmoothRel, PathCmd::QuadSmoothAbs, PathCmd::QuadSmoothRel,
    ]
    .get(c.0 as usize)
    .copied()
    .unwrap_or(PathCmd::ClosePath)
}

// --- shared array edit helpers ----------------------------------------------------------------

unsafe fn in_slice<'a, T>(p: *const T, n: u32) -> &'a [T] {
    if p.is_null() {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(p, n as usize) }
    }
}
/// `Update*`: overwrite/grow `buf[start..start+src.len()]`; `start > len` is out of range.
fn splice<T: Copy>(buf: &mut Vec<T>, start: u32, src: &[T], fill: T) -> WResult<()> {
    let start = start as usize;
    if start > buf.len() {
        return Err(err(E_INVALIDARG));
    }
    let end = start + src.len();
    if end > buf.len() {
        buf.resize(end, fill);
    }
    buf[start..end].copy_from_slice(src);
    Ok(())
}
fn remove_end<T>(buf: &mut Vec<T>, count: u32) -> WResult<()> {
    let c = count as usize;
    if c > buf.len() {
        return Err(err(E_INVALIDARG));
    }
    buf.truncate(buf.len() - c);
    Ok(())
}
/// Bounds-check a `Get*(dst, count, start)` request against a stored buffer of `len` elements.
fn get_range(len: usize, start: u32, count: u32) -> WResult<std::ops::Range<usize>> {
    let (s, c) = (start as usize, count as usize);
    if s + c > len {
        return Err(err(E_INVALIDARG));
    }
    Ok(s..s + c)
}

// --- document ----------------------------------------------------------------------------------

#[implement(ID2D1SvgDocument)]
pub(crate) struct SvgDocObj {
    pub(crate) doc: Rc<RefCell<Document>>,
    pub(crate) factory: Option<ID2D1Factory>,
}
impl SvgDocObj {
    pub(crate) fn new(doc: Document, factory: Option<ID2D1Factory>) -> Self {
        Self { doc: Rc::new(RefCell::new(doc)), factory }
    }
}
fn element(doc: &Rc<RefCell<Document>>, id: u32, factory: Option<ID2D1Factory>) -> ID2D1SvgElement {
    SvgElementObj { doc: doc.clone(), id, factory }.into()
}

impl ID2D1Resource_Impl for SvgDocObj_Impl {
    fn GetFactory(&self, f: OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
impl ID2D1SvgDocument_Impl for SvgDocObj_Impl {
    fn GetViewportSize(&self) -> D2D_SIZE_F {
        let [width, height] = self.doc.borrow().viewport;
        D2D_SIZE_F { width, height }
    }
    fn SetViewportSize(&self, s: &D2D_SIZE_F) -> WResult<()> {
        self.doc.borrow_mut().set_viewport(s.width, s.height).map_err(err)
    }
    fn GetRoot(&self, root: OutRef<ID2D1SvgElement>) {
        let id = self.doc.borrow().root_id();
        let _ = root.write(id.map(|id| element(&self.doc, id, self.factory.clone())));
    }
    fn SetRoot(&self, r: Ref<ID2D1SvgElement>) -> WResult<()> {
        let id = recover(&self.doc, &r)?;
        self.doc.borrow_mut().set_root(id).map_err(err)
    }
    fn FindElementById(&self, id: &PCWSTR) -> WResult<ID2D1SvgElement> {
        // Native returns S_OK with a *null* element when the id is absent (not an error); the only way
        // to express "S_OK, no out" through this Result is an Err carrying S_OK, which leaves the
        // caller's out null and returns 0 — matching native's ABI (verified against the WARP oracle).
        match self.doc.borrow().find_id(&pcwstr(id)) {
            Some(nid) => Ok(element(&self.doc, nid, self.factory.clone())),
            None => Err(err(0)),
        }
    }
    fn Serialize(&self, out: Ref<IStream>, subtree: Ref<ID2D1SvgElement>) -> WResult<()> {
        let stream = out.as_ref().ok_or(err(E_INVALIDARG))?;
        let xml = match subtree.as_ref() {
            Some(_) => {
                let id = recover(&self.doc, &subtree)?;
                self.doc.borrow().serialize_from(id).ok_or(err(E_INVALIDARG))?
            }
            None => self.doc.borrow().serialize(),
        };
        let bytes = xml.into_bytes();
        unsafe { stream.Write(bytes.as_ptr().cast(), bytes.len() as u32, None).ok()? };
        Ok(())
    }
    fn Deserialize(&self, input: Ref<IStream>) -> WResult<ID2D1SvgElement> {
        let stream = input.as_ref().ok_or(err(E_INVALIDARG))?;
        let bytes = read_stream(stream);
        let xml = String::from_utf8_lossy(&bytes);
        let viewport = self.doc.borrow().viewport;
        let parsed = Document::parse(&xml, viewport).map_err(err)?;
        let id = self.doc.borrow_mut().import_root(parsed);
        Ok(element(&self.doc, id, self.factory.clone()))
    }
    fn CreatePaint(&self, t: D2D1_SVG_PAINT_TYPE, color: *const D2D1_COLOR_F, id: &PCWSTR) -> WResult<ID2D1SvgPaint> {
        let c = if color.is_null() { [0.0; 4] } else { let c = unsafe { &*color }; [c.r, c.g, c.b, c.a] };
        let v = Value::Paint { ty: paint_type(t), color: c, id: pcwstr(id) };
        Ok(SvgPaintObj(AttrInner::detached(&self.doc, v, self.factory.clone())).into())
    }
    fn CreateStrokeDashArray(&self, d: *const D2D1_SVG_LENGTH, c: u32) -> WResult<ID2D1SvgStrokeDashArray> {
        let dashes = unsafe { in_slice(d, c) }
            .iter()
            .map(|l| (l.value, if l.units.0 == 1 { Units::Percentage } else { Units::Number }))
            .collect();
        Ok(SvgDashesObj(AttrInner::detached(&self.doc, Value::Dashes(dashes), self.factory.clone())).into())
    }
    fn CreatePointCollection(&self, p: *const Vector2, c: u32) -> WResult<ID2D1SvgPointCollection> {
        let pts = unsafe { in_slice(p, c) }.iter().map(|v| [v.X, v.Y]).collect();
        Ok(SvgPointsObj(AttrInner::detached(&self.doc, Value::Points(pts), self.factory.clone())).into())
    }
    fn CreatePathData(&self, s: *const f32, sc: u32, cmd: *const D2D1_SVG_PATH_COMMAND, cc: u32) -> WResult<ID2D1SvgPathData> {
        let data = unsafe { in_slice(s, sc) }.to_vec();
        let cmds = unsafe { in_slice(cmd, cc) }.iter().map(|c| path_cmd(*c)).collect();
        Ok(SvgPathDataObj(AttrInner::detached(&self.doc, Value::Path { cmds, data }, self.factory.clone())).into())
    }
}

/// Recover a same-document element handle's node id from a COM `ID2D1SvgElement`.
fn recover(doc: &Rc<RefCell<Document>>, r: &Ref<ID2D1SvgElement>) -> WResult<u32> {
    let e = r.as_ref().ok_or(err(E_INVALIDARG))?.cast_object_ref::<SvgElementObj>().map_err(|_| err(E_INVALIDARG))?;
    if !Rc::ptr_eq(&e.doc, doc) {
        return Err(err(E_INVALIDARG)); // element belongs to another document
    }
    Ok(e.id)
}

// --- element -----------------------------------------------------------------------------------

#[implement(ID2D1SvgElement)]
pub(crate) struct SvgElementObj {
    pub(crate) doc: Rc<RefCell<Document>>,
    pub(crate) id: u32,
    pub(crate) factory: Option<ID2D1Factory>,
}
impl SvgElementObj {
    fn child(&self, r: &Ref<ID2D1SvgElement>) -> WResult<u32> {
        recover(&self.doc, r)
    }
}

impl ID2D1Resource_Impl for SvgElementObj_Impl {
    fn GetFactory(&self, f: OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
impl ID2D1SvgElement_Impl for SvgElementObj_Impl {
    fn GetDocument(&self, document: OutRef<ID2D1SvgDocument>) {
        let _ = document.write(Some(SvgDocObj { doc: self.doc.clone(), factory: None }.into()));
    }
    fn GetTagName(&self, name: PWSTR, namecount: u32) -> WResult<()> {
        write_pwstr(name, namecount, &self.doc.borrow().tag_of(self.id).unwrap_or_default())
    }
    fn GetTagNameLength(&self) -> u32 {
        utf16_len(&self.doc.borrow().tag_of(self.id).unwrap_or_default())
    }
    fn IsTextContent(&self) -> BOOL {
        self.doc.borrow().is_text_content(self.id).into()
    }
    fn GetParent(&self, parent: OutRef<ID2D1SvgElement>) {
        let id = self.doc.borrow().parent_id(self.id);
        let _ = parent.write(id.map(|id| element(&self.doc, id, self.factory.clone())));
    }
    fn HasChildren(&self) -> BOOL {
        self.doc.borrow().has_children(self.id).into()
    }
    fn GetFirstChild(&self, child: OutRef<ID2D1SvgElement>) {
        let id = self.doc.borrow().first_child_id(self.id);
        let _ = child.write(id.map(|id| element(&self.doc, id, self.factory.clone())));
    }
    fn GetLastChild(&self, child: OutRef<ID2D1SvgElement>) {
        let id = self.doc.borrow().last_child_id(self.id);
        let _ = child.write(id.map(|id| element(&self.doc, id, self.factory.clone())));
    }
    fn GetPreviousChild(&self, referencechild: Ref<ID2D1SvgElement>) -> WResult<ID2D1SvgElement> {
        let r = self.child(&referencechild)?;
        let id = self.doc.borrow().sibling_id(self.id, r, -1).ok_or(err(E_INVALIDARG))?;
        Ok(element(&self.doc, id, self.factory.clone()))
    }
    fn GetNextChild(&self, referencechild: Ref<ID2D1SvgElement>) -> WResult<ID2D1SvgElement> {
        let r = self.child(&referencechild)?;
        let id = self.doc.borrow().sibling_id(self.id, r, 1).ok_or(err(E_INVALIDARG))?;
        Ok(element(&self.doc, id, self.factory.clone()))
    }
    fn InsertChildBefore(&self, newchild: Ref<ID2D1SvgElement>, referencechild: Ref<ID2D1SvgElement>) -> WResult<()> {
        let (c, r) = (self.child(&newchild)?, self.child(&referencechild)?);
        self.doc.borrow_mut().reparent(self.id, c, Some(r)).map_err(err)
    }
    fn AppendChild(&self, newchild: Ref<ID2D1SvgElement>) -> WResult<()> {
        let c = self.child(&newchild)?;
        self.doc.borrow_mut().reparent(self.id, c, None).map_err(err)
    }
    fn ReplaceChild(&self, newchild: Ref<ID2D1SvgElement>, oldchild: Ref<ID2D1SvgElement>) -> WResult<()> {
        let (n, o) = (self.child(&newchild)?, self.child(&oldchild)?);
        self.doc.borrow_mut().replace_child(self.id, n, o).map_err(err)
    }
    fn RemoveChild(&self, oldchild: Ref<ID2D1SvgElement>) -> WResult<()> {
        let o = self.child(&oldchild)?;
        self.doc.borrow_mut().remove_child(self.id, o).map_err(err)
    }
    fn CreateChild(&self, tagname: &PCWSTR) -> WResult<ID2D1SvgElement> {
        let id = self.doc.borrow_mut().create_child(self.id, &pcwstr(tagname)).map_err(err)?;
        Ok(element(&self.doc, id, self.factory.clone()))
    }
    fn IsAttributeSpecified(&self, name: &PCWSTR, inherited: *mut BOOL) -> BOOL {
        if !inherited.is_null() {
            unsafe { *inherited = BOOL(0) };
        }
        self.doc.borrow().is_specified(self.id, &pcwstr(name)).into()
    }
    fn GetSpecifiedAttributeCount(&self) -> u32 {
        self.doc.borrow().specified_count(self.id)
    }
    fn GetSpecifiedAttributeName(&self, index: u32, name: PWSTR, namecount: u32, inherited: *mut BOOL) -> WResult<()> {
        let n = self.doc.borrow().specified_name(self.id, index).ok_or(err(E_INVALIDARG))?;
        if !inherited.is_null() {
            unsafe { *inherited = BOOL(0) };
        }
        write_pwstr(name, namecount, &n)
    }
    fn GetSpecifiedAttributeNameLength(&self, index: u32, namelength: *mut u32, inherited: *mut BOOL) -> WResult<()> {
        let n = self.doc.borrow().specified_name(self.id, index).ok_or(err(E_INVALIDARG))?;
        unsafe {
            *namelength = utf16_len(&n);
            if !inherited.is_null() {
                *inherited = BOOL(0);
            }
        }
        Ok(())
    }
    fn RemoveAttribute(&self, name: &PCWSTR) -> WResult<()> {
        let name = pcwstr(name);
        let mut doc = self.doc.borrow_mut();
        if doc.attr_kind_of(self.id, &name).is_none() {
            return Err(err(E_INVALIDARG)); // not an attribute this element carries
        }
        doc.remove_attr(self.id, &name);
        Ok(())
    }
    fn SetTextValue(&self, value: &PCWSTR, valuecount: u32) -> WResult<()> {
        self.doc.borrow_mut().set_text_value(self.id, &pcwstr_n(value, valuecount)).map_err(err)
    }
    fn GetTextValue(&self, value: PWSTR, valuecount: u32) -> WResult<()> {
        let t = self.doc.borrow().text_value(self.id).ok_or(err(D2DERR_INVALID_CALL))?;
        write_pwstr(value, valuecount, &t)
    }
    fn GetTextValueLength(&self) -> u32 {
        self.doc.borrow().text_value(self.id).map_or(0, |t| utf16_len(&t))
    }
    fn SetAttributeValue(&self, name: &PCWSTR, value: Ref<ID2D1SvgAttribute>) -> WResult<()> {
        let name = pcwstr(name);
        let attr = value.as_ref().ok_or(err(E_INVALIDARG))?;
        let v = attr_value_of(attr).ok_or(err(E_INVALIDARG))?;
        self.doc.borrow_mut().set_attr_value(self.id, &name, v).map_err(err)?;
        bind_attr(attr, self.id, &name);
        Ok(())
    }
    fn SetAttributeValue2(&self, name: &PCWSTR, r#type: D2D1_SVG_ATTRIBUTE_POD_TYPE, value: *const core::ffi::c_void, size: u32) -> WResult<()> {
        let (kind, sz) = pod_kind(r#type).ok_or(err(E_INVALIDARG))?;
        if size as usize != sz || value.is_null() {
            return Err(err(E_INVALIDARG));
        }
        let v = unsafe { pod_to_value(kind, value) };
        self.doc.borrow_mut().set_attr_value(self.id, &pcwstr(name), v).map_err(err)
    }
    fn SetAttributeValue3(&self, name: &PCWSTR, _type: D2D1_SVG_ATTRIBUTE_STRING_TYPE, value: &PCWSTR) -> WResult<()> {
        self.doc.borrow_mut().set_attr_string(self.id, &pcwstr(name), &pcwstr(value)).map_err(err)
    }
    fn GetAttributeValue(&self, name: &PCWSTR, riid: *const GUID, value: *mut *mut core::ffi::c_void) -> WResult<()> {
        let name = pcwstr(name);
        let (kind, v) = {
            let doc = self.doc.borrow();
            let kind = doc.attr_kind_of(self.id, &name).ok_or(err(E_INVALIDARG))?;
            let v = doc.attr_value(self.id, &name).ok_or(err(E_INVALIDARG))?;
            (kind, v)
        };
        let inner = || AttrInner::bound(&self.doc, self.id, &name, v.clone(), self.factory.clone());
        let riid = unsafe { &*riid };
        unsafe {
            match kind {
                Kind::Paint => Into::<ID2D1SvgPaint>::into(SvgPaintObj(inner())).query(riid, value),
                Kind::PathData => Into::<ID2D1SvgPathData>::into(SvgPathDataObj(inner())).query(riid, value),
                Kind::Points => Into::<ID2D1SvgPointCollection>::into(SvgPointsObj(inner())).query(riid, value),
                Kind::DashArray => Into::<ID2D1SvgStrokeDashArray>::into(SvgDashesObj(inner())).query(riid, value),
                _ => return Err(err(E_INVALIDARG)), // not an object-valued attribute
            }
        }
        .ok()
    }
    fn GetAttributeValue2(&self, name: &PCWSTR, r#type: D2D1_SVG_ATTRIBUTE_POD_TYPE, value: *mut core::ffi::c_void, size: u32) -> WResult<()> {
        let (kind, sz) = pod_kind(r#type).ok_or(err(E_INVALIDARG))?;
        if size as usize != sz || value.is_null() {
            return Err(err(E_INVALIDARG));
        }
        let name = pcwstr(name);
        let doc = self.doc.borrow();
        if doc.attr_kind_of(self.id, &name) != Some(kind) {
            return Err(err(E_INVALIDARG));
        }
        let v = doc.attr_value(self.id, &name).ok_or(err(E_INVALIDARG))?;
        unsafe { value_to_pod(&v, value) };
        Ok(())
    }
    fn GetAttributeValue3(&self, name: &PCWSTR, _type: D2D1_SVG_ATTRIBUTE_STRING_TYPE, value: PWSTR, valuecount: u32) -> WResult<()> {
        let s = self.doc.borrow().attr_string(self.id, &pcwstr(name)).ok_or(err(E_INVALIDARG))?;
        write_pwstr(value, valuecount, &s)
    }
    fn GetAttributeValueLength(&self, name: &PCWSTR, _type: D2D1_SVG_ATTRIBUTE_STRING_TYPE) -> WResult<u32> {
        self.doc.borrow().attr_string(self.id, &pcwstr(name)).map(|s| utf16_len(&s)).ok_or(err(E_INVALIDARG))
    }
}

// --- attribute value objects -------------------------------------------------------------------

/// The backing shared by all four attribute objects: a local [`Value`], plus (when `bind` is set) a
/// live link to an element attribute so reads/writes go through the document.
struct AttrInner {
    doc: Rc<RefCell<Document>>,
    bind: RefCell<Option<(u32, String)>>,
    local: RefCell<Value>,
    factory: Option<ID2D1Factory>,
}
impl AttrInner {
    fn detached(doc: &Rc<RefCell<Document>>, v: Value, factory: Option<ID2D1Factory>) -> Self {
        Self { doc: doc.clone(), bind: RefCell::new(None), local: RefCell::new(v), factory }
    }
    fn bound(doc: &Rc<RefCell<Document>>, id: u32, name: &str, v: Value, factory: Option<ID2D1Factory>) -> Self {
        Self { doc: doc.clone(), bind: RefCell::new(Some((id, name.to_string()))), local: RefCell::new(v), factory }
    }
    fn read(&self) -> Value {
        if let Some((id, name)) = self.bind.borrow().as_ref() {
            if let Some(v) = self.doc.borrow().attr_value(*id, name) {
                return v;
            }
        }
        self.local.borrow().clone()
    }
    fn write(&self, v: Value) {
        if let Some((id, name)) = self.bind.borrow().as_ref() {
            let _ = self.doc.borrow_mut().set_attr_value(*id, name, v.clone());
        }
        *self.local.borrow_mut() = v;
    }
    fn get_element(&self, out: OutRef<ID2D1SvgElement>) {
        let e = self.bind.borrow().as_ref().map(|(id, _)| element(&self.doc, *id, self.factory.clone()));
        let _ = out.write(e);
    }
}

/// Read a `SetAttributeValue(name, attribute)` object back to its [`Value`], and bind it afterwards.
fn attr_value_of(a: &ID2D1SvgAttribute) -> Option<Value> {
    if let Ok(p) = a.cast_object_ref::<SvgPaintObj>() {
        Some(p.0.read())
    } else if let Ok(p) = a.cast_object_ref::<SvgPathDataObj>() {
        Some(p.0.read())
    } else if let Ok(p) = a.cast_object_ref::<SvgPointsObj>() {
        Some(p.0.read())
    } else if let Ok(p) = a.cast_object_ref::<SvgDashesObj>() {
        Some(p.0.read())
    } else {
        None
    }
}
fn bind_attr(a: &ID2D1SvgAttribute, id: u32, name: &str) {
    let set = |b: &RefCell<Option<(u32, String)>>| *b.borrow_mut() = Some((id, name.to_string()));
    if let Ok(p) = a.cast_object_ref::<SvgPaintObj>() {
        set(&p.0.bind);
    } else if let Ok(p) = a.cast_object_ref::<SvgPathDataObj>() {
        set(&p.0.bind);
    } else if let Ok(p) = a.cast_object_ref::<SvgPointsObj>() {
        set(&p.0.bind);
    } else if let Ok(p) = a.cast_object_ref::<SvgDashesObj>() {
        set(&p.0.bind);
    }
}

#[implement(ID2D1SvgPaint)]
pub(crate) struct SvgPaintObj(AttrInner);
impl SvgPaintObj {
    fn parts(&self) -> (PaintType, [f32; 4], String) {
        match self.0.read() {
            Value::Paint { ty, color, id } => (ty, color, id),
            _ => (PaintType::None, [0.0; 4], String::new()),
        }
    }
}
impl ID2D1Resource_Impl for SvgPaintObj_Impl {
    fn GetFactory(&self, f: OutRef<ID2D1Factory>) {
        let _ = f.write(self.0.factory.clone());
    }
}
impl ID2D1SvgAttribute_Impl for SvgPaintObj_Impl {
    fn GetElement(&self, element: OutRef<ID2D1SvgElement>) {
        self.0.get_element(element);
    }
    fn Clone(&self) -> WResult<ID2D1SvgAttribute> {
        Ok(Into::<ID2D1SvgPaint>::into(SvgPaintObj(AttrInner::detached(&self.0.doc, self.0.read(), self.0.factory.clone()))).cast()?)
    }
}
impl ID2D1SvgPaint_Impl for SvgPaintObj_Impl {
    fn SetPaintType(&self, t: D2D1_SVG_PAINT_TYPE) -> WResult<()> {
        let (_, color, id) = self.parts();
        self.0.write(Value::Paint { ty: paint_type(t), color, id });
        Ok(())
    }
    fn GetPaintType(&self) -> D2D1_SVG_PAINT_TYPE {
        D2D1_SVG_PAINT_TYPE(self.parts().0 as i32)
    }
    fn SetColor(&self, color: *const D2D1_COLOR_F) -> WResult<()> {
        let (ty, _, id) = self.parts();
        let c = unsafe { &*color };
        self.0.write(Value::Paint { ty, color: [c.r, c.g, c.b, c.a], id });
        Ok(())
    }
    fn GetColor(&self, color: *mut D2D1_COLOR_F) {
        let c = self.parts().1;
        unsafe { *color = D2D1_COLOR_F { r: c[0], g: c[1], b: c[2], a: c[3] } };
    }
    fn SetId(&self, id: &PCWSTR) -> WResult<()> {
        let (ty, color, _) = self.parts();
        self.0.write(Value::Paint { ty, color, id: pcwstr(id) });
        Ok(())
    }
    fn GetId(&self, id: PWSTR, idcount: u32) -> WResult<()> {
        write_pwstr(id, idcount, &self.parts().2)
    }
    fn GetIdLength(&self) -> u32 {
        utf16_len(&self.parts().2)
    }
}

#[implement(ID2D1SvgPathData)]
pub(crate) struct SvgPathDataObj(AttrInner);
impl SvgPathDataObj {
    fn parts(&self) -> (Vec<PathCmd>, Vec<f32>) {
        match self.0.read() {
            Value::Path { cmds, data } => (cmds, data),
            _ => (Vec::new(), Vec::new()),
        }
    }
}
impl ID2D1Resource_Impl for SvgPathDataObj_Impl {
    fn GetFactory(&self, f: OutRef<ID2D1Factory>) {
        let _ = f.write(self.0.factory.clone());
    }
}
impl ID2D1SvgAttribute_Impl for SvgPathDataObj_Impl {
    fn GetElement(&self, element: OutRef<ID2D1SvgElement>) {
        self.0.get_element(element);
    }
    fn Clone(&self) -> WResult<ID2D1SvgAttribute> {
        Ok(Into::<ID2D1SvgPathData>::into(SvgPathDataObj(AttrInner::detached(&self.0.doc, self.0.read(), self.0.factory.clone()))).cast()?)
    }
}
impl ID2D1SvgPathData_Impl for SvgPathDataObj_Impl {
    fn RemoveSegmentDataAtEnd(&self, count: u32) -> WResult<()> {
        let (cmds, mut data) = self.parts();
        remove_end(&mut data, count)?;
        self.0.write(Value::Path { cmds, data });
        Ok(())
    }
    fn UpdateSegmentData(&self, data: *const f32, count: u32, start: u32) -> WResult<()> {
        let (cmds, mut d) = self.parts();
        splice(&mut d, start, unsafe { in_slice(data, count) }, 0.0)?;
        self.0.write(Value::Path { cmds, data: d });
        Ok(())
    }
    fn GetSegmentData(&self, data: *mut f32, count: u32, start: u32) -> WResult<()> {
        let (_, d) = self.parts();
        let r = get_range(d.len(), start, count)?;
        (0..r.len()).for_each(|i| unsafe { data.add(i).write_unaligned(d[r.start + i]) });
        Ok(())
    }
    fn GetSegmentDataCount(&self) -> u32 {
        self.parts().1.len() as u32
    }
    fn RemoveCommandsAtEnd(&self, count: u32) -> WResult<()> {
        let (mut cmds, data) = self.parts();
        remove_end(&mut cmds, count)?;
        self.0.write(Value::Path { cmds, data });
        Ok(())
    }
    fn UpdateCommands(&self, commands: *const D2D1_SVG_PATH_COMMAND, count: u32, start: u32) -> WResult<()> {
        let (mut cmds, data) = self.parts();
        let src: Vec<PathCmd> = unsafe { in_slice(commands, count) }.iter().map(|c| path_cmd(*c)).collect();
        splice(&mut cmds, start, &src, PathCmd::ClosePath)?;
        self.0.write(Value::Path { cmds, data });
        Ok(())
    }
    fn GetCommands(&self, commands: *mut D2D1_SVG_PATH_COMMAND, count: u32, start: u32) -> WResult<()> {
        let (c, _) = self.parts();
        let r = get_range(c.len(), start, count)?;
        (0..r.len()).for_each(|i| unsafe { commands.add(i).write_unaligned(D2D1_SVG_PATH_COMMAND(c[r.start + i] as i32)) });
        Ok(())
    }
    fn GetCommandsCount(&self) -> u32 {
        self.parts().0.len() as u32
    }
    fn CreatePathGeometry(&self, _fillmode: D2D1_FILL_MODE) -> WResult<ID2D1PathGeometry1> {
        // ponytail: path-data → ID2D1PathGeometry1 is the geometry spec's surface, not the SVG DOM's;
        // add it when a scene needs a geometry built from an SVG path-data object.
        Err(E_NOTIMPL.into())
    }
}

#[implement(ID2D1SvgPointCollection)]
pub(crate) struct SvgPointsObj(AttrInner);
impl SvgPointsObj {
    fn pts(&self) -> Vec<[f32; 2]> {
        match self.0.read() {
            Value::Points(p) => p,
            _ => Vec::new(),
        }
    }
}
impl ID2D1Resource_Impl for SvgPointsObj_Impl {
    fn GetFactory(&self, f: OutRef<ID2D1Factory>) {
        let _ = f.write(self.0.factory.clone());
    }
}
impl ID2D1SvgAttribute_Impl for SvgPointsObj_Impl {
    fn GetElement(&self, element: OutRef<ID2D1SvgElement>) {
        self.0.get_element(element);
    }
    fn Clone(&self) -> WResult<ID2D1SvgAttribute> {
        Ok(Into::<ID2D1SvgPointCollection>::into(SvgPointsObj(AttrInner::detached(&self.0.doc, self.0.read(), self.0.factory.clone()))).cast()?)
    }
}
impl ID2D1SvgPointCollection_Impl for SvgPointsObj_Impl {
    fn RemovePointsAtEnd(&self, count: u32) -> WResult<()> {
        let mut p = self.pts();
        remove_end(&mut p, count)?;
        self.0.write(Value::Points(p));
        Ok(())
    }
    fn UpdatePoints(&self, points: *const Vector2, count: u32, start: u32) -> WResult<()> {
        let mut p = self.pts();
        let src: Vec<[f32; 2]> = unsafe { in_slice(points, count) }.iter().map(|v| [v.X, v.Y]).collect();
        splice(&mut p, start, &src, [0.0; 2])?;
        self.0.write(Value::Points(p));
        Ok(())
    }
    fn GetPoints(&self, points: *mut Vector2, count: u32, start: u32) -> WResult<()> {
        let p = self.pts();
        let r = get_range(p.len(), start, count)?;
        (0..r.len()).for_each(|i| unsafe { points.add(i).write_unaligned(Vector2 { X: p[r.start + i][0], Y: p[r.start + i][1] }) });
        Ok(())
    }
    fn GetPointsCount(&self) -> u32 {
        self.pts().len() as u32
    }
}

#[implement(ID2D1SvgStrokeDashArray)]
pub(crate) struct SvgDashesObj(AttrInner);
impl SvgDashesObj {
    fn dashes(&self) -> Vec<(f32, Units)> {
        match self.0.read() {
            Value::Dashes(d) => d,
            _ => Vec::new(),
        }
    }
}
impl ID2D1Resource_Impl for SvgDashesObj_Impl {
    fn GetFactory(&self, f: OutRef<ID2D1Factory>) {
        let _ = f.write(self.0.factory.clone());
    }
}
impl ID2D1SvgAttribute_Impl for SvgDashesObj_Impl {
    fn GetElement(&self, element: OutRef<ID2D1SvgElement>) {
        self.0.get_element(element);
    }
    fn Clone(&self) -> WResult<ID2D1SvgAttribute> {
        Ok(Into::<ID2D1SvgStrokeDashArray>::into(SvgDashesObj(AttrInner::detached(&self.0.doc, self.0.read(), self.0.factory.clone()))).cast()?)
    }
}
fn to_length(d: (f32, Units)) -> D2D1_SVG_LENGTH {
    D2D1_SVG_LENGTH { value: d.0, units: D2D1_SVG_LENGTH_UNITS(if d.1 == Units::Percentage { 1 } else { 0 }) }
}
impl ID2D1SvgStrokeDashArray_Impl for SvgDashesObj_Impl {
    fn RemoveDashesAtEnd(&self, count: u32) -> WResult<()> {
        let mut d = self.dashes();
        remove_end(&mut d, count)?;
        self.0.write(Value::Dashes(d));
        Ok(())
    }
    fn UpdateDashes(&self, dashes: *const D2D1_SVG_LENGTH, count: u32, start: u32) -> WResult<()> {
        let mut d = self.dashes();
        let src: Vec<(f32, Units)> = unsafe { in_slice(dashes, count) }
            .iter()
            .map(|l| (l.value, if l.units.0 == 1 { Units::Percentage } else { Units::Number }))
            .collect();
        splice(&mut d, start, &src, (0.0, Units::Number))?;
        self.0.write(Value::Dashes(d));
        Ok(())
    }
    fn UpdateDashes2(&self, dashes: *const f32, count: u32, start: u32) -> WResult<()> {
        let mut d = self.dashes();
        let src: Vec<(f32, Units)> = unsafe { in_slice(dashes, count) }.iter().map(|v| (*v, Units::Number)).collect();
        splice(&mut d, start, &src, (0.0, Units::Number))?;
        self.0.write(Value::Dashes(d));
        Ok(())
    }
    fn GetDashes(&self, dashes: *mut D2D1_SVG_LENGTH, count: u32, start: u32) -> WResult<()> {
        let d = self.dashes();
        let r = get_range(d.len(), start, count)?;
        (0..r.len()).for_each(|i| unsafe { dashes.add(i).write_unaligned(to_length(d[r.start + i])) });
        Ok(())
    }
    fn GetDashes2(&self, dashes: *mut f32, count: u32, start: u32) -> WResult<()> {
        let d = self.dashes();
        let r = get_range(d.len(), start, count)?;
        (0..r.len()).for_each(|i| unsafe { dashes.add(i).write_unaligned(d[r.start + i].0) });
        Ok(())
    }
    fn GetDashesCount(&self) -> u32 {
        self.dashes().len() as u32
    }
}
