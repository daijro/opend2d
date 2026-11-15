//! `ID2D1SvgDocument` / `ID2D1SvgElement`: a mutable SVG DOM, its XML parse and serialize, and
//! (in [`render`]) the lowering to ordinary Direct2D fills/strokes that native's SVG document
//! performs. Native d2d1 on WARP is the reference (from the native build); the managed classes are a
//! guide. SVG has no rasterizer of its own — every pixel comes from the geometry/brush code.
//!
//! The XML parse reuses `roxmltree`; only the supported element/attribute subset is retained, and
//! malformed input or a non-`svg` root maps to `E_INVALIDARG`, exactly as native reports it.

// `!(x > 0.0)` is deliberate: it rejects NaN the way native's positivity checks do.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod attr;
mod render;

pub use attr::{Kind, PaintType, PathCmd, Units, Value};

/// `E_INVALIDARG` — native's result for malformed XML, a bad root, or a type-mismatched set.
pub const E_INVALIDARG: i32 = -2147024809; // 0x80070057

/// A parsed SVG document: a root element tree plus the viewport size (DIPs) it renders at.
///
/// For the COM DOM (`ID2D1SvgElement`), every node carries a stable [`Element::id`] so a handle
/// survives reparenting; [`detached`](Document::detached) keeps subtrees that `RemoveChild` pulled
/// out of the tree but that a later `AppendChild` may re-insert, exactly as a live COM element does.
#[derive(Clone, Debug)]
pub struct Document {
    pub viewport: [f32; 2],
    root: Option<Element>,
    /// Node id allocator; 0 means "unassigned".
    next_id: u32,
    /// Subtrees removed from the tree but still referenced by a live COM element handle.
    detached: Vec<Element>,
}

/// One node of the SVG DOM. A node with an empty [`Element::tag`] is text content (`title`/`desc`).
#[derive(Clone, Debug)]
pub struct Element {
    /// Stable identity for COM handles (see [`Document`]); 0 until [`Document::assign_ids`] runs.
    id: u32,
    tag: String,
    text: String,
    /// Specified attributes (stored in insertion order). Native exposes them **alphabetically**
    /// (serialize, `GetSpecifiedAttributeName`), so those paths sort by name; lookups are by name.
    attrs: Vec<(String, Value)>,
    children: Vec<Element>,
}

impl Element {
    pub fn new(tag: &str) -> Element {
        Element { id: 0, tag: tag.to_string(), text: String::new(), attrs: Vec::new(), children: Vec::new() }
    }

    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn tag(&self) -> &str {
        &self.tag
    }
    pub fn is_text_content(&self) -> bool {
        self.tag.is_empty()
    }
    pub fn text(&self) -> &str {
        &self.text
    }
    pub fn set_text(&mut self, text: &str) {
        self.text = text.to_string();
    }
    pub fn children(&self) -> &[Element] {
        &self.children
    }

    /// The typed value of a specified attribute.
    pub fn attr(&self, name: &str) -> Option<&Value> {
        self.attrs.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }
    pub fn is_attribute_specified(&self, name: &str) -> bool {
        self.attrs.iter().any(|(n, _)| n == name)
    }
    pub fn specified_attribute_count(&self) -> usize {
        self.attrs.len()
    }
    /// `GetSpecifiedAttributeName`: the `i`-th specified attribute in **alphabetical** order — the
    /// order native keeps them in (verified against the WARP oracle), not insertion order.
    pub fn specified_attribute_name(&self, i: usize) -> Option<&str> {
        let mut names: Vec<&str> = self.attrs.iter().map(|(n, _)| n.as_str()).collect();
        names.sort_unstable();
        names.get(i).copied()
    }

    /// `GetAttributeValue(..., SvgAttributeStringType.Svg)`: the attribute reformatted to its SVG
    /// string, or `None` if unspecified.
    pub fn get_attribute(&self, name: &str) -> Option<String> {
        let v = self.attr(name)?;
        attr::attr_kind(&self.tag, name).map(|k| v.to_svg_string(k))
    }

    /// `SetAttributeValue(name, string)`: parse against the declared type and store. `E_INVALIDARG`
    /// if the element does not carry `name`, or the value does not parse as its type.
    pub fn set_attribute(&mut self, name: &str, value: &str) -> Result<(), i32> {
        let kind = attr::attr_kind(&self.tag, name).ok_or(E_INVALIDARG)?;
        let parsed = attr::parse(kind, value).ok_or(E_INVALIDARG)?;
        self.set_parsed(name, parsed);
        Ok(())
    }

    fn set_parsed(&mut self, name: &str, value: Value) {
        if let Some(slot) = self.attrs.iter_mut().find(|(n, _)| n == name) {
            slot.1 = value;
        } else {
            self.attrs.push((name.to_string(), value));
        }
    }

    /// `RemoveAttribute`: returns whether the attribute was present.
    pub fn remove_attribute(&mut self, name: &str) -> bool {
        let before = self.attrs.len();
        self.attrs.retain(|(n, _)| n != name);
        self.attrs.len() != before
    }

    // Tree mutation (`AppendChild`/`InsertChildBefore`/`RemoveChild`). The DOM is an owned tree, so
    // children are moved in/out by value rather than ref-counted as in the COM object model.
    pub fn append_child(&mut self, child: Element) {
        self.children.push(child);
    }
    pub fn insert_child(&mut self, index: usize, child: Element) {
        let i = index.min(self.children.len());
        self.children.insert(i, child);
    }
    pub fn remove_child(&mut self, index: usize) -> Option<Element> {
        (index < self.children.len()).then(|| self.children.remove(index))
    }

    fn find_by_id<'a>(&'a self, id: &str) -> Option<&'a Element> {
        if let Some(Value::Str(v)) = self.attr("id") {
            if v == id {
                return Some(self);
            }
        }
        self.children.iter().find_map(|c| c.find_by_id(id))
    }
}

impl Document {
    /// `CreateSvgDocument`: parse `xml` into a DOM at `viewport` (DIPs). Empty input yields an empty
    /// `<svg>` root. A viewport with a non-positive dimension, malformed XML, or an unsupported /
    /// non-`svg` root element all fail with `E_INVALIDARG`.
    pub fn parse(xml: &str, viewport: [f32; 2]) -> Result<Document, i32> {
        if !(viewport[0] > 0.0) || !(viewport[1] > 0.0) {
            return Err(E_INVALIDARG);
        }
        let root = if xml.trim().is_empty() {
            Some(Element::new("svg"))
        } else {
            Some(parse_root(xml)?)
        };
        let mut doc = Document { viewport, root, next_id: 1, detached: Vec::new() };
        doc.assign_ids();
        Ok(doc)
    }

    pub fn root(&self) -> Option<&Element> {
        self.root.as_ref()
    }
    pub fn root_mut(&mut self) -> Option<&mut Element> {
        self.root.as_mut()
    }

    /// `FindElementById`: depth-first search by `id` attribute.
    pub fn find_by_id(&self, id: &str) -> Option<&Element> {
        self.root.as_ref().and_then(|r| r.find_by_id(id))
    }

    /// `Serialize`: the document (or `subtree`) back to XML, matching native's formatting — CRLF
    /// newlines, two-space indent, the root `svg` re-emitting the `xmlns`/`xmlns:xlink` declarations,
    /// and attributes in specified order reformatted from their typed values.
    pub fn serialize(&self) -> String {
        let mut out = String::new();
        if let Some(root) = &self.root {
            write_element(&mut out, root, 0, true);
        }
        out
    }
}

// --- mutable DOM by stable node id (the COM `ID2D1SvgElement` model) --------------------------
//
// `ID2D1SvgElement` is a shared, re-parentable handle; our tree is owned by value, so handles name
// a node by its stable [`Element::id`] and every edit routes through these `Document` methods. This
// keeps one source of truth the FFI COM wrappers and the conformance replay both forward to.

/// `D2DERR_INVALID_CALL` — native's result for text get/set on a non-text element.
pub const D2DERR_INVALID_CALL: i32 = -2003238902; // 0x8899000A

fn find_in(el: &Element, id: u32) -> Option<&Element> {
    if el.id == id {
        return Some(el);
    }
    el.children.iter().find_map(|c| find_in(c, id))
}
fn find_in_mut(el: &mut Element, id: u32) -> Option<&mut Element> {
    if el.id == id {
        return Some(el);
    }
    el.children.iter_mut().find_map(|c| find_in_mut(c, id))
}
fn subtree_contains(el: &Element, id: u32) -> bool {
    el.id == id || el.children.iter().any(|c| subtree_contains(c, id))
}
fn parent_in(el: &Element, id: u32) -> Option<u32> {
    if el.children.iter().any(|c| c.id == id) {
        return Some(el.id);
    }
    el.children.iter().find_map(|c| parent_in(c, id))
}
/// Remove the node `id` from `el`'s subtree, returning it owned.
fn detach_from(el: &mut Element, id: u32) -> Option<Element> {
    if let Some(pos) = el.children.iter().position(|c| c.id == id) {
        return Some(el.children.remove(pos));
    }
    el.children.iter_mut().find_map(|c| detach_from(c, id))
}

/// How a parsed [`Value`] lines up with an attribute's declared [`Kind`] — the POD/object set path
/// validates this, matching native's type-mismatch `E_INVALIDARG`.
fn value_matches(kind: Kind, v: &Value) -> bool {
    matches!(
        (kind, v),
        (Kind::Length, Value::Length(..))
            | (Kind::Float, Value::Float(_))
            | (Kind::Color, Value::Color(_))
            | (Kind::Matrix, Value::Matrix(_))
            | (Kind::ViewBox, Value::ViewBox(_))
            | (Kind::AspectRatio, Value::Aspect { .. })
            | (Kind::Id | Kind::Href | Kind::FuncIri, Value::Str(_))
            | (Kind::Paint, Value::Paint { .. })
            | (Kind::PathData, Value::Path { .. })
            | (Kind::Points, Value::Points(_))
            | (Kind::DashArray, Value::Dashes(_))
            | (
                Kind::FillRule
                    | Kind::Display
                    | Kind::Visibility
                    | Kind::Overflow
                    | Kind::LineCap
                    | Kind::LineJoin
                    | Kind::UnitType
                    | Kind::ExtendMode,
                Value::Enum(_)
            )
    )
}

impl Document {
    /// Assign a sequential id to every node (pre-order), seeding the allocator past the last.
    fn assign_ids(&mut self) {
        fn walk(el: &mut Element, next: &mut u32) {
            el.id = *next;
            *next += 1;
            for c in &mut el.children {
                walk(c, next);
            }
        }
        let mut next = 1;
        if let Some(r) = self.root.as_mut() {
            walk(r, &mut next);
        }
        self.next_id = next;
    }

    fn node(&self, id: u32) -> Option<&Element> {
        let Document { root, detached, .. } = self;
        root.as_ref()
            .and_then(|r| find_in(r, id))
            .or_else(|| detached.iter().find_map(|e| find_in(e, id)))
    }
    fn node_mut(&mut self, id: u32) -> Option<&mut Element> {
        let Document { root, detached, .. } = self;
        root.as_mut()
            .and_then(|r| find_in_mut(r, id))
            .or_else(|| detached.iter_mut().find_map(|e| find_in_mut(e, id)))
    }

    /// Remove node `id` from wherever it lives (tree or the detached pool), returning it owned.
    fn take(&mut self, id: u32) -> Option<Element> {
        let Document { root, detached, .. } = self;
        if let Some(r) = root.as_mut() {
            if let Some(e) = detach_from(r, id) {
                return Some(e);
            }
        }
        if let Some(pos) = detached.iter().position(|e| e.id == id) {
            return Some(detached.remove(pos));
        }
        detached.iter_mut().find_map(|e| detach_from(e, id))
    }

    pub fn root_id(&self) -> Option<u32> {
        self.root.as_ref().map(|r| r.id)
    }

    /// `FindElementById` → the node id, or `None` (native maps not-found to `E_INVALIDARG`).
    pub fn find_id(&self, id: &str) -> Option<u32> {
        self.find_by_id(id).map(|e| e.id())
    }

    pub fn exists(&self, id: u32) -> bool {
        self.node(id).is_some()
    }
    pub fn tag_of(&self, id: u32) -> Option<String> {
        self.node(id).map(|e| e.tag().to_string())
    }
    pub fn is_text_content(&self, id: u32) -> bool {
        self.node(id).is_some_and(|e| e.is_text_content())
    }
    pub fn has_children(&self, id: u32) -> bool {
        self.node(id).is_some_and(|e| !e.children.is_empty())
    }
    pub fn parent_id(&self, id: u32) -> Option<u32> {
        self.root.as_ref().and_then(|r| parent_in(r, id))
    }
    pub fn first_child_id(&self, id: u32) -> Option<u32> {
        self.node(id).and_then(|e| e.children.first().map(|c| c.id))
    }
    pub fn last_child_id(&self, id: u32) -> Option<u32> {
        self.node(id).and_then(|e| e.children.last().map(|c| c.id))
    }
    /// `GetNextChild(reference)` / `GetPreviousChild(reference)` — `delta` is +1 / -1.
    pub fn sibling_id(&self, parent: u32, reference: u32, delta: i32) -> Option<u32> {
        let p = self.node(parent)?;
        let i = p.children.iter().position(|c| c.id == reference)? as i32 + delta;
        (i >= 0).then(|| p.children.get(i as usize)).flatten().map(|c| c.id)
    }

    /// `CreateChild(tag)`: make a new node (any tag, as native permits) and append it to `parent`.
    pub fn create_child(&mut self, parent: u32, tag: &str) -> Result<u32, i32> {
        let id = self.next_id;
        let p = self.node_mut(parent).ok_or(E_INVALIDARG)?;
        let mut el = Element::new(tag);
        el.id = id;
        p.children.push(el);
        self.next_id += 1;
        Ok(id)
    }

    /// `AppendChild` / `InsertChildBefore`: re-parent `child` under `parent` (at the end, or before
    /// `reference`). Rejects self-parenting and cycles (`parent` inside `child`'s subtree).
    pub fn reparent(&mut self, parent: u32, child: u32, before: Option<u32>) -> Result<(), i32> {
        if child == parent || before == Some(child) || Some(child) == self.root_id() {
            return Err(E_INVALIDARG);
        }
        if !self.exists(child) || !self.exists(parent) || self.is_descendant(parent, child) {
            return Err(E_INVALIDARG);
        }
        if let Some(r) = before {
            if !self.node(parent).unwrap().children.iter().any(|c| c.id == r) {
                return Err(E_INVALIDARG);
            }
        }
        let el = self.take(child).ok_or(E_INVALIDARG)?;
        let p = self.node_mut(parent).unwrap();
        let i = before
            .and_then(|r| p.children.iter().position(|c| c.id == r))
            .unwrap_or(p.children.len());
        p.children.insert(i, el);
        Ok(())
    }

    /// `RemoveChild`: detach `child` of `parent` into the pool (a live handle may re-insert it).
    pub fn remove_child(&mut self, parent: u32, child: u32) -> Result<(), i32> {
        let p = self.node_mut(parent).ok_or(E_INVALIDARG)?;
        let pos = p.children.iter().position(|c| c.id == child).ok_or(E_INVALIDARG)?;
        let el = p.children.remove(pos);
        self.detached.push(el);
        Ok(())
    }

    /// `ReplaceChild(new, old)`: put `new` where `old` sat among `parent`'s children; `old` → pool.
    pub fn replace_child(&mut self, parent: u32, new: u32, old: u32) -> Result<(), i32> {
        if new == old || new == parent || Some(new) == self.root_id() {
            return Err(E_INVALIDARG);
        }
        if !self.exists(new) || self.is_descendant(parent, new) {
            return Err(E_INVALIDARG);
        }
        if !self.node(parent).ok_or(E_INVALIDARG)?.children.iter().any(|c| c.id == old) {
            return Err(E_INVALIDARG);
        }
        let el = self.take(new).ok_or(E_INVALIDARG)?;
        let p = self.node_mut(parent).unwrap();
        let pos = p.children.iter().position(|c| c.id == old).unwrap();
        let old = std::mem::replace(&mut p.children[pos], el);
        self.detached.push(old);
        Ok(())
    }

    fn is_descendant(&self, a: u32, b: u32) -> bool {
        a != b && self.node(b).is_some_and(|n| subtree_contains(n, a))
    }

    // Attributes.
    pub fn set_attr_string(&mut self, id: u32, name: &str, value: &str) -> Result<(), i32> {
        self.node_mut(id).ok_or(E_INVALIDARG)?.set_attribute(name, value)
    }
    pub fn attr_string(&self, id: u32, name: &str) -> Option<String> {
        self.node(id).and_then(|e| e.get_attribute(name))
    }
    pub fn set_attr_value(&mut self, id: u32, name: &str, value: Value) -> Result<(), i32> {
        let e = self.node_mut(id).ok_or(E_INVALIDARG)?;
        let kind = attr::attr_kind(&e.tag, name).ok_or(E_INVALIDARG)?;
        if !value_matches(kind, &value) {
            return Err(E_INVALIDARG);
        }
        e.set_parsed(name, value);
        Ok(())
    }
    pub fn attr_value(&self, id: u32, name: &str) -> Option<Value> {
        self.node(id).and_then(|e| e.attr(name).cloned())
    }
    pub fn attr_kind_of(&self, id: u32, name: &str) -> Option<Kind> {
        self.node(id).and_then(|e| attr::attr_kind(&e.tag, name))
    }
    pub fn remove_attr(&mut self, id: u32, name: &str) -> bool {
        self.node_mut(id).is_some_and(|e| e.remove_attribute(name))
    }
    pub fn specified_count(&self, id: u32) -> u32 {
        self.node(id).map_or(0, |e| e.specified_attribute_count() as u32)
    }
    pub fn specified_name(&self, id: u32, index: u32) -> Option<String> {
        self.node(id).and_then(|e| e.specified_attribute_name(index as usize)).map(str::to_string)
    }
    pub fn is_specified(&self, id: u32, name: &str) -> bool {
        self.node(id).is_some_and(|e| e.is_attribute_specified(name))
    }

    // Text content.
    pub fn set_text_value(&mut self, id: u32, text: &str) -> Result<(), i32> {
        let e = self.node_mut(id).ok_or(E_INVALIDARG)?;
        if e.is_text_content() {
            // A text node: the value is stored here.
            e.text = text.to_string();
            Ok(())
        } else {
            // Any element node (including a created <title>/<desc> container, which has no text-node
            // child yet) rejects with E_INVALIDARG — native accepts SetTextValue only on the text node
            // itself, not on the container (verified on the VM: svgdom-10 op 4 returns 0x80070057).
            Err(E_INVALIDARG)
        }
    }
    pub fn text_value(&self, id: u32) -> Option<String> {
        self.node(id).filter(|e| e.is_text_content()).map(|e| e.text().to_string())
    }

    pub fn set_viewport(&mut self, w: f32, h: f32) -> Result<(), i32> {
        if !(w > 0.0) || !(h > 0.0) {
            return Err(E_INVALIDARG);
        }
        self.viewport = [w, h];
        Ok(())
    }

    /// `Serialize(subtree)`: the XML for one node (a non-root subtree drops the root's xmlns/indent).
    pub fn serialize_from(&self, id: u32) -> Option<String> {
        let is_root = Some(id) == self.root_id();
        let node = self.node(id)?;
        let mut out = String::new();
        write_element(&mut out, node, 0, is_root);
        Some(if is_root { out } else { out.trim_start_matches("\r\n").to_string() })
    }

    /// `Deserialize`: graft a freshly parsed document's root into this one (re-id'd, detached until a
    /// `SetRoot`), returning the new root node id — exactly native's "returns an element you set as
    /// root" flow.
    pub fn import_root(&mut self, parsed: Document) -> u32 {
        fn reid(el: &mut Element, next: &mut u32) {
            el.id = *next;
            *next += 1;
            for c in &mut el.children {
                reid(c, next);
            }
        }
        let mut root = parsed.root.unwrap_or_else(|| Element::new("svg"));
        let mut next = self.next_id;
        reid(&mut root, &mut next);
        self.next_id = next;
        let id = root.id;
        self.detached.push(root);
        id
    }

    /// `SetRoot`: promote node `id` to the document root (the old root is discarded).
    pub fn set_root(&mut self, id: u32) -> Result<(), i32> {
        if Some(id) == self.root_id() {
            return Ok(());
        }
        let el = self.take(id).ok_or(E_INVALIDARG)?;
        self.root = Some(el);
        Ok(())
    }
}

// --- parse ------------------------------------------------------------------------------------

const XLINK_NS: &str = "http://www.w3.org/1999/xlink";

fn parse_root(xml: &str) -> Result<Element, i32> {
    let doc = roxmltree::Document::parse(xml).map_err(|_| E_INVALIDARG)?;
    let node = doc.root_element();
    let tag = node.tag_name().name();
    if tag != "svg" {
        return Err(E_INVALIDARG);
    }
    Ok(read_element(node))
}

fn read_element(node: roxmltree::Node) -> Element {
    let tag = node.tag_name().name().to_string();
    let mut el = Element::new(&tag);
    let mut inline_style: Option<String> = None;

    for a in node.attributes() {
        let local = a.name();
        let ns = a.namespace();
        let name = if local == "href" && ns == Some(XLINK_NS) {
            "xlink:href".to_string()
        } else if ns.is_some() {
            continue; // prefixed attribute in another namespace → ignored
        } else {
            local.to_string()
        };
        if name == "style" {
            inline_style = Some(a.value().to_string());
        } else if let Some(kind) = attr::attr_kind(&tag, &name) {
            if let Some(v) = attr::parse(kind, a.value()) {
                el.set_parsed(&name, v);
            }
        }
    }

    if let Some(style) = inline_style {
        apply_inline_style(&mut el, &tag, &style);
    }

    for child in node.children() {
        if child.is_element() {
            let child_tag = child.tag_name().name();
            if attr::is_supported_element(child_tag) && attr::can_have_child(&tag, child_tag) {
                el.children.push(read_element(child));
            }
        } else if (child.is_text() || child.node_type() == roxmltree::NodeType::Text)
            && attr::can_have_child(&tag, "")
        {
            if let Some(text) = child.text() {
                let mut t = Element::new("");
                t.text = text.to_string();
                el.children.push(t);
            }
        }
    }
    el
}

fn apply_inline_style(el: &mut Element, tag: &str, style: &str) {
    for decl in style.split(';') {
        if let Some(colon) = decl.find(':') {
            let name = attr::trim(&decl[..colon]);
            let value = attr::trim(&decl[colon + 1..]);
            if let Some(kind) = attr::attr_kind(tag, name) {
                if let Some(v) = attr::parse(kind, value) {
                    el.set_parsed(name, v);
                }
            }
        }
    }
}

// --- serialize --------------------------------------------------------------------------------

fn write_element(out: &mut String, el: &Element, depth: usize, is_root: bool) {
    if el.is_text_content() {
        append_escaped_text(out, &el.text);
        return;
    }
    if !is_root {
        out.push_str("\r\n");
    }
    for _ in 0..depth {
        out.push_str("  ");
    }
    out.push('<');
    out.push_str(&el.tag);
    // Native emits attributes alphabetically by name, with the root's xmlns declarations sorted in
    // among them (verified against the WARP oracle).
    let mut pairs: Vec<(&str, String)> = el
        .attrs
        .iter()
        .filter_map(|(name, value)| attr::attr_kind(&el.tag, name).map(|k| (name.as_str(), value.to_svg_string(k))))
        .collect();
    if is_root && el.tag == "svg" {
        pairs.push(("xmlns", "http://www.w3.org/2000/svg".to_string()));
        pairs.push(("xmlns:xlink", "http://www.w3.org/1999/xlink".to_string()));
    }
    pairs.sort_by(|a, b| a.0.cmp(b.0));
    for (name, value) in &pairs {
        out.push(' ');
        out.push_str(name);
        out.push_str("=\"");
        append_escaped_attr(out, value);
        out.push('"');
    }
    if el.tag == "title" || el.tag == "desc" {
        if el.children.is_empty() {
            out.push_str(" />");
            return;
        }
        out.push('>');
        for c in &el.children {
            if c.is_text_content() {
                append_escaped_text(out, &c.text);
            }
        }
        out.push_str("</");
        out.push_str(&el.tag);
        out.push('>');
    } else {
        out.push('>');
        for c in &el.children {
            write_element(out, c, depth + 1, false);
        }
        out.push_str("\r\n");
        for _ in 0..depth {
            out.push_str("  ");
        }
        out.push_str("</");
        out.push_str(&el.tag);
        out.push('>');
    }
}

fn append_escaped_text(out: &mut String, text: &str) {
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(c),
        }
    }
}

fn append_escaped_attr(out: &mut String, value: &str) {
    for c in value.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
}
