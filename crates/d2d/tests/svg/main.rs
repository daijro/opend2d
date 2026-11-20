//! Phase 2E self-checks for the SVG DOM: parse, typed attributes, the mutable DOM API, and
//! serialization. These assert the behaviour ported from native's SVG document/string handling
//! (from the native build, with the managed classes as a guide). Pixel/render parity is measured by the
//! project meter in `conformance.rs` against the WARP oracle; serialization byte-parity vs native
//! cannot be measured through the frozen harness (no DOM query channel) — see the task report.

use d2d::svg::{attr, Document, PaintType, Units, Value};

fn parse(xml: &str) -> Document {
    Document::parse(xml, [128.0, 96.0]).expect("parse")
}

#[test]
fn parses_supported_tree() {
    let doc = parse(
        "<svg xmlns='http://www.w3.org/2000/svg' width='128' height='96'>\
         <rect x='10' y='10' width='60' height='40' fill='#3050c0'/>\
         <circle cx='90' cy='60' r='25' fill='#c05030'/></svg>",
    );
    let root = doc.root().unwrap();
    assert_eq!(root.tag(), "svg");
    assert_eq!(root.children().len(), 2);
    assert_eq!(root.children()[0].tag(), "rect");
    assert_eq!(root.children()[1].tag(), "circle");
    // width/height stored as user-unit lengths.
    assert_eq!(root.attr("width"), Some(&Value::Length(128.0, Units::Number)));
    // #3050c0 → straight sRGB floats.
    match root.children()[0].attr("fill") {
        Some(Value::Paint { ty: PaintType::Color, color, .. }) => {
            assert_eq!(*color, [0x30 as f32 / 255.0, 0x50 as f32 / 255.0, 0xc0 as f32 / 255.0, 1.0]);
        }
        other => panic!("fill = {other:?}"),
    }
}

#[test]
fn unsupported_and_namespaced_attrs_are_inert() {
    let doc = parse(
        "<svg xmlns='http://www.w3.org/2000/svg' width='1' height='1'>\
         <rect x='0' y='0' width='1' height='1' bogus='7' data-x='9'/>\
         <frobnicate/></svg>",
    );
    let root = doc.root().unwrap();
    // The unknown <frobnicate> element is dropped; only <rect> remains.
    assert_eq!(root.children().len(), 1);
    let rect = &root.children()[0];
    assert!(!rect.is_attribute_specified("bogus"));
    assert!(!rect.is_attribute_specified("data-x"));
    assert!(rect.is_attribute_specified("width"));
}

#[test]
fn non_svg_root_and_malformed_are_invalid() {
    assert_eq!(Document::parse("<g/>", [1.0, 1.0]).unwrap_err(), d2d::svg::E_INVALIDARG);
    assert_eq!(Document::parse("<svg><rect></svg>", [1.0, 1.0]).unwrap_err(), d2d::svg::E_INVALIDARG);
    assert_eq!(Document::parse("<svg/>", [0.0, 1.0]).unwrap_err(), d2d::svg::E_INVALIDARG);
}

#[test]
fn find_by_id_depth_first() {
    let doc = parse(
        "<svg xmlns='http://www.w3.org/2000/svg' width='1' height='1'>\
         <g id='outer'><rect id='inner' width='1' height='1'/></g></svg>",
    );
    assert_eq!(doc.find_by_id("inner").unwrap().tag(), "rect");
    assert_eq!(doc.find_by_id("outer").unwrap().tag(), "g");
    assert!(doc.find_by_id("missing").is_none());
}

#[test]
fn set_attribute_validates_type() {
    let mut doc = parse("<svg xmlns='http://www.w3.org/2000/svg' width='1' height='1'><rect/></svg>");
    let rect = &mut doc.root_mut().unwrap().remove_child(0).unwrap();
    assert!(rect.set_attribute("width", "40").is_ok());
    assert_eq!(rect.attr("width"), Some(&Value::Length(40.0, Units::Number)));
    // fill-rule only accepts its enum spellings.
    assert!(rect.set_attribute("fill-rule", "evenodd").is_ok());
    assert_eq!(rect.set_attribute("fill-rule", "bogus").unwrap_err(), d2d::svg::E_INVALIDARG);
    // rect carries no cx attribute.
    assert_eq!(rect.set_attribute("cx", "3").unwrap_err(), d2d::svg::E_INVALIDARG);
    assert!(rect.remove_attribute("width"));
    assert!(!rect.is_attribute_specified("width"));
}

#[test]
fn mutable_dom_api() {
    use d2d::svg::Element;
    let mut doc = parse("<svg xmlns='http://www.w3.org/2000/svg' width='1' height='1'><g/></svg>");
    let root = doc.root_mut().unwrap();

    // GetAttributeValue reflects a set, in SVG-normalized form.
    root.set_attribute("fill", "red").unwrap();
    assert_eq!(root.get_attribute("fill").as_deref(), Some("#FF0000"));
    assert_eq!(root.specified_attribute_count(), 3); // width, height, fill
    // Native exposes specified attributes alphabetically (oracle-verified), so index 2 is "width".
    assert_eq!(root.specified_attribute_name(0), Some("fill"));
    assert_eq!(root.specified_attribute_name(2), Some("width"));

    // AppendChild / InsertChildBefore / RemoveChild on the owned tree.
    root.append_child(Element::new("rect"));
    root.insert_child(0, Element::new("circle"));
    assert_eq!(root.children().iter().map(|c| c.tag()).collect::<Vec<_>>(), ["circle", "g", "rect"]);
    let removed = root.remove_child(1).unwrap();
    assert_eq!(removed.tag(), "g");
    assert_eq!(root.children().len(), 2);

    // Text content on a <desc>.
    let mut desc = Element::new("desc");
    desc.set_text("hi");
    assert_eq!(desc.text(), "hi");
}

#[test]
fn serialize_normalizes_and_formats() {
    // Named/short-hex colors normalize to #RRGGBB; the root re-emits the namespace declarations.
    let doc = parse(
        "<svg xmlns='http://www.w3.org/2000/svg' width='2' height='2'>\
         <rect x='0' y='0' width='2' height='2' fill='blue' fill-rule='evenodd'/></svg>",
    );
    let xml = doc.serialize();
    // Native drives an IXmlWriter and calls WriteIndent (CRLF + 2 spaces/level) before every end
    // tag, so even an attribute-only element emits a full `</rect>` rather than self-closing.
    // Attributes serialize alphabetically (oracle-verified: native keeps them sorted by name).
    let expected = "<svg height=\"2\" width=\"2\" xmlns=\"http://www.w3.org/2000/svg\" \
        xmlns:xlink=\"http://www.w3.org/1999/xlink\">\r\n  \
        <rect fill=\"#0000FF\" fill-rule=\"evenodd\" height=\"2\" width=\"2\" x=\"0\" y=\"0\">\r\n  \
        </rect>\r\n</svg>";
    assert_eq!(xml, expected);
}

#[test]
fn color_parsing_forms() {
    let c = |s: &str| match attr::parse(attr::Kind::Color, s) {
        Some(Value::Color(c)) => c,
        _ => panic!("color {s}"),
    };
    assert_eq!(c("#abc"), [0xaa as f32 / 255.0, 0xbb as f32 / 255.0, 0xcc as f32 / 255.0, 1.0]);
    assert_eq!(c("#aAbBcC"), [0xaa as f32 / 255.0, 0xbb as f32 / 255.0, 0xcc as f32 / 255.0, 1.0]);
    assert_eq!(c("red"), [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(c("rgb(255, 0, 128)"), [1.0, 0.0, 128.0 / 255.0, 1.0]);
    assert_eq!(c("rgb(100%, 0%, 0%)"), [1.0, 0.0, 0.0, 1.0]);
    assert!(attr::parse(attr::Kind::Color, "notacolor").is_none());
}

#[test]
fn path_data_round_trips() {
    let v = attr::parse(attr::Kind::PathData, "M10 10 H 20 V20 L 0,0 Z").expect("path");
    let Value::Path { cmds, data } = &v else { panic!("not a path") };
    use attr::PathCmd::*;
    assert_eq!(cmds, &[MoveAbs, HorizAbs, VertAbs, LineAbs, ClosePath]);
    assert_eq!(data, &[10.0, 10.0, 20.0, 20.0, 0.0, 0.0]);
    // Serialization uses the canonical letters and comma/space layout.
    assert_eq!(v.to_svg_string(attr::Kind::PathData), "M10,10H20V20L0,0Z");
}

#[test]
fn transform_composition_order() {
    // translate then scale: SVG `a b` composes as a*b (a applied to points first).
    let Some(Value::Matrix(m)) = attr::parse(attr::Kind::Matrix, "translate(10,20) scale(2)") else {
        panic!("matrix");
    };
    // point (1,1) → scale → (2,2) → translate → (12,22)
    assert_eq!([1.0 * m[0] + 1.0 * m[2] + m[4], 1.0 * m[1] + 1.0 * m[3] + m[5]], [12.0, 22.0]);
}
