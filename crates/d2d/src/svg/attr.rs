//! SVG attribute schema, value types, and the string↔value parsing/formatting that native
//! native's SVG element/string handling performs. Values are parsed on set and reformatted on get/serialize,
//! so serialization normalizes (e.g. `fill="blue"` → `#0000FF`), exactly as native does.
//!
//! Ported fresh from the behaviour of native d2d1 (from the native build) and the managed SVG strings
//! reference; `ref/` code is a guide only.

/// `D2D1_SVG_LENGTH` units we accept in presentation attributes (`px` is treated as a bare number).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Units {
    Number,
    Percentage,
}

/// `D2D1_SVG_PAINT_TYPE`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PaintType {
    None = 0,
    Color = 1,
    CurrentColor = 2,
    Uri = 3,
    UriNone = 4,
    UriColor = 5,
    UriCurrentColor = 6,
}

/// Which parser/formatter an attribute uses — mirrors `SvgAttributeKind`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Length,
    Float,
    Color,
    Matrix,
    FillRule,
    Display,
    Visibility,
    Overflow,
    LineCap,
    LineJoin,
    UnitType,
    ExtendMode,
    ViewBox,
    AspectRatio,
    Id,
    Href,
    FuncIri,
    Paint,
    PathData,
    Points,
    DashArray,
}

/// A parsed attribute value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Length(f32, Units),
    Float(f32),
    Color([f32; 4]),
    Matrix([f32; 6]),
    /// `D2D1_SVG_*` enum value (fill-rule, line cap/join, display, …). Interpretation is by `Kind`.
    Enum(i32),
    ViewBox([f32; 4]),
    Aspect { defer: bool, align: i32, slice: bool },
    Str(String), // id / href / funcIRI (stored verbatim after a trim)
    Paint { ty: PaintType, color: [f32; 4], id: String },
    Path { cmds: Vec<PathCmd>, data: Vec<f32> },
    Points(Vec<[f32; 2]>),
    Dashes(Vec<(f32, Units)>),
}

/// SVG path command letters, kept as their native `D2D1_SVG_PATH_COMMAND` discriminants so
/// serialization and `CreatePathGeometry` round-trip bit-for-bit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PathCmd {
    ClosePath = 0,
    MoveAbs = 1,
    MoveRel = 2,
    LineAbs = 3,
    LineRel = 4,
    CubicAbs = 5,
    CubicRel = 6,
    QuadAbs = 7,
    QuadRel = 8,
    ArcAbs = 9,
    ArcRel = 10,
    HorizAbs = 11,
    HorizRel = 12,
    VertAbs = 13,
    VertRel = 14,
    CubicSmoothAbs = 15,
    CubicSmoothRel = 16,
    QuadSmoothAbs = 17,
    QuadSmoothRel = 18,
}

impl PathCmd {
    pub fn arg_count(self) -> usize {
        use PathCmd::*;
        match self {
            ClosePath => 0,
            HorizAbs | HorizRel | VertAbs | VertRel => 1,
            MoveAbs | MoveRel | LineAbs | LineRel | QuadSmoothAbs | QuadSmoothRel => 2,
            CubicSmoothAbs | CubicSmoothRel | QuadAbs | QuadRel => 4,
            CubicAbs | CubicRel => 6,
            ArcAbs | ArcRel => 7,
        }
    }
    fn letter(self) -> char {
        use PathCmd::*;
        match self {
            ClosePath => 'Z',
            MoveAbs => 'M',
            MoveRel => 'm',
            LineAbs => 'L',
            LineRel => 'l',
            CubicAbs => 'C',
            CubicRel => 'c',
            QuadAbs => 'Q',
            QuadRel => 'q',
            ArcAbs => 'A',
            ArcRel => 'a',
            HorizAbs => 'H',
            HorizRel => 'h',
            VertAbs => 'V',
            VertRel => 'v',
            CubicSmoothAbs => 'S',
            CubicSmoothRel => 's',
            QuadSmoothAbs => 'T',
            QuadSmoothRel => 't',
        }
    }
}

// --- schema (element → attribute kinds, allowed children) -------------------------------------

/// Presentation attributes shared by every drawable element.
fn presentation_kind(name: &str) -> Option<Kind> {
    Some(match name {
        "fill" | "stroke" => Kind::Paint,
        "fill-opacity" | "stroke-opacity" | "stroke-miterlimit" | "opacity" | "stop-opacity" => Kind::Float,
        "fill-rule" | "clip-rule" => Kind::FillRule,
        "stroke-width" | "stroke-dashoffset" => Kind::Length,
        "stroke-linecap" => Kind::LineCap,
        "stroke-linejoin" => Kind::LineJoin,
        "stroke-dasharray" => Kind::DashArray,
        "color" | "stop-color" => Kind::Color,
        "display" => Kind::Display,
        "visibility" => Kind::Visibility,
        "overflow" => Kind::Overflow,
        "clip-path" => Kind::FuncIri,
        _ => return None,
    })
}

/// `true` if `tag` is a supported SVG element.
pub fn is_supported_element(tag: &str) -> bool {
    matches!(
        tag,
        "svg" | "g" | "defs" | "use" | "rect" | "circle" | "ellipse" | "line" | "path" | "polyline"
            | "polygon" | "image" | "linearGradient" | "radialGradient" | "stop" | "clipPath"
            | "title" | "desc"
    )
}

/// The declared kind of `name` on `tag`, or `None` if the element does not carry that attribute.
pub fn attr_kind(tag: &str, name: &str) -> Option<Kind> {
    if name == "id" {
        return is_supported_element(tag).then_some(Kind::Id);
    }
    // Element-specific attributes.
    let specific = match (tag, name) {
        ("svg", "x") | ("svg", "y") | ("svg", "width") | ("svg", "height") => Some(Kind::Length),
        ("svg", "viewBox") => Some(Kind::ViewBox),
        ("svg", "preserveAspectRatio") => Some(Kind::AspectRatio),
        ("g", "transform") | ("defs", "transform") | ("clipPath", "transform") => Some(Kind::Matrix),
        ("use", "x") | ("use", "y") | ("use", "width") | ("use", "height") => Some(Kind::Length),
        ("use", "transform") => Some(Kind::Matrix),
        ("use", "xlink:href") => Some(Kind::Href),
        ("rect", "x") | ("rect", "y") | ("rect", "width") | ("rect", "height") | ("rect", "rx")
        | ("rect", "ry") => Some(Kind::Length),
        ("rect", "transform") => Some(Kind::Matrix),
        ("circle", "cx") | ("circle", "cy") | ("circle", "r") => Some(Kind::Length),
        ("circle", "transform") => Some(Kind::Matrix),
        ("ellipse", "cx") | ("ellipse", "cy") | ("ellipse", "rx") | ("ellipse", "ry") => Some(Kind::Length),
        ("ellipse", "transform") => Some(Kind::Matrix),
        ("line", "x1") | ("line", "y1") | ("line", "x2") | ("line", "y2") => Some(Kind::Length),
        ("line", "transform") => Some(Kind::Matrix),
        ("path", "d") => Some(Kind::PathData),
        ("path", "transform") => Some(Kind::Matrix),
        ("polyline", "points") | ("polygon", "points") => Some(Kind::Points),
        ("polyline", "transform") | ("polygon", "transform") => Some(Kind::Matrix),
        ("image", "x") | ("image", "y") | ("image", "width") | ("image", "height") => Some(Kind::Length),
        ("image", "transform") => Some(Kind::Matrix),
        ("image", "preserveAspectRatio") => Some(Kind::AspectRatio),
        ("image", "xlink:href") => Some(Kind::Href),
        ("linearGradient", "x1") | ("linearGradient", "y1") | ("linearGradient", "x2")
        | ("linearGradient", "y2") => Some(Kind::Length),
        ("radialGradient", "cx") | ("radialGradient", "cy") | ("radialGradient", "r")
        | ("radialGradient", "fx") | ("radialGradient", "fy") => Some(Kind::Length),
        ("linearGradient", "gradientUnits") | ("radialGradient", "gradientUnits") => Some(Kind::UnitType),
        ("linearGradient", "gradientTransform") | ("radialGradient", "gradientTransform") => Some(Kind::Matrix),
        ("linearGradient", "spreadMethod") | ("radialGradient", "spreadMethod") => Some(Kind::ExtendMode),
        ("linearGradient", "xlink:href") | ("radialGradient", "xlink:href") => Some(Kind::Href),
        ("stop", "offset") => Some(Kind::Length),
        _ => None,
    };
    if specific.is_some() {
        return specific;
    }
    // Presentation attributes: every element above except title/desc carries them.
    if matches!(tag, "title" | "desc") {
        None
    } else if is_supported_element(tag) {
        presentation_kind(name)
    } else {
        None
    }
}

/// `true` if `child` (empty string = text content) may be a child of `parent`.
pub fn can_have_child(parent: &str, child: &str) -> bool {
    match parent {
        "title" | "desc" => child.is_empty(),
        "svg" | "g" | "defs" => matches!(
            child,
            "svg" | "g" | "defs" | "use" | "rect" | "circle" | "ellipse" | "line" | "path"
                | "polyline" | "polygon" | "image" | "linearGradient" | "radialGradient"
                | "clipPath" | "title" | "desc"
        ),
        "linearGradient" | "radialGradient" => matches!(child, "stop" | "title" | "desc"),
        "clipPath" => matches!(
            child,
            "use" | "rect" | "circle" | "ellipse" | "line" | "path" | "polyline" | "polygon"
                | "title" | "desc"
        ),
        "use" | "rect" | "circle" | "ellipse" | "line" | "path" | "polyline" | "polygon" | "image"
        | "stop" => matches!(child, "title" | "desc"),
        _ => false,
    }
}

pub fn allows_text_content(tag: &str) -> bool {
    matches!(tag, "title" | "desc")
}

// --- parsing (string → Value) -----------------------------------------------------------------

/// Parse `text` as `kind`; `None` means native would reject it (attribute left unset / `E_INVALIDARG`).
pub fn parse(kind: Kind, text: &str) -> Option<Value> {
    match kind {
        Kind::Length => parse_length(text).map(|(v, u)| Value::Length(v, u)),
        Kind::Float => parse_float(text).map(Value::Float),
        Kind::Color => parse_color(text).map(Value::Color),
        Kind::Matrix => parse_transform(text).map(Value::Matrix),
        Kind::FillRule => parse_enum(kind, text).map(Value::Enum),
        Kind::Display | Kind::Visibility | Kind::Overflow | Kind::LineCap | Kind::LineJoin
        | Kind::UnitType | Kind::ExtendMode => parse_enum(kind, text).map(Value::Enum),
        Kind::ViewBox => parse_view_box(text).map(Value::ViewBox),
        Kind::AspectRatio => parse_aspect(text),
        Kind::Id | Kind::Href => Some(Value::Str(text.to_string())),
        Kind::FuncIri => Some(Value::Str(trim(text).to_string())),
        Kind::Paint => parse_paint(text),
        Kind::PathData => parse_path(text),
        Kind::Points => parse_points(text).map(Value::Points),
        Kind::DashArray => parse_dashes(text).map(Value::Dashes),
    }
}

pub fn trim(s: &str) -> &str {
    s.trim_matches([' ', '\t', '\r', '\n', '\u{0C}'])
}

fn is_ws(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r' | '\n' | '\u{0C}')
}

/// A cursor over SVG number/token text, mirroring `SvgTokenReader`.
struct Reader<'a> {
    t: &'a [u8],
    p: usize,
}
impl<'a> Reader<'a> {
    fn new(s: &'a str) -> Self {
        Reader { t: s.as_bytes(), p: 0 }
    }
    fn at_end(&self) -> bool {
        self.p >= self.t.len()
    }
    fn cur(&self) -> u8 {
        self.t[self.p]
    }
    fn skip_ws(&mut self) {
        while self.p < self.t.len() && is_ws(self.t[self.p] as char) {
            self.p += 1;
        }
    }
    fn can_read_number(&self) -> bool {
        let mut i = self.p;
        while i < self.t.len() && is_ws(self.t[i] as char) {
            i += 1;
        }
        if i >= self.t.len() {
            return false;
        }
        let c = self.t[i];
        c.is_ascii_digit() || c == b'+' || c == b'-' || c == b'.'
    }
    fn can_read_number_or_comma(&self) -> bool {
        let mut i = self.p;
        while i < self.t.len() && is_ws(self.t[i] as char) {
            i += 1;
        }
        if i >= self.t.len() {
            return false;
        }
        let c = self.t[i];
        c.is_ascii_digit() || c == b'+' || c == b'-' || c == b'.' || c == b','
    }
    fn try_consume(&mut self, c: u8) -> bool {
        if !self.at_end() && self.cur() == c {
            self.p += 1;
            true
        } else {
            false
        }
    }
    /// `SvgTokenReader.TryConsumeCommaWhitespace`: ws, one optional comma, ws; true if anything moved.
    fn comma_ws(&mut self) -> bool {
        let start = self.p;
        self.skip_ws();
        self.try_consume(b',');
        self.skip_ws();
        self.p != start
    }
    fn read_number(&mut self) -> Option<f32> {
        self.skip_ws();
        let start = self.p;
        let mut i = self.p;
        if i < self.t.len() && (self.t[i] == b'+' || self.t[i] == b'-') {
            i += 1;
        }
        let mut int_digits = 0;
        while i < self.t.len() && self.t[i].is_ascii_digit() {
            i += 1;
            int_digits += 1;
        }
        let mut frac_digits = 0;
        if i < self.t.len() && self.t[i] == b'.' {
            i += 1;
            while i < self.t.len() && self.t[i].is_ascii_digit() {
                i += 1;
                frac_digits += 1;
            }
        }
        if int_digits == 0 && frac_digits == 0 {
            return None;
        }
        if i < self.t.len() && (self.t[i] == b'e' || self.t[i] == b'E') {
            let mut j = i + 1;
            if j < self.t.len() && (self.t[j] == b'+' || self.t[j] == b'-') {
                j += 1;
            }
            let mut exp_digits = 0;
            while j < self.t.len() && self.t[j].is_ascii_digit() {
                j += 1;
                exp_digits += 1;
            }
            if exp_digits > 0 {
                i = j;
            }
        }
        let s = std::str::from_utf8(&self.t[start..i]).ok()?;
        let v: f32 = s.parse().ok()?;
        self.p = i;
        Some(v)
    }
    fn read_flag(&mut self) -> Option<f32> {
        self.skip_ws();
        if !self.at_end() && (self.cur() == b'0' || self.cur() == b'1') {
            let v = (self.cur() - b'0') as f32;
            self.p += 1;
            Some(v)
        } else {
            None
        }
    }
    fn read_ident(&mut self) -> Option<&'a str> {
        self.skip_ws();
        let start = self.p;
        while !self.at_end() && (self.cur().is_ascii_alphabetic() || self.cur() == b'-') {
            self.p += 1;
        }
        if self.p > start {
            std::str::from_utf8(&self.t[start..self.p]).ok()
        } else {
            None
        }
    }
    fn read_command_letter(&mut self) -> Option<u8> {
        self.skip_ws();
        if !self.at_end() && self.cur().is_ascii_alphabetic() {
            let c = self.cur();
            self.p += 1;
            Some(c)
        } else {
            None
        }
    }
}

fn parse_float(text: &str) -> Option<f32> {
    let mut r = Reader::new(text);
    let v = r.read_number()?;
    r.skip_ws();
    r.at_end().then_some(v)
}

fn parse_length(text: &str) -> Option<(f32, Units)> {
    let mut r = Reader::new(text);
    let v = r.read_number()?;
    let units = if r.try_consume(b'%') {
        Units::Percentage
    } else if r.try_consume(b'p') {
        // `px` suffix is accepted and dropped; a bare `p` is rejected.
        if !r.try_consume(b'x') {
            return None;
        }
        Units::Number
    } else {
        Units::Number
    };
    r.skip_ws();
    r.at_end().then_some((v, units))
}

fn hex_digit(c: u8) -> Option<i32> {
    match c {
        b'0'..=b'9' => Some((c - b'0') as i32),
        b'a'..=b'f' => Some((c - b'a' + 10) as i32),
        b'A'..=b'F' => Some((c - b'A' + 10) as i32),
        _ => None,
    }
}

fn from_bytes(r: i32, g: i32, b: i32) -> [f32; 4] {
    [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0]
}

fn parse_color(text: &str) -> Option<[f32; 4]> {
    let s = trim(text);
    let b = s.as_bytes();
    if b.is_empty() {
        return None;
    }
    if b[0] == b'#' {
        let h = &b[1..];
        if h.len() == 3 {
            let (r, g, bl) = (hex_digit(h[0])?, hex_digit(h[1])?, hex_digit(h[2])?);
            return Some(from_bytes((r << 4) | r, (g << 4) | g, (bl << 4) | bl));
        }
        if h.len() == 6 {
            let r = (hex_digit(h[0])? << 4) | hex_digit(h[1])?;
            let g = (hex_digit(h[2])? << 4) | hex_digit(h[3])?;
            let bl = (hex_digit(h[4])? << 4) | hex_digit(h[5])?;
            return Some(from_bytes(r, g, bl));
        }
        return None;
    }
    if s.starts_with("rgb(") && s.ends_with(')') {
        let inner = &s[4..s.len() - 1];
        let mut r = Reader::new(inner);
        let mut v = [0f32; 3];
        let mut percent = false;
        for i in 0..3 {
            if i > 0 && !r.comma_ws() {
                return None;
            }
            v[i] = r.read_number()?;
            let p = r.try_consume(b'%');
            if i == 0 {
                percent = p;
            } else if p != percent {
                return None;
            }
        }
        r.skip_ws();
        if !r.at_end() {
            return None;
        }
        let comp = |x: f32| {
            if percent {
                round_half_up(x.clamp(0.0, 100.0) * 255.0 / 100.0) as i32
            } else {
                round_half_up(x).clamp(0.0, 255.0) as i32
            }
        };
        return Some(from_bytes(comp(v[0]), comp(v[1]), comp(v[2])));
    }
    named_color(s)
}

/// `MathF.Round(..., MidpointRounding.AwayFromZero)`.
fn round_half_up(x: f32) -> f32 {
    x.round()
}

fn parse_paint(text: &str) -> Option<Value> {
    let s = trim(text);
    if s == "none" {
        return Some(Value::Paint { ty: PaintType::None, color: [0.0; 4], id: String::new() });
    }
    if s == "currentColor" {
        return Some(Value::Paint { ty: PaintType::CurrentColor, color: [0.0; 4], id: String::new() });
    }
    if s.starts_with("url(") {
        let close = s.find(')')?;
        let iri = trim(&s[4..close]);
        if !iri.starts_with('#') {
            return None;
        }
        let id = iri[1..].to_string();
        let rest = trim(&s[close + 1..]);
        let (ty, color) = if rest.is_empty() {
            (PaintType::Uri, [0.0; 4])
        } else if rest == "none" {
            (PaintType::UriNone, [0.0; 4])
        } else if rest == "currentColor" {
            (PaintType::UriCurrentColor, [0.0; 4])
        } else if let Some(c) = parse_color(rest) {
            (PaintType::UriColor, c)
        } else {
            return None;
        };
        return Some(Value::Paint { ty, color, id });
    }
    parse_color(s).map(|c| Value::Paint { ty: PaintType::Color, color: c, id: String::new() })
}

fn parse_transform(text: &str) -> Option<[f32; 6]> {
    let mut r = Reader::new(text);
    r.skip_ws();
    let mut matrix = crate::geometry::IDENTITY;
    let mut any = false;
    while !r.at_end() {
        if any {
            r.comma_ws();
            if r.at_end() {
                break;
            }
        }
        let ident = r.read_ident()?.to_string();
        r.skip_ws();
        if !r.try_consume(b'(') {
            return None;
        }
        let mut args = [0f32; 6];
        let mut n = 0;
        r.skip_ws();
        while !r.try_consume(b')') {
            if n > 0 {
                r.comma_ws();
            }
            if n >= 6 {
                return None;
            }
            args[n] = r.read_number()?;
            n += 1;
            r.skip_ws();
        }
        let a = transform_op(&ident, &args[..n])?;
        matrix = if any { crate::geometry::mat_mul(&a, &matrix) } else { a };
        any = true;
        r.skip_ws();
    }
    any.then_some(matrix)
}

fn transform_op(name: &str, args: &[f32]) -> Option<[f32; 6]> {
    let deg = std::f32::consts::PI / 180.0;
    Some(match (name, args.len()) {
        ("matrix", 6) => [args[0], args[1], args[2], args[3], args[4], args[5]],
        ("translate", 1) => [1.0, 0.0, 0.0, 1.0, args[0], 0.0],
        ("translate", 2) => [1.0, 0.0, 0.0, 1.0, args[0], args[1]],
        ("scale", 1) => [args[0], 0.0, 0.0, args[0], 0.0, 0.0],
        ("scale", 2) => [args[0], 0.0, 0.0, args[1], 0.0, 0.0],
        ("rotate", 1) => rotation(args[0]),
        ("rotate", 3) => {
            // RotationAt(angle, cx, cy) = T(-c) · R · T(c).
            let r = rotation(args[0]);
            let (cx, cy) = (args[1], args[2]);
            let t1 = [1.0, 0.0, 0.0, 1.0, -cx, -cy];
            let t2 = [1.0, 0.0, 0.0, 1.0, cx, cy];
            crate::geometry::mat_mul(&crate::geometry::mat_mul(&t1, &r), &t2)
        }
        ("skewX", 1) => [1.0, 0.0, libm::tanf(args[0] * deg), 1.0, 0.0, 0.0],
        ("skewY", 1) => [1.0, libm::tanf(args[0] * deg), 0.0, 1.0, 0.0, 0.0],
        _ => return None,
    })
}

fn rotation(angle_deg: f32) -> [f32; 6] {
    let a = angle_deg * std::f32::consts::PI / 180.0;
    let (s, c) = (crate::crt::system_crt::sinf(a), crate::crt::system_crt::cosf(a));
    [c, s, -s, c, 0.0, 0.0]
}

fn parse_points(text: &str) -> Option<Vec<[f32; 2]>> {
    let mut r = Reader::new(text);
    r.skip_ws();
    let mut pts = Vec::new();
    while !r.at_end() {
        if !pts.is_empty() {
            r.comma_ws();
            if r.at_end() {
                break;
            }
        }
        let x = r.read_number()?;
        if !r.comma_ws() && !r.can_read_number() {
            return None;
        }
        let y = r.read_number()?;
        pts.push([x, y]);
        r.skip_ws();
    }
    Some(pts)
}

fn parse_dashes(text: &str) -> Option<Vec<(f32, Units)>> {
    let s = trim(text);
    if s == "none" {
        return Some(Vec::new());
    }
    let mut r = Reader::new(s);
    r.skip_ws();
    let mut out = Vec::new();
    while !r.at_end() {
        if !out.is_empty() {
            r.comma_ws();
            if r.at_end() {
                break;
            }
        }
        let v = r.read_number()?;
        let u = if r.try_consume(b'%') { Units::Percentage } else { Units::Number };
        out.push((v, u));
        r.skip_ws();
    }
    Some(out)
}

fn parse_view_box(text: &str) -> Option<[f32; 4]> {
    let mut r = Reader::new(text);
    let mut v = [0f32; 4];
    for i in 0..4 {
        if i > 0 {
            r.comma_ws();
        }
        v[i] = r.read_number()?;
    }
    r.skip_ws();
    r.at_end().then_some(v)
}

fn parse_aspect(text: &str) -> Option<Value> {
    let mut r = Reader::new(text);
    r.skip_ws();
    let mut ident = r.read_ident()?;
    let mut defer = false;
    if ident == "defer" {
        defer = true;
        r.skip_ws();
        ident = r.read_ident()?;
    }
    let align = match ident {
        "none" => 0,
        "xMinYMin" => 1,
        "xMidYMin" => 2,
        "xMaxYMin" => 3,
        "xMinYMid" => 4,
        "xMidYMid" => 5,
        "xMaxYMid" => 6,
        "xMinYMax" => 7,
        "xMidYMax" => 8,
        "xMaxYMax" => 9,
        _ => return None,
    };
    let mut slice = false;
    r.skip_ws();
    if let Some(id) = r.read_ident() {
        match id {
            "meet" => slice = false,
            "slice" => slice = true,
            _ => return None,
        }
    }
    r.skip_ws();
    r.at_end().then_some(Value::Aspect { defer, align, slice })
}

fn parse_enum(kind: Kind, text: &str) -> Option<i32> {
    let s = trim(text);
    Some(match kind {
        Kind::FillRule => match s {
            "nonzero" => 1,
            "evenodd" => 0,
            _ => return None,
        },
        Kind::Display => match s {
            "none" => 1,
            "inline" | "block" | "list-item" | "run-in" | "compact" | "marker" | "table"
            | "inline-table" | "table-row-group" | "table-header-group" | "table-footer-group"
            | "table-row" | "table-column-group" | "table-column" | "table-cell"
            | "table-caption" | "inline-block" => 0,
            _ => return None,
        },
        Kind::Visibility => match s {
            "visible" => 0,
            "hidden" | "collapse" => 1,
            _ => return None,
        },
        Kind::Overflow => match s {
            "visible" => 0,
            "hidden" => 1,
            _ => return None,
        },
        Kind::LineCap => match s {
            "butt" => 0,
            "square" => 1,
            "round" => 2,
            _ => return None,
        },
        Kind::LineJoin => match s {
            "bevel" => 1,
            "round" => 2,
            "miter" => 3,
            _ => return None,
        },
        Kind::UnitType => match s {
            "userSpaceOnUse" => 0,
            "objectBoundingBox" => 1,
            _ => return None,
        },
        Kind::ExtendMode => match s {
            "pad" => 0,
            "repeat" => 1,
            "reflect" => 2,
            _ => return None,
        },
        _ => return None,
    })
}

fn parse_path(text: &str) -> Option<Value> {
    let mut r = Reader::new(text);
    r.skip_ws();
    let mut cmds: Vec<PathCmd> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    let mut started = false;
    while !r.at_end() {
        let letter = r.read_command_letter()?;
        let rel = letter.is_ascii_lowercase();
        let upper = letter.to_ascii_uppercase();
        if !started && upper != b'M' {
            return None;
        }
        match upper {
            b'Z' => {
                cmds.push(PathCmd::ClosePath);
                r.skip_ws();
            }
            b'M' => {
                let mut first = true;
                loop {
                    if !first {
                        r.comma_ws();
                        if !r.can_read_number() {
                            break;
                        }
                    }
                    let x = match r.read_number() {
                        Some(v) => v,
                        // native `return !flag4`: a first-coordinate failure fails the whole parse,
                        // a later one stops with what was read so far.
                        None if first => return None,
                        None => return Some(Value::Path { cmds, data }),
                    };
                    r.comma_ws();
                    let y = r.read_number()?;
                    cmds.push(if first {
                        if rel { PathCmd::MoveRel } else { PathCmd::MoveAbs }
                    } else if rel {
                        PathCmd::LineRel
                    } else {
                        PathCmd::LineAbs
                    });
                    data.push(x);
                    data.push(y);
                    first = false;
                    started = true;
                    r.skip_ws();
                    if !r.can_read_number_or_comma() {
                        break;
                    }
                }
            }
            b'A' => {
                let mut first = true;
                loop {
                    if !first {
                        r.comma_ws();
                        if !r.can_read_number() {
                            break;
                        }
                    }
                    let rx = match r.read_number() {
                        Some(v) => v,
                        None if first => return None,
                        None => return Some(Value::Path { cmds, data }),
                    };
                    r.comma_ws();
                    let ry = r.read_number()?;
                    r.comma_ws();
                    let rot = r.read_number()?;
                    r.comma_ws();
                    let large = r.read_flag()?;
                    r.comma_ws();
                    let sweep = r.read_flag()?;
                    r.comma_ws();
                    let x = r.read_number()?;
                    r.comma_ws();
                    let y = r.read_number()?;
                    cmds.push(if rel { PathCmd::ArcRel } else { PathCmd::ArcAbs });
                    data.extend_from_slice(&[rx, ry, rot, large, sweep, x, y]);
                    first = false;
                    r.skip_ws();
                    if !r.can_read_number_or_comma() {
                        break;
                    }
                }
            }
            b'L' | b'T' | b'H' | b'V' | b'C' | b'S' | b'Q' => {
                let cmd = match (upper, rel) {
                    (b'L', false) => PathCmd::LineAbs,
                    (b'L', true) => PathCmd::LineRel,
                    (b'T', false) => PathCmd::QuadSmoothAbs,
                    (b'T', true) => PathCmd::QuadSmoothRel,
                    (b'H', false) => PathCmd::HorizAbs,
                    (b'H', true) => PathCmd::HorizRel,
                    (b'V', false) => PathCmd::VertAbs,
                    (b'V', true) => PathCmd::VertRel,
                    (b'C', false) => PathCmd::CubicAbs,
                    (b'C', true) => PathCmd::CubicRel,
                    (b'S', false) => PathCmd::CubicSmoothAbs,
                    (b'S', true) => PathCmd::CubicSmoothRel,
                    (b'Q', false) => PathCmd::QuadAbs,
                    (_, _) => PathCmd::QuadRel,
                };
                if !parse_number_groups(&mut r, cmd, &mut cmds, &mut data) {
                    return None;
                }
                started = true;
            }
            _ => return None,
        }
    }
    if !started {
        return cmds.is_empty().then_some(Value::Path { cmds, data });
    }
    Some(Value::Path { cmds, data })
}

fn parse_number_groups(r: &mut Reader, cmd: PathCmd, cmds: &mut Vec<PathCmd>, data: &mut Vec<f32>) -> bool {
    let n = cmd.arg_count();
    let mut first = true;
    loop {
        if !first {
            r.comma_ws();
            if !r.can_read_number() {
                break;
            }
        }
        let mut group = [0f32; 6];
        for (i, slot) in group.iter_mut().enumerate().take(n) {
            if i > 0 {
                r.comma_ws();
            }
            match r.read_number() {
                Some(v) => *slot = v,
                None => return !first && i == 0,
            }
        }
        cmds.push(cmd);
        data.extend_from_slice(&group[..n]);
        first = false;
        r.skip_ws();
        if !r.can_read_number_or_comma() {
            break;
        }
    }
    true
}

// --- formatting (Value → string, for get/serialize) -------------------------------------------

impl Value {
    pub fn to_svg_string(&self, kind: Kind) -> String {
        match self {
            Value::Length(v, u) => write_length(*v, *u),
            Value::Float(v) => write_float(*v),
            Value::Color(c) => write_color(*c),
            Value::Matrix(m) => write_matrix(m),
            Value::Enum(v) => write_enum(kind, *v).unwrap_or_default(),
            Value::ViewBox(vb) => {
                format!("{} {} {} {}", write_float(vb[0]), write_float(vb[1]), write_float(vb[2]), write_float(vb[3]))
            }
            Value::Aspect { defer, align, slice } => write_aspect(*defer, *align, *slice),
            Value::Str(s) => s.clone(),
            Value::Paint { ty, color, id } => write_paint(*ty, *color, id),
            Value::Path { cmds, data } => write_path(cmds, data),
            Value::Points(pts) => write_points(pts),
            Value::Dashes(d) => write_dashes(d),
        }
    }
}

/// `.NET float.ToString("G6")` for the common finite range, plus native's infinity/NaN spellings.
pub fn write_float(v: f32) -> String {
    if v.is_nan() {
        return "-1.#IND".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { "1.#INF" } else { "-1.#INF" }.to_string();
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let neg = v < 0.0;
    let a = v.abs();
    let exp = libm::log10f(a).floor() as i32;
    // ponytail: G6 scientific range (exp<-4 || exp>=6) is rare for SVG coords and not oracle-checked
    // here; fall back to a round-trippable fixed form. Revisit if a scene needs 1e6+ coordinates.
    let decimals = (5 - exp).max(0) as usize;
    let mut s = format!("{a:.decimals$}");
    if s.contains('.') {
        s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    }
    if neg {
        format!("-{s}")
    } else {
        s
    }
}

fn write_length(v: f32, u: Units) -> String {
    let s = write_float(v);
    if u == Units::Percentage {
        format!("{s}%")
    } else {
        s
    }
}

fn write_color(c: [f32; 4]) -> String {
    let b = |x: f32| round_half_up(x * 255.0).clamp(0.0, 255.0) as i32;
    format!("#{:02X}{:02X}{:02X}", b(c[0]), b(c[1]), b(c[2]))
}

fn write_matrix(m: &[f32; 6]) -> String {
    format!(
        "matrix({} {} {} {} {} {})",
        write_float(m[0]),
        write_float(m[1]),
        write_float(m[2]),
        write_float(m[3]),
        write_float(m[4]),
        write_float(m[5])
    )
}

fn write_paint(ty: PaintType, color: [f32; 4], id: &str) -> String {
    match ty {
        PaintType::None => "none".to_string(),
        PaintType::Color => write_color(color),
        PaintType::CurrentColor => "currentColor".to_string(),
        PaintType::Uri => format!("url(#{id})"),
        PaintType::UriNone => format!("url(#{id}) none"),
        PaintType::UriColor => format!("url(#{id}) {}", write_color(color)),
        PaintType::UriCurrentColor => format!("url(#{id}) currentColor"),
    }
}

fn write_points(pts: &[[f32; 2]]) -> String {
    let mut s = String::new();
    for (i, p) in pts.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        s.push_str(&write_float(p[0]));
        s.push(',');
        s.push_str(&write_float(p[1]));
    }
    s
}

fn write_dashes(d: &[(f32, Units)]) -> String {
    let mut s = String::new();
    for (i, (v, u)) in d.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        s.push_str(&write_length(*v, *u));
    }
    s
}

fn write_aspect(defer: bool, align: i32, slice: bool) -> String {
    let a = match align {
        0 => "none",
        1 => "xMinYMin",
        2 => "xMidYMin",
        3 => "xMaxYMin",
        4 => "xMinYMid",
        5 => "xMidYMid",
        6 => "xMaxYMid",
        7 => "xMinYMax",
        8 => "xMidYMax",
        9 => "xMaxYMax",
        _ => "none",
    };
    let base = if defer { format!("defer {a}") } else { a.to_string() };
    if slice {
        format!("{base} slice")
    } else {
        format!("{base} meet")
    }
}

fn write_enum(kind: Kind, v: i32) -> Option<String> {
    Some(match kind {
        Kind::FillRule => match v {
            1 => "nonzero",
            0 => "evenodd",
            _ => return None,
        },
        Kind::Display => match v {
            0 => "inline",
            1 => "none",
            _ => return None,
        },
        Kind::Visibility => match v {
            0 => "visible",
            1 => "hidden",
            _ => return None,
        },
        Kind::Overflow => match v {
            0 => "visible",
            1 => "hidden",
            _ => return None,
        },
        Kind::LineCap => match v {
            0 => "butt",
            2 => "round",
            1 => "square",
            _ => return None,
        },
        Kind::LineJoin => match v {
            3 => "miter",
            2 => "round",
            1 => "bevel",
            _ => return None,
        },
        Kind::UnitType => match v {
            0 => "userSpaceOnUse",
            1 => "objectBoundingBox",
            _ => return None,
        },
        Kind::ExtendMode => match v {
            0 => "pad",
            1 => "repeat",
            2 => "reflect",
            _ => return None,
        },
        _ => return None,
    }
    .to_string())
}

fn write_path(cmds: &[PathCmd], data: &[f32]) -> String {
    let mut s = String::new();
    let mut n = 0usize;
    for &c in cmds {
        let count = c.arg_count();
        if n + count > data.len() {
            break;
        }
        s.push(c.letter());
        let args = &data[n..n + count];
        match c {
            PathCmd::HorizAbs | PathCmd::HorizRel | PathCmd::VertAbs | PathCmd::VertRel => {
                s.push_str(&write_float(args[0]));
            }
            PathCmd::ArcAbs | PathCmd::ArcRel => {
                s.push_str(&write_float(args[0]));
                s.push(',');
                s.push_str(&write_float(args[1]));
                s.push(' ');
                s.push_str(&write_float(args[2]));
                s.push(' ');
                s.push_str(&write_float(args[3]));
                s.push(' ');
                s.push_str(&write_float(args[4]));
                s.push(' ');
                s.push_str(&write_float(args[5]));
                s.push(',');
                s.push_str(&write_float(args[6]));
            }
            PathCmd::ClosePath => {}
            _ => {
                let mut j = 0;
                while j < count {
                    if j > 0 {
                        s.push(' ');
                    }
                    s.push_str(&write_float(args[j]));
                    s.push(',');
                    s.push_str(&write_float(args[j + 1]));
                    j += 2;
                }
            }
        }
        n += count;
    }
    s
}

include!("named_colors.rs");
