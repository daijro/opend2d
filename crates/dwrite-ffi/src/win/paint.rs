//! `IDWritePaintReader` — a pull cursor over a font's COLR v1 paint graph, the object
//! `IDWriteFontFace7::CreatePaintReader` hands back. It reflects the OpenType paint tree directly
//! (the same graph the core renderer walks in `d2d::text::color`, but returning DirectWrite's *raw*
//! paint values rather than the renderer's projected/ realized ones): `SetCurrentGlyph` resolves a
//! glyph's root paint, clip box and attributes, and `MoveToFirstChild`/`MoveToNextSibling`/
//! `MoveToParent` walk it. Colours resolve through CPAL at the selected palette (or the custom
//! palette / text colour).
//!
//! Only the non-variable COLR v1 paints are decoded here — exactly the set the core's static walker
//! (`color::paint`) handles; variable (`PaintVar*`) paints are left to a follow-up (see the `_ =>`
//! arm), matching the core, which defers those to skrifa's renderer.

use super::E_INVALIDARG;
use core::cell::RefCell;
use skrifa::raw::tables::colr::{ClipBox, Colr, ColorLine, Extend, Paint};
use skrifa::raw::TableProvider;
use skrifa::{FontRef, GlyphId};
use std::sync::Arc;
use windows::Win32::Graphics::Direct2D::Common::{D2D1_COLOR_F, D2D1_GRADIENT_STOP, D2D_RECT_F};
use windows::Win32::Graphics::DirectWrite::*;
use windows_core::{implement, Result as WResult, HRESULT};

/// `dwrite_3.h` `DWRITE_E_NOCOLOR` — the glyph has no colour (paint) representation.
const DWRITE_E_NOCOLOR: HRESULT = HRESULT(0x8898_500Cu32 as i32);
/// The CPAL/COLR sentinel palette index meaning "the current text (foreground) colour".
const FOREGROUND: u16 = 0xFFFF;

/// One resolved paint node in a glyph's flattened paint tree. `kind` carries the raw DirectWrite
/// values; `children` index back into the arena; `stops` holds a gradient's raw colour stops.
struct Node {
    kind: Kind,
    children: Vec<usize>,
    parent: Option<usize>,
    stops: Vec<Stop>,
}

struct Stop {
    offset: f32,
    palette_index: u16,
    alpha: f32,
}

enum Kind {
    Layers,
    Solid { palette_index: u16, alpha: f32 },
    SolidGlyph { glyph: u32, palette_index: u16, alpha: f32 },
    Linear { extend: u32, x0: f32, y0: f32, x1: f32, y1: f32, x2: f32, y2: f32 },
    Radial { extend: u32, x0: f32, y0: f32, r0: f32, x1: f32, y1: f32, r1: f32 },
    Sweep { extend: u32, cx: f32, cy: f32, start: f32, end: f32 },
    Glyph { glyph: u32 },
    ColorGlyph { glyph: u32, clip: D2D_RECT_F },
    Transform { m: [f32; 6] },
    Composite { mode: i32 },
}

// --- OpenType affine paints → a single DWRITE_MATRIX (same formulas as `color::paint`) -----------
fn scale(x: f32, y: f32, cx: f32, cy: f32) -> [f32; 6] {
    [x, 0.0, 0.0, y, cx - x * cx, cy - y * cy]
}
fn rotate(angle: f32, cx: f32, cy: f32) -> [f32; 6] {
    // COLR angle is in counter-clockwise half-turns (value × 180°); provenance: OpenType COLR spec.
    let a = angle * std::f32::consts::PI;
    let (s, c) = (a.sin(), a.cos());
    [c, s, -s, c, s * cy + (1.0 - c) * cx, -s * cx + (1.0 - c) * cy]
}
fn skew(x: f32, y: f32, cx: f32, cy: f32) -> [f32; 6] {
    let tan = |v: f32| (v * std::f32::consts::PI).tan();
    let (x, y) = (tan(x), tan(y));
    [1.0, y, -x, 1.0, x * cy, -y * cx]
}

fn extend_mode(e: Extend) -> u32 {
    // OpenType Pad/Repeat/Reflect → D2D1_EXTEND_MODE CLAMP(0)/WRAP(1)/MIRROR(2).
    match e {
        Extend::Pad => 0,
        Extend::Repeat => 1,
        Extend::Reflect => 2,
        _ => 0,
    }
}

fn clip_of(colr: &Colr, gid: GlyphId) -> D2D_RECT_F {
    // ponytail: an absent COLR ClipBox yields a zero rect; native derives it from the glyph's ink
    // bounds. Deriving bounds here needs the outline machinery — a VM-oracle follow-up.
    match colr.v1_clip_box(gid) {
        Some(ClipBox::Format1(c)) => D2D_RECT_F {
            left: c.x_min().to_i16() as f32,
            top: c.y_min().to_i16() as f32,
            right: c.x_max().to_i16() as f32,
            bottom: c.y_max().to_i16() as f32,
        },
        Some(ClipBox::Format2(c)) => D2D_RECT_F {
            left: c.x_min().to_i16() as f32,
            top: c.y_min().to_i16() as f32,
            right: c.x_max().to_i16() as f32,
            bottom: c.y_max().to_i16() as f32,
        },
        _ => D2D_RECT_F::default(),
    }
}

/// Flattens the OpenType paint graph into a `Node` arena. `budget` bounds total nodes so a cyclic
/// `PaintColrGlyph` reference terminates (mirrors `color::paint`'s `remaining`).
struct Builder<'a> {
    colr: Colr<'a>,
    nodes: Vec<Node>,
    budget: u32,
}

impl<'a> Builder<'a> {
    fn push(&mut self, kind: Kind, parent: Option<usize>, stops: Vec<Stop>) -> usize {
        self.nodes.push(Node { kind, children: Vec::new(), parent, stops });
        self.nodes.len() - 1
    }

    fn stops(&self, line: &ColorLine) -> Vec<Stop> {
        line.color_stops()
            .iter()
            .map(|s| Stop { offset: s.stop_offset().to_f32(), palette_index: s.palette_index(), alpha: s.alpha().to_f32() })
            .collect()
    }

    /// Build the subtree for `paint`, returning its node index.
    fn build(&mut self, paint: Paint<'a>, parent: Option<usize>, depth: u32) -> Option<usize> {
        if depth >= 64 {
            return None;
        }
        self.budget = self.budget.checked_sub(1)?;
        // A transform paint: push the TRANSFORM node, then its single child.
        macro_rules! xf {
            ($m:expr, $p:expr) => {{
                let idx = self.push(Kind::Transform { m: $m }, parent, Vec::new());
                let c = self.build($p.paint().ok()?, Some(idx), depth + 1)?;
                self.nodes[idx].children = vec![c];
                Some(idx)
            }};
        }
        match paint {
            Paint::ColrLayers(p) => {
                let idx = self.push(Kind::Layers, parent, Vec::new());
                let mut kids = Vec::new();
                for i in p.first_layer_index()..p.first_layer_index().checked_add(p.num_layers() as u32)? {
                    let (layer, _) = self.colr.v1_layer(i as usize)?;
                    kids.push(self.build(layer, Some(idx), depth + 1)?);
                }
                self.nodes[idx].children = kids;
                Some(idx)
            }
            Paint::Solid(p) => Some(self.push(Kind::Solid { palette_index: p.palette_index(), alpha: p.alpha().to_f32() }, parent, Vec::new())),
            Paint::LinearGradient(p) => {
                let line = p.color_line().ok()?;
                let stops = self.stops(&line);
                Some(self.push(
                    Kind::Linear {
                        extend: extend_mode(line.extend()),
                        x0: p.x0().to_i16() as f32,
                        y0: p.y0().to_i16() as f32,
                        x1: p.x1().to_i16() as f32,
                        y1: p.y1().to_i16() as f32,
                        x2: p.x2().to_i16() as f32,
                        y2: p.y2().to_i16() as f32,
                    },
                    parent,
                    stops,
                ))
            }
            Paint::RadialGradient(p) => {
                let line = p.color_line().ok()?;
                let stops = self.stops(&line);
                Some(self.push(
                    Kind::Radial {
                        extend: extend_mode(line.extend()),
                        x0: p.x0().to_i16() as f32,
                        y0: p.y0().to_i16() as f32,
                        r0: p.radius0().to_u16() as f32,
                        x1: p.x1().to_i16() as f32,
                        y1: p.y1().to_i16() as f32,
                        r1: p.radius1().to_u16() as f32,
                    },
                    parent,
                    stops,
                ))
            }
            Paint::SweepGradient(p) => {
                let line = p.color_line().ok()?;
                let stops = self.stops(&line);
                Some(self.push(
                    Kind::Sweep {
                        extend: extend_mode(line.extend()),
                        cx: p.center_x().to_i16() as f32,
                        cy: p.center_y().to_i16() as f32,
                        // ponytail: raw OpenType angle in degrees (value × 180). Whether native applies
                        // the +180 render offset here is VM-pending; the raw angle is the literal value.
                        start: p.start_angle().to_f32() * 180.0,
                        end: p.end_angle().to_f32() * 180.0,
                    },
                    parent,
                    stops,
                ))
            }
            Paint::Glyph(p) => {
                let child = p.paint().ok()?;
                let glyph = p.glyph_id().to_u16() as u32;
                // A glyph filled directly with a solid colour collapses to SOLID_GLYPH (the union has
                // no transform slot there, so only a *direct* PaintSolid child qualifies).
                if let Paint::Solid(s) = child {
                    Some(self.push(Kind::SolidGlyph { glyph, palette_index: s.palette_index(), alpha: s.alpha().to_f32() }, parent, Vec::new()))
                } else {
                    let idx = self.push(Kind::Glyph { glyph }, parent, Vec::new());
                    let c = self.build(child, Some(idx), depth + 1)?;
                    self.nodes[idx].children = vec![c];
                    Some(idx)
                }
            }
            Paint::ColrGlyph(p) => {
                let gid: GlyphId = p.glyph_id().into();
                let idx = self.push(Kind::ColorGlyph { glyph: gid.to_u32(), clip: clip_of(&self.colr, gid) }, parent, Vec::new());
                let (root, _) = self.colr.v1_base_glyph(gid)?;
                let c = self.build(root, Some(idx), depth + 1)?;
                self.nodes[idx].children = vec![c];
                Some(idx)
            }
            Paint::Composite(p) => {
                let idx = self.push(Kind::Composite { mode: p.composite_mode() as i32 }, parent, Vec::new());
                // Children in draw order: backdrop first, then source (same order the core composites).
                let backdrop = self.build(p.backdrop_paint().ok()?, Some(idx), depth + 1)?;
                let source = self.build(p.source_paint().ok()?, Some(idx), depth + 1)?;
                self.nodes[idx].children = vec![backdrop, source];
                Some(idx)
            }
            Paint::Transform(p) => {
                let m = p.transform().ok()?;
                xf!([m.xx().to_f32(), m.yx().to_f32(), m.xy().to_f32(), m.yy().to_f32(), m.dx().to_f32(), m.dy().to_f32()], p)
            }
            Paint::Translate(p) => xf!([1.0, 0.0, 0.0, 1.0, p.dx().to_i16() as f32, p.dy().to_i16() as f32], p),
            Paint::Scale(p) => xf!(scale(p.scale_x().to_f32(), p.scale_y().to_f32(), 0.0, 0.0), p),
            Paint::ScaleAroundCenter(p) => xf!(scale(p.scale_x().to_f32(), p.scale_y().to_f32(), p.center_x().to_i16() as f32, p.center_y().to_i16() as f32), p),
            Paint::ScaleUniform(p) => xf!(scale(p.scale().to_f32(), p.scale().to_f32(), 0.0, 0.0), p),
            Paint::ScaleUniformAroundCenter(p) => xf!(scale(p.scale().to_f32(), p.scale().to_f32(), p.center_x().to_i16() as f32, p.center_y().to_i16() as f32), p),
            Paint::Rotate(p) => xf!(rotate(p.angle().to_f32(), 0.0, 0.0), p),
            Paint::RotateAroundCenter(p) => xf!(rotate(p.angle().to_f32(), p.center_x().to_i16() as f32, p.center_y().to_i16() as f32), p),
            Paint::Skew(p) => xf!(skew(p.x_skew_angle().to_f32(), p.y_skew_angle().to_f32(), 0.0, 0.0), p),
            Paint::SkewAroundCenter(p) => xf!(skew(p.x_skew_angle().to_f32(), p.y_skew_angle().to_f32(), p.center_x().to_i16() as f32, p.center_y().to_i16() as f32), p),
            // ponytail: PaintVar* (variable) paints are deferred, mirroring the core's static walker
            // which hands those to skrifa. A glyph whose tree needs them fails SetCurrentGlyph.
            _ => None,
        }
    }
}

/// The reader's mutable state: colour inputs, the current glyph's flattened tree, and the cursor.
struct State {
    palette_index: u32,
    custom: Option<Vec<DWRITE_COLOR_F>>,
    text_color: DWRITE_COLOR_F,
    palette: Vec<DWRITE_COLOR_F>,
    tree: Vec<Node>,
    cursor: usize,
}

#[implement(IDWritePaintReader)]
pub(crate) struct PaintReaderObj {
    bytes: Arc<Vec<u8>>,
    index: u32,
    state: RefCell<State>,
}

/// Build the reader for a face's bytes (the caller validated the format/COLR level).
pub(crate) fn new_reader(bytes: Arc<Vec<u8>>, index: u32) -> IDWritePaintReader {
    PaintReaderObj {
        bytes,
        index,
        state: RefCell::new(State {
            palette_index: 0,
            custom: None,
            text_color: DWRITE_COLOR_F { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            palette: Vec::new(),
            tree: Vec::new(),
            cursor: 0,
        }),
    }
    .into()
}

/// The CPAL palette `index`'s entries (or the custom palette) as straight-alpha colours.
fn cpal_palette(font: &FontRef, index: u32) -> Vec<DWRITE_COLOR_F> {
    let Ok(cpal) = font.cpal() else { return Vec::new() };
    let Some(Ok(records)) = cpal.color_records_array() else { return Vec::new() };
    let entries = cpal.num_palette_entries() as usize;
    let indices = cpal.color_record_indices();
    let p = (index as usize).min(cpal.num_palettes().saturating_sub(1).max(0) as usize);
    let base = indices.get(p).map(|i| i.get() as usize).unwrap_or(0);
    (0..entries)
        .map(|e| match records.get(base + e) {
            Some(r) => DWRITE_COLOR_F { r: r.red() as f32 / 255.0, g: r.green() as f32 / 255.0, b: r.blue() as f32 / 255.0, a: r.alpha() as f32 / 255.0 },
            None => DWRITE_COLOR_F { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
        })
        .collect()
}

impl State {
    /// Resolve a paint colour: a CPAL/custom palette entry, or the text colour for the 0xFFFF
    /// sentinel, with the stop/solid alpha carried separately (as native does).
    fn paint_color(&self, palette_index: u16, alpha: f32) -> DWRITE_PAINT_COLOR {
        let (value, colorAttributes) = if palette_index == FOREGROUND {
            (self.text_color, DWRITE_PAINT_ATTRIBUTES_USES_TEXT_COLOR)
        } else {
            let value = self.palette.get(palette_index as usize).copied().unwrap_or(DWRITE_COLOR_F { r: 0.0, g: 0.0, b: 0.0, a: 0.0 });
            (value, DWRITE_PAINT_ATTRIBUTES_USES_PALETTE)
        };
        DWRITE_PAINT_COLOR { value, paletteEntryIndex: palette_index, alphaMultiplier: alpha, colorAttributes }
    }

    /// The DirectWrite paint element for node `i`, filling the tagged union per its paint type.
    fn element(&self, i: usize) -> DWRITE_PAINT_ELEMENT {
        let node = &self.tree[i];
        let stop_count = node.stops.len() as u32;
        let (paintType, paint) = match node.kind {
            Kind::Layers => (DWRITE_PAINT_TYPE_LAYERS, DWRITE_PAINT_ELEMENT_0 { layers: DWRITE_PAINT_ELEMENT_0_0 { childCount: node.children.len() as u32 } }),
            Kind::Solid { palette_index, alpha } => (DWRITE_PAINT_TYPE_SOLID, DWRITE_PAINT_ELEMENT_0 { solid: self.paint_color(palette_index, alpha) }),
            Kind::SolidGlyph { glyph, palette_index, alpha } => (
                DWRITE_PAINT_TYPE_SOLID_GLYPH,
                DWRITE_PAINT_ELEMENT_0 { solidGlyph: DWRITE_PAINT_ELEMENT_0_1 { glyphIndex: glyph, color: self.paint_color(palette_index, alpha) } },
            ),
            Kind::Linear { extend, x0, y0, x1, y1, x2, y2 } => (
                DWRITE_PAINT_TYPE_LINEAR_GRADIENT,
                DWRITE_PAINT_ELEMENT_0 { linearGradient: DWRITE_PAINT_ELEMENT_0_2 { extendMode: extend, gradientStopCount: stop_count, x0, y0, x1, y1, x2, y2 } },
            ),
            Kind::Radial { extend, x0, y0, r0, x1, y1, r1 } => (
                DWRITE_PAINT_TYPE_RADIAL_GRADIENT,
                DWRITE_PAINT_ELEMENT_0 { radialGradient: DWRITE_PAINT_ELEMENT_0_3 { extendMode: extend, gradientStopCount: stop_count, x0, y0, radius0: r0, x1, y1, radius1: r1 } },
            ),
            Kind::Sweep { extend, cx, cy, start, end } => (
                DWRITE_PAINT_TYPE_SWEEP_GRADIENT,
                DWRITE_PAINT_ELEMENT_0 { sweepGradient: DWRITE_PAINT_ELEMENT_0_4 { extendMode: extend, gradientStopCount: stop_count, centerX: cx, centerY: cy, startAngle: start, endAngle: end } },
            ),
            Kind::Glyph { glyph } => (DWRITE_PAINT_TYPE_GLYPH, DWRITE_PAINT_ELEMENT_0 { glyph: DWRITE_PAINT_ELEMENT_0_5 { glyphIndex: glyph } }),
            Kind::ColorGlyph { glyph, clip } => (DWRITE_PAINT_TYPE_COLOR_GLYPH, DWRITE_PAINT_ELEMENT_0 { colorGlyph: DWRITE_PAINT_ELEMENT_0_6 { glyphIndex: glyph, clipBox: clip } }),
            Kind::Transform { m } => (DWRITE_PAINT_TYPE_TRANSFORM, DWRITE_PAINT_ELEMENT_0 { transform: DWRITE_MATRIX { m11: m[0], m12: m[1], m21: m[2], m22: m[3], dx: m[4], dy: m[5] } }),
            Kind::Composite { mode } => (DWRITE_PAINT_TYPE_COMPOSITE, DWRITE_PAINT_ELEMENT_0 { composite: DWRITE_PAINT_ELEMENT_0_7 { mode: DWRITE_COLOR_COMPOSITE_MODE(mode) } }),
        };
        DWRITE_PAINT_ELEMENT { paintType, paint }
    }

    /// Whether the glyph's tree reads palette / text colours (for SetCurrentGlyph's attributes).
    fn attributes(&self) -> DWRITE_PAINT_ATTRIBUTES {
        let mut a = DWRITE_PAINT_ATTRIBUTES_NONE;
        let mut note = |pal: u16| {
            if pal == FOREGROUND {
                a |= DWRITE_PAINT_ATTRIBUTES_USES_TEXT_COLOR;
            } else {
                a |= DWRITE_PAINT_ATTRIBUTES_USES_PALETTE;
            }
        };
        for n in &self.tree {
            match n.kind {
                Kind::Solid { palette_index, .. } | Kind::SolidGlyph { palette_index, .. } => note(palette_index),
                _ => {}
            }
            for s in &n.stops {
                note(s.palette_index);
            }
        }
        a
    }
}

/// Copy `elem` into the caller's buffer, bounded by the version's `structsize`.
unsafe fn write_element(out: *mut DWRITE_PAINT_ELEMENT, structsize: u32, elem: DWRITE_PAINT_ELEMENT) {
    if out.is_null() {
        return;
    }
    let n = (structsize as usize).min(core::mem::size_of::<DWRITE_PAINT_ELEMENT>());
    unsafe { core::ptr::copy_nonoverlapping(&elem as *const _ as *const u8, out as *mut u8, n) };
}

impl IDWritePaintReader_Impl for PaintReaderObj_Impl {
    fn SetCurrentGlyph(&self, glyphindex: u32, paintelement: *mut DWRITE_PAINT_ELEMENT, structsize: u32, clipbox: *mut D2D_RECT_F, glyphattributes: *mut DWRITE_PAINT_ATTRIBUTES) -> WResult<()> {
        let font = FontRef::from_index(&self.bytes, self.index).map_err(|_| E_INVALIDARG)?;
        let colr = font.colr().map_err(|_| DWRITE_E_NOCOLOR)?;
        let gid = GlyphId::new(glyphindex);
        let (root_paint, _) = colr.v1_base_glyph(gid).ok_or(DWRITE_E_NOCOLOR)?;
        let clip = clip_of(&colr, gid);
        let mut builder = Builder { colr, nodes: Vec::new(), budget: 16384 };
        let root = builder.build(root_paint, None, 0).ok_or(DWRITE_E_NOCOLOR)?;

        let mut st = self.state.borrow_mut();
        st.palette = st.custom.clone().unwrap_or_else(|| cpal_palette(&font, st.palette_index));
        st.tree = builder.nodes;
        st.cursor = root;
        let elem = st.element(root);
        let attrs = st.attributes();
        unsafe {
            write_element(paintelement, structsize, elem);
            if !clipbox.is_null() {
                *clipbox = clip;
            }
            if !glyphattributes.is_null() {
                *glyphattributes = attrs;
            }
        }
        Ok(())
    }

    fn SetTextColor(&self, textcolor: *const DWRITE_COLOR_F) -> WResult<()> {
        if textcolor.is_null() {
            return Err(E_INVALIDARG.into());
        }
        self.state.borrow_mut().text_color = unsafe { *textcolor };
        Ok(())
    }

    fn SetColorPaletteIndex(&self, colorpaletteindex: u32) -> WResult<()> {
        let mut st = self.state.borrow_mut();
        st.palette_index = colorpaletteindex;
        st.custom = None;
        Ok(())
    }

    fn SetCustomColorPalette(&self, paletteentries: *const DWRITE_COLOR_F, paletteentrycount: u32) -> WResult<()> {
        if paletteentries.is_null() {
            return Err(E_INVALIDARG.into());
        }
        // Native requires at least the font's CPAL palette-entry count; fewer → E_INVALIDARG
        // (VM-verified).
        let needed = FontRef::from_index(&self.bytes, self.index).ok().and_then(|f| f.cpal().ok()).map(|c| c.num_palette_entries() as u32).unwrap_or(0);
        if paletteentrycount < needed {
            return Err(E_INVALIDARG.into());
        }
        let entries = unsafe { core::slice::from_raw_parts(paletteentries, paletteentrycount as usize) }.to_vec();
        self.state.borrow_mut().custom = Some(entries);
        Ok(())
    }

    fn MoveToFirstChild(&self, paintelement: *mut DWRITE_PAINT_ELEMENT, structsize: u32) -> WResult<()> {
        let mut st = self.state.borrow_mut();
        let child = *st.tree[st.cursor].children.first().ok_or(E_INVALIDARG)?;
        st.cursor = child;
        unsafe { write_element(paintelement, structsize, st.element(child)) };
        Ok(())
    }

    fn MoveToNextSibling(&self, paintelement: *mut DWRITE_PAINT_ELEMENT, structsize: u32) -> WResult<()> {
        let mut st = self.state.borrow_mut();
        let cursor = st.cursor;
        let parent = st.tree[cursor].parent.ok_or(E_INVALIDARG)?;
        let sibs = &st.tree[parent].children;
        let pos = sibs.iter().position(|&c| c == cursor).ok_or(E_INVALIDARG)?;
        let next = *sibs.get(pos + 1).ok_or(E_INVALIDARG)?;
        st.cursor = next;
        unsafe { write_element(paintelement, structsize, st.element(next)) };
        Ok(())
    }

    fn MoveToParent(&self) -> WResult<()> {
        let mut st = self.state.borrow_mut();
        let parent = st.tree[st.cursor].parent.ok_or(E_INVALIDARG)?;
        st.cursor = parent;
        Ok(())
    }

    fn GetGradientStops(&self, firstgradientstopindex: u32, gradientstopcount: u32, gradientstops: *mut D2D1_GRADIENT_STOP) -> WResult<()> {
        let st = self.state.borrow();
        let stops = &st.tree[st.cursor].stops;
        for j in 0..gradientstopcount as usize {
            let s = stops.get(firstgradientstopindex as usize + j).ok_or(E_INVALIDARG)?;
            let c = st.paint_color(s.palette_index, s.alpha);
            // The realized stop colour: the palette/text colour with the stop alpha folded in.
            let color = D2D1_COLOR_F { r: c.value.r, g: c.value.g, b: c.value.b, a: c.value.a * s.alpha };
            unsafe { *gradientstops.add(j) = D2D1_GRADIENT_STOP { position: s.offset, color } };
        }
        Ok(())
    }

    fn GetGradientStopColors(&self, firstgradientstopindex: u32, gradientstopcount: u32, gradientstopcolors: *mut DWRITE_PAINT_COLOR) -> WResult<()> {
        let st = self.state.borrow();
        let stops = &st.tree[st.cursor].stops;
        for j in 0..gradientstopcount as usize {
            let s = stops.get(firstgradientstopindex as usize + j).ok_or(E_INVALIDARG)?;
            unsafe { *gradientstopcolors.add(j) = st.paint_color(s.palette_index, s.alpha) };
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reader(file: &str) -> IDWritePaintReader {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../d2d/src/text/color/testdata/");
        let bytes = Arc::new(std::fs::read(format!("{path}{file}")).unwrap());
        new_reader(bytes, 0)
    }

    /// The first glyph whose paint tree resolves, plus its root element.
    fn first_color_glyph(r: &IDWritePaintReader) -> Option<(u32, DWRITE_PAINT_ELEMENT)> {
        for gid in 0..4000u32 {
            let mut el = DWRITE_PAINT_ELEMENT::default();
            let mut clip = D2D_RECT_F::default();
            let mut attrs = DWRITE_PAINT_ATTRIBUTES_NONE;
            let ok = unsafe { r.SetCurrentGlyph(gid, &mut el, core::mem::size_of::<DWRITE_PAINT_ELEMENT>() as u32, &mut clip, Some(&mut attrs)) };
            if ok.is_ok() {
                return Some((gid, el));
            }
        }
        None
    }

    #[test]
    fn walks_a_colrv1_paint_tree() {
        for file in ["colrtest.ttf", "colrtest3.ttf"] {
            let r = reader(file);
            let (_, root) = first_color_glyph(&r).unwrap_or_else(|| panic!("{file}: no colour glyph"));
            // The root is a real paint type (not NONE), and SetCurrentGlyph populated the union.
            assert_ne!(root.paintType, DWRITE_PAINT_TYPE_NONE, "{file}: root paint type");

            // Descending to the first child then back to the parent returns the root element.
            let mut child = DWRITE_PAINT_ELEMENT::default();
            let size = core::mem::size_of::<DWRITE_PAINT_ELEMENT>() as u32;
            if unsafe { r.MoveToFirstChild(&mut child, size) }.is_ok() {
                assert_ne!(child.paintType, DWRITE_PAINT_TYPE_NONE, "{file}: child paint type");
                unsafe { r.MoveToParent() }.unwrap();
                // Re-descending yields the same first child (cursor is back at the root).
                let mut again = DWRITE_PAINT_ELEMENT::default();
                unsafe { r.MoveToFirstChild(&mut again, size) }.unwrap();
                assert_eq!(again.paintType, child.paintType, "{file}: re-descend");
            }
        }
    }

    const SIZE: u32 = core::mem::size_of::<DWRITE_PAINT_ELEMENT>() as u32;

    /// Depth-first walk from the node the cursor sits on (holding its already-read `el`), restoring
    /// the cursor to that node before returning. Reads back the stops of the first gradient found.
    fn dfs(r: &IDWritePaintReader, el: &DWRITE_PAINT_ELEMENT, found: &mut bool) {
        if *found {
            return;
        }
        if matches!(el.paintType, DWRITE_PAINT_TYPE_LINEAR_GRADIENT | DWRITE_PAINT_TYPE_RADIAL_GRADIENT | DWRITE_PAINT_TYPE_SWEEP_GRADIENT) {
            let count = unsafe { el.paint.linearGradient.gradientStopCount };
            if count > 0 {
                let mut stops = vec![D2D1_GRADIENT_STOP::default(); count as usize];
                unsafe { r.GetGradientStops(0, &mut stops) }.unwrap();
                assert!(stops.iter().all(|s| s.position.is_finite()), "finite stop positions");
                *found = true;
                return;
            }
        }
        let mut child = DWRITE_PAINT_ELEMENT::default();
        if unsafe { r.MoveToFirstChild(&mut child, SIZE) }.is_err() {
            return; // leaf: cursor unchanged
        }
        let mut cur = child;
        loop {
            dfs(r, &cur, found);
            if *found {
                return;
            }
            let mut sib = DWRITE_PAINT_ELEMENT::default();
            if unsafe { r.MoveToNextSibling(&mut sib, SIZE) }.is_err() {
                break;
            }
            cur = sib;
        }
        unsafe { r.MoveToParent() }.unwrap(); // restore to the node we descended from
    }

    #[test]
    fn reads_gradient_stops_when_present() {
        // The COLRv1 test fonts exercise gradients; find one anywhere in either font's trees.
        let mut found = false;
        for file in ["colrtest.ttf", "colrtest3.ttf"] {
            let r = reader(file);
            for gid in 0..4000u32 {
                let mut el = DWRITE_PAINT_ELEMENT::default();
                let mut clip = D2D_RECT_F::default();
                let mut attrs = DWRITE_PAINT_ATTRIBUTES_NONE;
                if unsafe { r.SetCurrentGlyph(gid, &mut el, SIZE, &mut clip, Some(&mut attrs)) }.is_ok() {
                    dfs(&r, &el, &mut found);
                }
                if found {
                    break;
                }
            }
            if found {
                break;
            }
        }
        assert!(found, "a COLRv1 test font has at least one gradient leaf");
    }
}
