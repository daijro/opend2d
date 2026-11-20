//! Command list: `SetTarget(list)` records draws device-INDEPENDENTLY (the DIP inputs + the
//! record-time transform/state), `Close` seals it, `DrawImage(list)` replays them. The contract is
//! **replay at transform T ≡ direct drawing at transform T**: native's
//! command list records resolution-independent primitives and re-rasterizes them under the draw-time
//! transform, so a scaled / rotated / skewed `DrawImage(commandList)` matches drawing the same
//! primitives directly under that transform. Replay re-issues each recorded draw through the *same*
//! public draw method with the context transform set to `record_world · replay_transform`, so there is
//! no separate replay rasterizer — brushes re-realize ([`crate::brush::BrushDef`]), curves re-flatten
//! and strokes re-widen under the composed transform, exactly as a direct draw would.
//!
//! [`RasterCmd`] is the *device-resolved* primitive the direct rasterizer consumes (never recorded);
//! [`RecCmd`] is the device-independent recorded draw. A list is a shared COM image (`Rc<RefCell<..>>`).

use windows_core::HRESULT;

use crate::bitmap::Bitmap;
use crate::brush::Brush;
use crate::context::PixelBrush;
use crate::geometry::{Geometry, StrokeStyle};
use crate::resource::Domain;
use crate::state::DrawState;
use crate::text::{Glyph, RenderingParams};
use std::cell::RefCell;
use std::rc::Rc;

/// A device-resolved primitive handed to the rasterizer (`DeviceContext::raster`). Not recorded —
/// the command list stores [`RecCmd`] instead. `src` is premultiplied BGRA8 (UNORM8 target); `src_f`
/// the same colour premultiplied RGBA f32 (float/HDR target).
#[derive(Clone)]
pub(crate) enum RasterCmd {
    FillRect { rect: [f32; 4], src: [u8; 4], src_f: [f32; 4], aa: u32, blend: u32 },
    // `geometry`: true for a FillGeometry/DrawGeometry fill (native's -0.5 pixel-centre device shift on
    // the COPY trapezoid path), false for a FillRectangle primitive (no shift).
    /// `warp28`: an aliased stroke WARP's own aliased edge walk rasterizes ([`crate::pixels::warp_geom`],
    /// device points taken to 28.4 as d2d1 does — [`crate::pixels::Target::fill_warp28`]).
    FillPolys { figures: Vec<Vec<[f32; 2]>>, winding: bool, src: [u8; 4], src_f: [f32; 4], aa: u32, blend: u32, geometry: bool, warp28: bool },
}

/// A recorded glyph run's foreground: a solid colour or a per-pixel gradient/bitmap/image brush.
#[derive(Clone)]
pub(crate) enum TextFg {
    Solid([f32; 4]),
    Brush(Brush),
    // A colour-bitmap glyph run (`DrawColorBitmapGlyphRun`) and its accepted image formats.
    ColorBitmap(u32),
    // An OT-SVG glyph run (`DrawSvgGlyphRun`) and its context foreground.
    Svg([f32; 4]),
}

/// One recorded, device-independent draw: the DIP inputs plus the record-time [`DrawState`] (which
/// carries the world transform, AA, blend and unit mode) and DPI. Replay re-issues it through the
/// matching public draw method with the transform composed with the replay transform.
#[derive(Clone)]
pub(crate) enum RecCmd {
    // A `Clear` recorded into a command list: native records it (it does NOT latch WRONG_STATE like a
    // Clear on a bitmap target) and, on replay, fills the list's drawn area with `color` — the list is
    // an infinite image, so an opaque Clear covers the whole target it is drawn into. `color` is
    // straight (non-premultiplied) RGBA; replay re-issues `clear(color)`.
    Clear { state: DrawState, dpi: (f32, f32), color: [f32; 4] },
    FillRect { state: DrawState, dpi: (f32, f32), rect: [f32; 4], color: [f32; 4], opacity: f32 },
    FillPolys { state: DrawState, dpi: (f32, f32), figures: Vec<Vec<[f32; 2]>>, winding: bool, color: [f32; 4], opacity: f32 },
    FillGeom { state: DrawState, dpi: (f32, f32), geom: Geometry, color: [f32; 4], opacity: f32 },
    StrokeGeom { state: DrawState, dpi: (f32, f32), geom: Geometry, color: [f32; 4], opacity: f32, width: f32, style: StrokeStyle },
    FillRectBrush { state: DrawState, dpi: (f32, f32), rect: [f32; 4], brush: PixelBrush },
    FillGeomBrush { state: DrawState, dpi: (f32, f32), geom: Geometry, brush: PixelBrush },
    StrokeGeomBrush { state: DrawState, dpi: (f32, f32), geom: Geometry, brush: PixelBrush, width: f32, style: StrokeStyle },
    // A `DrawLine`, kept as a line so playback widens it as a direct DrawLine does, under the playback
    // transform (the rectangle fill of a plain line, else the line stroke).
    Line { state: DrawState, dpi: (f32, f32), p0: [f32; 2], p1: [f32; 2], width: f32, style: Option<StrokeStyle>, color: [f32; 4], opacity: f32 },
    LineBrush { state: DrawState, dpi: (f32, f32), p0: [f32; 2], p1: [f32; 2], width: f32, style: Option<StrokeStyle>, brush: PixelBrush },
    GlyphRun {
        state: DrawState,
        dpi: (f32, f32),
        font: Rc<[u8]>,
        face_index: u32,
        em_size: f32,
        glyphs: Rc<[Glyph]>,
        baseline_origin: [f32; 2],
        measuring_mode: u32,
        is_sideways: bool,
        bidi_level: u32,
        no_snap: bool,
        simulations: u32,
        text_aa: u32,
        params: RenderingParams,
        fg: TextFg,
        palette: Option<u32>,
    },
    // A `DrawImage(bitmap)`: the image, its DIP top-left `offset`, the optional image-rectangle
    // `source`, interpolation and composite mode. Re-issued under the composed transform (an affine
    // DrawImage re-samples the bitmap), so a scaled/rotated image draw replays ≡ direct.
    DrawImage { state: DrawState, dpi: (f32, f32), bitmap: Rc<Bitmap>, offset: [f32; 2], source: Option<[f32; 4]>, interpolation: u32, composite: u32 },
    // A non-SOURCE_OVER `DrawImage(commandList)` nested in an outer list, or one with an image rectangle: the inner
    // list, its DIP offset, image rectangle and composite mode. Replayed as that DrawImage under the transform.
    DrawImageList { state: DrawState, dpi: (f32, f32), list: CommandList, offset: [f32; 2], rect: Option<[f32; 4]>, composite: u32 },
    // A `DrawImage(effect)`: the realized output bitmap and its pre-transform top-left `dest0`, blitted
    // via `blit_effect_bitmap` under the composed transform. (The effect output is a realized raster;
    // the effect is not re-realized at the replay transform — enough for the recorded scenes.)
    DrawEffect { state: DrawState, dpi: (f32, f32), bitmap: Rc<Bitmap>, dest0: [f32; 2], offset: [f32; 2], interp: u32, composite: u32, finite: bool, pointwise: bool },
    // A `DrawImage(effect)` onto a FLOAT target: the realized f32 output, its DIP DrawImage `offset`
    // and the realize device-bounds top-left `b_lt`; replay re-places it at the composed transform.
    BlitFloat { state: DrawState, dpi: (f32, f32), px: Rc<Vec<[f32; 4]>>, w: i32, h: i32, offset: [f32; 2], b_lt: [i32; 2] },
    // A `DrawSpriteBatch`: the batch's sprites and bitmap, drawn by the sprite tessellator at playback under the
    // composed transform (which also decides there whether a sprite's corners overflow).
    SpriteBatch { state: DrawState, dpi: (f32, f32), sprites: Rc<Vec<crate::extras::Sprite>>, bitmap: Rc<Bitmap>, interpolation: u32, options: u32 },
    // A `DrawInk` with a solid colour or a brush, and its style, stroked at playback under the composed transform.
    Ink { state: DrawState, dpi: (f32, f32), start: crate::extras::ink::InkPoint, segments: Rc<[crate::extras::ink::InkBezierSegment]>, color: [f32; 4], opacity: f32, style: Option<crate::extras::ink::InkStyle> },
    InkBrush { state: DrawState, dpi: (f32, f32), start: crate::extras::ink::InkPoint, segments: Rc<[crate::extras::ink::InkBezierSegment]>, brush: PixelBrush, style: Option<crate::extras::ink::InkStyle> },
    // A `DrawGradientMesh`, tessellated at playback under the composed transform.
    GradientMesh { state: DrawState, dpi: (f32, f32), patches: Rc<[crate::extras::mesh::MeshPatch]> },
    // A `DrawBitmap`, drawn at playback under the composed transform.
    DrawBitmap { state: DrawState, dpi: (f32, f32), bitmap: Rc<Bitmap>, dest: [f32; 4], source: [f32; 4], opacity: f32, interpolation: u32 },
    // A `FillOpacityMask` with a solid colour or a brush, filled at playback under the composed transform.
    OpacityMask { state: DrawState, dpi: (f32, f32), bitmap: Rc<Bitmap>, color: [f32; 4], opacity: f32, content: u32, dest: Option<[f32; 4]>, source: Option<[f32; 4]> },
    OpacityMaskBrush { state: DrawState, dpi: (f32, f32), bitmap: Rc<Bitmap>, brush: PixelBrush, content: u32, dest: Option<[f32; 4]>, source: Option<[f32; 4]> },
    // A `FillMesh` triangle with a solid colour or a brush, filled at playback under the composed transform.
    Mesh { state: DrawState, dpi: (f32, f32), tri: [[f32; 2]; 3], color: [f32; 4], opacity: f32 },
    MeshBrush { state: DrawState, dpi: (f32, f32), tri: [[f32; 2]; 3], brush: PixelBrush },
    // A `DrawImage(Flood)`: the infinite premultiplied colour, composited over the whole target.
    // `crop`: the DrawImage's offset and image rectangle, when one bounds the flood (replayed cropped, as drawn live).
    Flood { state: DrawState, dpi: (f32, f32), src: [i32; 4], interp: u32, composite: u32, crop: Option<([f32; 2], [f32; 4])> },
    // A `DrawImage(effect)`: the effect itself, as Direct2D's command list keeps it. It is evaluated where the list is
    // drawn, against that target's clip and precision and with the effect's state then (a property set after the draw,
    // even after Close, shows).
    Effect { state: DrawState, dpi: (f32, f32), effect: crate::effects::Effect, offset: [f32; 2], interpolation: u32, composite: u32 },
    // A layer: its DIP inputs (bounds, optional mask geometry + its AA/transform, opacity and
    // optional opacity brush, options). Replayed by re-running push_layer under the composed
    // transform, so the mask/bounds re-resolve at the replay transform.
    PushLayer {
        state: DrawState,
        dpi: (f32, f32),
        bounds: [f32; 4],
        mask: Option<Geometry>,
        mask_aa: u32,
        mask_transform: Option<[f32; 6]>,
        opacity: f32,
        opacity_brush: Option<PixelBrush>,
        layer_options: u32,
    },
    PopLayer,
    // An axis-aligned clip, its DIP rect and antialias mode: replayed by re-running the push under the
    // playback transform, so it clips the list's content where it is drawn.
    PushClip { state: DrawState, dpi: (f32, f32), rect: [f32; 4], antialias: u32 },
    PopClip,
}

/// Whether a SOURCE_OVER DrawImage may play these commands inline into the target — d2d1's
/// `SafeForDrawImageFastPathChecker`: a non-SOURCE_OVER primitive blend (set inside the recording or
/// before it), a Clear that is not opaque, a nested non-SOURCE_OVER DrawImage or an
/// INITIALIZE_FROM_BACKGROUND layer makes the list unsafe, and it is realized as an image and
/// composited instead (clsafe-* probes).
pub(crate) fn inline_safe(cmds: &[RecCmd]) -> bool {
    cmds.iter().all(|c| match c {
        RecCmd::Clear { state, color, .. } => state.blend == 0 && color[3] == 1.0,
        RecCmd::PushLayer { state, layer_options, .. } => state.blend == 0 && layer_options & 1 == 0,
        RecCmd::PushClip { .. } | RecCmd::PopClip => true,
        RecCmd::DrawImage { state, composite, .. } | RecCmd::DrawImageList { state, composite, .. } | RecCmd::DrawEffect { state, composite, .. }
        | RecCmd::Flood { state, composite, .. } | RecCmd::Effect { state, composite, .. } => {
            state.blend == 0 && *composite == 0
        }
        c => c.state().is_none_or(|s| s.blend == 0),
    })
}

impl RecCmd {
    /// The recorded draw state.
    pub(crate) fn state(&self) -> Option<&DrawState> {
        match self {
            RecCmd::Clear { state, .. }
            | RecCmd::FillRect { state, .. }
            | RecCmd::FillPolys { state, .. }
            | RecCmd::FillGeom { state, .. }
            | RecCmd::StrokeGeom { state, .. }
            | RecCmd::FillRectBrush { state, .. }
            | RecCmd::FillGeomBrush { state, .. }
            | RecCmd::StrokeGeomBrush { state, .. }
            | RecCmd::Line { state, .. }
            | RecCmd::LineBrush { state, .. }
            | RecCmd::GlyphRun { state, .. }
            | RecCmd::DrawImage { state, .. }
            | RecCmd::DrawImageList { state, .. }
            | RecCmd::DrawEffect { state, .. }
            | RecCmd::BlitFloat { state, .. }
            | RecCmd::SpriteBatch { state, .. }
            | RecCmd::Mesh { state, .. }
            | RecCmd::MeshBrush { state, .. }
            | RecCmd::DrawBitmap { state, .. }
            | RecCmd::Ink { state, .. }
            | RecCmd::InkBrush { state, .. }
            | RecCmd::GradientMesh { state, .. }
            | RecCmd::OpacityMask { state, .. }
            | RecCmd::OpacityMaskBrush { state, .. }
            | RecCmd::Flood { state, .. }
            | RecCmd::Effect { state, .. }
            | RecCmd::PushLayer { state, .. }
            | RecCmd::PushClip { state, .. } => Some(state),
            RecCmd::PopLayer | RecCmd::PopClip => None,
        }
    }

    /// The recorded draw state, for composing an outer transform into a nested inlined list.
    pub(crate) fn state_mut(&mut self) -> Option<&mut DrawState> {
        match self {
            RecCmd::Clear { state, .. }
            | RecCmd::FillRect { state, .. }
            | RecCmd::FillPolys { state, .. }
            | RecCmd::FillGeom { state, .. }
            | RecCmd::StrokeGeom { state, .. }
            | RecCmd::FillRectBrush { state, .. }
            | RecCmd::FillGeomBrush { state, .. }
            | RecCmd::StrokeGeomBrush { state, .. }
            | RecCmd::Line { state, .. }
            | RecCmd::LineBrush { state, .. }
            | RecCmd::GlyphRun { state, .. }
            | RecCmd::DrawImage { state, .. }
            | RecCmd::DrawImageList { state, .. }
            | RecCmd::DrawEffect { state, .. }
            | RecCmd::BlitFloat { state, .. }
            | RecCmd::SpriteBatch { state, .. }
            | RecCmd::Mesh { state, .. }
            | RecCmd::MeshBrush { state, .. }
            | RecCmd::DrawBitmap { state, .. }
            | RecCmd::Ink { state, .. }
            | RecCmd::InkBrush { state, .. }
            | RecCmd::GradientMesh { state, .. }
            | RecCmd::OpacityMask { state, .. }
            | RecCmd::OpacityMaskBrush { state, .. }
            | RecCmd::Flood { state, .. }
            | RecCmd::Effect { state, .. }
            | RecCmd::PushLayer { state, .. }
            | RecCmd::PushClip { state, .. } => Some(state),
            RecCmd::PopLayer | RecCmd::PopClip => None,
        }
    }
}

pub(crate) struct ListInner {
    pub(crate) domain: Domain,
    pub(crate) closed: bool,
    pub(crate) cmds: Vec<RecCmd>,
    /// The first draw error latched while this list was the recording target. Direct2D surfaces it
    /// from `ID2D1CommandList::Close` (the context also reports it at `EndDraw`), so a drop-in must
    /// return it from Close to match (fuzz probe: a nested `DrawImage` with an out-of-range composite
    /// mode latches `E_INVALIDARG`, and native's Close returns it). Stamped by `process_new_error`.
    pub(crate) rec_error: Option<HRESULT>,
    /// The clips (`true`) and layers (`false`) pushed while recording this list and not yet popped: a pop
    /// checks against them, and a list closed with any left open fails its Close.
    pub(crate) frames: Vec<bool>,
}

/// `ID2D1CommandList`. Usable as a `DrawImage` image only after [`close`](Self::close).
#[derive(Clone)]
pub struct CommandList(pub(crate) Rc<RefCell<ListInner>>);

impl CommandList {
    pub(crate) fn new(domain: Domain) -> Self {
        CommandList(Rc::new(RefCell::new(ListInner { domain, closed: false, cmds: Vec::new(), rec_error: None, frames: Vec::new() })))
    }

    /// The first draw error latched while this list was the recording target, which
    /// `ID2D1CommandList::Close` returns (see [`ListInner::rec_error`]).
    pub fn rec_error(&self) -> Option<HRESULT> {
        self.0.borrow().rec_error
    }

    /// `ID2D1CommandList::Close`: seal the list. Idempotent here (re-close is caller error in native
    /// but has no observable pixels; the state machine's WRONG_STATE lives in the context).
    pub fn close(&self) {
        self.0.borrow_mut().closed = true;
    }

    pub fn is_closed(&self) -> bool {
        self.0.borrow().closed
    }

    /// Whether a clip or layer pushed while recording was left open: Direct2D's Close then fails with
    /// D2DERR_PUSH_POP_UNBALANCED.
    pub fn unbalanced(&self) -> bool {
        !self.0.borrow().frames.is_empty()
    }

    pub(crate) fn domain(&self) -> Domain {
        self.0.borrow().domain
    }
}
