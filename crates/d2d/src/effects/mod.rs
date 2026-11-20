//! Phase 2B effect graph: `ID2D1Effect` creation/registration, the property system, the transform
//! graph and its rectangle mapping, and the few built-ins the `custom-fx-01..06` scenes exercise
//! (Crop, Border, Tile, Flood, Composite, 2D-affine, ColorMatrix).
//!
//! The graph is evaluated as a **pull**: each node is a function `eval(x,y) → premultiplied RGBA f32`
//! over its own pixel grid (DIPs = pixels at 96 DPI). WARP links the per-node library functions into
//! one pixel shader (R11), so a chain of simple-sampling nodes runs in one f32 pass with **no**
//! intermediate texture — the forced `8UNORM` precision in `custom-fx-05` never materialises, and the
//! chain is bit-identical to an unquantised f32 evaluation. We therefore evaluate the whole graph in
//! f32 and quantise once, when the output node is realised for `DrawImage`. (Buffer precision and
//! tiling are thus perf/quantisation concerns with no pixel effect for these point-sampled graphs;
//! `realize` is inherently untiled, so tiled≡untiled holds trivially.)
//!
//! Adding a built-in (for 2C): map its CLSID in [`kind_of`], give it a [`Kind`] tag, and handle that
//! tag in [`Effect::bounds`] and [`Effect::eval`] (plus input realisation if it resamples).

/// The 2C built-in families' pure per-pixel maths (colour/blend/filters/transform), which the engine
/// dispatches to as effects are registered.
pub mod builtin;

/// Custom effects: `RegisterEffectFromString`/`Stream`, `ID2D1EffectImpl`, the transform graph and
/// its nodes, `ID2D1EffectContext`, resource textures, the 3D LUT, and shader loads.
pub mod custom;

/// Shader-backed draw transform: runs a custom effect's pixel shader through `d2d-shader` (R11).
pub mod shader;

/// WARP's HLSL pixel-shader linear `sample` intrinsic (float texcoord → f32 bilinear), shared by the
/// blur/shadow/displacement filters and available to the lighting kernels and `d2d-shader`.
pub mod sample;

/// Effect-graph rectangle mapping (ffirects, this agent): `GetEffectRequiredInputRectangles`,
/// `GetEffectInvalidRectangles`, `GetImage{Local,World}Bounds`.
pub mod rects;
pub(crate) mod arena;
pub(crate) mod plan;

pub use custom::{EffectContext, EffectImpl, Lut3d, NodeId, PixelShaderEffect, PropKind, Props, ResourceTexture, Transform, TransformGraph};

use crate::bitmap::Bitmap;
use crate::command::{CommandList, RecCmd};
use crate::context::DeviceContext;
use d2d_types::abi::*;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use windows_core::{GUID, HRESULT};

const E_INVALIDARG: HRESULT = HRESULT(0x8007_0057u32 as i32);
/// `CreateEffect` on an unregistered CLSID returns raw `HRESULT_FROM_WIN32(ERROR_NOT_FOUND)`, not the
/// documented `D2DERR_EFFECT_IS_NOT_REGISTERED` (traced on native).
pub const NOTFOUND: HRESULT = HRESULT(0x8007_0490u32 as i32);

// --- device-pixel rectangles (R1) -------------------------------------------------------------

/// A device-pixel rect, right/bottom exclusive. Infinite = the native `{INT_MIN..INT_MAX}` sentinel.
#[derive(Clone, Copy)]
struct Rect {
    l: i32,
    t: i32,
    r: i32,
    b: i32,
}
const INF: Rect = Rect { l: i32::MIN, t: i32::MIN, r: i32::MAX, b: i32::MAX };

impl Rect {
    fn intersect(self, o: Rect) -> Rect {
        Rect { l: self.l.max(o.l), t: self.t.max(o.t), r: self.r.min(o.r), b: self.b.min(o.b) }
    }
    fn union(self, o: Rect) -> Rect {
        Rect { l: self.l.min(o.l), t: self.t.min(o.t), r: self.r.max(o.r), b: self.b.max(o.b) }
    }
    fn contains(self, x: i32, y: i32) -> bool {
        x >= self.l && x < self.r && y >= self.t && y < self.b
    }
    // i64 so the infinite rect (`INT_MAX − INT_MIN`) does not overflow before it is clipped.
    fn width(self) -> u32 {
        (self.r as i64 - self.l as i64).clamp(0, u32::MAX as i64) as u32
    }
    fn height(self) -> u32 {
        (self.b as i64 - self.t as i64).clamp(0, u32::MAX as i64) as u32
    }
}

// --- properties --------------------------------------------------------------------------------

/// A typed effect-property value. Variants mirror the harness `Value`/`D2D1_PROPERTY_TYPE` set so the
/// test driver converts 1:1. `Floats` (a blob, e.g. the ConvolveMatrix kernel) makes this non-`Copy`.
#[derive(Clone)]
pub enum Prop {
    F32(f32),
    U32(u32),
    I32(i32),
    Bool(bool),
    V2([f32; 2]),
    V3([f32; 3]),
    V4([f32; 4]),
    M32([f32; 6]),
    M44([f32; 16]),
    M54([f32; 20]),
    Floats(Rc<[f32]>),
}

/// The built-in effects the engine dispatches. 2B's own (Crop…ColorMatrix), the blend family and the
/// transform/generator/lighting family (`builtin::{blend,transform}`), plus the 3D LUT. Colour and
/// filter families register their own arms (owned by other agents) the same way; custom-registered
/// effects use [`Kind::Custom`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Crop,
    Border,
    Tile,
    Flood,
    Composite,
    Affine2D,
    /// Straighten: a rotate-about-centre (+ optional cover-scale) built into an affine transform,
    /// so it realises through the same affine resampler as [`Kind::Affine2D`] — only the matrix is
    /// computed (from Angle/MaintainSize/input bounds) and the interpolation is its ScaleMode.
    Straighten,
    ColorMatrix,
    // 2C filter family (neighbourhood/resampling — realised, not point-evaluated).
    GaussianBlur,
    DirectionalBlur,
    Shadow,
    ConvolveMatrix,
    Morphology,
    Sharpen,
    EdgeDetection,
    Emboss,
    DisplacementMap,
    Posterize,
    /// 2C colour & transfer family (per-pixel maths in [`builtin::color`]); the op selects the kernel.
    Color(ColorOp),
    /// ColorManagement (R26): a 1:1 colour-space transform ([`crate::color_context::ColorTransform`]
    /// in `Data::cms`). Non-linkable in native (binds a LUT), so its output is an intermediate at the
    /// context buffer precision — a FLOAT context (scRGB) keeps f32, which the pull graph does naturally
    /// (one output quantise), so a cms feeding another effect (asifcolorspace) stays f32 as native does.
    ColorManagement,
    /// OpacityMetadata: a metadata-only effect — it annotates its input's opaque region (an
    /// optimisation hint) and leaves the pixels untouched, so its output is input 0 verbatim.
    OpacityMetadata,
    /// YCbCr: Y'CbCr → RGB (BT.601). Input 0 is the Y plane, input 1 the CbCr plane; a 1:1
    /// (4:4:4) point conversion. Chroma subsampling (the upsampler transform) is a follow-up.
    YCbCr,
    /// HighlightsAndShadows: at default (no adjustment) native builds no shader graph and passes the
    /// input through. Its adjusted path (clarity edge-detect → Gaussian blur → gamma/YIQ tone curve)
    /// is a follow-up; until then this is the passthrough native produces at defaults.
    HighlightsShadows,
    /// BitmapSource: wraps an `IWICBitmapSource` as an image (the source is a property, not an
    /// input; the conformance runner bridges it as input 0). At default (Scale 1:1, no orientation)
    /// native produces no shader and the output is the source verbatim. Scale/orientation are a
    /// follow-up.
    BitmapSource,
    /// Histogram: a measurement sink. Its image output is empty (native's output rect is
    /// `[0,0,0,0]`); the result is the normalised per-bin counts read back through the
    /// `HistogramOutput` property ([`Effect::histogram_output`]).
    Histogram,
    // blend family (point-evaluated, `builtin::blend`)
    Blend,
    Arithmetic,
    CrossFade,
    Opacity,
    // transform / generator / lighting family (realized via `builtin::transform`)
    Atlas,
    Scale,
    Dpi,
    Transform3D,
    Perspective3D,
    Turbulence,
    PointDiffuse,
    DistantDiffuse,
    SpotDiffuse,
    PointSpecular,
    DistantSpecular,
    SpotSpecular,
    LookupTable3D,
    /// A custom effect registered from XML; its transform graph (built in `Initialize`) is the output.
    Custom,
}

impl Kind {
    /// The 2C filter family: realised (neighbourhood/resampling), not point-evaluated.
    fn is_filter(self) -> bool {
        use Kind::*;
        matches!(
            self,
            GaussianBlur
                | DirectionalBlur
                | Shadow
                | ConvolveMatrix
                | Morphology
                | Sharpen
                | EdgeDetection
                | Emboss
                | DisplacementMap
                | Posterize
        )
    }

    // Filters WARP links into a downstream point-wise consumer as a float
    // in-register intermediate (no 8-bit store between passes). Only the
    // single-pass pixel-shader filters link; the blur-based family
    // (GaussianBlur/DirectionalBlur/Shadow/Sharpen=unsharp) is multi-pass and
    // always materializes its own texture, so it does NOT link.
    fn is_linkable_filter(self) -> bool {
        use Kind::*;
        matches!(
            self,
            ConvolveMatrix | Morphology | EdgeDetection | Emboss | DisplacementMap | Posterize
        )
    }
}

/// The 2C colour/transfer effects dispatched through [`Effect::eval_color`]. Each is a 1:1 pixel map
/// (`AlphaMask` reads a second input) whose output bounds equal input 0's.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ColorOp {
    Invert,
    Grayscale,
    LuminanceToAlpha,
    Exposure,
    Brightness,
    Contrast,
    HueRotation,
    Saturation,
    Sepia,
    Tint,
    TemperatureTint,
    GammaTransfer,
    LinearTransfer,
    DiscreteTransfer,
    TableTransfer,
    Premultiply,
    UnPremultiply,
    AlphaMask,
    ChromaKey,
    WhiteLevel,
    Vignette,
    HdrToneMap,
    RgbToHue,
    HueToRgb,
}

fn kind_of(clsid: &GUID) -> Option<Kind> {
    let c = *clsid;
    // 2B built-ins
    if c == CLSID_D2D1Crop {
        Some(Kind::Crop)
    } else if c == CLSID_D2D1Border {
        Some(Kind::Border)
    } else if c == CLSID_D2D1Tile {
        Some(Kind::Tile)
    } else if c == CLSID_D2D1Flood {
        Some(Kind::Flood)
    } else if c == CLSID_D2D1Composite {
        Some(Kind::Composite)
    } else if c == CLSID_D2D12DAffineTransform {
        Some(Kind::Affine2D)
    } else if c == CLSID_D2D1ColorMatrix {
        Some(Kind::ColorMatrix)
    } else if c == CLSID_D2D1GaussianBlur {
        Some(Kind::GaussianBlur)
    } else if c == CLSID_D2D1DirectionalBlur {
        Some(Kind::DirectionalBlur)
    } else if c == CLSID_D2D1Shadow {
        Some(Kind::Shadow)
    } else if c == CLSID_D2D1ConvolveMatrix {
        Some(Kind::ConvolveMatrix)
    } else if c == CLSID_D2D1Morphology {
        Some(Kind::Morphology)
    } else if c == CLSID_D2D1Sharpen {
        Some(Kind::Sharpen)
    } else if c == CLSID_D2D1EdgeDetection {
        Some(Kind::EdgeDetection)
    } else if c == CLSID_D2D1Emboss {
        Some(Kind::Emboss)
    } else if c == CLSID_D2D1DisplacementMap {
        Some(Kind::DisplacementMap)
    } else if c == CLSID_D2D1Posterize {
        Some(Kind::Posterize)
    // blend family
    } else if c == CLSID_D2D1Blend {
        Some(Kind::Blend)
    } else if c == CLSID_D2D1ArithmeticComposite {
        Some(Kind::Arithmetic)
    } else if c == CLSID_D2D1CrossFade {
        Some(Kind::CrossFade)
    } else if c == CLSID_D2D1Opacity {
        Some(Kind::Opacity)
    // transform / generator / lighting family
    } else if c == CLSID_D2D1Atlas {
        Some(Kind::Atlas)
    } else if c == CLSID_D2D1Scale {
        Some(Kind::Scale)
    } else if c == CLSID_D2D1DpiCompensation {
        Some(Kind::Dpi)
    } else if c == CLSID_D2D13DTransform {
        Some(Kind::Transform3D)
    } else if c == CLSID_D2D13DPerspectiveTransform {
        Some(Kind::Perspective3D)
    } else if c == CLSID_D2D1Turbulence {
        Some(Kind::Turbulence)
    } else if c == CLSID_D2D1PointDiffuse {
        Some(Kind::PointDiffuse)
    } else if c == CLSID_D2D1DistantDiffuse {
        Some(Kind::DistantDiffuse)
    } else if c == CLSID_D2D1SpotDiffuse {
        Some(Kind::SpotDiffuse)
    } else if c == CLSID_D2D1PointSpecular {
        Some(Kind::PointSpecular)
    } else if c == CLSID_D2D1DistantSpecular {
        Some(Kind::DistantSpecular)
    } else if c == CLSID_D2D1SpotSpecular {
        Some(Kind::SpotSpecular)
    } else if c == CLSID_D2D1LookupTable3D {
        Some(Kind::LookupTable3D)
    } else if c == CLSID_D2D1ColorManagement {
        Some(Kind::ColorManagement)
    } else if c == CLSID_D2D1OpacityMetadata {
        Some(Kind::OpacityMetadata)
    } else if c == CLSID_D2D1Straighten {
        Some(Kind::Straighten)
    } else if c == CLSID_D2D1YCbCr {
        Some(Kind::YCbCr)
    } else if c == CLSID_D2D1HighlightsShadows {
        Some(Kind::HighlightsShadows)
    } else if c == CLSID_D2D1BitmapSource {
        Some(Kind::BitmapSource)
    } else if c == CLSID_D2D1Histogram {
        Some(Kind::Histogram)
    } else {
        color_op(&c).map(Kind::Color)
    }
}

/// Every built-in effect CLSID the engine models — the system effects `GetRegisteredEffects` reports.
/// In the d2d-types abi declaration order (alphabetical). ponytail: this is the modelled set; the
/// exact native registry *iteration order* is not reproduced (a C-only COM enumeration unreachable
/// from a browser — not a fingerprint surface), only the contract and the effect set.
pub const BUILTIN_EFFECT_CLSIDS: &[GUID] = &[
    CLSID_D2D12DAffineTransform, CLSID_D2D13DPerspectiveTransform, CLSID_D2D13DTransform, CLSID_D2D1AlphaMask,
    CLSID_D2D1ArithmeticComposite, CLSID_D2D1Atlas, CLSID_D2D1BitmapSource, CLSID_D2D1Blend, CLSID_D2D1Border,
    CLSID_D2D1Brightness, CLSID_D2D1ChromaKey, CLSID_D2D1ColorManagement, CLSID_D2D1ColorMatrix, CLSID_D2D1Composite,
    CLSID_D2D1Contrast, CLSID_D2D1ConvolveMatrix, CLSID_D2D1Crop, CLSID_D2D1CrossFade, CLSID_D2D1DirectionalBlur,
    CLSID_D2D1DiscreteTransfer, CLSID_D2D1DisplacementMap, CLSID_D2D1DistantDiffuse, CLSID_D2D1DistantSpecular,
    CLSID_D2D1DpiCompensation, CLSID_D2D1EdgeDetection, CLSID_D2D1Emboss, CLSID_D2D1Exposure, CLSID_D2D1Flood,
    CLSID_D2D1GammaTransfer, CLSID_D2D1GaussianBlur, CLSID_D2D1Grayscale, CLSID_D2D1HdrToneMap,
    CLSID_D2D1HighlightsShadows, CLSID_D2D1Histogram, CLSID_D2D1HueRotation, CLSID_D2D1HueToRgb, CLSID_D2D1Invert,
    CLSID_D2D1LinearTransfer, CLSID_D2D1LookupTable3D, CLSID_D2D1LuminanceToAlpha, CLSID_D2D1Morphology,
    CLSID_D2D1Opacity, CLSID_D2D1OpacityMetadata, CLSID_D2D1PointDiffuse, CLSID_D2D1PointSpecular, CLSID_D2D1Posterize,
    CLSID_D2D1Premultiply, CLSID_D2D1RgbToHue, CLSID_D2D1Saturation, CLSID_D2D1Scale, CLSID_D2D1Sepia,
    CLSID_D2D1Shadow, CLSID_D2D1Sharpen, CLSID_D2D1SpotDiffuse, CLSID_D2D1SpotSpecular, CLSID_D2D1Straighten,
    CLSID_D2D1TableTransfer, CLSID_D2D1TemperatureTint, CLSID_D2D1Tile, CLSID_D2D1Tint, CLSID_D2D1Turbulence,
    CLSID_D2D1UnPremultiply, CLSID_D2D1Vignette, CLSID_D2D1WhiteLevelAdjustment, CLSID_D2D1YCbCr,
];

/// 2C colour-family CLSID → [`ColorOp`] (Invert/Grayscale/LuminanceToAlpha route through the native
/// ColorMatrix shader but have their own CLSIDs).
fn color_op(c: &GUID) -> Option<ColorOp> {
    use ColorOp::*;
    Some(match *c {
        x if x == CLSID_D2D1Invert => Invert,
        x if x == CLSID_D2D1Grayscale => Grayscale,
        x if x == CLSID_D2D1LuminanceToAlpha => LuminanceToAlpha,
        x if x == CLSID_D2D1Exposure => Exposure,
        x if x == CLSID_D2D1Brightness => Brightness,
        x if x == CLSID_D2D1Contrast => Contrast,
        x if x == CLSID_D2D1HueRotation => HueRotation,
        x if x == CLSID_D2D1Saturation => Saturation,
        x if x == CLSID_D2D1Sepia => Sepia,
        x if x == CLSID_D2D1Tint => Tint,
        x if x == CLSID_D2D1TemperatureTint => TemperatureTint,
        x if x == CLSID_D2D1GammaTransfer => GammaTransfer,
        x if x == CLSID_D2D1LinearTransfer => LinearTransfer,
        x if x == CLSID_D2D1DiscreteTransfer => DiscreteTransfer,
        x if x == CLSID_D2D1TableTransfer => TableTransfer,
        x if x == CLSID_D2D1Premultiply => Premultiply,
        x if x == CLSID_D2D1UnPremultiply => UnPremultiply,
        x if x == CLSID_D2D1AlphaMask => AlphaMask,
        x if x == CLSID_D2D1ChromaKey => ChromaKey,
        x if x == CLSID_D2D1WhiteLevelAdjustment => WhiteLevel,
        x if x == CLSID_D2D1Vignette => Vignette,
        x if x == CLSID_D2D1HdrToneMap => HdrToneMap,
        x if x == CLSID_D2D1RgbToHue => RgbToHue,
        x if x == CLSID_D2D1HueToRgb => HueToRgb,
        _ => return None,
    })
}

/// True for the realized (float-scene-position) transform-family kinds: the resamplers, generators
/// and lighting effects baked through `builtin::transform`, as opposed to the point-evaluated kinds.
fn is_realized_transform(kind: Kind) -> bool {
    use Kind::*;
    matches!(
        kind,
        Atlas | Scale | Dpi | Transform3D | Perspective3D | Turbulence | PointDiffuse | DistantDiffuse | SpotDiffuse | PointSpecular | DistantSpecular | SpotSpecular
    )
}

/// The valid values of each built-in enum property; Direct2D's SetValue rejects any other with E_INVALIDARG,
/// at the top value + 1 and beyond (VM, every property here at its top value, + 1 and 100). The AlphaMode slots
/// take PREMULTIPLIED (1) or STRAIGHT (2) only. Straighten's ScaleMode has no HIGH_QUALITY_CUBIC.
fn enum_range(kind: Kind, index: u32) -> Option<(u32, u32)> {
    use Kind::*;
    Some(match (kind, index) {
        (Affine2D, 0) | (Scale, 2) | (Dpi, 0) | (ConvolveMatrix, 1) | (YCbCr, 2) => (0, 5),
        (PointDiffuse, 5) | (DistantDiffuse, 6) | (SpotDiffuse, 8) | (PointSpecular, 6) | (DistantSpecular, 7) | (SpotSpecular, 9) => (0, 5),
        (Transform3D, 0) | (Perspective3D, 0) | (Straighten, 2) | (YCbCr, 0) => (0, 4),
        (DisplacementMap, 1 | 2) | (Histogram, 1) | (ColorManagement, 1 | 3) => (0, 3),
        (Border, 0 | 1) | (GaussianBlur, 1) | (DirectionalBlur, 2) | (Shadow, 2) | (ColorManagement, 5) => (0, 2),
        (Affine2D, 1) | (Transform3D, 1) | (Perspective3D, 1) | (Scale, 3) | (Dpi, 1) | (Crop, 1) | (ConvolveMatrix, 9)
        | (GaussianBlur, 2) | (DirectionalBlur, 3) | (EdgeDetection, 2) | (Morphology, 0) | (Turbulence, 5)
        | (HighlightsShadows, 3) | (Color(ColorOp::HdrToneMap), 2) | (Color(ColorOp::RgbToHue), 0) | (Color(ColorOp::HueToRgb), 0) => (0, 1),
        (ColorMatrix, 1) | (EdgeDetection, 4) | (Color(ColorOp::Sepia), 1) | (LookupTable3D, 1) | (ColorManagement, 4) => (1, 2),
        (Composite, 0) => (0, 12),
        (Blend, 0) => (0, 25),
        _ => return None,
    })
}

/// Whether `r` is wider or taller than a realize can hold (4096 px a side): infinite, or near it.
fn beyond_realize_cap(r: Rect) -> bool {
    r.r as i64 - r.l as i64 > 1 << 12 || r.b as i64 - r.t as i64 > 1 << 12
}

/// The six lighting kinds, which light at one kernel unit per texel (`Effect::lighting_stage`).
fn is_lighting(kind: Kind) -> bool {
    use Kind::*;
    matches!(kind, PointDiffuse | DistantDiffuse | SpotDiffuse | PointSpecular | DistantSpecular | SpotSpecular)
}

// --- the effect object -------------------------------------------------------------------------

/// How many of the bitmap's tiles a Border drawn over `clip` spans: its width over the bitmap's on a wrapping or
/// mirroring axis (at least one), times the same down (d2d1's border cells past 25).
fn border_tiles(clip: Rect, bm: &Bitmap, modes: [u32; 2]) -> u64 {
    let n = |span: i64, dim: u32, mode: u32| if mode == 0 || dim == 0 { 1 } else { (span.max(0) as u64 / dim as u64).max(1) };
    n(clip.r as i64 - clip.l as i64, bm.width, modes[0]) * n(clip.b as i64 - clip.t as i64, bm.height, modes[1])
}

/// A bound input: a realized bitmap leaf, another effect's output, or a command list.
#[derive(Clone)]
pub enum Input {
    Bitmap(Rc<Bitmap>),
    Effect(Effect),
    /// A command list. Direct2D treats it as an image with no edge: each draw renders it over the rect that
    /// draw needs ([`DeviceContext::bake_lists`]), and the effect reads that rendering as a bitmap at its origin.
    List(Rc<ListInput>),
}

/// A command-list input and its rendering for the current draw (a bitmap and its origin in effect pixels).
pub struct ListInput {
    list: CommandList,
    baked: RefCell<Rc<(Bitmap, i32, i32)>>,
}

impl Input {
    pub fn list(list: CommandList) -> Self {
        Input::List(Rc::new(ListInput { list, baked: RefCell::new(Rc::new((float_bitmap(0, 0, std::iter::empty()), 0, 0))) }))
    }
}

impl ListInput {
    fn baked(&self) -> Rc<(Bitmap, i32, i32)> {
        self.baked.borrow().clone()
    }
}

struct Data {
    kind: Kind,
    props: BTreeMap<u32, Prop>,
    inputs: Vec<Input>,
    /// Context DPI captured at `CreateEffect` (DPI compensation reads it; `ID2D1EffectContext::GetDpi`).
    dpi: (f32, f32),
    /// 3D LUT resource bound to a `LookupTable3D` effect (`ID2D1LookupTable3D`).
    lut: Option<Rc<Lut3d>>,
    /// Custom-effect state (property schema + the transform graph built in `Initialize`); `Some`
    /// only when `kind == Custom`. See [`custom`].
    custom: Option<custom::CustomState>,
    /// Lazily-realized output bitmap (+ device origin), memoized so a point-eval consumer that reads a
    /// non-point-evaluable input (a blur/filter/realized-transform) bakes it once, not per pixel.
    /// Cleared on any property/input change.
    realized: Option<Rc<(Bitmap, i32, i32)>>,
    /// Lazily-realized FLOAT output of a pixel-shader FILTER read as a linked intermediate (R11 effect-
    /// shader linking): a filter feeding a point-wise consumer keeps the shader's f32 precision in-
    /// register (no 8-bit store), so its eval-intermediate is realized at R32 float even at an 8bpc
    /// buffer precision. Separate memo from `realized` (the 8bpc output-node store). Cleared with it.
    realized_f: Option<Rc<(Bitmap, i32, i32)>>,
    /// Memoized `bounds()` tagged with the [`BOUNDS_EPOCH`] it was computed at: the filter arm realizes
    /// the whole output to measure it, and the planner calls `bounds()` on every node every draw, so
    /// without this each measure re-bakes the chain. Any `set_input`/`set_value` on any effect bumps the
    /// epoch, so a stale entry (on this effect or anything upstream of it) is never returned. [perf]
    bounds_cache: std::cell::Cell<Option<(u64, Rect)>>,
    /// `D2D1_PROPERTY_PRECISION` (`D2D1_BUFFER_PRECISION`) set on this effect. `None` ⇒ the context's (or
    /// inherited) precision; otherwise this effect's output precision outright, above or below the context.
    precision: Option<u32>,
    /// The governing context bufferPrecision for the current render, pushed in by the drawing
    /// DeviceContext (`set_render_precision`) before each realize — per device context, not a
    /// thread-local, so interleaved contexts on one thread don't leak precision into each other.
    render_precision: u32,
    /// The precision an effect with no inputs takes under an UNKNOWN context (`render_precision` 0): the
    /// drawing target's (a Flood → Crop → Blur on a Float32 target with no bufferPrecision renders in float
    /// natively; 8bpc on a BGRA8 target).
    render_fallback: u32,
    /// `ColorManagement`'s colour transform (`kind == ColorManagement`); `None` = identity passthrough
    /// (unset contexts, or a SIMPLE-profile pair native fails to realise).
    cms: Option<crate::color_context::ColorTransform>,
    /// `Vignette`'s mesh for the last input rect + properties (key: rect, colour, transition, strength bits).
    vignette: RefCell<Option<([i32; 10], Rc<builtin::color::Vignette>)>>,
    /// `HdrToneMap`'s per-instance tone-map (matrices + anchors) for the last property set (key: the
    /// three property bits). Built once, not rebuilt per pixel.
    hdr: RefCell<Option<([i32; 3], Rc<builtin::color::HdrToneMap>)>>,
    /// Histogram's `HistogramOutput`, cached at the last drawn+flushed render (`None` until the
    /// effect is first drawn). `GetValue` returns `E_INVALIDARG` while `None`, else this value — and
    /// it holds across property/input changes (native returns the stale last-rendered histogram until
    /// the effect is drawn again, not `E_INVALIDARG`).
    histogram: Option<Vec<f32>>,
    /// Where Direct2D's intermediate allocator puts this blur's intermediates for the current draw
    /// ([`plan::apply`]); the origin when its graph has no plan.
    place: builtin::filters_hp::BlurPlace,
    /// The precision this effect's output is stored at when the last draw's plan gives it an intermediate of its
    /// own ([`plan::stages`]): its consumers read it from there. `None` when it is linked into its consumer.
    store: Option<u32>,
}

/// `ID2D1Effect`. The output is the effect itself (`GetOutput` returns `this` in managed), so an
/// `Effect` is used directly as a `DrawImage` image or as another effect's input. Cheap to clone.
#[derive(Clone)]
pub struct Effect(Rc<RefCell<Data>>);


/// A process-global monotonic counter bumped by every `set_input`/`set_value`; a
/// [`Data::bounds_cache`] entry is valid only while it carries the current value, so one bump
/// invalidates every effect's cached bounds at once (the cheap way to drop a downstream cache when
/// anything upstream changes). Atomic, not thread-local: device contexts can live on different threads,
/// and a mutation on any of them must be seen by a read on any other. `Relaxed` suffices — it is a bare
/// version tag, and a given effect's `Rc`-held cache is only ever read and written on one thread, so
/// that thread's bump happens-before its own read in program order. [perf]
static BOUNDS_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn bump_bounds_epoch() {
    BOUNDS_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

impl Effect {
    fn new_in(kind: Kind, dpi: (f32, f32)) -> Self {
        Effect(Rc::new(RefCell::new(Data { kind, props: BTreeMap::new(), inputs: Vec::new(), dpi, lut: None, custom: None, realized: None, realized_f: None, bounds_cache: std::cell::Cell::new(None), precision: None, render_precision: 1, render_fallback: 1, cms: None, histogram: None, vignette: RefCell::new(None), hdr: RefCell::new(None), place: Default::default(), store: None })))
    }

    /// `ColorManagement`: set its colour transform (built from the source/destination colour contexts
    /// and alpha mode). `None` = identity (a SIMPLE-profile pair native fails to realise; the caller
    /// latches that separately). The transform evaluates in f32 so the pull graph keeps the context
    /// (float, for scRGB) precision until the single output quantise.
    pub fn set_cms(&self, t: Option<crate::color_context::ColorTransform>) {
        self.0.borrow_mut().cms = t;
    }

    /// `SetInput(index, image)`: bind an input, growing the input list as needed.
    pub fn set_input(&self, index: u32, input: Input) {
        let mut d = self.0.borrow_mut();
        if index as usize >= d.inputs.len() {
            // Pad with a repeat of the first input is never needed by the scenes; grow with clones of
            // the new input so indices stay contiguous.
            d.inputs.resize(index as usize + 1, input.clone());
        }
        d.inputs[index as usize] = input;
        d.realized = None; // graph/input changed: drop the realize memo
        bump_bounds_epoch();
    }

    /// The minimum number of effect inputs this node's realize/eval will index. Generators take none;
    /// the two-input effects index input 1 (Blend/Arithmetic/CrossFade/YCbCr/DisplacementMap, and the
    /// AlphaMask colour op); every other built-in indexes input 0 (including Composite, which reads
    /// `inputs[0]` then folds the rest). A custom effect requires the inputs its transform graph
    /// actually references ([`custom::CustomState::referenced_input_count`]) — NOT the declared schema
    /// input count: a self-contained custom graph (a nested Flood generator, or a node fed by internal
    /// nodes) references none and renders with no external input, while a graph wiring an effect input
    /// needs it bound (native latches E_INVALIDARG for a referenced-but-unset input). `graph_valid`
    /// also recurses into any effect bound as an input, so a built-in missing an input is caught too.
    fn required_inputs(&self) -> usize {
        let d = self.0.borrow();
        match d.kind {
            Kind::Flood | Kind::Turbulence => 0,
            Kind::Custom => d.custom.as_ref().map_or(0, |c| c.referenced_input_count() as usize),
            Kind::Blend | Kind::Arithmetic | Kind::CrossFade | Kind::YCbCr | Kind::DisplacementMap => 2,
            Kind::Color(ColorOp::AlphaMask) => 2,
            _ => 1,
        }
    }

    /// The effect's DECLARED input count — what `ID2D1Effect::GetInputCount` reports on a freshly
    /// created effect (before any `SetInputCount`). Built-ins have a fixed count (generators 0, dual
    /// 2, else 1); a custom effect reports its registered `<Inputs>` count. Unlike [`required_inputs`]
    /// (which is the draw-time *referenced* count), this is the schema count the app queries.
    pub fn initial_input_count(&self) -> u32 {
        let d = self.0.borrow();
        match d.kind {
            Kind::Flood | Kind::Turbulence => 0,
            Kind::Custom => d.custom.as_ref().map_or(1, |c| c.input_count()),
            Kind::Blend | Kind::Arithmetic | Kind::CrossFade | Kind::YCbCr | Kind::DisplacementMap => 2,
            Kind::Color(ColorOp::AlphaMask) => 2,
            _ => 1,
        }
    }

    /// Recursively true when every effect node reachable from here carries at least the inputs its
    /// realize will index ([`Self::required_inputs`]). `DrawImage` checks this BEFORE realizing and
    /// rejects a short/missing-input graph with E_INVALIDARG, so the realize never indexes an input
    /// out of bounds. Deliberately panic-free (no `catch_unwind`): the core is also built for
    /// `wasm32-unknown-unknown`, which is `panic=abort`, so an out-of-bounds index there would kill
    /// the page and diverge from native. The effect graph is a DAG, so the recursion terminates.
    pub(crate) fn graph_valid(&self) -> bool {
        let need = self.required_inputs();
        let inputs = self.0.borrow().inputs.clone();
        if inputs.len() < need {
            return false;
        }
        inputs.iter().all(|inp| match inp {
            Input::Effect(e) => e.graph_valid(),
            Input::Bitmap(_) | Input::List(_) => true,
        })
    }

    /// Drive `PrepareForRender` once per draw on every custom effect reachable from here (native calls
    /// each `ID2D1EffectImpl::PrepareForRender` before realization). A no-op for built-ins and for a
    /// custom effect whose bridge does not override it; the COM bridge uses it to re-push a pixel-shader
    /// constant buffer set after `Initialize`. Called by `draw_image_effect` before realizing. [G2]
    /// The first failure of an author effect's `PrepareForRender` fails the draw (`Err`).
    pub(crate) fn prepare_for_render(&self) -> Result<(), HRESULT> {
        let inputs = {
            let d = self.0.borrow();
            if let Some(c) = d.custom.as_ref() {
                c.prepare_for_render()?;
            }
            d.inputs.clone()
        };
        for inp in &inputs {
            if let Input::Effect(e) = inp {
                e.prepare_for_render()?;
            }
        }
        Ok(())
    }

    // --- custom-effect author-property schema (exposed through the FFI `ID2D1Properties`) ----------
    // A registered custom effect carries its property schema; the FFI facade reads these to answer
    // GetPropertyCount/Name/Index/Type for it (a built-in uses the facade's own static table). `None`
    // on every accessor for a non-custom effect. [G2]

    /// Number of author properties, or `None` if this is not a custom effect.
    pub fn custom_property_count(&self) -> Option<u32> {
        self.0.borrow().custom.as_ref().map(|c| c.property_count())
    }
    /// The author property name at `index` (custom effects only).
    pub fn custom_property_name(&self, index: u32) -> Option<String> {
        self.0.borrow().custom.as_ref().and_then(|c| c.property_name(index).map(str::to_owned))
    }
    /// The author property [`PropKind`] at `index` (custom effects only).
    pub fn custom_property_kind(&self, index: u32) -> Option<PropKind> {
        self.0.borrow().custom.as_ref().and_then(|c| c.property_kind(index))
    }
    /// The author property index for `name` (custom effects only).
    pub fn custom_property_index(&self, name: &str) -> Option<u32> {
        self.0.borrow().custom.as_ref().and_then(|c| c.property_index(name))
    }

    /// Deliver a `SetValue` for custom author property `name` to its binding setter (the raw value
    /// bytes). No-op for a non-custom effect or an unbound property. [G2]
    pub fn set_bound_value(&self, name: &str, data: &[u8]) {
        if let Some(c) = self.0.borrow().custom.as_ref() {
            c.set_bound_value(name, data);
        }
    }

    /// The drop-in calls this when it draws an effect's output image. For Histogram it caches the
    /// `HistogramOutput` computed from the current input/properties (native fills it at draw+flush),
    /// so a later read returns that value — and keeps returning it, stale, across later property
    /// changes until the effect is drawn again. A no-op for every other effect.
    pub fn mark_drawn(&self) {
        if self.0.borrow().kind == Kind::Histogram {
            let h = self.histogram_output();
            self.0.borrow_mut().histogram = Some(h);
        }
    }

    /// The cached `HistogramOutput` (`None` until the effect is drawn); see [`Data::histogram`].
    pub fn histogram_cached(&self) -> Option<Vec<f32>> {
        self.0.borrow().histogram.clone()
    }

    /// `SetValue(index, value)`. System properties (high bit, e.g. `PRECISION`) are accepted and
    /// ignored — the graph runs in f32, so precision has no pixel effect (R10/R11). A value whose type
    /// does not match the slot is rejected `E_INVALIDARG`; out-of-range numerics are left to the
    /// effect to clamp in `eval`.
    pub fn set_value(&self, index: u32, value: Prop) -> Result<(), HRESULT> {
        if index & 0x8000_0000 != 0 {
            // D2D1_PROPERTY_PRECISION (0x80000007) feeds the precision rule; other system properties
            // (CLSID/DISPLAYNAME/…) have no pixel effect and are accepted-and-ignored.
            if index == 0x8000_0007 {
                let p = match value {
                    Prop::U32(p) => p,
                    Prop::I32(p) => p as u32,
                    _ => return Err(E_INVALIDARG),
                };
                let mut d = self.0.borrow_mut();
                d.precision = if p == 0 { None } else { Some(p) }; // 0 = UNKNOWN ⇒ no override
                d.realized = None;
            }
            return Ok(());
        }
        self.0.borrow_mut().realized = None; // property change invalidates the realize memo
        bump_bounds_epoch();
        // A custom effect validates against its XML schema, then rebuilds its transform graph.
        if self.0.borrow().kind == Kind::Custom {
            self.0.borrow().custom.as_ref().unwrap().validate(index, &value)?;
            self.0.borrow_mut().props.insert(index, value);
            return self.rebuild_custom();
        }
        if !self.type_ok(index, &value) {
            return Err(E_INVALIDARG);
        }
        // Direct2D rejects an enum property value outside its declared range with E_INVALIDARG rather than
        // clamping it. (Turbulence's Size/BaseFrequency/NumOctaves clamp, so they need no gate.)
        if let (Some((lo, hi)), Prop::U32(v)) = (enum_range(self.0.borrow().kind, index), &value) {
            if !(lo..=hi).contains(v) {
                return Err(E_INVALIDARG);
            }
        }
        // Composite's Mode takes a D2D1_COMPOSITE_MODE, SOURCE_OVER..MASK_INVERT; past it SetValue fails (r27).
        if let (Kind::Composite, 0, Prop::U32(v)) = (self.0.borrow().kind, index, &value) {
            if *v > 12 {
                return Err(E_INVALIDARG);
            }
        }
        self.0.borrow_mut().props.insert(index, value);
        Ok(())
    }

    /// True if `value`'s type matches the slot `index` for this effect's kind (the native property
    /// model rejects a type mismatch with `E_INVALIDARG`).
    fn type_ok(&self, index: u32, value: &Prop) -> bool {
        use Kind::*;
        let kind = self.0.borrow().kind;
        let expect_f32 = matches!(value, Prop::F32(_));
        let expect_v2 = matches!(value, Prop::V2(_));
        let expect_v4 = matches!(value, Prop::V4(_));
        let expect_enum = matches!(value, Prop::U32(_));
        let expect_bool = matches!(value, Prop::Bool(_));
        match (kind, index) {
            // A TableTransfer table needs two entries to interpolate between; a DiscreteTransfer one takes a single
            // entry (VM: a one-entry blue table fails SetValue with E_INVALIDARG for the first only). Either holds at most
            // 2048 (VM: 2048 accepted, 2049, 3826, 4096 and longer fail E_INVALIDARG and leave the table as it was).
            (Color(ColorOp::TableTransfer), 0 | 2 | 4 | 6) => matches!(&value, Prop::Floats(v) if (2..=2048).contains(&v.len())),
            (Color(ColorOp::DiscreteTransfer), 0 | 2 | 4 | 6) => matches!(&value, Prop::Floats(v) if (1..=2048).contains(&v.len())),
            (Crop, 0) | (Tile, 0) | (Flood, 0) => expect_v4,
            (Crop, 1) => expect_enum,
            (Border, 0) | (Border, 1) => expect_enum,
            (Composite, 0) => expect_enum,
            (Affine2D, 0) | (Affine2D, 1) => expect_enum,
            (Affine2D, 2) => matches!(value, Prop::M32(_)),
            (Affine2D, 3) => matches!(value, Prop::F32(_)),
            (ColorMatrix, 0) => matches!(value, Prop::M54(_)),
            (ColorMatrix, 1) => expect_enum,
            (ColorMatrix, 2) => expect_bool,
            // 2C filter family
            (GaussianBlur, 0) => expect_f32,
            (GaussianBlur, 1 | 2) => expect_enum,
            (DirectionalBlur, 0 | 1) => expect_f32,
            (DirectionalBlur, 2 | 3) => expect_enum,
            (Shadow, 0) => expect_f32,
            (Shadow, 1) => expect_v4,
            (Shadow, 2) => expect_enum,
            (ConvolveMatrix, 0 | 7) => expect_v2,
            (ConvolveMatrix, 1 | 2 | 3 | 9) => expect_enum,
            (ConvolveMatrix, 4) => matches!(value, Prop::Floats(_)),
            (ConvolveMatrix, 5 | 6) => expect_f32,
            (ConvolveMatrix, 8 | 10) => expect_bool,
            (Morphology, 0 | 1 | 2) => expect_enum,
            (Sharpen, 0 | 1) => expect_f32,
            (EdgeDetection, 0 | 1) => expect_f32,
            (EdgeDetection, 2 | 4) => expect_enum,
            (EdgeDetection, 3) => expect_bool,
            // Turbulence's registered property types: Offset/Size/BaseFrequency vector2, NumOctaves uint32,
            // Seed **int32** (a scene that passes it as uint32 mismatches and
            // native rejects: fuzz r15 20090/20096/…), Noise enum, Stitchable bool. Size/BaseFrequency/
            // NumOctaves clamp rather than reject, so the only type-reject is Seed.
            (Turbulence, 0 | 1 | 2) => expect_v2,
            (Turbulence, 3 | 5) => expect_enum,
            (Turbulence, 4) => matches!(value, Prop::I32(_)),
            (Turbulence, 6) => expect_bool,
            (Turbulence, _) => false,
            (Emboss, 0 | 1) => expect_f32,
            (DisplacementMap, 0) => expect_f32,
            (DisplacementMap, 1 | 2) => expect_enum,
            (Posterize, 0 | 1 | 2) => expect_enum,
            // The colour family's slots are many and all scene-set with the right type; no conformance
            // scene probes a type mismatch here, so accept (ponytail: per-slot typing if a scene needs it).
            (Color(_), _) => true,
            (GaussianBlur | DirectionalBlur | Shadow | ConvolveMatrix | Morphology | Sharpen | EdgeDetection | Emboss | DisplacementMap | Posterize, _) => false,
            (Crop | Tile | Flood | Border | Composite | Affine2D | ColorMatrix, _) => false, // unknown index for this 2B effect
            // The blend/transform/lighting families' setters store any property; their kernels read
            // the slots they need with defaults (no scene exercises a type-mismatch negative, which
            // would need the v2 `expect_hr` op). `Custom` validates against
            // its XML schema in its own setter path.
            _ => true,
        }
    }

    fn get_f32(&self, index: u32, default: f32) -> f32 {
        match self.0.borrow().props.get(&index) {
            Some(Prop::F32(v)) => *v,
            _ => default,
        }
    }

    /// A blur's standard deviation in effect pixels: the DIP property times [`DIP_SCALE`].
    fn sigma(&self) -> f32 {
        self.get_f32(0, 3.0).max(0.0) * DIP_SCALE.with(|s| s.get())
    }

    /// Whether any node of the graph samples its input's neighbourhood (a blur, convolution, …).
    /// Whether this point-wise graph leaves its bitmap as it is (an identity ColorMatrix, a white Tint, each over
    /// such a graph or the bitmap), which Direct2D then draws itself.
    /// A Border straight over a bitmap: the bitmap and the Border's edge modes.
    fn border_of_bitmap(&self) -> Option<(Rc<Bitmap>, [u32; 2])> {
        let d = self.0.borrow();
        match d.inputs.first() {
            // An alpha-only bitmap stays with the realize path (a bitmap brush rejects it). An ignored alpha reads as stored,
            // as an effect input reads it: the texture is the bitmap's own (VM: a 192-dpi IGNORE bitmap's Border drawn
            // MULTI_SAMPLE_LINEAR, axis-aligned and rotated, matches it marked PREMULTIPLIED).
            Some(Input::Bitmap(b)) if d.kind == Kind::Border && d.inputs.len() == 1 && b.format != DXGI_FORMAT_A8_UNORM => {
                let b = if b.alpha == D2D1_ALPHA_MODE_IGNORE { Rc::new(Bitmap { alpha: D2D1_ALPHA_MODE_PREMULTIPLIED, ..(**b).clone() }) } else { b.clone() };
                Some((b, [self.get_u32(0, 0), self.get_u32(1, 0)]))
            }
            // A Border over a Border's output (infinite) leaves it as it is, whatever its own modes (VM: a CLAMP×CLAMP and a
            // MIRROR×WRAP Border over a CLAMP×MIRROR Border of a bitmap draw alike, r27 32852).
            Some(Input::Effect(e)) if d.kind == Kind::Border && d.inputs.len() == 1 => e.border_of_bitmap(),
            _ => None,
        }
    }

    fn passes_through(&self) -> bool {
        let identity = self.is_identity_op();
        let d = self.0.borrow();
        identity && match &d.inputs[..] {
            [Input::Bitmap(_)] => true,
            [Input::Effect(e)] => e.passes_through(),
            _ => false,
        }
    }

    /// Whether this node leaves its input as it is, so Direct2D drops it from the graph: an identity ColorMatrix, a
    /// white Tint, an opaque Opacity, or a whole-pixel (or HARD) Crop covering its input.
    pub(super) fn is_identity_op(&self) -> bool {
        let d = self.0.borrow();
        match d.kind {
            Kind::ColorMatrix => self.color_matrix_rows() == core::array::from_fn(|i| if i < 16 && i % 5 == 0 { 1.0 } else { 0.0 }),
            Kind::Color(ColorOp::Tint) => self.get_v4(0, [1.0; 4]) == [1.0; 4],
            // Opacity 1.0 is an identity ColorMatrix (opacity scales alpha), so like the identity ColorMatrix
            // / white Tint arms above it resolves to its input bitmap and is drawn as a plain bitmap.
            Kind::Opacity => self.get_f32(0, 1.0) == 1.0,
            // A Crop draws no shader of its own when it is HARD or its rect in pixels (clamped to ±2.1e10, so an
            // infinite one too) is whole on every side; one that covers its input then leaves it as it is.
            Kind::Crop => {
                let r = self.rect_v4().map(|v| v.max(-2.1474836e10).min(2.1474836e10));
                let whole = self.get_u32(1, 0) == 1 || r.iter().all(|v| v % 1.0 == 0.0);
                let c = self.crop_rect();
                let covers = |b: Rect| c.l <= b.l && c.t <= b.t && c.r >= b.r && c.b >= b.b;
                whole && match d.inputs.first() {
                    Some(Input::Bitmap(b)) => covers(Rect { l: 0, t: 0, r: b.width as i32, b: b.height as i32 }),
                    Some(Input::Effect(e)) => covers(e.bounds()),
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// Whether a custom shader in this graph asks an infinite rect of a complex input Direct2D would have to realize
    /// unbounded: an intermediate with infinite bounds, or a Border folded over a bitmap with a WRAP or MIRROR edge.
    fn infinite_request_too_large(&self) -> bool {
        let d = self.0.borrow();
        (d.kind == Kind::Custom && d.custom.as_ref().is_some_and(|c| c.infinite_request_too_large(&d.inputs)))
            || d.inputs.iter().any(|i| matches!(i, Input::Effect(e) if e.infinite_request_too_large()))
    }

    fn has_filter(&self) -> bool {
        let d = self.0.borrow();
        d.kind.is_filter() || d.inputs.iter().any(|i| matches!(i, Input::Effect(e) if e.has_filter()))
    }

    /// Whether the graph can draw in device-pixel space: only bitmap inputs (whose pixels are the
    /// effect's, without DPI compensation) and effects with no DIP-valued property besides a blur's σ and a Scale's
    /// centre point (both scaled by `DIP_SCALE`).
    fn pixel_space_ok(&self) -> bool {
        let d = self.0.borrow();
        (matches!(d.kind, Kind::Flood | Kind::Composite | Kind::ColorMatrix | Kind::GaussianBlur | Kind::DirectionalBlur
            | Kind::Shadow | Kind::Posterize | Kind::Color(_) | Kind::OpacityMetadata | Kind::Blend | Kind::Arithmetic
            | Kind::CrossFade | Kind::Opacity | Kind::Border | Kind::Emboss | Kind::Scale | Kind::Crop | Kind::Affine2D | Kind::Straighten)
            || is_lighting(d.kind))
            && d.inputs.iter().all(|i| match i { Input::Bitmap(_) | Input::List(_) => true, Input::Effect(e) => e.pixel_space_ok() })
    }
    /// Whether the graph has a DIP-valued property that DIP_SCALE converts (a blur's σ, a Scale's centre, the lighting
    /// and Emboss DPI stage).
    fn dip_valued(&self) -> bool {
        let d = self.0.borrow();
        matches!(d.kind, Kind::GaussianBlur | Kind::DirectionalBlur | Kind::Shadow | Kind::Emboss | Kind::Scale | Kind::Affine2D) || is_lighting(d.kind)
            || d.inputs.iter().any(|i| matches!(i, Input::Effect(e) if e.dip_valued()))
    }
    /// Whether [`bounds`](Self::bounds) is Direct2D's own rect for the whole graph: Scale, DPI and the 3D
    /// transforms report an infinite stand-in.
    fn bounds_exact(&self) -> bool {
        let d = self.0.borrow();
        !matches!(d.kind, Kind::Scale | Kind::Dpi | Kind::Transform3D | Kind::Perspective3D)
            && d.inputs.iter().all(|i| match i { Input::Effect(e) => e.bounds_exact(), _ => true })
    }
    fn get_bool(&self, index: u32, default: bool) -> bool {
        match self.0.borrow().props.get(&index) {
            Some(Prop::Bool(v)) => *v,
            _ => default,
        }
    }
    fn get_v2(&self, index: u32, default: [f32; 2]) -> [f32; 2] {
        match self.0.borrow().props.get(&index) {
            Some(Prop::V2(v)) => *v,
            _ => default,
        }
    }
    fn get_v3(&self, index: u32, default: [f32; 3]) -> [f32; 3] {
        match self.0.borrow().props.get(&index) {
            Some(Prop::V3(v)) => *v,
            _ => default,
        }
    }

    fn get_u32(&self, index: u32, default: u32) -> u32 {
        match self.0.borrow().props.get(&index) {
            Some(Prop::U32(v)) => *v,
            _ => default,
        }
    }
    fn get_v4(&self, index: u32, default: [f32; 4]) -> [f32; 4] {
        match self.0.borrow().props.get(&index) {
            Some(Prop::V4(v)) => *v,
            _ => default,
        }
    }
    fn get_i32(&self, index: u32, default: i32) -> i32 {
        match self.0.borrow().props.get(&index) {
            Some(Prop::I32(v)) => *v,
            _ => default,
        }
    }
    fn get_m44(&self, index: u32, default: [f32; 16]) -> [f32; 16] {
        match self.0.borrow().props.get(&index) {
            Some(Prop::M44(v)) => *v,
            _ => default,
        }
    }
    /// A float-array property (a ConvolveMatrix kernel, a transfer table), or `default`.
    fn get_floats(&self, index: u32, default: &[f32]) -> Rc<[f32]> {
        match self.0.borrow().props.get(&index) {
            Some(Prop::Floats(v)) => v.clone(),
            _ => Rc::from(default),
        }
    }

    /// This effect's output-buffer precision (Direct2D's rule, verified on WARP): `D2D1_PROPERTY_PRECISION` when set,
    /// raising or lowering it (a 32bpc property over an 8bpc input and an 8bpc or unset context stores float;
    /// an 8bpc one under a Float32 context stores UNORM8), else the render context's bufferPrecision. SetOutputBuffer
    /// does not change it, and an input's precision does not once the context sets one; an UNKNOWN (unset)
    /// context takes the highest precision among the inputs (a float bitmap blurs in float, `precision-05`).
    pub(super) fn out_precision(&self) -> u32 {
        let d = self.0.borrow();
        let ctx = match d.render_precision {
            0 => d.inputs.iter().map(input_precision).max_by_key(|&p| prec_rank(p)).unwrap_or(d.render_fallback),
            p => p,
        };
        d.precision.unwrap_or(ctx)
    }

    /// The output precision when it is above 8bpc (16/32-bit float, 16-bit UNORM): native then keeps this
    /// effect's realized output and the filter family's intermediates at it ([`builtin::filters_hp`]).
    /// `None` at 8bpc, where every path is the UNORM8 one the 8bpc oracle pins.
    fn hp(&self) -> Option<u32> {
        let p = self.out_precision();
        (prec_rank(p) > 1).then_some(p)
    }

    /// Push the drawing device context's bufferPrecision into this effect and, recursively, every effect
    /// feeding it — so the realize quantizes each shader input at the context that is drawing it (per
    /// context, re-applied each draw). Called by `draw_image_effect` before realizing.
    pub(super) fn set_render_precision(&self, p: u32, fallback: u32) {
        {
            let mut d = self.0.borrow_mut();
            if d.render_precision != p || d.render_fallback != fallback {
                d.realized = None; // realized at another context's precision
                d.realized_f = None;
            }
            d.render_precision = p;
            d.render_fallback = fallback;
        }
        // Re-apply down the whole input graph every draw (small DAGs) so an interleaved draw by another
        // context always re-establishes its own precision, even on a shared sub-effect.
        let inputs = self.0.borrow().inputs.clone();
        for inp in &inputs {
            if let Input::Effect(e) = inp {
                e.set_render_precision(p, fallback);
            }
        }
    }

    // --- rectangle mapping (R1) ---------------------------------------------------------------

    /// This node's output rect (forward `MapInputRectsToOutputRect`). Infinite for the generators and
    /// extenders; the scenes always crop/composite them back to a finite rect before realising.
    fn bounds(&self) -> Rect {
        let epoch = BOUNDS_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if let Some((e, b)) = self.0.borrow().bounds_cache.get() {
            if e == epoch {
                return b;
            }
        }
        let b = self.bounds_uncached();
        self.0.borrow().bounds_cache.set(Some((epoch, b)));
        b
    }
    fn bounds_uncached(&self) -> Rect {
        if let Some(g) = self.lighting_stage() {
            return g.bounds();
        }
        let d = self.0.borrow();
        use Kind::*;
        match d.kind {
            Color(_) | ColorManagement | OpacityMetadata | YCbCr | HighlightsShadows => input_bounds(&d.inputs[0]), // 1:1 on input 0 (OpacityMetadata/HighlightsShadows pass through; YCbCr output = Y plane)
            BitmapSource => self.bitmapsource_bounds(),
            Histogram => Rect { l: 0, t: 0, r: 0, b: 0 }, // a measurement sink: empty image output
            Flood | Border | Tile => INF,
            // Scale/DPI/3D are logically infinite under hard/clamp sampling (and soft is clipped to
            // the target anyway); realize over the draw clip so both borders match native.
            Scale | Dpi | Transform3D | Perspective3D => INF,
            Crop => input_bounds(&d.inputs[0]).intersect(self.crop_rect()),
            // Native SOURCE_OUT keeps the last source bounds; unioning an infinite destination shifts the sampler DDA.
            Composite if self.get_u32(0, 0) == 4 => d.inputs.last().map(input_bounds).unwrap_or(INF),
            Composite | Blend | Arithmetic | CrossFade => {
                d.inputs.iter().map(input_bounds).reduce(Rect::union).unwrap_or(INF)
            }
            Opacity | ColorMatrix | LookupTable3D | PointDiffuse | DistantDiffuse | SpotDiffuse | PointSpecular | DistantSpecular | SpotSpecular => {
                input_bounds(&d.inputs[0])
            }
            Atlas => {
                let r = self.get_v4(0, [f32::MIN, f32::MIN, f32::MAX, f32::MAX]).map(|v| v.round() as i32);
                Rect { l: 0, t: 0, r: r[2] - r[0], b: r[3] - r[1] }
            }
            Turbulence => {
                let o = self.get_v2(0, [0.0, 0.0]);
                let s = self.get_v2(1, [0.0, 0.0]);
                Rect { l: o[0].round() as i32, t: o[1].round() as i32, r: (o[0] + s[0]).round() as i32, b: (o[1] + s[1]).round() as i32 }
            }
            Custom => d.custom.as_ref().unwrap().bounds(&d.inputs),
            Affine2D | Straighten => self.affine_output_bounds(input_bounds(&d.inputs[0])),
            // A HARD blur's output covers its input's rect, whatever part of it a draw asks for (and so realizes).
            GaussianBlur if self.get_u32(2, 0) == 1 => input_bounds(&d.inputs[0]),
            // The filter family is realised; its output rect is the realised bitmap's placement.
            // ponytail: realises to read the rect (and again to fill it); the native graph caches,
            // but the conformance inputs are tiny — cache in Data if a large-input scene appears.
            GaussianBlur
            | DirectionalBlur
            | Shadow
            | ConvolveMatrix
            | Morphology
            | Sharpen
            | EdgeDetection
            | Emboss
            | DisplacementMap
            | Posterize => {
                // A reach filter over an INFINITE input (Border/Tile/… extend) has infinite output;
                // native defers it and the draw clips to the target. Realizing it here would allocate
                // the infinite rect (overflow). Return INF; `realize_rect`/`draw_image_effect` clip to
                // the target and the input is realized over [output + reach] (`realize_filter_input`).
                let ib = input_bounds(&d.inputs[0]);
                if ib.l == i32::MIN || ib.t == i32::MIN || ib.r == i32::MAX || ib.b == i32::MAX {
                    INF
                } else {
                    let (bmp, ox, oy) = self.realize_filter(None);
                    Rect { l: ox, t: oy, r: ox + bmp.width as i32, b: oy + bmp.height as i32 }
                }
            }
        }
    }

    /// The Crop/Tile rect property. Crop sorts each axis, so an inverted rect crops the same area (sc203: HARD and SOFT,
    /// either axis or both; r27 32807's [10, 10, 2, 2] crop drawn SOURCE_COPY), and maps it from DIPs to the graph's
    /// pixels as `(dip·dpi)/96` (multiply then divide in f32: +1 ULP for a non-integer edge like 30.7 → 30.70000267;
    /// plan-bcrop2's bottom band).
    fn rect_v4(&self) -> [f32; 4] {
        let r = self.get_v4(0, [f32::MIN, f32::MIN, f32::MAX, f32::MAX]);
        let d = self.0.borrow();
        if d.kind != Kind::Crop {
            return r;
        }
        let (dx, dy) = d.dpi;
        let r = [r[0].min(r[2]), r[1].min(r[3]), r[0].max(r[2]), r[1].max(r[3])];
        [r[0] * dx / 96.0, r[1] * dy / 96.0, r[2] * dx / 96.0, r[3] * dy / 96.0]
    }

    /// Crop rect → integer pixel rect, round-half-up, as native's Crop prepares it.
    fn crop_rect(&self) -> Rect {
        let r = self.rect_v4();
        let q = |v: f32| (v + 0.5).floor().clamp(-(1 << 30) as f32, (1 << 30) as f32) as i32;
        Rect { l: q(r[0]), t: q(r[1]), r: q(r[2]), b: q(r[3]) }
    }

    /// Whether a HighlightsShadows effect is *active* (any of Highlights/Shadows/Clarity exceeds
    /// FLT_EPSILON) — native then builds its transform graph (`PrepareForRender`), which reads
    /// neighbours (edge + blur mask), so the whole effect bakes to a bitmap rather than point-evaluates.
    fn hs_baked(&self, kind: Kind) -> bool {
        const EPS: f32 = 1.192_092_9e-7; // FLT_EPSILON
        kind == Kind::HighlightsShadows
            && (self.get_f32(0, 0.0).abs() >= EPS
                || self.get_f32(1, 0.0).abs() >= EPS
                || self.get_f32(2, 0.0).abs() >= EPS)
    }

    // --- evaluation (R1–R8, R21) --------------------------------------------------------------

    /// Premultiplied RGBA f32 at integer pixel `(x, y)` (centre `x+0.5, y+0.5`). A non-point-evaluable
    /// kind (blur/filter/realized transform / 2D affine) is baked to a bitmap once (memoized) and
    /// sampled 1:1 — the way native's graph orders it — so a point-eval consumer that reads such an
    /// input (e.g. Shadow → Composite) never reaches an aborting panic.
    fn eval(&self, x: i32, y: i32) -> [f32; 4] {
        let v = self.eval_unstored(x, y);
        let d = self.0.borrow();
        // The effect drawn into a float target writes it, through a Border as well (Kind::Border): nothing is stored.
        let float_drawn = FLOAT_DRAWN.with(|c| c.get()) == Rc::as_ptr(&self.0) as *const () as usize;
        match d.store {
            Some(_) if float_drawn => v,
            Some(p) => quantize_to(p, v),
            // A custom shader drawn as its own pass stores its output at its precision before what reads it (VM: a box
            // shader under an Opacity rounds to the byte first, 874 px ±1 otherwise, into a float target too).
            None if d.kind == Kind::Custom && d.custom.as_ref().is_some_and(|c| c.draws_own_pass()) && !float_drawn => {
                drop(d);
                quantize_to(self.out_precision(), v)
            }
            None => v,
        }
    }

    /// [`Effect::eval`] before the store of this effect's own intermediate.
    fn eval_unstored(&self, x: i32, y: i32) -> [f32; 4] {
        let kind = self.0.borrow().kind;
        // A pixel-shader FILTER feeding this point-wise eval is a LINKED intermediate: read its float
        // output (no 8-bit store), so the chain stays f32 and quantizes once at the output node (R11).
        // At an already-float buffer precision `realized()` is float anyway; this matters at 8bpc.
        if kind.is_linkable_filter() && self.hp().is_none() {
            let r = self.realized_f();
            return bitmap_texel(&r.0, x - r.1, y - r.2);
        }
        if is_realized_transform(kind) || kind.is_filter() || kind == Kind::Affine2D || self.hs_baked(kind) {
            let r = self.realized();
            return bitmap_texel(&r.0, x - r.1, y - r.2);
        }
        let d = self.0.borrow();
        match d.kind {
            Kind::Flood => {
                // Stored verbatim, no premultiply. At an 8-bit precision the colour reaches the effects it feeds
                // clamped to [0, 1], as an 8-bit intermediate holds it (r27 32893: a Composite over a (2, −1, 0.5,
                // 0.5) Flood composited its −1 green).
                let v = self.get_v4(0, [0.0, 0.0, 0.0, 1.0]);
                if self.hp().is_none() { v.map(|c| c.clamp(0.0, 1.0)) } else { v }
            }
            Kind::Crop => {
                if self.get_u32(1, 0) == 1 {
                    // HARD: integer clip, bit-exact passthrough.
                    if self.crop_rect().contains(x, y) {
                        eval_input(&d.inputs[0], x, y)
                    } else {
                        [0.0; 4]
                    }
                } else {
                    // SOFT: antialiased edge — scale the premultiplied input by the fractional overlap
                    // of the crop RECT (in the graph's pixels) with the pixel box (analytic box coverage).
                    let r = self.rect_v4();
                    let cx = (r[2].min(x as f32 + 1.0) - r[0].max(x as f32)).clamp(0.0, 1.0);
                    let cy = (r[3].min(y as f32 + 1.0) - r[1].max(y as f32)).clamp(0.0, 1.0);
                    let cov = cx * cy;
                    eval_input(&d.inputs[0], x, y).map(|c| c * cov)
                }
            }
            Kind::Border => {
                // An input unbounded on an axis (another Border, a Flood, a Tile) has no edge there to
                // extend: that axis passes through (r27 32852).
                let b = input_bounds(&d.inputs[0]);
                let fx = if b.l == i32::MIN || b.r == i32::MAX { x } else { b.l + fold(x - b.l, b.width() as i32, self.get_u32(0, 0)) };
                let fy = if b.t == i32::MIN || b.b == i32::MAX { y } else { b.t + fold(y - b.t, b.height() as i32, self.get_u32(1, 0)) };
                // A Border drawn into a float target is the draw's sampler addressing, not a pass: what it reads is the
                // draw's last pass (VM: a box under a Border keeps its float output, under an Opacity rounds to the byte).
                let me = Rc::as_ptr(&self.0) as *const () as usize;
                match &d.inputs[0] {
                    Input::Effect(e) if FLOAT_DRAWN.with(|c| c.get()) == me => {
                        let prev = FLOAT_DRAWN.with(|c| c.replace(float_root(e)));
                        let v = e.eval(fx, fy);
                        FLOAT_DRAWN.with(|c| c.set(prev));
                        v
                    }
                    inp => eval_input(inp, fx, fy),
                }
            }
            Kind::Tile => {
                let r = self.crop_rect(); // Tile RECT uses the same round-half-up DIP→px
                let sx = tile_coord(x, r.l, r.width() as i32);
                let sy = tile_coord(y, r.t, r.height() as i32);
                eval_input(&d.inputs[0], sx, sy)
            }
            Kind::Composite => {
                let mode = self.get_u32(0, 0);
                let mut dst = eval_input(&d.inputs[0], x, y);
                for inp in &d.inputs[1..] {
                    // BOUNDED_SOURCE_COPY copies the source over its own bounds only, keeping the destination past them
                    // (VM: a 16×10 destination under an 11×14 source keeps its own texels beside the source).
                    if mode == 11 && !input_bounds(inp).contains(x, y) {
                        continue;
                    }
                    dst = composite_px(mode, eval_input(inp, x, y), dst);
                }
                dst
            }
            Kind::ColorMatrix => builtin::color::color_matrix(
                eval_input(&d.inputs[0], x, y),
                &self.color_matrix_rows(),
                builtin::color::AlphaMode::from_u32(self.get_u32(1, 1)), // 1 PREMUL (default), 2 STRAIGHT
                matches!(d.props.get(&2), Some(Prop::Bool(true))),
            ),
            Kind::Color(op) => self.eval_color(op, x, y),
            Kind::OpacityMetadata => eval_input(&d.inputs[0], x, y), // metadata-only: passthrough
            Kind::HighlightsShadows => {
                // Active HS bakes via `realize_filter` (the mask needs neighbours); `eval` intercepts
                // it at the top (`hs_baked`). This point path is the inactive pass-through — native
                // builds no graph until a strength exceeds FLT_EPSILON (PrepareForRender).
                eval_input(&d.inputs[0], x, y)
            }
            Kind::BitmapSource => self.eval_bitmapsource(x, y),
            Kind::Histogram => [0.0; 4], // empty image output (the result is HistogramOutput)
            Kind::YCbCr => self.eval_ycbcr(x, y),
            Kind::ColorManagement => {
                // 1:1 colour transform in f32 (pull graph keeps the context float precision until the
                // single output quantise); identity when no transform is set.
                let in0 = eval_input(&d.inputs[0], x, y);
                d.cms.as_ref().map_or(in0, |t| t.apply(in0))
            }
            Kind::Blend => {
                let (dst, src) = (eval_input(&d.inputs[0], x, y), eval_input(&d.inputs[1], x, y));
                // Dissolve (mode 4) reads the scene position for its positional noise (builtin::blend).
                builtin::blend::blend(self.get_u32(0, 0), dst, src, [x as f32 + 0.5, y as f32 + 0.5])
            }
            Kind::Arithmetic => builtin::blend::arithmetic_composite(
                self.get_v4(0, [0.0; 4]),
                self.get_bool(1, false),
                eval_input(&d.inputs[0], x, y),
                eval_input(&d.inputs[1], x, y),
            ),
            Kind::CrossFade => builtin::blend::cross_fade(
                self.get_f32(0, 0.0),
                eval_input(&d.inputs[0], x, y),
                eval_input(&d.inputs[1], x, y),
            ),
            Kind::Opacity => builtin::blend::opacity(self.get_f32(0, 1.0), eval_input(&d.inputs[0], x, y)),
            Kind::LookupTable3D => {
                let lut = self.lut3d();
                lut.sample(eval_input(&d.inputs[0], x, y))
            }
            // A custom effect draws only its output rect: past it, it is transparent however its shader would read
            // there (VM: a sub-texel tap drawn at a fractional offset leaves the row above the image transparent).
            Kind::Custom if !self.bounds().contains(x, y) => [0.0; 4],
            Kind::Custom => d.custom.as_ref().unwrap().eval(&d.inputs, x, y, d.render_precision, FLOAT_DRAWN.with(|c| c.get()) == Rc::as_ptr(&self.0) as *const () as usize),
            // The non-point-evaluable kinds (2D affine, straighten, realized transforms, filters/blurs)
            // are baked and sampled by the guard above and never reach here; transparent black is the
            // D2D empty result, so this stays graceful rather than panicking on any future kind.
            Kind::Affine2D | Kind::Straighten => [0.0; 4],
            k if is_realized_transform(k) => [0.0; 4],
            _ => [0.0; 4],
        }
    }

    /// YCbCr (BT.601): Y from input 0 (point, full-res), CbCr from input 1 bilinearly upsampled to
    /// the Y resolution, then the conversion pixel shader's arithmetic in its order. The chroma subsampling mode
    /// (4:2:0 / 4:2:2 / 4:4:0 / 4:4:4) is driven entirely by input 1's size relative to input 0, so
    /// one bilinear path covers them all — a half-texel-centred sample `(out+0.5)·in/out − 0.5`,
    /// edge-clamped (verified byte-exact against native 4:2:0 and 4:4:4 WARP captures).
    fn eval_ycbcr(&self, x: i32, y: i32) -> [f32; 4] {
        let d = self.0.borrow();
        let luma = eval_input(&d.inputs[0], x, y)[0];
        let b0 = input_bounds(&d.inputs[0]);
        let b1 = input_bounds(&d.inputs[1]);
        let (ow, oh) = (b0.width() as f32, b0.height() as f32);
        let (iw, ih) = (b1.width() as i32, b1.height() as i32);
        // Output-local position → input-1 texel space; clamp into the texel-centre range.
        let u = (((x - b0.l) as f32 + 0.5) * iw as f32 / ow - 0.5);
        let v = (((y - b0.t) as f32 + 0.5) * ih as f32 / oh - 0.5);
        let c = sample_bilinear(&d.inputs[1], b1, u, v);
        let cb = c[0] - 128.0 / 255.0;
        let cr = c[1] - 128.0 / 255.0;
        let sat = |v: f32| v.clamp(0.0, 1.0);
        let g = sat((cb * -0.34414 + cr * -0.71414) + luma); // dp2 then +Y
        [sat(cr * 1.402 + luma), g, sat(cb * 1.772 + luma), 1.0]
    }

    /// Histogram's `HistogramOutput`: the normalised per-bin counts of input 0 (compute shader
    /// ported as a plain count, which is order-independent). For each input pixel: unpremultiply the
    /// selected channel (ChannelSelect, property 1; the alpha channel 3 uses the raw stored alpha),
    /// `bin = min(floor(channel · NumBins), NumBins−1)`, count, then divide each bin by the pixel
    /// count. NumBins is property 0 (default 256).
    pub fn histogram_output(&self) -> Vec<f32> {
        let num_bins = self.get_u32(0, 256).max(1) as usize;
        let channel = self.get_u32(1, 0).min(3) as usize;
        let d = self.0.borrow();
        let b = input_bounds(&d.inputs[0]);
        let total = (b.width() * b.height()).max(1) as f32;
        let mut counts = vec![0u32; num_bins];
        for y in b.t..b.b {
            for x in b.l..b.r {
                let p = eval_input(&d.inputs[0], x, y);
                let a = p[3];
                let ch = if channel == 3 { a } else if a != 0.0 { p[channel] / a } else { 0.0 };
                let bin = (num_bins as f32 * ch).floor().clamp(0.0, (num_bins - 1) as f32) as usize;
                counts[bin] += 1;
            }
        }
        counts.iter().map(|&c| c as f32 / total).collect()
    }

    /// BitmapSource: the WIC source (bridged as input 0) scaled by the Scale property (property 1,
    /// the half-texel bilinear shared with the chroma upsample) and reoriented by the Orientation
    /// property (property 5, an axis-aligned flip/rotate). Non-linear InterpolationMode is a follow-up.
    fn eval_bitmapsource(&self, x: i32, y: i32) -> [f32; 4] {
        let d = self.0.borrow();
        let scale = self.get_v2(1, [1.0, 1.0]);
        let orient = self.get_u32(5, 1);
        let b = input_bounds(&d.inputs[0]);
        let (sw, sh) = bitmapsource_scaled(b, scale);
        let (sx, sy) = inverse_orient(orient, x, y, sw, sh);
        let u = (sx as f32 + 0.5) / scale[0] - 0.5;
        let v = (sy as f32 + 0.5) / scale[1] - 0.5;
        sample_bilinear_fixed8(&d.inputs[0], b, u, v)
    }

    /// BitmapSource output bounds: the source scaled by Scale, then the Orientation's 90°/270°
    /// rotations swap width and height.
    fn bitmapsource_bounds(&self) -> Rect {
        let scale = self.get_v2(1, [1.0, 1.0]);
        let orient = self.get_u32(5, 1);
        let (sw, sh) = bitmapsource_scaled(input_bounds(&self.0.borrow().inputs[0]), scale);
        let (ow, oh) = if (5..=8).contains(&orient) { (sh, sw) } else { (sw, sh) };
        Rect { l: 0, t: 0, r: ow, b: oh }
    }

    /// Dispatch a 2C colour/transfer effect to its [`builtin::color`] kernel, reading its properties.
    fn eval_color(&self, op: ColorOp, x: i32, y: i32) -> [f32; 4] {
        use builtin::color as fx;
        use ColorOp::*;
        let d = self.0.borrow();
        let in0 = eval_input(&d.inputs[0], x, y);
        match op {
            Invert => fx::invert(in0),
            Grayscale => fx::grayscale(in0),
            LuminanceToAlpha => fx::luminance_to_alpha(in0),
            Exposure => fx::exposure(in0, self.get_f32(0, 0.0)),
            Brightness => fx::brightness(in0, self.get_v2(0, [1.0, 1.0]), self.get_v2(1, [0.0, 0.0])),
            Contrast => fx::contrast(in0, self.get_f32(0, 0.0), self.get_bool(1, false)),
            HueRotation => fx::hue_rotation(in0, self.get_f32(0, 0.0)),
            Saturation => fx::saturation(in0, self.get_f32(0, 0.5)),
            Sepia => fx::sepia(in0, self.get_f32(0, 0.5), fx::AlphaMode::from_u32(self.get_u32(1, 1))),
            Tint => fx::tint(in0, self.get_v4(0, [1.0, 1.0, 1.0, 1.0]), self.get_bool(1, false)),
            TemperatureTint => fx::temperature_tint(in0, self.get_f32(0, 0.0), self.get_f32(1, 0.0)),
            GammaTransfer => {
                let ch = std::array::from_fn(|c| fx::GammaChannel {
                    amplitude: self.get_f32(c as u32 * 4, 1.0),
                    exponent: self.get_f32(c as u32 * 4 + 1, 1.0),
                    offset: self.get_f32(c as u32 * 4 + 2, 0.0),
                    disable: self.get_bool(c as u32 * 4 + 3, false),
                });
                fx::gamma_transfer(in0, &ch, self.get_bool(16, false))
            }
            LinearTransfer => {
                let ch = std::array::from_fn(|c| fx::LinearChannel {
                    y_intercept: self.get_f32(c as u32 * 3, 0.0),
                    slope: self.get_f32(c as u32 * 3 + 1, 1.0),
                    disable: self.get_bool(c as u32 * 3 + 2, false),
                });
                fx::linear_transfer(in0, &ch, self.get_bool(12, false))
            }
            // Each channel's table is property 2c (the default [0, 1]), its disable flag 2c + 1 (VM: a 3-entry red table
            // maps an over-range red to its last entry, r27 32888).
            DiscreteTransfer => {
                let t: [Rc<[f32]>; 4] = std::array::from_fn(|c| self.get_floats(c as u32 * 2, &[0.0, 1.0]));
                let dis = std::array::from_fn(|c| self.get_bool(c as u32 * 2 + 1, false));
                fx::discrete_transfer(in0, [&t[0], &t[1], &t[2], &t[3]], dis, self.get_bool(8, false))
            }
            TableTransfer => {
                let t: [Rc<[f32]>; 4] = std::array::from_fn(|c| self.get_floats(c as u32 * 2, &[0.0, 1.0]));
                let dis = std::array::from_fn(|c| self.get_bool(c as u32 * 2 + 1, false));
                fx::table_transfer(in0, [&t[0], &t[1], &t[2], &t[3]], dis, self.get_bool(8, false))
            }
            Premultiply => fx::premultiply_effect(in0),
            UnPremultiply => fx::unpremultiply_effect(in0),
            AlphaMask => fx::alpha_mask(in0, eval_input(&d.inputs[1], x, y)),
            ChromaKey => fx::chroma_key(in0, self.get_v3(0, [0.0, 0.0, 0.0]), self.get_f32(1, 0.1), self.get_bool(2, false)),
            WhiteLevel => fx::white_level(in0, self.get_f32(0, 80.0), self.get_f32(1, 80.0)),
            Vignette => {
                // The 9-quad mesh is built once per input rect and property set (a per-pixel rebuild is O(n²)).
                let b = input_bounds(&d.inputs[0]);
                let (color, ts, st) = (self.get_v4(0, [0.0, 0.0, 0.0, 1.0]), self.get_f32(1, 0.1), self.get_f32(2, 0.5));
                let key = [b.l, b.t, b.r, b.b, color[0].to_bits() as i32, color[1].to_bits() as i32, color[2].to_bits() as i32, color[3].to_bits() as i32, ts.to_bits() as i32, st.to_bits() as i32];
                let mut cache = d.vignette.borrow_mut();
                if cache.as_ref().map(|c| c.0) != Some(key) {
                    *cache = Some((key, Rc::new(fx::Vignette::new([b.l, b.t, b.r, b.b], color, ts, st))));
                }
                cache.as_ref().unwrap().1.pixel(x, y, in0)
            }
            HdrToneMap => {
                // The tone-map's matrices and anchors are built from the three properties only, so cache
                // the instance and reuse it across pixels instead of rebuilding it (with a matrix inverse)
                // every pixel.
                let (ml, md, mode) = (self.get_f32(0, 4000.0), self.get_f32(1, 300.0), self.get_u32(2, 0));
                let key = [ml.to_bits() as i32, md.to_bits() as i32, mode as i32];
                let mut cache = d.hdr.borrow_mut();
                if cache.as_ref().map(|c| c.0) != Some(key) {
                    *cache = Some((key, Rc::new(fx::HdrToneMap::new(ml, md, mode))));
                }
                cache.as_ref().unwrap().1.pixel(in0)
            }
            RgbToHue => fx::rgb_to_hue(in0, self.get_u32(0, 0)),
            HueToRgb => fx::hue_to_rgb(in0, self.get_u32(0, 0)),
        }
    }

    fn color_matrix_rows(&self) -> [f32; 20] {
        match self.0.borrow().props.get(&0) {
            Some(Prop::M54(m)) => *m,
            _ => {
                // identity 5×4: diag 1, zero translation
                let mut m = [0.0; 20];
                for i in 0..4 {
                    m[i * 4 + i] = 1.0;
                }
                m
            }
        }
    }

    // --- 2D affine resampler (R1–R3) ----------------------------------------------------------

    fn affine_matrix(&self) -> [f32; 6] {
        if self.0.borrow().kind == Kind::Straighten {
            return self.straighten_matrix();
        }
        match self.0.borrow().props.get(&2) {
            // The matrix is in DIPs; in device-pixel space (dpi/96 = k) it is conjugated by that scale: the linear part
            // stays and the translation scales by k (VM probes at 120 and 192 dpi).
            Some(Prop::M32(m)) => { let k = DIP_SCALE.with(|s| s.get()); [m[0], m[1], m[2], m[3], m[4] * k, m[5] * k] }
            _ => [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        }
    }

    /// The 3×2 affine for the Straighten effect, from the Angle
    /// (°, property 0), MaintainSize (property 1) and input-0 bounds. A rotation about the input
    /// centre — `RotationZ` with the *signed* angle, converted to radians in f32 (`×π/180`) — optionally
    /// scaled so the rotated image still covers the original (`sin|θ|·aspect + cos|θ|`, with the scale's
    /// radians converted in f64 as native does). Below ~FLT_EPSILON° native sets no transform (identity).
    fn straighten_matrix(&self) -> [f32; 6] {
        let angle = self.get_f32(0, 0.0);
        if angle.abs() <= 1.1920929e-7 {
            return [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        }
        let b = input_bounds(&self.0.borrow().inputs[0]);
        let (l, t) = (b.l as f32, b.t as f32);
        let (w, h) = (b.r as f32 - l, b.b as f32 - t);
        let (cx, cy) = (w * 0.5 + l, h * 0.5 + t);
        // Rotation: signed angle × (π/180 as f32), then the Windows CRT's sinf/cosf (via crt for cross-platform parity).
        let rad = angle * f32::from_bits(0x3c8e_fa35);
        let (sin, cos) = (crate::crt::system_crt::sinf(rad), crate::crt::system_crt::cosf(rad));
        let s = if self.get_bool(1, false) {
            let aspect = if h <= w { w / h } else { h / w };
            let rad_s = (angle.abs() as f64 * std::f64::consts::PI / 180.0) as f32;
            crate::crt::system_crt::sinf(rad_s) * aspect + crate::crt::system_crt::cosf(rad_s)
        } else {
            1.0
        };
        let (a, b_, c, d) = (s * cos, s * sin, -s * sin, s * cos);
        // Native Straighten composes rotation about the center, then scale about the center.
        [a, b_, c, d, ((cx - (cos * cx - sin * cy)) - cx) * s + cx, ((cy - (sin * cx + cos * cy)) - cy) * s + cy]
    }

    /// (linear?, soft-border?) for the affine resampler. Affine2D reads InterpolationMode(0)/
    /// BorderMode(1); Straighten's interpolation is its ScaleMode (property 2) and it always uses the
    /// soft border (its affine transform's default).
    fn affine_sampling(&self) -> (bool, bool) {
        if self.0.borrow().kind == Kind::Straighten {
            (self.get_u32(2, 1) != 0, true)
        } else {
            (self.get_u32(0, 1) != 0, self.get_u32(1, 0) == 0)
        }
    }

    /// Output bounds = forward transform of the input rect grown by the filter half-kernel (0.5 texel
    /// for linear), bounding-boxed (the 0.5-texel reach is the soft halo seen in the oracle).
    fn affine_output_bounds(&self, input: Rect) -> Rect {
        if input.width() == 0 || input.height() == 0 {
            return input;
        }
        // Straighten with MaintainSize keeps the output the same size as the input (it scales the
        // rotated image up to cover the original rect and crops to it), so the output rect is the
        // input rect, not the transformed bbox.
        if self.0.borrow().kind == Kind::Straighten && self.get_bool(1, false) {
            return input;
        }
        let mut m = self.affine_matrix();
        // HIGH_QUALITY_CUBIC of a pure scale (each > 0.01, no translation) is Direct2D's Scale effect. HARD, its last pass
        // maps the input rect by the whole scale and keeps the pixels whose centres it holds (top-left rounding).
        if self.0.borrow().kind == Kind::Affine2D && self.get_u32(0, 1) == 5 && self.get_u32(1, 0) == 1
            && m[1] == 0.0 && m[2] == 0.0 && m[0] > 0.01 && m[3] > 0.01 && m[4] == 0.0 && m[5] == 0.0
        {
            let edge = |v: i32, s: f32| (v as f32 * s - 0.5).ceil() as i32;
            return Rect { l: edge(input.l, m[0]), t: edge(input.t, m[3]), r: edge(input.r, m[0]), b: edge(input.b, m[3]) };
        }
        let mut input = input;
        let mode = self.get_u32(if self.0.borrow().kind == Kind::Straighten { 2 } else { 0 }, 1);
        if mode == 5 {
            let affine = self.0.borrow().kind == Kind::Affine2D;
            let factor = hq_prescale_factor(m);
            let hard = affine && self.get_u32(1, 0) == 1;
            let mut r = [input.l, input.t, input.r, input.b];
            for axis in 0..2 {
                let j = axis * 2;
                let scale = if affine { factor[axis] } else { crate::warpmath::sqrt(m[j] * m[j] + m[j + 1] * m[j + 1]) };
                if scale > 0.0 && scale < 1.0 {
                    (r[axis], r[axis + 2]) = hq_prescale_span(r[axis], r[axis + 2], scale, hard);
                    let k = 1.0 / scale;
                    m[j] *= k; m[j + 1] *= k;
                }
            }
            input = Rect { l: r[0], t: r[1], r: r[2], b: r[3] };
        }
        // HARD (not HIGH_QUALITY_CUBIC, whose prescale has its own rect): the transformed input rect itself, no filter
        // reach, rounded by the top-left rule — the pixels whose centres it holds.
        let hard = self.0.borrow().kind == Kind::Affine2D && self.get_u32(1, 0) == 1 && mode != 5;
        let k = match mode {
            _ if hard => [0.0; 2],
            0 => [0.0; 2],
            2 | 5 => [1.5; 2],
            3 | 4 => {
                let det = (m[0] * m[3] - m[1] * m[2]).abs();
                [(m[3].abs() + m[2].abs()) / det * 0.5 + 0.5, (m[1].abs() + m[0].abs()) / det * 0.5 + 0.5]
            }
            _ => [0.5; 2],
        };
        let (il, it) = (input.l as f32 - k[0], input.t as f32 - k[1]);
        let (ir, ib) = (input.r as f32 + k[0], input.b as f32 + k[1]);
        let fwd = |x: f32, y: f32| [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]];
        let corners = [fwd(il, it), fwd(ir, it), fwd(il, ib), fwd(ir, ib)];
        let minx = corners.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let maxx = corners.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
        let miny = corners.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        let maxy = corners.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);
        if hard {
            return Rect { l: (minx - 0.5).ceil() as i32, t: (miny - 0.5).ceil() as i32, r: (maxx - 0.5).ceil() as i32, b: (maxy - 0.5).ceil() as i32 };
        }
        Rect { l: minx.floor() as i32, t: miny.floor() as i32, r: maxx.ceil() as i32, b: maxy.ceil() as i32 }
    }

    // --- realisation ---------------------------------------------------------------------------

    /// Realise this effect's output over its full (clipped) bounds, with the bounds' top-left origin
    /// in DIP space. Used when another node needs this effect as a sampleable texture.
    fn realize(&self) -> (Bitmap, i32, i32) {
        // An unbounded output (Scale, DPI and 3D under hard sampling, Turbulence, lighting) bakes over
        // the draw's viewport, the only part a point-wise consumer reads.
        let mut b = self.bounds();
        if b.l == i32::MIN || b.t == i32::MIN || b.r == i32::MAX || b.b == i32::MAX {
            // ponytail: without a viewport (an output-bounds query) a fixed window around the origin; carry the
            // consumer's request down if one reads past it.
            let v = DRAW_VIEWPORT.with(|v| v.get()).unwrap_or(Rect { l: -1024, t: -1024, r: 1024, b: 1024 });
            b = b.intersect(v);
        }
        (self.realize_rect(b), b.l, b.t)
    }

    /// This effect's realized output bitmap (+ device origin), computed once and memoized. Used when a
    /// point-eval consumer reads a non-point-evaluable input: native bakes such an input to a texture
    /// before sampling it, rather than point-evaluating it (which is impossible for a blur/filter).
    fn realized(&self) -> Rc<(Bitmap, i32, i32)> {
        if let Some(r) = self.0.borrow().realized.clone() {
            return r;
        }
        let r = Rc::new(self.realize());
        self.0.borrow_mut().realized = Some(r.clone());
        r
    }

    /// A pixel-shader FILTER's realized output as a linked intermediate for a point-wise consumer:
    /// realized at R32 FLOAT with the filter's f32 arithmetic (`realize_filter_hp`), so the downstream
    /// point-wise effect reads the shader's unquantised value — native links the two into one shader
    /// (no 8-bit intermediate store) even at an 8bpc buffer precision (R11; precision-03 conv→ColorMatrix).
    /// At a buffer precision already > 8bpc the output store is float too, so `realized()` suffices; this
    /// is the 8bpc case. `realize_filter_hp(5, None)` realizes the whole (finite) filter output in float.
    fn realized_f(&self) -> Rc<(Bitmap, i32, i32)> {
        if let Some(r) = self.0.borrow().realized_f.clone() {
            return r;
        }
        let r = Rc::new(self.realize_filter_hp(5, None));
        self.0.borrow_mut().realized_f = Some(r.clone());
        r
    }

    /// Realise the output over the explicit (finite) rect `b` into a premultiplied BGRA8 bitmap. The
    /// single quantisation of the f32 graph (caller clips infinite bounds to the target first).
    fn realize_rect(&self, b: Rect) -> Bitmap {
        self.realize_rect_at(b, [0.0; 2])
    }

    fn realize_rect_at(&self, b: Rect, origin: [f32; 2]) -> Bitmap {
        self.realize_rect_clipped(b, origin, None)
    }

    fn realize_rect_clipped(&self, b: Rect, origin: [f32; 2], viewport: Option<Rect>) -> Bitmap {
        if let Some(p) = self.hp() {
            return self.realize_rect_hp(b, p, origin);
        }
        let kind = self.0.borrow().kind;
        if is_realized_transform(kind) {
            return self.realize_transform(b, origin, b);
        }
        // DoS guard (eval/affine/filter realize only): an INPUT-LESS infinite generator (Flood) drawn
        // under a near-singular device transform gets a ~1e9/side back-mapped clip, so this `vec` would
        // allocate petabytes (fuzz 169). is_realized_transform/hp effects returned above (they clip to
        // their input and never hit this), so here a real `b` is the effect's finite output or the
        // target-sized clip; cap each side (the degenerate output under such a transform is sub-pixel).
        let mut b = b;
        const REALIZE_CAP: i64 = 1 << 12; // ≤4096 px/side (≤64 MB)
        if b.r as i64 - b.l as i64 > REALIZE_CAP {
            b.r = (b.l as i64 + REALIZE_CAP) as i32;
        }
        if b.b as i64 - b.t as i64 > REALIZE_CAP {
            b.b = (b.t as i64 + REALIZE_CAP) as i32;
        }
        let (w, h) = (b.width(), b.height());
        let mut data = vec![0u8; w as usize * h as usize * 4];
        if kind == Kind::Affine2D || kind == Kind::Straighten {
            self.realize_affine(b, &mut data, origin);
        } else if kind.is_filter() || self.hs_baked(kind) {
            self.realize_filter_rect(b, &mut data, viewport);
        } else {
            for py in 0..h as i32 {
                for px in 0..w as i32 {
                    let c = self.eval(b.l + px, b.t + py);
                    store(&mut data, (py as usize * w as usize + px as usize) * 4, c);
                }
            }
        }
        Bitmap {
            width: w,
            height: h,
            premul_bgra: data,
            format: DXGI_FORMAT_B8G8R8A8_UNORM,
            alpha: D2D1_ALPHA_MODE_PREMULTIPLIED,
            dpi: (96.0, 96.0),
        }
    }

    /// [`realize_rect`] above 8bpc: a premultiplied R32G32B32A32_FLOAT bitmap holding each value at the
    /// effect's precision (32-bit float keeps its range; 16-bit float rounds to half; 16-bit UNORM clamps).
    /// Generators and lighting retain their existing byte intermediate.
    fn realize_rect_hp(&self, b: Rect, p: u32, origin: [f32; 2]) -> Bitmap {
        let kind = self.0.borrow().kind;
        let (w, h) = (b.width(), b.height());
        let px: Vec<[f32; 4]> = if kind.is_filter() || self.hs_baked(kind) {
            let (full, fox, foy) = self.realize_filter(Some(b));
            (0..h as i32).flat_map(|py| (0..w as i32).map(move |px| (px, py))).map(|(px, py)| bitmap_texel(&full, b.l + px - fox, b.t + py - foy)).collect()
        } else if kind == Kind::Affine2D || kind == Kind::Straighten {
            self.realize_affine_hp(b, origin)
        } else if is_realized_transform(kind) {
            let bmp = self.realize_transform(b, origin, b);
            (0..(w * h) as i32).map(|i| bitmap_texel(&bmp, i % w as i32, i / w as i32)).collect()
        } else {
            (0..h as i32).flat_map(|py| (0..w as i32).map(move |px| (px, py))).map(|(px, py)| self.eval(b.l + px, b.t + py)).collect()
        };
        float_bitmap(w, h, px.into_iter().map(|c| quantize_to(p, c)))
    }

    /// Realise a float-scene-position transform-family effect (resampler / generator / lighting) over
    /// rect `b` by building its `builtin::transform` kernel and baking it (texel centre at
    /// `b.l + px + 0.5`). This is the engine's wiring of the 2C transform kernels — the exact
    /// construction verified by the transform sampler sweeps against native WARP.
    fn realize_transform(&self, b: Rect, origin: [f32; 2], viewport: Rect) -> Bitmap {
        if let Some(g) = self.lighting_stage() {
            return g.realize_rect_clipped(b, origin, Some(viewport));
        }
        use builtin::transform as tfx;
        let d = self.0.borrow();
        let kind = d.kind;
        let highp = self.hp().is_some() && matches!(kind, Kind::Scale | Kind::Dpi | Kind::Transform3D | Kind::Perspective3D);
        let (w, h) = (b.width(), b.height());
        let (ox, oy) = (b.l as f32, b.t as f32);
        let interp = |i| tfx::Interp::from_mode(i);
        let border = |i| tfx::Border::from_mode(i);
        let scale3 = |c: [f32; 3], k: f32| [c[0] * k, c[1] * k, c[2] * k];

        // Kinds that take an input realise it once to a sampleable bitmap at its device origin.
        // Lighting reads a 3×3 neighbourhood: an infinite input (a Flood) realizes over `b` grown by 1, not whole.
        let ib = (!matches!(kind, Kind::Turbulence)).then(|| input_bounds(&d.inputs[0]));
        let lit_infinite = ib.is_some_and(beyond_realize_cap) && is_lighting(kind);
        let realized_in = ib.map(|_| if lit_infinite {
            realize_input_at(&d.inputs[0], Rect { l: b.l.saturating_sub(1), t: b.t.saturating_sub(1), r: b.r.saturating_add(1), b: b.b.saturating_add(1) }, highp)
        } else if highp {
            realize_input_hp(&d.inputs[0])
        } else {
            realize_input(&d.inputs[0])
        });
        let input = realized_in.as_ref().map(|(bm, iox, ioy)| {
            let input = tfx::Input::new(bm, *iox, *ioy);
            if highp { input.with_float_output() } else { input }
        });
        let inp = || input.as_ref().unwrap();

        match kind {
            Kind::Atlas => {
                let r = self.get_v4(0, [f32::MIN, f32::MIN, f32::MAX, f32::MAX]).map(|v| v.round() as i32);
                tfx::to_bitmap_precision(w, h, ox, oy, highp, tfx::atlas(inp(), r))
            }
            Kind::Scale => {
                // The centre point is in DIPs: in device-pixel space it scales by dpi/96 (sc124: at 120 dpi a ×2 Scale
                // centred at 2.43 DIPs pivots at 3.04 px, on its bitmap input's own pixels).
                let k = DIP_SCALE.with(|s| s.get());
                let m = tfx::scale_matrix(self.get_v2(0, [1.0, 1.0]), self.get_v2(1, [0.0, 0.0]).map(|v| v * k));
                let mode = match interp(self.get_u32(2, 1)) {
                    tfx::Interp::HqCubic(_) => tfx::Interp::HqCubic(self.get_f32(4, 0.0).clamp(0.0, 1.0)),
                    mode => mode,
                };
                // A content-realized (direct bitmap) input's `Sample` rounds the Q8 texel address
                // (cvtps2dq) at a non-power-of-two size; an intermediate is realized into a
                // power-of-two atlas, so its texcoord normalizes by the atlas dimension (which shifts
                // the sub-texel float by a ULP versus the content dimension) and the address floors.
                let (bm, iox, ioy) = realized_in.as_ref().unwrap();
                let is_bitmap = matches!(&d.inputs[0], Input::Bitmap(_));
                let mut si = tfx::Input::new(bm, *iox, *ioy);
                if highp { si = si.with_float_output(); }
                if is_bitmap && (!bm.width.is_power_of_two() || !bm.height.is_power_of_two()) {
                    si = si.with_rne_address();
                }
                if let tfx::Interp::HqCubic(sharpness) = mode {
                    if m[0] > 0.0 && m[3] > 0.0 && (m[0] < 1.0 || m[3] < 1.0) {
                        return tfx::hq_downscale(&si, m, sharpness, border(self.get_u32(3, 0)), [b.l, b.t, b.r, b.b], self.out_precision());
                    }
                }
                if !is_bitmap { si = si.in_atlas(); }
                tfx::to_bitmap_precision(w, h, ox, oy, highp, tfx::affine_at(&si, m, mode, border(self.get_u32(3, 0)), origin, true))
            }
            Kind::Dpi => {
                let m = tfx::dpi_matrix([d.dpi.0, d.dpi.1], self.get_v2(2, [96.0, 96.0]));
                if self.get_u32(0, 1) == 5 && m[0] > 0.0 && m[3] > 0.0 && (m[0] < 1.0 || m[3] < 1.0) {
                    return tfx::hq_downscale(inp(), m, 0.0, border(self.get_u32(1, 0)), [b.l, b.t, b.r, b.b], self.out_precision());
                }
                tfx::to_bitmap_precision(w, h, ox, oy, highp, tfx::affine_at(inp(), m, interp(self.get_u32(0, 1)), border(self.get_u32(1, 0)), origin, true))
            }
            Kind::Transform3D => {
                let m = self.get_m44(2, IDENTITY_M44).map(|v| v as f64);
                tfx::to_bitmap_precision(w, h, ox, oy, highp, tfx::projective_rect_at(inp(), m, interp(self.get_u32(0, 1)), border(self.get_u32(1, 0)), origin, Some([viewport.l as f32, viewport.t as f32, viewport.r as f32, viewport.b as f32])))
            }
            Kind::Perspective3D => {
                let m = tfx::perspective_matrix(
                    self.get_f32(2, 1000.0),
                    self.get_v2(3, [0.0, 0.0]),
                    self.get_v3(4, [0.0, 0.0, 0.0]),
                    self.get_v3(5, [0.0, 0.0, 0.0]),
                    self.get_v3(6, [0.0, 0.0, 0.0]),
                    self.get_v3(7, [0.0, 0.0, 0.0]),
                );
                tfx::to_bitmap_precision(w, h, ox, oy, highp, tfx::projective_rect_at(inp(), m, interp(self.get_u32(0, 1)), border(self.get_u32(1, 0)), origin, Some([viewport.l as f32, viewport.t as f32, viewport.r as f32, viewport.b as f32])))
            }
            Kind::Turbulence => {
                let o = self.get_v2(0, [0.0, 0.0]);
                let s = self.get_v2(1, [0.0, 0.0]);
                let rect = [o[0].round() as i32, o[1].round() as i32, (o[0] + s[0]).round() as i32, (o[1] + s[1]).round() as i32];
                let turb = tfx::Turbulence::new(self.get_i32(4, 0), self.get_v2(2, [0.01, 0.01]), self.get_u32(3, 1) as i32, self.get_u32(5, 0) == 0, rect);
                tfx::to_bitmap_precision(w, h, ox, oy, highp, move |x, y| turb.eval(x, y))
            }
            Kind::PointDiffuse => {
                let ss = self.get_f32(2, 1.0);
                let light = tfx::Light::Point { position: self.get_v3(0, [0.0; 3]), surface_scale: ss };
                tfx::to_bitmap_precision(w, h, ox, oy, highp, tfx::diffuse(inp(), light, scale3(self.get_v3(3, [1.0; 3]), self.get_f32(1, 1.0)), ss))
            }
            Kind::DistantDiffuse => {
                let ss = self.get_f32(3, 1.0);
                let light = tfx::Light::Distant { direction: tfx::distant_direction(self.get_f32(0, 0.0), self.get_f32(1, 0.0)) };
                tfx::to_bitmap_precision(w, h, ox, oy, highp, tfx::diffuse(inp(), light, scale3(self.get_v3(4, [1.0; 3]), self.get_f32(2, 1.0)), ss))
            }
            Kind::SpotDiffuse => {
                let (pos, ss) = (self.get_v3(0, [0.0; 3]), self.get_f32(5, 1.0));
                let (cos_outer, cos_inner) = tfx::spot_cone_cosines(self.get_f32(3, 90.0));
                let light = tfx::Light::Spot {
                    position: pos,
                    surface_scale: ss,
                    axis: tfx::spot_axis(pos, self.get_v3(1, [0.0; 3])),
                    focus: self.get_f32(2, 1.0),
                    cos_outer,
                    cos_inner,
                };
                tfx::to_bitmap_precision(w, h, ox, oy, highp, tfx::diffuse(inp(), light, scale3(self.get_v3(6, [1.0; 3]), self.get_f32(4, 1.0)), ss))
            }
            Kind::PointSpecular => {
                let ss = self.get_f32(3, 1.0);
                let light = tfx::Light::Point { position: self.get_v3(0, [0.0; 3]), surface_scale: ss };
                tfx::to_bitmap_precision(w, h, ox, oy, highp, tfx::specular(inp(), light, scale3(self.get_v3(4, [1.0; 3]), self.get_f32(2, 1.0)), self.get_f32(1, 1.0), ss))
            }
            Kind::DistantSpecular => {
                let ss = self.get_f32(4, 1.0);
                let dir = tfx::distant_direction(self.get_f32(0, 0.0), self.get_f32(1, 0.0));
                let light = tfx::Light::Distant { direction: tfx::distant_halfway(dir) };
                tfx::to_bitmap_precision(w, h, ox, oy, highp, tfx::specular(inp(), light, scale3(self.get_v3(5, [1.0; 3]), self.get_f32(3, 1.0)), self.get_f32(2, 1.0), ss))
            }
            Kind::SpotSpecular => {
                let (pos, ss) = (self.get_v3(0, [0.0; 3]), self.get_f32(6, 1.0));
                let (cos_outer, cos_inner) = tfx::spot_cone_cosines(self.get_f32(3, 90.0));
                let light = tfx::Light::Spot {
                    position: pos,
                    surface_scale: ss,
                    axis: tfx::spot_axis(pos, self.get_v3(1, [0.0; 3])),
                    focus: self.get_f32(2, 1.0),
                    cos_outer,
                    cos_inner,
                };
                tfx::to_bitmap_precision(w, h, ox, oy, highp, tfx::specular(inp(), light, scale3(self.get_v3(7, [1.0; 3]), self.get_f32(5, 1.0)), self.get_f32(4, 1.0), ss))
            }
            // Only the `is_realized_transform` kinds reach here (realize_rect dispatches on that); a
            // non-matching kind bakes to a transparent output rather than panicking.
            _ => tfx::to_bitmap_precision(w, h, ox, oy, highp, |_, _| [0.0; 4]),
        }
    }

    /// The 3D LUT bound to a `LookupTable3D` effect, or an identity LUT if none is set.
    fn lut3d(&self) -> Rc<Lut3d> {
        self.0.borrow().lut.clone().unwrap_or_else(|| Rc::new(Lut3d::identity()))
    }

    /// Realise the output over `b` as premultiplied RGBA **f32** (no 8-bit quantisation), for a float
    /// (HDR) target where the extended range must survive. Colour effects use the plain `eval` path,
    /// which is exact; affine/filter/realized-transform kinds retain their declared precision.
    fn realize_rect_f32(&self, b: Rect, origin: [f32; 2]) -> Vec<[f32; 4]> {
        let prev = FLOAT_DRAWN.with(|c| c.replace(float_root(self)));
        let out = self.realize_rect_f32_inner(b, origin);
        FLOAT_DRAWN.with(|c| c.set(prev));
        out
    }

    fn realize_rect_f32_inner(&self, b: Rect, origin: [f32; 2]) -> Vec<[f32; 4]> {
        let (w, h) = (b.width() as usize, b.height() as usize);
        let kind = self.0.borrow().kind;
        if kind == Kind::Affine2D || kind == Kind::Straighten || kind.is_filter() || is_realized_transform(kind) {
            let bmp = self.realize_rect_at(b, origin);
            return (0..w * h).map(|i| bitmap_texel(&bmp, (i % w) as i32, (i / w) as i32)).collect();
        }
        let mut out = Vec::with_capacity(w * h);
        for py in 0..h as i32 {
            for px in 0..w as i32 {
                out.push(self.eval(b.l + px, b.t + py));
            }
        }
        out
    }

    fn affine_input(&self, highp: bool) -> (Bitmap, i32, i32, [f32; 6]) {
        use builtin::transform as tfx;
        let (mut input, mut ox, mut oy) = if highp { realize_input_hp(&self.0.borrow().inputs[0]) } else { realize_input(&self.0.borrow().inputs[0]) };
        let mut matrix = self.affine_matrix();
        if self.0.borrow().kind == Kind::Affine2D && self.get_u32(0, 1) == 5 {
            let border = tfx::Border::from_mode(self.get_u32(1, 0));
            // Native's HIGH_QUALITY_CUBIC affine transform prefilters each shrinking matrix column before cubic placement.
            let mut prefiltered = false;
            let factor = hq_prescale_factor(matrix);
            // The prescale is a Scale effect given this transform's own sharpness and border mode.
            let sharpness = self.get_f32(3, 0.0).clamp(0.0, 1.0);
            for axis in 0..2 {
                let j = axis * 2;
                let scale = factor[axis];
                if scale < 1.0 {
                    let mut out = [ox, oy, ox + input.width as i32, oy + input.height as i32];
                    (out[axis], out[axis + 2]) = hq_prescale_span(out[axis], out[axis + 2], scale, border == tfx::Border::Hard);
                    let view = tfx::Input::new(&input, ox, oy);
                    let view = if prefiltered { view.in_atlas() } else { view };
                    input = tfx::hq_downscale_axis(&view, axis, scale, 0.0, sharpness, border, out, self.out_precision());
                    prefiltered = true;
                    ox = out[0]; oy = out[1];
                    // d2d1 forms the remaining transform as diag(1/sx, 1/sy)·M (a Matrix4x4F product), so the row
                    // multiplies by the factor's reciprocal (VM: a 45° HQ draw's vertices and inverse, r27 32499).
                    let k = 1.0 / scale;
                    matrix[j] *= k;
                    matrix[j + 1] *= k;
                }
            }
        }
        (input, ox, oy, matrix)
    }

    /// Affine resample: realize the input and filter through WARP's triangle attributes,
    /// with transparent SOFT borders or clamped HARD borders.
    fn realize_affine_axis_hq(&self, out: Rect) -> Option<Bitmap> {
        let m = self.affine_matrix();
        if self.0.borrow().kind != Kind::Affine2D || self.get_u32(0, 1) != 5
            || m[1] != 0.0 || m[2] != 0.0 || m[0] <= 0.0 || m[3] <= 0.0 || (m[0] >= 1.0 && m[3] >= 1.0) { return None; }
        // Native axis-aligned affine HQ collapses to the same Scale passes (DDI edge-affine-mixed).
        let (bitmap, ox, oy) = if self.hp().is_some() { realize_input_hp(&self.0.borrow().inputs[0]) } else { realize_input(&self.0.borrow().inputs[0]) };
        let input = builtin::transform::Input::new(&bitmap, ox, oy);
        Some(builtin::transform::hq_downscale(&input, m, 0.0, builtin::transform::Border::from_mode(self.get_u32(1, 0)), [out.l, out.t, out.r, out.b], self.out_precision()))
    }

    /// An axis-aligned, positive-scale 2D affine transform samples as the Scale effect does, in every
    /// interpolation mode (the mixed/one-axis sampler sweep: 40 native captures), extending the HQ
    /// collapse above.
    fn affine_is_axis_scale(&self, m: [f32; 6]) -> bool {
        self.0.borrow().kind == Kind::Affine2D && m == self.affine_matrix() && m[1] == 0.0 && m[2] == 0.0 && m[0] > 0.0 && m[3] > 0.0
    }

    fn realize_affine(&self, out: Rect, data: &mut [u8], origin: [f32; 2]) {
        if let Some(bitmap) = self.realize_affine_axis_hq(out) { data.copy_from_slice(&bitmap.premul_bgra); return; }
        for (i, c) in self.affine_pixels(out, origin, false).into_iter().enumerate() {
            store(data, i * 4, c);
        }
    }

    /// [`realize_affine`] above 8bpc: retain float input, filtering, and output precision.
    fn realize_affine_hp(&self, out: Rect, origin: [f32; 2]) -> Vec<[f32; 4]> {
        if let Some(bitmap) = self.realize_affine_axis_hq(out) {
            return (0..out.width() * out.height()).map(|i| bitmap.texel_rgba((i % out.width()) as i32, (i / out.width()) as i32)).collect();
        }
        self.affine_pixels(out, origin, true)
    }

    /// A HARD HIGH_QUALITY_CUBIC 2D affine transform's matrix-transform output: the transformed input rect's bounding box,
    /// the pixels whose centres it holds (d2d1's top-left rounding). `None` for any other effect or an unbounded input.
    /// ponytail: the raw matrix and input rect; the prescale's own rect and matrix would only move a pixel-centre tie.
    pub(crate) fn hq_hard_rect(&self) -> Option<Rect> {
        if self.0.borrow().kind != Kind::Affine2D || self.get_u32(0, 1) != 5 || self.get_u32(1, 0) != 1 {
            return None;
        }
        let ib = input_bounds(&self.0.borrow().inputs[0]);
        if ib.l == i32::MIN || ib.t == i32::MIN || ib.r == i32::MAX || ib.b == i32::MAX {
            return None;
        }
        let m = self.affine_matrix();
        let c = [(ib.l, ib.t), (ib.r, ib.t), (ib.l, ib.b), (ib.r, ib.b)].map(|(x, y)| {
            let (x, y) = (x as f32, y as f32);
            [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]
        });
        let lo = |k: usize| c.iter().map(|p| p[k]).fold(f32::INFINITY, f32::min);
        let hi = |k: usize| c.iter().map(|p| p[k]).fold(f32::NEG_INFINITY, f32::max);
        let e = |v: f32| (v - 0.5).ceil() as i32;
        Some(Rect { l: e(lo(0)), t: e(lo(1)), r: e(hi(0)), b: e(hi(1)) })
    }

    /// Where an unbatched slow-path draw's intermediate holds this HIGH_QUALITY_CUBIC transform's output, in effect pixels.
    /// Such a graph's intermediates get their own textures, each placed as d2d1 sizes a lone intermediate: the content
    /// (the request within the output rect) a texel in on a side where the request reaches past it, flush otherwise; its prescaled input sits at its texture's origin, the clamp addressing standing in for the left and
    /// top border cells. The texture origin then sets the triangle planes the cubic samples through (VM: r27 32499's
    /// DEST_ATOP draw, its HQ pass at slot (−8, +3) and the input unpadded; JIT trace of the cubic kernel, 94.53 of 255
    /// against SOURCE_OVER's 94.49 at the same pixel).
    /// ponytail: a small request (the one-texel margin case) and a HARD border only.
    fn hq_slot_origin(&self) -> Option<[i32; 2]> {
        let (id, r) = HQ_SLOW_REQ.with(|c| c.get())?;
        if id != Rc::as_ptr(&self.0) as usize {
            return None;
        }
        self.hq_slot(r)
    }

    /// [`Self::hq_slot_origin`] for request `r`.
    fn hq_slot(&self, r: Rect) -> Option<[i32; 2]> {
        let o = self.hq_hard_rect()?;
        Some([r.l.max(o.l - 1), r.t.max(o.t - 1)])
    }

    /// The 2D affine transform's output over `out`, row by row, at any precision (`highp` keeps the float input and
    /// output): the transformed input quad, transparent outside it.
    fn affine_pixels(&self, out: Rect, origin: [f32; 2], highp: bool) -> Vec<[f32; 4]> {
        let (inp, iox, ioy, m) = self.affine_input(highp);
        let det = m[0] * m[3] - m[1] * m[2];
        let (linear, soft) = self.affine_sampling();
        let mode = self.get_u32(if self.0.borrow().kind == Kind::Straighten { 2 } else { 0 }, 1);
        let cubic_input = builtin::transform::Input::new(&inp, iox, ioy);
        let cubic_input = if highp { cubic_input.with_float_output() } else { cubic_input };
        // A direct bitmap input rounds its Q8 address at a non-power-of-two size, as `realize_transform`'s does; an
        // intermediate sits in a power-of-two atlas and floors (VM probes: an identity LINEAR transform of a 13x11
        // bitmap read each texel 1/256 toward its up-left neighbour when floored; 16x16 exact either way).
        let is_bitmap = matches!(&self.0.borrow().inputs[0], Input::Bitmap(_));
        let cubic_input = if is_bitmap && (!inp.width.is_power_of_two() || !inp.height.is_power_of_two()) { cubic_input.with_rne_address() } else { cubic_input };
        let slot = self.hq_slot_origin();
        let cubic_input = if m != self.affine_matrix() { cubic_input.in_atlas().with_texture_offset(if slot.is_some() { [0; 2] } else { [-2; 2] }) } else { cubic_input };
        let origin = slot.map_or(origin, |t| [origin[0] - t[0] as f32, origin[1] - t[1] as f32]);
        // A lighting output is an intermediate WARP renders into an atlas texture (power of two, at least 512), and the
        // lighting stage's post-scale normalizes its texcoords by that texture: a pixel centre on a texel edge lands on
        // it exactly (sc78, sc82: NEAREST took the lower texel there, CUBIC was 1 off).
        // ponytail: NEAREST/LINEAR/CUBIC only; ANISOTROPIC's mip path doesn't take an atlas input yet.
        let atlas = mode <= 2 && matches!(&self.0.borrow().inputs[0], Input::Effect(e) if is_lighting(e.0.borrow().kind));
        let cubic_input = if atlas { cubic_input.in_atlas() } else { cubic_input };
        let axis_scale = self.affine_is_axis_scale(m);
        let cubic = builtin::transform::affine_at(&cubic_input, m, builtin::transform::Interp::from_mode(mode), if soft { builtin::transform::Border::Soft } else { builtin::transform::Border::Hard }, origin, axis_scale);
        let w = out.width() as i32;
        (0..out.height() as i32 * w)
            .map(|i| {
                let (sx, sy) = (out.l as f32 + (i % w) as f32 + 0.5, out.t as f32 + (i / w) as f32 + 0.5);
                if det == 0.0 {
                    return [0.0; 4]; // singular: inverse maps every pixel far away → transparent (R1)
                }
                // input DIP = inverse(M) · scenePos; then to the input bitmap's local texel space
                let ix = (m[3] * (sx - m[4]) - m[2] * (sy - m[5])) / det - iox as f32;
                let iy = (-m[1] * (sx - m[4]) + m[0] * (sy - m[5])) / det - ioy as f32;
                // The output is the transformed input quad, not its axis-aligned bbox. HARD border
                // clamps the filter's edge taps, but a pixel whose centre maps outside the input is
                // still off the quad → transparent (fixes `fx-transform-10`). SOFT already zero-fills
                // outside taps, which gives the soft-edge AA, so it needs no gate.
                if !soft && (ix < 0.0 || iy < 0.0 || ix >= inp.width as f32 || iy >= inp.height as f32) {
                    [0.0; 4]
                } else if matches!(mode, 0..=5) {
                    cubic(sx, sy)
                } else {
                    sample(&inp, ix, iy, linear, soft)
                }
            })
            .collect()
    }

    // --- 2C filter family (§ filter spec) -----------------------------------------------------

    /// Emboss as Direct2D builds it: a ColorMatrix whose output alpha is the input's luminance
    /// (0.299 R + 0.587 G + 0.114 B, colour zero) lit by a DistantDiffuse at elevation 50°, azimuth = Direction and
    /// surface scale = Height, at the defaults otherwise (diffuse constant 1, white, unit kernel).
    fn emboss_graph(&self) -> Effect {
        let d = self.0.borrow();
        let node = |kind| {
            let e = Effect::new_in(kind, d.dpi);
            (e.0.borrow_mut().render_precision, e.0.borrow_mut().render_fallback) = (d.render_precision, d.render_fallback);
            e
        };
        let cm = node(Kind::ColorMatrix);
        cm.set_input(0, d.inputs[0].clone());
        let mut m = [0.0f32; 20];
        (m[3], m[7], m[11]) = (0.299, 0.587, 0.114);
        cm.0.borrow_mut().props.insert(0, Prop::M54(m));
        let lit = node(Kind::DistantDiffuse);
        lit.set_input(0, Input::Effect(cm));
        let mut props = lit.0.borrow_mut();
        props.props.insert(0, Prop::F32(self.get_f32(1, 0.0).clamp(0.0, 360.0)));
        props.props.insert(1, Prop::F32(50.0));
        props.props.insert(3, Prop::F32(self.get_f32(0, 1.0).clamp(0.0, 10.0)));
        drop(props);
        lit
    }

    /// A lighting effect lights at one kernel unit per texel. Its kernel is k = KernelUnitLength · dpi/96 device
    /// pixels per axis; off a unit kernel Direct2D shrinks the input by 1/k with a HARD 2D affine, lights that, and
    /// scales the result back by k, both resamples at the effect's ScaleMode. `None` at a unit kernel.
    /// ponytail: HIGH_QUALITY_CUBIC scales through Direct2D's Scale effect, whose rect we don't model (sc78 m5:
    /// 22 px wide vs 12), and NEAREST ties at exact texel edges go low here, high natively (sc78 m0, max 30).
    fn lighting_stage(&self) -> Option<Effect> {
        use Kind::*;
        let d = self.0.borrow();
        // KernelUnitLength's index; ScaleMode follows it.
        let kul = match d.kind {
            PointDiffuse => 4,
            DistantDiffuse | PointSpecular => 5,
            DistantSpecular => 6,
            SpotDiffuse => 7,
            SpotSpecular => 8,
            _ => return None,
        };
        let s = DIP_SCALE.with(|s| s.get());
        let unit = self.get_v2(kul, [1.0, 1.0]);
        let k = [s * unit[0], s * unit[1]];
        // ponytail: an input past the realize cap (a Flood) lights unscaled; its interior is flat either way. Lighting
        // a near-infinite input at scale needs the 2D affine to realize its input over its need, not whole.
        if k.iter().all(|k| (1.0 - 1.0 / k).abs() <= 1e-5) || beyond_realize_cap(input_bounds(&d.inputs[0])) {
            return None;
        }
        let node = |kind| {
            let e = Effect::new_in(kind, d.dpi);
            (e.0.borrow_mut().render_precision, e.0.borrow_mut().render_fallback) = (d.render_precision, d.render_fallback);
            e
        };
        let mode = self.get_u32(kul + 1, 1);
        let scale = |input: Input, x: f32, y: f32| {
            let e = node(Affine2D);
            e.set_input(0, input);
            let mut p = e.0.borrow_mut();
            p.props.insert(0, Prop::U32(mode));
            p.props.insert(1, Prop::U32(1));
            p.props.insert(2, Prop::M32([x, 0.0, 0.0, y, 0.0, 0.0]));
            drop(p);
            e
        };
        let pre = scale(d.inputs[0].clone(), 1.0 / k[0], 1.0 / k[1]);
        // The same lighting at a unit kernel: (1/s) · s is within the tolerance, so it takes no stage of its own.
        let lit = node(d.kind);
        lit.0.borrow_mut().props = d.props.clone();
        lit.0.borrow_mut().props.insert(kul, Prop::V2([1.0 / s, 1.0 / s]));
        lit.0.borrow_mut().precision = d.precision;
        lit.set_input(0, Input::Effect(pre));
        Some(scale(Input::Effect(lit), k[0], k[1]))
    }

    /// Per-side device-pixel reach of a reach filter — how far an output pixel reads into input 0.
    /// Used to clip an INFINITE input (Border/Tile extend) to the finite sub-rect an output region
    /// needs, so the filter never realizes the infinite rect. Generous (≥ the true kernel radius);
    /// over-covering an infinite input is exact (taps past the reach are cropped with the output).
    fn filter_reach(&self) -> i32 {
        use Kind::*;
        match self.0.borrow().kind {
            GaussianBlur | Shadow | DirectionalBlur => ((self.sigma() * 3.0).ceil() as i32).saturating_add(8),
            ConvolveMatrix => (self.get_u32(2, 3).max(self.get_u32(3, 3)) as i32).saturating_add(2),
            Morphology => (self.get_u32(1, 1).max(self.get_u32(2, 1)) as i32).saturating_add(2),
            Sharpen | EdgeDetection | Emboss => 4,
            _ => 2,
        }
    }

    /// Realize filter input 0. An INFINITE input (its output bounds unbounded) with a known output
    /// `clip` is realized only over `[clip grown by filter_reach]`, never its infinite rect (the crash
    /// native avoids by deferring the extend). A finite input realizes whole, as before.
    fn realize_filter_input(&self, inp: &Input, clip: Option<Rect>, highp: bool) -> (Bitmap, i32, i32) {
        let ib = input_bounds(inp);
        let infinite = ib.l == i32::MIN || ib.t == i32::MIN || ib.r == i32::MAX || ib.b == i32::MAX;
        match (clip, infinite) {
            (Some(b), true) => {
                let k = self.filter_reach();
                realize_input_at(inp, Rect { l: b.l.saturating_sub(k), t: b.t.saturating_sub(k), r: b.r.saturating_add(k), b: b.b.saturating_add(k) }, highp)
            }
            _ if highp => realize_input_hp(inp),
            _ => realize_input(inp),
        }
    }

    /// Realise the whole filter output (the 2C per-pixel passes over the realised input) and its
    /// device-pixel origin. `realize_filter_input` gives the input bitmap + its own origin; the filter
    /// returns its output's origin relative to the input, which adds to give the device origin. `clip`
    /// is the output rect being realized (bounds the input when it is infinite); `None` realizes whole.
    fn realize_filter(&self, clip: Option<Rect>) -> (Bitmap, i32, i32) {
        self.realize_filter_at(clip, None)
    }

    fn realize_filter_at(&self, clip: Option<Rect>, viewport: Option<Rect>) -> (Bitmap, i32, i32) {
        if self.0.borrow().kind == Kind::Emboss {
            let highp = self.hp().is_some();
            return self.realize_filter_input(&Input::Effect(self.emboss_graph()), clip, highp);
        }
        if let Some(p) = self.hp() {
            return self.realize_filter_hp(p, clip);
        }
        use builtin::filters as f;
        // Only this Shadow's consumer asked for a stored colour; its own inputs did not.
        let stored = SHADOW_STORED.with(|c| c.replace(false));
        let d = self.0.borrow();
        // An image brush's border cells are for this (root) blur's Y pass: take them before the inputs
        // realize, so a blur inside the input graph never sees them.
        let cells = if d.kind == Kind::GaussianBlur { BORDER_CELLS.with(|c| c.take()) } else { None };
        if d.kind == Kind::GaussianBlur {
            if let (Some(b), Input::Effect(border)) = (clip, &d.inputs[0]) {
                if border.0.borrow().kind == Kind::Border {
                    let (inp, ox, oy) = realize_input(&border.0.borrow().inputs[0]);
                    let out = f::gaussian_blur_extended(&inp, self.sigma(), self.get_u32(1, 1), [border.get_u32(0, 0),border.get_u32(1, 0)], [b.l-ox,b.t-oy,b.r-ox,b.b-oy], false);
                    return (out.bitmap, ox+out.ox, oy+out.oy);
                }
            }
        }
        let vp = viewport.or_else(|| DRAW_VIEWPORT.with(|v| v.get()));
        // A blur's input is realized over what the blur reads of the draw's viewport (d2d1 maps each node's need back),
        // not over the viewport itself: a Shadow under a DirectionalBlur covers the blur's reach past the target (r19
        // 24502). The input's passes then start at that need.
        let ib = input_bounds(&d.inputs[0]);
        let nested = vp.filter(|_| d.kind == Kind::DirectionalBlur).and_then(|b| {
            f::directional_need(self.sigma(), self.get_f32(1, 0.0), self.get_u32(2, 1), self.get_u32(3, 0) == 1, [b.l, b.t, b.r, b.b], [ib.l, ib.t, ib.r, ib.b])
        });
        let prev = nested.map(|n| (DRAW_VIEWPORT.with(|v| v.replace(Some(Rect { l: n[0], t: n[1], r: n[2], b: n[3] }))), DRAW_ORIGIN.with(|o| o.replace(None))));
        // A HARD DirectionalBlur reads its input through a mirror border, whose cells colour a Shadow's stored blur.
        let border_read = d.kind == Kind::DirectionalBlur && self.get_u32(3, 0) == 1
            && matches!(&d.inputs[0], Input::Effect(e) if e.0.borrow().kind == Kind::Shadow);
        let prev_stored = SHADOW_STORED.with(|c| c.replace(border_read));
        let (inp, iox, ioy) = self.realize_filter_input(&d.inputs[0], clip, false);
        SHADOW_STORED.with(|c| c.set(prev_stored));
        if let Some((v, o)) = prev {
            DRAW_VIEWPORT.with(|c| c.set(v));
            DRAW_ORIGIN.with(|c| c.set(o));
        }
        // The draw's visible region in input pixels: native's blur passes cover only that.
        let req = vp.map(|b| {
            let o = DRAW_ORIGIN.with(|o| o.get()).unwrap_or([b.l, b.t]);
            f::Request { rect: [b.l - iox, b.t - ioy, b.r - iox, b.b - ioy], origin: [o[0] - iox, o[1] - ioy], input: [iox, ioy] }
        });
        let bool_at = |i| matches!(d.props.get(&i), Some(Prop::Bool(true)));
        let out = match d.kind {
            Kind::GaussianBlur => {
                let cells = cells.map(|c| f::BorderCells {
                    origin: [c.origin[0] - iox, c.origin[1] - ioy],
                    size: c.size,
                    cells: c.cells.into_iter().map(|(scr, i)| (scr, [i[0] - iox as f32, i[1] - ioy as f32, i[2] - iox as f32, i[3] - ioy as f32])).collect(),
                });
                f::gaussian_blur_in(&inp, self.sigma(), self.get_u32(1, 1), self.get_u32(2, 0) == 1, req, None, cells.as_ref())
            }
            Kind::DirectionalBlur => {
                f::directional_blur_with_options(&inp, self.sigma(), self.get_f32(1, 0.0), self.get_u32(2, 1), self.get_u32(3, 0) == 1, viewport.map(|b| [b.l-iox,b.t-ioy,b.r-iox,b.b-ioy]), [iox, ioy])
            }
            Kind::Shadow => {
                f::shadow_in(&inp, self.sigma(), self.get_v4(1, [0.0, 0.0, 0.0, 1.0]), self.get_u32(2, 1), req, stored)
            }
            Kind::Morphology => f::morphology(&inp, self.get_u32(0, 0) == 1, self.get_u32(1, 1), self.get_u32(2, 1)),
            Kind::Sharpen => {
                f::sharpen(&inp, self.get_f32(0, 0.0).clamp(0.0, 10.0), self.get_f32(1, 0.0).clamp(0.0, 10.0))
            }
            // ponytail: BLUR_RADIUS (pre-blur) is ignored — no scene exercises it; add a gaussian
            // pre-pass when one does.
            Kind::EdgeDetection => f::edge_detection(
                &inp,
                self.get_f32(0, 0.5).clamp(0.0, 1.0),
                self.get_u32(2, 0) == 1,
                bool_at(3),
                self.get_u32(4, 1) == 2,
            ),
            Kind::Emboss => {
                f::emboss(&inp, self.get_f32(0, 1.0).clamp(0.0, 10.0), self.get_f32(1, 0.0).clamp(0.0, 360.0))
            }
            Kind::Posterize => f::posterize(&inp, self.get_u32(0, 4), self.get_u32(1, 4), self.get_u32(2, 4)),
            Kind::HighlightsShadows => {
                // Props: 0 Highlights, 1 Shadows, 2 Clarity, 3 InputGamma (enum: LINEAR 0, SRGB 1 =
                // default), 4 MaskBlurRadius (default 1.25). LINEAR runs ps6 without the gamma pows.
                let (cx, cy) = builtin::color::highlights_shadows_gains(self.get_f32(0, 0.0), self.get_f32(1, 0.0));
                let cz = builtin::color::highlights_shadows_clarity(self.get_f32(2, 0.0));
                let linear = self.get_u32(3, 1) == 0;
                f::highlights_shadows(&inp, cx, cy, cz, self.get_f32(4, 1.25), linear)
            }
            Kind::ConvolveMatrix => {
                let kernel = self.get_floats(4, &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
                f::convolve(
                    &inp,
                    self.get_u32(2, 3),
                    self.get_u32(3, 3),
                    &kernel,
                    self.get_f32(5, 1.0),
                    self.get_f32(6, 0.0),
                    self.get_v2(7, [0.0, 0.0]),
                    bool_at(8),
                    bool_at(10),
                    self.get_u32(9, 0) == 1,
                )
            }
            Kind::DisplacementMap => {
                // An unbounded map is read only where the input lies.
                let at = Rect { l: iox, t: ioy, r: iox + inp.width as i32, b: ioy + inp.height as i32 };
                let prev = DRAW_VIEWPORT.with(|v| v.replace(Some(at)));
                let (dmap, _, _) = realize_input(&d.inputs[1]);
                DRAW_VIEWPORT.with(|v| v.set(prev));
                f::displacement_map(&inp, &dmap, self.get_f32(0, 0.0), self.get_u32(1, 3), self.get_u32(2, 3), disp_round(&d.inputs[0], &inp))
            }
            // Only `is_filter` kinds reach here (realize_rect dispatches on that); pass the input
            // through unchanged for any other kind rather than panicking.
            _ => return (inp, iox, ioy),
        };
        (out.bitmap, iox + out.ox, ioy + out.oy)
    }

    /// A HARD GaussianBlur's realized input when its input blur's last pass runs in its border (the plan's
    /// `fold`): that pass drawn over the border's quads at the border's place ([`builtin::filters_hp::border_fold`]).
    fn fold_apron(&self, d: &Data, inp: &Bitmap, org: [i32; 2], p: u32) -> Option<builtin::filters_hp::FOut> {
        use builtin::filters_hp as f;
        let Input::Effect(s) = &d.inputs[0] else { return None };
        let bounds = [org[0], org[1], org[0] + inp.width as i32, org[1] + inp.height as i32];
        let last = if d.place.fold { s.blur_last_pass(p)? } else { s.drawn_last_pass(d.place.src?, bounds, p)? };
        let r = f::hard_rects(bounds, f::blur_need(bounds, d.place.need), self.sigma(), self.get_u32(1, 1), false)?;
        Some(f::border_fold(&last, r.input?, d.place.input, r.pre.is_some(), p))
    }

    /// What a blur's border draws from this effect when it is drawn and the border samples it: a ConvolveMatrix on a
    /// float context, from its own intermediate (`src`: atlas place, rect) unclamped (its clamp links into the
    /// border), or, for a kernel that folds into the border, run there over its padded copy. `bounds` is the blur's
    /// input. Anything else keeps the exact mirror.
    fn drawn_last_pass(&self, (at, rect): ([i32; 2], [i32; 4]), bounds: [i32; 4], p: u32) -> Option<builtin::filters_hp::LastPass> {
        use builtin::filters_hp as f;
        let d = self.0.borrow();
        let bool_at = |i| matches!(d.props.get(&i), Some(Prop::Bool(true)));
        if d.kind != Kind::ConvolveMatrix || p != 5 || bool_at(8) {
            return None;
        }
        let (ksx, ksy) = (self.get_u32(2, 3), self.get_u32(3, 3));
        let kernel = self.get_floats(4, &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let (divisor, bias, offset, clamp) = (self.get_f32(5, 1.0), self.get_f32(6, 0.0), self.get_v2(7, [0.0, 0.0]), bool_at(10));
        let taps = f::native_taps(ksx as i32, ksy as i32, &kernel, divisor, bias, offset, clamp)?;
        let (inp, ox, oy) = self.realize_filter_input(&d.inputs[0], None, true);
        let src = if ksx == 3 && ksy == 3 {
            let mut c = f::convolve(&inp, ksx, ksy, &kernel, divisor, bias, offset, false, false, self.get_u32(9, 0) == 1);
            (c.ox, c.oy) = (c.ox + ox, c.oy + oy);
            f::regrid(&c, rect)
        } else {
            f::regrid(&f::FOut::of_at(&inp, [ox, oy]), rect)
        };
        let taps = (ksx != 3 || ksy != 3).then_some(taps);
        Some(f::LastPass::drawn(src, at, bounds, taps, clamp))
    }

    /// What this HARD GaussianBlur's last pass reads, for a blur that runs that pass in its own border.
    fn blur_last_pass(&self, p: u32) -> Option<builtin::filters_hp::LastPass> {
        let d = self.0.borrow();
        if d.kind != Kind::GaussianBlur || self.get_u32(2, 0) != 1 {
            return None;
        }
        let (inp, ox, oy) = self.realize_filter_input(&d.inputs[0], None, true);
        let apron = self.fold_apron(&d, &inp, [ox, oy], p);
        builtin::filters_hp::gaussian_blur_ex(&inp, [ox, oy], self.sigma(), self.get_u32(1, 1), true, matches!(d.inputs[0], Input::Bitmap(_) | Input::List(_)), p, &d.place, apron.as_ref()).1
    }

    /// [`realize_filter`] above 8bpc ([`builtin::filters_hp`]): the input in its own format, f32
    /// arithmetic, the result stored at precision `p`. HighlightsShadows keeps its 8bpc kernel.
    fn realize_filter_hp(&self, p: u32, clip: Option<Rect>) -> (Bitmap, i32, i32) {
        use builtin::filters_hp as f;
        let d = self.0.borrow();
        let (inp, iox, ioy) = self.realize_filter_input(&d.inputs[0], clip, true);
        let bool_at = |i| matches!(d.props.get(&i), Some(Prop::Bool(true)));
        let out = match d.kind {
            Kind::GaussianBlur => {
                let apron = self.fold_apron(&d, &inp, [iox, ioy], p);
                f::gaussian_blur_ex(&inp, [iox, ioy], self.sigma(), self.get_u32(1, 1), self.get_u32(2, 0) == 1, matches!(d.inputs[0], Input::Bitmap(_) | Input::List(_)), p, &d.place, apron.as_ref()).0
            }
            Kind::DirectionalBlur => {
                f::directional_blur(&inp, self.sigma(), self.get_f32(1, 0.0), self.get_u32(3, 0) == 1)
            }
            Kind::Shadow => {
                f::shadow(&inp, self.sigma(), self.get_v4(1, [0.0, 0.0, 0.0, 1.0]), self.get_u32(2, 1), p)
            }
            Kind::Morphology => f::morphology(&inp, self.get_u32(0, 0) == 1, self.get_u32(1, 1), self.get_u32(2, 1)),
            Kind::Sharpen => f::sharpen(&inp, self.get_f32(0, 0.0).clamp(0.0, 10.0), self.get_f32(1, 0.0).clamp(0.0, 10.0)),
            Kind::EdgeDetection => f::edge_detection(
                &inp,
                self.get_f32(0, 0.5).clamp(0.0, 1.0),
                self.get_u32(2, 0) == 1,
                bool_at(3),
                self.get_u32(4, 1) == 2,
            ),
            Kind::Emboss => f::emboss(&inp, self.get_f32(0, 1.0).clamp(0.0, 10.0), self.get_f32(1, 0.0).clamp(0.0, 360.0)),
            Kind::Posterize => f::posterize(&inp, self.get_u32(0, 4), self.get_u32(1, 4), self.get_u32(2, 4)),
            Kind::ConvolveMatrix => {
                let kernel = self.get_floats(4, &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
                f::convolve(
                    &inp,
                    self.get_u32(2, 3),
                    self.get_u32(3, 3),
                    &kernel,
                    self.get_f32(5, 1.0),
                    self.get_f32(6, 0.0),
                    self.get_v2(7, [0.0, 0.0]),
                    bool_at(8),
                    bool_at(10),
                    self.get_u32(9, 0) == 1,
                )
            }
            Kind::DisplacementMap => {
                let (dmap, _, _) = realize_input_hp(&d.inputs[1]);
                f::displacement_map(&inp, &dmap, self.get_f32(0, 0.0), self.get_u32(1, 3), self.get_u32(2, 3))
            }
            Kind::HighlightsShadows => {
                drop(d);
                let (bmp, ox, oy) = self.realize_filter_8bpc_hs();
                return (bmp, ox, oy);
            }
            _ => return (inp, iox, ioy),
        };
        let bmp = float_bitmap(out.w as u32, out.h as u32, out.px.into_iter().map(|c| quantize_to(p, c)));
        (bmp, iox + out.ox, ioy + out.oy)
    }

    /// HighlightsShadows above 8bpc: its 8bpc kernel over the 8bpc input (no float form yet).
    /// ponytail: the local-contrast mask graph is byte-exact at 8bpc only; a float form when a scene needs it.
    fn realize_filter_8bpc_hs(&self) -> (Bitmap, i32, i32) {
        let d = self.0.borrow();
        let (inp, iox, ioy) = realize_input(&d.inputs[0]);
        let (cx, cy) = builtin::color::highlights_shadows_gains(self.get_f32(0, 0.0), self.get_f32(1, 0.0));
        let cz = builtin::color::highlights_shadows_clarity(self.get_f32(2, 0.0));
        let out = builtin::filters::highlights_shadows(&inp, cx, cy, cz, self.get_f32(4, 1.25), self.get_u32(3, 1) == 0);
        (out.bitmap, iox + out.ox, ioy + out.oy)
    }

    /// Fill rect `b` from the realised filter output, transparent where `b` extends past it.
    fn realize_filter_rect(&self, b: Rect, data: &mut [u8], viewport: Option<Rect>) {
        let (full, fox, foy) = self.realize_filter_at(Some(b), viewport);
        let w = b.width() as usize;
        for py in 0..b.height() as i32 {
            for px in 0..w as i32 {
                let (lx, ly) = (b.l + px - fox, b.t + py - foy);
                let o = (py as usize * w + px as usize) * 4;
                if lx >= 0 && ly >= 0 && lx < full.width as i32 && ly < full.height as i32 {
                    let i = (ly as usize * full.width as usize + lx as usize) * 4;
                    data[o..o + 4].copy_from_slice(&full.premul_bgra[i..i + 4]);
                }
            }
        }
    }
}

// --- input helpers -----------------------------------------------------------------------------

fn input_bounds(inp: &Input) -> Rect {
    match inp {
        Input::Bitmap(b) => Rect { l: 0, t: 0, r: b.width as i32, b: b.height as i32 },
        Input::Effect(e) => e.bounds(),
        Input::List(l) => {
            let b = l.baked();
            Rect { l: b.1, t: b.2, r: b.1 + b.0.width as i32, b: b.2 + b.0.height as i32 }
        }
    }
}

fn eval_input(inp: &Input, x: i32, y: i32) -> [f32; 4] {
    match inp {
        Input::Bitmap(b) => bitmap_texel(b, x, y),
        Input::Effect(e) => e.eval(x, y),
        Input::List(l) => {
            let b = l.baked();
            bitmap_texel(&b.0, x - b.1, y - b.2)
        }
    }
}

/// Bilinear sample of `inp` (bounds `b`) at a fractional LOCAL texel position `(u, v)`, edge-clamped
/// — the GPU sampler's half-texel-centred bilinear (horizontal then vertical lerp). Shared by the
/// YCbCr chroma upsample and the BitmapSource scale.
fn sample_bilinear(inp: &Input, b: Rect, u: f32, v: f32) -> [f32; 4] {
    let (iw, ih) = (b.width() as i32, b.height() as i32);
    let u = u.clamp(0.0, (iw - 1).max(0) as f32);
    let v = v.clamp(0.0, (ih - 1).max(0) as f32);
    let (tu0, tv0) = (u.floor() as i32, v.floor() as i32);
    let (tu1, tv1) = ((tu0 + 1).min(iw - 1), (tv0 + 1).min(ih - 1));
    let (fu, fv) = (u - tu0 as f32, v - tv0 as f32);
    let s = |cx: i32, cy: i32| eval_input(inp, b.l + cx, b.t + cy);
    let lerp = |a: [f32; 4], c: [f32; 4], t: f32| core::array::from_fn(|i| a[i] * (1.0 - t) + c[i] * t);
    lerp(lerp(s(tu0, tv0), s(tu1, tv0), fu), lerp(s(tu0, tv1), s(tu1, tv1), fu), fv)
}

/// Fixed-point bilinear for the BitmapSource scale: native's bitmap sampler works on 8-bit texels
/// with integer weights `w = round(frac·128)` and a truncating `(a·(128−w) + b·w) >> 7` per axis
/// (horizontal then vertical), not a float-rounded blend — so a 0.25 tap of 255 is 63, not 64.
/// Exact at integer positions, so it leaves the orientation (1:1) cases unchanged.
fn sample_bilinear_fixed8(inp: &Input, b: Rect, u: f32, v: f32) -> [f32; 4] {
    let (iw, ih) = (b.width() as i32, b.height() as i32);
    let u = u.clamp(0.0, (iw - 1).max(0) as f32);
    let v = v.clamp(0.0, (ih - 1).max(0) as f32);
    let (tu0, tv0) = (u.floor() as i32, v.floor() as i32);
    let (tu1, tv1) = ((tu0 + 1).min(iw - 1), (tv0 + 1).min(ih - 1));
    let (wu, wv) = (((u - tu0 as f32) * 128.0).round() as i32, ((v - tv0 as f32) * 128.0).round() as i32);
    let t = |cx: i32, cy: i32| eval_input(inp, b.l + cx, b.t + cy).map(|c| (c * 255.0).round() as i32);
    let (c00, c10, c01, c11) = (t(tu0, tv0), t(tu1, tv0), t(tu0, tv1), t(tu1, tv1));
    core::array::from_fn(|i| {
        let r0 = (c00[i] * (128 - wu) + c10[i] * wu) >> 7;
        let r1 = (c01[i] * (128 - wu) + c11[i] * wu) >> 7;
        ((r0 * (128 - wv) + r1 * wv) >> 7) as f32 / 255.0
    })
}

/// BitmapSource scaled source dimensions: `round(size · Scale)`.
fn bitmapsource_scaled(b: Rect, scale: [f32; 2]) -> (i32, i32) {
    ((b.width() as f32 * scale[0]).round() as i32, (b.height() as f32 * scale[1]).round() as i32)
}

/// BitmapSource Orientation: map an output pixel `(ox, oy)` back to the scaled-source position,
/// given the scaled dims `(sw, sh)`. The `D2D1_BITMAPSOURCE_ORIENTATION` enum values are NOT
/// sequential — verified against native (unique-pixel captures): 1 default, 2 flip-h, 3 rotate180,
/// 4 rotate180+flip-h (= flip-v), 5 rotate270+flip-h (= anti-transpose), 6 rotate90cw, 7 rotate90+
/// flip-h (= transpose), 8 rotate270cw. Values 5–8 swap width and height.
fn inverse_orient(o: u32, ox: i32, oy: i32, sw: i32, sh: i32) -> (i32, i32) {
    match o {
        2 => (sw - 1 - ox, oy),          // flip horizontal
        3 => (sw - 1 - ox, sh - 1 - oy), // rotate 180
        4 => (ox, sh - 1 - oy),          // rotate 180 + flip-h = flip vertical
        5 => (sw - 1 - oy, sh - 1 - ox), // rotate 270 + flip-h (anti-transpose)
        6 => (oy, sh - 1 - ox),          // rotate clockwise 90
        7 => (oy, ox),                   // rotate 90 + flip-h (transpose)
        8 => (sw - 1 - oy, ox),          // rotate clockwise 270
        _ => (ox, oy),                   // default
    }
}

/// Rank for precision comparison (by bit depth / range); UNKNOWN folds to the 8bpc default.
pub(super) fn prec_rank(p: u32) -> u8 {
    match p {
        5 => 4, // 32BPC_FLOAT
        4 => 3, // 16BPC_FLOAT
        3 => 2, // 16BPC_UNORM
        _ => 1, // 8BPC_UNORM / _SRGB / UNKNOWN
    }
}


/// Quantise a realised f32 colour to a `D2D1_BUFFER_PRECISION`, matching how native stores an
/// intermediate before a downstream shader samples it: 8/16 UNORM round-and-clamp to [0,1], 16FLOAT
/// rounds to half, 32FLOAT (and the unknown-float case) passes through (extended range kept).
pub(crate) fn quantize_to(p: u32, c: [f32; 4]) -> [f32; 4] {
    match p {
        2 => [
            crate::sampling::srgb::roundtrip(c[0]),
            crate::sampling::srgb::roundtrip(c[1]),
            crate::sampling::srgb::roundtrip(c[2]),
            (c[3] * 255.0).round_ties_even().clamp(0.0, 255.0) / 255.0,
        ],
        5 => c,
        4 => c.map(|v| crate::float::f16_to_f32(crate::float::f32_to_f16(v))),
        3 => c.map(|v| (v * 65535.0).round_ties_even().clamp(0.0, 65535.0) / 65535.0),
        _ => c.map(|v| (v * 255.0).round_ties_even().clamp(0.0, 255.0) / 255.0),
    }
}

/// The effect a float-target draw of `e` writes from: `e` with the pass-throughs Direct2D drops peeled off (VM, r29
/// 40171: a box under an Opacity of 1 keeps its float output).
fn float_root(e: &Effect) -> usize {
    let mut e = e.clone();
    loop {
        let next = match e.0.borrow().inputs.first() {
            Some(Input::Effect(i)) if e.is_identity_op() => i.clone(),
            _ => break,
        };
        e = next;
    }
    Rc::as_ptr(&e.0) as *const () as usize
}

/// An input's buffer precision: a bitmap leaf is its own 8bpc format (B8G8R8A8), regardless of the
/// context; an effect input is that effect's output precision (context, optionally lowered by PROPERTY).
pub(super) fn input_precision(inp: &Input) -> u32 {
    match inp {
        Input::Bitmap(b) => format_precision(b.format),
        Input::Effect(e) => e.out_precision(),
        Input::List(l) => format_precision(l.baked().0.format),
    }
}

/// A DXGI format's D2D1_BUFFER_PRECISION (8BPC_UNORM for everything that is not a wide format).
fn format_precision(f: DXGI_FORMAT) -> u32 {
    match f {
        DXGI_FORMAT_R32G32B32A32_FLOAT => 5,
        DXGI_FORMAT_R16G16B16A16_FLOAT => 4,
        DXGI_FORMAT_R16G16B16A16_UNORM => 3,
        _ => 1,
    }
}

/// WARP's blit of an 8bpc image `bmp` whose top-left lands at the fractional device position `dev`, from the atlas
/// Direct2D renders it into: per device pixel, LINEAR rounds the texel position to 1/256, then lerps between the two
/// rows and between the two columns, each `a + ((b − a)·f + 128) >> 8`; NEAREST (`point`) takes the texel under the
/// pixel centre. Texels past the image are transparent. The result as a device-aligned bitmap and its top-left
/// device pixel.
///
/// Under PLUS (`plus`) WARP blits through its own integer kernel, which floors the texel position to 1/256, lerps each
/// column at 15 bits, `(256·top + fy·(bottom − top)) >> 1`, then across, `left + ⌊fx·(right − left)/256⌋`, and
/// truncates by `>> 7` before the saturating add.
fn atlas_blit(bmp: &Bitmap, dev: [f32; 2], point: bool, plus: bool) -> ([i32; 2], Bitmap) {
    let (w, h) = (bmp.width as i32, bmp.height as i32);
    let (l, t) = ((dev[0] - 1.0).floor() as i32, (dev[1] - 1.0).floor() as i32);
    let (r, b) = ((dev[0] + w as f32).ceil() as i32, (dev[1] + h as f32).ceil() as i32);
    let (ow, oh) = ((r - l).max(0) as u32, (b - t).max(0) as u32);
    let texel = |x: i32, y: i32| -> [i32; 4] {
        if x < 0 || y < 0 || x >= w || y >= h {
            return [0; 4];
        }
        let i = (y * w + x) as usize * 4;
        core::array::from_fn(|c| bmp.premul_bgra[i + c] as i32)
    };
    let lerp = |a: [i32; 4], b: [i32; 4], f: i32| -> [i32; 4] { core::array::from_fn(|c| a[c] + (((b[c] - a[c]) * f + 128) >> 8)) };
    let mut px = vec![0u8; (ow * oh * 4) as usize];
    for y in 0..oh as i32 {
        let pos = |v: f32| if plus { v.floor() as i32 } else { v.round() as i32 };
        let fv = pos(((t + y) as f32 - dev[1]) * 256.0);
        let (y0, ay) = (fv >> 8, fv & 255);
        for x in 0..ow as i32 {
            let fu = pos(((l + x) as f32 - dev[0]) * 256.0);
            let (x0, ax) = (fu >> 8, fu & 255);
            let v = if point {
                texel(((l + x) as f32 + 0.5 - dev[0]).floor() as i32, ((t + y) as f32 + 0.5 - dev[1]).floor() as i32)
            } else if plus {
                let column = |a: [i32; 4], b: [i32; 4]| -> [i32; 4] { core::array::from_fn(|c| (256 * a[c] + ay * (b[c] - a[c])) >> 1) };
                let (left, right) = (column(texel(x0, y0), texel(x0, y0 + 1)), column(texel(x0 + 1, y0), texel(x0 + 1, y0 + 1)));
                core::array::from_fn(|c| (left[c] + ((ax * (right[c] - left[c])) >> 8)) >> 7)
            } else {
                let left = lerp(texel(x0, y0), texel(x0, y0 + 1), ay);
                let right = lerp(texel(x0 + 1, y0), texel(x0 + 1, y0 + 1), ay);
                lerp(left, right, ax)
            };
            let i = (y as u32 * ow + x as u32) as usize * 4;
            for c in 0..4 {
                px[i + c] = v[c].clamp(0, 255) as u8;
            }
        }
    }
    ([l, t], Bitmap { width: ow, height: oh, premul_bgra: px, ..bmp.clone() })
}

/// A float (R32/R16_FLOAT) input bitmap decoded to the 8bpc premultiplied-BGRA the filter and affine
/// resample paths read. Those paths index `premul_bgra` as 4-byte BGRA8 (`builtin::filters::Buf` /
/// `premul_at` / `sample`), so a 16-/8-byte float texel would be read as garbage. Decode each texel
/// with the format-aware [`bitmap_texel`] and store it through the same UNORM8 round/clamp as a native
/// 8bpc node ([`store`]). ponytail: the filter family's working buffer is UNORM8, so an HDR value (>1)
/// clamps to 1 here; byte-exact HDR preservation for a float input would need the float-intermediate
/// rework (the open effect-precision item) plus a native oracle to pin it — neither available now.
pub(crate) fn float_input_to_bgra8(b: &Bitmap) -> Bitmap {
    let (w, h) = (b.width as usize, b.height as usize);
    let mut data = vec![0u8; w * h * 4];
    for i in 0..w * h {
        store(&mut data, i * 4, bitmap_texel(b, (i % w) as i32, (i / w) as i32));
    }
    Bitmap {
        width: b.width,
        height: b.height,
        premul_bgra: data,
        format: DXGI_FORMAT_B8G8R8A8_UNORM,
        alpha: D2D1_ALPHA_MODE_PREMULTIPLIED,
        dpi: b.dpi,
    }
}

thread_local! {
    /// Set while a distributed Composite's inputs draw ([`plan::distributed`]): their graph is already planned.
    static PLAN_HELD: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

thread_local! {
    /// The effect a DrawImage is drawing straight into a float target (its address; 0 when none): its
    /// last draw writes that target, so a custom shader there keeps a linear tap's full precision.
    static FLOAT_DRAWN: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The visible region of the effect draw being realized (device pixels), for the blurs inside it.
    static DRAW_VIEWPORT: std::cell::Cell<Option<Rect>> = const { std::cell::Cell::new(None) };
    /// The draw viewport's top-left in effect pixels when it is not the requested rect's (a direct
    /// `DrawImage` under a clip: the passes cover the clip, the last lands in target coordinates).
    static DRAW_ORIGIN: std::cell::Cell<Option<[i32; 2]>> = const { std::cell::Cell::new(None) };
    /// The drawn root effect (by identity) and its output's request when an unbatched slow-path CUBIC draw realizes it
    /// ([`Effect::hq_slot_origin`]).
    static HQ_SLOW_REQ: std::cell::Cell<Option<(usize, Rect)>> = const { std::cell::Cell::new(None) };
    /// Set while a consumer that reads its input through a mirror border (a HARD DirectionalBlur) realizes a Shadow:
    /// the border's cells then colour the Shadow's stored blur (`filters::shadow_in`).
    static SHADOW_STORED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The image rectangle (image space) a DrawImage of the root effect is cropped to: the realize covers only it.
    pub(crate) static IMAGE_CROP: std::cell::Cell<Option<[f32; 4]>> = const { std::cell::Cell::new(None) };
    /// [`DeviceContext::realize_effect_cells`]' request, in effect pixels, for the root blur.
    static BORDER_CELLS: std::cell::RefCell<Option<builtin::filters::BorderCells>> = const { std::cell::RefCell::new(None) };
    /// Pixels per DIP for the DIP-valued properties (the blurs' standard deviation) while an effect
    /// draws in device-pixel space (`DeviceContext::draw_image_effect` at a DPI other than 96).
    static DIP_SCALE: std::cell::Cell<f32> = const { std::cell::Cell::new(1.0) };
}

/// Realise an input to a bitmap + its origin (a bitmap leaf is used directly). Used by resampling
/// nodes that need a sampleable texture.
/// The per-axis round-vs-floor the DisplacementMap image `Sample` uses. WARP's `Sample` rounds
/// (cvtps2dq) the Q8 texel address by the realized texture size: a direct bitmap is content-realized
/// (a non-pow2 size rounds), an intermediate is placed in the pow2 realization atlas (floor). So round
/// only a content bitmap's non-pow2 axes; an intermediate (or a pow2 bitmap) floors as before.
fn disp_round(input: &Input, image: &Bitmap) -> [bool; 2] {
    let content = matches!(input, Input::Bitmap(_));
    [
        content && !(image.width as u32).is_power_of_two(),
        content && !(image.height as u32).is_power_of_two(),
    ]
}

fn realize_input(inp: &Input) -> (Bitmap, i32, i32) {
    match inp {
        // A float leaf is decoded to 8bpc first: its consumers (the 8bpc filter family + affine
        // resample) read `premul_bgra` as BGRA8, which would otherwise mis-read the raw float texels.
        Input::Bitmap(b) if b.format != DXGI_FORMAT_B8G8R8A8_UNORM => (float_input_to_bgra8(b), 0, 0),
        Input::Bitmap(b) => ((**b).clone(), 0, 0),
        // So is a float effect output an 8bpc consumer reads (it lowered its own precision).
        Input::Effect(e) => match e.realize() {
            (b, x, y) if crate::float::is_highp(b.format) => (float_input_to_bgra8(&b), x, y),
            r => r,
        },
        Input::List(l) => match &*l.baked() {
            (b, x, y) if b.format != DXGI_FORMAT_B8G8R8A8_UNORM => (float_input_to_bgra8(b), *x, *y),
            (b, x, y) => (b.clone(), *x, *y),
        },
    }
}

/// Realize an input over the finite device rect `r` (placed at its top-left) — for a filter whose
/// input is INFINITE (Border/Tile extend): realizing the input's own (infinite) bounds would overflow,
/// so the filter asks only for the sub-rect its output needs. A finite bitmap input has no infinite
/// case and realizes whole. `highp` keeps a float effect output in float (8bpc otherwise).
fn realize_input_at(inp: &Input, r: Rect, highp: bool) -> (Bitmap, i32, i32) {
    match inp {
        Input::Effect(e) => {
            let bmp = e.realize_rect(r);
            let bmp = if !highp && crate::float::is_highp(bmp.format) { float_input_to_bgra8(&bmp) } else { bmp };
            (bmp, r.l, r.t)
        }
        _ if highp => realize_input_hp(inp),
        _ => realize_input(inp),
    }
}

/// [`realize_input`] for a consumer above 8bpc: a float leaf or float effect output stays float (its
/// readers decode by format), so HDR and sub-LSB values reach the effect.
fn realize_input_hp(inp: &Input) -> (Bitmap, i32, i32) {
    match inp {
        Input::Bitmap(b) => ((**b).clone(), 0, 0),
        Input::Effect(e) => e.realize(),
        Input::List(l) => (*l.baked()).clone(),
    }
}

/// A premultiplied R32G32B32A32_FLOAT bitmap of `w × h` RGBA values.
fn float_bitmap(w: u32, h: u32, px: impl Iterator<Item = [f32; 4]>) -> Bitmap {
    Bitmap {
        width: w,
        height: h,
        premul_bgra: px.flat_map(|c| c.into_iter().flat_map(f32::to_le_bytes)).collect(),
        format: DXGI_FORMAT_R32G32B32A32_FLOAT,
        alpha: D2D1_ALPHA_MODE_PREMULTIPLIED,
        dpi: (96.0, 96.0),
    }
}

/// [`sample`] for a bitmap of any format, in f32: nearest, or bilinear with 8-bit fractional weights.
fn sample_hp(b: &Bitmap, tx: f32, ty: f32, linear: bool, soft: bool) -> [f32; 4] {
    let (w, h) = (b.width as i32, b.height as i32);
    let fetch = |x: i32, y: i32| {
        if soft {
            bitmap_texel(b, x, y)
        } else {
            bitmap_texel(b, x.clamp(0, w - 1), y.clamp(0, h - 1))
        }
    };
    if !linear {
        return fetch(tx.floor() as i32, ty.floor() as i32);
    }
    let (fx, fy) = (tx - 0.5, ty - 0.5);
    let (x0, y0) = (fx.floor(), fy.floor());
    let q = |f: f32| (f * 256.0).floor() / 256.0;
    let (wx, wy) = (q(fx - x0), q(fy - y0));
    let (x0, y0) = (x0 as i32, y0 as i32);
    let (c00, c10, c01, c11) = (fetch(x0, y0), fetch(x0 + 1, y0), fetch(x0, y0 + 1), fetch(x0 + 1, y0 + 1));
    core::array::from_fn(|k| {
        let top = c00[k] + (c10[k] - c00[k]) * wx;
        let bot = c01[k] + (c11[k] - c01[k]) * wx;
        top + (bot - top) * wy
    })
}

/// Row-major 4×4 identity (3D-transform default matrix).
const IDENTITY_M44: [f32; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

/// Premultiplied RGBA f32 at integer pixel `(x,y)`; transparent black outside the bitmap (D2D
/// zero-fill for an unbound/out-of-range input region).
fn bitmap_texel(b: &Bitmap, x: i32, y: i32) -> [f32; 4] {
    if x < 0 || y < 0 || x >= b.width as i32 || y >= b.height as i32 {
        return [0.0; 4];
    }
    let n = y as usize * b.width as usize + x as usize;
    let d = &b.premul_bgra;
    // A float source bitmap keeps its own precision (R32/R16 FLOAT, RGBA order); a byte bitmap is BGRA8.
    if b.format == DXGI_FORMAT_R32G32B32A32_FLOAT {
        let i = n * 16;
        core::array::from_fn(|k| f32::from_le_bytes([d[i + 4 * k], d[i + 4 * k + 1], d[i + 4 * k + 2], d[i + 4 * k + 3]]))
    } else if b.format == DXGI_FORMAT_R16G16B16A16_FLOAT {
        let i = n * 8;
        core::array::from_fn(|k| crate::float::f16_to_f32(u16::from_le_bytes([d[i + 2 * k], d[i + 2 * k + 1]])))
    } else if b.format == DXGI_FORMAT_R16G16B16A16_UNORM {
        let i = n * 8;
        core::array::from_fn(|k| u16::from_le_bytes([d[i + 2 * k], d[i + 2 * k + 1]]) as f32 / 65535.0)
    } else if b.format == DXGI_FORMAT_A8_UNORM {
        [0.0, 0.0, 0.0, d[n] as f32 / 255.0]
    } else if b.format == DXGI_FORMAT_R8_UNORM {
        [d[n] as f32 / 255.0, 0.0, 0.0, 1.0]
    } else if b.format == DXGI_FORMAT_R8G8_UNORM {
        [d[n * 2] as f32 / 255.0, d[n * 2 + 1] as f32 / 255.0, 0.0, 1.0]
    } else if b.format == DXGI_FORMAT_R8G8B8A8_UNORM {
        core::array::from_fn(|k| d[n * 4 + k] as f32 / 255.0)
    } else {
        let i = n * 4;
        [d[i + 2] as f32 / 255.0, d[i + 1] as f32 / 255.0, d[i] as f32 / 255.0, d[i + 3] as f32 / 255.0]
    }
}

/// Store premultiplied RGBA f32 as premultiplied BGRA8, rounding each channel to nearest-even and
/// clamping (the final/only quantisation; WARP's UNORM store).
fn store(data: &mut [u8], i: usize, c: [f32; 4]) {
    let q = |v: f32| (v * 255.0).round_ties_even().clamp(0.0, 255.0) as u8;
    data[i] = q(c[2]);
    data[i + 1] = q(c[1]);
    data[i + 2] = q(c[0]);
    data[i + 3] = q(c[3]);
}

// --- extend / tile / composite / colour-matrix math --------------------------------------------

/// Fold an index into `[0, n)` per a `D2D1_EXTEND_MODE` (0 clamp, 1 wrap, 2 mirror). `n ≥ 1`.
fn fold(c: i32, n: i32, mode: u32) -> i32 {
    if n <= 0 {
        return 0;
    }
    match mode {
        1 => c.rem_euclid(n),
        2 => {
            let m = c.rem_euclid(2 * n);
            if m < n {
                m
            } else {
                2 * n - 1 - m
            }
        }
        _ => c.clamp(0, n - 1),
    }
}

/// Tile point-sample index: wrap the scene centre into the tile window, then floor to a texel
/// (Tile: `wrapped = rel − floor(rel/size)·size`, MinMagMipPoint).
fn tile_coord(x: i32, origin: i32, size: i32) -> i32 {
    if size <= 0 {
        return origin;
    }
    let rel = x as f32 + 0.5 - origin as f32;
    let wrapped = rel - (rel / size as f32).floor() * size as f32;
    origin + wrapped.floor() as i32
}

/// Porter-Duff factors `(srcF, dstF)` for a `D2D1_COMPOSITE_MODE` (R21).
/// One Composite step of the premultiplied `src` onto `dst`. MASK_INVERT is no linear blend: rgb `(1 − αs)·d + (1 − d)·s`
/// and the destination's alpha, as the target's MASK_INVERT composite (VM: a Composite of two bitmaps, a source texel
/// past the destination keeping its colour at alpha 0, r27 32542).
fn composite_px(mode: u32, src: [f32; 4], dst: [f32; 4]) -> [f32; 4] {
    if mode == 12 {
        return [(1.0 - src[3]) * dst[0] + (1.0 - dst[0]) * src[0], (1.0 - src[3]) * dst[1] + (1.0 - dst[1]) * src[1],
            (1.0 - src[3]) * dst[2] + (1.0 - dst[2]) * src[2], dst[3]];
    }
    let (sf, df) = composite_factors(mode, src[3], dst[3]);
    core::array::from_fn(|c| sf * src[c] + df * dst[c])
}

fn composite_factors(mode: u32, src_a: f32, dst_a: f32) -> (f32, f32) {
    match mode {
        1 => (1.0 - dst_a, 1.0),       // DestinationOver
        2 => (dst_a, 0.0),             // SourceIn
        3 => (0.0, src_a),             // DestinationIn
        4 => (1.0 - dst_a, 0.0),       // SourceOut
        5 => (0.0, 1.0 - src_a),       // DestinationOut
        6 => (dst_a, 1.0 - src_a),     // SourceAtop
        7 => (1.0 - dst_a, src_a),     // DestinationAtop
        8 => (1.0 - dst_a, 1.0 - src_a), // Xor
        9 => (1.0, 1.0),               // Plus
        10 | 11 => (1.0, 0.0),         // SourceCopy, BoundedSourceCopy (its bounds are the caller's)
        _ => (1.0, 1.0 - src_a),       // SourceOver (0)
    }
}

// --- effect bilinear texture filter (R3) ------------------------------------------------------

/// Sample the input bitmap at texel-space `(tx, ty)` (texel centres at integer+0.5). SOFT reads
/// transparent black outside the bitmap, HARD clamps to the edge texel. Returns premultiplied RGBA
/// f32. Both filters go through WARP's JIT-confirmed effect `sample` kernel ([`sample::sample_linear`]:
/// `·257/65536` corner conversion, Q8 weights, f32 product-sum) — the affine transform's pixel shader
/// calls that same `Sample` intrinsic, so reusing it (not `/255` + a nested lerp) closes the ≤1-LSB gap.
fn sample(b: &Bitmap, tx: f32, ty: f32, linear: bool, soft: bool) -> [f32; 4] {
    let (w, h) = (b.width as i32, b.height as i32);
    let fetch = |x: i32, y: i32| -> [i32; 4] {
        let (x, y) = if soft {
            if x < 0 || y < 0 || x >= w || y >= h {
                return [0; 4];
            }
            (x, y)
        } else {
            (x.clamp(0, w - 1), y.clamp(0, h - 1))
        };
        let i = (y as usize * b.width as usize + x as usize) * 4;
        let d = &b.premul_bgra;
        [d[i + 2] as i32, d[i + 1] as i32, d[i] as i32, d[i + 3] as i32] // R,G,B,A bytes
    };
    if !linear {
        return fetch(tx.floor() as i32, ty.floor() as i32).map(|c| c as f32 / 255.0);
    }
    sample::sample_linear(tx, ty, fetch)
}

// --- DeviceContext wiring ----------------------------------------------------------------------

impl DeviceContext {
    /// Render a **closed** command list over the effect-pixel rect `r` into a transparent intermediate, SOURCE_OVER:
    /// what an effect reads of a command-list input. Effects run in device pixels, so the list draws at the
    /// context's DPI (scaled by [`DIP_SCALE`] inside a device-pixel draw) with no world transform, `r`'s top-left
    /// at the intermediate's origin. A float target, or a float buffer precision, renders it at float as native
    /// does (a gradient list under UnPremultiply into a Float32 target equals the direct fill; probe); else BGRA8.
    /// A list's content bounds in the space its input is rendered in (the context DPI times the effect's DIP
    /// scale, no world transform): the union of its draws' bounds, each taken as recorded and rounded out to
    /// pixels. `None` when a draw leaves them undecided (a glyph run, an effect) or they are unbounded.
    fn list_content(&mut self, list: &CommandList) -> Option<Rect> {
        let k = DIP_SCALE.with(|s| s.get());
        let (dx, dy) = self.get_dpi();
        let cmds = list.0.borrow().cmds.clone();
        let mut rects = Vec::new();
        if !self.cmds_rects(&cmds, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], (dx * k, dy * k), &mut rects) {
            return None;
        }
        let u = rects.iter().fold(None::<[f32; 4]>, |u, r| {
            Some(u.map_or(*r, |u| [u[0].min(r[0]), u[1].min(r[1]), u[2].max(r[2]), u[3].max(r[3])]))
        });
        let Some(u) = u else { return Some(Rect { l: 0, t: 0, r: 0, b: 0 }) };
        let lim = (1u32 << 30) as f32;
        if u.iter().any(|v| v.abs() >= lim) {
            return None;
        }
        Some(Rect { l: u[0] as i32, t: u[1] as i32, r: u[2] as i32, b: u[3] as i32 })
    }

    fn render_command_list(&mut self, list: &CommandList, r: Rect) -> Bitmap {
        let k = DIP_SCALE.with(|s| s.get());
        let (dx, dy) = self.get_dpi();
        let dpi = self.target_bitmap().map_or(96.0, |b| b.desc.dpi_x);
        let format = self.target_bitmap().map(|b| b.desc.format).filter(|f| crate::float::is_highp(*f))
            .or(match self.buffer_precision { 5 => Some(DXGI_FORMAT_R32G32B32A32_FLOAT), 4 => Some(DXGI_FORMAT_R16G16B16A16_FLOAT), _ => None });
        let (w, h) = (r.width(), r.height());
        let inter = match format {
            Some(f) => crate::float::float_target(w, h, dpi, f, D2D1_ALPHA_MODE_PREMULTIPLIED),
            None => crate::context::bitmap_target(w, h, dpi),
        };
        // The replay runs on this context (a list from another device latches its domain error here, as a draw
        // does), then its transform, DPI and target are put back.
        let saved = self.set_target_bitmap(inter);
        let (tr, unit) = (self.get_transform(), self.get_unit_mode());
        self.set_unit_mode(0);
        self.set_dpi(dx * k, dy * k);
        // ponytail: a fractional DIP_SCALE leaves r's origin a rounding off whole; no scene draws a list input there.
        self.set_transform([1.0, 0.0, 0.0, 1.0, -(r.l as f32) * 96.0 / (dx * k), -(r.t as f32) * 96.0 / (dy * k)]);
        let outer = std::mem::replace(&mut self.list_input, true);
        self.draw_image(list, [0.0, 0.0], 0);
        self.list_input = outer;
        self.set_dpi(dx, dy);
        self.set_transform(tr);
        self.set_unit_mode(unit);
        let realized = self.take_target().expect("intermediate was just bound");
        if let Some(prev) = saved {
            self.set_target_bitmap(prev);
        }
        Bitmap {
            width: w,
            height: h,
            premul_bgra: realized.data,
            format: realized.desc.format,
            alpha: realized.desc.alpha_mode,
            dpi: (realized.desc.dpi_x, realized.desc.dpi_y),
        }
    }

    /// Render every command-list input of `effect`'s graph for a draw that shows the effect-pixel rect `need`:
    /// each over `need` grown by the filter reach between it and the output, Direct2D's required-input rect
    /// for the filters. Run at every entry that evaluates a graph, before anything reads its bounds.
    /// ponytail: a transform between a list and the output (Affine2D, Scale, 3D) maps its need 1:1 here.
    fn bake_lists(&mut self, effect: &Effect, need: Rect) {
        fn walk(e: &Effect, reach: i32, depth: u32, out: &mut Vec<(Effect, u32, Rc<ListInput>, i32)>) {
            if depth > 64 {
                return;
            }
            let reach = reach.saturating_add(if e.0.borrow().kind.is_filter() { e.filter_reach().max(0) } else { 0 });
            let inputs = e.0.borrow().inputs.clone();
            for (i, inp) in inputs.iter().enumerate() {
                match inp {
                    Input::List(l) => out.push((e.clone(), i as u32, l.clone(), reach)),
                    Input::Effect(c) => walk(c, reach, depth + 1, out),
                    Input::Bitmap(_) => {}
                }
            }
        }
        let mut lists = Vec::new();
        walk(effect, 0, 0, &mut lists);
        for (e, i, l, k) in lists {
            let r = Rect { l: need.l.saturating_sub(k), t: need.t.saturating_sub(k), r: need.r.saturating_add(k), b: need.b.saturating_add(k) };
            // An unbounded or degenerate need (a singular transform) renders nothing, as a target-sized cap would.
            const MAX_DIM: u32 = 16384;
            let r = if r.width() <= MAX_DIM && r.height() <= MAX_DIM { r } else { Rect { l: 0, t: 0, r: 0, b: 0 } };
            // The list is the image of what it records: its bounds are its content's, within the need. One that records
            // nothing drawable (its only path has infinite points) is empty, and an effect reading it outputs nothing.
            let r = match self.list_content(&l.list) {
                Some(c) if c.width() == 0 || c.height() == 0 => Rect { l: r.l, t: r.t, r: r.l, b: r.t },
                Some(c) => {
                    let i = r.intersect(c);
                    if i.width() == 0 || i.height() == 0 { Rect { l: r.l, t: r.t, r: r.l, b: r.t } } else { i }
                }
                _ => r,
            };
            let bmp = self.render_command_list(&l.list, r);
            *l.baked.borrow_mut() = Rc::new((bmp, r.l, r.t));
            e.set_input(i, Input::List(l)); // as a rebind does: the consumer's memo is stale
        }
    }

    /// The drawing target's buffer precision (D2D1_BUFFER_PRECISION) from its DXGI format.
    fn target_precision(&self) -> u32 {
        self.target.as_ref().map_or(1, |t| format_precision(t.desc.format))
    }

    /// An effect's output over the integer pixel rect `r` ([l, t, r, b]), realized as `DrawImage`
    /// would and stored 8bpc: the texels an `ID2D1ImageBrush` over the effect samples. `r` (the
    /// brush's BorderRect tile) is also the passes' requested rect, drawn at an atlas origin.
    pub(crate) fn realize_effect_px(&mut self, effect: &Effect, r: [i32; 4]) -> Bitmap {
        self.at_dip_scale(|dc| dc.realize_effect_px_inner(effect, r))
    }

    /// Run `f` with [`DIP_SCALE`] at this context's DIP-to-pixel scale: an image brush realizes its effect in device
    /// pixels as `DrawImage` does, so a blur's σ in DIPs is σ·dpi/96 there too (r27 32161: a blur image brush at 120 dpi
    /// blurred by σ px). At unequal axis DPIs or in pixel units the scale stays, as `DrawImage` keeps it.
    fn at_dip_scale<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let (dpx, dpy) = self.get_dpi();
        if self.get_unit_mode() != 0 || dpx != dpy || dpx == 96.0 {
            return f(self);
        }
        let prev = DIP_SCALE.with(|s| s.replace(s.get() * dpx / 96.0));
        bump_bounds_epoch();
        let out = f(self);
        DIP_SCALE.with(|s| s.set(prev));
        bump_bounds_epoch();
        out
    }

    fn realize_effect_px_inner(&mut self, effect: &Effect, r: [i32; 4]) -> Bitmap {
        let r = Rect { l: r[0], t: r[1], r: r[2], b: r[3] };
        let _ = effect.prepare_for_render();
        effect.set_render_precision(self.buffer_precision, self.target_precision());
        plan::clear(effect);
        self.bake_lists(effect, r);
        let prev = DRAW_VIEWPORT.with(|v| v.replace(Some(r)));
        let bmp = if effect.graph_valid() && r.width() > 0 && r.height() > 0 { effect.realize_rect_clipped(r, [0.0; 2], Some(r)) } else { float_bitmap(0, 0, std::iter::empty()) };
        DRAW_VIEWPORT.with(|v| v.set(prev));
        if crate::float::is_highp(bmp.format) { float_input_to_bgra8(&bmp) } else { bmp }
    }

    /// An image brush's border cells over `effect` ([`builtin::filters::BorderCells`], effect pixels):
    /// the cells-drawn intermediate, or `None` when the effect is not a full-res GaussianBlur root or its last pass does
    /// not fold into the cells ([`builtin::filters::blur_folds_into_border`]; the caller then fills the cells from the
    /// realized tile `tile`). The passes cover `request`, what the border asks of its input.
    pub(crate) fn realize_effect_cells(&mut self, effect: &Effect, tile: [i32; 4], request: [i32; 4], cells: builtin::filters::BorderCells) -> Option<Bitmap> {
        self.at_dip_scale(|dc| dc.realize_effect_cells_inner(effect, tile, request, cells))
    }

    fn realize_effect_cells_inner(&mut self, effect: &Effect, tile: [i32; 4], request: [i32; 4], cells: builtin::filters::BorderCells) -> Option<Bitmap> {
        let d = effect.0.borrow();
        let full_res = d.kind == Kind::GaussianBlur && builtin::filters::blur_full_res(effect.sigma(), effect.get_u32(1, 1));
        drop(d);
        if !full_res || !effect.graph_valid() || cells.size[0] <= 0 || cells.size[1] <= 0 {
            return None;
        }
        let (b, o) = (effect.bounds(), cells.origin);
        let need = [o[0], o[1], o[0] + cells.size[0], o[1] + cells.size[1]];
        if !builtin::filters::blur_folds_into_border(effect.sigma(), effect.get_u32(1, 1), [b.l, b.t, b.r, b.b], request, need) {
            return None;
        }
        let _ = effect.prepare_for_render();
        effect.set_render_precision(self.buffer_precision, self.target_precision());
        plan::clear(effect);
        self.bake_lists(effect, Rect { l: tile[0], t: tile[1], r: tile[2], b: tile[3] });
        let n = Rect { l: cells.origin[0], t: cells.origin[1], r: cells.origin[0] + cells.size[0], b: cells.origin[1] + cells.size[1] };
        BORDER_CELLS.with(|c| c.replace(Some(cells)));
        let bmp = effect.realize_rect_clipped(n, [0.0; 2], Some(Rect { l: request[0], t: request[1], r: request[2], b: request[3] }));
        // Unconsumed (a high-precision graph realizes elsewhere): not cells-shaped.
        if BORDER_CELLS.with(|c| c.take()).is_some() || crate::float::is_highp(bmp.format) {
            return None;
        }
        Some(bmp)
    }

    /// `CreateEffect(clsid)`: build a built-in effect (consulting the built-in registry first), else a
    /// custom effect from the XML registry (running its `Initialize` to build the transform graph),
    /// else `NOTFOUND` (native returns raw `0x80070490`, not the documented
    /// `D2DERR_EFFECT_IS_NOT_REGISTERED`). The context DPI/unit are captured on the effect.
    pub fn create_effect(&self, clsid: GUID) -> Result<Effect, HRESULT> {
        let dpi = self.get_dpi();
        if let Some(kind) = kind_of(&clsid) {
            return Ok(Effect::new_in(kind, dpi));
        }
        custom::create_registered(clsid, dpi, self.factory_id())
    }

    /// `DrawImage(effect, offset, interpolation, composite)`: realise the effect output and draw it
    /// through the current device transform. All `custom-fx` scenes draw SOURCE_OVER; realised output
    /// is drawn by the shared bitmap path, so the draw transform scales/places it (R2/R3/R7/R9).
    /// Infinite output (Flood/Border/Tile at the root) is clipped to the target, as native does.
    /// `DrawImage` of an effect. A malformed effect graph — a reachable effect node with fewer inputs
    /// than its realize indexes (G5: a Paint.NET GPU plugin that leaves an input unset, directly or
    /// through a command-list input that was never realised) — would index an input out of bounds.
    /// Validate the whole graph FIRST and latch E_INVALIDARG without drawing, as native rejects a
    /// missing effect input before it draws. This is deliberately panic-free (no `catch_unwind`): the
    /// core is also built for `wasm32-unknown-unknown` (`panic=abort`), our main browser target, where
    /// an out-of-bounds index would kill the page and diverge from native. (The FFI keeps a
    /// `catch_unwind` net only for unwinding targets, as a last resort. Exact HRESULT per malformed
    /// shape is VM-gated; E_INVALIDARG is native's missing-input error.)
    /// DrawImage of an effect composites by its own `D2D1_COMPOSITE_MODE`, NOT the caller's
    /// `SetPrimitiveBlend`. Reset the primitive blend to the composite's mapped value for the realize so a
    /// nested sampled draw (the HQ-cubic output) does not inherit the outer blend (r27 32253: a PLUS-era
    /// ADD leaked into an effect drawn SOURCE_OVER). A recorded effect keeps the live state, replayed under it.
    pub fn draw_image_effect(&mut self, effect: &Effect, offset: [f32; 2], interpolation: u32, composite: u32) {
        let a = self.image_api();
        self.draw_image_effect_call(effect, offset, interpolation, composite);
        self.image_queue(a);
    }

    fn draw_image_effect_call(&mut self, effect: &Effect, offset: [f32; 2], interpolation: u32, composite: u32) {
        if self.recording.is_some() {
            return self.draw_image_effect_inner(effect, offset, interpolation, composite);
        }
        // Every composite mode defers into the open atlas set: SOURCE_OVER, PLUS and BOUNDED_SOURCE_COPY as an image
        // brush fill, the others through a command list drawn the same way.
        plan::batch_enter(std::mem::take(&mut self.effect_batch));
        self.draw_image_effect_batched(effect, offset, interpolation, composite);
        if let Some(set) = plan::batch_exit() {
            self.effect_batch = set;
        }
    }

    fn draw_image_effect_batched(&mut self, effect: &Effect, offset: [f32; 2], interpolation: u32, composite: u32) {
        if let Some((input, o)) = self.whole_pixel_offset(effect, offset) {
            match input {
                Input::Bitmap(b) => return self.draw_image_bitmap(&b, o, None, interpolation, composite),
                Input::Effect(e) => return self.draw_image_effect(&e, o, interpolation, composite),
                Input::List(_) => {}
            }
        }
        let prev = self.get_primitive_blend();
        self.set_primitive_blend(crate::context::composite_blend(composite));
        self.draw_image_effect_inner(effect, offset, interpolation, composite);
        self.set_primitive_blend(prev);
    }

    fn draw_image_effect_inner(&mut self, effect: &Effect, offset: [f32; 2], interpolation: u32, composite: u32) {
        if !self.draw_image_args_ok(interpolation, composite) {
            return;
        }
        if !effect.graph_valid() {
            self.process_new_error(E_INVALIDARG);
            return;
        }
        // A command list keeps the effect itself and evaluates the draw where the list is drawn (its target's clip and
        // precision, the effect's state then), as Direct2D's does; nothing about the recording context bounds it.
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            self.record(RecCmd::Effect { state, dpi, effect: effect.clone(), offset, interpolation, composite });
            return;
        }
        // TryDrawImageFastPath tests the effect's own placement, as for a bitmap: a NaN that passes it
        // draws nothing, or clears under the copy modes (w_bm ef2, r11 16982).
        if self.recording.is_none() && self.nan_fast_placement(offset, interpolation, false) {
            if matches!(composite, 10 | 11) {
                self.composite_empty(10);
            }
            return;
        }
        // Drive each custom effect's PrepareForRender once before realizing (native does so every
        // draw), so a constant buffer set after Initialize reaches the shader for this draw. [G2]
        // Before the request check: a ComputeSharp transform mapper's reach comes from those constants
        // (Koh's Relief passes ask an infinite rect until PrepareForRender, then 6 px).
        // A PrepareForRender that fails fails the draw with its HRESULT (VM: EndDraw returns E_INVALIDARG for a 16385-entry
        // resource texture created there).
        if let Err(hr) = effect.prepare_for_render() {
            self.process_new_error(hr);
            return;
        }
        // A custom shader asking an infinite rect of an input Direct2D must realize unbounded fails the draw.
        if effect.infinite_request_too_large() {
            self.process_new_error(crate::state::INTERMEDIATE_TOO_LARGE);
            return;
        }
        // HIGH_QUALITY_CUBIC under SOURCE_OVER, PLUS or BOUNDED_SOURCE_COPY sizes its intermediate as for a bitmap
        // ([`DeviceContext::hq_cubic_overflow`]): off the fast path, a placement that leaves a corner non-finite fails
        // with E_INVALIDARG (sc152: an effect under an infinite world element or at an infinite offset fails in those
        // three modes, DESTINATION_OVER and LINEAR draw; r27 32979). A finite effect only: an infinite one is unprobed.
        if interpolation == 5 && matches!(composite, 0 | 9 | 11) && self.replay_depth == 0 && !self.blending {
            let b = effect.bounds();
            let finite = b.l != i32::MIN && b.t != i32::MIN && b.r != i32::MAX && b.b != i32::MAX && b.width() > 0 && b.height() > 0;
            let w = self.get_transform();
            let fast = w[..4] == [1.0, 0.0, 0.0, 1.0]
                && crate::context::fast_whole(offset[0] + w[4]) && crate::context::fast_whole(offset[1] + w[5]);
            if finite && !fast {
                let r = [offset[0] + b.l as f32, offset[1] + b.t as f32, offset[0] + b.r as f32, offset[1] + b.b as f32];
                if self.hq_cubic_overflow(r, [b.width() as f32, b.height() as f32]) {
                    self.process_new_error(E_INVALIDARG);
                    return;
                }
            }
        }
        // A non-finite device placement that the whole-aligned NaN fast path above did not catch — a
        // ±inf or non-whole-NaN offset, or a non-finite world transform — is UNPLACEABLE: native realizes
        // the effect into an empty intermediate and composites THAT (`composite_empty`), so the keep-modes
        // (SOURCE_OVER/DEST_OVER/SOURCE_ATOP/…) keep the backdrop and the IN/OUT/COPY set clear it. We were
        // instead placing the effect at a finite-garbage offset and compositing a real (e.g. Flood) output
        // (fuzz r15 20770 offset.y=inf SOURCE_ATOP, 20898/20325 offset.x=-inf DEST_OVER — all keep the clear).
        if self.recording.is_none() {
            let w = self.get_transform();
            let (dpx, dpy) = self.get_dpi();
            let s = if self.get_unit_mode() == 0 { [dpx / 96.0, dpy / 96.0] } else { [1.0; 2] };
            let p = [(offset[0] + w[4]) * s[0], (offset[1] + w[5]) * s[1]];
            // A huge offset whose visible region collapses in image space is unplaceable too (sc71 clip16-of-64).
            let collapses = w[..4] == [1.0, 0.0, 0.0, 1.0] && self.offset_collapses(p);
            if !p[0].is_finite() || !p[1].is_finite() || w[..4].iter().any(|v| !v.is_finite()) || collapses {
                self.composite_empty(composite);
                return;
            }
        }
        // D2D effects run in device-pixel space: a bitmap input's pixels are the effect's (no DPI
        // compensation), DIP-valued properties scale by dpi/96, and the output lands 1:1 on device pixels
        // (r4 5077: a 13×8 bitmap's ColorMatrix covers 14×9 device px at 120 dpi; the 192-dpi blur
        // pre-downscales the bitmap by 2/3 for σ = 6 px). Drawing at 96 dpi with the offset and the world
        // translation scaled reproduces that placement.
        let (dpx, dpy) = self.get_dpi();
        // At unequal axis DPIs the output stays 1:1 on device pixels and only the offset and translation scale, per axis
        // (sc167: a ColorMatrix at 48x192, 96x192 and 192x96 dpi). A graph with a DIP-valued property is left to the DIP
        // path there: one DIP_SCALE cannot serve two axes, and native's per-axis rule for it is unprobed.
        if self.get_unit_mode() == 0 && (dpx != 96.0 || dpy != 96.0) && (dpx == dpy || !effect.dip_valued()) && self.recording.is_none() && effect.pixel_space_ok() {
            let (k, ky) = (dpx / 96.0, dpy / 96.0);
            let tr = self.get_transform();
            self.set_dpi(96.0, 96.0);
            self.set_transform([tr[0], tr[1], tr[2], tr[3], k * tr[4], ky * tr[5]]);
            // Bounds read DIP_SCALE (a blur's σ, the lighting stage), so the memoized ones go stale across the switch.
            let prev = DIP_SCALE.with(|s| s.replace(k));
            bump_bounds_epoch();
            self.draw_image_effect(effect, [offset[0] * k, offset[1] * ky], interpolation, composite);
            DIP_SCALE.with(|s| s.set(prev));
            bump_bounds_epoch();
            self.set_transform(tr);
            self.set_dpi(dpx, dpy);
            return;
        }
        // Under a scale or rotation an infinite effect goes through the transform stage's intermediate.
        if !PLAN_HELD.with(|h| h.get()) && self.get_transform()[..4] != [1.0, 0.0, 0.0, 1.0] && effect.bounds_exact() {
            let b = effect.bounds();
            if self.transformed_image_done([b.l, b.t, b.r, b.b], offset, interpolation, composite) {
                return;
            }
        }
        // Push this context's bufferPrecision down the effect graph (the governing term in Direct2D's
        // precision rule) so the realize quantizes each shader input at the drawing context's precision.
        effect.set_render_precision(self.buffer_precision, self.target_precision());
        let Some(clip) = self.effect_target_clip(offset) else { return };
        self.bake_lists(effect, clip);
        // A FINITE effect output is realized at its full declared bounds; clipping it to the target
        // back-map drops the resample apron and shifts a magnifying DrawImage's fixed-point texcoord
        // plane (19ec: the σ-blur apron extends past the back-map, so a 10× LINEAR magnify of the
        // clipped bmp reads a different source extent than native's). Only an INFINITE effect is capped.
        let fb = effect.bounds();
        let finite = fb.l != i32::MIN && fb.t != i32::MIN && fb.r != i32::MAX && fb.b != i32::MAX;
        // A finite effect is realized at its full declared bounds (keeping the resample apron a
        // magnifying DrawImage needs). But a pathologically large finite bounds — a degenerate fuzz
        // input whose coordinates saturate near ±2^30 — cannot be realized as a bitmap: native caps
        // at the device's max texture dimension, so fall back to the (bounded) target clip rather
        // than allocate `w*h*4` for a multi-gigapixel rect (which overflows `usize`).
        const MAX_DIM: u32 = 16384; // D2D feature-level 11 maximum bitmap dimension
        // An infinite output cropped to an image rectangle realizes that rectangle (with the filters' reach), not the
        // target mapped back into image space, which a tiny world scale makes huge (r27 32581: a Scale's infinite output
        // under a 1e-6 transform, ~29500² px; native renders the ~19×31 crop). The clip keeps placing the draw.
        let realize_clip = match IMAGE_CROP.with(|c| c.get()) {
            Some(r) => clip.intersect(Rect {
                l: (r[0].floor() as i32).saturating_sub(3),
                t: (r[1].floor() as i32).saturating_sub(3),
                r: (r[2].ceil() as i32).saturating_add(3),
                b: (r[3].ceil() as i32).saturating_add(3),
            }),
            None => clip,
        };
        let b = if finite && fb.width() <= MAX_DIM && fb.height() <= MAX_DIM { fb } else { realize_clip };
        let tr = self.get_transform();
        // An infinite output under a rotation is resampled into device space: native asks for the
        // back-mapped target padded by the filter reach (`MapOutputRectToInputRects`, 1 for linear), so
        // the corners sample real content, not the realized rect's transparent edge (fuzz r1 587, 722).
        // A cubic filter (CUBIC, HIGH_QUALITY_CUBIC) and MULTI_SAMPLE_LINEAR (taps at the pixel corners) reach a texel
        // past that rounded-out back-map at the target's edges too (sc142: a wrapped Border differed only on the
        // target's edge rows and columns; MULTI_SAMPLE_LINEAR at a whole offset as well).
        let b = if !finite && (tr[1] != 0.0 || tr[2] != 0.0 || matches!(interpolation, 2 | 3 | 5)) {
            Rect { l: b.l.saturating_sub(1), t: b.t.saturating_sub(1), r: b.r.saturating_add(1), b: b.b.saturating_add(1) }
        } else {
            b
        };
        if b.width() == 0 || b.height() == 0 || tr[0] * tr[3] - tr[1] * tr[2] == 0.0 {
            // Nothing to realize (an infinite effect under a non-finite transform back-maps to an empty
            // rect; a singular transform maps the image to no area), but native still composites the
            // empty source: the unbounded modes (SOURCE_IN, SOURCE_COPY, …) clear inside the clip (fuzz
            // 5364). SOURCE_OVER of nothing is a no-op, and so is BOUNDED_SOURCE_COPY, bounded by the
            // empty image (fxdraw-01..03).
            if composite != 0 && composite != 11 {
                let empty = Bitmap { width: 0, height: 0, premul_bgra: Vec::new(), format: DXGI_FORMAT_B8G8R8A8_UNORM, alpha: D2D1_ALPHA_MODE_PREMULTIPLIED, dpi: (96.0, 96.0) };
                self.place_effect_bitmap(empty, [0.0; 2], [0.0; 2], interpolation, composite, finite, false);
            }
            return;
        }
        // Where Direct2D's allocator puts each blur's intermediates for this draw of the rect `b`, given where its
        // output lands (a whole device pixel under a translation-only transform; otherwise a blur drawing straight
        // into the target gets no plan). A draw that shows nothing (above) renders no graph to plan.
        let at = [offset[0] + tr[4], offset[1] + tr[5]];
        let whole = tr[..4] == [1.0, 0.0, 0.0, 1.0] && at.iter().all(|v| v.fract() == 0.0 && v.abs() < 1e7);
        // The image brush a fractional position draws through: its request, and where its intermediate sits in its atlas.
        let mut brush_slot: Option<(Rect, [i32; 2])> = None;
        if !PLAN_HELD.with(|h| h.get()) {
            // Direct2D asks the graph for what the draw shows: its rect within the target and the pushed clips.
            let shown = if whole {
                let c = self.draw_clip_px().map_or(clip, |c| Rect {
                    l: (c[0] as f32 - at[0]).floor() as i32,
                    t: (c[1] as f32 - at[1]).floor() as i32,
                    r: (c[2] as f32 - at[0]).ceil() as i32,
                    b: (c[3] as f32 - at[1]).ceil() as i32,
                });
                b.intersect(clip).intersect(c)
            } else {
                b
            };
            // The deferred image context checks its arena first: past its limit the target renders its queue, which
            // frees the batch's memory and closes the atlas set. Then the DrawImage takes its operation, its render
            // graph and the atlas set's growth.
            let entry = self.effect_arena.measure();
            if entry > arena::FLUSH_BYTES {
                self.effect_arena.flush();
                self.batch.target_flushed();
                plan::batch_close();
            }
            let fractional = !whole && tr[..4] == [1.0, 0.0, 0.0, 1.0] && self.get_dpi() == (96.0, 96.0) && interpolation <= 2;
            let brush = fractional.then(|| self.fractional_request(b, self.effect_realize_origin(offset), interpolation));
            // An image brush asks the graph for its request.
            let asked = brush.unwrap_or(shown);
            let slot = plan::apply(effect, [asked.l, asked.t, asked.r, asked.b], whole.then(|| at.map(|v| v as i32)), composite, brush.map(|r| [r.l, r.t, r.r, r.b]));
            brush_slot = brush.map(|r| (r, slot.unwrap_or_default()));
            self.effect_arena.alloc(arena::DRAW_OPERATION);
            self.image_queued = true;
            let graph = (whole || fractional).then(|| arena::graph(effect, shown, composite, brush)).flatten();
            let atlas = graph.as_ref().and_then(|g| g.atlases.as_ref()).map(plan::add_atlases).unwrap_or_default();
            for &a in graph.iter().flat_map(|g| &g.allocs).chain(&atlas) {
                self.effect_arena.alloc(a);
            }
            if arena::trace() {
                eprintln!("arena entry {entry} graph {:?} atlas {atlas:?}", graph.as_ref().map(|g| &g.allocs));
            }
            // A Composite whose mode combines with this draw's draws each input straight onto the target in
            // Direct2D's order and blends, so they round as Direct2D's do; the graph stays planned as one draw. Through
            // an image brush (a float target's LINEAR draw at a fractional position) the inputs are drawn into the brush's
            // intermediate instead, and that is blitted once ([`Self::blit_float_fractional`]).
            let float_brush = brush.is_some() && composite == 0 && interpolation == 1 && !(crate::context::fast_whole(at[0]) && crate::context::fast_whole(at[1]))
                && self.target_bitmap().is_some_and(|t| crate::float::is_highp(t.desc.format));
            // Under a world that scales, rotates or skews, d2d1 roots the graph in the transform that resamples it onto the
            // target, so the Composite renders into its own intermediate and is not drawn input by input (VM: a skewed
            // MASK_INVERT Composite under SOURCE_COPY, r27 32542).
            let untransformed = self.get_transform()[..4] == [1.0, 0.0, 0.0, 1.0];
            if self.recording.is_none() && !float_brush && untransformed {
                if let Some(draws) = plan::distributed(effect, composite) {
                    if composite == 10 && self.distributed_float_copy(&draws, offset, interpolation) {
                        return;
                    }
                    PLAN_HELD.with(|h| h.set(true));
                    for (input, blend) in draws {
                        match input {
                            Input::Effect(e) => self.draw_image_effect(&e, offset, interpolation, blend),
                            // An effect reads a bitmap input texel for pixel whatever its DPI (VM: a 144-dpi 11×14 bitmap in
                            // a Composite drawn at 96 dpi covers 11×14 pixels, r27 32542), so it draws at the context's DPI.
                            Input::Bitmap(bm) => {
                                let at_context = Bitmap { dpi: self.get_dpi(), ..(*bm).clone() };
                                self.draw_image_bitmap(&at_context, offset, None, interpolation, blend)
                            }
                            // A command list drawn straight onto the target is the list's own draws.
                            Input::List(l) => self.draw_image(&l.list, offset, blend),
                        }
                    }
                    PLAN_HELD.with(|h| h.set(false));
                    return;
                }
            }
            // A graph whose intermediates outgrow the largest texture Direct2D renders a draw with is drawn in
            // tiles, each the whole graph for its part of what the draw shows ([`plan::tiles`]).
            // ponytail: SOURCE_OVER at 96 dpi outside a recording only; other draws stay one draw.
            if whole && composite == 0 && self.recording.is_none() && self.get_dpi() == (96.0, 96.0) {
                let at = at.map(|v| v as i32);
                let tiles = plan::tiles(effect, [shown.l, shown.t, shown.r, shown.b], Some(at), composite);
                if tiles.len() > 1 {
                    PLAN_HELD.with(|h| h.set(true));
                    for t in tiles {
                        plan::apply(effect, t, Some(at), composite, None);
                        let [l, t, r, b] = t.map(|v| v as f32);
                        self.push_axis_aligned_clip([l + offset[0], t + offset[1], r + offset[0], b + offset[1]], 1);
                        self.draw_image_effect(effect, offset, interpolation, composite);
                        self.pop_axis_aligned_clip();
                    }
                    PLAN_HELD.with(|h| h.set(false));
                    return;
                }
            }
        }
        // A float (HDR) target composites the effect's f32 output directly — the 8-bit realise would
        // clamp the extended range the colour effects produce (SOURCE_OVER only; the colour scenes draw
        // 1:1 at the offset). While recording, the f32 blit is kept as a BlitFloat and replayed. [G6]
        if (composite == 0 || self.recording.is_none()) && self.target_bitmap().is_some_and(|t| crate::float::is_highp(t.desc.format)) {
            let (dpx, dpy) = self.get_dpi();
            let (sx, sy) = if self.get_unit_mode() == 0 { (dpx / 96.0, dpy / 96.0) } else { (1.0, 1.0) };
            let w = self.get_transform();
            let (ox, oy) = (sx * (w[0] * offset[0] + w[2] * offset[1] + w[4]), sy * (w[1] * offset[0] + w[3] * offset[1] + w[5]));
            // A LINEAR SOURCE_OVER draw at a fractional position is an image-brush fill: the effect renders its f32
            // output into an atlas over the alpha blit's request, and WARP's float alpha blit samples it there,
            // each column's two texels lerped first, then across (VM probes of an exactly scaled float bitmap at
            // six fractional offsets: every interior pixel), and blends it over the target in float.
            let fractional = !crate::context::fast_whole(ox) || !crate::context::fast_whole(oy);
            if fractional && composite == 0 && interpolation == 1 && self.recording.is_none() && (sx, sy) == (1.0, 1.0)
                && w[..4] == [1.0, 0.0, 0.0, 1.0] && ox.abs() < 1e7 && oy.abs() < 1e7
            {
                // An infinite output fills the whole target the same way (VM, r29 40006: a MIRROR Border's resampled at the
                // offset, where it was placed rounded): its quad is the target mapped back, a pixel past for the lerp.
                let q = if finite { b } else { Rect { l: b.l.saturating_sub(1), t: b.t.saturating_sub(1), r: b.r.saturating_add(1), b: b.b.saturating_add(1) } };
                return self.blit_float_fractional(effect, q, offset, brush_slot);
            }
            // The draw asks for the part of the target it covers, as the BGRA8 path's direct viewport (VM: a box shader
            // reading an intermediate past a float target's edge reads its empty atlas texels there).
            let direct = tr[..4] == [1.0, 0.0, 0.0, 1.0] && self.get_dpi() == (96.0, 96.0);
            let asked = if direct && b.intersect(clip).width() > 0 && b.intersect(clip).height() > 0 { b.intersect(clip) } else { b };
            let prev = DRAW_VIEWPORT.with(|v| v.replace(Some(asked)));
            let px = effect.realize_rect_f32(b, [ox, oy]);
            DRAW_VIEWPORT.with(|v| v.set(prev));
            let (sw, sh) = (b.width() as i32, b.height() as i32);
            let origin = [b.l + ox.round() as i32, b.t + oy.round() as i32];
            if self.recording.is_some() {
                let (state, dpi) = self.rec_env();
                self.record(RecCmd::BlitFloat { state, dpi, px: Rc::new(px), w: sw, h: sh, offset, b_lt: [b.l, b.t] });
                return;
            }
            if composite != 0 {
                // Another composite: the f32 output in a transparent float intermediate, composited over
                // the reach (BOUNDED_SOURCE_COPY with no imageRectangle copies the whole reach too).
                // ponytail: DrawImage(effect) drops its imageRectangle on every target, here included.
                let r = self.reach_rect().map(|v| v as i32);
                let Some(target) = self.target.as_ref() else { return };
                let mut inter = target.clone();
                inter.data.fill(0);
                crate::float::FloatTarget::new(&mut inter).blit(&px, sw, sh, origin[0], origin[1]);
                crate::float::FloatTarget::new(self.target.as_mut().unwrap()).composite(&inter, composite, r);
                return;
            }
            if let Some(buf) = &mut self.target {
                crate::float::FloatTarget::new(buf).blit(&px, sw, sh, origin[0], origin[1]);
            }
            return;
        }
        // A HARD-border GaussianBlur is cropped on its own pixel grid, so at a fractional unit-scale offset Direct2D
        // realizes it there and places the output as a point-wise graph's (the atlas blit below), rather than running
        // its passes at the fractional origin (sc236/sc237: native's NEAREST at a fractional offset is its whole-offset
        // output, LINEAR the resampled one; r27 32897's DEST_IN edge).
        let hard_blur = { let d = effect.0.borrow(); d.kind == Kind::GaussianBlur } && effect.get_u32(2, 0) == 1;
        // A 2D affine transform or Straighten drawn NEAREST, LINEAR or ANISOTROPIC at a fractional unit-scale position
        // renders into an atlas on its own pixel grid and is resampled from there, as a point-wise graph is (DDI: a
        // rotating Affine2D's own pass is the first draw, into the 512 atlas; LINEAR then blits it to the target, and
        // ANISOTROPIC samples that atlas in a second draw).
        let tr = self.get_transform();
        let dev = [offset[0] + tr[4], offset[1] + tr[5]];
        let affine_atlas = matches!(effect.0.borrow().kind, Kind::Affine2D | Kind::Straighten) && matches!(interpolation, 0 | 1 | 4)
            && tr[..4] == [1.0, 0.0, 0.0, 1.0] && (self.get_unit_mode() == 1 || self.get_dpi() == (96.0, 96.0))
            && self.recording.is_none() && self.target_bitmap().is_some_and(|t| !crate::float::is_highp(t.desc.format))
            && !(crate::context::fast_whole(dev[0]) && crate::context::fast_whole(dev[1])) && dev.iter().all(|v| v.abs() < 1e7);
        let origin = if hard_blur || affine_atlas { [0.0; 2] } else { self.effect_realize_origin(offset) };
        let placed = if interpolation == 0
            && composite == 0
            && self.get_dpi() == (96.0, 96.0)
            && (tr[..4] != [1.0, 0.0, 0.0, 1.0] || origin.iter().any(|v| v.fract() != 0.0))
            && effect.0.borrow().kind == Kind::Transform3D
        {
            let (bm, ox, oy) = realize_input(&effect.0.borrow().inputs[0]);
            let input = builtin::transform::Input::new(&bm, ox, oy);
            builtin::transform::projective_bounds(
                &input,
                effect.get_m44(2, IDENTITY_M44).map(|v| v as f64),
                builtin::transform::Interp::from_mode(effect.get_u32(0, 1)),
                builtin::transform::Border::from_mode(effect.get_u32(1, 0)),
            )
        } else {
            None
        };
        // An unbatched composite (not SOURCE_OVER, PLUS or BOUNDED_SOURCE_COPY) drawing CUBIC at a fractional unit-scale
        // position renders the effect into a composite intermediate over the output's device rect grown by ½, which asks
        // the effect for that rect's CUBIC need (VM: r27 32499's DEST_ATOP HQ pass is asked [8, −4, 13, 4], its SOURCE_OVER
        // twin [8, −5, 14, 5]). A HIGH_QUALITY_CUBIC transform's output then lives in its own texture from its slot
        // ([`Effect::hq_slot_origin`]) to the request's far edges, which the CUBIC draw samples (DDI: the HQ pass's viewport
        // and scissor, slots (8, −3) and (6, −3) at offsets (−9.52, 14.25) and (−3.3, 5.7)).
        let cubic = interpolation == 2 || (interpolation == 5 && hq_draws_cubic(self.device_transform(), [b.l as f32 + offset[0], b.t as f32 + offset[1]]));
        let slow = (placed.is_none() && !matches!(composite, 0 | 9 | 11) && cubic && origin == [0.0; 2] && tr[..4] == [1.0, 0.0, 0.0, 1.0]
            && (self.get_unit_mode() == 1 || self.get_dpi() == (96.0, 96.0)) && dev.iter().any(|v| v.fract() != 0.0)
            && self.target_bitmap().is_some_and(|t| !crate::float::is_highp(t.desc.format)))
            .then(|| effect.hq_hard_rect()).flatten();
        // The composite intermediate holds the source only over that device rect: past it the composite sees transparency
        // (DDI: a 30° HQ draw under DEST_ATOP composites its intermediate over exactly that rect, its CUBIC ringing a
        // pixel beyond it absent natively). SOURCE_COPY copies its resample straight onto the target instead (no
        // composite draw), unclipped.
        // ponytail: SOURCE_COPY's own reach is unmodelled (two 30° probes keep a 1 px ringing difference, as before).
        let slow_rect = slow.filter(|_| composite != 10 && composite != 11).map(|o| self.fractional_device_rect(o, dev, 0.5));
        let slow = slow.map(|o| (Rc::as_ptr(&effect.0) as usize, self.fractional_need(o, dev, 0.5, 2.0)));
        let b = match slow {
            Some((_, r)) => {
                let t = effect.hq_slot(r).unwrap_or([r.l, r.t]);
                Rect { l: t[0], t: t[1], r: r.r, b: r.b }
            }
            None => b,
        };
        let bmp = if let Some([l, t, r, bottom]) = placed {
            if !self.ready_to_draw() {
                return;
            }
            let map = |x: f32, y: f32| [(x + offset[0]) * tr[0] + (y + offset[1]) * tr[2] + tr[4], (x + offset[0]) * tr[1] + (y + offset[1]) * tr[3] + tr[5]];
            let p = [map(l as f32, t as f32), map(r as f32, t as f32), map(r as f32, bottom as f32), map(l as f32, bottom as f32)];
            let lo: [f32; 2] = core::array::from_fn(|c| (p.iter().map(|p| p[c]).fold(f32::INFINITY, f32::min) + 0.5).floor());
            let hi: [f32; 2] = core::array::from_fn(|c| (p.iter().map(|p| p[c]).fold(f32::NEG_INFINITY, f32::max) - 0.5).ceil());
            let Some(target) = self.target.as_ref() else { return };
            let lo = [lo[0].max(0.0), lo[1].max(0.0)];
            let hi = [hi[0].min(target.desc.width as f32), hi[1].min(target.desc.height as f32)];
            if lo.iter().chain(&hi).any(|v| !v.is_finite()) || lo[0] >= hi[0] || lo[1] >= hi[1] {
                return;
            }
            let b = Rect { l, t, r, b: bottom }.intersect(b);
            if b.width() == 0 || b.height() == 0 {
                return;
            }
            let det = tr[0] * tr[3] - tr[1] * tr[2];
            if det == 0.0 || !det.is_finite() {
                return;
            }
            let k = 1.0 / det;
            let inv = [tr[3] * k, -tr[1] * k, -tr[2] * k, tr[0] * k];
            let t = [offset[0] * tr[0] + offset[1] * tr[2] + tr[4], offset[0] * tr[1] + offset[1] * tr[3] + tr[5]];
            let inverse_translation = [(tr[2] * t[1] - tr[3] * t[0]) * k, (tr[1] * t[0] - tr[0] * t[1]) * k];
            let back = |x: f32, y: f32| [x * inv[0] + y * inv[2] + inverse_translation[0], x * inv[1] + y * inv[3] + inverse_translation[1]];
            let direct = effect.get_u32(0, 1) == 0 && effect.get_m44(2, IDENTITY_M44) == IDENTITY_M44;
            let q = [back(lo[0] + 0.5, lo[1] + 0.5), back(hi[0] - 0.5, lo[1] + 0.5), back(hi[0] - 0.5, hi[1] - 0.5), back(lo[0] + 0.5, hi[1] - 0.5)];
            // The nearest consumer asks for a 1/32-texel guard around the back-mapped
            // pixel centers. Its atlas starts at the requested integer rectangle.
            let request = Rect {
                l: (q.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min) - 0.03125).floor() as i32,
                t: (q.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min) - 0.03125).floor() as i32,
                r: (q.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max) + 0.03125).ceil() as i32,
                b: (q.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max) + 0.03125).ceil() as i32,
            };
            let request = if direct { b } else { request };
            let Some(width) = request.width().checked_next_power_of_two() else { return };
            let Some(height) = request.height().checked_next_power_of_two() else { return };
            let dims = [width.max(512), height.max(512)];
            let Some(right) = (request.l as i64).checked_add(dims[0] as i64).and_then(|v| i32::try_from(v).ok()) else { return };
            let Some(bottom) = (request.t as i64).checked_add(dims[1] as i64).and_then(|v| i32::try_from(v).ok()) else { return };
            let viewport = Rect { l: request.l, t: request.t, r: right, b: bottom };
            let bitmap = effect.realize_transform(b, [-(request.l as f32), -(request.t as f32)], viewport);
            let start = back(lo[0] + 0.5, lo[1] + 0.5);
            let dx = [inv[0] / dims[0] as f32, inv[1] / dims[1] as f32];
            let dy = [inv[2] / dims[0] as f32, inv[3] / dims[1] as f32];
            let c = [(start[0] - request.l as f32) / dims[0] as f32, (start[1] - request.t as f32) / dims[1] as f32];
            if let Some(target) = &mut self.target {
                let x0 = (lo[0] as i32).max(0);
                let x1 = (hi[0] as i32).min(target.desc.width as i32);
                let y0 = (lo[1] as i32).max(0);
                let y1 = (hi[1] as i32).min(target.desc.height as i32);
                for tile in (y0.div_euclid(8) * 8..y1).step_by(8) {
                    let top = tile.max(y0);
                    let mut row: [f32; 2] = core::array::from_fn(|cidx| dx[cidx] * (x0 as f32 - lo[0]) + dy[cidx] * (top as f32 - lo[1]) + c[cidx]);
                    for y in top..(tile + 8).min(y1) {
                        let mut u: [[f32; 4]; 2] =
                            core::array::from_fn(|cidx| core::array::from_fn(|lane| row[cidx] * dims[cidx] as f32 + lane as f32 * (dx[cidx] * dims[cidx] as f32)));
                        for x in (x0..x1).step_by(4) {
                            for lane in 0..(x1 - x).min(4) as usize {
                                // System32 WARP elides identity NEAREST effects; its affine kernel
                                // evaluates X*dU/dX + (Y*dU/dY + c), without row accumulation.
                                let [tx, ty] = if direct {
                                    let p = [x as f32 + lane as f32 + 0.5, y as f32 + 0.5];
                                    [(p[0] * inv[0] + (p[1] * inv[2] + inverse_translation[0])).floor() as i64 - b.l as i64,
                                     (p[0] * inv[1] + (p[1] * inv[3] + inverse_translation[1])).floor() as i64 - b.t as i64]
                                } else { [u[0][lane].floor() as i64 + request.l as i64 - b.l as i64, u[1][lane].floor() as i64 + request.t as i64 - b.t as i64] };
                                if tx >= 0 && ty >= 0 && tx < bitmap.width as i64 && ty < bitmap.height as i64 {
                                    let src = ((ty as usize) * bitmap.width as usize + tx as usize) * 4;
                                    let dst = y as usize * target.stride + (x as usize + lane) * 4;
                                    let a = 255 - bitmap.premul_bgra[src + 3] as u32;
                                    for ch in 0..4 {
                                        let v = target.data[dst + ch] as u32 * a + 128;
                                        target.data[dst + ch] = (bitmap.premul_bgra[src + ch] as u32 + ((v + (v >> 8)) >> 8)).min(255) as u8;
                                    }
                                }
                            }
                            for cidx in 0..2 {
                                for v in &mut u[cidx] {
                                    *v += 4.0 * (dx[cidx] * dims[cidx] as f32);
                                }
                            }
                        }
                        for cidx in 0..2 {
                            row[cidx] += dy[cidx];
                        }
                    }
                }
            }
            return;
        } else {
            // Under a scaling or rotating transform native renders the effect into a power-of-two atlas
            // (512 minimum) at the requested rect's top-left, and that atlas is its draws' viewport.
            let direct = tr[..4] == [1.0, 0.0, 0.0, 1.0] && self.get_dpi() == (96.0, 96.0);
            let atlas = |l: i32, n: u32| n.checked_next_power_of_two().and_then(|n| i32::try_from(n.max(512)).ok()).and_then(|n| l.checked_add(n));
            let target = self.target_bitmap().map_or([0, 0], |t| [t.desc.width, t.desc.height]);
            let viewport = match (direct, atlas(clip.l, clip.width()), atlas(clip.t, clip.height())) {
                // A rotated LINEAR draw is an image-brush fill in Direct2D's DrawImage: the
                // effect is asked for [`alphablt_request`]'s rect, in an atlas at its top-left.
                (false, ..) if (tr[1] != 0.0 || tr[2] != 0.0) && interpolation == 1 => {
                    let (dpx, dpy) = self.get_dpi();
                    let (sx, sy) = if self.get_unit_mode() == 0 { (dpx / 96.0, dpy / 96.0) } else { (1.0, 1.0) };
                    let m = [sx * tr[0], sy * tr[1], sx * tr[2], sy * tr[3], sx * tr[4], sy * tr[5]];
                    let (_, q) = alphablt_request(&m, [b.l as f32 + offset[0], b.t as f32 + offset[1]], [b.width(), b.height()], composite, target);
                    let e = |v: f32, o: f32| (v - o).round() as i32;
                    Rect { l: e(q[0], offset[0]), t: e(q[1], offset[1]), r: e(q[2], offset[0]), b: e(q[3], offset[1]) }
                }
                (false, Some(r), Some(b)) => Rect { r, b, ..clip },
                // A direct draw asks only for the part of the target the pushed clips and layers leave.
                (true, ..) => self.draw_clip_px().map_or(clip, |c| {
                    let (ox, oy) = (origin[0], origin[1]);
                    let c = Rect { l: (c[0] as f32 - ox).floor() as i32, t: (c[1] as f32 - oy).floor() as i32, r: (c[2] as f32 - ox).ceil() as i32, b: (c[3] as f32 - oy).ceil() as i32 };
                    Rect { l: clip.l.max(c.l), t: clip.t.max(c.t), r: clip.r.min(c.r).max(clip.l.max(c.l)), b: clip.b.min(c.b).max(clip.t.max(c.t)) }
                }),
                _ => clip,
            };
            // Inputs a point-wise effect bakes (`realized`) see the draw's viewport through this.
            // ponytail: one draw-wide region, unmapped through intervening offsets; the memoized bake
            // keeps the first draw's. Map it per node if a moved/redrawn blur needs it.
            let mut draw_origin = direct.then_some([clip.l, clip.t]);
            // At a fractional origin the last pass can't draw straight onto the target: native's DrawImage is an image-brush fill
            // that renders the effect into a 512 atlas at its request's top-left and samples it from there. The fill covers the
            // output's bounds moved by the origin, grown by the sampler's apron (½ for LINEAR, 1½ for CUBIC, as the affine
            // transform's output rect) and rounded to the pixel centres it covers, within the target and the pushed clips. The
            // request is that rect's need (`image_brush_need`: its pixel centres mapped back, padded by ½ + 1/32, 1 or 2 and
            // rounded). VM captures: a sigma-12 blur drawn NEAREST at y 84.75 asks [0,−38,55,11] and one at x 5.5 starts at −6;
            // a sigma-6 blur drawn LINEAR and CUBIC at (10.3, 20.6) starts 1 and 3 rows above the output's top.
            // HIGH_QUALITY_CUBIC instead adds a cubic transform to the effect's graph: it renders the same device rect, and the
            // effect is asked for that rect's need within its output, in an intermediate cleared a 2-texel border around it
            // (native's render-graph log). A batched draw (SOURCE_OVER, PLUS, BOUNDED_SOURCE_COPY) atlases that intermediate
            // with the content 2 texels in; the other composites render a tile graph whose intermediates get their own
            // textures, the content 1 texel in (DDI VBs of a sigma-3 shadow drawn HQ at (20.39, 30.89) under each).
            let viewport = if direct && matches!(interpolation, 0 | 1 | 2 | 5) && origin.iter().any(|v| v.fract() != 0.0) {
                let req = self.fractional_request(b, origin, interpolation);
                let (req, inset) = if interpolation == 5 { (req.intersect(b), if matches!(composite, 0 | 9 | 11) { 2 } else { 1 }) } else { (req, 0) };
                draw_origin = Some([req.l - inset, req.t - inset]);
                req
            } else {
                viewport
            };
            let prev = DRAW_VIEWPORT.with(|v| v.replace(Some(viewport)));
            let prev_origin = DRAW_ORIGIN.with(|o| o.replace(draw_origin));
            let prev_slow = HQ_SLOW_REQ.with(|c| c.replace(slow));
            let bmp = effect.realize_rect_clipped(b, origin, Some(viewport));
            DRAW_VIEWPORT.with(|v| v.set(prev));
            DRAW_ORIGIN.with(|o| o.set(prev_origin));
            HQ_SLOW_REQ.with(|c| c.set(prev_slow));
            bmp
        };
        // Above 8bpc the realisation is float; the 8bpc target stores it (round-to-nearest, clamped).
        let bmp = if crate::float::is_highp(bmp.format) { float_input_to_bgra8(&bmp) } else { bmp };
        let dest0 = [b.l as f32 + offset[0], b.t as f32 + offset[1]];
        let pointwise = !effect.has_filter();
        // A point-wise graph that changes its bitmap, drawn NEAREST or LINEAR at a fractional device position onto an 8bpc
        // target: Direct2D renders it into an atlas at (1, 1) and blits it from there (r22 27378, DDI traces), sampling the
        // power-of-two texture as WARP's blit does ([`atlas_blit`]); the composite then lands on whole pixels, whatever its
        // mode (sc58: a NEAREST edge column under PLUS is the texel, not an antialiased half; r19 24601). An identity graph
        // draws its bitmap itself, which LINEAR samples another way.
        let unit = self.get_unit_mode() == 1 || self.get_dpi() == (96.0, 96.0);
        let dev = [dest0[0] + tr[4], dest0[1] + tr[5]];
        let eight = |f| !crate::float::is_highp(f);
        if (pointwise || hard_blur || affine_atlas) && interpolation <= 1 && finite && unit && tr[..4] == [1.0, 0.0, 0.0, 1.0]
            && eight(bmp.format) && self.target_bitmap().is_some_and(|t| eight(t.desc.format)) && self.recording.is_none()
            && !(crate::context::fast_whole(dev[0]) && crate::context::fast_whole(dev[1])) && dev.iter().all(|v| v.abs() < 1e7)
            && (interpolation == 0 || !effect.passes_through())
        {
            let (at, blit) = atlas_blit(&bmp, dev, interpolation == 0, composite == 9);
            self.draw_image_placed(&blit, [at[0] as f32 - tr[4], at[1] as f32 - tr[5]], None, 0, composite, false);
            return;
        }
        // Under a rotation or skew, a LINEAR graph that leaves its bitmap as it is (an identity ColorMatrix) draws the
        // bitmap itself, through WARP's textured draw (sc110: byte for byte the bitmap's, sc113). A point-wise graph that
        // changes it is realized and alpha-blitted below (sc115: a mixing ColorMatrix, either alpha mode).
        if pointwise && interpolation == 1 && finite && (tr[1] != 0.0 || tr[2] != 0.0) && effect.passes_through()
            && eight(bmp.format) && self.target_bitmap().is_some_and(|t| eight(t.desc.format)) && self.recording.is_none()
        {
            self.draw_image_placed(&bmp, dest0, None, 1, composite, false);
            return;
        }
        // A Border of a bitmap drawn straight into the target samples the bitmap itself, the Border's edge modes as the
        // sampler's address modes: d2d1 collapses the bitmap into the border operation drawing with the target, unless that
        // operation spans more than 25 of the bitmap's tiles over what it draws (then the border cells are drawn instead). VM: a WRAP Border
        // drawn MULTI_SAMPLE_LINEAR at a fractional offset is byte for byte a WRAP bitmap brush's fill (r27 32892). The collapse
        // holds under a rotation too: MULTI_SAMPLE_LINEAR is still that brush fill, and LINEAR is WARP's alpha blit of the
        // bitmap (VM: a MIRROR×CLAMP and a WRAP Border of a 64×1 bitmap under a ×7 rotation, every composite; r27 32107).
        // ponytail: an 8bpc target, an axis-aligned or a rotated NEAREST/LINEAR/MULTI_SAMPLE_LINEAR world and a bitmap input (or a Border chain over one) only; a
        // Border of an effect samples the effect's own intermediate the same way natively, unmodelled. A rotated LINEAR draw
        // under a slow-path composite blits into the atlas, whose height sets the blit's row bands ([`blit_target_height`]).
        let axis = tr[1] == 0.0 && tr[2] == 0.0 && tr[0] > 0.0 && tr[3] > 0.0;
        let rotated = (tr[1] != 0.0 || tr[2] != 0.0) && matches!(interpolation, 0 | 1 | 3) && tr.iter().all(|v| v.is_finite()) && tr[0] * tr[3] - tr[1] * tr[2] != 0.0;
        if self.recording.is_none() && unit && (axis || rotated) && self.target_bitmap().is_some_and(|t| eight(t.desc.format)) {
            // A LINEAR draw under PLUS is WARP's pixel kernel over the bitmap whatever the world or the tile count (VM: a 64×1
            // Border drawn PLUS axis-aligned, rotated, and WRAP×WRAP across 37 tiles, each exact as that kernel).
            let plus = composite == 9 && interpolation == 1;
            if let Some((bm, modes)) = effect.border_of_bitmap().filter(|(bm, m)| plus || border_tiles(realize_clip, bm, *m) <= 25) {
                if (rotated && interpolation <= 1) || plus {
                    return self.blit_border_rotated(&bm, modes, offset, composite, interpolation == 0);
                }
                return self.blit_border_bitmap(&bm, modes, offset, interpolation, composite);
            }
        }
        let before = slow_rect.and(self.target.clone());
        self.place_effect_bitmap(bmp, dest0, offset, interpolation, composite, finite, pointwise);
        if let (Some([l, t, r, b]), Some(before)) = (slow_rect, before) {
            let Some(drawn) = self.target.take() else { return };
            // Past the rect the composite sees a transparent source.
            let mut clear = before.clone();
            clear.data.fill(0);
            self.target = Some(before);
            self.composite_over_target(&clear, if composite == 11 { 10 } else { composite }, None);
            if let Some(out) = self.target.as_mut() {
                let (w, h) = (out.desc.width as i32, out.desc.height as i32);
                for y in t.max(0)..b.min(h) {
                    let i = y as usize * out.stride + l.clamp(0, w) as usize * 4;
                    let j = y as usize * out.stride + r.clamp(0, w) as usize * 4;
                    out.data[i..j].copy_from_slice(&drawn.data[i..j]);
                }
            }
        }
    }

    /// [`Self::draw_image_effect_inner`]'s collapsed Border: its bitmap filled as a bitmap brush with the Border's edge modes
    /// over the target, at the draw's offset (the brush's texel being the effect's pixel), composited with the draw's mode.
    fn blit_border_bitmap(&mut self, bm: &Bitmap, modes: [u32; 2], offset: [f32; 2], interp: u32, composite: u32) {
        let tr = self.get_transform();
        let k = [bm.dpi.0 / 96.0, bm.dpi.1 / 96.0];
        let brush = self.create_bitmap_brush(bm, modes, interp, 1.0, Some([k[0], 0.0, 0.0, k[1], offset[0], offset[1]]));
        let Some(t) = self.target_bitmap() else { return };
        let (w, h) = (t.desc.width as f32, t.desc.height as f32);
        let r = if tr[1] == 0.0 && tr[2] == 0.0 {
            [-tr[4] / tr[0], -tr[5] / tr[3], (w - tr[4]) / tr[0], (h - tr[5]) / tr[3]]
        } else {
            let inv = crate::brush::invert(&tr);
            let c = [[0.0, 0.0], [w, 0.0], [0.0, h], [w, h]].map(|[x, y]| [inv[0] * x + inv[2] * y + inv[4], inv[1] * x + inv[3] * y + inv[5]]);
            let (lo, hi) = (|k: usize| c.iter().map(|p| p[k]).fold(f32::INFINITY, f32::min), |k: usize| c.iter().map(|p| p[k]).fold(f32::NEG_INFINITY, f32::max));
            [lo(0) - 1.0, lo(1) - 1.0, hi(0) + 1.0, hi(1) + 1.0]
        };
        if composite == 0 {
            return self.fill_rectangle_brush(r, &brush);
        }
        let Some(saved) = self.target.take() else { return };
        let mut inter = saved.clone();
        inter.data.fill(0);
        self.target = Some(inter);
        let blend = self.get_primitive_blend();
        self.state.blend = 0;
        self.fill_rectangle_brush(r, &brush);
        self.state.blend = blend;
        let inter = self.target.replace(saved).unwrap();
        self.composite_over_target(&inter, if composite == 11 { 10 } else { composite }, None);
    }

    /// [`Self::draw_image_effect_inner`]'s collapsed Border drawn NEAREST or LINEAR under a rotation or skew, or LINEAR under PLUS:
    /// WARP's alpha blit samples the bitmap itself over the target, the Border's edge modes as its address modes and the plane normalized
    /// by the bitmap's size (the JIT kernel of r27 32107's draw masks a mirrored power-of-two column address and
    /// clamps the row). Its texel origin is the draw's offset under the device matrix.
    fn blit_border_rotated(&mut self, bm: &Bitmap, modes: [u32; 2], offset: [f32; 2], composite: u32, nearest: bool) {
        let tr = self.get_transform();
        let (dpx, dpy) = self.get_dpi();
        let (sx, sy) = if self.get_unit_mode() == 0 { (dpx / 96.0, dpy / 96.0) } else { (1.0, 1.0) };
        let m = [sx * tr[0], sy * tr[1], sx * tr[2], sy * tr[3], sx * tr[4], sy * tr[5]];
        let w = [offset[0] * m[0] + offset[1] * m[2] + m[4], offset[0] * m[1] + offset[1] * m[3] + m[5]];
        let ts = crate::brush::invert(&[m[0], m[1], m[2], m[3], w[0], w[1]]);
        let p = [w[0], w[1]];
        let texmap = self.image_texmap_pub(bm, offset, p);
        let Some(mut saved) = self.take_target() else { return };
        let (tw, th) = (saved.desc.width, saved.desc.height);
        let ex = modes.map(|v| match v { 1 => crate::sampling::Extend::Wrap, 2 => crate::sampling::Extend::Mirror, _ => crate::sampling::Extend::Clamp });
        let sampled = bm.as_sampled_bgra8();
        let tex = crate::sampling::Texture { data: &sampled.premul_bgra, width: bm.width, height: bm.height };
        if composite == 9 {
            // Under PLUS the draw is not the alpha blit but WARP's pixel kernel over the target: the bitmap's texcoord plane
            // fitted over a unit triangle, the Q8 address `(u·n − ½)·256` floored, the 15-bit bilinear truncated by `>> 7`,
            // and the sample added to the target saturating (the VM JIT of a rotated PLUS Border; r27 32107 probes).
            let fit = crate::warp::plane::warp_fit_affine(&texmap);
            let n = [bm.width as f32, bm.height as f32];
            let q = |u: f32, n: f32| {
                let a = ((u * n) + -0.5) * 256.0;
                let q = if a.is_nan() { 0 } else if a.abs() < 2147483648.0 { a.floor() as i32 } else { i32::MIN };
                (q >> 8, q & 0xff)
            };
            let st = saved.stride;
            for y in 0..th as usize {
                for x in 0..tw as usize {
                    let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                    let u = fit[0] * fx + (fit[2] * fy + fit[4]);
                    let v = fit[1] * fx + (fit[3] * fy + fit[5]);
                    let c = tex.bilinear_v128_all(ex, q(u, n[0]), q(v, n[1])).map(|c| c >> 7);
                    let i = y * st + x * 4;
                    for k in 0..4 {
                        saved.data[i + k] = (saved.data[i + k] as i32 + c[k]).clamp(0, 255) as u8;
                    }
                }
            }
            self.set_target_bitmap(saved);
            return;
        }
        let rect = [0, 0, tw as usize, th as usize];
        let mut px = vec![0u8; tw as usize * th as usize * 4];
        crate::sampling::alphablt_float_iter_ex(&ts, [bm.width, bm.height], [bm.width, bm.height], rect, rect, ex, nearest, blit_target_height(composite, th), |x, y, qx, qy| {
            let c = tex.sample_alphablt_q8(ex, qx, qy, nearest);
            let i = (y * tw as usize + x) * 4;
            px[i..i + 4].copy_from_slice(&c.map(|v| v as u8));
        });
        self.composite_device_blit(saved, &px, rect, 1, composite);
    }

    /// Composite a rotated alpha blit's device pixels `dev` (target-sized, written over `rect`) onto the target `saved`
    /// was taken from, with `composite`.
    fn composite_device_blit(&mut self, saved: d2d_types::contracts::PixelBuffer, dev: &[u8], rect: [usize; 4], interp: u32, composite: u32) {
        let (tw, th, st) = (saved.desc.width, saved.desc.height, saved.stride);
        if composite == 0 && interp == 1 {
            // The image-brush fill blends source-over as `s + d·(255 − sα)/255` rounded exactly (not the
            // blend unit's high-byte form; fxrot-15-clip-bottom/-bottom2).
            let mut saved = saved;
            let [x0, y0, x1, y1] = rect;
            for y in y0..y1 {
                for x in x0..x1 {
                    let (i, j) = (y * st + x * 4, (y * tw as usize + x) * 4);
                    let ia = 255 - dev[j + 3] as i32;
                    for k in 0..4 {
                        saved.data[i + k] = (dev[j + k] as i32 + crate::bitmap::unorm_mul(saved.data[i + k] as i32, ia)).min(255) as u8;
                    }
                }
            }
            self.set_target_bitmap(saved);
            return;
        }
        let mut inter = saved.clone();
        // BOUNDED_SOURCE_COPY replaces only what the alpha blit writes (its rect); the other modes see
        // the output transparent everywhere else.
        let [x0, y0, x1, y1] = if composite == 11 && interp == 1 { rect } else { [0, 0, tw as usize, th as usize] };
        for y in y0..y1 {
            inter.data[y * st + x0 * 4..y * st + x1 * 4].copy_from_slice(&dev[(y * tw as usize + x0) * 4..(y * tw as usize + x1) * 4]);
        }
        self.set_target_bitmap(saved);
        self.composite_over_target(&inter, if composite == 11 { 10 } else { composite }, None);
    }

    /// Record the realized effect output while recording a command list, else blit it now.
    #[allow(clippy::too_many_arguments)]
    fn place_effect_bitmap(&mut self, bmp: Bitmap, dest0: [f32; 2], offset: [f32; 2], interp: u32, composite: u32, finite: bool, pointwise: bool) {
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            self.record(RecCmd::DrawEffect { state, dpi, bitmap: Rc::new(bmp), dest0, offset, interp, composite, finite, pointwise });
            return;
        }
        self.blit_effect_bitmap(&bmp, dest0, offset, interp, composite, finite, pointwise);
    }

    /// A SOURCE_COPY-distributed Composite of a Flood and a bitmap drawn DEST_OVER, on an 8-bit target: native renders it
    /// in a float intermediate and stores it once. The Flood enters with its negative channels raised to 0 but its
    /// over-range ones kept, so an R of 2 stays saturated after the blend. The bitmap's CUBIC sample keeps its ringing
    /// below 0 and its LINEAR sample is the floored-Q8 bilinear in float, both straight into the DEST_OVER sum (r27 32893
    /// op 9: VM FP32 probes sc227 read native's sample; sc215/sc223 Δ8 → 1).
    fn distributed_float_copy(&mut self, draws: &[(Input, u32)], offset: [f32; 2], interp: u32) -> bool {
        let tr = self.get_transform();
        let unit = tr[..4] == [1.0, 0.0, 0.0, 1.0] && (self.get_unit_mode() == 1 || self.get_dpi() == (96.0, 96.0));
        let Some(t) = self.target_bitmap() else { return false };
        if !unit || crate::float::is_highp(t.desc.format) || draws.len() != 2 || interp == 0 { return false; }
        let (Input::Effect(e0), Input::Bitmap(bm)) = (&draws[0].0, &draws[1].0) else { return false };
        let flood = { let d = e0.0.borrow(); d.kind == Kind::Flood && d.inputs.is_empty() };
        if draws[1].1 != 1 || !flood || crate::float::is_highp(bm.format) { return false; }
        let (w, h) = (t.desc.width as i32, t.desc.height as i32);
        let mut acc = vec![e0.get_v4(0, [0.0, 0.0, 0.0, 1.0]).map(|c| c.max(0.0)); (w * h) as usize];
        let p = [offset[0] + tr[4], offset[1] + tr[5]];
        let texmap = self.image_texmap_pub(bm, offset, p);
        let fit = crate::warp::plane::warp_fit_affine(&texmap);
        let tex = crate::sampling::Texture { data: &bm.premul_bgra, width: bm.width, height: bm.height };
        let (bw, bh) = (bm.width as f32, bm.height as f32);
        let ex = [crate::sampling::Extend::Transparent; 2];
        let r = [(p[0] - 3.0).floor() as i32, (p[1] - 3.0).floor() as i32, (p[0] + bw + 3.0).ceil() as i32, (p[1] + bh + 3.0).ceil() as i32];
        for y in r[1].max(0)..r[3].min(h) { for x in r[0].max(0)..r[2].min(w) {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let u = [fit[0] * fx + (fit[2] * fy + fit[4]), fit[1] * fx + (fit[3] * fy + fit[5])];
            let [b, g, rr, a] = if interp == 1 {
                let q = |t: f32, n: f32| { let q = ((t * n - 0.5) * 256.0).floor() as i32; (q >> 8, q & 255) };
                let (qx, qy) = (q(u[0], bw), q(u[1], bh));
                core::array::from_fn(|ch| crate::sampling::norm_v128(tex.bilinear_v128(ex, ch, qx, qy)))
            } else {
                tex.sample_hqcubic_unclamped_in(ex, u, [u[0] * bw, u[1] * bh], [bm.width, bm.height])
            };
            let d = &mut acc[(y * w + x) as usize];
            let s = [rr, g, b, a];
            let ia = 1.0 - d[3];
            for k in 0..4 { d[k] += s[k] * ia; }
        }}
        let Some(buf) = self.target.as_mut() else { return false };
        let st = buf.stride;
        for y in 0..h { for x in 0..w {
            let c = acc[(y * w + x) as usize];
            let i = y as usize * st + x as usize * 4;
            let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round_ties_even() as u8;
            buf.data[i..i + 4].copy_from_slice(&[q(c[2]), q(c[1]), q(c[0]), q(c[3])]);
        }}
        true
    }

    /// [`Self::blit_effect_bitmap`]'s ANISOTROPIC atlas: `bmp` (placed at DIP `dest0`, drawn at `offset`) copied into a
    /// transparent power-of-two texture (at least 512) at the request's top-left, and where to draw that texture.
    fn aniso_effect_atlas(&self, bmp: &Bitmap, dest0: [f32; 2], offset: [f32; 2]) -> Option<(Bitmap, [f32; 2])> {
        let t = self.target_bitmap()?;
        let (tw, th) = (t.desc.width as i32, t.desc.height as i32);
        let tr = self.get_transform();
        let o = [offset[0] + tr[4], offset[1] + tr[5]];
        let b = [(dest0[0] - offset[0]) as i32, (dest0[1] - offset[1]) as i32];
        let (bw, bh) = (bmp.width as i32, bmp.height as i32);
        let c = self.draw_clip_px().map_or([0, 0, tw, th], |c| [c[0].max(0), c[1].max(0), c[2].min(tw), c[3].min(th)]);
        let near = |v: i32, o: f32| ((v as f32 + o - 0.5) + 0.5).floor() as i32;
        let far = |v: i32, o: f32| ((v as f32 + o + 0.5) - 0.5).ceil() as i32;
        let d = [c[0].max(near(b[0], o[0])), c[1].max(near(b[1], o[1])), c[2].min(far(b[0] + bw, o[0])), c[3].min(far(b[1] + bh, o[1]))];
        let lo = |v: i32, o: f32| (((v as f32 + 0.5 - o) - 1.0) + 0.5).floor() as i32;
        let hi = |v: i32, o: f32| (((v as f32 - 0.5 - o) + 1.0) - 0.5).ceil() as i32;
        if d[2] <= d[0] || d[3] <= d[1] { return None; }
        // The effect is realized over the request only: the part of `bmp` inside it, at its place in the atlas.
        let req = [lo(d[0], o[0]), lo(d[1], o[1]), hi(d[2], o[0]), hi(d[3], o[1])];
        let (x0, y0, x1, y1) = (req[0].max(b[0]), req[1].max(b[1]), req[2].min(b[0] + bw), req[3].min(b[1] + bh));
        if x1 <= x0 || y1 <= y0 { return None; }
        let n = [(req[2] - req[0]) as u32, (req[3] - req[1]) as u32].map(|v| v.next_power_of_two().max(512));
        if n[0] > 4096 || n[1] > 4096 { return None; }
        let mut px = vec![0u8; (n[0] * n[1] * 4) as usize];
        let row = (x1 - x0) as usize * 4;
        for y in y0..y1 {
            let s = ((y - b[1]) as usize * bw as usize + (x0 - b[0]) as usize) * 4;
            let dst = ((y - req[1]) as usize * n[0] as usize + (x0 - req[0]) as usize) * 4;
            px[dst..dst + row].copy_from_slice(&bmp.premul_bgra[s..s + row]);
        }
        let atlas = Bitmap { width: n[0], height: n[1], premul_bgra: px, format: bmp.format, alpha: bmp.alpha, dpi: bmp.dpi };
        Some((atlas, [req[0] as f32 + offset[0], req[1] as f32 + offset[1]]))
    }

    /// A 2D affine transform whose matrix, in device-pixel space, moves by whole pixels: Direct2D's 3D transform stage
    /// builds an offset graph for it, moving its input by the rounded translation without resampling (any interpolation
    /// but MULTI_SAMPLE_LINEAR; HIGH_QUALITY_CUBIC takes a Scale graph instead). "Whole" is within 1/512 above the
    /// truncated integer, a half rounding up. Drawn, it is its input drawn at the summed offset (VM probes: native
    /// renders the two byte for byte alike under NEAREST, LINEAR, CUBIC, HIGH_QUALITY_CUBIC and ANISOTROPIC draws; at
    /// 120 dpi a translation of 8 DIPs is 10 px but -2 is -2.5 px, so that one resamples). Returns the input and the
    /// offset, in DIPs, to draw it at. A bitmap input is only passed through at the context's DPI: an effect takes it at
    /// its pixel size, a bitmap draw at its own DPI.
    fn whole_pixel_offset(&self, effect: &Effect, offset: [f32; 2]) -> Option<(Input, [f32; 2])> {
        let d = effect.0.borrow();
        let m = match d.props.get(&2) { Some(Prop::M32(m)) if d.kind == Kind::Affine2D => *m, _ => return None };
        let (dpx, dpy) = self.get_dpi();
        // This draw's DIP-to-pixel scale; inside the device-pixel-space redraw the DPI reads 96 and `DIP_SCALE` holds it.
        let s = if self.get_unit_mode() == 1 { 1.0 } else { dpx / 96.0 };
        let k = s * DIP_SCALE.with(|v| v.get());
        let mode = effect.get_u32(0, 1);
        if matches!(mode, 3 | 5) || m[..4] != [1.0, 0.0, 0.0, 1.0] || dpx != dpy || IMAGE_CROP.with(|c| c.get()).is_some() || d.inputs.len() != 1 {
            return None;
        }
        // Native's pixel matrix is S(96/dpi)·M·S(dpi/96), the translation scaled by dpi/96.
        let t = [m[4] * k, m[5] * k];
        let whole = |v: f32| {
            let mut n = v as i32;
            if n as f32 - v == -0.5 { n += 1; }
            ((n as f32 - v).abs() < 1.0 / 512.0).then_some(n)
        };
        let round_sat = |v: f32| { let r = v.round_ties_even(); (if r - v == -0.5 { r + 1.0 } else { r }) as i32 };
        let (_, _) = (whole(t[0])?, whole(t[1])?);
        let dt = [round_sat(t[0]) as f32 / s, round_sat(t[1]) as f32 / s];
        let input = d.inputs[0].clone();
        if let Input::Bitmap(b) = &input {
            if self.get_unit_mode() == 0 && b.dpi != (dpx, dpy) { return None; }
        }
        Some((input, [offset[0] + dt[0], offset[1] + dt[1]]))
    }

    /// The device origin passed to `realize_rect_at`: for an identity world transform at 96 DPI the
    /// output is realized at the draw offset (the translation case); any other transform realizes at
    /// origin 0 (the resample families read the device transform themselves). Shared by the direct draw
    /// and the command-list recording so the realized pixels are identical.
    /// A LINEAR SOURCE_OVER DrawImage of `effect` (output rect `b`) at a fractional unit-scale position onto a float
    /// target, as Direct2D draws it: the effect's f32 output over [`alphablt_request`]'s rect in a power-of-two atlas
    /// (at least 512), sampled by WARP's float alpha-blit iterator and blended source-over in float.
    /// `brush` is the image brush's request and its intermediate's atlas place as the planner laid them out.
    fn blit_float_fractional(&mut self, effect: &Effect, b: Rect, offset: [f32; 2], brush: Option<(Rect, [i32; 2])>) {
        let Some(target) = self.target.as_ref() else { return };
        let (tw, th) = (target.desc.width, target.desc.height);
        let w = self.get_transform();
        let m = [1.0, 0.0, 0.0, 1.0, w[4], w[5]];
        let dest0 = [b.l as f32 + offset[0], b.t as f32 + offset[1]];
        let (rect, _) = alphablt_request(&m, dest0, [b.width(), b.height()], 0, [tw, th]);
        let (req, slot) = brush.unwrap_or_else(|| (self.fractional_request(b, self.effect_realize_origin(offset), 1), [0, 0]));
        if req.width() == 0 || req.height() == 0 {
            return;
        }
        let ae = [req.l as f32, req.t as f32];
        let hard_blur = { let d = effect.0.borrow(); d.kind == Kind::GaussianBlur } && effect.get_u32(2, 0) == 1;
        let origin = if hard_blur { [0.0; 2] } else { self.effect_realize_origin(offset) };
        let prev = DRAW_VIEWPORT.with(|v| v.replace(Some(req)));
        let px = effect.realize_rect_f32(req, origin);
        DRAW_VIEWPORT.with(|v| v.set(prev));
        // Device to atlas texel: the request's top-left draws at the world position of its effect-space origin, from
        // its place in the atlas.
        let at = [offset[0] + m[4], offset[1] + m[5]];
        let ts = crate::brush::invert(&[1.0, 0.0, 0.0, 1.0, (ae[0] + at[0]) - slot[0] as f32, (ae[1] + at[1]) - slot[1] as f32]);
        let (rw, rh) = (req.width() as i32, req.height() as i32);
        let size = [(rw + slot[0]) as u32, (rh + slot[1]) as u32].map(|v| v.next_power_of_two().max(512));
        let fetch = |x: i32, y: i32| {
            let (x, y) = (x - slot[0], y - slot[1]);
            if x < 0 || y < 0 || x >= rw || y >= rh { [0.0; 4] } else { px[(y * rw + x) as usize] }
        };
        // An 8bpc effect output sits in a UNORM8 atlas and the blit lerps its bytes in fixed point to value·128 (VM, r29:
        // a box shader over a bitmap drawn LINEAR at a fractional offset into a float target lands on 1/(255·128) steps).
        let eight = prec_rank(effect.out_precision()) == 1;
        let byte = |c: [f32; 4]| c.map(|v| (v * 255.0).round_ties_even().clamp(0.0, 255.0) as i32);
        let Some(buf) = self.target.as_mut() else { return };
        let target_h = buf.desc.height;
        let mut t = crate::float::FloatTarget::new(buf);
        crate::sampling::alphablt_float_iter_floor(&ts, size, rect, target_h, |x, y, qx, qy| {
            let c = if eight {
                let (x0, wx, y0, wy) = (qx >> 8, qx & 255, qy >> 8, qy & 255);
                let (c00, c10, c01, c11) = (byte(fetch(x0, y0)), byte(fetch(x0 + 1, y0)), byte(fetch(x0, y0 + 1)), byte(fetch(x0 + 1, y0 + 1)));
                // Each column's two texels to value·128 first (truncated), then across with a 16-bit high multiply.
                core::array::from_fn(|k| {
                    let l = ((c01[k] - c00[k]) * wy + (c00[k] << 8)) >> 1;
                    let r = ((c11[k] - c10[k]) * wy + (c10[k] << 8)) >> 1;
                    crate::sampling::norm_v128(l + ((((wx as i64) << 8) * (r - l) as i64) >> 16) as i32)
                })
            } else {
                crate::sampling::bilinear_f32((qx >> 8, qx & 255), (qy >> 8, qy & 255), fetch)
            };
            t.blend_px(x, y, c, 64, crate::pixels::Blend::SourceOver);
        });
    }

    /// What the image brush of a DrawImage at fractional device position `origin` asks of the effect's output `b`
    /// (NEAREST, LINEAR or CUBIC `interpolation`): the device rect the fill covers, the output moved by the origin and
    /// grown by the sampler's apron, rounded to the pixel centres it covers within the target and the pushed clips,
    /// then that rect's pixel centres mapped back and padded by the sampler's reach.
    fn fractional_request(&self, b: Rect, origin: [f32; 2], interpolation: u32) -> Rect {
        let (apron, pad) = [(0.0, 0.53125), (0.5, 1.0), (1.5, 2.0)][interpolation.min(2) as usize];
        self.fractional_need(b, origin, apron, pad)
    }

    /// The output `b` moved by `origin`, grown by `apron` and rounded to the pixel centres it covers within the target and
    /// the pushed clips.
    fn fractional_device_rect(&self, b: Rect, origin: [f32; 2], apron: f32) -> [i32; 4] {
        let target = self.target_bitmap().map_or([0, 0], |t| [t.desc.width, t.desc.height]);
        let (tw, th) = (target[0] as i32, target[1] as i32);
        let c = self.draw_clip_px().map_or([0, 0, tw, th], |c| [c[0].max(0), c[1].max(0), c[2].min(tw), c[3].min(th)]);
        let near = |v: i32, o: f32| ((v as f32 + o - apron) + 0.5).floor() as i32;
        let far = |v: i32, o: f32| ((v as f32 + o + apron) - 0.5).ceil() as i32;
        [c[0].max(near(b.l, origin[0])), c[1].max(near(b.t, origin[1])), c[2].min(far(b.r, origin[0])), c[3].min(far(b.b, origin[1]))]
    }

    /// [`Self::fractional_device_rect`]'s need for a sampler of `pad` texels.
    fn fractional_need(&self, b: Rect, origin: [f32; 2], apron: f32, pad: f32) -> Rect {
        let d = self.fractional_device_rect(b, origin, apron);
        let lo = |v: i32, o: f32| (((v as f32 + 0.5 - o) - pad) + 0.5).floor() as i32;
        let hi = |v: i32, o: f32| (((v as f32 - 0.5 - o) + pad) - 0.5).ceil() as i32;
        Rect { l: lo(d[0], origin[0]), t: lo(d[1], origin[1]), r: hi(d[2], origin[0]), b: hi(d[3], origin[1]) }
    }

    fn effect_realize_origin(&self, offset: [f32; 2]) -> [f32; 2] {
        let tr = self.get_transform();
        if tr[..4] == [1.0, 0.0, 0.0, 1.0] && self.get_dpi() == (96.0, 96.0) {
            [offset[0] + tr[4], offset[1] + tr[5]]
        } else {
            [0.0; 2]
        }
    }

    /// Place an already-realized 8bpc effect output `bmp`, whose top-left is at DIP `dest0` (drawn at the
    /// DrawImage `offset`), under the **current** device transform with `interp`/`composite` (`finite`
    /// enables the magnify-transparent ramp). The placement shared by a direct `DrawImage(effect)` and its
    /// command-list replay — the replay sets the baked transform (shifted by the replay offset) before
    /// calling this, so the two are byte-identical. [G6]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn blit_effect_bitmap(&mut self, bmp: &Bitmap, dest0: [f32; 2], offset: [f32; 2], interp: u32, composite: u32, finite: bool, pointwise: bool) {
        // HIGH_QUALITY_CUBIC is Direct2D's 2D affine transform effect: it builds a Scale node only for a pure positive scale
        // at the origin or for an axis it must pre-scale, and draws any other placement through its matrix transform in
        // CUBIC (VM: r27 32654's Blend drawn HIGH_QUALITY_CUBIC and CUBIC at (41.02, 28.28), byte-identical under
        // MASK_INVERT and SOURCE_COPY).
        let interp = if interp == 5 && hq_draws_cubic(self.device_transform(), dest0) { 2 } else { interp };
        let tr = self.get_transform();
        if (tr[1] != 0.0 || tr[2] != 0.0) && bmp.width > 0 && bmp.height > 0 {
            // A rotating/skewing transform: native applies it inside the effect pipeline, resampling the
            // output with its affine-transform sampler into device space (fxrot-01..06; fuzz 7234), then
            // composites that as a unit.
            let (dpx, dpy) = self.get_dpi();
            let (sx, sy) = if self.get_unit_mode() == 0 { (dpx / 96.0, dpy / 96.0) } else { (1.0, 1.0) };
            let m = [sx * tr[0], sy * tr[1], sx * tr[2], sy * tr[3], sx * tr[4], sy * tr[5]];
            let mat = [m[0], m[1], m[2], m[3], m[0] * dest0[0] + m[2] * dest0[1] + m[4], m[1] * dest0[0] + m[3] * dest0[1] + m[5]];
            let Some(saved) = self.take_target() else { return };
            let (tw, th) = (saved.desc.width, saved.desc.height);
            // A CUBIC resample reads the output from the 512 atlas it was rendered into, whose size normalizes the texcoords
            // (DDI: r27 32542's Composite inputs are copied into a 512 atlas slot that the skewed CUBIC draw then samples).
            // ponytail: CUBIC only (ANISOTROPIC's mip path takes no atlas input: r27 32223's Border drawn so is unchanged);
            // the slot's place in the atlas set is not modelled (the content at the atlas origin), which only matters at a
            // Q8 tie.
            let input = builtin::transform::Input::new(bmp, 0, 0);
            let input = if interp == 2 { input.in_atlas() } else { input };
            // LINEAR is WARP's float alpha-blit iterator, as a skewed bitmap-brush fill (fxrot-03 traces:
            // no shader draw), over [`alphablt_request`]'s rect from the atlas at its request.
            let (rect, req) = alphablt_request(&m, dest0, [bmp.width, bmp.height], composite, [tw, th]);
            let origin = [req[0], req[1]];
            let dev = if interp == 1 {
                let tex = crate::sampling::Texture { data: &bmp.premul_bgra, width: bmp.width, height: bmp.height };
                // Direct2D's DrawImage folds the offset into the world translation,
                // then places the atlas at its effect-space origin.
                let w = [offset[0] * m[0] + offset[1] * m[2] + m[4], offset[0] * m[1] + offset[1] * m[3] + m[5]];
                let ae = [origin[0] - offset[0], origin[1] - offset[1]].map(f32::round);
                let ts = crate::brush::invert(&[m[0], m[1], m[2], m[3], (ae[0] * m[0] + ae[1] * m[2]) + w[0], (ae[0] * m[1] + ae[1] * m[3]) + w[1]]);
                let shift = [dest0[0] - origin[0], dest0[1] - origin[1]].map(|v| v.round() as i32 * 256);
                let size = [bmp.width + 2, bmp.height + 2].map(|v| v.next_power_of_two().max(512));
                let mut px = vec![0u8; tw as usize * th as usize * 4];
                crate::sampling::alphablt_float_iter(&ts, size, rect, blit_target_height(composite, th), |x, y, qx, qy| {
                    let c = tex.sample_alphablt_q8([crate::sampling::Extend::Transparent; 2], qx - shift[0], qy - shift[1], false);
                    let i = (y * tw as usize + x) * 4;
                    px[i..i + 4].copy_from_slice(&c.map(|v| v as u8));
                });
                px
            } else {
                let eval = builtin::transform::affine_at(&input, mat, builtin::transform::Interp::from_mode(interp), builtin::transform::Border::Soft, [0.0; 2], false);
                builtin::transform::to_bitmap(tw, th, 0.0, 0.0, eval).premul_bgra
            };
            return self.composite_device_blit(saved, &dev, rect, interp, composite);
        }
        // At unit device scale a finite LINEAR or CUBIC point-wise output is native's DrawImage of the
        // realized bitmap (r5 5077: ColorMatrix(bitmap) at a fractional offset renders as DrawImage(bitmap)).
        // CUBIC addresses it in its atlas texture (r27 32199: a Crop at a fractional offset). MULTI_SAMPLE_LINEAR is
        // that DrawImage too, its taps reading transparent past the output (r27 32807: a Crop drawn SOURCE_COPY softens
        // its edge pixels). So is HIGH_QUALITY_CUBIC (r27 32654: a Blend output at a fractional offset, Δ24 as a
        // DrawBitmap, within the HQ DrawImage's own 2 this way). A filter (blur) graph places its fractional origin
        // inside its own passes instead, except that a CUBIC or HIGH_QUALITY_CUBIC draw still samples the atlas it rendered
        // into (VM probes of sigma-3 to 6 blurs and shadows drawn so at fractional offsets, under every composite and a
        // fractional world translation).
        let unit = tr[..4] == [1.0, 0.0, 0.0, 1.0] && (self.get_unit_mode() == 1 || self.get_dpi() == (96.0, 96.0));
        let frac = [dest0[0] + tr[4], dest0[1] + tr[5]].iter().any(|v| v.fract() != 0.0);
        // ANISOTROPIC within 1/32 px of a whole pixel is DrawImage's whole-pixel fast path too, the output drawn 1:1 at the
        // pixel it rounds to rather than resampled from an atlas (r27 32888: an ANISOTROPIC DEST_IN at device (0.0002,
        // 0.0002) leaves its opaque pixel's destination as it is; the atlas's floored Q8 address took 1/256 of each neighbour).
        let whole = crate::context::fast_whole(dest0[0] + tr[4]) && crate::context::fast_whole(dest0[1] + tr[5]);
        if (matches!(interp, 1 | 2 | 3 | 5) || (interp == 4 && whole)) && finite && unit && (pointwise || (matches!(interp, 2 | 5) && frac)) {
            self.effect_atlas = interp == 2 || (interp == 5 && !pointwise);
            self.draw_image_placed(bmp, dest0, None, interp, composite, false);
            self.effect_atlas = false;
            return;
        }
        // ANISOTROPIC at a fractional position samples the output from its atlas: a power-of-two (at least 512) texture
        // holding it at the image brush's request, the need of the device rect the output covers grown by ½ (DDI: a rotating
        // Affine2D's atlas starts at the need's top-left; the ANISOTROPIC draw reads that 512 texture).
        let atlased = (interp == 4 && finite && unit && pointwise && frac).then(|| self.aniso_effect_atlas(bmp, dest0, offset)).flatten();
        let (bmp, dest0) = match &atlased { Some((a, d)) => (a, *d), None => (bmp, dest0) };
        let (w, h) = (bmp.width as f32, bmp.height as f32);
        let dest = [dest0[0], dest0[1], dest0[0] + w, dest0[1] + h];
        if composite == 0 {
            // Magnifying LINEAR DrawImage of a FINITE effect output: native's sampler reads TRANSPARENT
            // past the realized bitmap, so the content ramps to 0 a half-texel beyond the output extent
            // (wider than the bounds-transformed dest). Extend dest+source by ±0.5 texel (scale
            // preserved) and blit with a transparent border so that ramp renders — matched vs the 19ec
            // oracle. Only finite+magnifying+LINEAR: an INFINITE effect
            // extends its content past the clip, so Clamp is correct there, not transparent.
            let tr = self.get_transform();
            let sc = (tr[0] * tr[0] + tr[1] * tr[1]).sqrt();
            if interp == 1 && sc > 1.0 && finite {
                let (dpx, dpy) = self.get_dpi();
                let (sx, sy) = if self.get_unit_mode() == 0 { (dpx / 96.0, dpy / 96.0) } else { (1.0, 1.0) };
                let map = |x: f32, y: f32| [sx * (tr[0] * x + tr[2] * y + tr[4]), sy * (tr[1] * x + tr[3] * y + tr[5])];
                let e = 0.5f32;
                let (a, c) = (map(dest[0] - e, dest[1] - e), map(dest[2] + e, dest[3] + e));
                let dev = [a[0].min(c[0]), a[1].min(c[1]), a[0].max(c[0]), a[1].max(c[1])];
                if let Some(buf) = &mut self.target {
                    crate::bitmap::draw_bitmap_transparent(buf, bmp, dev, [-e, -e, w + e, h + e]);
                }
            } else {
                self.draw_bitmap(bmp, dest, [0.0, 0.0, w, h], 1.0, interp);
            }
            return;
        }
        // Non-SOURCE_OVER: realize the effect output into a transparent target-sized intermediate at
        // its device rect, then composite it over the target as a unit (native's DrawImage slow path,
        // the same `composite_over_target` the command-list/bitmap paths use). The device rect is the
        // DIP `dest` mapped through the device transform, exactly as `draw_bitmap` maps it.
        let (dpx, dpy) = self.get_dpi();
        let (sx, sy) = if self.get_unit_mode() == 0 { (dpx / 96.0, dpy / 96.0) } else { (1.0, 1.0) };
        let tr = self.get_transform();
        let map = |x: f32, y: f32| [sx * ((tr[0] * x) + (tr[2] * y) + tr[4]), sy * ((tr[1] * x) + (tr[3] * y) + tr[5])];
        let a = map(dest[0], dest[1]);
        let c = map(dest[2], dest[3]);
        let dev = [a[0].min(c[0]), a[1].min(c[1]), a[0].max(c[0]), a[1].max(c[1])];
        let Some(saved) = self.take_target() else { return };
        let mut inter = saved.clone();
        inter.data.fill(0);
        if bmp.width > 0 && bmp.height > 0 {
            // A float output reads as premultiplied BGRA8, clamped, as `draw_bitmap` presents it on the SOURCE_OVER
            // path (r27 32893: a Composite over an over-range Flood drawn SOURCE_COPY read its float bytes as texels).
            let bmp = if crate::float::is_highp(bmp.format) { std::borrow::Cow::Owned(float_input_to_bgra8(bmp)) } else { std::borrow::Cow::Borrowed(bmp) };
            crate::bitmap::draw_bitmap(&mut inter, &bmp, dev, [0.0, 0.0, w, h], 1.0, interp);
        }
        self.set_target_bitmap(saved);
        // DrawImage of an effect carries no imageRectangle, so BOUNDED_SOURCE_COPY(11) is unbounded ≡
        // SOURCE_COPY(10) — a whole-target composite that clears everything outside the effect output.
        let mode = if composite == 11 { 10 } else { composite };
        self.composite_over_target(&inter, mode, None);
    }

    /// The effect-DIP rect whose pixels can land inside the target: the target rectangle back-mapped
    /// through the inverse device transform, minus the draw `offset`. Bounds realization is clipped to
    /// this so an infinite-output effect fills only the visible region. `None` if no target is bound.
    fn effect_target_clip(&self, offset: [f32; 2]) -> Option<Rect> {
        let buf = self.target_bitmap()?;
        let (tw, th) = (buf.desc.width as f32, buf.desc.height as f32);
        let w = self.get_transform();
        let (dpx, dpy) = self.get_dpi();
        let (sx, sy) = if self.get_unit_mode() == 0 { (dpx / 96.0, dpy / 96.0) } else { (1.0, 1.0) };
        // device = DpiScale ∘ world, as [m11,m12,m21,m22,dx,dy]
        let m = [sx * w[0], sy * w[1], sx * w[2], sy * w[3], sx * w[4], sy * w[5]];
        let det = m[0] * m[3] - m[1] * m[2];
        if det == 0.0 {
            return Some(INF);
        }
        let inv = |x: f32, y: f32| {
            let ix = (m[3] * (x - m[4]) - m[2] * (y - m[5])) / det - offset[0];
            let iy = (-m[1] * (x - m[4]) + m[0] * (y - m[5])) / det - offset[1];
            [ix, iy]
        };
        let c = [inv(0.0, 0.0), inv(tw, 0.0), inv(0.0, th), inv(tw, th)];
        let minx = c.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let maxx = c.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
        let miny = c.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        let maxy = c.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);
        Some(Rect { l: minx.floor() as i32, t: miny.floor() as i32, r: maxx.ceil() as i32, b: maxy.ceil() as i32 })
    }
}

#[cfg(test)]
mod float_input_tests {
    use super::*;
    use crate::bitmap::Bitmap;

    // A float (R32_FLOAT) input bitmap must decode to 8bpc premultiplied BGRA through `bitmap_texel`,
    // not be read as raw bytes — the filter/affine paths index it as BGRA8. In-range values round to
    // UNORM8; an HDR value (>1) clamps to 255 (the UNORM8 working-buffer ceiling).
    #[test]
    fn float_input_decodes_to_premul_bgra8() {
        // 2x1: texel0 in-range (R,G,B,A), texel1 HDR red (2.0) opaque.
        let px = [0.5f32, 0.25, 0.75, 1.0,  2.0, 0.0, 0.0, 1.0];
        let b = Bitmap::from_f32_rgba(2, 1, &px);
        let got = float_input_to_bgra8(&b);
        assert_eq!(got.format, DXGI_FORMAT_B8G8R8A8_UNORM);
        let q = |v: f32| (v * 255.0).round_ties_even().clamp(0.0, 255.0) as u8;
        // texel0 stored BGRA = (B,G,R,A)
        assert_eq!(&got.premul_bgra[0..4], &[q(0.75), q(0.25), q(0.5), 255]);
        // texel1: R=2.0 clamps to 255, G=B=0
        assert_eq!(&got.premul_bgra[4..8], &[0, 0, 255, 255]);
    }

    // The fix must not touch an 8bpc leaf: realize_input returns it unchanged.
    #[test]
    fn bgra8_input_passes_through() {
        let b = Bitmap::from_straight_rgba(1, 1, &[10, 20, 30, 255]);
        let (out, ox, oy) = realize_input(&Input::Bitmap(std::rc::Rc::new(b.clone())));
        assert_eq!((ox, oy), (0, 0));
        assert_eq!(out.premul_bgra, b.premul_bgra);
    }
}

#[cfg(test)]
mod missing_input_guard_tests {
    use super::*;
    // Runs in node (not run_in_browser): the core is pure CPU, so this needs no web APIs, and node
    // exercises the same wasm32 panic=abort build without a headless browser.

    // G5: drawing an effect whose required input is unset must defer an error to EndDraw WITHOUT
    // panicking. draw_image_effect validates the graph (graph_valid) and latches E_INVALIDARG, as
    // native rejects a missing effect input. Panic-free by design — on wasm32 (panic=abort, the
    // browser target) an out-of-bounds index would kill the page, so this runs on native AND wasm.
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn missing_effect_input_latches_error_not_panic() {
        let mut dc = crate::context::DeviceContext::new(crate::resource::Domain { factory_id: 0, device_id: 0 });
        dc.set_target_bitmap(crate::context::bitmap_target(8, 8, 96.0));
        dc.begin_draw();
        let eff = dc.create_effect(CLSID_D2D1GaussianBlur).expect("create GaussianBlur");
        // No set_input: the GaussianBlur has no input 0, so the pull indexes an out-of-bounds input.
        dc.draw_image_effect(&eff, [0.0, 0.0], 1, 0); // must not panic out of the draw
        let (hr, _, _) = dc.end_draw();
        assert_eq!(hr, E_INVALIDARG, "missing effect input must latch E_INVALIDARG, not abort");
    }

    // The D2D1_ALPHA_MODE range enforcement in native's setter: an
    // AlphaMode SetValue outside {PREMULTIPLIED=1, STRAIGHT=2} is E_INVALIDARG, not clamped. The same
    // enum backs EdgeDetection (index 4) and Sepia (index 1). fuzz r15 EdgeDetection + r16 Sepia.
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn alpha_mode_enforce_sepia_and_edge_detection() {
        let dc = crate::context::DeviceContext::new(crate::resource::Domain { factory_id: 0, device_id: 0 });
        let sepia = dc.create_effect(CLSID_D2D1Sepia).expect("create Sepia");
        assert!(sepia.set_value(1, Prop::U32(1)).is_ok()); // PREMULTIPLIED
        assert!(sepia.set_value(1, Prop::U32(2)).is_ok()); // STRAIGHT
        assert_eq!(sepia.set_value(1, Prop::U32(0)), Err(E_INVALIDARG)); // UNKNOWN
        assert_eq!(sepia.set_value(1, Prop::U32(3)), Err(E_INVALIDARG)); // out of range
        let edge = dc.create_effect(CLSID_D2D1EdgeDetection).expect("create EdgeDetection");
        assert!(edge.set_value(4, Prop::U32(2)).is_ok());
        assert_eq!(edge.set_value(4, Prop::U32(0)), Err(E_INVALIDARG));
        // Every enum property: the top value takes, one past it rejects (r24 29663 Composite, 29357 3D transform).
        for (clsid, index, top) in [(CLSID_D2D1Composite, 0, 12), (CLSID_D2D13DTransform, 0, 4), (CLSID_D2D1Straighten, 2, 4), (CLSID_D2D1Blend, 0, 25)] {
            let e = dc.create_effect(clsid).expect("create");
            assert!(e.set_value(index, Prop::U32(top)).is_ok());
            assert_eq!(e.set_value(index, Prop::U32(top + 1)), Err(E_INVALIDARG));
        }
    }

    // The memoized `bounds()` ([perf]: the planner calls it on every node every draw) must drop on any
    // property/input change on the effect OR anything upstream of it — the global `BOUNDS_EPOCH` makes
    // one mutation invalidate every cached bounds, so a downstream memo can't survive an upstream edit.
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn bounds_memo_drops_on_property_and_upstream_change() {
        let dc = crate::context::DeviceContext::new(crate::resource::Domain { factory_id: 0, device_id: 0 });
        let bmp = Input::Bitmap(std::rc::Rc::new(Bitmap::from_straight_rgba(100, 100, &vec![255u8; 100 * 100 * 4])));

        // Direct: a Crop's bounds are `input ∩ rect`; changing the rect must re-measure, not return the memo.
        let crop = dc.create_effect(CLSID_D2D1Crop).expect("create Crop");
        crop.set_input(0, bmp.clone());
        crop.set_value(0, Prop::V4([0.0, 0.0, 50.0, 50.0])).unwrap();
        let tup = |r: Rect| (r.l, r.t, r.r, r.b);
        assert_eq!(tup(crop.bounds()), (0, 0, 50, 50));
        crop.set_value(0, Prop::V4([0.0, 0.0, 30.0, 30.0])).unwrap();
        assert_eq!(tup(crop.bounds()), (0, 0, 30, 30), "a property change must drop the bounds memo");

        // Upstream: an outer Crop (rect larger than its input) is 1:1 on an inner Crop; changing the INNER
        // rect must drop the OUTER memo too. A per-effect-local invalidation would leave the outer stale here.
        let inner = dc.create_effect(CLSID_D2D1Crop).expect("create inner Crop");
        inner.set_input(0, bmp.clone());
        inner.set_value(0, Prop::V4([0.0, 0.0, 40.0, 40.0])).unwrap();
        let outer = dc.create_effect(CLSID_D2D1Crop).expect("create outer Crop");
        outer.set_input(0, Input::Effect(inner.clone()));
        outer.set_value(0, Prop::V4([0.0, 0.0, 1000.0, 1000.0])).unwrap();
        assert_eq!(tup(outer.bounds()), (0, 0, 40, 40)); // caches outer + inner at this epoch
        inner.set_value(0, Prop::V4([0.0, 0.0, 20.0, 20.0])).unwrap(); // upstream edit bumps the global epoch
        assert_eq!(tup(outer.bounds()), (0, 0, 20, 20), "an upstream change must drop a downstream bounds memo");
    }
}

/// The rect `[l, t, r, b]` a rotated LINEAR image-brush fill over `crop` (DIPs, under device matrix `m`) asks of an
/// effect drawn at DIP `offset`, in effect pixels: the fill's device pixels (the crop's device bounds rounded out, within
/// the target) back-mapped at their centres, grown by the LINEAR footprint as the need rect rounds it (round-saturate
/// `min − 1`, `⌈max + ½⌉`). Its top-left is where the effect's output sits in the atlas the fill samples (sc106: six
/// crop positions; the uncropped request is [`alphablt_request`]'s).
pub(crate) fn crop_need(m: &[f32; 6], crop: [f32; 4], offset: [f32; 2], target: [u32; 2]) -> [i32; 4] {
    let cs = [[crop[0], crop[1]], [crop[2], crop[1]], [crop[0], crop[3]], [crop[2], crop[3]]].map(|[x, y]| [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]);
    let ext = |k: usize, f: fn(f32, f32) -> f32, init: f32| cs.iter().map(|p| p[k]).fold(init, f);
    let px = |v: f32, n: u32| v.clamp(0.0, n as f32) as i32;
    let r = [
        px(ext(0, f32::min, f32::INFINITY).floor(), target[0]),
        px(ext(1, f32::min, f32::INFINITY).floor(), target[1]),
        px(ext(0, f32::max, f32::NEG_INFINITY).ceil(), target[0]),
        px(ext(1, f32::max, f32::NEG_INFINITY).ceil(), target[1]),
    ];
    let back = crate::brush::invert(m);
    let c = [[r[0], r[1]], [r[2] - 1, r[1]], [r[0], r[3] - 1], [r[2] - 1, r[3] - 1]].map(|[x, y]| {
        let (x, y) = (x as f32 + 0.5, y as f32 + 0.5);
        [back[0] * x + back[2] * y + back[4] - offset[0], back[1] * x + back[3] * y + back[5] - offset[1]]
    });
    let lo = |k: usize| (c.iter().map(|p| p[k]).fold(f32::INFINITY, f32::min) - 1.0 + 0.5).floor() as i32;
    let hi = |k: usize| (c.iter().map(|p| p[k]).fold(f32::NEG_INFINITY, f32::max) + 0.5).ceil() as i32;
    [lo(0), lo(1), hi(0), hi(1)]
}

/// The height of the render target a rotated LINEAR effect draw's alpha blit writes, which sets its row bands
/// ([`crate::sampling::alphablt_band`]): the target itself under SOURCE_OVER, PLUS and the copies, which draw there
/// directly, and otherwise the slow path's atlas, a power of two at least 512 tall holding the target-sized intermediate
/// (VM: a rotated Border on a 64×256 target bands at 16 rows under DEST_OVER, SOURCE_IN, SOURCE_OUT, DEST_ATOP, XOR and
/// MASK_INVERT and at 8 under SOURCE_COPY and BOUNDED_SOURCE_COPY; on a 64×600 one at 32 under XOR and DEST_OVER).
/// ponytail: an intermediate past 1024 rows bands at no single height (a 64×1100 XOR draw); unmodelled.
fn blit_target_height(composite: u32, th: u32) -> u32 {
    if matches!(composite, 0 | 9 | 10 | 11) { th } else { th.next_power_of_two().max(512) }
}

/// Direct2D's placement of a rotated LINEAR effect composite: the output realized over `size` texels at DIP
/// `dest0` under the device matrix `m`, composited onto a `target`-sized surface with `composite`.
/// Returns the alpha-blit rect and the effect's request `[l, t, r, b]` (DIP; its atlas starts at the
/// top-left). The rect is the output's device bounds grown by the half-texel LINEAR footprint, each edge
/// rounded as WARP's blit setup does (24.8 truncation, then `(v + 127) >> 8`), clipped to the target:
/// (25, 0, 45, 14) on fxrot-03, (40, 0, 56, 15) and (18, 0, 38, 11) on two captures. A copy requests the
/// output grown a texel; otherwise the request is the back-mapped rect's pixel centres grown by the
/// LINEAR footprint as Direct2D's need-rect calculation rounds it (round-saturate `min − 1`, `⌈max + ½⌉`;
/// the upscale destinations of ten rotated-blur DDI captures, clipped and fractional offsets included).
fn alphablt_request(m: &[f32; 6], dest0: [f32; 2], size: [u32; 2], composite: u32, target: [u32; 2]) -> ([usize; 4], [f32; 4]) {
    let (l, t) = (dest0[0] - 0.5, dest0[1] - 0.5);
    let (r, b) = (dest0[0] + size[0] as f32 + 0.5, dest0[1] + size[1] as f32 + 0.5);
    let corners = [[l, t], [r, t], [l, b], [r, b]].map(|[x, y]| [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]);
    let edge = |v: f32, n: u32| ((((v * 256.0) as i32).saturating_add(127) >> 8).clamp(0, n as i32)) as usize;
    let lo = |c: usize| edge(corners.iter().map(|p| p[c]).fold(f32::INFINITY, f32::min), target[c]);
    let hi = |c: usize| edge(corners.iter().map(|p| p[c]).fold(f32::NEG_INFINITY, f32::max), target[c]);
    let rect = [lo(0), lo(1), hi(0), hi(1)];
    if matches!(composite, 10 | 11) || rect[0] >= rect[2] || rect[1] >= rect[3] {
        let [x, y] = dest0;
        return (rect, [x - 1.0, y - 1.0, x + size[0] as f32 + 1.0, y + size[1] as f32 + 1.0]);
    }
    let back = crate::brush::invert(m);
    let c = [[rect[0], rect[1]], [rect[2] - 1, rect[1]], [rect[0], rect[3] - 1], [rect[2] - 1, rect[3] - 1]].map(|[x, y]| {
        let (x, y) = (x as f32 + 0.5, y as f32 + 0.5);
        [back[0] * x + back[2] * y + back[4], back[1] * x + back[3] * y + back[5]]
    });
    let lo = |k: usize| dest0[k] + (c.iter().map(|p| p[k]).fold(f32::INFINITY, f32::min) - dest0[k] - 0.5).floor();
    let hi = |k: usize| dest0[k] + (c.iter().map(|p| p[k]).fold(f32::NEG_INFINITY, f32::max) - dest0[k] + 0.5).ceil();
    (rect, [lo(0), lo(1), hi(0), hi(1)])
}

/// Native's HIGH_QUALITY_CUBIC 2D affine prescale, per matrix column. The column's length is measured between two
/// f32-translated points. It is clamped to [0.01, 1] and expressed as a power of 1/2 whose exponent snaps to the
/// nearest integer within 1e-5. A factor of 1 means no prescale (sc143: a 0.70710678 rotation gives exactly 1; 0.7071
/// prescales by 0.99999).
/// Whether a HIGH_QUALITY_CUBIC draw of an image placed at `at` under the device matrix `m` is the 2D affine transform
/// effect's CUBIC matrix transform alone: neither a pure positive scale at the origin (its Scale node) nor a placement it
/// pre-scales on an axis.
fn hq_draws_cubic(m: [f64; 6], at: [f32; 2]) -> bool {
    let (x, y) = (at[0] as f64, at[1] as f64);
    let a = [m[0] as f32, m[1] as f32, m[2] as f32, m[3] as f32, (m[0] * x + m[2] * y + m[4]) as f32, (m[1] * x + m[3] * y + m[5]) as f32];
    let pure_scale = a[1] == 0.0 && a[2] == 0.0 && a[0] > 0.01 && a[3] > 0.01 && a[4] == 0.0 && a[5] == 0.0;
    !pure_scale && hq_prescale_factor(a) == [1.0; 2]
}

pub(crate) fn hq_prescale_factor(m: [f32; 6]) -> [f32; 2] {
    let [a, b, c, d, dx, dy] = m;
    let z = 0.0f32;
    let (y0, x0) = ((b * z + d * z) + dy, (a * z + c * z) + dx);
    let col0 = [x0 - ((c * z + a) + dx), y0 - ((d * z + b) + dy)];
    let col1 = [x0 - ((a * z + c) + dx), y0 - ((b * z + d) + dy)];
    [col0, col1].map(|[u, v]| {
        let s = (v * v + u * u).sqrt();
        let s = 1.0f32.min(s).max(f32::from_bits(0x3c23_d70a));
        let x = (crate::crt::system_crt::logf(s) as f64 / -core::f64::consts::LN_2) as f32;
        let r = x.round_ties_even();
        let n = if (x - r).abs() >= 1e-5 { x } else { r };
        crate::crt::system_crt::powf(0.5, n)
    })
}

/// The span `[lo, hi)` a HIGH_QUALITY_CUBIC prescale by `scale` realizes. Under a HARD border it is the Scale effect's own
/// rect, the pixels whose centres the scaled span holds (a prescale by 0.99999 keeps the input's span: sc143). Under a
/// SOFT border it grows by the downscale filter's 2-px reach.
fn hq_prescale_span(lo: i32, hi: i32, scale: f32, hard: bool) -> (i32, i32) {
    if hard {
        ((lo as f32 * scale - 0.5).ceil() as i32, (hi as f32 * scale - 0.5).ceil() as i32)
    } else {
        ((lo as f32 * scale - 2.0).floor() as i32, (hi as f32 * scale + 2.0).ceil() as i32)
    }
}
