use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use std::{fs::File, io::BufWriter, path::Path};

// Optional instrumentation, off by default (behind the `native-detour` feature): the public
// reference renderer and the fuzzer build without it and render identically.
#[cfg(all(windows, feature = "native-detour"))]
pub mod warp_ddi;

// Seeded random scene generator for the differential fuzzer (the `fuzz-gen` bin and the fuzz runner).
pub mod fuzz;

/// Deserialize helpers that accept the strings `"nan"`, `"inf"`/`"+inf"`/`"infinity"` and
/// `"-inf"`/`"-infinity"` (case-insensitive) wherever a JSON number appears — the fuzzer feeds
/// non-finite coordinates and JSON has no NaN/Infinity literal. Plain numbers deserialize unchanged,
/// so every existing scene parses byte-identically. Applied via `#[serde(deserialize_with)]` on the
/// coordinate-bearing `f32`/`[f32; N]` fields.
pub mod num {
    use serde::de::{self, DeserializeSeed, SeqAccess, Visitor};
    use serde::Deserializer;
    use std::fmt;

    fn from_str(s: &str) -> Option<f32> {
        match s.trim().to_ascii_lowercase().as_str() {
            "nan" => Some(f32::NAN),
            "inf" | "+inf" | "infinity" | "+infinity" => Some(f32::INFINITY),
            "-inf" | "-infinity" => Some(f32::NEG_INFINITY),
            other => other.parse::<f32>().ok(),
        }
    }

    struct F32;
    impl<'de> Visitor<'de> for F32 {
        type Value = f32;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a number or \"nan\"/\"inf\"/\"-inf\"")
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> Result<f32, E> { Ok(v as f32) }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<f32, E> { Ok(v as f32) }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<f32, E> { Ok(v as f32) }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<f32, E> {
            from_str(v).ok_or_else(|| E::custom(format!("not a float: {v:?}")))
        }
    }
    impl<'de> DeserializeSeed<'de> for F32 {
        type Value = f32;
        fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<f32, D::Error> { d.deserialize_any(F32) }
    }

    pub fn scalar<'de, D: Deserializer<'de>>(d: D) -> Result<f32, D::Error> { d.deserialize_any(F32) }

    struct Arr<const N: usize>;
    impl<'de, const N: usize> Visitor<'de> for Arr<N> {
        type Value = [f32; N];
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "an array of {N} numbers/nan/inf")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<[f32; N], A::Error> {
            let mut out = [0f32; N];
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = seq.next_element_seed(F32)?.ok_or_else(|| de::Error::invalid_length(i, &self))?;
            }
            if seq.next_element::<de::IgnoredAny>()?.is_some() {
                return Err(de::Error::custom("array too long"));
            }
            Ok(out)
        }
    }
    pub fn arr2<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; 2], D::Error> { d.deserialize_seq(Arr::<2>) }
    pub fn arr3<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; 3], D::Error> { d.deserialize_seq(Arr::<3>) }
    pub fn arr4<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; 4], D::Error> { d.deserialize_seq(Arr::<4>) }
    pub fn arr6<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; 6], D::Error> { d.deserialize_seq(Arr::<6>) }
    pub fn arr16<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; 16], D::Error> { d.deserialize_seq(Arr::<16>) }
    pub fn arr20<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; 20], D::Error> { d.deserialize_seq(Arr::<20>) }

    /// `Vec<f32>` with each element accepting nan/inf (e.g. a stroke dash array).
    pub fn vecf<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f32>, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Vec<f32>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result { f.write_str("a list of numbers/nan/inf") }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<f32>, A::Error> {
                let mut out = Vec::new();
                while let Some(v) = seq.next_element_seed(F32)? {
                    out.push(v);
                }
                Ok(out)
            }
        }
        d.deserialize_seq(V)
    }

    /// `[f32; N]` as a `DeserializeSeed`, so option/vec visitors can nest a flexible fixed array.
    struct ArrSeed<const N: usize>;
    impl<'de, const N: usize> DeserializeSeed<'de> for ArrSeed<N> {
        type Value = [f32; N];
        fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<[f32; N], D::Error> { d.deserialize_seq(Arr::<N>) }
    }
    struct OptArr<const N: usize>;
    impl<'de, const N: usize> Visitor<'de> for OptArr<N> {
        type Value = Option<[f32; N]>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "null or {N} numbers/nan/inf") }
        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> { Ok(None) }
        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> { Ok(None) }
        fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> { d.deserialize_seq(Arr::<N>).map(Some) }
    }
    /// `Option<[f32; 4]>` / `Option<[f32; 6]>` accepting null → None and nan/inf per element.
    pub fn opt_arr4<'de, D: Deserializer<'de>>(d: D) -> Result<Option<[f32; 4]>, D::Error> { d.deserialize_option(OptArr::<4>) }
    pub fn opt_arr6<'de, D: Deserializer<'de>>(d: D) -> Result<Option<[f32; 6]>, D::Error> { d.deserialize_option(OptArr::<6>) }
    pub fn opt_arr2<'de, D: Deserializer<'de>>(d: D) -> Result<Option<[f32; 2]>, D::Error> { d.deserialize_option(OptArr::<2>) }
    /// `Vec<[[f32; 2]; 3]>` (fill-mesh triangles) with each coordinate accepting nan/inf.
    pub fn tris<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<[[f32; 2]; 3]>, D::Error> {
        struct Tri;
        impl<'de> DeserializeSeed<'de> for Tri {
            type Value = [[f32; 2]; 3];
            fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
                struct T;
                impl<'de> Visitor<'de> for T {
                    type Value = [[f32; 2]; 3];
                    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result { f.write_str("3 points") }
                    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                        let mut out = [[0f32; 2]; 3];
                        for (i, slot) in out.iter_mut().enumerate() {
                            *slot = seq.next_element_seed(ArrSeed::<2>)?.ok_or_else(|| de::Error::invalid_length(i, &self))?;
                        }
                        Ok(out)
                    }
                }
                d.deserialize_seq(T)
            }
        }
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Vec<[[f32; 2]; 3]>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result { f.write_str("a list of triangles") }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut out = Vec::new();
                while let Some(t) = seq.next_element_seed(Tri)? {
                    out.push(t);
                }
                Ok(out)
            }
        }
        d.deserialize_seq(V)
    }

    /// `[f32; 2]` as a `DeserializeSeed`, so a sequence of points can accept nan/inf per element.
    struct Pt;
    impl<'de> DeserializeSeed<'de> for Pt {
        type Value = [f32; 2];
        fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<[f32; 2], D::Error> { d.deserialize_seq(Arr::<2>) }
    }
    /// `[[f32; 2]; 16]` (a gradient-mesh patch's points) with each coordinate accepting nan/inf.
    pub fn arr2x16<'de, D: Deserializer<'de>>(d: D) -> Result<[[f32; 2]; 16], D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = [[f32; 2]; 16];
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "16 points") }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut out = [[0f32; 2]; 16];
                for (i, slot) in out.iter_mut().enumerate() {
                    *slot = seq.next_element_seed(Pt)?.ok_or_else(|| de::Error::invalid_length(i, &self))?;
                }
                if seq.next_element::<de::IgnoredAny>()?.is_some() {
                    return Err(de::Error::custom("array too long"));
                }
                Ok(out)
            }
        }
        d.deserialize_seq(V)
    }

    struct OptF32;
    impl<'de> Visitor<'de> for OptF32 {
        type Value = Option<f32>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result { f.write_str("null or a number/nan/inf") }
        fn visit_none<E: de::Error>(self) -> Result<Option<f32>, E> { Ok(None) }
        fn visit_unit<E: de::Error>(self) -> Result<Option<f32>, E> { Ok(None) }
        fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Option<f32>, D::Error> { d.deserialize_any(F32).map(Some) }
    }
    /// `Option<f32>` that accepts null → None and number-or-"nan"/"inf"/"-inf" → Some.
    pub fn opt_scalar<'de, D: Deserializer<'de>>(d: D) -> Result<Option<f32>, D::Error> { d.deserialize_option(OptF32) }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scene {
    pub version: u32,
    pub size: [u32; 2],
    pub dpi: f32,
    pub ops: Vec<Op>,
    // Target pixel format and alpha mode (DXGI_FORMAT / D2D1_ALPHA_MODE numeric values). Absent ⇒
    // B8G8R8A8_UNORM premultiplied, so every existing scene is byte-identical. Float targets
    // (R16G16B16A16_FLOAT=10, R32G32B32A32_FLOAT=2) and R8G8B8A8_UNORM=28 are the HDR additions.
    #[serde(default)]
    pub format: Option<u32>,
    #[serde(default)]
    pub alpha: Option<u32>,
    // `ID2D1DeviceContext::SetRenderingControls` bufferPrecision (`D2D1_BUFFER_PRECISION`): the DEFAULT
    // intermediate precision for the effect graph (5 = 32BPC_FLOAT). Absent ⇒ leave the DC default
    // (8bpc), so existing scenes are unchanged. This is the term that actually caps a custom effect's
    // intermediate precision (a per-transform SetOutputBuffer does not raise it past this).
    #[serde(default)]
    pub buffer_precision: Option<u32>,
}

// Numeric enum fields use the Win32 metadata values, keeping the format extensible.
/// One resource texture bound to a `register_shader_effect` draw transform: the D2D1_RESOURCE_TEXTURE
/// _PROPERTIES fields plus the raw texel bytes, bound at `index`. `precision`/`channel_depth`/`filter`/
/// `extend_modes` are the raw D2D1 enum values (precision 5 = 32BPC_FLOAT, channel_depth 1, filter 0 =
/// MIN_MAG_MIP_POINT, extend 0 = CLAMP); `extent`×dims and the bytes must agree.
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ResourceTextureSpec {
    pub index: u32,
    pub dimensions: u32,
    pub extents: Vec<u32>,
    #[serde(default)]
    pub precision: u32,
    #[serde(default)]
    pub channel_depth: u32,
    #[serde(default)]
    pub filter: u32,
    #[serde(default)]
    pub extend_modes: Vec<u32>,
    pub data: Vec<u8>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Op {
    Clear {
        color: [f32; 4],
    },
    Transform {
        #[serde(deserialize_with = "num::arr6")]
        matrix: [f32; 6],
    },
    Antialias {
        mode: u32,
    },
    Blend {
        mode: u32,
    },
    Solid {
        id: String,
        color: [f32; 4],
        #[serde(default)]
        opacity: Option<f32>,
        #[serde(default)]
        transform: Option<[f32; 6]>,
    },
    Linear {
        id: String,
        #[serde(deserialize_with = "num::arr2")]
        start: [f32; 2],
        #[serde(deserialize_with = "num::arr2")]
        end: [f32; 2],
        stops: Vec<Stop>,
        gamma: u32,
        extend: u32,
        #[serde(default)]
        opacity: Option<f32>,
        #[serde(default)]
        transform: Option<[f32; 6]>,
    },
    Radial {
        id: String,
        #[serde(deserialize_with = "num::arr2")]
        center: [f32; 2],
        #[serde(deserialize_with = "num::arr2")]
        offset: [f32; 2],
        #[serde(deserialize_with = "num::arr2")]
        radius: [f32; 2],
        stops: Vec<Stop>,
        gamma: u32,
        extend: u32,
        #[serde(default)]
        opacity: Option<f32>,
        #[serde(default)]
        transform: Option<[f32; 6]>,
    },
    Bitmap {
        id: String,
        size: [u32; 2],
        rgba: Vec<u8>,
        // Raw RGBA f32 texels (4 per pixel), uploaded verbatim as R32G32B32A32_FLOAT (as R16G16B16A16_FLOAT
        // halves when `format` is 10) — takes precedence
        // over `rgba`/`format` when non-empty. Lets a scene carry values outside [0,1] (e.g. a
        // scene-position image a SampleMap shader samples), which the u8 `rgba` path cannot express.
        #[serde(default)]
        rgba_f32: Vec<f32>,
        // Raw DXGI texels, used to test storage independently of upload conversion.
        #[serde(default)]
        data: Vec<u8>,
        // Optional bitmap pixel format / alpha mode (DXGI_FORMAT / D2D1_ALPHA_MODE). Absent ⇒
        // B8G8R8A8_UNORM premultiplied (v1 behaviour). A float format uploads the straight RGBA8
        // converted to float (values in [0,1]); the alpha mode controls premultiply-on-upload.
        #[serde(default)]
        format: Option<u32>,
        #[serde(default)]
        alpha: Option<u32>,
        // Optional bitmap DPI (default 96). A higher DPI makes DrawImage minify the bitmap (the COLR
        // paint-tree-image path draws an ss×DPI image at ~identity), for probing WARP's minify kernel.
        #[serde(default)]
        dpi: Option<f32>,
    },
    BitmapBrush {
        id: String,
        bitmap: String,
        extend: [u32; 2],
        interpolation: u32,
        #[serde(deserialize_with = "num::arr6")]
        matrix: [f32; 6],
        opacity: f32,
    },
    Geometry {
        id: String,
        shape: Shape,
    },
    // ID2D1Factory::CreateGeometryGroup — a group of existing geometry ids with one fill mode.
    GeometryGroup {
        id: String,
        fill: u32,
        geometries: Vec<String>,
    },
    // ID2D1Factory::CreateTransformedGeometry — an existing geometry id under a 3x2 transform.
    GeometryTransform {
        id: String,
        geometry: String,
        #[serde(deserialize_with = "num::arr6")]
        matrix: [f32; 6],
    },
    Primitive {
        shape: Shape,
        brush: String,
        // `deserialize_with` on an Option disables serde's implicit "missing ⇒ None", so restore it
        // with `default` (some scenes omit width).
        #[serde(default, deserialize_with = "num::opt_scalar")]
        width: Option<f32>,
    },
    // ID2D1Factory::CreateWicBitmapRenderTarget + IsSupported: render a filled rect into a WIC bitmap
    // and read the pixels back (FFI-only; the render-target variants have no core counterpart).
    WicRenderTarget {
        size: [u32; 2],
        fill_color: [f32; 4],
    },
    // ID2D1Factory::CreateDCRenderTarget + BindDC: bind a DIB-section memory DC, render a filled rect,
    // read the DIB pixels back.
    DcRenderTarget {
        size: [u32; 2],
        fill_color: [f32; 4],
    },
    // ID2D1GdiInteropRenderTarget GetDC/ReleaseDC: on a GDI-compatible WIC render target, GDI-fill a
    // rect through the interop DC, then read the bitmap back.
    GdiInteropDraw {
        size: [u32; 2],
        gdi_color: [u8; 3],
        rect: [i32; 4],
    },
    // ID2D1Factory::CreateHwndRenderTarget + GetHwnd/CheckWindowState/Resize over a hidden window
    // (headless: no window pixels, so semantic readback only).
    HwndRenderTarget {
        size: [u32; 2],
        resize: [u32; 2],
        color: [f32; 4],
    },
    // ID2D1Bitmap::CopyFromRenderTarget — draw, capture the frame into a bitmap, then redraw from it.
    CopyFromRtTest {
        rect: [f32; 4],
        fill: [f32; 4],
    },
    // ID2D1Factory::CreateGdiMetafile — build a real EMF via GDI, feed it to CreateGdiMetafile, and
    // record the HRESULT + accessors. Native delegates full EMF playback to the external
    // CreateMetafileRenderer apiset (absent ⇒ D2DERR_UNSUPPORTED_OPERATION 0x88990003), so this probes
    // the creation/delegation HRESULT rather than pixels.
    GdiMetafileProbe {},
    // ID2D1DeviceContext::DrawGdiMetafile — load an EMF(+)/EMF file from `file` (cwd-relative), create
    // an ID2D1GdiMetafile, and draw it at its natural size at the origin. Native plays it through the
    // CreateMetafileRenderer apiset (gdiplus); ours through the core EMF+ player + rasterizer. Pixel
    // comparison for the metafile playback port.
    DrawMetafile {
        // The EMF(+) source: either a cwd-relative `file` or, for a self-contained suite scene, the
        // raw bytes inline (like shader scenes carry their DXBC). `bytes` wins when non-empty.
        #[serde(default)]
        file: String,
        #[serde(default)]
        bytes: Vec<u8>,
    },
    // ID2D1RenderTarget::CreateCompatibleRenderTarget — render into an offscreen compatible target,
    // then GetBitmap + DrawBitmap it 1:1 onto the main target (an offscreen round-trip, pixel-exact).
    CompatibleRenderTarget {
        size: [u32; 2],
        bg: [f32; 4],
        rect: [f32; 4],
        fill: [f32; 4],
        dest: [f32; 2],
    },
    // ID2D1RenderTarget::DrawLine — strokes a single line segment (same path as DrawGeometry on a
    // one-segment open figure).
    DrawLine {
        #[serde(deserialize_with = "num::arr2")]
        p0: [f32; 2],
        #[serde(deserialize_with = "num::arr2")]
        p1: [f32; 2],
        brush: String,
        #[serde(deserialize_with = "num::scalar")]
        width: f32,
    },
    Stroke {
        id: String,
        caps: [u32; 3],
        join: u32,
        #[serde(deserialize_with = "num::scalar")]
        miter: f32,
        #[serde(deserialize_with = "num::scalar")]
        offset: f32,
        #[serde(deserialize_with = "num::vecf")]
        dashes: Vec<f32>,
        // `ID2D1StrokeStyle1` stroke transform type (0 normal / 1 fixed / 2 hairline); absent ⇒ 0,
        // so v1/v2 scenes are byte-identical.
        #[serde(default)]
        transform_type: u32,
        // Named `D2D1_DASH_STYLE` (0 SOLID/CUSTOM, 1 DASH, 2 DOT, 3 DASH_DOT, 4 DASH_DOT_DOT); absent ⇒ 0.
        // A named style ignores the `dashes` array and uses the predefined pattern (native), which we
        // expand to the same pattern for the core drive.
        #[serde(default)]
        dash_style: u32,
    },
    Fill {
        geometry: String,
        brush: String,
    },
    // ID2D1RenderTarget::CreateMesh + FillMesh: tessellate `triangles` (device-space triples) into a
    // mesh and fill it with `brush`. Used to probe native's FillMesh (aliased requirement,
    // per-triangle overlap, empty/degenerate handling).
    FillMesh {
        #[serde(deserialize_with = "num::tris")]
        triangles: Vec<[[f32; 2]; 3]>,
        brush: String,
    },
    Draw {
        geometry: String,
        brush: String,
        #[serde(deserialize_with = "num::scalar")]
        width: f32,
        style: Option<String>,
    },
    DrawBitmap {
        bitmap: String,
        #[serde(deserialize_with = "num::arr4")]
        dest: [f32; 4],
        #[serde(deserialize_with = "num::arr4")]
        source: [f32; 4],
        opacity: f32,
        interpolation: u32,
        /// Which COM overload to call: default = `ID2D1DeviceContext::DrawBitmap`; `"rt"` =
        /// `ID2D1RenderTarget::DrawBitmap`. The core render is identical — this exercises both C-API
        /// entry points so the FFI-vs-core gate covers both.
        #[serde(default)]
        overload: Option<String>,
    },
    PushClip {
        #[serde(deserialize_with = "num::arr4")]
        rect: [f32; 4],
        antialias: u32,
    },
    PopClip,
    PushLayer {
        #[serde(deserialize_with = "num::arr4")]
        bounds: [f32; 4],
        mask: Option<String>,
        opacity: f32,
        opacity_brush: Option<String>,
        #[serde(default)]
        mask_antialias: Option<u32>,
        #[serde(default, deserialize_with = "num::opt_arr6")]
        mask_transform: Option<[f32; 6]>,
        #[serde(default)]
        layer_options: Option<u32>,
    },
    PopLayer,
    Effect {
        id: String,
        clsid: String,
        inputs: Vec<String>,
        properties: Vec<Property>,
    },
    // Colour management (cms): create an ID2D1ColorContext in its own id-space. `space` picks the
    // constructor — "srgb"/"scrgb" (named), "custom" (ICC `profile` bytes), "dxgi" (`dxgi`
    // colour-space enum), "simple" (`simple` profile). An effect property of type `iunknown` names
    // one of these ids (used for ColorManagement's source/destination colour contexts).
    ColorContext {
        id: String,
        space: String,
        #[serde(default)]
        profile: Vec<u8>,
        #[serde(default)]
        dxgi: Option<u32>,
        #[serde(default)]
        simple: Option<SimpleProfile>,
    },
    // Phase 2B-3: register a custom effect whose single draw transform binds an authored pixel
    // shader (`dxbc`, a full ps_5_0 compiled from our own HLSL) with the `cbuffer` floats at b0 over
    // `inputs` simple inputs. A following `effect` op instantiates it by `clsid`, binds the inputs,
    // and `draw_image` draws the output. `filter` is the input description's D2D1_FILTER (pixel-inert
    // for the 1:1 simple inputs these scenes use).
    RegisterShaderEffect {
        clsid: String,
        dxbc: Vec<u8>,
        #[serde(default)]
        cbuffer: Vec<f32>,
        // Raw little-endian b0 bytes; takes precedence over `cbuffer` when non-empty. For shaders
        // whose constant buffer mixes int/uint/float fields (e.g. a PCG seed), which `cbuffer: f32`
        // cannot express.
        #[serde(default)]
        cbuffer_bytes: Vec<u8>,
        inputs: u32,
        #[serde(default)]
        filter: u32,
        // Resource textures the draw transform creates (CreateResourceTexture) and binds
        // (SetResourceTexture) — e.g. a 1-D LUT a shader reads via `ld`/resinfo. Empty for the
        // common simple-input shader.
        #[serde(default)]
        resource_textures: Vec<ResourceTextureSpec>,
        // `ID2D1TransformGraph::SetOutputBuffer` precision for the draw transform (`D2D1_BUFFER_PRECISION`
        // numeric: 5 = 32BPC_FLOAT, 4 = 16BPC_FLOAT, 1 = 8BPC_UNORM). Absent ⇒ inherit (the 8bpc default),
        // so every existing scene is unchanged. Lets a chain carry a float intermediate a downstream
        // shader reads at full precision.
        #[serde(default)]
        precision: Option<u32>,
        // The draw transform's MapOutputRectToInputRects: each input rect is the output rect inflated by
        // this many pixels, saturating, so 2147483647 is D2D's infinite rect (what ComputeSharp.D2D1 asks
        // for a complex input). Absent ⇒ identity, so every existing scene is unchanged.
        #[serde(default)]
        input_inflate: Option<i32>,
        // `ID2D1DrawInfo::SetPixelShader` pixel options (`D2D1_PIXEL_OPTIONS`): 1 = TRIVIAL_SAMPLING, which
        // with a shader blob carrying its export library (`D2D_FUNCTION` + `D2D_FULL_SHADER`, as ComputeSharp
        // compiles Paint.NET's) lets D2D link the effect into a neighbouring pass. Absent ⇒ NONE (unlinked).
        #[serde(default)]
        pixel_options: u32,
    },
    // Phase 3C: register a custom effect whose draw transform runs a custom vertex shader over a
    // vertex buffer (`vertices`: flat `[pos.x, pos.y, uv.x, uv.y]` per vertex, a POSITION+TEXCOORD
    // layout) that deforms a mesh the pixel shader (`ps_dxbc`) then shades, sampling the simple inputs
    // at the interpolated texcoord. `vs_cbuffer` is the VS `b0`; `viewport` is the effect output rect.
    RegisterVertexEffect {
        clsid: String,
        vs_dxbc: Vec<u8>,
        ps_dxbc: Vec<u8>,
        #[serde(default)]
        vs_cbuffer: Vec<f32>,
        vertices: Vec<f32>,
        inputs: u32,
        viewport: [i32; 4],
    },
    /// Create a `LookupTable3D` effect, build its `ID2D1LookupTable3D` from `extents`/`data`
    /// (flat RGBA, R fastest) and bind it, storing the effect output under `id`. `data` is given in
    /// f32; the harness encodes it into the table's native byte layout at `precision`
    /// (D2D1_BUFFER_PRECISION: 1=8BPC_UNORM, 3=16BPC_UNORM, 5=32BPC_FLOAT; default 32BPC_FLOAT), so a
    /// probe controls exactly what native decodes at each precision.
    #[serde(rename = "lookup_table3d")]
    LookupTable3D {
        id: String,
        input: String,
        extents: [u32; 3],
        data: Vec<f32>,
        #[serde(default)]
        precision: Option<u32>,
    },
    DrawImage {
        image: String,
        #[serde(deserialize_with = "num::arr2")]
        offset: [f32; 2],
        interpolation: u32,
        composite: u32,
        // The `imageRectangle` (image-space DIPs): the sub-rect of the image to draw, its top-left
        // placed at `offset`. Absent → the whole image (the v1/v2 default; native passes nullptr).
        #[serde(default)]
        image_rect: Option<[f32; 4]>,
    },
    Text {
        text: String,
        font: String,
        #[serde(deserialize_with = "num::scalar")]
        size: f32,
        #[serde(deserialize_with = "num::arr4")]
        rect: [f32; 4],
        brush: String,
        weight: u32,
        style: u32,
        locale: String,
        antialias: u32,
        measuring: u32,
        // DrawText options (D2D1_DRAW_TEXT_OPTIONS); absent → NONE. 4 = ENABLE_COLOR_FONT (emoji).
        #[serde(default)]
        options: Option<u32>,
        /// Which COM overload: default = `DrawText` (lays out + draws); `"layout"` =
        /// `DrawTextLayout` (build an IDWriteTextLayout, then draw it). Same core render — exercises
        /// the second text entry point so the FFI-vs-core gate covers it.
        #[serde(default)]
        overload: Option<String>,
        /// Text rendering params for this draw, `[gamma, enhanced contrast, grayscale enhanced contrast, ClearType
        /// level]` (flat geometry, default mode and grid fit); absent ⇒ the pinned 2.2 / 1 / 1 / 1.
        #[serde(default)]
        params: Option<[f32; 4]>,
    },
    // Records DWRITE_TEXT_METRICS of an IDWriteTextLayout (regular, en-us) into the query channel
    // (no pixels): the natural port of canvas measureText, for the font-advance ppem cases.
    TextMetrics {
        text: String,
        font: String,
        size: f32,
    },
    // Phase 2D-3: the DrawGlyphRun primitive directly. The run is built from `text` via the font's
    // cmap + design-metric advances (as DirectWrite's layout would), then drawn at `baseline` (device
    // DIPs) with a solid brush. `rendering_mode` (DWRITE_RENDERING_MODE1; 0 = default/recommended)
    // and `grid_fit` (DWRITE_GRID_FIT_MODE; 1 = disabled) are pinned via the DC's text rendering
    // params so a scene can isolate the raster+blend join from the hinting interpreter (grid-fit
    // disabled) or exercise the full hinted path (default).
    GlyphRun {
        text: String,
        font: String,
        size: f32,
        #[serde(default = "weight_normal")]
        weight: u32,
        #[serde(default)]
        style: u32,
        baseline: [f32; 2],
        brush: String,
        antialias: u32,
        measuring: u32,
        #[serde(default)]
        rendering_mode: u32,
        #[serde(default)]
        grid_fit: u32,
        // DWRITE_FONT_SIMULATIONS bits (1 = BOLD, 2 = OBLIQUE). The native oracle synthesizes these
        // from `weight`/`style` via GetFirstMatchingFont; our runner has no font-matching layer, so the
        // scene states them explicitly for the bold/oblique-sim cases.
        #[serde(default)]
        simulations: u32,
    },
    // --- v2 additions (strict superset; all of the above is v1) ---
    // A WIC bitmap source uploaded from inline straight RGBA8, realized as a D2D
    // bitmap via CreateBitmapFromWicBitmap; shares the bitmap/image ID namespace.
    WicBitmap {
        id: String,
        size: [u32; 2],
        rgba: Vec<u8>,
        #[serde(default)]
        format: Option<u32>,
        #[serde(default)]
        alpha: Option<u32>,
    },
    ImageBrush {
        id: String,
        image: String,
        #[serde(deserialize_with = "num::arr4")]
        source: [f32; 4],
        extend: [u32; 2],
        interpolation: u32,
        opacity: f32,
        #[serde(deserialize_with = "num::arr6")]
        matrix: [f32; 6],
    },
    // Records nested ops into an ID2D1CommandList image, usable by draw_image.
    CommandList {
        id: String,
        ops: Vec<Op>,
    },
    InkStyle {
        id: String,
        nib_shape: u32,
        #[serde(deserialize_with = "num::arr6")]
        nib_transform: [f32; 6],
    },
    Ink {
        id: String,
        #[serde(deserialize_with = "num::arr3")]
        start: [f32; 3],
        segments: Vec<InkSegment>,
    },
    // ID2D1Ink::RemoveSegmentsAtEnd — drops `count` bezier segments off the tail. Its HRESULT is
    // recorded per-op (covers count > segment-count).
    InkRemoveSegmentsAtEnd {
        ink: String,
        count: u32,
    },
    DrawInk {
        ink: String,
        brush: String,
        style: Option<String>,
    },
    GradientMesh {
        id: String,
        patches: Vec<MeshPatch>,
    },
    DrawGradientMesh {
        mesh: String,
    },
    SpriteBatch {
        id: String,
        sprites: Vec<Sprite>,
    },
    DrawSpriteBatch {
        batch: String,
        bitmap: String,
        interpolation: u32,
        options: u32,
    },
    SvgDocument {
        id: String,
        xml: String,
        viewport: [f32; 2],
    },
    DrawSvgDocument {
        doc: String,
    },
    // --- SVG DOM editing (phase 3A, `ID2D1SvgDocument`/`ID2D1SvgElement`) ----------------------
    // Element handles bind under a scene-wide name (`elem`) so later ops reference them; every
    // op forwards to the COM object (harness) or the core `Document` (conformance), so pixels and
    // query results must match the WARP oracle. Mutating ops are expected to succeed (`S_OK`);
    // `svg_query` folds any value/HRESULT result into the query channel.
    SvgGetRoot {
        doc: String,
        elem: String,
    },
    SvgCreateChild {
        parent: String,
        tag: String,
        elem: String,
    },
    SvgSetAttr {
        elem: String,
        name: String,
        value: String,
    },
    /// Capture `ID2D1SvgElement::GetFirstChild(parent)` as `elem` (e.g. a parsed element's text node).
    SvgFirstChild {
        parent: String,
        elem: String,
    },
    /// `ID2D1SvgElement::SetTextValue` on a text-content element (a text node).
    SvgSetText {
        elem: String,
        text: String,
    },
    /// `ID2D1SvgElement::RemoveAttribute`.
    SvgRemoveAttr {
        elem: String,
        name: String,
    },
    /// `ID2D1SvgElement::SetAttributeValue2` (a POD attribute): `pod` is the
    /// `D2D1_SVG_ATTRIBUTE_POD_TYPE`, `floats` the value's float words (FLOAT/COLOR/MATRIX/VIEWBOX/
    /// LENGTH).
    SvgSetAttrPod {
        elem: String,
        name: String,
        pod: u32,
        floats: Vec<f32>,
    },
    SvgSetObject {
        elem: String,
        name: String,
        object: SvgObject,
        /// When true, the harness builds the object with a SCRAMBLED value (different from `object`)
        /// and drives the setters (Update*/Set*/Remove*AtEnd, with a genuine removal) to reach
        /// `object` — so a setter that ignores its input leaves the scrambled value and diverges from
        /// the native oracle. The core sets `object` directly. Default false (the no-op re-set path).
        #[serde(default)]
        mutate: bool,
    },
    SvgAppendChild {
        parent: String,
        child: String,
    },
    SvgInsertChildBefore {
        parent: String,
        child: String,
        reference: String,
    },
    SvgRemoveChild {
        parent: String,
        child: String,
    },
    SvgReplaceChild {
        parent: String,
        new: String,
        old: String,
    },
    SvgFindById {
        doc: String,
        id: String,
        elem: String,
    },
    SvgSetViewport {
        doc: String,
        size: [f32; 2],
    },
    SvgDeserialize {
        doc: String,
        xml: String,
        elem: String,
    },
    SvgSetRoot {
        doc: String,
        elem: String,
    },
    // Exercises ID2D1SvgGlyphStyle end-to-end with fixed values (CreateSvgGlyphStyle, SetFill/GetFill,
    // SetStroke/GetStroke, GetStrokeDashesCount). Pushes two query results — defaults after create, and
    // the values after setting — so the FFI value holder is compared to native's round-trip.
    SvgGlyphStyleProbe,
    // Records a DOM read (or a deliberately-failing call's HRESULT) into the query channel.
    // `kind` ∈ serialize | attr | attr_pod | count | name | tag | specified | find_hr | attr_hr.
    SvgQuery {
        kind: String,
        #[serde(default)]
        doc: Option<String>,
        #[serde(default)]
        elem: Option<String>,
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        index: Option<u32>,
        #[serde(default)]
        pod: Option<u32>,
    },
    FillOpacityMask {
        bitmap: String,
        brush: String,
        content: u32,
        #[serde(default, deserialize_with = "num::opt_arr4")]
        dest: Option<[f32; 4]>,
        #[serde(default, deserialize_with = "num::opt_arr4")]
        source: Option<[f32; 4]>,
        /// Which COM overload: default = `ID2D1RenderTarget::FillOpacityMask` (5-arg, with `content`);
        /// `"dc"` = `ID2D1DeviceContext::FillOpacityMask` (4-arg, no `content`). Same core render.
        #[serde(default)]
        overload: Option<String>,
    },
    UnitMode {
        mode: u32,
    },
    SetTags {
        tag1: u64,
        tag2: u64,
    },
    SaveState {
        id: String,
    },
    RestoreState {
        id: String,
    },
    // Non-pixel assertion: records a geometry query result into output.json.
    // `tolerance` (flattening tolerance, absent ⇒ the default) and `world` (the query's worldTransform,
    // absent ⇒ none) are optional on every geometry query, so older scenes are unchanged.
    Query {
        geometry: String,
        kind: String,
        #[serde(default, deserialize_with = "num::opt_arr2")]
        point: Option<[f32; 2]>,
        #[serde(default, deserialize_with = "num::opt_scalar")]
        tolerance: Option<f32>,
        #[serde(default, deserialize_with = "num::opt_arr6")]
        world: Option<[f32; 6]>,
    },
    // 1C.2: value-assertion queries that need more than a point (ComputePointAtLength,
    // StrokeContainsPoint, GetWidenedBounds, CompareWithGeometry). A separate op so the v2 `Query`
    // stays byte-identical. `kind` ∈ point_at_length | stroke_contains | widened_bounds | compare.
    QueryGeom {
        geometry: String,
        kind: String,
        #[serde(default, deserialize_with = "num::opt_arr2")]
        point: Option<[f32; 2]>,
        #[serde(default, deserialize_with = "num::opt_scalar")]
        length: Option<f32>,
        #[serde(default, deserialize_with = "num::opt_scalar")]
        width: Option<f32>,
        #[serde(default)]
        style: Option<String>,
        #[serde(default)]
        other: Option<String>,
        // `compare`'s inputGeometryTransform, applied to `other`. None ⇒ no transform.
        #[serde(default, deserialize_with = "num::opt_arr6")]
        transform: Option<[f32; 6]>,
        #[serde(default, deserialize_with = "num::opt_scalar")]
        tolerance: Option<f32>,
        // The worldTransform of point_at_length / stroke_contains / widened_bounds.
        #[serde(default, deserialize_with = "num::opt_arr6")]
        world: Option<[f32; 6]>,
    },
    // 1C.2: derive a new geometry from an existing one via a sink-producing operation
    // (widen/outline/simplify/combine/tessellate), recording the sink's figures+segments into
    // output.json and binding the result under `id` so later fill/draw ops can rasterize it. This
    // is how Widen/Combine/Outline/Simplify output gets both value- and pixel-tested.
    GeometryOp {
        id: String,
        source: String,
        operation: String, // widen | outline | simplify | combine | tessellate
        #[serde(default)]
        other: Option<String>,
        #[serde(default)]
        mode: Option<u32>, // combine mode / simplify option
        #[serde(default, deserialize_with = "num::opt_scalar")]
        width: Option<f32>,
        #[serde(default)]
        style: Option<String>,
        // Optional world transform for `widen` (ID2D1Geometry::Widen's worldTransform) — lets a widen
        // oracle capture the DEVICE/elliptical-pen widening under a non-uniform transform, not just the
        // local circular widen (fuzz 5289) — and `combine`'s inputGeometryTransform, applied to `other`.
        // [a,b,c,d,e,f], row-major point·matrix. None ⇒ identity.
        #[serde(default, deserialize_with = "num::opt_arr6")]
        transform: Option<[f32; 6]>,
        #[serde(default, deserialize_with = "num::opt_scalar")]
        tolerance: Option<f32>,
        // The worldTransform of outline / simplify (widen's is `transform`).
        #[serde(default, deserialize_with = "num::opt_arr6")]
        world: Option<[f32; 6]>,
    },
    // 1C-7: record ID2D1Geometry::Tessellate's triangle stream (native's fan engine) into
    // output.json for a triangle-for-triangle value comparison. No pixels.
    Tessellate {
        geometry: String,
        #[serde(default, deserialize_with = "num::opt_scalar")]
        tolerance: Option<f32>,
        #[serde(default, deserialize_with = "num::opt_arr6")]
        world: Option<[f32; 6]>,
    },
    // 1C-7: ID2D1DeviceContext1 geometry realizations. Bakes an antialiased mesh for `geometry` at
    // `tolerance` (default 0.25); `stroke_width` present ⇒ a stroked realization (widened first),
    // else a filled one. `draw_realization` replays it under the current transform with a brush.
    Realize {
        id: String,
        geometry: String,
        #[serde(default, deserialize_with = "num::opt_scalar")]
        stroke_width: Option<f32>,
        #[serde(default)]
        style: Option<String>,
        #[serde(default, deserialize_with = "num::opt_scalar")]
        tolerance: Option<f32>,
    },
    DrawRealization {
        realization: String,
        brush: String,
    },
    // --- d2d1_3 extras: image sources, BlendImage, ComputeMaximumScaleFactor ---
    // CreateImageSourceFromWic over a WIC source built from inline straight RGBA8; stored in the
    // image id-space so draw_image can use it. `loading`=D2D1_IMAGE_SOURCE_LOADING_OPTIONS,
    // `alpha`=D2D1_ALPHA_MODE of the image source.
    ImageSource {
        id: String,
        size: [u32; 2],
        rgba: Vec<u8>,
        loading: u32,
        alpha: u32,
    },
    // CreateTransformedImageSource over an image-source id: orientation + scale + interpolation.
    TransformedImageSource {
        id: String,
        source: String,
        orientation: u32,
        scale: [f32; 2],
        interpolation: u32,
        options: u32,
    },
    // DC6 BlendImage: composite an image (bitmap/image-source id) over the target with a blend mode.
    BlendImage {
        image: String,
        mode: u32,
        offset: [f32; 2],
        #[serde(default)]
        target_rect: Option<[f32; 4]>,
        interpolation: u32,
    },
    // Records D2D1ComputeMaximumScaleFactor(matrix) into output.json's queries (non-pixel).
    MaxScaleFactor {
        matrix: [f32; 6],
    },
    // Phase 2D-2: build an IDWriteTextLayout and record native layout results — the
    // DWRITE_TEXT_METRICS, per-line metrics, and every glyph run DirectWrite emits
    // (glyph ids, advances, offsets, cluster map, baseline origin) — into output.json
    // `queries`, so the Rust layout/shaping engine can be checked against native. Pure
    // query: draws nothing. `wrap`/`align`/`para`/`reading`/`flow` default to the
    // IDWriteTextLayout defaults DrawText uses.
    TextLayout {
        text: String,
        font: String,
        #[serde(deserialize_with = "num::scalar")]
        size: f32,
        #[serde(deserialize_with = "num::scalar")]
        max_width: f32,
        #[serde(deserialize_with = "num::scalar")]
        max_height: f32,
        weight: u32,
        style: u32,
        stretch: u32,
        locale: String,
        measuring: u32,
        #[serde(default)]
        wrap: u32,
        #[serde(default)]
        align: u32,
        #[serde(default)]
        para: u32,
        #[serde(default)]
        reading: u32,
        #[serde(default)]
        flow: u32,
        // 2D-2 pt3 additive probes (omitted by every pt1/pt2 scene → the results below are
        // then empty/absent and those scenes' JSON is byte-identical):
        // trimming (SetTrimming + optional ellipsis sign), hit-test points/ranges, fixed-size
        // inline-object placeholders, and typography feature ranges reaching the shaper.
        #[serde(default)]
        trimming: Option<Trimming>,
        #[serde(default)]
        hit_test_points: Vec<[f32; 2]>,
        #[serde(default)]
        hit_test_ranges: Vec<[u32; 2]>,
        #[serde(default)]
        inline_objects: Vec<InlineObj>,
        #[serde(default)]
        features: Vec<FeatureRange>,
        // Per-range character formatting (SetFontSize/Weight/Style/Stretch/Underline/Strikethrough/
        // FontFamilyName over sub-ranges) — the rich-text runs browsers build. Omitted ⇒ absent.
        #[serde(default)]
        ranges: Vec<RangeFormat>,
        // When `brush` is set, DrawTextLayout the computed layout at `origin` (DIP) with that brush —
        // the pixel-rendering path. Absent ⇒ metrics-only (the pt1/pt2 scenes), byte-identical.
        #[serde(default)]
        brush: Option<String>,
        #[serde(default, deserialize_with = "num::arr2")]
        origin: [f32; 2],
        /// `SetLineSpacing(method, spacing, baseline)` on the format; absent ⇒ the default spacing.
        #[serde(default)]
        line_spacing: Option<[f32; 3]>,
        /// The `CreateGdiCompatibleTextLayout` transform `[m11, m12, m21, m22]` (GDI measuring only), as DrawText
        /// lays out under the world transform; absent ⇒ identity.
        #[serde(default)]
        gdi_transform: Option<[f32; 4]>,
    },
    // Colour fonts pt2: DrawPaintGlyphRun (COLRv1 paint tree) with a self-contained COLRv1 font
    // supplied inline. Isolates the paint/brush math from DirectWrite layout (explicit baseline, no
    // usWinAscent) and the glyph-coverage gate (the font's outlines are pixel-aligned rectangles ⇒
    // flat {0,255} coverage in any raster mode). `font` is the TTF bytes; `text` resolves glyph ids
    // + design advances from its cmap; `color` is the default-fill (foreground) brush; `palette` is
    // the CPAL index.
    ColorGlyphRun {
        font: Vec<u8>,
        text: String,
        em_size: f32,
        baseline: [f32; 2],
        #[serde(default)]
        palette: u32,
        color: [f32; 4],
        // Draw through ID2D1DeviceContext7::DrawGlyphRunWithColorSupport instead of DrawPaintGlyphRun
        // (same color-glyph path; both render the COLR run).
        #[serde(default)]
        with_color_support: bool,
    },
    /// Draw a color-bitmap (sbix / CBDT / CBLC) glyph run from an inline font through
    /// `ID2D1DeviceContext4::DrawColorBitmapGlyphRun` — browsers draw sbix/CBDT color web fonts here.
    /// `glyph_format` selects the `DWRITE_GLYPH_IMAGE_FORMATS` (png/jpeg/tiff), default PNG.
    ColorBitmapGlyphRun {
        font: Vec<u8>,
        text: String,
        em_size: f32,
        baseline: [f32; 2],
        #[serde(default)]
        glyph_format: String,
    },
    /// Draw an OT-SVG glyph run from an inline font through `ID2D1DeviceContext4::DrawSvgGlyphRun`
    /// (OT-SVG color web fonts). `color` is the context foreground brush SVG `context-fill` references.
    SvgGlyphRun {
        font: Vec<u8>,
        text: String,
        em_size: f32,
        baseline: [f32; 2],
        color: [f32; 4],
    },
    /// `ID2D1DeviceContext4::GetColorBitmapGlyphImage` for each glyph, then `SetTransform(glyphTransform)`
    /// + `DrawImage(glyphImage)` — the single-glyph image form of `ColorBitmapGlyphRun`, so the output
    /// must match it. Verifies the returned image + glyph→world transform.
    ColorBitmapGlyphImage {
        font: Vec<u8>,
        text: String,
        em_size: f32,
        baseline: [f32; 2],
        #[serde(default)]
        glyph_format: String,
    },
    /// `ID2D1DeviceContext4::GetSvgGlyphImage` for each glyph, then `SetTransform(glyphTransform)` +
    /// `DrawImage(glyphCommandList)` — the image (command-list) form of `SvgGlyphRun`; the output must
    /// match it. `color` is the `context-fill` foreground brush.
    SvgGlyphImage {
        font: Vec<u8>,
        text: String,
        em_size: f32,
        baseline: [f32; 2],
        color: [f32; 4],
    },
    /// Draw a monochrome glyph run by explicit glyph ids from an inline font, under an explicit world
    /// transform — reproduces a COLR paint-tree leaf's `DrawGlyphRun(em=1)` render through the public API
    /// (set a bitmap target at ss size + the leaf's ss-scale/residual transform + AA/rendering mode, then
    /// capture native's placed ss buffer). Analysis only.
    GlyphRunInline {
        font: Vec<u8>,
        glyphs: Vec<u16>,
        em_size: f32,
        baseline: [f32; 2],
        brush: String,
        #[serde(default = "mat_identity")]
        transform: [f32; 6],
        #[serde(default)]
        antialias: u32,
        #[serde(default)]
        rendering_mode: u32,
        #[serde(default)]
        grid_fit: u32,
        #[serde(default)]
        measuring: u32,
        /// Enhanced-contrast probe: Some(c) sets CreateCustomRenderingParams with that contrast;
        /// None skips SetTextRenderingParams so the DC default params are used (as the COLR paint path).
        #[serde(default)]
        contrast: Option<f32>,
    },
    // --- ffirects (this agent): bounds / rect-mapping / copy queries ---
    // ID2D1DeviceContext::GetGlyphRunWorldBounds on a run built exactly like `GlyphRun`.
    GlyphBounds {
        text: String,
        font: String,
        size: f32,
        #[serde(default = "weight_normal")]
        weight: u32,
        #[serde(default)]
        style: u32,
        baseline: [f32; 2],
        antialias: u32,
        measuring: u32,
        #[serde(default)]
        rendering_mode: u32,
        #[serde(default)]
        grid_fit: u32,
    },
    // GetImageLocalBounds (world=false) / GetImageWorldBounds (world=true) on an image id.
    ImageBounds {
        image: String,
        #[serde(default)]
        world: bool,
    },
    // GetEffectRequiredInputRectangles: the render output rect and the requested `(effect,index)` inputs.
    EffectRequiredInput {
        effect: String,
        output: [f32; 4],
        inputs: Vec<ReqInput>,
    },
    // InvalidateEffectInputRectangle for each entry, then GetEffectInvalidRectangles on the effect.
    EffectInvalidRects {
        effect: String,
        invalidate: Vec<InvRect>,
    },
    // Create an empty bitmap then CopyFromMemory inline straight RGBA8 into it (premultiplied + R/B
    // swapped like the `bitmap` upload); shares the bitmap/image id-space, drawable by draw_bitmap.
    CopyMemoryBitmap {
        id: String,
        size: [u32; 2],
        rgba: Vec<u8>,
    },
    // r24: a fresh zero-filled `size` bitmap `id` (the source's pixel format unless `bgra`, which forces
    // B8G8R8A8 premultiplied), then ID2D1Bitmap::CopyFromBitmap(point?, src, rect?) into it.
    CopyFromBitmap {
        id: String,
        size: [u32; 2],
        src: String,
        #[serde(default)]
        bgra: bool,
        #[serde(default)]
        point: Option<[u32; 2]>,
        #[serde(default)]
        rect: Option<[u32; 4]>,
    },
    // r24: a mid-frame readback: a CPU_READ | CANNOT_DRAW bitmap in the target's format,
    // CopyFromRenderTarget(0,0, rect?) and Map; the HRESULT and copied bytes go to the query channel.
    Readback {
        #[serde(default)]
        rect: Option<[u32; 4]>,
    },
    // r24: ID2D1RenderTarget::SetDpi mid-frame.
    SetDpi {
        #[serde(deserialize_with = "num::arr2")]
        dpi: [f32; 2],
    },
    // r24: ID2D1DeviceContext::SetRenderingControls mid-frame (D2D1_BUFFER_PRECISION, tile size).
    RenderingControls {
        precision: u32,
        tile: [u32; 2],
    },
    // census1 (coverage census, part 1): record a resource/state getter's out-parameters into the
    // query channel — the readback (`Get*`) surface scenes never otherwise exercise. `kind` selects
    // the getter group; `target` names a prior resource id for the resource groups (unused for the
    // `dc_*` device-context-state groups). Draws nothing; folds into output.json `queries`.
    GetProp {
        kind: String,
        #[serde(default)]
        target: Option<String>,
    },
    // Change an effect's property between draws (ID2D1Properties::SetValue). General — effects mutate
    // properties between draws — and it makes an effect's post-change state (e.g. HistogramOutput
    // invalidation) a real regression test.
    SetProperty {
        target: String,
        index: u32,
        value: Value,
    },
    // Rebind an effect's input `index` to image `input` (bitmap or effect) between draws
    // (ID2D1Effect::SetInput). Lets one effect instance draw with different inputs across draws, so a
    // per-draw cache keyed on pixel position (e.g. the derivative-quad cache) is exercised for stale
    // reuse — the second draw's pixels must reflect the new input, not the first draw's cached quads.
    SetInput {
        target: String,
        index: u32,
        input: String,
    },
    // Registers a transform-graph custom effect (border → offset → bounds-adjustment) under `clsid`,
    // exercising the effect-author transform + graph API. Drawn via the ordinary effect/draw_image ops.
    RegisterGraphEffect {
        clsid: String,
        offset: [i32; 2],
        border: [u32; 2],
        bounds: [i32; 4],
    },
    // Registers a 2-input blend custom effect (CreateBlendTransform + a D2D1_BLEND_DESCRIPTION).
    // `desc` = [src, dst, op, src_alpha, dst_alpha, op_alpha] (D2D1_BLEND / D2D1_BLEND_OPERATION).
    RegisterBlendEffect {
        clsid: String,
        desc: [u32; 6],
        factor: [f32; 4],
    },
    // Registers a 0-input compute custom effect (ID2D1ComputeTransform + a cs_5_0 shader that writes a
    // typed output UAV). `groups` are the dispatch group counts; `out_size` the generated image size.
    RegisterComputeEffect {
        clsid: String,
        dxbc: Vec<u8>,
        #[serde(default)]
        cbuffer: Vec<u8>,
        groups: [u32; 3],
        out_size: [i32; 2],
    },
    // Loads a font from bytes through IDWriteFactory5::CreateInMemoryFontFileLoader (the browser
    // web-font path) and draws a glyph run with the resulting face. `font` is a filename resolved
    // against D2D_FONTS (our side) or C:\Windows\Fonts (native). Records glyph indices + metrics, and
    // renders the run so the in-memory face is pixel-verified too.
    InMemoryFontRun {
        font: String,
        text: String,
        em_size: f32,
        baseline: [f32; 2],
        color: [f32; 4],
    },
    // Loads a font through app-provided custom loaders: CreateCustomFontFileReference (a custom
    // IDWriteFontFileLoader) and CreateCustomFontCollection (a custom IDWriteFontCollectionLoader),
    // comparing glyph indices from both paths and rendering the collection face.
    CustomFontRun {
        font: String,
        text: String,
        em_size: f32,
        baseline: [f32; 2],
        color: [f32; 4],
    },
    // AnalyzeContainerType + UnpackFontFile on a WOFF/WOFF2 container (`container` is a fixture
    // filename read from the working dir): decode to sfnt, build a face, compare glyph count/indices.
    UnpackFontFileTest {
        container: String,
        text: String,
    },
    // census2 (coverage census, part 2, DirectWrite): record a DirectWrite getter cluster's
    // out-parameters into the query channel — the IDWriteTextFormat / IDWriteRenderingParams /
    // IDWriteFontFace readback surface no other scene op reaches. Draws nothing. The core arm
    // (conformance.rs) records no result: these COM objects have no counterpart in the `d2d` core,
    // which only models the layout inputs it reproduces, so the getter values are verified against
    // native by the dwrite-ffi decisive run (crates/dwrite-ffi/decisive.py), exactly as the
    // text_layout query results are. `kind` ∈ format | params | fontface.
    DwriteQuery {
        kind: String,
        #[serde(default)]
        font: String,
        #[serde(default)]
        text: String,
        #[serde(default)]
        size: f32,
        #[serde(default = "weight_normal")]
        weight: u32,
        #[serde(default)]
        style: u32,
        #[serde(default)]
        stretch: u32,
        #[serde(default)]
        locale: String,
        // params kind: CreateCustomRenderingParams inputs — [gamma, enhanced_contrast,
        // grayscale_enhanced_contrast, cleartype_level] and [pixel_geometry, rendering_mode, grid_fit].
        #[serde(default)]
        params: [f32; 4],
        #[serde(default)]
        modes: [u32; 3],
        // fontface kind: ppem for the GDI-compatible metrics/advances readback.
        #[serde(default)]
        ppem: f32,
    },
    // census2 (coverage census, part 2, WIC): record a WIC component-info / capability getter
    // cluster's out-parameters into the query channel — the IWICBitmapDecoderInfo / IWICBitmap /
    // IWICBitmapLock / IWICFormatConverter readback no other scene op reaches. Draws nothing; the
    // core arm records nothing (verified against native like the dwrite query results).
    // `kind` ∈ decoderinfo | bitmap | converter.
    WicQuery {
        kind: String,
    },
}

/// One requested input for `EffectRequiredInput` (a `D2D1_EFFECT_INPUT_DESCRIPTION`).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReqInput {
    pub effect: String,
    pub index: u32,
    pub rect: [f32; 4],
}

/// One invalidation for `EffectInvalidRects`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InvRect {
    pub input: u32,
    pub rect: [f32; 4],
}

/// DWRITE_TRIMMING + whether to install an ellipsis trimming sign (`CreateEllipsisTrimmingSign`).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Trimming {
    pub granularity: u32, // DWRITE_TRIMMING_GRANULARITY: 0 none, 1 character, 2 word
    #[serde(default)]
    pub delimiter: u32,
    #[serde(default)]
    pub delimiter_count: u32,
    #[serde(default)]
    pub ellipsis: bool,
}

/// A fixed-size inline-object placeholder set over `[position, position+length)` via
/// `IDWriteTextLayout::SetInlineObject`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InlineObj {
    pub position: u32,
    pub length: u32,
    #[serde(deserialize_with = "num::arr2")]
    pub size: [f32; 2], // width, height (DIPs); nan/inf-aware like every other scene coordinate
    #[serde(deserialize_with = "num::scalar")]
    pub baseline: f32,
    #[serde(default)]
    pub supports_sideways: bool,
}

/// One typography feature over `[position, position+length)` (`IDWriteTypography::AddFontFeature`).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureRange {
    pub position: u32,
    pub length: u32,
    pub tag: String, // 4-char OpenType tag, e.g. "liga"
    pub value: u32,
}

/// Per-range character formatting over `[position, position+length)` on an IDWriteTextLayout: the
/// SetFontSize / SetFontWeight / SetFontStyle / SetFontStretch / SetUnderline / SetStrikethrough /
/// SetFontFamilyName setters browsers drive for rich-text runs (bold/italic/sized/underlined spans).
/// Each `None` field skips its setter.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RangeFormat {
    pub position: u32,
    pub length: u32,
    #[serde(default, deserialize_with = "num::opt_scalar")]
    pub size: Option<f32>,
    #[serde(default)]
    pub weight: Option<u32>,
    #[serde(default)]
    pub style: Option<u32>,
    #[serde(default)]
    pub stretch: Option<u32>,
    #[serde(default)]
    pub underline: Option<bool>,
    #[serde(default)]
    pub strikethrough: Option<bool>,
    #[serde(default)]
    pub family: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stop {
    #[serde(deserialize_with = "num::scalar")]
    pub position: f32,
    pub color: [f32; 4],
}

/// An SVG attribute object value for `svg_set_object` — created via `ID2D1SvgDocument::Create*`
/// then applied with `ID2D1SvgElement::SetAttributeValue`, exercising the four value interfaces.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum SvgObject {
    Paint {
        paint_type: u32,
        #[serde(default)]
        color: Option<[f32; 4]>,
        #[serde(default)]
        id: String,
    },
    Points {
        points: Vec<[f32; 2]>,
    },
    Path {
        commands: Vec<u32>,
        data: Vec<f32>,
    },
    Dashes {
        dashes: Vec<f32>,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Shape {
    Rect {
        #[serde(deserialize_with = "num::arr4")]
        rect: [f32; 4],
    },
    Rounded {
        #[serde(deserialize_with = "num::arr4")]
        rect: [f32; 4],
        #[serde(deserialize_with = "num::arr2")]
        radius: [f32; 2],
    },
    Ellipse {
        #[serde(deserialize_with = "num::arr2")]
        center: [f32; 2],
        #[serde(deserialize_with = "num::arr2")]
        radius: [f32; 2],
    },
    Path { fill: u32, figures: Vec<Figure> },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Figure {
    #[serde(deserialize_with = "num::arr2")]
    pub start: [f32; 2],
    pub filled: bool,
    pub closed: bool,
    pub segments: Vec<Segment>,
    /// Per-segment `D2D1_PATH_SEGMENT` flags (aligned with `segments`; absent/short ⇒ NONE). Set via
    /// `ID2D1GeometrySink::SetSegmentFlags` before each segment: 1 = FORCE_UNSTROKED, 2 =
    /// FORCE_ROUND_LINE_JOIN.
    #[serde(default)]
    pub seg_flags: Vec<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Segment {
    Line {
        #[serde(deserialize_with = "num::arr2")]
        to: [f32; 2],
    },
    Cubic {
        #[serde(deserialize_with = "num::arr2")]
        p1: [f32; 2],
        #[serde(deserialize_with = "num::arr2")]
        p2: [f32; 2],
        #[serde(deserialize_with = "num::arr2")]
        to: [f32; 2],
    },
    Quadratic {
        #[serde(deserialize_with = "num::arr2")]
        control: [f32; 2],
        #[serde(deserialize_with = "num::arr2")]
        to: [f32; 2],
    },
    Arc {
        #[serde(deserialize_with = "num::arr2")]
        to: [f32; 2],
        #[serde(deserialize_with = "num::arr2")]
        radius: [f32; 2],
        #[serde(deserialize_with = "num::scalar")]
        rotation: f32,
        sweep: u32,
        size: u32,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Property {
    pub index: u32,
    pub value: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InkSegment {
    #[serde(deserialize_with = "num::arr3")]
    pub p1: [f32; 3],
    #[serde(deserialize_with = "num::arr3")]
    pub p2: [f32; 3],
    #[serde(deserialize_with = "num::arr3")]
    pub end: [f32; 3],
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshPatch {
    #[serde(deserialize_with = "num::arr2x16")]
    pub points: [[f32; 2]; 16],
    pub colors: [[f32; 4]; 4],
    pub extend: [u32; 4],
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sprite {
    #[serde(deserialize_with = "num::arr4")]
    pub dest: [f32; 4],
    #[serde(default, deserialize_with = "num::opt_arr4")]
    pub source: Option<[f32; 4]>,
    pub color: Option<[f32; 4]>,
    #[serde(default, deserialize_with = "num::opt_arr6")]
    pub transform: Option<[f32; 6]>,
}

#[derive(Debug, Deserialize)]
#[serde(
    tag = "type",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum Value {
    // Float-valued properties accept nan/inf (gen_scene emits adversarial values; native parses the
    // scene and the effect SetValue then clamps/rejects, so read_scene must not reject at parse time).
    Float(#[serde(deserialize_with = "num::scalar")] f32),
    Uint32(u32),
    Int32(i32),
    Floats(#[serde(deserialize_with = "num::vecf")] Vec<f32>),
    Enum(u32),
    Bool(bool),
    Vector2(#[serde(deserialize_with = "num::arr2")] [f32; 2]),
    Vector3(#[serde(deserialize_with = "num::arr3")] [f32; 3]),
    Vector4(#[serde(deserialize_with = "num::arr4")] [f32; 4]),
    Matrix3x2(#[serde(deserialize_with = "num::arr6")] [f32; 6]),
    Matrix4x4(#[serde(deserialize_with = "num::arr16")] [f32; 16]),
    Matrix5x4(#[serde(deserialize_with = "num::arr20")] [f32; 20]),
    // An interface-valued property (D2D1_PROPERTY_TYPE_IUNKNOWN): the string names a color_context id.
    Iunknown(String),
}

// A D2D1_SIMPLE_COLOR_PROFILE (CreateColorContextFromSimpleColorProfile): CIE xy primaries + white
// point and a gamma selector (D2D1_GAMMA1: 0=2.2, 1=1.0, 2=PQ).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimpleProfile {
    pub red: [f32; 2],
    pub green: [f32; 2],
    pub blue: [f32; 2],
    pub white: [f32; 2],
    pub gamma: u32,
}

fn weight_normal() -> u32 {
    400
}
fn mat_identity() -> [f32; 6] {
    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
}

/// Effect-registration XML for a `RegisterShaderEffect`: the four required metadata properties and
/// `inputs` inputs, no author properties (the constant buffer is baked into the effect impl). Shared
/// by the native capture (`render.rs`) and the CPU conformance driver so both register the same
/// schema. `RegisterEffectFromString` parses this; the CPU parser counts the `<Input>` elements.
pub fn shader_effect_xml(inputs: u32) -> String {
    let inputs_xml: String = (0..inputs).map(|i| format!("<Input name='Source{i}'/>")).collect();
    format!(
        "<?xml version='1.0'?>\
<Effect>\
<Property name='DisplayName' type='string' value='FxShader'/>\
<Property name='Author' type='string' value='opend2d'/>\
<Property name='Category' type='string' value='Test'/>\
<Property name='Description' type='string' value='2B-3 shader effect'/>\
<Inputs>{inputs_xml}</Inputs>\
</Effect>"
    )
}

pub fn byte_len(size: [u32; 2]) -> Result<usize> {
    ensure!(
        size.iter().all(|&v| (1..=16384).contains(&v)),
        "dimensions must be 1..=16384"
    );
    Ok(size[0] as usize * size[1] as usize * 4)
}

/// Root of the conformance suite (scenes, oracles, ratchet, fixtures) — the git submodule at
/// `conformance/suite`. Resolution: the `D2D_SUITE` override, else the submodule in the current
/// superproject (initialised worktree), else the main checkout's copy (read-only fallback for a
/// worktree that didn't init the submodule, the same pattern as `ref/`). `None` means the suite is
/// absent (a public clone), so every runner and test that uses it skips cleanly.
pub fn suite_dir() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    let has_suite = |p: &std::path::Path| p.join("scenes").is_dir();
    if let Ok(p) = std::env::var("D2D_SUITE") {
        let p = PathBuf::from(p);
        if has_suite(&p) {
            return Some(p);
        }
    }
    // CARGO_MANIFEST_DIR is this crate (harness/); its parent is the superproject root.
    let local = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../conformance/suite");
    if has_suite(&local) {
        return Some(local);
    }
    let main = main_checkout()?.join("conformance/suite");
    has_suite(&main).then_some(main)
}

/// The main checkout of this repository — the parent of git's common dir, so a worktree finds the
/// checkout that holds the gitignored `ref/` and an initialised suite. `None` without git.
pub fn main_checkout() -> Option<std::path::PathBuf> {
    let out = std::process::Command::new("git")
        .args(["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "--path-format=absolute", "--git-common-dir"])
        .output()
        .ok()?;
    let gitdir = std::path::PathBuf::from(String::from_utf8(out.stdout).ok()?.trim());
    gitdir.parent().map(std::path::Path::to_path_buf)
}

/// The gitignored native reference data (`ref/`): the `D2D_REF` override, else this checkout's, else
/// the main checkout's. `None` when absent (a public clone), so its users skip cleanly.
pub fn ref_dir() -> Option<std::path::PathBuf> {
    let cands = [
        std::env::var_os("D2D_REF").map(std::path::PathBuf::from),
        Some(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../ref")),
        main_checkout().map(|m| m.join("ref")),
    ];
    cands.into_iter().flatten().find(|p| p.is_dir())
}

/// The native font files: the `D2D_FONTS` override, else `ref/windows/Fonts`.
pub fn fonts_dir() -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("D2D_FONTS") {
        return Some(p.into());
    }
    ref_dir().map(|r| r.join("windows/Fonts")).filter(|p| p.is_dir())
}

pub fn read_scene(path: &Path) -> Result<Scene> {
    let scene: Scene = serde_json::from_reader(File::open(path)?)
        .with_context(|| format!("parse {}", path.display()))?;
    check_scene(scene)
}

/// [`read_scene`] for a scene already in memory (the generator's self-check).
pub fn read_scene_value(v: &serde_json::Value) -> Result<Scene> {
    check_scene(serde_json::from_value(v.clone())?)
}

fn check_scene(scene: Scene) -> Result<Scene> {
    ensure!(
        (1..=2).contains(&scene.version),
        "unsupported scene version {}",
        scene.version
    );
    byte_len(scene.size)?;
    ensure!(scene.dpi > 0.0 && scene.dpi.is_finite(), "invalid DPI");
    // A frame may pop the wrong kind or end with a clip or layer open (EndDraw reports it, and its pixels
    // and HRESULT are still compared), so balance is checked inside command lists only, where a failed
    // Close would abort the scene.
    for (i, op) in scene.ops.iter().enumerate() {
        if !matches!(op, Op::PushClip { .. } | Op::PushLayer { .. } | Op::PopClip | Op::PopLayer) {
            validate_ops(std::slice::from_ref(op)).with_context(|| format!("op {i}"))?;
        }
    }
    Ok(scene)
}

// Clip/layer balance and inline bitmap sizes, recursing into command lists.
fn validate_ops(ops: &[Op]) -> Result<()> {
    validate_ops_in(ops, false)
}

// `D2D_ALLOW_UNBALANCED_LISTS` lets a probe record an unbalanced clip/layer inside a command list (the API
// allows it; the generator never emits one).
fn validate_ops_in(ops: &[Op], in_list: bool) -> Result<()> {
    let lax = in_list && std::env::var_os("D2D_ALLOW_UNBALANCED_LISTS").is_some();
    let mut stack = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        match op {
            Op::PushClip { .. } => stack.push("clip"),
            Op::PushLayer { .. } => stack.push("layer"),
            Op::PopClip => ensure!(stack.pop() == Some("clip") || lax, "op {i}: unbalanced clip"),
            Op::PopLayer => ensure!(stack.pop() == Some("layer") || lax, "op {i}: unbalanced layer"),
            Op::Bitmap { size, rgba, rgba_f32, data, format, .. } => {
                if !data.is_empty() {
                    let bpp = match format.unwrap_or(87) { 2 => 16, 10 | 11 => 8, 49 => 2, 61 | 65 => 1, _ => 4 };
                    ensure!(data.len() == byte_len(*size)? / 4 * bpp, "op {i}: bitmap raw byte count");
                } else if rgba_f32.is_empty() {
                    ensure!(rgba.len() == byte_len(*size)?, "op {i}: bitmap byte count")
                } else {
                    ensure!(rgba_f32.len() == (size[0] as usize * size[1] as usize * 4), "op {i}: bitmap float count")
                }
            }
            Op::WicBitmap { size, rgba, .. } | Op::ImageSource { size, rgba, .. } => {
                ensure!(rgba.len() == byte_len(*size)?, "op {i}: bitmap byte count")
            }
            Op::CommandList { ops, .. } => validate_ops_in(ops, true)?,
            _ => {}
        }
    }
    ensure!(stack.is_empty() || lax, "unclosed clip/layer");
    Ok(())
}

pub fn write_png(path: &Path, size: [u32; 2], rgba: &[u8]) -> Result<()> {
    ensure!(rgba.len() == byte_len(size)?, "invalid PNG byte count");
    let mut encoder = png::Encoder::new(BufWriter::new(File::create(path)?), size[0], size[1]);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(rgba)?;
    writer.finish()?;
    Ok(())
}

pub fn read_png(path: &Path) -> Result<([u32; 2], Vec<u8>)> {
    let mut decoder = png::Decoder::new(File::open(path)?);
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info()?;
    let mut data = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut data)?;
    data.truncate(info.buffer_size());
    let rgba = match info.color_type {
        png::ColorType::Rgba => data,
        png::ColorType::Rgb => data
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        png::ColorType::Grayscale => data.iter().flat_map(|&v| [v, v, v, 255]).collect(),
        png::ColorType::GrayscaleAlpha => data
            .chunks_exact(2)
            .flat_map(|p| [p[0], p[0], p[0], p[1]])
            .collect(),
        _ => bail!("unsupported PNG color type"),
    };
    Ok(([info.width, info.height], rgba))
}

// The preview PNG is straight alpha; this conversion must not be used for exact comparisons.
pub fn straight_rgba(bgra: &[u8]) -> Vec<u8> {
    bgra.chunks_exact(4)
        .flat_map(|p| {
            let a = p[3] as u32;
            let c = |v: u8| {
                if a == 0 {
                    0
                } else {
                    ((v as u32 * 255 + a / 2) / a).min(255) as u8
                }
            };
            [c(p[2]), c(p[1]), c(p[0]), p[3]]
        })
        .collect()
}
