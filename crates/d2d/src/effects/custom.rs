//! Custom effects: `RegisterEffectFromString`/`Stream` with an
//! XML property schema, `ID2D1EffectImpl`, the transform graph (`ID2D1TransformGraph`) and its
//! nodes, `ID2D1EffectContext` (nested effects, concrete transforms, resource textures, the 3D LUT,
//! shader loads), evaluated on the engine's CPU pull model.
//!
//! What runs end to end today: effects whose graph is built from CPU-evaluable nodes — a source
//! (author draws pixels), offset, border (extend), bounds-adjustment (crop), and composite (blend)
//! transforms, plus effect-input leaves. Custom **pixel/compute/vertex shaders** need `d2d-shader`
//! (phase 3C); `LoadPixelShader`/`LoadComputeShader`/`LoadVertexShader` and `CreateVertexBuffer`
//! return `E_NOTIMPL` until it lands, and a draw transform cannot bind a custom shader before then.
//! Resource textures and 3D LUTs are created and validated (the LUT feeds the built-in `LookupTable3D`
//! effect); resource textures are otherwise consumed only by the stubbed shader path.

use super::shader::ShaderNode;
use super::{composite_factors, eval_input, fold, input_bounds, Data, Effect, Input, Kind, Prop, Rect, E_INVALIDARG, INF, NOTFOUND};
use d2d_shader::exec::{Env, PixelInputs, Sampling, UavDesc};
use d2d_shader::sm4::Mem;
use d2d_shader::Shader;
use d2d_types::abi::*;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::rc::Rc;
use windows_core::{GUID, HRESULT};

/// `E_FAIL`: a pixel shader whose DXBC does not parse (`LoadPixelShader` create-failure analogue).
const E_FAIL: HRESULT = HRESULT(0x8000_4005u32 as i32);

// ============================================================================================
// 3D LUT resource (`ID2D1LookupTable3D`, `CreateLookupTable3D`)
// ============================================================================================

/// A 3D colour lookup table: `n[0]×n[1]×n[2]` premultiplied-RGBA entries laid out with the first
/// extent varying fastest in memory. Sampled trilinearly on straight RGB, alpha carried through —
/// the `LookupTable3D` effect's lookup.
///
/// Axis convention (ffi4-01): WARP's LookupTable3D shader indexes the lattice with the **blue**
/// input channel on the fastest (dim0) axis and **red** on the slowest (dim2) — the reverse of the
/// naïve "R fastest" reading. The flat buffer the harness and `CreateLookupTable3D` build is dim0
/// -fastest (`[i + j·n0 + k·n0·n1]`), so only the channel→axis mapping here needs flipping.
pub struct Lut3d {
    n: [usize; 3],
    data: Vec<[f32; 4]>,
}

impl Lut3d {
    /// A 2³ identity LUT (corners of the unit cube) — `set`-less `LookupTable3D` is a passthrough.
    /// Entry at lattice `(i, j, k)` (dim0/1/2) must output `(R=k, G=j, B=i)` so sampling with
    /// B→dim0, R→dim2 reproduces the input.
    pub fn identity() -> Self {
        let mut data = Vec::with_capacity(8);
        for k in 0..2 {
            for j in 0..2 {
                for i in 0..2 {
                    data.push([k as f32, j as f32, i as f32, 1.0]);
                }
            }
        }
        Lut3d { n: [2, 2, 2], data }
    }

    /// `CreateLookupTable3D`: `extents` are the per-dimension lattice sizes (each ≥ 2); `data` is
    /// straight or premultiplied RGBA laid out dim0-fastest, length `extents[0]*extents[1]*extents[2]`.
    pub fn new(extents: [u32; 3], data: Vec<[f32; 4]>) -> Result<Self, HRESULT> {
        let n = extents.map(|e| e as usize);
        if n.iter().any(|&e| e < 2) || data.len() != n[0] * n[1] * n[2] {
            return Err(E_INVALIDARG);
        }
        Ok(Lut3d { n, data })
    }

    /// Lattice entry at dim0/dim1/dim2 coordinates `(i, j, k)` (dim0 fastest in memory).
    fn at(&self, i: usize, j: usize, k: usize) -> [f32; 4] {
        self.data[i + j * self.n[0] + k * self.n[0] * self.n[1]]
    }

    /// Trilinear sample for a premultiplied input pixel: unpremultiply → index into the cube →
    /// trilinear interpolate → re-premultiply with the input alpha (alpha itself is not mapped).
    /// WARP drives dim0 from blue, dim1 from green, dim2 from red, sampling the LUT texture through
    /// its fixed-function trilinear sampler: per axis the lattice-space coordinate `c·(n−1)` is
    /// quantised to WARP's 8-bit sub-texel weight `q = floor(c·(n−1)·256)`, lower lattice point
    /// `q>>8` (clamped), weight `(q&0xff)/256`; the 8 corners (exact f32, the LUT is a float texture)
    /// are combined as an f32 product sum. The Q8 weight — not a full-f32 frac — is what makes the
    /// store round like native (ffi4-01; the same WARP sampler floor as [`crate::effects::sample`]).
    pub fn sample(&self, px: [f32; 4]) -> [f32; 4] {
        let a = px[3];
        let straight = if a != 0.0 { [px[0] / a, px[1] / a, px[2] / a] } else { [0.0; 3] };
        // (lower lattice point, Q8 sub-texel weight) along a dim of `n` points for channel `c`.
        let axis = |c: f32, n: usize| {
            let q = (c.clamp(0.0, 1.0) * (n as f32 - 1.0) * 256.0).floor() as i32;
            let t = (q >> 8).clamp(0, n as i32 - 2);
            (t as usize, (q - t * 256) as f32 / 256.0)
        };
        // dim0 ← blue, dim1 ← green, dim2 ← red.
        let (i0, fi) = axis(straight[2], self.n[0]);
        let (j0, fj) = axis(straight[1], self.n[1]);
        let (k0, fk) = axis(straight[0], self.n[2]);
        let (wi, wj, wk) = ([1.0 - fi, fi], [1.0 - fj, fj], [1.0 - fk, fk]);
        let mut out = [0.0f32; 4];
        for dk in 0..2 {
            for dj in 0..2 {
                for di in 0..2 {
                    let w = wi[di] * wj[dj] * wk[dk];
                    let corner = self.at(i0 + di, j0 + dj, k0 + dk);
                    for c in 0..4 {
                        out[c] += corner[c] * w;
                    }
                }
            }
        }
        [out[0] * a, out[1] * a, out[2] * a, a]
    }
}

/// A resource texture (`CreateResourceTexture`/`FindResourceTexture`): 1/2/3-D texel data plus the
/// sampler state a draw transform binds it with (`ID2D1DrawInfo::SetResourceTexture`). A custom pixel
/// shader samples it with the normalised texcoord it computes; [`ResourceTexture::sample`] maps that
/// coord through the per-axis extend mode and point/linear filter.
///
/// ponytail: the point/linear sampler here matches D3D normalised→texel addressing, not WARP's exact
/// Q8 sub-texel weight quantisation (the [`super::sample`] residual) — resource-texture sampling is
/// close but not yet pinned byte-exact to a WARP capture; upgrade when a resource-texture oracle lands.
pub struct ResourceTexture {
    pub dimensions: u32,
    pub extents: Vec<u32>,
    /// Texel bytes, interior-mutable so `ID2D1ResourceTexture::Update` can overwrite a sub-region.
    data: RefCell<Vec<u8>>,
    /// Channels per texel: 1 (`CHANNEL_DEPTH_1`) or 4 (`CHANNEL_DEPTH_4`).
    channels: usize,
    /// Bytes per channel and whether they are IEEE float, from `D2D1_BUFFER_PRECISION`.
    bpc: usize,
    float: bool,
    /// `D2D1_FILTER`: 0 (`MIN_MAG_MIP_POINT`) is point, anything else linear min/mag.
    filter: u32,
    /// Per-axis `D2D1_EXTEND_MODE` (clamp/wrap/mirror); 1-D defaults to clamp.
    extend: Vec<u32>,
}

impl ResourceTexture {
    /// Validate (as native's `CreateResourceTexture` does) and bake the sampler state.
    pub(crate) fn new(dimensions: u32, extents: &[u32], precision: u32, channel_depth: u32, filter: u32, extend_modes: &[u32], data: &[u8]) -> Result<Rc<ResourceTexture>, HRESULT> {
        if !(1..=3).contains(&dimensions) || extents.len() != dimensions as usize {
            return Err(E_INVALIDARG);
        }
        if precision == D2D1_BUFFER_PRECISION_UNKNOWN.0 as u32 || precision > D2D1_BUFFER_PRECISION_32BPC_FLOAT.0 as u32 {
            return Err(E_INVALIDARG);
        }
        let channels = if channel_depth == D2D1_CHANNEL_DEPTH_1.0 as u32 {
            1
        } else if channel_depth == D2D1_CHANNEL_DEPTH_4.0 as u32 {
            4
        } else {
            return Err(E_INVALIDARG);
        };
        if dimensions > 1 && extend_modes.len() != dimensions as usize {
            return Err(E_INVALIDARG);
        }
        // 8UNORM/8UNORM_SRGB → 1 byte; 16UNORM → 2; 16FLOAT → 2 float; 32FLOAT → 4 float.
        let (bpc, float) = match precision {
            1 | 2 => (1, false),
            3 => (2, false),
            4 => (2, true),
            _ => (4, true),
        };
        let extend = if extend_modes.is_empty() { vec![0; dimensions as usize] } else { extend_modes.to_vec() };
        Ok(Rc::new(ResourceTexture { dimensions, extents: extents.to_vec(), data: RefCell::new(data.to_vec()), channels, bpc, float, filter, extend }))
    }

    /// `ID2D1ResourceTexture::Update`: overwrite the texels in `[min, max)` per axis with tightly-packed
    /// `src`. Dimension mismatch, a region outside the extents, or too little data → `E_INVALIDARG`.
    pub fn update(&self, min: &[u32], max: &[u32], dims: u32, src: &[u8]) -> Result<(), HRESULT> {
        let d = self.dimensions as usize;
        if dims as usize != d || min.len() < d || max.len() < d {
            return Err(E_INVALIDARG);
        }
        let mut region = [1usize; 3];
        for a in 0..d {
            if min[a] > max[a] || max[a] > self.extents[a] {
                return Err(E_INVALIDARG);
            }
            region[a] = (max[a] - min[a]) as usize;
        }
        let texel = self.channels * self.bpc;
        if src.len() < region[0] * region[1] * region[2] * texel {
            return Err(E_INVALIDARG);
        }
        let mi = |a: usize| *min.get(a).unwrap_or(&0) as usize;
        let ea = |a: usize| *self.extents.get(a).unwrap_or(&1) as usize;
        let mut dst = self.data.borrow_mut();
        let mut s = 0usize;
        for k in 0..region[2] {
            for j in 0..region[1] {
                for i in 0..region[0] {
                    let lin = (mi(0) + i) + (mi(1) + j) * ea(0) + (mi(2) + k) * ea(0) * ea(1);
                    let o = lin * texel;
                    if o + texel <= dst.len() {
                        dst[o..o + texel].copy_from_slice(&src[s..s + texel]);
                    }
                    s += texel;
                }
            }
        }
        Ok(())
    }

    /// One texel's premultiplied-irrelevant RGBA (a resource texture carries raw channel data): a
    /// 1-channel texel reads as `(R,0,0,1)`, a 4-channel one as `(R,G,B,A)`. Indices fold per axis.
    /// `exact` picks the 8bpc UNORM decode: `ld` (Load) returns the format-exact `b/255` D3D specifies,
    /// while the filtered sampler uses WARP's fixed-point `b·257/65536` — these differ by up to 1 ULP,
    /// enough that a LUT read fed through `(int)(v·255)` (Clouds' permute table) is off by one.
    fn texel(&self, idx: [i32; 3], exact: bool) -> [f32; 4] {
        let mut lin = 0usize;
        let mut stride = 1usize;
        for a in 0..self.dimensions as usize {
            let i = fold(idx[a], self.extents[a] as i32, self.extend[a]) as usize;
            lin += i * stride;
            stride *= self.extents[a] as usize;
        }
        let base = lin * self.channels * self.bpc;
        let data = self.data.borrow();
        let dec = |c: usize| -> f32 {
            let o = base + c * self.bpc;
            match (self.bpc, self.float) {
                (1, _) if exact => *data.get(o).unwrap_or(&0) as f32 / 255.0,
                (1, _) => super::sample::unorm_to_f32(*data.get(o).unwrap_or(&0)),
                (2, false) => u16::from_le_bytes([data[o], data[o + 1]]) as f32 / 65535.0,
                (2, true) => half_to_f32(u16::from_le_bytes([data[o], data[o + 1]])),
                _ => f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]),
            }
        };
        if self.channels == 1 {
            [dec(0), 0.0, 0.0, 1.0]
        } else {
            [dec(0), dec(1), dec(2), dec(3)]
        }
    }

    /// `ld t#` (Texture.Load): a direct texel fetch at an integer index — format-exact decode, no
    /// filtering. Clouds reads its permute LUT this way; AddNoise its (float) noise LUT.
    pub fn load(&self, coord: [i32; 4]) -> [f32; 4] {
        self.texel([coord[0], coord[1], coord[2]], true)
    }

    /// Sample at normalised texcoord `coord` (`coord[a] ∈ [0,1]` per present axis), point or linear.
    pub fn sample(&self, coord: [f32; 4]) -> [f32; 4] {
        let d = self.dimensions as usize;
        if self.filter == 0 {
            // Point: texel whose cell contains the coordinate.
            let mut idx = [0i32; 3];
            for a in 0..d {
                idx[a] = (coord[a] * self.extents[a] as f32).floor() as i32;
            }
            return self.texel(idx, false);
        }
        // Linear: 2^d corners around `coord·extent − 0.5`, product-sum of per-axis weights.
        let mut lo = [0i32; 3];
        let mut fr = [0f32; 3];
        for a in 0..d {
            let c = coord[a] * self.extents[a] as f32 - 0.5;
            lo[a] = c.floor() as i32;
            fr[a] = c - c.floor();
        }
        let mut out = [0.0f32; 4];
        for corner in 0..(1usize << d) {
            let mut idx = [0i32; 3];
            let mut w = 1.0f32;
            for a in 0..d {
                let hi = (corner >> a) & 1 == 1;
                idx[a] = lo[a] + hi as i32;
                w *= if hi { fr[a] } else { 1.0 - fr[a] };
            }
            let t = self.texel(idx, false);
            for k in 0..4 {
                out[k] += t[k] * w;
            }
        }
        out
    }
}

/// IEEE half (`16FLOAT`) → f32 (resource-texture decode; subnormals/inf/NaN handled).
fn half_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) as u32 & 1;
    let exp = (h >> 10) as u32 & 0x1f;
    let mant = h as u32 & 0x3ff;
    let bits = match exp {
        0 if mant == 0 => sign << 31,
        0 => {
            // Subnormal: normalise.
            let mut e = -1i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | (((112 + e) as u32) << 23) | ((m & 0x3ff) << 13)
        }
        0x1f => (sign << 31) | 0x7f80_0000 | (mant << 13),
        _ => (sign << 31) | ((exp + 112) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

// ============================================================================================
// Transform graph and nodes (`ID2D1TransformGraph`, the transform-node family)
// ============================================================================================

/// A reference to a graph input slot's source: an effect input, or another node's output.
#[derive(Clone, Copy)]
enum Ref {
    EffectInput(u32),
    Node(usize),
}

/// A transform node's CPU behaviour. Shader-backed draw/compute transforms are not here — they need
/// `d2d-shader`; a source transform draws on the CPU, the rest are the concrete D2D transforms.
#[derive(Clone)]
enum NodeKind {
    /// `ID2D1SourceTransform`: author draws pixels at a scene position, logically infinite.
    Source(Rc<dyn Fn(f32, f32) -> [f32; 4]>),
    /// `ID2D1OffsetTransform`: integer-pixel shift, bit-exact.
    Offset(i32, i32),
    /// `ID2D1BorderTransform`: per-axis `D2D1_EXTEND_MODE`, input becomes logically infinite.
    Border(u32, u32),
    /// `ID2D1BoundsAdjustmentTransform` / crop: hard-clip the output to a fixed rect.
    BoundsAdjustment(Rect),
    /// Composite N inputs with a `D2D1_COMPOSITE_MODE` (the Porter-Duff path shared with the Composite
    /// effect; this is what `create_blend_transform` builds).
    Blend(u32),
    /// `CreateBlendTransform`: N inputs blended by a `D2D1_BLEND_DESCRIPTION` (the D3D11 output-merger
    /// equation), input 0 the initial destination, each later input the source over it.
    BlendDesc(BlendDesc),
    /// `ID2D1DrawTransform` with a custom pixel shader (`SetPixelShader` + `SetPixelShaderConstantBuffer`):
    /// its inputs sampled 1:1 and run through `d2d-shader` (see [`super::shader`]). The second field is
    /// the resource textures bound at their draw-info register (`SetResourceTexture`), sampled at the
    /// texcoord the shader computes.
    Shader(Rc<ShaderNode>, Rc<RefCell<Vec<(u32, Rc<ResourceTexture>)>>>),
    /// `CreateTransformNodeFromEffect`: a pre-wired sub-effect used as a leaf (0 graph inputs).
    Effect(Effect),
    /// `ID2D1ComputeTransform` with a custom compute shader (`SetComputeShader`): dispatched over its
    /// thread groups into an output image the node then reads per pixel (see [`ComputeNode`]).
    Compute(Rc<ComputeNode>),
    /// `ID2D1DrawTransform` with `SetVertexProcessing`: a custom vertex shader deforms a vertex buffer
    /// and the pixel shader shades the rasterized triangles (see [`VertexDrawNode`]).
    VertexDraw(Rc<VertexDrawNode>),
}

/// One realized `ID2D1ComputeTransform`: the `cs_5_0` shader, its user `b0` constants, input arity,
/// bound resource textures, the thread-group counts captured from `CalculateThreadgroups`, and the
/// output origin. The dispatch writes the whole output image once (lazily, cached) — the compute
/// "push" collapsed onto the engine's per-pixel pull.
///
/// ponytail: the groups are captured once at graph build (fixed-size / generator compute — shader2-09);
/// a dispatch re-sized to the live per-draw output rect needs a push-model pass, and the cache is not
/// refreshed by a bare `SetInput` without a graph rebuild. Upgrade both when a compute-effect corpus
/// exercises them.
pub struct ComputeNode {
    shader: Rc<Shader>,
    cbuffer: Vec<u8>,
    input_count: u32,
    textures: Vec<(u32, Rc<ResourceTexture>)>,
    groups: [u32; 3],
    origin: [i32; 2],
    cache: RefCell<Option<Vec<[f32; 4]>>>,
}

impl ComputeNode {
    /// Output image size: thread-group dims × dispatch group counts.
    fn size(&self) -> (i32, i32) {
        let tg = self.shader.program.thread_group;
        ((self.groups[0] * tg[0]) as i32, (self.groups[1] * tg[1]) as i32)
    }
    fn bounds(&self) -> Rect {
        let (w, h) = self.size();
        Rect { l: self.origin[0], t: self.origin[1], r: self.origin[0] + w, b: self.origin[1] + h }
    }
    /// Run the dispatch into the typed `u0` UAV and decode it to premultiplied RGBA per output texel.
    fn dispatch(&self, env: &dyn Env) -> Vec<[f32; 4]> {
        let (w, h) = self.size();
        let decl = self.shader.program.uavs[0];
        let mut uavs = [UavDesc { reg: decl.reg, kind: decl.mem, dims: [w as u32, h as u32, 1], data: vec![0u8; (w * h * 4).max(0) as usize] }];
        // D2D reserves b0 for its own compute system constant buffer; the user buffer set via
        // SetComputeShaderConstantBuffer binds at b1 (VM-verified). b0 is empty here — a generator /
        // b1-only shader does not read it (a shader that reads D2D's b0 system constants is a follow-on).
        self.shader.run_compute_bufs(env, &[&[], &self.cbuffer], self.groups, &mut uavs, &[]);
        let fmt = if let Mem::Typed(f) = decl.mem { f } else { [1u8; 4] };
        uavs[0].data.chunks_exact(4).map(|t| core::array::from_fn(|k| decode_texel(t[k], fmt[k]))).collect()
    }
    fn eval(&self, env: &dyn Env, x: i32, y: i32) -> [f32; 4] {
        let (w, h) = self.size();
        let (ox, oy) = (self.origin[0], self.origin[1]);
        if x < ox || y < oy || x >= ox + w || y >= oy + h {
            return [0.0; 4];
        }
        if self.cache.borrow().is_none() {
            *self.cache.borrow_mut() = Some(self.dispatch(env));
        }
        self.cache.borrow().as_ref().unwrap()[(y - oy) as usize * w as usize + (x - ox) as usize]
    }
}

/// Decode one typed-UAV channel byte to f32 (the executor stores 1 byte/channel): unorm `b/255`,
/// snorm, uint, else unorm — the inverse of the executor's `pack_component`.
fn decode_texel(b: u8, fmt: u8) -> f32 {
    match fmt {
        2 => (b as i8 as f32 / 127.0).clamp(-1.0, 1.0),
        4 => b as f32,
        _ => b as f32 / 255.0,
    }
}

/// Bridges the compute executor's texture reads to the engine: a resource-texture register samples
/// that texture; any other register reads the input slot's premultiplied texel (the point-pull value),
/// so a compute shader `Load`ing input `tN` at `tid.xy` gets input `N`'s pixel there.
struct ComputeSlotEnv<'a> {
    graph: &'a TransformGraph,
    node: &'a NodeDef,
    inputs: &'a [Input],
    textures: &'a [(u32, Rc<ResourceTexture>)],
}

impl Env for ComputeSlotEnv<'_> {
    fn sample(&self, resource: u32, _sampler: u32, coord: [f32; 4], _mode: Sampling) -> [f32; 4] {
        match self.textures.iter().find(|(r, _)| *r == resource) {
            Some((_, tex)) => tex.sample(coord),
            None => self.graph.eval_slot(self.node, resource as usize, self.inputs, coord[0] as i32, coord[1] as i32),
        }
    }
    fn load(&self, resource: u32, coord: [i32; 4]) -> [u32; 4] {
        let px = match self.textures.iter().find(|(r, _)| *r == resource) {
            Some((_, tex)) => tex.load(coord),
            None => self.graph.eval_slot(self.node, resource as usize, self.inputs, coord[0], coord[1]),
        };
        px.map(f32::to_bits)
    }
}

// ============================================================================================
// Vertex-processing draw transform (`SetVertexProcessing`: custom VS + vertex buffer + PS)
// ============================================================================================

/// A custom-effect vertex buffer (`CreateVertexBuffer`): interleaved vertex bytes + the per-attribute
/// layout resolved to vertex-shader input registers at draw time.
pub struct VertexBuffer {
    data: RefCell<Vec<u8>>,
    stride: usize,
    elements: Vec<VbElement>,
    mapped: Cell<bool>,
}

/// One vertex-buffer input element — its HLSL `semantic`+`index`, byte `offset` within a vertex, and
/// f32 `components` — matched to the vertex shader's input register by semantic.
pub struct VbElement {
    pub semantic: String,
    pub index: u32,
    pub offset: usize,
    pub components: usize,
}

impl VertexBuffer {
    fn count(&self) -> usize {
        if self.stride == 0 {
            0
        } else {
            self.data.borrow().len() / self.stride
        }
    }
    /// The vertex shader's input registers `v#` for vertex `i`: each VS input-signature element is
    /// filled (as raw float bits) from the buffer element matching its semantic+index.
    fn vs_inputs(&self, i: usize, sig: &[d2d_shader::dxbc::SigElem]) -> Vec<[u32; 4]> {
        let n = sig.iter().map(|e| e.register as usize + 1).max().unwrap_or(0);
        let mut v = vec![[0u32; 4]; n];
        let data = self.data.borrow();
        let base = i * self.stride;
        for s in sig {
            if let Some(el) = self.elements.iter().find(|e| e.semantic == s.semantic && e.index == s.semantic_index) {
                let mut r = [0f32; 4];
                for c in 0..el.components.min(4) {
                    let o = base + el.offset + c * 4;
                    if o + 4 <= data.len() {
                        r[c] = f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
                    }
                }
                v[s.register as usize] = r.map(f32::to_bits);
            }
        }
        v
    }

    /// `ID2D1VertexBuffer::Map`: a writable pointer to the buffer bytes + its length (the standard COM
    /// map — the caller writes in place and must not re-enter until `unmap`).
    pub fn map(&self) -> (*mut u8, usize) {
        self.mapped.set(true);
        let mut d = self.data.borrow_mut();
        (d.as_mut_ptr(), d.len())
    }

    /// `Unmap`: committing an unmapped buffer → `E_FAIL` (native).
    pub fn unmap(&self) -> Result<(), HRESULT> {
        if self.mapped.replace(false) {
            Ok(())
        } else {
            Err(E_FAIL)
        }
    }
}

/// One realized vertex-processing draw transform (see [`NodeKind::VertexDraw`]).
///
/// The VS input positions are in **scene space** (the output-rect pixel grid); the VS turns them into
/// clip space through the system cbuffer D2D binds at `b0` ([`VertexDrawNode::system_cbuffer`], native's
/// coordinate-mapping matrices), and `device()` maps that clip to the output rect. With that cbuffer
/// bound, `ffiauthor-01-vertex-deform` (a rotated quad) is byte-exact against the WARP oracle.
///
/// ponytail: w≡1 affine attribute interpolation (no perspective divide) and a single mesh with no
/// inter-triangle blend; the coverage replicates `crate::warp::fill_triangle` and the attribute
/// weights are barycentric (≤1-LSB vs WARP's value-at-origin plane) — [`raster_triangle`] is the one
/// swap site for the shared crates/d2d/src/warp attribute/texcoord pipeline (thr_aby9e4wyum). The
/// viewport is captured once at build (fixed-output); the cache is not refreshed by a bare SetInput.
pub struct VertexDrawNode {
    vs: Rc<Shader>,
    ps: Rc<Shader>,
    ps_cbuffer: Vec<u8>,
    vb: Rc<VertexBuffer>,
    input_count: u32,
    textures: Vec<(u32, Rc<ResourceTexture>)>,
    viewport: Rect,
    cache: RefCell<Option<Vec<[f32; 4]>>>,
}

impl VertexDrawNode {
    fn eval(&self, env: &VertexEnv, x: i32, y: i32) -> [f32; 4] {
        if !self.viewport.contains(x, y) {
            return [0.0; 4];
        }
        if self.cache.borrow().is_none() {
            *self.cache.borrow_mut() = Some(self.draw(env));
        }
        let w = (self.viewport.r - self.viewport.l) as usize;
        self.cache.borrow().as_ref().unwrap()[(y - self.viewport.t) as usize * w + (x - self.viewport.l) as usize]
    }

    /// The system vertex cbuffer D2D binds at `b0` (native's coordinate-mapping matrices): a
    /// scene→clip scale+bias for the output rect, then a scene→normalised texcoord transform per input.
    /// The VS reads this to turn scene-space (output-rect) vertex positions into clip space. The user
    /// `SetVertexShaderConstantBuffer` goes at `b1`, but the executor binds one cbuffer, so a VS that
    /// reads `b1` is not reproduced (ponytail: upgrade when a dual-cbuffer custom VS appears).
    fn system_cbuffer(&self, env: &VertexEnv) -> Vec<u8> {
        let (w, h) = ((self.viewport.r - self.viewport.l) as f32, (self.viewport.b - self.viewport.t) as f32);
        let (ox, oy) = (self.viewport.l as f32, self.viewport.t as f32);
        let mut cb: Vec<f32> = vec![2.0 / w, 2.0 * ox / w - 1.0, -2.0 / h, 1.0 - 2.0 * oy / h];
        for i in 0..self.input_count {
            let b = env.input_bounds(i as usize);
            let (iw, ih) = ((b.r - b.l).max(1) as f32, (b.b - b.t).max(1) as f32);
            cb.extend_from_slice(&[1.0 / iw, b.l as f32 / iw, 1.0 / ih, b.t as f32 / ih]);
        }
        cb.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// Run the VS per vertex, map to device pixels through the viewport, rasterize each triangle, and
    /// shade covered pixels with the PS at the interpolated varyings. Returns the viewport grid.
    fn draw(&self, env: &VertexEnv) -> Vec<[f32; 4]> {
        let (vl, vt) = (self.viewport.l, self.viewport.t);
        let (vw, vh) = ((self.viewport.r - vl) as usize, (self.viewport.b - vt) as usize);
        let mut grid = vec![[0.0f32; 4]; vw * vh];
        let svp = self.vs.outputs.iter().find(|e| e.system_value == 1).map(|e| e.register as usize).unwrap_or(0);
        let sys_cb = self.system_cbuffer(env);
        let verts: Vec<Vec<[f32; 4]>> = (0..self.vb.count()).map(|i| self.vs.run_vertex(env, &self.vb.vs_inputs(i, &self.vs.inputs), &sys_cb)).collect();
        if verts.is_empty() {
            return grid;
        }
        let nreg = verts[0].len();
        // Clip -> device pixel through the viewport (standard D3D: NDC [-1,1] -> [vl, vl+vw], y down).
        let device = |o: &[[f32; 4]]| {
            let c = o[svp];
            let w = if c[3] != 0.0 { c[3] } else { 1.0 };
            [vl as f32 + (c[0] / w * 0.5 + 0.5) * vw as f32, vt as f32 + (0.5 - c[1] / w * 0.5) * vh as f32]
        };
        for t in (0..verts.len().saturating_sub(2)).step_by(3) {
            let tri = [&verts[t], &verts[t + 1], &verts[t + 2]];
            let pos = [device(tri[0]), device(tri[1]), device(tri[2])];
            raster_triangle(pos, vl, vt, vw, vh, |px, py, bary| {
                let mut pv: Vec<[u32; 4]> = (0..nreg).map(|r| core::array::from_fn::<_, 4, _>(|k| bary[0] * tri[0][r][k] + bary[1] * tri[1][r][k] + bary[2] * tri[2][r][k]).map(f32::to_bits)).collect();
                pv[svp] = [(px as f32 + 0.5).to_bits(), (py as f32 + 0.5).to_bits(), 0u32, 1.0f32.to_bits()];
                grid[(py - vt) as usize * vw + (px - vl) as usize] = self.ps.run(env, &PixelInputs { inputs: &pv, cbuffer: &self.ps_cbuffer });
            });
        }
        grid
    }
}

/// Bridges the vertex/pixel shaders' texture reads to the engine: a resource-texture register samples
/// that texture; any other register point-samples the input slot at the (normalised) texcoord.
struct VertexEnv<'a> {
    graph: &'a TransformGraph,
    node: &'a NodeDef,
    inputs: &'a [Input],
    textures: &'a [(u32, Rc<ResourceTexture>)],
}

impl VertexEnv<'_> {
    /// Device bounds of input slot `i` (for the scene→texcoord transform in the system cbuffer).
    fn input_bounds(&self, i: usize) -> Rect {
        self.graph.bounds_slot(self.node, i, self.inputs)
    }
}

impl Env for VertexEnv<'_> {
    fn sample(&self, resource: u32, _sampler: u32, coord: [f32; 4], _mode: Sampling) -> [f32; 4] {
        if let Some((_, tex)) = self.textures.iter().find(|(r, _)| *r == resource) {
            return tex.sample(coord);
        }
        let b = self.graph.bounds_slot(self.node, resource as usize, self.inputs);
        let (w, h) = ((b.r - b.l).max(0), (b.b - b.t).max(0));
        if w == 0 || h == 0 {
            return [0.0; 4];
        }
        let tx = b.l + (coord[0] * w as f32).floor().clamp(0.0, (w - 1) as f32) as i32;
        let ty = b.t + (coord[1] * h as f32).floor().clamp(0.0, (h - 1) as f32) as i32;
        self.graph.eval_slot(self.node, resource as usize, self.inputs, tx, ty)
    }
    fn load(&self, resource: u32, coord: [i32; 4]) -> [u32; 4] {
        self.graph.eval_slot(self.node, resource as usize, self.inputs, coord[0], coord[1]).map(f32::to_bits)
    }
}

/// Rasterize one triangle (device-pixel positions) into `[vl, vl+vw)×[vt, vt+vh)` with WARP's snapped
/// 1/256-subpixel coverage + D3D11 top-left rule, calling `shade(px, py, barycentric)` per covered pixel.
fn raster_triangle(pos: [[f32; 2]; 3], vl: i32, vt: i32, vw: usize, vh: usize, mut shade: impl FnMut(i32, i32, [f32; 3])) {
    let snap = |v: f32| (v as f64 * 256.0).round_ties_even() as i64;
    let p: [[i64; 2]; 3] = core::array::from_fn(|i| [snap(pos[i][0]), snap(pos[i][1])]);
    let edge = |ax: i64, ay: i64, bx: i64, by: i64, cx: i64, cy: i64| (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    let area = edge(p[0][0], p[0][1], p[1][0], p[1][1], p[2][0], p[2][1]);
    if area == 0 {
        return;
    }
    let ccw = area > 0;
    let top_left = |ax: i64, ay: i64, bx: i64, by: i64| by < ay || (ay == by && bx > ax);
    let bias: [i64; 3] = core::array::from_fn(|i| {
        let (a, b) = (p[(i + 1) % 3], p[(i + 2) % 3]);
        let tl = if ccw { top_left(a[0], a[1], b[0], b[1]) } else { top_left(b[0], b[1], a[0], a[1]) };
        if tl {
            0
        } else {
            -1
        }
    });
    let ef = |i: usize, px: i64, py: i64| {
        let (a, b) = (p[(i + 1) % 3], p[(i + 2) % 3]);
        let s = edge(a[0], a[1], b[0], b[1], px, py);
        if ccw {
            s
        } else {
            -s
        }
    };
    let af = area.unsigned_abs() as f32;
    let minx = (p.iter().map(|q| q[0]).min().unwrap() >> 8).max(vl as i64);
    let maxx = ((p.iter().map(|q| q[0]).max().unwrap() + 255) >> 8).min((vl + vw as i32) as i64);
    let miny = (p.iter().map(|q| q[1]).min().unwrap() >> 8).max(vt as i64);
    let maxy = ((p.iter().map(|q| q[1]).max().unwrap() + 255) >> 8).min((vt + vh as i32) as i64);
    for y in miny..maxy {
        let sy = y * 256 + 128;
        for x in minx..maxx {
            let sx = x * 256 + 128;
            let e = [ef(0, sx, sy), ef(1, sx, sy), ef(2, sx, sy)];
            if (0..3).any(|i| e[i] + bias[i] < 0) {
                continue;
            }
            shade(x as i32, y as i32, [e[0] as f32 / af, e[1] as f32 / af, e[2] as f32 / af]);
        }
    }
}

struct NodeDef {
    kind: NodeKind,
    inputs: Vec<Option<Ref>>,
    /// `RemoveNode` tombstone: a removed node stays in the `Vec` (ids are indices) but may no longer
    /// be connected or made the output.
    removed: bool,
    /// `ID2D1TransformGraph::SetOutputBuffer` precision (`D2D1_BUFFER_PRECISION`): recorded for API
    /// fidelity but, as on native WARP, it does not change the realized intermediate precision — the
    /// device-context bufferPrecision governs — so nothing reads this back.
    #[allow(dead_code)]
    out_precision: Option<u32>,
}

/// `D2D1_BLEND_DESCRIPTION`: the D3D11 output-merger blend state a `CreateBlendTransform` applies over
/// premultiplied RGBA. Enum values are the Win32 `D2D1_BLEND` (1-based) / `D2D1_BLEND_OPERATION`.
#[derive(Clone)]
struct BlendDesc {
    inputs: u32,
    src: u32,
    dst: u32,
    op: u32,
    src_a: u32,
    dst_a: u32,
    op_a: u32,
    bf: [f32; 4],
}

impl BlendDesc {
    /// `D2D1_BLEND` → scalar factor for channel `c` (`c==3` is alpha; the `*_COLOR` factors then read
    /// the alpha component, as the output-merger does for the alpha slot).
    fn factor(b: u32, c: usize, src: [f32; 4], dst: [f32; 4], bf: [f32; 4]) -> f32 {
        match b {
            1 => 0.0,                               // ZERO
            3 => src[c],                            // SRC_COLOR
            4 => 1.0 - src[c],                      // INV_SRC_COLOR
            5 => src[3],                            // SRC_ALPHA
            6 => 1.0 - src[3],                      // INV_SRC_ALPHA
            7 => dst[3],                            // DEST_ALPHA
            8 => 1.0 - dst[3],                      // INV_DEST_ALPHA
            9 => dst[c],                            // DEST_COLOR
            10 => 1.0 - dst[c],                     // INV_DEST_COLOR
            11 if c == 3 => 1.0,                    // SRC_ALPHA_SAT (alpha: 1)
            11 => src[3].min(1.0 - dst[3]),         // SRC_ALPHA_SAT (colour)
            12 => bf[c],                            // BLEND_FACTOR
            13 => 1.0 - bf[c],                      // INV_BLEND_FACTOR
            _ => 1.0,                               // ONE (2) / unknown
        }
    }

    /// One `src over dst` step of the blend equation (no clamp — the premultiplied float result is
    /// clamped once at the UNORM store).
    fn blend(&self, src: [f32; 4], dst: [f32; 4]) -> [f32; 4] {
        core::array::from_fn(|c| {
            let (sb, db, op) = if c == 3 { (self.src_a, self.dst_a, self.op_a) } else { (self.src, self.dst, self.op) };
            match op {
                4 => src[c].min(dst[c]), // MIN — ignores factors
                5 => src[c].max(dst[c]), // MAX — ignores factors
                _ => {
                    let s = src[c] * Self::factor(sb, c, src, dst, self.bf);
                    let d = dst[c] * Self::factor(db, c, src, dst, self.bf);
                    match op {
                        2 => s - d, // SUBTRACT
                        3 => d - s, // REV_SUBTRACT
                        _ => s + d, // ADD
                    }
                }
            }
        })
    }
}

/// `ID2D1TransformGraph`: the DAG an effect's `Initialize` builds. Nodes are added, their inputs wired
/// to other nodes or effect inputs, and one node marked the output (or a passthrough to an input).
/// A realized intermediate: one node slot's raw premultiplied-f32 output over a device rect, computed
/// once per draw so a neighbourhood-sampling consumer indexes it instead of re-evaluating the whole
/// upstream subgraph per tap. The stored values are the exact `eval_slot` results, so a consumer that
/// reads them and quantises per tap (as it already does) is byte-identical to the per-tap pull — this
/// only collapses redundant work, it never changes a pixel. [perf]
struct Realized {
    l: i32,
    t: i32,
    w: i32,
    px: Vec<[f32; 4]>,
}
impl Realized {
    #[inline]
    fn get(&self, x: i32, y: i32) -> [f32; 4] {
        self.px[((y - self.t) * self.w + (x - self.l)) as usize]
    }
}

/// Above this many pixels an intermediate is NOT realized (it falls back to the per-tap pull): a guard
/// against an unbounded/huge requested region allocating a giant buffer. Real plugin intermediates are
/// far smaller; the pathological chains this targets are small and deep, not large.
const MAX_REALIZE_PX: i64 = 4 << 20;

pub struct TransformGraph {
    input_count: u32,
    nodes: Vec<NodeDef>,
    output: Option<Ref>,
    /// The drawing context's bufferPrecision for the current realize (set per eval by `CustomState::eval`);
    /// the governing term for this graph's shader-input precision. Default 8bpc.
    render_precision: core::cell::Cell<u32>,
    /// Per-draw memo of realized intermediates, keyed by `(consuming node index, input slot)`. Cleared
    /// before every draw by `clear_shader_caches` (which `prepare_for_render` calls down the whole input
    /// chain), so it can never serve a value from a draw whose inputs or constants differ. [perf]
    realize_cache: RefCell<HashMap<(usize, usize), Rc<Realized>>>,
}

/// A node handle returned by `add_node` / the `EffectContext` factories.
#[derive(Clone, Copy)]
pub struct NodeId(usize);

impl TransformGraph {
    fn new(input_count: u32) -> Self {
        TransformGraph { input_count, nodes: Vec::new(), output: None, render_precision: core::cell::Cell::new(1), realize_cache: RefCell::new(HashMap::new()) }
    }

    /// `GetInputCount`: the effect input count passed into the graph.
    pub fn get_input_count(&self) -> u32 {
        self.input_count
    }

    /// Drop every shader node's per-draw derivative-quad cache and the realized-intermediate memo
    /// (called from `prepare_for_render`, which recurses the whole input chain).
    pub(super) fn clear_shader_caches(&self) {
        for n in &self.nodes {
            if let NodeKind::Shader(sn, _) = &n.kind {
                sn.clear_quad_cache();
            }
        }
        self.realize_cache.borrow_mut().clear();
    }

    /// The number of effect inputs the built graph actually references: the highest `EffectInput(i)`
    /// index wired into any live node slot or the output, + 1 (0 if the graph is self-sourced). This,
    /// not the declared XML/`SetInputCount` input count, is what a draw must have bound — a
    /// self-contained generator (nested `Flood`, an internal `Source`) references none and renders with
    /// no input set, while a graph that wires an effect input needs it present (native latches
    /// `E_INVALIDARG` for a referenced-but-unset input). [G5]
    fn referenced_input_count(&self) -> u32 {
        self.nodes
            .iter()
            .filter(|n| !n.removed)
            .flat_map(|n| n.inputs.iter().flatten().copied())
            .chain(self.output)
            .filter_map(|r| match r {
                Ref::EffectInput(i) => Some(i + 1),
                Ref::Node(_) => None,
            })
            .max()
            .unwrap_or(0)
    }

    fn arity(kind: &NodeKind) -> usize {
        match kind {
            NodeKind::Source(_) | NodeKind::Effect(_) => 0,
            NodeKind::Compute(cn) => cn.input_count as usize,
            NodeKind::VertexDraw(vd) => vd.input_count as usize,
            NodeKind::Offset(..) | NodeKind::Border(..) | NodeKind::BoundsAdjustment(_) => 1,
            NodeKind::Blend(_) => 2,
            NodeKind::BlendDesc(b) => b.inputs as usize,
            NodeKind::Shader(sn, _) => sn.input_count() as usize,
        }
    }

    /// `AddNode`: register a transform node; its input slots start unconnected.
    pub fn add_node(&mut self, t: Transform) -> NodeId {
        let inputs = vec![None; Self::arity(&t.0)];
        self.nodes.push(NodeDef { kind: t.0, inputs, removed: false, out_precision: None });
        NodeId(self.nodes.len() - 1)
    }

    /// `ID2D1TransformGraph::SetOutputBuffer`: pin a node's output-buffer precision (the top-priority
    /// term in Direct2D's precision rule). `node` absent/removed → `E_INVALIDARG`.
    pub fn set_output_buffer(&mut self, node: NodeId, precision: u32) -> Result<(), HRESULT> {
        match self.nodes.get_mut(node.0) {
            Some(n) if !n.removed => {
                n.out_precision = Some(precision);
                Ok(())
            }
            _ => Err(E_INVALIDARG),
        }
    }

    /// `RemoveNode`: tombstone a node — ids are indices, so it stays in the `Vec` but can no longer be
    /// connected or made the output. Node absent / already removed → `E_INVALIDARG`.
    pub fn remove_node(&mut self, node: NodeId) -> Result<(), HRESULT> {
        let n = self.nodes.get_mut(node.0).ok_or(E_INVALIDARG)?;
        if n.removed {
            return Err(E_INVALIDARG);
        }
        n.removed = true;
        Ok(())
    }

    /// `ConnectNode`: wire `from.output → to.input[index]`.
    pub fn connect_node(&mut self, from: NodeId, to: NodeId, index: u32) -> Result<(), HRESULT> {
        if !matches!(self.nodes.get(from.0), Some(n) if !n.removed) {
            return Err(E_INVALIDARG);
        }
        self.set_slot(to, index, Ref::Node(from.0))
    }

    /// `ConnectToEffectInput`: wire `effectInput[i] → to.input[index]`.
    pub fn connect_to_effect_input(&mut self, effect_input: u32, to: NodeId, index: u32) -> Result<(), HRESULT> {
        if effect_input >= self.input_count {
            return Err(E_INVALIDARG);
        }
        self.set_slot(to, index, Ref::EffectInput(effect_input))
    }

    fn set_slot(&mut self, to: NodeId, index: u32, r: Ref) -> Result<(), HRESULT> {
        let node = self.nodes.get_mut(to.0).ok_or(E_INVALIDARG)?;
        if node.removed {
            return Err(E_INVALIDARG);
        }
        *node.inputs.get_mut(index as usize).ok_or(E_INVALIDARG)? = Some(r);
        Ok(())
    }

    /// `SetOutputNode`: mark the graph's output node.
    pub fn set_output_node(&mut self, node: NodeId) -> Result<(), HRESULT> {
        if !matches!(self.nodes.get(node.0), Some(n) if !n.removed) {
            return Err(E_INVALIDARG);
        }
        self.output = Some(Ref::Node(node.0));
        Ok(())
    }

    /// `SetSingleTransformNode`: clear, add `t`, wire effect input i → node input i for all i, mark
    /// it the output. The node arity must equal the effect input count.
    pub fn set_single_transform_node(&mut self, t: Transform) -> Result<(), HRESULT> {
        self.set_single_transform_node_with_precision(t, None)
    }

    /// As [`set_single_transform_node`], but also pins the node's output-buffer precision (SetOutputBuffer)
    /// when `precision` is `Some` — e.g. a custom effect that renders a float intermediate.
    pub fn set_single_transform_node_with_precision(&mut self, t: Transform, precision: Option<u32>) -> Result<(), HRESULT> {
        if Self::arity(&t.0) != self.input_count as usize {
            return Err(E_INVALIDARG);
        }
        self.clear();
        let id = self.add_node(t);
        for i in 0..self.input_count {
            self.connect_to_effect_input(i, id, i)?;
        }
        if let Some(p) = precision {
            self.set_output_buffer(id, p)?;
        }
        self.set_output_node(id)
    }

    /// `SetPassthroughGraph`: route effect input `i` straight to the output (no pass).
    pub fn set_passthrough_graph(&mut self, i: u32) -> Result<(), HRESULT> {
        if i >= self.input_count {
            return Err(E_INVALIDARG);
        }
        self.output = Some(Ref::EffectInput(i));
        Ok(())
    }

    /// `Clear`: remove all nodes and connections.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.output = None;
    }

    // --- evaluation (the pull model; see effects/mod.rs) --------------------------------------

    fn eval_ref(&self, r: Ref, inputs: &[Input], x: i32, y: i32) -> [f32; 4] {
        match r {
            Ref::EffectInput(i) => inputs.get(i as usize).map_or([0.0; 4], |inp| eval_input(inp, x, y)),
            Ref::Node(n) => self.eval_node(n, inputs, x, y),
        }
    }

    fn eval_slot(&self, node: &NodeDef, slot: usize, inputs: &[Input], x: i32, y: i32) -> [f32; 4] {
        node.inputs.get(slot).and_then(|s| *s).map_or([0.0; 4], |r| self.eval_ref(r, inputs, x, y))
    }

    /// Realize node `n`'s input `slot` over the device rect `b` once (memoized for this draw), so a
    /// neighbourhood-sampling shader reads each upstream pixel from the buffer instead of re-evaluating
    /// the whole subgraph per tap. `None` (fall back to the per-tap pull) when the region is empty or
    /// larger than [`MAX_REALIZE_PX`]. The buffer holds the exact `eval_slot` values, so the caller's
    /// per-tap quantise stays byte-identical to the pull. [perf]
    fn realized_slot(&self, n: usize, slot: usize, node: &NodeDef, inputs: &[Input], b: Rect) -> Option<Rc<Realized>> {
        let (w64, h64) = ((b.r - b.l) as i64, (b.b - b.t) as i64);
        if w64 <= 0 || h64 <= 0 || w64 * h64 > MAX_REALIZE_PX {
            return None;
        }
        if let Some(r) = self.realize_cache.borrow().get(&(n, slot)) {
            return Some(r.clone());
        }
        let (w, h) = (w64 as usize, h64 as usize);
        let mut px = vec![[0.0f32; 4]; w * h];
        for (yy, row) in px.chunks_exact_mut(w).enumerate() {
            for (xx, p) in row.iter_mut().enumerate() {
                *p = self.eval_slot(node, slot, inputs, b.l + xx as i32, b.t + yy as i32);
            }
        }
        let r = Rc::new(Realized { l: b.l, t: b.t, w: w64 as i32, px });
        self.realize_cache.borrow_mut().insert((n, slot), r.clone());
        Some(r)
    }

    fn eval_node(&self, n: usize, inputs: &[Input], x: i32, y: i32) -> [f32; 4] {
        let node = &self.nodes[n];
        match &node.kind {
            NodeKind::Source(f) => f(x as f32 + 0.5, y as f32 + 0.5),
            NodeKind::Offset(dx, dy) => self.eval_slot(node, 0, inputs, x - dx, y - dy),
            NodeKind::Border(ex, ey) => {
                let b = self.bounds_slot(node, 0, inputs);
                let fx = b.l + fold(x - b.l, b.width() as i32, *ex);
                let fy = b.t + fold(y - b.t, b.height() as i32, *ey);
                self.eval_slot(node, 0, inputs, fx, fy)
            }
            NodeKind::BoundsAdjustment(rect) => {
                if rect.contains(x, y) {
                    self.eval_slot(node, 0, inputs, x, y)
                } else {
                    [0.0; 4]
                }
            }
            NodeKind::Blend(mode) => {
                let mut dst = self.eval_slot(node, 0, inputs, x, y);
                for s in 1..node.inputs.len() {
                    let src = self.eval_slot(node, s, inputs, x, y);
                    let (sf, df) = composite_factors(*mode, src[3], dst[3]);
                    dst = core::array::from_fn(|c| sf * src[c] + df * dst[c]);
                }
                dst
            }
            NodeKind::BlendDesc(b) => {
                let mut dst = self.eval_slot(node, 0, inputs, x, y);
                for s in 1..node.inputs.len() {
                    let src = self.eval_slot(node, s, inputs, x, y);
                    dst = b.blend(src, dst);
                }
                dst
            }
            // Input `i` → texture register `ti`; a resource-texture register is sampled at the
            // texcoord the shader computed, an input slot at the device-pixel sample position `coord`
            // (`.xy`; see shader.rs). A tap on the current pixel centre is the 1:1 point-pull (simple
            // input, or a neighbourhood's centre); any other tap (a complex input's neighbourhood
            // read, `SampleInputAtOffset/AtPosition`) is WARP's fixed-point bilinear over the slot's
            // 8bpc texels. Inside the slot's bounds a texel is the input's own pixel. Past them, native
            // pads the realized input with transparent on each side where the transform's requested
            // input rect reaches past the bounds (an inflated or infinite request, ComputeSharp's
            // complex input) and edge-extends (clamps) on a side where it does not (the 1:1 identity).
            // ponytail: the request is taken for the node's whole output rect, not the drawn region
            // (a clipped draw whose request stops short of a bounds edge would still pad there), and an
            // identity request always clamps (a smaller input under a larger union is unprobed).
            NodeKind::Shader(sn, textures) => {
                // A shared cell: the app may bind a texture after the graph is built (ComputeSharp's
                // resource texture manager binds when its property is set, after SetDrawInfo).
                let textures = textures.borrow();
                // Each input's device rect (so the shader reads NORMALIZED texcoords, the default VS's
                // `float_plane` map) and TEXCOORD-normalizing size (content for a bitmap, the realization
                // atlas for an intermediate — see `slot_norm_dims`). Computed once per draw and reused by
                // every pixel: both are fixed for the draw, so a shader evaluated at every pixel of a layer
                // no longer re-walks the input bounds and re-allocates these per pixel. `fill_inputs` and
                // the sample closure both divide by the dim, so the shader's coordinate and the sampler agree.
                sn.ensure_draw_inputs(sn.input_count() as usize, |i| {
                    let b = self.bounds_slot(node, i, inputs);
                    let rect = [b.l, b.t, b.r, b.b];
                    // A simple input point-pulls at the device pixel (no texcoord normalization), and its
                    // `MapOutputRectToInputRects` may be infinite — which `slot_norm_dims` must not be
                    // handed (the content subtraction would overflow). A complex input normalizes by its
                    // realized texture size.
                    let dim = if sn.is_simple(i as u32) { (1.0, 1.0) } else { self.slot_norm_dims(rect, self.slot_is_bitmap(node, i, inputs)) };
                    (rect, dim)
                });
                let di = sn.draw_inputs();
                let (input_rects, input_dims) = (&di.0, &di.1);
                sn.eval(x, y, input_rects, input_dims, &|i, coord| match textures.iter().find(|(r, _)| *r == i) {
                Some((_, tex)) => tex.sample(coord),
                None => {
                    // Native realises the sampled input into an intermediate at the UPSTREAM buffer's
                    // precision (Direct2D's precision rule) and the shader reads that. An 8bpc intermediate
                    // (the common case, and a bitmap input) quantises to UNORM8; a float-precision
                    // upstream (SetOutputBuffer / D2D1_PROPERTY_PRECISION / float input) keeps its range.
                    let prec = self.slot_precision(node, i as usize, inputs);
                    let rect = input_rects[i as usize];
                    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                    if sn.is_simple(i) {
                        // A simple input (`__d2dinput_color`n, TRIVIAL_SAMPLING): the default VS wrote the
                        // device scene centre (`shader.rs::fill_inputs`) and the linker samples it 1:1 at
                        // this pixel, so the tap is the centre — the precision-exact point-pull. A
                        // non-centre device tap (which a simple input does not emit) falls back to a
                        // point-pull at the floored device coord.
                        if coord[0] == px && coord[1] == py {
                            return super::quantize_to(prec, self.eval_slot(node, i as usize, inputs, x, y));
                        }
                        let (dx, dy) = (coord[0].floor() as i32, coord[1].floor() as i32);
                        return super::quantize_to(prec, self.eval_slot(node, i as usize, inputs, dx, dy));
                    }
                    // A complex input normalizes by its realized texture size (content for a bitmap /
                    // non-atlased intermediate — see `slot_norm_dims`), so `coord·dim` recovers the texel
                    // at the sampler's `u·dim` scale (the rect/bitmap texture unit the custom `Sample`
                    // compiles to).
                    let (dim_x, dim_y) = input_dims[i as usize];
                    // The 1:1 identity tap (shader samples the current pixel's own normalized centre):
                    // point-pull, precision-exact. Checked FIRST — against the slot-indexed cached planes,
                    // so it costs a few mults and never walks the input bounds — the lazy path the
                    // pre-normalization code took, which a many-tap or whole-layer shader depends on.
                    let id = super::shader::eval_input_texcoord(&sn.planes_for(i, rect, (dim_x, dim_y)), rect, px, py);
                    if coord[0] == id[0] && coord[1] == id[1] {
                        return super::quantize_to(prec, self.eval_slot(node, i as usize, inputs, x, y));
                    }
                    // A non-centre tap — a shader-COMPUTED/displaced sample coordinate (the identity centre
                    // tap is point-pulled above). WARP's texture unit addresses `texel =
                    // floor((coord·dim − 0.5)·256) >> 8` in the input's own (0-based) texel space with the
                    // standard LINEAR sub-texel FLOOR (the `cvtps2dq`-round form over-applied here: it shifted
                    // a displaced sample's Q8 weight by one, +1 in the steepest channel — PdnV5 Ripple B / Wavey
                    // Transform G vs genuine Direct2D). The fetch adds the device origin, so the integer origin
                    // never perturbs the sub-texel floor ([`sample::sample_linear`]).
                    let [bl, bt, br, bb] = rect;
                    if br <= bl || bb <= bt {
                        return [0.0; 4];
                    }
                    let (lo_x, hi_x, lo_y, hi_y) = (bl, br - 1, bt, bb - 1);
                    // The requested rect: this node's output rect (its 1:1 bounds) grown by the reach.
                    let out = (0..node.inputs.len()).map(|s| self.bounds_slot(node, s, inputs)).reduce(Rect::union).unwrap_or(INF);
                    let [rl, rt, rr, rb] = sn.reach(i as usize);
                    let pad = [
                        rl > 0 && out.l.saturating_sub(rl) < bl,
                        rt > 0 && out.t.saturating_sub(rt) < bt,
                        rr > 0 && out.r.saturating_add(rr) > br,
                        rb > 0 && out.b.saturating_add(rb) > bb,
                    ];
                    let clear = |tx: i32, ty: i32| (pad[0] && tx < lo_x) || (pad[2] && tx > hi_x) || (pad[1] && ty < lo_y) || (pad[3] && ty > hi_y);
                    let (cx, cy) = (coord[0] * dim_x, coord[1] * dim_y);
                    // WARP's `Sample` floors the Q8 sub-texel address on a power-of-two realized texture
                    // dimension and rounds (cvtps2dq) any other — per axis (the built-in resample/cubic
                    // paths' rule). A direct **bitmap** input is realized at content, so a non-pow2 content
                    // dim rounds (VM-grounded byte-exact on a 24-px bitmap). An **intermediate** is in a
                    // power-of-two atlas, so it always floors — this keeps PdnV5 Ripple / Wavey (which
                    // sample a realized `EI-Effect` intermediate) byte-exact, as 04ee8129 did.
                    let is_bm = self.slot_is_bitmap(node, i as usize, inputs);
                    let (rx, ry) = if is_bm {
                        (!((br - bl) as u32).is_power_of_two(), !((bb - bt) as u32).is_power_of_two())
                    } else {
                        (false, false)
                    };
                    // Realize this input's output region once per draw so the neighbourhood taps below
                    // read a buffer instead of re-evaluating the upstream subgraph per tap (collapses a
                    // sampling chain from O(reach^depth) to O(reach·depth)); `fetch` reads the realized
                    // pixel, or re-evaluates on the fall-back when the region is too big to realize. The
                    // clamp is the input's extend (HARD): taps the `clear` guard did not zero land inside
                    // `[lo, hi]`, so the realized rect (= these bounds) always covers them. [perf]
                    let realized = self.realized_slot(n, i as usize, node, inputs, Rect { l: bl, t: bt, r: br, b: bb });
                    let fetch = |dx: i32, dy: i32| match &realized {
                        Some(r) => r.get(dx.clamp(lo_x, hi_x), dy.clamp(lo_y, hi_y)),
                        None => self.eval_slot(node, i as usize, inputs, dx.clamp(lo_x, hi_x), dy.clamp(lo_y, hi_y)),
                    };
                    if !sn.mag_linear() {
                        // POINT sampler (the `MIN_MAG_MIP_POINT` effect default): fetch the single texel
                        // the coordinate falls in — `floor(cx)` in the input's 0-based texel space — not
                        // the 2×2 bilinear. At a texel-aligned tap this equals the linear weight-0 result,
                        // so it only differs when a shader samples a fractional computed position.
                        let (dx, dy) = (bl + cx.floor() as i32, bt + cy.floor() as i32);
                        if clear(dx, dy) {
                            return [0.0; 4];
                        }
                        return super::quantize_to(prec, fetch(dx, dy));
                    }
                    if super::prec_rank(prec) == 1 {
                        // 8bpc: WARP's fixed-point bilinear over the UNORM8 intermediate (byte texels),
                        // Q8 address floored on a pow2 dim, rounded otherwise (`rx`/`ry`).
                        let q8 = |p: [f32; 4]| p.map(|c| (c * 255.0).round_ties_even().clamp(0.0, 255.0) as i32);
                        crate::effects::sample::sample_linear_sel(cx, cy, rx, ry, |tx, ty| {
                            let (dx, dy) = (bl + tx, bt + ty);
                            if clear(dx, dy) {
                                return [0; 4];
                            }
                            q8(fetch(dx, dy))
                        })
                    } else {
                        // Higher precision: float bilinear over the precision-quantised intermediate,
                        // same per-axis pow2-floor / non-pow2-round Q8 address as the 8bpc path.
                        crate::effects::sample::sample_linear_f32_sel(cx, cy, rx, ry, |tx, ty| {
                            let (dx, dy) = (bl + tx, bt + ty);
                            if clear(dx, dy) {
                                return [0.0; 4];
                            }
                            super::quantize_to(prec, fetch(dx, dy))
                        })
                    }
                }
            }, &|res, coord| {
                // `ld t#` on a bound resource texture (a noise/dither/clouds LUT): a format-exact texel
                // fetch at the integer index (UNORM decodes b/255, not the sampler's fixed-point — an
                // off-by-one there corrupts Clouds' permute reads). A non-texture load (input SRV) falls
                // back to the input slot's point-pull texel.
                match textures.iter().find(|(r, _)| *r == res) {
                    Some((_, tex)) => tex.load(coord).map(f32::to_bits),
                    None => self.eval_slot(node, res as usize, inputs, coord[0], coord[1]).map(f32::to_bits),
                }
            }, &|res| {
                // `resinfo` of a bound resource texture: its per-axis extents (a 1-D LUT's Width, etc).
                match textures.iter().find(|(r, _)| *r == res) {
                    Some((_, tex)) => core::array::from_fn(|i| if i < 3 { tex.extents.get(i).copied().unwrap_or(1) } else { 1 }),
                    None => [0; 4],
                }
            })
            }
            NodeKind::Compute(cn) => cn.eval(&ComputeSlotEnv { graph: self, node, inputs, textures: &cn.textures }, x, y),
            NodeKind::VertexDraw(vd) => vd.eval(&VertexEnv { graph: self, node, inputs, textures: &vd.textures }, x, y),
            // A built-in sub-effect node: `eval` bakes a non-point-evaluable one (blur/filter/realized
            // transform) before sampling, so this never reaches an aborting panic.
            NodeKind::Effect(e) => e.eval(x, y),
        }
    }

    fn bounds_ref(&self, r: Ref, inputs: &[Input]) -> Rect {
        match r {
            Ref::EffectInput(i) => inputs.get(i as usize).map_or(INF, input_bounds),
            Ref::Node(n) => self.bounds_node(n, inputs),
        }
    }

    fn bounds_slot(&self, node: &NodeDef, slot: usize, inputs: &[Input]) -> Rect {
        node.inputs.get(slot).and_then(|s| *s).map_or(INF, |r| self.bounds_ref(r, inputs))
    }

    /// Whether a shader input slot is a direct **bitmap** (vs an intermediate Direct2D realizes — a
    /// nested effect's output, a command list, or another transform node). Keys the realization: a
    /// bitmap is realized at its **content** size, an intermediate into a **power-of-two atlas** (the
    /// same split the built-in `Scale`/resample paths make — see [`effects::mod`] `Kind::Scale`).
    fn slot_is_bitmap(&self, node: &NodeDef, slot: usize, inputs: &[Input]) -> bool {
        matches!(node.inputs.get(slot).and_then(|s| *s), Some(Ref::EffectInput(i)) if matches!(inputs.get(i as usize), Some(Input::Bitmap(_))))
    }

    fn slot_norm_dims(&self, rect: [i32; 4], is_bitmap: bool) -> (f32, f32) {
        let [l, t, r, b] = rect;
        let (w, h) = ((r - l).max(1) as u32, (b - t).max(1) as u32);
        // A shader input's TEXCOORD normalizes by its realized texture size. Direct2D realizes a direct
        // **bitmap** input at its **content** size (`coord·content` recovers the texel 1:1; VM-grounded:
        // a 24-px bitmap read at `uv·24`, byte-exact), but an **intermediate** into the **next-power-of-
        // two** atlas (ComputeSharp Ripple / Wavey sample a realized intermediate — `EI-Effect` — at
        // `uv·pow2`). For a real displacement effect the shader scales its offset by the D2D-provided
        // `1/dim` (TEXCOORD0.zw) so the dim cancels and only the sub-texel Q8 rounding matters; the
        // rounding (floor on a pow2 dim, cvtps2dq-round otherwise) is chosen in the `Sample` call below.
        if is_bitmap { (w as f32, h as f32) } else { (w.next_power_of_two() as f32, h.next_power_of_two() as f32) }
    }

    /// Buffer precision of the source feeding `slot` — so a shader input is realised at its upstream
    /// buffer's precision. A bitmap source is 8bpc; any effect/intermediate takes this render's context
    /// precision (WARP ignores a per-transform SetOutputBuffer and does not raise for input precision).
    fn slot_precision(&self, node: &NodeDef, slot: usize, inputs: &[Input]) -> u32 {
        node.inputs.get(slot).and_then(|s| *s).map_or(self.render_precision.get(), |r| self.ref_precision(r, inputs))
    }
    fn ref_precision(&self, r: Ref, inputs: &[Input]) -> u32 {
        match r {
            Ref::EffectInput(i) => inputs.get(i as usize).map_or(self.render_precision.get(), super::input_precision),
            Ref::Node(n) => self.node_precision(n, inputs),
        }
    }
    /// An internal intermediate node's output precision is this render's context bufferPrecision (its
    /// SetOutputBuffer override is recorded but, like native, does not change the realized precision).
    fn node_precision(&self, _n: usize, _inputs: &[Input]) -> u32 {
        self.render_precision.get()
    }

    fn bounds_node(&self, n: usize, inputs: &[Input]) -> Rect {
        let node = &self.nodes[n];
        match &node.kind {
            NodeKind::Source(_) | NodeKind::Border(..) => INF,
            NodeKind::Offset(dx, dy) => {
                let b = self.bounds_slot(node, 0, inputs);
                Rect { l: b.l.saturating_add(*dx), t: b.t.saturating_add(*dy), r: b.r.saturating_add(*dx), b: b.b.saturating_add(*dy) }
            }
            NodeKind::BoundsAdjustment(rect) => self.bounds_slot(node, 0, inputs).intersect(*rect),
            NodeKind::Blend(_) | NodeKind::BlendDesc(_) => (0..node.inputs.len()).map(|s| self.bounds_slot(node, s, inputs)).reduce(Rect::union).unwrap_or(INF),
            // Draw transform `MapInputRectsToOutputRect` (identity 1:1): union of input rects, else
            // infinite for a 0-input generator.
            NodeKind::Shader(..) => (0..node.inputs.len()).map(|s| self.bounds_slot(node, s, inputs)).reduce(Rect::union).unwrap_or(INF),
            NodeKind::Compute(cn) => cn.bounds(),
            NodeKind::VertexDraw(vd) => vd.viewport,
            NodeKind::Effect(e) => e.bounds(),
        }
    }
}

/// A transform node created by the `EffectContext` factories (or [`Transform::source`]).
pub struct Transform(NodeKind);

impl Transform {
    /// An `ID2D1SourceTransform`: the author's CPU draw `(scenePos) -> premultiplied RGBA`.
    pub fn source(draw: impl Fn(f32, f32) -> [f32; 4] + 'static) -> Self {
        Transform(NodeKind::Source(Rc::new(draw)))
    }
}

// ============================================================================================
// EffectContext (`ID2D1EffectContext` given to Initialize / PrepareForRender)
// ============================================================================================

/// `ID2D1EffectContext`: the resource/transform factory handed to an effect's `Initialize`. Its stores
/// are shared, so a clone is the same context (an app may keep the interface past `Initialize`).
#[derive(Clone)]
pub struct EffectContext {
    dpi: (f32, f32),
    /// The owning factory's id, so a nested `create_effect` resolves against the same per-factory
    /// custom-effect registry.
    factory_id: u64,
    /// Pixel shaders loaded via `LoadPixelShader`, GUID-keyed (dedup R13); shared with the owning
    /// [`CustomState`] so a load survives graph rebuilds, as native's per-context store does.
    shaders: Rc<RefCell<HashMap<u128, Rc<Shader>>>>,
    /// Named resource textures (`CreateResourceTexture`/`FindResourceTexture`, GUID-keyed dedup R13),
    /// shared with [`CustomState`] the same way.
    resource_textures: Rc<RefCell<HashMap<u128, Rc<ResourceTexture>>>>,
    /// Named vertex buffers (`CreateVertexBuffer`/`FindVertexBuffer`, GUID-keyed dedup R13).
    vertex_buffers: Rc<RefCell<HashMap<u128, Rc<VertexBuffer>>>>,
}

impl EffectContext {
    fn new(dpi: (f32, f32), factory_id: u64, shaders: Rc<RefCell<HashMap<u128, Rc<Shader>>>>, resource_textures: Rc<RefCell<HashMap<u128, Rc<ResourceTexture>>>>, vertex_buffers: Rc<RefCell<HashMap<u128, Rc<VertexBuffer>>>>) -> Self {
        EffectContext { dpi, factory_id, shaders, resource_textures, vertex_buffers }
    }

    /// `GetDpi`.
    pub fn get_dpi(&self) -> (f32, f32) {
        self.dpi
    }

    /// `IsShaderLoaded`: whether a GUID has been `LoadPixelShader`'d into this context.
    pub fn is_shader_loaded(&self, id: GUID) -> bool {
        self.shaders.borrow().contains_key(&id.to_u128())
    }

    /// `CreateOffsetTransform`.
    pub fn create_offset_transform(&self, offset: [i32; 2]) -> Transform {
        Transform(NodeKind::Offset(offset[0], offset[1]))
    }
    /// `CreateBorderTransform`: per-axis `D2D1_EXTEND_MODE` (0 clamp, 1 wrap, 2 mirror).
    pub fn create_border_transform(&self, extend_x: u32, extend_y: u32) -> Transform {
        Transform(NodeKind::Border(extend_x, extend_y))
    }
    /// `CreateBoundsAdjustmentTransform`: fixed output bounds (device-pixel rect, r/b exclusive).
    pub fn create_bounds_adjustment_transform(&self, rect: [i32; 4]) -> Transform {
        Transform(NodeKind::BoundsAdjustment(Rect { l: rect[0], t: rect[1], r: rect[2], b: rect[3] }))
    }
    /// `CreateBlendTransform`: composite N inputs with a `D2D1_COMPOSITE_MODE` (CPU model, see
    /// [`NodeKind::Blend`]).
    pub fn create_blend_transform(&self, composite_mode: u32) -> Transform {
        Transform(NodeKind::Blend(composite_mode))
    }
    /// `CreateTransformNodeFromEffect`: wrap a pre-wired sub-effect as a leaf node.
    pub fn create_transform_node_from_effect(&self, effect: Effect) -> Transform {
        Transform(NodeKind::Effect(effect))
    }

    /// `CreateEffect`: a nested effect (built-in or registered), carrying this context's DPI.
    pub fn create_effect(&self, clsid: GUID) -> Result<Effect, HRESULT> {
        match super::kind_of(&clsid) {
            Some(kind) => Ok(Effect::new_in(kind, self.dpi)),
            None => create_registered(clsid, self.dpi, self.factory_id),
        }
    }

    /// `CreateResourceTexture`: validate `dimensions ∈ {1,2,3}`, precision usable (not `UNKNOWN`),
    /// channel depth set (not `DEFAULT`); multi-dim needs per-axis extend modes. Anonymous, point
    /// filter — the signature the core-level validation tests use.
    pub fn create_resource_texture(&self, dimensions: u32, extents: &[u32], precision: u32, channel_depth: u32, extend_modes: &[u32], data: &[u8]) -> Result<Rc<ResourceTexture>, HRESULT> {
        ResourceTexture::new(dimensions, extents, precision, channel_depth, 0, extend_modes, data)
    }

    /// `ID2D1EffectContext::CreateResourceTexture` with the full `D2D1_RESOURCE_TEXTURE_PROPERTIES`
    /// (filter) and an optional `resourceId`: a named texture dedups by GUID (R13 — a second create of
    /// a loaded GUID returns the first and ignores the new data); an anonymous one (`None`) is not stored.
    pub fn create_resource_texture_keyed(&self, id: Option<GUID>, dimensions: u32, extents: &[u32], precision: u32, channel_depth: u32, filter: u32, extend_modes: &[u32], data: &[u8]) -> Result<Rc<ResourceTexture>, HRESULT> {
        if let Some(id) = id {
            if let Some(t) = self.resource_textures.borrow().get(&id.to_u128()) {
                return Ok(t.clone());
            }
        }
        let tex = ResourceTexture::new(dimensions, extents, precision, channel_depth, filter, extend_modes, data)?;
        if let Some(id) = id {
            self.resource_textures.borrow_mut().insert(id.to_u128(), tex.clone());
        }
        Ok(tex)
    }

    /// `FindResourceTexture`: a GUID-keyed resource texture, else `NOTFOUND`.
    pub fn find_resource_texture(&self, id: GUID) -> Result<Rc<ResourceTexture>, HRESULT> {
        self.resource_textures.borrow().get(&id.to_u128()).cloned().ok_or(NOTFOUND)
    }

    /// `LoadPixelShader`: parse the DXBC and store it by GUID. Dedup R13 — a re-load of a GUID already
    /// loaded returns the first shader and ignores the new bytes. Unparseable DXBC → `E_FAIL`.
    pub fn load_pixel_shader(&self, id: GUID, dxbc: &[u8]) -> Result<(), HRESULT> {
        let key = id.to_u128();
        let mut m = self.shaders.borrow_mut();
        if m.contains_key(&key) {
            return Ok(());
        }
        let shader = Shader::parse(dxbc).map_err(|_| E_FAIL)?;
        m.insert(key, Rc::new(shader));
        Ok(())
    }

    /// `CreateBlendTransform`: `inputs` images blended by a `D2D1_BLEND_DESCRIPTION` (D3D11 output-merger
    /// equation). `inputs == 0` → `E_INVALIDARG`, as native's blend transform rejects it.
    #[allow(clippy::too_many_arguments)]
    pub fn create_blend_desc_transform(&self, inputs: u32, src: u32, dst: u32, op: u32, src_a: u32, dst_a: u32, op_a: u32, blend_factor: [f32; 4]) -> Result<Transform, HRESULT> {
        if inputs == 0 {
            return Err(E_INVALIDARG);
        }
        Ok(Transform(NodeKind::BlendDesc(BlendDesc { inputs, src, dst, op, src_a, dst_a, op_a, bf: blend_factor })))
    }

    /// `ID2D1DrawInfo::SetPixelShader(id)` + `SetPixelShaderConstantBuffer(cbuffer)` as one draw
    /// transform over `input_count` inputs, with any `SetResourceTexture`-bound textures. The GUID
    /// must have been `load_pixel_shader`'d, else `NOTFOUND` (R11); an opcode the executor cannot run
    /// fails here too (never a wrong pixel).
    /// `cbuffer` is a shared, interior-mutable cell (the FFI binds the SAME cell to the draw transform's
    /// `DrawInfo` so a custom effect's `PrepareForRender` can update the `b0` constants per draw). [G2]
    pub fn pixel_shader_transform(&self, id: GUID, cbuffer: Rc<RefCell<Vec<u8>>>, input_count: u32, resource_textures: Rc<RefCell<Vec<(u32, Rc<ResourceTexture>)>>>, reach: Rc<RefCell<Vec<[i32; 4]>>>, filter: u32) -> Result<Transform, HRESULT> {
        let shader = self.shaders.borrow().get(&id.to_u128()).cloned().ok_or(NOTFOUND)?;
        Ok(Transform(NodeKind::Shader(ShaderNode::new(shader, cbuffer, input_count, reach, filter)?, resource_textures)))
    }

    /// `LoadComputeShader`: parse the `cs_5_0` DXBC and store it by GUID — the same per-context
    /// GUID-keyed parse+dedup store as pixel shaders (R13); the executor runs compute the same way.
    pub fn load_compute_shader(&self, id: GUID, dxbc: &[u8]) -> Result<(), HRESULT> {
        self.load_pixel_shader(id, dxbc)
    }

    /// `ID2D1ComputeInfo::SetComputeShader(id)` as a compute draw transform over `input_count` inputs,
    /// with the bound resource textures, dispatched `groups` thread groups writing the output image at
    /// `origin`. The GUID must be loaded (else `NOTFOUND`); the shader must declare a typed output UAV
    /// at `u0` and run on the executor (else `E_FAIL`).
    pub fn compute_shader_transform(&self, id: GUID, cbuffer: Vec<u8>, input_count: u32, textures: Vec<(u32, Rc<ResourceTexture>)>, groups: [u32; 3], origin: [i32; 2]) -> Result<Transform, HRESULT> {
        let shader = self.shaders.borrow().get(&id.to_u128()).cloned().ok_or(NOTFOUND)?;
        if shader.unsupported_op().is_some() {
            return Err(E_FAIL);
        }
        if !matches!(shader.program.uavs.first().map(|d| d.mem), Some(Mem::Typed(_))) {
            return Err(E_FAIL); // an image-output compute transform writes a typed RWTexture at u0
        }
        Ok(Transform(NodeKind::Compute(Rc::new(ComputeNode { shader, cbuffer, input_count, textures, groups, origin, cache: RefCell::new(None) }))))
    }

    /// `LoadVertexShader`: parse the `vs_5_0` DXBC and store it by GUID (the same per-context store as
    /// pixel/compute shaders, R13 dedup; the executor runs it via `run_vertex`).
    pub fn load_vertex_shader(&self, id: GUID, dxbc: &[u8]) -> Result<(), HRESULT> {
        self.load_pixel_shader(id, dxbc)
    }

    /// `CreateVertexBuffer`: store an interleaved vertex buffer + its input-element layout, GUID-keyed
    /// (dedup R13 for a named buffer); anonymous (`None`) is not stored. `elements` are the per-attribute
    /// `(semantic, index, byte offset, component count)` resolved to vertex-shader input registers at draw.
    pub fn create_vertex_buffer(&self, id: Option<GUID>, data: &[u8], stride: u32, elements: Vec<VbElement>) -> Result<Rc<VertexBuffer>, HRESULT> {
        if let Some(id) = id {
            if let Some(vb) = self.vertex_buffers.borrow().get(&id.to_u128()) {
                return Ok(vb.clone());
            }
        }
        if stride == 0 {
            return Err(E_INVALIDARG);
        }
        let vb = Rc::new(VertexBuffer { data: RefCell::new(data.to_vec()), stride: stride as usize, elements, mapped: Cell::new(false) });
        if let Some(id) = id {
            self.vertex_buffers.borrow_mut().insert(id.to_u128(), vb.clone());
        }
        Ok(vb)
    }

    /// `FindVertexBuffer`: a GUID-keyed vertex buffer, else `NOTFOUND`.
    pub fn find_vertex_buffer(&self, id: GUID) -> Result<Rc<VertexBuffer>, HRESULT> {
        self.vertex_buffers.borrow().get(&id.to_u128()).cloned().ok_or(NOTFOUND)
    }

    /// `ID2D1DrawInfo::SetVertexProcessing` + `SetPixelShader`: a vertex-processing draw transform. The
    /// vertex shader `vs_id` transforms `vb`'s vertices into clip space, mapped to the `viewport` output
    /// rect; the pixel shader `ps_id` shades each rasterized pixel, sampling the `input_count` inputs +
    /// resource textures at the interpolated varyings. Both GUIDs must be loaded (else `NOTFOUND`); an
    /// opcode the executor can't run, or a vertex shader with no `SV_Position`, → `E_FAIL`.
    #[allow(clippy::too_many_arguments)]
    pub fn vertex_shader_transform(&self, vs_id: GUID, ps_id: GUID, _vs_cbuffer: Vec<u8>, ps_cbuffer: Vec<u8>, vb: Rc<VertexBuffer>, input_count: u32, textures: Vec<(u32, Rc<ResourceTexture>)>, viewport: [i32; 4]) -> Result<Transform, HRESULT> {
        // `_vs_cbuffer` is the user b1 cbuffer; the executor binds one cbuffer and the VS reads the
        // system scene→clip cbuffer at b0 ([`VertexDrawNode::system_cbuffer`]), so b1 is not bound.
        let get = |id: GUID| self.shaders.borrow().get(&id.to_u128()).cloned().ok_or(NOTFOUND);
        let vs = get(vs_id)?;
        let ps = get(ps_id)?;
        if vs.unsupported_op().is_some() || ps.unsupported_op().is_some() {
            return Err(E_FAIL);
        }
        if !vs.outputs.iter().any(|e| e.system_value == 1) {
            return Err(E_FAIL); // a vertex shader must write SV_Position
        }
        Ok(Transform(NodeKind::VertexDraw(Rc::new(VertexDrawNode { vs, ps, ps_cbuffer, vb, input_count, textures, viewport: Rect { l: viewport[0], t: viewport[1], r: viewport[2], b: viewport[3] }, cache: RefCell::new(None) }))))
    }
}

// ============================================================================================
// Property schema + the author trait
// ============================================================================================

/// The `D2D1_PROPERTY_TYPE` subset a custom effect's XML schema can declare (matches [`Prop`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PropKind {
    F32,
    U32,
    I32,
    Bool,
    V2,
    V3,
    V4,
    M32,
    M44,
    M54,
    /// A raw byte blob. ComputeSharp.D2D1 declares its pixel-shader constant buffer as
    /// `<Property name='Buffer' type='blob'>` (G1); the value arrives as [`Prop::Floats`] (the FFI
    /// decodes a `D2D1_PROPERTY_TYPE_BLOB` as little-endian f32s). Rejecting it blocked every
    /// ComputeSharp effect at `RegisterEffectFromString`.
    Blob,
    /// A COM interface pointer (`D2D1_PROPERTY_TYPE_IUNKNOWN`). ComputeSharp.D2D1 3.2.0 declares
    /// `<Property name='TransformMapper' type='iunknown'>` and `ResourceTextureManager0..15` on every
    /// pixel-shader effect (G14); rejecting the type blocked 16 effects at registration. Our DXBC
    /// executor does not consume these (no transform mapper / resource-texture binding), so the value
    /// has no core representation — the FFI accepts and ignores it (see `EffectObj::SetValue`).
    IUnknown,
}

impl PropKind {
    fn parse(ty: &str) -> Option<PropKind> {
        Some(match ty {
            "float" => PropKind::F32,
            "uint32" | "enum" => PropKind::U32,
            "int32" => PropKind::I32,
            "bool" => PropKind::Bool,
            "vector2" => PropKind::V2,
            "vector3" => PropKind::V3,
            "vector4" => PropKind::V4,
            "matrix3x2" => PropKind::M32,
            "matrix4x4" => PropKind::M44,
            "matrix5x4" => PropKind::M54,
            "blob" => PropKind::Blob,
            "iunknown" => PropKind::IUnknown,
            _ => return None,
        })
    }
    fn matches(self, v: &Prop) -> bool {
        matches!(
            (self, v),
            (PropKind::F32, Prop::F32(_))
                | (PropKind::U32, Prop::U32(_))
                | (PropKind::I32, Prop::I32(_))
                | (PropKind::Bool, Prop::Bool(_))
                | (PropKind::V2, Prop::V2(_))
                | (PropKind::V3, Prop::V3(_))
                | (PropKind::V4, Prop::V4(_))
                | (PropKind::M32, Prop::M32(_))
                | (PropKind::M44, Prop::M44(_))
                | (PropKind::M54, Prop::M54(_))
                | (PropKind::Blob, Prop::Floats(_))
        )
    }
}

/// One custom (non-system) effect property from the XML schema.
#[derive(Clone)]
struct PropDef {
    name: String,
    kind: PropKind,
}

#[derive(Clone)]
struct Schema {
    inputs: u32,
    props: Vec<PropDef>,
}

/// Read-only view of an effect's property values, handed to `EffectImpl::initialize`.
pub struct Props<'a> {
    map: &'a BTreeMap<u32, Prop>,
}

impl Props<'_> {
    pub fn get_f32(&self, index: u32, default: f32) -> f32 {
        if let Some(Prop::F32(v)) = self.map.get(&index) {
            *v
        } else {
            default
        }
    }
    pub fn get_u32(&self, index: u32, default: u32) -> u32 {
        if let Some(Prop::U32(v)) = self.map.get(&index) {
            *v
        } else {
            default
        }
    }
    pub fn get_v4(&self, index: u32, default: [f32; 4]) -> [f32; 4] {
        if let Some(Prop::V4(v)) = self.map.get(&index) {
            *v
        } else {
            default
        }
    }
}

/// `ID2D1EffectImpl`: the author object D2D drives. `initialize` builds the transform graph from the
/// current property values; `prepare_for_render` is a no-op here (no constant buffers without
/// shaders — the graph is rebuilt from properties on every `SetValue`).
pub trait EffectImpl {
    fn initialize(&self, props: &Props, ctx: &EffectContext, graph: &mut TransformGraph) -> Result<(), HRESULT>;
    /// Called once per draw, before the graph realizes (native `ID2D1EffectImpl::PrepareForRender`).
    /// Default no-op: an effect whose graph is rebuilt from its properties on every `SetValue` needs
    /// nothing here. The COM authoring bridge overrides it to drive the application effect's own
    /// `PrepareForRender`, which re-pushes its pixel-shader constant buffer through
    /// `SetPixelShaderConstantBuffer` — so a constant set AFTER `Initialize` (e.g. ComputeSharp's
    /// `Buffer` property) reaches the shader for this draw rather than the value baked at build. [G2]
    fn prepare_for_render(&self) {}
    /// Deliver a `SetValue` for author property `name` to the effect's own state (native routes it
    /// through the property's `D2D1_PROPERTY_BINDING` set-function). Default no-op: a built-from-
    /// properties effect reads its values from the graph rebuild instead. The COM bridge overrides it
    /// to call the app effect's bound setter, so a ComputeSharp `Buffer` reaches the shader via the
    /// effect's `PrepareForRender`. `data` is the raw little-endian value bytes. [G2]
    fn set_bound_value(&self, name: &str, data: &[u8]) {
        let _ = (name, data);
    }
}

/// A ready-made [`EffectImpl`] for the common single-pass pixel-shader effect: `Initialize` loads one
/// pixel shader (`LoadPixelShader`) and wires a draw transform over all `input_count` inputs
/// (`SetSingleTransformNode`). The bytecode, `b0` constant buffer and input count are fixed at
/// construction — what the `fxshader` scenes register. The DXBC must be authored/compiled fresh
/// (never copied from `ref/`); `cbuffer` is the little-endian `b0` bytes.
pub struct PixelShaderEffect {
    pub shader_id: GUID,
    pub dxbc: Vec<u8>,
    pub cbuffer: Vec<u8>,
    pub input_count: u32,
    /// SetOutputBuffer precision for the draw transform (`D2D1_BUFFER_PRECISION`); `None` ⇒ inherit.
    pub precision: Option<u32>,
    /// `MapOutputRectToInputRects` inflates the output rect by this much per input (saturating, so
    /// `i32::MAX` is infinite); `None` ⇒ the 1:1 identity.
    pub input_inflate: Option<i32>,
    /// The input sampler's `D2D1_FILTER` (the app's `SetInputDescription`); `0` = `MIN_MAG_MIP_POINT`,
    /// the effect default. See `shader::ShaderNode::mag_linear`.
    pub filter: u32,
}

impl EffectImpl for PixelShaderEffect {
    fn initialize(&self, _props: &Props, ctx: &EffectContext, graph: &mut TransformGraph) -> Result<(), HRESULT> {
        ctx.load_pixel_shader(self.shader_id, &self.dxbc)?;
        // A fixed-cbuffer effect: wrap the constant bytes in a fresh cell (never updated per draw).
        let reach = vec![[self.input_inflate.unwrap_or(0); 4]; self.input_count as usize];
        let t = ctx.pixel_shader_transform(self.shader_id, Rc::new(RefCell::new(self.cbuffer.clone())), self.input_count, Rc::default(), Rc::new(RefCell::new(reach)), self.filter)?;
        graph.set_single_transform_node_with_precision(t, self.precision)
    }
}

/// Live state of one custom effect instance: its schema, the author object, the current graph, and
/// the pixel shaders its `Initialize` loaded (kept across rebuilds, like native's per-context store).
pub struct CustomState {
    schema: Schema,
    imp: Box<dyn EffectImpl>,
    dpi: (f32, f32),
    /// The owning factory's id, for nested `create_effect` against its per-factory registry.
    factory_id: u64,
    graph: TransformGraph,
    shaders: Rc<RefCell<HashMap<u128, Rc<Shader>>>>,
    resource_textures: Rc<RefCell<HashMap<u128, Rc<ResourceTexture>>>>,
    vertex_buffers: Rc<RefCell<HashMap<u128, Rc<VertexBuffer>>>>,
}

impl CustomState {
    pub(super) fn input_count(&self) -> u32 {
        self.schema.inputs
    }

    /// `Some(n)` when the transform graph is one pixel shader over effect inputs `0..n` in order: a single draw
    /// for Direct2D's intermediate planning.
    pub(super) fn single_shader(&self) -> Option<u32> {
        let mut live = self.graph.nodes.iter().enumerate().filter(|(_, n)| !n.removed);
        let (i, n) = live.next()?;
        if live.next().is_some() || !matches!(self.graph.output, Some(Ref::Node(o)) if o == i) {
            return None;
        }
        let NodeKind::Shader(sn, _) = &n.kind else { return None };
        let count = sn.input_count();
        let wired = n.inputs.len() == count as usize && n.inputs.iter().enumerate().all(|(k, r)| matches!(r, Some(Ref::EffectInput(j)) if *j as usize == k));
        wired.then_some(count)
    }

    /// The number of effect inputs this effect's realized transform graph actually references — the
    /// draw-time requirement checked by `graph_valid`, not the declared `schema.inputs`. See
    /// [`TransformGraph::referenced_input_count`]. [G5]
    pub(super) fn referenced_input_count(&self) -> u32 {
        self.graph.referenced_input_count()
    }

    /// Drive the author effect's `PrepareForRender` before a draw (see [`EffectImpl::prepare_for_render`]).
    /// No-op unless the COM bridge overrides it. [G2]
    ///
    /// Also drops every shader node's per-draw derivative-quad cache: this draw may bind a different
    /// constant buffer or inputs than the last, so a memoized quad must not survive into it.
    pub(super) fn prepare_for_render(&self) {
        self.imp.prepare_for_render();
        self.graph.clear_shader_caches();
    }

    /// Deliver a `SetValue` for author property `name` to the effect's bound setter (see
    /// [`EffectImpl::set_bound_value`]). No-op unless the COM bridge overrides it. [G2]
    pub(super) fn set_bound_value(&self, name: &str, data: &[u8]) {
        self.imp.set_bound_value(name, data);
    }

    /// `GetPropertyIndex`: the index of a schema property by name, if any.
    pub(super) fn property_index(&self, name: &str) -> Option<u32> {
        self.schema.props.iter().position(|p| p.name == name).map(|i| i as u32)
    }

    /// `GetPropertyCount`: the number of author (schema) properties. [G2]
    pub(super) fn property_count(&self) -> u32 {
        self.schema.props.len() as u32
    }
    /// `GetPropertyName`: the author property name at `index`. [G2]
    pub(super) fn property_name(&self, index: u32) -> Option<&str> {
        self.schema.props.get(index as usize).map(|p| p.name.as_str())
    }
    /// `GetType`: the author property kind at `index`. [G2]
    pub(super) fn property_kind(&self, index: u32) -> Option<PropKind> {
        self.schema.props.get(index as usize).map(|p| p.kind)
    }

    /// Validate a `SetValue` against the schema: index in range and type matching → else
    /// `E_INVALIDARG` (as native's `SetValue` rejects it).
    pub(super) fn validate(&self, index: u32, value: &Prop) -> Result<(), HRESULT> {
        let def = self.schema.props.get(index as usize).ok_or(E_INVALIDARG)?;
        if def.kind.matches(value) {
            Ok(())
        } else {
            Err(E_INVALIDARG)
        }
    }

    /// Rebuild the transform graph from the current property values (after a `SetValue`).
    pub(super) fn rebuild(&mut self, props: &BTreeMap<u32, Prop>) -> Result<(), HRESULT> {
        let ctx = EffectContext::new(self.dpi, self.factory_id, self.shaders.clone(), self.resource_textures.clone(), self.vertex_buffers.clone());
        let mut graph = TransformGraph::new(self.schema.inputs);
        self.imp.initialize(&Props { map: props }, &ctx, &mut graph)?;
        self.graph = graph;
        Ok(())
    }

    pub(super) fn bounds(&self, inputs: &[Input]) -> Rect {
        match self.graph.output {
            Some(r) => self.graph.bounds_ref(r, inputs),
            None => INF,
        }
    }

    pub(super) fn eval(&self, inputs: &[Input], x: i32, y: i32, render_precision: u32) -> [f32; 4] {
        // The drawing context's bufferPrecision governs this graph's shader-input realize; stash it so
        // the precision queries read it without threading it through every value-eval call.
        self.graph.render_precision.set(if render_precision == 0 { 1 } else { render_precision });
        match self.graph.output {
            Some(r) => self.graph.eval_ref(r, inputs, x, y),
            None => [0.0; 4],
        }
    }
}

// ============================================================================================
// Registry (`RegisterEffectFromString`/`Stream`, keyed by CLSID; ref-counted)
// ============================================================================================

struct Registration {
    refcount: u32,
    schema: Schema,
    factory: Rc<dyn Fn() -> Box<dyn EffectImpl>>,
}

thread_local! {
    // Custom-effect registry, scoped **per factory** (outer key = `Factory::id`) as native D2D scopes
    // `RegisterEffectFromString`. Kept in a process-wide thread-local keyed by factory id rather than
    // owned by `Factory` to avoid threading an `Rc` through `Factory`→`Device`→`DeviceContext`; the id
    // (already carried by both) is the key. Each factory gets its own CLSID map, so two factories (or
    // two independently-authored effects replayed in one process, e.g. distinct per-process oracle
    // captures) never collide. See `register`.
    static REGISTRY: RefCell<HashMap<u64, HashMap<u128, Registration>>> = RefCell::new(HashMap::new());
}

/// Parse the effect-registration XML into a property schema: the input count (`<Inputs><Input/>…`)
/// and the author properties (top-level `<Property name type>` other than the reserved metadata
/// names). Reuses `roxmltree`, as the SVG DOM does.
fn parse_schema(xml: &str) -> Result<Schema, HRESULT> {
    let doc = roxmltree::Document::parse(xml).map_err(|_| E_INVALIDARG)?;
    let root = doc.root_element();
    let mut inputs = 0u32;
    let mut props = Vec::new();
    for node in root.children().filter(|n| n.is_element()) {
        match node.tag_name().name() {
            "Inputs" => inputs = node.children().filter(|c| c.is_element() && c.tag_name().name() == "Input").count() as u32,
            "Property" => {
                let name = node.attribute("name").unwrap_or_default();
                // Reserved system-metadata properties are not author properties.
                if matches!(name, "DisplayName" | "Author" | "Category" | "Description") {
                    continue;
                }
                let kind = PropKind::parse(node.attribute("type").unwrap_or_default()).ok_or(E_INVALIDARG)?;
                props.push(PropDef { name: name.to_owned(), kind });
            }
            _ => {}
        }
    }
    Ok(Schema { inputs, props })
}

/// `ID2D1Factory1::RegisterEffectFromString`: parse the XML schema, bind the author factory, add the
/// registration keyed by CLSID **within factory `fid`'s registry**. Re-registering the same CLSID on
/// the same factory bumps a refcount and keeps the first definition (native refcounts; a real app
/// re-registers the same definition). A CLSID equal to a built-in → `E_INVALIDARG`.
fn register(fid: u64, clsid: GUID, xml: &str, factory: Rc<dyn Fn() -> Box<dyn EffectImpl>>) -> Result<(), HRESULT> {
    if super::kind_of(&clsid).is_some() {
        return Err(E_INVALIDARG);
    }
    let schema = parse_schema(xml)?;
    REGISTRY.with(|reg| {
        let mut reg = reg.borrow_mut();
        match reg.entry(fid).or_default().entry(clsid.to_u128()) {
            std::collections::hash_map::Entry::Occupied(mut e) => e.get_mut().refcount += 1,
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(Registration { refcount: 1, schema, factory });
            }
        }
        Ok(())
    })
}

/// `CreateEffect` for a custom CLSID: instantiate the author object, build its initial graph from the
/// schema defaults, return a `Custom`-kind effect. Unregistered → `NOTFOUND`.
/// The custom effect CLSIDs registered on factory `fid` (`GetRegisteredEffects`'s non-system entries).
pub fn registered_custom_clsids(fid: u64) -> Vec<GUID> {
    REGISTRY.with(|reg| reg.borrow().get(&fid).map(|m| m.keys().map(|&k| GUID::from_u128(k)).collect()).unwrap_or_default())
}

pub(super) fn create_registered(clsid: GUID, dpi: (f32, f32), fid: u64) -> Result<Effect, HRESULT> {
    REGISTRY.with(|reg| {
        let reg = reg.borrow();
        let r = reg.get(&fid).and_then(|m| m.get(&clsid.to_u128())).ok_or(NOTFOUND)?;
        let mut state = CustomState {
            schema: r.schema.clone(),
            imp: (r.factory)(),
            dpi,
            factory_id: fid,
            graph: TransformGraph::new(r.schema.inputs),
            shaders: Rc::new(RefCell::new(HashMap::new())),
            resource_textures: Rc::new(RefCell::new(HashMap::new())),
            vertex_buffers: Rc::new(RefCell::new(HashMap::new())),
        };
        state.rebuild(&BTreeMap::new())?;
        Ok(Effect::from_custom(dpi, state))
    })
}

// ============================================================================================
// Public API on Factory / DeviceContext / Effect
// ============================================================================================

impl crate::Factory {
    /// `ID2D1Factory1::RegisterEffectFromString`. `factory` produces a fresh author object per
    /// `CreateEffect`. See [`register`].
    pub fn register_effect_from_string(&self, clsid: GUID, property_xml: &str, factory: Rc<dyn Fn() -> Box<dyn EffectImpl>>) -> Result<(), HRESULT> {
        register(self.id(), clsid, property_xml, factory)
    }

    /// The effect CLSIDs `ID2D1Factory1::GetRegisteredEffects` reports: the modelled built-in system
    /// effects followed by this factory's custom registrations.
    pub fn registered_effect_clsids(&self) -> Vec<GUID> {
        let mut list = super::BUILTIN_EFFECT_CLSIDS.to_vec();
        list.extend(registered_custom_clsids(self.id()));
        list
    }

    /// `ID2D1Factory1::RegisterEffectFromStream`: same, from already-read XML bytes (UTF-8).
    pub fn register_effect_from_stream(&self, clsid: GUID, property_xml: &[u8], factory: Rc<dyn Fn() -> Box<dyn EffectImpl>>) -> Result<(), HRESULT> {
        let xml = std::str::from_utf8(property_xml).map_err(|_| E_INVALIDARG)?;
        register(self.id(), clsid, xml, factory)
    }

    /// `UnregisterEffect`: decrement the CLSID's refcount in this factory's registry, removing it at
    /// zero. Not registered → `NOTFOUND`.
    pub fn unregister_effect(&self, clsid: GUID) -> Result<(), HRESULT> {
        REGISTRY.with(|reg| {
            let mut reg = reg.borrow_mut();
            let m = reg.get_mut(&self.id()).ok_or(NOTFOUND)?;
            let key = clsid.to_u128();
            let r = m.get_mut(&key).ok_or(NOTFOUND)?;
            r.refcount -= 1;
            if r.refcount == 0 {
                m.remove(&key);
            }
            Ok(())
        })
    }

    /// `GetRegisteredEffects`: the custom (non-built-in) registered CLSIDs for this factory.
    pub fn registered_effects(&self) -> Vec<GUID> {
        REGISTRY.with(|reg| reg.borrow().get(&self.id()).map(|m| m.keys().map(|&k| GUID::from_u128(k)).collect()).unwrap_or_default())
    }
}

impl crate::DeviceContext {
    /// `ID2D1EffectContext1::CreateLookupTable3D` at the device-context level, for binding to a
    /// `LookupTable3D` effect via [`Effect::set_lookup_table`].
    pub fn create_lookup_table3d(&self, extents: [u32; 3], data: Vec<[f32; 4]>) -> Result<Rc<Lut3d>, HRESULT> {
        Lut3d::new(extents, data).map(Rc::new)
    }
}

impl Effect {
    fn from_custom(dpi: (f32, f32), state: CustomState) -> Self {
        Effect(Rc::new(RefCell::new(Data { kind: Kind::Custom, props: BTreeMap::new(), inputs: Vec::new(), dpi, lut: None, custom: Some(state), realized: None, realized_f: None, bounds_cache: std::cell::Cell::new(None), precision: None, render_precision: 1, render_fallback: 1, cms: None, histogram: None, vignette: RefCell::new(None), hdr: RefCell::new(None), place: Default::default(), store: None })))
    }

    /// Bind a 3D LUT (`ID2D1LookupTable3D`) to a `LookupTable3D` effect (its lookup table property).
    pub fn set_lookup_table(&self, lut: Rc<Lut3d>) {
        self.0.borrow_mut().lut = Some(lut);
    }

    /// Rebuild a custom effect's transform graph from the current properties (after `SetValue`).
    pub(super) fn rebuild_custom(&self) -> Result<(), HRESULT> {
        let props = self.0.borrow().props.clone();
        let mut d = self.0.borrow_mut();
        if let Some(cs) = d.custom.as_mut() {
            cs.rebuild(&props)
        } else {
            Ok(())
        }
    }

    /// `SetValueByName`: resolve the property index from a custom effect's schema, then `SetValue`.
    /// Built-ins (no name schema here) → `E_INVALIDARG`.
    pub fn set_value_by_name(&self, name: &str, value: Prop) -> Result<(), HRESULT> {
        let index = self.0.borrow().custom.as_ref().and_then(|cs| cs.property_index(name)).ok_or(E_INVALIDARG)?;
        self.set_value(index, value)
    }

    /// `GetInputCount` for a custom effect (its schema input count); built-ins report their bound count.
    pub fn input_count(&self) -> u32 {
        let d = self.0.borrow();
        match &d.custom {
            Some(cs) => cs.input_count(),
            None => d.inputs.len() as u32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_desc_source_over_and_min() {
        // Premultiplied source-over: src·ONE + dst·INV_SRC_ALPHA, ADD, both colour and alpha.
        let over = BlendDesc { inputs: 2, src: 2, dst: 6, op: 1, src_a: 2, dst_a: 6, op_a: 1, bf: [0.0; 4] };
        let src = [0.5, 0.0, 0.0, 0.5];
        let dst = [0.0, 0.0, 0.5, 1.0];
        assert_eq!(over.blend(src, dst), [0.5, 0.0, 0.25, 1.0]);
        // MIN ignores the factors entirely.
        let min = BlendDesc { inputs: 2, src: 1, dst: 1, op: 4, src_a: 1, dst_a: 1, op_a: 4, bf: [0.0; 4] };
        assert_eq!(min.blend([0.2, 0.9, 0.1, 1.0], [0.7, 0.3, 0.4, 0.5]), [0.2, 0.3, 0.1, 0.5]);
    }

    // G1: a `type='blob'` property (ComputeSharp.D2D1's pixel-shader constant buffer) must register,
    // not reject the whole effect at `RegisterEffectFromString` with E_INVALIDARG.
    #[test]
    fn blob_property_type_registers() {
        let xml = "<?xml version='1.0'?><Effect>\
            <Property name='DisplayName' type='string' value='Test'/>\
            <Inputs><Input name='Source'/></Inputs>\
            <Property name='Buffer' type='blob'/></Effect>";
        let schema = parse_schema(xml).expect("blob property must register, not E_INVALIDARG");
        assert_eq!(schema.inputs, 1);
        assert!(schema.props.iter().any(|p| p.name == "Buffer" && matches!(p.kind, PropKind::Blob)));
        // A blob value (the FFI decodes D2D1_PROPERTY_TYPE_BLOB as Floats) validates against Blob.
        assert!(PropKind::Blob.matches(&Prop::Floats(std::rc::Rc::from(&[1.0f32, 2.0][..]))));
    }

    // Per-draw invalidation: `clear_shader_caches` (which `prepare_for_render` calls before every
    // draw) must empty a shader node's memoized derivative quads, so the next draw — which may bind a
    // different constant buffer or inputs — never reuses a quad realized from the previous draw.
    #[test]
    fn clear_shader_caches_drops_a_shader_nodes_quad_cache() {
        use d2d_shader::sm4::{Index, Inst, Opcode, Operand, Program, Sel};
        use std::rc::Rc;
        // deriv_rtx r0, v0 (marks the shader uses_deriv) ; mov o0, cb0[0].
        let o = |ty, sel, indices| Operand { ty, num_components: 4, sel, modifier: 0, indices, imm: vec![] };
        let inst = |op, operands| Inst { op, saturate: false, test_nz: false, operands, decl_count: 0, aux: 0 };
        let program = Program {
            program_type: 0, major: 5, minor: 0,
            insts: vec![
                inst(Opcode::DerivRtx, vec![o(0, Sel::Mask(0xf), vec![Index::Imm(0)]), o(1, Sel::Swizzle([0, 1, 2, 3]), vec![Index::Imm(0)])]),
                inst(Opcode::Mov, vec![o(2, Sel::Mask(0xf), vec![Index::Imm(0)]), o(8, Sel::Swizzle([0, 1, 2, 3]), vec![Index::Imm(0), Index::Imm(0)])]),
            ],
            temps: 1, indexable_temps: vec![], thread_group: [0; 3], uavs: vec![], tgsm: vec![], resources: vec![], icb: vec![],
        };
        let shader = Rc::new(Shader { program, inputs: vec![], outputs: vec![], input_simple: vec![] });
        let cb: Vec<u8> = [0.5f32, 0.0, 0.0, 0.0].iter().flat_map(|f| f.to_le_bytes()).collect();
        let node = ShaderNode::new(shader, Rc::new(RefCell::new(cb)), 0, Rc::default(), 0).expect("build");

        // A draw memoizes the quad; the graph holding the node must clear it on the next prepare.
        node.eval(2, 2, &[], &[], &|_r, _c| [0.0; 4], &|_r, _c| [0; 4], &|_r| [0; 4]);
        assert_eq!(node.quad_cache_len(), 1, "a draw memoizes a quad");
        let mut g = TransformGraph::new(0);
        g.add_node(Transform(NodeKind::Shader(node.clone(), Rc::default())));
        g.clear_shader_caches();
        assert_eq!(node.quad_cache_len(), 0, "clear_shader_caches empties the node's quad cache");
    }

    // G14: `type='iunknown'` properties (ComputeSharp.D2D1 3.2.0's `TransformMapper` and
    // `ResourceTextureManager0..15`) must register, not reject the whole effect. (G1 only added 'blob'.)
    #[test]
    fn iunknown_property_type_registers() {
        let xml = "<?xml version='1.0'?><Effect>\
            <Property name='DisplayName' type='string' value='Test'/>\
            <Inputs><Input name='Source'/></Inputs>\
            <Property name='TransformMapper' type='iunknown'/>\
            <Property name='ResourceTextureManager0' type='iunknown'/></Effect>";
        let schema = parse_schema(xml).expect("iunknown property must register, not E_INVALIDARG");
        assert_eq!(schema.props.iter().filter(|p| matches!(p.kind, PropKind::IUnknown)).count(), 2);
    }

    #[test]
    fn resource_texture_sampling() {
        // 1-D, 2 texels, RGBA8: texel0 black-opaque, texel1 white-opaque.
        let data = [0, 0, 0, 255, 255, 255, 255, 255];
        let w = super::super::sample::unorm_to_f32(255);
        let point = ResourceTexture::new(1, &[2], 1 /*8UNORM*/, 4 /*CHANNEL_4*/, 0 /*POINT*/, &[], &data).unwrap();
        assert_eq!(point.sample([0.25, 0.0, 0.0, 0.0]), [0.0, 0.0, 0.0, w]);
        assert_eq!(point.sample([0.75, 0.0, 0.0, 0.0]), [w, w, w, w]);
        // Linear at the texel-1 centre reads texel1 exactly (coord 0.75 → c = 0.75·2 − 0.5 = 1.0).
        let lin = ResourceTexture::new(1, &[2], 1, 4, 0x15 /*LINEAR*/, &[], &data).unwrap();
        assert_eq!(lin.sample([0.75, 0.0, 0.0, 0.0]), [w, w, w, w]);
        // Dimensions out of range / wrong extent count → E_INVALIDARG.
        assert!(ResourceTexture::new(4, &[2, 2, 2, 2], 1, 4, 0, &[0; 4], &[]).is_err());
        // 16FLOAT decode: a 1-texel R16 half of 1.0 (0x3C00) and 0.5 (0x3800) sample as 1.0 / 0.5.
        let half = ResourceTexture::new(1, &[2], 4 /*16FLOAT*/, 1 /*CHANNEL_1*/, 0, &[], &[0x00, 0x3c, 0x00, 0x38]).unwrap();
        assert_eq!(half.sample([0.25, 0.0, 0.0, 0.0])[0], 1.0);
        assert_eq!(half.sample([0.75, 0.0, 0.0, 0.0])[0], 0.5);
    }

    #[test]
    fn precision_rule() {
        use super::super::{prec_rank, quantize_to};
        // Quantise: 8bpc round-trips a stored byte exactly (== the old q8), clamps to [0,1]; 32FLOAT
        // keeps the extended range untouched (so a coord LUT / HDR value survives a shader input).
        for b in [0u8, 1, 128, 180, 255] {
            let v = b as f32 / 255.0;
            assert_eq!(quantize_to(1, [v, v, v, v])[0], v, "8bpc round-trip b={b}");
        }
        assert_eq!(quantize_to(1, [2.0, -1.0, 0.0, 1.0]), [1.0, 0.0, 0.0, 1.0]); // UNORM clamp
        assert_eq!(quantize_to(5, [17.0, -3.0, 0.123_456, 2.0]), [17.0, -3.0, 0.123_456, 2.0]); // 32F untouched
        assert!(prec_rank(5) > prec_rank(4) && prec_rank(4) > prec_rank(3) && prec_rank(3) > prec_rank(1));
        assert_eq!(prec_rank(0), 1); // UNKNOWN folds to 8bpc

        // The render's context precision governs an intermediate node; SetOutputBuffer is inert (WARP).
        let mut g = TransformGraph::new(1);
        let n = g.add_node(Transform(NodeKind::Offset(0, 0)));
        g.render_precision.set(1);
        assert_eq!(g.node_precision(n.0, &[]), 1); // 8bpc context
        g.set_output_buffer(n, 5).unwrap();
        assert_eq!(g.node_precision(n.0, &[]), 1); // SOB does NOT raise it
        g.render_precision.set(5);
        assert_eq!(g.node_precision(n.0, &[]), 5); // follows the render context
    }

    #[test]
    fn float_input_bitmap_keeps_precision() {
        use super::super::{eval_input, input_precision, Input};
        // A float (R32G32B32A32_FLOAT) source bitmap reports 32FLOAT precision and reads back its raw
        // texels — including values outside [0,1] (scene-position coords an 8bpc bitmap would clamp).
        let texels = [3.5f32, 7.25, 0.0, 1.0, 100.0, -2.0, 0.0, 1.0]; // 2×1, RGBA
        let inp = Input::Bitmap(std::rc::Rc::new(crate::bitmap::Bitmap::from_f32_rgba(2, 1, &texels)));
        assert_eq!(input_precision(&inp), 5); // 32FLOAT, not 8bpc
        assert_eq!(eval_input(&inp, 0, 0), [3.5, 7.25, 0.0, 1.0]);
        assert_eq!(eval_input(&inp, 1, 0), [100.0, -2.0, 0.0, 1.0]);
    }

    #[test]
    fn precision_per_context_no_leak() {
        // Two DeviceContexts on one thread at different bufferPrecisions, drawing the SAME effect
        // interleaved: each draw re-establishes its own context's precision on the effect (per-context
        // state pushed in at realize, not a process-wide thread-local), so one never leaks into the other.
        let factory = crate::Factory::new(false);
        let device = factory.create_device();
        let (mut a, mut b) = (device.create_device_context(), device.create_device_context());
        a.set_dpi(96.0, 96.0);
        b.set_dpi(96.0, 96.0);
        a.set_target_bitmap(crate::bitmap_target(4, 4, 96.0));
        b.set_target_bitmap(crate::bitmap_target(4, 4, 96.0));
        a.set_buffer_precision(5); // 32FLOAT
        b.set_buffer_precision(1); // 8bpc
        a.begin_draw();
        b.begin_draw();
        let flood = a.create_effect(GUID::from_u128(0x61c23c20_ae69_4d8e_94cf_50078df638f2)).unwrap();
        a.draw_image_effect(&flood, [0.0, 0.0], 1, 0);
        assert_eq!(flood.out_precision(), 5); // context A governs
        b.draw_image_effect(&flood, [0.0, 0.0], 1, 0);
        assert_eq!(flood.out_precision(), 1); // context B — no leak from A
        a.draw_image_effect(&flood, [0.0, 0.0], 1, 0);
        assert_eq!(flood.out_precision(), 5); // back to A — B's 8bpc did not stick
    }

    #[test]
    fn resource_texture_load_is_format_exact() {
        // `ld` must return the D3D format-exact UNORM decode b/255, NOT the sampler's fixed-point
        // b·257/65536 — otherwise a LUT read fed through `(int)(v·255)` (Clouds' permute table) is off
        // by one for every nonzero byte. 1-D R8_UNORM over all 256 byte values.
        let data: Vec<u8> = (0..=255u8).collect();
        let tex = ResourceTexture::new(1, &[256], 1 /*8UNORM*/, 1 /*CHANNEL_1*/, 0 /*POINT*/, &[], &data).unwrap();
        for b in 0..=255u8 {
            let v = tex.load([b as i32, 0, 0, 0])[0];
            assert_eq!(v, b as f32 / 255.0, "load b={b}");
            assert_eq!((v * 255.0) as i32, b as i32, "round-trip b={b}"); // the shader's recovery step
        }
        // The filtered sampler still uses the fixed-point expansion (so it differs from load for b>=1).
        assert_ne!(tex.load([180, 0, 0, 0])[0], tex.sample([(180.0 + 0.5) / 256.0, 0.0, 0.0, 0.0])[0]);
    }

    #[test]
    fn keyed_resource_and_vertex_find_roundtrip() {
        // CreateResourceTexture/CreateVertexBuffer with a GUID key store the resource so
        // FindResourceTexture/FindVertexBuffer return the SAME object (native's dedup R13); an
        // anonymous (None) create is not findable, and an unknown key is NOTFOUND.
        let ctx = EffectContext::new((96.0, 96.0), 0, Rc::new(RefCell::new(HashMap::new())), Rc::new(RefCell::new(HashMap::new())), Rc::new(RefCell::new(HashMap::new())));
        let key = GUID::from_u128(0x1111_2222);
        let other = GUID::from_u128(0x3333_4444);

        let tex = ctx.create_resource_texture_keyed(Some(key), 1, &[2], 1, 4, 0, &[], &[0, 0, 0, 255, 255, 255, 255, 255]).unwrap();
        assert!(Rc::ptr_eq(&tex, &ctx.find_resource_texture(key).unwrap()), "keyed texture is findable");
        // A second keyed create with the same GUID returns the first (dedup, no re-upload).
        assert!(Rc::ptr_eq(&tex, &ctx.create_resource_texture_keyed(Some(key), 1, &[2], 1, 4, 0, &[], &[]).unwrap()));
        assert_eq!(ctx.find_resource_texture(other).err(), Some(NOTFOUND));
        // Anonymous textures are not stored.
        let _anon = ctx.create_resource_texture_keyed(None, 1, &[2], 1, 4, 0, &[], &[0, 0, 0, 255, 255, 255, 255, 255]).unwrap();
        assert_eq!(ctx.find_resource_texture(other).err(), Some(NOTFOUND));

        let els = || vec![VbElement { semantic: "POSITION".into(), index: 0, offset: 0, components: 2 }];
        let vb = ctx.create_vertex_buffer(Some(key), &[0u8; 16], 8, els()).unwrap();
        assert!(Rc::ptr_eq(&vb, &ctx.find_vertex_buffer(key).unwrap()), "keyed vertex buffer is findable");
        assert!(Rc::ptr_eq(&vb, &ctx.create_vertex_buffer(Some(key), &[0u8; 16], 8, els()).unwrap()));
        assert_eq!(ctx.find_vertex_buffer(other).err(), Some(NOTFOUND));
    }

    #[test]
    fn transform_node_from_effect_renders_subeffect() {
        // CreateTransformNodeFromEffect wraps a pre-wired sub-effect as a graph leaf (NodeKind::Effect):
        // a Flood generator set to a colour, wrapped and made the graph's output node, floods that
        // colour at every pixel — the path the FFI EffectNodeObj drives.
        use super::super::CLSID_D2D1Flood;
        let ctx = EffectContext::new((96.0, 96.0), 0, Rc::new(RefCell::new(HashMap::new())), Rc::new(RefCell::new(HashMap::new())), Rc::new(RefCell::new(HashMap::new())));
        let flood = ctx.create_effect(CLSID_D2D1Flood).unwrap();
        flood.set_value(0, Prop::V4([0.25, 0.5, 0.75, 1.0])).unwrap();
        let node = ctx.create_transform_node_from_effect(flood);
        let mut g = TransformGraph::new(0);
        let n = g.add_node(node);
        g.set_output_node(n).unwrap();
        assert_eq!(g.eval_ref(Ref::Node(n.0), &[], 3, 7), [0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn compute_effect_matches_warp() {
        // shader2-09 is a 0-input compute generator (8×8 unorm RWTexture2D at u0, dispatch 2×2 of
        // [4,4,1] threads). Build it as a compute transform and check the rendered output byte-for-byte
        // against the WARP capture of the UAV (`.bin`), confirming the effect-graph compute path.
        let Some(suite) = d2d_reference::suite_dir() else { return };
        let dxbc = std::fs::read(suite.join("scenes/shader2-09-compute-uav.dxbc")).unwrap();
        let bin = std::fs::read(suite.join("oracle/win11-26100/shader2-09-compute-uav.bin")).unwrap();
        let ctx = EffectContext::new((96.0, 96.0), 0, Rc::new(RefCell::new(HashMap::new())), Rc::new(RefCell::new(HashMap::new())), Rc::new(RefCell::new(HashMap::new())));
        let id = GUID::from_u128(0xc0_f0);
        ctx.load_compute_shader(id, &dxbc).unwrap();
        let t = ctx.compute_shader_transform(id, Vec::new(), 0, Vec::new(), [2, 2, 1], [0, 0]).unwrap();
        let mut g = TransformGraph::new(0);
        let n = g.add_node(t);
        g.set_output_node(n).unwrap();
        for y in 0..8i32 {
            for x in 0..8i32 {
                let px = g.eval_ref(Ref::Node(n.0), &[], x, y);
                for k in 0..4 {
                    let got = (px[k] * 255.0).round_ties_even() as u8;
                    assert_eq!(got, bin[(y as usize * 8 + x as usize) * 4 + k], "compute pixel ({x},{y}) ch {k}");
                }
            }
        }
    }

    #[test]
    fn resource_texture_update_and_vertex_map() {
        // Update a 2×1 RGBA8 texture's second texel; the sample reflects it. Out-of-bounds → E_INVALIDARG.
        let w = super::super::sample::unorm_to_f32(255);
        let tex = ResourceTexture::new(1, &[2], 1, 4, 0, &[], &[0, 0, 0, 255, 0, 0, 0, 255]).unwrap();
        tex.update(&[1], &[2], 1, &[255, 255, 255, 255]).unwrap();
        assert_eq!(tex.sample([0.75, 0.0, 0.0, 0.0]), [w, w, w, w]);
        assert_eq!(tex.update(&[1], &[3], 1, &[0; 4]), Err(E_INVALIDARG)); // max past extent
        assert_eq!(tex.update(&[0], &[2], 1, &[0; 4]), Err(E_INVALIDARG)); // too little data
        // Vertex buffer Map returns a writable pointer; Unmap without Map → E_FAIL.
        let vb = VertexBuffer { data: RefCell::new(vec![0u8; 16]), stride: 16, elements: Vec::new(), mapped: Cell::new(false) };
        assert_eq!(vb.unmap(), Err(E_FAIL));
        let (ptr, len) = vb.map();
        assert!(!ptr.is_null() && len == 16);
        assert!(vb.unmap().is_ok());
    }

    #[test]
    fn remove_node_tombstone() {
        let mut g = TransformGraph::new(1);
        let a = g.add_node(Transform(NodeKind::Offset(1, 0)));
        let b = g.add_node(Transform(NodeKind::Offset(2, 0)));
        g.remove_node(b).unwrap();
        assert!(g.remove_node(b).is_err()); // already removed
        assert!(g.connect_node(b, a, 0).is_err()); // from a removed node
        assert!(g.set_output_node(b).is_err()); // output a removed node
        assert!(g.set_output_node(a).is_ok()); // a live node is fine
    }

    use super::Lut3d;

    #[test]
    fn lut3d_axis_convention() {
        // Identity LUT is a near-passthrough: B→dim0, R→dim2 (the Q8 sampler quantises the sub-texel
        // weight, so it is only exact on the UNORM8 round-trip — tolerance catches an axis flip).
        let id = Lut3d::identity();
        for px in [[0.08, 0.04, 1.0, 1.0], [0.3, 0.6, 0.9, 1.0], [0.0, 1.0, 0.5, 1.0]] {
            let out = id.sample(px);
            for k in 0..4 {
                assert!((out[k] - px[k]).abs() < 0.01, "identity LUT changed {px:?} -> {out:?}");
            }
        }
        // The ffi4-01 "swap-rb" cube is authored B-fastest/R-slowest, so WARP reads it as identity
        // (not a swap). Build it and confirm a mid pixel maps to itself, catching an axis flip.
        let data: Vec<[f32; 4]> = (0..8)
            .map(|f: usize| [(f >> 2 & 1) as f32, (f >> 1 & 1) as f32, (f & 1) as f32, 1.0])
            .collect();
        let lut = Lut3d::new([2, 2, 2], data).unwrap();
        let out = lut.sample([0.137, 0.039, 0.945, 1.0]);
        assert!(out[0] < 0.2 && out[2] > 0.9, "swap-rb cube must be identity, got {out:?}");
    }
}
