//! Shader-backed custom-effect draw transform: run a custom
//! effect's pixel shader through `d2d-shader`'s WARP-exact DXBC executor on the engine's CPU pull
//! model, reproducing how native Direct2D realizes an `ID2D1DrawTransform`.
//!
//! The realization, as Direct2D's draw info and `d2d1effecthelpers.hlsli` define it: the default effect
//! vertex shader (chosen by input count) feeds the pixel shader `SV_POSITION` (the scene pixel
//! centre), a `SCENE_POSITION` texcoord, and one `TEXCOORD`n per input carrying that input's sample
//! coordinate (`.xy` the coordinate, `.zw` the per-texel deltas). The full/linked pixel shader
//! samples input `tN` with sampler `sN` at `TEXCOORDn`, reads the effect constant buffer at `b0`,
//! and writes premultiplied RGBA to `o0`.
//!
//! For a **simple** input (`D2D1_PIXEL_OPTIONS_TRIVIAL_SAMPLING`, identity 1:1
//! `MapOutputRectToInputRects`) the coordinate maps the output pixel 1:1 into the input, so the
//! sample returns that input's texel at the current pixel — which the pull engine already has as
//! `eval(x, y)`. The engine is point-sampled and untiled (see `effects/mod.rs`), so a simple input's
//! bilinear weights, intermediate precision and tiling have no pixel effect. A **complex** input's
//! neighbourhood read (`D2DSampleInputAtOffset/AtPosition`) *is* reproduced: the default-effect-VS
//! deltas live in `TEXCOORDn.zw` (1 device pixel per offset unit, 1:1 input) so the macros address
//! the neighbour pixel, and the engine sampler closure (`effects/mod.rs`) resolves any non-centre tap
//! through `effects/sample.rs`'s WARP fixed-point bilinear over the slot's 8bpc texels, clamped to the
//! slot's device bounds (edge extend). The output premultiplied RGBA is returned to the engine, which
//! quantises once at `DrawImage` (`to_unorm8` ≡ WARP's UNORM store).
//! Not yet reproduced: a non-1:1 complex input's per-input delta (resamplers needing the real
//! scene→texel transform) and non-clamp sampler extend modes.

use d2d_shader::exec::{predecode_shader, run_predecoded_reuse, ExecScratch, Predecoded};
use d2d_shader::sm4::Opcode;
use d2d_shader::{Env, PixelInputs, Shader};
use std::cell::RefCell;
use std::rc::Rc;
use windows_core::HRESULT;

/// `E_FAIL` — a shader that cannot be realized on the CPU executor (malformed DXBC, or an opcode the
/// executor does not implement) fails `Initialize`, and never silently renders wrong pixels.
const E_FAIL: HRESULT = HRESULT(0x8000_4005u32 as i32);

/// What the default effect vertex shader writes into a pixel-shader input register: `SV_Position` and
/// `SCENE_POSITION` carry the device scene position; a `TEXCOORD`n carries input `n`'s *normalized*
/// sample coordinate (`[0,1]` across the input, the D3D sampler convention), with `.zw` the normalized
/// per-texel step for `SampleInputAtOffset` — **except** a `TEXCOORD`n for a **simple** input
/// (`__d2dinput_color`n), which the linker pre-samples 1:1, where the VS writes the device scene centre
/// (as `SCENE_POSITION`) and the sampler point-pulls it (`ShaderNode::is_simple`, handled in `fill_inputs`).
#[derive(Clone, Copy)]
enum Role {
    /// `SV_Position` — device scene pixel centre, z=0, w=1.
    SvPosition,
    /// `SCENE_POSITION` (or an unclassified semantic) — device scene centre with the 1-device-pixel step.
    Scene,
    /// `TEXCOORD`n for input index `n` — the input's normalized sample coordinate (a complex input), or
    /// the device scene centre when the input is simple (`ShaderNode::is_simple`).
    Input(u32),
}

/// A bitmap input's TEXCOORD normalizing size: its content dimensions `(r−l, b−t)`. Direct2D realizes
/// each effect-shader input into a texture and the texcoord is `(pos − origin)/texture_size`; for a plain
/// bitmap that texture *is* the bitmap, so `texture_size` is the content size and `coord·size` recovers
/// the texel 1:1. An **intermediate** (another transform's output, or an effect/command-list input D2D
/// must realize) is packed into the fixed realization atlas and normalizes by that instead — the input
/// kind decides, so the dim is chosen per input in `custom.rs::slot_norm_dims` and threaded in here; this
/// is only the fallback when a caller passes no dim. `rect` is the device bounds `[l, t, r, b]`.
fn content_dims(rect: [i32; 4]) -> (f32, f32) {
    let [l, t, r, b] = rect;
    ((r - l).max(1) as f32, (b - t).max(1) as f32)
}

/// Fit the per-input TEXCOORD plane the way WARP's rasteriser does: the default effect VS emits, at
/// each corner of the draw quad (= the input's 1:1 output rect `[l, t, r, b]`), the input's normalized
/// coordinate `(pos − origin)/texture_size` and its per-texel step `.zw`; WARP splits the quad into two
/// triangles and fits each of the four components with [`setup_triangle_attr`] (exact-area reciprocal).
/// Returns `[triangle][component]` plane coefficients. This matches native at the ±1-ULP edges that an
/// analytic `(pos − origin)/size` misses (the unit-triangle fit drops ~ULP — see `warp/plane.rs`), the
/// way `builtin::filters_hp` already does for the blur passes.
pub(super) fn fit_input_planes(rect: [i32; 4], dim_x: f32, dim_y: f32) -> [[[f32; 3]; 4]; 2] {
    use crate::warp::plane::setup_triangle_attr;
    let [l, t, r, b] = rect;
    let (w, h) = ((r - l) as f32, (b - t) as f32);
    // The content spans `0..w`/`0..h` in device pixels, but the texcoord normalizes by the *realized
    // texture* size `dim_x`/`dim_y` (a bitmap's content size, or the realization atlas for an
    // intermediate — see `custom.rs::slot_norm_dims`), so the `u·dim` round-trip at
    // `SampleInputAtOffset` taps is exact.
    let (ud_x, uc_x) = crate::sampling::float_plane(0.0, w, l as f32, r as f32, dim_x);
    let (ud_y, uc_y) = crate::sampling::float_plane(0.0, h, t as f32, b as f32, dim_y);
    // Per-vertex value: [u, v, per-texel step x, per-texel step y].
    let corner = |x: i32, y: i32| [ud_x * x as f32 + uc_x, ud_y * y as f32 + uc_y, ud_x, ud_y];
    let tris = [[(l, t), (r, b), (r, t)], [(l, b), (l, t), (r, b)]];
    tris.map(|tri| {
        let sx = tri.map(|(x, _)| x as i64 * 256);
        let sy = tri.map(|(_, y)| y as i64 * 256);
        let v = tri.map(|(x, y)| corner(x, y));
        core::array::from_fn(|c| setup_triangle_attr(sx, sy, [v[0][c], v[1][c], v[2][c]]))
    })
}

/// Evaluate the fitted TEXCOORD planes ([`fit_input_planes`]) at pixel centre `(px, py)`: pick the
/// triangle the pixel falls in (the quad's diagonal split, as WARP's rasteriser does) and evaluate the
/// four components in WARP's add order (`a[0]·px + (a[1]·py + a[2])`). Returns `[u, v, step_x, step_y]`.
pub(super) fn eval_input_texcoord(planes: &[[[f32; 3]; 4]; 2], rect: [i32; 4], px: f32, py: f32) -> [f32; 4] {
    let [l, t, r, b] = rect;
    let (w, h) = ((r - l) as f32, (b - t) as f32);
    let k = usize::from((py - t as f32) * w > (px - l as f32) * h);
    core::array::from_fn(|c| {
        let a = &planes[k][c];
        a[0] * px + (a[1] * py + a[2])
    })
}

/// One realized pixel-shader draw transform: the parsed shader, its `b0` bytes, the input arity and
/// the input-register layout the default vertex shader fills.
pub struct ShaderNode {
    shader: Rc<Shader>,
    /// The pixel shader's `b0` constants. A shared, interior-mutable cell so a custom effect's
    /// `PrepareForRender` (ComputeSharp's `SetPixelShaderConstantBuffer`) can update the constants
    /// seen by this draw without rebuilding the node: the FFI binds the SAME cell to its `DrawInfo`
    /// so the app writes straight into what the executor reads. [G2]
    cbuffer: Rc<RefCell<Vec<u8>>>,
    input_count: u32,
    /// `(register, Role)` for each input-signature element; the registers the default VS writes.
    regs: Vec<(u32, Role)>,
    n_regs: usize,
    /// The shader reads screen-space derivatives (`ddx`/`ddy`), so it must run over a 2×2 quad or the
    /// derivatives are zero (and a divide-by-derivative — e.g. a band-limited generator — is NaN).
    uses_deriv: bool,
    /// The shader lowered once for fast per-pixel execution (operand decoding hoisted out of the
    /// per-pixel loop); paired with a reusable register file so the common 1-lane path neither
    /// re-decodes operands nor reallocates registers per pixel.
    pre: Predecoded,
    scratch: RefCell<ExecScratch>,
    /// Derivative path only: the 2×2 quads realized this draw, keyed by top-left `(qx, qy)`. A `ddx`/
    /// `ddy` shader runs over the whole quad to derive one output pixel, so every pixel re-runs (and
    /// re-samples) its quad's other three lanes; this memoizes the quad so each is computed once per
    /// draw (≈4× fewer shader+sample passes), independent of pull order. Cleared before each draw by
    /// `prepare_for_render` (the quad depends on the per-draw constant buffer and inputs), so a stale
    /// quad can never leak across draws. Bounded by the output rect (output pixels / 4), like the
    /// compute/vertex nodes' whole-output caches.
    quad_cache: RefCell<std::collections::HashMap<(i32, i32), [[f32; 4]; 4]>>,
    /// Per input, how far (`[l, t, r, b]` px) the draw transform's requested input rect reaches past
    /// its output rect (`MapOutputRectToInputRects`): 0 for the 1:1 identity, `i32::MAX` for an
    /// infinite request (ComputeSharp's complex input). A shared cell, refreshed before each draw by
    /// the FFI from the app's transform like `cbuffer`; empty ⇒ identity.
    reach: Rc<RefCell<Vec<[i32; 4]>>>,
    /// The input sampler's `D2D1_FILTER` (the app's `SetInputDescription`; `0` =
    /// `MIN_MAG_MIP_POINT`, the effect default). Only the magnification bit (`0x04`) is consulted: a
    /// point sampler fetches the single nearest texel (`floor`), a linear sampler the 2×2 bilinear.
    /// Inert on texel-aligned taps (point == linear at weight 0), so it shows only when a shader
    /// samples a fractional computed position.
    filter: u32,
    /// Per input slot (indexed by slot — small and contiguous, so a `Vec`, not a hashed map, since this
    /// is read on the sampler's hottest path), the fitted TEXCOORD planes ([`fit_input_planes`]) with the
    /// `(rect, dim)` they were fit for. The planes depend only on the input rect and its normalizing dim
    /// (both fixed for a draw), yet a shader would otherwise fit them (8 `setup_triangle_attr`) once per
    /// pixel. [`planes_for`] reuses a cached entry only while `(rect, dim)` still match, so it is computed
    /// once per input per draw and a later draw with a different rect recomputes: self-invalidating, so
    /// it needs no explicit per-draw clear. [perf]
    input_planes: RefCell<Vec<Option<([i32; 4], (f32, f32), [[[f32; 3]; 4]; 2])>>>,
    /// The per-input `(device rect, normalizing dim)` for the current draw, computed once and reused by
    /// every pixel's `fill_inputs` and sample closure. Both are fixed for a draw (they come from the
    /// graph's input bounds and kinds), so [`ensure_draw_inputs`] fills this on the first pixel and the
    /// rest just index it — without this, a shader evaluated at millions of pixels re-walked the input
    /// bounds and re-allocated these vectors per pixel, which bound a whole-layer effect draw to a hang.
    /// Emptied before each draw by `clear_quad_cache` (`prepare_for_render`), as the inputs may change.
    draw_inputs: RefCell<(Vec<[i32; 4]>, Vec<(f32, f32)>)>,
}

impl ShaderNode {
    /// Build the node, failing if the executor cannot run the shader (`unsupported_op` → `E_FAIL`,
    /// honouring "never silently zero an unimplemented opcode"; the first such opcode is reported).
    pub fn new(shader: Rc<Shader>, cbuffer: Rc<RefCell<Vec<u8>>>, input_count: u32, reach: Rc<RefCell<Vec<[i32; 4]>>>, filter: u32) -> Result<Rc<ShaderNode>, HRESULT> {
        if shader.unsupported_op().is_some() {
            return Err(E_FAIL);
        }
        let regs: Vec<(u32, Role)> = shader
            .inputs
            .iter()
            .map(|e| {
                let role = if e.system_value == 1 {
                    Role::SvPosition
                } else if e.semantic.starts_with("TEXCOORD") {
                    Role::Input(e.semantic_index)
                } else {
                    // SCENE_POSITION (and any other arbitrary semantic) stays the device scene position.
                    Role::Scene
                };
                (e.register, role)
            })
            .collect();
        let n_regs = regs.iter().map(|&(r, _)| r as usize + 1).max().unwrap_or(0);
        let uses_deriv = shader.program.insts.iter().any(|i| {
            matches!(i.op, Opcode::DerivRtx | Opcode::DerivRty | Opcode::DerivRtxCoarse | Opcode::DerivRtxFine | Opcode::DerivRtyCoarse | Opcode::DerivRtyFine)
        });
        let pre = predecode_shader(&shader);
        let scratch = RefCell::new(ExecScratch::new(&shader));
        Ok(Rc::new(ShaderNode {
            shader,
            cbuffer,
            input_count,
            regs,
            n_regs,
            uses_deriv,
            reach,
            pre,
            scratch,
            quad_cache: RefCell::new(std::collections::HashMap::new()),
            filter,
            input_planes: RefCell::new(Vec::new()),
            draw_inputs: RefCell::new((Vec::new(), Vec::new())),
        }))
    }

    /// `ID2D1TransformNode::GetInputCount` — the node arity (`t0..` sampled by the shader).
    pub fn input_count(&self) -> u32 {
        self.input_count
    }

    /// Drop every memoized derivative quad. Called before each draw (`prepare_for_render`) because the
    /// next draw may bind a different constant buffer or inputs, so a quad from the previous draw must
    /// not be reused.
    pub fn clear_quad_cache(&self) {
        self.quad_cache.borrow_mut().clear();
        // The next draw may bind different inputs, so drop the cached per-input rects/dims; they are
        // repopulated lazily on the draw's first pixel by `ensure_draw_inputs`.
        let mut di = self.draw_inputs.borrow_mut();
        di.0.clear();
        di.1.clear();
    }

    /// Populate [`ShaderNode::draw_inputs`] for this draw if it is not already filled for `count` inputs:
    /// `f(i)` yields input `i`'s `(device rect, normalizing dim)`. Called on the draw's first pixel; every
    /// later pixel sees it filled and skips the recompute. Cleared per draw by [`clear_quad_cache`].
    pub(super) fn ensure_draw_inputs(&self, count: usize, f: impl Fn(usize) -> ([i32; 4], (f32, f32))) {
        let mut di = self.draw_inputs.borrow_mut();
        if di.0.len() == count {
            return;
        }
        di.0.clear();
        di.1.clear();
        for i in 0..count {
            let (rect, dim) = f(i);
            di.0.push(rect);
            di.1.push(dim);
        }
    }

    /// Borrow this draw's cached per-input `(rects, dims)` (see [`ShaderNode::draw_inputs`]).
    pub(super) fn draw_inputs(&self) -> std::cell::Ref<'_, (Vec<[i32; 4]>, Vec<(f32, f32)>)> {
        self.draw_inputs.borrow()
    }

    /// Number of memoized derivative quads (test-only, for the cross-draw invalidation tests).
    #[cfg(test)]
    pub(crate) fn quad_cache_len(&self) -> usize {
        self.quad_cache.borrow().len()
    }

    /// Input `slot`'s requested-rect reach past the output rect (`[l, t, r, b]`; see `reach`). A reach that saturates the
    /// request to Direct2D's infinite rect is infinite, `i32::MAX` (an infinite input rect's right and bottom edges, a pixel
    /// short of it, came through as finite: Paint.NET's Ripple and Wavey Transform then realized a giant input).
    pub fn reach(&self, slot: usize) -> [i32; 4] {
        self.reach.borrow().get(slot).copied().unwrap_or([0; 4]).map(|v| if v >= 1 << 30 { i32::MAX } else { v })
    }

    /// Whether the input sampler magnifies with a linear (bilinear) filter rather than point (nearest);
    /// the magnification bit of the input's `D2D1_FILTER` (see `filter`).
    pub(super) fn mag_linear(&self) -> bool {
        self.filter & 0x04 != 0
    }

    /// Whether input slot `i` is a **simple** (trivially-sampled 1:1) linked input (see
    /// [`d2d_shader::Shader::input_simple`]): its TEXCOORD is the device scene position and the sampler
    /// point-pulls it, instead of the normalized content/atlas texcoord a complex input samples.
    pub(super) fn is_simple(&self, i: u32) -> bool {
        self.shader.is_input_simple(i)
    }

    /// The fitted TEXCOORD planes ([`fit_input_planes`]) for input `slot` at device rect `rect`,
    /// normalized by `dim`, memoized in [`ShaderNode::input_planes`]: the fit is reused across this
    /// draw's pixels and sample taps and only recomputed when `(rect, dim)` change. Both `fill_inputs`
    /// (the TEXCOORD the VS writes) and the custom-effect sample closure (its identity-tap check) call
    /// this with the same `(rect, dim)`, so the coordinate the shader reads and the one the sampler
    /// compares against are fit from the identical planes.
    pub(super) fn planes_for(&self, slot: u32, rect: [i32; 4], dim: (f32, f32)) -> [[[f32; 3]; 4]; 2] {
        let s = slot as usize;
        if let Some(Some((r, d, p))) = self.input_planes.borrow().get(s) {
            if *r == rect && *d == dim {
                return *p;
            }
        }
        let p = fit_input_planes(rect, dim.0, dim.1);
        let mut cache = self.input_planes.borrow_mut();
        if cache.len() <= s {
            cache.resize(s + 1, None);
        }
        cache[s] = Some((rect, dim, p));
        p
    }

    /// The PS input registers the default effect VS fills for output pixel centre `(px, py)`.
    /// `input_rects[i]` is input `i`'s device rect `[l, t, r, b]`; each `TEXCOORD`n carries that input's
    /// *normalized* coordinate `(pos − origin)/size` built by Direct2D's affine source→dest map
    /// ([`crate::sampling::float_plane`]), with `.zw` the normalized per-texel step so
    /// `TEXCOORDn.xy + offset·TEXCOORDn.zw` lands on the neighbour texel — the D3D sampler convention
    /// the engine sampler closure then turns back into a texel via `coord·size` (WARP's `u·dim`).
    fn fill_inputs(&self, px: f32, py: f32, input_rects: &[[i32; 4]], input_dims: &[(f32, f32)]) -> Vec<[u32; 4]> {
        let mut inputs = vec![[0u32; 4]; self.n_regs];
        for &(reg, role) in &self.regs {
            inputs[reg as usize] = match role {
                Role::SvPosition => [px.to_bits(), py.to_bits(), 0, 1.0f32.to_bits()],
                Role::Scene => [px.to_bits(), py.to_bits(), 1.0f32.to_bits(), 1.0f32.to_bits()],
                Role::Input(i) if self.shader.is_input_simple(i) => {
                    // A **simple** input (`__d2dinput_color`n, `TRIVIAL_SAMPLING`): the linked shader
                    // samples it 1:1 at the output pixel's own scene position, so the default VS writes
                    // the device centre here, exactly as `SCENE_POSITION`. The sample closure point-pulls
                    // input `i` at this pixel — no content normalization, so an infinite-bounds simple
                    // input (its `MapOutputRectToInputRects` is unconstrained) never divides by a garbage
                    // realized size. The matching device identity tap is in `custom.rs`'s closure.
                    [px.to_bits(), py.to_bits(), 1.0f32.to_bits(), 1.0f32.to_bits()]
                }
                Role::Input(i) => {
                    let rect = input_rects.get(i as usize).copied().unwrap_or([0, 0, 1, 1]);
                    let dim = input_dims.get(i as usize).copied().unwrap_or_else(|| content_dims(rect));
                    eval_input_texcoord(&self.planes_for(i, rect, dim), rect, px, py).map(f32::to_bits)
                }
            };
        }
        inputs
    }

    /// Evaluate the shader for output pixel `(x, y)`. `sample(input_index, coord)` returns input
    /// `input_index`'s premultiplied RGBA at the device-pixel position `coord.xy` (a centre tap is the
    /// 1:1 point-pull; a neighbourhood tap is WARP-bilinear over the slot's texels — see the closure
    /// in `effects/mod.rs::eval_node`). A derivative-using shader runs as a 2×2 quad (the pixel's
    /// even-aligned quad) so `ddx`/`ddy` read real neighbour differences.
    pub fn eval(
        &self,
        x: i32,
        y: i32,
        input_rects: &[[i32; 4]],
        input_dims: &[(f32, f32)],
        sample: &dyn Fn(u32, [f32; 4]) -> [f32; 4],
        load: &dyn Fn(u32, [i32; 4]) -> [u32; 4],
        resinfo: &dyn Fn(u32) -> [u32; 4],
    ) -> [f32; 4] {
        let env = SampleEnv { sample, load, resinfo };
        let cbref = self.cbuffer.borrow();
        let cb: &[u8] = &cbref;
        if !self.uses_deriv {
            let inputs = self.fill_inputs(x as f32 + 0.5, y as f32 + 0.5, input_rects, input_dims);
            // Fast path: run the pre-decoded shader with the reusable register file (no per-pixel
            // operand decoding or register allocation). Bit-identical to `shader.run`.
            return run_predecoded_reuse(&self.shader, &self.pre, &env, &PixelInputs { inputs: &inputs, cbuffer: cb }, &mut self.scratch.borrow_mut());
        }
        // Even-aligned 2×2 quad (TL,TR,BL,BR), as the rasteriser groups pixels; return this pixel's lane.
        let (qx, qy) = (x & !1, y & !1);
        let within = ((x & 1) + 2 * (y & 1)) as usize;
        let compute = |env: &SampleEnv| {
            let pix = |lx: i32, ly: i32| self.fill_inputs(lx as f32 + 0.5, ly as f32 + 0.5, input_rects, input_dims);
            let (tl, tr, bl, br) = (pix(qx, qy), pix(qx + 1, qy), pix(qx, qy + 1), pix(qx + 1, qy + 1));
            let (tl, tr, bl, br) = (
                PixelInputs { inputs: &tl, cbuffer: cb },
                PixelInputs { inputs: &tr, cbuffer: cb },
                PixelInputs { inputs: &bl, cbuffer: cb },
                PixelInputs { inputs: &br, cbuffer: cb },
            );
            self.shader.run_quad(env, [&tl, &tr, &bl, &br])
        };
        // Reuse this quad if it was already realized this draw; otherwise run it and memoize. The cache
        // is cleared per draw (see `clear_quad_cache`), so the entry only ever reflects this draw's
        // constant buffer and inputs — bit-identical to recomputing, which `prepare_for_render` and the
        // two-draw tests guarantee.
        if let Some(q) = self.quad_cache.borrow().get(&(qx, qy)) {
            return q[within];
        }
        let q = compute(&env);
        self.quad_cache.borrow_mut().insert((qx, qy), q);
        q[within]
    }
}

/// Bridges the executor's texture `sample` to the pull engine's input sampler: forwards the register
/// and the shader's computed `coord` to the closure, which resolves it (centre tap → 1:1 point-pull,
/// neighbourhood tap → WARP bilinear over the slot).
struct SampleEnv<'a> {
    sample: &'a dyn Fn(u32, [f32; 4]) -> [f32; 4],
    load: &'a dyn Fn(u32, [i32; 4]) -> [u32; 4],
    resinfo: &'a dyn Fn(u32) -> [u32; 4],
}

impl Env for SampleEnv<'_> {
    fn sample(&self, resource: u32, _sampler: u32, coord: [f32; 4], _mode: d2d_shader::exec::Sampling) -> [f32; 4] {
        (self.sample)(resource, coord)
    }
    // `ld`/`resinfo` forward to the bound resource textures (SetResourceTexture); without this a
    // resource-texture Load (e.g. AddNoise's noise LUT) read zero and the shader fell through.
    fn load(&self, resource: u32, coord: [i32; 4]) -> [u32; 4] {
        (self.load)(resource, coord)
    }
    fn resinfo(&self, resource: u32) -> [u32; 4] {
        (self.resinfo)(resource)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Our own authored shader (fxshader-01-scale-bias.hlsl) compiled on the VM with fxc; the same
    // bytes the conformance scene embeds, read from the suite at runtime (skip when it is absent).
    // Checks the realization wiring: Shader::parse, the input register the sample reads, the b0 byte
    // layout, and the mad_sat + alpha-passthrough output.
    #[test]
    fn scale_bias_pixel() {
        let Some(suite) = d2d_reference::suite_dir() else { return };
        let dxbc = std::fs::read(suite.join("scenes/fxshader-01-scale-bias.dxbc")).unwrap();
        let shader = Rc::new(Shader::parse(&dxbc).expect("parse"));
        let k = [0.5f32, 0.1, -0.2, 0.0]; // scale, then per-channel RGB bias
        let cbuffer: Vec<u8> = k.iter().flat_map(|f| f.to_le_bytes()).collect();
        let node = ShaderNode::new(shader, Rc::new(RefCell::new(cbuffer)), 1, Rc::default(), 0).expect("build");
        // Opaque premultiplied input texel; the sampler is asked for input 0 only.
        let input = [128.0 / 255.0, 64.0 / 255.0, 200.0 / 255.0, 1.0];
        let out = node.eval(
            3,
            7,
            &[[0, 0, 1, 1]],
            &[(1.0, 1.0)],
            &|res, _coord| {
                assert_eq!(res, 0);
                input
            },
            &|_res, _coord| [0; 4],
            &|_res| [0; 4],
        );
        // o0.xyz = saturate(in.rgb * k.x + k.yzw); o0.w = in.a. f32 mul-then-add matches the executor.
        let want = |c: usize| (input[c] * k[0] + k[1 + c]).clamp(0.0, 1.0);
        assert_eq!(out, [want(0), want(1), want(2), input[3]]);
    }

    use d2d_shader::sm4::{Index, Inst, Operand, Program, Sel};

    /// A minimal derivative shader whose output is the constant buffer: `deriv_rtx r0, v0` (marks the
    /// shader `uses_deriv`, so eval takes the 2×2-quad cache path) then `mov o0, cb0[0]`. The output
    /// depends only on `b0`, which the test varies between draws to probe the quad cache.
    fn deriv_cb_shader() -> Rc<Shader> {
        let v0 = Operand { ty: 1, num_components: 4, sel: Sel::Swizzle([0, 1, 2, 3]), modifier: 0, indices: vec![Index::Imm(0)], imm: vec![] };
        let r0 = Operand { ty: 0, num_components: 4, sel: Sel::Mask(0xf), modifier: 0, indices: vec![Index::Imm(0)], imm: vec![] };
        let o0 = Operand { ty: 2, num_components: 4, sel: Sel::Mask(0xf), modifier: 0, indices: vec![Index::Imm(0)], imm: vec![] };
        let cb0 = Operand { ty: 8, num_components: 4, sel: Sel::Swizzle([0, 1, 2, 3]), modifier: 0, indices: vec![Index::Imm(0), Index::Imm(0)], imm: vec![] };
        let inst = |op, operands| Inst { op, saturate: false, test_nz: false, operands, decl_count: 0, aux: 0 };
        let program = Program {
            program_type: 0, major: 5, minor: 0,
            insts: vec![inst(Opcode::DerivRtx, vec![r0, v0]), inst(Opcode::Mov, vec![o0, cb0])],
            temps: 1, indexable_temps: vec![], thread_group: [0; 3], uavs: vec![], tgsm: vec![], resources: vec![], icb: vec![],
        };
        Rc::new(Shader { program, inputs: vec![], outputs: vec![], input_simple: vec![] })
    }

    fn cbuf(v: f32) -> Vec<u8> {
        [v, 0.0, 0.0, 0.0].iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    // The derivative-quad cache memoizes a quad for the draw and `clear_quad_cache` (what
    // `prepare_for_render` calls before every draw) drops it, so a quad computed with one constant
    // buffer can never leak into a later draw that binds a different one — the cross-draw parity bug.
    #[test]
    fn deriv_quad_cache_memoizes_then_clears() {
        let cb = Rc::new(RefCell::new(cbuf(0.25)));
        let node = ShaderNode::new(deriv_cb_shader(), cb.clone(), 0, Rc::default(), 0).expect("build");
        assert!(node.uses_deriv, "the shader must take the quad path");
        let eval = |n: &ShaderNode| n.eval(2, 2, &[], &[], &|_r, _c| [0.0; 4], &|_r, _c| [0; 4], &|_r| [0; 4]);

        // Draw 1 (b0 = 0.25): output is the constant buffer, and the quad is memoized.
        assert_eq!(eval(&node)[0], 0.25);

        // Rebind b0 = 0.75 but DON'T clear: the memoized quad still returns 0.25, so a stale quad would
        // leak if the cache were not cleared per draw — this is exactly the bug the clear prevents.
        *cb.borrow_mut() = cbuf(0.75);
        assert_eq!(eval(&node)[0], 0.25, "quad is memoized within a draw");

        // clear_quad_cache (the per-draw invalidation) → the next draw recomputes with the new b0.
        node.clear_quad_cache();
        assert_eq!(eval(&node)[0], 0.75, "cleared cache picks up the new constant buffer");
    }
}
