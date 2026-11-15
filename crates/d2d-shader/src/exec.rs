//! CPU executor for a decoded SM4/5 shader, reproducing the arithmetic WARP's JIT emits:
//! IEEE round-to-nearest-even float32, per-operation subnormal flush-to-signed-zero (raw `mov`
//! preserves bits), NaN-passing `min`/`max`, and the documented integer/conversion/rounding rules.
//!
//! The engine is lane-parallel (SIMT): one [`Machine`] holds N invocations and steps them through a
//! single program counter with a per-lane active mask, so structured control flow, derivatives
//! (which read a 2×2 quad's neighbour lanes) and compute thread groups (group-shared memory,
//! `sync`, atomics) all run on the same driver. A pixel shader is one lane ([`Shader::run`]); a
//! derivative-using pixel shader is a quad of four ([`Shader::run_quad`]); a compute dispatch runs
//! each thread group as N lanes ([`Shader::run_compute`]). Texture sampling and typed resource
//! loads are delegated to a caller-supplied [`Env`]; UAV / group-shared / structured memory is held
//! in the machine so the harness can read it back.
//!
//! Each lane stores every register as raw `u32`s so bit-preserving moves and float/int
//! reinterpretation are exact. Transcendentals (`exp2`/`log2`/`rsq`/`rcp`/`sqrt`/`frc`/`sincos`)
//! route through crates/d2d's `warpmath`, reproduced bit-for-bit from the x86 WARP's JIT emits, so
//! they match WARP exactly.

use crate::dxbc::Shader;
use crate::sm4::{Index, Inst, Mem, Opcode, Operand, Program, Sel};

/// Which sampling operation a `sample*`/`gather4` instruction performs. For the single-mip textures
/// the harness builds, `Lod`/`Bias`/`Grad` all resolve to the base level, as they do on WARP.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Sampling {
    Sample,
    Lod(f32),
    Bias(f32),
    Grad { ddx: [f32; 3], ddy: [f32; 3] },
    /// `sample_c`/`sample_c_lz`: compare the sampled value against `reference` (0/1 result).
    Cmp { reference: f32, lz: bool },
    /// `gather4`/`gather4_po`: the four footprint texels of channel `component` (0=r,1=g,2=b,3=a),
    /// the footprint shifted by an integer texel `offset` (`[0,0]` for plain `gather4`).
    Gather { component: u32, offset: [i32; 2] },
    /// `gather4_c`: each of the four footprint texels' first channel compared against `reference`
    /// (0/1).
    GatherCmp { reference: f32 },
}

/// Texture environment: the resources/samplers a shader reads. Implemented by the effect graph
/// (wrapping crates/d2d's WARP-exact `sampling.rs`) and by the test harness.
pub trait Env {
    /// `sample*`/`gather4 t#, s#` at the given coordinate; returns float components (RGBA).
    fn sample(&self, resource: u32, sampler: u32, coord: [f32; 4], mode: Sampling) -> [f32; 4];
    /// `ld t#` at integer texel coords (`.xyz` = u,v/array/element, `.w` = mip); raw components.
    fn load(&self, _resource: u32, _coord: [i32; 4]) -> [u32; 4] {
        [0; 4]
    }
    /// `resinfo` dimensions of `resource`: `(width, height, depth-or-elements, mip count)`.
    fn resinfo(&self, _resource: u32) -> [u32; 4] {
        [0; 4]
    }
    /// `sampleinfo` sample count of `resource` (1 for non-multisampled).
    fn sample_count(&self, _resource: u32) -> u32 {
        1
    }
}

/// Per-pixel inputs: the interpolated input registers (`v#`) and the `b0` constant-buffer bytes.
pub struct PixelInputs<'a> {
    /// Input registers indexed by register number; each is four raw lanes.
    pub inputs: &'a [[u32; 4]],
    /// Constant buffer bound at `b0` (16 bytes per `cb0[n]` element).
    pub cbuffer: &'a [u8],
}

/// A UAV / structured-or-raw resource buffer the harness binds for a compute dispatch. `data` is the
/// backing bytes (typed UAVs store packed texels; raw/structured store byte-addressed values); it is
/// read back after [`Shader::run_compute`].
#[derive(Clone)]
pub struct UavDesc {
    pub reg: u32,
    pub kind: Mem,
    /// Texture dimensions for typed UAVs (`[w, h, _]`); ignored for raw/structured.
    pub dims: [u32; 3],
    pub data: Vec<u8>,
}

/// Flush a subnormal to signed zero, as WARP does on f32 arithmetic inputs/outputs.
#[inline]
fn fl(x: f32) -> f32 {
    if x.is_subnormal() {
        (0.0f32).copysign(x)
    } else {
        x
    }
}

/// Saturate: NaN→0, clamp to `[0,1]` (output-merger / `_sat` rule). WARP emits `max(0, min(1, x))`,
/// and the `max` against `+0` turns a `-0.0` result into `+0.0` (where `clamp` would keep `-0.0`).
#[inline]
fn sat(x: f32) -> f32 {
    if x.is_nan() {
        0.0
    } else {
        // `+ 0.0` normalises a `-0.0` clamp result to `+0.0` and leaves every other value unchanged.
        x.clamp(0.0, 1.0) + 0.0
    }
}

/// Convert a shader float output to a UNORM8 byte: round-to-nearest-even of `saturate(x)*255`.
pub fn to_unorm8(x: f32) -> u8 {
    (sat(x) * 255.0).round_ties_even() as u8
}

/// The channel a `gather4*` selects, from its sampler operand's single-component swizzle.
fn gather_comp(op: &Operand) -> u32 {
    match op.sel {
        Sel::Select1(c) => c as u32,
        Sel::Swizzle(s) => s[0] as u32,
        _ => 0,
    }
}

/// One SIMT invocation's register file plus its control state.
#[derive(Clone)]
struct Lane {
    r: Vec<[u32; 4]>,
    x: Vec<Vec<[u32; 4]>>,
    o: Vec<[u32; 4]>,
    /// Input registers `v#` (pixel shaders).
    v: Vec<[u32; 4]>,
    /// Compute system values: dispatch id, group id, group-thread id, flattened group index.
    tid: [u32; 3],
    gid: [u32; 3],
    gtid: [u32; 3],
    flat: u32,
    active: bool,
    returned: bool,
}

impl Lane {
    fn new(prog: &Program) -> Lane {
        Lane {
            r: vec![[0; 4]; prog.temps as usize],
            // Each indexable temp `x#` gets one guard row past its declared size (zero-initialised).
            // WARP clamps a dynamic index to the register size, so an out-of-range row read or write
            // lands in this guard slot: both a read and a write go there, so an out-of-range write
            // followed by an out-of-range read returns the written value (observable, so it is ported
            // exactly rather than reading 0).
            x: prog.indexable_temps.iter().map(|&(_, n)| vec![[0u32; 4]; n as usize + 1]).collect(),
            o: vec![[0; 4]; 1],
            v: Vec::new(),
            tid: [0; 3],
            gid: [0; 3],
            gtid: [0; 3],
            flat: 0,
            active: true,
            returned: false,
        }
    }

    /// Reset the register file in place for the next pixel: zero the temps / indexable temps /
    /// outputs (the reference allocates them fresh and zeroed per pixel), load the input registers,
    /// and re-arm the lane. Reuses the existing allocations.
    fn reset(&mut self, inputs: &[[u32; 4]]) {
        for r in self.r.iter_mut() {
            *r = [0; 4];
        }
        for bank in self.x.iter_mut() {
            for row in bank.iter_mut() {
                *row = [0; 4];
            }
        }
        for o in self.o.iter_mut() {
            *o = [0; 4];
        }
        self.v.clear();
        self.v.extend_from_slice(inputs);
        self.active = true;
        self.returned = false;
    }
}

struct Machine<'a> {
    env: &'a dyn Env,
    /// Constant buffers indexed by register (`b0..b13`); a `cb[r][n]` read selects `cbuffers[r]`.
    /// Shorter than the referenced register reads empty (zeros), as a missing binding would.
    cbuffers: &'a [&'a [u8]],
    /// `dcl_immediateConstantBuffer` data (operand type 9 indexes into it).
    icb: &'a [[u32; 4]],
    lanes: Vec<Lane>,
    /// `true` while running a 2×2 quad so derivatives read neighbour lanes.
    quad: bool,
    /// Group-shared memory, one byte buffer per `g#` (reg, bytes), fresh per group.
    tgsm: Vec<(u32, Vec<u8>)>,
    /// UAV buffers (shared across the whole dispatch).
    uav: &'a mut [UavDesc],
    /// Read-only structured/raw resource buffers for compute.
    res: &'a [UavDesc],
}

impl Shader {
    /// The first opcode this executor does not implement, as its raw opcode number, or `None` if the
    /// whole program is runnable. The caller checks this before running and reports the blockage
    /// rather than letting `run` panic (owner rule: never silently zero an unimplemented opcode).
    pub fn unsupported_op(&self) -> Option<u16> {
        self.program.insts.iter().find_map(|i| {
            match i.op {
                Opcode::LdMs => Some(46),
                Opcode::Lod => Some(108),
                Opcode::SamplePos => Some(110),
                Opcode::ImmAtomicAlloc => Some(178),
                Opcode::ImmAtomicConsume => Some(179),
                Opcode::Other(code) if !is_declaration(code) => Some(code),
                _ => None,
            }
        })
    }

    /// Run this pixel shader for one output pixel; returns the `o0` output as premultiplied RGBA
    /// floats (see [`to_unorm8`] for an 8-bit target). Shaders that use derivatives need
    /// [`Shader::run_quad`].
    pub fn run(&self, env: &dyn Env, px: &PixelInputs) -> [f32; 4] {
        run(self, env, px)
    }

    /// Run this pixel shader for a 2×2 quad (lanes TL, TR, BL, BR) in lockstep, so `deriv_*` read the
    /// neighbour lanes; returns each lane's `o0`.
    pub fn run_quad(&self, env: &dyn Env, quad: [&PixelInputs; 4]) -> [[f32; 4]; 4] {
        let prog = &self.program;
        let mut m = Machine {
            env,
            cbuffers: &[quad[0].cbuffer],
            icb: &prog.icb,
            lanes: (0..4).map(|_| Lane::new(prog)).collect(),
            quad: true,
            tgsm: Vec::new(),
            uav: &mut [],
            res: &[],
        };
        for (l, q) in quad.iter().enumerate() {
            m.lanes[l].v = q.inputs.to_vec();
        }
        m.run(prog, &prog.insts);
        core::array::from_fn(|l| out_floats(&m.lanes[l]))
    }

    /// Run a compute dispatch: execute every thread group of `groups` (x,y,z group counts), sharing
    /// `uavs` (mutated in place) and reading `resources`. `cbuffer` binds at `b0` (a raw dispatch).
    pub fn run_compute(&self, env: &dyn Env, cbuffer: &[u8], groups: [u32; 3], uavs: &mut [UavDesc], resources: &[UavDesc]) {
        self.run_compute_bufs(env, &[cbuffer], groups, uavs, resources)
    }

    /// As [`Self::run_compute`], but binding a constant buffer per register (`cbuffers[r]` at `b{r}`).
    /// A D2D compute transform reserves `b0` for D2D's own system constant buffer and binds the user
    /// buffer (`SetComputeShaderConstantBuffer`) at `b1` — VM-verified — so that path passes
    /// `&[&[], user]`; a raw dispatch passes `&[cbuffer]`.
    pub fn run_compute_bufs(&self, env: &dyn Env, cbuffers: &[&[u8]], groups: [u32; 3], uavs: &mut [UavDesc], resources: &[UavDesc]) {
        let prog = &self.program;
        let [tx, ty, tz] = prog.thread_group;
        let n = (tx * ty * tz) as usize;
        for gz in 0..groups[2] {
            for gy in 0..groups[1] {
                for gx in 0..groups[0] {
                    let mut m = Machine {
                        env,
                        cbuffers,
                        icb: &prog.icb,
                        lanes: (0..n).map(|_| Lane::new(prog)).collect(),
                        quad: false,
                        tgsm: prog.tgsm.iter().map(|d| (d.reg, vec![0u8; mem_bytes(d.mem)])).collect(),
                        uav: uavs,
                        res: resources,
                    };
                    for (idx, lane) in m.lanes.iter_mut().enumerate() {
                        let li = idx as u32;
                        let (lx, ly, lz) = (li % tx, (li / tx) % ty, li / (tx * ty));
                        lane.gtid = [lx, ly, lz];
                        lane.gid = [gx, gy, gz];
                        lane.tid = [gx * tx + lx, gy * ty + ly, gz * tz + lz];
                        lane.flat = li;
                    }
                    m.run(prog, &prog.insts);
                    // Lanes borrow `uavs` through `m`; drop before the next group.
                }
            }
        }
    }

    /// Run this as a vertex shader for one vertex: `inputs` are the `v#` input registers (vertex
    /// attributes; `SV_VertexID`/`SV_InstanceID` as raw `u32` in their register's `.x`), `cbuffer`
    /// binds at `b0`. Returns every output register `o#` (the output signature's `SV_Position` plus
    /// varyings) as float4, indexed by register number — the triangle attributes the rasterizer and
    /// pixel shader then consume.
    pub fn run_vertex(&self, env: &dyn Env, inputs: &[[u32; 4]], cbuffer: &[u8]) -> Vec<[f32; 4]> {
        self.run_vertex_bufs(env, inputs, &[cbuffer])
    }

    /// As [`Self::run_vertex`], but binding a constant buffer per register (`cbuffers[r]` at `b{r}`).
    /// A shader that reads several `b#` slots (e.g. the ink body VS's `b0` atlas→NDC + `b1` transform
    /// array) needs each bound separately; the single-buffer [`Self::run_vertex`] only binds `b0`.
    pub fn run_vertex_bufs(&self, env: &dyn Env, inputs: &[[u32; 4]], cbuffers: &[&[u8]]) -> Vec<[f32; 4]> {
        let prog = &self.program;
        let n_out = self.outputs.iter().map(|e| e.register as usize + 1).max().unwrap_or(1);
        let mut lane = Lane::new(prog);
        lane.o = vec![[0; 4]; n_out];
        lane.v = inputs.to_vec();
        let mut m = Machine { env, cbuffers, icb: &prog.icb, lanes: vec![lane], quad: false, tgsm: Vec::new(), uav: &mut [], res: &[] };
        m.run(prog, &prog.insts);
        m.lanes[0].o.iter().map(|o| [f32::from_bits(o[0]), f32::from_bits(o[1]), f32::from_bits(o[2]), f32::from_bits(o[3])]).collect()
    }
}

fn mem_bytes(m: Mem) -> usize {
    match m {
        Mem::Raw { bytes } => bytes as usize,
        Mem::Structured { bytes, .. } => bytes as usize,
        Mem::Typed(_) => 0,
    }
}

fn out_floats(l: &Lane) -> [f32; 4] {
    let o = l.o[0];
    [f32::from_bits(o[0]), f32::from_bits(o[1]), f32::from_bits(o[2]), f32::from_bits(o[3])]
}

/// Run a pixel shader for one pixel; returns the `o0` output as four floats.
pub fn run(shader: &Shader, env: &dyn Env, px: &PixelInputs) -> [f32; 4] {
    let prog = &shader.program;
    let mut lane = Lane::new(prog);
    lane.v = px.inputs.to_vec();
    let mut m = Machine { env, cbuffers: &[px.cbuffer], icb: &prog.icb, lanes: vec![lane], quad: false, tgsm: Vec::new(), uav: &mut [], res: &[] };
    m.run(prog, &prog.insts);
    out_floats(&m.lanes[0])
}

/// Run a pixel shader for one pixel through the pre-decoded executor. Bit-for-bit identical to
/// [`run`]: the pre-decode pass only resolves per-operand swizzles, write masks and register slots
/// ahead of the per-pixel loop, keeping every IEEE operation (and its order) unchanged. The
/// differential test (`tests/predecode_diff.rs`) asserts `run == run_pre` across the shader corpora
/// and a battery of special-value inputs, so this stays exact as the pre-decode pass grows.
pub fn run_pre(shader: &Shader, env: &dyn Env, px: &PixelInputs) -> [f32; 4] {
    run_predecoded(shader, &predecode_shader(shader), env, px)
}

/// A shader lowered once for fast repeated execution. Built per shader by [`predecode_shader`] and
/// hoisted out of the per-pixel loop; every pixel runs through [`run_predecoded`], so the per-op
/// operand decoding (swizzle, bank, index, modifier, write mask) happens once per shader, not per
/// pixel. Bit-identical to [`run`] (the differential test guarantees it).
pub struct Predecoded {
    insts: Vec<Inst<ResolvedOperand>>,
}

/// Lower `shader` once for repeated pixel execution.
pub fn predecode_shader(shader: &Shader) -> Predecoded {
    Predecoded { insts: predecode(&shader.program) }
}

/// Run one pixel through a pre-lowered shader, without re-decoding the operands.
pub fn run_predecoded(shader: &Shader, pre: &Predecoded, env: &dyn Env, px: &PixelInputs) -> [f32; 4] {
    let prog = &shader.program;
    let mut lane = Lane::new(prog);
    lane.v = px.inputs.to_vec();
    let mut m = Machine { env, cbuffers: &[px.cbuffer], icb: &prog.icb, lanes: vec![lane], quad: false, tgsm: Vec::new(), uav: &mut [], res: &[] };
    m.run(prog, &pre.insts);
    out_floats(&m.lanes[0])
}

/// A pre-lowered shader bound to a constant buffer and environment for one draw, executing a tile of
/// pixels while reusing its register file: the shader is decoded once ([`predecode_shader`]) and the
/// per-pixel register buffers are reset in place instead of reallocated, so both the per-op operand
/// decoding and the per-pixel allocation leave the hot loop. Build one per draw and call [`pixel`]
/// for every pixel. Bit-identical to [`run`].
pub struct TileExec<'a> {
    prog: &'a Program,
    insts: &'a [Inst<ResolvedOperand>],
    m: Machine<'a>,
}

impl Shader {
    /// Bind this shader for a tile draw. `cbuffers` is the per-register constant-buffer slice (for a
    /// pixel shader, `&[b0]`); it and `env` live for the draw.
    pub fn tile_exec<'a>(&'a self, pre: &'a Predecoded, env: &'a dyn Env, cbuffers: &'a [&'a [u8]]) -> TileExec<'a> {
        let prog = &self.program;
        TileExec {
            prog,
            insts: &pre.insts,
            m: Machine { env, cbuffers, icb: &prog.icb, lanes: vec![Lane::new(prog)], quad: false, tgsm: Vec::new(), uav: &mut [], res: &[] },
        }
    }
}

impl TileExec<'_> {
    /// Run one pixel: reset the register file in place, set the input registers, execute.
    pub fn pixel(&mut self, inputs: &[[u32; 4]]) -> [f32; 4] {
        self.m.lanes[0].reset(inputs);
        self.m.run(self.prog, self.insts);
        out_floats(&self.m.lanes[0])
    }
}

/// A reusable register file for running a pre-decoded shader without per-pixel allocation when the
/// environment changes per pixel (the effect-graph `eval` path builds a fresh sampler env per pixel,
/// so it can't hold a [`TileExec`]). Build one per shader and run every pixel through
/// [`run_predecoded_reuse`], which resets the register file in place. Bit-identical to [`run`].
pub struct ExecScratch {
    lanes: Vec<Lane>,
}

impl ExecScratch {
    pub fn new(shader: &Shader) -> ExecScratch {
        ExecScratch { lanes: vec![Lane::new(&shader.program)] }
    }
}

/// Run one pixel through a pre-lowered shader with a reusable register file and a per-call
/// environment / constant buffer. Resets `scratch` in place instead of reallocating.
pub fn run_predecoded_reuse(shader: &Shader, pre: &Predecoded, env: &dyn Env, px: &PixelInputs, scratch: &mut ExecScratch) -> [f32; 4] {
    let prog = &shader.program;
    scratch.lanes[0].reset(px.inputs);
    // Lend the reusable register file to a fresh Machine (the env/cbuffer bindings change per pixel),
    // then take it back so its allocations persist for the next pixel.
    let lanes = core::mem::take(&mut scratch.lanes);
    let mut m = Machine { env, cbuffers: &[px.cbuffer], icb: &prog.icb, lanes, quad: false, tgsm: Vec::new(), uav: &mut [], res: &[] };
    m.run(prog, &pre.insts);
    let out = out_floats(&m.lanes[0]);
    scratch.lanes = m.lanes;
    out
}

/// A control-flow scope for SIMT reconvergence. `If` remembers the mask before the branch; `Loop`
/// remembers the entry mask and which lanes have `break`'d / `continue`'d this iteration. `Switch`
/// remembers the entry mask, each lane's selector, the lanes a label has selected (they run on, through
/// any later labels, until they `break`), the lanes that `break` out, and the switch's own case values
/// (`default` selects the lanes none of them match, wherever it sits).
enum Frame {
    If { prev: Mask, taken: Mask },
    Loop { entry: Mask, start: usize, broken: Mask, continued: Mask },
    Switch { entry: Mask, selector: Vec<u32>, selected: Mask, broken: Mask, cases: Vec<u32> },
}

/// A per-lane active mask. Inline bits for the pixel/quad/small-compute case (`n <= 64`, zero-alloc) and
/// a word vector for a large thread group — so the control-flow ops no longer heap-allocate a `Vec<bool>`
/// every If/Loop/Break on every loop iteration (the hot cost for a loop-heavy shader, and worse in wasm).
#[derive(Clone)]
enum Mask {
    Bits(u64),
    Wide(Vec<u64>),
}
impl Mask {
    #[inline]
    fn zeros(n: usize) -> Mask {
        if n <= 64 {
            Mask::Bits(0)
        } else {
            Mask::Wide(vec![0; n.div_ceil(64)])
        }
    }
    #[inline]
    fn build(n: usize, mut f: impl FnMut(usize) -> bool) -> Mask {
        if n <= 64 {
            let mut b = 0u64;
            for l in 0..n {
                if f(l) {
                    b |= 1u64 << l;
                }
            }
            Mask::Bits(b)
        } else {
            let mut w = vec![0u64; n.div_ceil(64)];
            for l in 0..n {
                if f(l) {
                    w[l / 64] |= 1u64 << (l % 64);
                }
            }
            Mask::Wide(w)
        }
    }
    #[inline]
    fn get(&self, l: usize) -> bool {
        match self {
            Mask::Bits(b) => (b >> l) & 1 != 0,
            Mask::Wide(w) => (w[l / 64] >> (l % 64)) & 1 != 0,
        }
    }
    #[inline]
    fn set(&mut self, l: usize, v: bool) {
        let bit = |word: &mut u64, m: u64| {
            if v {
                *word |= m;
            } else {
                *word &= !m;
            }
        };
        match self {
            Mask::Bits(b) => bit(b, 1u64 << l),
            Mask::Wide(w) => bit(&mut w[l / 64], 1u64 << (l % 64)),
        }
    }
    #[inline]
    fn any(&self) -> bool {
        match self {
            Mask::Bits(b) => *b != 0,
            Mask::Wide(w) => w.iter().any(|&x| x != 0),
        }
    }
}

/// The `case` values of the switch whose `switch` is at `pc` (not those of a nested switch).
fn switch_cases<P: Oper>(insts: &[Inst<P>], pc: usize) -> Vec<u32> {
    let mut depth = 0usize;
    let mut cases = Vec::new();
    for inst in &insts[pc + 1..] {
        match inst.op {
            Opcode::Switch => depth += 1,
            Opcode::Endswitch if depth == 0 => break,
            Opcode::Endswitch => depth -= 1,
            Opcode::Case if depth == 0 => cases.push(inst.operands[0].imm0()),
            _ => {}
        }
    }
    cases
}

/// Operand access abstracted so one `op_lane` / `run` executes both the reference [`Operand`] stream
/// and the pre-decoded operand stream. The reference impl resolves the bank, swizzle, modifier and
/// write mask by matching the operand's fields every time; the pre-decoded impl ([`ResolvedOperand`])
/// returns values resolved once by [`predecode`]. Results are bit-identical, which the differential
/// test guarantees.
trait Oper: Clone {
    /// Raw bits after base-register resolution and swizzle (no source modifier).
    fn swizzled(&self, m: &Machine<'_>, l: usize) -> [u32; 4];
    /// Source modifier bits (1 = neg, 2 = abs).
    fn modifier(&self) -> u8;
    /// Destination write mask (bit i = component i).
    fn dst_mask(&self) -> u8;
    /// Mutable destination slot, or None if out of range (the write is then dropped).
    fn slot_mut<'a>(&self, m: &'a mut Machine<'_>, l: usize) -> Option<&'a mut [u32; 4]>;
    /// The four swizzle components (identity `[0,1,2,3]` for a mask / no selection), used to swizzle a
    /// sampled value through a resource operand.
    fn swizzle(&self) -> [u8; 4];
    /// The component a `gather4` selects from its sampler operand's swizzle.
    fn sel_comp(&self) -> u32;
    /// The first immediate value (a `case` label, or `switch`/`case` selector constant).
    fn imm0(&self) -> u32;
    /// The operand-type number (13 = null, etc.).
    fn ty(&self) -> u8;
    /// The register number (first index), for resource / UAV / groupshared operands.
    fn reg(&self) -> u32;
}

impl Oper for Operand {
    fn swizzled(&self, m: &Machine<'_>, l: usize) -> [u32; 4] {
        m.base_swizzle(l, self)
    }
    fn modifier(&self) -> u8 {
        self.modifier
    }
    fn dst_mask(&self) -> u8 {
        Machine::dst_mask_op(self)
    }
    fn slot_mut<'a>(&self, m: &'a mut Machine<'_>, l: usize) -> Option<&'a mut [u32; 4]> {
        m.slot_mut_op(l, self)
    }
    fn swizzle(&self) -> [u8; 4] {
        match self.sel {
            Sel::Swizzle(s) => s,
            Sel::Select1(c) => [c; 4],
            _ => [0, 1, 2, 3],
        }
    }
    fn sel_comp(&self) -> u32 {
        gather_comp(self)
    }
    fn imm0(&self) -> u32 {
        self.imm.first().copied().unwrap_or(0)
    }
    fn ty(&self) -> u8 {
        self.ty
    }
    fn reg(&self) -> u32 {
        match self.indices.first() {
            Some(Index::Imm(v)) => *v as u32,
            _ => 0,
        }
    }
}

/// A pre-decoded operand: everything the per-pixel hot path recomputed per op is resolved once — the
/// bank type, its immediate indices, the swizzle, source modifier, write mask and immediate payload.
/// An operand with a register-relative index can't be fully resolved ahead of time (the index is a
/// per-lane value), so it keeps the reference [`Operand`] and takes the matching path.
#[derive(Clone)]
enum ResolvedOperand {
    Fast { ty: u8, i0: usize, i1: usize, swz: [u8; 4], modifier: u8, mask: u8, reg: u32, imm: Vec<u32> },
    Slow(Operand),
}

/// Lower a reference operand once. Fast-pathed unless it has a register-relative index.
fn resolve_operand(op: &Operand) -> ResolvedOperand {
    if !op.indices.iter().all(|i| matches!(i, Index::Imm(_))) {
        return ResolvedOperand::Slow(op.clone());
    }
    let imm_at = |n: usize| match op.indices.get(n) {
        Some(Index::Imm(v)) => *v as usize,
        _ => 0,
    };
    let (i0, i1) = match op.ty {
        9 => (
            match op.indices.last() {
                Some(Index::Imm(v)) => *v as usize,
                _ => 0,
            },
            0,
        ),
        _ => (imm_at(0), imm_at(1)),
    };
    ResolvedOperand::Fast {
        ty: op.ty,
        i0,
        i1,
        swz: op.swizzle(),
        modifier: op.modifier,
        mask: Machine::dst_mask_op(op),
        reg: op.reg(),
        imm: op.imm.clone(),
    }
}

/// Lower a whole program's instruction stream once (hoist this out of the per-pixel loop).
fn predecode(prog: &Program) -> Vec<Inst<ResolvedOperand>> {
    prog.insts
        .iter()
        .map(|i| Inst {
            op: i.op,
            saturate: i.saturate,
            test_nz: i.test_nz,
            operands: i.operands.iter().map(resolve_operand).collect(),
            decl_count: i.decl_count,
            aux: i.aux,
        })
        .collect()
}

impl Oper for ResolvedOperand {
    #[inline(always)]
    fn swizzled(&self, m: &Machine<'_>, l: usize) -> [u32; 4] {
        match self {
            ResolvedOperand::Slow(op) => m.base_swizzle(l, op),
            ResolvedOperand::Fast { ty, i0, i1, swz, imm, .. } => {
                let b = m.base_read(l, *ty, *i0, *i1, imm);
                // Identity swizzle (.xyzw) is the common full-register read: skip the reorder.
                if *swz == [0, 1, 2, 3] {
                    return b;
                }
                [b[swz[0] as usize], b[swz[1] as usize], b[swz[2] as usize], b[swz[3] as usize]]
            }
        }
    }
    fn modifier(&self) -> u8 {
        match self {
            ResolvedOperand::Fast { modifier, .. } => *modifier,
            ResolvedOperand::Slow(op) => op.modifier,
        }
    }
    fn dst_mask(&self) -> u8 {
        match self {
            ResolvedOperand::Fast { mask, .. } => *mask,
            ResolvedOperand::Slow(op) => Machine::dst_mask_op(op),
        }
    }
    #[inline(always)]
    fn slot_mut<'a>(&self, m: &'a mut Machine<'_>, l: usize) -> Option<&'a mut [u32; 4]> {
        match self {
            ResolvedOperand::Slow(op) => m.slot_mut_op(l, op),
            ResolvedOperand::Fast { ty, i0, i1, .. } => {
                let lane = &mut m.lanes[l];
                match ty {
                    0 => lane.r.get_mut(*i0),
                    2 => lane.o.get_mut(*i0),
                    3 => lane.x.get_mut(*i0).map(|b| {
                        let g = b.len() - 1;
                        &mut b[(*i1).min(g)]
                    }),
                    _ => None,
                }
            }
        }
    }
    fn swizzle(&self) -> [u8; 4] {
        match self {
            ResolvedOperand::Fast { swz, .. } => *swz,
            ResolvedOperand::Slow(op) => op.swizzle(),
        }
    }
    fn sel_comp(&self) -> u32 {
        match self {
            ResolvedOperand::Fast { swz, .. } => swz[0] as u32,
            ResolvedOperand::Slow(op) => gather_comp(op),
        }
    }
    fn imm0(&self) -> u32 {
        match self {
            ResolvedOperand::Fast { imm, .. } => imm.first().copied().unwrap_or(0),
            ResolvedOperand::Slow(op) => op.imm.first().copied().unwrap_or(0),
        }
    }
    fn ty(&self) -> u8 {
        match self {
            ResolvedOperand::Fast { ty, .. } => *ty,
            ResolvedOperand::Slow(op) => op.ty,
        }
    }
    fn reg(&self) -> u32 {
        match self {
            ResolvedOperand::Fast { reg, .. } => *reg,
            ResolvedOperand::Slow(op) => op.reg(),
        }
    }
}

impl Machine<'_> {
    fn n(&self) -> usize {
        self.lanes.len()
    }

    /// Step all lanes through the program with a shared PC and per-lane active masks.
    fn run<P: Oper>(&mut self, prog: &Program, insts: &[Inst<P>]) {
        let n = self.n();
        let mut cf: Vec<Frame> = Vec::new();
        let mut pc = 0usize;
        while pc < insts.len() {
            let inst = &insts[pc];
            use Opcode::*;
            match inst.op {
                Ret => {
                    for l in 0..n {
                        if self.lanes[l].active {
                            self.lanes[l].returned = true;
                            self.lanes[l].active = false;
                        }
                    }
                    pc += 1;
                }
                If => {
                    let prev = Mask::build(n, |l| self.lanes[l].active);
                    let taken = Mask::build(n, |l| prev.get(l) && self.cond(l, inst));
                    for l in 0..n {
                        self.lanes[l].active = taken.get(l);
                    }
                    cf.push(Frame::If { prev, taken });
                    pc += 1;
                }
                Else => {
                    let base: Option<Mask> = if let Some(Frame::If { prev, taken }) = cf.last() {
                        Some(Mask::build(n, |l| prev.get(l) && !taken.get(l)))
                    } else {
                        None
                    };
                    if let Some(base) = base {
                        self.restore(&base, &cf);
                    }
                    pc += 1;
                }
                Endif => {
                    if let Some(Frame::If { prev, .. }) = cf.pop() {
                        self.restore(&prev, &cf);
                    }
                    pc += 1;
                }
                Loop => {
                    let entry = Mask::build(n, |l| self.lanes[l].active);
                    cf.push(Frame::Loop { entry, start: pc + 1, broken: Mask::zeros(n), continued: Mask::zeros(n) });
                    pc += 1;
                }
                Switch => {
                    // No lane runs until a label selects it.
                    let entry = Mask::build(n, |l| self.lanes[l].active);
                    let selector: Vec<u32> = (0..n).map(|l| self.src_u(l, &inst.operands[0])[0]).collect();
                    for l in 0..n {
                        self.lanes[l].active = false;
                    }
                    let cases = switch_cases(insts, pc);
                    cf.push(Frame::Switch { entry, selector, selected: Mask::zeros(n), broken: Mask::zeros(n), cases });
                    pc += 1;
                }
                Case | Default => {
                    let run: Option<Mask> = if let Some(Frame::Switch { entry, selector, selected, broken, cases }) = cf.last_mut() {
                        for l in 0..n {
                            let hit = if inst.op == Case { selector[l] == inst.operands[0].imm0() } else { !cases.contains(&selector[l]) };
                            if entry.get(l) && hit {
                                selected.set(l, true);
                            }
                        }
                        Some(Mask::build(n, |l| selected.get(l) && !broken.get(l)))
                    } else {
                        None
                    };
                    if let Some(run) = run {
                        self.restore(&run, &cf);
                    }
                    pc += 1;
                }
                Endswitch => {
                    if let Some(Frame::Switch { entry, .. }) = cf.pop() {
                        self.restore(&entry, &cf);
                    }
                    pc += 1;
                }
                Break | Breakc => {
                    // `break` leaves the innermost loop or switch.
                    let brk = Mask::build(n, |l| self.lanes[l].active && cond_break(inst, self, l));
                    match cf.iter_mut().rev().find(|f| matches!(f, Frame::Loop { .. } | Frame::Switch { .. })) {
                        Some(Frame::Loop { broken, .. }) | Some(Frame::Switch { broken, .. }) => {
                            for l in 0..n {
                                if brk.get(l) {
                                    broken.set(l, true);
                                }
                            }
                        }
                        _ => {}
                    }
                    for l in 0..n {
                        if brk.get(l) {
                            self.lanes[l].active = false;
                        }
                    }
                    pc += 1;
                }
                Continue => {
                    let cont = Mask::build(n, |l| self.lanes[l].active);
                    if let Some(Frame::Loop { continued, .. }) = cf.iter_mut().rev().find(|f| matches!(f, Frame::Loop { .. })) {
                        for l in 0..n {
                            if cont.get(l) {
                                continued.set(l, true);
                            }
                        }
                    }
                    for l in 0..n {
                        if cont.get(l) {
                            self.lanes[l].active = false;
                        }
                    }
                    pc += 1;
                }
                Endloop => {
                    // Lanes still active here, or that hit `continue`, want another iteration unless
                    // broken or returned. Compute in a scoped borrow so we can pop/restore after.
                    let (loop_again, start, want, entry) = {
                        let frame = cf.last_mut().unwrap();
                        if let Frame::Loop { entry, start, broken, continued } = frame {
                            let want = Mask::build(n, |l| (self.lanes[l].active || continued.get(l)) && !broken.get(l) && !self.lanes[l].returned);
                            let again = want.any();
                            if again {
                                *continued = Mask::zeros(n);
                            }
                            (again, *start, want, entry.clone())
                        } else {
                            (false, 0, Mask::Bits(0), Mask::Bits(0))
                        }
                    };
                    if loop_again {
                        for l in 0..n {
                            self.lanes[l].active = want.get(l);
                        }
                        pc = start;
                    } else {
                        cf.pop();
                        self.restore(&entry, &cf);
                        pc += 1;
                    }
                }
                Sync => pc += 1, // lockstep execution is already a group barrier
                DerivRtx | DerivRty | DerivRtxCoarse | DerivRtxFine | DerivRtyCoarse | DerivRtyFine => {
                    self.deriv(inst);
                    pc += 1;
                }
                StoreRaw | StoreStructured | StoreUavTyped | LdRaw | LdStructured | LdUavTyped
                | AtomicAnd | AtomicOr | AtomicXor | AtomicIadd | AtomicImax | AtomicImin | AtomicUmax
                | AtomicUmin | AtomicCmpStore | ImmAtomicIadd | ImmAtomicAnd | ImmAtomicOr | ImmAtomicXor
                | ImmAtomicExch | ImmAtomicCmpExch | ImmAtomicImax | ImmAtomicImin | ImmAtomicUmax | ImmAtomicUmin => {
                    self.mem_op(prog, inst);
                    pc += 1;
                }
                // Declarations carried as metadata, flow markers handled above, discard (no output
                // channel in our comparison), and the no-op decls: nothing to execute.
                DclTemps | DclIndexableTemp | DclThreadGroup | DclUavTyped | DclUavRaw
                | DclUavStructured | DclTgsmRaw | DclTgsmStructured | DclResourceRaw
                | DclResourceStructured | Discard => pc += 1,
                Other(code) if is_declaration(code) => pc += 1,
                Other(code) => panic!("d2d-shader: unimplemented opcode {code}"),
                _ => {
                    for l in 0..n {
                        if self.lanes[l].active && !self.lanes[l].returned {
                            self.op_lane(l, inst);
                        }
                    }
                    pc += 1;
                }
            }
        }
    }

    /// Restore `active` to a saved scope mask, keeping already-returned lanes off and any lane that
    /// has `break`'d or `continue`'d in an enclosing loop (still on the `cf` stack) suspended.
    fn restore(&mut self, mask: &Mask, cf: &[Frame]) {
        for l in 0..self.n() {
            let suspended = cf.iter().any(|f| match f {
                Frame::Loop { broken, continued, .. } => broken.get(l) || continued.get(l),
                Frame::Switch { broken, .. } => broken.get(l),
                Frame::If { .. } => false,
            });
            self.lanes[l].active = mask.get(l) && !self.lanes[l].returned && !suspended;
        }
    }

    /// Per-lane boolean of an `if`/`breakc` instruction (operand 0, with its `_z`/`_nz` test).
    fn cond<P: Oper>(&self, l: usize, inst: &Inst<P>) -> bool {
        let v = self.src_u(l, &inst.operands[0])[0];
        (v != 0) == inst.test_nz
    }

    // --- operand reads ----------------------------------------------------

    fn base(&self, l: usize, op: &Operand) -> [u32; 4] {
        // Resolve the operand's indices to primary/secondary slots, then read the bank. The immediate
        // constant buffer (ty 9) addresses by its last index; everything else uses the first two.
        let idx_n = |n: usize| op.indices.get(n).map(|x| self.idx(l, x)).unwrap_or(0);
        let (i0, i1) = match op.ty {
            9 => (op.indices.last().map(|x| self.idx(l, x)).unwrap_or(0), 0),
            _ => (idx_n(0), idx_n(1)),
        };
        self.base_read(l, op.ty, i0, i1, &op.imm)
    }

    /// Read a register bank with its indices already resolved. Shared by the reference [`base`] and
    /// the pre-decoded operand so both read identically. Out-of-bounds reads return 0 (D3D11): an
    /// out-of-range temp / output / input / constant-buffer read is zero, and an out-of-range
    /// indexable-temp row clamps to the bank's guard slot (`len - 1`).
    #[inline(always)]
    fn base_read(&self, l: usize, ty: u8, i0: usize, i1: usize, imm: &[u32]) -> [u32; 4] {
        let lane = &self.lanes[l];
        match ty {
            0 => lane.r.get(i0).copied().unwrap_or([0; 4]),
            1 => lane.v.get(i0).copied().unwrap_or([0; 4]),
            2 => lane.o.get(i0).copied().unwrap_or([0; 4]),
            3 => lane.x.get(i0).map(|b| b[i1.min(b.len() - 1)]).unwrap_or([0; 4]),
            // immediate32 (4 or 1 comp) and immediate64 (two doubles = 4 dwords, or one = 2).
            4 | 5 => match imm.len() {
                4 => [imm[0], imm[1], imm[2], imm[3]],
                2 => [imm[0], imm[1], imm[0], imm[1]],
                _ => [imm.first().copied().unwrap_or(0); 4],
            },
            8 => {
                let base = i1 * 16;
                let b = self.cbuffers.get(i0).copied().unwrap_or(&[]);
                core::array::from_fn(|i| {
                    let o = base + i * 4;
                    if o + 4 <= b.len() {
                        u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
                    } else {
                        0
                    }
                })
            }
            9 => self.icb.get(i0).copied().unwrap_or([0; 4]),
            32 => [lane.tid[0], lane.tid[1], lane.tid[2], 0],
            33 => [lane.gid[0], lane.gid[1], lane.gid[2], 0],
            34 => [lane.gtid[0], lane.gtid[1], lane.gtid[2], 0],
            36 => [lane.flat; 4],
            _ => [0; 4],
        }
    }

    fn idx(&self, l: usize, index: &Index) -> usize {
        match index {
            Index::Imm(v) => *v as usize,
            Index::Rel(rel) => self.src_i(l, &**rel)[0] as usize,
            Index::ImmPlusRel(base, rel) => (*base as i64 + self.src_i(l, &**rel)[0] as i64) as usize,
        }
    }

    /// Raw bits after base-register resolution + swizzle, for a reference [`Operand`]. The
    /// pre-resolved operand returns the same four lanes directly; both go through [`Oper::swizzled`].
    #[inline(always)]
    fn base_swizzle(&self, l: usize, op: &Operand) -> [u32; 4] {
        let b = self.base(l, op);
        match op.sel {
            Sel::Swizzle(s) => [b[s[0] as usize], b[s[1] as usize], b[s[2] as usize], b[s[3] as usize]],
            Sel::Select1(c) => [b[c as usize]; 4],
            _ => b,
        }
    }

    /// Raw bits after swizzle, applying a neg/abs source modifier as an exact sign-bit operation.
    #[inline(always)]
    fn src_bits<P: Oper>(&self, l: usize, op: &P) -> [u32; 4] {
        let r = op.swizzled(self, l);
        let m = op.modifier();
        if m == 0 {
            return r;
        }
        core::array::from_fn(|i| {
            let mut b = r[i];
            if m & 2 != 0 {
                b &= 0x7fff_ffff;
            }
            // `neg` flips the sign bit, except WARP leaves a NaN's sign/payload untouched on this
            // data-movement path (so `movc`/`mov` select the NaN bit-for-bit, not a flipped copy).
            if m & 1 != 0 && b & 0x7fff_ffff <= 0x7f80_0000 {
                b ^= 0x8000_0000;
            }
            b
        })
    }

    #[inline(always)]
    fn src_f<P: Oper>(&self, l: usize, op: &P) -> [f32; 4] {
        let r = op.swizzled(self, l);
        let m = op.modifier();
        if m == 0 {
            // No source modifier is the common case: just the subnormal flush per component.
            return [fl(f32::from_bits(r[0])), fl(f32::from_bits(r[1])), fl(f32::from_bits(r[2])), fl(f32::from_bits(r[3]))];
        }
        core::array::from_fn(|i| {
            let mut v = fl(f32::from_bits(r[i]));
            if m & 2 != 0 {
                v = v.abs();
            }
            if m & 1 != 0 {
                v = -v;
            }
            v
        })
    }

    #[inline(always)]
    fn src_i<P: Oper>(&self, l: usize, op: &P) -> [i32; 4] {
        let r = op.swizzled(self, l);
        let m = op.modifier();
        core::array::from_fn(|i| {
            let mut v = r[i] as i32;
            if m & 1 != 0 {
                v = v.wrapping_neg();
            }
            v
        })
    }

    #[inline(always)]
    fn src_u<P: Oper>(&self, l: usize, op: &P) -> [u32; 4] {
        let r = op.swizzled(self, l);
        let m = op.modifier();
        core::array::from_fn(|i| {
            let mut v = r[i];
            if m & 1 != 0 {
                v = v.wrapping_neg();
            }
            v
        })
    }

    /// Read two doubles (`[c0|c1<<32]`, `[c2|c3<<32]`) from a double operand's swizzle.
    fn src_d<P: Oper>(&self, l: usize, op: &P) -> [f64; 2] {
        let r = op.swizzled(self, l);
        let m = op.modifier();
        let mut d = [f64::from_bits((r[0] as u64) | ((r[1] as u64) << 32)), f64::from_bits((r[2] as u64) | ((r[3] as u64) << 32))];
        if m & 2 != 0 {
            d = [d[0].abs(), d[1].abs()];
        }
        if m & 1 != 0 {
            d = [-d[0], -d[1]];
        }
        d
    }

    // --- destination writes ----------------------------------------------

    #[inline]
    fn dst_mask_op(op: &Operand) -> u8 {
        match op.sel {
            Sel::Mask(m) => m,
            Sel::Select1(c) => 1 << c,
            _ => 0xf,
        }
    }

    #[inline]
    fn store_f<P: Oper>(&mut self, l: usize, op: &P, sat_bit: bool, v: [f32; 4]) {
        let bits = if sat_bit {
            [sat(fl(v[0])).to_bits(), sat(fl(v[1])).to_bits(), sat(fl(v[2])).to_bits(), sat(fl(v[3])).to_bits()]
        } else {
            [fl(v[0]).to_bits(), fl(v[1]).to_bits(), fl(v[2]).to_bits(), fl(v[3]).to_bits()]
        };
        self.write(l, op, bits);
    }

    fn store_u<P: Oper>(&mut self, l: usize, op: &P, v: [u32; 4]) {
        self.write(l, op, v);
    }

    /// Store result doubles into a double destination: double0 goes to the lowest selected
    /// component pair (xy or zw), double1 to the next.
    fn store_d<P: Oper>(&mut self, l: usize, op: &P, v: [f64; 2]) {
        let mask = op.dst_mask();
        let pairs = [[v[0].to_bits() as u32, (v[0].to_bits() >> 32) as u32], [v[1].to_bits() as u32, (v[1].to_bits() >> 32) as u32]];
        let Some(slot) = op.slot_mut(self, l) else { return };
        let mut d = 0;
        for &(lo, hi) in &[(0usize, 1usize), (2, 3)] {
            if mask & (1 << lo) != 0 {
                slot[lo] = pairs[d][0];
                slot[hi] = pairs[d][1];
                d += 1;
            }
        }
    }

    /// Write result scalars to the destination's selected components in order (for `dtof`/`dtoi`/
    /// `dtou`, where N source doubles produce N scalars written to N masked components).
    fn write_consecutive<P: Oper>(&mut self, l: usize, op: &P, vals: [u32; 4]) {
        let mask = op.dst_mask();
        let Some(slot) = op.slot_mut(self, l) else { return };
        let mut k = 0;
        for i in 0..4 {
            if mask & (1 << i) != 0 {
                slot[i] = vals[k];
                k += 1;
            }
        }
    }

    #[inline(always)]
    fn write<P: Oper>(&mut self, l: usize, op: &P, v: [u32; 4]) {
        let mask = op.dst_mask();
        let Some(slot) = op.slot_mut(self, l) else { return };
        for i in 0..4 {
            if mask & (1 << i) != 0 {
                slot[i] = v[i];
            }
        }
    }

    /// Mutable 4-lane register a reference [`Operand`] destination names (temp/output/indexable-temp),
    /// or `None` when out of range. An out-of-range temp / output write is dropped; an out-of-range
    /// indexable-temp row clamps to the bank's guard slot (same slot reads clamp to, D3D11 / WARP).
    fn slot_mut_op(&mut self, l: usize, op: &Operand) -> Option<&mut [u32; 4]> {
        let i0 = op.indices.first().map(|x| self.idx(l, x)).unwrap_or(0);
        // Resolve the indexable-temp row (which may be register-relative) before the mutable borrow.
        let row = if op.ty == 3 { op.indices.get(1).map(|x| self.idx(l, x)).unwrap_or(0) } else { 0 };
        let lane = &mut self.lanes[l];
        match op.ty {
            0 => lane.r.get_mut(i0),
            2 => lane.o.get_mut(i0),
            // An out-of-range indexable-temp row writes the guard slot (`len - 1`), the same slot an
            // out-of-range read clamps to, so a later out-of-range read sees the written value.
            3 => lane.x.get_mut(i0).map(|b| {
                let g = b.len() - 1;
                &mut b[row.min(g)]
            }),
            _ => None,
        }
    }

    // --- per-lane ALU -----------------------------------------------------

    fn op_lane<P: Oper>(&mut self, l: usize, inst: &Inst<P>) {
        use Opcode::*;
        let o = &inst.operands;
        let d = || &o[0];
        macro_rules! f {
            ($i:expr) => {
                self.src_f(l, &o[$i])
            };
        }
        macro_rules! i32s {
            ($i:expr) => {
                self.src_i(l, &o[$i])
            };
        }
        macro_rules! u32s {
            ($i:expr) => {
                self.src_u(l, &o[$i])
            };
        }
        macro_rules! dd {
            ($i:expr) => {
                self.src_d(l, &o[$i])
            };
        }
        macro_rules! wf {
            ($v:expr) => {
                self.store_f(l, d(), inst.saturate, $v)
            };
        }
        macro_rules! wu {
            ($v:expr) => {
                self.store_u(l, d(), $v)
            };
        }
        macro_rules! wd {
            ($v:expr) => {
                self.store_d(l, d(), $v)
            };
        }
        match inst.op {
            Add => {
                let (a, b) = (f!(1), f!(2));
                wf!([a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]);
            }
            Mul => {
                let (a, b) = (f!(1), f!(2));
                wf!([a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]);
            }
            Div => {
                let (a, b) = (f!(1), f!(2));
                wf!([a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]]);
            }
            Mad => {
                let (a, b, c) = (f!(1), f!(2), f!(3));
                wf!([a[0] * b[0] + c[0], a[1] * b[1] + c[1], a[2] * b[2] + c[2], a[3] * b[3] + c[3]]);
            }
            Mov => {
                if inst.saturate {
                    wf!(f!(1));
                } else {
                    wu!(self.src_bits(l, &o[1]));
                }
            }
            Movc => {
                let s = u32s!(1);
                if inst.saturate {
                    // `movc_sat`: the selected value is a float result, flushed and clamped to [0,1].
                    let (a, b) = (f!(2), f!(3));
                    wf!([if s[0] != 0 { a[0] } else { b[0] }, if s[1] != 0 { a[1] } else { b[1] }, if s[2] != 0 { a[2] } else { b[2] }, if s[3] != 0 { a[3] } else { b[3] }]);
                } else {
                    let (a, b) = (self.src_bits(l, &o[2]), self.src_bits(l, &o[3]));
                    wu!([if s[0] != 0 { a[0] } else { b[0] }, if s[1] != 0 { a[1] } else { b[1] }, if s[2] != 0 { a[2] } else { b[2] }, if s[3] != 0 { a[3] } else { b[3] }]);
                }
            }
            Min => {
                let (a, b) = (f!(1), f!(2));
                wf!([fmin(a[0], b[0]), fmin(a[1], b[1]), fmin(a[2], b[2]), fmin(a[3], b[3])]);
            }
            Max => {
                let (a, b) = (f!(1), f!(2));
                wf!([fmax(a[0], b[0]), fmax(a[1], b[1]), fmax(a[2], b[2]), fmax(a[3], b[3])]);
            }
            Frc => {
                let a = f!(1);
                wf!([crate::warpmath::frc(a[0]), crate::warpmath::frc(a[1]), crate::warpmath::frc(a[2]), crate::warpmath::frc(a[3])]);
            }
            RoundNi => {
                let a = f!(1);
                wf!([a[0].floor(), a[1].floor(), a[2].floor(), a[3].floor()]);
            }
            RoundPi => {
                let a = f!(1);
                wf!([a[0].ceil(), a[1].ceil(), a[2].ceil(), a[3].ceil()]);
            }
            RoundZ => {
                let a = f!(1);
                wf!([a[0].trunc(), a[1].trunc(), a[2].trunc(), a[3].trunc()]);
            }
            RoundNe => {
                let a = f!(1);
                wf!([a[0].round_ties_even(), a[1].round_ties_even(), a[2].round_ties_even(), a[3].round_ties_even()]);
            }
            // Transcendentals route through crates/d2d's `warpmath`, reproduced bit-for-bit from the
            // x86 WARP's JIT emits (exact sqrtps/divps for sqrt/rsq/rcp, WARP's fdlibm-style
            // polynomials for exp2/log2).
            Sqrt => {
                let a = f!(1);
                wf!([crate::warpmath::sqrt(a[0]), crate::warpmath::sqrt(a[1]), crate::warpmath::sqrt(a[2]), crate::warpmath::sqrt(a[3])]);
            }
            Rsq => {
                let a = f!(1);
                wf!([crate::warpmath::rsqrt(a[0]), crate::warpmath::rsqrt(a[1]), crate::warpmath::rsqrt(a[2]), crate::warpmath::rsqrt(a[3])]);
            }
            Rcp => {
                let a = f!(1);
                wf!([crate::warpmath::rcp(a[0]), crate::warpmath::rcp(a[1]), crate::warpmath::rcp(a[2]), crate::warpmath::rcp(a[3])]);
            }
            Exp => {
                let a = f!(1);
                wf!([crate::warpmath::exp2(a[0]), crate::warpmath::exp2(a[1]), crate::warpmath::exp2(a[2]), crate::warpmath::exp2(a[3])]);
            }
            Log => {
                let a = f!(1);
                wf!([crate::warpmath::log2(a[0]), crate::warpmath::log2(a[1]), crate::warpmath::log2(a[2]), crate::warpmath::log2(a[3])]);
            }
            // `sincos dstSin, dstCos, src` — WARP's shared range-reduction + minimax, per lane/channel.
            Sincos => {
                let a = f!(2);
                let sc: [(f32, f32); 4] = [crate::warpmath::sincos(a[0]), crate::warpmath::sincos(a[1]), crate::warpmath::sincos(a[2]), crate::warpmath::sincos(a[3])];
                if o[0].ty() != 13 {
                    self.store_f(l, &o[0], inst.saturate, [sc[0].0, sc[1].0, sc[2].0, sc[3].0]);
                }
                if o[1].ty() != 13 {
                    self.store_f(l, &o[1], inst.saturate, [sc[0].1, sc[1].1, sc[2].1, sc[3].1]);
                }
            }
            Dp2 => {
                let (a, b) = (f!(1), f!(2));
                wf!([a[0] * b[0] + a[1] * b[1]; 4]);
            }
            Dp3 => {
                let (a, b) = (f!(1), f!(2));
                wf!([a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; 4]);
            }
            Dp4 => {
                let (a, b) = (f!(1), f!(2));
                wf!([a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; 4]);
            }
            Eq => {
                let (a, b) = (f!(1), f!(2));
                wu!([mask(a[0] == b[0]), mask(a[1] == b[1]), mask(a[2] == b[2]), mask(a[3] == b[3])]);
            }
            Ne => {
                let (a, b) = (f!(1), f!(2));
                wu!([mask(a[0] != b[0]), mask(a[1] != b[1]), mask(a[2] != b[2]), mask(a[3] != b[3])]);
            }
            Lt => {
                let (a, b) = (f!(1), f!(2));
                wu!([mask(a[0] < b[0]), mask(a[1] < b[1]), mask(a[2] < b[2]), mask(a[3] < b[3])]);
            }
            Ge => {
                let (a, b) = (f!(1), f!(2));
                wu!([mask(a[0] >= b[0]), mask(a[1] >= b[1]), mask(a[2] >= b[2]), mask(a[3] >= b[3])]);
            }
            Iadd => {
                let (a, b) = (i32s!(1), i32s!(2));
                wu!([a[0].wrapping_add(b[0]) as u32, a[1].wrapping_add(b[1]) as u32, a[2].wrapping_add(b[2]) as u32, a[3].wrapping_add(b[3]) as u32]);
            }
            // imul/umul/udiv have two destinations (hi, lo / quot, rem), so sources are o[2], o[3].
            Imul => {
                let (a, b) = (i32s!(2), i32s!(3));
                self.imul(l, inst, a, b);
            }
            Umul => {
                let (a, b) = (u32s!(2), u32s!(3));
                self.umul(l, inst, a, b);
            }
            Imad => {
                let (a, b, c) = (i32s!(1), i32s!(2), i32s!(3));
                wu!([a[0].wrapping_mul(b[0]).wrapping_add(c[0]) as u32, a[1].wrapping_mul(b[1]).wrapping_add(c[1]) as u32, a[2].wrapping_mul(b[2]).wrapping_add(c[2]) as u32, a[3].wrapping_mul(b[3]).wrapping_add(c[3]) as u32]);
            }
            Umad => {
                let (a, b, c) = (u32s!(1), u32s!(2), u32s!(3));
                wu!([a[0].wrapping_mul(b[0]).wrapping_add(c[0]), a[1].wrapping_mul(b[1]).wrapping_add(c[1]), a[2].wrapping_mul(b[2]).wrapping_add(c[2]), a[3].wrapping_mul(b[3]).wrapping_add(c[3])]);
            }
            Ineg => {
                let a = i32s!(1);
                wu!([a[0].wrapping_neg() as u32, a[1].wrapping_neg() as u32, a[2].wrapping_neg() as u32, a[3].wrapping_neg() as u32]);
            }
            Imin => {
                let (a, b) = (i32s!(1), i32s!(2));
                wu!([a[0].min(b[0]) as u32, a[1].min(b[1]) as u32, a[2].min(b[2]) as u32, a[3].min(b[3]) as u32]);
            }
            Imax => {
                let (a, b) = (i32s!(1), i32s!(2));
                wu!([a[0].max(b[0]) as u32, a[1].max(b[1]) as u32, a[2].max(b[2]) as u32, a[3].max(b[3]) as u32]);
            }
            Umin => {
                let (a, b) = (u32s!(1), u32s!(2));
                wu!([a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2]), a[3].min(b[3])]);
            }
            Umax => {
                let (a, b) = (u32s!(1), u32s!(2));
                wu!([a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2]), a[3].max(b[3])]);
            }
            And => {
                let (a, b) = (u32s!(1), u32s!(2));
                wu!([a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]]);
            }
            Or => {
                let (a, b) = (u32s!(1), u32s!(2));
                wu!([a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]]);
            }
            Xor => {
                let (a, b) = (u32s!(1), u32s!(2));
                wu!([a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]);
            }
            Not => {
                let a = u32s!(1);
                wu!([!a[0], !a[1], !a[2], !a[3]]);
            }
            Ishl => {
                let (a, b) = (u32s!(1), u32s!(2));
                wu!([a[0] << (b[0] & 31), a[1] << (b[1] & 31), a[2] << (b[2] & 31), a[3] << (b[3] & 31)]);
            }
            Ishr => {
                let (a, b) = (i32s!(1), u32s!(2));
                wu!([(a[0] >> (b[0] & 31)) as u32, (a[1] >> (b[1] & 31)) as u32, (a[2] >> (b[2] & 31)) as u32, (a[3] >> (b[3] & 31)) as u32]);
            }
            Ushr => {
                let (a, b) = (u32s!(1), u32s!(2));
                wu!([a[0] >> (b[0] & 31), a[1] >> (b[1] & 31), a[2] >> (b[2] & 31), a[3] >> (b[3] & 31)]);
            }
            Ieq => {
                let (a, b) = (i32s!(1), i32s!(2));
                wu!([mask(a[0] == b[0]), mask(a[1] == b[1]), mask(a[2] == b[2]), mask(a[3] == b[3])]);
            }
            Ine => {
                let (a, b) = (i32s!(1), i32s!(2));
                wu!([mask(a[0] != b[0]), mask(a[1] != b[1]), mask(a[2] != b[2]), mask(a[3] != b[3])]);
            }
            Ilt => {
                let (a, b) = (i32s!(1), i32s!(2));
                wu!([mask(a[0] < b[0]), mask(a[1] < b[1]), mask(a[2] < b[2]), mask(a[3] < b[3])]);
            }
            Ige => {
                let (a, b) = (i32s!(1), i32s!(2));
                wu!([mask(a[0] >= b[0]), mask(a[1] >= b[1]), mask(a[2] >= b[2]), mask(a[3] >= b[3])]);
            }
            Ult => {
                let (a, b) = (u32s!(1), u32s!(2));
                wu!([mask(a[0] < b[0]), mask(a[1] < b[1]), mask(a[2] < b[2]), mask(a[3] < b[3])]);
            }
            Uge => {
                let (a, b) = (u32s!(1), u32s!(2));
                wu!([mask(a[0] >= b[0]), mask(a[1] >= b[1]), mask(a[2] >= b[2]), mask(a[3] >= b[3])]);
            }
            Udiv => {
                let (a, b) = (u32s!(2), u32s!(3));
                self.udiv(l, inst, a, b);
            }
            Uaddc => {
                let (a, b) = (u32s!(1), u32s!(2));
                let sum: [u32; 4] = [a[0].wrapping_add(b[0]), a[1].wrapping_add(b[1]), a[2].wrapping_add(b[2]), a[3].wrapping_add(b[3])];
                let carry: [u32; 4] = [mask(sum[0] < a[0]), mask(sum[1] < a[1]), mask(sum[2] < a[2]), mask(sum[3] < a[3])];
                if o[0].ty() != 13 {
                    self.store_u(l, &o[0], sum);
                }
                self.store_u(l, &o[1], carry);
            }
            Usubb => {
                let (a, b) = (u32s!(1), u32s!(2));
                let diff: [u32; 4] = [a[0].wrapping_sub(b[0]), a[1].wrapping_sub(b[1]), a[2].wrapping_sub(b[2]), a[3].wrapping_sub(b[3])];
                let borrow: [u32; 4] = [mask(a[0] < b[0]), mask(a[1] < b[1]), mask(a[2] < b[2]), mask(a[3] < b[3])];
                if o[0].ty() != 13 {
                    self.store_u(l, &o[0], diff);
                }
                self.store_u(l, &o[1], borrow);
            }
            Countbits => {
                let a = u32s!(1);
                wu!([a[0].count_ones(), a[1].count_ones(), a[2].count_ones(), a[3].count_ones()]);
            }
            FirstbitLo => {
                let a = u32s!(1);
                wu!([if a[0] == 0 { 0xffff_ffff } else { a[0].trailing_zeros() }, if a[1] == 0 { 0xffff_ffff } else { a[1].trailing_zeros() }, if a[2] == 0 { 0xffff_ffff } else { a[2].trailing_zeros() }, if a[3] == 0 { 0xffff_ffff } else { a[3].trailing_zeros() }]);
            }
            FirstbitHi => {
                let a = u32s!(1);
                wu!([if a[0] == 0 { 0xffff_ffff } else { a[0].leading_zeros() }, if a[1] == 0 { 0xffff_ffff } else { a[1].leading_zeros() }, if a[2] == 0 { 0xffff_ffff } else { a[2].leading_zeros() }, if a[3] == 0 { 0xffff_ffff } else { a[3].leading_zeros() }]);
            }
            FirstbitShi => {
                let a = i32s!(1);
                wu!([firstbit_shi(a[0]), firstbit_shi(a[1]), firstbit_shi(a[2]), firstbit_shi(a[3])]);
            }
            Bfrev => {
                let a = u32s!(1);
                wu!([a[0].reverse_bits(), a[1].reverse_bits(), a[2].reverse_bits(), a[3].reverse_bits()]);
            }
            Swapc => {
                let (s, a, b) = (u32s!(2), self.src_bits(l, &o[3]), self.src_bits(l, &o[4]));
                let d0: [u32; 4] = [if s[0] != 0 { b[0] } else { a[0] }, if s[1] != 0 { b[1] } else { a[1] }, if s[2] != 0 { b[2] } else { a[2] }, if s[3] != 0 { b[3] } else { a[3] }];
                let d1: [u32; 4] = [if s[0] != 0 { a[0] } else { b[0] }, if s[1] != 0 { a[1] } else { b[1] }, if s[2] != 0 { a[2] } else { b[2] }, if s[3] != 0 { a[3] } else { b[3] }];
                self.store_u(l, &o[0], d0);
                self.store_u(l, &o[1], d1);
            }
            Ftoi => {
                let a = f!(1);
                wu!([ftoi(a[0]) as u32, ftoi(a[1]) as u32, ftoi(a[2]) as u32, ftoi(a[3]) as u32]);
            }
            Ftou => {
                let a = f!(1);
                wu!([ftou(a[0]), ftou(a[1]), ftou(a[2]), ftou(a[3])]);
            }
            Itof => {
                let a = i32s!(1);
                wf!([a[0] as f32, a[1] as f32, a[2] as f32, a[3] as f32]);
            }
            Utof => {
                let a = u32s!(1);
                wf!([a[0] as f32, a[1] as f32, a[2] as f32, a[3] as f32]);
            }
            F32tof16 => {
                let a = f!(1);
                wu!([f32_to_f16(a[0]) as u32, f32_to_f16(a[1]) as u32, f32_to_f16(a[2]) as u32, f32_to_f16(a[3]) as u32]);
            }
            F16tof32 => {
                let a = u32s!(1);
                wf!([f16_to_f32((a[0] & 0xffff) as u16), f16_to_f32((a[1] & 0xffff) as u16), f16_to_f32((a[2] & 0xffff) as u16), f16_to_f32((a[3] & 0xffff) as u16)]);
            }
            Ubfe => {
                let (w, off, src) = (u32s!(1), u32s!(2), u32s!(3));
                wu!([ubfe(w[0], off[0], src[0]), ubfe(w[1], off[1], src[1]), ubfe(w[2], off[2], src[2]), ubfe(w[3], off[3], src[3])]);
            }
            Ibfe => {
                let (w, off, src) = (u32s!(1), u32s!(2), i32s!(3));
                wu!([ibfe(w[0], off[0], src[0]) as u32, ibfe(w[1], off[1], src[1]) as u32, ibfe(w[2], off[2], src[2]) as u32, ibfe(w[3], off[3], src[3]) as u32]);
            }
            Bfi => {
                let (w, off, ins, base) = (u32s!(1), u32s!(2), u32s!(3), u32s!(4));
                wu!([bfi(w[0], off[0], ins[0], base[0]), bfi(w[1], off[1], ins[1], base[1]), bfi(w[2], off[2], ins[2], base[2]), bfi(w[3], off[3], ins[3], base[3])]);
            }
            // Double precision.
            Dadd => {
                let (a, b) = (dd!(1), dd!(2));
                wd!([a[0] + b[0], a[1] + b[1]]);
            }
            Dmul => {
                let (a, b) = (dd!(1), dd!(2));
                wd!([a[0] * b[0], a[1] * b[1]]);
            }
            Ddiv => {
                let (a, b) = (dd!(1), dd!(2));
                wd!([a[0] / b[0], a[1] / b[1]]);
            }
            Dfma => {
                let (a, b, c) = (dd!(1), dd!(2), dd!(3));
                wd!([a[0].mul_add(b[0], c[0]), a[1].mul_add(b[1], c[1])]);
            }
            Dmax => {
                let (a, b) = (dd!(1), dd!(2));
                wd!([dmax(a[0], b[0]), dmax(a[1], b[1])]);
            }
            Dmin => {
                let (a, b) = (dd!(1), dd!(2));
                wd!([dmin(a[0], b[0]), dmin(a[1], b[1])]);
            }
            Drcp => {
                let a = dd!(1);
                wd!([1.0 / a[0], 1.0 / a[1]]);
            }
            Dmov => {
                wd!(dd!(1));
            }
            Dmovc => {
                let s = u32s!(1);
                let (a, b) = (dd!(2), dd!(3));
                // The selector is read per double (components 0 and 2 of operand 1).
                wd!([if s[0] != 0 { a[0] } else { b[0] }, if s[2] != 0 { a[1] } else { b[1] }]);
            }
            Deq => {
                let (a, b) = (dd!(1), dd!(2));
                self.store_u(l, d(), dbool([a[0] == b[0], a[1] == b[1]]));
            }
            Dne => {
                let (a, b) = (dd!(1), dd!(2));
                self.store_u(l, d(), dbool([a[0] != b[0], a[1] != b[1]]));
            }
            Dlt => {
                let (a, b) = (dd!(1), dd!(2));
                self.store_u(l, d(), dbool([a[0] < b[0], a[1] < b[1]]));
            }
            Dge => {
                let (a, b) = (dd!(1), dd!(2));
                self.store_u(l, d(), dbool([a[0] >= b[0], a[1] >= b[1]]));
            }
            Dtof => {
                let a = dd!(1);
                // N source doubles → N floats, written to the destination's N selected components.
                self.write_consecutive(l, d(), [fl(a[0] as f32).to_bits(), fl(a[1] as f32).to_bits(), 0, 0]);
            }
            Ftod => {
                let a = f!(1);
                wd!([a[0] as f64, a[1] as f64]);
            }
            Dtoi => {
                let a = dd!(1);
                self.write_consecutive(l, d(), [a[0] as i32 as u32, a[1] as i32 as u32, 0, 0]);
            }
            Dtou => {
                let a = dd!(1);
                self.write_consecutive(l, d(), [a[0] as u32, a[1] as u32, 0, 0]);
            }
            Itod => {
                let a = i32s!(1);
                wd!([a[0] as f64, a[1] as f64]);
            }
            Utod => {
                let a = u32s!(1);
                wd!([a[0] as f64, a[1] as f64]);
            }
            // Sampling and resource queries.
            Sample => self.sample(l, inst, Sampling::Sample),
            SampleL => {
                let lod = self.src_f(l, &o[4])[0];
                self.sample(l, inst, Sampling::Lod(lod));
            }
            SampleB => {
                let bias = self.src_f(l, &o[4])[0];
                self.sample(l, inst, Sampling::Bias(bias));
            }
            SampleD => {
                let dx = self.src_f(l, &o[4]);
                let dy = self.src_f(l, &o[5]);
                self.sample(l, inst, Sampling::Grad { ddx: [dx[0], dx[1], dx[2]], ddy: [dy[0], dy[1], dy[2]] });
            }
            SampleC => {
                let r = self.src_f(l, &o[4])[0];
                self.sample(l, inst, Sampling::Cmp { reference: r, lz: false });
            }
            SampleCLz => {
                let r = self.src_f(l, &o[4])[0];
                self.sample(l, inst, Sampling::Cmp { reference: r, lz: true });
            }
            // `gather4 dst, coord, srv, sampler` — component from the sampler's swizzle, no offset.
            Gather4 => {
                let mode = Sampling::Gather { component: o[3].sel_comp(), offset: [0, 0] };
                self.sample_at(l, inst, 2, 3, mode);
            }
            // `gather4_po dst, coord, offset, srv, sampler` — footprint shifted by an integer texel
            // offset; srv/sampler move to operands 3/4.
            Gather4Po => {
                let off = i32s!(2);
                let mode = Sampling::Gather { component: o[4].sel_comp(), offset: [off[0], off[1]] };
                self.sample_at(l, inst, 3, 4, mode);
            }
            // `gather4_c dst, coord, srv, sampler, ref` — each footprint texel compared against ref.
            Gather4C => {
                let mode = Sampling::GatherCmp { reference: f!(4)[0] };
                self.sample_at(l, inst, 2, 3, mode);
            }
            Ld => {
                let c = i32s!(1);
                let res = o[2].reg();
                let v = self.env.load(res, c);
                self.write_resource(l, inst, &o[2], v);
            }
            Resinfo => {
                let res = o[2].reg();
                let dims = self.env.resinfo(res);
                // Return-type: aux bits 0..2 (0 float, 1 rcpFloat, 2 uint).
                let ret = inst.aux & 3;
                let v = if ret == 2 {
                    dims
                } else {
                    let f: [f32; 4] = [dims[0] as f32, dims[1] as f32, dims[2] as f32, dims[3] as f32];
                    [if ret == 1 && f[0] != 0.0 { (1.0 / f[0]).to_bits() } else { f[0].to_bits() }, if ret == 1 && f[1] != 0.0 { (1.0 / f[1]).to_bits() } else { f[1].to_bits() }, if ret == 1 && f[2] != 0.0 { (1.0 / f[2]).to_bits() } else { f[2].to_bits() }, if ret == 1 && f[3] != 0.0 { (1.0 / f[3]).to_bits() } else { f[3].to_bits() }]
                };
                self.write_resource(l, inst, &o[2], v);
            }
            SampleInfo => {
                let res = o[1].reg();
                let n = self.env.sample_count(res);
                // Default return type is float; uint when aux bit set.
                let v = if inst.aux & 1 != 0 { [n, 0, 0, 0] } else { [(n as f32).to_bits(), 0, 0, 0] };
                self.write_resource(l, inst, &o[1], v);
            }
            Bufinfo => {
                let res = o[1].reg();
                let dims = self.env.resinfo(res);
                self.write_resource(l, inst, &o[1], [dims[0], 0, 0, 0]);
            }
            _ => panic!("d2d-shader: op_lane reached non-ALU opcode {:?}", inst.op),
        }
    }

    fn imul<P: Oper>(&mut self, l: usize, inst: &Inst<P>, a: [i32; 4], b: [i32; 4]) {
        let lo: [u32; 4] = [a[0].wrapping_mul(b[0]) as u32, a[1].wrapping_mul(b[1]) as u32, a[2].wrapping_mul(b[2]) as u32, a[3].wrapping_mul(b[3]) as u32];
        let hi: [u32; 4] = [((a[0] as i64 * b[0] as i64) >> 32) as u32, ((a[1] as i64 * b[1] as i64) >> 32) as u32, ((a[2] as i64 * b[2] as i64) >> 32) as u32, ((a[3] as i64 * b[3] as i64) >> 32) as u32];
        if inst.operands[0].ty() != 13 {
            self.store_u(l, &inst.operands[0], hi);
        }
        self.store_u(l, &inst.operands[1], lo);
    }

    fn umul<P: Oper>(&mut self, l: usize, inst: &Inst<P>, a: [u32; 4], b: [u32; 4]) {
        let lo: [u32; 4] = [a[0].wrapping_mul(b[0]), a[1].wrapping_mul(b[1]), a[2].wrapping_mul(b[2]), a[3].wrapping_mul(b[3])];
        let hi: [u32; 4] = [((a[0] as u64 * b[0] as u64) >> 32) as u32, ((a[1] as u64 * b[1] as u64) >> 32) as u32, ((a[2] as u64 * b[2] as u64) >> 32) as u32, ((a[3] as u64 * b[3] as u64) >> 32) as u32];
        if inst.operands[0].ty() != 13 {
            self.store_u(l, &inst.operands[0], hi);
        }
        self.store_u(l, &inst.operands[1], lo);
    }

    fn udiv<P: Oper>(&mut self, l: usize, inst: &Inst<P>, a: [u32; 4], b: [u32; 4]) {
        let quot: [u32; 4] = [if b[0] == 0 { 0xffff_ffff } else { a[0] / b[0] }, if b[1] == 0 { 0xffff_ffff } else { a[1] / b[1] }, if b[2] == 0 { 0xffff_ffff } else { a[2] / b[2] }, if b[3] == 0 { 0xffff_ffff } else { a[3] / b[3] }];
        let rem: [u32; 4] = [if b[0] == 0 { 0xffff_ffff } else { a[0] % b[0] }, if b[1] == 0 { 0xffff_ffff } else { a[1] % b[1] }, if b[2] == 0 { 0xffff_ffff } else { a[2] % b[2] }, if b[3] == 0 { 0xffff_ffff } else { a[3] % b[3] }];
        if inst.operands[0].ty() != 13 {
            self.store_u(l, &inst.operands[0], quot);
        }
        if inst.operands[1].ty() != 13 {
            self.store_u(l, &inst.operands[1], rem);
        }
    }

    fn sample<P: Oper>(&mut self, l: usize, inst: &Inst<P>, mode: Sampling) {
        // `sample*` layout: dst, coord, srv@o[2], sampler@o[3].
        self.sample_at(l, inst, 2, 3, mode);
    }

    /// Sample/gather with the srv and sampler at the given operand indices (they shift for the
    /// `gather4_po*` forms, which carry an offset operand before the srv). `coord` is always o[1].
    fn sample_at<P: Oper>(&mut self, l: usize, inst: &Inst<P>, srv: usize, sampler: usize, mode: Sampling) {
        let o = &inst.operands;
        let coord = self.src_f(l, &o[1]);
        let res = o[srv].reg();
        let samp = o[sampler].reg();
        let v = self.env.sample(res, samp, coord, mode);
        self.write_resource(l, inst, &o[srv], v.map(f32::to_bits));
    }

    /// Write a resource result through the resource operand's swizzle into the destination mask.
    fn write_resource<P: Oper>(&mut self, l: usize, inst: &Inst<P>, res_op: &P, raw: [u32; 4]) {
        let s = res_op.swizzle();
        let swz = [raw[s[0] as usize], raw[s[1] as usize], raw[s[2] as usize], raw[s[3] as usize]];
        if inst.saturate {
            self.store_f(l, &inst.operands[0], true, swz.map(f32::from_bits));
        } else {
            self.store_u(l, &inst.operands[0], swz);
        }
    }

    // --- derivatives (cross-lane over the 2×2 quad) ----------------------

    fn deriv<P: Oper>(&mut self, inst: &Inst<P>) {
        use Opcode::*;
        if !self.quad {
            // No quad: derivatives are zero (a lone pixel has no neighbours).
            for l in 0..self.n() {
                if self.lanes[l].active {
                    self.store_f(l, &inst.operands[0], inst.saturate, [0.0; 4]);
                }
            }
            return;
        }
        // Read every lane's source first (so a write to o0 can't perturb a neighbour's read).
        let src: Vec<[f32; 4]> = (0..self.n()).map(|l| self.src_f(l, &inst.operands[1])).collect();
        for l in 0..self.n() {
            if !self.lanes[l].active {
                continue;
            }
            let base = l & !3;
            let within = l & 3; // TL=0, TR=1, BL=2, BR=3
            let (p, q) = match inst.op {
                // Coarse / base: top row for x, left column for y, broadcast to the quad.
                DerivRtx | DerivRtxCoarse => (base, base + 1),
                DerivRty | DerivRtyCoarse => (base, base + 2),
                // Fine: per-row (x) / per-column (y).
                DerivRtxFine => (base + (within & 2), base + (within & 2) + 1),
                DerivRtyFine => (base + (within & 1), base + (within & 1) + 2),
                _ => unreachable!(),
            };
            // The two pixels of a *fine* derivative share one difference; WARP computes it as
            // `partner - self`, so the high (right/bottom) lane evaluates `-(src[p] - src[q])`.
            // Numerically identical to `src[q] - src[p]` for every finite value, it differs only in
            // the sign of an exact zero: the low lane gets `+0.0`, the high lane `-0.0`. (Coarse/base
            // broadcast a single difference, so every lane keeps the same sign.)
            let hi = match inst.op {
                DerivRtxFine => within & 1 != 0,
                DerivRtyFine => within & 2 != 0,
                _ => false,
            };
            let v: [f32; 4] = [if hi { fl(-(src[p][0] - src[q][0])) } else { fl(src[q][0] - src[p][0]) }, if hi { fl(-(src[p][1] - src[q][1])) } else { fl(src[q][1] - src[p][1]) }, if hi { fl(-(src[p][2] - src[q][2])) } else { fl(src[q][2] - src[p][2]) }, if hi { fl(-(src[p][3] - src[q][3])) } else { fl(src[q][3] - src[p][3]) }];
            self.store_f(l, &inst.operands[0], inst.saturate, v);
        }
    }

    // --- compute memory and atomics --------------------------------------

    /// Byte offset of a (TGSM/UAV/resource) access and a `&mut Vec<u8>`/`&Vec<u8>` to its buffer.
    fn mem_op<P: Oper>(&mut self, prog: &Program, inst: &Inst<P>) {
        use Opcode::*;
        let n = self.n();
        match inst.op {
            StoreRaw | StoreStructured => {
                // dst (g#/u# with mask), [index], byteOffset, value.
                let structured = inst.op == StoreStructured;
                for l in 0..n {
                    if !self.lanes[l].active || self.lanes[l].returned {
                        continue;
                    }
                    let addr = self.store_addr(prog, inst, l, structured);
                    let mask = inst.operands[0].dst_mask();
                    let val = self.src_u(l, &inst.operands[inst.operands.len() - 1]);
                    self.mem_write(prog, inst.operands[0].ty(), inst.operands[0].reg(), addr, mask, val);
                }
            }
            StoreUavTyped => {
                for l in 0..n {
                    if !self.lanes[l].active || self.lanes[l].returned {
                        continue;
                    }
                    let coord = self.src_i(l, &inst.operands[1]);
                    let val = self.src_f(l, &inst.operands[2]);
                    self.uav_store_typed(inst.operands[0].reg(), coord, val);
                }
            }
            LdRaw | LdStructured => {
                let structured = inst.op == LdStructured;
                for l in 0..n {
                    if !self.lanes[l].active || self.lanes[l].returned {
                        continue;
                    }
                    let src_op = &inst.operands[inst.operands.len() - 1];
                    let addr = self.load_addr(prog, inst, l, structured);
                    let raw = self.mem_read(prog, src_op.ty(), src_op.reg(), addr);
                    self.write_resource(l, inst, src_op, raw);
                }
            }
            LdUavTyped => {
                for l in 0..n {
                    if !self.lanes[l].active || self.lanes[l].returned {
                        continue;
                    }
                    let coord = self.src_i(l, &inst.operands[1]);
                    let raw = self.uav_load_typed(inst.operands[2].reg(), coord);
                    self.write_resource(l, inst, &inst.operands[2], raw);
                }
            }
            _ => self.atomic(prog, inst),
        }
    }

    /// `[index,] byteOffset` → absolute byte offset for a store (operand layout: dst, [index], off, val).
    fn store_addr<P: Oper>(&self, prog: &Program, inst: &Inst<P>, l: usize, structured: bool) -> usize {
        if structured {
            let index = self.src_u(l, &inst.operands[1])[0] as usize;
            let off = self.src_u(l, &inst.operands[2])[0] as usize;
            let stride = self.stride(prog, inst.operands[0].ty(), inst.operands[0].reg());
            index * stride + off
        } else {
            self.src_u(l, &inst.operands[1])[0] as usize
        }
    }

    /// `[index,] byteOffset` → absolute byte offset for a load (operand layout: dst, [index], off, src).
    fn load_addr<P: Oper>(&self, prog: &Program, inst: &Inst<P>, l: usize, structured: bool) -> usize {
        let src_op = &inst.operands[inst.operands.len() - 1];
        if structured {
            let index = self.src_u(l, &inst.operands[1])[0] as usize;
            let off = self.src_u(l, &inst.operands[2])[0] as usize;
            index * self.stride(prog, src_op.ty(), src_op.reg()) + off
        } else {
            self.src_u(l, &inst.operands[1])[0] as usize
        }
    }

    fn stride(&self, prog: &Program, ty: u8, reg: u32) -> usize {
        let decls = if ty == 30 { &prog.uavs } else if ty == 31 { &prog.tgsm } else { &prog.resources };
        for d in decls {
            if d.reg == reg {
                if let Mem::Structured { stride, .. } = d.mem {
                    return stride as usize;
                }
            }
        }
        4
    }

    /// Mutable backing bytes for a UAV (ty 30) or TGSM (ty 31) register, or `None` when the bank is
    /// not bound — an unbound store is dropped rather than trapping.
    fn buf_mut(&mut self, ty: u8, reg: u32) -> Option<&mut Vec<u8>> {
        if ty == 31 {
            self.tgsm.iter_mut().find(|(r, _)| *r == reg).map(|(_, b)| b)
        } else {
            self.uav.iter_mut().find(|u| u.reg == reg).map(|u| &mut u.data)
        }
    }

    /// Read bytes from a UAV/TGSM (ty 30/31) or a read-only structured/raw resource (ty 7). An
    /// unbound bank reads as 0 (D3D11 unbound-resource rule) rather than trapping — a plugin shader
    /// that references a resource the host did not bind cannot crash the renderer.
    fn mem_read(&self, prog: &Program, ty: u8, reg: u32, addr: usize) -> [u32; 4] {
        let _ = prog;
        let buf: &[u8] = if ty == 31 {
            self.tgsm.iter().find(|(r, _)| *r == reg).map_or(&[], |(_, b)| b)
        } else if ty == 7 {
            self.res.iter().find(|r| r.reg == reg).map_or(&[], |r| &r.data)
        } else {
            self.uav.iter().find(|u| u.reg == reg).map_or(&[], |u| &u.data)
        };
        core::array::from_fn(|i| {
            let o = addr + i * 4;
            if o + 4 <= buf.len() {
                u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]])
            } else {
                0
            }
        })
    }

    fn mem_write(&mut self, _prog: &Program, ty: u8, reg: u32, addr: usize, mask: u8, val: [u32; 4]) {
        let Some(buf) = self.buf_mut(ty, reg) else { return };
        // A masked store writes consecutive selected components (store_raw/structured write .x, .xy…).
        let mut c = 0;
        for i in 0..4 {
            if mask & (1 << i) != 0 {
                let o = addr + c * 4;
                if o + 4 <= buf.len() {
                    buf[o..o + 4].copy_from_slice(&val[i].to_le_bytes());
                }
                c += 1;
            }
        }
    }

    fn uav_store_typed(&mut self, reg: u32, coord: [i32; 4], val: [f32; 4]) {
        let Some(u) = self.uav.iter_mut().find(|u| u.reg == reg) else { return };
        let (w, fmt) = (u.dims[0] as i32, if let Mem::Typed(f) = u.kind { f } else { [5; 4] });
        if coord[0] < 0 || coord[1] < 0 || coord[0] >= w || coord[1] >= u.dims[1] as i32 {
            return;
        }
        let px = (coord[1] as usize * w as usize + coord[0] as usize) * 4;
        if px + 4 <= u.data.len() {
            for i in 0..4 {
                u.data[px + i] = pack_component(val[i], fmt[i]);
            }
        }
    }

    fn uav_load_typed(&self, reg: u32, coord: [i32; 4]) -> [u32; 4] {
        let Some(u) = self.uav.iter().find(|u| u.reg == reg) else { return [0; 4] };
        let (w, fmt) = (u.dims[0] as i32, if let Mem::Typed(f) = u.kind { f } else { [5; 4] });
        if coord[0] < 0 || coord[1] < 0 || coord[0] >= w || coord[1] >= u.dims[1] as i32 {
            return [0; 4];
        }
        let px = (coord[1] as usize * w as usize + coord[0] as usize) * 4;
        [if px + 4 <= u.data.len() { unpack_component(u.data[px], fmt[0]) } else { 0 }, if px + 4 <= u.data.len() { unpack_component(u.data[px + 1], fmt[1]) } else { 0 }, if px + 4 <= u.data.len() { unpack_component(u.data[px + 2], fmt[2]) } else { 0 }, if px + 4 <= u.data.len() { unpack_component(u.data[px + 3], fmt[3]) } else { 0 }]
    }

    /// `atomic_*` (no return) and `imm_atomic_*` (returns the pre-op value). Lanes are applied in
    /// index order; for order-independent ops (iadd/and/or/xor/min/max) the stored value is
    /// deterministic, so authored scenes use those. ponytail: serial lane order, exact only for
    /// commutative atomics — order-dependent races (plain exch) are unspecified on real WARP too.
    fn atomic<P: Oper>(&mut self, prog: &Program, inst: &Inst<P>) {
        use Opcode::*;
        let imm = matches!(
            inst.op,
            ImmAtomicIadd | ImmAtomicAnd | ImmAtomicOr | ImmAtomicXor | ImmAtomicExch | ImmAtomicCmpExch | ImmAtomicImax | ImmAtomicImin | ImmAtomicUmax | ImmAtomicUmin
        );
        // Operand layout: [imm: dst_old,] resource, address, src0 [, src1 for cmp].
        let base = if imm { 1 } else { 0 };
        let res_op = inst.operands[base].clone();
        let (ty, reg) = (res_op.ty(), res_op.reg());
        let cmp = matches!(inst.op, AtomicCmpStore | ImmAtomicCmpExch);
        for l in 0..self.n() {
            if !self.lanes[l].active || self.lanes[l].returned {
                continue;
            }
            let addr = self.atomic_addr(prog, inst, l, base, ty, reg);
            let s0 = self.src_u(l, &inst.operands[base + 2])[0];
            let s1 = if cmp { self.src_u(l, &inst.operands[base + 3])[0] } else { 0 };
            let old = self.mem_read(prog, ty, reg, addr)[0];
            let new = atomic_apply(inst.op, old, s0, s1);
            self.mem_write(prog, ty, reg, addr, 1, [new, 0, 0, 0]);
            if imm {
                let dst = inst.operands[0].clone();
                self.store_u(l, &dst, [old; 4]);
            }
        }
    }

    /// Atomic address: structured UAVs use (index, byteOffset); typed/raw use a single address
    /// component. The address operand is at `base+1`.
    fn atomic_addr<P: Oper>(&self, prog: &Program, inst: &Inst<P>, l: usize, base: usize, ty: u8, reg: u32) -> usize {
        let a = self.src_u(l, &inst.operands[base + 1]);
        let is_struct = {
            let decls = if ty == 30 { &prog.uavs } else { &prog.tgsm };
            decls.iter().find(|d| d.reg == reg).map(|d| matches!(d.mem, Mem::Structured { .. })).unwrap_or(false)
        };
        if is_struct {
            a[0] as usize * self.stride(prog, ty, reg) + a[1] as usize
        } else {
            // Raw buffer: byte address in .x. (Typed-UAV atomics address by texel, unused here.)
            a[0] as usize
        }
    }
}


impl Operand {
    /// The register number (first/only index), or 0.
    fn reg(&self) -> u32 {
        match self.indices.first() {
            Some(Index::Imm(v)) => *v as u32,
            _ => 0,
        }
    }
}

/// Apply one atomic op to the memory value, returning the new value.
fn atomic_apply(op: Opcode, old: u32, s0: u32, s1: u32) -> u32 {
    use Opcode::*;
    match op {
        AtomicIadd | ImmAtomicIadd => old.wrapping_add(s0),
        AtomicAnd | ImmAtomicAnd => old & s0,
        AtomicOr | ImmAtomicOr => old | s0,
        AtomicXor | ImmAtomicXor => old ^ s0,
        AtomicUmax | ImmAtomicUmax => old.max(s0),
        AtomicUmin | ImmAtomicUmin => old.min(s0),
        AtomicImax | ImmAtomicImax => (old as i32).max(s0 as i32) as u32,
        AtomicImin | ImmAtomicImin => (old as i32).min(s0 as i32) as u32,
        ImmAtomicExch => s0,
        // cmp_store/cmp_exch: if old == compare(s0) write s1, else leave.
        AtomicCmpStore | ImmAtomicCmpExch => {
            if old == s0 {
                s1
            } else {
                old
            }
        }
        _ => old,
    }
}

/// WARP `min`/`max`: if exactly one operand is NaN, return the other (D3D rule).
fn fmin(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else if a < b {
        a
    } else {
        b
    }
}
fn fmax(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else if a > b {
        a
    } else {
        b
    }
}
fn dmin(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else if a < b {
        a
    } else {
        b
    }
}
fn dmax(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else if a > b {
        a
    } else {
        b
    }
}

fn mask(c: bool) -> u32 {
    if c {
        0xffff_ffff
    } else {
        0
    }
}

/// Double comparison → the single-dword `0xffffffff`/`0` mask doubled across the double's lanes.
fn dbool(c: [bool; 2]) -> [u32; 4] {
    [mask(c[0]), mask(c[0]), mask(c[1]), mask(c[1])]
}

/// `firstbit_shi`: first bit that differs from the sign bit, counted from the MSB; 0xffffffff for
/// 0 and -1.
fn firstbit_shi(src: i32) -> u32 {
    let v = if src < 0 { !(src as u32) } else { src as u32 };
    if v == 0 {
        0xffff_ffff
    } else {
        v.leading_zeros()
    }
}

/// `ftoi`: round-to-zero float→i32; NaN→0, out-of-range clamps to the i32 extremes.
fn ftoi(x: f32) -> i32 {
    if x.is_nan() {
        0
    } else if x >= 2147483648.0 {
        i32::MAX
    } else if x < -2147483648.0 {
        i32::MIN
    } else {
        x.trunc() as i32
    }
}

/// `ftou`: round-to-zero float→u32; NaN/negative→0, overflow→0xffffffff.
fn ftou(x: f32) -> u32 {
    if x.is_nan() || x < 0.0 {
        0
    } else if x >= 4294967296.0 {
        0xffff_ffff
    } else {
        x.trunc() as u32
    }
}

fn ubfe(width: u32, offset: u32, src: u32) -> u32 {
    let w = width & 31;
    let off = offset & 31;
    if w == 0 {
        0
    } else if w + off >= 32 {
        src >> off
    } else {
        (src >> off) & ((1 << w) - 1)
    }
}

fn ibfe(width: u32, offset: u32, src: i32) -> i32 {
    let w = width & 31;
    let off = offset & 31;
    if w == 0 {
        0
    } else if w + off >= 32 {
        src >> off
    } else {
        (src << (32 - w - off)) >> (32 - w)
    }
}

fn bfi(width: u32, offset: u32, ins: u32, base: u32) -> u32 {
    let w = width & 31;
    let off = offset & 31;
    let m = if w + off >= 32 { !0u32 << off } else { ((1u32 << w) - 1) << off };
    ((ins << off) & m) | (base & !m)
}

/// Pack a float into a UNORM8/SNORM8/UINT8/SINT8 byte per the typed-UAV return-type code.
fn pack_component(x: f32, fmt: u8) -> u8 {
    match fmt {
        1 => to_unorm8(x),                                                 // unorm
        2 => (x.clamp(-1.0, 1.0) * 127.0).round_ties_even() as i8 as u8,   // snorm
        3 => (x as i32).clamp(-128, 127) as i8 as u8,                      // sint
        4 => (x as u32).min(255) as u8,                                    // uint
        _ => to_unorm8(x),
    }
}

fn unpack_component(b: u8, fmt: u8) -> u32 {
    match fmt {
        1 => (b as f32 / 255.0).to_bits(),
        2 => ((b as i8 as f32 / 127.0).clamp(-1.0, 1.0)).to_bits(),
        3 => b as i8 as i32 as u32,
        4 => b as u32,
        _ => (b as f32 / 255.0).to_bits(),
    }
}

/// IEEE binary32 → binary16, round-to-nearest-even (the `f32tof16` conversion WARP performs).
fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        // Inf/NaN: keep a non-zero mantissa as NaN.
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7bff; // D3D f32tof16 clamps finite overflow to the max half, not inf
    }
    // WARP's f32tof16 truncates toward zero (drops the low mantissa bits), not round-to-nearest.
    if e <= 0 {
        if e < -10 {
            return sign; // too small → signed zero
        }
        let m = mant | 0x80_0000; // implicit 1
        let shift = (14 - e) as u32;
        return sign | (m >> shift) as u16;
    }
    sign | ((e as u16) << 10) | (mant >> 13) as u16
}

/// IEEE binary16 → binary32 (the `f16tof32` conversion).
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal half: value = mant * 2^-24 = 1.frac * 2^(p-24), p = top set bit of mant.
        let p = 31 - mant.leading_zeros();
        let fexp = (p as i32 - 24 + 127) as u32;
        let frac = (mant << (23 - p)) & 0x7f_ffff;
        return f32::from_bits(sign | (fexp << 23) | frac);
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }
    let fexp = exp + (127 - 15);
    f32::from_bits(sign | (fexp << 23) | (mant << 13))
}

/// `break`/`breakc` test for one lane (plain `break` is unconditional).
fn cond_break<P: Oper>(inst: &Inst<P>, m: &Machine, l: usize) -> bool {
    if inst.op == Opcode::Break {
        true
    } else {
        m.cond(l, inst)
    }
}

fn is_declaration(code: u16) -> bool {
    matches!(code, 44 | 58 | 88..=107 | 143..=154)
}

#[cfg(test)]
mod tests {
    use super::*;

    // WARP's saturate is `max(0, min(1, x))`: the `max` against `+0` normalises a `-0.0` result to
    // `+0.0` (a plain clamp would keep the `-0.0`). Byte-exact against the Paint.NET corpus relies on
    // this (e.g. shader 35a1…, where a saturated `-0.0` from `0*−k` must read back as `+0.0`).
    #[test]
    fn saturate_normalizes_negative_zero() {
        assert_eq!(sat(-0.0).to_bits(), 0x0000_0000);
        assert_eq!(sat(0.0).to_bits(), 0x0000_0000);
        assert_eq!(sat(1.5), 1.0);
        assert_eq!(sat(-3.0).to_bits(), 0x0000_0000);
        assert_eq!(sat(f32::NAN).to_bits(), 0x0000_0000);
    }

    // firstbit_shi and the carry/borrow ops have no HLSL intrinsic that fxc lowers cleanly, so they
    // are not covered by a WARP scene; pin their D3D-specified behaviour here.
    #[test]
    fn firstbit_shi_rules() {
        // Position (from the MSB) of the first bit that differs from the sign bit; 0xffffffff when
        // all bits are the sign bit.
        assert_eq!(firstbit_shi(0), 0xffff_ffff);
        assert_eq!(firstbit_shi(-1), 0xffff_ffff);
        assert_eq!(firstbit_shi(1), 31); // 0x00000001: only bit0 differs from sign 0
        assert_eq!(firstbit_shi(-2), 31); // 0xfffffffe: ~ = 1, bit0 differs from sign 1
        assert_eq!(firstbit_shi(0x4000_0000), 1); // bit30 differs from sign 0
    }
}
