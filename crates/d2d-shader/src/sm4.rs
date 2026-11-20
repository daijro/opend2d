//! Shader-model 4/5 tokenized-program (TPF) instruction decoder.
//!
//! Decodes the `SHEX`/`SHDR` token stream into [`Inst`]s with typed [`Operand`]s. Bit layouts and
//! the opcode/operand-type/index-representation numbers are the public ones from Microsoft's
//! `d3d11TokenizedProgramFormat.hpp` (same values the `winapi` crate re-publishes); only the short
//! numeric identifiers are used here. The executor in `exec.rs` consumes this; the decl opcodes we
//! do not execute are still length-walked so the token cursor stays aligned.

/// Opcode numbers we name. Everything else decodes as [`Opcode::Other`] and is length-skipped.
/// Numbers are the `D3D10_SB_OPCODE_TYPE` values from Microsoft's `d3d11TokenizedProgramFormat.hpp`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(non_camel_case_types)]
pub enum Opcode {
    Add, And, Break, Breakc, Continue, Discard, Div, Dp2, Dp3, Dp4, Else, Endif, Endloop, Eq, Exp,
    Frc, Ftoi, Ftou, Ge, Iadd, If, Ieq, Ige, Ilt, Imad, Imax, Imin, Imul, Ine, Ineg, Ishl, Ishr,
    Itof, Ld, Log, Loop, Lt, Mad, Min, Max, Mov, Movc, Mul, Ne, Not, Or, Rcp, Ret, RoundNe, RoundNi,
    RoundPi, RoundZ, Rsq, Sample, SampleL, Sincos, Sqrt, Udiv, Uge, Ult, Umad, Umax, Umin, Umul,
    Ushr, Utof, Xor, Ubfe, Ibfe, Bfi, F32tof16, F16tof32,
    // Switch flow control: `switch src`, `case <imm>`, `default`, `endswitch`.
    Switch, Case, Default, Endswitch,
    // Derivatives (quad): base (==coarse on WARP) and the SM5 coarse/fine forms.
    DerivRtx, DerivRty, DerivRtxCoarse, DerivRtxFine, DerivRtyCoarse, DerivRtyFine,
    // Sample / gather variants and resource queries.
    SampleC, SampleCLz, SampleD, SampleB, Lod, Gather4, Gather4C, Gather4Po, Gather4PoC,
    Resinfo, SamplePos, SampleInfo, Bufinfo, LdMs,
    // Integer completion.
    Uaddc, Usubb, Countbits, FirstbitHi, FirstbitLo, FirstbitShi, Bfrev, Swapc,
    // Double-precision.
    Dadd, Dmax, Dmin, Dmul, Deq, Dge, Dlt, Dne, Dmov, Dmovc, Dtof, Ftod, Ddiv, Dfma, Drcp,
    Dtoi, Dtou, Itod, Utod,
    // Compute: thread-group / UAV / groupshared / structured-resource declarations (payload kept).
    DclTemps, DclIndexableTemp, DclThreadGroup, DclUavTyped, DclUavRaw, DclUavStructured,
    DclTgsmRaw, DclTgsmStructured, DclResourceRaw, DclResourceStructured,
    // Compute memory access and atomics.
    LdUavTyped, StoreUavTyped, LdRaw, StoreRaw, LdStructured, StoreStructured,
    AtomicAnd, AtomicOr, AtomicXor, AtomicCmpStore, AtomicIadd, AtomicImax, AtomicImin, AtomicUmax,
    AtomicUmin, ImmAtomicAlloc, ImmAtomicConsume, ImmAtomicIadd, ImmAtomicAnd, ImmAtomicOr,
    ImmAtomicXor, ImmAtomicExch, ImmAtomicCmpExch, ImmAtomicImax, ImmAtomicImin, ImmAtomicUmax,
    ImmAtomicUmin, Sync,
    /// Any opcode we recognise structurally but do not execute (declarations we ignore).
    Other(u16),
}

impl Opcode {
    fn from_raw(n: u16) -> Opcode {
        use Opcode::*;
        match n {
            0 => Add, 1 => And, 2 => Break, 3 => Breakc, 6 => Case, 7 => Continue, 10 => Default,
            11 => DerivRtx,
            12 => DerivRty, 13 => Discard, 14 => Div, 15 => Dp2, 16 => Dp3, 17 => Dp4, 18 => Else,
            21 => Endif, 22 => Endloop, 23 => Endswitch, 24 => Eq, 25 => Exp, 26 => Frc, 27 => Ftoi, 28 => Ftou,
            29 => Ge, 30 => Iadd, 31 => If, 32 => Ieq, 33 => Ige, 34 => Ilt, 35 => Imad, 36 => Imax,
            37 => Imin, 38 => Imul, 39 => Ine, 40 => Ineg, 41 => Ishl, 42 => Ishr, 43 => Itof,
            45 => Ld, 46 => LdMs, 47 => Log, 48 => Loop, 49 => Lt, 50 => Mad, 51 => Min, 52 => Max,
            54 => Mov, 55 => Movc, 56 => Mul, 57 => Ne, 59 => Not, 60 => Or, 61 => Resinfo,
            62 => Ret, 64 => RoundNe, 65 => RoundNi, 66 => RoundPi, 67 => RoundZ, 68 => Rsq,
            69 => Sample, 70 => SampleC, 71 => SampleCLz, 72 => SampleL, 73 => SampleD, 74 => SampleB,
            75 => Sqrt, 76 => Switch, 77 => Sincos, 78 => Udiv, 79 => Ult, 80 => Uge, 81 => Umul, 82 => Umad,
            83 => Umax, 84 => Umin, 85 => Ushr, 86 => Utof, 87 => Xor, 104 => DclTemps,
            105 => DclIndexableTemp, 108 => Lod, 109 => Gather4, 110 => SamplePos, 111 => SampleInfo,
            121 => Bufinfo, 122 => DerivRtxCoarse, 123 => DerivRtxFine, 124 => DerivRtyCoarse,
            125 => DerivRtyFine, 126 => Gather4C, 127 => Gather4Po, 128 => Gather4PoC, 129 => Rcp,
            130 => F32tof16, 131 => F16tof32, 132 => Uaddc, 133 => Usubb, 134 => Countbits,
            135 => FirstbitHi, 136 => FirstbitLo, 137 => FirstbitShi, 138 => Ubfe, 139 => Ibfe,
            140 => Bfi, 141 => Bfrev, 142 => Swapc, 155 => DclThreadGroup, 156 => DclUavTyped,
            157 => DclUavRaw, 158 => DclUavStructured, 159 => DclTgsmRaw, 160 => DclTgsmStructured,
            161 => DclResourceRaw, 162 => DclResourceStructured, 163 => LdUavTyped,
            164 => StoreUavTyped, 165 => LdRaw, 166 => StoreRaw, 167 => LdStructured,
            168 => StoreStructured, 169 => AtomicAnd, 170 => AtomicOr, 171 => AtomicXor,
            172 => AtomicCmpStore, 173 => AtomicIadd, 174 => AtomicImax, 175 => AtomicImin,
            176 => AtomicUmax, 177 => AtomicUmin, 178 => ImmAtomicAlloc, 179 => ImmAtomicConsume,
            180 => ImmAtomicIadd, 181 => ImmAtomicAnd, 182 => ImmAtomicOr, 183 => ImmAtomicXor,
            184 => ImmAtomicExch, 185 => ImmAtomicCmpExch, 186 => ImmAtomicImax, 187 => ImmAtomicImin,
            188 => ImmAtomicUmax, 189 => ImmAtomicUmin, 190 => Sync, 191 => Dadd, 192 => Dmax,
            193 => Dmin, 194 => Dmul, 195 => Deq, 196 => Dge, 197 => Dlt, 198 => Dne, 199 => Dmov,
            // D3D11.1 doubles start at 210 (209 is reserved): ddiv, dfma, drcp, msad(213, unused),
            // dtoi, dtou, itod, utod.
            200 => Dmovc, 201 => Dtof, 202 => Ftod, 210 => Ddiv, 211 => Dfma, 212 => Drcp,
            214 => Dtoi, 215 => Dtou, 216 => Itod, 217 => Utod,
            other => Other(other),
        }
    }
}

const OPCODE_CUSTOMDATA: u16 = 53;

/// Component selection for a 4-component operand.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sel {
    /// Destination write mask (bit per component, x=1 y=2 z=4 w=8).
    Mask(u8),
    /// Source swizzle: four component indices (0..3 = x,y,z,w).
    Swizzle([u8; 4]),
    /// Single selected component (0..3).
    Select1(u8),
    /// 0- or 1-component operand (no selection field).
    None,
}

/// One index of an operand (register number, cbuffer element, …). May itself be register-relative.
#[derive(Clone, PartialEq, Debug)]
pub enum Index {
    Imm(u64),
    Rel(Box<Operand>),
    ImmPlusRel(u64, Box<Operand>),
}

/// A decoded operand. `ty` is the operand-type number (0 temp, 1 input, 2 output, 3 indexable-temp,
/// 4 immediate32, 6 sampler, 7 resource, 8 constant-buffer, 13 null, …).
#[derive(Clone, PartialEq, Debug)]
pub struct Operand {
    pub ty: u8,
    pub num_components: u8,
    pub sel: Sel,
    /// 0 none, 1 neg, 2 abs, 3 abs+neg (source modifier).
    pub modifier: u8,
    pub indices: Vec<Index>,
    /// Immediate payload when `ty` is immediate32 (1 or 4 values).
    pub imm: Vec<u32>,
}

/// A decoded instruction. Generic over the operand representation so the executor can run either the
/// reference [`Operand`]s or a pre-resolved operand type (the pre-decode pass) through one code path;
/// everything that decodes a shader uses the default `Inst = Inst<Operand>`.
#[derive(Clone, PartialEq, Debug)]
pub struct Inst<P = Operand> {
    pub op: Opcode,
    pub saturate: bool,
    /// `if`/`breakc`/`continuec` boolean test: true = branch when the operand is non-zero.
    pub test_nz: bool,
    pub operands: Vec<P>,
    /// `dcl_temps` count / `dcl_indexableTemp` register count (else 0).
    pub decl_count: u32,
    /// Instruction-specific control field (token bits 11..24): e.g. `resinfo` return-type.
    pub aux: u16,
}

/// How a UAV / resource / groupshared bank is addressed, from its declaration.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mem {
    /// Typed (e.g. `unorm×4`) texture/buffer UAV or resource: `fmt` is the four return-type codes
    /// (1 unorm, 2 snorm, 3 sint, 4 uint, 5 float). Addressed by texel coords.
    Typed([u8; 4]),
    /// Byte-addressed raw buffer (`ByteAddressBuffer`). `bytes` is the groupshared size (0 otherwise).
    Raw { bytes: u32 },
    /// Structured buffer: `stride` bytes per element; `bytes` is the groupshared size (0 otherwise).
    Structured { stride: u32, bytes: u32 },
}

/// A declared `u#`/`t#`/`g#` bank: register number plus its addressing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Decl {
    pub reg: u32,
    pub mem: Mem,
}

/// A decoded program: version word plus the instruction stream.
#[derive(Clone, PartialEq, Debug)]
pub struct Program {
    /// Program-type (ps=0, vs=1, cs=5, …) from the version token.
    pub program_type: u16,
    pub major: u8,
    pub minor: u8,
    pub insts: Vec<Inst>,
    /// Number of `r#` temp registers (`dcl_temps`).
    pub temps: u32,
    /// `(id, register_count)` for each `x#` indexable-temp bank (`dcl_indexableTemp`).
    pub indexable_temps: Vec<(u32, u32)>,
    /// `dcl_thread_group` dimensions (compute); `[0;3]` for non-compute.
    pub thread_group: [u32; 3],
    /// `u#` UAV declarations (typed/raw/structured).
    pub uavs: Vec<Decl>,
    /// `g#` thread-group-shared-memory declarations (raw/structured, with byte size).
    pub tgsm: Vec<Decl>,
    /// `t#` raw/structured resource declarations (typed resources are handled by the sampler env).
    pub resources: Vec<Decl>,
    /// `dcl_immediateConstantBuffer` data: the baked `float4` array operand type 9 indexes into.
    pub icb: Vec<[u32; 4]>,
}

/// Walk one operand starting at `t[0]`, returning it and the number of dwords consumed.
fn read_operand(t: &[u32]) -> (Operand, usize) {
    let t0 = t[0];
    let num_components = match t0 & 3 {
        0 => 0,
        1 => 1,
        _ => 4,
    };
    let sel = if num_components == 4 {
        match (t0 >> 2) & 3 {
            0 => Sel::Mask(((t0 >> 4) & 0xf) as u8),
            1 => {
                let s = (t0 >> 4) & 0xff;
                Sel::Swizzle([(s & 3) as u8, ((s >> 2) & 3) as u8, ((s >> 4) & 3) as u8, ((s >> 6) & 3) as u8])
            }
            _ => Sel::Select1(((t0 >> 4) & 3) as u8),
        }
    } else {
        Sel::None
    };
    let ty = ((t0 >> 12) & 0xff) as u8;
    let index_dim = ((t0 >> 20) & 3) as usize;
    let mut modifier = 0u8;
    let mut cur = 1usize;
    // Extended operand tokens (modifier). Chain continues while bit31 of the current ext token set.
    if t0 >> 31 != 0 {
        loop {
            let e = t[cur];
            cur += 1;
            if e & 0x3f == 1 {
                modifier = ((e >> 6) & 0xff) as u8;
            }
            if e >> 31 == 0 {
                break;
            }
        }
    }
    let mut indices = Vec::with_capacity(index_dim);
    for d in 0..index_dim {
        let rep = (t0 >> (22 + 3 * d)) & 3;
        match rep {
            0 => {
                indices.push(Index::Imm(t[cur] as u64));
                cur += 1;
            }
            1 => {
                indices.push(Index::Imm((t[cur] as u64) | ((t[cur + 1] as u64) << 32)));
                cur += 2;
            }
            2 => {
                let (rel, n) = read_operand(&t[cur..]);
                indices.push(Index::Rel(Box::new(rel)));
                cur += n;
            }
            _ => {
                // 3 = imm32+relative (imm64+relative, rep 4, does not occur in ps corpus).
                let base = t[cur] as u64;
                cur += 1;
                let (rel, n) = read_operand(&t[cur..]);
                indices.push(Index::ImmPlusRel(base, Box::new(rel)));
                cur += n;
            }
        }
    }
    let mut imm = Vec::new();
    if ty == 4 {
        let n = if num_components == 4 { 4 } else { 1 };
        imm.extend_from_slice(&t[cur..cur + n]);
        cur += n;
    } else if ty == 5 {
        // immediate64: a 4-component operand is two doubles (4 dwords); a 1-component one double.
        let n = if num_components == 4 { 4 } else { 2 };
        imm.extend_from_slice(&t[cur..cur + n]);
        cur += n;
    }
    (Operand { ty, num_components, sel, modifier, indices, imm }, cur)
}

/// Decode a `SHEX`/`SHDR` program-token slice (starting at the version token).
pub fn decode(tokens: &[u32]) -> Program {
    let ver = tokens[0];
    let len = tokens[1] as usize;
    let mut prog = Program {
        program_type: ((ver >> 16) & 0xffff) as u16,
        major: ((ver >> 4) & 0xf) as u8,
        minor: (ver & 0xf) as u8,
        insts: Vec::new(),
        temps: 0,
        indexable_temps: Vec::new(),
        thread_group: [0; 3],
        uavs: Vec::new(),
        tgsm: Vec::new(),
        resources: Vec::new(),
        icb: Vec::new(),
    };
    let mut i = 2usize;
    while i < len {
        let tok = tokens[i];
        let raw = (tok & 0x7ff) as u16;
        if raw == OPCODE_CUSTOMDATA {
            let dlen = tokens[i + 1] as usize;
            // Custom-data class is bits [31:11]; class 3 = dcl_immediateConstantBuffer (float4 array).
            if tok >> 11 == 3 {
                prog.icb = tokens[i + 2..i + dlen.max(2)].chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
            }
            i += dlen.max(2);
            continue;
        }
        let ilen = ((tok >> 24) & 0x7f) as usize;
        let op = Opcode::from_raw(raw);
        let saturate = (tok >> 13) & 1 != 0;
        let test_nz = (tok >> 18) & 1 != 0;
        let aux = ((tok >> 11) & 0x1fff) as u16;
        // Skip extended opcode tokens (sample controls / resource dims) before the operands.
        let mut start = i + 1;
        if tok >> 31 != 0 {
            while tokens[start] >> 31 != 0 {
                start += 1;
            }
            start += 1;
        }
        let end = i + ilen;
        let mut operands = Vec::new();
        let mut decl_count = 0u32;
        // Register number of the single operand of a declaration, plus the dword just after it.
        let decl_reg = |t: &[u32]| {
            let (o, n) = read_operand(&t[start..end]);
            let reg = match o.indices.first() {
                Some(Index::Imm(v)) => *v as u32,
                _ => 0,
            };
            (reg, start + n)
        };
        match op {
            Opcode::DclTemps => decl_count = tokens[start],
            Opcode::DclIndexableTemp => {
                prog.indexable_temps.push((tokens[start], tokens[start + 1]));
            }
            Opcode::DclThreadGroup => {
                prog.thread_group = [tokens[start], tokens[start + 1], tokens[start + 2]];
            }
            Opcode::DclUavTyped => {
                let (reg, c) = decl_reg(tokens);
                let d = tokens[c];
                let fmt = [(d & 0xf) as u8, ((d >> 4) & 0xf) as u8, ((d >> 8) & 0xf) as u8, ((d >> 12) & 0xf) as u8];
                prog.uavs.push(Decl { reg, mem: Mem::Typed(fmt) });
            }
            Opcode::DclUavRaw => {
                let (reg, _) = decl_reg(tokens);
                prog.uavs.push(Decl { reg, mem: Mem::Raw { bytes: 0 } });
            }
            Opcode::DclUavStructured => {
                let (reg, c) = decl_reg(tokens);
                prog.uavs.push(Decl { reg, mem: Mem::Structured { stride: tokens[c], bytes: 0 } });
            }
            Opcode::DclTgsmRaw => {
                let (reg, c) = decl_reg(tokens);
                prog.tgsm.push(Decl { reg, mem: Mem::Raw { bytes: tokens[c] } });
            }
            Opcode::DclTgsmStructured => {
                let (reg, c) = decl_reg(tokens);
                let (stride, count) = (tokens[c], tokens[c + 1]);
                prog.tgsm.push(Decl { reg, mem: Mem::Structured { stride, bytes: stride * count } });
            }
            Opcode::DclResourceRaw => {
                let (reg, _) = decl_reg(tokens);
                prog.resources.push(Decl { reg, mem: Mem::Raw { bytes: 0 } });
            }
            Opcode::DclResourceStructured => {
                let (reg, c) = decl_reg(tokens);
                prog.resources.push(Decl { reg, mem: Mem::Structured { stride: tokens[c], bytes: 0 } });
            }
            Opcode::Other(_) => {}
            _ => {
                let mut c = start;
                while c < end {
                    let (o, n) = read_operand(&tokens[c..end]);
                    c += n;
                    operands.push(o);
                }
            }
        }
        if op == Opcode::DclTemps {
            prog.temps = decl_count;
        }
        prog.insts.push(Inst { op, saturate, test_nz, operands, decl_count, aux });
        i += ilen;
    }
    prog
}
