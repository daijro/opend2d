//! The TrueType bytecode interpreter: font/CVT/glyph programs, the full instruction
//! set, graphics state, twilight zone and function/instruction definitions, reproducing
//! DirectWrite's classic rasterizer. Instruction decoding reuses read-fonts'
//! bytecode `Decoder`; everything else is implemented here.

use skrifa::raw::tables::glyf::bytecode::{Decoder, InlineOperands, Opcode};

use super::math::{self};
use super::overscale;
use super::state::{CoordAxis, GraphicsState, Point, ZonePointer};

/// Hard cap on executed instructions per program, guarding against malformed fonts
/// that would otherwise loop forever. Well above any real glyph/prep program.
const MAX_INSTRUCTIONS: u64 = 8_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HintError {
    StackUnderflow,
    StackOverflow,
    DivideByZero,
    CallStackOverflow,
    NestedDefinition,
    DefinitionInGlyphProgram,
    TooManyDefinitions,
    InvalidDefinition,
    UnhandledOpcode(u8),
    InvalidZone,
    UnexpectedEnd,
    Budget,
}

pub type R = Result<(), HintError>;

/// The glyph zone's coordinates (F26Dot6) after one point-mover op (MDAP/MDRP/MIRP/IP/IUP),
/// matching native dwrite's point-mover dump detour. Collected only while tracing is enabled, to
/// diff our hinting against a native capture.
#[derive(Clone, Debug)]
pub struct MoverRec {
    pub x: Vec<i32>,
    pub y: Vec<i32>,
    pub op: u8,
    pub ip: u32,
}

thread_local! {
    static TRACE: std::cell::RefCell<Option<Vec<MoverRec>>> = const { std::cell::RefCell::new(None) };
}

/// Start collecting a mover trace on this thread; pair with [`take_trace`] after `Instance::hint`.
pub fn enable_trace() {
    TRACE.with(|t| *t.borrow_mut() = Some(Vec::new()));
}

/// Take and clear the collected mover trace.
pub fn take_trace() -> Vec<MoverRec> {
    TRACE.with(|t| t.borrow_mut().take().unwrap_or_default())
}

/// Internal metric preparation is not part of the requested render-outline mover trace.
pub(super) fn without_trace<T>(f: impl FnOnce() -> T) -> T {
    let saved = TRACE.with(|t| t.borrow_mut().take());
    let result = f();
    TRACE.with(|t| *t.borrow_mut() = saved);
    result
}

fn is_mover(op: u8) -> bool {
    // MDAP/IUP/SHP/SHC (0x2E–0x35), SHPIX/IP/MSIRP/ALIGNRP (0x38–0x3C), MIAP (0x3E/0x3F),
    // MDRP/MIRP (0xC0–0xFF). SHZ (0x36/0x37) is excluded — it has no native detour in the capture.
    matches!(op, 0x2E..=0x35 | 0x38..=0x3C | 0x3E | 0x3F | 0xC0..=0xFF)
}

/// A function (FDEF) or instruction (IDEF) definition: a byte range within one program.
#[derive(Copy, Clone, Default)]
pub struct Definition {
    pub program: u8,
    pub start: u32,
    pub key: i32,
    pub active: bool,
    /// A VTT/compiler fpgm idiom DirectWrite recognises and neutralises under sub-pixel positioning:
    /// a function that gates a SHPIX pixel-snap on an MPPEM range/equality test. While it runs, native
    /// suppresses its x-direction SHPIX (see [`is_snap_function`] and `op_shpix`).
    pub neutral: bool,
}

/// Recognised font-tool functions ([`super::state::Retained::known_functions`]).
pub(crate) const KNOWN_GUARD22: u8 = 1;
pub(crate) const KNOWN_GUARD24: u8 = 2;
pub(crate) const KNOWN_FLAG8: u8 = 4;

/// Whether an fpgm function body is one of the MPPEM-gated SHPIX pixel-snap idioms DirectWrite adds to
/// its sub-pixel suppression list (an FDEF, matched by exact bytecode). Pattern A is
/// `MPPEM EQ IF SHPIX ELSE POP POP EIF`; pattern B is `MPPEM GTEQ SWAP MPPEM LTEQ AND IF SHPIX …` (the
/// Palatino/Courier `#` size-range snap). Matching the exact leading bytes mirrors native, which keys
/// off them to drop the integer x-snap so it never disturbs the fractional sub-pixel x.
fn is_snap_function(code: &[u8]) -> bool {
    const A: [u8; 9] = [0x4B, 0x54, 0x58, 0x38, 0x1B, 0x21, 0x21, 0x59, 0x2D];
    const B: [u8; 13] = [0x4B, 0x53, 0x23, 0x4B, 0x51, 0x5A, 0x58, 0x38, 0x1B, 0x21, 0x21, 0x59, 0x2D];
    code.starts_with(&A) || code.starts_with(&B)
}

/// One active function/loop-call invocation.
#[derive(Copy, Clone, Default)]
struct CallRecord {
    caller_program: u8,
    return_pc: usize,
    count: u32,
    def_program: u8,
    def_start: u32,
    neutral: bool,
    saved_cutin: Option<i32>,
    /// A storage slot this call reads as 0, and the value it gets back on return.
    saved_storage: Option<(usize, i32)>,
}

pub struct Engine<'a> {
    pub stack: Vec<i32>,
    pub max_stack: usize,
    pub gs: GraphicsState,
    pub cvt: Vec<i32>,
    pub storage: Vec<i32>,
    pub functions: Vec<Definition>,
    pub instructions: Vec<Definition>,
    /// True while running fpgm/prep (definitions allowed), false in a glyph program.
    pub can_define: bool,
    bytecode: [&'a [u8]; 3],
    current: u8,
    decoder: Decoder<'a>,
    call_stack: Vec<CallRecord>,
    instr_count: u64,
    /// Nesting depth of active calls into [`Definition::neutral`] functions — native's call-nesting flag,
    /// set by the CALL handler for the duration of such a call. While > 0, `op_shpix` drops x-snaps.
    neutral_depth: u32,
    /// Which axes' `IUP[]` have run this program — native's IUP-ran flags (bit 0 = x, bit 1 = y), set at
    /// the end of the IUP handler. Under sub-pixel positioning a post-IUP `DELTAP`
    /// on that axis is suppressed (see `deltap_allowed` / the delta engine).
    iup_axes: u8,
}

impl<'a> Engine<'a> {
    pub fn new(
        gs: GraphicsState,
        cvt: Vec<i32>,
        storage: Vec<i32>,
        functions: Vec<Definition>,
        instructions: Vec<Definition>,
        max_stack: usize,
        fpgm: &'a [u8],
        prep: &'a [u8],
    ) -> Self {
        Self {
            stack: Vec::with_capacity(max_stack.min(4096)),
            max_stack,
            gs,
            cvt,
            storage,
            functions,
            instructions,
            can_define: true,
            bytecode: [fpgm, prep, &[]],
            current: 0,
            decoder: Decoder::new(fpgm, 0),
            call_stack: Vec::new(),
            instr_count: 0,
            neutral_depth: 0,
            iup_axes: 0,
        }
    }

    /// Run one of the three programs (0 = fpgm, 1 = prep, 2 = glyph). `glyph_code` is the
    /// glyph's instruction stream (ignored for fpgm/prep).
    pub fn run_program(&mut self, program: u8, glyph_code: &'a [u8]) -> R {
        self.bytecode[2] = glyph_code;
        self.current = program;
        self.decoder = Decoder::new(self.bytecode[program as usize], 0);
        self.call_stack.clear();
        self.stack.clear();
        self.neutral_depth = 0;
        self.iup_axes = 0;
        self.can_define = program != 2;
        // Volatile graphics state resets at the start of every program; retained state
        // (set by prep: CVT cut-in, scan control, instruct control, …) persists.
        self.gs.reset_volatile();
        self.run()
    }

    fn run(&mut self) -> R {
        let itrace = std::env::var_os("D2D_ITRACE").is_some();
        loop {
            let ip = self.decoder.pc;
            let Some(ins) = self.decoder.decode() else { break };
            let ins = ins.map_err(|_| HintError::UnexpectedEnd)?;
            let op = ins.opcode as u8;
            if itrace && self.gs.is_glyph {
                let top: Vec<i32> = self.stack.iter().rev().take(4).cloned().collect();
                eprintln!("IT ip={ip} op={op:#04x} sp={} top={top:?} proj={:?}/{},{} free={:?}/{},{} rp0={} rp1={} rp2={}",
                    self.stack.len(), self.gs.proj_axis, self.gs.proj.x, self.gs.proj.y,
                    self.gs.free_axis, self.gs.free.x, self.gs.free.y,
                    self.gs.rp0, self.gs.rp1, self.gs.rp2);
            }
            self.dispatch(op, &ins.inline_operands)?;
            if self.gs.is_glyph && is_mover(op) && !self.gs.zp1.is_twilight() {
                let z = self.gs.zone(self.gs.zp1);
                let (x, y) = (z.cur.iter().map(|p| p.x).collect(), z.cur.iter().map(|p| p.y).collect());
                TRACE.with(|t| {
                    if let Some(v) = t.borrow_mut().as_mut() {
                        v.push(MoverRec { x, y, op, ip: ip as u32 });
                    }
                });
            }
            self.instr_count += 1;
            if self.instr_count > MAX_INSTRUCTIONS {
                return Err(HintError::Budget);
            }
        }
        Ok(())
    }

    // --- value stack helpers -------------------------------------------------

    fn push(&mut self, v: i32) -> R {
        if self.stack.len() >= self.max_stack {
            return Err(HintError::StackOverflow);
        }
        self.stack.push(v);
        Ok(())
    }
    fn pop(&mut self) -> Result<i32, HintError> {
        self.stack.pop().ok_or(HintError::StackUnderflow)
    }
    fn pop_usize(&mut self) -> Result<usize, HintError> {
        Ok(self.pop()? as usize)
    }
    fn binary(&mut self, f: impl Fn(i32, i32) -> i32) -> R {
        let b = self.pop()?;
        let a = self.pop()?;
        self.push(f(a, b))
    }
    fn unary(&mut self, f: impl Fn(i32) -> i32) -> R {
        let a = self.pop()?;
        self.push(f(a))
    }

    // --- program flow --------------------------------------------------------

    fn enter(&mut self, def: &Definition, count: u32) -> R {
        if self.call_stack.len() >= 256 {
            return Err(HintError::CallStackOverflow);
        }
        self.call_stack.push(CallRecord {
            caller_program: self.current,
            return_pc: self.decoder.pc,
            count,
            def_program: def.program,
            def_start: def.start,
            neutral: def.neutral,
            saved_cutin: None,
            saved_storage: None,
        });
        if def.neutral {
            self.neutral_depth += 1;
        }
        self.current = def.program;
        self.decoder = Decoder::new(self.bytecode[def.program as usize], def.start as usize);
        Ok(())
    }

    fn leave(&mut self) -> R {
        let mut rec = self.call_stack.pop().ok_or(HintError::UnexpectedEnd)?;
        if rec.count > 1 {
            rec.count -= 1;
            self.decoder = Decoder::new(self.bytecode[rec.def_program as usize], rec.def_start as usize);
            self.call_stack.push(rec);
        } else {
            if let Some(cutin) = rec.saved_cutin {
                self.gs.ret.control_value_cutin = cutin;
            }
            if let Some((i, v)) = rec.saved_storage {
                if let Some(s) = self.storage.get_mut(i) {
                    *s = v;
                }
            }
            if rec.neutral {
                self.neutral_depth = self.neutral_depth.saturating_sub(1);
            }
            self.current = rec.caller_program;
            self.decoder = Decoder::new(self.bytecode[rec.caller_program as usize], rec.return_pc);
        }
        Ok(())
    }

    fn find_def(defs: &[Definition], key: i32) -> Option<Definition> {
        if let Some(d) = defs.get(key as usize) {
            if d.active && d.key == key {
                return Some(*d);
            }
        }
        defs.iter().rev().find(|d| d.active && d.key == key).copied()
    }

    fn do_def(&mut self, key: i32, instruction: bool) -> R {
        if !self.can_define {
            return Err(HintError::DefinitionInGlyphProgram);
        }
        let start = self.decoder.pc;
        // Scan past the matching ENDF so the outer program resumes after the body.
        loop {
            match self.decoder.decode() {
                Some(Ok(ins)) => match ins.opcode {
                    Opcode::FDEF | Opcode::IDEF => return Err(HintError::NestedDefinition),
                    Opcode::ENDF => break,
                    _ => {}
                },
                _ => return Err(HintError::UnexpectedEnd),
            }
        }
        // DirectWrite recognises MPPEM-gated SHPIX pixel-snap functions at FDEF (fpgm only) and
        // neutralises their x-snap under sub-pixel positioning (see `is_snap_function`).
        let neutral = !instruction
            && self.current == 0
            && self
                .bytecode
                .get(self.current as usize)
                .and_then(|c| c.get(start..))
                .is_some_and(is_snap_function);
        // DirectWrite also recognises four common font-tool functions by their opening bytes and, under sub-pixel
        // positioning, overrides what they read (see `op_call` and `op_rs`): function 58's slope cut-in, a
        // storage-22 guard in functions 64-66, a storage-24 guard in functions 0-2, 4, 7 and 8 (after an x-axis
        // SVTCA, with or without RTG), and the storage-8 flag function 0 sets.
        if !instruction {
            let code = &self.bytecode[self.current as usize][start..];
            let ret = &mut self.gs.ret;
            match key {
                58 if code.starts_with(&[0x20, 0x20, 0xb0, 0x01, 0x60, 0x46, 0xb0, 0x40, 0x23, 0x42]) => ret.slope_function = true,
                64..=66 if code.starts_with(&[0xb0, 0x16, 0x43, 0x58]) => ret.known_functions |= KNOWN_GUARD22,
                0..=2 | 4 | 7 | 8 if code.starts_with(&[0x01, 0xb0, 0x18, 0x43, 0x58]) || code.starts_with(&[0x01, 0x18, 0xb0, 0x18, 0x43, 0x58]) => {
                    ret.known_functions |= KNOWN_GUARD24
                }
                _ => {}
            }
            if key == 0 && code.starts_with(&[0x45, 0x23, 0x46, 0x60, 0x20, 0xb0, 0x26]) {
                ret.known_functions |= KNOWN_FLAG8;
            }
        }
        let def = Definition { program: self.current, start: start as u32, key, active: true, neutral };
        let defs = if instruction { &mut self.instructions } else { &mut self.functions };
        // Prefer key-as-index, else reuse an existing entry with this key, else the
        // highest inactive slot (reference `DefinitionMap::allocate`).
        let slot = if defs.get(key as usize).map_or(false, |d| !d.active || d.key == key) {
            Some(key as usize)
        } else {
            let mut last_inactive = None;
            for (i, d) in defs.iter().enumerate().rev() {
                if d.active {
                    if d.key == key {
                        last_inactive = Some(i);
                        break;
                    }
                } else if last_inactive.is_none() {
                    last_inactive = Some(i);
                }
            }
            last_inactive
        };
        match slot.and_then(|i| defs.get_mut(i)) {
            Some(d) => {
                *d = def;
                Ok(())
            }
            None => Err(HintError::TooManyDefinitions),
        }
    }

    fn do_call(&mut self, key: i32, count: u32, instruction: bool) -> R {
        if count == 0 {
            return Ok(());
        }
        let defs = if instruction { &self.instructions } else { &self.functions };
        match Self::find_def(defs, key) {
            Some(d) => self.enter(&d, count),
            None if instruction => Err(HintError::UnhandledOpcode(key as u8)),
            None => Err(HintError::InvalidDefinition),
        }
    }

    // --- dispatch ------------------------------------------------------------

    fn dispatch(&mut self, op: u8, operands: &InlineOperands) -> R {
        match op {
            0x00 | 0x01 | 0x02 | 0x03 | 0x04 | 0x05 => self.op_svtca(op),
            0x06 | 0x07 | 0x08 | 0x09 => self.op_svtl(op),
            0x0A => self.op_spvfs(),
            0x0B => self.op_sfvfs(),
            0x0C => self.op_gpv(),
            0x0D => self.op_gfv(),
            0x0E => self.op_sfvtpv(),
            0x0F => self.op_isect(),
            0x10 => { let p = self.pop_usize()?; self.gs.rp0 = p; Ok(()) }
            0x11 => { let p = self.pop_usize()?; self.gs.rp1 = p; Ok(()) }
            0x12 => { let p = self.pop_usize()?; self.gs.rp2 = p; Ok(()) }
            0x13 => { let z = self.pop_zone()?; self.gs.zp0 = z; Ok(()) }
            0x14 => { let z = self.pop_zone()?; self.gs.zp1 = z; Ok(()) }
            0x15 => { let z = self.pop_zone()?; self.gs.zp2 = z; Ok(()) }
            0x16 => { let z = self.pop_zone()?; self.gs.zp0 = z; self.gs.zp1 = z; self.gs.zp2 = z; Ok(()) }
            0x17 => self.op_sloop(),
            0x18 => { self.gs.round.mode = math::RoundMode::Grid; Ok(()) }
            0x19 => { self.gs.round.mode = math::RoundMode::HalfGrid; Ok(()) }
            0x1A => { self.gs.ret.min_distance = self.pop()?; Ok(()) }
            0x1B => self.op_else(),
            0x1C => self.op_jmpr(),
            0x1D => { self.gs.ret.control_value_cutin = self.pop()?; Ok(()) }
            0x1E => { self.gs.ret.single_width_cutin = self.pop()?; Ok(()) }
            0x1F => { let n = self.pop()?; self.gs.ret.single_width = math::fixmul(n, self.gs.ret.scale); Ok(()) }
            0x20 => self.op_dup(),
            0x21 => { self.pop()?; Ok(()) }
            0x22 => { self.stack.clear(); Ok(()) }
            0x23 => self.op_swap(),
            0x24 => { let n = self.stack.len() as i32; self.push(n) }
            0x25 => self.op_cindex(),
            0x26 => self.op_mindex(),
            0x27 => self.op_alignpts(),
            0x29 => self.op_utp(),
            0x2A => self.op_loopcall(),
            0x2B => self.op_call(),
            0x2C => { let k = self.pop()?; self.do_def(k, false) }
            0x2D => self.leave(),
            0x2E | 0x2F => self.op_mdap(op),
            0x30 | 0x31 => self.op_iup(op),
            0x32 | 0x33 => self.op_shp(op),
            0x34 | 0x35 => self.op_shc(op),
            0x36 | 0x37 => self.op_shz(op),
            0x38 => self.op_shpix(),
            0x39 => self.op_ip(),
            0x3A | 0x3B => self.op_msirp(op),
            0x3C => self.op_alignrp(),
            0x3D => { self.gs.round.mode = math::RoundMode::DoubleGrid; Ok(()) }
            0x3E | 0x3F => self.op_miap(op),
            0x40 | 0x41 => self.op_push(operands),
            0x42 => self.op_ws(),
            0x43 => self.op_rs(),
            0x44 => self.op_wcvtp(),
            0x45 => self.op_rcvt(),
            0x46 | 0x47 => self.op_gc(op),
            0x48 => self.op_scfs(),
            0x49 | 0x4A => self.op_md(op),
            0x4B => { let p = self.proj_ppem(); self.push(p) }
            0x4C => { let p = self.gs.ret.ppem.saturating_mul(64); self.push(p) }
            0x4D => { self.gs.ret.auto_flip = true; Ok(()) }
            0x4E => { self.gs.ret.auto_flip = false; Ok(()) }
            0x4F => { self.pop()?; Ok(()) } // DEBUG
            0x50 => self.binary(|a, b| (a < b) as i32),
            0x51 => self.binary(|a, b| (a <= b) as i32),
            0x52 => self.binary(|a, b| (a > b) as i32),
            0x53 => self.binary(|a, b| (a >= b) as i32),
            0x54 => self.binary(|a, b| (a == b) as i32),
            0x55 => self.binary(|a, b| (a != b) as i32),
            0x56 => self.op_odd(),
            0x57 => self.op_even(),
            0x58 => self.op_if(),
            0x59 => Ok(()), // EIF
            0x5A => self.binary(|a, b| (a != 0 && b != 0) as i32),
            0x5B => self.binary(|a, b| (a != 0 || b != 0) as i32),
            0x5C => self.unary(|a| (a == 0) as i32),
            0x5D => self.op_deltap(0),
            0x5E => { self.gs.ret.delta_base = self.pop()? as u16; Ok(()) }
            0x5F => { let n = self.pop()?; self.gs.ret.delta_shift = (n as u16) & 0xFFFF; Ok(()) }
            0x60 => self.binary(|a, b| a.wrapping_add(b)),
            0x61 => self.binary(|a, b| a.wrapping_sub(b)),
            0x62 => self.op_div(),
            0x63 => self.binary(|a, b| math::mul_div(a, b, 64)),
            0x64 => self.unary(|a| a.wrapping_abs()),
            0x65 => self.unary(|a| a.wrapping_neg()),
            0x66 => self.unary(math::floor),
            0x67 => self.unary(math::ceil),
            0x68 | 0x69 | 0x6A | 0x6B => self.op_round(),
            0x6C | 0x6D | 0x6E | 0x6F => Ok(()), // NROUND (engine compensation = 0)
            0x70 => self.op_wcvtf(),
            0x71 => self.op_deltap(1),
            0x72 => self.op_deltap(2),
            0x73 => self.op_deltac(0),
            0x74 => self.op_deltac(1),
            0x75 => self.op_deltac(2),
            0x76 => { let n = self.pop()?; self.gs.round.set_super(n, false); Ok(()) }
            0x77 => { let n = self.pop()?; self.gs.round.set_super(n, true); Ok(()) }
            0x78 => self.op_jrot(true),
            0x79 => self.op_jrot(false),
            0x7A => { self.gs.round.mode = math::RoundMode::Off; Ok(()) }
            0x7C => { self.gs.round.mode = math::RoundMode::UpToGrid; Ok(()) }
            0x7D => { self.gs.round.mode = math::RoundMode::DownToGrid; Ok(()) }
            0x7E => { self.pop()?; Ok(()) } // SANGW
            0x7F => Ok(()),                 // AA
            0x80 => self.op_flippt(),
            0x81 => self.op_fliprg(true),
            0x82 => self.op_fliprg(false),
            0x85 => self.op_scanctrl(),
            0x86 | 0x87 => self.op_sdpvtl(op),
            0x88 => self.op_getinfo(),
            0x89 => { let k = self.pop()?; self.do_def(k, true) }
            0x8A => self.op_roll(),
            0x8B => self.binary(|a, b| a.max(b)),
            0x8C => self.binary(|a, b| a.min(b)),
            0x8D => { self.gs.ret.scan_type = self.pop()? & 0xFFFF; Ok(()) }
            0x8E => self.op_instctrl(),
            0x91 => self.op_getvariation(),
            0x92 => self.op_unknown(0x92),
            0xB0..=0xB7 => self.op_push(operands),
            0xB8..=0xBF => self.op_push(operands),
            0xC0..=0xDF => self.op_mdrp(op),
            0xE0..=0xFF => self.op_mirp(op),
            _ => self.op_unknown(op),
        }
    }

    fn op_unknown(&mut self, op: u8) -> R {
        self.do_call(op as i32, 1, true)
    }

    fn pop_zone(&mut self) -> Result<ZonePointer, HintError> {
        ZonePointer::from_stack(self.pop()?).ok_or(HintError::InvalidZone)
    }

    // --- vectors / graphics state -------------------------------------------

    fn op_svtca(&mut self, op: u8) -> R {
        let x = ((op as i32) & 1) << 14;
        let y = x ^ 0x4000;
        let v = Point::new(x, y);
        if op < 4 {
            self.gs.proj = v;
            self.gs.dual = v;
            self.gs.proj_generic = false;
            // Native SVTCA/SPVTCA forget the SPVTL line points, so a later MDRP does not
            // take the InterAlign branch off a stale line.
            self.gs.proj_ref = (-1, -1);
        }
        if op & 2 == 0 {
            self.gs.free = v;
        }
        self.gs.update_projection();
        Ok(())
    }

    fn op_svtl(&mut self, op: u8) -> R {
        let i1 = self.pop_usize()?;
        let i2 = self.pop_usize()?;
        let parallel = op & 1 == 0;
        let p1 = self.gs.zone(self.gs.zp1).point(i2);
        let p2 = self.gs.zone(self.gs.zp2).point(i1);
        let v = line_vector(p1, p2, parallel);
        if op < 8 {
            self.gs.proj = v;
            self.gs.dual = v;
            // Native SPVTL records the two line points for MDRP's inter-align branch.
            self.gs.proj_ref = (i1 as i32, i2 as i32);
            self.gs.proj_generic = true;
        } else {
            self.gs.free = v;
        }
        self.gs.update_projection();
        Ok(())
    }

    fn op_sdpvtl(&mut self, op: u8) -> R {
        let i1 = self.pop_usize()?;
        let i2 = self.pop_usize()?;
        let parallel = op & 1 == 0;
        let o1 = self.gs.zone(self.gs.zp1).original(i2);
        let o2 = self.gs.zone(self.gs.zp2).original(i1);
        self.gs.dual = line_vector(o1, o2, parallel);
        let p1 = self.gs.zone(self.gs.zp1).point(i2);
        let p2 = self.gs.zone(self.gs.zp2).point(i1);
        self.gs.proj = line_vector(p1, p2, parallel);
        self.gs.proj_ref = (i1 as i32, i2 as i32);
        self.gs.proj_generic = false;
        self.gs.update_projection();
        Ok(())
    }

    fn op_spvfs(&mut self) -> R {
        let y = self.pop()? as i16 as i32;
        let x = self.pop()? as i16 as i32;
        let v = if (x, y) == (0, 0) { self.gs.proj } else { let (a, b) = math::normalize14(x, y); Point::new(a, b) };
        self.gs.proj = v;
        self.gs.dual = v;
        self.gs.proj_generic = true;
        self.gs.proj_ref = (-1, -1); // native's vector writes forget the SPVTL line points too
        self.gs.update_projection();
        Ok(())
    }
    fn op_sfvfs(&mut self) -> R {
        let y = self.pop()? as i16 as i32;
        let x = self.pop()? as i16 as i32;
        let v = if (x, y) == (0, 0) { self.gs.free } else { let (a, b) = math::normalize14(x, y); Point::new(a, b) };
        self.gs.free = v;
        self.gs.update_projection();
        Ok(())
    }
    fn op_gpv(&mut self) -> R {
        let (x, y) = (self.gs.proj.x, self.gs.proj.y);
        self.push(x)?;
        self.push(y)
    }
    fn op_gfv(&mut self) -> R {
        let (x, y) = (self.gs.free.x, self.gs.free.y);
        self.push(x)?;
        self.push(y)
    }
    fn op_sfvtpv(&mut self) -> R {
        self.gs.free = self.gs.proj;
        self.gs.update_projection();
        // Native's SFVTPV handling sets this exactly,
        // independent of the rounded 2.14 vector's measured length.
        self.gs.fdotp = 0x4000;
        Ok(())
    }

    fn op_sloop(&mut self) -> R {
        let n = self.pop()?;
        if n < 0 {
            return Err(HintError::StackUnderflow);
        }
        self.gs.loop_counter = (n as u32).min(0xFFFF);
        Ok(())
    }

    fn op_instctrl(&mut self) -> R {
        let selector = self.pop()? as u32;
        let value = self.pop()? as u32;
        if !(1..=3).contains(&selector) {
            return Ok(());
        }
        let flag = 1u32 << (selector - 1);
        if value != 0 && value != flag {
            return Ok(());
        }
        // Only the control-value program may set the persistent flags; the classic
        // rasterizer has no glyph-level backward-compatibility toggle.
        if self.current == 1 {
            self.gs.ret.instruct_control &= !(flag as u8);
            self.gs.ret.instruct_control |= value as u8;
        }
        Ok(())
    }

    fn op_scanctrl(&mut self) -> R {
        // Native SCANCTRL only stores the word; the glyph setup evaluates it per glyph (Retained::dropout_on).
        self.gs.ret.scan_word = self.pop()? & 0xFFFF;
        Ok(())
    }

    fn op_getinfo(&mut self) -> R {
        let sel = self.pop()?;
        let g = &self.gs.ret.getinfo;
        let mut r = 0i32;
        if sel & 0x01 != 0 {
            r = g.version;
        }
        // Native GETINFO reports rotated (sel 2) and stretched (sel 4) only while a scaler flag is clear that
        // DirectWrite always sets (every VM dump has it set). So a quarter turn, a shear or a stretch never
        // shows here; the rotate/stretch state still drives dropout control (`Retained::dropout_on`).
        if sel & 0x08 != 0 {
            r |= 0x400;
        }
        if sel & 0x10 != 0 {
            r |= 0x800;
        }
        // Selector 0x20 (ClearType enabled) answers from a scaler byte DirectWrite leaves clear on every pass:
        // VM GETINFO dumps of Courier New NATURAL, NATURAL_SYMMETRIC and GDI ClearType all answer 0. Courier's
        // prep keys its render-mode storage on it, so answering it skipped the NATURAL ClearType SHPIX snaps.
        if sel & 0x40 != 0 && g.subpixel_hinting {
            r |= 0x2000;
        }
        // Native carries compatible widths in the pass flags' bit 1 only on the subpixel render pass.
        if sel & 0x80 != 0 && g.compatible_width && g.subpixel_hinting {
            r |= 0x4000;
        }
        if sel & 0x100 != 0 && g.vertical_lcd {
            r |= 0x8000;
        }
        if sel & 0x200 != 0 && g.bgr {
            r |= 0x10000;
        }
        if sel & 0x400 != 0 && g.subpixel_positioned {
            r |= 0x20000;
        }
        if sel & 0x800 != 0 && g.symmetric {
            r |= 0x40000;
        }
        if sel & 0x1000 != 0 && g.grayscale_cleartype {
            r |= 0x80000;
        }
        self.push(r)
    }

    fn op_getvariation(&mut self) -> R {
        // Non-variable instance: no axes. Fall back to IDEF resolution like the reference.
        self.op_unknown(0x91)
    }

    // --- stack ---------------------------------------------------------------

    fn op_dup(&mut self) -> R {
        let v = *self.stack.last().ok_or(HintError::StackUnderflow)?;
        self.push(v)
    }
    fn op_swap(&mut self) -> R {
        let n = self.stack.len();
        if n < 2 {
            return Err(HintError::StackUnderflow);
        }
        self.stack.swap(n - 1, n - 2);
        Ok(())
    }
    fn op_cindex(&mut self) -> R {
        let k = self.pop()?;
        if k <= 0 || k as usize > self.stack.len() {
            return Err(HintError::StackUnderflow);
        }
        let v = self.stack[self.stack.len() - k as usize];
        self.push(v)
    }
    fn op_mindex(&mut self) -> R {
        let k = self.pop()?;
        if k <= 0 || k as usize > self.stack.len() {
            return Err(HintError::StackUnderflow);
        }
        let idx = self.stack.len() - k as usize;
        let v = self.stack.remove(idx);
        self.push(v)
    }
    fn op_roll(&mut self) -> R {
        let n = self.stack.len();
        if n < 3 {
            return Err(HintError::StackUnderflow);
        }
        let a = self.stack[n - 3];
        self.stack[n - 3] = self.stack[n - 2];
        self.stack[n - 2] = self.stack[n - 1];
        self.stack[n - 1] = a;
        Ok(())
    }
    fn op_push(&mut self, operands: &InlineOperands) -> R {
        for v in operands.values() {
            self.push(v)?;
        }
        Ok(())
    }

    // --- arithmetic ----------------------------------------------------------

    fn op_div(&mut self) -> R {
        let b = self.pop()?;
        let a = self.pop()?;
        if b == 0 {
            return Err(HintError::DivideByZero);
        }
        self.push(math::mul_div_no_round(a, 64, b))
    }
    fn op_round(&mut self) -> R {
        let n = self.pop()?;
        let r = self.gs.round_dist(n);
        self.push(r)
    }
    fn op_odd(&mut self) -> R {
        let n = self.pop()?;
        let r = self.gs.round_dist(n);
        self.push((r & 64 == 64) as i32)
    }
    fn op_even(&mut self) -> R {
        let n = self.pop()?;
        let r = self.gs.round_dist(n);
        self.push((r & 64 == 0) as i32)
    }

    // --- storage / cvt -------------------------------------------------------

    fn op_rs(&mut self) -> R {
        let i = self.pop_usize()?;
        // Under sub-pixel positioning, a font whose function 0 is the recognised flag function reads its storage-8
        // flag as 0, so the glyph program's whole-pixel grid-fit of curve points is skipped (VM: Arial 'u').
        let g = &self.gs.ret.getinfo;
        let v = if i == 8 && self.gs.ret.known_functions & KNOWN_FLAG8 != 0 && g.subpixel_hinting && self.gs.ret.instruct_control & 4 == 0 {
            0
        } else {
            self.storage.get(i).copied().unwrap_or(0)
        };
        self.push(v)
    }
    fn op_ws(&mut self) -> R {
        let v = self.pop()?;
        let i = self.pop_usize()?;
        if let Some(s) = self.storage.get_mut(i) {
            *s = v;
        }
        Ok(())
    }
    fn op_wcvtp(&mut self) -> R {
        let mut v = self.pop()?;
        let i = self.pop_usize()?;
        // WCVTP: a non-square instance stores the pixel value unstretched.
        if !self.gs.ret.square() && v != 0 {
            v = math::fixdiv(v, self.cvt_scale());
        }
        if let Some(c) = self.cvt.get_mut(i) {
            *c = v;
        }
        Ok(())
    }
    fn op_wcvtf(&mut self) -> R {
        let v = self.pop()?;
        let i = self.pop_usize()?;
        // WCVTF scales FUnits→F26Dot6 through native's outline scaler, which rounds
        // **half toward +∞** (`math::scale`), not half-away like `fixmul`: a value landing exactly on
        // a .5 grid goes up, so −4fu@20px (−2.5) becomes −2, not −3. Read from the oracle (Segoe UI
        // 'ñ' composite WCVTF cvt[12]) — the 1-unit difference flips a downstream ROUND and mislands
        // the tilde by 0.0625px. (Native's WCVT-write path: the same scaler as outline points.)
        let scaled = math::scale(v, self.gs.ret.scale);
        if let Some(c) = self.cvt.get_mut(i) {
            *c = scaled;
        }
        Ok(())
    }
    fn op_rcvt(&mut self) -> R {
        let i = self.pop_usize()?;
        let v = self.cvt_get(i);
        self.push(v)
    }
    /// Reading a CVT entry: a non-square instance stretches it along the projection vector.
    fn cvt_get(&self, i: usize) -> i32 {
        let v = self.cvt.get(i).copied().unwrap_or(0);
        if self.gs.ret.square() { v } else { math::fixmul(v, self.cvt_scale()) }
    }
    /// The CVT scale: the x ratio on an x projection, the y ratio on a y projection, otherwise
    /// FracSqrt(px²·xr² + py²·yr²).
    fn cvt_scale(&self) -> i32 {
        let (px, py, r) = (self.gs.proj.x, self.gs.proj.y, &self.gs.ret);
        if py == 0 {
            return r.x_ratio;
        }
        if px == 0 {
            return r.y_ratio;
        }
        let sq = |v: i32| 4 * (((v * v + 0x2000) >> 14) as i16 as i32);
        let s = math::fixmul(sq(px), math::fixmul(r.x_ratio, r.x_ratio)) + math::fixmul(sq(py), math::fixmul(r.y_ratio, r.y_ratio));
        if s <= 0x10000 { (math::frac_sqrt(s << 14) + 0x2000) >> 14 } else { 0x10000 }
    }
    /// MPPEM and the DELTA instructions: a non-square instance's ppem along the projection vector.
    fn proj_ppem(&self) -> i32 {
        let p = self.gs.ret.ppem;
        if self.gs.ret.square() { p } else { math::fixmul(p, self.cvt_scale()) }
    }
    /// An original FUnit distance (`a − b`) in pixels along the dual projection: native MDRP/MD scale the
    /// projected FUnits when square, else each axis by its own scale before projecting.
    fn fu_distance(&self, a: Point, b: Point) -> i32 {
        let r = &self.gs.ret;
        if r.square() || self.gs.is_composite {
            return math::scale(self.gs.dual_project(a, b), self.gs.unscaled_to_pixels());
        }
        let d = Point::new(math::scale(a.x.wrapping_sub(b.x), r.x_scale), math::scale(a.y.wrapping_sub(b.y), r.y_scale));
        self.gs.dual_project(d, Point::default())
    }

    // --- data ----------------------------------------------------------------

    fn op_gc(&mut self, op: u8) -> R {
        let p = self.pop_usize()?;
        if !self.gs.in_bounds(&[(self.gs.zp2, p)]) {
            return self.push(0);
        }
        let z = self.gs.zone(self.gs.zp2);
        let v = if op & 1 != 0 {
            self.gs.dual_project(z.original(p), Point::default())
        } else {
            self.gs.project(z.point(p), Point::default())
        };
        self.push(v)
    }
    fn op_scfs(&mut self) -> R {
        let value = self.pop()?;
        let p = self.pop_usize()?;
        let zp2 = self.gs.zp2;
        let proj = self.gs.project(self.gs.zone(zp2).point(p), Point::default());
        self.gs.move_point(zp2, p, value.wrapping_sub(proj));
        if zp2.is_twilight() {
            let z = self.gs.zone_mut(ZonePointer::Twilight);
            if let (Some(&cur), Some(orig)) = (z.cur.get(p), z.original.get_mut(p)) {
                *orig = cur;
            }
        }
        Ok(())
    }
    fn op_md(&mut self, op: u8) -> R {
        let p1 = self.pop_usize()?;
        let p2 = self.pop_usize()?;
        if !self.gs.in_bounds(&[(self.gs.zp0, p2), (self.gs.zp1, p1)]) {
            return self.push(0);
        }
        let d = if op & 1 != 0 {
            self.gs.project(self.gs.zone(self.gs.zp0).point(p2), self.gs.zone(self.gs.zp1).point(p1))
        } else if self.gs.zp0.is_twilight() || self.gs.zp1.is_twilight() {
            self.gs.dual_project(self.gs.zone(self.gs.zp0).original(p2), self.gs.zone(self.gs.zp1).original(p1))
        } else {
            let (a, b) = (self.gs.zone(self.gs.zp0).unscaled(p2), self.gs.zone(self.gs.zp1).unscaled(p1));
            if self.gs.ret.square() || self.gs.is_composite {
                math::fixmul(self.gs.dual_project(a, b), self.gs.unscaled_to_pixels())
            } else {
                self.fu_distance(a, b)
            }
        };
        self.push(d)
    }

    // --- control flow --------------------------------------------------------

    fn op_if(&mut self) -> R {
        if self.pop()? == 0 {
            let mut depth = 1;
            loop {
                let op = self.decode_opcode()?;
                match op {
                    Opcode::IF => depth += 1,
                    Opcode::ELSE => {
                        if depth == 1 {
                            break;
                        }
                    }
                    Opcode::EIF => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }
    fn op_else(&mut self) -> R {
        let mut depth = 1;
        while depth != 0 {
            match self.decode_opcode()? {
                Opcode::IF => depth += 1,
                Opcode::EIF => depth -= 1,
                _ => {}
            }
        }
        Ok(())
    }
    fn op_jmpr(&mut self) -> R {
        let off = self.pop()?.wrapping_sub(1);
        self.decoder.pc = self.decoder.pc.wrapping_add_signed(off as isize);
        self.jump_budget()
    }
    fn op_jrot(&mut self, want: bool) -> R {
        let e = self.pop()?;
        let off = self.pop()?.wrapping_sub(1);
        if (e != 0) == want {
            self.decoder.pc = self.decoder.pc.wrapping_add_signed(off as isize);
            self.jump_budget()?;
        }
        Ok(())
    }
    fn jump_budget(&mut self) -> R {
        self.instr_count += 1;
        if self.instr_count > MAX_INSTRUCTIONS {
            Err(HintError::Budget)
        } else {
            Ok(())
        }
    }
    fn decode_opcode(&mut self) -> Result<Opcode, HintError> {
        self.instr_count += 1;
        if self.instr_count > MAX_INSTRUCTIONS {
            return Err(HintError::Budget);
        }
        Ok(self.decoder.decode().ok_or(HintError::UnexpectedEnd)?.map_err(|_| HintError::UnexpectedEnd)?.opcode)
    }

    // --- definitions ---------------------------------------------------------

    fn op_call(&mut self) -> R {
        let f = self.pop()?;
        self.do_call(f, 1, false)?;
        // A recognised guard function reads its storage flag as 0 for the call, so its pixel-snap body is skipped
        // (VM: Impact, whose function 66 is not one of them, keeps its flag; its GDI-natural 'A', 'V' and 'W'
        // advance 13, 14 and 21 px at ppem 26).
        let subpixel = self.gs.ret.getinfo.subpixel_hinting && self.gs.ret.instruct_control & 4 == 0;
        let guard = match f {
            64..=66 if self.gs.ret.known_functions & KNOWN_GUARD22 != 0 => Some(22),
            0..=2 | 4 | 7 | 8 if self.gs.ret.known_functions & KNOWN_GUARD24 != 0 => Some(24),
            _ => None,
        };
        if let Some(i) = guard.filter(|_| subpixel) {
            if let Some(v) = self.storage.get(i).copied() {
                self.call_stack.last_mut().unwrap().saved_storage = Some((i, v));
                self.storage[i] = 0;
            }
        }
        if f == 58 && self.gs.ret.slope_function && self.gs.ret.getinfo.subpixel_hinting
            && self.gs.ret.instruct_control & 4 == 0
        {
            self.call_stack.last_mut().unwrap().saved_cutin = Some(self.gs.ret.control_value_cutin);
            self.gs.ret.control_value_cutin = i32::MAX;
        }
        Ok(())
    }
    fn op_loopcall(&mut self) -> R {
        let f = self.pop()?;
        let count = self.pop()?;
        if count > 0 {
            self.do_call(f, count as u32, false)
        } else {
            Ok(())
        }
    }

    // --- outline-moving instructions -----------------------------------------

    fn op_flippt(&mut self) -> R {
        let count = self.gs.loop_counter as usize;
        self.gs.loop_counter = 1;
        for _ in 0..count {
            let p = self.pop_usize()?;
            self.gs.zone_mut(ZonePointer::Glyph).flip_on_curve(p);
        }
        Ok(())
    }
    fn op_fliprg(&mut self, on: bool) -> R {
        let hi = self.pop_usize()?;
        let lo = self.pop_usize()?;
        self.gs.zone_mut(ZonePointer::Glyph).set_on_curve(lo, hi.saturating_add(1), on);
        Ok(())
    }
    fn op_utp(&mut self) -> R {
        let p = self.pop_usize()?;
        let axis = match (self.gs.free.x != 0, self.gs.free.y != 0) {
            (true, true) => Some(CoordAxis::Both),
            (true, false) => Some(CoordAxis::X),
            (false, true) => Some(CoordAxis::Y),
            (false, false) => None,
        };
        if let Some(axis) = axis {
            self.gs.zone_mut(self.gs.zp0).untouch(p, axis);
        }
        Ok(())
    }

    fn op_shp(&mut self, op: u8) -> R {
        let (_, reference, dx, dy) = self.gs.point_displacement(op);
        let count = self.gs.loop_counter;
        self.gs.loop_counter = 1;
        for _ in 0..count {
            let p = self.pop_usize()?;
            self.gs.move_zp2(p, dx, dy, true);
            self.gdi_link_shp(reference, p);
        }
        Ok(())
    }
    fn op_shc(&mut self, op: u8) -> R {
        let c = self.pop_usize()?;
        if c >= self.gs.zone(self.gs.zp2).contours.len() {
            return Ok(());
        }
        let (zone, pi, dx, dy) = self.gs.point_displacement(op);
        let z = self.gs.zone(self.gs.zp2);
        let start = if c != 0 { z.contour(c - 1) as usize + 1 } else { 0 };
        let end = if self.gs.zp2.is_twilight() { z.len() } else { z.contour(c) as usize + 1 };
        for i in start..end {
            if zone != self.gs.zp2 || pi != i {
                self.gs.move_zp2(i, dx, dy, true);
            }
        }
        Ok(())
    }
    fn op_shz(&mut self, op: u8) -> R {
        let _e = self.pop_zone()?;
        let (zone, pi, dx, dy) = self.gs.point_displacement(op);
        let z = self.gs.zone(self.gs.zp2);
        let end = if self.gs.zp2.is_twilight() {
            z.len()
        } else if !z.contours.is_empty() {
            *z.contours.last().unwrap() as usize + 1
        } else {
            0
        };
        for i in 0..end {
            if zone != self.gs.zp2 || i != pi {
                self.gs.move_zp2(i, dx, dy, false);
            }
        }
        Ok(())
    }
    fn op_shpix(&mut self) -> R {
        let amount = self.pop()?;
        let dx = math::mul14(amount, self.gs.free.x);
        let dy = math::mul14(amount, self.gs.free.y);
        // Inside a recognised pixel-snap function, native (the common SHP handler under the call-nesting flag)
        // drops the shift on the subpixel axis: only a pure non-subpixel-axis shift of a touched point
        // survives. Elsewhere SHPIX applies unconditionally.
        let suppress = self.neutral_depth > 0;
        let count = self.gs.loop_counter;
        self.gs.loop_counter = 1;
        for _ in 0..count {
            let p = self.pop_usize()?;
            if suppress && !self.shpix_axis_allowed(p) {
                continue;
            }
            self.gs.move_zp2(p, dx, dy, true);
        }
        Ok(())
    }

    /// In a neutralised snap function, whether a SHPIX may move `p`: only a pure non-subpixel-axis
    /// shift of a point already touched on that axis (vertical LCD → x on an x-touched point, else y
    /// on a y-touched point), matching the common SHP handler's gate. Outside subpixel hinting (or
    /// with INSTCTRL bit 2 set) the shift is never gated. Native's CALL handler applies
    /// the same suppression in compatible-width rendering.
    fn shpix_axis_allowed(&self, p: usize) -> bool {
        const ONE: i32 = 0x4000;
        let g = &self.gs.ret.getinfo;
        if !g.subpixel_hinting || (self.gs.ret.instruct_control & 4) != 0 {
            return true;
        }
        let (fx, fy) = (self.gs.free.x, self.gs.free.y);
        let z = self.gs.zone(self.gs.zp2);
        if g.vertical_lcd {
            fx == ONE && fy == 0 && self.iup_axes & 1 == 0 && z.is_touched(p, CoordAxis::X)
        } else {
            fy == ONE && fx == 0 && self.iup_axes & 2 == 0 && z.is_touched(p, CoordAxis::Y)
        }
    }
    fn op_msirp(&mut self, op: u8) -> R {
        let mut distance = self.pop()?;
        let p = self.pop_usize()?;
        if !self.gs.in_bounds(&[(self.gs.zp1, p), (self.gs.zp0, self.gs.rp0)]) {
            return Ok(());
        }
        if self.gs.zp1.is_twilight() {
            // Native/FreeType Ins_MSIRP twilight branch seeds the point's ORIGINAL (not cur) from rp0's
            // original (`zp1.org[p] = zp0.org[rp0]`), moves the original along the freedom vector, then copies
            // original->cur. Seeding cur instead left the point's stale original in place, so a twilight point
            // reused across stems kept an earlier reference's off-axis coordinate — in Arial italic's non-square
            // (stretched) stem dance point 2 stuck at the cap-height anchor (y=733) instead of rp0's mid anchor
            // (y=501), flattening the stem and under-stretching every stem glyph's x on a non-uniform transform.
            let base = self.gs.zone(self.gs.zp0).original(self.gs.rp0);
            if let Some(orig) = self.gs.zone_mut(self.gs.zp1).original.get_mut(p) {
                *orig = base;
            }
            self.gs.move_original(self.gs.zp1, p, distance);
            let orig = self.gs.zone(self.gs.zp1).original(p);
            if let Some(cur) = self.gs.zone_mut(self.gs.zp1).cur.get_mut(p) {
                *cur = orig;
            }
        }
        // native MSIRP cut-in: on the subpixel projection while the prep has not set INSTCTRL selector 3
        // (bit 2 of the INSTCTRL word), a commanded distance deviating from the original (design) distance by
        // more than the control-value cut-in (16·|operand−orig|) reverts to the design distance. It keeps a
        // composite accent on its design anchor (é/ñ) and a simple glyph's advance phantom where an MSIRP
        // would move it (Times 'g' GDI_NATURAL @23: 612+108 → design 736), GDI-classic's render pass included
        // (Arial Bold 'j' @33: the advance phantom stays at its design 587).
        if self.gs.subpixel_round && self.gs.ret.instruct_control & 4 == 0 {
            let orig = self.gs.dual_project(self.gs.zone(self.gs.zp1).original(p), self.gs.zone(self.gs.zp0).original(self.gs.rp0));
            if orig != 0 && (16i64 * distance.wrapping_sub(orig) as i64).abs() > self.gs.ret.control_value_cutin as i64 {
                distance = orig;
            }
        }
        let d = self.gs.project(self.gs.zone(self.gs.zp1).point(p), self.gs.zone(self.gs.zp0).point(self.gs.rp0));
        self.gs.move_point(self.gs.zp1, p, distance.wrapping_sub(d));
        self.gdi_link_mirp(1, p, self.gs.rp0); // MSIRP's link block (DoubleCheckLinkColor default colour 1)
        self.gs.rp1 = self.gs.rp0;
        self.gs.rp2 = p;
        if op & 1 != 0 {
            self.gs.rp0 = p;
        }
        Ok(())
    }
    fn op_mdap(&mut self, op: u8) -> R {
        let p = self.pop_usize()?;
        if !self.gs.in_bounds(&[(self.gs.zp0, p)]) {
            self.gs.rp0 = p;
            self.gs.rp1 = p;
            return Ok(());
        }
        let distance = if op & 1 != 0 {
            let cur = self.gs.project(self.gs.zone(self.gs.zp0).point(p), Point::default());
            self.gs.round_dist(cur).wrapping_sub(cur)
        } else {
            0
        };
        self.gs.move_point(self.gs.zp0, p, distance);
        self.gs.rp0 = p;
        self.gs.rp1 = p;
        Ok(())
    }
    fn op_miap(&mut self, op: u8) -> R {
        let entry = self.pop_usize()?;
        let p = self.pop_usize()?;
        let mut distance = self.cvt_get(entry);
        if self.gs.zp0.is_twilight() {
            let fv = self.gs.free;
            let np = Point::new(math::mul14(distance, fv.x), math::mul14(distance, fv.y));
            let z = self.gs.zone_mut(ZonePointer::Twilight);
            if let Some(o) = z.original.get_mut(p) {
                *o = np;
            }
            if let Some(c) = z.cur.get_mut(p) {
                *c = np;
            }
        }
        let orig_dist = self.gs.project(self.gs.zone(self.gs.zp0).point(p), Point::default());
        if op & 1 != 0 {
            if (distance.wrapping_sub(orig_dist)).abs() > self.gs.ret.control_value_cutin {
                distance = orig_dist;
            }
            distance = self.gs.round_dist(distance);
        }
        self.gs.move_point(self.gs.zp0, p, distance.wrapping_sub(orig_dist));
        self.gs.rp0 = p;
        self.gs.rp1 = p;
        Ok(())
    }
    fn op_mdrp(&mut self, op: u8) -> R {
        let p = self.pop_usize()?;
        if !self.gs.in_bounds(&[(self.gs.zp1, p), (self.gs.zp0, self.gs.rp0)]) {
            self.gs.rp1 = self.gs.rp0;
            self.gs.rp2 = p;
            if op & 16 != 0 {
                self.gs.rp0 = p;
            }
            return Ok(());
        }
        let mut orig = if self.gs.zp0.is_twilight() || self.gs.zp1.is_twilight() {
            self.gs.dual_project(self.gs.zone(self.gs.zp1).original(p), self.gs.zone(self.gs.zp0).original(self.gs.rp0))
        } else {
            let v1 = self.gs.zone(self.gs.zp1).unscaled(p);
            let v2 = self.gs.zone(self.gs.zp0).unscaled(self.gs.rp0);
            // Native's MDRP uses the outline scaler, including its positive half-tie bias.
            self.fu_distance(v1, v2)
        };
        orig = self.apply_single_width(orig);
        let mut distance = if op & 4 != 0 { self.gs.round_dist(orig) } else { orig };
        if op & 8 != 0 {
            let min = self.gs.min_dist();
            if orig >= 0 {
                if distance < min {
                    distance = min;
                }
            } else if distance > -min {
                distance = -min;
            }
        }
        let cur = self.gs.project(self.gs.zone(self.gs.zp1).point(p), self.gs.zone(self.gs.zp0).point(self.gs.rp0));
        self.gs.move_point(self.gs.zp1, p, distance.wrapping_sub(cur));
        self.gdi_link_mdrp(p, self.gs.rp0);
        self.gs.rp1 = self.gs.rp0;
        self.gs.rp2 = p;
        if op & 16 != 0 {
            self.gs.rp0 = p;
        }
        Ok(())
    }
    fn op_mirp(&mut self, op: u8) -> R {
        let n = self.pop()?.wrapping_add(1) as usize;
        let p = self.pop_usize()?;
        if !self.gs.in_bounds(&[(self.gs.zp1, p), (self.gs.zp0, self.gs.rp0)]) || n > self.cvt.len() {
            self.gs.rp1 = self.gs.rp0;
            if op & 16 != 0 {
                self.gs.rp0 = p;
            }
            self.gs.rp2 = p;
            return Ok(());
        }
        let mut cvt = if n == 0 { 0 } else { self.cvt_get(n - 1) };
        // single width
        if (cvt.wrapping_sub(self.gs.ret.single_width)).abs() < self.gs.ret.single_width_cutin {
            cvt = if cvt >= 0 { self.gs.ret.single_width } else { -self.gs.ret.single_width };
        }
        if self.gs.zp1.is_twilight() {
            let fv = self.gs.free;
            let base = self.gs.zone(self.gs.zp0).original(self.gs.rp0);
            let np = Point::new(
                base.x.wrapping_add(math::fixmul(cvt, fv.x)),
                base.y.wrapping_add(math::fixmul(cvt, fv.y)),
            );
            if let Some(orig) = self.gs.zone_mut(self.gs.zp1).original.get_mut(p) {
                *orig = np;
            }
            if let Some(cur) = self.gs.zone_mut(self.gs.zp1).cur.get_mut(p) {
                *cur = np;
            }
        }
        let orig = self.gs.dual_project(self.gs.zone(self.gs.zp1).original(p), self.gs.zone(self.gs.zp0).original(self.gs.rp0));
        let cur = self.gs.project(self.gs.zone(self.gs.zp1).point(p), self.gs.zone(self.gs.zp0).point(self.gs.rp0));
        if self.gs.ret.auto_flip && (orig ^ cvt) < 0 {
            cvt = -cvt;
        }
        // CVT cut-in: use the actual (original) distance instead of the CVT stem width when they
        // differ by more than the cut-in. On the subpixel axis the comparison is 16·(cvt − orig) and
        // native applies it before picking RoundToGridSP vs RoundOffSP, except that INSTCTRL 4
        // suppresses it for unrounded instructions. Off the subpixel axis it requires the round bit.
        // Native's MIRP handling.
        if self.gs.subpixel_round {
            if (self.gs.ret.instruct_control & 4 == 0 || op & 4 != 0)
                && (16i64 * cvt.wrapping_sub(orig) as i64).abs() > self.gs.ret.control_value_cutin as i64
            {
                cvt = orig;
            }
        } else if op & 4 != 0 && self.gs.zp0 == self.gs.zp1 && (cvt.wrapping_sub(orig)).abs() > self.gs.ret.control_value_cutin {
            cvt = orig;
        }
        let mut distance = if op & 4 != 0 { self.gs.round_dist(cvt) } else { cvt };
        if op & 8 != 0 {
            let min = self.gs.min_dist();
            if orig >= 0 {
                if distance < min {
                    distance = min;
                }
            } else if distance > -min {
                distance = -min;
            }
        }
        self.gs.move_point(self.gs.zp1, p, distance.wrapping_sub(cur));
        self.gdi_link_mirp((op & 3) as i32, p, self.gs.rp0);
        self.gs.rp1 = self.gs.rp0;
        if op & 16 != 0 {
            self.gs.rp0 = p;
        }
        self.gs.rp2 = p;
        Ok(())
    }
    fn apply_single_width(&self, mut distance: i32) -> i32 {
        let (cutin, value) = (self.gs.ret.single_width_cutin, self.gs.ret.single_width);
        if cutin > 0 && distance < value + cutin && distance > value - cutin {
            distance = if distance >= 0 { value } else { -value };
        }
        distance
    }
    fn op_alignrp(&mut self) -> R {
        let count = self.gs.loop_counter;
        self.gs.loop_counter = 1;
        for _ in 0..count {
            let p = self.pop_usize()?;
            let d = self.gs.project(self.gs.zone(self.gs.zp1).point(p), self.gs.zone(self.gs.zp0).point(self.gs.rp0));
            self.gs.move_point(self.gs.zp1, p, -d);
            self.gdi_link_mdrp(p, self.gs.rp0); // ALIGNRP's link block is MDRP's (distance/inter-align link)
        }
        Ok(())
    }
    fn op_alignpts(&mut self) -> R {
        let p2 = self.pop_usize()?;
        let p1 = self.pop_usize()?;
        let d = self.gs.project(self.gs.zone(self.gs.zp0).point(p2), self.gs.zone(self.gs.zp1).point(p1)) / 2;
        self.gs.move_point(self.gs.zp1, p1, d);
        self.gs.move_point(self.gs.zp0, p2, -d);
        Ok(())
    }
    fn op_ip(&mut self) -> R {
        let count = self.gs.loop_counter;
        self.gs.loop_counter = 1;
        if !self.gs.in_bounds(&[(self.gs.zp0, self.gs.rp1), (self.gs.zp1, self.gs.rp2)]) {
            return Ok(());
        }
        let twilight = self.gs.zp0.is_twilight() || self.gs.zp1.is_twilight() || self.gs.zp2.is_twilight();
        let orus_base = if twilight {
            self.gs.zone(self.gs.zp0).original(self.gs.rp1)
        } else {
            self.gs.zone(self.gs.zp0).unscaled(self.gs.rp1)
        };
        let cur_base = self.gs.zone(self.gs.zp0).point(self.gs.rp1);
        let old_range = if twilight {
            self.gs.dual_project(self.gs.zone(self.gs.zp1).original(self.gs.rp2), orus_base)
        } else {
            self.gs.dual_project(self.gs.zone(self.gs.zp1).unscaled(self.gs.rp2), orus_base)
        };
        let cur_range = self.gs.project(self.gs.zone(self.gs.zp1).point(self.gs.rp2), cur_base);
        for _ in 0..count {
            let p = self.pop_usize()?;
            if !self.gs.in_bounds(&[(self.gs.zp2, p)]) {
                continue;
            }
            let orig = if twilight {
                self.gs.dual_project(self.gs.zone(self.gs.zp2).original(p), orus_base)
            } else {
                self.gs.dual_project(self.gs.zone(self.gs.zp2).unscaled(p), orus_base)
            };
            let cur = self.gs.project(self.gs.zone(self.gs.zp2).point(p), cur_base);
            let new = if orig != 0 {
                if old_range != 0 {
                    math::mul_div(orig, cur_range, old_range)
                } else {
                    orig
                }
            } else {
                0
            };
            self.gs.move_point(self.gs.zp2, p, new.wrapping_sub(cur));
            self.gdi_link_ip(p);
        }
        Ok(())
    }
    fn op_iup(&mut self, op: u8) -> R {
        let axis = if op & 1 != 0 { CoordAxis::X } else { CoordAxis::Y };
        // GDI-classic render pass: native's IUP handler runs the phase-control pass at its top on the subpixel
        // IUP axis (gate `g363==2 && (g448&3)==3 && (g448&4)!=(op&1) && !elem+96`) — the x-overscale
        // squeeze happens here, *before* the interpolation, not as a separate post-pass (hint-13). For
        // horizontal subpixel that is `IUP[x]`; it squeezes the touched stem x by the overscale and the
        // interpolation then carries the untouched points along.
        let g = &self.gs.ret.getinfo;
        if axis == CoordAxis::X
            && self.gs.gdi_overscale
            && g.subpixel_hinting
            && g.compatible_width
            && !g.vertical_lcd
            && !self.gs.phase_done
        {
            self.phase_control();
        }
        self.gs.zone_mut(ZonePointer::Glyph).iup(axis);
        // Native's IUP handler tail sets the touch flags `((op&1)==0)+1` — bit 0 after IUP[x], bit 1
        // after IUP[y]. Marks completed axes for SHPIX/DELTAP suppression.
        self.iup_axes |= if axis == CoordAxis::X { 1 } else { 2 };
        Ok(())
    }

    /// Native's phase-control pass on the glyph zone, once per glyph program: the GDI-classic x-overscale
    /// squeeze. Native runs it at the top of the subpixel `IUP[x]`, or — when the program never reaches one
    /// — at the end of the glyph program (GDI-classic render pass only), see [`Self::finish_glyph`].
    fn phase_control(&mut self) {
        self.gs.phase_done = true;
        let z = &self.gs.zones[1];
        let last = *z.contours.last().unwrap_or(&0) as i32;
        let mut x: Vec<i32> = z.cur.iter().map(|p| p.x).collect();
        let org_x: Vec<i32> = z.original.iter().map(|p| p.x).collect();
        overscale::execute_phase_control(&mut x, &org_x, &mut self.gs.links, self.gs.overscale, last);
        for (p, &nx) in self.gs.zones[1].cur.iter_mut().zip(&x) {
            p.x = nx;
        }
    }

    /// End of the glyph program (native's interpreter tail): a GDI-classic render pass whose program had
    /// no subpixel `IUP[x]` (e.g. Segoe UI Bold 'A') still gets the phase-control squeeze here.
    // ponytail: native SHC also runs it early (reference point in the glyph zone) and adds the
    // reference's phase shift to the SHC amount; port that when a seed needs it.
    pub(super) fn finish_glyph(&mut self) {
        let g = &self.gs.ret.getinfo;
        if self.gs.gdi_overscale && g.subpixel_hinting && g.compatible_width && !self.gs.phase_done {
            self.phase_control();
        }
    }

    /// GDI-classic x-overscale link building (native's GDI-classic link-build gate): active only
    /// in the glyph program, GDI-classic mode, with a projection other than the exact positive
    /// non-subpixel axis and the moved point in the glyph zone. A composite's own program builds none, so its
    /// phase pass leaves the assembled components where their own passes put them (VM phase dump, Arial U+00D1
    /// at ppem 14: the composite element's link memory is empty; ours squeezed two points 42/64 px left).
    fn gdi_link_active(&self, zp: ZonePointer) -> bool {
        // Native's SPVTL handler, the diagonal-projection slot: diagonal projections also build
        // links; only the exact positive non-subpixel axis disables the gate.
        let g = &self.gs.ret.getinfo;
        let non_subpixel = if g.vertical_lcd { Point::new(0x4000, 0) } else { Point::new(0, 0x4000) };
        self.gs.gdi_overscale && self.gs.is_glyph && !self.gs.is_composite && g.subpixel_hinting
            && self.gs.proj != non_subpixel && !zp.is_twilight()
    }
    /// MIRP/MSIRP link (native's inline distance link with the link-colour check).
    /// `default_color` is MIRP's `op&3` or MSIRP's constant 1 (the colour used for a cross-contour /
    /// non-adjacent pair).
    fn gdi_link_mirp(&mut self, default_color: i32, p: usize, rp0: usize) {
        if !self.gdi_link_active(self.gs.zp1) || self.gs.zp0.is_twilight() {
            return;
        }
        let color = overscale::double_check_link_color(&self.gs.zones[1], &self.gs.fill_bits, rp0 as i32, p as i32, default_color);
        overscale::add_distance(&self.gs.zones[1], &mut self.gs.links, rp0 as i32, p as i32, color);
    }
    /// MDRP link: inter-align → two-reference proportion link, else one-reference distance link.
    fn gdi_link_mdrp(&mut self, p: usize, rp0: usize) {
        if !self.gdi_link_active(self.gs.zp1) || self.gs.zp0.is_twilight() {
            return;
        }
        let (r1, r2) = self.gs.proj_ref;
        let aligned = r1 != -1 && r2 != -1 && overscale::inter_align(&self.gs.zones[1], r1, p as i32, r2);
        if aligned {
            overscale::add_proportion(&mut self.gs.links, r1, p as i32, r2);
        } else {
            overscale::add_distance(&self.gs.zones[1], &mut self.gs.links, rp0 as i32, p as i32, 3);
        }
    }
    /// IP link (native's `AddProportion` between the two interpolation references rp1/rp2).
    fn gdi_link_ip(&mut self, p: usize) {
        if !self.gdi_link_active(self.gs.zp2) || self.gs.zp0.is_twilight() || self.gs.zp1.is_twilight() {
            return;
        }
        overscale::add_proportion(&mut self.gs.links, self.gs.rp1 as i32, p as i32, self.gs.rp2 as i32);
    }
    /// GDI-classic x-overscale link built by SHP (native's common SHP handler, the inline block under
    /// native's GDI-classic gate): a single-parent distance-style link from the shifted point to
    /// the SHP reference (rp1 for SHP[1], rp2 for SHP[0]), walking the reference's parent chain while the
    /// unscaled x is unchanged. This is how the diagonal glyphs that hint via MDAP/SHP/MIAP (no MIRP/MDRP)
    /// — e.g. Segoe 'x' — tie their points to the advance phantom so the post-pass squeezes them.
    fn gdi_link_shp(&mut self, reference: usize, p: usize) {
        if !self.gdi_link_active(self.gs.zp2) {
            return;
        }
        overscale::add_distance(&self.gs.zones[1], &mut self.gs.links, reference as i32, p as i32, 0);
    }
    fn op_isect(&mut self) -> R {
        let b1 = self.pop_usize()?;
        let b0 = self.pop_usize()?;
        let a1 = self.pop_usize()?;
        let a0 = self.pop_usize()?;
        let p = self.pop_usize()?;
        let za = self.gs.zone(self.gs.zp1);
        let (pa0, pa1) = (za.point(a0), za.point(a1));
        let zb = self.gs.zone(self.gs.zp0);
        let (pb0, pb1) = (zb.point(b0), zb.point(b1));
        let dbx = pb1.x.wrapping_sub(pb0.x);
        let dby = pb1.y.wrapping_sub(pb0.y);
        let dax = pa1.x.wrapping_sub(pa0.x);
        let day = pa1.y.wrapping_sub(pa0.y);
        use math::mul_div;
        // Native's ISECT handler: phase-control links follow the edge with the stronger
        // x-intersection contribution, independent of the projection-vector gate.
        if self.gs.gdi_overscale && self.gs.is_glyph && self.gs.ret.getinfo.compatible_width
            && !self.gs.zp2.is_twilight()
        {
            let (r0, r1) = if mul_div(day, dbx, 0x40).wrapping_abs() > mul_div(dax, dby, 0x40).wrapping_abs() {
                (a0, a1)
            } else { (b0, b1) };
            overscale::add_proportion(&mut self.gs.links, r0 as i32, p as i32, r1 as i32);
        }
        // Native ISECT: walk line a to its crossing with line b, the crossing parameter t = num/den
        // formed through b's dominant axis, every quotient rounded half away from zero (`CompDiv`); an
        // axis-parallel b takes its direct form, and parallel lines (den 0) average the four endpoints.
        let cd = |d: i32, n: i64| -> i32 {
            if d == 0 {
                return (0x7FFF_FFFFi64 + (n < 0) as i64) as i32;
            }
            let h = (d / 2) as i64;
            ((if (d < 0) == (n < 0) { n + h } else { n - h }) / d as i64) as i32
        };
        let (xa0, ya0, xb0, yb0) = (pa0.x, pa0.y, pb0.x, pb0.y);
        let solve = |num: i32, den: i32| {
            if den != 0 {
                Point::new(xa0.wrapping_add(cd(den, num as i64 * dax as i64)), ya0.wrapping_add(cd(den, num as i64 * day as i64)))
            } else {
                Point::new(
                    xa0.wrapping_add(xb0).wrapping_add(dbx >> 1).wrapping_add(dax >> 1) >> 1,
                    ya0.wrapping_add(dby >> 1).wrapping_add(yb0).wrapping_add(day >> 1) >> 1,
                )
            }
        };
        let np = if dby != 0 {
            if dbx != 0 {
                let dy0 = ya0.wrapping_sub(yb0);
                if dbx.wrapping_abs() >= dby.wrapping_abs() {
                    solve(dy0.wrapping_sub(cd(dbx, dby as i64 * xa0.wrapping_sub(xb0) as i64)), cd(dbx, dby as i64 * dax as i64).wrapping_sub(day))
                } else {
                    solve(cd(dby, dbx as i64 * dy0 as i64).wrapping_add(xb0).wrapping_sub(xa0), dax.wrapping_sub(cd(dby, dbx as i64 * day as i64)))
                }
            } else if day != 0 {
                solve(xa0.wrapping_sub(xb0), dax.wrapping_neg())
            } else {
                Point::new(xb0, ya0)
            }
        } else if dax != 0 {
            solve(ya0.wrapping_sub(yb0), day.wrapping_neg())
        } else {
            Point::new(xa0, yb0)
        };
        if let Some(pt) = self.gs.zone_mut(self.gs.zp2).cur.get_mut(p) {
            *pt = np;
        }
        self.gs.zone_mut(self.gs.zp2).touch(p, CoordAxis::Both);
        Ok(())
    }

    // --- deltas --------------------------------------------------------------

    fn op_deltap(&mut self, set: u32) -> R {
        let ppem = self.proj_ppem() as u32;
        let point_count = self.gs.zone(self.gs.zp0).len();
        let mut n = self.pop()?.max(0) as usize;
        n = n.min(self.stack.len() / 2);
        let bias = set * 16 + self.gs.ret.delta_base as u32;
        let shift = self.gs.ret.delta_shift as i32;
        for _ in 0..n {
            let point = self.pop_usize()?;
            let mut b = self.pop()?;
            if point >= point_count {
                continue;
            }
            let c = ((b as u32 & 0xF0) >> 4) + bias;
            if ppem == c {
                b = (b & 0xF) - 8;
                if b >= 0 {
                    b += 1;
                }
                b *= 1 << (6 - shift);
                if self.deltap_allowed(point) {
                    self.gs.move_point(self.gs.zp0, point, b);
                }
            }
        }
        Ok(())
    }

    /// Whether a DELTAP exception may actually move `point`, reproducing native's DELTA engine
    /// gate: outside subpixel mode (or with `INSTCTRL` bit 2 set) deltas
    /// move freely, but under subpixel positioning they move a point only *along the non-subpixel
    /// axis* and only when one of:
    ///   * the composite flag is set — the **composite** program (the composite inner gridfit sets it to 1;
    ///     the simple inner gridfit clears it to 0), where the delta applies unconditionally; or
    ///   * the point is **touched** on that axis **and** `IUP[]` has not yet run on it (the IUP-ran flag).
    /// So a post-`IUP[y]` Y-delta on a Y-touched point is *suppressed* — this is what keeps native from
    /// re-moving the 'e' bowl off-curve control of 'é' etc. after its IUP+SHPIX (textx-glyph-251/283/…).
    /// Horizontal (x) DELTAP in the smooth modes stays suppressed. DELTAC is never gated (CVT only).
    /// The axis test is on the **projection** vector (native's delta engine reads the
    /// projection vector, and SFVTCA writes the freedom vector), so a delta measured along y
    /// still moves along a slanted italic freedom vector (Arial Bold Italic 'g' @9: pt0 dy −64, dx −8).
    fn deltap_allowed(&self, point: usize) -> bool {
        const ONE: i32 = 0x4000; // 2.14 unit vector = 1.0
        let g = &self.gs.ret.getinfo;
        if !g.subpixel_hinting || (self.gs.ret.instruct_control & 4) != 0 {
            return true;
        }
        let (fx, fy) = (self.gs.proj.x, self.gs.proj.y);
        let composite = self.gs.is_composite; // the composite flag — set for the composite's own program
        let zone = self.gs.zone(self.gs.zp0);
        if g.vertical_lcd {
            fx == ONE && fy == 0 && (composite || (zone.is_touched(point, CoordAxis::X) && (self.iup_axes & 1) == 0))
        } else {
            fy == ONE && fx == 0 && (composite || (zone.is_touched(point, CoordAxis::Y) && (self.iup_axes & 2) == 0))
        }
    }

    fn op_deltac(&mut self, set: u32) -> R {
        let ppem = self.proj_ppem() as u32;
        let mut n = self.pop()?.max(0) as usize;
        n = n.min(self.stack.len() / 2);
        let bias = set * 16 + self.gs.ret.delta_base as u32;
        let shift = self.gs.ret.delta_shift as i32;
        for _ in 0..n {
            let ci = self.pop_usize()?;
            let mut b = self.pop()?;
            let c = ((b as u32 & 0xF0) >> 4) + bias;
            if ppem == c {
                b = (b & 0xF) - 8;
                if b >= 0 {
                    b += 1;
                }
                b *= 1 << (6 - shift);
                // Changing a CVT entry: a non-square instance unstretches the delta.
                if !self.gs.ret.square() {
                    b = math::fixdiv(b, self.cvt_scale());
                }
                if let Some(cv) = self.cvt.get_mut(ci) {
                    *cv = cv.wrapping_add(b);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Definition, Engine};
    use super::super::state::{GraphicsState, Retained, Zone};

    fn empty_engine<'a>(fpgm: &'a [u8], max_func: usize) -> Engine<'a> {
        let gs = GraphicsState::new(Retained::default(), Zone::default(), Zone::default());
        Engine::new(gs, Vec::new(), vec![0; 4], vec![Definition::default(); max_func], Vec::new(), 128, fpgm, &[])
    }

    #[test]
    fn compatible_diagonal_projection_builds_links() {
        use super::super::state::{Point, ZonePointer};
        let mut e = empty_engine(&[], 0);
        e.gs.gdi_overscale = true;
        e.gs.is_glyph = true;
        e.gs.ret.getinfo.subpixel_hinting = true;
        e.gs.proj = Point::new(14033, -8457); // captured Arial 4@13 SPVTL
        assert!(e.gdi_link_active(ZonePointer::Glyph));
        e.gs.proj = Point::new(0, 0x4000);
        assert!(!e.gdi_link_active(ZonePointer::Glyph));
    }

    #[test]
    fn compatible_intersection_preserves_phase_dependencies() {
        use super::super::state::Point;
        let mut e = empty_engine(&[], 0);
        e.gs.gdi_overscale = true;
        e.gs.is_glyph = true;
        e.gs.ret.getinfo.compatible_width = true;
        let points = vec![Point::new(0, 0), Point::new(0, 64), Point::new(-64, 32), Point::new(64, 32), Point::default()];
        e.gs.zones[1] = Zone { unscaled: points.clone(), original: points.clone(), cur: points, flags: vec![0; 5], contours: vec![4] };
        e.gs.links = vec![super::super::state::LinkRec::default(); 5];
        for p in [4, 0, 1, 2, 3] { e.push(p).unwrap(); }
        e.op_isect().unwrap();
        assert_eq!(e.gs.zones[1].cur[4], Point::new(0, 32));
        assert_eq!((e.gs.links[4].p1, e.gs.links[4].p2), (0, 1));
    }

    #[test]
    fn compatible_width_snap_function_preserves_untouched_points() {
        use super::super::state::Point;
        // Native's CALL/SHP handlers: compatible-width rendering still suppresses
        // an MPPEM-gated snap unless the non-subpixel axis is already touched.
        for (axis, flags, control, iup, expected) in [(0, 0, 0, 0, 0), (0, 4, 0, 0, 40), (1, 2, 0, 0, 0), (0, 0, 4, 0, 40), (0, 4, 0, 2, 0)] {
            let mut e = empty_engine(&[], 0);
            e.gs.ret.getinfo.subpixel_hinting = true;
            e.gs.ret.getinfo.compatible_width = true;
            e.gs.ret.instruct_control = control;
            e.gs.zones[1] = Zone { unscaled: vec![Point::new(0, 0)], original: vec![Point::new(0, 0)],
                cur: vec![Point::new(0, 0)], flags: vec![flags], contours: vec![0] };
            e.op_svtca(axis).unwrap();
            e.neutral_depth = 1;
            e.iup_axes = iup;
            e.push(0).unwrap();
            e.push(40).unwrap();
            e.op_shpix().unwrap();
            assert_eq!(e.gs.zones[1].cur[0].x + e.gs.zones[1].cur[0].y, expected);
        }
    }

    #[test]
    fn unrounded_subpixel_mirp_respects_instruction_control() {
        use super::super::state::Point;
        // INSTCTRL 4 bypasses CVT cut-in for unrounded subpixel instructions only.
        for (control, opcode, expected) in [(0, 0xe0, 2156), (4, 0xe0, 512), (4, 0xe4, 2156)] {
            let mut e = empty_engine(&[], 0);
            e.gs.ret.instruct_control = control;
            e.gs.subpixel_round = true;
            let points = vec![Point::new(0, 0), Point::new(2156, 0)];
            e.gs.zones[1] = Zone { unscaled: points.clone(), original: points.clone(), cur: points, flags: vec![0; 2], contours: vec![1] };
            e.cvt = vec![512];
            e.push(1).unwrap();
            e.push(0).unwrap();
            e.op_mirp(opcode).unwrap();
            assert_eq!(e.gs.zones[1].cur[1].x, expected, "control={control}, opcode={opcode:x}");
        }
    }

    #[test]
    fn recognized_slope_call_temporarily_disables_cutin() {
        let fpgm = [0xb0, 58, 0x2c, 0x20, 0x20, 0xb0, 1, 0x60, 0x46, 0xb0, 0x40, 0x23, 0x42, 0x2d];
        for (subpixel, control, expected) in [(true, 0, i32::MAX), (false, 0, 144), (true, 4, 144)] {
            let mut e = empty_engine(&fpgm, 59);
            e.run_program(0, &[]).unwrap();
            assert!(e.gs.ret.slope_function);
            e.gs.ret.control_value_cutin = 144;
            e.gs.ret.getinfo.subpixel_hinting = subpixel;
            e.gs.ret.instruct_control = control;
            e.push(58).unwrap();
            e.op_call().unwrap();
            assert_eq!(e.gs.ret.control_value_cutin, expected);
            e.leave().unwrap();
            assert_eq!(e.gs.ret.control_value_cutin, 144);
        }
    }

    #[test]
    fn wcvtf_scales_negative_half_toward_positive() {
        let mut e = empty_engine(&[], 0);
        e.gs.ret.scale = 40960;
        e.cvt = vec![0];
        e.push(0).unwrap();
        e.push(-4).unwrap();
        e.op_wcvtf().unwrap();
        assert_eq!(e.cvt[0], -2);
    }

    #[test]
    fn sfvtpv_sets_exact_unit_projection_factor() {
        let mut e = empty_engine(&[], 0);
        e.gs.proj = super::super::state::Point::new(15958, -3707);
        e.op_sfvtpv().unwrap();
        assert_eq!(e.gs.free, e.gs.proj);
        assert_eq!(e.gs.fdotp, 0x4000);
    }

    #[test]
    fn perpendicular_zero_length_line_rotates_native_default_vector() {
        use super::super::state::Point;
        assert_eq!(super::line_vector(Point::default(), Point::default(), false), Point::new(0, 0x4000));
    }

    #[test]
    fn fdef_call_add_store() {
        // Define function 1 = { push 0,2,3; ADD; WS }, then call it; storage[0] == 5.
        #[rustfmt::skip]
        let fpgm = [
            0xB0, 0x01,                   // PUSHB[0] 1  (function number)
            0x2C,                         // FDEF
              0x40, 0x03, 0x00, 0x02, 0x03, // NPUSHB 3: 0,2,3
              0x60,                       // ADD -> 0,5
              0x42,                       // WS  -> storage[0] = 5
            0x2D,                         // ENDF
            0xB0, 0x01,                   // PUSHB[0] 1
            0x2B,                         // CALL
        ];
        let mut e = empty_engine(&fpgm, 2);
        e.run_program(0, &[]).unwrap();
        assert_eq!(e.storage[0], 5);
        assert!(e.functions[1].active);
    }

    #[test]
    fn if_else_skips_branch() {
        // push 0; IF { push 7; WS@1 } ELSE { push 9; WS@1 } EIF  -> storage[1] = 9
        #[rustfmt::skip]
        let fpgm = [
            0xB0, 0x00,             // PUSHB[0] 0  (false)
            0x58,                   // IF
              0xB1, 0x01, 0x07,     // PUSHB[1] 1 7   (loc 1, val 7)
              0x42,                 // WS
            0x1B,                   // ELSE
              0xB1, 0x01, 0x09,     // PUSHB[1] 1 9
              0x42,                 // WS
            0x59,                   // EIF
        ];
        let mut e = empty_engine(&fpgm, 1);
        e.run_program(0, &[]).unwrap();
        assert_eq!(e.storage[1], 9);
    }
}

/// Build a 2.14 unit vector from a line through two points (`SxVTL`/`SDPVTL`).
/// `parallel` keeps the line direction, otherwise rotates 90°.
fn line_vector(p1: Point, p2: Point, parallel: bool) -> Point {
    // Native's SPVTL / SDPVTL normalize first, including the zero-vector default,
    // then rotate the rounded 2.14 vector for the perpendicular variant.
    let (x, y) = math::normalize14(p1.x.wrapping_sub(p2.x), p1.y.wrapping_sub(p2.y));
    if parallel { Point::new(x, y) } else { Point::new(-y, x) }
}
