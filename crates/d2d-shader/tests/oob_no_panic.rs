//! Out-of-bounds register indexing must never trap. A plugin shader is third-party input, and in
//! wasm a panic traps the whole page's renderer; the D3D11 rule (which WARP follows) is that an
//! out-of-bounds register read returns 0 and an out-of-bounds write is discarded, not a crash.
//!
//! This test runs every pixel shader in the corpora through `exec::run` over a battery of inputs
//! chosen to drive register-relative indices out of range — NaN/Inf/denormal/extreme/random bit
//! patterns fed into operands that a shader may use as an index — and asserts that NONE of them
//! panics. It also asserts the rule directly on a hand-built program that reads and writes an
//! indexable temp out of bounds.

use d2d_shader::dxbc::Shader;
use d2d_shader::exec::{self, Env, PixelInputs, Sampling};
use d2d_shader::sm4::{Index, Inst, Opcode, Operand, Program, Sel};

struct DetEnv;
fn mix(seed: u64) -> u32 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) as u32
}
impl Env for DetEnv {
    fn sample(&self, r: u32, s: u32, c: [f32; 4], _m: Sampling) -> [f32; 4] {
        let base = (r as u64) << 40 ^ (s as u64) << 32 ^ c[0].to_bits() as u64;
        core::array::from_fn(|i| f32::from_bits(mix(base ^ (i as u64) << 56)))
    }
    fn load(&self, r: u32, c: [i32; 4]) -> [u32; 4] {
        core::array::from_fn(|i| mix((r as u64) << 40 ^ c[0] as u64 ^ (i as u64) << 48))
    }
    fn resinfo(&self, r: u32) -> [u32; 4] {
        [256, 256, 1, 1].map(|d: u32| d ^ (r & 1))
    }
}

const SPECIAL: &[u32] = &[
    0, 0x8000_0000, 0x3f80_0000, 0xbf80_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0x7f80_0001,
    0x0000_0001, 0x007f_ffff, 0x0080_0000, 0x7f7f_ffff, 0xffff_ffff, 0x0000_000a, 0x0000_0003,
    // large values that make especially nasty register-relative indices:
    0x7fff_ffff, 0x0001_0000, 0x000f_4240,
];
fn bank(seed: u64) -> [u32; 4] {
    core::array::from_fn(|i| {
        let s = mix(seed ^ (i as u64) << 20);
        if s & 1 == 0 { SPECIAL[(s as usize >> 1) % SPECIAL.len()] } else { mix(seed.wrapping_mul(0x100_0001) ^ i as u64) }
    })
}
fn input_battery() -> Vec<(Vec<[u32; 4]>, Vec<u8>)> {
    (0..24u64)
        .map(|case| {
            let inputs: Vec<[u32; 4]> = (0..8).map(|r| bank(case << 8 ^ r)).collect();
            let mut cb = Vec::with_capacity(512);
            for e in 0..32u64 {
                for c in bank(case << 16 ^ e) {
                    cb.extend_from_slice(&c.to_le_bytes());
                }
            }
            (inputs, cb)
        })
        .collect()
}
fn shader_dirs() -> Vec<std::path::PathBuf> {
    let mut dirs = Vec::new();
    if let Some(suite) = d2d_reference::suite_dir() {
        dirs.push(suite.join("scenes"));
    }
    if let Some(root) = d2d_reference::main_checkout() {
        dirs.push(root.join("ref/windows/shaders"));
    }
    if let Ok(extra) = std::env::var("SHADER_DIFF_DIR") {
        for p in extra.split([':', ',']).filter(|s| !s.is_empty()) {
            dirs.push(std::path::PathBuf::from(p));
        }
    }
    dirs.retain(|d| d.is_dir());
    dirs.sort();
    dirs.dedup();
    dirs
}

#[test]
fn corpus_never_traps_on_out_of_range_index() {
    let dirs = shader_dirs();
    if dirs.iter().all(|d| !d.is_dir()) {
        eprintln!("oob_no_panic: no shader corpus found; skipping");
        return;
    }
    let battery = input_battery();
    let env = DetEnv;
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let (mut ran, mut cases, mut panics) = (0usize, 0usize, 0usize);
    for dir in &dirs {
        let Ok(entries) = std::fs::read_dir(dir) else { continue };
        for e in entries {
            let path = e.unwrap().path();
            if path.extension().and_then(|s| s.to_str()) != Some("dxbc") {
                continue;
            }
            let Ok(shader) = Shader::parse(&std::fs::read(&path).unwrap()) else { continue };
            if shader.program.program_type != 0 || shader.unsupported_op().is_some() {
                continue;
            }
            ran += 1;
            for (inputs, cb) in &battery {
                let px = PixelInputs { inputs, cbuffer: cb };
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| exec::run(&shader, &env, &px))).is_err() {
                    panics += 1;
                    if panics <= 3 {
                        eprintln!("  TRAP in {}", path.file_stem().unwrap().to_string_lossy());
                    }
                }
                cases += 1;
            }
        }
    }
    std::panic::set_hook(prev);
    eprintln!("oob_no_panic: {ran} pixel shaders × {} inputs = {cases} cases, {panics} traps", battery.len());
    assert_eq!(panics, 0, "out-of-range register indexing must never trap, but {panics} cases did");
    assert!(ran > 0, "no runnable pixel shaders found");
}

/// Assert the OOB guard rules directly on a hand-built program that reads and writes an indexable
/// temp `x0` (declared with 2 rows) at a constant out-of-range row. WARP clamps the index to a guard
/// slot past the declared rows, so a fresh out-of-range read is 0, an out-of-range write lands in the
/// guard (not a valid row), and an out-of-range write followed by an out-of-range read returns the
/// written value.
#[test]
fn indexable_temp_out_of_bounds_clamps_to_zero_guard() {
    let o0 = Operand { ty: 2, num_components: 4, sel: Sel::Mask(0xf), modifier: 0, indices: vec![Index::Imm(0)], imm: vec![] };
    let x0_src = |row: u64| Operand { ty: 3, num_components: 4, sel: Sel::Swizzle([0, 1, 2, 3]), modifier: 0, indices: vec![Index::Imm(0), Index::Imm(row)], imm: vec![] };
    let x0_dst = |row: u64| Operand { ty: 3, num_components: 4, sel: Sel::Mask(0xf), modifier: 0, indices: vec![Index::Imm(0), Index::Imm(row)], imm: vec![] };
    let imm = |v: u32| Operand { ty: 4, num_components: 4, sel: Sel::None, modifier: 0, indices: vec![], imm: vec![v; 4] };
    let mov = |dst: Operand, src: Operand| Inst { op: Opcode::Mov, saturate: false, test_nz: false, operands: vec![dst, src], decl_count: 0, aux: 0 };
    let prog = |insts: Vec<Inst>| Program {
        program_type: 0, major: 5, minor: 0, insts, temps: 0, indexable_temps: vec![(0, 2)],
        thread_group: [0; 3], uavs: vec![], tgsm: vec![], resources: vec![], icb: vec![],
    };
    let shader = |p: Program| Shader { program: p, inputs: vec![], outputs: vec![], input_simple: vec![] };
    let px = PixelInputs { inputs: &[], cbuffer: &[] };

    // Fresh out-of-range read x0[9] → the zero-initialised guard → 0.
    let got = exec::run(&shader(prog(vec![mov(o0.clone(), x0_src(9))])), &DetEnv, &px);
    assert_eq!(got.map(|f| f.to_bits()), [0; 4], "fresh OOB indexable-temp read is 0");

    // Out-of-range write to x0[9] lands in the guard, not a valid row, so x0[0] stays 0.
    let got = exec::run(&shader(prog(vec![mov(x0_dst(9), imm(0x3f80_0000)), mov(o0.clone(), x0_src(0))])), &DetEnv, &px);
    assert_eq!(got.map(|f| f.to_bits()), [0; 4], "OOB write does not touch a valid row");

    // Out-of-range write to x0[9] then out-of-range read x0[8]: both clamp to the guard, so the read
    // returns the written value — the observable fingerprint WARP exposes.
    let got = exec::run(&shader(prog(vec![mov(x0_dst(9), imm(0x3f80_0000)), mov(o0, x0_src(8))])), &DetEnv, &px);
    assert_eq!(got.map(|f| f.to_bits()), [0x3f80_0000; 4], "OOB write then OOB read returns the written value (guard)");
}

/// A NEGATIVE register-relative row index (e.g. `-1`) must clamp to the guard slot as an unsigned
/// compare — the same place a large positive index lands — not fall back to row 0. WARP compares the
/// index to the register size as an unsigned value, so `-1` reads as `0xFFFFFFFF`, far above the size,
/// and clamps to the guard.
#[test]
fn negative_relative_index_clamps_to_guard_not_row_zero() {
    let o0 = Operand { ty: 2, num_components: 4, sel: Sel::Mask(0xf), modifier: 0, indices: vec![Index::Imm(0)], imm: vec![] };
    // r0 temp, read as `.x` (Select1) — the relative index source.
    let r0_x = || Operand { ty: 0, num_components: 4, sel: Sel::Select1(0), modifier: 0, indices: vec![Index::Imm(0)], imm: vec![] };
    let r0_dst = || Operand { ty: 0, num_components: 4, sel: Sel::Mask(0xf), modifier: 0, indices: vec![Index::Imm(0)], imm: vec![] };
    // x0[r0.x]: ty 3, bank 0, row = register-relative on r0.x.
    let x0_rel_src = || Operand { ty: 3, num_components: 4, sel: Sel::Swizzle([0, 1, 2, 3]), modifier: 0, indices: vec![Index::Imm(0), Index::Rel(Box::new(r0_x()))], imm: vec![] };
    let x0_rel_dst = || Operand { ty: 3, num_components: 4, sel: Sel::Mask(0xf), modifier: 0, indices: vec![Index::Imm(0), Index::Rel(Box::new(r0_x()))], imm: vec![] };
    let x0_0_dst = Operand { ty: 3, num_components: 4, sel: Sel::Mask(0xf), modifier: 0, indices: vec![Index::Imm(0), Index::Imm(0)], imm: vec![] };
    let x0_0_src = Operand { ty: 3, num_components: 4, sel: Sel::Swizzle([0, 1, 2, 3]), modifier: 0, indices: vec![Index::Imm(0), Index::Imm(0)], imm: vec![] };
    let imm = |v: u32| Operand { ty: 4, num_components: 4, sel: Sel::None, modifier: 0, indices: vec![], imm: vec![v; 4] };
    let mov = |dst: Operand, src: Operand| Inst { op: Opcode::Mov, saturate: false, test_nz: false, operands: vec![dst, src], decl_count: 0, aux: 0 };
    let prog = |insts: Vec<Inst>| Program {
        program_type: 0, major: 5, minor: 0, insts, temps: 1, indexable_temps: vec![(0, 2)],
        thread_group: [0; 3], uavs: vec![], tgsm: vec![], resources: vec![], icb: vec![],
    };
    let shader = |p: Program| Shader { program: p, inputs: vec![], outputs: vec![], input_simple: vec![] };
    let px = PixelInputs { inputs: &[], cbuffer: &[] };

    // r0.x = -1; x0[0] = 2.0 (row-0 marker); x0[-1] = 1.0 (negative → guard); o0 = x0[0].
    // The negative write must NOT corrupt row 0, so o0 stays 2.0.
    let insts = vec![
        mov(r0_dst(), imm(0xffff_ffff)),
        mov(x0_0_dst, imm(0x4000_0000)),
        mov(x0_rel_dst(), imm(0x3f80_0000)),
        mov(o0.clone(), x0_0_src),
    ];
    let got = exec::run(&shader(prog(insts)), &DetEnv, &px);
    assert_eq!(got.map(|f| f.to_bits()), [0x4000_0000; 4], "a negative relative write must hit the guard, not row 0");

    // r0.x = -1; x0[-1] = 1.0 (write → guard); o0 = x0[-1] (read → guard): returns the written value.
    let insts = vec![
        mov(r0_dst(), imm(0xffff_ffff)),
        mov(x0_rel_dst(), imm(0x3f80_0000)),
        mov(o0, x0_rel_src()),
    ];
    let got = exec::run(&shader(prog(insts)), &DetEnv, &px);
    assert_eq!(got.map(|f| f.to_bits()), [0x3f80_0000; 4], "negative relative read clamps to the same guard slot");
}
