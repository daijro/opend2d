//! Differential test for the pre-decoded shader executor: `exec::run` (reference) vs `exec::run_pre`
//! (pre-decoded) must agree BIT-FOR-BIT on every pixel shader in the corpora, across a battery of
//! inputs chosen to hit the IEEE edge cases the pre-decode must not perturb — NaN (quiet, signalling,
//! both signs, payloads), ±Inf, ±0, denormals, the f32 extremes, and seeded-random bit patterns.
//!
//! This is the safety net the pre-decode pass is built behind: it fails the instant the lowered
//! executor diverges from the reference on any op, operand modifier, swizzle, write mask or control
//! path. Correctness is byte-exact — no FMA, no reassociation — so a single mismatched bit fails.
//!
//! Corpora: the conformance `scenes/*.dxbc` (via `d2d_reference::suite_dir`, which falls back to the
//! main checkout, so this runs from a worktree) plus any directory named in `SHADER_DIFF_DIR`.

use d2d_shader::dxbc::Shader;
use d2d_shader::exec::{self, Env, PixelInputs, Sampling};
use d2d_shader::sm4::{Index, Inst, Opcode, Operand, Program, Sel};

/// A deterministic environment: `sample`/`load`/`resinfo` return values derived only from their
/// arguments, so the reference and pre-decoded runs see identical texture reads (the differential is
/// about the executor, not the textures). Values are spread across the f32 range, including the odd
/// NaN, to exercise the sampling result paths too.
struct DetEnv;

fn mix(seed: u64) -> u32 {
    // SplitMix64-style avalanche → a well-spread u32 bit pattern.
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) as u32
}

impl Env for DetEnv {
    fn sample(&self, resource: u32, sampler: u32, coord: [f32; 4], _mode: Sampling) -> [f32; 4] {
        let base = (resource as u64) << 40 ^ (sampler as u64) << 32 ^ coord[0].to_bits() as u64 ^ (coord[1].to_bits() as u64) << 8;
        core::array::from_fn(|i| f32::from_bits(mix(base ^ (i as u64) << 56)))
    }
    fn load(&self, resource: u32, coord: [i32; 4]) -> [u32; 4] {
        let base = (resource as u64) << 40 ^ coord[0] as u64 ^ (coord[1] as u64) << 16;
        core::array::from_fn(|i| mix(base ^ (i as u64) << 48))
    }
    fn resinfo(&self, resource: u32) -> [u32; 4] {
        [256, 256, 1, 1].map(|d: u32| d ^ (resource & 1))
    }
    fn sample_count(&self, _resource: u32) -> u32 {
        1
    }
}

/// The IEEE-interesting f32 bit patterns a correct executor must carry through unperturbed.
const SPECIAL: &[u32] = &[
    0x0000_0000, // +0.0
    0x8000_0000, // -0.0
    0x3f80_0000, // 1.0
    0xbf80_0000, // -1.0
    0x4000_0000, // 2.0
    0x3f00_0000, // 0.5
    0x7f80_0000, // +Inf
    0xff80_0000, // -Inf
    0x7fc0_0000, // qNaN
    0xffc0_0000, // -qNaN
    0x7f80_0001, // sNaN
    0x7fbf_ffff, // sNaN, max payload
    0x0000_0001, // smallest +denormal
    0x8000_0001, // smallest -denormal
    0x007f_ffff, // largest +denormal
    0x0080_0000, // smallest +normal
    0x7f7f_ffff, // f32 max
    0xff7f_ffff, // -f32 max
    0x3f7f_ffff, // just below 1.0
    0x0000_000a, // small int 10
    0xffff_ffff, // all ones / -NaN / int -1
    0x0000_0003, // int 3 (loop counts, indices)
];

/// Build one input bank: four components, each a bit pattern selected from SPECIAL / random by `seed`.
fn bank(seed: u64) -> [u32; 4] {
    core::array::from_fn(|i| {
        let s = mix(seed ^ (i as u64) << 20);
        // Half the time a SPECIAL value, half a random one — so edge cases AND arbitrary bits appear.
        if s & 1 == 0 {
            SPECIAL[(s as usize >> 1) % SPECIAL.len()]
        } else {
            mix(seed.wrapping_mul(0x100_0001) ^ i as u64)
        }
    })
}

/// A battery of PixelInputs: several input registers and a constant buffer, filled from the special
/// and random patterns. Enough registers/cbuffer bytes to cover what the corpus shaders read.
fn input_battery() -> Vec<(Vec<[u32; 4]>, Vec<u8>)> {
    let mut out = Vec::new();
    for case in 0..24u64 {
        let inputs: Vec<[u32; 4]> = (0..8).map(|r| bank(case << 8 ^ r)).collect();
        // 512-byte cbuffer (32 float4 elements) of the same patterns.
        let mut cb = Vec::with_capacity(512);
        for e in 0..32u64 {
            for c in bank(case << 16 ^ e).iter() {
                cb.extend_from_slice(&c.to_le_bytes());
            }
        }
        out.push((inputs, cb));
    }
    out
}

/// Every shader-corpus directory to sweep. The committed baseline is the conformance `scenes`; the
/// Paint.NET plugin corpora (pdn-host's extracted builtin + plugin `ps_5_0` dumps, the V5 samples and
/// the BoltBait packs — all compiled through this same exec path) live under the reference tree
/// (`ref/windows/shaders`) and, for a worktree run, wherever `SHADER_DIFF_DIR` points (colon- or
/// comma-separated). Absent dirs are skipped, so a public clone with only the suite still runs.
fn shader_dirs() -> Vec<std::path::PathBuf> {
    use std::path::PathBuf;
    let mut dirs = Vec::new();
    if let Some(suite) = d2d_reference::suite_dir() {
        dirs.push(suite.join("scenes"));
    }
    // The extracted native/plugin shader corpus in the reference tree (pdn-host dumps).
    if let Some(root) = d2d_reference::main_checkout() {
        dirs.push(root.join("ref/windows/shaders"));
    }
    if let Ok(extra) = std::env::var("SHADER_DIFF_DIR") {
        for p in extra.split([':', ',']).filter(|s| !s.is_empty()) {
            dirs.push(PathBuf::from(p));
        }
    }
    dirs.retain(|d| d.is_dir());
    dirs.sort();
    dirs.dedup();
    dirs
}

/// A temp read before it is written must read the same value `TileExec` and the reference give — and
/// since the reference allocates a fresh, zero-initialised register file per pixel (`Lane::new`), the
/// reused register file must be reset to 0 each pixel, so a value a previous pixel wrote cannot leak
/// in. (Reading an uninitialised temp is undefined in the D3D11 model, so this exactness is the
/// reference's zero-init behaviour, which the passing WARP oracles are consistent with; the point of
/// the test is that `TileExec` reproduces it.) The shader `mov o0, r0; mov r0, 5.0` reads r0 into o0
/// *before* writing it, so a stale r0 from the previous pixel would surface as 5.0 instead of 0.
#[test]
fn tile_exec_resets_temp_read_before_write() {
    let o0 = Operand { ty: 2, num_components: 4, sel: Sel::Mask(0xf), modifier: 0, indices: vec![Index::Imm(0)], imm: vec![] };
    let r0_src = Operand { ty: 0, num_components: 4, sel: Sel::Swizzle([0, 1, 2, 3]), modifier: 0, indices: vec![Index::Imm(0)], imm: vec![] };
    let r0_dst = Operand { ty: 0, num_components: 4, sel: Sel::Mask(0xf), modifier: 0, indices: vec![Index::Imm(0)], imm: vec![] };
    let imm5 = Operand { ty: 4, num_components: 4, sel: Sel::None, modifier: 0, indices: vec![], imm: vec![0x4040_0000; 4] };
    let mov = |dst: Operand, src: Operand| Inst { op: Opcode::Mov, saturate: false, test_nz: false, operands: vec![dst, src], decl_count: 0, aux: 0 };
    let program = Program {
        program_type: 0, major: 5, minor: 0,
        insts: vec![mov(o0, r0_src), mov(r0_dst, imm5)],
        temps: 1, indexable_temps: vec![], thread_group: [0; 3], uavs: vec![], tgsm: vec![], resources: vec![], icb: vec![],
    };
    let shader = Shader { program, inputs: vec![], outputs: vec![], input_simple: vec![] };
    let env = DetEnv;
    let px = PixelInputs { inputs: &[], cbuffer: &[] };

    // Reference: fresh register file each pixel → r0 reads 0.
    assert_eq!(exec::run(&shader, &env, &px).map(|f| f.to_bits()), [0; 4]);

    // TileExec reuses the register file: pixel 1 writes r0 = 5.0, pixel 2 reads r0 before writing it;
    // the reset must give 0, not the stale 5.0.
    let pre = exec::predecode_shader(&shader);
    let cbs: [&[u8]; 1] = [&[]];
    let mut te = shader.tile_exec(&pre, &env, &cbs);
    assert_eq!(te.pixel(&[]).map(|f| f.to_bits()), [0; 4], "pixel 1 reads r0 == 0");
    assert_eq!(te.pixel(&[]).map(|f| f.to_bits()), [0; 4], "pixel 2 must reset r0 to 0, not reuse the written 5.0");
}

#[test]
fn run_pre_matches_run_bit_exact() {
    let dirs = shader_dirs();
    if dirs.iter().all(|d| !d.is_dir()) {
        eprintln!("predecode_diff: no shader corpus found (conformance suite absent); skipping");
        return;
    }
    let battery = input_battery();
    let env = DetEnv;
    let (mut ran, mut skipped, mut cases) = (0usize, 0usize, 0usize);
    // Out-of-range register indexing follows the guard rule and never traps, so run_pre must agree
    // with run on every case AND neither may panic. catch_unwind only turns a (bug) trap into a clean
    // failure message instead of a raw unwind.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let run_catch = |f: &dyn Fn() -> [f32; 4]| std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).ok();

    for dir in &dirs {
        let Ok(entries) = std::fs::read_dir(dir) else { continue };
        for e in entries {
            let path = e.unwrap().path();
            if path.extension().and_then(|s| s.to_str()) != Some("dxbc") {
                continue;
            }
            let name = path.file_stem().unwrap().to_string_lossy().into_owned();
            let bytes = std::fs::read(&path).unwrap();
            let Ok(shader) = Shader::parse(&bytes) else {
                skipped += 1;
                continue;
            };
            // Pixel shaders only (program_type 0); run/run_pre drive one pixel lane.
            if shader.program.program_type != 0 || shader.unsupported_op().is_some() {
                skipped += 1;
                continue;
            }
            ran += 1;
            for (inputs, cb) in &battery {
                let px = PixelInputs { inputs, cbuffer: cb };
                let a = run_catch(&|| exec::run(&shader, &env, &px));
                let b = run_catch(&|| exec::run_pre(&shader, &env, &px));
                // The tile executor runs the pre-decoded shader while reusing its register file; call
                // it twice so the second call also exercises the in-place register reset.
                let pre = exec::predecode_shader(&shader);
                let cbs: [&[u8]; 1] = [cb.as_slice()];
                let d = run_catch(&|| {
                    let mut te = shader.tile_exec(&pre, &env, &cbs);
                    te.pixel(inputs);
                    te.pixel(inputs)
                });
                // run_predecoded_reuse: per-call env, reused register file (the effect-graph path).
                // Call twice so the in-place reset is exercised.
                let e = run_catch(&|| {
                    let mut scratch = exec::ExecScratch::new(&shader);
                    exec::run_predecoded_reuse(&shader, &pre, &env, &px, &mut scratch);
                    exec::run_predecoded_reuse(&shader, &pre, &env, &px, &mut scratch)
                });
                // Out-of-range indexing never traps (the guard rule), so all must return — a panic in
                // any is a failure, not something to tolerate.
                let (Some(a), Some(b), Some(d), Some(e)) = (a, b, d, e) else {
                    std::panic::set_hook(prev_hook);
                    panic!("{name}: shader trapped on input (OOB indexing must not panic) — run={} run_pre={} tile={} reuse={} (inputs[0]={:08x?})", a.is_some(), b.is_some(), d.is_some(), e.is_some(), inputs[0]);
                };
                for c in 0..4 {
                    assert_eq!(
                        a[c].to_bits(),
                        b[c].to_bits(),
                        "{name}: channel {c} run vs run_pre: 0x{:08x} vs 0x{:08x} (inputs[0]={:08x?})",
                        a[c].to_bits(),
                        b[c].to_bits(),
                        inputs[0],
                    );
                    assert_eq!(
                        a[c].to_bits(),
                        d[c].to_bits(),
                        "{name}: channel {c} run vs tile_exec: 0x{:08x} vs 0x{:08x} (inputs[0]={:08x?})",
                        a[c].to_bits(),
                        d[c].to_bits(),
                        inputs[0],
                    );
                    assert_eq!(
                        a[c].to_bits(),
                        e[c].to_bits(),
                        "{name}: channel {c} run vs run_predecoded_reuse: 0x{:08x} vs 0x{:08x} (inputs[0]={:08x?})",
                        a[c].to_bits(),
                        e[c].to_bits(),
                        inputs[0],
                    );
                }
                cases += 1;
            }
        }
    }
    std::panic::set_hook(prev_hook);
    eprintln!("predecode_diff: {ran} pixel shaders × {} inputs → {cases} cases bit-exact, 0 traps; {skipped} skipped", battery.len());
    assert!(ran > 0, "no runnable pixel shaders found in corpora: {dirs:?}");
}
