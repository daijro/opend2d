//! Decoder validation against the 326 real DXBC blobs from `d2d1.dll` and their
//! `D3DDisassemble` output (`ref/windows/shaders/`, gitignored).
//! No VM needed: the token walk must land on exactly the instruction boundaries the disassembler
//! saw, and a hand-verified shader's operands must decode correctly.

use d2d_shader::dxbc::Shader;
use d2d_shader::sm4::{Index, Mem, Opcode, Sel};
use std::path::{Path, PathBuf};

fn corpus() -> Option<PathBuf> {
    let p = d2d_reference::ref_dir()?.join("windows/shaders");
    p.is_dir().then_some(p)
}

/// Count the instruction/declaration lines a `.asm` disassembly holds (one per line after the
/// profile line). Blobs with an inline immediate-constant-buffer block (`{`) are reported as `None`
/// because that customdata block spans several lines but is one skipped token.
fn asm_inst_lines(asm: &str) -> Option<usize> {
    // Empty / failure marker: one of the 13 PRIV lib_5_0/function link-containers D3DDisassemble
    // cannot render (nested, not a standalone stage).
    if asm.trim().is_empty() || asm.starts_with("; D3DDisassemble failed") {
        return None;
    }
    // Count one per instruction/declaration line, skipping the leading profile line and the whole
    // `dcl_immediateConstantBuffer { … }` block (a multi-line customdata token the decoder skips).
    let mut n = 0;
    let mut depth = 0i32;
    for (i, line) in asm.lines().enumerate() {
        let t = line.trim();
        if t.is_empty() || i == 0 {
            continue;
        }
        let braces = line.matches('{').count() as i32 - line.matches('}').count() as i32;
        if depth > 0 {
            depth += braces; // continuation of the icb block
            continue;
        }
        if t.starts_with("dcl_immediateConstantBuffer") {
            depth += braces; // the icb declaration itself is a skipped customdata token
            continue;
        }
        n += 1;
    }
    Some(n)
}

#[test]
fn walk_lands_on_every_instruction_boundary() {
    let Some(dir) = corpus() else {
        eprintln!("skip: reference shader corpus absent");
        return;
    };
    let (mut checked, mut skipped) = (0u32, 0u32);
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("dxbc") {
            continue;
        }
        // Nested PRIV lib/function link-containers disassemble as a failure marker and some have no
        // standalone code chunk — skip them; the standalone stages are what we validate.
        let asm = std::fs::read_to_string(path.with_extension("asm")).unwrap();
        let Some(expected) = asm_inst_lines(&asm) else {
            skipped += 1;
            continue;
        };
        let bytes = std::fs::read(&path).unwrap();
        let shader = Shader::parse(&bytes)
            .unwrap_or_else(|e| panic!("{}: parse failed: {e}", path.display()));
        assert_eq!(
            shader.program.insts.len(),
            expected,
            "{}: decoded {} insts, disassembly shows {expected}",
            path.display(),
            shader.program.insts.len()
        );
        checked += 1;
    }
    assert!(checked > 200, "only checked {checked} blobs");
    eprintln!("decode: {checked} blobs matched instruction count ({skipped} had customdata blocks)");
}

#[test]
fn operands_of_a_known_shader() {
    let Some(dir) = corpus() else {
        return;
    };
    // 0x3b54a0: the smallest ps_4_0; its body is
    //   itof r0.xy, cb0[0].xyxx / add r0.xy, -r0.xyxx, v1.xyxx / ftoi r0.xy, r0.xyxx
    //   iadd r0.z, -cb0[0].x, cb0[0].z / imad r0.x, r0.y, r0.z, r0.x / ld_structured … / ret
    let path = dir.join("0x3b54a0.dxbc");
    if !path.exists() {
        return;
    }
    let s = Shader::parse(&std::fs::read(&path).unwrap()).unwrap();
    let body: Vec<_> = s.program.insts.iter().filter(|i| !matches!(i.op, Opcode::Other(_))).collect();
    // First executable op: itof r0.xy, cb0[0].xyxx — but itof is executable, Other skips dcls.
    let itof = body.iter().find(|i| i.op == Opcode::Itof).expect("itof present");
    assert_eq!(itof.operands[0].ty, 0, "dest is a temp r#");
    assert_eq!(itof.operands[0].sel, Sel::Mask(0b0011), "dest mask .xy");
    assert_eq!(itof.operands[1].ty, 8, "src is constant buffer cb#");
    assert!(matches!(itof.operands[1].indices[0], Index::Imm(0)), "cb slot 0");
    assert!(matches!(itof.operands[1].indices[1], Index::Imm(0)), "cb element 0");

    let add = body.iter().find(|i| i.op == Opcode::Add).expect("add present");
    assert_eq!(add.operands[1].ty, 0, "add src0 is temp");
    assert_eq!(add.operands[1].modifier, 1, "add src0 negated");
    assert_eq!(add.operands[1].sel, Sel::Swizzle([0, 1, 0, 0]), "src0 swizzle .xyxx");
    assert_eq!(add.operands[2].ty, 1, "add src1 is input v#");

    assert_eq!(s.program.insts.last().unwrap().op, Opcode::Ret);
}

#[test]
fn compute_declarations_decode() {
    let Some(dir) = corpus() else {
        return;
    };
    // 0x5804b0: the smallest cs_5_0 — `dcl_uav_structured u0, 12` / `dcl_thread_group 1, 1, 1`.
    let path = dir.join("0x5804b0.dxbc");
    if !path.exists() {
        return;
    }
    let p = Shader::parse(&std::fs::read(&path).unwrap()).unwrap().program;
    assert_eq!(p.program_type, 5, "compute program");
    assert_eq!(p.thread_group, [1, 1, 1]);
    assert_eq!(p.uavs.len(), 1);
    assert_eq!(p.uavs[0].reg, 0);
    assert_eq!(p.uavs[0].mem, Mem::Structured { stride: 12, bytes: 0 });

    // 0x57f6d0 (ink): tgsm_raw g0 (4 bytes), thread_group 10×1×1, uav_structured u0.
    let ink = dir.join("0x57f6d0.dxbc");
    if ink.exists() {
        let p = Shader::parse(&std::fs::read(&ink).unwrap()).unwrap().program;
        assert_eq!(p.thread_group, [10, 1, 1]);
        assert_eq!(p.tgsm, vec![d2d_shader::sm4::Decl { reg: 0, mem: Mem::Raw { bytes: 4 } }]);
    }
}

#[test]
fn signatures_parse() {
    let Some(dir) = corpus() else {
        return;
    };
    let path = dir.join("0x3b54a0.dxbc");
    if !path.exists() {
        return;
    }
    let s = Shader::parse(&std::fs::read(&path).unwrap()).unwrap();
    assert!(s.inputs.iter().any(|e| e.semantic == "SCENE_POSITION" || e.semantic == "SV_POSITION"));
    assert!(s.outputs.iter().any(|e| e.semantic == "SV_Target"));
    let _ = Path::new("");
}

/// The version-1 signature chunks (`ISG1`/`OSG1`) carry a leading u32 `Stream` and a trailing u32
/// `MinPrecision` around the 24-byte `ISGN` body, so each element is 32 bytes (not 36). This is the
/// captured text-outline realization VS, whose nine-element `ISG1` only decodes correctly at the
/// right stride — at the wrong one every element past the first drifts by 4·index into the name
/// table and reports nonsense registers/masks.
#[test]
fn isg1_wide_signature_decodes() {
    let Some(suite) = d2d_reference::suite_dir() else { return };
    let path = suite.join("fixtures/d2d-shader/text_outline_vs.dxbc");
    let s = Shader::parse(&std::fs::read(&path).unwrap()).unwrap();
    let ins: Vec<(&str, u32, u8)> = s.inputs.iter().map(|e| (e.semantic.as_str(), e.register, e.mask)).collect();
    assert_eq!(
        ins,
        vec![
            ("POSITION", 0, 1),
            ("INTERIOR_OFFSET", 1, 1),
            ("BOUNDARY_VERTEX_TYPE", 2, 1),
            ("TRANSFORM_OFFSET", 3, 3),
            ("GLYPH_START_VERTEX", 4, 1),
            ("SEQUENTIAL_START_VERTEX", 5, 1),
            ("INDEXED_START_INDEX", 6, 1),
            ("INSTANCE_TYPE", 7, 1),
            ("CONSTANT_DATA_INDEX", 8, 1),
        ]
    );
    // SV_POSITION at o0, the coverage varying at o1 (not the mis-strided o15).
    let outs: Vec<(&str, u32)> = s.outputs.iter().map(|e| (e.semantic.as_str(), e.register)).collect();
    assert_eq!(outs, vec![("SV_POSITION", 0), ("TEXCOORD", 1)]);
}
