//! Decode-parity on representative *built-in* shader bytecode (phase 3C.1): the 95 ComputeSharp.D2D1
//! pixel shaders Paint.NET embeds. Their bytecode is Microsoft/Paint.NET property, never committed —
//! this test extracts each blob from the gitignored C# dump at runtime and is skipped when `ref/`
//! is absent. (The 326 d2d1.dll blobs are already walked by `decode.rs`.)
//!
//! It asserts every ComputeSharp `ps_5_0` parses and its whole instruction stream decodes to known
//! executable opcodes or declarations — no stray `Other(n)` with an unrecognised code — so the
//! executor sees real D2D-compiled shaders faithfully.

use d2d_shader::sm4::Opcode;
use d2d_shader::Shader;
use std::collections::BTreeSet;
use std::path::PathBuf;

fn corpus() -> Option<PathBuf> {
    let p = d2d_reference::ref_dir()?.join("managed-d2d/ComputeSharp.D2D1.Generated");
    p.is_dir().then_some(p)
}

/// Pull the `private static ReadOnlySpan<byte> Data => new byte[N] { … }` array out of a generated
/// memory-manager `.cs` file.
fn extract_bytecode(cs: &str) -> Option<Vec<u8>> {
    let at = cs.find("new byte[")?;
    let rest = &cs[at + "new byte[".len()..];
    let n: usize = rest[..rest.find(']')?].trim().parse().ok()?;
    let open = at + rest.find('{')? + "new byte[".len();
    let body = &cs[open + 1..];
    let end = body.find('}')?;
    let bytes: Vec<u8> = body[..end]
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|t| !t.is_empty())
        .filter_map(|t| t.parse().ok())
        .collect();
    (bytes.len() >= n).then(|| bytes[..n].to_vec())
}

/// Declaration opcodes we recognise structurally but do not execute (`dcl_*`, incl. the SM5 range).
fn is_known_decl(code: u16) -> bool {
    matches!(code, 88..=107 | 143..=162)
}

#[test]
fn computesharp_pixel_shaders_decode() {
    let Some(dir) = corpus() else {
        eprintln!("skip: ComputeSharp corpus absent");
        return;
    };
    let mut count = 0u32;
    let mut runnable = 0u32;
    let mut missing: BTreeSet<u16> = BTreeSet::new();
    let mut unknown: BTreeSet<u16> = BTreeSet::new();
    let mut opcodes: BTreeSet<String> = BTreeSet::new();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if !name.contains("HlslBytecodeMemoryManager") {
            continue;
        }
        let cs = std::fs::read_to_string(&path).unwrap();
        let Some(bytes) = extract_bytecode(&cs) else { continue };
        let shader = Shader::parse(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(shader.program.major, 5, "{name}: expected shader model 5");
        for inst in &shader.program.insts {
            match inst.op {
                Opcode::Other(code) if !is_known_decl(code) => {
                    unknown.insert(code);
                }
                _ => {
                    opcodes.insert(format!("{:?}", inst.op));
                }
            }
        }
        match shader.unsupported_op() {
            None => runnable += 1,
            Some(code) => {
                missing.insert(code);
            }
        }
        count += 1;
    }
    eprintln!(
        "builtin_decode: {count} ComputeSharp ps_5_0 decoded; {} distinct opcodes; {runnable}/{count} fully runnable by the executor (unimplemented opcodes: {missing:?})",
        opcodes.len()
    );
    assert!(count >= 90, "only decoded {count} shaders");
    assert!(unknown.is_empty(), "unrecognised opcodes (not declarations): {unknown:?}");
}
