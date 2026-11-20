//! Throwaway: disassemble a captured .dxbc (path in D2D_DISASM) — prints each instruction.
use d2d_shader::Shader;

#[test]
fn disasm() {
    let Ok(path) = std::env::var("D2D_DISASM") else {
        eprintln!("set D2D_DISASM=path.dxbc");
        return;
    };
    let bytes = std::fs::read(&path).unwrap();
    let sh = Shader::parse(&bytes).unwrap();
    eprintln!("== {path} : type={} {}.{}  temps={} ==", sh.program.program_type, sh.program.major, sh.program.minor, sh.program.temps);
    for s in &sh.inputs {
        eprintln!("  IN  {} idx{} reg{} sv{} mask{:#x}", s.semantic, s.semantic_index, s.register, s.system_value, s.mask);
    }
    for (i, inst) in sh.program.insts.iter().enumerate() {
        eprintln!("{i:3}: {:?}{}", inst.op, if inst.saturate { " [sat]" } else { "" });
        for o in &inst.operands {
            eprintln!("        {o:?}");
        }
    }
}
