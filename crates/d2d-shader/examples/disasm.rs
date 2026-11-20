//! Reference-only DXBC disassembler: `cargo run --example disasm -- <file.dxbc>`.
//! Prints the decoded signature + instruction stream so a captured native shader can be read and
//! ported op-for-op. Not part of the library; the captured bytecode it reads is never committed.
use d2d_shader::dxbc::Shader;
use d2d_shader::sm4::{Index, Inst, Operand, Sel};

fn sel(s: &Sel) -> String {
    match s {
        Sel::Mask(m) => {
            let mut o = String::from(".");
            for (b, c) in [(1, 'x'), (2, 'y'), (4, 'z'), (8, 'w')] {
                if m & b != 0 {
                    o.push(c);
                }
            }
            o
        }
        Sel::Swizzle(sw) => format!(".{}{}{}{}", "xyzw".as_bytes()[sw[0] as usize] as char, "xyzw".as_bytes()[sw[1] as usize] as char, "xyzw".as_bytes()[sw[2] as usize] as char, "xyzw".as_bytes()[sw[3] as usize] as char),
        Sel::Select1(c) => format!(".{}", "xyzw".as_bytes()[*c as usize] as char),
        Sel::None => String::new(),
    }
}

fn idx(i: &Index) -> String {
    match i {
        Index::Imm(n) => n.to_string(),
        Index::Rel(o) => format!("[{}]", op(o)),
        Index::ImmPlusRel(n, o) => format!("{n}+[{}]", op(o)),
    }
}

fn op(o: &Operand) -> String {
    let base = match o.ty {
        0 => format!("r{}", o.indices.first().map(idx).unwrap_or_default()),
        1 => format!("v{}", o.indices.first().map(idx).unwrap_or_default()),
        2 => format!("o{}", o.indices.first().map(idx).unwrap_or_default()),
        3 => format!("x{}", o.indices.iter().map(idx).collect::<Vec<_>>().join("][")),
        4 => {
            // immediate32: show as float and hex (gradient constants are floats)
            if o.imm.len() == 1 {
                let f = f32::from_bits(o.imm[0]);
                format!("imm({f}=0x{:08x})", o.imm[0])
            } else {
                let v: Vec<String> = o.imm.iter().map(|&b| format!("{}", f32::from_bits(b))).collect();
                format!("imm({})", v.join(","))
            }
        }
        6 => format!("s{}", o.indices.first().map(idx).unwrap_or_default()),
        7 => format!("t{}", o.indices.first().map(idx).unwrap_or_default()),
        8 => format!("cb{}[{}]", o.indices.first().map(idx).unwrap_or_default(), o.indices.get(1).map(idx).unwrap_or_default()),
        13 => "null".into(),
        n => format!("ty{n}{}", o.indices.iter().map(|i| format!("[{}]", idx(i))).collect::<Vec<_>>().join("")),
    };
    let m = match o.modifier {
        1 => "-",
        2 => "|abs|",
        3 => "-abs",
        _ => "",
    };
    format!("{m}{base}{}", sel(&o.sel))
}

fn inst(i: &Inst) -> String {
    let sat = if i.saturate { "_sat" } else { "" };
    let ops: Vec<String> = i.operands.iter().map(op).collect();
    format!("{:?}{sat} {}", i.op, ops.join(", "))
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: disasm <file.dxbc>");
    let bytes = std::fs::read(&path).expect("read dxbc");
    let sh = Shader::parse(&bytes).expect("parse dxbc");
    let p = &sh.program;
    println!("// {path}");
    println!("// type={} sm{}.{} temps={} icb={}", p.program_type, p.major, p.minor, p.temps, p.icb.len());
    for s in &sh.inputs {
        println!("// in  {} idx{} reg{} sv{} mask{:#x}", s.semantic, s.semantic_index, s.register, s.system_value, s.mask);
    }
    for s in &sh.outputs {
        println!("// out {} idx{} reg{} sv{} mask{:#x}", s.semantic, s.semantic_index, s.register, s.system_value, s.mask);
    }
    if !p.icb.is_empty() {
        println!("// icb:");
        for (n, v) in p.icb.iter().enumerate() {
            println!("//   [{n}] = {},{},{},{}", f32::from_bits(v[0]), f32::from_bits(v[1]), f32::from_bits(v[2]), f32::from_bits(v[3]));
        }
    }
    for (n, i) in p.insts.iter().enumerate() {
        println!("{n:3}: {}", inst(i));
    }
}
