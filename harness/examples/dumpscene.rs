// Fuzz-diff triage tool. `cargo run -p d2d-reference --example dumpscene -- <seed>` prints one scene's
// gen_scene JSON; `-- <start> <count> <outdir>` writes <outdir>/fz-<seed:010>.json for the range (the
// filename fuzz_diff / cap.py expect), for a native capture batch.
use std::io::Write;
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.len() {
        0 | 1 => {
            let seed: u64 = a.first().and_then(|s| s.parse().ok()).unwrap_or(0);
            println!("{:#}", d2d_reference::fuzz::gen_scene(seed));
        }
        _ => {
            let start: u64 = a[0].parse().expect("start");
            let count: u64 = a[1].parse().expect("count");
            let dir = std::path::Path::new(&a[2]);
            std::fs::create_dir_all(dir).unwrap();
            for seed in start..start + count {
                let p = dir.join(format!("fz-{seed:010}.json"));
                let mut f = std::fs::File::create(&p).unwrap();
                write!(f, "{}", d2d_reference::fuzz::gen_scene(seed)).unwrap();
            }
            eprintln!("wrote {count} scenes to {}", dir.display());
        }
    }
}
