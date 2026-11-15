//! Emit seeded random fuzz scenes and self-check each by round-tripping through the real scene
//! validator (`read_scene`). Host-native (no Windows deps): the generator only writes JSON.
//!
//!   fuzz-gen <out-dir> <count> [start-seed]
//!
//! Writes `<out-dir>/fz-<seed>.json` for seeds `start..start+count`. Exits non-zero if any generated
//! scene fails validation (a generator bug), so the runner can trust every scene it captures.

use anyhow::{anyhow, Context, Result};
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!((3..=4).contains(&args.len()), "usage: fuzz-gen <out-dir> <count> [start-seed]");
    let dir = &args[1];
    let count: u64 = args[2].parse().context("count")?;
    let start: u64 = args.get(3).map(|s| s.parse()).transpose().context("start-seed")?.unwrap_or(0);
    std::fs::create_dir_all(dir)?;
    for i in 0..count {
        let seed = start + i;
        let scene = d2d_reference::fuzz::gen_scene(seed);
        let path = format!("{dir}/fz-{seed:010}.json");
        std::fs::write(&path, serde_json::to_string(&scene)?)?;
        d2d_reference::read_scene(Path::new(&path)).map_err(|e| anyhow!("seed {seed} invalid: {e:#}"))?;
    }
    eprintln!("generated + validated {count} scenes (seeds {start}..{}) in {dir}", start + count);
    Ok(())
}
