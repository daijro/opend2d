use anyhow::{ensure, Result};
use d2d_reference::{read_png, write_png};
use std::path::Path;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error:#}");
        std::process::exit(2);
    }
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 4 || (args.len() == 5 && args[4] == "--bgra"),
        "usage: png-diff EXPECTED.png ACTUAL.png DIFF.png [--bgra]"
    );
    let (size, mut a) = read_png(Path::new(&args[1]))?;
    let (other, mut b) = read_png(Path::new(&args[2]))?;
    ensure!(
        size == other,
        "image dimensions differ: {size:?} vs {other:?}"
    );
    let raw = args.len() == 5;
    if raw {
        a = std::fs::read(Path::new(&args[1]).with_extension("bgra"))?;
        b = std::fs::read(Path::new(&args[2]).with_extension("bgra"))?;
        ensure!(
            a.len() == size[0] as usize * size[1] as usize * 4 && b.len() == a.len(),
            "invalid BGRA sidecar size"
        );
    }
    let mut max = [0u8; 4];
    let mut sums = [0u64; 4];
    let mut diff = Vec::with_capacity(a.len());
    let mut changed = 0u64;
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let d: [u8; 4] = std::array::from_fn(|c| x[c].abs_diff(y[c]));
        changed += u64::from(d != [0; 4]);
        for c in 0..4 {
            max[c] = max[c].max(d[c]);
            sums[c] += d[c] as u64;
        }
        // Show alpha-only errors too; output alpha is always opaque.
        let rgb = if raw {
            [d[2], d[1], d[0]]
        } else {
            [d[0], d[1], d[2]]
        };
        diff.extend([rgb[0].max(d[3]), rgb[1].max(d[3]), rgb[2].max(d[3]), 255]);
    }
    let count = size[0] as u64 * size[1] as u64;
    let mean = sums.map(|v| v as f64 / count as f64);
    println!(
        "{}",
        serde_json::json!({
            "encoding": if raw { "premultiplied BGRA8" } else { "decoded RGBA8" },
            "max": max.iter().max().unwrap(), "mean": sums.iter().sum::<u64>() as f64 / (count * 4) as f64,
            "channel_max": max, "channel_mean": mean, "changed_pixels": changed, "pixels": count
        })
    );
    write_png(Path::new(&args[3]), size, &diff)?;
    if changed != 0 {
        std::process::exit(1);
    }
    Ok(())
}
