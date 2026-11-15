//! No-oracle robustness fuzz for the WIC image decoders and the font loader: a decoder (which in a
//! browser runs on untrusted files) must REJECT malformed input with an `Err`/`None`, never panic,
//! hang or allocate unboundedly. This drives each decoder with random bytes, format-magic-prefixed
//! random, and byte-mutated valid images (encoded via the matching encoder), under `catch_unwind`.
//!
//! Panics are caught and reported here (the test fails with the offending input saved to
//! `$TMPDIR/wic_fuzz_fail_*.bin`). Hangs and unbounded allocations cannot be caught in-process — run
//! this under a watchdog: `ulimit -v 3000000; timeout 300 cargo test -p d2d --test wic_fuzz --release`.
//! The per-input progress line (iter + format + input hash) on stderr identifies a hang/OOM repro.
//!
//! Deterministic: the input stream is a seeded xorshift (FUZZ_SEED env, default 1), FUZZ_ITERS the
//! count (default 20000). A found panic is printed with its seed+iter so it replays exactly.

use std::panic::{catch_unwind, AssertUnwindSafe};

/// Tiny deterministic xorshift64* PRNG — no dev-dependency, reproducible from a seed.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next() % n as u64) as usize
    }
    fn byte(&mut self) -> u8 {
        self.next() as u8
    }
}

/// A seed image for mutation: a small valid file in each encodable format (so mutants reach past the
/// header into the real decode paths). Not every decoder has an encoder here; those are fuzzed from
/// random + magic only, which still exercises their header/parse paths.
fn seed_images() -> Vec<(&'static str, Vec<u8>)> {
    use d2d::wic::{encode_bmp, encode_gif, encode_tiff, Decoded};
    // A 3×2 BGRA32 test image (BGRA32 = the WIC pixel-format GUID the encoders accept).
    let (w, h) = (3usize, 2usize);
    let mut px = vec![0u8; w * h * 4];
    for (i, c) in px.chunks_mut(4).enumerate() {
        c.copy_from_slice(&[(i * 40) as u8, (i * 20) as u8, (i * 60) as u8, 255]);
    }
    let img = Decoded {
        width: w as u32,
        height: h as u32,
        format: "6fddc324-4e03-4bfe-b185-3d77768dc90f".into(), // fmt::BGRA32
        stride: w * 4,
        pixels: px,
        palette: Vec::new(),
    };
    let mut out: Vec<(&'static str, Vec<u8>)> = Vec::new();
    if let Ok(b) = encode_bmp(&img, 96.0) {
        out.push(("bmp", b));
    }
    if let Ok(b) = encode_gif(&img) {
        out.push(("gif", b));
    }
    if let Ok(b) = encode_tiff(&img) {
        out.push(("tiff", b));
    }
    // A real TrueType font so byte-mutated seeds drive the font loader past the sfnt header into the
    // table-directory / per-table parse (the "malformed sfnt tables" the loader must reject, not panic).
    out.push(("font", include_bytes!("../src/text/color/testdata/colrtest.ttf").to_vec()));
    out
}

/// The format magic prefixes, so random bodies reach each decoder's header parse.
const MAGIC: &[(&str, &[u8])] = &[
    ("bmp", b"BM"),
    ("png", b"\x89PNG\r\n\x1a\n"),
    ("gif", b"GIF89a"),
    ("jpeg", b"\xff\xd8\xff\xe0"),
    ("tiff", b"II*\x00"),
    ("tiff", b"MM\x00*"),
    ("ico", b"\x00\x00\x01\x00"),
    ("dds", b"DDS "),
    ("jxr", b"II\xbc\x01"),
    ("font", b"\x00\x01\x00\x00"), // TrueType sfnt
    ("font", b"OTTO"),             // CFF OpenType
    ("font", b"ttcf"),             // TrueType collection
    ("font", b"true"),             // legacy Apple sfnt
];

/// Run every decoder on `input` under catch_unwind; return the name of the first that panicked, if any.
/// When `trace` is set, log each decoder before calling it (flushed), so an uncatchable abort (OOM,
/// stack overflow) leaves the culprit as the last stderr line.
fn decode_all(input: &[u8]) -> Option<&'static str> {
    let trace = std::env::var_os("FUZZ_TRACE").is_some();
    macro_rules! one {
        ($name:literal, $call:expr) => {
            if trace {
                eprintln!("  try {} ({} bytes)", $name, input.len());
                use std::io::Write;
                let _ = std::io::stderr().flush();
            }
            if catch_unwind(AssertUnwindSafe(|| {
                let _ = $call;
            }))
            .is_err()
            {
                return Some($name);
            }
        };
    }
    one!("bmp", d2d::wic::decode_bmp(input));
    one!("png", d2d::wic::decode_png(input));
    one!("gif", d2d::wic::decode_gif(input));
    one!("jpeg", d2d::wic::decode_jpeg(input));
    one!("jpeg_scaled", d2d::wic::decode_jpeg_scaled(input, 2));
    one!("tiff", d2d::wic::decode_tiff(input));
    one!("ico", d2d::wic::decode_ico(input));
    one!("dds", d2d::wic::decode_dds(input));
    one!("jxr", d2d::wic::decode_jxr(input));
    one!("font", d2d::text::shaping::Font::from_bytes(input.to_vec(), 0));
    None
}

/// Replay a saved crash input directly (no catch_unwind) to see the exact panic/abort + backtrace:
/// `FUZZ_REPRO=$TMPDIR/wic_fuzz_cur.bin FUZZ_REPRO_DEC=font RUST_BACKTRACE=1 cargo test -p d2d --test
/// wic_fuzz --release repro -- --nocapture`.
#[test]
fn repro() {
    let Ok(path) = std::env::var("FUZZ_REPRO") else {
        return;
    };
    let input = std::fs::read(&path).expect("read repro");
    let dec = std::env::var("FUZZ_REPRO_DEC").unwrap_or_else(|_| "font".into());
    eprintln!("repro {dec} on {} ({} bytes)", path, input.len());
    match dec.as_str() {
        "font" => {
            let _ = d2d::text::shaping::Font::from_bytes(input, 0);
        }
        "tiff" => {
            let _ = d2d::wic::decode_tiff(&input);
        }
        "gif" => {
            let _ = d2d::wic::decode_gif(&input);
        }
        "bmp" => {
            let _ = d2d::wic::decode_bmp(&input);
        }
        "png" => {
            let _ = d2d::wic::decode_png(&input);
        }
        "ico" => {
            let _ = d2d::wic::decode_ico(&input);
        }
        "jpeg" => {
            let _ = d2d::wic::decode_jpeg(&input);
        }
        "dds" => {
            let _ = d2d::wic::decode_dds(&input);
        }
        "jxr" => {
            let _ = d2d::wic::decode_jxr(&input);
        }
        _ => panic!("unknown decoder {dec}"),
    }
    eprintln!("repro returned (no crash)");
}

/// One fuzz input, chosen by the same deterministic strategy mix (pure random, format-magic + random
/// body, or a byte-mutated seed image). Shared by the robustness fuzz and the identity digest so both
/// drive the decoders with the identical input stream for a given seed.
fn gen_input(rng: &mut Rng, seeds: &[(&'static str, Vec<u8>)]) -> Vec<u8> {
    match rng.below(4) {
        0 => {
            let n = rng.below(512);
            (0..n).map(|_| rng.byte()).collect()
        }
        1 => {
            let (_, m) = MAGIC[rng.below(MAGIC.len())];
            let n = rng.below(256);
            let mut v = m.to_vec();
            v.extend((0..n).map(|_| rng.byte()));
            v
        }
        _ if !seeds.is_empty() => {
            let (_, base) = &seeds[rng.below(seeds.len())];
            let mut v = base.clone();
            if !v.is_empty() {
                let muts = 1 + rng.below(8);
                for _ in 0..muts {
                    if v.is_empty() {
                        break;
                    }
                    match rng.below(3) {
                        0 => {
                            let i = rng.below(v.len());
                            v[i] ^= rng.byte();
                        }
                        1 => {
                            let i = rng.below(v.len());
                            v.insert(i, rng.byte());
                        }
                        _ => {
                            let i = rng.below(v.len());
                            v.truncate(i);
                        }
                    }
                }
            }
            v
        }
        _ => (0..rng.below(512)).map(|_| rng.byte()).collect(),
    }
}

// Identity gate, image-decoder side: one combined digest of every decoder's output (dimensions and
// pixels, or the error/panic outcome) across the deterministic input stream, so a base build and a
// candidate build can be compared. Gated by WIC_HASH; FUZZ_SEED/FUZZ_ITERS size the run.
#[test]
fn wic_digest() {
    if std::env::var_os("WIC_HASH").is_none() {
        return;
    }
    let seed: u64 = std::env::var("FUZZ_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let iters: usize = std::env::var("FUZZ_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let seeds = seed_images();
    let mut rng = Rng(seed | 1);
    std::panic::set_hook(Box::new(|_| {}));
    use std::hash::Hasher;
    let mut h = std::hash::DefaultHasher::new();
    // Decoders return heterogeneous shapes (a single image, or a frame list); hash each result's full
    // debug form so dimensions, pixels and error all fold in uniformly.
    macro_rules! hd {
        ($name:literal, $e:expr) => {{
            h.write($name.as_bytes());
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| format!("{:?}", $e))) {
                Ok(s) => {
                    h.write_u8(1);
                    h.write(s.as_bytes());
                }
                Err(_) => h.write_u8(2),
            }
        }};
    }
    for _ in 0..iters {
        let input = gen_input(&mut rng, &seeds);
        hd!("bmp", d2d::wic::decode_bmp(&input));
        hd!("png", d2d::wic::decode_png(&input));
        hd!("gif", d2d::wic::decode_gif(&input));
        hd!("jpeg", d2d::wic::decode_jpeg(&input));
        hd!("jpeg_scaled", d2d::wic::decode_jpeg_scaled(&input, 2));
        hd!("tiff", d2d::wic::decode_tiff(&input));
        hd!("ico", d2d::wic::decode_ico(&input));
        hd!("dds", d2d::wic::decode_dds(&input));
        hd!("jxr", d2d::wic::decode_jxr(&input));
    }
    let _ = std::panic::take_hook();
    println!("WIC-DIGEST {seed} {iters} {:#018x}", h.finish());
}

#[test]
fn wic_decoders_never_panic() {
    let seed: u64 = std::env::var("FUZZ_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let iters: usize = std::env::var("FUZZ_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let seeds = seed_images();
    let mut rng = Rng(seed | 1);
    // Suppress the default panic printout; catch_unwind still returns Err so we record it ourselves.
    std::panic::set_hook(Box::new(|_| {}));
    let mut fails: Vec<(usize, &'static str, Vec<u8>)> = Vec::new();

    for iter in 0..iters {
        let input = gen_input(&mut rng, &seeds);

        if std::env::var_os("FUZZ_TRACE").is_some() {
            eprintln!("iter {iter} ({} bytes)", input.len());
        }
        // Write the current input before decoding so an UNCATCHABLE abort (stack overflow / OOM, which
        // catch_unwind cannot intercept) leaves its culprit on disk as the last-written file.
        if let Ok(dir) = std::env::var("FUZZ_CUR") {
            let _ = std::fs::write(format!("{dir}/wic_fuzz_cur.bin"), &input);
        }
        if let Some(name) = decode_all(&input) {
            fails.push((iter, name, input));
            if fails.len() > 20 {
                break;
            }
        }
    }

    let _ = std::panic::take_hook();
    if !fails.is_empty() {
        let dir = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
        for (iter, name, input) in &fails {
            let p = format!("{dir}/wic_fuzz_fail_{name}_{iter}.bin");
            let _ = std::fs::write(&p, input);
            eprintln!("PANIC seed={seed} iter={iter} decoder={name} ({} bytes) -> {p}", input.len());
        }
        panic!("{} decoder input(s) panicked (seed {seed}); see $TMPDIR/wic_fuzz_fail_*.bin", fails.len());
    }
}
