//! Phase 4A browser runner: the conformance suite, in the browser and under Node.
//!
//! No driver or corpus is duplicated here. The whole suite — scene lowering, the public-API
//! driver, the byte-for-byte comparison, and the oracle corpus baked in for wasm — lives in
//! `crates/d2d/tests/conformance.rs` and is pulled in verbatim as a module. This file only wraps
//! its platform-free `report()` for JS, so a browser page and the Node/headless-browser test run
//! identical code and therefore produce identical results.
#![cfg(target_arch = "wasm32")]

#[path = "../../crates/d2d/tests/conformance.rs"]
mod conformance;

use wasm_bindgen::prelude::wasm_bindgen;

/// Run the full suite and return the result as JSON for the page to render.
#[wasm_bindgen]
pub fn run() -> String {
    let r = conformance::report();
    serde_json::json!({
        "pass": r.pass,
        "skip": r.skip,
        "fail": r.fail,
        "fails": r.fails,
        "regressed": r.regressed,
        "missing": r.missing,
        "digest": format!("{:#018x}", r.digest),
    })
    .to_string()
}

/// Fuzz parity: the digest of driving gen_scene(seed) for seeds [start, start+n), the wasm side of
/// the native/wasm cross-platform check (native prints the same via the `fuzz_parity` test).
#[wasm_bindgen]
pub fn fuzz_digest(start: u64, n: u64) -> String {
    format!("{:#018x}", conformance::fuzz_digest(start, n))
}

/// Benchmark: drive one scene (given as its JSON) and return the rendered pixels. The caller (a Node
/// or browser harness) times repeated calls with its own clock, so the same core is measured in wasm
/// as natively. Returns an empty buffer if the scene does not parse or render.
#[wasm_bindgen]
pub fn drive_bytes(scene_json: &str) -> Vec<u8> {
    conformance::drive_json_pixels(scene_json)
}
