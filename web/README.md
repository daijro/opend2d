# OpenD2D conformance in the browser

This crate compiles the conformance suite to wasm32 and runs it in a browser. It uses exactly the
same driver and corpus as `crates/d2d/tests/conformance.rs`, with no logic duplicated: `src/lib.rs`
includes that test file as a module and wraps its `report()` for JavaScript. The wasm32 build has
no filesystem, so the corpus is baked in at compile time from `crates/d2d/tests/embed/corpus.rs`.
Regenerate it with `scripts/gen-wasm-embed.sh`.

## The page

```sh
scripts/conformance.sh web            # builds web/pkg
python3 -m http.server -d web 8000    # then open http://localhost:8000
```

The page runs every scene and shows which ones passed, were skipped or failed, along with the
full-output digest and any regression.

## One-command checks

Each of these runs the same suite against the same Windows captures. `parity` checks that the
output is bit-identical across all the targets the host can run. Set `OFFLINE=1` to use only the
local cargo cache.

```sh
scripts/conformance.sh native         # x86_64 host
scripts/conformance.sh wasm           # wasm32 under Node
scripts/conformance.sh wasm-browser   # wasm32 in a real headless browser
scripts/conformance.sh windows        # x86_64-pc-windows-gnu under Wine
scripts/conformance.sh checks         # compile for aarch64 / macOS / FreeBSD
scripts/conformance.sh parity         # assert identical digest on native + wasm + Windows
```

The wasm runs need a `wasm-bindgen-cli` that matches the pinned `wasm-bindgen` version (0.2.128),
and `wasm-browser` also needs `chromedriver` on `PATH`. CI (`.github/workflows/conformance.yml`)
runs every target, including aarch64 under qemu, Windows itself, macOS and FreeBSD, and fails
unless the digests are identical.
