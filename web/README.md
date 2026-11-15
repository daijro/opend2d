# opend2d conformance in the browser (phase 4A)

This crate compiles the conformance suite — the exact driver and oracle corpus from
`crates/d2d/tests/conformance.rs` — to wasm32 and runs it in a browser. No logic is duplicated:
`src/lib.rs` includes that test file as a module and wraps its `report()` for JS (see
`src/lib.rs`). The wasm32 build has no filesystem, so the corpus is baked in at compile time from
`crates/d2d/tests/embed/corpus.rs` (regenerate with `scripts/gen-wasm-embed.sh`).

## The page

```sh
scripts/conformance.sh web            # builds web/pkg
python3 -m http.server -d web 8000    # then open http://localhost:8000
```

The page runs every scene and shows pass/skip/fail, the full-output digest, and any regression.

## One-command checks

Each drives the same suite against the same oracle; `parity` proves bit-identical output across the
targets runnable on the host. Set `OFFLINE=1` to use only the local cargo cache.

```sh
scripts/conformance.sh native         # x86_64 host
scripts/conformance.sh wasm           # wasm32 under Node
scripts/conformance.sh wasm-browser   # wasm32 in a real headless browser
scripts/conformance.sh windows        # x86_64-pc-windows-gnu under Wine
scripts/conformance.sh checks         # compile for aarch64 / macOS / FreeBSD
scripts/conformance.sh parity         # assert identical digest on native + wasm + Windows
```

The wasm runs need `wasm-bindgen-cli` matching the pinned `wasm-bindgen` (0.2.128) and, for
`wasm-browser`, `chromedriver` on `PATH`. CI (`.github/workflows/conformance.yml`) runs all targets,
including aarch64 (qemu), native Windows, macOS and FreeBSD, and gates on identical digests.
