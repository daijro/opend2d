# d2d-types

Portable (native + `wasm32-unknown-unknown`) Direct2D / DirectWrite / WIC / DXGI value types,
enums, constants, GUIDs/CLSIDs, a public-interface inventory (method order, slots, IIDs), and the
frozen cross-crate contracts (phase 0A).

## What is generated vs hand-written

| File | Lines | Source |
| --- | ---: | --- |
| `src/abi.rs` | ~9.2k | **generated** — value types, enums, constants, GUIDs/CLSIDs |
| `src/interfaces.rs` | ~10.1k | **generated** — 310 interfaces, 1458 methods, IIDs, slots |
| `src/lib.rs` | ~130 | hand-written — inventory types + portable `prelude` |
| `src/contracts.rs` | ~90 | hand-written — the 7 frozen contracts |
| `gen/src/main.rs` | ~260 | hand-written — the generator |
| `tests/inventory.rs` | ~110 | hand-written — tests (incl. `#[cfg(windows)]` parity) |

The two generated files are `@generated` — **do not edit them by hand**; change `gen/` and rerun.

## Regenerating

Pinned input: the `windows` crate **0.62.2** source (MIT OR Apache-2.0), as resolved in `Cargo.lock`
and unpacked under the Cargo registry. No `Windows.Win32.winmd` / `windows-bindgen` is available
locally, so the generator transforms that source (the sanctioned fallback): it strips the COM/link
machinery, flattens the four namespaces into one `abi` module, lowers COM-interface-typed fields to
a pointer-sized `prelude::ComPtr`, and reads the interface inventory from the `define_interface!`,
`interface_hierarchy!` and `*_Vtbl` declarations.

```sh
# default input = windows 0.62.2 in the Cargo registry; output = ../src/{abi,interfaces}.rs
cd crates/d2d-types/gen && cargo run --offline
# or point at an explicit source tree and output dir:
cargo run --offline -- \
  ~/.cargo/registry/src/index.crates.io-*/windows-0.62.2/src/Windows/Win32/Graphics \
  ../src
```

The generator asserts it found exactly 310 interfaces and 1458 methods, and reports any value type
it skips (0 for windows 0.62.2). `gen/` is excluded from the workspace (see the root `Cargo.toml`).

## Tests

- `cargo test -p d2d-types` (native) and `cargo build -p d2d-types --target wasm32-unknown-unknown`:
  inventory structure/slot invariants and representative portable layouts.
- `cargo test -p d2d-types --target x86_64-pc-windows-gnu` (run under Wine): the `windows_parity`
  module proves sampled sizes, alignments, enum values, GUIDs, IIDs and vtable slot counts match the
  `windows` crate itself.

## License

Generated declarations are derived from the `windows` crate (Microsoft), MIT OR Apache-2.0. The MIT
notice is retained; see <https://github.com/microsoft/windows-rs>. No `ref/` content is used.
