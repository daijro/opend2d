# d2d-types

Portable Direct2D, DirectWrite, WIC and DXGI types that build both for native targets and for
`wasm32-unknown-unknown`. The crate holds the value types, enums, constants, GUIDs and CLSIDs, an
inventory of the public interfaces (method order, vtable slots and IIDs), and the frozen contracts
shared between crates.

## Generated and hand-written files

| File | Lines | Source |
| --- | ---: | --- |
| `src/abi.rs` | ~9.2k | generated: value types, enums, constants, GUIDs and CLSIDs |
| `src/interfaces.rs` | ~10.1k | generated: 310 interfaces, 1458 methods, IIDs and slots |
| `src/lib.rs` | ~130 | hand-written: inventory types and the portable `prelude` |
| `src/contracts.rs` | ~90 | hand-written: the 7 frozen contracts |
| `gen/src/main.rs` | ~260 | hand-written: the generator |
| `tests/inventory.rs` | ~110 | hand-written: tests, including the `#[cfg(windows)]` parity checks |

The two generated files are marked `@generated`. Do not edit them by hand; change `gen/` and run it
again.

## Regenerating

The input is pinned to the source of the `windows` crate, version 0.62.2 (MIT OR Apache-2.0), as
resolved in `Cargo.lock` and unpacked in the Cargo registry. Neither `Windows.Win32.winmd` nor
`windows-bindgen` is available locally, so the generator transforms that source instead, which is
the sanctioned fallback. It strips the COM and link machinery, flattens the four namespaces into a
single `abi` module, and lowers fields typed as COM interfaces to a pointer-sized
`prelude::ComPtr`. It reads the interface inventory from the `define_interface!`,
`interface_hierarchy!` and `*_Vtbl` declarations.

```sh
# default input = windows 0.62.2 in the Cargo registry; output = ../src/{abi,interfaces}.rs
cd crates/d2d-types/gen && cargo run --offline
# or point at an explicit source tree and output dir:
cargo run --offline -- \
  ~/.cargo/registry/src/index.crates.io-*/windows-0.62.2/src/Windows/Win32/Graphics \
  ../src
```

The generator asserts that it found exactly 310 interfaces and 1458 methods, and it reports any
value type it skips (none for windows 0.62.2). `gen/` is excluded from the workspace (see the root
`Cargo.toml`).

## Tests

- `cargo test -p d2d-types` (native) and `cargo build -p d2d-types --target wasm32-unknown-unknown`
  check the inventory's structure and slot invariants and a representative set of portable layouts.
- `cargo test -p d2d-types --target x86_64-pc-windows-gnu`, run under Wine, runs the
  `windows_parity` module. It checks that sampled sizes, alignments, enum values, GUIDs, IIDs and
  vtable slot counts match the `windows` crate itself.

## License

The generated declarations are derived from Microsoft's `windows` crate, which is MIT OR
Apache-2.0. The MIT notice is retained; see <https://github.com/microsoft/windows-rs>. No `ref/`
content is used.
