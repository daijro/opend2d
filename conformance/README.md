# conformance

The cross-platform conformance runner and its test corpus. Native `d2d1.dll` on Windows WARP is the
oracle; a case passes only on **exact** pixel match (raw-bit for float targets), never a tolerance.

## suite/ — the corpus (private git submodule)

All scenes, native WARP/DirectWrite captures (oracles), the ratchet, the scene generators, and the
per-suite fixture data live in a **separate private repository**, added here as the git submodule
`conformance/suite` so that opend2d itself tracks only code. Layout and the add-a-scene / capture
workflow are in that repo's own `README.md`.

- Initialise it in a worktree with `git submodule update --init conformance/suite` (clones from the
  configured URL — a local path for now, repointed to the private remote by the coordinator).
- The runner finds it via `d2d_reference::suite_dir()` (and the matching resolver in the decisive.py
  runners and `scripts/gen-wasm-embed.sh`): `$D2D_SUITE`, else this in-tree submodule, else the main
  checkout's `conformance/suite`, else absent. **Absent ⇒ every runner and test skips cleanly**, so a
  public clone without the submodule builds and its tests pass.

## the runner

`crates/d2d/tests/conformance.rs` replays the corpus on Windows, Linux, macOS, FreeBSD and `wasm32`
in a browser, reports exact-pass counts and the first differing coordinate/channel, and holds the
line against `suite/passing.txt`. `scripts/conformance.sh <native|wasm|windows|aarch64|parity|…>`
drives one target each; `parity` proves the full-output digest is identical across targets. The
`wasm32` build has no filesystem, so `scripts/gen-wasm-embed.sh` bakes the suite into
`crates/d2d/tests/embed/` (committed empty, overwritten locally — never commit the populated form).

See `crates/d2d-types/` for the frozen value/layout/command contracts, and `harness/` for the
reference-capture and image-diff tooling (`harness/vm.py` drives the reference VM; captured oracles
are written into `conformance/suite/oracle/win11-26100/`).
