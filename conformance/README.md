# conformance

This directory holds the cross-platform conformance runner and its test corpus. Direct2D (d2d1.dll
on Windows WARP) is the reference. A case passes only on an exact pixel match, never within a
tolerance, and float targets are compared on their raw bits.

## The corpus: suite/

The scenes, the Windows WARP and DirectWrite captures they are checked against, the ratchet, the
scene generators and the per-suite fixture data all live in a separate private repository. It is
added here as the git submodule `conformance/suite`, so OpenD2D itself tracks only code. That
repository's own `README.md` describes its layout and how to add and capture a scene.

To initialise it in a worktree, run `git submodule update --init conformance/suite`. This clones
from the configured URL, which is a local path for now and will be repointed at the private remote.

The runner finds the suite through `d2d_reference::suite_dir()`, and the decisive.py runners and
`scripts/gen-wasm-embed.sh` use a matching resolver. They look, in order, at `$D2D_SUITE`, this
in-tree submodule, and the main checkout's `conformance/suite`. If none of these exists, every
runner and test skips cleanly, so a public clone without the submodule still builds and passes its
tests.

## The runner

`crates/d2d/tests/conformance.rs` replays the corpus on Windows, Linux, macOS, FreeBSD and in a
browser as `wasm32`. It reports how many cases pass exactly and, for each failure, the first
coordinate and channel that differ. It also holds the line against `suite/passing.txt`.

`scripts/conformance.sh <native|wasm|windows|aarch64|parity|…>` drives one target per run. The
`parity` target checks that the full-output digest is identical across targets.

The `wasm32` build has no filesystem, so `scripts/gen-wasm-embed.sh` bakes the suite into
`crates/d2d/tests/embed/`. That directory is committed empty and overwritten locally. Never commit
the populated version.

The frozen value, layout and command contracts are in `crates/d2d-types/`. The reference-capture
and image-diff tooling is in `harness/`: `harness/vm.py` drives the reference VM, and captured
Windows output is written to `conformance/suite/oracle/win11-26100/`.
