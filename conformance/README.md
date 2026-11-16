# conformance

This directory is where the conformance test corpus goes. Direct2D (d2d1.dll on Windows WARP) is
the reference. A case passes only on an exact pixel match, never within a
tolerance, and float targets are compared on their raw bits.

## The corpus: suite/

The scenes, the Windows WARP and DirectWrite captures they are checked against, the pass list, the
scene generators and the per-suite fixture data are kept privately and are not part of this
repository, so OpenD2D itself tracks only code. If you have a copy, put it at `conformance/suite` or
point `D2D_SUITE` at it.

The runner finds the suite through `d2d_reference::suite_dir()`, and the decisive.py runners and
`scripts/gen-wasm-embed.sh` use a matching resolver. They look, in order, at `$D2D_SUITE`, this
directory's `suite/`, and the main checkout's `conformance/suite`. If none of these exists, every
runner and test skips cleanly, so a public clone without the suite still builds and passes its
tests.

## The runner

`crates/d2d/tests/conformance.rs` replays the corpus on Windows, Linux, macOS, FreeBSD and in a
browser as `wasm32`. It reports how many cases pass exactly and, for each failure, the first
coordinate and channel that differ. When the suite has a pass list, the test fails if any scene on it
stops matching.

`scripts/conformance.sh <native|wasm|wasm-browser|windows|aarch64|checks|parity|web>` drives one
target per run. The `parity` target checks that the full-output digest is identical across targets.

The `wasm32` build has no filesystem, so `scripts/gen-wasm-embed.sh` bakes the suite into
`crates/d2d/tests/embed/`. That directory is committed empty and overwritten locally. Never commit
the populated version.

The frozen value, layout and command contracts are in `crates/d2d-types/`. The reference renderer
and image-diff tooling is in `harness/`. `d2d-ref.exe` draws a scene with Microsoft's Direct2D on a
Windows machine and writes the pixels and result codes; the suite keeps those captures in
`oracle/win11-26100/`. `png-diff` compares two images.
