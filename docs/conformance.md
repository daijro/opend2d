# How conformance is measured

opend2d's only correctness criterion is **byte-for-byte equality with native `d2d1.dll` on Microsoft
WARP**. Nothing here is judged by eye or by a tolerance; a scene passes only when every pixel and
every per-op `HRESULT`/tag/geometry-query result equals the captured native result exactly.

## The oracle

Native `d2d1.dll` **10.0.26100.8875** (Windows 11 LTSC 26100, SHA-256 verified) renders each scene on
a **D3D11 WARP** software device (adapter vendor `0x1414`, device `0x008c`, software flag, feature
level 11_0) and the result is read back before any desktop/display colour management. Two independent
captures were byte-identical across all artifacts, so the capture is deterministic.

Each scene produces, in `conformance/oracle/win11-26100/<scene>.{bgra,json}`:

- `.bgra` — raw, tightly-packed, top-down target pixels (default premultiplied BGRA8), no header.
- `.json` — size/DPI, adapter/feature-level provenance, and the recorded per-op `HRESULT` + primitive
  tags, the `EndDraw` result, and any geometry-query answers. A scene whose native `EndDraw` fails has
  JSON only (no pixels) so negative/error behaviour is captured too.

Scenes are authored as `harness/scenes/<name>.json` and captured with `harness/vm.py` against the
reference VM. The harness (`harness/src/`) is a thin, format-stable driver — the same binary renders
on the VM (native), under Wine (our DLLs) and on every target (our core), so results are comparable
by construction.

## The three levels

1. **Core (CPU)** — `cargo test -p d2d --test conformance` drives the renderer directly against the
   oracle. `conformance/passing.txt` is the set that is byte-exact; the test fails if any listed
   scene regresses. It only ever grows. **6132 of 6159 scenes** pass today (7 skip on an op the
   `DeviceContext` API does not expose, 20 not yet byte-exact). (Refresh: run the command above and
   read the final `conformance: N pass, N skip, N fail` line.)

2. **Cross-target parity** — the *same* suite and oracle run on x86-64, `wasm32` (Node and a real
   headless browser), `aarch64` (qemu) and `x86_64-pc-windows-gnu` (Wine). `scripts/conformance.sh
   parity` asserts an identical full-output digest across every target runnable on the host, proving
   the renderer is bit-identical everywhere, not just on the dev machine.

3. **Drop-in DLL (decisive)** — the exported `d2d1.dll` + `dwrite.dll` are dropped beside the
   *unchanged* harness `d2d-ref.exe` and driven through the genuine COM ABI under Wine, comparing to
   the oracle (pixels **and** the HRESULT/tag/query channel). This is the end-to-end proof that the
   exported DLL — not just the Rust core — reproduces native. The C-API gate renders the FFI scene
   set this way: **95 of 95 byte-exact** against the oracle, and **0 of 88 differences** between the
   drop-in DLL and the Rust core on the same scenes — the DLL reproduces the core exactly through the
   ABI. The deeper `crates/d2d-ffi/decisive-test.sh` run (every passing scene through the DLL) is a
   standalone verification; its only remaining deltas are non-Latin fallback-glyph residuals in the
   core text path (see [gaps.md](gaps.md)), not the COM/ABI layer.

The consumer examples add a fourth check: `crates/d2d-ffi/examples/run-examples.sh` builds a C, C++
and Rust program against the standard API and asserts each one's readback is byte-identical to the
oracle through the raw COM ABI.

## Reproducing

```sh
scripts/conformance.sh native        # CPU core vs oracle (x86-64 host)
scripts/conformance.sh wasm          # same suite under Node
scripts/conformance.sh windows       # same suite as a Windows exe under Wine
scripts/conformance.sh parity        # assert identical digest across all host-runnable targets
crates/d2d-ffi/decisive-test.sh      # the drop-in DLL vs oracle under Wine (sandbox off)
crates/d2d-ffi/examples/run-examples.sh   # the C/C++/Rust consumer examples vs oracle
```

A one-LSB difference is a diagnostic, never a pass. Floating-point targets are compared by raw bits.
