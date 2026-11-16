# How conformance is measured

OpenD2D has one measure of correctness: its output has to be identical, byte for byte, to what
Direct2D produces on Windows for the same scene. Nothing is judged by eye or within a tolerance. A
scene passes only when every pixel matches and every result Direct2D reports (the HRESULT of each
call, the tag recorded with it and the answer to any geometry query) matches too. A difference of
one in the lowest bit counts as a failure. Float targets are compared by their raw bits.

## The Windows captures

The reference output comes from Direct2D's d2d1.dll version 10.0.26100.8875 on Windows 11 LTSC
26100, drawing on the WARP software renderer. Each scene is captured as its raw pixels plus a small
JSON file with the size, the DPI, the device used and every result code. When Direct2D's EndDraw
fails, only the JSON file is kept, so error behaviour is tested as well. Two separate capture runs
gave identical files, so the captures are repeatable.

The captures and the list of passing scenes are kept privately and are not part of this repository.
The test programs that compare against them are.

## What is checked

The renderer itself. The conformance test draws each scene with the Rust renderer and compares it
with the capture. A scene that matches is added to the pass list, and the test fails if any listed
scene stops matching, so the list only grows. Today 6132 of 6159 scenes match. 7 are skipped
because they use an operation the DeviceContext API doesn't offer, and 20 don't match yet.

Every platform. The same suite runs on x86-64, as WebAssembly (in Node and in a headless browser),
on aarch64 under qemu, and as a Windows program under Wine. `scripts/conformance.sh parity` checks
that every platform the machine can run produces exactly the same output.

The drop-in DLLs. OpenD2D's d2d1.dll and dwrite.dll are placed next to an unchanged Windows test
program and driven through the real COM interface under Wine, and the results are compared with the
captures in the same way. 95 of 95 scenes in that set match Direct2D, and the DLLs produce exactly
the same output as the Rust renderer on all 88 scenes compared, so the DLL layer adds no
differences. `crates/d2d-ffi/decisive-test.sh` runs every passing scene through the DLLs; the only
differences left there are the fallback-font text cases described in [gaps.md](gaps.md).

The examples. `crates/d2d-ffi/examples/run-examples.sh` builds small C, C++ and Rust programs
against the standard API and checks that what each one reads back matches the capture.

The fuzzer. Besides the hand-written scenes, a fuzzer generates random scenes and compares them in
the same way. 98.1% of about 20,600 such scenes match. The scene generator and the differ are in `tools/fuzz`
and `harness`; capturing new scenes needs a Windows machine (see
[tools/fuzz/README.md](../tools/fuzz/README.md)).

## Running the checks

These need the Windows captures, pointed to by `D2D_SUITE`.

```sh
scripts/conformance.sh native        # the renderer on x86-64
scripts/conformance.sh wasm          # the same suite under Node
scripts/conformance.sh windows       # the same suite as a Windows program under Wine
scripts/conformance.sh parity        # check every platform gives the same output
crates/d2d-ffi/decisive-test.sh      # the drop-in DLLs under Wine
crates/d2d-ffi/examples/run-examples.sh   # the C, C++ and Rust examples
```
