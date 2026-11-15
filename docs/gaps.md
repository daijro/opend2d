# Known gaps

What is not yet byte-exact or not yet implemented, as distinct from the
[deliberate native-matching rejections](api.md#deliberate-native-matching-rejections). Behavioural
truth is [`conformance/passing.txt`](../conformance/passing.txt) (6132 of 6159 suite scenes
byte-exact; 7 skip, 20 not yet exact) and the differential fuzzer (98.1% of ~20,600 random scenes
byte-exact); this file explains the delta.

## Text — non-Latin fallback glyphs

The scenes that pass in the core but differ when driven through the drop-in DLL are non-Latin
fallback-glyph residuals in the DirectWrite path (hinted device placement for complex scripts), not a
COM/ABI fault — the same scenes differ by the same pixels whether driven by the Rust core directly or
through the exported DLLs. ClearType filtering and the coverage scan are proven byte-identical
to native; the residual is hinting/placement for fallback faces. Tracked in the core text work.

## Effects and float targets

A handful of effect and highp-float (`R16/R32` target) paths are still being matched; the 64 built-in
effects are implemented but not all are certified byte-exact on every input. `CopyFromBitmap` /
`Map` read-back of a highp-float target is not yet supported (the 4-byte BGRA/RGBA read-back path is).

## Platform build coverage (this host)

`scripts/package.sh` builds and symbol-checks what the local toolchain can link —
`x86_64-pc-windows-gnu` (MinGW) and `x86_64-unknown-linux-gnu` here — and compiles the core for
`wasm32`. The other targets (`i686`/`aarch64` Windows, `aarch64` Linux, macOS, FreeBSD) `cargo check`
cleanly but need their own linker/SDK to produce a shared library; run `package.sh` on such a host, or
add a cross toolchain, to fill `dist/` in. Cross-target *rendering* parity is already proven for every
target runnable here via `scripts/conformance.sh parity`.

## Portable (non-Wine) COM surface

The native Linux/macOS/FreeBSD `libd2d1.*` currently implements the render-and-read-back core
(factory → device context → target bitmap → geometry/solid-brush fill → map), exercised by
`crates/d2d-ffi/portable-test.c`. Text, effects and WIC save run through the Windows-ABI DLLs under
Wine today; broadening the hand-built portable vtables to the full surface is incremental work
(`crates/d2d-ffi/src/portable.rs`). On Windows and under Wine the whole COM surface above is live.

## ABI fixes made while packaging

- `ID2D1Bitmap1::CopyFromBitmap` now reads a rendered **target** bitmap's pixels (not only an uploaded
  source), so the standard offscreen read-back — render to a `D2D1_BITMAP_OPTIONS_TARGET` bitmap,
  `CopyFromBitmap` into a `CPU_READ|CANNOT_DRAW` staging bitmap, `Map` — works from a stand-alone app
  (`crates/d2d-ffi/src/win/resources.rs`). Without it the standard read-back pattern returned
  `E_NOTIMPL`.
- The off-Windows `windowscodecs` library now also exports `DllCanUnloadNow`, matching native's
  in-proc-server symbol surface (`crates/wic-ffi/src/lib.rs`).
- `windowscodecs` now exports `WICCreateImagingFactory_Proxy` (the SDK's registration-free factory
  call), so WIC can be deployed **app-local** without overwriting the system DLL — alongside a
  registration-free COM manifest and `DllGetClassObject`. All three app-local routes, and the native
  `ID2D1PathGeometry1::ComputePointAndSegmentAtLength` values, were verified by running the native and
  our DLLs under Wine (geometry needs no device).
- `ID2D1Factory1::CreatePathGeometry` (the `ID2D1PathGeometry1` overload) and
  `ID2D1PathGeometry1::ComputePointAndSegmentAtLength` are now implemented (byte-exact vs native for
  line/cubic/quadratic figures; arcs fan out to multiple segments, the known arc gap).

## `E_NOTIMPL` stub inventory (`d2d-ffi`)

A sweep of every method in `crates/d2d-ffi/src/win/**` that can return `E_NOTIMPL`, separating genuine
gaps from correct behaviour. Most `E_NOTIMPL` occurrences are **null-argument guards** — e.g.
`CreateLinearGradientBrush`/`CreateImageBrush`/`CreateBitmapBrush` do `arg.as_ref().ok_or(E_NOTIMPL)?`
and only error on a null input; those methods are implemented. The genuine whole-body stubs:

**Implemented this session (were stubs):**
- `ID2D1Factory::CreateWicBitmapRenderTarget` — a render target over an `IWICBitmap`'s memory, portable
  (no OS surface). Byte-exact vs native WARP for the present scenes through the real COM ABI under Wine
  (`present-ref` `wic` mode, `present.py`).
- `ID2D1RenderTarget::CreateBitmap` / `CreateBitmapFromWicBitmap` / `CreateBitmapBrush` (the **1.0**
  overloads) — forward to the verified `ID2D1DeviceContext` 1.1 overloads (`ID2D1Bitmap1`/
  `ID2D1BitmapBrush1` carry the 1.0 interfaces). The current harness drives the 1.1 interface, so these
  are not independently oracle-tested, but the 1.1 target they forward to is byte-exact.

**Correct stubs — native also returns `E_NOTIMPL` (or the method is optional):**
- `ID2D1EffectImpl::SetGraph` on the author support/test effects — the DDI makes `SetGraph` optional
  (only effects whose input count varies implement it); native effects return `E_NOTIMPL` too.

**By design off-Windows — implemented on Windows and under Wine, `E_NOTIMPL` only off-Windows:**
- `ID2D1Factory::CreateHwndRenderTarget` / `CreateDxgiSurfaceRenderTarget` / `CreateDCRenderTarget` /
  `CreatePrintControl`, `ID2D1DeviceContext::CreateBitmapFromDxgiSurface` / `CreateImageSourceFromDxgi`
  — each needs an OS surface (HWND / DXGI / GDI DC / XPS). The drop-in `d2d1.dll` implements them; the
  native Linux/macOS library returns `E_NOTIMPL` because those surfaces don't exist there.

**Real gaps — native implements them, not yet wired (1.0/1.1 object-returning methods):**
- `ID2D1RenderTarget::CreateSharedBitmap` — a bitmap aliasing a DXGI/WIC/memory handle.
- `ID2D1RenderTarget::CreateCompatibleRenderTarget` — an offscreen `ID2D1BitmapRenderTarget`.
- `ID2D1RenderTarget::CreateLayer` — an explicit `ID2D1Layer` (the core applies `PushLayer` params
  directly, so layered scenes render, but the handle itself isn't vended).
- `ID2D1RenderTarget::CreateMesh` — an `ID2D1Mesh` for `FillMesh`.
- `ID2D1DeviceContext::GetGradientMeshWorldBounds`.
- `ID2D1DeviceContext2/3::CreateSvgGlyphStyle` / `GetColorBitmapGlyphImage` / `GetSvgGlyphImage` —
  D2D1.3 colour/SVG glyph imaging.
- `ID2D1Factory1::GetRegisteredEffects` / `GetEffectProperties` — effect-registry introspection.
- `ID2D1PathGeometry::Stream` — replay a stored path's figures into a sink (the inverse of `Open`).

## Out of scope here

The GPU backend (`crates/d2d-wgpu`) and arbitrary pixel/compute-shader translation
(`crates/d2d-shader`) are secondary acceleration tracks behind the authoritative CPU renderer and are
not part of the drop-in parity guarantee. Paint.NET plugin hosting lives in a separate project.
