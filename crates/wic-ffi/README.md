# wic-ffi

The drop-in **`windowscodecs.dll`** — the WIC COM surface Direct2D apps load and save images through.
The `[lib]` is named `windowscodecs` (`cdylib` + `rlib`). It exports the COM in-proc-server entry
points `DllGetClassObject` / `DllCanUnloadNow` plus `WICCreateImagingFactory_Proxy` (the SDK's
registration-free factory call); `CoCreateInstance(CLSID_WICImagingFactory)` hands back
a real `IWICImagingFactory` (`#[implement]` with the official `windows` IIDs/vtables), forwarding to
[`d2d::wic`](../d2d) — decoders, format converter, scalers, flip-rotator, clipper, and the
PNG/TIFF/GIF/BMP/JPEG encoders.

Our WIC bitmaps implement the public `IWICBitmapSource` ABI, so our `d2d1.dll`'s
`CreateBitmapFromWicBitmap` / `CreateImageSourceFromWic` consume them with no extra interop.

- 23 WIC interfaces (factory, decoder/frame-decode, encoder/frame-encode, format converter, palette,
  scaler/clipper/flip-rotator, stream, colour context, metadata, pixel-format/codec info) plus the
  `IClassFactory`/enumerator plumbing.
- Off Windows the WIC definitions are `cfg(windows)`-gated, so the library exposes only
  `DllGetClassObject`/`DllCanUnloadNow` returning `E_NOTIMPL`.

**Deployment note:** WIC is a COM in-process server the loader finds by the absolute registry path
(`…\System32\windowscodecs.dll`), so `CoCreateInstance` alone won't pick up an app-local copy, and the
protected system DLL should not be overwritten. Deploy app-local via `WICCreateImagingFactory_Proxy`,
a registration-free COM manifest, or `DllGetClassObject` directly (README quick start (a); all verified
under Wine). Under Wine, dropping it in the prefix's `system32` also works.

```sh
cargo build --release -p wic-ffi --target x86_64-pc-windows-gnu   # windowscodecs.dll
```

Verified under Wine by the metadata/decode oracle (`crates/d2d/tests/wic_com/`) and the consumer
examples' WIC-PNG save (`crates/d2d-ffi/examples/run-examples.sh`).

## License

Fresh Rust over the `windows` crate (MIT OR Apache-2.0). No `ref/` content is copied.
