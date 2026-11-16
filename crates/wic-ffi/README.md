# wic-ffi

This crate builds the drop-in windowscodecs.dll, the WIC COM surface that Direct2D applications
load and save images through. The `[lib]` is named `windowscodecs` and is built as `cdylib` and
`rlib`.

It exports the COM in-process server entry points DllGetClassObject and DllCanUnloadNow, plus
WICCreateImagingFactory_Proxy, the SDK's registration-free factory call. Calling
CoCreateInstance(CLSID_WICImagingFactory) returns a real IWICImagingFactory, built with
`#[implement]` on the official `windows` IIDs and vtables, that forwards to [`d2d::wic`](../d2d).
That covers the decoders, format converter, scalers, flip-rotator, clipper, and the PNG, TIFF, GIF,
BMP and JPEG encoders.

OpenD2D's WIC bitmaps implement the public IWICBitmapSource ABI, so OpenD2D's d2d1.dll can consume
them in CreateBitmapFromWicBitmap and CreateImageSourceFromWic with no extra interop.

It implements 23 WIC interfaces, plus the IClassFactory and enumerator plumbing. They cover the
factory, decoder and frame decode, encoder and frame encode, format converter, palette, scaler,
clipper, flip-rotator, stream, colour context, metadata, and pixel format and codec info.

Off Windows, the WIC definitions are gated behind `cfg(windows)`, so the library only exposes
DllGetClassObject and DllCanUnloadNow, both returning E_NOTIMPL.

## Deployment

WIC is a COM in-process server that the loader finds through an absolute registry path
(`…\System32\windowscodecs.dll`). CoCreateInstance on its own therefore will not pick up a copy
next to the application, and the protected system DLL should not be overwritten. To use this copy
app-locally, call WICCreateImagingFactory_Proxy, use a registration-free COM manifest, or call
DllGetClassObject directly (see Getting started in the top-level [README](../../README.md)). All
three are verified under Wine. Under Wine, putting it in the prefix's `system32` also works.

```sh
cargo build --release -p wic-ffi --target x86_64-pc-windows-gnu   # windowscodecs.dll
```

It is verified under Wine by the metadata and decode checks in `crates/d2d/tests/wic_com/`, and by
the consumer examples saving a PNG through WIC (`crates/d2d-ffi/examples/run-examples.sh`).

## License

Fresh Rust over the `windows` crate (MIT OR Apache-2.0). No `ref/` content is copied.
