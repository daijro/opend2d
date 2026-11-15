//! `wic-ffi`: the drop-in `windowscodecs.dll` and the WIC COM surface Direct2D apps load and save
//! images through (phase: drop-in windowscodecs over `d2d::wic`).
//!
//! On Windows the crate is a `cdylib` named `windowscodecs.dll` that exports the COM in-proc-server
//! entry points (`DllGetClassObject`/`DllCanUnloadNow`). `CoCreateInstance(CLSID_WICImagingFactory)`
//! hands back a real `IWICImagingFactory`, implemented with the official `windows` IIDs/vtables via
//! `#[implement]` and forwarding to the `d2d::wic` Rust core (decoders, format converter, scalers,
//! flip-rotator, clipper, JPEG encoder). Dropping it beside the unchanged harness `wic-dump.exe`
//! reproduces native WIC output byte-for-byte.
//!
//! Our WIC bitmaps implement the public `IWICBitmapSource` ABI (same `windows`-crate vtable both
//! DLLs use), so our `d2d1.dll`'s `CreateBitmapFromWicBitmap` / `CreateImageSourceFromWic` consume
//! them with no extra interop — WIC's own public ABI already carries the pixels.
//!
//! Off Windows the `windows` interface definitions are `#[cfg(windows)]`-gated upstream, so the
//! crate builds on the `windows-shim`/`windows-core-shim` crates (renamed to `windows`/`windows_core`)
//! — the one COM implementation in [`win`] serves every platform.

pub mod win;
