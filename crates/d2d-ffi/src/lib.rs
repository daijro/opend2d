//! `d2d-ffi`: the drop-in `d2d1.dll` and its C ABI (phase 3A).
//!
//! On Windows the crate is a `cdylib` named `d2d1.dll` that exports `D2D1CreateFactory` (and the
//! other d2d1.dll entry points) and hands back real Direct2D COM objects — implemented with the
//! official `windows` IIDs/vtables via `#[implement]`, each method forwarding to the `d2d` Rust
//! core. Dropping it beside the unchanged harness `d2d-ref.exe` reproduces native d2d1's output.
//! Methods outside the exercised subset return `E_NOTIMPL`; [`win::NOT_IMPL`] lists them.
//!
//! Off Windows (Linux/macOS/FreeBSD/wasm) the same code runs: the `windows`-crate interface
//! definitions it builds on are `#[cfg(windows)]`-gated upstream, so off Windows the crate depends
//! on the `windows-shim`/`windows-core-shim` crates (windows-bindgen output over `windows-core`)
//! renamed to `windows`/`windows_core`. One COM implementation ([`win`]) serves every platform;
//! only the HWND/DXGI/GDI/print bits that need the OS at runtime are `#[cfg(windows)]`-gated and
//! return native-matching errors elsewhere.

pub use d2d::{abi, contracts, interfaces};

pub mod win;

mod exports_math;
