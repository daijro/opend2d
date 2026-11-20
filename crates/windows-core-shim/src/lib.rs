//! Off-Windows superset of `windows-core`.
//!
//! `windows-core` compiles everywhere, but it gates the pointer-string types (`PCWSTR`, `PWSTR`,
//! `PCSTR`, `PSTR`, `BSTR`) and the handle-`Free` trait behind `#[cfg(windows)]` (they live in the
//! likewise-gated `windows-strings`). The windows-bindgen output in `windows-shim` and the ffi
//! crates' `win/*` reference those as `windows_core::*`, so off Windows we re-export the real crate
//! and add the handful of missing items. The ffi crates consume this crate under the name
//! `windows_core` via a Cargo `package =` rename, so no source changes.
//!
//! On Windows this is just a re-export of the real `windows-core`.

pub use ::windows_core::*;

#[cfg(not(windows))]
mod ext;
#[cfg(not(windows))]
pub use ext::*;
