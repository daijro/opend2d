//! Off-Windows stand-in for the `windows` crate.
//!
//! The real `windows` crate is `#![cfg(windows)]`, so off Windows it is empty and the ffi crates'
//! `win/*` (which `use windows::Win32::…` and `use windows::core::…`) would not compile. This crate
//! provides those paths from windows-bindgen output (`bindings.rs`) over `windows-core`, so the one
//! COM implementation in `win/*` builds and runs on every platform. The ffi crates depend on it
//! under the name `windows` via a Cargo `package =` rename, so no `win/*` source changes.
//!
//! On Windows this is just a re-export of the real `windows` crate.
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals, dead_code, clippy::all)]
// windows-bindgen compares vtable function pointers for interface identity — expected and fine.
#![allow(unpredictable_function_pointer_comparisons)]

#[cfg(windows)]
pub use ::windows::*;

#[cfg(not(windows))]
#[path = "bindings.rs"]
mod bindings;

// `windows::Win32::…` (bindgen nests everything under a top-level `Windows` module).
#[cfg(not(windows))]
pub use bindings::Windows::*;

// `windows::core::…` — the ffi crates import a few items (BOOL, Interface, PCWSTR …) this way.
// `::windows_core` here is the superset shim (see Cargo.toml).
#[cfg(not(windows))]
pub mod core {
    pub use ::windows_core::*;
}
