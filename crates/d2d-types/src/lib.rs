//! Portable (native + `wasm32-unknown-unknown`) definitions of the Direct2D, DirectWrite, WIC and
//! DXGI value types, enums, constants and GUIDs/CLSIDs ([`abi`]), the public-interface inventory
//! (method order, slots, IIDs — [`interfaces`]), and the frozen cross-crate [`contracts`].
//!
//! [`abi`] and [`interfaces`] are `@generated` by `gen/` from the pinned `windows 0.62.2` Win32
//! source (MIT OR Apache-2.0; its MIT notice is retained in `README.md`). Do not edit them by
//! hand — change the generator and rerun it (see `README.md`). Everything else here is written by
//! hand and is tiny.

pub mod abi;
pub mod contracts;
pub mod interfaces;

/// One newly-declared method of a COM interface, in vtable order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Method {
    pub name: &'static str,
    /// Absolute vtable slot (IUnknown occupies 0..3).
    pub slot: u32,
    pub support: Support,
}

/// A COM interface's identity and method layout. `slots` is the full vtable length including
/// inherited methods; `methods` are only those newly declared by this interface.
#[derive(Clone, Copy, Debug)]
pub struct Interface {
    pub name: &'static str,
    pub iid: windows_core::GUID,
    pub slots: u32,
    pub base: Option<&'static Interface>,
    pub methods: &'static [Method],
}

/// Implementation status of a method. Phase 0A freezes the inventory only, so every method is
/// `Unsupported`; later phases add variants as behaviour lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Support {
    Unsupported,
}

/// Portable stand-ins for the foreign Win32 types the emitted value structs actually reference.
/// Layouts match the `windows` crate on x64; `ComPtr` replaces every COM-interface-typed field
/// (all pointer-sized). Only these nine are used — vtable-only types are not emitted, so they are
/// not provided here.
#[allow(non_snake_case)]
pub mod prelude {
    use core::ffi::c_void;

    /// Pointer-sized wrapper: a transparent pointer with a null (`zeroed`) `Default`. Covers the
    /// opaque COM interface stand-in (`ComPtr`), the referenced Win32 handles, and the string
    /// pointers windows-core only re-exports on Windows.
    macro_rules! ptr {
        ($($n:ident($t:ty)),* $(,)?) => {$(
            #[repr(transparent)]
            #[derive(Clone, Copy, Debug, PartialEq, Eq)]
            pub struct $n(pub $t);
            impl Default for $n {
                fn default() -> Self { unsafe { core::mem::zeroed() } }
            }
        )*};
    }
    ptr! {
        ComPtr(*mut c_void), HANDLE(*mut c_void), HWND(*mut c_void), HMONITOR(*mut c_void),
        PCWSTR(*const u16), PCSTR(*const u8),
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct RECT { pub left: i32, pub top: i32, pub right: i32, pub bottom: i32 }
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct POINT { pub x: i32, pub y: i32 }
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct LUID { pub LowPart: u32, pub HighPart: i32 }
}
