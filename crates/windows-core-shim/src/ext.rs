//! The `windows-core` / `windows-strings` items that are `#[cfg(windows)]`-gated upstream, defined
//! here so off-Windows builds have them. ABI-identical to windows-rs (transparent pointer newtypes);
//! they are only passed to our own core, never to the OS.

macro_rules! str_ptr {
    ($name:ident, $ptr:ty, $null:expr) => {
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name(pub $ptr);
        impl $name {
            pub const fn null() -> Self {
                Self($null)
            }
            pub fn as_ptr(&self) -> $ptr {
                self.0
            }
            pub fn is_null(&self) -> bool {
                self.0.is_null()
            }
        }
        impl Default for $name {
            fn default() -> Self {
                Self::null()
            }
        }
    };
}

str_ptr!(PCWSTR, *const u16, core::ptr::null());
str_ptr!(PWSTR, *mut u16, core::ptr::null_mut());
str_ptr!(PCSTR, *const u8, core::ptr::null());
str_ptr!(PSTR, *mut u8, core::ptr::null_mut());
str_ptr!(BSTR, *const u16, core::ptr::null());

impl PCWSTR {
    /// # Safety
    /// Must point to a NUL-terminated UTF-16 string. Mirrors `windows-strings`.
    pub unsafe fn as_wide(&self) -> &[u16] {
        let mut len = 0usize;
        while unsafe { *self.0.add(len) } != 0 {
            len += 1;
        }
        unsafe { core::slice::from_raw_parts(self.0, len) }
    }
    /// # Safety
    /// See [`as_wide`](Self::as_wide).
    pub unsafe fn to_string(&self) -> Result<String, std::string::FromUtf16Error> {
        String::from_utf16(unsafe { self.as_wide() })
    }
}
impl PCSTR {
    /// # Safety
    /// Must point to a NUL-terminated byte string. Mirrors `windows-strings`.
    pub unsafe fn as_bytes(&self) -> &[u8] {
        let mut len = 0usize;
        while unsafe { *self.0.add(len) } != 0 {
            len += 1;
        }
        unsafe { core::slice::from_raw_parts(self.0, len) }
    }
    /// # Safety
    /// See [`as_bytes`](Self::as_bytes).
    pub unsafe fn to_string(&self) -> Result<String, std::str::Utf8Error> {
        std::str::from_utf8(unsafe { self.as_bytes() }).map(str::to_owned)
    }
}
impl PWSTR {
    /// # Safety
    /// See [`PCWSTR::as_wide`].
    pub unsafe fn as_wide(&self) -> &[u16] {
        let mut len = 0usize;
        while unsafe { *self.0.add(len) } != 0 {
            len += 1;
        }
        unsafe { core::slice::from_raw_parts(self.0 as *const u16, len) }
    }
    /// # Safety
    /// See [`PCWSTR::as_wide`].
    pub unsafe fn to_string(&self) -> Result<String, std::string::FromUtf16Error> {
        String::from_utf16(unsafe { self.as_wide() })
    }
}

/// `windows-core`'s `s!` (ANSI/UTF-8 string literal → NUL-terminated `PCSTR`), provided off Windows
/// where the real macro lives in the `#[cfg(windows)]`-gated `windows-strings`.
#[macro_export]
macro_rules! s {
    ($s:literal) => {
        $crate::PCSTR(::core::concat!($s, "\0").as_ptr())
    };
}

/// The handle-release trait windows-bindgen emits `impl`s for (`HANDLE`, `HGLOBAL`, GDI handles …).
pub trait Free {
    /// # Safety
    /// Releases the underlying OS handle; the handle must not be used afterwards.
    unsafe fn free(&mut self);
}

// The `TypeKind`/`Param` impls `windows-strings` provides only under `#[cfg(windows)]`, so the
// generated `fn …<P0: Param<PCWSTR>>` wrappers are well-formed. All treated as plain copy values
// (they are never marshalled to the OS off Windows); mirrors windows-strings' `PWSTR→PCWSTR` etc.
macro_rules! copy_typekind {
    ($($t:ty),*) => {$(
        impl ::windows_core::TypeKind for $t {
            type TypeKind = ::windows_core::CopyType;
        }
    )*};
}
copy_typekind!(PCWSTR, PWSTR, PCSTR, PSTR, BSTR);

impl ::windows_core::Param<PCWSTR> for PWSTR {
    unsafe fn param(self) -> ::windows_core::ParamValue<PCWSTR> {
        ::windows_core::ParamValue::Owned(PCWSTR(self.0))
    }
}
impl ::windows_core::Param<PCSTR> for PSTR {
    unsafe fn param(self) -> ::windows_core::ParamValue<PCSTR> {
        ::windows_core::ParamValue::Owned(PCSTR(self.0))
    }
}
