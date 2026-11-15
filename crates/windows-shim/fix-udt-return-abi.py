#!/usr/bin/env python3
"""Post-process windows-bindgen output (src/bindings.rs) so COM getters that return a WinRT numerics
value struct by value use the MSVC member-function ABI (a hidden result pointer as the second
argument) instead of a by-value return.

windows-bindgen emits `-> windows_numerics::Vector2` for the D2D1_POINT_2F getters
(GetStartPoint/GetEndPoint/GetCenter/GetGradientOriginOffset). Real COM callers — native d2d1 (MSVC)
and TerraFX — use the member-function struct-return convention: the callee writes `*result` (2nd arg)
and the slot is read back. On x86_64-pc-windows-gnu the by-value form is mis-lowered (the value never
reaches the caller); the out-param form is what windows-bindgen already uses for every Win32 POD
struct return (GetSize, GetColor, …), so this just makes these four match. Idempotent; run after
regenerating bindings.rs (see README).
"""
import re
import sys
import pathlib

BINDINGS = pathlib.Path(__file__).with_name("src") / "bindings.rs"
METHODS = ("GetStartPoint", "GetEndPoint", "GetCenter", "GetGradientOriginOffset")
VEC = "windows_numerics::Vector2"

src = BINDINGS.read_text()
changed = 0
for m in METHODS:
    # 1. vtable struct field: `pub M: unsafe extern "system" fn( *mut c_void, ) -> Vec,`
    field = re.compile(
        r'pub ' + m + r':\s*unsafe extern "system" fn\(\s*\*mut core::ffi::c_void,\s*\)\s*-> '
        + re.escape(VEC) + r',')
    src, n = field.subn(
        f'pub {m}: unsafe extern "system" fn(*mut core::ffi::c_void, *mut {VEC}),', src)
    changed += n
    # 2. call wrapper: forward a zeroed result slot, then return it.
    call = re.compile(
        r'(pub unsafe fn ' + m + r'\(&self\) -> ' + re.escape(VEC) + r' \{\s*unsafe \{\s*)'
        r'\(windows_core::Interface::vtable\(self\)\.' + m + r'\)\(\s*'
        r'windows_core::Interface::as_raw\(self\),\s*\)(\s*\}\s*\})')
    src, n = call.subn(
        r'\1let mut result__ = core::mem::zeroed(); '
        r'(windows_core::Interface::vtable(self).' + m + r')('
        r'windows_core::Interface::as_raw(self), &mut result__); result__\2', src)
    changed += n
    # 3. #[implement] thunk: write the impl's return through the result pointer.
    thunk = re.compile(
        r'(unsafe extern "system" fn ' + m + r'<\s*Identity: (\w+),\s*const OFFSET: isize,\s*>\(\s*'
        r'this: \*mut core::ffi::c_void,\s*)\) -> ' + re.escape(VEC) + r' \{'
        r'(\s*unsafe \{\s*let this: &Identity = [^;]*?;\s*)'
        r'(\2::' + m + r'\(this\))(\s*\}\s*\})')
    src, n = thunk.subn(r'\1result__: *mut ' + VEC + r') {\3*result__ = \4\5', src)
    changed += n

BINDINGS.write_text(src)
print(f"rewrote {changed} sites across {len(METHODS)} methods "
      f"(expect {len(METHODS)*3} on a fresh bindings.rs, 0 if already applied)")
sys.exit(0 if changed in (0, len(METHODS) * 3) else 1)
