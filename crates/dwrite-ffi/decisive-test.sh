#!/bin/sh
# Decisive drop-in-dwrite test (needs an x86_64-pc-windows-gnu toolchain + Wine): build our drop-in
# dwrite.dll + d2d1.dll and the unchanged harness d2d-ref.exe, drop them in a stage dir, and render
# every text scene that is byte-exact in the core through the real DirectWrite + Direct2D COM ABI,
# comparing to the native WARP oracle. See decisive.py.
#
# Wine needs a writable server socket, so run this with the shell sandbox disabled.
set -eu
cd "$(dirname "$0")/../.."           # repo root
T=x86_64-pc-windows-gnu
cargo build -p dwrite-ffi --target $T --offline
cargo build -p d2d-ffi --target $T --offline
cargo build --manifest-path harness/Cargo.toml --target $T --release --bin d2d-ref --offline

STAGE="$(pwd)/target/$T/dwrite-decisive-stage"
mkdir -p "$STAGE"
cp "target/$T/debug/d2d1.dll" "target/$T/debug/dwrite.dll" "$STAGE/"
cp "harness/target/$T/release/d2d-ref.exe" "$STAGE/"

# Fonts: the VM's own files live in the main checkout's gitignored ref/ (absent from worktrees); the
# system font collection scans that directory (the harness/core use the same files).
REF_FONTS="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/ref/windows/Fonts"

export STAGE
export WINEPREFIX="${WINEPREFIX:-$HOME/.wine-d2dffi}"
WINEDLLOVERRIDES="mscoree,mshtml=d" WINEDEBUG=-all wineboot -u >/dev/null 2>&1 || true
export D2D_FONTS="$(WINEDEBUG=-all winepath -w "$REF_FONTS")"
exec python3 -I crates/dwrite-ffi/decisive.py
