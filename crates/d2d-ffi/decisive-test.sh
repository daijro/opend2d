#!/bin/sh
# Decisive 3A test (needs an x86_64-pc-windows-gnu toolchain + Wine): build the drop-in d2d1.dll and
# the unchanged harness d2d-ref.exe, drop the DLL beside the exe, and render every passing scene
# through the real COM ABI, comparing to the native WARP oracle. See decisive.py for the comparison.
#
# Wine needs a writable server socket, so run this with the shell sandbox disabled.
set -eu
cd "$(dirname "$0")/../.."           # repo root
T=x86_64-pc-windows-gnu
cargo build -p d2d-ffi --target $T --offline
cargo build -p dwrite-ffi --target $T --offline   # our dwrite.dll (text: DrawText/DrawGlyphRun)
cargo build --manifest-path harness/Cargo.toml --target $T --release --bin d2d-ref --offline

STAGE="$(pwd)/target/$T/decisive-stage"
mkdir -p "$STAGE"
cp "target/$T/debug/d2d1.dll" "$STAGE/"
# Stage our dwrite.dll beside the exe so the loader (exe dir first) picks it over the prefix's stale
# copy — keeps text rendering in lock-step with this build without clobbering the shared prefix.
cp "target/$T/debug/dwrite.dll" "$STAGE/"
cp "harness/target/$T/release/d2d-ref.exe" "$STAGE/"

export STAGE
export WINEPREFIX="${WINEPREFIX:-$HOME/.wine-d2dffi}"
WINEDLLOVERRIDES="mscoree,mshtml=d" WINEDEBUG=-all wineboot -u >/dev/null 2>&1 || true
# Point our dwrite.dll's system font collection at the VM's own fonts (in the main checkout's
# gitignored ref/), the same files the core/harness resolve — else CreateTextFormat can't find
# families like Arial and all text renders blank.
REF_FONTS="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/ref/windows/Fonts"
export D2D_FONTS="$(WINEDEBUG=-all winepath -w "$REF_FONTS")"
exec python3 -I crates/d2d-ffi/decisive.py
