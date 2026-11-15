#!/bin/sh
# Present-path 3A test (needs an x86_64-pc-windows-gnu toolchain + Wine): build the drop-in d2d1.dll
# and the present-ref harness, drop the DLL beside the exe, and render the hwnd-* scenes into a GDI
# DC, an HWND window, a DXGI surface render target and a swap-chain back buffer through the real COM
# ABI, comparing each read-back to the native WARP oracle. See present.py for the comparison.
#
# Wine needs a writable server socket (and a display for the window/swap-chain targets), so run this
# with the shell sandbox disabled.
set -eu
cd "$(dirname "$0")/../.."           # repo root
T=x86_64-pc-windows-gnu
cargo build -p d2d-ffi --target $T --offline
cargo build --manifest-path harness/Cargo.toml --target $T --release --bin present-ref --offline

STAGE="$(pwd)/target/$T/present-stage"
mkdir -p "$STAGE"
cp "target/$T/debug/d2d1.dll" "$STAGE/"
cp "harness/target/$T/release/present-ref.exe" "$STAGE/"

export STAGE
export WINEPREFIX="${WINEPREFIX:-$HOME/.wine-d2dffi}"
WINEDLLOVERRIDES="mscoree,mshtml=d" WINEDEBUG=-all wineboot -u >/dev/null 2>&1 || true
exec python3 -I crates/d2d-ffi/present.py
