#!/bin/sh
# Build the native-vs-opend2d showcase: render each showcased scene through the drop-in d2d1.dll
# under Wine and compare it byte-for-byte with the native WARP oracle, emitting docs/showcase/*.png
# and the showcase table in README.md (see scripts/showcase.py for the scene list).
#
# Needs an x86_64-pc-windows-gnu toolchain + Wine + python3 with Pillow. Wine needs a writable server
# socket, so run with the shell sandbox disabled. Rerun as conformance improves. Set OFFLINE=1 for the
# local cargo cache only.
set -eu
cd "$(dirname "$0")/.."
T=x86_64-pc-windows-gnu
OFF=${OFFLINE:+--offline}

cargo build -p d2d-ffi -p dwrite-ffi -p wic-ffi --target $T $OFF
cargo build --manifest-path harness/Cargo.toml --target $T --release --bin d2d-ref $OFF

STAGE="$(pwd)/target/$T/showcase-stage"
rm -rf "$STAGE"; mkdir -p "$STAGE"
DLL="$(pwd)/target/$T/debug"
cp "$DLL/d2d1.dll" "$DLL/dwrite.dll" "$DLL/windowscodecs.dll" "$STAGE/"
cp "harness/target/$T/release/d2d-ref.exe" "$STAGE/"

export STAGE
export WINEPREFIX="${WINEPREFIX:-$HOME/.wine-d2d-showcase}"
WINEDLLOVERRIDES="mscoree,mshtml=d" WINEDEBUG=-all wineboot -u >/dev/null 2>&1 || true
cp "$DLL/windowscodecs.dll" "$WINEPREFIX/drive_c/windows/system32/" # WIC is loaded by registry path
# Point our dwrite.dll's font collection at the shared ref fonts (gitignored) so text scenes render.
REF_FONTS="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/ref/windows/Fonts"
[ -d "$REF_FONTS" ] && export D2D_FONTS="$(WINEDEBUG=-all winepath -w "$REF_FONTS" 2>/dev/null)"
export WINEDLLOVERRIDES="d2d1,dwrite,windowscodecs=n;mscoree,mshtml=d"
export WINEDEBUG=-all

exec python3 -I scripts/showcase.py
