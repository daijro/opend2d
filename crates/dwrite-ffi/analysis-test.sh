#!/bin/sh
# Decisive glyph-run-analysis probe (needs an x86_64-pc-windows-gnu toolchain + Wine): build our
# drop-in dwrite.dll and the unchanged harness dwrite-dump.exe, stage them, and run dwrite-dump
# through our dwrite.dll under Wine for the committed native glyph-run-analysis fixtures, comparing
# the alpha textures + bounds byte-for-byte. See analysis.py.
#
# Wine needs a writable server socket, so run this with the shell sandbox disabled.
set -eu
cd "$(dirname "$0")/../.."           # repo root
T=x86_64-pc-windows-gnu
cargo build -p dwrite-ffi --target $T --offline
cargo build -p dwrite-ffi --target $T --offline --bin dwrite-probe
cargo build --manifest-path harness/Cargo.toml --target $T --release --bin dwrite-dump --offline

STAGE="$(pwd)/target/$T/dwrite-analysis-stage"
mkdir -p "$STAGE"
cp "target/$T/debug/dwrite.dll" "target/$T/debug/dwrite-probe.exe" "$STAGE/"
cp "harness/target/$T/release/dwrite-dump.exe" "$STAGE/"

# Fonts: the VM's own files live in the main checkout's gitignored ref/ (absent from worktrees); the
# system font collection scans that directory, as the fixtures were captured against.
REF_FONTS="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/ref/windows/Fonts"

export STAGE
export WINEPREFIX="${WINEPREFIX:-$HOME/.wine-d2dffi}"
WINEDLLOVERRIDES="mscoree,mshtml=d" WINEDEBUG=-all wineboot -u >/dev/null 2>&1 || true
export D2D_FONTS="$(WINEDEBUG=-all winepath -w "$REF_FONTS")"
exec python3 -I crates/dwrite-ffi/analysis.py
