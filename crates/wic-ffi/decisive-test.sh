#!/bin/sh
# Decisive drop-in-windowscodecs test (needs an x86_64-pc-windows-gnu toolchain + Wine): build our
# drop-in windowscodecs.dll and the unchanged harness wic-dump.exe, stage them beside a fresh Wine
# prefix, register our DLL for the WIC CLSIDs, and run wic-dump over the committed WIC fixtures
# through the real WIC COM ABI, comparing byte-for-byte to the oracle captures. See decisive.py.
#
# Wine needs a writable server socket, so run this with the shell sandbox disabled.
set -eu
cd "$(dirname "$0")/../.."           # repo root
T=x86_64-pc-windows-gnu
cargo build -p wic-ffi --target $T --offline
cargo build --manifest-path harness/Cargo.toml --target $T --release --bin wic-dump --offline

STAGE="$(pwd)/target/$T/wic-decisive-stage"
mkdir -p "$STAGE"
cp "target/$T/debug/windowscodecs.dll" "$STAGE/"
cp "harness/target/$T/release/wic-dump.exe" "$STAGE/"

export STAGE
export WINEPREFIX="${WINEPREFIX:-$HOME/.wine-wicffi}"
export WINEDEBUG=-all
WINEDLLOVERRIDES="mscoree,mshtml=d" wineboot -u >/dev/null 2>&1 || true

# True drop-in: replace the prefix's windowscodecs.dll with ours. The prefix already registers the
# WIC CLSIDs (imaging factory, PNG decoder, …) against this path, so `windowscodecs=n` (decisive.py)
# makes CoCreateInstance load our native DLL and reach our DllGetClassObject.
cp "$STAGE/windowscodecs.dll" "$WINEPREFIX/drive_c/windows/system32/windowscodecs.dll"

exec python3 -I crates/wic-ffi/decisive.py
