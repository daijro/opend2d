#!/bin/sh
# Build and run the opend2d consumer examples and compare their output to the native WARP oracle.
#
# Proves a project consuming opend2d the standard way gets native-identical pixels through the raw
# COM ABI: each example renders harness scene 02-fractional-rect op-for-op and asserts its readback
# equals the suite's oracle/win11-26100/02-fractional-rect.bgra, then exercises geometry, text, an
# effect and a WIC PNG save. The C and C++ programs are built with stock MinGW-w64 Windows headers;
# the Rust program uses the `windows` crate. All run under Wine against our staged drop-in DLLs
# (d2d1.dll, dwrite.dll, windowscodecs.dll) — the exact Windows drop-in deployment, cross-checked.
#
# Needs an x86_64-pc-windows-gnu toolchain + MinGW-w64 gcc/g++ + Wine. Wine needs a writable server
# socket, so run this with the shell sandbox disabled. Exit 0 and "EXAMPLES OK" when all pass.
set -eu
cd "$(dirname "$0")/../../.."          # repo root
T=x86_64-pc-windows-gnu
OFF=${OFFLINE:+--offline}

cargo build -p d2d-ffi -p dwrite-ffi -p wic-ffi --target $T $OFF
cargo build -p d2d-ffi --example example --target $T $OFF

STAGE="$(pwd)/target/$T/examples-stage"
rm -rf "$STAGE"; mkdir -p "$STAGE"
DLL="$(pwd)/target/$T/debug"
cp "$DLL/d2d1.dll" "$DLL/dwrite.dll" "$DLL/windowscodecs.dll" "$STAGE/"
cp "$DLL/examples/example.exe" "$STAGE/example-rs.exe"
x86_64-w64-mingw32-gcc -O2 -o "$STAGE/example-c.exe"   crates/d2d-ffi/examples/example.c   -ld2d1 -ldwrite -lole32
x86_64-w64-mingw32-g++ -O2 -o "$STAGE/example-cpp.exe" crates/d2d-ffi/examples/example.cpp -ld2d1 -ldwrite -lwindowscodecs -lole32 -luuid
S=${D2D_SUITE:-conformance/suite}; [ -d "$S/oracle" ] || S="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/conformance/suite"
cp "$S/oracle/win11-26100/02-fractional-rect.bgra" "$STAGE/oracle.bgra"

export WINEPREFIX="${WINEPREFIX:-$HOME/.wine-d2d-examples}"
WINEDLLOVERRIDES="mscoree,mshtml=d" WINEDEBUG=-all wineboot -u >/dev/null 2>&1 || true
# windowscodecs is a COM in-proc server loaded by absolute path from the registry, so the drop-in
# copy must live in system32 (unlike d2d1/dwrite, which load by name from the exe dir).
SYS32="$WINEPREFIX/drive_c/windows/system32"
cp "$DLL/d2d1.dll" "$DLL/dwrite.dll" "$DLL/windowscodecs.dll" "$SYS32/"
# Point our dwrite.dll's system font collection at the shared ref fonts (gitignored, main checkout),
# else CreateTextFormat can't find Arial and text renders blank (parity still holds — no text there).
REF_FONTS="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/ref/windows/Fonts"
[ -d "$REF_FONTS" ] && export D2D_FONTS="$(WINEDEBUG=-all winepath -w "$REF_FONTS" 2>/dev/null)"

cd "$STAGE"
ov="d2d1,dwrite,windowscodecs=n;mscoree,mshtml=d"
rc=0
for ex in example-c example-cpp example-rs; do
  echo "== $ex =="
  if WINEDLLOVERRIDES="$ov" WINEDEBUG=-all wine "$ex.exe" oracle.bgra . 2>&1 | grep -q "byte-exact vs native oracle"; then
    echo "  PASS (byte-exact vs native oracle)"
  else
    echo "  FAIL"; rc=1
  fi
done
[ $rc -eq 0 ] && echo "EXAMPLES OK (PNGs in $STAGE)" || echo "EXAMPLES FAILED"
exit $rc
