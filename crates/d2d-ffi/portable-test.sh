#!/bin/sh
# Portable C-ABI test (off Windows): build the d2d1 shared library for the host, compile the C
# driver against it, and run it. Proves the one COM implementation works through the raw COM ABI on
# Linux/macOS/FreeBSD and renders scene 02-fractional-rect byte-for-byte like native WARP.
# On macOS the library is libd2d1.dylib and the loader var is DYLD_LIBRARY_PATH.
set -eu
cd "$(dirname "$0")/../.."           # repo root
cargo build -p d2d-ffi ${OFFLINE:+--offline}
OUT="$(pwd)/target/debug"
cc -O2 -Wall -o "$OUT/portable-test" crates/d2d-ffi/portable-test.c -L"$OUT" -ld2d1
# Oracle lives in the conformance suite (D2D_SUITE, else the in-tree submodule, else the main checkout).
for d in "${D2D_SUITE:-}" "$(pwd)/conformance/suite" "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/conformance/suite"; do
  [ -n "$d" ] && [ -f "$d/oracle/win11-26100/02-fractional-rect.bgra" ] && ORACLE="$d/oracle/win11-26100/02-fractional-rect.bgra" && break
done
LD_LIBRARY_PATH="$OUT" DYLD_LIBRARY_PATH="$OUT" exec "$OUT/portable-test" "$ORACLE"
