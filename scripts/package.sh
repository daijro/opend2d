#!/usr/bin/env bash
# Build opend2d's release artifacts for every supported target into dist/<target>/, verify each
# built library exports the entry points native's DLLs do, and write dist/manifest.txt with versions
# and SHA-256s. Targets whose linker/SDK is absent on this host are attempted and recorded as skipped
# (cross-linking Windows/macOS/FreeBSD from Linux needs their toolchains); re-run on a host that has
# them to fill dist/ in. Set OFFLINE=1 for the local cargo cache only.
#
#   scripts/package.sh [dist-dir]      # default dist/
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-dist}"
OFF=${OFFLINE:+--offline}
rm -rf "$OUT"; mkdir -p "$OUT"
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' crates/d2d-ffi/Cargo.toml | head -1)
MAN="$OUT/manifest.txt"
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
nm_exports() { # list exported symbol names of a built library, format-agnostic
  case "$1" in
    *.dll) llvm-readobj --coff-exports "$1" 2>/dev/null | sed -n 's/^  Name: \(.*\)/\1/p' | grep -v '\.dll$' ;;
    *)     llvm-nm -D --defined-only "$1" 2>/dev/null | awk '{print $NF}' ;;
  esac
}
# Entry points native's DLLs export that a drop-in must provide (the documented d2d1.dll projection
# minus the deferred GradientMesh-CoonsPatch helper; dwrite's factory; WIC's COM in-proc entries).
req_d2d1="D2D1CreateFactory D2D1CreateDevice D2D1CreateDeviceContext D2D1ComputeMaximumScaleFactor D2D1ConvertColorSpace D2D1InvertMatrix D2D1IsMatrixInvertible D2D1MakeRotateMatrix D2D1MakeSkewMatrix D2D1SinCos D2D1Tan D2D1Vec3Length"
req_dwrite="DWriteCreateFactory"
req_wic="DllGetClassObject DllCanUnloadNow WICCreateImagingFactory_Proxy"

{ echo "opend2d $VERSION — release artifact manifest"
  echo "generated $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(uname -srm)"
  echo "rustc $(rustc --version 2>/dev/null | cut -d' ' -f2)"
  echo; } > "$MAN"

# target | pretty library filenames for that platform
targets="x86_64-pc-windows-gnu:dll i686-pc-windows-gnu:dll aarch64-pc-windows-msvc:dll \
         x86_64-unknown-linux-gnu:so aarch64-unknown-linux-gnu:so \
         x86_64-apple-darwin:dylib aarch64-apple-darwin:dylib x86_64-unknown-freebsd:so"

libnames() { # echo the three library filenames for a platform kind
  case "$1" in
    dll)   echo "d2d1.dll dwrite.dll windowscodecs.dll" ;;
    so)    echo "libd2d1.so libdwrite.so libwindowscodecs.so" ;;
    dylib) echo "libd2d1.dylib libdwrite.dylib libwindowscodecs.dylib" ;;
  esac
}

check() { # check <built-lib> <required-symbols> ; echo ok/MISSING:...
  local f=$1; shift
  local have; have=$(nm_exports "$f")
  local miss=""
  for s in "$@"; do echo "$have" | grep -qx "$s" || miss="$miss $s"; done
  [ -z "$miss" ] && echo "exports ok" || echo "MISSING:$miss"
}

for entry in $targets; do
  t="${entry%%:*}"; kind="${entry##*:}"
  echo "== $t =="
  if cargo build --release --target "$t" $OFF -p d2d-ffi -p dwrite-ffi -p wic-ffi >/dev/null 2>&1; then
    d="$OUT/$t"; mkdir -p "$d"
    set -- $(libnames "$kind")
    built="target/$t/release"
    # cdylib names: our [lib] names are d2d1/dwrite/windowscodecs (no 'lib' on windows, 'lib' elsewhere)
    for pair in "d2d1:$1:$req_d2d1" "dwrite:$2:$req_dwrite" "windowscodecs:$3:$req_wic"; do
      base="${pair%%:*}"; rest="${pair#*:}"; fname="${rest%%:*}"; reqs="${rest#*:}"
      src=$(ls "$built"/*"$base".dll "$built"/lib"$base".so "$built"/lib"$base".dylib 2>/dev/null | head -1 || true)
      [ -n "$src" ] || continue
      cp "$src" "$d/$fname"
      printf '%-28s %-24s %s  %s\n' "$t" "$fname" "$(sha "$d/$fname")" "$(check "$d/$fname" $reqs)" | tee -a "$MAN"
    done
    cp -r crates/d2d-ffi/examples "$d/examples"
  else
    printf '%-28s skipped (no linker/SDK on this host; build on a %s host)\n' "$t" "$t" | tee -a "$MAN"
  fi
done

echo "== wasm32-unknown-unknown =="
if cargo build --release --target wasm32-unknown-unknown $OFF -p d2d >/dev/null 2>&1; then
  d="$OUT/wasm32-unknown-unknown"; mkdir -p "$d"
  cp target/wasm32-unknown-unknown/release/libd2d.rlib "$d/" 2>/dev/null || true
  printf '%-28s %s  CPU renderer rlib — link into a wasm-bindgen app (see web/)\n' wasm32-unknown-unknown "$(sha "$d/libd2d.rlib" 2>/dev/null || echo -)" | tee -a "$MAN"
else
  printf '%-28s skipped\n' wasm32-unknown-unknown | tee -a "$MAN"
fi

# Consumer kit: the example sources and a one-line pointer to the headers/bindings a consumer needs.
{ echo "Headers/bindings per consumption path (opend2d ships none of its own — it IS Direct2D):"
  echo "  C / C++  : stock Windows headers d2d1.h/d2d1_1.h, dwrite.h, wincodec.h (MinGW-w64 or the Windows SDK; Wine on Linux). See examples/example.c, examples/example.cpp."
  echo "  Rust     : the 'windows' crate (Windows targets). See examples/example.rs."
  echo "  Rust/CPU : depend on the 'd2d' crate directly (all platforms incl. wasm32)."
} > "$OUT/CONSUMING.txt"

echo; echo "wrote $OUT/ (manifest: $MAN)"
