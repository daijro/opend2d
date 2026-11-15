#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p out/wine-builtin
export WINEPREFIX="${WINEPREFIX:-$PWD/wine-prefix}"
export WINEARCH=win64
export WINEDLLOVERRIDES='d2d1,d3d11,d3d10_1,d3d10core,dxgi,dwrite,d3dcompiler_47=b;winemenubuilder.exe='
export WINEDEBUG=+loaddll
wine --version > out/wine-builtin/wine-version.txt
# A prefix/server failure is environmental: stop before attempting every scene.
timeout 45s wineboot -u >out/wine-builtin/wineboot.log 2>&1
failed=0
for scene in scenes/*.json; do
  name=$(basename "$scene" .json)
  if timeout 45s wine target/x86_64-pc-windows-gnu/release/d2d-ref.exe \
    "$scene" "out/wine-builtin/$name.png" --wine-smoke >"out/wine-builtin/$name.log" 2>&1; then
    echo "PASS $name (Wine builtin, not reference pixels)"
  else
    echo "FAIL $name: see out/wine-builtin/$name.log"
    failed=1
  fi
done
exit "$failed"
