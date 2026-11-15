#!/usr/bin/env bash
# One-command conformance check per platform (phase 4A). Every target drives the *same* suite
# (crates/d2d/tests/conformance.rs) against the *same* oracle, so results are identical by
# construction; `parity` proves it by comparing the full-output digest across the targets runnable
# on this host. Set OFFLINE=1 to use only the local cargo cache.
#
# Usage: scripts/conformance.sh <native|wasm|wasm-browser|windows|aarch64|checks|parity|web>
set -euo pipefail
shopt -s inherit_errexit # a failed build inside $(...) must fail the run, not reuse a stale artifact
cd "$(dirname "$0")/.."

off=${OFFLINE:+--offline}
runner_env=(CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner)

# Text scenes need the shared gitignored fonts (ref/windows/Fonts). Point D2D_FONTS at them so the
# native and qemu-user runs resolve fonts the same as conformance.rs fonts_dir() would (git-derived);
# the wine run converts this to a Windows path, and wasm bakes fonts in (see gen-wasm-embed.sh).
fonts_dir() {
  local g; g=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || return 0
  local d="$(dirname "$g")/ref/windows/Fonts"; [ -d "$d" ] && printf %s "$d"
}
f=${D2D_FONTS:-$(fonts_dir)}; [ -n "$f" ] && export D2D_FONTS="$f"
# Bake the corpus and fonts into the wasm build (wasm has no filesystem). Idempotent; run before any
# wasm/web build so embed/fonts.rs matches the current ref/windows/Fonts (empty when ref/ is absent).
embed() { bash scripts/gen-wasm-embed.sh >&2; }
# Windows path (Z:\…) of the font dir for a wine run, or empty; wine reads D2D_FONTS as a native path.
wine_fonts() { [ -n "${D2D_FONTS:-}" ] && WINEPREFIX=$1 winepath -w "$D2D_FONTS" 2>/dev/null || true; }

# Build web/ to wasm and emit Node bindings into $1; echo the digest JSON field `run()` returns.
web_digest() {
  local dir=$1
  embed
  ( cd web && cargo build --target wasm32-unknown-unknown --release $off >&2 )
  wasm-bindgen --target nodejs --out-dir "$dir" \
    web/target/wasm32-unknown-unknown/release/d2d_web.wasm >&2
  node -e "const{run}=require('$dir/d2d_web.js');process.stdout.write(JSON.parse(run()).digest)"
}

# Build the suite static for aarch64 (musl, so no cross-glibc/gcc is needed) and run it under
# qemu-user; clang+lld cross-link it (the host `cc`/`ld` rejects aarch64 flags). Args pass to the test.
aarch64_run() {
  CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER=clang \
  CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_RUSTFLAGS="-Clink-arg=--target=aarch64-unknown-linux-musl -Clink-arg=-fuse-ld=lld" \
  CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_RUNNER=qemu-aarch64 \
  cargo test -p d2d --test conformance --target aarch64-unknown-linux-musl $off -- "$@"
}

case "${1:-}" in
native)
  cargo test -p d2d --test conformance $off -- --nocapture ;;

wasm)  # headless Node
  embed
  env "${runner_env[@]}" cargo test -p d2d --test conformance \
    --target wasm32-unknown-unknown $off ;;

wasm-browser)  # real headless browser
  embed
  env "${runner_env[@]}" cargo test -p d2d --test conformance \
    --target wasm32-unknown-unknown --features conformance-browser $off ;;

windows)  # x86_64-pc-windows-gnu under Wine
  cargo test -p d2d --test conformance --target x86_64-pc-windows-gnu --no-run $off
  exe=$(ls -t target/x86_64-pc-windows-gnu/debug/deps/conformance-*.exe | head -1)
  wp=${WINEPREFIX:-$HOME/.wine-d2d-4a}; wf=$(wine_fonts "$wp")
  WINEPREFIX=$wp WINEDLLOVERRIDES="mscoree,mshtml=d" WINEDEBUG=-all D2D_FONTS="$wf" \
    wine "$exe" --nocapture --test-threads=1 ;;

aarch64)  # aarch64-unknown-linux-musl, static, run under qemu-user
  aarch64_run --nocapture ;;

checks)  # compile the suite for every supported target (run happens in CI / per-OS)
  for t in aarch64-unknown-linux-gnu x86_64-apple-darwin aarch64-apple-darwin x86_64-unknown-freebsd; do
    echo "== check $t =="
    cargo check -p d2d --test conformance --target "$t" $off
  done ;;

parity)  # assert bit-identical full output across every target runnable on this host
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  native=$(cargo test -p d2d --test conformance $off -- --nocapture 2>/dev/null \
    | sed -n 's/.*output digest \(0x[0-9a-f]*\).*/\1/p')
  wasm=$(web_digest "$tmp/node" 2>/dev/null)
  win=""
  if command -v wine >/dev/null; then
    cargo test -p d2d --test conformance --target x86_64-pc-windows-gnu --no-run $off >/dev/null 2>&1
    exe=$(ls -t target/x86_64-pc-windows-gnu/debug/deps/conformance-*.exe | head -1)
    wp=${WINEPREFIX:-$HOME/.wine-d2d-4a}; wf=$(wine_fonts "$wp")
    win=$(WINEPREFIX=$wp WINEDLLOVERRIDES="mscoree,mshtml=d" WINEDEBUG=-all D2D_FONTS="$wf" \
      wine "$exe" --nocapture --test-threads=1 2>/dev/null \
      | sed -n 's/.*output digest \(0x[0-9a-f]*\).*/\1/p')
  fi
  arm=""
  if command -v qemu-aarch64 >/dev/null && command -v clang >/dev/null; then
    arm=$(aarch64_run --nocapture 2>/dev/null | sed -n 's/.*output digest \(0x[0-9a-f]*\).*/\1/p')
  fi
  echo "native(x86_64-linux): $native"
  echo "wasm32 (node)       : $wasm"
  echo "windows(wine)       : ${win:-skipped}"
  echo "aarch64 (qemu-user) : ${arm:-skipped}"
  for d in "$wasm" $win $arm; do
    [ "$d" = "$native" ] || { echo "PARITY FAIL: $d != $native"; exit 1; }
  done
  echo "PARITY OK: identical full-corpus output digest $native" ;;

web)  # build the browser page into web/pkg
  embed
  ( cd web && cargo build --target wasm32-unknown-unknown --release $off )
  wasm-bindgen --target web --out-dir web/pkg \
    web/target/wasm32-unknown-unknown/release/d2d_web.wasm
  echo "built web/pkg — serve web/ (e.g. 'python3 -m http.server -d web') and open index.html" ;;

*)
  echo "usage: $0 <native|wasm|wasm-browser|windows|aarch64|checks|parity|web>" >&2
  exit 2 ;;
esac
