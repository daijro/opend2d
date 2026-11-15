#!/bin/sh
# Measure current rendering performance and print the markdown table for docs/perf.md and the README.
#
# One build of each target (native x86_64 and wasm32 Node, simd128), then the minimum wall time over
# REPS renders of each photo-size scene — native and wasm back-to-back per scene so both see the same
# machine state. Run on a QUIET machine (no other heavy jobs) for clean numbers; min-of-reps drops the
# slow outliers but cannot hide sustained contention.
#
# Usage:  REPS=9 scripts/perf-bench.sh
# Needs:  a wasm32 target + wasm-bindgen + node; the bench generator (scratch/gen-bench-scenes.py).
set -eu
cd "$(dirname "$0")/.."
REPS=${REPS:-9}
SCENES=${SCENES:-/tmp/bench-scenes}

echo "generating scenes -> $SCENES" >&2
python3 -I scratch/gen-bench-scenes.py "$SCENES" >&2

echo "building native bench" >&2
cargo test -p d2d --test conformance --release --no-run >&2
BIN=$(ls -t target/release/deps/conformance-* | grep -v '\.d$' | head -1)

echo "building wasm (simd128, nodejs)" >&2
sh scripts/gen-wasm-embed.sh >&2 2>/dev/null || true
( cd web && RUSTFLAGS="-C target-feature=+simd128" cargo build --target wasm32-unknown-unknown --release >&2 )
WASM=$(ls -t web/target/wasm32-unknown-unknown/release/d2d_web.wasm target/wasm32-unknown-unknown/release/d2d_web.wasm 2>/dev/null | head -1)
wasm-bindgen --target nodejs --out-dir /tmp/wasmpkg "$WASM" >&2

echo "timing (REPS=$REPS)" >&2
# native: "BENCH scenes <us>us <w>x<h> <name>" -> "<name> <us>"
D2D_BENCH=scenes D2D_BENCH_SCENES="$SCENES" D2D_BENCH_REPS="$REPS" "$BIN" bench --nocapture 2>&1 \
  | sed -nE 's/.*BENCH scenes[[:space:]]+([0-9]+)us[[:space:]]+[0-9]+x[0-9]+[[:space:]]+(.*)/\2 \1/p' \
  | sort > /tmp/perf-native.txt
# wasm: "<name>.json wasm-min <ms> bytes <n>" -> "<name> <ms>" (strip .json to match the native stem)
node scripts/wasm-bench.mjs /tmp/wasmpkg "$REPS" "$SCENES"/*.json 2>/dev/null \
  | sed -nE 's/^([^[:space:]]+)\.json wasm-min ([0-9.]+) .*/\1 \2/p' \
  | sort > /tmp/perf-wasm.txt

# join and emit the markdown table (native us -> ms, 1 decimal)
echo "| scene | native (ms) | wasm (ms) |"
echo "|-------|------------:|----------:|"
join -a1 /tmp/perf-native.txt /tmp/perf-wasm.txt | while read -r name us wms; do
  nms=$(awk "BEGIN{printf \"%.1f\", $us/1000}")
  printf "| %s | %s | %s |\n" "$name" "$nms" "${wms:-—}"
done
