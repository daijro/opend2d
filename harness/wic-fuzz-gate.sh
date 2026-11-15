#!/bin/sh
# WIC decoder no-panic fuzz with integer overflow checks forced ON in a RELEASE build.
#
# The normal gate runs the fuzz in release, where an out-of-range integer operation (an oversized
# shift, an add that wraps u8, a multiply that wraps usize) is silent: it produces a wrong value
# instead of erroring, so a robustness bug can hide. Building release with `-C overflow-checks=on`
# turns that whole class into a panic the gate catches, while keeping release codegen. A dedicated
# target dir keeps these overflow-checked artifacts from clobbering the normal release build cache.
#
# Usage: harness/wic-fuzz-gate.sh         (defaults below; override via env)
#        FUZZ_ITERS=200000 WIC_FUZZ_SEEDS="1 2 3 4 5 6 7 8" harness/wic-fuzz-gate.sh
set -eu
cd "$(dirname "$0")/.."                                   # repo root (this script lives in harness/)

SEEDS="${WIC_FUZZ_SEEDS:-1 2 3 4}"
export FUZZ_ITERS="${FUZZ_ITERS:-50000}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/wic-fuzz-overflow}"
export RUSTFLAGS="${RUSTFLAGS:-} -C overflow-checks=on"

echo "wic-fuzz-gate: release + overflow-checks, ${FUZZ_ITERS} iters/seed, seeds: ${SEEDS}"
# Build once so the per-seed loop only re-runs.
cargo test -p d2d --release --test wic_fuzz wic_decoders_never_panic --no-run
for s in ${SEEDS}; do
    echo "--- seed ${s} ---"
    FUZZ_SEED="${s}" cargo test -p d2d --release --test wic_fuzz wic_decoders_never_panic -- --exact
done
echo "wic-fuzz-gate: OK"
