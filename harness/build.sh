#!/bin/sh
set -eu
cd "$(dirname "$0")"
cargo build --offline --locked --release --bin png-diff
if rustup target list --installed | grep -qx x86_64-pc-windows-gnu; then
  cargo build --offline --locked --release --target x86_64-pc-windows-gnu --bin d2d-ref
else
  # This machine has rust-src for 1.93, but no GNU target and no network access.
  python3 prepare-std.py
  __CARGO_TESTS_ONLY_SRC_ROOT="$PWD/target/rust-src/library" RUSTC_BOOTSTRAP=1 \
    CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc \
    cargo +1.93 build --offline --locked -Z build-std=std,panic_abort --release \
    --target x86_64-pc-windows-gnu --bin d2d-ref
fi
# Required notices travel with this machine's MinGW/GCC-linked executable.
cp /usr/share/licenses/mingw-w64-crt/COPYING.MinGW-w64-runtime.txt target/x86_64-pc-windows-gnu/release/
cp /usr/share/licenses/gcc/RUNTIME.LIBRARY.EXCEPTION target/x86_64-pc-windows-gnu/release/
