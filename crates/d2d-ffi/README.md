# d2d-ffi

This crate builds the drop-in d2d1.dll for Windows and Wine, and libd2d1 (`.so` or `.dylib`) for
Linux, macOS and FreeBSD, together with its C and COM ABI. The `[lib]` is named `d2d1` and is built
both as a `cdylib`, which is the loadable library, and as an `rlib`, so the in-process conformance
test can drive the same objects without the OS loader.

It is a thin shell. The COM methods forward to the [`d2d`](../d2d) core, which produces the pixels.

The same COM implementation (`src/win/`) is built for every platform. It provides real Direct2D
COM objects, built with `#[implement]` on the official IIDs and vtables from the `windows` crate (off
Windows, from the `windows-shim` and `windows-core-shim` crates). It also has the D2D1CreateFactory,
D2D1CreateDevice and D2D1CreateDeviceContext exports and the matrix and colour helper exports
(`src/exports_math.rs`). It implements about 55 interfaces.

On Linux, macOS and FreeBSD, libd2d1 has the full COM API, including text, effects and WIC, and
needs nothing from Microsoft. Only the methods that need a window, a GDI device context, DXGI or a
printer are left out there (`src/win/present.rs` and `src/win/print.rs` are Windows only).

## Build and verify

```sh
cargo build --release -p d2d-ffi --target x86_64-pc-windows-gnu   # d2d1.dll
cargo build --release -p d2d-ffi                                  # libd2d1.so
sh portable-test.sh                           # raw C ABI on Linux, macOS or FreeBSD
sh decisive-test.sh                           # drop-in d2d1.dll vs the Windows captures, under Wine
sh examples/run-examples.sh                   # C, C++ and Rust examples vs the Windows captures
```

`examples/` has C, C++ and Rust programs that use the standard Direct2D, DirectWrite and WIC API
exactly as on Windows: they create a factory and a render target, then draw geometry with a brush,
text and an effect, and save a PNG through WIC. Each one asserts that its output equals the Windows
capture byte for byte. `portable-test.c` is a raw ABI driver that needs no headers; it draws one
scene through the COM vtables and checks a by-value return. decisive-test.sh and run-examples.sh
compare against the Windows captures, which are not in this repository; point `D2D_SUITE` at them.

## License

Fresh Rust over the `windows` crate's COM machinery (MIT OR Apache-2.0). No `ref/` content is
copied.
