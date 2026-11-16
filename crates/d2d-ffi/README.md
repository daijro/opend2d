# d2d-ffi

This crate builds the drop-in d2d1.dll for Windows and Wine, and libd2d1 (`.so` or `.dylib`) for
Linux, macOS and FreeBSD, together with its C and COM ABI. The `[lib]` is named `d2d1` and is built
both as a `cdylib`, which is the loadable library, and as an `rlib`, so the in-process conformance
test can drive the same objects without the OS loader.

It is a thin shell. The COM methods forward to the [`d2d`](../d2d) core, which produces the pixels.

On Windows and Wine (`src/win/`), it provides real Direct2D COM objects, built with `#[implement]`
on the official IIDs and vtables from the `windows` crate. It also has the D2D1CreateFactory,
D2D1CreateDevice and D2D1CreateDeviceContext exports and the matrix and colour helper exports
(`src/exports_math.rs`). It implements about 55 interfaces. Methods that Direct2D itself stubs out
stay E_NOTIMPL (`src/win/mod.rs::NOT_IMPL`).

Off Windows (`src/portable.rs`), it uses hand-built COM vtables whose layout and QueryInterface
behaviour come from the generated `d2d-types` inventory. These implement the core render and
read-back path, so D2D1CreateFactory returns a working object through the raw C ABI.

## Build and verify

```sh
cargo build --release -p d2d-ffi --target x86_64-pc-windows-gnu   # d2d1.dll
cargo build --release -p d2d-ffi                                  # libd2d1.so
sh portable-test.sh                           # raw C ABI, native Linux/macOS/FreeBSD
sh decisive-test.sh                           # drop-in d2d1.dll vs native oracle under Wine
sh examples/run-examples.sh                   # C / C++ / Rust consumer examples vs oracle
```

`examples/` has C, C++ and Rust programs that use the standard Direct2D, DirectWrite and WIC API
exactly as on Windows: they create a factory and a render target, then draw geometry with a brush,
text and an effect, and save a PNG through WIC. Each one asserts that its output equals the Windows
capture byte for byte. `portable-test.c` is a raw ABI driver that needs no headers.

## License

Fresh Rust over the `windows` crate's COM machinery (MIT OR Apache-2.0). No `ref/` content is
copied.
