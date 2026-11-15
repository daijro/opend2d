# d2d-ffi

The drop-in **`d2d1.dll`** (Windows / Wine) and **`libd2d1.{so,dylib}`** (Linux/macOS/FreeBSD), plus
its C/COM ABI. The `[lib]` is named `d2d1` and built `cdylib` (the loadable library) + `rlib` (so the
in-process conformance test can drive the same objects without the OS loader).

It is a thin shell: the COM methods forward to the [`d2d`](../d2d) core, which produces the pixels.

- **Windows / Wine** (`src/win/`): real Direct2D COM objects built with the official `windows` crate
  IIDs/vtables via `#[implement]`, and the `D2D1CreateFactory` / `D2D1CreateDevice` /
  `D2D1CreateDeviceContext` + matrix/colour DLL exports (`src/exports_math.rs`). ~55 interfaces;
  methods native itself stubs stay `E_NOTIMPL` (`src/win/mod.rs::NOT_IMPL`).
- **Off Windows** (`src/portable.rs`): hand-built COM vtables whose layout/QI are driven by the
  generated `d2d-types` inventory, implementing the render-and-read-back core so `D2D1CreateFactory`
  returns a working object through the raw C ABI.

## Build & verify

```sh
cargo build --release -p d2d-ffi --target x86_64-pc-windows-gnu   # d2d1.dll
cargo build --release -p d2d-ffi                                  # libd2d1.so
sh portable-test.sh                           # raw C ABI, native Linux/macOS/FreeBSD
sh decisive-test.sh                           # drop-in d2d1.dll vs native oracle under Wine
sh examples/run-examples.sh                   # C / C++ / Rust consumer examples vs oracle
```

`examples/` has C, C++ and Rust programs using the standard Direct2D/DirectWrite/WIC API exactly as
on Windows (factory → render target → geometry → brush → text → effect → save-PNG-through-WIC), each
asserting byte-equality with the native oracle. `portable-test.c` is a header-free raw-ABI driver.

## License

Fresh Rust over the `windows` crate COM machinery (MIT OR Apache-2.0). No `ref/` content is copied.
