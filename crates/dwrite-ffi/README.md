# dwrite-ffi

This crate builds the drop-in dwrite.dll, the DirectWrite COM surface that Direct2D applications
use to make text. The `[lib]` is named `dwrite` and is built as `cdylib` and `rlib`. It exports
DWriteCreateFactory, which returns real DirectWrite COM objects built with `#[implement]` on the
official `windows` IIDs and vtables. Layout, shaping and glyph metrics are forwarded to
[`d2d::text`](../d2d).

Text objects hand their font bytes and layout to OpenD2D's d2d1.dll, for DrawText and
DrawGlyphRun, through a private COM interface called ITextInterop that crosses the DLL boundary.
That keeps rendering in the shared core, so both DLLs draw exactly the core's pixels.

It implements 29 DirectWrite interfaces plus ITextInterop. They cover the factory, text format and
layout, font files, faces, collections and fallback, the text analyzer, glyph-run analysis,
rendering params, and the colour glyph enumerator.

Off Windows, the DirectWrite definitions in the `windows` crate are gated behind `cfg(windows)`, so
the library only exposes a DWriteCreateFactory that returns E_NOTIMPL. Text runs through the
Windows ABI DLL under Wine instead.

For font resolution, the system font collection reads the directory given in `D2D_FONTS` (a
Windows path under Wine). These are the same faces the core and the harness resolve.

```sh
cargo build --release -p dwrite-ffi --target x86_64-pc-windows-gnu   # dwrite.dll
```

It is verified together with d2d1.dll by `crates/d2d-ffi/decisive-test.sh` and
`crates/d2d-ffi/examples/run-examples.sh`.

## License

Fresh Rust over the `windows` crate (MIT OR Apache-2.0). No `ref/` content is copied.
