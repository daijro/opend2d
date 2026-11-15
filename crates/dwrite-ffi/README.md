# dwrite-ffi

The drop-in **`dwrite.dll`** — the DirectWrite COM surface Direct2D apps use to make text. The
`[lib]` is named `dwrite` (`cdylib` + `rlib`). It exports `DWriteCreateFactory` and hands back real
DirectWrite COM objects (`#[implement]` with the official `windows` IIDs/vtables), forwarding layout,
shaping and glyph metrics to [`d2d::text`](../d2d).

Text objects carry their font bytes and layout to our `d2d1.dll`'s `DrawText`/`DrawGlyphRun` across
the DLL boundary through a private `ITextInterop` COM interface, so rendering stays in the shared
core and both DLLs draw exactly the core's pixels.

- 29 DirectWrite interfaces (factory, text format/layout, font file/face/collection/fallback,
  analyzer, glyph-run analysis, rendering params, colour-glyph enumerator) plus `ITextInterop`.
- Off Windows the `windows` DirectWrite definitions are `cfg(windows)`-gated, so the library exposes
  only a `DWriteCreateFactory` returning `E_NOTIMPL`; text runs through the Windows-ABI DLL under Wine.

Font resolution: our system font collection reads the directory given in `D2D_FONTS` (a Windows path
under Wine), the same faces the core and harness resolve.

```sh
cargo build --release -p dwrite-ffi --target x86_64-pc-windows-gnu   # dwrite.dll
```

Verified together with `d2d1.dll` by `crates/d2d-ffi/decisive-test.sh` and `examples/run-examples.sh`.

## License

Fresh Rust over the `windows` crate (MIT OR Apache-2.0). No `ref/` content is copied.
