# d2d

This crate is the portable CPU renderer with a safe Rust API. It is OpenD2D's authoritative
implementation: every pixel it draws is checked byte for byte against Direct2D (d2d1.dll on WARP),
as described in [../../docs/conformance.md](../../docs/conformance.md). It has no `cfg(windows)`
gate, so it compiles and runs identically on native targets and on `wasm32-unknown-unknown`.

It is also the lightest way to use OpenD2D from Rust. There is no COM; you call the renderer
directly:

```toml
[dependencies]
d2d = { path = "crates/d2d" }
```

## Modules

| Module | Contents |
| --- | --- |
| `geometry`, `pixels` | path, stroke and clip geometry; the coverage rasteriser, winding and composition |
| `brush`, `sampling` | solid, gradient, bitmap and image brushes; the WARP-exact texture sampler |
| `text` | glyph-run outlines, hinting, grayscale and ClearType, and the DirectWrite text blend |
| `effects` | the effect graph and the 64 built-in effects |
| `bitmap`, `wic`, `color_context` | bitmaps, WIC decoding and encoding, ICC and scRGB colour |
| `layer`, `svg`, `metafile`, `extras` | layers and clips, the SVG DOM, GDI metafiles, and ink, meshes, sprites and LUT3D |
| `float`, `crt`, `warp` | high-precision float targets, bit-exact CRT maths, and the WARP triangle pipeline |

The FFI crates (`d2d-ffi`, `dwrite-ffi` and `wic-ffi`) are thin COM shells over these modules. The
pixels are produced here.

## Tests

`cargo test -p d2d --test conformance` runs every module against the captured Windows output. The
list of scenes that match exactly (6132 of 6159 today, kept with the private captures) only grows. The same test runs on
wasm32, aarch64 and Windows to check that the output is identical across targets
(`scripts/conformance.sh`).

## License

Fresh Rust code that uses MIT OR Apache-2.0 crates (`kurbo`, `skrifa`, `rustybuzz`, `image` and
others). No `ref/` content is copied.
