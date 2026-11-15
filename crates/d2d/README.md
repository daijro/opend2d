# d2d

The portable CPU renderer and safe Rust API — opend2d's **authoritative** implementation. Every
pixel it draws is checked byte-for-byte against native `d2d1.dll` on WARP (see
[../../docs/conformance.md](../../docs/conformance.md)). No `cfg(windows)` gate: it compiles and runs
identically on native targets and `wasm32-unknown-unknown`.

This is the lightest way to use opend2d from Rust — no COM, just call the renderer:

```toml
[dependencies]
d2d = { path = "crates/d2d" }
```

## Modules

| Module | What |
| --- | --- |
| `geometry`, `pixels` | path/stroke/clip geometry; coverage rasteriser, winding, composition |
| `brush`, `sampling` | solid/gradient/bitmap/image brushes; the WARP-exact texture sampler |
| `text` | glyph-run outlines, hinting, grayscale/ClearType, the DirectWrite text blend |
| `effects` | the effect graph and the 64 built-in effects |
| `bitmap`, `wic`, `color_context` | bitmaps, WIC decode/encode, ICC/scRGB colour |
| `layer`, `svg`, `metafile`, `extras` | layers/clips, SVG DOM, GDI metafile, ink/mesh/sprite/LUT3D |
| `float`, `crt`, `warp` | highp-float targets, bit-exact CRT math, the WARP triangle pipeline |

The FFI crates (`d2d-ffi`, `dwrite-ffi`, `wic-ffi`) are thin COM shells over these modules; the
pixels are produced here.

## Tests

`cargo test -p d2d --test conformance` drives every module against the captured oracle;
`conformance/passing.txt` is the byte-exact set (1190 scenes) and only grows. The same test runs on
wasm32/aarch64/Windows for cross-target parity (`scripts/conformance.sh`).

## License

Fresh Rust; reuses MIT-OR-Apache-2.0 crates (`kurbo`, `skrifa`, `rustybuzz`, `image`, …). No `ref/`
content is copied.
