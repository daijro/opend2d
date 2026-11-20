# d2d-shader

This crate runs Direct2D pixel-shader bytecode on the CPU, for both custom and built-in effect
passes. It reads a DXBC container, decodes its shader model 4 and 5 tokens, and executes them with
the same float and integer semantics that WARP's JIT emits. As a result, an effect's pixel shader
produces the same bytes on every platform (native targets and `wasm32`) as Direct2D does on
Windows WARP.

It is a leaf crate with no GPU or platform state. `d2d` re-exports its WARP-exact shader maths
(`warpmath`), so the core can depend on it without a cycle.

## API

`d2d`'s effect graph uses it in three steps:

1. Call `Shader::parse` once on the pass's DXBC, then call `Shader::unsupported_op`. If that names
   an opcode, report the blockage instead of running the shader. The executor never silently
   produces zeros for an unimplemented opcode.
2. For each output pixel, build `PixelInputs` (the interpolated input registers and the `b0`
   constant bytes) and an `exec::Env` that samples each bound input image or resource texture. The
   `Env` is wired to `d2d::sampling`.
3. `Shader::run` returns the premultiplied RGBA `o0`. For an 8-bit target, apply `exec::to_unorm8`
   to each channel.

The texture instructions `sample` and `ld` are delegated to the caller's `Env`, so the WARP-exact
sampling lives in `d2d`.

This is a secondary track behind the authoritative CPU renderer. It advances as arbitrary-shader
effects are closed against Direct2D's output.

## Tests

`cargo test -p d2d-shader` runs the sampler parity test, which compares against `d2d::sampling`.

## License

Fresh Rust. DXBC is parsed, never copied into tracked files. No `ref/` content is used.
