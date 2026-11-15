# d2d-shader

CPU execution of Direct2D pixel-shader bytecode, for custom and built-in effect passes. It reads a
DXBC container, decodes its shader-model 4/5 tokens, and runs them on the CPU with the float/integer
semantics WARP's JIT emits — so an effect's pixel shader produces the same bytes on every platform
(native and `wasm32`) as native Direct2D on Windows WARP.

A leaf crate: it holds no GPU or platform state, and `d2d` re-exports its WARP-exact shader maths
(`warpmath`) so the core can depend on it without a cycle.

## API (used by `d2d`'s effect graph)

1. `Shader::parse` the pass's DXBC once, then `Shader::unsupported_op` — if it names an opcode, report
   that blockage instead of running (the executor never silently zeroes an unimplemented opcode).
2. Per output pixel, build `PixelInputs` (interpolated input registers + the `b0` constant bytes) and
   an `exec::Env` that samples each bound input image / resource texture (wired to `d2d::sampling`).
3. `Shader::run` returns the premultiplied RGBA `o0`; `exec::to_unorm8` per channel for an 8-bit
   target.

Texture `sample`/`ld` are delegated to the caller's `Env`, so the WARP-exact sampling lives in `d2d`.

This is a secondary track behind the authoritative CPU renderer; it advances as arbitrary-shader
effects are closed against the oracle.

## Tests

`cargo test -p d2d-shader` — the sampler-parity test compares against `d2d::sampling`.

## License

Fresh Rust. DXBC is parsed, never copied into tracked files; no `ref/` content is used.
