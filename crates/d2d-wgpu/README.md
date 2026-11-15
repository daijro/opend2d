# d2d-wgpu

An **optional** GPU backend. The CPU path in [`d2d`](../d2d) is authoritative; this crate only
accelerates operations it can prove **byte-identical** to that path, and every caller keeps a CPU
fallback (`Gpu::new` returns `None` with no adapter — FreeBSD, headless, or a browser without WebGPU).

## Why integer-only

WGSL explicitly permits floating-point reassociation, contraction (FMA) and accuracy variation, so
identical float WGSL on two conformant devices need not produce identical bits. Integer (`u32`)
arithmetic has no such licence. So the accelerated set is exactly the premultiplied-BGRA8 operations
that reduce to integer arithmetic matching `d2d::pixels`:

- clear / solid fill of a pixel-aligned (full-coverage) region, and image composite / bitmap blit at
  an integer offset, under the integer primitive-blend modes Copy/Min/Max/Add (opaque `SourceOver` is
  `Copy`).

Translucent `SourceOver`, anti-aliased fringe pixels, gradient-LUT sampling and bilinear filtering
stay on the CPU — their arithmetic is floating-point and cannot be proven identical here. There is no
DXBC→WGSL path for arbitrary shaders; those are floating point, so `d2d-shader`'s CPU executor remains
the authority.

Backends are selected per target inside `wgpu` (Vulkan/Metal/DX12/GLES natively, WebGPU on wasm32);
D2D callers are synchronous, so native readback blocks (`pollster`) while the browser drives
completion itself.

## Tests

`cargo test -p d2d-wgpu` — parity tests compare GPU readback byte-for-byte against the CPU rasteriser
(skipped with no adapter).

## License

Fresh Rust over `wgpu` (MIT OR Apache-2.0). No `ref/` content is used.
