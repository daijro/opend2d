# d2d-wgpu

An optional GPU backend. The CPU path in [`d2d`](../d2d) is authoritative. This crate only
accelerates operations it can prove are byte-identical to that path, and every caller keeps a CPU
fallback: `Gpu::new` returns `None` when there is no adapter, for example on FreeBSD, when headless,
or in a browser without WebGPU.

## Why integer-only

WGSL explicitly allows floating-point reassociation, contraction (FMA) and variation in accuracy,
so the same float WGSL on two conformant devices need not produce the same bits. Integer (`u32`)
arithmetic has no such allowance. The accelerated set is therefore exactly the premultiplied BGRA8
operations that reduce to integer arithmetic matching `d2d::pixels`. Those are clears and solid
fills of pixel-aligned, fully covered regions, and image composites and bitmap blits at integer
offsets, under the integer primitive blend modes Copy, Min, Max and Add. Opaque SourceOver counts
as Copy.

Translucent SourceOver, anti-aliased fringe pixels, gradient lookup-table sampling and bilinear
filtering stay on the CPU, because their arithmetic is floating point and cannot be proven
identical here. There is no DXBC to WGSL path for arbitrary shaders either, since those are
floating point too, so the CPU executor in `d2d-shader` remains the authority.

`wgpu` selects the backend per target: Vulkan, Metal, DX12 or GLES on native targets, and WebGPU on
wasm32. Direct2D callers are synchronous, so native readback blocks (using `pollster`), while the
browser drives completion itself.

## Tests

`cargo test -p d2d-wgpu` runs parity tests that compare GPU readback byte for byte with the CPU
rasteriser. They are skipped when there is no adapter.

## License

Fresh Rust over `wgpu` (MIT OR Apache-2.0). No `ref/` content is used.
