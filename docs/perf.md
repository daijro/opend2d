# Rendering performance

Photo-size, real-use benchmark scenes (full-HD and 4K fills, strokes, gradients, image draws with
scaling, and effect chains). Every optimisation below is **output-identical** — byte-for-byte across
the conformance suite, every fuzz corpus and the WIC decoders (the `identity-gate`).

## Reproduce

One build of each target, then the minimum wall time over `REPS` renders of each photo-size scene,
native and wasm back-to-back per scene so both see the same machine state. Run on a quiet machine —
`min` drops slow outliers but cannot hide sustained contention.

```sh
REPS=9 scripts/perf-bench.sh      # writes the scenes, builds both targets, prints the table below
```

The wasm build is `simd128` only — never `relaxed-simd` or FMA contraction (determinism: the web and
native targets must produce identical pixels).

## Results

Minimum wall time in milliseconds over the reps. `native` is x86_64; `wasm` is Node with `simd128`.
Refresh with `scripts/perf-bench.sh`.

These figures were measured on commit `060c4791` (2026-11-14). An interleaved A/B of the same scenes
against the current release renderer shows no change — per-scene native ratios of 0.975 and 1.062,
within measurement noise — so they stand for this release.

| scene (size)          | native (ms) | wasm (ms) |
|-----------------------|------------:|----------:|
| image-up (1920×1080)  |         582 |      1371 |
| fill-4k (3840×2160)   |          51 |       129 |
| fill-1080 (1920×1080) |          10 |         — |
| gradient-linear-4k    |         359 |       711 |
| gradient-radial-4k    |         364 |         — |
| gradient-linear-1080  |          90 |         — |
| gradient-radial-1080  |          88 |         — |
| blur-4k               |         283 |         — |
| blur-1080             |         273 |         — |
| coloradjust-1080      |          51 |         — |

## Techniques

- **HQ-cubic window combine fetches each texel once across the four channels** — the cubic-magnify
  `DrawBitmap` sampler runs the identical per-channel math across the four BGRA lanes from a single
  texel fetch rather than re-reading each texel per channel. This carries image magnification.
- **Full-coverage opaque source-over writes the source directly** — a UNORM8 solid/path/stroke fill
  skips the destination read and the per-pixel float blend when the result is provably the source,
  which carries the solid-fill interiors; a test proves it for every input it fires on.
- **Opaque source-over brush pixel writes the rounded source directly** — the gradient/bitmap-brush
  counterpart (gradient brushes realise to a 1-row ramp and fill through this path), proved against the
  general formula by test.
