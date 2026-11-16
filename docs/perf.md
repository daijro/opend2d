# Performance

These benchmarks time OpenD2D drawing photo-size scenes like the ones real applications draw:
full-HD and 4K fills, gradients, scaled images and effects.

All the times are OpenD2D's own. The x86-64 column is OpenD2D compiled to native code; the
WebAssembly column is the same code compiled to wasm32 with SIMD and run in Node. There is no
Direct2D column, because Direct2D only runs on Windows and these were measured on Linux.

None of the optimisations change the output. Every one is checked against the full test suite, all
the fuzz scenes and the WIC decoders, which must stay identical byte for byte.

## Results

Each figure is the fastest of 9 runs, in milliseconds. Not every scene has been timed in
WebAssembly yet.

The figures were measured on commit 060c4791 (2026-11-14). Running the old and the release code side
by side on two of the scenes gave ratios of 0.975 and 1.062, which is within measurement noise, so
they also hold for this release.

| Scene (size) | OpenD2D, x86-64 (ms) | OpenD2D, WebAssembly (ms) |
| --- | ---: | ---: |
| Image scaled up (1920×1080) | 582 | 1371 |
| Fill (3840×2160) | 51 | 129 |
| Fill (1920×1080) | 10 | not measured |
| Linear gradient (3840×2160) | 359 | 711 |
| Radial gradient (3840×2160) | 364 | not measured |
| Linear gradient (1920×1080) | 90 | not measured |
| Radial gradient (1920×1080) | 88 | not measured |
| Blur (3840×2160) | 283 | not measured |
| Blur (1920×1080) | 273 | not measured |
| Color adjustment (1920×1080) | 51 | not measured |

## Reproducing

```sh
REPS=9 scripts/perf-bench.sh      # writes the scenes, builds both targets, prints the table below
```

The script builds each target once, then times each scene `REPS` times and keeps the fastest run.
The x86-64 and WebAssembly runs of a scene happen back to back, so both see the same machine state.
Use an otherwise idle machine: keeping the fastest run hides the odd slow run, but not a machine that
is busy for the whole benchmark.

The WebAssembly build uses plain SIMD only, without relaxed SIMD or fused multiply-add, because the
browser build has to produce exactly the same pixels as every other build.

## What made it fast

Scaling an image up with the high-quality cubic filter now reads each source pixel once and works on
all four color channels together, instead of reading the pixel again for each channel.

When a solid, path or stroke fill is fully opaque and fully covers a pixel, the result is just the
fill color, so OpenD2D writes it directly instead of reading the destination and blending. A test
proves this gives the same result as the full blend for every case it applies to. Opaque gradient
and bitmap brushes take the same shortcut.
