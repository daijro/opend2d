# opend2d parity backlog

Known, un-fixed byte-exactness gaps that **no corpus scene currently reaches** — deliberately deferred,
listed so the fuzzer can aim at them later. Distinct from [`docs/gaps.md`](docs/gaps.md) (areas
implemented but not yet certified) and the deliberate native-matching rejections in
[`docs/api.md`](docs/api.md).

## Text raster

- **Simulated-bold SIMPLE overscale mode (ClearType).** Native's overscale-to-bold path switches from
  its LUT run-darkening to a pure horizontal bit-dilation when the embolden amount exceeds 24 or the
  overscaled bitmap is taller than 48 rows — i.e. a synthetic-bold ClearType glyph taller than ~48
  device px. Our `embolden_bits` OR-dilate is byte-exact for the LUT regime (em ≤ 48, verified against
  the native sims=1 ladder) but diverges in SIMPLE mode (the em-80 ladder step, maxd 255). Unported
  because no corpus glyph is a large synthetic-bold ClearType run — the only families that resolve to
  synthetic bold (Lucida Console, Impact; no bold-face file) appear only at small em. Measured via VM
  probes of the native overscale path. Fuzzer target: a weight ≥ 600 ClearType run at > 48 px em for a
  family with no dedicated bold face.
