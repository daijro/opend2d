# OpenD2D parity backlog

This file lists known byte-exactness gaps that no scene in the corpus reaches yet. They are
deferred on purpose and written down so the fuzzer can aim at them later. This is separate from
[docs/gaps.md](docs/gaps.md), which covers areas that are implemented but not yet certified, and
from the deliberate rejections listed in [docs/api.md](docs/api.md).

## Text raster

### Simulated bold in ClearType SIMPLE overscale mode

When Direct2D emboldens an overscaled ClearType glyph, it normally darkens runs using a lookup
table. If the embolden amount exceeds 24, or the overscaled bitmap is taller than 48 rows, it
switches to a plain horizontal bit dilation instead (SIMPLE mode). In practice that means a
synthetic-bold ClearType glyph taller than about 48 device pixels.

OpenD2D's `embolden_bits` OR-dilation is byte-exact in the lookup-table regime (em up to 48,
checked against Direct2D's sims=1 ladder) but diverges in SIMPLE mode: the em-80 step of the ladder
differs by up to 255. This was measured by probing Direct2D's overscale path on the reference VM.

It has not been ported because no corpus glyph is a large synthetic-bold ClearType run. The only
families that resolve to synthetic bold, Lucida Console and Impact (neither has a bold face file),
appear only at small em sizes.

Fuzzer target: a ClearType run with weight 600 or more, at an em size above 48 px, in a family with
no dedicated bold face.
