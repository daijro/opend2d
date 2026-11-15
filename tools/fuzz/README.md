# Differential fuzzer

OpenD2D is a drop-in reimplementation of the Windows 2D graphics stack. It must render **byte-for-byte**
what the system libraries render — on the web, any pixel or `HRESULT` that differs from the real thing is
a cross-site fingerprinting signal. This directory is the harness that proves equivalence: it generates
random draw programs ("scenes"), renders each one twice — once with the Windows system libraries on a
reference machine, once with the OpenD2D build under Wine — and reports every difference in pixels, in the
per-operation `HRESULT`/tag channel, and in query results.

Nothing here contains captured reference data. You supply a Windows reference machine; the tooling drives
it, and the native captures, pinned corpora and private test suite stay out of the repo (see
[What is not in the repo](#what-is-not-in-the-repo)).

## Layout

The generator, the differ and the minimiser are Rust and live with the rest of the harness; this directory
holds the orchestration around them.

| Piece | Where | What it does |
|-------|-------|--------------|
| Scene generator | `harness/src/bin/fuzz-gen.rs` | emit seeded random scenes, self-validating each |
| End-to-end fuzzer | `harness/fuzz.py` | generate → capture both sides → diff → minimise → bucket by area |
| Reference-machine I/O | `harness/vm.py` | push/run/pull over JSON-RPC to the reference machine |
| Regression differ | `crates/d2d/tests/conformance.rs` (`fuzz_diff`) | compare OpenD2D vs pinned native captures |
| Op attribution | same (`fuzz_attr`), + `prefix_bisect.py` | find which op first paints a mismatching pixel |
| WIC differ | `crates/d2d/tests/wic_fuzz.rs`, `wic/` | the same idea for the imaging-codec layer |
| `vmcap.py` | here | shared native-capture driver (per-run dir, cold fonts, chunked render) |
| `capture.py` | here | capture native oracles for a scene directory (warm or cold) |
| `prefix_bisect.py` | here | native prefix-bisect of a mismatching pixel |
| `gate.sh` | here | no-regression A/B across every corpus, for the merge gate |
| `dump_scene.py` | here | print a scene's ops with painter-op indexing |
| `route.py` | here | bucket attribution output by subsystem for triage |

## Prerequisites

* **A Windows reference machine** reachable over HTTP, running the small JSON-RPC bridge that `harness/vm.py`
  talks to (`push` / `run` / `pull`). Point the tooling at it with `D2D_VM_URL`.
* **Wine**, to run the OpenD2D build on this (Linux) host.
* The harness renderers built for Windows (`d2d-ref.exe`, `wic-dump.exe`) and the OpenD2D drop-in build
  staged for Wine. `harness/fuzz-setup.sh` builds and stages them; see it for the exact targets.
* Python 3 (plus Pillow for the WIC generator).

Common environment variables (defaults in parentheses):

| Variable | Meaning |
|----------|---------|
| `D2D_VM_URL` | reference-machine bridge endpoint |
| `D2D_VM_ROOT` | Windows working root on that machine (`C:\d2d-rs`) |
| `D2D_ROOT` | repository root (auto-detected from the script location) |
| `VM_CAP` / `BB_THREAD_ID` | names a per-run working directory so parallel runs do not collide |
| `D2D_VM_FONTS` | source directory of the measured faces (`C:\Windows\Fonts`) |
| `STAGE` | directory holding the staged OpenD2D build for Wine (used by `harness/fuzz.py`) |

## Workflows

### 1. Discover new divergences

```sh
STAGE=/path/to/staged-build COUNT=1000 SEED=12345 python3 harness/fuzz.py
```

Generates `COUNT` scenes from `SEED`, captures both sides, and for every mismatch writes a **minimised**
repro grouped by area (text, sampling, effects, raster, …). A run where the native side produced nothing
for any scene fails loudly rather than reporting "0 mismatches".

### 2. Pin a corpus and run the regression differ

Discovery regenerates scenes each time; the merge gate instead compares against **pinned** native captures
so a result is reproducible. Generate a corpus, capture the native oracle once, then diff as often as you
like:

```sh
cargo run -q -p harness --bin fuzz-gen -- scenes/ 1000 25000      # scenes/fz-<seed>.json
python3 tools/fuzz/capture.py scenes/ captures/round-x            # add --cold for text
cargo test -p d2d --test conformance --release fuzz_diff -- --nocapture   # with the env below
```

`capture.py` renders each scene through the system libraries and writes the oracle `fz-<seed>.bgra` +
`fz-<seed>.json`, pinning the exact input beside it as `fz-<seed>.scene.json`. `--cold` captures
first-time text rasterization (the text-conformance target) from a custom font collection of the
measured faces only, and records them in `cold-faces.txt`; when the capture directory has that file,
`fuzz_diff` restricts its font matcher to the same faces, as native's was.

The `fuzz_diff` test reads its inputs from the environment:

| Variable | Meaning |
|----------|---------|
| `FUZZ_DIFF_DIR` | the capture directory |
| `FUZZ_START`, `FUZZ_N` | first seed and count |
| `FUZZ_BROWSER` | `1` to emulate the browser adapter/flags |

It prints `FDIFF <seed>: …` per divergence and `FDIFF-SKIP <seed>` for inputs the native side could not
render (not comparable).

### 3. Attribute a mismatch

`fuzz_attr` (same test binary, set `FUZZ_ATTR=1` alongside the `FUZZ_DIFF_DIR`/`FUZZ_START`/`FUZZ_N` above)
reports *our* painter op for each mismatching pixel. To find the **native** painter — whose subsystem we
fail to match — bisect the pinned scene on the reference machine:

```sh
python3 tools/fuzz/prefix_bisect.py captures/round-x 25098:0:0 25757:4:12   # <seed>:<px>:<py> …
```

`dump_scene.py <seed> [dir]` prints the scene with painter indexing so an "op #k" reads back to a real op;
`route.py <attr_dir>` buckets a round of attribution output by subsystem for triage.

### 4. The merge gate

`gate.sh` runs `fuzz_diff` for a baseline and a candidate build across every corpus and fails if the
candidate diverges on any seed the baseline matched (or panics). It reports seeds the candidate *fixes*
without gating on them.

```sh
BASE=$(cargo test -p d2d --test conformance --release --no-run 2>&1 | grep -oE '/.*conformance-[0-9a-f]+')
# build the candidate the same way into CAND, then:
GATE_CORPORA=tools/fuzz/gate.corpora tools/fuzz/gate.sh "$BASE" "$CAND"
```

The corpus list is a config file (`gate.corpora`, gitignored) because it points at the pinned captures,
which are private. Copy [`gate.corpora.example`](gate.corpora.example) and edit it. `GATE_ACCEPTED` may
point at a file of owner-approved divergences that are reported but not gated (each pending a fix).

### 5. WIC (imaging codecs)

The imaging layer has its own generator/capture/diff that mirror the above:

```sh
python3 tools/fuzz/wic/wic_gen.py corpus/ 1000 0                      # needs Pillow
python3 tools/fuzz/wic/wic_cap.py corpus/ wic-native/                 # system codecs on the reference machine
python3 tools/fuzz/wic/wic_ab.py ours corpus/ /path/to/stage wic-ours/
python3 tools/fuzz/wic/wic_ab.py diff wic-native/ wic-ours/
```

`crates/d2d/tests/wic_fuzz.rs` is the in-process WIC fuzz loop (`FUZZ_SEED`, `FUZZ_ITERS`, and
`FUZZ_REPRO`/`FUZZ_REPRO_DEC` to re-run one seed).

## What is not in the repo

The owner keeps three things out, and the `.gitignore` here enforces it:

* **native captures** — `*.bgra`, the oracle `fz-*.json` / `wic-*` files, capture tarballs;
* **pinned corpora** — kept wherever your `gate.corpora` points, outside the tree;
* **the private test suite** — referenced by the harness, not vendored here.

Keep reverse-engineering notes (how a behaviour was derived) out of the repo too; describe the *behaviour*
in comments, not where it came from.
