# Differential fuzzer

OpenD2D is a drop-in reimplementation of the Windows 2D graphics stack, and it has to render exactly
what the system libraries render, byte for byte. On the web, any pixel or HRESULT that differs from
Windows is a signal that can fingerprint a user across sites.

This directory holds the harness that checks this. It generates random draw programs, called
scenes, and renders each one twice: once with the Windows system libraries on a reference machine,
and once with the OpenD2D build under Wine. It then reports every difference in pixels, in the
per-operation HRESULT and tag channel, and in query results.

Nothing here contains captured reference data. The Windows captures, pinned corpora and private
test suite stay out of the repository (see [What is not in the repo](#what-is-not-in-the-repo)).
So does the bridge that drives the Windows reference machine (`harness/vm.py`) and the end-to-end
runner built on it. The scripts here that talk to Windows (`capture.py`, `prefix_bisect.py` and
`wic/wic_cap.py`) import that bridge, so they do not run from a public clone. To capture Direct2D's
output yourself, build the harness's reference renderer `d2d-ref.exe` and run it on a Windows
machine (see [Capturing on Windows](#capturing-on-windows)). The generator, the differ, the gate
and the triage scripts work without the private files.

## Layout

The scene generator, the differ and the minimiser are written in Rust and live with the rest of the
harness. This directory holds the orchestration around them.

| Piece | Where | What it does |
|-------|-------|--------------|
| Scene generator | `harness/src/bin/fuzz-gen.rs` | emits seeded random scenes and validates each one |
| Reference renderer | `harness/src/bin/d2d-ref.rs` | draws one scene with Windows' Direct2D and writes its pixels and results |
| Regression differ | `crates/d2d/tests/conformance.rs` (`fuzz_diff`) | compares OpenD2D with pinned Windows captures |
| Op attribution | the same file (`fuzz_attr`), plus `prefix_bisect.py` | finds which op first paints a mismatching pixel |
| WIC differ | `crates/d2d/tests/wic_fuzz.rs`, `wic/` | the same idea for the imaging codec layer |
| `vmcap.py` | here | shared Windows capture driver (per-run directory, cold fonts, chunked rendering); needs the private bridge |
| `capture.py` | here | captures Windows output for a scene directory, warm or cold; needs the private bridge |
| `prefix_bisect.py` | here | prefix-bisects a mismatching pixel on Windows; needs the private bridge |
| `gate.sh` | here | no-regression A/B run across every corpus, for the merge gate |
| `dump_scene.py` | here | prints a scene's ops with painter-op indexing |
| `route.py` | here | buckets attribution output by subsystem for triage |

## Prerequisites

- A Windows machine, to capture Direct2D's output with `d2d-ref.exe`.
- An x86_64-pc-windows-gnu Rust toolchain to build it:
  `cargo build --release --manifest-path harness/Cargo.toml --target x86_64-pc-windows-gnu --bin d2d-ref`.
- Python 3, plus Pillow for the WIC generator.

`vmcap.py` and the scripts built on it also read these variables (defaults in parentheses). They
only matter with the private bridge:

| Variable | Meaning |
|----------|---------|
| `D2D_VM_URL` | reference-machine bridge endpoint |
| `D2D_VM_ROOT` | Windows working root on that machine (`C:\d2d-rs`) |
| `D2D_ROOT` | repository root (detected from the script location) |
| `VM_CAP` / `BB_THREAD_ID` | names a per-run working directory so parallel runs do not collide |
| `D2D_VM_FONTS` | source directory of the measured faces (`C:\Windows\Fonts`) |

## Capturing on Windows

`d2d-ref.exe SCENE.json OUTPUT.png` draws one scene with the system's Direct2D on WARP. Run it from
a directory with no OpenD2D DLL beside it, so Windows loads its own libraries. It writes the image,
plus the raw pixels and the result codes next to it with the same stem (`.bgra` and `.json`). Copy
the generated scenes to the Windows machine, run each one, and copy the `.bgra` and `.json` files
back into a capture directory named `fz-<seed>.bgra` and `fz-<seed>.json`, with the input scene
beside them as `fz-<seed>.scene.json`. That is the layout `capture.py` writes and `fuzz_diff` reads.

## Workflows

### 1. Discover new divergences

The end-to-end discovery runner, which generates scenes, captures both sides, minimises and buckets
each mismatch, is private. Without it, generate a fresh corpus, capture it on Windows and run the
differ, as below.

### 2. Pin a corpus and run the regression differ

Discovery generates fresh scenes every time. The merge gate instead compares against pinned
Windows captures, so its results are reproducible. Generate a corpus, capture the Windows output
once, then diff as often as you like:

```sh
cargo run -q --manifest-path harness/Cargo.toml --bin fuzz-gen -- scenes/ 1000 25000   # scenes/fz-<seed>.json
python3 tools/fuzz/capture.py scenes/ captures/round-x            # private bridge; add --cold for text
cargo test -p d2d --test conformance --release fuzz_diff -- --nocapture   # with the env below
```

`capture.py` renders each scene through the system libraries and writes the Windows output as
`fz-<seed>.bgra` and `fz-<seed>.json`, with the exact input pinned beside it as
`fz-<seed>.scene.json`. It needs the private bridge; without it, capture with `d2d-ref.exe` as
described in [Capturing on Windows](#capturing-on-windows).

`--cold` captures first-time text rasterization, which is what text conformance targets. It renders
from a custom font collection that holds only the measured faces, and records those faces in
`cold-faces.txt`. When a capture directory has that file, `fuzz_diff` restricts its font matcher to
the same faces, as the Windows capture was restricted.

The `fuzz_diff` test reads its inputs from the environment:

| Variable | Meaning |
|----------|---------|
| `FUZZ_DIFF_DIR` | the capture directory |
| `FUZZ_START`, `FUZZ_N` | first seed and count |
| `FUZZ_BROWSER` | if set, generate browser-style scenes (realistic values, only the ops a browser canvas uses); the differ only uses it when a capture has no pinned `.scene.json` and the scene has to be regenerated. `fuzz-gen` reads it too |

It prints `FDIFF <seed>: …` for each divergence, and `FDIFF-SKIP <seed>` for inputs Windows could
not render, which cannot be compared.

### 3. Attribute a mismatch

`fuzz_attr` is in the same test binary. Set `FUZZ_ATTR=1` together with the `FUZZ_DIFF_DIR`,
`FUZZ_START` and `FUZZ_N` variables above, and it reports which OpenD2D op painted each
mismatching pixel. To find the op that painted it on Windows, and so the subsystem OpenD2D fails to
match, bisect the pinned scene on the reference machine (this needs the private bridge):

```sh
python3 tools/fuzz/prefix_bisect.py captures/round-x 25098:0:0 25757:4:12   # <seed>:<px>:<py> …
```

`dump_scene.py <seed> [dir]` prints the scene with painter indexing, so an "op #k" can be traced
back to a real op. `route.py <attr_dir>` buckets a round of attribution output by subsystem for
triage.

### 4. The merge gate

`gate.sh` runs `fuzz_diff` for a baseline build and a candidate build across every corpus. It fails
if the candidate diverges on any seed the baseline matched, or if it panics. It also reports seeds
the candidate fixes, without gating on them.

```sh
BASE=$(cargo test -p d2d --test conformance --release --no-run 2>&1 | grep -oE '/.*conformance-[0-9a-f]+')
# build the candidate the same way into CAND, then:
GATE_CORPORA=tools/fuzz/gate.corpora tools/fuzz/gate.sh "$BASE" "$CAND"
```

The corpus list is a config file, `gate.corpora`. It is gitignored because it points at the pinned
captures, which are private. Copy [`gate.corpora.example`](gate.corpora.example) and edit it.
`GATE_ACCEPTED` can point at a file of owner-approved divergences, each waiting on a fix, which are
reported but not gated.

### 5. WIC (imaging codecs)

The imaging layer has its own generator, capture and diff tools that mirror the ones above. The
capture and A/B steps run `wic-dump.exe`, a harness program that is not in this repository, and
`wic_cap.py` also needs the private bridge. Only `wic_gen.py` and the in-process `wic_fuzz.rs` loop
work from a public clone:

```sh
python3 tools/fuzz/wic/wic_gen.py corpus/ 1000 0                      # needs Pillow
python3 tools/fuzz/wic/wic_cap.py corpus/ wic-native/                 # system codecs on the reference machine
python3 tools/fuzz/wic/wic_ab.py ours corpus/ /path/to/stage wic-ours/
python3 tools/fuzz/wic/wic_ab.py diff wic-native/ wic-ours/
```

`crates/d2d/tests/wic_fuzz.rs` is the in-process WIC fuzz loop. It reads `FUZZ_SEED` and
`FUZZ_ITERS`, and `FUZZ_REPRO` or `FUZZ_REPRO_DEC` re-runs a single seed.

## What is not in the repo

The owner keeps three things out of the repository, and the `.gitignore` here enforces it:

- Windows captures: `*.bgra`, the captured `fz-*.json` and `wic-*` files, and capture tarballs.
- Pinned corpora, kept outside the tree wherever your `gate.corpora` points.
- The private test suite, which the harness references but does not vendor.

Keep reverse-engineering notes, meaning how a behaviour was derived, out of the repository as well.
Comments should describe the behaviour itself, not where it came from.
