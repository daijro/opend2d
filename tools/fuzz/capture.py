#!/usr/bin/env python3
r"""Capture native oracles for the differential diff.

    tools/fuzz/capture.py <scenes_dir> <out_dir> [--cold] [--faces=r25|a,b,c]

*scenes_dir* holds generated scenes ``fz-<seed>.json`` (write them with the generator, e.g.
``cargo run -q -p harness --bin fuzz-gen -- <scenes_dir> <count> <start-seed>``). This renders each one
through the Windows system renderer on the reference machine and writes the oracle ``fz-<seed>.bgra`` +
``fz-<seed>.json`` into *out_dir*, then pins the exact input beside each as ``fz-<seed>.scene.json`` so
the differ compares the scene that was actually rendered (the seed->scene mapping drifts as the
generator grows, and the pulled ``fz-<seed>.json`` is the oracle, not the input).

``--cold`` captures first-time (cold) text rasterization using the bumped measured faces — the target
for text conformance. See ``vmcap.py`` for the reference-machine environment variables.
"""
import os
import pathlib
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find vmcap when run with -I / from elsewhere
import vmcap


def main(argv):
    cold = "--cold" in argv
    # --faces=r25 (the round-25 set) or --faces=a,b,c (face file stems): the cold collection's faces.
    faces = next((a.split("=", 1)[1] for a in argv if a.startswith("--faces=")), None)
    faces = vmcap.FONTS25 if faces == "r25" else faces.split(",") if faces else None
    argv = [a for a in argv if a != "--cold" and not a.startswith("--faces=")]
    if len(argv) != 2:
        print(__doc__)
        return 2
    scenes_dir, out_dir = pathlib.Path(argv[0]), pathlib.Path(argv[1])
    exe = vmcap.ROOT / "harness/target/x86_64-pc-windows-gnu/release/d2d-ref.exe"
    if not exe.exists():
        raise SystemExit(f"renderer not built: {exe} (build the Windows d2d-ref target first)")

    # Pin the exact input scene beside each capture BEFORE the oracle is written: if out_dir == scenes_dir
    # the pulled oracle ``fz-<seed>.json`` would overwrite the input, so pin first. Afterwards the
    # ``.scene.json`` holds the real scene and the ``.json`` holds the native oracle, whether or not the
    # two directories are the same.
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = sorted(p.name for p in scenes_dir.glob("fz-*.json"))
    for s in scenes:
        shutil.copyfile(scenes_dir / s, out_dir / (s[:-5] + ".scene.json"))

    vmcap.capture(exe, scenes_dir, out_dir, cold=cold, faces=faces)
    print(f"CAPTURE-DONE native bgra: {len(list(out_dir.glob('*.bgra')))} "
          f"json: {len(list(out_dir.glob('fz-*[0-9].json')))} pinned scenes: {len(scenes)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
