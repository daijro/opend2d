#!/usr/bin/env python3
r"""Native prefix-bisect: find which op first paints a given pixel on the Windows system renderer.

    tools/fuzz/prefix_bisect.py <scenes_dir> <seed>:<px>:<py> [<seed>:<px>:<py> ...]

For each case, render every prefix ``ops[0:k]`` of the pinned scene (closing any open clips/layers with
synthetic pops so each prefix is balanced) through the system renderer, read the mismatch pixel, and
report the last ``k`` whose op changes it — that op is the native painter of the pixel. The op-attribution
pass over the differ gives *our* painter; this gives native's, which tells whose subsystem fails to draw
when the two disagree.

*scenes_dir* holds the pinned ``fz-<seed>.scene.json`` files (as written by ``capture.py``). Captures run
cold, matching the text-conformance target, with the corpus's own cold face set (its ``cold-faces.txt``).
See ``vmcap.py`` for reference-machine environment variables.
"""
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find vmcap when run with -I / from elsewhere
import vmcap


def truncations(scene):
    """Yield (k, balanced-prefix-scene) for every prefix length k, closing open scopes."""
    ops = scene["ops"]
    for k in range(1, len(ops) + 1):
        arr, open_ = ops[:k], []
        for o in arr:
            t = o.get("op")
            if t == "push_clip":
                open_.append({"op": "pop_clip"})
            elif t == "push_layer":
                open_.append({"op": "pop_layer"})
            elif t in ("pop_clip", "pop_layer") and open_:
                open_.pop()
        yield k, dict(scene, ops=arr + list(reversed(open_)))


def px_at(path, px, py, w):
    if not path.exists():
        return None
    b = path.read_bytes()
    o = (py * w + px) * 4
    return list(b[o:o + 4]) if o + 4 <= len(b) else None


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    scenes_dir = pathlib.Path(argv[0])
    cases = [tuple(int(x) for x in c.split(":")) for c in argv[1:]]

    work = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / f"fuzz-bisect-{vmcap.AGENT}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    index = {}  # seed -> (px, py, size, [(k, opname)])
    for seed, px, py in cases:
        scene = json.loads((scenes_dir / f"fz-{seed:010}.scene.json").read_text())
        kops = []
        for k, s in truncations(scene):
            json.dump(s, open(work / f"fz-{seed}{k:04d}.json", "w"))
            kops.append((k, scene["ops"][k - 1]["op"]))
        index[seed] = (px, py, scene["size"], kops)

    # Reuse the shared capture path. It keys outputs by the scene file stem, so each prefix lands as
    # fz-<seed><k>.bgra in the capture directory.
    exe = vmcap.ROOT / "harness/target/x86_64-pc-windows-gnu/release/d2d-ref.exe"
    natdir = work / "nat"
    vmcap.capture(exe, work, natdir, cold=True, faces=vmcap.faces_of(scenes_dir))

    print("\n==== native painter per case ====")
    for seed, px, py in cases:
        pxp, pyp, size, kops = index[seed]
        w = size[0]
        painter, prev, traj = None, None, []
        for k, opname in kops:
            v = px_at(natdir / f"fz-{seed}{k:04d}.bgra", pxp, pyp, w)
            if v is None:
                traj.append((k, opname, "kill/err"))
                continue
            if v != prev:
                painter, prev = (k, opname, v), v
                traj.append((k, opname, v))
        pt = f"op#{painter[0] - 1}={painter[1]} -> {painter[2]}" if painter else "never painted"
        print(f"\n{seed} @({pxp},{pyp}) size={size}: native painter = {pt}")
        for k, opname, v in traj:
            print(f"    k={k:3d} op#{k - 1}={opname:20s} px={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
