#!/usr/bin/env python3
# Bucket op-attribution output (the differ's `fuzz_attr` pass) by subsystem and op, so a round of
# divergences can be triaged at a glance and routed to whoever owns each subsystem. Each `fuzz_attr`
# line names the corpus, seed, painter op and first-mismatch pixel; this resolves the op back to the
# scene to classify it (and to see which brush/image a draw used).
#
#   tools/fuzz/route.py <attr_dir>
#
# <attr_dir> holds the saved `fuzz_attr` output files. Scenes are looked up as
# <root>/<corpus>/fz-<seed>.scene.json then <root>/fz-<seed>.scene.json for each root in
# $FUZZ_CAP_DIRS (colon-separated). Writes rows.json beside the input for further analysis.
import collections
import json
import os
import pathlib
import re
import sys

LINE = re.compile(
    r"^(?P<c>[\w-]+):(?P<s>\d+): op#(?P<i>\d+|18446744073709551615)="
    r"(?P<t>\S+) delta=(?P<d>\d+) at \((?P<x>\d+),(?P<y>\d+)\)(?P<e>.*)$")


def find_scene(roots, corpus, seed):
    for root in roots:
        for cand in (root / corpus / f"fz-{seed:010d}.scene.json", root / f"fz-{seed:010d}.scene.json"):
            if cand.exists():
                return cand
    return None


def subsystem(op, brush_kind):
    if op in ("draw_image", "draw_bitmap", "draw_sprite_batch", "draw_sprites", "sprite_batch",
              "fill_opacity_mask") or brush_kind in ("image_brush", "bitmap_brush"):
        return "sampling"
    if op in ("text", "text_layout", "glyph_run"):
        return "text"
    if op in ("push_layer", "push_clip", "pop_layer", "pop_clip", "clear"):
        return "layers"
    return "raster"


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    attr_dir = pathlib.Path(argv[0])
    roots = [pathlib.Path(os.path.expanduser(r)) for r in os.environ.get("FUZZ_CAP_DIRS", ".").split(":") if r]

    rows = []
    for f in sorted(attr_dir.iterdir()):
        if not f.is_file():
            continue
        for line in f.read_text(errors="replace").splitlines():
            m = LINE.match(line.strip())
            if not m:
                continue
            c, s, i, t, d = m["c"], int(m["s"]), int(m["i"]), m["t"], int(m["d"])
            sc = find_scene(roots, c, s)
            try:
                ops = json.loads(sc.read_text())["ops"] if sc else []
            except Exception:  # noqa: BLE001
                ops = []
            ids = {}

            def walk(lst):
                for o in lst:
                    if "id" in o:
                        ids[o["id"]] = o
                    if o.get("op") == "command_list":
                        walk(o.get("ops", []))
            walk(ops)
            op = ops[i] if i < len(ops) else {}
            brush = ids.get(op.get("brush")) if isinstance(op.get("brush"), str) else None
            bk = brush.get("op") if brush else None
            img = ids.get(op.get("image")) if isinstance(op.get("image"), str) else None
            ik = img.get("op") if img else None
            kind = f"draw_image[{ik}]" if t == "draw_image" else (f"{t}[{bk}]" if bk else t)
            rows.append((subsystem(t, bk), kind, d, c, s, i, int(m["x"]), int(m["y"]), m["e"].strip()))

    by = collections.defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
    for sub in sorted(by):
        ks = collections.Counter(r[1] for r in by[sub])
        print(f"== {sub}: {len(by[sub])}  " + ", ".join(f"{k} {n}" for k, n in ks.most_common()))
    for sub in sorted(by):
        print(f"\n-- {sub} by delta")
        for r in sorted(by[sub], key=lambda r: -r[2]):
            print(f"  d{r[2]:<3} {r[3]}:{r[4]} op{r[5]} {r[1]} at ({r[6]},{r[7]})")
    (attr_dir / "rows.json").write_text(json.dumps([list(r) for r in rows]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
