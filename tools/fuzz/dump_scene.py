#!/usr/bin/env python3
# Print one scene's ops with painter-op indexing, so a divergence reported as "op #k" can be read back
# to the actual op. Resource-definition ops (brushes, geometries, fonts, ...) do not paint, so they are
# not numbered; only drawing ops get an op# that matches the differ's per-op index.
#
#   tools/fuzz/dump_scene.py <seed> [scenes_dir]
#
# scenes_dir defaults to $FUZZ_DIFF_DIR, then the current directory. Reads fz-<seed>.scene.json.
import json
import os
import pathlib
import sys

RESOURCES = {
    "solid", "linear", "radial", "bitmap", "effect", "geometry", "geometry_group", "command_list",
    "image_brush", "bitmap_brush", "stroke_style", "text_format", "gradient_mesh", "path",
    "ellipse_geometry", "transformed_geometry", "font", "image_source", "sprite_batch",
}


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    seed = int(argv[0])
    d = pathlib.Path(os.path.expanduser(
        argv[1] if len(argv) > 1 else os.environ.get("FUZZ_DIFF_DIR", ".")))
    scene = json.loads((d / f"fz-{seed:010d}.scene.json").read_text())
    print(seed, scene["size"], scene.get("dpi"))
    painter = 0
    for i, o in enumerate(scene["ops"]):
        tag = ""
        if o["op"] not in RESOURCES:
            tag = "op#%d" % painter
            painter += 1
        trimmed = {k: (v if len(str(v)) < 160 else str(v)[:160] + "…") for k, v in o.items()}
        print("%3d %6s" % (i, tag), trimmed)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
