#!/usr/bin/env python3
"""Emit the cf3-NN colour-fonts-part-3 scenes (composite modes + nested clip) into the suite (conformance/suite/scenes/).

Each scene carries colrtest3.ttf inline and draws one glyph with the `color_glyph_run` op
(DrawPaintGlyphRun), em 50 at baseline (12,52) on 64×64 — the same flat-coverage setup as the
colorfont2 scenes, so only the composite/clip math is under test.

Run on the host after gen_font3.py: `python3 gen_scenes3.py`.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# testdata → color → text → src → d2d → crates → repo root
ROOT = os.path.normpath(os.path.join(HERE, *([".."] * 6)))
SCENES = os.path.join(ROOT, "conformance", "suite", "scenes")

with open(os.path.join(HERE, "colrtest3.ttf"), "rb") as f:
    FONT = list(f.read())

# (scene suffix, char). The char selects the colour glyph (cmap a..f).
CASES = [
    ("01-composite-src-in", "a"),
    ("02-composite-xor", "b"),
    ("03-composite-plus", "c"),
    ("04-composite-multiply", "d"),
    ("05-composite-hard-light", "e"),
    ("06-nested-clip", "f"),
]


def scene(char):
    return {
        "version": 2,
        "size": [64, 64],
        "dpi": 96,
        "ops": [
            {"op": "clear", "color": [1.0, 1.0, 1.0, 1.0]},
            {"op": "color_glyph_run", "font": FONT, "text": char, "em_size": 50.0,
             "baseline": [12.0, 52.0], "palette": 0, "color": [0.0, 0.0, 0.0, 1.0]},
        ],
    }


def main():
    for suffix, char in CASES:
        path = os.path.join(SCENES, f"cf3-{suffix}.json")
        with open(path, "w") as f:
            json.dump(scene(char), f, separators=(",", ":"))
        print("wrote", os.path.basename(path))


if __name__ == "__main__":
    main()
