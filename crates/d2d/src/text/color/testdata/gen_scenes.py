#!/usr/bin/env python3
"""Emit the colorfont2-NN per-paint-type scenes into the suite (conformance/suite/scenes/).

Each scene is self-contained: it carries colrtest.ttf inline (a flat byte array, like the WicBitmap
op's `rgba`) and draws one glyph of it with the `color_glyph_run` op (DrawPaintGlyphRun). em 50 at
baseline (12,52) on a 64×64 target puts the shared rectangle outline at device x[17,57) y[17,47),
pixel-aligned, so leaf coverage is flat {0,255} and only the paint/brush math is under test.

Run on the host after gen_font.py: `python3 gen_scenes.py`.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# testdata → color → text → src → d2d → crates → repo root
ROOT = os.path.normpath(os.path.join(HERE, *([".."] * 6)))
SCENES = os.path.join(ROOT, "conformance", "suite", "scenes")

with open(os.path.join(HERE, "colrtest.ttf"), "rb") as f:
    FONT = list(f.read())

# (scene suffix, char, note). The char selects the colour glyph (cmap A..I).
CASES = [
    ("01-solid", "A"),
    ("02-linear", "B"),
    ("03-radial-focal", "C"),
    ("04-radial-two-circle", "D"),
    ("05-transform", "G"),
    ("06-composite-srcover", "F"),
    ("07-colr-reuse", "H"),
    ("08-variable-alpha", "I"),
    ("09-sweep", "E"),
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
        path = os.path.join(SCENES, f"colorfont2-{suffix}.json")
        with open(path, "w") as f:
            json.dump(scene(char), f, separators=(",", ":"))
        print("wrote", os.path.basename(path))


if __name__ == "__main__":
    main()
