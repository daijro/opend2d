#!/usr/bin/env python3
"""Generate colrtest3.ttf — a COLRv1 font exercising colour fonts part 3: `PaintComposite`
(non-SrcOver Porter-Duff and blend modes) and nested glyph clips.

Two base rectangles overlap in a middle band so each composite shows three regions
(backdrop-only, overlap, source-only); at em 50 / baseline (12,52) / 64² both land on whole
device pixels, so leaf coverage is flat {0,255} and only the composite/clip math is under test:
  rectL design x[100,600] y[100,700] → device x[17,42) y[17,47)
  rectR design x[400,900] y[100,700] → device x[32,57) y[17,47)   (overlap device x[32,42))

Run on the host after fontTools is available: `python3 gen_font3.py` → writes colrtest3.ttf next
to this file. Committed with its output; this task's own test asset (not a system font).
"""
import os
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

UPEM = 1000

# CPAL palette. Indices used by the paint trees below.
PALETTE = [
    (1.0, 0.0, 0.0, 1.0),   # 0 red (opaque)          — Porter-Duff backdrop
    (0.0, 0.0, 1.0, 0.6),   # 1 blue @ 60% alpha       — Porter-Duff source
    (1.0, 0.5, 0.0, 1.0),   # 2 orange (opaque)        — blend backdrop
    (0.0, 1.0, 1.0, 1.0),   # 3 cyan (opaque)          — blend source
    (0.0, 0.6, 0.0, 1.0),   # 4 green (opaque)         — clip fill
]

RECT_L = [(100, 100), (600, 100), (600, 700), (100, 700)]
RECT_R = [(400, 100), (900, 100), (900, 700), (400, 700)]


def rect_glyph(rect):
    pen = TTGlyphPen(None)
    pen.moveTo(rect[0])
    for p in rect[1:]:
        pen.lineTo(p)
    pen.closePath()
    return pen.glyph()


def empty_glyph():
    return TTGlyphPen(None).glyph()


def solid(index):
    return {"Format": 2, "PaletteIndex": index, "Alpha": 1.0}


def paint_glyph(glyph, paint):
    return {"Format": 10, "Glyph": glyph, "Paint": paint}


def composite(mode, src_pal, bg_pal):
    # source (rectR) <mode> backdrop (rectL), each a glyph-clipped solid fill.
    return {"Format": 32,
            "SourcePaint": paint_glyph("rectR", solid(src_pal)),
            "CompositeMode": mode,
            "BackdropPaint": paint_glyph("rectL", solid(bg_pal))}


# One colour glyph per case; cmap char selects it.
COLOR = {
    "cSrcIn": composite("src_in", 1, 0),       # Porter-Duff: source ∩ backdrop
    "cXor": composite("xor", 1, 0),            # Porter-Duff: symmetric difference
    "cPlus": composite("plus", 1, 0),          # Porter-Duff: additive (saturating)
    "cMul": composite("multiply", 3, 2),       # blend: multiply (separable)
    "cHard": composite("hard_light", 3, 2),    # blend: hard-light (asymmetric)
    # nested clip: fill rectR with green, clipped to rectL → their intersection band.
    "cClip": paint_glyph("rectL", paint_glyph("rectR", solid(4))),
}

CMAP = {ord(c): name for c, name in zip("abcdef", COLOR.keys())}
GLYPHS = [".notdef", "rectL", "rectR"] + list(COLOR.keys())


def main():
    fb = FontBuilder(UPEM, isTTF=True)
    fb.setupGlyphOrder(GLYPHS)
    fb.setupCharacterMap(CMAP)
    glyphs = {".notdef": empty_glyph(), "rectL": rect_glyph(RECT_L), "rectR": rect_glyph(RECT_R)}
    for name in COLOR:
        glyphs[name] = empty_glyph()
    fb.setupGlyf(glyphs)
    # lsb must equal each base glyph's xMin or the outline shifts relative to the (unshifted) paint.
    xmin = {"rectL": 100, "rectR": 400}
    fb.setupHorizontalMetrics({g: (1000, xmin.get(g, 0)) for g in GLYPHS})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "D2DColrTest3", "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200)
    fb.setupPost()
    fb.setupCPAL([[tuple(c) for c in PALETTE]])
    fb.setupCOLR(COLOR, version=1)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colrtest3.ttf")
    fb.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
