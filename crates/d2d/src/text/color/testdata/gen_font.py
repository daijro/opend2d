#!/usr/bin/env python3
"""Generate colrtest.ttf — a minimal COLRv1 font with one glyph per paint type.

Each colour glyph is `PaintGlyph(<brush>, "rect")`: the brush under test filling one
shared axis-aligned rectangle outline. upem is 1000; a scene draws at em 50 (scale 0.05)
with the rectangle's design coords on a 20-unit grid, so every edge lands on a device
pixel boundary and the leaf coverage is a flat {0,255} regardless of the rasterizer —
isolating the paint/brush math from the (separately-owned) glyph-coverage gate.

Run on the host: `python3 gen_font.py` → writes colrtest.ttf next to this file.
Committed with its output; it is this task's own test asset (not a system font).
"""
import os
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

UPEM = 1000

# CPAL palette. 0xFFFF as a stop/solid palette index means "text foreground colour".
PALETTE = [
    (1.0, 0.0, 0.0, 1.0),   # 0 red
    (0.0, 1.0, 0.0, 1.0),   # 1 green
    (0.0, 0.0, 1.0, 1.0),   # 2 blue
    (1.0, 1.0, 0.0, 1.0),   # 3 yellow
    (1.0, 0.0, 1.0, 0.5),   # 4 magenta @ 50% alpha
    (0.0, 0.0, 0.0, 1.0),   # 5 black
    (1.0, 1.0, 1.0, 1.0),   # 6 white
]

# The shared rectangle outline, design units on a 20-unit grid (pixel-aligned at em 50).
RECT = [(100, 100), (900, 100), (900, 700), (100, 700)]


def rect_glyph():
    pen = TTGlyphPen(None)
    pen.moveTo(RECT[0])
    for p in RECT[1:]:
        pen.lineTo(p)
    pen.closePath()
    return pen.glyph()


def empty_glyph():
    return TTGlyphPen(None).glyph()


def stops(pairs, extend="pad"):
    return {"Extend": extend, "ColorStop": [
        {"StopOffset": o, "PaletteIndex": i, "Alpha": a} for (o, i, a) in pairs]}


# Three-stop ramp red→green→blue spanning the rect.
RAMP = [(0.0, 0, 1.0), (0.5, 1, 1.0), (1.0, 2, 1.0)]


def paint_glyph(brush):
    return {"Format": 10, "Glyph": "rect", "Paint": brush}


# One colour glyph per paint type. Gradient geometry is in font design units (the paint
# tree's own space), transformed to device by the same base matrix as the outline.
COLOR = {
    # solid: flat palette colour 2 (blue).
    "colA": paint_glyph({"Format": 2, "PaletteIndex": 2, "Alpha": 1.0}),
    # linear gradient left→right across the rect.
    "colB": paint_glyph({"Format": 4, "ColorLine": stops(RAMP),
                         "x0": 100, "y0": 400, "x1": 900, "y1": 400, "x2": 100, "y2": 500}),
    # radial A: r0 == 0, focal at the centre (D2D radial-brush case).
    "colC": paint_glyph({"Format": 6, "ColorLine": stops(RAMP),
                         "x0": 500, "y0": 400, "r0": 0, "x1": 500, "y1": 400, "r1": 400}),
    # radial B: two-circle, r0 != 0, offset centres (the conical-brush case).
    "colD": paint_glyph({"Format": 6, "ColorLine": stops(RAMP),
                         "x0": 360, "y0": 400, "r0": 80, "x1": 560, "y1": 400, "r1": 320}),
    # sweep gradient around the centre.
    "colE": paint_glyph({"Format": 8, "ColorLine": stops(RAMP),
                         "centerX": 500, "centerY": 400, "startAngle": 0, "endAngle": 360}),
    # composite: blue source SRC_OVER a red backdrop (both glyph-clipped to the rect).
    "colF": {"Format": 32,
             "SourcePaint": paint_glyph({"Format": 2, "PaletteIndex": 2, "Alpha": 1.0}),
             "CompositeMode": "src_over",
             "BackdropPaint": paint_glyph({"Format": 2, "PaletteIndex": 0, "Alpha": 1.0})},
    # transform: scale 0.6 about origin applied to a solid green fill. 0.6 maps the rectangle's
    # 20-unit grid to a 12-unit one, i.e. still whole device pixels at em 50 — so the scaled outline
    # stays pixel-aligned and its coverage flat, isolating the transform from the AA-coverage gate.
    "colG": {"Format": 12, "Paint": paint_glyph({"Format": 2, "PaletteIndex": 1, "Alpha": 1.0}),
             "Transform": {"xx": 0.6, "yx": 0.0, "xy": 0.0, "yy": 0.6, "dx": 0.0, "dy": 0.0}},
    # PaintColrGlyph reuse: reference colB's whole paint tree.
    "colH": {"Format": 11, "Glyph": "colB"},
    # variable alpha: solid magenta already carries a 0.5 CPAL alpha, times a 0.5 stop alpha.
    "colI": paint_glyph({"Format": 2, "PaletteIndex": 4, "Alpha": 0.5}),
}

CMAP = {ord(c): "col" + c for c in "ABCDEFGHI"}
GLYPHS = [".notdef", "rect"] + list(COLOR.keys())


def main():
    fb = FontBuilder(UPEM, isTTF=True)
    fb.setupGlyphOrder(GLYPHS)
    fb.setupCharacterMap(CMAP)
    glyphs = {".notdef": empty_glyph(), "rect": rect_glyph()}
    for name in COLOR:
        glyphs[name] = empty_glyph()
    fb.setupGlyf(glyphs)
    # lsb must equal each glyph's xMin, or DirectWrite/skrifa shift the outline by (lsb-xMin)
    # while leaving the COLR paint coordinates unshifted — misaligning brush and outline. The
    # rectangle's xMin is 100; the empty base glyphs have none (0).
    fb.setupHorizontalMetrics({g: (1000, min((p[0] for p in RECT), default=0) if g == "rect" else 0) for g in GLYPHS})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "D2DColrTest", "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200)
    fb.setupPost()
    fb.setupCPAL([[tuple(c) for c in PALETTE]])
    fb.setupCOLR(COLOR, version=1)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colrtest.ttf")
    fb.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
