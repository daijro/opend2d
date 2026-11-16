#!/usr/bin/env python3
"""Generate the Direct2D-vs-OpenD2D showcase: docs/showcase/*.png and the table in README.md.

For each showcased scene: render it with opend2d through the drop-in d2d1.dll under Wine (our render),
read the captured native WARP oracle, and write direct2d.png / opend2d.png / a diff.png (black where
equal, amplified where not), with the byte-compare result. Every image is placed on the same 512x384
frame, scaled up by a whole number so pixels stay sharp. Rerun as conformance improves. Invoked by scripts/showcase.sh (which builds + stages the DLLs and d2d-ref.exe).

Public scenes only: the private corpus (canvas-*/textx-*/sundial) is never shown. Env: STAGE (dir with
d2d-ref.exe + our d2d1/dwrite/windowscodecs), WINEPREFIX, D2D_FONTS. Run with python3 -I."""
import json, os, pathlib, subprocess, sys
from PIL import Image, ImageChops

ROOT = pathlib.Path(__file__).resolve().parents[1]
STAGE = pathlib.Path(os.environ["STAGE"])
SUITE = next(pathlib.Path(p) for p in (os.environ.get("D2D_SUITE"), ROOT / "conformance/suite",
             pathlib.Path(subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True).stdout.strip() or ".").parent / "conformance/suite") if p and pathlib.Path(p, "scenes").is_dir())
ORACLE = SUITE / "oracle/win11-26100"
OUT = ROOT / "docs/showcase"
FRAME = (512, 384)  # every showcase image is this size; scenes are scaled up by a whole number to fit
AMP = 16        # diff amplification: a 1/255 difference shows as 16/255

# area, scene, one-line description. First 12 are byte-exact (in conformance/passing.txt); the last
# two are honest "not yet exact" examples shown with their max channel error.
SCENES = [
    ("antialiasing", "02-fractional-rect", "Antialiased rectangles at fractional coordinates"),
    ("curves", "05-curves-arcs", "Bézier curves and arcs"),
    ("strokes", "dash-01-caps", "Dashed strokes with caps"),
    ("gradients", "09-linear-gamma", "Gamma-correct linear gradient"),
    ("bitmaps", "bitmaps-04-magnify-cubic", "Bitmap scaled up with the cubic filter"),
    ("layers", "2a2-01-opacity-mask", "Layer with an opacity mask"),
    ("effects", "drawimage-fx-01-dest-over", "DrawImage compositing"),
    ("shaders", "custom-fx-01-crop-output-rect", "Custom pixel shader effect"),
    ("text", "textmodes-04-cleartype-mmm", "ClearType text: mmmmmmmmmmlli"),
    ("geometry", "comb-03-simplify-rect", "Combined and simplified geometry"),
    ("ink", "inkfast-01-round-multiseg-alias", "Ink stroke"),
    ("svg", "svg-01-rects", "SVG document"),
    ("bitmaps", "bitmaps-05-magnify-hqcubic", "Bitmap scaled up with the high-quality cubic filter"),
    ("effects", "fx-filters-04-shadow", "Shadow effect"),
]

def load_bgra(path, w, h):
    b = path.read_bytes()
    return b if len(b) == w * h * 4 else None

def straight_rgba(bgra, w, h):
    """Premultiplied BGRA8 → straight RGBA for display (matches the harness preview)."""
    img = Image.new("RGBA", (w, h))
    px = bytearray(w * h * 4)
    for i in range(0, len(bgra), 4):
        bl, g, r, a = bgra[i], bgra[i + 1], bgra[i + 2], bgra[i + 3]
        if a:
            r = min(255, (r * 255 + a // 2) // a); g = min(255, (g * 255 + a // 2) // a); bl = min(255, (bl * 255 + a // 2) // a)
        px[i:i + 4] = bytes((r, g, bl, a))
    img.frombytes(bytes(px))
    return img

def up(img):
    """Scale up by the largest whole number that fits FRAME and centre on a transparent FRAME canvas."""
    fw, fh = FRAME
    k = max(1, min(fw // img.width, fh // img.height))
    if img.width > fw or img.height > fh:
        s = min(fw / img.width, fh / img.height)
        big = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)
    else:
        big = img.resize((img.width * k, img.height * k), Image.NEAREST)
    canvas = Image.new("RGBA", FRAME, (0, 0, 0, 0))
    canvas.paste(big, ((fw - big.width) // 2, (fh - big.height) // 2))
    return canvas

def main():
    entries = []
    for area, name, desc in SCENES:
        oj = ORACLE / f"{name}.json"
        size = json.loads(oj.read_text())["size"]
        w, h = size
        ours_png = STAGE / f"sc_{name}.png"
        for ext in (".bgra", ".png", ".json"):
            ours_png.with_suffix(ext).unlink(missing_ok=True)
        subprocess.run(["wine", "d2d-ref.exe", str(SUITE / "scenes" / f"{name}.json"), str(ours_png), "--wine-smoke"],
                       cwd=STAGE, env=os.environ, capture_output=True, timeout=120)
        ours = load_bgra(ours_png.with_suffix(".bgra"), w, h)
        oracle = load_bgra(ORACLE / f"{name}.bgra", w, h)
        if ours is None or oracle is None:
            print(f"SKIP {name}: no render ({'ours' if ours is None else 'oracle'} missing)", file=sys.stderr)
            continue
        maxerr = max((abs(a - b) for a, b in zip(ours, oracle)), default=0)
        d = OUT / area
        d.mkdir(parents=True, exist_ok=True)
        nat, op = straight_rgba(oracle, w, h), straight_rgba(ours, w, h)
        up(nat).save(d / f"{name}.direct2d.png")
        up(op).save(d / f"{name}.opend2d.png")
        diff = ImageChops.difference(nat.convert("RGB"), op.convert("RGB")).point(lambda x: min(255, x * AMP))
        up(diff).save(d / f"{name}.diff.png")
        result = "identical" if maxerr == 0 else f"max error {maxerr}/255"
        entries.append((area, name, desc, result))
        print(f"{name:34} {w}x{h}  {result}")

    # One table in README.md, between the showcase markers.
    rows = ["| Scene | Direct2D | OpenD2D | Difference | Result |", "| --- | --- | --- | --- | --- |"]
    for area, name, desc, result in entries:
        rel = f"docs/showcase/{area}/{name}"
        img = lambda kind: f'<img src="{rel}.{kind}.png" width="240">'
        rows.append(f"| {desc} | {img('direct2d')} | {img('opend2d')} | {img('diff')} | {result} |")
    readme = ROOT / "README.md"
    start, end = "<!-- showcase:start -->", "<!-- showcase:end -->"
    t = readme.read_text()
    i, j = t.index(start) + len(start), t.index(end)
    readme.write_text(t[:i] + "\n" + "\n".join(rows) + "\n" + t[j:])
    print(f"\nwrote the README showcase table and {OUT}")

if __name__ == "__main__":
    main()
