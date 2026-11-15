#!/usr/bin/env python3
"""Decisive glyph-run-analysis probe: run the unchanged harness `dwrite-dump.exe` through OUR
drop-in `dwrite.dll` under Wine (so every `IDWriteFactory3::CreateGlyphRunAnalysis` /
`IDWriteGlyphRunAnalysis::{GetAlphaTextureBounds,CreateAlphaTexture,GetAlphaBlendParams}` call —
plus the `GetSystemFontCollection`/`CreateFontFace`/`GetGlyphIndices`/`GetDesignGlyphMetrics`
front end — hits our implementation) and compare the alpha textures + bounds byte-for-byte to the
native `dwrite-dump` captures in the suite's `fixtures/text_raster/fixtures`.

Those fixtures are native-`dwrite.dll` output on the reference VM (grid-fit DISABLED), so this is a
VM-free native comparison. Invoked by `analysis-test.sh`. Run with `python3 -I`.
Env: STAGE (dir with dwrite-dump.exe + dwrite.dll), WINEPREFIX, D2D_FONTS."""
import json, os, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
STAGE = pathlib.Path(os.environ["STAGE"])
SUITE = pathlib.Path(next(p for p in (os.environ.get("D2D_SUITE"), ROOT / "conformance/suite", pathlib.Path(subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True).stdout.strip() or ".").parent / "conformance/suite") if p and pathlib.Path(p, "scenes").is_dir()))
FIX = SUITE / "fixtures/text_raster/fixtures"

# Single-glyph Arial fixtures (grid-fit DISABLED, natural-symmetric). Name = <char><em>. The core
# `text::raster` is byte-exact vs native on GRAY_EXACT / CT_EXACT and within one 4x4 sample
# elsewhere (dropout control, see the raster module); the COM path must reproduce that exactly.
FIXTURES = ["o20", "H20", "e20", "g20", "o16", "R40", "a13", "vo34", "veacute34"]
GRAY_EXACT = {"H20", "e20", "a13", "vo34", "veacute34"}
CT_EXACT = {"H20", "a13", "vo34", "veacute34"}

env = dict(os.environ, WINEDLLOVERRIDES="dwrite=n;mscoree,mshtml=d", WINEDEBUG="-all")


def text_of(name):
    return re.match(r"([^\d]+)", name).group(1)  # strip the em-size suffix


def run_dump(name, fx):
    spec = {
        "font": fx.get("family", "Arial"),  # font FAMILY name; the fixture's "font" is the capture's file
        "em_size": fx["em_size"],
        "dpi": fx.get("dpi", 96.0),
        "text": fx.get("text", text_of(name)),
        "baseline": fx["baseline"],
        "measuring": fx.get("measuring", 0),
        "rendering_mode": fx.get("rendering_mode", 5),
        "grid_fit": fx.get("grid_fit", 1),
        "transform": fx.get("transform"),
    }
    spec_path = STAGE / f"a_{name}.json"
    spec_path.write_text(json.dumps(spec))
    prefix = STAGE / f"a_{name}"
    r = subprocess.run(["wine", "dwrite-dump.exe", str(spec_path), str(prefix)],
                       cwd=STAGE, env=env, capture_output=True, timeout=120)
    if r.returncode != 0:
        return None, r.stderr.decode("utf-8", "replace")[-300:]
    return json.loads(r.stdout.decode("utf-8")), None


npass, fails, notes = 0, [], []
for name in FIXTURES:
    fx = json.loads((FIX / f"{name}.json").read_text())
    out, err = run_dump(name, fx)
    if out is None:
        fails.append((name, f"dwrite-dump failed: {err}"))
        continue
    ok = True
    # Bounds (control-point box) must be byte-exact on every fixture.
    for tex in ("gray", "cleartype"):
        want = fx[tex]["bounds"]
        got = out[tex]["bounds"] if out[tex] else None
        if got != want:
            fails.append((name, f"{tex} bounds got {got} want {want}"))
            ok = False
    # Texture bytes: exact on the known-exact set; report max |delta| otherwise.
    for tex, bpp, exact_set in (("gray", 1, GRAY_EXACT), ("ct", 3, CT_EXACT)):
        got = (STAGE / f"a_{name}.{tex}.bin")
        want = FIX / f"{name}.{tex}.bin"
        if not want.exists() or not got.exists():
            continue
        a, b = got.read_bytes(), want.read_bytes()
        if len(a) != len(b):
            fails.append((name, f"{tex} size {len(a)} vs {len(b)}"))
            ok = False
            continue
        md = max((abs(x - y) for x, y in zip(a, b)), default=0)
        if name in exact_set:
            if a != b:
                fails.append((name, f"{tex} bytes differ (max |Δ|={md})"))
                ok = False
        elif md:
            notes.append(f"{name} {tex}: max |Δ|={md} (core scan residual)")
    if ok:
        npass += 1

print(f"\nANALYSIS: {npass}/{len(FIXTURES)} glyph-run-analysis fixtures bounds-exact + "
      f"byte-exact on the exact set, through our dwrite.dll under Wine")
for n in notes:
    print("  note", n)
for name, why in fails:
    print("  FAIL", name, "--", why)

# --- dwrite-probe: the font-file + text-analyzer surface dwrite-dump doesn't cover ---------------
# Oracle = the same probe run natively on the reference VM (arial.ttf "HeogaR"): confirmed identical
# to these values except match_count (native 5 — DirectWrite's WWS family grouping folds Arial Black,
# absent from the committed font subset, into "Arial"; ours lists the 4 real Arial faces present).
EXPECT = {
    "analyze": {"supported": True, "file_type": 2, "face_type": 1, "num_faces": 1},
    "cmap_ids": [43, 72, 82, 74, 68, 53],              # H e o g a R (native GetGlyphIndices)
    "analyzer_glyphs": [43, 72, 82, 74, 68, 53],        # IDWriteTextAnalyzer::GetGlyphs
    "analyzer_advances": [11.554688, 8.898438, 8.898438, 8.898438, 8.898438, 11.554688],  # GetGlyphPlacements @ em 16
    # CreateRenderingParams system defaults (native on the reference VM):
    "rparams": {"gamma": 1.8, "enhanced": 0.5, "grayscale": 1.0, "cleartype": 1.0, "pixel_geometry": 1, "rendering_mode": 0},
    # CreateTextLayout("Hello World Wide") DetermineMinWidth + GetMetrics width (native on the VM):
    "min_width": 51.865234,
    "metrics_width": 154.121094,
    # MapCharacters "HeogaR" base "Arial" + a base collection: the base run, Arial (native VM). The
    # built fallback maps the whole BMP to Arial. (With a NULL base collection native instead returns
    # its Latin default "Segoe UI"; D2D always passes a collection, the case verified here.)
    "fallback_sys": [6, "Arial"],
    "fallback_built": [6, "Arial"],
    # GetOverhangMetrics of the "Hello World Wide" layout in the 1000x1000 box (native VM):
    "overhang": [-1.601562, -4.443359, -846.708984, -981.005859],
    # GetFontDownloadQueue: always-empty local queue (IsEmpty TRUE, generation 0) — native VM.
    "dq": [True, 0],
    # A font set built from arial.ttf's face reference: GetFontCount, GetPropertyValues(FAMILY_NAME)
    # count, first family — native VM.
    "fontset": [1, 1, "Arial"],
    # CreateFontCollectionFromFontSet → GetFontFamilyCount + the Collection1/FontFamily1/Font3 chain
    # (GetFontFamily1(0).GetFont(0).GetFontFamily().GetFamilyNames()) — native VM.
    "coll_from_set": [1, "Arial"],
    # TranslateColorGlyphRun on the COLRv0 emoji U+1F600 (Segoe UI Emoji): sub-run count + each
    # [layer glyph, palette index, r,g,b,a] — native VM.
    "color_run": [6, [[45442, 64399, 1.0, 0.6902, 0.1804, 1.0], [45439, 64291, 0.7333, 0.1137, 0.502, 1.0], [45440, 64417, 1.0, 1.0, 1.0, 1.0], [45441, 64417, 1.0, 1.0, 1.0, 1.0], [45443, 64174, 0.251, 0.1647, 0.1961, 1.0], [45444, 64174, 0.251, 0.1647, 0.1961, 1.0]]],
    # GdiInterop ConvertFontFaceToLOGFONT(Arial) [faceName, weight, italic, charset] + a 48x24 bitmap
    # render target [sizeX, sizeY, pixelsPerDip] — native VM. (DrawGlyphRun's GDI DIB blend is
    # best-effort, not pixel-compared here — see win/gdi.rs.)
    "logfont": ["Arial", 400, 0, 1],
    "brt": [48, 24, 1.0],
    # CreateEllipsisTrimmingSign(Arial 20) GetMetrics [width, height, baseline] + a NO_WRAP layout of
    # "Hello World Wide" in a 60px box, character-trimmed with the sign: [lineCount, line0 length,
    # isTrimmed, trimmed width] — native VM.
    "ellipsis": [20.0, 0.0, 0.0],
    "trim": [1, 16, True, 54.4531],
    # TextAnalyzer GetGdiCompatibleGlyphPlacements (ppd 1, 16px, whole-pixel advances),
    # AnalyzeLineBreakpoints (per-char break bitfields), AnalyzeNumberSubstitution (sink fires 0×
    # for Latin "HeogaR") — native VM.
    "gdi_advances": [11.0, 9.0, 9.0, 9.0, 9.0, 11.0],
    "line_breaks": [10, 10, 10, 10, 10, 6],
    "numsub_calls": 0,
    # FontDownloadQueue AddListener (1-based token) + RemoveListener(token) — native VM.
    "dq_listener": [1, True],
    # FontFallbackBuilder::AddMappings: merge the built fallback into a fresh builder; it maps
    # "HeogaR" to Arial exactly like the source (native VM).
    "fallback_merged": [6, "Arial"],
}
arial = os.path.join(os.environ["D2D_FONTS"], "arial.ttf")
pr = subprocess.run(["wine", "dwrite-probe.exe", arial, "HeogaR"], cwd=STAGE, env=env, capture_output=True, timeout=120)
pfail = []
try:
    got = json.loads(pr.stdout.decode("utf-8").strip().splitlines()[-1])
    for k, want in EXPECT.items():
        if got.get(k) != want:
            pfail.append(f"{k}: got {got.get(k)} want {want}")
    if got.get("match_count", 0) < 4:
        pfail.append(f"match_count {got.get('match_count')} < 4")
except Exception as e:
    pfail.append(f"no probe output: {e}; stderr {pr.stderr.decode('utf-8','replace')[-200:]}")
print("\nPROBE: font-file (CreateFontFileReference/Analyze/CreateFontFace) + text analyzer "
      f"(GetGlyphs/GetGlyphPlacements) {'OK' if not pfail else 'FAIL'} vs native")
for f in pfail:
    print("  FAIL", f)

sys.exit(1 if fails or pfail else 0)
