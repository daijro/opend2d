#!/usr/bin/env python3
"""Decisive drop-in-dwrite test: run the unchanged harness `d2d-ref.exe` against our drop-in
`d2d1.dll` + `dwrite.dll` for every text scene that renders byte-exact in the core, and compare to
the native WARP + DirectWrite oracle — raw BGRA bytes plus the per-op HRESULT/tag channel, EndDraw,
and text_metrics/text_layout query results. Exit non-zero on any mismatch.

Invoked by `decisive-test.sh`, which builds the DLLs + harness and stages them beside a fresh Wine
prefix with D2D_FONTS pointing at the VM's fonts. Run with `python3 -I`.
Env: STAGE (dir with d2d-ref.exe + d2d1.dll + dwrite.dll), WINEPREFIX, D2D_FONTS."""
import json, os, pathlib, struct, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
STAGE = pathlib.Path(os.environ["STAGE"])


def suite():
    """Conformance suite root (the conformance/suite submodule): D2D_SUITE, else the in-tree submodule,
    else the main checkout's copy. None ⇒ absent (public clone), so this decisive run skips."""
    for cand in (os.environ.get("D2D_SUITE"), str(ROOT / "conformance/suite"),
                 pathlib.Path(subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True).stdout.strip() or ".").parent / "conformance/suite"):
        if cand and (pathlib.Path(cand) / "scenes").is_dir():
            return pathlib.Path(cand)
    return None


SUITE = suite()
if SUITE is None:
    print("SKIP: conformance suite absent (conformance/suite submodule not initialised)")
    sys.exit(0)
ORACLE = SUITE / "oracle/win11-26100"

# The text scenes that render byte-exact in the core (cargo test -p d2d --test conformance with the
# VM fonts present): one glyph_run, DrawText glyph/font scenes, a text_metrics scene and the
# text_layout align scene. These are the drop-in dwrite + d2d1 decisive set.
SCENES = [
    "texte2e-01-gfdisabled-stem",
    "textx-font-249", "textx-fontquant",
    "textx-glyph-019", "textx-glyph-020", "textx-glyph-032", "textx-glyph-128",
    "textx-glyph-224", "textx-glyph-228", "textx-glyph-232", "textx-glyph-499",
    "textx-glyph-500", "textx-glyph-512", "textx-glyph-516", "textx-glyph-520",
    "textx-glyph-612", "textx-glyph-616",
    "text2b-01-align",
]

# Getter-readback scenes: these draw nothing (blank 16×16, pixels match the blank oracle trivially)
# and carry only `dwrite_query`/`get_prop` ops, so the real check is the query comparison — this is
# where the drop-in IDWriteTextFormat/Font/FontFace/TextLayout/FontFallback getters are verified
# against the native oracle. Auto-discovered from the suite (census2-dwrite-* + census-dw-*).
QUERY_SCENES = sorted(
    p.stem for p in (SUITE / "scenes").glob("*.json")
    if any(o.get("op") in ("dwrite_query",) for o in json.loads(p.read_text()).get("ops", []))
)
SCENES += QUERY_SCENES

# d2d1=n;dwrite=n: load our drop-in DLLs, not Wine's built-ins.
env = dict(os.environ, WINEDLLOVERRIDES="d2d1=n;dwrite=n;mscoree,mshtml=d", WINEDEBUG="-all")


def bits(a):
    return struct.unpack('<I', struct.pack('<f', float(a)))[0]


def cmp_query(kind, got, want):
    # text_metrics/text_layout: every value the oracle recorded must match exactly. We match on the
    # oracle's own keys and ignore keys only our (newer) harness adds — some committed oracles were
    # captured before the harness gained the run `font` name and the `hit_points`/`hit_ranges`
    # query fields, and those additive fields are not a rendering/layout divergence.
    def matches(g, w):
        if isinstance(w, dict):
            return isinstance(g, dict) and all(k in g and matches(g[k], v) for k, v in w.items())
        if isinstance(w, list):
            return isinstance(g, list) and len(g) == len(w) and all(matches(a, b) for a, b in zip(g, w))
        return g == w
    return matches(got, want)


npass, fails = 0, []
for s in SCENES:
    out = STAGE / f"o_{s}.png"
    r = subprocess.run(["wine", "d2d-ref.exe", str(SUITE / "scenes" / f"{s}.json"), str(out), "--wine-smoke"],
                       cwd=STAGE, env=env, capture_output=True, timeout=120)
    bgra = out.with_suffix(".bgra")
    if not bgra.exists():
        fails.append((s, "no output: " + r.stderr.decode('utf-8', 'replace')[-200:]))
        continue
    a, b = bgra.read_bytes(), (ORACLE / f"{s}.bgra").read_bytes()
    if a != b:
        i = next(k for k in range(min(len(a), len(b))) if a[k] != b[k])
        fails.append((s, f"pixel byte {i}: got {a[i]} want {b[i]} (len {len(a)} vs {len(b)})"))
        continue
    got = json.loads(out.with_suffix(".json").read_text())
    want = json.loads((ORACLE / f"{s}.json").read_text())
    why = None
    for g, w in zip(got["ops"], want["ops"]):
        if (g["hr"], g["tag1"], g["tag2"]) != (w["hr"], w["tag1"], w["tag2"]):
            why = f"op {g['op']} got ({g['hr']},{g['tag1']},{g['tag2']}) want ({w['hr']},{w['tag1']},{w['tag2']})"
            break
    gd, wd = got["end_draw"], want["end_draw"]
    if why is None and (gd["hr"], gd["tag1"], gd["tag2"]) != (wd["hr"], wd["tag1"], wd["tag2"]):
        why = f"end_draw got {gd} want {wd}"
    wq = {q["op"]: q for q in want.get("queries", [])}
    for q in got.get("queries", []):
        if why:
            break
        w = wq.get(q["op"])
        if not w or not cmp_query(q["kind"].split(':')[0], q["result"], w["result"]):
            why = f"query op {q['op']} ({q['kind']}) mismatch"
    if why:
        fails.append((s, why))
    else:
        npass += 1

print(f"\nDECISIVE: {npass}/{len(SCENES)} text scenes byte-exact + HRESULT/tag/query-exact through d2d1.dll + dwrite.dll")
for s, why in fails:
    print("  FAIL", s, "--", why)
sys.exit(1 if fails else 0)
