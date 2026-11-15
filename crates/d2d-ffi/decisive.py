#!/usr/bin/env python3
"""Decisive 3A test: run the unchanged harness `d2d-ref.exe` against our drop-in `d2d1.dll` for
every scene in `conformance/passing.txt` and compare to the native WARP oracle — raw BGRA bytes plus
the per-op HRESULT/tag channel, EndDraw, and geometry query results. Exit non-zero on any mismatch.

Invoked by `decisive-test.sh`, which builds the DLL + harness and stages them beside a fresh Wine
prefix. Run with `python3 -I`. Env: STAGE (dir with d2d-ref.exe + d2d1.dll), WINEPREFIX."""
import json, os, pathlib, re, struct, subprocess, sys

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
scenes = [l.strip() for l in (SUITE / "passing.txt").read_text().splitlines()
          if l.strip() and not l.startswith('#')]
# Force our d2d1.dll and dwrite.dll native (both staged beside the exe, so they win over the prefix's
# copies); our dwrite must match this build or text rendering (ITextInterop) breaks.
env = dict(os.environ, WINEDLLOVERRIDES="d2d1,dwrite=n;mscoree,mshtml=d", WINEDEBUG="-all")

def bits(a): return struct.unpack('<I', struct.pack('<f', float(a)))[0]
def ulp(a, b): return abs(struct.unpack('<i', struct.pack('<f', float(a)))[0] -
                          struct.unpack('<i', struct.pack('<f', float(b)))[0])
def bbit(v): return 0x7fc0_0000 if v is None else bits(v)  # non-finite bound → one NaN sentinel
def cmp_query(kind, got, want):
    # Mirror crates/d2d/tests/conformance.rs: area ≤1 ULP, length/bounds exact bits, else exact.
    if kind == 'area': return ulp(got, want) <= 1
    if kind == 'length': return bits(got) == bits(want)
    if kind == 'bounds': return all(bits(g) == bits(w) for g, w in zip(got, want))
    if kind == 'geometry_op':
        # Exact figure/segment counts and exact-bit bounds; the output `stream` only when the oracle
        # has it (it predates the recorder for some scenes — conformance.rs::compare_stream).
        if got.get('figures') != want.get('figures') or got.get('segments') != want.get('segments'):
            return False
        if not all(bbit(g) == bbit(w) for g, w in zip(got.get('bounds', []), want.get('bounds', []))):
            return False
        ws = want.get('stream')
        return ws is None or got.get('stream') == ws
    return got == want

npass, fails = 0, []
for s in scenes:
    out = STAGE / f"o_{s}.png"
    for ext in (".bgra", ".png", ".json"):  # drop any stale output so we only judge this run
        out.with_suffix(ext).unlink(missing_ok=True)
    try:
        r = subprocess.run(["wine", "d2d-ref.exe", str(SUITE / "scenes" / f"{s}.json"),
                            str(out), "--wine-smoke"], cwd=STAGE, env=env, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        # One hung scene must not abort the whole run — reap our own lingering wine (matched by this
        # run's unique STAGE path, so other agents on the shared prefix are untouched), record it as a
        # DLL-fail, and move on.
        subprocess.run(["pkill", "-9", "-f", str(STAGE)], capture_output=True)
        fails.append((s, "timeout (>60s)")); continue
    oj, bgra, oracle_bgra = out.with_suffix(".json"), out.with_suffix(".bgra"), ORACLE / f"{s}.bgra"
    if not oj.exists():
        fails.append((s, "no output: " + r.stderr.decode('utf-8', 'replace')[-160:])); continue
    # A negative scene whose native EndDraw failed has JSON only (no oracle pixels); compare its
    # HRESULTs/tags only, as crates/d2d/tests/conformance.rs does. Otherwise byte-compare our pixels.
    if oracle_bgra.exists():
        if not bgra.exists():
            fails.append((s, "no output: " + r.stderr.decode('utf-8', 'replace')[-160:])); continue
        a, b = bgra.read_bytes(), oracle_bgra.read_bytes()
        if a != b:
            i = next(k for k in range(min(len(a), len(b))) if a[k] != b[k])
            fails.append((s, f"pixel byte {i}: got {a[i]} want {b[i]}")); continue
    got, want, why = json.loads(oj.read_text()), json.loads((ORACLE / f"{s}.json").read_text()), None
    for g, w in zip(got["ops"], want["ops"]):
        if (g["hr"], g["tag1"], g["tag2"]) != (w["hr"], w["tag1"], w["tag2"]):
            why = f"op {g['op']} got ({g['hr']},{g['tag1']},{g['tag2']}) want ({w['hr']},{w['tag1']},{w['tag2']})"; break
    gd, wd = got["end_draw"], want["end_draw"]
    if why is None and (gd["hr"], gd["tag1"], gd["tag2"]) != (wd["hr"], wd["tag1"], wd["tag2"]):
        why = f"end_draw got {gd} want {wd}"
    wq = {q["op"]: q for q in want.get("queries", [])}
    for q in got.get("queries", []):
        if why: break
        w = wq.get(q["op"])
        if not w or not cmp_query(q["kind"].split(':')[0], q["result"], w["result"]):
            why = f"query op {q['op']} ({q['kind']}) got {q['result']} want {w['result'] if w else None}"
    fails.append((s, why)) if why else None
    npass += why is None

print(f"\nDECISIVE: {npass}/{len(scenes)} scenes byte-exact + HRESULT/tag/query-exact through d2d1.dll")

# Per-family counts (family = scene name up to its first `-<digit>`). Every scene here passes in the
# core (it is in passing.txt), so any family not at full pass is the core-pass / DLL-fail gap.
def family(s): return re.sub(r'-\d.*$', '', s)
total, passed = {}, {}
failset = {s for s, _ in fails}
for s in scenes:
    f = family(s)
    total[f] = total.get(f, 0) + 1
    passed[f] = passed.get(f, 0) + (s not in failset)
gaps = sorted(f for f in total if passed[f] != total[f])
print("\nPer-family counts (families with a core-pass / DLL-fail gap):")
for f in gaps:
    print(f"  {f:28} {passed[f]}/{total[f]}")
if not gaps:
    print("  (none — every family passes in the DLL)")

print(f"\nPasses in core but fails in DLL ({len(fails)}):")
for s, why in sorted(fails):
    print("  FAIL", s, "--", why)
sys.exit(1 if fails else 0)
