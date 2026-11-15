#!/usr/bin/env python3
"""FFI conformance mode (group 5): run EVERY oracle scene through the drop-in d2d1.dll/dwrite.dll via
the unchanged d2d-ref.exe under Wine, and report three things the coordinator asked for:
  (a) wall-clock runtime,
  (b) scenes where the FFI pixels differ from the CORE pixels — any such diff is a C-API plumbing
      bug (the COM method didn't route to the core the way the core test does), whatever native shows,
  (c) FFI-vs-oracle byte-exact pass count.

Env: STAGE (dir with d2d-ref.exe + d2d1.dll + dwrite.dll), CORE_DUMP (dir of <scene>.core.bgra from
`D2D_CORE_DUMP=<dir> cargo test -p d2d --test conformance`), D2D_FONTS. Both modes run in their own
throwaway Wine prefix under $TMPDIR, so $WINEPREFIX is ignored. Run with python3 -I.

`--gate`: run only a covering subset (≥1 scene per scene op and per overload, plus every scene whose
C-API entry point changed since GATE_BASE — default origin/main) in a fresh throwaway Wine prefix,
aiming for ~2 min, for the per-merge gate. Without it, the full oracle corpus runs."""
import json, os, pathlib, shutil, subprocess, sys, tempfile, time

GATE = "--gate" in sys.argv
ROOT = pathlib.Path(__file__).resolve().parents[2]
STAGE = pathlib.Path(os.environ["STAGE"])
CORE = pathlib.Path(os.environ["CORE_DUMP"])
SUITE = pathlib.Path(os.environ.get("D2D_SUITE", ROOT / "conformance/suite"))
ORACLE = SUITE / "oracle/win11-26100"

# The scene ops each scene uses, as tags: the op name, and `op#overload` for an overload field — so a
# covering subset exercises every op and every overload at least once.
def scene_tags(name):
    try:
        j = json.loads((SUITE / "scenes" / f"{name}.json").read_text())
    except Exception:
        return set()
    tags = set()
    for op in j.get("ops", []):
        t = op.get("op")
        if t:
            tags.add(t)
            if op.get("overload"):
                tags.add(f"{t}#{op['overload']}")
    return tags

all_scenes = sorted(p.stem for p in ORACLE.glob("*.bgra"))
if GATE:
    tags = {s: scene_tags(s) for s in all_scenes}
    # Greedy set cover: smallest scenes first so each covers few tags and the subset stays tight.
    need = set().union(*tags.values()) if tags else set()
    covered, chosen = set(), []
    for s in sorted(all_scenes, key=lambda s: (len(tags[s]), s)):
        if tags[s] - covered:
            chosen.append(s)
            covered |= tags[s]
    # Plus every scene whose C-API entry point changed since GATE_BASE: ops named (snake or the draw
    # verb) in the diff of the FFI/harness since the baseline. Keeps changed code broadly exercised.
    base = os.environ.get("GATE_BASE", "origin/main")
    diff = subprocess.run(["git", "-C", str(ROOT), "diff", f"{base}...HEAD", "--", "crates/d2d-ffi", "crates/d2d/src", "harness/src"],
                          capture_output=True, text=True).stdout.lower()
    changed = {s for s in all_scenes for t in tags[s] if t.split('#')[0] in diff}
    scenes = sorted(set(chosen) | changed)
else:
    scenes = all_scenes
    # FFI_OFFSET/FFI_LIMIT slice the corpus so the full run can be chunked under a wall-clock cap
    # (each chunk its own prefix); the per-chunk reports are additive — sum (b) and concat no-output.
    off = int(os.environ.get("FFI_OFFSET", 0))
    if off or os.environ.get("FFI_LIMIT"):
        lim = int(os.environ.get("FFI_LIMIT", len(scenes)))
        scenes = scenes[off : off + lim]

# Prefix setup (both modes): always make a throwaway prefix in the system temp dir so concurrent runs
# never race a shared one. It must sit on a filesystem whose wineserver socket dir (/tmp/.wine-$UID/
# server-<dev>-<inode>) is creatable — the default /tmp works; a sandbox that makes that path
# read-only (or forbids the AF_UNIX socket) breaks Wine outright, so run this unsandboxed. We wineboot
# the prefix, hold a persistent wineserver (else it exits + restarts between scenes, ~6s/scene), and
# warm the client once (the first `wine` call pays the whole PE-map + DLL-load + font-scan cost and
# would otherwise time the first scene out — a spurious no-output).
gate_prefix = tempfile.mkdtemp(prefix="wine-ffi-")
os.environ["WINEPREFIX"] = gate_prefix
benv = dict(os.environ, WINEDLLOVERRIDES="mscoree,mshtml=d", WINEDEBUG="-all")
subprocess.run(["wineboot", "-u"], env=benv, capture_output=True, timeout=120)
# Hold the server persistent (no capture: the persistent daemon would keep a captured pipe open and
# the call would never return — send its fds to /dev/null instead).
subprocess.run(["wineserver", "-p"], env=benv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
env = dict(os.environ, WINEDLLOVERRIDES="d2d1,dwrite=n;mscoree,mshtml=d", WINEDEBUG="-all")
if scenes:
    try:
        subprocess.run(["wine", "d2d-ref.exe", str(SUITE / "scenes" / f"{scenes[0]}.json"), str(STAGE / "warmup.png"), "--wine-smoke"],
                       cwd=STAGE, env=env, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-9", "-f", str(STAGE)], capture_output=True)
t0 = time.time()
ffi_core_bugs, ffi_oracle_pass, no_output, considered, ffi_oracle_fail = [], 0, [], 0, []
for s in scenes:
    out = STAGE / f"o_{s}.png"
    for ext in (".bgra", ".png", ".json"):
        out.with_suffix(ext).unlink(missing_ok=True)
    try:
        subprocess.run(["wine", "d2d-ref.exe", str(SUITE / "scenes" / f"{s}.json"), str(out), "--wine-smoke"],
                       cwd=STAGE, env=env, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-9", "-f", str(STAGE)], capture_output=True)
        no_output.append((s, "timeout")); continue
    ffi = out.with_suffix(".bgra")
    if not ffi.exists():
        # No pixels: a negative scene (native EndDraw failed, oracle has none) is expected; else a fail.
        if not (ORACLE / f"{s}.bgra").read_bytes():
            pass
        else:
            no_output.append((s, "no FFI output"))
        continue
    fb = ffi.read_bytes()
    # (b) FFI vs CORE — the C-API correctness check.
    core_p = CORE / f"{s}.core.bgra"
    if core_p.exists():
        considered += 1
        cb = core_p.read_bytes()
        if fb != cb:
            i = next((k for k in range(min(len(fb), len(cb))) if fb[k] != cb[k]), -1)
            d = abs(fb[i] - cb[i]) if i >= 0 else -1
            ffi_core_bugs.append((s, f"byte {i}: ffi {fb[i] if i>=0 else '?'} vs core {cb[i] if i>=0 else '?'} (len {len(fb)}/{len(cb)}, Δ{d})"))
    # (c) FFI vs ORACLE — the native byte-exact check.
    ob = (ORACLE / f"{s}.bgra").read_bytes()
    if ob and fb == ob:
        ffi_oracle_pass += 1
    elif ob:
        # Classify: a fail where FFI == CORE is a core residual (not an FFI bug); FFI != CORE is a
        # real C-API plumbing bug (also in (b) when a core dump exists).
        cls = "FFI==CORE (core residual)" if core_p.exists() and fb == core_p.read_bytes() else ("FFI!=CORE (C-API bug)" if core_p.exists() else "no core dump")
        i = next((k for k in range(min(len(fb), len(ob))) if fb[k] != ob[k]), -1)
        ffi_oracle_fail.append((s, f"byte {i} ffi {fb[i] if i>=0 else '?'} vs oracle {ob[i] if i>=0 else '?'} (len {len(fb)}/{len(ob)}) [{cls}]"))
secs = time.time() - t0
# Release the persistent server and remove our throwaway prefix.
subprocess.run(["wineserver", "-k"], env=dict(os.environ, WINEDEBUG="-all"), capture_output=True, timeout=30)
shutil.rmtree(gate_prefix, ignore_errors=True)

mode = f"GATE subset ({len(scenes)} of {len(all_scenes)})" if GATE else f"{len(scenes)} oracle scenes"
print(f"\n=== FFI conformance mode: {mode}, {secs:.0f}s ===")
print(f"(a) runtime: {secs:.1f}s  ({secs/max(len(scenes),1)*1000:.0f} ms/scene)")
print(f"(b) FFI-vs-core differences (C-API bugs): {len(ffi_core_bugs)} / {considered} compared")
for s, why in ffi_core_bugs:
    print("    BUG", s, "--", why)
print(f"(c) FFI-vs-oracle byte-exact: {ffi_oracle_pass} / {len(scenes)}")
for s, why in ffi_oracle_fail:
    print("    ORACLE-FAIL", s, "--", why)
if no_output:
    print(f"\nscenes with no FFI output ({len(no_output)}):")
    for s, why in no_output:
        print("    ", s, "--", why)
# The gate fails on any FFI-vs-core difference (a C-API bug) or a scene that produced no FFI output
# (a crash / unimplemented path through the C API that the core renders).
sys.exit(1 if ffi_core_bugs or no_output else 0)
