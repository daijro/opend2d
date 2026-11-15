#!/usr/bin/env python3
"""Present-path 3A test: render each `hwnd-*` scene through the drop-in `d2d1.dll` into every
presenting target — a GDI DC render target, an HWND window render target, a DXGI surface render
target, a swap-chain back buffer (ID2D1DeviceContext + CreateBitmapFromDxgiSurface), and a
GDI-interop GetDC/ReleaseDC round-trip — and compare the read-back pixels to the native WARP oracle.

The oracle (`<scene>.present.bgra`) was captured on the reference VM; its `dc`, `surf` and `gdi`
captures are byte-identical, so one file stands for every target. The `hwnd` and `swap` targets need
an interactive desktop (the VM's session 0 and headless hosts have none), so they are best-effort:
skipped when the window/swap chain cannot be created, but byte-checked when it can.

Invoked by `present-test.sh`, which builds the DLL + harness and stages them beside a Wine prefix.
Run with `python3 -I`. Env: STAGE (dir with present-ref.exe + d2d1.dll), WINEPREFIX."""
import os, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
STAGE = pathlib.Path(os.environ["STAGE"])
SUITE = pathlib.Path(next(p for p in (os.environ.get("D2D_SUITE"), ROOT / "conformance/suite", pathlib.Path(subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True).stdout.strip() or ".").parent / "conformance/suite") if p and pathlib.Path(p, "scenes").is_dir()))
ORACLE = SUITE / "oracle/win11-26100"
SCENES = ["hwnd-01-rects", "hwnd-02-shapes"]
REQUIRED = ["dc", "surf", "read", "gdi", "wic"] # session-0 safe: must render and match
BEST_EFFORT = ["hwnd", "swap"]          # need an interactive desktop: match if produced
env = dict(os.environ, WINEDLLOVERRIDES="d2d1=n;mscoree,mshtml=d", WINEDEBUG="-all")

npass, skips, fails = 0, [], []
for s in SCENES:
    want = (ORACLE / f"{s}.present.bgra").read_bytes()
    for m in REQUIRED + BEST_EFFORT:
        out = STAGE / f"p_{s}_{m}"
        r = subprocess.run(["wine", "present-ref.exe", str(SUITE / "scenes" / f"{s}.json"), str(out), m],
                           cwd=STAGE, env=env, capture_output=True, timeout=120)
        bgra = out.with_suffix(".bgra")
        if not bgra.exists():
            msg = r.stderr.decode("utf-8", "replace")[-120:]
            (skips if m in BEST_EFFORT else fails).append((s, m, "no output: " + msg))
            continue
        got = bgra.read_bytes()
        if got != want:
            i = next(k for k in range(min(len(got), len(want))) if got[k] != want[k])
            fails.append((s, m, f"byte {i}: got {got[i]} want {want[i]}"))
            continue
        npass += 1

print(f"\nPRESENT: {npass} (scene x target) byte-exact vs native WARP through d2d1.dll; {len(skips)} best-effort skipped")
for s, m, why in skips:
    print("  skip", s, m, "--", why)
for s, m, why in fails:
    print("  FAIL", s, m, "--", why)
sys.exit(1 if fails else 0)
