#!/usr/bin/env python3
r"""Shared helper for capturing native oracles on a Windows reference machine.

The differential fuzzer renders every scene twice: once with the drop-in OpenD2D build and once with
the Windows system Direct2D. This module drives the native side. It talks to the reference machine
through ``harness/vm.py`` (a tiny JSON-RPC client; set ``D2D_VM_URL`` to point it at your machine), then
on that machine it: wipes a clean working directory, uploads the renderer and the scenes, renders each
scene through the *system* renderer (the renderer .exe runs from a directory with no drop-in DLL beside
it, so the operating system's own libraries load), and pulls the ``.bgra``/``.json`` results back.

Two knobs matter:

* A **per-run remote directory** keyed on the caller. Several runs may share one reference machine, and a
  single fixed directory would let them overwrite each other's scenes mid-render. The directory is
  ``<VM_ROOT>\\<agent>\\fuzz`` where ``agent`` is ``$VM_CAP`` or ``cap-$BB_THREAD_ID`` or ``cap-shared``.
* A **cold-font** option. Text rendering caches per font; to capture the first-time (cold) rasterization
  that the conformance target uses, we copy the measured system faces, append a few zero bytes (a new
  cache key, byte-identical outline), and expose them so every text op rasterizes fresh. The renderer
  builds a custom font collection from exactly those faces, so they are also the only faces its font
  matcher can pick; the capture records them in ``cold-faces.txt`` and the differ replays that set.

Environment:
  D2D_VM_URL   reference-machine endpoint (default lives in harness/vm.py)
  D2D_VM_ROOT  Windows working root on that machine (default ``C:\d2d-rs``)
  VM_CAP       explicit per-run directory name (overrides the BB_THREAD_ID-derived default)
  BB_THREAD_ID used to derive the per-run directory name when VM_CAP is unset
  D2D_ROOT     repository root (default: detected from this file's location)
"""
import os
import pathlib
import subprocess
import sys
import tarfile
import time

ROOT = pathlib.Path(os.environ.get("D2D_ROOT", pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "harness"))
import vm  # noqa: E402  (harness/vm.py)

VM_ROOT = os.environ.get("D2D_VM_ROOT", r"C:\d2d-rs")
AGENT = os.environ.get("VM_CAP") or "cap-" + os.environ.get("BB_THREAD_ID", "shared")[-10:]
REMOTE = f"{AGENT}/fuzz"          # path under VM_ROOT, as harness/vm.py expects it
RBASE = rf"{VM_ROOT}\{AGENT}\fuzz"  # the same path spelled for PowerShell

# The measured system faces, cold-bumped for first-time rasterization. Rounds up to r22 used the first
# eight; the italics let native's matcher pick a real italic face, as the system collection would.
FONTS = ["arial", "arialbd", "consola", "consolab", "segoeui", "segoeuib", "times", "timesbd",
         "arialbi", "ariali", "timesbi", "timesi", "consolai", "consolaz", "segoeuii", "segoeuiz"]
# Round 25 (text): every single-file system face the text generator's families resolve to, so most runs
# draw from the cold collection; strings in scripts these faces lack still go through system fallback.
FONTS25 = ["arial", "arialbd", "arialbi", "ariali", "ariblk", "bahnschrift", "calibri", "comic", "consola", "consolab",
           "consolai", "consolaz", "cour", "georgia", "georgiai", "impact", "LeelawUI", "lucon", "malgun", "pala",
           "segoeui", "segoeuib", "segoeuii", "segoeuiz", "seguiemj", "tahoma", "times", "timesbd", "timesbi", "timesi",
           "trebuc", "verdana"]
FONT_SRC = os.environ.get("D2D_VM_FONTS", r"C:\Windows\Fonts")
CHUNK = 50  # scenes per remote render call, so no single call runs too long


def run_retry(script, timeout=1200, tries=5, pause=150, cwd=RBASE):
    """Run a PowerShell script on the reference machine, retrying hangs/failures.

    The RPC bridge occasionally wedges; a host watchdog restarts it within a few minutes, so a timeout
    or error is retried after a pause rather than aborting the whole capture. ``cwd`` defaults to the
    per-run directory, but the first call — which creates that directory — must run from a path that
    already exists (a fresh agent's RBASE does not), or PowerShell fails to set its working directory.
    """
    for attempt in range(tries):
        try:
            res = vm.run(script, cwd=cwd, timeout_ms=int(timeout * 1000))
            return (res.get("stdout") or "").strip()
        except Exception as e:  # noqa: BLE001 — any RPC/timeout error is retryable
            err = str(e)[-300:]
        print(f"vm run attempt {attempt + 1}/{tries} failed: {err}; waiting for watchdog...", flush=True)
        if attempt + 1 < tries:
            time.sleep(pause)
    raise SystemExit(f"vm run failed after {tries} tries")


def faces_of(capture_dir):
    """The cold face set a capture directory was rendered with (its ``cold-faces.txt``), else ``FONTS``."""
    try:
        names = (pathlib.Path(capture_dir) / "cold-faces.txt").read_text().split()
    except OSError:
        return FONTS
    return [n[:-4] if n.endswith(".ttf") else n for n in names]


def _cold_fonts(faces):
    """Build the cold-bumped faces on the reference machine; return the env value listing them."""
    srcs = "; ".join(
        f"[IO.File]::WriteAllBytes('{RBASE}\\fonts\\{f}_c.ttf', "
        f"[IO.File]::ReadAllBytes('{FONT_SRC}\\{f}.ttf') + $z)"
        for f in faces
    )
    run_retry(rf"New-Item -ItemType Directory -Force {RBASE}\fonts | Out-Null; "
              rf"$z = New-Object byte[] 64; {srcs}; echo fontsbuilt")
    return ";".join(f"{RBASE}\\fonts\\{f}_c.ttf" for f in faces)


def capture(exe_path, scenes_dir, out_dir, cold=False, faces=None):
    """Render every ``fz-*.json`` in *scenes_dir* through the system renderer and collect results.

    Writes ``fz-<seed>.bgra`` + ``fz-<seed>.json`` (native oracle) into *out_dir*; a cold capture also
    writes ``cold-faces.txt``, the faces (*faces*, default ``FONTS``) its font collection held. A scene
    that hangs or crashes the native renderer leaves no ``.bgra`` — the differ treats that as
    native-unrenderable and skips it, which is the correct outcome for an adversarial input. Returns the
    list of scene names.
    """
    scenes_dir, out_dir = pathlib.Path(scenes_dir), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = sorted(p.name for p in scenes_dir.glob("fz-*.json"))
    print(f"{len(scenes)} scenes from {scenes_dir} -> {out_dir} (cold={cold})", flush=True)

    tar = out_dir / "scenes.tar"
    with tarfile.open(tar, "w") as t:
        for s in scenes:
            t.add(scenes_dir / s, arcname=s)

    # Run the (re)create from VM_ROOT, not RBASE: for a fresh agent RBASE does not exist yet, so a
    # cwd=RBASE call would fail before this script could create it.
    run_retry(rf"Remove-Item -Recurse -Force {RBASE} -ErrorAction SilentlyContinue; "
              rf"New-Item -ItemType Directory -Force {RBASE}\in, {RBASE}\out | Out-Null; echo ok", cwd=VM_ROOT)
    vm.push(str(tar), f"{REMOTE}/scenes.tar")
    vm.push(str(exe_path), f"{REMOTE}/d2d-ref.exe")
    faces = faces or FONTS
    cold_env = _cold_fonts(faces) if cold else None
    if cold:
        (out_dir / "cold-faces.txt").write_text("".join(f"{f}.ttf\n" for f in faces))
    run_retry(rf"cd {RBASE}; tar -xf scenes.tar -C in; (Get-ChildItem in\*.json).Count")

    prefix = f"$env:DWRITE_COLD_FONT='{cold_env}'; " if cold_env else ""
    for i in range(0, len(scenes), CHUNK):
        names = [s[:-5] for s in scenes[i:i + CHUNK]]  # strip .json
        lst = ",".join(f"'{n}'" for n in names)
        # Each render gets a hard timeout + Kill: an adversarial scene can hang the native renderer too,
        # and without a per-scene cap one hang would stall the whole chunk.
        ps = (prefix + rf"cd {RBASE}; @({lst}) | ForEach-Object " +
              r"{ $p = Start-Process -FilePath .\d2d-ref.exe -ArgumentList ('in\'+$_+'.json'),('out\'+$_+'.png') "
              r"-PassThru -NoNewWindow -RedirectStandardOutput $env:TEMP\r.txt -RedirectStandardError $env:TEMP\e.txt; "
              r"if (-not $p.WaitForExit(5000)) { $p.Kill(); $p.WaitForExit() } }; "
              r"(Get-ChildItem out\*.bgra).Count")
        cnt = run_retry(ps)
        print(f"chunk {i // CHUNK}: out bgra count = {cnt.splitlines()[-1] if cnt else '?'}", flush=True)

    run_retry(rf"cd {RBASE}; tar -cf ..\out.tar -C out .; echo done")
    outtar = out_dir / "out.tar"
    vm.pull(f"{AGENT}/out.tar", str(outtar))
    with tarfile.open(outtar) as t:
        t.extractall(out_dir)
    outtar.unlink()
    return scenes
