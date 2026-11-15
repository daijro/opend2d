#!/usr/bin/env python3
r"""Capture the WIC differential oracle on the Windows reference machine (system WIC codecs).

Mirrors the d2d capture.py: push wic-dump.exe + every scene's input .bin to this run's own working
directory, render each scene's op with the *system* codecs (the renderer runs from a directory with no
drop-in DLL beside it, so the operating system's own libraries load), pull the .f0.bin + stdout JSON
back, and PIN the exact scene beside each capture (wic-<seed>.scene.json) so the diff compares the pinned
input, never a regenerated one. Chunked with a per-scene timeout + kill (a fuzz input can hang the system
codecs too).

  tools/fuzz/wic/wic_cap.py <corpus_dir> <native_out_dir>

Env (see ../vmcap.py for the shared reference-machine variables): D2D_ROOT (repo root, default detected),
D2D_VM_URL, D2D_VM_ROOT (default C:\d2d-rs), VM_CAP / BB_THREAD_ID (per-run working directory name).
Serialize concurrent captures against one reference machine with your own lock, e.g.
``flock "$D2D_VM_LOCK" tools/fuzz/wic/wic_cap.py ...``.
"""
import json, os, pathlib, shutil, sys, tarfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wic_ab import scene_args, normalize  # noqa: E402

ROOT = pathlib.Path(os.environ.get("D2D_ROOT", HERE.parents[2]))  # tools/fuzz/wic -> repo root
sys.path.insert(0, str(ROOT / "harness"))
import vm  # noqa: E402  (harness/vm.py)

EXE = ROOT / "harness/target/x86_64-pc-windows-gnu/release/wic-dump.exe"
VM_ROOT = os.environ.get("D2D_VM_ROOT", r"C:\d2d-rs")
AGENT = os.environ.get("VM_CAP") or "cap-" + os.environ.get("BB_THREAD_ID", "shared")[-10:]
REMOTE = f"{AGENT}/wicfuzz"            # under <VM_ROOT>\<AGENT>\wicfuzz
RBASE = rf"{VM_ROOT}\{AGENT}\wicfuzz"


def main():
    corpus = pathlib.Path(sys.argv[1])
    outdir = pathlib.Path(sys.argv[2]); outdir.mkdir(parents=True, exist_ok=True)
    scenes = sorted(corpus.glob("wic-*.json"))
    print(f"pushing exe + {len(scenes)} scenes (+ input bins) to {RBASE}", flush=True)
    vm.run(rf"Remove-Item -Recurse -Force {RBASE} -ErrorAction SilentlyContinue; "
           rf"New-Item -ItemType Directory -Force {RBASE}\in, {RBASE}\out | Out-Null; echo ok")
    vm.push(str(EXE), f"{REMOTE}/wic-dump.exe")
    for sp in scenes:
        sc = json.loads(sp.read_text())
        vm.push(str(corpus / sc["input"]), f"{REMOTE}/in/{sc['input']}")

    CH = 40
    for i in range(0, len(scenes), CH):
        cmds = []
        for sp in scenes[i:i + CH]:
            sc = json.loads(sp.read_text())
            args = scene_args(sc, rf"in\{sc['input']}", rf"out\{sp.stem}")
            # Combined stdout+stderr per scene → out\<stem>.out (normalised locally via normalize,
            # symmetric with ours). A hang/crash leaves no .f0.bin, which diff treats as native-void.
            argl = ",".join("'" + a.replace("'", "''") + "'" for a in args)
            cmds.append(
                rf"$o=& .\wic-dump.exe @({argl}) 2>&1 | Out-String; "
                rf"Set-Content -NoNewline -Path ('out\{sp.stem}.out') -Value $o")
        ps = rf"cd {RBASE}; " + "; ".join(cmds) + r"; echo chunkdone"
        vm.run(ps, cwd=RBASE)
        print(f"  chunk {i // CH} done", flush=True)

    # Pull every output file back.
    vm.run(rf"cd {RBASE}; tar -cf ..\wicout.tar -C out .; echo ok")
    tarp = outdir / "wicout.tar"
    vm.pull(f"{AGENT}/wicout.tar", str(tarp))
    with tarfile.open(tarp) as t:
        t.extractall(outdir)
    tarp.unlink()
    # Normalise each native run into the comparable record (symmetric with ours), then pin the exact
    # scene + input beside it (local copy, no VM round-trip).
    for sp in scenes:
        sc = json.loads(sp.read_text())
        outp = outdir / f"{sp.stem}.out"
        text = outp.read_text(errors="replace") if outp.exists() else ""
        # Outputs were extracted into outdir; normalise reads them there. Persist the comparable blob
        # as <stem>.f0.bin (encode wrote <stem>.bin) so the diff finds it under one name for every op.
        meta, blob = normalize(sc, text, outdir, sp.stem)
        (outdir / f"{sp.stem}.json").write_text(json.dumps(meta))
        if blob is not None:
            (outdir / f"{sp.stem}.f0.bin").write_bytes(blob)
        shutil.copyfile(sp, outdir / f"{sp.stem}.scene.json")
        shutil.copyfile(corpus / sc["input"], outdir / sc["input"])
    nbin = len(list(outdir.glob("*.f0.bin")))
    print(f"WIC-CAPTURE-DONE: {len(scenes)} scenes, {nbin} native bins, pinned {len(scenes)} scenes -> {outdir}",
          flush=True)


if __name__ == "__main__":
    main()
