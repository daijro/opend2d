#!/usr/bin/env python3
r"""WIC differential A/B — shared arg-builder + ours-runner (Wine) + diff.

A WIC "scene" (wic-<seed>.json from wic_gen.py) is executed by the unchanged harness `wic-dump.exe`
through the real WIC COM ABI — identically against the Windows system WIC codecs (reference machine,
wic_cap.py) and our drop-in build (Wine, here). Each op writes <out>.f0.bin (tightly-strided output
pixels) + one stdout JSON line {format,width,height,bpp,stride,palette}, or {"error":"0x...."} on
failure. The differential: their output vs ours must match the bin byte-for-byte AND the JSON metadata
AND the error.

Modes:
  wic_ab.py args   <scene.json> <in_rel> <out_prefix>   # print the wic-dump arg list (debug)
  wic_ab.py ours   <corpus_dir> <stage_dir> <out_dir>   # run every scene under Wine → <out_dir>
  wic_ab.py diff   <native_dir> <ours_dir>              # compare; bucket + route; exit 1 on mismatch
"""
import json, os, pathlib, shutil, subprocess, sys

META = ("format", "width", "height", "bpp", "stride", "palette")


def scene_args(sc, in_rel, out_prefix):
    """The wic-dump argv (after the exe) for a scene, with input/output as caller-supplied rel paths.
    Transform ops + decode write <out_prefix>.f0.bin + stdout JSON; encode writes <out_prefix>.bin
    (+ .txt for metadata); metaread writes <out_prefix>.txt."""
    op = sc["op"]
    if op in ("convert", "scale", "fliprotate", "clip"):
        src, w, h = sc["src"], str(sc["w"]), str(sc["h"])
        if op == "convert":
            return ["convert", src, w, h, in_rel, sc["dst"], out_prefix, str(sc["dither"]), str(sc["alpha"])]
        if op == "scale":
            return ["scale", src, w, h, in_rel, str(sc["dw"]), str(sc["dh"]), str(sc["mode"]), out_prefix]
        if op == "fliprotate":
            return ["fliprotate", src, w, h, in_rel, str(sc["opt"]), out_prefix]
        return ["clip", src, w, h, in_rel, str(sc["x"]), str(sc["y"]), str(sc["cw"]), str(sc["ch"]), out_prefix]
    if op == "encode":
        return (["encode", sc["container"], sc["src"], str(sc["w"]), str(sc["h"]), str(sc["stride"]),
                 in_rel, out_prefix, "-"] + sc.get("edits", []))
    if op == "decode":
        return [in_rel, out_prefix]            # wic-dump's default decode (argv len 3)
    if op == "metaread":
        return ["meta", in_rel, out_prefix]
    raise SystemExit(f"unknown op {op!r}")


def normalize(sc, text, srcdir, stem):
    """Normalise one wic-dump run into (meta_dict, blob_bytes|None) — the comparable record, symmetric
    across native + ours. `srcdir` holds the op's output files; `text` is combined stdout+stderr.
      transform / decode: meta = frame-0 JSON line (decode adds nframes); blob = <stem>.f0.bin
      encode:  blob = <stem>.bin (the encoded container); meta = {ok} / error, + .txt (metadata writes)
      metaread: meta = {meta: <stem>.txt tree}; no blob
    Errors surface as a {"error":"0x…"} JSON line (stdout) or the first 0x… HRESULT in the text."""
    import re
    srcdir = pathlib.Path(srcdir)
    op = sc["op"]
    frames = [json.loads(l) for l in text.splitlines()
              if l.strip().startswith("{") and l.strip().endswith("}")]
    errframe = next((f for f in frames if "error" in f), None)
    rd = lambda suffix: srcdir / f"{stem}{suffix}"
    txt = lambda suffix: rd(suffix).read_text(errors="replace") if rd(suffix).exists() else None

    if op in ("convert", "scale", "fliprotate", "clip", "decode"):
        if errframe:
            return errframe, None
        if not frames:  # no JSON at all ⇒ a stderr-propagated (anyhow) failure: pull the HRESULT
            m = re.search(r"0x[0-9A-Fa-f]{8}", text)
            return {"error": m.group(0).lower() if m else "err"}, None
        meta = dict(frames[0])
        if op == "decode":
            meta["nframes"] = len(frames)  # frame count (multi-frame GIF/TIFF); matches the gate normalize
        blob = rd(".f0.bin")
        return meta, (blob.read_bytes() if blob.exists() else None)

    if op == "encode":
        # Success prints nothing + writes a non-empty container; failure prints {"error":…} and leaves
        # the 0-byte stream InitializeFromFilename created — so key off the error line, not the file.
        blob, t = rd(".bin"), txt(".txt")
        ok = errframe is None and blob.exists() and blob.stat().st_size > 0
        m = {"ok": 1} if ok else (dict(errframe) if errframe else {"error": "err"})
        if t is not None:
            m["txt"] = t
        return m, (blob.read_bytes() if ok else None)

    if op == "metaread":  # success writes .txt + prints nothing; failure prints {"error":…}
        return (errframe, None) if errframe else ({"meta": txt(".txt") or ""}, None)

    return (errframe or {"error": "err"}), None


def record_differs(nm, om, nbin: pathlib.Path, obin: pathlib.Path):
    """Compare a native/ours record pair; return the divergence channel ('error-mismatch' /
    'error-code' / 'meta' / 'pixels') or None if they match. Used by both diff and gate."""
    if ("error" in nm) != ("error" in om):
        return "error-mismatch"
    if "error" in nm:
        if nm.get("error") != om.get("error") or nm.get("stage") != om.get("stage"):
            return "error-code"
        return None
    if nm != om:
        return "meta"
    a = nbin.read_bytes() if nbin.exists() else b""
    b = obin.read_bytes() if obin.exists() else b""
    return "pixels" if a != b else None


# ---- ours: run every scene under Wine against our drop-in WIC build -------------------------------
def run_ours(corpus, stage, outdir):
    import concurrent.futures as cf
    corpus, stage, outdir = map(pathlib.Path, (corpus, stage, outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    work_in, work_out = stage / "work" / "in", stage / "work" / "out"
    work_in.mkdir(parents=True, exist_ok=True); work_out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, WINEDLLOVERRIDES="windowscodecs=n", WINEDEBUG="-all")
    scenes = sorted(corpus.glob("wic-*.json"))
    # Wine per-invocation startup dominates, so run the scenes concurrently (each is an independent
    # wic-dump process with a unique input/output name — no shared-state collision). WIC_JOBS workers.
    for sp in scenes:  # stage every input first (serial, cheap)
        shutil.copy(corpus / json.loads(sp.read_text())["input"], work_in / json.loads(sp.read_text())["input"])
    jobs = int(os.environ.get("WIC_JOBS", "12"))

    def one(sp):
        sc = json.loads(sp.read_text()); stem = sp.stem
        args = scene_args(sc, f"work\\in\\{sc['input']}", f"work\\out\\{stem}")
        try:
            r = subprocess.run(["wine", "wic-dump.exe", *args], cwd=stage, env=env,
                               capture_output=True, timeout=120)
            text = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
        except subprocess.TimeoutExpired:
            text = '{"error":"timeout"}'
        meta, pix = normalize(sc, text, work_out, stem)
        (outdir / f"{stem}.json").write_text(json.dumps(meta))
        if pix is not None:
            (outdir / f"{stem}.f0.bin").write_bytes(pix)
        return "error" in meta

    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        res = list(ex.map(one, scenes))
    err = sum(res); ok = len(res) - err
    print(f"ours: {len(scenes)} scenes, {ok} ok, {err} error/void -> {outdir} (jobs={jobs})")


# ---- diff: native vs ours -------------------------------------------------------------------------
def diff(native, ours):
    native, ours = map(pathlib.Path, (native, ours))
    scenes = sorted(p.stem for p in native.glob("wic-*.scene.json"))
    buckets = {}  # reason -> [seeds]
    nmatch = 0
    for st in scenes:
        stem = st[:-6] if st.endswith(".scene") else st  # strip ".scene"
        sc = json.loads((native / f"{stem}.scene.json").read_text())
        nmeta_p, ometa_p = native / f"{stem}.json", ours / f"{stem}.json"
        if not nmeta_p.exists():
            continue  # native skipped/killed this input (unrenderable) — not comparable
        nmeta = json.loads(nmeta_p.read_text())
        if not ometa_p.exists():
            buckets.setdefault(f"{sc['op']}:ours-missing", []).append(stem); continue
        ometa = json.loads(ometa_p.read_text())
        ch = record_differs(nmeta, ometa, native / f"{stem}.f0.bin", ours / f"{stem}.f0.bin")
        if ch:
            buckets.setdefault(f"{sc['op']}:{ch}", []).append(stem)
        else:
            nmatch += 1
    total = sum(len(v) for v in buckets.values()) + nmatch
    print(f"WIC-DIFF: {nmatch}/{total} match; {sum(len(v) for v in buckets.values())} diverge")
    for reason in sorted(buckets):
        v = buckets[reason]
        print(f"  {reason}: {len(v)}  {' '.join(s.split('-')[-1] for s in v[:20])}")
    return 1 if any(buckets.values()) else 0


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "args":
        sc = json.loads(pathlib.Path(sys.argv[2]).read_text())
        print(" ".join(scene_args(sc, sys.argv[3], sys.argv[4])))
    elif mode == "ours":
        run_ours(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == "diff":
        sys.exit(diff(sys.argv[2], sys.argv[3]))
    elif mode == "gate":
        # gate <native_dir> <ours_dir> <baseline.txt> : the A/B merge-gate step. A CAND build's ours_dir
        # is diffed vs the pinned native; any divergent seed NOT in the accepted baseline is NEW ⇒ fail.
        native, ours, basef = sys.argv[2], sys.argv[3], pathlib.Path(sys.argv[4])
        base = set(basef.read_text().split()) if basef.exists() else set()
        # Recompute the cand divergence set with the same record comparison diff uses.
        div = set()
        natp = pathlib.Path(native)
        for sp in sorted(natp.glob("wic-*.scene.json")):
            stem = sp.name[:-11]
            sc = json.loads(sp.read_text())
            np_, op_ = natp / f"{stem}.json", pathlib.Path(ours) / f"{stem}.json"
            if not np_.exists():
                continue
            nm = json.loads(np_.read_text())
            if not op_.exists():
                div.add(stem); continue
            om = json.loads(op_.read_text())
            if record_differs(nm, om, natp / f"{stem}.f0.bin", pathlib.Path(ours) / f"{stem}.f0.bin"):
                div.add(stem)
        new = sorted(div - base)
        fixed = sorted(base - div)
        print(f"WIC-GATE: cand diverge {len(div)}, baseline {len(base)}, new {len(new)}, fixed {len(fixed)}")
        if fixed:
            print("  fixed:", " ".join(s.split('-')[1] for s in fixed[:40]))
        if new:
            print("  NEW:", " ".join(s.split('-')[1] for s in new))
        print("WIC-GATE", "FAIL" if new else "PASS")
        sys.exit(1 if new else 0)
    else:
        raise SystemExit(__doc__)
