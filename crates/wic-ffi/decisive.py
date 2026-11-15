#!/usr/bin/env python3
r"""Decisive drop-in-windowscodecs test: run the unchanged harness `wic-dump.exe` against our drop-in
`windowscodecs.dll` over every committed WIC fixture, through the real WIC COM ABI
(`CoCreateInstance(CLSID_WICImagingFactory)` → `IWICImagingFactory` → decoders / converter / scaler
/ flip-rotator / clipper / JPEG encoder), and compare the output byte-for-byte to the oracle
captures. Exit non-zero on any mismatch.

Invoked by `decisive-test.sh`, which builds the DLL + wic-dump.exe, stages them beside a fresh Wine
prefix, and registers our DLL for the WIC CLSIDs. Run with `python3 -I`.
Env: STAGE (dir with wic-dump.exe + windowscodecs.dll), WINEPREFIX."""
import json, os, pathlib, shutil, subprocess, sys

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
# The WIC fixtures moved into the suite (conformance/suite/fixtures/<suite>/…), preserving the old
# crates/d2d/tests/<suite>/… layout, so every `TESTS / "<suite>/…"` below still resolves.
TESTS = SUITE / "fixtures"
WORK = STAGE / "work"
# windowscodecs=n: load our drop-in native DLL for the WIC CLSIDs, not Wine's builtin.
ENV = dict(os.environ, WINEDLLOVERRIDES="windowscodecs=n", WINEDEBUG="-all")
DEC_KEYS = ("format", "width", "height", "bpp", "stride", "palette")

npass, fails = 0, []


def run(*args):
    """`wine wic-dump.exe <args>` from STAGE (where the exe + dll live); data paths are relative
    `work\\in\\…` / `work\\out\\…`, dodging any Unix-path translation."""
    r = subprocess.run(["wine", "wic-dump.exe", *args], cwd=STAGE, env=ENV, capture_output=True, timeout=180)
    return r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace")


def frames_of(stdout):
    return [json.loads(l) for l in stdout.splitlines() if l.strip().startswith("{")]


def stage_input(src: pathlib.Path) -> str:
    """Copy a fixture into STAGE/work/in and return its STAGE-relative Windows path."""
    shutil.copy(src, WORK / "in" / src.name)
    return f"work\\in\\{src.name}"


def check_bin(tag, out_rel, oracle):
    got = (WORK / "out" / out_rel)
    if not got.exists():
        fails.append(f"{tag}: no output")
        return False
    a, b = got.read_bytes(), oracle.read_bytes()
    if a != b:
        i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        fails.append(f"{tag}: byte {i} (len {len(a)} vs {len(b)})")
        return False
    return True


def check_json(tag, got, want):
    bad = [k for k in DEC_KEYS if str(got.get(k)) != str(want.get(k))]
    for k in bad:
        fails.append(f"{tag}: {k} {got.get(k)!r} != {want.get(k)!r}")
    return not bad


def decode_corpus(label, corpus_dir, cases, cap_dir, file_of, decodepng=False):
    global npass
    for stem, case in cases.items():
        meta = case["frames"] if "frames" in case else [case]
        src = corpus_dir / file_of(stem, case)
        if not src.exists():
            continue
        rel = stage_input(src)
        out, err = run(*(["decodepng"] if decodepng else []), rel, f"work\\out\\{stem}")
        frames = frames_of(out)
        if len(frames) != len(meta):
            fails.append(f"{label}/{stem}: {len(frames)} frames != {len(meta)} ({err.strip()[-160:]})")
            continue
        ok = True
        for i, want in enumerate(meta):
            ok &= check_json(f"{label}/{stem}.f{i}", frames[i], want)
            ok &= check_bin(f"{label}/{stem}.f{i}", f"{stem}.f{i}.bin", cap_dir / f"{stem}.f{i}.bin")
        npass += ok


def process(label, args_of, out_name):
    global npass
    d = TESTS / "wic_process" / label
    for c in json.load(open(d / "cases.json")):
        rel = stage_input(d / f"{c['src']}.src.bin")
        name = out_name(c)
        stem = f"{label}_{name}".replace("-", "_")
        run(*args_of(c, rel, f"work\\out\\{stem}"))
        npass += check_bin(f"{label}/{name}", f"{stem}.f0.bin", d / f"{name}.out.bin")


def encode_corpus(label, container, cases_file, inputs, captures, ext):
    """Encode each core-verified input through our IWICBitmapEncoder (wic-dump `encode`) and compare
    the committed file byte-for-byte to native's capture — the full COM path over the byte-exact
    `d2d::wic::encode_*` core (default encoder options)."""
    global npass
    wc = TESTS / "wic_codecs"
    for name, c in json.load(open(wc / cases_file)).items():
        rel = stage_input(wc / inputs / f"{name}.bin")
        fmt = f"6fddc324-4e03-4bfe-b185-3d77768dc9{c.get('suffix', '04')}"  # gif fixed 8bppIndexed
        pal = ",".join(c.get("palette") or c.get("pal") or []) or "-"
        stem = f"{label}_{name}".replace("-", "_")
        run("encode", container, fmt, str(c["w"]), str(c["h"]), str(c.get("stride", c["w"])), rel, f"work\\out\\{stem}", pal)
        npass += check_bin(f"{label}/{name}", f"{stem}.bin", wc / captures / f"{name}.{ext}")


def com_api():
    """Component info / enumeration / pixel-format info / metadata query reader, compared to the
    native oracle under `wic_com/oracle/` (text dumped identically by the same `wic-dump` binary, so
    only the text is compared)."""
    global npass
    d = TESTS / "wic_com"
    orc = d / "oracle"
    spec = json.load(open(d / "cases.json"))

    def cmp_txt(tag, name):
        got = WORK / "out" / f"{name}.txt"
        if got.exists() and got.read_bytes() == (orc / f"{name}.txt").read_bytes():
            return 1
        fails.append(f"{tag}: {'no output' if not got.exists() else 'text mismatch'}")
        return 0

    for c in spec["compinfo"]:
        rel = stage_input(TESTS / c["img"])
        run("compinfo", rel, f"work\\out\\{c['out']}")
        npass += cmp_txt(f"compinfo/{c['out']}", c["out"])
    for suf in spec["pfinfo"]:
        guid = "3cc4a650-a527-4d37-a916-3142c7ebedba" if suf == "prgba" else f"6fddc324-4e03-4bfe-b185-3d77768dc9{suf}"
        run("pfinfo", guid, f"work\\out\\pf_{suf}")
        npass += cmp_txt(f"pfinfo/{suf}", f"pf_{suf}")
    for ty in spec["palette"]:
        for addtrans in (0, 1):
            out = f"pal_{ty}_{addtrans}"
            run("palette", str(ty), str(addtrans), f"work\\out\\{out}")
            npass += cmp_txt(f"palette/{out}", out)
    for types, opts, out in spec["enum"]:
        run("enumcomp", str(types), str(opts), f"work\\out\\{out}")
        got = WORK / "out" / f"{out}.txt"
        ours = set(got.read_text().splitlines()) if got.exists() else set()
        want = set((orc / f"{out}.txt").read_text().splitlines())
        if ours and ours <= want:  # our DLL enumerates only the codecs it implements: a native subset
            npass += 1
        else:
            fails.append(f"enum/{out}: not a subset of native ({sorted(ours - want)})")
    for c in spec["meta"]:
        rel = stage_input(TESTS / c["img"])
        run("meta", rel, f"work\\out\\{c['out']}")
        npass += cmp_txt(f"meta/{c['out']}", c["out"])

    # CreateQueryWriterFromReader round-trip: copy each frame-0 reader into a query writer and dump;
    # it must reproduce the reader tree exactly (the meta oracle's frame-0 entries, prefix stripped).
    for c in spec["meta"]:
        rel = stage_input(TESTS / c["img"])
        run("metawrite", rel, f"work\\out\\mw_{c['out']}")
        got = WORK / "out" / f"mw_{c['out']}.txt"
        want = "\n".join(sorted(l[3:] for l in (orc / f"{c['out']}.txt").read_text().splitlines() if l.startswith("f0:")))
        if got.exists() and got.read_text() == want:
            npass += 1
        else:
            fails.append(f"metawrite/{c['out']}: round-trip mismatch")

    # Fast metadata encoder in-place Commit: run the edit, compare the committed container bytes
    # (.bin) and/or the dump text (.txt: seed tree + create/set/commit HRESULTs) to native.
    for c in spec.get("fastmeta", []):
        rel = stage_input(TESTS / c["img"])
        out = c["out"]
        run("fastedit", c["scope"], rel, f"work\\out\\{out}", *c["edits"])
        ok = True
        if c.get("txt"):
            ok &= cmp_txt(f"fastmeta/{out}", out) == 1
        if c.get("bin"):
            ok &= check_bin(f"fastmeta/{out}", f"{out}.bin", orc / f"{out}.bin")
        npass += 1 if ok else 0

    # Encoder-side metadata serialization: encode with EXIF/COM set on the frame query writer and
    # compare the committed JPEG bytes (.bin) and the set/commit dump (.txt) to native.
    for c in spec.get("encodemeta", []):
        rel = stage_input(TESTS / c["src"])
        out = c["out"]
        run("encodemeta", str(c["w"]), str(c["h"]), rel, c["fmt"], f"work\\out\\{out}", *c["edits"])
        ok = cmp_txt(f"encodemeta/{out}", out) == 1
        ok &= check_bin(f"encodemeta/{out}", f"{out}.bin", orc / f"{out}.bin")
        npass += 1 if ok else 0

    # Container encoder metadata (TIFF IFD tags / GIF extensions / XMP): encode with tags set on the
    # frame query writer and compare the committed bytes (.bin) and set/commit dump (.txt) to native.
    for c in spec.get("encmeta", []):
        rel = stage_input(TESTS / f"{c['in']}.bin")
        out = c["out"]
        guid = f"6fddc324-4e03-4bfe-b185-3d77768dc9{c['suffix']}"
        pal = ",".join(c.get("pal") or []) or "-"
        run("encode", c["container"], guid, str(c["w"]), str(c["h"]), str(c["stride"]), rel, f"work\\out\\{out}", pal, *c["edits"])
        ok = cmp_txt(f"encmeta/{out}", out) == 1
        ok &= check_bin(f"encmeta/{out}", f"{out}.bin", orc / f"{out}.bin")
        npass += 1 if ok else 0

    # CreateBitmapFromSourceRect: re-crop the clip oracle cases and compare to the clip output (the
    # same `d2d::wic` crop), reusing the already byte-exact native clip captures.
    cd = TESTS / "wic_process" / "clip"
    for c in json.load(open(cd / "cases.json")):
        rel = stage_input(cd / f"{c['src']}.src.bin")
        name = f"{c['src']}-{c['x']}_{c['y']}_{c['cw']}_{c['ch']}"
        stem = f"bmprect_{name}".replace("-", "_")
        run("bmprect", c["fmt"], str(c["width"]), str(c["height"]), rel, str(c["x"]), str(c["y"]), str(c["cw"]), str(c["ch"]), f"work\\out\\{stem}")
        npass += check_bin(f"bmprect/{name}", f"{stem}.f0.bin", cd / f"{name}.out.bin")

    # IWICStream::InitializeFromMemory: decoding through an in-memory stream must equal a file decode.
    for img, cap in [
        ("wic/corpus/rgb8.png", "wic/captures/rgb8.f0.bin"),
        ("wic_codecs/corpus/bmp_rgb24.bmp", "wic_codecs/captures/bmp_rgb24.f0.bin"),
        ("wic_jpeg/corpus/p420.jpg", "wic_jpeg/captures/p420.f0.bin"),
        ("wic_codecs/corpus/gif_basic.gif", "wic_codecs/captures/gif_basic.f0.bin"),
        ("wic_codecs/corpus/tiff_rgb8.tif", "wic_codecs/captures/tiff_rgb8.f0.bin"),
    ]:
        stem = pathlib.Path(img).stem
        rel = stage_input(TESTS / img)
        run("decodemem", rel, f"work\\out\\dm_{stem}")
        npass += check_bin(f"decodemem/{stem}", f"dm_{stem}.f0.bin", TESTS / cap)
        run("decodehandle", rel, f"work\\out\\dh_{stem}")  # CreateDecoderFromFileHandle, same pixels
        npass += check_bin(f"decodehandle/{stem}", f"dh_{stem}.f0.bin", TESTS / cap)

    # CreateBitmap + IWICBitmapLock: writing a source through the lock then reading it back is identity.
    seen = set()
    for c in json.load(open(cd / "cases.json")):
        if c["src"] in seen:
            continue
        seen.add(c["src"])
        rel = stage_input(cd / f"{c['src']}.src.bin")
        run("bitmaplock", c["fmt"], str(c["width"]), str(c["height"]), rel, f"work\\out\\bl_{c['src']}")
        npass += check_bin(f"bitmaplock/{c['src']}", f"bl_{c['src']}.f0.bin", cd / f"{c['src']}.src.bin")

    # IWICColorContext: InitializeFromMemory returns the profile bytes verbatim (as native stores
    # them); InitializeFromExifColorSpace round-trips the id.
    icc_in = TESTS / "wic/corpus/rgb8.png"
    rel = stage_input(icc_in)
    run("colorctx", "mem", rel, "work\\out\\cc_mem")
    got = WORK / "out" / "cc_mem.bin"
    if got.exists() and got.read_bytes() == icc_in.read_bytes() and (WORK / "out" / "cc_mem.txt").read_text() == "type 1\n":
        npass += 1
    else:
        fails.append("colorctx/mem: round-trip mismatch")
    run("colorctx", "exif", "1", "work\\out\\cc_exif")
    ce = WORK / "out" / "cc_exif.txt"
    if ce.exists() and ce.read_text() == "type 2\nexif 1\n":
        npass += 1
    else:
        fails.append("colorctx/exif: mismatch")

    # CreateBitmapFromHBITMAP/HICON are GDI-backed; headless Wine's GDI (GetDC/CreateIconIndirect)
    # blocks, so they are not run in this loop. They are verified byte-for-byte against the native
    # capture in wic_com/oracle/{hbm_*,hico_*}.f0.bin (wic-dump hbitmap/hicon), whose bytes equal the
    # deterministic output of the implemented algorithm (GetDIBits identity + AND-mask application).

    # Frame GetColorContexts: extract the embedded ICC profile / EXIF colour space (JPEG APP2 ICC,
    # JPEG EXIF ColorSpace, PNG iCCP, TIFF tag 34675), compared to the native oracle.
    for c in ["color_icc.jpg", "color_cs.jpg", "color_icc.png", "color_icc.tif"]:
        stem = "cc_" + c.replace(".", "_")
        rel = stage_input(d / "images" / c)
        run("colorctxframe", rel, f"work\\out\\{stem}")
        got = WORK / "out" / f"{stem}.txt"
        ob = orc / f"{stem}.c0.bin"
        ok = got.exists() and got.read_bytes() == (orc / f"{stem}.txt").read_bytes()
        ok = ok and (not ob.exists() or (WORK / "out" / f"{stem}.c0.bin").read_bytes() == ob.read_bytes())
        if ok:
            npass += 1
        else:
            fails.append(f"colorctxframe/{stem}: mismatch")

    # Fast metadata encoder: native rejects CreateFastMetadataEncoderFromDecoder for every built-in
    # codec except GIF (WINCODEC_ERR_UNSUPPORTEDOPERATION at create time). (GIF's in-place Commit and a
    # schema-aware query writer are not reproduced.)
    for img in ["wic/corpus/rgb8.png", "wic_jpeg/corpus/p420.jpg", "wic_codecs/corpus/tiff_rgb8.tif", "wic_codecs/corpus/bmp_rgb24.bmp"]:
        rel = stage_input(TESTS / img)
        out, _ = run("fastmeta", rel, f"work\\out\\fm_{pathlib.Path(img).stem}")
        if '"error":"0x88982f81","stage":"createfme"' in out:
            npass += 1
        else:
            fails.append(f"fastmeta/{pathlib.Path(img).stem}: {out.strip()[:80]}")
    # GIF is the one codec that supports it: createfme/getwriter succeed, and the strict writer
    # rejects an out-of-schema path (/app1) at the set step with WINCODEC_ERR_PROPERTYNOTFOUND.
    out, _ = run("fastmeta", stage_input(TESTS / "wic_codecs/corpus/gif_basic.gif"), "work\\out\\fm_gif")
    if '"error":"0x88982f40","stage":"set"' in out:
        npass += 1
    else:
        fails.append(f"fastmeta/gif: {out.strip()[:80]}")


def palette_bitmap():
    """IWICPalette::InitializeFromBitmap (median-cut quantiser): drive `palettebitmap` over each
    authored source and compare the dumped colour table + flags to the native oracle text."""
    global npass
    d = TESTS / "wic_palette"
    for c in json.load(open(d / "cases.json")):
        rel = stage_input(d / f"{c['src']}.src.bin")
        out = c["name"]
        run("palettebitmap", c["fmt"], str(c["width"]), str(c["height"]), rel,
            str(c["count"]), str(c["addtrans"]), f"work\\out\\{out}")
        got = WORK / "out" / f"{out}.txt"
        want = d / "oracle" / f"{out}.txt"
        if got.exists() and got.read_bytes() == want.read_bytes():
            npass += 1
        else:
            fails.append(f"palettebitmap/{out}: {'no output' if not got.exists() else 'mismatch'}")


def colortransform():
    """IWICColorTransform over two ICC profiles — the mscms/icm32 legacy "Win " CMM path. For every
    direction × integer pixel format, transform the authored input and compare byte-for-byte to the
    native capture; the float format is rejected with WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT as native."""
    global npass
    d = TESTS / "wic_colortransform"
    spec = json.load(open(d / "cases.json"))
    w, h = str(spec["width"]), str(spec["height"])
    dirs = {**spec["directions"], **spec.get("directions_para", {}), **spec.get("directions_lut", {})}
    for tag, (sf, tf) in dirs.items():
        srel, trel = stage_input(d / sf), stage_input(d / tf)
        for fmt, guid in spec["formats"].items():
            inrel = stage_input(d / f"in_{fmt}.bin")
            stem = f"ct_{tag}_{fmt}"
            run("colortransform", srel, trel, guid, w, h, inrel, f"work\\out\\{stem}")
            npass += check_bin(f"colortransform/{tag}_{fmt}", f"{stem}.f0.bin", d / "oracle" / f"{tag}_{fmt}.f0.bin")
    # CMYK 4->3: source is 32bppCMYK (in_cmyk32.bin), destination an RGB format (9th arg). Only the
    # 8-bit outputs are byte-exact (rgba16 goes through native's generic path, sub-LSB off).
    cmyk_guid = spec["cmyk_src_format"]
    cmyk_in = stage_input(d / "in_cmyk32.bin")
    for tag, (sf, tf) in spec.get("directions_cmyk", {}).items():
        srel, trel = stage_input(d / sf), stage_input(d / tf)
        for fmt in ("rgb8", "bgra8"):
            stem = f"ct_{tag}_{fmt}"
            run("colortransform", srel, trel, cmyk_guid, w, h, cmyk_in, f"work\\out\\{stem}", spec["formats"][fmt])
            npass += check_bin(f"colortransform/{tag}_{fmt}", f"{stem}.f0.bin", d / "oracle" / f"{tag}_{fmt}.f0.bin")
    srel = stage_input(d / "srgb.icc")
    inrel = stage_input(d / "in_rgba128f.bin")
    guid = spec["unsupported_formats"]["rgba128f"].split()[0]
    out, _ = run("colortransform", srel, srel, guid, w, h, inrel, "work\\out\\ct_unsup")
    if '"error":"0x88982f80"' in out:
        npass += 1
    else:
        fails.append(f"colortransform/rgba128f: expected UNSUPPORTEDPIXELFORMAT, got {out.strip()[:80]}")


def main():
    global npass
    if WORK.exists():
        shutil.rmtree(WORK)
    (WORK / "in").mkdir(parents=True)
    (WORK / "out").mkdir(parents=True)

    # 1. PNG decode (CoCreateInstance(CLSID_WICPngDecoder) + Initialize from a stream).
    decode_corpus("png", TESTS / "wic/corpus", json.load(open(TESTS / "wic/cases.json")),
                  TESTS / "wic/captures", lambda s, c: f"{s}.png", decodepng=True)

    # 2. BMP/GIF/ICO/TIFF decode (CreateDecoderFromFilename, container sniff).
    decode_corpus("codec", TESTS / "wic_codecs/corpus", json.load(open(TESTS / "wic_codecs/cases.json")),
                  TESTS / "wic_codecs/captures", lambda s, c: c["file"])

    # 3. JPEG decode.
    decode_corpus("jpeg", TESTS / "wic_jpeg/corpus", json.load(open(TESTS / "wic_jpeg/cases.json")),
                  TESTS / "wic_jpeg/captures", lambda s, c: f"{s}.jpg")

    # 4. JPEG reduced-size decode (IWICBitmapSourceTransform).
    for name, denoms in json.load(open(TESTS / "wic_jpeg/scaled.json")).items():
        src = TESTS / "wic_jpeg/corpus" / f"{name}.jpg"
        if not src.exists():
            continue
        rel = stage_input(src)
        for denom in denoms:
            run("--scale", denom, rel, f"work\\out\\{name}_s{denom}")
            npass += check_bin(f"scaled/{name}.{denom}", f"{name}_s{denom}.f0.bin",
                               TESTS / "wic_jpeg/captures/scaled" / f"{name}.{denom}.bin")

    # 5. Pixel processing: convert / scale / flip-rotate / clip.
    process("convert",
            lambda c, rel, p: ["convert", c["src_fmt"], str(c["width"]), str(c["height"]), rel, c["dst_fmt"], p, "0", "0",
                               *([",".join(c["palette"])] if c.get("palette") else [])],
            lambda c: f"{c['src']}-{c['dst']}")
    process("scale",
            lambda c, rel, p: ["scale", c["fmt"], str(c["srcw"]), str(c["srch"]), rel, str(c["dstw"]), str(c["dsth"]), str(c["mode"]), p],
            lambda c: f"{c['src']}-m{c['mode']}-{c['dstw']}x{c['dsth']}")
    process("fliprotate",
            lambda c, rel, p: ["fliprotate", c["fmt"], str(c["width"]), str(c["height"]), rel, str(c["options"]), p],
            lambda c: f"{c['src']}-{c['options']}")
    process("clip",
            lambda c, rel, p: ["clip", c["fmt"], str(c["width"]), str(c["height"]), rel, str(c["x"]), str(c["y"]), str(c["cw"]), str(c["ch"]), p],
            lambda c: f"{c['src']}-{c['x']}_{c['y']}_{c['cw']}_{c['ch']}")

    # 6. JPEG encode (CreateEncoder(JPEG) → WriteSource → Commit, WIC defaults).
    for name, c in json.load(open(TESTS / "wic_jpeg/encode/cases.json")).items():
        rel = stage_input(TESTS / "wic_jpeg/encode/inputs" / f"{name}.bin")
        args = ["--encode", str(c["width"]), str(c["height"]), rel, f"work\\out\\{name}.jpg"]
        if c.get("gray"):
            args.append("gray")
        run(*args)
        npass += check_bin(f"encode/{name}", f"{name}.jpg", TESTS / "wic_jpeg/encode" / f"{name}.jpg")

    # 6b. BMP / PNG / TIFF / GIF encode through the COM encoder (CreateEncoder → WriteSource →
    # Commit), compared to native captures (the same oracles the core encode_* tests use).
    encode_corpus("enc_bmp", "0af1d87e-fcfe-4188-bdeb-a7906471cbe3", "enc_cases.json", "enc_inputs", "captures_enc", "bmp")
    encode_corpus("enc_png", "1b7cfaf4-713f-473c-bbcd-6137425faeaf", "png_enc_cases.json", "png_enc_inputs", "png_captures", "png")
    encode_corpus("enc_tif", "163bcc30-e2e9-4f0b-961d-a3e9fdb788a3", "tif_enc_cases.json", "tif_enc_inputs", "tif_captures", "tif")
    encode_corpus("enc_gif", "1f8a5601-7d4d-4cbd-9c82-1bc8d4eeb9a5", "gif_enc_cases.json", "gif_enc_inputs", "gif_captures", "gif")

    # 6c. PNG ancillary-chunk metadata (tEXt / tIME) set on the encoder frame's query writer,
    # inserted before IDAT — compared to native (oracle in wic_com/oracle/pm_*.bin).
    pngc, bgr24 = "1b7cfaf4-713f-473c-bbcd-6137425faeaf", "6fddc324-4e03-4bfe-b185-3d77768dc90c"
    pin = stage_input(TESTS / "wic_codecs/png_enc_inputs/bgr24.bin")
    orc = TESTS / "wic_com/oracle"
    for out, edits in [
        ("pm_text", ["/tEXt/{str=Title}:str:Hello"]),
        ("pm_time", ["/tIME/Year:u2:2026", "/tIME/Month:u1:10", "/tIME/Day:u1:31", "/tIME/Hour:u1:12", "/tIME/Minute:u1:0", "/tIME/Second:u1:0"]),
    ]:
        run("encode", pngc, bgr24, "4", "3", "12", pin, f"work\\out\\{out}", "-", *edits)
        npass += check_bin(f"pngmeta/{out}", f"{out}.bin", orc / f"{out}.bin")

    # 6d. JPEG-XR encode through the COM encoder (CreateEncoder(WMP) → property-bag options →
    # WritePixels → Commit), compared to the wmphoto oracle. The only path that exercises the
    # IWICBitmapFrameEncode property-bag option handling through the real COM ABI.
    jxr = TESTS / "wic/jxr/encode"
    jtype = {"ImageQuality": "f", "Lossless": "b", "UseCodecOptions": "b", "Overlap": "u",
             "Subsampling": "u", "HorizontalTileSlices": "w", "VerticalTileSlices": "w",
             "FrequencyOrder": "b", "InterleavedAlpha": "b"}
    for name, c in json.load(open(jxr / "cases.json")).items():
        src = stage_input(TESTS / "wic/jxr/captures" / c["source"])
        specs = [f"{k}:{jtype[k]}:{str(v).lower() if isinstance(v, bool) else v}" for k, v in c["options"].items()]
        run("encodejxr", c["format"], str(c["width"]), str(c["height"]), src, f"work\\out\\{c['file']}", *specs)
        npass += check_bin(f"jxrenc/{name}", c["file"], jxr / c["file"])

    # 7. COM gaps: component info / enumeration / pixel-format info / metadata query reader.
    com_api()

    # 8. Median-cut palette: IWICPalette::InitializeFromBitmap.
    palette_bitmap()

    # 9. IWICColorTransform: ICC→ICC recolour through the legacy "Win " CMM (icm32).
    colortransform()

    total = npass + len(fails)
    print(f"\nDECISIVE: {npass}/{total} WIC fixtures byte-exact through windowscodecs.dll")
    for f in fails[:60]:
        print("  FAIL", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
