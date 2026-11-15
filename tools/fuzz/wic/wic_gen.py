#!/usr/bin/env python3
# WIC differential fuzz generator. Emits a corpus of WIC "scenes" that `wic-dump` runs identically
# against the Windows system WIC codecs (reference machine) and ours (Wine) — the byte-exact driver.
#
# Round 1-2 ops (pure codec math, no container parse): input = raw source pixels.
#   convert  <srcGuid> w h <bin> <dstGuid> <out> [dither] [alpha]   # IWICFormatConverter
#   scale    <srcGuid> w h <bin> dw dh <mode> <out>                 # IWICBitmapScaler
#   fliprotate <srcGuid> w h <bin> <opt> <out>                      # IWICBitmapFlipRotator
#   clip     <srcGuid> w h <bin> x y cw ch <out>                    # IWICBitmapClipper
# Round 3 ops (container codecs): encode raw pixels OR decode/read a container.
#   encode   raw pixels -> IWICBitmapEncoder(container) -> bytes     (+ optional metadata writes)
#   decode   container file -> IWICBitmapDecoder -> frame-0 pixels + per-frame meta
#   metaread container file (with embedded metadata) -> IWICMetadataQueryReader tree
# For decode/metaread the input .bin IS a valid container, built here with Pillow (codec-agnostic:
# the differential is their-decode vs ours-decode of the SAME bytes, so the source needn't be the
# system codec's own encoder). encode is self-contained (input = raw pixels, like the round-1 ops).
#
# Each seed writes <outdir>/wic-<seed>.json (the pinned scene) + wic-<seed>.bin (its input). The
# pinned .scene.json beside each capture is the source of truth — gen drift can never desync oracle.
#
#   wic_gen.py <outdir> <count> [start-seed] [ops]
#     ops = comma list to restrict the op mix (default all 7); round 3 e.g. "encode,decode,metaread".
import io, json, os, sys, pathlib

# Non-indexed WIC source pixel formats (GUID hex-32, bits-per-pixel) — sized to fill the source buffer.
FMT = {
    "BGRA32":  ("6fddc3244e034bfeb1853d77768dc90f", 32),
    "PBGRA32": ("6fddc3244e034bfeb1853d77768dc910", 32),
    "BGR32":   ("6fddc3244e034bfeb1853d77768dc90e", 32),
    "BGR24":   ("6fddc3244e034bfeb1853d77768dc90c", 24),
    "RGB24":   ("6fddc3244e034bfeb1853d77768dc90d", 24),
    "Gray8":   ("6fddc3244e034bfeb1853d77768dc908", 8),
    "Gray16":  ("6fddc3244e034bfeb1853d77768dc90b", 16),
    "BGR565":  ("6fddc3244e034bfeb1853d77768dc90a", 16),
    "BGR555":  ("6fddc3244e034bfeb1853d77768dc909", 16),
    "RGBA32":  ("f5c7ad2d6a8d43dda7a8a29935261ae9", 32),
    "PRGBA32": ("3cc4a650a5274d37a9163142c7ebedba", 32),
    "RGBA64":  ("6fddc3244e034bfeb1853d77768dc916", 64),
}
NAMES = list(FMT)

# WIC container-format GUIDs (hex-32, standard registry string order, dashes stripped).
CONTAINERS = {
    "Png":  "1b7cfaf4713f473cbbcd6137425faeaf",
    "Bmp":  "0af1d87efcfe4188bdeba7906471cbe3",
    "Gif":  "1f8a56017d4d4cbd9c821bc8d4eeb9a5",
    "Ico":  "a3a860c4338f4c17919afba4b5628f21",
    "Jpeg": "19e4a5aa56624fc5a0c01758028e1057",
    "Tiff": "163bcc30e2e94f0b961da3e9fdb788a3",
    "Wmp":  "57a37caa367a4540916bf183c5093a4b",  # JPEG-XR
    "Dds":  "9967cb952e854ac88ca283d7ccd425c9",
}
CNAMES = list(CONTAINERS)

# Source formats each container's encoder is likely to accept — bias encode toward valid output (the
# rest of the time a random format exercises the reject path, also a differential).
CONTAINER_SRC = {
    "Png":  ["BGRA32", "BGR24", "Gray8", "Gray16", "RGBA64", "RGBA32"],
    "Bmp":  ["BGRA32", "BGR24", "BGR32", "Gray8", "BGR555", "BGR565"],
    "Gif":  ["Gray8", "BGR24"],
    "Ico":  ["BGRA32", "BGR24"],
    "Jpeg": ["BGR24", "Gray8"],
    "Tiff": ["BGRA32", "BGR24", "Gray8", "Gray16", "RGBA64", "BGR565"],
    "Wmp":  ["BGRA32", "BGR24", "Gray8", "RGBA64"],
    "Dds":  ["BGRA32"],
}

# Candidate encoder-side metadata writes (path:vt:value) per container — exercise the query writer +
# Commit serialization. Need not all be valid: an invalid path yields a SetMetadataByName HRESULT that
# must match, so a bad path is still a differential. make_pv vt codes: u1/u2/u4/u8/bool/str/wstr.
CONTAINER_EDITS = {
    "Tiff": ["/ifd/{ushort=270}:wstr:WICFuzz", "/ifd/{ushort=305}:wstr:opend2d",
             "/ifd/{ushort=274}:u2:1"],
    "Gif":  ["/grctlext/Delay:u2:50", "/grctlext/TransparencyFlag:bool:1",
             "/imgdesc/InterlaceFlag:bool:0"],
    "Png":  ["/tEXt/{str=Comment}:str:wicfuzz", "/tEXt/{str=Title}:str:d2d"],
    "Jpeg": ["/app1/ifd/{ushort=270}:wstr:WICFuzz", "/app1/ifd/exif/{ushort=33434}:u8:100"],
}

# Containers Pillow can WRITE (decode/metaread inputs). Pillow has no JPEG-XR/DDS writer.
PIL_WRITE = {"Png": "PNG", "Bmp": "BMP", "Gif": "GIF", "Ico": "ICO", "Jpeg": "JPEG", "Tiff": "TIFF"}
PIL_META = ["Png", "Jpeg", "Tiff"]  # containers we attach metadata to for metaread


class R:
    def __init__(self, seed): self.s = (seed * 2862933555777941757 + 3037000493) & ((1 << 64) - 1)
    def nxt(self):
        x = self.s; x ^= (x << 13) & ((1 << 64) - 1); x ^= x >> 7; x ^= (x << 17) & ((1 << 64) - 1)
        self.s = x; return x
    def below(self, n): return self.nxt() % n
    def byte(self): return self.nxt() & 0xFF
    def pick(self, seq): return seq[self.below(len(seq))]


def raw_source(r):
    """A random small source: (src_name, guid, w, h, stride, pixel bytes)."""
    src = r.pick(NAMES)
    sg, bpp = FMT[src]
    w = 1 + r.below(16 if r.below(4) else 5)
    h = 1 + r.below(16 if r.below(4) else 5)
    stride = (w * bpp + 7) // 8
    return src, sg, w, h, stride, bytes(r.byte() for _ in range(stride * h))


def pil_bytes(r, cname, with_meta):
    """A valid container file (bytes) built with Pillow from random pixels — decode/metaread input.
    Round 5 broadens the decode differential: multi-frame GIF/TIFF (frame count + per-frame metadata),
    progressive JPEG, and an occasional truncated prefix (partial-decode HRESULT differential)."""
    from PIL import Image
    fmt = PIL_WRITE[cname]
    w, h = 1 + r.below(8), 1 + r.below(8)
    mode = r.pick(["RGB", "RGBA", "L", "P"] if fmt not in ("JPEG",) else ["RGB", "L"])
    bands = len(Image.new(mode, (1, 1)).getbands())

    def frame():
        img = Image.frombytes(mode, (w, h), bytes(r.byte() for _ in range(w * h * bands)))
        if fmt == "GIF" and mode != "P":
            img = img.convert("P")
        if fmt == "ICO" and mode != "RGBA":
            img = img.convert("RGBA")
        return img

    img = frame()
    buf = io.BytesIO()
    kw = {}
    if with_meta:
        if fmt == "PNG":
            from PIL import PngImagePlugin
            pi = PngImagePlugin.PngInfo(); pi.add_text("Comment", "wicfuzz"); pi.add_text("Title", "d2d")
            kw["pnginfo"] = pi
        else:  # JPEG / TIFF: an EXIF ImageDescription + Software
            ex = Image.Exif(); ex[0x010e] = "WICFuzz"; ex[0x0131] = "opend2d"
            kw["exif"] = ex.tobytes()
    # ~1/3 multi-frame for the frame-count codecs (animated GIF, multi-page TIFF).
    if fmt in ("GIF", "TIFF") and not with_meta and r.below(3) == 0:
        kw["save_all"] = True
        kw["append_images"] = [frame() for _ in range(1 + r.below(3))]
    if fmt == "JPEG" and r.below(3) == 0:  # progressive JPEG decode path
        kw["progressive"] = True
    img.save(buf, format=fmt, **kw)
    data = buf.getvalue()
    # ~1/6 of plain decode inputs keep only a prefix: a short/partial container the decoder must reject
    # or decode partially, matching native's HRESULT (never truncate a metadata-read input).
    if not with_meta and len(data) > 16 and r.below(6) == 0:
        data = data[: 16 + r.below(len(data) - 16)]
    return data


def gen(seed, ops):
    r = R(seed + 1)
    op = r.pick(ops)
    sc = {"seed": seed, "op": op, "input": f"wic-{seed:010d}.bin"}

    if op in ("convert", "scale", "fliprotate", "clip"):
        src, sg, w, h, _, px = raw_source(r)
        sc.update(src=sg, src_name=src, w=w, h=h)
        if op == "convert":
            dst = r.pick(NAMES)
            sc["dst"] = FMT[dst][0]; sc["dst_name"] = dst
            sc["dither"] = r.below(5)
            sc["alpha"] = [0.0, 50.0, 100.0][r.below(3)]
        elif op == "scale":
            sc["dw"] = 1 + r.below(2 * w + 1); sc["dh"] = 1 + r.below(2 * h + 1)
            # 0..4: NearestNeighbor, Linear, Cubic, Fant, HighQualityCubic (mode 4 = round-5 coverage).
            sc["mode"] = r.below(5)
        elif op == "fliprotate":
            sc["opt"] = r.below(8)
        elif op == "clip":
            sc["x"] = r.below(w); sc["y"] = r.below(h)
            sc["cw"] = 1 + r.below(w - sc["x"]); sc["ch"] = 1 + r.below(h - sc["y"])
        return sc, px

    if op == "encode":
        cname = r.pick(CNAMES)
        # 70% a container-preferred source (mostly-valid encode), else any (reject-path coverage).
        src = r.pick(CONTAINER_SRC[cname]) if r.below(10) < 7 else r.pick(NAMES)
        sg, bpp = FMT[src]
        w = 1 + r.below(8); h = 1 + r.below(8)
        stride = (w * bpp + 7) // 8
        px = bytes(r.byte() for _ in range(stride * h))
        sc.update(container=CONTAINERS[cname], container_name=cname, src=sg, src_name=src,
                  w=w, h=h, stride=stride, edits=[])
        # ~1/3 carry 1-2 metadata writes, when the container has a candidate set.
        if cname in CONTAINER_EDITS and r.below(3) == 0:
            cand = CONTAINER_EDITS[cname]
            sc["edits"] = [r.pick(cand)] if r.below(2) else cand[:2]
        return sc, px

    if op in ("decode", "metaread"):
        with_meta = op == "metaread"
        cname = r.pick(PIL_META if with_meta else list(PIL_WRITE))
        sc.update(container_name=cname)
        return sc, pil_bytes(r, cname, with_meta)

    raise SystemExit(f"unknown op {op!r}")


def main():
    outdir = pathlib.Path(sys.argv[1]); outdir.mkdir(parents=True, exist_ok=True)
    count = int(sys.argv[2]); start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    ops = (sys.argv[4].split(",") if len(sys.argv) > 4
           else ["convert", "scale", "fliprotate", "clip", "encode", "decode", "metaread"])
    for i in range(count):
        seed = start + i
        sc, px = gen(seed, ops)
        (outdir / f"wic-{seed:010d}.json").write_text(json.dumps(sc))
        (outdir / f"wic-{seed:010d}.bin").write_bytes(px)
    print(f"generated {count} WIC scenes (seeds {start}..{start + count}, ops={','.join(ops)}) in {outdir}")


if __name__ == "__main__":
    main()
