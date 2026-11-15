#!/usr/bin/env python3
# Build an OFL test font from colrtest.ttf with an SVG table, an sbix PNG strike, and a CBDT/CBLC PNG
# strike, to confirm per-glyph GetGlyphImageFormats (SVG + bitmap branches) against native.
import struct, zlib, sys
from fontTools.ttLib import TTFont, newTable

SRC = sys.argv[1]; OUT = sys.argv[2]
WANT = set((sys.argv[3] if len(sys.argv) > 3 else 'svg,sbix,cbdt').split(','))

def png(w, h, rgba):
    def chunk(typ, data):
        c = typ + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    raw = b''.join(b'\x00' + bytes(rgba) * w for _ in range(h))
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw))
            + chunk(b'IEND', b''))

f = TTFont(SRC)
go = f.getGlyphOrder()
PIMG = png(8, 8, (255, 0, 0, 255))

# --- SVG table: 'rect' (gid 1) ---
svg = newTable('SVG ')
svg.docList = [('<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">'
                '<rect id="glyph1" x="0" y="0" width="500" height="500" fill="#f00"/></svg>', 1, 1)]
if 'svg' in WANT: f['SVG '] = svg

# --- sbix table: 'colB' PNG strike at ppem 24 ---
from fontTools.ttLib.tables.sbixStrike import Strike
from fontTools.ttLib.tables.sbixGlyph import Glyph as SbixGlyph
sbix = newTable('sbix')
sbix.version = 1; sbix.flags = 1
st = Strike(ppem=24, resolution=72)
st.glyphs = {g: SbixGlyph(glyphName=g) for g in go}
st.glyphs['colB'] = SbixGlyph(glyphName='colB', graphicType='png ', imageData=PIMG, originOffsetX=0, originOffsetY=0)
sbix.strikes = {24: st}
if 'sbix' in WANT: f['sbix'] = sbix

# --- CBLC/CBDT: 'colA' PNG strike at ppem 16 (format 17 small-metrics PNG) ---
from fontTools.ttLib.tables.E_B_L_C_ import Strike as EblcStrike, SbitLineMetrics, eblc_index_sub_table_1
from fontTools.ttLib.tables.BitmapGlyphMetrics import SmallGlyphMetrics
from fontTools.ttLib.tables.C_B_D_T_ import cbdt_bitmap_format_17
colA = f.getGlyphID('colA')
cblc = newTable('CBLC'); cbdt = newTable('CBDT')
cblc.version = cbdt.version = 3.0
strike = EblcStrike()
bst = strike.bitmapSizeTable
bst.startGlyphIndex = colA
bst.endGlyphIndex = colA
bst.ppemX = 16
bst.ppemY = 16
bst.bitDepth = 32
bst.flags = 1
bst.colorRef = 0
bst.numberOfIndexSubTables = 1
bst.indexSubTableArrayOffset = 0
bst.indexTablesSize = 0
for d in ('hori', 'vert'):
    m = SbitLineMetrics()
    for a in ('ascender','descender','widthMax','caretSlopeNumerator','caretSlopeDenominator','caretOffset','minOriginSB','minAdvanceSB','maxBeforeBL','minAfterBL','pad1','pad2'):
        setattr(m, a, 0)
    m.ascender, m.descender, m.widthMax = 8, 0, 8
    setattr(strike.bitmapSizeTable, d, m)
ist = eblc_index_sub_table_1(b'', None)
ist.indexFormat = 1
ist.imageFormat = 17
ist.names = ['colA']
strike.indexSubTables = [ist]
cblc.strikes = [strike]
bmp = cbdt_bitmap_format_17(b'', None)
met = SmallGlyphMetrics()
met.height, met.width, met.BearingX, met.BearingY, met.Advance = 8, 8, 0, 8, 8
bmp.metrics = met
bmp.imageData = PIMG
cbdt.strikeData = [{'colA': bmp}]
if 'cbdt' in WANT:
    f['CBLC'] = cblc; f['CBDT'] = cbdt

f.save(OUT)
# round-trip sanity
g = TTFont(OUT)
print('OK tables:', [t for t in ('SVG ','sbix','CBLC','CBDT') if t in g])
print('glyphIDs rect/colA/colB:', f.getGlyphID('rect'), f.getGlyphID('colA'), f.getGlyphID('colB'))
