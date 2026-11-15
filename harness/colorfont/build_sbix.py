import json, sys, io
from fontTools.ttLib import TTFont, newTable
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib.tables.sbixStrike import Strike
from fontTools.ttLib.tables.sbixGlyph import Glyph as SbixGlyph
from PIL import Image
UPEM=2048; PPEM=24
# Distinctive 24x24 PNG: left half red, right half blue, opaque; a 2px green top stripe (orientation).
img=Image.new("RGBA",(PPEM,PPEM),(0,0,0,0))
for y in range(PPEM):
    for x in range(PPEM):
        c=(220,20,20,255) if x<PPEM//2 else (20,20,220,255)
        if y<2: c=(20,200,20,255)
        img.putpixel((x,y),c)
buf=io.BytesIO(); img.save(buf,"PNG"); png=buf.getvalue()
order=[".notdef","A"]
fb=FontBuilder(UPEM,isTTF=True); fb.setupGlyphOrder(order); fb.setupCharacterMap({0x41:"A"})
# A box glyph spanning the em (so DWrite has a bounding outline; sbix bitmap overrides the visual).
pen=TTGlyphPen(None)
for (x0,y0,x1,y1) in [(0,0,UPEM,UPEM)]:
    pen.moveTo((x0,y0)); pen.lineTo((x1,y0)); pen.lineTo((x1,y1)); pen.lineTo((x0,y1)); pen.closePath()
box=pen.glyph()
empty=TTGlyphPen(None); empty.moveTo((0,0)); empty.lineTo((0,0)); empty.lineTo((0,0)); empty.closePath()
fb.setupGlyf({".notdef":empty.glyph(),"A":box})
fb.setupHorizontalMetrics({".notdef":(UPEM,0),"A":(UPEM,0)})
fb.setupHorizontalHeader(ascent=1638,descent=-410)
fb.setupNameTable({"familyName":"SbixTest","styleName":"Regular"})
fb.setupOS2(sTypoAscender=1638,sTypoDescender=-410,usWinAscent=1638,usWinDescent=410)
fb.setupPost()
sbix=newTable("sbix"); sbix.version=1; sbix.flags=1; sbix.strikes={}
strike=Strike(ppem=PPEM,resolution=72)
strike.glyphs={".notdef":SbixGlyph(glyphName=".notdef"),
               "A":SbixGlyph(glyphName="A",graphicType="png ",imageData=png,originOffsetX=0,originOffsetY=0)}
sbix.strikes[PPEM]=strike
fb.font["sbix"]=sbix
out=sys.argv[1]; fb.save(out)
b=open(out,'rb').read()
scene={"version":2,"size":[40,40],"dpi":96,"ops":[{"op":"clear","color":[1.0,1.0,1.0,1.0]},
  {"op":"color_bitmap_glyph_run","font":list(b),"text":"A","em_size":float(PPEM),"baseline":[4.0,32.0],"glyph_format":"png"}]}
json.dump(scene,open(sys.argv[2],'w')); print("wrote sbix",out,len(b),"png",len(png))
