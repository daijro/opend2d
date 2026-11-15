import json, sys
from fontTools.ttLib import newTable
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
UPEM=2048
fb=FontBuilder(UPEM,isTTF=True); fb.setupGlyphOrder([".notdef","A"]); fb.setupCharacterMap({0x41:"A"})
pen=TTGlyphPen(None); pen.moveTo((0,0)); pen.lineTo((UPEM,0)); pen.lineTo((UPEM,UPEM)); pen.lineTo((0,UPEM)); pen.closePath()
empty=TTGlyphPen(None); empty.moveTo((0,0)); empty.lineTo((0,0)); empty.lineTo((0,0)); empty.closePath()
fb.setupGlyf({".notdef":empty.glyph(),"A":pen.glyph()})
fb.setupHorizontalMetrics({".notdef":(UPEM,0),"A":(UPEM,0)})
fb.setupHorizontalHeader(ascent=1638,descent=-410)
fb.setupNameTable({"familyName":"SvgTest","styleName":"Regular"})
fb.setupOS2(sTypoAscender=1638,sTypoDescender=-410,usWinAscent=1638,usWinDescent=410); fb.setupPost()
svg='<svg xmlns="http://www.w3.org/2000/svg"><g id="glyph1">'\
    '<rect x="0" y="0" width="1024" height="2048" fill="rgb(220,20,20)"/>'\
    '<rect x="1024" y="0" width="1024" height="2048" fill="rgb(20,20,220)"/>'\
    '<rect x="0" y="0" width="2048" height="170" fill="rgb(20,200,20)"/></g></svg>'
t=newTable("SVG "); t.docList=[(svg,1,1)]; fb.font["SVG "]=t
out=sys.argv[1]; fb.save(out); b=open(out,'rb').read()
scene={"version":2,"size":[40,40],"dpi":96,"ops":[{"op":"clear","color":[1.0,1.0,1.0,1.0]},
  {"op":"svg_glyph_run","font":list(b),"text":"A","em_size":24.0,"baseline":[4.0,32.0],"color":[0.0,0.0,0.0,1.0]}]}
json.dump(scene,open(sys.argv[2],'w')); print("wrote svg",len(b))
