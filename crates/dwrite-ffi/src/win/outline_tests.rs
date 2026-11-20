use super::fonts::{FaceObj, FontSource};
use std::{cell::RefCell, sync::Arc};
use windows::Win32::Graphics::Direct2D::Common::*;
use windows::Win32::Graphics::DirectWrite::*;
use windows_core::{implement, Interface, Result};
use windows_numerics::Vector2;

#[implement(ID2D1SimplifiedGeometrySink)]
struct Sink { lines: RefCell<Vec<String>> }
impl Sink { fn push(&self, text: String) { self.lines.borrow_mut().push(text); } }
fn pt(p: &Vector2) -> String { format!(" {:08x} {:08x}",p.X.to_bits(),p.Y.to_bits()) }
impl ID2D1SimplifiedGeometrySink_Impl for Sink_Impl {
    fn SetFillMode(&self, m: D2D1_FILL_MODE) { self.push(format!("fill {}",m.0)); }
    fn SetSegmentFlags(&self, f: D2D1_PATH_SEGMENT) { self.push(format!("flags {}",f.0)); }
    fn BeginFigure(&self, p: &Vector2, b: D2D1_FIGURE_BEGIN) { self.push(format!("move {}{}",b.0,pt(p))); }
    fn AddLines(&self, points: *const Vector2, n: u32) {
        for p in unsafe { std::slice::from_raw_parts(points,n as usize) } { self.push(format!("line{}",pt(p))); }
    }
    fn AddBeziers(&self, segments: *const D2D1_BEZIER_SEGMENT, n: u32) {
        for s in unsafe { std::slice::from_raw_parts(segments,n as usize) } { self.push(format!("cubic{}{}{}",pt(&s.point1),pt(&s.point2),pt(&s.point3))); }
    }
    fn EndFigure(&self, e: D2D1_FIGURE_END) { self.push(format!("end {}",e.0)); }
    fn Close(&self) -> Result<()> { self.push("close".into()); Ok(()) }
}
#[test]
fn glyph_outline_stream_matches_native() {
    let root = std::env::var_os("D2D_SUITE").map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/suite"));
    if !root.join("scenes").is_dir() { return; }
    let Some(fontdir) = std::env::var_os("D2D_FONTS") else { return };
    for (filename, oracle, family) in [("arial.ttf","com-g7-outline.txt","Arial"),("YuGothR.ttc","com-g7-outline-yu.txt","Yu Gothic")] {
    let Ok(expected) = std::fs::read_to_string(root.join("oracle/win11-26100").join(oracle)) else { continue };
    let bytes = Arc::new(std::fs::read(std::path::Path::new(&fontdir).join(filename)).unwrap());
    let face: IDWriteFontFace7 = FaceObj { bytes: bytes.clone(), index: 0, family: family.into(), source: FontSource::Bytes(bytes), sims: 0, axis_values: Vec::new() }.into();
    let ids: Vec<u16> = expected.lines().nth(1).unwrap().split_whitespace().skip(1).map(|v|v.parse().unwrap()).collect();
    let mut actual = vec!["DLL C:\\Windows\\SYSTEM32\\DWrite.dll".to_string(),expected.lines().nth(1).unwrap().to_string()];
    for sideways in [false,true] { for rtl in [false,true] { for custom in [false,true] {
        actual.push(format!("case {} {} {}",sideways as u8,rtl as u8,custom as u8));
        let sink: ID2D1SimplifiedGeometrySink = Sink { lines: RefCell::new(Vec::new()) }.into();
        let advances = [11.25,13.5,9.125];
        let offsets = [DWRITE_GLYPH_OFFSET { advanceOffset: 1.25, ascenderOffset: 2.5 },DWRITE_GLYPH_OFFSET { advanceOffset: -0.75, ascenderOffset: -1.5 },DWRITE_GLYPH_OFFSET { advanceOffset: 2.0, ascenderOffset: 0.25 }];
        unsafe { face.GetGlyphRunOutline(17.25,ids.as_ptr(), if custom { Some(advances.as_ptr()) } else { None }, if custom { Some(offsets.as_ptr()) } else { None },3,sideways,rtl,&sink).unwrap(); }
        actual.extend(sink.cast_object_ref::<Sink>().unwrap().lines.borrow().iter().cloned());
        actual.push("hr 00000000".into());
    }}}
    let actual = actual.join("\n")+"\n";
    // Keep failures small enough to identify the first point/order difference.
    for (i,(got,want)) in actual.lines().zip(expected.lines()).enumerate() { assert_eq!(got,want,"line {}",i+1); }
    assert_eq!(actual.lines().count(),expected.lines().count());
    }
}

// End-to-end: a bold-simulated face (sims=1) returns an EMBOLDENED outline from the C-API
// GetGlyphRunOutline. At emSize = upm the output is design units, so the embolden is exact:
// Impact 'H' starts at the apex corner, which native bold shifts to (1093, -1659) screen.
#[test]
fn bold_glyph_run_outline_is_emboldened() {
    let root = std::env::var_os("D2D_FONTS").map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ref/windows/Fonts"));
    let Ok(data) = std::fs::read(root.join("impact.ttf")) else { return };
    let bytes = Arc::new(data);
    let font = skrifa::FontRef::from_index(&bytes, 0).unwrap();
    let h = skrifa::MetadataProvider::charmap(&font).map('H').unwrap().to_u32() as u16;
    let make = |sims: u16| -> Vec<String> {
        let face: IDWriteFontFace7 = FaceObj { bytes: bytes.clone(), index: 0, family: "Impact".into(), source: FontSource::Bytes(bytes.clone()), sims, axis_values: Vec::new() }.into();
        let sink: ID2D1SimplifiedGeometrySink = Sink { lines: RefCell::new(Vec::new()) }.into();
        unsafe { face.GetGlyphRunOutline(2048.0, [h].as_ptr(), None, None, 1, false, false, &sink).unwrap(); }
        sink.cast_object_ref::<Sink>().unwrap().lines.borrow().clone()
    };
    let reg = make(0);
    let bold = make(DWRITE_FONT_SIMULATIONS_BOLD.0 as u16);
    assert_ne!(reg, bold, "bold simulation must change the outline");
    // first op is the fill mode, second is the opening move; apex corner at (1093, -1659).
    let want = format!("move 0{}", pt(&Vector2 { X: 1093.0, Y: -1659.0 }));
    assert_eq!(bold[1], want, "bold apex move must be byte-exact to native; got {:?}", bold[1]);
    // regular apex is (1052, -1619); confirm the delta is the embolden, not a no-op.
    assert_eq!(reg[1], format!("move 0{}", pt(&Vector2 { X: 1052.0, Y: -1619.0 })));
}
