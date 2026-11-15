//! Unhinted glyph geometry emitted through IDWriteFontFace::GetGlyphRunOutline.
//!
//! Synthetic-bold faces (DWRITE_FONT_SIMULATIONS_BOLD) embolden the outline: native's scaler
//! offsets every contour edge along its normal and intersects neighbours, in design space at
//! ppem = unitsPerEm, then scales by emSize/upm. Ported from the dwrite grayscale embolden
//! and the fixed-point primitives (vector-normalize, Mul26Dot6, Intersect26Dot6, CompDiv,
//! DWRITE_FracSqrt). Strength: boldY = (upm*20 - 10)/1000, boldX = boldY + 1 (design units).
use skrifa::{FontRef, GlyphId, MetadataProvider};
use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{DrawSettings, OutlinePen};
use skrifa::raw::TableProvider;
use windows::Win32::Graphics::Direct2D::Common::*;
use windows::Win32::Graphics::DirectWrite::DWRITE_GLYPH_OFFSET;
use windows_core::Result;
use windows_numerics::Vector2;

use d2d::text::outline_embold as embold;

struct Pen<'a> {
    sink: &'a ID2D1SimplifiedGeometrySink,
    scale: f32,
    origin: [f32; 2],
    sideways: bool,
    current: Vector2,
}
impl Pen<'_> {
    fn point(&self, x: f32, y: f32) -> Vector2 {
        if self.sideways {
            Vector2 { X: self.origin[0]-y*self.scale, Y: self.origin[1]-x*self.scale }
        } else {
            Vector2 { X: self.origin[0]+x*self.scale, Y: self.origin[1]-y*self.scale }
        }
    }
    fn quad(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        let c = self.point(cx, cy);
        let end = self.point(x, y);
        // Native weights are 1/3 and (1 - 1/3), rounded separately in f32.
        let third = 1.0f32/3.0;
        let control = |p: Vector2| Vector2 { X: c.X*(1.0-third)+p.X*third, Y: c.Y*(1.0-third)+p.Y*third };
        let seg = D2D1_BEZIER_SEGMENT { point1: control(self.current), point2: control(end), point3: end };
        unsafe { self.sink.AddBeziers(&[seg]); }
        self.current = end;
    }
}
impl OutlinePen for Pen<'_> {
    fn move_to(&mut self,x:f32,y:f32) {
        self.current=self.point(x,y);
        unsafe { self.sink.BeginFigure(self.current,D2D1_FIGURE_BEGIN_FILLED); }
    }
    fn line_to(&mut self,x:f32,y:f32) {
        self.current=self.point(x,y);
        unsafe { self.sink.AddLines(&[self.current]); }
    }
    fn quad_to(&mut self,cx:f32,cy:f32,x:f32,y:f32) { self.quad(cx,cy,x,y); }
    fn curve_to(&mut self,ax:f32,ay:f32,bx:f32,by:f32,x:f32,y:f32) {
        let seg=D2D1_BEZIER_SEGMENT { point1:self.point(ax,ay),point2:self.point(bx,by),point3:self.point(x,y) };
        unsafe { self.sink.AddBeziers(&[seg]); }
        self.current=seg.point3;
    }
    fn close(&mut self) { unsafe { self.sink.EndFigure(D2D1_FIGURE_END_CLOSED); } }
}

pub(super) fn emit(font:&FontRef<'_>,em:f32,ids:&[u16],advances:Option<&[f32]>,offsets:Option<&[DWRITE_GLYPH_OFFSET]>,sideways:bool,rtl:bool,sims:u16,sink:&ID2D1SimplifiedGeometrySink) -> Result<()> {
    let metrics=font.metrics(Size::unscaled(),LocationRef::default());
    let upm=metrics.units_per_em as i32;
    let scale=em/metrics.units_per_em as f32;
    let gm=font.glyph_metrics(Size::unscaled(),LocationRef::default());
    let outlines=font.outline_glyphs();
    let (ascent,descent)=font.os2().map(|t|(t.s_typo_ascender() as f32,t.s_typo_descender() as f32)).unwrap_or((metrics.ascent,metrics.descent));
    let bold = sims & (windows::Win32::Graphics::DirectWrite::DWRITE_FONT_SIMULATIONS_BOLD.0 as u16) != 0;
    let mut pen_x=0.0;
    unsafe { sink.SetFillMode(D2D1_FILL_MODE_WINDING); }
    for (i,&id) in ids.iter().enumerate() {
        let gid=GlyphId::new(id as u32);
        let aw=gm.advance_width(gid).unwrap_or(0.0);
        let nominal=if sideways { ascent-descent } else { aw }*scale;
        let offset=offsets.map(|o|o[i]).unwrap_or_default();
        let x=if rtl { -pen_x-nominal-offset.advanceOffset } else { pen_x+offset.advanceOffset };
        let origin=if sideways { [x+ascent*scale,aw*scale*0.5-offset.ascenderOffset] } else { [x,0.0-offset.ascenderOffset] };
        if let Some(glyph)=outlines.get(gid) {
            if bold {
                let mut p=Pen { sink,scale,origin,sideways,current:Vector2::default() };
                embold::draw_bold(&glyph, upm, &mut p).map_err(|_|windows_core::Error::from(super::E_INVALIDARG))?;
            } else {
                let mut p=Pen { sink,scale,origin,sideways,current:Vector2::default() };
                glyph.draw(DrawSettings::unhinted(Size::unscaled(),LocationRef::default()),&mut p).map_err(|_|windows_core::Error::from(super::E_INVALIDARG))?;
            }
        }
        pen_x+=advances.map(|a|a[i]).unwrap_or(nominal);
    }
    Ok(())
}
