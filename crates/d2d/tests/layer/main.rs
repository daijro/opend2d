//! Phase 2A-2 layer/mask error tests: the `FillOpacityMask` validation the oracle pixel scenes
//! can't reach (a latched draw error renders no bitmap, so native's own corpus has no such scene).
//! The order and HRESULTs follow Direct2D's `FillOpacityMask`, confirmed by the
//! `2a2-07-opacity-mask-not-aliased` capture, whose `EndDraw`
//! returns `0x88990001` (WRONG_STATE) tagged with the failing op.

use d2d::bitmap::Bitmap;
use d2d::state::*;
use d2d::{bitmap_target, Device, DeviceContext, Factory};

fn ctx(dev: &Device, w: u32, h: u32) -> DeviceContext {
    let mut dc = dev.create_device_context();
    dc.set_target_bitmap(bitmap_target(w, h, 96.0));
    dc
}

fn mask() -> Bitmap {
    Bitmap::from_straight_rgba(4, 4, &[255, 255, 255, 128].repeat(16))
}

// Native requires the current antialias mode to be ALIASED (this+82==1); the default PER_PRIMITIVE
// latches WRONG_STATE, surfaced at EndDraw with the failing op's tag — exactly the 2a2-07 oracle.
#[test]
fn fill_opacity_mask_requires_aliased() {
    let f = Factory::new(false);
    let mut dc = ctx(&f.create_device(), 64, 48);
    let brush = dc.create_solid_color_brush([0.2, 0.5, 0.9, 1.0], None);
    dc.begin_draw();
    dc.set_tags(3, 0);
    dc.fill_opacity_mask(&mask(), &brush, 0, Some([8.0, 8.0, 56.0, 40.0]), Some([0.0, 0.0, 4.0, 4.0]));
    assert_eq!(dc.end_draw(), (WRONG_STATE, 3, 0));
}

// ALIASED set: a valid fill latches nothing.
#[test]
fn fill_opacity_mask_aliased_ok() {
    let f = Factory::new(false);
    let mut dc = ctx(&f.create_device(), 64, 48);
    let brush = dc.create_solid_color_brush([0.2, 0.5, 0.9, 1.0], None);
    dc.begin_draw();
    dc.set_antialias_mode(1);
    dc.fill_opacity_mask(&mask(), &brush, 0, None, None);
    assert_eq!(dc.end_draw(), (S_OK, 0, 0));
}

// `content > 2` is E_INVALIDARG — but only after the ALIASED check, so content is validated in
// aliased mode (in non-aliased mode WRONG_STATE wins).
#[test]
fn fill_opacity_mask_content_out_of_range() {
    let f = Factory::new(false);
    let mut dc = ctx(&f.create_device(), 64, 48);
    let brush = dc.create_solid_color_brush([0.2, 0.5, 0.9, 1.0], None);
    dc.begin_draw();
    dc.set_antialias_mode(1);
    dc.fill_opacity_mask(&mask(), &brush, 3, None, None);
    assert_eq!(dc.end_draw().0, E_INVALIDARG);
}

// A brush from another factory is WRONG_FACTORY (the aliased check passes first).
#[test]
fn fill_opacity_mask_cross_factory_brush() {
    let f1 = Factory::new(false);
    let f2 = Factory::new(false);
    let mut dc = ctx(&f1.create_device(), 64, 48);
    let foreign = ctx(&f2.create_device(), 64, 48).create_solid_color_brush([1.0, 0.0, 0.0, 1.0], None);
    dc.begin_draw();
    dc.set_antialias_mode(1);
    dc.fill_opacity_mask(&mask(), &foreign, 0, None, None);
    assert_eq!(dc.end_draw().0, WRONG_FACTORY);
}

// Outside a draw it is WRONG_STATE (the first thing native checks).
#[test]
fn fill_opacity_mask_outside_draw() {
    let f = Factory::new(false);
    let mut dc = ctx(&f.create_device(), 64, 48);
    let brush = dc.create_solid_color_brush([0.2, 0.5, 0.9, 1.0], None);
    dc.set_antialias_mode(1);
    dc.fill_opacity_mask(&mask(), &brush, 0, None, None);
    assert_eq!(dc.end_draw().0, WRONG_STATE);
}
