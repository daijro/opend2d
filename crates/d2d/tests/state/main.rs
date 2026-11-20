//! Phase 1A state-machine tests: the HRESULT/tag/latch rules, resource-domain checks, drawing-state
//! blocks, push/pop balance and command-list replay that the oracle pixel scenes don't reach
//! (native's only error scene uses effects). Rules follow the method table, error/tag model
//! and layer push/pop rules.

use d2d::bitmap::Bitmap;
use d2d::effects::{Input, Prop};
use d2d::state::*;
use d2d::{bitmap_target, Device, DeviceContext, Factory};
use std::rc::Rc;

fn ctx(dev: &Device, w: u32, h: u32) -> DeviceContext {
    let mut dc = dev.create_device_context();
    dc.set_target_bitmap(bitmap_target(w, h, 96.0));
    dc
}

#[test]
fn begin_end_roundtrip() {
    let f = Factory::new(false);
    let mut dc = ctx(&f.create_device(), 16, 16);
    dc.begin_draw();
    dc.clear(Some([1.0, 1.0, 1.0, 1.0]));
    assert_eq!(dc.end_draw(), (S_OK, 0, 0));
    assert!(!dc.is_in_draw());
}

#[test]
fn end_without_begin_is_wrong_state() {
    let f = Factory::new(false);
    let mut dc = ctx(&f.create_device(), 16, 16);
    assert_eq!(dc.end_draw().0, WRONG_STATE);
}

#[test]
fn begin_while_drawing_is_wrong_state() {
    let f = Factory::new(false);
    let mut dc = ctx(&f.create_device(), 16, 16);
    dc.begin_draw();
    dc.begin_draw(); // latches WRONG_STATE
    assert_eq!(dc.end_draw().0, WRONG_STATE);
}

#[test]
fn draw_outside_begin_is_gated() {
    // Not drawing ⇒ the fill latches WRONG_STATE and produces no pixels. (The latch is unobservable
    // via EndDraw because the next BeginDraw clears the error; the gate is what matters.)
    let f = Factory::new(false);
    let mut dc = ctx(&f.create_device(), 16, 16);
    let brush = dc.create_solid_color_brush([0.0, 0.0, 0.0, 1.0], None);
    dc.fill_rectangle([0.0, 0.0, 8.0, 8.0], &brush);
    assert!(dc.target_bitmap().unwrap().data.iter().all(|&b| b == 0));
}

#[test]
fn placed_effect_obeys_draw_state() {
    use d2d::bitmap::Bitmap;
    use d2d::effects::Input;
    use std::rc::Rc;
    use windows_core::GUID;

    let f = Factory::new(false);
    let mut dc = ctx(&f.create_device(), 16, 16);
    let effect = dc.create_effect(GUID::from_u128(0xe8467b04_ec61_4b8a_b5de_d4d73debea5a)).unwrap();
    effect.set_input(0, Input::Bitmap(Rc::new(Bitmap::from_straight_rgba(4, 4, &[255; 64]))));
    dc.set_transform([1.0, 0.0, 0.0, 1.0, 0.25, 0.375]);

    dc.draw_image_effect(&effect, [0.0; 2], 0, 0);
    assert!(dc.target_bitmap().unwrap().data.iter().all(|&b| b == 0));

    dc.begin_draw();
    dc.set_tags(7, 9);
    dc.latch_error(E_INVALIDARG);
    dc.draw_image_effect(&effect, [0.0; 2], 0, 0);
    assert!(dc.target_bitmap().unwrap().data.iter().all(|&b| b == 0));
    assert_eq!(dc.end_draw(), (E_INVALIDARG, 7, 9));

    dc.begin_draw();
    dc.draw_image_effect(&effect, [0.0; 2], 0, 0);
    assert_eq!(dc.end_draw().0, S_OK);
    assert!(dc.target_bitmap().unwrap().data.iter().any(|&b| b != 0));
}

#[test]
fn error_latch_keeps_first_error_and_freezes_tags() {
    let f = Factory::new(false);
    let dev = f.create_device();
    let other = Factory::new(false);
    let foreign = ctx(&other.create_device(), 16, 16).create_solid_color_brush([0.0; 4], None);

    let mut dc = ctx(&dev, 16, 16);
    dc.begin_draw();
    dc.set_tags(7, 7);
    dc.fill_rectangle([0.0, 0.0, 8.0, 8.0], &foreign); // WRONG_FACTORY, err tags frozen at (7,7)
    dc.set_tags(9, 9);
    dc.fill_rectangle([0.0, 0.0, 8.0, 8.0], &foreign); // in error ⇒ no-op, no overwrite
    assert_eq!(dc.get_tags(), (9, 9)); // current tags still move
    assert_eq!(dc.end_draw(), (WRONG_FACTORY, 7, 7)); // first error + frozen tags
}

#[test]
fn foreign_device_same_factory_is_wrong_resource_domain() {
    let f = Factory::new(false);
    let dev_a = f.create_device();
    let brush_b = ctx(&f.create_device(), 16, 16).create_solid_color_brush([0.0; 4], None);
    let mut dc = ctx(&dev_a, 16, 16);
    dc.begin_draw();
    dc.fill_rectangle([0.0, 0.0, 8.0, 8.0], &brush_b);
    assert_eq!(dc.end_draw().0, WRONG_RESOURCE_DOMAIN);
}

#[test]
fn begin_resets_error_but_tags_persist() {
    let f = Factory::new(false);
    let foreign = ctx(&Factory::new(false).create_device(), 16, 16).create_solid_color_brush([0.0; 4], None);
    let mut dc = ctx(&f.create_device(), 16, 16);
    dc.begin_draw();
    dc.set_tags(5, 6);
    dc.fill_rectangle([0.0, 0.0, 8.0, 8.0], &foreign);
    assert_eq!(dc.end_draw().0, WRONG_FACTORY);
    dc.begin_draw(); // clears the error
    assert_eq!(dc.get_tags(), (5, 6)); // last SetTags persists across pairs
    assert_eq!(dc.end_draw(), (S_OK, 0, 0));
}

#[test]
fn push_pop_rules() {
    let f = Factory::new(false);
    let dev = f.create_device();

    // open clip left at EndDraw ⇒ PUSH_POP_UNBALANCED
    let mut dc = ctx(&dev, 16, 16);
    dc.begin_draw();
    dc.push_axis_aligned_clip([0.0, 0.0, 16.0, 16.0], 0);
    assert_eq!(dc.end_draw().0, PUSH_POP_UNBALANCED);

    // pop with empty stack ⇒ PUSH_POP_UNBALANCED
    let mut dc = ctx(&dev, 16, 16);
    dc.begin_draw();
    dc.pop_axis_aligned_clip();
    assert_eq!(dc.end_draw().0, PUSH_POP_UNBALANCED);

    // popping a layer when the top is a clip ⇒ POP_DID_NOT_MATCH_PUSH
    let mut dc = ctx(&dev, 16, 16);
    dc.begin_draw();
    dc.push_axis_aligned_clip([0.0, 0.0, 16.0, 16.0], 0);
    dc.pop_layer();
    assert_eq!(dc.end_draw().0, POP_DID_NOT_MATCH_PUSH);

    // bad antialias mode on a clip ⇒ E_INVALIDARG
    let mut dc = ctx(&dev, 16, 16);
    dc.begin_draw();
    dc.push_axis_aligned_clip([0.0, 0.0, 16.0, 16.0], 2);
    assert_eq!(dc.end_draw().0, E_INVALIDARG);
}

#[test]
fn setters_validate_ranges() {
    let f = Factory::new(false);
    let mut dc = ctx(&f.create_device(), 16, 16);
    dc.set_antialias_mode(5);
    assert_eq!(dc.get_antialias_mode(), 0);
    dc.set_antialias_mode(1);
    assert_eq!(dc.get_antialias_mode(), 1);
    dc.set_primitive_blend(9);
    assert_eq!(dc.get_primitive_blend(), 0);
    dc.set_primitive_blend(4);
    assert_eq!(dc.get_primitive_blend(), 4);
    dc.set_unit_mode(2);
    assert_eq!(dc.get_unit_mode(), 0);
    dc.set_text_antialias_mode(7);
    assert_eq!(dc.get_text_antialias_mode(), 0);
    dc.set_text_antialias_mode(3);
    assert_eq!(dc.get_text_antialias_mode(), 3);
}

#[test]
fn dpi_rules() {
    let f = Factory::new(false);
    let mut dc = ctx(&f.create_device(), 16, 16);
    dc.set_dpi(192.0, 144.0);
    assert_eq!(dc.get_dpi(), (192.0, 144.0));
    dc.set_dpi(-1.0, 5.0); // mixed/negative ⇒ dropped
    assert_eq!(dc.get_dpi(), (192.0, 144.0));
    dc.set_dpi(0.0, 0.0); // ⇒ default 96
    assert_eq!(dc.get_dpi(), (96.0, 96.0));
}

#[test]
fn drawing_state_block_roundtrip_and_factory_guard() {
    let f = Factory::new(false);
    let mut dc = ctx(&f.create_device(), 16, 16);
    dc.set_antialias_mode(0);
    dc.set_transform([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    dc.set_tags(1, 2);
    let mut block = f.create_drawing_state_block();
    dc.save_drawing_state(&mut block);

    dc.set_antialias_mode(1);
    dc.set_transform([2.0, 0.0, 0.0, 2.0, 5.0, 5.0]);
    dc.set_tags(9, 9);
    dc.set_primitive_blend(3);
    dc.set_unit_mode(1);
    dc.restore_drawing_state(&block);
    assert_eq!(dc.get_antialias_mode(), 0);
    assert_eq!(dc.get_transform(), [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    assert_eq!(dc.get_tags(), (1, 2));
    assert_eq!(dc.get_primitive_blend(), 0);
    assert_eq!(dc.get_unit_mode(), 0);

    // a block from another factory is a silent no-op both ways
    let other = Factory::new(false);
    let mut foreign_block = other.create_drawing_state_block();
    dc.set_antialias_mode(1);
    dc.save_drawing_state(&mut foreign_block); // no-op
    dc.restore_drawing_state(&foreign_block); // no-op
    assert_eq!(dc.get_antialias_mode(), 1);
}

#[test]
fn flush_keeps_draw_open_and_clears_error() {
    let f = Factory::new(false);
    let foreign = ctx(&Factory::new(false).create_device(), 16, 16).create_solid_color_brush([0.0; 4], None);
    let mut dc = ctx(&f.create_device(), 16, 16);
    dc.begin_draw();
    dc.set_tags(3, 4);
    dc.fill_rectangle([0.0, 0.0, 8.0, 8.0], &foreign);
    assert_eq!(dc.flush(), (WRONG_FACTORY, 3, 4)); // reports then clears
    assert!(dc.is_in_draw()); // pair stays open
    assert_eq!(dc.end_draw(), (S_OK, 0, 0)); // error was cleared by flush
}

#[test]
fn flush_rejects_a_recording_target_but_clear_records() {
    let f = Factory::new(false);
    let mut dc = ctx(&f.create_device(), 16, 16);
    let list = dc.create_command_list();
    dc.begin_draw();
    dc.record_into(&list);
    // Flush on a command-list target ⇒ WRONG_STATE (native).
    // Flush reports the error then clears it.
    assert_eq!(dc.flush().0, WRONG_STATE);
    // Clear on a command-list target RECORDS — it does NOT reject (G19, VM-verified: EndDraw S_OK,
    // the recorded Clear replays within the list's drawn area). Draws nothing to the device here.
    dc.clear(Some([1.0; 4]));
    dc.stop_recording();
    // The flush's WRONG_STATE was cleared by the flush; the recorded Clear latches nothing ⇒ S_OK.
    assert_eq!(dc.end_draw().0, S_OK);
}

#[test]
fn command_list_replay_equals_direct() {
    let f = Factory::new(false);
    let dev = f.create_device();

    let mut direct = ctx(&dev, 64, 48);
    direct.begin_draw();
    direct.clear(Some([1.0, 1.0, 1.0, 1.0]));
    let red = direct.create_solid_color_brush([0.8, 0.1, 0.1, 1.0], None);
    let blue = direct.create_solid_color_brush([0.1, 0.1, 0.8, 0.6], None);
    direct.fill_rectangle([8.0, 8.0, 40.0, 30.0], &red);
    direct.fill_rectangle([20.0, 18.0, 56.0, 42.0], &blue);
    let _ = direct.end_draw();
    let direct = direct.take_target().unwrap();

    let mut replay = ctx(&dev, 64, 48);
    replay.begin_draw();
    replay.clear(Some([1.0, 1.0, 1.0, 1.0]));
    let list = replay.create_command_list();
    replay.record_into(&list);
    let red = replay.create_solid_color_brush([0.8, 0.1, 0.1, 1.0], None);
    let blue = replay.create_solid_color_brush([0.1, 0.1, 0.8, 0.6], None);
    replay.fill_rectangle([8.0, 8.0, 40.0, 30.0], &red);
    replay.fill_rectangle([20.0, 18.0, 56.0, 42.0], &blue);
    list.close();
    replay.stop_recording();
    replay.draw_image(&list, [0.0, 0.0], 0);
    let _ = replay.end_draw();
    let replay = replay.take_target().unwrap();

    assert_eq!(direct.data, replay.data, "replayed command list must match direct rendering");
}

#[test]
fn draw_image_replay_equals_direct() {
    // G12: DrawImage of a bitmap AND of an effect, recorded into a command list, must replay
    // byte-for-byte identically to drawing them directly (the record model's replay ≡ direct).
    let f = Factory::new(false);
    let dev = f.create_device();
    let blue = Rc::new(Bitmap::from_straight_rgba(8, 8, &[0u8, 0, 255, 255].repeat(64)));

    // Both scenes: clear white, a Flood effect (source-over, fills the target), then an opaque 8×8
    // bitmap DrawImage at an offset on top — exercising both the effect- and bitmap-record paths.
    let scene = |dc: &mut DeviceContext, record: bool| {
        dc.begin_draw();
        dc.clear(Some([1.0, 1.0, 1.0, 1.0]));
        let list = dc.create_command_list();
        if record {
            dc.record_into(&list);
        }
        let flood = dc.create_effect(d2d::abi::CLSID_D2D1Flood).unwrap();
        flood.set_value(0, Prop::V4([0.2, 0.6, 0.3, 1.0])).unwrap();
        dc.draw_image_effect(&flood, [0.0, 0.0], 0, 0);
        dc.draw_image_bitmap(&blue, [20.0, 16.0], None, 0, 0);
        if record {
            list.close();
            dc.stop_recording();
            dc.draw_image(&list, [0.0, 0.0], 0);
        }
        let _ = dc.end_draw();
    };

    let mut direct = ctx(&dev, 64, 48);
    scene(&mut direct, false);
    let direct = direct.take_target().unwrap();

    let mut replay = ctx(&dev, 64, 48);
    scene(&mut replay, true);
    let replay = replay.take_target().unwrap();

    assert_eq!(direct.data, replay.data, "recorded DrawImage (bitmap + effect) must replay ≡ direct");
    // Sanity: the overlay actually drew (blue block centre is blue, not the flood colour).
    let i = 20 * 64 * 4 + 24 * 4; // pixel (24,20) inside the 8×8 block at (20,16)
    assert_eq!(&replay.data[i..i + 4], &[255, 0, 0, 255], "blue bitmap overlay (BGRA) at its centre");
}

#[test]
fn brush_fill_replay_equals_direct() {
    // G6: a gradient-brush fill recorded into a command list re-rasters on replay (not flattened), so
    // it is byte-exact to a direct draw at a non-zero replay offset — the brush pattern translates with
    // the geometry, as native's replay does. A non-SOURCE_OVER primitive blend makes the list unsafe to
    // inline (SafeForDrawImageFastPathChecker): native composites it as an image, so replay ≡ direct
    // only where the blend is invisible (an opaque COPY); ADD differs on native too (suite
    // inv-brush-b3-o0-*).
    let f = Factory::new(false);
    let dev = f.create_device();
    let stops = [(0.0, [1.0, 0.0, 0.0, 1.0]), (1.0, [0.0, 0.0, 1.0, 1.0])];

    // `blend` is the primitive blend (0 SOURCE_OVER, 1 COPY, 3 ADD); `off` the DrawImage replay offset.
    let check = |blend: u32, off: [f32; 2]| {
        let scene = |dc: &mut DeviceContext, record: bool| {
            dc.begin_draw();
            dc.clear(Some([1.0, 1.0, 1.0, 1.0]));
            dc.set_primitive_blend(blend);
            let list = dc.create_command_list();
            if record {
                // Record at the original position; DrawImage(list, off) translates the whole fill
                // (rect AND gradient) by `off`, as native does.
                let g = dc.create_linear_gradient_brush([8.0, 0.0], [56.0, 0.0], &stops, 0, 0, 1.0, None);
                dc.record_into(&list);
                dc.fill_rectangle_brush([8.0, 8.0, 56.0, 40.0], &g);
                list.close();
                dc.stop_recording();
                dc.set_primitive_blend(0);
                dc.draw_image(&list, off, 0);
            } else {
                // Direct equivalent of the translated replay: the rect AND the gradient endpoints both
                // shifted by `off`.
                let g = dc.create_linear_gradient_brush([8.0 + off[0], off[1]], [56.0 + off[0], off[1]], &stops, 0, 0, 1.0, None);
                dc.fill_rectangle_brush([8.0 + off[0], 8.0 + off[1], 56.0 + off[0], 40.0 + off[1]], &g);
            }
            let _ = dc.end_draw();
        };
        let mut direct = ctx(&dev, 72, 56);
        scene(&mut direct, false);
        let direct = direct.take_target().unwrap();
        let mut replay = ctx(&dev, 72, 56);
        scene(&mut replay, true);
        let replay = replay.take_target().unwrap();
        assert_eq!(direct.data, replay.data, "recorded brush fill (blend {blend}, offset {off:?}) must replay ≡ direct");
        replay.data
    };

    let d = check(0, [0.0, 0.0]); // SOURCE_OVER, no offset
    check(1, [0.0, 0.0]); // COPY of an opaque gradient (realized as an image, still ≡ direct)
    check(0, [6.0, 4.0]); // SOURCE_OVER at a non-zero replay offset (brush translates with it)

    // Sanity: the gradient actually painted (left edge red-dominant, right edge blue-dominant, BGRA).
    let left = 8 * 72 * 4 + 9 * 4;
    let right = 8 * 72 * 4 + 54 * 4;
    assert!(d[left + 2] > d[left], "left of gradient is red-dominant");
    assert!(d[right] > d[right + 2], "right of gradient is blue-dominant");
}

#[test]
fn glyph_run_replay_equals_direct() {
    // G6: a glyph run recorded into a command list must replay byte-for-byte like drawing it directly.
    // Text can't flatten (its gamma blend reads the destination), so the run is re-rastered on replay;
    // this pins that against a direct draw. Uses the tiny embedded COLR test font (no system fonts).
    use d2d::text::{Glyph, GlyphRun, RenderingParams};
    let font: &[u8] = include_bytes!("../../src/text/color/testdata/colrtest.ttf");
    let f = Factory::new(false);
    let dev = f.create_device();
    let glyphs = [Glyph { id: 1, advance: 24.0, offset: [0.0, 0.0] }, Glyph { id: 2, advance: 24.0, offset: [0.0, 0.0] }];
    let run = |origin: [f32; 2]| GlyphRun {
        font_data: font,
        face_index: 0,
        em_size: 32.0,
        glyphs: &glyphs,
        baseline_origin: origin,
        measuring_mode: 0,
        is_sideways: false,
        bidi_level: 0,
        no_snap: false,
        simulations: 0,
        coords: &[],
    };

    let scene = |dc: &mut DeviceContext, record: bool| {
        let brush = dc.create_solid_color_brush([0.0, 0.0, 0.0, 1.0], None);
        let params = RenderingParams::default();
        dc.begin_draw();
        dc.clear(Some([1.0, 1.0, 1.0, 1.0]));
        let list = dc.create_command_list();
        if record {
            dc.record_into(&list);
        }
        dc.draw_glyph_run(&run([6.0, 44.0]), &brush, &params);
        if record {
            list.close();
            dc.stop_recording();
            dc.draw_image(&list, [0.0, 0.0], 0);
        }
        let _ = dc.end_draw();
    };

    let mut direct = ctx(&dev, 80, 64);
    scene(&mut direct, false);
    let direct = direct.take_target().unwrap();

    let mut replay = ctx(&dev, 80, 64);
    scene(&mut replay, true);
    let replay = replay.take_target().unwrap();

    assert_eq!(direct.data, replay.data, "recorded glyph run must replay ≡ direct");
    // Sanity: the glyph run actually painted something (not an all-white target).
    assert!(direct.data.chunks_exact(4).any(|p| p != [255, 255, 255, 255]), "glyph run should paint coverage");
}

#[test]
fn glyph_run_rejects_params_that_conflict_with_text_aa() {
    // DrawGlyphRun latches E_INVALIDARG when the params' rendering mode conflicts with the text AA mode,
    // or when an explicit bitmap mode meets a 2^15-px em; a negative em draws nothing, without error.
    use d2d::text::{Glyph, GlyphRun, RenderingParams};
    let font: &[u8] = include_bytes!("../../src/text/color/testdata/colrtest.ttf");
    let dev = Factory::new(false).create_device();
    let glyphs = [Glyph { id: 1, advance: 24.0, offset: [0.0, 0.0] }];
    let hr = |aa: u32, mode: u32, em: f32, scale: f32| {
        let run = GlyphRun { font_data: font, face_index: 0, em_size: em, glyphs: &glyphs, baseline_origin: [4.0, 12.0], measuring_mode: 0, is_sideways: false, bidi_level: 0, no_snap: false, simulations: 0, coords: &[] };
        let mut dc = ctx(&dev, 16, 16);
        let brush = dc.create_solid_color_brush([0.0, 0.0, 0.0, 1.0], None);
        dc.begin_draw();
        dc.set_transform([scale, 0.0, 0.0, scale, 0.0, 0.0]);
        dc.set_text_antialias_mode(aa);
        dc.draw_glyph_run(&run, &brush, &RenderingParams { rendering_mode: mode, ..Default::default() });
        dc.end_draw().0
    };
    // (text AA, mode): ClearType rejects ALIASED/OUTLINE, grayscale ALIASED, ALIASED the GDI/natural
    // modes (NATURAL_SYMMETRIC_DOWNSAMPLED reads as NATURAL_SYMMETRIC); DEFAULT text AA accepts any.
    for (aa, mode, bad) in [(1, 1, true), (1, 6, true), (1, 4, false), (2, 1, true), (2, 6, false), (3, 2, true), (3, 5, true), (3, 7, true), (3, 1, false), (3, 6, false), (0, 1, false), (0, 3, false)] {
        assert_eq!(hr(aa, mode, 10.0, 1.0) == E_INVALIDARG, bad, "text aa {aa}, mode {mode}");
    }
    assert_eq!(hr(3, 2, -10.0, 1.0), S_OK, "a negative em returns before the check");
    assert_eq!(hr(2, 4, 10.0, 4000.0), E_INVALIDARG, "explicit mode at a 40000-px em");
    assert_eq!(hr(2, 0, 10.0, 4000.0), S_OK, "DEFAULT resolves a huge em to OUTLINE");
    assert_eq!(hr(2, 6, 10.0, 4000.0), S_OK, "OUTLINE never reaches the bitmap cache");
}

#[test]
fn nested_composited_list_replay_is_an_image() {
    // G6: a non-SOURCE_OVER DrawImage(commandList) recorded into an outer list makes the outer list
    // unsafe to inline: native realizes it as an image (the inner PLUS lands on transparent) and
    // composites that SOURCE_OVER, so the replay is NOT the direct PLUS (suite inv-nested-*).
    let f = Factory::new(false);
    let dev = f.create_device();
    let inner_of = |dc: &mut DeviceContext| {
        let list = dc.create_command_list();
        dc.record_into(&list);
        let red = dc.create_solid_color_brush([0.8, 0.2, 0.1, 0.7], None);
        dc.fill_rectangle([10.0, 10.0, 50.0, 38.0], &red);
        list.close();
        dc.stop_recording();
        list
    };
    let composite = 9; // PLUS (additive) — a non-SOURCE_OVER mode

    let mut direct = ctx(&dev, 64, 48);
    direct.begin_draw();
    direct.clear(Some([0.3, 0.5, 0.7, 1.0]));
    let di = inner_of(&mut direct);
    direct.draw_image(&di, [4.0, 2.0], composite);
    let _ = direct.end_draw();
    let direct = direct.take_target().unwrap();

    let mut replay = ctx(&dev, 64, 48);
    replay.begin_draw();
    replay.clear(Some([0.3, 0.5, 0.7, 1.0]));
    let ri = inner_of(&mut replay);
    let outer = replay.create_command_list();
    replay.record_into(&outer);
    replay.draw_image(&ri, [4.0, 2.0], composite); // recorded as a DrawImageList unit
    outer.close();
    replay.stop_recording();
    replay.draw_image(&outer, [0.0, 0.0], 0);
    let _ = replay.end_draw();
    let replay = replay.take_target().unwrap();

    assert_ne!(direct.data, replay.data, "a nested PLUS list composites as an image, not inline");
    // The image's content is the inner fill over transparent, composited over the backdrop.
    let mut over = ctx(&dev, 64, 48);
    over.begin_draw();
    over.clear(Some([0.3, 0.5, 0.7, 1.0]));
    let oi = inner_of(&mut over);
    over.draw_image(&oi, [4.0, 2.0], 0);
    let _ = over.end_draw();
    assert_eq!(over.take_target().unwrap().data, replay.data, "the unsafe outer list replays as its image SOURCE_OVER");
}

#[test]
fn effect_transformed_replay_equals_direct() {
    // G6: a DrawImage(effect) under a scaled / rotated transform recorded into a command list must
    // replay byte-for-byte like drawing it directly (the realized output + transform state are
    // recorded, and the same placement re-run on replay). A non-SOURCE_OVER one makes the list unsafe
    // to inline, and native composites it as an image instead (suite inv-effect-2-*).
    use d2d::effects::Prop;
    let f = Factory::new(false);
    let dev = f.create_device();
    let check = |xform: [f32; 6], composite: u32| {
        let scene = |dc: &mut DeviceContext, record: bool| {
            let flood = dc.create_effect(d2d::abi::CLSID_D2D1Flood).unwrap();
            flood.set_value(0, Prop::V4([0.2, 0.6, 0.3, 0.8])).unwrap();
            dc.begin_draw();
            dc.clear(Some([1.0, 1.0, 1.0, 1.0]));
            let list = dc.create_command_list();
            if record {
                dc.record_into(&list);
                dc.set_transform(xform);
                dc.draw_image_effect(&flood, [2.0, 2.0], 0, composite);
                dc.set_transform([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
                list.close();
                dc.stop_recording();
                dc.draw_image(&list, [0.0, 0.0], 0);
            } else {
                dc.set_transform(xform);
                dc.draw_image_effect(&flood, [2.0, 2.0], 0, composite);
                dc.set_transform([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
            }
            let _ = dc.end_draw();
        };
        let mut direct = ctx(&dev, 64, 48);
        scene(&mut direct, false);
        let direct = direct.take_target().unwrap();
        let mut replay = ctx(&dev, 64, 48);
        scene(&mut replay, true);
        let replay = replay.take_target().unwrap();
        assert_eq!(direct.data, replay.data, "effect (xform {xform:?}, composite {composite}) must replay ≡ direct");
        direct.data
    };
    let d = check([2.0, 0.0, 0.0, 2.0, 0.0, 0.0], 0); // 2× scale, SOURCE_OVER
    check([1.5, 0.5, -0.5, 1.5, 4.0, 2.0], 0); // rotate+scale+translate, SOURCE_OVER
    // Sanity: the scaled flood actually painted (centre pixel is not the white clear).
    let i = 24 * 64 * 4 + 24 * 4;
    assert_ne!(&d[i..i + 4], &[255, 255, 255, 255], "scaled flood effect should paint");
}

#[test]
fn draw_image_requires_a_closed_list() {
    let f = Factory::new(false);
    let mut dc = ctx(&f.create_device(), 16, 16);
    let list = dc.create_command_list(); // left open
    dc.begin_draw();
    dc.draw_image(&list, [0.0, 0.0], 0);
    assert_eq!(dc.end_draw().0, WRONG_STATE);
}
