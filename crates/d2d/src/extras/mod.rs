//! Direct2D 1.1–1.3 extras: the newer resources no other spec owns —
//! sprite batches, ink, gradient meshes, image sources, SVG, 3D LUTs, printing and `BlendImage`.
//!
//! **Sprite batches** are WARP primitives, not `DrawBitmap`s: native's `DrawSpriteBatch`
//! tessellates each sprite's quad, rasterizes it aliased and samples the bitmap in the sprite pixel
//! shader (see [`DeviceContext::draw_sprite_batch`]). Nearest is byte-exact on the native sprite
//! captures; linear keeps a ±1 residue on some (`sprite-syn-i1-*`).
//!
//! The other extras (gradient mesh, ink, image sources, SVG, LUT3D, …) are blocked, not exact yet.

// Image sources, BlendImage, LUT3D, GDI metafiles and print control (own files).
pub mod blend_image;
pub mod image_source;
pub mod lut3d;
pub mod ink;
pub mod metafile;
pub mod print;
pub mod mesh;

use crate::bitmap::Bitmap;
use crate::context::DeviceContext;

/// One sprite of an `ID2D1SpriteBatch`: destination rect (DIPs), optional source rect (texels,
/// default the whole bitmap) and optional straight colour (default opaque white).
#[derive(Clone, Copy)]
pub struct Sprite {
    pub dest: [f32; 4],
    pub source: Option<[f32; 4]>,
    pub color: Option<[f32; 4]>,
    /// Optional per-sprite transform (a 2×3 affine) applied to the destination rectangle before the
    /// context transform — native `ID2D1SpriteBatch`'s `transforms` array. `None` = identity.
    pub transform: Option<[f32; 6]>,
}

/// Compose two 2×3 affines: `apply(sprite_compose(a, b), p) == apply(a, apply(b, p))` (matches
/// `context::compose` / `metafile::compose6`). Used to place a sprite's transform under the context's.
fn sprite_compose(a: [f32; 6], b: [f32; 6]) -> [f32; 6] {
    [
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
        a[0] * b[4] + a[2] * b[5] + a[4],
        a[1] * b[4] + a[3] * b[5] + a[5],
    ]
}

/// `ID2D1SpriteBatch`: the parallel per-sprite arrays, kept as one struct-of-fields vec. Immutable
/// here beyond the initial fill — `AddSprites`/`SetSprites` would push/replace entries.
pub struct SpriteBatch {
    pub sprites: Vec<Sprite>,
}

impl DeviceContext {
    /// `CreateSpriteBatch` + `AddSprites`: an empty batch seeded with `sprites`.
    pub fn create_sprite_batch(&self, sprites: Vec<Sprite>) -> SpriteBatch {
        SpriteBatch { sprites }
    }

    /// `DrawSpriteBatch`. Native's `DrawSpriteBatch`
    /// draws the batch as D3D primitives: a sprite-batch tessellator emits each sprite's
    /// destination quad (sprite transform, then the context transform), the rasterizer is the
    /// aliased one `DrawSpriteBatch` requires, and the sprite pixel shader samples the bitmap at the
    /// interpolated texcoord `source·(1/size)` (point or linear sampler) times the sprite colour. So a
    /// sprite edge is WARP's pixel-centre top-left coverage, not `DrawBitmap`'s antialiased edge.
    pub fn draw_sprite_batch(&mut self, batch: &SpriteBatch, bitmap: &Bitmap, interpolation: u32, options: u32) {
        let _a = self.api(48, 0, None);
        // A command list keeps the batch and draws it at playback, as Direct2D's does: under the playback transform,
        // onto the playback target.
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            let (sprites, bitmap) = (std::rc::Rc::new(batch.sprites.clone()), std::rc::Rc::new(bitmap.clone()));
            self.record(crate::command::RecCmd::SpriteBatch { state, dpi, sprites, bitmap, interpolation, options });
            return;
        }
        // An A8 (alpha-only) sprite source is NOT a colour-format reject (unlike DrawBitmap/DrawImage,
        // which latch WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT): native samples the alpha byte broadcast to
        // all channels, (a,a,a,a), so an untinted sprite is invisible and a tinted one is tint·a
        // (VM-probed direct and command-list-recorded; both S_OK, no latch). Broadcast up front so the
        // float/command-list DrawBitmap fallback below also treats it as a colour source, not a reject.
        let a8;
        let bitmap = if bitmap.format.0 == 65 { a8 = bitmap.a8_as_bgra(); &a8 } else { bitmap };
        let highp = crate::float::is_highp(bitmap.format) || self.target.as_ref().is_some_and(|t| crate::float::is_highp(t.desc.format));
        if highp {
            return self.draw_sprites_as_bitmaps(batch, bitmap, interpolation, options);
        }
        if !self.ready_to_draw() {
            return;
        }
        // Present an R8G8B8A8 sprite source to the BGRA8 sampler with R/B swapped (A8 is already BGRA).
        let sampled = bitmap.as_sampled_bgra8();
        let bitmap = sampled.as_ref();
        let m = self.device_transform().map(|v| v as f32);
        let blend = crate::context::blend_of(self.get_primitive_blend());
        let tex = crate::sampling::Texture { data: &bitmap.premul_bgra, width: bitmap.width, height: bitmap.height };
        let (bw, bh) = (bitmap.width as f32, bitmap.height as f32);
        let inv = [1.0 / bw, 1.0 / bh];
        let clamp = options & 1 != 0; // D2D1_SPRITE_OPTIONS_CLAMP_TO_SOURCE_RECTANGLE
        // One sprite PS per batch: once any sprite carries a colour (D2DSpriteBatch::HasColors) every
        // sprite takes the coloured PS, white where it has none.
        let colored = batch.sprites.iter().any(|s| s.color.is_some());
        // Triangles of sprites with a non-finite corner, for WARP's rasterizer-fault bands after the loop.
        let mut faulting: Vec<[[f32; 3]; 3]> = Vec::new();
        let Some(buf) = &mut self.target else { return };
        for s in &batch.sprites {
            // AddSprites clamps the integer source rect to the bitmap.
            let src = s.source.unwrap_or([0.0, 0.0, bw, bh]);
            let src = [src[0].min(bw), src[1].min(bh), src[2].min(bw), src[3].min(bh)];
            // Sprite VS (shader dumps): the unit-quad corner `v0·(r−l) + l`; a batch with transforms
            // applies the sprite's own matrix first (`dp3((x, y, 1), (m11, m21, dx))`, identity for a
            // sprite without one), then the context matrix `x·A + y·C + E` from the constant buffer.
            // Texcoords `(v0·(src.r−src.l) + src.l)·(1/size)`, clamp bounds `(src ± 0.5)·(1/size)`.
            let d = s.dest;
            let (dw, dh) = (d[2] - d[0], d[3] - d[1]);
            let corner = |ux: f32, uy: f32| {
                let (mut x, mut y) = (ux * dw + d[0], uy * dh + d[1]);
                if let Some(t) = s.transform {
                    (x, y) = ((x * t[0] + y * t[2]) + t[4], (x * t[1] + y * t[3]) + t[5]);
                }
                [(x * m[0] + y * m[2]) + m[4], (x * m[1] + y * m[3]) + m[5]]
            };
            let unit = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
            let pos = unit.map(|u| corner(u[0], u[1]));
            // A destination corner that overflows f32 in DEVICE space (the sprite transform and the
            // context/DPI matrix applied) leaves nothing finite to place: a 3.4e38 edge draws at 96 DPI but
            // not at 120 or 192, where the DPI scale pushes it past FLT_MAX. Usually the sprite is
            // simply dropped, but an overflowing right or bottom edge can instead wipe the whole target
            // ([`overflow_wipes`]); under pushed clips and layers their bounds (sorted, intersected with the target)
            // take the viewport's place for the sprite's other axis and for the finite edge's low side.
            if pos.iter().any(|p| !p[0].is_finite() || !p[1].is_finite()) {
                // Its triangles, (TL, TR, BR) and (BL, TL, BR), still reach WARP's setup, where a non-finite corner
                // can fault the frame's batch (r22 27911: the bands it faults read back transparent, wiping a
                // later FillOpacityMask there).
                for ids in [[0usize, 1, 3], [2, 0, 3]] {
                    faulting.push(ids.map(|i| [pos[i][0], pos[i][1], 1.0]));
                }
                let (w, h) = (buf.desc.width as f32, buf.desc.height as f32);
                let mut s = [0.0f32, 0.0, w, h];
                for f in self.stack.iter() {
                    let r = match f { crate::layer::Frame::Clip { rect, .. } => *rect, crate::layer::Frame::Layer { params, .. } => params.bounds };
                    s = [s[0].max(r[0].min(r[2])), s[1].max(r[1].min(r[3])), s[2].min(r[0].max(r[2])), s[3].min(r[1].max(r[3]))];
                }
                if overflow_wipes(pos, [w, h], s) {
                    buf.data.fill(0);
                    for f in self.stack.iter_mut() {
                        let (crate::layer::Frame::Clip { saved, .. } | crate::layer::Frame::Layer { saved, .. }) = f;
                        saved.data.fill(0);
                    }
                }
                continue;
            }
            let uv = unit.map(|u| [(u[0] * (src[2] - src[0]) + src[0]) * inv[0], (u[1] * (src[3] - src[1]) + src[1]) * inv[1]]);
            let lo = [(src[0] + 0.5) * inv[0], (src[1] + 0.5) * inv[1]];
            let hi = [(src[2] - 0.5) * inv[0], (src[3] - 0.5) * inv[1]];
            let c = s.color.unwrap_or([1.0; 4]);
            let tint = [c[2] * c[3], c[1] * c[3], c[0] * c[3], c[3]]; // premultiplied, BGRA order
            // Quad split on the 0–3 diagonal (top-left to bottom-right): measured best against
            // native's sprite tessellation on the linear-sampled captures.
            // WARP frustum-clips a triangle with a corner outside the viewport (the target) before setting up its planes,
            // through the clipper a mesh's triangles take ([`crate::warp::clip_triangle`]): only the planes a corner lies
            // outside, each cut from the vertex inside it, the later planes' distances interpolated along with the position
            // rather than recomputed from it (VM sweep of 396 singular-world sprites around r24 29845, far edges to 3e9 px,
            // every sign and corner order: all exact, where recomputing the distances collapsed the far corners elsewhere).
            let viewport = [0.0, 0.0, buf.desc.width as f32, buf.desc.height as f32];
            let outside = pos.iter().any(|p| p[0] < viewport[0] || p[0] > viewport[2] || p[1] < viewport[1] || p[1] > viewport[3]);
            // The index buffer draws (TL, TR, BR) and (BL, TL, BR); a clipped fan keeps that vertex order,
            // while an unclipped triangle's setup starts at TL.
            let mut tris = Vec::new();
            for (tri, unclipped) in [([0usize, 1, 3], [0usize, 1, 3]), ([2, 0, 3], [0, 3, 2])] {
                let corner = |ids: [usize; 3]| ids.map(|i| [pos[i][0], pos[i][1], uv[i][0], uv[i][1]]);
                if !outside {
                    tris.push(corner(unclipped));
                    continue;
                }
                match crate::warp::clip_triangle(corner(tri), viewport) {
                    crate::warp::Clipped::Rejected => {}
                    crate::warp::Clipped::Inside => tris.push(corner(unclipped)),
                    crate::warp::Clipped::Fan(f) => tris.extend(f),
                }
            }
            for t in tris {
                let tpos = t.map(|v| [v[0], v[1]]);
                let (p, _) = crate::warp::snap_tri(tpos);
                let (sx, sy) = (p.map(|q| q[0]), p.map(|q| q[1]));
                let pu = crate::warp::plane::setup_triangle_attr(sx, sy, t.map(|v| v[2]));
                let pv = crate::warp::plane::setup_triangle_attr(sx, sy, t.map(|v| v[3]));
                crate::warp::fill_triangle_brush_cov(buf, tpos, [1.0; 3], blend, |px, py| {
                    // WARP's sprite pixel kernel (jit-dump of the linear sprite PS): the plane is
                    // evaluated `px·dadx + (py·dady + c0)`, texcoords clamp to ±10 (NaN → 0).
                    let ev = |p: [f32; 3]| px * p[0] + (py * p[1] + p[2]);
                    let fix = |t: f32| if t.is_nan() { 0.0 } else { t.clamp(-10.0, 10.0) };
                    let (mut u, mut v) = (fix(ev(pu)), fix(ev(pv)));
                    if clamp {
                        (u, v) = (u.max(lo[0]).min(hi[0]), v.max(lo[1]).min(hi[1]));
                    }
                    let ex = [crate::sampling::Extend::Clamp; 2];
                    if interpolation == 0 {
                        let t = tex.sample(crate::sampling::Filter::Nearest, ex, u * bw, v * bh);
                        return core::array::from_fn(|k| t[k] * tint[k] * 255.0);
                    }
                    // The Q8 texel/weight (jit-dumps of the sprite kernel on 15 texture sizes): x of a
                    // power-of-two-wide texture, and y of one that is a power of two on both axes, take
                    // `floor((t·n − 0.5)·256)` and clamp texels after; every other axis clamps to [0, 1]
                    // first, then `rne(t·(256·n)) − 128` (a 5×8 or 3×4 texture's y included).
                    // Then the `value·128` bilinear; a colourless batch's PS stores it truncated
                    // (`>> 7`), a coloured one multiplies the float sample and rounds.
                    let pow2 = [bitmap.width.is_power_of_two(), bitmap.width.is_power_of_two() && bitmap.height.is_power_of_two()];
                    let q8 = |t: f32, n: u32, x_axis: bool| {
                        let q = if pow2[usize::from(!x_axis)] {
                            ((t * n as f32 + -0.5) * 256.0).floor() as i32
                        } else {
                            (t.clamp(0.0, 1.0) * (256.0 * n as f32)).round_ties_even() as i32 - 128
                        };
                        (q >> 8, q & 0xff)
                    };
                    let (ax, ay) = (q8(u, bitmap.width, true), q8(v, bitmap.height, false));
                    let w: [i32; 4] = core::array::from_fn(|ch| tex.bilinear_v128(ex, ch, ax, ay));
                    if !colored {
                        w.map(|w| (w >> 7) as f32)
                    } else {
                        core::array::from_fn(|k| crate::sampling::norm_v128(w[k]) * tint[k] * 255.0)
                    }
                });
            }
        }
        for t in faulting {
            self.note_fault_triangle(t);
        }
    }

    /// The `DrawBitmap`-per-sprite fallback for float bitmaps and targets.
    fn draw_sprites_as_bitmaps(&mut self, batch: &SpriteBatch, bitmap: &Bitmap, interpolation: u32, _options: u32) {
        let whole = [0.0, 0.0, bitmap.width as f32, bitmap.height as f32];
        let saved = self.get_transform();
        for s in &batch.sprites {
            let source = s.source.unwrap_or(whole);
            let opacity = s.color.map_or(1.0, |c| c[3]);
            let has_xf = s.transform.is_some();
            if let Some(xf) = s.transform {
                self.set_transform(sprite_compose(saved, xf));
            }
            // Same device-space overflow rule the WARP sprite path uses: native skips a sprite whose
            // destination corner is non-finite once the sprite transform and the context/DPI matrix are
            // applied. The high-precision-target and command-list sprite paths realize per sprite here,
            // so guard them too (VM-probed: a device-overflow sprite voids on a float target as it does
            // on 8bpc). The transform set above is in effect, so `device_transform` already folds it in.
            let m = self.device_transform().map(|v| v as f32);
            let d = s.dest;
            let finite = [[d[0], d[1]], [d[2], d[1]], [d[0], d[3]], [d[2], d[3]]]
                .iter()
                .all(|&[x, y]| ((x * m[0] + y * m[2]) + m[4]).is_finite() && ((x * m[1] + y * m[3]) + m[5]).is_finite());
            if finite {
                self.draw_bitmap(bitmap, s.dest, source, opacity, interpolation);
            }
            if has_xf {
                self.set_transform(saved);
            }
        }
    }
}

/// Whether a sprite whose device corners `pos` (unit-quad order: (0,0), (1,0), (0,1), (1,1)) overflow wipes a `size`
/// target, under pushed clips and layers whose bounds intersect to `scissor` (the whole target when none). The finite
/// edge's low side and the other axis's pixel centres are taken in `scissor`, the finite edge's high side in the target
/// (sc230/sc241: a clip below a bottom-overflowing sprite's top still wipes; one beside it does not; inverted clips are
/// sorted; a clip inverted in x with a −inf top, whose device rect is empty, keeps it: sc242, r19 24487). VM probes
/// (sc217-sc222, r11 16908) show this is the outcome of WARP's clipping of the overflowed quad. The sprite turns the
/// WHOLE target and every saved clip/layer buffer transparent, whatever the blend, colour or clamp option. It does so
/// only when:
/// - its right edge is ±inf and its left edge l lies in (1.5, width]; or its bottom edge is ±inf and its top t lies in
///   (0.5, height]; and
/// - its extent across the other axis covers a pixel centre (top-left rule).
///
/// An infinite left or top edge, or a finite edge outside those ranges, drops the sprite instead.
fn overflow_wipes(pos: [[f32; 2]; 4], size: [f32; 2], scissor: [f32; 4]) -> bool {
    let fin = |p: [f32; 2]| p[0].is_finite() && p[1].is_finite();
    // Some pixel centre j + 0.5 inside [lo, hi) lies in [min(a, b), max(a, b)).
    let covers = |a: f32, b: f32, lo: f32, hi: f32| {
        let j = (a.min(b).max(lo) - 0.5).ceil();
        j + 0.5 < hi && j + 0.5 < a.max(b)
    };
    let [w, h] = size;
    let [sl, st, sr, sb] = scissor;
    if fin(pos[0]) && fin(pos[2]) && pos[1][1].is_finite() && pos[3][1].is_finite() && !pos[1][0].is_finite() && !pos[3][0].is_finite() {
        let l = pos[0][0];
        return sl + 1.5 < l && l <= w && covers(pos[0][1], pos[2][1], st, sb);
    }
    if fin(pos[0]) && fin(pos[1]) && pos[2][0].is_finite() && pos[3][0].is_finite() && !pos[2][1].is_finite() && !pos[3][1].is_finite() {
        let t = pos[0][1];
        return st + 0.5 < t && t <= h && covers(pos[0][0], pos[1][0], sl, sr);
    }
    false
}

#[cfg(test)]
mod overflow_wipe_tests {
    use super::overflow_wipes;
    const I: f32 = f32::INFINITY;
    const V: [f32; 4] = [0.0, 0.0, 64.0, 32.0];

    // A 64x32 target; x edges at the VM thresholds, y extent [8, 24] (sc220-sc222).
    #[test]
    fn right_edge_thresholds() {
        let q = |l: f32, r: f32| [[l, 8.0], [r, 8.0], [l, 24.0], [r, 24.0]];
        for (l, want) in [(1.5, false), (1.515625, true), (20.0, true), (64.0, true), (64.015625, false), (-20.0, false)] {
            assert_eq!(overflow_wipes(q(l, -I), [64.0, 32.0], V), want, "l={l} r=-inf");
            assert_eq!(overflow_wipes(q(l, I), [64.0, 32.0], V), want, "l={l} r=+inf");
        }
        // An infinite left edge never wipes.
        assert!(!overflow_wipes([[-I, 8.0], [20.0, 8.0], [-I, 24.0], [20.0, 24.0]], [64.0, 32.0], V));
    }

    #[test]
    fn bottom_edge_and_other_axis() {
        let q = |t: f32, b: f32| [[8.0, t], [40.0, t], [8.0, b], [40.0, b]];
        for (t, want) in [(0.5, false), (0.515625, true), (32.0, true), (32.015625, false)] {
            assert_eq!(overflow_wipes(q(t, -I), [64.0, 32.0], V), want, "t={t}");
        }
        // The other axis must cover a pixel centre: y [-8, 1] does, [-8, 0.5] and [32, 40] do not.
        let x = |a: f32, b: f32| [[20.0, a], [-I, a], [20.0, b], [-I, b]];
        assert!(overflow_wipes(x(-8.0, 1.0), [64.0, 32.0], V));
        assert!(!overflow_wipes(x(-8.0, 0.5), [64.0, 32.0], V));
        assert!(overflow_wipes(x(31.5, 40.0), [64.0, 32.0], V));
        assert!(!overflow_wipes(x(32.0, 40.0), [64.0, 32.0], V));
    }

    // Under a clip (sc241): a bottom-overflowing sprite (top 8, x 8..40) wipes under a clip whose rows end above its
    // top, but not under one beside it; a right-overflowing one (left 20) does not wipe under a clip left of it.
    #[test]
    fn under_a_clip() {
        let y = [[8.0, 8.0], [40.0, 8.0], [8.0, I], [40.0, I]];
        assert!(overflow_wipes(y, [64.0, 32.0], [4.6, 0.0, 21.2, 0.8]));
        assert!(!overflow_wipes(y, [64.0, 32.0], [48.0, 4.0, 60.0, 20.0]));
        let x = [[20.0, 8.0], [-I, 8.0], [20.0, 24.0], [-I, 24.0]];
        assert!(!overflow_wipes(x, [64.0, 32.0], [48.0, 4.0, 60.0, 20.0]));
        assert!(!overflow_wipes(x, [64.0, 32.0], [4.6, 0.0, 21.2, 0.8]));
        assert!(overflow_wipes(x, [64.0, 32.0], [4.0, 4.0, 20.0, 20.0]));
    }
}
