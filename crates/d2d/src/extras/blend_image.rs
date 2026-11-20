//! `ID2D1DeviceContext6::BlendImage` (D2D 1.3): draw an image over the current target with one of
//! the 26 `D2D1_BLEND_MODE`s instead of the current primitive blend.
//!
//! Native draws it as a DrawImage that carries the blend mode in place of a composite mode: modes 0..=25 are
//! valid, and the image is placed and resampled as DrawImage places it (world transform, image rectangle,
//! interpolation). Each pixel the image covers is then the W3C separable or non-separable blend `B(Cb,Cs)` on
//! the unpremultiplied colours, composited source-over. That math is the Blend effect's
//! (`effects::builtin::blend`), byte-exact for all 26 modes, reused here. Only Dissolve (mode 4) is left out:
//! its noise coordinate origin under `BlendImage` is unconfirmed (see dissolve-native-noise).

use crate::bitmap::Bitmap;
use crate::context::DeviceContext;
use d2d_types::contracts::PixelBuffer;

impl DeviceContext {
    /// `BlendImage` (§2.8): composite `image` over the current target at `offset` (DIPs) with a
    /// `D2D1_BLEND_MODE`. Native implements this as `DrawImage(... | blendFlag)`: the image is placed and resampled
    /// as DrawImage places it (the world transform, the image rectangle `rect` with its top-left at `offset`, the
    /// interpolation), and each pixel it covers is the W3C `B(Cb,Cs)` on the unpremultiplied colours, then
    /// source-over compositing.
    pub fn blend_image(&mut self, image: &Bitmap, mode: u32, offset: [f32; 2], rect: Option<[f32; 4]>, interpolation: u32) {
        let _a = self.api(0, 0, None); // ponytail: BlendImage's own command is not counted
        let Some(saved) = self.take_target() else { return };
        let mut inter = saved.clone();
        inter.data.fill(0);
        self.set_target_bitmap(inter);
        self.blending = true;
        // On DrawImage's whole-pixel fast path (unit device scale, an origin within 1/32 px of a pixel corner, as d2d1 tests
        // it) the blend effect reads the bitmap itself, 1:1 at the rounded origin, over the bitmap's own pixel
        // rect there (r27 32338: an offset of (14.02, 21.01) blends a 4x10 quad at (14, 21), DDI dump).
        let fast = if rect.is_none() { self.image_fast_origin(image, offset, interpolation) } else { None };
        if fast.is_some() || interpolation == 0 {
            // NEAREST has no filter reach, so the slow-path list below picks the texels the direct SOURCE_OVER draw picks,
            // ties on a texel boundary under magnification included (VM: a 96-dpi bitmap at dpi 144 under SOURCE_OUT and
            // SOURCE_OVER, and as a BlendImage, picks the same texel at every tie, where our slow-path NEAREST resample
            // does not).
            self.draw_image_bitmap(image, offset, rect, 0, 0);
        } else {
            // Otherwise Direct2D records the image as a SOURCE_OVER DrawImage in the same slow-path command list a SOURCE_OUT
            // DrawImage builds, and blends what that list renders, so the resample is SOURCE_OUT's over a transparent
            // intermediate: its intermediate origin, slow-path DPI scale and samplers (r27 32983's LINEAR form, 32727's
            // whole-pixel CUBIC form).
            self.draw_image_bitmap(image, offset, rect, interpolation, 4);
        }
        self.blending = false;
        let Some(mut src) = self.take_target() else { return };
        // The blend covers only the pixels the placed image's device rect touches (rounded out), not the resampler's
        // reach past it as a DrawImage does (sc148/sc149: at 1.5x, 2x and 3x a centre within half a pixel of the rect
        // blends, one past that keeps the backdrop).
        let (w, h) = (image.width as f32 * 96.0 / image.dpi.0, image.height as f32 * 96.0 / image.dpi.1);
        let [l, t, r, b] = match rect {
            Some(q) => {
                let q = [q[0].min(q[2]), q[1].min(q[3]), q[0].max(q[2]), q[1].max(q[3])];
                [offset[0] + (q[0].max(0.0) - q[0]), offset[1] + (q[1].max(0.0) - q[1]), offset[0] + (q[2].min(w) - q[0]), offset[1] + (q[3].min(h) - q[1])]
            }
            None => [offset[0], offset[1], offset[0] + w, offset[1] + h],
        };
        let m = self.device_transform();
        let c = [[l, t], [r, t], [l, b], [r, b]].map(|[x, y]| [m[0] * x as f64 + m[2] * y as f64 + m[4], m[1] * x as f64 + m[3] * y as f64 + m[5]]);
        let dim = [image.width as f64, image.height as f64];
        let lo = |k: usize| fast.map_or_else(|| c.iter().map(|p| p[k]).fold(f64::INFINITY, f64::min).floor(), |o| o[k] as f64);
        let hi = |k: usize| fast.map_or_else(|| c.iter().map(|p| p[k]).fold(f64::NEG_INFINITY, f64::max).ceil(), |o| o[k] as f64 + dim[k]);
        let mut quad = None;
        if [lo(0), lo(1), hi(0), hi(1)].iter().all(|v| v.is_finite()) {
            let (sw, sh) = (src.desc.width as f64, src.desc.height as f64);
            let [x0, y0, x1, y1] = [lo(0).clamp(0.0, sw), lo(1).clamp(0.0, sh), hi(0).clamp(0.0, sw), hi(1).clamp(0.0, sh)].map(|v| v as usize);
            quad = Some([x0, y0, x1, y1]);
            for y in 0..src.desc.height as usize {
                for x in 0..src.desc.width as usize {
                    if !(x0..x1).contains(&x) || !(y0..y1).contains(&y) {
                        src.data[y * src.stride + x * 4..y * src.stride + x * 4 + 4].fill(0);
                    }
                }
            }
        }
        let mut buf = saved;
        match quad {
            Some(q) => blend_quad(&mut buf, &src, mode, q),
            None => blend_from(&mut buf, &src, mode),
        }
        self.set_target_bitmap(buf);
    }
}

/// Blend the resampled image `src` (a target-sized premultiplied buffer, transparent where the image does not reach)
/// into `buf`, pixel by pixel.
fn blend_from(buf: &mut PixelBuffer, src: &PixelBuffer, mode: u32) {
    let (w, h) = (buf.desc.width as usize, buf.desc.height as usize);
    for y in 0..h {
        for x in 0..w {
            let (si, di) = (y * src.stride + x * 4, y * buf.stride + x * 4);
            let s: [u8; 4] = src.data[si..si + 4].try_into().unwrap();
            if s == [0; 4] {
                continue;
            }
            let d: [u8; 4] = buf.data[di..di + 4].try_into().unwrap();
            buf.data[di..di + 4].copy_from_slice(&blend_px(s, d, mode, [x as f32 + 0.5, y as f32 + 0.5]));
        }
    }
}


/// The blend pass over the rounded-out rect `[x0, y0, x1, y1]`. Direct2D draws it as one quad (DDI dump: the vertex
/// list TR, TL, BR, BL, TL, BR, texcoords 1/512 per texel) over two textures placed at a 512-texel atlas's origin: the
/// resampled image and a copy of the destination. Each is read bilinearly (Q8 floor address, 8.7 sample) through the
/// triangle's own fitted texcoord plane, and the blend runs on those unquantized samples. The planes of (TR, TL, BR) are
/// fitted at TL and those of (BL, TL, BR) at BL: of all 36 vertex orders only those match 48 random VM probes (any
/// size, fractional offset, dpi 96 to 144, six blend modes), and the order of the other two vertices cannot change an
/// axis-aligned quad's plane. Fitted there, a u plane can land a hair under a texel centre and blend 1/256 of the left
/// neighbour in (r27 32727: ±1 scattered over the image, alpha 254 down an opaque image's first column).
fn blend_quad(buf: &mut PixelBuffer, src: &PixelBuffer, mode: u32, [x0, y0, x1, y1]: [usize; 4]) {
    if x0 >= x1 || y0 >= y1 {
        return;
    }
    let (rw, rh) = ((x1 - x0) as i32, (y1 - y0) as i32);
    let atlas = [rw, rh].map(|v| (v as u32).next_power_of_two().max(512) as f32);
    let dst = buf.clone();
    let texel = |b: &PixelBuffer, i: i32, j: i32, c: usize| -> i32 {
        let (i, j) = (i.clamp(0, atlas[0] as i32 - 1), j.clamp(0, atlas[1] as i32 - 1));
        if i < rw && j < rh { b.data[(y0 + j as usize) * b.stride + (x0 + i as usize) * 4 + c] as i32 } else { 0 }
    };
    let [l, t, r, b] = [x0, y0, x1, y1].map(|v| v as f32);
    let (u_end, v_end) = (rw as f32 / atlas[0], rh as f32 / atlas[1]);
    let corner = |x: f32, y: f32, u: f32, v: f32| ([x, y], [u, v]);
    let (tr, tl, br, bl) = (corner(r, t, u_end, 0.0), corner(l, t, 0.0, 0.0), corner(r, b, u_end, v_end), corner(l, b, 0.0, v_end));
    let (w, h) = (buf.desc.width as i64, buf.desc.height as i64);
    // Each triangle listed from the vertex its planes are fitted at.
    for tri in [[tl, br, tr], [bl, tl, br]] {
        let pos = tri.map(|v| v.0);
        let (p, area) = crate::warp::snap_tri(pos);
        let (sx, sy) = (p.map(|v| v[0]), p.map(|v| v[1]));
        let pu = crate::warp::plane::setup_triangle_attr(sx, sy, tri.map(|v| v.1[0]));
        let pv = crate::warp::plane::setup_triangle_attr(sx, sy, tri.map(|v| v.1[1]));
        crate::warp::cover_tri(p, area, w, h, |x, y| {
            let (xc, yc) = (x as f32 + 0.5, y as f32 + 0.5);
            let q = |pl: [f32; 3], n: f32| (((xc * pl[0] + (yc * pl[1] + pl[2])) * n) * 256.0 - 128.0).floor() as i32;
            let (qu, qv) = (q(pu, atlas[0]), q(pv, atlas[1]));
            let (tx, fx, ty, fy) = (qu >> 8, qu & 0xff, qv >> 8, qv & 0xff);
            let sample = |bf: &PixelBuffer| -> [f32; 4] {
                let ch = |c: usize| {
                    let s = texel(bf, tx, ty, c) * (256 - fx) * (256 - fy) + texel(bf, tx + 1, ty, c) * fx * (256 - fy)
                        + texel(bf, tx, ty + 1, c) * (256 - fx) * fy + texel(bf, tx + 1, ty + 1, c) * fx * fy;
                    (s >> 9) as f32 / (128.0 * 255.0)
                };
                [ch(2), ch(1), ch(0), ch(3)]
            };
            let (sf, df) = (sample(src), sample(&dst));
            let out = if sf[3] == 0.0 { core::array::from_fn(|c| if c == 3 { df[3] } else { df[c] + (1.0 - df[3]) * sf[c] }) }
                else { crate::effects::builtin::blend::blend(mode, df, sf, [xc, yc]) };
            let q8 = |v: f32| (v * 255.0).round_ties_even().clamp(0.0, 255.0) as u8;
            let i = y * buf.stride + x * 4;
            buf.data[i..i + 4].copy_from_slice(&[q8(out[2]), q8(out[1]), q8(out[0]), q8(out[3])]);
        });
    }
}

/// Blend one premultiplied BGRA8 source over one premultiplied BGRA8 backdrop, reusing the 2C Blend
/// effect's byte-exact W3C `B(Cb,Cs)` + source-over compositing (`effects::builtin::blend`) — the
/// same math native's `DrawImage(... | blendFlag)` runs. The target pixel is the backdrop (input 0),
/// the image the source (input 1); both are UNORM8 → f32 (`c/255`) and the result stores back with
/// `round_ties_even`, exactly as the effect does. `scene_pos` is the device pixel centre (only
/// Dissolve uses it).
fn blend_px(s: [u8; 4], d: [u8; 4], mode: u32, scene_pos: [f32; 2]) -> [u8; 4] {
    let to_f = |p: [u8; 4]| [p[2] as f32 / 255.0, p[1] as f32 / 255.0, p[0] as f32 / 255.0, p[3] as f32 / 255.0];
    let (df, sf) = (to_f(d), to_f(s));
    // A source with no alpha but colour (a HIGH_QUALITY_CUBIC resample's ringing past the image) still adds its stored
    // colour where the backdrop is transparent: dst + (1 − dst.a)·src, alpha kept (r27 32844; every mode alike).
    let out = if s[3] == 0 { core::array::from_fn(|c| if c == 3 { df[3] } else { df[c] + (1.0 - df[3]) * sf[c] }) }
        else { crate::effects::builtin::blend::blend(mode, df, sf, scene_pos) };
    let q = |v: f32| (v * 255.0).round_ties_even().clamp(0.0, 255.0) as u8;
    [q(out[2]), q(out[1]), q(out[0]), q(out[3])]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_min_max_exact() {
        // Opaque backdrop/source: darken = min, lighten = max, both byte-exact, alpha stays 255.
        let d = [40u8, 200, 100, 255];
        let s = [90u8, 50, 100, 255];
        assert_eq!(blend_px(s, d, 2, [0.5, 0.5]), [40, 50, 100, 255]); // DARKEN
        assert_eq!(blend_px(s, d, 3, [0.5, 0.5]), [90, 200, 100, 255]); // LIGHTEN
    }
}
