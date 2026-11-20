//! Phase 1A device context: the state machine shared by every render target. It owns the draw
//! state, the error/tag latch, the push/pop stack and the target, and forwards the in-scope draws
//! (Clear, FillRectangle, FillGeometry on line figures) to `crate::pixels`. Everything here follows
//! the method table, error/tag model and layer push/pop rules.

use crate::bitmap::{self, Bitmap, Composite};
use crate::brush::{self, Brush};
use crate::command::{CommandList, ListInner, RasterCmd, RecCmd, TextFg};
use crate::pixels::{premultiply, Aa, Blend, FillMode, Target};
use crate::resource::{Domain, SolidColorBrush};
use crate::state::*;
use d2d_types::abi::{D2D1_ALPHA_MODE_IGNORE, D2D1_ALPHA_MODE_PREMULTIPLIED, DXGI_FORMAT_B8G8R8A8_UNORM};
use d2d_types::contracts::{ImageDesc, PixelBuffer};
use std::cell::RefCell;
use std::rc::Rc;
use windows_core::HRESULT;

// 3×2 affine `[m11,m12,m21,m22,dx,dy]` in f64; `compose(a,b)` applies b then a. Coordinates are
// cast f32→f64→transform→f32 exactly as the rasterizer expects device-space input.
type Mat = [f64; 6];
/// The one-segment open figure `DrawLine` strokes when it is not a rectangle fill.
pub(crate) fn line_geometry(p0: [f32; 2], p1: [f32; 2]) -> crate::geometry::Geometry {
    use crate::geometry::{Figure, FillMode, Geometry, Segment};
    Geometry::Path {
        fill: FillMode::Alternate,
        figures: vec![Figure { start: p0, filled: false, closed: false, segments: vec![Segment::Line(p1)], seg_flags: Vec::new() }],
    }
}

fn apply(m: &Mat, x: f32, y: f32) -> [f32; 2] {
    let (x, y) = (x as f64, y as f64);
    [(m[0] * x + m[2] * y + m[4]) as f32, (m[1] * x + m[3] * y + m[5]) as f32]
}
/// A mesh vertex through `MILMatrix3x2::GetTransformFunction`'s point transform, as
/// `MeshRealization::SendGeometry` applies it: f32, with native's specialisations — identity,
/// translation only, translate-or-scale (no cross terms, so an infinite x never reaches y as `0·inf`),
/// else the general `(y·m21 + x·m11) + dx`.
fn transform_mesh_point(m: &Mat, x: f32, y: f32) -> [f32; 2] {
    let f: [f32; 6] = m.map(|v| v as f32);
    if f[0] == 1.0 && f[1] == 0.0 && f[2] == 0.0 && f[3] == 1.0 {
        if f[4] == 0.0 && f[5] == 0.0 {
            return [x, y];
        }
        return [x + f[4], y + f[5]];
    }
    if f[1] == 0.0 && f[2] == 0.0 {
        return [x * f[0] + f[4], y * f[3] + f[5]];
    }
    [(y * f[2] + x * f[0]) + f[4], (y * f[3] + x * f[1]) + f[5]]
}
fn compose(a: &Mat, b: &Mat) -> Mat {
    [
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
        a[0] * b[4] + a[2] * b[5] + a[4],
        a[1] * b[4] + a[3] * b[5] + a[5],
    ]
}

/// d2d1's device→texcoord map of a DrawImage'd bitmap, its VS rows `(x·t0 + y·t2) + t4` (bit-exact to
/// the traced vertex attributes): device→DIP by the inverse DPI scale `inv` (`96·(1/dpi)`, see
/// [`DeviceContext::image_dip`]), then the unit-image→DIP matrix (bitmap size folded in first, the
/// origin `o` subtracted from the translation) inverted with one reciprocal determinant. At dpi 120 a
/// 7-wide bitmap's u row is fl(1/7)·fl(96·fl(1/120)), not the product's inverse (npd120 VS JIT, l88).
fn image_texmap(world: [f32; 6], pd: [f32; 2], inv: (f32, f32), dims: [f32; 2], o: [f32; 2]) -> [f32; 6] {
    let (dx, dy) = inv;
    let (a, b, c, d) = (world[0] * dims[0], world[1] * dims[0], world[2] * dims[1], world[3] * dims[1]);
    let (e, f) = (pd[0] - o[0] * dx, pd[1] - o[1] * dy);
    let i = 1.0 / (a * d - b * c);
    [dx * (d * i), dx * (-b * i), dy * (-c * i), dy * (a * i), (c * f - d * e) * i, (b * e - a * f) * i]
}

/// [`hw_bitmap_row`]'s forward matrix, normalized texcoord → device: diag(size)·B, then ·W.
pub(crate) fn hw_bitmap_fwd(size: [f32; 2], b: [f32; 6], w: [f32; 6]) -> [f32; 6] {
    let (sw, sh, z) = (size[0], size[1], 0.0f32);
    let (m00, m01, m10, m11) = ((b[0] * sw) + (b[2] * z), (b[1] * sw) + (b[3] * z), (b[2] * sh) + (b[0] * z), (b[3] * sh) + (b[1] * z));
    let (m20, m21) = (((b[0] * z) + (b[2] * z)) + b[4], ((b[1] * z) + (b[3] * z)) + b[5]);
    let (a00, a01) = ((w[2] * m01) + (w[0] * m00), (w[3] * m01) + (w[1] * m00));
    let (a10, a11) = ((m11 * w[2]) + (m10 * w[0]), (m11 * w[3]) + (w[1] * m10));
    [a00, a01, a10, a11, ((m21 * w[2]) + (m20 * w[0])) + w[4], ((m21 * w[3]) + (m20 * w[1])) + w[5]]
}

/// `HwBitmapBrush::SetBrushAndContext`'s device→texcoord row, op for op: diag(size)·B, then ·W, then
/// the inverse by one reciprocal determinant.
pub(crate) fn hw_bitmap_row(size: [f32; 2], b: [f32; 6], w: [f32; 6]) -> [f32; 6] {
    let [a00, a01, a10, a11, a20, a21] = hw_bitmap_fwd(size, b, w);
    let i = 1.0 / ((a11 * a00) - (a10 * a01));
    [a11 * i, -a01 * i, -a10 * i, a00 * i, ((a10 * a21) - (a11 * a20)) * i, ((a01 * a20) - (a00 * a21)) * i]
}

/// The frame's shared 512x512 atlas of slow-path DrawImage intermediates. Direct2D packs each draw's rect (its device
/// bounds rounded out) on shelves with no gap, clears the slot and replays the draw into it unclipped, so a filter's
/// reach past the rect lands on whatever slot sits there; the composites read the atlas once the frame's batch runs.
#[derive(Default)]
struct SlowAtlas {
    cursor: [i32; 2],
    shelf_h: i32,
    slots: Vec<AtlasSlot>,
    texels: Vec<u8>,
}

/// One packed intermediate: its atlas origin, device rect, composite, and the target as it was before and after its
/// composite (to replay the composite when a later draw spills into the slot).
struct AtlasSlot {
    at: [i32; 2],
    rect: [i32; 4],
    composite: u32,
    reach: [f32; 4],
    before: Vec<u8>,
    after: Vec<u8>,
}

impl SlowAtlas {
    /// The slots for an ink's mask tiles, one each (the atlas origin for a tile that does not fit).
    fn tile_slots(&mut self, tiles: &[[f32; 4]]) -> Vec<[i32; 2]> {
        tiles.iter().map(|t| self.alloc((t[2] - t[0]) as i32, (t[3] - t[1]) as i32).unwrap_or([0; 2])).collect()
    }

    /// Where [`Self::alloc`] would place a `w` x `h` rect, without placing it.
    fn next(&self, w: i32, h: i32) -> Option<[i32; 2]> {
        let mut probe = SlowAtlas { cursor: self.cursor, shelf_h: self.shelf_h, slots: Vec::new(), texels: Vec::new() };
        probe.alloc(w, h)
    }

    /// The slot for a `w` x `h` rect, or `None` when it does not fit the atlas.
    fn alloc(&mut self, w: i32, h: i32) -> Option<[i32; 2]> {
        const SIZE: i32 = 512;
        if w <= 0 || h <= 0 || w > SIZE || h > SIZE {
            return None;
        }
        if w >= SIZE - self.cursor[0] {
            self.cursor = [0, self.cursor[1] + self.shelf_h];
            self.shelf_h = 0;
        }
        if h > SIZE - self.cursor[1] {
            return None;
        }
        let at = self.cursor;
        self.cursor[0] += w;
        self.shelf_h = self.shelf_h.max(h);
        Some(at)
    }
}

/// WARP's NEAREST blit of an axis-aligned bitmap brush with per-texel scale `s` and texel-0 corner `o`, in the pixels of
/// the surface it fills (the target for a direct draw, the atlas for a slow-path one). The image covers the pixels its
/// edge rule takes. WARP addresses them on its 16.16 blit iterator, or on a float plane when, on either axis, the first
/// or the last covered pixel centre lies within 1/32 texel inside the image's edge (JIT dumps of the blit kernels on 194
/// VM probe scenes, direct and slow-path draws alike: a far edge whose rounded f32 position falls on a pixel centre leaves
/// its last covered centre a whole pixel inside, and stays on the iterator). The plane is the
/// unit-triangle fit of the brush's texcoord rows (bitmap size folded in, the translation divided out), read through a
/// Q8 floor address; its coefficients match the ones read out of the running kernel.
struct NearestBlit {
    lo: [f32; 2],
    hi: [f32; 2],
    n: [f32; 2],
    plane: Option<[[f32; 3]; 2]>,
    step: [(f32, f32); 2],
}

impl NearestBlit {
    /// `k` is the fill's world under the surface's DPI scale per axis. `replay` is the slow-path draw into the atlas,
    /// whose plane takes the translation -o/(s*n); a direct draw's keeps the texcoord row's own (VM: the running
    /// kernel's coefficients on direct draws at dpi 96 and 120).
    fn new(s: [f32; 2], o: [f32; 2], n: [f32; 2], k: [f32; 2], replay: bool) -> Self {
        let lo = [0, 1].map(|c| (o[c] + 0.5).floor());
        let far = [0, 1].map(|c| o[c] + n[c] * s[c]);
        let hi = [0, 1].map(|c| (far[c] - 0.5).ceil());
        let near = (0..2).any(|c| {
            let s = s[c] as f64;
            let (first, last) = ((lo[c] as f64 + 0.5) - o[c] as f64, far[c] as f64 - (hi[c] as f64 - 0.5));
            (first < 1.0 && first / s < 1.0 / 32.0) || last / s < 1.0 / 32.0
        });
        let plane = near.then(|| {
            let mut row = hw_bitmap_row(n, [s[0], 0.0, 0.0, s[1], o[0], o[1]], [k[0], 0.0, 0.0, k[1], 0.0, 0.0]);
            if replay {
                row[4] = -o[0] / (s[0] * n[0]);
                row[5] = -o[1] / (s[1] * n[1]);
            }
            let tri = [(0.0f32, 0.0f32), (1.0, 0.0), (1.0, 1.0)].map(|(x, y)| [(x * row[0] + y * row[2]) + row[4], (x * row[1] + y * row[3]) + row[5]]);
            [0, 1].map(|c| crate::warp::plane::setup_triangle_attr([0, 256, 256], [0, 0, 256], tri.map(|t| t[c])))
        });
        let i = 1.0 / (s[0] * s[1]);
        let step = [(i * s[1], (-(s[1] * o[0])) * i), (i * s[0], (-(s[0] * o[1])) * i)];
        NearestBlit { lo, hi, n, plane, step }
    }

    /// The texel the surface pixel (`x`, `y`) reads, or `None` where the image does not cover it.
    fn texel(&self, x: i64, y: i64) -> Option<(i32, i32)> {
        let p = [x, y];
        if !(0..2).all(|c| (self.lo[c] as i64..self.hi[c] as i64).contains(&p[c])) {
            return None;
        }
        let n = self.n.map(|v| v as i32);
        let t: [i32; 2] = core::array::from_fn(|c| match self.plane {
            Some(planes) => {
                let (xc, yc) = (x as f32 + 0.5, y as f32 + 0.5);
                let pl = planes[c];
                (((((xc * pl[0] + (yc * pl[1] + pl[2])) * self.n[c]) * 256.0).floor() as i32) >> 8).clamp(0, n[c] - 1)
            }
            None => crate::sampling::dda_texel(self.step[c].0, self.step[c].1, self.lo[c], p[c] as f32 + 0.5),
        });
        (0..2).all(|c| (0..n[c]).contains(&t[c])).then_some((t[0], t[1]))
    }
}

// System32 WARP samplerprobe traces: DrawImage's direct kernels use a soft border and unit-triangle fit.
#[allow(clippy::too_many_arguments)]
fn bitmap_image_axis(bitmap: &Bitmap, matrix: [f32; 6], interpolation: u32, float_target: bool, plus: bool, over: bool, copy_prim: bool, texmap: [f32; 6], tex: [u32; 2]) -> impl Fn(f32, f32) -> [f32; 4] + '_ {
    let dims = [bitmap.width as f32, bitmap.height as f32];
    let scale = [matrix[0], matrix[3]];
    let offset = [matrix[4], matrix[5]];
    let pad = if interpolation == 0 { 0.0 } else if matches!(interpolation, 2 | 5) { 1.5 } else { 0.5 };
    // PLUS's textured rect covers MULTI_SAMPLE_LINEAR's whole tap reach (±0.375 px past the half-texel
    // LINEAR footprint); SOURCE_OVER's stops at the half texel (msl-c9-*/msl-c0-*, dimg-sc-plus-i3-*).
    let extra = if plus && interpolation == 3 { 0.375 } else { 0.0 };
    // A SOURCE_COPY under the COPY primitive blend reaches half a device pixel past the image, whatever the filter's reach
    // (VM geo/c2000c: LINEAR and CUBIC at dpi 96 and 120 leave the tail column past that transparent, while SOURCE_OVER
    // and PLUS under it keep the filter's reach; r3 2000).
    let rect = if copy_prim && interpolation != 0 {
        [(offset[0] - 0.5 + 0.5).floor(), (offset[1] - 0.5 + 0.5).floor(),
            (offset[0] + dims[0] * scale[0] + 0.5 - 0.5).ceil(), (offset[1] + dims[1] * scale[1] + 0.5 - 0.5).ceil()]
    } else {
        [(offset[0] - pad * scale[0] - extra + 0.5).floor(), (offset[1] - pad * scale[1] - extra + 0.5).floor(),
            (offset[0] + (dims[0] + pad) * scale[0] + extra - 0.5).ceil(), (offset[1] + (dims[1] + pad) * scale[1] + extra - 0.5).ceil()]
    };
    let mut brush = Brush::bitmap_source(bitmap.clone(), &matrix, [0, 0], interpolation, 1.0);
    if let Brush::Bitmap { extend, to_source, .. } = &mut brush {
        *extend = [crate::sampling::Extend::Transparent; 2];
        if interpolation == 1 && !crate::float::is_highp(bitmap.format) { *to_source = crate::warp::plane::warp_fit_affine(to_source); }
    }
    let highp = crate::float::is_highp(bitmap.format);
    // An integer device rect minifies on WARP's 16.16 DDA (the sampler sweeps: no triangle setup is
    // traced); a fractional one is a unit-triangle WARP draw (SetupTriangleInternal2 traces of 8×8 at a
    // fractional offset, scale 1 and ×0.6/×0.75), sampled from the fitted plane below.
    let whole_rect = (0..2).all(|c| offset[c].fract() == 0.0 && (offset[c] + dims[c] * scale[c]).fract() == 0.0);
    let fixed: [(i32, i32); 2] = core::array::from_fn(|c| crate::sampling::axis_fixed(0.0, dims[c], offset[c], offset[c] + dims[c] * scale[c]));
    // A texture whose width is not a power of two samples on a 16.16 DDA at any scale, the arithmetic of
    // AlphaBltAttributeSetup's fixed path: the device→texel matrix (`MILMatrix3x2::Invert`) at the rect's
    // first pixel centre and the step, each truncated to 16.16, plus LINEAR's −½-texel bias
    // (`+0x80 − 0x8000`). Byte-exact on r5 5930 (7×8 at dpi 120), 7×8 under ×1.25/×1.6/×2.5 and r5 5077's
    // 13×8 at a fractional offset and under ×0.6/×0.75 (traced as a unit-triangle WARP draw). A
    // power-of-two width with any height is the WARP draw below (lsc-2x13-*, r8 9499's 2×13), and so is a SOURCE_OVER
    // draw of a texture whose height is not a power of two either
    // (VM, 15 and 13 wide at dpi 120 and 144: heights 5, 6, 7, 10, 12, 15 and 24 sample the fitted plane exactly, 2, 4, 8
    // and 16 keep the iterator; found on r27 32983's 15×7). The slow-path composites keep the iterator (r14 19650: a
    // 3×5 SOURCE_OUT draw is exact on it, its SOURCE_OVER twin on the plane).
    let fixed_iter = interpolation == 1 && !highp && !float_target && !bitmap.width.is_power_of_two()
        && (bitmap.height.is_power_of_two() || !over);
    let blit: [(i32, i32); 2] = core::array::from_fn(|c| {
        let i = 1.0 / (scale[0] * scale[1]);
        let m0 = i * scale[1 - c];
        let m4 = -(scale[1 - c] * offset[c]) * i;
        let start = m0 * (rect[c] + 0.5) + m4;
        (((start * 65536.0) as i32).wrapping_add(0x80 - 0x8000), (m0 * 65536.0) as i32)
    });
    // LINEAR and CUBIC sample the fitted VS texcoord rows, CUBIC's phase `u·n`: at dpi 120 the factored
    // v row lands one ulp past a texel centre, where the device inverse lands on it (cubq-dpi120,
    // dimg-sc-*-i2-d120f).
    let fit = crate::warp::plane::warp_fit_affine(&texmap);
    let texd = tex.map(|v| v as f32);
    move |x, y| {
        if x < rect[0] || y < rect[1] || x >= rect[2] || y >= rect[3] { return [0.0; 4]; }
        if interpolation == 2 && !highp && !float_target {
            let tex = crate::sampling::Texture { data: &bitmap.premul_bgra, width: bitmap.width, height: bitmap.height };
            let u = [fit[0] * x + (fit[2] * y + fit[4]), fit[1] * x + (fit[3] * y + fit[5])];
            let [b, g, r, a] = tex.sample_hqcubic_in([crate::sampling::Extend::Transparent; 2], u, [u[0] * texd[0], u[1] * texd[1]], texd.map(|v| v as u32));
            return [r, g, b, a];
        }
        if !highp && !float_target && interpolation == 0 {
            // NEAREST addresses its texel on the sampler's 16.16 DDA from the rect's first pixel.
            let t: [i32; 2] = core::array::from_fn(|c| {
                let i = 1.0 / (scale[0] * scale[1]);
                crate::sampling::dda_texel(i * scale[1 - c], -(scale[1 - c] * offset[c]) * i, rect[c], [x, y][c])
            });
            return bitmap.texel_rgba(t[0], t[1]);
        }
        if fixed_iter {
            let q = |c: usize, p: f32| { let u = blit[c].0.wrapping_add(((p.floor() - rect[c]) as i32).wrapping_mul(blit[c].1)); (u >> 16, (u >> 8) & 255) };
            let texture = crate::sampling::Texture { data: &bitmap.premul_bgra, width: bitmap.width, height: bitmap.height };
            let (qx, qy) = (q(0, x), q(1, y));
            let [b, g, r, a]: [f32; 4] = core::array::from_fn(|ch| (crate::sampling::norm_v128(texture.bilinear_v128([crate::sampling::Extend::Transparent; 2], ch, qx, qy)) * 255.0).floor() / 255.0);
            return [r, g, b, a];
        }
        if !highp && !float_target && interpolation == 1 && scale[0] <= 1.0 && scale[1] <= 1.0 && whole_rect {
            let q = |c: usize, p: f32| fixed[c].0.wrapping_add((p.floor() as i32).wrapping_mul(fixed[c].1));
            let texture = crate::sampling::Texture { data: &bitmap.premul_bgra, width: bitmap.width, height: bitmap.height };
            let [b, g, r, a] = texture.sample_fixed([crate::sampling::Extend::Transparent; 2], q(0, x), q(1, y));
            return [r, g, b, a];
        }
        if interpolation == 1 && !highp && !float_target {
            // LINEAR off the fixed iterator is the unit-triangle WARP draw of the VS rows (the fit
            // above): WARP's Q8 address (a power-of-two dimension floors, any other rounds) and the
            // sample stored truncated, `>> 7` (lsc-* probes; lsc-8x8-d120 traced).
            let tex = crate::sampling::Texture { data: &bitmap.premul_bgra, width: bitmap.width, height: bitmap.height };
            let u = [fit[0] * x + (fit[2] * y + fit[4]), fit[1] * x + (fit[3] * y + fit[5])];
            let q = |t: f32, n: f32| {
                let a = (t.clamp(-10.0, 10.0) * n - 0.5) * 256.0;
                let q = if (n as u32).is_power_of_two() { a.floor() } else { a.round_ties_even() } as i32;
                (q >> 8, q & 0xff)
            };
            let (qx, qy) = (q(u[0], dims[0]), q(u[1], dims[1]));
            let [b, g, r, a]: [f32; 4] = core::array::from_fn(|ch| (tex.bilinear_v128([crate::sampling::Extend::Transparent; 2], ch, qx, qy) >> 7) as f32 / 255.0);
            return [r, g, b, a];
        }
        let [b, g, r, a] = if highp || float_target { brush.source_at_f_rect(x, y, rect) }
            else { brush.source_at_rect(x, y, true, rect).map(|v| if interpolation == 1 { v.floor() / 255.0 } else if interpolation == 2 { v * (1.0 / 255.0) } else { v / 255.0 }) };
        [r, g, b, a]
    }
}

// Native DrawImage mixed HQ uses a padded axis prefilter, then an affine cubic atlas pass.
fn bitmap_image_hq_mixed(bitmap: &Bitmap, matrix: [f32; 6], width: u32, height: u32, precision: u32) -> Bitmap {
    use crate::effects::builtin::transform as tfx;
    let axis = usize::from(matrix[3] < 1.0);
    let scale = matrix[axis * 3];
    let mut bounds = [0, 0, bitmap.width as i32, bitmap.height as i32];
    bounds[axis] = -2;
    bounds[axis + 2] = (bounds[axis + 2] as f32 * scale + 2.0).ceil() as i32;
    let precision = if precision == 0 { if crate::float::is_highp(bitmap.format) { 5 } else { 1 } } else { precision };
    let intermediate = tfx::hq_downscale_axis(&tfx::Input::new(bitmap, 0, 0), axis, scale, 0.0, 0.0, tfx::Border::Soft, bounds, precision);
    let input = tfx::Input::new(&intermediate, bounds[0], bounds[1]).in_atlas().with_texture_offset([-2; 2]);
    let input = if precision > 1 { input.with_float_output() } else { input };
    let mut remaining = matrix;
    remaining[axis * 3] = 1.0;
    let rect = [
        ((bounds[0] as f32 - 1.5) * remaining[0] + remaining[4] + 0.5).floor(),
        ((bounds[1] as f32 - 1.5) * remaining[3] + remaining[5] + 0.5).floor(),
        ((bounds[2] as f32 + 1.5) * remaining[0] + remaining[4] - 0.5).ceil(),
        ((bounds[3] as f32 + 1.5) * remaining[3] + remaining[5] - 0.5).ceil(),
    ];
    let dims = [intermediate.width, intermediate.height].map(|v| v.next_power_of_two().max(512));
    let texture_origin = [bounds[0] - 2, bounds[1] - 2];
    tfx::to_bitmap_precision(width, height, 0.0, 0.0, precision > 1, |x, y| {
        if x < rect[0] || y < rect[1] || x >= rect[2] || y >= rect[3] { return [0.0; 4]; }
        let planes = crate::brush::bitmap_quad_at(&remaining, dims, texture_origin, rect, x, y, [rect[0], rect[1]]);
        let uv = planes.map(|a| a[0] * (x - rect[0]) + (a[1] * (y - rect[1]) + a[2]));
        let phase = [(x - remaining[4]) / remaining[0], (y - remaining[5]) / remaining[3]];
        input.cubic(phase, uv, tfx::Border::Soft)
    })
}

fn aa_of(mode: u32) -> Aa {
    if mode == 1 { Aa::Aliased } else { Aa::PerPrimitive }
}
pub(crate) fn blend_of(mode: u32) -> Blend {
    match mode {
        1 => Blend::Copy,
        2 => Blend::Min,
        3 => Blend::Add,
        4 => Blend::Max,
        _ => Blend::SourceOver,
    }
}

/// A `D2D1_COMPOSITE_MODE` mapped to the `SetPrimitiveBlend` mode its direct draw uses: SOURCE_OVER(0)→
/// SOURCE_OVER, PLUS(9)→ADD(3), BOUNDED_SOURCE_COPY(11)→COPY(1). Every other composite realizes into a
/// transparent intermediate (whose internal draws are SOURCE_OVER) and composites that, so it maps to
/// SOURCE_OVER(0) too. Used to reset `state.blend` at the entry of a DrawImage/effect realize so a nested
/// sampled draw does not inherit the caller's outer `SetPrimitiveBlend` (r27 32253).
pub(crate) fn composite_blend(composite: u32) -> u32 {
    match composite {
        9 => 3,
        11 => 1,
        _ => 0,
    }
}

/// `ID2D1DeviceContext` / the `D2DDeviceContextBase` state engine. The bound bitmap stays owned in
/// `target`; `recording` is an overlay (`SetTarget(commandList)`) that diverts draws to a list while
/// leaving the bitmap bound to resume afterwards.
pub struct DeviceContext {
    domain: Domain,
    pub(crate) state: DrawState,
    dpi: (f32, f32),
    in_draw: bool,
    // Error latch: `error == S_OK` means no error yet; the first error freezes `err_tags`.
    error: HRESULT,
    err_tags: (u64, u64),
    pub(crate) stack: Vec<crate::layer::Frame>,
    pub(crate) target: Option<PixelBuffer>,
    pub(crate) recording: Option<Rc<RefCell<ListInner>>>,
    /// Saved outer `recording` targets so a NESTED command-list recording (a command_list op recorded
    /// inside another list's ops) restores the OUTER list — not `None` — when it stops. `SetTarget` is
    /// not a stack in D2D, but the drive/FFI record a nested list by record_into/stop_recording pairs,
    /// so stack the previous target here (fuzz r20: a nested list voided the main target otherwise).
    pub(crate) recording_saved: Vec<Option<Rc<RefCell<ListInner>>>>,
    /// `SetRenderingControls` bufferPrecision: the default intermediate precision for the effect graph
    /// (`D2D1_BUFFER_PRECISION`; 0 = UNKNOWN, the native default; 1 = 8bpc). It governs a realized effect
    /// intermediate's precision; UNKNOWN leaves each effect at its inputs' precision.
    pub(crate) buffer_precision: u32,
    /// `SetRenderingControls` tileSize (default 1024×1024): the tiles a layer pop walks. A pop that
    /// fails partway stops in its first tile (see [`crate::layer::layer_back`]).
    pub(crate) tile: [u32; 2],
    /// The 8-row bands where a FillMesh triangle faulted WARP's rasterizer this frame (empty: none;
    /// see [`mesh_fault_bands`]). WARP runs the frame's batch as one job per band, and a faulting job
    /// aborts the batch: every band whose job had not completed reads back as it was before the frame
    /// (transparent black) — whatever was drawn there before or after the mesh. Only the pixels: every
    /// op still validates and latches its errors as usual (r23 28091). Applied wherever the frame's
    /// pixels become observable ([`Self::wipe_if_poisoned`]); resets at `begin_draw`.
    pub(crate) mesh_bad: Vec<bool>,
    /// The bands FillMesh triangles reached this frame, in the order they first reached them. WARP's
    /// workers take those bands newest first (VM probes: a fill reaching a band before or after the
    /// mesh does not move it), so the bands that complete before a fault are the ones taken ahead of
    /// the first faulting band.
    mesh_bands: Vec<usize>,
    /// WARP's worker pool already ran before the frame's batch (a bitmap or gradient texture was
    /// uploaded on the device): its workers then start the band jobs together and a fault aborts every
    /// band, the ones taken ahead of it included (VM probes: a gradient or bitmap created anywhere before
    /// EndDraw, even after the mesh, loses the band an idle pool keeps).
    warp_warm: std::cell::Cell<bool>,
    /// Set while a command list replays into an image brush's (or effect input's) intermediate: errors a
    /// direct playback latches at the drawing op do not surface from there (r24 29528).
    realizing_list: bool,
    /// Set while a command list plays as one image (a DrawImage composite other than an inline SOURCE_OVER): an
    /// error its draws latch carries the playing op's tags rather than the recorded ones.
    playing_as_image: bool,
    /// Clips/layers left open in command lists that stopped recording: Direct2D keeps them on the
    /// context's stack, so EndDraw fails with PUSH_POP_UNBALANCED.
    pub(crate) rec_open: usize,
    /// Nesting depth of command-list playback ([`Self::replay`]).
    pub(crate) replay_depth: u32,
    /// The tags of the op playing the outermost command list back, while one plays.
    play_tags: Option<(u64, u64)>,
    /// A command list plays back at an infinite placement, which places nothing.
    play_unplaced: bool,
    /// A DrawImage fills its image rectangle with the bitmap as a LINEAR image brush, through the pixel-shader path.
    crop_fill: bool,
    /// The list playing back holds an HQ-cubic DrawBitmap and its recorded bounds are the whole plane (a Clear, or an
    /// image under a mode where transparent black matters), so Direct2D plays it even at a non-finite placement.
    play_unbounded: bool,
    /// An error Direct2D only finds once the frame's batch has run: EndDraw latches it under the frame's last tags.
    end_error: Option<HRESULT>,
    /// The DrawImage being placed, in its own terms: the DIP world, the DIP offset and the bitmap's DPI fold (96/dpi).
    dip_place: Option<([f32; 6], [f32; 2], [f32; 2])>,
    /// The frame's shared atlas of slow-path DrawImage intermediates.
    atlas: SlowAtlas,
    /// A BlendImage is drawing its image through DrawImage (`extras::blend_image`).
    pub(crate) blending: bool,
    /// The DrawImage source is an effect output: a texture in a power-of-two atlas of at least 512 texels.
    pub(crate) effect_atlas: bool,
    /// The atlas set Direct2D's deferred image context holds open for DrawImages of effects
    /// ([`crate::effects::plan::batch_enter`]). It closes where the target processes its queued operations: BeginDraw,
    /// EndDraw, Flush, SetTarget and a rendering-controls change.
    pub(crate) effect_batch: crate::effects::plan::AtlasSet,
    /// The deferred image context's arena: what the batch's DrawImages of effects hold ([`crate::effects::arena`]).
    pub(crate) effect_arena: crate::effects::arena::Arena,
    /// Direct2D's command batch and the target's queue of primitives: where the target renders ([`crate::batch`]).
    pub(crate) batch: crate::batch::Batch,
    /// How deep the API calls in progress are nested ([`Self::api`]).
    api_depth: std::rc::Rc<std::cell::Cell<u32>>,
    /// The DrawImage of an effect being made queued its deferred operation.
    pub(crate) image_queued: bool,
}

/// An API call in progress ([`DeviceContext::api`]); `.1`: it is the outermost one and recorded.
pub(crate) struct Api(std::rc::Rc<std::cell::Cell<u32>>, bool);

impl Drop for Api {
    fn drop(&mut self) {
        self.0.set(self.0.get() - 1);
    }
}

impl DeviceContext {
    pub(crate) fn new(domain: Domain) -> Self {
        Self {
            domain,
            state: DrawState::default(),
            dpi: (96.0, 96.0),
            in_draw: false,
            realizing_list: false,
            playing_as_image: false,
            error: S_OK,
            err_tags: (0, 0),
            stack: Vec::new(),
            target: None,
            recording: None,
            recording_saved: Vec::new(),
            buffer_precision: 0,
            tile: [1024, 1024],
            mesh_bad: Vec::new(),
            mesh_bands: Vec::new(),
            warp_warm: std::cell::Cell::new(false),
            rec_open: 0,
            replay_depth: 0,
            play_tags: None,
            play_unplaced: false,
            play_unbounded: false,
            end_error: None,
            dip_place: None,
            atlas: SlowAtlas::default(),
            crop_fill: false,
            blending: false,
            effect_atlas: false,
            effect_batch: Default::default(),
            effect_arena: Default::default(),
            batch: Default::default(),
            api_depth: Default::default(),
            image_queued: false,
        }
    }

    /// An API call: the outermost call of a draw with no error latched records its command of `bytes` (checking for
    /// `extra` bytes beyond the batch's headroom) and queues its primitive, as Direct2D's device context does; the
    /// calls our implementation makes on itself do not. Hold the result for the call.
    pub(crate) fn api(&mut self, bytes: u32, extra: u32, prim: Option<crate::batch::Prim>) -> Api {
        let depth = self.api_depth.clone();
        let live = depth.get() == 0 && self.in_draw && self.error == S_OK && self.recording.is_none();
        depth.set(depth.get() + 1);
        if live {
            if bytes > 0 {
                let f = self.batch.command(bytes, extra);
                self.rendered(f);
            }
            if let Some(p) = prim {
                let f = self.batch.prim(p);
                self.rendered(f);
            }
        }
        Api(depth, live)
    }

    /// What the target rendering its queue, or the command batch ending, does to the deferred image context.
    pub(crate) fn rendered(&mut self, f: crate::batch::Flush) {
        use crate::batch::Flush;
        if f != Flush::None && crate::effects::arena::trace() {
            eprintln!("batch {f:?} at op {}", self.state.tags.0);
        }
        match f {
            Flush::None => {}
            Flush::Target => {
                self.effect_batch.clear();
                self.effect_arena.flush();
            }
            // The DrawImage's own memory is not queued yet, so nothing is freed; the set it packed into closes.
            Flush::TargetInImage => self.effect_batch.clear(),
            Flush::Batch => {
                self.effect_batch.clear();
                self.effect_arena.reset();
                self.atlas = SlowAtlas::default();
            }
        }
    }

    /// A DrawImage of an effect's API call ([`Self::api`]); [`Self::image_queue`] ends it.
    pub(crate) fn image_api(&mut self) -> Api {
        let a = self.api(64, 0, None);
        if a.1 {
            self.image_queued = false;
        }
        a
    }

    /// The DrawImage of an effect made in `a` queues its deferred operation (once it has packed its atlas set).
    pub(crate) fn image_queue(&mut self, a: Api) {
        if a.1 && std::mem::take(&mut self.image_queued) {
            let f = self.batch.prim(crate::batch::Prim::Image);
            self.rendered(f);
        }
    }

    /// `SetTextRenderingParams` (`set`: with params, not null): the context records them in a draw; without, its next
    /// glyph run makes and records the default ones.
    pub fn note_text_rendering_params(&mut self, set: bool) {
        let drawing = self.in_draw && self.error == S_OK && self.recording.is_none() && self.api_depth.get() == 0;
        let f = self.batch.set_text_params(set, drawing);
        self.rendered(f);
    }

    /// A glyph run's API call ([`Self::api`]): its command (the default rendering params first if the context has none)
    /// with its brush copy of `brush` bytes and its glyph data, and its primitive.
    pub(crate) fn glyph_api(&mut self, run: &crate::text::GlyphRun, brush: u32) -> Api {
        let a = self.api(0, 0, None);
        if a.1 {
            let f = self.batch.glyph_run(brush, run.glyphs.len() as u32, run.glyphs.iter().any(|g| g.offset != [0.0, 0.0]));
            self.rendered(f);
            let f = self.batch.prim(crate::batch::Prim::Text);
            self.rendered(f);
        }
        a
    }

    /// `ID2D1DeviceContext::SetRenderingControls` bufferPrecision: the default precision D2D uses for
    /// the effect graph's intermediate buffers. Subsequent effect draws realize their intermediates at
    /// this precision; 0/UNKNOWN (the default) leaves each effect at its inputs' precision (native: an
    /// unset context blurs a float bitmap in float, keeping HDR — oracle `precision-05`).
    pub fn set_buffer_precision(&mut self, precision: u32) {
        // Stored per device context; pushed into the effect graph at each effect draw
        // (`set_render_precision`), so interleaved contexts on one thread never leak precision.
        self.buffer_precision = precision;
    }

    /// `SetRenderingControls`: Direct2D ignores the whole call (precision too) for an invalid
    /// precision (above 32BPC_FLOAT) or a zero tile dimension.
    pub fn set_rendering_controls(&mut self, precision: u32, tile: [u32; 2]) {
        if precision <= 5 && tile[0] != 0 && tile[1] != 0 {
            // The next DrawImage of an effect flushes the target's queue when the rendering controls changed.
            if (precision, tile) != (self.buffer_precision, self.tile) {
                self.batch.target_flushed();
                self.rendered(crate::batch::Flush::Target);
            }
            self.buffer_precision = precision;
            self.tile = tile;
        }
    }

    // --- target binding -----------------------------------------------------------------------

    /// Bind a premultiplied BGRA8 bitmap as the target. Returns the previously bound bitmap, if any.
    pub fn set_target_bitmap(&mut self, buf: PixelBuffer) -> Option<PixelBuffer> {
        self.wipe_if_poisoned();
        self.effect_batch.clear();
        self.effect_arena.reset();
        self.batch.end();
        self.target.replace(buf)
    }

    /// `SetTarget(commandList)`: divert subsequent draws into the list (the bitmap stays bound). Stacks
    /// the previous recording target so a nested list recorded inside this one restores THIS list, not
    /// the bitmap, when it stops.
    pub fn record_into(&mut self, list: &CommandList) {
        self.effect_batch.clear();
        self.effect_arena.reset();
        self.batch.end();
        // Leaving the bitmap target for a list inside a zero-area layer turns that layer's corner pixel live
        // (`LayerParams::corner`): what was drawn there before stays clipped out.
        if self.recording.is_none() {
            if let Some(t) = self.target.as_mut() {
                for f in self.stack.iter_mut() {
                    if let crate::layer::Frame::Layer { saved, params } = f {
                        if let (Some((x, y)), false) = (params.corner, params.corner_live) {
                            params.corner_live = true;
                            crate::layer::copy_pixel(t, saved, x, y);
                        }
                    }
                }
            }
        }
        self.recording_saved.push(self.recording.take());
        self.recording = Some(list.0.clone());
    }

    /// Stop recording and resume drawing into the enclosing target — the outer command list when this
    /// was a nested recording, else the bound bitmap (`SetTarget(bitmap)` after a list).
    pub fn stop_recording(&mut self) {
        self.effect_batch.clear();
        self.effect_arena.reset();
        // ponytail: the target set again queues its entry (+1) as at BeginDraw; not counted.
        self.batch.end();
        if let Some(list) = &self.recording {
            self.rec_open += list.borrow().frames.len();
        }
        self.recording = self.recording_saved.pop().flatten();
    }

    /// Unbind the target, returning the bound bitmap if there was one (for pixel readback).
    pub fn take_target(&mut self) -> Option<PixelBuffer> {
        self.target.take()
    }

    /// Borrow the bound target bitmap's pixels (for tests/readback mid-session).
    pub fn target_bitmap(&self) -> Option<&PixelBuffer> {
        self.target.as_ref()
    }

    // --- resource creation --------------------------------------------------------------------

    /// `CreateSolidColorBrush`. `opacity` defaults to 1.0 when `None`.
    pub fn create_solid_color_brush(&self, color: [f32; 4], opacity: Option<f32>) -> SolidColorBrush {
        SolidColorBrush { domain: self.domain, color, opacity: opacity.unwrap_or(1.0) }
    }

    /// `CreateLinearGradientBrush`. The gradient texture is realized against the current device
    /// transform composed with the optional brush transform.
    // ponytail: realize at create with the current transform; native realizes lazily per draw —
    // identical unless the transform changes between create and fill (not exercised).
    #[allow(clippy::too_many_arguments)]
    pub fn create_linear_gradient_brush(&self, start: [f32; 2], end: [f32; 2], stops: &[(f32, [f32; 4])], gamma: u32, extend: u32, opacity: f32, transform: Option<[f32; 6]>) -> PixelBrush {
        let xform = transform.unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        self.note_texture_upload();
        PixelBrush {
            domain: self.domain,
            brush: Brush::linear(stops, gamma, extend, opacity, start, end, &self.brush_to_device(transform)),
            def: Some(crate::brush::BrushDef::Linear { stops: stops.to_vec(), gamma, extend, opacity, start, end, xform }),
            image: None,
            crop: None,
            fail: None,
            batch: 56,
            rows: Some(2 + (opacity != 1.0) as u32),
        }
    }

    /// `CreateRadialGradientBrush`.
    #[allow(clippy::too_many_arguments)]
    pub fn create_radial_gradient_brush(&self, center: [f32; 2], offset: [f32; 2], radius: [f32; 2], stops: &[(f32, [f32; 4])], gamma: u32, extend: u32, opacity: f32, transform: Option<[f32; 6]>) -> PixelBrush {
        let xform = transform.unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        self.note_texture_upload();
        PixelBrush {
            domain: self.domain,
            brush: Brush::radial_world(stops, gamma, extend, opacity, center, radius, offset, &self.brush_to_device(transform), &self.device_transform().map(|v| v as f32), &xform),
            def: Some(crate::brush::BrushDef::Radial { stops: stops.to_vec(), gamma, extend, opacity, center, radii: radius, offset, xform }),
            image: None,
            crop: None,
            fail: None,
            batch: 64,
            rows: Some(3 + (offset != [0.0, 0.0]) as u32 + (opacity != 1.0) as u32),
        }
    }

    /// A bitmap brush's texel→device matrix under the current transform, DPI and unit mode. A bitmap occupies its own
    /// DIPs, so a non-96-dpi bitmap is `texels·96/dpi` DIP wide (a 144-dpi bitmap tiles tighter): the brush→device
    /// matrix maps brush DIPs, so its linear part scales by `96/dpi` to sample texels (`CreateImageBrush` does the same
    /// via `source·dpi/96`). Under D2D1_UNIT_MODE_PIXELS a texel is a pixel whatever the bitmap's DPI (VM probe-bdpi,
    /// as DrawBitmap's source rect).
    fn bitmap_placement(&self, transform: Option<[f32; 6]>, dpi: (f32, f32)) -> [f32; 6] {
        let (sx, sy) = if self.state.unit == 1 { (1.0, 1.0) } else { (96.0 / dpi.0, 96.0 / dpi.1) };
        let b = self.brush_to_device(transform);
        [b[0] * sx, b[1] * sx, b[2] * sy, b[3] * sy, b[4], b[5]]
    }

    /// A NEAREST, LINEAR or MULTI_SAMPLE_LINEAR bitmap brush's device→texcoord VS row, as d2d1 builds it from the brush
    /// matrix and the world·DPI matrix apart, op for op ([`hw_bitmap_row`]). Inverting the placement, which composes them
    /// first, lands its coefficients an ulp off, enough to move a sampler address across a Q8 step (r27 32579: an MSL
    /// WRAP tap on a tie; VM sweeps geo/c579s, c579w and c579l of brush fills under every extend, world transforms,
    /// bitmap DPIs and the pixel unit mode: 27 MSL and 6 LINEAR/NEAREST probes exact, the DDI VS constants bit for bit).
    fn bitmap_vs_row(&self, transform: Option<[f32; 6]>, dpi: (f32, f32), size: [u32; 2], interpolation: u32) -> Option<[f32; 6]> {
        if !matches!(interpolation, 0 | 1 | 3) {
            return None;
        }
        // The bitmap's size in DIPs, `(n·96)/dpi` (pixels in the pixel unit mode), scales the brush matrix as it stands.
        let dips = if self.state.unit == 1 { size.map(|v| v as f32) } else { [(size[0] as f32 * 96.0) / dpi.0, (size[1] as f32 * 96.0) / dpi.1] };
        let b = transform.unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        Some(hw_bitmap_row(dips, b, self.device_transform().map(|v| v as f32)))
    }

    /// `CreateBitmapBrush`: a bitmap brush from a realized bitmap, its extend/interpolation modes,
    /// brush transform and opacity.
    pub fn create_bitmap_brush(&self, bitmap: &Bitmap, extend: [u32; 2], interpolation: u32, opacity: f32, transform: Option<[f32; 6]>) -> PixelBrush {
        // Present an R8G8B8A8 source to the BGRA8 sampler with R/B swapped; A8 stays A8 so `check_brush`
        // still rejects it at the fill.
        let sampled = bitmap.as_sampled_bgra8();
        let bitmap = sampled.as_ref();
        let b2d = self.bitmap_placement(transform, bitmap.dpi);
        let mut b = Brush::bitmap_source(bitmap.clone(), &b2d, extend, interpolation, opacity);
        if let Some(row) = self.bitmap_vs_row(transform, bitmap.dpi, [bitmap.width, bitmap.height], interpolation) { b.set_vs_row(row); }
        b.set_intermediate_precision(self.buffer_precision);
        let xform = transform.unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        PixelBrush {
            domain: self.domain,
            brush: b,
            def: Some(crate::brush::BrushDef::Bitmap { texels: bitmap.premul_bgra.clone(), size: [bitmap.width, bitmap.height], extend, interpolation, opacity, xform, dpi: bitmap.dpi }),
            image: None,
            crop: None,
            // HIGH_QUALITY_CUBIC can't place the bitmap through a non-finite matrix entry, as for an image brush below.
            fail: (interpolation == 5 && transform.is_some_and(|m| !m.iter().all(|v| v.is_finite()))).then_some(E_INVALIDARG),
            batch: 56,
            rows: Some(2 + (opacity != 1.0) as u32),
        }
    }

    /// `CreateImageBrush`. A bitmap image is cropped to `source` (texels) and sampled as a bitmap brush
    /// with the crop's top-left at the brush origin. An effect or command list follows native's
    /// Native's image-brush graph: the image is realized
    /// over a rounded pixel rect (BorderRect: that rect plus the extend modes) and mapped onto the
    /// exact `source` rect, whose top-left is the brush origin (AffineTransform, brush→device):
    ///   * effect: `source·dpi/96` rounded half-up, realized at the context DPI, the rounded rect
    ///     stretched back onto the fractional one;
    ///   * command list: rendered resolution-independently at the brush's scale
    ///     (`ExtractScaleAndTranslate` of brush→device, `source` fitted onto its rounded device rect),
    ///     sampled LINEAR whatever the brush's interpolation.
    #[allow(clippy::too_many_arguments)]
    pub fn create_image_brush(&mut self, image: BrushImage, source: [f32; 4], extend: [u32; 2], interpolation: u32, opacity: f32, transform: Option<[f32; 6]>) -> PixelBrush {
        PixelBrush { batch: 80, ..self.image_brush(image, source, extend, interpolation, opacity, transform) }
    }

    fn image_brush(&mut self, image: BrushImage, source: [f32; 4], extend: [u32; 2], interpolation: u32, opacity: f32, transform: Option<[f32; 6]>) -> PixelBrush {
        // HIGH_QUALITY_CUBIC can't place a bitmap or effect through a non-finite brush matrix: a draw with
        // the brush fails with E_INVALIDARG, ahead of INTERMEDIATE_TOO_LARGE (r24 29252; VM: NaN or inf in
        // any slot, translation included; other modes draw nothing, and a command list never fails).
        let hq_matrix = interpolation == 5
            && !matches!(image, BrushImage::CommandList(_))
            && transform.is_some_and(|m| !m.iter().all(|v| v.is_finite()));
        // `D2DImageBrush::CloneBrushIntoMemory` normalises an inverted source rect.
        let s = [source[0].min(source[2]), source[1].min(source[3]), source[0].max(source[2]), source[1].max(source[3])];
        if let BrushImage::Bitmap(b) = image {
            // d2d1 draws an image brush over a bitmap as a plain bitmap brush when its source rect, in the bitmap's
            // pixels, is the whole bitmap or a whole-pixel sub-rect of it. It is still an image brush: a non-finite
            // matrix entry voids it (unlike a plain CreateBitmapBrush).
            let (kx, ky) = (b.dpi.0 / 96.0, b.dpi.1 / 96.0);
            let px = [s[0] * kx, s[1] * ky, s[2] * kx, s[3] * ky];
            let fast = crate::bitmap::fast_path_bitmap(px, [b.width, b.height], extend, interpolation, false, true);
            let m = transform.unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
            let (brush_m, ext) = match fast {
                // A whole-bitmap source d2d1 does not convert (ANISOTROPIC, HIGH_QUALITY_CUBIC) goes through the image
                // brush's effect graph, which samples the whole bitmap through that filter as the bitmap brush does.
                // ponytail: modelled by the bitmap brush; our graph route is not faithful for these filters.
                crate::bitmap::FastPathBitmap::None if source == [0.0, 0.0, b.width as f32, b.height as f32] => (Some(transform), extend),
                crate::bitmap::FastPathBitmap::None => (None, extend),
                crate::bitmap::FastPathBitmap::Copy => (Some(transform), extend),
                // The sub-rect's top-left (in DIPs) is the brush origin.
                crate::bitmap::FastPathBitmap::Scaled(r) => {
                    let (l, t) = (r[0] / kx, r[1] / ky);
                    (Some(Some([m[0], m[1], m[2], m[3], m[4] - (l * m[0] + t * m[2]), m[5] - (l * m[1] + t * m[3])])), [0, 0])
                }
            };
            if let Some(bm) = brush_m {
                let mut pb = self.create_bitmap_brush(b, ext, interpolation, opacity, bm);
                pb.brush.set_origin(crate::brush::BrushOrigin::BitmapImage);
                if let crate::bitmap::FastPathBitmap::Scaled(r) = fast {
                    let (iw, ih) = (1.0 / b.width as f32, 1.0 / b.height as f32);
                    pb.brush.set_clamp([iw * (r[0] + 0.5), ih * (r[1] + 0.5), iw * (r[2] - 0.5), ih * (r[3] - 0.5)]);
                }
                pb.fail = hq_matrix.then_some(E_INVALIDARG);
                return pb;
            }
        }
        let user = transform.unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let round = |v: f32| (v + 0.5).floor(); // native's cvtss2si + "−0.5 ⇒ +1": round half up
        let (dx, dy) = (self.dpi.0, self.dpi.1);
        // A command list under a brush whose 2×2 collapses to zero puts its whole source rect on one device point. Direct2D
        // renders the list into one atlas texel covering that rect (each primitive's AA coverage on 8 sample columns,
        // composited in order) and every pixel samples texel (0, 0): a constant, not the singular brush's void (recap-f 7284).
        if let BrushImage::CommandList(l) = image {
            let at_origin = [user[0], user[1], user[2], user[3], user[4] - s[0] * user[0] - s[1] * user[2], user[5] - s[0] * user[1] - s[1] * user[3]];
            // MULTI_SAMPLE_LINEAR and ANISOTROPIC keep the singular void (r23 28483, 28902).
            if !matches!(interpolation, 3 | 4) && self.brush_to_device(Some(at_origin))[..4] == [0.0; 4] && s[2] > s[0] && s[3] > s[1] {
                let (kx, ky) = (1.0 / (s[2] - s[0]), 1.0 / (s[3] - s[1]));
                let texel = self.realize_list_px(l, [kx, ky, -s[0] * kx, -s[1] * ky], [1, 1]);
                return PixelBrush { rows: None, ..self.create_bitmap_brush(&texel, [0, 0], 0, opacity, Some([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])) };
            }
        }
        // `r`: the rounded pixel rect; `tb` = [ax, ay, bx, by]: texel (relative to r) → brush space.
        let mut hq_fail = hq_matrix;
        let (r, tb, interp) = match image {
            // A bitmap's source rect is in its own texels: rounded, stretched onto the exact rect, and
            // transparent past the bitmap (r5 5930: a WRAP tile of 24.04×39.23 over a 7×8 bitmap).
            BrushImage::Effect(_) | BrushImage::Bitmap(_) => {
                let (dx, dy) = if let BrushImage::Bitmap(b) = image { b.dpi } else { (dx, dy) };
                let e = [dx * s[0] / 96.0, dy * s[1] / 96.0, dx * s[2] / 96.0, dy * s[3] / 96.0];
                // HIGH_QUALITY_CUBIC can't size an image whose scaled source edge overflowed: a draw with the
                // brush fails with E_INVALIDARG (r24 29306, 29974; VM: a bitmap's edge fails from FLT_MAX/96
                // whatever the context dpi, brush matrix or extend; other modes draw nothing).
                hq_fail |= interpolation == 5 && !e.iter().all(|v| v.is_finite());
                let r = e.map(round);
                // Stretch the rounded rect onto the exact one: px' = k·px + o (f64, then f32).
                let k = |a: f32, b: f32, ra: f32, rb: f32| ((b - a) as f64 / (rb - ra) as f64) as f32;
                let (kx, ky) = (k(e[0], e[2], r[0], r[2]), k(e[1], e[3], r[1], r[3]));
                let (ox, oy) = ((e[0] as f64 - kx as f64 * r[0] as f64) as f32, (e[1] as f64 - ky as f64 * r[1] as f64) as f32);
                let tb = [kx * 96.0 / dx, ky * 96.0 / dy, (kx * r[0] + ox) * 96.0 / dx - s[0], (ky * r[1] + oy) * 96.0 / dy - s[1]];
                (r, tb, interpolation)
            }
            BrushImage::CommandList(_) => {
                // brush→device with the source origin at the brush origin, then its row scales.
                let at_origin = [user[0], user[1], user[2], user[3], user[4] - s[0] * user[0] - s[1] * user[2], user[5] - s[0] * user[1] - s[1] * user[3]];
                let m = self.brush_to_device(Some(at_origin));
                let (r, tb) = Self::image_device_scale_realize(&m, s);
                (r, tb, 1)
            }
        };
        let (w, h) = ((r[2] - r[0]).max(0.0), (r[3] - r[1]).max(0.0));
        let win = self.image_brush_window([w, h], tb, &user);
        // Saturating: a 1e12 brush scale puts `r` past i32 (9-corpus seed 130 creates one, never draws it).
        // ponytail: drawing with such a brush is unverified against native; probe it if a seed ever does.
        let px: [i32; 4] = std::array::from_fn(|k| (r[k % 2] as i32).saturating_add(win[k]));
        let mut bm = match image {
            BrushImage::Effect(e) => self.realize_effect_px(e, px),
            BrushImage::CommandList(l) => {
                // DIP → texel: the fitted q = k·p + o, less the window's origin.
                let (kx, ky) = (1.0 / tb[0], 1.0 / tb[1]);
                let (ox, oy) = (r[0] - kx * (tb[2] + s[0]), r[1] - ky * (tb[3] + s[1]));
                self.realize_list_px(l, [kx, ky, ox - px[0] as f32, oy - px[1] as f32], [px[2] - px[0], px[3] - px[1]])
            }
            // Texels past the bitmap are transparent whatever its alpha mode: a STRAIGHT bitmap is made premultiplied, and an
            // IGNORE one keeps its stored alpha, before the crop pads it, or its padding would sample as opaque black (r27 32016).
            BrushImage::Bitmap(b) => b.as_cropped_bgra8().crop(px.map(|v| v as f32)),
        };
        if bm.width == 0 || bm.height == 0 {
            bm = Bitmap { width: 1, height: 1, premul_bgra: vec![0; 4], format: DXGI_FORMAT_B8G8R8A8_UNORM, alpha: D2D1_ALPHA_MODE_PREMULTIPLIED, dpi: self.dpi };
        }
        let (ax, ay) = (tb[0], tb[1]);
        let (bx, by) = (tb[2] + ax * win[0] as f32, tb[3] + ay * win[1] as f32);
        let mut m = [ax * user[0], ax * user[1], ay * user[2], ay * user[3], bx * user[0] + by * user[2] + user[4], bx * user[1] + by * user[3] + user[5]];
        // `create_image_brush` already folds the source's dpi into `m` (the realized tile is placed in
        // device space here), so the realized intermediate must sample 1:1 — reset its dpi to 96 so
        // `create_bitmap_brush`'s texel→DIP scale is a no-op and does not double-apply the dpi.
        bm.dpi = (96.0, 96.0);
        let mut extend = extend;
        // A command list whose brush collapses one source axis (a zero column) realizes that axis as one texel over the
        // whole source, and every pixel samples it there: a unit column clamped to that texel, not the singular void (sc88).
        if let (BrushImage::CommandList(_), false) = (&image, matches!(interpolation, 3 | 4)) {
            for k in 0..2 {
                let j = 2 * k;
                if m[j] == 0.0 && m[j + 1] == 0.0 && (m[2 - j] != 0.0 || m[3 - j] != 0.0) {
                    (m[j], m[j + 1]) = if k == 0 { (m[3], -m[2]) } else { (-m[1], m[0]) };
                    extend[k] = 0;
                }
            }
        }
        let mut pb = self.create_bitmap_brush(&bm, extend, interp, opacity, Some(m));
        // An image brush realizes through the full matrix, so a non-finite entry voids it (per kind).
        pb.brush.set_origin(match &image {
            BrushImage::CommandList(_) => crate::brush::BrushOrigin::OtherImage,
            _ => crate::brush::BrushOrigin::StagedImage,
        });
        // Like an effect image, the bitmap's tile is drawn from a power-of-two (≥512) atlas.
        if let (BrushImage::Bitmap(_), Brush::Bitmap { texture_size, shader_draw, .. }) = (&image, &mut pb.brush) {
            *texture_size = [bm.width, bm.height].map(|v| v.next_power_of_two().max(512));
            *shader_draw = true;
        }
        // The image's own pixels within the realized window; past them a source rect holds transparent padding.
        let image_px = match &image {
            BrushImage::Bitmap(b) => Some([0.0, 0.0, b.width as f32, b.height as f32]),
            BrushImage::Effect(e) => e.intrinsic_bounds(),
            BrushImage::CommandList(_) => None,
        };
        if let (Some(b), Brush::Bitmap { content, .. }) = (image_px, &mut pb.brush) {
            let b = b.map(|v| v as i32);
            // Saturating: an unbounded effect's bounds and window sit at the i32 limits.
            *content = Some([b[0].max(px[0]).saturating_sub(px[0]), b[1].max(px[1]).saturating_sub(px[1]), b[2].min(px[2]).saturating_sub(px[0]), b[3].min(px[3]).saturating_sub(px[1])]);
        }
        // A command list's realization is an atlas too: geometry fills sample it in texels (r5 5592).
        if let (BrushImage::CommandList(_), Brush::Bitmap { texture_size, .. }) = (&image, &mut pb.brush) {
            *texture_size = [bm.width, bm.height].map(|v| v.next_power_of_two().max(512));
        }
        if let (BrushImage::Bitmap(_), true) = (&image, win == [0, 0, w as i32, h as i32]) {
            pb.crop = Some(ImageCrop { tile: px, texels: std::rc::Rc::new(bm.clone()), m, extend, interp, opacity });
        }
        if let (BrushImage::Effect(e), true) = (image, win == [0, 0, w as i32, h as i32]) {
            pb.image = Some(std::rc::Rc::new(ImageBrushDef { effect: e.clone(), tile: px, m, extend, interp, opacity }));
        }
        if hq_fail {
            pb.fail = Some(E_INVALIDARG);
        }
        // Not the fast path's bitmap brush: the fill is an operation of the deferred image context.
        pb.rows = None;
        pb
    }

    /// Native realizes an effect image brush per draw (d2d1 `BrushRedirection`): from the draw's device
    /// rect `d` it maps back the need region (`CBrushTransformNode::MapOutputRectToInputRects`: the
    /// pixel-centre inset rect through the inverse matrix, padded by the filter reach), renders the
    /// BorderRect output there — the effect's last pass drawn as border cells (a nine-grid when both
    /// axes CLAMP), or past 25 tiles (`ShouldUseVsAndPsForBorder`) the Border shader over the tile —
    /// and the draw samples that intermediate through the AffineTransform matrix. Text samples it on
    /// WARP's unit-triangle plane.
    /// `None` keeps the brush realized at creation.
    pub(crate) fn image_brush_for_draw(&mut self, brush: &PixelBrush, d: [i32; 4], text: bool, latch: bool) -> Option<PixelBrush> {
        use crate::effects::builtin::filters::{border_ps, BorderCells};
        if let Some(crop) = brush.crop.clone() {
            return self.crop_brush_for_draw(brush, &crop, d, text);
        }
        let def = brush.image.as_ref()?;
        if d[2] <= d[0] || d[3] <= d[1] {
            return None;
        }
        let tile = def.tile;
        let size = [tile[2] - tile[0], tile[3] - tile[1]];
        let inv = self.image_brush_inverse(tile, def.m)?;
        if latch && self.image_brush_border_too_large(def.extend, tile, &inv, d) {
            self.process_new_error(INTERMEDIATE_TOO_LARGE);
            return None;
        }
        // A need too large to realize (a brush minified to a speck: scale 1e-3 to 1e-6) is the Border shader over the
        // tile itself, the draw's pixel shader sampling the tile through the brush's extend modes (sc154, sc162-sc165:
        // CLAMP, WRAP and MIRROR at those scales within 1, where the realized need drew nothing).
        let need = Self::image_brush_need(&inv, &self.image_brush_forward(tile, def.m), def.interp, d)?;
        let nsize = need.map_or([i32::MAX; 2], |n| [n[2] - n[0], n[3] - n[1]]);
        let Some(n) = need else {
            let t = self.realize_effect_px(&def.effect, tile);
            let mut pb = self.create_bitmap_brush(&t, def.extend, def.interp, def.opacity, Some(def.m));
            if let Brush::Bitmap { texture_size, shader_draw, .. } = &mut pb.brush {
                *texture_size = [t.width, t.height].map(|v| v.next_power_of_two().max(512));
                *shader_draw = true;
            }
            return Some(pb);
        };
        let tiles = |k: usize| (nsize[k] / size[k]).max(1) as i64;
        let bm = if tiles(0) * tiles(1) > 25 {
            let t = self.realize_effect_px(&def.effect, tile);
            border_ps(&t, [tile[0], tile[1]], n, def.extend).bitmap
        } else {
            let cells = BorderCells { origin: [n[0], n[1]], size: nsize, cells: border_cells(tile, n, def.extend) };
            // The border asks its input for what the need folds onto (d2d1's border transform), which places the blur's passes.
            match self.realize_effect_cells(&def.effect, tile, border_request(tile, n, def.extend), cells) {
                Some(bm) => bm,
                None => border_fill(&self.realize_effect_px(&def.effect, tile), tile, n, def.extend),
            }
        };
        let (dx, dy) = ((n[0] - tile[0]) as f32, (n[1] - tile[1]) as f32);
        let a = def.m;
        let mi = [a[0], a[1], a[2], a[3], a[0] * dx + a[2] * dy + a[4], a[1] * dx + a[3] * dy + a[5]];
        let mut pb = self.create_bitmap_brush(&bm, [0, 0], def.interp, def.opacity, Some(mi));
        if let Brush::Bitmap { to_source, texture_size, .. } = &mut pb.brush {
            // The realization lives in a power-of-two (≥512) atlas, which AlphaBlt normalizes by.
            *texture_size = [bm.width, bm.height].map(|v| v.next_power_of_two().max(512));
            if text {
                *to_source = crate::warp::plane::warp_fit_affine(to_source);
            }
        }
        Some(pb)
    }

    /// An image brush's need region over device rect `d` ([`Self::image_brush_for_draw`]): `d`'s pixel-centre inset rect
    /// through the inverse matrix `inv`, padded by the filter's reach and rounded. `None` when nothing can be sized;
    /// `Some(None)` for a need too large to realize as border cells (past 2^22 texels or outside ±1e6).
    /// A brush whose AffineTransform stays in its graph ([`kept_affine`]) takes that graph's need instead: its matrix
    /// transform maps the inset corners back through the adjugate of its matrix over the determinant, snaps each bound
    /// within 1e-4 of a whole texel and widens the floor and ceiling by its reach; a HIGH_QUALITY_CUBIC pre-scale in
    /// front (a Scale effect by `ps`, the matrix then `S(1/ps)·fwd`) maps that back per axis, `(v ± 0.5)/s` widened by
    /// the scaler's reach (1 at a unit scale, 4 above half, else `⌊2/s + 0.5⌋`).
    fn image_brush_need(inv: &[f32; 6], fwd: &[f32; 6], interp: u32, d: [i32; 4]) -> Option<Option<[i32; 4]>> {
        let inset = |v: i32, lo: i32| v as f32 + if v == lo { 0.5 } else { -0.5 };
        let corners = [(d[0], d[1]), (d[2], d[1]), (d[0], d[3]), (d[2], d[3])].map(|(x, y)| (inset(x, d[0]), inset(y, d[1])));
        let kept = kept_affine(interp, fwd);
        let pts = match kept {
            None => corners.map(|(x, y)| [inv[0] * x + inv[2] * y + inv[4], inv[1] * x + inv[3] * y + inv[5]]),
            Some((_, ps)) => {
                let (kx, ky) = (1.0 / ps[0], 1.0 / ps[1]);
                let [a, b, c, d, e, f] = if ps == [1.0; 2] { *fwd } else { [fwd[0] * kx, fwd[1] * kx, fwd[2] * ky, fwd[3] * ky, fwd[4], fwd[5]] };
                let (det, tx, ty) = ((a * d) - (c * b), (c * f) - (e * d), (e * b) - (a * f));
                corners.map(|(x, y)| [((x * d + y * -c) + tx) / det, ((x * -b + y * a) + ty) / det])
            }
        };
        let lo = |k: usize| pts.iter().map(|p| p[k]).fold(f32::INFINITY, f32::min);
        let hi = |k: usize| pts.iter().map(|p| p[k]).fold(f32::NEG_INFINITY, f32::max);
        let fit = |v: f32| v.is_finite() && v.abs() < 1e6;
        let (l, t, r, bt) = match kept {
            None => {
                let pad = match interp { 0 => 0.53125, 2 => 2.0, _ => 1.0 };
                (lo(0) - pad, lo(1) - pad, hi(0) + pad - 0.5, hi(1) + pad - 0.5)
            }
            Some((e, ps)) => {
                let snap = |v: f32| { let n = v.round(); if (v - n).abs() < 1e-4 { n } else { v } };
                let m = [snap(lo(0)).floor() - e, snap(lo(1)).floor() - e, snap(hi(0)).ceil() + e, snap(hi(1)).ceil() + e];
                if ps == [1.0; 2] {
                    (m[0], m[1], m[2] - 0.5, m[3] - 0.5)
                } else {
                    let reach = |s: f32| if s > 1.0 { 2.0 } else if s == 1.0 { 1.0 } else if s > 0.5 { 4.0 } else { (2.0 / s + 0.5).floor() };
                    let (ex, ey) = (reach(ps[0]), reach(ps[1]));
                    ((m[0] + 0.5) / ps[0] - ex, (m[1] + 0.5) / ps[1] - ey, ((m[2] - 0.5) / ps[0] + ex) - 0.5, ((m[3] - 0.5) / ps[1] + ey) - 0.5)
                }
            }
        };
        if ![l, t, r, bt].into_iter().all(f32::is_finite) {
            return None;
        }
        let n = [l, t, r, bt].into_iter().all(fit).then(|| [(l + 0.5).floor() as i32, (t + 0.5).floor() as i32, r.ceil() as i32, bt.ceil() as i32]);
        let nsize = n.map_or([i32::MAX; 2], |n| [n[2] - n[0], n[3] - n[1]]);
        if nsize[0] <= 0 || nsize[1] <= 0 {
            return None;
        }
        Some(n.filter(|_| nsize[0] as i64 * nsize[1] as i64 <= 1 << 22))
    }

    /// Realize an image brush's input at the brush's DEVICE SCALE, not the full source resolution: the
    /// per-axis scale is the brush→device column magnitude, the source rect `s` rounds HALF-UP against it,
    /// and the source fits back onto the rounded rect. Returns the rounded realization rect `r` (device
    /// texels) and `tb` = [ax, ay, bx, by] mapping a texel in `r` back to brush space. A minifying brush
    /// therefore realizes a SMALL intermediate (the device footprint), which the filter then samples.
    /// Shared by the command-list brush realization and a minifying cropped bitmap brush.
    fn image_device_scale_realize(m: &[f32; 6], s: [f32; 4]) -> ([f32; 4], [f32; 4]) {
        let round = |v: f32| (v + 0.5).floor(); // native cvtss2si + "−0.5 ⇒ +1": round half up
        let (sx, sy) = ((m[1] * m[1] + m[0] * m[0]).sqrt(), (m[3] * m[3] + m[2] * m[2]).sqrt());
        let mut r = [round(s[0] * sx + m[4]), round(s[1] * sy + m[5]), round(s[2] * sx + m[4]), round(s[3] * sy + m[5])];
        if r[0] > r[2] || r[2] - r[0] == 0.0 {
            r[2] += 1.0;
        }
        if r[1] > r[3] || r[3] - r[1] == 0.0 {
            r[3] += 1.0;
        }
        let fit = |a: f32, b: f32, ra: f32, rb: f32, scale: f32, t: f32| {
            // The source fits the rounded rect, a one-pixel one included (sc88: a 1e-6 brush samples its whole source
            // there); an empty source has nothing to fit and keeps the extracted scale (r20 25199 draws nothing).
            if b == a { return (scale, t); }
            let k = (rb - ra) as f64 / (b - a) as f64;
            (k as f32, (ra as f64 - a as f64 * k) as f32)
        };
        let (kx, ox) = fit(s[0], s[2], r[0], r[2], sx, m[4]);
        let (ky, oy) = fit(s[1], s[3], r[1], r[3], sy, m[5]);
        (r, [1.0 / kx, 1.0 / ky, (r[0] - ox) / kx - s[0], (r[1] - oy) / ky - s[1]])
    }

    /// A draw with a cropped bitmap image brush over device rect `d`. Its graph's border node passes through when the
    /// draw's need region lies inside the tile: the draw then samples the image itself, at its own size, through the brush
    /// (DDI: only the bitmap bound, no atlas). A need reaching past the tile renders the border cells into the frame's
    /// atlas, ahead of any slow-path DrawImage after it (DDI: a NEAREST CLAMP/MIRROR brush's 22x6 need, 18 texels of
    /// source and 4 clamped, at atlas texel 0, the next DrawImage's slot after it); that realization the creation-time
    /// crop already samples as. A LINEAR, CUBIC or MULTI_SAMPLE_LINEAR fill then addresses the cells at their slot;
    /// otherwise `None` keeps the creation-time brush and only the slot is taken here.
    /// ponytail: past 25 tiles (the Border shader over the tile) no slot is taken; a later image's filter reach landing
    /// on the slot does not reach the draw.
    fn crop_brush_for_draw(&mut self, brush: &PixelBrush, crop: &ImageCrop, d: [i32; 4], text: bool) -> Option<PixelBrush> {
        if d[2] <= d[0] || d[3] <= d[1] {
            return None;
        }
        let tile = crop.tile;
        let inv = self.image_brush_inverse(tile, crop.m)?;
        let n = Self::image_brush_need(&inv, &self.image_brush_forward(tile, crop.m), crop.interp, d)??;
        if n[0] >= tile[0] && n[1] >= tile[1] && n[2] <= tile[2] && n[3] <= tile[3] {
            // A need within the bitmap's own texels samples the bitmap itself, at its own size, not the tile with its
            // transparent padding (VM: an ANISOTROPIC fill's plane normalized over a 16x4 bitmap, not its 24x5 tile).
            let own = match &brush.brush {
                Brush::Bitmap { content: Some(c), .. } if [c[0], c[1], c[2], c[3]] != [0, 0, tile[2] - tile[0], tile[3] - tile[1]]
                    && n[0] >= tile[0] + c[0] && n[1] >= tile[1] + c[1] && n[2] <= tile[0] + c[2] && n[3] <= tile[1] + c[3] => Some(*c),
                _ => None,
            };
            let (texels, m) = match own {
                Some(c) => {
                    let (m, o) = (crop.m, [c[0] as f32, c[1] as f32]);
                    (crop.texels.crop(c.map(|v| v as f32)), [m[0], m[1], m[2], m[3], (o[0] * m[0] + o[1] * m[2]) + m[4], (o[0] * m[1] + o[1] * m[3]) + m[5]])
                }
                None => ((*crop.texels).clone(), crop.m),
            };
            let mut pb = self.create_bitmap_brush(&texels, crop.extend, crop.interp, crop.opacity, Some(m));
            if let (true, Brush::Bitmap { to_source, .. }) = (text, &mut pb.brush) {
                *to_source = crate::warp::plane::warp_fit_affine(to_source);
            }
            pb.brush.set_origin(crate::brush::BrushOrigin::StagedImage);
            return Some(pb);
        }
        let (size, ns) = ([tile[2] - tile[0], tile[3] - tile[1]], [n[2] - n[0], n[3] - n[1]]);
        // Past 25 tiles the Border shader realizes the whole need into its own square power-of-two intermediate (at least
        // 512), the need at its origin, which a LINEAR, CUBIC or MULTI_SAMPLE_LINEAR fill samples as it samples cells in the
        // atlas (VM DDI of CUBIC crop fills, needs 360 to 1787 texels: 512, 1024 and 2048 targets, the fill's texel step
        // 1/size; r22 27703).
        let cells = (ns[0] / size[0]).max(1) as i64 * (ns[1] / size[1]).max(1) as i64 <= 25;
        let (at, atlas_size) = if cells { (self.atlas.alloc(ns[0], ns[1]), 512) }
            else if matches!(crop.interp, 1..=3) { (Some([0, 0]), (ns[0].max(ns[1]) as u32).next_power_of_two().max(512)) }
            else { (None, 512) };
        // A LINEAR, CUBIC, MULTI_SAMPLE_LINEAR or ANISOTROPIC fill samples the cells through the atlas's own row: the crop's matrix
        // after the cell's offset, over the atlas size, under the device transform (VM: the fill kernel's plane, all six
        // coefficients). The offset is two f32 translations: the brush's own, to the tile origin, then the redirect's to
        // the need region's atlas place, each `(y·m21 + x·m11) + dx` (d2d1's RedirectToBitmap; a CUBIC sub-rect's plane
        // constants differ in the last bits from one combined step).
        let cell = at.filter(|_| matches!(crop.interp, 1..=4) && !text).map(|at| {
            let c = [at[0] + tile[0] - n[0], at[1] + tile[1] - n[1]];
            let tr = |m: [f32; 6], o: [i32; 2]| {
                let o = o.map(|v| v as f32);
                [m[0], m[1], m[2], m[3], (o[0] * m[0] + o[1] * m[2]) + m[4], (o[0] * m[1] + o[1] * m[3]) + m[5]]
            };
            let b = tr(tr(crop.m, [-tile[0], -tile[1]]), [n[0] - at[0], n[1] - at[1]]);
            let w = self.device_transform().map(|v| v as f32);
            // The CUBIC pixel shader's phase matrix is the crop's before the redirect: texels from the tile origin.
            let fwd = hw_bitmap_fwd([1.0; 2], tr(crop.m, [-tile[0], -tile[1]]), w);
            (hw_bitmap_row([atlas_size as f32; 2], b, w), c, (fwd, d.map(|v| v as f32)))
        });
        // The border cells render into an intermediate of the draw's buffer precision, which the fill samples at that
        // precision: a float or sRGB one through the float sampler, holding each texel quantized to it.
        if self.buffer_precision >= 2 {
            let t = &crop.texels;
            let mut data = Vec::with_capacity(t.width as usize * t.height as usize * 16);
            for y in 0..t.height as i32 {
                for x in 0..t.width as i32 {
                    for v in crate::effects::quantize_to(self.buffer_precision, t.texel_rgba(x, y)) { data.extend_from_slice(&v.to_le_bytes()); }
                }
            }
            let fb = Bitmap { width: t.width, height: t.height, premul_bgra: data, format: d2d_types::abi::DXGI_FORMAT_R32G32B32A32_FLOAT, alpha: t.alpha, dpi: t.dpi };
            let mut pb = self.create_bitmap_brush(&fb, crop.extend, crop.interp, crop.opacity, Some(crop.m));
            pb.brush.set_origin(crate::brush::BrushOrigin::StagedImage);
            if let Some((row, c, fwd)) = cell { pb.brush.set_atlas_cell(row, c, atlas_size, fwd); }
            return Some(pb);
        }
        cell.map(|(row, c, fwd)| {
            let mut pb = brush.clone();
            pb.brush.set_atlas_cell(row, c, atlas_size, fwd);
            pb
        })
    }

    /// An ink's mask tiles take slots of the frame's atlas, one each, in order.
    pub(crate) fn ink_tile_slots(&mut self, tiles: &[[f32; 4]]) -> Vec<[i32; 2]> {
        self.atlas.tile_slots(tiles)
    }

    /// The inverse of an image brush's image-pixel→device matrix `m` under the current transform (texel-relative to the
    /// `tile`'s top-left), or `None` when the tile is empty or the matrix singular or non-finite.
    fn image_brush_inverse(&self, tile: [i32; 4], m: [f32; 6]) -> Option<[f32; 6]> {
        let m = self.image_brush_forward(tile, m);
        let det = m[0] * m[3] - m[1] * m[2];
        if tile[2] == tile[0] || tile[3] == tile[1] || det == 0.0 || !det.is_finite() {
            return None;
        }
        Some([m[3] / det, -m[1] / det, -m[2] / det, m[0] / det, (m[2] * m[5] - m[3] * m[4]) / det, (m[1] * m[4] - m[0] * m[5]) / det])
    }

    /// An image brush's matrix from its image's pixels to device pixels, the tile `m` places at its own position.
    fn image_brush_forward(&self, tile: [i32; 4], m: [f32; 6]) -> [f32; 6] {
        let b = self.brush_to_device(Some(m));
        [b[0], b[1], b[2], b[3], b[4] - (b[0] * tile[0] as f32 + b[2] * tile[1] as f32), b[5] - (b[1] * tile[0] as f32 + b[3] * tile[1] as f32)]
    }

    /// Whether an effect image brush's Border fails the draw with INTERMEDIATE_TOO_LARGE. Native folds a WRAP or MIRROR
    /// Border into the sampler's address mode only while the draw's need on that axis stays within the tile size times
    /// the device's texture wrap count (16384 at feature level 11) past the tile; otherwise it realizes the Border's
    /// output, and that fails once even a minimal 16-pixel output tile needs more than a texture's 16384 texels of it
    /// (sc161-sc169: a 2-texel WRAP tile fails at 32771.5 and draws at 32710, a CLAMP axis never fails; r23 28264's
    /// need lies 400000 out but spans 48, and draws; r27 32005, 32241). Only the part of the draw the clips and
    /// layers leave counts: native processes no tile of a draw clipped away (recap 6340, r13 18734, r15 20496).
    /// ponytail: a tile the effect's output does not cover (a source rectangle outside it) fails at any scale natively.
    fn image_brush_border_too_large(&self, extend: [u32; 2], tile: [i32; 4], inv: &[f32; 6], d: [i32; 4]) -> bool {
        let v = self.reach_rect();
        let e = [d[0].max(v[0] as i32), d[1].max(v[1] as i32), d[2].min(v[2] as i32), d[3].min(v[3] as i32)];
        if e[2] <= e[0] || e[3] <= e[1] {
            return false;
        }
        let pts = [(e[0], e[1]), (e[2], e[1]), (e[0], e[3]), (e[2], e[3])].map(|(x, y)| {
            let (x, y) = (x as f32 + if x == e[0] { 0.5 } else { -0.5 }, y as f32 + if y == e[1] { 0.5 } else { -0.5 });
            [inv[0] * x + inv[2] * y + inv[4], inv[1] * x + inv[3] * y + inv[5]]
        });
        (0..2).any(|k| {
            let lo = pts.iter().map(|p| p[k]).fold(f32::INFINITY, f32::min);
            let hi = pts.iter().map(|p| p[k]).fold(f32::NEG_INFINITY, f32::max);
            let reach = (tile[k + 2] - tile[k]) as f32 * 16384.0;
            extend[k] != 0 && lo.is_finite() && hi.is_finite()
                && (hi - tile[k + 2] as f32 > reach || tile[k] as f32 - lo > reach)
                && 16.0 * (inv[k].abs() + inv[k + 2].abs()) > 16384.0
        })
    }

    /// The texel window ([x0, y0, x1, y1] within `size`) an effect / command-list brush realizes.
    /// Native renders the image only where the draw samples it, so a huge `sourceRectangle` (fuzz
    /// 6276) must not become a huge bitmap: past ~4M texels, realize only what the target can sample
    /// (the target inverse-mapped through texel→device, plus a 2-texel apron, at least one texel).
    /// ponytail: a WRAP/MIRROR period boundary inside that window is not reproduced.
    fn image_brush_window(&self, size: [f32; 2], tb: [f32; 4], user: &[f32; 6]) -> [i32; 4] {
        let full = [0, 0, size[0] as i32, size[1] as i32];
        if size[0] as f64 * size[1] as f64 <= (1u64 << 22) as f64 {
            return full;
        }
        let t2b = [tb[0] * user[0], tb[0] * user[1], tb[1] * user[2], tb[1] * user[3], tb[2] * user[0] + tb[3] * user[2] + user[4], tb[2] * user[1] + tb[3] * user[3] + user[5]];
        let m = self.brush_to_device(Some(t2b));
        let det = m[0] * m[3] - m[1] * m[2];
        let (w, h) = self.target.as_ref().map_or((0.0, 0.0), |t| (t.desc.width as f32, t.desc.height as f32));
        if det == 0.0 || !det.is_finite() || w == 0.0 {
            return [0, 0, 0, 0]; // nothing can be sampled
        }
        let inv = |x: f32, y: f32| {
            let (x, y) = (x - m[4], y - m[5]);
            [(m[3] * x - m[2] * y) / det, (m[0] * y - m[1] * x) / det]
        };
        let c = [inv(0.0, 0.0), inv(w, 0.0), inv(0.0, h), inv(w, h)];
        let span = |k: usize, n: f32| {
            let lo = (c.iter().map(|p| p[k]).fold(f32::INFINITY, f32::min).floor() - 2.0).clamp(0.0, n - 1.0);
            let hi = (c.iter().map(|p| p[k]).fold(f32::NEG_INFINITY, f32::max).ceil() + 2.0).clamp(lo + 1.0, n);
            (lo as i32, hi as i32)
        };
        let ((x0, x1), (y0, y1)) = (span(0, size[0]), span(1, size[1]));
        [x0, y0, x1, y1]
    }

    /// Replay a command list into a transparent `size` bitmap under the DIP→pixel map
    /// `[kx, ky, ox, oy]` (pixel = k·dip + o). Works outside BeginDraw and never records into an open
    /// command list.
    fn realize_list_px(&mut self, list: &CommandList, map: [f32; 4], size: [i32; 2]) -> Bitmap {
        let dpi = self.dpi;
        let (w, h) = (size[0].max(0) as u32, size[1].max(0) as u32);
        let saved = self.set_target_bitmap(bitmap_target(w, h, dpi.0));
        let recording = self.recording.take();
        let in_draw = core::mem::replace(&mut self.in_draw, true);
        let realizing = core::mem::replace(&mut self.realizing_list, true);
        let cmds = list.0.borrow().cmds.clone();
        // replay's transform is in DIPs and the device adds the DPI scale.
        let (sx, sy) = (96.0 / dpi.0, 96.0 / dpi.1);
        self.replay(&cmds, [map[0] * sx, 0.0, 0.0, map[1] * sy, map[2] * sx, map[3] * sy], dpi);
        self.realizing_list = realizing;
        self.in_draw = in_draw;
        self.recording = recording;
        let realized = self.take_target().expect("intermediate was just bound");
        if let Some(prev) = saved {
            self.set_target_bitmap(prev);
        }
        Bitmap { width: w, height: h, premul_bgra: realized.data, format: realized.desc.format, alpha: realized.desc.alpha_mode, dpi }
    }

    /// `brush→device` = current device transform (world∘DPI) composed with the brush transform,
    /// in f32 for the brush realizer (the device transform's integer scenes cast exactly).
    fn brush_to_device(&self, transform: Option<[f32; 6]>) -> [f32; 6] {
        let bt = transform.unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).map(|v| v as f64);
        compose(&self.device_transform(), &bt).map(|v| v as f32)
    }

    /// `CreateCommandList`: an open, device-stamped list.
    pub fn create_command_list(&self) -> CommandList {
        CommandList::new(self.domain)
    }

    // --- lifecycle ----------------------------------------------------------------------------

    /// `BeginDraw`: clear the error, enter the draw. Beginning while already drawing latches
    /// WRONG_STATE. Tags persist across pairs.
    pub fn begin_draw(&mut self) {
        self.effect_batch.clear();
        self.effect_arena.reset();
        if self.in_draw {
            self.process_new_error(WRONG_STATE);
            return;
        }
        self.error = S_OK;
        self.err_tags = (0, 0);
        self.in_draw = true;
        self.batch.begin_draw();
        self.mesh_bad.clear();
        self.mesh_bands.clear();
        self.rec_open = 0;
        self.end_error = None;
        self.atlas = SlowAtlas::default();
    }

    /// `EndDraw`: return `(hresult, tag1, tag2)` — the first latched error and the tags frozen with
    /// it (both 0 on success) — then reset error/tags/inDraw so the target may begin again. An open
    /// clip/layer maps to PUSH_POP_UNBALANCED; calling outside a draw is WRONG_STATE.
    pub fn end_draw(&mut self) -> (HRESULT, u64, u64) {
        self.effect_batch.clear();
        self.effect_arena.reset();
        self.batch.end();
        self.atlas = SlowAtlas::default();
        // ponytail: a Flush or CopyFromRenderTarget between may surface it earlier natively; unprobed.
        if let Some(hr) = self.end_error.take() {
            self.process_new_error(hr);
        }
        if !self.in_draw {
            self.process_new_error(WRONG_STATE);
        } else if !self.stack.is_empty() || self.rec_open != 0 {
            self.process_new_error(PUSH_POP_UNBALANCED);
        }
        self.rec_open = 0;
        let result = (self.error, self.err_tags.0, self.err_tags.1);
        self.error = S_OK;
        self.err_tags = (0, 0);
        self.in_draw = false;
        // Unbalanced stack: discard the open offscreens and rebind the original backdrop so the
        // target stays coherent (native latches PUSH_POP_UNBALANCED; pixels are undefined).
        while let Some(frame) = self.stack.pop() {
            self.target = Some(match frame {
                crate::layer::Frame::Clip { saved, .. } => saved,
                crate::layer::Frame::Layer { saved, .. } => saved,
            });
        }
        self.wipe_if_poisoned();
        result
    }

    /// `ID2D1Bitmap::CopyFromRenderTarget` from this context's own target into a fresh bitmap the size of
    /// `rect` (default: the whole target; at least 1×1): the target's bytes in `rect`, clipped to the
    /// target, the rest zero. It flushes first, so a latched draw error comes back (and stays latched);
    /// a clip or layer still pushed fails with D2DERR_RENDER_TARGET_HAS_LAYER_OR_CLIPRECT (r24 probes).
    /// Our draws are immediate, so the target is current without a flush.
    pub fn copy_from_target(&self, rect: Option<[u32; 4]>) -> Result<Vec<u8>, HRESULT> {
        if self.error != S_OK {
            return Err(self.error);
        }
        if !self.stack.is_empty() {
            return Err(HRESULT(0x8899_0017u32 as i32));
        }
        let Some(t) = &self.target else { return Err(WRONG_STATE) };
        let (tw, th) = (t.desc.width, t.desc.height);
        let [l, top, r, b] = rect.unwrap_or([0, 0, tw, th]);
        let (w, h) = (r.saturating_sub(l).max(1) as usize, b.saturating_sub(top).max(1) as usize);
        let bpp = crate::bitmap::bytes_per_pixel(t.desc.format).unwrap_or(4);
        let mut out = vec![0u8; w * h * bpp];
        let (cr, cb) = (r.min(tw), b.min(th));
        if l < cr {
            for y in top..cb {
                let src = &t.data[y as usize * t.stride + l as usize * bpp..y as usize * t.stride + cr as usize * bpp];
                let d = (y - top) as usize * w * bpp;
                out[d..d + src.len()].copy_from_slice(src);
            }
        }
        Ok(out)
    }

    /// `Flush`: report then clear the error/tags but keep the draw open. Flushing while not drawing,
    /// or while recording a command list, is WRONG_STATE.
    pub fn flush(&mut self) -> (HRESULT, u64, u64) {
        self.effect_batch.clear();
        self.effect_arena.reset();
        self.batch.end();
        self.atlas = SlowAtlas::default();
        if !self.in_draw || self.recording.is_some() {
            self.process_new_error(WRONG_STATE);
        }
        self.wipe_if_poisoned();
        let result = (self.error, self.err_tags.0, self.err_tags.1);
        self.error = S_OK;
        self.err_tags = (0, 0);
        result
    }

    /// After a FillMesh fault, wherever the frame's pixels become observable (EndDraw, Flush, a target
    /// switch) the bands whose job did not complete are transparent black, on the target and on every
    /// buffer a pushed clip or layer will restore or composite back from (native's clip is only a
    /// scissor: r13 18808). An idle worker pool completes the jobs taken ahead of the first faulting
    /// band; a warm one completes none ([`Self::warp_warm`]).
    // ponytail: a band the pool had not taken when the abort landed sometimes completes (a race:
    // repeated VM captures of one scene differ); not modelled.
    fn wipe_if_poisoned(&mut self) {
        if !self.mesh_bad.contains(&true) {
            return;
        }
        let mut keep = vec![false; self.mesh_bad.len()];
        if !self.warp_warm.get() {
            for &b in self.mesh_bands.iter().rev() {
                match keep.get_mut(b) {
                    Some(k) if !self.mesh_bad[b] => *k = true,
                    _ => break,
                }
            }
        }
        let wipe = |buf: &mut PixelBuffer| {
            let rows = buf.data.len() / buf.stride.max(1);
            for y in 0..rows {
                if !keep.get(y / 8).copied().unwrap_or(false) {
                    buf.data[y * buf.stride..(y + 1) * buf.stride].fill(0);
                }
            }
        };
        if let Some(buf) = &mut self.target {
            wipe(buf);
        }
        for f in &mut self.stack {
            let (crate::layer::Frame::Clip { saved, .. } | crate::layer::Frame::Layer { saved, .. }) = f;
            wipe(saved);
        }
    }

    /// A texture upload on the device (a bitmap created with pixels, a gradient's stop texture) runs
    /// WARP's worker pool, which changes how a later FillMesh fault ends ([`Self::warp_warm`]).
    pub fn note_texture_upload(&self) {
        self.warp_warm.set(true);
    }

    // --- error / tag model --------------------------------------------------------------------

    /// Native's first-error latch: store `hr` only if no error is latched yet, freezing the current
    /// tags into the error tags on that first error.
    pub(crate) fn process_new_error(&mut self, hr: HRESULT) {
        if self.error == S_OK && hr != S_OK {
            self.error = hr;
            self.err_tags = self.state.tags;
            // A list being recorded owns the first error latched while it is the target: native's
            // `ID2D1CommandList::Close` returns it (API-observable), though the context also surfaces
            // it at `EndDraw`. This is the first latch, so it is the one Close reports.
            if let Some(list) = &self.recording {
                list.borrow_mut().rec_error = Some(hr);
            }
        }
    }

    /// Latch `hr` as a draw error (surfaced at `EndDraw` with the current tags), for a caller that
    /// cannot proceed with a draw — e.g. the FFI layer when a bound resource won't resolve. Native's
    /// draw methods defer such errors to EndDraw rather than failing the void call, so a drop-in must
    /// latch, not silently draw nothing and return S_OK. Public wrapper over `process_new_error`.
    pub fn latch_error(&mut self, hr: HRESULT) {
        self.process_new_error(hr);
    }

    /// True if a draw may proceed. Not drawing latches WRONG_STATE; an already-latched error makes
    /// pixel-producing ops silent no-ops.
    pub(crate) fn ready_to_draw(&mut self) -> bool {
        if !self.in_draw {
            self.process_new_error(WRONG_STATE);
            return false;
        }
        self.error == S_OK
    }

    /// The owning factory's id — the key for its per-factory custom-effect registry (effects/custom.rs).
    pub(crate) fn factory_id(&self) -> u64 {
        self.domain.factory_id
    }

    /// [`check_domain`](Self::check_domain) for a pixel brush; a singular bitmap/image brush paints
    /// nothing, so its draw is skipped whatever the blend (fuzz 5701: text and fills, SOURCE_OVER
    /// and COPY alike).
    pub(crate) fn check_brush(&mut self, brush: &PixelBrush) -> bool {
        self.check_fill_brush(brush, false)
    }

    /// [`check_brush`](Self::check_brush) for a rectangle fill, opacity-mask fill or layer pop, where an all-zero
    /// HIGH_QUALITY_CUBIC brush matrix fails the draw wherever the clip leaves a pixel, whatever the draw covers (r27
    /// 32978; r27 32630: fills under an empty clip pass; VM: zero-matrix rect fills and pops). A rank-one matrix, or the
    /// same brush stroked or inked, does not fail (r23 28212 28571 28844, r27 32278 32487).
    pub(crate) fn check_fill_brush(&mut self, brush: &PixelBrush, zero_fails: bool) -> bool {
        if self.a8_brush_rejected(brush) {
            return false;
        }
        if let Some(hr) = self.brush_error(brush) {
            // A played-back draw fails as the brush is placed for the list's image, under the playing op's tags
            // (r27 32544: an image brush under a NaN offset).
            let tags = self.state.tags;
            self.state.tags = self.play_tags.unwrap_or(tags);
            self.process_new_error(hr);
            self.state.tags = tags;
            return false;
        }
        if !self.check_domain(brush.domain) {
            return false;
        }
        // A draw places the brush under its own transform (a recorded one where the list plays back).
        let placed = self.recording.is_none().then(|| self.realize_brush_for_draw(brush)).flatten();
        let brush = placed.as_ref().unwrap_or(brush);
        if brush.brush.singular() {
            if zero_fails && self.recording.is_none() && self.brush_need_too_large(brush, self.reach_rect()) {
                self.process_new_error(INTERMEDIATE_TOO_LARGE);
            }
            return false;
        }
        true
    }

    /// The error a draw with `brush` latches whatever it covers, if any. A brush that can't be placed fails where it
    /// is drawn; recorded into a command list, it fails where the list plays back instead (r27 32279: a
    /// HIGH_QUALITY_CUBIC image brush with an infinite translate used in a list, whose Close native leaves S_OK).
    fn brush_error(&self, brush: &PixelBrush) -> Option<HRESULT> {
        if self.recording.is_some() {
            return None;
        }
        // HIGH_QUALITY_CUBIC places a bitmap through the brush matrix under the draw's transform: a non-finite entry in
        // either fails (r27 32165: a 1e-6 brush drawn under an infinite translate), though their product overflowing
        // does not (r27 32705).
        // A list played back under a finite singular transform checks nothing (r27 32870: a zero matrix), nor one placed
        // at an infinite offset (r27 32073), though one under a NaN translation does (r27 32544).
        let m = self.device_transform();
        let flat = m[0] * m[3] - m[1] * m[2] == 0.0 || self.play_unplaced;
        if let Some(crate::brush::BrushDef::Bitmap { interpolation: 5, xform, .. }) = &brush.def.as_ref().filter(|_| self.replay_depth == 0 || !flat) {
            if !xform.iter().all(|v| v.is_finite()) || !self.state.transform.iter().all(|v| v.is_finite()) {
                return Some(E_INVALIDARG);
            }
        }
        brush.fail
    }

    /// Whether a draw with `brush` over the device rect `rect` needs an intermediate Direct2D can't allocate. A brush
    /// samples through an image-effect stage, which renders its image tile by tile as a transformed image does (see
    /// [`transformed_image_need`](Self::transformed_image_need)), when it is:
    /// - HIGH_QUALITY_CUBIC: the image is pre-scaled by the affine pre-scale factors (`hq_scales`, at least 0.01),
    ///   and the stage takes the rest of the transform;
    /// - ANISOTROPIC, through its mip intermediate ([`Brush::aniso_need_too_large`], over the pixels the rect touches);
    /// - an image brush over an effect or a cropped bitmap, whatever its interpolation.
    /// A singular matrix needs nothing (a zero ANISOTROPIC matrix samples the image's coarsest mip), except an all-zero
    /// HIGH_QUALITY_CUBIC one (see [`check_fill_brush`](Self::check_fill_brush)).
    pub(crate) fn brush_need_too_large(&self, brush: &PixelBrush, rect: [f32; 4]) -> bool {
        let crate::brush::Brush::Bitmap { transform: m, interpolation, origin, .. } = &brush.brush else { return false };
        let staged = *origin == crate::brush::BrushOrigin::StagedImage;
        // An all-zero HIGH_QUALITY_CUBIC matrix fails whatever its translation; any other brush that voids (singular or
        // non-finite; HIGH_QUALITY_CUBIC fails a non-finite one in `brush_error`) needs nothing (suite brsing-ib-tiny-b1).
        if m[..4] == [0.0; 4] {
            return *interpolation == 5 && self.draw_pixels(rect).is_some();
        }
        if brush.brush.singular() || !m.iter().all(|v| v.is_finite()) {
            return false;
        }
        let (m, limit) = match interpolation {
            5 => {
                let s = crate::brush::hq_scales(m);
                ([m[0] / s[0], m[1] / s[0], m[2] / s[1], m[3] / s[1], m[4], m[5]], 4096)
            }
            4 if !staged => {
                let c = self.reach_rect();
                let region = [rect[0].floor().max(c[0]), rect[1].floor().max(c[1]), rect[2].ceil().min(c[2]), rect[3].ceil().min(c[3])].map(|v| v as i32);
                return region[0] < region[2] && region[1] < region[3] && brush.brush.aniso_need_too_large(region);
            }
            // An effect image brush realized per draw has its own Border rule ([`Self::image_brush_border_too_large`]):
            // a CLAMP speck draws through the Border shader (suite imgbrush-fx-speck-*).
            _ if staged && brush.image.is_none() => (*m, 4096),
            _ => return false,
        };
        self.draw_pixels(rect).is_some_and(|reg| matches!(self.tile_need(m, reg, *interpolation, limit), ImageNeed::TooLarge))
    }

    /// `brush` with a bitmap brush's brush→device matrix taken under the current transform, as a draw places it.
    fn brush_under_current(&self, brush: &PixelBrush) -> PixelBrush {
        let mut b = brush.clone();
        if let Some(crate::brush::BrushDef::Bitmap { xform, dpi, size, interpolation, .. }) = &brush.def {
            if let crate::brush::Brush::Bitmap { transform, .. } = &mut b.brush { *transform = self.bitmap_placement(Some(*xform), *dpi); }
            if let Some(row) = self.bitmap_vs_row(Some(*xform), *dpi, *size, *interpolation) { b.brush.set_vs_row(row); }
        }
        b
    }

    /// Whether a draw with an image brush whose source rect runs past its image (a bitmap, or an effect's output bounds),
    /// over the device rect `rect`, needs only the source rect's transparent padding. Native realizes the image over the
    /// draw's need, and when that need lies within the source rect yet holds none of the image it skips the draw: it
    /// changes nothing, even under COPY, whatever the extend modes (VM probe-862e; r27 32862). A need that leaves the
    /// source rect draws: the border extends its edge (probe-862d, a rotated brush clamped past the rect's left edge).
    /// The need is the pixel centres the draw covers mapped back to atlas texels and grown by the filter's reach, as
    /// [`image_brush_for_draw`](Self::image_brush_for_draw) sizes it.
    pub(crate) fn brush_samples_padding_only(&self, brush: &PixelBrush, rect: [f32; 4]) -> bool {
        let Brush::Bitmap { content: Some(c), to_source: t, interpolation, bitmap, .. } = &brush.brush else { return false };
        let Some(reg) = self.draw_pixels(rect) else { return false };
        let [x0, y0, x1, y1] = reg.map(|v| v as f32);
        let p = [[x0 + 0.5, y0 + 0.5], [x1 - 0.5, y0 + 0.5], [x0 + 0.5, y1 - 0.5], [x1 - 0.5, y1 - 0.5]]
            .map(|[x, y]| [t[0] * x + t[2] * y + t[4], t[1] * x + t[3] * y + t[5]]);
        let reach = match interpolation { 0 => 0.53125, 2 => 2.0, _ => 1.0 };
        let need: [[f32; 2]; 2] = core::array::from_fn(|k| [
            (p.iter().map(|q| q[k]).fold(f32::INFINITY, f32::min) - reach + 0.5).floor(),
            (p.iter().map(|q| q[k]).fold(f32::NEG_INFINITY, f32::max) + reach - 0.5).ceil(),
        ]);
        let size = [bitmap.width as f32, bitmap.height as f32];
        let inside = (0..2).all(|k| need[k][0].is_finite() && need[k][1].is_finite() && need[k][0] >= 0.0 && need[k][1] <= size[k]);
        inside && (0..2).any(|k| need[k][1] <= c[k] as f32 || need[k][0] >= c[k + 2] as f32)
    }

    /// The pixels whose centres the device rect `rect` covers within the clip and target, if any.
    pub(crate) fn draw_pixels(&self, rect: [f32; 4]) -> Option<[i32; 4]> {
        let clip = self.reach_rect();
        let reg = [
            (rect[0] - 0.5).ceil().max(clip[0]),
            (rect[1] - 0.5).ceil().max(clip[1]),
            (rect[2] - 0.5).ceil().min(clip[2]),
            (rect[3] - 0.5).ceil().min(clip[3]),
        ];
        (reg[0] < reg[2] && reg[1] < reg[3]).then(|| reg.map(|v| v as i32))
    }

    /// Filling with a brush whose source is an A8 (alpha-only) bitmap is rejected: native latches
    /// WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT at the fill (not at brush creation — an unused A8 brush is
    /// fine, r23 28128 vs 28744) and draws nothing.
    pub(crate) fn a8_brush_rejected(&mut self, brush: &PixelBrush) -> bool {
        if let crate::brush::Brush::Bitmap { bitmap, .. } = &brush.brush {
            if bitmap.format.0 == 65 {
                self.process_new_error(UNSUPPORTED_PIXEL_FORMAT);
                return true;
            }
        }
        false
    }

    pub(crate) fn check_domain(&mut self, d: Domain) -> bool {
        if d.factory_id != self.domain.factory_id {
            self.process_new_error(WRONG_FACTORY);
            return false;
        }
        if d.device_id != self.domain.device_id {
            self.process_new_error(WRONG_RESOURCE_DOMAIN);
            return false;
        }
        true
    }

    /// `(factory_id, device_id)` a resource created on this context records, so a later draw of it on
    /// another context domain-checks correctly (the FFI bitmap path; brushes carry `Domain` directly).
    pub fn domain_ids(&self) -> (u64, u64) {
        (self.domain.factory_id, self.domain.device_id)
    }

    /// A fresh device context in the same resource domain (same factory + device). `CreateCompatible-
    /// RenderTarget` builds its offscreen target this way so the bitmap it hands back draws on the
    /// parent without latching WRONG_RESOURCE_DOMAIN, exactly as native's compatible target (which
    /// shares the parent's device) does.
    pub fn sibling(&self) -> DeviceContext {
        DeviceContext::new(self.domain)
    }

    /// Public [`check_domain`] for the FFI: latch WRONG_FACTORY / WRONG_RESOURCE_DOMAIN (deferred to
    /// EndDraw) and return false when a drawn resource's `(factory_id, device_id)` is not this domain.
    pub fn check_draw_domain(&mut self, factory_id: u64, device_id: u64) -> bool {
        self.check_domain(Domain { factory_id, device_id })
    }

    // --- state setters / getters --------------------------------------------------------------

    /// `SetTransform`. Any finite matrix is accepted. Takes effect on the next draw even when set
    /// outside a draw (plain field — the native deferred-batch replay is an optimization with no
    /// observable difference for the CPU path).
    pub fn set_transform(&mut self, m: [f32; 6]) {
        let _a = self.api(40, 0, None);
        self.state.transform = m;
    }
    pub fn get_transform(&self) -> [f32; 6] {
        self.state.transform
    }

    /// `SetAntialiasMode`: stored only when `mode <= 1`; otherwise silently ignored.
    pub fn set_antialias_mode(&mut self, mode: u32) {
        if mode <= 1 {
            self.state.aa = mode;
        }
    }
    pub fn get_antialias_mode(&self) -> u32 {
        self.state.aa
    }

    /// `SetTextAntialiasMode`: stored only when `mode <= 3`.
    pub fn set_text_antialias_mode(&mut self, mode: u32) {
        if mode <= 3 {
            let _a = self.api(24, 0, None);
            self.state.text_aa = mode;
        }
    }
    pub fn get_text_antialias_mode(&self) -> u32 {
        self.state.text_aa
    }

    /// The text antialias mode a glyph run draws with. Native resolves DEFAULT against the alpha mode of
    /// the surface drawn to, and inside a layer pushed with IGNORE_ALPHA that surface ignores alpha, so
    /// DEFAULT draws ClearType there. Elsewhere DEFAULT stays for the target to resolve.
    pub(crate) fn text_aa_for_draw(&self) -> u32 {
        let aa = self.state.text_aa;
        let layer_ignores_alpha = self.stack.iter().rev().find_map(|f| match f {
            crate::layer::Frame::Layer { params, .. } => Some(params.ignore_alpha),
            _ => None,
        });
        if aa == 0 && layer_ignores_alpha == Some(true) { 1 } else { aa }
    }

    /// `FillMesh` and `DrawSpriteBatch` require aliased antialias mode: native latches
    /// `D2DERR_WRONG_STATE` (the sticky error, surfaced at the next HRESULT-returning call and at
    /// `EndDraw` with the op's tags) and draws nothing in per-primitive mode. Returns whether the draw
    /// may proceed. (Native's `FillMesh` / `DrawSpriteBatch`: both
    /// `if (antialiasMode != ALIASED) error = 0x88990001`; `DrawGradientMesh` has NO such check.)
    pub fn aliased_draw_check(&mut self) -> bool {
        // The aliased-mode precondition latches WRONG_STATE even when an earlier inf-X FillMesh poisoned
        // the pipeline (7000-7999 rerun 7750). `in_draw` comes first (outside BeginDraw is WRONG_STATE).
        if !self.in_draw {
            self.process_new_error(WRONG_STATE);
            return false;
        }
        if self.state.aa != 1 {
            self.process_new_error(WRONG_STATE);
            return false;
        }
        if !self.ready_to_draw() {
            return false;
        }
        true
    }

    /// The latched draw error (`S_OK` if none), WITHOUT clearing it. Native keeps a sticky error state
    /// and returns it from subsequent HRESULT-returning methods (e.g. `CreateCommandList`) until
    /// `EndDraw`/`Flush` clears it, so a resource/command call after a failed draw fails too.
    pub fn current_error(&self) -> HRESULT {
        self.error
    }

    /// Fill mesh triangles flat with a solid brush, each one on its own through WARP's D3D triangle
    /// rasterizer (1/256-subpixel snap + top-left rule) — the aliased rule native's FillMesh uses, not
    /// the geometry scan converter (which places the triangle edge a pixel differently, VM-confirmed).
    /// Call only after [`aliased_draw_check`] passes.
    pub fn fill_mesh_solid(&mut self, tris: &[[f32; 2]; 3], brush: &SolidColorBrush) {
        // Recording, a command list keeps the triangle and fills it where it is drawn (r23 28478).
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            self.record(RecCmd::Mesh { state, dpi, tri: *tris, color: brush.color, opacity: brush.opacity });
            return;
        }
        let m = self.device_transform();
        let col = brush.color();
        let a = col[3] * brush.opacity();
        let rgba = [col[0] * a, col[1] * a, col[2] * a, a]; // premultiplied, full coverage
        let blend = blend_of(self.state.blend);
        let clip = self.mesh_clip();
        if let Some(buf) = &mut self.target {
            let (w, h) = (buf.desc.width as f32, buf.desc.height as f32);
            for t in mesh_triangles(&m, tris, w, h) {
                let xy = t.map(|v| [v[0], v[1]]);
                note_mesh_bands(&mut self.mesh_bands, &mut self.mesh_bad, t, clip, h as usize, true, |x, y| crate::warp::blend_px(buf, x, y, rgba, blend, true));
                if xy.iter().flatten().any(|v| !v.is_finite()) {
                    // A front-facing fan with a non-finite clipped position but finite clip values rasterizes
                    // at its converted positions in the solid colour (VM probe: r19 24242's fan off an
                    // infinite vertex, drawn solid). One whose clip values are NaN is not modelled here.
                    let q = cvt_nan_positions(t);
                    if t.iter().all(|v| !v[2].is_nan()) && crate::warp::setup_accepts(q) && crate::warp::exact_front_facing(q) {
                        let (p, area) = crate::warp::snap_tri(q);
                        crate::warp::cover_tri(p, area, w as i64, h as i64, |x, y| crate::warp::blend_px(buf, x, y, rgba, blend, true));
                    }
                    continue;
                }
                // SetupTriangle culls a back face: AddTriangles' front-facing swap keys on an f32 cross that
                // overflows to NaN on a huge vertex and swaps, so the fan can come out back-facing (r18 23314).
                if !crate::warp::setup_accepts(xy) {
                    continue;
                }
                let v: [crate::warp::Vert; 3] = core::array::from_fn(|k| crate::warp::Vert { pos: xy[k], rgba });
                crate::warp::fill_triangle(buf, v, blend, true);
            }
        }
    }

    /// The pixel columns and rows `[x0, y0, x1, y1)` a mesh draw rasterizes: the target under every
    /// pushed clip and layer, as WARP's scissor (pixel centres inside). A layer's is its device pixel
    /// rect, cut to its geometric mask's bounds when it has one.
    /// Record where a sprite triangle (`[x, y, 1]` device px) with a non-finite corner faults WARP's
    /// rasterizer ([`note_mesh_bands`]), after WARP's viewport clip; it paints nothing. Sprites are not
    /// culled, so a back face is set up with its winding reversed.
    pub(crate) fn note_fault_triangle(&mut self, t: [[f32; 3]; 3]) {
        let clip = self.mesh_clip();
        let Some(buf) = &self.target else { return };
        let (w, h) = (buf.desc.width as f32, buf.desc.height as f32);
        let tris = match crate::warp::clip_triangle(t, [0.0, 0.0, w, h]) {
            crate::warp::Clipped::Rejected => Vec::new(),
            crate::warp::Clipped::Inside => vec![t],
            crate::warp::Clipped::Fan(f) => f,
        };
        for t in tris {
            let t = if crate::warp::setup_accepts(cvt_nan_positions(t)) { t } else { [t[0], t[2], t[1]] };
            note_mesh_bands(&mut self.mesh_bands, &mut self.mesh_bad, t, clip, h as usize, false, |_, _| {});
        }
    }

    fn mesh_clip(&self) -> [i64; 4] {
        let (w, h) = self.target.as_ref().map_or((0, 0), |t| (t.desc.width as i64, t.desc.height as i64));
        let centre = |v: f32| if v.is_nan() { 0 } else { (v as f64 - 0.5).ceil().clamp(-1e15, 1e15) as i64 };
        self.stack.iter().fold([0, 0, w, h], |r, f| {
            let b = match f {
                crate::layer::Frame::Clip { rect, .. } => rect.map(centre),
                crate::layer::Frame::Layer { params, .. } => match (&params.native, params.mask_bounds) {
                    (Some(n), _) if n.nan => [0; 4],
                    (Some(n), _) => n.rect.map(i64::from),
                    (None, Some(m)) => {
                        let b = params.bounds;
                        [b[0].max(m[0]), b[1].max(m[1]), b[2].min(m[2]), b[3].min(m[3])].map(centre)
                    }
                    (None, None) => params.bounds.map(centre),
                },
            };
            [r[0].max(b[0]), r[1].max(b[1]), r[2].min(b[2]), r[3].min(b[3])]
        })
    }

    /// Fill one mesh triangle with a gradient/bitmap/image brush, through the *same* WARP D3D
    /// triangle rule as [`fill_mesh_solid`] (1/256 snap + top-left) with a per-pixel brush source —
    /// the aliased, per-triangle rule native's `FillMesh` uses for every brush. Call only after
    /// [`aliased_draw_check`] passes.
    pub fn fill_mesh_brush(&mut self, tris: &[[f32; 2]; 3], brush: &PixelBrush) {
        // FillMesh validates its brush like any other fill (as fill_rectangle_brush does): an A8
        // (alpha-only) image brush latches WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT, and native runs this check
        // even while RECORDING a command list, so the error tags to the nested fill_mesh index (r23 28733:
        // the A8 brush inside a recorded list latches at the list's nested tag, not a later draw).
        // check_brush also skips a singular brush, replacing the old singular-only guard; the mesh then
        // records (replay re-realizes the brush and re-enters this with recording off, rerunning the check).
        if !self.check_brush(brush) {
            return;
        }
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            self.record(RecCmd::MeshBrush { state, dpi, tri: *tris, brush: brush.clone() });
            return;
        }
        // Gradient brushes realize lazily per draw under the current device transform (as fills/strokes).
        let grad_rr = self.realize_brush_for_draw(brush);
        let brush = grad_rr.as_ref().unwrap_or(brush);
        let m = self.device_transform();
        // A cropped bitmap image brush is realized for the draw over the triangle's device bounds, as a fill's is (VM: an
        // ANISOTROPIC mesh fill's texcoord plane is normalized over the bitmap's own 16 texels, not its 24-texel tile).
        let per_draw = match (brush.crop.is_some(), &self.target) {
            (true, Some(t)) => {
                let q = tris.map(|p| apply(&m, p[0], p[1]));
                let lo = |k: usize| q.iter().map(|p| p[k]).fold(f32::INFINITY, f32::min);
                let hi = |k: usize| q.iter().map(|p| p[k]).fold(f32::NEG_INFINITY, f32::max);
                let d = [lo(0).floor().max(0.0), lo(1).floor().max(0.0), hi(0).ceil().min(t.desc.width as f32), hi(1).ceil().min(t.desc.height as f32)];
                d.iter().all(|v| v.is_finite()).then(|| self.image_brush_for_draw(brush, d.map(|v| v as i32), false, true)).flatten().map(|mut pb| { pb.brush.keep_cell_quad(false); pb })
            }
            _ => None,
        };
        let brush = per_draw.as_ref().unwrap_or(brush);
        let blend = blend_of(self.state.blend);
        let clip = self.mesh_clip();
        if let Some(buf) = &mut self.target {
            let (w, h) = (buf.desc.width as f32, buf.desc.height as f32);
            // The brush is evaluated at the mesh triangle's own vertices and the frustum clip carries those
            // values onto each piece with its position, so a piece cut from a far vertex takes the planes
            // fitted from them (VM probes: FillMesh triangles reaching ±1e9 or 3.4e38 under radial, bitmap
            // and linear brushes; 23213's linear pieces match native's planes bit for bit).
            let setup = mesh_setup_triangle(&m, tris);
            let attrs = crate::brush::brush_vertex_attrs(setup.map(|v| [v[0], v[1]]), &brush.brush)
                .filter(|_| setup.iter().flatten().all(|v| v.is_finite()));
            let tri: [[f32; 5]; 3] = core::array::from_fn(|i| {
                let a = attrs.map_or([0.0; 2], |a| a[i]);
                [setup[i][0], setup[i][1], 1.0, a[0], a[1]]
            });
            // An unclipped triangle fits its planes from the provoking (last) vertex, a clipped piece from
            // its fan vertex.
            let (pieces, order) = match crate::warp::clip_triangle(tri, [0.0, 0.0, w, h]) {
                crate::warp::Clipped::Rejected => (Vec::new(), [0, 1, 2]),
                crate::warp::Clipped::Inside => (vec![tri], [2, 0, 1]),
                crate::warp::Clipped::Fan(f) => (f, [0, 1, 2]),
            };
            let nan_faults = !matches!(brush.brush, crate::brush::Brush::Bitmap { origin, .. } if !matches!(origin, crate::brush::BrushOrigin::PlainBitmap));
            for piece in pieces {
                let t = piece.map(|v| [v[0], v[1], v[2]]);
                let xy = t.map(|v| [v[0], v[1]]);
                // A brush's inverted quads take its sample at the clipped vertex's NaN texcoord.
                note_mesh_bands(&mut self.mesh_bands, &mut self.mesh_bad, t, clip, h as usize, nan_faults, |x, y| crate::brush::fill_px_brush_nan_texcoord(buf, x, y, &brush.brush, blend));
                if xy.iter().flatten().any(|v| !v.is_finite()) {
                    // A front-facing fan with a non-finite clipped vertex still rasterizes at its converted
                    // positions. The brush values clipped onto that vertex come out NaN (a NaN clip factor, or
                    // zero times the far vertex's infinite texcoord), so it samples at a NaN texcoord (r14
                    // 19786; r19 24242's fan off an infinite vertex).
                    let q = cvt_nan_positions(t);
                    if crate::warp::setup_accepts(q) && crate::warp::exact_front_facing(q) {
                        crate::brush::fill_triangle_brush_nan_texcoord(buf, q, &brush.brush, blend);
                    }
                    continue;
                }
                if !crate::warp::setup_accepts(xy) {
                    continue; // a back face (see `fill_mesh_solid`)
                }
                if attrs.is_some() {
                    crate::brush::fill_mesh_piece_brush(buf, piece, order, &brush.brush, blend);
                } else {
                    crate::brush::fill_triangle_brush(buf, xy, &brush.brush, blend);
                }
            }
        }
    }

    /// `SetPrimitiveBlend`: stored only when `mode <= 4`.
    pub fn set_primitive_blend(&mut self, mode: u32) {
        if mode <= 4 {
            let _a = self.api(24, 0, None);
            self.state.blend = mode;
        }
    }
    pub fn get_primitive_blend(&self) -> u32 {
        self.state.blend
    }

    /// `SetUnitMode`: stored only when `mode <= 1`.
    pub fn set_unit_mode(&mut self, mode: u32) {
        if mode <= 1 {
            let _a = self.api(24, 0, None);
            self.state.unit = mode;
        }
    }
    pub fn get_unit_mode(&self) -> u32 {
        self.state.unit
    }

    /// `SetDpi`. `(0,0)` resets to the default (96 — no desktop factory here); any other pair with
    /// both components `> 0` is stored; a negative/mixed pair is dropped.
    pub fn set_dpi(&mut self, x: f32, y: f32) {
        if x == 0.0 && y == 0.0 {
            self.dpi = (96.0, 96.0);
        } else if x > 0.0 && y > 0.0 {
            self.dpi = (x, y);
        }
    }
    pub fn get_dpi(&self) -> (f32, f32) {
        self.dpi
    }

    /// `SetTags` / `GetTags`: the current tags (not the error tags).
    pub fn set_tags(&mut self, tag1: u64, tag2: u64) {
        let _a = self.api(32, 0, None);
        self.state.tags = (tag1, tag2);
    }
    pub fn get_tags(&self) -> (u64, u64) {
        self.state.tags
    }

    pub fn is_in_draw(&self) -> bool {
        self.in_draw
    }

    // --- drawing-state blocks -----------------------------------------------------------------

    /// `SaveDrawingState`: copy this context's state into `block`. Silent no-op if the block belongs
    /// to a different factory.
    pub fn save_drawing_state(&self, block: &mut DrawingStateBlock) {
        if block.factory_id == self.domain.factory_id {
            block.state = self.state;
        }
    }

    /// `RestoreDrawingState`: apply `block`'s state back. Silent no-op across factories. The block
    /// only ever holds values its setters already validated, so a direct assignment matches
    /// restoring field by field.
    pub fn restore_drawing_state(&mut self, block: &DrawingStateBlock) {
        if block.factory_id == self.domain.factory_id {
            self.state = block.state;
        }
    }

    // --- push / pop (phase 2A: clips, layers, composition) ------------------------------------
    // A push swaps the bound target for a working offscreen — a backdrop copy for a clip (so draws
    // blend against the real target) or transparent for a layer (group isolation) — and the matching
    // pop composites it back (`crate::layer`). The stack still carries the 1A balance rules so
    // EndDraw reports PUSH_POP_UNBALANCED and a mismatched Pop reports POP_DID_NOT_MATCH_PUSH.

    /// `PushAxisAlignedClip(rect, antialiasMode)`. `rect` is normalised and mapped through the device
    /// transform (axis-aligned under scale/translate; a rotated transform would need a quad clip, not
    /// reached by the 2A scenes). `antialias > 1` ⇒ `E_INVALIDARG`.
    pub fn push_axis_aligned_clip(&mut self, rect: [f32; 4], antialias: u32) {
        let a = self.api(40, 0, None);
        if !self.in_draw {
            self.process_new_error(WRONG_STATE);
            return;
        }
        if antialias > 1 {
            self.process_new_error(E_INVALIDARG);
            return;
        }
        // Recorded into a command list: replayed under the playback transform (clclip-* probes). The live
        // clip stack is untouched (r23 28711, 28261: a recorded push left a stray frame there).
        if let Some(list) = self.recording.clone() {
            let (mut state, dpi) = self.rec_env();
            // The clip is axis-aligned where it is recorded: the box of the rect's corners under the
            // recording transform, which only the playback transform may then turn (clclip p2/p3).
            let w = state.transform;
            let c = [[rect[0], rect[1]], [rect[2], rect[1]], [rect[0], rect[3]], [rect[2], rect[3]]]
                .map(|[x, y]| [w[0] * x + w[2] * y + w[4], w[1] * x + w[3] * y + w[5]]);
            let lo = |k: usize| c.iter().map(|p| p[k]).fold(f32::INFINITY, f32::min);
            let hi = |k: usize| c.iter().map(|p| p[k]).fold(f32::NEG_INFINITY, f32::max);
            state.transform = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
            let rect = [lo(0), lo(1), hi(0), hi(1)];
            self.record(RecCmd::PushClip { state, dpi, rect, antialias });
            list.borrow_mut().frames.push(true);
            return;
        }
        // A clip pushed under a singular turn or skew is the box of the rect's four corners, not empty: native
        // keeps drawing inside that box, aliased or antialiased (VM probes on r19 24013's singular matrix).
        let m = self.device_transform();
        let dev = if m[1] != 0.0 || m[2] != 0.0 { self.device_corner_box(rect) } else { self.device_clip_rect(rect) };
        let finite = rect.iter().all(|v| v.is_finite()) && self.device_transform().iter().all(|v| v.is_finite());
        let corner = self.target.as_ref().filter(|_| finite).and_then(|t| {
            crate::layer::degenerate_clip_pixel(dev, antialias == 1, t.desc.width as i32, t.desc.height as i32)
        });
        let saved = self.swap_working(true);
        self.stack.push(crate::layer::Frame::Clip { saved, rect: dev, aliased: antialias == 1, min_rule: Vec::new(), corner });
        // An antialiased clip copies each fractional edge's strip of the target, then sets the clip.
        let edges = if antialias == 1 { [false; 4] } else { self.clip_strip_edges(dev) };
        for _ in edges.iter().filter(|&&e| e) {
            self.queue(&a, crate::batch::Prim::Mark);
        }
        self.queue(&a, crate::batch::Prim::Mark);
    }

    /// Which edges of an antialiased clip's device rect `r` have a strip, left, right, top and bottom: a fractional edge
    /// inside the target.
    // ponytail: d2d1 also skips an edge on its parent clip's; not modelled.
    fn clip_strip_edges(&self, r: [f32; 4]) -> [bool; 4] {
        let Some(t) = self.target.as_ref() else { return [false; 4] };
        let (w, h) = (t.desc.width as f32, t.desc.height as f32);
        if !(r[2] > r[0] && r[3] > r[1]) {
            return [false; 4];
        }
        let strip = |v: f32, n: f32| v.is_finite() && v.fract() != 0.0 && v > 0.0 && v < n;
        [strip(r[0], w), strip(r[2], w), strip(r[1], h), strip(r[3], h)]
    }

    /// A primitive the API call `a` queues on the target ([`Self::api`]).
    pub(crate) fn queue(&mut self, a: &Api, p: crate::batch::Prim) {
        if a.1 {
            let f = self.batch.prim(p);
            self.rendered(f);
        }
    }

    pub fn pop_axis_aligned_clip(&mut self) {
        use crate::batch::Prim;
        let a = self.api(16, 0, None);
        let top = match self.stack.last() {
            Some(crate::layer::Frame::Clip { rect, aliased, .. }) => Some((*rect, *aliased)),
            _ => None,
        };
        let depth = self.stack.len();
        self.pop_clip_or_layer(true);
        let Some((r, aliased)) = top.filter(|_| self.stack.len() < depth) else { return };
        // The parent clip set again, then an antialiased clip's edge strips redrawn, left, right, top, bottom, then the
        // corners between them.
        self.queue(&a, Prim::Mark);
        let e = if aliased { [false; 4] } else { self.clip_strip_edges(r) };
        for k in (0..4).filter(|&k| e[k]) {
            self.queue(&a, Prim::Strip(k < 2));
        }
        for v in &e[..2] {
            for h in &e[2..] {
                if *v && *h {
                    self.queue(&a, Prim::Corner);
                }
            }
        }
    }

    /// `PushLayer(D2D1_LAYER_PARAMETERS1, layer?)`: redirect drawing into an offscreen and capture the
    /// composite-back parameters — content bounds (device space), constant opacity, the geometric-mask
    /// coverage (the mask filled at `mask_aa` under `maskTransform × world`) and the opacity-brush
    /// per-pixel alpha. `layer_options` (LAYER_OPTIONS1): bit 1
    /// INITIALIZE_FROM_BACKGROUND starts the offscreen from the backdrop instead of transparent; bit 2
    /// IGNORE_ALPHA treats the layer as opaque at composite-back (native `CBaseRenderTarget::PushLayer`).
    #[allow(clippy::too_many_arguments)]
    pub fn push_layer(
        &mut self,
        bounds: [f32; 4],
        mask: Option<&crate::geometry::Geometry>,
        mask_aa: u32,
        mask_transform: Option<[f32; 6]>,
        opacity: f32,
        opacity_brush: Option<&PixelBrush>,
        layer_options: u32,
    ) {
        let a = self.api(104, 0, None);
        let depth = self.stack.len();
        self.push_layer_as(bounds, mask, mask_aa, mask_transform, opacity, opacity_brush, layer_options, false);
        // Its begin, a copy and its clip, then its clear unless it starts from the background.
        if self.stack.len() > depth && opacity_brush.is_none() {
            for _ in 0..3 {
                self.queue(&a, crate::batch::Prim::Mark);
            }
            if layer_options & 1 == 0 {
                self.queue(&a, crate::batch::Prim::LayerFill(1));
            }
        }
    }

    /// [`Self::push_layer`]; `clip` marks the layer Direct2D makes of an axis-aligned clip replayed under
    /// a rotation or skew, which pops differently (`LayerParams::init_quad`).
    #[allow(clippy::too_many_arguments)]
    fn push_layer_as(
        &mut self,
        bounds: [f32; 4],
        mask: Option<&crate::geometry::Geometry>,
        mask_aa: u32,
        mask_transform: Option<[f32; 6]>,
        opacity: f32,
        opacity_brush: Option<&PixelBrush>,
        layer_options: u32,
        clip: bool,
    ) {
        if !self.in_draw {
            self.process_new_error(WRONG_STATE);
            return;
        }
        // A mask antialias mode past ALIASED fails the push with E_INVALIDARG (r27).
        if mask_aa > 1 {
            self.process_new_error(E_INVALIDARG);
        }
        // While recording, keep the DIP inputs — replay re-resolves the bounds/mask under the replay
        // transform (native records the layer device-independently).
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            self.record(RecCmd::PushLayer {
                state,
                dpi,
                bounds,
                mask: mask.cloned(),
                mask_aa,
                mask_transform,
                opacity,
                opacity_brush: opacity_brush.cloned(),
                layer_options,
            });
            if let Some(list) = &self.recording {
                list.borrow_mut().frames.push(false);
            }
            return;
        }
        let (w, h) = self.target.as_ref().map(|t| (t.desc.width, t.desc.height)).unwrap_or((0, 0));
        let dm = self.device_transform();
        // The pop clamps the layer opacity to the target's range, [0, 1] on an 8bpc target, ±65504 on a
        // half-float one, ±FLT_MAX on a float one, a NaN taking the top (r27 32477: 1.5 composites as 1).
        let opacity = {
            let (lo, hi) = match self.target.as_ref().map(|t| t.desc.format.0) {
                Some(10) => (-65504.0, 65504.0),
                Some(2) => (-f32::MAX, f32::MAX),
                _ => (0.0, 1.0),
            };
            if opacity <= hi { opacity.max(lo) } else { hi }
        };
        // A geometric mask makes the layer's clip geometry the mask itself — when its own bounds (under the
        // mask transform) already lie inside the content bounds, are empty, or the content bounds are
        // unbounded on both ends of an axis — or else the mask cut by the content bounds, the cut holding
        // infinite bounds to ±FLT_MAX (fuzz r19 24248: an infinite right edge still cuts finitely).
        let mask_only = mask.is_some_and(|g| {
            let [l, t, r, b] = bounds;
            let unbounded = l <= -f32::MAX && r >= f32::MAX || t <= -f32::MAX && b >= f32::MAX;
            let mt = mask_transform.unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
            let [ml, mtop, mr, mb] = g.get_bounds(&mt);
            unbounded || ml >= mr || mtop >= mb || (ml >= l && mtop >= t && r >= mr && b >= mb)
        });
        let bounds = if mask.is_some() { bounds.map(|v| v.clamp(-f32::MAX, f32::MAX)) } else { bounds };
        // A rotated/sheared/singular device transform turns the layer bounds into a non-axis-aligned
        // (or zero-area) quad; native clips the layer to that transformed quad as geometry, not to its
        // AABB (fuzz 1254/983: a collinear det-0 transform cleared the whole AABB box instead of the
        // thin line). Rasterise the quad to a content-bounds coverage mask and keep `bounds` as its
        // AABB (just the loop extent). Axis-aligned stays `layer_bounds_cova`.
        let (dev_bounds, bounds_mask) = if dm[1] != 0.0 || dm[2] != 0.0 || dm[0] * dm[3] - dm[1] * dm[2] == 0.0 {
            let cs = [apply(&dm, bounds[0], bounds[1]), apply(&dm, bounds[2], bounds[1]), apply(&dm, bounds[2], bounds[3]), apply(&dm, bounds[0], bounds[3])];
            let (xs, ys) = (cs.map(|c| c[0]), cs.map(|c| c[1]));
            let aabb = [
                xs.iter().copied().fold(f32::INFINITY, f32::min),
                ys.iter().copied().fold(f32::INFINITY, f32::min),
                xs.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                ys.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            ];
            let cov = if mask_aa == 1 && cs.iter().flatten().all(|v| v.is_finite()) {
                crate::layer::aliased_quad_cov(&cs, w as usize, h as usize)
            } else {
                self.render_mask(&crate::geometry::Geometry::Rect(bounds), mask_aa, &dm, w, h)
            };
            (aabb, Some((cov, cs)))
        } else {
            (self.device_rect(bounds), None)
        };
        let (bounds_mask, bounds_quad) = bounds_mask.map_or((None, None), |(m, q)| (Some(m), Some(q)));
        let mt = mask_transform.unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mask_mat = compose(&self.device_transform(), &mt.map(|v| v as f64));
        // The clip geometry is the mask, or else the mask under its transform strip-clipped to the
        // content bounds (sorted per axis): one geometry, so an edge pixel both cross takes the inner
        // edge's coverage, not the product of the two (fuzz r20 25292).
        // A geometry mask pushed under the COPY blend is filled as a COPY fill is, by the trapezoid walk; the blend at the
        // pop does not matter (probe-cmk; r27 32114).
        let copy = self.state.blend == 1;
        let mask_cov = mask.map(|g| {
            if mask_only || g.any_nonfinite() {
                self.render_mask_in(g, mask_aa, &mask_mat, w, h, copy)
            } else if bounds.iter().any(|v| v.is_nan()) {
                vec![0; w as usize * h as usize] // a NaN bound leaves the layer no pixel rect (r16 21737)
            } else {
                // Each bound is held to ±FLT_MAX, then each axis sorts. Direct2D cuts the mask in layer space,
                // antialiased or aliased, flattening only a curve the cut crosses (`Geometry::clip_to_rect`), so
                // the strip clipper's tolerance is in DIPs whatever the device transform (r27 32360, 32069, 32923;
                // mask-cut VM probes at dpi 48 to 192, under a scale and a rotation).
                let [l, t, r, b] = bounds.map(|v| v.min(f32::MAX).max(-f32::MAX));
                let clip = g.clip_to_rect(&mt, [l.min(r), t.min(b), l.max(r), t.max(b)]);
                self.render_mask_in(&clip, mask_aa, &dm, w, h, copy)
            }
        });
        let mask_bounds = mask.filter(|g| !g.any_nonfinite()).map(|g| g.get_bounds(&mask_mat.map(|v| v as f32)));
        // With an opacity brush, native folds the constant opacity into the brush's per-pixel alpha
        // in float (one round) rather than applying it as a second UNORM8 multiply — two integer
        // rounds inflate the product by a LSB (`15-layer-mask`). Without a brush it stays the
        // standalone opacity factor (`layers-03`). The geometric mask is always a separate multiply.
        // A gradient opacity brush is realized at the push, under the transform and DPI then, as a fill
        // realizes its brush at the draw (r24 29105: SetDpi(120) between creating the brush and the push).
        let realized = opacity_brush.and_then(|b| self.realize_brush_for_draw(b));
        let opacity_brush = realized.as_ref().or(opacity_brush);
        let brush_opacity = opacity;
        // A singular bitmap or image opacity brush paints nothing, and the pop then fills the layer in one of four ways
        // (VM probe-spop: every brush kind, NEAREST to ANISOTROPIC, zero and rank-one matrices, opacity 1 and 0.5):
        // - ANISOTROPIC samples the image's coarsest mip, a constant alpha, and fades as usual;
        // - at an opacity other than 1 the pop draws nothing: the layer keeps its content on the cleared layer;
        // - at opacity 1, CUBIC fades the content out and puts the backdrop back (as does HIGH_QUALITY_CUBIC,
        //   unprobed), a plain NEAREST or LINEAR bitmap brush draws nothing, and the rest put the content over the
        //   backdrop unfaded (r27 32330).
        // The pop fills the layer's pixel rect (or its content bounds within its mask's bounds) with the brush.
        let native = mask.is_none().then(|| self.native_layer_clip(bounds, &dm, w, h, bounds_mask.is_none(), mask_aa));
        let pop_region = match (&native, mask_bounds) {
            (Some(n), _) if !n.nan => n.rect.map(|v| v as f32),
            (Some(_), _) => [0.0; 4],
            (None, Some(m)) => [dev_bounds[0].max(m[0]), dev_bounds[1].max(m[1]), dev_bounds[2].min(m[2]), dev_bounds[3].min(m[3])],
            (None, None) => dev_bounds,
        };
        // An effect image opacity brush is realized for the pop over the pop region, as for any draw: with a geometric
        // mask that region is the mask's bounds, and the creation-time realization samples a hair off (r27 32095).
        let realized = opacity_brush.filter(|b| b.image.is_some() || b.crop.is_some()).and_then(|b| {
            let r = pop_region;
            let d = [r[0].floor().max(0.0) as i32, r[1].floor().max(0.0) as i32, r[2].ceil().min(w as f32) as i32, r[3].ceil().min(h as f32) as i32];
            self.image_brush_for_draw(b, d, false, false).map(|mut pb| { pb.brush.keep_cell_quad(false); pb })
        });
        let (mut brush_alpha, brush_alpha_f) = realized.as_ref().or(opacity_brush).map(|b| self.opacity_alpha(&b.brush, opacity, w, h, layer_options & 2 != 0, pop_region)).unzip();
        let mut brush_alpha_f = brush_alpha_f.flatten();
        // A brush whose pop samples only an image crop's padding paints nothing there too (r27 32862).
        let paints_nothing = |b: &PixelBrush| b.brush.singular() || self.brush_samples_padding_only(b, pop_region);
        let (mut void_pop, mut keep_pop) = (false, false);
        if let Some(crate::brush::Brush::Bitmap { interpolation, origin, bitmap, .. }) = opacity_brush.filter(|b| paints_nothing(b)).map(|b| &b.brush) {
            match interpolation {
                4 => {
                    let n = (bitmap.width * bitmap.height).max(1) as f32;
                    let mean = bitmap.premul_bgra.chunks_exact(4).map(|p| p[3] as f32).sum::<f32>() / n / 255.0;
                    let a = (mean * opacity).clamp(0.0, 1.0);
                    brush_alpha = Some(vec![(a * 255.0).round_ties_even() as u8; w as usize * h as usize]);
                    brush_alpha_f = brush_alpha_f.map(|v| vec![a; v.len()]);
                }
                _ if opacity != 1.0 && *interpolation < 5 => keep_pop = true,
                2 | 5 => void_pop = true,
                0 | 1 if *origin == crate::brush::BrushOrigin::PlainBitmap => keep_pop = true,
                _ => (brush_alpha, brush_alpha_f) = (None, None),
            }
        }
        let (opacity8, opacity) = if opacity_brush.is_some() {
            (255, 1.0)
        } else {
            ((opacity * 255.0).round_ties_even().clamp(0.0, 255.0) as i32, opacity)
        };
        let pop_brush = opacity_brush.and_then(|b| {
            let r = pop_region;
            // A pop that fills no pixel checks nothing (r27 32279: a failing brush under an infinite transform).
            self.draw_pixels(r)?;
            let xform = match &b.def { Some(crate::brush::BrushDef::Bitmap { xform, .. }) => Some(*xform), _ => None };
            Some(crate::layer::PopBrush { brush: b.clone(), region: r, b2d: self.brush_to_device(xform), unit: self.state.unit, opacity: brush_opacity })
        });
        let params = crate::layer::LayerParams {
            native,
            bounds: dev_bounds,
            bounds_mask,
            init_quad: bounds_quad.filter(|_| clip || copy),
            bounds_nan: bounds.iter().any(|v| v.is_nan()),
            void_pop,
            keep_pop,
            init: layer_options & 1 != 0,
            pop_brush,
            opacity8,
            opacity,
            mask: mask_cov,
            mask_bounds,
            brush_alpha,
            brush_alpha_f,
            ignore_alpha: layer_options & 2 != 0,
            aliased: mask_aa == 1,
            inverted: bounds[0] > bounds[2] || bounds[1] > bounds[3],
            blend: blend_of(self.state.blend),
            // The degenerate-corner pass (this branch only runs for a mask-less layer) is NOT gated on the
            // mask antialias mode: native passes a mask-less zero-width/height layer's device-floor corner
            // through the degenerate flatten regardless of that mode, carrying the 1px-clipped content there
            // (r27 32446: a zero-width layer with mask_antialias ALIASED whose corner still takes the content).
            corner: (mask.is_none() && bounds_quad.is_none() && bounds.iter().chain(dm.iter().map(|v| *v as f32).collect::<Vec<_>>().iter()).all(|v| v.is_finite()))
                .then(|| crate::layer::degenerate_clip_pixel(dev_bounds, false, w as i32, h as i32)).flatten(),
            corner_live: false,
        };
        let init = layer_options & 1 != 0;
        let in_ignore = self.stack.iter().rev().find_map(|f| match f {
            crate::layer::Frame::Layer { params, .. } => Some(params.ignore_alpha),
            _ => None,
        }) == Some(true);
        let saved = self.swap_working(init);
        // An INIT layer inside an IGNORE_ALPHA one starts from that surface read as opaque (r17 22261).
        if init && in_ignore {
            if let Some(t) = self.target.as_mut().filter(|t| !crate::float::is_highp(t.desc.format)) {
                t.data.chunks_exact_mut(4).for_each(|p| p[3] = 255);
            }
        }
        // Under IGNORE_ALPHA the push's clear of the layer rect writes opaque black, so the content
        // draws over an opaque destination (a destination-alpha composite inside sees alpha 1).
        if layer_options & 3 == 2 {
            if let Some(t) = self.target.as_mut().filter(|t| !crate::float::is_highp(t.desc.format)) {
                t.data.chunks_exact_mut(4).for_each(|p| p[3] = 255);
            }
        }
        self.stack.push(crate::layer::Frame::Layer { saved, params });
    }

    pub fn pop_layer(&mut self) {
        use crate::batch::Prim;
        let a = self.api(32, 0, None);
        let top = match self.stack.last() {
            Some(crate::layer::Frame::Layer { params, .. }) if params.pop_brush.is_none() => Some(params.opacity != 1.0 || params.mask.is_some()),
            _ => None,
        };
        let depth = self.stack.len();
        self.pop_clip_or_layer(false);
        // Its clip, its draw-back (an opacity or a mask first fills once more), its end and the clip restored.
        let Some(twice) = top.filter(|_| self.stack.len() < depth) else { return };
        self.queue(&a, Prim::Mark);
        if twice {
            self.queue(&a, Prim::LayerFill(1));
        }
        for p in [Prim::LayerFill(2), Prim::Mark, Prim::Mark] {
            self.queue(&a, p);
        }
    }

    /// Direct2D's setup of a mask-less layer (see [`crate::layer::NativeClip`]). Bounds unbounded on
    /// both ends of one axis (at or past ±FLT_MAX) need no bounds geometry: the raw, untransformed bounds
    /// are the device rect. Otherwise the bounds become a rectangle geometry: its device bounds give the
    /// rect, and it stays the pop's clip geometry unless the transform keeps axes and the bounds land on
    /// whole pixels. The pop fills that geometry's inverse; when its device corners leave the safe box
    /// the fill first clips the figure to the box, and a NaN the clip leaves behind fails the fill.
    fn native_layer_clip(&self, bounds: [f32; 4], dm: &Mat, w: u32, h: u32, axis_aligned: bool, mask_aa: u32) -> crate::layer::NativeClip {
        let m32: [f32; 6] = dm.map(|v| v as f32);
        let [l, t, r, b] = bounds;
        let (nl, nr) = if l > r { (r, l) } else { (l, r) };
        let (nt, nb) = if t > b { (b, t) } else { (t, b) };
        // Unbounded is judged on the sorted rect: an inverted infinite axis [inf, -inf] spans the plane too
        // (r15 20968), not a NaN corner.
        let unbounded = nl <= -f32::MAX && nr >= f32::MAX || nt <= -f32::MAX && nb >= f32::MAX;
        let (dev, geom) = if unbounded {
            ([nl, nt, nr, nb], false)
        } else {
            let d = crate::layer::rect_fill_bounds(bounds, &m32);
            (d, !(crate::layer::axis_preserving(&m32) && crate::layer::pixel_aligned(d)))
        };
        let clip = self.draw_clip_px().map_or([0.0, 0.0, 2147483520.0, 2147483520.0], |c| c.map(|v| v as f32));
        let rect = crate::layer::layer_rect(dev, w as f32, h as f32, clip);
        let nan = rect.is_some_and(|r| r.iter().any(|v| v.is_nan()));
        let px = rect.filter(|_| !nan).map_or([0; 4], |r| {
            let c = |v: f32, n: u32| v.clamp(0.0, n as f32) as i32;
            [c(r[0], w), c(r[1], h), c(r[2], w), c(r[3], h)]
        });
        let off_box = |v: f32| v.is_nan() || v.abs() > 524287.75;
        let needs_clip = geom && crate::layer::rect_corners([nl, nt, nr, nb], &m32).iter().flatten().any(|&v| off_box(v));
        let (mut fail, mut exact) = (false, false);
        if needs_clip {
            // The figure the fill clips: the rect's corners through Direct2D's specialised point
            // transform, starting at the top-left and going clockwise.
            let quad: Vec<[f32; 2]> = [[nl, nt], [nr, nt], [nr, nb], [nl, nb]].iter().map(|c| transform_mesh_point(dm, c[0], c[1])).collect();
            let segs: Vec<crate::pixels::Seg> = quad[1..].iter().map(|&q| crate::pixels::Seg::Line(q)).collect();
            let clipped = crate::pixels::clip_figure(quad[0], &segs).map(|(s, g)| {
                std::iter::once(s).chain(g.iter().map(|g| match *g { crate::pixels::Seg::Line(q) | crate::pixels::Seg::Cubic(_, _, q) => q })).collect::<Vec<_>>()
            });
            fail = clipped.as_ref().is_some_and(|c| c.iter().flatten().any(|v| v.is_nan()));
            exact = axis_aligned
                && quad.iter().flatten().all(|v| v.is_finite())
                && clipped.as_ref().is_some_and(|c| c.iter().flatten().all(|v| v.is_finite()));
        }
        // An axis-aligned rect whose clip leaves infinite (not NaN) crossings rasterises as that clipped
        // figure, the crossings at the far corner: the fill path's own huge-shape clip.
        let far = (needs_clip && axis_aligned && !fail && !exact)
            .then(|| self.render_mask(&crate::geometry::Geometry::Rect(bounds), mask_aa, dm, w, h));
        crate::layer::NativeClip { rect: px, nan, fail, exact, hard: !geom, far }
    }

    /// Normalised device-space rect: map both DIP corners through the device transform and sort.
    fn device_rect(&self, rect: [f32; 4]) -> [f32; 4] {
        let m = self.device_transform();
        // A singular (det 0) device transform collapses the rect to a zero-area line/point, so a layer
        // or clip built from it covers no area — native clips to that degenerate region (≈ nothing),
        // but our axis-aligned AABB of the two endpoints would span a large box through the origin
        // (fuzz 1254/983: a COPY layer under a collinear transform cleared the whole box). Collapse to
        // empty. (ponytail: exact zero area; a thin line native may still rasterise is a sub-case.)
        if m[0] * m[3] - m[1] * m[2] == 0.0 {
            return [0.0, 0.0, 0.0, 0.0];
        }
        let a = apply(&m, rect[0], rect[1]);
        let c = apply(&m, rect[2], rect[3]);
        [a[0].min(c[0]), a[1].min(c[1]), a[0].max(c[0]), a[1].max(c[1])]
    }

    /// The axis-aligned device clip rect for `PushAxisAlignedClip`: the AABB of the transformed rect
    /// over all FOUR corners (native stays axis-aligned under a rotation/skew — a PushLayer instead
    /// clips to the quad). The two-corner diagonal AABB `device_rect` uses is too narrow under a
    /// rotation/skew (fuzz-clip-01/02). Singular transform → empty (as device_rect).
    fn device_clip_rect(&self, rect: [f32; 4]) -> [f32; 4] {
        let m = self.device_transform();
        if m[1] == 0.0 && m[2] == 0.0 {
            return self.device_rect(rect); // axis-aligned: byte-exact two-corner path
        }
        if m[0] * m[3] - m[1] * m[2] == 0.0 {
            return [0.0, 0.0, 0.0, 0.0];
        }
        self.device_corner_box(rect)
    }

    /// The box of the rect's four corners under the device transform: `device_clip_rect` past its axis-aligned and
    /// singular cases.
    fn device_corner_box(&self, rect: [f32; 4]) -> [f32; 4] {
        let m = self.device_transform();
        let cs = [apply(&m, rect[0], rect[1]), apply(&m, rect[2], rect[1]), apply(&m, rect[2], rect[3]), apply(&m, rect[0], rect[3])];
        // A corner the transform cannot place (an infinite rect turned, inf - inf) leaves the clip NaN, as the
        // axis-aligned path does: nothing draws inside it (r27 32327; sc129: under a rotation or skew, aliased or not).
        if cs.iter().flatten().any(|v| v.is_nan()) {
            return [f32::NAN; 4];
        }
        let (xs, ys) = (cs.map(|c| c[0]), cs.map(|c| c[1]));
        [
            xs.iter().copied().fold(f32::INFINITY, f32::min),
            ys.iter().copied().fold(f32::INFINITY, f32::min),
            xs.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            ys.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        ]
    }

    /// The device rect a BOUNDED_SOURCE_COPY slow path's image must reach to copy: the target within
    /// the pushed clips and layer bounds (an image outside them keeps the content: b11reach-* probes,
    /// r6 6816's bitmap left of its layer).
    pub(crate) fn reach_rect(&self) -> [f32; 4] {
        let (tw, th) = self.target.as_ref().map_or((0, 0), |b| (b.desc.width as i32, b.desc.height as i32));
        let c = self.draw_clip_px().unwrap_or([0, 0, tw, th]);
        [c[0].max(0), c[1].max(0), c[2].min(tw), c[3].min(th)].map(|v| v as f32)
    }

    /// Whether a command list is being played back.
    pub(crate) fn in_replay(&self) -> bool {
        self.replay_depth > 0
    }

    /// The pixel rect the pushed axis-aligned clips and layer content bounds leave (an aliased clip
    /// keeps the pixels whose centres it contains, an antialiased clip or a layer every pixel it
    /// touches); `None` with nothing pushed.
    pub(crate) fn draw_clip_px(&self) -> Option<[i32; 4]> {
        let out = |r: &[f32; 4]| [r[0].floor() as i32, r[1].floor() as i32, r[2].ceil() as i32, r[3].ceil() as i32];
        self.stack.iter().fold(None, |acc, f| {
            let px = match f {
                crate::layer::Frame::Clip { rect: r, aliased: true, .. } => r.map(|v| (v - 0.5).ceil() as i32),
                crate::layer::Frame::Clip { rect: r, .. } => out(r),
                // A layer extends over its content bounds within its geometric mask's bounds, intersected before
                // rounding out: a mask beside the bounds leaves nothing, even inside one pixel (r21 26687).
                // A layer whose device bounds came out NaN (an infinite edge times a zero matrix term, as an
                // inverted infinite axis gives) reaches over its own pixel rect: it spans the plane, it does
                // not empty a layer nested inside it (r15 20968).
                crate::layer::Frame::Layer { params, .. } if !params.bounds_nan && params.bounds.iter().any(|v| v.is_nan()) && params.native.is_some() => {
                    params.native.as_ref().map_or([0; 4], |n| n.rect)
                }
                crate::layer::Frame::Layer { params, .. } if !params.bounds_nan => {
                    let (b, m) = (params.bounds, params.mask_bounds.unwrap_or(params.bounds));
                    let b = [b[0].max(m[0]), b[1].max(m[1]), b[2].min(m[2]), b[3].min(m[3])];
                    if b[0] > b[2] || b[1] > b[3] { [0; 4] } else { out(&b) }
                }
                crate::layer::Frame::Layer { .. } => return acc,
            };
            Some(acc.map_or(px, |a: [i32; 4]| [a[0].max(px[0]), a[1].max(px[1]), a[2].min(px[2]), a[3].min(px[3])]))
        })
    }

    /// Swap the bound target for a working offscreen, returning the backdrop. `copy` ⇒ a backdrop
    /// copy (clip draws blend against it); else transparent (layer group isolation).
    fn swap_working(&mut self, copy: bool) -> PixelBuffer {
        let saved = self.target.take().unwrap_or_else(|| bitmap_target(0, 0, self.dpi.0));
        let mut working = saved.clone();
        if !copy {
            working.data.fill(0);
        }
        self.target = Some(working);
        saved
    }

    /// Rasterize `geom` as opaque white into a transparent scratch at `mask_aa`, under the device
    /// matrix `m` (`maskTransform × world × DPI`), returning the alpha channel as per-pixel mask
    /// coverage (the fill path's coverage is exactly what the mask needs).
    fn render_mask(&self, geom: &crate::geometry::Geometry, mask_aa: u32, m: &Mat, w: u32, h: u32) -> Vec<u8> {
        self.render_mask_in(geom, mask_aa, m, w, h, false)
    }

    /// [`Self::render_mask`]; `copy` draws an antialiased mask by the COPY blend's trapezoid walk
    /// ([`crate::pixels::copy_tess::fill_copy`]), as native does for a mask pushed under COPY.
    fn render_mask_in(&self, geom: &crate::geometry::Geometry, mask_aa: u32, m: &Mat, w: u32, h: u32, copy: bool) -> Vec<u8> {
        let dm: crate::geometry::Matrix =
            [m[0] as f32, m[1] as f32, m[2] as f32, m[3] as f32, m[4] as f32, m[5] as f32];
        // A mask geometry with ANY non-finite-position figure voids the WHOLE mask — native's figure
        // sink rejects the geometry (not just the bad figure), so the layer passes nothing (fuzz r16
        // 21603: g1's fig0 has a -inf cubic endpoint, fig1 is finite; native shows nothing where our
        // per-figure flatten dropped only fig0 and leaked fig1's 33 px). This is the whole-geometry void
        // `any_nonfinite`, stricter than the flatten path's per-figure `has_nonfinite` drop.
        if geom.any_nonfinite() {
            return vec![0u8; w as usize * h as usize];
        }
        // Layer masks go through the ordinary (28.4) fill path, not the WARP tessellator gate.
        let polys: Vec<Vec<[f32; 2]>> = geom.flatten_fill(&dm).into_iter().filter(|p| p.filled).map(|p| p.pts).collect();
        let fill = if geom.fill_mode() == crate::geometry::FillMode::Winding { FillMode::Winding } else { FillMode::Alternate };
        let mut scratch = bitmap_target(w, h, self.dpi.0);
        if copy && mask_aa == 0 {
            let slices: Vec<&[[f32; 2]]> = polys.iter().map(|f| f.as_slice()).collect();
            let mut out = vec![0u8; w as usize * h as usize];
            // A half-covered pixel rounds down, as the mask coverage path below does (r16 21639).
            crate::pixels::copy_tess::fill_copy(&slices, fill == FillMode::Alternate, w as usize, h as usize, true, |x, y, cov, _| {
                out[y * w as usize + x] = match cov.c8() { 128 => 127, c => c as u8 };
            });
            return out;
        }
        // An aliased mask other than a rectangle is drawn by WARP's aliased edge walk, as an aliased fill is (probe-amd:
        // an ellipse and an arc path at dpi 48 to 192).
        if mask_aa == 1 && !matches!(geom, crate::geometry::Geometry::Rect(_)) {
            Target::new(&mut scratch).fill_warp28(&polys, fill == FillMode::Winding, [255, 255, 255, 255], Blend::SourceOver);
        } else {
            Target::new(&mut scratch).fill_path(&polys, fill, [255, 255, 255, 255], aa_of(mask_aa), Blend::SourceOver, true);
        }
        // The fill rasteriser quantises coverage round-to-even, but the mask/clip coverage path
        // rounds half **down**; the two differ only at the half sample (the 64ths
        // grid's 32/64 → 127.5), so remap that one value to match native (`layers-04`).
        scratch.data.chunks_exact(4).map(|p| if p[3] == 128 { 127 } else { p[3] }).collect()
    }

    /// The opacity-brush alpha at each device pixel centre, with the layer's constant `opacity`
    /// folded in in float (one round), matching native's opacity-mask term. The
    /// fold uses the brush's *sub-byte* alpha (`source_at_f`) so a gradient opacity brush folds the
    /// opacity into its unquantised value, as WARP does — the byte-rounded gradient would tip a LSB
    /// (`15-layer-mask`). A bitmap opacity brush has no sub-byte form and falls back to the byte path.
    /// An opacity brush's per-pixel alpha × the layer opacity: as bytes, and for an IGNORE_ALPHA layer unrounded too.
    /// `region` is the device rect the pop fills, over which a rotated HIGH_QUALITY_CUBIC brush realizes its intermediate.
    fn opacity_alpha(&self, brush: &Brush, opacity: f32, w: u32, h: u32, ignore: bool, region: [f32; 4]) -> (Vec<u8>, Option<Vec<f32>>) {
        let hq = brush.prepare_rotated([region[0].max(0.0), region[1].max(0.0), region[2].min(w as f32), region[3].min(h as f32)]);
        let mut brush = hq.unwrap_or_else(|| brush.clone());
        // A HIGH_QUALITY_CUBIC opacity brush is realized at full opacity into an intermediate of its precision, then its
        // own opacity scales the stored sample (r27 32584, 32649, 32717: Δ1 at their pushes).
        let stored = match &mut brush {
            Brush::Bitmap { interpolation: 5, opacity, intermediate_precision, .. } => Some((std::mem::replace(opacity, 1.0).clamp(0.0, 1.0), *intermediate_precision)),
            _ => None,
        };
        // A CUBIC opacity brush on an 8-bit target saturates its sample before its own opacity scales it (a float
        // target keeps the overshoot); clamping the scaled sample to [0, opacity] is the same (r27 32539's probes).
        let cubic_cap = match &brush {
            Brush::Bitmap { interpolation: 2, opacity, .. } if !self.target.as_ref().is_some_and(|t| crate::float::is_highp(t.desc.format)) => Some(opacity.clamp(0.0, 1.0)),
            _ => None,
        };
        let alpha = |x: f32, y: f32| {
            // An ANISOTROPIC brush hands the pop its sampler's stored byte, as it does a brush fill (VM: an ANISOTROPIC
            // bitmap opacity brush under INITIALIZE_FROM_BACKGROUND and/or IGNORE_ALPHA, 196-387 px off with the float
            // sample, exact with the byte; a LINEAR one is exact with the float; r27 32837).
            let a = if matches!(brush, Brush::Bitmap { interpolation: 4, .. }) { brush.source_at(x, y, true)[3] / 255.0 } else { brush.source_at_f(x, y)[3] };
            let a = cubic_cap.map_or(a, |o| a.clamp(0.0, o));
            stored.map_or(a, |(o, p)| crate::effects::quantize_to(p, [a.clamp(0.0, 1.0); 4])[3] * o)
        };
        let f = ignore.then(|| {
            let mut v = vec![0f32; w as usize * h as usize];
            for y in 0..h as usize {
                for x in 0..w as usize {
                    // A brush that paints nothing (a non-finite matrix samples NaN) fades to nothing, as the
                    // byte path's NaN → 0 (r27 32054).
                    v[y * w as usize + x] = (alpha(x as f32 + 0.5, y as f32 + 0.5) * opacity).max(0.0).min(1.0);
                }
            }
            v
        });
        let mut v = vec![0u8; w as usize * h as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let a = alpha(x as f32 + 0.5, y as f32 + 0.5) * 255.0 * opacity;
                v[y * w as usize + x] = a.round_ties_even().clamp(0.0, 255.0) as u8;
            }
        }
        (v, f)
    }

    fn pop_clip_or_layer(&mut self, want_clip: bool) {
        if !self.in_draw {
            self.process_new_error(WRONG_STATE);
            return;
        }
        // Recording: a `pop_layer` records a `PopLayer` command (its `push_layer` recorded `PushLayer`);
        // clips are not recorded into lists (no 2A scene nests a clip in a command list).
        // Recording: the pop must match a clip/layer pushed into this list, else it fails as on the live
        // stack (clclip probes: a pop with nothing open is PUSH_POP_UNBALANCED, the wrong kind
        // POP_DID_NOT_MATCH_PUSH, both at the pop).
        if let Some(rc) = self.recording.clone() {
            let top = rc.borrow().frames.last().copied();
            match top {
                None => self.process_new_error(PUSH_POP_UNBALANCED),
                Some(clip) if clip != want_clip => self.process_new_error(POP_DID_NOT_MATCH_PUSH),
                Some(_) => {
                    let mut l = rc.borrow_mut();
                    l.frames.pop();
                    l.cmds.push(if want_clip { RecCmd::PopClip } else { RecCmd::PopLayer });
                }
            }
            return;
        }
        match self.stack.last() {
            None => {
                self.process_new_error(PUSH_POP_UNBALANCED);
                return;
            }
            Some(crate::layer::Frame::Clip { .. }) if !want_clip => {
                self.process_new_error(POP_DID_NOT_MATCH_PUSH);
                return;
            }
            Some(crate::layer::Frame::Layer { .. }) if want_clip => {
                self.process_new_error(POP_DID_NOT_MATCH_PUSH);
                return;
            }
            _ => {}
        }
        // The pop fills the layer with its opacity brush under the transform then, so a brush that fill can't draw fails
        // the pop (r27 32705: a rank-one brush pushed under the identity, popped under a zero matrix).
        if let Some(crate::layer::Frame::Layer { params: crate::layer::LayerParams { pop_brush: Some(crate::layer::PopBrush { brush: b, region: r, .. }), .. }, .. }) = self.stack.last() {
            let (b, r) = (self.brush_under_current(b), *r);
            // An effect image brush is realized for the pop as for any draw, so its Border can fail it too (r27 32792: a
            // WRAP brush of an effect at scale 1e-6 fails the pop, not the push).
            let border = |c: &Self| {
                let (Some(def), Some(d)) = (b.image.as_ref(), c.draw_pixels(r)) else { return false };
                c.image_brush_inverse(def.tile, def.m).is_some_and(|inv| c.image_brush_border_too_large(def.extend, def.tile, &inv, d))
            };
            if let Some(hr) = self.brush_error(&b).or_else(|| (self.brush_need_too_large(&b, r) || border(self)).then_some(INTERMEDIATE_TOO_LARGE)) {
                self.process_new_error(hr);
            }
        }
        // The opacity brush keeps its placement from the push, but a bitmap's own DPI scales it by the unit mode at the
        // pop (r27 32580: pushed in DIPs, popped in pixels; r27 32291: pushed in pixels, popped in DIPs).
        let redo = match self.stack.last() {
            Some(crate::layer::Frame::Layer { params, .. }) if params.brush_alpha.is_some() && !params.keep_pop && !params.void_pop => {
                params.pop_brush.clone().filter(|p| p.unit != self.state.unit).map(|p| (p, params.ignore_alpha))
            }
            _ => None,
        };
        if let (Some((p, ignore)), Some((w, h))) = (redo, self.target.as_ref().map(|t| (t.desc.width, t.desc.height))) {
            if let Some(crate::brush::BrushDef::Bitmap { dpi, .. }) = &p.brush.def {
                if *dpi != (96.0, 96.0) && !p.brush.brush.singular() {
                    let (sx, sy) = if self.state.unit == 1 { (1.0, 1.0) } else { (96.0 / dpi.0, 96.0 / dpi.1) };
                    let mut b = p.brush.clone();
                    b.brush.set_placement([p.b2d[0] * sx, p.b2d[1] * sx, p.b2d[2] * sy, p.b2d[3] * sy, p.b2d[4], p.b2d[5]]);
                    let (a, af) = self.opacity_alpha(&b.brush, p.opacity, w, h, ignore, p.region);
                    if let Some(crate::layer::Frame::Layer { params, .. }) = self.stack.last_mut() {
                        (params.brush_alpha, params.brush_alpha_f) = (Some(a), af);
                    }
                }
            }
        }
        let working = self.target.take().unwrap_or_else(|| bitmap_target(0, 0, self.dpi.0));
        // A float (HDR) backdrop composites the working offscreen back in f32 with no 8-bit
        // quantisation (`crate::float`); the BGRA8/UNORM path keeps the integer composite.
        let saved = match self.stack.pop().unwrap() {
            crate::layer::Frame::Clip { mut saved, rect, aliased, min_rule, corner } => {
                if crate::float::is_highp(saved.desc.format) {
                    crate::float::clip_back_f(&mut saved, &working, rect, aliased, corner);
                } else {
                    let ignore = self.stack.iter().rev().find_map(|f| match f {
                        crate::layer::Frame::Layer { params, .. } => Some(params.ignore_alpha),
                        _ => None,
                    });
                    crate::layer::clip_back(&mut saved, &working, rect, aliased, &min_rule, ignore == Some(true), corner);
                }
                saved
            }
            crate::layer::Frame::Layer { mut saved, params } => {
                if params.void_pop {
                    // nothing composites: the backdrop as it was at the push, erased under a geometric mask
                    crate::layer::void_pop_erase(&mut saved, &working, &params);
                } else if crate::float::is_highp(saved.desc.format) {
                    crate::float::layer_back_f(&mut saved, &working, &params);
                } else {
                    crate::layer::layer_back(&mut saved, &working, &params, self.tile);
                }
                if let (Some((x, y)), true) = (params.corner, params.corner_live) {
                    crate::layer::copy_pixel(&mut saved, &working, x, y);
                }
                saved
            }
        };
        self.target = Some(saved);
    }

    /// `DrawImage` of a flood (constant-colour) image under a composite mode: fill
    /// the bound target with `composite(src, dst, mode)`. The flood is infinite, so the surrounding
    /// axis-aligned clip bounds it. `src` is premultiplied BGRA (0..=255).
    /// `rect`: the DrawImage image rectangle, which makes the flood a finite image: the placed rectangle filled
    /// with its colour ([`draw_image_cropped`](Self::draw_image_cropped)).
    // ponytail: a recorded flood keeps no rectangle (replayed uncropped); record it when a list scene needs one.
    pub fn draw_flood(&mut self, src: [i32; 4], offset: [f32; 2], rect: Option<[f32; 4]>, interp: u32, composite: u32) {
        let _a = self.api(64, 0, None);
        if !self.ready_to_draw() || !self.draw_image_args_ok(interp, composite) {
            return;
        }
        match rect {
            Some(r) if self.recording.is_none() => self.draw_image_cropped(offset, r, composite, |dc, o| dc.flood(src, o, true, interp, 0)),
            // A list keeps the image rectangle and replays the flood cropped to it (r27 32363: a 0.03-DIP-wide crop draws nothing).
            Some(r) => {
                let (state, dpi) = self.rec_env();
                self.record(RecCmd::Flood { state, dpi, src, interp, composite, crop: Some((offset, r)) });
            }
            None => self.flood(src, offset, false, interp, composite),
        }
    }

    /// [`draw_flood`](Self::draw_flood) of the whole flood; `finite` when an image rectangle bounds it.
    fn flood(&mut self, src: [i32; 4], offset: [f32; 2], finite: bool, interp: u32, composite: u32) {
        // A DrawImage of the (infinite) Flood effect at an unplaceable offset is realized empty and that
        // empty image is composited — the keep-modes keep the backdrop, the IN/OUT/COPY set clear (fuzz
        // r15 20770 offset.y=inf SOURCE_ATOP, 20898/20325 offset.x=-inf DEST_OVER). "Unplaceable" is not
        // only ±inf/NaN: a huge finite offset collapses the visible region in image space
        // ([`offset_collapses`](Self::offset_collapses); fuzz r16 21027, r20 25696). Our draw_flood otherwise
        // ignores the offset and composites the flood everywhere. (A recorded-in-a-list flood replays at
        // offset 0, so it is placeable.)
        if self.recording.is_none() {
            // Direct2D's fast path tests the Flood's placement as for any image: a NaN that passes it draws
            // nothing, or clears under the copy modes (r23 28546, offset (1e9, NaN) at 144 dpi).
            if self.nan_fast_placement(offset, interp, false) {
                if matches!(composite, 10 | 11) {
                    self.composite_empty(10);
                }
                return;
            }
            let w = self.state.transform;
            let s = if self.state.unit == 0 { [self.dpi.0 / 96.0, self.dpi.1 / 96.0] } else { [1.0; 2] };
            let p = [(offset[0] + w[4]) * s[0], (offset[1] + w[5]) * s[1]];
            // The transform is unplaceable when native's inverse of it fails. Native inverts with the
            // shared matrix routine, which fails when det is 0 OR the reciprocal 1/det is non-finite as
            // an f32 — a near-singular det whose reciprocal overflows f32 is as degenerate as an exactly
            // singular one, by that same test. So a 1e-40 det (whose f32 1/det overflows to ±inf) is
            // unplaceable just like the det-0 seeds, with no guessed tolerance. The reciprocal follows
            // native's op order (f64 divide, cast to f32). A non-finite det — a 3.4e38 entry whose
            // product overflows to ±inf (fuzz 21571) — stays unplaceable too; the effect realizes into a
            // zero-area / overflowed intermediate and composites empty (also fuzz r16 21434, singular
            // 2×2), matching the degenerate-transform "draws nothing" rule.
            let det = w[0] * w[3] - w[1] * w[2];
            let inv_fails = det == 0.0 || !((1.0_f64 / det as f64) as f32).is_finite();
            if self.offset_collapses(p) || w[..4].iter().any(|v| !v.is_finite()) || !det.is_finite() || inv_fails {
                // BOUNDED_SOURCE_COPY of the (empty) unplaceable flood normally leaves the backdrop
                // (bounded by the empty image), but under a degenerate (non-invertible) transform with
                // ANISOTROPIC resampling it does not: anisotropic takes its output rectangle from the
                // device Jacobian, which the collapse spans across the whole reach, so it copies the
                // empty source across the reach exactly like SOURCE_COPY. (VM matrix under a singular
                // transform: comp 11 wipes iff interp==ANISOTROPIC; SOURCE_COPY wipes for every interp;
                // fuzz 26380/26101.) Any singular form triggers it (a rank-1 [1,0,2,0] with a non-zero
                // linear part wipes too, not only the all-zero collapse). reach_rect is clip-bounded —
                // 26101 wipes only its pushed clip, 26380 the whole target. (An even tinier *visible*
                // scale — e.g. 1e-18, det 1e-36, whose 1/det is still finite — also wipes natively, but
                // that tail is past a native crash zone: det 1e-6 crashes the reference, so there is no
                // clean oracle boundary there, and no corpus scene reaches it.)
                if composite == 11 && interp == 4 && inv_fails {
                    let r = self.reach_rect().map(|v| v as i32);
                    if let Some(t) = self.target.as_ref() {
                        let mut inter = t.clone();
                        inter.data.fill(0);
                        self.composite_over_target(&inter, 10, Some(r));
                    }
                } else {
                    self.composite_empty(composite);
                }
                return;
            }
            // Under a scale or rotation the infinite Flood goes through the transform stage's intermediate.
            // A flood with an image rectangle is finite (draw_image_cropped), so only a bare one is checked as infinite (r24 29914).
            if !finite && self.transformed_image_done([i32::MIN, i32::MIN, i32::MAX, i32::MAX], offset, interp, composite) {
                return;
            }
        }
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            self.record(RecCmd::Flood { state, dpi, src, interp, composite, crop: None });
            return;
        }
        // Translated only, the flood lands on the device rect Direct2D's f32 placement gives it (all of a target at an
        // ordinary offset). Past it, an atlas blit composites transparent; a direct draw leaves the target alone, except
        // the two copy modes, which clear it (sc90).
        // Off unit DPI scale the direct rule holds at a device offset whole on both axes (sc89), and a fractional one
        // takes the image-brush atlas rule (flood_brush_span).
        // A fractional HIGH_QUALITY_CUBIC one pre-scaled on an axis ([`crate::brush::flood_hq`]) places its intermediate with
        // the residual cubic; unscaled, it is the flood colour everywhere.
        let w = self.state.transform;
        let s = if self.state.unit == 0 { [self.dpi.0 / 96.0, self.dpi.1 / 96.0] } else { [1.0; 2] };
        let p = [(offset[0] + w[4]) * s[0], (offset[1] + w[5]) * s[1]];
        let whole = fast_whole(p[0]) && fast_whole(p[1]);
        let hq = if interp == 5 && w[..4] == [1.0, 0.0, 0.0, 1.0] && s != [1.0; 2] && !whole {
            self.flood_matrix(offset, composite).and_then(|m| crate::brush::flood_hq(src.map(|v| v as f32 / 255.0), m, self.reach_rect()))
        } else {
            None
        };
        let mut msl_end = None;
        let span = if w[..4] != [1.0, 0.0, 0.0, 1.0] {
            None
        } else if s == [1.0; 2] || whole {
            self.flood_span(p, interp)
        } else {
            self.flood_brush_span(offset, interp, composite).map(|(r, ends)| {
                msl_end = (interp == 3).then_some(ends);
                (r, true)
            })
        };
        let clear_outside = span.is_some_and(|(_, atlas)| atlas) || matches!(composite, 10 | 11);
        let span = span.map(|(r, _)| r);
        let Some(buf) = &mut self.target else { return };
        let stride = buf.stride;
        let (w, h) = (buf.desc.width as usize, buf.desc.height as usize);
        for y in 0..h {
            for x in 0..w {
                let i = y * stride + x * 4;
                let dst = [buf.data[i], buf.data[i + 1], buf.data[i + 2], buf.data[i + 3]];
                let inside = span.is_none_or(|r| (x as i32) >= r[0] && (y as i32) >= r[1] && (x as i32) < r[2] && (y as i32) < r[3]);
                if !inside && !clear_outside {
                    continue;
                }
                let mut src = if inside { src } else { [0; 4] };
                if let Some(f) = &hq {
                    src = f(x as f32 + 0.5, y as f32 + 0.5).map(|v| (v * 255.0).round() as i32);
                }
                if let Some(e) = msl_end.filter(|_| inside) {
                    let cov = msl_edge_cover(x as f32, e[0]) * msl_edge_cover(y as f32, e[1]);
                    src = src.map(|v| (v as f32 * cov).round() as i32);
                }
                if let Some(out) = crate::layer::composite(src, dst, composite) {
                    buf.data[i..i + 4].copy_from_slice(&out);
                }
            }
        }
    }

    /// `FillOpacityMask(bitmap, brush, content, dest?, src?)`. Native
    /// Native's `FillOpacityMask` validates in this order: not in a draw → `WRONG_STATE`;
    /// **current antialias mode must be ALIASED** else `WRONG_STATE`; `content > 2` → `E_INVALIDARG`;
    /// brush from another factory → `WRONG_FACTORY`. Then it fills `dest` (DIPs, default = the
    /// bitmap's size) with the solid `brush` modulated by the mask alpha over the `src` texel sub-rect
    /// (default = the whole bitmap), aliased, under the current primitive blend. `content` only
    /// selects text-gamma for non-A8 masks (no pixel effect for a graphics mask), so it is validated
    /// but not otherwise used here.
    pub fn fill_opacity_mask(
        &mut self,
        bitmap: &Bitmap,
        brush: &SolidColorBrush,
        content: u32,
        dest: Option<[f32; 4]>,
        source: Option<[f32; 4]>,
    ) {
        let _a = self.api(80 + 24, 0, None);
        if !self.ready_to_draw() {
            return;
        }
        if self.state.aa != 1 {
            self.process_new_error(WRONG_STATE);
            return;
        }
        if content > 2 {
            self.process_new_error(E_INVALIDARG);
            return;
        }
        if !self.check_domain(brush.domain) {
            return;
        }
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            self.record(RecCmd::OpacityMask { state, dpi, bitmap: Rc::new(bitmap.clone()), color: brush.color, opacity: brush.opacity, content, dest, source });
            return;
        }
        let (bw, bh) = (bitmap.width as f32, bitmap.height as f32);
        // `device_rect` normalises + maps `dest` (DIPs→device); `src` is texels, normalised in place.
        let [sl, st, sr, sb] = source.unwrap_or([0.0, 0.0, bw, bh]);
        let src = [sl.min(sr), st.min(sb), sl.max(sr), st.max(sb)];
        // No destination rectangle: the source rectangle's size within the mask, placed at the origin (r27 32591:
        // a 1×6 source of a 3×10 mask paints 1×6 pixels).
        let dip = dest.unwrap_or([0.0, 0.0, src[2].min(bw) - src[0].max(0.0), src[3].min(bh) - src[1].max(0.0)]);
        let premul = brush.premul_f();
        let blend = blend_of(self.state.blend);
        let m = self.device_transform();
        // Native's `FillOpacityMask` voids only a dest that is itself NON-FINITE — a dest
        // rect with an ±inf/NaN edge (fuzz r13 18466 dest top/bottom ±inf, 18028 left -inf). A FINITE
        // dest that merely overflows the 28.4 device grid under the dpi/transform scale (a 3.4e38 edge
        // → ±inf device) still DRAWS: native clamps it to the target and fills the mask's texel-0 colour
        // (r14 19612 dest left 3.4e38 paints x 0..25). So guard on the raw DIP dest, not the transformed
        // corners.
        if !dip.iter().all(|v| v.is_finite()) {
            return;
        }
        if self.fom_huge_dest(bitmap, src, dip, &m, [premul[2], premul[1], premul[0], premul[3]], None) {
            return;
        }
        // A rotation/skew maps the dest to a quad; `device_rect`'s diagonal AABB would paint far
        // outside it (fuzz 14312), so route non-axis-aligned transforms through the quad fill.
        let axis_aligned = (m[1] == 0.0 && m[2] == 0.0) || (m[0] == 0.0 && m[3] == 0.0);
        let dest = self.device_rect(dip);
        let uv = self.fom_uv(bitmap, source, dip);
        if let Some(buf) = &mut self.target {
            if axis_aligned {
                let Some(uv) = uv else { return };
                crate::layer::opacity_mask_fill(buf, bitmap, premul, dest, uv, blend);
            } else {
                let dip = [dip[0].min(dip[2]), dip[1].min(dip[3]), dip[0].max(dip[2]), dip[1].max(dip[3])];
                let bgra = [premul[2], premul[1], premul[0], premul[3]];
                crate::layer::opacity_mask_fill_xform(buf, bitmap, dip, m, src, uv, blend, |_, _| bgra);
            }
        }
    }

    /// A FillOpacityMask whose dest has a device corner the 28.4 edge conversion cannot hold (|v| ≥ 2³¹,
    /// ±∞ or NaN) leaves the rect path: Direct2D fills the dest as a general shape through the safe-box
    /// strip clip (the overflow edge crosses to the far corner and draws a slope-1 wedge, r20 25198), and
    /// the collapsed source samples the mask's NEAREST source-origin texel times the brush's coarsest
    /// sample `s` (BGRA premultiplied) — a constant. A merely large finite dest (1e9) stays per-pixel.
    /// Returns whether it handled the fill (fuzz r21 26216; r15 20464 for a solid brush).
    /// The FOM mask plane ([`crate::layer::fom_uv_plane`]): `source` is in the mask's DIPs (default
    /// its whole DIP size), `dip` the DIP dest, under the current device transform.
    fn fom_uv(&self, bitmap: &Bitmap, source: Option<[f32; 4]>, dip: [f32; 4]) -> Option<[f32; 6]> {
        let size = [bitmap.width as f32 * 96.0 / bitmap.dpi.0, bitmap.height as f32 * 96.0 / bitmap.dpi.1];
        let world = self.device_transform().map(|v| v as f32);
        crate::layer::fom_uv_plane(source.unwrap_or([0.0, 0.0, size[0], size[1]]), dip, world, size)
    }

    fn fom_huge_dest(&mut self, bitmap: &Bitmap, src: [f32; 4], dip: [f32; 4], m: &Mat, s: [f32; 4], brush: Option<&Brush>) -> bool {
        let corners = [apply(m, dip[0], dip[1]), apply(m, dip[2], dip[1]), apply(m, dip[2], dip[3]), apply(m, dip[0], dip[3])];
        if !corners.iter().any(|q| !q[0].is_finite() || !q[1].is_finite() || q[0].abs() >= 2147483648.0 || q[1].abs() >= 2147483648.0) {
            return false;
        }
        let tex = crate::sampling::Texture { data: &bitmap.premul_bgra, width: bitmap.width, height: bitmap.height };
        let m_a = crate::sampling::norm_v128(tex.bilinear_v128([crate::sampling::Extend::Clamp; 2], 3, (src[0].floor() as i32, 0), (src[1].floor() as i32, 0)));
        let strip: [f32; 4] = core::array::from_fn(|c| (m_a * s[c]).clamp(0.0, 1.0));
        let src_u8 = strip.map(|c| (c * 255.0).round_ties_even() as u8);
        let dm: crate::geometry::Matrix = [m[0] as f32, m[1] as f32, m[2] as f32, m[3] as f32, m[4] as f32, m[5] as f32];
        let figures: Vec<Vec<[f32; 2]>> = crate::geometry::Geometry::Rect(dip).flatten_fill(&dm).into_iter().filter(|p| p.filled).map(|p| p.pts).collect();
        // A gradient brush computes its coordinate per pixel from the device position, so it is sampled
        // per pixel across the wedge like any gradient fill (fuzz r19 24317 radial, recap 1264 and r17
        // 22924 linear: the coarsest level only matched where the ramp clamps to an end stop); the mask
        // texel stays the constant.
        if self.state.blend == 0 {
            if let (Some(b @ (Brush::Radial { .. } | Brush::Linear { .. })), Some(buf)) = (brush, self.target.as_mut().filter(|t| !crate::float::is_highp(t.desc.format))) {
                let Some(figs) = crate::pixels::clip_huge(&figures) else { return true };
                let edges = crate::pixels::figure_edges(&figs);
                let px = |x: usize, y: usize| {
                    let s = b.source_for_mask(x as f32 + 0.5, y as f32 + 0.5);
                    s.map(|c| ((m_a * c).clamp(0.0, 1.0) * 255.0).round_ties_even() as u8)
                };
                Target::new(buf).fill_edges_px(&edges, FillMode::Winding, aa_of(self.state.aa), Blend::SourceOver, px);
                return true;
            }
        }
        self.raster(RasterCmd::FillPolys { figures, winding: true, src: src_u8, src_f: strip, aa: self.state.aa, blend: self.state.blend, geometry: true, warp28: false });
        true
    }

    /// `FillOpacityMask` with a gradient/bitmap/image brush (the common bitmap/gradient-brush form:
    /// the mask selects where the brush shows). Same aliased requirement, content range and rects as
    /// the solid [`fill_opacity_mask`]; the per-pixel brush source replaces the flat premul colour.
    pub fn fill_opacity_mask_brush(
        &mut self,
        bitmap: &Bitmap,
        brush: &PixelBrush,
        content: u32,
        dest: Option<[f32; 4]>,
        source: Option<[f32; 4]>,
    ) {
        let _a = self.api(80 + brush.batch, 0, None);
        if !self.ready_to_draw() {
            return;
        }
        if self.state.aa != 1 {
            self.process_new_error(WRONG_STATE);
            return;
        }
        if content > 2 {
            self.process_new_error(E_INVALIDARG);
            return;
        }
        if !self.check_fill_brush(brush, true) {
            return;
        }
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            self.record(RecCmd::OpacityMaskBrush { state, dpi, bitmap: Rc::new(bitmap.clone()), brush: brush.clone(), content, dest, source });
            return;
        }
        // A gradient realizes under the device transform of the draw, as for a fill (r19 24363: a radial
        // brush made under DIPs, filled under unit mode PIXELS).
        let grad_rr = self.realize_brush_for_draw(brush);
        let brush = grad_rr.as_ref().unwrap_or(brush);
        let (bw, bh) = (bitmap.width as f32, bitmap.height as f32);
        let [sl, st, sr, sb] = source.unwrap_or([0.0, 0.0, bw, bh]);
        let src = [sl.min(sr), st.min(sb), sl.max(sr), st.max(sb)];
        // No destination rectangle: the source rectangle's size within the mask, placed at the origin (r27 32591:
        // a 1×6 source of a 3×10 mask paints 1×6 pixels).
        let dip = dest.unwrap_or([0.0, 0.0, src[2].min(bw) - src[0].max(0.0), src[3].min(bh) - src[1].max(0.0)]);
        let blend = blend_of(self.state.blend);
        let m = self.device_transform();
        // See `fill_opacity_mask`: only a NON-FINITE dest (±inf/NaN edge) voids; a finite dest that
        // overflows the device grid still draws. Guard the raw DIP dest.
        if !dip.iter().all(|v| v.is_finite()) {
            return;
        }
        let axis_aligned = (m[1] == 0.0 && m[2] == 0.0) || (m[0] == 0.0 && m[3] == 0.0);
        let dest = self.device_rect(dip);
        let staged = brush;
        // An image brush re-realizes its effect per draw over the device need-region, exactly as a fill
        // or `fill_rectangle_brush` does — `FillOpacityMask` lacked this and kept the creation-time
        // realization. When native's placement cannot size that region (a degenerate or huge brush
        // transform), the realization is empty and the FOM paints nothing (r16 21080: a 3.4e38 brush
        // matrix element). `image_brush_for_draw` returns `None` for such a brush (its `fit`/`det` guards),
        // which here means void, not the "use the creation brush" fallback a non-image brush takes.
        let per_draw = match (brush.image.is_some() || brush.crop.is_some(), self.target.as_ref().map(|t| (t.desc.width as f32, t.desc.height as f32))) {
            (true, Some((tw, th))) => {
                let d = [dest[0].floor().max(0.0) as i32, dest[1].floor().max(0.0) as i32, dest[2].ceil().min(tw) as i32, dest[3].ceil().min(th) as i32];
                // A dest past the device grid draws the brush's coarse sample as one strip (`fom_huge_dest`): no
                // effect graph runs, so its Border cannot fail (recap 6031).
                let huge = [[dip[0], dip[1]], [dip[2], dip[1]], [dip[2], dip[3]], [dip[0], dip[3]]]
                    .map(|[x, y]| apply(&m, x, y)).iter().any(|q| !q[0].is_finite() || !q[1].is_finite() || q[0].abs() >= 2147483648.0 || q[1].abs() >= 2147483648.0);
                match self.image_brush_for_draw(brush, d, false, !huge) {
                    Some(mut r) => { r.brush.keep_cell_quad(false); Some(r) }
                    None if brush.image.is_none() => None,
                    None => return,
                }
            }
            _ => None,
        };
        let brush = per_draw.as_ref().unwrap_or(brush);
        if self.fom_huge_dest(bitmap, src, dip, &m, brush.brush.coarse_sample(), Some(&brush.brush)) {
            return;
        }
        // A huge dest draws nothing first (r6 6031).
        if self.brush_need_too_large(staged, dest) {
            self.process_new_error(INTERMEDIATE_TOO_LARGE);
            return;
        }
        if self.brush_samples_padding_only(staged, dest) {
            return;
        }
        // A gradient FOM under an AA clip composites back with the clip's nearest-edge coverage
        // (`layer::clip_min_cova`, r13 18771): mark the pixels it fills in the enclosing clip frame.
        if axis_aligned && matches!(brush.brush, crate::brush::Brush::Linear { .. } | crate::brush::Brush::Radial { .. }) {
            if let (Some(t), Some(crate::layer::Frame::Clip { aliased: false, min_rule, .. })) = (&self.target, self.stack.last_mut()) {
                let (w, h) = (t.desc.width as i32, t.desc.height as i32);
                let (x0, x1, y0, y1) = crate::layer::opacity_mask_span(dest, w, h);
                if min_rule.is_empty() {
                    *min_rule = vec![0; (w * h) as usize];
                }
                for y in y0..y1 {
                    min_rule[(y * w + x0) as usize..(y * w + x1) as usize].fill(1);
                }
            }
        }
        let uv = self.fom_uv(bitmap, source, dip);
        // A HIGH_QUALITY_CUBIC brush is rendered into the frame's atlas over the dest's whole-pixel quad, at the next slot
        // (VM: the mask kernel's plane addresses (x - 5)/512 alone and (x + 91)/512 after a PLUS DrawImage took the
        // target-sized region at the atlas origin).
        let hq_slot = match (&brush.brush, &self.target) {
            (crate::brush::Brush::Bitmap { interpolation: 5, .. }, Some(t)) => {
                let (w, h) = (t.desc.width as f32, t.desc.height as f32);
                let dq = [dest[0].max(0.0).floor(), dest[1].max(0.0).floor(), dest[2].min(w).ceil(), dest[3].min(h).ceil()];
                self.atlas.alloc((dq[2] - dq[0]) as i32, (dq[3] - dq[1]) as i32)
            }
            _ => None,
        }.unwrap_or([0; 2]);
        if let Some(buf) = &mut self.target {
            let (w, h) = (buf.desc.width as f32, buf.desc.height as f32);
            let mut hq = brush.brush.prepare_rotated([dest[0].max(0.0), dest[1].max(0.0), dest[2].min(w), dest[3].min(h)]);
            if let Some(b) = &mut hq { b.set_hq_slot(hq_slot); }
            let prepared = hq.is_some();
            let mut fill = hq.unwrap_or_else(|| brush.brush.clone());
            // A HIGH_QUALITY_CUBIC brush reaches the mask through an 8-bit intermediate at full opacity: its sample is
            // saturated and stored as a byte before the brush opacity scales it (r27 32114: an overshooting alpha at
            // opacity 0.93; the byte store took its 521 px Δ1 to 4).
            let hq_opacity = match &mut fill {
                crate::brush::Brush::Bitmap { interpolation: 5, opacity, intermediate_precision, .. } => Some((std::mem::replace(opacity, 1.0).clamp(0.0, 1.0), *intermediate_precision)),
                _ => None,
            };
            let dq = [dest[0].max(0.0).floor(), dest[1].max(0.0).floor(), dest[2].min(w).ceil(), dest[3].min(h).ceil()];
            let src_at = |x: f32, y: f32| {
                // Its intermediate is placed over the dest's whole-pixel quad from that quad's origin (r27 32114); a
                // prescaled brush's cached placement already is.
                let s = if hq_opacity.is_some() && !prepared { fill.source_at_f_rect_origin(x, y, dq, [dq[0] - hq_slot[0] as f32, dq[1] - hq_slot[1] as f32]) } else { fill.source_for_mask(x, y) };
                hq_opacity.map_or(s, |(o, p)| crate::effects::quantize_to(p, s.map(|v| v.clamp(0.0, 1.0))).map(|v| v * o))
            };
            if axis_aligned {
                let Some(uv) = uv else { return };
                crate::layer::opacity_mask_fill_brush(buf, bitmap, src_at, dest, uv, blend);
            } else {
                // A non-axis-aligned transform maps the dest to a quad (fuzz 14312 family); fill only it.
                let dip = [dip[0].min(dip[2]), dip[1].min(dip[3]), dip[0].max(dip[2]), dip[1].max(dip[3])];
                crate::layer::opacity_mask_fill_xform(buf, bitmap, dip, m, src, uv, blend, src_at);
            }
        }
    }

    // --- drawing ------------------------------------------------------------------------------

    /// `Clear`: fill the whole target with `color` (premultiplied), ignoring transform and blend.
    /// `None` is transparent black. WRONG_STATE outside a draw or on a command-list target; no-op in
    /// error.
    pub fn clear(&mut self, color: Option<[f32; 4]>) {
        let _a = self.api(32, 0, Some(crate::batch::Prim::Clear));
        if !self.ready_to_draw() {
            return;
        }
        let c = color.unwrap_or([0.0; 4]);
        // A Clear on a command-list target is RECORDED, not WRONG_STATE (VM-verified: EndDraw S_OK).
        // On replay it fills the list's drawn area — the list is an infinite image, so an opaque Clear
        // covers the whole target it is drawn into (G19; BoltBait Blur/Feather Selection Edge).
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            self.record(RecCmd::Clear { state, dpi, color: c });
            return;
        }
        // An IGNORE_ALPHA layer's surface is opaque: a Clear inside it stores its colour with the alpha
        // ignored, which the pop then fades as opaque content (r27 32046). A list played back renders into
        // its own intermediate, where a recorded Clear keeps its alpha (r27 32313).
        let ignore = self.replay_depth == 0 && self.stack.iter().rev().find_map(|f| match f {
            crate::layer::Frame::Layer { params, .. } => Some(params.ignore_alpha),
            _ => None,
        }) == Some(true);
        let c = if ignore { [c[0], c[1], c[2], 1.0] } else { c };
        if let Some(buf) = &mut self.target {
            if crate::float::is_highp(buf.desc.format) {
                crate::float::FloatTarget::new(buf).clear(crate::float::premul_f(c, 1.0));
            } else {
                Target::new(buf).clear(premultiply(c));
            }
        }
    }

    /// `FillRectangle`: fill an axis-aligned DIP rectangle with a solid brush under the current
    /// device transform. A rotated/skewed transform turns it into a winding quad fill.
    pub fn fill_rectangle(&mut self, rect: [f32; 4], brush: &SolidColorBrush) {
        let _a = self.api(48 + 24, 0, Some(crate::batch::Prim::Fill));
        if !self.ready_to_draw() || !self.check_domain(brush.domain) {
            return;
        }
        // Native draws nothing for a rect with a non-finite coordinate (fuzz 189: a NaN edge → whole
        // scene stays the clear; we otherwise rasterised NaN bounds over the origin).
        if !rect.iter().all(|v| v.is_finite()) {
            return;
        }
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            self.record(RecCmd::FillRect { state, dpi, rect, color: brush.color, opacity: brush.opacity });
            return;
        }
        let m = self.device_transform();
        self.fill_rect_under(rect, brush, m);
    }

    /// The rectangle fill behind `FillRectangle`, with the rect mapped to device space by `m` (the
    /// device transform, or `DrawLine`'s line matrix) and the brush under the device transform.
    fn fill_rect_under(&mut self, rect: [f32; 4], brush: &SolidColorBrush, m: Mat) {
        let (src, src_f, aa, blend) = (brush.premul(), brush.premul_f(), self.state.aa, self.state.blend);
        // Singular (det 0) device transform collapses the rect to a line/point — zero area, native draws
        // nothing (see fill_rectangle_brush; fuzz r12 17584).
        if m[0] * m[3] - m[1] * m[2] == 0.0 {
            return;
        }
        let a = apply(&m, rect[0], rect[1]);
        let c = apply(&m, rect[2], rect[3]);
        // Native normalises an inverted rect (right<left / bottom<top), filling the min→max region
        // rather than nothing (census1-12); take the device extent.
        let dev = [a[0].min(c[0]), a[1].min(c[1]), a[0].max(c[0]), a[1].max(c[1])];
        // WARP's unit-rect eighths fast path (our `rasterize_rect`'s `e8`) is taken only for a
        // unit-scale axis-aligned rect whose ±0.5-inflated device corners all stay within |524287|
        // (0x48FFFFE0). A non-unit scale (dpi≠96 or a world scale) or any rotation/skew rounds a ~0.31
        // edge to 2/8 under WARP's 8×8 scan but 3/8 under e8 (q8829 dpi-120 vs aarect-frac
        // dpi-96); and once any inflated corner exceeds |524287|, native leaves the rect fast path and
        // fills the rect as rectangle geometry (rfill-huge-*). Both land on the 8×8
        // scan, which our polygon fill reproduces.
        // A quarter turn keeps the rectangle axis-aligned in device space, so it takes the rect paths an unrotated
        // transform does (VM probes, a strikethrough turned with a vertical run, aliased: the edge snaps to the 24.8
        // grid as the rect fast path's does, not to the aliased edge walk's 28.4).
        let quarter = m[0] == 0.0 && m[3] == 0.0;
        let unit = (m[1] == 0.0 && m[2] == 0.0 && m[0].abs() == 1.0 && m[3].abs() == 1.0)
            || (quarter && m[1].abs() == 1.0 && m[2].abs() == 1.0);
        let in_range = [dev[0] - 0.5, dev[1] - 0.5, dev[2] + 0.5, dev[3] + 0.5].iter().all(|v| v.abs() <= 524287.0);
        if unit && in_range {
            self.raster(RasterCmd::FillRect { rect: dev, src, src_f, aa, blend });
        } else {
            let rotated = (m[1] != 0.0 || m[2] != 0.0) && !quarter;
            // A rotated or out-of-range rect fills as a `CRectangle` shape: its corners go through the
            // transform sink in f32, under the alternate fill mode it sets (see `fill_rect_brush_under`).
            let figure = if rotated || !in_range {
                [[rect[0], rect[1]], [rect[2], rect[1]], [rect[2], rect[3]], [rect[0], rect[3]]].map(|p| transform_mesh_point(&m, p[0], p[1])).to_vec()
            } else {
                vec![a, apply(&m, rect[2], rect[1]), c, apply(&m, rect[0], rect[3])]
            };
            // An ALIASED rect under a rotation or skew misses the rectangle fast paths and fills as rectangle
            // geometry through WARP's aliased edge walk, pixel-centre shift included (r4 5071).
            let warp28 = aa == 1 && rotated;
            self.raster(RasterCmd::FillPolys {
                figures: vec![figure],
                winding: !(rotated || !in_range),
                src,
                src_f,
                aa,
                blend,
                geometry: warp28, // the FillRectangle primitive has no pixel-centre shift; the geometry fill does
                warp28,
            });
        }
    }

    /// `FillGeometry` for pre-flattened line figures (the rasterizer path; curve flattening and
    /// stroking are phase 1C). `figures` are DIP polygons; `winding` selects the fill rule.
    pub fn fill_polygons(&mut self, figures: &[Vec<[f32; 2]>], winding: bool, brush: &SolidColorBrush) {
        if !self.ready_to_draw() || !self.check_domain(brush.domain) {
            return;
        }
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            self.record(RecCmd::FillPolys { state, dpi, figures: figures.to_vec(), winding, color: brush.color, opacity: brush.opacity });
            return;
        }
        let (src, src_f, aa, blend, m) =
            (brush.premul(), brush.premul_f(), self.state.aa, self.state.blend, self.device_transform());
        let dev: Vec<Vec<[f32; 2]>> =
            figures.iter().map(|f| f.iter().map(|p| apply(&m, p[0], p[1])).collect()).collect();
        self.raster(RasterCmd::FillPolys { figures: dev, winding, src, src_f, aa, blend, geometry: true, warp28: false });
    }

    /// `FillRectangle` with a gradient or bitmap/image brush (`brush.rs`). The per-pixel source is
    /// composited directly into the target with source-over at full coverage — enough for the
    /// pixel-aligned rectangle fills the brush scenes use. Fractional-edge anti-aliasing of a brush
    /// fill, non-SOURCE_OVER blends, and recording into a command list are not wired yet.
    pub fn fill_rectangle_brush(&mut self, rect: [f32; 4], brush: &PixelBrush) {
        let _a = self.api(48 + brush.batch, 0, Some(brush.queued()));
        if !self.ready_to_draw() || !self.check_fill_brush(brush, true) {
            return;
        }
        if !rect.iter().all(|v| v.is_finite()) {
            return; // native draws nothing for a non-finite rect (see fill_rectangle)
        }
        // While recording a command list, record the DIP rect + the brush (with its device-independent
        // definition); replay re-realizes the brush under the replay transform. [G6]
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            self.record(RecCmd::FillRectBrush { state, dpi, rect, brush: brush.clone() });
            return;
        }
        // Gradient brushes realize lazily per draw under the current device transform (see
        // `realize_brush_for_draw`); a world transform set after CreateBrush must apply.
        let m = self.device_transform();
        self.fill_rect_brush_under(rect, brush, m);
    }

    /// The rectangle fill behind `FillRectangle` with a pixel brush, the rect mapped by `m` (see
    /// [`Self::fill_rect_under`]); the brush stays under the device transform.
    fn fill_rect_brush_under(&mut self, rect: [f32; 4], brush: &PixelBrush, m: Mat) {
        let grad_rr = self.realize_brush_for_fill(brush);
        let brush = grad_rr.as_ref().unwrap_or(brush);
        // Singular (det 0) device transform collapses the rect to a line/point — zero area, native draws
        // nothing. The 2-corner AABB below would otherwise span a stray box (two diagonal corners of a
        // collapsed rect still differ), which a COPY blend then wipes (fuzz r12 17584).
        if m[0] * m[3] - m[1] * m[2] == 0.0 {
            return;
        }
        let a = apply(&m, rect[0], rect[1]);
        let c = apply(&m, rect[2], rect[3]);
        // An inverted rect fills its min→max extent, as the solid fill's (recap9 10919's image brush).
        let (lo, hi) = ([a[0].min(c[0]), a[1].min(c[1])], [a[0].max(c[0]), a[1].max(c[1])]);
        if let Some(buf) = &self.target {
            let visible = [lo[0].max(0.0), lo[1].max(0.0), hi[0].min(buf.desc.width as f32), hi[1].min(buf.desc.height as f32)];
            if visible[0] < visible[2] && visible[1] < visible[3]
                && (brush.brush.hq_request_bounds(visible).is_some_and(|request| {
                    request.iter().any(|v| !v.is_finite() || *v < -2_147_483_648.0 || *v >= 2_147_483_648.0)
                }) || self.brush_need_too_large(brush, visible))
            {
                // Native CDImageContext::ProcessTile rejects an unrepresentable intermediate request.
                self.process_new_error(INTERMEDIATE_TOO_LARGE);
                return;
            }
        }
        if self.brush_samples_padding_only(brush, [lo[0], lo[1], hi[0], hi[1]]) {
            return;
        }
        let blend = blend_of(self.state.blend);
        let per_draw = match (brush.image.is_some() || brush.crop.is_some(), &self.target) {
            (true, Some(t)) => {
                let q = [a, apply(&m, rect[2], rect[1]), c, apply(&m, rect[0], rect[3])];
                let lo = |k: usize| q.iter().map(|p| p[k]).fold(f32::INFINITY, f32::min);
                let hi = |k: usize| q.iter().map(|p| p[k]).fold(f32::NEG_INFINITY, f32::max);
                let d = [lo(0).floor().max(0.0) as i32, lo(1).floor().max(0.0) as i32, hi(0).ceil().min(t.desc.width as f32) as i32, hi(1).ceil().min(t.desc.height as f32) as i32];
                let whole = crate::brush::whole_px_rect(&q) && matches!(brush.brush, crate::brush::Brush::Bitmap { opacity, .. } if opacity >= 1.0);
                self.image_brush_for_draw(brush, d, false, true).map(|mut pb| { pb.brush.keep_cell_quad(whole); pb })
            }
            _ => None,
        };
        // An image brush whose per-draw realization native cannot place (a degenerate/huge brush
        // transform) paints nothing, as in FillOpacityMask — VM-probed: a 3.4e38-matrix effect image
        // brush voids here too. `image_brush_for_draw` returns `None` for it, which for an image brush
        // means void, not the creation-brush fallback a non-image (solid/gradient/bitmap) brush takes.
        if brush.image.is_some() && per_draw.is_none() {
            return;
        }
        let brush = per_draw.as_ref().unwrap_or(brush);
        // A rotated/skewed transform turns the rectangle into a quad, not the axis-aligned bbox. So does a
        // rect with a device corner past |524287|: native's
        // `FillAxisAlignedAntiAliasedRectangle_Cutout` and the aliased `CSmallRegionFillTessellator` both
        // leave the rect fast path for native's non-overlapping-rectangles slow path, which fills a rectangle shape
        // shape through `FillShapeData` and its `ClipWithRect` (r13 18547: a 3.4e38 corner draws nothing).
        // The shape streams its corners through the transform sink in f32 and sets the alternate fill
        // mode; the strip clip's f32 crossings then decide how a huge rotated rect lands (a 1e9-wide
        // diagonal DrawLine's clipped figure covers nothing, r24 29432). The general rasterizer takes the
        // corners as they are, without the rect special case's 16.8 rounding.
        // A NaN device corner (a NaN transform term) takes the same geometry path, whose sink voids the
        // fill, as the solid fill does (r15 20489: `[1e6, 0, 0, 1e6, NaN, …]` draws nothing).
        // A CUBIC bitmap brush is not one of the rect special case's brushes either: native fills the rect as geometry,
        // whose edge snapping counts an eighth the rect path's doesn't (sc239/sc240: a top edge at 5.564 covers 3/8,
        // not the rect path's 4/8; r27 32321's image-rect fill).
        // Nor is the LINEAR image brush a DrawImage fills its image rectangle with: its pixel shader samples the bitmap
        // through the fitted plane and stores the bilinear truncated, where the rect case's blit rounds (r27 32839).
        let cubic = (matches!(&brush.brush, Brush::Bitmap { interpolation: 2, .. }) || self.crop_fill && matches!(&brush.brush, Brush::Bitmap { interpolation: 1, .. }))
            && self.target.as_ref().is_some_and(|t| !crate::float::is_highp(t.desc.format));
        // A quarter turn keeps the rect axis-aligned in device space (see `fill_rect_under`).
        let rotated = (m[1] != 0.0 || m[2] != 0.0) && !(m[0] == 0.0 && m[3] == 0.0);
        // Nor is an aliased radial gradient: d2d1's aliased rectangle fast path on WARP takes solid, gradient-ramp and bitmap
        // brushes only and hands any other to the slow path's geometry, as a rotated rect (r8 9948: an aliased radial rect's
        // left edge at 5.5014 leaves column 5 uncovered where the fast path's 24.8 snap covers it; VM probes g3c/e948 under
        // SOURCE_OVER and COPY, a solid brush still covering it).
        let aliased_radial = self.state.aa == 1 && matches!(&brush.brush, Brush::Radial { .. });
        if cubic || rotated || aliased_radial || [lo[0], lo[1], hi[0], hi[1]].iter().any(|v| !(v.abs() <= 524287.0)) {
            let q = [[rect[0], rect[1]], [rect[2], rect[1]], [rect[2], rect[3]], [rect[0], rect[3]]].map(|p| transform_mesh_point(&m, p[0], p[1]));
            let aa = aa_of(self.state.aa);
            // The aliased rotated/skewed rect fills through WARP's aliased edge walk (see `fill_rectangle`).
            let warp28 = aa == crate::pixels::Aa::Aliased && (rotated || aliased_radial);
            let clip = self.raster_clip();
            if let Some(buf) = &mut self.target {
                brush::fill_geometry_cov(buf, &[q.to_vec()], false, aa, &brush.brush, blend, warp28, false, clip, false);
            }
            return;
        }
        if let Some(buf) = &mut self.target {
            // A float (HDR) target samples the *same* 8-bit gradient ramp with full-precision f32
            // bilinear and composites in f32 (`crate::float`, oracle `hdr-11`), rather than the BGRA8
            // UNORM8 path. (Radial/bitmap brushes to a float target reuse the same entry; only the
            // linear gradient has a captured scene.)
            if crate::float::is_highp(buf.desc.format) {
                crate::float::fill_rect_brush_f(buf, [lo[0], lo[1], hi[0], hi[1]], &brush.brush, blend);
            } else {
                brush::fill_rect(buf, [lo[0], lo[1], hi[0], hi[1]], &brush.brush, blend, aa_of(self.state.aa));
            }
        }
    }

    /// `DrawLine` with a solid brush. See [`Self::line_rect`]: a line without a stroke style (or one the
    /// rectangle widening accepts) fills a rectangle under the line's own matrix; otherwise it strokes
    /// the one-segment line.
    pub fn draw_line(&mut self, p0: [f32; 2], p1: [f32; 2], brush: &SolidColorBrush, width: f32, style: Option<&crate::geometry::StrokeStyle>) {
        let _a = self.api(64 + 24, 0, Some(crate::batch::Prim::Fill));
        // A command list keeps the line and widens it at playback, under the playback transform.
        if self.recording.is_some() {
            if self.ready_to_draw() && self.check_domain(brush.domain) {
                let (state, dpi) = self.rec_env();
                self.record(RecCmd::Line { state, dpi, p0, p1, width, style: style.cloned(), color: brush.color, opacity: brush.opacity });
            }
            return;
        }
        {
            if let Some((rect, m)) = self.line_rect(p0, p1, width, style) {
                if self.ready_to_draw() && self.check_domain(brush.domain) && rect.iter().all(|v| v.is_finite()) {
                    self.fill_rect_under(rect, brush, m);
                }
                return;
            }
        }
        self.draw_geometry(&line_geometry(p0, p1), brush, width, style.unwrap_or(&crate::geometry::StrokeStyle::default()));
    }

    /// `DrawLine` with a gradient or bitmap/image brush (see [`Self::draw_line`]).
    pub fn draw_line_brush(&mut self, p0: [f32; 2], p1: [f32; 2], brush: &PixelBrush, width: f32, style: Option<&crate::geometry::StrokeStyle>) {
        let _a = self.api(64 + brush.batch, 0, Some(brush.queued()));
        if self.recording.is_some() {
            if self.ready_to_draw() && self.check_brush(brush) {
                let (state, dpi) = self.rec_env();
                self.record(RecCmd::LineBrush { state, dpi, p0, p1, width, style: style.cloned(), brush: brush.clone() });
            }
            return;
        }
        {
            if let Some((rect, m)) = self.line_rect(p0, p1, width, style) {
                if self.ready_to_draw() && self.check_brush(brush) && rect.iter().all(|v| v.is_finite()) {
                    self.fill_rect_brush_under(rect, brush, m);
                }
                return;
            }
        }
        self.draw_geometry_brush(&line_geometry(p0, p1), brush, width, style.unwrap_or(&crate::geometry::StrokeStyle::default()));
    }

    /// The rectangle `DrawLine` fills and the device matrix it fills it under, or `None` when the line
    /// strokes as a geometry. A line is widened to the rectangle `[0, -w/2, length, w/2]` whose x axis the
    /// matrix maps along the line from `p0` — all in f32, so a far endpoint's rounding moves where the
    /// line ends (a line from x=1e9 to 30.5 runs to 0, r23 28137). With no stroke style that is always the
    /// route. With one, it is taken when the line is at most a device pixel wide (or a hairline) and not
    /// dashed, or when both caps are flat or square, it is not dashed, and a fixed-width stroke has equal
    /// DPIs. A square cap (or any non-flat cap on a line at most a pixel wide) extends that end by w/2.
    fn line_rect(&self, p0: [f32; 2], p1: [f32; 2], width: f32, style: Option<&crate::geometry::StrokeStyle>) -> Option<([f32; 4], Mat)> {
        use crate::geometry::stroke::{Cap, TransformType};
        let a = self.device_transform().map(|v| v as f32);
        let (dpi_x, dpi_y) = if self.state.unit == 1 { (96.0, 96.0) } else { self.dpi };
        let wabs = width.abs();
        let tt = style.map_or(TransformType::Normal, |s| s.transform_type);
        if let Some(s) = style {
            let dashed = !s.dashes.is_empty();
            let mut thin = false;
            if a[0] == a[3] && a[1] == -a[2] && dpi_x == dpi_y {
                let scale = (a[2] * a[2] + a[0] * a[0]).sqrt();
                thin = match tt {
                    TransformType::Normal => wabs * scale <= 1.0,
                    TransformType::Fixed => dpi_x / 96.0 * wabs <= 1.0,
                    TransformType::Hairline => true,
                };
            }
            let squared = |c: Cap| matches!(c, Cap::Flat | Cap::Square);
            let fast = if thin { !dashed } else { squared(s.start_cap) && squared(s.end_cap) && !dashed && (tt != TransformType::Fixed || dpi_x == dpi_y) };
            if !fast {
                return None;
            }
        }
        let (mut dx, mut dy) = (p1[0] - p0[0], p1[1] - p0[1]);
        let (mut ox, mut oy) = (p0[0], p0[1]);
        let w = match tt {
            TransformType::Normal => wabs,
            TransformType::Fixed => wabs * dpi_x / 96.0,
            TransformType::Hairline => 1.0,
        };
        if tt != TransformType::Normal {
            // A fixed-width or hairline line widens in device space.
            (dx, dy) = (a[0] * dx + a[2] * dy, dy * a[3] + a[1] * dx);
            (ox, oy) = ((p0[1] * a[2] + p0[0] * a[0]) + a[4], (p0[1] * a[3] + p0[0] * a[1]) + a[5]);
        }
        let len = (dy * dy + dx * dx).sqrt();
        let inv = 1.0 / len;
        let (mut ux, mut uy) = (inv * dx, inv * dy);
        let mut right = len;
        if let Some(s) = style {
            if len * len < 0.000025 {
                (ux, uy) = (1.0, 0.0);
            }
            let ext = |c: Cap| c == Cap::Square || (c != Cap::Flat && w <= 1.0);
            if ext(s.start_cap) {
                ox -= ux * w * 0.5;
                oy -= uy * w * 0.5;
                right = len + w * 0.5;
            }
            if ext(s.end_cap) {
                right += w * 0.5;
            }
        }
        let rect = [0.0, w * -0.5, right, w * 0.5];
        let m: [f32; 6] = if tt != TransformType::Normal {
            [ux, uy, -uy, ux, ox, oy]
        } else {
            let nuy = -uy;
            [
                uy * a[2] + a[0] * ux,
                uy * a[3] + a[1] * ux,
                a[2] * ux + a[0] * nuy,
                a[1] * nuy + a[3] * ux,
                (ox * a[0] + a[2] * oy) + a[4],
                (a[1] * ox + oy * a[3]) + a[5],
            ]
        };
        Some((rect, m.map(|v| v as f64)))
    }

    /// `FillGeometry` for a retained geometry (phase 1C). Flattens curves the way native fills do —
    /// the MIL `CMILBezier` cracker on 28.4 device coordinates (`CFigureData::SendToD2DSinkInternal`),
    /// not the float query flattener — then feeds the exact device-space polyline to the rasterizer.
    pub fn fill_geometry(&mut self, geom: &crate::geometry::Geometry, brush: &SolidColorBrush) {
        let _a = self.api(128 + 24, 0, Some(crate::batch::Prim::Fill));
        if !self.ready_to_draw() || !self.check_domain(brush.domain) {
            return;
        }
        if geom.fill_nonfinite_void() {
            return; // native's geometry sink voids a path with ANY non-finite point (fuzz r11 16347)
        }
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            self.record(RecCmd::FillGeom { state, dpi, geom: geom.clone(), color: brush.color, opacity: brush.opacity });
            return;
        }
        let (src, src_f, aa, blend, m) =
            (brush.premul(), brush.premul_f(), self.state.aa, self.state.blend, self.device_transform());
        // A singular (det 0) device transform maps the plane onto a line. Under COPY the fill draws nothing
        // (fuzz r12 17584: an inverted/−3.4e38 rect under a det-0 transform keeps the backdrop native
        // keeps). Any other per-primitive fill streams its f32-mapped points into the 28.4 rasterizer like
        // any shape, where the rounding leaves a sliver along the line that draws (r26 31024, 31575).
        let singular = m[0] * m[3] - m[1] * m[2] == 0.0;
        let dm: crate::geometry::Matrix =
            [m[0] as f32, m[1] as f32, m[2] as f32, m[3] as f32, m[4] as f32, m[5] as f32];
        let winding = geom.fill_mode() == crate::geometry::FillMode::Winding;
        // An aliased fill of anything but a rect is a WARP geometry draw: `CreateGeometryGeneratorFromShape`
        // hands the outline to WARP's geometry sink and WARP's own aliased edge walk
        // rasterizes it (no triangles; 9711). Rect geometries take FillPath's rectangle path instead, unless a
        // rotation or skew term leaves them no rectangle path (r26 31156), as for FillRectangle. Under a
        // singular transform the outline collapses onto a line, and the edge walk still lights the pixel
        // centres its snapped edges pass through (26191).
        if aa == 1 && (!matches!(geom, crate::geometry::Geometry::Rect(_)) || dm[1] != 0.0 || dm[2] != 0.0) {
            let figures = geom.flatten_fill(&dm).into_iter().filter(|p| p.filled).map(|p| p.pts).collect();
            self.raster(RasterCmd::FillPolys { figures, winding, src, src_f, aa, blend, geometry: true, warp28: true });
            return;
        }
        if singular && blend == 1 {
            return;
        }
        // The rest go through native's 28.4 edge list (`CHwRasterizer`); the finer 26.6 grid is only
        // `DrawInk`'s AsGeometry path.
        // Every edge keeps its exact device position: an anti-aliased fill samples the unsnapped outline,
        // whatever the rect's orientation (VM probes: edges a few hundredths either side of a sample centre,
        // in all four inversions, at unit and fractional scale). Huge fills are strip-clipped inside
        // flatten_fill (d5e4393), so no inline clip_huge here.
        let figures: Vec<Vec<[f32; 2]>> = geom
            .flatten_fill_on(&dm, self.copy_target())
            .into_iter()
            .filter(|p| p.filled)
            .map(|p| p.pts)
            .collect();
        // A rectangle geometry under an axis-aligned transform, its ±0.5-inflated device corners within |524287|, fills
        // through the rectangle path FillRectangle takes, its fringe composited with that path's once-rounded
        // source-over (r8 9477: 237 where the output merger's form gives 236). Past the box it fills as geometry
        // (hugeclip-copy-fill), and so does any other shape (r26 31009, r27 32616).
        let rect_path = match geom {
            crate::geometry::Geometry::Rect(r) if dm[1] == 0.0 && dm[2] == 0.0 => {
                let (x0, x1, y0, y1) = (r[0] * dm[0] + dm[4], r[2] * dm[0] + dm[4], r[1] * dm[3] + dm[5], r[3] * dm[3] + dm[5]);
                [x0.min(x1) - 0.5, y0.min(y1) - 0.5, x0.max(x1) + 0.5, y0.max(y1) + 0.5].iter().all(|v| v.abs() <= 524287.0)
            }
            _ => false,
        };
        self.raster(RasterCmd::FillPolys { figures, winding, src, src_f, aa, blend, geometry: !rect_path, warp28: false });
    }

    /// `DrawGeometry` (phase 1C): widen the geometry into stroke fill pieces (device space) and
    /// rasterize them under nonzero winding, as native unions `CWidener`'s pieces. Axis-aligned
    /// stroke rectangles are byte-exact; cap/join/dash shapes are geometrically faithful (see
    /// `geometry::stroke`), and sloped-edge coverage carries the 1B rasterizer residual.
    pub fn draw_geometry(
        &mut self,
        geom: &crate::geometry::Geometry,
        brush: &SolidColorBrush,
        width: f32,
        style: &crate::geometry::StrokeStyle,
    ) {
        let _a = self.api(144 + 24, 0, Some(crate::batch::Prim::Fill));
        // A rectangle geometry's shape holds its rect in order, and the target strokes that shape as
        // `DrawRectangle` does. A recording keeps the geometry stroke, which replays through here.
        match geom {
            crate::geometry::Geometry::Rect(r) if self.recording.is_none() => self.draw_rectangle(*r, brush, width, style),
            _ => self.draw_geometry_impl(geom, brush, width, style, true),
        }
    }

    /// [`Self::draw_geometry`]; `warp` = false is `DrawRectangle`'s small-region rect fill (see
    /// [`Self::draw_rectangle`]), whose aliased coverage is the plain scan, not WARP's 28.4 rasterizer.
    fn draw_geometry_impl(
        &mut self,
        geom: &crate::geometry::Geometry,
        brush: &SolidColorBrush,
        width: f32,
        style: &crate::geometry::StrokeStyle,
        warp: bool,
    ) {
        if !self.ready_to_draw() || !self.check_domain(brush.domain) {
            return;
        }
        // A non-finite STROKE is NOT blanket-voided like a fill: native widens per figure (degenerating
        // non-finite controls to their endpoint) and only voids the whole DrawGeometry on an edge-list
        // overflow or a pen BAD_NUMBER tangent — both handled inside `widen_inner`.
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            self.record(RecCmd::StrokeGeom { state, dpi, geom: geom.clone(), color: brush.color, opacity: brush.opacity, width, style: style.clone() });
            return;
        }
        let (src, src_f, aa, blend, m) =
            (brush.premul(), brush.premul_f(), self.state.aa, self.state.blend, self.device_transform());
        let dm: crate::geometry::Matrix =
            [m[0] as f32, m[1] as f32, m[2] as f32, m[3] as f32, m[4] as f32, m[5] as f32];
        let dpi_scale = [self.dpi.0 / 96.0, self.dpi.1 / 96.0];
        // Aliased strokes reach WARP's own geometry rasterizer (`RasterizeGeometry`, not triangles): d2d1
        // hands WARP the widened outline's float vertices (no 1/64 grid snap: that put 9 of aastroke-01's
        // 47 traced vertices a 28.4 unit off), converted as every geometry fill ([`Target::fill_warp28`]).
        let figures = geom.widen_scaled(width, style, &dm, dpi_scale, crate::geometry::DEFAULT_TOLERANCE, 16.0, self.copy_target(), self.target_size());
        self.raster(RasterCmd::FillPolys { figures, winding: true, src, src_f, aa, blend, geometry: true, warp28: aa == 1 && warp });
    }

    /// `DrawRectangle`. d2d1's fast aliased-rect path strokes an
    /// ALIASED rect as `FillRectangles` over the fast axis-aligned-rectangle widen when
    /// the style allows it, never through WARP's geometry rasterizer (fuzz 14666, 18370). One outset
    /// rect takes the rect fast path (7161: the bitmap-brush sampler); the 8 border cells go to
    /// the fast aliased-rectangles path, which sends any count ≠ 1 to
    /// `FillTessellation(ALIASED, CSmallRegionFillTessellator)`: the plain aliased scan of the cells'
    /// union (= the widened outline) with the gradient pixel shader. Without the fast widen it is the
    /// geometry stroke. `DrawGeometry` of a rectangle geometry takes this same stroke: the geometry's
    /// shape keeps its rect in order (9584).
    /// Per-primitive, a WARP device takes the same fast widen (it never uses the dedicated rectangle-stroke
    /// tessellator): one outset rect fills through the rectangle path, and the 8 cells are filled as one
    /// shape of rectangle figures, even-odd. Under COPY that shape goes through the trapezoid edge walk,
    /// whose bands depend on the cells' edges (the inner cell edge ends at `bottom − w/2`, an off-target
    /// side keeps its own 1-pixel pair), not on the pen frame's.
    /// ponytail: a recorded 8-cell stroke replays as a geometry stroke (warp28); record the rect
    /// stroke as its own command if an aliased command-list rect stroke shows up in a diff.
    pub fn draw_rectangle(&mut self, rect: [f32; 4], brush: &SolidColorBrush, width: f32, style: &crate::geometry::StrokeStyle) {
        let _a = self.api(64 + 24, 0, Some(crate::batch::Prim::Fill));
        let rect = normalized_rect(rect);
        let g = crate::geometry::Geometry::Rect(rect);
        match self.fast_rect_stroke(rect, width, style) {
            Some(rects) if rects.len() == 1 => self.fill_rectangle(rects[0], brush),
            Some(rects) if self.state.aa != 1 => self.fill_geometry(&rect_cells(rects), brush),
            Some(_) => self.draw_geometry_impl(&g, brush, width, style, false),
            None => self.draw_geometry_impl(&g, brush, width, style, true),
        }
    }

    /// [`Self::draw_rectangle`] with a gradient/bitmap/image brush.
    pub fn draw_rectangle_brush(&mut self, rect: [f32; 4], brush: &PixelBrush, width: f32, style: &crate::geometry::StrokeStyle) {
        let _a = self.api(64 + brush.batch, 0, Some(brush.queued()));
        let rect = normalized_rect(rect);
        let g = crate::geometry::Geometry::Rect(rect);
        match self.fast_rect_stroke(rect, width, style) {
            Some(rects) if rects.len() == 1 => self.fill_rectangle_brush(rects[0], brush),
            Some(rects) if self.state.aa != 1 => self.fill_geometry_brush(&rect_cells(rects), brush),
            Some(rects) => self.draw_geometry_brush_impl(&g, brush, width, style, Some(&rects)),
            None => self.draw_geometry_brush_impl(&g, brush, width, style, None),
        }
    }

    /// The rects native's fast rectangle stroke fills for a `DrawRectangle` (its rect already
    /// [`normalized_rect`]), or `None` when it falls
    /// back to the geometry stroke. The style must pass the fast-widen eligibility test:
    /// solid, MITER or MITER_OR_BEVEL join, miter limit ≥ √2, NORMAL transform type.
    /// A stroke wider than the rect (or an inverted rect) is one outset rect; else the 8 border
    /// cells, each normalised `right = maxps(left, right)`, `bottom = maxps(top, bottom)`.
    fn fast_rect_stroke(&self, [l, t, r, b]: [f32; 4], width: f32, style: &crate::geometry::StrokeStyle) -> Option<Vec<[f32; 4]>> {
        use crate::geometry::stroke::{Join, TransformType};
        let widenable = style.dashes.is_empty()
            && matches!(style.join, Join::Miter | Join::MiterOrBevel)
            && style.miter_limit >= std::f32::consts::SQRT_2
            && style.transform_type == TransformType::Normal;
        if !widenable {
            return None;
        }
        let w = width.abs();
        let h = w * 0.5;
        if w > r - l || w > b - t {
            return Some(vec![[l - h, t - h, r + h, b + h]]);
        }
        let maxps = |a: f32, b: f32| if a > b { a } else { b };
        let cells = [
            [l - h, t - h, l + h, t + h], [l + h, t - h, r - h, t + h], [r - h, t - h, r + h, t + h],
            [l - h, t + h, l + h, b - h], [r - h, t + h, r + h, b - h],
            [l - h, b - h, l + h, b + h], [l + h, b - h, r - h, b + h], [r - h, b - h, r + h, b + h],
        ];
        Some(cells.iter().map(|&[l, t, r, b]| [l, t, maxps(l, r), maxps(t, b)]).collect())
    }

    /// `DrawImage(commandList)` under a `D2D1_COMPOSITE_MODE`. The list must
    /// be closed. SOURCE_OVER (0) replays each primitive directly into the target with the blend it
    /// was recorded under, so replay ≡ direct. Every other mode composites the list **as a unit**:
    /// native wraps it in a slow-path command list (realizes the content into an intermediate)
    /// then composites that image with the mode. We mirror that — realize the list into a transparent
    /// intermediate, then blend it over the whole target with `layer::composite`. Where the source is
    /// transparent (outside the list's content) the over/dest-over/dest-out/atop/xor/plus modes are a
    /// no-op on dst, while the in/out/copy/dest-atop modes zero dst — the "unbounded" effect. Native's
    /// BOUNDED_SOURCE_COPY bounds the copy to the DrawImage `imageRectangle`; with none supplied (the
    /// only form the harness can record) native copies over the whole target, so it equals SOURCE_COPY
    /// here (the fast image-draw path, SOURCE_COPY over the target rect).
    pub fn draw_image(&mut self, list: &CommandList, offset: [f32; 2], composite: u32) {
        self.draw_image_list(list, offset, None, 1, composite);
    }

    /// `DrawImage(commandList, offset, imageRectangle, interpolation, composite)`. The interpolation is only
    /// validated: Direct2D draws a command list at device resolution whatever the mode. The image rectangle (in the
    /// list's own space) is the part drawn, its top-left landing at `offset`: Direct2D clips the list's draws to
    /// `[offset, offset + rect size]` with the context's antialias mode.
    pub fn draw_image_list(&mut self, list: &CommandList, offset: [f32; 2], rect: Option<[f32; 4]>, interpolation: u32, composite: u32) {
        let _a = self.api(64, 0, None);
        if !self.ready_to_draw() || !self.check_domain(list.domain()) {
            return;
        }
        if !list.is_closed() {
            self.process_new_error(WRONG_STATE);
            return;
        }
        // D2D1_INTERPOLATION_MODE runs 0..=5 and D2D1_COMPOSITE_MODE 0..=12: anything past them is an invalid
        // argument (fuzz 15010: composite 13 → 0x80070057), checked before the list is drawn.
        if !self.draw_image_args_ok(interpolation, composite) {
            return;
        }
        // The list is drawn under the effective transform `t` = the DIP `offset` less the image rectangle's
        // top-left, then the replay context's world transform. Each recorded primitive replays under
        // `record_world · t`, so a translated / scaled / rotated / skewed DrawImage(commandList) matches drawing
        // the recorded primitives directly under that transform (native's resolution-independent command list).
        // An inverted image rectangle is its min→max extent, its top-left at the offset (sc151: native places an
        // x-inverted rect at its smaller x; r27 32144).
        let rect = rect.map(|r| [r[0].min(r[2]), r[1].min(r[3]), r[0].max(r[2]), r[1].max(r[3])]);
        let o = rect.map_or(offset, |r| [offset[0] - r[0], offset[1] - r[1]]);
        let t = crate::geometry::mat_mul(&[1.0, 0.0, 0.0, 1.0, o[0], o[1]], &self.state.transform);
        let clip = rect.map(|r| [offset[0], offset[1], offset[0] + (r[2] - r[0]), offset[1] + (r[3] - r[1])]);
        let cmds = list.0.borrow().cmds.clone();
        let w = self.state.transform;
        let axis_aligned = (w[1] == 0.0 && w[2] == 0.0) || (w[0] == 0.0 && w[3] == 0.0);
        // A list with an image rectangle is kept whole while recording: its clip applies where it is drawn.
        if self.recording.is_some() && rect.is_some() {
            let (state, dpi) = self.rec_env();
            self.record(RecCmd::DrawImageList { state, dpi, list: list.clone(), offset, rect, composite });
            return;
        }
        let inline = composite == 0 && crate::command::inline_safe(&cmds) && (clip.is_none() || axis_aligned);
        if let (true, Some(c)) = (inline, clip) {
            self.push_axis_aligned_clip(c, self.state.aa);
            self.replay(&cmds, t, self.dpi);
            self.pop_axis_aligned_clip();
            return;
        }
        if inline {
            // SOURCE_OVER inlines a list the fast-path checker passes. While recording, append its draws
            // (their transforms composed with `t`) to the open list so a nested source-over DrawImage
            // flattens into the outer recording; otherwise replay them directly under `t`.
            if self.recording.is_some() {
                for c in &cmds {
                    let mut c = c.clone();
                    if let Some(s) = c.state_mut() {
                        s.transform = crate::geometry::mat_mul(&s.transform, &t);
                    }
                    self.record(c);
                }
                return;
            }
            self.replay(&cmds, t, self.dpi);
            return;
        }
        // Non-SOURCE_OVER composites (and an unsafe list's SOURCE_OVER) as a unit. Recorded into an open list, keep it as a nested
        // DrawImageList replayed the same way; otherwise composite it now. [G6]
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            self.record(RecCmd::DrawImageList { state, dpi, list: list.clone(), offset, rect, composite });
            return;
        }
        self.composite_cmds(&cmds, t, self.dpi, composite, clip);
    }

    /// Realize a recorded command stream into a transparent target-sized intermediate under transform
    /// `t`, then blend it over the bound target as a unit under `composite` — native's
    /// DrawImage(commandList) slow path. A command list is an infinite image, so (like cmdlist-11) even
    /// BOUNDED_SOURCE_COPY copies over the whole target. Shared by `draw_image` and command-list replay.
    ///
    /// `clip` is an image rectangle's place (DIPs, world space). Direct2D draws the cropped list one of two ways:
    /// - a list it can inline under an axis-aligned transform renders clipped to it with the context's antialias mode;
    /// - otherwise it fills the rectangle with the rendered list as an image brush, at the rasteriser's coverage.
    ///   SOURCE_OVER, PLUS and BOUNDED_SOURCE_COPY fill the target directly (blend SOURCE_OVER, ADD, COPY). The
    ///   other modes fill a transparent intermediate and composite that over the whole target.
    fn composite_cmds(&mut self, cmds: &[RecCmd], t: [f32; 6], dpi: (f32, f32), composite: u32, clip: Option<[f32; 4]>) {
        // At a non-finite DPI (a SetDpi(inf, 96)) the list can't be placed: the draw leaves the target as it is, except that
        // SOURCE_COPY clears it (sc92: every mode at dpi inf; r24 29514). A non-finite world transform or offset still
        // composites the empty image (recap-a 634, r11 16410, r16 21569, r19 24529 clear under the IN modes).
        if self.get_unit_mode() == 0 && !(self.dpi.0.is_finite() && self.dpi.1.is_finite()) {
            if composite == 10 {
                self.composite_empty(10);
            }
            return;
        }
        // BOUNDED_SOURCE_COPY(11) of a command list is a whole-target SOURCE_COPY; under a rotation/shear
        // The fast image-draw path bails and DrawImage's textured rect copies the list over the target, as
        // long as its content reaches it (cl11rot-*, r7 8465). And an image with NO drawable content
        // copies nothing: SOURCE_COPY of an empty realized image leaves the backdrop, it does not clear
        // (fuzz 9171, a NaN-radius primitive). Otherwise it clears the whole target + the content (fuzz
        // 34, VM cl11-small-int/frac/span, byte-exact).
        // The fast image-draw path takes a translation-only transform whose device position lies within 1/32
        // of a whole pixel and draws the list there RoundSat'd to that pixel, for every mode it is tried
        // with (all but SOURCE_OVER, which inlines the list): 9-corpus seed 539's SOURCE_IN at
        // (−3.027, 23.002) puts a rect edge on 23.5, aliased-covered; cloff-* probes. BOUNDED_SOURCE_COPY
        // then copies over the whole clip whatever the list holds (bounded-copy probes: content outside
        // the target still clears it). Otherwise the slow path copies only when the list's content
        // bounds reach the target, transparent content included (fuzz r4 5501; r1 80 keeps the backdrop
        // for content wholly outside).
        let (sx, sy) = if self.get_unit_mode() == 0 { (dpi.0 / 96.0, dpi.1 / 96.0) } else { (1.0, 1.0) };
        let fast = t[..4] == [1.0, 0.0, 0.0, 1.0] && fast_whole(t[4] * sx) && fast_whole(t[5] * sy);
        if fast && (t[4].is_nan() || t[5].is_nan()) && !matches!(composite, 10 | 11) {
            return; // see `fast_whole`
        }
        // ponytail: off dpi 96 the snapped DIP translation n/scale is inexact in f32 (~1e-7 px), a
        // device-space replay origin if an edge ever lands on that sub-ulp.
        let rt = if fast { [t[0], t[1], t[2], t[3], (t[4] * sx).round() / sx, (t[5] * sy).round() / sy] } else { t };
        let w = self.state.transform;
        let clipped = clip.is_some() && !matches!(composite, 0 | 9 | 11) && ((w[1] == 0.0 && w[2] == 0.0) || (w[0] == 0.0 && w[3] == 0.0)) && crate::command::inline_safe(cmds);
        // The image brush a fill draws with renders the list over the fill's rounded device rect only: one that
        // covers no pixel renders nothing (r23 28478: a Flood in the list never reaches its intermediate check).
        let (tw, th) = self.target.as_ref().map_or((0.0, 0.0), |b| (b.desc.width as f32, b.desc.height as f32));
        let region = clip.filter(|_| !clipped).map(|c| {
            let d = self.device_clip_rect(c);
            [d[0].floor().max(0.0), d[1].floor().max(0.0), d[2].ceil().min(tw), d[3].ceil().min(th)]
        });
        let unseen = region.is_some_and(|p| !(p[0] < p[2] && p[1] < p[3]));
        if unseen && matches!(composite, 0 | 9 | 11) {
            return;
        }
        let Some(saved) = self.target.take() else { return };
        let mut inter = saved.clone();
        inter.data.fill(0);
        self.target = Some(inter);
        let (error, tags) = (self.error, self.state.tags);
        match (clipped, clip, region) {
            (true, Some(c), _) => self.push_axis_aligned_clip(c, self.state.aa),
            (_, _, Some(p)) if !unseen => {
                let (t, u) = (self.state.transform, self.state.unit);
                (self.state.transform, self.state.unit) = ([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 1);
                self.push_axis_aligned_clip(p, 1);
                (self.state.transform, self.state.unit) = (t, u);
            }
            _ => {}
        }
        // Off the fast path Direct2D renders the list into an intermediate over its own bounds, rounded out to whole
        // device pixels, so content past them never reaches the composite (r21 26088: a degenerate rounded rect's stroke
        // sliver above its widened bounds, cleared by the SOURCE_COPY).
        // ponytail: only a list of plain fills and strokes, whose bounds `cmds_rects` measures as native does; layers,
        // clips, images and text keep the unbounded intermediate (their measured bounds run short of native's: r24 29158).
        let plain = cmds.iter().all(|c| matches!(c, RecCmd::FillRect { .. } | RecCmd::FillPolys { .. } | RecCmd::FillGeom { .. }
            | RecCmd::StrokeGeom { .. } | RecCmd::FillRectBrush { .. } | RecCmd::FillGeomBrush { .. } | RecCmd::StrokeGeomBrush { .. }
            | RecCmd::Line { .. } | RecCmd::LineBrush { .. }));
        let bounds = if fast || !plain { None } else {
            let mut rects = Vec::new();
            self.cmds_rects(cmds, rt, dpi, &mut rects);
            rects.iter().filter(|q| !q.iter().any(|v| v.is_nan())).copied().reduce(|u, q| [u[0].min(q[0]), u[1].min(q[1]), u[2].max(q[2]), u[3].max(q[3])])
        };
        let bounds = bounds.map(|u| [u[0].floor().max(0.0), u[1].floor().max(0.0), u[2].ceil().min(tw), u[3].ceil().min(th)])
            .filter(|b| *b != [0.0, 0.0, tw, th]);
        // The list's intermediate takes a slot of the frame's atlas the size of those bounds, the target's when unbounded,
        // under every composite but SOURCE_COPY and BOUNDED_SOURCE_COPY (VM: an HQ cubic FOM's slot after a list of one
        // 30-wide fill drawn DESTINATION_OVER, SOURCE_OUT or PLUS is at x 30, after a SOURCE_COPY one at 0; after two
        // unbounded DESTINATION_OVER lists on a 128-wide target at 256). A list placed off to infinity draws nothing and
        // takes none.
        if !matches!(composite, 0 | 10 | 11) && !unseen && rt.iter().all(|v| v.is_finite()) {
            let mut rects = Vec::new();
            if plain { self.cmds_rects(cmds, rt, dpi, &mut rects); }
            let b = rects.iter().filter(|q| !q.iter().any(|v| v.is_nan())).copied().reduce(|u, q| [u[0].min(q[0]), u[1].min(q[1]), u[2].max(q[2]), u[3].max(q[3])])
                .map_or([0.0, 0.0, tw, th], |u| [u[0].floor().max(0.0), u[1].floor().max(0.0), u[2].ceil().min(tw), u[3].ceil().min(th)]);
            self.atlas.alloc((b[2] - b[0]) as i32, (b[3] - b[1]) as i32);
        }
        if let Some(b) = bounds.filter(|_| !unseen) {
            let (t, u) = (self.state.transform, self.state.unit);
            (self.state.transform, self.state.unit) = ([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 1);
            self.push_axis_aligned_clip(b, 1);
            (self.state.transform, self.state.unit) = (t, u);
        }
        if !unseen {
            let as_image = core::mem::replace(&mut self.playing_as_image, true);
            // The list's own slow-path draws do not take slots of the frame's atlas (VM: the FOM's slot after two lists
            // each holding a slow-path DrawImage is the lists' two regions only).
            let frame_atlas = core::mem::take(&mut self.atlas);
            self.replay(cmds, rt, dpi); // replay saves/clears/restores `recording` itself
            self.atlas = frame_atlas;
            self.playing_as_image = as_image;
        }
        if bounds.is_some() && !unseen {
            self.pop_axis_aligned_clip();
        }
        if clipped || (region.is_some() && !unseen) {
            self.pop_axis_aligned_clip();
        }
        // An image too large for its intermediate fails while the list renders as one image, so Direct2D reports it
        // with the tags of the draw that drew the list (sc49 list-xor).
        if error == S_OK && self.error == INTERMEDIATE_TOO_LARGE {
            self.err_tags = tags;
        }
        let inter = self.target.take().unwrap();
        self.target = Some(saved);
        let inter = match clip {
            Some(c) if !clipped => {
                let direct = matches!(composite, 0 | 9 | 11);
                let blend = [(0, 0), (9, 3), (11, 1)].iter().find(|p| p.0 == composite).map_or(0, |p| p.1);
                let saved = if direct { None } else { self.target.replace({ let mut t = inter.clone(); t.data.fill(0); t }) };
                self.fill_rect_with_buffer(c, &inter, blend);
                match saved {
                    None => return,
                    Some(s) => self.target.replace(s).unwrap(),
                }
            }
            _ => inter,
        };
        if composite == 11 && !fast {
            match self.cmds_reach_target(cmds, t, dpi) {
                Some(false) => return,
                // An empty list copies nothing where its bounds miss the target: placed finitely, at an infinite
                // translation (r23 28360) or under a singular linear part (r23 28746). At a NaN translation under an
                // invertible one its bounds are NaN, miss nothing, and its empty copy clears the target (r27 32637:
                // 3.4e38 times 1e6 plus -inf).
                None if inter.data.iter().all(|&b| b == 0) && !((t[4].is_nan() || t[5].is_nan()) && {
                    let det = t[0] * t[3] - t[1] * t[2];
                    det.is_finite() && det != 0.0
                }) => return,
                _ => {}
            }
        }
        // At a huge whole-pixel offset the list draws directly over the device span its f32 placement gives it (as a
        // Flood does, [`flood_span`](Self::flood_span)), so the blend stops there; the copy modes still clear the
        // rest (sc87, sc91: DEST_IN at −1e9 keeps the target where that span misses it; r24 29383).
        let p = [t[4] * sx, t[5] * sy];
        if fast && clip.is_none() && !matches!(composite, 10 | 11) && p.iter().any(|v| v.abs() >= 4_194_304.0) {
            if let Some((r, false)) = self.flood_span(p, 1) {
                if r[0] < r[2] && r[1] < r[3] {
                    self.composite_over_target(&inter, composite, Some(r));
                }
                return;
            }
        }
        self.composite_over_target(&inter, composite, None);
    }

    /// `DrawImage` of an effect output; with an image rectangle, the output cropped to it
    /// ([`draw_image_cropped`](Self::draw_image_cropped)).
    pub fn draw_image_effect_rect(&mut self, effect: &crate::effects::Effect, offset: [f32; 2], rect: Option<[f32; 4]>, interpolation: u32, composite: u32) {
        let a = self.image_api();
        self.draw_image_effect_rect_call(effect, offset, rect, interpolation, composite);
        self.image_queue(a);
    }

    fn draw_image_effect_rect_call(&mut self, effect: &crate::effects::Effect, offset: [f32; 2], rect: Option<[f32; 4]>, interpolation: u32, composite: u32) {
        match rect {
            Some(r) if self.recording.is_none() => {
                if self.ready_to_draw() && self.draw_image_args_ok(interpolation, composite) && effect.graph_valid() {
                    let w = self.state.transform;
                    if (w[1] != 0.0 || w[2] != 0.0) && interpolation == 1 && (self.state.unit == 1 || self.dpi == (96.0, 96.0)) {
                        self.draw_effect_crop_brush(effect, offset, r, composite);
                    } else {
                        let n = [r[0].min(r[2]), r[1].min(r[3]), r[0].max(r[2]), r[1].max(r[3])];
                        self.draw_image_cropped(offset, r, composite, |dc, o| {
                            let prev = crate::effects::IMAGE_CROP.with(|c| c.replace(Some(n)));
                            dc.draw_image_effect(effect, o, interpolation, 0);
                            crate::effects::IMAGE_CROP.with(|c| c.set(prev));
                        });
                    }
                } else {
                    self.draw_image_effect(effect, offset, interpolation, composite);
                }
            }
            // ponytail: a recorded effect keeps no rectangle (replayed uncropped); record it when a list scene needs one.
            _ => self.draw_image_effect(effect, offset, interpolation, composite),
        }
    }

    /// A rotated LINEAR [`draw_image_effect_rect`](Self::draw_image_effect_rect): Direct2D fills the placed crop with
    /// the effect as an image brush, the effect rendered unrotated into an atlas over [`crate::effects::crop_need`]'s
    /// rect (texel 0 at its top-left) and sampled by the brush, not alpha-blitted as an uncropped rotated draw is
    /// (sc94 fx-imgrect-blur-rotated, sc106: DDI dumps place that atlas at the crop's need).
    fn draw_effect_crop_brush(&mut self, effect: &crate::effects::Effect, offset: [f32; 2], rect: [f32; 4], composite: u32) {
        let n = [rect[0].min(rect[2]), rect[1].min(rect[3]), rect[0].max(rect[2]), rect[1].max(rect[3])];
        let crop = [offset[0], offset[1], offset[0] + (n[2] - n[0]), offset[1] + (n[3] - n[1])];
        let eo = [offset[0] - n[0], offset[1] - n[1]];
        let Some(target) = self.target.as_ref() else { return };
        let (tw, th) = (target.desc.width, target.desc.height);
        let m = self.device_transform().map(|v| v as f32);
        let need = crate::effects::crop_need(&m, crop, eo, [tw, th]);
        let (nw, nh) = (need[2] - need[0], need[3] - need[1]);
        if nw > 0 && nh > 0 {
            let side = (nw.max(nh) as u32).next_power_of_two().max(512);
            let mut atlas = target.clone();
            atlas.desc.width = side;
            atlas.desc.height = side;
            atlas.stride = side as usize * 4;
            atlas.data = vec![0; side as usize * side as usize * 4];
            let saved = self.target.replace(atlas);
            let (t, u) = (self.state.transform, self.state.unit);
            (self.state.transform, self.state.unit) = ([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 1);
            self.push_axis_aligned_clip([0.0, 0.0, nw as f32, nh as f32], 1);
            self.draw_image_effect(effect, [-need[0] as f32, -need[1] as f32], 1, 0);
            self.pop_axis_aligned_clip();
            (self.state.transform, self.state.unit) = (t, u);
            let atlas = std::mem::replace(&mut self.target, saved).unwrap();
            let bmp = Bitmap { width: side, height: side, premul_bgra: atlas.data, format: atlas.desc.format, alpha: crate::abi::D2D1_ALPHA_MODE_PREMULTIPLIED, dpi: (96.0, 96.0) };
            let brush = self.create_bitmap_brush(&bmp, [0, 0], 1, 1.0, Some([1.0, 0.0, 0.0, 1.0, eo[0] + need[0] as f32, eo[1] + need[1] as f32]));
            self.fill_crop_with_brush(crop, composite, &brush);
        } else if !matches!(composite, 0 | 9 | 11) {
            self.composite_empty(composite);
        }
    }

    /// Fill the DIP `crop` with `brush` under `composite`, as an image rectangle's fill: SOURCE_OVER, PLUS and
    /// BOUNDED_SOURCE_COPY fill the target (blend SOURCE_OVER, ADD, COPY), the other modes a transparent intermediate
    /// composited over the whole target.
    /// BOUNDED_SOURCE_COPY's image-rectangle fill off the fast path (a scaled draw, or a device offset off the whole
    /// pixel): native draws the DIP `crop` as one textured mesh from the opaque rasterizer's edge walk, with a geometry
    /// fill's pixel-centre convention ([`crate::pixels::copy_tess`]). Its whole rows ramp the coverage linearly across
    /// each vertical edge and its cap rows take the 8×8 scan, where the rect special case counts 8×8 samples
    /// everywhere. Each triangle samples `brush` through its own texcoord plane, transparent past the bitmap, and the
    /// output merger lerps the target toward that sample by the coverage (DDI: one draw of that mesh, its pixel shader
    /// a plain sample with the coverage as the second output; sc127 at 120 dpi and at a fractional offset, sc200).
    fn bcopy_crop_walk(&mut self, crop: [f32; 4], brush: &crate::PixelBrush) {
        let m = self.device_transform();
        let poly = [apply(&m, crop[0], crop[1]), apply(&m, crop[2], crop[1]), apply(&m, crop[2], crop[3]), apply(&m, crop[0], crop[3])];
        let Some(buf) = self.target.as_mut() else { return };
        let (w, h, stride) = (buf.desc.width as usize, buf.desc.height as usize, buf.stride);
        let vp = [0.0, 0.0, w as f32, h as f32];
        crate::pixels::copy_tess::fill_copy(&[&poly], true, w, h, true, |x, y, cov, tri| {
            let s = crate::brush::triangle_brush_sampler(tri, &brush.brush, false, vp)(x, y);
            let src = s.map(|v| v.round_ties_even().clamp(0.0, 255.0) as u8);
            let i = y * stride + x * 4;
            let out = cov.lerp(buf.data[i..i + 4].try_into().unwrap(), src);
            buf.data[i..i + 4].copy_from_slice(&out);
        });
    }

    pub(crate) fn image_texmap_pub(&self, bm: &Bitmap, offset: [f32; 2], p: [f32; 2]) -> [f32; 6] {
        self.image_texmap(bm, offset, p, [0.0; 2], 1)
    }

    fn fill_crop_with_brush(&mut self, crop: [f32; 4], composite: u32, brush: &crate::PixelBrush) {
        let direct = matches!(composite, 0 | 9 | 11);
        let blend = [(0, 0), (9, 3), (11, 1)].iter().find(|p| p.0 == composite).map_or(0, |p| p.1);
        // Precision store: with 16/32-bit-float rendering controls native realizes the draw's intermediate in that
        // float format (op8cap: the aniso realize atlas is R16G16B16A16_FLOAT at precision 4), so the sub-byte fraction
        // the 8bpc floor drops survives the opacity multiply and quantizes to UNORM8 only when it composites back. The
        // realize carries NO placement coverage — native's stage-2 places the float atlas with the fractional edge as
        // an 8-bit coverage ring — so the float intermediate is filled over the crop's whole-pixel bounds at full
        // coverage, and the eighths are applied at the write-back ([`Self::composite_float_inter_over_target`]). Gated
        // to an identity device transform (the fill is then whole-device-pixel aligned); other transforms and an
        // already-float target keep the 8bpc / whole-float path unchanged.
        let dt = self.device_transform();
        let fp = matches!(self.buffer_precision, 4 | 5) && !direct
            && matches!(brush.brush, crate::brush::Brush::Bitmap { interpolation: 4, .. })
            && dt == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            && self.target.as_ref().is_some_and(|t| !crate::float::is_highp(t.desc.format));
        let fill_rect = if fp { [crop[0].floor(), crop[1].floor(), crop[2].ceil(), crop[3].ceil()] } else { crop };
        let saved = if direct {
            None
        } else if fp {
            let t = self.target.as_ref().unwrap();
            let fmt = if self.buffer_precision == 5 { d2d_types::abi::DXGI_FORMAT_R32G32B32A32_FLOAT } else { d2d_types::abi::DXGI_FORMAT_R16G16B16A16_FLOAT };
            self.target.replace(crate::float::float_target(t.desc.width, t.desc.height, t.desc.dpi_x, fmt, t.desc.alpha_mode))
        } else {
            self.target.replace({ let mut z = self.target.clone().unwrap(); z.data.fill(0); z })
        };
        let aa = aa_of(self.state.aa);
        let prev = std::mem::replace(&mut self.state.blend, blend);
        self.fill_rectangle_brush(fill_rect, brush);
        self.state.blend = prev;
        if let Some(s) = saved {
            let filled = std::mem::replace(&mut self.target, Some(s)).unwrap();
            if fp {
                self.composite_float_inter_over_target(&filled, composite, Some((crop, aa)));
            } else if crate::float::is_highp(filled.desc.format) {
                let whole = [0, 0, filled.desc.width as i32, filled.desc.height as i32];
                crate::float::FloatTarget::new(self.target.as_mut().unwrap()).composite(&filled, composite, whole);
            } else {
                self.composite_over_target(&filled, composite, None);
            }
        }
    }

    /// Composite a premultiplied FLOAT intermediate `inter` (RGBA slot order) over the 8bpc BGRA target under
    /// `D2D1_COMPOSITE_MODE` `composite`, the precision-store path's quantize-on-write-back ([`Self::fill_crop_with_brush`]).
    /// The float realize value is rounded to UNORM8 ties-even, then — when `cov` gives the placement rectangle and its
    /// AA mode — weighted by the fractional eighths as WARP's fill kernel does (`cova = round(cov·255/64)`, product
    /// `(t + (t>>8) + 128) >> 8`), reproducing native's stage-2 placement ring; the blend is WARP's integer unit
    /// ([`crate::layer::composite`], byte-identical to the 8bpc path).
    fn composite_float_inter_over_target(&mut self, inter: &PixelBuffer, composite: u32, cov: Option<([f32; 4], crate::pixels::Aa)>) {
        let (w, h) = (inter.desc.width as usize, inter.desc.height as usize);
        // Per-pixel 8-bit placement coverage (0 outside the rect, so the unbounded modes clear there as the 8bpc path does).
        let covmap = cov.map(|(rect, aa)| {
            let mut m = vec![0u8; w * h];
            crate::pixels::rasterize_rect(rect, aa, w, h, |x, y, c| {
                m[y * w + x] = (c.min(64) as f32 * 255.0 / 64.0).round_ties_even() as u8;
            });
            m
        });
        let Some(target) = self.target.as_mut() else { return };
        let st = target.stride;
        for y in 0..h {
            for x in 0..w {
                let i = y * st + x * 4;
                let f = crate::float::read_pixel(inter, x, y); // [R, G, B, A], 0..1 premul
                let mut s: [i32; 4] = [f[2], f[1], f[0], f[3]].map(|c| (c * 255.0).round_ties_even().clamp(0.0, 255.0) as i32);
                if let Some(m) = &covmap {
                    let k = m[y * w + x] as i32;
                    s = s.map(|v| { let t = v * k; ((t + (t >> 8) + 128) >> 8).clamp(0, 255) });
                }
                let d = [target.data[i], target.data[i + 1], target.data[i + 2], target.data[i + 3]];
                if let Some(out) = crate::layer::composite(s, d, composite) {
                    target.data[i..i + 4].copy_from_slice(&out);
                }
            }
        }
    }

    /// `DrawImage` of an image cropped to its image rectangle `rect` (image space, either corner order), whose
    /// top-left lands at `offset`. Direct2D fills the placed rectangle with the image as an image brush at the
    /// rasteriser's coverage, as it does a command list off its inline path ([`composite_cmds`](Self::composite_cmds)):
    /// `draw` renders the uncropped image, at the offset less the rectangle's top-left, into a transparent target over
    /// the fill's rounded device rect only. SOURCE_OVER, PLUS and BOUNDED_SOURCE_COPY then fill the target with it
    /// (blend SOURCE_OVER, ADD, COPY: a coverage-weighted copy at the edges); the other modes fill a transparent
    /// intermediate, composited over the whole target, so SOURCE_COPY and the IN modes clear outside the rectangle,
    /// all of it for an empty one (sc94: Flood, GaussianBlur and ColorMatrix outputs, every interpolation).
    fn draw_image_cropped(&mut self, offset: [f32; 2], rect: [f32; 4], composite: u32, draw: impl FnOnce(&mut Self, [f32; 2])) {
        let r = [rect[0].min(rect[2]), rect[1].min(rect[3]), rect[0].max(rect[2]), rect[1].max(rect[3])];
        let clip = [offset[0], offset[1], offset[0] + (r[2] - r[0]), offset[1] + (r[3] - r[1])];
        let (tw, th) = self.target.as_ref().map_or((0.0, 0.0), |b| (b.desc.width as f32, b.desc.height as f32));
        let d = self.device_clip_rect(clip);
        let region = [d[0].floor().max(0.0), d[1].floor().max(0.0), d[2].ceil().min(tw), d[3].ceil().min(th)];
        let unseen = !(region[0] < region[2] && region[1] < region[3]);
        let direct = matches!(composite, 0 | 9 | 11);
        if unseen && direct {
            return;
        }
        let Some(saved) = self.target.take() else { return };
        let mut inter = saved.clone();
        inter.data.fill(0);
        self.target = Some(inter);
        if !unseen {
            let (t, u) = (self.state.transform, self.state.unit);
            (self.state.transform, self.state.unit) = ([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 1);
            self.push_axis_aligned_clip(region, 1);
            (self.state.transform, self.state.unit) = (t, u);
            draw(self, [offset[0] - r[0], offset[1] - r[1]]);
            self.pop_axis_aligned_clip();
        }
        let inter = self.target.replace(saved).unwrap();
        let blend = [(0, 0), (9, 3), (11, 1)].iter().find(|p| p.0 == composite).map_or(0, |p| p.1);
        let saved = if direct { None } else { self.target.replace({ let mut t = inter.clone(); t.data.fill(0); t }) };
        self.fill_rect_with_buffer(clip, &inter, blend);
        if let Some(s) = saved {
            let filled = self.target.replace(s).unwrap();
            self.composite_over_target(&filled, composite, None);
        }
    }

    /// Fill the DIP `rect` under the current transform with the device-aligned `px` as an image brush (each device
    /// pixel reads its own texel) under primitive blend `blend`: how Direct2D draws a command list cropped to an image
    /// rectangle off its inline path.
    fn fill_rect_with_buffer(&mut self, rect: [f32; 4], px: &PixelBuffer, blend: u32) {
        let (w, h) = (px.desc.width, px.desc.height);
        let mut texels = Vec::with_capacity(w as usize * h as usize * 4);
        for y in 0..h as usize {
            texels.extend_from_slice(&px.data[y * px.stride..y * px.stride + w as usize * 4]);
        }
        let bmp = Bitmap { width: w, height: h, premul_bgra: texels, format: px.desc.format, alpha: crate::abi::D2D1_ALPHA_MODE_PREMULTIPLIED, dpi: (96.0, 96.0) };
        let to_device = crate::brush::invert(&self.device_transform().map(|v| v as f32));
        let brush = self.create_bitmap_brush(&bmp, [0, 0], 0, 1.0, Some(to_device));
        let prev = std::mem::replace(&mut self.state.blend, blend);
        self.fill_rectangle_brush(rect, &brush);
        self.state.blend = prev;
    }

    /// Each recorded command's device bounds under `record_world · t` at `dpi`, rounded out to whole pixels, into
    /// `out` (a nested list's commands too); false when a glyph run, a float blit or an effect leaves them undecided.
    pub(crate) fn cmds_rects(&mut self, cmds: &[RecCmd], t: [f32; 6], dpi: (f32, f32), out: &mut Vec<[f32; 4]>) -> bool {
        let (saved_state, saved_dpi) = (self.state, self.dpi);
        let mut undecided = false;
        // The open layers' bounds and clips, in device space: a list's bounds are its content cut by each
        // enclosing one (r24 29813: content inside layers that lie past the target gives a
        // BOUNDED_SOURCE_COPY nothing to copy; cllay probes).
        let mut cuts: Vec<[f32; 4]> = Vec::new();
        for cmd in cmds {
            let corners = |m: &[f32; 6], r: [f32; 4]| [[r[0], r[1]], [r[2], r[1]], [r[0], r[3]], [r[2], r[3]]].map(|[x, y]| [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]);
            let hull = |p: &[[f32; 2]]| {
                let f = |k: usize, lo: bool| p.iter().map(|q| q[k]).fold(if lo { f32::INFINITY } else { f32::NEG_INFINITY }, |a, v| if lo { a.min(v) } else { a.max(v) });
                [f(0, true), f(1, true), f(0, false), f(1, false)]
            };
            let start = out.len();
            let bounds = match cmd {
                RecCmd::Clear { .. } | RecCmd::Flood { crop: None, .. } => Some([f32::NEG_INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::INFINITY]),
                // A cropped flood draws its placed image rectangle.
                RecCmd::Flood { state, crop: Some((o, r)), .. } => {
                    self.set_replay_state(state, &t, dpi);
                    let c = self.device_clip_rect([o[0], o[1], o[0] + (r[2] - r[0]).abs(), o[1] + (r[3] - r[1]).abs()]);
                    Some([c[0].floor(), c[1].floor(), c[2].ceil(), c[3].ceil()])
                }
                RecCmd::PushLayer { state, bounds: r, .. } | RecCmd::PushClip { state, rect: r, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    let m = self.device_transform().map(|v| v as f32);
                    let c = hull(&corners(&m, *r));
                    // A NaN edge (an infinite bound under a zero scale) cuts nothing.
                    cuts.push(if c.iter().any(|v| v.is_nan()) { [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::INFINITY] } else { c });
                    None
                }
                RecCmd::PopLayer | RecCmd::PopClip => {
                    cuts.pop();
                    None
                }
                RecCmd::GlyphRun { .. } | RecCmd::BlitFloat { .. } | RecCmd::Effect { .. } | RecCmd::Ink { .. } | RecCmd::InkBrush { .. } => {
                    undecided = true;
                    None
                }
                // An image drawn under a mode where transparent black still changes the target (the IN, OUT, DEST_ATOP
                // and copy modes) bounds the list by its image rectangle, or without one by the whole plane, whatever it
                // holds (r27 32486: a SOURCE_OUT bitmap makes a bounded copy of the list clear a target its content misses).
                RecCmd::DrawImage { composite, source: None, .. } | RecCmd::DrawEffect { composite, .. } | RecCmd::DrawImageList { composite, rect: None, .. }
                    if transparent_black_affects_output(*composite) =>
                {
                    Some([-f32::MAX, -f32::MAX, f32::MAX, f32::MAX])
                }
                RecCmd::DrawImageList { state, offset, rect: Some(r), composite, .. } if transparent_black_affects_output(*composite) => {
                    self.set_replay_state(state, &t, dpi);
                    Some(self.device_clip_rect([offset[0], offset[1], offset[0] + (r[2] - r[0]), offset[1] + (r[3] - r[1])]))
                }
                RecCmd::DrawImageList { state, list, offset, rect, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    let o = rect.map_or(*offset, |r| [offset[0] - r[0], offset[1] - r[1]]);
                    let inner_t = crate::geometry::mat_mul(&[1.0, 0.0, 0.0, 1.0, o[0], o[1]], &self.state.transform);
                    let inner = list.0.borrow().cmds.clone();
                    let n = out.len();
                    undecided |= !self.cmds_rects(&inner, inner_t, dpi, out);
                    // An image rectangle bounds what the nested list draws.
                    if let Some(r) = rect {
                        let c = self.device_clip_rect([offset[0], offset[1], offset[0] + (r[2] - r[0]), offset[1] + (r[3] - r[1])]);
                        let c = [c[0].floor(), c[1].floor(), c[2].ceil(), c[3].ceil()];
                        let kept: Vec<[f32; 4]> = out.drain(n..).map(|q| [q[0].max(c[0]), q[1].max(c[1]), q[2].min(c[2]), q[3].min(c[3])]).filter(|q| q[0] < q[2] && q[1] < q[3]).collect();
                        out.extend(kept);
                    }
                    None
                }
                _ => {
                    let state = match cmd {
                        RecCmd::FillRect { state, .. } | RecCmd::FillPolys { state, .. } | RecCmd::FillGeom { state, .. } | RecCmd::StrokeGeom { state, .. }
                        | RecCmd::FillRectBrush { state, .. } | RecCmd::FillGeomBrush { state, .. } | RecCmd::StrokeGeomBrush { state, .. }
                        | RecCmd::Line { state, .. } | RecCmd::LineBrush { state, .. }
                        | RecCmd::DrawImage { state, .. } | RecCmd::DrawEffect { state, .. } | RecCmd::SpriteBatch { state, .. }
                        | RecCmd::Mesh { state, .. } | RecCmd::MeshBrush { state, .. } | RecCmd::DrawBitmap { state, .. }
                        | RecCmd::OpacityMask { state, .. } | RecCmd::OpacityMaskBrush { state, .. } | RecCmd::GradientMesh { state, .. } => state,
                        _ => unreachable!(),
                    };
                    let under = |this: &mut Self, t: &[f32; 6]| {
                        this.set_replay_state(state, t, dpi);
                        let m = this.device_transform().map(|v| v as f32);
                        match cmd {
                            RecCmd::FillRect { rect, .. } | RecCmd::FillRectBrush { rect, .. } => hull(&corners(&m, *rect)),
                            RecCmd::DrawBitmap { dest, .. } => hull(&corners(&m, *dest)),
                            RecCmd::GradientMesh { patches, .. } => hull(&patches.iter().flat_map(|p| p.points).map(|[x, y]| [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]).collect::<Vec<_>>()),
                            RecCmd::OpacityMask { bitmap, dest, .. } | RecCmd::OpacityMaskBrush { bitmap, dest, .. } => {
                                hull(&corners(&m, dest.unwrap_or([0.0, 0.0, bitmap.width as f32, bitmap.height as f32])))
                            }
                            RecCmd::Mesh { tri, .. } | RecCmd::MeshBrush { tri, .. } => hull(&tri.map(|[x, y]| [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]])),
                            RecCmd::FillPolys { figures, .. } => hull(&figures.iter().flatten().map(|&[x, y]| [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]).collect::<Vec<_>>()),
                            RecCmd::FillGeom { geom, .. } | RecCmd::FillGeomBrush { geom, .. } => geom.get_bounds(&m),
                            RecCmd::StrokeGeom { geom, width, style, .. } | RecCmd::StrokeGeomBrush { geom, width, style, .. } => geom.widened_bounds(*width, style, &m, 0.25),
                            RecCmd::Line { p0, p1, width, style, .. } | RecCmd::LineBrush { p0, p1, width, style, .. } => {
                                line_geometry(*p0, *p1).widened_bounds(*width, &style.clone().unwrap_or_default(), &m, 0.25)
                            }
                            RecCmd::DrawImage { bitmap, offset, source, .. } => {
                                let s = source.unwrap_or([0.0, 0.0, bitmap.width as f32, bitmap.height as f32]);
                                hull(&corners(&m, [offset[0], offset[1], offset[0] + s[2] - s[0], offset[1] + s[3] - s[1]]))
                            }
                            RecCmd::DrawEffect { bitmap, dest0, .. } => hull(&corners(&m, [dest0[0], dest0[1], dest0[0] + bitmap.width as f32, dest0[1] + bitmap.height as f32])),
                            // Every sprite's destination corners, through its own matrix and then the context's.
                            RecCmd::SpriteBatch { sprites, .. } => hull(&sprites.iter().flat_map(|sp| {
                                let t = sp.transform.unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
                                corners(&t, sp.dest).map(|[x, y]| [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]])
                            }).collect::<Vec<_>>()),
                            _ => unreachable!(),
                        }
                    };
                    // The list's bounds are taken as it records: a draw whose recorded bounds are not finite (an
                    // infinite rect, an ellipse of infinite radius) adds none, whatever the transform it is drawn under
                    // (r20 25168). Finite ones are then placed under it, even where that overflows them.
                    let recorded = under(self, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
                    recorded.iter().all(|v| v.is_finite()).then(|| under(self, &t))
                }
            };
            // The bound is rounded out to whole pixels: a zero width or height inside a pixel covers it (r15
            // 20109: an ellipse of x radius 1e-20 clears the target), one on a pixel edge covers nothing (a rect
            // collapsed onto x = 10 keeps the backdrop). A NaN or empty (inf, −inf) bound covers nothing.
            if let Some([l, t0, r, b]) = bounds && r >= l && b >= t0 {
                let px = [l.floor(), t0.floor(), r.ceil(), b.ceil()];
                if px[2] > px[0] && px[3] > px[1] {
                    out.push(px);
                }
            }
            // Cut what this command added (a nested list's rects too) by the open layers and clips.
            let added: Vec<[f32; 4]> = out.drain(start..).collect();
            out.extend(added.into_iter().filter_map(|mut px| {
                for c in &cuts {
                    px = [px[0].max(c[0].floor()), px[1].max(c[1].floor()), px[2].min(c[2].ceil()), px[3].min(c[3].ceil())];
                }
                (px[2] > px[0] && px[3] > px[1]).then_some(px)
            }));
        }
        (self.state, self.dpi) = (saved_state, saved_dpi);
        !undecided
    }

    /// Whether the list's device bounds reach the bound target, the slow path's emptiness test. The bounds are the
    /// union of every recorded command's (under `record_world · t` at `dpi`, [`Self::cmds_rects`]), so draws that each
    /// miss the target but straddle it reach it (r23 28504: a FillMesh's triangles, or two draws, on either side of
    /// it; sc103). `None` when a glyph run leaves it undecided.
    fn cmds_reach_target(&mut self, cmds: &[RecCmd], t: [f32; 6], dpi: (f32, f32)) -> Option<bool> {
        let [cl, ct, tw, th] = self.reach_rect();
        let mut rects = Vec::new();
        let decided = self.cmds_rects(cmds, t, dpi, &mut rects);
        let union = rects.iter().filter(|q| !q.iter().any(|v| v.is_nan())).copied().reduce(|u, q| [u[0].min(q[0]), u[1].min(q[1]), u[2].max(q[2]), u[3].max(q[3])]);
        if union.is_some_and(|[l, t0, r, b]| l < tw && t0 < th && r > cl && b > ct) {
            Some(true)
        } else if decided {
            Some(false)
        } else {
            None
        }
    }

    /// How an image whose bounds reach past ±2^21 (an infinite image) draws under a world transform that is not a
    /// translation. Direct2D renders it into an intermediate in its own pixel space, one tile of the draw at a time:
    /// - The draw covers the pixels whose centres the image's mapped bounds contain, within the clip.
    /// - Each side of that rect is halved until it is 16 px or less; those are the tiles.
    /// - A tile needs the image over its pixel centres (pixel edges for MULTI_SAMPLE_LINEAR), mapped back through the
    ///   transform's adjoint in f32. That rect is grown by the filter's reach and saturated to i32.
    /// - ANISOTROPIC asks for its tile grown by a pixel each side within the target.
    /// - A tile whose need is empty draws nothing.
    /// - A need wider or taller than 4096 px fails the whole draw with INTERMEDIATE_TOO_LARGE.
    /// `m` maps image pixels to device pixels. (167 VM probes: scales, rotations, clips, every interpolation.)
    pub(crate) fn transformed_image_need(&self, m: [f32; 6], bounds: [i32; 4], interp: u32) -> ImageNeed {
        let [a, b, c, d, tx, ty] = m;
        let map = |x: f32, y: f32| [a * x + c * y + tx, b * x + d * y + ty];
        let [l, t, r, bt] = bounds.map(|v| v as f32);
        let q = [map(l, t), map(r, t), map(l, bt), map(r, bt)];
        let lo = |k: usize| q.iter().map(|p| p[k]).fold(f32::INFINITY, f32::min);
        let hi = |k: usize| q.iter().map(|p| p[k]).fold(f32::NEG_INFINITY, f32::max);
        let clip = self.reach_rect();
        let reg = [
            (lo(0) - 0.5).ceil().max(clip[0]),
            (lo(1) - 0.5).ceil().max(clip[1]),
            (hi(0) - 0.5).ceil().min(clip[2]),
            (hi(1) - 0.5).ceil().min(clip[3]),
        ]
        .map(|v| v as i32);
        self.tile_need(m, reg, interp, 4096)
    }

    /// The tiles of the device pixel rect `reg` that [`transformed_image_need`](Self::transformed_image_need)
    /// describes, each needing its image through `m` within `limit` px on a side.
    fn tile_need(&self, m: [f32; 6], reg: [i32; 4], interp: u32, limit: i64) -> ImageNeed {
        if reg[0] >= reg[2] || reg[1] >= reg[3] {
            return ImageNeed::Nothing;
        }
        let (mut xs, mut ys) = (Vec::new(), Vec::new());
        halves(reg[0], reg[2], &mut xs);
        halves(reg[1], reg[3], &mut ys);
        let (tw, th) = self.target.as_ref().map_or((0, 0), |b| (b.desc.width as i32, b.desc.height as i32));
        let mut draws = false;
        for &(x0, x1) in &xs {
            for &(y0, y1) in &ys {
                let tile = if interp == 4 { [(x0 - 1).max(0), (y0 - 1).max(0), (x1 + 1).min(tw), (y1 + 1).min(th)] } else { [x0, y0, x1, y1] };
                let need = adjoint_need(m, tile, interp);
                let (w, h) = (need[2] as i64 - need[0] as i64, need[3] as i64 - need[1] as i64);
                if w <= 0 || h <= 0 {
                    continue;
                }
                if w > limit || h > limit {
                    return ImageNeed::TooLarge;
                }
                draws = true;
            }
        }
        if draws { ImageNeed::Draws } else { ImageNeed::Nothing }
    }

    /// [`transformed_image_need`](Self::transformed_image_need) for a DrawImage of an image with `bounds` at `offset`
    /// under the current world transform, acted on: a draw that fails latches INTERMEDIATE_TOO_LARGE, and one that
    /// covers nothing copies its empty image under SOURCE_COPY and is otherwise a no-op. True when the draw is done.
    /// Images render in device-pixel space, so the transform stage scales by the world matrix alone.
    pub(crate) fn transformed_image_done(&mut self, bounds: [i32; 4], offset: [f32; 2], interp: u32, composite: u32) -> bool {
        let w = self.state.transform;
        const REACH: i32 = 1 << 21;
        // A transform with no f32 inverse keeps the unplaceable image's own handling (draw_flood's `inv_fails`).
        let det = w[0] * w[3] - w[1] * w[2];
        let invertible = det != 0.0 && ((1.0_f64 / det as f64) as f32).is_finite();
        if w[..4] == [1.0, 0.0, 0.0, 1.0] || !invertible || interp > 5 || !(bounds[0] < -REACH || bounds[1] < -REACH || bounds[2] > REACH || bounds[3] > REACH) {
            return false;
        }
        let k = if self.state.unit == 0 { [self.dpi.0 / 96.0, self.dpi.1 / 96.0] } else { [1.0; 2] };
        let m = [w[0], w[1], w[2], w[3], k[0] * (w[0] * offset[0] + w[2] * offset[1] + w[4]), k[1] * (w[1] * offset[0] + w[3] * offset[1] + w[5])];
        // Inside a layer whose bound is NaN the image covers nothing: native draws nothing there, so it never reaches the
        // size check (sc93: a Flood in such a layer under a 1e-6 scale is S_OK, bare or in a finite layer it fails).
        let nan_layer = self.stack.iter().any(|f| matches!(f, crate::layer::Frame::Layer { params, .. } if params.bounds_nan));
        let need = if nan_layer { ImageNeed::Nothing } else { self.transformed_image_need(m, bounds, interp) };
        match need {
            ImageNeed::Draws => return false,
            ImageNeed::TooLarge => self.process_new_error(INTERMEDIATE_TOO_LARGE),
            ImageNeed::Nothing if composite == 10 => self.composite_empty(10),
            ImageNeed::Nothing => {}
        }
        true
    }

    /// Whether an image drawn at device offset `p` shows nothing because its visible region (the clips within the
    /// target), taken into the image's own space in f32, collapses on an axis: past about 2^25 the f32 spacing outgrows
    /// a narrow region, so `lo − p` and `hi − p` round together (1e9 has a spacing of 64: a 32-px-wide target draws
    /// nothing, a 33-px one draws). A non-finite offset collapses too. VM probes atlasplan sc67–sc71; r20 25696.
    // ponytail: a merely huge offset can also leave part of a wide region undrawn (r20 25991: 64 of 96 columns at
    // 1e9 under DEST_OVER); a pixel-centre need explains that one but not a SOURCE_ATOP/cubic draw (r21 26106).
    /// The device rect a DrawImage of the (infinite) Flood covers at device offset `p`, as Direct2D places it in f32: at
    /// a huge offset the image-space rect `visible − p` rounds to the f32 grid, and that rounding decides how much of the
    /// target the flood covers (r20 25991; sc85: 1e9 fills 48 or 64 of 96 columns).
    /// - A fractional position renders an atlas over the visible pixel centres in image space, `⌊c₀+½−p − ½⌋` to
    ///   `⌈(c₁−½−p) + ½⌉` (each sum rounded to f32), and blits it back at `+p`.
    /// - A whole position draws the image-space rect `c − p` directly through the viewport's NDC transform:
    ///   `ndc = x·a + ((±2p)/size ∓ 1)` with `a = ±2/size` in f32, the offset's term rounded once (a quotient, not the
    ///   product `p·a`), covering the pixel centres between. At 1e9+40 (f32 1e9+64) both corners land on one NDC value
    ///   and the flood draws nothing; at 120 dpi the device offset `(offset + translation)·dpi/96` takes the same rule
    ///   (sc85, sc89: 23 whole positions).
    /// Which path depends on the interpolation (sc86): NEAREST/LINEAR/CUBIC as above; ANISOTROPIC always draws directly;
    /// HIGH_QUALITY_CUBIC draws directly at a whole position and covers the whole reach at a fractional one; and
    /// MULTI_SAMPLE_LINEAR always covers the whole reach (`None`).
    /// Returns the rect and whether it came from the atlas path.
    pub(crate) fn flood_span(&self, p: [f32; 2], interp: u32) -> Option<([i32; 4], bool)> {
        let c = self.reach_rect();
        let size = self.target.as_ref().map_or([0.0; 2], |t| [t.desc.width as f32, t.desc.height as f32]);
        let mut r = [0i32; 4];
        let whole = fast_whole(p[0]) && fast_whole(p[1]);
        if interp == 3 || (interp == 5 && !whole) {
            return None;
        }
        let atlas = !whole && interp != 4;
        for k in 0..2 {
            let (lo, hi) = if atlas {
                let lo = ((c[k] + 0.5 - p[k]) - 0.5).floor() as f64;
                let hi = ((c[k + 2] - 0.5 - p[k]) + 0.5).ceil() as f64;
                ((lo + p[k] as f64).floor(), (hi + p[k] as f64).ceil())
            } else {
                let (a, sign) = if k == 0 { (2.0 / size[0], -1.0) } else { (-2.0 / size[1], 1.0) };
                let t = (p[k] * if k == 0 { 2.0 } else { -2.0 }) / size[k] + sign;
                let dev = |x: f32| {
                    let ndc = (x * a + t) as f64;
                    if k == 0 { (ndc + 1.0) * size[0] as f64 / 2.0 } else { (1.0 - ndc) * size[1] as f64 / 2.0 }
                };
                let (d0, d1) = (dev(c[k] - p[k]), dev(c[k + 2] - p[k]));
                // The pixels whose centres the span holds.
                ((d0.min(d1) - 0.5).ceil(), (d0.max(d1) - 0.5).ceil())
            };
            let clamp = |v: f64| v.clamp(i32::MIN as f64, i32::MAX as f64) as i32;
            (r[k], r[k + 2]) = (clamp(lo), clamp(hi));
        }
        Some((r, atlas))
    }

    /// The device rect the Flood covers at a fractional position off unit scale, NEAREST/LINEAR/CUBIC/MULTI_SAMPLE_LINEAR,
    /// and the device edge `hi·a + b` per axis. Such a DrawImage
    /// leaves the fast path and fills the target with the flood as an image brush, mapped to the device by `x·a + b`:
    /// - the brush's atlas holds the flood over the image rect the visible pixel centres need through that map
    ///   ([`adjoint_need`]), at the atlas origin;
    /// - the fill maps that rect's right edge forward (`hi·a + b`) and clamps left of the rect, so the flood covers the
    ///   visible rect up to that edge, and past it the brush is transparent.
    /// SOURCE_OVER, PLUS and BOUNDED_SOURCE_COPY fill directly, `a = 1` and `b` the device offset. The other modes record
    /// the draw into a command list and composite its replay; the replay target reapplies the user transform `T` as
    /// `(T·S)·S⁻¹` in f32 (`S` the dpi scale), and at 120 and 240 dpi that round trip leaves a scale one ulp under 1, so
    /// `a` and `b` land a few ulps from 1 and the device offset.
    /// MULTI_SAMPLE_LINEAR reads the atlas edge with its two sub-samples per axis, so the flood fades over the pixel
    /// before and after that edge ([`msl_edge_cover`]).
    /// (d2d1 reading of both paths; VM DDI atlas quads geo/fl and geo/fls, 40 of 40 exact, and their outputs; MSL:
    /// VM probes geo/flw at 120 dpi, offsets 5e8 and 7e8, SOURCE_OVER and DESTINATION_OVER.)
    // ponytail: the replay tile is taken at the target origin (no tile offset), and a dpi whose scale and 96/dpi do
    // not multiply back to 1 keeps the whole-target draw; extend both when a scene needs them.
    /// The image-to-device matrix of a translated flood off unit dpi: the target's device transform (T a translation) is
    /// T·S, or for a composite mode that replays T·S·S⁻¹·S. `None` where the dpi scale's own reciprocal is inexact.
    fn flood_matrix(&self, offset: [f32; 2], composite: u32) -> Option<[f32; 6]> {
        let s = [self.dpi.0 / 96.0, self.dpi.1 / 96.0];
        let r = [96.0 / self.dpi.0, 96.0 / self.dpi.1];
        if s[0] * r[0] != 1.0 || s[1] * r[1] != 1.0 {
            return None;
        }
        let w = self.state.transform;
        let replay = !matches!(composite, 0 | 9 | 11);
        let inv = 1.0 / (s[0] * s[1]);
        let mut m = [0.0f32; 6];
        for k in 0..2 {
            let q = s[1 - k] * inv;
            let (dm, dt) = if replay { (s[k] * (q * s[k]), s[k] * (q * (s[k] * w[4 + k]))) } else { (s[k], s[k] * w[4 + k]) };
            m[k * 3] = dm * r[k];
            m[4 + k] = offset[k] * dm + dt;
        }
        Some(m)
    }

    fn flood_brush_span(&self, offset: [f32; 2], interp: u32, composite: u32) -> Option<([i32; 4], [f32; 2])> {
        if interp > 4 {
            return None;
        }
        let m = self.flood_matrix(offset, composite)?;
        let c = self.reach_rect().map(|v| v as i32);
        let need = adjoint_need(m, c, interp);
        if need[2] <= need[0] || need[3] <= need[1] {
            return Some(([c[0], c[1], c[0], c[1]], [f32::INFINITY; 2]));
        }
        let (mut out, mut ends) = (c, [f32::INFINITY; 2]);
        if interp == 4 {
            // ANISOTROPIC: the brush is redirected, so a quad over the reach draws the atlas through per-vertex texcoords
            // `f32(corner − need)` (need in image space, a ≈ 1). Huge, they round to the f32 grid, and the quad
            // interpolates them exactly between the corners: the flood shows where that runs inside the atlas's
            // `need` width (VM DDI geo/flw at 120 and 192 dpi: a 96-px quad spanning 128 texels of a 64-texel atlas
            // covers 48 columns).
            for k in 0..2 {
                // The atlas sits at the reach's left edge, so the texel is the interpolated value less `cn` and that edge.
                let cn = -(need[k] as f32);
                let edge = c[k] as f64;
                let (q0, q1) = (c[k] as f32, c[k + 2] as f32);
                let (w0, w1) = ((q0 + cn) as f64 - cn as f64 - edge, (q1 + cn) as f64 - cn as f64 - edge);
                let width = need[k + 2] as f64 - need[k] as f64;
                let texel = |x: i32| w0 + (w1 - w0) * ((x as f64 + 0.5 - q0 as f64) / (q1 as f64 - q0 as f64));
                let lo = (c[k]..c[k + 2]).find(|&x| texel(x) >= 0.0).unwrap_or(c[k + 2]);
                out[k] = lo;
                out[k + 2] = (lo..c[k + 2]).find(|&x| texel(x) >= width).unwrap_or(c[k + 2]);
            }
            return Some((out, ends));
        }
        for k in 0..2 {
            let end = need[k + 2] as f32 * m[k * 3] + m[4 + k];
            // MSL's last sub-sample at x + 0.25 still reads the edge texel below end + 0.5.
            let last = if interp == 3 { end + 0.25 } else { end - 0.5 };
            out[k + 2] = (last.ceil() as i32).max(c[k]).min(c[k + 2]);
            ends[k] = end;
        }
        Some((out, ends))
    }

    pub(crate) fn offset_collapses(&self, p: [f32; 2]) -> bool {
        let c = self.reach_rect();
        (0..2).any(|k| !p[k].is_finite() || !((c[k] - p[k]) < (c[k + 2] - p[k])))
    }

    /// Whether the fast image-draw path places this DrawImage at a NaN device position ([`fast_whole`]).
    /// It needs a translation-only world transform and places per axis, `(offset + translation)·dpi/96`
    /// (no 0·NaN cross term). It is never taken for MULTI_SAMPLE_LINEAR, nor for a bitmap off dpi 96,
    /// whose own 96 dpi then mismatches the context's.
    pub(crate) fn nan_fast_placement(&self, offset: [f32; 2], interpolation: u32, bitmap: bool) -> bool {
        let w = self.state.transform;
        let s = if self.state.unit == 0 { [self.dpi.0 / 96.0, self.dpi.1 / 96.0] } else { [1.0; 2] };
        let p = [(offset[0] + w[4]) * s[0], (offset[1] + w[5]) * s[1]];
        w[..4] == [1.0, 0.0, 0.0, 1.0] && (!bitmap || s == [1.0; 2]) && interpolation != 3
            && p.iter().any(|v| v.is_nan()) && fast_whole(p[0]) && fast_whole(p[1])
    }

    /// An unplaceable image (non-finite or singular transform) realizes empty and still composites:
    /// the modes that zero the destination under a transparent source (IN, DEST_IN, OUT, DEST_ATOP,
    /// COPY) clear it, the rest keep it, and BOUNDED_SOURCE_COPY has no bounds to copy (dimg-inf-*,
    /// dimg-sing-*, recap9 11709).
    pub(crate) fn composite_empty(&mut self, composite: u32) {
        if composite == 11 { return; }
        let Some(t) = self.target.as_ref() else { return };
        let mut inter = t.clone();
        inter.data.fill(0);
        self.composite_over_target(&inter, composite, None);
    }

    /// Composite a fully-realized, target-sized `inter` image over the bound target as a unit, with
    /// the full `D2D1_COMPOSITE_MODE` table (`layer::composite`) — native's DrawImage slow path
    /// (native's slow-path command list): the image is realized into a transparent
    /// intermediate, then blended over every target pixel. `bounds` (device pixels) limits the blend
    /// to a rectangle — native's bounded fast path for a *finite* image (BOUNDED_SOURCE_COPY leaves
    /// the target outside the image rect untouched); `None` blends the whole target. Shared by
    /// `draw_image` (command list), `draw_image_bitmap` and `draw_image_effect`.
    pub(crate) fn composite_over_target(&mut self, inter: &PixelBuffer, composite: u32, bounds: Option<[i32; 4]>) {
        let Some(target) = self.target.as_mut() else { return };
        let (w, h, st) = (target.desc.width as i32, target.desc.height as i32, target.stride);
        let [x0, y0, x1, y1] = bounds.unwrap_or([0, 0, w, h]);
        let (x0, y0, x1, y1) = (x0.clamp(0, w), y0.clamp(0, h), x1.clamp(0, w), y1.clamp(0, h));
        for y in y0..y1 {
            for x in x0..x1 {
                let i = y as usize * st + x as usize * 4;
                let s: [i32; 4] = core::array::from_fn(|k| inter.data[i + k] as i32);
                let d = [target.data[i], target.data[i + 1], target.data[i + 2], target.data[i + 3]];
                if let Some(out) = crate::layer::composite(s, d, composite) {
                    target.data[i..i + 4].copy_from_slice(&out);
                }
            }
        }
    }

    /// Set the live draw state for replaying a recorded command: its record-time state (AA/blend/unit)
    /// with the world transform composed with the replay transform `t`, and the replay DPI.
    fn set_replay_state(&mut self, state: &crate::state::DrawState, t: &[f32; 6], dpi: (f32, f32)) {
        self.state = *state;
        self.state.transform = crate::geometry::mat_mul(&state.transform, t);
        self.dpi = dpi;
    }

    /// Replay a recorded list under transform `t` (the DrawImage placement composed with the replay
    /// context's world transform) at `dpi`. Each recorded draw is re-issued through the *same* public
    /// draw method with the context transform set to `record_world · t`, so replay ≡ direct at that
    /// transform: curves re-flatten, strokes re-widen and brushes re-realize under the composed
    /// transform. Saves/restores the live state, DPI and `recording` (draws go to the target, not back
    /// into the list).
    fn replay(&mut self, cmds: &[RecCmd], t: [f32; 6], dpi: (f32, f32)) {
        let saved_state = self.state;
        let saved_dpi = self.dpi;
        let saved_rec = self.recording.take();
        // Each replayed clip: pushed as an axis-aligned clip (false) or, under a transform that does not
        // keep axes, as a layer masked by the rect (true).
        let mut clip_as_layer: Vec<bool> = Vec::new();
        self.replay_depth += 1;
        let outer_tags = self.play_tags;
        self.play_tags = Some(outer_tags.unwrap_or(saved_state.tags));
        let outer_unplaced = self.play_unplaced;
        // An infinite offset places nothing (the NaN beside it is the offset composed through a zero term).
        self.play_unplaced |= t.iter().any(|v| v.is_infinite());
        // Bounds that cover the whole plane stay that way under any placement, so a list holding them plays at a
        // non-finite one too (the DPI scale can overflow a finite offset); finite bounds map to nothing there. Only an
        // HQ-cubic DrawBitmap asks.
        let outer_unbounded = self.play_unbounded;
        self.play_unbounded = cmds.iter().any(|c| matches!(c, RecCmd::DrawBitmap { interpolation: 5, .. })) && {
            let mut rects = Vec::new();
            self.cmds_rects(cmds, t, dpi, &mut rects);
            rects.iter().any(|r| r[0] <= -f32::MAX && r[1] <= -f32::MAX && r[2] >= f32::MAX && r[3] >= f32::MAX)
        };
        for cmd in cmds {
            match cmd {
                RecCmd::Clear { state, color, .. } => {
                    // A Clear fills the list's drawn area regardless of transform (the list image is
                    // infinite), so re-issue it as a target clear — an opaque Clear covers the whole
                    // target, matching native (replaying onto the composite intermediate for a
                    // non-SOURCE_OVER list, or directly for SOURCE_OVER). ponytail: a transparent/semi
                    // Clear under a SOURCE_OVER inline replay replaces rather than composites; no corpus
                    // scene exercises that (Clears here are opaque).
                    self.set_replay_state(state, &t, dpi);
                    self.clear(Some(*color));
                }
                RecCmd::FillRect { state, rect, color, opacity, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    let b = crate::resource::SolidColorBrush { domain: self.domain, color: *color, opacity: *opacity };
                    self.fill_rectangle(*rect, &b);
                }
                RecCmd::FillPolys { state, figures, winding, color, opacity, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    let b = crate::resource::SolidColorBrush { domain: self.domain, color: *color, opacity: *opacity };
                    self.fill_polygons(figures, *winding, &b);
                }
                RecCmd::FillGeom { state, geom, color, opacity, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    let b = crate::resource::SolidColorBrush { domain: self.domain, color: *color, opacity: *opacity };
                    self.fill_geometry(geom, &b);
                }
                RecCmd::StrokeGeom { state, geom, color, opacity, width, style, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    let b = crate::resource::SolidColorBrush { domain: self.domain, color: *color, opacity: *opacity };
                    self.draw_geometry(geom, &b, *width, style);
                }
                RecCmd::Ink { state, start, segments, color, opacity, style, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    let b = crate::resource::SolidColorBrush { domain: self.domain, color: *color, opacity: *opacity };
                    let ink = crate::extras::ink::Ink { start: *start, segments: segments.to_vec() };
                    self.draw_ink(&ink, &b, style.as_ref());
                }
                RecCmd::InkBrush { state, start, segments, brush, style, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    let b = self.rerealize(brush);
                    let ink = crate::extras::ink::Ink { start: *start, segments: segments.to_vec() };
                    self.draw_ink_brush(&ink, &b, style.as_ref());
                }
                RecCmd::GradientMesh { state, patches, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    self.draw_gradient_mesh(&crate::extras::mesh::GradientMesh { patches: patches.to_vec() });
                }
                RecCmd::DrawBitmap { state, bitmap, dest, source, opacity, interpolation, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    self.draw_bitmap(bitmap, *dest, *source, *opacity, *interpolation);
                }
                RecCmd::OpacityMask { state, bitmap, color, opacity, content, dest, source, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    let b = crate::resource::SolidColorBrush { domain: self.domain, color: *color, opacity: *opacity };
                    self.fill_opacity_mask(bitmap, &b, *content, *dest, *source);
                }
                RecCmd::OpacityMaskBrush { state, bitmap, brush, content, dest, source, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    let b = self.rerealize(brush);
                    self.fill_opacity_mask_brush(bitmap, &b, *content, *dest, *source);
                }
                RecCmd::Mesh { state, tri, color, opacity, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    let b = crate::resource::SolidColorBrush { domain: self.domain, color: *color, opacity: *opacity };
                    self.fill_mesh_solid(tri, &b);
                }
                RecCmd::MeshBrush { state, tri, brush, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    let b = self.rerealize(brush);
                    self.fill_mesh_brush(tri, &b);
                }
                RecCmd::FillRectBrush { state, rect, brush, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    let b = self.rerealize(brush);
                    self.fill_rectangle_brush(*rect, &b);
                }
                RecCmd::FillGeomBrush { state, geom, brush, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    let b = self.rerealize(brush);
                    self.fill_geometry_brush(geom, &b);
                }
                RecCmd::StrokeGeomBrush { state, geom, brush, width, style, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    let b = self.rerealize(brush);
                    self.draw_geometry_brush(geom, &b, *width, style);
                }
                RecCmd::Line { state, p0, p1, width, style, color, opacity, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    let b = crate::resource::SolidColorBrush { domain: self.domain, color: *color, opacity: *opacity };
                    self.draw_line(*p0, *p1, &b, *width, style.as_ref());
                }
                RecCmd::LineBrush { state, p0, p1, width, style, brush, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    let b = self.rerealize(brush);
                    self.draw_line_brush(*p0, *p1, &b, *width, style.as_ref());
                }
                RecCmd::GlyphRun {
                    state, font, face_index, em_size, glyphs, baseline_origin, measuring_mode, is_sideways, bidi_level, no_snap, simulations, text_aa, params, fg, palette, ..
                } => {
                    self.set_replay_state(state, &t, dpi);
                    // Text honours SetPrimitiveBlend like every other primitive: carry the recorded
                    // primitive blend into the render params so the composite can COPY/MIN/ADD/MAX.
                    let params = &crate::text::RenderingParams { blend: blend_of(state.blend), raster_clip: self.raster_clip(), clip_px: self.centre_clip_px(), ..*params };
                    let run = crate::text::GlyphRun {
                        font_data: font,
                        face_index: *face_index,
                        em_size: *em_size,
                        glyphs,
                        baseline_origin: *baseline_origin,
                        measuring_mode: *measuring_mode,
                        is_sideways: *is_sideways,
                        bidi_level: *bidi_level,
                        no_snap: *no_snap,
                        simulations: *simulations,
                        coords: &[],
                    };
                    // Played back by DrawImage, the run meets DrawGlyphRun's bitmap em limit under the replay
                    // transform and latches E_INVALIDARG (r24 29278: a recorded run drawn under ×1e6). A list
                    // inlined by SOURCE_OVER replays its recorded tags with its draws, so the error carries the
                    // run's own (r23 28696); a list composited as one image carries the playing op's. A list
                    // realized for an image brush does not latch it (r24 29528: the same run under a ×1e6
                    // image brush leaves the frame's error to a later op).
                    if !self.realizing_list {
                        let own = self.state.tags;
                        if self.playing_as_image {
                            self.state.tags = saved_state.tags;
                        }
                        let rejected = self.glyph_run_em_rejected(&run, params);
                        self.state.tags = own;
                        if rejected {
                            continue;
                        }
                    }
                    // Colour-bitmap and SVG runs replay through their own draws under the composed state.
                    match fg {
                        crate::command::TextFg::ColorBitmap(formats) => {
                            self.draw_color_bitmap_glyph_run(&run, *formats, params);
                            continue;
                        }
                        crate::command::TextFg::Svg(color) => {
                            self.draw_svg_glyph_run(&run, *color, params);
                            continue;
                        }
                        _ => {}
                    }
                    let dt = self.device_transform();
                    if let Some(buf) = &mut self.target {
                        match (palette, fg) {
                            (Some(p), crate::command::TextFg::Solid(c)) => {
                                crate::text::color::render_color_glyph_run(buf, &dt, &run, crate::text::TextFg::Solid(*c), *p, *text_aa, params, None);
                            }
                            (_, crate::command::TextFg::Solid(c)) => {
                                crate::text::render_glyph_run(buf, &dt, &run, *c, *text_aa, params);
                            }
                            (_, crate::command::TextFg::Brush(b)) => {
                                crate::text::render_glyph_run_brush(buf, &dt, &run, b, *text_aa, params);
                            }
                            (_, crate::command::TextFg::ColorBitmap(_) | crate::command::TextFg::Svg(_)) => {}
                        }
                    }
                }
                RecCmd::DrawImage { state, bitmap, offset, source, interpolation, composite, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    // Recorded from inside draw_image_placed: the state already carries the bitmap's texel→DIP scale and
                    // the offset and rect are in texels, so replay places it as is (r27 32929: a 48-dpi bitmap in a list
                    // replayed at twice its size).
                    self.draw_image_placed(bitmap, *offset, *source, *interpolation, *composite, true);
                }
                RecCmd::DrawImageList { state, list, offset, rect: Some(r), composite, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    self.draw_image_list(list, *offset, Some(*r), 1, *composite);
                }
                RecCmd::DrawImageList { state, list, offset, composite, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    // Compose the nested placement with the outer replay transform.
                    let inner_t = crate::geometry::mat_mul(&[1.0, 0.0, 0.0, 1.0, offset[0], offset[1]], &self.state.transform);
                    let inner = list.0.borrow().cmds.clone();
                    self.composite_cmds(&inner, inner_t, dpi, *composite, None);
                }
                RecCmd::DrawEffect { state, bitmap, dest0, offset, interp, composite, finite, pointwise, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    self.blit_effect_bitmap(bitmap, *dest0, *offset, *interp, *composite, *finite, *pointwise);
                }
                RecCmd::Flood { state, src, interp, composite, crop, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    self.draw_flood(*src, crop.map_or([0.0, 0.0], |c| c.0), crop.map(|c| c.1), *interp, *composite);
                }
                RecCmd::SpriteBatch { state, sprites, bitmap, interpolation, options, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    let batch = crate::extras::SpriteBatch { sprites: sprites.to_vec() };
                    self.draw_sprite_batch(&batch, bitmap, *interpolation, *options);
                }
                RecCmd::Effect { state, effect, offset, interpolation, composite, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    self.draw_image_effect(effect, *offset, *interpolation, *composite);
                }
                RecCmd::BlitFloat { state, px, w, h, offset, b_lt, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    // Device offset of the DIP DrawImage offset under the composed transform (world∘dpi),
                    // added to the record-time realize bounds origin.
                    let we = self.state.transform;
                    let (dpx, dpy) = self.dpi;
                    let (sx, sy) = if self.get_unit_mode() == 0 { (dpx / 96.0, dpy / 96.0) } else { (1.0, 1.0) };
                    let ox = sx * (we[0] * offset[0] + we[2] * offset[1] + we[4]);
                    let oy = sy * (we[1] * offset[0] + we[3] * offset[1] + we[5]);
                    let (dx, dy) = (b_lt[0] + ox.round() as i32, b_lt[1] + oy.round() as i32);
                    if let Some(buf) = &mut self.target {
                        crate::float::FloatTarget::new(buf).blit(px, *w, *h, dx, dy);
                    }
                }
                RecCmd::PushLayer { state, bounds, mask, mask_aa, mask_transform, opacity, opacity_brush, layer_options, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    self.push_layer(*bounds, mask.as_ref(), *mask_aa, *mask_transform, *opacity, opacity_brush.as_ref(), *layer_options);
                }
                RecCmd::PopLayer => self.pop_layer(),
                RecCmd::PushClip { state, rect, antialias, .. } => {
                    self.set_replay_state(state, &t, dpi);
                    // Under a rotation or skew at playback Direct2D turns the clip into an
                    // INITIALIZE_FROM_BACKGROUND layer bounded by the rect (IGNORE_ALPHA too on an
                    // ignore-alpha target), so the clip is the rotated rect itself (clclip p2).
                    let d = self.device_transform().map(|v| v as f32);
                    let rotated = !crate::layer::axis_preserving(&d);
                    if rotated {
                        let ignore = self.target.as_ref().is_some_and(|t| t.desc.alpha_mode == D2D1_ALPHA_MODE_IGNORE);
                        self.push_layer_as(*rect, None, *antialias, None, 1.0, None, if ignore { 3 } else { 1 }, true);
                    } else {
                        self.push_axis_aligned_clip(*rect, *antialias);
                    }
                    clip_as_layer.push(rotated);
                }
                RecCmd::PopClip => {
                    if clip_as_layer.pop() == Some(true) { self.pop_layer() } else { self.pop_axis_aligned_clip() }
                }
            }
        }
        self.replay_depth -= 1;
        self.play_tags = outer_tags;
        self.play_unplaced = outer_unplaced;
        self.play_unbounded = outer_unbounded;
        self.recording = saved_rec;
        self.state = saved_state;
        self.dpi = saved_dpi;
    }

    /// Re-realize a recorded brush under the current (replay) device transform; a brush with no
    /// retained definition (image brush / COLR ramp) keeps its baked realization.
    fn rerealize(&self, brush: &PixelBrush) -> PixelBrush {
        match &brush.def {
            Some(def) => {
                let m = self.device_transform();
                let dev = [m[0] as f32, m[1] as f32, m[2] as f32, m[3] as f32, m[4] as f32, m[5] as f32];
                PixelBrush { domain: brush.domain, brush: def.realize(&dev), def: brush.def.clone(), image: None, crop: brush.crop.clone(), fail: brush.fail, batch: brush.batch, rows: brush.rows }
            }
            None => brush.clone(),
        }
    }

    /// A brush placed for a draw under the current world transform, DPI and unit mode, or `None` when its realization
    /// already is. Native places a brush per draw (`GetRealizationPlacementData` runs at fill time), so a transform,
    /// DPI or unit mode set between creating the brush and drawing with it applies (fuzz r5 5011/5862: a gradient
    /// under a rotate and shear set after it; VM probe-bdraw: a bitmap brush under a rotation, SetDpi or unit mode).
    /// A gradient re-realizes its ramp; a bitmap or image brush keeps its texels and takes the new brush→device
    /// matrix (an effect image brush also realizes its effect per draw, [`image_brush_for_draw`](Self::image_brush_for_draw)).
    pub(crate) fn realize_brush_for_draw(&self, brush: &PixelBrush) -> Option<PixelBrush> {
        match (&brush.def, &brush.brush) {
            (Some(crate::brush::BrushDef::Linear { .. } | crate::brush::BrushDef::Radial { .. }), _) => Some(self.rerealize(brush)),
            (Some(crate::brush::BrushDef::Bitmap { xform, dpi, .. }), Brush::Bitmap { transform, intermediate_precision, bitmap, .. }) => {
                let m = self.bitmap_placement(Some(*xform), *dpi);
                // Its intermediates take the buffer precision current at the draw (r27 32017: a brush made before
                // SetRenderingControls(16 bpc) and drawn after it).
                let p = match self.buffer_precision { 0 => if crate::float::is_highp(bitmap.format) { 5 } else { 1 }, p => p };
                // ponytail: a re-placed draw clones the brush's texels and mips; share them (Rc) if draw-heavy callers
                // that change the transform per draw show it.
                (m.map(f32::to_bits) != transform.map(f32::to_bits) || p != *intermediate_precision).then(|| {
                    let mut b = brush.clone();
                    b.brush.set_placement(m);
                    if let Some(crate::brush::BrushDef::Bitmap { size, interpolation, .. }) = &brush.def {
                        if let Some(row) = self.bitmap_vs_row(Some(*xform), *dpi, *size, *interpolation) { b.brush.set_vs_row(row); }
                    }
                    b.brush.set_intermediate_precision(p);
                    b
                })
            }
            _ => None,
        }
    }

    /// [`Self::realize_brush_for_draw`] for a geometry/rect fill or stroke, which also takes the NaN-LOD
    /// coarsest level ([`crate::brush::Brush::nan_lod`]).
    pub(crate) fn realize_brush_for_fill(&self, brush: &PixelBrush) -> Option<PixelBrush> {
        let r = self.realize_brush_for_draw(brush);
        let base = r.as_ref().unwrap_or(brush);
        match base.brush.nan_lod() {
            Some(b) => Some(PixelBrush { brush: b, ..base.clone() }),
            None => r,
        }
    }

    /// HIGH_QUALITY_CUBIC sizes its intermediate from the device rect's minimum corner pushed out by one
    /// source unit's device span; when that is not finite the draw fails with E_INVALIDARG and paints
    /// nothing (r23 28560: a dest top of -3e38 over a 7-DIP source span fails, -2.9e38 draws, a 1-DIP span
    /// fails at -2.6e38, dpi scales it, no maximum edge fails). A NaN corner drops out of the min and max,
    /// so an axis with no finite corner fails too (r24 29175: an -inf offset under a singular transform).
    pub(crate) fn hq_cubic_overflow(&self, rect: [f32; 4], src: [f32; 2]) -> bool {
        let m = self.device_transform();
        let c = [apply(&m, rect[0], rect[1]), apply(&m, rect[2], rect[1]), apply(&m, rect[2], rect[3]), apply(&m, rect[0], rect[3])];
        (0..2).any(|k| {
            let (lo, hi) = (c.iter().map(|p| p[k]).fold(f32::INFINITY, f32::min), c.iter().map(|p| p[k]).fold(f32::NEG_INFINITY, f32::max));
            !(lo - (hi - lo) / src[k]).is_finite()
        })
    }

    /// `DrawBitmap` (1E): composite `bitmap` from `source` (texels) into `dest` (DIPs) under the
    /// current device transform, premultiplied source-over, scaling `opacity` into the sample. A pure
    /// scale/translate keeps the fast axis-aligned blit; a rotation/shear routes to the rotated
    /// textured-quad raster (`bitmap::draw_bitmap_affine`). No-op while recording a command list.
    pub fn draw_bitmap(&mut self, bitmap: &Bitmap, dest: [f32; 4], source: [f32; 4], opacity: f32, interpolation: u32) {
        let _a = self.api(136, 0, Some(crate::batch::Prim::Bitmap));
        if !self.ready_to_draw() {
            return;
        }
        // An A8 (alpha-only) bitmap is not a valid colour source — native latches
        // WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT at the draw (the op still records S_OK; the sticky error
        // surfaces at the next HRESULT call and at EndDraw) and paints nothing (r23 A8 draw_bitmap). It does so while
        // a command list records too, which makes the list's Close fail (r24 29113, 29143, 29296).
        if bitmap.format.0 == 65 {
            self.process_new_error(UNSUPPORTED_PIXEL_FORMAT);
            return;
        }
        // Recording, a command list keeps the draw and makes it where it is drawn (r23 28559: a DrawBitmap in a list
        // painted the live target while the list recorded).
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            self.record(RecCmd::DrawBitmap { state, dpi, bitmap: Rc::new(bitmap.clone()), dest, source, opacity, interpolation });
            return;
        }
        // An R8G8B8A8 colour source stores RGBA; present it to the BGRA8 sampler with R/B swapped.
        let sampled = bitmap.as_sampled_bgra8();
        let bitmap = sampled.as_ref();
        // DrawBitmap's `sourceRectangle` is in the bitmap's own DIPs, not texels, so a non-96-dpi bitmap
        // scales it to texels by `dpi/96` before the sort/clamp and sampling below (a 144-dpi 16-texel
        // bitmap is 10.67 DIP wide; r23 28063 draws an inverted source rect of one). A 96-dpi bitmap —
        // the common case — scales by exactly 1.0 and is unchanged.
        // Under D2D1_UNIT_MODE_PIXELS the source rectangle is already in texels: the bitmap's dpi does not scale it
        // (r27 32861: a 192-dpi bitmap drawn in pixel units samples texel x 4.5, not 9).
        let (kx, ky) = if self.state.unit == 0 { (bitmap.dpi.0 / 96.0, bitmap.dpi.1 / 96.0) } else { (1.0, 1.0) };
        let source = [source[0] * kx, source[1] * ky, source[2] * kx, source[3] * ky];
        // Native draws nothing for a source rect with a NaN edge in ANY slot (VM ladder fx7k/bs2: each of
        // the four edges alone, both orders, one or both axes — r12 17124/17420/17862, r13 18361) — not
        // only where native's bitmap-draw per-axis `l <= r` sort and
        // `[0, dim]` clamp would leave a NaN span. That sort+clamp does decide the finite case: an edge
        // clamped onto the other (span 0, e.g. [16, 0, 14, 10] on a 14-wide bitmap) gives an infinite
        // mapping and draws nothing too, as does an axis whose two edges are the same infinity, which the
        // clamp leaves an infinite span and the mapping a zero scale (VM: [-inf, -inf] and [inf, inf] on
        // either axis draw nothing, NEAREST and LINEAR; [-inf, inf] and a finite span wholly past an edge draw).
        {
            let (bw, bh) = (bitmap.width as f32, bitmap.height as f32);
            let srt = |a: f32, b: f32| if a <= b { (a, b) } else { (b, a) };
            let (xmin, xmax) = srt(source[0], source[2]);
            let (ymin, ymax) = srt(source[1], source[3]);
            let xmin = if xmin <= 0.0 { 0.0 } else { xmin };
            let ymin = if ymin <= 0.0 { 0.0 } else { ymin };
            let xmax = if xmax < bw { xmax } else { bw };
            let ymax = if ymax < bh { ymax } else { bh };
            let void = |d: f32| !(d > 0.0 || d < 0.0) || d.is_infinite(); // NaN, zero or infinite
            // HIGH_QUALITY_CUBIC fails where its intermediate can't be sized ([`Self::hq_cubic_overflow`]),
            // which a source edge clamped onto the other (span 0) or a NaN source edge also does, before the
            // silent voids below. A NaN dest edge fails only off opacity 1, which takes the general path
            // (r24 29902 at opacity 0.89; VM: each dest edge alone at opacity 0, 0.89, 0.999 and 1.5 fails,
            // at exactly 1 none does, and no other mode fails).
            let dest_nan = dest.iter().any(|v| v.is_nan());
            // Recording into a command list checks nothing: the draw fails where it is played back
            // (r23 28619: recorded under a NaN translate, never drawn; VM: every failing case recorded
            // then drawn fails at the list's nested index). On replay the NaN-source check still fires
            // (below) under any finite transform, singular included (r24 29424); only the HQ sizing test
            // is gated on the transform placing (r24 29494; 29809/29982 an image brush with a singular matrix).
            let m = self.device_transform();
            let m_finite = m.iter().all(|v| v.is_finite());
            // On replay, opacity 1 with a non-finite dest takes the fast blit that silently voids (VM:
            // a replayed draw_bitmap at opacity 1 with a NaN/inf dest draws nothing; opacity ≠ 1 or a
            // finite dest still fails).
            let void_1 = self.replay_depth > 0 && opacity == 1.0 && dest.iter().any(|v| !v.is_finite());
            if interpolation == 5 && self.recording.is_none() && !void_1 {
                // A NaN source edge fails regardless of the transform's placement: a replay under a
                // singular or -inf-offset transform still fails it (r24 29424: a command-list draw_bitmap
                // with a NaN source, replayed under a -inf outer offset, fails E_INVALIDARG at the nested
                // index; VM cl/op probes: NaN source fails at every finite transform, identity or singular).
                if source.iter().any(|v| v.is_nan()) {
                    self.process_new_error(E_INVALIDARG);
                    return;
                }
                // The HQ-cubic intermediate sizing (a NaN dest off opacity 1, else the overflow test) is
                // placement-gated: a replay under a non-finite transform can't size it, so it is skipped, unless the
                // list's bounds cover the whole plane and it plays there anyway. That failure only shows once the
                // frame's batch has run: EndDraw reports it under the frame's last tags, whatever follows the draw
                // (r27 32411: a list with a Clear and an HQ-cubic bitmap, drawn at an offset the DPI scale overflows
                // to -inf; VM: trailing state ops, a layer, a fill, SOURCE_OVER and an explicit SetTags all report the
                // last op's tag).
                let unplaced = self.replay_depth > 0 && !m_finite;
                if (!unplaced || self.play_unbounded)
                    && (if dest_nan { opacity != 1.0 } else { self.hq_cubic_overflow(dest, [xmax - xmin, ymax - ymin]) })
                {
                    if unplaced {
                        self.end_error.get_or_insert(E_INVALIDARG);
                    } else {
                        self.process_new_error(E_INVALIDARG);
                    }
                    return;
                }
            }
            // Nor can it pre-scale the bitmap under a zero linear transform, where the dest collapses to the translation:
            // the draw fails when that point lies strictly inside a pixel the clip leaves, on both axes, and off the
            // 1/16-px grid's pixel corners (VM probe-hz: (10.5, 10.5) and (31.5, 5.5) fail; an integer coordinate, a
            // point within 1/32 px of a corner, or one outside the target or the clip succeeds). r27 32656 fails; r23
            // 28659, whose point lies outside the target, does not; a rank-one transform draws (r23 28212 28831).
            let reach = self.reach_rect();
            let inside = |v: f64, lo: f32, hi: f32| v.fract() != 0.0 && v.floor() >= lo as f64 && v.floor() < hi as f64;
            let corner = |v: f64| (v - v.round()).abs() < 1.0 / 32.0;
            if interpolation == 5 && self.recording.is_none() && self.replay_depth == 0 && m[..4] == [0.0; 4]
                && inside(m[4], reach[0], reach[2]) && inside(m[5], reach[1], reach[3]) && !(corner(m[4]) && corner(m[5]))
            {
                self.process_new_error(INTERMEDIATE_TOO_LARGE);
                return;
            }
            if source.iter().any(|v| v.is_nan()) || void(xmax - xmin) || void(ymax - ymin) {
                return;
            }
            // Native stretches the WHOLE bitmap onto the dest: the source→dest scale is
            // `sorted_dest_dim / clamped_source_dim` (native's bitmap-draw affine), and
            // the realization spans the full bitmap, `bitmap_dim × scale`. When that footprint overflows
            // f32 to a non-finite value the realization is degenerate and native draws nothing — even when
            // the scale itself is finite, and for every interpolation mode (VM-probed both sides of the
            // FLT_MAX boundary, across bitmap widths and both axes; the anisotropic thin-source scene voids
            // here too). Native's dest sort: X by `left <= right`, Y by `top > bottom`, so a NaN dest edge
            // propagates a non-finite span here exactly as it does there.
            let (xl, xr) = if dest[0] <= dest[2] { (dest[0], dest[2]) } else { (dest[2], dest[0]) };
            let (yt, yb) = if dest[1] > dest[3] { (dest[3], dest[1]) } else { (dest[1], dest[3]) };
            if !(bw * ((xr - xl) / (xmax - xmin))).is_finite() || !(bh * ((yb - yt) / (ymax - ymin))).is_finite() {
                return;
            }
        }
        let m = self.device_transform();
        // The cubic/anisotropic/HQ paths realize a DEVICE-resolution intermediate for the dest before
        // sampling; NEAREST/LINEAR instead fill the dest as geometry and need no such intermediate. When
        // the device-space dest area (width × height of the transformed-corner bounds, in f32) overflows
        // f32, the intermediate cannot be sized and native draws nothing — VM-probed both sides of the
        // FLT_MAX boundary (dest area, not the footprint: NEAREST/LINEAR draw the same dest, and the area
        // is measured in device space, so dpi scales the threshold). cubic/aniso/HQ only.
        if interpolation >= 2 {
            let c = [apply(&m, dest[0], dest[1]), apply(&m, dest[2], dest[1]), apply(&m, dest[2], dest[3]), apply(&m, dest[0], dest[3])];
            let span = |k: usize| c.iter().map(|p| p[k]).fold(f32::NEG_INFINITY, f32::max) - c.iter().map(|p| p[k]).fold(f32::INFINITY, f32::min);
            if !(span(0) * span(1)).is_finite() {
                return;
            }
        }
        let Some(buf) = &mut self.target else { return };
        // Rotation/shear (off-diagonal terms) can't be an axis-aligned rect; rasterize the quad.
        if m[1] != 0.0 || m[2] != 0.0 {
            // Native's per-axis sort and clamp: the low edge clamps up to 0 and the high edge down to the size, each
            // on its own, so a source rect wholly past the bitmap ([3.23, 12.39] on a 3-wide bitmap: [3.23, 3]) maps
            // through its inverted sliver rather than collapsing to nothing (r27 32522).
            let source = {
                let (bw, bh) = (bitmap.width as f32, bitmap.height as f32);
                let (l, r) = if source[0] <= source[2] { (source[0], source[2]) } else { (source[2], source[0]) };
                let (t, b) = if source[1] <= source[3] { (source[1], source[3]) } else { (source[3], source[1]) };
                [if l <= 0.0 { 0.0 } else { l }, if t <= 0.0 { 0.0 } else { t }, if r < bw { r } else { bw }, if b < bh { b } else { bh }]
            };
            // d2d1 draws a rotated DrawBitmap as FillRectangles of the dest under a bitmap brush: the rectangle's corners
            // go through the transform sink in f32, as a rotated FillRectangle's do (VM: an edge of slope -0.4 at dpi
            // 144 runs through a row of sample points, which the f64 product put on the other side).
            let corners = [
                transform_mesh_point(&m, dest[0], dest[1]),
                transform_mesh_point(&m, dest[2], dest[1]),
                transform_mesh_point(&m, dest[2], dest[3]),
                transform_mesh_point(&m, dest[0], dest[3]),
            ];
            // Preserve the source mapping before rounded device corners lose matrix precision.
            let sx = (dest[2] - dest[0]) / (source[2] - source[0]);
            let sy = (dest[3] - dest[1]) / (source[3] - source[1]);
            let dx = dest[0] - sx * source[0];
            let dy = dest[1] - sy * source[1];
            let f = m.map(|v| v as f32);
            let mapping = [f[0] * sx, f[1] * sx, f[2] * sy, f[3] * sy,
                (f[0] * dx + f[2] * dy) + f[4], (f[1] * dx + f[3] * dy) + f[5]];
            let aa = if self.state.aa == 1 { crate::pixels::Aa::Aliased } else { crate::pixels::Aa::PerPrimitive };
            // The brush the sampled quad draws with: d2d1's source→dest matrix (built in double from the sorted dest and the
            // clamped source, stored f32) through the bitmap brush's device→texcoord row (VM: the LINEAR kernel's
            // texcoords at its address, every lane of a rotated dpi-192 draw).
            let (xl, xr) = if dest[0] <= dest[2] { (dest[0], dest[2]) } else { (dest[2], dest[0]) };
            let (yt, yb) = if dest[1] > dest[3] { (dest[3], dest[1]) } else { (dest[1], dest[3]) };
            let k = [(xr as f64 - xl as f64) / (source[2] as f64 - source[0] as f64), (yb as f64 - yt as f64) / (source[3] as f64 - source[1] as f64)];
            let b = [k[0] as f32, 0.0, 0.0, k[1] as f32, (xl as f64 - source[0] as f64 * k[0]) as f32, (yt as f64 - source[1] as f64 * k[1]) as f32];
            let row = hw_bitmap_row([bitmap.width as f32, bitmap.height as f32], b, f);
            bitmap::draw_bitmap_affine_precision(buf, bitmap, corners, source, opacity, interpolation, self.buffer_precision, Some(mapping), aa, blend_of(self.state.blend), Some(row));
            return;
        }
        let a = apply(&m, dest[0], dest[1]);
        let c = apply(&m, dest[2], dest[3]);
        let dev = [a[0].min(c[0]), a[1].min(c[1]), a[0].max(c[0]), a[1].max(c[1])];
        // Under an explicit SetAntialiasMode(ALIASED) every sampled DrawBitmap draws its dest-rect edges by
        // WARP's aliased centre test, not the 8×8 fringe (VM probe palias; r27 32901 cubic, 32474 MSL).
        let aa = if self.state.aa == 1 { crate::pixels::Aa::Aliased } else { crate::pixels::Aa::PerPrimitive };
        if self.state.blend == 1 && interpolation == 1
            && (crate::float::is_highp(buf.desc.format) || crate::float::is_highp(bitmap.format)) {
            let source = bitmap::norm_source(source, bitmap.width as f32, bitmap.height as f32);
            crate::float::draw_bitmap_copy_linear(buf, bitmap, dev, source, opacity);
        } else if crate::float::is_highp(buf.desc.format) {
            crate::float::draw_bitmap_precision(buf, bitmap, dev, source, opacity, interpolation, self.buffer_precision, blend_of(self.state.blend), aa);
        } else if self.buffer_precision != 0 && interpolation >= 2 {
            crate::float::draw_bitmap_precision(buf, bitmap, dev, source, opacity, interpolation, self.buffer_precision, blend_of(self.state.blend), aa);
        } else {
            // Off the alpha-blit path a LINEAR draw fills its dest with a bitmap brush whose matrix is d2d1's
            // source→dest affine (built in double, stored f32): WARP samples the brush's device→texcoord row on its
            // unit-triangle plane (VM: the brush fill kernel's texcoords at its address, every lane exact).
            let plane = (interpolation == 1).then(|| {
                let (bw, bh) = (bitmap.width as f32, bitmap.height as f32);
                let (smin, smax) = ([source[0].min(source[2]).max(0.0), source[1].min(source[3]).max(0.0)], [source[0].max(source[2]).min(bw), source[1].max(source[3]).min(bh)]);
                let (xl, xr) = if dest[0] <= dest[2] { (dest[0], dest[2]) } else { (dest[2], dest[0]) };
                let (yt, yb) = if dest[1] > dest[3] { (dest[3], dest[1]) } else { (dest[1], dest[3]) };
                let s = [(xr as f64 - xl as f64) / (smax[0] as f64 - smin[0] as f64), (yb as f64 - yt as f64) / (smax[1] as f64 - smin[1] as f64)];
                let b = [s[0] as f32, 0.0, 0.0, s[1] as f32, (xl as f64 - smin[0] as f64 * s[0]) as f32, (yt as f64 - smin[1] as f64 * s[1]) as f32];
                crate::warp::plane::warp_fit_affine(&hw_bitmap_row([bw, bh], b, m.map(|v| v as f32)))
            });
            bitmap::draw_bitmap_with(buf, bitmap, dev, source, opacity, interpolation, blend_of(self.state.blend), aa, plane);
        }
    }

    /// `DrawImage` of a bitmap (1E) at `offset` (DIPs) under any
    /// `D2D1_COMPOSITE_MODE`. `source` is the optional `imageRectangle` (image-space texels): the
    /// sub-rect of the bitmap to draw, its top-left placed at `offset`; `None` draws the whole bitmap.
    ///
    /// Native's `DrawImage` splits by mode: SOURCE_OVER(0) and PLUS(9) take the
    /// fast path — a textured rectangle with the mapped primitive blend (OVER / ADD), leaving the rest
    /// of the target untouched. BOUNDED_SOURCE_COPY(11) copies within the *imageRectangle* only; with
    /// no imageRectangle it is unbounded ≡ SOURCE_COPY(10), which clears the whole target. Every other
    /// mode (the Porter-Duff/MASK_INVERT) takes the slow path: realize the image into a transparent
    /// intermediate, then composite it over the whole target. We keep the proven whole-bitmap blit for
    /// 0/9/10 (and no-rect 11 ≡ copy), and realize-then-`composite_over_target` for the rest — bounded
    /// to the imageRectangle for mode 11 with a `source`, whole-target otherwise.
    pub fn draw_image_bitmap(&mut self, bitmap: &Bitmap, offset: [f32; 2], source: Option<[f32; 4]>, interpolation: u32, composite: u32) {
        let _a = self.api(64, 0, None);
        if !self.draw_image_args_ok(interpolation, composite) {
            return;
        }
        // A bitmap's image space is its own DIPs: a texel spans 96/dpi of them (r23 28075: a 192-dpi
        // bitmap covers half its texel size). Draw it in texel space, under the world transform scaled by
        // that, with the offset and image rectangle carried back through the scale. Under UNIT_MODE_PIXELS a texel is a
        // pixel whatever the bitmap's DPI (sc156: a 144-dpi or 48-dpi bitmap draws at its texel size; r27 32395).
        let (sx, sy) = if self.state.unit == 1 { (1.0, 1.0) } else { (96.0 / bitmap.dpi.0, 96.0 / bitmap.dpi.1) };
        self.dip_place = (self.state.unit == 0).then_some((self.state.transform, offset, [sx, sy]));
        if (sx, sy) == (1.0, 1.0) || !(sx.is_finite() && sy.is_finite()) {
            self.draw_image_placed(bitmap, offset, source, interpolation, composite, true);
            self.dip_place = None;
            return;
        }
        let world = self.state.transform;
        let [a, b, c, d, e, f] = world;
        self.state.transform = [a * sx, b * sx, c * sy, d * sy, e, f];
        let source = source.map(|[l, t, r, b]| [l / sx, t / sy, r / sx, b / sy]);
        self.draw_image_placed(bitmap, [offset[0] / sx, offset[1] / sy], source, interpolation, composite, true);
        self.state.transform = world;
        self.dip_place = None;
    }

    /// The whole device pixel `bitmap`'s origin rounds to when a DrawImage of it at DIP `offset` takes the whole-pixel fast
    /// path (unit device scale, origin within 1/32 px of a pixel corner, not MULTI_SAMPLE_LINEAR), as `draw_image_bitmap`
    /// places it.
    pub(crate) fn image_fast_origin(&self, bitmap: &Bitmap, offset: [f32; 2], interpolation: u32) -> Option<[f32; 2]> {
        let (sx, sy) = if self.state.unit == 1 { (1.0, 1.0) } else { (96.0 / bitmap.dpi.0, 96.0 / bitmap.dpi.1) };
        let w = self.state.transform;
        let (dx, dy) = if self.state.unit == 0 { (self.dpi.0 / 96.0, self.dpi.1 / 96.0) } else { (1.0, 1.0) };
        let m = [w[0] * sx * dx, w[1] * sx * dy, w[2] * sy * dx, w[3] * sy * dy, w[4] * dx, w[5] * dy].map(|v| v as f64);
        let p = apply(&m, offset[0] / sx, offset[1] / sy);
        (interpolation != 3 && m[..4] == [1.0, 0.0, 0.0, 1.0] && fast_whole(p[0]) && fast_whole(p[1])).then(|| p.map(f32::round))
    }

    /// DrawImage's argument check, made before it draws anything (inside a draw; outside one the draw latches
    /// WRONG_STATE first): an interpolation past HIGH_QUALITY_CUBIC or a composite past MASK_INVERT latches
    /// E_INVALIDARG (r11 16045, 16814: composite 13 on an effect).
    pub(crate) fn draw_image_args_ok(&mut self, interpolation: u32, composite: u32) -> bool {
        if self.in_draw && (interpolation > 5 || composite > 12) {
            self.process_new_error(E_INVALIDARG);
            return false;
        }
        true
    }

    /// [`draw_image_bitmap`](Self::draw_image_bitmap), or (`own_placement` false) an effect's realized
    /// output, whose NaN placement `draw_image_effect` already tested on the effect's own offset.
    pub(crate) fn draw_image_placed(&mut self, bitmap: &Bitmap, offset: [f32; 2], source: Option<[f32; 4]>, interpolation: u32, composite: u32, own_placement: bool) {
        if !self.ready_to_draw() {
            return;
        }
        // An A8 image source is rejected like a DrawBitmap A8 source (see `draw_bitmap`): latch
        // WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT and draw nothing (r23 A8 draw_image).
        // An A8 (alpha-only) colour source is rejected at the draw whether or not a command list is open:
        // a recorded DrawImage of an A8 bitmap makes the list's Close fail with UNSUPPORTEDPIXELFORMAT, as
        // native does (r23 28152/28201; draw_bitmap and the sprite batch already latch unconditionally).
        if bitmap.format.0 == 65 {
            self.process_new_error(UNSUPPORTED_PIXEL_FORMAT);
            return;
        }
        // DrawImage reads an IGNORE-alpha bitmap as premultiplied, stored alpha and all, under every composite: native
        // draws it exactly as the same bitmap marked PREMULTIPLIED (sc132, sc133, sc138: SOURCE_OVER, DESTINATION_OVER and
        // BlendImage at every interpolation), and the copy modes write the stored alpha bytes as they are (r27 32237: a
        // bounded copy leaves alpha 36 where an opaque read gave 249).
        let sampled = if bitmap.alpha == D2D1_ALPHA_MODE_IGNORE && bitmap.premul_bgra.len() == bitmap.width as usize * bitmap.height as usize * 4 {
            std::borrow::Cow::Owned(Bitmap { alpha: D2D1_ALPHA_MODE_PREMULTIPLIED, ..bitmap.clone() }.as_sampled_bgra8().into_owned())
        } else {
            bitmap.as_sampled_bgra8()
        };
        let bitmap = sampled.as_ref();
        // While recording, keep the DIP offset + image; replay re-issues under the replay transform. [G12]
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            self.record(RecCmd::DrawImage { state, dpi, bitmap: Rc::new(bitmap.clone()), offset, source, interpolation, composite });
            return;
        }
        let m = self.device_transform();
        let p = apply(&m, offset[0], offset[1]);
        // HIGH_QUALITY_CUBIC at extreme axis-aligned minification would need ~2/scale cubic taps per
        // output pixel below the HQ downscale's per-pass scale limit (`HQ_MIN_SCALE`) — an unbounded loop,
        // i.e. a hang (fuzz r21 26308: a 1e-6 downscale). Native's DrawImage DDI drops to the anisotropic
        // sampler for that footprint, which mip-prefilters to a bounded probe. Verified byte-exact against
        // native's own render on the 1e-6 downscale.
        let hq = interpolation == 5;
        let interpolation = if interpolation == 5
            && m[1] == 0.0 && m[2] == 0.0 && m[0] > 0.0 && m[3] > 0.0 && m[0] <= 1.0 && m[3] <= 1.0
            && (m[0] < crate::sampling::HQ_MIN_SCALE as f64 || m[3] < crate::sampling::HQ_MIN_SCALE as f64)
        {
            4
        } else {
            interpolation
        };
        // The fast image-draw path: at a unit device scale a device offset within 1/32 of a whole pixel draws
        // the bitmap RoundSat'd to that pixel whatever the mode, MULTI_SAMPLE_LINEAR excepted (bmfast-*).
        // A rotated or skewed filtered DrawImage with an image rectangle is the placed rectangle filled with the bitmap as
        // a clamped bitmap brush of that interpolation, through WARP's textured-triangle sampler (sc113: native's cropped
        // bitmap and cropped identity ColorMatrix render alike; sc114: that fill matches it byte for byte; sc160: CUBIC
        // and MULTI_SAMPLE_LINEAR exact, ANISOTROPIC and HQ within 2), not the axis-aligned blit, which ignores the
        // rotation. A rectangle past the bitmap is filled whole, the bitmap sampled transparent past its edges (sc159,
        // sc160: LINEAR, CUBIC and the copy exact, MULTI_SAMPLE_LINEAR within 1; r27 32929). Axis-aligned, a SOURCE_OVER
        // draw with an image rectangle is that fill too, in every mode: its fractional edges cover their pixels in part
        // (sc194/sc195: NEAREST to CUBIC exact at 48 and 96 dpi, HQ within 2; r27 32654's bitmap draws). The other modes
        // fill a transparent intermediate the same way and composite it, on a float target too, so a pixel the rect only
        // touches still blends its covered share (sc200: every mode exact; MASK_INVERT's sc193 edges within 1).
        // BOUNDED_SOURCE_COPY keeps its copy-nothing rules ahead of the fill, and off the fast path fills through the
        // opaque edge walk ([`Self::bcopy_crop_walk`]); its ANISOTROPIC draw stays with the blit, which clears the whole
        // rect where the anisotropic fill covers only the bitmap.
        // Off the fast path, the composites Direct2D draws straight into the target (SOURCE_OVER, PLUS,
        // BOUNDED_SOURCE_COPY) size HIGH_QUALITY_CUBIC's intermediate as DrawBitmap does
        // ([`Self::hq_cubic_overflow`]): an infinite or NaN placement fails with E_INVALIDARG (r24 29175; VM:
        // ±inf and NaN offsets and non-finite world translations fail in exactly those three modes, none in
        // the other ten, a -3.4e38 offset draws nothing, and no other interpolation mode fails), an image
        // rectangle's fill included (r27 32449: a BOUNDED_SOURCE_COPY rect reaching -3.4e38). The fast path
        // places per axis, so a NaN x with a fractional y is off it (no 0·NaN cross term).
        let fast_axis = m[..4] == [1.0, 0.0, 0.0, 1.0] && interpolation != 3
            && fast_whole((offset[0] as f64 + m[4]) as f32) && fast_whole((offset[1] as f64 + m[5]) as f32);
        // A recorded DrawImage played back draws without the check (r23 28554, r24 29477 29881).
        // A BlendImage blends through its own intermediate, so it is not one of those direct draws (r27 32342 32494 32713:
        // a HIGH_QUALITY_CUBIC BlendImage at a 3.4e38 offset is S_OK).
        let hq_overflow = own_placement && !fast_axis && hq && matches!(composite, 0 | 9 | 11) && self.replay_depth == 0 && !self.blending && {
            let s = source.unwrap_or([0.0, 0.0, bitmap.width as f32, bitmap.height as f32]);
            let (sw, sh) = (s[2] - s[0], s[3] - s[1]);
            self.hq_cubic_overflow([offset[0], offset[1], offset[0] + sw, offset[1] + sh], [sw.abs(), sh.abs()])
        };
        let axis = m[1] == 0.0 && m[2] == 0.0;
        if let (Some(r), true) = (source, axis && !(composite == 11 && interpolation == 4) || !axis && interpolation != 0) {
            if hq_overflow {
                self.process_new_error(E_INVALIDARG);
                return;
            }
            let n = [r[0].min(r[2]), r[1].min(r[3]), r[0].max(r[2]), r[1].max(r[3])];
            if composite == 11 && axis && self.bcopy_rect_copies_nothing(bitmap, p, n, interpolation) {
                return;
            }
            // The world already carries the bitmap's DPI (`draw_image_bitmap`), so the brush maps texels 1:1.
            let texels = (bitmap.dpi != (96.0, 96.0) && self.state.unit == 0).then(|| Bitmap { dpi: (96.0, 96.0), ..bitmap.clone() });
            // ANISOTROPIC realizes the crop with a transparent border (extend 3), which also builds the
            // mip chain UNPADDED (image texels only) so the sampler's border — not a clamp ring — handles
            // out-of-bounds, as WARP's realize does; every other mode keeps the clamp extend.
            let crop_extend = if interpolation == 4 { [3, 3] } else { [0, 0] };
            let mut brush = self.create_bitmap_brush(texels.as_ref().unwrap_or(bitmap), crop_extend, interpolation, 1.0, Some([1.0, 0.0, 0.0, 1.0, offset[0] - n[0], offset[1] - n[1]]));
            // The fill is DrawImage's own: a HIGH_QUALITY_CUBIC image at a non-finite placement fails only in the modes
            // drawn straight into the target, as the uncropped image does (r24 29752: SOURCE_COPY at a NaN offset draws).
            if !matches!(composite, 0 | 9 | 11) {
                brush.fail = None;
            }
            let crop = if interpolation == 4 {
                // ANISOTROPIC realizes the crop as WARP's two-stage draw-00 does: the bitmap sampled with a
                // transparent border (set above) into the composite intermediate, which is then placed 1:1 and
                // composited (VM DDI iso32663: the realize quad into a 512 atlas is byte-exact, and the final
                // image is that atlas placed 1:1 — op8cap). A crop edge the image_rect carries PAST the bitmap
                // (o[k]!=n[k], clamped to the bitmap edge) fades continuously through that border, so the fill
                // rect extends to the whole pixel the fade reaches and the border does it. A crop edge INSIDE
                // the bitmap (o[k]==n[k]) is the fill rect's own AA edge, faded by the rasteriser's unit-rect
                // eighths coverage. (Replaces fcd99642's per-side clamp, which the clamp-padded mip chain made
                // a no-op — the aniso crop always clamped; here the unpadded chain lets the border fade.)
                let (bw, bh) = (bitmap.width as f32, bitmap.height as f32);
                let o = [n[0].max(0.0), n[1].max(0.0), n[2].min(bw), n[3].min(bh)];
                let o = if o[0] < o[2] && o[1] < o[3] { o } else { [n[0]; 4] };
                core::array::from_fn(|k| {
                    let e = offset[k & 1] + (o[k] - n[k & 1]);
                    if o[k] == n[k] { e } else if k < 2 { e.floor() } else { e.ceil() }
                })
            } else {
                if let Brush::Bitmap { extend, .. } = &mut brush.brush { *extend = [crate::sampling::Extend::Transparent; 2]; }
                brush.def = None;
                let mut crop = [offset[0], offset[1], offset[0] + (n[2] - n[0]), offset[1] + (n[3] - n[1])];
                // MULTI_SAMPLE_LINEAR: a crop edge the image_rect carries past the bitmap stops at the pixel centres
                // within half a texel of the bitmap's edge, where the filter's reach into the transparent border ends,
                // not at the crop's own edge (VM c136: 22 probes at 0.5x to 4x with each side past, SOURCE_OVER and PLUS
                // exact; r27 32136). A mode that composites through an intermediate renders the image into one bounded
                // by its edges rounded out on the 1/256 grid, so the fill stops at the nearer of the two (VM c136-c138:
                // 35 SOURCE_IN and SOURCE_ATOP probes at 0.5x to 2x, edge fractions on both sides of each cut, exact).
                let lim = [0.0, 0.0, bitmap.width as f32, bitmap.height as f32];
                let bounded = !matches!(composite, 0 | 9 | 10 | 11);
                if interpolation == 3 && axis && m[0] > 0.0 && m[3] > 0.0 && n[0].max(0.0) < n[2].min(lim[2]) && n[1].max(0.0) < n[3].min(lim[3]) {
                    for k in 0..4 {
                        if (k < 2) == (n[k] < lim[k]) && n[k] != lim[k] {
                            let (sc, tr) = if k & 1 == 0 { (m[0], m[4]) } else { (m[3], m[5]) };
                            let edge = (offset[k & 1] + (lim[k] - n[k & 1])) as f64;
                            let reach = (sc * (edge + if k < 2 { -0.5 } else { 0.5 }) + tr).round();
                            let out = ((sc * edge + tr) * 256.0).round() / 256.0;
                            let d = if !bounded { reach } else if k < 2 { out.floor().max(reach) } else { out.ceil().min(reach) };
                            crop[k] = ((d - tr) / sc) as f32;
                        }
                    }
                }
                crop
            };
            let float = self.target.as_ref().is_some_and(|t| crate::float::is_highp(t.desc.format));
            if composite == 11 && axis && !float && !(m[..4] == [1.0, 0.0, 0.0, 1.0] && fast_whole(p[0]) && fast_whole(p[1])) {
                self.bcopy_crop_walk(crop, &brush);
                return;
            }
            // The shader route is taken where the LINEAR footprint (the rect grown by the half texel the filter reads)
            // passes the bitmap, so the transparent border is sampled; a footprint inside it samples the bitmap as the
            // rect case does (VM: a right edge at 16.758 of 18 texels rounds, at 17.99 or 18.5 truncates). A
            // BlendImage's blend effect samples the bitmap as its own input instead (suite blend-image-lineardodge-linear-rect).
            // ponytail: an edge exactly half a texel from the bitmap's is taken as inside; unprobed.
            let (bw, bh) = (bitmap.width as f32, bitmap.height as f32);
            let past = !(n[0] - 0.5 >= 0.0 && n[1] - 0.5 >= 0.0 && n[2] + 0.5 <= bw && n[3] + 0.5 <= bh);
            self.crop_fill = interpolation == 1 && !self.blending && past;
            self.fill_crop_with_brush(crop, composite, &brush);
            self.crop_fill = false;
            return;
        }
        if own_placement && self.nan_fast_placement(offset, interpolation, true) {
            if matches!(composite, 10 | 11) {
                self.composite_empty(10);
            }
            return;
        }
        let fast = m[..4] == [1.0, 0.0, 0.0, 1.0] && interpolation != 3 && fast_whole(p[0]) && fast_whole(p[1]);
        if hq_overflow {
            self.process_new_error(E_INVALIDARG);
            return;
        }
        let p = if fast { p.map(f32::round) } else { p };
        // A non-finite device transform (or placed origin) can't place the image: native realizes it
        // empty and composites that — BOUNDED_SOURCE_COPY keeps the backdrop (fuzz 9222, m[1]=-inf),
        // the destination-zeroing modes clear it (recap9 11709's SOURCE_OUT under ty=inf).
        if !m.iter().all(|v| v.is_finite()) || !p.iter().all(|v| v.is_finite()) {
            self.composite_empty(composite);
            return;
        }
        // BOUNDED_SOURCE_COPY without an imageRectangle: the fast image-draw path (unit device scale, device
        // position within 1/32 of a whole pixel) copies over the whole clip even for an image wholly
        // outside it. Off that path native copies only when the image reaches the target, and then draws
        // the bounded-copy geometry below over the whole target (bm11-* probes, DDI dump of d192-int-in).
        let mut slow11 = false;
        if composite == 11 && source.is_none() {
            if interpolation == 3 || !(m[..4] == [1.0, 0.0, 0.0, 1.0] && fast_whole(p[0]) && fast_whole(p[1])) {
                let (w, h) = (bitmap.width as f32, bitmap.height as f32);
                let c = [[0.0, 0.0], [w, 0.0], [0.0, h], [w, h]].map(|[x, y]| apply(&m, offset[0] + x, offset[1] + y));
                let [cl, ct, cr, cb] = self.reach_rect();
                let lo = |k: usize| c.iter().map(|q| q[k]).fold(f32::INFINITY, f32::min);
                let hi = |k: usize| c.iter().map(|q| q[k]).fold(f32::NEG_INFINITY, f32::max);
                // Native realizes the scaled source as an intermediate of round(deviceFootprint) texels
                // per axis; when BOTH axes round to 0 the intermediate is 0×0 (empty), and SOURCE_COPY of
                // an empty source replaces the whole bounded region (the target, no image rect) with
                // transparent. That fires only when the empty intermediate, at the image's rounded device
                // position, lies inside that region (sc131: at 1e-6 it clears at (10,10) but keeps the backdrop
                // at x 31.8 of 32, or past any edge); a singular transform clears nothing (sc130, r27 32417).
                // The footprint is the source size under the transform's LINEAR part only: a huge
                // offset (3.4e38) would lose the size to f32 precision in the placed corners. VM-probed:
                // empty iff round(w·scale)==0 && round(h·scale)==0 (fuzz r20 25584, an 11×3 bitmap at 1e-6;
                // a 0.8px-wide footprint still draws, 0.4px wipes — round at 0.5, both axes).
                if m[0] * m[3] - m[1] * m[2] == 0.0 {
                    return;
                }
                let lc = [[0.0, 0.0], [w as f64, 0.0], [0.0, h as f64], [w as f64, h as f64]]
                    .map(|[x, y]| [m[0] * x + m[2] * y, m[1] * x + m[3] * y]);
                let ext = |k: usize| lc.iter().map(|q| q[k]).fold(f64::NEG_INFINITY, f64::max)
                    - lc.iter().map(|q| q[k]).fold(f64::INFINITY, f64::min);
                if (ext(0) as f32).round() == 0.0 && (ext(1) as f32).round() == 0.0 {
                    let q = [p[0].round(), p[1].round()];
                    if !(cl <= q[0] && q[0] < cr && ct <= q[1] && q[1] < cb) {
                        return;
                    }
                    let r = [cl as i32, ct as i32, cr as i32, cb as i32];
                    if let Some(t) = self.target.as_ref() {
                        let mut inter = t.clone();
                        inter.data.fill(0);
                        self.composite_over_target(&inter, 10, Some(r));
                    }
                    return;
                }
                // Axis-aligned, the copy fires when a pixel centre falls inside the image rect grown by the
                // filter's footprint, texels·scale + pixels of (0,0) NEAREST, (0.5,0) LINEAR, (1.5,0) CUBIC/HQ,
                // (0.5,0.5) MULTI_SAMPLE_LINEAR, (0.5,1) ANISOTROPIC (bm11 reach probes at scale 1.25 and 2.5).
                let reach = if m[1] == 0.0 && m[2] == 0.0 {
                    let (rt, rp) = [(0.0, 0.0), (0.5, 0.0), (1.5, 0.0), (0.5, 0.5), (0.5, 1.0), (1.5, 0.0)][interpolation.min(5) as usize];
                    let axis = |k: usize, c0: f32, c1: f32| {
                        let r = rt * (if k == 0 { m[0] } else { m[3] }).abs() as f32 + rp;
                        ((lo(k) - r - 0.5).floor() + 1.0).max(c0) <= ((hi(k) + r - 0.5).ceil() - 1.0).min(c1 - 1.0)
                    };
                    axis(0, cl, cr) && axis(1, ct, cb)
                } else {
                    lo(0) < cr && lo(1) < cb && hi(0) > cl && hi(1) > ct
                };
                if !reach {
                    return;
                }
                slow11 = true;
            }
        }
        // An unscaled draw at a fractional offset resamples like a scaled one in every filtered mode, not
        // only LINEAR (a unit-triangle WARP draw, traced for r5 5077's 13×8 and an 8×8; CUBIC, ANISOTROPIC
        // and HQ cubic VM probes, r19 24025 24820, r20 25766).
        let frac_linear = interpolation != 0 && (p[0].fract() != 0.0 || p[1].fract() != 0.0);
        // Unscaled, a LINEAR power-of-two-wide texture whose height is not a power of two is the WARP draw below
        // (r5 5028's 1×11, traced), not the fixed iterator. CUBIC and HQ resample as for any other shape (sc173: an 8×18
        // and a 16×18 bitmap at a fractional offset; r27 32199).
        let unit_draw = frac_linear && interpolation == 1 && matches!(composite, 0 | 9) && m[..4] == [1.0, 0.0, 0.0, 1.0]
            && bitmap.width.is_power_of_two() && !bitmap.height.is_power_of_two();
        // PLUS and SOURCE_COPY under an axis-aligned non-unit scale draw this same textured rect, with an
        // ADD blend / replacing the target (dimg-sc-* probes; 9-corpus seed 306 at dpi 120). PLUS NEAREST is
        // the WARP point draw below instead; SOURCE_COPY NEAREST is this rect's AlphaBlt DDA too (traced
        // dimg-sc-copy-n-w2: a 16.16 blit kernel, no triangle setup, its top tie row excluded). Unscaled at
        // a fractional offset PLUS LINEAR is SOURCE_OVER's draw with ADD too, the unit triangle included
        // (dplus-frac-*, r7 8089), and SOURCE_COPY's CUBIC, ANISOTROPIC and HQ cubic resample through this rect as
        // well (sc213: Δ13-34 as a copy, exact this way; r27 32850). Its LINEAR is the slow WARP draw below.
        let axis_scaled = matches!(composite, 9 | 10) && (interpolation != 0 || composite == 10) && m[1] == 0.0 && m[2] == 0.0 && m[..4] != [1.0, 0.0, 0.0, 1.0]
            || composite == 9 && frac_linear && m[..4] == [1.0, 0.0, 0.0, 1.0] && !unit_draw
            || composite == 10 && frac_linear && m[..4] == [1.0, 0.0, 0.0, 1.0] && matches!(interpolation, 2 | 4 | 5);
        // MULTI_SAMPLE_LINEAR is never the fast path's copy: unscaled under any composite it is the same four-tap
        // textured rect as scaled, its taps reading transparent past the image (bmfast-ms-c0-near, msu-*, msl-whole-*:
        // native softens even at a whole offset). SOURCE_COPY draws it here, BOUNDED_SOURCE_COPY as its slow copy
        // above, the other modes through the intermediate below.
        let msl_unscaled = interpolation == 3 && matches!(composite, 0 | 9 | 10);
        // CUBIC, MULTI_SAMPLE_LINEAR, ANISOTROPIC and HQ cubic under a slow-path composite resample through
        // the same sampler as the source-over draw, into a transparent intermediate the mode then composites
        // (VM probes: SOURCE_IN/OUT and MASK_INVERT edges maxd 19-36 -> 1; r23 28044, 28075's ring).
        let filtered_slow = interpolation >= 2 && !matches!(composite, 0 | 9 | 10 | 11) && m[1] == 0.0 && m[2] == 0.0
            && (m[..4] != [1.0, 0.0, 0.0, 1.0] || frac_linear || interpolation == 3);
        // A NEAREST SOURCE_OVER draw off the whole-pixel fast path fills the target with the bitmap as an image brush whose
        // matrix is the device transform with the DIP offset applied, not the DPI-folded placement we place by (VM: a 72-dpi
        // bitmap at dpi 144 lands at 6.25, where the folded chain gives 6.2499995 and moves ties and edge columns).
        let highp_target = self.target.as_ref().is_some_and(|t| crate::float::is_highp(t.desc.format));
        if source.is_none() && composite == 0 && interpolation == 0 && !fast && m[1] == 0.0 && m[2] == 0.0 && !highp_target
            && !crate::float::is_highp(bitmap.format)
        {
            if let Some(blit) = self.dip_place.and_then(|place| self.direct_nearest_blit(bitmap, place)) {
                let Some(buf) = self.target.as_mut() else { return };
                let (w, h) = (buf.desc.width as i64, buf.desc.height as i64);
                for y in (blit.lo[1] as i64).max(0)..(blit.hi[1] as i64).min(h) {
                    for x in (blit.lo[0] as i64).max(0)..(blit.hi[0] as i64).min(w) {
                        let Some((tx, ty)) = blit.texel(x, y) else { continue };
                        let [r, g, b, a] = bitmap.texel_rgba(tx, ty);
                        let i = y as usize * buf.stride + x as usize * 4;
                        bitmap::over_premul_f(buf, i, [b, g, r, a], 1.0);
                    }
                }
                return;
            }
        }
        if source.is_none() && (composite == 0 && (m[..4] != [1.0, 0.0, 0.0, 1.0] || frac_linear) && !unit_draw || axis_scaled || msl_unscaled || filtered_slow) {
            use crate::effects::builtin::transform as tfx;
            let texmap = self.image_texmap(bitmap, offset, p, [0.0; 2], composite);
            let tex_dims = self.texture_dims(bitmap);
            let slow_matrix = self.image_brush_matrix(offset, composite);
            let vis = self.reach_rect();
            let copy_prim = self.state.blend == 1 && composite == 10;
            let Some(buf) = self.target.as_mut() else { return };
            let (w, h) = (buf.desc.width, buf.desc.height);
            let input = if self.effect_atlas { tfx::Input::new(bitmap, 0, 0).in_atlas() } else { tfx::Input::new(bitmap, 0, 0) };
            // Native DrawImage ANISO retains the source mip format, then stores into the requested
            // output precision. Its UNORM8 sampler truncates before an 8-bit intermediate store.
            let input = if interpolation == 4 && self.buffer_precision != 0 { input.with_output_precision(self.buffer_precision) }
                else if crate::float::is_highp(buf.desc.format) { input.with_float_output() } else { input };
            let matrix = [m[0] as f32, m[1] as f32, m[2] as f32, m[3] as f32, p[0], p[1]];
            // The slow path (SOURCE_COPY here) places the image by its own device matrix (msltie-*-c10:
            // its 1.2499999 scale includes the footprint's tie pixel).
            let matrix = if composite == 10 { slow_matrix.unwrap_or(matrix) } else { matrix };
            let highp = crate::float::is_highp(bitmap.format) || crate::float::is_highp(buf.desc.format);
            let precision = if self.buffer_precision == 0 { if highp { 5 } else { 1 } } else { self.buffer_precision };
            let axis = matrix[1] == 0.0 && matrix[2] == 0.0;
            // Native HQ magnification and anisotropic minification use an atlas at the rounded
            // output bounds; their vertex attributes retain the original image coordinates.
            let placed_filter = (interpolation == 4 && (matrix[0] < 1.0 || matrix[3] < 1.0))
                || (interpolation == 5 && matrix[0] >= 1.0 && matrix[3] >= 1.0);
            // The HQ cubic magnification's slot in the frame's atlas: its output bounds rounded to nearest.
            let mut placed_rect: Option<[i32; 4]> = None;
            let realized = if placed_filter && axis {
                let mode = if interpolation == 5 { tfx::Interp::Cubic } else { tfx::Interp::from_mode(interpolation) };
                let m44 = [matrix[0] as f64, matrix[1] as f64, 0.0, 0.0, matrix[2] as f64, matrix[3] as f64, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, p[0] as f64, p[1] as f64, 0.0, 1.0];
                // The atlas starts at the Scale primitive's top-left corner, which ANISOTROPIC rounds
                // out (floor) and HQ cubic to nearest (aniso trace: native's quad spans y 0..62 of its
                // atlas, so the v plane's intercept is native's at the v·H = 5.0 ties).
                let extent = tfx::projective_extent(&input, m44, mode, tfx::Border::Soft);
                if interpolation == 5 {
                    placed_rect = extent.map(|(lo, hi)| [lo[0], lo[1], hi[0], hi[1]].map(|v| (v + 0.5).floor() as i32));
                }
                let lo = extent.map_or([0.0; 2], |(lo, _)| lo.map(|v| if interpolation == 4 { v.floor() } else { (v + 0.5).floor() }));
                let sample = tfx::affine_at(&input, matrix, mode, tfx::Border::Soft, [-lo[0], -lo[1]], true);
                tfx::to_bitmap_precision(w, h, 0.0, 0.0, highp, |x, y| {
                    let value = sample(x, y);
                    if self.buffer_precision == 0 { value } else { crate::effects::quantize_to(precision, value) }
                })
            } else if axis && matrix[0] > 0.0 && matrix[3] > 0.0 && matches!(interpolation, 0 | 1 | 2 | 3 | 5)
                && !(interpolation == 5 && (matrix[0] < 1.0 || matrix[3] < 1.0)) {
                tfx::to_bitmap_precision(w, h, 0.0, 0.0, highp, bitmap_image_axis(bitmap, matrix, interpolation, crate::float::is_highp(buf.desc.format), composite == 9, composite == 0, copy_prim, texmap, tex_dims))
            } else if interpolation == 5 && axis && matrix[0] > 0.0 && matrix[3] > 0.0
                && (matrix[0] < 1.0 || matrix[3] < 1.0) {
                if matrix[0] > 1.0 || matrix[3] > 1.0 {
                    bitmap_image_hq_mixed(bitmap, matrix, w, h, precision)
                } else if matrix[4].fract() != 0.0 || matrix[5].fract() != 0.0 {
                    // At a fractional device offset native shrinks the bitmap at the origin, as the two Scale passes
                    // of its HQ downscale (the y axis, then x) over the shrunk extent padded to [-3, ⌈n·s⌉ + 2), and
                    // then places it with a third, HQ cubic Scale pass: the whole translation, at unit scale. At a
                    // whole offset that last pass is a plain copy (DDI: three draws against two; sc211's ×0.75
                    // probes Δ67 → 1, r27 32400).
                    let ext = [(bitmap.width as f32 * matrix[0]).ceil() as i32, (bitmap.height as f32 * matrix[3]).ceil() as i32];
                    let bounds = [-3, -3, ext[0] + 2, ext[1] + 2];
                    let down = tfx::hq_downscale(&input, [matrix[0], 0.0, 0.0, matrix[3], 0.0, 0.0], 0.0, tfx::Border::Soft, bounds, precision);
                    let mid = tfx::Input::new(&down, bounds[0], bounds[1]).in_atlas();
                    let mid = if precision > 1 { mid.with_float_output() } else { mid };
                    let sample = tfx::affine_at(&mid, [1.0, 0.0, 0.0, 1.0, matrix[4], matrix[5]], tfx::Interp::HqCubic(0.0), tfx::Border::Soft, [0.0; 2], true);
                    tfx::to_bitmap_precision(w, h, 0.0, 0.0, precision > 1, sample)
                } else {
                    tfx::hq_downscale(&input, matrix, 0.0, tfx::Border::Soft, [0, 0, w as i32, h as i32], precision)
                }
            } else if interpolation == 4 {
                let local = [matrix[0], matrix[1], matrix[2], matrix[3], 0.0, 0.0];
                let sample = tfx::affine_at_vp(&input, local, tfx::Interp::Anisotropic, tfx::Border::Soft, p, axis, Some([0.0, 0.0, w as f32, h as f32]));
                tfx::to_bitmap_precision(w, h, 0.0, 0.0, highp, |x, y| {
                    let value = sample(x - p[0], y - p[1]);
                    if self.buffer_precision == 0 { value } else { crate::effects::quantize_to(precision, value) }
                })
            } else if interpolation == 1 && !axis && !highp && composite == 0 {
                // A rotated or skewed LINEAR SOURCE_OVER draw is WARP's textured draw with the bitmap brush's texcoord
                // rows (diag(size)·B·W inverted by one reciprocal determinant, fitted over the unit triangle, as the
                // geometry sampler's), over the image grown by half a texel with a transparent border; the address
                // floors a power-of-two dimension and rounds another, and the sample is stored truncated. The
                // transform sampler's quad planes land a Q8 step off in places (sc113: 24 px; sc110: the same for an
                // identity ColorMatrix, which draws as its bitmap).
                let (bw, bh) = (bitmap.width as f32, bitmap.height as f32);
                let row = crate::warp::plane::warp_fit_affine(&crate::brush::invert(&[matrix[0] * bw, matrix[1] * bw, matrix[2] * bh, matrix[3] * bh, matrix[4], matrix[5]]));
                let tex = crate::sampling::Texture { data: &bitmap.premul_bgra, width: bitmap.width, height: bitmap.height };
                tfx::to_bitmap_precision(w, h, 0.0, 0.0, false, |x, y| {
                    let u = [row[0] * x + (row[2] * y + row[4]), row[1] * x + (row[3] * y + row[5])];
                    let t = [u[0] * bw, u[1] * bh];
                    if !(t[0] >= -0.5 && t[0] < bw + 0.5 && t[1] >= -0.5 && t[1] < bh + 0.5) { return [0.0; 4]; }
                    let q = |t: f32, n: f32, pow2: bool| {
                        let a = (t.clamp(-10.0, 10.0) * n - 0.5) * 256.0;
                        let q = if pow2 { a.floor() } else { a.round_ties_even() } as i32;
                        (q >> 8, q & 255)
                    };
                    let pow2 = [bitmap.width.is_power_of_two(), bitmap.width.is_power_of_two() && bitmap.height.is_power_of_two()];
                    let (qx, qy) = (q(u[0], bw, pow2[0]), q(u[1], bh, pow2[1]));
                    let v: [f32; 4] = core::array::from_fn(|ch| (tex.bilinear_v128([crate::sampling::Extend::Transparent; 2], ch, qx, qy) >> 7) as f32 / 255.0);
                    [v[2], v[1], v[0], v[3]]
                })
            } else {
                tfx::to_bitmap_precision(w, h, 0.0, 0.0, highp,
                    tfx::affine_at(&input, matrix, tfx::Interp::from_mode(interpolation), tfx::Border::Soft, [0.0; 2], axis))
            };
            if filtered_slow && !crate::float::is_highp(buf.desc.format) {
                let mut inter = buf.clone();
                inter.data.fill(0);
                // The intermediate holds only the image's device rect rounded out: the filter's spill past
                // it stays transparent (r19 24197's XOR beside the target edge; DEST_OVER/ATOP probes).
                let (xs, ys) = ([matrix[4], matrix[4] + matrix[0] * bitmap.width as f32], [matrix[5], matrix[5] + matrix[3] * bitmap.height as f32]);
                let (x0, x1, y0, y1) = (xs[0].min(xs[1]).floor(), xs[0].max(xs[1]).ceil(), ys[0].min(ys[1]).floor(), ys[0].max(ys[1]).ceil());
                // A CUBIC or MULTI_SAMPLE_LINEAR draw resamples in its slot of the frame's atlas under the replay's brush,
                // as the LINEAR one does: the plane is fitted at the atlas origin, the slot's offset carried through the
                // DPI round trip (r27 32025: MSL's taps on the device plane land a Q8 step off native's atlas).
                let slot_rect = Self::atlas_visible([x0, y0, x1, y1].map(|v| v as i32), vis);
                let replay = (matches!(interpolation, 2 | 3) && axis && matrix[0] > 0.0 && matrix[3] > 0.0 && !highp && !self.effect_atlas)
                    .then(|| self.atlas.next(slot_rect[2] - slot_rect[0], slot_rect[3] - slot_rect[1])).flatten()
                    .and_then(|slot| self.dip_place.and_then(|place| self.replay_brush(place, [(slot[0] - slot_rect[0]) as f32, (slot[1] - slot_rect[1]) as f32]))
                        .map(|r| (slot, r)));
                let replay = replay.map(|(slot, (rs, ro, rk))| {
                    let row = hw_bitmap_row(tex_dims.map(|v| v as f32), [rs[0], 0.0, 0.0, rs[1], ro[0], ro[1]], [rk[0], 0.0, 0.0, rk[1], 0.0, 0.0]);
                    // The textured rect of the image's soft border, as for the target-space resample: MSL's taps reach
                    // half a texel, CUBIC's one and a half.
                    let (n, sc, of) = ([bitmap.width as f32, bitmap.height as f32], [matrix[0], matrix[3]], [matrix[4], matrix[5]]);
                    let pad = if interpolation == 3 { 0.5 } else { 1.5 };
                    let rect: [f32; 4] = [(of[0] - pad * sc[0] + 0.5).floor(), (of[1] - pad * sc[1] + 0.5).floor(),
                        (of[0] + (n[0] + pad) * sc[0] - 0.5).ceil(), (of[1] + (n[1] + pad) * sc[1] - 0.5).ceil()];
                    (crate::warp::plane::warp_fit_affine(&row), [slot_rect[0] - slot[0], slot_rect[1] - slot[1]], rect, row)
                });
                let texd = tex_dims.map(|v| v as f32);
                let tex = crate::sampling::Texture { data: &bitmap.premul_bgra, width: bitmap.width, height: bitmap.height };
                let quantize = |v: [f32; 4]| v.map(|c| (c.clamp(0.0, 1.0) * 255.0).round_ties_even() as u8);
                let msl = replay.filter(|_| interpolation == 3).map(|(.., row)| {
                    let mut brush = Brush::bitmap_source(bitmap.clone(), &matrix, [0, 0], 3, 1.0);
                    if let Brush::Bitmap { extend, .. } = &mut brush { *extend = [crate::sampling::Extend::Transparent; 2]; }
                    brush.set_vs_row(row);
                    brush
                });
                let px = |x: i32, y: i32| -> [u8; 4] {
                    match replay {
                        Some((fit, org, rect, _)) => {
                            let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
                            if cx < rect[0] || cy < rect[1] || cx >= rect[2] || cy >= rect[3] { return [0; 4]; }
                            let (ax, ay) = ((x - org[0]) as f32 + 0.5, (y - org[1]) as f32 + 0.5);
                            if let Some(brush) = &msl {
                                return quantize(brush.source_at_rect(ax, ay, true, [f32::MIN, f32::MIN, f32::MAX, f32::MAX]).map(|c| c / 255.0));
                            }
                            let u = [fit[0] * ax + (fit[2] * ay + fit[4]), fit[1] * ax + (fit[3] * ay + fit[5])];
                            let v = tex.sample_hqcubic_in([crate::sampling::Extend::Transparent; 2], u, [u[0] * texd[0], u[1] * texd[1]], tex_dims);
                            quantize(v.map(|c| (c * 255.0).round_ties_even() / 255.0))
                        }
                        None => { let [r, g, b, a] = realized.texel_rgba(x, y); quantize([b, g, r, a]) }
                    }
                };
                for y in 0..h { for x in 0..w {
                    if !((x0..x1).contains(&(x as f32)) && (y0..y1).contains(&(y as f32))) { continue; }
                    let i = y as usize * inter.stride + x as usize * 4;
                    inter.data[i..i + 4].copy_from_slice(&px(x as i32, y as i32));
                }}
                let spill = |x: i32, y: i32| px(x, y);
                // The filter reaches 1.5 texels past the image (2 for the HQ cubic's prefilter), plus the pixel it lands in.
                let reach = (2.0 * matrix[0].abs().max(matrix[3].abs())).ceil() as i32 + 1;
                self.atlas_composite(&inter, [x0, y0, x1, y1].map(|v| v as i32), composite, Some((&spill, reach.clamp(1, 64))));
                return;
            }
            // A SOURCE_OVER HQ cubic magnification is resampled into its slot of the frame's atlas and composited from
            // there (DDI: a 24x16 slot at atlas texel 0, the next slow-path image's slot after it), so a later image's
            // filter reach landing on the slot shows in it.
            if let (Some(rect), 0, false) = (placed_rect, composite, crate::float::is_highp(buf.desc.format)) {
                let mut inter = buf.clone();
                inter.data.fill(0);
                for y in rect[1].max(0)..rect[3].min(h as i32) { for x in rect[0].max(0)..rect[2].min(w as i32) {
                    let [r, g, b, a] = realized.texel_rgba(x, y);
                    let i = y as usize * inter.stride + x as usize * 4;
                    inter.data[i..i + 4].copy_from_slice(&[b, g, r, a].map(|v| (v.clamp(0.0, 1.0) * 255.0).round_ties_even() as u8));
                }}
                self.atlas_composite(&inter, rect, 0, None);
                return;
            }
            if crate::float::is_highp(buf.desc.format) {
                let pixels: Vec<[f32; 4]> = (0..h).flat_map(|y| (0..w).map(move |x| (x, y)))
                    .map(|(x, y)| realized.texel_rgba(x as i32, y as i32)).collect();
                crate::float::FloatTarget::new(buf).blit(&pixels, w as i32, h as i32, 0, 0);
            } else {
                // SOURCE_COPY issues its textured draw only when the image's exact device rect meets the
                // target; past an edge it copies transparent over the whole target (DDI: no draw, only the
                // clear; SOURCE_COPY reach probes: LINEAR/CUBIC 15.95 draws, 16.05 does not).
                let (xs, ys) = ([matrix[4], matrix[4] + matrix[0] * bitmap.width as f32], [matrix[5], matrix[5] + matrix[3] * bitmap.height as f32]);
                let miss = composite == 10 && axis && !(xs[0].min(xs[1]) < w as f32 && xs[0].max(xs[1]) > 0.0 && ys[0].min(ys[1]) < h as f32 && ys[0].max(ys[1]) > 0.0);
                // An image the target or a clip cuts copies only its device rect rounded out: the filter's
                // spill past it stays transparent. One wholly inside keeps its spill (SOURCE_COPY cubic
                // probes: the image across the right, left or bottom edge or a clip, vs wholly inside).
                let (lo, hi) = ([xs[0].min(xs[1]), ys[0].min(ys[1])], [xs[0].max(xs[1]), ys[0].max(ys[1])]);
                let cut = composite == 10 && axis && !(lo[0] >= vis[0] && lo[1] >= vis[1] && hi[0] <= vis[2] && hi[1] <= vis[3]);
                let snap = |v: f32| (v * 256.0).round_ties_even() / 256.0; // the rect's edges on the 1/256 grid
                let (cx0, cy0, cx1, cy1) = (snap(lo[0]).floor(), snap(lo[1]).floor(), snap(hi[0]).ceil(), snap(hi[1]).ceil());
                // A PLUS HQ cubic magnification renders at its device position in a target-sized region of the frame's atlas,
                // at its origin (DDI: the image at atlas texel (29, 48), the next draw's slot at x 96 on a 96-wide target).
                if composite == 9 && placed_rect.is_some() {
                    self.atlas.alloc(w as i32, h as i32);
                }
                for y in 0..h { for x in 0..w {
                    let outside = cut && !((cx0..cx1).contains(&(x as f32)) && (cy0..cy1).contains(&(y as f32)));
                    let [r, g, b, a] = if miss || outside { [0.0; 4] } else { realized.texel_rgba(x as i32, y as i32) };
                    let i = y as usize * buf.stride + x as usize * 4;
                    match composite {
                        9 => bitmap::add_premul_f(buf, i, [b, g, r, a]),
                        10 => buf.data[i..i + 4].copy_from_slice(&[b, g, r, a].map(|v| (v.clamp(0.0, 1.0) * 255.0).round_ties_even() as u8)),
                        _ => bitmap::over_premul_f(buf, i, [b, g, r, a], 1.0),
                    }
                }}
            }
            return;
        }
        // Rotated/sheared transform (non-axis, non-singular) with a non-zero composite: native's
        // DrawImage slow path realizes the image through the same 2D-affine transform graph as the
        // source-over draw above (soft border), into a transparent target-sized intermediate, then
        // composites it with the requested mode (drawimage-bm-rot-05/-11, drawimage-bm-shear-12).
        // COPY/BOUNDED copy the whole target: the image is transparent outside its footprint.
        // A LINEAR draw under a non-fast composite is the same WARP draw into an intermediate whatever
        // the transform (composite probes cmpl-*: the axis-aligned resample was ±1 at a fractional offset).
        let slow_linear = interpolation == 1 && (!matches!(composite, 0 | 9 | 11) || unit_draw);
        // NEAREST under a slow-path composite (not SOURCE_OVER, PLUS, COPY or BOUNDED) is the same WARP
        // point draw, its unit triangle at the screen origin and its address floored (r9 14261's
        // SOURCE_OUT: the traced texture mapping is image_texmap's at origin 0; nptq-* probes).
        let slow_point = !matches!(composite, 0 | 9 | 10 | 11);
        let point = (matches!(composite, 9 | 10) || slow_point) && interpolation == 0 && m[..4] != [1.0, 0.0, 0.0, 1.0];
        // That block resamples into an 8-bit intermediate; on a float target an axis-aligned draw composites through
        // [`Self::blit_image`]'s float intermediate instead (a LINEAR DrawImage under DEST_OVER, SOURCE_OUT, DEST_ATOP,
        // XOR or SOURCE_COPY wrote 8-bit samples into the float target's buffer: inf and garbage).
        let float_axis = m[1] == 0.0 && m[2] == 0.0 && self.target.as_ref().is_some_and(|t| crate::float::is_highp(t.desc.format));
        if source.is_none() && !float_axis && slow_point && interpolation == 0 && !fast && m[1] == 0.0 && m[2] == 0.0 {
            if let Some((inter, rect)) = self.dip_place.and_then(|place| self.slow_nearest_intermediate(bitmap, place)) {
                self.atlas_composite(&inter, rect, composite, None);
                return;
            }
        }
        if source.is_none() && !float_axis && (slow11 || slow_linear || point || m[1] != 0.0 || m[2] != 0.0) && (m[0] * m[3] - m[1] * m[2]) != 0.0 {
            use crate::effects::builtin::transform as tfx;
            let reach = self.reach_rect();
            let Some(saved) = self.target.take() else { return };
            let (w, h) = (saved.desc.width, saved.desc.height);
            let matrix = [m[0] as f32, m[1] as f32, m[2] as f32, m[3] as f32, p[0], p[1]];
            if crate::float::is_highp(saved.desc.format) {
                // A rotated or skewed draw on a float target resamples in float into a transparent float intermediate and
                // composites that. The 8-bit kernels below wrote bytes into the float buffer: inf and garbage. sc234:
                // NEAREST exact; CUBIC rings within 11/255 of native's.
                let input = tfx::Input::new(bitmap, 0, 0).with_float_output();
                let realized = tfx::to_bitmap_precision(w, h, 0.0, 0.0, true,
                    tfx::affine_at(&input, matrix, tfx::Interp::from_mode(interpolation), tfx::Border::Soft, [0.0; 2], false));
                let mut pixels: Vec<[f32; 4]> = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| realized.texel_rgba(x as i32, y as i32)).collect();
                if interpolation == 1 && composite != 11 {
                    // LINEAR is the 8-bit path's WARP draw (unit-triangle planes at the device bbox's origin, the Q8
                    // address), its bilinear sample kept in float rather than stored `>> 7` (sc234: exact for every mode
                    // but BOUNDED_SOURCE_COPY, whose quad pieces keep the affine sampler: within 1.2/255).
                    let (bw, bh) = (bitmap.width as f32, bitmap.height as f32);
                    let cs = [(0.0, 0.0), (bw, 0.0), (0.0, bh), (bw, bh)].map(|(x, y)| [matrix[0] * x + matrix[2] * y + matrix[4], matrix[1] * x + matrix[3] * y + matrix[5]]);
                    let lo = |k: usize| cs.iter().map(|c| c[k]).fold(f32::INFINITY, f32::min).floor().max(0.0) as i32;
                    let hi = |k: usize, n: u32| (cs.iter().map(|c| c[k]).fold(f32::NEG_INFINITY, f32::max).ceil() as i32).min(n as i32);
                    let (bx0, by0, bx1, by1) = if composite == 10 { (0, 0, w as i32, h as i32) } else { (lo(0), lo(1), hi(0, w), hi(1, h)) };
                    let (ox, oy) = if matches!(composite, 0 | 9) { (0, 0) } else { (bx0, by0) };
                    let local = self.image_texmap(bitmap, offset, p, [ox as f32, oy as f32], composite);
                    let eval = |n: [f32; 6], x: f32, y: f32| [(x * n[0] + y * n[2]) + n[4], (x * n[1] + y * n[3]) + n[5]];
                    let tri = [(0.0f32, 0.0f32), (1.0, 0.0), (1.0, 1.0)].map(|(x, y)| eval(local, x, y));
                    let pu = crate::warp::plane::setup_triangle_attr([0, 256, 256], [0, 0, 256], tri.map(|t| t[0]));
                    let pv = crate::warp::plane::setup_triangle_attr([0, 256, 256], [0, 0, 256], tri.map(|t| t[1]));
                    let tex = crate::sampling::Texture { data: &bitmap.premul_bgra, width: bitmap.width, height: bitmap.height };
                    let q = |t: f32, n: f32, floor: bool| {
                        let a = (t.clamp(-10.0, 10.0) * n - 0.5) * 256.0;
                        let q = if floor { a.floor() } else { a.round_ties_even() } as i32;
                        (q >> 8, q & 0xff)
                    };
                    let v2 = bitmap.width.is_power_of_two() && bitmap.height.is_power_of_two();
                    pixels.iter_mut().for_each(|v| *v = [0.0; 4]);
                    for y in by0..by1 { for x in bx0..bx1 {
                        let (px, py) = ((x - ox) as f32 + 0.5, (y - oy) as f32 + 0.5);
                        let (u, v) = (px * pu[0] + (py * pu[1] + pu[2]), px * pv[0] + (py * pv[1] + pv[2]));
                        let (qx, qy) = (q(u, bw, bitmap.width.is_power_of_two()), q(v, bh, v2));
                        let s: [f32; 4] = core::array::from_fn(|ch| crate::sampling::norm_v128(tex.bilinear_v128([crate::sampling::Extend::Transparent; 2], ch, qx, qy)));
                        pixels[y as usize * w as usize + x as usize] = [s[2], s[1], s[0], s[3]];
                    }}
                }
                let mut inter = saved.clone();
                inter.data.fill(0);
                crate::float::FloatTarget::new(&mut inter).blit(&pixels, w as i32, h as i32, 0, 0);
                self.target = Some(saved);
                let mode = if composite == 11 { 10 } else { composite };
                crate::float::FloatTarget::new(self.target.as_mut().unwrap()).composite(&inter, mode, [0, 0, w as i32, h as i32]);
                return;
            }
            // The resample is WARP's geometry setup kernel, whose Q8 texel address rounds to nearest
            // (drawimage-bm-shear-12 exact).
            let mut inter = saved.clone();
            let mut ring_buf = saved.clone();
            ring_buf.data.fill(0);
            let mut ring_rect: Option<[i32; 4]> = None;
            if slow11 && interpolation == 0 && m[..4] == [1.0, 0.0, 0.0, 1.0] {
                // Unscaled NEAREST is a copy at the half-up rounded position (dcopy-c11-i0-*: 5.5 lands on 6,
                // where SOURCE_COPY's sampler drops the column instead).
                inter.data.fill(0);
                bitmap::draw_image(&mut inter, bitmap, p.map(|v| (v + 0.5).floor()), 0, Composite::SourceCopy);
            } else if interpolation != 1 && !point {
                // Axis-aligned at a positive scale, CUBIC copies through the same textured draw as
                // SOURCE_COPY's and samples as it does; a magnified HIGH_QUALITY_CUBIC samples cubic from an
                // atlas placed at the output's rounded corner, as the source-over draw does (bm11 copy probes).
                let pos = slow11 && matrix[1] == 0.0 && matrix[2] == 0.0 && matrix[0] > 0.0 && matrix[3] > 0.0;
                let hq_mag = pos && interpolation == 5 && matrix[0] >= 1.0 && matrix[3] >= 1.0;
                let shared = pos && matches!(interpolation, 2 | 3) || hq_mag;
                let input = tfx::Input::new(bitmap, 0, 0).with_rne_address();
                let realized = if hq_mag {
                    let input = tfx::Input::new(bitmap, 0, 0);
                    let m44 = [matrix[0] as f64, 0.0, 0.0, 0.0, 0.0, matrix[3] as f64, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, p[0] as f64, p[1] as f64, 0.0, 1.0];
                    let lo = tfx::projective_extent(&input, m44, tfx::Interp::Cubic, tfx::Border::Soft).map_or([0.0; 2], |(lo, _)| lo.map(|v| (v + 0.5).floor()));
                    let sample = tfx::affine_at(&input, matrix, tfx::Interp::Cubic, tfx::Border::Soft, [-lo[0], -lo[1]], true);
                    tfx::to_bitmap_precision(w, h, 0.0, 0.0, false, |x, y| sample(x, y))
                } else if shared {
                    // The copy's whole-target quad covers MULTI_SAMPLE_LINEAR's whole tap reach, as PLUS's rect does.
                    // It places the image by its own device matrix, as SOURCE_COPY's draw does.
                    let placed = self.image_brush_matrix(offset, 10).unwrap_or(matrix);
                    tfx::to_bitmap_precision(w, h, 0.0, 0.0, false, bitmap_image_axis(bitmap, placed, interpolation, false, interpolation == 3, false, false,
                        self.image_texmap(bitmap, offset, p, [0.0; 2], 10), self.texture_dims(bitmap)))
                } else {
                    tfx::to_bitmap_precision(w, h, 0.0, 0.0, false,
                        tfx::affine_at(&input, matrix, tfx::Interp::from_mode(interpolation), tfx::Border::Soft, [0.0; 2], false))
                };
                for y in 0..h { for x in 0..w {
                    let [r, g, b, a] = realized.texel_rgba(x as i32, y as i32);
                    let i = y as usize * inter.stride + x as usize * 4;
                    let q = |v: f32| if shared { (v.clamp(0.0, 1.0) * 255.0).round_ties_even() as u8 } else { (v * 255.0).round() as u8 };
                    inter.data[i..i + 4].copy_from_slice(&[b, g, r, a].map(q));
                }}
            } else {
                // LINEAR is a WARP draw into the intermediate (SetupTriangleInternal2 traces of
                // drawimage-bm-rot-05/-11): texcoords are the inverse map normalized by the bitmap size,
                // the pixel kernel rounds its Q8 address to nearest and stores the sample truncated
                // (`>> 7`). Every mode but BOUNDED_SOURCE_COPY fits the plane over one unit triangle at
                // the device bbox's origin (SOURCE_COPY too, rot-15); BOUNDED_SOURCE_COPY draws the
                // target-covering quad (0.5, 1/16)..(w−0.5, h+1/16) as (BL, TL, TR), (BL, TR, BR),
                // frustum-clipped at the viewport, one plane per piece.
                let (bw, bh) = (bitmap.width as f32, bitmap.height as f32);
                let texmap = |o: [f32; 2]| self.image_texmap(bitmap, offset, p, o, composite);
                let eval = |n: [f32; 6], x: f32, y: f32| [(x * n[0] + y * n[2]) + n[4], (x * n[1] + y * n[3]) + n[5]];
                let device = texmap([0.0, 0.0]);
                let inv = |x: f32, y: f32| eval(device, x, y);
                let tex = crate::sampling::Texture { data: &bitmap.premul_bgra, width: bitmap.width, height: bitmap.height };
                // WARP's Q8 address: a power-of-two texture dimension floors, any other rounds to nearest
                // (bm11-d192-int-in / -d120-int-in: a 4×4 bitmap lands one weight below nearest).
                let sample = |u: f32, v: f32| -> [u8; 4] {
                    let q = |t: f32, n: f32, floor: bool| {
                        let a = (t.clamp(-10.0, 10.0) * n - if point { 0.0 } else { 0.5 }) * 256.0;
                        let q = if floor { a.floor() } else { a.round_ties_even() } as i32;
                        (q >> 8, q & 0xff)
                    };
                    // PLUS's point kernel floors whatever the size (npx2/npd120 JIT: roundps 1 on u·(n·256)), as
                    // do the slow-path composites'.
                    let plus = point && (composite == 9 || slow_point);
                    // v takes the power-of-two path only when both dimensions are (r9 14527's 3×8).
                    let v2 = bitmap.width.is_power_of_two() && bitmap.height.is_power_of_two();
                    let (qx, qy) = (q(u, bw, plus || bitmap.width.is_power_of_two()), q(v, bh, plus || v2));
                    let (qx, qy) = if point { ((qx.0, 0), (qy.0, 0)) } else { (qx, qy) };
                    core::array::from_fn(|ch| (tex.bilinear_v128([crate::sampling::Extend::Transparent; 2], ch, qx, qy) >> 7) as u8)
                };
                inter.data.fill(0);
                let stride = inter.stride;
                // An unscaled draw of a texture whose width is not a power of two is the fixed iterator
                // instead (r5 5077's 13×8 under SOURCE_OUT at a fractional offset; r5 5028's 1×11 is
                // this draw), BOUNDED_SOURCE_COPY's quad included (r8 9622's 6×4, b11-c11-*); scaled,
                // it stays this draw (fuzz b7 9993, r3 2541).
                let axis = matrix[..4] == [1.0, 0.0, 0.0, 1.0] && !bitmap.width.is_power_of_two();
                let blit = bitmap_image_axis(bitmap, matrix, 1, false, false, false, false, device, [bitmap.width, bitmap.height]);
                // A LINEAR slow-path draw reaches half a texel past its rect; that ring goes to the frame's atlas.
                let ring = if interpolation == 1 && !point && composite != 10 && matrix[1] == 0.0 && matrix[2] == 0.0 {
                    (0.5 * matrix[0].abs().max(matrix[3].abs())).ceil() as i32 + 1
                } else { 0 };
                if composite != 11 {
                    let cs = [(0.0, 0.0), (bw, 0.0), (0.0, bh), (bw, bh)].map(|(x, y)| [matrix[0] * x + matrix[2] * y + matrix[4], matrix[1] * x + matrix[3] * y + matrix[5]]);
                    let lo = |k: usize| cs.iter().map(|c| c[k]).fold(f32::INFINITY, f32::min).floor().max(0.0) as i32;
                    let hi = |k: usize, n: u32| (cs.iter().map(|c| c[k]).fold(f32::NEG_INFINITY, f32::max).ceil() as i32).min(n as i32);
                    // SOURCE_COPY realizes over the whole target (unit triangle at its origin, rot-15).
                    let (bx0, by0, bx1, by1) = if composite == 10 { (0, 0, w as i32, h as i32) } else { (lo(0), lo(1), hi(0, w), hi(1, h)) };
                    // A SOURCE_OVER draw goes straight to the target, its unit triangle at the screen origin
                    // (r5 5028's 1×11 trace), as does a slow-path NEAREST draw (r9 14261's traced mapping); a
                    // LINEAR intermediate's sits at the origin of the frame's atlas, where the image's visible part
                    // takes the next slot.
                    let span = |k: usize| (cs.iter().map(|c| c[k]).fold(f32::INFINITY, f32::min).floor() as i32, cs.iter().map(|c| c[k]).fold(f32::NEG_INFINITY, f32::max).ceil() as i32);
                    let full = { let ((x0, x1), (y0, y1)) = (span(0), span(1)); [x0, y0, x1, y1] };
                    let vis = Self::atlas_visible(full, reach);
                    let slot = if ring > 0 { self.atlas.next(vis[2] - vis[0], vis[3] - vis[1]).unwrap_or([0, 0]) } else { [0, 0] };
                    let direct = matches!(composite, 0 | 9) || (slow_point && interpolation == 0);
                    let (ox, oy) = if direct { (0, 0) } else if ring > 0 { (vis[0] - slot[0], vis[1] - slot[1]) } else { (bx0, by0) };
                    // In the atlas the brush is the replay's (the slot shift through the DPI round trip), under a fill
                    // world of the replay's DPI scale; a draw straight to the target keeps its own mapping.
                    let replay = (!direct && ring > 0 && !self.effect_atlas).then_some(self.dip_place).flatten()
                        .and_then(|place| self.replay_brush(place, [(slot[0] - vis[0]) as f32, (slot[1] - vis[1]) as f32]));
                    let local = match replay {
                        Some((s, o, k)) => hw_bitmap_row(self.texture_dims(bitmap).map(|v| v as f32), [s[0], 0.0, 0.0, s[1], o[0], o[1]], [k[0], 0.0, 0.0, k[1], 0.0, 0.0]),
                        None => texmap([ox as f32, oy as f32]),
                    };
                    let tri = [(0.0f32, 0.0f32), (1.0, 0.0), (1.0, 1.0)].map(|(x, y)| eval(local, x, y));
                    let pu = crate::warp::plane::setup_triangle_attr([0, 256, 256], [0, 0, 256], tri.map(|t| t[0]));
                    let pv = crate::warp::plane::setup_triangle_attr([0, 256, 256], [0, 0, 256], tri.map(|t| t[1]));
                    for y in (by0 - ring).max(0)..(by1 + ring).min(h as i32) { for x in (bx0 - ring).max(0)..(bx1 + ring).min(w as i32) {
                        let (px, py) = ((x - ox) as f32 + 0.5, (y - oy) as f32 + 0.5);
                        let i = y as usize * stride + x as usize * 4;
                        let out = if (bx0..bx1).contains(&x) && (by0..by1).contains(&y) { &mut inter } else { &mut ring_buf };
                        if axis {
                            let [r, g, b, a] = blit(x as f32 + 0.5, y as f32 + 0.5);
                            out.data[i..i + 4].copy_from_slice(&[b, g, r, a].map(|v| (v * 255.0).round() as u8));
                            continue;
                        }
                        out.data[i..i + 4].copy_from_slice(&sample(px * pu[0] + (py * pu[1] + pu[2]), px * pv[0] + (py * pv[1] + pv[2])));
                    }}
                    if ring > 0 { ring_rect = Some(full); }
                } else {
                    // The right-edge AA ring (x ∈ [w−0.5, w+0.5], coverage 1 → 0) is drawn with its
                    // coverage attribute: the kernel forms `c = rne(clamp01(cov)·255)` and lerps the sample
                    // into the target as `(s·c + (s·c>>8) + d·(255−c) + (d·(255−c)>>8) + 128) >> 8`
                    // (jit-dump of drawimage-bm-rot-13); interior pieces have coverage 1 (`= s`).
                    // The quad's x edges sit 1/(1024·h) left of the half-pixel lines and each ring spans
                    // one pixel from them (DDI vertex buffers at 32×48, 40×24, 64×64, 100×30); the
                    // texcoords are evaluated at those raw positions.
                    let (fw, fh) = (w as f32, h as f32);
                    let shift = 1.0 / (1024.0 * fh);
                    let (xl, xr) = (0.5 - shift, (fw - 0.5) - shift);
                    let corner = |x: f32, y: f32, cov: f32| { let t = inv(x, y); [x, y, t[0], t[1], cov] };
                    let (tl, tr) = (corner(xl, 0.0625, 1.0), corner(xr, 0.0625, 1.0));
                    let (bl, br) = (corner(xl, fh + 0.0625, 1.0), corner(xr, fh + 0.0625, 1.0));
                    let (rtl, rtr) = (corner(xr, 0.0625, 1.0), corner(xr + 1.0, 0.0625, 0.0));
                    let (rbl, rbr) = (corner(xr, fh + 0.0625, 1.0), corner(xr + 1.0, fh + 0.0625, 0.0));
                    let (ltl, ltr) = (corner(xl - 1.0, 0.0625, 0.0), corner(xl, 0.0625, 1.0));
                    let (lbl, lbr) = (corner(xl - 1.0, fh + 0.0625, 0.0), corner(xl, fh + 0.0625, 1.0));
                    let viewport = [0.0, 0.0, fw, fh];
                    let pieces: Vec<[[f32; 5]; 3]> = [[lbl, ltl, ltr], [lbl, ltr, lbr], [bl, tl, tr], [bl, tr, br], [rbl, rtl, rtr], [rbl, rtr, rbr]]
                        .into_iter().flat_map(|t| crate::warp::clip_fan(t, viewport)).collect();
                    // Aliased, the pieces only bound the copy: the image samples the unit-triangle plane at
                    // the screen origin, as the other modes' draws (r9 14527, traced: one unit triangle).
                    let unit = self.state.aa == 1;
                    let tri = [(0.0f32, 0.0f32), (1.0, 0.0), (1.0, 1.0)].map(|(x, y)| eval(device, x, y));
                    let gu = crate::warp::plane::setup_triangle_attr([0, 256, 256], [0, 0, 256], tri.map(|t| t[0]));
                    let gv = crate::warp::plane::setup_triangle_attr([0, 256, 256], [0, 0, 256], tri.map(|t| t[1]));
                    for t in pieces {
                        let pos = t.map(|v| [v[0], v[1]]);
                        let (p, area) = crate::warp::snap_tri(pos);
                        let (sx, sy) = (p.map(|q| q[0]), p.map(|q| q[1]));
                        let pu = crate::warp::plane::setup_triangle_attr(sx, sy, t.map(|v| v[2]));
                        let pv = crate::warp::plane::setup_triangle_attr(sx, sy, t.map(|v| v[3]));
                        let pc = crate::warp::plane::setup_triangle_attr(sx, sy, t.map(|v| v[4]));
                        crate::warp::cover_tri(p, area, w as i64, h as i64, |x, y| {
                            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                            let i = y * stride + x * 4;
                            let s = if axis { let [r, g, b, a] = blit(px, py); [b, g, r, a].map(|v| (v * 255.0).round() as u8) }
                                else if unit { sample(px * gu[0] + (py * gu[1] + gu[2]), px * gv[0] + (py * gv[1] + gv[2])) }
                                else { sample(px * pu[0] + (py * pu[1] + pu[2]), px * pv[0] + (py * pv[1] + pv[2])) };
                            let c = ((px * pc[0] + (py * pc[1] + pc[2])).clamp(0.0, 1.0) * 255.0).round_ties_even() as i32;
                            let out = crate::pixels::copy_tess::lerp8(saved.data[i..i + 4].try_into().unwrap(), s, c);
                            inter.data[i..i + 4].copy_from_slice(&out);
                        });
                    }
                }
            }
            self.target = Some(saved);
            let mode = if composite == 11 { 10 } else { composite };
            match ring_rect {
                Some(rect) => {
                    let ring_px = |x: i32, y: i32| { let i = y as usize * ring_buf.stride + x as usize * 4; ring_buf.data[i..i + 4].try_into().unwrap() };
                    let reach = (0.5 * m[0].abs().max(m[3].abs())).ceil() as i32 + 1;
                    self.atlas_composite(&inter, rect, mode, Some((&ring_px, reach)));
                }
                None => self.composite_over_target(&inter, mode, None),
            }
            return;
        }
        self.blit_image(bitmap, p, source, interpolation, composite);
    }

    /// Composite `bitmap` at device top-left `p` under `composite` — the device-space core of
    /// `DrawImage(bitmap)`, shared by the direct draw and command-list replay. Per-mode fast/slow-path
    /// reasoning is documented on [`Self::draw_image_bitmap`].
    fn blit_image(&mut self, bitmap: &Bitmap, p: [f32; 2], source: Option<[f32; 4]>, interpolation: u32, composite: u32) {
        if self.target.as_ref().is_some_and(|b| crate::float::is_highp(b.desc.format)) {
            // Float target: source-over DrawImage of a whole bitmap ≡ a 1:1 DrawBitmap. A source sub-rect
            // or another composite draws that sub-rect 1:1 into a transparent float intermediate and
            // composites it (native copies the texels exactly under every mode: float DrawImage probes).
            let (bw, bh) = (bitmap.width as f32, bitmap.height as f32);
            if composite == 0 && source.is_none() {
                let buf = self.target.as_mut().unwrap();
                crate::float::draw_bitmap(buf, bitmap, [p[0], p[1], p[0] + bw, p[1] + bh], [0.0, 0.0, bw, bh], 1.0, interpolation);
                return;
            }
            let [sl, st, sr, sb] = source.unwrap_or([0.0, 0.0, bw, bh]);
            let src = [sl.min(sr).max(0.0), st.min(sb).max(0.0), sl.max(sr).min(bw), st.max(sb).min(bh)];
            let dest = [p[0], p[1], p[0] + (src[2] - src[0]), p[1] + (src[3] - src[1])];
            let target = self.target.as_ref().unwrap();
            let mut inter = target.clone();
            inter.data.fill(0);
            crate::float::draw_bitmap(&mut inter, bitmap, dest, src, 1.0, interpolation);
            let r = self.reach_rect();
            // BOUNDED_SOURCE_COPY is bounded by the imageRectangle; without one it copies the whole reach
            // (the fast path's whole-target copy). ponytail: native's non-unit-scale bail is not modelled.
            let bounds = if composite == 11 && source.is_some() {
                [dest[0].floor() as i32, dest[1].floor() as i32, dest[2].ceil() as i32, dest[3].ceil() as i32]
            } else {
                [r[0] as i32, r[1] as i32, r[2] as i32, r[3] as i32]
            };
            crate::float::FloatTarget::new(self.target.as_mut().unwrap()).composite(&inter, composite, bounds);
            return;
        }
        let m = self.device_transform();
        let det = m[0] * m[3] - m[1] * m[2];
        // Whole-bitmap fast path for the native bounded/copy modes (byte-exact, proven by bitmaps-11/12).
        // BOUNDED_SOURCE_COPY(11) with no imageRectangle runs native's GPU fast path
        // (native's fast image-draw path, SOURCE_COPY), which only
        // engages when the device offset is integer-aligned (its RoundSat snap bails on a fractional
        // offset). When it engages it copies the whole target — clearing outside the image's extent, so
        // the "bound" is the imageRectangle, not the bitmap (drawimage-bm-11). When the offset is
        // fractional the fast path bails and nothing is drawn (fuzz 5451/5984/6238 leave the backdrop
        // untouched, even where the bitmap overlaps the target).
        if source.is_none() {
            // A singular (det 0) device transform maps the bitmap to zero area: native's SOURCE_COPY
            // then copies an empty source over the whole target (clears it), while the other modes
            // composite nothing (fuzz 7309 composite 10 under a collinear transform).
            if det == 0.0 {
                self.composite_empty(composite);
                return;
            }
            // A non-finite device offset (singular transform → NaN/inf) draws nothing (fuzz 6238).
            if !p[0].is_finite() || !p[1].is_finite() {
                return;
            }
            // BOUNDED_SOURCE_COPY(11) with no imageRectangle runs native's GPU fast path
            // (the fast image-draw path, SOURCE_COPY) — a device 1:1 whole-target copy that clears outside the
            // image to transparent. It engages only at UNIT device scale (identity world + dpi 96); a
            // fractional offset still erases there (fuzz 3275: colored clear + comp11 at (24.78,52.74) →
            // (0,0) transparent — the old "fractional → draw nothing" was mis-validated on transparent
            // backdrops where an erase and a no-op look identical). At non-unit scale (e.g. dpi 192) the
            // 1:1 fast path can't represent the magnification, so an integer offset still takes it but a
            // fractional one bails and native draws nothing (fuzz 2232/2935 at dpi 192 leave the backdrop).
            let fast = match composite {
                0 => Some(Composite::SourceOver),
                9 => Some(Composite::Plus),
                10 | 11 => Some(Composite::SourceCopy),
                _ => None,
            };
            if let Some(mode) = fast {
                if let Some(buf) = &mut self.target {
                    bitmap::draw_image(buf, bitmap, p, interpolation, mode);
                }
                return;
            }
        }
        // Slow / bounded path: realize the (sub-rect of the) bitmap into a transparent target-sized
        // intermediate at the device rect, then composite it over the target. DrawImage maps one image
        // pixel to one DIP, so the device footprint is the source size times the device transform's axis
        // scale (dpi/96 × world scale) — at dpi 96 with no world scale this is 1:1, but e.g. dpi 120
        // stretches the bitmap ×1.25 (fuzz 5145: composite DESTINATION_OUT at dpi 120).
        let (sx, sy) = (m[0] as f32, m[3] as f32);
        let (bw, bh) = (bitmap.width as f32, bitmap.height as f32);
        let [sl, st, sr, sb] = source.unwrap_or([0.0, 0.0, bw, bh]);
        let src = [sl.min(sr), st.min(sb), sl.max(sr), st.max(sb)];
        let dest = [p[0], p[1], p[0] + (src[2] - src[0]) * sx, p[1] + (src[3] - src[1]) * sy];
        // The image is its bitmap: the part of the image rectangle past the bitmap is transparent, so draw only the
        // rectangle's overlap with it, where it lands (BlendImage with a rect past a 40 px bitmap draws 20 px).
        let clip = [src[0].max(0.0), src[1].max(0.0), src[2].min(bw), src[3].min(bh)];
        if composite == 11 && self.bcopy_rect_copies_nothing(bitmap, p, src, interpolation) {
            return;
        }
        let Some(saved) = self.target.take() else { return };
        let mut inter = saved.clone();
        inter.data.fill(0);
        if clip[0] < clip[2] && clip[1] < clip[3] {
            let at = |k: usize, v: f32| p[k] + (v - src[k]) * if k == 0 { sx } else { sy };
            bitmap::draw_image_rect(&mut inter, bitmap, [at(0, clip[0]), at(1, clip[1]), at(0, clip[2]), at(1, clip[3])], clip, interpolation);
        }
        self.target = Some(saved);
        let bounds = (composite == 11)
            .then(|| [dest[0].floor() as i32, dest[1].floor() as i32, dest[2].ceil() as i32, dest[3].ceil() as i32]);
        let mode = if composite == 11 { 10 } else { composite };
        self.composite_over_target(&inter, mode, bounds);
    }

    /// Whether a BOUNDED_SOURCE_COPY DrawImage of `bitmap` with the image rect `src` (sorted, texels) placed at device
    /// `p` copies nothing. Off the fast path (MULTI_SAMPLE_LINEAR, or not a unit scale at a whole offset) it copies
    /// nothing when the rect misses the bitmap entirely, while on it it clears the rect (sc136, r27 32162). Nor does it
    /// copy when that overlap, placed, reaches no pixel of the target: the no-rect slow path's reach test (a pixel
    /// centre inside the placed rect grown by the filter's footprint) on the overlap (r27 32550: the overlap lands
    /// above the target, and the rest of the rectangle keeps the backdrop).
    fn bcopy_rect_copies_nothing(&self, bitmap: &Bitmap, p: [f32; 2], src: [f32; 4], interpolation: u32) -> bool {
        let m = self.device_transform();
        if m[..4] == [1.0, 0.0, 0.0, 1.0] && interpolation != 3 && fast_whole(p[0]) && fast_whole(p[1]) {
            return false;
        }
        let (sx, sy) = (m[0] as f32, m[3] as f32);
        let (bw, bh) = (bitmap.width as f32, bitmap.height as f32);
        let clip = [src[0].max(0.0), src[1].max(0.0), src[2].min(bw), src[3].min(bh)];
        if !(clip[0] < clip[2] && clip[1] < clip[3]) {
            return true;
        }
        let [cl, ct, cr, cb] = self.reach_rect();
        let (rt, rp) = [(0.0, 0.0), (0.5, 0.0), (1.5, 0.0), (0.5, 0.5), (0.5, 1.0), (1.5, 0.0)][interpolation.min(5) as usize];
        let reaches = |k: usize, c0: f32, c1: f32| {
            let s = if k == 0 { sx } else { sy };
            let (a, b) = (p[k] + (clip[k] - src[k]) * s, p[k] + (clip[k + 2] - src[k]) * s);
            let (lo, hi, r) = (a.min(b), a.max(b), rt * s.abs() + rp);
            ((lo - r - 0.5).floor() + 1.0).max(c0) <= ((hi + r - 0.5).ceil() - 1.0).min(c1 - 1.0)
        };
        !(reaches(0, cl, cr) && reaches(1, ct, cb))
    }

    // --- internals ----------------------------------------------------------------------------

    /// The inverse DPI scale (1 in pixel units) and `bitmap`'s DIP position at DIP `offset` (from the
    /// world alone; the device position `p` itself at dpi 96). d2d1 forms the inverse as `96·(1/dpi)`:
    /// at dpi 120 that product is the exact f32 midpoint above fl(0.8) and rounds up to 0.80000007
    /// (lsc-8x8-d120 trace: u(1,0) = fl(fl(0.8)+1ulp)/8 − 0.5).
    fn image_dip(&self, offset: [f32; 2], p: [f32; 2]) -> ((f32, f32), [f32; 2]) {
        if self.state.unit != 0 || self.dpi == (96.0, 96.0) { return ((1.0, 1.0), p); }
        ((96.0 * (1.0 / self.dpi.0), 96.0 * (1.0 / self.dpi.1)), apply(&self.state.transform.map(|v| v as f64), offset[0], offset[1]))
    }

    /// The intermediate a slow-path composite reads for an axis-aligned NEAREST DrawImage of a bitmap (the composites
    /// other than SOURCE_OVER, PLUS and the copies), built as Direct2D builds it. Its list replays the SOURCE_OVER draw
    /// into an atlas whose whole-pixel origin is the image's DIP offset under world x dpi/96, floored. The replay hands
    /// its transform over in DIPs: the world is scaled to device pixels, moved by that origin and taken back through the
    /// inverse DPI matrix (one reciprocal determinant), and the atlas target applies dpi/96 again. At dpi 120 that is a
    /// 1.24999988 scale and a y translation of -4.99999952 for an origin of 5. The draw fills the atlas with the bitmap as
    /// an image brush at the offset under that matrix, sampled as [`NearestBlit`] says. VM probes: 64 intermediate readbacks
    /// (SOURCE_OUT onto transparent, one colour per texel), 56 origin-sweep scenes at 1.5, 2.5 and 4 px per texel, and
    /// d2d1-side captures of the brush matrices; every one exact. `None` leaves the draw to the general resample: a
    /// flipped or skewed world, or an image the target or a clip cuts on its left or top, whose atlas origin is unprobed.
    fn slow_nearest_intermediate(&self, bitmap: &Bitmap, (w, offset, fold): ([f32; 6], [f32; 2], [f32; 2])) -> Option<(PixelBuffer, [i32; 4])> {
        if w[1] != 0.0 || w[2] != 0.0 || !(w[0] > 0.0 && w[3] > 0.0) {
            return None;
        }
        let target = self.target.as_ref()?;
        let (dx, dy) = (self.dpi.0 / 96.0, self.dpi.1 / 96.0);
        let origin = [((offset[0] * (w[0] * dx)) + offset[1] * (w[2] * dx)) + w[4] * dx, ((offset[0] * (w[1] * dy)) + offset[1] * (w[3] * dy)) + w[5] * dy]
            .map(f32::floor);
        let reach = self.reach_rect();
        if !origin.iter().all(|v| v.is_finite()) || origin[0] < reach[0] || origin[1] < reach[1] {
            return None;
        }
        // The device rect it packs into the atlas: its bounds rounded out.
        let far = [(((offset[0] + bitmap.width as f32 * fold[0]) * (w[0] * dx)) + offset[1] * (w[2] * dx)) + w[4] * dx,
            ((offset[0] * (w[1] * dy)) + (offset[1] + bitmap.height as f32 * fold[1]) * (w[3] * dy)) + w[5] * dy].map(f32::ceil);
        if !far.iter().all(|v| v.is_finite()) || far[0] - origin[0] > 4096.0 || far[1] - origin[1] > 4096.0 {
            return None;
        }
        let rect = [origin[0] as i32, origin[1] as i32, far[0] as i32, far[1] as i32];
        let vis = Self::atlas_visible(rect, self.reach_rect());
        let at = self.atlas.next(vis[2] - vis[0], vis[3] - vis[1]).unwrap_or([0, 0]);
        let (s, o, k) = self.replay_brush((w, offset, fold), [(at[0] - rect[0]) as f32, (at[1] - rect[1]) as f32])?;
        let blit = NearestBlit::new(s, o, [bitmap.width as f32, bitmap.height as f32], k, true);
        let mut inter = target.clone();
        inter.data.fill(0);
        // Atlas rows and columns that land on the target (a huge placement saturates and covers nothing).
        let org = [(rect[0] - at[0]) as i64, (rect[1] - at[1]) as i64];
        let len = [inter.desc.width as i64, inter.desc.height as i64];
        let span = |c: usize| (blit.lo[c] as i64).max(org[c].saturating_neg())..(blit.hi[c] as i64).min(len[c].saturating_sub(org[c]));
        for ay in span(1) {
            for ax in span(0) {
                let Some((tx, ty)) = blit.texel(ax, ay) else { continue };
                let [r, g, b, a] = bitmap.texel_rgba(tx, ty);
                let di = (ay + org[1]) as usize * inter.stride + (ax + org[0]) as usize * 4;
                inter.data[di..di + 4].copy_from_slice(&[b, g, r, a].map(|v| (v * 255.0).round() as u8));
            }
        }
        Some((inter, rect))
    }

    /// Composite a slow-path DrawImage's intermediate `inter` (target-sized, holding the device `rect`) through the frame's
    /// atlas. The draw's whole rendering, `spill` past `rect` included, is source-over'd into the atlas at its slot, so it
    /// lands on what earlier draws left there and an earlier slot it reaches changes under that slot's composite, which
    /// runs again over its own rect from the target it saw, provided nothing has touched that rect since and the clip is
    /// the same (r27 32938: a CUBIC MASK_INVERT image's left filter column lands on the previous XOR image's last column,
    /// one row lower; DDI: both composites read one atlas, the atlas cleared once per frame).
    fn atlas_composite(&mut self, inter: &PixelBuffer, rect: [i32; 4], composite: u32, spill: Option<(&dyn Fn(i32, i32) -> [u8; 4], i32)>) {
        const SIZE: i32 = 512;
        let reach = self.reach_rect();
        let Some(target) = self.target.as_ref() else { return };
        let (tw, th) = (target.desc.width as i32, target.desc.height as i32);
        let clip_rect = |r: [i32; 4]| [r[0].max(0), r[1].max(0), r[2].min(tw), r[3].min(th)];
        // The slot holds the part of the image the draw needs. When that is not the whole image, the replay into the
        // atlas is clipped to it and the filter's reach stays out (VM probes: a CUBIC image cut by the target's
        // bottom edge leaves nothing under the next slot).
        let clipped = Self::atlas_visible(rect, reach);
        let spill = if clipped == rect { spill } else { None };
        let rect = clipped;
        let copy_rect = |b: &PixelBuffer, r: [i32; 4]| -> Vec<u8> {
            let mut v = Vec::new();
            for y in r[1]..r[3] { for x in r[0]..r[2] { let i = y as usize * b.stride + x as usize * 4; v.extend_from_slice(&b.data[i..i + 4]); } }
            v
        };
        let Some(at) = self.atlas.alloc(rect[2] - rect[0], rect[3] - rect[1]) else {
            self.composite_over_target(inter, composite, None);
            return;
        };
        if self.atlas.texels.is_empty() {
            self.atlas.texels = vec![0u8; (SIZE * SIZE) as usize * 4];
        }
        // The draw into the atlas: inside its rect the intermediate, past it the filter's reach (device pixels on the
        // target only), each source-over'd onto the atlas.
        let over = |dst: &mut [u8], src: [u8; 4]| {
            let ia = 255 - src[3] as i32;
            for c in 0..4 {
                let t = dst[c] as i32 * ia;
                dst[c] = ((((src[c] as i32) << 8) + t + (t >> 8) + 128) >> 8).min(255) as u8;
            }
        };
        let reach_px = spill.map_or(0, |(_, r)| r);
        let mut touched = [i32::MAX, i32::MAX, i32::MIN, i32::MIN];
        for y in (rect[1] - reach_px).max(0)..(rect[3] + reach_px).min(th) {
            for x in (rect[0] - reach_px).max(0)..(rect[2] + reach_px).min(tw) {
                let (u, v) = (at[0] + x - rect[0], at[1] + y - rect[1]);
                if !(0..SIZE).contains(&u) || !(0..SIZE).contains(&v) { continue; }
                let inside = (rect[0]..rect[2]).contains(&x) && (rect[1]..rect[3]).contains(&y);
                let src: [u8; 4] = if inside {
                    let i = y as usize * inter.stride + x as usize * 4;
                    inter.data[i..i + 4].try_into().unwrap()
                } else if let Some((spill, _)) = spill { spill(x, y) } else { continue };
                if src == [0; 4] { continue; }
                let ai = (v * SIZE + u) as usize * 4;
                over(&mut self.atlas.texels[ai..ai + 4], src);
                if !inside { touched = [touched[0].min(u), touched[1].min(v), touched[2].max(u + 1), touched[3].max(v + 1)]; }
            }
        }
        // Earlier slots the spill reached composite again over their own rect.
        for j in (0..self.atlas.slots.len()).rev() {
            let s = &self.atlas.slots[j];
            let (sw, sh) = (s.rect[2] - s.rect[0], s.rect[3] - s.rect[1]);
            if touched[0] >= s.at[0] + sw || touched[2] <= s.at[0] || touched[1] >= s.at[1] + sh || touched[3] <= s.at[1] { continue; }
            let sr = clip_rect(s.rect);
            let Some(target) = self.target.as_ref() else { return };
            if s.reach != reach || copy_rect(target, sr) != s.after { continue; }
            let mut redo = target.clone();
            redo.data.fill(0);
            let (at_j, rect_j, before, mode) = (s.at, s.rect, s.before.clone(), s.composite);
            let mut k = 0;
            for y in sr[1]..sr[3] { for x in sr[0]..sr[2] {
                let i = y as usize * redo.stride + x as usize * 4;
                let ai = ((at_j[1] + y - rect_j[1]) * SIZE + (at_j[0] + x - rect_j[0])) as usize * 4;
                redo.data[i..i + 4].copy_from_slice(&self.atlas.texels[ai..ai + 4]);
                let tgt = self.target.as_mut().unwrap();
                tgt.data[i..i + 4].copy_from_slice(&before[k..k + 4]);
                k += 4;
            }}
            self.composite_over_target(&redo, mode, Some(sr));
            let after = copy_rect(self.target.as_ref().unwrap(), sr);
            self.atlas.slots[j].after = after;
        }
        // This draw's composite reads its slot, under it whatever earlier draws spilled there.
        let Some(target) = self.target.as_ref() else { return };
        let before = copy_rect(target, clipped);
        let mut own = target.clone();
        own.data.fill(0);
        for y in clipped[1]..clipped[3] { for x in clipped[0]..clipped[2] {
            let (u, v) = (at[0] + x - rect[0], at[1] + y - rect[1]);
            let i = y as usize * own.stride + x as usize * 4;
            if (0..SIZE).contains(&u) && (0..SIZE).contains(&v) {
                let ai = (v * SIZE + u) as usize * 4;
                own.data[i..i + 4].copy_from_slice(&self.atlas.texels[ai..ai + 4]);
            } else {
                own.data[i..i + 4].copy_from_slice(&inter.data[i..i + 4]);
            }
        }}
        self.composite_over_target(&own, composite, None);
        let after = copy_rect(self.target.as_ref().unwrap(), clipped);
        self.atlas.slots.push(AtlasSlot { at, rect, composite, reach, before, after });
    }

    /// A slow-path DrawImage's replay into its atlas slot, for an axis-aligned positive world `w`: the image brush's
    /// per-texel scale `s` and origin `o` in atlas pixels, and the fill's world scale `k`. The replay's DIP transform is the
    /// world moved by `shift` (the slot less the rect's corner) through the DPI round trip, which the atlas target scales by
    /// dpi/96 again (1.24999988 and -4.99999952 at dpi 120); the brush is the DIP `offset` under it.
    fn replay_brush(&self, (w, offset, fold): ([f32; 6], [f32; 2], [f32; 2]), shift: [f32; 2]) -> Option<([f32; 2], [f32; 2], [f32; 2])> {
        if w[1] != 0.0 || w[2] != 0.0 || !(w[0] > 0.0 && w[3] > 0.0) {
            return None;
        }
        let (dx, dy) = (self.dpi.0 / 96.0, self.dpi.1 / 96.0);
        let v = 1.0 / (dx * dy);
        if !v.is_finite() {
            return None;
        }
        let (m11, m22) = (((dy * v) * (dx * w[0])) * dx, ((dx * v) * (dy * w[3])) * dy);
        let (t11, t22) = (((dy * v) * ((dx * w[4]) + shift[0])) * dx, ((dx * v) * ((dy * w[5]) + shift[1])) * dy);
        let o = [offset[0] * m11 + t11, offset[1] * m22 + t22];
        Some(([m11 * fold[0], m22 * fold[1]], o, [(96.0 / self.dpi.0) * dx, (96.0 / self.dpi.1) * dy]))
    }

    /// The part of a slow-path image's device `rect` its atlas slot holds: the rect within `reach` (the target and the
    /// draw clip, [`Self::reach_rect`]), all zero when nothing of it is left (an image off the target takes no slot).
    fn atlas_visible(rect: [i32; 4], reach: [f32; 4]) -> [i32; 4] {
        let r = reach.map(|v| v as i32);
        let v = [rect[0].max(r[0]), rect[1].max(r[1]), rect[2].min(r[2]), rect[3].min(r[3])];
        if v[0] < v[2] && v[1] < v[3] { v } else { [0; 4] }
    }

    /// The direct draw's NEAREST blit: the image brush at the DIP offset under the device transform, its per-texel scale
    /// that transform's times the bitmap's DPI fold. `None` for a flipped or skewed world.
    fn direct_nearest_blit(&self, bitmap: &Bitmap, (w, offset, fold): ([f32; 6], [f32; 2], [f32; 2])) -> Option<NearestBlit> {
        if w[1] != 0.0 || w[2] != 0.0 || !(w[0] > 0.0 && w[3] > 0.0) {
            return None;
        }
        let (dx, dy) = (self.dpi.0 / 96.0, self.dpi.1 / 96.0);
        let (m11, m22) = (w[0] * dx, w[3] * dy);
        let o = [((offset[0] * m11) + offset[1] * (w[2] * dx)) + w[4] * dx, ((offset[0] * (w[1] * dy)) + offset[1] * m22) + w[5] * dy];
        if !o.iter().all(|v| v.is_finite()) {
            return None;
        }
        // The fill's temporary world 96/dpi under the target's DPI scale, which is 1/(96*(1/dpi)): 0.99999994 at dpi 120.
        let k = [(96.0 / self.dpi.0) * (1.0 / (96.0 * (1.0 / self.dpi.0))), (96.0 / self.dpi.1) * (1.0 / (96.0 * (1.0 / self.dpi.1)))];
        Some(NearestBlit::new([m11 * fold[0], m22 * fold[1]], o, [bitmap.width as f32, bitmap.height as f32], k, false))
    }

    /// [`image_texmap`] of `bitmap` drawn at DIP `offset` (device position `p`).
    fn image_texmap(&self, bitmap: &Bitmap, offset: [f32; 2], p: [f32; 2], o: [f32; 2], composite: u32) -> [f32; 6] {
        let (inv, pd) = self.image_dip(offset, p);
        // Off dpi 96 the row is native's DrawImage brush, op for op (the texture
        // mapping SetTextureMapping uploads, traced bit-exact on five draws): the brush is the offset
        // under the device transform, its DPI scale dpi/96 on the direct composites but 1/(96·(1/dpi))
        // on the slow path (slow-path command list: 1.2499999 at dpi 120); the fill runs under a
        // temporary world 96/dpi, which the context's DPI scale 1/inv turns into k = (96/dpi)·(1/inv)
        // (0.99999994 at 120); HwBitmapBrush folds the bitmap size in and inverts.
        if let Some(b) = self.image_brush_matrix(offset, composite) {
            let dpi = [self.dpi.0, self.dpi.1];
            let k: [f32; 2] = core::array::from_fn(|c| (96.0 / dpi[c]) * (1.0 / [inv.0, inv.1][c]));
            return hw_bitmap_row(self.texture_dims(bitmap).map(|v| v as f32), b, [k[0], 0.0, 0.0, k[1], -o[0], -o[1]]);
        }
        image_texmap(self.state.transform, pd, inv, self.texture_dims(bitmap).map(|v| v as f32), o)
    }

    /// The texture a DrawImage source lives in: its own size, or an effect output's atlas (power of two, at least 512).
    fn texture_dims(&self, bitmap: &Bitmap) -> [u32; 2] {
        let dims = [bitmap.width, bitmap.height];
        if self.effect_atlas { dims.map(|v| v.next_power_of_two().max(512)) } else { dims }
    }

    /// Off dpi 96, DrawImage's device brush matrix: the DIP `offset` under the device transform, op for
    /// op as native's DrawImage forms it, whose DPI scale is dpi/96 on the direct
    /// composites (SOURCE_OVER, PLUS, BOUNDED_SOURCE_COPY) but 1/(96·(1/dpi)) on the slow path
    /// (slow-path command list: 1.2499999 at dpi 120). `None` at dpi 96.
    fn image_brush_matrix(&self, offset: [f32; 2], composite: u32) -> Option<[f32; 6]> {
        let (inv, _) = self.image_dip(offset, [0.0; 2]);
        if inv == (1.0, 1.0) { return None; }
        let slow = !matches!(composite, 0 | 9 | 11);
        let (dpi, w) = ([self.dpi.0, self.dpi.1], self.state.transform);
        let s: [f32; 2] = core::array::from_fn(|c| if slow { 1.0 / [inv.0, inv.1][c] } else { dpi[c] / 96.0 });
        let m = [w[0] * s[0], w[1] * s[1], w[2] * s[0], w[3] * s[1], w[4] * s[0], w[5] * s[1]];
        Some([(m[2] * 0.0) + m[0], (m[3] * 0.0) + m[1], (m[0] * 0.0) + m[2], (m[1] * 0.0) + m[3],
            ((m[2] * offset[1]) + (offset[0] * m[0])) + m[4], ((offset[0] * m[1]) + (offset[1] * m[3])) + m[5]])
    }


    /// deviceTransform = world × DpiScale, with DpiScale = diag(dpiX/96, dpiY/96) in DIPS mode and
    /// identity in PIXELS mode.
    pub fn device_transform(&self) -> Mat {
        // Direct2D scales each element of the world matrix by its axis's DPI factor in f32, so an
        // infinite element never meets the scale's zeros (recap 11891: m11 = -inf keeps m12 finite).
        let w = self.state.transform;
        let (sx, sy) = if self.state.unit == 0 { (self.dpi.0 / 96.0, self.dpi.1 / 96.0) } else { (1.0, 1.0) };
        [w[0] * sx, w[1] * sy, w[2] * sx, w[3] * sy, w[4] * sx, w[5] * sy].map(|v| v as f64)
    }

    /// The record-time state + DPI stamped on every recorded [`RecCmd`].
    pub(crate) fn rec_env(&self) -> (crate::state::DrawState, (f32, f32)) {
        (self.state, self.dpi)
    }

    /// Append a device-independent draw to the open command list, returning whether one was open.
    pub(crate) fn record(&mut self, cmd: RecCmd) -> bool {
        match &self.recording {
            Some(rc) => {
                rc.borrow_mut().cmds.push(cmd);
                true
            }
            None => false,
        }
    }

    /// The pixel size of the bound target.
    pub(crate) fn target_size(&self) -> Option<[usize; 2]> {
        self.target.as_ref().map(|t| [t.desc.width as usize, t.desc.height as usize])
    }

    /// The pixel rect a draw's edge list is clipped to: the target within the pushed clips and layer
    /// bounds ([`Self::draw_clip_px`]).
    pub(crate) fn raster_clip(&self) -> Option<[i32; 4]> {
        let t = self.target.as_ref()?;
        let (w, h) = (t.desc.width as i32, t.desc.height as i32);
        let c = self.draw_clip_px().unwrap_or([0, 0, w, h]);
        Some([c[0].max(0), c[1].max(0), c[2].min(w), c[3].min(h)])
    }

    /// The pushed clips' pixel rect by the pixel-centre rule, antialiased or not: a pixel is in when its centre is
    /// (VM DDI of r25 30509: a CUBIC brush's text intermediates under an antialiased clip at 21.82..39.20 ×
    /// 5.04..24.42 are drawn over columns 22..39 and rows 5..24).
    pub(crate) fn centre_clip_px(&self) -> Option<[i32; 4]> {
        self.stack.iter().fold(None, |acc, f| match f {
            crate::layer::Frame::Clip { rect: r, .. } => {
                let px = r.map(|v| (v - 0.5).ceil() as i32);
                Some(acc.map_or(px, |a: [i32; 4]| [a[0].max(px[0]), a[1].max(px[1]), a[2].min(px[2]), a[3].min(px[3])]))
            }
            _ => acc,
        })
    }

    /// The edge list's clip rect when the current draw is a per-primitive COPY: that draw's fill goes
    /// through the edge list, which leaves a curve off that rect unsubdivided.
    pub(crate) fn copy_target(&self) -> Option<[i32; 4]> {
        self.raster_clip().filter(|_| self.state.blend == 1 && self.state.aa != 1)
    }

    /// Rasterize a device-resolved primitive into the bound bitmap (the direct, non-recording path;
    /// a command list records a device-independent [`RecCmd`] instead).
    fn raster(&mut self, cmd: RasterCmd) {
        let clip = self.raster_clip();
        let ignore_alpha_layer = self.stack.iter().rev().find_map(|f| match f {
            crate::layer::Frame::Layer { params, .. } => Some(params.ignore_alpha),
            _ => None,
        }) == Some(true);
        let Some(buf) = &mut self.target else { return };
        // A float (HDR) target composites in f32 with no 8-bit quantisation of the source and no
        // clamp; the BGRA8/UNORM target keeps the integer pipeline. Coverage/scan-conversion is
        // shared (`pixels::rasterize_*`), so only the composite/store differ (`crate::float`).
        if crate::float::is_highp(buf.desc.format) {
            let mut t = crate::float::FloatTarget::new(buf);
            match cmd {
                RasterCmd::FillRect { rect, src_f, aa, blend, .. } => {
                    t.fill_rect(rect, src_f, aa_of(aa), blend_of(blend))
                }
                RasterCmd::FillPolys { figures, winding, src_f, aa, blend, .. } => t.fill_path(
                    &figures,
                    if winding { FillMode::Winding } else { FillMode::Alternate },
                    src_f,
                    aa_of(aa),
                    blend_of(blend),
                ),
            }
            return;
        }
        // A MIN/ADD/MAX fill passes its unquantised float source so the blend rounds once (see
        // Target::src_f); SOURCE_OVER/COPY keep the pre-quantised [u8;4] path.
        fn min_src(t: Target<'_>, blend: u32, src_f: [f32; 4]) -> Target<'_> {
            match blend_of(blend) {
                Blend::Min => t.with_src_f(src_f),
                // A COPY walk inside an IGNORE_ALPHA layer composites the unquantised source too.
                Blend::Copy => t.with_src_f(src_f),
                _ => t,
            }
        }
        let t = Target::new(buf).with_clip(clip).in_ignore_alpha_layer(ignore_alpha_layer);
        match cmd {
            RasterCmd::FillRect { rect, src, src_f, aa, blend, .. } => min_src(t, blend, src_f).fill_rect(rect, src, aa_of(aa), blend_of(blend)),
            RasterCmd::FillPolys { figures, winding, src, src_f, blend, warp28: true, .. } => min_src(t, blend, src_f).fill_warp28(&figures, winding, src, blend_of(blend)),
            RasterCmd::FillPolys { figures, winding, src, src_f, aa, blend, geometry, .. } => min_src(t, blend, src_f).fill_path(
                &figures,
                if winding { FillMode::Winding } else { FillMode::Alternate },
                src,
                aa_of(aa),
                blend_of(blend),
                geometry,
            ),
        }
    }
}

/// A gradient or bitmap/image brush: a per-pixel premultiplied source ([`Brush`]) realized against
/// the device, plus the device domain for cross-device validation.
#[derive(Clone)]
pub struct PixelBrush {
    pub(crate) domain: Domain,
    pub(crate) brush: Brush,
    /// The device-independent definition, retained so a command-list replay can re-realize the brush
    /// under the replay transform (native realizes brushes at draw time). `None` for a brush with no
    /// re-realizable definition (an image brush / COLR ramp) — such a brush falls back to translation.
    pub(crate) def: Option<crate::brush::BrushDef>,
    /// An effect image brush's definition, realized again per draw ([`DeviceContext::image_brush_for_draw`]).
    pub(crate) image: Option<std::rc::Rc<ImageBrushDef>>,
    /// A cropped bitmap image brush's tile, which each draw realizes into the frame's atlas
    /// ([`DeviceContext::image_brush_for_draw`]); the draw itself samples the creation-time crop.
    pub(crate) crop: Option<ImageCrop>,
    /// The error a draw with this brush latches instead of drawing (an image brush native can't realize).
    pub(crate) fail: Option<HRESULT>,
    /// What a command's copy of the brush takes in the command batch ([`crate::batch`]): 56 bytes for a linear gradient
    /// or a bitmap brush, 64 for a radial gradient, 80 for an image brush (24 for a solid colour).
    pub(crate) batch: u32,
    /// The constant rows a fill with the brush writes in the target's queue ([`crate::batch`]); `None` for an image brush of
    /// an effect, a command list or a bitmap crop, whose fill is an operation of the deferred image context.
    pub(crate) rows: Option<u32>,
}

impl PixelBrush {
    /// What a fill with the brush queues on the target.
    pub(crate) fn queued(&self) -> crate::batch::Prim {
        match self.rows {
            Some(r) => crate::batch::Prim::Brush(self as *const PixelBrush as usize, r),
            None => crate::batch::Prim::Image,
        }
    }
}

/// What [`DeviceContext::image_brush_for_draw`] needs of an effect image brush: the effect, its
/// BorderRect tile (pixel rect), the tile-texel → brush matrix and the brush's modes.
pub(crate) struct ImageBrushDef {
    effect: crate::effects::Effect,
    tile: [i32; 4],
    m: [f32; 6],
    extend: [u32; 2],
    interp: u32,
    opacity: f32,
}

/// What a draw with a cropped bitmap image brush needs to size that brush's realization: its border tile (pixel rect),
/// the tile-texel → brush matrix and the interpolation.
#[derive(Clone)]
pub(crate) struct ImageCrop {
    tile: [i32; 4],
    texels: std::rc::Rc<Bitmap>,
    m: [f32; 6],
    extend: [u32; 2],
    interp: u32,
    opacity: f32,
}

/// The reach and pre-scale of the matrix transform an image brush keeps in its graph, or `None` where d2d1 folds the
/// brush's AffineTransform into the brush itself (NEAREST and LINEAR; CUBIC and MULTI_SAMPLE_LINEAR for a lone brush).
/// ANISOTROPIC keeps it, reaching 1 texel; HIGH_QUALITY_CUBIC other than a pure scale draws through it as CUBIC,
/// 2 texels, behind a Scale effect by its pre-scale factors where an axis minifies.
/// ponytail: a pure-scale HIGH_QUALITY_CUBIC (both above 0.01, no translation: a Scale effect alone) keeps the folded reach.
fn kept_affine(interp: u32, m: &[f32; 6]) -> Option<(f32, [f32; 2])> {
    match interp {
        4 => Some((1.0, [1.0; 2])),
        5 if !(m[0] > 0.01 && m[1] == 0.0 && m[2] == 0.0 && m[3] > 0.01 && m[4] == 0.0 && m[5] == 0.0) => {
            Some((2.0, crate::effects::hq_prescale_factor(*m)))
        }
        _ => None,
    }
}

/// What a border over input rect `inner` asks of its input for output rect `need`, per axis by its extend mode: CLAMP
/// the need clamped into the input; WRAP the need wrapped into one period when it fits there unbroken, else all of it;
/// MIRROR the need folded into the input when its ends fold the same way within one period, from the edge they share to
/// the farther end when they fold apart within two, else all of it (d2d1's border transform's input request).
fn border_request(inner: [i32; 4], need: [i32; 4], extend: [u32; 2]) -> [i32; 4] {
    let axis = |k: usize| -> (i32, i32) {
        let (l, r1, a, b) = (inner[k] as i64, inner[k + 2] as i64 - 1, need[k] as i64, need[k + 2] as i64 - 1);
        let p = r1 - l + 1;
        if p <= 0 || b < a {
            return (inner[k], inner[k + 2]);
        }
        let (lo, hi) = match extend[k] {
            1 => {
                let (ma, mb) = ((a - l).rem_euclid(p) + l, (b - l).rem_euclid(p) + l);
                if b - a + 1 > p || ma > mb { (l, r1) } else { (ma, mb) }
            }
            2 => {
                let fold = |v: i64| -> (i64, bool) {
                    let k = (v - l).div_euclid(p);
                    let m = (v - l).rem_euclid(p);
                    if k & 1 == 1 { (r1 - m, true) } else { (l + m, false) }
                };
                let ((ma, fa), (mb, fb)) = (fold(a), fold(b));
                let span = b - a + 1;
                if fa == fb {
                    if span < p { if fa { (mb, ma) } else { (ma, mb) } } else { (l, r1) }
                } else if span >= 2 * p {
                    (l, r1)
                } else if fa || !fb {
                    (l, ma.max(mb))
                } else {
                    (ma.min(mb), r1)
                }
            }
            _ => (a.min(r1).max(l), b.max(l).min(r1)),
        };
        (lo as i32, (hi + 1) as i32)
    };
    let ((l, r), (t, b)) = (axis(0), axis(1));
    [l, t, r, b]
}

/// The border cells over need region `n`: d2d1 `GenerateNineGrid` when both axes CLAMP (per axis
/// `[n.l, l)`, `[l, r)`, `[r, n.r)`), else `GenerateRepeatingCells` (the tile grid aligned to `tile`),
/// each cell clipped to `n` with its screen rect relative to `n`. The edges' input coordinates follow
/// `CBorderVertexRenderData::GetVertexInput`: a CLAMP vertex past the tile samples the edge texel's
/// centre, WRAP repeats, MIRROR reverses the odd tiles.
fn border_cells(tile: [i32; 4], n: [i32; 4], extend: [u32; 2]) -> Vec<([i32; 4], [f32; 4])> {
    let input = |k: usize, x: i32, vt: i32| {
        let (l, r) = (tile[k], tile[k + 2]);
        let (w, v) = (r - l, x - vt);
        match extend[k] {
            0 if v < l => l as f32 + 0.5,
            0 if v >= r => r as f32 - 0.5,
            0 => x as f32,
            1 => ((v - l).rem_euclid(w) + l) as f32 + vt as f32,
            _ if (v - l).div_euclid(w) & 1 == 1 => (r - 1 - (v - l).rem_euclid(w)) as f32 + (1 - vt) as f32,
            _ => ((v - l).rem_euclid(w) + l) as f32 + vt as f32,
        }
    };
    let axis = |k: usize| {
        let (t0, t1, n0, n1) = (tile[k], tile[k + 2], n[k], n[k + 2]);
        let edges: Vec<i32> = if extend == [0, 0] {
            vec![n0, t0, t1, n1]
        } else {
            let w = t1 - t0;
            let mut d = t0.rem_euclid(w) - n0.rem_euclid(w);
            if d > 0 {
                d -= w;
            }
            (n0 + d..n1).step_by(w as usize).chain([n1.max(n0 + d)]).collect()
        };
        edges.windows(2).map(|e| (e[0].max(n0), e[1].min(n1))).filter(|(a, b)| a < b).collect::<Vec<_>>()
    };
    let (xs, ys) = (axis(0), axis(1));
    let mut cells = Vec::with_capacity(xs.len() * ys.len());
    for &(y0, y1) in &ys {
        for &(x0, x1) in &xs {
            cells.push(([x0 - n[0], y0 - n[1], x1 - n[0], y1 - n[1]], [input(0, x0, 0), input(1, y0, 0), input(0, x1, 1), input(1, y1, 1)]));
        }
    }
    cells
}

/// [`border_cells`] filled from the realized `tile` by texel copies (an effect whose last pass the
/// cells cannot replace).
fn border_fill(t: &Bitmap, tile: [i32; 4], n: [i32; 4], extend: [u32; 2]) -> Bitmap {
    let (w, h) = ((n[2] - n[0]) as usize, (n[3] - n[1]) as usize);
    let at = |v: i32, len: i32, e: u32| match e {
        0 => v.clamp(0, len - 1) as usize,
        _ => {
            let p = v.rem_euclid(if e == 2 { 2 * len } else { len });
            (if p >= len { 2 * len - 1 - p } else { p }) as usize
        }
    };
    let mut px = vec![0u8; w * h * 4];
    if t.width as i32 == tile[2] - tile[0] && t.height as i32 == tile[3] - tile[1] {
        for y in 0..h {
            let sy = at(n[1].saturating_add(y as i32).saturating_sub(tile[1]), t.height as i32, extend[1]);
            for x in 0..w {
                let sx = at(n[0].saturating_add(x as i32).saturating_sub(tile[0]), t.width as i32, extend[0]);
                let i = (sy * t.width as usize + sx) * 4;
                px[(y * w + x) * 4..][..4].copy_from_slice(&t.premul_bgra[i..i + 4]);
            }
        }
    }
    Bitmap { width: w as u32, height: h as u32, premul_bgra: px, format: DXGI_FORMAT_B8G8R8A8_UNORM, alpha: D2D1_ALPHA_MODE_PREMULTIPLIED, dpi: t.dpi }
}

/// Allocate a premultiplied BGRA8 target bitmap of `width`×`height` at `dpi` (what a bitmap render
/// target wraps). Pixels start transparent black.
/// The image an `ID2D1ImageBrush` samples ([`DeviceContext::create_image_brush`]).
/// What [`DeviceContext::transformed_image_need`] finds a transformed infinite image's draw does.
pub(crate) enum ImageNeed {
    Draws,
    Nothing,
    TooLarge,
}

pub enum BrushImage<'a> {
    Bitmap(&'a Bitmap),
    Effect(&'a crate::effects::Effect),
    CommandList(&'a CommandList),
}

pub fn bitmap_target(width: u32, height: u32, dpi: f32) -> PixelBuffer {
    PixelBuffer {
        desc: ImageDesc {
            width,
            height,
            format: DXGI_FORMAT_B8G8R8A8_UNORM,
            alpha_mode: D2D1_ALPHA_MODE_PREMULTIPLIED,
            dpi_x: dpi,
            dpi_y: dpi,
        },
        stride: width as usize * 4,
        data: vec![0; width as usize * height as usize * 4],
    }
}

/// The fast image-draw path's placement test: it rejects a device position when
/// `|RoundSat(v) − v| ≥ 1/32`. So a NaN passes, placed at RoundSat's INT_MIN, and a value past
/// ±2147483600 saturates and fails. At that NaN placement the copy modes still clear the target and
/// every other mode draws nothing (w_bm di2/di3, r11 16512); a ±3e9 offset takes the slow path.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
/// The image rect a draw through `m` (image pixels to device pixels, `[a, b, c, d, tx, ty]`) needs for the device
/// pixel rect `r`: the pixel centres (pixel edges for MULTI_SAMPLE_LINEAR) mapped back through `m`'s adjoint in f32,
/// grown by the filter's reach and saturated to i32 (Direct2D's bounding-box need rect).
/// The share of device pixel `x` that MULTI_SAMPLE_LINEAR reads inside an atlas edge at device `end`: the mean of
/// its sub-samples at x + 0.25 and x + 0.75, each the bilinear weight of the texels below the edge.
fn msl_edge_cover(x: f32, end: f32) -> f32 {
    let w = |p: f32| (end + 0.5 - p).clamp(0.0, 1.0);
    (w(x + 0.25) + w(x + 0.75)) * 0.5
}

fn adjoint_need(m: [f32; 6], r: [i32; 4], interp: u32) -> [i32; 4] {
    let [a, b, c, d, tx, ty] = m;
    // Saturating f32 → i32, NaN to i32::MIN as the conversion instruction gives: nearest with ties up, and ceiling.
    let round_sat = |v: f32| if v.is_nan() { i32::MIN } else { let r = v.round_ties_even(); (if r - v == -0.5 { r + 1.0 } else { r }) as i32 };
    let ceil_sat = |v: f32| if v.is_nan() { i32::MIN + 1 } else { v.ceil() as i32 };
    let det = a * d - b * c;
    let adj = [d, -b, -c, a, c * ty - d * tx, b * tx - a * ty];
    // ponytail: HIGH_QUALITY_CUBIC takes CUBIC's reach; r27 32905's 1e-6 replay fails and the layers lane's 84
    // controls (scales 0.1 to 2, rotations, skews) pass either way, so its reach at the 4096 edge is unprobed.
    let reach = match interp { 0 => 0.53125, 2 | 5 => 2.0, _ => 1.0 };
    let inset = if interp == 3 { 0.0 } else { 0.5 };
    let (mut nx, mut ny) = ([f32::INFINITY, f32::NEG_INFINITY], [f32::INFINITY, f32::NEG_INFINITY]);
    for x in [r[0] as f32 + inset, r[2] as f32 - inset] {
        for y in [r[1] as f32 + inset, r[3] as f32 - inset] {
            let (ix, iy) = ((x * adj[0] + y * adj[2] + adj[4]) / det, (x * adj[1] + y * adj[3] + adj[5]) / det);
            nx = [nx[0].min(ix), nx[1].max(ix)];
            ny = [ny[0].min(iy), ny[1].max(iy)];
        }
    }
    if det * det > 0.0 {
        [round_sat(nx[0] - reach), round_sat(ny[0] - reach), ceil_sat(nx[1] + reach - 0.5), ceil_sat(ny[1] + reach - 0.5)]
    } else {
        [i32::MIN, i32::MIN, i32::MAX, i32::MAX]
    }
}

pub(crate) fn fast_whole(v: f32) -> bool {
    let r = if v < -2147483600.0 { i32::MIN as f32 } else if v >= 2147483600.0 { i32::MAX as f32 } else { v.round() };
    !((r - v).abs() >= 0.03125)
}

#[cfg(test)]
mod aliased_check_tests {
    use super::*;
    use crate::resource::Domain;

    // A FillMesh/DrawSpriteBatch under PER_PRIMITIVE antialias latches WRONG_STATE even when an earlier
    // inf-X FillMesh has nan-poisoned the pipeline — native validates the mode regardless of poison
    // (7000-7999 rerun 7750: poisoned op9, op11 FillMesh under aa=0 still latches @11).
    #[test]
    fn aliased_check_latches_wrong_state_under_poison() {
        let mut dc = DeviceContext::new(Domain { factory_id: 0, device_id: 0 });
        dc.begin_draw();
        dc.mesh_bad = vec![true]; // an earlier inf-X FillMesh faulted the frame
        dc.set_antialias_mode(0); // PER_PRIMITIVE
        assert!(!dc.aliased_draw_check(), "per-primitive FillMesh rejects");
        assert_eq!(dc.current_error(), WRONG_STATE, "and latches WRONG_STATE despite the poison");

        // Under ALIASED the draw proceeds (the poison drops only its pixels) with no error latched.
        let mut dc2 = DeviceContext::new(Domain { factory_id: 0, device_id: 0 });
        dc2.begin_draw();
        dc2.mesh_bad = vec![true];
        dc2.set_antialias_mode(1); // ALIASED
        assert!(dc2.aliased_draw_check(), "the poison does not stop the op");
        assert_eq!(dc2.current_error(), S_OK, "and no WRONG_STATE — aa is aliased");
    }
}

/// Record where one clipped mesh triangle (`[x, y, attr]` device px) reaches WARP's 8-row bands inside
/// the mesh `clip` on an `h`-row target: bands it reaches first are appended to `bands`, and bands where
/// it faults are set in `bad` (sized to the target's bands on the first fault; see
/// [`mesh_fault_bands`]). `paint` receives the pixels its inverted quads still colour. `nan_faults`: a fan with a
/// NaN clip value can fault (any mesh fill but an image brush's).
fn note_mesh_bands(bands: &mut Vec<usize>, bad: &mut Vec<bool>, t: [[f32; 3]; 3], clip: [i64; 4], h: usize, nan_faults: bool, paint: impl FnMut(usize, usize)) {
    let q = cvt_nan_positions(t);
    let p = warp_fixed(q);
    let Some((r0, r1)) = tri_rows(&p, clip) else { return };
    // A back face never reaches setup; a front face's bands are queued top-down.
    if !crate::warp::setup_accepts(q) {
        return;
    }
    for b in (r0 / 8) as usize..=(r1 / 8) as usize {
        if !bands.contains(&b) {
            bands.push(b);
        }
    }
    if t.iter().all(|v| v[0].is_finite() && v[1].is_finite()) {
        return;
    }
    let mut hit = vec![false; h.div_ceil(8)];
    mesh_fault_bands(t, &p, (r0, r1), clip, &mut hit, nan_faults, paint);
    if hit.contains(&true) {
        bad.resize(hit.len(), false);
        bad.iter_mut().zip(hit).for_each(|(b, h)| *b |= h);
    }
}

/// Set the 8-row bands where a clipped mesh triangle with a non-finite position (24.8 positions `p`,
/// pixel rows `r0..=r1` inside `clip`) faults WARP's rasterizer.
///
/// Setup keys the triangle's winding on the wrapped 32-bit area ([`crate::warp::setup_accepts`]), but
/// the edge walk steps the true edges. When the exact winding disagrees
/// ([`crate::warp::exact_front_facing`]) every row's span comes out inverted: it starts at the
/// triangle's right edge and ends at its left. Spans are clamped to the clip columns, and the fill
/// kernel walks each 2-row quad row from `min(start) >> 1` to `(max(end) + 1) >> 1` quads, counting
/// down to an exact zero; a row outside the triangle is `(0x7FFF, 0)`. A quad row whose end comes
/// before its start never reaches zero, runs off the band buffer and faults that band's job. A clipped
/// vertex carrying a NaN clip value faults the same way under a solid, gradient or bitmap brush (VM probes: a
/// 3.4e38-y triangle's fan wipes the frame, r18 23277), but not under an image brush (the same mesh with one
/// draws, r14 19650), nor a sprite's (`nan_faults`). Its colour comes out NaN, so its quad rows that do run
/// paint nothing (r11 16690, r14 19786).
///
/// A quad row whose end lies past its start does run: the kernel colours each of its rows' pixels `x`
/// with `(x − start) < (end − start)` compared unsigned in 32 bits, so an inverted span one pixel short
/// colours its `start` pixel (and, wrapping, all but `end` of the quads), an empty one nothing (VM
/// probes: 214 non-finite meshes pixel-exact). `paint` gets those pixels inside the clip.
fn mesh_fault_bands(t: [[f32; 3]; 3], p: &[[i64; 2]; 3], (r0, r1): (i64, i64), clip: [i64; 4], bad: &mut [bool], nan_faults: bool, mut paint: impl FnMut(usize, usize)) {
    let nan_colour = t.iter().any(|v| v[2].is_nan());
    if (nan_colour && !nan_faults) || crate::warp::exact_front_facing(cvt_nan_positions(t)) {
        return;
    }
    // Per row: (start, end) pixel columns of the inverted span, clamped to the clip.
    let span = |row: i64| -> (i64, i64) {
        if row < r0 || row > r1 {
            return (0x7FFF, 0);
        }
        let yc = (row * 256 + 128) as i128;
        let (mut start, mut end) = (i128::MIN, i128::MAX);
        for k in 0..3 {
            let ([xa, ya], [xb, yb]) = (p[k].map(i128::from), p[(k + 1) % 3].map(i128::from));
            if ya == yb || yc < ya.min(yb) || yc > ya.max(yb) {
                continue;
            }
            // First pixel whose centre is at or right of the edge's x at this row centre.
            let (mut num, mut den) = (xa * (yb - ya) + (xb - xa) * (yc - ya) - 128 * (yb - ya), 256 * (yb - ya));
            if den < 0 {
                (num, den) = (-num, -den);
            }
            let px = num.div_euclid(den) + i128::from(num.rem_euclid(den) != 0);
            start = start.max(px);
            end = end.min(px);
        }
        if start == i128::MIN {
            return (0x7FFF, 0);
        }
        let c = |v: i128| v.clamp(clip[0] as i128, clip[2].max(clip[0]) as i128) as i64;
        (c(start), c(end))
    };
    for qr in r0.div_euclid(2)..=r1.div_euclid(2) {
        let (a, b) = (span(2 * qr), span(2 * qr + 1));
        let (x0, x1) = (a.0.min(b.0) >> 1, (a.1.max(b.1) + 1) >> 1);
        if x1 < x0 {
            if let Some(band) = bad.get_mut((2 * qr / 8) as usize) {
                *band = true;
            }
        } else if x1 > x0 && !nan_colour {
            for (row, (start, end)) in [(2 * qr, a), (2 * qr + 1, b)] {
                for x in (2 * x0).max(clip[0])..(2 * x1).min(clip[2]) {
                    if ((x - start) as u32) < ((end - start) as u32) && row >= clip[1] && row < clip[3] {
                        paint(x as usize, row as usize);
                    }
                }
            }
        }
    }
}

/// A triangle's positions in 24.8 as WARP's setup converts them (`cvtss2si`, round to nearest even):
/// a non-finite or out-of-range coordinate becomes `i32::MIN`.
fn warp_fixed(pos: [[f32; 2]; 3]) -> [[i64; 2]; 3] {
    pos.map(|v| {
        v.map(|c| {
            let s = c * 256.0;
            if s.is_finite() && s >= i32::MIN as f32 && s < i32::MAX as f32 { s.round_ties_even() as i64 } else { i32::MIN as i64 }
        })
    })
}

/// The pixel rows `[r0, r1]` whose centres a triangle's 24.8 positions span (the top vertex's row
/// in, the bottom vertex's out), cut to the clip rows; `None` when empty.
fn tri_rows(p: &[[i64; 2]; 3], clip: [i64; 4]) -> Option<(i64, i64)> {
    let (ymin, ymax) = (p.iter().map(|v| v[1]).min()?, p.iter().map(|v| v[1]).max()?);
    let r0 = (ymin - 128 + 255).div_euclid(256).max(clip[1]);
    let r1 = ((ymax - 128 + 255).div_euclid(256) - 1).min(clip[3] - 1);
    (r0 <= r1 && clip[0] < clip[2]).then_some((r0, r1))
}

/// A clipped mesh vertex's position as WARP's setup sees it: `cvtss2si` turns a NaN coordinate into
/// `i32::MIN` in 24.8, i.e. −2^23 px.
fn cvt_nan_positions(t: [[f32; 3]; 3]) -> [[f32; 2]; 3] {
    t.map(|v| [v[0], v[1]].map(|c| if c.is_nan() { -8388608.0 } else { c }))
}

/// The triangles WARP rasterizes for one `FillMesh` triangle (DIPs) on a `w×h` target, as native
/// gets there: `MeshRealization::AddTriangles` stores it front-facing (`p1`/`p2` swapped unless
/// `(x1−x0)(y2−y1) − (y1−y0)(x2−x1) > 0`), `SendGeometry` transforms it ([`transform_mesh_point`]) and the
/// vertex writer swaps again unless the transform's determinant is positive; then WARP clips it
/// ([`crate::warp::clip_triangle`]): none when a clip distance is NaN, itself when inside, else the fan.
/// Each vertex is `[x, y, 1]`: the constant third attribute comes out NaN where the clip interpolated
/// with a NaN factor. A fan vertex whose clipped x/y came out NaN stays non-finite; such a triangle
/// never draws, and may fault the frame whatever the brush ([`mesh_fault_bands`], r14 19919 radial).
fn mesh_triangles(m: &Mat, tri: &[[f32; 2]; 3], w: f32, h: f32) -> Vec<[[f32; 3]; 3]> {
    let pos = mesh_setup_triangle(m, tri);
    match crate::warp::clip_triangle(pos, [0.0, 0.0, w, h]) {
        crate::warp::Clipped::Rejected => Vec::new(),
        crate::warp::Clipped::Inside => vec![pos],
        crate::warp::Clipped::Fan(f) => f,
    }
}

/// A mesh triangle in device space, its vertices in the order setup receives them.
fn mesh_setup_triangle(m: &Mat, tri: &[[f32; 2]; 3]) -> [[f32; 3]; 3] {
    let [p0, mut p1, mut p2] = *tri;
    // Both tests are `comiss`-based: only a positive value keeps the order, so a NaN swaps too (fuzz r10
    // 15119's NaN cross: native clips (p0, p2, p1)).
    if !((p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0]) > 0.0) {
        core::mem::swap(&mut p1, &mut p2);
    }
    let f: [f32; 6] = m.map(|v| v as f32);
    if !(f[3] * f[0] - f[2] * f[1] > 0.0) {
        core::mem::swap(&mut p1, &mut p2);
    }
    [p0, p1, p2].map(|p| {
        let [x, y] = transform_mesh_point(m, p[0], p[1]);
        [x, y, 1.0]
    })
}

/// The fast-widened border cells as the one shape native fills: rectangle figures, even-odd.
fn rect_cells(rects: Vec<[f32; 4]>) -> crate::geometry::Geometry {
    use crate::geometry::{FillMode, Geometry};
    Geometry::Group { fill: FillMode::Alternate, children: rects.into_iter().map(Geometry::Rect).collect() }
}

/// Native's `DrawRectangle` batches the rect swapped into order — left/right
/// when `left > right`, top/bottom when `top > bottom` (a NaN compares false and stays) — so the
/// target's `DrawRectangle` never sees an inverted rect (fuzz r12 17547, 17579; r13 18216).
fn normalized_rect([l, t, r, b]: [f32; 4]) -> [f32; 4] {
    let (l, r) = if l > r { (r, l) } else { (l, r) };
    let (t, b) = if t > b { (b, t) } else { (t, b) };
    [l, t, r, b]
}

/// Direct2D's tiling of a draw: each side of the pixel span `a..b` halved until it is 16 px or less.
pub(crate) fn halves(a: i32, b: i32, out: &mut Vec<(i32, i32)>) {
    if b - a <= 16 {
        out.push((a, b));
    } else {
        let m = a + ((b - a) >> 1);
        halves(a, m, out);
        halves(m, b, out);
    }
}

/// Whether a transparent source still changes the target under composite mode `c`: every Porter-Duff mode that
/// clears or replaces the destination outside the source (SOURCE_IN, DESTINATION_IN, SOURCE_OUT, DESTINATION_ATOP,
/// SOURCE_COPY, BOUNDED_SOURCE_COPY).
fn transparent_black_affects_output(c: u32) -> bool {
    !matches!(c, 0 | 1 | 5 | 6 | 8 | 9 | 12)
}
