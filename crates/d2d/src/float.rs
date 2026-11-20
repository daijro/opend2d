//! High-precision / HDR render targets: `R16G16B16A16_FLOAT` and `R32G32B32A32_FLOAT` (scRGB,
//! extended range). Reproduces WARP's float pipeline read byte-for-byte from the oracle
//! (`harness/scenes/hdr-01..04`, captures in `conformance/oracle/win11-26100`):
//!
//! * The source colour is premultiplied in **f32 with no 8-bit quantisation** (unlike BGRA8, which
//!   quantises the brush colour first). No sRGB/linear conversion — the raw colour value is stored.
//! * The 8×8 coverage and its **8-bit quantisation** `cova = round_ties_even(samples·255/64)` are
//!   *identical* to BGRA8 (shared from [`crate::pixels`]): confirmed by `hdr-01`'s half-pixel column
//!   (0.15019607… = the 128/255 path) and the `hdr-04` coverage ramp.
//! * Source-over / copy / min / add / max run in **f32 with no clamp** — ADD is extended-range
//!   (`hdr-03`: 0.7+0.6 → 1.3, alpha → 2.0), the float store never saturates to [0,1].
//! * The result is stored RGBA as IEEE half (round-to-nearest-even) or f32.
//!
//! Callers pass **device-space** coordinates and a premultiplied RGBA f32 source; clipping and the
//! transform∘DPI mapping are the context's job, as for the BGRA8 target.

use crate::pixels::{rasterize_edges, rasterize_rect, Aa, Blend, FillMode};
use d2d_types::abi::{
    D2D1_ALPHA_MODE, DXGI_FORMAT, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT,
};
use d2d_types::contracts::{ImageDesc, PixelBuffer};

/// Is this a float target this module renders (vs the BGRA8/R8G8B8A8 `pixels::Target`)?
pub fn is_highp(format: DXGI_FORMAT) -> bool {
    format == DXGI_FORMAT_R16G16B16A16_FLOAT || format == DXGI_FORMAT_R32G32B32A32_FLOAT
}

/// Allocate a float render target (RGBA, `format`/`alpha`). Zero bytes = transparent black (0.0 in
/// both half and f32), matching the implicit clear the harness/context start with.
pub fn float_target(width: u32, height: u32, dpi: f32, format: DXGI_FORMAT, alpha: D2D1_ALPHA_MODE) -> PixelBuffer {
    let bpp = if format == DXGI_FORMAT_R32G32B32A32_FLOAT { 16 } else { 8 };
    let stride = width as usize * bpp;
    PixelBuffer {
        desc: ImageDesc { width, height, format, alpha_mode: alpha, dpi_x: dpi, dpi_y: dpi },
        stride,
        data: vec![0; stride * height as usize],
    }
}

/// Premultiply a straight RGBA colour (f32, 0..1) with a brush opacity, in **f32** (no quantisation),
/// for the float source. Mirrors `SolidColorBrush::premul` but keeps full precision.
pub fn premul_f(color: [f32; 4], opacity: f32) -> [f32; 4] {
    let a = color[3] * opacity;
    [color[0] * a, color[1] * a, color[2] * a, a]
}

/// A float render target wrapping a [`PixelBuffer`] whose format [`is_highp`].
pub struct FloatTarget<'a> {
    buf: &'a mut PixelBuffer,
    f32fmt: bool,
}

impl<'a> FloatTarget<'a> {
    pub fn new(buf: &'a mut PixelBuffer) -> Self {
        let f32fmt = buf.desc.format == DXGI_FORMAT_R32G32B32A32_FLOAT;
        Self { buf, f32fmt }
    }

    fn w(&self) -> usize {
        self.buf.desc.width as usize
    }
    fn h(&self) -> usize {
        self.buf.desc.height as usize
    }
    fn bpp(&self) -> usize {
        if self.f32fmt { 16 } else { 8 }
    }

    /// `Clear`: store premultiplied RGBA `premul` to every pixel. No coverage/blend.
    pub fn clear(&mut self, premul: [f32; 4]) {
        let (w, h) = (self.w(), self.h());
        for y in 0..h {
            for x in 0..w {
                self.store(x, y, premul);
            }
        }
    }

    /// Fill an axis-aligned device rectangle with premultiplied RGBA f32 `src`.
    pub fn fill_rect(&mut self, rect: [f32; 4], src: [f32; 4], aa: Aa, blend: Blend) {
        let (w, h) = (self.w(), self.h());
        rasterize_rect(rect, aa, w, h, |x, y, cov| self.blend_px(x, y, src, cov, blend));
    }

    /// Fill closed device-space figures under `fill` with premultiplied RGBA f32 `src`.
    pub fn fill_path(&mut self, figures: &[Vec<[f32; 2]>], fill: FillMode, src: [f32; 4], aa: Aa, blend: Blend) {
        let Some(figures) = crate::pixels::clip_huge(figures) else { return };
        self.fill_edges(&crate::pixels::figure_edges(&figures), fill, src, aa, blend);
    }

    /// Rasterize flattened device-space line `edges` with premultiplied RGBA f32 `src`.
    pub fn fill_edges(&mut self, edges: &[[f32; 4]], fill: FillMode, src: [f32; 4], aa: Aa, blend: Blend) {
        let (w, h) = (self.w(), self.h());
        rasterize_edges(edges, fill, aa, w, h, |x, y, cov| self.blend_px(x, y, src, cov, blend));
    }

    /// Composite a premultiplied RGBA f32 source rect (`src`, `sw`×`sh`) over the target at device
    /// origin `(dx, dy)` with full-coverage source-over — the float path for `DrawImage` of an effect
    /// output (`crate::effects`). Pixels outside the target are skipped.
    pub fn blit(&mut self, src: &[[f32; 4]], sw: i32, sh: i32, dx: i32, dy: i32) {
        let (w, h) = (self.w() as i32, self.h() as i32);
        for sy in 0..sh {
            for sx in 0..sw {
                let (x, y) = (dx + sx, dy + sy);
                if x >= 0 && y >= 0 && x < w && y < h {
                    self.blend_px(x as usize, y as usize, src[(sy * sw + sx) as usize], 64, Blend::SourceOver);
                }
            }
        }
    }

    /// Composite a target-sized premultiplied float intermediate `inter` onto the target under
    /// `D2D1_COMPOSITE_MODE` `mode`, within the device rect `bounds`: Porter-Duff in f32, `inter`
    /// transparent outside the image, so the unbounded modes (SOURCE_IN, SOURCE_COPY, …) clear there.
    /// SOURCE_OVER is `blit`'s `s + d·(1 − sa)`; MASK_INVERT (12) inverts the destination by the source.
    pub fn composite(&mut self, inter: &PixelBuffer, mode: u32, bounds: [i32; 4]) {
        let (w, h) = (self.w() as i32, self.h() as i32);
        let [x0, y0, x1, y1] = bounds;
        for y in y0.clamp(0, h)..y1.clamp(0, h) {
            for x in x0.clamp(0, w)..x1.clamp(0, w) {
                let (s, d) = (read_pixel(inter, x as usize, y as usize), self.load(x as usize, y as usize));
                let (sa, da) = (s[3], d[3]);
                if mode == 12 {
                    // MASK_INVERT, as the 8-bit path's `MaskInvertShader`: rgb (1 − αs)·d + (1 − d)·s, alpha αd.
                    let out = [(1.0 - sa) * d[0] + (1.0 - d[0]) * s[0], (1.0 - sa) * d[1] + (1.0 - d[1]) * s[1], (1.0 - sa) * d[2] + (1.0 - d[2]) * s[2], da];
                    self.store(x as usize, y as usize, out);
                    continue;
                }
                let (fa, fb) = match mode {
                    0 => (1.0, 1.0 - sa),
                    1 => (1.0 - da, 1.0),
                    2 => (da, 0.0),
                    3 => (0.0, sa),
                    4 => (1.0 - da, 0.0),
                    5 => (0.0, 1.0 - sa),
                    6 => (da, 1.0 - sa),
                    7 => (1.0 - da, sa),
                    8 => (1.0 - da, 1.0 - sa),
                    9 => (1.0, 1.0),
                    10 | 11 => (1.0, 0.0),
                    _ => continue,
                };
                let out = if mode == 0 { core::array::from_fn(|i| s[i] + d[i] * fb) } else { core::array::from_fn(|i| s[i] * fa + d[i] * fb) };
                self.store(x as usize, y as usize, out);
            }
        }
    }

    pub(crate) fn load_px(&self, x: usize, y: usize) -> [f32; 4] {
        self.load(x, y)
    }

    pub(crate) fn store_px(&mut self, x: usize, y: usize, c: [f32; 4]) {
        self.store(x, y, c)
    }

    pub(crate) fn blend_px(&mut self, x: usize, y: usize, src: [f32; 4], cov: u32, blend: Blend) {
        let dst = self.load(x, y);
        self.store(x, y, composite_f(dst, src, cov, blend));
    }

    fn pix(&self, x: usize, y: usize) -> usize {
        y * self.buf.stride + x * self.bpp()
    }

    /// Read the stored premultiplied RGBA at `(x, y)` as f32 (half → f32, or f32 verbatim).
    fn load(&self, x: usize, y: usize) -> [f32; 4] {
        read_pixel(self.buf, x, y)
    }

    /// Store premultiplied RGBA f32 at `(x, y)` (f32 verbatim, or half round-to-nearest-even). No
    /// clamp: the float target holds the extended range.
    fn store(&mut self, x: usize, y: usize, c: [f32; 4]) {
        let (i, f32fmt) = (self.pix(x, y), self.f32fmt);
        let d = &mut self.buf.data[i..i + if f32fmt { 16 } else { 8 }];
        if f32fmt {
            for (c, out) in c.iter().zip(d.chunks_exact_mut(4)) {
                out.copy_from_slice(&c.to_le_bytes());
            }
        } else {
            for (c, out) in c.iter().zip(d.chunks_exact_mut(2)) {
                out.copy_from_slice(&f32_to_f16(*c).to_le_bytes());
            }
        }
    }
}

/// `DrawBitmap` to a float target: sample the source rect, scale by `opacity` (in f32 — oracle
/// `hdr-09`: not the BGRA8 path's 8-bit opacity), and source-over composite in f32. The sampler is
/// shared with the BGRA8 path (`crate::sampling`), but its premultiplied-float sample is used
/// directly instead of being re-quantised to UNORM8, so an 8-bit source gives `byte/255` for nearest
/// and WARP's 8-bit-weight bilinear for linear (`hdr-10`). Channels come back BGRA and are stored
/// RGBA. Fractional-edge coverage is 8-bit (`fbitmap-07`); a float bitmap's own texels keep their
/// range (`fbitmap-06..08`).
pub fn draw_bitmap(
    buf: &mut PixelBuffer,
    bitmap: &crate::bitmap::Bitmap,
    dest: [f32; 4],
    source: [f32; 4],
    opacity: f32,
    interpolation: u32,
) {
    let source = crate::bitmap::norm_source(source, bitmap.width as f32, bitmap.height as f32);
    draw_sampled_bitmap_precision(buf, bitmap, dest, source, opacity, interpolation, 0, Blend::SourceOver, crate::pixels::Aa::PerPrimitive);
}

/// A DrawBitmap at a buffer precision or onto a float target, under the primitive blend. The sampled modes map the
/// source as the 8-bit path does ([`crate::bitmap::draw_bitmap_with`]): per axis the min held at >= 0 and the max at
/// <= the bitmap's size, so a source wholly past an edge keeps its reversed span.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_bitmap_precision(
    buf: &mut PixelBuffer, bitmap: &crate::bitmap::Bitmap, dest: [f32; 4],
    source: [f32; 4], opacity: f32, interpolation: u32, precision: u32, blend: Blend, aa: crate::pixels::Aa,
) {
    let (w, h) = (bitmap.width as f32, bitmap.height as f32);
    let source = if matches!(interpolation, 2..=5) {
        [source[0].min(source[2]).max(0.0), source[1].min(source[3]).max(0.0), source[0].max(source[2]).min(w), source[1].max(source[3]).min(h)]
    } else {
        crate::bitmap::norm_source(source, w, h)
    };
    draw_sampled_bitmap_precision(buf, bitmap, dest, source, opacity, interpolation, precision, blend, aa);
}

/// Shared smooth DrawBitmap sampler, preserving float source and target precision.
pub(crate) fn draw_sampled_bitmap(
    buf: &mut PixelBuffer, bitmap: &crate::bitmap::Bitmap, dest: [f32; 4],
    source: [f32; 4], opacity: f32, interpolation: u32, blend: Blend, aa: crate::pixels::Aa,
) {
    draw_sampled_bitmap_precision(buf, bitmap, dest, source, opacity, interpolation, 0, blend, aa);
}

pub(crate) fn draw_sampled_bitmap_precision(
    buf: &mut PixelBuffer, bitmap: &crate::bitmap::Bitmap, dest: [f32; 4],
    source: [f32; 4], opacity: f32, interpolation: u32, precision: u32, blend: Blend, aa: crate::pixels::Aa,
) {
    let [l, t, r, b] = dest;
    // The COPY walk's size limit is on the unclipped rect's snapped interior, as the rect fill's.
    let full = dest;
    let (iw, ih) = (r.floor() - l.ceil(), b.floor() - t.ceil());
    let big = iw > 0.0 && ih > 0.0 && iw * ih > 47863.0;
    // A reversed source span (DrawBitmap's half-clamp) maps backwards; only an empty one draws nothing.
    if !(r > l && b > t && source[2] != source[0] && source[3] != source[1]) || source.iter().any(|v| v.is_nan()) { return; }
    // Native's DrawBitmap affine: the scale divides in f64 and the translation uses the unrounded scale.
    let (sx, sy) = ((r as f64 - l as f64) / (source[2] as f64 - source[0] as f64), (b as f64 - t as f64) / (source[3] as f64 - source[1] as f64));
    let matrix = [sx as f32, 0.0, 0.0, sy as f32, (l as f64 - source[0] as f64 * sx) as f32, (t as f64 - source[1] as f64 * sy) as f32];
    let dest = [l.max(0.0), t.max(0.0), r.min(buf.desc.width as f32), b.min(buf.desc.height as f32)];
    let [l, t, r, b] = dest;
    if r <= l || b <= t { return; }
    // DrawBitmap's anisotropic and HQ-cubic kernels store the filtered sample as UNORM8, and the opacity
    // then modulates that byte as a byte (r23 28586 28015: a half-level product rounded the other way).
    // CUBIC and MULTI_SAMPLE_LINEAR keep the float product, as does a bitmap or image brush fill. With
    // 16/32-bit-float rendering controls (precision ≥ 3) the realize is that float format too, so aniso/HQ
    // keep the float product like the rest — the fraction survives the opacity multiply and quantizes to
    // UNORM8 only at the coverage ring below (op8cap r24 29094).
    //
    // EXCEPTION — an ANISOTROPIC (mode 4) DrawBitmap/DrawImage modulates the opacity in FLOAT,
    // `round(sample·opacity)`, when EITHER the primitive blend is one of the arithmetic modes MIN / ADD /
    // MAX, OR the unclipped device destination spans ≥ 2^19 pixels on an axis. Both route through native's
    // float output path; a SOURCE_OVER or COPY draw at a normal size keeps the byte `unorm_mul`. It is NOT
    // the magnification depth or the tap count — a reduced 28960 (scale 5.4×3.1, SOURCE_OVER) is byte, and
    // only its full-scene MAX blend makes it float (r27 28960 15→0 via the MAX, 32314 266→0 via its ~1e9-wide
    // dest). VM sweeps: ADD and MAX flip byte→float at every scale, COPY and SOURCE_OVER stay byte; the dest
    // switch is a clean 2^19 device px (65535·8 = 524280 byte, 65536·8 = 524288 float, and it tracks the
    // device span, not the scale — a 4-texel source flips at 2^17, a 32-texel source at 2^14, each at dest
    // 2^19). A geometry/image-brush fill never reaches here.
    let byte_opacity = matches!(interpolation, 4 | 5) && !is_highp(bitmap.format) && !is_highp(buf.desc.format) && precision < 3;
    let aniso_float_opacity = interpolation == 4
        && (matches!(blend, Blend::Min | Blend::Add | Blend::Max)
            || full[2] - full[0] >= 524288.0 || full[3] - full[1] >= 524288.0);
    // ANISOTROPIC at a float buffer precision realizes into the context-format (f16/f32) atlas BEFORE the
    // draw opacity: native rounds the magnified bilinear to the atlas format, then the composite applies
    // opacity in float. So carry opacity OUT of the sample, round the raw sample to the atlas format, then
    // multiply opacity (r24 29094: f16 atlas store resolves the 4-px ±1 tie). Only MAGNIFICATION realizes
    // into that context-format atlas — a MINIFYING aniso samples the mip chain straight to the target with no
    // context-format round, so quantising it there snaps the sample to the 16-bit grid and perturbs the low
    // float bits of a float render target (suite precision-drawbitmap p3/p4 min-aniso: a 16×16→10×10 minify is
    // byte-identical without the round, maxd-254 bit divergence with it). `magnifying` mirrors the brush's own
    // minifying test (set_intermediate_precision): this path is axis-aligned, so both scale axes ≥ 1 ⟺ no mip.
    let magnifying = sx.abs().min(sy.abs()) >= 1.0;
    let aniso_hp = interpolation == 4 && matches!(precision, 3 | 4) && magnifying;
    let o8 = (opacity.clamp(0.0, 1.0) * 255.0).round_ties_even() as i32;
    let mut brush = crate::brush::Brush::bitmap_source(bitmap.clone(), &matrix, [0, 0], interpolation, if byte_opacity || aniso_hp { 1.0 } else { opacity });
    brush.set_intermediate_precision(precision);
    let prepared = brush.prepare_rect(dest);
    let brush = prepared.as_ref().unwrap_or(&brush);
    // Antialiased edges use the shared 8×8 scan converter (eighths per axis, `cov ∈ 0..=64`), the same
    // one the solid/gradient rect fill runs — NOT a box area overlap, which over-covers a sub-eighth
    // sliver (native draws nothing where a sub-sample centre misses the edge: direct cubic DrawBitmap A,
    // left edge x=9 at 9.95 is 0 eighths, so the pixel stays the backdrop). Under ALIASED antialias mode
    // DrawBitmap instead takes WARP's aliased centre test — a dest pixel is full or empty by whether its
    // centre lands in the 24.8-snapped rect, no fringe — for every interpolation mode alike (VM probe
    // palias, interp 0/2/3/4; NEAREST/LINEAR reach it through `blit`, the smooth modes and highp here)
    // (r27 32901 cubic, 32474 MULTI_SAMPLE_LINEAR, each drawn after an explicit SetAntialiasMode(ALIASED)).
    let (bw, bh, stride) = (buf.desc.width as usize, buf.desc.height as usize, buf.stride);
    // Under COPY a CUBIC or MULTI_SAMPLE_LINEAR draw onto an 8-bit target covers its rect by the COPY trapezoid walk, as
    // the bitmap-brush rect fill does, below the big-rect area where the 8x8 scan takes over (VM: the fringe columns of
    // an MSL DrawBitmap at 7/16 and 5/16, not the eighths).
    if blend == Blend::Copy && aa == crate::pixels::Aa::PerPrimitive && matches!(interpolation, 2 | 3) && !big && !is_highp(buf.desc.format) {
        // The vertex shader's texcoord: the device-to-normalized plane at the vertex.
        let np = crate::brush::Brush::plane_of_b2d(&matrix, bitmap.width, bitmap.height);
        let tc = |x: f32, y: f32| [np[0] * x + (np[2] * y + np[4]), np[1] * x + (np[3] * y + np[5])];
        copy_rect_cover(full, bw as i64, bh as i64, tc, |x, y, c8, plane| {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let s = if is_highp(bitmap.format) { brush.source_at_f_rect(px, py, dest) }
                else { brush.source_at_rect_plane(px, py, dest, plane).map(|v| v / 255.0) };
            let s8 = s.map(|v| (v.clamp(0.0, 1.0) * 255.0).round_ties_even() as u8);
            let i = y * stride + x * 4;
            let dst: [u8; 4] = buf.data[i..i + 4].try_into().unwrap();
            buf.data[i..i + 4].copy_from_slice(&crate::pixels::copy_tess::lerp8(dst, s8, c8 as i32));
        });
        return;
    }
    crate::pixels::rasterize_rect(dest, aa, bw, bh, |x, y, cov| {
        let cov8 = (cov.min(64) as f32 * 255.0 / 64.0).round_ties_even();
        let s = if is_highp(bitmap.format) || is_highp(buf.desc.format) || precision >= 3 { let s = brush.source_at_f_rect(x as f32 + 0.5, y as f32 + 0.5, dest); if aniso_hp { crate::effects::quantize_to(precision, s).map(|v| v * opacity.clamp(0.0, 1.0)) } else { s } }
            else if byte_opacity { brush.source_at_rect(x as f32 + 0.5, y as f32 + 0.5, true, dest).map(|v|
                if aniso_float_opacity { (v * opacity.clamp(0.0, 1.0)).round_ties_even() / 255.0 }
                else { crate::layer::unorm_mul(v.round_ties_even() as i32, o8) as f32 / 255.0 }) }
            else { brush.source_at_rect(x as f32 + 0.5, y as f32 + 0.5, true, dest).map(|v| v / 255.0) };
        if is_highp(buf.desc.format) {
            let s = s.map(|v| v * (cov8 / 255.0));
            FloatTarget::new(buf).blend_px(x, y, [s[2], s[1], s[0], s[3]], 64, blend);
        } else {
            // An 8bpc target rounds the sample to 8 bits before the coverage multiply (fbitmap-02).
            let s = s.map(|v| crate::bitmap::unorm_mul((v.clamp(0.0, 1.0) * 255.0).round_ties_even() as i32, cov8 as i32) as f32 / 255.0);
            let i = y * stride + x * 4;
            // The primitive blend (D2D1_PRIMITIVE_BLEND) applies to the sampled modes too, not just
            // NEAREST/LINEAR's blit (r23 28960: a MAX-blended DrawBitmap interp-4 kept the backdrop's
            // channel). SOURCE_OVER keeps the exact WARP 8-bit over formula.
            if blend == Blend::SourceOver {
                crate::bitmap::over_premul_f(buf, i, s, 1.0);
            } else {
                let src: [i32; 4] = s.map(|v| (v.clamp(0.0, 1.0) * 255.0).round_ties_even() as i32);
                let cov_keep = 255 - cov8 as i32; // dst-keep for COPY (1 − coverage)
                for c in 0..4 {
                    let d = buf.data[i + c] as i32;
                    buf.data[i + c] = match blend {
                        Blend::Max => src[c].max(d),
                        Blend::Add => (src[c] + d).min(255),
                        Blend::Min => (src[c] + (255 - src[3])).min(d),
                        Blend::Copy => ((src[c] << 8) + d * cov_keep + (d * cov_keep >> 8) + 128 >> 8).min(255),
                        Blend::SourceOver => d,
                    }.clamp(0, 255) as u8;
                }
            }
        }
    });
}

/// `DrawBitmap` LINEAR under PrimitiveBlend COPY with a float bitmap or target: native fills the dest rect
/// as a bitmap-brush FillRectangle (native's bitmap-draw path) through the COPY
/// tessellation (DDI vertex buffers, `float DrawBitmap` probes). Six triangles: a left ring, the interior
/// and a right ring, each one pixel wide, coverage 0 → 1 → 0, their x edges `1/(1024·h)` left of the
/// half-pixel lines (h = dest height), y edges 1/16 below the rect's. The VS forms the brush texcoords from
/// those raw positions while the plane is set up on the 24.8-snapped ones, so a pixel centre lands just
/// short of its texel: WARP's Q8 address floors on a power-of-two texture dimension (taking 1/256 of the
/// texel to its left) and rounds to nearest otherwise. 8-bit bilinear weights, CLAMP addressing.
/// ponytail: a fractional dest edge's rings are not traced (integer rects, as Paint.NET's tile draw, are exact).
pub(crate) fn draw_bitmap_copy_linear(buf: &mut PixelBuffer, bitmap: &crate::bitmap::Bitmap, dest: [f32; 4], source: [f32; 4], opacity: f32) {
    let [l, t, r, b] = dest;
    if !(r > l && b > t && source[2] > source[0] && source[3] > source[1]) { return; }
    // The brush native's bitmap-draw path forms (f64, stored f32), then the bitmap brush's texture row.
    let (m11, m22) = (((r - l) as f64 / (source[2] - source[0]) as f64), ((b - t) as f64 / (source[3] - source[1]) as f64));
    let brush = [m11 as f32, 0.0, 0.0, m22 as f32, (l as f64 - source[0] as f64 * m11) as f32, (t as f64 - source[1] as f64 * m22) as f32];
    let (bw, bh) = (bitmap.width, bitmap.height);
    let tex = crate::context::hw_bitmap_row([bw as f32, bh as f32], brush, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let shift = 1.0 / (1024.0 * (b - t));
    let (xl, xr) = ((l + 0.5) - shift, (r - 0.5) - shift);
    let (yt, yb) = (t + 0.0625, b + 0.0625);
    let v = |x: f32, y: f32, c: f32| [x, y, (x * tex[0] + y * tex[2]) + tex[4], (x * tex[1] + y * tex[3]) + tex[5], c];
    let (ol, or) = (xl - 1.0, xr + 1.0);
    let tris = [
        [v(ol, yt, 0.0), v(xl, yt, 1.0), v(ol, yb, 0.0)], [v(xl, yt, 1.0), v(xl, yb, 1.0), v(ol, yb, 0.0)],
        [v(xl, yt, 1.0), v(xr, yt, 1.0), v(xl, yb, 1.0)], [v(xr, yt, 1.0), v(xr, yb, 1.0), v(xl, yb, 1.0)],
        [v(xr, yt, 1.0), v(or, yt, 0.0), v(xr, yb, 1.0)], [v(or, yt, 0.0), v(or, yb, 0.0), v(xr, yb, 1.0)],
    ];
    let q8 = |t: f32, n: u32, floor: bool| {
        let a = (t * n as f32 - 0.5) * 256.0;
        let q = if floor { a.floor() } else { a.round_ties_even() } as i32;
        (q >> 8, (q & 0xff) as f32 * (1.0 / 256.0))
    };
    let (fx, fy) = (bw.is_power_of_two(), bw.is_power_of_two() && bh.is_power_of_two());
    let texel = |x: i32, y: i32| bitmap.texel_rgba(x.clamp(0, bw as i32 - 1), y.clamp(0, bh as i32 - 1));
    let (w, h, highp) = (buf.desc.width as i64, buf.desc.height as i64, is_highp(buf.desc.format));
    // WARP's triangle setup, as `warp::fill_triangle`: a triangle crossing the viewport goes through the
    // frustum clip and each fan piece keys on its own first vertex (the rect's bottom edge sits 1/16 below a
    // full tile); one inside keys on its min clip-space vertex, positions through the clip-space round trip.
    let (fw, fh) = (w as f32, h as f32);
    let mut pieces: Vec<([[f32; 5]; 3], [usize; 3])> = Vec::new();
    for tri in tris {
        if tri.iter().any(|q| q[0] < 0.0 || q[0] > fw || q[1] < 0.0 || q[1] > fh) {
            match crate::warp::clip_triangle(tri, [0.0, 0.0, fw, fh]) {
                crate::warp::Clipped::Rejected => {}
                crate::warp::Clipped::Inside => pieces.push((tri, [0, 1, 2])),
                crate::warp::Clipped::Fan(f) => pieces.extend(f.into_iter().map(|t| (t, [0, 1, 2]))),
            }
        } else {
            let o = crate::warp::clip_rotation(tri.map(|q| [q[0], q[1]]), fw, fh);
            pieces.push((tri.map(|q| { let r = crate::warp::viewport_round_trip([q[0], q[1]], fw, fh); [r[0], r[1], q[2], q[3], q[4]] }), o));
        }
    }
    for (tri, o) in pieces {
        let (p, area) = crate::warp::snap_tri(tri.map(|q| [q[0], q[1]]));
        let (sx, sy) = (o.map(|i| p[i][0]), o.map(|i| p[i][1]));
        let [pu, pv, pc] = [2, 3, 4].map(|k| crate::warp::plane::setup_triangle_attr(sx, sy, o.map(|i| tri[i][k])));
        crate::warp::cover_tri(p, area, w, h, |x, y| {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let at = |n: [f32; 3]| px * n[0] + (py * n[1] + n[2]);
            let ((ix, wx), (iy, wy)) = (q8(at(pu), bw, fx), q8(at(pv), bh, fy));
            let (t00, t10, t01, t11) = (texel(ix, iy), texel(ix + 1, iy), texel(ix, iy + 1), texel(ix + 1, iy + 1));
            let s: [f32; 4] = core::array::from_fn(|c| {
                // WARP's float lerp is `a + (b − a)·w` (bit-exact on the random-texel float probes).
                let left = t00[c] + (t01[c] - t00[c]) * wy;
                let right = t10[c] + (t11[c] - t10[c]) * wy;
                (left + (right - left) * wx) * opacity
            });
            let c8 = (at(pc).clamp(0.0, 1.0) * 255.0).round_ties_even();
            if highp {
                let mut ft = FloatTarget::new(buf);
                let d = ft.load(x, y);
                let c = c8 / 255.0;
                ft.store(x, y, core::array::from_fn(|i| s[i] * c + d[i] * (1.0 - c)));
            } else {
                let i = y * buf.stride + x * 4;
                let s8 = [s[2], s[1], s[0], s[3]].map(|v| (v.clamp(0.0, 1.0) * 255.0).round_ties_even() as u8);
                let out = crate::pixels::copy_tess::lerp8(buf.data[i..i + 4].try_into().unwrap(), s8, c8 as i32);
                buf.data[i..i + 4].copy_from_slice(&out);
            }
        });
    }
}

/// Native's COPY tessellation of a rectangle drawn with a sampled bitmap brush (a CUBIC or MULTI_SAMPLE_LINEAR
/// DrawBitmap under COPY), from its vertex buffer:
/// - a fractional top or bottom row as one triangle of constant 8x8 coverage per corner pixel and one over the middle,
///   `(x0, y + 1/2), (x1, y), (x1, y + 1)`;
/// - then the six-triangle rings of [`draw_bitmap_copy_linear`] from `ceil(t) + 1/16` to `floor(b) + 1/16`, their x
///   edges on the 1/16 ceiling of the dest's, less `1/(1024 h)`.
///
/// Each goes through WARP's frustum clip and triangle setup. `tc` is the vertex shader's texcoord at a vertex, taken
/// before the vertex snaps to the 24.8 grid. `put(x, y, c8, plane)` is called for each covered pixel with its 8-bit
/// coverage and the texcoord plane of its triangle (`[du/dx, dv/dx, du/dy, dv/dy, u0, v0]`).
fn copy_rect_cover(dest: [f32; 4], w: i64, h: i64, tc: impl Fn(f32, f32) -> [f32; 2], mut put: impl FnMut(usize, usize, f32, [f32; 6])) {
    let [l, t, r, b] = dest;
    let (fw, fh) = (w as f32, h as f32);
    // A pixel's 8x8 sample count inside the rect along one axis (DDI: 32, 64 and 16 along a 0.22 top row).
    let count = |lo: f32, hi: f32, base: f32| (0..8).filter(|k| { let s = base + (*k as f32 + 0.5) / 8.0; s >= lo && s < hi }).count() as f32;
    let v = |x: f32, y: f32, c: f32| { let [tu, tv] = tc(x, y); [x, y, c, tu, tv] };
    let shift = 1.0 / (1024.0 * (b - t));
    let c16 = |v: f32| (v * 16.0).ceil() / 16.0;
    let (xl, xr) = ((c16(l) + 0.5) - shift, (c16(r) - 0.5) - shift);
    // Edges under a pixel apart form no trapezoid band. A dest over more than one pixel column is the walk's
    // complex scan, row by row.
    let (x0, x1) = (l.floor(), r.ceil());
    if xr < xl && x1 - x0 > 1.0 {
        let poly = [[l, t], [r, t], [r, b], [l, b]];
        crate::pixels::copy_tess::fill_copy_in(&[&poly], true, w as usize, h as usize, true, [0, 0, w as i32, h as i32], |x, y, cov, tri| {
            let (p, _) = crate::warp::snap_tri(tri);
            let (sx, sy) = (p.map(|q| q[0]), p.map(|q| q[1]));
            let at = tri.map(|q| tc(q[0], q[1]));
            let [pu, pv] = [0, 1].map(|k| crate::warp::plane::setup_triangle_attr(sx, sy, at.map(|a| a[k])));
            put(x, y, cov.c8() as f32, [pu[0], pv[0], pu[1], pv[1], pu[2], pv[2]]);
        });
        return;
    }
    let mut tris: Vec<[[f32; 5]; 3]> = Vec::new();
    let mut row = |y: f32| {
        let cy = count(t, b, y);
        let (x0, x1) = (l.floor(), r.ceil());
        let mut cut = vec![x0];
        cut.extend([x0 + 1.0, x1 - 1.0].into_iter().filter(|&c| c > x0 && c < x1));
        cut.dedup();
        cut.push(x1);
        for k in cut.windows(2) {
            let c = count(l, r, k[0]) * cy / 64.0;
            if c > 0.0 { tris.push([v(k[0], y + 0.5, c), v(k[1], y, c), v(k[1], y + 1.0, c)]); }
        }
    };
    if t.fract() != 0.0 { row(t.floor()); }
    if b.fract() != 0.0 && b.floor() > t.floor() { row(b.floor()); }
    let (yt, yb) = (t.ceil() + 0.0625, b.floor() + 0.0625);
    if xr < xl && t.ceil() < b.floor() {
        // Within one pixel column the walk draws the whole rows as one vertical strip at the column's 8x8 count
        // (DDI: a 0.7-wide dest's middle is two triangles at 40/64).
        let (y0, y1, c) = (t.ceil(), b.floor(), count(l, r, x0) / 8.0);
        tris.extend([[v(x0, y0, c), v(x1, y0, c), v(x0, y1, c)], [v(x1, y0, c), v(x1, y1, c), v(x0, y1, c)]]);
    } else if yb > yt {
        let (ol, or) = (xl - 1.0, xr + 1.0);
        tris.extend([
            [v(ol, yt, 0.0), v(xl, yt, 1.0), v(ol, yb, 0.0)], [v(xl, yt, 1.0), v(xl, yb, 1.0), v(ol, yb, 0.0)],
            [v(xl, yt, 1.0), v(xr, yt, 1.0), v(xl, yb, 1.0)], [v(xr, yt, 1.0), v(xr, yb, 1.0), v(xl, yb, 1.0)],
            [v(xr, yt, 1.0), v(or, yt, 0.0), v(xr, yb, 1.0)], [v(or, yt, 0.0), v(or, yb, 0.0), v(xr, yb, 1.0)],
        ]);
    }
    let mut pieces: Vec<([[f32; 5]; 3], [usize; 3])> = Vec::new();
    for tri in tris {
        if tri.iter().any(|q| q[0] < 0.0 || q[0] > fw || q[1] < 0.0 || q[1] > fh) {
            match crate::warp::clip_triangle(tri, [0.0, 0.0, fw, fh]) {
                crate::warp::Clipped::Rejected => {}
                crate::warp::Clipped::Inside => pieces.push((tri, [0, 1, 2])),
                crate::warp::Clipped::Fan(f) => pieces.extend(f.into_iter().map(|t| (t, [0, 1, 2]))),
            }
        } else {
            let o = crate::warp::clip_rotation(tri.map(|q| [q[0], q[1]]), fw, fh);
            pieces.push((tri.map(|q| { let r = crate::warp::viewport_round_trip([q[0], q[1]], fw, fh); [r[0], r[1], q[2], q[3], q[4]] }), o));
        }
    }
    for (tri, o) in pieces {
        let (p, area) = crate::warp::snap_tri(tri.map(|q| [q[0], q[1]]));
        let (sx, sy) = (o.map(|i| p[i][0]), o.map(|i| p[i][1]));
        let [pc, pu, pv] = [2, 3, 4].map(|k| crate::warp::plane::setup_triangle_attr(sx, sy, o.map(|i| tri[i][k])));
        let plane = [pu[0], pv[0], pu[1], pv[1], pu[2], pv[2]];
        crate::warp::cover_tri(p, area, w, h, |x, y| {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            put(x, y, ((px * pc[0] + (py * pc[1] + pc[2])).clamp(0.0, 1.0) * 255.0).round_ties_even(), plane);
        });
    }
}

/// Read a premultiplied RGBA f32 pixel from a float [`PixelBuffer`] (half→f32, or f32 verbatim) — the
/// working offscreen's pixels when compositing a float layer/clip back. Slot order is RGBA (the float
/// target stores RGBA, not the 8bpc path's BGRA), so a caller compositing onto an 8bpc BGRA target
/// swaps `[r[2], r[1], r[0], r[3]]`.
pub(crate) fn read_pixel(buf: &PixelBuffer, x: usize, y: usize) -> [f32; 4] {
    let f32fmt = buf.desc.format == DXGI_FORMAT_R32G32B32A32_FLOAT;
    let bpp = if f32fmt { 16 } else { 8 };
    let d = &buf.data[y * buf.stride + x * bpp..][..bpp];
    if f32fmt {
        core::array::from_fn(|c| f32::from_le_bytes(d[c * 4..c * 4 + 4].try_into().unwrap()))
    } else {
        core::array::from_fn(|c| f16_to_f32(u16::from_le_bytes(d[c * 2..c * 2 + 2].try_into().unwrap())))
    }
}

/// Snap a device coordinate to the aliased-clip grid: round half **down** = `ceil(x − 0.5)` (the same
/// scissor rule as `crate::layer::clip_back`).
fn snap(x: f32) -> i32 {
    (x - 0.5).ceil() as i32
}

/// Analytic per-axis fractional overlap (0..=1) of the device rect `rect` with pixel `(px,py)` —
/// the float content-bounds / clip-edge coverage, kept in f32 (no 8-bit quantisation).
fn overlap(rect: [f32; 4], px: i32, py: i32) -> f32 {
    let [l, t, r, b] = rect;
    let (x, y) = (px as f32, py as f32);
    let cx = (r.min(x + 1.0) - l.max(x)).clamp(0.0, 1.0);
    let cy = (b.min(y + 1.0) - t.max(y)).clamp(0.0, 1.0);
    cx * cy
}

/// `PopLayer` composite-back onto a **float** backdrop (`hdr-12`). The working offscreen is the same
/// float format; load each premultiplied pixel in f32, modulate it by the layer factors — opacity ×
/// content-bounds coverage × geometric mask × opacity-brush alpha — then source-over in f32 and
/// truncate-store. The factors are **f32** multiplies (oracle: opacity 0.5 exact, not the 8-bit path's
/// 128/255), applied one at a time as native binds them as distinct constants/textures.
///
/// ponytail: a *fractional* content-bounds edge or an opacity mask on a float target have no oracle
/// yet — the bounds coverage is the analytic per-axis overlap and the mask/brush bytes are taken as
/// `byte/255`; the integer-aligned bounds every captured float layer uses are exact.
pub(crate) fn layer_back_f(saved: &mut PixelBuffer, working: &PixelBuffer, p: &crate::layer::LayerParams) {
    let (w, h) = (saved.desc.width as i32, saved.desc.height as i32);
    let [l, t, r, b] = p.bounds;
    let (x0, x1) = ((l.floor() as i32).clamp(0, w), (r.ceil() as i32).clamp(0, w));
    let (y0, y1) = ((t.floor() as i32).clamp(0, h), (b.ceil() as i32).clamp(0, h));
    let mut dst = FloatTarget::new(saved);
    // Mirror layer_back's degenerate-clip clear (see layer/mod.rs) on a float/HDR target so the 8-bit
    // and float builds stay byte-identical: a non-finite-overflow layer bound hard-clears the clamped
    // rect to transparent. Dormant for every captured float scene (none use degenerate bounds).
    // Mirror layer_back's asymmetric degenerate-clip trigger (see layer/mod.rs).
    let huge_v = |x: f32| !(x * 256.0).is_finite();
    // A NaN in any raw bound bails native's degenerate-clip path (see layer/mod.rs).
    if !p.bounds_nan && (!l.is_finite() || (huge_v(t) ^ huge_v(b))) && r > l && b > t {
        for py in y0..y1 {
            for px in x0..x1 {
                dst.store(px as usize, py as usize, [0.0; 4]);
            }
        }
        return;
    }
    for py in y0..y1 {
        for px in x0..x1 {
            let pix = (py * w + px) as usize;
            // The transformed-quad mask coverage for a non-axis-aligned device transform, else the
            // analytic axis-aligned rect coverage (mirrors layer_back).
            let bc = match &p.bounds_mask {
                Some(m) => m[pix] as f32 / 255.0,
                None => overlap(p.bounds, px, py),
            };
            if bc <= 0.0 {
                continue;
            }
            let mut src = read_pixel(working, px as usize, py as usize);
            if p.ignore_alpha {
                src[3] = 1.0;
            }
            for f in [p.opacity, bc]
                .into_iter()
                .chain(p.mask.as_ref().map(|m| m[pix] as f32 / 255.0))
                .chain(p.brush_alpha.as_ref().map(|a| a[pix] as f32 / 255.0))
            {
                src = src.map(|c| c * f);
            }
            // Match layer_back: only a full-coverage COPY layer replaces the backdrop; else source-over.
            let b = if p.blend == Blend::Copy && bc >= 1.0 { Blend::Copy } else { Blend::SourceOver };
            dst.blend_px(px as usize, py as usize, src, 64, b);
        }
    }
}

/// `PopAxisAlignedClip` composite-back onto a **float** backdrop. Aliased: a hard scissor copying the
/// working pixels back inside the round-half-down integer rect. Antialiased: reveal the working result
/// through the analytic per-axis edge coverage in f32, `out = working·c + saved·(1−c)`.
/// ponytail: the AA-clip float coverage has no oracle yet (every captured float clip is aliased); the
/// aliased hard-copy is exact.
pub(crate) fn clip_back_f(saved: &mut PixelBuffer, working: &PixelBuffer, rect: [f32; 4], aliased: bool, corner: Option<(usize, usize)>) {
    let (w, h) = (saved.desc.width as i32, saved.desc.height as i32);
    if let Some((x, y)) = corner {
        let v = read_pixel(working, x, y);
        FloatTarget::new(saved).store(x, y, v);
        return;
    }
    let [l, t, r, b] = rect;
    let (x0, x1, y0, y1) = if aliased {
        (snap(l).clamp(0, w), snap(r).clamp(0, w), snap(t).clamp(0, h), snap(b).clamp(0, h))
    } else {
        ((l.floor() as i32).clamp(0, w), (r.ceil() as i32).clamp(0, w), (t.floor() as i32).clamp(0, h), (b.ceil() as i32).clamp(0, h))
    };
    let mut dst = FloatTarget::new(saved);
    for py in y0..y1 {
        for px in x0..x1 {
            let (xu, yu) = (px as usize, py as usize);
            let wv = read_pixel(working, xu, yu);
            if aliased {
                dst.store(xu, yu, wv);
            } else {
                let c = overlap(rect, px, py);
                let sv = dst.load(xu, yu);
                dst.store(xu, yu, core::array::from_fn(|k| wv[k] * c + sv[k] * (1.0 - c)));
            }
        }
    }
}

/// Fill a device-space rectangle with a gradient/bitmap `brush` on a **float** target (`hdr-11`): the
/// brush's premultiplied per-pixel source (BGRA float 0..1, from `brush::Brush::source_at_f` — the
/// gradient sampled with full-precision f32 bilinear) is composited in f32 under `blend` and
/// truncate-stored. Pixel-aligned rectangle, full coverage (`composite_f` already covers every
/// primitive-blend mode) — the float gradient scene's case (mirrors `brush::fill_rect`'s bounds).
pub(crate) fn fill_rect_brush_f(buf: &mut PixelBuffer, rect: [f32; 4], brush: &crate::brush::Brush, blend: Blend) {
    let rect = [rect[0].max(0.0), rect[1].max(0.0), rect[2].min(buf.desc.width as f32), rect[3].min(buf.desc.height as f32)];
    if rect[2] <= rect[0] || rect[3] <= rect[1] { return; }
    if let Some(tiles) = brush.split_rect(rect) {
        for tile in tiles { fill_rect_brush_f(buf, tile, brush, blend); }
        return;
    }
    let prepared = brush.prepare_rect(rect);
    let brush = prepared.as_ref().unwrap_or(brush);
    let (w, h) = (buf.desc.width as usize, buf.desc.height as usize);
    let x0 = (rect[0].round().max(0.0) as usize).min(w);
    let y0 = (rect[1].round().max(0.0) as usize).min(h);
    let x1 = (rect[2].round().max(0.0) as usize).min(w);
    let y1 = (rect[3].round().max(0.0) as usize).min(h);
    let mut t = FloatTarget::new(buf);
    for y in y0..y1 {
        for x in x0..x1 {
            let s = brush.source_at_f_rect(x as f32 + 0.5, y as f32 + 0.5, rect); // premultiplied BGRA 0..1
            t.blend_px(x, y, [s[2], s[1], s[0], s[3]], 64, blend);
        }
    }
}

/// Composite premultiplied RGBA f32 `src` at coverage `cov∈0..=64` over `dst` under `blend`, in f32
/// with no clamp. Coverage is quantised to 8 bits exactly as the BGRA8 path (oracle-confirmed).
fn composite_f(dst: [f32; 4], src: [f32; 4], cov: u32, blend: Blend) -> [f32; 4] {
    let cova = (cov.min(64) as f32 * 255.0 / 64.0).round_ties_even();
    let c = cova / 255.0;
    let cs = src.map(|s| s * c);
    match blend {
        Blend::SourceOver => {
            let sa = cs[3];
            core::array::from_fn(|i| cs[i] + dst[i] * (1.0 - sa))
        }
        Blend::Copy => core::array::from_fn(|i| cs[i] + dst[i] * (1.0 - c)),
        // MIN takes the source over WHITE first: `min(dst, src + (1 − src.a))` (native probe grid).
        Blend::Min => core::array::from_fn(|i| (cs[i] + (1.0 - cs[3])).min(dst[i])),
        Blend::Add => core::array::from_fn(|i| cs[i] + dst[i]),
        Blend::Max => core::array::from_fn(|i| cs[i].max(dst[i])),
    }
}

/// IEEE 754 binary16 → f32 (exact; half is a strict subset of f32).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // subnormal: normalise into f32's exponent range
            let mut e = 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | (((e + (127 - 15)) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | (((exp + (127 - 15)) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// f32 → IEEE 754 binary16, **round toward zero** (truncation). This is WARP's half store, not the
/// usual round-to-nearest-even: confirmed byte-for-byte against the oracle (`hdr-01` fillA blue
/// 0.39988→0.399658, the clear of 0.3→0.2998, every blend in `hdr-03`). The low mantissa bits are
/// dropped with no rounding increment; a finite overflow saturates to the largest half (truncation
/// never rounds up to inf); NaN is preserved; subnormals are shifted and truncated.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;
    if exp == 0xff {
        // inf → inf; nan → a quiet NaN (non-zero mantissa)
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7bff; // finite overflow truncates to the largest half (not inf)
    }
    if e <= 0 {
        if e < -10 {
            return sign; // too small → ±0
        }
        // subnormal half: shift the implicit-1 mantissa down and truncate
        let m = mant | 0x80_0000;
        return sign | (m >> (14 - e) as u32) as u16;
    }
    sign | (((e as u32) << 10) | (mant >> 13)) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_truncation_matches_warp() {
        // Exact halves survive the round trip.
        for v in [0.0f32, 1.0, 0.5, 0.25, -2.0, 2048.0] {
            assert_eq!(f16_to_f32(f32_to_f16(v)), v, "exact {v}");
        }
        // WARP truncates toward zero (not round-to-nearest). Oracle-observed stores from hdr-01/03:
        assert_eq!(f16_to_f32(f32_to_f16(0.3)), 0.2998046875); // round-nearest would be 0.30004883
        assert_eq!(f16_to_f32(f32_to_f16(0.399882813)), 0.399658203125); // fillA blue
        assert_eq!(f16_to_f32(f32_to_f16(0.1)), 0.0999755859375);
        // Truncation never rounds the magnitude up: 0.9999 stays below 1.0.
        assert!(f16_to_f32(f32_to_f16(0.99999)) < 1.0);
        // Negative truncates toward zero too.
        assert_eq!(f16_to_f32(f32_to_f16(-0.399882813)), -0.399658203125);
        // Finite overflow saturates to the largest half, not inf.
        assert_eq!(f32_to_f16(70000.0), 0x7bff);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f16_to_f32(1), f32::from_bits(103 << 23)); // smallest half subnormal: 2^-24
        assert_eq!(f16_to_f32(0x3ff), 1023.0 / 16_777_216.0);
        assert_eq!(f16_to_f32(0x8001), -f32::from_bits(103 << 23));
    }
}
