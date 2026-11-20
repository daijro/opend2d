//! GDI Enhanced-Metafile (EMF) parsing + playback for `ID2D1GdiMetafile`.
//!
//! Scope / ceiling. Native `d2d1.dll` does **not** interpret EMF/EMF+ itself: `D2DGdiMetafile::Create`
//! hands the stream to `CreateMetafileRenderer` from the `ext-ms-win-direct2d-desktop-l1-1-0` apiset
//! (from the native build for this task — there is no record switch, and no gdiplus/`PlayEnhMetaFile`
//! imports), and `DrawGdiMetafile` only records a deferred command the external renderer replays
//! through GDI at batch execution. That renderer is outside the reference material and its pixels are
//! only observable on the WARP VM, so DrawGdiMetafile **pixel parity is out of reach here**.
//!
//! What this module reproduces and that *is* deterministic: the EMF header geometry the
//! `ID2D1GdiMetafile`/`ID2D1GdiMetafile1` accessors report (GetBounds / GetDpi / GetSourceBounds) and
//! the record enumeration `Stream` feeds an `ID2D1GdiMetafileSink`. Plus a best-effort playback of the
//! common filled-primitive records so `DrawGdiMetafile` is not a no-op.
//!
//! ponytail: minimal record set (filled rectangle / ellipse / polygon, current brush colour, default
//! MM_TEXT mapping). The full GDI record coverage, pen strokes, mapping modes and world transforms
//! live in the external renderer and are not pixel-verifiable from the material here; add them only if
//! a VM pixel oracle for metafiles is ever wired.

use crate::context::DeviceContext;
use crate::geometry::Geometry;
use std::collections::HashMap;

fn i32_at(b: &[u8], o: usize) -> i32 {
    i32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

const EMR_HEADER: u32 = 1;
const ENHMETA_SIGNATURE: u32 = 0x464D_4520; // ' EMF'

// EMF record types this playback maps (all others are enumerated by `records`/`Stream` but skipped).
const EMR_POLYGON: u32 = 3;
const EMR_SELECTOBJECT: u32 = 37;
const EMR_ELLIPSE: u32 = 42;
const EMR_RECTANGLE: u32 = 43;
const EMR_CREATEBRUSHINDIRECT: u32 = 39;
const EMR_POLYGON16: u32 = 86;

/// A parsed EMF: the raw bytes plus the header geometry the `ID2D1GdiMetafile` accessors report.
pub struct GdiMetafile {
    pub bytes: Vec<u8>,
    frame: [i32; 4],   // rclFrame — the picture frame, in 0.01 mm
    bounds: [i32; 4],  // rclBounds — inclusive content bounds, in device units
    device_px: [i32; 2],
    device_mm: [i32; 2],
}

impl GdiMetafile {
    /// Parse an EMF byte stream, or `None` when it is not a valid EMF (`EMR_HEADER` + `' EMF'`).
    pub fn parse(bytes: Vec<u8>) -> Option<Self> {
        if bytes.len() < 88 || u32_at(&bytes, 0) != EMR_HEADER || u32_at(&bytes, 40) != ENHMETA_SIGNATURE {
            return None;
        }
        let rd4 = |o: usize| [i32_at(&bytes, o), i32_at(&bytes, o + 4), i32_at(&bytes, o + 8), i32_at(&bytes, o + 12)];
        Some(Self {
            frame: rd4(24),
            bounds: rd4(8),
            device_px: [i32_at(&bytes, 72), i32_at(&bytes, 76)],
            device_mm: [i32_at(&bytes, 80), i32_at(&bytes, 84)],
            bytes,
        })
    }

    /// Reference device resolution (DPI). Ported 1:1 from gdiplus `GetEmfHeader`: `(szlDevice /
    /// szlMillimeters) × 25.4`, entirely in f32 (native uses `divss`/`mulss` with `25.4f`).
    ///
    /// ponytail: ≤2-ULP residual vs native's reported DPI. Native's GdipGetImageHorizontalResolution
    /// returns a value 1-2 ULP below both `(dev/mm)·25.4f` and `(dev·25.4f)/mm` (rt-07: 1024/271 →
    /// native 95.97637176 vs 95.97638702 here) — GDI+ derives the metafile resolution through its own
    /// reference-DC/reciprocal path, not reproducible from the header fields by arithmetic. The algorithm
    /// (and the downstream bounds) are otherwise exact; this last-bit base propagates to bounds/source.
    pub fn dpi(&self) -> (f32, f32) {
        let d = |px: i32, mm: i32| (px as f32 / mm as f32) * 25.4_f32;
        (d(self.device_px[0], self.device_mm[0]), d(self.device_px[1], self.device_mm[1]))
    }

    /// The GDI+ image bounds `(x, y, w, h)` in the metafile's device pixels, ported 1:1 from
    /// `GpMetafile::GetRealBounds` (what `GdipGetImageBounds` returns for a metafile): the **rclFrame**
    /// (0.01 mm, un-normalised) scaled by `dpi/2540`, with native's `+1.0` px on each extent. All f32
    /// (native is `divss`/`mulss`/`addss` throughout, `2540.0f` and `1.0f`).
    fn image_bounds(&self) -> [f32; 4] {
        let (dx, dy) = self.dpi();
        let sx = dx / 2540.0_f32;
        let sy = dy / 2540.0_f32;
        let [fl, ft, fr, fb] = self.frame;
        let x = sx * fl as f32;
        let y = sy * ft as f32;
        let w = (fr - fl) as f32 * sx + 1.0_f32;
        let h = (fb - ft) as f32 * sy + 1.0_f32;
        [x, y, w, h]
    }

    /// GetSourceBounds: the image bounds mapped into 96-DPI DIPs (×96/dpi), un-normalised — native's
    /// `MetafileRenderer::GetBounds` `[left, top, right, bottom]` = `[x·m, y·m, (x+w)·m, (y+h)·m]`.
    pub fn source_bounds(&self) -> [f32; 4] {
        let (dx, dy) = self.dpi();
        let mx = 96.0_f32 / dx;
        let my = 96.0_f32 / dy;
        let [x, y, w, h] = self.image_bounds();
        [x * mx, y * my, (w + x) * mx, (h + y) * my]
    }

    /// GetBounds: the image extent mapped into 96-DPI DIPs at the origin — native's
    /// `ID2D1GdiMetafile::GetBounds`. The size is mapped from the normalised `(0,0,w,h)` (so the height
    /// can differ by a ULP from `source[3] − source[1]`, as native's does).
    pub fn bounds(&self) -> [f32; 4] {
        let (dx, dy) = self.dpi();
        let mx = 96.0_f32 / dx;
        let my = 96.0_f32 / dy;
        let [_, _, w, h] = self.image_bounds();
        [0.0, 0.0, w * mx, h * my]
    }

    /// Enumerate records as `(iType, record_bytes)`, the whole record including its 8-byte
    /// `{iType,nSize}` header. Stops at a malformed size (what a sink would see as the end).
    pub fn records(&self) -> impl Iterator<Item = (u32, &[u8])> {
        let b = &self.bytes;
        let mut off = 0usize;
        std::iter::from_fn(move || {
            if off + 8 > b.len() {
                return None;
            }
            let itype = u32_at(b, off);
            let nsize = u32_at(b, off + 4) as usize;
            if nsize < 8 || off + nsize > b.len() {
                return None;
            }
            let rec = &b[off..off + nsize];
            off += nsize;
            Some((itype, rec))
        })
    }
}

/// Compose two D2D 2×3 affines (`apply(compose(a,b), p) == apply(a, apply(b, p))`), matching
/// `context::compose`.
fn compose6(a: [f32; 6], b: [f32; 6]) -> [f32; 6] {
    [
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
        a[0] * b[4] + a[2] * b[5] + a[4],
        a[1] * b[4] + a[3] * b[5] + a[5],
    ]
}

/// GDI `COLORREF` (0x00BBGGRR) → straight RGBA with opaque alpha.
fn colorref(c: u32) -> [f32; 4] {
    [(c & 0xFF) as f32 / 255.0, ((c >> 8) & 0xFF) as f32 / 255.0, ((c >> 16) & 0xFF) as f32 / 255.0, 1.0]
}

// GDI stock-brush colours selected by `SELECTOBJECT(0x80000000 | index)`.
fn stock_brush(index: u32) -> Option<[f32; 4]> {
    match index {
        0 => Some([1.0, 1.0, 1.0, 1.0]),                   // WHITE_BRUSH
        1 => Some([0.75, 0.75, 0.75, 1.0]),                // LTGRAY_BRUSH
        2 => Some([0.5, 0.5, 0.5, 1.0]),                   // GRAY_BRUSH
        3 => Some([0.25, 0.25, 0.25, 1.0]),                // DKGRAY_BRUSH
        4 => Some([0.0, 0.0, 0.0, 1.0]),                   // BLACK_BRUSH
        _ => None,                                         // NULL_BRUSH (5) and non-brush stock objects
    }
}

impl DeviceContext {
    /// Best-effort playback of a GDI metafile into `dest` (DIPs). See the module ceiling note: this
    /// reproduces the common filled-primitive records (rectangle, ellipse, polygon) with the current
    /// brush colour under the default MM_TEXT mapping, mapping the metafile's device bounds onto
    /// `dest`; other records are skipped. `_src` sub-rect selection is not reproduced.
    pub fn draw_gdi_metafile(&mut self, mf: &GdiMetafile, dest: [f32; 4], src: Option<[f32; 4]>) {
        let _a = self.api(0, 0, None); // ponytail: DrawGdiMetafile's own command is not counted
        // A GDI+ metafile (EMF+ records embedded in EMR_COMMENTs) plays through the EMF+ record
        // interpreter + GDI+-style rasteriser rather than the GDI-EMF path below.
        if !mf.emfplus_stream().is_empty() {
            return self.draw_emfplus(mf, dest, src);
        }
        let b = mf.bounds;
        let (sw, sh) = ((b[2] - b[0]) as f32, (b[3] - b[1]) as f32);
        if sw <= 0.0 || sh <= 0.0 {
            return;
        }
        let (sx, sy) = ((dest[2] - dest[0]) / sw, (dest[3] - dest[1]) / sh);
        // Map logical(device) → dest, applied *before* the DC's current world transform.
        let map = [sx, 0.0, 0.0, sy, dest[0] - b[0] as f32 * sx, dest[1] - b[1] as f32 * sy];
        let saved = self.get_transform();
        self.set_transform(compose6(saved, map));

        let mut objs: HashMap<u32, [f32; 4]> = HashMap::new();
        let mut brush_color: Option<[f32; 4]> = Some([1.0, 1.0, 1.0, 1.0]);
        for (it, rec) in mf.records() {
            match it {
                EMR_CREATEBRUSHINDIRECT if rec.len() >= 20 => {
                    // ihBrush @8; LOGBRUSH32 { lbStyle @12, lbColor @16 }. Style 1 = BS_NULL (no fill).
                    let color = if u32_at(rec, 12) == 1 { None } else { Some(colorref(u32_at(rec, 16))) };
                    if let Some(c) = color {
                        objs.insert(u32_at(rec, 8), c);
                    } else {
                        objs.remove(&u32_at(rec, 8));
                    }
                }
                EMR_SELECTOBJECT if rec.len() >= 12 => {
                    let h = u32_at(rec, 8);
                    brush_color = if h & 0x8000_0000 != 0 { stock_brush(h & 0x7FFF_FFFF) } else { objs.get(&h).copied() };
                }
                EMR_RECTANGLE if rec.len() >= 24 => {
                    if let Some(c) = brush_color {
                        let brush = self.create_solid_color_brush(c, None);
                        let r = [i32_at(rec, 8) as f32, i32_at(rec, 12) as f32, i32_at(rec, 16) as f32, i32_at(rec, 20) as f32];
                        self.fill_rectangle(r, &brush);
                    }
                }
                EMR_ELLIPSE if rec.len() >= 24 => {
                    if let Some(c) = brush_color {
                        let brush = self.create_solid_color_brush(c, None);
                        let (l, t, r, bo) = (i32_at(rec, 8) as f32, i32_at(rec, 12) as f32, i32_at(rec, 16) as f32, i32_at(rec, 20) as f32);
                        let g = Geometry::Ellipse { center: [(l + r) / 2.0, (t + bo) / 2.0], radius: [(r - l) / 2.0, (bo - t) / 2.0] };
                        self.fill_geometry(&g, &brush);
                    }
                }
                EMR_POLYGON | EMR_POLYGON16 if rec.len() >= 28 => {
                    if let Some(c) = brush_color {
                        let n = u32_at(rec, 24) as usize;
                        let fig = read_points(rec, n, it == EMR_POLYGON16);
                        if fig.len() >= 3 {
                            let brush = self.create_solid_color_brush(c, None);
                            self.fill_polygons(&[fig], false, &brush);
                        }
                    }
                }
                _ => {}
            }
        }
        self.set_transform(saved);
    }

    /// Play a GDI+ (EMF+) metafile: interpret its records to draw-intents, then rasterise them the way
    /// GDI+ does during metafile playback — aliased (SmoothingMode None), logical coordinates mapped
    /// natural-size onto `dest`. First increment: solid `FillRects`; curves/stroke follow.
    fn draw_emfplus(&mut self, mf: &GdiMetafile, dest: [f32; 4], _src: Option<[f32; 4]>) {
        use crate::emfplus::{DrawCmd, Player};
        // Natural-size playback maps the metafile's logical pixel space 1:1 onto `dest` (the apiset plays
        // at 96-DPI pixels; GetBounds ≈ the frame in pixels), so the map is just the dest origin offset.
        let map = [1.0, 0.0, 0.0, 1.0, dest[0], dest[1]];
        let saved = self.get_transform();
        let saved_aa = self.get_antialias_mode();
        // GDI+ metafile playback defaults to SmoothingMode None → aliased fills (mode 1).
        self.set_antialias_mode(1);
        for cmd in Player::interpret(mf) {
            match cmd {
                DrawCmd::Clear(c) => {
                    self.set_transform(saved);
                    self.clear(Some(argb(c)));
                }
                DrawCmd::FillRects { brush, rects, xform } => {
                    self.set_transform(compose6(saved, compose6(map, xform)));
                    let b = self.create_solid_color_brush(argb(brush), None);
                    for r in rects {
                        // EMF+ RectF is (x, y, w, h); fill_rectangle takes left/top/right/bottom.
                        self.fill_rectangle([r[0], r[1], r[0] + r[2], r[1] + r[3]], &b);
                    }
                }
                DrawCmd::FillEllipse { brush, rect, xform: _ } => {
                    // Port of GDI+'s aliased FillEllipse: AddEllipse kappa-beziers → cFlatten (×16) under
                    // the device transform (scale 16 + the measured playback half-pixel + dest origin) →
                    // edge-setup DDA → even-odd span fill. Byte-exact vs native on 8
                    // diverse ellipses. Each span is blitted as a 1px axis-aligned rect (FillRects-exact).
                    // ponytail: EMF+ world-xform composition deferred (identity for un-transformed play).
                    use crate::emfplus_raster as raster;
                    // Device transform, derived per-metafile (not a fixed constant): logical→device
                    // scale 16·96/dpi per axis (EMF+ logical 96-dpi normalized to the recorded dpi;
                    // the 96/dpi is what places large ellipses' extreme edges exactly) + PixelOffsetMode
                    // Half (-0.5px) + dest origin.
                    let (sx, sy) = raster::device_scale(mf.dpi());
                    let (w, h) = (dest[2].ceil().max(0.0) as i32, dest[3].ceil().max(0.0) as i32);
                    let tx = raster::HALF_PIXEL_X16 + dest[0] * 16.0;
                    let ty = raster::HALF_PIXEL_X16 + dest[1] * 16.0;
                    let poly = raster::flatten_ellipse(rect, sx, sy, tx, ty);
                    let rows = raster::rasterize(&raster::init_edges(&poly), w, h);
                    self.blit_aliased_spans(&rows, argb(brush));
                }
                DrawCmd::FillPolygon { brush, points, xform: _ } => {
                    // Same aliased pipeline as FillEllipse, but the polygon is the record's points
                    // straight (lines, no bezier flatten): snap each under the device transform, then
                    // edge setup + even-odd span fill. ponytail: world-xform deferred.
                    use crate::emfplus_raster as raster;
                    let (sx, sy) = raster::device_scale(mf.dpi());
                    let (w, h) = (dest[2].ceil().max(0.0) as i32, dest[3].ceil().max(0.0) as i32);
                    let (tx, ty) = (raster::HALF_PIXEL_X16 + dest[0] * 16.0, raster::HALF_PIXEL_X16 + dest[1] * 16.0);
                    let poly: Vec<(i32, i32)> = points
                        .iter()
                        .map(|p| (raster::snap_coord(p[0], sx, tx), raster::snap_coord(p[1], sy, ty)))
                        .collect();
                    let rows = raster::rasterize(&raster::init_edges(&poly), w, h);
                    self.blit_aliased_spans(&rows, argb(brush));
                }
                DrawCmd::DrawLines { pen, points, closed: _, xform: _ } => {
                    // Stroke: pre-transform the polyline to device space (scale 96/dpi, like the fill
                    // coords), widen with GDI+'s pen widener (miter/bevel joins, flat caps), then fill
                    // the outline by nonzero winding (strokes self-overlap) + Half. The offset edges
                    // match native when widened in device space.
                    // ponytail: square/round caps, round joins, dashes and closed polylines are
                    // follow-ups (pen.start_cap/end_cap/dash are parsed, wiring awaits byte-exact caps).
                    use crate::emfplus_raster as raster;
                    if points.len() >= 2 {
                        let (dpix, dpiy) = mf.dpi();
                        let (s96x, s96y) = (96.0f32 / dpix, 96.0f32 / dpiy);
                        let dev: Vec<(f64, f64)> =
                            points.iter().map(|p| ((p[0] * s96x) as f64, (p[1] * s96y) as f64)).collect();
                        let outline = raster::widen_stroke(&dev, pen.width as f64, pen.join);
                        if !outline.is_empty() {
                            let (w, h) = (dest[2].ceil().max(0.0) as i32, dest[3].ceil().max(0.0) as i32);
                            let (tx, ty) = (raster::HALF_PIXEL_X16 + dest[0] * 16.0, raster::HALF_PIXEL_X16 + dest[1] * 16.0);
                            let poly: Vec<(i32, i32)> = outline
                                .iter()
                                .map(|&(x, y)| (raster::snap_coord(x as f32, 16.0, tx), raster::snap_coord(y as f32, 16.0, ty)))
                                .collect();
                            let rows = raster::rasterize_winding(&raster::init_edges(&poly), w, h);
                            self.blit_aliased_spans(&rows, argb(pen.color));
                        }
                    }
                }
            }
        }
        self.set_antialias_mode(saved_aa);
        self.set_transform(saved);
    }

    /// Composite a solid colour over aliased span-rows (absolute device pixels from the GDI+ scan
    /// converter), each contiguous run as a 1px axis-aligned rect (axis-aligned FillRects is byte-
    /// exact). Resets the transform to identity since the spans are already in device space.
    fn blit_aliased_spans(&mut self, rows: &[Vec<i32>], color: [f32; 4]) {
        let b = self.create_solid_color_brush(color, None);
        self.set_transform([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        for (y, cols) in rows.iter().enumerate() {
            let mut i = 0;
            while i < cols.len() {
                let mut j = i;
                while j + 1 < cols.len() && cols[j + 1] == cols[j] + 1 {
                    j += 1;
                }
                self.fill_rectangle([cols[i] as f32, y as f32, (cols[j] + 1) as f32, (y + 1) as f32], &b);
                i = j + 1;
            }
        }
    }
}

/// An EMF+ straight-RGBA colour (`[R,G,B,A]` 0-255) → the `[f32;4]` straight RGBA our brushes take.
fn argb(c: crate::emfplus::Argb) -> [f32; 4] {
    c.0.map(|v| v as f32 / 255.0)
}


/// Read `n` points starting at offset 28 of a polygon record: `POINTL` (i32×2) or `POINTS` (i16×2).
fn read_points(rec: &[u8], n: usize, short: bool) -> Vec<[f32; 2]> {
    let stride = if short { 4 } else { 8 };
    (0..n)
        .filter_map(|i| {
            let o = 28 + i * stride;
            if o + stride > rec.len() {
                return None;
            }
            Some(if short {
                let x = i16::from_le_bytes([rec[o], rec[o + 1]]);
                let y = i16::from_le_bytes([rec[o + 2], rec[o + 3]]);
                [x as f32, y as f32]
            } else {
                [i32_at(rec, o) as f32, i32_at(rec, o + 4) as f32]
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // A hand-built minimal EMF: header (88 bytes) + one EMR_RECTANGLE (24) + EOF (20).
    fn sample_emf() -> Vec<u8> {
        let mut b = vec![0u8; 88];
        let put = |b: &mut [u8], o: usize, v: u32| b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        let puti = |b: &mut [u8], o: usize, v: i32| b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        put(&mut b, 0, EMR_HEADER);
        put(&mut b, 4, 88); // nSize of header
        // rclBounds = 0,0,100,50 device units
        puti(&mut b, 8, 0);
        puti(&mut b, 12, 0);
        puti(&mut b, 16, 100);
        puti(&mut b, 20, 50);
        // rclFrame = 0,0,2540,1270 (0.01 mm) → GDI+ image 97×49 px → 97×49 DIP at 96 dpi (+1-px extent)
        puti(&mut b, 24, 0);
        puti(&mut b, 28, 0);
        puti(&mut b, 32, 2540);
        puti(&mut b, 36, 1270);
        put(&mut b, 40, ENHMETA_SIGNATURE);
        // szlDevice 960×480 px over szlMillimeters 254×127 mm (10×5 in) → exactly 96 dpi both axes
        puti(&mut b, 72, 960);
        puti(&mut b, 76, 480);
        puti(&mut b, 80, 254);
        puti(&mut b, 84, 127);
        // EMR_RECTANGLE
        let mut r = vec![0u8; 24];
        put(&mut r, 0, EMR_RECTANGLE);
        put(&mut r, 4, 24);
        b.extend_from_slice(&r);
        // EMR_EOF (type 14)
        let mut eof = vec![0u8; 20];
        put(&mut eof, 0, 14);
        put(&mut eof, 4, 20);
        b.extend_from_slice(&eof);
        b
    }

    #[test]
    fn parses_header_and_enumerates_records() {
        let mf = GdiMetafile::parse(sample_emf()).expect("valid EMF");
        // GetBounds (gdiplus): rclFrame 0,0,2540,1270 (0.01 mm) at 96 dpi → GDI+ image px
        // frameW·dpi/2540 + 1 = 96+1 = 97 (and 48+1 = 49), ×96/dpi → DIPs, normalised to the origin.
        let b = mf.bounds();
        assert_eq!([b[0], b[1]], [0.0, 0.0]);
        assert!((b[2] - 97.0).abs() < 1e-3 && (b[3] - 49.0).abs() < 1e-3, "bounds={b:?}");
        // dpi = szlDevice / szlMillimeters × 25.4.
        let (dx, dy) = mf.dpi();
        assert!((dx - 96.0).abs() < 1e-3 && (dy - 96.0).abs() < 1e-3);
        // GetSourceBounds is the same extent as GetBounds, positioned at the frame origin (0,0 here).
        let s = mf.source_bounds();
        assert!((s[2] - s[0] - 97.0).abs() < 1e-3 && (s[3] - s[1] - 49.0).abs() < 1e-3, "source={s:?}");
        // header + rectangle + eof = 3 records, in order.
        let types: Vec<u32> = mf.records().map(|(t, _)| t).collect();
        assert_eq!(types, vec![EMR_HEADER, EMR_RECTANGLE, 14]);
    }

    #[test]
    fn rejects_non_emf() {
        assert!(GdiMetafile::parse(vec![0u8; 8]).is_none());
        assert!(GdiMetafile::parse(b"not an emf at all......................................................................".to_vec()).is_none());
    }
}
