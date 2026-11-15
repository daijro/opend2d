use anyhow::{ensure, Context, Result};
use d2d_reference::*;

// Optional instrumentation, off by default (behind the `native-detour` feature).
#[cfg(feature = "native-detour")]
#[path = "ink_flat_hook.rs"]
mod ink_flat_hook;
use std::{cell::RefCell, collections::HashMap, mem::ManuallyDrop, path::Path, rc::Rc};
use windows::{
    core::{implement, s, w, Interface, GUID, HSTRING},
    Foundation::Numerics::Matrix3x2,
    Win32::{
        Foundation::{HMODULE, BOOL},
        Graphics::{
            Direct2D::{Common::*, *},
            Direct3D::*,
            Direct3D11::*,
            DirectWrite::*,
            Dxgi::{Common::*, *},
            Imaging::*,
        },
        System::{Com::*, LibraryLoader::*},
        UI::Shell::SHCreateMemStream,
    },
};

fn point([x, y]: [f32; 2]) -> D2D_POINT_2F {
    D2D_POINT_2F { x, y }
}
fn rect([left, top, right, bottom]: [f32; 4]) -> D2D_RECT_F {
    D2D_RECT_F {
        left,
        top,
        right,
        bottom,
    }
}
fn rect_u([left, top, right, bottom]: [f32; 4]) -> D2D_RECT_U {
    D2D_RECT_U {
        left: left as u32,
        top: top as u32,
        right: right as u32,
        bottom: bottom as u32,
    }
}
fn color([r, g, b, a]: [f32; 4]) -> D2D1_COLOR_F {
    D2D1_COLOR_F { r, g, b, a }
}
fn ink_point([x, y, radius]: [f32; 3]) -> D2D1_INK_POINT {
    D2D1_INK_POINT { x, y, radius }
}
fn matrix(m: [f32; 6]) -> Matrix3x2 {
    Matrix3x2 {
        M11: m[0],
        M12: m[1],
        M21: m[2],
        M22: m[3],
        M31: m[4],
        M32: m[5],
    }
}
fn identity() -> Matrix3x2 {
    matrix([1., 0., 0., 1., 0., 0.])
}
// Phase 2D-2 oracle: an IDWriteTextRenderer that captures every glyph run a layout
// emits instead of drawing it, so native layout/shaping can be compared to the Rust
// engine. Pixel snapping is reported DISABLED so the baseline origins come through as
// the raw fractional layout positions (snapping is a DrawText-time concern).
// IDWriteTextRenderer1 too: DirectWrite refuses a vertical reading direction (TEXTRENDERERINCOMPATIBLE)
// unless the renderer takes orientation angles. A non-zero angle is recorded as `orientation`; an
// unrotated run records exactly what the base methods do.
#[implement(IDWriteTextRenderer1)]
struct GlyphCapture {
    runs: Rc<RefCell<Vec<serde_json::Value>>>,
}

impl IDWritePixelSnapping_Impl for GlyphCapture_Impl {
    fn IsPixelSnappingDisabled(&self, _ctx: *const core::ffi::c_void) -> windows::core::Result<windows::Win32::Foundation::BOOL> {
        Ok(true.into())
    }
    fn GetCurrentTransform(&self, _ctx: *const core::ffi::c_void, transform: *mut DWRITE_MATRIX) -> windows::core::Result<()> {
        unsafe { *transform = DWRITE_MATRIX { m11: 1.0, m12: 0.0, m21: 0.0, m22: 1.0, dx: 0.0, dy: 0.0 } };
        Ok(())
    }
    fn GetPixelsPerDip(&self, _ctx: *const core::ffi::c_void) -> windows::core::Result<f32> {
        Ok(1.0)
    }
}

impl IDWriteTextRenderer_Impl for GlyphCapture_Impl {
    fn DrawGlyphRun(
        &self,
        _ctx: *const core::ffi::c_void,
        x: f32,
        y: f32,
        measuring: DWRITE_MEASURING_MODE,
        run: *const DWRITE_GLYPH_RUN,
        desc: *const DWRITE_GLYPH_RUN_DESCRIPTION,
        _effect: Option<&windows::core::IUnknown>,
    ) -> windows::core::Result<()> {
        unsafe {
            // DirectWrite may pass NULL glyphAdvances/glyphOffsets (both optional) and a NULL run
            // description; slicing a NULL pointer faulted the whole capture (r11: ~30% of seeds).
            fn slice<'a, T>(p: *const T, n: usize) -> &'a [T] {
                if p.is_null() || n == 0 { &[] } else { unsafe { std::slice::from_raw_parts(p, n) } }
            }
            let run = &*run;
            let n = run.glyphCount as usize;
            let ids = slice(run.glyphIndices, n);
            let adv = slice(run.glyphAdvances, n);
            let off = slice(run.glyphOffsets, n);
            let (text, cmap, text_pos) = match desc.as_ref() {
                Some(d) => {
                    let sl = d.stringLength as usize;
                    (String::from_utf16_lossy(slice(d.string.0, sl)), slice(d.clusterMap, sl), d.textPosition)
                }
                None => (String::new(), &[][..], 0),
            };
            // Resolved family of the (possibly fallback-substituted) face this run shaped with.
            let family = run.fontFace.as_ref().and_then(family_name).unwrap_or_default();
            self.runs.borrow_mut().push(serde_json::json!({
                "x": x, "y": y, "measuring": measuring.0,
                "em": run.fontEmSize, "bidi": run.bidiLevel,
                "sideways": run.isSideways.as_bool(),
                "glyphs": ids,
                "advances": adv,
                "offsets": off.iter().map(|o| [o.advanceOffset, o.ascenderOffset]).collect::<Vec<_>>(),
                "clusters": cmap,
                "text_pos": text_pos, "text": text, "font": family,
            }));
        }
        Ok(())
    }
    fn DrawUnderline(&self, _c: *const core::ffi::c_void, x: f32, y: f32, u: *const DWRITE_UNDERLINE, _e: Option<&windows::core::IUnknown>) -> windows::core::Result<()> {
        unsafe {
            let u = &*u;
            self.runs.borrow_mut().push(serde_json::json!({
                "underline": { "x": x, "y": y, "width": u.width, "thickness": u.thickness, "offset": u.offset }
            }));
        }
        Ok(())
    }
    fn DrawStrikethrough(&self, _c: *const core::ffi::c_void, x: f32, y: f32, s: *const DWRITE_STRIKETHROUGH, _e: Option<&windows::core::IUnknown>) -> windows::core::Result<()> {
        unsafe {
            let s = &*s;
            self.runs.borrow_mut().push(serde_json::json!({
                "strikethrough": { "x": x, "y": y, "width": s.width, "thickness": s.thickness, "offset": s.offset }
            }));
        }
        Ok(())
    }
    fn DrawInlineObject(&self, _c: *const core::ffi::c_void, x: f32, y: f32, o: Option<&IDWriteInlineObject>, s: windows::Win32::Foundation::BOOL, r: windows::Win32::Foundation::BOOL, _e: Option<&windows::core::IUnknown>) -> windows::core::Result<()> {
        // The ellipsis trimming sign and any SetInlineObject placeholder funnel here; record the
        // origin DirectWrite placed it at plus the metrics it queried (width distinguishes the
        // ellipsis sign from a fixed-size test object).
        let m = match o {
            Some(o) => unsafe { o.GetMetrics()? },
            None => DWRITE_INLINE_OBJECT_METRICS::default(),
        };
        self.runs.borrow_mut().push(serde_json::json!({
            "inline": { "x": x, "y": y, "sideways": s.as_bool(), "rtl": r.as_bool(),
                        "width": m.width, "height": m.height, "baseline": m.baseline }
        }));
        Ok(())
    }
}

impl GlyphCapture_Impl {
    fn tag_orientation(&self, angle: DWRITE_GLYPH_ORIENTATION_ANGLE) {
        if angle.0 != 0 {
            if let Some(serde_json::Value::Object(m)) = self.runs.borrow_mut().last_mut() {
                m.insert("orientation".into(), serde_json::json!(angle.0));
            }
        }
    }
}

impl IDWriteTextRenderer1_Impl for GlyphCapture_Impl {
    fn DrawGlyphRun(&self, c: *const core::ffi::c_void, x: f32, y: f32, angle: DWRITE_GLYPH_ORIENTATION_ANGLE, m: DWRITE_MEASURING_MODE, run: *const DWRITE_GLYPH_RUN, desc: *const DWRITE_GLYPH_RUN_DESCRIPTION, e: Option<&windows::core::IUnknown>) -> windows::core::Result<()> {
        IDWriteTextRenderer_Impl::DrawGlyphRun(self, c, x, y, m, run, desc, e)?;
        self.tag_orientation(angle);
        Ok(())
    }
    fn DrawUnderline(&self, c: *const core::ffi::c_void, x: f32, y: f32, angle: DWRITE_GLYPH_ORIENTATION_ANGLE, u: *const DWRITE_UNDERLINE, e: Option<&windows::core::IUnknown>) -> windows::core::Result<()> {
        IDWriteTextRenderer_Impl::DrawUnderline(self, c, x, y, u, e)?;
        self.tag_orientation(angle);
        Ok(())
    }
    fn DrawStrikethrough(&self, c: *const core::ffi::c_void, x: f32, y: f32, angle: DWRITE_GLYPH_ORIENTATION_ANGLE, st: *const DWRITE_STRIKETHROUGH, e: Option<&windows::core::IUnknown>) -> windows::core::Result<()> {
        IDWriteTextRenderer_Impl::DrawStrikethrough(self, c, x, y, st, e)?;
        self.tag_orientation(angle);
        Ok(())
    }
    fn DrawInlineObject(&self, c: *const core::ffi::c_void, x: f32, y: f32, angle: DWRITE_GLYPH_ORIENTATION_ANGLE, o: Option<&IDWriteInlineObject>, s: windows::Win32::Foundation::BOOL, r: windows::Win32::Foundation::BOOL, e: Option<&windows::core::IUnknown>) -> windows::core::Result<()> {
        IDWriteTextRenderer_Impl::DrawInlineObject(self, c, x, y, o, s, r, e)?;
        self.tag_orientation(angle);
        Ok(())
    }
}

// A fixed-size inline object (`SetInlineObject` placeholder). Draw is a no-op; only its metrics
// and break conditions affect layout, which is what the pt3 inline-object scenes measure.
#[implement(IDWriteInlineObject)]
struct TestInline {
    size: [f32; 2],
    baseline: f32,
    sideways: bool,
}

impl IDWriteInlineObject_Impl for TestInline_Impl {
    fn Draw(&self, _c: *const core::ffi::c_void, _r: Option<&IDWriteTextRenderer>, _x: f32, _y: f32, _s: windows::Win32::Foundation::BOOL, _rtl: windows::Win32::Foundation::BOOL, _e: Option<&windows::core::IUnknown>) -> windows::core::Result<()> {
        Ok(())
    }
    fn GetMetrics(&self) -> windows::core::Result<DWRITE_INLINE_OBJECT_METRICS> {
        Ok(DWRITE_INLINE_OBJECT_METRICS {
            width: self.size[0],
            height: self.size[1],
            baseline: self.baseline,
            supportsSideways: self.sideways.into(),
        })
    }
    fn GetOverhangMetrics(&self) -> windows::core::Result<DWRITE_OVERHANG_METRICS> {
        Ok(DWRITE_OVERHANG_METRICS::default())
    }
    fn GetBreakConditions(&self, before: *mut DWRITE_BREAK_CONDITION, after: *mut DWRITE_BREAK_CONDITION) -> windows::core::Result<()> {
        unsafe {
            *before = DWRITE_BREAK_CONDITION_NEUTRAL;
            *after = DWRITE_BREAK_CONDITION_NEUTRAL;
        }
        Ok(())
    }
}

// App-provided custom font loaders D2D invokes as callbacks: a font-file loader + stream (serving a
// font's bytes from memory for CreateCustomFontFileReference) and a collection loader + enumerator
// (yielding one path-based file for CreateCustomFontCollection) — the DirectWrite web-font paths.
#[implement(IDWriteFontFileStream)]
struct MemStream {
    data: Vec<u8>,
}
impl IDWriteFontFileStream_Impl for MemStream_Impl {
    fn ReadFileFragment(&self, start: *mut *mut core::ffi::c_void, offset: u64, size: u64, ctx: *mut *mut core::ffi::c_void) -> windows::core::Result<()> {
        if offset.checked_add(size).is_none_or(|end| end > self.data.len() as u64) {
            return Err(windows::Win32::Foundation::E_INVALIDARG.into());
        }
        unsafe {
            *start = self.data.as_ptr().add(offset as usize) as *mut core::ffi::c_void;
            *ctx = std::ptr::null_mut();
        }
        Ok(())
    }
    fn ReleaseFileFragment(&self, _ctx: *mut core::ffi::c_void) {}
    fn GetFileSize(&self) -> windows::core::Result<u64> {
        Ok(self.data.len() as u64)
    }
    fn GetLastWriteTime(&self) -> windows::core::Result<u64> {
        Ok(0)
    }
}
#[implement(IDWriteFontFileLoader)]
struct MemLoader {
    data: Vec<u8>,
}
impl IDWriteFontFileLoader_Impl for MemLoader_Impl {
    fn CreateStreamFromKey(&self, _key: *const core::ffi::c_void, _keysize: u32) -> windows::core::Result<IDWriteFontFileStream> {
        Ok(MemStream { data: self.data.clone() }.into())
    }
}
#[implement(IDWriteFontCollectionLoader)]
struct PathCollLoader {
    path: String,
}
impl IDWriteFontCollectionLoader_Impl for PathCollLoader_Impl {
    fn CreateEnumeratorFromKey(&self, factory: Option<&IDWriteFactory>, _key: *const core::ffi::c_void, _keysize: u32) -> windows::core::Result<IDWriteFontFileEnumerator> {
        Ok(PathEnum { factory: factory.unwrap().clone(), path: self.path.clone(), idx: std::cell::Cell::new(0) }.into())
    }
}
#[implement(IDWriteFontFileEnumerator)]
struct PathEnum {
    factory: IDWriteFactory,
    path: String,
    idx: std::cell::Cell<i32>,
}
impl IDWriteFontFileEnumerator_Impl for PathEnum_Impl {
    fn MoveNext(&self) -> windows::core::Result<windows::Win32::Foundation::BOOL> {
        let n = self.idx.get();
        self.idx.set(n + 1);
        Ok((n == 0).into())
    }
    fn GetCurrentFontFile(&self) -> windows::core::Result<IDWriteFontFile> {
        unsafe { self.factory.CreateFontFileReference(&HSTRING::from(self.path.as_str()), None) }
    }
}

/// DWRITE_HIT_TEST_METRICS → JSON (one range/point box).
fn hit_metrics_json(hm: &DWRITE_HIT_TEST_METRICS) -> serde_json::Value {
    serde_json::json!({
        "pos": hm.textPosition, "length": hm.length,
        "left": hm.left, "top": hm.top, "width": hm.width, "height": hm.height,
        "bidi": hm.bidiLevel, "is_text": hm.isText.as_bool(), "is_trimmed": hm.isTrimmed.as_bool(),
    })
}

/// Index-0 family name of a glyph run's resolved face, for recording system fallback results.
fn family_name(face: &IDWriteFontFace) -> Option<String> {
    unsafe {
        let names = face.cast::<IDWriteFontFace3>().ok()?.GetFamilyNames().ok()?;
        let len = names.GetStringLength(0).ok()? as usize;
        let mut buf = vec![0u16; len + 1];
        names.GetString(0, &mut buf).ok()?;
        Some(String::from_utf16_lossy(&buf[..len]))
    }
}

// Records every call native makes to a geometry-op output sink (fill mode, figure starts,
// begin/end flags, and each segment with its exact points) AND forwards them to an inner path
// sink so the result is still a fillable geometry. The recorded stream drives a point-by-point
// output-geometry diff against the Rust engine, independent of pixels.
#[implement(ID2D1SimplifiedGeometrySink)]
struct RecordingSink {
    inner: ID2D1SimplifiedGeometrySink,
    fill_mode: Rc<std::cell::Cell<i32>>,
    figures: Rc<RefCell<Vec<serde_json::Value>>>,
    cur: RefCell<Option<(D2D_POINT_2F, i32, Vec<serde_json::Value>)>>,
}

impl ID2D1SimplifiedGeometrySink_Impl for RecordingSink_Impl {
    fn SetFillMode(&self, m: D2D1_FILL_MODE) {
        self.fill_mode.set(m.0);
        unsafe { self.inner.SetFillMode(m) };
    }
    fn SetSegmentFlags(&self, f: D2D1_PATH_SEGMENT) {
        unsafe { self.inner.SetSegmentFlags(f) };
    }
    fn BeginFigure(&self, p: &D2D_POINT_2F, b: D2D1_FIGURE_BEGIN) {
        *self.cur.borrow_mut() = Some((*p, b.0, Vec::new()));
        unsafe { self.inner.BeginFigure(*p, b) };
    }
    fn AddLines(&self, points: *const D2D_POINT_2F, n: u32) {
        let pts = unsafe { std::slice::from_raw_parts(points, n as usize) };
        if let Some((_, _, segs)) = self.cur.borrow_mut().as_mut() {
            for p in pts {
                segs.push(serde_json::json!({"t": "line", "p": [p.x, p.y]}));
            }
        }
        unsafe { self.inner.AddLines(pts) };
    }
    fn AddBeziers(&self, beziers: *const D2D1_BEZIER_SEGMENT, n: u32) {
        let bs = unsafe { std::slice::from_raw_parts(beziers, n as usize) };
        if let Some((_, _, segs)) = self.cur.borrow_mut().as_mut() {
            for b in bs {
                segs.push(serde_json::json!({"t": "bezier",
                    "p": [[b.point1.x, b.point1.y], [b.point2.x, b.point2.y], [b.point3.x, b.point3.y]]}));
            }
        }
        unsafe { self.inner.AddBeziers(bs) };
    }
    fn EndFigure(&self, e: D2D1_FIGURE_END) {
        if let Some((start, begin, segments)) = self.cur.borrow_mut().take() {
            self.figures.borrow_mut().push(serde_json::json!({
                "start": [start.x, start.y], "begin": begin, "end": e.0, "segments": segments
            }));
        }
        unsafe { self.inner.EndFigure(e) };
    }
    fn Close(&self) -> windows::core::Result<()> {
        unsafe { self.inner.Close() }
    }
}

// Records every triangle native emits to a tessellation sink (ID2D1Geometry::Tessellate), for a
// triangle-for-triangle value diff against the Rust fan tessellator.
#[implement(ID2D1TessellationSink)]
struct RecordingTessSink {
    tris: Rc<RefCell<Vec<serde_json::Value>>>,
}

impl ID2D1TessellationSink_Impl for RecordingTessSink_Impl {
    fn AddTriangles(&self, triangles: *const D2D1_TRIANGLE, n: u32) {
        let ts = unsafe { std::slice::from_raw_parts(triangles, n as usize) };
        for t in ts {
            self.tris.borrow_mut().push(serde_json::json!([
                [t.point1.x, t.point1.y], [t.point2.x, t.point2.y], [t.point3.x, t.point3.y]
            ]));
        }
    }
    fn Close(&self) -> windows::core::Result<()> {
        Ok(())
    }
}

// Straight RGBA8 → tightly packed premultiplied BGRA8, matching the upload the
// `bitmap` op has always used; shared with `wic_bitmap`.
fn premultiply_bgra(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .flat_map(|p| {
            let c = |v: u8| ((v as u32 * p[3] as u32 + 127) / 255) as u8;
            [c(p[2]), c(p[1]), c(p[0]), p[3]]
        })
        .collect()
}

// f32 → IEEE half, round-to-nearest-even (preview/upload only). Mirrors the CPU renderer's store.
fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7f_ffff;
    if ((b >> 23) & 0xff) == 0xff {
        return sign | 0x7c00 | if m != 0 { 0x200 } else { 0 }; // inf/nan
    }
    if e >= 0x1f {
        return sign | 0x7c00; // overflow → inf
    }
    if e <= 0 {
        if e < -10 {
            return sign; // underflow → ±0
        }
        let m = m | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let round = if rem > (1 << (shift - 1)) || (rem == (1 << (shift - 1)) && half & 1 == 1) { 1 } else { 0 };
        return sign | (half + round) as u16;
    }
    let half = ((e as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    let round = if rem > 0x1000 || (rem == 0x1000 && half & 1 == 1) { 1 } else { 0 };
    sign | (half + round) as u16
}

// Encode straight RGBA8 into one tightly-packed bitmap of `format`/`alpha` for a CreateBitmap upload.
// Default (BGRA8 premultiplied) is byte-identical to `premultiply_bgra`. Float/R8G8B8A8 inputs let a
// scene define a high-precision source bitmap (values stay in [0,1], from the RGBA8).
fn encode_bitmap(rgba: &[u8], format: DXGI_FORMAT, alpha: D2D1_ALPHA_MODE) -> Vec<u8> {
    if format == DXGI_FORMAT_B8G8R8A8_UNORM && alpha == D2D1_ALPHA_MODE_PREMULTIPLIED {
        return premultiply_bgra(rgba);
    }
    let premul = alpha == D2D1_ALPHA_MODE_PREMULTIPLIED;
    rgba.chunks_exact(4)
        .flat_map(|p| {
            let a = p[3] as f32 / 255.0;
            // [r,g,b] straight in [0,1]; premultiply if the alpha mode asks for it.
            let f = |c: usize| if premul { p[c] as f32 / 255.0 * a } else { p[c] as f32 / 255.0 };
            let (r, g, b) = (f(0), f(1), f(2));
            match format {
                DXGI_FORMAT_R32G32B32A32_FLOAT => [r, g, b, a].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
                DXGI_FORMAT_R16G16B16A16_FLOAT => [r, g, b, a].iter().flat_map(|v| f32_to_f16(*v).to_le_bytes()).collect(),
                _ => {
                    // R8G8B8A8_UNORM (RGBA order)
                    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    vec![q(r), q(g), q(b), p[3]]
                }
            }
        })
        .collect()
}
// Optional brush opacity/transform (v2). None reproduces v1's null properties exactly.
fn brush_props(opacity: Option<f32>, transform: Option<[f32; 6]>) -> Option<D2D1_BRUSH_PROPERTIES> {
    (opacity.is_some() || transform.is_some()).then(|| D2D1_BRUSH_PROPERTIES {
        opacity: opacity.unwrap_or(1.0),
        transform: transform.map(matrix).unwrap_or_else(identity),
    })
}
fn fmt_hr(hr: i32) -> String {
    format!("0x{:08X}", hr as u32)
}
fn hr_of(e: &anyhow::Error) -> Option<i32> {
    e.chain()
        .find_map(|c| c.downcast_ref::<windows::core::Error>())
        .map(|w| w.code().0)
}
fn get<'a, T>(map: &'a HashMap<String, T>, id: &str) -> Result<&'a T> {
    map.get(id)
        .with_context(|| format!("undefined resource {id:?}"))
}
/// `"S_OK"` or the HRESULT (for SVG-DOM negative-test queries that record a call's result code).
fn hr_str(r: windows::core::Result<()>) -> String {
    match r {
        Ok(()) => "S_OK".into(),
        Err(e) => fmt_hr(e.code().0),
    }
}
/// Record one print-scene `variant` into `rec`'s command list, for capturing native's XPS
/// serialisation of that primitive part-by-part (the print serializer's target primitives:
/// glyph runs, gradients, image/bitmap brushes, strokes/dashes, layers/clips, opacity).
unsafe fn record_print_primitive(rec: &ID2D1DeviceContext, dw: &IDWriteFactory, variant: &str) -> Result<()> {
    let rt: ID2D1RenderTarget = rec.cast()?;
    let fill = D2D_RECT_F { left: 10.0, top: 10.0, right: 60.0, bottom: 40.0 };
    match variant {
        // A glyph run → native emits an XPS <Glyphs> element plus an embedded font part.
        "glyphs" => {
            let mut coll = None;
            dw.GetSystemFontCollection(&mut coll, false)?;
            let coll = coll.context("no system font collection")?;
            let (mut idx, mut exists) = (0u32, windows::Win32::Foundation::BOOL(0));
            coll.FindFamilyName(&HSTRING::from("Arial"), &mut idx, &mut exists)?;
            ensure!(exists.as_bool(), "Arial not found");
            let face = coll
                .GetFontFamily(idx)?
                .GetFirstMatchingFont(DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STRETCH_NORMAL, DWRITE_FONT_STYLE_NORMAL)?
                .CreateFontFace()?;
            let codes = ['A' as u32, 'g' as u32];
            let mut ids = [0u16; 2];
            face.GetGlyphIndices(codes.as_ptr(), 2, ids.as_mut_ptr())?;
            let mut fm = DWRITE_FONT_METRICS::default();
            face.GetMetrics(&mut fm);
            let mut gm = [DWRITE_GLYPH_METRICS::default(); 2];
            face.GetDesignGlyphMetrics(ids.as_ptr(), 2, gm.as_mut_ptr(), false)?;
            let upem = fm.designUnitsPerEm as f32;
            let advances: Vec<f32> = gm.iter().map(|m| m.advanceWidth as f32 * 16.0 / upem).collect();
            let offsets = [DWRITE_GLYPH_OFFSET { advanceOffset: 0.0, ascenderOffset: 0.0 }; 2];
            let run = DWRITE_GLYPH_RUN {
                fontFace: ManuallyDrop::new(Some(face.clone())),
                fontEmSize: 16.0,
                glyphCount: 2,
                glyphIndices: ids.as_ptr(),
                glyphAdvances: advances.as_ptr(),
                glyphOffsets: offsets.as_ptr(),
                isSideways: false.into(),
                bidiLevel: 0,
            };
            let brush = rt.CreateSolidColorBrush(&color([0.0, 0.0, 0.0, 1.0]), None)?;
            rec.DrawGlyphRun(D2D_POINT_2F { x: 12.0, y: 40.0 }, &run, None, &brush, DWRITE_MEASURING_MODE_NATURAL);
            let _ = ManuallyDrop::into_inner(run.fontFace);
        }
        // Gradient / brush fills → native emits XPS <LinearGradientBrush>/<RadialGradientBrush>/<ImageBrush>.
        "lineargrad" => {
            let stops = [
                D2D1_GRADIENT_STOP { position: 0.0, color: color([1.0, 0.0, 0.0, 1.0]) },
                D2D1_GRADIENT_STOP { position: 1.0, color: color([0.0, 0.0, 1.0, 1.0]) },
            ];
            let gsc = rt.CreateGradientStopCollection(&stops, D2D1_GAMMA_2_2, D2D1_EXTEND_MODE_CLAMP)?;
            let brush = rt.CreateLinearGradientBrush(
                &D2D1_LINEAR_GRADIENT_BRUSH_PROPERTIES { startPoint: point([10.0, 10.0]), endPoint: point([60.0, 40.0]) },
                None,
                &gsc,
            )?;
            rt.FillRectangle(&fill, &brush);
        }
        "radialgrad" => {
            let stops = [
                D2D1_GRADIENT_STOP { position: 0.0, color: color([1.0, 1.0, 0.0, 1.0]) },
                D2D1_GRADIENT_STOP { position: 1.0, color: color([0.0, 0.5, 0.0, 1.0]) },
            ];
            let gsc = rt.CreateGradientStopCollection(&stops, D2D1_GAMMA_2_2, D2D1_EXTEND_MODE_CLAMP)?;
            let brush = rt.CreateRadialGradientBrush(
                &D2D1_RADIAL_GRADIENT_BRUSH_PROPERTIES {
                    center: point([35.0, 25.0]),
                    gradientOriginOffset: point([0.0, 0.0]),
                    radiusX: 25.0,
                    radiusY: 15.0,
                },
                None,
                &gsc,
            )?;
            rt.FillRectangle(&fill, &brush);
        }
        "bitmapbrush" => {
            let px: [u8; 16] = [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255];
            let props = D2D1_BITMAP_PROPERTIES {
                pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT_B8G8R8A8_UNORM, alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED },
                dpiX: 96.0,
                dpiY: 96.0,
            };
            let bmp = rt.CreateBitmap(D2D_SIZE_U { width: 2, height: 2 }, Some(px.as_ptr() as *const _), 8, &props)?;
            let brush = rt.CreateBitmapBrush(&bmp, None, None)?;
            rt.FillRectangle(&fill, &brush);
        }
        // WRAP-extend bitmap-brush variants to derive the XPS tile Viewbox formula empirically (vary the
        // bitmap size `bw`×`bh` and the fill rect). WRAP tiles directly (no CLAMP-emulation nesting).
        v if v.starts_with("bbwrap") || v.starts_with("bbmir") || v.starts_with("bbclamp") || v.starts_with("bbcmp") => {
            // bbwrap* = WRAP/WRAP; bbmir = MIRROR/MIRROR; bbmirx = WRAP x / MIRROR y; bbmiry = MIRROR x / WRAP y;
            // bbclamp* = CLAMP/CLAMP (nested clamp-emulation); bbcmpx = CLAMP x / WRAP y; bbcmpy = WRAP x / CLAMP y;
            // bbcmpmx = CLAMP x / MIRROR y; bbcmpmy = MIRROR x / CLAMP y (mixed clamp emulation). Size/fill
            // suffixes: *4 = 4x4; *fill = fill rect (5,5,45,25); *ov = fill (0,0,60,40) (overlaps the origin).
            let (bw, bh) = if v.ends_with('4') { (4u32, 4u32) } else { (2u32, 2u32) };
            let fillr = if v.ends_with("fill") {
                D2D_RECT_F { left: 5.0, top: 5.0, right: 45.0, bottom: 25.0 }
            } else if v.ends_with("ov") {
                D2D_RECT_F { left: 0.0, top: 0.0, right: 60.0, bottom: 40.0 }
            } else {
                fill
            };
            let (ex, ey) = match v {
                "bbmirx" => (D2D1_EXTEND_MODE_WRAP, D2D1_EXTEND_MODE_MIRROR),
                "bbmiry" => (D2D1_EXTEND_MODE_MIRROR, D2D1_EXTEND_MODE_WRAP),
                v if v.starts_with("bbmir") => (D2D1_EXTEND_MODE_MIRROR, D2D1_EXTEND_MODE_MIRROR),
                v if v.starts_with("bbclamp") => (D2D1_EXTEND_MODE_CLAMP, D2D1_EXTEND_MODE_CLAMP),
                v if v.starts_with("bbcmpmx") => (D2D1_EXTEND_MODE_CLAMP, D2D1_EXTEND_MODE_MIRROR),
                v if v.starts_with("bbcmpmy") => (D2D1_EXTEND_MODE_MIRROR, D2D1_EXTEND_MODE_CLAMP),
                v if v.starts_with("bbcmpx") => (D2D1_EXTEND_MODE_CLAMP, D2D1_EXTEND_MODE_WRAP),
                v if v.starts_with("bbcmpy") => (D2D1_EXTEND_MODE_WRAP, D2D1_EXTEND_MODE_CLAMP),
                _ => (D2D1_EXTEND_MODE_WRAP, D2D1_EXTEND_MODE_WRAP),
            };
            let px: Vec<u8> = (0..bw * bh).flat_map(|i| [(i * 37) as u8, (i * 53) as u8, (i * 71) as u8, 255]).collect();
            let props = D2D1_BITMAP_PROPERTIES {
                pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT_B8G8R8A8_UNORM, alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED },
                dpiX: 96.0,
                dpiY: 96.0,
            };
            let bmp = rt.CreateBitmap(D2D_SIZE_U { width: bw, height: bh }, Some(px.as_ptr() as *const _), bw * 4, &props)?;
            let bbp = D2D1_BITMAP_BRUSH_PROPERTIES {
                extendModeX: ex,
                extendModeY: ey,
                interpolationMode: D2D1_BITMAP_INTERPOLATION_MODE_LINEAR,
            };
            let brush = rt.CreateBitmapBrush(&bmp, Some(&bbp), None)?;
            rt.FillRectangle(&fillr, &brush);
        }
        // A stroked, dashed line → native emits an XPS <Path> with Stroke / StrokeDashArray.
        "stroke" => {
            let factory: ID2D1Factory = rt.GetFactory()?;
            let style = factory.CreateStrokeStyle(
                &D2D1_STROKE_STYLE_PROPERTIES {
                    startCap: D2D1_CAP_STYLE_FLAT,
                    endCap: D2D1_CAP_STYLE_FLAT,
                    dashCap: D2D1_CAP_STYLE_FLAT,
                    lineJoin: D2D1_LINE_JOIN_MITER,
                    miterLimit: 10.0,
                    dashStyle: D2D1_DASH_STYLE_DASH,
                    dashOffset: 0.0,
                },
                None,
            )?;
            let brush = rt.CreateSolidColorBrush(&color([0.0, 0.0, 1.0, 1.0]), None)?;
            rt.DrawLine(point([10.0, 10.0]), point([60.0, 40.0]), &brush, 3.0, &style);
        }
        // A clip layer with partial opacity → native emits an XPS <Canvas> with Clip + Opacity.
        "layer" => {
            let brush = rt.CreateSolidColorBrush(&color([1.0, 0.0, 0.0, 1.0]), None)?;
            let mut params = D2D1_LAYER_PARAMETERS1 {
                contentBounds: D2D_RECT_F { left: 15.0, top: 15.0, right: 55.0, bottom: 35.0 },
                geometricMask: ManuallyDrop::new(None),
                maskAntialiasMode: D2D1_ANTIALIAS_MODE_PER_PRIMITIVE,
                maskTransform: identity(),
                opacity: 0.5,
                opacityBrush: ManuallyDrop::new(None),
                layerOptions: D2D1_LAYER_OPTIONS1(0),
            };
            rec.PushLayer(&params, None);
            rt.FillRectangle(&fill, &brush);
            rec.PopLayer();
            ManuallyDrop::drop(&mut params.geometricMask);
            ManuallyDrop::drop(&mut params.opacityBrush);
        }
        "opacity" => {
            let brush = rt.CreateSolidColorBrush(&color([1.0, 0.0, 0.0, 1.0]), None)?;
            brush.SetOpacity(0.5);
            rt.FillRectangle(&fill, &brush);
        }
        // Default "rect": one opaque solid fill (the cov-12 baseline).
        _ => {
            let brush = rt.CreateSolidColorBrush(&color([1.0, 0.0, 0.0, 1.0]), None)?;
            rt.FillRectangle(&fill, &brush);
        }
    }
    Ok(())
}
/// Byte size of a `D2D1_SVG_ATTRIBUTE_POD_TYPE` value struct (for `GetAttributeValue2` buffers).
fn pod_size(t: D2D1_SVG_ATTRIBUTE_POD_TYPE) -> usize {
    match t.0 {
        1 => 16,  // COLOR (D2D1_COLOR_F)
        8 => 24,  // MATRIX (D2D1_MATRIX_3X2_F)
        11 => 12, // PRESERVE_ASPECT_RATIO
        12 => 16, // VIEWBOX
        13 => 8,  // LENGTH
        _ => 4,   // FLOAT / enums
    }
}
/// Read an `IStream` from the start into a string (the `Serialize` output).
unsafe fn stream_to_string(s: &IStream) -> Result<String> {
    s.Seek(0, STREAM_SEEK_SET, None)?;
    let mut out = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        let mut read = 0u32;
        s.Read(buf.as_mut_ptr().cast(), buf.len() as u32, Some(&mut read)).ok()?;
        if read == 0 {
            break;
        }
        out.extend_from_slice(&buf[..read as usize]);
    }
    Ok(String::from_utf8_lossy(&out).into_owned())
}
fn put<T>(map: &mut HashMap<String, T>, id: &str, value: T) -> Result<()> {
    ensure!(!map.contains_key(id), "duplicate resource {id:?}");
    map.insert(id.into(), value);
    Ok(())
}

unsafe fn geometry(factory: &ID2D1Factory, shape: &Shape) -> Result<ID2D1Geometry> {
    Ok(match shape {
        Shape::Rect { rect: r } => factory.CreateRectangleGeometry(&rect(*r))?.cast()?,
        Shape::Rounded { rect: r, radius } => factory
            .CreateRoundedRectangleGeometry(&D2D1_ROUNDED_RECT {
                rect: rect(*r),
                radiusX: radius[0],
                radiusY: radius[1],
            })?
            .cast()?,
        Shape::Ellipse { center, radius } => factory
            .CreateEllipseGeometry(&D2D1_ELLIPSE {
                point: point(*center),
                radiusX: radius[0],
                radiusY: radius[1],
            })?
            .cast()?,
        Shape::Path { fill, figures } => {
            let path = factory.CreatePathGeometry()?;
            let sink = path.Open()?;
            sink.SetFillMode(D2D1_FILL_MODE(*fill as i32));
            for f in figures {
                sink.BeginFigure(
                    point(f.start),
                    if f.filled {
                        D2D1_FIGURE_BEGIN_FILLED
                    } else {
                        D2D1_FIGURE_BEGIN_HOLLOW
                    },
                );
                for (si, segment) in f.segments.iter().enumerate() {
                    sink.SetSegmentFlags(D2D1_PATH_SEGMENT(f.seg_flags.get(si).copied().unwrap_or(0) as i32));
                    match segment {
                        Segment::Line { to } => sink.AddLine(point(*to)),
                        Segment::Cubic { p1, p2, to } => sink.AddBezier(&D2D1_BEZIER_SEGMENT {
                            point1: point(*p1),
                            point2: point(*p2),
                            point3: point(*to),
                        }),
                        Segment::Quadratic { control, to } => {
                            sink.AddQuadraticBezier(&D2D1_QUADRATIC_BEZIER_SEGMENT {
                                point1: point(*control),
                                point2: point(*to),
                            })
                        }
                        Segment::Arc {
                            to,
                            radius,
                            rotation,
                            sweep,
                            size,
                        } => sink.AddArc(&D2D1_ARC_SEGMENT {
                            point: point(*to),
                            size: D2D_SIZE_F {
                                width: radius[0],
                                height: radius[1],
                            },
                            rotationAngle: *rotation,
                            sweepDirection: D2D1_SWEEP_DIRECTION(*sweep as i32),
                            arcSize: D2D1_ARC_SIZE(*size as i32),
                        }),
                    }
                }
                sink.EndFigure(if f.closed {
                    D2D1_FIGURE_END_CLOSED
                } else {
                    D2D1_FIGURE_END_OPEN
                });
            }
            sink.Close()?;
            path.cast()?
        }
    })
}

fn property(value: &Value) -> (D2D1_PROPERTY_TYPE, Vec<u8>) {
    let floats = |v: &[f32]| v.iter().flat_map(|v| v.to_le_bytes()).collect();
    match value {
        Value::Float(v) => (D2D1_PROPERTY_TYPE_FLOAT, v.to_le_bytes().to_vec()),
        Value::Uint32(v) => (D2D1_PROPERTY_TYPE_UINT32, v.to_le_bytes().to_vec()),
        Value::Int32(v) => (D2D1_PROPERTY_TYPE_INT32, v.to_le_bytes().to_vec()),
        Value::Floats(v) => (D2D1_PROPERTY_TYPE_BLOB, floats(v)),
        Value::Enum(v) => (D2D1_PROPERTY_TYPE_ENUM, v.to_le_bytes().to_vec()),
        Value::Bool(v) => (
            D2D1_PROPERTY_TYPE_BOOL,
            u32::from(*v).to_le_bytes().to_vec(),
        ),
        Value::Vector2(v) => (D2D1_PROPERTY_TYPE_VECTOR2, floats(v)),
        Value::Vector3(v) => (D2D1_PROPERTY_TYPE_VECTOR3, floats(v)),
        Value::Vector4(v) => (D2D1_PROPERTY_TYPE_VECTOR4, floats(v)),
        Value::Matrix3x2(v) => (D2D1_PROPERTY_TYPE_MATRIX_3X2, floats(v)),
        Value::Matrix4x4(v) => (D2D1_PROPERTY_TYPE_MATRIX_4X4, floats(v)),
        Value::Matrix5x4(v) => (D2D1_PROPERTY_TYPE_MATRIX_5X4, floats(v)),
        Value::Iunknown(_) => unreachable!("iunknown properties are resolved in the effect op"),
    }
}

// ------------------------------------------------------------------------------------------------
// Phase 2B-3: custom shader-effect authoring. `register_shader_effect` registers a real
// ID2D1Effect (ID2D1EffectImpl + a single ID2D1DrawTransform) whose draw transform binds an authored
// pixel shader, so native Direct2D runs that shader through its own effect pipeline on the WARP software rasterizer — the oracle
// for crates/d2d's `effects::shader` realization. The constant buffer is baked into the effect impl
// (no author properties), passed from the op through this thread-local to the D2D-invoked factory.
#[derive(Clone)]
struct ShaderData {
    guid: windows::core::GUID,
    dxbc: Vec<u8>,
    cbuffer: Vec<u8>,
    inputs: u32,
    filter: D2D1_FILTER,
    resource_textures: Vec<ResourceTextureSpec>,
    precision: Option<u32>,
    input_inflate: Option<i32>,
    pixel_options: u32,
}

thread_local! {
    static PENDING_SHADER: RefCell<Option<ShaderData>> = const { RefCell::new(None) };
}

/// A stable per-resource key derived from the effect's GUID and a slot index, so keyed resources
/// (resource textures, vertex buffers) are distinct per effect+slot and re-findable across
/// Initialize calls. `data1 ^= 0x8000_0000 ^ index` keeps it collision-free within one effect.
fn derive_resource_key(base: &windows::core::GUID, index: u32) -> windows::core::GUID {
    let mut g = *base;
    g.data1 ^= 0x8000_0000 ^ index;
    g
}

#[implement(ID2D1EffectImpl)]
struct ShaderEffect {
    data: ShaderData,
}

impl ID2D1EffectImpl_Impl for ShaderEffect_Impl {
    fn Initialize(&self, ctx: Option<&ID2D1EffectContext>, graph: Option<&ID2D1TransformGraph>) -> windows::core::Result<()> {
        let (ctx, graph) = (ctx.unwrap(), graph.unwrap());
        unsafe {
            ctx.LoadPixelShader(&self.data.guid, &self.data.dxbc)?;
            // Create each bound resource texture (CreateResourceTexture) now; the draw transform
            // binds them per register in SetDrawInfo (SetResourceTexture). `extents`/`extendModes`
            // are raw pointers read synchronously by the (copying) create call, so the backing Vecs
            // only need to outlive it.
            let mut textures: Vec<(u32, ID2D1ResourceTexture)> = Vec::new();
            for rt in &self.data.resource_textures {
                let extends: Vec<D2D1_EXTEND_MODE> = if rt.extend_modes.is_empty() {
                    vec![D2D1_EXTEND_MODE(0); rt.dimensions as usize]
                } else {
                    rt.extend_modes.iter().map(|&m| D2D1_EXTEND_MODE(m as i32)).collect()
                };
                let props = D2D1_RESOURCE_TEXTURE_PROPERTIES {
                    extents: rt.extents.as_ptr(),
                    dimensions: rt.dimensions,
                    bufferPrecision: D2D1_BUFFER_PRECISION(rt.precision as i32),
                    channelDepth: D2D1_CHANNEL_DEPTH(rt.channel_depth as i32),
                    filter: D2D1_FILTER(rt.filter as i32),
                    extendModes: extends.as_ptr(),
                };
                // CreateResourceTexture needs the per-axis byte strides for 2-D/3-D textures (row then
                // slab pitch, dimensions-1 of them); a null stride is only valid for a 1-D texture, so
                // a 2-D resource texture (e.g. an ordered-dither threshold matrix) created with None
                // fails and the effect renders nothing. strides[k] = Π(extents[0..=k]) · texel bytes.
                let bpc = match rt.precision { 1 | 2 => 1u32, 3 | 4 => 2, _ => 4 };
                let texel = bpc * if rt.channel_depth == 1 { 1 } else { 4 };
                let strides: Vec<u32> = (0..rt.dimensions as usize - 1)
                    .map(|k| rt.extents[..=k].iter().product::<u32>() * texel)
                    .collect();
                let strides_ptr = if strides.is_empty() { None } else { Some(strides.as_ptr()) };
                let tex = ctx.CreateResourceTexture(None, &props, Some(&rt.data), strides_ptr)?;
                textures.push((rt.index, tex));
            }
            let node: ID2D1DrawTransform = ShaderTransform { data: self.data.clone(), textures }.into();
            graph.SetSingleTransformNode(&node)
        }
    }
    fn PrepareForRender(&self, _change: D2D1_CHANGE_TYPE) -> windows::core::Result<()> {
        Ok(())
    }
    fn SetGraph(&self, _graph: Option<&ID2D1TransformGraph>) -> windows::core::Result<()> {
        Err(windows::Win32::Foundation::E_NOTIMPL.into()) // fixed input count
    }
}

#[implement(ID2D1DrawTransform)]
struct ShaderTransform {
    data: ShaderData,
    textures: Vec<(u32, ID2D1ResourceTexture)>,
}

impl ID2D1TransformNode_Impl for ShaderTransform_Impl {
    fn GetInputCount(&self) -> u32 {
        self.data.inputs
    }
}

impl ID2D1Transform_Impl for ShaderTransform_Impl {
    fn MapOutputRectToInputRects(&self, output: *const windows::Win32::Foundation::RECT, inputs: *mut windows::Win32::Foundation::RECT, count: u32) -> windows::core::Result<()> {
        unsafe {
            let o = *output;
            let d = self.data.input_inflate.unwrap_or(0); // identity 1:1 draw transform unless inflated
            for i in 0..count as usize {
                *inputs.add(i) = windows::Win32::Foundation::RECT {
                    left: o.left.saturating_sub(d),
                    top: o.top.saturating_sub(d),
                    right: o.right.saturating_add(d),
                    bottom: o.bottom.saturating_add(d),
                };
            }
        }
        Ok(())
    }
    fn MapInputRectsToOutputRect(&self, inputs: *const windows::Win32::Foundation::RECT, _opaque: *const windows::Win32::Foundation::RECT, count: u32, output: *mut windows::Win32::Foundation::RECT, output_opaque: *mut windows::Win32::Foundation::RECT) -> windows::core::Result<()> {
        unsafe {
            *output = if count == 0 {
                windows::Win32::Foundation::RECT { left: i32::MIN, top: i32::MIN, right: i32::MAX, bottom: i32::MAX }
            } else {
                *inputs // union == inputs[0] for the single-input case
            };
            *output_opaque = windows::Win32::Foundation::RECT { left: 0, top: 0, right: 0, bottom: 0 };
        }
        Ok(())
    }
    fn MapInvalidRect(&self, _index: u32, invalid: &windows::Win32::Foundation::RECT) -> windows::core::Result<windows::Win32::Foundation::RECT> {
        Ok(*invalid)
    }
}

impl ID2D1DrawTransform_Impl for ShaderTransform_Impl {
    fn SetDrawInfo(&self, info: Option<&ID2D1DrawInfo>) -> windows::core::Result<()> {
        let info = info.unwrap();
        unsafe {
            info.SetPixelShader(&self.data.guid, D2D1_PIXEL_OPTIONS(self.data.pixel_options as i32))?;
            if !self.data.cbuffer.is_empty() {
                info.SetPixelShaderConstantBuffer(&self.data.cbuffer)?;
            }
            for i in 0..self.data.inputs {
                info.SetInputDescription(i, D2D1_INPUT_DESCRIPTION { filter: self.data.filter, levelOfDetailCount: 0 })?;
            }
            for (index, tex) in &self.textures {
                info.SetResourceTexture(*index, tex)?;
            }
            // SetOutputBuffer pins this transform's intermediate precision (the top term in the D2D1
            // precision rule) — e.g. a float buffer a downstream shader reads without 8bpc truncation.
            if let Some(p) = self.data.precision {
                info.SetOutputBuffer(D2D1_BUFFER_PRECISION(p as i32), D2D1_CHANNEL_DEPTH(4))?;
            }
            // Write-only transform-graph hints with no pixel effect: SetCached (whether D2D may cache
            // this transform's output) and SetInstructionCountHint (shader instruction count). Exercise
            // them so the ID2D1RenderInfo + ID2D1ConcreteTransform SetCached/SetInstructionCountHint
            // surface is covered (they ride the register_shader_effect scenes unchanged).
            info.SetCached(false);
            info.SetInstructionCountHint(0);
        }
        Ok(())
    }
}

// A never-invoked effect factory for registration-only coverage (the effect is never CreateEffect'd).
unsafe extern "system" fn dummy_effect_factory(_out: *mut Option<windows::core::IUnknown>) -> windows::core::HRESULT {
    windows::Win32::Foundation::E_NOTIMPL
}

// The PD2D1_EFFECT_FACTORY D2D invokes at CreateEffect; builds the effect from the pending data.
unsafe extern "system" fn shader_effect_factory(out: *mut Option<windows::core::IUnknown>) -> windows::core::HRESULT {
    match PENDING_SHADER.with(|p| p.borrow().clone()) {
        Some(data) => {
            let effect: ID2D1EffectImpl = ShaderEffect { data }.into();
            match effect.cast::<windows::core::IUnknown>() {
                Ok(unk) => {
                    *out = Some(unk);
                    windows::core::HRESULT(0)
                }
                Err(e) => e.code(),
            }
        }
        None => windows::Win32::Foundation::E_FAIL,
    }
}

// A custom effect whose transform graph is built from the non-shader built-in transforms (border →
// offset → bounds-adjustment), exercising the ID2D1EffectContext Create*Transform API, the transform
// setters/getters, and the multi-node ID2D1TransformGraph wiring (AddNode/ConnectToEffectInput/
// ConnectNode/SetOutputNode). Native d2d1 runs the graph; the core effects::custom engine is the oracle.
#[derive(Clone)]
struct GraphData {
    offset: [i32; 2],
    border: [u32; 2],
    bounds: [i32; 4],
}
thread_local! {
    static PENDING_GRAPH: RefCell<Option<GraphData>> = const { RefCell::new(None) };
}
#[implement(ID2D1EffectImpl)]
struct GraphEffect {
    data: GraphData,
}
impl ID2D1EffectImpl_Impl for GraphEffect_Impl {
    fn Initialize(&self, ctx: Option<&ID2D1EffectContext>, graph: Option<&ID2D1TransformGraph>) -> windows::core::Result<()> {
        let (ctx, graph) = (ctx.unwrap(), graph.unwrap());
        let b = self.data.bounds;
        unsafe {
            // EffectContext device-capability getters (not graph-affecting; exercised for coverage).
            let _ = ctx.GetMaximumSupportedFeatureLevel(&[D3D_FEATURE_LEVEL_9_1, D3D_FEATURE_LEVEL_10_0, D3D_FEATURE_LEVEL_11_0]);
            let mut doubles = 0i32;
            let _ = ctx.CheckFeatureSupport(D2D1_FEATURE_DOUBLES, &mut doubles as *mut i32 as *mut _, 4);
            let border = ctx.CreateBorderTransform(D2D1_EXTEND_MODE(self.data.border[0] as i32), D2D1_EXTEND_MODE(self.data.border[1] as i32))?;
            let offset = ctx.CreateOffsetTransform(windows::Win32::Foundation::POINT { x: 0, y: 0 })?;
            offset.SetOffset(windows::Win32::Foundation::POINT { x: self.data.offset[0], y: self.data.offset[1] });
            let _ = offset.GetOffset();
            let bounds = ctx.CreateBoundsAdjustmentTransform(&RECT { left: 0, top: 0, right: 0, bottom: 0 })?;
            bounds.SetOutputBounds(&RECT { left: b[0], top: b[1], right: b[2], bottom: b[3] });
            let _ = bounds.GetOutputBounds();
            let bn: ID2D1TransformNode = border.cast()?;
            let on: ID2D1TransformNode = offset.cast()?;
            let dn: ID2D1TransformNode = bounds.cast()?;
            // ID2D1TransformGraph::SetPassthroughGraph coverage: set the output to effect-input 0, then
            // build the real graph below — the final SetOutputNode overrides it, so the output is
            // unchanged (both engines end with the node as output).
            graph.SetPassthroughGraph(0)?;
            graph.AddNode(&bn)?;
            graph.AddNode(&on)?;
            graph.AddNode(&dn)?;
            graph.ConnectToEffectInput(0, &bn, 0)?;
            graph.ConnectNode(&bn, &on, 0)?;
            graph.ConnectNode(&on, &dn, 0)?;
            graph.SetOutputNode(&dn)?;
            // Coverage (not wired into the output graph, so the rendered result is unchanged):
            // CreateResourceTexture(keyed) → FindResourceTexture round-trips the same texture, and
            // CreateEffect → CreateTransformNodeFromEffect wraps a Flood sub-effect as a node.
            let rkey = windows::core::GUID::from_u128(0x0d2d_0001);
            let extents = [2u32];
            let extends = [D2D1_EXTEND_MODE(0)];
            let rprops = D2D1_RESOURCE_TEXTURE_PROPERTIES {
                extents: extents.as_ptr(), dimensions: 1,
                bufferPrecision: D2D1_BUFFER_PRECISION(1), channelDepth: D2D1_CHANNEL_DEPTH(1),
                filter: D2D1_FILTER(0), extendModes: extends.as_ptr(),
            };
            if let Ok(tex) = ctx.CreateResourceTexture(Some(&rkey), &rprops, Some(&[0u8, 255]), None) {
                let _ = ctx.FindResourceTexture(&rkey);
                // ID2D1ResourceTexture::Update — overwrite the 1-D texels.
                let (minext, maxext) = ([0u32], [2u32]);
                let _ = tex.Update(Some(minext.as_ptr()), Some(maxext.as_ptr()), None, 1, &[128u8, 64]);
            }
            if let Ok(sub) = ctx.CreateEffect(&windows::Win32::Graphics::Direct2D::CLSID_D2D1Flood) {
                let _ = ctx.CreateTransformNodeFromEffect(&sub);
            }
            // ID2D1TransformGraph::RemoveNode — add a throwaway node and remove it (graph unchanged).
            if let Ok(extra) = ctx.CreateBorderTransform(D2D1_EXTEND_MODE(0), D2D1_EXTEND_MODE(0)) {
                if let Ok(en) = extra.cast::<ID2D1TransformNode>() {
                    if graph.AddNode(&en).is_ok() {
                        let _ = graph.RemoveNode(&en);
                    }
                }
            }
        }
        Ok(())
    }
    fn PrepareForRender(&self, _c: D2D1_CHANGE_TYPE) -> windows::core::Result<()> {
        Ok(())
    }
    fn SetGraph(&self, _g: Option<&ID2D1TransformGraph>) -> windows::core::Result<()> {
        Err(windows::Win32::Foundation::E_NOTIMPL.into()) // fixed input count
    }
}
unsafe extern "system" fn graph_effect_factory(out: *mut Option<windows::core::IUnknown>) -> windows::core::HRESULT {
    match PENDING_GRAPH.with(|p| p.borrow().clone()) {
        Some(data) => {
            let effect: ID2D1EffectImpl = GraphEffect { data }.into();
            match effect.cast::<windows::core::IUnknown>() {
                Ok(unk) => {
                    *out = Some(unk);
                    windows::core::HRESULT(0)
                }
                Err(e) => e.code(),
            }
        }
        None => windows::Win32::Foundation::E_FAIL,
    }
}

// --- print capture: a custom IPrintDocumentPackageTarget that hands native an XPS-OM package writer
// writing the parts to a file, so native's CreatePrintControl/AddPage produce vector XPS on disk to
// diff against ours. Windows-only (the XPS OM is a Windows API).
#[cfg(windows)]
#[windows::core::implement(windows::Win32::Storage::Xps::Printing::IPrintDocumentPackageTarget)]
struct XpsCaptureTarget {
    factory: windows::Win32::Storage::Xps::IXpsOMObjectFactory,
    path: HSTRING,
}

#[cfg(windows)]
impl windows::Win32::Storage::Xps::Printing::IPrintDocumentPackageTarget_Impl for XpsCaptureTarget_Impl {
    fn GetPackageTargetTypes(&self, targetcount: *mut u32, targettypes: *mut *mut windows::core::GUID) -> windows::core::Result<()> {
        unsafe {
            let p = windows::Win32::System::Com::CoTaskMemAlloc(core::mem::size_of::<windows::core::GUID>()) as *mut windows::core::GUID;
            if p.is_null() {
                return Err(windows::Win32::Foundation::E_OUTOFMEMORY.into());
            }
            *p = windows::Win32::Storage::Xps::Printing::ID_DOCUMENTPACKAGETARGET_OPENXPS;
            *targettypes = p;
            *targetcount = 1;
        }
        Ok(())
    }
    fn GetPackageTarget(&self, _guid: *const windows::core::GUID, riid: *const windows::core::GUID, ppv: *mut *mut core::ffi::c_void) -> windows::core::Result<()> {
        let inner: windows::core::IUnknown = XpsInner { factory: self.factory.clone(), path: self.path.clone() }.into();
        unsafe { inner.query(riid, ppv).ok() }
    }
    fn Cancel(&self) -> windows::core::Result<()> {
        Ok(())
    }
}

#[cfg(windows)]
#[windows::core::implement(windows::Win32::Storage::Xps::IXpsDocumentPackageTarget)]
struct XpsInner {
    factory: windows::Win32::Storage::Xps::IXpsOMObjectFactory,
    path: HSTRING,
}

#[cfg(windows)]
impl windows::Win32::Storage::Xps::IXpsDocumentPackageTarget_Impl for XpsInner_Impl {
    fn GetXpsOMPackageWriter(&self, seq: Option<&windows::Win32::Storage::Packaging::Opc::IOpcPartUri>, discard: Option<&windows::Win32::Storage::Packaging::Opc::IOpcPartUri>) -> windows::core::Result<windows::Win32::Storage::Xps::IXpsOMPackageWriter> {
        unsafe {
            self.factory.CreatePackageWriterOnFile(
                windows::core::PCWSTR(self.path.as_ptr()),
                core::ptr::null(),
                0,
                false,
                windows::Win32::Storage::Xps::XPS_INTERLEAVING_OFF,
                seq.unwrap(),
                None,
                None,
                None,
                discard,
            )
        }
    }
    fn GetXpsOMFactory(&self) -> windows::core::Result<windows::Win32::Storage::Xps::IXpsOMObjectFactory> {
        Ok(self.factory.clone())
    }
    fn GetXpsType(&self) -> windows::core::Result<windows::Win32::Storage::Xps::XPS_DOCUMENT_TYPE> {
        Ok(windows::Win32::Storage::Xps::XPS_DOCUMENT_TYPE_OPENXPS)
    }
}

// A 2-input custom effect that blends its inputs with a D2D1_BLEND_DESCRIPTION — exercises
// ID2D1EffectContext::CreateBlendTransform, ID2D1BlendTransform Set/GetDescription, and the
// multi-input ConnectToEffectInput graph wiring.
#[derive(Clone)]
struct BlendData {
    desc: [u32; 6], // src, dst, op, src_a, dst_a, op_a
    factor: [f32; 4],
}
thread_local! {
    static PENDING_BLEND: RefCell<Option<BlendData>> = const { RefCell::new(None) };
}
#[implement(ID2D1EffectImpl)]
struct BlendEffect {
    data: BlendData,
}
impl ID2D1EffectImpl_Impl for BlendEffect_Impl {
    fn Initialize(&self, ctx: Option<&ID2D1EffectContext>, graph: Option<&ID2D1TransformGraph>) -> windows::core::Result<()> {
        let (ctx, graph) = (ctx.unwrap(), graph.unwrap());
        let d = &self.data.desc;
        let desc = D2D1_BLEND_DESCRIPTION {
            sourceBlend: D2D1_BLEND(d[0] as i32),
            destinationBlend: D2D1_BLEND(d[1] as i32),
            blendOperation: D2D1_BLEND_OPERATION(d[2] as i32),
            sourceBlendAlpha: D2D1_BLEND(d[3] as i32),
            destinationBlendAlpha: D2D1_BLEND(d[4] as i32),
            blendOperationAlpha: D2D1_BLEND_OPERATION(d[5] as i32),
            blendFactor: self.data.factor,
        };
        unsafe {
            let blend = ctx.CreateBlendTransform(2, &desc)?;
            blend.SetDescription(&desc);
            let mut gd = D2D1_BLEND_DESCRIPTION::default();
            blend.GetDescription(&mut gd);
            let bn: ID2D1TransformNode = blend.cast()?;
            graph.AddNode(&bn)?;
            graph.ConnectToEffectInput(0, &bn, 0)?;
            graph.ConnectToEffectInput(1, &bn, 1)?;
            graph.SetOutputNode(&bn)?;
        }
        Ok(())
    }
    fn PrepareForRender(&self, _c: D2D1_CHANGE_TYPE) -> windows::core::Result<()> {
        Ok(())
    }
    fn SetGraph(&self, _g: Option<&ID2D1TransformGraph>) -> windows::core::Result<()> {
        Err(windows::Win32::Foundation::E_NOTIMPL.into())
    }
}
unsafe extern "system" fn blend_effect_factory(out: *mut Option<windows::core::IUnknown>) -> windows::core::HRESULT {
    match PENDING_BLEND.with(|p| p.borrow().clone()) {
        Some(data) => {
            let effect: ID2D1EffectImpl = BlendEffect { data }.into();
            match effect.cast::<windows::core::IUnknown>() {
                Ok(unk) => {
                    *out = Some(unk);
                    windows::core::HRESULT(0)
                }
                Err(e) => e.code(),
            }
        }
        None => windows::Win32::Foundation::E_FAIL,
    }
}

// A 0-input custom effect whose transform is an ID2D1ComputeTransform — exercises LoadComputeShader,
// ID2D1ComputeInfo::SetComputeShader, and ID2D1ComputeTransform::SetComputeInfo/CalculateThreadgroups.
// The compute shader generates a fixed `out_size` image (one thread group of the DXBC's thread dims).
#[derive(Clone)]
struct ComputeData {
    guid: windows::core::GUID,
    dxbc: Vec<u8>,
    cbuffer: Vec<u8>,
    groups: [u32; 3],
    out_size: [i32; 2],
}
thread_local! {
    static PENDING_COMPUTE: RefCell<Option<ComputeData>> = const { RefCell::new(None) };
}
#[implement(ID2D1EffectImpl)]
struct ComputeEffect {
    data: ComputeData,
}
impl ID2D1EffectImpl_Impl for ComputeEffect_Impl {
    fn Initialize(&self, ctx: Option<&ID2D1EffectContext>, graph: Option<&ID2D1TransformGraph>) -> windows::core::Result<()> {
        let (ctx, graph) = (ctx.unwrap(), graph.unwrap());
        unsafe {
            ctx.LoadComputeShader(&self.data.guid, &self.data.dxbc)?;
            // IsShaderLoaded must report the shader we just loaded (load-bearing: a wrong answer here
            // fails the effect, so a byte-exact render confirms it).
            if !ctx.IsShaderLoaded(&self.data.guid).as_bool() {
                return Err(windows::Win32::Foundation::E_FAIL.into());
            }
            let node: ID2D1ComputeTransform = ComputeXform { data: self.data.clone() }.into();
            let tn: ID2D1TransformNode = node.cast()?;
            graph.SetSingleTransformNode(&tn)
        }
    }
    fn PrepareForRender(&self, _c: D2D1_CHANGE_TYPE) -> windows::core::Result<()> {
        Ok(())
    }
    fn SetGraph(&self, _g: Option<&ID2D1TransformGraph>) -> windows::core::Result<()> {
        Err(windows::Win32::Foundation::E_NOTIMPL.into())
    }
}
#[implement(ID2D1ComputeTransform)]
struct ComputeXform {
    data: ComputeData,
}
impl ID2D1TransformNode_Impl for ComputeXform_Impl {
    fn GetInputCount(&self) -> u32 {
        0 // generator
    }
}
impl ID2D1Transform_Impl for ComputeXform_Impl {
    fn MapOutputRectToInputRects(&self, _output: *const windows::Win32::Foundation::RECT, _inputs: *mut windows::Win32::Foundation::RECT, _count: u32) -> windows::core::Result<()> {
        Ok(()) // no inputs
    }
    fn MapInputRectsToOutputRect(&self, _inputs: *const windows::Win32::Foundation::RECT, _opaque: *const windows::Win32::Foundation::RECT, _count: u32, output: *mut windows::Win32::Foundation::RECT, output_opaque: *mut windows::Win32::Foundation::RECT) -> windows::core::Result<()> {
        unsafe {
            *output = windows::Win32::Foundation::RECT { left: 0, top: 0, right: self.data.out_size[0], bottom: self.data.out_size[1] };
            *output_opaque = windows::Win32::Foundation::RECT { left: 0, top: 0, right: 0, bottom: 0 };
        }
        Ok(())
    }
    fn MapInvalidRect(&self, _index: u32, _invalid: &windows::Win32::Foundation::RECT) -> windows::core::Result<windows::Win32::Foundation::RECT> {
        Ok(windows::Win32::Foundation::RECT { left: 0, top: 0, right: self.data.out_size[0], bottom: self.data.out_size[1] })
    }
}
impl ID2D1ComputeTransform_Impl for ComputeXform_Impl {
    fn SetComputeInfo(&self, info: Option<&ID2D1ComputeInfo>) -> windows::core::Result<()> {
        let info = info.unwrap();
        unsafe {
            info.SetComputeShader(&self.data.guid)?;
            if !self.data.cbuffer.is_empty() {
                info.SetComputeShaderConstantBuffer(&self.data.cbuffer)?;
            }
        }
        Ok(())
    }
    fn CalculateThreadgroups(&self, _output: *const windows::Win32::Foundation::RECT, dx: *mut u32, dy: *mut u32, dz: *mut u32) -> windows::core::Result<()> {
        unsafe {
            *dx = self.data.groups[0];
            *dy = self.data.groups[1];
            *dz = self.data.groups[2];
        }
        Ok(())
    }
}
unsafe extern "system" fn compute_effect_factory(out: *mut Option<windows::core::IUnknown>) -> windows::core::HRESULT {
    match PENDING_COMPUTE.with(|p| p.borrow().clone()) {
        Some(data) => {
            let effect: ID2D1EffectImpl = ComputeEffect { data }.into();
            match effect.cast::<windows::core::IUnknown>() {
                Ok(unk) => {
                    *out = Some(unk);
                    windows::core::HRESULT(0)
                }
                Err(e) => e.code(),
            }
        }
        None => windows::Win32::Foundation::E_FAIL,
    }
}

// Phase 3C: a custom vertex-shader effect — the VS deforms a vertex-buffer mesh, the PS shades it.
#[derive(Clone)]
struct VertexData {
    vs_guid: windows::core::GUID,
    ps_guid: windows::core::GUID,
    vs_dxbc: Vec<u8>,
    ps_dxbc: Vec<u8>,
    vs_cbuffer: Vec<u8>,
    vertices: Vec<u8>,
    inputs: u32,
    viewport: [i32; 4],
}

thread_local! {
    static PENDING_VERTEX: RefCell<Option<VertexData>> = const { RefCell::new(None) };
}

#[implement(ID2D1EffectImpl)]
struct VertexEffect {
    data: VertexData,
}

impl ID2D1EffectImpl_Impl for VertexEffect_Impl {
    fn Initialize(&self, ctx: Option<&ID2D1EffectContext>, graph: Option<&ID2D1TransformGraph>) -> windows::core::Result<()> {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32_FLOAT;
        let (ctx, graph) = (ctx.unwrap(), graph.unwrap());
        unsafe {
            ctx.LoadVertexShader(&self.data.vs_guid, &self.data.vs_dxbc)?;
            ctx.LoadPixelShader(&self.data.ps_guid, &self.data.ps_dxbc)?;
            let elems = [
                D2D1_INPUT_ELEMENT_DESC { semanticName: windows::core::s!("POSITION"), semanticIndex: 0, format: DXGI_FORMAT_R32G32_FLOAT, inputSlot: 0, alignedByteOffset: 0 },
                D2D1_INPUT_ELEMENT_DESC { semanticName: windows::core::s!("TEXCOORD"), semanticIndex: 0, format: DXGI_FORMAT_R32G32_FLOAT, inputSlot: 0, alignedByteOffset: 8 },
            ];
            let props = D2D1_VERTEX_BUFFER_PROPERTIES { inputCount: 1, usage: D2D1_VERTEX_USAGE_STATIC, data: self.data.vertices.as_ptr(), byteWidth: self.data.vertices.len() as u32 };
            let custom = D2D1_CUSTOM_VERTEX_BUFFER_PROPERTIES { shaderBufferWithInputSignature: self.data.vs_dxbc.as_ptr(), shaderBufferSize: self.data.vs_dxbc.len() as u32, inputElements: elems.as_ptr(), elementCount: 2, stride: 16 };
            // Keyed find-or-create (native pattern): bind the found buffer so the deform is
            // byte-exact while exercising CreateVertexBuffer(keyed) and FindVertexBuffer.
            let key = derive_resource_key(&self.data.vs_guid, 0);
            let vb = match ctx.FindVertexBuffer(&key) {
                Ok(v) => v,
                Err(_) => ctx.CreateVertexBuffer(&props, Some(&key), Some(&custom))?,
            };
            let xform: ID2D1DrawTransform = VertexTransform { data: self.data.clone(), vb }.into();
            let node: ID2D1TransformNode = xform.cast()?;
            graph.SetSingleTransformNode(&node)
        }
    }
    fn PrepareForRender(&self, _change: D2D1_CHANGE_TYPE) -> windows::core::Result<()> {
        Ok(())
    }
    fn SetGraph(&self, _graph: Option<&ID2D1TransformGraph>) -> windows::core::Result<()> {
        Err(windows::Win32::Foundation::E_NOTIMPL.into())
    }
}

#[implement(ID2D1DrawTransform)]
struct VertexTransform {
    data: VertexData,
    vb: ID2D1VertexBuffer,
}

impl ID2D1TransformNode_Impl for VertexTransform_Impl {
    fn GetInputCount(&self) -> u32 {
        self.data.inputs
    }
}

impl ID2D1Transform_Impl for VertexTransform_Impl {
    fn MapOutputRectToInputRects(&self, output: *const windows::Win32::Foundation::RECT, inputs: *mut windows::Win32::Foundation::RECT, count: u32) -> windows::core::Result<()> {
        unsafe {
            for i in 0..count as usize {
                *inputs.add(i) = *output;
            }
        }
        Ok(())
    }
    fn MapInputRectsToOutputRect(&self, _inputs: *const windows::Win32::Foundation::RECT, _opaque: *const windows::Win32::Foundation::RECT, _count: u32, output: *mut windows::Win32::Foundation::RECT, output_opaque: *mut windows::Win32::Foundation::RECT) -> windows::core::Result<()> {
        let v = self.data.viewport;
        unsafe {
            *output = windows::Win32::Foundation::RECT { left: v[0], top: v[1], right: v[2], bottom: v[3] };
            *output_opaque = windows::Win32::Foundation::RECT { left: 0, top: 0, right: 0, bottom: 0 };
        }
        Ok(())
    }
    fn MapInvalidRect(&self, _index: u32, invalid: &windows::Win32::Foundation::RECT) -> windows::core::Result<windows::Win32::Foundation::RECT> {
        Ok(*invalid)
    }
}

impl ID2D1DrawTransform_Impl for VertexTransform_Impl {
    fn SetDrawInfo(&self, info: Option<&ID2D1DrawInfo>) -> windows::core::Result<()> {
        let info = info.unwrap();
        unsafe {
            info.SetVertexProcessing(&self.vb, D2D1_VERTEX_OPTIONS_NONE, None, None, Some(&self.data.vs_guid))?;
            if !self.data.vs_cbuffer.is_empty() {
                info.SetVertexShaderConstantBuffer(&self.data.vs_cbuffer)?;
            }
            info.SetPixelShader(&self.data.ps_guid, D2D1_PIXEL_OPTIONS_NONE)?;
            for i in 0..self.data.inputs {
                info.SetInputDescription(i, D2D1_INPUT_DESCRIPTION { filter: D2D1_FILTER_MIN_MAG_MIP_POINT, levelOfDetailCount: 0 })?;
            }
        }
        Ok(())
    }
}

unsafe extern "system" fn vertex_effect_factory(out: *mut Option<windows::core::IUnknown>) -> windows::core::HRESULT {
    match PENDING_VERTEX.with(|p| p.borrow().clone()) {
        Some(data) => {
            let effect: ID2D1EffectImpl = VertexEffect { data }.into();
            match effect.cast::<windows::core::IUnknown>() {
                Ok(unk) => {
                    *out = Some(unk);
                    windows::core::HRESULT(0)
                }
                Err(e) => e.code(),
            }
        }
        None => windows::Win32::Foundation::E_FAIL,
    }
}

// All per-scene resources and the device context. Methods take `&mut self` so
// `command_list` can replay nested ops that define and use further resources.
struct Runner<'a> {
    dc: &'a ID2D1DeviceContext,
    rt: ID2D1RenderTarget,
    factory: &'a ID2D1Factory,
    brushes: HashMap<String, ID2D1Brush>,
    geometries: HashMap<String, ID2D1Geometry>,
    strokes: HashMap<String, ID2D1StrokeStyle>,
    bitmaps: HashMap<String, ID2D1Bitmap1>,
    images: HashMap<String, ID2D1Image>,
    effects: HashMap<String, ID2D1Effect>, // ffirects: effect objects by id (for the rect queries)
    states: HashMap<String, ID2D1DrawingStateBlock>,
    inks: HashMap<String, ID2D1Ink>,
    ink_styles: HashMap<String, ID2D1InkStyle>,
    meshes: HashMap<String, ID2D1GradientMesh>,
    batches: HashMap<String, ID2D1SpriteBatch>,
    svgs: HashMap<String, ID2D1SvgDocument>,
    svg_elems: HashMap<String, ID2D1SvgElement>,
    image_sources: HashMap<String, ID2D1ImageSource>,
    color_contexts: HashMap<String, ID2D1ColorContext>,
    wic_sources: HashMap<String, IWICBitmapSource>, // IWICBitmap kept for the BitmapSource effect property
    realizations: HashMap<String, ID2D1GeometryRealization>, // 1C-7 geometry realizations
    text_factory: Option<IDWriteFactory>,
    wic: Option<IWICImagingFactory>,
    queries: Vec<serde_json::Value>,
}

struct Recorded {
    ops: Vec<serde_json::Value>,
    queries: Vec<serde_json::Value>,
    end_draw: serde_json::Value,
    ok: bool,
}

impl<'a> Runner<'a> {
    unsafe fn new(dc: &'a ID2D1DeviceContext, factory: &'a ID2D1Factory) -> Result<Self> {
        Ok(Self {
            rt: dc.cast()?,
            dc,
            factory,
            brushes: HashMap::new(),
            geometries: HashMap::new(),
            strokes: HashMap::new(),
            bitmaps: HashMap::new(),
            images: HashMap::new(),
            effects: HashMap::new(),
            states: HashMap::new(),
            inks: HashMap::new(),
            ink_styles: HashMap::new(),
            meshes: HashMap::new(),
            batches: HashMap::new(),
            svgs: HashMap::new(),
            svg_elems: HashMap::new(),
            image_sources: HashMap::new(),
            color_contexts: HashMap::new(),
            wic_sources: HashMap::new(),
            realizations: HashMap::new(),
            text_factory: None,
            wic: None,
            queries: Vec::new(),
        })
    }

    unsafe fn wic(&mut self) -> Result<&IWICImagingFactory> {
        if self.wic.is_none() {
            self.wic = Some(CoCreateInstance(
                &CLSID_WICImagingFactory,
                None,
                CLSCTX_INPROC_SERVER,
            )?);
        }
        Ok(self.wic.as_ref().unwrap())
    }

    // Lazily create the DirectWrite factory and pin the DC's text rendering params (gamma 2.2,
    // contrast 1, ClearType 1, flat geometry, default mode) — identical to the original inline init.
    unsafe fn text_factory(&mut self) -> Result<&IDWriteFactory> {
        if self.text_factory.is_none() {
            let factory: IDWriteFactory = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED)?;
            let params = factory.CreateCustomRenderingParams(
                2.2,
                1.0,
                1.0,
                DWRITE_PIXEL_GEOMETRY_FLAT,
                DWRITE_RENDERING_MODE_DEFAULT,
            )?;
            self.dc.SetTextRenderingParams(&params);
            self.text_factory = Some(factory);
        }
        Ok(self.text_factory.as_ref().unwrap())
    }

    unsafe fn stops(&self, stops: &[Stop], gamma: u32, extend: u32) -> Result<ID2D1GradientStopCollection> {
        let stops: Vec<_> = stops
            .iter()
            .map(|s| D2D1_GRADIENT_STOP {
                position: s.position,
                color: color(s.color),
            })
            .collect();
        Ok(self.rt.CreateGradientStopCollection(
            &stops,
            D2D1_GAMMA(gamma as i32),
            D2D1_EXTEND_MODE(extend as i32),
        )?)
    }

    unsafe fn op(&mut self, i: usize, op: &Op) -> Result<()> {
        match op {
            Op::Clear { color: c } => self.dc.Clear(Some(&color(*c))),
            Op::Transform { matrix: m } => self.dc.SetTransform(&matrix(*m)),
            Op::Antialias { mode } => self.dc.SetAntialiasMode(D2D1_ANTIALIAS_MODE(*mode as i32)),
            Op::Blend { mode } => self.dc.SetPrimitiveBlend(D2D1_PRIMITIVE_BLEND(*mode as i32)),
            Op::UnitMode { mode } => self.dc.SetUnitMode(D2D1_UNIT_MODE(*mode as i32)),
            Op::SetTags { tag1, tag2 } => self.dc.SetTags(*tag1, *tag2),
            Op::Solid {
                id,
                color: c,
                opacity,
                transform,
            } => {
                let bp = brush_props(*opacity, *transform);
                let brush = self
                    .dc
                    .CreateSolidColorBrush(&color(*c), bp.as_ref().map(std::ptr::from_ref))?;
                put(&mut self.brushes, id, brush.cast()?)?;
            }
            Op::Linear {
                id,
                start,
                end,
                stops,
                gamma,
                extend,
                opacity,
                transform,
            } => {
                let stops = self.stops(stops, *gamma, *extend)?;
                let bp = brush_props(*opacity, *transform);
                let brush = self.dc.CreateLinearGradientBrush(
                    &D2D1_LINEAR_GRADIENT_BRUSH_PROPERTIES {
                        startPoint: point(*start),
                        endPoint: point(*end),
                    },
                    bp.as_ref().map(std::ptr::from_ref),
                    &stops,
                )?;
                put(&mut self.brushes, id, brush.cast()?)?;
            }
            Op::Radial {
                id,
                center,
                offset,
                radius,
                stops,
                gamma,
                extend,
                opacity,
                transform,
            } => {
                let stops = self.stops(stops, *gamma, *extend)?;
                let bp = brush_props(*opacity, *transform);
                let brush = self.dc.CreateRadialGradientBrush(
                    &D2D1_RADIAL_GRADIENT_BRUSH_PROPERTIES {
                        center: point(*center),
                        gradientOriginOffset: point(*offset),
                        radiusX: radius[0],
                        radiusY: radius[1],
                    },
                    bp.as_ref().map(std::ptr::from_ref),
                    &stops,
                )?;
                put(&mut self.brushes, id, brush.cast()?)?;
            }
            Op::Bitmap { id, size, rgba, rgba_f32, data: raw, format, alpha, dpi } => {
                // Raw-float texels upload verbatim as R32G32B32A32_FLOAT (coords/HDR outside [0,1]);
                // otherwise the straight-RGBA8 path at the requested format.
                let (fmt, am, bpp, data) = if !raw.is_empty() {
                    let (fmt, am, bpp) = pixel_fmt(*format, *alpha);
                    (fmt, am, bpp, raw.clone())
                } else if !rgba_f32.is_empty() {
                    let (_, am, _) = pixel_fmt(*format, *alpha);
                    if *format == Some(DXGI_FORMAT_R16G16B16A16_FLOAT.0 as u32) {
                        // format 10: the same raw texels as half floats (HDR in an R16 bitmap).
                        (DXGI_FORMAT_R16G16B16A16_FLOAT, am, 8usize, rgba_f32.iter().flat_map(|&v| f32_to_f16(v).to_le_bytes()).collect::<Vec<u8>>())
                    } else {
                        (DXGI_FORMAT_R32G32B32A32_FLOAT, am, 16usize, rgba_f32.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())
                    }
                } else {
                    let (fmt, am, bpp) = pixel_fmt(*format, *alpha);
                    (fmt, am, bpp, encode_bitmap(rgba, fmt, am))
                };
                let bitmap = self.dc.CreateBitmap(
                    D2D_SIZE_U {
                        width: size[0],
                        height: size[1],
                    },
                    Some(data.as_ptr().cast()),
                    size[0] * bpp as u32,
                    &bitmap_properties_fmt(dpi.unwrap_or(96.), D2D1_BITMAP_OPTIONS_NONE, fmt, am),
                )?;
                put(&mut self.images, id, bitmap.cast()?)?;
                put(&mut self.bitmaps, id, bitmap)?;
            }
            Op::WicBitmap { id, size, rgba, format, alpha } => {
                // The WIC source stays premultiplied BGRA8; D2D converts it to the requested bitmap
                // format (incl. float) on CreateBitmapFromWicBitmap.
                let (fmt, am, _) = pixel_fmt(*format, *alpha);
                let bgra = premultiply_bgra(rgba);
                let src = self.wic()?.CreateBitmapFromMemory(
                    size[0],
                    size[1],
                    &GUID_WICPixelFormat32bppPBGRA,
                    size[0] * 4,
                    &bgra,
                )?;
                let bitmap = self
                    .dc
                    .CreateBitmapFromWicBitmap(&src, Some(&bitmap_properties_fmt(96., D2D1_BITMAP_OPTIONS_NONE, fmt, am)))?;
                put(&mut self.images, id, bitmap.cast()?)?;
                put(&mut self.bitmaps, id, bitmap)?;
                // Keep the IWICBitmap so a BitmapSource effect can bind it as its WicBitmapSource.
                self.wic_sources.insert(id.clone(), src.cast()?);
            }
            Op::BitmapBrush {
                id,
                bitmap,
                extend,
                interpolation,
                matrix: m,
                opacity,
            } => {
                let brush = self.dc.CreateBitmapBrush(
                    get(&self.bitmaps, bitmap)?,
                    Some(&D2D1_BITMAP_BRUSH_PROPERTIES1 {
                        extendModeX: D2D1_EXTEND_MODE(extend[0] as i32),
                        extendModeY: D2D1_EXTEND_MODE(extend[1] as i32),
                        interpolationMode: D2D1_INTERPOLATION_MODE(*interpolation as i32),
                    }),
                    Some(&D2D1_BRUSH_PROPERTIES {
                        opacity: *opacity,
                        transform: matrix(*m),
                    }),
                )?;
                put(&mut self.brushes, id, brush.cast()?)?;
            }
            Op::ImageBrush {
                id,
                image,
                source,
                extend,
                interpolation,
                opacity,
                matrix: m,
            } => {
                let brush = self.dc.CreateImageBrush(
                    get(&self.images, image)?,
                    &D2D1_IMAGE_BRUSH_PROPERTIES {
                        sourceRectangle: rect(*source),
                        extendModeX: D2D1_EXTEND_MODE(extend[0] as i32),
                        extendModeY: D2D1_EXTEND_MODE(extend[1] as i32),
                        interpolationMode: D2D1_INTERPOLATION_MODE(*interpolation as i32),
                    },
                    Some(&D2D1_BRUSH_PROPERTIES {
                        opacity: *opacity,
                        transform: matrix(*m),
                    }),
                )?;
                put(&mut self.brushes, id, brush.cast()?)?;
            }
            Op::Geometry { id, shape } => {
                let g = geometry(self.factory, shape)?;
                put(&mut self.geometries, id, g)?;
            }
            Op::GeometryGroup { id, fill, geometries } => {
                let children: Vec<Option<ID2D1Geometry>> =
                    geometries.iter().map(|g| get(&self.geometries, g).cloned().map(Some)).collect::<Result<_>>()?;
                let group = self.factory.CreateGeometryGroup(D2D1_FILL_MODE(*fill as i32), &children)?;
                put(&mut self.geometries, id, group.cast()?)?;
            }
            Op::GeometryTransform { id, geometry: src, matrix: m } => {
                let g = get(&self.geometries, src)?;
                let tg = self.factory.CreateTransformedGeometry(g, &matrix(*m))?;
                put(&mut self.geometries, id, tg.cast()?)?;
            }
            Op::Primitive {
                shape,
                brush,
                width,
            } => {
                let brush = get(&self.brushes, brush)?;
                match shape {
                    Shape::Rect { rect: r } => {
                        if let Some(w) = width {
                            self.dc.DrawRectangle(&rect(*r), brush, *w, None);
                        } else {
                            self.dc.FillRectangle(&rect(*r), brush);
                        }
                    }
                    Shape::Ellipse { center, radius } => {
                        let e = D2D1_ELLIPSE {
                            point: point(*center),
                            radiusX: radius[0],
                            radiusY: radius[1],
                        };
                        if let Some(w) = width {
                            self.dc.DrawEllipse(&e, brush, *w, None);
                        } else {
                            self.dc.FillEllipse(&e, brush);
                        }
                    }
                    Shape::Rounded { rect: r, radius } => {
                        let r = D2D1_ROUNDED_RECT {
                            rect: rect(*r),
                            radiusX: radius[0],
                            radiusY: radius[1],
                        };
                        if let Some(w) = width {
                            self.dc.DrawRoundedRectangle(&r, brush, *w, None);
                        } else {
                            self.dc.FillRoundedRectangle(&r, brush);
                        }
                    }
                    Shape::Path { .. } => {
                        // A primitive with a path shape: build the native geometry and fill/stroke it,
                        // so the oracle captures native's real pixels (not a harness bail) — the fuzz
                        // differential needs these ops driven, matching conformance drive().
                        let g = geometry(self.factory, shape)?;
                        if let Some(w) = width {
                            self.dc.DrawGeometry(&g, brush, *w, None);
                        } else {
                            self.dc.FillGeometry(&g, brush, None);
                        }
                    }
                }
            }
            Op::Stroke {
                id,
                caps,
                join,
                miter,
                offset,
                dashes,
                transform_type,
                dash_style,
            } => {
                // A named D2D1_DASH_STYLE (scene `dash_style` 1..=4) uses its predefined pattern and
                // ignores the dashes array (null dash_arg); 0 falls back to CUSTOM (with the array) or
                // SOLID (empty). A dashes array is only valid with CUSTOM, else D2D returns E_INVALIDARG.
                let dash_style = if *dash_style != 0 {
                    D2D1_DASH_STYLE(*dash_style as i32)
                } else if dashes.is_empty() {
                    D2D1_DASH_STYLE_SOLID
                } else {
                    D2D1_DASH_STYLE_CUSTOM
                };
                let dash_arg = if dash_style == D2D1_DASH_STYLE_CUSTOM { Some(dashes.as_slice()) } else { None };
                // transform_type 0 (normal) keeps the base `CreateStrokeStyle` so v1/v2 captures are
                // byte-identical; fixed/hairline need `ID2D1StrokeStyle1` via `CreateStrokeStyle1`.
                let style: ID2D1StrokeStyle = if *transform_type == 0 {
                    self.factory.CreateStrokeStyle(
                        &D2D1_STROKE_STYLE_PROPERTIES {
                            startCap: D2D1_CAP_STYLE(caps[0] as i32),
                            endCap: D2D1_CAP_STYLE(caps[1] as i32),
                            dashCap: D2D1_CAP_STYLE(caps[2] as i32),
                            lineJoin: D2D1_LINE_JOIN(*join as i32),
                            miterLimit: *miter,
                            dashStyle: dash_style,
                            dashOffset: *offset,
                        },
                        dash_arg,
                    )?
                } else {
                    let f1: ID2D1Factory1 = self.factory.cast()?;
                    f1.CreateStrokeStyle(
                        &D2D1_STROKE_STYLE_PROPERTIES1 {
                            startCap: D2D1_CAP_STYLE(caps[0] as i32),
                            endCap: D2D1_CAP_STYLE(caps[1] as i32),
                            dashCap: D2D1_CAP_STYLE(caps[2] as i32),
                            lineJoin: D2D1_LINE_JOIN(*join as i32),
                            miterLimit: *miter,
                            dashStyle: dash_style,
                            dashOffset: *offset,
                            transformType: D2D1_STROKE_TRANSFORM_TYPE(*transform_type as i32),
                        },
                        dash_arg,
                    )?
                    .cast()?
                };
                put(&mut self.strokes, id, style)?;
            }
            Op::Fill { geometry, brush } => {
                // A geometry whose creation native rejected (e.g. invalid arc radius) is undefined;
                // native's Fill on it is a no-op, so skip (mirror drive()), don't record an error.
                if let Some(g) = self.geometries.get(geometry) {
                    self.dc.FillGeometry(g, get(&self.brushes, brush)?, None);
                }
            }
            Op::DrawLine { p0, p1, brush, width } => {
                self.dc.DrawLine(point(*p0), point(*p1), get(&self.brushes, brush)?, *width, None);
            }
            Op::CopyFromBitmap { id, size, src, bgra, point, rect: rc } => {
                let s: ID2D1Bitmap = get(&self.bitmaps, src)?.cast()?;
                let pf = s.GetPixelFormat();
                let (fmt, am) = if *bgra { (DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED) } else { (pf.format, pf.alphaMode) };
                let bpp = match fmt.0 { 2 => 16, 10 | 11 => 8, 49 => 2, 61 | 65 => 1, _ => 4 };
                let zeros = vec![0u8; size[0] as usize * size[1] as usize * bpp];
                let d = self.dc.CreateBitmap(
                    D2D_SIZE_U { width: size[0], height: size[1] },
                    Some(zeros.as_ptr().cast()),
                    size[0] * bpp as u32,
                    &bitmap_properties_fmt(96., D2D1_BITMAP_OPTIONS_NONE, fmt, am),
                )?;
                put(&mut self.images, id, d.cast()?)?;
                put(&mut self.bitmaps, id, d.clone())?;
                let p = point.map(|p| D2D_POINT_2U { x: p[0], y: p[1] });
                let r = rc.map(|r| D2D_RECT_U { left: r[0], top: r[1], right: r[2], bottom: r[3] });
                d.CopyFromBitmap(p.as_ref().map(|p| p as *const _), &s, r.as_ref().map(|r| r as *const _))?;
            }
            Op::Readback { rect: rc } => {
                let sz = self.rt.GetPixelSize();
                let pf = self.rt.GetPixelFormat();
                let bpp = match pf.format.0 { 2 => 16, 10 | 11 => 8, _ => 4 };
                let r = rc.unwrap_or([0, 0, sz.width, sz.height]);
                let (w, h) = (r[2].saturating_sub(r[0]).max(1), r[3].saturating_sub(r[1]).max(1));
                let bm = self.dc.CreateBitmap(
                    D2D_SIZE_U { width: w, height: h },
                    None,
                    0,
                    &bitmap_properties_fmt(96., D2D1_BITMAP_OPTIONS_CPU_READ | D2D1_BITMAP_OPTIONS_CANNOT_DRAW, pf.format, pf.alphaMode),
                )?;
                let dst = D2D_POINT_2U { x: 0, y: 0 };
                let src = D2D_RECT_U { left: r[0], top: r[1], right: r[2], bottom: r[3] };
                let copied = bm.CopyFromRenderTarget(Some(&dst), &self.rt, rc.map(|_| &src as *const _));
                let hr = hr_str(copied.clone());
                let mut bytes = Vec::new();
                if copied.is_ok() {
                    let m = bm.Map(D2D1_MAP_OPTIONS_READ)?;
                    for row in 0..h as usize {
                        bytes.extend_from_slice(std::slice::from_raw_parts(m.bits.add(row * m.pitch as usize), w as usize * bpp));
                    }
                    bm.Unmap()?;
                }
                self.queries.push(serde_json::json!({"op": i, "kind": "readback", "result": {"hr": hr, "size": [w, h], "bytes": bytes}}));
            }
            Op::SetDpi { dpi } => self.dc.SetDpi(dpi[0], dpi[1]),
            Op::RenderingControls { precision, tile } => {
                let rc = D2D1_RENDERING_CONTROLS {
                    bufferPrecision: D2D1_BUFFER_PRECISION(*precision as i32),
                    tileSize: D2D_SIZE_U { width: tile[0], height: tile[1] },
                };
                self.dc.SetRenderingControls(&rc);
            }
            Op::CopyFromRtTest { rect: r, fill } => {
                // Draw a rect, capture the whole frame via CopyFromRenderTarget into a bitmap, clear,
                // then DrawBitmap the capture back 1:1 — the output is the captured frame.
                let sz = self.rt.GetPixelSize();
                let brush = self.dc.CreateSolidColorBrush(&color(*fill), None)?;
                self.dc.FillRectangle(&rect(*r), &brush);
                let bmp = self.dc.CreateBitmap(sz, None, 0, &bitmap_properties_fmt(96.0, D2D1_BITMAP_OPTIONS_NONE, DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED))?;
                bmp.CopyFromRenderTarget(None, &self.rt, None)?;
                self.dc.Clear(Some(&color([1.0, 1.0, 1.0, 1.0])));
                self.dc.DrawBitmap(&bmp, None, 1.0, D2D1_INTERPOLATION_MODE_NEAREST_NEIGHBOR, None, None);
            }
            Op::GdiMetafileProbe {} => {
                use windows::Win32::Graphics::Gdi::{CloseEnhMetaFile, CreateEnhMetaFileW, DeleteEnhMetaFile, GetEnhMetaFileBits, Rectangle};
                // Build a tiny EMF in memory (draw one rectangle), then hand its bytes to
                // ID2D1Factory::CreateGdiMetafile and record what native/ours return.
                let emf_dc = CreateEnhMetaFileW(None, windows::core::PCWSTR::null(), None, windows::core::PCWSTR::null());
                let _ = Rectangle(emf_dc, 10, 10, 50, 50);
                let hemf = CloseEnhMetaFile(emf_dc);
                let n = GetEnhMetaFileBits(hemf, None);
                let mut bytes = vec![0u8; n as usize];
                GetEnhMetaFileBits(hemf, Some(&mut bytes));
                let _ = DeleteEnhMetaFile(hemf);
                // Raw ENHMETAHEADER rects: rclBounds (device units) at offset 8, rclFrame (0.01mm) at 24.
                let rd4 = |o: usize| -> [i32; 4] { std::array::from_fn(|k| i32::from_le_bytes(bytes[o + k * 4..o + k * 4 + 4].try_into().unwrap())) };
                let rd2 = |o: usize| -> [i32; 2] { std::array::from_fn(|k| i32::from_le_bytes(bytes[o + k * 4..o + k * 4 + 4].try_into().unwrap())) };
                let (rcl_bounds, rcl_frame) = (rd4(8), rd4(24));
                // szlDevice (px) at 72, szlMillimeters (mm) at 80 — the reference-resolution source.
                let (szl_device, szl_mm) = (rd2(72), rd2(80));
                let stream = SHCreateMemStream(Some(&bytes)).context("SHCreateMemStream failed")?;
                let factory1: ID2D1Factory1 = self.factory.cast()?;
                let (hr, bounds, src_bounds, dpi) = match factory1.CreateGdiMetafile(&stream) {
                    Ok(mf) => {
                        let b = mf.GetBounds().unwrap_or_default();
                        let sb: D2D_RECT_F = {
                            let m1: windows::Win32::Graphics::Direct2D::ID2D1GdiMetafile1 = mf.cast()?;
                            m1.GetSourceBounds().unwrap_or_default()
                        };
                        let (mut dx, mut dy) = (0.0f32, 0.0f32);
                        let m1: windows::Win32::Graphics::Direct2D::ID2D1GdiMetafile1 = mf.cast()?;
                        let _ = m1.GetDpi(&mut dx, &mut dy);
                        ("S_OK".to_string(), [b.left, b.top, b.right, b.bottom], [sb.left, sb.top, sb.right, sb.bottom], [dx, dy])
                    }
                    Err(e) => (hr_str(Err(e)), [0.0; 4], [0.0; 4], [0.0; 2]),
                };
                self.queries.push(serde_json::json!({"op": i, "kind": "gdi_metafile", "result": {
                    "emf_bytes": bytes.len(), "create_hr": hr, "bounds": bounds, "source_bounds": src_bounds, "dpi": dpi,
                    "rcl_bounds": rcl_bounds, "rcl_frame": rcl_frame, "szl_device": szl_device, "szl_mm": szl_mm,
                }}));
            }
            Op::DrawMetafile { file, bytes } => {
                let bytes =
                    if !bytes.is_empty() { bytes.clone() } else { std::fs::read(file).with_context(|| format!("read metafile {file}"))? };
                let stream = SHCreateMemStream(Some(&bytes)).context("SHCreateMemStream failed")?;
                let factory1: ID2D1Factory1 = self.factory.cast()?;
                let mf = factory1.CreateGdiMetafile(&stream)?;
                self.dc.DrawGdiMetafile(&mf, None);
            }
            Op::CompatibleRenderTarget { size, bg, rect: r, fill, dest } => {
                use windows::Win32::Graphics::Direct2D::{D2D1_BITMAP_INTERPOLATION_MODE_LINEAR, D2D1_COMPATIBLE_RENDER_TARGET_OPTIONS_NONE};
                let sz = D2D_SIZE_U { width: size[0], height: size[1] };
                let crt = self.rt.CreateCompatibleRenderTarget(None, Some(&sz), None, D2D1_COMPATIBLE_RENDER_TARGET_OPTIONS_NONE)?;
                crt.BeginDraw();
                crt.Clear(Some(&color(*bg)));
                let b = crt.CreateSolidColorBrush(&color(*fill), None)?;
                crt.FillRectangle(&rect(*r), &b);
                crt.EndDraw(None, None)?;
                let bmp = crt.GetBitmap()?;
                // 1:1 identity blit (no resample) so the round-trip is pixel-exact.
                let dr = D2D_RECT_F { left: dest[0], top: dest[1], right: dest[0] + size[0] as f32, bottom: dest[1] + size[1] as f32 };
                self.rt.DrawBitmap(&bmp, Some(&dr), 1.0, D2D1_BITMAP_INTERPOLATION_MODE_LINEAR, None);
            }
            Op::HwndRenderTarget { size, resize, color: c } => {
                use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
                use windows::Win32::System::LibraryLoader::GetModuleHandleW;
                use windows::Win32::UI::WindowsAndMessaging::{CreateWindowExW, DefWindowProcW, DestroyWindow, RegisterClassW, WINDOW_EX_STYLE, WNDCLASSW, WS_POPUP};
                extern "system" fn wndproc(h: HWND, m: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
                    unsafe { DefWindowProcW(h, m, wp, lp) }
                }
                let hinst = windows::Win32::Foundation::HINSTANCE(GetModuleHandleW(None)?.0);
                let cls = windows::core::w!("d2dref_hwndrt");
                let _ = RegisterClassW(&WNDCLASSW { lpfnWndProc: Some(wndproc), hInstance: hinst, lpszClassName: cls, ..Default::default() });
                let hwnd = CreateWindowExW(WINDOW_EX_STYLE(0), cls, cls, WS_POPUP, 0, 0, size[0] as i32, size[1] as i32, None, None, hinst, None)?;
                let rtp = windows::Win32::Graphics::Direct2D::D2D1_RENDER_TARGET_PROPERTIES::default();
                let hp = D2D1_HWND_RENDER_TARGET_PROPERTIES { hwnd, pixelSize: D2D_SIZE_U { width: size[0], height: size[1] }, presentOptions: D2D1_PRESENT_OPTIONS_NONE };
                let rt = self.factory.CreateHwndRenderTarget(&rtp, &hp)?;
                let hwnd_match = (rt.GetHwnd() == hwnd) as u32;
                let state = rt.CheckWindowState().0;
                rt.BeginDraw();
                rt.Clear(Some(&color(*c)));
                let end_hr = hr_str(rt.EndDraw(None, None));
                let resize_hr = hr_str(rt.Resize(&D2D_SIZE_U { width: resize[0], height: resize[1] }));
                let sz = rt.GetPixelSize();
                let _ = DestroyWindow(hwnd);
                self.queries.push(serde_json::json!({"op": i, "kind": "hwnd_rt", "result": {
                    "hwnd_match": hwnd_match, "window_state": state, "end_hr": end_hr, "resize_hr": resize_hr,
                    "size_after": [sz.width, sz.height],
                }}));
            }
            Op::GdiInteropDraw { size, gdi_color, rect } => {
                use windows::Win32::Graphics::Gdi::{CreateSolidBrush, DeleteObject, FillRect, HGDIOBJ};
                use windows::Win32::Graphics::Imaging::{GUID_WICPixelFormat32bppPBGRA, WICBitmapCacheOnLoad};
                let wicbmp = self.wic()?.CreateBitmap(size[0], size[1], &GUID_WICPixelFormat32bppPBGRA, WICBitmapCacheOnLoad)?;
                let props = windows::Win32::Graphics::Direct2D::D2D1_RENDER_TARGET_PROPERTIES {
                    pixelFormat: windows::Win32::Graphics::Direct2D::Common::D2D1_PIXEL_FORMAT {
                        format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM,
                        alphaMode: windows::Win32::Graphics::Direct2D::Common::D2D1_ALPHA_MODE_PREMULTIPLIED,
                    },
                    usage: windows::Win32::Graphics::Direct2D::D2D1_RENDER_TARGET_USAGE_GDI_COMPATIBLE,
                    ..Default::default()
                };
                let rt = self.factory.CreateWicBitmapRenderTarget(&wicbmp, &props)?;
                rt.BeginDraw();
                rt.Clear(Some(&color([1.0, 1.0, 1.0, 1.0])));
                let gi: ID2D1GdiInteropRenderTarget = rt.cast()?;
                let hdc = gi.GetDC(D2D1_DC_INITIALIZE_MODE_COPY)?;
                let brush = CreateSolidBrush(windows::Win32::Foundation::COLORREF(gdi_color[0] as u32 | (gdi_color[1] as u32) << 8 | (gdi_color[2] as u32) << 16));
                let r = windows::Win32::Foundation::RECT { left: rect[0], top: rect[1], right: rect[2], bottom: rect[3] };
                FillRect(hdc, &r, brush);
                let _ = DeleteObject(HGDIOBJ(brush.0));
                gi.ReleaseDC(None)?;
                rt.EndDraw(None, None)?;
                let mut buf = vec![0u8; (size[0] * size[1] * 4) as usize];
                wicbmp.CopyPixels(std::ptr::null(), size[0] * 4, &mut buf)?;
                let c = ((size[1] / 2 * size[0] + size[0] / 2) * 4) as usize;
                self.queries.push(serde_json::json!({"op": i, "kind": "gdi_interop", "result": {
                    "center": [buf[c], buf[c + 1], buf[c + 2], buf[c + 3]], "corner": [buf[0], buf[1], buf[2], buf[3]],
                }}));
            }
            Op::DcRenderTarget { size, fill_color } => {
                use windows::Win32::Graphics::Gdi::{CreateCompatibleDC, CreateDIBSection, DeleteDC, DeleteObject, GdiFlush, SelectObject, BITMAPINFO, BITMAPINFOHEADER, BI_RGB, DIB_RGB_COLORS, HDC, HGDIOBJ};
                let (w, h) = (size[0] as i32, size[1] as i32);
                let hdc = CreateCompatibleDC(HDC::default());
                let mut bmi = BITMAPINFO::default();
                bmi.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
                bmi.bmiHeader.biWidth = w;
                bmi.bmiHeader.biHeight = -h; // top-down
                bmi.bmiHeader.biPlanes = 1;
                bmi.bmiHeader.biBitCount = 32;
                bmi.bmiHeader.biCompression = BI_RGB.0;
                let mut bits: *mut core::ffi::c_void = std::ptr::null_mut();
                let hbm = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, &mut bits, windows::Win32::Foundation::HANDLE::default(), 0)?;
                let old = SelectObject(hdc, HGDIOBJ(hbm.0));
                // A DC render target must be DXGI_FORMAT_B8G8R8A8_UNORM; the default UNKNOWN is rejected.
                let props = windows::Win32::Graphics::Direct2D::D2D1_RENDER_TARGET_PROPERTIES {
                    pixelFormat: windows::Win32::Graphics::Direct2D::Common::D2D1_PIXEL_FORMAT {
                        format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM,
                        alphaMode: windows::Win32::Graphics::Direct2D::Common::D2D1_ALPHA_MODE_PREMULTIPLIED,
                    },
                    ..Default::default()
                };
                let rt = self.factory.CreateDCRenderTarget(&props)?;
                let rect = windows::Win32::Foundation::RECT { left: 0, top: 0, right: w, bottom: h };
                rt.BindDC(hdc, &rect)?;
                rt.BeginDraw();
                rt.Clear(Some(&color([1.0, 1.0, 1.0, 1.0])));
                let brush = rt.CreateSolidColorBrush(&color(*fill_color), None)?;
                rt.FillRectangle(&D2D_RECT_F { left: 2.0, top: 2.0, right: w as f32 - 2.0, bottom: h as f32 - 2.0 }, &brush);
                rt.EndDraw(None, None)?;
                let _ = GdiFlush();
                let buf = std::slice::from_raw_parts(bits as *const u8, (w * h * 4) as usize).to_vec();
                SelectObject(hdc, old);
                let _ = DeleteObject(HGDIOBJ(hbm.0));
                let _ = DeleteDC(hdc);
                // Reject path: the default UNKNOWN pixel format must give D2DERR_UNSUPPORTED_PIXEL_FORMAT.
                let bad = self.factory.CreateDCRenderTarget(&windows::Win32::Graphics::Direct2D::D2D1_RENDER_TARGET_PROPERTIES::default());
                let bad_hr = bad.err().map(|e| format!("0x{:08X}", e.code().0 as u32)).unwrap_or_else(|| "S_OK".into());
                let c = ((h / 2 * w + w / 2) * 4) as usize;
                self.queries.push(serde_json::json!({"op": i, "kind": "dc_rt", "result": {
                    "center": [buf[c], buf[c + 1], buf[c + 2], buf[c + 3]], "corner": [buf[0], buf[1], buf[2], buf[3]],
                    "unknown_format_hr": bad_hr,
                }}));
            }
            Op::WicRenderTarget { size, fill_color } => {
                use windows::Win32::Graphics::Imaging::{GUID_WICPixelFormat32bppPBGRA, WICBitmapCacheOnLoad};
                let wicbmp = self.wic()?.CreateBitmap(size[0], size[1], &GUID_WICPixelFormat32bppPBGRA, WICBitmapCacheOnLoad)?;
                let props = windows::Win32::Graphics::Direct2D::D2D1_RENDER_TARGET_PROPERTIES::default();
                let rt = self.factory.CreateWicBitmapRenderTarget(&wicbmp, &props)?;
                let supported = rt.IsSupported(&props).as_bool();
                rt.BeginDraw();
                rt.Clear(Some(&color([1.0, 1.0, 1.0, 1.0])));
                let brush = rt.CreateSolidColorBrush(&color(*fill_color), None)?;
                rt.FillRectangle(&D2D_RECT_F { left: 2.0, top: 2.0, right: size[0] as f32 - 2.0, bottom: size[1] as f32 - 2.0 }, &brush);
                rt.EndDraw(None, None)?;
                let mut buf = vec![0u8; (size[0] * size[1] * 4) as usize];
                wicbmp.CopyPixels(std::ptr::null(), size[0] * 4, &mut buf)?;
                let c = ((size[1] / 2 * size[0] + size[0] / 2) * 4) as usize;
                self.queries.push(serde_json::json!({"op": i, "kind": "wic_rt", "result": {
                    "supported": supported as u32,
                    "center": [buf[c], buf[c + 1], buf[c + 2], buf[c + 3]],
                    "corner": [buf[0], buf[1], buf[2], buf[3]],
                }}));
            }
            Op::FillMesh { triangles, brush } => {
                let mesh = self.dc.CreateMesh()?;
                let sink = mesh.Open()?;
                let tris: Vec<D2D1_TRIANGLE> = triangles
                    .iter()
                    .map(|t| D2D1_TRIANGLE { point1: point(t[0]), point2: point(t[1]), point3: point(t[2]) })
                    .collect();
                sink.AddTriangles(&tris);
                sink.Close()?;
                self.dc.FillMesh(&mesh, get(&self.brushes, brush)?)
            }
            Op::Draw {
                geometry,
                brush,
                width,
                style,
            } => {
                // As Op::Fill: an undefined (native-rejected) geometry draws nothing, so skip.
                if let Some(g) = self.geometries.get(geometry) {
                    let style = style.as_ref().map(|s| get(&self.strokes, s)).transpose()?;
                    self.dc.DrawGeometry(g, get(&self.brushes, brush)?, *width, style);
                }
            }
            Op::DrawBitmap {
                bitmap,
                dest,
                source,
                opacity,
                interpolation,
                overload,
            } => {
                if overload.as_deref() == Some("rt") {
                    // ID2D1RenderTarget::DrawBitmap (5-arg, D2D1_BITMAP_INTERPOLATION_MODE).
                    self.rt.DrawBitmap(
                        get(&self.bitmaps, bitmap)?,
                        Some(&rect(*dest)),
                        *opacity,
                        D2D1_BITMAP_INTERPOLATION_MODE(*interpolation as i32),
                        Some(&rect(*source)),
                    )
                } else {
                    self.dc.DrawBitmap(
                        get(&self.bitmaps, bitmap)?,
                        Some(&rect(*dest)),
                        *opacity,
                        D2D1_INTERPOLATION_MODE(*interpolation as i32),
                        Some(&rect(*source)),
                        None,
                    )
                }
            }
            Op::PushClip { rect: r, antialias } => self
                .dc
                .PushAxisAlignedClip(&rect(*r), D2D1_ANTIALIAS_MODE(*antialias as i32)),
            Op::PopClip => self.dc.PopAxisAlignedClip(),
            Op::PushLayer {
                bounds,
                mask,
                opacity,
                opacity_brush,
                mask_antialias,
                mask_transform,
                layer_options,
            } => {
                let mask = mask
                    .as_ref()
                    .map(|s| get(&self.geometries, s).cloned())
                    .transpose()?;
                let brush = opacity_brush
                    .as_ref()
                    .map(|s| get(&self.brushes, s).cloned())
                    .transpose()?;
                let mut params = D2D1_LAYER_PARAMETERS1 {
                    contentBounds: rect(*bounds),
                    geometricMask: ManuallyDrop::new(mask),
                    maskAntialiasMode: D2D1_ANTIALIAS_MODE(
                        mask_antialias.unwrap_or(D2D1_ANTIALIAS_MODE_PER_PRIMITIVE.0 as u32) as i32,
                    ),
                    maskTransform: mask_transform.map(matrix).unwrap_or_else(identity),
                    opacity: *opacity,
                    opacityBrush: ManuallyDrop::new(brush),
                    layerOptions: D2D1_LAYER_OPTIONS1(layer_options.unwrap_or(0) as i32),
                };
                self.dc.PushLayer(&params, None);
                ManuallyDrop::drop(&mut params.geometricMask);
                ManuallyDrop::drop(&mut params.opacityBrush);
            }
            Op::PopLayer => self.dc.PopLayer(),
            Op::FillOpacityMask {
                bitmap,
                brush,
                content,
                dest,
                source,
                overload,
            } => {
                if overload.as_deref() == Some("dc") {
                    // ID2D1DeviceContext::FillOpacityMask (4-arg, no content).
                    self.dc.FillOpacityMask(
                        get(&self.bitmaps, bitmap)?,
                        get(&self.brushes, brush)?,
                        dest.map(|r| rect(r)).as_ref().map(std::ptr::from_ref),
                        source.map(|r| rect(r)).as_ref().map(std::ptr::from_ref),
                    )
                } else {
                    self.rt.FillOpacityMask(
                        get(&self.bitmaps, bitmap)?,
                        get(&self.brushes, brush)?,
                        D2D1_OPACITY_MASK_CONTENT(*content as i32),
                        dest.map(|r| rect(r)).as_ref().map(std::ptr::from_ref),
                        source.map(|r| rect(r)).as_ref().map(std::ptr::from_ref),
                    )
                }
            }
            Op::Effect {
                id,
                clsid,
                inputs,
                properties,
            } => {
                let guid = GUID::from_u128(
                    u128::from_str_radix(&clsid.replace('-', ""), 16)
                        .context("invalid effect CLSID")?,
                );
                let effect = self.dc.CreateEffect(&guid)?;
                effect.SetInputCount(inputs.len().try_into()?)?;
                for (n, input) in inputs.iter().enumerate() {
                    effect.SetInput(n as u32, get(&self.images, input)?, true);
                }
                for p in properties {
                    if let Value::Iunknown(oid) = &p.value {
                        // An interface-pointer property: a colour context (ColorManagement) or an
                        // IWICBitmapSource (BitmapSource's WicBitmapSource). Resolve by which map holds it.
                        if let Some(cc) = self.color_contexts.get(oid) {
                            let ptr = (cc.as_raw() as usize as u64).to_le_bytes();
                            effect
                                .SetValue(p.index, D2D1_PROPERTY_TYPE_COLOR_CONTEXT, &ptr)
                                .with_context(|| format!("effect property {}", p.index))?;
                        } else {
                            let ptr = (get(&self.wic_sources, oid)?.as_raw() as usize as u64).to_le_bytes();
                            effect
                                .SetValue(p.index, D2D1_PROPERTY_TYPE_IUNKNOWN, &ptr)
                                .with_context(|| format!("effect property {}", p.index))?;
                        }
                    } else {
                        let (kind, bytes) = property(&p.value);
                        effect
                            .SetValue(p.index, kind, &bytes)
                            .with_context(|| format!("effect property {}", p.index))?;
                    }
                }
                put(&mut self.images, id, effect.GetOutput()?)?;
                self.effects.insert(id.clone(), effect); // ffirects: keep the effect for the rect queries
            }
            Op::SetProperty { target, index, value } => {
                // Change an effect property between draws (ID2D1Properties::SetValue).
                let effect = get(&self.effects, target)?;
                let (kind, bytes) = property(value);
                effect.SetValue(*index, kind, &bytes).with_context(|| format!("set_property {} index {}", target, index))?;
            }
            Op::SetInput { target, index, input } => {
                // Rebind an effect's input between draws (ID2D1Effect::SetInput).
                let effect = get(&self.effects, target)?;
                effect.SetInput(*index, get(&self.images, input)?, true);
            }
            Op::ColorContext { id, space, profile, dxgi, simple } => {
                let cc: ID2D1ColorContext = match space.as_str() {
                    "srgb" => self.dc.CreateColorContext(D2D1_COLOR_SPACE_SRGB, None)?,
                    "scrgb" => self.dc.CreateColorContext(D2D1_COLOR_SPACE_SCRGB, None)?,
                    "custom" => self.dc.CreateColorContext(D2D1_COLOR_SPACE_CUSTOM, Some(profile))?,
                    "dxgi" => self
                        .dc5()?
                        .CreateColorContextFromDxgiColorSpace(DXGI_COLOR_SPACE_TYPE(
                            dxgi.context("dxgi color context without a color space")? as i32,
                        ))?
                        .cast()?,
                    "simple" => {
                        let s = simple.as_ref().context("simple color context without a profile")?;
                        let p = D2D1_SIMPLE_COLOR_PROFILE {
                            redPrimary: point(s.red),
                            greenPrimary: point(s.green),
                            bluePrimary: point(s.blue),
                            whitePointXZ: point(s.white),
                            gamma: D2D1_GAMMA1(s.gamma as i32),
                        };
                        let cc1: ID2D1ColorContext1 = self.dc5()?.CreateColorContextFromSimpleColorProfile(&p)?;
                        // ID2D1ColorContext1::GetSimpleColorProfile — read the profile back.
                        let mut got = D2D1_SIMPLE_COLOR_PROFILE::default();
                        cc1.GetSimpleColorProfile(&mut got)?;
                        self.queries.push(serde_json::json!({"op": i, "kind": "simple_color_profile", "result": {
                            "red": [got.redPrimary.x, got.redPrimary.y], "green": [got.greenPrimary.x, got.greenPrimary.y],
                            "blue": [got.bluePrimary.x, got.bluePrimary.y], "white": [got.whitePointXZ.x, got.whitePointXZ.y],
                            "gamma": got.gamma.0,
                        }}));
                        cc1.cast()?
                    }
                    other => anyhow::bail!("unknown color context space {other}"),
                };
                put(&mut self.color_contexts, id, cc)?;
            }
            Op::RegisterShaderEffect { clsid, dxbc, cbuffer, cbuffer_bytes, inputs, filter, resource_textures, precision, input_inflate, pixel_options } => {
                let guid = GUID::from_u128(
                    u128::from_str_radix(&clsid.replace('-', ""), 16).context("invalid shader effect CLSID")?,
                );
                let cb: Vec<u8> = if !cbuffer_bytes.is_empty() {
                    cbuffer_bytes.clone()
                } else {
                    cbuffer.iter().flat_map(|f| f.to_le_bytes()).collect()
                };
                PENDING_SHADER.with(|p| {
                    *p.borrow_mut() = Some(ShaderData { guid, dxbc: dxbc.clone(), cbuffer: cb, inputs: *inputs, filter: D2D1_FILTER(*filter as i32), resource_textures: resource_textures.clone(), precision: *precision, input_inflate: *input_inflate, pixel_options: *pixel_options });
                });
                let xml = HSTRING::from(shader_effect_xml(*inputs));
                let factory1: ID2D1Factory1 = self.factory.cast()?;
                factory1.RegisterEffectFromString(&guid, &xml, None, Some(shader_effect_factory))?;
            }
            Op::RegisterGraphEffect { clsid, offset, border, bounds } => {
                let guid = GUID::from_u128(u128::from_str_radix(&clsid.replace('-', ""), 16).context("invalid graph effect CLSID")?);
                PENDING_GRAPH.with(|p| *p.borrow_mut() = Some(GraphData { offset: *offset, border: *border, bounds: *bounds }));
                let xml = HSTRING::from(shader_effect_xml(1));
                let factory1: ID2D1Factory1 = self.factory.cast()?;
                factory1.RegisterEffectFromString(&guid, &xml, None, Some(graph_effect_factory))?;
            }
            Op::RegisterComputeEffect { clsid, dxbc, cbuffer, groups, out_size } => {
                let guid = GUID::from_u128(u128::from_str_radix(&clsid.replace('-', ""), 16).context("invalid compute effect CLSID")?);
                PENDING_COMPUTE.with(|p| *p.borrow_mut() = Some(ComputeData { guid, dxbc: dxbc.clone(), cbuffer: cbuffer.clone(), groups: *groups, out_size: *out_size }));
                let xml = HSTRING::from(shader_effect_xml(0));
                let factory1: ID2D1Factory1 = self.factory.cast()?;
                factory1.RegisterEffectFromString(&guid, &xml, None, Some(compute_effect_factory))?;
            }
            Op::RegisterBlendEffect { clsid, desc, factor } => {
                let guid = GUID::from_u128(u128::from_str_radix(&clsid.replace('-', ""), 16).context("invalid blend effect CLSID")?);
                PENDING_BLEND.with(|p| *p.borrow_mut() = Some(BlendData { desc: *desc, factor: *factor }));
                let xml = HSTRING::from(shader_effect_xml(2));
                let factory1: ID2D1Factory1 = self.factory.cast()?;
                factory1.RegisterEffectFromString(&guid, &xml, None, Some(blend_effect_factory))?;
            }
            Op::RegisterVertexEffect { clsid, vs_dxbc, ps_dxbc, vs_cbuffer, vertices, inputs, viewport } => {
                let n = u128::from_str_radix(&clsid.replace('-', ""), 16).context("invalid vertex effect CLSID")?;
                let vs_cb: Vec<u8> = vs_cbuffer.iter().flat_map(|f| f.to_le_bytes()).collect();
                let verts: Vec<u8> = vertices.iter().flat_map(|f| f.to_le_bytes()).collect();
                PENDING_VERTEX.with(|p| {
                    *p.borrow_mut() = Some(VertexData {
                        vs_guid: GUID::from_u128(n ^ 1),
                        ps_guid: GUID::from_u128(n ^ 2),
                        vs_dxbc: vs_dxbc.clone(),
                        ps_dxbc: ps_dxbc.clone(),
                        vs_cbuffer: vs_cb,
                        vertices: verts,
                        inputs: *inputs,
                        viewport: *viewport,
                    });
                });
                let xml = HSTRING::from(shader_effect_xml(*inputs));
                let factory1: ID2D1Factory1 = self.factory.cast()?;
                factory1.RegisterEffectFromString(&GUID::from_u128(n), &xml, None, Some(vertex_effect_factory))?;
            }
            Op::LookupTable3D {
                id,
                input,
                extents,
                data,
                precision,
            } => {
                let effect = self.dc.CreateEffect(&CLSID_D2D1LookupTable3D)?;
                effect.SetInput(0, get(&self.images, input)?, true);
                // Encode the f32 `data` into the table's native byte layout at `precision` (RGBA, R
                // fastest, tightly packed). bpe = bytes per entry (4 channels).
                let prec = precision.unwrap_or(5);
                let (bpe, buf): (u32, Vec<u8>) = match prec {
                    1 => (4, data.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect()),
                    3 => (8, data.iter().flat_map(|&v| ((v.clamp(0.0, 1.0) * 65535.0).round() as u16).to_le_bytes()).collect()),
                    _ => (16, data.iter().flat_map(|&v| v.to_le_bytes()).collect()),
                };
                let d2d_prec = D2D1_BUFFER_PRECISION(prec as i32);
                let strides = [extents[0] * bpe, extents[0] * extents[1] * bpe];
                let bytes = &buf[..];
                let dc2: ID2D1DeviceContext2 = self.dc.cast()?;
                let lut = dc2.CreateLookupTable3D(d2d_prec, extents, bytes, &strides)?;
                // The LUT property (index 0) is an IUnknown: pass the interface pointer's bytes.
                let ptr = (lut.as_raw() as usize).to_ne_bytes();
                effect.SetValue(0, D2D1_PROPERTY_TYPE_IUNKNOWN, &ptr)?;
                put(&mut self.images, id, effect.GetOutput()?)?;
            }
            Op::DrawImage {
                image,
                offset,
                interpolation,
                composite,
                image_rect,
            } => {
                let ir = image_rect.map(rect);
                self.dc.DrawImage(
                    get(&self.images, image)?,
                    Some(&point(*offset)),
                    ir.as_ref().map(std::ptr::from_ref),
                    D2D1_INTERPOLATION_MODE(*interpolation as i32),
                    D2D1_COMPOSITE_MODE(*composite as i32),
                )
            }
            Op::CommandList { id, ops } => {
                let list = self.dc.CreateCommandList()?;
                let prev = self.dc.GetTarget()?;
                self.dc.SetTarget(&list);
                let result = (|| -> Result<()> {
                    for (j, nested) in ops.iter().enumerate() {
                        self.dc.SetTags(j as u64, 0);
                        self.op(j, nested)?;
                    }
                    Ok(())
                })();
                self.dc.SetTarget(&prev);
                result?;
                list.Close()?;
                put(&mut self.images, id, list.cast()?)?;
            }
            Op::InkStyle {
                id,
                nib_shape,
                nib_transform,
            } => {
                // Create with default properties, then drive the nib through the setters
                // (SetNibShape/SetNibTransform) — same resulting style as constructing with the
                // properties, so every ink scene byte-identically exercises the setter path too.
                let style = self.dc2()?.CreateInkStyle(None)?;
                style.SetNibShape(D2D1_INK_NIB_SHAPE(*nib_shape as i32));
                style.SetNibTransform(&matrix(*nib_transform));
                put(&mut self.ink_styles, id, style)?;
            }
            Op::Ink {
                id,
                start,
                segments,
            } => {
                let ink = self.dc2()?.CreateInk(&ink_point(*start))?;
                let segs: Vec<_> = segments
                    .iter()
                    .map(|s| D2D1_INK_BEZIER_SEGMENT {
                        point1: ink_point(s.p1),
                        point2: ink_point(s.p2),
                        point3: ink_point(s.end),
                    })
                    .collect();
                ink.AddSegments(&segs)?;
                put(&mut self.inks, id, ink)?;
            }
            Op::InkRemoveSegmentsAtEnd { ink, count } => {
                get(&self.inks, ink)?.RemoveSegmentsAtEnd(*count)?;
            }
            Op::DrawInk { ink, brush, style } => {
                let style = style.as_ref().map(|s| get(&self.ink_styles, s)).transpose()?;
                self.dc2()?
                    .DrawInk(get(&self.inks, ink)?, get(&self.brushes, brush)?, style);
            }
            Op::GradientMesh { id, patches } => {
                let patches: Vec<_> = patches
                    .iter()
                    .map(|p| {
                        let pt = |n: usize| point(p.points[n]);
                        let col = |n: usize| color(p.colors[n]);
                        let edge = |n: usize| D2D1_PATCH_EDGE_MODE(p.extend[n] as i32);
                        D2D1_GRADIENT_MESH_PATCH {
                            point00: pt(0),
                            point01: pt(1),
                            point02: pt(2),
                            point03: pt(3),
                            point10: pt(4),
                            point11: pt(5),
                            point12: pt(6),
                            point13: pt(7),
                            point20: pt(8),
                            point21: pt(9),
                            point22: pt(10),
                            point23: pt(11),
                            point30: pt(12),
                            point31: pt(13),
                            point32: pt(14),
                            point33: pt(15),
                            color00: col(0),
                            color03: col(1),
                            color30: col(2),
                            color33: col(3),
                            // extend is [left, top, right, bottom] per-edge modes.
                            leftEdgeMode: edge(0),
                            topEdgeMode: edge(1),
                            rightEdgeMode: edge(2),
                            bottomEdgeMode: edge(3),
                        }
                    })
                    .collect();
                let mesh = self.dc2()?.CreateGradientMesh(&patches)?;
                put(&mut self.meshes, id, mesh)?;
            }
            Op::DrawGradientMesh { mesh } => self.dc2()?.DrawGradientMesh(get(&self.meshes, mesh)?),
            Op::SpriteBatch { id, sprites } => {
                let batch = self.dc3()?.CreateSpriteBatch()?;
                for s in sprites {
                    let dest = rect(s.dest);
                    let src = s.source.map(rect_u);
                    let col = s.color.map(color);
                    let xf = s.transform.map(matrix);
                    batch.AddSprites(
                        1,
                        &dest,
                        src.as_ref().map(std::ptr::from_ref),
                        col.as_ref().map(std::ptr::from_ref),
                        xf.as_ref().map(std::ptr::from_ref),
                        std::mem::size_of::<D2D_RECT_F>() as u32,
                        std::mem::size_of::<D2D_RECT_U>() as u32,
                        std::mem::size_of::<D2D1_COLOR_F>() as u32,
                        std::mem::size_of::<Matrix3x2>() as u32,
                    )?;
                }
                // Exercise SetSprites by re-setting the first sprite with its own values (an identical
                // batch), so the sprite scenes cover the update path, not only AddSprites.
                if let Some(s) = sprites.first() {
                    let dest = rect(s.dest);
                    let src = s.source.map(rect_u);
                    let col = s.color.map(color);
                    let xf = s.transform.map(matrix);
                    batch.SetSprites(
                        0,
                        1,
                        Some(&dest),
                        src.as_ref().map(std::ptr::from_ref),
                        col.as_ref().map(std::ptr::from_ref),
                        xf.as_ref().map(std::ptr::from_ref),
                        std::mem::size_of::<D2D_RECT_F>() as u32,
                        std::mem::size_of::<D2D_RECT_U>() as u32,
                        std::mem::size_of::<D2D1_COLOR_F>() as u32,
                        std::mem::size_of::<Matrix3x2>() as u32,
                    )?;
                }
                put(&mut self.batches, id, batch)?;
            }
            Op::DrawSpriteBatch {
                batch,
                bitmap,
                interpolation,
                options,
            } => {
                let b = get(&self.batches, batch)?;
                self.dc3()?.DrawSpriteBatch(
                    b,
                    0,
                    b.GetSpriteCount(),
                    get(&self.bitmaps, bitmap)?,
                    D2D1_BITMAP_INTERPOLATION_MODE(*interpolation as i32),
                    D2D1_SPRITE_OPTIONS(*options as i32),
                );
            }
            Op::SvgDocument { id, xml, viewport } => {
                let stream = SHCreateMemStream(Some(xml.as_bytes()))
                    .context("SHCreateMemStream failed")?;
                let doc = self.dc5()?.CreateSvgDocument(
                    &stream,
                    D2D_SIZE_F {
                        width: viewport[0],
                        height: viewport[1],
                    },
                )?;
                put(&mut self.svgs, id, doc)?;
            }
            Op::DrawSvgDocument { doc } => self.dc5()?.DrawSvgDocument(get(&self.svgs, doc)?),
            Op::SvgGetRoot { doc, elem } => {
                let root = get(&self.svgs, doc)?.GetRoot()?;
                self.svg_elems.insert(elem.clone(), root);
            }
            Op::SvgCreateChild { parent, tag, elem } => {
                let child = get(&self.svg_elems, parent)?.CreateChild(&HSTRING::from(tag.as_str()))?;
                self.svg_elems.insert(elem.clone(), child);
            }
            Op::SvgSetAttr { elem, name, value } => {
                get(&self.svg_elems, elem)?.SetAttributeValue3(
                    &HSTRING::from(name.as_str()),
                    D2D1_SVG_ATTRIBUTE_STRING_TYPE_SVG,
                    &HSTRING::from(value.as_str()),
                )?;
            }
            Op::SvgFirstChild { parent, elem } => {
                let c = get(&self.svg_elems, parent)?.GetFirstChild()?;
                self.svg_elems.insert(elem.clone(), c);
            }
            Op::SvgSetText { elem, text } => {
                let u16s: Vec<u16> = text.encode_utf16().collect();
                get(&self.svg_elems, elem)?.SetTextValue(&u16s)?;
            }
            Op::SvgRemoveAttr { elem, name } => {
                get(&self.svg_elems, elem)?.RemoveAttribute(&HSTRING::from(name.as_str()))?;
            }
            Op::SvgSetAttrPod { elem, name, pod, floats } => {
                let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
                get(&self.svg_elems, elem)?.SetAttributeValue2(
                    &HSTRING::from(name.as_str()),
                    D2D1_SVG_ATTRIBUTE_POD_TYPE(*pod as i32),
                    bytes.as_ptr().cast(),
                    bytes.len() as u32,
                )?;
            }
            Op::SvgSetObject { elem, name, object, mutate } => {
                let e = get(&self.svg_elems, elem)?;
                let doc = e.GetDocument()?;
                // `mutate`: build the object with a SCRAMBLED value (different from `object`, one extra
                // element), then drive the setters to reach `object` — so an input-ignoring setter
                // leaves the scrambled value and diverges from the native oracle. Else: net no-op
                // re-set (value unchanged, setters still covered).
                let m = *mutate;
                let attr: ID2D1SvgAttribute = match object {
                    SvgObject::Paint { paint_type, color, id } => {
                        let c = color.map(|c| D2D1_COLOR_F { r: c[0], g: c[1], b: c[2], a: c[3] });
                        let init = if m { D2D1_SVG_PAINT_TYPE(if *paint_type == 1 { 0 } else { 1 }) } else { D2D1_SVG_PAINT_TYPE(*paint_type as i32) };
                        let init_id = if m { HSTRING::new() } else { HSTRING::from(id.as_str()) };
                        let p = doc.CreatePaint(init, c.as_ref().map(|c| c as *const _), &init_id)?;
                        p.SetPaintType(D2D1_SVG_PAINT_TYPE(*paint_type as i32))?;
                        p.SetId(&HSTRING::from(id.as_str()))?;
                        p.cast()?
                    }
                    SvgObject::Points { points } => {
                        let pts: Vec<D2D_POINT_2F> = points.iter().map(|p| point(*p)).collect();
                        if m {
                            let mut init = vec![D2D_POINT_2F { x: 1.0, y: 1.0 }; pts.len() + 1];
                            let pc = doc.CreatePointCollection(Some(&init))?;
                            pc.UpdatePoints(&pts, 0)?; // overwrite the scrambled points with the real ones
                            pc.RemovePointsAtEnd(1)?;  // drop the extra point
                            init.clear();
                            pc.cast()?
                        } else {
                            let pc = doc.CreatePointCollection(Some(&pts))?;
                            pc.UpdatePoints(&pts, 0)?;
                            pc.RemovePointsAtEnd(0)?;
                            pc.cast()?
                        }
                    }
                    SvgObject::Path { commands, data } => {
                        let cmds: Vec<D2D1_SVG_PATH_COMMAND> = commands.iter().map(|c| D2D1_SVG_PATH_COMMAND(*c as i32)).collect();
                        if m {
                            // Scramble: ones for data + one extra HorizAbs command (1 data), then fix up.
                            let mut idata: Vec<f32> = vec![1.0; data.len() + 1];
                            let mut icmds = cmds.clone();
                            icmds.push(D2D1_SVG_PATH_COMMAND_HORIZONTAL_ABSOLUTE);
                            let pd = doc.CreatePathData(Some(&idata), Some(&icmds))?;
                            pd.UpdateSegmentData(data, 0)?;   // overwrite scrambled data with the real data
                            pd.UpdateCommands(&cmds, 0)?;     // re-set the real commands
                            pd.RemoveCommandsAtEnd(1)?;       // drop the extra HorizAbs
                            pd.RemoveSegmentDataAtEnd(1)?;    // drop its extra datum
                            idata.clear();
                            pd.cast()?
                        } else {
                            let pd = doc.CreatePathData(Some(data), Some(&cmds))?;
                            pd.UpdateSegmentData(data, 0)?;
                            pd.UpdateCommands(&cmds, 0)?;
                            pd.RemoveSegmentDataAtEnd(0)?;
                            pd.RemoveCommandsAtEnd(0)?;
                            pd.cast()?
                        }
                    }
                    SvgObject::Dashes { dashes } => {
                        let ds: Vec<D2D1_SVG_LENGTH> =
                            dashes.iter().map(|v| D2D1_SVG_LENGTH { value: *v, units: D2D1_SVG_LENGTH_UNITS_NUMBER }).collect();
                        if m {
                            let init = vec![D2D1_SVG_LENGTH { value: 1.0, units: D2D1_SVG_LENGTH_UNITS_NUMBER }; ds.len() + 1];
                            let da = doc.CreateStrokeDashArray(Some(&init))?;
                            da.UpdateDashes2(dashes, 0)?;  // overwrite the scrambled dashes
                            da.UpdateDashes(&ds, 0)?;      // re-set via the length form too
                            da.RemoveDashesAtEnd(1)?;      // drop the extra dash
                            da.cast()?
                        } else {
                            let da = doc.CreateStrokeDashArray(Some(&ds))?;
                            da.UpdateDashes(&ds, 0)?;
                            da.UpdateDashes2(dashes, 0)?;
                            da.RemoveDashesAtEnd(0)?;
                            da.cast()?
                        }
                    }
                };
                e.SetAttributeValue(&HSTRING::from(name.as_str()), &attr)?;
                // SvgAttribute base: the attribute now belongs to `e`; Clone returns an equal copy.
                let _ = attr.GetElement();
                let _ = attr.Clone();
            }
            Op::SvgAppendChild { parent, child } => {
                let c = get(&self.svg_elems, child)?;
                get(&self.svg_elems, parent)?.AppendChild(c)?;
            }
            Op::SvgInsertChildBefore { parent, child, reference } => {
                let (c, r) = (get(&self.svg_elems, child)?, get(&self.svg_elems, reference)?);
                get(&self.svg_elems, parent)?.InsertChildBefore(c, r)?;
            }
            Op::SvgRemoveChild { parent, child } => {
                let c = get(&self.svg_elems, child)?;
                get(&self.svg_elems, parent)?.RemoveChild(c)?;
            }
            Op::SvgReplaceChild { parent, new, old } => {
                let (n, o) = (get(&self.svg_elems, new)?, get(&self.svg_elems, old)?);
                get(&self.svg_elems, parent)?.ReplaceChild(n, o)?;
            }
            Op::SvgFindById { doc, id, elem } => {
                let e = get(&self.svgs, doc)?.FindElementById(&HSTRING::from(id.as_str()))?;
                self.svg_elems.insert(elem.clone(), e);
            }
            Op::SvgSetViewport { doc, size } => {
                get(&self.svgs, doc)?.SetViewportSize(D2D_SIZE_F { width: size[0], height: size[1] })?;
            }
            Op::SvgDeserialize { doc, xml, elem } => {
                let stream = SHCreateMemStream(Some(xml.as_bytes())).context("SHCreateMemStream failed")?;
                let e = get(&self.svgs, doc)?.Deserialize(&stream)?;
                self.svg_elems.insert(elem.clone(), e);
            }
            Op::SvgSetRoot { doc, elem } => {
                let e = get(&self.svg_elems, elem)?;
                get(&self.svgs, doc)?.SetRoot(e)?;
            }
            // ID2D1SvgGlyphStyle round-trip (FFI-only value holder). Read defaults after create, then
            // set a red fill + a blue stroke (width 2.5, dashes [1,2,3], offset 0.5) and read them all
            // back. One composite query result (queries are matched by op index). Fixed values so the
            // FFI DLL's getters are compared to native's byte-for-byte.
            Op::SvgGlyphStyleProbe => {
                let gs = self.dc4()?.CreateSvgGlyphStyle()?;
                let has_fill0 = unsafe { gs.GetFill() }.is_ok() as u32 as f32;
                let mut w0 = 0f32;
                unsafe { gs.GetStroke(None, Some(&mut w0), None, None) };
                let cnt0 = unsafe { gs.GetStrokeDashesCount() } as f32;
                let fb = self.dc.CreateSolidColorBrush(&color([1.0, 0.0, 0.0, 1.0]), None)?;
                unsafe { gs.SetFill(&fb) }?;
                let sb = self.dc.CreateSolidColorBrush(&color([0.0, 0.0, 1.0, 1.0]), None)?;
                unsafe { gs.SetStroke(&sb, 2.5, Some(&[1.0, 2.0, 3.0]), 0.5) }?;
                let fc = unsafe { gs.GetFill() }?.cast::<ID2D1SolidColorBrush>()?.GetColor();
                let cnt = unsafe { gs.GetStrokeDashesCount() };
                let mut sbrush: Option<ID2D1Brush> = None;
                let (mut w, mut off) = (0f32, 0f32);
                let mut dashes = vec![0f32; cnt as usize];
                unsafe { gs.GetStroke(Some(&mut sbrush), Some(&mut w), Some(&mut dashes), Some(&mut off)) };
                let sc = sbrush.unwrap().cast::<ID2D1SolidColorBrush>()?.GetColor();
                self.queries.push(serde_json::json!({"op": i, "kind": "glyphstyle", "result": [
                    has_fill0, cnt0, cnt as f32,
                    fc.r, fc.g, fc.b, fc.a, sc.r, sc.g, sc.b, sc.a, w0, w, off,
                    dashes[0], dashes[1], dashes[2]
                ]}));
            }
            Op::SvgQuery { kind, doc, elem, name, index, pod } => {
                let elem_of = |r: &Self| -> Result<ID2D1SvgElement> { get(&r.svg_elems, elem.as_deref().unwrap_or("")).cloned() };
                let name = || HSTRING::from(name.as_deref().unwrap_or(""));
                // Tag name of an element, "" for a null (navigation off the end / root's parent).
                let tag_of = |e: &ID2D1SvgElement| -> String {
                    let len = unsafe { e.GetTagNameLength() };
                    let mut buf = vec![0u16; len as usize + 1];
                    let _ = unsafe { e.GetTagName(&mut buf) };
                    String::from_utf16_lossy(&buf[..len as usize])
                };
                let result = match kind.as_str() {
                    "serialize" => {
                        let d = get(&self.svgs, doc.as_deref().unwrap_or(""))?;
                        let stream = SHCreateMemStream(None).context("SHCreateMemStream failed")?;
                        d.Serialize(&stream, None)?;
                        serde_json::json!(stream_to_string(&stream)?)
                    }
                    "attr" => {
                        let e = elem_of(self)?;
                        let len = e.GetAttributeValueLength(&name(), D2D1_SVG_ATTRIBUTE_STRING_TYPE_SVG)?;
                        let mut buf = vec![0u16; len as usize + 1];
                        e.GetAttributeValue3(&name(), D2D1_SVG_ATTRIBUTE_STRING_TYPE_SVG, &mut buf)?;
                        serde_json::json!(String::from_utf16_lossy(&buf[..len as usize]))
                    }
                    "attr_pod" => {
                        let e = elem_of(self)?;
                        let t = D2D1_SVG_ATTRIBUTE_POD_TYPE(pod.unwrap_or(0) as i32);
                        let size = pod_size(t);
                        let mut buf = vec![0u8; size];
                        e.GetAttributeValue2(&name(), t, buf.as_mut_ptr().cast(), size as u32)?;
                        serde_json::json!(buf.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<u32>>())
                    }
                    "count" => serde_json::json!(elem_of(self)?.GetSpecifiedAttributeCount()),
                    "name" => {
                        let e = elem_of(self)?;
                        let i = index.unwrap_or(0);
                        let mut len = 0u32;
                        e.GetSpecifiedAttributeNameLength(i, &mut len, None)?;
                        let mut buf = vec![0u16; len as usize + 1];
                        e.GetSpecifiedAttributeName(i, &mut buf, None)?;
                        serde_json::json!(String::from_utf16_lossy(&buf[..len as usize]))
                    }
                    "tag" => {
                        let e = elem_of(self)?;
                        let len = e.GetTagNameLength();
                        let mut buf = vec![0u16; len as usize + 1];
                        e.GetTagName(&mut buf)?;
                        serde_json::json!(String::from_utf16_lossy(&buf[..len as usize]))
                    }
                    "specified" => serde_json::json!(elem_of(self)?.IsAttributeSpecified(&name(), None).as_bool()),
                    "has_children" => serde_json::json!(elem_of(self)?.HasChildren().as_bool()),
                    // DOM navigation: the tag of the element GetParent/GetFirstChild/GetLastChild/
                    // GetNextChild/GetPreviousChild returns ("" past the end), and IsTextContent.
                    "parent_tag" => serde_json::json!(unsafe { elem_of(self)?.GetParent() }.map(|e| tag_of(&e)).unwrap_or_default()),
                    "first_child_tag" => serde_json::json!(unsafe { elem_of(self)?.GetFirstChild() }.map(|e| tag_of(&e)).unwrap_or_default()),
                    "last_child_tag" => serde_json::json!(unsafe { elem_of(self)?.GetLastChild() }.map(|e| tag_of(&e)).unwrap_or_default()),
                    "next_tag" => {
                        let e = elem_of(self)?;
                        let parent = unsafe { e.GetParent() }?;
                        serde_json::json!(unsafe { parent.GetNextChild(&e) }.map(|c| tag_of(&c)).unwrap_or_default())
                    }
                    "prev_tag" => {
                        let e = elem_of(self)?;
                        let parent = unsafe { e.GetParent() }?;
                        serde_json::json!(unsafe { parent.GetPreviousChild(&e) }.map(|c| tag_of(&c)).unwrap_or_default())
                    }
                    "is_text" => serde_json::json!(unsafe { elem_of(self)?.IsTextContent() }.as_bool()),
                    // Object-valued attribute readback (GetAttributeValue → the typed object, then its
                    // getters): SvgPaint, SvgPathData, SvgPointCollection, SvgStrokeDashArray.
                    "paint_type" => {
                        let p: ID2D1SvgPaint = unsafe { elem_of(self)?.GetAttributeValue(&name()) }?;
                        serde_json::json!(unsafe { p.GetPaintType() }.0)
                    }
                    "paint_id" => {
                        let p: ID2D1SvgPaint = unsafe { elem_of(self)?.GetAttributeValue(&name()) }?;
                        let n = unsafe { p.GetIdLength() };
                        let mut b = vec![0u16; n as usize + 1];
                        let _ = unsafe { p.GetId(&mut b) };
                        serde_json::json!(String::from_utf16_lossy(&b[..n as usize]))
                    }
                    "path_cmds" => {
                        let p: ID2D1SvgPathData = unsafe { elem_of(self)?.GetAttributeValue(&name()) }?;
                        let n = unsafe { p.GetCommandsCount() } as usize;
                        let mut b = vec![D2D1_SVG_PATH_COMMAND::default(); n];
                        if n > 0 { let _ = unsafe { p.GetCommands(&mut b, 0) }; }
                        serde_json::json!(b.iter().map(|c| c.0).collect::<Vec<i32>>())
                    }
                    "path_data" => {
                        let p: ID2D1SvgPathData = unsafe { elem_of(self)?.GetAttributeValue(&name()) }?;
                        let n = unsafe { p.GetSegmentDataCount() } as usize;
                        let mut b = vec![0f32; n];
                        if n > 0 { let _ = unsafe { p.GetSegmentData(&mut b, 0) }; }
                        serde_json::json!(b)
                    }
                    "points" => {
                        let p: ID2D1SvgPointCollection = unsafe { elem_of(self)?.GetAttributeValue(&name()) }?;
                        let n = unsafe { p.GetPointsCount() } as usize;
                        let mut b = vec![D2D_POINT_2F::default(); n];
                        if n > 0 { let _ = unsafe { p.GetPoints(&mut b, 0) }; }
                        serde_json::json!(b.iter().flat_map(|v| [v.x, v.y]).collect::<Vec<f32>>())
                    }
                    "dashes" => {
                        let p: ID2D1SvgStrokeDashArray = unsafe { elem_of(self)?.GetAttributeValue(&name()) }?;
                        let n = unsafe { p.GetDashesCount() } as usize;
                        let mut b = vec![0f32; n];
                        if n > 0 { let _ = unsafe { p.GetDashes2(&mut b, 0) }; }
                        serde_json::json!(b)
                    }
                    "viewport" => {
                        let s = unsafe { get(&self.svgs, doc.as_deref().unwrap_or(""))?.GetViewportSize() };
                        serde_json::json!([s.width, s.height])
                    }
                    "text_value" => {
                        let e = elem_of(self)?;
                        let n = unsafe { e.GetTextValueLength() };
                        let mut b = vec![0u16; n as usize + 1];
                        let _ = unsafe { e.GetTextValue(&mut b) };
                        serde_json::json!(String::from_utf16_lossy(&b[..n as usize]))
                    }
                    "find_hr" => {
                        let r = get(&self.svgs, doc.as_deref().unwrap_or(""))?.FindElementById(&name());
                        serde_json::json!(hr_str(r.map(|_| ())))
                    }
                    "attr_hr" => {
                        let r = elem_of(self)?.GetAttributeValueLength(&name(), D2D1_SVG_ATTRIBUTE_STRING_TYPE_SVG);
                        serde_json::json!(hr_str(r.map(|_| ())))
                    }
                    other => anyhow::bail!("unknown svg query kind {other:?}"),
                };
                self.queries.push(serde_json::json!({"op": i, "kind": format!("svg_{kind}"), "result": result}));
            }
            Op::ImageSource { id, size, rgba, loading, alpha } => {
                let bgra = premultiply_bgra(rgba);
                let src = self.wic()?.CreateBitmapFromMemory(
                    size[0],
                    size[1],
                    &GUID_WICPixelFormat32bppPBGRA,
                    size[0] * 4,
                    &bgra,
                )?;
                let image = self.dc2()?.CreateImageSourceFromWic(
                    &src,
                    D2D1_IMAGE_SOURCE_LOADING_OPTIONS(*loading as i32),
                    D2D1_ALPHA_MODE(*alpha as i32),
                )?;
                put(&mut self.images, id, image.cast()?)?;
                put(&mut self.image_sources, id, image.cast()?)?;
            }
            Op::TransformedImageSource { id, source, orientation, scale, interpolation, options } => {
                let image = self.dc2()?.CreateTransformedImageSource(
                    get(&self.image_sources, source)?,
                    &D2D1_TRANSFORMED_IMAGE_SOURCE_PROPERTIES {
                        orientation: D2D1_ORIENTATION(*orientation as i32),
                        scaleX: scale[0],
                        scaleY: scale[1],
                        interpolationMode: D2D1_INTERPOLATION_MODE(*interpolation as i32),
                        options: D2D1_TRANSFORMED_IMAGE_SOURCE_OPTIONS(*options as i32),
                    },
                )?;
                put(&mut self.images, id, image.cast()?)?;
            }
            Op::BlendImage { image, mode, offset, target_rect, interpolation } => {
                let tr = target_rect.map(rect);
                self.dc6()?.BlendImage(
                    get(&self.images, image)?,
                    D2D1_BLEND_MODE(*mode as i32),
                    Some(&point(*offset)),
                    tr.as_ref().map(std::ptr::from_ref),
                    D2D1_INTERPOLATION_MODE(*interpolation as i32),
                );
            }
            Op::MaxScaleFactor { matrix: m } => {
                let v = D2D1ComputeMaximumScaleFactor(&matrix(*m));
                self.queries
                    .push(serde_json::json!({"op": i, "kind": "max_scale", "result": v}));
            }
            Op::SaveState { id } => {
                let block = self.factory.CreateDrawingStateBlock(None, None)?;
                self.dc.SaveDrawingState(&block);
                put(&mut self.states, id, block)?;
            }
            Op::RestoreState { id } => self.dc.RestoreDrawingState(get(&self.states, id)?),
            Op::Query {
                geometry,
                kind,
                point: at,
                tolerance,
                world,
            } => {
                let g = get(&self.geometries, geometry)?;
                let tol = tolerance.unwrap_or(D2D1_DEFAULT_FLATTENING_TOLERANCE);
                let wm = world.map(matrix);
                let wp = || wm.as_ref().map(|x| x as *const _);
                let result = match kind.as_str() {
                    "area" => serde_json::json!(g.ComputeArea(wp(), tol)?),
                    "length" => serde_json::json!(g.ComputeLength(wp(), tol)?),
                    "bounds" => {
                        let b = g.GetBounds(wp())?;
                        serde_json::json!([b.left, b.top, b.right, b.bottom])
                    }
                    "contains_point" => {
                        let p = point(at.context("contains_point requires point")?);
                        serde_json::json!(g.FillContainsPoint(p, wp(), tol)?.as_bool())
                    }
                    // ID2D1PathGeometry::Stream — re-emit the path into a fresh path geometry's sink,
                    // then read back the copy's counts + bounds (equal to the source if Stream is faithful).
                    "stream" => {
                        let src: ID2D1PathGeometry = g.cast()?;
                        let dst = self.factory.CreatePathGeometry()?;
                        let sink = dst.Open()?;
                        src.Stream(&sink)?;
                        sink.Close()?;
                        // The streamed copy is geometrically identical to the source (EQUAL = 4).
                        let dstg: ID2D1Geometry = dst.cast()?;
                        let relation = src.CompareWithGeometry(&dstg, None, tol)?.0;
                        serde_json::json!({
                            "src_figures": src.GetFigureCount()?, "src_segments": src.GetSegmentCount()?,
                            "dst_figures": dst.GetFigureCount()?, "dst_segments": dst.GetSegmentCount()?,
                            "relation": relation,
                        })
                    }
                    other => anyhow::bail!("unknown query kind {other:?}"),
                };
                self.queries
                    .push(serde_json::json!({"op": i, "geometry": geometry, "kind": kind, "result": result}));
            }
            Op::QueryGeom {
                geometry,
                kind,
                point: at,
                length,
                width,
                style,
                other,
                transform,
                tolerance,
                world,
            } => {
                let g = get(&self.geometries, geometry)?;
                let tol = tolerance.unwrap_or(D2D1_DEFAULT_FLATTENING_TOLERANCE);
                let wm = world.map(matrix);
                let wp = || wm.as_ref().map(|x| x as *const _);
                let style_of = |s: &Option<String>| -> Result<Option<ID2D1StrokeStyle>> {
                    s.as_ref().map(|s| get(&self.strokes, s).cloned()).transpose()
                };
                let result = match kind.as_str() {
                    "point_at_length" => {
                        let len = length.context("point_at_length requires length")?;
                        let mut p = D2D_POINT_2F::default();
                        let mut t = D2D_POINT_2F::default();
                        g.ComputePointAtLength(len, wp(), tol, Some(&mut p), Some(&mut t))?;
                        serde_json::json!([p.x, p.y, t.x, t.y])
                    }
                    "stroke_contains" => {
                        let p = point(at.context("stroke_contains requires point")?);
                        let w = width.context("stroke_contains requires width")?;
                        let st = style_of(style)?;
                        serde_json::json!(g.StrokeContainsPoint(p, w, st.as_ref(), wp(), tol)?.as_bool())
                    }
                    "widened_bounds" => {
                        let w = width.context("widened_bounds requires width")?;
                        let st = style_of(style)?;
                        let b = g.GetWidenedBounds(w, st.as_ref(), wp(), tol)?;
                        serde_json::json!([b.left, b.top, b.right, b.bottom])
                    }
                    "compare" => {
                        let o = get(&self.geometries, other.as_ref().context("compare requires other")?)?;
                        let xf = transform.map(matrix);
                        serde_json::json!(g.CompareWithGeometry(o, xf.as_ref().map(|x| x as *const _), tol)?.0)
                    }
                    other => anyhow::bail!("unknown query kind {other:?}"),
                };
                self.queries
                    .push(serde_json::json!({"op": i, "geometry": geometry, "kind": kind, "result": result}));
            }
            Op::GeometryOp {
                id,
                source,
                operation,
                other,
                mode,
                width,
                style,
                transform,
                tolerance,
                world,
            } => {
                let src = get(&self.geometries, source)?.clone();
                let tol = tolerance.unwrap_or(D2D1_DEFAULT_FLATTENING_TOLERANCE);
                let wm = world.map(matrix);
                let wp = || wm.as_ref().map(|x| x as *const _);
                let path = self.factory.CreatePathGeometry()?;
                let sink = path.Open()?;
                let inner: ID2D1SimplifiedGeometrySink = sink.cast()?;
                // Tee the output through RecordingSink: it logs the exact stream and forwards to the
                // path sink (still fillable). Native drives `recorder`; `recorder` drives `inner`.
                let fill_mode = Rc::new(std::cell::Cell::new(0));
                let figures = Rc::new(RefCell::new(Vec::new()));
                let recorder: ID2D1SimplifiedGeometrySink = RecordingSink {
                    inner: inner.clone(),
                    fill_mode: fill_mode.clone(),
                    figures: figures.clone(),
                    cur: RefCell::new(None),
                }
                .into();
                match operation.as_str() {
                    "widen" => {
                        let w = width.context("widen requires width")?;
                        let st = style.as_ref().map(|s| get(&self.strokes, s).cloned()).transpose()?;
                        // Optional worldTransform: captures the device/elliptical-pen widening under a
                        // non-uniform transform (fuzz 5289), not just the local circular widen.
                        let xf = transform.map(matrix);
                        src.Widen(w, st.as_ref(), xf.as_ref().map(|x| x as *const _), tol, &recorder)?;
                    }
                    "outline" => src.Outline(wp(), tol, &recorder)?,
                    "simplify" => src.Simplify(
                        D2D1_GEOMETRY_SIMPLIFICATION_OPTION(mode.unwrap_or(0) as i32),
                        wp(),
                        tol,
                        &recorder,
                    )?,
                    "combine" => {
                        let o = get(&self.geometries, other.as_ref().context("combine requires other")?)?;
                        let xf = transform.map(matrix);
                        src.CombineWithGeometry(o, D2D1_COMBINE_MODE(mode.unwrap_or(0) as i32), xf.as_ref().map(|x| x as *const _), tol, &recorder)?;
                    }
                    o => anyhow::bail!("unknown geometry op {o:?}"),
                }
                sink.Close()?; // native does not close the output sink; the caller must, to finalize the path
                drop(recorder); // release the inner-sink ref the tee held
                let stream = serde_json::json!({
                    "fill_mode": fill_mode.get(),
                    "figures": figures.borrow().clone(),
                });
                // Record the sink-output summary (figures, segments, bounds) plus the full stream.
                let b = path.GetBounds(None)?;
                self.queries.push(serde_json::json!({
                    "op": i, "geometry": id, "kind": format!("geometry_op:{operation}"),
                    "result": {
                        "figures": path.GetFigureCount()?,
                        "segments": path.GetSegmentCount()?,
                        "bounds": [b.left, b.top, b.right, b.bottom],
                        "stream": stream,
                    }
                }));
                put(&mut self.geometries, id, path.cast()?)?;
            }
            Op::Tessellate { geometry, tolerance, world } => {
                let g = get(&self.geometries, geometry)?;
                let tris = Rc::new(RefCell::new(Vec::new()));
                let sink: ID2D1TessellationSink = RecordingTessSink { tris: tris.clone() }.into();
                let wm = world.map(matrix);
                g.Tessellate(wm.as_ref().map(|x| x as *const _), tolerance.unwrap_or(D2D1_DEFAULT_FLATTENING_TOLERANCE), &sink)?;
                self.queries.push(serde_json::json!({
                    "op": i, "geometry": geometry, "kind": "tessellate",
                    "result": { "triangles": tris.borrow().clone() }
                }));
            }
            Op::Realize { id, geometry, stroke_width, style, tolerance } => {
                let dc1: ID2D1DeviceContext1 = self.dc.cast()?;
                let g = get(&self.geometries, geometry)?;
                let tol = tolerance.unwrap_or(D2D1_DEFAULT_FLATTENING_TOLERANCE);
                let r = match stroke_width {
                    Some(w) => {
                        let st = style.as_ref().map(|s| get(&self.strokes, s).cloned()).transpose()?;
                        dc1.CreateStrokedGeometryRealization(g, tol, *w, st.as_ref())?
                    }
                    None => dc1.CreateFilledGeometryRealization(g, tol)?,
                };
                put(&mut self.realizations, id, r)?;
            }
            Op::DrawRealization { realization, brush } => {
                let dc1: ID2D1DeviceContext1 = self.dc.cast()?;
                dc1.DrawGeometryRealization(get(&self.realizations, realization)?, get(&self.brushes, brush)?);
            }
            Op::Text {
                text,
                font,
                size,
                rect: r,
                brush,
                weight,
                style,
                locale,
                antialias,
                measuring,
                options,
                overload,
                params,
            } => {
                // Diagnostic: `DWRITE_COLD_FONT=<path>` resolves the family through a custom font set
                // built from that file, a face identity the shared glyph cache has never seen, so d2d1's
                // DrawText rasterizes cold and the cold glyph-fill detour ([`d2d_contour_hook`]) fires.
                let cold: Option<IDWriteFontCollection> = match std::env::var_os("DWRITE_COLD_FONT") {
                    Some(p) => {
                        let f = self.text_factory()?;
                        let f3: IDWriteFactory3 = f.cast()?;
                        let builder: IDWriteFontSetBuilder1 = f3.CreateFontSetBuilder()?.cast()?;
                        // ';'-separated for multi-font scenes (same as Op::TextLayout); a single path is one part.
                        for part in p.to_string_lossy().split(';').filter(|s| !s.is_empty()) {
                            let file = f.CreateFontFileReference(&HSTRING::from(part), None)?;
                            builder.AddFontFile(&file)?;
                        }
                        let set = builder.CreateFontSet()?;
                        Some(f3.CreateFontCollectionFromFontSet(&set)?.cast()?)
                    }
                    None => None,
                };
                let format = self.text_factory()?.CreateTextFormat(
                    &HSTRING::from(font),
                    cold.as_ref(),
                    DWRITE_FONT_WEIGHT(*weight as i32),
                    DWRITE_FONT_STYLE(*style as i32),
                    DWRITE_FONT_STRETCH_NORMAL,
                    *size,
                    &HSTRING::from(locale),
                )?;
                self.dc
                    .SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE(*antialias as i32));
                // A scene's own rendering params for this draw; the pinned ones come back after it.
                let set_params = |dc: &ID2D1DeviceContext, f: &IDWriteFactory, p: [f32; 4]| -> Result<()> {
                    let rp: IDWriteRenderingParams = f.cast::<IDWriteFactory3>()?
                        .CreateCustomRenderingParams(p[0], p[1], p[2], p[3], DWRITE_PIXEL_GEOMETRY_FLAT, DWRITE_RENDERING_MODE1_DEFAULT, DWRITE_GRID_FIT_MODE_DEFAULT)?
                        .cast()?;
                    unsafe { dc.SetTextRenderingParams(&rp) };
                    Ok(())
                };
                if let Some(p) = params {
                    set_params(&self.dc, self.text_factory()?, *p)?;
                }
                let opts = options.map_or(D2D1_DRAW_TEXT_OPTIONS_NONE, |o| D2D1_DRAW_TEXT_OPTIONS(o as i32));
                let u16s = text.encode_utf16().collect::<Vec<_>>();
                if overload.as_deref() == Some("layout") {
                    // ID2D1RenderTarget::DrawTextLayout: build the layout from the format and draw it at
                    // the rect's top-left (natural measuring; a GDI-compatible layout is a separate op).
                    let rc = rect(*r);
                    let layout = self.text_factory()?.CreateTextLayout(&u16s, &format, rc.right - rc.left, rc.bottom - rc.top)?;
                    self.dc.DrawTextLayout(D2D_POINT_2F { x: rc.left, y: rc.top }, &layout, get(&self.brushes, brush)?, opts);
                } else {
                    self.dc.DrawText(
                        &u16s,
                        &format,
                        &rect(*r),
                        get(&self.brushes, brush)?,
                        opts,
                        DWRITE_MEASURING_MODE(*measuring as i32),
                    );
                }
                if params.is_some() {
                    set_params(&self.dc, self.text_factory()?, [2.2, 1.0, 1.0, 1.0])?;
                }
            }
            Op::TextLayout {
                text, font, size, max_width, max_height, weight, style, stretch, locale,
                measuring, wrap, align, para, reading, flow,
                trimming, hit_test_points, hit_test_ranges, inline_objects, features, ranges,
                brush, origin, line_spacing, gdi_transform,
            } => {
                self.text_factory()?;
                let f = self.text_factory.as_ref().unwrap();
                // Same cold-font path as Op::Text: `DWRITE_COLD_FONT` (';'-separated for multi-font scenes)
                // builds a custom font set the shared glyph cache has never seen, so DrawTextLayout also
                // rasterizes cold and the cold-glyph contour detour fires for text_layout glyphs too.
                // `DWRITE_COLD_BOLD=1` adds the cold face with DWRITE_FONT_SIMULATIONS_BOLD, so a cold
                // DrawText synthesizes bold exactly as the system face does (a single-file custom set
                // otherwise returns the matched face WITHOUT the simulation). Lets the contour detour
                // capture native's synthetic-bold embold on a cold (cache-cold) glyph.
                let cold: Option<IDWriteFontCollection> = match std::env::var_os("DWRITE_COLD_FONT") {
                    Some(p) => {
                        let f3: IDWriteFactory3 = f.cast()?;
                        let builder: IDWriteFontSetBuilder1 = f3.CreateFontSetBuilder()?.cast()?;
                        let sims = if std::env::var_os("DWRITE_COLD_BOLD").is_some() {
                            DWRITE_FONT_SIMULATIONS_BOLD
                        } else {
                            DWRITE_FONT_SIMULATIONS_NONE
                        };
                        for part in p.to_string_lossy().split(';').filter(|s| !s.is_empty()) {
                            if sims == DWRITE_FONT_SIMULATIONS_NONE {
                                let file = f.CreateFontFileReference(&HSTRING::from(part), None)?;
                                builder.AddFontFile(&file)?;
                            } else {
                                let r = f3.CreateFontFaceReference2(&HSTRING::from(part), None, 0, sims)?;
                                builder.AddFontFaceReference(&r, &[])?;
                            }
                        }
                        Some(f3.CreateFontCollectionFromFontSet(&builder.CreateFontSet()?)?.cast()?)
                    }
                    None => None,
                };
                let format = f.CreateTextFormat(
                    &HSTRING::from(font), cold.as_ref(),
                    DWRITE_FONT_WEIGHT(*weight as i32),
                    DWRITE_FONT_STYLE(*style as i32),
                    DWRITE_FONT_STRETCH(*stretch as i32),
                    *size, &HSTRING::from(locale),
                )?;
                format.SetWordWrapping(DWRITE_WORD_WRAPPING(*wrap as i32))?;
                format.SetTextAlignment(DWRITE_TEXT_ALIGNMENT(*align as i32))?;
                format.SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT(*para as i32))?;
                format.SetReadingDirection(DWRITE_READING_DIRECTION(*reading as i32))?;
                format.SetFlowDirection(DWRITE_FLOW_DIRECTION(*flow as i32))?;
                if let Some(t) = trimming {
                    let opts = DWRITE_TRIMMING {
                        granularity: DWRITE_TRIMMING_GRANULARITY(t.granularity as i32),
                        delimiter: t.delimiter,
                        delimiterCount: t.delimiter_count,
                    };
                    let sign = if t.ellipsis { Some(f.CreateEllipsisTrimmingSign(&format)?) } else { None };
                    format.SetTrimming(&opts, sign.as_ref())?;
                }
                if let Some([m, sp, base]) = line_spacing {
                    format.SetLineSpacing(DWRITE_LINE_SPACING_METHOD(*m as i32), *sp, *base)?;
                }
                let u16s: Vec<u16> = text.encode_utf16().collect();
                let (mut dpix, mut dpiy) = (0.0f32, 0.0f32);
                self.dc.GetDpi(&mut dpix, &mut dpiy);
                let layout = if *measuring == 0 {
                    f.CreateTextLayout(&u16s, &format, *max_width, *max_height)?
                } else {
                    let [m11, m12, m21, m22] = gdi_transform.unwrap_or([1.0, 0.0, 0.0, 1.0]);
                    let m = DWRITE_MATRIX { m11, m12, m21, m22, dx: 0.0, dy: 0.0 };
                    f.CreateGdiCompatibleTextLayout(
                        &u16s, &format, *max_width, *max_height, dpix / 96.0, Some(&m), *measuring == 2,
                    )?
                };
                for o in inline_objects {
                    let obj: IDWriteInlineObject = TestInline { size: o.size, baseline: o.baseline, sideways: o.supports_sideways }.into();
                    layout.SetInlineObject(&obj, DWRITE_TEXT_RANGE { startPosition: o.position, length: o.length })?;
                }
                for fr in features {
                    let typo = f.CreateTypography()?;
                    let b = fr.tag.as_bytes();
                    let tag = DWRITE_FONT_FEATURE_TAG(u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                    typo.AddFontFeature(DWRITE_FONT_FEATURE { nameTag: tag, parameter: fr.value })?;
                    layout.SetTypography(&typo, DWRITE_TEXT_RANGE { startPosition: fr.position, length: fr.length })?;
                }
                // Per-range character formatting (the browser rich-text runs): SetFontSize/Weight/Style/
                // Stretch/Underline/Strikethrough/FontFamilyName over [position, position+length).
                for rf in ranges {
                    let range = DWRITE_TEXT_RANGE { startPosition: rf.position, length: rf.length };
                    if let Some(s) = rf.size { layout.SetFontSize(s, range)?; }
                    if let Some(w) = rf.weight { layout.SetFontWeight(DWRITE_FONT_WEIGHT(w as i32), range)?; }
                    if let Some(st) = rf.style { layout.SetFontStyle(DWRITE_FONT_STYLE(st as i32), range)?; }
                    if let Some(sr) = rf.stretch { layout.SetFontStretch(DWRITE_FONT_STRETCH(sr as i32), range)?; }
                    if let Some(u) = rf.underline { layout.SetUnderline(u, range)?; }
                    if let Some(sk) = rf.strikethrough { layout.SetStrikethrough(sk, range)?; }
                    if let Some(fam) = &rf.family { layout.SetFontFamilyName(&HSTRING::from(fam), range)?; }
                }
                let runs = Rc::new(RefCell::new(Vec::new()));
                let renderer: IDWriteTextRenderer1 = GlyphCapture { runs: runs.clone() }.into();
                layout.Draw(None, &renderer, 0.0, 0.0)?;
                // Pixel-render path: DrawTextLayout the computed layout at `origin` with `brush`.
                if let Some(bid) = brush {
                    let b = get(&self.brushes, bid)?;
                    self.dc.DrawTextLayout(
                        windows::Win32::Graphics::Direct2D::Common::D2D_POINT_2F { x: origin[0], y: origin[1] },
                        &layout, b, windows::Win32::Graphics::Direct2D::D2D1_DRAW_TEXT_OPTIONS_NONE,
                    );
                }
                let mut tm = DWRITE_TEXT_METRICS::default();
                layout.GetMetrics(&mut tm)?;
                let mut nl = 0u32;
                let _ = layout.GetLineMetrics(None, &mut nl);
                let mut lines = vec![DWRITE_LINE_METRICS::default(); nl as usize];
                layout.GetLineMetrics(Some(&mut lines), &mut nl)?;
                let mut nc = 0u32;
                let _ = layout.GetClusterMetrics(None, &mut nc);
                let mut clusters = vec![DWRITE_CLUSTER_METRICS::default(); nc as usize];
                layout.GetClusterMetrics(Some(&mut clusters), &mut nc)?;
                // Caret hit-test: leading edge of every text position 0..=len.
                let mut carets = Vec::new();
                for p in 0..=u16s.len() as u32 {
                    let (mut px, mut py) = (0.0f32, 0.0f32);
                    let mut hm = DWRITE_HIT_TEST_METRICS::default();
                    layout.HitTestTextPosition(p, false, &mut px, &mut py, &mut hm)?;
                    carets.push(serde_json::json!({
                        "pos": p, "x": px, "y": py,
                        "left": hm.left, "top": hm.top, "width": hm.width, "height": hm.height,
                        "bidi": hm.bidiLevel, "is_text": hm.isText.as_bool(), "is_trimmed": hm.isTrimmed.as_bool(),
                    }));
                }
                // HitTestPoint: (x,y) → the cluster hit, whether the trailing half was hit, and
                // whether the point lies inside the text.
                let mut hit_points = Vec::new();
                for &[x, y] in hit_test_points {
                    let (mut trailing, mut inside) = (Default::default(), Default::default());
                    let mut hm = DWRITE_HIT_TEST_METRICS::default();
                    layout.HitTestPoint(x, y, &mut trailing, &mut inside, &mut hm)?;
                    let mut j = hit_metrics_json(&hm);
                    j["x"] = x.into();
                    j["y"] = y.into();
                    j["is_trailing"] = trailing.as_bool().into();
                    j["is_inside"] = inside.as_bool().into();
                    hit_points.push(j);
                }
                // HitTestTextRange: one box per line-segment the range covers (count-probe then fill).
                let mut hit_ranges = Vec::new();
                for &[pos, len] in hit_test_ranges {
                    let mut n = 0u32;
                    let _ = layout.HitTestTextRange(pos, len, 0.0, 0.0, None, &mut n);
                    let mut boxes = vec![DWRITE_HIT_TEST_METRICS::default(); n as usize];
                    layout.HitTestTextRange(pos, len, 0.0, 0.0, Some(&mut boxes), &mut n)?;
                    hit_ranges.push(boxes.iter().map(hit_metrics_json).collect::<Vec<_>>());
                }
                self.queries.push(serde_json::json!({
                    "op": i, "kind": "text_layout",
                    "result": {
                        "metrics": {
                            "left": tm.left, "top": tm.top, "width": tm.width,
                            "width_trailing": tm.widthIncludingTrailingWhitespace,
                            "height": tm.height, "layout_width": tm.layoutWidth,
                            "layout_height": tm.layoutHeight,
                            "bidi_depth": tm.maxBidiReorderingDepth, "line_count": tm.lineCount,
                        },
                        "lines": lines.iter().map(|l| serde_json::json!({
                            "length": l.length, "trailing_ws": l.trailingWhitespaceLength,
                            "newline": l.newlineLength, "height": l.height,
                            "baseline": l.baseline, "trimmed": l.isTrimmed.as_bool(),
                        })).collect::<Vec<_>>(),
                        "clusters": clusters.iter().map(|c| serde_json::json!({
                            "width": c.width, "length": c.length, "bits": c._bitfield,
                        })).collect::<Vec<_>>(),
                        "carets": carets,
                        "hit_points": hit_points,
                        "hit_ranges": hit_ranges,
                        "runs": &*runs.borrow(),
                    }
                }));
            }
            Op::TextMetrics { text, font, size } => {
                let format = self.text_factory()?.CreateTextFormat(
                    &HSTRING::from(font),
                    None,
                    DWRITE_FONT_WEIGHT_NORMAL,
                    DWRITE_FONT_STYLE_NORMAL,
                    DWRITE_FONT_STRETCH_NORMAL,
                    *size,
                    w!("en-us"),
                )?;
                let layout = self.text_factory()?.CreateTextLayout(
                    &text.encode_utf16().collect::<Vec<_>>(),
                    &format,
                    f32::MAX,
                    f32::MAX,
                )?;
                let mut m = DWRITE_TEXT_METRICS::default();
                layout.GetMetrics(&mut m)?;
                self.queries.push(serde_json::json!({
                    "op": i, "kind": "text_metrics",
                    "result": {
                        "width": m.width,
                        "widthIncludingTrailingWhitespace": m.widthIncludingTrailingWhitespace,
                        "height": m.height,
                    }
                }));
            }
            Op::GlyphRun { text, font, size, weight, style, baseline, brush, antialias, measuring, rendering_mode, grid_fit, simulations: _ } => {
                let factory = self.text_factory()?.clone();
                let factory3: IDWriteFactory3 = factory.cast()?;
                // Resolve the font face from the system collection (family + weight/style), build the
                // run from cmap ids + natural design advances — the same inputs the Rust runner uses.
                let collection: IDWriteFontCollection = if let Some(p) = std::env::var_os("DWRITE_COLD_FONT") {
                    // Use a fresh file identity to bypass the shared glyph cache during native diagnostics;
                    // ';'-separated for multi-font scenes, as for Op::Text.
                    let builder: IDWriteFontSetBuilder1 = factory3.CreateFontSetBuilder()?.cast()?;
                    for part in p.to_string_lossy().split(';').filter(|s| !s.is_empty()) {
                        builder.AddFontFile(&factory.CreateFontFileReference(&HSTRING::from(part), None)?)?;
                    }
                    factory3.CreateFontCollectionFromFontSet(&builder.CreateFontSet()?)?.cast()?
                } else {
                    let mut collection = None;
                    factory.GetSystemFontCollection(&mut collection, false)?;
                    collection.context("no system font collection")?
                };
                let (mut index, mut exists) = (0u32, windows::Win32::Foundation::BOOL(0));
                collection.FindFamilyName(&HSTRING::from(font), &mut index, &mut exists)?;
                ensure!(exists.as_bool(), "font family not found: {font}");
                // `DWRITE_COLD_BOLD=1` forces DWRITE_FONT_SIMULATIONS_BOLD on the cold single-file face
                // directly (a custom single-file collection's GetFirstMatchingFont returns it WITHOUT the
                // simulation), so a cold DrawGlyphRun synthesizes bold exactly as the system face does —
                // reproducing native's synthetic-bold embold for a regular-only face (e.g. Segoe UI Emoji).
                let face = match (std::env::var_os("DWRITE_COLD_FONT"), std::env::var_os("DWRITE_COLD_BOLD")) {
                    (Some(p), Some(_)) => {
                        let s = p.to_string_lossy();
                        let part = s.split(';').next().unwrap_or("");
                        let file = factory.CreateFontFileReference(&HSTRING::from(part), None)?;
                        factory.CreateFontFace(DWRITE_FONT_FACE_TYPE_TRUETYPE, &[Some(file)], 0, DWRITE_FONT_SIMULATIONS_BOLD)?
                    }
                    _ => collection
                        .GetFontFamily(index)?
                        .GetFirstMatchingFont(DWRITE_FONT_WEIGHT(*weight as i32), DWRITE_FONT_STRETCH_NORMAL, DWRITE_FONT_STYLE(*style as i32))?
                        .CreateFontFace()?,
                };
                let mut fm = DWRITE_FONT_METRICS::default();
                face.GetMetrics(&mut fm);
                let upem = fm.designUnitsPerEm as f32;
                let codes: Vec<u32> = text.chars().map(|c| c as u32).collect();
                let mut ids = vec![0u16; codes.len()];
                face.GetGlyphIndices(codes.as_ptr(), codes.len() as u32, ids.as_mut_ptr())?;
                let mut gm = vec![DWRITE_GLYPH_METRICS::default(); codes.len()];
                face.GetDesignGlyphMetrics(ids.as_ptr(), codes.len() as u32, gm.as_mut_ptr(), false)?;
                let advances: Vec<f32> = gm.iter().map(|m| m.advanceWidth as f32 * size / upem).collect();
                let offsets = vec![DWRITE_GLYPH_OFFSET { advanceOffset: 0.0, ascenderOffset: 0.0 }; codes.len()];
                let run = DWRITE_GLYPH_RUN {
                    fontFace: ManuallyDrop::new(Some(face.clone())),
                    fontEmSize: *size,
                    glyphCount: codes.len() as u32,
                    glyphIndices: ids.as_ptr(),
                    glyphAdvances: advances.as_ptr(),
                    glyphOffsets: offsets.as_ptr(),
                    isSideways: false.into(),
                    bidiLevel: 0,
                };
                // Pin the harness render params (gamma 2.2 / contrast 1 / grayscale contrast 1 /
                // level 1 / flat) plus the scene's explicit rendering mode + grid-fit, so DrawGlyphRun's
                // coverage is exactly what the Rust path reproduces.
                let params: IDWriteRenderingParams = factory3
                    .CreateCustomRenderingParams(
                        2.2, 1.0, 1.0, 1.0, DWRITE_PIXEL_GEOMETRY_FLAT,
                        DWRITE_RENDERING_MODE1(*rendering_mode as i32),
                        DWRITE_GRID_FIT_MODE(*grid_fit as i32),
                    )?
                    .cast()?;
                self.dc.SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE(*antialias as i32));
                self.dc.SetTextRenderingParams(&params);
                self.dc.DrawGlyphRun(
                    D2D_POINT_2F { x: baseline[0], y: baseline[1] },
                    &run,
                    None,
                    get(&self.brushes, brush)?,
                    DWRITE_MEASURING_MODE(*measuring as i32),
                );
                let _ = ManuallyDrop::into_inner(run.fontFace);
            }
            Op::InMemoryFontRun { font, text, em_size, baseline, color: fg } => {
                let factory = self.text_factory()?.clone();
                let dir = std::env::var("D2D_FONTS").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from("C:\\Windows\\Fonts"));
                let bytes = std::fs::read(dir.join(font))?;
                // Browser web-font path: an in-memory loader + reference, registered on the factory.
                let f5: IDWriteFactory5 = factory.cast()?;
                let loader = f5.CreateInMemoryFontFileLoader()?;
                f5.RegisterFontFileLoader(&loader)?;
                let file = loader.CreateInMemoryFontFileReference(&factory, bytes.as_ptr() as *const _, bytes.len() as u32, None)?;
                let file_count = loader.GetFileCount();
                let face = factory.CreateFontFace(DWRITE_FONT_FACE_TYPE_TRUETYPE, &[Some(file)], 0, DWRITE_FONT_SIMULATIONS_NONE)?;
                let mut fm = DWRITE_FONT_METRICS::default();
                face.GetMetrics(&mut fm);
                let upem = fm.designUnitsPerEm as f32;
                let codes: Vec<u32> = text.chars().map(|c| c as u32).collect();
                let mut ids = vec![0u16; codes.len()];
                face.GetGlyphIndices(codes.as_ptr(), codes.len() as u32, ids.as_mut_ptr())?;
                let mut gm = vec![DWRITE_GLYPH_METRICS::default(); codes.len()];
                face.GetDesignGlyphMetrics(ids.as_ptr(), codes.len() as u32, gm.as_mut_ptr(), false)?;
                self.queries.push(serde_json::json!({"op": i, "kind": "inmem_font", "result": {
                    "file_count": file_count, "upem": fm.designUnitsPerEm, "ascent": fm.ascent, "descent": fm.descent,
                    "glyphs": ids, "advances": gm.iter().map(|m| m.advanceWidth).collect::<Vec<u32>>(),
                }}));
                let advances: Vec<f32> = gm.iter().map(|m| m.advanceWidth as f32 * em_size / upem).collect();
                let offsets = vec![DWRITE_GLYPH_OFFSET { advanceOffset: 0.0, ascenderOffset: 0.0 }; codes.len()];
                let run = DWRITE_GLYPH_RUN {
                    fontFace: ManuallyDrop::new(Some(face.clone())),
                    fontEmSize: *em_size, glyphCount: codes.len() as u32,
                    glyphIndices: ids.as_ptr(), glyphAdvances: advances.as_ptr(), glyphOffsets: offsets.as_ptr(),
                    isSideways: false.into(), bidiLevel: 0,
                };
                let fgb = self.dc.CreateSolidColorBrush(&color(*fg), None)?;
                self.dc.SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE_GRAYSCALE);
                self.dc.DrawGlyphRun(point(*baseline), &run, None, &fgb, DWRITE_MEASURING_MODE_NATURAL);
                let _ = ManuallyDrop::into_inner(run.fontFace);
                f5.UnregisterFontFileLoader(&loader)?;
            }
            Op::CustomFontRun { font, text, em_size, baseline, color: fg } => {
                let factory = self.text_factory()?.clone();
                let dir = std::env::var("D2D_FONTS").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from("C:\\Windows\\Fonts"));
                let path = dir.join(font);
                let bytes = std::fs::read(&path)?;
                let codes: Vec<u32> = text.chars().map(|c| c as u32).collect();
                // (a) custom font-file loader → CreateCustomFontFileReference.
                let file_loader: IDWriteFontFileLoader = MemLoader { data: bytes }.into();
                factory.RegisterFontFileLoader(&file_loader)?;
                let key = [1u8, 2, 3, 4];
                let file = factory.CreateCustomFontFileReference(key.as_ptr() as *const _, key.len() as u32, &file_loader)?;
                let fface = factory.CreateFontFace(DWRITE_FONT_FACE_TYPE_TRUETYPE, &[Some(file)], 0, DWRITE_FONT_SIMULATIONS_NONE)?;
                let mut file_ids = vec![0u16; codes.len()];
                fface.GetGlyphIndices(codes.as_ptr(), codes.len() as u32, file_ids.as_mut_ptr())?;
                factory.UnregisterFontFileLoader(&file_loader)?;
                // (b) custom font-collection loader → CreateCustomFontCollection.
                let coll_loader: IDWriteFontCollectionLoader = PathCollLoader { path: path.to_string_lossy().into_owned() }.into();
                factory.RegisterFontCollectionLoader(&coll_loader)?;
                let key2 = [9u8, 9];
                let coll = factory.CreateCustomFontCollection(&coll_loader, key2.as_ptr() as *const _, key2.len() as u32)?;
                let fam_count = coll.GetFontFamilyCount();
                let cface = coll.GetFontFamily(0)?.GetFirstMatchingFont(DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STRETCH_NORMAL, DWRITE_FONT_STYLE_NORMAL)?.CreateFontFace()?;
                let mut coll_ids = vec![0u16; codes.len()];
                cface.GetGlyphIndices(codes.as_ptr(), codes.len() as u32, coll_ids.as_mut_ptr())?;
                factory.UnregisterFontCollectionLoader(&coll_loader)?;
                self.queries.push(serde_json::json!({"op": i, "kind": "custom_font", "result": {
                    "file_glyphs": file_ids, "coll_glyphs": coll_ids, "family_count": fam_count,
                }}));
                // Render the collection face so the custom path is pixel-verified.
                let mut fm = DWRITE_FONT_METRICS::default();
                cface.GetMetrics(&mut fm);
                let upem = fm.designUnitsPerEm as f32;
                let mut gm = vec![DWRITE_GLYPH_METRICS::default(); codes.len()];
                cface.GetDesignGlyphMetrics(coll_ids.as_ptr(), codes.len() as u32, gm.as_mut_ptr(), false)?;
                let advances: Vec<f32> = gm.iter().map(|m| m.advanceWidth as f32 * em_size / upem).collect();
                let offsets = vec![DWRITE_GLYPH_OFFSET { advanceOffset: 0.0, ascenderOffset: 0.0 }; codes.len()];
                let run = DWRITE_GLYPH_RUN {
                    fontFace: ManuallyDrop::new(Some(cface.clone())),
                    fontEmSize: *em_size, glyphCount: codes.len() as u32,
                    glyphIndices: coll_ids.as_ptr(), glyphAdvances: advances.as_ptr(), glyphOffsets: offsets.as_ptr(),
                    isSideways: false.into(), bidiLevel: 0,
                };
                let fgb = self.dc.CreateSolidColorBrush(&color(*fg), None)?;
                self.dc.SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE_GRAYSCALE);
                self.dc.DrawGlyphRun(point(*baseline), &run, None, &fgb, DWRITE_MEASURING_MODE_NATURAL);
                let _ = ManuallyDrop::into_inner(run.fontFace);
            }
            Op::UnpackFontFileTest { container, text } => {
                let bytes = std::fs::read(container)?;
                let factory = self.text_factory()?.clone();
                let f5: IDWriteFactory5 = factory.cast()?;
                let ct = f5.AnalyzeContainerType(bytes.as_ptr() as *const _, bytes.len() as u32);
                let stream = f5.UnpackFontFile(ct, bytes.as_ptr() as *const _, bytes.len() as u32)?;
                let size = stream.GetFileSize()?;
                let (mut start, mut sctx): (*mut core::ffi::c_void, *mut core::ffi::c_void) = (std::ptr::null_mut(), std::ptr::null_mut());
                stream.ReadFileFragment(&mut start, 0, size, &mut sctx)?;
                let sfnt = unsafe { std::slice::from_raw_parts(start as *const u8, size as usize) }.to_vec();
                stream.ReleaseFileFragment(sctx);
                // Build a face from the unpacked sfnt via the in-memory loader → compare glyph data.
                let loader = f5.CreateInMemoryFontFileLoader()?;
                f5.RegisterFontFileLoader(&loader)?;
                let file = loader.CreateInMemoryFontFileReference(&factory, sfnt.as_ptr() as *const _, sfnt.len() as u32, None)?;
                let face = factory.CreateFontFace(DWRITE_FONT_FACE_TYPE_TRUETYPE, &[Some(file)], 0, DWRITE_FONT_SIMULATIONS_NONE)?;
                let glyph_count = face.GetGlyphCount();
                let codes: Vec<u32> = text.chars().map(|c| c as u32).collect();
                let mut ids = vec![0u16; codes.len()];
                face.GetGlyphIndices(codes.as_ptr(), codes.len() as u32, ids.as_mut_ptr())?;
                f5.UnregisterFontFileLoader(&loader)?;
                self.queries.push(serde_json::json!({"op": i, "kind": "unpack_font", "result": {
                    "container": ct.0, "glyph_count": glyph_count, "glyphs": ids,
                }}));
            }
            Op::ColorGlyphRun { font, text, em_size, baseline, palette, color: fg, with_color_support } => {
                use std::hash::{Hash, Hasher};
                let factory = self.text_factory()?.clone();
                // Materialise the inline TTF to a (content-named) temp file and open face 0.
                let mut h = std::collections::hash_map::DefaultHasher::new();
                font.hash(&mut h);
                let path = std::env::temp_dir().join(format!("d2d_colr_{:016x}.ttf", h.finish()));
                std::fs::write(&path, font)?;
                let file = factory.CreateFontFileReference(&HSTRING::from(path.to_string_lossy().as_ref()), None)?;
                let face = factory.CreateFontFace(DWRITE_FONT_FACE_TYPE_TRUETYPE, &[Some(file)], 0, DWRITE_FONT_SIMULATIONS_NONE)?;
                let mut fm = DWRITE_FONT_METRICS::default();
                face.GetMetrics(&mut fm);
                let upem = fm.designUnitsPerEm as f32;
                let codes: Vec<u32> = text.chars().map(|c| c as u32).collect();
                let mut ids = vec![0u16; codes.len()];
                face.GetGlyphIndices(codes.as_ptr(), codes.len() as u32, ids.as_mut_ptr())?;
                let mut gm = vec![DWRITE_GLYPH_METRICS::default(); codes.len()];
                face.GetDesignGlyphMetrics(ids.as_ptr(), codes.len() as u32, gm.as_mut_ptr(), false)?;
                let advances: Vec<f32> = gm.iter().map(|m| m.advanceWidth as f32 * em_size / upem).collect();
                let offsets = vec![DWRITE_GLYPH_OFFSET { advanceOffset: 0.0, ascenderOffset: 0.0 }; codes.len()];
                let run = DWRITE_GLYPH_RUN {
                    fontFace: ManuallyDrop::new(Some(face.clone())),
                    fontEmSize: *em_size,
                    glyphCount: codes.len() as u32,
                    glyphIndices: ids.as_ptr(),
                    glyphAdvances: advances.as_ptr(),
                    glyphOffsets: offsets.as_ptr(),
                    isSideways: false.into(),
                    bidiLevel: 0,
                };
                let fgb = self.dc.CreateSolidColorBrush(&color(*fg), None)?;
                // Grayscale coverage; pixel-aligned rect outlines make it a flat {0,255} either way.
                self.dc.SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE_GRAYSCALE);
                if *with_color_support {
                    self.dc7()?.DrawGlyphRunWithColorSupport(
                        point(*baseline), &run, None, &fgb, None,
                        *palette, DWRITE_MEASURING_MODE_NATURAL, D2D1_COLOR_BITMAP_GLYPH_SNAP_OPTION_DEFAULT,
                    );
                } else {
                    self.dc7()?.DrawPaintGlyphRun(point(*baseline), &run, &fgb, *palette, DWRITE_MEASURING_MODE_NATURAL);
                }
                let _ = ManuallyDrop::into_inner(run.fontFace);
                let _ = std::fs::remove_file(&path);
            }
            Op::ColorBitmapGlyphRun { font, text, em_size, baseline, glyph_format } => {
                let (face, ids, advances, offsets) = self.inline_run(font, text, *em_size)?;
                let run = DWRITE_GLYPH_RUN {
                    fontFace: ManuallyDrop::new(Some(face.clone())),
                    fontEmSize: *em_size, glyphCount: ids.len() as u32,
                    glyphIndices: ids.as_ptr(), glyphAdvances: advances.as_ptr(), glyphOffsets: offsets.as_ptr(),
                    isSideways: false.into(), bidiLevel: 0,
                };
                let fmt = match glyph_format.as_str() {
                    "jpeg" => DWRITE_GLYPH_IMAGE_FORMATS_JPEG,
                    "tiff" => DWRITE_GLYPH_IMAGE_FORMATS_TIFF,
                    "pbgra" => DWRITE_GLYPH_IMAGE_FORMATS_PREMULTIPLIED_B8G8R8A8,
                    _ => DWRITE_GLYPH_IMAGE_FORMATS_PNG,
                };
                self.dc4()?.DrawColorBitmapGlyphRun(fmt, point(*baseline), &run, DWRITE_MEASURING_MODE_NATURAL, D2D1_COLOR_BITMAP_GLYPH_SNAP_OPTION_DEFAULT);
                let _ = ManuallyDrop::into_inner(run.fontFace);
            }
            Op::SvgGlyphRun { font, text, em_size, baseline, color: fg } => {
                let (face, ids, advances, offsets) = self.inline_run(font, text, *em_size)?;
                let run = DWRITE_GLYPH_RUN {
                    fontFace: ManuallyDrop::new(Some(face.clone())),
                    fontEmSize: *em_size, glyphCount: ids.len() as u32,
                    glyphIndices: ids.as_ptr(), glyphAdvances: advances.as_ptr(), glyphOffsets: offsets.as_ptr(),
                    isSideways: false.into(), bidiLevel: 0,
                };
                let fgb = self.dc.CreateSolidColorBrush(&color(*fg), None)?;
                self.dc4()?.DrawSvgGlyphRun(point(*baseline), &run, &fgb, None, 0, DWRITE_MEASURING_MODE_NATURAL);
                let _ = ManuallyDrop::into_inner(run.fontFace);
            }
            Op::ColorBitmapGlyphImage { font, text, em_size, baseline, glyph_format } => {
                let (face, ids, advances, _offsets) = self.inline_run(font, text, *em_size)?;
                let fmt = match glyph_format.as_str() {
                    "jpeg" => DWRITE_GLYPH_IMAGE_FORMATS_JPEG,
                    "tiff" => DWRITE_GLYPH_IMAGE_FORMATS_TIFF,
                    "pbgra" => DWRITE_GLYPH_IMAGE_FORMATS_PREMULTIPLIED_B8G8R8A8,
                    _ => DWRITE_GLYPH_IMAGE_FORMATS_PNG,
                };
                let dc4 = self.dc4()?;
                let mut pen_x = baseline[0];
                for (i, &gid) in ids.iter().enumerate() {
                    let mut gt = Matrix3x2::default();
                    let mut img_out: Option<ID2D1Image> = None;
                    // NULL world transform (identity); the returned glyphTransform places the image.
                    let r = dc4.GetColorBitmapGlyphImage(fmt, point([pen_x, baseline[1]]), &face, *em_size, gid, false, None, 96.0, 96.0, &mut gt, &mut img_out);
                    pen_x += advances[i];
                    if r.is_ok() {
                        if let Some(img) = img_out {
                            self.dc.SetTransform(&gt);
                            self.dc.DrawImage(&img, None, None, D2D1_INTERPOLATION_MODE_LINEAR, D2D1_COMPOSITE_MODE_SOURCE_OVER);
                            self.dc.SetTransform(&matrix([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]));
                        }
                    }
                }
            }
            Op::SvgGlyphImage { font, text, em_size, baseline, color: fg } => {
                let (face, ids, advances, _offsets) = self.inline_run(font, text, *em_size)?;
                let fgb = self.dc.CreateSolidColorBrush(&color(*fg), None)?;
                let dc4 = self.dc4()?;
                let mut pen_x = baseline[0];
                for (i, &gid) in ids.iter().enumerate() {
                    let mut gt = Matrix3x2::default();
                    let mut list_out: Option<ID2D1CommandList> = None;
                    let r = dc4.GetSvgGlyphImage(point([pen_x, baseline[1]]), &face, *em_size, gid, false, None, &fgb, None, 0, &mut gt, &mut list_out);
                    pen_x += advances[i];
                    if r.is_ok() {
                        if let Some(list) = list_out {
                            self.dc.SetTransform(&gt);
                            self.dc.DrawImage(&list, None, None, D2D1_INTERPOLATION_MODE_LINEAR, D2D1_COMPOSITE_MODE_SOURCE_OVER);
                            self.dc.SetTransform(&matrix([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]));
                        }
                    }
                }
            }
            Op::GlyphRunInline { font, glyphs, em_size, baseline, brush, transform, antialias, rendering_mode, grid_fit, measuring, contrast } => {
                use std::hash::{Hash, Hasher};
                let factory = self.text_factory()?.clone();
                let factory3: IDWriteFactory3 = factory.cast()?;
                let mut h = std::collections::hash_map::DefaultHasher::new();
                font.hash(&mut h);
                let path = std::env::temp_dir().join(format!("d2d_gri_{:016x}.ttf", h.finish()));
                std::fs::write(&path, font)?;
                let file = factory.CreateFontFileReference(&HSTRING::from(path.to_string_lossy().as_ref()), None)?;
                let face = factory.CreateFontFace(DWRITE_FONT_FACE_TYPE_TRUETYPE, &[Some(file)], 0, DWRITE_FONT_SIMULATIONS_NONE)?;
                let mut fm = DWRITE_FONT_METRICS::default();
                face.GetMetrics(&mut fm);
                let upem = fm.designUnitsPerEm as f32;
                let mut gm = vec![DWRITE_GLYPH_METRICS::default(); glyphs.len()];
                face.GetDesignGlyphMetrics(glyphs.as_ptr(), glyphs.len() as u32, gm.as_mut_ptr(), false)?;
                let advances: Vec<f32> = gm.iter().map(|m| m.advanceWidth as f32 * em_size / upem).collect();
                let offsets = vec![DWRITE_GLYPH_OFFSET { advanceOffset: 0.0, ascenderOffset: 0.0 }; glyphs.len()];
                let run = DWRITE_GLYPH_RUN {
                    fontFace: ManuallyDrop::new(Some(face.clone())),
                    fontEmSize: *em_size,
                    glyphCount: glyphs.len() as u32,
                    glyphIndices: glyphs.as_ptr(),
                    glyphAdvances: advances.as_ptr(),
                    glyphOffsets: offsets.as_ptr(),
                    isSideways: false.into(),
                    bidiLevel: 0,
                };
                self.dc.SetTransform(&matrix(*transform));
                self.dc.SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE(*antialias as i32));
                // Some(c): custom params with enhanced-contrast c. None: leave the DC's default params
                // in place (what DrawPaintGlyphRun's leaves use), to probe the default contrast/gamma.
                if let Some(c) = *contrast {
                    let params: IDWriteRenderingParams = factory3
                        .CreateCustomRenderingParams(
                            2.2, 1.0, c, 1.0, DWRITE_PIXEL_GEOMETRY_FLAT,
                            DWRITE_RENDERING_MODE1(*rendering_mode as i32),
                            DWRITE_GRID_FIT_MODE(*grid_fit as i32),
                        )?
                        .cast()?;
                    self.dc.SetTextRenderingParams(&params);
                }
                self.dc.DrawGlyphRun(point(*baseline), &run, None, get(&self.brushes, brush)?, DWRITE_MEASURING_MODE(*measuring as i32));
                self.dc.SetTransform(&matrix([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]));
                let _ = ManuallyDrop::into_inner(run.fontFace);
                let _ = std::fs::remove_file(&path);
            }
            // --- ffirects (this agent): bounds / rect-mapping / copy ---
            Op::GlyphBounds { text, font, size, weight, style, baseline, antialias, measuring, rendering_mode, grid_fit } => {
                // Build the run exactly like `GlyphRun` (cmap ids + design advances) and record
                // GetGlyphRunWorldBounds under the same AA + rendering params.
                let factory = self.text_factory()?.clone();
                let factory3: IDWriteFactory3 = factory.cast()?;
                let mut collection: Option<IDWriteFontCollection> = None;
                factory.GetSystemFontCollection(&mut collection, false)?;
                let collection = collection.context("no system font collection")?;
                let (mut index, mut exists) = (0u32, windows::Win32::Foundation::BOOL(0));
                collection.FindFamilyName(&HSTRING::from(font), &mut index, &mut exists)?;
                ensure!(exists.as_bool(), "font family not found: {font}");
                // `DWRITE_COLD_BOLD=1` forces DWRITE_FONT_SIMULATIONS_BOLD on the cold single-file face
                // directly (a custom single-file collection's GetFirstMatchingFont returns it WITHOUT the
                // simulation), so a cold DrawGlyphRun synthesizes bold exactly as the system face does —
                // reproducing native's synthetic-bold embold for a regular-only face (e.g. Segoe UI Emoji).
                let face = match (std::env::var_os("DWRITE_COLD_FONT"), std::env::var_os("DWRITE_COLD_BOLD")) {
                    (Some(p), Some(_)) => {
                        let s = p.to_string_lossy();
                        let part = s.split(';').next().unwrap_or("");
                        let file = factory.CreateFontFileReference(&HSTRING::from(part), None)?;
                        factory.CreateFontFace(DWRITE_FONT_FACE_TYPE_TRUETYPE, &[Some(file)], 0, DWRITE_FONT_SIMULATIONS_BOLD)?
                    }
                    _ => collection
                        .GetFontFamily(index)?
                        .GetFirstMatchingFont(DWRITE_FONT_WEIGHT(*weight as i32), DWRITE_FONT_STRETCH_NORMAL, DWRITE_FONT_STYLE(*style as i32))?
                        .CreateFontFace()?,
                };
                let mut fm = DWRITE_FONT_METRICS::default();
                face.GetMetrics(&mut fm);
                let upem = fm.designUnitsPerEm as f32;
                let codes: Vec<u32> = text.chars().map(|c| c as u32).collect();
                let mut ids = vec![0u16; codes.len()];
                face.GetGlyphIndices(codes.as_ptr(), codes.len() as u32, ids.as_mut_ptr())?;
                let mut gm = vec![DWRITE_GLYPH_METRICS::default(); codes.len()];
                face.GetDesignGlyphMetrics(ids.as_ptr(), codes.len() as u32, gm.as_mut_ptr(), false)?;
                let advances: Vec<f32> = gm.iter().map(|m| m.advanceWidth as f32 * size / upem).collect();
                let offsets = vec![DWRITE_GLYPH_OFFSET { advanceOffset: 0.0, ascenderOffset: 0.0 }; codes.len()];
                let run = DWRITE_GLYPH_RUN {
                    fontFace: ManuallyDrop::new(Some(face.clone())),
                    fontEmSize: *size,
                    glyphCount: codes.len() as u32,
                    glyphIndices: ids.as_ptr(),
                    glyphAdvances: advances.as_ptr(),
                    glyphOffsets: offsets.as_ptr(),
                    isSideways: false.into(),
                    bidiLevel: 0,
                };
                let params: IDWriteRenderingParams = factory3
                    .CreateCustomRenderingParams(2.2, 1.0, 1.0, 1.0, DWRITE_PIXEL_GEOMETRY_FLAT, DWRITE_RENDERING_MODE1(*rendering_mode as i32), DWRITE_GRID_FIT_MODE(*grid_fit as i32))?
                    .cast()?;
                self.dc.SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE(*antialias as i32));
                self.dc.SetTextRenderingParams(&params);
                let b = self.dc.GetGlyphRunWorldBounds(point(*baseline), &run, DWRITE_MEASURING_MODE(*measuring as i32))?;
                let _ = ManuallyDrop::into_inner(run.fontFace);
                self.queries.push(serde_json::json!({"op": i, "kind": "glyph_bounds", "result": [b.left, b.top, b.right, b.bottom]}));
            }
            Op::ImageBounds { image, world } => {
                let img = get(&self.images, image)?;
                let b = if *world { self.dc.GetImageWorldBounds(img)? } else { self.dc.GetImageLocalBounds(img)? };
                self.queries.push(serde_json::json!({"op": i, "kind": "image_bounds", "result": [b.left, b.top, b.right, b.bottom]}));
            }
            Op::EffectRequiredInput { effect, output, inputs } => {
                let eff = self.effects.get(effect).with_context(|| format!("unknown effect {effect}"))?.clone();
                let out = D2D_RECT_F { left: output[0], top: output[1], right: output[2], bottom: output[3] };
                let mut descs: Vec<D2D1_EFFECT_INPUT_DESCRIPTION> = Vec::with_capacity(inputs.len());
                for ri in inputs {
                    let e = self.effects.get(&ri.effect).with_context(|| format!("unknown effect {}", ri.effect))?.clone();
                    descs.push(D2D1_EFFECT_INPUT_DESCRIPTION {
                        effect: ManuallyDrop::new(Some(e)),
                        inputIndex: ri.index,
                        inputRectangle: D2D_RECT_F { left: ri.rect[0], top: ri.rect[1], right: ri.rect[2], bottom: ri.rect[3] },
                    });
                }
                let mut outrects = vec![D2D_RECT_F::default(); descs.len()];
                self.dc.GetEffectRequiredInputRectangles(&eff, Some(&out), descs.as_ptr(), outrects.as_mut_ptr(), descs.len() as u32)?;
                for d in descs {
                    let _ = ManuallyDrop::into_inner(d.effect);
                }
                let result: Vec<[f32; 4]> = outrects.iter().map(|r| [r.left, r.top, r.right, r.bottom]).collect();
                self.queries.push(serde_json::json!({"op": i, "kind": "required_input", "result": result}));
            }
            Op::EffectInvalidRects { effect, invalidate } => {
                let eff = self.effects.get(effect).with_context(|| format!("unknown effect {effect}"))?.clone();
                for inv in invalidate {
                    let r = D2D_RECT_F { left: inv.rect[0], top: inv.rect[1], right: inv.rect[2], bottom: inv.rect[3] };
                    self.dc.InvalidateEffectInputRectangle(&eff, inv.input, &r)?;
                }
                let count = self.dc.GetEffectInvalidRectangleCount(&eff)?;
                let mut rects = vec![D2D_RECT_F::default(); count as usize];
                if count > 0 {
                    self.dc.GetEffectInvalidRectangles(&eff, &mut rects)?;
                }
                let result: Vec<[f32; 4]> = rects.iter().map(|r| [r.left, r.top, r.right, r.bottom]).collect();
                self.queries.push(serde_json::json!({"op": i, "kind": "invalid_rects", "result": {"count": count, "rects": result}}));
            }
            Op::CopyMemoryBitmap { id, size, rgba } => {
                // Create an empty BGRA8-premultiplied bitmap, then CopyFromMemory the premultiplied rows.
                let data = premultiply_bgra(rgba);
                let bitmap = self.dc.CreateBitmap(
                    D2D_SIZE_U { width: size[0], height: size[1] },
                    None,
                    0,
                    &bitmap_properties_fmt(96., D2D1_BITMAP_OPTIONS_NONE, DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED),
                )?;
                let rect = D2D_RECT_U { left: 0, top: 0, right: size[0], bottom: size[1] };
                bitmap.CopyFromMemory(Some(&rect), data.as_ptr().cast(), size[0] * 4)?;
                put(&mut self.images, id, bitmap.cast()?)?;
                put(&mut self.bitmaps, id, bitmap)?;
            }
            Op::GetProp { kind, target } => {
                let t = target.as_deref().unwrap_or("");
                let result = match kind.as_str() {
                    // Small deterministic getter/creator coverage: ID2D1Factory::ReloadSystemMetrics,
                    // ID2D1Resource::GetFactory, ID2D1RenderTarget::CreateLayer + GetTextRenderingParams,
                    // ID2D1Device::Set/GetMaximumTextureMemory.
                    "coverage_misc" => {
                        let reload = hr_str(self.factory.ReloadSystemMetrics());
                        let brush = self.dc.CreateSolidColorBrush(&color([1.0, 0.0, 0.0, 1.0]), None)?;
                        let _fac: ID2D1Factory = brush.GetFactory()?;
                        let layer = self.rt.CreateLayer(Some(&D2D_SIZE_F { width: 40.0, height: 24.0 }))?;
                        let lsz = layer.GetSize();
                        let has_params = self.rt.GetTextRenderingParams().is_ok() as u32;
                        let dev: ID2D1Device = self.dc.GetDevice()?;
                        dev.SetMaximumTextureMemory(1_234_567);
                        // ID2D1Device::ClearResources — free device resources unused for ≥0 ms. A no-op
                        // for our CPU core (nothing cached); exercised here so the method is covered.
                        dev.ClearResources(0);
                        serde_json::json!({
                            "reload_hr": reload,
                            "has_factory": 1u32,
                            "layer_size": [lsz.width, lsz.height],
                            "has_text_params": has_params,
                            "max_texture_memory": dev.GetMaximumTextureMemory(),
                        })
                    }
                    // ID2D1ImageSourceFromWic EnsureCached/TrimCache/GetSource + CreateTransformedImageSource
                    // GetSource/GetProperties.
                    "image_source" => {
                        use windows::Win32::Graphics::Imaging::{GUID_WICPixelFormat32bppPBGRA, WICBitmapCacheOnLoad};
                        let wic = self.wic()?.CreateBitmap(4, 4, &GUID_WICPixelFormat32bppPBGRA, WICBitmapCacheOnLoad)?;
                        let dc2: ID2D1DeviceContext2 = self.dc.cast()?;
                        let isrc = dc2.CreateImageSourceFromWic(&wic, D2D1_IMAGE_SOURCE_LOADING_OPTIONS_NONE, D2D1_ALPHA_MODE_PREMULTIPLIED)?;
                        let rect = D2D_RECT_U { left: 0, top: 0, right: 4, bottom: 4 };
                        let ensure_null = hr_str(isrc.EnsureCached(None));
                        let ensure_rect = hr_str(isrc.EnsureCached(Some(&rect)));
                        let trim_null = hr_str(isrc.TrimCache(None));
                        let wic_src = isrc.GetSource().is_ok();
                        let is: ID2D1ImageSource = isrc.cast()?;
                        let tp = D2D1_TRANSFORMED_IMAGE_SOURCE_PROPERTIES {
                            orientation: D2D1_ORIENTATION_ROTATE_CLOCKWISE90, scaleX: 2.0, scaleY: 3.0,
                            interpolationMode: D2D1_INTERPOLATION_MODE_LINEAR, options: D2D1_TRANSFORMED_IMAGE_SOURCE_OPTIONS_NONE,
                        };
                        let tsrc = dc2.CreateTransformedImageSource(&is, &tp)?;
                        let tsrc_src = tsrc.GetSource().is_ok();
                        let mut gp = D2D1_TRANSFORMED_IMAGE_SOURCE_PROPERTIES::default();
                        tsrc.GetProperties(&mut gp);
                        // ID2D1ImageSource::OfferResources / TryReclaimResources (DXGI residency).
                        let offer = hr_str(is.OfferResources());
                        let reclaim = is.TryReclaimResources().map(|b| b.as_bool()).unwrap_or(false);
                        serde_json::json!({
                            "ensure_null": ensure_null, "ensure_rect": ensure_rect, "trim_null": trim_null,
                            "wic_source": wic_src as u32, "tsrc_source": tsrc_src as u32,
                            "props": [gp.orientation.0, gp.scaleX as i64, gp.scaleY as i64, gp.interpolationMode.0, gp.options.0],
                            "offer": offer, "reclaim": reclaim as u32,
                        })
                    }
                    // ID2D1Factory1 RegisterEffectFromStream / GetRegisteredEffects / GetEffectProperties /
                    // UnregisterEffect. The registered-effect list differs from native (our built-in set is
                    // a subset), so verify the registered-count DELTA (register +1, unregister -1), which
                    // is build-independent, plus GetEffectProperties' user-property count.
                    "effect_registry" => {
                        let f1: ID2D1Factory1 = self.factory.cast()?;
                        let mut reg0 = 0u32;
                        f1.GetRegisteredEffects(None, None, Some(&mut reg0))?;
                        let guid = windows::core::GUID::from_u128(0x0d2d_0002_1111_2222_3333_444455556666);
                        let xml = HSTRING::from(shader_effect_xml(1));
                        let stream = SHCreateMemStream(Some(xml.to_string().as_bytes())).context("SHCreateMemStream")?;
                        f1.RegisterEffectFromStream(&guid, &stream, None, Some(dummy_effect_factory))?;
                        let mut reg1 = 0u32;
                        f1.GetRegisteredEffects(None, None, Some(&mut reg1))?;
                        f1.UnregisterEffect(&guid)?;
                        let mut reg2 = 0u32;
                        f1.GetRegisteredEffects(None, None, Some(&mut reg2))?;
                        // ID2D1Factory1::GetEffectProperties on a built-in (Flood): the schema's author
                        // property + its default value, no instance needed.
                        let ep = f1.GetEffectProperties(&CLSID_D2D1Flood)?;
                        let ep_count = ep.GetPropertyCount();
                        let nlen = ep.GetPropertyNameLength(0) as usize;
                        let mut nb = vec![0u16; nlen + 1];
                        let _ = ep.GetPropertyName(0, &mut nb);
                        let ep_name = String::from_utf16_lossy(&nb[..nlen]);
                        let mut def = [0u8; 16];
                        ep.GetValue(0, D2D1_PROPERTY_TYPE_VECTOR4, &mut def)?;
                        let ep_default: Vec<f32> = def.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                        serde_json::json!({
                            "delta_register": reg1 as i64 - reg0 as i64,
                            "delta_unregister": reg1 as i64 - reg2 as i64,
                            "ep_count": ep_count, "ep_name": ep_name, "ep_default": ep_default,
                        })
                    }
                    // ID2D1Effect::GetInput/GetInputCount + ID2D1DrawingStateBlock::GetTextRenderingParams.
                    "coverage2" => {
                        let fx = self.dc.CreateEffect(&CLSID_D2D1ColorMatrix)?;
                        let in_count = fx.GetInputCount();
                        let inp = fx.GetInput(0);
                        let sb = self.factory.CreateDrawingStateBlock(None, None)?;
                        let sp = sb.GetTextRenderingParams();
                        serde_json::json!({
                            "input_count": in_count,
                            "has_input": inp.is_ok() as u32,
                            "sb_has_text_params": sp.is_ok() as u32,
                        })
                    }
                    // ID2D1Properties by-name path on a Flood effect (exposes a "Color" VECTOR4 prop):
                    // GetPropertyIndex (hit + miss), GetValueByName (read + zero-size E_INVALIDARG),
                    // SetValueByName round-trip, GetSubProperties (valid leaf + out-of-range).
                    "props_byname" => {
                        let fx = self.dc.CreateEffect(&CLSID_D2D1Flood)?;
                        let p: ID2D1Properties = fx.cast()?;
                        let color = HSTRING::from("Color");
                        let missing = HSTRING::from("NoSuchProp");
                        let idx_color = p.GetPropertyIndex(&color);
                        let idx_missing = p.GetPropertyIndex(&missing);
                        let mut buf = [0u8; 16];
                        p.GetValueByName(&color, D2D1_PROPERTY_TYPE_VECTOR4, &mut buf)?;
                        let read0: Vec<f32> = buf.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                        let zero_hr = hr_str(p.GetValueByName(&color, D2D1_PROPERTY_TYPE_VECTOR4, &mut []));
                        let set_bytes: Vec<u8> = [1.0f32, 0.0, 0.5, 1.0].iter().flat_map(|f| f.to_le_bytes()).collect();
                        let set_hr = hr_str(p.SetValueByName(&color, D2D1_PROPERTY_TYPE_VECTOR4, &set_bytes));
                        let mut buf2 = [0u8; 16];
                        p.GetValueByName(&color, D2D1_PROPERTY_TYPE_VECTOR4, &mut buf2)?;
                        let read1: Vec<f32> = buf2.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                        let sub_oob = hr_str(p.GetSubProperties(999).map(|_| ()));
                        // The valid-leaf sub-property object: DISPLAYNAME/ISREADONLY/DEFAULT (verified
                        // byte-exact vs native), MIN/MAX/FIELDS/INDEX → 0x88990029.
                        let sp = p.GetSubProperties(idx_color)?;
                        let sub_count = sp.GetPropertyCount();
                        let mut dn = vec![0u8; sp.GetValueSize(0x8000_0000) as usize];
                        sp.GetValue(0x8000_0000, D2D1_PROPERTY_TYPE_STRING, &mut dn)?;
                        let dn_str = String::from_utf16_lossy(&dn.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).take_while(|&w| w != 0).collect::<Vec<_>>());
                        let mut ro = [0u8; 4];
                        sp.GetValue(0x8000_0001, D2D1_PROPERTY_TYPE_BOOL, &mut ro)?;
                        let mut def = vec![0u8; sp.GetValueSize(0x8000_0004) as usize];
                        sp.GetValue(0x8000_0004, D2D1_PROPERTY_TYPE_VECTOR4, &mut def)?;
                        let def_f: Vec<f32> = def.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                        let sys_hr: Vec<String> = [0x8000_0002u32, 0x8000_0003, 0x8000_0005, 0x8000_0006]
                            .iter().map(|&k| hr_str(sp.GetValue(k, D2D1_PROPERTY_TYPE_UNKNOWN, &mut [0u8; 4]))).collect();
                        serde_json::json!({
                            "idx_color": idx_color, "idx_missing": idx_missing,
                            "read0": read0, "zero_hr": zero_hr, "set_hr": set_hr, "read1": read1,
                            "sub_oob": sub_oob, "sub_count": sub_count, "sub_displayname": dn_str,
                            "sub_readonly": u32::from_le_bytes(ro), "sub_default": def_f, "sub_sys_hr": sys_hr,
                        })
                    }
                    // ID2D1RenderTarget::CreateSharedBitmap(IWICBitmap) across RT types: native rejects
                    // sharing a CPU WIC bitmap into a GPU/DXGI device context (this harness's self.dc)
                    // but accepts it on a software WIC-bitmap render target.
                    "shared_bitmap" => {
                        use windows::Win32::Graphics::Imaging::{GUID_WICPixelFormat32bppPBGRA, WICBitmapCacheOnLoad, WICBitmapLockRead, IWICBitmapLock};
                        use windows::Win32::Graphics::Imaging::WICRect;
                        let bprops = D2D1_BITMAP_PROPERTIES { pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT_B8G8R8A8_UNORM, alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED }, dpiX: 96.0, dpiY: 96.0 };
                        let rect = WICRect { X: 0, Y: 0, Width: 4, Height: 4 };
                        let rtp = windows::Win32::Graphics::Direct2D::D2D1_RENDER_TARGET_PROPERTIES {
                            pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT_B8G8R8A8_UNORM, alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED },
                            ..Default::default()
                        };
                        let wicf = self.wic()?.clone();
                        let wicrt_bmp = wicf.CreateBitmap(8, 8, &GUID_WICPixelFormat32bppPBGRA, WICBitmapCacheOnLoad)?;
                        let wicrt = self.factory.CreateWicBitmapRenderTarget(&wicrt_bmp, &rtp)?;
                        // Share an IWICBitmapLock (a supported CreateSharedBitmap resource) on each RT.
                        let b1 = wicf.CreateBitmap(4, 4, &GUID_WICPixelFormat32bppPBGRA, WICBitmapCacheOnLoad)?;
                        let lock1: IWICBitmapLock = b1.Lock(&rect, WICBitmapLockRead.0 as u32)?;
                        let dc_rt: ID2D1RenderTarget = self.dc.cast()?;
                        let mut o1: Option<ID2D1Bitmap> = None;
                        let dc_lock_hr = hr_str(dc_rt.CreateSharedBitmap(&IWICBitmapLock::IID, lock1.as_raw(), Some(&bprops), &mut o1));
                        let dc_lock_ok = o1.is_some() as u32;
                        let b2 = wicf.CreateBitmap(4, 4, &GUID_WICPixelFormat32bppPBGRA, WICBitmapCacheOnLoad)?;
                        let lock2: IWICBitmapLock = b2.Lock(&rect, WICBitmapLockRead.0 as u32)?;
                        let mut o2: Option<ID2D1Bitmap> = None;
                        let wic_lock_hr = hr_str(wicrt.CreateSharedBitmap(&IWICBitmapLock::IID, lock2.as_raw(), Some(&bprops), &mut o2));
                        let wic_lock_ok = o2.is_some() as u32;
                        // Share an ID2D1Bitmap on each RT.
                        let d2dbmp = self.dc.CreateBitmap(D2D_SIZE_U { width: 4, height: 4 }, None, 0, &D2D1_BITMAP_PROPERTIES1 { pixelFormat: bprops.pixelFormat, dpiX: 96.0, dpiY: 96.0, bitmapOptions: D2D1_BITMAP_OPTIONS_NONE, colorContext: ManuallyDrop::new(None) })?;
                        let mut out_db: Option<ID2D1Bitmap> = None;
                        let dc_d2d_hr = hr_str(self.dc.CreateSharedBitmap(&ID2D1Bitmap::IID, d2dbmp.as_raw(), Some(&bprops), &mut out_db));
                        // A bare IWICBitmap (not a lock) is NOT a supported resource → E_INVALIDARG.
                        let badbmp = wicf.CreateBitmap(4, 4, &GUID_WICPixelFormat32bppPBGRA, WICBitmapCacheOnLoad)?;
                        let mut out_bad: Option<ID2D1Bitmap> = None;
                        let bad_hr = hr_str(self.dc.CreateSharedBitmap(&windows::Win32::Graphics::Imaging::IWICBitmap::IID, badbmp.as_raw(), Some(&bprops), &mut out_bad));
                        serde_json::json!({
                            "dc_lock_hr": dc_lock_hr, "dc_lock_ok": dc_lock_ok,
                            "wic_lock_hr": wic_lock_hr, "wic_lock_ok": wic_lock_ok,
                            "dc_d2d_hr": dc_d2d_hr, "dc_d2d_ok": out_db.is_some() as u32,
                            "bad_hr": bad_hr, "bad_ok": out_bad.is_some() as u32,
                        })
                    }
                    // ID2D1DeviceContext::CreateColorContextFromFilename (the system sRGB .icm, which
                    // native + we classify SRGB via logical-equality recognition) +
                    // CreateColorContextFromWicColorContext (Exif sRGB / Adobe RGB).
                    "color_context_create" => {
                        use windows::Win32::Graphics::Imaging::IWICColorContext;
                        let path = HSTRING::from(r"C:\Windows\System32\spool\drivers\color\sRGB Color Space Profile.icm");
                        let cc_file = self.dc.CreateColorContextFromFilename(&path)?;
                        let fsize = cc_file.GetProfileSize();
                        let mut fprof = vec![0u8; fsize as usize];
                        cc_file.GetProfile(&mut fprof)?;
                        let wic = self.wic()?.clone();
                        let w_srgb: IWICColorContext = wic.CreateColorContext()?;
                        w_srgb.InitializeFromExifColorSpace(1)?;
                        let cc_srgb = self.dc.CreateColorContextFromWicColorContext(&w_srgb)?;
                        let w_adobe: IWICColorContext = wic.CreateColorContext()?;
                        w_adobe.InitializeFromExifColorSpace(2)?;
                        let cc_adobe = self.dc.CreateColorContextFromWicColorContext(&w_adobe)?;
                        serde_json::json!({
                            "file_space": cc_file.GetColorSpace().0, "file_size": fsize,
                            "file_first8": fprof.get(..8).unwrap_or(&[]),
                            "wic_srgb_space": cc_srgb.GetColorSpace().0,
                            "wic_adobe_space": cc_adobe.GetColorSpace().0,
                        })
                    }
                    // Print a simple scene (one solid-colour rectangle) into a command list, then run
                    // CreatePrintControl/AddPage through the custom XPS-capture target so native's vector
                    // XPS is written to print_out.xps in the cwd for part-by-part diffing.
                    "print_capture" => {
                        use windows::Win32::System::Com::{CoCreateInstance, CLSCTX_INPROC_SERVER};
                        // `target` selects the primitive (default "rect"); an optional `@mode` suffix picks
                        // the font-subset mode (none/eachpage/default) — see record_print_primitive.
                        let tgt = target.as_deref().unwrap_or("rect");
                        let (variant, mode) = tgt.split_once('@').unwrap_or((tgt, "default"));
                        let (variant, mode) = (variant.to_string(), mode.to_string());
                        let dw = self.text_factory()?.clone();
                        // Record the page on a FRESH device context so the scene's outer BeginDraw/EndDraw
                        // lifecycle on self.dc is untouched (a nested BeginDraw is D2DERR_WRONG_STATE).
                        let device: ID2D1Device = self.dc.GetDevice()?;
                        let rec = device.CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE)?;
                        let list = rec.CreateCommandList()?;
                        let img: ID2D1Image = list.cast()?;
                        rec.SetTarget(&img);
                        rec.BeginDraw();
                        record_print_primitive(&rec, &dw, &variant)?;
                        rec.EndDraw(None, None)?;
                        list.Close()?;
                        let xps_factory: windows::Win32::Storage::Xps::IXpsOMObjectFactory =
                            unsafe { CoCreateInstance(&windows::Win32::Storage::Xps::XpsOMObjectFactory, None, CLSCTX_INPROC_SERVER)? };
                        let out = HSTRING::from("print_out.xps");
                        let _ = std::fs::remove_file("print_out.xps");
                        let target: windows::Win32::Storage::Xps::Printing::IPrintDocumentPackageTarget =
                            XpsCaptureTarget { factory: xps_factory, path: out }.into();
                        let wic = self.wic()?.clone();
                        let font_subset = match mode.as_str() {
                            "none" => D2D1_PRINT_FONT_SUBSET_MODE_NONE,
                            "eachpage" => D2D1_PRINT_FONT_SUBSET_MODE_EACHPAGE,
                            _ => D2D1_PRINT_FONT_SUBSET_MODE_DEFAULT,
                        };
                        let props = D2D1_PRINT_CONTROL_PROPERTIES { fontSubset: font_subset, ..Default::default() };
                        let pc = device.CreatePrintControl(&wic, &target, Some(&props))?;
                        let page_hr = hr_str(pc.AddPage(&list, D2D_SIZE_F { width: 100.0, height: 100.0 }, None, None, None));
                        let close_hr = hr_str(pc.Close());
                        let size = std::fs::metadata("print_out.xps").map(|m| m.len()).unwrap_or(0);
                        serde_json::json!({ "variant": variant, "page_hr": page_hr, "close_hr": close_hr, "xps_size": size })
                    }
                    // device-context / render-target state getters
                    "dc_transform" => {
                        let mut m = identity();
                        self.dc.GetTransform(&mut m);
                        serde_json::json!([m.M11, m.M12, m.M21, m.M22, m.M31, m.M32])
                    }
                    "dc_antialias" => serde_json::json!(self.dc.GetAntialiasMode().0),
                    "dc_text_antialias" => serde_json::json!(self.dc.GetTextAntialiasMode().0),
                    "dc_blend" => serde_json::json!(self.dc.GetPrimitiveBlend().0),
                    "dc_unit" => serde_json::json!(self.dc.GetUnitMode().0),
                    "dc_dpi" => {
                        let (mut x, mut y) = (0.0f32, 0.0f32);
                        self.dc.GetDpi(&mut x, &mut y);
                        serde_json::json!([x, y])
                    }
                    "dc_tags" => {
                        let (mut a, mut b) = (0u64, 0u64);
                        self.dc.GetTags(Some(&mut a), Some(&mut b));
                        serde_json::json!([a, b])
                    }
                    // solid brush
                    "brush_color" => {
                        let b: ID2D1SolidColorBrush = get(&self.brushes, t)?.cast()?;
                        let c = b.GetColor();
                        serde_json::json!([c.r, c.g, c.b, c.a])
                    }
                    "brush_opacity" => serde_json::json!(get(&self.brushes, t)?.GetOpacity()),
                    // stroke style (+ v1 transform type)
                    "stroke" => {
                        let s = get(&self.strokes, t)?;
                        let s1: ID2D1StrokeStyle1 = s.cast()?;
                        let n = s.GetDashesCount();
                        let mut dashes = vec![0f32; n as usize];
                        if n > 0 { s.GetDashes(&mut dashes); }
                        serde_json::json!({
                            "start_cap": s.GetStartCap().0, "end_cap": s.GetEndCap().0,
                            "dash_cap": s.GetDashCap().0, "line_join": s.GetLineJoin().0,
                            "miter": s.GetMiterLimit(), "dash_style": s.GetDashStyle().0,
                            "dash_offset": s.GetDashOffset(), "dash_count": n, "dashes": dashes,
                            "transform_type": s1.GetStrokeTransformType().0,
                        })
                    }
                    // geometry shape getters
                    "geom_rect" => {
                        let g: ID2D1RectangleGeometry = get(&self.geometries, t)?.cast()?;
                        let r = g.GetRect();
                        serde_json::json!([r.left, r.top, r.right, r.bottom])
                    }
                    "geom_rounded" => {
                        let g: ID2D1RoundedRectangleGeometry = get(&self.geometries, t)?.cast()?;
                        let mut rr = D2D1_ROUNDED_RECT::default();
                        g.GetRoundedRect(&mut rr);
                        serde_json::json!([rr.rect.left, rr.rect.top, rr.rect.right, rr.rect.bottom, rr.radiusX, rr.radiusY])
                    }
                    "geom_ellipse" => {
                        let g: ID2D1EllipseGeometry = get(&self.geometries, t)?.cast()?;
                        let e = g.GetEllipse();
                        serde_json::json!([e.point.x, e.point.y, e.radiusX, e.radiusY])
                    }
                    "geom_group" => {
                        let g: ID2D1GeometryGroup = get(&self.geometries, t)?.cast()?;
                        serde_json::json!([g.GetFillMode().0, g.GetSourceGeometryCount()])
                    }
                    // bitmap
                    "bitmap" => {
                        let b = get(&self.bitmaps, t)?;
                        let sz = b.GetSize();
                        let ps = b.GetPixelSize();
                        let (mut dx, mut dy) = (0.0f32, 0.0f32);
                        b.GetDpi(&mut dx, &mut dy);
                        let pf = b.GetPixelFormat();
                        serde_json::json!({
                            "size": [sz.width, sz.height], "pixel_size": [ps.width, ps.height],
                            "dpi": [dx, dy], "format": pf.format.0, "alpha": pf.alphaMode.0,
                        })
                    }
                    // color context
                    "colorctx_type" => {
                        let c: ID2D1ColorContext1 = get(&self.color_contexts, t)?.cast()?;
                        serde_json::json!(c.GetColorContextType().0)
                    }
                    // ink / ink style
                    "ink" => {
                        let k = get(&self.inks, t)?;
                        let p = k.GetStartPoint();
                        serde_json::json!({"start": [p.x, p.y, p.radius], "segments": k.GetSegmentCount()})
                    }
                    // StreamAsGeometry vertex capture: the swept-nib outline figures native emits for the
                    // ink (with the scene's single ink style), teed through RecordingSink. Diffs against
                    // our `Ink::stream_as_geometry`. The flattening tolerance is the nib's own
                    // GetInkFlatteningTolerance (round 0.5 / else 0.25) so the captured geometry matches
                    // what DrawInk fills — NOT D2D1_DEFAULT_FLATTENING_TOLERANCE (0.25), which over-flattens.
                    "ink_geom" => {
                        let k = get(&self.inks, t)?;
                        let style = self.ink_styles.values().next().cloned();
                        let tol = match style.as_ref().map(|s| s.GetNibShape().0) {
                            Some(0) | None => 0.5f32, // round
                            _ => 0.25f32,             // square
                        };
                        let path = self.factory.CreatePathGeometry()?;
                        let sink = path.Open()?;
                        let inner: ID2D1SimplifiedGeometrySink = sink.cast()?;
                        let figures = Rc::new(RefCell::new(Vec::new()));
                        let recorder: ID2D1SimplifiedGeometrySink = RecordingSink {
                            inner: inner.clone(),
                            fill_mode: Rc::new(std::cell::Cell::new(0)),
                            figures: figures.clone(),
                            cur: RefCell::new(None),
                        }
                        .into();
                        k.StreamAsGeometry(style.as_ref(), None, tol, &recorder)?;
                        sink.Close()?;
                        serde_json::json!({"figures": &*figures.borrow()})
                    }
                    "ink_style" => {
                        let s = get(&self.ink_styles, t)?;
                        let mut m = identity();
                        s.GetNibTransform(&mut m);
                        serde_json::json!({"nib_shape": s.GetNibShape().0, "nib_transform": [m.M11, m.M12, m.M21, m.M22, m.M31, m.M32]})
                    }
                    // census3 (part 3): the remaining ID2D1 getter surface, read off the same stored
                    // native objects. Each reads the real interface so the census attributes the
                    // method to this op and the recorded value is the native ground truth.
                    // --- gradient brushes + their stop collection ---
                    "grad_linear" => {
                        let b: ID2D1LinearGradientBrush = get(&self.brushes, t)?.cast()?;
                        let (sp, ep) = (b.GetStartPoint(), b.GetEndPoint());
                        let sc = b.GetGradientStopCollection()?;
                        let n = sc.GetGradientStopCount();
                        let mut stops = vec![D2D1_GRADIENT_STOP::default(); n as usize];
                        sc.GetGradientStops(&mut stops);
                        // ID2D1GradientStopCollection1: the extended color-space / buffer-precision
                        // getters (a base collection QIs to the 1-interface and reports its defaults).
                        let sc1: ID2D1GradientStopCollection1 = sc.cast()?;
                        let mut stops1 = vec![D2D1_GRADIENT_STOP::default(); n as usize];
                        sc1.GetGradientStops1(&mut stops1);
                        serde_json::json!({
                            "start": [sp.x, sp.y], "end": [ep.x, ep.y], "stop_count": n,
                            "stops": stops.iter().map(|s| serde_json::json!([s.position, s.color.r, s.color.g, s.color.b, s.color.a])).collect::<Vec<_>>(),
                            "gamma": sc.GetColorInterpolationGamma().0, "extend": sc.GetExtendMode().0,
                            "stops1": stops1.iter().map(|s| serde_json::json!([s.position, s.color.r, s.color.g, s.color.b, s.color.a])).collect::<Vec<_>>(),
                            "color_interp_mode": sc1.GetColorInterpolationMode().0,
                            "pre_space": sc1.GetPreInterpolationSpace().0,
                            "post_space": sc1.GetPostInterpolationSpace().0,
                            "buffer_precision": sc1.GetBufferPrecision().0,
                        })
                    }
                    "grad_radial" => {
                        let b: ID2D1RadialGradientBrush = get(&self.brushes, t)?.cast()?;
                        let (c, o) = (b.GetCenter(), b.GetGradientOriginOffset());
                        serde_json::json!({
                            "center": [c.x, c.y], "offset": [o.x, o.y],
                            "radius_x": b.GetRadiusX(), "radius_y": b.GetRadiusY(),
                        })
                    }
                    // --- gradient-brush mutators: set new geometry, then read it back (the by-value
                    // point setters SetEndPoint/SetCenter/SetGradientOriginOffset + the radii). ---
                    "grad_linear_set" => {
                        let b: ID2D1LinearGradientBrush = get(&self.brushes, t)?.cast()?;
                        b.SetStartPoint(point([3.0, 4.0]));
                        b.SetEndPoint(point([20.0, 25.0]));
                        let (sp, ep) = (b.GetStartPoint(), b.GetEndPoint());
                        serde_json::json!({ "start": [sp.x, sp.y], "end": [ep.x, ep.y] })
                    }
                    "grad_radial_set" => {
                        let b: ID2D1RadialGradientBrush = get(&self.brushes, t)?.cast()?;
                        b.SetCenter(point([10.0, 12.0]));
                        b.SetGradientOriginOffset(point([1.5, -2.0]));
                        b.SetRadiusX(7.5);
                        b.SetRadiusY(9.0);
                        let (c, o) = (b.GetCenter(), b.GetGradientOriginOffset());
                        serde_json::json!({
                            "center": [c.x, c.y], "offset": [o.x, o.y],
                            "radius_x": b.GetRadiusX(), "radius_y": b.GetRadiusY(),
                        })
                    }
                    // --- geometry-sink batch adds (AddLines / AddBeziers / AddQuadraticBeziers): build a
                    // throwaway closed figure with the plural methods, read its bounds. ---
                    "geom_sink_plural" => {
                        let path = self.factory.CreatePathGeometry()?;
                        let sink = path.Open()?;
                        sink.SetFillMode(D2D1_FILL_MODE_WINDING);
                        sink.BeginFigure(point([1.0, 1.0]), D2D1_FIGURE_BEGIN_FILLED);
                        sink.AddLines(&[point([10.0, 1.0]), point([10.0, 10.0])]);
                        sink.AddQuadraticBeziers(&[D2D1_QUADRATIC_BEZIER_SEGMENT { point1: point([6.0, 14.0]), point2: point([1.0, 10.0]) }]);
                        sink.AddBeziers(&[D2D1_BEZIER_SEGMENT { point1: point([0.0, 6.0]), point2: point([0.0, 3.0]), point3: point([1.0, 1.0]) }]);
                        sink.EndFigure(D2D1_FIGURE_END_CLOSED);
                        sink.Close()?;
                        let b = path.GetBounds(None)?;
                        serde_json::json!({ "bounds": [b.left, b.top, b.right, b.bottom] })
                    }
                    // --- bitmap / image brush extend + interpolation modes ---
                    "brush_extend" => {
                        let b: ID2D1BitmapBrush1 = get(&self.brushes, t)?.cast()?;
                        serde_json::json!({
                            "extend_x": b.GetExtendModeX().0, "extend_y": b.GetExtendModeY().0,
                            "interp": b.GetInterpolationMode().0, "interp1": b.GetInterpolationMode1().0,
                        })
                    }
                    "image_brush_extend" => {
                        let b: ID2D1ImageBrush = get(&self.brushes, t)?.cast()?;
                        serde_json::json!({
                            "extend_x": b.GetExtendModeX().0, "extend_y": b.GetExtendModeY().0,
                            "interp": b.GetInterpolationMode().0,
                        })
                    }
                    // --- ID2D1Bitmap1 options ---
                    "bitmap_opts" => serde_json::json!(get(&self.bitmaps, t)?.GetOptions().0),
                    // --- colour context profile data (ID2D1ColorContext / ColorContext1) ---
                    "colorctx_data" => {
                        let c = get(&self.color_contexts, t)?;
                        let n = c.GetProfileSize();
                        let mut prof = vec![0u8; n as usize];
                        if n > 0 { c.GetProfile(&mut prof)?; }
                        let c1: ID2D1ColorContext1 = c.cast()?;
                        serde_json::json!({
                            "color_space": c.GetColorSpace().0, "profile_size": n, "profile": prof,
                            "dxgi_color_space": c1.GetDXGIColorSpace().0,
                        })
                    }
                    // --- geometry source accessors (transformed geometry, geometry group) ---
                    "geom_src" => {
                        let g: ID2D1TransformedGeometry = get(&self.geometries, t)?.cast()?;
                        let mut m = identity();
                        g.GetTransform(&mut m);
                        let src = g.GetSourceGeometry()?; // ID2D1TransformedGeometry::GetSourceGeometry
                        let b = src.GetBounds(None)?;
                        serde_json::json!({"transform": [m.M11, m.M12, m.M21, m.M22, m.M31, m.M32],
                            "source_bounds": [b.left, b.top, b.right, b.bottom]})
                    }
                    "geom_group_src" => {
                        let g: ID2D1GeometryGroup = get(&self.geometries, t)?.cast()?;
                        let n = g.GetSourceGeometryCount();
                        let mut kids = vec![None; n as usize];
                        g.GetSourceGeometries(&mut kids);
                        let bounds: Vec<_> = kids.iter().filter_map(|k| k.as_ref()).map(|k| {
                            let b = k.GetBounds(None).unwrap_or_default();
                            serde_json::json!([b.left, b.top, b.right, b.bottom])
                        }).collect();
                        serde_json::json!({"count": n, "fill": g.GetFillMode().0, "child_bounds": bounds})
                    }
                    // --- CHistogram HistogramOutput (property 2, a FLOAT[NumBins] blob) ---
                    "histogram_output" => {
                        // The histogram compute runs when the effect output is drawn + the device
                        // flushes, so flush the queued draw before reading the output property.
                        self.dc.Flush(None, None)?;
                        let e = get(&self.effects, t)?;
                        let sz = e.GetValueSize(2) as usize; // D2D1_HISTOGRAM_PROP_HISTOGRAM_OUTPUT
                        let mut buf = vec![0u8; sz];
                        e.GetValue(2, D2D1_PROPERTY_TYPE_BLOB, &mut buf)?;
                        let floats: Vec<f32> = buf.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                        serde_json::json!(floats)
                    }
                    // --- effect / Properties metadata (ID2D1Properties on an ID2D1Effect) ---
                    "effect_props" => {
                        let e = get(&self.effects, t)?;
                        let n = e.GetPropertyCount();
                        let mut props = Vec::new();
                        for idx in 0..n {
                            let nl = e.GetPropertyNameLength(idx);
                            let mut nm = vec![0u16; nl as usize + 1];
                            e.GetPropertyName(idx, &mut nm)?;
                            props.push(serde_json::json!({
                                "name": String::from_utf16_lossy(&nm[..nl as usize]),
                                "type": e.GetType(idx).0, "value_size": e.GetValueSize(idx),
                            }));
                        }
                        serde_json::json!({"count": n, "input_count": e.GetInputCount(), "props": props})
                    }
                    // --- gradient mesh patches ---
                    "mesh" => {
                        let m = get(&self.meshes, t)?;
                        let n = m.GetPatchCount();
                        let mut patches = vec![D2D1_GRADIENT_MESH_PATCH::default(); n as usize];
                        if n > 0 { m.GetPatches(0, &mut patches)?; }
                        let p0 = patches.first().map(|p| serde_json::json!([p.point00.x, p.point00.y, p.point33.x, p.point33.y]));
                        // ID2D1DeviceContext2::GetGradientMeshWorldBounds — the device-space bounds.
                        let wb = self.dc2()?.GetGradientMeshWorldBounds(m)?;
                        serde_json::json!({"patch_count": n, "first_corners": p0,
                            "world_bounds": [wb.left, wb.top, wb.right, wb.bottom]})
                    }
                    // --- sprite batch sprite readback ---
                    "sprites" => {
                        let sb = get(&self.batches, t)?;
                        let n = sb.GetSpriteCount();
                        let mut dests = vec![D2D_RECT_F::default(); n as usize];
                        if n > 0 {
                            sb.GetSprites(0, n, Some(dests.as_mut_ptr()), None, None, None)?;
                        }
                        serde_json::json!({"count": n,
                            "dests": dests.iter().map(|r| serde_json::json!([r.left, r.top, r.right, r.bottom])).collect::<Vec<_>>()})
                    }
                    // --- ID2D1PathGeometry1::ComputePointAndSegmentAtLength ---
                    "path_seg" => {
                        let g: ID2D1PathGeometry1 = get(&self.geometries, t)?.cast()?;
                        let mut pd = D2D1_POINT_DESCRIPTION::default();
                        g.ComputePointAndSegmentAtLength(10.0, 0, None, 0.25, &mut pd)?;
                        serde_json::json!({
                            "point": [pd.point.x, pd.point.y], "tangent": [pd.unitTangentVector.x, pd.unitTangentVector.y],
                            "end_segment": pd.endSegment, "end_figure": pd.endFigure, "len_to_end": pd.lengthToEndSegment,
                        })
                    }
                    // --- ink segment readback + mutators (ID2D1Ink setters) ---
                    "ink_more" => {
                        let k = get(&self.inks, t)?;
                        let n = k.GetSegmentCount();
                        let mut segs = vec![D2D1_INK_BEZIER_SEGMENT::default(); n as usize];
                        if n > 0 { k.GetSegments(0, &mut segs)?; }
                        // exercise the setters on the live ink (re-sets identical geometry):
                        k.SetStartPoint(&k.GetStartPoint());
                        if n > 0 {
                            k.SetSegments(0, &segs[..1])?;
                            k.SetSegmentAtEnd(&segs[n as usize - 1])?;
                        }
                        serde_json::json!({"segments": n,
                            "first_end": segs.first().map(|s| serde_json::json!([s.point3.x, s.point3.y, s.point3.radius]))})
                    }
                    // --- device-context / factory / device capability queries (no target) ---
                    "dc_caps" => {
                        let (mut ddx, mut ddy) = (0.0f32, 0.0f32);
                        self.factory.GetDesktopDpi(&mut ddx, &mut ddy);
                        let dev = self.dc.GetDevice()?;
                        let dc7: ID2D1DeviceContext7 = self.dc.cast()?;
                        serde_json::json!({
                            "desktop_dpi": [ddx, ddy], "max_bitmap_size": self.dc.GetMaximumBitmapSize(),
                            "bgra_supported": self.dc.IsDxgiFormatSupported(DXGI_FORMAT_B8G8R8A8_UNORM).as_bool(),
                            "r8_supported": self.dc.IsDxgiFormatSupported(DXGI_FORMAT_R8_UNORM).as_bool(),
                            "prec8_supported": self.dc.IsBufferPrecisionSupported(D2D1_BUFFER_PRECISION_8BPC_UNORM).as_bool(),
                            "prec_f32_supported": self.dc.IsBufferPrecisionSupported(D2D1_BUFFER_PRECISION_32BPC_FLOAT).as_bool(),
                            "max_texture_memory": dev.GetMaximumTextureMemory(),
                            "paint_feature_level": dc7.GetPaintFeatureLevel().0,
                        })
                    }
                    // --- brush mutators: set new values, then read them back (round-trips the
                    // newly-wired Set* against native; scalar/out-param readbacks only, so the
                    // 8-byte by-value point getters are not exercised here). ---
                    "bitmap_brush_set" => {
                        let b: ID2D1BitmapBrush1 = get(&self.brushes, t)?.cast()?;
                        b.SetExtendModeX(D2D1_EXTEND_MODE_WRAP);
                        b.SetExtendModeY(D2D1_EXTEND_MODE_MIRROR);
                        b.SetInterpolationMode1(D2D1_INTERPOLATION_MODE_ANISOTROPIC);
                        b.SetOpacity(0.5);
                        b.SetTransform(&matrix([2.0, 0.0, 0.0, 3.0, 4.0, 5.0]));
                        let mut gt = identity();
                        b.GetTransform(&mut gt);
                        // SetBitmap/GetBitmap: swap in a distinct 4x4 bitmap, read its size back.
                        let nb = self.dc.CreateBitmap(
                            D2D_SIZE_U { width: 4, height: 4 }, None, 0,
                            &bitmap_properties_fmt(96., D2D1_BITMAP_OPTIONS_NONE, DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED))?;
                        b.SetBitmap(&nb);
                        let got = b.GetBitmap()?.GetPixelSize();
                        serde_json::json!({
                            "extend_x": b.GetExtendModeX().0, "extend_y": b.GetExtendModeY().0,
                            "interp": b.GetInterpolationMode().0, "interp1": b.GetInterpolationMode1().0,
                            "opacity": b.GetOpacity(), "transform": [gt.M11, gt.M12, gt.M21, gt.M22, gt.M31, gt.M32],
                            "bitmap_size": [got.width, got.height],
                        })
                    }
                    "image_brush_set" => {
                        let b: ID2D1ImageBrush = get(&self.brushes, t)?.cast()?;
                        b.SetExtendModeX(D2D1_EXTEND_MODE_WRAP);
                        b.SetExtendModeY(D2D1_EXTEND_MODE_MIRROR);
                        b.SetInterpolationMode(D2D1_INTERPOLATION_MODE_NEAREST_NEIGHBOR);
                        b.SetSourceRectangle(&D2D_RECT_F { left: 0.0, top: 0.0, right: 1.0, bottom: 1.0 });
                        b.SetOpacity(0.25);
                        let sr = b.GetSourceRectangle();
                        // SetImage/GetImage: swap in a distinct 5x3 bitmap (an ID2D1Image), read back.
                        let nb = self.dc.CreateBitmap(
                            D2D_SIZE_U { width: 5, height: 3 }, None, 0,
                            &bitmap_properties_fmt(96., D2D1_BITMAP_OPTIONS_NONE, DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED))?;
                        b.SetImage(&nb);
                        let img = b.GetImage()?;
                        let got = img.cast::<ID2D1Bitmap>()?.GetPixelSize();
                        serde_json::json!({
                            "extend_x": b.GetExtendModeX().0, "extend_y": b.GetExtendModeY().0,
                            "interp": b.GetInterpolationMode().0, "opacity": b.GetOpacity(),
                            "source": [sr.left, sr.top, sr.right, sr.bottom],
                            "image_size": [got.width, got.height],
                        })
                    }
                    "solid_set" => {
                        let b: ID2D1SolidColorBrush = get(&self.brushes, t)?.cast()?;
                        b.SetColor(&D2D1_COLOR_F { r: 0.1, g: 0.2, b: 0.3, a: 0.4 });
                        b.SetOpacity(0.75);
                        b.SetTransform(&matrix([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
                        let c = b.GetColor();
                        let mut gt = identity();
                        b.GetTransform(&mut gt);
                        serde_json::json!({
                            "color": [c.r, c.g, c.b, c.a], "opacity": b.GetOpacity(),
                            "transform": [gt.M11, gt.M12, gt.M21, gt.M22, gt.M31, gt.M32],
                        })
                    }
                    other => anyhow::bail!("unknown getprop kind {other:?}"),
                };
                self.queries.push(serde_json::json!({"op": i, "kind": format!("getprop:{kind}"), "result": result}));
            }
            Op::DwriteQuery { kind, font, text, size, weight, style, stretch, locale, params, modes, ppem } => {
                let f = self.text_factory()?.clone();
                let result = match kind.as_str() {
                    // IDWriteTextFormat: set the non-default state, then read every getter back.
                    "format" => {
                        let fmt = f.CreateTextFormat(
                            &HSTRING::from(font.as_str()), None,
                            DWRITE_FONT_WEIGHT(*weight as i32), DWRITE_FONT_STYLE(*style as i32),
                            DWRITE_FONT_STRETCH(*stretch as i32), *size, &HSTRING::from(locale.as_str()),
                        )?;
                        fmt.SetTextAlignment(DWRITE_TEXT_ALIGNMENT(modes[0] as i32))?;
                        fmt.SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT(modes[1] as i32))?;
                        fmt.SetWordWrapping(DWRITE_WORD_WRAPPING(modes[2] as i32))?;
                        fmt.SetReadingDirection(DWRITE_READING_DIRECTION(params[0] as i32))?;
                        fmt.SetFlowDirection(DWRITE_FLOW_DIRECTION(params[1] as i32))?;
                        fmt.SetIncrementalTabStop(params[2])?;
                        fmt.SetLineSpacing(DWRITE_LINE_SPACING_METHOD(params[3] as i32), *size * 1.5, *size)?;
                        let fam_n = fmt.GetFontFamilyNameLength();
                        let mut fam = vec![0u16; fam_n as usize + 1];
                        fmt.GetFontFamilyName(&mut fam)?;
                        let (mut lm, mut ls, mut lb) = (DWRITE_LINE_SPACING_METHOD_DEFAULT, 0.0f32, 0.0f32);
                        fmt.GetLineSpacing(&mut lm, &mut ls, &mut lb)?;
                        // IDWriteTextFormat1/3: set the non-default vertical/optical/last-line/fallback
                        // and variable-font axis state, then read every getter back.
                        let f3: IDWriteTextFormat3 = fmt.cast()?;
                        f3.SetVerticalGlyphOrientation(DWRITE_VERTICAL_GLYPH_ORIENTATION_STACKED)?;
                        f3.SetLastLineWrapping(false)?;
                        f3.SetOpticalAlignment(DWRITE_OPTICAL_ALIGNMENT_NO_SIDE_BEARINGS)?;
                        let sysfb = f.cast::<IDWriteFactory2>()?.GetSystemFontFallback()?;
                        f3.SetFontFallback(&sysfb)?;
                        f3.SetFontAxisValues(&[DWRITE_FONT_AXIS_VALUE { axisTag: DWRITE_FONT_AXIS_TAG_WEIGHT, value: 600.0 }])?;
                        f3.SetAutomaticFontAxes(DWRITE_AUTOMATIC_FONT_AXES_OPTICAL_SIZE)?;
                        let ac = f3.GetFontAxisValueCount();
                        let mut av = vec![DWRITE_FONT_AXIS_VALUE::default(); ac as usize];
                        if ac > 0 { f3.GetFontAxisValues(&mut av)?; }
                        serde_json::json!({
                            "weight": fmt.GetFontWeight().0, "style": fmt.GetFontStyle().0,
                            "stretch": fmt.GetFontStretch().0, "size": fmt.GetFontSize(),
                            "align": fmt.GetTextAlignment().0, "para": fmt.GetParagraphAlignment().0,
                            "wrap": fmt.GetWordWrapping().0, "reading": fmt.GetReadingDirection().0,
                            "flow": fmt.GetFlowDirection().0, "tab": fmt.GetIncrementalTabStop(),
                            "line_method": lm.0, "line_spacing": ls, "line_baseline": lb,
                            "family": String::from_utf16_lossy(&fam[..fam_n as usize]),
                            "family_len": fam_n, "locale_len": fmt.GetLocaleNameLength(),
                            "vgo": f3.GetVerticalGlyphOrientation().0,
                            "lastline": f3.GetLastLineWrapping().as_bool() as u32,
                            "optical": f3.GetOpticalAlignment().0,
                            "fallback_present": f3.GetFontFallback().is_ok() as u32,
                            "axis_count": ac,
                            "axis0_tag": av.first().map(|a| a.axisTag.0).unwrap_or(0),
                            "axis0_val": av.first().map(|a| a.value).unwrap_or(0.0),
                            "auto_axes": f3.GetAutomaticFontAxes().0,
                        })
                    }
                    // IDWriteRenderingParams1/2/3: read every getter off a custom params object.
                    "params" => {
                        let p = f.cast::<IDWriteFactory3>()?.CreateCustomRenderingParams(
                            params[0], params[1], params[2], params[3],
                            DWRITE_PIXEL_GEOMETRY(modes[0] as i32),
                            DWRITE_RENDERING_MODE1(modes[1] as i32), DWRITE_GRID_FIT_MODE(modes[2] as i32),
                        )?;
                        serde_json::json!({
                            "gamma": p.GetGamma(), "enhanced_contrast": p.GetEnhancedContrast(),
                            "grayscale_contrast": p.GetGrayscaleEnhancedContrast(),
                            "cleartype_level": p.GetClearTypeLevel(), "pixel_geometry": p.GetPixelGeometry().0,
                            "rendering_mode": p.GetRenderingMode().0, "grid_fit": p.GetGridFitMode().0,
                            "rendering_mode1": p.GetRenderingMode1().0,
                        })
                    }
                    // IDWriteFontFace/1/2/3: resolve a system face (family + weight/style) and read its
                    // metric/query getters — the same resolution path as GlyphRun.
                    "fontface" => {
                        let mut coll: Option<IDWriteFontCollection> = None;
                        f.GetSystemFontCollection(&mut coll, false)?;
                        let coll = coll.context("no system font collection")?;
                        let (mut idx, mut ex) = (0u32, windows::Win32::Foundation::BOOL(0));
                        coll.FindFamilyName(&HSTRING::from(font.as_str()), &mut idx, &mut ex)?;
                        ensure!(ex.as_bool(), "font family not found: {font}");
                        let fobj = coll.GetFontFamily(idx)?.GetFirstMatchingFont(
                            DWRITE_FONT_WEIGHT(*weight as i32), DWRITE_FONT_STRETCH(*stretch as i32),
                            DWRITE_FONT_STYLE(*style as i32))?;
                        let face = fobj.CreateFontFace()?;
                        let mut m = DWRITE_FONT_METRICS::default();
                        face.GetMetrics(&mut m);
                        let mut gm = DWRITE_FONT_METRICS::default();
                        let face1: IDWriteFontFace1 = face.cast()?;
                        face.GetGdiCompatibleMetrics(*ppem, 1.0, None, &mut gm)?;
                        let codes: Vec<u32> = text.chars().map(|c| c as u32).collect();
                        let mut ids = vec![0u16; codes.len()];
                        face.GetGlyphIndices(codes.as_ptr(), codes.len() as u32, ids.as_mut_ptr())?;
                        let mut adv = vec![0i32; ids.len()];
                        face1.GetDesignGlyphAdvances(ids.len() as u32, ids.as_ptr(), adv.as_mut_ptr(), false)?;
                        let mut kern = vec![0i32; ids.len()];
                        face1.GetKerningPairAdjustments(ids.len() as u32, ids.as_ptr(), kern.as_mut_ptr())?;
                        let mut uranges = 0u32;
                        let _ = face1.GetUnicodeRanges(None, &mut uranges);
                        let face2: IDWriteFontFace2 = face.cast()?;
                        let mut file_count = 0u32;
                        face.GetFiles(&mut file_count, None)?;
                        serde_json::json!({
                            "glyph_count": face.GetGlyphCount(), "index": face.GetIndex(),
                            "simulations": face.GetSimulations().0, "is_symbol": face.IsSymbolFont().as_bool(),
                            "upem": m.designUnitsPerEm, "ascent": m.ascent, "descent": m.descent,
                            "line_gap": m.lineGap, "cap_height": m.capHeight, "x_height": m.xHeight,
                            "gdi_ascent": gm.ascent, "gdi_descent": gm.descent, "gdi_upem": gm.designUnitsPerEm,
                            "glyphs": ids, "design_advances": adv, "kerning": kern,
                            "has_kerning": face1.HasKerningPairs().as_bool(),
                            "is_monospaced": face1.IsMonospacedFont().as_bool(), "unicode_ranges": uranges,
                            "is_color": face2.IsColorFont().as_bool(),
                            "color_palettes": face2.GetColorPaletteCount(),
                            "palette_entries": face2.GetPaletteEntryCount(),
                            "file_count": file_count,
                        })
                    }
                    // IDWriteFontFace2::GetPaletteEntries: resolve a COLR/CPAL face and read palette 0's
                    // entries (straight RGBA f32), plus the out-of-range HRESULT (span past the end).
                    "palette" => {
                        let mut coll: Option<IDWriteFontCollection> = None;
                        f.GetSystemFontCollection(&mut coll, false)?;
                        let coll = coll.context("no system font collection")?;
                        let (mut idx, mut ex) = (0u32, windows::Win32::Foundation::BOOL(0));
                        coll.FindFamilyName(&HSTRING::from(font.as_str()), &mut idx, &mut ex)?;
                        ensure!(ex.as_bool(), "font family not found: {font}");
                        let face = coll.GetFontFamily(idx)?.GetFirstMatchingFont(
                            DWRITE_FONT_WEIGHT(*weight as i32), DWRITE_FONT_STRETCH(*stretch as i32), DWRITE_FONT_STYLE(*style as i32))?.CreateFontFace()?;
                        let face2: IDWriteFontFace2 = face.cast()?;
                        let (npal, nent) = (face2.GetColorPaletteCount(), face2.GetPaletteEntryCount());
                        // Read the first min(nent, 16) entries from a mid-palette start (so the check
                        // exercises firstentryindex≠0), keeping the oracle tiny for large emoji palettes.
                        let take = nent.min(16);
                        let first = nent.saturating_sub(take).min(4);
                        let mut entries = vec![DWRITE_COLOR_F::default(); take as usize];
                        if npal > 0 && take > 0 {
                            face2.GetPaletteEntries(0, first, &mut entries)?;
                        }
                        let colors: Vec<f32> = entries.iter().flat_map(|c| [c.r, c.g, c.b, c.a]).collect();
                        // span one past the end → native E_INVALIDARG
                        let mut oob = vec![DWRITE_COLOR_F::default(); nent as usize + 1];
                        let oob_hr = face2.GetPaletteEntries(0, 0, &mut oob).err().map(|e| e.code().0 as u32).unwrap_or(0);
                        serde_json::json!({ "palettes": npal, "entries": nent, "first": first, "colors": colors, "oob_hr": format!("0x{oob_hr:08X}") })
                    }
                    // IDWriteFont(1/2): resolve a system font (family + weight/stretch/style) and read
                    // its attribute/metric getters — the IDWriteFont surface distinct from FontFace.
                    "font" => {
                        let mut coll: Option<IDWriteFontCollection> = None;
                        f.GetSystemFontCollection(&mut coll, false)?;
                        let coll = coll.context("no system font collection")?;
                        let (mut idx, mut ex) = (0u32, windows::Win32::Foundation::BOOL(0));
                        coll.FindFamilyName(&HSTRING::from(font.as_str()), &mut idx, &mut ex)?;
                        ensure!(ex.as_bool(), "font family not found: {font}");
                        let fo = coll.GetFontFamily(idx)?.GetFirstMatchingFont(
                            DWRITE_FONT_WEIGHT(*weight as i32), DWRITE_FONT_STRETCH(*stretch as i32), DWRITE_FONT_STYLE(*style as i32))?;
                        let mut m = DWRITE_FONT_METRICS::default();
                        fo.GetMetrics(&mut m);
                        let font1: IDWriteFont1 = fo.cast()?;
                        let panose = font1.GetPanose();
                        let mut uranges = 0u32;
                        let _ = font1.GetUnicodeRanges(None, &mut uranges);
                        let font2: IDWriteFont2 = fo.cast()?;
                        let ch = text.chars().next().unwrap_or('A') as u32;
                        serde_json::json!({
                            "weight": fo.GetWeight().0, "stretch": fo.GetStretch().0, "style": fo.GetStyle().0,
                            "is_symbol": fo.IsSymbolFont().as_bool(), "simulations": fo.GetSimulations().0,
                            "ascent": m.ascent, "descent": m.descent, "line_gap": m.lineGap,
                            "cap_height": m.capHeight, "x_height": m.xHeight, "upem": m.designUnitsPerEm,
                            "panose": unsafe { panose.values }.to_vec(), "unicode_ranges": uranges,
                            "is_monospaced": font1.IsMonospacedFont().as_bool(),
                            "is_color": font2.IsColorFont().as_bool(), "has_char": fo.HasCharacter(ch)?.as_bool(),
                        })
                    }
                    // IDWriteTextLayout: create a layout and read its max-size + per-range format
                    // getters (whole-text range, no overrides) + GetMetrics width. params = [maxW, maxH].
                    "layout" => {
                        let fmt = f.CreateTextFormat(
                            &HSTRING::from(font.as_str()), None,
                            DWRITE_FONT_WEIGHT(*weight as i32), DWRITE_FONT_STYLE(*style as i32),
                            DWRITE_FONT_STRETCH(*stretch as i32), *size, &HSTRING::from(locale.as_str()))?;
                        let wtext: Vec<u16> = text.encode_utf16().collect();
                        let layout = f.CreateTextLayout(&wtext, &fmt, params[0], params[1])?;
                        let mut range = DWRITE_TEXT_RANGE::default();
                        let (mut w, mut st, mut sw, mut sz) = (DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 0f32);
                        layout.GetFontWeight(0, &mut w, Some(&mut range))?;
                        layout.GetFontStyle(0, &mut st, None)?;
                        layout.GetFontStretch(0, &mut sw, None)?;
                        layout.GetFontSize(0, &mut sz, None)?;
                        let mut famlen = 0u32;
                        layout.GetFontFamilyNameLength(0, &mut famlen, None)?;
                        let mut fam = vec![0u16; famlen as usize + 1];
                        layout.GetFontFamilyName(0, &mut fam, None)?;
                        let mut loclen = 0u32;
                        layout.GetLocaleNameLength(0, &mut loclen, None)?;
                        let mut tm = DWRITE_TEXT_METRICS::default();
                        layout.GetMetrics(&mut tm)?;
                        // IDWriteTextLayout1/2/3/4: per-range pair-kerning + character spacing, the
                        // whole-layout vertical/optical/last-line/fallback state, the variable-font axes,
                        // and InvalidateLayout — set non-default values, then read them back.
                        let l4: IDWriteTextLayout4 = layout.cast()?;
                        let full = DWRITE_TEXT_RANGE { startPosition: 0, length: wtext.len() as u32 };
                        l4.SetPairKerning(true, full)?;
                        l4.SetCharacterSpacing(1.0, 2.0, 0.5, full)?;
                        l4.SetVerticalGlyphOrientation(DWRITE_VERTICAL_GLYPH_ORIENTATION_STACKED)?;
                        l4.SetLastLineWrapping(false)?;
                        l4.SetOpticalAlignment(DWRITE_OPTICAL_ALIGNMENT_NO_SIDE_BEARINGS)?;
                        let sysfb = f.cast::<IDWriteFactory2>()?.GetSystemFontFallback()?;
                        l4.SetFontFallback(&sysfb)?;
                        l4.SetFontAxisValues(&[DWRITE_FONT_AXIS_VALUE { axisTag: DWRITE_FONT_AXIS_TAG_WEIGHT, value: 600.0 }], full)?;
                        l4.SetAutomaticFontAxes(DWRITE_AUTOMATIC_FONT_AXES_OPTICAL_SIZE)?;
                        l4.InvalidateLayout()?;
                        let (mut pk, mut pkr) = (windows::Win32::Foundation::BOOL(0), DWRITE_TEXT_RANGE::default());
                        l4.GetPairKerning(0, &mut pk, Some(&mut pkr))?;
                        let (mut cl, mut ct, mut cm, mut csr) = (0f32, 0f32, 0f32, DWRITE_TEXT_RANGE::default());
                        l4.GetCharacterSpacing(0, &mut cl, &mut ct, &mut cm, Some(&mut csr))?;
                        serde_json::json!({
                            "max_width": layout.GetMaxWidth(), "max_height": layout.GetMaxHeight(),
                            "weight": w.0, "style": st.0, "stretch": sw.0, "size": sz,
                            "family": String::from_utf16_lossy(&fam[..famlen as usize]), "family_len": famlen,
                            "range_start": range.startPosition, "range_len": range.length,
                            "locale_len": loclen, "metrics_width": tm.width, "line_count": tm.lineCount,
                            "pair_kerning": pk.as_bool() as u32, "pk_range": [pkr.startPosition, pkr.length],
                            "char_leading": cl, "char_trailing": ct, "char_minadv": cm,
                            "cs_range": [csr.startPosition, csr.length],
                            "vgo": l4.GetVerticalGlyphOrientation().0,
                            "lastline": l4.GetLastLineWrapping().as_bool() as u32,
                            "optical": l4.GetOpticalAlignment().0,
                            "l_fallback_present": l4.GetFontFallback().is_ok() as u32,
                            "auto_axes": l4.GetAutomaticFontAxes().0,
                        })
                    }
                    // census3 (part 3): the remaining IDWrite getter surface. Each kind calls the
                    // real methods so the census attributes them and the recorded native value is the
                    // ground truth (the core arm records nothing — blank pixels — so these pass the
                    // conformance ratchet trivially and the drop-in is checked by the decisive run).
                    // --- IDWriteFont face/informational names + IDWriteFont3 identity ---
                    "font_names" => {
                        let mut coll: Option<IDWriteFontCollection> = None;
                        f.GetSystemFontCollection(&mut coll, false)?;
                        let coll = coll.context("no system font collection")?;
                        let (mut idx, mut ex) = (0u32, BOOL(0));
                        coll.FindFamilyName(&HSTRING::from(font.as_str()), &mut idx, &mut ex)?;
                        ensure!(ex.as_bool(), "font family not found: {font}");
                        let fo = coll.GetFontFamily(idx)?.GetFirstMatchingFont(
                            DWRITE_FONT_WEIGHT(*weight as i32), DWRITE_FONT_STRETCH(*stretch as i32), DWRITE_FONT_STYLE(*style as i32))?;
                        let names = fo.GetFaceNames()?;
                        let n = names.GetCount();
                        let slen = names.GetStringLength(0)?;
                        let mut sbuf = vec![0u16; slen as usize + 1];
                        names.GetString(0, &mut sbuf)?;
                        let llen = names.GetLocaleNameLength(0)?;
                        let mut lbuf = vec![0u16; llen as usize + 1];
                        names.GetLocaleName(0, &mut lbuf)?;
                        let (mut li, mut lex) = (0u32, BOOL(0));
                        let _ = names.FindLocaleName(w!("en-us"), &mut li, &mut lex);
                        let mut info: Option<IDWriteLocalizedStrings> = None;
                        let mut iex = BOOL(0);
                        fo.GetInformationalStrings(DWRITE_INFORMATIONAL_STRING_FULL_NAME, &mut info, &mut iex)?;
                        let f3: IDWriteFont3 = fo.cast()?;
                        let r3 = f3.GetFontFaceReference()?;
                        serde_json::json!({
                            "face_name_count": n, "face_name": String::from_utf16_lossy(&sbuf[..slen as usize]),
                            "face_locale": String::from_utf16_lossy(&lbuf[..llen as usize]),
                            "find_enus": lex.as_bool(), "find_enus_index": li,
                            "info_exists": iex.as_bool(), "info_count": info.as_ref().map(|s| s.GetCount()).unwrap_or(0),
                            "self_equals": f3.Equals(&fo).as_bool(), "locality": f3.GetLocality().0,
                            "ref_index": r3.GetFontFaceIndex(),
                        })
                    }
                    // Full system-font enumeration oracle: legacy GetSystemFontCollection (WSS model)
                    // family names/order/per-family counts + every face's reported file path, and the
                    // typographic GetSystemFontSet model (font count + CreateFontCollectionFromFontSet
                    // families). Drives the font-set rework, compared against the reference enumeration.
                    "font_enum" => {
                        let enus = |names: &IDWriteLocalizedStrings| -> Result<String> {
                            let (mut li, mut lx) = (0u32, BOOL(0));
                            let _ = names.FindLocaleName(w!("en-us"), &mut li, &mut lx);
                            let idx = if lx.as_bool() { li } else { 0 };
                            let sl = names.GetStringLength(idx)?;
                            let mut sb = vec![0u16; sl as usize + 1];
                            names.GetString(idx, &mut sb)?;
                            Ok(String::from_utf16_lossy(&sb[..sl as usize]))
                        };
                        let info = |fo: &IDWriteFont, id: DWRITE_INFORMATIONAL_STRING_ID| -> Option<String> {
                            let (mut s, mut ex) = (None, BOOL(0));
                            fo.GetInformationalStrings(id, &mut s, &mut ex).ok()?;
                            if !ex.as_bool() { return None; }
                            let s = s?;
                            let sl = s.GetStringLength(0).ok()?;
                            let mut sb = vec![0u16; sl as usize + 1];
                            s.GetString(0, &mut sb).ok()?;
                            Some(String::from_utf16_lossy(&sb[..sl as usize]))
                        };
                        let mut coll: Option<IDWriteFontCollection> = None;
                        f.GetSystemFontCollection(&mut coll, false)?;
                        let coll = coll.context("no system font collection")?;
                        let n = coll.GetFontFamilyCount();
                        let (mut fams, mut per) = (Vec::new(), Vec::new());
                        let mut paths = std::collections::BTreeSet::new();
                        let mut faces = Vec::new();
                        for i in 0..n {
                            let fam = coll.GetFontFamily(i)?;
                            fams.push(enus(&fam.GetFamilyNames()?)?);
                            let fc = fam.GetFontCount();
                            per.push(fc);
                            for j in 0..fc {
                                let fo = fam.GetFont(j)?;
                                let mut path = String::new();
                                let face = fo.CreateFontFace()?;
                                let mut nf = 0u32;
                                face.GetFiles(&mut nf, None)?;
                                let mut files = vec![None; nf as usize];
                                face.GetFiles(&mut nf, Some(files.as_mut_ptr()))?;
                                if let Some(file) = files.first().and_then(|f| f.as_ref()) {
                                    let (mut key, mut ks) = (std::ptr::null_mut(), 0u32);
                                    file.GetReferenceKey(&mut key, &mut ks)?;
                                    if let Ok(local) = file.GetLoader()?.cast::<IDWriteLocalFontFileLoader>() {
                                        let pl = local.GetFilePathLengthFromKey(key, ks)?;
                                        let mut pb = vec![0u16; pl as usize + 1];
                                        local.GetFilePathFromKey(key, ks, &mut pb)?;
                                        path = String::from_utf16_lossy(&pb[..pl as usize]);
                                        paths.insert(path.clone());
                                    }
                                }
                                faces.push(serde_json::json!({
                                    "fam": fams[i as usize], "idx": face.GetIndex(),
                                    "n1": info(&fo, DWRITE_INFORMATIONAL_STRING_WIN32_FAMILY_NAMES),
                                    "w": fo.GetWeight().0, "st": fo.GetStretch().0, "it": fo.GetStyle().0,
                                    "sim": fo.GetSimulations().0, "path": path,
                                }));
                            }
                        }
                        let f3: IDWriteFactory3 = f.cast()?;
                        let set = f3.GetSystemFontSet()?;
                        let sc = f3.CreateFontCollectionFromFontSet(&set)?;
                        let scn = sc.GetFontFamilyCount();
                        let mut setfams = Vec::new();
                        for i in 0..scn {
                            setfams.push(enus(&sc.GetFontFamily(i)?.GetFamilyNames()?)?);
                        }
                        serde_json::json!({
                            "coll_count": n, "coll_families": fams, "coll_fonts_per": per,
                            "paths": paths.into_iter().collect::<Vec<_>>(), "faces": faces,
                            "set_count": set.GetFontCount(), "setcoll_count": scn, "setcoll_families": setfams,
                        })
                    }
                    // GetFirstMatchingFont sweep oracle: for a set of families, which face native
                    // returns for each (weight, stretch, style) request.
                    "match_sweep" => {
                        let mut coll: Option<IDWriteFontCollection> = None;
                        f.GetSystemFontCollection(&mut coll, false)?;
                        let coll = coll.context("no system font collection")?;
                        let mut out = Vec::new();
                        for famname in ["Arial", "Impact", "Yu Gothic", "Bahnschrift", "Segoe UI"] {
                            let (mut idx, mut ex) = (0u32, BOOL(0));
                            coll.FindFamilyName(&HSTRING::from(famname), &mut idx, &mut ex)?;
                            if !ex.as_bool() { continue; }
                            let fam = coll.GetFontFamily(idx)?;
                            for w in [100u16, 300, 400, 500, 600, 700, 900] {
                                for st in [3u16, 5] {
                                    for style in [0i32, 1, 2] {
                                        let fo = fam.GetFirstMatchingFont(DWRITE_FONT_WEIGHT(w as i32), DWRITE_FONT_STRETCH(st as i32), DWRITE_FONT_STYLE(style))?;
                                        out.push(serde_json::json!({
                                            "fam": famname, "req": [w, st, style as u16],
                                            "got": [fo.GetWeight().0, fo.GetStretch().0, fo.GetStyle().0, fo.GetSimulations().0],
                                        }));
                                    }
                                }
                            }
                        }
                        serde_json::json!({ "sweep": out })
                    }
                    // --- IDWriteFontFace1/2/3 getters (caret, gdi metrics/advances, palette, locality) ---
                    "fontface_ex" => {
                        let face = self.resolve_face(&f, font, *weight, *style, *stretch)?;
                        let f1: IDWriteFontFace1 = face.cast()?;
                        let f2: IDWriteFontFace2 = face.cast()?;
                        let f3: IDWriteFontFace3 = face.cast()?;
                        let caret = f1.GetCaretMetrics();
                        let codes: Vec<u32> = text.chars().map(|c| c as u32).collect();
                        let codes = if codes.is_empty() { vec!['A' as u32] } else { codes };
                        let mut ids = vec![0u16; codes.len()];
                        face.GetGlyphIndices(codes.as_ptr(), codes.len() as u32, ids.as_mut_ptr())?;
                        let mut gm = vec![DWRITE_GLYPH_METRICS::default(); ids.len()];
                        face.GetGdiCompatibleGlyphMetrics(*size, 1.0, None, BOOL(1), ids.as_ptr(), ids.len() as u32, gm.as_mut_ptr(), BOOL(0))?;
                        let mut gadv = vec![0i32; ids.len()];
                        f1.GetGdiCompatibleGlyphAdvances(*size, 1.0, None, BOOL(1), BOOL(0), ids.len() as u32, ids.as_ptr(), gadv.as_mut_ptr())?;
                        let has_vert = f1.HasVerticalGlyphVariants().as_bool();
                        let mut vids = vec![0u16; ids.len()];
                        let _ = f1.GetVerticalGlyphVariants(ids.len() as u32, ids.as_ptr(), vids.as_mut_ptr());
                        let chars: Vec<u16> = text.encode_utf16().collect();
                        let chars = if chars.is_empty() { vec!['A' as u16] } else { chars };
                        let (mut td, mut ts, mut tc, mut te) = (std::ptr::null_mut(), 0u32, std::ptr::null_mut(), BOOL(0));
                        face.TryGetFontTable(u32::from_le_bytes(*b"cmap"), &mut td, &mut ts, &mut tc, &mut te)?;
                        if te.as_bool() { face.ReleaseFontTable(tc); }
                        let path = self.factory.CreatePathGeometry()?;
                        let sink = path.Open()?;
                        let inner: ID2D1SimplifiedGeometrySink = sink.cast()?;
                        // Tee the glyph outline through RecordingSink so the exact bezier stream is
                        // captured (native vs our skrifa port) — diagnosing the feather flat-top corner.
                        let go_figs = Rc::new(RefCell::new(Vec::new()));
                        let go_rec: ID2D1SimplifiedGeometrySink = RecordingSink {
                            inner: inner.clone(),
                            fill_mode: Rc::new(std::cell::Cell::new(0)),
                            figures: go_figs.clone(),
                            cur: RefCell::new(None),
                        }
                        .into();
                        face.GetGlyphRunOutline(*size, ids.as_ptr(), None, None, ids.len() as u32, BOOL(0), BOOL(0), &go_rec)?;
                        sink.Close()?;
                        drop(go_rec);
                        let r3 = f3.GetFontFaceReference()?;
                        serde_json::json!({
                            "caret_rise": caret.slopeRise, "caret_run": caret.slopeRun, "caret_offset": caret.offset,
                            "gdi_advance0": gadv[0], "gdi_metric0_adv": gm[0].advanceWidth,
                            "has_vertical": has_vert, "palette_count": f2.GetColorPaletteCount(),
                            "char_local": f3.IsCharacterLocal(codes[0]).as_bool(), "glyph_local": f3.IsGlyphLocal(ids[0]).as_bool(),
                            "all_chars_local": f3.AreCharactersLocal(&chars, BOOL(0))?.as_bool(),
                            "all_glyphs_local": f3.AreGlyphsLocal(&ids, BOOL(0))?.as_bool(),
                            "cmap_exists": te.as_bool(), "cmap_size": ts, "ref_sims": r3.GetSimulations().0,
                            "glyph_outline": go_figs.borrow().clone(),
                        })
                    }
                    // --- IDWriteFontFile + IDWriteFontFileLoader/Stream + IDWriteLocalFontFileLoader ---
                    "fontfile" => {
                        let face = self.resolve_face(&f, font, *weight, *style, *stretch)?;
                        let mut nf = 0u32;
                        face.GetFiles(&mut nf, None)?;
                        let mut files = vec![None; nf as usize];
                        face.GetFiles(&mut nf, Some(files.as_mut_ptr()))?;
                        let file = files[0].as_ref().context("no font file")?;
                        let (mut key, mut keysize) = (std::ptr::null_mut(), 0u32);
                        file.GetReferenceKey(&mut key, &mut keysize)?;
                        let key_hex = {
                            let kb = unsafe { std::slice::from_raw_parts(key as *const u8, keysize as usize) };
                            kb.iter().map(|b| format!("{b:02x}")).collect::<String>()
                        };
                        let loader = file.GetLoader()?;
                        let local: IDWriteLocalFontFileLoader = loader.cast()?;
                        let plen = local.GetFilePathLengthFromKey(key, keysize)?;
                        let mut pbuf = vec![0u16; plen as usize + 1];
                        local.GetFilePathFromKey(key, keysize, &mut pbuf)?;
                        let wt = local.GetLastWriteTimeFromKey(key, keysize)?;
                        let stream = loader.CreateStreamFromKey(key, keysize)?;
                        let (mut frag, mut fctx) = (std::ptr::null_mut(), std::ptr::null_mut());
                        stream.ReadFileFragment(&mut frag, 0, 12, &mut fctx)?;
                        stream.ReleaseFileFragment(fctx);
                        let (mut sup, mut ft, mut fct, mut nfaces) = (BOOL(0), DWRITE_FONT_FILE_TYPE(0), DWRITE_FONT_FACE_TYPE(0), 0u32);
                        file.Analyze(&mut sup, &mut ft, Some(&mut fct), &mut nfaces)?;
                        serde_json::json!({
                            "files": nf, "key_size": keysize, "key_hex": key_hex, "path": String::from_utf16_lossy(&pbuf[..plen as usize]),
                            "path_write": (wt.dwLowDateTime as u64) | ((wt.dwHighDateTime as u64) << 32),
                            "stream_size": stream.GetFileSize()?, "stream_write": stream.GetLastWriteTime().unwrap_or(0u64),
                            "supported": sup.as_bool(), "file_type": ft.0, "face_type": fct.0, "num_faces": nfaces,
                        })
                    }
                    // --- IDWriteFontFaceReference (IDWriteFactory3::CreateFontFaceReference*) ---
                    "fontref" => {
                        let face = self.resolve_face(&f, font, *weight, *style, *stretch)?;
                        let mut nf = 0u32;
                        face.GetFiles(&mut nf, None)?;
                        let mut files = vec![None; nf as usize];
                        face.GetFiles(&mut nf, Some(files.as_mut_ptr()))?;
                        let file = files[0].as_ref().context("no font file")?;
                        let (mut key, mut keysize) = (std::ptr::null_mut(), 0u32);
                        file.GetReferenceKey(&mut key, &mut keysize)?;
                        let local: IDWriteLocalFontFileLoader = file.GetLoader()?.cast()?;
                        let plen = local.GetFilePathLengthFromKey(key, keysize)?;
                        let mut pbuf = vec![0u16; plen as usize + 1];
                        local.GetFilePathFromKey(key, keysize, &mut pbuf)?;
                        let path = HSTRING::from(String::from_utf16_lossy(&pbuf[..plen as usize]));
                        let f3 = f.cast::<IDWriteFactory3>()?;
                        let r = f3.CreateFontFaceReference(file, face.GetIndex(), face.GetSimulations())?; // slot 33
                        let r2 = f3.CreateFontFaceReference2(&path, None, face.GetIndex(), face.GetSimulations())?;
                        let _ff = r.CreateFontFace()?;
                        let _ffs = r.CreateFontFaceWithSimulations(DWRITE_FONT_SIMULATIONS_BOLD)?;
                        let ftime = r.GetFileTime()?;
                        let _file2 = r.GetFontFile()?;
                        serde_json::json!({
                            "index": r.GetFontFaceIndex(), "sims": r.GetSimulations().0,
                            "local_size": r.GetLocalFileSize(), "file_size": r.GetFileSize(),
                            "locality": r.GetLocality().0, "equals_same": r.Equals(&r).as_bool(),
                            "equals_other": r.Equals(&r2).as_bool(),
                            "file_time": (ftime.dwLowDateTime as u64) | ((ftime.dwHighDateTime as u64) << 32),
                        })
                    }
                    // --- IDWriteFontSet / FontSetBuilder / FontCollection(1) ---
                    "fontset" => {
                        let f3 = f.cast::<IDWriteFactory3>()?;
                        let set = f3.GetSystemFontSet()?;
                        let n = set.GetFontCount();
                        let r0 = set.GetFontFaceReference(0)?;
                        let coll = f3.CreateFontCollectionFromFontSet(&set)?;
                        let builder = f3.CreateFontSetBuilder()?;
                        builder.AddFontFaceReference(&r0, &[])?;
                        builder.AddFontSet(&set)?;
                        let built = builder.CreateFontSet()?;
                        let cset = coll.GetFontSet()?;
                        serde_json::json!({
                            "set_count": n, "built_count": built.GetFontCount(),
                            "coll_families": coll.GetFontFamilyCount(), "coll_set_count": cset.GetFontCount(),
                            "ref0_index": r0.GetFontFaceIndex(),
                        })
                    }
                    // --- IDWriteFontCollection::GetFontFromFontFace + FontFamily1 + FontList ---
                    "collection" => {
                        let mut coll: Option<IDWriteFontCollection> = None;
                        f.GetSystemFontCollection(&mut coll, false)?;
                        let coll = coll.context("no system font collection")?;
                        let (mut idx, mut ex) = (0u32, BOOL(0));
                        coll.FindFamilyName(&HSTRING::from(font.as_str()), &mut idx, &mut ex)?;
                        let fam = coll.GetFontFamily(idx)?;
                        let fo = fam.GetFirstMatchingFont(DWRITE_FONT_WEIGHT(*weight as i32), DWRITE_FONT_STRETCH(*stretch as i32), DWRITE_FONT_STYLE(*style as i32))?;
                        let face = fo.CreateFontFace()?;
                        let back = coll.GetFontFromFontFace(&face)?;
                        let fam1: IDWriteFontFamily1 = fam.cast()?;
                        let f0 = fam1.GetFont(0)?;
                        let fr = fam1.GetFontFaceReference(0)?;
                        let list = fam.GetMatchingFonts(DWRITE_FONT_WEIGHT(*weight as i32), DWRITE_FONT_STRETCH(*stretch as i32), DWRITE_FONT_STYLE(*style as i32))?;
                        let list_coll = list.GetFontCollection()?;
                        serde_json::json!({
                            "back_weight": back.GetWeight().0, "fam1_font0_weight": f0.GetWeight().0,
                            "fam1_locality": fam1.GetFontLocality(0).0, "fam1_ref_index": fr.GetFontFaceIndex(),
                            "list_count": list.GetFontCount(), "list_coll_families": list_coll.GetFontFamilyCount(),
                        })
                    }
                    // --- IDWriteTextFormat/TextLayout setters+getters + Typography + InlineObject ---
                    "textfmt_ex" => {
                        let fmt = f.CreateTextFormat(&HSTRING::from(font.as_str()), None,
                            DWRITE_FONT_WEIGHT(*weight as i32), DWRITE_FONT_STYLE(*style as i32),
                            DWRITE_FONT_STRETCH(*stretch as i32), *size, &HSTRING::from(locale.as_str()))?;
                        let sign = f.CreateEllipsisTrimmingSign(&fmt)?;
                        fmt.SetTrimming(&DWRITE_TRIMMING { granularity: DWRITE_TRIMMING_GRANULARITY_CHARACTER, delimiter: 0, delimiterCount: 0 }, &sign)?;
                        let mut trim = DWRITE_TRIMMING::default();
                        let mut tsign: Option<IDWriteInlineObject> = None;
                        fmt.GetTrimming(&mut trim, &mut tsign)?;
                        let _fc = fmt.GetFontCollection()?;
                        let flen = fmt.GetLocaleNameLength();
                        let mut lbuf = vec![0u16; flen as usize + 1];
                        fmt.GetLocaleName(&mut lbuf)?;
                        let (mut bb, mut ba) = (DWRITE_BREAK_CONDITION(0), DWRITE_BREAK_CONDITION(0));
                        sign.GetBreakConditions(&mut bb, &mut ba)?;
                        let typo = f.CreateTypography()?;
                        typo.AddFontFeature(DWRITE_FONT_FEATURE { nameTag: DWRITE_FONT_FEATURE_TAG(u32::from_le_bytes(*b"liga")), parameter: 1 })?;
                        let feat0 = typo.GetFontFeature(0)?;
                        let wtext: Vec<u16> = text.encode_utf16().collect();
                        let wtext = if wtext.is_empty() { vec![b'A' as u16] } else { wtext };
                        let layout = f.CreateTextLayout(&wtext, &fmt, params[0].max(1.0), params[1].max(1.0))?;
                        let whole = DWRITE_TEXT_RANGE { startPosition: 0, length: wtext.len() as u32 };
                        layout.SetMaxWidth(200.0)?;
                        layout.SetMaxHeight(100.0)?;
                        layout.SetFontCollection(&_fc, whole)?;
                        layout.SetFontFamilyName(&HSTRING::from(font.as_str()), whole)?;
                        layout.SetFontWeight(DWRITE_FONT_WEIGHT_BOLD, whole)?;
                        layout.SetFontStyle(DWRITE_FONT_STYLE_ITALIC, whole)?;
                        layout.SetFontStretch(DWRITE_FONT_STRETCH_EXPANDED, whole)?;
                        layout.SetFontSize(*size + 2.0, whole)?;
                        layout.SetUnderline(BOOL(1), whole)?;
                        layout.SetStrikethrough(BOOL(1), whole)?;
                        layout.SetLocaleName(w!("en-us"), whole)?;
                        layout.SetDrawingEffect(&sign, whole)?;
                        layout.SetInlineObject(&sign, whole)?;
                        layout.SetTypography(&typo, whole)?;
                        let (mut ul, mut strike) = (BOOL(0), BOOL(0));
                        layout.GetUnderline(0, &mut ul, None)?;
                        layout.GetStrikethrough(0, &mut strike, None)?;
                        let mut lcoll: Option<IDWriteFontCollection> = None;
                        layout.GetFontCollection(0, &mut lcoll, None)?;
                        let mut lt: Option<IDWriteTypography> = None;
                        layout.GetTypography(0, &mut lt, None)?;
                        let mut lio: Option<IDWriteInlineObject> = None;
                        layout.GetInlineObject(0, &mut lio, None)?;
                        let mut lde: Option<windows::core::IUnknown> = None;
                        layout.GetDrawingEffect(0, &mut lde, None)?;
                        let mut loc_len = 0u32;
                        layout.GetLocaleNameLength(0, &mut loc_len, None)?;
                        let mut locbuf = vec![0u16; loc_len as usize + 1];
                        layout.GetLocaleName(0, &mut locbuf, None)?;
                        serde_json::json!({
                            "trim_gran": trim.granularity.0, "has_trim_sign": tsign.is_some(),
                            "fmt_locale": String::from_utf16_lossy(&lbuf[..flen as usize]),
                            "break_before": bb.0, "break_after": ba.0,
                            "feat_count": typo.GetFontFeatureCount(), "feat0_param": feat0.parameter,
                            "lay_max_w": layout.GetMaxWidth(), "lay_max_h": layout.GetMaxHeight(),
                            "lay_underline": ul.as_bool(), "lay_strike": strike.as_bool(),
                            "lay_has_typo": lt.is_some(), "lay_has_io": lio.is_some(), "lay_has_de": lde.is_some(),
                            "lay_has_coll": lcoll.is_some(),
                            "lay_locale": String::from_utf16_lossy(&locbuf[..loc_len as usize]),
                        })
                    }
                    // --- IDWriteFactory number substitution / EUDC + GdiInterop bitmap render target ---
                    "factory_ex" => {
                        let ns = f.CreateNumberSubstitution(DWRITE_NUMBER_SUBSTITUTION_METHOD_NONE, w!("en-us"), BOOL(0))?;
                        let _ = ns; // created; no readable getters on IDWriteNumberSubstitution
                        let f1 = f.cast::<IDWriteFactory1>()?;
                        let mut eudc: Option<IDWriteFontCollection> = None;
                        f1.GetEudcFontCollection(&mut eudc, BOOL(0))?;
                        let gdi = f.GetGdiInterop()?;
                        let brt = gdi.CreateBitmapRenderTarget(windows::Win32::Graphics::Gdi::HDC::default(), 32, 16)?;
                        let dc_null = brt.GetMemoryDC().0.is_null();
                        let ppd0 = brt.GetPixelsPerDip();
                        brt.SetPixelsPerDip(2.0)?;
                        let mut xf = DWRITE_MATRIX::default();
                        brt.GetCurrentTransform(&mut xf)?;
                        brt.SetCurrentTransform(Some(&xf))?;
                        brt.Resize(40, 20)?;
                        let sz = brt.GetSize()?;
                        serde_json::json!({
                            "eudc_families": eudc.as_ref().map(|c| c.GetFontFamilyCount()).unwrap_or(0),
                            "brt_memdc_null": dc_null, "brt_ppd0": ppd0, "brt_ppd1": brt.GetPixelsPerDip(),
                            "brt_m11": xf.m11, "brt_size": [sz.cx, sz.cy],
                        })
                    }
                    // IDWriteFactory6::CreateFontResource over a variable font (axis metadata), plus
                    // IDWriteFactory5::AnalyzeContainerType (WOFF/WOFF2 magic) and CreateHttpFontFileLoader.
                    "font_resource" => {
                        let dir = std::env::var("D2D_FONTS").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from("C:\\Windows\\Fonts"));
                        let file = f.CreateFontFileReference(&HSTRING::from(dir.join(font).to_string_lossy().as_ref()), None)?;
                        let f6: IDWriteFactory6 = f.cast()?;
                        let res = f6.CreateFontResource(&file, 0)?;
                        let ac = res.GetFontAxisCount();
                        let mut defs = vec![DWRITE_FONT_AXIS_VALUE::default(); ac as usize];
                        if ac > 0 { res.GetDefaultFontAxisValues(&mut defs)?; }
                        let mut ranges = vec![DWRITE_FONT_AXIS_RANGE::default(); ac as usize];
                        if ac > 0 { res.GetFontAxisRanges(&mut ranges)?; }
                        let attrs: Vec<i32> = (0..ac).map(|a| res.GetFontAxisAttributes(a).0).collect();
                        let f5: IDWriteFactory5 = f.cast()?;
                        let woff = f5.AnalyzeContainerType(b"wOFF\0\0\0\0".as_ptr() as *const _, 8).0;
                        let woff2 = f5.AnalyzeContainerType(b"wOF2\0\0\0\0".as_ptr() as *const _, 8).0;
                        let ttf = f5.AnalyzeContainerType([0u8, 1, 0, 0, 0, 0, 0, 0].as_ptr() as *const _, 8).0;
                        let http_ok = f5.CreateHttpFontFileLoader(windows::core::PCWSTR::null(), windows::core::PCWSTR::null()).is_ok();
                        serde_json::json!({
                            "axis_count": ac,
                            "defaults": defs.iter().map(|v| serde_json::json!([v.axisTag.0, v.value])).collect::<Vec<_>>(),
                            "ranges": ranges.iter().map(|r| serde_json::json!([r.axisTag.0, r.minValue, r.maxValue])).collect::<Vec<_>>(),
                            "attributes": attrs,
                            "container": [woff, woff2, ttf], "http_loader": http_ok as u32,
                        })
                    }
                    "glyph_origins" => {
                        // IDWriteFactory4::ComputeGlyphOrigins / ComputeGlyphOrigins2 — baseline +
                        // advance accumulation for a 3-glyph run.
                        let dir = std::env::var("D2D_FONTS").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from("C:\\Windows\\Fonts"));
                        let file = f.CreateFontFileReference(&HSTRING::from(dir.join(font).to_string_lossy().as_ref()), None)?;
                        let face = f.CreateFontFace(DWRITE_FONT_FACE_TYPE_TRUETYPE, &[Some(file)], 0, DWRITE_FONT_SIMULATIONS_NONE)?;
                        // windows-rs 0.58 binds the glyphOrigins out-array as a single returned point,
                        // so probe a 1-glyph run (origin = baseline).
                        let (ids, advances, offsets) = ([10u16], [12.0f32], [DWRITE_GLYPH_OFFSET::default()]);
                        let run = DWRITE_GLYPH_RUN {
                            fontFace: std::mem::ManuallyDrop::new(Some(face.clone())),
                            fontEmSize: 16.0, glyphCount: 1,
                            glyphIndices: ids.as_ptr(), glyphAdvances: advances.as_ptr(), glyphOffsets: offsets.as_ptr(),
                            isSideways: false.into(), bidiLevel: 0,
                        };
                        let f4: IDWriteFactory4 = f.cast()?;
                        let base = D2D_POINT_2F { x: 10.0, y: 20.0 };
                        let o1 = f4.ComputeGlyphOrigins(&run, base)?;
                        let o2 = f4.ComputeGlyphOrigins2(&run, DWRITE_MEASURING_MODE_NATURAL, base, None)?;
                        serde_json::json!({ "origin": [o1.x, o1.y], "origin2": [o2.x, o2.y] })
                    }
                    "enqueue" => {
                        // IDWriteFontFaceReference EnqueueFont/Character/Glyph/FileFragment DownloadRequest
                        // for a local font (nothing to download).
                        let f3: IDWriteFactory3 = f.cast()?;
                        let dir = std::env::var("D2D_FONTS").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from("C:\\Windows\\Fonts"));
                        let file = f.CreateFontFileReference(&HSTRING::from(dir.join(font).to_string_lossy().as_ref()), None)?;
                        let fref = f3.CreateFontFaceReference(&file, 0, DWRITE_FONT_SIMULATIONS_NONE)?;
                        let chars: Vec<u16> = "AB".encode_utf16().collect();
                        let glyphs = [10u16, 20];
                        serde_json::json!({
                            "font": hr_str(fref.EnqueueFontDownloadRequest()),
                            "character": hr_str(fref.EnqueueCharacterDownloadRequest(&chars)),
                            "glyph": hr_str(fref.EnqueueGlyphDownloadRequest(&glyphs)),
                            "fragment": hr_str(fref.EnqueueFileFragmentDownloadRequest(0, 100)),
                        })
                    }
                    "download_queue" => {
                        // IDWriteFontDownloadQueue IsEmpty / GetGenerationCount / BeginDownload / CancelDownload
                        // (a local-only system has an empty queue).
                        let f3: IDWriteFactory3 = f.cast()?;
                        let q = f3.GetFontDownloadQueue()?;
                        let is_empty = q.IsEmpty().as_bool();
                        let gen0 = q.GetGenerationCount();
                        let begin_hr = hr_str(q.BeginDownload(None));
                        let cancel_hr = hr_str(q.CancelDownload());
                        serde_json::json!({ "is_empty": is_empty as u32, "generation": gen0, "begin_hr": begin_hr, "cancel_hr": cancel_hr })
                    }
                    "fontset_misc" => {
                        // IDWriteFontSet2::GetExpirationEvent + IDWriteFontCollection3::GetExpirationEvent
                        // (null for a static set) + IDWriteFontSet::GetPropertyOccurrenceCount (impossible→0).
                        let f3: IDWriteFactory3 = f.cast()?;
                        let set = f3.GetSystemFontSet()?;
                        let s2: IDWriteFontSet2 = set.cast()?;
                        let exp_set = s2.GetExpirationEvent().0.is_null();
                        let f6: IDWriteFactory6 = f.cast()?;
                        let coll = f6.GetSystemFontCollection(false, DWRITE_FONT_FAMILY_MODEL_WEIGHT_STRETCH_STYLE)?;
                        let c3: IDWriteFontCollection3 = coll.cast()?;
                        let exp_coll = c3.GetExpirationEvent().0.is_null();
                        let fam: Vec<u16> = "ZZZNoSuchFamily\0".encode_utf16().collect();
                        let loc: Vec<u16> = "en-us\0".encode_utf16().collect();
                        let prop = DWRITE_FONT_PROPERTY { propertyId: DWRITE_FONT_PROPERTY_ID_FAMILY_NAME, propertyValue: windows::core::PCWSTR(fam.as_ptr()), localeName: windows::core::PCWSTR(loc.as_ptr()) };
                        let occ = set.GetPropertyOccurrenceCount(&prop)?;
                        serde_json::json!({ "exp_set_null": exp_set as u32, "exp_coll_null": exp_coll as u32, "impossible_occurrence": occ })
                    }
                    "fontset_filter" => {
                        // IDWriteFontSet1 GetFilteredFonts / GetFilteredFonts2 / GetFilteredFontIndices.
                        // The system set's total differs native-vs-ours, so use content-independent
                        // invariants: select indices [0,1,2] → 3; an impossible axis range → 0.
                        let f3: IDWriteFactory3 = f.cast()?;
                        let set = f3.GetSystemFontSet()?;
                        let s1: IDWriteFontSet1 = set.cast()?;
                        let sub = s1.GetFilteredFonts(&[0u32, 1, 2])?;
                        let impossible = [DWRITE_FONT_AXIS_RANGE { axisTag: DWRITE_FONT_AXIS_TAG_WEIGHT, minValue: 99999.0, maxValue: 99999.0 }];
                        let sub2 = s1.GetFilteredFonts2(&impossible, false)?;
                        let mut idx = [0u32; 8];
                        let mut n = 0u32;
                        s1.GetFilteredFontIndices(&impossible, false, &mut idx, &mut n)?;
                        // By-property filter with a family name no font has → empty.
                        let fam: Vec<u16> = "ZZZNoSuchFamily\0".encode_utf16().collect();
                        let loc: Vec<u16> = "en-us\0".encode_utf16().collect();
                        let prop = [DWRITE_FONT_PROPERTY { propertyId: DWRITE_FONT_PROPERTY_ID_FAMILY_NAME, propertyValue: windows::core::PCWSTR(fam.as_ptr()), localeName: windows::core::PCWSTR(loc.as_ptr()) }];
                        let sub3 = s1.GetFilteredFonts3(Some(&prop), false)?;
                        let mut idx2 = [0u32; 8];
                        let mut n2 = 0u32;
                        s1.GetFilteredFontIndices2(&prop, false, &mut idx2, &mut n2)?;
                        serde_json::json!({
                            "by_indices_count": sub.GetFontCount(),
                            "impossible_range_count": sub2.GetFontCount(),
                            "impossible_indices_count": n,
                            "impossible_prop_count": sub3.GetFontCount(),
                            "impossible_prop_indices": n2,
                        })
                    }
                    "wss_axis" => {
                        // IDWriteFontSet4::ConvertWeightStretchStyleToFontAxisValues — pure WSS+size →
                        // axis-value conversion (set-independent).
                        let f3: IDWriteFactory3 = f.cast()?;
                        let set = f3.GetSystemFontSet()?;
                        let s4: IDWriteFontSet4 = set.cast()?;
                        let cases = [
                            (700u16, 5u16, DWRITE_FONT_STYLE_NORMAL, 0.0f32),
                            (400, 3, DWRITE_FONT_STYLE_ITALIC, 24.0),
                            (900, 7, DWRITE_FONT_STYLE_OBLIQUE, 12.0),
                        ];
                        let rows: Vec<_> = cases.iter().map(|&(w, st, sty, sz)| {
                            let mut out = [DWRITE_FONT_AXIS_VALUE::default(); 5];
                            let n = s4.ConvertWeightStretchStyleToFontAxisValues(None, DWRITE_FONT_WEIGHT(w as i32), DWRITE_FONT_STRETCH(st as i32), sty, sz, &mut out);
                            serde_json::json!(out[..n as usize].iter().map(|v| serde_json::json!([v.axisTag.0, v.value])).collect::<Vec<_>>())
                        }).collect();
                        serde_json::json!({ "cases": rows })
                    }
                    "collection_model" => {
                        // IDWriteFontCollection2::GetFontFamilyModel — the model the collection was made with.
                        let f6: IDWriteFactory6 = f.cast()?;
                        let typo = f6.GetSystemFontCollection(false, DWRITE_FONT_FAMILY_MODEL_TYPOGRAPHIC)?;
                        let wss = f6.GetSystemFontCollection(false, DWRITE_FONT_FAMILY_MODEL_WEIGHT_STRETCH_STYLE)?;
                        serde_json::json!({ "typographic": typo.GetFontFamilyModel().0, "wss": wss.GetFontFamilyModel().0 })
                    }
                    "paint_reader" => {
                        // IDWriteFontFace7::CreatePaintReader + IDWritePaintReader traversal over a COLR v1
                        // glyph (`font` is loaded via the in-memory loader; gid from the cmap of 'a').
                        let bytes = std::fs::read(font)?;
                        let f5: IDWriteFactory5 = f.cast()?;
                        let loader = f5.CreateInMemoryFontFileLoader()?;
                        f5.RegisterFontFileLoader(&loader)?;
                        let file = loader.CreateInMemoryFontFileReference(&f, bytes.as_ptr() as *const _, bytes.len() as u32, None)?;
                        let face = f.CreateFontFace(DWRITE_FONT_FACE_TYPE_TRUETYPE, &[Some(file)], 0, DWRITE_FONT_SIMULATIONS_NONE)?;
                        let f7: IDWriteFontFace7 = face.cast()?;
                        let mut gid = [0u16; 1];
                        face.GetGlyphIndices([0x61u32].as_ptr(), 1, gid.as_mut_ptr())?;
                        let reader = f7.CreatePaintReader(DWRITE_GLYPH_IMAGE_FORMATS_COLR_PAINT_TREE, DWRITE_PAINT_FEATURE_LEVEL_COLR_V1)?;
                        let sz = std::mem::size_of::<DWRITE_PAINT_ELEMENT>() as u32;
                        let mut el = DWRITE_PAINT_ELEMENT::default();
                        let mut clip = D2D_RECT_F::default();
                        let mut attrs = DWRITE_PAINT_ATTRIBUTES::default();
                        reader.SetCurrentGlyph(gid[0] as u32, &mut el, sz, &mut clip, Some(&mut attrs))?;
                        let root_kind = el.paintType.0;
                        let mut children = Vec::new();
                        if reader.MoveToFirstChild(&mut el, sz).is_ok() {
                            children.push(el.paintType.0);
                            while reader.MoveToNextSibling(&mut el, sz).is_ok() {
                                children.push(el.paintType.0);
                                if children.len() > 16 { break; }
                            }
                        }
                        let parent_ok = reader.MoveToParent().is_ok();
                        // Palette/colour setters (record success).
                        let set_text = reader.SetTextColor(&DWRITE_COLOR_F { r: 1.0, g: 0.0, b: 0.0, a: 1.0 }).is_ok();
                        let set_pal = reader.SetColorPaletteIndex(0).is_ok();
                        let one = [DWRITE_COLOR_F { r: 0.5, g: 0.5, b: 0.5, a: 1.0 }];
                        let five = [DWRITE_COLOR_F { r: 0.5, g: 0.5, b: 0.5, a: 1.0 }; 5];
                        let set_custom_few = reader.SetCustomColorPalette(&one).is_ok();
                        let set_custom_full = reader.SetCustomColorPalette(&five).is_ok();
                        serde_json::json!({
                            "root_kind": root_kind, "children": children, "parent_ok": parent_ok as u32,
                            "clip": [clip.left, clip.top, clip.right, clip.bottom], "attrs": attrs.0,
                            "set_text": set_text as u32, "set_pal": set_pal as u32,
                            "set_custom_few": set_custom_few as u32, "set_custom_full": set_custom_full as u32,
                        })
                    }
                    "fontset_builder" => {
                        // IDWriteFontSetBuilder2::AddFont (in-memory font) → FontSet3 source getters +
                        // FontSet1 GetFirstFontResources/GetFontAxisRanges2.
                        let bytes = std::fs::read(font)?;
                        let f3: IDWriteFactory3 = f.cast()?;
                        let f5: IDWriteFactory5 = f.cast()?;
                        let loader = f5.CreateInMemoryFontFileLoader()?;
                        f5.RegisterFontFileLoader(&loader)?;
                        let file = loader.CreateInMemoryFontFileReference(&f, bytes.as_ptr() as *const _, bytes.len() as u32, None)?;
                        let builder = f3.CreateFontSetBuilder()?;
                        let b2: IDWriteFontSetBuilder2 = builder.cast()?;
                        b2.AddFont(&file, 0, DWRITE_FONT_SIMULATIONS_NONE, &[], &[], &[])?;
                        let set = builder.CreateFontSet()?;
                        let count = set.GetFontCount();
                        let s3: IDWriteFontSet3 = set.cast()?;
                        let src_type = s3.GetFontSourceType(0).0;
                        let name_len = s3.GetFontSourceNameLength(0);
                        let mut namebuf = vec![0u16; name_len as usize + 1];
                        let name_hr = hr_str(s3.GetFontSourceName(0, &mut namebuf));
                        let s1: IDWriteFontSet1 = set.cast()?;
                        let first = s1.GetFirstFontResources()?;
                        let first_count = first.GetFontCount();
                        // FontSet property getters: a manually-built AddFont set with no explicit
                        // properties indexes none (native + ours), so these report empty / not-exists.
                        let enus = HSTRING::from("en-us");
                        let pv_base = set.GetPropertyValues(DWRITE_FONT_PROPERTY_ID_FAMILY_NAME).map(|l| l.GetCount() as i64).unwrap_or(-1);
                        let pv2 = set.GetPropertyValues2(DWRITE_FONT_PROPERTY_ID_FAMILY_NAME, &enus).map(|l| l.GetCount() as i64).unwrap_or(-1);
                        let (mut pv3_ex, mut pv3_ls) = (BOOL(0), None);
                        let _ = set.GetPropertyValues3(0, DWRITE_FONT_PROPERTY_ID_FAMILY_NAME, &mut pv3_ex, &mut pv3_ls);
                        // FontSet query methods: FindFontFaceReference (round-trip a member's ref) and
                        // GetMatchingFonts2 (an empty property filter returns the whole set).
                        let fref = set.GetFontFaceReference(0)?;
                        let (mut fidx, mut fex) = (0u32, BOOL(0));
                        let find_hr = hr_str(set.FindFontFaceReference(&fref, &mut fidx, &mut fex));
                        // FindFontFace: a bare font face created from the set member.
                        let member_face: IDWriteFontFace = s1.CreateFontFace(0)?.cast()?;
                        let (mut ffidx, mut ffex) = (0u32, BOOL(0));
                        let ff_hr = hr_str(set.FindFontFace(&member_face, &mut ffidx, &mut ffex));
                        let matched = set.GetMatchingFonts2(&[]);
                        let matched_hr = hr_str(matched.as_ref().map(|_| ()).map_err(|e| e.clone()));
                        let matched_count = matched.map(|m| m.GetFontCount()).unwrap_or(u32::MAX);
                        // Explicit-property model: build a second set giving AddFont a FAMILY_NAME
                        // property; GetPropertyValues then serves exactly that (native's indexed store).
                        let b2b: IDWriteFontSetBuilder2 = f3.CreateFontSetBuilder()?.cast()?;
                        let fam = HSTRING::from("ZZTestFamily");
                        let loc = HSTRING::from("en-us");
                        let prop = DWRITE_FONT_PROPERTY { propertyId: DWRITE_FONT_PROPERTY_ID_FAMILY_NAME, propertyValue: windows_core::PCWSTR(fam.as_ptr()), localeName: windows_core::PCWSTR(loc.as_ptr()) };
                        b2b.AddFont(&file, 0, DWRITE_FONT_SIMULATIONS_NONE, &[], &[], &[prop])?;
                        let set2: IDWriteFontSet = b2b.cast::<IDWriteFontSetBuilder>()?.CreateFontSet()?;
                        let ex_list = set2.GetPropertyValues(DWRITE_FONT_PROPERTY_ID_FAMILY_NAME)?;
                        let ex_count = ex_list.GetCount();
                        let ex_val = if ex_count > 0 {
                            let n = ex_list.GetStringLength(0)? as usize;
                            let mut b = vec![0u16; n + 1];
                            ex_list.GetString(0, &mut b)?;
                            String::from_utf16_lossy(&b[..n])
                        } else { String::new() };
                        let mut arcnt = 0u32;
                        let _ = s1.GetFontAxisRanges2(&mut [], &mut arcnt);
                        let mut ranges = vec![DWRITE_FONT_AXIS_RANGE::default(); arcnt as usize];
                        if arcnt > 0 { s1.GetFontAxisRanges2(&mut ranges, &mut arcnt)?; }
                        let rr: Vec<_> = ranges.iter().map(|r| {
                            let t = r.axisTag.0.to_le_bytes();
                            serde_json::json!([String::from_utf8_lossy(&t).to_string(), r.minValue, r.maxValue])
                        }).collect();
                        serde_json::json!({
                            "count": count, "src_type": src_type, "name_len": name_len, "name_hr": name_hr,
                            "first_count": first_count, "axis_range_count": arcnt, "ranges": rr,
                            "find_idx": fidx, "find_exists": fex.as_bool() as u32, "find_hr": find_hr,
                            "ff_idx": ffidx, "ff_exists": ffex.as_bool() as u32, "ff_hr": ff_hr,
                            "matched_hr": matched_hr, "matched_count": matched_count,
                            "pv_base": pv_base, "pv2": pv2, "pv3_exists": pv3_ex.as_bool() as u32,
                            "ex_count": ex_count, "ex_val": ex_val,
                        })
                    }
                    "paint_gradient" => {
                        // IDWritePaintReader::GetGradientStops + GetGradientStopColors: DFS the COLR v1
                        // paint tree for the first gradient node (over every glyph), then read its stops.
                        let bytes = std::fs::read(font)?;
                        let f5: IDWriteFactory5 = f.cast()?;
                        let loader = f5.CreateInMemoryFontFileLoader()?;
                        f5.RegisterFontFileLoader(&loader)?;
                        let file = loader.CreateInMemoryFontFileReference(&f, bytes.as_ptr() as *const _, bytes.len() as u32, None)?;
                        let face = f.CreateFontFace(DWRITE_FONT_FACE_TYPE_TRUETYPE, &[Some(file)], 0, DWRITE_FONT_SIMULATIONS_NONE)?;
                        let f7: IDWriteFontFace7 = face.cast()?;
                        let reader = f7.CreatePaintReader(DWRITE_GLYPH_IMAGE_FORMATS_COLR_PAINT_TREE, DWRITE_PAINT_FEATURE_LEVEL_COLR_V1)?;
                        let sz = std::mem::size_of::<DWRITE_PAINT_ELEMENT>() as u32;
                        // Leaves the cursor at the first gradient node found; returns (paintType, stopCount).
                        fn dfs(reader: &IDWritePaintReader, el: &DWRITE_PAINT_ELEMENT, sz: u32, depth: u32) -> Option<(i32, u32)> {
                            let sc = unsafe {
                                match el.paintType.0 {
                                    4 => Some(el.paint.linearGradient.gradientStopCount),
                                    5 => Some(el.paint.radialGradient.gradientStopCount),
                                    6 => Some(el.paint.sweepGradient.gradientStopCount),
                                    _ => None,
                                }
                            };
                            if let Some(sc) = sc { return Some((el.paintType.0, sc)); }
                            if depth > 8 { return None; }
                            let mut child = DWRITE_PAINT_ELEMENT::default();
                            if unsafe { reader.MoveToFirstChild(&mut child, sz) }.is_ok() {
                                loop {
                                    if let Some(r) = dfs(reader, &child, sz, depth + 1) { return Some(r); }
                                    if unsafe { reader.MoveToNextSibling(&mut child, sz) }.is_err() { break; }
                                }
                                let _ = unsafe { reader.MoveToParent() };
                            }
                            None
                        }
                        let mut found = None;
                        for gid in 0..1024u32 {
                            let mut el = DWRITE_PAINT_ELEMENT::default();
                            let mut clip = D2D_RECT_F::default();
                            let mut attrs = DWRITE_PAINT_ATTRIBUTES::default();
                            if reader.SetCurrentGlyph(gid, &mut el, sz, &mut clip, Some(&mut attrs)).is_err() { continue; }
                            if let Some((pt, sc)) = dfs(&reader, &el, sz, 0) {
                                let mut stops = vec![D2D1_GRADIENT_STOP::default(); sc as usize];
                                reader.GetGradientStops(0, &mut stops)?;
                                let mut cols = vec![DWRITE_PAINT_COLOR::default(); sc as usize];
                                reader.GetGradientStopColors(0, &mut cols)?;
                                found = Some(serde_json::json!({
                                    "gid": gid, "paint_type": pt, "stop_count": sc,
                                    "positions": stops.iter().map(|s| s.position).collect::<Vec<_>>(),
                                    "stop_colors": stops.iter().map(|s| [s.color.r, s.color.g, s.color.b, s.color.a]).collect::<Vec<_>>(),
                                    "paint_colors": cols.iter().map(|c| [c.value.r, c.value.g, c.value.b, c.value.a]).collect::<Vec<_>>(),
                                    "palette_idx": cols.iter().map(|c| c.paletteEntryIndex).collect::<Vec<_>>(),
                                    "alpha_mult": cols.iter().map(|c| c.alphaMultiplier).collect::<Vec<_>>(),
                                }));
                                break;
                            }
                        }
                        found.unwrap_or(serde_json::json!({"found": false}))
                    }
                    "remote_loader" => {
                        // IDWriteFactory5::CreateHttpFontFileLoader + IDWriteRemoteFontFileLoader:
                        // CreateFontFileReferenceFromUrl (valid URL from D2D_REMOTE_URL) → the lazy
                        // reference's key → GetLocalityFromKey + CreateRemoteStreamFromKey. Observable
                        // pre-download state only (no BeginDownload).
                        use windows::Win32::Graphics::DirectWrite::*;
                        let url = HSTRING::from(std::env::var("D2D_REMOTE_URL").unwrap_or_default());
                        let f5: IDWriteFactory5 = f.cast()?;
                        let rl = f5.CreateHttpFontFileLoader(windows_core::PCWSTR::null(), windows_core::PCWSTR::null())?;
                        // The HTTP loader must be registered with the factory before it mints references.
                        let reg_hr = hr_str(f5.RegisterFontFileLoader(&rl));
                        let base = HSTRING::from("");
                        let (mut url_hr, mut locality, mut stream_hr, mut key_len) = ("none".to_string(), -1i32, "none".to_string(), 0u32);
                        let mut stream_state = serde_json::json!(null);
                        match rl.CreateFontFileReferenceFromUrl(&f, &base, &url) {
                            Ok(file) => {
                                url_hr = "S_OK".to_string();
                                let (mut kp, mut ks) = (core::ptr::null_mut::<core::ffi::c_void>(), 0u32);
                                if file.GetReferenceKey(&mut kp, &mut ks).is_ok() {
                                    key_len = ks;
                                    locality = rl.GetLocalityFromKey(kp, ks).map(|l| l.0).unwrap_or(-2);
                                    match rl.CreateRemoteStreamFromKey(kp, ks) {
                                        Ok(stream) => {
                                            stream_hr = "S_OK".to_string();
                                            // The stream's pre-download state.
                                            let (mut islocal, mut partial) = (BOOL(0), 0u64);
                                            let frag_hr = hr_str(stream.GetFileFragmentLocality(0, 4, &mut islocal, &mut partial));
                                            let ls = stream.GetLocalFileSize().map(|v| v as i64).unwrap_or(-1);
                                            let sloc = stream.GetLocality().0;
                                            // BeginDownload the whole file, wait, then re-read the stream.
                                            unsafe extern "system" { fn WaitForSingleObject(h: *mut core::ffi::c_void, ms: u32) -> u32; }
                                            let gid = windows::core::GUID::from_u128(0);
                                            let fsize_pre = hr_str(stream.GetFileSize().map(|_| ()));
                                            // Try BeginDownload variants to learn native's precondition.
                                            let begin = |frags: &[DWRITE_FILE_FRAGMENT]| -> String {
                                                match stream.BeginDownload(&gid, frags) {
                                                    Ok(ar) => { let h = ar.GetWaitHandle(); if !h.is_invalid() { WaitForSingleObject(h.0 as *mut _, 10000); } format!("ok:{}", hr_str(ar.GetResult())) }
                                                    Err(e) => hr_str(Err(e)),
                                                }
                                            };
                                            let dl_empty = begin(&[]);
                                            let dl_one = begin(&[DWRITE_FILE_FRAGMENT { fileOffset: 0, fragmentSize: 1 }]);
                                            let fsize_post = stream.GetFileSize().map(|v| v as i64).unwrap_or(-1);
                                            let lsize_post = stream.GetLocalFileSize().map(|v| v as i64).unwrap_or(-1);
                                            stream_state = serde_json::json!({ "frag_hr": frag_hr, "frag_local": islocal.as_bool() as u32, "frag_partial": partial, "local_size": ls, "stream_locality": sloc,
                                                "fsize_pre": fsize_pre, "dl_empty": dl_empty, "dl_one": dl_one, "fsize_post": fsize_post, "lsize_post": lsize_post });
                                        }
                                        Err(e) => stream_hr = hr_str(Err(e)),
                                    };
                                }
                            }
                            Err(e) => url_hr = hr_str(Err(e)),
                        }
                        serde_json::json!({ "reg_hr": reg_hr, "url_hr": url_hr, "locality": locality, "stream_hr": stream_hr, "key_len": key_len, "stream_state": stream_state })
                    }
                    "glyph_formats_mem" => {
                        // Load `font` (cwd-relative) via the in-memory loader and probe per-glyph
                        // GetGlyphImageFormats for the SVG (gid 1), CBDT (gid 2 @ppem16) and sbix
                        // (gid 3 @ppem24) glyphs across ppem ranges, to confirm the bitmap/SVG branch.
                        let bytes = std::fs::read(font)?;
                        let f5: IDWriteFactory5 = f.cast()?;
                        let loader = f5.CreateInMemoryFontFileLoader()?;
                        f5.RegisterFontFileLoader(&loader)?;
                        let file = loader.CreateInMemoryFontFileReference(&f, bytes.as_ptr() as *const _, bytes.len() as u32, None)?;
                        let face = f.CreateFontFace(DWRITE_FONT_FACE_TYPE_TRUETYPE, &[Some(file)], 0, DWRITE_FONT_SIMULATIONS_NONE)?;
                        let f4: IDWriteFontFace4 = face.cast()?;
                        let probe = |gid: u16, ranges: &[(u32, u32)]| -> Vec<i32> {
                            ranges.iter().map(|&(lo, hi)| f4.GetGlyphImageFormats(gid, lo, hi).map(|x| x.0).unwrap_or(-1)).collect()
                        };
                        // IDWriteFontFace4::GetGlyphImageData for the sbix PNG glyph (gid 3 @ppem 24).
                        let mut gdata = DWRITE_GLYPH_IMAGE_DATA::default();
                        let mut gctx: *mut core::ffi::c_void = core::ptr::null_mut();
                        let (img_size, px, first4) = if f4.GetGlyphImageData(3, 24, DWRITE_GLYPH_IMAGE_FORMATS_PNG, &mut gdata, &mut gctx).is_ok() {
                            let bytes = unsafe { core::slice::from_raw_parts(gdata.imageData as *const u8, gdata.imageDataSize as usize) };
                            let f4b: Vec<u8> = bytes.iter().take(4).copied().collect();
                            let r = (gdata.imageDataSize, [gdata.pixelSize.width, gdata.pixelSize.height], f4b);
                            f4.ReleaseGlyphImageData(gctx);
                            r
                        } else { (0, [0, 0], vec![]) };
                        serde_json::json!({
                            "all": f4.GetGlyphImageFormats2().0,
                            "svg_gid1": probe(1, &[(0, 0), (0, 24), (100, 200)]),
                            "cbdt_gid2": probe(2, &[(16, 16), (0, 16), (24, 24), (0, 8)]),
                            "sbix_gid3": probe(3, &[(24, 24), (16, 24), (0, 16)]),
                            "img_data_size": img_size, "img_pixel_size": px, "img_first4": first4,
                        })
                    }
                    "glyph_formats" => {
                        // Probe matrix: GetGlyphImageFormats per glyph × ppem, to port native's rule.
                        let dir = std::env::var("D2D_FONTS").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from("C:\\Windows\\Fonts"));
                        let file = f.CreateFontFileReference(&HSTRING::from(dir.join(font).to_string_lossy().as_ref()), None)?;
                        let files = [Some(file.clone())];
                        let face = f.CreateFontFace(DWRITE_FONT_FACE_TYPE_TRUETYPE, &files, 0, DWRITE_FONT_SIMULATIONS_NONE)?;
                        let f4: IDWriteFontFace4 = face.cast()?;
                        let chars: [u32; 4] = [0x20, 0x41, 0x1F600, 0x263A]; // space, 'A', emoji, ☺
                        let mut cgids = [0u16; 4];
                        face.GetGlyphIndices(chars.as_ptr(), chars.len() as u32, cgids.as_mut_ptr())?;
                        let gids: Vec<u16> = std::iter::once(0u16).chain(cgids).collect();
                        let ranges = [(0u32, 0u32), (0, 24), (12, 24)];
                        let glyphs: Vec<_> = gids.iter().map(|&gid| {
                            let per: Vec<i32> = ranges.iter().map(|&(lo, hi)| f4.GetGlyphImageFormats(gid, lo, hi).map(|x| x.0).unwrap_or(-1)).collect();
                            serde_json::json!({"gid": gid, "f": per})
                        }).collect();
                        serde_json::json!({"all": f4.GetGlyphImageFormats2().0, "glyphs": glyphs})
                    }
                    "face_misc" => {
                        // IDWriteFontFace4::GetGlyphImageFormats + IDWriteFontFace5::GetFontResource.
                        let dir = std::env::var("D2D_FONTS").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from("C:\\Windows\\Fonts"));
                        let file = f.CreateFontFileReference(&HSTRING::from(dir.join(font).to_string_lossy().as_ref()), None)?;
                        let files = [Some(file.clone())];
                        let face = f.CreateFontFace(DWRITE_FONT_FACE_TYPE_TRUETYPE, &files, 0, DWRITE_FONT_SIMULATIONS_NONE)?;
                        let f4: IDWriteFontFace4 = face.cast()?;
                        let all_formats = f4.GetGlyphImageFormats2().0;
                        let f5: IDWriteFontFace5 = face.cast()?;
                        let res = f5.GetFontResource()?;
                        // ponytail: per-glyph GetGlyphImageFormats(gid, ppem..) left uncovered — native
                        // reports 0 for an outline-only .notdef where ours reports TRUETYPE; the exact
                        // per-glyph format rule needs its own oracle.
                        serde_json::json!({ "glyph_image_formats": all_formats, "resource_axis_count": res.GetFontAxisCount() })
                    }
                    "gdi_face" => {
                        // IDWriteGdiInterop::CreateFontFaceFromHdc + IDWriteFactory::CreateMonitorRenderingParams.
                        use windows::Win32::Graphics::Gdi::{CreateCompatibleDC, CreateFontIndirectW, DeleteDC, DeleteObject, SelectObject, HGDIOBJ, LOGFONTW};
                        let gi = f.GetGdiInterop()?;
                        let hdc = CreateCompatibleDC(None);
                        let mut lf = LOGFONTW { lfHeight: -16, ..Default::default() };
                        for (i, c) in font.trim_end_matches(".ttf").encode_utf16().take(31).enumerate() { lf.lfFaceName[i] = c; }
                        let hfont = CreateFontIndirectW(&lf);
                        let old = SelectObject(hdc, HGDIOBJ(hfont.0));
                        let glyph_count = gi.CreateFontFaceFromHdc(hdc).map(|face| face.GetGlyphCount()).unwrap_or(0);
                        SelectObject(hdc, old);
                        let _ = DeleteObject(HGDIOBJ(hfont.0));
                        let _ = DeleteDC(hdc);
                        // CreateMonitorRenderingParams for the primary monitor.
                        use windows::Win32::Graphics::Gdi::{MonitorFromPoint, MONITOR_DEFAULTTOPRIMARY};
                        let mon = MonitorFromPoint(windows::Win32::Foundation::POINT { x: 0, y: 0 }, MONITOR_DEFAULTTOPRIMARY);
                        let mparams = f.CreateMonitorRenderingParams(mon)?;
                        serde_json::json!({
                            "glyph_count": glyph_count,
                            "mon_gamma": mparams.GetGamma(), "mon_contrast": mparams.GetEnhancedContrast(),
                            "mon_cleartype": mparams.GetClearTypeLevel(), "mon_geometry": mparams.GetPixelGeometry().0, "mon_mode": mparams.GetRenderingMode().0,
                        })
                    }
                    "gdi_interop" => {
                        // IDWriteGdiInterop CreateFontFromLOGFONT → ConvertFontToLOGFONT round-trip
                        // for a system face: the facename/weight/italic/isSystemFont native reports.
                        use windows::Win32::Graphics::Gdi::LOGFONTW;
                        let gi = f.GetGdiInterop()?;
                        let mut lf = LOGFONTW { lfWeight: *weight as i32, lfItalic: if *style != 0 { 1 } else { 0 }, ..Default::default() };
                        for (i, c) in font.encode_utf16().take(31).enumerate() { lf.lfFaceName[i] = c; }
                        let fnt = gi.CreateFontFromLOGFONT(&lf)?;
                        let mut out = LOGFONTW::default();
                        let mut is_sys = windows::Win32::Foundation::BOOL(0);
                        gi.ConvertFontToLOGFONT(&fnt, &mut out, &mut is_sys)?;
                        let end = out.lfFaceName.iter().position(|&c| c == 0).unwrap_or(32);
                        serde_json::json!({
                            "face": String::from_utf16_lossy(&out.lfFaceName[..end]),
                            "weight": out.lfWeight, "italic": out.lfItalic as i32, "is_system": is_sys.0,
                        })
                    }
                    "font_axis_names" => {
                        // IDWriteFontResource::GetAxisNames / GetAxisValueNameCount — per-axis name
                        // (fvar name) and STAT value-name count. Readback compared native-vs-ours.
                        let dir = std::env::var("D2D_FONTS").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from("C:\\Windows\\Fonts"));
                        let file = f.CreateFontFileReference(&HSTRING::from(dir.join(font).to_string_lossy().as_ref()), None)?;
                        let f6: IDWriteFactory6 = f.cast()?;
                        let res = f6.CreateFontResource(&file, 0)?;
                        let ac = res.GetFontAxisCount();
                        let mut axes = Vec::new();
                        for a in 0..ac {
                            let name = match res.GetAxisNames(a) {
                                Ok(strs) => {
                                    let n = strs.GetStringLength(0).unwrap_or(0) as usize;
                                    let mut buf = vec![0u16; n + 1];
                                    let name = if strs.GetString(0, &mut buf).is_ok() { String::from_utf16_lossy(&buf[..n]) } else { String::new() };
                                    name
                                }
                                Err(_) => String::new(),
                            };
                            let vn = res.GetAxisValueNameCount(a);
                            let mut value_names = Vec::new();
                            for vi in 0..vn {
                                let mut range = DWRITE_FONT_AXIS_RANGE::default();
                                let mut strs: Option<IDWriteLocalizedStrings> = None;
                                if res.GetAxisValueNames(a, vi, &mut range, &mut strs).is_ok() {
                                    if let Some(strs) = strs {
                                        let n = strs.GetStringLength(0).unwrap_or(0) as usize;
                                        let mut buf = vec![0u16; n + 1];
                                        let s = if strs.GetString(0, &mut buf).is_ok() { String::from_utf16_lossy(&buf[..n]) } else { String::new() };
                                        value_names.push(serde_json::json!([s, range.minValue, range.maxValue]));
                                    }
                                }
                            }
                            axes.push(serde_json::json!([name, vn, value_names]));
                        }
                        serde_json::json!({ "axis_count": ac, "axes": axes })
                    }
                    other => anyhow::bail!("unknown dwrite_query kind {other:?}"),
                };
                self.queries.push(serde_json::json!({"op": i, "kind": format!("dwrite:{kind}"), "result": result}));
            }
            Op::WicQuery { kind } => {
                let wic = self.wic()?.clone();
                let getstr = |f: &dyn Fn(&mut [u16], *mut u32) -> windows::core::Result<()>| -> String {
                    let mut n = 0u32;
                    let _ = f(&mut [], &mut n);
                    if n == 0 { return String::new(); }
                    let mut buf = vec![0u16; n as usize];
                    if f(&mut buf, &mut n).is_err() { return String::new(); }
                    String::from_utf16_lossy(&buf[..n.saturating_sub(1) as usize])
                };
                let result = match kind.as_str() {
                    // IWICComponentInfo / IWICBitmapDecoderInfo on the built-in PNG decoder.
                    "decoderinfo" => {
                        let di: IWICBitmapDecoderInfo = wic.CreateComponentInfo(&CLSID_WICPngDecoder)?.cast()?;
                        let (mut pat_count, mut pat_cb) = (0u32, 0u32);
                        let _ = di.GetPatterns(0, None, Some(&mut pat_count), &mut pat_cb);
                        // GetSigningStatus/MatchesMimeType can return E_NOTIMPL on native — that
                        // response is itself the ground truth, so record it rather than bailing.
                        serde_json::json!({
                            "signing_status": di.GetSigningStatus().ok(),
                            "manufacturer": getstr(&|b, p| unsafe { di.GetDeviceManufacturer(b, p) }),
                            "models": getstr(&|b, p| unsafe { di.GetDeviceModels(b, p) }),
                            "mime_png": di.MatchesMimeType(w!("image/png")).map(|b| b.as_bool()).unwrap_or(false),
                            "mime_jpeg": di.MatchesMimeType(w!("image/jpeg")).map(|b| b.as_bool()).unwrap_or(false),
                            "pattern_count": pat_count,
                        })
                    }
                    // IWICBitmap resolution + IWICBitmapLock stride on a small PBGRA bitmap.
                    "bitmap" => {
                        let bmp = wic.CreateBitmapFromMemory(
                            2, 2, &GUID_WICPixelFormat32bppPBGRA, 8, &[0u8; 16])?;
                        bmp.SetResolution(120.0, 72.0)?;
                        let (mut dx, mut dy) = (0.0f64, 0.0f64);
                        bmp.GetResolution(&mut dx, &mut dy)?;
                        let r = WICRect { X: 0, Y: 0, Width: 2, Height: 2 };
                        let lock = bmp.Lock(&r, WICBitmapLockRead.0 as u32)?;
                        serde_json::json!({
                            "dpi_x": dx, "dpi_y": dy, "stride": lock.GetStride()?,
                        })
                    }
                    // IWICFormatConverter::CanConvert for a few format pairs.
                    "converter" => {
                        let c = wic.CreateFormatConverter()?;
                        serde_json::json!({
                            "pbgra_to_bgr": c.CanConvert(&GUID_WICPixelFormat32bppPBGRA, &GUID_WICPixelFormat24bppBGR)?.as_bool(),
                            "pbgra_to_gray8": c.CanConvert(&GUID_WICPixelFormat32bppPBGRA, &GUID_WICPixelFormat8bppGray)?.as_bool(),
                        })
                    }
                    // census3 (part 3): encode a 2x2 BGRA bitmap to PNG in a growable memory IStream,
                    // then decode it back — exercising the encoder/frame-encoder metadata setters and
                    // the decoder preview/thumbnail/frame readback + decoder-info capability probing.
                    // PNG has no thumbnail/preview/colour-context support, so those record native HRESULTs.
                    "codec" => {
                        let src = wic.CreateBitmapFromMemory(2, 2, &GUID_WICPixelFormat32bppBGRA, 8, &[64u8; 16])?;
                        let mem = SHCreateMemStream(None).context("SHCreateMemStream failed")?;
                        let stream = wic.CreateStream()?;
                        stream.InitializeFromIStream(&mem)?; // IWICStream::InitializeFromIStream
                        let enc = wic.CreateEncoder(&GUID_ContainerFormatPng, std::ptr::null())?;
                        enc.Initialize(&stream, WICBitmapEncoderNoCache)?;
                        let _enc_info = enc.GetEncoderInfo()?; // IWICBitmapEncoder::GetEncoderInfo
                        let enc_set_cc = hr_str(enc.SetColorContexts(&[]));
                        let enc_set_thumb = hr_str(enc.SetThumbnail(&src));
                        let enc_set_prev = hr_str(enc.SetPreview(&src));
                        let mut frame: Option<IWICBitmapFrameEncode> = None;
                        let mut bag = None;
                        enc.CreateNewFrame(&mut frame, &mut bag)?;
                        let frame = frame.context("no frame")?;
                        frame.Initialize(bag.as_ref())?;
                        frame.SetSize(2, 2)?;
                        let mut fmt = GUID_WICPixelFormat32bppBGRA;
                        frame.SetPixelFormat(&mut fmt)?;
                        let fr_set_cc = hr_str(frame.SetColorContexts(&[]));
                        let fr_set_thumb = hr_str(frame.SetThumbnail(&src));
                        frame.WritePixels(2, 8, &[64u8; 16])?;
                        frame.Commit()?;
                        enc.Commit()?;
                        mem.Seek(0, STREAM_SEEK_SET, None)?;
                        let dec = wic.CreateDecoderFromStream(&stream, std::ptr::null(), WICDecodeMetadataCacheOnDemand)?;
                        let preview = hr_str(dec.GetPreview().map(|_| ()));
                        let thumb = hr_str(dec.GetThumbnail().map(|_| ()));
                        let frame0 = dec.GetFrame(0)?;
                        let frame_thumb = hr_str(frame0.GetThumbnail().map(|_| ()));
                        // IWICBitmapSourceTransform lives on the frame decode (as for CopyPixels/GetClosestSize).
                        let st: IWICBitmapSourceTransform = frame0.cast()?;
                        let mut dstfmt = GUID_WICPixelFormat32bppBGRA;
                        let gcp = hr_str(st.GetClosestPixelFormat(&mut dstfmt));
                        let supports = st.DoesSupportTransform(WICBitmapTransformRotate0)?.as_bool();
                        let di: IWICBitmapDecoderInfo = wic.CreateComponentInfo(&CLSID_WICPngDecoder)?.cast()?;
                        mem.Seek(0, STREAM_SEEK_SET, None)?;
                        let matches = di.MatchesPattern(&mem).map(|b| b.as_bool()).unwrap_or(false);
                        let fresh = di.CreateInstance()?;
                        mem.Seek(0, STREAM_SEEK_SET, None)?;
                        let cap = fresh.QueryCapability(&mem).unwrap_or(0);
                        serde_json::json!({
                            "enc_set_cc": enc_set_cc, "enc_set_thumb": enc_set_thumb, "enc_set_prev": enc_set_prev,
                            "fr_set_cc": fr_set_cc, "fr_set_thumb": fr_set_thumb,
                            "preview": preview, "thumb": thumb, "frame_thumb": frame_thumb,
                            "matches_pattern": matches, "query_capability": cap,
                            "closest_pixel_format": gcp, "supports_rotate0": supports,
                        })
                    }
                    // census3: palette copy, pixel-format colour context, source transform, the
                    // CreateDecoder/CreateBitmapFromSource/CreateQueryWriter factory paths, metadata
                    // writer location/remove, encoder-info CreateInstance, and IStream region init.
                    "misc" => {
                        let src = wic.CreateBitmapFromMemory(4, 4, &GUID_WICPixelFormat32bppBGRA, 16, &[32u8; 64])?;
                        let p1 = wic.CreatePalette()?;
                        p1.InitializePredefined(WICBitmapPaletteTypeFixedHalftone256, false)?;
                        let p2 = wic.CreatePalette()?;
                        p2.InitializeFromPalette(&p1)?; // IWICPalette::InitializeFromPalette
                        let palette_copy_count = p2.GetColorCount()?;
                        let pfi: IWICPixelFormatInfo = wic.CreateComponentInfo(&GUID_WICPixelFormat32bppBGRA)?.cast()?;
                        let pf_cc = hr_str(pfi.GetColorContext().map(|_| ()));
                        let copy = wic.CreateBitmapFromSource(&src, WICBitmapNoCache)?;
                        let (mut cw, mut ch) = (0u32, 0u32);
                        copy.GetSize(&mut cw, &mut ch)?;
                        let _dec = wic.CreateDecoder(&GUID_ContainerFormatPng, std::ptr::null())?;
                        let ei: IWICBitmapEncoderInfo = wic.CreateComponentInfo(&CLSID_WICPngEncoder)?.cast()?;
                        let _fresh_enc = ei.CreateInstance()?; // IWICBitmapEncoderInfo::CreateInstance
                        let qw = wic.CreateQueryWriter(&GUID_MetadataFormatChunktEXt, std::ptr::null())?;
                        let mut nsbuf = vec![0u16; 64];
                        let mut nslen = 0u32;
                        let loc = hr_str(qw.GetLocation(&mut nsbuf, &mut nslen));
                        let rm = hr_str(qw.RemoveMetadataByName(w!("/tEXt/NoSuchKey")));
                        let mem = SHCreateMemStream(Some(&[0u8; 32])).context("SHCreateMemStream failed")?;
                        let region = wic.CreateStream()?;
                        region.InitializeFromIStreamRegion(&mem, 0, 16)?; // IWICStream::InitializeFromIStreamRegion
                        serde_json::json!({
                            "palette_copy_count": palette_copy_count, "pf_colorcontext": pf_cc,
                            "copy_size": [cw, ch], "meta_location": loc, "meta_location_len": nslen,
                            "meta_remove": rm,
                        })
                    }
                    other => anyhow::bail!("unknown wic_query kind {other:?}"),
                };
                self.queries.push(serde_json::json!({"op": i, "kind": format!("wic:{kind}"), "result": result}));
            }
        }
        Ok(())
    }

    // Resolve a system font family + weight/style/stretch to an IDWriteFontFace, the same path the
    // `fontface`/`font` query kinds use; shared by the census3 DirectWrite readback kinds.
    unsafe fn resolve_face(&self, f: &IDWriteFactory, family: &str, weight: u32, style: u32, stretch: u32) -> Result<IDWriteFontFace> {
        let mut coll: Option<IDWriteFontCollection> = None;
        f.GetSystemFontCollection(&mut coll, false)?;
        let coll = coll.context("no system font collection")?;
        let (mut idx, mut ex) = (0u32, windows::Win32::Foundation::BOOL(0));
        coll.FindFamilyName(&HSTRING::from(family), &mut idx, &mut ex)?;
        ensure!(ex.as_bool(), "font family not found: {family}");
        Ok(coll.GetFontFamily(idx)?.GetFirstMatchingFont(
            DWRITE_FONT_WEIGHT(weight as i32), DWRITE_FONT_STRETCH(stretch as i32),
            DWRITE_FONT_STYLE(style as i32))?.CreateFontFace()?)
    }

    unsafe fn dc2(&self) -> Result<ID2D1DeviceContext2> {
        Ok(self.dc.cast()?)
    }
    unsafe fn dc3(&self) -> Result<ID2D1DeviceContext3> {
        Ok(self.dc.cast()?)
    }
    unsafe fn dc5(&self) -> Result<ID2D1DeviceContext5> {
        Ok(self.dc.cast()?)
    }
    unsafe fn dc6(&self) -> Result<ID2D1DeviceContext6> {
        Ok(self.dc.cast()?)
    }
    unsafe fn dc7(&self) -> Result<ID2D1DeviceContext7> {
        Ok(self.dc.cast()?)
    }
    unsafe fn dc4(&self) -> Result<ID2D1DeviceContext4> {
        Ok(self.dc.cast()?)
    }
    /// Build a single-glyph-run face + run from inline TTF bytes (shared by the color-bitmap / SVG
    /// glyph-run ops). Returns the face (keep alive), the glyph ids, advances, and offsets.
    unsafe fn inline_run(&mut self, font: &[u8], text: &str, em_size: f32) -> Result<(IDWriteFontFace, Vec<u16>, Vec<f32>, Vec<DWRITE_GLYPH_OFFSET>)> {
        use std::hash::{Hash, Hasher};
        let factory = self.text_factory()?.clone();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        font.hash(&mut h);
        let path = std::env::temp_dir().join(format!("d2d_cb_{:016x}.ttf", h.finish()));
        std::fs::write(&path, font)?;
        let file = factory.CreateFontFileReference(&HSTRING::from(path.to_string_lossy().as_ref()), None)?;
        let face = factory.CreateFontFace(DWRITE_FONT_FACE_TYPE_TRUETYPE, &[Some(file)], 0, DWRITE_FONT_SIMULATIONS_NONE)?;
        let mut fm = DWRITE_FONT_METRICS::default();
        face.GetMetrics(&mut fm);
        let upem = fm.designUnitsPerEm as f32;
        let codes: Vec<u32> = text.chars().map(|c| c as u32).collect();
        let mut ids = vec![0u16; codes.len()];
        face.GetGlyphIndices(codes.as_ptr(), codes.len() as u32, ids.as_mut_ptr())?;
        let mut gm = vec![DWRITE_GLYPH_METRICS::default(); codes.len()];
        face.GetDesignGlyphMetrics(ids.as_ptr(), codes.len() as u32, gm.as_mut_ptr(), false)?;
        let advances: Vec<f32> = gm.iter().map(|m| m.advanceWidth as f32 * em_size / upem).collect();
        let offsets = vec![DWRITE_GLYPH_OFFSET { advanceOffset: 0.0, ascenderOffset: 0.0 }; codes.len()];
        let _ = std::fs::remove_file(&path);
        Ok((face, ids, advances, offsets))
    }
}

/// Run `scene`'s ops on `dc` (BeginDraw…EndDraw), discarding the record — for probes that drive the
/// oracle runner on their own device.
#[allow(dead_code)]
pub(crate) unsafe fn execute_scene(scene: &Scene, dc: &ID2D1DeviceContext, factory: &ID2D1Factory) -> Result<()> {
    execute(scene, dc, factory).map(|_| ())
}

unsafe fn execute(scene: &Scene, dc: &ID2D1DeviceContext, factory: &ID2D1Factory) -> Result<Recorded> {
    let mut runner = Runner::new(dc, factory)?;
    // The device-context default intermediate precision (SetRenderingControls.bufferPrecision) caps the
    // effect graph's intermediates — a custom effect needs this at 32FLOAT to keep a float intermediate.
    if let Some(bp) = scene.buffer_precision {
        let mut rc = dc.GetRenderingControls();
        rc.bufferPrecision = D2D1_BUFFER_PRECISION(bp as i32);
        dc.SetRenderingControls(&rc);
    }
    #[cfg(feature = "native-detour")]
    let rlog = render_log_begin(dc)?;
    dc.BeginDraw();
    dc.Clear(Some(&color([0.; 4])));
    let mut ops = Vec::new();
    for (i, op) in scene.ops.iter().enumerate() {
        dc.SetTags(i as u64, 0);
        let result = runner.op(i, op);
        let (mut tag1, mut tag2) = (0u64, 0u64);
        dc.GetTags(Some(&mut tag1), Some(&mut tag2));
        // Record the HRESULT and tags, then continue: a negative-test op captures its
        // error instead of aborting. Valid v1 scenes never error, so this is a no-op for
        // them and output pixels are unchanged. "S_OK" = success, "0x…" = a D2D HRESULT,
        // "err: …" = a harness-level failure (e.g. an undefined resource id).
        let hr = match &result {
            Ok(()) => "S_OK".to_string(),
            Err(e) => hr_of(e)
                .map(fmt_hr)
                .unwrap_or_else(|| format!("err: {e:#}").chars().take(160).collect()),
        };
        ops.push(serde_json::json!({ "op": i, "hr": hr, "tag1": tag1, "tag2": tag2 }));
    }
    let (mut tag1, mut tag2) = (0u64, 0u64);
    let end = dc.EndDraw(Some(&mut tag1), Some(&mut tag2));
    let end_hr = end.as_ref().err().map(|e| e.code().0).unwrap_or(0);
    #[cfg(feature = "native-detour")]
    render_log_end(rlog)?;
    Ok(Recorded {
        ops,
        queries: runner.queries,
        end_draw: serde_json::json!({"hr": fmt_hr(end_hr), "tag1": tag1, "tag2": tag2}),
        ok: end.is_ok(),
    })
}

#[cfg(feature = "native-detour")]
include!("render_detour.rs");



// Map the scene's optional DXGI_FORMAT / D2D1_ALPHA_MODE numbers to the pixel format, alpha mode and
// bytes-per-pixel. Absent ⇒ premultiplied B8G8R8A8_UNORM (the v1 default), so untagged scenes are
// byte-identical. The HDR/high-precision additions are the two float formats and R8G8B8A8.
fn pixel_fmt(format: Option<u32>, alpha: Option<u32>) -> (DXGI_FORMAT, D2D1_ALPHA_MODE, usize) {
    let f = format.unwrap_or(DXGI_FORMAT_B8G8R8A8_UNORM.0 as u32);
    let bpp = match DXGI_FORMAT(f as i32) {
        DXGI_FORMAT_R32G32B32A32_FLOAT => 16,
        DXGI_FORMAT_R16G16B16A16_FLOAT | DXGI_FORMAT_R16G16B16A16_UNORM => 8,
        DXGI_FORMAT_A8_UNORM => 1,
        _ => 4, // B8G8R8A8_UNORM / R8G8B8A8_UNORM
    };
    let a = alpha.unwrap_or(D2D1_ALPHA_MODE_PREMULTIPLIED.0 as u32);
    (DXGI_FORMAT(f as i32), D2D1_ALPHA_MODE(a as i32), bpp)
}

fn bitmap_properties_fmt(
    dpi: f32,
    options: D2D1_BITMAP_OPTIONS,
    format: DXGI_FORMAT,
    alpha: D2D1_ALPHA_MODE,
) -> D2D1_BITMAP_PROPERTIES1 {
    D2D1_BITMAP_PROPERTIES1 {
        pixelFormat: D2D1_PIXEL_FORMAT { format, alphaMode: alpha },
        dpiX: dpi,
        dpiY: dpi,
        bitmapOptions: options,
        ..Default::default()
    }
}

// Decode one raw target pixel to straight RGBA8 for the preview PNG (cosmetic; the `.bgra` sidecar is
// the oracle). Premultiplied float/UNORM targets are unpremultiplied and clamped to [0,1].
fn preview(raw: &[u8], size: [u32; 2], format: DXGI_FORMAT, alpha: D2D1_ALPHA_MODE) -> Vec<u8> {
    if format == DXGI_FORMAT_B8G8R8A8_UNORM {
        return straight_rgba(raw); // unchanged v1 path
    }
    let premul = alpha == D2D1_ALPHA_MODE_PREMULTIPLIED;
    let bpp = match format {
        DXGI_FORMAT_R32G32B32A32_FLOAT => 16,
        DXGI_FORMAT_R16G16B16A16_FLOAT | DXGI_FORMAT_R16G16B16A16_UNORM => 8,
        DXGI_FORMAT_A8_UNORM => 1,
        _ => 4,
    };
    let chan = |px: &[u8], c: usize| -> f32 {
        match format {
            DXGI_FORMAT_R32G32B32A32_FLOAT => f32::from_le_bytes(px[c * 4..c * 4 + 4].try_into().unwrap()),
            DXGI_FORMAT_R16G16B16A16_FLOAT => f16_to_f32(u16::from_le_bytes(px[c * 2..c * 2 + 2].try_into().unwrap())),
            _ => px[c] as f32 / 255.0, // R8G8B8A8_UNORM
        }
    };
    let mut out = Vec::with_capacity(size[0] as usize * size[1] as usize * 4);
    for px in raw.chunks_exact(bpp) {
        let a = chan(px, 3);
        let s = |c: usize| {
            let v = chan(px, c);
            let v = if premul && a > 0.0 { v / a } else { v };
            (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
        };
        out.extend_from_slice(&[s(0), s(1), s(2), (a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8]);
    }
    out
}

// IEEE half → f32 (preview only; parity rounding lives in the CPU renderer).
fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let bits = if exp == 0 {
        if mant == 0 { (sign as u32) << 31 } else {
            // subnormal
            let mut e = -1i32;
            let mut m = mant as u32;
            loop { e += 1; m <<= 1; if m & 0x400 != 0 { break; } }
            ((sign as u32) << 31) | (((127 - 15 - e) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        ((sign as u32) << 31) | (0xff << 23) | ((mant as u32) << 13)
    } else {
        ((sign as u32) << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | ((mant as u32) << 13)
    };
    f32::from_bits(bits)
}

unsafe fn modules() -> serde_json::Value {
    let mut result = serde_json::Map::new();
    for name in [
        "d2d1.dll",
        "d3d11.dll",
        "d3d10warp.dll",
        "dxgi.dll",
        "dwrite.dll",
        "d3dcompiler_47.dll",
        "windowscodecs.dll",
    ] {
        if let Ok(module) = GetModuleHandleW(&HSTRING::from(name)) {
            let mut buf = [0u16; 32768];
            let len = GetModuleFileNameW(module, &mut buf) as usize;
            result.insert(name.into(), String::from_utf16_lossy(&buf[..len]).into());
        }
    }
    result.into()
}





















pub fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        (3..=4).contains(&args.len()),
        "usage: d2d-ref SCENE.json OUTPUT.png [--wine-smoke|--wine-native]"
    );
    let mode = args.get(3).map(String::as_str).unwrap_or("windows-warp");
    ensure!(
        ["windows-warp", "--wine-smoke", "--wine-native"].contains(&mode),
        "unknown mode {mode}"
    );
    let scene = read_scene(Path::new(&args[1]))?;
    unsafe {
        let wine =
            GetProcAddress(GetModuleHandleW(w!("ntdll.dll"))?, s!("wine_get_version")).is_some();
        ensure!(
            !wine || mode != "windows-warp",
            "Wine detected; select --wine-smoke or --wine-native explicitly"
        );
        CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;
        let result = capture(&scene, Path::new(&args[2]), mode, wine);
        CoUninitialize();
        result
    }
}

unsafe fn capture(scene: &Scene, output: &Path, mode: &str, wine: bool) -> Result<()> {
    #[cfg(feature = "native-detour")]
    let want_draws = std::env::var_os("D2D_DUMP_DRAWS").is_some();
    #[cfg(feature = "native-detour")]
    if want_draws { draws_dump::install(); }
    let mut d3d = None;
    let mut level = D3D_FEATURE_LEVEL::default();
    D3D11CreateDevice(
        None,
        D3D_DRIVER_TYPE_WARP,
        HMODULE::default(),
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        Some(&[D3D_FEATURE_LEVEL_11_0]),
        D3D11_SDK_VERSION,
        Some(&mut d3d),
        Some(&mut level),
        None,
    )
    .context("D3D11CreateDevice(WARP, FL11_0)")?;
    let d3d_dev = d3d.context("missing D3D11 device")?;
    // Optional instrumentation hook (native-detour feature, off by default): applied to the
    // freshly-created D3D11 device before any rendering.
    #[cfg(feature = "native-detour")]
    if let Some(h) = DEVICE_HOOK.get() {
        h(&d3d_dev);
    }
    // Analysis-only: dump every built-in effect's DXBC as D2D hands it to D3D (public
    // the public shader-creation path — shader creation goes through it
    // even though draws don't). Render a scene with exactly ONE built-in effect to attribute cleanly.
    #[cfg(feature = "native-detour")]
    if std::env::var_os("D2D_DUMP_SHADERS").is_some() {
        shader_dump::install(d3d_dev.as_raw());
    }
    #[cfg(feature = "native-detour")]
    let want_vbib = std::env::var_os("D2D_DUMP_VBIB").is_some();
    #[cfg(feature = "native-detour")]
    if want_vbib {
        vbib_dump::install(d3d_dev.as_raw());
    }
    #[cfg(feature = "native-detour")]
    if std::env::var_os("D2D_DUMP_TEXTURES").is_some() {
        texture_dump::install(d3d_dev.as_raw());
    }
    let dxgi: IDXGIDevice = d3d_dev.cast()?;
    let adapter: IDXGIAdapter1 = dxgi.GetAdapter()?.cast()?;
    let desc = adapter.GetDesc1()?;
    let name = String::from_utf16_lossy(&desc.Description)
        .trim_end_matches('\0')
        .to_owned();
    eprintln!("mode={mode} wine={wine} adapter={name:?} vendor={:04x} device={:04x} flags={} feature_level={:x}", desc.VendorId, desc.DeviceId, desc.Flags, level.0);
    if mode != "--wine-smoke" {
        ensure!(
            desc.VendorId == 0x1414
                && desc.DeviceId == 0x8c
                && desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
            "adapter is not Microsoft WARP; refusing reference capture"
        );
        GetModuleHandleW(w!("d3d10warp.dll")).context("WARP DLL is not loaded")?;
    }
    let factory: ID2D1Factory1 = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, None)?;
    // The reference renderer is now loaded; install the optional instrumentation (off by default;
    // only built with the native-detour feature).
    #[cfg(feature = "native-detour")]
    ink_flat_hook::install();
    let dc = factory
        .CreateDevice(&dxgi)?
        .CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE)?;
    dc.SetDpi(scene.dpi, scene.dpi);
    let size = D2D_SIZE_U {
        width: scene.size[0],
        height: scene.size[1],
    };
    let (fmt, am, bpp) = pixel_fmt(scene.format, scene.alpha);
    let target = dc.CreateBitmap(
        size,
        None,
        0,
        &bitmap_properties_fmt(scene.dpi, D2D1_BITMAP_OPTIONS_TARGET, fmt, am),
    )?;
    dc.SetTarget(&target);
    #[cfg(feature = "native-detour")]
    if std::env::var_os("D2D_INK_TRACE").is_some() {
        ink_trace::install();
    }
    #[cfg(feature = "native-detour")]
    let want_contours = std::env::var_os("DWRITE_DUMP_CONTOURS").is_some();
    #[cfg(feature = "native-detour")]
    if want_contours {
        d2d_contour_hook::install();
    }
    // `DWRITE_DUMP_FILLBITMAP`: capture the scan-converter mono bitmaps D2D rasterises internally
    // (the COLR paint-tree leaf glyph run) — the leaf's post-dropout 1-bit grid at its ss render em.
    #[cfg(feature = "native-detour")]
    let want_fillbm = std::env::var_os("DWRITE_DUMP_FILLBITMAP").is_some();
    #[cfg(feature = "native-detour")]
    if want_fillbm {
        fillbitmap_hook::install();
    }
    #[cfg(feature = "native-detour")]
    let want_fillsubrect = std::env::var_os("DWRITE_DUMP_FILLSUBRECT").is_some();
    #[cfg(feature = "native-detour")]
    if want_fillsubrect {
        fillsubrect_hook::install();
    }
    #[cfg(feature = "native-detour")]
    let want_emgray = std::env::var_os("DWRITE_DUMP_EMBOLDENGRAY").is_some();
    #[cfg(feature = "native-detour")]
    if want_emgray {
        embolden_gray_hook::install();
    }
    // `DWRITE_DUMP_GRAYMAP`: capture the 6×6 grayscale-downsample OUTPUT during the real DrawText
    // render — the only path that reaches it (CreateGlyphRunAnalysis(ALIASED_1x1) skips it). Needs
    // DWRITE_COLD_FONT. If the array is empty for a glyph, that glyph's render did NOT take the 4× path.
    #[cfg(feature = "native-detour")]
    let want_graymap = std::env::var_os("DWRITE_DUMP_GRAYMAP").is_some();
    #[cfg(feature = "native-detour")]
    if want_graymap {
        graymap_hook::install();
    }
    // `DWRITE_DUMP_RASTER`: capture the glyph-cache producer the real cold DrawText uses
    // — the 1-bit mono bitmap + the rasterizer identity + the 2x2-downsample gate flags.
    #[cfg(feature = "native-detour")]
    let want_scale = std::env::var_os("DWRITE_DUMP_SCALE").is_some();
    #[cfg(feature = "native-detour")]
    if want_scale { scale_hook::install(); }
    #[cfg(feature = "native-detour")]
    let want_raster = std::env::var_os("DWRITE_DUMP_RASTER").is_some();
    #[cfg(feature = "native-detour")]
    if want_raster {
        rasterize_hook::install();
    }
    // `DWRITE_DUMP_DS2X2`: capture per-glyph DownSample2x2 inputs (own-frame left/top + phase a2) and
    // the downsampled output, for the NATURAL_SYMMETRIC_DOWNSAMPLED coverage port. Needs DWRITE_COLD_FONT.
    #[cfg(feature = "native-detour")]
    let want_ds2x2 = std::env::var_os("DWRITE_DUMP_DS2X2").is_some();
    #[cfg(feature = "native-detour")]
    if want_ds2x2 {
        downsample_hook::install();
    }
    // `DWRITE_DUMP_CROSS`: capture per-crossing scan marks during the REAL DrawText render (not
    // CreateGlyphRunAnalysis) — the true DrawText-path crossings. Needs DWRITE_COLD_FONT so the scan runs.
    #[cfg(feature = "native-detour")]
    let want_cross = std::env::var_os("DWRITE_DUMP_CROSS").is_some();
    #[cfg(feature = "native-detour")]
    if want_cross {
        crossing_hook::install();
    }
    // `D2D_DUMP_COLRBITMAP`: capture the realized supersampled paint-tree bitmap(s) the COLR place-back
    // minifies (native's actual ss leaf buffer), dumped after EndDraw.
    #[cfg(feature = "native-detour")]
    let want_colrbm = std::env::var_os("D2D_DUMP_COLRBITMAP").is_some();
    #[cfg(feature = "native-detour")]
    if want_colrbm {
        colr_bitmap_hook::install();
    }
    // `D2D_DUMP_LEAFGLYPH`: capture the paint-tree sync transform + DrawGlyphRun for each
    // paint-tree leaf (the folded matrix + glyph run + measuring) — to replay exactly via public API.
    #[cfg(feature = "native-detour")]
    let want_leaf = std::env::var_os("D2D_DUMP_LEAFGLYPH").is_some();
    #[cfg(feature = "native-detour")]
    if want_leaf {
        leaf_capture_hook::install();
    }
    // `D2D_DUMP_TEXTLUT`: capture the text-blend coverage→BGRA LUT (mode + color params + entries).
    #[cfg(feature = "native-detour")]
    let want_lut = std::env::var_os("D2D_DUMP_TEXTLUT").is_some();
    #[cfg(feature = "native-detour")]
    if want_lut {
        leaf_lut_hook::install();
    }
    // `D2D_DUMP_GRADSPAN`: capture the CPU gradient-span generator (width + the realized
    // packed-BGRA ramp row + stops) — native's exact CPU gradient ramp, for the conical/sweep compare.
    #[cfg(feature = "native-detour")]
    let want_gradspan = std::env::var_os("D2D_DUMP_GRADSPAN").is_some();
    #[cfg(feature = "native-detour")]
    if want_gradspan {
        gradient_span_hook::install();
    }
    // `D2D_DUMP_CALCTEX`: capture the texture-size calculation (brush type, footprint, chosen width) — to compare our
    // conical/sweep footprint to native's (the width formula already matches our radial_width).
    #[cfg(feature = "native-detour")]
    let want_calctex = std::env::var_os("D2D_DUMP_CALCTEX").is_some();
    #[cfg(feature = "native-detour")]
    if want_calctex {
        gradient_calctex_hook::install();
    }
    #[cfg(feature = "native-detour")]
    let want_brushctx = std::env::var_os("D2D_DUMP_BRUSHCTX").is_some();
    #[cfg(feature = "native-detour")]
    if want_brushctx {
        brush_ctx_hook::install();
    }
    // `D2D_DUMP_REALIZEINFO`: capture the realization-info / texture-mapping query (chosen width + brush matrix) —
    // native's sampled width for LINEAR leaves (which skip that calculation).
    #[cfg(feature = "native-detour")]
    let want_realizeinfo = std::env::var_os("D2D_DUMP_REALIZEINFO").is_some();
    #[cfg(feature = "native-detour")]
    if want_realizeinfo {
        realizeinfo_hook::install();
    }
    #[cfg(feature = "native-detour")]
    let want_ctm = std::env::var_os("D2D_DUMP_CTM").is_some();
    #[cfg(feature = "native-detour")]
    if want_ctm {
        calctexmap_hook::install();
    }
    #[cfg(feature = "native-detour")]
    let want_linplace = std::env::var_os("D2D_DUMP_LINPLACE").is_some();
    #[cfg(feature = "native-detour")]
    if want_linplace {
        linplace_hook::install();
        linctm_hook::install();
    }
    // `D2D_DUMP_PTGLYPH`: per-leaf GetPaintTreeGlyphImage em/mode/origin/matrix/out-transform.
    #[cfg(feature = "native-detour")]
    let want_ptglyph = std::env::var_os("D2D_DUMP_PTGLYPH").is_some();
    #[cfg(feature = "native-detour")]
    if want_ptglyph {
        ptglyph_hook::install();
    }
    #[cfg(feature = "native-detour")]
    if want_ptglyph || want_leaf {
        sync_hook::install();
    }
    #[cfg(feature = "native-detour")]
    let probe_brush: Option<ID2D1Brush> = if want_leaf && std::env::var_os("D2D_LEAF_SOLID").is_some() {
        Some(dc.CreateSolidColorBrush(&color([0.0, 0.0, 0.0, 1.0]), None)?.cast()?)
    } else { None };
    #[cfg(feature = "native-detour")]
    if let Some(b) = &probe_brush { leaf_capture_hook::set_brush(b.as_raw() as usize); }
    #[cfg(feature = "native-detour")]
    let jit_pre = std::env::var_os("D2D_DUMP_JIT").map(|_| jit_regions::bases());
    let recorded = execute(scene, &dc, &factory.cast()?)?;
    #[cfg(feature = "native-detour")]
    if want_draws { std::fs::write(output.with_extension("draws.json"), serde_json::to_vec(&draws_dump::take())?)?; }
    #[cfg(feature = "native-detour")]
    if want_vbib { std::fs::write(output.with_extension("vbib.json"), serde_json::to_vec(&vbib_dump::take())?)?; }
    #[cfg(feature = "native-detour")]
    leaf_capture_hook::set_brush(0);
    #[cfg(feature = "native-detour")]
    if want_scale {
        scale_hook::remove();
        std::fs::write(output.with_extension("scale.json"), serde_json::to_vec(&scale_hook::take())?)?;
    }
    #[cfg(feature = "native-detour")]
    if want_ptglyph {
        ptglyph_hook::remove();
        std::fs::write(output.with_extension("ptglyph.json"), serde_json::to_vec(&ptglyph_hook::take())?)?;
    }
    #[cfg(feature = "native-detour")]
    if want_ptglyph || want_leaf {
        sync_hook::remove();
        std::fs::write(output.with_extension("syncxf.json"), serde_json::to_vec(&sync_hook::take())?)?;
    }
    #[cfg(feature = "native-detour")]
    if want_brushctx {
        brush_ctx_hook::remove();
        std::fs::write(output.with_extension("brushctx.json"), serde_json::to_vec(&brush_ctx_hook::take())?)?;
    }
    #[cfg(feature = "native-detour")]
    if want_realizeinfo {
        realizeinfo_hook::remove();
        std::fs::write(output.with_extension("realizeinfo.json"), serde_json::to_vec(&realizeinfo_hook::take())?)?;
    }
    #[cfg(feature = "native-detour")]
    if want_ctm {
        calctexmap_hook::remove();
        std::fs::write(output.with_extension("ctm.json"), serde_json::to_vec(&calctexmap_hook::take())?)?;
    }
    #[cfg(feature = "native-detour")]
    if want_linplace {
        linplace_hook::remove();
        std::fs::write(output.with_extension("linplace.json"), serde_json::to_vec(&linplace_hook::take())?)?;
        linctm_hook::remove();
        std::fs::write(output.with_extension("linctm.json"), serde_json::to_vec(&linctm_hook::take())?)?;
    }
    #[cfg(feature = "native-detour")]
    if want_gradspan {
        gradient_span_hook::remove();
        std::fs::write(output.with_extension("gradspan.json"), serde_json::to_vec(&gradient_span_hook::take())?)?;
    }
    #[cfg(feature = "native-detour")]
    if want_calctex {
        gradient_calctex_hook::remove();
        std::fs::write(output.with_extension("calctex.json"), serde_json::to_vec(&gradient_calctex_hook::take())?)?;
    }
    #[cfg(feature = "native-detour")]
    if want_cross {
        crossing_hook::remove();
        std::fs::write(output.with_extension("cross.json"), serde_json::to_vec(&crossing_hook::take())?)?;
    }
    #[cfg(feature = "native-detour")]
    if want_graymap {
        graymap_hook::remove();
        std::fs::write(output.with_extension("graymap.json"), serde_json::to_vec(&graymap_hook::take())?)?;
    }
    #[cfg(feature = "native-detour")]
    if want_raster {
        rasterize_hook::remove();
        std::fs::write(output.with_extension("raster.json"), serde_json::to_vec(&rasterize_hook::take())?)?;
    }
    #[cfg(feature = "native-detour")]
    if want_ds2x2 {
        downsample_hook::remove();
        std::fs::write(output.with_extension("ds2x2.json"), serde_json::to_vec(&downsample_hook::take())?)?;
    }
    #[cfg(feature = "native-detour")]
    if want_leaf {
        leaf_capture_hook::remove();
        std::fs::write(output.with_extension("leaf.json"), serde_json::to_vec(&leaf_capture_hook::take())?)?;
    }
    #[cfg(feature = "native-detour")]
    if want_lut {
        leaf_lut_hook::remove();
        std::fs::write(output.with_extension("textlut.json"), serde_json::to_vec(&leaf_lut_hook::take())?)?;
    }
    #[cfg(feature = "native-detour")]
    if want_fillbm {
        fillbitmap_hook::remove();
        std::fs::write(output.with_extension("fillbm.json"), serde_json::to_vec(&fillbitmap_hook::take())?)?;
    }
    #[cfg(feature = "native-detour")]
    if want_fillsubrect {
        fillsubrect_hook::remove();
        std::fs::write(output.with_extension("fillsubrect.json"), serde_json::to_vec(&fillsubrect_hook::take())?)?;
    }
    #[cfg(feature = "native-detour")]
    if want_emgray {
        embolden_gray_hook::remove();
        std::fs::write(output.with_extension("emgray.json"), serde_json::to_vec(&embolden_gray_hook::take())?)?;
    }
    #[cfg(feature = "native-detour")]
    if want_colrbm {
        colr_bitmap_hook::remove();
    }
    #[cfg(feature = "native-detour")]
    if want_contours {
        d2d_contour_hook::remove();
        std::fs::write(output.with_extension("d2dcontours.json"), serde_json::to_vec(&d2d_contour_hook::take())?)?;
    }
    dc.SetTarget(None);
    let metadata = serde_json::json!({
        "mode": mode, "wine": wine, "reference_candidate": mode == "windows-warp" && !wine,
        "size": scene.size, "dpi": scene.dpi, "adapter": name, "vendor": desc.VendorId, "device": desc.DeviceId,
        "adapter_flags": desc.Flags, "feature_level": level.0, "modules": modules(),
        "format": fmt.0, "alpha_mode": am.0, "bytes_per_pixel": bpp,
        "raw": "top-down tightly packed target pixels (default premultiplied BGRA8), no header", "png": "straight RGBA8 preview",
        "text_rendering": { "gamma": 2.2, "contrast": 1.0, "cleartype_level": 1.0, "pixel_geometry": "flat", "rendering_mode": "default" },
        "ops": recorded.ops, "queries": recorded.queries, "end_draw": recorded.end_draw
    });
    if recorded.ok {
        let staging = dc.CreateBitmap(
            size,
            None,
            0,
            &bitmap_properties_fmt(
                scene.dpi,
                D2D1_BITMAP_OPTIONS_CPU_READ | D2D1_BITMAP_OPTIONS_CANNOT_DRAW,
                fmt,
                am,
            ),
        )?;
        staging.CopyFromBitmap(None, &target, None)?;
        let row_bytes = size.width as usize * bpp;
        let mut raw = Vec::with_capacity(row_bytes * size.height as usize);
        let mapped = staging.Map(D2D1_MAP_OPTIONS_READ)?;
        for row in 0..size.height as usize {
            raw.extend_from_slice(std::slice::from_raw_parts(
                mapped.bits.add(row * mapped.pitch as usize),
                row_bytes,
            ));
        }
        staging.Unmap()?;
        write_png(output, scene.size, &preview(&raw, scene.size, fmt, am))?;
        std::fs::write(output.with_extension("bgra"), &raw)?;
    }
    // Read the target first so D2D's deferred rendering has completed before copying its atlases.
    #[cfg(feature = "native-detour")]
    if let Ok(dir) = std::env::var("D2D_DUMP_TEXTURES") {
        texture_dump::dump(&d3d_dev, &dir);
    }
    #[cfg(feature = "native-detour")]
    if want_colrbm {
        // Each captured COLR paint-tree image is a realized ss bitmap; copy to a CPU-readable staging
        // bitmap (matching its pixel size/B8G8R8A8-premul) and dump its texels + size as JSON.
        let mut dumps = Vec::new();
        for (n, raw_img) in colr_bitmap_hook::take().into_iter().enumerate() {
            let image = ID2D1Image::from_raw(raw_img as *mut _); // takes the detour's AddRef
            if let Ok(bmp) = image.cast::<ID2D1Bitmap>() {
                let ps = bmp.GetPixelSize();
                let st = dc.CreateBitmap(
                    ps, None, 0,
                    &bitmap_properties_fmt(96.0, D2D1_BITMAP_OPTIONS_CPU_READ | D2D1_BITMAP_OPTIONS_CANNOT_DRAW, DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED),
                )?;
                if st.CopyFromBitmap(None, &bmp, None).is_ok() {
                    let rb = ps.width as usize * 4;
                    let mut raw = Vec::with_capacity(rb * ps.height as usize);
                    let m = st.Map(D2D1_MAP_OPTIONS_READ)?;
                    for row in 0..ps.height as usize {
                        raw.extend_from_slice(std::slice::from_raw_parts(m.bits.add(row * m.pitch as usize), rb));
                    }
                    st.Unmap()?;
                    std::fs::write(output.with_extension(format!("colrbm{n}.bgra")), &raw)?;
                    dumps.push(serde_json::json!({"n": n, "w": ps.width, "h": ps.height}));
                }
            }
        }
        std::fs::write(output.with_extension("colrbm.json"), serde_json::to_vec(&dumps)?)?;
    }
    std::fs::write(
        output.with_extension("json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    #[cfg(feature = "native-detour")]
    if let Some(pre) = jit_pre { jit_regions::dump(&pre, &output.with_extension("jit").to_string_lossy())?; }
    ensure!(
        recorded.ok,
        "scene failed to render: EndDraw {}",
        recorded.end_draw["hr"]
    );
    eprintln!("wrote {} and BGRA/provenance sidecars", output.display());
    Ok(())
}

// --- present-path capture (hwnd / DC / DXGI surface / swap chain) -------------------------------
//
// Proves the drop-in d2d1.dll presents into a real window, a GDI DC and a DXGI surface. The CPU
// core still renders every pixel; only the final EndDraw copy differs per target. The readback is
// identical code for the native oracle and our DLL, so (for opaque scenes, where our CPU pixels are
// byte-exact to WARP) the readback bytes match. Modes: dc, hwnd, surf, swap, gdi.
use windows::Win32::Graphics::Gdi::{
    BitBlt, CreateCompatibleDC, CreateDIBSection, DeleteDC, DeleteObject, GdiFlush, GetDC as GdiGetDC,
    ReleaseDC as GdiReleaseDC, SelectObject, BITMAPINFO, BITMAPINFOHEADER, BI_RGB, DIB_RGB_COLORS, HDC,
    HGDIOBJ, SRCCOPY,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, RegisterClassW, ShowWindow, HMENU, SW_SHOWNA,
    WNDCLASSW, WS_POPUP,
};
use windows::Win32::Foundation::{HINSTANCE, HWND, LPARAM, LRESULT, RECT, WPARAM};

unsafe extern "system" fn present_wndproc(h: HWND, m: u32, w: WPARAM, l: LPARAM) -> LRESULT {
    DefWindowProcW(h, m, w, l)
}

pub fn present_run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        (4..=5).contains(&args.len()),
        "usage: present-ref SCENE.json OUT_PREFIX MODE[dc|hwnd|surf|read|swap|gdi|wic] [--wine-smoke]"
    );
    let scene = read_scene(Path::new(&args[1]))?;
    let prefix = args[2].clone();
    let mode = args[3].clone();
    unsafe {
        CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;
        let r = present_capture(&scene, &prefix, &mode);
        CoUninitialize();
        r
    }
}

unsafe fn present_capture(scene: &Scene, prefix: &str, mode: &str) -> Result<()> {
    let (w, h) = (scene.size[0], scene.size[1]);
    let factory: ID2D1Factory1 = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, None)?;
    let f: ID2D1Factory = factory.cast()?;
    let raw = match mode {
        "dc" => capture_dc(&factory, &f, scene, w, h)?,
        "hwnd" => capture_hwnd(&factory, &f, scene, w, h)?,
        "gdi" => capture_gdi(&factory, &f, scene, w, h)?,
        "wic" => capture_wic(&factory, &f, scene, w, h)?,
        "surf" => capture_dxgi_surface(&factory, &f, scene, w, h)?,
        "read" => capture_read(&factory, &f, scene, w, h)?,
        "swap" => match capture_swapchain(&factory, &f, scene, w, h) {
            Ok(raw) => raw,
            // A real swap chain needs a GPU/compositor; where it can't be made (e.g. headless Wine)
            // skip rather than fail — the surf mode already proves the DXGI present path.
            Err(e) => {
                eprintln!("swap mode skipped: {e:#}");
                return Ok(());
            }
        },
        other => anyhow::bail!("unknown mode {other}"),
    };
    std::fs::write(format!("{prefix}.bgra"), &raw)?;
    eprintln!("wrote {prefix}.bgra ({mode}, {}x{})", w, h);
    Ok(())
}

fn bmi_topdown(w: u32, h: u32) -> BITMAPINFO {
    BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: w as i32,
            biHeight: -(h as i32),
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB.0,
            ..Default::default()
        },
        ..Default::default()
    }
}

// A top-down 32bpp DIB selected into a fresh memory DC; `bits` is the pixel memory.
unsafe fn dib_dc(w: u32, h: u32) -> Result<(HDC, windows::Win32::Graphics::Gdi::HBITMAP, windows::Win32::Graphics::Gdi::HGDIOBJ, *mut u8)> {
    let bmi = bmi_topdown(w, h);
    let mut bits: *mut core::ffi::c_void = std::ptr::null_mut();
    let bmp = CreateDIBSection(None, &bmi, DIB_RGB_COLORS, &mut bits, None, 0)?;
    let dc = CreateCompatibleDC(None);
    let old = SelectObject(dc, HGDIOBJ(bmp.0));
    Ok((dc, bmp, old, bits.cast()))
}

unsafe fn read_dib(bits: *const u8, w: u32, h: u32) -> Vec<u8> {
    std::slice::from_raw_parts(bits, (w * h * 4) as usize).to_vec()
}

unsafe fn free_dib(memdc: HDC, bmp: windows::Win32::Graphics::Gdi::HBITMAP, old: HGDIOBJ) {
    SelectObject(memdc, old);
    let _ = DeleteDC(memdc);
    let _ = DeleteObject(HGDIOBJ(bmp.0));
}

fn rt_props(dpi: f32) -> D2D1_RENDER_TARGET_PROPERTIES {
    D2D1_RENDER_TARGET_PROPERTIES {
        r#type: D2D1_RENDER_TARGET_TYPE_DEFAULT,
        pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT_B8G8R8A8_UNORM, alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED },
        dpiX: dpi,
        dpiY: dpi,
        usage: D2D1_RENDER_TARGET_USAGE_NONE,
        minLevel: D2D1_FEATURE_LEVEL_DEFAULT,
    }
}

unsafe fn capture_dc(factory: &ID2D1Factory1, f: &ID2D1Factory, scene: &Scene, w: u32, h: u32) -> Result<Vec<u8>> {
    let (memdc, bmp, old, bits) = dib_dc(w, h)?;
    let rt = factory.CreateDCRenderTarget(&rt_props(scene.dpi))?;
    rt.BindDC(memdc, &RECT { left: 0, top: 0, right: w as i32, bottom: h as i32 })?;
    rt.SetDpi(scene.dpi, scene.dpi);
    let dc: ID2D1DeviceContext = rt.cast()?;
    execute(scene, &dc, f)?;
    GdiFlush().ok()?;
    let raw = read_dib(bits, w, h);
    free_dib(memdc, bmp, old);
    Ok(raw)
}

unsafe fn capture_gdi(factory: &ID2D1Factory1, f: &ID2D1Factory, scene: &Scene, w: u32, h: u32) -> Result<Vec<u8>> {
    // Same as `dc`, but wrap the draw in a GdiInterop GetDC(COPY)/ReleaseDC round-trip (no GDI
    // drawing between them): proves the interop DC mirrors the target and composites back unchanged.
    let (memdc, bmp, old, bits) = dib_dc(w, h)?;
    let rt = factory.CreateDCRenderTarget(&rt_props(scene.dpi))?;
    rt.BindDC(memdc, &RECT { left: 0, top: 0, right: w as i32, bottom: h as i32 })?;
    rt.SetDpi(scene.dpi, scene.dpi);
    let dc: ID2D1DeviceContext = rt.cast()?;
    let gdi: ID2D1GdiInteropRenderTarget = rt.cast()?;
    dc.BeginDraw();
    dc.Clear(Some(&color([0.; 4])));
    let mut runner = Runner::new(&dc, f)?;
    for (i, op) in scene.ops.iter().enumerate() {
        let _ = runner.op(i, op);
    }
    let hdc = gdi.GetDC(D2D1_DC_INITIALIZE_MODE_COPY)?;
    let _ = hdc; // no GDI drawing: the COPY→ReleaseDC round-trip must be identity
    gdi.ReleaseDC(None)?;
    dc.EndDraw(None, None)?;
    GdiFlush().ok()?;
    let raw = read_dib(bits, w, h);
    free_dib(memdc, bmp, old);
    Ok(raw)
}

// `ID2D1Factory::CreateWicBitmapRenderTarget`: render the scene into a 32bppPBGRA IWICBitmap, then
// read its pixels back. A premultiplied-BGRA WIC bitmap stores the same bytes as the DC/surf oracle,
// so the read-back is byte-compared against the shared `<scene>.present.bgra`.
unsafe fn capture_wic(factory: &ID2D1Factory1, f: &ID2D1Factory, scene: &Scene, w: u32, h: u32) -> Result<Vec<u8>> {
    let wicf: IWICImagingFactory = CoCreateInstance(&CLSID_WICImagingFactory, None, CLSCTX_INPROC_SERVER)?;
    let bmp = wicf.CreateBitmap(w, h, &GUID_WICPixelFormat32bppPBGRA, WICBitmapCacheOnLoad)?;
    let rt = factory.CreateWicBitmapRenderTarget(&bmp, &rt_props(scene.dpi))?;
    rt.SetDpi(scene.dpi, scene.dpi);
    let dc: ID2D1DeviceContext = rt.cast()?;
    execute(scene, &dc, f)?;
    let rect = WICRect { X: 0, Y: 0, Width: w as i32, Height: h as i32 };
    let lock = bmp.Lock(&rect, WICBitmapLockRead.0 as u32)?;
    let stride = lock.GetStride()? as usize;
    let (mut size, mut data) = (0u32, std::ptr::null_mut());
    lock.GetDataPointer(&mut size, &mut data)?;
    let row = w as usize * 4;
    let mut raw = vec![0u8; row * h as usize];
    for y in 0..h as usize {
        raw[y * row..y * row + row].copy_from_slice(std::slice::from_raw_parts(data.add(y * stride), row));
    }
    Ok(raw)
}

unsafe fn make_window(w: u32, h: u32) -> Result<HWND> {
    let hinst = GetModuleHandleW(None)?;
    let hinst = HINSTANCE(hinst.0);
    let class = w!("d2dpresent");
    let wc = WNDCLASSW {
        lpfnWndProc: Some(present_wndproc),
        hInstance: hinst,
        lpszClassName: class,
        ..Default::default()
    };
    RegisterClassW(&wc); // 0 if already registered this process — harmless
    let hwnd = CreateWindowExW(
        Default::default(),
        class,
        w!(""),
        WS_POPUP,
        -20000,
        -20000,
        w as i32,
        h as i32,
        HWND::default(),
        HMENU::default(),
        hinst,
        None,
    )?;
    let _ = ShowWindow(hwnd, SW_SHOWNA); // give it a backing surface, off-screen
    Ok(hwnd)
}

unsafe fn capture_hwnd(factory: &ID2D1Factory1, f: &ID2D1Factory, scene: &Scene, w: u32, h: u32) -> Result<Vec<u8>> {
    let hwnd = make_window(w, h)?;
    let props = D2D1_HWND_RENDER_TARGET_PROPERTIES {
        hwnd,
        pixelSize: D2D_SIZE_U { width: w, height: h },
        presentOptions: D2D1_PRESENT_OPTIONS_NONE,
    };
    let rt = factory.CreateHwndRenderTarget(&rt_props(scene.dpi), &props)?;
    rt.SetDpi(scene.dpi, scene.dpi);
    let dc: ID2D1DeviceContext = rt.cast()?;
    execute(scene, &dc, f)?;
    GdiFlush().ok()?;
    // Read the window client area back through its DC.
    let windc = GdiGetDC(hwnd);
    let (memdc, bmp, old, bits) = dib_dc(w, h)?;
    let _ = BitBlt(memdc, 0, 0, w as i32, h as i32, windc, 0, 0, SRCCOPY);
    let raw = read_dib(bits, w, h);
    free_dib(memdc, bmp, old);
    GdiReleaseDC(hwnd, windc);
    let _ = DestroyWindow(hwnd);
    Ok(raw)
}

unsafe fn make_d3d() -> Result<ID3D11Device> {
    let mut d3d = None;
    D3D11CreateDevice(
        None,
        D3D_DRIVER_TYPE_WARP,
        HMODULE::default(),
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        Some(&[D3D_FEATURE_LEVEL_11_0]),
        D3D11_SDK_VERSION,
        Some(&mut d3d),
        None,
        None,
    )
    .context("D3D11CreateDevice(WARP)")?;
    d3d.context("missing D3D11 device")
}

unsafe fn make_bgra_texture(d3d: &ID3D11Device, w: u32, h: u32, bind: u32) -> Result<ID3D11Texture2D> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: w,
        Height: h,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: bind,
        ..Default::default()
    };
    let mut tex = None;
    d3d.CreateTexture2D(&desc, None, Some(&mut tex))?;
    tex.context("CreateTexture2D returned null")
}

// Copy a BGRA texture to a CPU-readable staging texture and read its rows tightly packed.
unsafe fn read_texture(d3d: &ID3D11Device, tex: &ID3D11Texture2D, w: u32, h: u32) -> Result<Vec<u8>> {
    let ctx = d3d.GetImmediateContext()?;
    let desc = D3D11_TEXTURE2D_DESC {
        Width: w,
        Height: h,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Usage: D3D11_USAGE_STAGING,
        CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
        ..Default::default()
    };
    let mut staging = None;
    d3d.CreateTexture2D(&desc, None, Some(&mut staging))?;
    let staging = staging.context("staging CreateTexture2D null")?;
    ctx.CopyResource(&staging, tex);
    let mut m = D3D11_MAPPED_SUBRESOURCE::default();
    ctx.Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut m))?;
    let mut raw = Vec::with_capacity((w * h * 4) as usize);
    for row in 0..h as usize {
        raw.extend_from_slice(std::slice::from_raw_parts((m.pData as *const u8).add(row * m.RowPitch as usize), w as usize * 4));
    }
    ctx.Unmap(&staging, 0);
    Ok(raw)
}

unsafe fn capture_dxgi_surface(factory: &ID2D1Factory1, f: &ID2D1Factory, scene: &Scene, w: u32, h: u32) -> Result<Vec<u8>> {
    let d3d = make_d3d()?;
    let tex = make_bgra_texture(&d3d, w, h, (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32)?;
    let surface: IDXGISurface = tex.cast()?;
    let rt = factory.CreateDxgiSurfaceRenderTarget(&surface, &rt_props(scene.dpi))?;
    rt.SetDpi(scene.dpi, scene.dpi);
    let dc: ID2D1DeviceContext = rt.cast()?;
    execute(scene, &dc, f)?;
    read_texture(&d3d, &tex, w, h)
}

// ID2D1DeviceContext over a real DXGI swap chain's back buffer (the scenario the task names).
unsafe fn capture_swapchain(factory: &ID2D1Factory1, f: &ID2D1Factory, scene: &Scene, w: u32, h: u32) -> Result<Vec<u8>> {
    let d3d = make_d3d()?;
    let dxgi: IDXGIDevice = d3d.cast()?;
    let dxgi_factory: IDXGIFactory2 = dxgi.GetAdapter()?.GetParent()?;
    let hwnd = make_window(w, h)?;
    let desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: w,
        Height: h,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: 2,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
        AlphaMode: DXGI_ALPHA_MODE_IGNORE,
        ..Default::default()
    };
    let swapchain: IDXGISwapChain1 = dxgi_factory.CreateSwapChainForHwnd(&d3d, hwnd, &desc, None, None)?;
    let backbuf: ID3D11Texture2D = swapchain.GetBuffer(0)?;
    let surface: IDXGISurface = backbuf.cast()?;
    let device = factory.CreateDevice(&dxgi)?;
    let dc = device.CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE)?;
    dc.SetDpi(scene.dpi, scene.dpi);
    let bprops = D2D1_BITMAP_PROPERTIES1 {
        pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT_B8G8R8A8_UNORM, alphaMode: D2D1_ALPHA_MODE_IGNORE },
        dpiX: scene.dpi,
        dpiY: scene.dpi,
        bitmapOptions: D2D1_BITMAP_OPTIONS_TARGET | D2D1_BITMAP_OPTIONS_CANNOT_DRAW,
        ..Default::default()
    };
    let bitmap = dc.CreateBitmapFromDxgiSurface(&surface, Some(&bprops as *const _))?;
    dc.SetTarget(&bitmap);
    execute(scene, &dc, f)?;
    // Read the back buffer before Present (flip discards it).
    let raw = read_texture(&d3d, &backbuf, w, h)?;
    let _ = swapchain.Present(0, DXGI_PRESENT(0));
    let _ = DestroyWindow(hwnd);
    Ok(raw)
}

// Readable-DXGI round-trip (GetSurface / CreateBitmapFromDxgiSurface readback /
// CreateImageSourceFromDxgi): render the scene into surface A, then read A back through those three
// APIs and copy it 1:1 into surface B. The round-trip reproduces the scene, so the read-back is
// checked against the same `<scene>.present.bgra` oracle `surf` uses — no new capture needed.
unsafe fn capture_read(factory: &ID2D1Factory1, f: &ID2D1Factory, scene: &Scene, w: u32, h: u32) -> Result<Vec<u8>> {
    let d3d = make_d3d()?;
    let bind = (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32;
    // 1. render the scene into surface A.
    let texa = make_bgra_texture(&d3d, w, h, bind)?;
    let surfa: IDXGISurface = texa.cast()?;
    let rt = factory.CreateDxgiSurfaceRenderTarget(&surfa, &rt_props(scene.dpi))?;
    rt.SetDpi(scene.dpi, scene.dpi);
    execute(scene, &rt.cast::<ID2D1DeviceContext>()?, f)?;
    // 2. a device-context target over a second surface B.
    let dxgi: IDXGIDevice = d3d.cast()?;
    let device = factory.CreateDevice(&dxgi)?;
    let dc = device.CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE)?;
    dc.SetDpi(scene.dpi, scene.dpi);
    let texb = make_bgra_texture(&d3d, w, h, bind)?;
    let surfb: IDXGISurface = texb.cast()?;
    let bprops = |opts| D2D1_BITMAP_PROPERTIES1 {
        pixelFormat: D2D1_PIXEL_FORMAT { format: DXGI_FORMAT_B8G8R8A8_UNORM, alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED },
        dpiX: scene.dpi,
        dpiY: scene.dpi,
        bitmapOptions: opts,
        ..Default::default()
    };
    let target = dc.CreateBitmapFromDxgiSurface(&surfb, Some(&bprops(D2D1_BITMAP_OPTIONS_TARGET)))?;
    // 3. read surface A back: a source bitmap (via GetSurface) feeds an image source.
    let srcbmp = dc.CreateBitmapFromDxgiSurface(&surfa, Some(&bprops(D2D1_BITMAP_OPTIONS_NONE)))?;
    let via_getsurface = srcbmp.GetSurface()?; // must be A — proven by the round-trip pixels below
    let dc2: ID2D1DeviceContext2 = dc.cast()?;
    let imgsrc = dc2.CreateImageSourceFromDxgi(
        &[Some(via_getsurface)],
        DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,
        D2D1_IMAGE_SOURCE_FROM_DXGI_OPTIONS(0),
    )?;
    // 4. copy it 1:1 into B and read B.
    dc.SetTarget(&target);
    dc.BeginDraw();
    dc.DrawImage(&imgsrc, None, None, D2D1_INTERPOLATION_MODE_NEAREST_NEIGHBOR, D2D1_COMPOSITE_MODE_SOURCE_COPY);
    dc.EndDraw(None, None)?;
    read_texture(&d3d, &texb, w, h)
}







