//! Geometry objects and the path sink. One `GeometryObj` type implements all four concrete geometry
//! interfaces (rectangle/rounded/ellipse/path) over a `d2d::geometry::Geometry`; `FillGeometry`/
//! `DrawGeometry`/`Open` recover it with `cast_object_ref`.

use super::resources::StrokeStyleObj;
use super::{E_INVALIDARG, E_NOTIMPL, mat_opt, vec2};
use d2d::geometry::{combine, tessellate, Figure, FillMode, Geometry, Segment, StrokeStyle, IDENTITY};

/// `D2DERR_BAD_NUMBER`: a scanned query whose workspace rejects the shape.
const BAD_NUMBER: windows_core::HRESULT = windows_core::HRESULT(0x8899_0011u32 as i32);
use std::cell::RefCell;
use std::rc::Rc;
use windows::Win32::Graphics::Direct2D::Common::*;
use windows::Win32::Graphics::Direct2D::*;
use windows_core::{BOOL, Interface, OutRef, Ref, Result as WResult, implement};
use windows_numerics::{Matrix3x2, Vector2};

/// A path geometry under construction by its sink; shared (`Rc`) between the `GeometryObj` and the
/// `ID2D1GeometrySink` its `Open` handed out.
pub(crate) struct PathBuild {
    fill: FillMode,
    figures: Vec<Figure>,
    cur: Option<Figure>,
    /// Current `D2D1_PATH_SEGMENT` flags (sticky, set by `SetSegmentFlags`), recorded per segment.
    cur_flags: u32,
}
impl Default for PathBuild {
    fn default() -> Self {
        Self { fill: FillMode::Alternate, figures: Vec::new(), cur: None, cur_flags: 0 }
    }
}

pub(crate) enum GeomRepr {
    Shape(Geometry),
    Path(Rc<RefCell<PathBuild>>),
    /// A geometry group / transformed geometry keeps its source COM object(s) (so the source getters
    /// hand them back, as native does) alongside the fill mode / transform.
    Group { fill: FillMode, children: Vec<ID2D1Geometry> },
    Transformed { matrix: [f32; 6], source: ID2D1Geometry },
}

#[implement(ID2D1PathGeometry1, ID2D1PathGeometry, ID2D1RectangleGeometry, ID2D1RoundedRectangleGeometry, ID2D1EllipseGeometry, ID2D1GeometryGroup, ID2D1TransformedGeometry)]
pub(crate) struct GeometryObj {
    repr: GeomRepr,
    /// The factory that created this geometry, returned by `GetFactory` (native resources hold a
    /// strong factory reference). Set via [`Self::with_factory`] at every creation site.
    factory: Option<ID2D1Factory>,
}

impl GeometryObj {
    pub(crate) fn shape(g: Geometry) -> Self {
        Self { repr: GeomRepr::Shape(g), factory: None }
    }
    pub(crate) fn path() -> Self {
        Self { repr: GeomRepr::Path(Rc::new(RefCell::new(PathBuild::default()))), factory: None }
    }
    pub(crate) fn group(fill: FillMode, children: Vec<ID2D1Geometry>) -> Self {
        Self { repr: GeomRepr::Group { fill, children }, factory: None }
    }
    pub(crate) fn transformed(matrix: [f32; 6], source: ID2D1Geometry) -> Self {
        Self { repr: GeomRepr::Transformed { matrix, source }, factory: None }
    }
    /// Record the creating factory (chained before `.into()`), so `GetFactory` is faithful.
    pub(crate) fn with_factory(mut self, factory: ID2D1Factory) -> Self {
        self.factory = Some(factory);
        self
    }
    /// This geometry's owning factory (for a draw's cross-factory check), if it carries one.
    pub(crate) fn owning_factory(&self) -> Option<ID2D1Factory> {
        self.factory.clone()
    }
    /// The core geometry this object stands for (a finished path is assembled from its figures; a
    /// group/transformed recurses into its source objects).
    pub(crate) fn geometry(&self) -> Geometry {
        match &self.repr {
            GeomRepr::Shape(g) => g.clone(),
            GeomRepr::Path(b) => {
                let b = b.borrow();
                Geometry::Path { fill: b.fill, figures: b.figures.clone() }
            }
            GeomRepr::Group { fill, children } => Geometry::Group {
                fill: *fill,
                children: children.iter().filter_map(|c| c.cast_object_ref::<GeometryObj>().ok().map(|o| o.geometry())).collect(),
            },
            GeomRepr::Transformed { matrix, source } => Geometry::Transformed {
                matrix: *matrix,
                child: Box::new(source.cast_object_ref::<GeometryObj>().map(|o| o.geometry()).unwrap_or(Geometry::Rect([0.0; 4]))),
            },
        }
    }
}

fn rect_f(b: [f32; 4]) -> D2D_RECT_F {
    D2D_RECT_F { left: b[0], top: b[1], right: b[2], bottom: b[3] }
}

fn pt(p: [f32; 2]) -> Vector2 {
    Vector2 { X: p[0], Y: p[1] }
}

/// The core geometry behind another `ID2D1Geometry` arg (for combine/compare).
fn other_geom(g: Ref<ID2D1Geometry>) -> Option<Geometry> {
    g.as_ref()?.cast_object_ref::<GeometryObj>().ok().map(|o| o.geometry())
}

/// The stroke style behind an `ID2D1StrokeStyle` arg (null / foreign ⇒ default), for the widen
/// queries. Mirrors `context.rs::stroke_of`.
fn stroke_of(s: Ref<ID2D1StrokeStyle>) -> StrokeStyle {
    s.as_ref().and_then(|s| s.cast_object_ref::<StrokeStyleObj>().ok()).map(|o| o.style.clone()).unwrap_or_default()
}

/// Replay a combine/simplify/outline result (always a `Geometry::Path`) into the caller's
/// `ID2D1SimplifiedGeometrySink`, exactly as native drives it: fill mode, then per figure a
/// `BeginFigure`, runs of lines/beziers, and `EndFigure`. Native leaves the sink open (the caller
/// closes it), so we do not call `Close`.
fn replay(geom: &Geometry, sink: &ID2D1SimplifiedGeometrySink) {
    let Geometry::Path { fill, figures } = geom else { return };
    unsafe {
        sink.SetFillMode(D2D1_FILL_MODE(if *fill == FillMode::Winding { 1 } else { 0 }));
        for f in figures {
            sink.BeginFigure(pt(f.start), D2D1_FIGURE_BEGIN(if f.filled { 0 } else { 1 }));
            // The recorded stream is per-segment, so batching buys nothing. Quadratic/Arc never
            // appear in a simplify/outline/combine result.
            for s in &f.segments {
                match s {
                    Segment::Line(p) => sink.AddLines(&[pt(*p)]),
                    Segment::Cubic(a, b, c) => sink.AddBeziers(&[D2D1_BEZIER_SEGMENT { point1: pt(*a), point2: pt(*b), point3: pt(*c) }]),
                    _ => {}
                }
            }
            sink.EndFigure(D2D1_FIGURE_END(if f.closed { 1 } else { 0 }));
        }
    }
}

// ID2D1Geometry (shared base of all four concrete interfaces): the four query kinds the corpus uses
// are forwarded; the stroke/boolean/stream queries are E_NOTIMPL (no passing scene uses them).
impl ID2D1Geometry_Impl for GeometryObj_Impl {
    fn GetBounds(&self, worldtransform: *const Matrix3x2) -> WResult<D2D_RECT_F> {
        Ok(rect_f(self.geometry().get_bounds(&mat_opt(worldtransform))))
    }
    fn ComputeArea(&self, worldtransform: *const Matrix3x2, tol: f32) -> WResult<f32> {
        let (g, m) = (self.geometry(), mat_opt(worldtransform));
        if g.area_bad_number(&m) {
            return Err(BAD_NUMBER.into());
        }
        Ok(g.compute_area(&m, tol))
    }
    fn ComputeLength(&self, worldtransform: *const Matrix3x2, tol: f32) -> WResult<f32> {
        Ok(self.geometry().compute_length(&mat_opt(worldtransform), tol))
    }
    fn FillContainsPoint(&self, point: &Vector2, worldtransform: *const Matrix3x2, tol: f32) -> WResult<BOOL> {
        Ok(self.geometry().fill_contains(vec2(point), &mat_opt(worldtransform), tol).into())
    }
    fn GetWidenedBounds(&self, w: f32, s: Ref<ID2D1StrokeStyle>, t: *const Matrix3x2, tol: f32) -> WResult<D2D_RECT_F> {
        let (st, m) = (stroke_of(s), mat_opt(t));
        if self.geometry().widen_query_bad_number(w, &st, &m, tol, None) {
            return Err(BAD_NUMBER.into());
        }
        Ok(rect_f(self.geometry().widened_bounds(w, &st, &m, tol)))
    }
    fn StrokeContainsPoint(&self, p: &Vector2, w: f32, s: Ref<ID2D1StrokeStyle>, t: *const Matrix3x2, tol: f32) -> WResult<BOOL> {
        let (st, m) = (stroke_of(s), mat_opt(t));
        if self.geometry().widen_query_bad_number(w, &st, &m, tol, Some(vec2(p))) {
            return Err(BAD_NUMBER.into());
        }
        Ok(self.geometry().stroke_contains(vec2(p), w, &st, &m, tol).into())
    }
    fn CompareWithGeometry(&self, g: Ref<ID2D1Geometry>, t: *const Matrix3x2, tol: f32) -> WResult<D2D1_GEOMETRY_RELATION> {
        let other = other_geom(g).ok_or(windows_core::Error::from(E_NOTIMPL))?;
        if self.geometry().compare_bad_number(&other, &mat_opt(t)) {
            return Err(BAD_NUMBER.into());
        }
        Ok(D2D1_GEOMETRY_RELATION(combine::compare_with_geometry(&self.geometry(), &other, &mat_opt(t), tol) as i32))
    }
    fn Simplify(&self, o: D2D1_GEOMETRY_SIMPLIFICATION_OPTION, t: *const Matrix3x2, tol: f32, s: Ref<ID2D1SimplifiedGeometrySink>) -> WResult<()> {
        let sink = s.as_ref().ok_or(windows_core::Error::from(E_NOTIMPL))?;
        replay(&combine::simplify(&self.geometry(), o.0 as u32, &mat_opt(t), tol), sink);
        Ok(())
    }
    fn Tessellate(&self, t: *const Matrix3x2, tol: f32, s: Ref<ID2D1TessellationSink>) -> WResult<()> {
        let sink = s.as_ref().ok_or(windows_core::Error::from(E_NOTIMPL))?;
        if self.geometry().tessellate_bad_number(&mat_opt(t)) {
            return Err(BAD_NUMBER.into());
        }
        let tris: Vec<D2D1_TRIANGLE> = tessellate::tessellate(&self.geometry(), &mat_opt(t), tol)
            .iter()
            .map(|tr| D2D1_TRIANGLE { point1: pt(tr[0]), point2: pt(tr[1]), point3: pt(tr[2]) })
            .collect();
        unsafe { sink.AddTriangles(&tris) };
        Ok(())
    }
    fn CombineWithGeometry(&self, g: Ref<ID2D1Geometry>, m: D2D1_COMBINE_MODE, t: *const Matrix3x2, tol: f32, s: Ref<ID2D1SimplifiedGeometrySink>) -> WResult<()> {
        let other = other_geom(g).ok_or(windows_core::Error::from(E_NOTIMPL))?;
        let sink = s.as_ref().ok_or(windows_core::Error::from(E_NOTIMPL))?;
        if self.geometry().combine_bad_number(&other, &IDENTITY, &mat_opt(t)) {
            return Err(BAD_NUMBER.into());
        }
        replay(&combine::combine_with_geometry(&self.geometry(), &other, m.0 as u32, &mat_opt(t), tol), sink);
        Ok(())
    }
    fn Outline(&self, t: *const Matrix3x2, tol: f32, s: Ref<ID2D1SimplifiedGeometrySink>) -> WResult<()> {
        let sink = s.as_ref().ok_or(windows_core::Error::from(E_NOTIMPL))?;
        if self.geometry().outline_bad_number(&mat_opt(t)) {
            return Err(BAD_NUMBER.into());
        }
        replay(&combine::outline(&self.geometry(), &mat_opt(t), tol), sink);
        Ok(())
    }
    fn ComputePointAtLength(&self, len: f32, t: *const Matrix3x2, tol: f32, p: *mut Vector2, u: *mut Vector2) -> WResult<()> {
        // Native returns the point + unit tangent at arc-length `len`; empty geometry ⇒ origin/zero.
        let (point, tangent) = self.geometry().point_at_length(len, &mat_opt(t), tol).unwrap_or(([0.0; 2], [0.0; 2]));
        unsafe {
            if !p.is_null() {
                *p = pt(point);
            }
            if !u.is_null() {
                *u = pt(tangent);
            }
        }
        Ok(())
    }
    fn Widen(&self, w: f32, s: Ref<ID2D1StrokeStyle>, t: *const Matrix3x2, tol: f32, sink: Ref<ID2D1SimplifiedGeometrySink>) -> WResult<()> {
        // Stream the widened region exactly as native does (see `Geometry::widen_figures`): an
        // axis-aligned rect as clean alternate-fill rings, else winding open polylines. Exact for
        // rects and flat-cap / miter-or-bevel-join strokes; round joins/caps (beziers) and square
        // caps (separate figure) differ — a core-widener limitation noted on `widen_figures`.
        let sink = sink.as_ref().ok_or(windows_core::Error::from(E_NOTIMPL))?;
        let (st, m) = (stroke_of(s), mat_opt(t));
        if self.geometry().widen_query_bad_number(w, &st, &m, tol, None) {
            return Err(BAD_NUMBER.into());
        }
        replay(&self.geometry().widen_figures(w, &st, &m, [1.0, 1.0], tol), sink);
        Ok(())
    }
}

impl ID2D1Resource_Impl for GeometryObj_Impl {
    fn GetFactory(&self, factory: windows_core::OutRef<ID2D1Factory>) {
        let _ = factory.write(self.factory.clone());
    }
}

impl ID2D1PathGeometry_Impl for GeometryObj_Impl {
    fn Open(&self) -> WResult<ID2D1GeometrySink> {
        match &self.repr {
            GeomRepr::Path(b) => Ok(GeometrySinkObj { build: b.clone() }.into()),
            _ => Err(E_NOTIMPL.into()),
        }
    }
    fn Stream(&self, sink: Ref<ID2D1GeometrySink>) -> WResult<()> {
        // `ID2D1PathGeometry::Stream`: re-emit the stored path to the caller's sink — fill mode, then
        // per figure a BeginFigure, each segment (with its SetSegmentFlags), and EndFigure — leaving the
        // sink open for the caller to Close, as native does. Unlike `replay` (combine/simplify output,
        // lines+beziers only) a streamed path keeps quadratics and arcs verbatim.
        let Some(sink) = sink.as_ref() else { return Err(E_INVALIDARG.into()) };
        let Geometry::Path { fill, figures } = self.geometry() else { return Err(E_NOTIMPL.into()) };
        unsafe {
            sink.SetFillMode(D2D1_FILL_MODE(if fill == FillMode::Winding { 1 } else { 0 }));
            for f in &figures {
                sink.BeginFigure(pt(f.start), D2D1_FIGURE_BEGIN(if f.filled { 0 } else { 1 }));
                for (k, s) in f.segments.iter().enumerate() {
                    sink.SetSegmentFlags(D2D1_PATH_SEGMENT(f.seg_flags.get(k).copied().unwrap_or(0) as i32));
                    match s {
                        Segment::Line(p) => sink.AddLine(pt(*p)),
                        Segment::Cubic(a, b, c) => sink.AddBezier(&D2D1_BEZIER_SEGMENT { point1: pt(*a), point2: pt(*b), point3: pt(*c) }),
                        Segment::Quadratic(a, b) => sink.AddQuadraticBezier(&D2D1_QUADRATIC_BEZIER_SEGMENT { point1: pt(*a), point2: pt(*b) }),
                        Segment::Arc { to, radius, rotation, clockwise, large } => sink.AddArc(&D2D1_ARC_SEGMENT {
                            point: pt(*to),
                            size: D2D_SIZE_F { width: radius[0], height: radius[1] },
                            rotationAngle: *rotation,
                            sweepDirection: D2D1_SWEEP_DIRECTION(if *clockwise { 1 } else { 0 }),
                            arcSize: D2D1_ARC_SIZE(if *large { 1 } else { 0 }),
                        }),
                    }
                }
                sink.EndFigure(D2D1_FIGURE_END(if f.closed { 1 } else { 0 }));
            }
        }
        Ok(())
    }
    fn GetSegmentCount(&self) -> WResult<u32> {
        // Native counts each closed figure's implicit closing segment (`combine::segment_count`),
        // not just the streamed segments.
        Ok(combine::segment_count(&self.geometry()) as u32)
    }
    fn GetFigureCount(&self) -> WResult<u32> {
        match &self.repr {
            GeomRepr::Path(b) => Ok(b.borrow().figures.len() as u32),
            _ => Ok(0),
        }
    }
}

impl ID2D1PathGeometry1_Impl for GeometryObj_Impl {
    fn ComputePointAndSegmentAtLength(&self, len: f32, start: u32, t: *const Matrix3x2, tol: f32, out: *mut D2D1_POINT_DESCRIPTION) -> WResult<()> {
        // Point + unit tangent + the segment/figure the point lies on and the length before it;
        // empty geometry ⇒ all zero. Values verified against native (see docs/gaps.md).
        let (p, u, seg, fig, l) = self.geometry().point_and_segment_at_length(len, start, &mat_opt(t), tol).unwrap_or(([0.0; 2], [0.0; 2], 0, 0, 0.0));
        if !out.is_null() {
            unsafe {
                *out = D2D1_POINT_DESCRIPTION { point: pt(p), unitTangentVector: pt(u), endSegment: seg, endFigure: fig, lengthToEndSegment: l };
            }
        }
        Ok(())
    }
}

// The concrete shape getters hand back the authored shape (native echoes what the geometry was
// created with); QI exposes all four on every GeometryObj.
impl ID2D1RectangleGeometry_Impl for GeometryObj_Impl {
    fn GetRect(&self, rect: *mut D2D_RECT_F) {
        if let Geometry::Rect(r) = self.geometry() {
            unsafe { *rect = rect_f(r) };
        }
    }
}
impl ID2D1RoundedRectangleGeometry_Impl for GeometryObj_Impl {
    fn GetRoundedRect(&self, rr: *mut D2D1_ROUNDED_RECT) {
        if let Geometry::Rounded { rect, radius } = self.geometry() {
            unsafe { *rr = D2D1_ROUNDED_RECT { rect: rect_f(rect), radiusX: radius[0], radiusY: radius[1] } };
        }
    }
}
impl ID2D1EllipseGeometry_Impl for GeometryObj_Impl {
    fn GetEllipse(&self, e: *mut D2D1_ELLIPSE) {
        if let Geometry::Ellipse { center, radius } = self.geometry() {
            unsafe { *e = D2D1_ELLIPSE { point: Vector2 { X: center[0], Y: center[1] }, radiusX: radius[0], radiusY: radius[1] } };
        }
    }
}

// Geometry group / transformed geometry: the source getters hand back the stored COM objects, as
// native does. (QI exposes these on every GeometryObj, as with the four shape interfaces above.)
impl ID2D1GeometryGroup_Impl for GeometryObj_Impl {
    fn GetFillMode(&self) -> D2D1_FILL_MODE {
        match &self.repr {
            GeomRepr::Group { fill, .. } => D2D1_FILL_MODE(if *fill == FillMode::Winding { 1 } else { 0 }),
            _ => D2D1_FILL_MODE(0),
        }
    }
    fn GetSourceGeometryCount(&self) -> u32 {
        match &self.repr {
            GeomRepr::Group { children, .. } => children.len() as u32,
            _ => 0,
        }
    }
    fn GetSourceGeometries(&self, geometries: *mut Option<ID2D1Geometry>, count: u32) {
        if let GeomRepr::Group { children, .. } = &self.repr {
            for (i, c) in children.iter().take(count as usize).enumerate() {
                unsafe { geometries.add(i).write(Some(c.clone())) };
            }
        }
    }
}
impl ID2D1TransformedGeometry_Impl for GeometryObj_Impl {
    fn GetSourceGeometry(&self, sourcegeometry: OutRef<ID2D1Geometry>) {
        if let GeomRepr::Transformed { source, .. } = &self.repr {
            let _ = sourcegeometry.write(Some(source.clone()));
        }
    }
    fn GetTransform(&self, transform: *mut Matrix3x2) {
        if let GeomRepr::Transformed { matrix: m, .. } = &self.repr {
            unsafe { *transform = Matrix3x2 { M11: m[0], M12: m[1], M21: m[2], M22: m[3], M31: m[4], M32: m[5] } };
        }
    }
}

// --- path sink ---------------------------------------------------------------------------------

#[implement(ID2D1GeometrySink)]
pub(crate) struct GeometrySinkObj {
    build: Rc<RefCell<PathBuild>>,
}

impl ID2D1SimplifiedGeometrySink_Impl for GeometrySinkObj_Impl {
    fn SetFillMode(&self, fillmode: D2D1_FILL_MODE) {
        self.build.borrow_mut().fill = if fillmode.0 == 1 { FillMode::Winding } else { FillMode::Alternate };
    }
    // `SetSegmentFlags` is sticky: it sets the `D2D1_PATH_SEGMENT` flags recorded for each segment
    // added until the next call. The core stroker honours FORCE_UNSTROKED (segment is a gap); the
    // flags are ignored by fill. [SetSegmentFlags]
    fn SetSegmentFlags(&self, flags: D2D1_PATH_SEGMENT) {
        self.build.borrow_mut().cur_flags = flags.0 as u32;
    }
    fn BeginFigure(&self, startpoint: &Vector2, figurebegin: D2D1_FIGURE_BEGIN) {
        self.build.borrow_mut().cur =
            Some(Figure { start: vec2(startpoint), filled: figurebegin.0 == 0, closed: false, segments: Vec::new(), seg_flags: Vec::new() });
    }
    fn AddLines(&self, points: *const Vector2, count: u32) {
        let mut b = self.build.borrow_mut();
        let fl = b.cur_flags;
        if let Some(f) = b.cur.as_mut() {
            for i in 0..count as isize {
                f.segments.push(Segment::Line(vec2(unsafe { &*points.offset(i) })));
                f.seg_flags.push(fl);
            }
        }
    }
    fn AddBeziers(&self, beziers: *const D2D1_BEZIER_SEGMENT, count: u32) {
        let mut b = self.build.borrow_mut();
        let fl = b.cur_flags;
        if let Some(f) = b.cur.as_mut() {
            for i in 0..count as isize {
                let s = unsafe { &*beziers.offset(i) };
                f.segments.push(Segment::Cubic(vec2(&s.point1), vec2(&s.point2), vec2(&s.point3)));
                f.seg_flags.push(fl);
            }
        }
    }
    fn EndFigure(&self, figureend: D2D1_FIGURE_END) {
        let mut b = self.build.borrow_mut();
        if let Some(mut f) = b.cur.take() {
            f.closed = figureend.0 == 1;
            b.figures.push(f);
        }
    }
    fn Close(&self) -> WResult<()> {
        // Native's sink rejects a path arc with a negative radius: Close returns E_INVALIDARG and the
        // geometry is never created (`Geometry::has_invalid_arc_radius`; fuzz seeds 20003/20041).
        let b = self.build.borrow();
        let g = Geometry::Path { fill: b.fill, figures: b.figures.clone() };
        if g.has_invalid_arc_radius() {
            return Err(windows_core::Error::from(E_INVALIDARG));
        }
        Ok(())
    }
}

impl ID2D1GeometrySink_Impl for GeometrySinkObj_Impl {
    fn AddLine(&self, point: &Vector2) {
        let mut b = self.build.borrow_mut();
        let fl = b.cur_flags;
        if let Some(f) = b.cur.as_mut() {
            f.segments.push(Segment::Line(vec2(point)));
            f.seg_flags.push(fl);
        }
    }
    fn AddBezier(&self, bezier: *const D2D1_BEZIER_SEGMENT) {
        let s = unsafe { &*bezier };
        let mut b = self.build.borrow_mut();
        let fl = b.cur_flags;
        if let Some(f) = b.cur.as_mut() {
            f.segments.push(Segment::Cubic(vec2(&s.point1), vec2(&s.point2), vec2(&s.point3)));
            f.seg_flags.push(fl);
        }
    }
    fn AddQuadraticBezier(&self, bezier: *const D2D1_QUADRATIC_BEZIER_SEGMENT) {
        let s = unsafe { &*bezier };
        let mut b = self.build.borrow_mut();
        let fl = b.cur_flags;
        if let Some(f) = b.cur.as_mut() {
            f.segments.push(Segment::Quadratic(vec2(&s.point1), vec2(&s.point2)));
            f.seg_flags.push(fl);
        }
    }
    fn AddQuadraticBeziers(&self, beziers: *const D2D1_QUADRATIC_BEZIER_SEGMENT, count: u32) {
        let mut b = self.build.borrow_mut();
        let fl = b.cur_flags;
        if let Some(f) = b.cur.as_mut() {
            for i in 0..count as isize {
                let s = unsafe { &*beziers.offset(i) };
                f.segments.push(Segment::Quadratic(vec2(&s.point1), vec2(&s.point2)));
                f.seg_flags.push(fl);
            }
        }
    }
    fn AddArc(&self, arc: *const D2D1_ARC_SEGMENT) {
        let s = unsafe { &*arc };
        let mut b = self.build.borrow_mut();
        let fl = b.cur_flags;
        if let Some(f) = b.cur.as_mut() {
            f.seg_flags.push(fl);
            f.segments.push(Segment::Arc {
                to: [s.point.X, s.point.Y],
                radius: [s.size.width, s.size.height],
                rotation: s.rotationAngle,
                clockwise: s.sweepDirection.0 == 1,
                large: s.arcSize.0 == 1,
            });
        }
    }
}

// --- mesh (ID2D1Mesh: a tessellated triangle set, filled by ID2D1RenderTarget::FillMesh) ---------

/// A mesh is just the triangle list its tessellation sink collected (device/DIP space). `FillMesh`
/// fills each triangle on its own in aliased mode, exactly as native does.
#[implement(ID2D1Mesh)]
pub(crate) struct MeshObj {
    pub(crate) tris: Rc<RefCell<Vec<[[f32; 2]; 3]>>>,
    pub(crate) factory: Option<ID2D1Factory>,
}
impl MeshObj {
    pub(crate) fn new(factory: ID2D1Factory) -> Self {
        Self { tris: Rc::new(RefCell::new(Vec::new())), factory: Some(factory) }
    }
}
impl ID2D1Resource_Impl for MeshObj_Impl {
    fn GetFactory(&self, f: OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
impl ID2D1Mesh_Impl for MeshObj_Impl {
    fn Open(&self) -> WResult<ID2D1TessellationSink> {
        Ok(MeshSinkObj { tris: self.tris.clone() }.into())
    }
}

#[implement(ID2D1TessellationSink)]
struct MeshSinkObj {
    tris: Rc<RefCell<Vec<[[f32; 2]; 3]>>>,
}
impl ID2D1TessellationSink_Impl for MeshSinkObj_Impl {
    fn AddTriangles(&self, triangles: *const D2D1_TRIANGLE, count: u32) {
        let mut t = self.tris.borrow_mut();
        for i in 0..count as isize {
            let tr = unsafe { &*triangles.offset(i) };
            t.push([[tr.point1.X, tr.point1.Y], [tr.point2.X, tr.point2.Y], [tr.point3.X, tr.point3.Y]]);
        }
    }
    fn Close(&self) -> WResult<()> {
        Ok(())
    }
}
