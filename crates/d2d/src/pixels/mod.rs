//! Phase 1B CPU rasterizer: `Clear`, rectangle and path fills, 8×8 per-primitive
//! coverage anti-aliasing, aliased center/top-left sampling, and the primitive blend
//! modes — composited into a premultiplied `B8G8R8A8_UNORM` [`PixelBuffer`] to match
//! native `d2d1.dll` on the WARP software device.
//!
//! Polygon fills reproduce native `CHwRasterizer`'s scan converter
//! (edge setup then active-edge-list walk): device coordinates go
//! to 28.4 fixed point (`ceil(16·(v−0.5))`, the −0.5 pixel-centre bias folded in), then ×8+56
//! into an 8× supersampling grid; each edge walks a fixed-point DDA one sub-scanline at a time,
//! and the covered sub-samples accumulate per pixel as `samples ∈ 0..64` ([`Edge`],
//! [`add_interval`]). The extra 28.4 bit is DDA headroom — the X quantisation step is exactly
//! 1/8 (Q2, confirmed byte-for-byte by `1b2-01-sixteenth-offsets`).
//!
//! Coverage then becomes colour exactly as part 1 pinned it against the oracle: premultiply with
//! round-to-nearest-even (WARP's float→UNORM8 rule); quantise coverage to 8 bits as
//! `round(samples*255/64)` before normalising; blend the coverage-weighted premultiplied source
//! in f32 single-stage source-over. Callers pass **device-space** coordinates (world transform ∘
//! DPI already applied); clipping is not this layer's job.

use d2d_types::contracts::PixelBuffer;

pub(crate) mod copy_tess;
pub(crate) mod warp_geom;

/// Anti-alias mode for a fill (`D2D1_ANTIALIAS_MODE`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Aa {
    /// 8×8 coverage anti-aliasing (default).
    PerPrimitive,
    /// Single centre sample with the top-left fill rule.
    Aliased,
}

/// Primitive blend mode (`D2D1_PRIMITIVE_BLEND`). Operates on premultiplied channels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Blend {
    SourceOver,
    Copy,
    Min,
    Add,
    Max,
}

/// Fill rule for a path (`D2D1_FILL_MODE`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FillMode {
    Alternate,
    Winding,
}

/// Premultiply a straight RGBA colour (D2D `D2D1_COLOR_F`, f32 channels in `[0,1]`) into an
/// 8-bit premultiplied `[B, G, R, A]`, rounding each channel to nearest-even — WARP's
/// float→UNORM8 rule. Channels outside `[0,1]` are clamped.
pub fn premultiply([r, g, b, a]: [f32; 4]) -> [u8; 4] {
    let q = |c: f32| (c * a * 255.0).round_ties_even().clamp(0.0, 255.0) as u8;
    [q(b), q(g), q(r), (a * 255.0).round_ties_even().clamp(0.0, 255.0) as u8]
}

/// A render target wrapping a premultiplied 8-bit [`PixelBuffer`]. The compositing works in a fixed
/// logical BGRA order; `order` maps that to the buffer's byte slots so `R8G8B8A8_UNORM` reuses the
/// whole pipeline unchanged (its bytes are RGBA, oracle-confirmed the same round-ties-even premul).
pub struct Target<'a> {
    buf: &'a mut PixelBuffer,
    order: [usize; 4],
    /// Optional **unquantised** premultiplied source (BGRA, 0..1 — the `src_f` the float target uses)
    /// for the MIN/ADD/MAX primitive blends. Those blends `min`/`+`/`max` the coverage-weighted source
    /// against the dest; taking the source already rounded to a byte (`[u8;4]`) double-rounds and is
    /// ±1 near halves (native keeps the source float through the blend — `min(dst, src+1−srcα)` etc.).
    /// SOURCE_OVER / COPY keep the pre-quantised `[u8;4]` path (their layered-translucent byte-exactness
    /// depends on quantising `cs` first), so this is read only by MIN/ADD/MAX in [`Self::blend_px`].
    src_f: Option<[f32; 4]>,
    /// The draw's clip as a pixel rect (see [`copy_tess::fill_copy_in`]); `None` is the whole target.
    clip: Option<[i32; 4]>,
}

impl<'a> Target<'a> {
    /// This target drawing under the pushed clips' pixel rect.
    pub(crate) fn with_clip(self, clip: Option<[i32; 4]>) -> Self {
        Target { clip, ..self }
    }

    pub fn new(buf: &'a mut PixelBuffer) -> Self {
        // logical [B,G,R,A] → buffer slot. B8G8R8A8 is identity; R8G8B8A8 swaps the B/R slots.
        let order = if buf.desc.format == d2d_types::abi::DXGI_FORMAT_R8G8B8A8_UNORM {
            [2, 1, 0, 3]
        } else {
            [0, 1, 2, 3]
        };
        Self { buf, order, src_f: None, clip: None }
    }

    /// Supply the unquantised premultiplied source (0..1 BGRA) for a MIN/ADD/MAX fill, so its blend
    /// rounds once instead of double-rounding the `[u8;4]` source (see [`Self::src_f`]).
    pub fn with_src_f(mut self, src_f: [f32; 4]) -> Self {
        self.src_f = Some(src_f);
        self
    }

    fn w(&self) -> usize {
        self.buf.desc.width as usize
    }
    fn h(&self) -> usize {
        self.buf.desc.height as usize
    }

    /// `Clear`: write `premul` (premultiplied BGRA) to every pixel. Independent of transform
    /// and primitive blend; ignores anti-alias mode.
    pub fn clear(&mut self, premul: [u8; 4]) {
        let (w, h, stride, o) = (self.w(), self.h(), self.buf.stride, self.order);
        for y in 0..h {
            let row = &mut self.buf.data[y * stride..];
            for x in 0..w {
                let p = &mut row[x * 4..x * 4 + 4];
                for l in 0..4 {
                    p[o[l]] = premul[l];
                }
            }
        }
    }

    /// Fill an axis-aligned device-space rectangle with premultiplied BGRA `src`. Empty,
    /// inverted or non-finite rectangles produce no pixels. Pixel-aligned opaque rectangles
    /// fall out of the general coverage path as full-coverage (no fringe), matching native's
    /// `FillRegionShapeWriteOnce` fast path byte-for-byte without a separate branch.
    pub fn fill_rect(&mut self, rect: [f32; 4], src: [u8; 4], aa: Aa, blend: Blend) {
        let (w, h) = (self.w(), self.h());
        rasterize_rect(rect, aa, w, h, |x, y, cov| self.blend_px_rounded(x, y, src, cov, blend, true));
    }

    /// Fill a path (one or more closed figures of device-space vertices) with premultiplied
    /// BGRA `src` under `fill`. Figures are implicitly closed; this just expands them to the
    /// edge list [`fill_edges`] rasterizes.
    pub fn fill_path(
        &mut self,
        figures: &[Vec<[f32; 2]>],
        fill: FillMode,
        src: [u8; 4],
        aa: Aa,
        blend: Blend,
        geometry: bool,
    ) {
        self.fill_path_in(figures, fill, src, aa, blend, geometry, true);
    }

    /// [`Self::fill_path`]. `rect_fixed` gives a single axis-aligned rectangle the rect special case's 16.8 corners;
    /// a rectangle shape Direct2D sends to its general rasterizer instead (`FillShapeData`) keeps exact ones.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fill_path_in(
        &mut self,
        figures: &[Vec<[f32; 2]>],
        fill: FillMode,
        src: [u8; 4],
        aa: Aa,
        blend: Blend,
        geometry: bool,
        rect_fixed: bool,
    ) {
        let Some(figures) = clip_huge(figures) else { return };
        // A clipped fill is native's FillShapeData geometry, pixel-centre shift included.
        let geometry = geometry || matches!(figures, std::borrow::Cow::Owned(_));
        let figures = &*figures;
        // COPY-blend (opaque) anti-aliased fills take native's CHwRasterizer trapezoid-fringe coverage,
        // not the 8×8 area the SOURCE_OVER path uses, on their slanted edges (`IsAlphaBlendingRequired`
        // is false for COPY, so native takes the opaque trapezoid path). Route single-contour CONVEX
        // fills through the faithful active-edge-list walk ([`copy_tess::fill_convex_copy`]); everything
        // else (concave, multi-contour, aliased, non-COPY) keeps the area path. Convexity guarantees the
        // one-left/one-right-edge-per-band the walk assumes. `geometry` selects native's pixel-centre
        // convention (FillGeometry shifts −0.5, FillRectangle doesn't). The trapezoid/span triangles lerp
        // the target toward the source by their 8-bit coverage in one rounding ([`copy_tess::Cov::lerp`]).
        // Byte-exact on rects/rotated rects and on curve trapezoid rails; curve cap/vertex rows carry the
        // shared scan-converter edge residual (≤10). Strictly no-worse than the old area path. Axis-aligned
        // rects skip the walk — native fills them via its rect special case, so the −0.5 geometry shift is
        // wrong for them (the fractional-rect COPY fills fuzz 8876/8982 that the walk regressed).
        // A single axis-aligned rect keeps native's rect special case (no −0.5 pixel-centre shift); every
        // other COPY per-primitive fill — convex, concave, self-intersecting, or multi-contour — takes the
        // faithful active-edge-list walk ([`copy_tess::fill_copy`]), which now pairs active edges by winding
        // so non-convex and multi-contour shapes are byte-exact too (star/overlap probes), not just convex.
        // Every COPY per-primitive fill except a single axis-aligned rect (native's rect special case)
        // takes the faithful active-edge-list walk — Winding and Alternate (even-odd), convex, concave,
        // self-intersecting, or multi-contour — which pairs active edges by the fill rule.
        let single_axis_rect = is_single_axis_rect(figures);
        // MIN of an opaque brush takes the same opaque walk (it needs no alpha blending either).
        let opaque_min = blend == Blend::Min && src[3] == 255;
        if (blend == Blend::Copy || opaque_min) && aa == Aa::PerPrimitive && !single_axis_rect && !figures.iter().all(|f| f.is_empty()) {
            let (w, h) = (self.w(), self.h());
            let slices: Vec<&[[f32; 2]]> = figures.iter().map(|f| f.as_slice()).collect();
            let alternate = fill == FillMode::Alternate;
            let clip = self.clip.unwrap_or([0, 0, w as i32, h as i32]);
            copy_tess::fill_copy_in(&slices, alternate, w, h, geometry, clip, |x, y, cov, _| {
                let (i, o) = (y * self.buf.stride + x * 4, self.order);
                let p = &mut self.buf.data[i..i + 4];
                let dst = [p[o[0]], p[o[1]], p[o[2]], p[o[3]]];
                let out = if opaque_min { cov.min_white(dst, src) } else { cov.lerp(dst, src) };
                for l in 0..4 {
                    p[o[l]] = out[l];
                }
            });
            return;
        }
        // An anti-aliased single axis-aligned rectangle fills with its corners in 16.8 fixed point: each
        // device coordinate rounds to the nearest 1/256 (ties to even) before the 8×8 coverage (VM probes:
        // an edge up to half a step past a sample centre still takes the sample, for a rect path, rect
        // geometry or FillRectangle in any orientation). Other outlines keep their exact vertices.
        let snapped: Vec<Vec<[f32; 2]>>;
        let figures = if rect_fixed && aa == Aa::PerPrimitive && is_single_axis_rect(figures) {
            let q = |v: f32| ((v as f64 * 256.0).round_ties_even() / 256.0) as f32;
            snapped = figures.iter().map(|f| f.iter().map(|p| [q(p[0]), q(p[1])]).collect()).collect();
            &snapped[..]
        } else {
            figures
        };
        // A rectangle's own fill (not a geometry) keeps the once-rounded source-over (see `blend_px_rounded`).
        let (w, h) = (self.w(), self.h());
        rasterize_edges(&figure_edges(figures), fill, aa, w, h, |x, y, cov| self.blend_px_rounded(x, y, src, cov, blend, !geometry));
    }

    /// An aliased fill of device-space figures through WARP's aliased edge walk
    /// ([`warp_geom::fill_aliased`]). d2d1's WARP geometry edge-list sink
    /// converts each point to 28.4 as `⌈16x − 8⌉` (the adapter matrix is 16·M with the −½-pixel centre
    /// shift). A non-finite vertex voids the fill, as elsewhere; a figure past |524287| is strip-clipped
    /// first ([`clip_huge`]), as the shape path that reaches this walk does.
    pub fn fill_warp28(&mut self, figures: &[Vec<[f32; 2]>], winding: bool, src: [u8; 4], blend: Blend) {
        let Some(figures) = clip_huge(figures) else { return };
        let figs: Vec<Vec<[i32; 2]>> = figures
            .iter()
            .map(|f| f.iter().map(|p| [((p[0] - 0.5) * 16.0).ceil() as i32, ((p[1] - 0.5) * 16.0).ceil() as i32]).collect())
            .collect();
        let (w, h) = (self.w(), self.h());
        warp_geom::fill_aliased(&figs, winding, w, h, |y, a, b| {
            for x in a..b {
                self.blend_px(x, y, src, 64, blend);
            }
        });
    }

    /// Rasterize a list of already-flattened line edges (device-space `[x0,y0,x1,y1]`, each
    /// implicitly part of a closed contour) under `fill`, compositing premultiplied BGRA `src`.
    /// This is the entry point curves/strokes (1C) flatten into. `PerPrimitive` reproduces
    /// native `CHwRasterizer`'s fixed-point 8×8 scan conversion (see module docs)
    /// `Aliased` is a single centre sample with the top-left rule.
    pub fn fill_edges(&mut self, edges: &[[f32; 4]], fill: FillMode, src: [u8; 4], aa: Aa, blend: Blend) {
        let (w, h) = (self.w(), self.h());
        rasterize_edges(edges, fill, aa, w, h, |x, y, cov| self.blend_px(x, y, src, cov, blend));
    }

    /// [`Self::fill_edges`] with a per-pixel premultiplied BGRA source `src(x, y)`.
    pub(crate) fn fill_edges_px(&mut self, edges: &[[f32; 4]], fill: FillMode, aa: Aa, blend: Blend, src: impl Fn(usize, usize) -> [u8; 4]) {
        let (w, h) = (self.w(), self.h());
        rasterize_edges(edges, fill, aa, w, h, |x, y, cov| self.blend_px(x, y, src(x, y), cov, blend));
    }

    fn blend_px(&mut self, x: usize, y: usize, src: [u8; 4], cov: u32, blend: Blend) {
        self.blend_px_rounded(x, y, src, cov, blend, false)
    }

    /// [`Self::blend_px`]; `rounded` takes the rectangle fill's source-over, which rounds `dst·(1 − a)`
    /// once ([`over_rounded`]) where a geometry fill goes through the output merger's 16-bit form
    /// ([`blend_cov`]): a FillRectangle's fringe needs it (r26 31705), a FillGeometry's the other
    /// (r26 31009 31250, r27 32616).
    fn blend_px_rounded(&mut self, x: usize, y: usize, src: [u8; 4], cov: u32, blend: Blend, rounded: bool) {
        let (i, o) = (y * self.buf.stride + x * 4, self.order);
        let p = &mut self.buf.data[i..i + 4];
        // Full-coverage opaque source-over reduces to writing the source: `composite` returns `src`
        // exactly when `cov == 64` and the source alpha is 255 (`cs8 == src`, `1 − sa == 0`). Skip the
        // destination read and the per-pixel float blend. Byte-identical.
        if blend == Blend::SourceOver && cov == 64 && src[3] == 255 {
            for l in 0..4 {
                p[o[l]] = src[l];
            }
            return;
        }
        let dst = [p[o[0]], p[o[1]], p[o[2]], p[o[3]]];
        // MIN/ADD/MAX round the coverage-weighted source ONCE from the unquantised float source (if the
        // caller supplied it), not from an already-rounded `[u8;4]` — matching the float target and
        // killing the ±1-near-halves fingerprint (blend-min-grid-2). SOURCE_OVER/COPY keep `composite`.
        let out = match (self.src_f, blend) {
            (Some(sf), Blend::Min) => {
                // `sf` is the straight-order premultiplied RGBA source in 0..1 (the float target's
                // `src_f`); `dst`/`out` are logical BGRA, so read `sf` through the R↔B swap [2,1,0,3].
                // Native's MIN is the source over WHITE, `min(dst, re(cs + (255 − csα)))` — the SAME
                // formula as `blend_cov`'s Min, but on the UNQUANTISED float source `cs = src·255·c`
                // (one rounding) instead of the pre-quantised `[u8;4]·c`, which kills the ±1-near-halves
                // fingerprint (blend-min-grid-2). ADD/MAX keep `composite`'s [u8;4] path (the one-round
                // float form regressed Max fuzz 20153/21107 — their native quantisation differs).
                let c = (cov.min(64) as f32 * 255.0 / 64.0).round_ties_even() / 255.0;
                const BGRA: [usize; 4] = [2, 1, 0, 3];
                let csa = sf[3] * 255.0 * c; // coverage-weighted source alpha, 0..255
                let re = |v: f32| v.round_ties_even().clamp(0.0, 255.0) as u8;
                core::array::from_fn(|k| re(sf[BGRA[k]] * 255.0 * c + (255.0 - csa)).min(dst[k]))
            }
            (_, Blend::SourceOver) if rounded => over_rounded(dst, src, cov),
            _ => composite(dst, src, cov, blend),
        };
        for l in 0..4 {
            p[o[l]] = out[l];
        }
    }
}

/// The device point [`clip_huge`] puts an infinite crossing at: `ceil(16·v − 8)` = −2²³ on both axes.
pub(crate) const FAR: [f32; 2] = [-524287.5; 2];

/// Native `CShapeBase::ClipWithRect` for a fill whose device extent passes
/// |524287|: the shape generator tests each bound corner against `0x48FFFFE0` and then
/// cuts every figure to `[-524287, 524287]²` with two `CStripClipper`s, y then x ([`strip`]). A
/// crossing's free coordinate is `((b−q)·p' − (b−p)·q')/(p−q)` in f32, so once that product passes
/// `FLT_MAX` (a rect with `w·h ≥ FLT_MAX`, w_bm dbx/dbl) it comes out ±inf or NaN. A NaN vertex voids
/// the whole fill (`None`, w_bm dbt/dbr-b, rfill-huge-b). An infinite one reaches 28.4 as
/// `(x·16 + y·0) − 8` = NaN, which `cvttss2si` and the edge setup's ±2²³ clamp put at (−2²³, −2²³):
/// the slope-1 edge to the far top-left that native draws (rfill-huge-l, w_bm dbl), our [`FAR`]. A
/// non-finite input vertex voids the fill, as before.
/// ponytail: clips in device space; native clips an axis-aligned non-identity world transform in shape
/// space, which rounds a finite crossing differently. Thread the shape matrix here if that shows up.
pub(crate) fn clip_huge(figures: &[Vec<[f32; 2]>]) -> Option<std::borrow::Cow<'_, [Vec<[f32; 2]>]>> {
    if !figures.iter().flatten().flatten().all(|v| v.is_finite()) {
        return None;
    }
    // `FAR` is our own stand-in from an earlier pass (fill_geometry's pre-grid clip), not a new crossing.
    if !figures.iter().flatten().any(|&p| p != FAR && p.iter().any(|v| v.abs() > 524287.0)) {
        return Some(figures.into());
    }
    let out: Vec<Vec<[f32; 2]>> = figures
        .iter()
        .filter_map(|f| {
            let (&s, rest) = f.split_first()?;
            let (s, segs) = clip_figure(s, &rest.iter().map(|&q| Seg::Line(q)).collect::<Vec<_>>())?;
            Some([s].into_iter().chain(segs.iter().map(|g| match *g { Seg::Line(q) | Seg::Cubic(_, _, q) => q })).collect())
        })
        .collect();
    if out.iter().flatten().flatten().any(|v| v.is_nan()) {
        return None;
    }
    let far = |p: [f32; 2]| if p.iter().all(|v| v.is_finite()) { p } else { FAR };
    Some(out.into_iter().map(|f| f.into_iter().map(far).collect::<Vec<_>>()).collect::<Vec<_>>().into())
}

/// The closed figures' edge list (`[x0, y0, x1, y1]` per segment); a figure under two points has none.
pub(crate) fn figure_edges(figures: &[Vec<[f32; 2]>]) -> Vec<[f32; 4]> {
    let ring = |f: &[[f32; 2]]| {
        let n = if f.len() < 2 { 0 } else { f.len() };
        (0..n).map(|i| [f[i][0], f[i][1], f[(i + 1) % n][0], f[(i + 1) % n][1]]).collect::<Vec<_>>()
    };
    figures.iter().flat_map(|f| ring(f)).collect()
}

/// A figure segment for the strip clip: a line to its point, or a cubic through its three.
#[derive(Clone, Copy)]
pub(crate) enum Seg {
    Line([f32; 2]),
    Cubic([f32; 2], [f32; 2], [f32; 2]),
}

/// [`clip_huge`]'s two strip passes (y, then x) over one closed figure of lines and cubics, or `None`
/// when it never touches the strips. Crossings come back as computed: ±inf, NaN and all.
pub(crate) fn clip_figure(start: [f32; 2], segs: &[Seg]) -> Option<([f32; 2], Vec<Seg>)> {
    clip_figure_to(start, segs, 524287.0)
}

/// [`clip_figure`] against the box `[-bound, bound]²` instead of the safe device box.
pub(crate) fn clip_figure_to(start: [f32; 2], segs: &[Seg], bound: f32) -> Option<([f32; 2], Vec<Seg>)> {
    clip_figure_rect(start, segs, [-bound, -bound, bound, bound])
}

/// [`clip_figure`] against the rect `[l, t, r, b]`, its vertical bounds first.
pub(crate) fn clip_figure_rect(start: [f32; 2], segs: &[Seg], [l, t, r, b]: [f32; 4]) -> Option<([f32; 2], Vec<Seg>)> {
    let (s, segs) = Strip::run(start, segs, 1, t, b)?;
    Strip::run(s, &segs, 0, l, r)
}

/// One `CStripClipper` pass along `axis` (0 = x, 1 = y), ported from d2d1's `BeginFigure`/`AddLines`/
/// `AddBeziers`/`EndFigure`. The outcode is
/// the sign bits of `v − lo` (1) and `hi − v` (2); x86's default NaN is negative, so a NaN is 3. A cubic
/// whose control points all sit inside passes through, or all outside on the current side is skipped.
/// Any other cubic is flattened at ClipWithRect's tolerance (`NormalizeTolerance(0)` = 0.25) by the
/// float `CBezierFlattener` and clipped as lines. The closing segment back to the start emits its
/// crossings only.
struct Strip {
    axis: usize,
    first: Option<[f32; 2]>,
    out: Vec<Seg>,
    p: [f32; 2],
    cp: u8,
    lo: f32,
    hi: f32,
}

impl Strip {
    fn run(start: [f32; 2], segs: &[Seg], axis: usize, lo: f32, hi: f32) -> Option<([f32; 2], Vec<Seg>)> {
        let mut st = Strip { axis, first: None, out: Vec::new(), p: start, cp: 0, lo, hi };
        st.cp = st.code(start);
        if st.cp == 0 {
            st.first = Some(start);
        }
        for seg in segs {
            match *seg {
                Seg::Line(q) => st.line(q, false),
                Seg::Cubic(p1, p2, p3) => st.cubic(p1, p2, p3),
            }
        }
        if st.first.is_some() {
            st.line(start, true);
        }
        Some((st.first?, st.out))
    }

    fn code(&self, p: [f32; 2]) -> u8 {
        let v = p[self.axis];
        if v.is_nan() { 3 } else { (u8::from(self.hi - v < 0.0) << 1) | u8::from(v - self.lo < 0.0) }
    }

    fn emit(&mut self, q: [f32; 2]) {
        if self.first.is_none() {
            self.first = Some(q);
        } else {
            self.out.push(Seg::Line(q));
        }
    }

    fn cross(&self, c: u8, q: [f32; 2]) -> [f32; 2] {
        let (a, p) = (self.axis, self.p);
        let (b, o, d) = (if c == 1 { self.lo } else { self.hi }, 1 - a, p[a] - q[a]);
        let mut r = p;
        r[a] = b;
        if d != 0.0 {
            r[o] = ((b - q[a]) * p[o] - (b - p[a]) * q[o]) / d;
        }
        r
    }

    fn line(&mut self, q: [f32; 2], close: bool) {
        let (cp, cq) = (self.cp, self.code(q));
        if cp != cq {
            if cp != 0 {
                self.emit(self.cross(cp, q));
            }
            if cq != 0 {
                self.emit(self.cross(cq, q));
            }
        }
        if cq == 0 && !close {
            self.emit(q);
        }
        (self.p, self.cp) = (q, cq);
    }

    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn cubic(&mut self, p1: [f32; 2], p2: [f32; 2], p3: [f32; 2]) {
        let c = [p1, p2, p3].map(|q| q[self.axis]);
        let (lo, hi) = (self.lo, self.hi);
        let pass_or_skip = match self.cp {
            1 => c.iter().all(|&v| !(v >= lo)),
            2 => c.iter().all(|&v| !(hi >= v)),
            _ => c.iter().all(|&v| !(lo > v || v > hi)),
        };
        if pass_or_skip {
            if !matches!(self.cp, 1 | 2) {
                self.first.get_or_insert(self.p);
                self.out.push(Seg::Cubic(p1, p2, p3));
            }
            self.p = p3; // the outcode stays: p3 lies on the same side
            return;
        }
        let mut pts = Vec::new();
        crate::geometry::flatten::flatten_cubic(self.p, p1, p2, p3, 0.25, &mut pts);
        for q in pts {
            self.line(q, false);
        }
    }
}

/// True when every edge of the (finite) polygon is axis-parallel — an axis-aligned rectangle/box.
/// Such a fill has no slanted edges, so the COPY trapezoid walk adds no fringe and its −0.5
/// FillGeometry pixel-centre shift is wrong (native fills an axis rect via its rect special case, not
/// the shifted geometry grid); those fills stay on the 8×8 area path, which is byte-exact (layers-07,
/// and the fractional-rect COPY fills fuzz 8876/8982 that the walk regressed).
/// One axis-aligned rectangle (four distinct corners), which a COPY fill takes through native's rect
/// special case. A widened stroke strip-clipped to the safe box can also run along the axes, but it is a
/// longer contour, and native fills it through the edge walk.
pub(crate) fn is_single_axis_rect(figures: &[Vec<[f32; 2]>]) -> bool {
    let [f] = figures else { return false };
    let mut c = f.clone();
    c.dedup();
    if c.len() > 1 && c.first() == c.last() {
        c.pop();
    }
    c.len() == 4 && is_axis_aligned(f)
}

pub(crate) fn is_axis_aligned(poly: &[[f32; 2]]) -> bool {
    let n = poly.len();
    if n < 3 || !poly.iter().all(|p| p[0].is_finite() && p[1].is_finite()) {
        return false;
    }
    (0..n).all(|i| {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        (a[0] - b[0]).abs() < 1e-4 || (a[1] - b[1]).abs() < 1e-4
    })
}

/// A single-contour polygon all of whose turns go the same way (and whose points are finite): native's
/// trapezoid rasteriser assumes one left + one right edge per scanline, which convexity guarantees.
/// Near-coincident vertices (curve-flattening jitter at an ellipse's axis extrema, and the explicit
/// closing point) are dropped first — they create spurious sub-pixel backward turns on a shape that is
/// genuinely convex. A real concavity is a vertex far from its neighbours, so it survives the dedup.
pub(crate) fn is_convex(poly: &[[f32; 2]]) -> bool {
    if !poly.iter().all(|p| p[0].is_finite() && p[1].is_finite()) {
        return false;
    }
    // Squared distance (< 0.1²), not hypot — hypot comes from the platform libm and isn't bit-identical
    // across platforms (clippy disallows it); squared distance is exact and sufficient for the dedup.
    let near = |a: [f32; 2], b: [f32; 2]| {
        let (dx, dy) = (a[0] - b[0], a[1] - b[1]);
        dx * dx + dy * dy < 0.01
    };
    let mut pts: Vec<[f32; 2]> = Vec::with_capacity(poly.len());
    for &p in poly {
        if pts.last().map(|&q| near(q, p)).unwrap_or(false) {
            continue;
        }
        pts.push(p);
    }
    while pts.len() > 1 && near(pts[0], pts[pts.len() - 1]) {
        pts.pop();
    }
    let n = pts.len();
    if n < 3 {
        return false;
    }
    let mut sign = 0i8;
    for i in 0..n {
        let (a, b, c) = (pts[i], pts[(i + 1) % n], pts[(i + 2) % n]);
        let cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
        if cross.abs() < 1e-4 {
            continue; // collinear vertices don't change the turn direction
        }
        let s = if cross > 0.0 { 1 } else { -1 };
        if sign == 0 {
            sign = s;
        } else if s != sign {
            return false;
        }
    }
    sign != 0 // reject fully-degenerate (collinear) contours
}

/// Composite `src` (premultiplied BGRA) at coverage `cov ∈ 0..=64` over `dst` under `blend`.
///
/// Coverage is quantised to 8 bits (`cova = round(cov*255/64)`) then normalised — the
/// managed/native rule confirmed against the oracle. Source-over
/// runs single-stage in f32 with one round-to-nearest-even; min/add/max use the coverage-weighted
/// 8-bit premultiplied source.
pub(crate) fn composite(dst: [u8; 4], src: [u8; 4], cov: u32, blend: Blend) -> [u8; 4] {
    let cova = (cov.min(64) as f32 * 255.0 / 64.0).round_ties_even();
    if blend == Blend::SourceOver {
        // WARP's fill kernel weights each 8-bit premultiplied channel by the 8-bit coverage with its
        // UNORM product `(t + (t >> 8) + 128) >> 8`, t = colour·coverage (as the brush fringe does),
        // then source-overs that.
        let k = cova as i32;
        let cs8 = src.map(|s| {
            let t = s as i32 * k;
            ((t + (t >> 8) + 128) >> 8) as u8
        });
        return blend_cov(dst, cs8, 1.0, blend);
    }
    blend_cov(dst, src, cova / 255.0, blend)
}

/// Source-over of `src` at `cov`/64 with `dst·(1 − a)` rounded once: the coverage-weighted source
/// quantised to 8 bits, then `re(cs8 + dst·(1 − cs8.a/255))`.
fn over_rounded(dst: [u8; 4], src: [u8; 4], cov: u32) -> [u8; 4] {
    let c = (cov.min(64) as f32 * 255.0 / 64.0).round_ties_even() / 255.0;
    let re = |v: f32| v.round_ties_even().clamp(0.0, 255.0) as u8;
    let cs8 = src.map(|s| re(s as f32 * c));
    let sa = cs8[3] as f32 / 255.0;
    core::array::from_fn(|i| re(cs8[i] as f32 + dst[i] as f32 * (1.0 - sa)))
}

/// [`composite`] at coverage `c` (`cov8/255`): the primitive blend of an 8-bit premultiplied source.
pub(crate) fn blend_cov(dst: [u8; 4], src: [u8; 4], c: f32, blend: Blend) -> [u8; 4] {
    let cs = src.map(|s| s as f32 * c);
    let re = |v: f32| v.round_ties_even().clamp(0.0, 255.0) as u8;
    let add8 = |a: u8, b: u8| (a as u16 + b as u16).min(255) as u8;
    match blend {
        Blend::SourceOver => {
            // WARP takes the coverage-weighted premultiplied source as an 8-bit value (quantised
            // first, like Min/Add/Max below and the bitmap/triangle UNORM8 path),
            // then does source-over on 8-bit premul in its output merger: `out = cs8 + dst·(255 −
            // cs8.a)/255`, the product divided as `(t + (t>>8) + 128) >> 8` (as brush text, layers and
            // bitmaps). Quantising `cs` before the blend (not a single f32 stage) is what makes layered
            // translucent curve fills — overlapping ellipses (canvas-comp-*) — byte-exact.
            let cs8 = cs.map(re);
            let ia = 255 - cs8[3] as i32;
            core::array::from_fn(|i| {
                let t = dst[i] as i32 * ia;
                (cs8[i] as i32 + ((t + (t >> 8) + 128) >> 8)).clamp(0, 255) as u8
            })
        }
        // COPY lerps dst→src by coverage, taking the coverage-weighted premultiplied source as an
        // 8-bit value FIRST (`re(src·c)`, like SourceOver/Min/Add/Max — not an unquantised f32), then
        // `out = cs8 + dst·(1−c)`. The pre-quantise is what makes a fractional-coverage COPY edge
        // byte-exact (fillcopy-half/-opq: an un-quantised lerp rounds R/G one step high). Aliased
        // (cov∈{0,64}) is unaffected: cs8=src at c=1, 0 at c=0.
        Blend::Copy => {
            let cs8 = cs.map(re);
            core::array::from_fn(|i| re(cs8[i] as f32 + dst[i] as f32 * (1.0 - c)))
        }
        // MIN takes the source over WHITE: `min(dst, src + 1 − src.a)`. Native forms it on the float
        // source; this path only has the 8-bit premultiplied one (ponytail: ±1 where c·a·255 sits near
        // a half; carry the float source here to close it).
        Blend::Min => core::array::from_fn(|i| re(cs[i] + (255.0 - cs[3])).min(dst[i])),
        Blend::Add => core::array::from_fn(|i| add8(re(cs[i]), dst[i])),
        Blend::Max => core::array::from_fn(|i| re(cs[i]).max(dst[i])),
    }
}

/// Rasterize an axis-aligned device rectangle's coverage, calling `f(x, y, samples∈1..=64)` once per
/// covered pixel. Format-independent — the BGRA8 [`Target`] and the float target (`crate::float`)
/// share it, so the 8×8 coverage rule is pinned in one place. Empty/inverted/non-finite rects and
/// the `Aliased` centre-sample top-left test reproduce the oracle exactly.
pub(crate) fn rasterize_rect(rect: [f32; 4], aa: Aa, w: usize, h: usize, mut f: impl FnMut(usize, usize, u32)) {
    let [l, t, r, b] = rect.map(|v| v as f64);
    if !(r > l && b > t) || ![l, t, r, b].iter().all(|v| v.is_finite()) {
        return;
    }
    if aa == Aa::Aliased {
        // WARP snaps each edge to the 24.8 device grid (round-half-up) before the centre/top-left
        // test; an edge landing within 1/256 above a pixel centre thus covers that pixel (clal-d).
        // `v as f32` matches the device coords' single-precision origin, as the AA `e8` path does.
        let snap = |v: f64| ((v as f32 * 256.0 + 0.5).floor() as f64) / 256.0;
        let (l, t, r, b) = (snap(l), snap(t), snap(r), snap(b));
        for y in 0..h {
            let cy = y as f64 + 0.5;
            if !(t <= cy && cy < b) {
                continue;
            }
            for x in 0..w {
                let cx = x as f64 + 0.5;
                if l <= cx && cx < r {
                    f(x, y, 64);
                }
            }
        }
        return;
    }
    // Per-primitive: WARP's AA rect rasteriser: each edge snaps to
    // 1/256 to nearest, then goes to eighths as `(8·q + 127) >> 8`, and a pixel's coverage is its
    // x-eighths × y-eighths of 64 (r5 5887: x 23.6228 covers 3/8, y 56.6894 3/8; r3 2957: y 27.1960
    // covers 6/8; a .6905 edge covers 2/8, not truncation's 3/8: q8829-*, bre-*, r9 14672).
    let e8 = |v: f64| (8 * (v as f32 * 256.0).round_ties_even() as i32 as i64 + 127) >> 8;
    let (l8, t8, r8, b8) = (e8(l), e8(t), e8(r), e8(b));
    let span = |lo: i64, hi: i64, n: usize| ((lo >> 3).clamp(0, n as i64) as usize, ((hi + 7) >> 3).clamp(0, n as i64) as usize);
    let ((x0, x1), (y0, y1)) = (span(l8, r8, w), span(t8, b8, h));
    let cover = |p: usize, lo: i64, hi: i64| ((p as i64 * 8 + 8).min(hi) - (p as i64 * 8).max(lo)).max(0) as u32;
    for x in x0..x1 {
        let hc = cover(x, l8, r8);
        if hc == 0 {
            continue;
        }
        for y in y0..y1 {
            let vc = cover(y, t8, b8);
            if vc > 0 {
                f(x, y, vc * hc);
            }
        }
    }
}

/// Rasterize flattened device-space line `edges` (each part of a closed contour) under `fill`,
/// calling `f(x, y, samples∈1..=64)` per covered pixel. `PerPrimitive` runs native `CHwRasterizer`'s
/// fixed-point 8×8 scan conversion; `Aliased` is the centre-sample top-left test. Shared by the
/// BGRA8 and float targets so there is one copy of the scan converter.
pub(crate) fn rasterize_edges(edges: &[[f32; 4]], fill: FillMode, aa: Aa, w: usize, h: usize, mut f: impl FnMut(usize, usize, u32)) {
    // A geometry with any non-finite device vertex draws nothing, matching native: WARP realizes the
    // path in f32 device space, so a DIP coordinate that overflows f32 under the world/DPI transform
    // (e.g. FLT_MAX·dpi → ±inf) — or a literal inf/NaN point — makes the whole FillGeometry empty, not
    // a clamped polygon. Verified on the VM: a huge/inf vertex anywhere voids the entire fill (fuzz
    // seeds 20015/20022 draw nothing natively; a finite 1.4e8-device edge still fills). One non-finite
    // vertex voids the contour it is in and every other contour of the same fill.
    if edges.iter().any(|e| !e.iter().all(|v| v.is_finite())) {
        return;
    }
    if aa == Aa::Aliased {
        // The point-in-path test is O(w·h·edges), so drop edges that cross no scanline centre in
        // [0,h]: `inside`'s `(y0<=cy) != (y1<=cy)` is false for every in-target `cy` when both ends sit
        // on the same side of the band, so the edge never affects any in-target pixel — byte-identical,
        // but it stops a fill whose geometry extends far off-target (fuzz 1258: a huge nib ellipse
        // flattened to ~573k edges, almost all outside the 128×96 viewport) from spinning billions of
        // no-op ray tests. Bounds the per-pixel edge count to those actually crossing the viewport rows.
        // Same aliased edge rule as `rasterize_rect`: snap every vertex to the 24.8 device grid
        // (round-half-up) before the centre/top-left point-in-path test, so a boundary within 1/256 of
        // a pixel centre falls on native's side of it (clal-d geometry fills, not just axis rects).
        // (General non-rect aliased fills are pre-snapped to native's 1/64→1/16 grid upstream in
        // `fill_geometry`/`draw_geometry`; this 1/256 snap is then a no-op on them.)
        let snap = |v: f32| ((v * 256.0 + 0.5).floor() as f64) / 256.0;
        let e: Vec<[f64; 4]> = edges
            .iter()
            .map(|s| s.map(snap))
            .filter(|&[_, y0, _, y1]| {
                y0 != y1 && [y0, y1].iter().all(|v| v.is_finite()) && y0.min(y1) < h as f64 && y0.max(y1) > 0.0
            })
            .collect();
        if e.is_empty() {
            return;
        }
        for y in 0..h {
            let cy = y as f64 + 0.5;
            for x in 0..w {
                if inside(&e, fill, x as f64 + 0.5, cy) {
                    f(x, y, 64);
                }
            }
        }
        return;
    }
    let mut es: Vec<Edge> = edges.iter().filter_map(|&s| Edge::new(s)).collect();
    if es.is_empty() {
        return;
    }
    es.sort_by_key(|e| e.y_top);
    let y_end = es.iter().map(|e| e.y_bot).max().unwrap().min(h as i32 * 8);
    let row0 = (es[0].y_top >> 3).max(0);
    let row1 = ((y_end + 7) >> 3).min(h as i32);
    let mut cov = vec![0i32; w];
    let mut active: Vec<usize> = Vec::new();
    let mut next = 0usize;
    let mut xs: Vec<(i32, i32)> = Vec::new();
    for row in row0..row1 {
        cov.iter_mut().for_each(|c| *c = 0);
        for sub in 0..8 {
            let sl = row * 8 + sub;
            while next < es.len() && es[next].y_top <= sl {
                // An edge whose span begins above the first processed sub-scanline — geometry
                // extending above the target's top row, so `row0` was clamped up to 0 — must have
                // its DDA advanced to `sl` before use, or its x still reflects the off-target top
                // vertex. Native's edge setup does the same clip-to-top,
                // advancing x/err to the clip top when `clipTop > edgeTop`; stepping one
                // sub-scanline at a time is the closed form native computes. A no-op when y_top==sl.
                let ytop = es[next].y_top;
                let n = sl - ytop;
                // Byte-exact stepping for geometry clipped a little above the target (the common case);
                // a closed-form jump only for a far-off-screen top (>512 px), which would otherwise be
                // millions of steps. `advance` reproduces `step` exactly, so the threshold is a pure
                // perf split, not a rendering change.
                if n > 4096 {
                    es[next].advance(n);
                } else {
                    for _ in 0..n {
                        es[next].step();
                    }
                }
                active.push(next);
                next += 1;
            }
            active.retain(|&i| es[i].y_bot > sl);
            xs.clear();
            for &i in &active {
                xs.push((es[i].x, es[i].wind));
            }
            xs.sort_by_key(|t| t.0);
            let mut wind = 0;
            for j in 0..xs.len().saturating_sub(1) {
                wind += xs[j].1;
                let fill_span = match fill {
                    FillMode::Winding => wind != 0,
                    FillMode::Alternate => j % 2 == 0,
                };
                if fill_span {
                    add_interval(&mut cov, xs[j].0, xs[j + 1].0);
                }
            }
            for &i in &active {
                es[i].step();
            }
        }
        for x in 0..w {
            if cov[x] > 0 {
                f(x, row as usize, cov[x] as u32);
            }
        }
    }
}

/// One active edge in native `CHwRasterizer`'s fixed-point DDA form. `x`/`err` are the current
/// column in eighth-of-a-pixel device units and the sub-sample DDA remainder; `dx`/`err_up`/
/// `err_down` step them one sub-scanline at a time; `[y_top, y_bot)` is the sub-scanline span
/// (8 per pixel row). Ported from native's edge setup and active-edge-list walk.
#[derive(Clone)]
struct Edge {
    x: i32,
    dx: i32,
    err: i32,
    err_up: i32,
    err_down: i32,
    y_top: i32,
    y_bot: i32,
    wind: i32,
}

impl Edge {
    /// Build an edge from a device-space segment, or `None` if horizontal, degenerate or
    /// non-finite. Device coords go to 28.4 as `ceil(16·(v−0.5))` (the −0.5 pixel-centre bias
    /// folded in), clamp to ±2^23 sub-units, then ×8+56 into the 8× supersampling grid.
    fn new([x0f, y0f, x1f, y1f]: [f32; 4]) -> Option<Edge> {
        if ![x0f, y0f, x1f, y1f].iter().all(|v| v.is_finite()) {
            return None;
        }
        // The geometry edge-list sink converts with cvttss2si, so an out-of-i32-range
        // 28.4 value becomes i32::MIN (±1), which the edge setup then clamps to
        // ±2^23 like any other point: an overflow lands at −2^23 whatever its sign. [`clip_huge`] keeps
        // fills inside ±524287 px, so only direct edge callers can still overflow here.
        let t = |v: f32| -> i32 {
            let c = (16.0 * v - 8.0).ceil();
            let c = if (-2147483648.0..2147483648.0).contains(&c) { c as i32 } else { i32::MIN };
            8 * c.clamp(-8388608, 8388608) + 56
        };
        let (x0, y0, x1, y1) = (t(x0f), t(y0f), t(x1f), t(y1f));
        let (dy, wind, dx, y_top_f, y_bot_f, x_top, y_start) = if y1 - y0 < 0 {
            (y0 - y1, -1, x0 - x1, y1 + 15, y0 + 15, x1, y1)
        } else {
            (y1 - y0, 1, x1 - x0, y0 + 15, y1 + 15, x0, y0)
        };
        let (y_top, y_bot) = (y_top_f >> 4, y_bot_f >> 4);
        if y_bot <= y_top {
            return None;
        }
        let (dx_step, err_up) = if dx < 0 {
            let adx = -dx;
            if adx < dy {
                (-1, dy + dx)
            } else {
                let (mut q, mut r) = (-(adx / dy), adx % dy);
                if r > 0 {
                    q -= 1;
                    // Two's-complement wrap (as native's integer rasterizer does, see the `t` comment):
                    // `dy` can be i32::MIN when an off-screen vertex's coordinate negation overflowed, and
                    // `dy - r` would then panic in a debug / panic=abort build (7000-7999 rerun 7203: a
                    // FillGeometry of a figure with a -1e9 vertex). A crash, not an E_INVALIDARG, in a
                    // panic=abort drop-in.
                    r = dy.wrapping_sub(r);
                }
                (q, r)
            }
        } else if dx >= dy {
            (dx / dy, dx % dy)
        } else {
            (0, dx)
        };
        // Advance the DDA across the first vertex's fractional sub-scanline to snap onto the grid.
        let mut err = -1i32;
        let mut x = x_top;
        let frac = y_start & 0xF;
        if frac != 0 {
            for _ in 0..(16 - frac) {
                let e_new = err.saturating_add(err_up);
                let newx = dx_step.saturating_add(x);
                err = e_new;
                if e_new >= 0 {
                    err = err.saturating_sub(dy);
                    x = newx.saturating_add(1);
                } else {
                    x = newx;
                }
            }
        }
        if x & 0xF != 0 {
            err = err.saturating_sub(dy.saturating_mul(16 - (x & 0xF)));
            x = x.saturating_add(15);
        }
        Some(Edge { x: x >> 4, dx: dx_step, err: err >> 4, err_up, err_down: dy, y_top, y_bot, wind })
    }

    /// Advance the DDA by `n` sub-scanlines in closed form, identical to calling [`step`] `n` times
    /// (Bresenham: the err invariant `err ∈ [-err_down, 0)`, `err_up ∈ [0, err_down)` gives at most one
    /// wrap per step, so over `n` steps `wraps = (err + n·err_up).div_euclid(err_down) + 1` and
    /// `x += n·dx + wraps`). Lets the top-skip jump a far-off-screen edge to the first visible
    /// sub-scanline without millions of steps (fuzz seed 1764 was a 120 s hang). i64 math + saturating
    /// x so an extreme off-screen edge can't overflow; its saturated x is clipped by the span fill.
    fn advance(&mut self, n: i32) {
        if n <= 0 {
            return;
        }
        let (n, edown) = (n as i64, self.err_down as i64);
        let s = self.err as i64 + n * self.err_up as i64;
        let wraps = s.div_euclid(edown) + 1;
        self.err = (s.rem_euclid(edown) - edown) as i32;
        let x = self.x as i64 + n * self.dx as i64 + wraps;
        self.x = x.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }

    /// Advance X and the DDA remainder by one sub-scanline. Saturating: an edge from extreme
    /// (off-screen, ±524288 px clamp) geometry can be caught up over millions of sub-scanlines
    /// (rasterize_edges' top-skip), accumulating `x`/`err` past i32 — a panic (fuzz seed 455). The
    /// saturated x stays off-screen and is clipped by the span fill, so visible pixels are unchanged;
    /// a legit in-target edge never approaches the limit.
    fn step(&mut self) {
        let e_new = self.err.saturating_add(self.err_up);
        self.x = self.x.saturating_add(self.dx);
        self.err = e_new;
        if e_new >= 0 {
            self.err = self.err.saturating_sub(self.err_down);
            self.x = self.x.saturating_add(1);
        }
    }
}

/// Add one filled span `[l, r)` (eighth-of-a-pixel device units) of a sub-scanline into the
/// per-pixel coverage row: the left pixel gets `8−(l&7)` samples, interior pixels 8, the right
/// pixel `r&7` (a single-pixel span gets `r−l`). This is native `CCoverageBuffer::AddInterval`.
fn add_interval(cov: &mut [i32], l: i32, r: i32) {
    if r <= l {
        return;
    }
    let w = cov.len() as i32;
    let (lpx, rpx) = (l >> 3, r >> 3);
    let mut put = |px: i32, v: i32| {
        if px >= 0 && px < w {
            cov[px as usize] += v;
        }
    };
    if lpx == rpx {
        put(lpx, r - l);
    } else {
        put(lpx, 8 - (l & 7));
        // Clamp the interior sweep to the buffer: `put` already no-ops an out-of-range pixel, so this
        // is byte-identical output, but it stops a span from a far-off-screen geometry (device x up to
        // the ±2^23 edge clamp, e.g. a 1e9-radius ellipse — fuzz seed 7798) from spinning ~16M no-op
        // iterations per sub-scanline, which wedged the fill for seconds (a wasm panic=abort hang).
        for px in (lpx + 1).max(0)..rpx.min(w) {
            put(px, 8);
        }
        put(rpx, r & 7);
    }
}

/// Native `CHwRasterizer`'s trapezoidal AA (WPF `hwrasterizer.cpp`/`hwvertexbuffer.cpp`), the coverage
/// d2d1 bakes for geometry that leaves the safe box (the huge-coordinate fill route). Per pixel row it
/// pairs the active edges (left/right of each wound span) and, where their AA expand regions don't
/// overlap, outputs a TRAPEZOID: the coverage ramps linearly over a HORIZONTAL half-width
/// `delta = 0.5 + 0.5·|dx/dy|` on each edge (outer = 0, inner = 1), interior = 1 — so a slope-1 diagonal
/// reads 0.5 at the pixel centre (`AddTrapezoidStandard`). Where the expand regions DO overlap (edges
/// within `leftDelta+rightDelta`, e.g. a triangle's apex corner) it falls back to the 8×8 complex-scan
/// subsample (`rasterize_edges`, = `nCoverage/64`). This reproduces native's 32-straight/36-apex split
/// that the pure subsample (uniform 36) and the normal-displacement feather both miss.
pub(crate) fn trapezoid_coverage(figures: &[Vec<[f32; 2]>], w: usize, h: usize, aa: Aa) -> Vec<u8> {
    // Complex-scan fallback coverage: the 8×8 subsample, identical to the edge-DDA path.
    // d2d1 bakes the coverage as a Q24 attribute whose alpha is the high byte (`cov >> 16`), i.e. the
    // nCoverage/64 fraction times 256 (clamped), NOT the ×255 the WARP edge-DDA path uses: the complex-scan
    // apex vertex 0x900000 >> 16 = 144, not 143. So a full span (64) → 255, and nCoverage → (nCoverage·4).
    let edges = figure_edges(figures);
    let mut complex = vec![0u8; w * h];
    rasterize_edges(&edges, FillMode::Winding, aa, w, h, |x, y, c| {
        complex[y * w + x] = (c.min(64) * 4).min(255) as u8;
    });
    // Trapezoid edges in device space (non-horizontal), oriented top→bottom with a winding sign.
    struct TEdge { x0: f32, y0: f32, x1: f32, y1: f32, wind: i32 }
    let mut tedges: Vec<TEdge> = Vec::new();
    for f in figures {
        let n = f.len();
        if n < 2 { continue; }
        for i in 0..n {
            let (a, b) = (f[i], f[(i + 1) % n]);
            if !(a[1].is_finite() && a[0].is_finite() && b[1].is_finite() && b[0].is_finite()) || a[1] == b[1] { continue; }
            let (x0, y0, x1, y1, wind) = if a[1] < b[1] { (a[0], a[1], b[0], b[1], 1) } else { (b[0], b[1], a[0], a[1], -1) };
            tedges.push(TEdge { x0, y0, x1, y1, wind });
        }
    }
    let mut cov = vec![0u8; w * h];
    let mut active: Vec<(f32, f32, i32)> = Vec::new(); // (x_at_cy, inv_slope, wind)
    for py in 0..h {
        let cy = py as f32 + 0.5;
        active.clear();
        for e in &tedges {
            if cy >= e.y0 && cy < e.y1 {
                let inv = (e.x1 - e.x0) / (e.y1 - e.y0);
                let x = e.x0 + (cy - e.y0) * inv;
                active.push((x, inv, e.wind));
            }
        }
        active.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut wind = 0;
        for i in 0..active.len().saturating_sub(1) {
            wind += active[i].2;
            if wind == 0 { continue; }
            let (xl, invl, _) = active[i];
            let (xr, invr, _) = active[i + 1];
            let (dl, dr) = (0.5 + 0.5 * invl.abs(), 0.5 + 0.5 * invr.abs());
            if xl + dl > xr - dr {
                // Expand regions overlap → complex scan (the 8×8 subsample) over the span's pixels.
                let (p0, p1) = ((xl.floor().max(0.0) as usize).min(w), ((xr.ceil().max(0.0)) as usize).min(w));
                for px in p0..p1 { cov[py * w + px] = complex[py * w + px]; }
            } else {
                // Simple trapezoid: horizontal AA ramp of half-width delta on each edge.
                let (p0, p1) = (((xl - dl).floor().max(0.0)) as usize, (((xr + dr).ceil().max(0.0)) as usize).min(w));
                for px in p0.min(w)..p1 {
                    let cx = px as f32 + 0.5;
                    let cl = ((cx - (xl - dl)) / (2.0 * dl)).clamp(0.0, 1.0);
                    let cr = (((xr + dr) - cx) / (2.0 * dr)).clamp(0.0, 1.0);
                    cov[py * w + px] = ((cl.min(cr) * 256.0).round_ties_even() as u32).min(255) as u8;
                }
            }
        }
    }
    cov
}

/// Point-in-path test for aliased fills: ray cast with half-open edges in y and a strict
/// right comparison, which reproduces WARP's centre-sample top-left rule on the oracle.
fn inside(edges: &[[f64; 4]], fill: FillMode, cx: f64, cy: f64) -> bool {
    let mut wind = 0i32;
    let mut cnt = 0u32;
    for &[x0, y0, x1, y1] in edges {
        if (y0 <= cy) != (y1 <= cy) {
            let xi = x0 + (cy - y0) / (y1 - y0) * (x1 - x0);
            if xi > cx {
                cnt += 1;
                wind += if y1 > y0 { 1 } else { -1 };
            }
        }
    }
    match fill {
        FillMode::Winding => wind != 0,
        FillMode::Alternate => cnt % 2 == 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The strip clip's f32 crossings at `w·h ≥ FLT_MAX` (w_bm dbl/dbt): +inf → FAR, NaN → nothing.
    #[test]
    fn clip_huge_overflowed_crossings() {
        let left = [vec![[-1e38, 0.0], [20.0, 0.0], [20.0, 12.0], [-1e38, 12.0]]];
        assert_eq!(&*clip_huge(&left).unwrap(), &[vec![[-524287.0, 0.0], [20.0, 0.0], [20.0, 12.0], FAR]]);
        assert!(clip_huge(&[vec![[0.0, -1e38], [20.0, -1e38], [20.0, 12.0], [0.0, 12.0]]]).is_none());
    }

    // `Edge::advance(n)` must be byte-identical to calling `step()` n times (the closed-form top-skip).
    #[test]
    fn advance_matches_stepping() {
        // A spread of edge slopes/spans, including a far-off-screen top that needs the closed form.
        let cases = [[3.0, -5.0, 61.0, 90.0], [1.0, -100.0, 40.0, 12.0], [90.0, -4000.0, 5.0, 60.0], [10.0, 2.0, 11.0, 300.0], [-200.0, -50.0, 400.0, 70.0]];
        for seg in cases {
            let Some(base) = Edge::new(seg) else { continue };
            for n in [0, 1, 7, 50, 999, 50_000] {
                let (mut a, mut b) = (base.clone(), base.clone());
                a.advance(n);
                for _ in 0..n {
                    b.step();
                }
                assert_eq!((a.x, a.err), (b.x, b.err), "advance({n}) != step×{n} for {seg:?}");
            }
        }
    }

    // A contour with any non-finite device vertex produces zero coverage (native empties the whole
    // FillGeometry — fuzz seeds 20015/20022). A finite square over the same area still covers.
    #[test]
    fn non_finite_vertex_voids_fill() {
        let square = [[2.0, 2.0, 6.0, 2.0], [6.0, 2.0, 6.0, 6.0], [6.0, 6.0, 2.0, 6.0], [2.0, 6.0, 2.0, 2.0]];
        for aa in [Aa::Aliased, Aa::PerPrimitive] {
            let mut covered = 0u32;
            rasterize_edges(&square, FillMode::Winding, aa, 8, 8, |_, _, _| covered += 1);
            assert!(covered > 0, "finite square must cover something");

            // Push the right edge to +inf (e.g. FLT_MAX·dpi overflow): whole fill voids.
            let mut inf = square;
            inf[0][2] = f32::INFINITY;
            inf[1][0] = f32::INFINITY;
            let mut n = 0u32;
            rasterize_edges(&inf, FillMode::Winding, aa, 8, 8, |_, _, _| n += 1);
            assert_eq!(n, 0, "non-finite vertex must void the fill");
        }
    }

    // `blend_px`'s full-coverage opaque source-over fast path writes `src` directly instead of calling
    // `composite`. Prove that shortcut is byte-identical for EVERY UNORM8 input it triggers on — cov 64
    // and source alpha 255 — not just the ones the corpora happen to exercise: `composite` must return
    // `src` for all destinations. (This is the u8 `Target`; the float `FloatTarget` has its own
    // `blend_px`/`composite_f` and is untouched, so its NaN / ±inf / −0 cases do not arise here.)
    #[test]
    fn opaque_full_source_over_equals_src() {
        // Per channel: with alpha 255 the `dst·(1 − sa)` term is exactly 0 and `re(src)` is `src`, so
        // the result is independent of dst. Exhaustive over (src channel, dst) × every value.
        for s in 0u16..=255 {
            for d in 0u16..=255 {
                let (src, dst) = ([s as u8, s as u8, s as u8, 255], [d as u8; 4]);
                assert_eq!(composite(dst, src, 64, Blend::SourceOver), src, "s={s} d={d}");
            }
        }
        // Mixed channels against extreme destinations (per-channel independence).
        for src in [[0, 128, 255, 255], [1, 2, 254, 255], [37, 200, 9, 255]] {
            for dst in [[0, 0, 0, 0], [255, 255, 255, 255], [128, 64, 200, 17]] {
                assert_eq!(composite(dst, src, 64, Blend::SourceOver), src, "src={src:?} dst={dst:?}");
            }
        }
    }

    // A FINITE but extreme vertex (e.g. -1e9) drives an off-screen edge whose 28.4 coord clamps to
    // i32::MIN; the DDA setup must wrap in two's-complement, not overflow-panic (7000-7999 rerun 7203 —
    // a panic crashes a panic=abort drop-in, not just latches E_INVALIDARG).
    #[test]
    fn finite_huge_vertex_does_not_panic() {
        for aa in [Aa::Aliased, Aa::PerPrimitive] {
            let tri = [[46.35, -1e9, 36.15, 29.1], [36.15, 29.1, 0.0, 0.0], [0.0, 0.0, 46.35, -1e9]];
            let mut n = 0u32;
            rasterize_edges(&tri, FillMode::Winding, aa, 8, 8, |_, _, _| n += 1); // must not panic
            let _ = n;
        }
    }
}
