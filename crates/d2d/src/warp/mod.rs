//! WARP's Direct3D 11 general triangle pipeline — the path gradient meshes (and later ink / custom
//! vertex shaders) draw through on the WARP software device, distinct from Direct2D's own
//! coverage rasterizer in [`crate::pixels`]. Confirmed byte-for-byte against the gradient-mesh
//! oracle.
//!
//! What a triangle does here:
//! - **Vertex snapping**: device positions snap to 1/256-subpixel fixed point for the coverage test.
//! - **Membership**: one sample at the pixel centre `(x+0.5, y+0.5)` with the D3D11 top-left fill
//!   rule (a sample exactly on a top or left edge is inside), i.e. aliased — gradient meshes carry
//!   no edge anti-aliasing, matching the hard patch boundaries in the oracle.
//! - **Attributes** (colour) interpolate as a *screen-space plane*: value at the origin plus per-pixel
//!   `d/dx`, `d/dy` gradients taken relative to the 3rd (provoking) vertex, all in f32. This exact
//!   form is what makes the oracle byte-exact — a vertex-relative barycentric sum differs by up to
//!   1 LSB in the steep channel. w ≡ 1 here, so perspective-correct interpolation is affine.
//!
//! Output is premultiplied BGRA8, composited with the current primitive blend at full coverage.

pub mod plane;

use crate::pixels::Blend;
use d2d_types::contracts::PixelBuffer;

/// A triangle vertex: device-space position (world∘DPI already applied) and a straight RGBA colour.
#[derive(Clone, Copy)]
pub struct Vert {
    pub pos: [f32; 2],
    pub rgba: [f32; 4],
}

/// WARP's per-pixel linear-attribute interpolation for **one axis** of an axis-aligned quad draw,
/// handed to the texture sampler as a 16.16 fixed-point DDA. This is the texcoord counterpart of the
/// f32 Gouraud plane in [`fill_triangle`]: the rasterizer carries the attribute (source coordinate)
/// as `base + px·step` in 16.16, so no per-pixel float coordinate is re-rounded.
///
/// Given the source coordinate at the quad's two edges on this axis (`src0` at `dst0`, `src1` at
/// `dst1`, in texel units — centres at integer+0.5), the rasterizer:
/// - takes the gradient `ratio = (src1−src0)/(dst1−dst0)` and rounds it to 16.16 (`step`);
/// - rounds the attribute to 16.16 **once**, at the draw's first covered pixel `xref = floor(dst0)`
///   (`Ubase`), and reaches every other pixel by the integer recurrence `Ubase + (px−xref)·step`;
/// - folds the sampler's `−0x7f80` bias (half a texel minus half an 8-bit weight unit) into the base.
///
/// Returns `(base, step)` so the sampler reads `u16(px) = base + px·step`, with texel `u16 >> 16` and
/// 8-bit weight bits `[15:8]` (see [`crate::sampling::sample_fixed`]). Anchoring the single rounding
/// at the draw's own start pixel — not at `px = 0` — is what makes a fractional-dest `DrawBitmap`
/// (`warpjit-01`), an integer-dest magnified bitmap brush (`brushes-06`), and `FillOpacityMask`
/// (`v2-opacity-mask`) all byte-exact at once, which no fixed `px = 0` rounding bias could.
pub fn linear_attr_axis(src0: f32, src1: f32, dst0: f32, dst1: f32) -> (i32, i32) {
    let ratio = (src1 as f64 - src0 as f64) / (dst1 as f64 - dst0 as f64);
    let step = (ratio * 65536.0).round_ties_even() as i64;
    let xref = dst0.floor() as i64;
    let u_ref = src0 as f64 + (xref as f64 + 0.5 - dst0 as f64) * ratio;
    // Wrapping arithmetic: a degenerate/extreme DrawBitmap (zero-extent dest ⇒ huge `step`, or a
    // far-off-canvas dest ⇒ huge `xref`) makes `xref·step` exceed i64 — an overflow panic that aborts
    // across the COM ABI (a public DrawBitmap must never crash; such draws clip to an empty region
    // anyway). Non-overflowing inputs are unaffected, so every existing scene stays byte-identical;
    // on overflow the result is deterministic (same bits on every target — no browser fingerprint).
    let base = ((u_ref * 65536.0).round_ties_even() as i64)
        .wrapping_sub(xref.wrapping_mul(step))
        .wrapping_sub(0x7f80);
    (base as i32, step as i32)
}

/// Snap a device coordinate to 1/256-subpixel fixed point (D3D11's rasterizer grid). Clamped to ±2²⁹
/// subpixels (±2M device px, far beyond the 16384px max target, so a no-op for every real coordinate):
/// an extreme/NaN/inf vertex from a public draw otherwise snaps to `i64::MAX` and overflows the `edge`
/// area product — a panic that aborts across the COM ABI. The clamp bounds the edge/area math so it
/// can't overflow, deterministically on every target.
fn snap(v: f32) -> i64 {
    ((v as f64 * 256.0).round_ties_even() as i64).clamp(-(1 << 29), 1 << 29)
}

/// Signed area ×2 of the triangle A,B,C in the subpixel grid (`>0` ⇒ one winding, `<0` the other).
fn edge(ax: i64, ay: i64, bx: i64, by: i64, cx: i64, cy: i64) -> i64 {
    (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
}

/// D3D11 top-left test for the directed edge A→B of an area>0 (CCW, `e≥0` = inside) triangle: a
/// sample lying exactly on the edge is covered only when it is a left edge (interior to its right,
/// `by < ay`) or a top edge (horizontal, interior below, `bx > ax`). This assigns a cell's shared
/// diagonal to the lower-right triangle, matching the oracle on exactly-on-diagonal pixels.
fn top_left(ax: i64, ay: i64, bx: i64, by: i64) -> bool {
    by < ay || (ay == by && bx > ax)
}

/// Rasterize one aliased Gouraud triangle into `buf` (premultiplied BGRA8) under `blend`. Vertices
/// are supplied in index-buffer order; the 3rd is the plane origin for attribute interpolation.
///
/// `premul` selects what the `rgba` attribute carries, to match the native VS:
///   * `false` — straight colour with coverage in alpha; [`blend_px`] premultiplies `rgb·a`. Use for
///     draws whose source alpha is a separate coverage term (gradient mesh).
///   * `true` — the attribute is already `premultiplied_colour · coverage` (one float4, exactly the
///     realization/ink VS's output `o1`), so the Gouraud plane interpolates the *product*
///     `plane(c·cov)` and [`blend_px`] quantises it directly — never `c·plane(cov)`, which rounds a
///     LSB apart at a feather edge (`realize-02`).
pub fn fill_triangle(buf: &mut PixelBuffer, v: [Vert; 3], blend: Blend, premul: bool) {
    fill_triangle_cull(buf, v, blend, premul, Cull::None);
}

/// [`fill_triangle`] under a rasterizer cull mode: a triangle whose snapped setup area `cull` rejects
/// draws nothing.
pub(crate) fn fill_triangle_cull(buf: &mut PixelBuffer, v: [Vert; 3], blend: Blend, premul: bool, cull: Cull) {
    let (fw, fh) = (buf.desc.width as f32, buf.desc.height as f32);
    // A triangle crossing the target edge goes through WARP's frustum clip ([`clip_triangle`]); each fan
    // piece is set up keyed on its own first vertex (realize-stroke-normal-2's 120-dpi ellipse leaves
    // the target at x 96). One inside goes to setup whole, rotated by [`clip_rotation`].
    if v.iter().any(|t| t.pos[0] < 0.0 || t.pos[0] > fw || t.pos[1] < 0.0 || t.pos[1] > fh) {
        let tri: [[f32; 6]; 3] = v.map(|t| [t.pos[0], t.pos[1], t.rgba[0], t.rgba[1], t.rgba[2], t.rgba[3]]);
        let fan = match clip_triangle(tri, [0.0, 0.0, fw, fh]) {
            Clipped::Rejected => return,
            Clipped::Inside => vec![tri],
            Clipped::Fan(f) => f,
        };
        for s in fan {
            let vert = |t: [f32; 6]| Vert { pos: [t[0], t[1]], rgba: [t[2], t[3], t[4], t[5]] };
            fill_triangle_keyed(buf, s.map(vert), blend, premul, [0, 1, 2], cull);
        }
        return;
    }
    let o = clip_rotation([v[0].pos, v[1].pos, v[2].pos], fw, fh);
    // The position reaches setup through the VS's clip-space output and WARP's viewport transform, both
    // rounded in f32 (`x·(2/W) − 1`, then `·(W/2) + W/2`), as [`clip_triangle`]'s fan does: at W = 96 the
    // trip moves realize-stroke-normal-3's (74.1074, 29.5713) to subpixel x 18972, not 18971.
    let v = v.map(|t| Vert { pos: viewport_round_trip(t.pos, fw, fh), ..t });
    fill_triangle_keyed(buf, v, blend, premul, o, cull);
}

/// [`fill_triangle`]'s rasterizer: attribute planes from WARP's attribute-plane setup over the
/// 1/256-snapped positions, set up over the vertices in order `o` (the first is the reference), evaluated
/// as the JIT'd kernel does: `dx·x + (dy·y + c0)`, separate f32 mul/add. A feather pixel whose
/// `colour·cov` lands on a .5 quantisation tie takes native's side (realize-stroke-normal-1 (29,16) keys
/// on v0: B 118.50014; its mirror (66,47) on v2).
fn fill_triangle_keyed(buf: &mut PixelBuffer, v: [Vert; 3], blend: Blend, premul: bool, o: [usize; 3], cull: Cull) {
    let (w, h) = (buf.desc.width as i64, buf.desc.height as i64);
    let (p, area) = snap_tri([v[0].pos, v[1].pos, v[2].pos]);
    if area == 0 || !cull.keeps(area) {
        return;
    }
    let plane: [[f32; 3]; 4] = core::array::from_fn(|k| {
        plane::setup_triangle_attr(o.map(|i| p[i][0]), o.map(|i| p[i][1]), o.map(|i| v[i].rgba[k]))
    });
    cover_tri(p, area, w, h, |x, y| {
        let (pxf, pyf) = (x as f32 + 0.5, y as f32 + 0.5);
        let rgba: [f32; 4] = std::array::from_fn(|k| (pxf * plane[k][0] + (pyf * plane[k][1] + plane[k][2])).clamp(0.0, 1.0));
        blend_px(buf, x, y, rgba, blend, premul);
    });
}

/// A device position through the VS's clip-space output and WARP's viewport transform, as setup sees
/// it (`x·(2/W) − 1`, then `·(W/2) + W/2`; y flipped), each op rounded in f32.
pub(crate) fn viewport_round_trip(q: [f32; 2], w: f32, h: f32) -> [f32; 2] {
    let c = [q[0] * (2.0 / w) + -1.0, q[1] * (-2.0 / h) + 1.0];
    [c[0] * (w * 0.5) + w * 0.5, c[1] * (-h * 0.5) + h * 0.5]
}

/// WARP's triangle clipper rotates every triangle, keeping its winding, to start at its
/// lexicographically smallest clip-space vertex (`x`, then `y`; the vertex shader's
/// `o0 = pos·(2/W, −2/H) + (−1, 1)`), through a fixed rotation table, before triangle setup; the
/// attribute setup keys on that first vertex. Returns the vertex order.
pub(crate) fn clip_rotation(pos: [[f32; 2]; 3], w: f32, h: f32) -> [usize; 3] {
    let (sx, sy) = (2.0 / w, -2.0 / h);
    let c = pos.map(|q| [q[0] * sx + -1.0, q[1] * sy + 1.0]);
    let gt = |a: [f32; 2], b: [f32; 2]| a[0] > b[0] || (a[0] == b[0] && a[1] > b[1]);
    let mut k = usize::from(gt(c[0], c[1]));
    if gt(c[k], c[2]) {
        k = 2;
    }
    [k, (k + 1) % 3, (k + 2) % 3]
}

/// The rasterizer state's cull mode for a draw: `Back` keeps only front faces (positive snapped
/// area, clockwise on screen), `Front` only back faces, `None` both.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Cull {
    None,
    Back,
    Front,
}

impl Cull {
    fn keeps(self, area: i64) -> bool {
        match self {
            Cull::None => true,
            Cull::Back => area > 0,
            Cull::Front => area < 0,
        }
    }
}

/// Rasterize one Gouraud triangle whose colour is a **per-pixel brush source** modulated by an
/// interpolated scalar coverage — the realization/ink feather replayed with a gradient/bitmap/image
/// brush. `cov` is the coverage at the three vertices, interpolated through the *same* snapped-1/256
/// Gouraud plane as [`fill_triangle`] (so the coverage is byte-identical to the solid feather fill);
/// `src(px, py)` returns the premultiplied BGRA source (0..255) at the pixel centre. The product
/// `src · cov` is quantised **once** (round-ties-even) then source-over blended through WARP's output
/// merger, exactly as [`fill_triangle`]'s `premul` path does for the baked `colour · cov`.
pub(crate) fn fill_triangle_brush_cov(
    buf: &mut PixelBuffer,
    pos: [[f32; 2]; 3],
    cov: [f32; 3],
    blend: Blend,
    mut src: impl FnMut(f32, f32) -> [f32; 4],
) {
    let (w, h) = (buf.desc.width as usize, buf.desc.height as usize);
    visit_triangle_brush_cov(w, h, pos, cov, blend != Blend::Copy, Cull::None, |x, y, cv| {
        composite_cov(buf, x, y, src(x as f32 + 0.5, y as f32 + 0.5), cv, blend);
    });
}

/// [`fill_triangle_brush_cov`]'s rasterisation on a `w×h` target, handing each covered pixel and its
/// interpolated coverage to `visit`; a triangle `cull` rejects at setup visits nothing.
pub(crate) fn visit_triangle_brush_cov(
    w: usize,
    h: usize,
    pos: [[f32; 2]; 3],
    cov: [f32; 3],
    clip: bool,
    cull: Cull,
    mut visit: impl FnMut(usize, usize, f32),
) {
    // The same clip, rotation and viewport round trip as [`fill_triangle`] (r16 21008's source-over
    // radial realization at the target's left edge needs the fan). A COPY draw keeps its triangles whole:
    // recap 9448's COPY radial realization matches native only unclipped at x 127.
    // ponytail: native's COPY realization takes a path whose viewport holds the whole mesh (not ported);
    // port it if a COPY seed needs that viewport's own round trip.
    let (fw, fh) = (w as f32, h as f32);
    if clip && pos.iter().any(|q| q[0] < 0.0 || q[0] > fw || q[1] < 0.0 || q[1] > fh) {
        let tri: [[f32; 3]; 3] = core::array::from_fn(|i| [pos[i][0], pos[i][1], cov[i]]);
        let fan = match clip_triangle(tri, [0.0, 0.0, fw, fh]) {
            Clipped::Rejected => return,
            Clipped::Inside => vec![tri],
            Clipped::Fan(f) => f,
        };
        for s in fan {
            visit_brush_cov_keyed(w, h, s.map(|t| [t[0], t[1]]), s.map(|t| t[2]), [0, 1, 2], cull, &mut visit);
        }
        return;
    }
    let o = clip_rotation(pos, fw, fh);
    visit_brush_cov_keyed(w, h, pos.map(|q| viewport_round_trip(q, fw, fh)), cov, o, cull, &mut visit);
}

fn visit_brush_cov_keyed(w: usize, h: usize, pos: [[f32; 2]; 3], cov: [f32; 3], o: [usize; 3], cull: Cull, visit: &mut impl FnMut(usize, usize, f32)) {
    let (p, area) = snap_tri(pos);
    if area == 0 || !cull.keeps(area) {
        return;
    }
    let [dadx, dady, c0] = plane::setup_triangle_attr(o.map(|i| p[i][0]), o.map(|i| p[i][1]), o.map(|i| cov[i]));
    cover_tri(p, area, w as i64, h as i64, |x, y| {
        let (xf, yf) = (x as f32 + 0.5, y as f32 + 0.5);
        visit(x, y, (xf * dadx + (yf * dady + c0)).clamp(0.0, 1.0));
    });
}

/// Composite the premultiplied BGRA source `s` (0..255) at pixel `(x, y)` through coverage `cv` under
/// `blend`, as WARP's brush-through-coverage kernel does.
pub(crate) fn composite_cov(buf: &mut PixelBuffer, x: usize, y: usize, s: [f32; 4], cv: f32, blend: Blend) {
    // The coverage reaches the brush as an 8-bit value (native fills a brush-sourced realization
    // through an A8 coverage intermediate): `src · q8(cov)/255`, rounded once (r14 19242's radial
    // stroked realization; a float coverage rounds an alpha LSB high on 9 px).
    let cvq = (cv * 255.0).round_ties_even() / 255.0;
    let sc: [i32; 4] = core::array::from_fn(|k| (s[k] * cvq).round_ties_even().clamp(0.0, 255.0) as i32);
    let i = y * buf.stride + x * 4;
    let out: [i32; 4] = match blend {
        // COPY lerps the target toward the byte source by the 8-bit coverage, as native's AA COPY
        // fills do (r13 18746's realization fringe keeps 60% of the backdrop at coverage 0.40).
        Blend::Copy => {
            let s8: [u8; 4] = core::array::from_fn(|k| s[k].round_ties_even().clamp(0.0, 255.0) as u8);
            let d: [u8; 4] = buf.data[i..i + 4].try_into().unwrap();
            crate::pixels::copy_tess::lerp8(d, s8, (cv * 255.0).round_ties_even() as i32).map(i32::from)
        }
        Blend::Min => core::array::from_fn(|k| {
            ((s[k] * cvq + (255.0 - s[3] * cvq)).round_ties_even().clamp(0.0, 255.0) as i32).min(buf.data[i + k] as i32)
        }),
        Blend::Max => core::array::from_fn(|k| sc[k].max(buf.data[i + k] as i32)),
        Blend::Add => core::array::from_fn(|k| sc[k] + buf.data[i + k] as i32),
        Blend::SourceOver => {
            // WARP's pixel kernel fuses the blend as `src + (p + (p>>8) + 128) >> 8`,
            // `p = dst·(255 − srcA)`.
            let ia = 255 - sc[3];
            core::array::from_fn(|k| { let p = buf.data[i + k] as i32 * ia; sc[k] + ((p + (p >> 8) + 128) >> 8) })
        }
    };
    buf.data[i..i + 4].copy_from_slice(&out.map(|v| v.clamp(0, 255) as u8));
}

/// WARP's viewport frustum clip of one triangle `[x, y, attrs…]` (device px, w = 1) against the
/// viewport `[l, t, r, b]`, fanned from the first vertex of the clipped polygon. The same arithmetic as the transform
/// sampler's clip: positions map to clip space (`x·2/(r−l) − (r+l)/(r−l)`, y flipped), Sutherland–Hodgman
/// over w, w+x, w−x, w−y, w+y interpolating from the inside endpoint (differences in f32, the
/// multiply/divide/add in f64), then back through the viewport. `N ≤ 9`.
pub(crate) fn viewport_clip<const N: usize>(tri: [[f32; N]; 3], [l, t, r, b]: [f32; 4]) -> Vec<[[f32; N]; 3]> {
    // [x_clip, y_clip, w, attrs…]
    let mut poly: Vec<[f32; 10]> = tri.iter().map(|v| {
        let mut o = [0.0; 10];
        o[0] = v[0] * (2.0 / (r - l)) + (-(r + l) / (r - l));
        o[1] = v[1] * (-2.0 / (b - t)) + ((b + t) / (b - t));
        o[2] = 1.0;
        o[3..N + 1].copy_from_slice(&v[2..N]);
        o
    }).collect();
    // WARP's clipper first rotates the triangle (cyclically, through its rotation table) to start at
    // its min clip-space x vertex, ties to the min clip y (the bottom-most); the fan then starts there.
    let first = if poly[0][0] > poly[1][0] || (poly[1][0] == poly[0][0] && poly[0][1] > poly[1][1]) { 1 } else { 0 };
    let first = if poly[first][0] > poly[2][0] || (poly[2][0] == poly[first][0] && poly[first][1] > poly[2][1]) { 2 } else { first };
    poly.rotate_left(first);
    let dist = |v: &[f32; 10], plane: usize| match plane { 0 => v[2], 1 => v[0] + v[2], 2 => v[2] - v[0], 3 => v[2] - v[1], _ => v[1] + v[2] };
    for plane in 0..5 {
        let mut out = Vec::new();
        for i in 0..poly.len() {
            let (a, c) = (poly[i], poly[(i + 1) % poly.len()]);
            let (da, dc) = (dist(&a, plane), dist(&c, plane));
            if da >= 0.0 { out.push(a); }
            if (da >= 0.0) != (dc >= 0.0) {
                let (a, c, da, dc) = if da >= 0.0 { (a, c, da, dc) } else { (c, a, dc, da) };
                out.push(core::array::from_fn(|k| ((c[k] - a[k]) as f64 * da as f64 / (da - dc) as f64 + a[k] as f64) as f32));
            }
        }
        poly = out;
    }
    let back = |v: [f32; 10]| {
        let rw = 1.0 / v[2];
        let mut o = [0.0; N];
        o[0] = (v[0] * rw) * ((r - l) * 0.5) + (r + l) * 0.5;
        o[1] = (v[1] * rw) * ((t - b) * 0.5) + (b + t) * 0.5;
        for k in 2..N { o[k] = v[k + 1] * rw; }
        o
    };
    (1..poly.len().saturating_sub(1)).map(|i| [back(poly[0]), back(poly[i]), back(poly[i + 1])]).collect()
}

/// SSE-style scalar max: `dst > src ? dst : src` — the SECOND operand whenever either is NaN (not
/// IEEE `fmax`/`f32::max`, which drop the NaN). WARP's clipper clamps screen positions with it.
fn sse_max(dst: f32, src: f32) -> f32 {
    if dst > src {
        dst
    } else {
        src
    }
}

/// WARP's clipper outcome for one triangle (see [`clip_triangle`]).
pub(crate) enum Clipped<const N: usize> {
    /// A NaN clip distance (`ComputeClipCodesPacked` forces the AND code to 1) or all three vertices
    /// outside one plane: native draws nothing.
    Rejected,
    /// All inside (OR code 0): the triangle goes straight to setup, unclipped.
    Inside,
    /// The fan of the clipped polygon. A position whose clipped x/y came out NaN stays non-finite here
    /// (native's SSE-style max clamp lets the NaN through and the float→int conversion makes it `i32::MIN`).
    Fan(Vec<[[f32; N]; 3]>),
}

/// WARP's frustum clip of one triangle `[x, y, attrs…]` (device px, w = 1) against the viewport
/// `[l, t, r, b]`, in WARP's operation order: clip codes (distances `w+x, w−x, w−y, w+y`;
/// sign-bit codes; any NaN distance rejects), the clip (rotate to the min clip-x vertex, then per
/// set plane, with outside meaning `0 > d` (false for NaN): `if out(d_a) { if !out(d_c) emit I(c→a) }
/// else { emit a; if out(d_c) emit I(a→c) }`), the interpolation (from the inside vertex,
/// `(c−a)·d_a/(d_a−d_c) + a` with the difference in f32 and the multiply/divide in f64, later planes'
/// distances too), and the clipped vertices' screen mapping (`x' = w > sse_max(x,−w) ? sse_max(−w,x) : w`,
/// then `x'/w·(r−l)/2 + (r+l)/2`). `N ≤ 9`.
pub(crate) fn clip_triangle<const N: usize>(tri: [[f32; N]; 3], [l, t, r, b]: [f32; 4]) -> Clipped<N> {
    // [x_clip, y_clip, w, attrs…] and the four x/y plane distances (indices 2..=5 as native's).
    let vert = |v: &[f32; N]| {
        let mut o = [0.0f32; 10];
        o[0] = v[0] * (2.0 / (r - l)) + (-(r + l) / (r - l));
        o[1] = v[1] * (-2.0 / (b - t)) + ((b + t) / (b - t));
        o[2] = 1.0;
        o[3..N + 1].copy_from_slice(&v[2..N]);
        let d = [0.0, 0.0, o[2] + o[0], o[2] - o[0], o[2] - o[1], o[1] + o[2]];
        (o, d)
    };
    let mut poly: Vec<([f32; 10], [f32; 6])> = tri.iter().map(vert).collect();
    if poly.iter().any(|(_, d)| d[2..].iter().any(|v| v.is_nan())) {
        return Clipped::Rejected;
    }
    let code = |d: &[f32; 6]| (2..6).fold(0u32, |c, k| c | (u32::from(d[k].is_sign_negative()) << k));
    let (or, and) = poly.iter().fold((0u32, !0u32), |(o, a), (_, d)| (o | code(d), a & code(d)));
    if and != 0 {
        return Clipped::Rejected;
    }
    if or == 0 {
        return Clipped::Inside;
    }
    let first = if poly[0].0[0] > poly[1].0[0] || (poly[1].0[0] == poly[0].0[0] && poly[0].0[1] > poly[1].0[1]) { 1 } else { 0 };
    let first = if poly[first].0[0] > poly[2].0[0] || (poly[2].0[0] == poly[first].0[0] && poly[first].0[1] > poly[2].0[1]) { 2 } else { first };
    poly.rotate_left(first);
    // Interpolate from vertex `a` (distance `d_a`) toward `c`: the clip-space position (and the plane
    // distances) as `(v_c − v_a)·d_a/(d_a − d_c) + v_a` rounded once from f64, the attributes in f32 as
    // `v_a + t·(v_c − v_a)` with `t = d_a/(d_a − d_c)` (the clipped vertices WARP sets up carry exactly these).
    let interp = |a: &([f32; 10], [f32; 6]), c: &([f32; 10], [f32; 6]), da: f32, dc: f32, plane: usize| {
        let lerp = |va: f32, vc: f32| ((vc - va) as f64 * da as f64 / (da - dc) as f64 + va as f64) as f32;
        let t = da / (da - dc);
        let attr = |va: f32, vc: f32| va + t * (vc - va);
        let mut o = (core::array::from_fn(|k| if k >= 3 { attr(a.0[k], c.0[k]) } else { lerp(a.0[k], c.0[k]) }), a.1);
        for k in plane + 1..6 {
            if or & (1 << k) != 0 {
                o.1[k] = lerp(a.1[k], c.1[k]);
            }
        }
        o
    };
    for plane in 2..6 {
        if or & (1 << plane) == 0 {
            continue;
        }
        let mut out = Vec::with_capacity(poly.len() + 1);
        for i in 0..poly.len() {
            let (a, c) = (&poly[i], &poly[(i + 1) % poly.len()]);
            let (da, dc) = (a.1[plane], c.1[plane]);
            if da < 0.0 {
                if dc < 0.0 {
                    continue;
                }
                out.push(interp(c, a, dc, da, plane));
            } else {
                out.push(*a);
                // `0 > d_c`: only a definitely-negative d_c clips — a NaN counts as inside here.
                if !(dc < 0.0) {
                    continue;
                }
                out.push(interp(a, c, da, dc, plane));
            }
        }
        poly = out;
        if poly.is_empty() {
            return Clipped::Fan(Vec::new());
        }
    }
    let back = |v: &[f32; 10]| {
        let w = sse_max(f32::MIN_POSITIVE, v[2]);
        let clamp = |p: f32| if w > sse_max(p, -w) { sse_max(-w, p) } else { w };
        let rw = 1.0 / w;
        let mut o = [0.0; N];
        o[0] = (rw * clamp(v[0])) * ((r - l) * 0.5) + (r + l) * 0.5;
        o[1] = (rw * clamp(v[1])) * ((t - b) * 0.5) + (b + t) * 0.5;
        for k in 2..N {
            o[k] = v[k + 1] * rw;
        }
        o
    };
    Clipped::Fan((1..poly.len().saturating_sub(1)).map(|i| [back(&poly[0].0), back(&poly[i].0), back(&poly[i + 1].0)]).collect())
}

/// Whether WARP's triangle setup passes this triangle on to rasterization, from its 24.8
/// float→int positions (a non-finite or out-of-range value is `i32::MIN`): the snapped x span and y
/// span must each contain a pixel centre (`v + lut[v & 0xff]`, the next centre at or past `v`, differs
/// between the min and the max), and the signed area `(x1−x0)·(y2−y0) − (x0−x2)·(y0−y1)` — 32-bit
/// differences that wrap, 64-bit products — must be positive: Direct2D culls back faces, so a zero or
/// negative area (a NaN-clipped vertex's wrapped arithmetic included) never reaches setup.
/// Whether the triangle's exact (unwrapped) 24.8 winding agrees with the positive wrapped area
/// [`setup_accepts`] keys on: setup takes the wrapped sign, but the edge walk steps the true edges, so a
/// triangle whose exact winding is negative covers no pixel (r16 21562: an `i32::MIN` fan triangle with
/// a positive wrapped area and a negative exact one draws nothing natively). Its pixel-stage fault does
/// not care ([`crate::context`]'s `poisons` keeps the wrapped test alone, fillmesh-nan-* probes).
pub(crate) fn exact_front_facing(pos: [[f32; 2]; 3]) -> bool {
    let cvt = |v: f32| {
        let s = v * 256.0;
        if s.is_finite() && s >= i32::MIN as f32 && s < i32::MAX as f32 { s.round_ties_even() as i128 } else { i32::MIN as i128 }
    };
    let p: [[i128; 2]; 3] = core::array::from_fn(|k| [cvt(pos[k][0]), cvt(pos[k][1])]);
    (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[0][0] - p[2][0]) * (p[0][1] - p[1][1]) > 0
}

pub(crate) fn setup_accepts(pos: [[f32; 2]; 3]) -> bool {
    let cvt = |v: f32| {
        let s = v * 256.0;
        if s.is_finite() && s >= i32::MIN as f32 && s < i32::MAX as f32 { s.round_ties_even() as i32 } else { i32::MIN }
    };
    let p: [[i32; 2]; 3] = core::array::from_fn(|k| [cvt(pos[k][0]), cvt(pos[k][1])]);
    let centre = |v: i32| v.wrapping_add(128i32.wrapping_sub(v & 255) & 255);
    let spans = |k: usize| {
        let (lo, hi) = (p[0][k].min(p[1][k]).min(p[2][k]), p[0][k].max(p[1][k]).max(p[2][k]));
        centre(lo) != centre(hi)
    };
    if !spans(0) || !spans(1) {
        return false;
    }
    let d = |a: i32, b: i32| a.wrapping_sub(b) as i64;
    let area = d(p[1][0], p[0][0]).wrapping_mul(d(p[2][1], p[0][1])).wrapping_sub(d(p[0][0], p[2][0]).wrapping_mul(d(p[0][1], p[1][1])));
    area > 0
}

/// Snap a triangle's 3 device points to the 1/256-subpixel grid and return them with the signed
/// area×2 (`>0` ⇒ CCW). Shared setup for [`fill_triangle`] and [`cover_tri`].
pub(crate) fn snap_tri(pos: [[f32; 2]; 3]) -> ([[i64; 2]; 3], i64) {
    let p: [[i64; 2]; 3] = [
        [snap(pos[0][0]), snap(pos[0][1])],
        [snap(pos[1][0]), snap(pos[1][1])],
        [snap(pos[2][0]), snap(pos[2][1])],
    ];
    let area = edge(p[0][0], p[0][1], p[1][0], p[1][1], p[2][0], p[2][1]);
    (p, area)
}

/// Visit every pixel of the aliased triangle `p` (snapped 1/256 points, signed area `area`) under
/// D3D11's top-left rule, clipped to `w×h`. The single source of the mesh/Gouraud coverage rule:
/// [`fill_triangle`] (colour) and the mesh brush fill (`brush::fill_triangle_brush`) both walk it.
pub(crate) fn cover_tri(p: [[i64; 2]; 3], area: i64, w: i64, h: i64, mut visit: impl FnMut(usize, usize)) {
    if area == 0 {
        return;
    }
    // Orient to a single winding so the inside test is "all edge functions ≥ 0".
    let ccw = area > 0;
    let e = |i: usize, px: i64, py: i64| -> i64 {
        let (a, b) = ([p[(i + 1) % 3][0], p[(i + 1) % 3][1]], [p[(i + 2) % 3][0], p[(i + 2) % 3][1]]);
        let s = edge(a[0], a[1], b[0], b[1], px, py);
        if ccw { s } else { -s }
    };
    // Per-edge top-left bias: on-edge samples (edge fn == 0) count only for top-left edges.
    let bias: [i64; 3] = std::array::from_fn(|i| {
        let (a, b) = (p[(i + 1) % 3], p[(i + 2) % 3]);
        let tl = if ccw { top_left(a[0], a[1], b[0], b[1]) } else { top_left(b[0], b[1], a[0], a[1]) };
        if tl { 0 } else { -1 }
    });
    let minx = (p.iter().map(|q| q[0]).min().unwrap() >> 8).max(0);
    let maxx = ((p.iter().map(|q| q[0]).max().unwrap() + 255) >> 8).min(w);
    let miny = (p.iter().map(|q| q[1]).min().unwrap() >> 8).max(0);
    let maxy = ((p.iter().map(|q| q[1]).max().unwrap() + 255) >> 8).min(h);
    for y in miny..maxy {
        let sy = y * 256 + 128;
        for x in minx..maxx {
            let sx = x * 256 + 128;
            if (0..3).any(|i| e(i, sx, sy) + bias[i] < 0) {
                continue;
            }
            visit(x as usize, y as usize);
        }
    }
}

/// Rasterize one aliased triangle into a single-channel coverage `mask` (`w×h`, row-major f32),
/// keeping the **maximum** of the existing value and the plane-interpolated coverage `cov` at each
/// covered pixel. This is the ink fast path's mask accumulation (`DrawInk_ISM` draws the instanced
/// feathered nib into a coverage mask with a MAX blend, then composites the brush through it): same
/// snapped-1/256 positions and top-left membership as [`fill_triangle`], the attribute plane from
/// [`plane::setup_triangle_attr`] (WARP's attribute-plane setup), but one scalar attribute evaluated
/// with WARP's pixel loop's fused FMA and a MAX write instead of a colour source-over.
pub fn fill_triangle_cov(mask: &mut [f32], w: usize, h: usize, pos: [[f32; 2]; 3], cov: [f32; 3]) {
    visit_triangle_cov(w, h, pos, cov, |x, y, v| {
        let m = &mut mask[y * w + x];
        if v > *m {
            *m = v;
        }
    });
}

/// [`fill_triangle_cov`]'s rasterisation, handing each covered pixel and its coverage to `visit`.
pub(crate) fn visit_triangle_cov(w: usize, h: usize, pos: [[f32; 2]; 3], cov: [f32; 3], mut visit: impl FnMut(usize, usize, f32)) {
    let (wi, hi) = (w as i64, h as i64);
    let p: [[i64; 2]; 3] = [
        [snap(pos[0][0]), snap(pos[0][1])],
        [snap(pos[1][0]), snap(pos[1][1])],
        [snap(pos[2][0]), snap(pos[2][1])],
    ];
    let area = edge(p[0][0], p[0][1], p[1][0], p[1][1], p[2][0], p[2][1]);
    if area == 0 {
        return;
    }
    let ccw = area > 0;
    let e = |i: usize, px: i64, py: i64| -> i64 {
        let (a, b) = ([p[(i + 1) % 3][0], p[(i + 1) % 3][1]], [p[(i + 2) % 3][0], p[(i + 2) % 3][1]]);
        let s = edge(a[0], a[1], b[0], b[1], px, py);
        if ccw { s } else { -s }
    };
    let bias: [i64; 3] = std::array::from_fn(|i| {
        let (a, b) = (p[(i + 1) % 3], p[(i + 2) % 3]);
        let tl = if ccw { top_left(a[0], a[1], b[0], b[1]) } else { top_left(b[0], b[1], a[0], a[1]) };
        if tl { 0 } else { -1 }
    });
    // Attribute plane via WARP's attribute-plane setup ([`plane`]); the per-pixel
    // value is then WARP's pixel loop's **fused** `fma(xc, dadx, fma(yc, dady, c0))` at the sample
    // centre. The fusion is what makes a feather-rim pixel byte-exact: separate f32 multiply-adds
    // accumulate ~6 ULP and round the true 0.9 coverage one A8 step low (229 vs native 230), while the
    // two-FMA form lands either side of the 0.9 tie exactly as WARP does. `mul_add` is a single-rounded
    // fma on every target (hardware or libm), so the mask is identical on native and wasm.
    let [dadx, dady, c0] = plane::setup_triangle_attr([p[0][0], p[1][0], p[2][0]], [p[0][1], p[1][1], p[2][1]], cov);
    let minx = (p.iter().map(|q| q[0]).min().unwrap() >> 8).max(0);
    let maxx = ((p.iter().map(|q| q[0]).max().unwrap() + 255) >> 8).min(wi);
    let miny = (p.iter().map(|q| q[1]).min().unwrap() >> 8).max(0);
    let maxy = ((p.iter().map(|q| q[1]).max().unwrap() + 255) >> 8).min(hi);
    for y in miny..maxy {
        let sy = y * 256 + 128;
        let row = (y as f32 + 0.5).mul_add(dady, c0);
        for x in minx..maxx {
            let sx = x * 256 + 128;
            if (0..3).any(|i| e(i, sx, sy) + bias[i] < 0) {
                continue;
            }
            visit(x as usize, y as usize, (x as f32 + 0.5).mul_add(dadx, row).clamp(0.0, 1.0));
        }
    }
}

/// Composite a straight `rgba` fragment over the pixel with `blend`. WARP's UNORM8 render target
/// quantises the shader's premultiplied float output to 8 bits (round to nearest even) and then
/// blends in integer fixed point, exactly as the DrawBitmap pipeline does — a float blend is ~1 LSB off on the semi-transparent gradient-mesh oracle.
pub(crate) fn blend_px(buf: &mut PixelBuffer, x: usize, y: usize, rgba: [f32; 4], blend: Blend, premul: bool) {
    let a = rgba[3];
    let q = |c: f32| (c * 255.0).round_ties_even().clamp(0.0, 255.0) as i32;
    // Premultiplied source in BGRA order, quantised to UNORM8. When `premul`, the plane already
    // interpolated `colour·coverage` so the channels are taken as-is; otherwise `rgb` is straight and
    // we premultiply by the coverage alpha here (see `fill_triangle`).
    let src = if premul {
        [q(rgba[2]), q(rgba[1]), q(rgba[0]), q(a)]
    } else {
        [q(rgba[2] * a), q(rgba[1] * a), q(rgba[0] * a), q(a)]
    };
    let i = y * buf.stride + x * 4;
    let dst: [u8; 4] = buf.data[i..i + 4].try_into().unwrap();
    let out: [i32; 4] = match blend {
        Blend::Copy => src,
        // MIN: the float source over WHITE, then the UNORM store and the min (native probe grid).
        Blend::Min => {
            let f = if premul { [rgba[2], rgba[1], rgba[0], a] } else { [rgba[2] * a, rgba[1] * a, rgba[0] * a, a] };
            std::array::from_fn(|k| q(f[k] + (1.0 - a)).min(dst[k] as i32))
        }
        Blend::Max => std::array::from_fn(|k| src[k].max(dst[k] as i32)),
        Blend::Add => std::array::from_fn(|k| src[k] + dst[k] as i32),
        // The output merger: `src + ((t + (t >> 8) + 128) >> 8)`, t = dst·(255 − src.a) (the realization
        // kernel's pmullw/paddusw merge), not the sampler's `unorm_mul`.
        Blend::SourceOver => {
            let ia = 255 - src[3];
            std::array::from_fn(|k| {
                let t = dst[k] as i32 * ia;
                src[k] + ((t + (t >> 8) + 128) >> 8)
            })
        }
    };
    buf.data[i..i + 4].copy_from_slice(&out.map(|v| v.clamp(0, 255) as u8));
}

#[cfg(test)]
mod tests {
    use super::{clip_triangle, fill_triangle_cov, linear_attr_axis, setup_accepts, Clipped};

    // A DrawBitmap with an extreme/degenerate dest or source (fuzzer seed 5003: source Y = -3.4e38,
    // so a huge `step`) must not overflow-panic — a public DrawBitmap aborting across the COM ABI is a
    // browser crash. In a debug (overflow-checked) build this panicked before the wrapping fix; the
    // wrapped result is deterministic and byte-matches native WARP for that scene.
    #[test]
    fn linear_attr_axis_degenerate_does_not_overflow() {
        let _ = linear_attr_axis(-3.4e38, 1.174, -6.558, 5.316); // huge source extent ⇒ huge step
        let _ = linear_attr_axis(0.0, 1.0, 5.0, 5.0); // zero-extent dest ⇒ ratio = ±inf
        let _ = linear_attr_axis(0.0, 0.0, 7.0, 7.0); // 0/0 ⇒ ratio = NaN
        let _ = linear_attr_axis(1e30, -1e30, 1e-20, -1e-20); // every term extreme
    }

    // The v2-ink start-cap rim triangle: its coverage at pixel (17,70) is the exact 0.9 tie, which the
    // JIT pixel loop's fused FMA rounds up to A8 230 (native). A naive `c0 + x·dadx + y·dady` with
    // separate multiply-adds drifts ~6 ULP low and quantises to 229 — the one-step regression this
    // guards. (The whole scene is gated by `v2-ink` in conformance/passing.txt.)
    #[test]
    fn feather_rim_09_tie_rounds_up() {
        let (w, h) = (20usize, 72usize);
        let mut mask = vec![0.0f32; w * h];
        fill_triangle_cov(&mut mask, w, h, [[17.6903, 70.9567], [16.5, 70.0], [17.5, 70.0]], [1.0, 0.0, 1.0]);
        let c8 = (mask[70 * w + 17] * 255.0).round_ties_even() as i32;
        assert_eq!(c8, 230, "feather-rim coverage must quantise to the native A8 step (fused FMA eval)");
    }

    // Texel at a given device pixel from the 16.16 DDA.
    fn at(base: i32, step: i32, px: i32) -> (i32, i32) {
        let u = base + px * step;
        (u >> 16, (u >> 8) & 0xff) // (texel, 8-bit weight)
    }

    #[test]
    fn identity_samples_texel_centres() {
        // src [0,W] over dst [0,W]: each device pixel reads its own texel's centre (weight 0).
        let (base, step) = linear_attr_axis(0.0, 64.0, 0.0, 64.0);
        assert_eq!(step, 65536);
        for px in 0..64 {
            assert_eq!(at(base, step, px), (px, 0));
        }
    }

    #[test]
    fn brushes06_anchor_weight() {
        // brushes-06 x-axis: 2×2 texture scaled 40× offset 24 → to_source[0]=0.025, [4]=−0.6; the
        // rect spans device x ∈ [0,128]. At px=70 the anchored DDA gives lower texel 0 with 8-bit
        // weight 169 — the WARP-oracle value (a per-pixel float coord rounds to 170, one too high).
        let (base, step) = linear_attr_axis(-0.6, 2.6, 0.0, 128.0);
        assert_eq!(at(base, step, 70), (0, 169));
    }

    /// WARP's clipper on non-finite mesh vertices (native triangle-setup traces, fx7k):
    /// 15202 tri0 (one +inf x, winding-normalised as Direct2D's mesh stores it) clips to
    /// a fan whose NaN-x vertices reach setup at i32::MIN with a non-zero area (the frame faults);
    /// 7463 tri3 (one −inf x) never reaches setup; a NaN distance rejects outright.
    #[test]
    fn clip_triangle_nonfinite_matches_native() {
        let s = 1.25f32; // 120 dpi
        let dev = |p: [f32; 2]| [p[0] * s, p[1] * s];
        let fan = |t: [[f32; 2]; 3], w: f32, h: f32| match clip_triangle(t, [0.0, 0.0, w, h]) {
            Clipped::Fan(f) => f,
            Clipped::Inside => vec![t],
            Clipped::Rejected => Vec::new(),
        };
        let t15202 = [dev([20.154417, 31.710327]), dev([f32::INFINITY, 27.258366]), dev([24.79615, 35.325249])];
        let f = fan(t15202, 32.0, 96.0);
        assert_eq!(f.len(), 2, "native: two setup triangles");
        assert!(f.iter().any(|t| t.iter().flatten().any(|v| !v.is_finite()) && setup_accepts(*t)));
        // 7463 tri3, front-facing order (p0, p2, p1): native's three fan triangles all stop in triangle setup
        // (one has no pixel-centre row, two wrap to a non-positive area) — none rasterizes, no fault.
        let t7463 = [dev([f32::NEG_INFINITY, 0.716_228_7]), dev([6.170_997, 0.239_715_1]), dev([15.56396, 0.25])];
        let f = fan(t7463, 16.0, 1.0);
        assert_eq!(f.len(), 3, "native: three triangle-setup calls");
        assert!(!f.iter().any(|t| setup_accepts(*t)));
        // 15119 (96 dpi, 48×48; −inf x and −1e9 y; a NaN winding cross swaps ⇒ (p0, p2, p1)): native's fan is two
        // triangles, both accepted by setup (the frame faults).
        let t15119 = [[46.52422, 2.856_017], [0.287_332_8, -1.0e9], [f32::NEG_INFINITY, 22.358_162]];
        let f = fan(t15119, 48.0, 48.0);
        assert_eq!(f.len(), 2);
        assert!(f.iter().all(|t| setup_accepts(*t)));
        assert!(matches!(clip_triangle(dev3([0.0, f32::NAN, 1.0, 1.0, 2.0, 2.0]), [0.0, 0.0, 16.0, 16.0]), Clipped::Rejected));
        fn dev3(v: [f32; 6]) -> [[f32; 2]; 3] {
            [[v[0], v[1]], [v[2], v[3]], [v[4], v[5]]]
        }
    }
}

