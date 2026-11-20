//! `ID2D1GradientMesh` + `DrawGradientMesh`, drawn through WARP's
//! triangle pipeline ([`crate::warp`]).
//!
//! A patch is a bicubic Bézier surface (4×4 control points) with four corner colours. Native's
//! gradient-mesh draw builds a device-independent *canonical* vertex buffer (a `tessU × tessV` grid
//! plus a four-edge border ring at strip levels 2 and 3) and a triangle index list, runs a vertex
//! shader that evaluates the surface, Bernstein-interpolates the premultiplied corner colours, and
//! extrudes the border strips half a pixel along the edge normal when the patch's extend modes call
//! for it, then composites every triangle premultiplied-source-over in index order (no depth buffer).
//! [`canonical`] reproduces the buffers and [`mesh_vs`] the shader, both byte-exact against WARP.
//!
//! The grid size is native's tessellation curvature rule: take the patch's top row (u-edge) and left column (v-edge) cubics,
//! map them to device space, and recursively bisect while the control-polygon length exceeds the
//! chord by more than 0.5 px (depth capped at 5). `tess = 2^(depth+1)` vertices per side, so a flat
//! patch gives `tess = 2` → one cell → two triangles — which reproduces the oracle exactly.

use crate::context::DeviceContext;
use crate::pixels::Blend;
use crate::warp::{self, Vert};
use d2d_types::contracts::PixelBuffer;

/// One `D2D1_GRADIENT_MESH_PATCH`: the 4×4 control points `point[row*4+col]` (row = v, col = u),
/// the four corner colours (`color00, color03, color30, color33`) and per-edge modes (unused for
/// the aliased interior; see the module note on edge AA).
#[derive(Clone, Copy)]
pub struct MeshPatch {
    pub points: [[f32; 2]; 16],
    pub colors: [[f32; 4]; 4],
    pub extend: [u32; 4],
}

/// `ID2D1GradientMesh`: an immutable patch array.
pub struct GradientMesh {
    pub patches: Vec<MeshPatch>,
}

impl DeviceContext {
    /// `CreateGradientMesh`.
    pub fn create_gradient_mesh(&self, patches: Vec<MeshPatch>) -> GradientMesh {
        GradientMesh { patches }
    }

    /// `GetGradientMeshWorldBounds`: the mesh's loose (control-point) bounds in the context's unit
    /// space, as native computes them: the AABB of every
    /// patch's 16 control points under the device transform (world post-scaled by dpi/96, applied
    /// per-component as native does), then scaled back to DIPs (×96/dpi) in DIPS unit mode.
    pub fn gradient_mesh_world_bounds(&self, mesh: &GradientMesh) -> [f32; 4] {
        let w = self.get_transform();
        let (dpx, dpy) = self.get_dpi();
        let (sx, sy) = (dpx / 96.0, dpy / 96.0);
        // Device matrix = world × diag(sx, sy): native multiplies the x-column by sx, the y-column by sy.
        let dev = [w[0] * sx, w[1] * sy, w[2] * sx, w[3] * sy, w[4] * sx, w[5] * sy];
        let (mut l, mut t, mut r, mut b) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for patch in &mesh.patches {
            for p in &patch.points {
                let x = p[0] * dev[0] + p[1] * dev[2] + dev[4];
                let y = p[0] * dev[1] + p[1] * dev[3] + dev[5];
                l = l.min(x);
                t = t.min(y);
                r = r.max(x);
                b = b.max(y);
            }
        }
        if self.get_unit_mode() == 0 && r >= l {
            // DIPS: convert the device-space bounds back to DIPs (×96/dpi), mirroring native's two-step
            // arithmetic exactly (not folded into `dev`, so the rounding matches at dpi ≠ 96).
            let (ix, iy) = (96.0 / dpx, 96.0 / dpy);
            l *= ix;
            r *= ix;
            t *= iy;
            b *= iy;
        }
        [l, t, r, b]
    }

    /// `DrawGradientMesh`: tessellate and Gouraud-shade every patch through the WARP triangle
    /// pipeline under the current device transform and primitive blend.
    // ponytail: patches are drawn in array order (painter's), not depth-resolved — native uses a
    // depth buffer to pick a winner where patches overlap. No overlapping-patch oracle scene yet;
    // add a depth buffer in `warp` when one needs it. Edge extend modes / antialiased edges and
    // recording into a command list are likewise unexercised and skipped.
    pub fn draw_gradient_mesh(&mut self, mesh: &GradientMesh) {
        let _a = self.api(24, 0, None);
        // The shared draw gate: a latched error leaves it a silent no-op, as every other draw.
        if !self.ready_to_draw() {
            return;
        }
        // Recording, a command list keeps the mesh and tessellates it where the list is drawn.
        if self.recording.is_some() {
            let (state, dpi) = self.rec_env();
            self.record(crate::command::RecCmd::GradientMesh { state, dpi, patches: mesh.patches.clone().into() });
            return;
        }
        let m = device_matrix(self);
        let blend = blend_of(self.get_primitive_blend());
        // Native tessellates the WHOLE mesh at one (tu, tv). It picks ONE row and ONE column at
        // CreateGradientMesh time, over the DIP-space patches ([`curviest`]); each draw then maps
        // just those two cubics to device space per draw and takes `2 << depth` of each. So a flat patch
        // inherits a curvy sibling's density, and a device-space max over every row (our old rule) differs
        // whenever the transform changes which cubic is deepest (r13 18538: 64×2 natively, not 64×4).
        if mesh.patches.is_empty() {
            return;
        }
        let ((pr, r), (pc, c)) = curviest(&mesh.patches);
        let dev = |p: usize, i: usize| apply(&m, mesh.patches[p].points[i]);
        let tu = tess_number(&[dev(pr, r * 4), dev(pr, r * 4 + 1), dev(pr, r * 4 + 2), dev(pr, r * 4 + 3)]);
        let tv = tess_number(&[dev(pc, c), dev(pc, c + 4), dev(pc, c + 8), dev(pc, c + 12)]);
        let (tu, tv) = (tu as usize, tv as usize);
        let Some(mut buf) = self.take_target() else {
            return;
        };
        for patch in &mesh.patches {
            draw_patch(&mut buf, patch, &m, tu, tv, blend);
        }
        self.set_target_bitmap(buf);
    }
}

/// Tessellate and draw one patch through the gradient-mesh vertex shader (`mesh_vs`), reproducing
/// native's canonical vertex/index buffers (vertex, index and border generation) and its premultiplied source-over compositing in triangle-index order.
fn draw_patch(buf: &mut PixelBuffer, patch: &MeshPatch, m: &[f64; 6], tu: usize, tv: usize, blend: Blend) {
    // Device-space control points, row-major point[row*4+col], as 4 rows of x and of y (VS inputs).
    let dev: Vec<[f32; 2]> = patch.points.iter().map(|&p| apply(m, p)).collect();
    let mut xr = [[0f32; 4]; 4];
    let mut yr = [[0f32; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            xr[r][c] = dev[r * 4 + c][0];
            yr[r][c] = dev[r * 4 + c][1];
        }
    }
    // `tu`/`tv` are the mesh-global tessellation (computed once by the caller over all patches).

    // Premultiplied corner colours (the vertex shader Bernstein-interpolates these; its colour mask is always 1).
    let pcol: [[f32; 4]; 4] =
        core::array::from_fn(|i| { let c = patch.colors[i]; [c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3]] });
    // The vertex shader's boundary attribute is derived from the extend field (= [left, top, right, bottom] edge
    // modes) by a value-map then a within-pair swap, captured from native's per-instance stream across
    // six scenes: the mode→inflate-level map is 0(ALIASED)→0, 1(ANTIALIASED)→2, 2(ALIASED_INFLATED)→1
    // (antialiased edges inflate both strips, inflated only the outer), and boundary = [vm(e1), vm(e0),
    // vm(e3), vm(e2)]. e.g. extend [1,1,2,2] → [2,2,1,1]; [0,1,0,0] → [2,0,0,0]; [2,1,2,1] → [2,1,2,1].
    let e = &patch.extend;
    let vm = |v: u32| -> f32 { if v == 0 { 0.0 } else { (3 - v.min(2)) as f32 } };
    let ext = [vm(e[1]), vm(e[0]), vm(e[3]), vm(e[2])];

    let (cverts, tris) = canonical(tu, tv);
    let shaded: Vec<(Vert, f32)> = cverts
        .iter()
        .map(|&(u, v, f)| {
            let (pos, rgba, mask) = mesh_vs(u, v, &f, &xr, &yr, &pcol, ext);
            (Vert { pos, rgba }, mask)
        })
        .collect();
    // A triangle that crosses the target edge goes through WARP's own frustum clip (`warp::clip_triangle`,
    // in WARP's operation order: each set plane from the inside vertex, NaN-reject codes, the
    // SSE-style max screen clamp) before rasterizing, so the clip-edge pixels match native; fully-inside triangles
    // go straight through unchanged (keeps interiors exact). A clip that recomputes each plane's distances lost all precision
    // between ±3e31 vertices: r13 18538's patch 1 (3.4e32 control points) painted 784 px of slivers that
    // native's clipper leaves above the target (traced: clipped, never reaching triangle setup).
    let (fw, fh) = (buf.desc.width as f32, buf.desc.height as f32);
    for &[a, b, c] in &tris {
        let v = [shaded[a], shaded[b], shaded[c]];
        // A non-finite vertex position (an overflow control point drives the extrude's `0·inf` to NaN)
        // is culled by WARP's clip, so drop the triangle — `crosses` below never fires on NaN (every
        // `<`/`>` is false), which would otherwise snap it to the origin and paint a garbage fan.
        if v.iter().any(|(t, _)| !t.pos[0].is_finite() || !t.pos[1].is_finite()) {
            continue;
        }
        let crosses = v.iter().any(|(t, _)| t.pos[0] < 0.0 || t.pos[0] > fw || t.pos[1] < 0.0 || t.pos[1] > fh);
        if crosses {
            let tri7: [[f32; 7]; 3] = core::array::from_fn(|i| {
                let (t, m) = v[i];
                [t.pos[0], t.pos[1], t.rgba[0], t.rgba[1], t.rgba[2], t.rgba[3], m]
            });
            let fan = match warp::clip_triangle(tri7, [0.0, 0.0, fw, fh]) {
                warp::Clipped::Rejected => continue,
                warp::Clipped::Inside => vec![tri7],
                warp::Clipped::Fan(f) => f,
            };
            for s in fan {
                let vert = |t: [f32; 7]| (Vert { pos: [t[0], t[1]], rgba: [t[2], t[3], t[4], t[5]] }, t[6]);
                let (p, q, r) = (vert(s[0]), vert(s[1]), vert(s[2]));
                fill_mesh_tri(buf, [p.0, q.0, r.0], [p.1, q.1, r.1], blend, 0);
            }
        } else {
            let o = warp::clip_rotation([v[0].0.pos, v[1].0.pos, v[2].0.pos], fw, fh);
            let rt = |t: Vert| Vert { pos: warp::viewport_round_trip(t.pos, fw, fh), ..t };
            fill_mesh_tri(buf, [rt(v[0].0), rt(v[1].0), rt(v[2].0)], [v[0].1, v[1].1, v[2].1], blend, o[0]);
        }
    }
}

/// Rasterize one gradient-mesh triangle. The mesh pixel shader is `colour = colour · mask`, so the
/// premultiplied colour (`v[*].rgba`) and the coverage mask are interpolated through *separate*
/// attribute planes (WARP's attribute-plane setup) and multiplied per pixel — baking the mask into the
/// vertex colour would interpolate `plane(colour·mask)`, which differs from `plane(colour)·plane(mask)` and
/// mis-fades the fold. Coverage, source-over and quantisation are WARP's shared aliased path
/// ([`warp::cover_tri`] + [`warp::blend_px`]), so flat/curved patches stay byte-exact.
fn fill_mesh_tri(buf: &mut PixelBuffer, v: [Vert; 3], mask: [f32; 3], blend: Blend, key: usize) {
    let (w, h) = (buf.desc.width as i64, buf.desc.height as i64);
    let (p, area) = warp::snap_tri([v[0].pos, v[1].pos, v[2].pos]);
    if area == 0 {
        return; // degenerate (border joints; every border quad when extend = 0)
    }
    // Attribute planes [value-at-origin, d/dx, d/dy] from WARP's attribute-plane setup, keyed on vertex 0,
    // over the 1/256-snapped positions ([`warp::plane::setup_triangle_attr`]). With the kernel's
    // unfused `dx·x + (dy·y + origin)` evaluation and its mask handling below, this reproduces WARP's
    // mesh kernel's ties (r14 19957: B lands on 83.5 under a vertex-2-referenced fit, native rounds it to 83).
    let o = [key, (key + 1) % 3, (key + 2) % 3];
    let pl = |a: [f32; 3]| -> [f32; 3] {
        let q = warp::plane::setup_triangle_attr(o.map(|k| p[k][0]), o.map(|k| p[k][1]), o.map(|k| a[k]));
        [q[2], q[0], q[1]]
    };
    let cp: [[f32; 3]; 4] = core::array::from_fn(|k| pl([v[0].rgba[k], v[1].rgba[k], v[2].rgba[k]]));
    let mp = pl(mask);
    let stride = buf.stride;
    let highp = crate::float::is_highp(buf.desc.format);
    warp::cover_tri(p, area, w, h, |x, y| {
        let (pxf, pyf) = (x as f32 + 0.5, y as f32 + 0.5);
        // WARP's mesh kernel (traced for r13 18947 COPY and r14 19957 SOURCE_OVER) evaluates every plane
        // as dx·x + (dy·y + origin), separate SSE mul/add (no FMA): the row base once per row, then dx·x
        // per pixel. Colour planes are R,G,B,A; the source-over kernel multiplies the colour by the
        // UNCLAMPED mask plane before clamping: `clamp(C·m)`.
        let m = pxf * mp[1] + (pyf * mp[2] + mp[0]);
        let col = |k: usize| ((pxf * cp[k][1] + (pyf * cp[k][2] + cp[k][0])) * m).clamp(0.0, 1.0);
        // Float (R16/R32) target: store the premultiplied f32 colour with no 8-bit quantise, WARP's f32
        // blend (shared with the clear / DrawBitmap float path). The aliased mesh is full-coverage, so the
        // whole covered pixel takes the source (cov = 64).
        if highp {
            crate::float::FloatTarget::new(buf).blend_px(x, y, [col(0), col(1), col(2), col(3)], 64, blend);
            return;
        }
        let q = |c: f32| (c * 255.0).round_ties_even().clamp(0.0, 255.0) as i32;
        // 8bpc buffer order B,G,R,A.
        let s = |k: usize| q(col(k));
        let src = [s(2), s(1), s(0), s(3)];
        let i = y * stride + x * 4;
        let dst: [i32; 4] = core::array::from_fn(|k| buf.data[i + k] as i32);
        // The coverage mask `m` is per-pixel COVERAGE, not a plain colour multiplier: the fragment covers
        // fraction `m` of the pixel, so the composite lerps the backdrop in by (1−m). SOURCE-OVER folds
        // that in through the premultiplied src alpha (src.a = C.a·m) and ADD through src itself, but
        // COPY otherwise replaces the interior with a faded (m→0) border strip — native keeps the
        // interior: `out = C·m + dst·(1−m)` (= src + dst·(1−m)). cmesh7500 (extend [0,1,2,2], COPY).
        // Aliased meshes (extend 0) have m≡1 so the term vanishes and the exact paths below are kept.
        let out: [i32; 4] = match blend {
            // COPY: WARP's mesh kernel (traced on r13 18947's COPY border strip) quantizes the clamped
            // colour WITHOUT the mask, `c = round_even(clamp(C)·255)`, and the mask plane on its own,
            // unfused `m8 = round_even(clamp(x·dx + (y·dy + c0))·255)`, then lerps in 16-bit fixed point:
            // `(c·m8 + (c·m8 >> 8) + d·(255−m8) + (d·(255−m8) >> 8) + 128) >> 8`.
            Blend::Copy => {
                let mu = (pxf * mp[1] + (pyf * mp[2] + mp[0])).clamp(0.0, 1.0);
                let m8 = (mu * 255.0).round_ties_even() as i32;
                let craw = |k: usize| q((pxf * cp[k][1] + (pyf * cp[k][2] + cp[k][0])).clamp(0.0, 1.0));
                let c = [craw(2), craw(1), craw(0), craw(3)];
                core::array::from_fn(|k| {
                    let (sm, dm) = (c[k] * m8, dst[k] * (255 - m8));
                    (sm + (sm >> 8) + dm + (dm >> 8) + 128) >> 8
                })
            }
            // MIN composites the float source over WHITE before the store: `min(dst, src + 1 − src.a)`.
            Blend::Min => {
                let a = col(3);
                let w = [q(col(2) + 1.0 - a), q(col(1) + 1.0 - a), q(col(0) + 1.0 - a), 255];
                core::array::from_fn(|k| w[k].min(dst[k]))
            }
            Blend::Max => core::array::from_fn(|k| src[k].max(dst[k])),
            Blend::Add => core::array::from_fn(|k| src[k] + dst[k]),
            // WARP's blend unit: ONE passes x·256, an alpha factor f gives p+(p>>8) (p=x·f), ZERO drops
            // it, and the sum rounds once ((Σ+128)>>8) — unlike the shared unorm_mul, which folds the
            // +128 into the high-byte correction and so over-rounds a fold-overlap pixel by 1.
            Blend::SourceOver => {
                let ia = 255 - src[3];
                core::array::from_fn(|k| { let p = dst[k] * ia; (src[k] * 256 + p + (p >> 8) + 128) >> 8 })
            }
        };
        buf.data[i..i + 4].copy_from_slice(&out.map(|v| v.clamp(0, 255) as u8));
    });
}

/// g-flag index: top→g4, left→g5, bottom→g6, right→g7.
const GT: usize = 0;
const GL: usize = 1;
const GB: usize = 2;
const GR: usize = 3;

/// Reproduce native's canonical vertex buffer (`(u,v)` + 8 flag bytes) and triangle index list for a
/// `tu × tv` tessellation: the interior grid (`idx = row*tu + col`) followed by the border ring
/// (strip level 3 outer, then strip level 2 inner). Byte-exact against the captured level-32 buffers.
fn canonical(tu: usize, tv: usize) -> (Vec<(f32, f32, [u8; 8])>, Vec<[usize; 3]>) {
    let uv = |col: usize, row: usize| (col as f32 / (tu - 1) as f32, row as f32 / (tv - 1) as f32);
    // (top, bottom, left, right) edge membership and the "natural" (f0, f1) direction codes.
    let edges = |row: usize, col: usize| (row == 0, row == tv - 1, col == 0, col == tu - 1);
    let natf = |(t, b, l, r): (bool, bool, bool, bool)| (1 + b as i32 - t as i32, 1 + l as i32 - r as i32);

    let mut v: Vec<(f32, f32, [u8; 8])> = Vec::new();
    // Interior grid.
    for row in 0..tv {
        for col in 0..tu {
            let e @ (t, b, l, r) = edges(row, col);
            let (f0, f1) = natf(e);
            let corner = (t as u8 + b as u8 + l as u8 + r as u8) == 2;
            let val = if corner { 2 } else { 1 };
            let mut g = [0u8; 4];
            if t { g[GT] = val }
            if l { g[GL] = val }
            if b { g[GB] = val }
            if r { g[GR] = val }
            let f2 = (t || b || l || r) as u8;
            let (uu, vv) = uv(col, row);
            v.push((uu, vv, [f0 as u8, f1 as u8, f2, 0, g[0], g[1], g[2], g[3]]));
        }
    }
    // Interior triangles: two per cell, TR→BL anti-diagonal split.
    let mut tris: Vec<[usize; 3]> = Vec::new();
    for row in 0..tv - 1 {
        for col in 0..tu - 1 {
            let i = |r: usize, c: usize| r * tu + c;
            tris.push([i(row, col), i(row, col + 1), i(row + 1, col)]);
            tris.push([i(row, col + 1), i(row + 1, col + 1), i(row + 1, col)]);
        }
    }
    // Border ring walk: TOP, RIGHT, BOTTOM, LEFT (own-edge index into g).
    let mut walk: Vec<(usize, usize, usize)> = Vec::new(); // (row, col, own)
    for col in 0..tu { walk.push((0, col, GT)); }
    for row in 0..tv { walk.push((row, tu - 1, GR)); }
    for col in 0..tu { walk.push((tv - 1, col, GB)); }
    for row in 0..tv { walk.push((row, 0, GL)); }
    // Strip level 3 (outer): reflected natural direction, own-edge g = 1, constant per run.
    let base3 = v.len();
    for &(row, col, own) in &walk {
        let on_e = (own == GT, own == GB, own == GL, own == GR);
        let (f0, f1) = natf(on_e);
        let mut g = [0u8; 4];
        g[own] = 1;
        let (uu, vv) = uv(col, row);
        v.push((uu, vv, [(2 - f0) as u8, (2 - f1) as u8, 3, 0, g[0], g[1], g[2], g[3]]));
    }
    // Strip level 2 (inner): natural direction, own-edge g = 1, other-edge g = 2 at corners.
    let base2 = v.len();
    for &(row, col, own) in &walk {
        let e @ (t, b, l, r) = edges(row, col);
        let (f0, f1) = natf(e);
        let mut g = [0u8; 4];
        g[own] = 1;
        if t && own != GT { g[GT] = 2 }
        if l && own != GL { g[GL] = 2 }
        if b && own != GB { g[GB] = 2 }
        if r && own != GR { g[GR] = 2 }
        let (uu, vv) = uv(col, row);
        v.push((uu, vv, [f0 as u8, f1 as u8, 2, 0, g[0], g[1], g[2], g[3]]));
    }
    // Border quads between strip3 (outer) and strip2 (inner); skip the 3 inter-edge joints.
    let joints = [tu - 1, tu + tv - 1, 2 * tu + tv - 1];
    for i in 0..walk.len() - 1 {
        if joints.contains(&i) {
            continue;
        }
        let (s3, s3n, s2, s2n) = (base3 + i, base3 + i + 1, base2 + i, base2 + i + 1);
        tris.push([s3, s3n, s2]);
        tris.push([s3n, s2n, s2]);
    }
    (v, tris)
}

/// Rust transcription of the gradient-mesh vertex shader (vs0.dxbc, 133 instructions), byte-exact
/// against WARP's DXBC executor. Evaluates the bicubic surface at `(u,v)`, Bernstein-interpolates the
/// premultiplied corner colours, and (for border-strip vertices whose extend gate `f2² + 5·ext_sel > 10`
/// fires) extrudes the point half a device pixel along the edge normal. `f` = the 8 canonical flag
/// bytes; `xr`/`yr` = the control-point rows' device x/y; `col` = the premultiplied corners; `ext` =
/// the patch's extend field. The extrude magnitude `cb0[2].w` is the constant half-pixel 0.5.
#[allow(clippy::too_many_arguments)]
fn mesh_vs(u: f32, v: f32, f: &[u8; 8], xr: &[[f32; 4]; 4], yr: &[[f32; 4]; 4], col: &[[f32; 4]; 4], ext: [f32; 4]) -> ([f32; 2], [f32; 4], f32) {
    let (f0, f1, f2) = (f[0] as i32, f[1] as i32, f[2] as i32);
    let (g4, g5, g6, g7) = (f[4] as i32, f[5] as i32, f[6] as i32, f[7] as i32);
    let (r0x, r0y, r0z, r0w) = (1.0 - u, 1.0 - v, 3.0 * u, 3.0 * v);
    let (r1x, r1y, r1z) = (r0x * r0x, r0y * r0y, r0y * r0x);
    let (r2x, r2y) = (r0z * u, r0w * v); // 3u², 3v²
    let r3z = v * u;
    let bu = [r0x * r1x, r0z * r1x, r0x * r2x, (u * u) * u]; // B0..B3(u)
    let bv = [r0y * r1y, r0w * r1y, r0y * r2y, (v * v) * v]; // B0..B3(v)
    let mut r5 = [0f32; 4]; // X v-interp per column
    let mut r4 = [0f32; 4]; // Y v-interp per column
    for c in 0..4 {
        r5[c] = bv[1] * xr[1][c]; r5[c] = bv[0] * xr[0][c] + r5[c]; r5[c] = bv[2] * xr[2][c] + r5[c]; r5[c] = bv[3] * xr[3][c] + r5[c];
        r4[c] = bv[1] * yr[1][c]; r4[c] = bv[0] * yr[0][c] + r4[c]; r4[c] = bv[2] * yr[2][c] + r4[c]; r4[c] = bv[3] * yr[3][c] + r4[c];
    }
    let dot4 = |a: &[f32; 4], b: &[f32; 4]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let mut sx = dot4(&r5, &bu);
    let mut sy = dot4(&r4, &bu);
    // Extrude gate.
    let r8f = [g4.min(1) as f32, g5.min(1) as f32, g6.min(1) as f32, g7.min(1) as f32];
    let sel = (g4 + g6) == 1;
    let r2z = (r8f[0] * ext[0] + r8f[2] * ext[2]).trunc();
    let r2w = (r8f[1] * ext[1] + r8f[3] * ext[3]).trunc();
    let ext_sel = if sel { r2z } else { r2w };
    // Coverage mask (the PS multiplies the colour by it): the outer strip (f2 = 3) of an edge that
    // folds (its extend ≥ 2) fades to 0; everything else stays fully covered. Interpolated per-pixel.
    let mask = if f2 <= 2 || ext_sel <= 1.0 { 1.0f32 } else { 0.0 };
    // Bilinear premultiplied colour (accumulated C1, C0, C2, C3).
    let (wc0, wc1, wc2, wc3) = (r1z, r0y * u, r0x * v, r3z);
    let mut rgba = [0f32; 4];
    for k in 0..4 {
        let mut a = wc1 * col[1][k];
        a = wc0 * col[0][k] + a;
        a = wc2 * col[2][k] + a;
        a = wc3 * col[3][k] + a;
        rgba[k] = a;
    }
    if (f2 * f2 + 5 * ext_sel as i32) > 10 {
        let mut off_u = f0 - 1;
        let mut off_v = f1 - 1;
        if off_v * off_u != 0 {
            off_u *= if r2z == 2.0 { 1 } else { 0 };
            off_v *= if r2w == 2.0 { 1 } else { 0 };
        }
        let (r9a, r9b, r9c, r9d) = (1.0 - 3.0 * u, 2.0 - 3.0 * u, 1.0 - 3.0 * v, 2.0 - 3.0 * v);
        let dbu = [r1x * (-3.0), (r0x * r9a) * 3.0, r0z * r9b, r2x]; // dB0..3/du
        let dbv = [r1y * (-3.0), (r0y * r9c) * 3.0, r0w * r9d, r2y]; // dB0..3/dv
        let dxdu = dot4(&r5, &dbu);
        let dydu = dot4(&r4, &dbu);
        let mut txcol = [0f32; 4];
        let mut tycol = [0f32; 4];
        for c in 0..4 {
            txcol[c] = dbv[1] * xr[1][c]; txcol[c] = dbv[0] * xr[0][c] + txcol[c]; txcol[c] = dbv[2] * xr[2][c] + txcol[c]; txcol[c] = dbv[3] * xr[3][c] + txcol[c];
            tycol[c] = dbv[1] * yr[1][c]; tycol[c] = dbv[0] * yr[0][c] + tycol[c]; tycol[c] = dbv[2] * yr[2][c] + tycol[c]; tycol[c] = dbv[3] * yr[3][c] + tycol[c];
        }
        let dxdv = dot4(&txcol, &bu);
        let dydv = dot4(&tycol, &bu);
        let norm = |x: f32, y: f32| 1.0 / (x * x + y * y).sqrt().max(0.001);
        let nu = norm(dxdu, dydu);
        let (nux, nuy) = (nu * dxdu * 0.5, nu * dydu * 0.5);
        let nv_ = norm(dydv, dxdv);
        let (nvy, nvx) = (nv_ * dydv * 0.5, nv_ * dxdv * 0.5);
        let cross = nux * nvy - nvx * nuy;
        let sign = if cross < 0.0 { -0.5f32 } else { 0.5 };
        let (ofu, ofv) = (off_u as f32, off_v as f32);
        let r2ax = ofu * nuy + nvy * ofv;
        let r2ay = (-ofu) * nux + nvx * (-ofv);
        let on = norm(r2ax, r2ay);
        sx = (on * r2ax) * sign + sx;
        sy = (on * r2ay) * sign + sy;
    }
    ([sx, sy], rgba, mask)
}

/// `tess = 2^(depth+1)` vertices per side from the control-polygon-minus-chord subdivision rule.
fn tess_number(c: &[[f32; 2]; 4]) -> u32 {
    2u32 << bezier_depth(c, 0)
}

/// Native's curviest-cubic search over the mesh's DIP-space patches:
/// the `(patch, row)` and `(patch, column)` whose cubic subdivides deepest under the flatness test
/// (0.5 tolerance), ties going to the larger polygon-minus-chord excess — measured against a best that
/// resets to 0 at each patch, so an equal-depth cubic in a later patch wins on any positive excess —
/// and each search stopping at the first depth-5 cubic.
fn curviest(patches: &[MeshPatch]) -> ((usize, usize), (usize, usize)) {
    let (mut row, mut col) = (((0, 0), 0u32), ((0, 0), 0u32));
    let (mut row_done, mut col_done) = (false, false);
    for (pi, p) in patches.iter().enumerate() {
        let pt = |i: usize| p.points[i];
        for (done, sel, cubic) in [
            (&mut row_done, &mut row, &(|i: usize| [pt(i * 4), pt(i * 4 + 1), pt(i * 4 + 2), pt(i * 4 + 3)]) as &dyn Fn(usize) -> [[f32; 2]; 4]),
            (&mut col_done, &mut col, &(|j: usize| [pt(j), pt(j + 4), pt(j + 8), pt(j + 12)])),
        ] {
            if *done {
                continue;
            }
            let mut best_len = 0.0f32;
            for k in 0..4 {
                let (depth, len) = bezier_depth_excess(&cubic(k), 0);
                if sel.1 < depth || (sel.1 == depth && len > best_len) {
                    *sel = ((pi, k), depth);
                    best_len = len;
                    if depth == 5 {
                        *done = true;
                        break;
                    }
                }
            }
        }
    }
    (row.0, col.0)
}

/// The flatness test's depth and its excess output: the polygon-minus-chord of the leaf that
/// set the depth — the deeper half's, or `fmaxf` of both on a tie.
fn bezier_depth_excess(c: &[[f32; 2]; 4], depth: u32) -> (u32, f32) {
    let d = |a: [f32; 2], b: [f32; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    let excess = (d(c[0], c[1]) + d(c[1], c[2]) + d(c[2], c[3])) - d(c[0], c[3]);
    if !(excess > 0.5) || depth >= 5 {
        return (depth, excess); // a NaN excess counts as flat, as `bezier_depth`
    }
    let (l, r) = split(c);
    let ((dl, el), (dr, er)) = (bezier_depth_excess(&l, depth + 1), bezier_depth_excess(&r, depth + 1));
    if dl <= dr {
        (dr, if dl < dr { er } else { el.max(er) })
    } else {
        (dl, el)
    }
}

/// Recursive bisection depth: how many halvings until the cubic's control polygon is within 0.5 px
/// of its chord (native's flatness test, capped at 5). f32 to match native arithmetic near the cutoff.
fn bezier_depth(c: &[[f32; 2]; 4], depth: u32) -> u32 {
    let d = |a: [f32; 2], b: [f32; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    let poly = d(c[0], c[1]) + d(c[1], c[2]) + d(c[2], c[3]);
    let chord = d(c[0], c[3]);
    // `!(poly - chord > 0.5)` not `<= 0.5`: identical for finite values, but when an overflow
    // control point makes poly == chord == inf the difference is NaN — native's flatness test
    // treats that as flat (stop) where `<= 0.5` would be false and recurse to the cap. (Matches
    // native on 3300/3300 finite+overflow cubics; `<= 0.5` misses the 101.)
    if !(poly - chord > 0.5) || depth >= 5 {
        return depth;
    }
    let (l, r) = split(c);
    bezier_depth(&l, depth + 1).max(bezier_depth(&r, depth + 1))
}

/// de Casteljau split of a cubic at t=0.5.
fn split(c: &[[f32; 2]; 4]) -> ([[f32; 2]; 4], [[f32; 2]; 4]) {
    let mid = |a: [f32; 2], b: [f32; 2]| [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
    let (p01, p12, p23) = (mid(c[0], c[1]), mid(c[1], c[2]), mid(c[2], c[3]));
    let (p012, p123) = (mid(p01, p12), mid(p12, p23));
    let p0123 = mid(p012, p123);
    ([c[0], p01, p012, p0123], [p0123, p123, p23, c[3]])
}

/// The device transform the draw maps through: the context's, each world element scaled by its axis's f32 DPI
/// factor (native's f32 composition; an f64 product differs at a DPI whose /96 is no f32 value, and a full
/// matrix product would meet an infinite element with the scale's zeros).
fn device_matrix(dc: &DeviceContext) -> [f64; 6] {
    dc.device_transform()
}

/// A control point to device space as Direct2D maps a patch (for its tessellation number and for the vertex shader's
/// input): native transforms the points, `(y·m21 + x·m11) + dx` in f32, every op rounded. An f64 product rounded
/// once moves a vertex far out along a collapsed (rank-one) mesh by an ulp, and the clipper's cancellation there turns
/// that into a different band (r24 29507).
fn apply(m: &[f64; 6], p: [f32; 2]) -> [f32; 2] {
    let f = m.map(|v| v as f32);
    let (x, y) = (p[0], p[1]);
    [(y * f[2] + x * f[0]) + f[4], (y * f[3] + x * f[1]) + f[5]]
}

fn blend_of(mode: u32) -> Blend {
    match mode {
        1 => Blend::Copy,
        2 => Blend::Min,
        3 => Blend::Add,
        4 => Blend::Max,
        _ => Blend::SourceOver,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bitmap_target, Factory};

    // meshgentle-03 (RGBW gently-curved patch) is a tracked ±1 gradient-floor scene; this guards its
    // mismatched-byte count against the baseline (82) so a colour-interpolation change can't regress it
    // beyond the merge gate without failing a fast local test.
    #[test]
    fn meshgentle03_probe() {
        let Some(suite) = d2d_reference::suite_dir() else { return };
        let patch = MeshPatch {
            points: [
                [20.0, 15.0], [30.0, 14.0], [40.0, 16.0], [50.0, 15.0],
                [19.0, 25.0], [31.0, 26.0], [41.0, 24.0], [51.0, 26.0],
                [20.0, 35.0], [30.0, 34.0], [40.0, 36.0], [50.0, 35.0],
                [20.0, 45.0], [30.0, 44.0], [40.0, 46.0], [50.0, 45.0],
            ],
            colors: [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            extend: [0, 0, 0, 0],
        };
        let factory = Factory::new(false);
        let mut dc = factory.create_device().create_device_context();
        dc.set_dpi(96.0, 96.0);
        dc.set_target_bitmap(bitmap_target(64, 56, 96.0));
        dc.begin_draw();
        dc.clear(Some([0.0, 0.0, 0.0, 0.0]));
        let mesh = dc.create_gradient_mesh(vec![patch]);
        dc.draw_gradient_mesh(&mesh);
        let _ = dc.end_draw();
        let px = dc.take_target().unwrap().data;
        let oracle = std::fs::read(suite.join("oracle/win11-26100/meshgentle-03.bgra")).unwrap();
        let bytes = px.iter().zip(&oracle).filter(|(a, b)| a != b).count();
        assert!(bytes <= 82, "meshgentle-03 regressed past the gate baseline: {bytes} mismatched bytes (>82)");
    }

    #[test]
    fn canonical_counts() {
        // Interior grid (tu·tv) + a four-edge border ring at strip levels 2 and 3 (2·(tu+tv) each);
        // interior triangles 2·(tu-1)·(tv-1) + 4·(edge-1) border quads ×2. Byte-exactness against the
        // captured native buffers is covered by `curved_mesh_byte_exact` and the mesh suite scenes.
        let (tu, tv) = (32, 24);
        let (v, tris) = canonical(tu, tv);
        assert_eq!(v.len(), tu * tv + 2 * (2 * (tu + tv)));
        assert_eq!(tris.len(), 2 * (tu - 1) * (tv - 1) + 2 * (2 * (tu + tv) - 1 - 3));
    }

    /// The `warp3d-02-mesh-curved` oracle patch (steep interior colour gradients) — the one whose
    /// WARP-internal f32 residual this module exists to close.
    #[test]
    fn curved_mesh_byte_exact() {
        let Some(suite) = d2d_reference::suite_dir() else { return };
        let patch = MeshPatch {
            points: [
                [12.0, 12.0], [46.0, 40.0], [82.0, 40.0], [116.0, 12.0],
                [12.0, 37.0], [46.0, 55.0], [82.0, 55.0], [116.0, 37.0],
                [12.0, 60.0], [46.0, 42.0], [82.0, 42.0], [116.0, 60.0],
                [12.0, 84.0], [46.0, 70.0], [82.0, 70.0], [116.0, 84.0],
            ],
            colors: [[0.9, 0.15, 0.1, 1.0], [0.1, 0.75, 0.2, 1.0], [0.15, 0.2, 0.85, 1.0], [0.9, 0.85, 0.15, 1.0]],
            extend: [0, 0, 0, 0],
        };
        let factory = Factory::new(false);
        let mut dc = factory.create_device().create_device_context();
        dc.set_dpi(96.0, 96.0);
        dc.set_target_bitmap(bitmap_target(128, 96, 96.0));
        dc.begin_draw();
        dc.clear(Some([1.0, 1.0, 1.0, 1.0]));
        let mesh = dc.create_gradient_mesh(vec![patch]);
        dc.draw_gradient_mesh(&mesh);
        let _ = dc.end_draw();
        let px = dc.take_target().unwrap().data;
        let oracle = std::fs::read(suite.join("oracle/win11-26100/warp3d-02-mesh-curved.bgra")).unwrap();
        let diffs = px.chunks_exact(4).zip(oracle.chunks_exact(4)).filter(|(a, b)| a != b).count();
        assert_eq!(diffs, 0, "{diffs} pixels differ from the WARP oracle");
    }
}
