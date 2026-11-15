//! The scanner's workspace transform (`CWorkspaceTransform`, as `CScannerModuleD::Set` sets it): the
//! signed fixed-point grid every shape is scanned on.

/// Workspace scale for small coordinates: the fixed grid every shape within reach of it uses.
const SCALE: f64 = 1_048_576.0; // 2^20
const P48: f64 = 281_474_976_710_656.0; // 2^48

/// The scanner's fixed-point workspace: per axis, `grid = round_half_away((v − centre)·scale)` and back
/// `device = grid·(1/scale) + centre`. Shapes whose bounds stay within the fixed `2^20` grid (centre 0)
/// use it; larger bounds take a scale that fits their extent in `2^48`, centred on them.
#[derive(Clone, Copy, Debug)]
pub struct Workspace {
    centre: [f64; 2],
    scale: [f64; 2],
    inv: [f64; 2],
}

impl Workspace {
    /// The fixed `2^20` grid at centre 0.
    pub const FIXED: Workspace = Workspace { centre: [0.0, 0.0], scale: [SCALE, SCALE], inv: [1.0 / SCALE, 1.0 / SCALE] };

    /// The workspace for tight bounds `[l, t, r, b]`: empty (inverted) bounds take a unit box; a bound
    /// corner that would pass `2^48` on the fixed grid (`(l−1)·2^20·1.5 ≤ −2^48`, `(r+1)·2^20·1.5 ≥
    /// 2^48`, likewise top/bottom) sets the per-axis scale `2^48/(extent·1.5+2)` and snaps the centre to it.
    pub fn for_bounds(b: [f32; 4]) -> Workspace {
        let [mut l, mut t, mut r, mut bt] = b.map(|v| v as f64);
        if l > r || t > bt {
            (l, t, r, bt) = (0.0, 0.0, 1.0, 1.0);
        }
        let overflow = (l - 1.0) * SCALE * 1.5 <= -P48 || (t - 1.0) * SCALE * 1.5 <= -P48 || (r + 1.0) * SCALE * 1.5 >= P48 || (bt + 1.0) * SCALE * 1.5 >= P48;
        if !overflow {
            return Workspace::FIXED;
        }
        Self::fit(l, t, r, bt)
    }

    /// The per-axis fit of bounds too far out for a fixed grid: each axis scaled so its extent fits
    /// `2^48`, centred on the bounds and snapped to the grid.
    fn fit(l: f64, t: f64, r: f64, bt: f64) -> Workspace {
        let axis = |lo: f64, hi: f64| {
            let mut c = hi * 0.5 + lo * 0.5;
            let n = (hi - lo) * 1.5 + 2.0;
            if n <= 3.131513062514021e-294 {
                return (c, 1.0, 1.0);
            }
            let s = P48 / n;
            let inv = 1.0 / s;
            c = (s * c + 0.5).floor() * inv;
            (c, s, inv)
        };
        let ((cx, sx, ix), (cy, sy, iy)) = (axis(l, r), axis(t, bt));
        Workspace { centre: [cx, cy], scale: [sx, sy], inv: [ix, iy] }
    }

    /// The tessellator's workspace (`CGeneralFillTessellator::SendToMglSink`) for non-empty fill bounds
    /// `b`, and the extent its tolerance floor scales: the bounds are widened to at least the
    /// `±524287` safe box and scanned on a 64 grid at centre 0, unless a corner passes `2^48` there.
    pub fn for_tessellation(b: [f32; 4]) -> (Workspace, f64) {
        let lo = |v: f32| (if v > -524287.0 { -524287.0 } else { v }) as f64;
        let hi = |v: f32| (if v < 524287.0 { 524287.0 } else { v }) as f64;
        let (l, t, r, bt) = (lo(b[0]), lo(b[1]), hi(b[2]), hi(b[3]));
        let extent = (r - l).max(bt - t);
        let overflow = (l - 1.0) * 64.0 * 1.5 <= -P48 || (t - 1.0) * 64.0 * 1.5 <= -P48 || (r + 1.0) * 64.0 * 1.5 >= P48 || (bt + 1.0) * 64.0 * 1.5 >= P48;
        let ws = if overflow { Self::fit(l, t, r, bt) } else { Workspace { centre: [0.0, 0.0], scale: [64.0, 64.0], inv: [1.0 / 64.0, 1.0 / 64.0] } };
        (ws, extent)
    }

    /// Whether the tessellator's workspace for non-empty fill bounds `b` rejects the stream with
    /// `D2DERR_BAD_NUMBER`: far bounds whose extent is not finite cannot be fitted, and a point the
    /// workspace maps past `2^48` (or to NaN) fails the sink.
    pub fn tessellation_rejects(b: [f32; 4], pts: impl IntoIterator<Item = [f32; 2]>) -> bool {
        let lo = |v: f32| (if v > -524287.0 { -524287.0 } else { v }) as f64;
        let hi = |v: f32| (if v < 524287.0 { 524287.0 } else { v }) as f64;
        let (l, t, r, bt) = (lo(b[0]), lo(b[1]), hi(b[2]), hi(b[3]));
        let overflow = (l - 1.0) * 64.0 * 1.5 <= -P48 || (t - 1.0) * 64.0 * 1.5 <= -P48 || (r + 1.0) * 64.0 * 1.5 >= P48 || (bt + 1.0) * 64.0 * 1.5 >= P48;
        if overflow && !((r - l).is_finite() && (bt - t).is_finite()) {
            return true;
        }
        let (ws, _) = Self::for_tessellation(b);
        let out = |v: f64, k: usize| {
            let d = (v - ws.centre[k]) * ws.scale[k];
            let r = if d < 0.0 { d - 0.5 } else { d + 0.5 };
            let i: i64 = if (-9.223372036854776e18..9.223372036854776e18).contains(&r) { r as i64 } else { i64::MIN };
            (i.wrapping_add(P48 as i64) as u64) > ((P48 as u64) << 1)
        };
        pts.into_iter().any(|q| out(q[0] as f64, 0) || out(q[1] as f64, 1))
    }

    /// Device point → grid.
    pub fn grid(&self, p: [f32; 2]) -> GP {
        let g = |k: usize| round_to_grid((p[k] as f64 - self.centre[k]) * self.scale[k]);
        [g(0), g(1)]
    }

    /// Grid coordinate (possibly fractional) on axis `k` → device.
    pub(crate) fn device_f(&self, v: f64, k: usize) -> f32 {
        (self.inv[k] * v + self.centre[k]) as f32
    }

    /// Grid coordinate on axis `k` → device, kept in f64.
    pub(crate) fn device_d(&self, v: f64, k: usize) -> f64 {
        self.inv[k] * v + self.centre[k]
    }

    /// Both inverse scales' product, the grid-area to device-area factor.
    pub fn inv_scale_product(&self) -> f64 {
        self.inv[0] * self.inv[1]
    }

    /// The grid flattening tolerance: the larger scale times the normalized tolerance, or the extent's
    /// `3e-12` share when that is larger.
    pub fn tol_grid(&self, tol: f64, extent: f64) -> f64 {
        self.scale[0].max(self.scale[1]) * tol.max(extent * 3.0e-12)
    }
}

pub type GP = [i64; 2];

/// A grid coordinate rounded half away from zero and truncated to an integer as the scanner's 64-bit
/// conversion does: a value out of the integer range (or NaN) becomes the integer minimum.
pub fn round_to_grid(v: f64) -> i64 {
    let r = if v < 0.0 { v - 0.5 } else { v + 0.5 };
    if (-9.223372036854776e18..9.223372036854776e18).contains(&r) { r as i64 } else { i64::MIN }
}
