//! Curve support for the region engine: the double-precision grid Bézier flattener
//! that feeds the sweep, and the de Casteljau trim the reconstructor uses to re-emit the original
//! curve. The flattener
//! runs on grid-space control points, emits grid-rounded vertices tagged with their Bézier param,
//! and uses the grid tolerance times 6. Reconstruction trims the original
//! double-precision curve to the run's param range.

/// A cubic as 8 doubles: P0x,P0y,P1x,P1y,P2x,P2y,P3x,P3y.
pub type Bez = [f64; 8];

/// Trim a cubic, keeping the sub-curve on `[0, t]` (reparametrised to `[0,1]`).
pub fn trim_to_end_at(b: &mut Bez, t: f64) {
    let mt = 1.0 - t;
    let u1 = b[4] * t;
    let u2 = b[7] * t + b[5] * (1.0 - t);
    b[6] = t * b[6] + b[4] * (1.0 - t);
    b[7] = u2;
    let u3 = b[2] * t;
    let u4 = b[3] * (1.0 - t) + b[5] * t;
    b[4] = b[2] * (1.0 - t) + u1;
    b[5] = u4;
    let u5 = (1.0 - t) * b[0];
    let u6 = b[3] * t;
    let u7 = b[1] * mt;
    b[2] = u5 + u3;
    b[3] = u6 + u7;
    let u8 = b[4] * t;
    let u9 = b[5] * mt + b[7] * t;
    b[6] = t * b[6] + b[4] * mt;
    b[7] = u9;
    let u10 = mt * b[3] + t * b[5];
    b[4] = mt * b[2] + u8;
    b[5] = u10;
    let u11 = b[7] * t + u10 * mt;
    b[6] = t * b[6] + mt * b[4];
    b[7] = u11;
}

/// Trim a cubic, keeping the sub-curve on `[t, 1]` (reparametrised to `[0,1]`).
pub fn trim_to_start_at(b: &mut Bez, t: f64) {
    let mt = 1.0 - t;
    let u1 = b[3] * t;
    let u2 = b[1] * (1.0 - t);
    let u3 = b[2] * (1.0 - t);
    b[0] = (1.0 - t) * b[0] + b[2] * t;
    b[1] = u1 + u2;
    let u4 = b[4] * (1.0 - t);
    let u5 = b[3] * (1.0 - t) + b[5] * t;
    b[2] = b[4] * t + u3;
    b[3] = u5;
    let u6 = t * b[6];
    let u7 = b[7] * t;
    let u8 = b[5] * mt;
    b[4] = u4 + u6;
    b[5] = u8 + u7;
    let u9 = b[2] * mt;
    let u10 = b[1] * mt + b[3] * t;
    b[0] = mt * b[0] + b[2] * t;
    b[1] = u10;
    let u11 = mt * b[3] + t * b[5];
    b[2] = t * b[4] + u9;
    b[3] = u11;
    let u12 = b[1] * mt + u11 * t;
    b[0] = mt * b[0] + t * b[2];
    b[1] = u12;
}

/// The sub-curve of `b` on the parameter range `[ta, tb] ⊆ [0,1]`, using native's trim order
/// (`TrimToEndAt(tb)` then `TrimToStartAt`), which is how the reconstructor re-emits a
/// boundary run. The start param rescales by `tb` because the end-trim reparametrises `[0,tb]→[0,1]`.
pub fn trim(mut b: Bez, ta: f64, tb: f64) -> Bez {
    if tb < 1.0 {
        trim_to_end_at(&mut b, tb);
    }
    if ta > 0.0 {
        trim_to_start_at(&mut b, if tb < 1.0 { ta / tb } else { ta });
    }
    b
}

// Adaptive forward-differencing flattener state (double precision), mirroring `flatten.rs` but on
// grid coordinates, tracking the Bézier param per step and rounding each emitted vertex to the grid.
struct GridFlattener {
    b3: [f64; 2],
    cur: [f64; 2],
    e0: [f64; 2],
    e1: [f64; 2],
    e2: [f64; 2],
    err_bound: f64,
    fuzz: f64,
    csteps: i32,
    stepsize: f64,
    param: f64,
}

fn amax(p: [f64; 2]) -> f64 {
    p[0].abs().max(p[1].abs())
}

impl GridFlattener {
    fn new(tol: f64, p0: [f64; 2], p1: [f64; 2], p2: [f64; 2], p3: [f64; 2]) -> Self {
        let err_bound = tol.max(0.0) * 6.0;
        let mut f = GridFlattener {
            b3: p3,
            cur: p0,
            e0: [p3[0] - p0[0], p3[1] - p0[1]],
            e1: [6.0 * (p1[0] - 2.0 * p2[0] + p3[0]), 6.0 * (p1[1] - 2.0 * p2[1] + p3[1])],
            e2: [6.0 * (p0[0] - 2.0 * p1[0] + p2[0]), 6.0 * (p0[1] - 2.0 * p1[1] + p2[1])],
            err_bound,
            fuzz: err_bound * 0.25,
            csteps: 1,
            stepsize: 1.0,
            param: 0.0,
        };
        while !(amax(f.e1) <= f.err_bound && amax(f.e2) <= f.err_bound) && f.stepsize > 0.001 {
            f.halve();
        }
        f
    }
    fn halve(&mut self) {
        let ne1 = [(self.e1[0] + self.e2[0]) * 0.125, (self.e1[1] + self.e2[1]) * 0.125];
        self.e0 = [(self.e0[0] - ne1[0]) * 0.5, (self.e0[1] - ne1[1]) * 0.5];
        self.e2 = [self.e2[0] * 0.25, self.e2[1] * 0.25];
        self.e1 = ne1;
        self.csteps *= 2;
        self.stepsize *= 0.5;
    }
}

/// Flatten a grid-space cubic, appending `(grid vertex, param)` for every point **excluding P0,
/// including P3 at param 1.0** — exactly Direct2D's grid Bézier flattener emission. `tol` is the grid
/// tolerance; `err_bound = tol·6`.
pub fn flatten_grid(p0: [f64; 2], p1: [f64; 2], p2: [f64; 2], p3: [f64; 2], tol: f64, out: &mut Vec<([i64; 2], f64)>) {
    let mut f = GridFlattener::new(tol, p0, p1, p2, p3);
    let emit = |out: &mut Vec<([i64; 2], f64)>, pt: [f64; 2], param: f64| {
        out.push(([super::workspace::round_to_grid(pt[0]), super::workspace::round_to_grid(pt[1])], param));
    };
    while f.csteps > 1 {
        // One forward-difference step; the param advances by the current stepsize.
        f.cur = [f.cur[0] + f.e0[0], f.cur[1] + f.e0[1]];
        let o1 = f.e1;
        f.e0 = [f.e0[0] + f.e1[0], f.e0[1] + f.e1[1]];
        f.e1 = [2.0 * o1[0] - f.e2[0], 2.0 * o1[1] - f.e2[1]];
        f.e2 = o1;
        f.csteps -= 1;
        f.param += f.stepsize;
        emit(out, f.cur, f.param);
        if amax(f.e1) <= f.err_bound || f.stepsize <= 0.001 {
            while f.csteps & 1 == 0 {
                let d = [2.0 * f.e1[0] - f.e2[0], 2.0 * f.e1[1] - f.e2[1]];
                if f.fuzz < f.e2[0].abs() || f.fuzz < f.e2[1].abs() || f.fuzz < d[0].abs() || f.fuzz < d[1].abs() {
                    break;
                }
                f.e0 = [2.0 * f.e0[0] + f.e1[0], 2.0 * f.e0[1] + f.e1[1]];
                f.e2 = [f.e2[0] * 4.0, f.e2[1] * 4.0];
                f.e1 = [d[0] * 4.0, d[1] * 4.0];
                f.stepsize *= 2.0;
                f.csteps /= 2;
            }
        } else {
            f.halve();
        }
    }
    emit(out, f.b3, 1.0);
}

