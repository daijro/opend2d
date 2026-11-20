//! Native `d2d1.dll` curve flattener, ported bit-for-bit from `CBezierFlattener<float,CPoint2F>`
//! (its ctor, `SetPoints`, `Flatten` and `HalveTheStep` routines). It is MIL adaptive
//! *forward-differencing* — NOT the recursive midpoint
//! subdivision the managed reference uses — and it decides every flattened vertex, hence every
//! length/area/point-at-length result and every curve edge pixel. All arithmetic is `f32` in the
//! native operation order (D2D runs with MXCSR `0x1F80`: round-to-nearest-even, exceptions masked).
//!
//! Verified bit-exact: the ellipse polyline it produces matches native `ComputeLength`
//! (`conformance/oracle/win11-26100/v2-query.json`, `0x435ccf9e`).

pub type Point = [f32; 2];

/// One flattened cubic Bézier, appending points **excluding `p0`, including the final `p3`** to
/// `out` — exactly what native emits (the caller seeds the start point once per figure).
pub fn flatten_cubic(p0: Point, p1: Point, p2: Point, p3: Point, tol: f32, out: &mut Vec<Point>) {
    let mut f = Flattener::new(tol, p0, p1, p2, p3);
    // Native pulls points in batches of 10 (`Flatten(..., 0xA, ...)`); the batch size is just a
    // buffer bound and does not change the result, so a single drain is equivalent.
    loop {
        let more = f.step_batch(out, None, None);
        if !more {
            break;
        }
    }
}

/// [`flatten_cubic`] as Simplify's flattening sink runs it: a step's flatness is
/// `maxss(|x|, |y|)` (the second operand when either is NaN, so a NaN y gives NaN and a NaN x gives
/// `|y|`), halved only on an ordered `> bound`. A cubic whose second differences are NaN in y is
/// flat, its endpoint alone; one NaN in x subdivides by its y differences. (The safe-box clipper's
/// flattener halves on a NaN instead, r13 18657, which [`flatten_cubic`] keeps.)
pub fn flatten_cubic_sink(p0: Point, p1: Point, p2: Point, p3: Point, tol: f32, out: &mut Vec<Point>) {
    let mut f = Flattener::new_with(tol, p0, p1, p2, p3, true);
    // Native pulls points in batches of 10 (`Flatten(..., 0xA, ...)`); the batch size is just a
    // buffer bound and does not change the result, so a single drain is equivalent.
    loop {
        let more = f.step_batch(out, None, None);
        if !more {
            break;
        }
    }
}

/// Flatten a cubic emitting, per output point (excluding `p0`), the analytic **tangent vector**
/// native's `CBezierFlattener::Flatten` writes to its second array: interior points get
/// `6·e0 − e1 − 2·e2` (the forward-difference derivative), the final point `GetLastTangent`
/// (`P3−P2`, falling back to `P3−P1`/`P3−P0`). Used by `ComputePointAtLength`, whose flattener
/// takes the sink's flatness test ([`flatten_cubic_sink`]).
pub fn flatten_cubic_t(p0: Point, p1: Point, p2: Point, p3: Point, tol: f32, pts: &mut Vec<Point>, tans: &mut Vec<Point>) {
    let mut f = Flattener::new_with(tol, p0, p1, p2, p3, true);
    loop {
        let more = f.step_batch(pts, Some(tans), None);
        if !more {
            break;
        }
    }
}

/// [`flatten_cubic_t`] with, per output point, the curve parameter native's `Flatten` writes to its
/// third array: the running f32 sum of the step sizes for interior points, exactly 1 for the end point.
#[allow(clippy::too_many_arguments)]
pub fn flatten_cubic_tp(p0: Point, p1: Point, p2: Point, p3: Point, tol: f32, pts: &mut Vec<Point>, tans: &mut Vec<Point>, params: &mut Vec<f32>) {
    let mut f = Flattener::new(tol, p0, p1, p2, p3);
    while f.step_batch(pts, Some(tans), Some(params)) {}
}

/// The flattener's squared length fuzz `tol²·1e-4` (f32), which the tangent fallbacks gate on.
fn length_fuzz(tol: f32) -> f32 {
    (tol * tol) * 0.000_099_999_997
}

/// `GetFirstTangent`: the cubic's initial tangent vector `P1−P0`, falling back to `P2−P0` then
/// `P3−P0` when the earlier difference is within the squared length fuzz ([`length_fuzz`]).
pub fn first_tangent(p0: Point, p1: Point, p2: Point, p3: Point, tol: f32) -> Point {
    let fuzz = length_fuzz(tol); // a difference within it (or NaN) falls back
    let d = |a: Point| [a[0] - p0[0], a[1] - p0[1]];
    for c in [p1, p2] {
        let t = d(c);
        if t[0] * t[0] + t[1] * t[1] > fuzz {
            return t;
        }
    }
    d(p3)
}

/// Whether `GetFirstTangent` finds a direction: it fails (BAD_NUMBER) when `P1−P0`, `P2−P0` and
/// `P3−P0` all fall within the squared length fuzz `tol²·1e-4` (an ordered compare, so a NaN difference
/// or a NaN fuzz never fails it). The length walk then skips the curve.
pub fn first_tangent_ok(p0: Point, p1: Point, p2: Point, p3: Point, tol: f32) -> bool {
    let fuzz = length_fuzz(tol);
    let d = |a: Point| { let t = [a[0] - p0[0], a[1] - p0[1]]; t[0] * t[0] + t[1] * t[1] };
    d(p1) > fuzz || d(p2) > fuzz || !(fuzz >= d(p3))
}

// Field names mirror the native object (byte offsets in comments). `e0` is the first forward
// difference, `e1` the second, `e2` the previous second difference; see memory
// `geometry-native-flattener`.
struct Flattener {
    b3: Point,
    p0: Point, // control points, for GetLastTangent on the endpoint
    p1: Point,
    p2: Point,
    cur: Point,
    e0: Point, // +64
    e1: Point, // +72
    e2: Point, // +80
    tol: f32,       // the tolerance it was built with (`length_fuzz`)
    err_bound: f32, // +40  max(tol,0)*6
    fuzz: f32,      // +48  err_bound*0.25
    maxss: bool,    // the sink's NaN-ordered flatness test (`flatten_cubic_sink`)
    csteps: i32,    // +88
    param: f32,     // +92  the curve parameter of `cur`
    stepsize: f32,  // +96
}

/// `TryDoubleTheStep`'s test: the step doubles only when the second difference `e2` and the next one
/// `d = 2·e1 − e2` are within `fuzz` in both coordinates. A NaN fails it (native's compares take an
/// unordered value as out of range), so a NaN difference stops the doubling rather than allowing it.
#[inline]
pub(crate) fn step_doubles(e2: [f64; 2], d: [f64; 2], fuzz: f64) -> bool {
    e2[0].abs() <= fuzz && e2[1].abs() <= fuzz && d[0].abs() <= fuzz && d[1].abs() <= fuzz
}

#[inline]
fn fmax_abs(p: Point) -> f32 {
    p[0].abs().max(p[1].abs())
}

impl Flattener {
    fn new(tol: f32, p0: Point, p1: Point, p2: Point, p3: Point) -> Self {
        Self::new_with(tol, p0, p1, p2, p3, false)
    }

    fn new_with(tol: f32, p0: Point, p1: Point, p2: Point, p3: Point, maxss: bool) -> Self {
        let err_bound = tol.max(0.0) * 6.0;
        let mut f = Flattener {
            b3: p3,
            p0,
            p1,
            p2,
            cur: p0,
            e0: [p3[0] - p0[0], p3[1] - p0[1]],
            e1: [
                6.0 * (p1[0] - 2.0 * p2[0] + p3[0]),
                6.0 * (p1[1] - 2.0 * p2[1] + p3[1]),
            ],
            e2: [
                6.0 * (p0[0] - 2.0 * p1[0] + p2[0]),
                6.0 * (p0[1] - 2.0 * p1[1] + p2[1]),
            ],
            tol,
            err_bound,
            fuzz: err_bound * 0.25,
            maxss,
            csteps: 1,
            param: 0.0,
            stepsize: 1.0,
        };
        // SetPoints' initial halving loop: subdivide the single step until the second differences
        // fall under the error bound (or the step floor 0.001 is hit).
        while (f.over(f.e1) || f.over(f.e2)) && f.stepsize as f64 > 0.001 {
            f.halve();
        }
        f
    }

    /// Whether a second difference is over the error bound (see `maxss`).
    fn over(&self, p: Point) -> bool {
        if self.maxss {
            let (a, b) = (p[0].abs(), p[1].abs());
            (if a > b { a } else { b }) > self.err_bound
        } else {
            !(fmax_abs(p) <= self.err_bound)
        }
    }

    fn halve(&mut self) {
        let ne1 = [(self.e1[0] + self.e2[0]) * 0.125, (self.e1[1] + self.e2[1]) * 0.125];
        self.e0 = [(self.e0[0] - ne1[0]) * 0.5, (self.e0[1] - ne1[1]) * 0.5];
        self.e2 = [self.e2[0] * 0.25, self.e2[1] * 0.25];
        self.e1 = ne1;
        self.csteps *= 2;
        self.stepsize = ((self.stepsize as f64) * 0.5) as f32;
    }

    // Emits up to 10 points; returns true if the curve is not yet fully flattened. When `tans` is
    // given, appends the analytic tangent vector at each emitted point (native's 2nd output array);
    // `params` likewise the curve parameter (the 3rd).
    fn step_batch(&mut self, out: &mut Vec<Point>, mut tans: Option<&mut Vec<Point>>, mut params: Option<&mut Vec<f32>>) -> bool {
        let mut emitted = 0;
        if self.csteps > 1 {
            while emitted < 10 {
                // One forward-difference step.
                self.cur = [self.cur[0] + self.e0[0], self.cur[1] + self.e0[1]];
                let o1 = self.e1;
                self.e0 = [self.e0[0] + self.e1[0], self.e0[1] + self.e1[1]];
                self.e1 = [2.0 * o1[0] - self.e2[0], 2.0 * o1[1] - self.e2[1]];
                self.e2 = o1;
                self.csteps -= 1;
                self.param += self.stepsize;
                out.push(self.cur);
                if let Some(p) = params.as_deref_mut() {
                    p.push(self.param);
                }
                if let Some(t) = tans.as_deref_mut() {
                    // Interior point tangent: 6·e0 − e1 − 2·e2 (post-step forward differences).
                    t.push([
                        (self.e0[0] * 6.0 - self.e1[0]) - (self.e2[0] + self.e2[0]),
                        (self.e0[1] * 6.0 - self.e1[1]) - (self.e2[1] + self.e2[1]),
                    ]);
                }
                emitted += 1;

                if !(self.over(self.e1) && self.stepsize as f64 > 0.001) {
                    // Try to double the step while the third differences stay within `fuzz`.
                    while self.csteps & 1 == 0 {
                        let d = [2.0 * self.e1[0] - self.e2[0], 2.0 * self.e1[1] - self.e2[1]];
                        let wide = |p: Point| [p[0] as f64, p[1] as f64];
                        if !step_doubles(wide(self.e2), wide(d), self.fuzz as f64) {
                            break;
                        }
                        self.e0 = [2.0 * self.e0[0] + self.e1[0], 2.0 * self.e0[1] + self.e1[1]];
                        self.e2 = [self.e2[0] * 4.0, self.e2[1] * 4.0];
                        self.e1 = [d[0] * 4.0, d[1] * 4.0];
                        self.stepsize *= 2.0;
                        self.csteps /= 2;
                    }
                } else {
                    self.halve();
                }
                if self.csteps <= 1 {
                    break;
                }
            }
            if emitted >= 10 && self.csteps > 1 {
                return true;
            }
        }
        if emitted >= 10 {
            return true;
        }
        out.push(self.b3);
        if let Some(p) = params.as_deref_mut() {
            p.push(1.0);
        }
        if let Some(t) = tans.as_deref_mut() {
            // Endpoint tangent = GetLastTangent: P3−P2, falling back to P3−P1 / P3−P0 when tiny.
            let gate = length_fuzz(self.tol) * 0.125; // a difference within it falls back
            let d = |a: Point| [self.b3[0] - a[0], self.b3[1] - a[1]];
            let mut lt = d(self.p2);
            if lt[0] * lt[0] + lt[1] * lt[1] <= gate {
                lt = d(self.p1);
                if lt[0] * lt[0] + lt[1] * lt[1] <= gate {
                    lt = d(self.p0);
                }
            }
            t.push(lt);
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A NaN difference never doubles the step (native's compares fail on an unordered value), and the
    // parameter output runs with the points: increasing interior values, exactly 1 at the end.
    #[test]
    fn nan_stops_doubling_and_params_track_points() {
        assert!(step_doubles([0.0, 0.0], [0.0, 0.0], 0.1));
        assert!(!step_doubles([f64::NAN, 0.0], [0.0, 0.0], 0.1));
        assert!(!step_doubles([0.0, 0.0], [0.0, f64::NAN], 0.1));
        let (p0, p1, p2, p3) = ([0.0, 0.0], [10.0, 40.0], [50.0, 40.0], [60.0, 0.0]);
        let (mut plain, mut pts, mut tans, mut params) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        flatten_cubic(p0, p1, p2, p3, 0.25, &mut plain);
        flatten_cubic_tp(p0, p1, p2, p3, 0.25, &mut pts, &mut tans, &mut params);
        assert_eq!(plain, pts);
        assert_eq!((params.len(), tans.len()), (pts.len(), pts.len()));
        assert!(params.windows(2).all(|w| w[0] < w[1]) && *params.last().unwrap() == 1.0);
    }
}
