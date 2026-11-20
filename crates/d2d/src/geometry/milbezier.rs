//! Native `d2d1.dll` **fill** bezier flattener: the MIL `CMILBezier` hybrid-forward-difference
//! cracker that runs on 28.4 fixed-point device coordinates — NOT the float `CBezierFlattener` the
//! length/area queries use. Fills stream each figure through native's sink-to-edge-list adapter, which
//! converts control points to 28.4 and flattens cubics with
//! `CMILBezier::Flatten` (driven from `CFigureData`'s sink). Reproduced here so curve
//! fills feed `pixels::fill_edges` the exact polyline native's scan converter sees.
//!
//! Ported from the MIL algorithm (Kirk Olynyk's HFD cracker; MIT-licensed `dotnet/wpf`
//! `WpfGfx/core/geometry/bezier.{h,cpp}`), with the constants taken from this build:
//! `BEZIER64_FRACTION=28`, `geqErrorHigh=0x300000000000`, `geqErrorLow=0x180000000`, and the 32-bit
//! `HFD32_*` shifts/magnitudes. All arithmetic is integer, matching native bit-for-bit.

// 32-bit cracker (18.14 → 15.17) constants.
const HFD32_INITIAL_SHIFT: i32 = 10;
const HFD32_ADDITIONAL_SHIFT: i32 = 3;
const HFD32_SHIFT: i32 = HFD32_INITIAL_SHIFT + HFD32_ADDITIONAL_SHIFT; // 13
const HFD32_ROUND: i32 = 1 << (HFD32_SHIFT - 1); // 4096
const HFD32_TOLERANCE: i32 = 24; // 6 × (1/4 px in 28.4)
const HFD32_INITIAL_TEST_MAGNITUDE: i32 = HFD32_TOLERANCE << HFD32_INITIAL_SHIFT; // 24576
const HFD32_TEST_MAGNITUDE: i32 = HFD32_INITIAL_TEST_MAGNITUDE << HFD32_ADDITIONAL_SHIFT; // 196608
const HFD32_MAX_ERROR: i32 = HFD32_TOLERANCE << ((2 * HFD32_INITIAL_SHIFT) / 3); // 24 << 6 = 1536
const HFD32_MAX_SIZE: i32 = 0xffffc000u32 as i32;

// 64-bit cracker (36.28) constants.
const BEZIER64_FRACTION: i64 = 28;
const BEZIER64_ROUND: i64 = 1 << (BEZIER64_FRACTION - 1);
const GEQ_ERROR_HIGH: i64 = 0x300000000000; // 6·2^15 >> 4, << 32
const GEQ_ERROR_LOW: i64 = 0x180000000; // 3 << 31

type P = [i32; 2];

fn bbox(pts: &[P; 4]) -> (i32, i32, i32, i32) {
    let (mut l, mut t, mut r, mut b) = (pts[0][0], pts[0][1], pts[0][0], pts[0][1]);
    for p in &pts[1..] {
        l = l.min(p[0]);
        t = t.min(p[1]);
        r = r.max(p[0]);
        b = b.max(p[1]);
    }
    (l - 16, t - 16, r + 16, b + 16) // bounds one pixel loose (half-pixel in 28.4 each side)
}

// --- 32-bit HFD basis (per axis) ------------------------------------------------------------

#[derive(Default, Clone, Copy)]
struct Hfd32 {
    e0: i32,
    e1: i32,
    e2: i32,
    e3: i32,
}

impl Hfd32 {
    fn l_error(&self) -> i32 {
        self.e2.wrapping_abs().max(self.e3.wrapping_abs())
    }
    fn l_parent_error_div4(&self) -> i32 {
        self.e3.wrapping_abs().max((self.e2.wrapping_add(self.e2).wrapping_sub(self.e3)).wrapping_abs())
    }
    fn fx_value(&self) -> i32 {
        (self.e0.wrapping_add(HFD32_ROUND)) >> HFD32_SHIFT
    }
    fn b_init(&mut self, p1: i32, p2: i32, p3: i32, p4: i32, tol: i32) -> bool {
        self.e0 = p1 << HFD32_INITIAL_SHIFT;
        self.e1 = (p4 - p1) << HFD32_INITIAL_SHIFT;
        self.e2 = 6 * (p2 - p3 - p3 + p4);
        self.e3 = 6 * (p1 - p2 - p2 + p3);
        if self.l_error() >= HFD32_MAX_ERROR * tol {
            return false;
        }
        self.e2 <<= HFD32_INITIAL_SHIFT;
        self.e3 <<= HFD32_INITIAL_SHIFT;
        true
    }
    fn v_lazy_halve_step(&mut self, c_shift: i32) {
        self.e2 = (self.e2.wrapping_add(self.e3)) >> 1;
        self.e1 = (self.e1.wrapping_sub(self.e2 >> c_shift)) >> 1;
    }
    fn v_steady_state(&mut self, c_shift: i32) {
        self.e0 <<= HFD32_ADDITIONAL_SHIFT;
        self.e1 <<= HFD32_ADDITIONAL_SHIFT;
        let shift = c_shift - HFD32_ADDITIONAL_SHIFT;
        if shift < 0 {
            self.e2 <<= -shift;
            self.e3 <<= -shift;
        } else {
            self.e2 >>= shift;
            self.e3 >>= shift;
        }
    }
    fn v_halve_step(&mut self) {
        self.e2 = (self.e2.wrapping_add(self.e3)) >> 3;
        self.e1 = (self.e1.wrapping_sub(self.e2)) >> 1;
        self.e3 >>= 2;
    }
    fn v_double_step(&mut self) {
        self.e1 = self.e1.wrapping_add(self.e1).wrapping_add(self.e2);
        self.e3 <<= 2;
        self.e2 = (self.e2 << 3).wrapping_sub(self.e3);
    }
    fn v_take_step(&mut self) {
        self.e0 = self.e0.wrapping_add(self.e1);
        let t = self.e2;
        self.e1 = self.e1.wrapping_add(t);
        self.e2 = self.e2.wrapping_add(t.wrapping_sub(self.e3));
        self.e3 = t;
    }
}

// --- 64-bit HFD basis (per axis) ------------------------------------------------------------

#[derive(Default, Clone, Copy)]
struct Hfd64 {
    e0: i64,
    e1: i64,
    e2: i64,
    e3: i64,
}

impl Hfd64 {
    fn v_error(&self) -> i64 {
        self.e2.wrapping_abs().max(self.e3.wrapping_abs())
    }
    fn v_parent_error(&self) -> i64 {
        (self.e3 << 2).wrapping_abs().max(((self.e2 << 3).wrapping_sub(self.e3 << 2)).wrapping_abs())
    }
    fn fx_value(&self) -> i32 {
        ((self.e0 + BEZIER64_ROUND) >> BEZIER64_FRACTION) as i32
    }
    fn v_init(&mut self, p1: i32, p2: i32, p3: i32, p4: i32) {
        let (p1, p2, p3, p4) = (p1 as i64, p2 as i64, p3 as i64, p4 as i64);
        self.e0 = p1;
        self.e1 = p4;
        self.e2 = p2 - p3 - p3 + p4;
        self.e3 = self.e0 - p2 - p2 + p3;
        self.e1 -= self.e0;
        self.e0 <<= BEZIER64_FRACTION;
        self.e1 <<= BEZIER64_FRACTION;
        self.e2 = (self.e2 * 3) << (BEZIER64_FRACTION + 1);
        self.e3 = (self.e3 * 3) << (BEZIER64_FRACTION + 1);
    }
    fn v_untransform(&self) -> [i32; 4] {
        // p0=e0; p1=e0+(6e1-e2-2e3)/18; p2=e0+(12e1-2e2-e3)/18; p3=e0+e1, each rounded from 36.28.
        let mut p1 = self.e1 + self.e1 + self.e1; // 3e1
        let mut p2 = p1 + p1; // 6e1 (reuse: p1 holds 3e1; mirror native temp use)
        p2 -= self.e2; // 6e1 - e2
        let p2b = p2 + p2 - self.e3; // 12e1 - 2e2 - e3
        p1 = p2 - self.e3 - self.e3; // 6e1 - e2 - 2e3
        p1 /= 18;
        let mut p2 = p2b / 18;
        p1 += self.e0;
        p2 += self.e0;
        let p3 = self.e0 + self.e1;
        let r = |eq: i64| ((eq + BEZIER64_ROUND) >> BEZIER64_FRACTION) as i32;
        [r(self.e0), r(p1), r(p2), r(p3)]
    }
    fn v_halve_step(&mut self) {
        self.e2 = (self.e2 + self.e3) >> 3;
        self.e1 = (self.e1 - self.e2) >> 1;
        self.e3 >>= 2;
    }
    fn v_double_step(&mut self) {
        self.e1 <<= 1;
        self.e1 += self.e2;
        self.e3 <<= 2;
        self.e2 = (self.e2 << 3) - self.e3;
    }
    fn v_take_step(&mut self) {
        self.e0 += self.e1;
        let t = self.e2;
        self.e1 += t;
        self.e2 += t - self.e3;
        self.e3 = t;
    }
}

// --- the two crackers -----------------------------------------------------------------------

struct Bezier32 {
    c_steps: i32,
    x: Hfd32,
    y: Hfd32,
    base: [i32; 2],
    tol: i32, // error-threshold multiplier: 1 for 28.4 (1/16), G/16 for a 1/G grid (so the segment
              // count tracks the device tolerance regardless of the sub-pixel scale of the input).
}

impl Bezier32 {
    fn b_init(pts: &[P; 4], tol: i32, clip: Option<[i32; 4]>) -> Option<Bezier32> {
        let (bl, bt, br, bb) = bbox(pts);
        let mut a = *pts;
        let mut fx_or = 0i32;
        for p in &mut a {
            p[0] -= bl;
            p[1] -= bt;
            fx_or |= p[0] | p[1];
        }
        if fx_or & HFD32_MAX_SIZE != 0 {
            return None;
        }
        let mut x = Hfd32::default();
        let mut y = Hfd32::default();
        if !x.b_init(a[0][0], a[1][0], a[2][0], a[3][0], tol) {
            return None;
        }
        if !y.b_init(a[0][1], a[1][1], a[2][1], a[3][1], tol) {
            return None;
        }
        let mut c_steps = 1i32;
        let mut c_shift = 0i32;
        // A curve whose loose bounds miss the clip is not subdivided: it steps straight to its end.
        let visible = clip.is_none_or(|[l, t, r, b]| bl < r && bt < b && br > l && bb > t);
        if visible {
            loop {
                let test = (HFD32_INITIAL_TEST_MAGNITUDE * tol) << c_shift;
                if x.l_error() <= test && y.l_error() <= test {
                    break;
                }
                c_shift += 2;
                x.v_lazy_halve_step(c_shift);
                y.v_lazy_halve_step(c_shift);
                c_steps <<= 1;
            }
        }
        x.v_steady_state(c_shift);
        y.v_steady_state(c_shift);
        x.v_take_step();
        y.v_take_step();
        c_steps -= 1;
        Some(Bezier32 { c_steps, x, y, base: [bl, bt], tol })
    }

    fn flatten(&mut self, out: &mut Vec<P>) {
        let test_mag = HFD32_TEST_MAGNITUDE * self.tol;
        loop {
            out.push([self.x.fx_value() + self.base[0], self.y.fx_value() + self.base[1]]);
            if self.c_steps == 0 {
                return;
            }
            if self.x.l_error().max(self.y.l_error()) > test_mag {
                self.x.v_halve_step();
                self.y.v_halve_step();
                self.c_steps <<= 1;
            }
            while self.c_steps & 1 == 0
                && self.x.l_parent_error_div4() <= (test_mag >> 2)
                && self.y.l_parent_error_div4() <= (test_mag >> 2)
            {
                self.x.v_double_step();
                self.y.v_double_step();
                self.c_steps >>= 1;
            }
            self.c_steps -= 1;
            self.x.v_take_step();
            self.y.v_take_step();
        }
    }
}

struct Bezier64 {
    x_low: Hfd64,
    y_low: Hfd64,
    x_high: Hfd64,
    y_high: Hfd64,
    c_steps_high: i32,
    c_steps_low: i32,
    hi: i64, // GEQ_ERROR_HIGH × tol
    lo: i64, // GEQ_ERROR_LOW × tol
    clip: Option<[i32; 4]>,
}

impl Bezier64 {
    fn v_init(pts: &[P; 4], tol: i64, clip: Option<[i32; 4]>) -> Bezier64 {
        let mut b = Bezier64 {
            x_low: Hfd64::default(),
            y_low: Hfd64::default(),
            x_high: Hfd64::default(),
            y_high: Hfd64::default(),
            c_steps_high: 1,
            c_steps_low: 0,
            hi: GEQ_ERROR_HIGH * tol,
            lo: GEQ_ERROR_LOW * tol,
            clip,
        };
        b.x_high.v_init(pts[0][0], pts[1][0], pts[2][0], pts[3][0]);
        b.y_high.v_init(pts[0][1], pts[1][1], pts[2][1], pts[3][1]);
        while b.x_high.v_error() > b.hi || b.y_high.v_error() > b.hi {
            b.c_steps_high <<= 1;
            b.x_high.v_halve_step();
            b.y_high.v_halve_step();
        }
        b
    }

    fn flatten(&mut self, out: &mut Vec<P>) {
        let (hi, lo) = (self.hi, self.lo);
        loop {
            if self.c_steps_low == 0 {
                let ax = self.x_high.v_untransform();
                let ay = self.y_high.v_untransform();
                self.x_low.v_init(ax[0], ax[1], ax[2], ax[3]);
                self.y_low.v_init(ay[0], ay[1], ay[2], ay[3]);
                self.c_steps_low = 1;
                // A high step whose piece's loose bounds miss the clip is not subdivided: it steps straight
                // to the piece's end, as the 32-bit cracker does for the whole curve.
                let (bl, bt, br, bb) = bbox(&core::array::from_fn(|k| [ax[k], ay[k]]));
                let visible = self.clip.is_none_or(|[l, t, r, b]| bl < r && bt < b && br > l && bb > t);
                while visible && (self.x_low.v_error() > lo || self.y_low.v_error() > lo) {
                    self.c_steps_low <<= 1;
                    self.x_low.v_halve_step();
                    self.y_low.v_halve_step();
                }
                self.c_steps_high -= 1;
                if self.c_steps_high != 0 {
                    self.x_high.v_take_step();
                    self.y_high.v_take_step();
                    if self.x_high.v_error() > hi || self.y_high.v_error() > hi {
                        self.c_steps_high <<= 1;
                        self.x_high.v_halve_step();
                        self.y_high.v_halve_step();
                    }
                    while self.c_steps_high & 1 == 0
                        && self.x_high.v_parent_error() <= hi
                        && self.y_high.v_parent_error() <= hi
                    {
                        self.x_high.v_double_step();
                        self.y_high.v_double_step();
                        self.c_steps_high >>= 1;
                    }
                }
            }
            self.x_low.v_take_step();
            self.y_low.v_take_step();
            out.push([self.x_low.fx_value(), self.y_low.fx_value()]);
            self.c_steps_low -= 1;
            if self.c_steps_low == 0 && self.c_steps_high == 0 {
                return;
            }
            if self.x_low.v_error() > lo || self.y_low.v_error() > lo {
                self.c_steps_low <<= 1;
                self.x_low.v_halve_step();
                self.y_low.v_halve_step();
            }
            while self.c_steps_low & 1 == 0
                && self.x_low.v_parent_error() <= lo
                && self.y_low.v_parent_error() <= lo
            {
                self.x_low.v_double_step();
                self.y_low.v_double_step();
                self.c_steps_low >>= 1;
            }
        }
    }
}

/// Flatten one cubic given its four control points in **28.4 fixed point**, appending the
/// flattened interior+end points (excluding `p0`) to `out`. This is `CMILBezier::Flatten`: the fast
/// 32-bit cracker for small curves, the 64-bit cracker otherwise.
pub fn flatten_cubic_28_4(p0: P, p1: P, p2: P, p3: P, out: &mut Vec<P>) {
    flatten_cubic_scaled(p0, p1, p2, p3, 1, None, out);
}

/// As [`flatten_cubic_28_4`] but with an error-threshold multiplier `tol` for inputs on a finer grid
/// than 28.4: a 1/G grid uses `tol = G/16`, so the MIL cracker emits the same segment count as 28.4
/// while its vertices land on the finer grid (native's WARP fill tessellator path).
pub fn flatten_cubic_scaled(p0: P, p1: P, p2: P, p3: P, tol: i32, clip: Option<[i32; 4]>, out: &mut Vec<P>) {
    let pts = [p0, p1, p2, p3];
    match Bezier32::b_init(&pts, tol, clip) {
        Some(mut b) => b.flatten(out),
        None => Bezier64::v_init(&pts, tol as i64, clip).flatten(out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A non-trivial cubic must produce a monotone-ish chain that starts past p0 and ends at p3
    // (within the flattener's documented sub-unit endpoint tolerance), with >1 segment.
    #[test]
    fn flattens_cubic() {
        // 1c-05 cubic in 28.4 (device = DIP at dpi96): ceil(16·v − 8).
        let c = |v: f32| (16.0 * v - 8.0).ceil() as i32;
        let p = |x: f32, y: f32| [c(x), c(y)];
        let mut out = Vec::new();
        flatten_cubic_28_4(p(20.0, 60.0), p(40.0, 12.0), p(92.0, 12.0), p(110.0, 60.0), &mut out);
        assert!(out.len() > 4, "expected a subdivided chain, got {}", out.len());
        let end = *out.last().unwrap();
        // End within ~1 unit (28.4) of the true endpoint p3.
        assert!((end[0] - c(110.0)).abs() <= 2 && (end[1] - c(60.0)).abs() <= 2, "end {end:?}");
    }

    // A curve whose loose bounds miss the clip steps straight to its end; one touching it subdivides.
    // A curve too big for the 32-bit cracker takes the 64-bit one, which makes the same test per high
    // step: its off-clip pieces are single steps.
    #[test]
    fn off_clip_curve_is_one_step() {
        let (p0, p1, p2, p3) = ([320, 480], [352, 384], [416, 384], [448, 480]);
        let mut out = Vec::new();
        flatten_cubic_scaled(p0, p1, p2, p3, 1, Some([0, 0, 160, 160]), &mut out);
        assert_eq!(out, vec![p3]);
        out.clear();
        flatten_cubic_scaled(p0, p1, p2, p3, 1, Some([0, 0, 1600, 1600]), &mut out);
        assert!(out.len() > 4 && *out.last().unwrap() == p3);
        // A big curve (64-bit cracker) dipping into the clip only at its start: the pieces past the clip
        // are single steps, so it emits fewer points than unclipped and still ends at p3.
        let (q0, q1, q2, q3) = ([0, 0], [40000, 0], [40000, 40000], [0, 40000]);
        let (mut full, mut cut) = (Vec::new(), Vec::new());
        flatten_cubic_scaled(q0, q1, q2, q3, 1, None, &mut full);
        flatten_cubic_scaled(q0, q1, q2, q3, 1, Some([-160, -160, 160, 160]), &mut cut);
        assert!(cut.len() < full.len() && *cut.last().unwrap() == q3, "{} vs {}", cut.len(), full.len());
    }

    // The error-threshold multiplier (`flatten_cubic_scaled`) must give a finer grid the SAME segment
    // count as 28.4, not more: a `tol = G/16` keeps the device tolerance fixed. A quarter-circle KAPPA
    // cubic flattens to 4 segments at 28.4 (tol 1) AND at 26.6 (×64 input, tol 4) — the WARP tessellator
    // path native's aliased ink fill takes. (Without the tol scale, ×64 over-subdivides to 8.)
    #[test]
    fn scaled_flatten_matches_segment_count() {
        let k = 4.0f32 * 0.5522848;
        // KAPPA quarter arc (r=4): (4,0)->(4,k)->(k,4)->(0,4), centred far from origin to exercise bbox.
        let quad = [(104.0, 40.0), (104.0, 40.0 + k), (100.0 + k, 44.0), (100.0, 44.0)];
        let seg = |scale: f32, tol: i32| {
            let c = |v: f32| (scale * v - scale * 0.5).round() as i32;
            let mut out = Vec::new();
            flatten_cubic_scaled(c1(quad[0], c), c1(quad[1], c), c1(quad[2], c), c1(quad[3], c), tol, None, &mut out);
            out.len()
        };
        fn c1((x, y): (f32, f32), c: impl Fn(f32) -> i32) -> P {
            [c(x), c(y)]
        }
        assert_eq!(seg(16.0, 1), seg(64.0, 4), "26.6 (tol 4) must match 28.4 (tol 1) segment count");
        assert!(seg(64.0, 1) > seg(64.0, 4), "without the tol scale, the finer grid over-subdivides");
    }
}
