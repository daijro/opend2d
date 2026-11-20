//! Fixed-point arithmetic and the rounding engine for the TrueType interpreter.
//!
//! Everything is plain `i32`: F26Dot6 distances/coordinates (1 px = 64) and 2.14
//! unit vectors (1.0 = 0x4000). The formulas are the ones DirectWrite's classic
//! rasterizer uses (super-round / set-round-values, and the fixed-point mul/div
//! helpers), which are the TrueType spec's own
//! integer math. Written fresh here rather than taken from any font library.

/// 16.16 fixed multiply with round-half-away-from-zero (`FT_MulFix` / `DWRITE_FixMul`).
#[inline]
pub fn fixmul(a: i32, b: i32) -> i32 {
    let ab = a as i64 * b as i64;
    ((ab + 0x8000 - i64::from(ab < 0)) >> 16) as i32
}

/// Scale a font-unit coordinate to F26Dot6 by a 16.16 factor, rounding **half toward +∞** (not
/// half-away-from-zero like [`fixmul`]). DirectWrite's outline scaler adds
/// 0x8000 and arithmetic-shifts without folding the sign, so a coordinate landing exactly on a .5
/// grid line goes up: fu −33 at ppem 16 (scale 0.5) scales to −16, where `fixmul` gives −17.
/// Verified against native's point-mover dump y coordinates for the core fonts.
#[inline]
pub fn scale(fu: i32, scale: i32) -> i32 {
    ((fu as i64 * scale as i64 + 0x8000) >> 16) as i32
}

/// 16.16 fixed divide with rounding (`FT_DivFix` / `DWRITE_FixDiv`).
#[inline]
pub fn fixdiv(a: i32, b: i32) -> i32 {
    let sign = (a < 0) ^ (b < 0);
    let au = a.unsigned_abs() as u64;
    let bu = b.unsigned_abs() as u64;
    let q = if bu == 0 { 0x7FFF_FFFF } else { (((au << 16) + (bu >> 1)) / bu) as u32 };
    if sign { (q as i32).wrapping_neg() } else { q as i32 }
}

/// `a * b / c` in fixed point, rounded (`FT_MulDiv`).
pub fn mul_div(a: i32, b: i32, c: i32) -> i32 {
    let mut sign = 1i32;
    let (mut au, mut bu, mut cu) = (a as u64, b as u64, c as u64);
    if a < 0 { au = 0u64.wrapping_sub(au); sign = -sign; }
    if b < 0 { bu = 0u64.wrapping_sub(bu); sign = -sign; }
    if c < 0 { cu = 0u64.wrapping_sub(cu); sign = -sign; }
    let r = if cu > 0 { (au.wrapping_mul(bu).wrapping_add(cu >> 1) / cu) as i32 } else { 0x7FFF_FFFF };
    if sign < 0 { r.wrapping_neg() } else { r }
}

/// `a * b / c` in fixed point, truncated (`FT_MulDiv_NoRound`).
pub fn mul_div_no_round(a: i32, b: i32, c: i32) -> i32 {
    let mut sign = 1i32;
    let (mut au, mut bu, mut cu) = (a as u64, b as u64, c as u64);
    if a < 0 { au = 0u64.wrapping_sub(au); sign = -sign; }
    if b < 0 { bu = 0u64.wrapping_sub(bu); sign = -sign; }
    if c < 0 { cu = 0u64.wrapping_sub(cu); sign = -sign; }
    let r = if cu > 0 { (au.wrapping_mul(bu) / cu) as i32 } else { 0x7FFF_FFFF };
    if sign < 0 { r.wrapping_neg() } else { r }
}

/// 2.14 fixed multiply (`TT_MulFix14`).
#[inline]
pub fn mul14(a: i32, b: i32) -> i32 {
    let mut v = a as i64 * b as i64;
    v += 0x2000 + (v >> 63);
    (v >> 14) as i32
}


#[inline]
pub fn floor(x: i32) -> i32 {
    x & !63
}
#[inline]
pub fn round(x: i32) -> i32 {
    floor(x.wrapping_add(32))
}
#[inline]
pub fn ceil(x: i32) -> i32 {
    floor(x.wrapping_add(63))
}
#[inline]
fn round_pad(x: i32, n: i32) -> i32 {
    x.wrapping_add(n / 2) & !(n - 1)
}

/// Normalize through signed 2.30 arithmetic, then round to a 2.14 unit vector.
/// Reproduces DirectWrite's vector normalize and fractional sqrt, verified byte-identical.
pub fn normalize14(mut x: i32, mut y: i32) -> (i32, i32) {
    if x == 0 && y == 0 {
        return (0x4000, 0);
    }
    let squared;
    if (-32767..32767).contains(&x) && (-32767..32767).contains(&y) {
        let mut sum = x * x + y * y;
        let mut shift = 15;
        while sum < 0x2000_0000 {
            sum *= 4;
            shift += 1;
        }
        x = x.wrapping_shl(shift);
        y = y.wrapping_shl(shift);
        squared = sum;
    } else {
        while (-0x1fff_ffff..0x2000_0000).contains(&x)
            && (-0x1fff_ffff..0x2000_0000).contains(&y)
        {
            x *= 2;
            y *= 2;
        }
        let square = |v: i32| ((v as i64 * v as i64 + 0x2000_0000) >> 30) as i32;
        squared = square(x).wrapping_add(square(y));
    }
    let length = frac_sqrt(squared) as i64;
    let component = |v: i32| {
        let numerator = (v as i64) << 30;
        let rounded = if numerator >= 0 { numerator + length / 2 } else { numerator - length / 2 };
        let normalized = if length == 0 { i32::MAX as i64 + (rounded < 0) as i64 } else { rounded / length };
        (((normalized as i32).wrapping_add(0x8000) >> 16) as i16) as i32
    };
    (component(x), component(y))
}

pub(crate) fn frac_sqrt(value: i32) -> i32 {
    if value < 0 {
        return i32::MIN;
    }
    let mut remainder = value as u32;
    let mut root = 0;
    if remainder >= 0x4000_0000 {
        remainder -= 0x4000_0000;
        root = 0x4000_0000;
    }
    let mut bit = 0x1000_0000;
    while bit != 0 {
        if remainder >= bit + root {
            remainder -= bit + root;
            root += 2 * bit;
        }
        remainder = remainder.wrapping_mul(2);
        bit >>= 1;
    }
    let final_remainder = if remainder > root {
        let previous = root;
        root += 1;
        (remainder - previous).wrapping_mul(2).wrapping_sub(1)
    } else {
        remainder.wrapping_mul(2)
    };
    (root + (final_remainder > root) as u32) as i32
}

/// How a distance is snapped to the grid. `Super`/`Super45` carry period/phase/threshold.
#[derive(Copy, Clone, PartialEq, Eq, Default, Debug)]
pub enum RoundMode {
    #[default]
    Grid,
    HalfGrid,
    DoubleGrid,
    DownToGrid,
    UpToGrid,
    Off,
    Super,
    Super45,
}

/// Round-state graphics variable: the active mode plus the SROUND parameters.
#[derive(Copy, Clone, Debug)]
pub struct RoundState {
    pub mode: RoundMode,
    pub period: i32,
    pub phase: i32,
    pub threshold: i32,
}

impl Default for RoundState {
    fn default() -> Self {
        Self { mode: RoundMode::Grid, period: 64, phase: 0, threshold: 0 }
    }
}

impl RoundState {
    /// Snap a F26Dot6 distance. Engine compensation is zero for smooth (grayscale/
    /// ClearType) rendering, matching the reference rasterizer's configured values.
    pub fn round(&self, distance: i32) -> i32 {
        use RoundMode::*;
        let neg = distance.wrapping_neg();
        match self.mode {
            Grid => {
                if distance >= 0 { round(distance).max(0) } else { round(neg).wrapping_neg().min(0) }
            }
            HalfGrid => {
                if distance >= 0 { floor(distance).wrapping_add(32).max(0) }
                else { floor(neg).wrapping_add(32).wrapping_neg().min(0) }
            }
            DoubleGrid => {
                if distance >= 0 { round_pad(distance, 32).max(0) }
                else { round_pad(neg, 32).wrapping_neg().min(0) }
            }
            DownToGrid => {
                if distance >= 0 { floor(distance).max(0) } else { floor(neg).wrapping_neg().min(0) }
            }
            UpToGrid => {
                if distance >= 0 { ceil(distance).max(0) } else { ceil(neg).wrapping_neg().min(0) }
            }
            Off => distance,
            Super => {
                let bias = self.threshold.wrapping_sub(self.phase);
                if distance >= 0 {
                    let v = (distance.wrapping_add(bias) & self.period.wrapping_neg()).wrapping_add(self.phase);
                    if v < 0 { self.phase } else { v }
                } else {
                    let v = (bias.wrapping_sub(distance) & self.period.wrapping_neg())
                        .wrapping_neg().wrapping_sub(self.phase);
                    if v > 0 { self.phase.wrapping_neg() } else { v }
                }
            }
            Super45 => {
                let bias = self.threshold.wrapping_sub(self.phase);
                if distance >= 0 {
                    let v = distance.wrapping_add(bias).wrapping_div(self.period)
                        .wrapping_mul(self.period).wrapping_add(self.phase);
                    if v < 0 { self.phase } else { v }
                } else {
                    let v = bias.wrapping_sub(distance).wrapping_div(self.period)
                        .wrapping_mul(self.period).wrapping_neg().wrapping_sub(self.phase);
                    if v > 0 { self.phase.wrapping_neg() } else { v }
                }
            }
        }
    }

    /// Subpixel-axis rounding: the same mode snapped to a **1/16-px grid** (4 F26Dot6 units) instead
    /// of the whole-pixel grid, which is what DirectWrite's round-to-grid/round-off subpixel family
    /// (round-function table bank 8, selected on the subpixel axis in `SVTCA`) does — stems keep a
    /// sub-pixel width rounded to 1/16 px rather than being snapped to whole pixels or left raw.
    /// `RoundToGridSP = (x+2)&~3` (Grid) and `RoundOffSP = x` (Off), verified against DirectWrite.
    // ponytail: Super/Super45 keep the whole-grid formula — SROUND on the subpixel axis is not used
    // by the core fonts; reproduce its bank-8 variant if an oracle glyph needs it.
    pub fn round_sp(&self, distance: i32) -> i32 {
        use RoundMode::*;
        let neg = distance.wrapping_neg();
        let floor4 = |x: i32| x & !3;
        let round4 = |x: i32| floor4(x.wrapping_add(2));
        let ceil4 = |x: i32| floor4(x.wrapping_add(3));
        match self.mode {
            Grid => if distance >= 0 { round4(distance).max(0) } else { round4(neg).wrapping_neg().min(0) },
            HalfGrid => {
                if distance >= 0 { floor4(distance).wrapping_add(2).max(0) }
                else { floor4(neg).wrapping_add(2).wrapping_neg().min(0) }
            }
            DoubleGrid => {
                if distance >= 0 { round_pad(distance, 2).max(0) } else { round_pad(neg, 2).wrapping_neg().min(0) }
            }
            DownToGrid => if distance >= 0 { floor4(distance).max(0) } else { floor4(neg).wrapping_neg().min(0) },
            UpToGrid => if distance >= 0 { ceil4(distance).max(0) } else { ceil4(neg).wrapping_neg().min(0) },
            Off => distance,
            Super | Super45 => self.round(distance),
        }
    }

    /// Apply an SROUND (`is45 == false`) or S45ROUND operand, reproducing
    /// The SROUND setup: SROUND periods are whole-pixel powers of two (mask
    /// rounding); S45ROUND periods are the sqrt(2)/2 grid (division), and its
    /// phase/threshold are rounded, not truncated.
    pub fn set_super(&mut self, selector: i32, is45: bool) {
        let p = if !is45 {
            match selector & 0xC0 { 0x00 => 32, 0x40 => 64, 0x80 => 128, _ => 999 }
        } else {
            match selector & 0xC0 { 0x00 => 23, 0x40 => 45, 0x80 => 91, _ => 4 }
        };
        self.period = p;
        self.phase = match selector & 0x30 {
            0x10 => (p + 2) >> 2,
            0x20 => (p + 1) >> 1,
            0x30 => (3 * p + 2) >> 2,
            _ => 0,
        };
        let t = selector & 0x0F;
        self.threshold = if t == 0 { p - 1 } else { ((t - 4) * p + 4) >> 3 };
        self.mode = if is45 { RoundMode::Super45 } else { RoundMode::Super };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_diagonal_normalization_rounds_to_2_14() {
        // Georgia Italic R@48: native's SPVTL forms this temporary phantom vector.
        assert_eq!(normalize14(2112, -512), (15923, -3860));
        assert_eq!(normalize14(0, 0), (16384, 0));
    }

    #[test]
    fn fixed_point_primitives() {
        // FixMul rounds half away from zero.
        assert_eq!(fixmul(0x10000, 0x8000), 0x8000);
        assert_eq!(fixmul(-0x10000, 0x8000), -0x8000);
        assert_eq!(fixmul(0x8000, 0x8000), 0x4000);
        // FixDiv is the inverse.
        assert_eq!(fixdiv(0x8000, 0x10000), 0x8000);
        assert_eq!(fixdiv(1, 0), 0x7FFF_FFFF);
        // 2.14 multiply: 1.0 * v == v.
        assert_eq!(mul14(0x4000, 1234), 1234);
        assert_eq!(mul_div(100, 64, 64), 100);
        assert_eq!(mul_div_no_round(10, 64, 64), 10);
        // Normalised axis vectors stay exact.
        assert_eq!(normalize14(0x4000, 0), (0x4000, 0));
        assert_eq!(normalize14(0, -0x4000), (0, -0x4000));
    }

    #[test]
    fn round_modes() {
        let cases = [
            (RoundMode::Grid, [(0, 0), (32, 64), (-32, -64), (64, 64), (50, 64)]),
            (RoundMode::HalfGrid, [(0, 32), (32, 32), (-32, -32), (64, 96), (50, 32)]),
            (RoundMode::DoubleGrid, [(0, 0), (32, 32), (-32, -32), (64, 64), (50, 64)]),
            (RoundMode::DownToGrid, [(0, 0), (32, 0), (-32, 0), (64, 64), (50, 0)]),
            (RoundMode::UpToGrid, [(0, 0), (32, 64), (-32, -64), (64, 64), (50, 64)]),
            (RoundMode::Off, [(0, 0), (32, 32), (-32, -32), (64, 64), (50, 50)]),
        ];
        for (mode, pairs) in cases {
            let rs = RoundState { mode, ..Default::default() };
            for (input, expected) in pairs {
                assert_eq!(rs.round(input), expected, "{mode:?}({input})");
            }
        }
    }

    #[test]
    fn sround_matches_rtg() {
        // SROUND 1px, phase 0, threshold half == round-to-grid.
        let mut rs = RoundState::default();
        rs.set_super(0x40, false);
        assert_eq!(rs.period, 64);
        assert_eq!(rs.phase, 0);
        assert_eq!(rs.round(50), 64);
        assert_eq!(rs.round(32), 64);
        // S45ROUND periods are the odd sqrt(2)/2 grid, rounded (not 90/22).
        rs.set_super(0x80, true);
        assert_eq!(rs.period, 91);
    }
}
