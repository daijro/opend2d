// Ported from dotnet/wpf (MIT, Copyright (c) .NET Foundation and Contributors):
// src/Microsoft.DotNet.Wpf/src/WpfGfx/core/geometry/ExactArithmetic.{cpp,h}. See THIRD-PARTY-NOTICES.

//! The scanner's exact layer: a fixed-width signed integer wide enough for every exact predicate on
//! workspace coordinates (|v| < 2^49). The deepest one compares two segment intersections' rational
//! coordinates, about 2^245 in magnitude. WPF sizes its digit classes for 30-bit coordinates; the
//! predicates are exact either way, so this keeps one wide type instead.

use std::cmp::Ordering;

const LIMBS: usize = 6; // 384 bits of magnitude

/// A signed integer as sign and 384-bit magnitude (little-endian limbs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Big {
    neg: bool,
    mag: [u64; LIMBS],
}

impl Big {
    pub const ZERO: Big = Big { neg: false, mag: [0; LIMBS] };

    pub fn from_i128(v: i128) -> Big {
        let m = v.unsigned_abs();
        let mut mag = [0; LIMBS];
        mag[0] = m as u64;
        mag[1] = (m >> 64) as u64;
        Big { neg: v < 0, mag }
    }

    pub fn is_zero(&self) -> bool {
        self.mag.iter().all(|&l| l == 0)
    }

    /// -1, 0 or 1.
    pub fn signum(&self) -> i32 {
        if self.is_zero() {
            0
        } else if self.neg {
            -1
        } else {
            1
        }
    }

    fn cmp_mag(a: &[u64; LIMBS], b: &[u64; LIMBS]) -> Ordering {
        for i in (0..LIMBS).rev() {
            match a[i].cmp(&b[i]) {
                Ordering::Equal => continue,
                o => return o,
            }
        }
        Ordering::Equal
    }

    fn mag_plus(a: &[u64; LIMBS], b: &[u64; LIMBS]) -> [u64; LIMBS] {
        let mut out = [0; LIMBS];
        let mut carry = 0u128;
        for i in 0..LIMBS {
            let s = a[i] as u128 + b[i] as u128 + carry;
            out[i] = s as u64;
            carry = s >> 64;
        }
        debug_assert!(carry == 0, "exact layer overflow");
        out
    }

    /// `a − b` for `a ≥ b`.
    fn mag_minus(a: &[u64; LIMBS], b: &[u64; LIMBS]) -> [u64; LIMBS] {
        let mut out = [0; LIMBS];
        let mut borrow = 0i128;
        for i in 0..LIMBS {
            let d = a[i] as i128 - b[i] as i128 - borrow;
            out[i] = d as u64;
            borrow = i128::from(d < 0);
        }
        out
    }

    pub fn neg(self) -> Big {
        Big { neg: !self.neg && !self.is_zero(), ..self }
    }

    pub fn add(self, o: Big) -> Big {
        if self.neg == o.neg {
            return Big { neg: self.neg, mag: Big::mag_plus(&self.mag, &o.mag) };
        }
        match Big::cmp_mag(&self.mag, &o.mag) {
            Ordering::Equal => Big::ZERO,
            Ordering::Greater => Big { neg: self.neg, mag: Big::mag_minus(&self.mag, &o.mag) },
            Ordering::Less => Big { neg: o.neg, mag: Big::mag_minus(&o.mag, &self.mag) },
        }
    }

    pub fn sub(self, o: Big) -> Big {
        self.add(o.neg())
    }

    pub fn mul(self, o: Big) -> Big {
        let mut out = [0u64; LIMBS];
        for i in 0..LIMBS {
            if self.mag[i] == 0 {
                continue;
            }
            let mut carry = 0u128;
            for j in 0..LIMBS - i {
                let t = self.mag[i] as u128 * o.mag[j] as u128 + out[i + j] as u128 + carry;
                out[i + j] = t as u64;
                carry = t >> 64;
            }
        }
        let r = Big { neg: self.neg != o.neg, mag: out };
        if r.is_zero() { Big::ZERO } else { r }
    }

    pub fn cmp(&self, o: &Big) -> Ordering {
        match (self.signum(), o.signum()) {
            (a, b) if a != b => a.cmp(&b),
            (0, _) => Ordering::Equal,
            (1, _) => Big::cmp_mag(&self.mag, &o.mag),
            _ => Big::cmp_mag(&o.mag, &self.mag),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_matches_i128_and_extends_past_it() {
        let v = [0i128, 1, -1, 7, -12345678901234567, 1 << 100, -(1 << 100) + 3];
        for &a in &v {
            for &b in &v {
                let (x, y) = (Big::from_i128(a), Big::from_i128(b));
                assert_eq!(x.add(y), Big::from_i128(a + b));
                assert_eq!(x.sub(y), Big::from_i128(a - b));
                assert_eq!(x.cmp(&y), a.cmp(&b));
                if a.unsigned_abs() < 1 << 62 && b.unsigned_abs() < 1 << 62 {
                    assert_eq!(x.mul(y), Big::from_i128(a * b));
                }
            }
        }
        // (2^100)^3 = 2^300: past i128, still ordered and signed.
        let p = Big::from_i128(1 << 100);
        let c = p.mul(p).mul(p);
        assert_eq!(c.signum(), 1);
        assert_eq!(c.neg().cmp(&c), Ordering::Less);
        assert_eq!(c.sub(c), Big::ZERO);
        assert_eq!(c.cmp(&p.mul(p)), Ordering::Greater);
    }
}
