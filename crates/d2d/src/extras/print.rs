//! `ID2D1PrintControl` and `D2D1ComputeMaximumScaleFactor` (D2D 1.1/1.3).
//!
//! `D2D1ComputeMaximumScaleFactor` is pure math: for a shear-free matrix (`m12==0 && m21==0`) it is
//! `max(|m22|, |m11|)`; otherwise
//! the larger singular value of the 2×2 linear part — `sqrt((T + sqrt((r0-r1)² + 4·c²)) / 2)` with
//! `r0=m11²+m12²`, `r1=m21²+m22²`, `c=m21·m11+m22·m12`, `T=r0+r1`. Ported bit-for-bit in the native
//! f32 op order; `sqrtf` is IEEE correctly-rounded, so the result is identical on any CPU (the
//! answer is not CPU-vendor dependent — see warp-unorm8-sampler).
//!
//! `ID2D1PrintControl` is **blocked for CPU parity**: it serialises the command stream into an XPS
//! document package (a print target), so the offscreen BGRA harness has no comparable pixel output.
//! No scene; documented and stopped.

/// `D2D1ComputeMaximumScaleFactor(matrix)`: the maximum scale factor of the 3×2 transform's linear
/// part. `matrix` is `[m11, m12, m21, m22, dx, dy]`; translation is ignored.
pub fn compute_maximum_scale_factor(m: [f32; 6]) -> f32 {
    let (m11, m12, m21, m22) = (m[0], m[1], m[2], m[3]);
    if m12 == 0.0 && m21 == 0.0 {
        return m22.abs().max(m11.abs());
    }
    let r0 = m11 * m11 + m12 * m12;
    let c = m21 * m11 + m22 * m12;
    let r1 = m21 * m21 + m22 * m22;
    let disc = ((c * 4.0) * c + (r0 - r1) * (r0 - r1)).sqrt();
    ((disc + (r1 + r0)) * 0.5).sqrt()
}

/// `D2D1MakeRotateMatrix(angle, center)` — the rotate-about-`center` 3×2 affine
/// `[m11,m12,m21,m22,dx,dy]` (row-vector convention). Native reduces the degree
/// angle into `[0,360)` with `fmodf`, returns an exact sin/cos at the axis angles 0/90/180/270
/// (so those matrices have clean 0/±1 entries, not a sinf round-off), and otherwise uses the system C runtime's
/// `sinf`/`cosf` of `angle·0.017453292` (π/180 as f32, `0x3c8efa35`).
pub fn make_rotate_matrix(angle_deg: f32, cx: f32, cy: f32) -> [f32; 6] {
    // `%` on f32 is C `fmodf` (truncated remainder); normalise its signed result into [0,360).
    let mut a = angle_deg % 360.0;
    if a < 0.0 {
        a += 360.0;
    }
    let (s, c) = if a == 0.0 {
        (0.0, 1.0)
    } else if a == 90.0 {
        (1.0, 0.0)
    } else if a == 180.0 {
        (0.0, -1.0)
    } else if a == 270.0 {
        (-1.0, 0.0)
    } else {
        let rad = a * f32::from_bits(0x3c8e_fa35);
        (crate::crt::system_crt::sinf(rad), crate::crt::system_crt::cosf(rad))
    };
    [c, s, -s, c, cx - cx * c + cy * s, cy - cx * s - cy * c]
}

#[cfg(test)]
mod tests {
    use super::compute_maximum_scale_factor as msf;

    #[test]
    fn scale_factor() {
        assert_eq!(msf([1.0, 0.0, 0.0, 1.0, 5.0, 9.0]), 1.0); // identity (+ translation)
        assert_eq!(msf([2.0, 0.0, 0.0, 3.0, 0.0, 0.0]), 3.0); // axis scale -> max axis
        assert_eq!(msf([-4.0, 0.0, 0.0, 2.0, 0.0, 0.0]), 4.0); // sign ignored
        // Pure rotation preserves lengths: max scale factor 1.
        let (c, s) = (0.6f32, 0.8f32); // 3-4-5 rotation, exactly representable
        assert!((msf([c, s, -s, c, 0.0, 0.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rotate_axis_angles_are_exact() {
        use super::make_rotate_matrix as rot;
        // The four axis angles give clean 0/±1 matrices (no sinf round-off), about the origin.
        assert_eq!(rot(0.0, 0.0, 0.0), [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(rot(90.0, 0.0, 0.0), [0.0, 1.0, -1.0, 0.0, 0.0, 0.0]);
        assert_eq!(rot(180.0, 0.0, 0.0), [-1.0, 0.0, 0.0, -1.0, 0.0, 0.0]);
        assert_eq!(rot(270.0, 0.0, 0.0), [0.0, -1.0, 1.0, 0.0, 0.0, 0.0]);
        // Reduction: 450° == 90°, -90° == 270°.
        assert_eq!(rot(450.0, 0.0, 0.0), rot(90.0, 0.0, 0.0));
        assert_eq!(rot(-90.0, 0.0, 0.0), rot(270.0, 0.0, 0.0));
        // About a centre, 180° maps (cx,cy)·2 into the translation.
        assert_eq!(rot(180.0, 3.0, 4.0), [-1.0, 0.0, 0.0, -1.0, 6.0, 8.0]);
    }
}
