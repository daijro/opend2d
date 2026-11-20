//! The pure-math Direct2D exports — no COM, so they are portable and compile/run on every target.
//! Layouts are `repr(C)` matching `windows_numerics` / Direct2D, so the exported ABI is identical to
//! native. `D2D1GetGradientMeshInteriorPointsFromCoonsPatch` is deferred (its native formula is
//! unverified against WARP and nothing here exercises it) rather than fabricated.
//!
//! The COM entry points (`D2D1CreateFactory`/`CreateDevice`/`CreateDeviceContext`) live in
//! [`crate::win`] on every platform.

#![allow(non_snake_case)]

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}
/// Row-major 3×2 affine, memory order `m11,m12,m21,m22,m31,m32` (windows_numerics::Matrix3x2).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Mat3x2 {
    pub m: [f32; 6],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ColorF {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

const RAD: f32 = core::f32::consts::PI / 180.0;

#[unsafe(no_mangle)]
pub extern "system" fn D2D1Tan(angle: f32) -> f32 {
    // Native returns the system C runtime's `tanf` of the radian angle.
    d2d::crt::system_crt::tanf(angle)
}

#[unsafe(no_mangle)]
pub extern "system" fn D2D1Vec3Length(x: f32, y: f32, z: f32) -> f32 {
    // sqrtf is IEEE correctly-rounded, so the f32 op order below matches native bit-for-bit.
    (x * x + y * y + z * z).sqrt()
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn D2D1SinCos(angle: f32, s: *mut f32, c: *mut f32) {
    // Native computes the sine/cosine of the radian angle with the system C runtime's sinf/cosf.
    unsafe {
        *s = d2d::crt::system_crt::sinf(angle);
        *c = d2d::crt::system_crt::cosf(angle);
    }
}

/// Determinant of the 2×2 linear part.
fn det(m: &[f32; 6]) -> f32 {
    m[0] * m[3] - m[1] * m[2]
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn D2D1IsMatrixInvertible(matrix: *const Mat3x2) -> i32 {
    // d2d1!D2D1IsMatrixInvertible (0x1800…): invertible iff the determinant is non-zero AND its
    // f32 reciprocal is finite (a tiny det whose 1/det overflows to ∞ is reported non-invertible).
    let d = det(unsafe { &(*matrix).m });
    (d != 0.0 && (1.0 / d).is_finite()) as i32
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn D2D1InvertMatrix(matrix: *mut Mat3x2) -> i32 {
    // D2D1InvertMatrix: det = m22·m11 − m12·m21 (the `det` helper, commutative-
    // equal); reject on det == 0 or a non-finite reciprocal; the six entries use native's exact order.
    let m = unsafe { &mut (*matrix).m };
    let d = det(m);
    if d == 0.0 {
        return 0;
    }
    let inv = 1.0 / d;
    if !inv.is_finite() {
        return 0;
    }
    let [a, b, c, e, f, g] = *m; // m11,m12,m21,m22,dx,dy
    // Inverse of [[a,b,0],[c,e,0],[f,g,1]] (row-vector affine).
    *m = [
        e * inv,
        -b * inv,
        -c * inv,
        a * inv,
        (c * g - e * f) * inv,
        (b * f - a * g) * inv,
    ];
    1
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn D2D1MakeRotateMatrix(angle: f32, center: Vec2, matrix: *mut Mat3x2) {
    // Matches Direct2D: fmodf reduction into [0,360), exact sin/cos at the axis
    // angles, else the system C runtime's sinf/cosf — so 90°/180°/270° give clean matrices, not sinf round-off.
    unsafe {
        (*matrix).m = d2d::extras::print::make_rotate_matrix(angle, center.x, center.y);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn D2D1MakeSkewMatrix(anglex: f32, angley: f32, center: Vec2, matrix: *mut Mat3x2) {
    // Shear factors = the system C runtime's tanf of the radian angles (RAD = π/180 f32 = 0x3c8efa35, native's constant).
    let tx = d2d::crt::system_crt::tanf(anglex * RAD);
    let ty = d2d::crt::system_crt::tanf(angley * RAD);
    unsafe {
        (*matrix).m = [1.0, ty, tx, 1.0, -center.y * tx, -center.x * ty];
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn D2D1ComputeMaximumScaleFactor(matrix: *const Mat3x2) -> f32 {
    // The maximum-scale-factor computation, ported bit-for-bit in the core (shear-free fast path +
    // exact f32 op order) — forward so the result matches WARP to the last bit.
    d2d::extras::print::compute_maximum_scale_factor(unsafe { (*matrix).m })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn D2D1ConvertColorSpace(src: i32, dst: i32, color: *const ColorF) -> ColorF {
    let c = unsafe { *color };
    let [r, g, b, a] = d2d::bitmap::convert_color_space(src as u32, dst as u32, [c.r, c.g, c.b, c.a]);
    ColorF { r, g, b, a }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compose two row-vector affines `a·b`.
    fn mul(a: &[f32; 6], b: &[f32; 6]) -> [f32; 6] {
        [
            a[0] * b[0] + a[1] * b[2],
            a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2],
            a[2] * b[1] + a[3] * b[3],
            a[4] * b[0] + a[5] * b[2] + b[4],
            a[4] * b[1] + a[5] * b[3] + b[5],
        ]
    }

    #[test]
    fn invert_roundtrips_and_rejects_singular() {
        let orig = Mat3x2 { m: [2.0, 1.0, -0.5, 3.0, 10.0, -4.0] };
        let mut inv = orig;
        assert_eq!(unsafe { D2D1InvertMatrix(&mut inv) }, 1);
        assert_eq!(unsafe { D2D1IsMatrixInvertible(&orig) }, 1);
        let i = mul(&orig.m, &inv.m);
        let id = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        for k in 0..6 {
            assert!((i[k] - id[k]).abs() < 1e-5, "M·M⁻¹ entry {k} = {}", i[k]);
        }

        // Singular (det = 2·2 − 4·1 = 0): rejected, and reported non-invertible.
        let mut sing = Mat3x2 { m: [2.0, 4.0, 1.0, 2.0, 0.0, 0.0] };
        assert_eq!(unsafe { D2D1InvertMatrix(&mut sing) }, 0);
        assert_eq!(unsafe { D2D1IsMatrixInvertible(&sing) }, 0);
    }
}
