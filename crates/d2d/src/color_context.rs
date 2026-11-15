//! `ID2D1ColorContext` / `ID2D1ColorContext1` and the colour-transform engine that native d2d1's
//! `CColorManagement` effect (and, for the same named spaces, WIC) run on WARP.
//!
//! Native parses both colour contexts into a module pipeline (`CColorProfileParser` →
//! `CGPUColorTranslator`). For the sRGB/scRGB named contexts it takes the `CSRGBToLinearTranslator`
//! fast path: a **1024-entry 16-bit sRGB→linear LUT** sampled
//! in a shader at texcoord `(x·1023 + 0.5)/1024`, and its numeric inverse (`ColorUtils::InvertCurve`)
//! for the linear→sRGB direction. sRGB and scRGB share Rec.709 primaries and the D65 white point,
//! so no colour matrix is involved. Reproduced byte-for-byte against WARP
//! (`conformance/oracle/win11-26100/cms-0*`); rendering intent and quality do not change the result
//! for this space (verified: `cms-01`/`06`/`07`/`08` are identical).
//!
//! The sRGB and scRGB spaces are reproduced whether they arrive as the named ICC contexts or as the
//! `RGB_FULL_G22/G10_NONE_P709` DXGI colour spaces: native maps those two DXGI spaces onto the same
//! named contexts, so `CreateColorContextFromDxgiColorSpace(G22 P709)` → scRGB applies the exact same
//! sRGB→linear curve as the named context (verified byte-for-byte: `cms2-01/02/03`). A
//! `CreateColorContextFromSimpleColorProfile` context is a different animal — native parses it into a
//! matrix-only (no-TRC) colour-processing module that cannot be linked into the sRGB/scRGB named
//! pipeline, so a ColorManagement transform touching one fails `EndDraw` with `E_INVALIDARG` on WARP
//! (`cms2-04/05`); this is reproduced as a construction failure, not a colour transform.
//!
//! RGB matrix/TRC ICC profiles use the native NORMAL/FAST 32³/16³ byte LUT pipeline or the BEST
//! matrix and 1024-entry UNORM16 curves. Absolute colorimetric intent includes the media-white
//! scaling, and the straight-alpha LUT path retains WARP's truncating store. LUT/CLUT, CMYK and gray
//! profiles use the native CPU/GPU pipeline in `icc::lut`.

use d2d_types::abi::*;
use std::sync::{Arc, LazyLock};

mod icc;

/// Device colour space (managed `DeviceColorSpace`): what drives the transform. `Custom` is any ICC
/// profile not recognised as the canonical sRGB or scRGB.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Space {
    Srgb,
    ScRgb,
    Custom,
}

/// Native (`ColorProfileHelpers::AreProfileBytesLogicallyEqual`) reports a loaded ICC profile's
/// colour space as SRGB when it is *logically* the system sRGB profile. Reproduced by comparing an
/// FNV-1a 64 digest of the profile with the volatile ICC header fields zeroed — the profile/creation
/// date-time (24..36), rendering intent + flags (44..48, 64..68) and the profile ID (84..100), which
/// "logically equal" ignores — against the digest of Windows' `sRGB Color Space Profile.icm`
/// (3144 bytes). (scRGB has no system .icm file, so only sRGB is recognised; add its digest if one
/// surfaces.) Only the digest is stored, not Microsoft's profile bytes.
fn recognize_profile_space(profile: &[u8]) -> Space {
    const SRGB_LEN: usize = 3144;
    const SRGB_FNV: u64 = 0x065c_981f_227e_4765;
    if profile.len() != SRGB_LEN || profile.len() < 100 {
        return Space::Custom;
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (i, &b) in profile.iter().enumerate() {
        let v = if matches!(i, 24..=35 | 44..=47 | 64..=67 | 84..=99) { 0 } else { b };
        h ^= v as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    if h == SRGB_FNV { Space::Srgb } else { Space::Custom }
}

/// A CIE-xy simple colour profile (`CreateColorContextFromSimpleColorProfile`): primaries, white
/// point and a `D2D1_GAMMA1` selector.
#[derive(Clone, Copy, Debug)]
pub struct SimpleProfile {
    pub red: [f32; 2],
    pub green: [f32; 2],
    pub blue: [f32; 2],
    pub white: [f32; 2],
    pub gamma: u32,
}

/// `ID2D1ColorContext` (+ `ID2D1ColorContext1`). Stores the profile bytes and the classification the
/// transform engine needs; the five `Create*` constructors mirror `ManagedDeviceColorContext`.
pub struct ColorContext {
    ctype: D2D1_COLOR_CONTEXT_TYPE,
    space: Space,
    dxgi: DXGI_COLOR_SPACE_TYPE,
    simple: Option<SimpleProfile>,
    profile: Vec<u8>,
}

impl ColorContext {
    /// `CreateColorContext(D2D1_COLOR_SPACE_SRGB | _SCRGB, nullptr)` — a named ICC context. The
    /// canonical profile bytes native returns are Microsoft's and are not shipped here, so
    /// [`Self::get_profile`] is empty for a named context (it is exact for a custom one).
    pub fn from_color_space(space: D2D1_COLOR_SPACE) -> Self {
        let (sp, dxgi) = if space == D2D1_COLOR_SPACE_SCRGB {
            (Space::ScRgb, DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709)
        } else {
            (Space::Srgb, DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709)
        };
        Self { ctype: D2D1_COLOR_CONTEXT_TYPE_ICC, space: sp, dxgi, simple: None, profile: Vec::new() }
    }

    /// `CreateColorContext(D2D1_COLOR_SPACE_CUSTOM, profile)` — an ICC context from raw profile
    /// bytes, kept verbatim for [`Self::get_profile`]. Native's `ColorProfileHelpers::
    /// AreProfileBytesLogicallyEqual` recognises a loaded profile that is logically the sRGB (or
    /// scRGB) profile and reports its colour space as SRGB/ScRGB, not CUSTOM; reproduced via
    /// [`recognize_profile_space`]. The DXGI space + managed classification follow, so a recognised
    /// sRGB context color-manages like the named one.
    pub fn from_profile_bytes(profile: &[u8]) -> Self {
        let (space, dxgi) = match recognize_profile_space(profile) {
            Space::Srgb => (Space::Srgb, DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709),
            Space::ScRgb => (Space::ScRgb, DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709),
            _ => (Space::Custom, DXGI_COLOR_SPACE_CUSTOM),
        };
        Self {
            ctype: D2D1_COLOR_CONTEXT_TYPE_ICC,
            space,
            dxgi,
            simple: None,
            profile: profile.to_vec(),
        }
    }

    /// `CreateColorContextFromDxgiColorSpace`. G22/G10 P709 map to sRGB/scRGB (managed), the other
    /// supported spaces to `Custom`.
    pub fn from_dxgi_color_space(cs: DXGI_COLOR_SPACE_TYPE) -> Self {
        let space = if cs == DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709 {
            Space::Srgb
        } else if cs == DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709 {
            Space::ScRgb
        } else {
            Space::Custom
        };
        Self { ctype: D2D1_COLOR_CONTEXT_TYPE_DXGI, space, dxgi: cs, simple: None, profile: Vec::new() }
    }

    /// `CreateColorContextFromSimpleColorProfile`.
    pub fn from_simple_color_profile(p: SimpleProfile) -> Self {
        Self {
            ctype: D2D1_COLOR_CONTEXT_TYPE_SIMPLE,
            space: Space::Custom,
            dxgi: DXGI_COLOR_SPACE_CUSTOM,
            simple: Some(p),
            profile: Vec::new(),
        }
    }

    /// `ID2D1ColorContext1::GetColorContextType`.
    pub fn color_context_type(&self) -> D2D1_COLOR_CONTEXT_TYPE {
        self.ctype
    }
    /// `ID2D1ColorContext::GetProfile` (+ `GetProfileSize` = `.len()`).
    pub fn get_profile(&self) -> &[u8] {
        &self.profile
    }
    /// `ID2D1ColorContext1::GetDXGIColorSpace`.
    pub fn dxgi_color_space(&self) -> DXGI_COLOR_SPACE_TYPE {
        self.dxgi
    }
    /// `ID2D1ColorContext1::GetSimpleColorProfile`.
    pub fn simple_color_profile(&self) -> Option<&SimpleProfile> {
        self.simple.as_ref()
    }
    pub fn space(&self) -> Space {
        self.space
    }
    /// Whether this is a `CreateColorContextFromSimpleColorProfile` context. Native parses such a
    /// profile into a matrix-only module (no TRC) that cannot be linked into the sRGB/scRGB named
    /// pipeline, so a ColorManagement transform touching one fails `EndDraw` with `E_INVALIDARG`
    /// (reproduced: `cms2-04/05`). DXGI G22/G10 P709 contexts, by contrast, behave like named sRGB/scRGB.
    pub fn is_simple(&self) -> bool {
        self.ctype == D2D1_COLOR_CONTEXT_TYPE_SIMPLE
    }
}

// --- the sRGB↔linear transform engine (native CSRGBToLinearTranslator) -------------------------

const N: usize = 1024;

/// sRGB → linear (IEC 61966-2-1), f64 for the LUT build.
fn srgb_to_linear(x: f64) -> f64 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        crate::crt::pow((x + 0.055) / 1.055, 2.4)
    }
}

/// The sRGB→linear LUT: `round(srgb_to_linear(i/1023) · 65535)`, the 1024-entry 16-bit forward curve.
static FORWARD: LazyLock<[u16; N]> = LazyLock::new(|| {
    std::array::from_fn(|i| {
        let v = srgb_to_linear(i as f64 / (N - 1) as f64) * 65535.0;
        (v + 0.5).floor().clamp(0.0, 65535.0) as u16
    })
});

/// The inverse LUT: linear → sRGB as the numeric inverse of the forward curve.
/// This is the 16.16 fixed-point inversion native runs, not the analytic `1.055·x^(1/2.4)−0.055`
/// (the two differ by up to ~1 LSB, which is why the analytic form is not byte-exact for the
/// backward direction — `cms-02`).
static BACKWARD: LazyLock<[u16; N]> = LazyLock::new(|| invert_curve(&FORWARD));

fn invert_curve(src: &[u16; N]) -> [u16; N] {
    let mut dst = [0u16; N];
    let (src_len, dst_len) = (N as i64, N as i64);
    // target(j) = round(j·65535 / (dstLen−1)) via native's `((dstLen−1)/2 + 65535·j) / (dstLen−1)`.
    let target = |j: i64| (((dst_len - 1) / 2) + 65535 * j) / (dst_len - 1);
    let mut j: i64 = 0;
    let s0 = src[0] as i64;
    let mut v = target(0);
    while j < dst_len && v <= s0 {
        dst[j as usize] = 0;
        j += 1;
        v = target(j);
    }
    for i in 1..src_len as usize {
        let (prev, cur) = (src[i - 1] as i64, src[i] as i64);
        while j < dst_len && v <= cur {
            let mut out = ((i as i64 - 1) << 16) / (src_len - 1);
            if cur != prev {
                out += ((v - prev) << 16) / ((src_len - 1) * (cur - prev));
                if out & 0x1_0000 != 0 {
                    out = 0xFFFF;
                }
            }
            dst[j as usize] = (out & 0xFFFF) as u16;
            j += 1;
            v = target(j);
        }
    }
    while j < dst_len {
        dst[j as usize] = 0xFFFF;
        j += 1;
    }
    dst
}

/// Sample a 16-bit LUT as the shader does on WARP: clamp to `[0,1]`, texcoord `(x·1023 + 0.5)/1024`,
/// the sampler's Q8 address (`floor`, the 1024-texel width being a power of two) and its float lerp
/// `a + (b − a)·w` over the normalised texels. Bit-exact against native's Float32 output (a zone plate
/// and a 0..1 ramp, scRGB → sRGB); an 8-bit store hides the 1/256 weight, which the float pipelines see.
fn sample(lut: &[u16; N], x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    let u = (x * (N - 1) as f32 + 0.5) / N as f32;
    let q = ((u * N as f32 - 0.5) * 256.0).floor() as i32;
    let at = |i: i32| lut[i.clamp(0, N as i32 - 1) as usize] as f32 / 65535.0;
    let (a, b, w) = (at(q >> 8), at((q >> 8) + 1), (q & 0xff) as f32 / 256.0);
    a + (b - a) * w
}

/// A colour transform between two [`ColorContext`]s: a per-channel 1-D LUT (`None` = identity) plus
/// the alpha mode. Premultiplied RGBA f32 in and out, matching the effect-graph pixel convention.
#[derive(Clone)]
pub struct ColorTransform {
    lut: Option<&'static [u16; N]>,
    icc: Option<Arc<icc::Transform>>,
    straight: bool,
}

impl ColorTransform {
    /// Build a NORMAL-quality transform with perceptual source and destination intents.
    /// Unsupported profile models return `None` so COM can fail realization.
    pub fn new(src: &ColorContext, dst: &ColorContext, straight_alpha: bool) -> Option<Self> {
        Self::with_intents(src, dst, straight_alpha, [0, 0])
    }

    pub fn with_intents(src: &ColorContext, dst: &ColorContext, straight_alpha: bool, intents: [u32; 2]) -> Option<Self> {
        Self::with_options(src, dst, straight_alpha, intents, 1)
    }

    pub fn with_options(src: &ColorContext, dst: &ColorContext, straight_alpha: bool, intents: [u32; 2], quality: u32) -> Option<Self> {
        let srgb_family = |c: &ColorContext| {
            c.ctype != D2D1_COLOR_CONTEXT_TYPE_SIMPLE && matches!(c.space, Space::Srgb | Space::ScRgb)
        };
        if !srgb_family(src) || !srgb_family(dst) {
            return Some(Self { lut: None, icc: Some(Arc::new(icc::Transform::new(src, dst, intents, quality)?)), straight: straight_alpha });
        }
        let lut = match (src.space, dst.space) {
            (Space::Srgb, Space::ScRgb) => Some(&*FORWARD),
            (Space::ScRgb, Space::Srgb) => Some(&*BACKWARD),
            _ => None, // same space → identity
        };
        Some(ColorTransform { lut, icc: None, straight: straight_alpha })
    }

    /// An identity transform (no colour change). Used to model native where the transform is
    /// constructed but realization later fails (a SIMPLE-profile source/destination).
    pub fn identity(straight_alpha: bool) -> Self {
        ColorTransform { lut: None, icc: None, straight: straight_alpha }
    }

    /// Transform one premultiplied RGBA f32 pixel. Premultiplied mode unpremultiplies, applies the
    /// per-channel curve and re-premultiplies; straight mode applies the curve to the premultiplied
    /// rgb directly and keeps alpha (native `straightAlpha`).
    pub fn apply(&self, p: [f32; 4]) -> [f32; 4] {
        if let Some(t) = &self.icc { return t.apply(p, self.straight); }
        let Some(lut) = self.lut else { return p };
        let a = p[3];
        if self.straight {
            [sample(lut, p[0]), sample(lut, p[1]), sample(lut, p[2]), a]
        } else {
            let s = if a != 0.0 { [p[0] / a, p[1] / a, p[2] / a] } else { [0.0; 3] };
            [sample(lut, s[0]) * a, sample(lut, s[1]) * a, sample(lut, s[2]) * a, a]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_lut_matches_native_anchors() {
        // The native sRGB→linear LUT: toe is linear (step 5), top rolls off to 0xffff.
        assert_eq!(FORWARD[0], 0);
        assert_eq!(FORWARD[1], 5);
        assert_eq!(FORWARD[1022], 0xff6d);
        assert_eq!(FORWARD[1023], 0xffff);
    }

    #[test]
    fn srgb_to_scrgb_mid_gray() {
        // sRGB 128/255 → linear ≈ 0.2159 → 55/255 (cms-01, premultiplied, opaque).
        let srgb = ColorContext::from_color_space(D2D1_COLOR_SPACE_SRGB);
        let scrgb = ColorContext::from_color_space(D2D1_COLOR_SPACE_SCRGB);
        let t = ColorTransform::new(&srgb, &scrgb, false).unwrap();
        let g = 128.0 / 255.0;
        let out = t.apply([g, g, g, 1.0]);
        let q = (out[0] * 255.0).round() as u8;
        assert_eq!(q, 55);
    }

    #[test]
    fn identity_and_unsupported() {
        let srgb = ColorContext::from_color_space(D2D1_COLOR_SPACE_SRGB);
        let srgb2 = ColorContext::from_color_space(D2D1_COLOR_SPACE_SRGB);
        // same space → identity passthrough
        let id = ColorTransform::new(&srgb, &srgb2, false).unwrap();
        assert_eq!(id.apply([0.3, 0.6, 0.9, 1.0]), [0.3, 0.6, 0.9, 1.0]);
        // a custom ICC profile is not reproduced → None
        let custom = ColorContext::from_profile_bytes(&[0u8; 4]);
        assert!(ColorTransform::new(&srgb, &custom, false).is_none());
    }

    #[test]
    fn dxgi_g22_g10_are_the_named_contexts() {
        // Native maps the P709 G22/G10 DXGI spaces onto named sRGB/scRGB (cms2-01/02/03): G22→scRGB
        // applies the sRGB→linear curve, G10→scRGB and G22→sRGB are identity.
        let scrgb = ColorContext::from_color_space(D2D1_COLOR_SPACE_SCRGB);
        let g22 = ColorContext::from_dxgi_color_space(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709);
        let g10 = ColorContext::from_dxgi_color_space(DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709);
        assert!(!g22.is_simple() && !g10.is_simple());
        let fwd = ColorTransform::new(&g22, &scrgb, false).unwrap();
        let named = ColorTransform::new(&ColorContext::from_color_space(D2D1_COLOR_SPACE_SRGB), &scrgb, false).unwrap();
        assert_eq!(fwd.apply([0.5, 0.5, 0.5, 1.0]), named.apply([0.5, 0.5, 0.5, 1.0]));
        assert_eq!(ColorTransform::new(&g10, &scrgb, false).unwrap().apply([0.3, 0.6, 0.9, 1.0]), [0.3, 0.6, 0.9, 1.0]);
    }

    #[test]
    fn simple_profile_is_rejected() {
        // Native fails EndDraw with E_INVALIDARG for a simple-profile context (cms2-04/05): no
        // reproducible transform, flagged so the effect can surface the failure.
        let simple = ColorContext::from_simple_color_profile(SimpleProfile {
            red: [0.64, 0.33], green: [0.30, 0.60], blue: [0.15, 0.06], white: [0.3127, 0.3290], gamma: 0,
        });
        let scrgb = ColorContext::from_color_space(D2D1_COLOR_SPACE_SCRGB);
        assert!(simple.is_simple());
        assert!(ColorTransform::new(&simple, &scrgb, false).is_none());
    }
}
