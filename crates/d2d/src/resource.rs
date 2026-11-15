//! Phase 1A resources and their domain. Every resource records the factory+device it was created
//! on; a draw that uses one from a different factory latches WRONG_FACTORY, from a different device
//! WRONG_RESOURCE_DOMAIN (the cross-domain rule, native's image-domain validation).
//!
//! Only `SolidColorBrush` is built here — the one brush the 1A drawing subset (Clear + rectangle)
//! needs. Gradient/bitmap/image brushes are `brush.rs`/`bitmap.rs` (phases 1D/1E).

use crate::pixels::premultiply;

/// The factory + device a resource belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Domain {
    pub factory_id: u64,
    pub device_id: u64,
}

/// `ID2D1SolidColorBrush`: a straight `D2D1_COLOR_F` plus an opacity multiplier. Created on a
/// device context, so it carries that context's domain.
#[derive(Clone, Copy, Debug)]
pub struct SolidColorBrush {
    pub(crate) domain: Domain,
    pub(crate) color: [f32; 4],
    pub(crate) opacity: f32,
}

impl SolidColorBrush {
    /// Straight `D2D1_COLOR_F` (ffirects agent: `ID2D1SolidColorBrush::GetColor`).
    pub fn color(&self) -> [f32; 4] {
        self.color
    }
    /// Opacity multiplier (ffirects agent: `ID2D1Brush::GetOpacity`).
    pub fn opacity(&self) -> f32 {
        self.opacity
    }
    /// The brush as a constant per-pixel source, for a layer opacity brush (native takes any brush's
    /// alpha there).
    pub fn as_pixel_brush(&self) -> crate::context::PixelBrush {
        crate::context::PixelBrush { domain: self.domain, brush: crate::brush::Brush::solid(self.color, self.opacity), def: None, image: None, fail: None }
    }
    /// Premultiplied BGRA for the rasterizer: opacity folds into the alpha channel before
    /// premultiplication (`color.a · opacity`), matching the brush-opacity modulation.
    pub(crate) fn premul(&self) -> [u8; 4] {
        let [r, g, b, a] = self.color;
        premultiply([r, g, b, a * self.opacity])
    }

    /// Premultiplied RGBA f32 for a float (HDR) target: no 8-bit quantisation (the quantise happens
    /// only at the BGRA8 store), matching WARP's float pipeline (`crate::float`).
    pub(crate) fn premul_f(&self) -> [f32; 4] {
        crate::float::premul_f(self.color, self.opacity)
    }
}
