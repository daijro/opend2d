//! Frozen cross-crate contracts. Only the integration/0A owner
//! changes these; other crates request a change rather than editing this file. They
//! define the rules and the backend boundary between crates.

use crate::abi::{D2D1_ALPHA_MODE, DXGI_FORMAT};
use windows_core::{GUID, HRESULT};

/// Stable handle to a backend-owned resource. `generation` is bumped when a slot is reused, so a
/// stale id compares unequal to the live one instead of aliasing it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ResourceId {
    pub index: u32,
    pub generation: u32,
}

/// Shape, pixel format and resolution of an image — no pixels. The `format`/`alpha_mode` use the
/// generated ABI enums so every crate speaks the same vocabulary.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ImageDesc {
    pub width: u32,
    pub height: u32,
    pub format: DXGI_FORMAT,
    pub alpha_mode: D2D1_ALPHA_MODE,
    pub dpi_x: f32,
    pub dpi_y: f32,
}

/// The CPU-authoritative pixel buffer the renderer reads and writes. `stride` is the row pitch in
/// bytes (`>= width * bytes-per-pixel`); `data.len()` is `stride * height`.
#[derive(Clone, PartialEq, Debug)]
pub struct PixelBuffer {
    pub desc: ImageDesc,
    pub stride: usize,
    pub data: Vec<u8>,
}

/// One retained drawing command, executed in order into a target [`PixelBuffer`]. The backend
/// boundary is: an ordered `&[Command]` plus resources in, typed pixel buffers out. Variants are
/// added by contract change as drawing phases land; these are the phase-0A seed.
#[derive(Clone, PartialEq, Debug)]
pub enum Command {
    /// 3×2 affine transform (row-major: `[m11, m12, m21, m22, dx, dy]`).
    SetTransform([f32; 6]),
    /// Clear to a premultiplied RGBA color, or to transparent (`None`).
    Clear(Option<[f32; 4]>),
    /// Fill a rectangle (`[left, top, right, bottom]`, DIPs) with a brush resource.
    FillRectangle { rect: [f32; 4], brush: ResourceId },
    /// Draw an image at a target offset (DIPs).
    DrawImage { image: ResourceId, target_offset: [f32; 2] },
}

/// A node in the effect graph: an effect CLSID and its ordered input resources. Effect property
/// values are added by contract change when the effect phases need them.
#[derive(Clone, PartialEq, Debug)]
pub struct EffectNode {
    pub clsid: GUID,
    pub inputs: Vec<ResourceId>,
}

/// A pixel/compute shader program: its GUID and bytecode (DXBC on Windows). Populated at runtime by
/// callers — never from `ref/`.
#[derive(Clone, PartialEq, Debug)]
pub struct ShaderProgram {
    pub id: GUID,
    pub bytecode: Vec<u8>,
}

/// A rendering failure: the HRESULT and the D2D tag pair reported at `EndDraw`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RenderError {
    pub hresult: HRESULT,
    pub tags: (u64, u64),
}

impl core::fmt::Display for RenderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "render error {:?} (tags {}, {})", self.hresult, self.tags.0, self.tags.1)
    }
}
impl std::error::Error for RenderError {}
