//! Phase 1A state vocabulary: the HRESULTs this area returns, the per-context draw-state bundle,
//! and the factory-owned drawing-state block.

use windows_core::HRESULT;

const fn hr(code: u32) -> HRESULT {
    HRESULT(code as i32)
}

// HRESULTs the 1A state machine returns. The
// spec lists more for this area, but ones with no path in the CPU subset — RECREATE_TARGET
// (device-lost), UNSUPPORTED_PIXEL_FORMAT / UNSUPPORTED_OPERATION (target creation), E_NOINTERFACE
// (QI), LAYER_ALREADY_IN_USE (explicit layer reuse) — belong to the phases that can reach them.
pub const S_OK: HRESULT = hr(0x0000_0000);
pub const E_INVALIDARG: HRESULT = hr(0x8007_0057);
pub const E_NOTIMPL: HRESULT = hr(0x8000_4001);
pub const WRONG_STATE: HRESULT = hr(0x8899_0001);
pub const WRONG_FACTORY: HRESULT = hr(0x8899_0012);
pub const POP_DID_NOT_MATCH_PUSH: HRESULT = hr(0x8899_0014);
pub const WRONG_RESOURCE_DOMAIN: HRESULT = hr(0x8899_0015);
pub const PUSH_POP_UNBALANCED: HRESULT = hr(0x8899_0016);
/// An image drawn under a transform needs an intermediate larger than Direct2D renders one at.
pub const INTERMEDIATE_TOO_LARGE: HRESULT = hr(0x8899_0027);
/// `WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT` — native rejects an A8 (alpha-only) bitmap used as a colour
/// source (DrawBitmap/image brush/bitmap brush/sprite batch), latching this at the draw.
pub const UNSUPPORTED_PIXEL_FORMAT: HRESULT = hr(0x8898_2F80);

/// The mutable render state a context carries and a [`DrawingStateBlock`] round-trips: the fields
/// `SaveDrawingState` copies. DPI and text-rendering params are
/// context-level and are *not* part of the block, matching native.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DrawState {
    /// World transform (row-major `[m11, m12, m21, m22, dx, dy]`).
    pub transform: [f32; 6],
    /// `D2D1_ANTIALIAS_MODE` (0 PER_PRIMITIVE, 1 ALIASED).
    pub aa: u32,
    /// `D2D1_TEXT_ANTIALIAS_MODE` (0 DEFAULT .. 3 ALIASED).
    pub text_aa: u32,
    /// `D2D1_PRIMITIVE_BLEND` (0 SOURCE_OVER .. 4 MAX).
    pub blend: u32,
    /// `D2D1_UNIT_MODE` (0 DIPS, 1 PIXELS).
    pub unit: u32,
    /// Current tags (SetTags target / GetTags source) — not the error tags.
    pub tags: (u64, u64),
}

impl Default for DrawState {
    fn default() -> Self {
        Self {
            transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            aa: 0,
            text_aa: 0,
            blend: 0,
            unit: 0,
            tags: (0, 0),
        }
    }
}

/// A `ID2D1DrawingStateBlock`: a snapshot of [`DrawState`] bound to the factory that created it.
/// Save/Restore are silent no-ops across factories (the same-factory guard).
#[derive(Clone, Copy, Debug)]
pub struct DrawingStateBlock {
    pub(crate) factory_id: u64,
    pub(crate) state: DrawState,
}

impl DrawingStateBlock {
    /// The block's `D2D1_DRAWING_STATE_DESCRIPTION` fields — `(transform, antialiasMode,
    /// textAntialiasMode, (tag1, tag2))` — for `GetDescription` across the C-API boundary.
    pub fn description(&self) -> ([f32; 6], u32, u32, (u64, u64)) {
        (self.state.transform, self.state.aa, self.state.text_aa, self.state.tags)
    }
    /// Set the block's description fields (`SetDescription`); leaves the (not-in-description) blend
    /// and unit mode untouched, as native does.
    pub fn set_description(&mut self, transform: [f32; 6], aa: u32, text_aa: u32, tags: (u64, u64)) {
        self.state.transform = transform;
        self.state.aa = aa;
        self.state.text_aa = text_aa;
        self.state.tags = tags;
    }
    /// The `D2D1_DRAWING_STATE_DESCRIPTION1` fields — the base four plus `(primitiveBlend, unitMode)`
    /// (`ID2D1DrawingStateBlock1::GetDescription`).
    pub fn description1(&self) -> ([f32; 6], u32, u32, (u64, u64), u32, u32) {
        (self.state.transform, self.state.aa, self.state.text_aa, self.state.tags, self.state.blend, self.state.unit)
    }
    /// Set all six `..._DESCRIPTION1` fields (`ID2D1DrawingStateBlock1::SetDescription`).
    pub fn set_description1(&mut self, transform: [f32; 6], aa: u32, text_aa: u32, tags: (u64, u64), blend: u32, unit: u32) {
        self.set_description(transform, aa, text_aa, tags);
        self.state.blend = blend;
        self.state.unit = unit;
    }
}
