//! Phase 1A factory and device: the top of the object graph and the source of resource-domain
//! identity. `Factory` ≈ `ID2D1Factory`, `Device` ≈ `ID2D1Device`. A `DeviceContext` is created
//! from a `Device`.

use crate::context::DeviceContext;
use crate::resource::Domain;
use crate::state::DrawingStateBlock;
use std::sync::atomic::{AtomicU64, Ordering};

// Monotonic ids stamp factories and devices so resources can be checked for domain at draw time
// (WRONG_FACTORY / WRONG_RESOURCE_DOMAIN). Rust ownership makes generations unnecessary here —
// those belong to the FFI/backend boundary (phase 3A), not this in-process API.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// `ID2D1Factory`. Single- vs multi-threaded only governs `GetMultithreadProtected` here; there is
/// no shared mutable state to lock yet, so the factory lock is a no-op (add it with shared resource
/// tables). `ponytail: no lock, add a Mutex when factory-owned tables are shared`.
pub struct Factory {
    id: u64,
    multithreaded: bool,
}

impl Factory {
    /// `D2D1CreateFactory(type)`. `multithreaded` selects `D2D1_FACTORY_TYPE_MULTI_THREADED`.
    pub fn new(multithreaded: bool) -> Self {
        Self { id: next_id(), multithreaded }
    }

    /// `Factory1::CreateDevice`. Portable: no DXGI device to wrap.
    pub fn create_device(&self) -> Device {
        Device { factory_id: self.id, id: next_id() }
    }

    /// `Factory::CreateDrawingStateBlock`: a zeroed block bound to this factory.
    pub fn create_drawing_state_block(&self) -> DrawingStateBlock {
        DrawingStateBlock { factory_id: self.id, state: Default::default() }
    }

    /// `ID2D1Multithread::GetMultithreadProtected`.
    pub fn is_multithreaded(&self) -> bool {
        self.multithreaded
    }

    /// This factory's identity, the key for its per-factory custom-effect registry (effects/custom.rs)
    /// and resource domain. Native scopes `RegisterEffectFromString` per `ID2D1Factory`.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }
}

/// `ID2D1Device`. Owns context creation and the device half of a resource's domain.
pub struct Device {
    factory_id: u64,
    id: u64,
}

impl Device {
    /// `Device::CreateDeviceContext`.
    pub fn create_device_context(&self) -> DeviceContext {
        DeviceContext::new(Domain { factory_id: self.factory_id, device_id: self.id })
    }
}
