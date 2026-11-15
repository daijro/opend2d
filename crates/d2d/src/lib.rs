//! `d2d`: safe resource/API semantics and the CPU renderer (phases 1–2). Phase 0A only wires the
//! crate up and re-exports the frozen contracts so downstream code shares one vocabulary.

pub use d2d_types::{abi, contracts, interfaces};

pub mod brush;
pub mod effects;
pub mod float;
pub mod geometry;
pub mod pixels;
pub use d2d_shader::warpmath; // WARP-exact shader maths; lives in the leaf crate so d2d-shader needs no d2d
// d2d1's CPU-side CRT maths (ucrtbase sinf/cosf/…) split into a dependency-free crate so pdn-core can
// share it; re-exported so `d2d::crt` and in-crate `crate::crt::…` paths are unchanged.
pub use crt;


// Colour management: ID2D1ColorContext/ColorContext1 + the sRGB↔scRGB transform engine.
pub mod color_context;

// Phase 1E: bitmaps, imaging and the reusable texture sampler.
pub mod bitmap;
pub mod sampling;
pub mod wic;

// WARP's Direct3D 11 general triangle pipeline (gradient meshes, ink).
pub mod warp;

// Direct2D 1.1–1.3 extras: sprite batches, ink, gradient meshes, image sources, SVG, LUT3D.
pub mod extras;

// Phase 2A: layers, clips and composition.
pub mod layer;

// Phase 2E: SVG documents (mutable DOM, parse/serialize, render-to-D2D-calls).
pub mod svg;

// COM gaps part 3: GDI metafile (EMF) parse + playback for ID2D1GdiMetafile (see module ceiling note).
pub mod emfplus;
pub mod emfplus_raster;
pub mod metafile;

// Phase 2D: glyph-run rendering (outlines, hinting, rasterization, ClearType, the D2D text blend).
pub mod text;



// Phase 1A: state, lifecycle and the public API.
pub mod command;
/// `ID2D1CommandList::Stream` replay (ffirects agent).
pub mod command_stream;
pub mod context;
pub mod factory;
pub mod resource;
pub mod state;

pub use command::CommandList;
pub use context::{bitmap_target, BrushImage, DeviceContext, PixelBrush};
pub use factory::{Device, Factory};
pub use resource::SolidColorBrush;
pub use state::DrawingStateBlock;
