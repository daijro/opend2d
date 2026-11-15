//! `ID2D1LookupTable3D` and the built-in LookupTable3D effect (D2D 1.2).
//!
//! **Blocked** — grounded, not a gap to fill here: the LUT3D object is an
//! opaque 3D-texture resource that is *only* sampled by the built-in LookupTable3D effect's pixel
//! shader. This project has no effect-graph infrastructure yet (phases 2B/2C are unbuilt), so the
//! effect cannot be run end-to-end to produce a comparable pixel scene. The sampling itself
//! (trilinear/tetrahedral interpolation plus the `D2D1_BUFFER_PRECISION` conversion) lives in that
//! WARP f32 pixel shader and carries the same ≤1-LSB residual as every other WARP f32 path
//! (warp-unorm8-sampler), so it would not be byte-exact without the whole effect substrate.
//! `CreateLookupTable3D` forwards `dataCount`/stride validation to the inner device
//! factory through an unnamed indirect call, so the exact rule is not readable from the public
//! method. No scene; documented and stopped until the effect graph lands.
