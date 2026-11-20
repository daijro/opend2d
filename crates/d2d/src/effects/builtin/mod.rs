//! The built-in effect families, each a module of per-pixel maths verified against the native
//! WARP oracle: [`blend`] (blending), [`color`] (colour and transfers), [`filters`] (neighbourhood
//! filters, with [`filters_hp`] their form at a float/16-bit buffer precision), [`transform`] (transforms,
//! generators, lighting).
pub mod blend;
pub mod color;
pub mod filters;
pub mod filters_hp;
pub mod transform;
