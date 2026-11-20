//! Direct2D shader bytecode execution for the CPU renderer (phase 3C).
//!
//! Reads a DXBC pixel-shader container ([`dxbc::Shader`]), decodes its shader-model 4/5 tokens
//! ([`sm4`]), and runs it on the CPU ([`exec`]) with the float/integer semantics WARP's JIT emits,
//! so a custom or built-in effect pixel shader produces the same bytes on every platform as native
//! Direct2D on Windows WARP. Texture `sample`/`ld` are delegated to a caller-supplied [`exec::Env`]
//! so the effect graph wires in crates/d2d's WARP-exact `sampling.rs`.
//!
//! # API for the effect graph
//!
//! The effect graph (crates/d2d `effects`) uses this per custom/built-in pixel-shader pass:
//!
//! 1. [`Shader::parse`] the pass's DXBC once (dedup it by GUID yourself — `LoadPixelShader`
//!    semantics live in the effect context, not here), then check [`Shader::unsupported_op`]: if it
//!    returns an opcode, report that blockage instead of running (the executor never silently
//!    zeroes an unimplemented opcode). [`Shader::inputs`] gives the input-register layout
//!    (`SV_Position`, one texcoord per input image) so you know which `v#` to fill.
//! 2. For each output pixel, build [`PixelInputs`] — the interpolated input registers and the `b0`
//!    constant-buffer bytes — and an [`Env`] that samples each bound input image / resource texture.
//! 3. Call [`Shader::run`]; it returns the premultiplied RGBA `o0` output. Use [`exec::to_unorm8`]
//!    per channel for an 8-bit target.
//!
//! The executor holds no GPU/platform state, so the same call runs identically on every target
//! (native, wasm32).

pub mod dxbc;
// warpmath split into its own dependency-free crate so pdn-core can share it; re-exported here so
// `d2d_shader::warpmath` and in-crate `crate::warpmath::…` paths are unchanged.
pub use warpmath;
pub mod exec;
pub mod sm4;

pub use dxbc::Shader;
pub use exec::{Env, PixelInputs};
