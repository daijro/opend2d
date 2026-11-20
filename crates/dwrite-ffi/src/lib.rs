//! `dwrite-ffi`: the drop-in `dwrite.dll` and the DirectWrite COM surface Direct2D apps use to make
//! text (phase: drop-in dwrite over `d2d::text`).
//!
//! On Windows the crate is a `cdylib` named `dwrite.dll` that exports `DWriteCreateFactory` and hands
//! back real DirectWrite COM objects — implemented with the official `windows` IIDs/vtables via
//! `#[implement]`, each method forwarding to the `d2d::text` Rust core (layout, shaping, glyph
//! metrics). Dropping it beside the unchanged harness `d2d-ref.exe` together with our `d2d1.dll`
//! reproduces native DirectWrite + Direct2D text output. Methods outside the exercised text subset
//! return `E_NOTIMPL`/defaults.
//!
//! The font bytes and layout a text object carries reach our `d2d1.dll`'s `DrawText`/`DrawGlyphRun`
//! through [`ITextInterop`], a private COM interface our text objects implement and `d2d1.dll` QIs —
//! the rendering stays in the shared `d2d` core, so both DLLs draw exactly the core's pixels.
//!
//! Off Windows the `windows` interface definitions are `#[cfg(windows)]`-gated upstream, so the
//! crate builds on the `windows-shim`/`windows-core-shim` crates (renamed to `windows`/`windows_core`)
//! — the one COM implementation in [`win`] serves every platform.

pub mod win;

pub use win::{default_rendering_params, font_file_rasterizer, resolve_face, resolve_face_sims, resolve_face_sims_covering, resolve_face_sims_covering_typographic, resolve_face_sims_in, resolve_face_sims_typographic, wss_to_axis_values, ITextInterop};
