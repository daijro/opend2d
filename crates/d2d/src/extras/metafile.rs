//! GDI metafile playback (`ID2D1GdiMetafile`, D2D 1.0/1.1).
//!
//! **Blocked for CPU parity** — grounded: playback replays EMF/EMF+ records through GDI
//! (`D2DGdiMetafile`), which needs a GDI device and an EMF byte stream the offscreen BGRA harness
//! cannot author or consume; it is out of CPU-parity scope. Once a
//! record stream is decoded, replay reduces to the geometry/bitmap/text specs this module does not
//! own anyway. No scene; documented and stopped.
