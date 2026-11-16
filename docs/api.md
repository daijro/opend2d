# API coverage

This page counts what OpenD2D itself exposes, interface by interface. The numbers come from the COM
objects the FFI crates implement with `#[implement(...)]`, and from the methods they deliberately
leave returning E_NOTIMPL.

Implementing an interface or wiring up a method is necessary but not sufficient. The real measure
is behaviour: 6132 of the 6159 test scenes match Direct2D byte for byte (see
[conformance.md](conformance.md)). A wired method can still be under refinement
against Windows' output, and an E_NOTIMPL is only a gap if Direct2D implements that method.

## By DLL

| Drop-in | COM interfaces implemented | E_NOTIMPL method sites | Flat exports |
| --- | ---: | ---: | --- |
| d2d1.dll (`crates/d2d-ffi`) | about 55 Direct2D interfaces, plus the effect-author transform, info and graph set | 84 | D2D1CreateFactory, D2D1CreateDevice, D2D1CreateDeviceContext and 9 matrix and colour helpers (D2D1GetGradientMeshInteriorPointsFromCoonsPatch is deferred) |
| dwrite.dll (`crates/dwrite-ffi`) | 29 DirectWrite interfaces, plus the private ITextInterop | 25 | DWriteCreateFactory |
| windowscodecs.dll (`crates/wic-ffi`) | 23 WIC interfaces, plus COM plumbing (IClassFactory, enumerators) | 10 | DllGetClassObject, DllCanUnloadNow |

The implemented Direct2D interfaces cover the whole device-context drawing path: the factory,
device and context, every geometry type and sink, solid, gradient, bitmap and image brushes, stroke
styles, layers, command lists, sprite batches, ink, gradient meshes, LUT3D, the SVG document object
model, GDI metafiles, colour contexts, and the bridges to WIC and DXGI bitmaps. They also cover the
effect graph: ID2D1Effect, ID2D1EffectContext, and the transform, draw, compute, source and
analysis interfaces used to write effects.

## Deliberate native-matching rejections

The methods below return E_NOTIMPL, or a specific HRESULT, either because Direct2D does the same or
because they serve registration or host data that Direct2D keeps outside the renderer. They are not
missing work.

- ID2D1Factory::CreateHwndRenderTarget, ID2D1HwndRenderTarget, and swap-chain presentation as a
  renderer feature. OpenD2D renders offscreen, and presentation is the host's job.
- The property metadata getters for built-in effects: ID2D1Properties::GetPropertyName,
  GetPropertyCount, GetType and GetSubProperties. Effect property names, types and counts live in
  the registration XML inside d2d1.dll, not in the renderer. GetValue and GetValueSize do read the
  stored value back.
- The versioned factory overload ID2D1Factory1::CreateDrawingStateBlock (which returns
  ID2D1DrawingStateBlock1), plus GetRegisteredEffects and GetEffectProperties. Create drawing state
  blocks through the base ID2D1Factory method instead. ID2D1Factory1::CreatePathGeometry is
  implemented: it returns an ID2D1PathGeometry1 whose ComputePointAndSegmentAtLength matches
  Direct2D byte for byte for line, cubic and quadratic figures.
- The legacy render-target creators ID2D1RenderTarget::CreateSharedBitmap,
  CreateCompatibleRenderTarget, CreateLayer, CreateMesh and CreateWicBitmapRenderTarget. The
  device-context path supersedes them.
- Colour bitmap and SVG glyph image extraction: ID2D1DeviceContext4::CreateSvgGlyphStyle,
  GetColorBitmapGlyphImage and GetSvgGlyphImage. No passing scene reaches them yet.

## Implemented, but not at parity yet

These work, but don't match Direct2D in every case yet.

- ID2D1GdiMetafile playback (DrawGdiMetafile) covers a minimal set of primitive records. Direct2D
  hands full EMF playback to a separate Windows component, which OpenD2D doesn't reproduce yet.
- ID2D1PrintControl produces an XPS package through the operating system's XPS object model, in the
  whole-page raster form that Direct2D uses. It does not serialise arbitrary command-list
  primitives as vectors.
- DXGI surface read-back and image sources are single BGRA planes. Multi-plane YUV is not modelled.
- ID2D1Geometry::Widen returns byte-exact widened regions for rectangles and for strokes with flat
  caps and miter or bevel joins. Round joins and caps, and square caps, still differ from the
  widened region stream that Direct2D produces.

## Still missing

The remaining rendering differences, such as fallback fonts for non-Latin scripts and a few effect
and float paths, are described with examples in [gaps.md](gaps.md). The GPU backend (`d2d-wgpu`) and arbitrary shader translation (`d2d-shader`)
are secondary to the authoritative CPU path and advance incrementally.

The source always has the current lists. In `crates/d2d-ffi/src/win/mod.rs`, `NOT_IMPL` lists the
Direct2D rejections with the reason for each, and running `grep '#\[implement('` in each FFI crate
lists the implemented interfaces.
