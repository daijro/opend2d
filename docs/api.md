# API coverage of the opend2d implementation

This counts what **opend2d itself** exposes, by interface. The numbers here come from the COM objects
our FFI crates actually implement (`#[implement(...)]`) and the methods they deliberately leave as
`E_NOTIMPL`.

Structural coverage (an interface is implemented, a method is wired) is necessary but not
sufficient — the real measure is behavioural: **1190** scenes byte-exact in the core and the decisive
drop-in run (see [conformance.md](conformance.md)). A method can be "wired" and still be refined
against the oracle; conversely an `E_NOTIMPL` is only a gap if native implements it.

## By DLL

| Drop-in | COM interfaces implemented | `E_NOTIMPL` method sites | Flat exports |
| --- | ---: | ---: | --- |
| `d2d1.dll` (`crates/d2d-ffi`) | ~55 Direct2D interfaces + the effect-author transform/info/graph set | 84 | `D2D1CreateFactory`, `D2D1CreateDevice`, `D2D1CreateDeviceContext`, and the 9 matrix/colour helpers (the deferred `…GradientMeshInteriorPointsFromCoonsPatch` aside) |
| `dwrite.dll` (`crates/dwrite-ffi`) | 29 DirectWrite interfaces + the private `ITextInterop` | 25 | `DWriteCreateFactory` |
| `windowscodecs.dll` (`crates/wic-ffi`) | 23 WIC interfaces + COM plumbing (`IClassFactory`, enumerators) | 10 | `DllGetClassObject`, `DllCanUnloadNow` |

The implemented Direct2D interfaces cover the full device-context drawing path: factory/device/
context, every geometry type and sink, solid/gradient/bitmap/image brushes, stroke styles, layers,
command lists, the effect graph (`ID2D1Effect` + `ID2D1EffectContext` + the transform/draw/compute/
source/analysis authoring interfaces), sprite batches, ink, gradient meshes, LUT3D, the SVG document
object model, GDI metafile, colour contexts, and the WIC/DXGI bitmap bridges.

## Deliberate native-matching rejections

These return `E_NOTIMPL` (or a specific `HRESULT`) **because native does, or because they are
registration/host data Direct2D itself keeps outside the renderer**. They are not missing work:

- `ID2D1Factory::CreateHwndRenderTarget` / `ID2D1HwndRenderTarget`, and swap-chain presentation as a
  *renderer* — opend2d renders offscreen; presentation is the host's job.
- Built-in-effect **property metadata** getters — `ID2D1Properties::{GetPropertyName,
  GetPropertyCount,GetType,GetSubProperties}`. Names/types/counts live in `d2d1.dll`'s registration
  XML, not in the renderer; `GetValue`/`GetValueSize` *do* read the stored value back.
- The versioned factory overload `ID2D1Factory1::CreateDrawingStateBlock` (→`…1`), and
  `GetRegisteredEffects`/`GetEffectProperties`. Create drawing state blocks through the base
  `ID2D1Factory` method. (`ID2D1Factory1::CreatePathGeometry` → `ID2D1PathGeometry1` *is* implemented,
  with `ComputePointAndSegmentAtLength`, byte-exact vs native for line/cubic/quadratic figures.)
- Legacy render-target creators `ID2D1RenderTarget::{CreateSharedBitmap,CreateCompatibleRenderTarget,
  CreateLayer,CreateMesh,CreateWicBitmapRenderTarget}` — superseded by the device-context path.
- Colour-bitmap/SVG **glyph** image extraction (`ID2D1DeviceContext4::{CreateSvgGlyphStyle,
  GetColorBitmapGlyphImage,GetSvgGlyphImage}`) — no passing scene reaches them yet.

## Documented ceilings (implemented, but not full native parity)

- `ID2D1GdiMetafile` playback (`DrawGdiMetafile`) covers a minimal primitive record set; native
  delegates full EMF playback to an external apiset, so metafile *pixel* parity is out of reach here.
- `ID2D1PrintControl` produces an XPS package via the OS XPS object model (native's whole-page-raster
  form); vector serialisation of arbitrary command-list primitives is the ceiling.
- DXGI-surface read-back / image sources are single BGRA planes; multi-plane YUV is not modelled.
- `ID2D1Geometry::Widen` returns byte-exact widened regions for rects and flat-cap/miter-bevel
  strokes; round joins/caps and square caps still differ from native's widened-region stream.

## Genuinely missing (tracked)

Non-Latin fallback-glyph rendering and a few effect/float paths are still being closed; these are the
~35 decisive-run residuals. See [gaps.md](gaps.md). The GPU backend (`d2d-wgpu`) and arbitrary-shader
translation (`d2d-shader`) are secondary to the authoritative CPU path and advance incrementally.

The authoritative, always-current lists are in the source: `crates/d2d-ffi/src/win/mod.rs`
(`NOT_IMPL`) enumerates the Direct2D rejections with rationale; `grep '#\[implement('` in each FFI
crate enumerates the implemented interfaces.
