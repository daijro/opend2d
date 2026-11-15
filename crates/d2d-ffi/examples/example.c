/* opend2d C example — the standard Direct2D / DirectWrite / WIC API, exactly as on Windows.
 *
 * Built with stock MinGW-w64 Windows headers (d2d1.h/d2d1_1.h, dwrite.h, wincodec.h) and linked
 * against the import libs; at runtime Wine resolves d2d1.dll / dwrite.dll / windowscodecs.dll to
 * our drop-in DLLs (WINEDLLOVERRIDES, staged beside the exe). Nothing here knows it is opend2d.
 *
 * MinGW-w64's *derived*-interface D2D COBJMACROS (e.g. ID2D1DeviceContext_Clear) are buggy — they
 * reach inherited methods at the wrong vtable level. So we call the standard COM ABI through the
 * stock headers' correct base-interface macros (ID2D1RenderTarget_*, ID2D1Bitmap_*, …) plus the
 * `->lpVtbl->` slot for each interface's own methods, with plain pointer casts up the COM chain.
 *
 * Two render passes:
 *   (1) parity — reproduces harness scene `02-fractional-rect` op-for-op (clear + two solid-brush
 *       FillRectangle), reads the target back through a CPU staging bitmap exactly as the oracle
 *       capture does, and asserts the bytes equal the committed native oracle .bgra.
 *   (2) showcase — path geometry and DrawText (DirectWrite) to exercise the rest of the API.
 *       Both passes are saved to PNG through WIC. (Effects need the ID2D1Effect C vtable, which
 *       MinGW's stock C headers omit — example.cpp drives an effect through the C++ interface.)
 *
 * Built and run by examples/run-examples.sh. Prints "C EXAMPLE OK" and exits 0 on success. */
#define COBJMACROS
#define D2D_USE_C_DEFINITIONS
#define WIN32_LEAN_AND_MEAN
#include <initguid.h> /* instantiate the referenced IIDs/CLSIDs locally, so no -luuid is needed */
#include <d2d1_1.h>
#include <dwrite.h>
#include <wincodec.h>
#include <objbase.h>
#include <stdio.h>
#include <stdlib.h>

#define CK(hr, what) do { HRESULT _h = (hr); if (FAILED(_h)) { printf("FAIL %s hr=%08lx\n", what, (unsigned long)_h); return 1; } } while (0)
#define REL(p) IUnknown_Release((IUnknown *)(p))   /* QI/AddRef/Release are slot-stable on every vtable */
#define RT(dc) ((ID2D1RenderTarget *)(dc))         /* the inherited RenderTarget face of a DeviceContext */

static const UINT W = 128, H = 96;

/* Create a BGRA8-premultiplied device-context bitmap with the given options at 96 DPI. */
static ID2D1Bitmap1 *make_bitmap(ID2D1DeviceContext *dc, D2D1_BITMAP_OPTIONS opts) {
    D2D1_BITMAP_PROPERTIES1 p = {{DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED}, 96.0f, 96.0f, opts, NULL};
    D2D1_SIZE_U size = {W, H};
    ID2D1Bitmap1 *b = NULL;
    if (FAILED(dc->lpVtbl->CreateBitmap(dc, size, NULL, 0, &p, &b))) return NULL;
    return b;
}

/* Copy `target` into a CPU-readable staging bitmap and read its rows out tightly packed (top-down
 * premultiplied BGRA8), exactly as the oracle capture does. `out` is malloc'd for the caller. */
static int readback(ID2D1DeviceContext *dc, ID2D1Bitmap1 *target, unsigned char **out) {
    ID2D1Bitmap1 *staging = make_bitmap(dc, D2D1_BITMAP_OPTIONS_CPU_READ | D2D1_BITMAP_OPTIONS_CANNOT_DRAW);
    if (!staging) { printf("FAIL staging bitmap\n"); return 1; }
    CK(ID2D1Bitmap_CopyFromBitmap((ID2D1Bitmap *)staging, NULL, (ID2D1Bitmap *)target, NULL), "CopyFromBitmap");
    D2D1_MAPPED_RECT m;
    CK(ID2D1Bitmap1_Map(staging, D2D1_MAP_OPTIONS_READ, &m), "Map");
    unsigned char *buf = malloc((size_t)W * H * 4);
    for (UINT y = 0; y < H; y++) memcpy(buf + (size_t)y * W * 4, m.bits + (size_t)y * m.pitch, (size_t)W * 4);
    ID2D1Bitmap1_Unmap(staging);
    REL(staging);
    *out = buf;
    return 0;
}

/* Save tightly-packed BGRA8 pixels to `path` as a PNG through WIC — the standard encoder chain. */
static int save_png(IWICImagingFactory *wic, const wchar_t *path, const unsigned char *bgra) {
    IWICStream *stream = NULL;
    IWICBitmapEncoder *enc = NULL;
    IWICBitmapFrameEncode *frame = NULL;
    CK(IWICImagingFactory_CreateStream(wic, &stream), "CreateStream");
    CK(IWICStream_InitializeFromFilename(stream, path, GENERIC_WRITE), "InitializeFromFilename");
    CK(IWICImagingFactory_CreateEncoder(wic, &GUID_ContainerFormatPng, NULL, &enc), "CreateEncoder");
    CK(IWICBitmapEncoder_Initialize(enc, (IStream *)stream, WICBitmapEncoderNoCache), "Encoder.Initialize");
    CK(IWICBitmapEncoder_CreateNewFrame(enc, &frame, NULL), "CreateNewFrame");
    CK(IWICBitmapFrameEncode_Initialize(frame, NULL), "Frame.Initialize");
    CK(IWICBitmapFrameEncode_SetSize(frame, W, H), "SetSize");
    WICPixelFormatGUID fmt = GUID_WICPixelFormat32bppBGRA;
    CK(IWICBitmapFrameEncode_SetPixelFormat(frame, &fmt), "SetPixelFormat");
    CK(IWICBitmapFrameEncode_WritePixels(frame, H, W * 4, (UINT)W * H * 4, (BYTE *)bgra), "WritePixels");
    CK(IWICBitmapFrameEncode_Commit(frame), "Frame.Commit");
    CK(IWICBitmapEncoder_Commit(enc), "Encoder.Commit");
    REL(frame); REL(enc); REL(stream);
    return 0;
}

int main(int argc, char **argv) {
    const char *oracle_path = argc > 1 ? argv[1] : NULL; /* 02-fractional-rect.bgra, optional */
    wchar_t wdir[512];
    mbstowcs(wdir, argc > 2 ? argv[2] : ".", 512);

    CK(CoInitializeEx(NULL, COINIT_MULTITHREADED), "CoInitializeEx");

    /* factory → device → context (our CreateDevice ignores the DXGI device: it is a CPU renderer) */
    ID2D1Factory1 *factory = NULL;
    CK(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &IID_ID2D1Factory1, NULL, (void **)&factory), "D2D1CreateFactory");
    ID2D1Device *device = NULL;
    CK(factory->lpVtbl->CreateDevice(factory, NULL, &device), "CreateDevice");
    ID2D1DeviceContext *dc = NULL;
    CK(device->lpVtbl->CreateDeviceContext(device, D2D1_DEVICE_CONTEXT_OPTIONS_NONE, &dc), "CreateDeviceContext");
    ID2D1RenderTarget_SetDpi(RT(dc), 96.0f, 96.0f);

    IWICImagingFactory *wic = NULL;
    CK(CoCreateInstance(&CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, &IID_IWICImagingFactory, (void **)&wic), "CoCreateInstance(WIC)");

    /* ---- pass 1: parity — op-for-op copy of scene 02-fractional-rect ---- */
    ID2D1Bitmap1 *target = make_bitmap(dc, D2D1_BITMAP_OPTIONS_TARGET);
    if (!target) { printf("FAIL target bitmap\n"); return 1; }
    dc->lpVtbl->SetTarget(dc, (ID2D1Image *)target);
    ID2D1RenderTarget_BeginDraw(RT(dc));
    D2D1_COLOR_F clear0 = {0, 0, 0, 0}, white = {1, 1, 1, 1};
    ID2D1RenderTarget_Clear(RT(dc), &clear0);       /* harness clears transparent first */
    ID2D1RenderTarget_Clear(RT(dc), &white);        /* scene op: clear white */

    ID2D1SolidColorBrush *ink = NULL, *red = NULL;
    D2D1_COLOR_F ink_c = {0.12f, 0.35f, 0.8f, 1.0f}, red_c = {1.0f, 0.0f, 0.0f, 0.5f};
    CK(ID2D1RenderTarget_CreateSolidColorBrush(RT(dc), &ink_c, NULL, &ink), "CreateSolidColorBrush(ink)");
    CK(ID2D1RenderTarget_CreateSolidColorBrush(RT(dc), &red_c, NULL, &red), "CreateSolidColorBrush(red)");
    D2D1_RECT_F r1 = {10.25f, 12.5f, 110.75f, 82.125f}, r2 = {30.0f, 25.0f, 126.0f, 55.0f};
    ID2D1RenderTarget_FillRectangle(RT(dc), &r1, (ID2D1Brush *)ink);
    ID2D1RenderTarget_FillRectangle(RT(dc), &r2, (ID2D1Brush *)red);

    D2D1_TAG t1 = 0, t2 = 0;
    CK(ID2D1RenderTarget_EndDraw(RT(dc), &t1, &t2), "EndDraw(parity)");
    dc->lpVtbl->SetTarget(dc, NULL);

    unsigned char *pixels = NULL;
    if (readback(dc, target, &pixels)) return 1;

    int parity = -1; /* -1 = no oracle to compare */
    if (oracle_path) {
        FILE *f = fopen(oracle_path, "rb");
        if (!f) { printf("FAIL open oracle %s\n", oracle_path); return 1; }
        unsigned char *oracle = malloc((size_t)W * H * 4);
        size_t n = fread(oracle, 1, (size_t)W * H * 4, f);
        fclose(f);
        parity = (n == (size_t)W * H * 4) && memcmp(pixels, oracle, n) == 0;
        if (!parity)
            for (size_t i = 0; i < n; i++)
                if (pixels[i] != oracle[i]) { printf("parity mismatch at byte %zu: got %u want %u\n", i, pixels[i], oracle[i]); break; }
        free(oracle);
    }

    wchar_t path[1024];
    swprintf(path, 1024, L"%ls/c-parity.png", wdir);
    if (save_png(wic, path, pixels)) return 1;
    free(pixels);
    REL(ink); REL(red); REL(target);

    /* ---- pass 2: showcase — path geometry, text, effect ---- */
    ID2D1Bitmap1 *canvas = make_bitmap(dc, D2D1_BITMAP_OPTIONS_TARGET);
    if (!canvas) { printf("FAIL canvas bitmap\n"); return 1; }
    dc->lpVtbl->SetTarget(dc, (ID2D1Image *)canvas);
    ID2D1RenderTarget_BeginDraw(RT(dc));
    ID2D1RenderTarget_Clear(RT(dc), &white);

    /* geometry: a filled triangle */
    ID2D1PathGeometry *tri = NULL;
    CK(ID2D1Factory_CreatePathGeometry((ID2D1Factory *)factory, &tri), "CreatePathGeometry");
    ID2D1GeometrySink *sink = NULL;
    CK(ID2D1PathGeometry_Open(tri, &sink), "PathGeometry.Open");
    ID2D1SimplifiedGeometrySink *s = (ID2D1SimplifiedGeometrySink *)sink;
    D2D1_POINT_2F a = {20, 20};
    ID2D1SimplifiedGeometrySink_BeginFigure(s, a, D2D1_FIGURE_BEGIN_FILLED);
    D2D1_POINT_2F pts[2] = {{100, 40}, {30, 90}};
    ID2D1SimplifiedGeometrySink_AddLines(s, pts, 2);
    ID2D1SimplifiedGeometrySink_EndFigure(s, D2D1_FIGURE_END_CLOSED);
    CK(ID2D1SimplifiedGeometrySink_Close(s), "GeometrySink.Close");
    REL(sink);
    ID2D1SolidColorBrush *fill = NULL;
    D2D1_COLOR_F teal = {0.0f, 0.5f, 0.5f, 1.0f};
    CK(ID2D1RenderTarget_CreateSolidColorBrush(RT(dc), &teal, NULL, &fill), "CreateSolidColorBrush(fill)");
    ID2D1RenderTarget_FillGeometry(RT(dc), (ID2D1Geometry *)tri, (ID2D1Brush *)fill, NULL);
    REL(tri);

    /* text: DirectWrite format + DrawText (needs a system font via D2D_FONTS; blank if absent) */
    IDWriteFactory *dw = NULL;
    CK(DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, &IID_IDWriteFactory, (IUnknown **)&dw), "DWriteCreateFactory");
    IDWriteTextFormat *tf = NULL;
    CK(dw->lpVtbl->CreateTextFormat(dw, L"Arial", NULL, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL,
                                    DWRITE_FONT_STRETCH_NORMAL, 18.0f, L"en-us", &tf), "CreateTextFormat");
    ID2D1SolidColorBrush *text_brush = NULL;
    D2D1_COLOR_F black = {0, 0, 0, 1};
    CK(ID2D1RenderTarget_CreateSolidColorBrush(RT(dc), &black, NULL, &text_brush), "CreateSolidColorBrush(text)");
    D2D1_RECT_F layout = {8, 60, 126, 94};
    const wchar_t *msg = L"opend2d";
    ID2D1RenderTarget_DrawText(RT(dc), msg, (UINT32)wcslen(msg), tf, &layout, (ID2D1Brush *)text_brush,
                               D2D1_DRAW_TEXT_OPTIONS_NONE, DWRITE_MEASURING_MODE_NATURAL);
    REL(tf); REL(dw);

    /* effect: MinGW-w64's stock *C* headers omit the ID2D1Effect / ID2D1Properties C vtables
     * (they ship only the C++ definitions), so an effect graph cannot be driven from C with stock
     * headers. example.cpp shows the standard CreateEffect → SetValue → GetOutput → DrawImage flow. */

    CK(ID2D1RenderTarget_EndDraw(RT(dc), &t1, &t2), "EndDraw(showcase)");
    dc->lpVtbl->SetTarget(dc, NULL);

    unsigned char *shot = NULL;
    if (readback(dc, canvas, &shot)) return 1;
    swprintf(path, 1024, L"%ls/c-showcase.png", wdir);
    if (save_png(wic, path, shot)) return 1;
    free(shot);
    REL(fill); REL(text_brush); REL(canvas);

    REL(wic); REL(dc); REL(device); REL(factory);
    CoUninitialize();

    printf("C EXAMPLE OK (parity: %s; geometry+text+WIC PNG saved; effects: see example.cpp)\n",
           parity == 1 ? "byte-exact vs native oracle" : parity == 0 ? "MISMATCH" : "no oracle given");
    return parity == 0 ? 1 : 0;
}
