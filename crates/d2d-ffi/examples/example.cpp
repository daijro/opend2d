// opend2d C++ example — the standard Direct2D / DirectWrite / WIC API, exactly as on Windows.
//
// Built with stock MinGW-w64 headers (d2d1_1.h, d2d1effects.h, dwrite.h, wincodec.h) using the
// native C++ COM interface (->Method(), __uuidof); at runtime Wine resolves d2d1.dll / dwrite.dll /
// windowscodecs.dll to our drop-in DLLs (WINEDLLOVERRIDES). Unlike the C headers, the C++ headers
// expose ID2D1Effect, so this one adds an effect to the pipeline example.c could not reach from C.
//
//   pass 1 parity   : reproduces harness scene 02-fractional-rect op-for-op, reads the target back
//                     through a CPU staging bitmap, asserts byte-equality with the native oracle.
//   pass 2 showcase : path geometry + DrawText (DirectWrite) + a Flood effect (DrawImage).
//   both passes are saved to PNG through WIC.
//
// Built and run by examples/run-examples.sh. Prints "CPP EXAMPLE OK" and exits 0 on success.
#define WIN32_LEAN_AND_MEAN
#include <d2d1_1.h>
#include <d2d1effects.h>
#include <dwrite.h>
#include <wincodec.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>

static const UINT W = 128, H = 96;

#define CK(hr, what) do { HRESULT h_ = (hr); if (FAILED(h_)) { std::printf("FAIL %s hr=%08lx\n", what, (unsigned long)h_); return 1; } } while (0)

// A BGRA8-premultiplied device-context bitmap at 96 DPI.
static HRESULT make_bitmap(ID2D1DeviceContext *dc, D2D1_BITMAP_OPTIONS opts, ID2D1Bitmap1 **out) {
    D2D1_BITMAP_PROPERTIES1 p = {};
    p.pixelFormat = {DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED};
    p.dpiX = p.dpiY = 96.0f;
    p.bitmapOptions = opts;
    return dc->CreateBitmap({W, H}, nullptr, 0, &p, out);
}

// Copy a rendered target into a CPU-readable staging bitmap and read it out tightly packed.
static int readback(ID2D1DeviceContext *dc, ID2D1Bitmap1 *target, unsigned char *buf) {
    ID2D1Bitmap1 *staging = nullptr;
    CK(make_bitmap(dc, D2D1_BITMAP_OPTIONS_CPU_READ | D2D1_BITMAP_OPTIONS_CANNOT_DRAW, &staging), "staging");
    CK(staging->CopyFromBitmap(nullptr, target, nullptr), "CopyFromBitmap");
    D2D1_MAPPED_RECT m;
    CK(staging->Map(D2D1_MAP_OPTIONS_READ, &m), "Map");
    for (UINT y = 0; y < H; y++) std::memcpy(buf + (size_t)y * W * 4, m.bits + (size_t)y * m.pitch, (size_t)W * 4);
    staging->Unmap();
    staging->Release();
    return 0;
}

static int save_png(IWICImagingFactory *wic, const wchar_t *path, const unsigned char *bgra) {
    IWICStream *stream = nullptr; IWICBitmapEncoder *enc = nullptr; IWICBitmapFrameEncode *frame = nullptr;
    CK(wic->CreateStream(&stream), "CreateStream");
    CK(stream->InitializeFromFilename(path, GENERIC_WRITE), "InitializeFromFilename");
    CK(wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc), "CreateEncoder");
    CK(enc->Initialize(stream, WICBitmapEncoderNoCache), "Encoder.Initialize");
    CK(enc->CreateNewFrame(&frame, nullptr), "CreateNewFrame");
    CK(frame->Initialize(nullptr), "Frame.Initialize");
    CK(frame->SetSize(W, H), "SetSize");
    WICPixelFormatGUID fmt = GUID_WICPixelFormat32bppBGRA;
    CK(frame->SetPixelFormat(&fmt), "SetPixelFormat");
    CK(frame->WritePixels(H, W * 4, (UINT)W * H * 4, (BYTE *)bgra), "WritePixels");
    CK(frame->Commit(), "Frame.Commit");
    CK(enc->Commit(), "Encoder.Commit");
    frame->Release(); enc->Release(); stream->Release();
    return 0;
}

int main(int argc, char **argv) {
    const char *oracle_path = argc > 1 ? argv[1] : nullptr;
    wchar_t wdir[512]; std::mbstowcs(wdir, argc > 2 ? argv[2] : ".", 512);

    CK(CoInitializeEx(nullptr, COINIT_MULTITHREADED), "CoInitializeEx");

    ID2D1Factory1 *factory = nullptr;
    CK(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, __uuidof(ID2D1Factory1), nullptr, (void **)&factory), "D2D1CreateFactory");
    ID2D1Device *device = nullptr;
    CK(factory->CreateDevice(nullptr, &device), "CreateDevice");           // our CreateDevice ignores the DXGI device
    ID2D1DeviceContext *dc = nullptr;
    CK(device->CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE, &dc), "CreateDeviceContext");
    dc->SetDpi(96.0f, 96.0f);

    IWICImagingFactory *wic = nullptr;
    CK(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, __uuidof(IWICImagingFactory), (void **)&wic), "CoCreateInstance(WIC)");

    // ---- pass 1: parity — op-for-op copy of scene 02-fractional-rect ----
    ID2D1Bitmap1 *target = nullptr;
    CK(make_bitmap(dc, D2D1_BITMAP_OPTIONS_TARGET, &target), "target");
    dc->SetTarget(target);
    dc->BeginDraw();
    dc->Clear(D2D1::ColorF(0, 0, 0, 0));          // harness clears transparent first
    dc->Clear(D2D1::ColorF(1, 1, 1, 1));          // scene op: clear white
    ID2D1SolidColorBrush *ink = nullptr, *red = nullptr;
    CK(dc->CreateSolidColorBrush(D2D1::ColorF(0.12f, 0.35f, 0.8f, 1.0f), &ink), "brush(ink)");
    CK(dc->CreateSolidColorBrush(D2D1::ColorF(1.0f, 0.0f, 0.0f, 0.5f), &red), "brush(red)");
    dc->FillRectangle(D2D1::RectF(10.25f, 12.5f, 110.75f, 82.125f), ink);
    dc->FillRectangle(D2D1::RectF(30.0f, 25.0f, 126.0f, 55.0f), red);
    D2D1_TAG t1 = 0, t2 = 0;
    CK(dc->EndDraw(&t1, &t2), "EndDraw(parity)");
    dc->SetTarget(nullptr);

    unsigned char *pixels = (unsigned char *)std::malloc((size_t)W * H * 4);
    if (readback(dc, target, pixels)) return 1;

    int parity = -1;
    if (oracle_path) {
        FILE *f = std::fopen(oracle_path, "rb");
        if (!f) { std::printf("FAIL open oracle %s\n", oracle_path); return 1; }
        unsigned char *oracle = (unsigned char *)std::malloc((size_t)W * H * 4);
        size_t n = std::fread(oracle, 1, (size_t)W * H * 4, f);
        std::fclose(f);
        parity = (n == (size_t)W * H * 4) && std::memcmp(pixels, oracle, n) == 0;
        std::free(oracle);
    }
    wchar_t path[1024];
    std::swprintf(path, 1024, L"%ls/cpp-parity.png", wdir);
    if (save_png(wic, path, pixels)) return 1;
    std::free(pixels);
    ink->Release(); red->Release(); target->Release();

    // ---- pass 2: showcase — path geometry, text, effect ----
    ID2D1Bitmap1 *canvas = nullptr;
    CK(make_bitmap(dc, D2D1_BITMAP_OPTIONS_TARGET, &canvas), "canvas");
    dc->SetTarget(canvas);
    dc->BeginDraw();
    dc->Clear(D2D1::ColorF(1, 1, 1, 1));

    ID2D1PathGeometry *tri = nullptr;
    CK(factory->CreatePathGeometry(&tri), "CreatePathGeometry");
    ID2D1GeometrySink *sink = nullptr;
    CK(tri->Open(&sink), "PathGeometry.Open");
    sink->BeginFigure(D2D1::Point2F(20, 20), D2D1_FIGURE_BEGIN_FILLED);
    D2D1_POINT_2F pts[2] = {{100, 40}, {30, 90}};
    sink->AddLines(pts, 2);
    sink->EndFigure(D2D1_FIGURE_END_CLOSED);
    CK(sink->Close(), "sink.Close");
    sink->Release();
    ID2D1SolidColorBrush *fill = nullptr;
    CK(dc->CreateSolidColorBrush(D2D1::ColorF(0.0f, 0.5f, 0.5f, 1.0f), &fill), "brush(fill)");
    dc->FillGeometry(tri, fill, nullptr);
    tri->Release();

    IDWriteFactory *dw = nullptr;
    CK(DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), (IUnknown **)&dw), "DWriteCreateFactory");
    IDWriteTextFormat *tf = nullptr;
    CK(dw->CreateTextFormat(L"Arial", nullptr, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL,
                            DWRITE_FONT_STRETCH_NORMAL, 18.0f, L"en-us", &tf), "CreateTextFormat");
    ID2D1SolidColorBrush *text_brush = nullptr;
    CK(dc->CreateSolidColorBrush(D2D1::ColorF(0, 0, 0, 1), &text_brush), "brush(text)");
    const wchar_t *msg = L"opend2d";
    dc->DrawText(msg, (UINT32)std::wcslen(msg), tf, D2D1::RectF(8, 60, 126, 94), text_brush,
                 D2D1_DRAW_TEXT_OPTIONS_NONE, DWRITE_MEASURING_MODE_NATURAL);
    tf->Release(); dw->Release();

    // effect: Flood a translucent colour and composite it with DrawImage
    ID2D1Effect *flood = nullptr;
    CK(dc->CreateEffect(CLSID_D2D1Flood, &flood), "CreateEffect(Flood)");
    float fc[4] = {1.0f, 0.6f, 0.0f, 0.35f}; // D2D1_FLOOD_PROP_COLOR, a D2D1_VECTOR_4F
    CK(flood->SetValue(D2D1_FLOOD_PROP_COLOR, D2D1_PROPERTY_TYPE_VECTOR4, (const BYTE *)fc, sizeof fc), "Flood.SetValue");
    ID2D1Image *flood_out = nullptr;
    flood->GetOutput(&flood_out);
    dc->DrawImage(flood_out, D2D1::Point2F(0, 0), D2D1::RectF(0, 0, 64, 24),
                  D2D1_INTERPOLATION_MODE_LINEAR, D2D1_COMPOSITE_MODE_SOURCE_OVER);
    flood_out->Release(); flood->Release();

    CK(dc->EndDraw(&t1, &t2), "EndDraw(showcase)");
    dc->SetTarget(nullptr);

    unsigned char *shot = (unsigned char *)std::malloc((size_t)W * H * 4);
    if (readback(dc, canvas, shot)) return 1;
    std::swprintf(path, 1024, L"%ls/cpp-showcase.png", wdir);
    if (save_png(wic, path, shot)) return 1;
    std::free(shot);
    fill->Release(); text_brush->Release(); canvas->Release();

    wic->Release(); dc->Release(); device->Release(); factory->Release();
    CoUninitialize();

    std::printf("CPP EXAMPLE OK (parity: %s; geometry+text+effect+WIC PNG saved)\n",
                parity == 1 ? "byte-exact vs native oracle" : parity == 0 ? "MISMATCH" : "no oracle given");
    return parity == 0 ? 1 : 0;
}
