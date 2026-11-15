/* Portable C-ABI test for the off-Windows d2d1 shared library.
 *
 * Drives the real COM vtables from plain C — no windows headers, no Rust — to prove the one COM
 * implementation (the `win` module) hands back a working object graph through the raw ABI on
 * Linux/macOS/FreeBSD, and that it renders byte-for-byte like native WARP. Each method is called
 * through its vtable slot (the layout the inventory in d2d-types fixes).
 *
 * It renders harness scene `02-fractional-rect` (128x96: white clear, an opaque fractional-rect
 * fill, then a half-transparent red fill) into a target bitmap, reads the premultiplied BGRA back,
 * and — when given the scene's oracle .bgra as argv[1] — asserts every byte equals native WARP.
 *
 * Built and run by `portable-test.sh`. Exit 0 and "PORTABLE OK" on success. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint32_t d1; uint16_t d2, d3; uint8_t d4[8]; } GUID;
typedef struct { float r, g, b, a; } ColorF;
typedef struct { float left, top, right, bottom; } RectF;
typedef struct { uint32_t w, h; } SizeU;
typedef struct { uint32_t pitch; uint8_t *bits; } MappedRect;

extern int D2D1CreateFactory(int type, const GUID *riid, const void *opts, void **out);

/* obj's first word is its vtable pointer; call slot n through it. */
#define SLOT(obj, n) ((*(void ***)(obj))[n])

#define W 128u
#define H 96u

static void *solid(void *ctx, ColorF c) {
    void *brush = NULL;
    ((int (*)(void *, const ColorF *, const void *, void **)) SLOT(ctx, 8))(ctx, &c, NULL, &brush);
    return brush;
}
static void fill(void *ctx, RectF r, void *brush) {
    ((void (*)(void *, const RectF *, void *)) SLOT(ctx, 17))(ctx, &r, brush);
}

typedef struct { float x, y; } Vec2;
typedef struct { float position; ColorF color; } GradStop;
typedef struct { Vec2 start, end; } LinProps;

/* A COM method returning a UDT by value uses the MSVC member ABI: the callee writes the result
 * through a hidden pointer passed as the second argument. That is exactly how TerraFX / the CLR and
 * MSVC C++ lower `D2D1_POINT_2F GetStartPoint()`. Create a linear gradient brush with a known
 * start/end and read GetStartPoint/GetEndPoint that way; the drop-in must hand back the authored
 * point, not [0,0]. (Regression guard for the by-value-return ABI — see windows-shim/fix-udt-return-abi.py.) */
static int test_point_getter_abi(void *ctx) {
    GradStop stops[2] = {{0.f, {1, 0, 0, 1}}, {1.f, {0, 0, 1, 1}}};
    void *coll = NULL;
    int hr = ((int (*)(void *, const GradStop *, uint32_t, int, int, void **)) SLOT(ctx, 9))(ctx, stops, 2, 0, 1, &coll);
    if (hr != 0 || !coll) { printf("FAIL CreateGradientStopCollection hr=%08x\n", hr); return 0; }
    LinProps lp = {{3.f, 5.f}, {7.f, 11.f}};
    void *brush = NULL;
    hr = ((int (*)(void *, const LinProps *, const void *, void *, void **)) SLOT(ctx, 10))(ctx, &lp, NULL, coll, &brush);
    if (hr != 0 || !brush) { printf("FAIL CreateLinearGradientBrush hr=%08x\n", hr); return 0; }
    Vec2 sp = {-1.f, -1.f}, ep = {-1.f, -1.f};
    ((void (*)(void *, Vec2 *)) SLOT(brush, 10))(brush, &sp); /* GetStartPoint */
    ((void (*)(void *, Vec2 *)) SLOT(brush, 11))(brush, &ep); /* GetEndPoint */
    int ok = sp.x == 3.f && sp.y == 5.f && ep.x == 7.f && ep.y == 11.f;
    printf("GetStartPoint=[%g,%g] GetEndPoint=[%g,%g] %s\n", sp.x, sp.y, ep.x, ep.y,
           ok ? "(MSVC member ABI ok)" : "FAIL by-value point return dropped");
    ((uint32_t (*)(void *)) SLOT(brush, 2))(brush);
    ((uint32_t (*)(void *)) SLOT(coll, 2))(coll);
    return ok;
}

int main(int argc, char **argv) {
    /* IID_ID2D1Factory1 = bb12d362-daee-4b9a-aa1d-14ba401cfa1f */
    GUID iid_factory1 = {0xbb12d362, 0xdaee, 0x4b9a, {0xaa, 0x1d, 0x14, 0xba, 0x40, 0x1c, 0xfa, 0x1f}};
    void *factory = NULL, *device = NULL, *ctx = NULL, *target = NULL;

    int hr = D2D1CreateFactory(0 /* single-threaded */, &iid_factory1, NULL, &factory);
    if (hr != 0 || !factory) { printf("FAIL CreateFactory hr=%08x\n", hr); return 1; }
    hr = ((int (*)(void *, void *, void **)) SLOT(factory, 17))(factory, NULL, &device);
    if (hr != 0 || !device) { printf("FAIL CreateDevice hr=%08x\n", hr); return 1; }
    hr = ((int (*)(void *, uint32_t, void **)) SLOT(device, 4))(device, 0, &ctx);
    if (hr != 0 || !ctx) { printf("FAIL CreateDeviceContext hr=%08x\n", hr); return 1; }

    int abi_ok = test_point_getter_abi(ctx);

    SizeU size = {W, H};
    hr = ((int (*)(void *, SizeU, const void *, uint32_t, const void *, void **)) SLOT(ctx, 57))(ctx, size, NULL, 0, NULL, &target);
    if (hr != 0 || !target) { printf("FAIL CreateBitmap hr=%08x\n", hr); return 1; }

    ((void (*)(void *, void *)) SLOT(ctx, 74))(ctx, target);          /* SetTarget */
    ((void (*)(void *, float, float)) SLOT(ctx, 51))(ctx, 96.f, 96.f); /* SetDpi */
    ((void (*)(void *)) SLOT(ctx, 48))(ctx);                           /* BeginDraw */

    ColorF white = {1, 1, 1, 1};
    ((void (*)(void *, const ColorF *)) SLOT(ctx, 47))(ctx, &white);   /* Clear */
    void *ink = solid(ctx, (ColorF){0.12f, 0.35f, 0.8f, 1.f});
    fill(ctx, (RectF){10.25f, 12.5f, 110.75f, 82.125f}, ink);
    void *red = solid(ctx, (ColorF){1.f, 0.f, 0.f, 0.5f});
    fill(ctx, (RectF){30.f, 25.f, 126.f, 55.f}, red);

    uint64_t t1 = 0, t2 = 0;
    hr = ((int (*)(void *, uint64_t *, uint64_t *)) SLOT(ctx, 49))(ctx, &t1, &t2); /* EndDraw */
    if (hr != 0) { printf("FAIL EndDraw hr=%08x\n", hr); return 1; }
    ((void (*)(void *, void *)) SLOT(ctx, 74))(ctx, NULL); /* SetTarget(NULL): buffer back to bitmap */

    MappedRect mapped;
    memset(&mapped, 0, sizeof mapped);
    hr = ((int (*)(void *, uint32_t, MappedRect *)) SLOT(target, 14))(target, 1 /* READ */, &mapped);
    if (hr != 0 || !mapped.bits) { printf("FAIL Map hr=%08x\n", hr); return 1; }

    int ok = 1;
    if (argc > 1) {
        /* Byte-for-byte against the native WARP oracle (row-major BGRA, no padding). */
        FILE *f = fopen(argv[1], "rb");
        if (!f) { printf("FAIL open oracle %s\n", argv[1]); return 1; }
        uint8_t *oracle = malloc(W * H * 4);
        if (fread(oracle, 1, W * H * 4, f) != W * H * 4) { printf("FAIL read oracle\n"); return 1; }
        fclose(f);
        for (uint32_t y = 0; y < H && ok; y++)
            if (memcmp(mapped.bits + (size_t) y * mapped.pitch, oracle + (size_t) y * W * 4, W * 4) != 0)
                ok = 0;
        free(oracle);
        printf(ok ? "byte-exact vs native oracle\n" : "FAIL pixel mismatch vs oracle\n");
    } else {
        uint8_t *px = mapped.bits; /* half-red over white, premultiplied BGRA at (30,25) */
        printf("rendered 128x96 (pitch %u); top-left BGRA = %u,%u,%u,%u\n", mapped.pitch, px[0], px[1], px[2], px[3]);
    }

    ((int (*)(void *)) SLOT(target, 15))(target); /* Unmap */
    ((uint32_t (*)(void *)) SLOT(red, 2))(red);
    ((uint32_t (*)(void *)) SLOT(ink, 2))(ink);
    ((uint32_t (*)(void *)) SLOT(target, 2))(target);
    ((uint32_t (*)(void *)) SLOT(ctx, 2))(ctx);
    ((uint32_t (*)(void *)) SLOT(device, 2))(device);
    ((uint32_t (*)(void *)) SLOT(factory, 2))(factory);

    ok = ok && abi_ok;
    printf(ok ? "PORTABLE OK\n" : "PORTABLE FAILED\n");
    return ok ? 0 : 1;
}
