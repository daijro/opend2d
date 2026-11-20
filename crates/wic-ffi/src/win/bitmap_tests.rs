//! `IWICImagingFactory::CreateBitmap` + `IWICBitmapLock` round-trip, driven through the real COM
//! vtable in-process (no Wine). Mirrors `wic-dump bitmaplock` / the decisive identity test: a flat
//! write through the lock's data pointer then `CopyPixels` back out must equal the input, which holds
//! only when the bitmap's internal stride is the native TIGHT `ceil(w*bpp/8)` (not DWORD-aligned).

use super::factory::FactoryObj;
use windows::Win32::Graphics::Imaging::*;
use windows_core::{GUID, Interface};

// (The off-Windows CoTaskMemAlloc/CoTaskMemFree/PropVariantClear symbols are now defined for real in
// `meta.rs` as library code, so this test binary links against those, not local stubs.)

fn guid(s: &str) -> GUID {
    super::guid_from_str(s)
}

/// Flat-write `src` through a write lock, read back tight, assert identity + the tight data size.
fn assert_identity(fmt: GUID, w: u32, h: u32, bpp: u32, src: &[u8]) {
    unsafe {
        let f: IWICImagingFactory = FactoryObj::new().into();
        let bmp = f.CreateBitmap(w, h, &fmt, WICBitmapCacheOnLoad).unwrap();
        {
            let lock = bmp.Lock(std::ptr::null(), WICBitmapLockWrite.0 as u32).unwrap();
            // WIC hands back a tight buffer for a full-bitmap lock.
            assert_eq!(lock.GetStride().unwrap(), (w * bpp).div_ceil(8), "{fmt:?} lock stride");
            let (mut cb, mut ppb) = (0u32, std::ptr::null_mut());
            lock.GetDataPointer(&mut cb, &mut ppb).unwrap();
            assert_eq!(cb as usize, src.len(), "{fmt:?} data size");
            let n = (cb as usize).min(src.len());
            std::ptr::copy_nonoverlapping(src.as_ptr(), ppb, n);
        }
        let stride = (w * bpp).div_ceil(8);
        let mut out = vec![0u8; stride as usize * h as usize];
        bmp.CopyPixels(std::ptr::null(), stride, &mut out).unwrap();
        assert_eq!(out, src, "{fmt:?} {w}x{h} identity round-trip");
    }
}

#[test]
fn create_bitmap_lock_identity_tight_stride() {
    // 5-wide rgb24: tight row 15 (DWORD-aligned would be 16 and shift every row past the first).
    let rgb24: Vec<u8> = (0..4u8).flat_map(|row| (0..5u8).flat_map(move |col| [col, row, 0])).collect();
    assert_eq!(rgb24.len(), 60);
    assert_identity(guid("6fddc324-4e03-4bfe-b185-3d77768dc90d"), 5, 4, 24, &rgb24);

    // 5-wide gray8: tight row 5 (DWORD-aligned would be 8).
    let gray8: Vec<u8> = (0..4u8).flat_map(|row| (0..5u8).map(move |col| row * 5 + col)).collect();
    assert_eq!(gray8.len(), 20);
    assert_identity(guid("6fddc324-4e03-4bfe-b185-3d77768dc908"), 5, 4, 8, &gray8);

    // 32bpp bgra, already DWORD-aligned, so tight == DWORD (regression guard, not a change).
    let bgra: Vec<u8> = (0..3 * 4 * 4).map(|i| i as u8).collect();
    assert_identity(guid("6fddc324-4e03-4bfe-b185-3d77768dc90f"), 3, 4, 32, &bgra);
}
