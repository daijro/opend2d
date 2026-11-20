//! Native Windows 11 WIC: memory-bitmap buffer and lock contracts.
#![cfg(windows)]
use windows::Win32::Graphics::Imaging::*;
use windows_core::{Interface, Result};

unsafe fn factory() -> IWICImagingFactory {
    let mut raw=std::ptr::null_mut();
    unsafe { windowscodecs::win::WICCreateImagingFactory_Proxy(0x0236,&mut raw).ok().unwrap(); IWICImagingFactory::from_raw(raw) }
}
fn hr<T>(r: Result<T>) -> u32 { r.err().map_or(0,|e|e.code().0 as u32) }

#[test]
fn bitmap_memory_and_copy_bounds() {
    unsafe {
        let f=factory();
        let data:Vec<u8>=(0..21).collect();
        assert_eq!(hr(f.CreateBitmapFromMemory(3,2,&GUID_WICPixelFormat24bppBGR,8,&data)),0x80070057);
        assert_eq!(hr(f.CreateBitmapFromMemory(3,2,&GUID_WICPixelFormat24bppBGR,12,&data[..20])),0x88982f8c);
        assert_eq!(hr(f.CreateBitmap(0,1,&GUID_WICPixelFormat24bppBGR,WICBitmapCacheOnLoad)),0x80070057);
        let b=f.CreateBitmapFromMemory(3,2,&GUID_WICPixelFormat24bppBGR,12,&data).unwrap();
        let lock=b.Lock(std::ptr::null(),WICBitmapLockRead.0 as u32).unwrap();
        assert_eq!(lock.GetStride().unwrap(),12);
        let(mut len,mut p)=(0,std::ptr::null_mut());lock.GetDataPointer(&mut len,&mut p).unwrap();assert_eq!(len,21);
        drop(lock);
        let mut out=[0xcc;24];
        b.CopyPixels(std::ptr::null(),12,&mut out).unwrap();
        assert_eq!(&out[..9],&data[..9]);assert_eq!(&out[9..12],&[0xcc;3]);assert_eq!(&out[12..21],&data[12..]);assert_eq!(&out[21..],&[0xcc;3]);
        for (r,expected) in [(WICRect{X:-1,Y:0,Width:1,Height:1},0x80070216),
            (WICRect{X:0,Y:0,Width:0,Height:1},0x80070057),
            (WICRect{X:2,Y:0,Width:2,Height:1},0x80070057)] {
            assert_eq!(hr(b.CopyPixels(&r,12,&mut out)),expected);
        }
        assert_eq!(hr(b.CopyPixels(std::ptr::null(),12,&mut out[..20])),0x88982f8c);
    }
}

#[test]
fn bitmap_lock_lifetime_and_exclusion() {
    unsafe {
        let f=factory();let b=f.CreateBitmap(3,2,&GUID_WICPixelFormat24bppBGR,WICBitmapCacheOnLoad).unwrap();
        assert_eq!(hr(b.Lock(std::ptr::null(),0)),0x80070057);
        assert_eq!(hr(b.Lock(std::ptr::null(),4)),0x80070057);
        let read=b.Lock(std::ptr::null(),1).unwrap();let read2=b.Lock(std::ptr::null(),1).unwrap();
        assert_eq!(hr(b.Lock(std::ptr::null(),2)),0x88982f0d);drop(read);drop(read2);
        let write=b.Lock(std::ptr::null(),2).unwrap();
        assert_eq!(hr(b.Lock(std::ptr::null(),1)),0x88982f0d);assert_eq!(hr(b.Lock(std::ptr::null(),2)),0x88982f0d);
        let mut out=[0;32];assert_eq!(hr(b.CopyPixels(std::ptr::null(),12,&mut out)),0x88982f0d);
        let(mut len,mut p)=(0,std::ptr::null_mut());write.GetDataPointer(&mut len,&mut p).unwrap();*p=0xa7;drop(write);
        b.CopyPixels(std::ptr::null(),12,&mut out).unwrap();assert_eq!(out[0],0xa7);
        let lock=b.Lock(std::ptr::null(),5).unwrap();drop(b);lock.GetDataPointer(&mut len,&mut p).unwrap();assert_eq!(*p,0xa7);
    }
}

#[test]
fn packed_bitmap_subrect_writeback() {
    unsafe {
        let f=factory();
        for (format,bpp) in [(GUID_WICPixelFormat1bppIndexed,1),(GUID_WICPixelFormat2bppIndexed,2),(GUID_WICPixelFormat4bppIndexed,4)] {
            let stride=(9*bpp+7)/8;
            let data=vec![0x96;stride as usize*2];
            let b=f.CreateBitmapFromMemory(9,2,&format,stride,&data).unwrap();
            let r=WICRect {X:1,Y:0,Width:3,Height:2};
            let lock=b.Lock(&r,3).unwrap();assert_eq!(lock.GetStride().unwrap(),4);
            let(mut size,mut p)=(0,std::ptr::null_mut());lock.GetDataPointer(&mut size,&mut p).unwrap();assert_eq!(size,8);
            for i in 0..4 {*p.add(i)=0xff;*p.add(4+i)=0;}drop(lock);
            let mut out=vec![0;data.len()];b.CopyPixels(std::ptr::null(),stride,&mut out).unwrap();
            for y in 0..2 {for x in 0..9*bpp {
                let bit=(out[y*stride as usize+x as usize/8] >> (7-(x&7)))&1;
                let expected=if x>=bpp && x<4*bpp {if y==0 {1}else{0}}else{(0x96 >> (7-(x&7)))&1};
                assert_eq!(bit,expected,"bpp{bpp} y{y} bit{x}");
            }}
        }
    }
}
