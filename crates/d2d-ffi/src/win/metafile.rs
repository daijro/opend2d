//! `ID2D1GdiMetafile` / `ID2D1GdiMetafile1` COM object, wrapping the core [`GdiMetafile`].
//!
//! Direct2D delegates all EMF/EMF+ work to the external `CreateMetafileRenderer` apiset,
//! so only the deterministic, reproducible surface is wired here: the header
//! geometry the accessors report (`GetBounds`/`GetDpi`/`GetSourceBounds`) and the record enumeration
//! `Stream` feeds an `ID2D1GdiMetafileSink`. Playback lives in the core (`d2d::metafile`); its pixel
//! parity ceiling is documented there.

use super::E_NOTIMPL;
use d2d::metafile::GdiMetafile;
use windows::Win32::Graphics::Direct2D::Common::D2D_RECT_F;
use windows::Win32::Graphics::Direct2D::*;
use windows_core::{Ref, Result as WResult, implement};

#[implement(ID2D1GdiMetafile1)]
pub(crate) struct MetafileObj {
    pub(crate) mf: GdiMetafile,
    pub(crate) factory: Option<ID2D1Factory>,
}

fn rect(r: [f32; 4]) -> D2D_RECT_F {
    D2D_RECT_F { left: r[0], top: r[1], right: r[2], bottom: r[3] }
}

impl ID2D1Resource_Impl for MetafileObj_Impl {
    fn GetFactory(&self, f: windows_core::OutRef<ID2D1Factory>) {
        let _ = f.write(self.factory.clone());
    }
}
impl ID2D1GdiMetafile_Impl for MetafileObj_Impl {
    fn Stream(&self, sink: Ref<ID2D1GdiMetafileSink>) -> WResult<()> {
        let sink = sink.as_ref().ok_or(windows_core::Error::from(E_NOTIMPL))?;
        for (itype, record) in self.mf.records() {
            // ProcessRecord receives the record's data — the bytes after the 8-byte {iType,nSize}
            // EMR header — and its size.
            let data = &record[8..];
            unsafe { sink.ProcessRecord(itype, Some(data.as_ptr().cast()), data.len() as u32)? };
        }
        Ok(())
    }
    fn GetBounds(&self) -> WResult<D2D_RECT_F> {
        Ok(rect(self.mf.bounds()))
    }
}
impl ID2D1GdiMetafile1_Impl for MetafileObj_Impl {
    fn GetDpi(&self, dpix: *mut f32, dpiy: *mut f32) -> WResult<()> {
        let (x, y) = self.mf.dpi();
        unsafe {
            *dpix = x;
            *dpiy = y;
        }
        Ok(())
    }
    fn GetSourceBounds(&self) -> WResult<D2D_RECT_F> {
        Ok(rect(self.mf.source_bounds()))
    }
}
