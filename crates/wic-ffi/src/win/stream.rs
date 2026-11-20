//! Native windowscodecs stream wrappers (memory stream + sub-stream), Windows 11 26100.
use super::e;
use std::cell::RefCell;
use std::fs::File;
use std::io::{Read,Write,Seek,SeekFrom};
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Imaging::*;
use windows::Win32::System::Com::*;
use windows_core::{HRESULT,Ref,Result as WResult,implement};

#[derive(Default)]
enum Back {
    #[default]
    None,
    File(File),
    Memory { data:*mut u8, len:u32, pos:u32 },
    Stream(IStream),
    Region { stream:IStream, offset:u64, len:u64, pos:u64 },
}

#[implement(IWICStream)]
#[derive(Default)]
pub(crate) struct StreamObj { back:RefCell<Back> }

impl StreamObj {
    fn uninitialized(&self) -> WResult<()> {
        if matches!(*self.back.borrow(),Back::None) { Ok(()) } else { Err(e(WINCODEC_ERR_WRONGSTATE)) }
    }
}

impl IWICStream_Impl for StreamObj_Impl {
    fn InitializeFromIStream(&self,s:Ref<IStream>) -> WResult<()> {
        let stream=s.as_ref().ok_or(e(E_INVALIDARG))?;
        self.uninitialized()?;
        *self.back.borrow_mut()=Back::Stream(stream.clone());Ok(())
    }
    fn InitializeFromFilename(&self,path:&windows_core::PCWSTR,access:u32) -> WResult<()> {
        if path.is_null() { return Err(e(E_INVALIDARG)); }
        self.uninitialized()?;
        let path=unsafe {path.to_string()}.map_err(|_|e(E_INVALIDARG))?;
        let (read,write)=(access&0x80000000!=0,access&0x40000000!=0);
        let file=if read&&write {std::fs::OpenOptions::new().read(true).write(true).open(path)}
            else if write {File::create(path)} else {File::open(path)};
        *self.back.borrow_mut()=Back::File(file.map_err(|_|e(WINCODEC_ERR_WIN32ERROR))?);Ok(())
    }
    fn InitializeFromMemory(&self,data:*const u8,len:u32) -> WResult<()> {
        if data.is_null() {return Err(e(E_INVALIDARG));}
        self.uninitialized()?;
        *self.back.borrow_mut()=Back::Memory {data:data as *mut u8,len,pos:0};Ok(())
    }
    fn InitializeFromIStreamRegion(&self,s:Ref<IStream>,offset:u64,len:u64) -> WResult<()> {
        let stream=s.as_ref().ok_or(e(E_INVALIDARG))?;
        self.uninitialized()?;
        offset.checked_add(len).ok_or(e(HRESULT(0x80070216u32 as i32)))?;
        *self.back.borrow_mut()=Back::Region {stream:stream.clone(),offset,len,pos:0};Ok(())
    }
}

fn region_io(stream:&IStream,offset:u64,pos:&mut u64,len:u64,mut io:impl FnMut(u32,*mut u32)->HRESULT,count:u32,actual:*mut u32) -> HRESULT {
    let mut saved=0;
    let result=(|| -> WResult<HRESULT> {unsafe {
        stream.Seek(0,STREAM_SEEK_CUR,Some(&mut saved))?;
        let absolute=offset.checked_add(*pos).filter(|&n|n<=i64::MAX as u64).ok_or(e(HRESULT(0x80070216u32 as i32)))?;
        stream.Seek(absolute as i64,STREAM_SEEK_SET,None)?;
        let mut got=0;
        let result=io((count as u64).min(len.saturating_sub(*pos)) as u32,&mut got);
        let restore=stream.Seek(saved as i64,STREAM_SEEK_SET,None);
        result.ok()?;restore?;
        *pos+=got as u64;
        if !actual.is_null() {*actual=got;}
        Ok(result)
    }})();
    result.unwrap_or_else(|e|e.code())
}

impl ISequentialStream_Impl for StreamObj_Impl {
    fn Read(&self,pv:*mut core::ffi::c_void,cb:u32,actual:*mut u32) -> HRESULT {
        let mut back=self.back.borrow_mut();
        if matches!(*back,Back::None) {return WINCODEC_ERR_NOTINITIALIZED;}
        if pv.is_null() {return E_INVALIDARG;}
        let count=match &mut *back {
            Back::None=>return WINCODEC_ERR_NOTINITIALIZED,
            Back::Memory {data,len,pos}=>{
                let count=cb.min(*len-*pos);
                unsafe {core::ptr::copy((*data).add(*pos as usize),pv as *mut u8,count as usize);}
                *pos+=count;count
            }
            Back::File(f)=>match f.read(unsafe {core::slice::from_raw_parts_mut(pv as *mut u8,cb as usize)}) {Ok(n)=>n as u32,Err(_)=>return E_FAIL},
            Back::Stream(s)=>return unsafe {s.Read(pv,cb,Some(actual))},
            Back::Region {stream,offset,len,pos}=>return region_io(stream,*offset,pos,*len,|n,p|unsafe {stream.Read(pv,n,Some(p))},cb,actual),
        };
        if !actual.is_null() {unsafe {*actual=count;}}
        S_OK
    }
    fn Write(&self,pv:*const core::ffi::c_void,cb:u32,actual:*mut u32) -> HRESULT {
        let mut back=self.back.borrow_mut();
        if matches!(*back,Back::None) {return WINCODEC_ERR_NOTINITIALIZED;}
        if pv.is_null() {return E_INVALIDARG;}
        match &mut *back {
            Back::None=>return WINCODEC_ERR_NOTINITIALIZED,
            Back::Memory {data,len,pos}=>{
                let Some(end)=pos.checked_add(cb) else {return HRESULT(0x80070216u32 as i32)};
                if end>*len {return STG_E_MEDIUMFULL;}
                unsafe {core::ptr::copy(pv as *const u8,(*data).add(*pos as usize),cb as usize);}
                *pos=end;
            }
            Back::File(f)=>if f.write_all(unsafe {core::slice::from_raw_parts(pv as *const u8,cb as usize)}).is_err() {return E_FAIL;},
            Back::Stream(s)=>return unsafe {s.Write(pv,cb,Some(actual))},
            Back::Region {stream,offset,len,pos}=>return region_io(stream,*offset,pos,*len,|n,p|unsafe {stream.Write(pv,n,Some(p))},cb,actual),
        }
        if !actual.is_null() {unsafe {*actual=cb;}}
        S_OK
    }
}

fn position(pos:u64,len:u64,offset:i64,origin:STREAM_SEEK,max:u64) -> WResult<u64> {
    let base=match origin {STREAM_SEEK_SET=>0,STREAM_SEEK_CUR=>pos,STREAM_SEEK_END=>len,_=>return Err(e(E_INVALIDARG))};
    let next=base as i128+offset as i128;
    if next<0 || next>max as i128 {return Err(e(HRESULT(0x80070216u32 as i32)));}
    if next>len as i128 {return Err(e(E_INVALIDARG));}
    Ok(next as u64)
}

impl IStream_Impl for StreamObj_Impl {
    fn Seek(&self,offset:i64,origin:STREAM_SEEK,out:*mut u64) -> WResult<()> {
        let mut back=self.back.borrow_mut();
        let next=match &mut *back {
            Back::None=>return Err(e(WINCODEC_ERR_NOTINITIALIZED)),
            Back::Stream(s)=>return unsafe {s.Seek(offset,origin,Some(out))},
            Back::Memory {len,pos,..}=>{let next=position(*pos as u64,*len as u64,offset,origin,u32::MAX as u64)?;*pos=next as u32;next},
            Back::Region {stream,offset:base,len,pos}=>{
                let mut stat=STATSTG::default();unsafe {stream.Stat(&mut stat,STATFLAG_NONAME)?;}
                let size=(*len).min(stat.cbSize.saturating_sub(*base));
                let next=position(*pos,size,offset,origin,i64::MAX as u64)?;*pos=next;next
            }
            Back::File(f)=>{
                let from=match origin {STREAM_SEEK_SET if offset>=0=>SeekFrom::Start(offset as u64),STREAM_SEEK_CUR=>SeekFrom::Current(offset),STREAM_SEEK_END=>SeekFrom::End(offset),_=>return Err(e(E_INVALIDARG))};
                f.seek(from).map_err(|_|e(E_FAIL))?
            }
        };
        if !out.is_null() {unsafe {*out=next;}}Ok(())
    }
    fn SetSize(&self,size:u64) -> WResult<()> {
        match &mut *self.back.borrow_mut() {
            Back::None=>Err(e(WINCODEC_ERR_NOTINITIALIZED)),
            Back::Stream(s)=>unsafe {s.SetSize(size)},
            Back::File(f)=>f.set_len(size).map_err(|_|e(E_FAIL)),
            _=>Err(e(E_NOTIMPL)),
        }
    }
    fn CopyTo(&self,dest:Ref<IStream>,count:u64,read:*mut u64,written:*mut u64) -> WResult<()> {
        let dest=dest.as_ref().ok_or(e(E_INVALIDARG))?;
        let(mut total,mut output)=(0,0);
        let mut buf=[0u8;65536];
        while total<count {
            let mut got=0;
            self.Read(buf.as_mut_ptr() as *mut _,(count-total).min(buf.len() as u64) as u32,&mut got).ok()?;
            if got==0 {break;}
            let mut sent=0;unsafe {dest.Write(buf.as_ptr() as *const _,got,Some(&mut sent))}.ok()?;
            total+=got as u64;output+=sent as u64;
            if sent<got {break;}
        }
        unsafe {if !read.is_null() {*read=total;}if !written.is_null() {*written=output;}}Ok(())
    }
    fn Commit(&self,flags:&STGC) -> WResult<()> {
        match &mut *self.back.borrow_mut() {
            Back::None=>Err(e(WINCODEC_ERR_NOTINITIALIZED)),
            Back::File(f)=>f.flush().map_err(|_|e(E_FAIL)),
            Back::Stream(s)|Back::Region {stream:s,..}=>unsafe {s.Commit(*flags)},
            _=>Ok(()),
        }
    }
    fn Revert(&self) -> WResult<()> {
        match &*self.back.borrow() {Back::None=>Err(e(WINCODEC_ERR_NOTINITIALIZED)),Back::Stream(s)|Back::Region {stream:s,..}=>unsafe {s.Revert()},_=>Err(e(E_NOTIMPL))}
    }
    fn LockRegion(&self,offset:u64,count:u64,kind:&LOCKTYPE) -> WResult<()> {
        match &*self.back.borrow() {Back::None=>Err(e(WINCODEC_ERR_NOTINITIALIZED)),Back::Stream(s)=>unsafe {s.LockRegion(offset,count,*kind)},_=>Err(e(E_NOTIMPL))}
    }
    fn UnlockRegion(&self,offset:u64,count:u64,kind:u32) -> WResult<()> {
        match &*self.back.borrow() {Back::None=>Err(e(WINCODEC_ERR_NOTINITIALIZED)),Back::Stream(s)=>unsafe {s.UnlockRegion(offset,count,kind)},_=>Err(e(E_NOTIMPL))}
    }
    fn Stat(&self,out:*mut STATSTG,flags:&STATFLAG) -> WResult<()> {
        let back=self.back.borrow();
        if matches!(*back,Back::None) {return Err(e(WINCODEC_ERR_NOTINITIALIZED));}
        if out.is_null() {return Err(e(E_INVALIDARG));}
        let size=match &*back {
            Back::None=>return Err(e(WINCODEC_ERR_NOTINITIALIZED)),
            Back::Stream(s)=>return unsafe {s.Stat(out,*flags)},
            Back::Region {stream,offset,len,..}=>{
                unsafe {stream.Stat(out,*flags)?;(*out).cbSize=(*out).cbSize.saturating_sub(*offset).min(*len);}return Ok(());
            }
            Back::Memory {len,..}=>*len as u64,
            Back::File(f)=>f.metadata().map_err(|_|e(E_FAIL))?.len(),
        };
        unsafe {*out=STATSTG {r#type:STGTY_STREAM.0 as u32,cbSize:size,..Default::default()};}Ok(())
    }
    fn Clone(&self) -> WResult<IStream> {
        match &*self.back.borrow() {Back::None=>Err(e(WINCODEC_ERR_NOTINITIALIZED)),Back::Stream(s)=>unsafe {s.Clone()},_=>Err(e(E_NOTIMPL))}
    }
}
