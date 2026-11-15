//! Metadata query readers — `GetMetadataQueryReader` on the decoder and each frame.
//!
//! A small in-memory tree ([`Value`] leaves / `Value::Sub` nested readers) is parsed out of the raw
//! container bytes per format, then served through `IWICMetadataQueryReader`: `GetMetadataByName`
//! navigates a `/a/b/c` path and returns a `PROPVARIANT` (a nested block comes back as `VT_UNKNOWN`
//! wrapping another reader), `GetEnumerator` lists the immediate child names. Paths, VARIANT types
//! and tree shape are the exact ones native WIC produces (captured on the reference VM):
//! JPEG `/app0` (JFIF) and `/app1` (EXIF IFD tree, with `/{ushort=0}` IFDs and
//! sub-IFDs), the DQT `/luminance`+`/chrominance`, TIFF `/ifd`, PNG text/gamma/time chunks, and GIF
//! `/logscrdesc` (decoder) + `/imgdesc` (frame).

use super::components::put_wstr;
use super::e;
use d2d::wic::{xmp_packet, GifGce, GifMeta, TiffField};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Imaging::*;
use windows::Win32::System::Com::IEnumString;
use windows::Win32::System::Com::CoTaskMemAlloc;
use windows::Win32::System::Com::{IStream, STREAM_SEEK_SET};
use windows::Win32::System::Com::StructuredStorage::PROPVARIANT;
use windows_core::{GUID, HRESULT, Interface, PCWSTR, PWSTR, Result as WResult, implement};

// Container GUIDs.
const PNG: u128 = 0x1b7cfaf4_713f_473c_bbcd_6137425faeaf;
const JPEG: u128 = 0x19e4a5aa_5662_4fc5_a0c0_1758028e1057;
const TIFF: u128 = 0x163bcc30_e2e9_4f0b_961d_a3e9fdb788a3;
pub(crate) const GIF: u128 = 0x1f8a5601_7d4d_4cbd_9c82_1bc8d4eeb9a5;
const PROPERTYNOTFOUND: u32 = 0x8898_2F40;

/// One metadata node: a leaf PROPVARIANT value, or a nested reader (`VT_UNKNOWN`).
#[derive(Clone)]
enum Value {
    U1(u8),
    U2(u16),
    U4(u32),
    U8(u64),
    Bool(bool),
    Str(String),  // VT_LPSTR
    WStr(String), // VT_LPWSTR
    Blob(Vec<u8>),
    VecU1(Vec<u8>),
    VecU2(Vec<u16>),
    VecU4(Vec<u32>),
    VecU8(Vec<u64>),
    Sub(Tree),
}

/// An ordered, mutable map of child name (`/name`) → value. Shared via `Rc<RefCell>` so a reader
/// borrows it and a query writer (and its nested sub-writers) mutate it in place.
type Tree = Rc<RefCell<Vec<(String, Value)>>>;

fn tree(v: Vec<(String, Value)>) -> Tree {
    Rc::new(RefCell::new(v))
}

// --- byte readers ------------------------------------------------------------------------------

fn rd16(b: &[u8], o: usize, le: bool) -> u16 {
    match b.get(o..o + 2) {
        Some(x) if le => u16::from_le_bytes([x[0], x[1]]),
        Some(x) => u16::from_be_bytes([x[0], x[1]]),
        None => 0,
    }
}
fn rd32(b: &[u8], o: usize, le: bool) -> u32 {
    match b.get(o..o + 4) {
        Some(x) if le => u32::from_le_bytes([x[0], x[1], x[2], x[3]]),
        Some(x) => u32::from_be_bytes([x[0], x[1], x[2], x[3]]),
        None => 0,
    }
}

// --- PROPVARIANT construction (raw layout + CoTaskMemAlloc) ------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
struct PvCa {
    celems: u32,
    pelems: *mut core::ffi::c_void,
}
#[repr(C)]
union PvVal {
    u8_: u8,
    i16_: i16,
    u16_: u16,
    u32_: u32,
    u64_: u64,
    ptr: *mut core::ffi::c_void,
    ca: PvCa,
}
#[repr(C)]
struct PvRaw {
    vt: u16,
    _r: [u16; 3],
    val: PvVal,
}

/// CoTaskMem copy of `bytes` (NUL-free); null for an empty slice.
unsafe fn alloc(bytes: &[u8]) -> *mut core::ffi::c_void {
    if bytes.is_empty() {
        return core::ptr::null_mut();
    }
    let p = unsafe { CoTaskMemAlloc(bytes.len()) };
    if !p.is_null() {
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), p as *mut u8, bytes.len()) };
    }
    p
}

fn u16s_to_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u32s_to_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u64s_to_bytes(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Build a `PROPVARIANT` for a value. Pointers are CoTaskMem (the caller's `PropVariantClear` frees
/// them); a nested reader is an AddRef'd `IWICMetadataQueryReader` under `VT_UNKNOWN`.
fn build_pv(v: &Value, container: GUID, location: &str, writer: bool, strict: bool) -> PROPVARIANT {
    let mut pv = PROPVARIANT::default();
    let r = &mut pv as *mut PROPVARIANT as *mut PvRaw;
    unsafe {
        match v {
            Value::U1(x) => { (*r).vt = 17; (*r).val.u8_ = *x; }
            Value::U2(x) => { (*r).vt = 18; (*r).val.u16_ = *x; }
            Value::U4(x) => { (*r).vt = 19; (*r).val.u32_ = *x; }
            Value::U8(x) => { (*r).vt = 21; (*r).val.u64_ = *x; }
            Value::Bool(b) => { (*r).vt = 11; (*r).val.i16_ = if *b { -1 } else { 0 }; }
            Value::Str(s) => {
                let mut bytes: Vec<u8> = s.bytes().collect();
                bytes.push(0);
                (*r).vt = 30;
                (*r).val.ptr = alloc(&bytes);
            }
            Value::WStr(s) => {
                let mut w: Vec<u16> = s.encode_utf16().collect();
                w.push(0);
                (*r).vt = 31;
                (*r).val.ptr = alloc(&u16s_to_bytes(&w));
            }
            Value::Blob(b) => { (*r).vt = 65; (*r).val.ca = PvCa { celems: b.len() as u32, pelems: alloc(b) }; }
            Value::VecU1(b) => { (*r).vt = 0x1011; (*r).val.ca = PvCa { celems: b.len() as u32, pelems: alloc(b) }; }
            Value::VecU2(x) => { (*r).vt = 0x1012; (*r).val.ca = PvCa { celems: x.len() as u32, pelems: alloc(&u16s_to_bytes(x)) }; }
            Value::VecU4(x) => { (*r).vt = 0x1013; (*r).val.ca = PvCa { celems: x.len() as u32, pelems: alloc(&u32s_to_bytes(x)) }; }
            Value::VecU8(x) => { (*r).vt = 0x1015; (*r).val.ca = PvCa { celems: x.len() as u32, pelems: alloc(&u64s_to_bytes(x)) }; }
            Value::Sub(t) => {
                // A nested block comes back as a reader or (from a query writer) a live sub-writer
                // sharing the same tree, so edits through it reach the parent.
                let unk: windows_core::IUnknown = if writer {
                    let w: IWICMetadataQueryWriter = QueryWriterObj { tree: t.clone(), container, location: location.to_string(), strict }.into();
                    w.into()
                } else {
                    let rd: IWICMetadataQueryReader = QueryReaderObj { tree: t.clone(), container, location: location.to_string() }.into();
                    rd.into()
                };
                (*r).vt = 13;
                (*r).val.ptr = unk.into_raw();
            }
        }
    }
    pv
}

/// A `CA*` counted-array as a `Vec<T>` (empty when the pointer is null or the count is zero — WIC
/// uses a null `pElems` for an empty blob/vector).
unsafe fn ca_vec<T: Copy>(ca: &PvCa) -> Vec<T> {
    if ca.pelems.is_null() || ca.celems == 0 {
        Vec::new()
    } else {
        unsafe { core::slice::from_raw_parts(ca.pelems as *const T, ca.celems as usize) }.to_vec()
    }
}

/// Parse an incoming `PROPVARIANT` (from `SetMetadataByName`) into a [`Value`]. Common leaf types
/// only; an unrecognised variant yields `None` (the set is rejected).
unsafe fn pv_to_value(pv: *const PROPVARIANT) -> Option<Value> {
    let r = unsafe { &*(pv as *const PvRaw) };
    let read_ptr_str = |p: *const u8| -> String {
        let mut s = Vec::new();
        let mut q = p;
        while !q.is_null() && unsafe { *q } != 0 {
            s.push(unsafe { *q });
            q = unsafe { q.add(1) };
        }
        s.iter().map(|&c| c as char).collect()
    };
    unsafe {
        Some(match r.vt {
            17 => Value::U1(r.val.u8_),
            18 => Value::U2(r.val.u16_),
            19 => Value::U4(r.val.u32_),
            21 => Value::U8(r.val.u64_),
            11 => Value::Bool(r.val.i16_ != 0),
            30 => Value::Str(read_ptr_str(r.val.ptr as *const u8)),
            31 => {
                let mut w = Vec::new();
                let mut q = r.val.ptr as *const u16;
                while !q.is_null() && *q != 0 {
                    w.push(*q);
                    q = q.add(1);
                }
                Value::WStr(String::from_utf16_lossy(&w))
            }
            65 => Value::Blob(ca_vec::<u8>(&r.val.ca)),
            0x1011 => Value::VecU1(ca_vec::<u8>(&r.val.ca)),
            0x1012 => Value::VecU2(ca_vec::<u16>(&r.val.ca)),
            0x1013 => Value::VecU4(ca_vec::<u32>(&r.val.ca)),
            0x1015 => Value::VecU8(ca_vec::<u64>(&r.val.ca)),
            _ => return None,
        })
    }
}

// --- shared tree navigation --------------------------------------------------------------------

/// Resolve `/a/b/c` to the tree holding the final component and that component's key. With `create`,
/// missing intermediate blocks are created as empty sub-trees (query-writer auto-create).
fn navigate(root: &Tree, path: &str, create: bool) -> Option<(Tree, String)> {
    let parts: Vec<&str> = path.trim_start_matches('/').split('/').filter(|p| !p.is_empty()).collect();
    if parts.is_empty() {
        return None;
    }
    let mut cur = root.clone();
    for part in &parts[..parts.len() - 1] {
        let key = format!("/{part}");
        let next = {
            let mut b = cur.borrow_mut();
            match b.iter().find(|(k, _)| *k == key) {
                Some((_, Value::Sub(t))) => Some(t.clone()),
                Some(_) => None,
                None if create => {
                    let t = tree(Vec::new());
                    b.push((key, Value::Sub(t.clone())));
                    Some(t)
                }
                None => None,
            }
        };
        cur = next?;
    }
    Some((cur, format!("/{}", parts[parts.len() - 1])))
}

/// The immediate child names of a tree, NUL-terminated UTF-16 (for `IEnumString`).
fn enum_names(t: &Tree) -> Vec<Vec<u16>> {
    t.borrow().iter().map(|(k, _)| k.encode_utf16().chain([0]).collect()).collect()
}

/// `GetMetadataByName` for either a reader (`writer=false`) or writer (`writer=true`).
fn get_metadata(t: &Tree, container: GUID, location: &str, name: &PCWSTR, pv: *mut PROPVARIANT, writer: bool, strict: bool) -> WResult<()> {
    let path = unsafe { name.to_string() }.map_err(|_| e(E_INVALIDARG))?;
    let (parent, key) = navigate(t, &path, false).ok_or(e(HRESULT(PROPERTYNOTFOUND as i32)))?;
    let v = parent.borrow().iter().find(|(k, _)| *k == key).map(|(_, v)| v.clone()).ok_or(e(HRESULT(PROPERTYNOTFOUND as i32)))?;
    let loc = format!("{}{}", location.trim_end_matches('/'), key);
    unsafe { pv.write(build_pv(&v, container, &loc, writer, strict)) };
    Ok(())
}

// --- the query reader COM object ---------------------------------------------------------------

#[implement(IWICMetadataQueryReader)]
struct QueryReaderObj {
    tree: Tree,
    container: GUID,
    location: String,
}

impl IWICMetadataQueryReader_Impl for QueryReaderObj_Impl {
    fn GetContainerFormat(&self) -> WResult<GUID> {
        Ok(self.container)
    }
    fn GetLocation(&self, cch: u32, wz: PWSTR, actual: *mut u32) -> WResult<()> {
        put_wstr(&self.location, cch, wz, actual)
    }
    fn GetMetadataByName(&self, name: &PCWSTR, pv: *mut PROPVARIANT) -> WResult<()> {
        get_metadata(&self.tree, self.container, &self.location, name, pv, false, false)
    }
    fn GetEnumerator(&self) -> WResult<IEnumString> {
        Ok(EnumStringObj { names: enum_names(&self.tree), pos: Cell::new(0) }.into())
    }
}

/// The query writer: the reader surface plus `SetMetadataByName`/`RemoveMetadataByName`, mutating the
/// shared tree. `strict` writers (the fast metadata encoder) reject paths whose block does not exist
/// (as native's schema-backed writers do); non-strict writers (`CreateQueryWriter[FromReader]`)
/// auto-create intermediate blocks.
#[implement(IWICMetadataQueryWriter)]
struct QueryWriterObj {
    tree: Tree,
    container: GUID,
    location: String,
    strict: bool,
}

impl IWICMetadataQueryReader_Impl for QueryWriterObj_Impl {
    fn GetContainerFormat(&self) -> WResult<GUID> {
        Ok(self.container)
    }
    fn GetLocation(&self, cch: u32, wz: PWSTR, actual: *mut u32) -> WResult<()> {
        put_wstr(&self.location, cch, wz, actual)
    }
    fn GetMetadataByName(&self, name: &PCWSTR, pv: *mut PROPVARIANT) -> WResult<()> {
        get_metadata(&self.tree, self.container, &self.location, name, pv, true, self.strict)
    }
    fn GetEnumerator(&self) -> WResult<IEnumString> {
        Ok(EnumStringObj { names: enum_names(&self.tree), pos: Cell::new(0) }.into())
    }
}

impl IWICMetadataQueryWriter_Impl for QueryWriterObj_Impl {
    fn SetMetadataByName(&self, name: &PCWSTR, pv: *const PROPVARIANT) -> WResult<()> {
        let path = unsafe { name.to_string() }.map_err(|_| e(E_INVALIDARG))?;
        let value = unsafe { pv_to_value(pv) }.ok_or(e(WINCODEC_ERR_UNSUPPORTEDOPERATION))?;
        let notfound = || e(HRESULT(PROPERTYNOTFOUND as i32));
        if self.strict {
            // Edit an existing value only; a path whose block is absent is rejected, as native does.
            let (parent, key) = navigate(&self.tree, &path, false).ok_or_else(notfound)?;
            let mut b = parent.borrow_mut();
            b.iter_mut().find(|(k, _)| *k == key).ok_or_else(notfound)?.1 = value;
        } else {
            let (parent, key) = navigate(&self.tree, &path, true).ok_or(e(E_INVALIDARG))?;
            let mut b = parent.borrow_mut();
            // An unindexed PNG `/tEXt/{str=KEYWORD}` write targets the first tEXt block, which holds a
            // single keyword — so native replaces whatever keyword was there. Writing a second
            // unindexed tEXt (any keyword) drops the first (verified: setting Comment then Title emits
            // only a `Title` tEXt chunk). Collapse to one entry; indexed `/[n]tEXt` paths are distinct
            // blocks and navigate to their own subtrees, so they are unaffected.
            if path.starts_with("/tEXt/{str=") {
                b.clear();
            }
            if let Some(entry) = b.iter_mut().find(|(k, _)| *k == key) {
                entry.1 = value;
            } else {
                b.push((key, value));
            }
        }
        Ok(())
    }
    fn RemoveMetadataByName(&self, name: &PCWSTR) -> WResult<()> {
        let path = unsafe { name.to_string() }.map_err(|_| e(E_INVALIDARG))?;
        let (parent, key) = navigate(&self.tree, &path, false).ok_or(e(HRESULT(PROPERTYNOTFOUND as i32)))?;
        let mut b = parent.borrow_mut();
        let before = b.len();
        b.retain(|(k, _)| *k != key);
        if b.len() == before {
            return Err(e(HRESULT(PROPERTYNOTFOUND as i32)));
        }
        Ok(())
    }
}

/// `IEnumString` over a reader's child names (each copy CoTaskMem for the caller to free).
#[implement(IEnumString)]
struct EnumStringObj {
    names: Vec<Vec<u16>>,
    pos: Cell<usize>,
}

impl windows::Win32::System::Com::IEnumString_Impl for EnumStringObj_Impl {
    fn Next(&self, celt: u32, rgelt: *mut PWSTR, fetched: *mut u32) -> HRESULT {
        let out = unsafe { core::slice::from_raw_parts_mut(rgelt, celt as usize) };
        let mut n = 0usize;
        while n < celt as usize && self.pos.get() < self.names.len() {
            let w = &self.names[self.pos.get()];
            let p = unsafe { alloc(&u16s_to_bytes(w)) } as *mut u16;
            out[n] = PWSTR(p);
            self.pos.set(self.pos.get() + 1);
            n += 1;
        }
        if !fetched.is_null() {
            unsafe { *fetched = n as u32 };
        }
        if n == celt as usize { S_OK } else { S_FALSE }
    }
    fn Skip(&self, celt: u32) -> HRESULT {
        self.pos.set((self.pos.get() + celt as usize).min(self.names.len()));
        S_OK
    }
    fn Reset(&self) -> WResult<()> {
        self.pos.set(0);
        Ok(())
    }
    fn Clone(&self) -> WResult<IEnumString> {
        Ok(EnumStringObj { names: self.names.clone(), pos: Cell::new(self.pos.get()) }.into())
    }
}

// --- dispatch ----------------------------------------------------------------------------------

fn reader_obj(entries: Vec<(String, Value)>, container: u128) -> IWICMetadataQueryReader {
    QueryReaderObj { tree: tree(entries), container: GUID::from_u128(container), location: "/".into() }.into()
}

pub(crate) fn frame_query_reader(container: GUID, raw: Rc<Vec<u8>>, index: u32) -> WResult<IWICMetadataQueryReader> {
    let c = container.to_u128();
    let entries = match c {
        PNG => parse_png(&raw),
        JPEG => parse_jpeg_frame(&raw),
        TIFF => parse_tiff_frame(&raw, index),
        GIF => parse_gif_frame(&raw, index),
        _ => return Err(e(WINCODEC_ERR_UNSUPPORTEDOPERATION)),
    };
    Ok(reader_obj(entries, c))
}

pub(crate) fn decoder_query_reader(container: GUID, raw: Rc<Vec<u8>>) -> WResult<IWICMetadataQueryReader> {
    let c = container.to_u128();
    // GIF has container-level metadata (logical screen descriptor); the others keep all metadata at
    // the frame level, so the decoder reader is empty (native shows no decoder-level entries).
    let entries = if c == GIF { parse_gif_logscr(&raw) } else { Vec::new() };
    Ok(reader_obj(entries, c))
}

/// Deep-copy an arbitrary query reader's tree (read through the public ABI, so any reader works).
fn copy_reader_to_tree(reader: &IWICMetadataQueryReader) -> Tree {
    let mut entries: Vec<(String, Value)> = Vec::new();
    if let Ok(en) = unsafe { reader.GetEnumerator() } {
        loop {
            let mut name = PWSTR::null();
            let mut got = 0u32;
            if unsafe { en.Next(core::slice::from_mut(&mut name), Some(&mut got)) }.is_err() || got == 0 {
                break;
            }
            let key = unsafe { name.to_string() }.unwrap_or_default();
            let mut pv = PROPVARIANT::default();
            if unsafe { reader.GetMetadataByName(PCWSTR(name.as_ptr()), &mut pv) }.is_err() {
                continue;
            }
            let raw = unsafe { &*(&pv as *const PROPVARIANT as *const PvRaw) };
            if raw.vt == 13 {
                let p = unsafe { raw.val.ptr };
                if let Some(unk) = unsafe { windows_core::IUnknown::from_raw_borrowed(&p) } {
                    if let Ok(sub) = unk.cast::<IWICMetadataQueryReader>() {
                        entries.push((key, Value::Sub(copy_reader_to_tree(&sub))));
                    }
                }
            } else if let Some(v) = unsafe { pv_to_value(&pv) } {
                entries.push((key, v));
            }
            // StructuredStorage::PROPVARIANT has no Drop, so free its CoTaskMem / interface ref.
            unsafe { windows::Win32::System::Com::StructuredStorage::PropVariantClear(&mut pv) }.ok();
        }
    }
    tree(entries)
}

pub(crate) fn query_writer_empty(container: GUID) -> IWICMetadataQueryWriter {
    QueryWriterObj { tree: tree(Vec::new()), container, location: "/".into(), strict: false }.into()
}

pub(crate) fn query_writer_from_reader(reader: &IWICMetadataQueryReader) -> WResult<IWICMetadataQueryWriter> {
    let container = unsafe { reader.GetContainerFormat() }.unwrap_or_default();
    Ok(QueryWriterObj { tree: copy_reader_to_tree(reader), container, location: "/".into(), strict: false }.into())
}

/// Among the built-in codecs only GIF supports the fast metadata encoder; native's
/// CreateFastMetadataEncoderFrom* returns `WINCODEC_ERR_UNSUPPORTEDOPERATION` at create time for the
/// others (PNG/JPEG/TIFF/BMP/ICO/DDS — verified on the VM).
pub(crate) fn reject_fast_encoder(container: GUID) -> WResult<()> {
    if container.to_u128() == GIF {
        Ok(())
    } else {
        Err(e(WINCODEC_ERR_UNSUPPORTEDOPERATION))
    }
}

/// Which metadata blocks the fast encoder's `Commit` rewrites in place: the GIF logical screen
/// descriptor (decoder scope) or a frame's image descriptor (frame scope).
pub(crate) enum FastScope {
    Decoder,
    Frame(u32),
}

/// Seed the fast metadata encoder from a decoder/frame's metadata reader, retaining the writable
/// stream and raw bytes so `Commit` can rewrite the edited blocks in place.
pub(crate) fn fast_encoder(container: GUID, reader: Option<IWICMetadataQueryReader>, stream: Option<IStream>, raw: Rc<Vec<u8>>, scope: FastScope) -> IWICFastMetadataEncoder {
    let t = match reader {
        Some(r) => copy_reader_to_tree(&r),
        None => tree(Vec::new()),
    };
    FastEncoderObj { tree: t.clone(), writer: QueryWriterObj { tree: t, container, location: "/".into(), strict: true }.into(), container, stream, raw, scope }.into()
}

/// `IWICFastMetadataEncoder`: exposes a seeded query writer and, on `Commit`, re-serialises the
/// edited metadata back into the container stream in place (native only accepts GIF for this).
#[implement(IWICFastMetadataEncoder)]
struct FastEncoderObj {
    tree: Tree,
    writer: IWICMetadataQueryWriter,
    container: GUID,
    stream: Option<IStream>,
    raw: Rc<Vec<u8>>,
    scope: FastScope,
}

impl IWICFastMetadataEncoder_Impl for FastEncoderObj_Impl {
    fn Commit(&self) -> WResult<()> {
        let stream = self.stream.as_ref().ok_or(e(WINCODEC_ERR_NOTINITIALIZED))?;
        // Only GIF's fixed-layout blocks are re-serialised; for other containers nothing is dirty
        // yet, so leaving the stream untouched (output == input) matches native's no-op commit.
        if self.container.to_u128() != GIF {
            return Ok(());
        }
        // Same-size in-place overwrite (every GIF descriptor field is fixed-width), so the stream
        // length is unchanged — just rewrite from the start.
        let out = serialize_gif_inplace(&self.raw, &self.tree, &self.scope);
        unsafe {
            stream.Seek(0, STREAM_SEEK_SET, None)?;
            let mut w = 0u32;
            stream.Write(out.as_ptr() as *const _, out.len() as u32, Some(&mut w)).ok()?;
        }
        Ok(())
    }
    fn GetMetadataQueryWriter(&self) -> WResult<IWICMetadataQueryWriter> {
        Ok(self.writer.clone())
    }
}

/// The named child sub-tree of `root` (e.g. `/logscrdesc`), if present.
fn sub_of(root: &Tree, name: &str) -> Option<Tree> {
    root.borrow().iter().find(|(k, _)| k == name).and_then(|(_, v)| match v {
        Value::Sub(t) => Some(t.clone()),
        _ => None,
    })
}

/// A GIF descriptor field's numeric value (all are `U1`/`U2`/`Bool`; 0 if absent).
fn field_u(t: &Tree, name: &str) -> u64 {
    t.borrow().iter().find(|(k, _)| k == name).map(|(_, v)| match v {
        Value::U1(x) => *x as u64,
        Value::U2(x) => *x as u64,
        Value::Bool(b) => *b as u64,
        _ => 0,
    }).unwrap_or(0)
}

/// The byte offset of the `index`-th image descriptor (the `0x2C` separator) in a GIF.
fn gif_imgdesc_offset(raw: &[u8], index: u32) -> Option<usize> {
    if raw.len() < 13 {
        return None;
    }
    let packed = raw[10];
    let mut o = 13;
    if packed & 0x80 != 0 {
        o += 3 * (1 << ((packed & 7) + 1));
    }
    let mut frame = 0u32;
    while let Some(&b) = raw.get(o) {
        match b {
            0x3B => break,
            0x21 => o = gif_skip_subblocks(raw, o + 2),
            0x2C => {
                let p = raw.get(o + 9).copied().unwrap_or(0);
                if frame == index {
                    return Some(o);
                }
                o += 10;
                if p & 0x80 != 0 {
                    o += 3 * (1 << ((p & 7) + 1));
                }
                o = gif_skip_subblocks(raw, o + 1);
                frame += 1;
            }
            _ => break,
        }
    }
    None
}

/// Rewrite the edited GIF block (logical screen descriptor or image descriptor) back into a copy of
/// the raw bytes at its original offset — the exact in-place overwrite native's fast encoder does.
fn serialize_gif_inplace(raw: &[u8], root: &Tree, scope: &FastScope) -> Vec<u8> {
    let mut out = raw.to_vec();
    match scope {
        FastScope::Decoder => {
            if let Some(t) = sub_of(root, "/logscrdesc") {
                if out.len() >= 13 {
                    out[6..8].copy_from_slice(&(field_u(&t, "/Width") as u16).to_le_bytes());
                    out[8..10].copy_from_slice(&(field_u(&t, "/Height") as u16).to_le_bytes());
                    out[10] = ((field_u(&t, "/GlobalColorTableFlag") as u8) << 7)
                        | (((field_u(&t, "/ColorResolution") as u8) & 7) << 4)
                        | ((field_u(&t, "/SortFlag") as u8) << 3)
                        | (field_u(&t, "/GlobalColorTableSize") as u8 & 7);
                    out[11] = field_u(&t, "/BackgroundColorIndex") as u8;
                    out[12] = field_u(&t, "/PixelAspectRatio") as u8;
                }
            }
        }
        FastScope::Frame(n) => {
            if let (Some(t), Some(d)) = (sub_of(root, "/imgdesc"), gif_imgdesc_offset(raw, *n)) {
                if out.len() >= d + 10 {
                    out[d + 1..d + 3].copy_from_slice(&(field_u(&t, "/Left") as u16).to_le_bytes());
                    out[d + 3..d + 5].copy_from_slice(&(field_u(&t, "/Top") as u16).to_le_bytes());
                    out[d + 5..d + 7].copy_from_slice(&(field_u(&t, "/Width") as u16).to_le_bytes());
                    out[d + 7..d + 9].copy_from_slice(&(field_u(&t, "/Height") as u16).to_le_bytes());
                    out[d + 9] = ((field_u(&t, "/LocalColorTableFlag") as u8) << 7)
                        | ((field_u(&t, "/InterlaceFlag") as u8) << 6)
                        | ((field_u(&t, "/SortFlag") as u8) << 5)
                        | (field_u(&t, "/LocalColorTableSize") as u8 & 7);
                }
            }
        }
    }
    out
}

// --- encoder-side serialization ----------------------------------------------------------------

/// The tag of a `/{ushort=NNN}` IFD key, if that's its form.
fn parse_ushort_key(k: &str) -> Option<u16> {
    k.strip_prefix("/{ushort=")?.strip_suffix('}')?.parse().ok()
}

/// A metadata value as an IFD field: `(TIFF type, count, data bytes)`, big-endian (`le=false`, JPEG
/// APP1 EXIF) or little-endian (`le=true`, TIFF `II`). The PROPVARIANT VARTYPE picks the TIFF type,
/// exactly as native's IFD writer does (SHORT←UI2, LONG←UI4, …).
fn exif_encode_value(v: &Value, le: bool) -> Option<(u16, u32, Vec<u8>)> {
    let u16b = |x: u16| if le { x.to_le_bytes().to_vec() } else { x.to_be_bytes().to_vec() };
    let u32b = |x: u32| if le { x.to_le_bytes().to_vec() } else { x.to_be_bytes().to_vec() };
    // RATIONAL: the reader packs (den << 32) | num, so num is the low dword, den the high.
    let ratb = |x: u64| {
        let mut d = u32b(x as u32);
        d.extend(u32b((x >> 32) as u32));
        d
    };
    Some(match v {
        Value::U1(x) => (1, 1, vec![*x]),                                      // BYTE
        Value::U2(x) => (3, 1, u16b(*x)),                                      // SHORT
        Value::U4(x) => (4, 1, u32b(*x)),                                      // LONG
        Value::U8(x) => (5, 1, ratb(*x)),                                      // RATIONAL
        // An ASCII tag may be set as VT_LPSTR or VT_LPWSTR; native narrows the wide string to EXIF
        // ASCII either way (e.g. ImageDescription 0x010e written as `wstr`). ponytail: non-ASCII wide
        // chars narrow by UTF-8 bytes here, not native's code-page narrowing — fine for ASCII tags.
        Value::Str(s) | Value::WStr(s) => {
            let mut b: Vec<u8> = s.bytes().collect();
            b.push(0);
            (2, b.len() as u32, b) // ASCII (count includes NUL)
        }
        Value::Blob(b) => (7, b.len() as u32, b.clone()),                      // UNDEFINED
        Value::VecU2(v) => (3, v.len() as u32, v.iter().flat_map(|x| u16b(*x)).collect()), // SHORT[]
        Value::VecU4(v) => (4, v.len() as u32, v.iter().flat_map(|x| u32b(*x)).collect()), // LONG[]
        Value::VecU8(v) => (5, v.len() as u32, v.iter().flat_map(|x| ratb(*x)).collect()), // RATIONAL[]
        _ => return None,
    })
}

/// One IFD entry: a direct value, or a pointer to a nested sub-IFD block.
enum IfdEntry {
    Value(u16, u16, u32, Vec<u8>), // tag, TIFF type, count, big-endian data
    Sub(u16, Tree),                // pointer tag (Exif/GPS/Interop), child subtree
}

/// The entries of an IFD subtree, sorted ascending by tag: direct `/{ushort=TAG}` leaves plus
/// pointer entries for the nested `/exif` (34665) and `/gps` (34853) blocks.
fn collect_ifd(tree: &Tree) -> Vec<IfdEntry> {
    let mut e: Vec<IfdEntry> = Vec::new();
    for (k, v) in tree.borrow().iter() {
        if let Some(tag) = parse_ushort_key(k) {
            if let Some((ty, cnt, data)) = exif_encode_value(v, false) {
                e.push(IfdEntry::Value(tag, ty, cnt, data));
            }
        } else if let Value::Sub(child) = v {
            if let Some(tag) = match k.as_str() { "/exif" => Some(34665u16), "/gps" => Some(34853), _ => None } {
                e.push(IfdEntry::Sub(tag, child.clone()));
            }
        }
    }
    e.sort_by_key(|x| match x { IfdEntry::Value(t, ..) => *t, IfdEntry::Sub(t, _) => *t });
    e
}

/// Emit one IFD block — table, then its data area (values >4 bytes, each even-padded), then its
/// nested sub-IFD blocks (each padded to a multiple of 4) — with every offset relative to the TIFF
/// start, given this IFD's table begins at `base`. Native's exact layout.
fn emit_ifd(tree: &Tree, base: usize) -> Vec<u8> {
    let entries = collect_ifd(tree);
    let n = entries.len();
    let data_start = base + 2 + 12 * n + 4;
    // Data area for over-long values, assigning each its offset.
    let mut data = Vec::new();
    let value_off: Vec<Option<usize>> = entries.iter().map(|e| match e {
        IfdEntry::Value(_, _, _, d) if d.len() > 4 => {
            let off = data_start + data.len();
            data.extend_from_slice(d);
            if d.len() % 2 == 1 { data.push(0); }
            Some(off)
        }
        _ => None,
    }).collect();
    // Sub-IFD blocks after the data area; each padded to a multiple of 4.
    let mut subs = Vec::new();
    let mut cursor = data_start + data.len();
    let sub_off: Vec<Option<usize>> = entries.iter().map(|e| match e {
        IfdEntry::Sub(_, child) => {
            let off = cursor;
            let mut block = emit_ifd(child, off);
            while block.len() % 4 != 0 { block.push(0); }
            cursor += block.len();
            subs.extend_from_slice(&block);
            Some(off)
        }
        _ => None,
    }).collect();
    // Build the table with resolved offsets.
    let mut out = Vec::new();
    out.extend_from_slice(&(n as u16).to_be_bytes());
    for (i, e) in entries.iter().enumerate() {
        match e {
            IfdEntry::Value(tag, ty, cnt, d) => {
                out.extend_from_slice(&tag.to_be_bytes());
                out.extend_from_slice(&ty.to_be_bytes());
                out.extend_from_slice(&cnt.to_be_bytes());
                match value_off[i] {
                    Some(off) => out.extend_from_slice(&(off as u32).to_be_bytes()),
                    None => { let mut f = d.clone(); f.resize(4, 0); out.extend_from_slice(&f); }
                }
            }
            IfdEntry::Sub(tag, _) => {
                out.extend_from_slice(&tag.to_be_bytes());
                out.extend_from_slice(&4u16.to_be_bytes()); // LONG
                out.extend_from_slice(&1u32.to_be_bytes());
                out.extend_from_slice(&(sub_off[i].unwrap() as u32).to_be_bytes());
            }
        }
    }
    out.extend_from_slice(&0u32.to_be_bytes()); // next-IFD offset (none)
    out.extend_from_slice(&data);
    out.extend_from_slice(&subs);
    out
}

/// A big-endian TIFF stream (header + IFD0 and nested sub-IFDs) for the `/app1/ifd` subtree, or
/// empty if it has no serialisable entries.
fn build_tiff_ifd0(ifd0: &Tree) -> Vec<u8> {
    if collect_ifd(ifd0).is_empty() {
        return Vec::new();
    }
    let mut tiff = Vec::new();
    tiff.extend_from_slice(b"MM");
    tiff.extend_from_slice(&42u16.to_be_bytes());
    tiff.extend_from_slice(&8u32.to_be_bytes()); // IFD0 at offset 8
    tiff.extend_from_slice(&emit_ifd(ifd0, 8));
    tiff
}

/// Scalar XMP properties `(prefix, local, value)` set under `/xmp` on a query writer, in order.
/// The child keys are `/{prefix}:{local}` with VT_LPWSTR values; arrays/structs (nested) are skipped.
fn xmp_props_of(tree: &Tree) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    if let Some(xmp) = sub_of(tree, "/xmp") {
        for (k, v) in xmp.borrow().iter() {
            let val = match v {
                Value::WStr(s) | Value::Str(s) => s.clone(),
                _ => continue,
            };
            if let Some((pfx, local)) = k.trim_start_matches('/').split_once(':') {
                out.push((pfx.to_string(), local.to_string(), val));
            }
        }
    }
    out
}

pub(crate) fn xmp_props(writer: &IWICMetadataQueryReader) -> Vec<(String, String, String)> {
    xmp_props_of(&copy_reader_to_tree(writer))
}

/// Serialise a JPEG encoder frame's query-writer tree into the APPn/COM segments native inserts
/// after the APP0 JFIF marker: APP1 `Exif` (IFD0 from `/app1/ifd`), a `/com/TextEntry` COM segment,
/// and an APP1 XMP packet (`/xmp`). Returns the concatenated segment bytes (empty if none set).
pub(crate) fn jpeg_metadata_segments(writer: &IWICMetadataQueryReader) -> Vec<u8> {
    let tree = copy_reader_to_tree(writer);
    let mut out = Vec::new();
    if let Some(tiff) = sub_of(&tree, "/app1").and_then(|a| sub_of(&a, "/ifd")).map(|ifd| build_tiff_ifd0(&ifd)) {
        if !tiff.is_empty() {
            let seg_len = 2 + 6 + tiff.len(); // length field + "Exif\0\0" + TIFF
            out.extend_from_slice(&[0xFF, 0xE1]);
            out.extend_from_slice(&(seg_len as u16).to_be_bytes());
            out.extend_from_slice(b"Exif\0\0");
            out.extend_from_slice(&tiff);
        }
    }
    if let Some(com) = sub_of(&tree, "/com") {
        if let Some((_, Value::Str(s))) = com.borrow().iter().find(|(k, _)| k == "/TextEntry") {
            let mut payload: Vec<u8> = s.bytes().collect();
            payload.push(0);
            out.extend_from_slice(&[0xFF, 0xFE]);
            out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
            out.extend_from_slice(&payload);
        }
    }
    if let Some(packet) = xmp_packet(&xmp_props_of(&tree)) {
        const ID: &[u8] = b"http://ns.adobe.com/xap/1.0/\0"; // 29 bytes
        out.extend_from_slice(&[0xFF, 0xE1]);
        out.extend_from_slice(&((2 + ID.len() + packet.len()) as u16).to_be_bytes());
        out.extend_from_slice(ID);
        out.extend_from_slice(&packet);
    }
    out
}

/// The `/ifd` subtree as little-endian [`TiffField`]s (`/exif`/`/gps` become sub-IFD pointers), for
/// merging into the TIFF encoder's directory via `encode_tiff_meta`.
fn tree_to_tiff_fields(t: &Tree) -> Vec<TiffField> {
    let mut out = Vec::new();
    for (k, v) in t.borrow().iter() {
        if let Some(tag) = parse_ushort_key(k) {
            if let Some((ty, count, data)) = exif_encode_value(v, true) {
                out.push(TiffField::Value { tag, ty, count, data });
            }
        } else if let Value::Sub(child) = v {
            if let Some(tag) = match k.as_str() { "/exif" => Some(34665u16), "/gps" => Some(34853), _ => None } {
                out.push(TiffField::Sub { tag, children: tree_to_tiff_fields(child) });
            }
        }
    }
    out
}

/// The TIFF IFD0 fields set through a frame encoder's metadata query writer (the `/ifd` subtree).
/// Empty if none were set.
pub(crate) fn tiff_ifd_fields(writer: &IWICMetadataQueryReader) -> Vec<TiffField> {
    let tree = copy_reader_to_tree(writer);
    sub_of(&tree, "/ifd").map(|ifd| tree_to_tiff_fields(&ifd)).unwrap_or_default()
}

/// The GIF comment/application/graphic-control extensions and image-descriptor overrides set on a
/// frame encoder's query writer, for `encode_gif_meta`.
pub(crate) fn gif_meta(writer: &IWICMetadataQueryReader) -> GifMeta {
    let tree = copy_reader_to_tree(writer);
    let mut m = GifMeta::default();
    if let Some(c) = sub_of(&tree, "/commentext") {
        if let Some((_, Value::Str(s))) = c.borrow().iter().find(|(k, _)| k == "/TextEntry") {
            m.comment = Some(s.bytes().collect());
        }
    }
    if let Some(g) = sub_of(&tree, "/grctlext") {
        m.gce = Some(GifGce {
            packed: ((field_u(&g, "/Disposal") as u8 & 7) << 2)
                | ((field_u(&g, "/UserInputFlag") as u8 & 1) << 1)
                | (field_u(&g, "/TransparencyFlag") as u8 & 1),
            delay: field_u(&g, "/Delay") as u16,
            trans_index: field_u(&g, "/TransparentColorIndex") as u8,
        });
    }
    if let Some(a) = sub_of(&tree, "/appext") {
        let bytes = |name: &str| a.borrow().iter().find(|(k, _)| k == name).and_then(|(_, v)| match v {
            Value::VecU1(b) => Some(b.clone()),
            _ => None,
        });
        if let (Some(id), Some(data)) = (bytes("/Application"), bytes("/Data")) {
            m.app_id = Some(id);
            m.app_data = data;
        }
    }
    if let Some(d) = sub_of(&tree, "/imgdesc") {
        let field = |name: &str| d.borrow().iter().find(|(k, _)| k == name).map(|_| field_u(&d, name) as u16);
        (m.left, m.top, m.width, m.height) = (field("/Left"), field("/Top"), field("/Width"), field("/Height"));
    }
    m
}

/// Serialise a PNG encoder frame's query-writer tree into the ancillary chunks native inserts
/// between the `gAMA` chunk and the first `IDAT`: a `tEXt` per `/tEXt/{str=KEYWORD}` entry and a
/// `tIME` from `/tIME`. Returns the concatenated chunk bytes (empty if none set).
pub(crate) fn png_metadata_chunks(writer: &IWICMetadataQueryReader) -> Vec<u8> {
    let tree = copy_reader_to_tree(writer);
    let mut out = Vec::new();
    if let Some(t) = sub_of(&tree, "/tEXt") {
        for (k, v) in t.borrow().iter() {
            if let (Some(kw), Value::Str(text)) = (k.strip_prefix("/{str=").and_then(|s| s.strip_suffix('}')), v) {
                let mut data: Vec<u8> = kw.bytes().collect();
                data.push(0);
                data.extend(text.bytes());
                out.extend(d2d::wic::png_chunk(b"tEXt", &data));
            }
        }
    }
    if let Some(t) = sub_of(&tree, "/tIME") {
        let mut data = (field_u(&t, "/Year") as u16).to_be_bytes().to_vec();
        data.extend([field_u(&t, "/Month") as u8, field_u(&t, "/Day") as u8, field_u(&t, "/Hour") as u8, field_u(&t, "/Minute") as u8, field_u(&t, "/Second") as u8]);
        out.extend(d2d::wic::png_chunk(b"tIME", &data));
    }
    out
}

// --- PNG ---------------------------------------------------------------------------------------

/// A latin-1 (byte-per-char) string up to the first NUL.
fn latin1_z(b: &[u8]) -> String {
    b.iter().take_while(|&&c| c != 0).map(|&c| c as char).collect()
}

fn parse_png(raw: &[u8]) -> Vec<(String, Value)> {
    let mut out: Vec<(String, Value)> = Vec::new();
    let mut counts: HashMap<&'static str, u32> = HashMap::new();
    let push = |out: &mut Vec<(String, Value)>, counts: &mut HashMap<&'static str, u32>, base: &'static str, v: Value| {
        let n = counts.entry(base).or_insert(0);
        let name = if *n == 0 { format!("/{base}") } else { format!("/[{n}]{base}") };
        *n += 1;
        out.push((name, v));
    };
    let mut o = 8;
    while o + 8 <= raw.len() {
        let len = rd32(raw, o, false) as usize;
        let typ = &raw[o + 4..o + 8];
        let data = raw.get(o + 8..o + 8 + len).unwrap_or(&[]);
        match typ {
            b"tEXt" => {
                let z = data.iter().position(|&c| c == 0).unwrap_or(data.len());
                let kw = latin1_z(&data[..z]);
                let text = latin1_z(data.get(z + 1..).unwrap_or(&[]));
                push(&mut out, &mut counts, "tEXt", Value::Sub(tree(vec![(format!("/{{str={kw}}}"), Value::Str(text))])));
            }
            b"gAMA" => {
                push(&mut out, &mut counts, "gAMA", Value::Sub(tree(vec![("/ImageGamma".into(), Value::U4(rd32(data, 0, false)))])));
            }
            b"sRGB" if !data.is_empty() => {
                push(&mut out, &mut counts, "sRGB", Value::Sub(tree(vec![("/RenderingIntent".into(), Value::U1(data[0]))])));
            }
            b"tIME" if data.len() >= 7 => {
                push(&mut out, &mut counts, "tIME", Value::Sub(tree(vec![
                    ("/Year".into(), Value::U2(rd16(data, 0, false))),
                    ("/Month".into(), Value::U1(data[2])),
                    ("/Day".into(), Value::U1(data[3])),
                    ("/Hour".into(), Value::U1(data[4])),
                    ("/Minute".into(), Value::U1(data[5])),
                    ("/Second".into(), Value::U1(data[6])),
                ])));
            }
            b"iTXt" => {
                let kz = data.iter().position(|&c| c == 0).unwrap_or(0);
                if &data[..kz] == b"XML:com.adobe.xmp" {
                    // XMP iTXt: skip keyword\0, compression flag + method, language tag\0 and
                    // translated keyword\0; the remainder is the (uncompressed) XMP packet.
                    let rest = data.get(kz + 3..).unwrap_or(&[]);
                    let lz = rest.iter().position(|&c| c == 0).map_or(rest.len(), |i| i + 1);
                    let rest2 = rest.get(lz..).unwrap_or(&[]);
                    let tz = rest2.iter().position(|&c| c == 0).map_or(rest2.len(), |i| i + 1);
                    push(&mut out, &mut counts, "xmp", Value::Sub(tree(parse_xmp(rest2.get(tz..).unwrap_or(&[])))));
                } else {
                    push(&mut out, &mut counts, "iTXt", Value::Sub(tree(parse_itxt(data))));
                }
            }
            b"IEND" => break,
            _ => {}
        }
        o += 12 + len;
    }
    out
}

fn parse_itxt(data: &[u8]) -> Vec<(String, Value)> {
    let kz = data.iter().position(|&c| c == 0).unwrap_or(0);
    let keyword = latin1_z(&data[..kz]);
    let compflag = data.get(kz + 1).copied().unwrap_or(0);
    // kz+2 is the compression method (not exposed).
    let rest = data.get(kz + 3..).unwrap_or(&[]);
    let lz = rest.iter().position(|&c| c == 0).unwrap_or(rest.len());
    let langtag = latin1_z(&rest[..lz]);
    let rest2 = rest.get(lz + 1..).unwrap_or(&[]);
    let tz = rest2.iter().position(|&c| c == 0).unwrap_or(rest2.len());
    let transkw = String::from_utf8_lossy(&rest2[..tz]).into_owned();
    let text = String::from_utf8_lossy(rest2.get(tz + 1..).unwrap_or(&[])).into_owned();
    vec![
        ("/CompressionFlag".into(), Value::U1(compflag)),
        ("/Keyword".into(), Value::Str(keyword)),
        ("/LanguageTag".into(), Value::Str(langtag)),
        ("/TextEntry".into(), Value::WStr(text)),
        ("/TranslatedKeyword".into(), Value::WStr(transkw)),
        ("/xmp".into(), Value::Sub(tree(Vec::new()))),
    ]
}

// --- XMP (RDF/XML) reader ----------------------------------------------------------------------
//
// Parses an XMP packet into the exact tree native exposes under `/xmp` (JPEG APP1 / PNG iTXt) or
// `/ifd/{ushort=700}` (TIFF): scalar properties as VT_LPWSTR `/{prefix}:{local}`, rdf:Bag/Seq as a
// sub-reader indexed `/{ulong=N}`, rdf:Alt as a sub-reader keyed by `xml:lang`, and structs
// (parseType=Resource or a nested rdf:Description) as a sub-reader of `/{prefix}:{local}` fields.
// Non-standard document prefixes are remapped to the registered prefix (native does this).

const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
const XML_NS: &str = "http://www.w3.org/XML/1998/namespace";

/// Registered XMP namespace URI → prefix (the prefix native exposes regardless of the document's).
const XMP_NS: &[(&str, &str)] = &[
    ("http://purl.org/dc/elements/1.1/", "dc"),
    ("http://ns.adobe.com/xap/1.0/", "xmp"),
    ("http://ns.adobe.com/xap/1.0/mm/", "xmpMM"),
    ("http://ns.adobe.com/xap/1.0/rights/", "xmpRights"),
    ("http://ns.adobe.com/tiff/1.0/", "tiff"),
    ("http://ns.adobe.com/exif/1.0/", "exif"),
    ("http://ns.adobe.com/exif/1.0/aux/", "aux"),
    ("http://ns.adobe.com/photoshop/1.0/", "photoshop"),
    ("http://ns.adobe.com/pdf/1.3/", "pdf"),
    ("http://ns.adobe.com/xap/1.0/sType/ResourceRef#", "stRef"),
    ("http://ns.adobe.com/xap/1.0/sType/ResourceEvent#", "stEvt"),
    ("http://ns.adobe.com/xap/1.0/sType/Dimensions#", "stDim"),
    ("http://ns.adobe.com/xap/1.0/sType/Version#", "stVer"),
];

/// The path prefix for `uri`: its registered prefix, else the document's own (for unlisted URIs).
fn xmp_prefix(node: roxmltree::Node, uri: &str) -> String {
    XMP_NS
        .iter()
        .find(|(u, _)| *u == uri)
        .map(|(_, p)| (*p).to_string())
        .or_else(|| node.lookup_prefix(uri).map(str::to_string))
        .unwrap_or_default()
}

/// Struct fields (non-rdf attributes + child elements) as a `/{prefix}:{local}` sub-tree.
fn xmp_struct(node: roxmltree::Node) -> Value {
    let mut fields: Vec<(String, Value)> = Vec::new();
    for a in node.attributes() {
        let ns = a.namespace().unwrap_or("");
        if ns.is_empty() || ns == RDF_NS || ns == XML_NS {
            continue;
        }
        fields.push((format!("/{}:{}", xmp_prefix(node, ns), a.name()), Value::WStr(a.value().to_string())));
    }
    for f in node.children().filter(roxmltree::Node::is_element) {
        let ns = f.tag_name().namespace().unwrap_or("");
        fields.push((format!("/{}:{}", xmp_prefix(f, ns), f.tag_name().name()), xmp_node_value(f)));
    }
    Value::Sub(tree(fields))
}

/// The value of a property/li node: scalar text, an array (Bag/Seq → `{ulong=N}`, Alt → by lang),
/// or a struct.
fn xmp_node_value(node: roxmltree::Node) -> Value {
    let elems: Vec<_> = node.children().filter(roxmltree::Node::is_element).collect();
    if elems.len() == 1 && elems[0].tag_name().namespace() == Some(RDF_NS) {
        let c = elems[0];
        let items = || c.children().filter(roxmltree::Node::is_element);
        match c.tag_name().name() {
            "Bag" | "Seq" => {
                return Value::Sub(tree(items().enumerate().map(|(i, li)| (format!("/{{ulong={i}}}"), xmp_node_value(li))).collect()));
            }
            "Alt" => {
                return Value::Sub(tree(items().enumerate().map(|(i, li)| {
                    let key = li.attribute((XML_NS, "lang")).map(|l| format!("/{l}")).unwrap_or_else(|| format!("/{{ulong={i}}}"));
                    (key, xmp_node_value(li))
                }).collect()));
            }
            "Description" => return xmp_struct(c),
            _ => {}
        }
    }
    if node.attribute((RDF_NS, "parseType")) == Some("Resource") {
        return xmp_struct(node);
    }
    Value::WStr(node.text().unwrap_or("").to_string())
}

/// Parse an XMP packet into the `/xmp` child entries. Empty on malformed input.
fn parse_xmp(packet: &[u8]) -> Vec<(String, Value)> {
    let Ok(s) = std::str::from_utf8(packet) else { return Vec::new() };
    // Slice the well-formed root out of the xpacket/padding wrapper.
    let (start, close) = match s.find("<x:xmpmeta") {
        Some(i) => (i, "</x:xmpmeta>"),
        None => match s.find("<rdf:RDF") {
            Some(i) => (i, "</rdf:RDF>"),
            None => return Vec::new(),
        },
    };
    let Some(rel) = s[start..].find(close) else { return Vec::new() };
    let Ok(doc) = roxmltree::Document::parse(&s[start..start + rel + close.len()]) else { return Vec::new() };
    let Some(rdf) = doc.descendants().find(|n| n.has_tag_name((RDF_NS, "RDF"))) else { return Vec::new() };
    let mut out: Vec<(String, Value)> = Vec::new();
    for desc in rdf.children().filter(|n| n.has_tag_name((RDF_NS, "Description"))) {
        for a in desc.attributes() {
            let ns = a.namespace().unwrap_or("");
            if ns.is_empty() || ns == RDF_NS || ns == XML_NS {
                continue;
            }
            out.push((format!("/{}:{}", xmp_prefix(desc, ns), a.name()), Value::WStr(a.value().to_string())));
        }
        for p in desc.children().filter(roxmltree::Node::is_element) {
            let ns = p.tag_name().namespace().unwrap_or("");
            out.push((format!("/{}:{}", xmp_prefix(p, ns), p.tag_name().name()), xmp_node_value(p)));
        }
    }
    out
}

// --- JPEG --------------------------------------------------------------------------------------

fn parse_jpeg_frame(raw: &[u8]) -> Vec<(String, Value)> {
    let mut out: Vec<(String, Value)> = Vec::new();
    let (mut lum, mut chrom): (Option<Vec<u16>>, Option<Vec<u16>>) = (None, None);
    let mut o = 2;
    while o + 1 < raw.len() && raw[o] == 0xFF {
        let mut m = o + 1;
        while m < raw.len() && raw[m] == 0xFF {
            m += 1;
        }
        let marker = raw[m];
        o = m + 1;
        if marker == 0xD9 || marker == 0xDA {
            break; // EOI / start of scan
        }
        if marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            continue; // standalone markers carry no length
        }
        let seglen = rd16(raw, o, false) as usize;
        if seglen < 2 || o + seglen > raw.len() {
            break;
        }
        let payload = &raw[o + 2..o + seglen];
        match marker {
            0xE0 if payload.starts_with(b"JFIF\0") => out.push(("/app0".into(), Value::Sub(tree(parse_jfif(&payload[5..]))))),
            0xE1 if payload.starts_with(b"Exif\0\0") => out.push(("/app1".into(), Value::Sub(tree(parse_exif(&payload[6..]))))),
            0xE1 if payload.starts_with(b"http://ns.adobe.com/xap/1.0/\0") => {
                out.push(("/xmp".into(), Value::Sub(tree(parse_xmp(&payload[29..])))));
            }
            0xDB => parse_dqt(payload, &mut lum, &mut chrom),
            _ => {}
        }
        o += seglen;
    }
    if let Some(v) = chrom {
        out.push(("/chrominance".into(), Value::Sub(tree(vec![("/TableEntry".into(), Value::VecU2(v))]))));
    }
    if let Some(v) = lum {
        out.push(("/luminance".into(), Value::Sub(tree(vec![("/TableEntry".into(), Value::VecU2(v))]))));
    }
    out
}

/// JFIF APP0 body (after `JFIF\0`): version, units, X/Y density, X/Y thumbnail, thumbnail blob.
fn parse_jfif(b: &[u8]) -> Vec<(String, Value)> {
    vec![
        ("/{ushort=0}".into(), Value::U2(rd16(b, 0, false))),
        ("/{ushort=1}".into(), Value::U1(b.get(2).copied().unwrap_or(0))),
        ("/{ushort=2}".into(), Value::U2(rd16(b, 3, false))),
        ("/{ushort=3}".into(), Value::U2(rd16(b, 5, false))),
        ("/{ushort=4}".into(), Value::U1(b.get(7).copied().unwrap_or(0))),
        ("/{ushort=5}".into(), Value::U1(b.get(8).copied().unwrap_or(0))),
        ("/{ushort=6}".into(), Value::Blob(b.get(9..).unwrap_or(&[]).to_vec())),
    ]
}

fn parse_dqt(payload: &[u8], lum: &mut Option<Vec<u16>>, chrom: &mut Option<Vec<u16>>) {
    let mut p = 0;
    while p < payload.len() {
        let pq = payload[p];
        p += 1;
        let (prec, id) = (pq >> 4, pq & 0xf);
        let mut vals = Vec::with_capacity(64);
        for i in 0..64 {
            vals.push(if prec == 0 { payload.get(p + i).copied().unwrap_or(0) as u16 } else { rd16(payload, p + i * 2, false) });
        }
        p += if prec == 0 { 64 } else { 128 };
        match id {
            0 => *lum = Some(vals),
            1 => *chrom = Some(vals),
            _ => {}
        }
    }
}

// --- EXIF / TIFF IFD ---------------------------------------------------------------------------

fn type_size(t: u16) -> usize {
    match t {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 => 4,
        5 | 10 | 12 => 8,
        _ => 1,
    }
}

fn pack_rational(d: &[u8], le: bool) -> u64 {
    ((rd32(d, 4, le) as u64) << 32) | rd32(d, 0, le) as u64
}

/// EXIF `UserComment`: an 8-byte character-code id then the comment (ASCII or UNICODE/UTF-16LE).
fn decode_user_comment(d: &[u8]) -> String {
    if d.len() < 8 {
        return String::new();
    }
    let (id, rest) = d.split_at(8);
    if id == b"UNICODE\0" {
        String::from_utf16_lossy(&rest.chunks_exact(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect::<Vec<_>>())
    } else {
        rest.iter().map(|&c| c as char).collect()
    }
}

/// Parse one IFD at `off`; return its `/{ushort=TAG}` children and the next-IFD offset.
fn parse_ifd(t: &[u8], off: u32, le: bool, depth: u8) -> (Vec<(String, Value)>, u32) {
    let mut out = Vec::new();
    let o = off as usize;
    if depth > 4 || o + 2 > t.len() {
        return (out, 0);
    }
    let n = rd16(t, o, le) as usize;
    for i in 0..n {
        let e = o + 2 + i * 12;
        if e + 12 > t.len() {
            break;
        }
        let (tag, typ, cnt) = (rd16(t, e, le), rd16(t, e + 2, le), rd32(t, e + 4, le) as usize);
        let total = type_size(typ) * cnt;
        let dpos = if total <= 4 { e + 8 } else { rd32(t, e + 8, le) as usize };
        let data = t.get(dpos..dpos + total).unwrap_or(&[]);
        if let Some(v) = ifd_value(t, tag, typ, cnt, data, le, depth) {
            out.push((format!("/{{ushort={tag}}}"), v));
        }
    }
    (out, rd32(t, o + 2 + n * 12, le))
}

fn ifd_value(t: &[u8], tag: u16, typ: u16, cnt: usize, data: &[u8], le: bool, depth: u8) -> Option<Value> {
    // Sub-IFD pointers (Exif, GPS, Interop) expand to nested readers.
    if matches!(tag, 34665 | 34853 | 40965) && typ == 4 {
        let (children, _) = parse_ifd(t, rd32(data, 0, le), le, depth + 1);
        return Some(Value::Sub(tree(children)));
    }
    // Tag 700 (XMP) is exposed as a nested XMP metadata reader, not a raw byte blob.
    if tag == 700 {
        return Some(Value::Sub(tree(parse_xmp(data))));
    }
    if tag == 37510 && typ == 7 {
        return Some(Value::WStr(decode_user_comment(data)));
    }
    Some(match typ {
        1 | 6 if cnt == 1 => Value::U1(data.first().copied().unwrap_or(0)),
        1 | 6 => Value::VecU1(data.to_vec()),
        2 => Value::Str(latin1_z(data)),
        3 | 8 if cnt == 1 => Value::U2(rd16(data, 0, le)),
        3 | 8 => Value::VecU2((0..cnt).map(|i| rd16(data, i * 2, le)).collect()),
        4 | 9 if cnt == 1 => Value::U4(rd32(data, 0, le)),
        4 | 9 => Value::VecU4((0..cnt).map(|i| rd32(data, i * 4, le)).collect()),
        5 | 10 if cnt == 1 => Value::U8(pack_rational(data, le)),
        5 | 10 => Value::VecU8((0..cnt).map(|i| pack_rational(&data[i * 8..], le)).collect()),
        7 => Value::Blob(data.to_vec()),
        _ => return None,
    })
}

/// EXIF APP1 TIFF (from the `II`/`MM` header): one `/{ushort=N}` entry per IFD.
fn parse_exif(t: &[u8]) -> Vec<(String, Value)> {
    let le = t.get(0..2) != Some(b"MM");
    let mut out = Vec::new();
    let mut off = rd32(t, 4, le);
    let (mut idx, mut guard) = (0u32, 0);
    while off != 0 && guard < 16 {
        let (children, next) = parse_ifd(t, off, le, 0);
        out.push((format!("/{{ushort={idx}}}"), Value::Sub(tree(children))));
        off = next;
        idx += 1;
        guard += 1;
    }
    out
}

fn parse_tiff_frame(raw: &[u8], index: u32) -> Vec<(String, Value)> {
    let le = raw.get(0..2) != Some(b"MM");
    let mut off = rd32(raw, 4, le);
    let mut i = 0;
    while off != 0 && i < index {
        off = parse_ifd(raw, off, le, 0).1;
        i += 1;
    }
    if off == 0 {
        return Vec::new();
    }
    let (children, _) = parse_ifd(raw, off, le, 0);
    vec![("/ifd".into(), Value::Sub(tree(children)))]
}

// --- GIF ---------------------------------------------------------------------------------------

fn parse_gif_logscr(raw: &[u8]) -> Vec<(String, Value)> {
    if raw.len() < 13 {
        return Vec::new();
    }
    let packed = raw[10];
    let desc = vec![
        ("/Signature".into(), Value::VecU1(raw[0..6].to_vec())),
        ("/Width".into(), Value::U2(rd16(raw, 6, true))),
        ("/Height".into(), Value::U2(rd16(raw, 8, true))),
        ("/GlobalColorTableFlag".into(), Value::Bool(packed & 0x80 != 0)),
        ("/ColorResolution".into(), Value::U1((packed >> 4) & 7)),
        ("/SortFlag".into(), Value::Bool(packed & 0x08 != 0)),
        ("/GlobalColorTableSize".into(), Value::U1(packed & 7)),
        ("/BackgroundColorIndex".into(), Value::U1(raw[11])),
        ("/PixelAspectRatio".into(), Value::U1(raw[12])),
    ];
    vec![("/logscrdesc".into(), Value::Sub(tree(desc)))]
}

fn gif_skip_subblocks(raw: &[u8], mut o: usize) -> usize {
    while let Some(&len) = raw.get(o) {
        if len == 0 {
            return o + 1;
        }
        o += 1 + len as usize;
    }
    o
}

fn parse_gif_frame(raw: &[u8], index: u32) -> Vec<(String, Value)> {
    if raw.len() < 13 {
        return Vec::new();
    }
    let packed = raw[10];
    let mut o = 13;
    if packed & 0x80 != 0 {
        o += 3 * (1 << ((packed & 7) + 1));
    }
    let mut frame = 0u32;
    while let Some(&b) = raw.get(o) {
        match b {
            0x3B => break,
            0x21 => o = gif_skip_subblocks(raw, o + 2),
            0x2C => {
                let p = raw.get(o + 9).copied().unwrap_or(0);
                if frame == index {
                    return vec![("/imgdesc".into(), Value::Sub(tree(vec![
                        ("/Left".into(), Value::U2(rd16(raw, o + 1, true))),
                        ("/Top".into(), Value::U2(rd16(raw, o + 3, true))),
                        ("/Width".into(), Value::U2(rd16(raw, o + 5, true))),
                        ("/Height".into(), Value::U2(rd16(raw, o + 7, true))),
                        ("/LocalColorTableFlag".into(), Value::Bool(p & 0x80 != 0)),
                        ("/InterlaceFlag".into(), Value::Bool(p & 0x40 != 0)),
                        ("/SortFlag".into(), Value::Bool(p & 0x20 != 0)),
                        ("/LocalColorTableSize".into(), Value::U1(p & 7)),
                    ])))];
                }
                o += 10;
                if p & 0x80 != 0 {
                    o += 3 * (1 << ((p & 7) + 1));
                }
                o = gif_skip_subblocks(raw, o + 1); // skip LZW min-code-size + image data
                frame += 1;
            }
            _ => break,
        }
    }
    Vec::new()
}
