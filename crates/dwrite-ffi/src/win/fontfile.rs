//! Font-file references, the built-in local file loader + stream, and font-face references — the
//! `CreateFontFileReference` / `CreateCustomFontFileReference` / `CreateFontFace` /
//! `CreateFontFaceReference*` surface. A font face built here is the same [`FaceObj`] the system
//! collection produces (glyph indices, metrics and `CreateGlyphRunAnalysis` all go through it), so an
//! app that loads its own font renders byte-identically to a system font.

use super::fonts::FaceObj;
use super::{E_INVALIDARG, wide_to_string};
use core::cell::RefCell;
use core::slice;
use flate2::read::ZlibDecoder;
use skrifa::string::StringId;
use skrifa::{FontRef, MetadataProvider};
use std::io::Read;
use std::sync::Arc;
use windows::Win32::Foundation::{FILETIME, FALSE, TRUE};
use windows::Win32::Graphics::DirectWrite::*;
use windows_core::{BOOL, Interface, PCWSTR, PWSTR, Ref, Result as WResult, implement};

/// The canonical (English-or-first) family name of a face, for `FaceObj`/`FontObj` identity.
pub(crate) fn family_name(bytes: &[u8], index: u32) -> String {
    FontRef::from_index(bytes, index)
        .ok()
        .and_then(|f| f.localized_strings(StringId::FAMILY_NAME).english_or_first().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// `IDWriteFontFile::Analyze` by parsing with skrifa: supported ⇔ it parses; the type is a collection
/// when the wrapper is `ttcf` (or >1 face), CFF when a `CFF ` table is present, else plain TrueType.
fn analyze(bytes: &[u8]) -> (bool, DWRITE_FONT_FILE_TYPE, DWRITE_FONT_FACE_TYPE, u32) {
    let mut faces = 0u32;
    let mut has_cff = false;
    while let Ok(font) = FontRef::from_index(bytes, faces) {
        if faces == 0 {
            has_cff = font.table_data(skrifa::Tag::new(b"CFF ")).is_some();
        }
        faces += 1;
    }
    if faces == 0 {
        return (false, DWRITE_FONT_FILE_TYPE_UNKNOWN, DWRITE_FONT_FACE_TYPE_UNKNOWN, 0);
    }
    let collection = bytes.starts_with(b"ttcf");
    let (file, face) = if collection {
        (DWRITE_FONT_FILE_TYPE_TRUETYPE_COLLECTION, DWRITE_FONT_FACE_TYPE_OPENTYPE_COLLECTION)
    } else if has_cff {
        (DWRITE_FONT_FILE_TYPE_CFF, DWRITE_FONT_FACE_TYPE_CFF)
    } else {
        (DWRITE_FONT_FILE_TYPE_TRUETYPE, DWRITE_FONT_FACE_TYPE_TRUETYPE)
    };
    (true, file, face, faces)
}

// --- IDWriteFontFile ---------------------------------------------------------------------------

#[implement(IDWriteFontFile)]
pub(crate) struct FontFileObj {
    pub(crate) bytes: Arc<Vec<u8>>,
    /// The reference key: a local file's path as UTF-16 code units (little-endian bytes), so the
    /// local loader can round-trip it back to a path. Empty for a custom/in-memory file.
    key: Vec<u8>,
    /// `true` ⇒ the key is a file path and [`GetLoader`](FontFileObj::GetLoader) hands back the
    /// local file loader (native's file-backed form); `false` ⇒ an in-memory loader that serves
    /// `bytes` directly, so an in-memory face still round-trips GetReferenceKey→CreateStreamFromKey.
    local: bool,
    /// `true` ⇒ an un-downloaded HTTP/remote reference (`CreateFontFileReferenceFromUrl`): the key is
    /// the URL, `bytes` is empty, and [`GetLoader`](FontFileObj::GetLoader) hands back the remote loader.
    remote: bool,
}

/// An un-downloaded HTTP font reference: the key is the resolved URL as UTF-16LE (native's key is the
/// URL too — 60 bytes for a 30-char URL), no bytes yet.
pub(crate) fn file_remote(key: Vec<u8>) -> IDWriteFontFile {
    FontFileObj { bytes: Arc::new(Vec::new()), key, local: false, remote: true }.into()
}

/// A font file over already-loaded bytes (an in-memory face): the loader serves the bytes, the key
/// is empty — what `IDWriteFontFace::GetFiles` hands back for a memory-backed face.
pub(crate) fn file_from_bytes(bytes: Arc<Vec<u8>>) -> IDWriteFontFile {
    FontFileObj { bytes, key: Vec::new(), local: false, remote: false }.into()
}

impl FontFileObj {
    /// The [`FontSource`](super::fonts::FontSource) a face built from this file should carry, so its
    /// `GetFiles` reproduces this file's form: the on-disk path (decoded from the key) for a local
    /// file, else the in-memory bytes.
    pub(crate) fn as_source(&self) -> super::fonts::FontSource {
        if self.local {
            super::fonts::FontSource::Path(decode_key(&self.key).1)
        } else {
            super::fonts::FontSource::Bytes(self.bytes.clone())
        }
    }
}

/// The local font file reference key. For a file in the Windows fonts directory this is native's
/// system-loader form — the file's last-write time (FILETIME, 8 bytes LE), a constant `0x002a` tag,
/// then the UPPERCASE filename as UTF-16LE + NUL (byte-exact: Arial/Times/Calibri/Comic on the VM).
/// A font elsewhere (an app `CreateFontFileReference`) keeps the full UTF-16LE path, so it round-trips
/// without the fonts-dir assumption; the two forms are told apart by the `0x2a` tag (a real path's
/// 5th UTF-16 unit is never `*`).
pub(crate) fn local_key(path: &std::path::Path) -> Vec<u8> {
    let in_fonts_dir = path.parent().is_some_and(|p| p.to_string_lossy().eq_ignore_ascii_case(&super::fonts::fonts_dir().to_string_lossy()));
    if in_fonts_dir {
        let mut k = path_write_time(&path.to_string_lossy()).to_le_bytes().to_vec();
        k.extend_from_slice(&[0x2a, 0x00]);
        let name = path.file_name().map(|n| n.to_string_lossy().to_ascii_uppercase()).unwrap_or_default();
        k.extend(name.encode_utf16().chain(core::iter::once(0)).flat_map(|u| u.to_le_bytes()));
        k
    } else {
        path.to_string_lossy().encode_utf16().flat_map(|u| u.to_le_bytes()).collect()
    }
}

/// `(last-write time, on-disk path)` from a [`local_key`]: the compact system form (FILETIME + 0x2a +
/// filename resolved under the fonts dir) or a full UTF-16LE path.
fn decode_key(key: &[u8]) -> (u64, std::path::PathBuf) {
    if key.len() >= 12 && key[8] == 0x2a && key[9] == 0 {
        let wt = u64::from_le_bytes(key[..8].try_into().unwrap());
        let name: Vec<u16> = key[10..].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).take_while(|&u| u != 0).collect();
        (wt, super::fonts::fonts_dir().join(String::from_utf16_lossy(&name)))
    } else {
        let units: Vec<u16> = key.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).take_while(|&u| u != 0).collect();
        let p = std::path::PathBuf::from(String::from_utf16_lossy(&units));
        (path_write_time(&p.to_string_lossy()), p)
    }
}

/// A file-backed reference over already-loaded `bytes` keyed by `path` (no re-read): the local loader
/// resolves the key back to the font on disk. The file-backed form `IDWriteFontFace::GetFiles` returns.
pub(crate) fn file_ref_path(bytes: Arc<Vec<u8>>, path: &std::path::Path) -> IDWriteFontFile {
    FontFileObj { bytes, key: local_key(path), local: true, remote: false }.into()
}

/// `CreateFontFileReference(path)`: read the file now and key it by its path. ponytail: uses the
/// system-font key form (filename under the fonts dir) — exact for a fonts-dir file, which is all the
/// corpus references; a font elsewhere would need its directory kept too.
pub(crate) fn file_from_path(path: &str) -> WResult<IDWriteFontFile> {
    let bytes = std::fs::read(path).map_err(|_| windows_core::Error::from(E_INVALIDARG))?;
    Ok(FontFileObj { bytes: Arc::new(bytes), key: local_key(std::path::Path::new(path)), local: true, remote: false }.into())
}

/// `CreateCustomFontFileReference(key, loader)`: read the bytes through the app loader now (so the
/// face never needs the loader again), keeping the opaque key for `GetReferenceKey`.
pub(crate) fn file_from_loader(key: &[u8], loader: &IDWriteFontFileLoader) -> WResult<IDWriteFontFile> {
    let bytes = read_via_loader(loader, key)?;
    // The app's loader was consumed at creation (bytes read once), so the file is now in-memory.
    Ok(FontFileObj { bytes: Arc::new(bytes), key: key.to_vec(), local: false, remote: false }.into())
}

impl IDWriteFontFile_Impl for FontFileObj_Impl {
    fn GetReferenceKey(&self, key: *mut *mut core::ffi::c_void, keysize: *mut u32) -> WResult<()> {
        unsafe {
            *key = self.key.as_ptr() as *mut core::ffi::c_void;
            *keysize = self.key.len() as u32;
        }
        Ok(())
    }
    fn GetLoader(&self) -> WResult<IDWriteFontFileLoader> {
        if self.remote {
            remote_loader().cast()
        } else if self.local {
            let l: IDWriteLocalFontFileLoader = LocalLoaderObj.into();
            l.cast()
        } else {
            Ok(MemLoaderObj { bytes: self.bytes.clone() }.into())
        }
    }
    fn Analyze(&self, issupported: *mut BOOL, filetype: *mut DWRITE_FONT_FILE_TYPE, facetype: *mut DWRITE_FONT_FACE_TYPE, numfaces: *mut u32) -> WResult<()> {
        let (ok, ft, fat, n) = analyze(&self.bytes);
        unsafe {
            *issupported = if ok { TRUE } else { FALSE };
            *filetype = ft;
            if !facetype.is_null() {
                *facetype = fat;
            }
            *numfaces = n;
        }
        Ok(())
    }
}

// --- IDWriteLocalFontFileLoader + IDWriteFontFileStream ----------------------------------------

#[implement(IDWriteLocalFontFileLoader)]
pub(crate) struct LocalLoaderObj;

/// A path's last-write time as a 64-bit Windows FILETIME (100 ns ticks since 1601-01-01), 0 if the
/// file is unreadable. On Windows this reproduces the NTFS file time native's GetLastWriteTime returns.
pub(crate) fn path_write_time(path: &str) -> u64 {
    const EPOCH_1601_TO_1970: u128 = 116_444_736_000_000_000; // 100 ns ticks between the two epochs
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| (d.as_nanos() / 100 + EPOCH_1601_TO_1970) as u64)
        .unwrap_or(0)
}

fn filetime(ticks: u64) -> FILETIME {
    FILETIME { dwLowDateTime: ticks as u32, dwHighDateTime: (ticks >> 32) as u32 }
}

/// Native reports a system-font path (one under the Windows fonts directory) uppercased, as its font
/// cache stores it; an app-provided path comes back exactly as given. Presentation only — the stored
/// key keeps the real on-disk casing so reads (incl. case-sensitive Wine) still resolve.
fn present_path(path: String) -> String {
    // `to_ascii_uppercase` is a per-unit, length-preserving map like native's RtlUpcaseUnicodeString;
    // `to_uppercase` would expand some characters (ß -> SS) and native never changes the length.
    let fonts = super::fonts::fonts_dir().to_string_lossy().to_ascii_lowercase();
    if !fonts.is_empty() && path.to_ascii_lowercase().starts_with(&fonts) {
        path.to_ascii_uppercase()
    } else {
        path
    }
}

/// `(last-write time, on-disk path string)` from a raw local loader key. See [`decode_key`].
fn read_key(key: *const core::ffi::c_void, size: u32) -> (u64, String) {
    let (wt, p) = decode_key(unsafe { slice::from_raw_parts(key as *const u8, size as usize) });
    (wt, p.to_string_lossy().into_owned())
}

/// Read a font file, falling back to a case-insensitive match in its directory (a compact key stores
/// the uppercase filename; Windows/Wine resolve it directly, a case-sensitive host needs the scan).
fn read_font_path(path: &str) -> WResult<Vec<u8>> {
    if let Ok(b) = std::fs::read(path) {
        return Ok(b);
    }
    let p = std::path::Path::new(path);
    if let (Some(dir), Some(name)) = (p.parent(), p.file_name()) {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for de in rd.flatten() {
                if de.file_name().eq_ignore_ascii_case(name) {
                    if let Ok(b) = std::fs::read(de.path()) {
                        return Ok(b);
                    }
                }
            }
        }
    }
    Err(E_INVALIDARG.into())
}

impl IDWriteFontFileLoader_Impl for LocalLoaderObj_Impl {
    fn CreateStreamFromKey(&self, key: *const core::ffi::c_void, keysize: u32) -> WResult<IDWriteFontFileStream> {
        let (wt, path) = read_key(key, keysize);
        Ok(StreamObj { bytes: Arc::new(read_font_path(&path)?), write: wt }.into())
    }
}

impl IDWriteLocalFontFileLoader_Impl for LocalLoaderObj_Impl {
    fn GetFilePathLengthFromKey(&self, key: *const core::ffi::c_void, keysize: u32) -> WResult<u32> {
        Ok(present_path(read_key(key, keysize).1).encode_utf16().count() as u32)
    }
    fn GetFilePathFromKey(&self, key: *const core::ffi::c_void, keysize: u32, filepath: PWSTR, filepathsize: u32) -> WResult<()> {
        super::write_wide(&present_path(read_key(key, keysize).1), filepath, filepathsize)
    }
    fn GetLastWriteTimeFromKey(&self, key: *const core::ffi::c_void, keysize: u32) -> WResult<FILETIME> {
        Ok(filetime(read_key(key, keysize).0))
    }
}

/// In-memory font file loader: holds the face's bytes and serves them for any key, so a memory-backed
/// `IDWriteFontFile` round-trips `GetReferenceKey` → `CreateStreamFromKey` → `ReadFileFragment` without
/// a path. (Native's custom-loader key is the app's opaque token + its own loader; we read the bytes
/// once at creation, so our loader serves those stored bytes instead — same round-trip, our key.)
#[implement(IDWriteFontFileLoader)]
pub(crate) struct MemLoaderObj {
    bytes: Arc<Vec<u8>>,
}

impl IDWriteFontFileLoader_Impl for MemLoaderObj_Impl {
    fn CreateStreamFromKey(&self, _key: *const core::ffi::c_void, _keysize: u32) -> WResult<IDWriteFontFileStream> {
        // In-memory: no file, so no write time (native's custom streams report what the app supplies).
        Ok(StreamObj { bytes: self.bytes.clone(), write: 0 }.into())
    }
}

#[implement(IDWriteFontFileStream)]
pub(crate) struct StreamObj {
    bytes: Arc<Vec<u8>>,
    write: u64,
}

impl IDWriteFontFileStream_Impl for StreamObj_Impl {
    fn ReadFileFragment(&self, start: *mut *mut core::ffi::c_void, offset: u64, size: u64, context: *mut *mut core::ffi::c_void) -> WResult<()> {
        if offset + size > self.bytes.len() as u64 {
            return Err(E_INVALIDARG.into());
        }
        unsafe {
            *start = self.bytes.as_ptr().add(offset as usize) as *mut core::ffi::c_void;
            *context = core::ptr::null_mut();
        }
        Ok(())
    }
    // Legitimate no-op: ReadFileFragment hands back a pointer into our owned `bytes` with a null
    // context (no per-fragment allocation), so there is nothing to free here — as native frees the
    // loader context it allocated, and ours is null.
    fn ReleaseFileFragment(&self, _context: *mut core::ffi::c_void) {}
    fn GetFileSize(&self) -> WResult<u64> {
        Ok(self.bytes.len() as u64)
    }
    fn GetLastWriteTime(&self) -> WResult<u64> {
        Ok(self.write)
    }
}

// --- IDWriteInMemoryFontFileLoader / IDWriteRemoteFontFileLoader (Factory5) ---------------------

/// An app's in-memory font loader (`IDWriteFactory5::CreateInMemoryFontFileLoader`): each
/// `CreateInMemoryFontFileReference` copies the caller's bytes and hands back a self-serving file.
#[implement(IDWriteInMemoryFontFileLoader)]
pub(crate) struct InMemoryLoaderObj {
    blobs: RefCell<Vec<Arc<Vec<u8>>>>,
}

pub(crate) fn in_memory_loader() -> IDWriteInMemoryFontFileLoader {
    InMemoryLoaderObj { blobs: RefCell::new(Vec::new()) }.into()
}

impl IDWriteFontFileLoader_Impl for InMemoryLoaderObj_Impl {
    fn CreateStreamFromKey(&self, key: *const core::ffi::c_void, keysize: u32) -> WResult<IDWriteFontFileStream> {
        // Our reference key is the blob index (u32 LE) assigned at creation.
        if keysize < 4 {
            return Err(E_INVALIDARG.into());
        }
        let idx = u32::from_le_bytes(unsafe { slice::from_raw_parts(key as *const u8, 4) }.try_into().unwrap()) as usize;
        let bytes = self.blobs.borrow().get(idx).cloned().ok_or(E_INVALIDARG)?;
        Ok(StreamObj { bytes, write: 0 }.into())
    }
}

impl IDWriteInMemoryFontFileLoader_Impl for InMemoryLoaderObj_Impl {
    fn CreateInMemoryFontFileReference(&self, _factory: Ref<IDWriteFactory>, fontdata: *const core::ffi::c_void, fontdatasize: u32, _ownerobject: Ref<windows_core::IUnknown>) -> WResult<IDWriteFontFile> {
        // Copy the caller's bytes now. ponytail: ownerObject is ignored — we own the copy, so there
        // is no caller-lifetime dependency to track.
        let bytes = Arc::new(unsafe { slice::from_raw_parts(fontdata as *const u8, fontdatasize as usize) }.to_vec());
        self.blobs.borrow_mut().push(bytes.clone());
        Ok(file_from_bytes(bytes))
    }
    fn GetFileCount(&self) -> u32 {
        self.blobs.borrow().len() as u32
    }
}

/// An app's HTTP/remote font loader (`IDWriteFactory5::CreateHttpFontFileLoader`). ponytail: this is
/// a local-only drop-in with no network, so every key is remote-and-unresolvable — a real loader
/// object (not an E_NOTIMPL factory method) whose download paths report `DWRITE_E_REMOTEFONT`.
#[implement(IDWriteRemoteFontFileLoader)]
pub(crate) struct RemoteLoaderObj;

pub(crate) fn remote_loader() -> IDWriteRemoteFontFileLoader {
    RemoteLoaderObj.into()
}

impl IDWriteFontFileLoader_Impl for RemoteLoaderObj_Impl {
    fn CreateStreamFromKey(&self, _key: *const core::ffi::c_void, _keysize: u32) -> WResult<IDWriteFontFileStream> {
        Err(DWRITE_E_REMOTEFONT.into())
    }
}

impl IDWriteRemoteFontFileLoader_Impl for RemoteLoaderObj_Impl {
    fn CreateRemoteStreamFromKey(&self, _fontfilereferencekey: *const core::ffi::c_void, _fontfilereferencekeysize: u32) -> WResult<IDWriteRemoteFontFileStream> {
        // An un-downloaded remote stream: it exists (native returns S_OK) but nothing is local yet, so
        // every read/download path reports the remote, not-yet-fetched state. ponytail: no HTTP client
        // — the font is never actually fetched (BeginDownload is unsupported).
        Ok(RemoteStreamObj.into())
    }
    fn GetLocalityFromKey(&self, _fontfilereferencekey: *const core::ffi::c_void, _fontfilereferencekeysize: u32) -> WResult<DWRITE_LOCALITY> {
        Ok(DWRITE_LOCALITY_REMOTE)
    }
    fn CreateFontFileReferenceFromUrl(&self, _factory: Ref<IDWriteFactory>, baseurl: &PCWSTR, fontfileurl: &PCWSTR) -> WResult<IDWriteFontFile> {
        // Mint a lazy reference keyed by the resolved URL as UTF-16LE (native's key is the URL too).
        // The HTTP loader must be registered first, else native E_INVALIDARGs the call — handled by the
        // caller (RegisterFontFileLoader). A null/empty URL is E_INVALIDARG.
        let base = unsafe { wide_to_string(baseurl.0, None) };
        let rest = unsafe { wide_to_string(fontfileurl.0, None) };
        let url = format!("{base}{rest}");
        if url.is_empty() {
            return Err(E_INVALIDARG.into());
        }
        let key: Vec<u8> = url.encode_utf16().flat_map(|u| u.to_le_bytes()).collect();
        Ok(file_remote(key))
    }
}

/// An un-downloaded `IDWriteRemoteFontFileStream`: the font lives at a remote URL and has not been
/// fetched, so it is fully remote with nothing local. ponytail: no HTTP client, so `BeginDownload`
/// (the actual fetch) is unsupported — only the pre-download observable state is reproduced.
#[implement(IDWriteRemoteFontFileStream)]
pub(crate) struct RemoteStreamObj;

impl IDWriteFontFileStream_Impl for RemoteStreamObj_Impl {
    fn ReadFileFragment(&self, _s: *mut *mut core::ffi::c_void, _o: u64, _n: u64, _c: *mut *mut core::ffi::c_void) -> WResult<()> {
        Err(DWRITE_E_REMOTEFONT.into()) // nothing is local to read yet
    }
    fn ReleaseFileFragment(&self, context: *mut core::ffi::c_void) {
        // Verified native no-op: the remote file stream's ReleaseFileFragment returns immediately
        // (a bare `ret`, folded with other trivial no-ops).
        // Our ReadFileFragment hands back no fragment, so there is never a context to release.
        let _ = context;
    }
    fn GetFileSize(&self) -> WResult<u64> {
        Err(DWRITE_E_REMOTEFONT.into()) // the size is unknown until downloaded
    }
    fn GetLastWriteTime(&self) -> WResult<u64> {
        Err(DWRITE_E_REMOTEFONT.into())
    }
}

impl IDWriteRemoteFontFileStream_Impl for RemoteStreamObj_Impl {
    fn GetLocalFileSize(&self) -> WResult<u64> {
        Ok(0) // nothing downloaded
    }
    fn GetFileFragmentLocality(&self, fileoffset: u64, fragmentsize: u64, islocal: *mut BOOL, partialsize: *mut u64) -> WResult<()> {
        // Nothing is downloaded, so the local size is 0: a fragment that extends past it is out of
        // range (native returns E_INVALIDARG for e.g. offset 0 / size 4 pre-download).
        if fileoffset.saturating_add(fragmentsize) > 0 {
            return Err(E_INVALIDARG.into());
        }
        unsafe {
            if !islocal.is_null() { *islocal = FALSE; }
            if !partialsize.is_null() { *partialsize = 0; }
        }
        Ok(())
    }
    fn GetLocality(&self) -> DWRITE_LOCALITY {
        DWRITE_LOCALITY_REMOTE
    }
    fn BeginDownload(&self, _id: *const windows_core::GUID, frags: *const DWRITE_FILE_FRAGMENT, count: u32) -> WResult<IDWriteAsyncResult> {
        // A specific fragment can't be validated before the size is known → E_INVALIDARG (native). An
        // empty fragment list means "download the whole file": native starts it and it completes with
        // DWRITE_E_FILEACCESS on the reference VM (the font never downloads there — matched byte-exact).
        // We do not reproduce a successful fetch, which native itself does not perform here.
        let _ = frags;
        if count != 0 {
            return Err(E_INVALIDARG.into());
        }
        Ok(AsyncResultObj { result: DWRITE_E_FILEACCESS }.into())
    }
}

/// `DWRITE_E_FILEACCESS` — the result a whole-file remote download completes with on the reference VM.
const DWRITE_E_FILEACCESS: windows_core::HRESULT = windows_core::HRESULT(0x8898_5004u32 as i32);

/// A completed `IDWriteAsyncResult` carrying a fixed HRESULT (the remote download's outcome). The wait
/// handle is null — the operation is already finished, so a waiter proceeds straight to `GetResult`.
#[implement(IDWriteAsyncResult)]
pub(crate) struct AsyncResultObj {
    result: windows_core::HRESULT,
}

impl IDWriteAsyncResult_Impl for AsyncResultObj_Impl {
    fn GetWaitHandle(&self) -> windows::Win32::Foundation::HANDLE {
        windows::Win32::Foundation::HANDLE(core::ptr::null_mut())
    }
    fn GetResult(&self) -> WResult<()> {
        self.result.ok()
    }
}

// --- Container (WOFF) analysis + unpack (Factory5) ---------------------------------------------

/// `IDWriteFactory5::AnalyzeContainerType`: the wrapper format from the 4-byte signature.
pub(crate) fn analyze_container(data: &[u8]) -> DWRITE_CONTAINER_TYPE {
    match data.get(0..4) {
        Some(b"wOFF") => DWRITE_CONTAINER_TYPE_WOFF,
        Some(b"wOF2") => DWRITE_CONTAINER_TYPE_WOFF2,
        _ => DWRITE_CONTAINER_TYPE_UNKNOWN,
    }
}

/// `IDWriteFactory5::UnpackFontFile`: decode a container to a plain sfnt stream. WOFF is decoded.
pub(crate) fn unpack(container: DWRITE_CONTAINER_TYPE, data: &[u8]) -> WResult<IDWriteFontFileStream> {
    let sfnt = match container {
        DWRITE_CONTAINER_TYPE_WOFF => woff_to_sfnt(data)?,
        DWRITE_CONTAINER_TYPE_WOFF2 => woff2_to_sfnt(data)?,
        // Native returns DWRITE_E_FILEFORMAT (0x88985000) for a format it cannot unpack.
        _ => return Err(windows_core::HRESULT(0x8898_5000u32 as i32).into()),
    };
    Ok(StreamObj { bytes: Arc::new(sfnt), write: 0 }.into())
}

/// Reassemble a WOFF 1.0 container into a plain sfnt: per-table zlib inflate (when compressed) and a
/// fresh sfnt table directory. (OpenType WOFF File Format 1.0 — 44-byte header, 20-byte table dir.)
fn woff_to_sfnt(data: &[u8]) -> WResult<Vec<u8>> {
    let bad = || windows_core::Error::from(E_INVALIDARG);
    if data.len() < 44 || &data[0..4] != b"wOFF" {
        return Err(bad());
    }
    let be16 = |o: usize| u16::from_be_bytes([data[o], data[o + 1]]);
    let be32 = |o: usize| u32::from_be_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
    let flavor = be32(4);
    let num = be16(12) as usize;
    if data.len() < 44 + num * 20 {
        return Err(bad());
    }
    // sfnt offset table: sfntVersion, numTables, searchRange = 2^floor(log2(num))*16, entrySelector,
    // rangeShift = numTables*16 - searchRange.
    let entry_sel = 15u16.saturating_sub((num.max(1) as u16).leading_zeros() as u16);
    let search = (1u16 << entry_sel) * 16;
    let mut sfnt = Vec::new();
    sfnt.extend_from_slice(&flavor.to_be_bytes());
    sfnt.extend_from_slice(&(num as u16).to_be_bytes());
    sfnt.extend_from_slice(&search.to_be_bytes());
    sfnt.extend_from_slice(&entry_sel.to_be_bytes());
    sfnt.extend_from_slice(&((num as u16 * 16).wrapping_sub(search)).to_be_bytes());
    let dir_end = 12 + num * 16;
    let (mut dir, mut body) = (Vec::new(), Vec::new());
    for i in 0..num {
        let e = 44 + i * 20;
        let (off, comp, orig, checksum) = (be32(e + 4) as usize, be32(e + 8) as usize, be32(e + 12) as usize, be32(e + 16));
        if off.checked_add(comp).is_none_or(|end| end > data.len()) {
            return Err(bad());
        }
        let raw = &data[off..off + comp];
        let table = if comp != orig {
            let mut out = Vec::with_capacity(orig);
            ZlibDecoder::new(raw).read_to_end(&mut out).map_err(|_| bad())?;
            out
        } else {
            raw.to_vec()
        };
        dir.extend_from_slice(&data[e..e + 4]); // tag
        dir.extend_from_slice(&checksum.to_be_bytes());
        dir.extend_from_slice(&((dir_end + body.len()) as u32).to_be_bytes());
        dir.extend_from_slice(&(orig as u32).to_be_bytes());
        body.extend_from_slice(&table);
        while body.len() % 4 != 0 {
            body.push(0); // tables are 4-byte aligned in an sfnt
        }
    }
    sfnt.extend_from_slice(&dir);
    sfnt.extend_from_slice(&body);
    Ok(sfnt)
}

/// WOFF2 known-table-tag table: `flags & 0x3f` indexes it (63 ⇒ an explicit 4-byte tag follows).
/// (W3C WOFF2 §4.1, Table 2.)
#[rustfmt::skip]
const WOFF2_KNOWN_TAGS: [&[u8; 4]; 63] = [
    b"cmap", b"head", b"hhea", b"hmtx", b"maxp", b"name", b"OS/2", b"post",
    b"cvt ", b"fpgm", b"glyf", b"loca", b"prep", b"CFF ", b"VORG", b"EBDT",
    b"EBLC", b"gasp", b"hdmx", b"kern", b"LTSH", b"PCLT", b"VDMX", b"vhea",
    b"vmtx", b"BASE", b"GDEF", b"GPOS", b"GSUB", b"EBSC", b"JSTF", b"MATH",
    b"CBDT", b"CBLC", b"COLR", b"CPAL", b"SVG ", b"sbix", b"acnt", b"avar",
    b"bdat", b"bloc", b"bsln", b"cvar", b"fdsc", b"feat", b"fmtx", b"fvar",
    b"gvar", b"hsty", b"just", b"lcar", b"mort", b"morx", b"opbd", b"prop",
    b"trak", b"Zapf", b"Silf", b"Glat", b"Gloc", b"Feat", b"Sill",
];

/// A big-endian byte cursor with the two WOFF2 variable-length integer readers. Any read past the end
/// returns `None`.
struct Cur<'a> {
    d: &'a [u8],
    p: usize,
}
impl<'a> Cur<'a> {
    fn new(d: &'a [u8]) -> Self {
        Cur { d, p: 0 }
    }
    fn u8(&mut self) -> Option<u8> {
        let b = *self.d.get(self.p)?;
        self.p += 1;
        Some(b)
    }
    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_be_bytes([self.u8()?, self.u8()?]))
    }
    fn i16(&mut self) -> Option<i16> {
        Some(self.u16()? as i16)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_be_bytes([self.u8()?, self.u8()?, self.u8()?, self.u8()?]))
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.d.get(self.p..self.p.checked_add(n)?)?;
        self.p += n;
        Some(s)
    }
    /// `UIntBase128` (WOFF2 §6.1.2): up to five base-128 bytes, high bit = "more follow".
    fn base128(&mut self) -> Option<u32> {
        let mut accum: u32 = 0;
        for i in 0..5 {
            let b = self.u8()?;
            if i == 0 && b == 0x80 {
                return None; // no leading zeros
            }
            if accum & 0xfe00_0000 != 0 {
                return None; // the next <<7 would overflow u32
            }
            accum = (accum << 7) | (b & 0x7f) as u32;
            if b & 0x80 == 0 {
                return Some(accum);
            }
        }
        None
    }
    /// `255UInt16` (WOFF2 §6.1.1).
    fn pack255(&mut self) -> Option<u16> {
        match self.u8()? {
            253 => self.u16(),
            255 => Some(253 + self.u8()? as u16),
            254 => Some(506 + self.u8()? as u16),
            b => Some(b as u16),
        }
    }
}

/// `wOF2` container → a plain sfnt (W3C WOFF2). Parses the table directory, brotli-decompresses the
/// single data stream, reconstructs a transformed `glyf`/`loca` pair, passes other tables through, and
/// reassembles the sfnt. WOFF2 is the web-font wire format browsers hand to `UnpackFontFile`.
fn woff2_to_sfnt(data: &[u8]) -> WResult<Vec<u8>> {
    let bad = || windows_core::Error::from(E_INVALIDARG);
    let mut h = Cur::new(data);
    if h.u32() != Some(0x774F_4632) {
        return Err(bad()); // 'wOF2'
    }
    let flavor = h.u32().ok_or_else(bad)?;
    let _length = h.u32().ok_or_else(bad)?;
    let num_tables = h.u16().ok_or_else(bad)? as usize;
    let _reserved = h.u16().ok_or_else(bad)?;
    let _total_sfnt_size = h.u32().ok_or_else(bad)?;
    let total_compressed = h.u32().ok_or_else(bad)? as usize;
    let _ver = (h.u16().ok_or_else(bad)?, h.u16().ok_or_else(bad)?);
    let _meta = (h.u32(), h.u32(), h.u32());
    let _privd = (h.u32(), h.u32());

    // Table directory: flag byte (tag index + transform version), origLength, and transformLength when
    // the table carries a transform (glyf/loca at version 0).
    struct Tbl {
        tag: [u8; 4],
        orig_len: usize,
        stream_len: usize,
        transformed: bool,
    }
    let mut tables: Vec<Tbl> = Vec::with_capacity(num_tables);
    for _ in 0..num_tables {
        let flags = h.u8().ok_or_else(bad)?;
        let idx = (flags & 0x3f) as usize;
        let tag = if idx == 63 {
            let t = h.take(4).ok_or_else(bad)?;
            [t[0], t[1], t[2], t[3]]
        } else {
            **WOFF2_KNOWN_TAGS.get(idx).ok_or_else(bad)?
        };
        let tver = flags >> 6;
        let orig_len = h.base128().ok_or_else(bad)? as usize;
        // glyf/loca are transformed at version 0 (version 3 = null transform); other tables carry no
        // transform we reverse (a non-zero version on them would be the rare hmtx transform — rejected).
        let is_glyf_loca = &tag == b"glyf" || &tag == b"loca";
        let transformed = if is_glyf_loca { tver == 0 } else { tver != 0 };
        let stream_len = if transformed { h.base128().ok_or_else(bad)? as usize } else { orig_len };
        tables.push(Tbl { tag, orig_len, stream_len, transformed });
    }

    // One brotli stream holds every table's data concatenated (no padding), in directory order.
    let comp = h.take(total_compressed).ok_or_else(bad)?;
    let mut raw: Vec<u8> = Vec::new();
    brotli_decompressor::BrotliDecompress(&mut { comp }, &mut raw).map_err(|_| bad())?;

    // Slice each table out of the decompressed stream.
    let mut pos = 0usize;
    let mut entries: Vec<([u8; 4], bool, &[u8])> = Vec::with_capacity(num_tables);
    for t in &tables {
        let s = raw.get(pos..pos.checked_add(t.stream_len).ok_or_else(bad)?).ok_or_else(bad)?;
        pos += t.stream_len;
        entries.push((t.tag, t.transformed, s));
        let _ = t.orig_len; // origLength is advisory for untransformed tables; stream_len == orig_len there
    }

    // Reconstruct glyf + loca from the transformed glyf stream (if any); loca's own stream is empty.
    let (mut glyf_recon, mut loca_recon) = (None, None);
    if let Some((_, _, gs)) = entries.iter().find(|(tag, tr, _)| tag == b"glyf" && *tr) {
        let (g, l) = reconstruct_glyf(gs)?;
        glyf_recon = Some(g);
        loca_recon = Some(l);
    }

    let mut out: Vec<([u8; 4], Vec<u8>)> = Vec::with_capacity(num_tables);
    for (tag, tr, slice) in &entries {
        let buf = if tag == b"glyf" && *tr {
            glyf_recon.take().ok_or_else(bad)?
        } else if tag == b"loca" && *tr {
            loca_recon.take().ok_or_else(bad)?
        } else if *tr {
            // ponytail: a transformed non-glyf/loca table is the optional hmtx transform (WOFF2 §5.3);
            // woff2_compress does not emit it. Reverse it when a font that uses it turns up.
            return Err(bad());
        } else {
            slice.to_vec()
        };
        out.push((*tag, buf));
    }

    Ok(assemble_sfnt(flavor, out))
}

/// Build an sfnt from `(tag, data)` tables (sorted by tag, 4-byte aligned, per-table checksums).
fn assemble_sfnt(flavor: u32, mut tables: Vec<([u8; 4], Vec<u8>)>) -> Vec<u8> {
    tables.sort_by(|a, b| a.0.cmp(&b.0));
    let num = tables.len();
    let entry_sel = 15u16.saturating_sub((num.max(1) as u16).leading_zeros() as u16);
    let search = (1u16 << entry_sel) * 16;
    let mut sfnt = Vec::new();
    sfnt.extend_from_slice(&flavor.to_be_bytes());
    sfnt.extend_from_slice(&(num as u16).to_be_bytes());
    sfnt.extend_from_slice(&search.to_be_bytes());
    sfnt.extend_from_slice(&entry_sel.to_be_bytes());
    sfnt.extend_from_slice(&((num as u16 * 16).wrapping_sub(search)).to_be_bytes());
    let dir_end = 12 + num * 16;
    let (mut dir, mut body) = (Vec::new(), Vec::new());
    for (tag, buf) in &tables {
        dir.extend_from_slice(tag);
        dir.extend_from_slice(&table_checksum(buf).to_be_bytes());
        dir.extend_from_slice(&((dir_end + body.len()) as u32).to_be_bytes());
        dir.extend_from_slice(&(buf.len() as u32).to_be_bytes());
        body.extend_from_slice(buf);
        while body.len() % 4 != 0 {
            body.push(0);
        }
    }
    sfnt.extend_from_slice(&dir);
    sfnt.extend_from_slice(&body);
    sfnt
}

/// sfnt table checksum: the sum of the table's big-endian `u32` words, zero-padded to 4 bytes.
fn table_checksum(buf: &[u8]) -> u32 {
    let mut sum = 0u32;
    let mut i = 0;
    while i < buf.len() {
        let mut w = [0u8; 4];
        let n = (buf.len() - i).min(4);
        w[..n].copy_from_slice(&buf[i..i + n]);
        sum = sum.wrapping_add(u32::from_be_bytes(w));
        i += 4;
    }
    sum
}

/// WOFF2 point delta sign: bit 0 of `flag` selects the sign of `base` (§5.2).
fn with_sign(flag: u8, base: i32) -> i32 {
    if flag & 1 != 0 { base } else { -base }
}

/// Decode `n` triplet-encoded points from the glyph stream into absolute `(x, y)` coordinates
/// (WOFF2 §5.2, the Google `woff2` reference `TripletDecode`). `flags[i]` bit 7 = off-curve.
fn triplet_decode(flags: &[u8], g: &mut Cur, n: usize) -> Option<(Vec<i16>, Vec<i16>)> {
    let (mut xs, mut ys) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut x, mut y) = (0i32, 0i32);
    for &fl in flags.iter().take(n) {
        let flag = fl & 0x7f;
        let (dx, dy);
        if flag < 10 {
            let b = g.u8()? as i32;
            dx = 0;
            dy = with_sign(flag, ((flag as i32 & 14) << 7) + b);
        } else if flag < 20 {
            let b = g.u8()? as i32;
            dx = with_sign(flag, (((flag as i32 - 10) & 14) << 7) + b);
            dy = 0;
        } else if flag < 84 {
            let b0 = flag as i32 - 20;
            let b1 = g.u8()? as i32;
            dx = with_sign(flag, 1 + (b0 & 0x30) + (b1 >> 4));
            dy = with_sign(flag >> 1, 1 + ((b0 & 0x0c) << 2) + (b1 & 0x0f));
        } else if flag < 120 {
            let b0 = flag as i32 - 84;
            let b1 = g.u8()? as i32;
            let b2 = g.u8()? as i32;
            dx = with_sign(flag, 1 + ((b0 / 12) << 8) + b1);
            dy = with_sign(flag >> 1, 1 + (((b0 % 12) >> 2) << 8) + b2);
        } else if flag < 124 {
            let b1 = g.u8()? as i32;
            let b2 = g.u8()? as i32;
            let b3 = g.u8()? as i32;
            dx = with_sign(flag, (b1 << 4) + (b2 >> 4));
            dy = with_sign(flag >> 1, ((b2 & 0x0f) << 8) + b3);
        } else {
            let b1 = g.u8()? as i32;
            let b2 = g.u8()? as i32;
            let b3 = g.u8()? as i32;
            let b4 = g.u8()? as i32;
            dx = with_sign(flag, (b1 << 8) + b2);
            dy = with_sign(flag >> 1, (b3 << 8) + b4);
        }
        x += dx;
        y += dy;
        xs.push(x as i16);
        ys.push(y as i16);
    }
    Some((xs, ys))
}

/// Reconstruct a standard `glyf` table and its `loca` from the WOFF2 transformed-glyf stream
/// (WOFF2 §5.1). Returns `(glyf, loca)`. Points are re-emitted uncompressed (1 flag byte + 16-bit
/// deltas per point) — a valid, if un-minimised, glyf whose outlines equal the original.
fn reconstruct_glyf(data: &[u8]) -> WResult<(Vec<u8>, Vec<u8>)> {
    let bad = || windows_core::Error::from(E_INVALIDARG);
    let mut c = Cur::new(data);
    let _reserved = c.u16().ok_or_else(bad)?;
    let option_flags = c.u16().ok_or_else(bad)?;
    let num_glyphs = c.u16().ok_or_else(bad)? as usize;
    let index_format = c.u16().ok_or_else(bad)?;
    let sizes: Vec<usize> = (0..7).map(|_| c.u32().map(|v| v as usize)).collect::<Option<_>>().ok_or_else(bad)?;
    let mut n_contour = Cur::new(c.take(sizes[0]).ok_or_else(bad)?);
    let mut n_points = Cur::new(c.take(sizes[1]).ok_or_else(bad)?);
    let mut flag_s = Cur::new(c.take(sizes[2]).ok_or_else(bad)?);
    let mut glyph_s = Cur::new(c.take(sizes[3]).ok_or_else(bad)?);
    let mut composite_s = Cur::new(c.take(sizes[4]).ok_or_else(bad)?);
    let bbox_stream = c.take(sizes[5]).ok_or_else(bad)?;
    let mut instr_s = Cur::new(c.take(sizes[6]).ok_or_else(bad)?);
    // The bbox stream opens with a one-bit-per-glyph "has explicit bbox" bitmap.
    let bitmap_len = num_glyphs.div_ceil(8);
    let (bbox_bitmap, bbox_data) = bbox_stream.split_at(bitmap_len.min(bbox_stream.len()));
    let mut bbox_s = Cur::new(bbox_data);
    // `overlapSimpleBitmap` (optionFlags bit 0) is a per-glyph OVERLAP_SIMPLE hint we drop (cosmetic).
    let _overlap = (option_flags & 1 != 0).then(|| ()); // present but unused by outline reconstruction

    let has_bbox = |gi: usize| bbox_bitmap.get(gi / 8).is_some_and(|b| b >> (7 - (gi % 8)) & 1 == 1);
    let mut glyf: Vec<u8> = Vec::new();
    let mut loca: Vec<u32> = Vec::with_capacity(num_glyphs + 1);
    loca.push(0);
    for gi in 0..num_glyphs {
        let nc = n_contour.i16().ok_or_else(bad)?;
        if nc > 0 {
            let ncontours = nc as usize;
            let mut end_pts = Vec::with_capacity(ncontours);
            let mut total = 0usize;
            for _ in 0..ncontours {
                total += n_points.pack255().ok_or_else(bad)? as usize;
                end_pts.push((total as u16).wrapping_sub(1));
            }
            let flags = flag_s.take(total).ok_or_else(bad)?;
            let (xs, ys) = triplet_decode(flags, &mut glyph_s, total).ok_or_else(bad)?;
            let instr_len = glyph_s.pack255().ok_or_else(bad)? as usize;
            let instrs = instr_s.take(instr_len).ok_or_else(bad)?;
            let (xmin, ymin, xmax, ymax) = if has_bbox(gi) {
                (bbox_s.i16().ok_or_else(bad)?, bbox_s.i16().ok_or_else(bad)?, bbox_s.i16().ok_or_else(bad)?, bbox_s.i16().ok_or_else(bad)?)
            } else {
                bbox_from_points(&xs, &ys)
            };
            glyf.extend_from_slice(&(ncontours as i16).to_be_bytes());
            for v in [xmin, ymin, xmax, ymax] {
                glyf.extend_from_slice(&v.to_be_bytes());
            }
            for &e in &end_pts {
                glyf.extend_from_slice(&e.to_be_bytes());
            }
            glyf.extend_from_slice(&(instr_len as u16).to_be_bytes());
            glyf.extend_from_slice(instrs);
            for &fl in flags {
                glyf.push(if fl >> 7 == 0 { 0x01 } else { 0x00 }); // ON_CURVE_POINT, no short/same compression
            }
            let mut prev = 0i16;
            for &xv in &xs {
                glyf.extend_from_slice(&xv.wrapping_sub(prev).to_be_bytes());
                prev = xv;
            }
            let mut prev = 0i16;
            for &yv in &ys {
                glyf.extend_from_slice(&yv.wrapping_sub(prev).to_be_bytes());
                prev = yv;
            }
        } else if nc == -1 {
            // Composite: the component records are verbatim standard-glyf in the composite stream.
            let start = composite_s.p;
            let mut have_instr = false;
            loop {
                let flags = composite_s.u16().ok_or_else(bad)?;
                let _glyph_index = composite_s.u16().ok_or_else(bad)?;
                composite_s.take(if flags & 0x0001 != 0 { 4 } else { 2 }).ok_or_else(bad)?; // args
                if flags & 0x0008 != 0 {
                    composite_s.take(2).ok_or_else(bad)?; // WE_HAVE_A_SCALE
                } else if flags & 0x0040 != 0 {
                    composite_s.take(4).ok_or_else(bad)?; // X_AND_Y_SCALE
                } else if flags & 0x0080 != 0 {
                    composite_s.take(8).ok_or_else(bad)?; // 2x2
                }
                have_instr |= flags & 0x0100 != 0; // WE_HAVE_INSTRUCTIONS
                if flags & 0x0020 == 0 {
                    break; // no MORE_COMPONENTS
                }
            }
            let comp_bytes = composite_s.d.get(start..composite_s.p).ok_or_else(bad)?;
            // Composites always carry an explicit bbox.
            let (xmin, ymin, xmax, ymax) = (bbox_s.i16().ok_or_else(bad)?, bbox_s.i16().ok_or_else(bad)?, bbox_s.i16().ok_or_else(bad)?, bbox_s.i16().ok_or_else(bad)?);
            glyf.extend_from_slice(&(-1i16).to_be_bytes());
            for v in [xmin, ymin, xmax, ymax] {
                glyf.extend_from_slice(&v.to_be_bytes());
            }
            glyf.extend_from_slice(comp_bytes);
            if have_instr {
                let instr_len = glyph_s.pack255().ok_or_else(bad)? as usize;
                let instrs = instr_s.take(instr_len).ok_or_else(bad)?;
                glyf.extend_from_slice(&(instr_len as u16).to_be_bytes());
                glyf.extend_from_slice(instrs);
            }
        }
        // nc == 0 ⇒ empty glyph, no data.
        while glyf.len() % 2 != 0 {
            glyf.push(0); // even-align so short loca offsets (offset/2) stay exact
        }
        loca.push(glyf.len() as u32);
    }
    let loca_bytes = if index_format == 0 {
        loca.iter().flat_map(|&o| ((o / 2) as u16).to_be_bytes()).collect()
    } else {
        loca.iter().flat_map(|&o| o.to_be_bytes()).collect()
    };
    Ok((glyf, loca_bytes))
}

/// Bounding box of absolute point coordinates (an empty glyph has all-zero bounds).
fn bbox_from_points(xs: &[i16], ys: &[i16]) -> (i16, i16, i16, i16) {
    if xs.is_empty() {
        return (0, 0, 0, 0);
    }
    (
        *xs.iter().min().unwrap(),
        *ys.iter().min().unwrap(),
        *xs.iter().max().unwrap(),
        *ys.iter().max().unwrap(),
    )
}

// --- CreateFontFace ----------------------------------------------------------------------------

/// Read a whole font file's bytes: fast-path our own [`FontFileObj`], else go through the file's
/// loader + stream (an app's custom file, or a foreign `IDWriteFontFile`).
pub(crate) fn read_font_file(file: &IDWriteFontFile) -> WResult<Arc<Vec<u8>>> {
    if let Ok(obj) = file.cast_object_ref::<FontFileObj>() {
        return Ok(obj.bytes.clone());
    }
    let loader = unsafe { file.GetLoader()? };
    let (mut key, mut keysize) = (core::ptr::null_mut(), 0u32);
    unsafe { file.GetReferenceKey(&mut key, &mut keysize)? };
    Ok(Arc::new(read_via_loader(&loader, unsafe { slice::from_raw_parts(key as *const u8, keysize as usize) })?))
}

/// Read a file's bytes through a loader + stream (`CreateStreamFromKey` → `ReadFileFragment`).
fn read_via_loader(loader: &IDWriteFontFileLoader, key: &[u8]) -> WResult<Vec<u8>> {
    unsafe {
        let stream = loader.CreateStreamFromKey(key.as_ptr() as *const core::ffi::c_void, key.len() as u32)?;
        let size = stream.GetFileSize()?;
        let (mut frag, mut ctx) = (core::ptr::null_mut(), core::ptr::null_mut());
        stream.ReadFileFragment(&mut frag, 0, size, &mut ctx)?;
        let bytes = slice::from_raw_parts(frag as *const u8, size as usize).to_vec();
        stream.ReleaseFileFragment(ctx);
        Ok(bytes)
    }
}

/// `CreateFontFace`: build a face from the first file (one file per TrueType/OpenType face) at
/// `index`. Returns the same [`FaceObj`] the system collection yields.
pub(crate) fn make_font_face(numfiles: u32, files: *const Option<IDWriteFontFile>, index: u32) -> WResult<IDWriteFontFace> {
    if numfiles == 0 || files.is_null() {
        return Err(E_INVALIDARG.into());
    }
    let file = unsafe { (*files).as_ref() }.ok_or(E_INVALIDARG)?;
    let bytes = read_font_file(file)?;
    if FontRef::from_index(&bytes, index).is_err() {
        return Err(E_INVALIDARG.into());
    }
    let family = family_name(&bytes, index);
    // Preserve this file's form (path vs in-memory) so the face's GetFiles round-trips like native.
    let source = file.cast_object_ref::<FontFileObj>().map(|f| f.as_source()).unwrap_or_else(|_| super::fonts::FontSource::Bytes(bytes.clone()));
    let face: IDWriteFontFace7 = FaceObj { bytes, index, family, source, sims: 0, axis_values: Vec::new() }.into();
    Ok(face.cast()?)
}

// --- IDWriteFontFaceReference(1) ---------------------------------------------------------------

#[implement(IDWriteFontFaceReference1)]
pub(crate) struct FaceRefObj {
    pub(crate) bytes: Arc<Vec<u8>>,
    pub(crate) index: u32,
    pub(crate) sims: DWRITE_FONT_SIMULATIONS,
    /// The on-disk path (for `GetFileTime`), or `None` for an in-memory reference.
    pub(crate) path: Option<String>,
}

/// A `FaceRefObj` as the base `IDWriteFontFaceReference` (the common return type), built from the
/// `IDWriteFontFaceReference1` leaf the object now implements.
pub(crate) fn faceref(bytes: Arc<Vec<u8>>, index: u32, sims: DWRITE_FONT_SIMULATIONS, path: Option<String>) -> IDWriteFontFaceReference {
    let r: IDWriteFontFaceReference1 = FaceRefObj { bytes, index, sims, path }.into();
    r.cast().unwrap()
}

/// `CreateFontFaceReference(fontFile, index, sims)`.
pub(crate) fn faceref_from_file(file: &IDWriteFontFile, index: u32, sims: DWRITE_FONT_SIMULATIONS) -> WResult<IDWriteFontFaceReference> {
    let path = file.cast_object_ref::<FontFileObj>().ok().and_then(|f| f.as_source().path_str());
    Ok(faceref(read_font_file(file)?, index, sims, path))
}

/// `CreateFontFaceReference2(filePath, index, sims)`.
pub(crate) fn faceref_from_path(path: &PCWSTR, index: u32, sims: DWRITE_FONT_SIMULATIONS) -> WResult<IDWriteFontFaceReference> {
    let p = wide_to_string(path.0, None);
    let bytes = std::fs::read(&p).map_err(|_| windows_core::Error::from(E_INVALIDARG))?;
    Ok(faceref(Arc::new(bytes), index, sims, Some(p)))
}

impl FaceRefObj {
    pub(crate) fn source(&self) -> super::fonts::FontSource {
        self.path.as_ref().map(|p| super::fonts::FontSource::Path(p.into()))
            .unwrap_or_else(|| super::fonts::FontSource::Bytes(self.bytes.clone()))
    }
    fn face3(&self) -> WResult<IDWriteFontFace3> {
        if FontRef::from_index(&self.bytes, self.index).is_err() {
            return Err(E_INVALIDARG.into());
        }
        let f: IDWriteFontFace7 = FaceObj { bytes: self.bytes.clone(), index: self.index, family: family_name(&self.bytes, self.index), source: self.source(), sims: self.sims.0 as u16, axis_values: Vec::new() }.into();
        Ok(f.cast()?)
    }
    /// The axis values this reference resolves to (`IDWriteFontFaceReference1::GetFontAxisValues`).
    /// We create references without explicit axis values, so these are the face's own values — a
    /// variable font's `fvar` defaults, or a static font's synthesized set.
    fn axis_values(&self) -> Vec<DWRITE_FONT_AXIS_VALUE> {
        FontRef::from_index(&self.bytes, self.index).ok().map(|f| super::fonts::face_axis_values(&f)).unwrap_or_default()
    }
}

impl IDWriteFontFaceReference1_Impl for FaceRefObj_Impl {
    fn CreateFontFace(&self) -> WResult<IDWriteFontFace5> {
        Ok(self.face3()?.cast()?)
    }
    fn GetFontAxisValueCount(&self) -> u32 {
        self.axis_values().len() as u32
    }
    fn GetFontAxisValues(&self, fontaxisvalues: *mut DWRITE_FONT_AXIS_VALUE, fontaxisvaluecount: u32) -> WResult<()> {
        super::text::write_axis_values(&self.axis_values(), fontaxisvalues, fontaxisvaluecount)
    }
}

impl IDWriteFontFaceReference_Impl for FaceRefObj_Impl {
    fn CreateFontFace(&self) -> WResult<IDWriteFontFace3> {
        self.face3()
    }
    fn CreateFontFaceWithSimulations(&self, _sims: DWRITE_FONT_SIMULATIONS) -> WResult<IDWriteFontFace3> {
        self.face3()
    }
    fn Equals(&self, other: Ref<IDWriteFontFaceReference>) -> BOOL {
        // Native equality is (same font file, same face index, same simulations). Two references to
        // the same file loaded separately (CreateFontFaceReference from a file vs from a path) hold
        // distinct buffers, so fall back from the pointer fast-path to a content compare.
        let same = other.as_ref().and_then(|o| o.cast_object_ref::<FaceRefObj>().ok()).is_some_and(|o| {
            o.index == self.index && o.sims == self.sims && (Arc::ptr_eq(&o.bytes, &self.bytes) || o.bytes == self.bytes)
        });
        if same { TRUE } else { FALSE }
    }
    fn GetFontFaceIndex(&self) -> u32 {
        self.index
    }
    fn GetSimulations(&self) -> DWRITE_FONT_SIMULATIONS {
        self.sims
    }
    fn GetFontFile(&self) -> WResult<IDWriteFontFile> {
        Ok(FontFileObj { bytes: self.bytes.clone(), key: Vec::new(), local: false, remote: false }.into())
    }
    fn GetLocalFileSize(&self) -> u64 {
        self.bytes.len() as u64
    }
    fn GetFileSize(&self) -> u64 {
        self.bytes.len() as u64
    }
    fn GetFileTime(&self) -> WResult<FILETIME> {
        Ok(filetime(self.path.as_deref().map(path_write_time).unwrap_or(0)))
    }
    fn GetLocality(&self) -> DWRITE_LOCALITY {
        DWRITE_LOCALITY_LOCAL
    }
    fn EnqueueFontDownloadRequest(&self) -> WResult<()> {
        Ok(())
    }
    fn EnqueueCharacterDownloadRequest(&self, _characters: &PCWSTR, _count: u32) -> WResult<()> {
        Ok(())
    }
    fn EnqueueGlyphDownloadRequest(&self, _glyphindices: *const u16, _count: u32) -> WResult<()> {
        Ok(())
    }
    fn EnqueueFileFragmentDownloadRequest(&self, _offset: u64, _size: u64) -> WResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::fonts::FontSource;

    // Read a font file through the public loader/stream chain, exactly as `IDWriteFontFace::GetFiles`'
    // consumer (our d2d-ffi, or a foreign app) does: GetReferenceKey → GetLoader → CreateStreamFromKey
    // → ReadFileFragment. Returns the bytes the stream serves.
    fn read_back(file: &IDWriteFontFile) -> Vec<u8> {
        unsafe {
            let mut key: *mut core::ffi::c_void = core::ptr::null_mut();
            let mut ks = 0u32;
            file.GetReferenceKey(&mut key, &mut ks).unwrap();
            let loader = file.GetLoader().unwrap();
            let stream = loader.CreateStreamFromKey(key as *const core::ffi::c_void, ks).unwrap();
            let size = stream.GetFileSize().unwrap();
            let mut frag: *mut core::ffi::c_void = core::ptr::null_mut();
            let mut ctx: *mut core::ffi::c_void = core::ptr::null_mut();
            stream.ReadFileFragment(&mut frag, 0, size, &mut ctx).unwrap();
            let b = slice::from_raw_parts(frag as *const u8, size as usize).to_vec();
            stream.ReleaseFileFragment(ctx);
            b
        }
    }

    fn face_files(face: &IDWriteFontFace) -> IDWriteFontFile {
        unsafe {
            let mut n = 0u32;
            face.GetFiles(&mut n, None).unwrap();
            assert_eq!(n, 1);
            let mut files: Vec<Option<IDWriteFontFile>> = vec![None; 1];
            face.GetFiles(&mut n, Some(files.as_mut_ptr())).unwrap();
            files.into_iter().flatten().next().unwrap()
        }
    }

    // A memory-backed face's GetFiles round-trips through the in-memory loader (no path needed).
    #[test]
    fn memory_face_getfiles_roundtrips() {
        let bytes = Arc::new(b"not-a-real-font-but-the-loader-serves-it-verbatim".to_vec());
        let face: IDWriteFontFace7 = FaceObj {
            bytes: bytes.clone(),
            index: 0,
            family: "Test".into(),
            source: FontSource::Bytes(bytes.clone()),
            sims: 0,
            axis_values: Vec::new(),
        }
        .into();
        let file = face_files(&face.cast().unwrap());
        assert_eq!(read_back(&file), *bytes, "in-memory GetFiles must serve the face's bytes");
    }

    // A file-backed face's GetFiles hands back a path-keyed local-loader file that reads the font from
    // disk — the key round-trips to the path and the stream serves the file's bytes.
    #[test]
    fn file_face_getfiles_roundtrips() {
        let dir = std::env::temp_dir();
        let path = dir.join("dwrite_getfiles_roundtrip.bin");
        let data = b"on-disk font bytes".to_vec();
        std::fs::write(&path, &data).unwrap();
        let face: IDWriteFontFace7 = FaceObj {
            bytes: Arc::new(data.clone()),
            index: 0,
            family: "Test".into(),
            source: FontSource::Path(path.clone()),
            sims: 0,
            axis_values: Vec::new(),
        }
        .into();
        let file = face_files(&face.cast().unwrap());
        // The reference key must decode back to the font's path (native's file-backed form).
        let f = file.cast_object_ref::<FontFileObj>().unwrap();
        assert!(f.local, "a file-backed face's file must use the local loader");
        assert_eq!(read_back(&file), data, "local-loader GetFiles must read the font from disk");
        let _ = std::fs::remove_file(&path);
    }

    // WOFF2 unpack: the fixture is `woff2_compress`(colrtest.ttf), so its glyf/loca are transformed.
    // It must decode to an sfnt skrifa parses, with every glyph's outline identical to the original.
    #[test]
    fn woff2_unpacks_to_sfnt_with_identical_outlines() {
        use skrifa::instance::{LocationRef, Size};
        use skrifa::outline::{DrawSettings, OutlinePen};
        use skrifa::raw::TableProvider;
        use skrifa::{FontRef, GlyphId, MetadataProvider};

        #[derive(Default, PartialEq, Debug)]
        struct Rec(Vec<String>);
        impl OutlinePen for Rec {
            fn move_to(&mut self, x: f32, y: f32) {
                self.0.push(format!("M {x} {y}"));
            }
            fn line_to(&mut self, x: f32, y: f32) {
                self.0.push(format!("L {x} {y}"));
            }
            fn quad_to(&mut self, a: f32, b: f32, x: f32, y: f32) {
                self.0.push(format!("Q {a} {b} {x} {y}"));
            }
            fn curve_to(&mut self, a: f32, b: f32, c: f32, d: f32, x: f32, y: f32) {
                self.0.push(format!("C {a} {b} {c} {d} {x} {y}"));
            }
            fn close(&mut self) {
                self.0.push("Z".into());
            }
        }
        fn outline(font: &FontRef, g: u16) -> Rec {
            let mut r = Rec::default();
            if let Some(gl) = font.outline_glyphs().get(GlyphId::new(g as u32)) {
                let _ = gl.draw(DrawSettings::unhinted(Size::unscaled(), LocationRef::default()), &mut r);
            }
            r
        }

        let woff2 = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/colrtest.woff2"));
        let ttf = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../d2d/src/text/color/testdata/colrtest.ttf"));
        assert_eq!(analyze_container(woff2), DWRITE_CONTAINER_TYPE_WOFF2);
        let sfnt = woff2_to_sfnt(woff2).unwrap();
        let out = FontRef::new(&sfnt).unwrap();
        let orig = FontRef::new(ttf.as_slice()).unwrap();
        let n = orig.maxp().unwrap().num_glyphs();
        assert!(n > 1, "fixture should have several glyphs");
        for g in 0..n {
            assert_eq!(outline(&out, g), outline(&orig, g), "glyph {g} outline differs after WOFF2 unpack");
        }
    }
}
