//! Small value-holder COM objects a text client creates and hands back to layout/analysis:
//! `CreateTypography` (an OpenType feature list) and `CreateNumberSubstitution` (an opaque number
//! substitution config — the interface has no methods; layout reads it via QueryInterface identity).

use super::{E_INVALIDARG, write_wide};
use std::cell::RefCell;
use windows::Win32::Graphics::DirectWrite::*;
use windows_core::{BOOL, PWSTR, Ref, Result as WResult, implement};

#[implement(IDWriteTypography)]
pub(crate) struct TypographyObj {
    features: RefCell<Vec<DWRITE_FONT_FEATURE>>,
}

pub(crate) fn typography() -> IDWriteTypography {
    TypographyObj { features: RefCell::new(Vec::new()) }.into()
}

impl IDWriteTypography_Impl for TypographyObj_Impl {
    fn AddFontFeature(&self, fontfeature: &DWRITE_FONT_FEATURE) -> WResult<()> {
        self.features.borrow_mut().push(*fontfeature);
        Ok(())
    }
    fn GetFontFeatureCount(&self) -> u32 {
        self.features.borrow().len() as u32
    }
    fn GetFontFeature(&self, fontfeatureindex: u32) -> WResult<DWRITE_FONT_FEATURE> {
        self.features.borrow().get(fontfeatureindex as usize).copied().ok_or(E_INVALIDARG.into())
    }
}

#[implement(IDWriteNumberSubstitution)]
pub(crate) struct NumberSubstitutionObj;

pub(crate) fn number_substitution() -> IDWriteNumberSubstitution {
    NumberSubstitutionObj.into()
}

impl IDWriteNumberSubstitution_Impl for NumberSubstitutionObj_Impl {}

/// The font download queue. Every font this collection serves is a local file, so the queue is
/// always empty and downloads are no-ops — matching native's local-font behaviour (`IsEmpty` TRUE,
/// `BeginDownload` succeeds with nothing to do).
#[implement(IDWriteFontDownloadQueue)]
pub(crate) struct DownloadQueueObj {
    next_token: core::cell::Cell<u32>,
}

pub(crate) fn download_queue() -> IDWriteFontDownloadQueue {
    DownloadQueueObj { next_token: core::cell::Cell::new(0) }.into()
}

impl IDWriteFontDownloadQueue_Impl for DownloadQueueObj_Impl {
    fn AddListener(&self, _listener: Ref<IDWriteFontDownloadListener>) -> WResult<u32> {
        // Native hands out 1-based tokens (verified: the first listener's token is 1); the queue is
        // always empty so the listener never fires, but the token must be non-zero.
        let token = self.next_token.get() + 1;
        self.next_token.set(token);
        Ok(token)
    }
    fn RemoveListener(&self, _token: u32) -> WResult<()> {
        Ok(())
    }
    fn IsEmpty(&self) -> BOOL {
        true.into()
    }
    fn BeginDownload(&self, _context: Ref<windows_core::IUnknown>) -> WResult<()> {
        Ok(())
    }
    fn CancelDownload(&self) -> WResult<()> {
        Ok(())
    }
    fn GetGenerationCount(&self) -> u64 {
        0
    }
}

/// An `IDWriteStringList` of English strings (font-set property values). Each entry is one string
/// with locale "en-us".
#[implement(IDWriteStringList)]
pub(crate) struct StringListObj {
    values: Vec<String>,
}

pub(crate) fn string_list(values: Vec<String>) -> IDWriteStringList {
    StringListObj { values }.into()
}

impl IDWriteStringList_Impl for StringListObj_Impl {
    fn GetCount(&self) -> u32 {
        self.values.len() as u32
    }
    fn GetLocaleNameLength(&self, _index: u32) -> WResult<u32> {
        Ok(5) // "en-us"
    }
    fn GetLocaleName(&self, _index: u32, name: PWSTR, size: u32) -> WResult<()> {
        write_wide("en-us", name, size)
    }
    fn GetStringLength(&self, index: u32) -> WResult<u32> {
        Ok(self.values.get(index as usize).map(|s| s.encode_utf16().count() as u32).unwrap_or(0))
    }
    fn GetString(&self, index: u32, buffer: PWSTR, size: u32) -> WResult<()> {
        write_wide(self.values.get(index as usize).map(String::as_str).unwrap_or(""), buffer, size)
    }
}
