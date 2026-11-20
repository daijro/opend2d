//! Font sets and the font-set builder — `GetSystemFontSet` / `CreateFontSetBuilder` /
//! `CreateFontSet`. A font set is a flat list of `(source, faceIndex)` members; each yields the same
//! [`FaceRefObj`]/[`FaceObj`] the rest of the surface uses, so a set member renders identically.
//!
//! ponytail: the enumeration + build path (GetFontCount / GetFontFaceReference / AddFontFaceReference
//! / CreateFontSet) is implemented; the property-index queries (GetPropertyValues*,
//! Find*, GetMatchingFonts) need a per-property name index that no d2d::text path feeds yet, so they
//! report empty/not-found — add the index when a font-set property-query scene exists.

use super::fontfile::FaceRefObj;
use super::fonts::{FontSource, localized};
use super::misc::string_list;
use super::{E_INVALIDARG, E_NOTIMPL, wide_to_string, write_wide};
use skrifa::string::StringId;
use skrifa::{FontRef, MetadataProvider};
use std::cell::RefCell;
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::DirectWrite::*;
use windows_core::{BOOL, Interface, PCWSTR, PWSTR, Ref, Result as WResult, implement};

/// The skrifa name id behind a `DWRITE_FONT_PROPERTY_ID` (the ones a font set indexes), or `None`.
fn name_id(id: DWRITE_FONT_PROPERTY_ID) -> Option<StringId> {
    Some(match id {
        DWRITE_FONT_PROPERTY_ID_FAMILY_NAME | DWRITE_FONT_PROPERTY_ID_WIN32_FAMILY_NAME => StringId::FAMILY_NAME,
        DWRITE_FONT_PROPERTY_ID_TYPOGRAPHIC_FAMILY_NAME => StringId::TYPOGRAPHIC_FAMILY_NAME,
        DWRITE_FONT_PROPERTY_ID_FACE_NAME => StringId::SUBFAMILY_NAME,
        DWRITE_FONT_PROPERTY_ID_FULL_NAME => StringId::FULL_NAME,
        DWRITE_FONT_PROPERTY_ID_POSTSCRIPT_NAME => StringId::POSTSCRIPT_NAME,
        _ => return None,
    })
}

/// A font-set member: its source + face index, plus how its properties are served —
/// `None` synthesizes them from the font's name tables (the system/collection sets, which native
/// indexes exhaustively), `Some(props)` serves only the explicit `(property-id, value)` pairs the
/// set was built with (a builder set via AddFont/AddFontFaceReference — native indexes nothing more,
/// so an AddFont set with no properties reports none). The choice travels with each member through
/// subset `clone`s, so filtered sub-sets inherit it automatically.
pub(crate) type Member = (FontSource, u32, Option<Vec<(u32, String)>>);

#[implement(IDWriteFontSet4)]
pub(crate) struct FontSetObj {
    faces: Vec<Member>,
}

/// A font set as the most-derived `IDWriteFontSet4` leaf, so the QI chain exposes FontSet1/2/3/4.
fn set_obj(faces: Vec<Member>) -> IDWriteFontSet4 {
    FontSetObj { faces }.into()
}

/// A synthesized set (system / collection fonts): every member reports name-table properties.
pub(crate) fn font_set(faces: Vec<(FontSource, u32)>) -> IDWriteFontSet {
    set_obj(faces.into_iter().map(|(s, f)| (s, f, None)).collect()).cast().unwrap()
}

/// A built set whose members carry only their explicit properties (the font-set builder).
pub(crate) fn font_set_explicit(faces: Vec<Member>) -> IDWriteFontSet {
    set_obj(faces).cast().unwrap()
}

impl IDWriteFontSet_Impl for FontSetObj_Impl {
    fn GetFontCount(&self) -> u32 {
        self.faces.len() as u32
    }
    fn GetFontFaceReference(&self, index: u32) -> WResult<IDWriteFontFaceReference> {
        let (src, face, _) = self.faces.get(index as usize).ok_or(E_NOTIMPL)?;
        let bytes = src.read().ok_or(E_NOTIMPL)?;
        Ok(super::fontfile::faceref(bytes, *face, DWRITE_FONT_SIMULATIONS_NONE, src.path_str()))
    }
    fn FindFontFaceReference(&self, r: Ref<IDWriteFontFaceReference>, listindex: *mut u32, exists: *mut BOOL) -> WResult<()> {
        let want = r.as_ref().and_then(|r| r.cast_object_ref::<FaceRefObj>().ok());
        let hit = want.and_then(|w| self.index_of(&w.bytes, w.index));
        unsafe {
            *listindex = hit.unwrap_or(0) as u32;
            *exists = hit.is_some().into();
        }
        Ok(())
    }
    fn FindFontFace(&self, f: Ref<IDWriteFontFace>, listindex: *mut u32, exists: *mut BOOL) -> WResult<()> {
        // Match the face's font bytes + face index against the set's members (our faces carry both,
        // like a face reference). Native finds an in-set face (VM-verified: exists, index 0).
        let hit = f.as_ref()
            .and_then(|f| f.cast_object_ref::<super::fonts::FaceObj>().ok())
            .and_then(|fo| self.index_of(&fo.bytes, fo.index));
        unsafe {
            *listindex = hit.unwrap_or(0) as u32;
            *exists = hit.is_some().into();
        }
        Ok(())
    }
    fn GetPropertyValues(&self, id: DWRITE_FONT_PROPERTY_ID) -> WResult<IDWriteStringList> {
        Ok(string_list(self.distinct(id)))
    }
    fn GetPropertyValues2(&self, id: DWRITE_FONT_PROPERTY_ID, _locales: &PCWSTR) -> WResult<IDWriteStringList> {
        Ok(string_list(self.distinct(id)))
    }
    fn GetPropertyValues3(&self, index: u32, id: DWRITE_FONT_PROPERTY_ID, exists: *mut BOOL, values: windows_core::OutRef<IDWriteLocalizedStrings>) -> WResult<()> {
        match self.prop(index as usize, id) {
            Some(v) => {
                unsafe { *exists = true.into() };
                values.write(Some(localized(v)))
            }
            None => {
                unsafe { *exists = false.into() };
                Ok(())
            }
        }
    }
    fn GetPropertyOccurrenceCount(&self, property: *const DWRITE_FONT_PROPERTY) -> WResult<u32> {
        let p = unsafe { &*property };
        let want = wide_to_string(p.propertyValue.0, None);
        let n = (0..self.faces.len()).filter(|&i| self.prop(i, p.propertyId).as_deref() == Some(want.as_str())).count();
        Ok(n as u32)
    }
    fn GetMatchingFonts(&self, _familyname: &PCWSTR, _weight: DWRITE_FONT_WEIGHT, _stretch: DWRITE_FONT_STRETCH, _style: DWRITE_FONT_STYLE) -> WResult<IDWriteFontSet> {
        // Property-matched sub-sets aren't reproduced (the font set is a flat member list); return
        // the whole set so the caller can still enumerate.
        Ok(font_set_explicit(self.faces.clone()))
    }
    fn GetMatchingFonts2(&self, _properties: *const DWRITE_FONT_PROPERTY, _count: u32) -> WResult<IDWriteFontSet> {
        Ok(font_set_explicit(self.faces.clone()))
    }
}

impl FontSetObj {
    /// One member's property value (`None` for an unindexed property). A synthesized member
    /// (`explicit == None`) reads the name from its font tables; an explicit member serves only the
    /// `(id, value)` pairs it was built with — native's indexed-store model.
    fn prop(&self, index: usize, id: DWRITE_FONT_PROPERTY_ID) -> Option<String> {
        let (src, face, explicit) = self.faces.get(index)?;
        if let Some(props) = explicit {
            return props.iter().find(|(pid, _)| *pid == id.0 as u32).map(|(_, v)| v.clone());
        }
        let sid = name_id(id)?;
        let bytes = src.read()?;
        let font = FontRef::from_index(&bytes, *face).ok()?;
        font.localized_strings(sid).english_or_first().map(|s| s.to_string())
    }
    /// The set's distinct values for `id`, in member order.
    fn distinct(&self, id: DWRITE_FONT_PROPERTY_ID) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for i in 0..self.faces.len() {
            if let Some(v) = self.prop(i, id) {
                if !out.contains(&v) {
                    out.push(v);
                }
            }
        }
        out
    }
    /// The first member whose bytes + face index match (content equality), for `FindFontFaceReference`.
    fn index_of(&self, bytes: &[u8], face: u32) -> Option<usize> {
        (0..self.faces.len()).find(|&i| self.faces[i].1 == face && self.faces[i].0.read().as_deref().map(Vec::as_slice) == Some(bytes))
    }

    /// A member read into `(bytes, faceIndex, source)`, or `E_NOTIMPL` if the index is out of range or
    /// its bytes cannot be read — the member accessors (`CreateFontFace`/`GetFontFaceReference`/…).
    fn member(&self, index: u32) -> WResult<(std::sync::Arc<Vec<u8>>, u32, FontSource)> {
        let (src, face, _) = self.faces.get(index as usize).ok_or(E_NOTIMPL)?;
        Ok((src.read().ok_or(E_NOTIMPL)?, *face, src.clone()))
    }
    /// A member's `fvar` axis ranges (empty for a static face).
    fn member_ranges(&self, index: u32) -> Vec<DWRITE_FONT_AXIS_RANGE> {
        self.faces
            .get(index as usize)
            .and_then(|(src, face, _)| {
                let bytes = src.read()?;
                let font = FontRef::from_index(&bytes, *face).ok()?;
                let fvar = super::fonts::fvar_axes(&font);
                if !fvar.is_empty() {
                    return Some(fvar.iter().map(|a| DWRITE_FONT_AXIS_RANGE { axisTag: a.tag, minValue: a.min, maxValue: a.max }).collect());
                }
                // A static font exposes the four standard axes (wght/wdth/ital/slnt) as degenerate
                // ranges, matching native GetFontAxisRanges/GetFontAxisRanges2 (VM-verified: bmtest →
                // wght 400, wdth 100, ital 0, slnt 0).
                Some(super::fonts::face_axis_values(&font).iter().map(|v| DWRITE_FONT_AXIS_RANGE { axisTag: v.axisTag, minValue: v.value, maxValue: v.value }).collect())
            })
            .unwrap_or_default()
    }
    /// The union of every member's axis ranges, widening `[min,max]` per tag (`GetFontAxisRanges2`).
    fn union_ranges(&self) -> Vec<DWRITE_FONT_AXIS_RANGE> {
        let mut out: Vec<DWRITE_FONT_AXIS_RANGE> = Vec::new();
        for i in 0..self.faces.len() as u32 {
            for r in self.member_ranges(i) {
                match out.iter_mut().find(|o| o.axisTag == r.axisTag) {
                    Some(o) => {
                        o.minValue = o.minValue.min(r.minValue);
                        o.maxValue = o.maxValue.max(r.maxValue);
                    }
                    None => out.push(r),
                }
            }
        }
        out
    }
    /// Members deduplicated to the first face of each distinct font resource (file), for
    /// `GetFirstFontResources`.
    fn first_resources(&self) -> Vec<Member> {
        let mut out: Vec<Member> = Vec::new();
        for m in &self.faces {
            if !out.iter().any(|(s, _, _)| same_resource(s, &m.0)) {
                out.push(m.clone());
            }
        }
        out
    }
    /// Does member `index` carry an axis matching every (or, with `any`, any) of `ranges` — a tag whose
    /// `[min,max]` overlaps the requested range? Empty `ranges` matches everything.
    fn ranges_match(&self, index: u32, ranges: &[DWRITE_FONT_AXIS_RANGE], any: BOOL) -> bool {
        if ranges.is_empty() {
            return true;
        }
        let have = self.member_ranges(index);
        let one = |r: &DWRITE_FONT_AXIS_RANGE| have.iter().any(|h| h.axisTag == r.axisTag && h.minValue <= r.maxValue && h.maxValue >= r.minValue);
        if any.as_bool() { ranges.iter().any(one) } else { ranges.iter().all(one) }
    }
    /// Does member `index` match every (or, with `any`, any) of `props` by value? Empty matches all.
    fn props_match(&self, index: usize, props: &[DWRITE_FONT_PROPERTY], any: BOOL) -> bool {
        if props.is_empty() {
            return true;
        }
        let one = |p: &DWRITE_FONT_PROPERTY| self.prop(index, p.propertyId).as_deref() == Some(wide_to_string(p.propertyValue.0, None).as_str());
        if any.as_bool() { props.iter().any(one) } else { props.iter().all(one) }
    }
    /// The members matching a single optional name property (null → all), for `FontSet1::GetMatchingFonts`.
    fn filter_by_property(&self, property: *const DWRITE_FONT_PROPERTY) -> Vec<Member> {
        if property.is_null() {
            return self.faces.clone();
        }
        let p = unsafe { &*property };
        (0..self.faces.len()).filter(|&i| self.props_match(i, std::slice::from_ref(p), BOOL(0))).map(|i| self.faces[i].clone()).collect()
    }
    /// The members whose family name equals `family` (`FontSet4::GetMatchingFonts`); empty name → all.
    fn filter_by_family(&self, family: &PCWSTR) -> Vec<Member> {
        let want = wide_to_string(family.0, None);
        if want.is_empty() {
            return self.faces.clone();
        }
        (0..self.faces.len()).filter(|&i| self.prop(i, DWRITE_FONT_PROPERTY_ID_FAMILY_NAME).as_deref() == Some(want.as_str())).map(|i| self.faces[i].clone()).collect()
    }
}

/// Two members share a font resource when they come from the same file (by path), or the same bytes.
fn same_resource(a: &FontSource, b: &FontSource) -> bool {
    match (a.path_str(), b.path_str()) {
        (Some(pa), Some(pb)) => pa == pb,
        (None, None) => a.read().as_deref() == b.read().as_deref(),
        _ => false,
    }
}

/// A caller `*const T`/`count` as a slice (empty when null or zero).
unsafe fn slice_or_empty<'a, T>(ptr: *const T, count: u32) -> &'a [T] {
    if ptr.is_null() || count == 0 { &[] } else { unsafe { std::slice::from_raw_parts(ptr, count as usize) } }
}

/// Fill a caller `u32[max]` buffer with `vals`, `*actual` = the full count (the `Get*Indices`
/// truncation protocol: a caller sizes the buffer from `*actual` and calls again).
fn write_indices(vals: &[u32], out: *mut u32, max: u32, actual: *mut u32) -> WResult<()> {
    unsafe { *actual = vals.len() as u32 };
    let n = vals.len().min(max as usize);
    if !out.is_null() && n > 0 {
        unsafe { std::slice::from_raw_parts_mut(out, n) }.copy_from_slice(&vals[..n]);
    }
    Ok(())
}

/// Fill a caller `DWRITE_FONT_AXIS_RANGE[max]` buffer with `ranges`, `*actual` = the full count.
fn write_ranges(ranges: &[DWRITE_FONT_AXIS_RANGE], out: *mut DWRITE_FONT_AXIS_RANGE, max: u32, actual: *mut u32) -> WResult<()> {
    unsafe { *actual = ranges.len() as u32 };
    let n = ranges.len().min(max as usize);
    if !out.is_null() && n > 0 {
        unsafe { std::slice::from_raw_parts_mut(out, n) }.copy_from_slice(&ranges[..n]);
    }
    Ok(())
}

impl IDWriteFontSet1_Impl for FontSetObj_Impl {
    fn GetMatchingFonts(&self, fontproperty: *const DWRITE_FONT_PROPERTY, _fontaxisvalues: *const DWRITE_FONT_AXIS_VALUE, _fontaxisvaluecount: u32) -> WResult<IDWriteFontSet1> {
        // ponytail: filter by the name property when given; axis-value specialisation needs instancing
        // (not done), so axis values do not narrow the result. Null property → whole set.
        Ok(set_obj(self.filter_by_property(fontproperty)).cast()?)
    }
    fn GetFirstFontResources(&self) -> WResult<IDWriteFontSet1> {
        Ok(set_obj(self.first_resources()).cast()?)
    }
    fn GetFilteredFonts(&self, indices: *const u32, indexcount: u32) -> WResult<IDWriteFontSet1> {
        let idx = unsafe { slice_or_empty(indices, indexcount) };
        let faces = idx.iter().filter_map(|&i| self.faces.get(i as usize).cloned()).collect();
        Ok(set_obj(faces).cast()?)
    }
    fn GetFilteredFonts2(&self, fontaxisranges: *const DWRITE_FONT_AXIS_RANGE, fontaxisrangecount: u32, selectanyrange: BOOL) -> WResult<IDWriteFontSet1> {
        let ranges = unsafe { slice_or_empty(fontaxisranges, fontaxisrangecount) };
        let faces = (0..self.faces.len() as u32).filter(|&i| self.ranges_match(i, ranges, selectanyrange)).filter_map(|i| self.faces.get(i as usize).cloned()).collect();
        Ok(set_obj(faces).cast()?)
    }
    fn GetFilteredFonts3(&self, properties: *const DWRITE_FONT_PROPERTY, propertycount: u32, selectanyproperty: BOOL) -> WResult<IDWriteFontSet1> {
        let props = unsafe { slice_or_empty(properties, propertycount) };
        let faces = (0..self.faces.len()).filter(|&i| self.props_match(i, props, selectanyproperty)).map(|i| self.faces[i].clone()).collect();
        Ok(set_obj(faces).cast()?)
    }
    fn GetFilteredFontIndices(&self, fontaxisranges: *const DWRITE_FONT_AXIS_RANGE, fontaxisrangecount: u32, selectanyrange: BOOL, indices: *mut u32, maxindexcount: u32, actualindexcount: *mut u32) -> WResult<()> {
        let ranges = unsafe { slice_or_empty(fontaxisranges, fontaxisrangecount) };
        let matched: Vec<u32> = (0..self.faces.len() as u32).filter(|&i| self.ranges_match(i, ranges, selectanyrange)).collect();
        write_indices(&matched, indices, maxindexcount, actualindexcount)
    }
    fn GetFilteredFontIndices2(&self, properties: *const DWRITE_FONT_PROPERTY, propertycount: u32, selectanyproperty: BOOL, indices: *mut u32, maxindexcount: u32, actualindexcount: *mut u32) -> WResult<()> {
        let props = unsafe { slice_or_empty(properties, propertycount) };
        let matched: Vec<u32> = (0..self.faces.len() as u32).filter(|&i| self.props_match(i as usize, props, selectanyproperty)).collect();
        write_indices(&matched, indices, maxindexcount, actualindexcount)
    }
    fn GetFontAxisRanges(&self, listindex: u32, fontaxisranges: *mut DWRITE_FONT_AXIS_RANGE, maxfontaxisrangecount: u32, actualfontaxisrangecount: *mut u32) -> WResult<()> {
        write_ranges(&self.member_ranges(listindex), fontaxisranges, maxfontaxisrangecount, actualfontaxisrangecount)
    }
    fn GetFontAxisRanges2(&self, fontaxisranges: *mut DWRITE_FONT_AXIS_RANGE, maxfontaxisrangecount: u32, actualfontaxisrangecount: *mut u32) -> WResult<()> {
        write_ranges(&self.union_ranges(), fontaxisranges, maxfontaxisrangecount, actualfontaxisrangecount)
    }
    fn GetFontFaceReference(&self, listindex: u32) -> WResult<IDWriteFontFaceReference1> {
        let (bytes, face, src) = self.member(listindex)?;
        Ok(super::fontfile::faceref(bytes, face, DWRITE_FONT_SIMULATIONS_NONE, src.path_str()).cast()?)
    }
    fn CreateFontResource(&self, listindex: u32) -> WResult<IDWriteFontResource> {
        let (bytes, face, src) = self.member(listindex)?;
        Ok(super::fonts::FontResourceObj { bytes, index: face, source: src }.into())
    }
    fn CreateFontFace(&self, listindex: u32) -> WResult<IDWriteFontFace5> {
        let (bytes, face, src) = self.member(listindex)?;
        if FontRef::from_index(&bytes, face).is_err() {
            return Err(E_INVALIDARG.into());
        }
        let f: IDWriteFontFace7 = super::fonts::FaceObj { bytes: bytes.clone(), index: face, family: super::fontfile::family_name(&bytes, face), source: src, sims: 0, axis_values: Vec::new() }.into();
        Ok(f.cast()?)
    }
    fn GetFontLocality(&self, _listindex: u32) -> DWRITE_LOCALITY {
        // Every member of our sets is a local, resident font.
        DWRITE_LOCALITY_LOCAL
    }
}

impl IDWriteFontSet2_Impl for FontSetObj_Impl {
    fn GetExpirationEvent(&self) -> HANDLE {
        super::fonts::expiration_event()
    }
}

impl IDWriteFontSet3_Impl for FontSetObj_Impl {
    fn GetFontSourceType(&self, fontindex: u32) -> DWRITE_FONT_SOURCE_TYPE {
        // ponytail: a file-backed member is treated as an installed (per-machine) font; a memory
        // member has no source type. Per-machine vs per-user is VM-pending.
        match self.faces.get(fontindex as usize) {
            Some((FontSource::Path(_), _, _)) => DWRITE_FONT_SOURCE_TYPE_PER_MACHINE,
            _ => DWRITE_FONT_SOURCE_TYPE_UNKNOWN,
        }
    }
    fn GetFontSourceNameLength(&self, listindex: u32) -> u32 {
        self.faces.get(listindex as usize).and_then(|(s, _, _)| s.path_str()).map(|p| p.encode_utf16().count() as u32).unwrap_or(0)
    }
    fn GetFontSourceName(&self, listindex: u32, stringbuffer: PWSTR, stringbuffersize: u32) -> WResult<()> {
        let name = self.faces.get(listindex as usize).and_then(|(s, _, _)| s.path_str()).unwrap_or_default();
        write_wide(&name, stringbuffer, stringbuffersize)
    }
}

impl IDWriteFontSet4_Impl for FontSetObj_Impl {
    fn ConvertWeightStretchStyleToFontAxisValues(&self, inputaxisvalues: *const DWRITE_FONT_AXIS_VALUE, inputaxiscount: u32, fontweight: DWRITE_FONT_WEIGHT, fontstretch: DWRITE_FONT_STRETCH, fontstyle: DWRITE_FONT_STYLE, fontsize: f32, outputaxisvalues: *mut DWRITE_FONT_AXIS_VALUE) -> u32 {
        let input = unsafe { slice_or_empty(inputaxisvalues, inputaxiscount) };
        let out = super::fonts::convert_wss_axis_values(input, fontweight, fontstretch, fontstyle, fontsize);
        // Native's output never exceeds the five standard axes (the client buffer is `[_; 5]`).
        let n = out.len().min(5);
        if !outputaxisvalues.is_null() && n > 0 {
            unsafe { std::slice::from_raw_parts_mut(outputaxisvalues, n) }.copy_from_slice(&out[..n]);
        }
        n as u32
    }
    fn GetMatchingFonts(&self, familyname: &PCWSTR, _fontaxisvalues: *const DWRITE_FONT_AXIS_VALUE, _fontaxisvaluecount: u32, _allowedsimulations: DWRITE_FONT_SIMULATIONS) -> WResult<IDWriteFontSet4> {
        Ok(set_obj(self.filter_by_family(familyname)).cast()?)
    }
}

// --- IDWriteFontSetBuilder ---------------------------------------------------------------------

#[implement(IDWriteFontSetBuilder2)]
pub(crate) struct FontSetBuilderObj {
    faces: RefCell<Vec<Member>>,
}

pub(crate) fn builder() -> IDWriteFontSetBuilder {
    let b: IDWriteFontSetBuilder2 = FontSetBuilderObj { faces: RefCell::new(Vec::new()) }.into();
    b.cast().unwrap()
}

/// The explicit `(property-id, value)` pairs an `AddFont`/`AddFontFaceReference` call supplies; a
/// built set serves only these (never the font's name tables), as native's indexed store does.
fn explicit_props(properties: *const DWRITE_FONT_PROPERTY, count: u32) -> Vec<(u32, String)> {
    unsafe { slice_or_empty(properties, count) }
        .iter()
        .map(|p| (p.propertyId.0 as u32, wide_to_string(p.propertyValue.0, None)))
        .collect()
}

impl FontSetBuilderObj {
    /// Take the in-memory bytes + face index a face reference carries (ours are [`FaceRefObj`]), to
    /// store as a `Bytes` source so the built set never needs the original file again.
    fn push(&self, r: Ref<IDWriteFontFaceReference>, props: Vec<(u32, String)>) -> WResult<()> {
        let fr = r.as_ref().ok_or(E_NOTIMPL)?.cast_object_ref::<FaceRefObj>().map_err(|_| windows_core::Error::from(E_NOTIMPL))?;
        self.faces.borrow_mut().push((fr.source(), fr.index, Some(props)));
        Ok(())
    }
    /// Add every face of a font file to the set (`IDWriteFontSetBuilder1::AddFontFile`), reading the
    /// bytes once so the set is self-contained. No properties are indexed (native indexes none here).
    fn push_file(&self, file: &IDWriteFontFile) -> WResult<()> {
        let bytes = super::fontfile::read_font_file(file)?;
        let source = file.cast_object_ref::<super::fontfile::FontFileObj>().map(|f| f.as_source()).unwrap_or_else(|_| FontSource::Bytes(bytes.clone()));
        let mut faces = 0u32;
        while FontRef::from_index(&bytes, faces).is_ok() {
            self.faces.borrow_mut().push((source.clone(), faces, Some(Vec::new())));
            faces += 1;
        }
        if faces == 0 { Err(E_NOTIMPL.into()) } else { Ok(()) }
    }
}

impl IDWriteFontSetBuilder_Impl for FontSetBuilderObj_Impl {
    fn AddFontFaceReference(&self, r: Ref<IDWriteFontFaceReference>, properties: *const DWRITE_FONT_PROPERTY, count: u32) -> WResult<()> {
        self.push(r, explicit_props(properties, count))
    }
    fn AddFontFaceReference2(&self, r: Ref<IDWriteFontFaceReference>) -> WResult<()> {
        self.push(r, Vec::new())
    }
    fn AddFontSet(&self, set: Ref<IDWriteFontSet>) -> WResult<()> {
        let set = set.as_ref().ok_or(E_NOTIMPL)?.cast_object_ref::<FontSetObj>().map_err(|_| windows_core::Error::from(E_NOTIMPL))?;
        self.faces.borrow_mut().extend(set.faces.iter().cloned());
        Ok(())
    }
    fn CreateFontSet(&self) -> WResult<IDWriteFontSet> {
        Ok(font_set_explicit(self.faces.borrow().clone()))
    }
}

impl IDWriteFontSetBuilder1_Impl for FontSetBuilderObj_Impl {
    fn AddFontFile(&self, fontfile: Ref<IDWriteFontFile>) -> WResult<()> {
        self.push_file(fontfile.as_ref().ok_or(E_NOTIMPL)?)
    }
}

impl IDWriteFontSetBuilder2_Impl for FontSetBuilderObj_Impl {
    fn AddFont(&self, fontfile: Ref<IDWriteFontFile>, fontfaceindex: u32, _fontsimulations: DWRITE_FONT_SIMULATIONS, _fontaxisvalues: *const DWRITE_FONT_AXIS_VALUE, _fontaxisvaluecount: u32, _fontaxisranges: *const DWRITE_FONT_AXIS_RANGE, _fontaxisrangecount: u32, properties: *const DWRITE_FONT_PROPERTY, propertycount: u32) -> WResult<()> {
        // The member stores its explicit properties (and source + face index); simulations / axis
        // values / ranges are not reproduced per member (no instancing).
        let file = fontfile.as_ref().ok_or(E_NOTIMPL)?;
        let bytes = super::fontfile::read_font_file(file)?;
        let source = file.cast_object_ref::<super::fontfile::FontFileObj>().map(|f| f.as_source()).unwrap_or_else(|_| FontSource::Bytes(bytes));
        self.faces.borrow_mut().push((source, fontfaceindex, Some(explicit_props(properties, propertycount))));
        Ok(())
    }
    fn AddFontFile(&self, filepath: &PCWSTR) -> WResult<()> {
        let file = super::fontfile::file_from_path(&wide_to_string(filepath.0, None))?;
        self.push_file(&file)
    }
}

/// The `(source, faceIndex)` members of one of our font sets — for `CreateFontCollectionFromFontSet`.
pub(crate) fn set_faces(set: &IDWriteFontSet) -> Option<Vec<(FontSource, u32)>> {
    set.cast_object_ref::<FontSetObj>().ok().map(|s| s.faces.iter().map(|(src, face, _)| (src.clone(), *face)).collect())
}
