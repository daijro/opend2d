//! The system font collection and its font/family/face objects, parsed with `skrifa` (upem, cmap,
//! design advances, family name, weight/style) from the platform font directory. These feed the
//! harness's `GetGlyphIndices`/`GetDesignGlyphMetrics` (the inputs it packs into a `DrawGlyphRun`)
//! and, through [`ITextInterop`], the font bytes our `d2d1.dll` renders.

use super::{E_INVALIDARG, E_NOTIMPL, ITextInterop, ITextInterop_Impl, wide_to_string, write_wide};
use core::cell::RefCell;
use core::slice;
use d2d::text::{self, Glyph as TGlyph, GlyphRun as TRun, RenderingParams, fallback};
use std::collections::HashMap;
use skrifa::bitmap::{BitmapData, BitmapFormat, BitmapStrikes};
use skrifa::instance::{LocationRef, Size};
use skrifa::raw::tables::kern::{Kern, SubtableKind};
use skrifa::raw::tables::os2::SelectionFlags;
use skrifa::raw::tables::stat::AxisValue;
use skrifa::raw::{FontRead, TableProvider, types::Tag};
use skrifa::string::StringId;
use skrifa::{FontRef, GlyphId, MetadataProvider};
use std::path::PathBuf;
use std::sync::Arc;
use windows::Win32::Foundation::{HANDLE, POINT};
use windows::Win32::Graphics::Direct2D::Common::{D2D_SIZE_U, ID2D1SimplifiedGeometrySink};
use windows::Win32::Graphics::DirectWrite::*;
use windows_core::{BOOL, HRESULT, Interface, PCWSTR, PWSTR, Ref, Result as WResult, implement};

/// Where a font's bytes come from: a file path (the system collection / `CreateFontFileReference`)
/// or already-loaded bytes (a custom collection / font set built from app-supplied or enumerated
/// files). `Bytes` keeps the data so a custom loader is read exactly once, at creation.
#[derive(Clone)]
pub(crate) enum FontSource {
    Path(PathBuf),
    Bytes(Arc<Vec<u8>>),
}

impl FontSource {
    pub(crate) fn rasterizer(&self) -> d2d::text::hinting::Rasterizer {
        use d2d::text::hinting::Rasterizer;
        let Self::Path(path) = self else { return Rasterizer::DirectWrite; };
        let dir = fonts_dir();
        let same = |a: &std::path::Path, b: &std::path::Path| {
            a.canonicalize().unwrap_or_else(|_| a.to_path_buf()) == b.canonicalize().unwrap_or_else(|_| b.to_path_buf())
        };
        #[cfg(target_os = "linux")]
        if std::env::var_os("D2D_FONTS").is_none() && super::font_discovery::paths().iter().any(|p| same(p, path)) {
            return Rasterizer::FontCache;
        }
        if path.parent().is_some_and(|p| same(p, &dir)) || registry_font_paths(&dir).iter().any(|p| same(p, path)) {
            Rasterizer::FontCache
        } else {
            Rasterizer::DirectWrite
        }
    }
    pub(crate) fn read(&self) -> Option<Arc<Vec<u8>>> {
        match self {
            FontSource::Path(p) => std::fs::read(p).ok().map(Arc::new),
            FontSource::Bytes(b) => Some(b.clone()),
        }
    }
    /// The on-disk path (for the file's last-write time), or `None` for in-memory bytes.
    pub(crate) fn path_str(&self) -> Option<String> {
        match self {
            FontSource::Path(p) => Some(p.to_string_lossy().into_owned()),
            FontSource::Bytes(_) => None,
        }
    }
}

/// One parsed font face: its family and the weight/stretch/style used to pick it, plus where its
/// bytes live. `sims` is the `DWRITE_FONT_SIMULATIONS` (0 real, 1 bold, 2 oblique, 3 bold+oblique) —
/// native synthesizes bold/oblique faces to fill the family's RBIZ grid.
struct FontEntry {
    family: String,
    weight: u16,
    stretch: u16,
    style: u8, // DWRITE_FONT_STYLE: 0 normal, 1 oblique, 2 italic
    sims: u16,
    source: FontSource,
    face: u32,
    /// For an fvar named-instance face, the instance's user axis values; empty for a static face or
    /// the default instance. Threaded onto the rendered `FaceObj` so the glyph outlines instance.
    coords: Vec<DWRITE_FONT_AXIS_VALUE>,
}

impl FontEntry {
    fn italic(&self) -> bool {
        self.style != 0
    }
}

struct FamilyData {
    name: String,
    name_lc: String,
    faces: Vec<usize>, // indices into `entries`
}

/// DirectWrite's weight-stretch-style family name: the GDI family name (`name` id 1) with trailing
/// weight / stretch / slope tokens removed — how `GetSystemFontCollection` groups faces into families
/// ("Arial Black" → "Arial", "Sitka Small Semibold" → "Sitka Small", "Bahnschrift SemiLight
/// Condensed" → "Bahnschrift"). Verified against the native system collection: this reproduces its 80
/// families and every face's derived family exactly.
fn wws_family(name: &str) -> String {
    const TOKENS: &[&str] = &[
        "thin", "extralight", "ultralight", "light", "semilight", "medium", "demibold", "semibold",
        "bold", "extrabold", "ultrabold", "black", "heavy", "extrablack", "ultrablack", "condensed",
        "semicondensed", "extracondensed", "ultracondensed", "semiexpanded", "expanded",
        "extraexpanded", "ultraexpanded", "italic", "oblique",
    ];
    let mut parts: Vec<&str> = name.split(' ').collect();
    while parts.len() > 1 && TOKENS.contains(&parts[parts.len() - 1].to_ascii_lowercase().as_str()) {
        parts.pop();
    }
    parts.join(" ")
}

/// The weight a GDI face name's trailing style word implies ("Candara Light" → 300), or `None`.
fn weight_token(name: &str) -> Option<u16> {
    Some(match name.rsplit(' ').next()?.to_ascii_lowercase().as_str() {
        "thin" => 100,
        "extralight" | "ultralight" => 200,
        "light" => 300,
        "semilight" => 350,
        "medium" => 500,
        "demibold" | "semibold" => 600,
        "bold" => 700,
        "extrabold" | "ultrabold" => 800,
        "black" | "heavy" => 900,
        "extrablack" | "ultrablack" => 950,
        _ => return None,
    })
}

/// A variable font's `wdth` axis value (percent) → the `DWRITE_FONT_STRETCH` (usWidthClass 1..9).
fn stretch_from_wdth(pct: f32) -> u16 {
    const PCTS: [f32; 9] = [50.0, 62.5, 75.0, 87.5, 100.0, 112.5, 125.0, 150.0, 200.0];
    (1..=9u16).min_by_key(|&i| ((PCTS[i as usize - 1] - pct).abs() * 1000.0) as i64).unwrap_or(5)
}

/// The STAT optical-size (`opsz`) axis-value name nearest `value` — the optical name native appends to
/// a variable font's family ("Small"/"Text"/"Display"/…). `None` if the font has no STAT opsz values.
fn opsz_name(font: &FontRef, value: f32) -> Option<String> {
    let stat = font.stat().ok()?;
    let opsz = stat.design_axes().ok()?.iter().position(|a| a.axis_tag() == Tag::new(b"opsz"))? as u16;
    let mut best: Option<(f32, StringId)> = None;
    for av in stat.offset_to_axis_values()?.ok()?.axis_values().iter().flatten() {
        let (idx, v, nid) = match av {
            AxisValue::Format1(a) => (a.axis_index(), a.value(), a.value_name_id()),
            AxisValue::Format2(a) => (a.axis_index(), a.nominal_value(), a.value_name_id()),
            AxisValue::Format3(a) => (a.axis_index(), a.value(), a.value_name_id()),
            _ => continue, // Format4 is multi-axis; opsz uses the single-axis formats
        };
        if idx != opsz {
            continue;
        }
        let d = (v.to_f32() - value).abs();
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, nid));
        }
    }
    font.localized_strings(best?.1).english_or_first().map(|s| s.to_string())
}

/// The STAT axis-value records for the design axis carrying `dwrite_tag`, as
/// `(value name id, range min, max)` — what `IDWriteFontResource::GetAxisValueName{Count,s}` reports.
/// Single-value records (Format 1/3) give a degenerate `[v, v]` range; Format 2 gives its nominal
/// range. Empty when the font has no STAT or no values for that axis. DWRITE packs a tag as the four
/// chars little-endian, so `to_le_bytes` recovers the OpenType tag skrifa compares.
fn stat_axis_values(font: &FontRef, dwrite_tag: DWRITE_FONT_AXIS_TAG) -> Vec<(StringId, f32, f32)> {
    let mut out = Vec::new();
    let tag = Tag::new(&dwrite_tag.0.to_le_bytes());
    let Ok(stat) = font.stat() else { return out };
    let Ok(axes) = stat.design_axes() else { return out };
    let Some(sidx) = axes.iter().position(|a| a.axis_tag() == tag) else { return out };
    let sidx = sidx as u16;
    let Some(Ok(avs)) = stat.offset_to_axis_values() else { return out };
    for av in avs.axis_values().iter().flatten() {
        let (idx, nid, lo, hi) = match av {
            AxisValue::Format1(a) => (a.axis_index(), a.value_name_id(), a.value().to_f32(), a.value().to_f32()),
            AxisValue::Format2(a) => {
                // An open range bound is stored as the Fixed extreme (±32768); native reports it as
                // ±Infinity, not the sentinel value.
                let lo = a.range_min_value().to_f32();
                let hi = a.range_max_value().to_f32();
                let lo = if lo <= -32767.0 { f32::NEG_INFINITY } else { lo };
                let hi = if hi >= 32767.0 { f32::INFINITY } else { hi };
                (a.axis_index(), a.value_name_id(), lo, hi)
            }
            AxisValue::Format3(a) => (a.axis_index(), a.value_name_id(), a.value().to_f32(), a.value().to_f32()),
            _ => continue, // Format4 is multi-axis; GetAxisValueNames reports the single-axis formats
        };
        if idx == sidx {
            out.push((nid, lo, hi));
        }
    }
    out
}

/// The parsed platform font set, shared (`Arc`) by the collection, families and fonts.
pub(crate) struct SystemFonts {
    families: Vec<FamilyData>,
    entries: Vec<FontEntry>,
    /// Every spelling that resolves to a family: its WWS name, each face's GDI (name 1) and
    /// typographic (name 16) names, lowercased → family index. Only the WWS families are enumerated;
    /// the GDI/typographic aliases let `FindFamilyName` / fallback still locate every face.
    aliases: HashMap<String, usize>,
    /// Which `DWRITE_FONT_FAMILY_MODEL` the faces are grouped under: `false` = weight-stretch-style
    /// (GDI families, trailing style tokens stripped), `true` = typographic (grouped by the
    /// typographic/preferred family, name id 16).
    typographic: bool,
}

/// The font directory: `D2D_FONTS` if set (how the cross-platform/Wine tests point at the VM's
/// fonts), else the Windows fonts directory.
pub(crate) fn fonts_dir() -> PathBuf {
    if let Ok(p) = std::env::var("D2D_FONTS") {
        return PathBuf::from(p);
    }
    let win = std::env::var("SystemRoot").or_else(|_| std::env::var("windir")).unwrap_or_else(|_| "C:\\Windows".into());
    PathBuf::from(win).join("Fonts")
}

/// Full paths of the fonts registered in `HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts`,
/// in the registry's storage (enumeration) order — the order native's font collection enumerates. A
/// bare filename resolves under `dir`; an absolute value is taken as-is. Empty on non-Windows.
#[cfg(windows)]
fn registry_font_paths(dir: &std::path::Path) -> Vec<PathBuf> {
    use std::os::raw::c_void;
    type Hkey = *mut c_void;
    const HKLM: Hkey = 0xFFFF_FFFF_8000_0002u64 as Hkey; // (HKEY)(ULONG_PTR)(LONG)0x80000002
    const KEY_READ: u32 = 0x2_0019;
    const ERROR_SUCCESS: i32 = 0;
    #[link(name = "advapi32")]
    unsafe extern "system" {
        fn RegOpenKeyExW(k: Hkey, sub: *const u16, opt: u32, sam: u32, out: *mut Hkey) -> i32;
        fn RegEnumValueW(k: Hkey, i: u32, name: *mut u16, name_len: *mut u32, res: *mut u32, typ: *mut u32, data: *mut u8, data_len: *mut u32) -> i32;
        fn RegCloseKey(k: Hkey) -> i32;
    }
    let sub: Vec<u16> = "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Fonts\0".encode_utf16().collect();
    let mut key: Hkey = core::ptr::null_mut();
    if unsafe { RegOpenKeyExW(HKLM, sub.as_ptr(), 0, KEY_READ, &mut key) } != ERROR_SUCCESS {
        return Vec::new();
    }
    let mut paths = Vec::new();
    for i in 0.. {
        let mut name = [0u16; 256];
        let mut data = [0u16; 520];
        let (mut nl, mut dl) = (name.len() as u32, (data.len() * 2) as u32);
        let r = unsafe {
            RegEnumValueW(key, i, name.as_mut_ptr(), &mut nl, core::ptr::null_mut(), core::ptr::null_mut(), data.as_mut_ptr() as *mut u8, &mut dl)
        };
        if r != ERROR_SUCCESS {
            break; // ERROR_NO_MORE_ITEMS (259) or any error ends enumeration
        }
        let units = (dl as usize / 2).saturating_sub(1).min(data.len()); // drop the NUL terminator
        let value = String::from_utf16_lossy(&data[..units]);
        if value.is_empty() {
            continue;
        }
        let p = std::path::Path::new(&value);
        paths.push(if p.is_absolute() { p.to_path_buf() } else { dir.join(p) });
    }
    unsafe { RegCloseKey(key) };
    paths
}

#[cfg(not(windows))]
fn registry_font_paths(_dir: &std::path::Path) -> Vec<PathBuf> {
    Vec::new()
}

impl SystemFonts {
    // ponytail: O(n) directory scan reading every font's name/attributes once. Fine for a pinned
    // font dir (the corpus); cache/lazy-parse per family if a full C:\Windows\Fonts ever gets slow.
    pub(crate) fn scan() -> Arc<SystemFonts> {
        Self::scan_model(false)
    }
    /// Like [`scan`] but grouping the system fonts under the requested family model (`true` =
    /// typographic). The typographic collection is not cached (requested rarely).
    pub(crate) fn scan_model(typographic: bool) -> Arc<SystemFonts> {
        let dir = fonts_dir();
        // Native's GetSystemFontCollection enumerates fonts in the HKLM `Fonts` registry's storage
        // order (RegEnumValue order ≈ install order, not alphabetical), then any font files present
        // but unregistered (e.g. Marlett). D2D_FONTS (the Wine/cross-platform tests) points at an
        // arbitrary directory with no matching registry, so there we just scan it, sorted.
        let use_registry = cfg!(windows) && std::env::var_os("D2D_FONTS").is_none();
        let mut sources = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let push = |p: PathBuf, sources: &mut Vec<FontSource>, seen: &mut std::collections::HashSet<String>| {
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
            if matches!(ext.as_str(), "ttf" | "ttc" | "otf") && seen.insert(p.to_string_lossy().to_ascii_lowercase()) {
                sources.push(FontSource::Path(p));
            }
        };
        if use_registry {
            for p in registry_font_paths(&dir) {
                push(p, &mut sources, &mut seen);
            }
        }
        // Remaining files in the directory, filename-sorted (the unregistered leftovers on Windows;
        // the whole set otherwise).
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir).into_iter().flatten().flatten().map(|de| de.path()).collect();
        files.sort_by_key(|p| p.to_string_lossy().to_ascii_lowercase());
        #[cfg(target_os = "linux")]
        if std::env::var_os("D2D_FONTS").is_none() {
            files = super::font_discovery::paths();
        }
        for p in files {
            push(p, &mut sources, &mut seen);
        }
        Self::from_sources_model(sources, typographic)
    }

    /// Parse each source's faces (family / weight / italic) into entries grouped into families —
    /// the grouping used by the system scan and custom collections (every face of every file).
    pub(crate) fn from_sources(sources: Vec<FontSource>) -> Arc<SystemFonts> {
        Self::from_sources_model(sources, false)
    }
    fn from_sources_model(sources: Vec<FontSource>, typographic: bool) -> Arc<SystemFonts> {
        let mut sf = SystemFonts { families: Vec::new(), entries: Vec::new(), aliases: HashMap::new(), typographic };
        for src in sources {
            let Some(bytes) = src.read() else { continue };
            let mut face = 0u32;
            while FontRef::from_index(&bytes, face).is_ok() {
                sf.add_face(&bytes, &src, face);
                face += 1;
            }
        }
        sf.decollide_weights();
        sf.synthesize_simulations();
        Arc::new(sf)
    }

    /// Like [`from_sources`] but for a specific set of `(source, faceIndex)` faces — a font set
    /// turned into a collection (`CreateFontCollectionFromFontSet`), not every face of each file.
    pub(crate) fn from_faces(faces: Vec<(FontSource, u32)>) -> Arc<SystemFonts> {
        Self::from_faces_model(faces, false)
    }
    pub(crate) fn from_faces_model(faces: Vec<(FontSource, u32)>, typographic: bool) -> Arc<SystemFonts> {
        let mut sf = SystemFonts { families: Vec::new(), entries: Vec::new(), aliases: HashMap::new(), typographic };
        for (src, face) in faces {
            if let Some(bytes) = src.read() {
                // Font-set members are explicit faces, so add each as itself (no fvar re-expansion).
                if let Ok(font) = FontRef::from_index(&bytes, face) {
                    sf.add_static_face(&font, &src, face);
                }
            }
        }
        Arc::new(sf)
    }

    /// Parse one face from `bytes`. A variable font is expanded into its fvar named instances (what
    /// native enumerates); a static face is added as itself.
    fn add_face(&mut self, bytes: &[u8], src: &FontSource, face: u32) {
        let Ok(font) = FontRef::from_index(bytes, face) else { return };
        if font.named_instances().is_empty() {
            self.add_static_face(&font, src, face);
        } else {
            self.add_variable_faces(&font, src, face);
        }
    }

    /// A static (non-variable) face: grouped under its weight-stretch-style family (the GDI family,
    /// name 1, minus trailing weight/width/slope tokens — "Arial Black" → "Arial"). The raw GDI name
    /// and the typographic family (name 16) resolve to it too so FindFamilyName / fallback still find
    /// every spelling.
    fn add_static_face(&mut self, font: &FontRef, src: &FontSource, face: u32) {
        let Some(n1) = font.localized_strings(StringId::FAMILY_NAME).english_or_first().map(|s| s.to_string()) else { return };
        let typo = font.localized_strings(StringId::TYPOGRAPHIC_FAMILY_NAME).english_or_first().map(|s| s.to_string());
        // Weight/stretch/style from OS/2 (usWeightClass/usWidthClass/fsSelection), as native reports
        // them — not skrifa's Attributes, which normalises e.g. Candara Light's 300 to 400.
        let (weight, stretch, style) = face_attrs(font);
        // Typographic model groups by the preferred family (name 16, falling back to the GDI name);
        // weight-stretch-style groups by the GDI family with trailing style tokens stripped.
        let family = if self.typographic {
            typo.clone().unwrap_or_else(|| n1.clone())
        } else {
            wws_family(&n1)
        };
        self.add_entry(family, Some((n1, typo)), weight, stretch, style as u8, 0, src, face);
    }

    /// A variable font: native's collection enumerates one face per fvar named instance, grouped by
    /// the typographic family plus the optical-size (opsz) name from STAT ("Sitka" + "Small" → "Sitka
    /// Small"); the weight is the instance's `wght` axis value. ponytail: rendering uses the file's
    /// default instance (the core text path has no variation support), which no corpus scene reaches.
    fn add_variable_faces(&mut self, font: &FontRef, src: &FontSource, face: u32) {
        let typo = font
            .localized_strings(StringId::TYPOGRAPHIC_FAMILY_NAME)
            .english_or_first()
            .or_else(|| font.localized_strings(StringId::FAMILY_NAME).english_or_first())
            .map(|s| s.to_string())
            .unwrap_or_default();
        let axes: Vec<Tag> = font.axes().iter().map(|a| a.tag()).collect();
        let wght_i = axes.iter().position(|t| *t == Tag::new(b"wght"));
        let wdth_i = axes.iter().position(|t| *t == Tag::new(b"wdth"));
        let opsz_i = axes.iter().position(|t| *t == Tag::new(b"opsz"));
        let style = face_attrs(font).2 as u8; // the file's style (upright vs italic variant)
        for inst in font.named_instances().iter() {
            let coords: Vec<f32> = inst.user_coords().collect();
            let weight = wght_i.and_then(|i| coords.get(i)).map(|w| w.round() as u16).unwrap_or(400);
            let stretch = wdth_i.and_then(|i| coords.get(i)).map(|&w| stretch_from_wdth(w)).unwrap_or(5);
            let optical = opsz_i.and_then(|i| coords.get(i)).and_then(|&o| opsz_name(font, o));
            let family = match optical {
                Some(o) if !o.is_empty() && !typo.is_empty() => format!("{typo} {o}"),
                _ => typo.clone(),
            };
            if !family.is_empty() {
                // Typographic model keeps the whole variable font under its typographic family (the
                // opsz split is a weight-stretch-style grouping); WSS strips style tokens.
                let key = if self.typographic { typo.clone() } else { wws_family(&family) };
                self.add_entry(key, None, weight, stretch, style, 0, src, face);
                // This instance's user axis values (DWRITE packs each tag little-endian), so the
                // rendered face instances the gvar outline at the instance rather than the default.
                let axis_vals: Vec<DWRITE_FONT_AXIS_VALUE> = axes
                    .iter()
                    .zip(&coords)
                    .map(|(t, &v)| DWRITE_FONT_AXIS_VALUE { axisTag: DWRITE_FONT_AXIS_TAG(u32::from_le_bytes(t.to_be_bytes())), value: v })
                    .collect();
                if let Some(e) = self.entries.last_mut() {
                    e.coords = axis_vals;
                }
            }
        }
    }

    /// Add one face as an entry under its WWS `family`. `names` (a static face's GDI name 1 and
    /// typographic name 16) also alias to the family, in the weight-stretch-style model only. The entry's displayed family name is the GDI
    /// name for a static face, else the WWS family.
    #[allow(clippy::too_many_arguments)]
    fn add_entry(&mut self, family: String, names: Option<(String, Option<String>)>, weight: u16, stretch: u16, style: u8, sims: u16, src: &FontSource, face: u32) {
        let entry = self.entries.len();
        let display = names.as_ref().map(|(n1, _)| n1.clone()).unwrap_or_else(|| family.clone());
        self.entries.push(FontEntry { family: display, weight, stretch, style, sims, source: src.clone(), face, coords: Vec::new() });
        let fi = self.family_index(&family);
        self.families[fi].faces.push(entry);
        self.aliases.entry(family.to_ascii_lowercase()).or_insert(fi);
        // A typographic collection answers only to its typographic family names: native's cold font-set
        // collection does not find "Arial Black" (ariblk sits under "Arial") and lays it out in the
        // fallback Segoe UI.
        if let Some((n1, typo)) = names.filter(|_| !self.typographic) {
            self.aliases.entry(n1.to_ascii_lowercase()).or_insert(fi);
            if let Some(t) = typo {
                self.aliases.entry(t.to_ascii_lowercase()).or_insert(fi);
            }
        }
    }

    /// De-collide weights native disambiguates within a family: when two real faces of the same
    /// stretch/style share a usWeightClass (e.g. Candara's Light and Regular are both 400 in OS/2 —
    /// a font quirk), native reassigns the one whose GDI name carries a weight word to that word's
    /// weight ("Candara Light" → 300). A unique weight (Franklin Gothic Medium, alone at 400) is left.
    fn decollide_weights(&mut self) {
        for fam in 0..self.families.len() {
            let faces: Vec<usize> = self.families[fam].faces.iter().copied().filter(|&e| self.entries[e].sims == 0).collect();
            for &e in &faces {
                let Some(tw) = weight_token(&self.entries[e].family) else { continue };
                let (w, st, sty) = (self.entries[e].weight, self.entries[e].stretch, self.entries[e].style);
                if tw != w && faces.iter().any(|&o| o != e && self.entries[o].weight == w && self.entries[o].stretch == st && self.entries[o].style == sty) {
                    self.entries[e].weight = tw;
                }
            }
        }
    }

    /// Add the bold/oblique simulated faces native synthesizes to fill each family's RBIZ grid, after
    /// all real faces are in. Per `(stretch, style∈{normal,italic})` group: if a real weight < 700 is
    /// present and 700 is not, add a BOLD (700) face; and add an OBLIQUE face for every upright face
    /// (real or bold-sim → sim OBLIQUE or BOLD+OBLIQUE). Verified against the native collection: this
    /// reproduces its per-family face counts and simulation flags exactly.
    fn synthesize_simulations(&mut self) {
        for fam in 0..self.families.len() {
            let reals = self.families[fam].faces.clone();
            // Group real faces by (stretch, style) to decide bold synthesis.
            let mut groups: HashMap<(u16, u8), Vec<u16>> = HashMap::new();
            for &e in &reals {
                let en = &self.entries[e];
                groups.entry((en.stretch, en.style)).or_default().push(en.weight);
            }
            let mut sims: Vec<FontEntry> = Vec::new();
            // Bold (700) for a normal/italic group with a lighter weight but no 700.
            let mut bold_uprights: Vec<usize> = Vec::new();
            for &e in &reals {
                let en = &self.entries[e];
                if en.style == 1 {
                    continue;
                }
                let ws = &groups[&(en.stretch, en.style)];
                // Only the "regular" (weight nearest 400) of the group seeds the bold.
                let regular = reals.iter().filter(|&&r| self.entries[r].stretch == en.stretch && self.entries[r].style == en.style)
                    .min_by_key(|&&r| (self.entries[r].weight as i32 - 400).abs()).copied();
                if regular == Some(e) && ws.iter().any(|&w| w < 700) && !ws.contains(&700) {
                    sims.push(FontEntry { family: en.family.clone(), weight: 700, stretch: en.stretch, style: en.style, sims: 1, source: en.source.clone(), face: en.face, coords: en.coords.clone() });
                    if en.style == 0 {
                        bold_uprights.push(sims.len() - 1);
                    }
                }
            }
            // Oblique for every upright real face.
            for &e in &reals {
                let en = &self.entries[e];
                if en.style == 0 {
                    sims.push(FontEntry { family: en.family.clone(), weight: en.weight, stretch: en.stretch, style: 1, sims: 2, source: en.source.clone(), face: en.face, coords: en.coords.clone() });
                }
            }
            // Bold+oblique for each synthesized upright bold.
            for &bi in &bold_uprights {
                let b = &sims[bi];
                sims.push(FontEntry { family: b.family.clone(), weight: 700, stretch: b.stretch, style: 1, sims: 3, source: b.source.clone(), face: b.face, coords: b.coords.clone() });
            }
            for ne in sims {
                let idx = self.entries.len();
                self.entries.push(ne);
                self.families[fam].faces.push(idx);
            }
            // Native's within-family face order: real faces before simulated, then by weight, with
            // normal width before the others, then style. Verified exact across the collection.
            let mut faces = std::mem::take(&mut self.families[fam].faces);
            faces.sort_by_key(|&e| {
                let en = &self.entries[e];
                (en.sims != 0, en.weight, en.stretch != 5, en.stretch, en.style)
            });
            self.families[fam].faces = faces;
        }
    }

    /// Index of the enumerated WWS family named `wws`, creating it if new (keyed case-insensitively).
    fn family_index(&mut self, wws: &str) -> usize {
        let lc = wws.to_ascii_lowercase();
        self.families.iter().position(|f| f.name_lc == lc).unwrap_or_else(|| {
            self.families.push(FamilyData { name: wws.to_string(), name_lc: lc, faces: Vec::new() });
            self.families.len() - 1
        })
    }

    /// Every face as a `(source, faceIndex)` pair — the system font set's members.
    pub(crate) fn face_refs(&self) -> Vec<(FontSource, u32)> {
        self.entries.iter().map(|e| (e.source.clone(), e.face)).collect()
    }

    fn find_family(&self, name: &str) -> Option<usize> {
        self.aliases.get(&name.to_ascii_lowercase()).copied()
    }

    /// The family a format or range names but the collection lacks lays its text out with: the system
    /// fallback's Segoe UI; without Segoe UI (a machine without Microsoft's fonts), the family
    /// `D2D_FALLBACK_FAMILY` names, else the collection's first family, so a draw never fails for lack of
    /// Segoe UI. `None` only for an empty collection.
    pub(crate) fn stand_in(&self) -> Option<String> {
        ["Segoe UI".to_string()]
            .into_iter()
            .chain(std::env::var("D2D_FALLBACK_FAMILY").ok())
            .chain(self.families.first().map(|f| f.name.clone()))
            .find(|f| self.find_family(f).is_some())
    }

    /// `GetFirstMatchingFont`: the family face nearest the requested stretch/style/weight, by the CSS
    /// (DWrite) font-matching order — stretch first, then **style**, then weight — with a stable tie-break
    /// on the family's face order (so a real face beats a simulated one at equal distance). Style outranks
    /// weight: an upright heavier face must not beat an italic lighter one for an italic request — e.g.
    /// requesting Arial italic at weight 900 keeps Arial Bold Italic (700) rather than jumping to the
    /// upright Arial Black (900), whose different advances shift the whole run.
    fn match_entry(&self, fam: usize, weight: u16, stretch: u16, style: u8) -> Option<usize> {
        self.ranked(fam, weight, stretch, style).first().copied()
    }

    /// The family's faces (real and simulated) in [`Self::match_entry`]'s rank order, its match first.
    fn ranked(&self, fam: usize, weight: u16, stretch: u16, style: u8) -> Vec<usize> {
        let mut faces = self.families[fam].faces.clone();
        faces.sort_by_key(|&e| {
            let en = &self.entries[e];
            (stretch_rank(en.stretch, stretch), slant_rank(en.style, style), weight_rank(en.weight, weight), style_rank(en.style, style))
        });
        faces
    }

    /// Font fallback's face of a family for `ch` and the simulations it takes: the first real face in
    /// [`fallback::wws_fallback_order`] whose cmap maps it, so a Segoe UI request at weight 900 takes Segoe UI
    /// Bold for Arabic, which Segoe UI Black lacks (VM r25 30951). The faces start in font-set order: the
    /// system set's registry order, a font-set collection's own. `None` when no face of the family has it.
    /// `bytes` reads (and may cache) an entry's file.
    fn match_entry_covering(&self, fam: usize, weight: u16, stretch: u16, style: u8, ch: char, bytes: impl Fn(usize) -> Option<Arc<Vec<u8>>>) -> Option<(usize, u16)> {
        let mut reals: Vec<usize> = self.families[fam].faces.iter().copied().filter(|&e| self.entries[e].sims == 0).collect();
        // ponytail: registry order modelled as names sorted, a file's faces in index order (native lists Yu
        // Gothic UI Semibold before Bold, from one file); the scanned order if the registry ever diverges.
        if !self.typographic {
            reals.sort_by_cached_key(|&e| (bytes(e).map(|b| fallback::set_order_key(&b)).unwrap_or_default(), self.entries[e].face));
        }
        let faces: Vec<_> = reals.iter().map(|&e| (self.entries[e].weight, self.entries[e].stretch, self.entries[e].style)).collect();
        fallback::wws_fallback_order(&faces, weight, stretch, style)
            .into_iter()
            .map(|(i, sims)| (reals[i], sims))
            .find(|&(e, _)| bytes(e).is_some_and(|b| FontRef::from_index(&b, self.entries[e].face).ok().and_then(|f| f.charmap().map(ch)).is_some()))
    }

    /// Resolve a family name + weight/style to its file bytes, face index and canonical family name.
    pub(crate) fn resolve(&self, family: &str, weight: u16, italic: bool) -> Option<(Arc<Vec<u8>>, u32, String)> {
        self.resolve_sims(family, weight, italic).map(|(b, i, f, _)| (b, i, f))
    }

    /// As [`resolve`], also returning the matched entry's `DWRITE_FONT_SIMULATIONS` so a bold/oblique
    /// synthesized match (e.g. Impact at weight 700) can carry its simulation into layout/advances.
    pub(crate) fn resolve_sims(&self, family: &str, weight: u16, italic: bool) -> Option<(Arc<Vec<u8>>, u32, String, u16)> {
        let e = self.match_entry(self.find_family(family)?, weight, 5, if italic { 2 } else { 0 })?;
        let ent = &self.entries[e];
        Some((ent.source.read()?, ent.face, ent.family.clone(), ent.sims))
    }

    /// As [`resolve_sims`] but with the full `DWRITE` style (0 normal, 1 oblique, 2 italic) and stretch
    /// (1-9, 5 normal) a per-range `SetFontStyle`/`SetFontStretch` can carry, matched by the same
    /// `GetFirstMatchingFont` order (stretch, style, weight).
    pub(crate) fn resolve_full(&self, family: &str, weight: u16, style: u8, stretch: u16) -> Option<(Arc<Vec<u8>>, u32, String, u16)> {
        let e = self.match_entry(self.find_family(family)?, weight, stretch, style)?;
        let ent = &self.entries[e];
        Some((ent.source.read()?, ent.face, ent.family.clone(), ent.sims))
    }

    pub(crate) fn resolve_source(&self, family: &str, weight: u16, italic: bool) -> Option<FontSource> {
        let e = self.match_entry(self.find_family(family)?, weight, 5, if italic { 2 } else { 0 })?;
        Some(self.entries[e].source.clone())
    }

    /// The family index that owns `entry` (its canonical family name's group).
    fn family_of(&self, entry: usize) -> Option<usize> {
        self.find_family(&self.entries[entry].family)
    }

    /// The entry a family name + weight/italic resolves to (`CreateFontFromLOGFONT`/GDI interop).
    #[cfg(windows)] // only the Windows-only GDI interop (`win::gdi`) resolves fonts this way.
    pub(crate) fn find_match(&self, family: &str, weight: u16, italic: bool) -> Option<usize> {
        self.match_entry(self.find_family(family)?, weight, 5, if italic { 2 } else { 0 })
    }
}

/// Resolve `family` + `weight`/`style`/`stretch` to a concrete face file in the font directory `dir`,
/// via the system-collection matcher (`GetFirstMatchingFont`: stretch, then style, then weight) — the
/// same matching the drop-in `dwrite.dll` does, verified against the native collection. Returns the
/// on-disk path and face index, or `None` if the directory has no matching family. `weight` is the
/// `DWRITE_FONT_WEIGHT` (100-900), `style` the `DWRITE_FONT_STYLE` (0 normal, 1 oblique, 2 italic),
/// `stretch` the `DWRITE_FONT_STRETCH` (1-9, 5 normal).
///
/// For test/fuzz harnesses that drive the `d2d` core by font bytes: one call replaces a hand-maintained
/// family→file table with native's actual weight/stretch/style face selection, so every family and
/// weight resolves as DirectWrite does. The scanned collection is cached per directory (a run pins its
/// font dir), matching `SystemFonts::scan` on a `D2D_FONTS`-style directory (sorted file order).
pub fn resolve_face(dir: &std::path::Path, family: &str, weight: u32, style: u32, stretch: u32) -> Option<(PathBuf, u32)> {
    resolve_face_sims(dir, family, weight, style, stretch).map(|(p, f, _)| (p, f))
}

/// [`resolve_face`] plus the matched face's `DWRITE_FONT_SIMULATIONS` (bold 1, oblique 2): the face
/// `GetFirstMatchingFont` returns may be a synthesized one (an oblique request matches a family's
/// simulated-oblique upright face), which the text drawn with it must render with.
pub fn resolve_face_sims(dir: &std::path::Path, family: &str, weight: u32, style: u32, stretch: u32) -> Option<(PathBuf, u32, u16)> {
    // Mirror SystemFonts::scan's non-registry listing (the filename order is the tie-break the matcher
    // relies on).
    resolve_face_sims_in(&dir_files(dir), family, weight, style, stretch)
}

/// [`resolve_face_sims`] over exactly `files`, in that order: the collection a font-set builder makes
/// from those files alone (a custom collection, as a cold capture builds), so a face missing from the
/// list can never be matched. The scanned collection is cached per file list.
pub fn resolve_face_sims_in(files: &[PathBuf], family: &str, weight: u32, style: u32, stretch: u32) -> Option<(PathBuf, u32, u16)> {
    resolve_in_model(files, false, family, weight, style, stretch)
}

/// [`resolve_face_sims_in`] with the files grouped by typographic family name, as a collection built from a
/// font set groups them: a face whose legacy family is a style of a larger family ("Arial Black" is Arial at
/// weight 900) is found only under that family.
pub fn resolve_face_sims_typographic(files: &[PathBuf], family: &str, weight: u32, style: u32, stretch: u32) -> Option<(PathBuf, u32, u16)> {
    resolve_in_model(files, true, family, weight, style, stretch)
}

/// [`resolve_face_sims`] for font fallback: the family's best-ranked face that maps `ch`, or `None` when
/// none does (fallback then moves on to the next family).
pub fn resolve_face_sims_covering(dir: &std::path::Path, family: &str, weight: u32, style: u32, stretch: u32, ch: char) -> Option<(PathBuf, u32, u16)> {
    covering_in_model(&dir_files(dir), false, family, weight, style, stretch, ch)
}

/// [`resolve_face_sims_covering`] over exactly `files` grouped by typographic family, as
/// [`resolve_face_sims_typographic`]: the layout's own family in a custom (font-set) collection.
pub fn resolve_face_sims_covering_typographic(files: &[PathBuf], family: &str, weight: u32, style: u32, stretch: u32, ch: char) -> Option<(PathBuf, u32, u16)> {
    covering_in_model(files, true, family, weight, style, stretch, ch)
}

fn covering_in_model(files: &[PathBuf], typographic: bool, family: &str, weight: u32, style: u32, stretch: u32, ch: char) -> Option<(PathBuf, u32, u16)> {
    static BYTES: std::sync::OnceLock<std::sync::Mutex<HashMap<PathBuf, Arc<Vec<u8>>>>> = std::sync::OnceLock::new();
    let sf = scanned(files, typographic);
    let read = |e: usize| match &sf.entries[e].source {
        FontSource::Path(p) => {
            let mut c = BYTES.get_or_init(|| std::sync::Mutex::new(HashMap::new())).lock().unwrap();
            if let Some(b) = c.get(p) {
                return Some(b.clone());
            }
            let b = Arc::new(std::fs::read(p).ok()?);
            c.insert(p.clone(), b.clone());
            Some(b)
        }
        FontSource::Bytes(b) => Some(b.clone()),
    };
    let (e, sims) = sf.match_entry_covering(sf.find_family(family)?, weight as u16, stretch as u16, style as u8, ch, read)?;
    entry_path(&sf, e).map(|(p, f, _)| (p, f, sims))
}

/// Every .ttf/.ttc/.otf in `dir`, filename-sorted, as SystemFonts::scan lists a non-registry directory.
fn dir_files(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir).into_iter().flatten().flatten().map(|de| de.path()).collect();
    files.sort_by_key(|p| p.to_string_lossy().to_ascii_lowercase());
    files.retain(|p| matches!(p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase().as_str(), "ttf" | "ttc" | "otf"));
    files
}

/// The collection scanned from exactly `files` (cached per file list and family model).
fn scanned(files: &[PathBuf], typographic: bool) -> Arc<SystemFonts> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<HashMap<(Vec<PathBuf>, bool), Arc<SystemFonts>>>> = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    cache
        .lock()
        .unwrap()
        .entry((files.to_vec(), typographic))
        .or_insert_with(|| SystemFonts::from_sources_model(files.iter().cloned().map(FontSource::Path).collect(), typographic))
        .clone()
}

fn entry_path(sf: &SystemFonts, e: usize) -> Option<(PathBuf, u32, u16)> {
    match &sf.entries[e].source {
        FontSource::Path(p) => Some((p.clone(), sf.entries[e].face, sf.entries[e].sims)),
        FontSource::Bytes(_) => None,
    }
}

fn resolve_in_model(files: &[PathBuf], typographic: bool, family: &str, weight: u32, style: u32, stretch: u32) -> Option<(PathBuf, u32, u16)> {
    let sf = scanned(files, typographic);
    let e = sf.match_entry(sf.find_family(family)?, weight as u16, stretch as u16, style as u8)?;
    match &sf.entries[e].source {
        FontSource::Path(p) => Some((p.clone(), sf.entries[e].face, sf.entries[e].sims)),
        FontSource::Bytes(_) => None,
    }
}

/// CSS/DWrite stretch distance (lower = better): at or below the normal width prefer narrower, else
/// wider, matching `GetFirstMatchingFont`.
fn stretch_rank(have: u16, want: u16) -> (u8, u16) {
    if want <= 5 {
        if have <= want { (0, want - have) } else { (1, have - want) }
    } else if have >= want {
        (0, have - want)
    } else {
        (1, want - have)
    }
}

/// CSS/DWrite style distance: exact style first, then the other slanted style, then upright.
/// Slant first: a slanted request takes italic and oblique alike before weight, the exact style breaking a
/// weight tie (r25 30859: Arial at weight 900 italic is Arial Black's simulated oblique, not Arial Bold Italic);
/// an upright request ranks normal, oblique, italic.
fn slant_rank(have: u8, want: u8) -> u8 {
    match (want, have) {
        (0, _) => have.min(2),
        (_, 0) => 2,
        _ => 0,
    }
}

fn style_rank(have: u8, want: u8) -> u8 {
    match want {
        2 => [2, 1, 0][have.min(2) as usize], // italic: italic < oblique < normal
        1 => match have { 1 => 0, 2 => 1, _ => 2 }, // oblique: oblique < italic < normal
        _ => have.min(2),                     // normal: normal < oblique < italic
    }
}

/// DWrite weight distance: nearest requested weight, and on an equal-distance tie the heavier weight
/// wins (500 → 600 over 400, 600 → 700 over 500). Matches `GetFirstMatchingFont` across the corpus.
fn weight_rank(have: u16, want: u16) -> (u16, u16) {
    ((have as i32 - want as i32).unsigned_abs() as u16, u16::MAX - have)
}

/// An entry as a base `IDWriteFont` (GDI interop's `CreateFontFromLOGFONT`).
#[cfg(windows)] // only the Windows-only GDI interop (`win::gdi`) needs a bare `IDWriteFont`.
pub(crate) fn font_iface(fonts: Arc<SystemFonts>, entry: usize) -> WResult<IDWriteFont> {
    Ok(font3(fonts, entry).cast()?)
}

/// `GetRecommendedRenderingMode`: `DWRITE_RENDERING_MODE(1)` for a device em-pixel size and measuring
/// mode, via the `text::recommended_rendering_mode` port of native `GetRecommendedRenderingModeInternal`.
/// `em_px` is the already-device em size; `aa` (`D2D1_TEXT_ANTIALIAS_MODE`: 2 grayscale, 3 aliased)
/// picks the 100-px antialiased vs 350-px aliased outline threshold; an explicit `params_mode`
/// (non-DEFAULT) passes through.
fn recommend(font_data: &[u8], face_index: u32, em_px: f32, measuring: u32, aa: u32, params_mode: u32, dpi: [f32; 2]) -> u32 {
    let run = TRun {
        font_data,
        face_index,
        em_size: em_px,
        glyphs: &[] as &[TGlyph],
        baseline_origin: [0.0, 0.0],
        measuring_mode: measuring,
        is_sideways: false,
        bidi_level: 0,
        no_snap: false,
        simulations: 0,
        coords: &[],
    };
    let params = RenderingParams { rendering_mode: params_mode, dpi, ..Default::default() };
    // dt = identity since `em_px` is already device pixels.
    text::recommended_rendering_mode(&run, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], aa, &params)
}

// Native Face0..2 cannot expose the mode7 introduced by DWRITE_RENDERING_MODE1.
fn legacy_recommend(mode: u32) -> u32 {
    if mode == 7 { 5 } else { mode }
}

/// Device em-pixel size for the Face1/2/3 overloads: em × (dpi/96) × the transform's max axis scale.
/// The params' rendering mode as DirectWrite reads it for a recommendation: the full
/// DWRITE_RENDERING_MODE1 value (the legacy getter reports NATURAL_SYMMETRIC_DOWNSAMPLED as 5).
fn params_mode(p: &IDWriteRenderingParams) -> u32 {
    unsafe { p.cast::<IDWriteRenderingParams3>().map_or_else(|_| p.GetRenderingMode().0, |p3| p3.GetRenderingMode1().0) as u32 }
}

fn device_em(fontemsize: f32, dpi: f32, transform: *const DWRITE_MATRIX) -> f32 {
    let ts = unsafe { transform.as_ref() }.map(|m| m.m11.abs().max(m.m22.abs())).unwrap_or(1.0);
    fontemsize * (dpi / 96.0) * ts
}

/// `DWRITE_OUTLINE_THRESHOLD`: ALIASED(1) uses the 350-px aliased cutoff, else the 100-px
/// antialiased cutoff — expressed as the `D2D1_TEXT_ANTIALIAS_MODE` `recommend` keys off (3/2).
fn aa_for_threshold(th: DWRITE_OUTLINE_THRESHOLD) -> u32 {
    if th == DWRITE_OUTLINE_THRESHOLD_ALIASED { 3 } else { 2 }
}

/// Grid-fit mode the Face2/Face3 recommendation pairs with `mode`: an explicit params grid-fit wins;
/// OUTLINE disables grid-fit; a rotated/skewed transform disables grid-fit (text spec R7); else
/// enabled. ponytail: best-effort pairing (the mode itself is the ported-routine result) — pin the
/// exact off-axis/OUTLINE grid-fit against the VM when a GetRecommendedRenderingMode scene exists.
fn recommend_gridfit(transform: *const DWRITE_MATRIX, mode: u32, params: &Ref<IDWriteRenderingParams>) -> DWRITE_GRID_FIT_MODE {
    if let Some(g) = params.as_ref().and_then(|p| p.cast::<IDWriteRenderingParams2>().ok()).map(|p2| unsafe { p2.GetGridFitMode() }) {
        if g != DWRITE_GRID_FIT_MODE_DEFAULT {
            return g;
        }
    }
    if mode == 6 {
        return DWRITE_GRID_FIT_MODE_DISABLED; // OUTLINE
    }
    let rotated = unsafe { transform.as_ref() }.map(|m| m.m12 != 0.0 || m.m21 != 0.0).unwrap_or(false);
    if rotated { DWRITE_GRID_FIT_MODE_DISABLED } else { DWRITE_GRID_FIT_MODE_ENABLED }
}

// --- Face metadata extracted straight from the font tables (skrifa) ----------------------------
// The getter bodies below (FaceObj/FontObj) read these; keeping them here means a system font and an
// app-loaded font report identical metrics/attributes.

/// `DWRITE_FONT_METRICS` in design units, from skrifa's unscaled metrics (hhea ascent/descent/gap,
/// OS/2 cap/x-height, post underline, OS/2 strikeout). hhea descent is negative; DWRITE reports its
/// magnitude.
fn design_metrics(font: &FontRef) -> DWRITE_FONT_METRICS {
    let m = font.metrics(Size::unscaled(), LocationRef::default());
    let und = m.underline.unwrap_or_default();
    let strike = m.strikeout.unwrap_or_default();
    let (ascent, descent, line_gap) = line_metrics(font).unwrap_or((m.ascent.round() as i32, (-m.descent).round() as i32, m.leading.round() as i32));
    DWRITE_FONT_METRICS {
        designUnitsPerEm: m.units_per_em,
        ascent: ascent as u16,
        descent: descent as u16,
        lineGap: line_gap as i16,
        capHeight: m.cap_height.unwrap_or(0.0).round() as u16,
        xHeight: m.x_height.unwrap_or(0.0).round() as u16,
        underlinePosition: und.offset.round() as i16,
        underlineThickness: und.thickness.round() as u16,
        strikethroughPosition: strike.offset.round() as i16,
        strikethroughThickness: strike.thickness.round() as u16,
    }
}

/// The ascent, descent and line gap DirectWrite reports: OS/2's typographic values when its USE_TYPO_METRICS
/// flag is set, else `usWinAscent` / `usWinDescent` with the line gap the part of hhea's line height beyond
/// them (VM GetMetrics over 22 faces: Segoe UI Emoji 2210 / 514 / 0, Calibri 1950 / 550 / 0, Arial 1854 /
/// 434 / 67, where hhea gives Segoe UI Emoji 1491 / 431 / 269). The layout's line box uses the same rule.
fn line_metrics(font: &FontRef) -> Option<(i32, i32, i32)> {
    let os2 = font.os2().ok()?;
    if os2.fs_selection().contains(SelectionFlags::USE_TYPO_METRICS) {
        return Some((os2.s_typo_ascender() as i32, -(os2.s_typo_descender() as i32), os2.s_typo_line_gap() as i32));
    }
    let (wa, wd) = (os2.us_win_ascent() as i32, os2.us_win_descent() as i32);
    let hhea_lh = font.hhea().ok().map_or(0, |h| h.ascender().to_i16() as i32 - h.descender().to_i16() as i32 + h.line_gap().to_i16() as i32);
    Some((wa, wd, (hhea_lh - (wa + wd)).max(0)))
}

/// `DWRITE_FONT_METRICS1`: the base metrics plus the glyph-extent box (`head`). The sub/superscript
/// and typographic fields are left zero (no corpus reads them). ponytail: fill from OS/2 if a scene
/// ever needs them.
fn design_metrics1(font: &FontRef) -> DWRITE_FONT_METRICS1 {
    let b = design_metrics(font);
    let bb = font.metrics(Size::unscaled(), LocationRef::default()).bounds.unwrap_or_default();
    DWRITE_FONT_METRICS1 {
        Base: b,
        glyphBoxLeft: bb.x_min.round() as i16,
        glyphBoxTop: bb.y_max.round() as i16,
        glyphBoxRight: bb.x_max.round() as i16,
        glyphBoxBottom: bb.y_min.round() as i16,
        ..Default::default()
    }
}

/// `(weight, stretch, style)` straight from OS/2 `usWeightClass`/`usWidthClass`/`fsSelection` — the
/// source DWRITE reports for `IDWriteFont*`/`IDWriteFontFace3` weight/stretch/style.
fn face_attrs(font: &FontRef) -> (u16, u16, u32) {
    let Ok(os2) = font.os2() else { return (400, 5, 0) };
    let style = if os2.fs_selection().contains(SelectionFlags::ITALIC) {
        2 // DWRITE_FONT_STYLE_ITALIC
    } else if os2.fs_selection().contains(SelectionFlags::OBLIQUE) {
        1 // DWRITE_FONT_STYLE_OBLIQUE
    } else {
        0
    };
    (os2.us_weight_class(), os2.us_width_class(), style)
}

/// One `fvar` variation axis in DirectWrite's terms, backing the `IDWriteFontFace5`/
/// `IDWriteFontResource` axis surface.
pub(crate) struct FvarAxis {
    pub(crate) tag: DWRITE_FONT_AXIS_TAG,
    pub(crate) default: f32,
    pub(crate) min: f32,
    pub(crate) max: f32,
    /// The raw `fvar` axis flags (bit 0 = `HIDDEN_AXIS`).
    pub(crate) flags: u16,
    /// The `axisNameID` (a `name`-table id) for `GetAxisNames`.
    pub(crate) name_id: u16,
}

/// The axis values `IDWriteFontFace5`/`IDWriteFontFaceReference1` report for a face created without
/// explicit axis values: a variable font's `fvar` default coordinates, or a static font's standard
/// wght/wdth/ital synthesized from OS/2.
pub(crate) fn face_axis_values(font: &FontRef) -> Vec<DWRITE_FONT_AXIS_VALUE> {
    let fvar = fvar_axes(font);
    if !fvar.is_empty() {
        return fvar.iter().map(|a| DWRITE_FONT_AXIS_VALUE { axisTag: a.tag, value: a.default }).collect();
    }
    // A static font synthesises the four standard axes wght/wdth/ital/slnt, in that order.
    // VM-verified against native IDWriteFontFace5::GetFontAxisValues on win11-26100: Arial →
    // [wght 400, wdth 100, ital 0, slnt 0], Arial Italic → [wght 400, wdth 100, ital 1, slnt -12].
    // wght = OS/2 usWeightClass, wdth = width% from usWidthClass, ital = the italic bit, slnt = the
    // `post` table italic angle (0 upright, the font's negative slant for an italic).
    let (weight, width_class, style) = face_attrs(font);
    let slant = font.post().map(|p| p.italic_angle().to_f32()).unwrap_or(0.0);
    vec![
        DWRITE_FONT_AXIS_VALUE { axisTag: DWRITE_FONT_AXIS_TAG_WEIGHT, value: weight as f32 },
        DWRITE_FONT_AXIS_VALUE { axisTag: DWRITE_FONT_AXIS_TAG_WIDTH, value: width_pct(width_class) },
        DWRITE_FONT_AXIS_VALUE { axisTag: DWRITE_FONT_AXIS_TAG_ITALIC, value: (style == 2) as u8 as f32 },
        DWRITE_FONT_AXIS_VALUE { axisTag: DWRITE_FONT_AXIS_TAG_SLANT, value: slant },
    ]
}

/// Insert or override one axis value in `out` by tag (axis values are a tag-keyed set).
fn push_axis(out: &mut Vec<DWRITE_FONT_AXIS_VALUE>, tag: DWRITE_FONT_AXIS_TAG, value: f32) {
    match out.iter_mut().find(|v| v.axisTag == tag) {
        Some(v) => v.value = value,
        None => out.push(DWRITE_FONT_AXIS_VALUE { axisTag: tag, value }),
    }
}

/// `ConvertWeightStretchStyleToFontAxisValues`: weight/stretch/style/size → the wght/wdth/slnt|ital/
/// opsz axes, with caller `input` axes overriding and extending the synthesized set.
/// ponytail: the slant angle for oblique (−20°), opsz-only-when-size>0, and the override order are
/// the documented defaults; exact tie-breaks are VM-pending (shared with the static-axis synthesis).
pub fn wss_to_axis_values(input: &[DWRITE_FONT_AXIS_VALUE], weight: DWRITE_FONT_WEIGHT, stretch: DWRITE_FONT_STRETCH, style: DWRITE_FONT_STYLE, size: f32) -> Vec<DWRITE_FONT_AXIS_VALUE> {
    let mut out = Vec::new();
    push_axis(&mut out, DWRITE_FONT_AXIS_TAG_WEIGHT, weight.0 as f32);
    push_axis(&mut out, DWRITE_FONT_AXIS_TAG_WIDTH, width_pct(stretch.0 as u16));
    match style.0 {
        2 => push_axis(&mut out, DWRITE_FONT_AXIS_TAG_ITALIC, 1.0), // DWRITE_FONT_STYLE_ITALIC
        1 => push_axis(&mut out, DWRITE_FONT_AXIS_TAG_SLANT, -20.0), // DWRITE_FONT_STYLE_OBLIQUE
        _ => {}
    }
    if size > 0.0 {
        push_axis(&mut out, DWRITE_FONT_AXIS_TAG_OPTICAL_SIZE, size);
    }
    for v in input {
        push_axis(&mut out, v.axisTag, v.value);
    }
    out
}

/// `IDWriteFontSet4::ConvertWeightStretchStyleToFontAxisValues` — the PUBLIC conversion, which differs
/// from the internal font-matching [`wss_to_axis_values`] (VM-verified):
/// it ALWAYS emits wght, wdth, ital and slnt (order fixed), with ital/slnt per style
/// NORMAL=(0,0)/ITALIC=(1,−12)/OBLIQUE=(0,−20), and opsz in POINTS (= size DIP × 72/96) only when
/// size>0; caller `input` axes then override/extend by tag.
pub fn convert_wss_axis_values(input: &[DWRITE_FONT_AXIS_VALUE], weight: DWRITE_FONT_WEIGHT, stretch: DWRITE_FONT_STRETCH, style: DWRITE_FONT_STYLE, size: f32) -> Vec<DWRITE_FONT_AXIS_VALUE> {
    let mut out = Vec::new();
    push_axis(&mut out, DWRITE_FONT_AXIS_TAG_WEIGHT, weight.0 as f32);
    push_axis(&mut out, DWRITE_FONT_AXIS_TAG_WIDTH, width_pct(stretch.0 as u16));
    let (ital, slnt) = match style.0 {
        2 => (1.0, -12.0), // ITALIC
        1 => (0.0, -20.0), // OBLIQUE
        _ => (0.0, 0.0),   // NORMAL
    };
    push_axis(&mut out, DWRITE_FONT_AXIS_TAG_ITALIC, ital);
    push_axis(&mut out, DWRITE_FONT_AXIS_TAG_SLANT, slnt);
    if size > 0.0 {
        push_axis(&mut out, DWRITE_FONT_AXIS_TAG_OPTICAL_SIZE, size * 72.0 / 96.0);
    }
    for v in input {
        push_axis(&mut out, v.axisTag, v.value);
    }
    out
}

/// A font's `fvar` variation axes in table order; empty for a non-variable font.
pub(crate) fn fvar_axes(font: &FontRef) -> Vec<FvarAxis> {
    let Ok(fvar) = font.fvar() else { return Vec::new() };
    let Ok(axes) = fvar.axes() else { return Vec::new() };
    axes.iter()
        .map(|a| FvarAxis {
            // DWRITE packs the four tag bytes little-endian ('wght' → 0x74686777).
            tag: DWRITE_FONT_AXIS_TAG(u32::from_le_bytes(a.axis_tag().to_be_bytes())),
            default: a.default_value().to_f32(),
            min: a.min_value().to_f32(),
            max: a.max_value().to_f32(),
            flags: a.flags(),
            name_id: a.axis_name_id().to_u16(),
        })
        .collect()
}

/// The axes `IDWriteFontResource` reports: the `fvar` axes followed by any of the four standard
/// registered axes (wght/wdth/ital/slnt) absent from `fvar`, each fixed at its metric-derived value
/// (min == max, so non-variable). VM-verified against native CreateFontResource on win11-26100:
/// Segoe UI Variable → [opsz 5–36, wght 300–700, wdth 100, ital 0, slnt 0].
pub(crate) fn resource_axes(font: &FontRef) -> Vec<FvarAxis> {
    let mut axes = fvar_axes(font);
    let (weight, width_class, style) = face_attrs(font);
    let slant = font.post().map(|p| p.italic_angle().to_f32()).unwrap_or(0.0);
    for (tag, val) in [
        (DWRITE_FONT_AXIS_TAG_WEIGHT, weight as f32),
        (DWRITE_FONT_AXIS_TAG_WIDTH, width_pct(width_class)),
        (DWRITE_FONT_AXIS_TAG_ITALIC, (style == 2) as u8 as f32),
        (DWRITE_FONT_AXIS_TAG_SLANT, slant),
    ] {
        if !axes.iter().any(|a| a.tag == tag) {
            axes.push(FvarAxis { tag, default: val, min: val, max: val, flags: 0, name_id: 0 });
        }
    }
    // Native reports the registered axes in a fixed canonical order (opsz, wght, wdth, ital, slnt),
    // ahead of any custom axis (which keep their fvar order) — VM-verified: Segoe UI Variable's fvar
    // is [wght, opsz] but CreateFontResource returns [opsz, wght, wdth, ital, slnt]. Stable-sort by it.
    let rank = |t: DWRITE_FONT_AXIS_TAG| match t {
        DWRITE_FONT_AXIS_TAG_OPTICAL_SIZE => 0,
        DWRITE_FONT_AXIS_TAG_WEIGHT => 1,
        DWRITE_FONT_AXIS_TAG_WIDTH => 2,
        DWRITE_FONT_AXIS_TAG_ITALIC => 3,
        DWRITE_FONT_AXIS_TAG_SLANT => 4,
        _ => 5,
    };
    axes.sort_by_key(|a| rank(a.tag));
    axes
}

/// OS/2 `usWidthClass` (1–9) → width percentage, the `wdth` axis value DirectWrite synthesizes.
fn width_pct(width_class: u16) -> f32 {
    match width_class {
        1 => 50.0,
        2 => 62.5,
        3 => 75.0,
        4 => 87.5,
        6 => 112.5,
        7 => 125.0,
        8 => 150.0,
        9 => 200.0,
        _ => 100.0, // 5 (medium) and any out-of-range value
    }
}

/// The `COLR` table version (0 = layered v0, ≥1 = paint tree v1), or `None` with no `COLR` table.
fn colr_version(font: &FontRef) -> Option<u16> {
    let colr = font.table_data(Tag::new(b"COLR"))?;
    let b = colr.as_bytes();
    (b.len() >= 2).then(|| u16::from_be_bytes([b[0], b[1]]))
}

/// Coalesced Unicode ranges the cmap covers (`IDWriteFont1`/`IDWriteFontFace1::GetUnicodeRanges`):
/// consecutive mapped code points merge into one `[first,last]` range.
fn unicode_ranges(font: &FontRef) -> Vec<DWRITE_UNICODE_RANGE> {
    let mut cps: Vec<u32> = font.charmap().mappings().map(|(c, _)| c).collect();
    cps.sort_unstable();
    cps.dedup();
    let mut out: Vec<DWRITE_UNICODE_RANGE> = Vec::new();
    for c in cps {
        match out.last_mut() {
            Some(r) if r.last + 1 == c => r.last = c,
            _ => out.push(DWRITE_UNICODE_RANGE { first: c, last: c }),
        }
    }
    out
}

/// `GetUnicodeRanges` buffer protocol: set `*count` to the range count; fill `dst` when it is big
/// enough, else return `HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER)` (what native returns so a
/// caller can size its buffer).
fn write_unicode_ranges(font: &FontRef, max: u32, dst: *mut DWRITE_UNICODE_RANGE, count: *mut u32) -> WResult<()> {
    let ranges = unicode_ranges(font);
    unsafe { *count = ranges.len() as u32 };
    if dst.is_null() || (max as usize) < ranges.len() {
        return Err(HRESULT(0x8007_007Au32 as i32).into());
    }
    let out = unsafe { slice::from_raw_parts_mut(dst, ranges.len()) };
    out.copy_from_slice(&ranges);
    Ok(())
}

/// Legacy `kern` table (format 0) pair adjustments for consecutive glyphs — the design-unit values
/// `IDWriteFontFace1::GetKerningPairAdjustments` returns (`adj[i]` kerns glyph `i`→`i+1`; last is 0).
fn kern_adjustments(font: &FontRef, glyphs: &[u16]) -> Vec<i32> {
    let mut out = vec![0i32; glyphs.len()];
    let Some(kern) = font.table_data(Tag::new(b"kern")).and_then(|d| Kern::read(d).ok()) else { return out };
    for i in 0..glyphs.len().saturating_sub(1) {
        let (l, r) = (GlyphId::new(glyphs[i] as u32), GlyphId::new(glyphs[i + 1] as u32));
        for st in kern.subtables().flatten() {
            if st.is_horizontal() && !st.is_cross_stream() {
                if let Ok(SubtableKind::Format0(s0)) = st.kind() {
                    if let Some(v) = s0.kerning(l, r) {
                        out[i] += v;
                    }
                }
            }
        }
    }
    out
}

/// Does the font carry pairwise kerning (`HasKerningPairs`)? True when a `kern` table is present.
fn has_kern(font: &FontRef) -> bool {
    font.table_data(Tag::new(b"kern")).is_some()
}

/// `DWRITE_PANOSE` from the OS/2 10-byte PANOSE classification.
fn panose(font: &FontRef) -> DWRITE_PANOSE {
    let mut p = DWRITE_PANOSE::default();
    if let Ok(os2) = font.os2() {
        for (d, s) in unsafe { p.values.iter_mut() }.zip(os2.panose_10()) {
            *d = *s;
        }
    }
    p
}

// --- Interface constructors (most-derived object, cast to the base a method returns) -----------

/// The font collection as the most-derived `IDWriteFontCollection3` leaf (QI chain exposes 1/2/3).
fn collection3(fonts: Arc<SystemFonts>) -> IDWriteFontCollection3 {
    CollectionObj { fonts }.into()
}
/// The font collection as `IDWriteFontCollection1` (carries `GetFontSet`/`GetFontFamily1`).
pub(crate) fn collection1(fonts: Arc<SystemFonts>) -> IDWriteFontCollection1 {
    collection3(fonts).cast().unwrap()
}
/// The base `IDWriteFontCollection` every family/collection getter returns.
pub(crate) fn collection(fonts: Arc<SystemFonts>) -> IDWriteFontCollection {
    collection1(fonts).cast().unwrap()
}
fn family2(fonts: Arc<SystemFonts>, family: usize) -> IDWriteFontFamily2 {
    FamilyObj { fonts, family }.into()
}
fn family1(fonts: Arc<SystemFonts>, family: usize) -> IDWriteFontFamily1 {
    family2(fonts, family).cast().unwrap()
}
fn font_list2(fonts: Arc<SystemFonts>, entries: Vec<usize>) -> IDWriteFontList2 {
    FontListObj { fonts, entries }.into()
}
fn font_list1(fonts: Arc<SystemFonts>, entries: Vec<usize>) -> IDWriteFontList1 {
    font_list2(fonts, entries).cast().unwrap()
}
fn font3(fonts: Arc<SystemFonts>, entry: usize) -> IDWriteFont3 {
    FontObj { fonts, entry }.into()
}

/// The `(source, faceIndex)` members behind entry indices — the FontSet1 a collection/family/list
/// hands back (`GetFontSet`).
fn entry_refs(fonts: &SystemFonts, entries: impl IntoIterator<Item = usize>) -> Vec<(FontSource, u32)> {
    entries.into_iter().map(|e| (fonts.entries[e].source.clone(), fonts.entries[e].face)).collect()
}

// --- IDWriteFontCollection(1) ------------------------------------------------------------------

#[implement(IDWriteFontCollection3)]
pub(crate) struct CollectionObj {
    pub(crate) fonts: Arc<SystemFonts>,
}

impl IDWriteFontCollection_Impl for CollectionObj_Impl {
    fn GetFontFamilyCount(&self) -> u32 {
        self.fonts.families.len() as u32
    }
    fn GetFontFamily(&self, index: u32) -> WResult<IDWriteFontFamily> {
        if (index as usize) < self.fonts.families.len() {
            Ok(family1(self.fonts.clone(), index as usize).cast()?)
        } else {
            Err(E_NOTIMPL.into())
        }
    }
    fn FindFamilyName(&self, familyname: &PCWSTR, index: *mut u32, exists: *mut BOOL) -> WResult<()> {
        let name = super::wide_to_string(familyname.0, None);
        match self.fonts.find_family(&name) {
            Some(i) => unsafe {
                *index = i as u32;
                *exists = true.into();
            },
            None => unsafe {
                *index = 0;
                *exists = false.into();
            },
        }
        Ok(())
    }
    fn GetFontFromFontFace(&self, fontface: Ref<IDWriteFontFace>) -> WResult<IDWriteFont> {
        // Our own faces carry their canonical family + face index; map back to the owning entry.
        let fo = fontface.as_ref().ok_or(E_INVALIDARG)?.cast_object_ref::<FaceObj>().map_err(|_| windows_core::Error::from(E_INVALIDARG))?;
        let fam = self.fonts.find_family(&fo.family).ok_or(E_INVALIDARG)?;
        let entry = *self.fonts.families[fam].faces.iter().find(|&&e| self.fonts.entries[e].face == fo.index).ok_or(E_INVALIDARG)?;
        Ok(font3(self.fonts.clone(), entry).cast()?)
    }
}

impl IDWriteFontCollection1_Impl for CollectionObj_Impl {
    fn GetFontSet(&self) -> WResult<IDWriteFontSet> {
        Ok(super::fontset::font_set(self.fonts.face_refs()))
    }
    fn GetFontFamily(&self, index: u32) -> WResult<IDWriteFontFamily1> {
        if (index as usize) < self.fonts.families.len() {
            Ok(family1(self.fonts.clone(), index as usize))
        } else {
            Err(E_NOTIMPL.into())
        }
    }
}

impl IDWriteFontCollection2_Impl for CollectionObj_Impl {
    fn GetFontFamily(&self, index: u32) -> WResult<IDWriteFontFamily2> {
        if (index as usize) < self.fonts.families.len() {
            Ok(family2(self.fonts.clone(), index as usize))
        } else {
            Err(E_NOTIMPL.into())
        }
    }
    fn GetMatchingFonts(&self, familyname: &PCWSTR, _fontaxisvalues: *const DWRITE_FONT_AXIS_VALUE, _fontaxisvaluecount: u32) -> WResult<IDWriteFontList2> {
        // ponytail: match by family name (axis-value specialisation needs instancing). An unknown
        // family yields an empty list, as native does for a name the collection does not carry.
        let name = super::wide_to_string(familyname.0, None);
        let entries = self.fonts.find_family(&name).map(|f| self.fonts.families[f].faces.clone()).unwrap_or_default();
        Ok(font_list2(self.fonts.clone(), entries))
    }
    fn GetFontFamilyModel(&self) -> DWRITE_FONT_FAMILY_MODEL {
        if self.fonts.typographic {
            DWRITE_FONT_FAMILY_MODEL_TYPOGRAPHIC
        } else {
            DWRITE_FONT_FAMILY_MODEL_WEIGHT_STRETCH_STYLE
        }
    }
    fn GetFontSet(&self) -> WResult<IDWriteFontSet1> {
        Ok(super::fontset::font_set(self.fonts.face_refs()).cast()?)
    }
}

impl IDWriteFontCollection3_Impl for CollectionObj_Impl {
    fn GetExpirationEvent(&self) -> HANDLE {
        expiration_event()
    }
}

/// The `GetExpirationEvent` handle for a system font set / collection. Native hands back a real
/// (never-signaled) event even for a set that does not change — a non-null handle, VM-verified —
/// so a drop-in must too; ours is a process-wide manual-reset event that
/// never fires (our scan is static). Off Windows there are no OS handles, so it is null there.
#[cfg(windows)]
pub(crate) fn expiration_event() -> HANDLE {
    use std::sync::OnceLock;
    // kernel32 directly — the shim windows crate dwrite-ffi builds against omits System::Threading.
    unsafe extern "system" {
        fn CreateEventW(attrs: *const core::ffi::c_void, manual_reset: i32, initial_state: i32, name: *const u16) -> *mut core::ffi::c_void;
    }
    static EVENT: OnceLock<usize> = OnceLock::new();
    let h = *EVENT.get_or_init(|| unsafe { CreateEventW(core::ptr::null(), 1, 0, core::ptr::null()) as usize });
    HANDLE(h as *mut core::ffi::c_void)
}
#[cfg(not(windows))]
pub(crate) fn expiration_event() -> HANDLE {
    HANDLE(core::ptr::null_mut())
}

// --- IDWriteFontFamily(1/2) (over IDWriteFontList) --------------------------------------------

#[implement(IDWriteFontFamily2)]
pub(crate) struct FamilyObj {
    fonts: Arc<SystemFonts>,
    family: usize,
}

impl IDWriteFontList_Impl for FamilyObj_Impl {
    fn GetFontCollection(&self) -> WResult<IDWriteFontCollection> {
        Ok(collection(self.fonts.clone()))
    }
    fn GetFontCount(&self) -> u32 {
        self.fonts.families[self.family].faces.len() as u32
    }
    fn GetFont(&self, index: u32) -> WResult<IDWriteFont> {
        let e = *self.fonts.families[self.family].faces.get(index as usize).ok_or(E_NOTIMPL)?;
        Ok(font3(self.fonts.clone(), e).cast()?)
    }
}

impl IDWriteFontFamily_Impl for FamilyObj_Impl {
    fn GetFamilyNames(&self) -> WResult<IDWriteLocalizedStrings> {
        Ok(localized(self.fonts.families[self.family].name.clone()))
    }
    fn GetFirstMatchingFont(&self, weight: DWRITE_FONT_WEIGHT, stretch: DWRITE_FONT_STRETCH, style: DWRITE_FONT_STYLE) -> WResult<IDWriteFont> {
        let e = self.fonts.match_entry(self.family, weight.0 as u16, stretch.0 as u16, style.0 as u8).ok_or(E_NOTIMPL)?;
        Ok(font3(self.fonts.clone(), e).cast()?)
    }
    fn GetMatchingFonts(&self, weight: DWRITE_FONT_WEIGHT, _stretch: DWRITE_FONT_STRETCH, style: DWRITE_FONT_STYLE) -> WResult<IDWriteFontList> {
        // The family's faces ordered by match to the requested style/weight (the same nearest rule
        // as `match_entry`): italic match first, then weight distance.
        // GetMatchingFonts lists the family's real faces ranked by match (native excludes the
        // simulated faces here; GetFirstMatchingFont still returns a simulated face when asked).
        let italic = style.0 != 0;
        let mut entries: Vec<usize> = self.fonts.families[self.family].faces.iter().copied().filter(|&e| self.fonts.entries[e].sims == 0).collect();
        entries.sort_by_key(|&e| ((self.fonts.entries[e].italic() != italic) as u32, (self.fonts.entries[e].weight as i32 - weight.0).unsigned_abs()));
        Ok(font_list1(self.fonts.clone(), entries).cast()?)
    }
}

impl IDWriteFontFamily1_Impl for FamilyObj_Impl {
    fn GetFontLocality(&self, _index: u32) -> DWRITE_LOCALITY {
        DWRITE_LOCALITY_LOCAL
    }
    fn GetFont(&self, index: u32) -> WResult<IDWriteFont3> {
        let e = *self.fonts.families[self.family].faces.get(index as usize).ok_or(E_NOTIMPL)?;
        Ok(font3(self.fonts.clone(), e))
    }
    fn GetFontFaceReference(&self, index: u32) -> WResult<IDWriteFontFaceReference> {
        let e = *self.fonts.families[self.family].faces.get(index as usize).ok_or(E_NOTIMPL)?;
        let ent = &self.fonts.entries[e];
        Ok(super::fontfile::faceref(ent.source.read().ok_or(E_NOTIMPL)?, ent.face, DWRITE_FONT_SIMULATIONS_NONE, ent.source.path_str()))
    }
}

impl IDWriteFontFamily2_Impl for FamilyObj_Impl {
    fn GetMatchingFonts(&self, _fontaxisvalues: *const DWRITE_FONT_AXIS_VALUE, _fontaxisvaluecount: u32) -> WResult<IDWriteFontList2> {
        // ponytail: axis-value matching needs instancing; return the family's real (non-simulated)
        // faces, the set an axis query filters from.
        let entries: Vec<usize> = self.fonts.families[self.family].faces.iter().copied().filter(|&e| self.fonts.entries[e].sims == 0).collect();
        Ok(font_list2(self.fonts.clone(), entries))
    }
    fn GetFontSet(&self) -> WResult<IDWriteFontSet1> {
        let refs = entry_refs(&self.fonts, self.fonts.families[self.family].faces.iter().copied());
        Ok(super::fontset::font_set(refs).cast()?)
    }
}

// --- IDWriteFontList(1/2) (GetMatchingFonts result) -------------------------------------------

#[implement(IDWriteFontList2)]
pub(crate) struct FontListObj {
    fonts: Arc<SystemFonts>,
    entries: Vec<usize>,
}

impl IDWriteFontList_Impl for FontListObj_Impl {
    fn GetFontCollection(&self) -> WResult<IDWriteFontCollection> {
        Ok(collection(self.fonts.clone()))
    }
    fn GetFontCount(&self) -> u32 {
        self.entries.len() as u32
    }
    fn GetFont(&self, index: u32) -> WResult<IDWriteFont> {
        let entry = *self.entries.get(index as usize).ok_or(E_INVALIDARG)?;
        Ok(font3(self.fonts.clone(), entry).cast()?)
    }
}

impl IDWriteFontList1_Impl for FontListObj_Impl {
    fn GetFontLocality(&self, _listindex: u32) -> DWRITE_LOCALITY {
        // Every face in a locally-scanned collection is present on disk.
        DWRITE_LOCALITY_LOCAL
    }
    fn GetFont(&self, listindex: u32) -> WResult<IDWriteFont3> {
        let entry = *self.entries.get(listindex as usize).ok_or(E_INVALIDARG)?;
        Ok(font3(self.fonts.clone(), entry))
    }
    fn GetFontFaceReference(&self, listindex: u32) -> WResult<IDWriteFontFaceReference> {
        let entry = *self.entries.get(listindex as usize).ok_or(E_INVALIDARG)?;
        let ent = &self.fonts.entries[entry];
        Ok(super::fontfile::faceref(ent.source.read().ok_or(E_NOTIMPL)?, ent.face, DWRITE_FONT_SIMULATIONS_NONE, ent.source.path_str()))
    }
}

impl IDWriteFontList2_Impl for FontListObj_Impl {
    fn GetFontSet(&self) -> WResult<IDWriteFontSet1> {
        let refs = entry_refs(&self.fonts, self.entries.iter().copied());
        Ok(super::fontset::font_set(refs).cast()?)
    }
}

// --- IDWriteFont(1/2/3) ------------------------------------------------------------------------

#[implement(IDWriteFont3)]
pub(crate) struct FontObj {
    fonts: Arc<SystemFonts>,
    entry: usize,
}

impl FontObj {
    /// Parse this font's file and run `f` over the face (for cmap / subfamily queries). `None` if the
    /// file can't be read/parsed. ponytail: re-reads the file per call — fine for the few
    /// metadata queries a D2D app makes; cache bytes in the entry if a hot path appears.
    fn with_font<R>(&self, f: impl FnOnce(&FontRef) -> R) -> Option<R> {
        let ent = &self.fonts.entries[self.entry];
        let bytes = ent.source.read()?;
        Some(f(&FontRef::from_index(&bytes, ent.face).ok()?))
    }
}

impl IDWriteFont_Impl for FontObj_Impl {
    fn GetFontFamily(&self) -> WResult<IDWriteFontFamily> {
        let family = self.fonts.family_of(self.entry).ok_or(E_INVALIDARG)?;
        Ok(family1(self.fonts.clone(), family).cast()?)
    }
    fn GetWeight(&self) -> DWRITE_FONT_WEIGHT {
        DWRITE_FONT_WEIGHT(self.fonts.entries[self.entry].weight as i32)
    }
    fn GetStretch(&self) -> DWRITE_FONT_STRETCH {
        DWRITE_FONT_STRETCH(self.fonts.entries[self.entry].stretch as i32)
    }
    fn GetStyle(&self) -> DWRITE_FONT_STYLE {
        DWRITE_FONT_STYLE(self.fonts.entries[self.entry].style as i32)
    }
    fn IsSymbolFont(&self) -> BOOL {
        false.into()
    }
    fn GetFaceNames(&self) -> WResult<IDWriteLocalizedStrings> {
        // All localized subfamily-name records (native enumerates every locale, not just en-US).
        let e = self.with_font(|f| localized_names(f, StringId::SUBFAMILY_NAME)).unwrap_or_default();
        Ok(StringsObj { entries: if e.is_empty() { vec![("en-us".into(), "Regular".into())] } else { e } }.into())
    }
    fn GetInformationalStrings(&self, id: DWRITE_INFORMATIONAL_STRING_ID, s: windows_core::OutRef<IDWriteLocalizedStrings>, exists: *mut BOOL) -> WResult<()> {
        let e = info_string_id(id).map(|sid| self.with_font(|f| localized_names(f, sid)).unwrap_or_default()).unwrap_or_default();
        unsafe { *exists = (!e.is_empty()).into() };
        if e.is_empty() {
            return Ok(());
        }
        s.write(Some(StringsObj { entries: e }.into()))
    }
    fn GetSimulations(&self) -> DWRITE_FONT_SIMULATIONS {
        DWRITE_FONT_SIMULATIONS(self.fonts.entries[self.entry].sims as i32)
    }
    fn GetMetrics(&self, fontmetrics: *mut DWRITE_FONT_METRICS) {
        if let Some(m) = self.with_font(design_metrics) {
            unsafe { *fontmetrics = m };
        }
    }
    fn HasCharacter(&self, unicodevalue: u32) -> WResult<BOOL> {
        let has = self.with_font(|f| char::from_u32(unicodevalue).and_then(|c| f.charmap().map(c)).is_some()).unwrap_or(false);
        Ok(has.into())
    }
    fn CreateFontFace(&self) -> WResult<IDWriteFontFace> {
        let ent = &self.fonts.entries[self.entry];
        let bytes = ent.source.read().ok_or(E_NOTIMPL)?;
        let face: IDWriteFontFace7 = FaceObj { bytes, index: ent.face, family: ent.family.clone(), source: ent.source.clone(), sims: ent.sims, axis_values: ent.coords.clone() }.into();
        Ok(face.cast()?)
    }
}

impl IDWriteFont1_Impl for FontObj_Impl {
    fn GetMetrics(&self, fontmetrics: *mut DWRITE_FONT_METRICS1) {
        if let Some(m) = self.with_font(design_metrics1) {
            unsafe { *fontmetrics = m };
        }
    }
    fn GetPanose(&self, panose_out: *mut DWRITE_PANOSE) {
        if let Some(p) = self.with_font(panose) {
            unsafe { *panose_out = p };
        }
    }
    fn GetUnicodeRanges(&self, maxrangecount: u32, unicoderanges: *mut DWRITE_UNICODE_RANGE, actualrangecount: *mut u32) -> WResult<()> {
        match self.with_font(|f| write_unicode_ranges(f, maxrangecount, unicoderanges, actualrangecount)) {
            Some(r) => r,
            None => {
                unsafe { *actualrangecount = 0 };
                Ok(())
            }
        }
    }
    fn IsMonospacedFont(&self) -> BOOL {
        self.with_font(|f| f.metrics(Size::unscaled(), LocationRef::default()).is_monospace).unwrap_or(false).into()
    }
}

impl IDWriteFont2_Impl for FontObj_Impl {
    fn IsColorFont(&self) -> BOOL {
        self.with_font(|f| f.table_data(skrifa::Tag::new(b"COLR")).is_some()).unwrap_or(false).into()
    }
}

impl IDWriteFont3_Impl for FontObj_Impl {
    fn CreateFontFace(&self) -> WResult<IDWriteFontFace3> {
        let ent = &self.fonts.entries[self.entry];
        let f: IDWriteFontFace7 = FaceObj { bytes: ent.source.read().ok_or(E_NOTIMPL)?, index: ent.face, family: ent.family.clone(), source: ent.source.clone(), sims: ent.sims, axis_values: ent.coords.clone() }.into();
        Ok(f.cast()?)
    }
    fn Equals(&self, font: Ref<IDWriteFont>) -> BOOL {
        font.as_ref().and_then(|f| f.cast_object_ref::<FontObj>().ok()).is_some_and(|o| Arc::ptr_eq(&o.fonts, &self.fonts) && o.entry == self.entry).into()
    }
    fn GetFontFaceReference(&self) -> WResult<IDWriteFontFaceReference> {
        let ent = &self.fonts.entries[self.entry];
        Ok(super::fontfile::faceref(ent.source.read().ok_or(E_NOTIMPL)?, ent.face, DWRITE_FONT_SIMULATIONS_NONE, ent.source.path_str()))
    }
    fn HasCharacter(&self, unicodevalue: u32) -> BOOL {
        self.with_font(|f| char::from_u32(unicodevalue).and_then(|c| f.charmap().map(c)).is_some()).unwrap_or(false).into()
    }
    fn GetLocality(&self) -> DWRITE_LOCALITY {
        DWRITE_LOCALITY_LOCAL
    }
}

// --- IDWriteFontFace (up to FontFace7) + ITextInterop ------------------------------------------

#[implement(IDWriteFontFace7, ITextInterop)]
pub(crate) struct FaceObj {
    pub(crate) bytes: Arc<Vec<u8>>,
    pub(crate) index: u32,
    pub(crate) family: String,
    /// Where the face's bytes came from, so `GetFiles` hands back native's form: a path-keyed local
    /// loader for a file-backed face, an in-memory loader for a memory-backed one. [`FontSource`].
    pub(crate) source: FontSource,
    /// `DWRITE_FONT_SIMULATIONS` this face carries (a bold/oblique-simulated collection face).
    pub(crate) sims: u16,
    /// For an fvar named-instance face, the instance's user axis values (empty = default instance /
    /// static face). `GetFontAxisValues` reports these, and the render path instances the outline.
    pub(crate) axis_values: Vec<DWRITE_FONT_AXIS_VALUE>,
}

impl FaceObj {
    fn font(&self) -> Option<FontRef<'_>> {
        FontRef::from_index(&self.bytes, self.index).ok()
    }
    /// Whether the face's cmap maps `cp` to a non-`.notdef` glyph (`HasCharacter`/`IsCharacterLocal`).
    fn covers(&self, cp: u32) -> bool {
        self.font().and_then(|f| char::from_u32(cp).and_then(|c| f.charmap().map(c))).is_some()
    }
    /// The subfamily (style) name — "Regular"/"Bold"/… — for `GetFaceNames`.
    fn subfamily(&self) -> String {
        self.font()
            .and_then(|f| f.localized_strings(StringId::SUBFAMILY_NAME).english_or_first().map(|s| s.to_string()))
            .unwrap_or_else(|| "Regular".into())
    }
    /// This face's `fvar` variation axes (empty for a non-variable font). [`fvar_axes`].
    fn fvar_axes(&self) -> Vec<FvarAxis> {
        self.font().map(|f| fvar_axes(&f)).unwrap_or_default()
    }
    /// The per-face axis values `IDWriteFontFace5::GetFontAxisValues` reports. A named-instance face
    /// reports its instance coordinates; otherwise the fvar defaults / static synthesis.
    /// [`face_axis_values`].
    fn face_axis_values(&self) -> Vec<DWRITE_FONT_AXIS_VALUE> {
        if !self.axis_values.is_empty() {
            return self.axis_values.clone();
        }
        self.font().map(|f| face_axis_values(&f)).unwrap_or_default()
    }
    /// Family/face names in the requested [`DWRITE_FONT_FAMILY_MODEL`] (`IDWriteFontFace6`): the
    /// typographic model prefers the typographic `name` ids, the weight-stretch-style model the
    /// WIN32 ids; either falls back to the other id, then to `default`.
    fn model_names(&self, primary: StringId, fallback: StringId, default: &str) -> WResult<IDWriteLocalizedStrings> {
        let e = self
            .font()
            .map(|f| {
                let p = localized_names(&f, primary);
                if p.is_empty() { localized_names(&f, fallback) } else { p }
            })
            .unwrap_or_default();
        Ok(StringsObj { entries: if e.is_empty() { vec![("en-us".into(), default.to_string())] } else { e } }.into())
    }
}

impl ITextInterop_Impl for FaceObj_Impl {
    unsafe fn FontRasterizer(&self, rasterizer: *mut u32) -> HRESULT {
        if rasterizer.is_null() { return E_INVALIDARG; }
        unsafe { *rasterizer = (self.source.rasterizer() == d2d::text::hinting::Rasterizer::FontCache) as u32; }
        HRESULT(0)
    }
    unsafe fn FontBlob(&self, data: *mut *const u8, len: *mut usize, index: *mut u32) -> HRESULT {
        unsafe {
            *data = self.bytes.as_ptr();
            *len = self.bytes.len();
            *index = self.index;
        }
        HRESULT(0)
    }
    unsafe fn FormatParams(&self, _em: *mut f32, _align: *mut u32, _wrap: *mut u32, _base_rtl: *mut i32, _sims: *mut u32) -> HRESULT {
        E_NOTIMPL
    }
}

impl IDWriteFontFace_Impl for FaceObj_Impl {
    fn GetType(&self) -> DWRITE_FONT_FACE_TYPE {
        if self.bytes.starts_with(b"ttcf") {
            DWRITE_FONT_FACE_TYPE_OPENTYPE_COLLECTION
        } else if self.font().is_some_and(|f| f.table_data(Tag::new(b"CFF ")).is_some()) {
            DWRITE_FONT_FACE_TYPE_CFF
        } else {
            DWRITE_FONT_FACE_TYPE_TRUETYPE
        }
    }
    fn GetFiles(&self, numberoffiles: *mut u32, fontfiles: windows_core::OutRef<IDWriteFontFile>) -> WResult<()> {
        // One file per face (how `CreateFontFace` builds us). The count-only probe passes a null array.
        unsafe { *numberoffiles = 1 };
        if !fontfiles.is_null() {
            // File-backed ⇒ path key + local loader (reads disk); memory-backed ⇒ in-memory loader
            // serving the stored bytes. Both round-trip GetReferenceKey→CreateStreamFromKey→ReadFile.
            let file = match &self.source {
                FontSource::Path(p) => super::fontfile::file_ref_path(self.bytes.clone(), p),
                FontSource::Bytes(_) => super::fontfile::file_from_bytes(self.bytes.clone()),
            };
            fontfiles.write(Some(file))?;
        }
        Ok(())
    }
    fn GetIndex(&self) -> u32 {
        self.index
    }
    fn GetSimulations(&self) -> DWRITE_FONT_SIMULATIONS {
        DWRITE_FONT_SIMULATIONS(self.sims as i32)
    }
    fn IsSymbolFont(&self) -> BOOL {
        false.into()
    }
    fn GetMetrics(&self, fontfacemetrics: *mut DWRITE_FONT_METRICS) {
        let m = self.font().map(|f| design_metrics(&f)).unwrap_or(DWRITE_FONT_METRICS { designUnitsPerEm: 2048, ..Default::default() });
        unsafe { *fontfacemetrics = m };
    }
    fn GetGlyphCount(&self) -> u16 {
        self.font().map(|f| f.metrics(Size::unscaled(), LocationRef::default()).glyph_count).unwrap_or(0)
    }
    fn GetDesignGlyphMetrics(&self, glyphindices: *const u16, glyphcount: u32, glyphmetrics: *mut DWRITE_GLYPH_METRICS, _issideways: BOOL) -> WResult<()> {
        let Some(font) = self.font() else { return Err(E_NOTIMPL.into()) };
        let gm = font.glyph_metrics(Size::unscaled(), LocationRef::default());
        // Bold simulation widens each outline, so native's GetDesignGlyphMetrics adds a constant
        // design-unit advance delta = round(upem/50) (0.02 em: 41 @2048, 82 @4096, VM-verified) to
        // every glyph that HAS an outline; an empty glyph (e.g. the space) keeps its advance.
        let bold_delta = if self.sims & DWRITE_FONT_SIMULATIONS_BOLD.0 as u16 != 0 { (design_metrics(&font).designUnitsPerEm as f32 / 50.0).round() as i64 } else { 0 };
        let ids = unsafe { slice::from_raw_parts(glyphindices, glyphcount as usize) };
        let dst = unsafe { slice::from_raw_parts_mut(glyphmetrics, glyphcount as usize) };
        for (d, &g) in dst.iter_mut().zip(ids) {
            let gid = GlyphId::new(g as u32);
            let aw = gm.advance_width(gid).unwrap_or(0.0).round() as i64;
            let delta = if bold_delta != 0 && gm.bounds(gid).is_some() { bold_delta } else { 0 };
            *d = DWRITE_GLYPH_METRICS { advanceWidth: (aw + delta).max(0) as u32, ..Default::default() };
        }
        Ok(())
    }
    fn GetGlyphIndices(&self, codepoints: *const u32, codepointcount: u32, glyphindices: *mut u16) -> WResult<()> {
        let Some(font) = self.font() else { return Err(E_NOTIMPL.into()) };
        let cmap = font.charmap();
        let cps = unsafe { slice::from_raw_parts(codepoints, codepointcount as usize) };
        let dst = unsafe { slice::from_raw_parts_mut(glyphindices, codepointcount as usize) };
        for (d, &cp) in dst.iter_mut().zip(cps) {
            *d = char::from_u32(cp).and_then(|c| cmap.map(c)).map(|g| g.to_u32() as u16).unwrap_or(0);
        }
        Ok(())
    }
    fn TryGetFontTable(&self, tag: u32, tabledata: *mut *mut core::ffi::c_void, tablesize: *mut u32, tablecontext: *mut *mut core::ffi::c_void, exists: *mut BOOL) -> WResult<()> {
        // The DWRITE tag is the 4 chars in memory order, so `to_le_bytes` gives skrifa's tag bytes.
        // The slice borrows `self.bytes` (a stable Arc buffer), so the raw pointer stays valid until
        // the face dies; `ReleaseFontTable` has nothing to free (no context handle).
        let found = self.font().and_then(|f| f.table_data(Tag::new(&tag.to_le_bytes())).map(|d| {
            let b = d.as_bytes();
            (b.as_ptr(), b.len())
        }));
        unsafe {
            match found {
                Some((ptr, len)) => {
                    *tabledata = ptr as *mut core::ffi::c_void;
                    *tablesize = len as u32;
                    *exists = true.into();
                }
                None => {
                    *tabledata = core::ptr::null_mut();
                    *tablesize = 0;
                    *exists = false.into();
                }
            }
            *tablecontext = core::ptr::null_mut();
        }
        Ok(())
    }
    // Legitimate no-op: TryGetFontTable returns a slice of our owned font bytes with a null table
    // context (no allocation), so there is nothing to free — mirroring native freeing its context.
    fn ReleaseFontTable(&self, _tablecontext: *const core::ffi::c_void) {}
    fn GetGlyphRunOutline(&self, emsize: f32, glyphindices: *const u16, glyphadvances: *const f32, glyphoffsets: *const DWRITE_GLYPH_OFFSET, glyphcount: u32, issideways: BOOL, isrighttoleft: BOOL, geometrysink: Ref<ID2D1SimplifiedGeometrySink>) -> WResult<()> {
        let sink = geometrysink.as_ref().ok_or(windows_core::Error::from(E_INVALIDARG))?;
        if !emsize.is_finite() || emsize < 0.0 || (glyphcount != 0 && glyphindices.is_null()) { return Err(E_INVALIDARG.into()); }
        let font = self.font().ok_or(windows_core::Error::from(E_INVALIDARG))?;
        let n = glyphcount as usize;
        let ids = if n == 0 { &[] } else { unsafe { slice::from_raw_parts(glyphindices, n) } };
        let advances = if glyphadvances.is_null() { None } else { Some(unsafe { slice::from_raw_parts(glyphadvances, n) }) };
        let offsets = if glyphoffsets.is_null() { None } else { Some(unsafe { slice::from_raw_parts(glyphoffsets, n) }) };
        super::outline::emit(&font, emsize, ids, advances, offsets, issideways.as_bool(), isrighttoleft.as_bool(), self.sims, sink)
    }
    fn GetRecommendedRenderingMode(&self, emsize: f32, pixelsperdip: f32, measuringmode: DWRITE_MEASURING_MODE, renderingparams: Ref<IDWriteRenderingParams>) -> WResult<DWRITE_RENDERING_MODE> {
        let pm = renderingparams.as_ref().map_or(0, params_mode);
        Ok(DWRITE_RENDERING_MODE(legacy_recommend(recommend(&self.bytes, self.index, emsize * pixelsperdip, measuringmode.0 as u32, 2, pm, [96.0 * pixelsperdip; 2])) as i32))
    }
    fn GetGdiCompatibleMetrics(&self, emsize: f32, pixelsperdip: f32, transform: *const DWRITE_MATRIX, fontfacemetrics: *mut DWRITE_FONT_METRICS) -> WResult<()> {
        let Some(font) = self.font() else { return Err(E_INVALIDARG.into()) };
        unsafe { *fontfacemetrics = gdi_compat_metrics(&font, emsize, pixelsperdip, transform) };
        Ok(())
    }
    fn GetGdiCompatibleGlyphMetrics(&self, emsize: f32, pixelsperdip: f32, transform: *const DWRITE_MATRIX, usegdinatural: BOOL, glyphindices: *const u16, glyphcount: u32, glyphmetrics: *mut DWRITE_GLYPH_METRICS, _issideways: BOOL) -> WResult<()> {
        // Hinted whole-pixel advance (design units) per glyph. ponytail: only advanceWidth is filled
        // (what any scene reads), like GetDesignGlyphMetrics; the hinted bounding-box fields stay 0.
        let Some(font) = self.font() else { return Err(E_INVALIDARG.into()) };
        let upem = design_metrics(&font).designUnitsPerEm;
        let classic = !usegdinatural.as_bool();
        let xf = unsafe { transform.as_ref() }.map(|t| [t.m11, t.m12, t.m21, t.m22]);
        let bold = self.sims & DWRITE_FONT_SIMULATIONS_BOLD.0 as u16 != 0;
        let ids = unsafe { slice::from_raw_parts(glyphindices, glyphcount as usize) };
        let dst = unsafe { slice::from_raw_parts_mut(glyphmetrics, glyphcount as usize) };
        for (d, &g) in dst.iter_mut().zip(ids) {
            // GDI bold +1px applies to every nonzero-advance glyph (the space included); the gate is in gdi_compat_advance_du.
            let aw = d2d::text::gdi::gdi_compat_advance_du(&font, g, emsize, pixelsperdip, classic, xf, upem, bold).unwrap_or(0);
            *d = DWRITE_GLYPH_METRICS { advanceWidth: aw as u32, ..Default::default() };
        }
        Ok(())
    }
}

/// `GetGdiCompatibleMetrics`: design metrics with ascent/descent replaced by whole device pixels at the
/// float device em (`emSize·pixelsPerDip·transformScale`) mapped back to design units: the VDMX record's
/// below 255 ppem, else the design ascent and descent rounded to pixels (VM: Segoe UI Emoji, which has no
/// VDMX, reports 2191 / 510 at device em 40.19). The line gap is dropped.
fn gdi_compat_metrics(font: &FontRef, emsize: f32, ppd: f32, transform: *const DWRITE_MATRIX) -> DWRITE_FONT_METRICS {
    use d2d::text::gdi::round_half_up;
    let mut m = design_metrics(font);
    let ts = unsafe { transform.as_ref() }.map(|t| t.m11.abs().max(t.m22.abs())).unwrap_or(1.0);
    let dev = emsize * ppd * ts;
    let upem = m.designUnitsPerEm as f32;
    let (asc, desc) = d2d::text::gdi::gdi_compat_vmetrics(font, dev, m.designUnitsPerEm).unwrap_or_else(|| {
        let du = |v: u16| round_half_up(round_half_up(v as f32 / (upem / dev)) as f32 * (upem / dev)) as u16;
        (du(m.ascent), du(m.descent))
    });
    m.ascent = asc;
    m.descent = desc;
    m.lineGap = 0;
    m
}

impl IDWriteFontFace1_Impl for FaceObj_Impl {
    fn GetMetrics(&self, fontmetrics: *mut DWRITE_FONT_METRICS1) {
        let m = self.font().map(|f| design_metrics1(&f)).unwrap_or_default();
        unsafe { *fontmetrics = m };
    }
    fn GetGdiCompatibleMetrics(&self, emsize: f32, pixelsperdip: f32, transform: *const DWRITE_MATRIX, fontmetrics: *mut DWRITE_FONT_METRICS1) -> WResult<()> {
        let Some(font) = self.font() else { return Err(E_INVALIDARG.into()) };
        let mut m = design_metrics1(&font);
        m.Base = gdi_compat_metrics(&font, emsize, pixelsperdip, transform);
        unsafe { *fontmetrics = m };
        Ok(())
    }
    fn GetCaretMetrics(&self, caretmetrics: *mut DWRITE_CARET_METRICS) {
        let c = self.font().and_then(|f| f.hhea().ok().map(|h| DWRITE_CARET_METRICS { slopeRise: h.caret_slope_rise(), slopeRun: h.caret_slope_run(), offset: h.caret_offset() })).unwrap_or_default();
        unsafe { *caretmetrics = c };
    }
    fn GetUnicodeRanges(&self, maxrangecount: u32, unicoderanges: *mut DWRITE_UNICODE_RANGE, actualrangecount: *mut u32) -> WResult<()> {
        let Some(font) = self.font() else { return Err(E_INVALIDARG.into()) };
        write_unicode_ranges(&font, maxrangecount, unicoderanges, actualrangecount)
    }
    fn IsMonospacedFont(&self) -> BOOL {
        self.font().map(|f| f.metrics(Size::unscaled(), LocationRef::default()).is_monospace).unwrap_or(false).into()
    }
    fn GetDesignGlyphAdvances(&self, glyphcount: u32, glyphindices: *const u16, glyphadvances: *mut i32, _issideways: BOOL) -> WResult<()> {
        let Some(font) = self.font() else { return Err(E_INVALIDARG.into()) };
        let gm = font.glyph_metrics(Size::unscaled(), LocationRef::default());
        let bold_delta = if self.sims & DWRITE_FONT_SIMULATIONS_BOLD.0 as u16 != 0 { (design_metrics(&font).designUnitsPerEm as f32 / 50.0).round() as i32 } else { 0 };
        let ids = unsafe { slice::from_raw_parts(glyphindices, glyphcount as usize) };
        let dst = unsafe { slice::from_raw_parts_mut(glyphadvances, glyphcount as usize) };
        for (d, &g) in dst.iter_mut().zip(ids) {
            let gid = GlyphId::new(g as u32);
            let delta = if bold_delta != 0 && gm.bounds(gid).is_some() { bold_delta } else { 0 };
            *d = gm.advance_width(gid).unwrap_or(0.0).round() as i32 + delta;
        }
        Ok(())
    }
    fn GetGdiCompatibleGlyphAdvances(&self, emsize: f32, pixelsperdip: f32, transform: *const DWRITE_MATRIX, usegdinatural: BOOL, _issideways: BOOL, glyphcount: u32, glyphindices: *const u16, glyphadvances: *mut i32) -> WResult<()> {
        // Hinted whole-pixel advance per glyph, in design units (the core hinter, via gdi).
        let Some(font) = self.font() else { return Err(E_INVALIDARG.into()) };
        let upem = design_metrics(&font).designUnitsPerEm;
        let classic = !usegdinatural.as_bool();
        let xf = unsafe { transform.as_ref() }.map(|t| [t.m11, t.m12, t.m21, t.m22]);
        let bold = self.sims & DWRITE_FONT_SIMULATIONS_BOLD.0 as u16 != 0;
        let ids = unsafe { slice::from_raw_parts(glyphindices, glyphcount as usize) };
        let dst = unsafe { slice::from_raw_parts_mut(glyphadvances, glyphcount as usize) };
        for (d, &g) in dst.iter_mut().zip(ids) {
            // GDI bold +1px applies to every nonzero-advance glyph (the space included); the gate is in gdi_compat_advance_du.
            *d = d2d::text::gdi::gdi_compat_advance_du(&font, g, emsize, pixelsperdip, classic, xf, upem, bold).unwrap_or(0);
        }
        Ok(())
    }
    fn GetKerningPairAdjustments(&self, glyphcount: u32, glyphindices: *const u16, glyphadvanceadjustments: *mut i32) -> WResult<()> {
        let Some(font) = self.font() else { return Err(E_INVALIDARG.into()) };
        let ids = unsafe { slice::from_raw_parts(glyphindices, glyphcount as usize) };
        let dst = unsafe { slice::from_raw_parts_mut(glyphadvanceadjustments, glyphcount as usize) };
        dst.copy_from_slice(&kern_adjustments(&font, ids));
        Ok(())
    }
    fn HasKerningPairs(&self) -> BOOL {
        self.font().map(|f| has_kern(&f)).unwrap_or(false).into()
    }
    fn GetRecommendedRenderingMode(&self, fontemsize: f32, dpix: f32, dpiy: f32, transform: *const DWRITE_MATRIX, _issideways: BOOL, outlinethreshold: DWRITE_OUTLINE_THRESHOLD, measuringmode: DWRITE_MEASURING_MODE) -> WResult<DWRITE_RENDERING_MODE> {
        Ok(DWRITE_RENDERING_MODE(legacy_recommend(recommend(&self.bytes, self.index, device_em(fontemsize, dpix, transform), measuringmode.0 as u32, aa_for_threshold(outlinethreshold), 0, [dpix, dpiy])) as i32))
    }
    fn GetVerticalGlyphVariants(&self, _glyphcount: u32, _nominalglyphindices: *const u16, _verticalglyphindices: *mut u16) -> WResult<()> {
        Ok(())
    }
    fn HasVerticalGlyphVariants(&self) -> BOOL {
        Default::default()
    }
}

impl IDWriteFontFace2_Impl for FaceObj_Impl {
    fn IsColorFont(&self) -> BOOL {
        self.font().is_some_and(|f| f.table_data(Tag::new(b"COLR")).is_some()).into()
    }
    fn GetColorPaletteCount(&self) -> u32 {
        self.font().and_then(|f| f.cpal().ok().map(|c| c.num_palettes() as u32)).unwrap_or(0)
    }
    fn GetPaletteEntryCount(&self) -> u32 {
        self.font().and_then(|f| f.cpal().ok().map(|c| c.num_palette_entries() as u32)).unwrap_or(0)
    }
    fn GetPaletteEntries(&self, colorpaletteindex: u32, firstentryindex: u32, entrycount: u32, paletteentries: *mut DWRITE_COLOR_F) -> WResult<()> {
        // Read the CPAL palette `colorpaletteindex`, entries `[first, first+count)`, as straight RGBA
        // f32 (CPAL stores BGRA u8). Native returns E_INVALIDARG for an out-of-range palette or span.
        let font = self.font().ok_or(E_INVALIDARG)?;
        let palettes = font.color_palettes();
        let palette = palettes.get(colorpaletteindex as u16).ok_or(E_INVALIDARG)?;
        let all: Vec<DWRITE_COLOR_F> = palette
            .colors()
            .iter()
            // Native converts each CPAL byte with a multiply by 1/255 (not a divide) — the two round
            // to different f32 for most bytes (e.g. 238 → 0x3f6eeef0, not 0x3f6eeeef).
            .map(|c| DWRITE_COLOR_F { r: c.red as f32 * (1.0 / 255.0), g: c.green as f32 * (1.0 / 255.0), b: c.blue as f32 * (1.0 / 255.0), a: c.alpha as f32 * (1.0 / 255.0) })
            .collect();
        let (first, count) = (firstentryindex as usize, entrycount as usize);
        if first + count > all.len() {
            return Err(E_INVALIDARG.into());
        }
        unsafe { slice::from_raw_parts_mut(paletteentries, count) }.copy_from_slice(&all[first..first + count]);
        Ok(())
    }
    fn GetRecommendedRenderingMode(&self, fontemsize: f32, dpix: f32, dpiy: f32, transform: *const DWRITE_MATRIX, _issideways: BOOL, outlinethreshold: DWRITE_OUTLINE_THRESHOLD, measuringmode: DWRITE_MEASURING_MODE, renderingparams: Ref<IDWriteRenderingParams>, renderingmode: *mut DWRITE_RENDERING_MODE, gridfitmode: *mut DWRITE_GRID_FIT_MODE) -> WResult<()> {
        let pm = renderingparams.as_ref().map_or(0, params_mode);
        let m = recommend(&self.bytes, self.index, device_em(fontemsize, dpix, transform), measuringmode.0 as u32, aa_for_threshold(outlinethreshold), pm, [dpix, dpiy]);
        unsafe {
            *renderingmode = DWRITE_RENDERING_MODE(legacy_recommend(m) as i32);
            *gridfitmode = recommend_gridfit(transform, m, &renderingparams);
        }
        Ok(())
    }
}

impl IDWriteFontFace3_Impl for FaceObj_Impl {
    fn GetFontFaceReference(&self) -> WResult<IDWriteFontFaceReference> {
        Ok(super::fontfile::faceref(self.bytes.clone(), self.index, DWRITE_FONT_SIMULATIONS_NONE, self.source.path_str()))
    }
    fn GetPanose(&self, panose_out: *mut DWRITE_PANOSE) {
        if let Some(f) = self.font() {
            unsafe { *panose_out = panose(&f) };
        }
    }
    fn GetWeight(&self) -> DWRITE_FONT_WEIGHT {
        DWRITE_FONT_WEIGHT(self.font().map(|f| face_attrs(&f).0 as i32).unwrap_or(400))
    }
    fn GetStretch(&self) -> DWRITE_FONT_STRETCH {
        DWRITE_FONT_STRETCH(self.font().map(|f| face_attrs(&f).1 as i32).unwrap_or(5))
    }
    fn GetStyle(&self) -> DWRITE_FONT_STYLE {
        DWRITE_FONT_STYLE(self.font().map(|f| face_attrs(&f).2 as i32).unwrap_or(0))
    }
    fn GetFamilyNames(&self) -> WResult<IDWriteLocalizedStrings> {
        let e = self.font().map(|f| localized_names(&f, StringId::FAMILY_NAME)).unwrap_or_default();
        Ok(StringsObj { entries: if e.is_empty() { vec![("en-us".into(), self.family.clone())] } else { e } }.into())
    }
    fn GetFaceNames(&self) -> WResult<IDWriteLocalizedStrings> {
        let e = self.font().map(|f| localized_names(&f, StringId::SUBFAMILY_NAME)).unwrap_or_default();
        Ok(StringsObj { entries: if e.is_empty() { vec![("en-us".into(), self.subfamily())] } else { e } }.into())
    }
    fn GetInformationalStrings(&self, id: DWRITE_INFORMATIONAL_STRING_ID, s: windows_core::OutRef<IDWriteLocalizedStrings>, exists: *mut BOOL) -> WResult<()> {
        let e = info_string_id(id).and_then(|sid| self.font().map(|f| localized_names(&f, sid))).unwrap_or_default();
        unsafe { *exists = (!e.is_empty()).into() };
        if e.is_empty() {
            return Ok(());
        }
        s.write(Some(StringsObj { entries: e }.into()))
    }
    fn HasCharacter(&self, unicodevalue: u32) -> BOOL {
        self.covers(unicodevalue).into()
    }
    fn GetRecommendedRenderingMode(&self, fontemsize: f32, dpix: f32, dpiy: f32, transform: *const DWRITE_MATRIX, _issideways: BOOL, outlinethreshold: DWRITE_OUTLINE_THRESHOLD, measuringmode: DWRITE_MEASURING_MODE, renderingparams: Ref<IDWriteRenderingParams>, renderingmode: *mut DWRITE_RENDERING_MODE1, gridfitmode: *mut DWRITE_GRID_FIT_MODE) -> WResult<()> {
        let pm = renderingparams.as_ref().map_or(0, params_mode);
        let m = recommend(&self.bytes, self.index, device_em(fontemsize, dpix, transform), measuringmode.0 as u32, aa_for_threshold(outlinethreshold), pm, [dpix, dpiy]);
        unsafe {
            *renderingmode = DWRITE_RENDERING_MODE1(m as i32);
            *gridfitmode = recommend_gridfit(transform, m, &renderingparams);
        }
        Ok(())
    }
    fn IsCharacterLocal(&self, unicodevalue: u32) -> BOOL {
        self.covers(unicodevalue).into()
    }
    fn IsGlyphLocal(&self, glyphid: u16) -> BOOL {
        // A local file face has every glyph it reports; `.notdef` (0) and any id below the count are
        // present. We keep it simple: non-null glyph id ⇒ local (the corpus never streams a face).
        (glyphid != 0 || self.font().is_some()).into()
    }
    fn AreCharactersLocal(&self, characters: &PCWSTR, charactercount: u32, _enqueueifnotlocal: BOOL) -> WResult<BOOL> {
        let chars = unsafe { slice::from_raw_parts(characters.0, charactercount as usize) };
        // A locally-loaded face is fully local; "local" here means every character maps to a glyph.
        Ok(String::from_utf16_lossy(chars).chars().all(|c| self.covers(c as u32)).into())
    }
    fn AreGlyphsLocal(&self, _glyphindices: *const u16, _glyphcount: u32, _enqueueifnotlocal: BOOL) -> WResult<BOOL> {
        Ok(self.font().is_some().into())
    }
}

impl IDWriteFontFace4_Impl for FaceObj_Impl {
    fn GetGlyphImageFormats2(&self) -> DWRITE_GLYPH_IMAGE_FORMATS {
        self.font().map(|f| face_image_formats(&f)).unwrap_or(DWRITE_GLYPH_IMAGE_FORMATS_NONE)
    }
    fn GetGlyphImageFormats(&self, glyphid: u16, pixelsperemfirst: u32, pixelsperemlast: u32) -> WResult<DWRITE_GLYPH_IMAGE_FORMATS> {
        let Some(font) = self.font() else { return Ok(DWRITE_GLYPH_IMAGE_FORMATS_NONE) };
        Ok(glyph_image_formats(&font, glyphid, pixelsperemfirst, pixelsperemlast))
    }
    fn GetGlyphImageData(&self, glyphid: u16, pixelsperem: u32, format: DWRITE_GLYPH_IMAGE_FORMATS, glyphdata: *mut DWRITE_GLYPH_IMAGE_DATA, glyphdatacontext: *mut *mut core::ffi::c_void) -> WResult<()> {
        let font = self.font().ok_or(E_INVALIDARG)?;
        let g = GlyphId::new(glyphid as u32);
        let mut data = DWRITE_GLYPH_IMAGE_DATA::default();
        let found = if format == DWRITE_GLYPH_IMAGE_FORMATS_SVG {
            svg_document(&font, glyphid).map(|bytes| {
                data.imageData = bytes.as_ptr() as *const core::ffi::c_void;
                data.imageDataSize = bytes.len() as u32;
                data.uniqueDataId = unique_data_id(glyphid, 0, format);
                // SVG is resolution-independent: native leaves ppem/pixelSize/origins at 0.
            })
        } else if format == DWRITE_GLYPH_IMAGE_FORMATS_PNG || format == DWRITE_GLYPH_IMAGE_FORMATS_PREMULTIPLIED_B8G8R8A8 {
            bitmap_glyph_data(&font, g, pixelsperem, format, &mut data)
        } else {
            None
        };
        // A format the glyph does not provide → E_INVALIDARG, exactly like native.
        if found.is_none() {
            return Err(E_INVALIDARG.into());
        }
        unsafe {
            *glyphdata = data;
            if !glyphdatacontext.is_null() {
                *glyphdatacontext = core::ptr::null_mut();
            }
        }
        Ok(())
    }
    // Legitimate no-op: GetGlyphImageData hands back a slice of our owned font bytes with a null
    // glyphDataContext (no allocation), so there is nothing to free — mirroring native, whose
    // ReleaseGlyphImageData only drops the context handle its GetGlyphImageData allocated (none here).
    fn ReleaseGlyphImageData(&self, _glyphdatacontext: *mut core::ffi::c_void) {}
}

impl IDWriteFontFace5_Impl for FaceObj_Impl {
    fn GetFontAxisValueCount(&self) -> u32 {
        self.face_axis_values().len() as u32
    }
    fn GetFontAxisValues(&self, fontaxisvalues: *mut DWRITE_FONT_AXIS_VALUE, fontaxisvaluecount: u32) -> WResult<()> {
        super::text::write_axis_values(&self.face_axis_values(), fontaxisvalues, fontaxisvaluecount)
    }
    fn HasVariations(&self) -> BOOL {
        (!self.fvar_axes().is_empty()).into()
    }
    fn GetFontResource(&self) -> WResult<IDWriteFontResource> {
        Ok(FontResourceObj { bytes: self.bytes.clone(), index: self.index, source: self.source.clone() }.into())
    }
    fn Equals(&self, fontface: Ref<IDWriteFontFace>) -> BOOL {
        // Same face identity: same font file (same bytes), face index, and simulations — what native
        // compares (the underlying font reference). We do not instance, so axis values do not differ.
        fontface
            .as_ref()
            .and_then(|f| f.cast_object_ref::<FaceObj>().ok())
            .is_some_and(|o| self.index == o.index && self.sims == o.sims && self.bytes == o.bytes)
            .into()
    }
}

impl IDWriteFontFace6_Impl for FaceObj_Impl {
    fn GetFamilyNames(&self, fontfamilymodel: DWRITE_FONT_FAMILY_MODEL) -> WResult<IDWriteLocalizedStrings> {
        // Typographic model: name id 16 (→ 1); weight-stretch-style model: id 1 (→ 16).
        let (primary, fallback) = if fontfamilymodel == DWRITE_FONT_FAMILY_MODEL_TYPOGRAPHIC {
            (StringId::new(16), StringId::FAMILY_NAME)
        } else {
            (StringId::FAMILY_NAME, StringId::new(16))
        };
        self.model_names(primary, fallback, &self.family)
    }
    fn GetFaceNames(&self, fontfamilymodel: DWRITE_FONT_FAMILY_MODEL) -> WResult<IDWriteLocalizedStrings> {
        let (primary, fallback) = if fontfamilymodel == DWRITE_FONT_FAMILY_MODEL_TYPOGRAPHIC {
            (StringId::new(17), StringId::SUBFAMILY_NAME)
        } else {
            (StringId::SUBFAMILY_NAME, StringId::new(17))
        };
        let sub = self.subfamily();
        self.model_names(primary, fallback, &sub)
    }
}

impl IDWriteFontFace7_Impl for FaceObj_Impl {
    fn GetPaintFeatureLevel(&self, glyphimageformat: DWRITE_GLYPH_IMAGE_FORMATS) -> DWRITE_PAINT_FEATURE_LEVEL {
        // The COLR paint support this face offers for the requested format: a v1 paint tree →
        // COLR_V1, a v0 layered table → COLR_V0, otherwise none. Only the COLR formats carry one.
        if glyphimageformat.0 & (DWRITE_GLYPH_IMAGE_FORMATS_COLR_PAINT_TREE.0 | DWRITE_GLYPH_IMAGE_FORMATS_COLR.0) == 0 {
            return DWRITE_PAINT_FEATURE_LEVEL_NONE;
        }
        match self.font().and_then(|f| colr_version(&f)) {
            Some(v) if v >= 1 => DWRITE_PAINT_FEATURE_LEVEL_COLR_V1,
            Some(_) => DWRITE_PAINT_FEATURE_LEVEL_COLR_V0,
            None => DWRITE_PAINT_FEATURE_LEVEL_NONE,
        }
    }
    fn CreatePaintReader(&self, glyphimageformat: DWRITE_GLYPH_IMAGE_FORMATS, _paintfeaturelevel: DWRITE_PAINT_FEATURE_LEVEL) -> WResult<IDWritePaintReader> {
        // Only the COLR paint-tree format produces a paint reader, and only a COLR v1 font has one.
        if glyphimageformat.0 & DWRITE_GLYPH_IMAGE_FORMATS_COLR_PAINT_TREE.0 == 0 {
            return Err(E_INVALIDARG.into());
        }
        if !self.font().and_then(|f| colr_version(&f)).is_some_and(|v| v >= 1) {
            return Err(E_INVALIDARG.into());
        }
        Ok(super::paint::new_reader(self.bytes.clone(), self.index))
    }
}

// --- IDWriteFontResource -----------------------------------------------------------------------

/// `IDWriteFontResource` — the un-instanced variable font behind a face: its file, face index, and
/// `fvar` axis model (count, defaults, ranges, attributes, names). Reached via
/// `IDWriteFontFace5::GetFontResource`.
#[implement(IDWriteFontResource)]
pub(crate) struct FontResourceObj {
    pub(crate) bytes: Arc<Vec<u8>>,
    pub(crate) index: u32,
    pub(crate) source: FontSource,
}

impl FontResourceObj {
    fn font(&self) -> Option<FontRef<'_>> {
        FontRef::from_index(&self.bytes, self.index).ok()
    }
    fn axes(&self) -> Vec<FvarAxis> {
        self.font().map(|f| resource_axes(&f)).unwrap_or_default()
    }
}

impl IDWriteFontResource_Impl for FontResourceObj_Impl {
    fn GetFontFile(&self) -> WResult<IDWriteFontFile> {
        Ok(match &self.source {
            FontSource::Path(p) => super::fontfile::file_ref_path(self.bytes.clone(), p),
            _ => super::fontfile::file_from_bytes(self.bytes.clone()),
        })
    }
    fn GetFontFaceIndex(&self) -> u32 {
        self.index
    }
    fn GetFontAxisCount(&self) -> u32 {
        self.axes().len() as u32
    }
    fn GetDefaultFontAxisValues(&self, fontaxisvalues: *mut DWRITE_FONT_AXIS_VALUE, fontaxisvaluecount: u32) -> WResult<()> {
        let vals: Vec<_> = self.axes().iter().map(|a| DWRITE_FONT_AXIS_VALUE { axisTag: a.tag, value: a.default }).collect();
        super::text::write_axis_values(&vals, fontaxisvalues, fontaxisvaluecount)
    }
    fn GetFontAxisRanges(&self, fontaxisranges: *mut DWRITE_FONT_AXIS_RANGE, fontaxisrangecount: u32) -> WResult<()> {
        let ranges: Vec<_> = self.axes().iter().map(|a| DWRITE_FONT_AXIS_RANGE { axisTag: a.tag, minValue: a.min, maxValue: a.max }).collect();
        if (fontaxisrangecount as usize) < ranges.len() {
            return Err(HRESULT(0x8007_007Au32 as i32).into()); // ERROR_INSUFFICIENT_BUFFER
        }
        if !fontaxisranges.is_null() && !ranges.is_empty() {
            unsafe { slice::from_raw_parts_mut(fontaxisranges, ranges.len()) }.copy_from_slice(&ranges);
        }
        Ok(())
    }
    fn GetFontAxisAttributes(&self, axisindex: u32) -> DWRITE_FONT_AXIS_ATTRIBUTES {
        match self.axes().get(axisindex as usize) {
            Some(a) => {
                let mut attr = DWRITE_FONT_AXIS_ATTRIBUTES_NONE.0;
                if a.min != a.max {
                    attr |= DWRITE_FONT_AXIS_ATTRIBUTES_VARIABLE.0;
                }
                if a.flags & 0x0001 != 0 {
                    attr |= DWRITE_FONT_AXIS_ATTRIBUTES_HIDDEN.0; // fvar HIDDEN_AXIS
                }
                DWRITE_FONT_AXIS_ATTRIBUTES(attr)
            }
            None => DWRITE_FONT_AXIS_ATTRIBUTES_NONE,
        }
    }
    fn GetAxisNames(&self, axisindex: u32) -> WResult<IDWriteLocalizedStrings> {
        let font = self.font().ok_or(E_INVALIDARG)?;
        let axis = self.axes().into_iter().nth(axisindex as usize).ok_or(E_INVALIDARG)?;
        // Only a real fvar axis name (name id >= 256) has a name; a synthesized/unnamed axis returns
        // an empty collection like native — NOT name record 0 (the copyright string).
        let entries = if axis.name_id >= 256 { localized_names(&font, StringId::new(axis.name_id)) } else { Vec::new() };
        Ok(StringsObj { entries }.into())
    }
    fn GetAxisValueNameCount(&self, axisindex: u32) -> u32 {
        let Some(font) = self.font() else { return 0 };
        let Some(axis) = self.axes().into_iter().nth(axisindex as usize) else { return 0 };
        stat_axis_values(&font, axis.tag).len() as u32
    }
    fn GetAxisValueNames(&self, axisindex: u32, axisvalueindex: u32, fontaxisrange: *mut DWRITE_FONT_AXIS_RANGE, names: windows_core::OutRef<IDWriteLocalizedStrings>) -> WResult<()> {
        let font = self.font().ok_or(E_INVALIDARG)?;
        let axis = self.axes().into_iter().nth(axisindex as usize).ok_or(E_INVALIDARG)?;
        let vals = stat_axis_values(&font, axis.tag);
        let (nid, lo, hi) = vals.get(axisvalueindex as usize).copied().ok_or(E_INVALIDARG)?;
        if !fontaxisrange.is_null() {
            unsafe { *fontaxisrange = DWRITE_FONT_AXIS_RANGE { axisTag: axis.tag, minValue: lo, maxValue: hi } };
        }
        names.write(Some(StringsObj { entries: localized_names(&font, nid) }.into())).ok();
        Ok(())
    }
    fn HasVariations(&self) -> BOOL {
        (!self.axes().is_empty()).into()
    }
    fn CreateFontFace(&self, fontsimulations: DWRITE_FONT_SIMULATIONS, fontaxisvalues: *const DWRITE_FONT_AXIS_VALUE, fontaxisvaluecount: u32) -> WResult<IDWriteFontFace5> {
        if FontRef::from_index(&self.bytes, self.index).is_err() {
            return Err(E_INVALIDARG.into());
        }
        // The requested axis values specialise the face: the render path instances the gvar outline
        // at these coordinates (empty → the default instance). Simulations are honoured.
        let axis_values = if fontaxisvalues.is_null() || fontaxisvaluecount == 0 {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(fontaxisvalues, fontaxisvaluecount as usize) }.to_vec()
        };
        let face: IDWriteFontFace7 = FaceObj {
            bytes: self.bytes.clone(),
            index: self.index,
            family: super::fontfile::family_name(&self.bytes, self.index),
            source: self.source.clone(),
            sims: fontsimulations.0 as u16,
            axis_values,
        }
        .into();
        Ok(face.cast()?)
    }
    fn CreateFontFaceReference(&self, fontsimulations: DWRITE_FONT_SIMULATIONS, _fontaxisvalues: *const DWRITE_FONT_AXIS_VALUE, _fontaxisvaluecount: u32) -> WResult<IDWriteFontFaceReference1> {
        // ponytail: core does not instance variable coordinates, so the requested axis values do not
        // specialise the reference (it resolves to the face's own values); simulations are honoured.
        let faceref = super::fontfile::faceref(self.bytes.clone(), self.index, fontsimulations, self.source.path_str());
        Ok(faceref.cast()?)
    }
}

/// The `DWRITE_GLYPH_IMAGE_FORMATS` a font can produce face-wide, from its tables: outline formats
/// (glyf → TrueType, CFF/CFF2 → CFF), COLR v0 layered vs v1 paint tree (from the COLR header version
/// and v0 base-record count), SVG, and a colour-bitmap strike (sbix/CBDT) as PNG.
fn face_image_formats(font: &FontRef) -> DWRITE_GLYPH_IMAGE_FORMATS {
    let mut f = DWRITE_GLYPH_IMAGE_FORMATS_NONE;
    if font.table_data(Tag::new(b"glyf")).is_some() {
        f |= DWRITE_GLYPH_IMAGE_FORMATS_TRUETYPE;
    }
    if font.table_data(Tag::new(b"CFF ")).is_some() || font.table_data(Tag::new(b"CFF2")).is_some() {
        f |= DWRITE_GLYPH_IMAGE_FORMATS_CFF;
    }
    if let Some(colr) = font.table_data(Tag::new(b"COLR")) {
        let b = colr.as_bytes();
        if b.len() >= 4 {
            let version = u16::from_be_bytes([b[0], b[1]]);
            // Native's font-wide set reports COLR (v0 format) whenever a COLR table is present — even a
            // v1 table with zero v0 base-glyph records (VM-verified on a COLRv1 font) — and adds
            // COLR_PAINT_TREE for v1.
            f |= DWRITE_GLYPH_IMAGE_FORMATS_COLR;
            if version >= 1 {
                f |= DWRITE_GLYPH_IMAGE_FORMATS_COLR_PAINT_TREE;
            }
        }
    }
    if font.table_data(Tag::new(b"SVG ")).is_some() {
        f |= DWRITE_GLYPH_IMAGE_FORMATS_SVG;
    }
    // sbix/CBDT colour strikes are PNG in the Windows/corpus fonts; EBDT is monochrome (no DWRITE
    // bit). ponytail: a BGRA-raw strike would be PREMULTIPLIED_B8G8R8A8 — VM-verify (no corpus font).
    if matches!(font.bitmap_strikes().format(), Some(skrifa::bitmap::BitmapFormat::Sbix | skrifa::bitmap::BitmapFormat::Cbdt)) {
        f |= DWRITE_GLYPH_IMAGE_FORMATS_PNG;
    }
    f
}

/// The per-glyph image formats `IDWriteFontFace4::GetGlyphImageFormats(gid, ppemFirst, ppemLast)`
/// reports — the formats in which this glyph is stored as image DATA: an SVG document covering it, and
/// a colour bitmap from a strike whose ppem falls in `[ppemFirst, ppemLast]`. It does NOT report
/// outlines (TRUETYPE/CFF) or COLR — those are constructed, not stored per-glyph image data, and
/// native returns 0 for them here (VM-verified: segoeui 'A', SegUIVar, and
/// seguiemj's COLR glyphs all report 0 at every ppem range incl. `[0,…]`, while GetGlyphImageFormats
/// *2* — the font-wide set — still reports TRUETYPE/COLR). The font-wide set is [`face_image_formats`].
fn glyph_image_formats(font: &FontRef, gid: u16, ppem_first: u32, ppem_last: u32) -> DWRITE_GLYPH_IMAGE_FORMATS {
    let mut f = DWRITE_GLYPH_IMAGE_FORMATS_NONE;
    let g = GlyphId::new(gid as u32);
    if svg_document(font, gid).is_some() {
        f |= DWRITE_GLYPH_IMAGE_FORMATS_SVG;
    }
    // Check every colour-bitmap table independently (native reads sbix AND CBDT; skrifa's default
    // `bitmap_strikes()` exposes only the first, so a font carrying both would hide one).
    for fmt in [BitmapFormat::Sbix, BitmapFormat::Cbdt, BitmapFormat::Ebdt] {
        let Some(strikes) = BitmapStrikes::with_format(font, fmt) else { continue };
        for i in 0..strikes.len() {
            let Some(strike) = strikes.get(i) else { continue };
            let ppem = strike.ppem().round() as u32;
            if ppem < ppem_first || ppem > ppem_last {
                continue;
            }
            if let Some(bg) = strike.get(g) {
                f |= match bg.data {
                    BitmapData::Png(_) => DWRITE_GLYPH_IMAGE_FORMATS_PNG,
                    BitmapData::Bgra(_) => DWRITE_GLYPH_IMAGE_FORMATS_PREMULTIPLIED_B8G8R8A8,
                    BitmapData::Mask(_) => DWRITE_GLYPH_IMAGE_FORMATS_NONE,
                };
            }
        }
    }
    f
}

/// Fill `data` from the first colour-bitmap strike at `ppem` that has `g` in the requested `want`
/// format. Returns `None` if no such strike/format — the caller reports `E_INVALIDARG`.
fn bitmap_glyph_data(font: &FontRef, g: GlyphId, ppem: u32, want: DWRITE_GLYPH_IMAGE_FORMATS, data: &mut DWRITE_GLYPH_IMAGE_DATA) -> Option<()> {
    let strikes = font.bitmap_strikes();
    for i in 0..strikes.len() {
        let strike = strikes.get(i)?;
        if strike.ppem().round() as u32 != ppem {
            continue;
        }
        if let Some(bg) = strike.get(g) {
            let (bytes, fmt) = match bg.data {
                BitmapData::Png(b) => (b, DWRITE_GLYPH_IMAGE_FORMATS_PNG),
                BitmapData::Bgra(b) => (b, DWRITE_GLYPH_IMAGE_FORMATS_PREMULTIPLIED_B8G8R8A8),
                BitmapData::Mask(_) => continue,
            };
            if fmt != want {
                continue;
            }
            data.imageData = bytes.as_ptr() as *const core::ffi::c_void;
            data.imageDataSize = bytes.len() as u32;
            data.uniqueDataId = unique_data_id(g.to_u32() as u16, ppem, fmt);
            data.pixelsPerEm = strike.ppem().round() as u32;
            data.pixelSize = D2D_SIZE_U { width: bg.width, height: bg.height };
            // Placement origins (pixels). ponytail: bearing-derived best-effort; native's exact
            // per-origin values are VM-pending (no corpus font ships a PNG/BGRA strike).
            let p = POINT { x: bg.inner_bearing_x.round() as i32, y: -bg.inner_bearing_y.round() as i32 };
            data.horizontalLeftOrigin = p;
            data.horizontalRightOrigin = p;
            data.verticalTopOrigin = p;
            data.verticalBottomOrigin = p;
            return Some(());
        }
    }
    None
}

/// The raw SVG document bytes for `gid` from the `SVG ` table (returned verbatim — gzip/SVGZ included,
/// as native does), as a borrow of the font's owned bytes. `None` if no document covers the glyph.
fn svg_document<'a>(font: &FontRef<'a>, gid: u16) -> Option<&'a [u8]> {
    let svg = font.table_data(Tag::new(b"SVG "))?;
    let d = svg.as_bytes();
    // SVG header: version u16, offsetToSVGDocumentList u32. The document list: numEntries u16, then
    // per entry { startGlyphID u16, endGlyphID u16, svgDocOffset u32, svgDocLength u32 }, the doc
    // offset measured from the start of the document list.
    let list_off = u32::from_be_bytes([*d.get(2)?, *d.get(3)?, *d.get(4)?, *d.get(5)?]) as usize;
    let list = d.get(list_off..)?;
    let num = u16::from_be_bytes([*list.get(0)?, *list.get(1)?]) as usize;
    for i in 0..num {
        let e = list.get(2 + i * 12..2 + i * 12 + 12)?;
        let start = u16::from_be_bytes([e[0], e[1]]);
        let end = u16::from_be_bytes([e[2], e[3]]);
        if gid >= start && gid <= end {
            let off = u32::from_be_bytes([e[4], e[5], e[6], e[7]]) as usize;
            let len = u32::from_be_bytes([e[8], e[9], e[10], e[11]]) as usize;
            return d.get(list_off + off..list_off + off + len);
        }
    }
    None
}

/// A stable, non-zero id for a glyph image blob (the `DWRITE_GLYPH_IMAGE_DATA::uniqueDataId` cache
/// key). ponytail: native's exact value is an internal handle; this is a deterministic stand-in —
/// apps use it only to dedupe identical blobs, which this still does per (glyph, ppem, format).
fn unique_data_id(gid: u16, ppem: u32, fmt: DWRITE_GLYPH_IMAGE_FORMATS) -> u32 {
    (gid as u32).wrapping_mul(2654435761) ^ (ppem.wrapping_mul(40503)) ^ ((fmt.0 as u32) << 24)
}

// --- IDWriteLocalizedStrings (all name-table locale records) -----------------------------------

#[implement(IDWriteLocalizedStrings)]
pub(crate) struct StringsObj {
    /// `(locale, value)` per localized record, in the font's name-record order (native's ordering).
    /// Locale names are lowercase BCP-47 ("en-us"), matching native; a record with no language is "".
    entries: Vec<(String, String)>,
}

/// One English localized string (a font-set per-font property value).
pub(crate) fn localized(value: String) -> IDWriteLocalizedStrings {
    StringsObj { entries: vec![("en-us".into(), value)] }.into()
}

/// Every localized record for name `id` on `font`, in name-record order, as native's
/// IDWriteLocalizedStrings would enumerate them (lowercase locale; skrifa already maps the platform/
/// language id to BCP-47). Empty when the font has no such name.
pub(crate) fn localized_names(font: &FontRef, id: StringId) -> Vec<(String, String)> {
    // Native enumerates Windows-platform (id 3) name records only, in table order. skrifa's
    // `localized_strings` yields every matching record across platforms, so mask it against the same
    // records (same order, same name-id filter) and keep the ones whose platform is Windows — this
    // drops the Mac/Unicode duplicates (e.g. a bare "en" vs the Windows "en-us").
    let Ok(name) = font.name() else { return Vec::new() };
    let windows = name.name_record().iter().filter(|r| r.name_id() == id).map(|r| r.platform_id() == 3);
    font.localized_strings(id)
        .zip(windows)
        .filter_map(|(s, win)| if win { s.language().map(|l| (l.to_ascii_lowercase(), s.to_string())) } else { None })
        .collect()
}

/// `DWRITE_INFORMATIONAL_STRING_ID` → the OpenType `name` table id it reads, as an skrifa `StringId`.
/// `None` for ids with no `name`-record backing (NONE, script-language tags), where native reports the
/// string as absent.
fn info_string_id(id: DWRITE_INFORMATIONAL_STRING_ID) -> Option<StringId> {
    let name_id: u16 = match id.0 {
        1 => 0,   // COPYRIGHT_NOTICE
        2 => 5,   // VERSION_STRINGS
        3 => 7,   // TRADEMARK
        4 => 8,   // MANUFACTURER
        5 => 9,   // DESIGNER
        6 => 12,  // DESIGNER_URL
        7 => 10,  // DESCRIPTION
        8 => 11,  // FONT_VENDOR_URL
        9 => 13,  // LICENSE_DESCRIPTION
        10 => 14, // LICENSE_INFO_URL
        11 => 1,  // WIN32_FAMILY_NAMES
        12 => 2,  // WIN32_SUBFAMILY_NAMES
        13 => 16, // TYPOGRAPHIC_FAMILY_NAMES
        14 => 17, // TYPOGRAPHIC_SUBFAMILY_NAMES
        15 => 19, // SAMPLE_TEXT
        16 => 4,  // FULL_NAME
        17 => 6,  // POSTSCRIPT_NAME
        18 => 20, // POSTSCRIPT_CID_NAME
        _ => return None,
    };
    Some(StringId::new(name_id))
}

impl StringsObj {
    fn get(&self, index: u32) -> Option<&(String, String)> {
        self.entries.get(index as usize)
    }
}

impl IDWriteLocalizedStrings_Impl for StringsObj_Impl {
    fn GetCount(&self) -> u32 {
        self.entries.len() as u32
    }
    fn FindLocaleName(&self, localename: &PCWSTR, index: *mut u32, exists: *mut BOOL) -> WResult<()> {
        let want = unsafe { localename.to_string() }.unwrap_or_default().to_ascii_lowercase();
        let hit = self.entries.iter().position(|(l, _)| *l == want);
        unsafe {
            *index = hit.unwrap_or(0) as u32;
            *exists = hit.is_some().into();
        }
        Ok(())
    }
    fn GetLocaleNameLength(&self, index: u32) -> WResult<u32> {
        Ok(self.get(index).map(|(l, _)| l.encode_utf16().count() as u32).unwrap_or(0))
    }
    fn GetLocaleName(&self, index: u32, localename: PWSTR, size: u32) -> WResult<()> {
        write_wide(self.get(index).map(|(l, _)| l.as_str()).unwrap_or(""), localename, size)
    }
    fn GetStringLength(&self, index: u32) -> WResult<u32> {
        Ok(self.get(index).map(|(_, v)| v.encode_utf16().count() as u32).unwrap_or(0))
    }
    fn GetString(&self, index: u32, stringbuffer: PWSTR, size: u32) -> WResult<()> {
        write_wide(self.get(index).map(|(_, v)| v.as_str()).unwrap_or(""), stringbuffer, size)
    }
}

// --- IDWriteFontFallback (system + custom-built font fallback) ---------------------------------

/// One `IDWriteFontFallbackBuilder::AddMapping` entry: Unicode ranges → an ordered family list,
/// gated by a locale (empty = any).
#[derive(Clone)]
pub(crate) struct CustomMapping {
    pub(crate) ranges: Vec<(u32, u32)>,
    pub(crate) families: Vec<String>,
    pub(crate) locale: String,
}

impl CustomMapping {
    fn contains(&self, cp: u32) -> bool {
        self.ranges.iter().any(|&(a, b)| cp >= a && cp <= b)
    }
}

#[implement(IDWriteFontFallback)]
pub(crate) struct FallbackObj {
    pub(crate) fonts: Arc<SystemFonts>,
    /// Empty ⇒ the system fallback (the baked `text::fallback` table); non-empty ⇒ a builder-made
    /// fallback that uses only these mappings.
    pub(crate) mappings: Vec<CustomMapping>,
}

/// A vertical layout's upright trimming-sign extent per em: the vertical ellipsis's face is the first system
/// fallback family for U+22EE that maps it, matched to the format, as the d2d core resolves it
/// ([`text::layout::upright_sign_per_em`]); 0 when none does.
pub(crate) fn upright_sign(fonts: &SystemFonts, locale: &str, weight: u16, italic: bool) -> (f32, f32) {
    let cache = RefCell::new(HashMap::new());
    fallback::candidates(0x22EE, locale)
        .into_iter()
        .find_map(|f| FallbackObj::covering(fonts, &cache, f, weight, italic, '\u{22EE}'))
        .and_then(|e| {
            let en = &fonts.entries[e];
            text::layout::upright_sign_per_em(en.source.read()?.to_vec(), en.face, en.sims as u32)
        })
        .unwrap_or((0.0, 0.0))
}

impl FallbackObj {
    /// The family's fallback face for `ch` ([`SystemFonts::match_entry_covering`]), reading each entry's
    /// bytes once per call through `cache`.
    fn covering(fonts: &SystemFonts, cache: &RefCell<HashMap<usize, Option<Arc<Vec<u8>>>>>, fam: &str, weight: u16, italic: bool, ch: char) -> Option<usize> {
        let read = |e: usize| cache.borrow_mut().entry(e).or_insert_with(|| fonts.entries[e].source.read()).clone();
        let fi = fonts.find_family(fam)?;
        let (e, sims) = fonts.match_entry_covering(fi, weight, 5, if italic { 2 } else { 0 }, ch, read)?;
        // The family's entry for that face and simulation; the real face when the family lacks one (a
        // simulated bold of a face other than its regular).
        let same = |o: usize| fonts.entries[o].face == fonts.entries[e].face && fonts.entries[o].sims == sims && match (&fonts.entries[o].source, &fonts.entries[e].source) {
            (FontSource::Path(a), FontSource::Path(b)) => a == b,
            (FontSource::Bytes(a), FontSource::Bytes(b)) => Arc::ptr_eq(a, b),
            _ => false,
        };
        Some(fonts.families[fi].faces.iter().copied().find(|&o| same(o)).unwrap_or(e))
    }
}

impl IDWriteFontFallback_Impl for FallbackObj_Impl {
    fn MapCharacters(&self, analysissource: Ref<IDWriteTextAnalysisSource>, textposition: u32, textlength: u32, _basefontcollection: Ref<IDWriteFontCollection>, basefamilyname: &PCWSTR, baseweight: DWRITE_FONT_WEIGHT, basestyle: DWRITE_FONT_STYLE, _basestretch: DWRITE_FONT_STRETCH, mappedlength: *mut u32, mappedfont: windows_core::OutRef<IDWriteFont>, scale: *mut f32) -> WResult<()> {
        unsafe {
            *mappedlength = 0;
            *scale = 1.0;
        }
        let src = analysissource.as_ref().ok_or(E_INVALIDARG)?;
        // The source owns the text buffer; `GetTextAtPosition` hands back the run from `textposition`.
        let (mut tp, mut tn) = (core::ptr::null_mut::<u16>(), 0u32);
        unsafe { src.GetTextAtPosition(textposition, &mut tp, &mut tn)? };
        if tp.is_null() || tn == 0 || textlength == 0 {
            return Ok(());
        }
        let avail = (tn as usize).min(textlength as usize);
        let text = String::from_utf16_lossy(unsafe { slice::from_raw_parts(tp, avail) });
        let locale = {
            let (mut ln, mut lp) = (0u32, core::ptr::null_mut::<u16>());
            let _ = unsafe { src.GetLocaleName(textposition, &mut ln, &mut lp) };
            if lp.is_null() { "en-us".to_string() } else { wide_to_string(lp, Some(ln as usize)) }
        };
        let base = wide_to_string(basefamilyname.0, None);
        let (weight, italic) = (baseweight.0 as u16, basestyle.0 != 0);

        // Coverage closures over the system fonts: the base family's own face, and (cached) each
        // fallback-candidate family's face. `text::fallback::map_characters` is the native
        // `IDWriteFontFallback::MapCharacters` resolution ported over this.
        let base_bytes = self.fonts.resolve(&base, weight, italic);
        let base_covers = |ch: char| -> bool {
            base_bytes.as_ref().and_then(|(b, i, _)| FontRef::from_index(b, *i).ok().and_then(|f| f.charmap().map(ch))).is_some()
        };
        let cache = RefCell::new(HashMap::new());
        // `fallback::OWN_FAMILY` is the base family's other faces, looked at before the fallback list.
        let covers = |fam: &str, ch: char| FallbackObj::covering(&self.fonts, &cache, if fam == fallback::OWN_FAMILY { &base } else { fam }, weight, italic, ch).is_some();

        // The first run's resolved family (None = base) and its UTF-16 length. The system fallback
        // uses the baked table (`text::fallback`); a builder-made fallback uses only its mappings.
        let (mapped16, resolved): (u32, Option<String>) = if self.mappings.is_empty() {
            let spans = fallback::map_characters(&text, &locale, base_covers, covers);
            let first = spans.first().copied().unwrap_or(fallback::Span { start: 0, end: text.len(), family: None });
            (text[first.start..first.end].encode_utf16().count() as u32, first.family.map(str::to_string))
        } else {
            let resolve = |ch: char| -> Option<String> {
                if base_covers(ch) {
                    return None;
                }
                self.mappings.iter().find_map(|m| {
                    (m.locale.is_empty() || m.locale.eq_ignore_ascii_case(&locale)).then(|| m.contains(ch as u32).then(|| m.families.iter().find(|f| covers(f, ch)).cloned()).flatten()).flatten()
                })
            };
            let first = resolve(text.chars().next().unwrap());
            let n16 = text.chars().take_while(|&ch| resolve(ch) == first).map(|ch| ch.len_utf16() as u32).sum();
            (n16, first)
        };
        unsafe { *mappedlength = mapped16 };

        // The resolved fallback family's face for the run's first character, or the base family's matching
        // face when unmapped.
        let entry = match resolved.as_deref() {
            Some(fam) => FallbackObj::covering(&self.fonts, &cache, if fam == fallback::OWN_FAMILY { &base } else { fam }, weight, italic, text.chars().next().unwrap_or(' ')),
            None => self.fonts.find_family(&base).and_then(|fi| self.fonts.match_entry(fi, weight, 5, if italic { 2 } else { 0 })),
        };
        if let Some(entry) = entry {
            mappedfont.write(Some(font3(self.fonts.clone(), entry).cast()?))?;
        }
        Ok(())
    }
}

// --- IDWriteFontFallbackBuilder ----------------------------------------------------------------

#[implement(IDWriteFontFallbackBuilder)]
pub(crate) struct FallbackBuilderObj {
    fonts: Arc<SystemFonts>,
    mappings: RefCell<Vec<CustomMapping>>,
}

pub(crate) fn fallback_builder(fonts: Arc<SystemFonts>) -> IDWriteFontFallbackBuilder {
    FallbackBuilderObj { fonts, mappings: RefCell::new(Vec::new()) }.into()
}

impl IDWriteFontFallbackBuilder_Impl for FallbackBuilderObj_Impl {
    fn AddMapping(&self, ranges: *const DWRITE_UNICODE_RANGE, rangescount: u32, targetfamilynames: *const *const u16, targetfamilynamescount: u32, _fontcollection: Ref<IDWriteFontCollection>, localename: &PCWSTR, _basefamilyname: &PCWSTR, _scale: f32) -> WResult<()> {
        let ranges = unsafe { slice::from_raw_parts(ranges, rangescount as usize) }.iter().map(|r| (r.first, r.last)).collect();
        let families = unsafe { slice::from_raw_parts(targetfamilynames, targetfamilynamescount as usize) }.iter().map(|&p| super::wide_to_string(p, None)).collect();
        self.mappings.borrow_mut().push(CustomMapping { ranges, families, locale: super::wide_to_string(localename.0, None) });
        Ok(())
    }
    fn AddMappings(&self, fontfallback: Ref<IDWriteFontFallback>) -> WResult<()> {
        if let Some(fb) = fontfallback.as_ref().and_then(|f| f.cast_object_ref::<FallbackObj>().ok()) {
            self.mappings.borrow_mut().extend(fb.mappings.iter().cloned());
        }
        Ok(())
    }
    fn CreateFontFallback(&self) -> WResult<IDWriteFontFallback> {
        Ok(FallbackObj { fonts: self.fonts.clone(), mappings: self.mappings.borrow().clone() }.into())
    }
}

// Verify the skrifa-backed getters against the native values captured on the reference VM for Arial
// (the census2-02 fontface oracle: "AVag" @ ppem 16). Reads the font from D2D_FONTS or the main
// checkout's ref/; skips when neither is present (public clone).
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn font_source_selects_native_rasterizer_for_faces_and_formats() {
        // A path in the system font directory selects the font-cache rasterizer, so this needs Arial
        // there (D2D_FONTS, or Windows' own); skip without it, as the gate's workspace run has none.
        let path = fonts_dir().join("arial.ttf");
        let Ok(bytes) = std::fs::read(&path) else { return; };
        let bytes = Arc::new(bytes);
        let factory: IDWriteFactory3 = { let f: IDWriteFactory7 = super::super::factory::FactoryObj::new().into(); f.cast().unwrap() };
        for (source, expected) in [(FontSource::Path(path), 1), (FontSource::Bytes(bytes.clone()), 0)] {
            let face: IDWriteFontFace7 = FaceObj { bytes: bytes.clone(), index: 0, family: "Arial".into(), source: source.clone(), sims: 0, axis_values: Vec::new() }.into();
            let it: ITextInterop = face.cast().unwrap();
            let mut backend = u32::MAX;
            unsafe { it.FontRasterizer(&mut backend).unwrap(); }
            assert_eq!(backend, expected);
            let collection = collection(SystemFonts::from_sources(vec![source]));
            let (family, locale): (Vec<u16>, Vec<u16>) = ("Arial\0".encode_utf16().collect(), "en-US\0".encode_utf16().collect());
            let format = unsafe { factory.CreateTextFormat(windows_core::PCWSTR(family.as_ptr()), &collection, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 16.0, windows_core::PCWSTR(locale.as_ptr())) }.unwrap();
            let it: ITextInterop = format.cast().unwrap();
            unsafe { it.FontRasterizer(&mut backend).unwrap(); }
            assert_eq!(backend, expected);
        }
    }

    #[test]
    fn rendering_recommendations_preserve_font_dpi_and_legacy_modes() {
        let Some(bytes) = arial() else { return; };
        let face = FaceObj { bytes: Arc::new(bytes), index: 0, family: "Arial".into(), source: FontSource::Bytes(Arc::new(Vec::new())), sims: 0, axis_values: Vec::new() };
        let face3: IDWriteFontFace7 = face.into();
        let face0: IDWriteFontFace = face3.cast().unwrap();
        let face1: IDWriteFontFace1 = face3.cast().unwrap();
        let face2: IDWriteFontFace2 = face3.cast().unwrap();
        // Native: Arial gasp selects 4 at 12px, and only Face3
        // exposes the high-DPI downsampled mode7; earlier interface enums report5.
        for (em, dpi, old, modern, grid) in [(12.0, 96.0, 4, 4, 2), (12.0, 172.8, 5, 7, 2), (100.0, 96.0, 6, 6, 1)] {
            unsafe {
                let factory: IDWriteFactory3 = { let f: IDWriteFactory7 = super::super::factory::FactoryObj::new().into(); f.cast().unwrap() };
                let p = factory.CreateRenderingParams().unwrap();
                assert_eq!(face0.GetRecommendedRenderingMode(em, dpi / 96.0, DWRITE_MEASURING_MODE_NATURAL, &p).unwrap().0, old);
                assert_eq!(face1.GetRecommendedRenderingMode(em, dpi, dpi, None, false, DWRITE_OUTLINE_THRESHOLD_ANTIALIASED, DWRITE_MEASURING_MODE_NATURAL).unwrap().0, old);
                let (mut m2, mut g2) = (DWRITE_RENDERING_MODE_DEFAULT, DWRITE_GRID_FIT_MODE_DEFAULT);
                face2.GetRecommendedRenderingMode(em, dpi, dpi, None, false, DWRITE_OUTLINE_THRESHOLD_ANTIALIASED, DWRITE_MEASURING_MODE_NATURAL, &p, &mut m2, &mut g2).unwrap();
                assert_eq!((m2.0, g2.0), (old, grid));
                let (mut m3, mut g3) = (DWRITE_RENDERING_MODE1_DEFAULT, DWRITE_GRID_FIT_MODE_DEFAULT);
                face3.GetRecommendedRenderingMode(em, dpi, dpi, None, false, DWRITE_OUTLINE_THRESHOLD_ANTIALIASED, DWRITE_MEASURING_MODE_NATURAL, &p, &mut m3, &mut g3).unwrap();
                assert_eq!((m3.0, g3.0), (modern, grid));
            }
        }
    }

    fn arial() -> Option<Vec<u8>> {
        for p in [
            std::env::var("D2D_FONTS").ok().map(|d| format!("{d}/arial.ttf")),
            Some(concat!(env!("CARGO_MANIFEST_DIR"), "/../../ref/windows/Fonts/arial.ttf").into()),
        ]
        .into_iter()
        .flatten()
        {
            if let Ok(b) = std::fs::read(&p) {
                return Some(b);
            }
        }
        None
    }



    #[test]
    fn arial_face_metrics_match_native() {
        let Some(bytes) = arial() else {
            eprintln!("SKIP: Arial not present");
            return;
        };
        let f = FontRef::from_index(&bytes, 0).unwrap();
        let m = design_metrics(&f);
        assert_eq!((m.designUnitsPerEm, m.ascent, m.descent, m.lineGap, m.capHeight, m.xHeight), (2048, 1854, 434, 67, 1467, 1062));
        assert_eq!(f.metrics(Size::unscaled(), LocationRef::default()).glyph_count, 4548);
        let (gm, glyphs) = (f.glyph_metrics(Size::unscaled(), LocationRef::default()), [36u16, 57, 68, 74]);
        let adv: Vec<i32> = glyphs.iter().map(|&g| gm.advance_width(GlyphId::new(g as u32)).unwrap_or(0.0).round() as i32).collect();
        assert_eq!(adv, [1366, 1366, 1139, 1139]);
        assert_eq!(kern_adjustments(&f, &glyphs), [-152, -152, 0, 0]);
        assert!(has_kern(&f));
        assert_eq!(unicode_ranges(&f).len(), 140);
        assert_eq!(d2d::text::gdi::gdi_compat_vmetrics(&f, 16.0, 2048), Some((1920, 384)));
        // GetMetrics takes usWin with the hhea surplus as the gap (VM probe), where hhea alone differs.
        let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../ref/windows/Fonts"));
        for (file, want) in [("seguiemj.ttf", (2210, 514, 0)), ("calibri.ttf", (1950, 550, 0)), ("consola.ttf", (1884, 514, 0)), ("pala.ttf", (2150, 613, 0))] {
            let dir = std::env::var("D2D_FONTS").map(std::path::PathBuf::from).unwrap_or_else(|_| dir.to_path_buf());
            let Ok(b) = std::fs::read(dir.join(file)) else { continue };
            let m = design_metrics(&FontRef::from_index(&b, 0).unwrap());
            assert_eq!((m.ascent, m.descent, m.lineGap), want, "{file}");
        }
        assert_eq!(face_attrs(&f), (400, 5, 0));
        assert_eq!(unsafe { panose(&f).values }, [2, 11, 6, 4, 2, 2, 2, 2, 2, 4]);
    }
}

