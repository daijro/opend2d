//! DirectWrite system font fallback — reproduces `IDWriteFontFallback::MapCharacters`
//! and the built-in `FONTFALLBACK`
//! table DirectWrite loads from its own resource.
//!
//! The native algorithm, per text position:
//!  1. If the base family's font covers the character, keep the base family (`MatchFamilyToText`).
//!  2. Otherwise binary-search the character in the table's Unicode-range groups; each group holds
//!     an ordered list of *lookups* gated by a BCP-47 locale range (`RangeIncludes`) and an optional
//!     base-family filter. Concatenate the eligible lookups' family lists in order and take the
//!     first family whose font covers the character (`FindNextMapping` + `MatchFontToText`).
//!  3. If nothing covers it, fall back to the base family (its `.notdef`).
//! A run extends while consecutive characters resolve to the same family.
//!
//! This module is pure (no font I/O, so it stays wasm-clean): the caller passes coverage closures
//! backed by loaded faces. `text::layout` turns the resulting [`Span`]s into per-run fonts.
//!
//! The table below is the reference VM's `FONTFALLBACK`/`FALLBACK` resource in full: all 515 range
//! groups with their lookups. Family names and Unicode ranges are data (short identifiers), not copied
//! code. Every lookup's scale is 1.0 and its base-family filter empty, so neither is modelled.

/// One lookup inside a range group: an ordered candidate family list, gated by a BCP-47 locale
/// range (`""` matches any locale). The native table also allows a base-family filter per lookup,
/// but every baked group leaves it empty, so it is omitted — re-add it with the dwrite resource's
/// filter string if a family-scoped group ever gets a scene.
pub struct Lookup {
    pub locale: &'static str,
    pub families: &'static [&'static str],
}

/// A Unicode-range group `[first, last]` and its ordered lookups — one entry of dwrite's range table.
pub struct Group {
    pub first: u32,
    pub last: u32,
    pub lookups: &'static [Lookup],
}

// Generated from the reference dwrite.dll FONTFALLBACK/FALLBACK resource (every range group, in
// resource order): each group's lookups as shared lists.
const L0: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI", "Segoe UI Symbol", "Ebrima", "Calibri", "Sans Serif Collection"] }, Lookup { locale: "", families: &["Tahoma", "Arial", "Simplified Arabic", "Traditional Arabic", "Arial Unicode MS", "Microsoft Sans Serif", "Lucida Sans Unicode"] }];
const L1: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI", "Calibri"] }, Lookup { locale: "", families: &["Sylfaen", "Arial Unicode MS"] }];
const L2: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI", "Simplified Arabic"] }, Lookup { locale: "", families: &["Tahoma", "Arial", "Traditional Arabic", "Arial Unicode MS", "Microsoft Sans Serif", "Lucida Sans Unicode"] }];
const L3: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI Historic"] }, Lookup { locale: "", families: &["Estrangelo Edessa"] }];
const L4: &[Lookup] = &[Lookup { locale: "", families: &["MV Boli"] }];
const L5: &[Lookup] = &[Lookup { locale: "", families: &["Ebrima"] }];
const L6: &[Lookup] = &[Lookup { locale: "", families: &["Sans Serif Collection"] }];
const L7: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI Historic"] }];
const L8: &[Lookup] = &[];
const L9: &[Lookup] = &[Lookup { locale: "", families: &["Nirmala UI"] }, Lookup { locale: "", families: &["Mangal"] }];
const L10: &[Lookup] = &[Lookup { locale: "", families: &["Nirmala UI"] }, Lookup { locale: "", families: &["Vrinda"] }];
const L11: &[Lookup] = &[Lookup { locale: "", families: &["Nirmala UI"] }, Lookup { locale: "", families: &["Raavi"] }];
const L12: &[Lookup] = &[Lookup { locale: "", families: &["Nirmala UI"] }, Lookup { locale: "", families: &["Shruti"] }];
const L13: &[Lookup] = &[Lookup { locale: "", families: &["Nirmala UI"] }, Lookup { locale: "", families: &["Kalinga"] }];
const L14: &[Lookup] = &[Lookup { locale: "", families: &["Nirmala UI"] }, Lookup { locale: "", families: &["Latha"] }];
const L15: &[Lookup] = &[Lookup { locale: "", families: &["Nirmala UI"] }, Lookup { locale: "", families: &["Gautami"] }];
const L16: &[Lookup] = &[Lookup { locale: "", families: &["Nirmala UI"] }, Lookup { locale: "", families: &["Tunga"] }];
const L17: &[Lookup] = &[Lookup { locale: "", families: &["Nirmala UI"] }, Lookup { locale: "", families: &["Kartika"] }];
const L18: &[Lookup] = &[Lookup { locale: "", families: &["Nirmala UI"] }, Lookup { locale: "", families: &["Iskoola Pota"] }];
const L19: &[Lookup] = &[Lookup { locale: "", families: &["Leelawadee UI"] }, Lookup { locale: "", families: &["Leelawadee", "Tahoma"] }];
const L20: &[Lookup] = &[Lookup { locale: "", families: &["Leelawadee UI"] }, Lookup { locale: "", families: &["Lao UI", "DokChampa"] }];
const L21: &[Lookup] = &[Lookup { locale: "", families: &["Microsoft Himalaya"] }];
const L22: &[Lookup] = &[Lookup { locale: "", families: &["Myanmar Text"] }];
const L23: &[Lookup] = &[Lookup { locale: "", families: &["Malgun Gothic", "Gulim"] }];
const L24: &[Lookup] = &[Lookup { locale: "", families: &["Ebrima"] }, Lookup { locale: "", families: &["Nyala"] }];
const L25: &[Lookup] = &[Lookup { locale: "", families: &["Gadugi"] }, Lookup { locale: "", families: &["Plantagenet Cherokee"] }];
const L26: &[Lookup] = &[Lookup { locale: "", families: &["Gadugi"] }, Lookup { locale: "", families: &["Euphemia"] }];
const L27: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI Historic", "Segoe UI Symbol"] }];
const L28: &[Lookup] = &[Lookup { locale: "", families: &["Leelawadee UI", "Khmer UI", "DaunPenh"] }];
const L29: &[Lookup] = &[Lookup { locale: "", families: &["Mongolian Baiti"] }];
const L30: &[Lookup] = &[Lookup { locale: "", families: &["Microsoft Tai Le"] }];
const L31: &[Lookup] = &[Lookup { locale: "", families: &["Microsoft New Tai Lue"] }];
const L32: &[Lookup] = &[Lookup { locale: "", families: &["Leelawadee UI"] }];
const L33: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI", "Segoe UI Symbol", "Ebrima", "Calibri", "Sans Serif Collection"] }];
const L34: &[Lookup] = &[Lookup { locale: "", families: &["Nirmala UI"] }];
const L35: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI", "Calibri"] }];
const L36: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI", "Calibri"] }, Lookup { locale: "", families: &["Sans Serif Collection"] }];
const L37: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI", "Calibri"] }, Lookup { locale: "", families: &["Nirmala UI", "Sans Serif Collection"] }];
const L38: &[Lookup] = &[Lookup { locale: "", families: &["Nirmala UI", "Sans Serif Collection"] }];
const L39: &[Lookup] = &[Lookup { locale: "zh-hans", families: &["Microsoft YaHei UI", "Microsoft YaHei", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Segoe UI", "Segoe UI Symbol", "SimSun", "MS Gothic", "MingLiU", "Tahoma"] }, Lookup { locale: "zh-hant", families: &["Microsoft JhengHei UI", "Microsoft JhengHei", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Segoe UI", "Segoe UI Symbol", "Microsoft YaHei UI", "Microsoft YaHei", "MingLiU", "MS Gothic", "Tahoma"] }, Lookup { locale: "ja", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "Segoe UI", "Segoe UI Symbol", "Microsoft JhengHei UI", "Microsoft JhengHei", "Microsoft YaHei UI", "Microsoft YaHei", "Malgun Gothic", "MS Gothic", "MingLiU", "Tahoma"] }, Lookup { locale: "ko", families: &["Malgun Gothic", "Segoe UI", "Segoe UI Symbol", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft JhengHei UI", "Microsoft JhengHei", "Microsoft YaHei UI", "Microsoft YaHei", "Gulim", "MS Gothic", "MingLiU", "Tahoma"] }, Lookup { locale: "", families: &["Segoe UI", "Segoe UI Symbol", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft JhengHei UI", "Microsoft JhengHei", "Microsoft YaHei UI", "Microsoft YaHei", "Malgun Gothic", "Tahoma", "MS Gothic", "MingLiU", "MingLiU"] }, Lookup { locale: "", families: &["Lucida Sans Unicode", "Arial", "Arial Unicode MS"] }];
const L40: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI Symbol"] }];
const L41: &[Lookup] = &[Lookup { locale: "zh-hans", families: &["Microsoft YaHei UI", "Microsoft YaHei", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Segoe UI", "Segoe UI Symbol", "SimSun", "MS Gothic", "MingLiU"] }, Lookup { locale: "zh-hant", families: &["Microsoft JhengHei UI", "Microsoft JhengHei", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Segoe UI", "Segoe UI Symbol", "Microsoft YaHei UI", "Microsoft YaHei", "MingLiU", "MS Gothic"] }, Lookup { locale: "ja", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "Segoe UI", "Segoe UI Symbol", "Microsoft JhengHei UI", "Microsoft JhengHei", "Microsoft YaHei UI", "Microsoft YaHei", "Malgun Gothic", "MS Gothic", "MingLiU"] }, Lookup { locale: "ko", families: &["Malgun Gothic", "Segoe UI", "Segoe UI Symbol", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft JhengHei UI", "Microsoft JhengHei", "Microsoft YaHei UI", "Microsoft YaHei", "Gulim", "MS Gothic", "MingLiU"] }, Lookup { locale: "", families: &["Segoe UI", "Segoe UI Symbol", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft JhengHei UI", "Microsoft JhengHei", "Microsoft YaHei UI", "Microsoft YaHei", "Malgun Gothic", "MS Gothic", "MingLiU", "MingLiU"] }, Lookup { locale: "", families: &["Tahoma", "Lucida Sans Unicode", "Arial", "Arial Unicode MS"] }];
const L42: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI Emoji"] }, Lookup { locale: "zh-hans", families: &["Microsoft YaHei UI", "Microsoft YaHei", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Segoe UI", "Segoe UI Symbol", "SimSun", "MS Gothic", "MingLiU"] }, Lookup { locale: "zh-hant", families: &["Microsoft JhengHei UI", "Microsoft JhengHei", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Segoe UI", "Segoe UI Symbol", "Microsoft YaHei UI", "Microsoft YaHei", "MingLiU", "MS Gothic"] }, Lookup { locale: "ja", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "Segoe UI", "Segoe UI Symbol", "Microsoft JhengHei UI", "Microsoft JhengHei", "Microsoft YaHei UI", "Microsoft YaHei", "Malgun Gothic", "MS Gothic", "MingLiU"] }, Lookup { locale: "ko", families: &["Malgun Gothic", "Segoe UI", "Segoe UI Symbol", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft JhengHei UI", "Microsoft JhengHei", "Microsoft YaHei UI", "Microsoft YaHei", "Gulim", "MS Gothic", "MingLiU"] }, Lookup { locale: "", families: &["Segoe UI", "Segoe UI Symbol", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft JhengHei UI", "Microsoft JhengHei", "Microsoft YaHei UI", "Microsoft YaHei", "Malgun Gothic", "MS Gothic", "MingLiU", "MingLiU"] }, Lookup { locale: "", families: &["Tahoma", "Lucida Sans Unicode", "Arial", "Arial Unicode MS"] }];
const L43: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI Symbol"] }, Lookup { locale: "", families: &["Lucida Sans Unicode"] }];
const L44: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI Symbol"] }, Lookup { locale: "", families: &["Arial Unicode MS"] }];
const L45: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI Emoji"] }, Lookup { locale: "zh-hans", families: &["Microsoft YaHei UI", "Microsoft YaHei", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Segoe UI", "Segoe UI Symbol", "SimSun", "MS Gothic", "MingLiU", "Tahoma"] }, Lookup { locale: "zh-hant", families: &["Microsoft JhengHei UI", "Microsoft JhengHei", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Segoe UI", "Segoe UI Symbol", "Microsoft YaHei UI", "Microsoft YaHei", "MingLiU", "MS Gothic", "Tahoma"] }, Lookup { locale: "ja", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "Segoe UI", "Segoe UI Symbol", "Microsoft JhengHei UI", "Microsoft JhengHei", "Microsoft YaHei UI", "Microsoft YaHei", "Malgun Gothic", "MS Gothic", "MingLiU", "Tahoma"] }, Lookup { locale: "ko", families: &["Malgun Gothic", "Segoe UI", "Segoe UI Symbol", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft JhengHei UI", "Microsoft JhengHei", "Microsoft YaHei UI", "Microsoft YaHei", "Gulim", "MS Gothic", "MingLiU", "Tahoma"] }, Lookup { locale: "", families: &["Segoe UI", "Segoe UI Symbol", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft JhengHei UI", "Microsoft JhengHei", "Microsoft YaHei UI", "Microsoft YaHei", "Malgun Gothic", "Tahoma", "MS Gothic", "MingLiU", "MingLiU"] }, Lookup { locale: "", families: &["Lucida Sans Unicode", "Arial", "Arial Unicode MS"] }];
const L46: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI Symbol", "Cambria Math", "MS Gothic"] }];
const L47: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI Emoji"] }, Lookup { locale: "", families: &["Segoe UI Symbol", "Cambria Math", "MS Gothic"] }];
const L48: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI Historic", "Segoe UI Symbol", "Sans Serif Collection"] }];
const L49: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI", "Segoe UI Symbol", "Tahoma", "Times New Roman"] }];
const L50: &[Lookup] = &[Lookup { locale: "zh-hans", families: &["Microsoft YaHei UI", "Microsoft YaHei", "SimSun", "Yu Gothic UI", "Meiryo UI", "Meiryo", "MingLiu"] }, Lookup { locale: "zh-hant", families: &["Microsoft JhengHei UI", "Microsoft YaHei UI", "Microsoft YaHei", "MingLiU", "Yu Gothic UI", "Meiryo UI", "Meiryo", "SimSun"] }, Lookup { locale: "ja", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft YaHei UI", "Microsoft YaHei", "MingLiU", "SimSun"] }, Lookup { locale: "ko", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft YaHei UI", "Microsoft YaHei", "MingLiU", "SimSun"] }, Lookup { locale: "", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft YaHei UI", "Microsoft YaHei", "MingLiU", "SimSun"] }];
const L51: &[Lookup] = &[Lookup { locale: "", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo"] }];
const L52: &[Lookup] = &[Lookup { locale: "", families: &["Microsoft YaHei UI", "SimSun"] }];
const L53: &[Lookup] = &[Lookup { locale: "zh-hans", families: &["Microsoft YaHei UI", "Microsoft YaHei", "SimSun", "MS Gothic"] }, Lookup { locale: "zh-hant", families: &["Microsoft JhengHei UI", "Microsoft JhengHei", "MingLiu", "MS Gothic"] }, Lookup { locale: "ko", families: &["Malgun Gothic", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft YaHei UI", "Microsoft YaHei", "Gulim", "MS Gothic", "MingLiu"] }, Lookup { locale: "ja", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft YaHei UI", "Microsoft YaHei", "MS Gothic", "MingLiu"] }, Lookup { locale: "", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft YaHei UI", "Microsoft YaHei", "MS Gothic", "MingLiu"] }, Lookup { locale: "", families: &["Arial Unicode MS"] }];
const L54: &[Lookup] = &[Lookup { locale: "zh-hans", families: &["Microsoft YaHei UI", "Microsoft YaHei", "SimSun"] }, Lookup { locale: "zh-hant", families: &["Microsoft JhengHei UI", "Microsoft JhengHei", "MingLiU"] }, Lookup { locale: "ja", families: &["Microsoft JhengHei UI", "Microsoft JhengHei", "MingLiu"] }, Lookup { locale: "ko", families: &["Microsoft JhengHei UI", "Microsoft JhengHei", "MingLiU"] }, Lookup { locale: "", families: &["Microsoft JhengHei UI", "Microsoft JhengHei", "MingLiu"] }, Lookup { locale: "", families: &["Arial Unicode MS"] }];
const L55: &[Lookup] = &[Lookup { locale: "zh-hans", families: &["Microsoft YaHei UI", "Microsoft YaHei", "MingLiU"] }, Lookup { locale: "zh-hant", families: &["Microsoft JhengHei UI", "Microsoft JhengHei", "MingLiU"] }, Lookup { locale: "ja", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft JhengHei UI", "Microsoft JhengHei", "MingLiU"] }, Lookup { locale: "ko", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft JhengHei UI", "Microsoft JhengHei", "MingLiU"] }, Lookup { locale: "", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft JhengHei UI", "Microsoft JhengHei", "MingLiU"] }];
const L56: &[Lookup] = &[Lookup { locale: "", families: &["Microsoft JhengHei UI", "MingLiU"] }];
const L57: &[Lookup] = &[Lookup { locale: "zh-hans", families: &["Microsoft YaHei UI", "Microsoft YaHei", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Segoe UI Symbol", "SimSun", "MS Gothic"] }, Lookup { locale: "zh-hant", families: &["Microsoft JhengHei UI", "Microsoft JhengHei", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Segoe UI Symbol", "MingLiU", "MS Gothic"] }, Lookup { locale: "ja", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "Segoe UI Symbol", "MS Gothic"] }, Lookup { locale: "ko", families: &["Segoe UI Symbol", "Yu Gothic UI", "Meiryo UI", "Meiryo", "MS Gothic"] }, Lookup { locale: "", families: &["Segoe UI Symbol", "Yu Gothic UI", "Meiryo UI", "Meiryo", "MS Gothic"] }];
const L58: &[Lookup] = &[Lookup { locale: "", families: &["Yu Gothic UI", "Meiryo UI"] }];
const L59: &[Lookup] = &[Lookup { locale: "zh-hans", families: &["Microsoft YaHei UI", "Microsoft YaHei", "Microsoft JhengHei UI", "Microsoft JhengHei", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Malgun Gothic", "Segoe UI Symbol", "SimSun", "MS Gothic"] }, Lookup { locale: "zh-hant", families: &["Microsoft JhengHei UI", "Microsoft JhengHei", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Malgun Gothic", "Segoe UI Symbol", "PMingLiU", "MS Gothic"] }, Lookup { locale: "ko", families: &["Malgun Gothic", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Segoe UI Symbol", "Gulim", "MS Gothic"] }, Lookup { locale: "ja", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "Segoe UI Symbol", "MS Gothic"] }, Lookup { locale: "", families: &["Segoe UI Symbol", "Yu Gothic UI", "Meiryo UI", "Meiryo", "MS Gothic"] }];
const L60: &[Lookup] = &[Lookup { locale: "zh-hans", families: &["Microsoft YaHei UI", "Microsoft YaHei", "SimSun", "SimSun-18030", "SimSun-ExtB", "MingLiU"] }, Lookup { locale: "zh-hk", families: &["Microsoft JhengHei UI", "Microsoft JhengHei", "Microsoft YaHei UI", "Microsoft YaHei", "MingLiU_HKSCS", "MingLiU", "SimSun-18030", "SimSun-ExtB", "SimSun"] }, Lookup { locale: "zh-hant", families: &["Microsoft JhengHei UI", "Microsoft JhengHei", "Microsoft YaHei UI", "Microsoft YaHei", "MingLiU", "SimSun-18030", "SimSun-ExtB", "SimSun"] }, Lookup { locale: "ja", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "MS Gothic", "Microsoft JhengHei UI", "Microsoft JhengHei", "Microsoft YaHei UI", "Microsoft YaHei", "MingLiU"] }, Lookup { locale: "ko", families: &["Malgun Gothic", "Gulim", "Microsoft JhengHei UI", "Microsoft JhengHei", "Microsoft YaHei UI", "Microsoft YaHei", "MS Gothic", "MingLiU", "Simsun"] }, Lookup { locale: "", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft JhengHei UI", "Microsoft JhengHei", "Microsoft YaHei UI", "Microsoft YaHei", "MS Gothic", "MingLiU", "Simsun"] }];
const L61: &[Lookup] = &[Lookup { locale: "", families: &["Microsoft Yi Baiti"] }, Lookup { locale: "", families: &["SimSun-18030", "SimSun-ExtB"] }];
const L62: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI"] }];
const L63: &[Lookup] = &[Lookup { locale: "", families: &["Microsoft PhagsPa"] }];
const L64: &[Lookup] = &[Lookup { locale: "", families: &["Javanese Text"] }];
const L65: &[Lookup] = &[Lookup { locale: "", families: &["Gadugi"] }];
const L66: &[Lookup] = &[Lookup { locale: "", families: &["Malgun Gothic", "Gulim"] }, Lookup { locale: "", families: &["Malgun Gothic", "Gulim"] }];
const L67: &[Lookup] = &[Lookup { locale: "zh-hk", families: &["MingLiU_HKSCS"] }];
const L68: &[Lookup] = &[Lookup { locale: "zh-cn", families: &["Microsoft YaHei UI", "Microsoft YaHei", "SimSun"] }, Lookup { locale: "zh-hans", families: &["Microsoft YaHei UI", "Microsoft YaHei", "SimSun"] }, Lookup { locale: "zh-hk", families: &["MingLiU_HKSCS"] }];
const L69: &[Lookup] = &[Lookup { locale: "zh-hans", families: &["Microsoft YaHei UI", "Microsoft YaHei", "Microsoft JhengHei UI", "Microsoft JhengHei", "Yu Gothic UI", "Meiryo UI", "Meiryo", "SimSun", "PMingLiU", "MS Gothic"] }, Lookup { locale: "zh-hant", families: &["Microsoft JhengHei UI", "Microsoft JhengHei", "Yu Gothic UI", "Meiryo UI", "Meiryo", "PMingLiU", "MS Mincho"] }, Lookup { locale: "ko", families: &["Malgun Gothic", "Gulim", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft JhengHei UI", "Microsoft JhengHei", "MS Gothic", "PMingLiU", "SimSun"] }, Lookup { locale: "ja", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft JhengHei UI", "Microsoft JhengHei", "MS Gothic", "PMingLiU", "SimSun", "Gulim"] }, Lookup { locale: "", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft JhengHei UI", "Microsoft JhengHei", "MS Gothic", "PMingLiU", "SimSun", "Gulim"] }];
const L70: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI", "Calibri"] }, Lookup { locale: "", families: &["Segoe UI", "Calibri"] }, Lookup { locale: "", families: &["Sylfaen", "Arial Unicode MS"] }];
const L71: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI", "Segoe UI Symbol", "Ebrima", "Calibri", "Sans Serif Collection"] }, Lookup { locale: "", families: &["Segoe UI", "Simplified Arabic"] }, Lookup { locale: "", families: &["Tahoma", "Arial", "Simplified Arabic", "Traditional Arabic", "Arial Unicode MS", "Microsoft Sans Serif", "Lucida Sans Unicode"] }, Lookup { locale: "", families: &["Tahoma", "Arial", "Traditional Arabic", "Arial Unicode MS", "Microsoft Sans Serif", "Lucida Sans Unicode"] }];
const L72: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI", "Simplified Arabic"] }, Lookup { locale: "", families: &["Tahoma", "Traditional Arabic", "Arial Unicode MS"] }];
const L73: &[Lookup] = &[Lookup { locale: "zh-hans", families: &["Microsoft YaHei UI", "Microsoft YaHei"] }, Lookup { locale: "zh-hant", families: &["Microsoft JhengHei UI", "Microsoft JhengHei"] }, Lookup { locale: "ko", families: &["Malgun Gothic", "Microsoft JhengHei UI", "Microsoft JhengHei"] }, Lookup { locale: "ja", families: &["Microsoft JhengHei UI", "Microsoft JhengHei"] }, Lookup { locale: "", families: &["Microsoft JhengHei UI", "Microsoft JhengHei"] }];
const L74: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI", "Segoe UI Symbol", "Ebrima", "Calibri", "Sans Serif Collection"] }, Lookup { locale: "", families: &["Sans Serif Collection"] }, Lookup { locale: "", families: &["Tahoma", "Simplified Arabic", "Traditional Arabic", "Arial Unicode MS"] }];
const L75: &[Lookup] = &[Lookup { locale: "zh-hans", families: &["Microsoft YaHei UI", "Microsoft YaHei", "SimSun"] }, Lookup { locale: "zh-hant", families: &["Microsoft JhengHei UI", "Microsoft JhengHei", "MingLiU"] }, Lookup { locale: "ja", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft JhengHei UI", "Microsoft JhengHei", "MingLiu"] }, Lookup { locale: "ko", families: &["Malgun Gothic", "Microsoft JhengHei UI", "Microsoft JhengHei", "MingLiu"] }, Lookup { locale: "", families: &["Microsoft JhengHei UI", "Microsoft JhengHei", "MingLiu"] }];
const L76: &[Lookup] = &[Lookup { locale: "zh-hans", families: &["Microsoft YaHei UI", "Microsoft YaHei", "MS Gothic"] }, Lookup { locale: "zh-hant", families: &["Microsoft JhengHei UI", "Microsoft JhengHei", "MS Gothic"] }, Lookup { locale: "ko", families: &["Malgun Gothic", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Gulim", "MS Gothic"] }, Lookup { locale: "ja", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "MS Gothic"] }, Lookup { locale: "", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "MS Gothic"] }];
const L77: &[Lookup] = &[Lookup { locale: "zh-hans", families: &["Microsoft YaHei UI", "Microsoft YaHei", "MingLiU"] }, Lookup { locale: "zh-hant", families: &["Microsoft JhengHei UI", "Microsoft JhengHei", "MingLiU"] }, Lookup { locale: "ja", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "MS Gothic"] }, Lookup { locale: "ko", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "MS Gothic"] }, Lookup { locale: "", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "MS Gothic"] }];
const L78: &[Lookup] = &[Lookup { locale: "zh-hans", families: &["Microsoft YaHei UI", "Microsoft YaHei", "Microsoft JhengHei UI", "Microsoft JhengHei"] }, Lookup { locale: "zh-hant", families: &["Microsoft JhengHei UI", "Microsoft JhengHei"] }, Lookup { locale: "ja", families: &["Microsoft JhengHei UI", "Microsoft JhengHei"] }, Lookup { locale: "ko", families: &["Malgun Gothic", "Microsoft JhengHei UI", "Microsoft JhengHei"] }, Lookup { locale: "", families: &["Microsoft JhengHei UI", "Microsoft JhengHei"] }];
const L79: &[Lookup] = &[Lookup { locale: "zh-hans", families: &["Microsoft YaHei UI", "Microsoft YaHei", "MS Gothic", "Gulim"] }, Lookup { locale: "zh-hant", families: &["Microsoft JhengHei UI", "Microsoft JhengHei", "MS Gothic", "Gulim"] }, Lookup { locale: "ja", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "MS Gothic", "Gulim"] }, Lookup { locale: "ko", families: &["Malgun Gothic", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Microsoft JhengHei UI", "Microsoft JhengHei", "Gulim", "MS Gothic"] }, Lookup { locale: "", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "MS Gothic", "Gulim"] }];
const L80: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI"] }, Lookup { locale: "", families: &["Tahoma"] }];
const L81: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI", "Calibri", "Sans Serif Collection"] }];
const L82: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI Symbol", "Cambria Math"] }];
const L83: &[Lookup] = &[Lookup { locale: "", families: &["Segoe UI Emoji"] }, Lookup { locale: "", families: &["Segoe UI Symbol"] }];
const L84: &[Lookup] = &[Lookup { locale: "zh-hans", families: &["SimSun-ExtB"] }, Lookup { locale: "zh-hk", families: &["MingLiU_HKSCS-ExtB", "MingLiU-ExtB"] }, Lookup { locale: "ja", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "MS Gothic", "MingLiU-ExtB"] }, Lookup { locale: "", families: &["MingLiU-ExtB", "Yu Gothic UI", "Yu Gothic", "Microsoft YaHei UI", "Microsoft YaHei"] }];
const L85: &[Lookup] = &[Lookup { locale: "zh-hk", families: &["MingLiU_HKSCS-ExtB"] }, Lookup { locale: "", families: &["MingLiU-ExtB", "Yu Gothic UI", "Yu Gothic", "Microsoft YaHei UI", "Microsoft YaHei"] }];
const L86: &[Lookup] = &[Lookup { locale: "", families: &["SimSun-ExtB", "Yu Gothic UI", "Yu Gothic", "Microsoft YaHei UI", "Microsoft YaHei"] }];
const L87: &[Lookup] = &[Lookup { locale: "zh-hans", families: &["SimSun-ExtB", "MingLiU-ExtB", "Yu Gothic UI", "Meiryo UI", "Meiryo"] }, Lookup { locale: "zh-hk", families: &["MingLiU_HKSCS-ExtB", "MingLiU-ExtB", "Yu Gothic UI", "Meiryo UI", "Meiryo"] }, Lookup { locale: "zh-hant", families: &["MingLiU-ExtB", "Yu Gothic UI", "Meiryo UI", "Meiryo"] }, Lookup { locale: "ja", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "MingLiU-ExtB"] }, Lookup { locale: "ko", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "MingLiU-ExtB"] }, Lookup { locale: "", families: &["Yu Gothic UI", "Meiryo UI", "Meiryo", "MingLiU-ExtB"] }];
static TABLE: &[Group] = &[
    Group { first: 0x0, last: 0x52F, lookups: L0 },
    Group { first: 0x530, last: 0x58F, lookups: L1 },
    Group { first: 0x590, last: 0x5FF, lookups: L0 },
    Group { first: 0x600, last: 0x6FF, lookups: L2 },
    Group { first: 0x700, last: 0x74F, lookups: L3 },
    Group { first: 0x750, last: 0x77F, lookups: L2 },
    Group { first: 0x780, last: 0x7BF, lookups: L4 },
    Group { first: 0x7C0, last: 0x7FF, lookups: L5 },
    Group { first: 0x800, last: 0x85F, lookups: L6 },
    Group { first: 0x860, last: 0x86F, lookups: L7 },
    Group { first: 0x870, last: 0x89F, lookups: L8 },
    Group { first: 0x8A0, last: 0x8FF, lookups: L2 },
    Group { first: 0x900, last: 0x97F, lookups: L9 },
    Group { first: 0x980, last: 0x9FF, lookups: L10 },
    Group { first: 0xA00, last: 0xA7F, lookups: L11 },
    Group { first: 0xA80, last: 0xAFF, lookups: L12 },
    Group { first: 0xB00, last: 0xB7F, lookups: L13 },
    Group { first: 0xB80, last: 0xBFF, lookups: L14 },
    Group { first: 0xC00, last: 0xC7F, lookups: L15 },
    Group { first: 0xC80, last: 0xCFF, lookups: L16 },
    Group { first: 0xD00, last: 0xD7F, lookups: L17 },
    Group { first: 0xD80, last: 0xDFF, lookups: L18 },
    Group { first: 0xE00, last: 0xE7F, lookups: L19 },
    Group { first: 0xE80, last: 0xEFF, lookups: L20 },
    Group { first: 0xF00, last: 0xFFF, lookups: L21 },
    Group { first: 0x1000, last: 0x109F, lookups: L22 },
    Group { first: 0x10A0, last: 0x10FF, lookups: L1 },
    Group { first: 0x1100, last: 0x11FF, lookups: L23 },
    Group { first: 0x1200, last: 0x139F, lookups: L24 },
    Group { first: 0x13A0, last: 0x13FF, lookups: L25 },
    Group { first: 0x1400, last: 0x167F, lookups: L26 },
    Group { first: 0x1680, last: 0x16FF, lookups: L27 },
    Group { first: 0x1700, last: 0x177F, lookups: L6 },
    Group { first: 0x1780, last: 0x17FF, lookups: L28 },
    Group { first: 0x1800, last: 0x18AF, lookups: L29 },
    Group { first: 0x18B0, last: 0x18FF, lookups: L26 },
    Group { first: 0x1900, last: 0x194F, lookups: L6 },
    Group { first: 0x1950, last: 0x197F, lookups: L30 },
    Group { first: 0x1980, last: 0x19DF, lookups: L31 },
    Group { first: 0x19E0, last: 0x19FF, lookups: L28 },
    Group { first: 0x1A00, last: 0x1A1F, lookups: L32 },
    Group { first: 0x1A20, last: 0x1AAF, lookups: L6 },
    Group { first: 0x1AB0, last: 0x1AFF, lookups: L33 },
    Group { first: 0x1B00, last: 0x1C4F, lookups: L6 },
    Group { first: 0x1C50, last: 0x1C7F, lookups: L34 },
    Group { first: 0x1C80, last: 0x1C8F, lookups: L0 },
    Group { first: 0x1C90, last: 0x1CBF, lookups: L35 },
    Group { first: 0x1CC0, last: 0x1CCF, lookups: L36 },
    Group { first: 0x1CD0, last: 0x1CFB, lookups: L37 },
    Group { first: 0x1CFC, last: 0x1CFF, lookups: L38 },
    Group { first: 0x1D00, last: 0x1FFF, lookups: L0 },
    Group { first: 0x2000, last: 0x202E, lookups: L39 },
    Group { first: 0x202F, last: 0x202F, lookups: L29 },
    Group { first: 0x2030, last: 0x20CF, lookups: L39 },
    Group { first: 0x20D0, last: 0x20FF, lookups: L40 },
    Group { first: 0x2100, last: 0x2138, lookups: L41 },
    Group { first: 0x2139, last: 0x2139, lookups: L42 },
    Group { first: 0x213A, last: 0x214F, lookups: L41 },
    Group { first: 0x2150, last: 0x22FF, lookups: L39 },
    Group { first: 0x2300, last: 0x2319, lookups: L41 },
    Group { first: 0x231A, last: 0x231B, lookups: L42 },
    Group { first: 0x231C, last: 0x2327, lookups: L41 },
    Group { first: 0x2328, last: 0x2328, lookups: L42 },
    Group { first: 0x2329, last: 0x23CE, lookups: L41 },
    Group { first: 0x23CF, last: 0x23CF, lookups: L42 },
    Group { first: 0x23D0, last: 0x23E8, lookups: L41 },
    Group { first: 0x23E9, last: 0x23F3, lookups: L42 },
    Group { first: 0x23F4, last: 0x23F7, lookups: L41 },
    Group { first: 0x23F8, last: 0x23FA, lookups: L42 },
    Group { first: 0x23FB, last: 0x23FF, lookups: L41 },
    Group { first: 0x2400, last: 0x243F, lookups: L43 },
    Group { first: 0x2440, last: 0x245F, lookups: L44 },
    Group { first: 0x2460, last: 0x25FC, lookups: L39 },
    Group { first: 0x25FD, last: 0x25FE, lookups: L45 },
    Group { first: 0x25FF, last: 0x25FF, lookups: L39 },
    Group { first: 0x2600, last: 0x2604, lookups: L42 },
    Group { first: 0x2605, last: 0x260D, lookups: L41 },
    Group { first: 0x260E, last: 0x260E, lookups: L42 },
    Group { first: 0x260F, last: 0x2610, lookups: L41 },
    Group { first: 0x2611, last: 0x2611, lookups: L42 },
    Group { first: 0x2612, last: 0x2613, lookups: L41 },
    Group { first: 0x2614, last: 0x2615, lookups: L42 },
    Group { first: 0x2616, last: 0x2617, lookups: L41 },
    Group { first: 0x2618, last: 0x2618, lookups: L42 },
    Group { first: 0x2619, last: 0x261F, lookups: L41 },
    Group { first: 0x2620, last: 0x2620, lookups: L42 },
    Group { first: 0x2621, last: 0x2621, lookups: L41 },
    Group { first: 0x2622, last: 0x2623, lookups: L42 },
    Group { first: 0x2624, last: 0x2625, lookups: L41 },
    Group { first: 0x2626, last: 0x2626, lookups: L42 },
    Group { first: 0x2627, last: 0x2629, lookups: L41 },
    Group { first: 0x262A, last: 0x262A, lookups: L42 },
    Group { first: 0x262B, last: 0x262D, lookups: L41 },
    Group { first: 0x262E, last: 0x262F, lookups: L42 },
    Group { first: 0x2630, last: 0x2637, lookups: L41 },
    Group { first: 0x2638, last: 0x263A, lookups: L42 },
    Group { first: 0x263B, last: 0x263F, lookups: L41 },
    Group { first: 0x2640, last: 0x2640, lookups: L42 },
    Group { first: 0x2641, last: 0x2641, lookups: L41 },
    Group { first: 0x2642, last: 0x2642, lookups: L42 },
    Group { first: 0x2643, last: 0x2647, lookups: L41 },
    Group { first: 0x2648, last: 0x2653, lookups: L42 },
    Group { first: 0x2654, last: 0x265E, lookups: L41 },
    Group { first: 0x265F, last: 0x265F, lookups: L42 },
    Group { first: 0x2660, last: 0x2667, lookups: L41 },
    Group { first: 0x2668, last: 0x2668, lookups: L42 },
    Group { first: 0x2669, last: 0x267A, lookups: L41 },
    Group { first: 0x267B, last: 0x267B, lookups: L42 },
    Group { first: 0x267C, last: 0x267D, lookups: L41 },
    Group { first: 0x267E, last: 0x267F, lookups: L42 },
    Group { first: 0x2680, last: 0x2691, lookups: L41 },
    Group { first: 0x2692, last: 0x2697, lookups: L42 },
    Group { first: 0x2698, last: 0x2698, lookups: L41 },
    Group { first: 0x2699, last: 0x2699, lookups: L42 },
    Group { first: 0x269A, last: 0x269A, lookups: L41 },
    Group { first: 0x269B, last: 0x269C, lookups: L42 },
    Group { first: 0x269D, last: 0x269F, lookups: L41 },
    Group { first: 0x26A0, last: 0x26A1, lookups: L42 },
    Group { first: 0x26A2, last: 0x26A6, lookups: L41 },
    Group { first: 0x26A7, last: 0x26A7, lookups: L42 },
    Group { first: 0x26A8, last: 0x26A9, lookups: L41 },
    Group { first: 0x26AA, last: 0x26AB, lookups: L42 },
    Group { first: 0x26AC, last: 0x26AF, lookups: L41 },
    Group { first: 0x26B0, last: 0x26B1, lookups: L42 },
    Group { first: 0x26B2, last: 0x26BC, lookups: L41 },
    Group { first: 0x26BD, last: 0x26BE, lookups: L42 },
    Group { first: 0x26BF, last: 0x26C3, lookups: L41 },
    Group { first: 0x26C4, last: 0x26C5, lookups: L42 },
    Group { first: 0x26C6, last: 0x26C7, lookups: L41 },
    Group { first: 0x26C8, last: 0x26C8, lookups: L42 },
    Group { first: 0x26C9, last: 0x26CD, lookups: L41 },
    Group { first: 0x26CE, last: 0x26CF, lookups: L42 },
    Group { first: 0x26D0, last: 0x26D0, lookups: L41 },
    Group { first: 0x26D1, last: 0x26D1, lookups: L42 },
    Group { first: 0x26D2, last: 0x26D2, lookups: L41 },
    Group { first: 0x26D3, last: 0x26D4, lookups: L42 },
    Group { first: 0x26D5, last: 0x26E8, lookups: L41 },
    Group { first: 0x26E9, last: 0x26EA, lookups: L42 },
    Group { first: 0x26EB, last: 0x26EF, lookups: L41 },
    Group { first: 0x26F0, last: 0x26F5, lookups: L42 },
    Group { first: 0x26F6, last: 0x26F6, lookups: L41 },
    Group { first: 0x26F7, last: 0x26F8, lookups: L42 },
    Group { first: 0x26F9, last: 0x26F9, lookups: L41 },
    Group { first: 0x26FA, last: 0x26FA, lookups: L42 },
    Group { first: 0x26FB, last: 0x26FC, lookups: L41 },
    Group { first: 0x26FD, last: 0x26FD, lookups: L42 },
    Group { first: 0x26FE, last: 0x2701, lookups: L41 },
    Group { first: 0x2702, last: 0x2702, lookups: L42 },
    Group { first: 0x2703, last: 0x2704, lookups: L41 },
    Group { first: 0x2705, last: 0x2705, lookups: L42 },
    Group { first: 0x2706, last: 0x2707, lookups: L41 },
    Group { first: 0x2708, last: 0x270C, lookups: L42 },
    Group { first: 0x270D, last: 0x270E, lookups: L41 },
    Group { first: 0x270F, last: 0x270F, lookups: L42 },
    Group { first: 0x2710, last: 0x2711, lookups: L41 },
    Group { first: 0x2712, last: 0x2712, lookups: L42 },
    Group { first: 0x2713, last: 0x2713, lookups: L41 },
    Group { first: 0x2714, last: 0x2714, lookups: L42 },
    Group { first: 0x2715, last: 0x2715, lookups: L41 },
    Group { first: 0x2716, last: 0x2716, lookups: L42 },
    Group { first: 0x2717, last: 0x271C, lookups: L41 },
    Group { first: 0x271D, last: 0x271D, lookups: L42 },
    Group { first: 0x271E, last: 0x2720, lookups: L41 },
    Group { first: 0x2721, last: 0x2721, lookups: L42 },
    Group { first: 0x2722, last: 0x2727, lookups: L41 },
    Group { first: 0x2728, last: 0x2728, lookups: L42 },
    Group { first: 0x2729, last: 0x2732, lookups: L41 },
    Group { first: 0x2733, last: 0x2734, lookups: L42 },
    Group { first: 0x2735, last: 0x2743, lookups: L41 },
    Group { first: 0x2744, last: 0x2744, lookups: L42 },
    Group { first: 0x2745, last: 0x2746, lookups: L41 },
    Group { first: 0x2747, last: 0x2747, lookups: L42 },
    Group { first: 0x2748, last: 0x274B, lookups: L41 },
    Group { first: 0x274C, last: 0x274C, lookups: L42 },
    Group { first: 0x274D, last: 0x274D, lookups: L41 },
    Group { first: 0x274E, last: 0x274E, lookups: L42 },
    Group { first: 0x274F, last: 0x2752, lookups: L41 },
    Group { first: 0x2753, last: 0x2755, lookups: L42 },
    Group { first: 0x2756, last: 0x2756, lookups: L41 },
    Group { first: 0x2757, last: 0x2757, lookups: L42 },
    Group { first: 0x2758, last: 0x2762, lookups: L41 },
    Group { first: 0x2763, last: 0x2764, lookups: L42 },
    Group { first: 0x2765, last: 0x2794, lookups: L41 },
    Group { first: 0x2795, last: 0x2797, lookups: L42 },
    Group { first: 0x2798, last: 0x27A0, lookups: L41 },
    Group { first: 0x27A1, last: 0x27A1, lookups: L42 },
    Group { first: 0x27A2, last: 0x27AF, lookups: L41 },
    Group { first: 0x27B0, last: 0x27B0, lookups: L42 },
    Group { first: 0x27B1, last: 0x27BE, lookups: L41 },
    Group { first: 0x27BF, last: 0x27BF, lookups: L42 },
    Group { first: 0x27C0, last: 0x2B1A, lookups: L46 },
    Group { first: 0x2B1B, last: 0x2B1C, lookups: L47 },
    Group { first: 0x2B1D, last: 0x2B4F, lookups: L46 },
    Group { first: 0x2B50, last: 0x2B50, lookups: L47 },
    Group { first: 0x2B51, last: 0x2B54, lookups: L46 },
    Group { first: 0x2B55, last: 0x2B55, lookups: L47 },
    Group { first: 0x2B56, last: 0x2BFF, lookups: L46 },
    Group { first: 0x2C00, last: 0x2C5F, lookups: L48 },
    Group { first: 0x2C60, last: 0x2C7F, lookups: L0 },
    Group { first: 0x2C80, last: 0x2CFF, lookups: L27 },
    Group { first: 0x2D00, last: 0x2D2F, lookups: L1 },
    Group { first: 0x2D30, last: 0x2D7F, lookups: L5 },
    Group { first: 0x2D80, last: 0x2DDF, lookups: L24 },
    Group { first: 0x2DE0, last: 0x2DFF, lookups: L33 },
    Group { first: 0x2E00, last: 0x2E7F, lookups: L49 },
    Group { first: 0x2E80, last: 0x2EFF, lookups: L50 },
    Group { first: 0x2F00, last: 0x2FDF, lookups: L51 },
    Group { first: 0x2FE0, last: 0x2FEF, lookups: L8 },
    Group { first: 0x2FF0, last: 0x2FFF, lookups: L52 },
    Group { first: 0x3000, last: 0x30FF, lookups: L53 },
    Group { first: 0x3100, last: 0x312F, lookups: L54 },
    Group { first: 0x3130, last: 0x318F, lookups: L23 },
    Group { first: 0x3190, last: 0x319F, lookups: L55 },
    Group { first: 0x31A0, last: 0x31BF, lookups: L54 },
    Group { first: 0x31C0, last: 0x31EF, lookups: L56 },
    Group { first: 0x31F0, last: 0x31FF, lookups: L53 },
    Group { first: 0x3200, last: 0x321F, lookups: L23 },
    Group { first: 0x3220, last: 0x324F, lookups: L57 },
    Group { first: 0x3250, last: 0x3250, lookups: L58 },
    Group { first: 0x3251, last: 0x325F, lookups: L57 },
    Group { first: 0x3260, last: 0x327F, lookups: L23 },
    Group { first: 0x3280, last: 0x32CB, lookups: L57 },
    Group { first: 0x32CC, last: 0x32CF, lookups: L58 },
    Group { first: 0x32D0, last: 0x3376, lookups: L57 },
    Group { first: 0x3377, last: 0x337A, lookups: L58 },
    Group { first: 0x337B, last: 0x337F, lookups: L57 },
    Group { first: 0x3380, last: 0x33DF, lookups: L59 },
    Group { first: 0x33E0, last: 0x33FF, lookups: L57 },
    Group { first: 0x3400, last: 0x4DBF, lookups: L60 },
    Group { first: 0x4DC0, last: 0x4DFF, lookups: L40 },
    Group { first: 0x4E00, last: 0x9FFF, lookups: L60 },
    Group { first: 0xA000, last: 0xA4CF, lookups: L61 },
    Group { first: 0xA4D0, last: 0xA4FF, lookups: L62 },
    Group { first: 0xA500, last: 0xA63F, lookups: L5 },
    Group { first: 0xA640, last: 0xA69F, lookups: L0 },
    Group { first: 0xA6A0, last: 0xA6FF, lookups: L5 },
    Group { first: 0xA700, last: 0xA71F, lookups: L62 },
    Group { first: 0xA720, last: 0xA7FF, lookups: L0 },
    Group { first: 0xA800, last: 0xA82F, lookups: L6 },
    Group { first: 0xA830, last: 0xA83F, lookups: L34 },
    Group { first: 0xA840, last: 0xA87F, lookups: L63 },
    Group { first: 0xA880, last: 0xA8DF, lookups: L6 },
    Group { first: 0xA8E0, last: 0xA8FF, lookups: L34 },
    Group { first: 0xA900, last: 0xA95F, lookups: L6 },
    Group { first: 0xA960, last: 0xA97F, lookups: L23 },
    Group { first: 0xA980, last: 0xA9DF, lookups: L64 },
    Group { first: 0xA9E0, last: 0xA9FF, lookups: L22 },
    Group { first: 0xAA00, last: 0xAA5F, lookups: L6 },
    Group { first: 0xAA60, last: 0xAA7F, lookups: L22 },
    Group { first: 0xAA80, last: 0xAAFF, lookups: L6 },
    Group { first: 0xAB00, last: 0xAB2F, lookups: L24 },
    Group { first: 0xAB30, last: 0xAB6F, lookups: L0 },
    Group { first: 0xAB70, last: 0xABBF, lookups: L65 },
    Group { first: 0xABC0, last: 0xABFF, lookups: L6 },
    Group { first: 0xAC00, last: 0xD7AF, lookups: L23 },
    Group { first: 0xD7B0, last: 0xD7FF, lookups: L66 },
    Group { first: 0xD800, last: 0xDFFF, lookups: L8 },
    Group { first: 0xE000, last: 0xE76B, lookups: L67 },
    Group { first: 0xE76C, last: 0xE76C, lookups: L68 },
    Group { first: 0xE76D, last: 0xE78C, lookups: L67 },
    Group { first: 0xE78D, last: 0xE796, lookups: L68 },
    Group { first: 0xE797, last: 0xE7C6, lookups: L67 },
    Group { first: 0xE7C7, last: 0xE7C8, lookups: L68 },
    Group { first: 0xE7C9, last: 0xE7E6, lookups: L67 },
    Group { first: 0xE7E7, last: 0xE7F3, lookups: L68 },
    Group { first: 0xE7F4, last: 0xE814, lookups: L67 },
    Group { first: 0xE815, last: 0xE864, lookups: L68 },
    Group { first: 0xE865, last: 0xEEB7, lookups: L67 },
    Group { first: 0xEEB8, last: 0xF302, lookups: L8 },
    Group { first: 0xF303, last: 0xF34B, lookups: L67 },
    Group { first: 0xF34C, last: 0xF39F, lookups: L8 },
    Group { first: 0xF3A0, last: 0xF5F1, lookups: L67 },
    Group { first: 0xF5F2, last: 0xF633, lookups: L8 },
    Group { first: 0xF634, last: 0xF848, lookups: L67 },
    Group { first: 0xF849, last: 0xF8FF, lookups: L8 },
    Group { first: 0xF900, last: 0xFAFF, lookups: L69 },
    Group { first: 0xFB00, last: 0xFB0F, lookups: L0 },
    Group { first: 0xFB10, last: 0xFB1C, lookups: L70 },
    Group { first: 0xFB1D, last: 0xFB4F, lookups: L0 },
    Group { first: 0xFB50, last: 0xFBFF, lookups: L71 },
    Group { first: 0xFC00, last: 0xFDCF, lookups: L72 },
    Group { first: 0xFDD0, last: 0xFDEF, lookups: L8 },
    Group { first: 0xFDF0, last: 0xFDFF, lookups: L72 },
    Group { first: 0xFE00, last: 0xFE0F, lookups: L8 },
    Group { first: 0xFE10, last: 0xFE1F, lookups: L73 },
    Group { first: 0xFE20, last: 0xFE2F, lookups: L74 },
    Group { first: 0xFE30, last: 0xFE6F, lookups: L75 },
    Group { first: 0xFE70, last: 0xFEFE, lookups: L2 },
    Group { first: 0xFEFF, last: 0xFEFF, lookups: L62 },
    Group { first: 0xFF00, last: 0xFF60, lookups: L76 },
    Group { first: 0xFF61, last: 0xFF9F, lookups: L77 },
    Group { first: 0xFFA0, last: 0xFFDF, lookups: L78 },
    Group { first: 0xFFE0, last: 0xFFEF, lookups: L79 },
    Group { first: 0xFFF0, last: 0xFFFD, lookups: L80 },
    Group { first: 0xFFFE, last: 0xFFFF, lookups: L8 },
    Group { first: 0x10000, last: 0x1007F, lookups: L6 },
    Group { first: 0x10080, last: 0x100FF, lookups: L6 },
    Group { first: 0x10100, last: 0x1013F, lookups: L6 },
    Group { first: 0x10140, last: 0x1027F, lookups: L8 },
    Group { first: 0x10280, last: 0x102DF, lookups: L7 },
    Group { first: 0x102E0, last: 0x102FF, lookups: L6 },
    Group { first: 0x10300, last: 0x1034F, lookups: L27 },
    Group { first: 0x10350, last: 0x1037F, lookups: L6 },
    Group { first: 0x10380, last: 0x103DF, lookups: L7 },
    Group { first: 0x103E0, last: 0x103FF, lookups: L8 },
    Group { first: 0x10400, last: 0x1044F, lookups: L40 },
    Group { first: 0x10450, last: 0x1047F, lookups: L7 },
    Group { first: 0x10480, last: 0x104AF, lookups: L5 },
    Group { first: 0x104B0, last: 0x104FF, lookups: L65 },
    Group { first: 0x10500, last: 0x1077F, lookups: L6 },
    Group { first: 0x10780, last: 0x107BF, lookups: L81 },
    Group { first: 0x107C0, last: 0x107FF, lookups: L8 },
    Group { first: 0x10800, last: 0x1083F, lookups: L7 },
    Group { first: 0x10840, last: 0x1085F, lookups: L7 },
    Group { first: 0x10860, last: 0x108AF, lookups: L6 },
    Group { first: 0x108B0, last: 0x108DF, lookups: L8 },
    Group { first: 0x108E0, last: 0x108FF, lookups: L6 },
    Group { first: 0x10900, last: 0x1093F, lookups: L7 },
    Group { first: 0x10940, last: 0x1097F, lookups: L8 },
    Group { first: 0x10980, last: 0x109FF, lookups: L48 },
    Group { first: 0x10A00, last: 0x10A7F, lookups: L7 },
    Group { first: 0x10A80, last: 0x10B3F, lookups: L6 },
    Group { first: 0x10B40, last: 0x10B7F, lookups: L7 },
    Group { first: 0x10B80, last: 0x10BAF, lookups: L6 },
    Group { first: 0x10BB0, last: 0x10BFF, lookups: L8 },
    Group { first: 0x10C00, last: 0x10C4F, lookups: L48 },
    Group { first: 0x10C50, last: 0x10C7F, lookups: L8 },
    Group { first: 0x10C80, last: 0x10FFF, lookups: L6 },
    Group { first: 0x11000, last: 0x1107F, lookups: L7 },
    Group { first: 0x11080, last: 0x110CF, lookups: L6 },
    Group { first: 0x110D0, last: 0x110FF, lookups: L34 },
    Group { first: 0x11100, last: 0x1114F, lookups: L34 },
    Group { first: 0x11150, last: 0x111DF, lookups: L6 },
    Group { first: 0x111E0, last: 0x111FF, lookups: L34 },
    Group { first: 0x11200, last: 0x1124F, lookups: L6 },
    Group { first: 0x11250, last: 0x1127F, lookups: L8 },
    Group { first: 0x11280, last: 0x1165F, lookups: L6 },
    Group { first: 0x11660, last: 0x1167F, lookups: L29 },
    Group { first: 0x11680, last: 0x11AAF, lookups: L6 },
    Group { first: 0x11AB0, last: 0x11ABF, lookups: L65 },
    Group { first: 0x11AC0, last: 0x11AFF, lookups: L6 },
    Group { first: 0x11B00, last: 0x11BFF, lookups: L8 },
    Group { first: 0x11C00, last: 0x11FFF, lookups: L6 },
    Group { first: 0x12000, last: 0x1247F, lookups: L7 },
    Group { first: 0x12480, last: 0x12F8F, lookups: L8 },
    Group { first: 0x12F90, last: 0x12FFF, lookups: L6 },
    Group { first: 0x13000, last: 0x13438, lookups: L7 },
    Group { first: 0x13439, last: 0x143FF, lookups: L8 },
    Group { first: 0x14400, last: 0x1467F, lookups: L6 },
    Group { first: 0x14680, last: 0x167FF, lookups: L8 },
    Group { first: 0x16800, last: 0x16A3F, lookups: L7 },
    Group { first: 0x16A40, last: 0x16B8F, lookups: L6 },
    Group { first: 0x16B90, last: 0x16E3F, lookups: L8 },
    Group { first: 0x16E40, last: 0x16FFF, lookups: L6 },
    Group { first: 0x17000, last: 0x1AFFF, lookups: L8 },
    Group { first: 0x1B000, last: 0x1B12F, lookups: L58 },
    Group { first: 0x1B130, last: 0x1B16F, lookups: L8 },
    Group { first: 0x1B170, last: 0x1BCAF, lookups: L6 },
    Group { first: 0x1BCB0, last: 0x1CFFF, lookups: L8 },
    Group { first: 0x1D000, last: 0x1D0FF, lookups: L6 },
    Group { first: 0x1D100, last: 0x1D1FF, lookups: L40 },
    Group { first: 0x1D200, last: 0x1D24F, lookups: L6 },
    Group { first: 0x1D250, last: 0x1D2DF, lookups: L8 },
    Group { first: 0x1D2E0, last: 0x1D2FF, lookups: L6 },
    Group { first: 0x1D300, last: 0x1D35F, lookups: L40 },
    Group { first: 0x1D360, last: 0x1D3FF, lookups: L8 },
    Group { first: 0x1D400, last: 0x1D7FF, lookups: L82 },
    Group { first: 0x1D800, last: 0x1DEFF, lookups: L8 },
    Group { first: 0x1DF00, last: 0x1DFFF, lookups: L81 },
    Group { first: 0x1E000, last: 0x1E02F, lookups: L48 },
    Group { first: 0x1E030, last: 0x1E0FF, lookups: L8 },
    Group { first: 0x1E100, last: 0x1E2FF, lookups: L6 },
    Group { first: 0x1E300, last: 0x1E4CF, lookups: L8 },
    Group { first: 0x1E4D0, last: 0x1E4FF, lookups: L6 },
    Group { first: 0x1E500, last: 0x1E7DF, lookups: L8 },
    Group { first: 0x1E7E0, last: 0x1E7FF, lookups: L5 },
    Group { first: 0x1E800, last: 0x1E8DF, lookups: L6 },
    Group { first: 0x1E8E0, last: 0x1E8FF, lookups: L8 },
    Group { first: 0x1E900, last: 0x1E95F, lookups: L5 },
    Group { first: 0x1E960, last: 0x1EC6F, lookups: L8 },
    Group { first: 0x1EC70, last: 0x1ECBF, lookups: L6 },
    Group { first: 0x1ECC0, last: 0x1ECFF, lookups: L8 },
    Group { first: 0x1ED00, last: 0x1ED4F, lookups: L6 },
    Group { first: 0x1ED50, last: 0x1EFFF, lookups: L8 },
    Group { first: 0x1F000, last: 0x1F01F, lookups: L40 },
    Group { first: 0x1F020, last: 0x1F020, lookups: L83 },
    Group { first: 0x1F021, last: 0x1F18D, lookups: L40 },
    Group { first: 0x1F18E, last: 0x1F18E, lookups: L83 },
    Group { first: 0x1F18F, last: 0x1F190, lookups: L40 },
    Group { first: 0x1F191, last: 0x1F19A, lookups: L83 },
    Group { first: 0x1F19B, last: 0x1F1E5, lookups: L40 },
    Group { first: 0x1F1E6, last: 0x1F1FF, lookups: L83 },
    Group { first: 0x1F200, last: 0x1F200, lookups: L40 },
    Group { first: 0x1F201, last: 0x1F201, lookups: L83 },
    Group { first: 0x1F202, last: 0x1F219, lookups: L40 },
    Group { first: 0x1F21A, last: 0x1F21A, lookups: L83 },
    Group { first: 0x1F21B, last: 0x1F22E, lookups: L40 },
    Group { first: 0x1F22F, last: 0x1F22F, lookups: L83 },
    Group { first: 0x1F230, last: 0x1F231, lookups: L40 },
    Group { first: 0x1F232, last: 0x1F236, lookups: L83 },
    Group { first: 0x1F237, last: 0x1F237, lookups: L40 },
    Group { first: 0x1F238, last: 0x1F23A, lookups: L83 },
    Group { first: 0x1F23B, last: 0x1F24F, lookups: L40 },
    Group { first: 0x1F250, last: 0x1F251, lookups: L83 },
    Group { first: 0x1F252, last: 0x1F2FF, lookups: L40 },
    Group { first: 0x1F300, last: 0x1F321, lookups: L83 },
    Group { first: 0x1F322, last: 0x1F323, lookups: L40 },
    Group { first: 0x1F324, last: 0x1F393, lookups: L83 },
    Group { first: 0x1F394, last: 0x1F395, lookups: L40 },
    Group { first: 0x1F396, last: 0x1F397, lookups: L83 },
    Group { first: 0x1F398, last: 0x1F398, lookups: L40 },
    Group { first: 0x1F399, last: 0x1F39B, lookups: L83 },
    Group { first: 0x1F39C, last: 0x1F39D, lookups: L40 },
    Group { first: 0x1F39E, last: 0x1F3F0, lookups: L83 },
    Group { first: 0x1F3F1, last: 0x1F3F2, lookups: L40 },
    Group { first: 0x1F3F3, last: 0x1F3F5, lookups: L83 },
    Group { first: 0x1F3F6, last: 0x1F3F6, lookups: L40 },
    Group { first: 0x1F3F7, last: 0x1F4FD, lookups: L83 },
    Group { first: 0x1F4FE, last: 0x1F4FE, lookups: L40 },
    Group { first: 0x1F4FF, last: 0x1F53D, lookups: L83 },
    Group { first: 0x1F53E, last: 0x1F548, lookups: L40 },
    Group { first: 0x1F549, last: 0x1F54E, lookups: L83 },
    Group { first: 0x1F54F, last: 0x1F54F, lookups: L40 },
    Group { first: 0x1F550, last: 0x1F567, lookups: L83 },
    Group { first: 0x1F568, last: 0x1F56E, lookups: L40 },
    Group { first: 0x1F56F, last: 0x1F570, lookups: L83 },
    Group { first: 0x1F571, last: 0x1F572, lookups: L40 },
    Group { first: 0x1F573, last: 0x1F57A, lookups: L83 },
    Group { first: 0x1F57B, last: 0x1F586, lookups: L40 },
    Group { first: 0x1F587, last: 0x1F587, lookups: L83 },
    Group { first: 0x1F588, last: 0x1F589, lookups: L40 },
    Group { first: 0x1F58A, last: 0x1F58D, lookups: L83 },
    Group { first: 0x1F58E, last: 0x1F58F, lookups: L40 },
    Group { first: 0x1F590, last: 0x1F590, lookups: L83 },
    Group { first: 0x1F591, last: 0x1F594, lookups: L40 },
    Group { first: 0x1F595, last: 0x1F596, lookups: L83 },
    Group { first: 0x1F597, last: 0x1F5A3, lookups: L40 },
    Group { first: 0x1F5A4, last: 0x1F5A5, lookups: L83 },
    Group { first: 0x1F5A6, last: 0x1F5A7, lookups: L40 },
    Group { first: 0x1F5A8, last: 0x1F5A8, lookups: L83 },
    Group { first: 0x1F5A9, last: 0x1F5B0, lookups: L40 },
    Group { first: 0x1F5B1, last: 0x1F5B2, lookups: L83 },
    Group { first: 0x1F5B3, last: 0x1F5BB, lookups: L40 },
    Group { first: 0x1F5BC, last: 0x1F5BC, lookups: L83 },
    Group { first: 0x1F5BD, last: 0x1F5C1, lookups: L40 },
    Group { first: 0x1F5C2, last: 0x1F5C4, lookups: L83 },
    Group { first: 0x1F5C5, last: 0x1F5D0, lookups: L40 },
    Group { first: 0x1F5D1, last: 0x1F5D3, lookups: L83 },
    Group { first: 0x1F5D4, last: 0x1F5DB, lookups: L40 },
    Group { first: 0x1F5DC, last: 0x1F5DE, lookups: L83 },
    Group { first: 0x1F5DF, last: 0x1F5E0, lookups: L40 },
    Group { first: 0x1F5E1, last: 0x1F5E1, lookups: L83 },
    Group { first: 0x1F5E2, last: 0x1F5E2, lookups: L40 },
    Group { first: 0x1F5E3, last: 0x1F5E3, lookups: L83 },
    Group { first: 0x1F5E4, last: 0x1F5E7, lookups: L40 },
    Group { first: 0x1F5E8, last: 0x1F5E8, lookups: L83 },
    Group { first: 0x1F5E9, last: 0x1F5EE, lookups: L40 },
    Group { first: 0x1F5EF, last: 0x1F5EF, lookups: L83 },
    Group { first: 0x1F5F0, last: 0x1F5F2, lookups: L40 },
    Group { first: 0x1F5F3, last: 0x1F5F3, lookups: L83 },
    Group { first: 0x1F5F4, last: 0x1F5F9, lookups: L40 },
    Group { first: 0x1F5FA, last: 0x1F64F, lookups: L83 },
    Group { first: 0x1F650, last: 0x1F67F, lookups: L40 },
    Group { first: 0x1F680, last: 0x1F6C5, lookups: L83 },
    Group { first: 0x1F6C6, last: 0x1F6CA, lookups: L40 },
    Group { first: 0x1F6CB, last: 0x1F6D2, lookups: L83 },
    Group { first: 0x1F6D3, last: 0x1F6D4, lookups: L40 },
    Group { first: 0x1F6D5, last: 0x1F6D7, lookups: L83 },
    Group { first: 0x1F6D8, last: 0x1F6DB, lookups: L40 },
    Group { first: 0x1F6DC, last: 0x1F6E5, lookups: L83 },
    Group { first: 0x1F6E6, last: 0x1F6E8, lookups: L40 },
    Group { first: 0x1F6E9, last: 0x1F6E9, lookups: L83 },
    Group { first: 0x1F6EA, last: 0x1F6EA, lookups: L40 },
    Group { first: 0x1F6EB, last: 0x1F6EC, lookups: L83 },
    Group { first: 0x1F6ED, last: 0x1F6EF, lookups: L40 },
    Group { first: 0x1F6F0, last: 0x1F6F0, lookups: L83 },
    Group { first: 0x1F6F1, last: 0x1F6F2, lookups: L40 },
    Group { first: 0x1F6F3, last: 0x1F6FC, lookups: L83 },
    Group { first: 0x1F6FD, last: 0x1F6FF, lookups: L40 },
    Group { first: 0x1F700, last: 0x1F77F, lookups: L40 },
    Group { first: 0x1F780, last: 0x1F7DF, lookups: L39 },
    Group { first: 0x1F7E0, last: 0x1F7EB, lookups: L83 },
    Group { first: 0x1F7EC, last: 0x1F7EF, lookups: L40 },
    Group { first: 0x1F7F0, last: 0x1F7F0, lookups: L83 },
    Group { first: 0x1F7F1, last: 0x1F7FF, lookups: L40 },
    Group { first: 0x1F800, last: 0x1F8FF, lookups: L46 },
    Group { first: 0x1F900, last: 0x1F90B, lookups: L40 },
    Group { first: 0x1F90C, last: 0x1F93A, lookups: L83 },
    Group { first: 0x1F93B, last: 0x1F93B, lookups: L40 },
    Group { first: 0x1F93C, last: 0x1F945, lookups: L83 },
    Group { first: 0x1F946, last: 0x1F946, lookups: L40 },
    Group { first: 0x1F947, last: 0x1F9FF, lookups: L83 },
    Group { first: 0x1FA00, last: 0x1FA6F, lookups: L8 },
    Group { first: 0x1FA70, last: 0x1FA7C, lookups: L83 },
    Group { first: 0x1FA7D, last: 0x1FA7F, lookups: L40 },
    Group { first: 0x1FA80, last: 0x1FA88, lookups: L83 },
    Group { first: 0x1FA89, last: 0x1FA8F, lookups: L40 },
    Group { first: 0x1FA90, last: 0x1FABD, lookups: L83 },
    Group { first: 0x1FABE, last: 0x1FABE, lookups: L40 },
    Group { first: 0x1FABF, last: 0x1FAC5, lookups: L83 },
    Group { first: 0x1FAC6, last: 0x1FACD, lookups: L40 },
    Group { first: 0x1FACE, last: 0x1FADB, lookups: L83 },
    Group { first: 0x1FADC, last: 0x1FADF, lookups: L40 },
    Group { first: 0x1FAE0, last: 0x1FAE8, lookups: L83 },
    Group { first: 0x1FAE9, last: 0x1FAEF, lookups: L40 },
    Group { first: 0x1FAF0, last: 0x1FAF8, lookups: L83 },
    Group { first: 0x1FAF9, last: 0x1FAFF, lookups: L40 },
    Group { first: 0x1FB00, last: 0x1FFFF, lookups: L8 },
    Group { first: 0x20000, last: 0x2A6DF, lookups: L84 },
    Group { first: 0x2A6E0, last: 0x2A6FF, lookups: L8 },
    Group { first: 0x2A700, last: 0x2B81F, lookups: L85 },
    Group { first: 0x2B820, last: 0x2EBEF, lookups: L86 },
    Group { first: 0x2EBF0, last: 0x2F7FF, lookups: L8 },
    Group { first: 0x2F800, last: 0x2FA1F, lookups: L87 },
    Group { first: 0x2FA20, last: 0xFFFFFFFF, lookups: L8 },
];

/// Code-point ranges dwrite's `FONTFALLBACK` resource maps (primary lookup) to **Segoe UI Emoji**,
/// with **Segoe UI Symbol** as the secondary — baked from the same resource (all 127 ranges whose
/// first lookup family is Segoe UI Emoji). Both lookups are locale-neutral, so the candidate list is
/// the same for every locale. The BMP symbol ranges (0x2139–0x2B55) carry extra locale-gated CJK
/// lookups in the resource, but those only win for a character Segoe UI Emoji does not cover, which no
/// emoji character is — so the resolved family is Segoe UI Emoji for every locale and they are not
/// modelled. Sorted, non-overlapping; binary-searched.
static EMOJI: &[(u32, u32)] = &[
    (0x2139, 0x2139), (0x231A, 0x231B), (0x2328, 0x2328), (0x23CF, 0x23CF), (0x23E9, 0x23F3),
    (0x23F8, 0x23FA), (0x25FD, 0x25FE), (0x2600, 0x2604), (0x260E, 0x260E), (0x2611, 0x2611),
    (0x2614, 0x2615), (0x2618, 0x2618), (0x2620, 0x2620), (0x2622, 0x2623), (0x2626, 0x2626),
    (0x262A, 0x262A), (0x262E, 0x262F), (0x2638, 0x263A), (0x2640, 0x2640), (0x2642, 0x2642),
    (0x2648, 0x2653), (0x265F, 0x265F), (0x2668, 0x2668), (0x267B, 0x267B), (0x267E, 0x267F),
    (0x2692, 0x2697), (0x2699, 0x2699), (0x269B, 0x269C), (0x26A0, 0x26A1), (0x26A7, 0x26A7),
    (0x26AA, 0x26AB), (0x26B0, 0x26B1), (0x26BD, 0x26BE), (0x26C4, 0x26C5), (0x26C8, 0x26C8),
    (0x26CE, 0x26CF), (0x26D1, 0x26D1), (0x26D3, 0x26D4), (0x26E9, 0x26EA), (0x26F0, 0x26F5),
    (0x26F7, 0x26F8), (0x26FA, 0x26FA), (0x26FD, 0x26FD), (0x2702, 0x2702), (0x2705, 0x2705),
    (0x2708, 0x270C), (0x270F, 0x270F), (0x2712, 0x2712), (0x2714, 0x2714), (0x2716, 0x2716),
    (0x271D, 0x271D), (0x2721, 0x2721), (0x2728, 0x2728), (0x2733, 0x2734), (0x2744, 0x2744),
    (0x2747, 0x2747), (0x274C, 0x274C), (0x274E, 0x274E), (0x2753, 0x2755), (0x2757, 0x2757),
    (0x2763, 0x2764), (0x2795, 0x2797), (0x27A1, 0x27A1), (0x27B0, 0x27B0), (0x27BF, 0x27BF),
    (0x2B1B, 0x2B1C), (0x2B50, 0x2B50), (0x2B55, 0x2B55), (0x1F020, 0x1F020), (0x1F18E, 0x1F18E),
    (0x1F191, 0x1F19A), (0x1F1E6, 0x1F1FF), (0x1F201, 0x1F201), (0x1F21A, 0x1F21A), (0x1F22F, 0x1F22F),
    (0x1F232, 0x1F236), (0x1F238, 0x1F23A), (0x1F250, 0x1F251), (0x1F300, 0x1F321), (0x1F324, 0x1F393),
    (0x1F396, 0x1F397), (0x1F399, 0x1F39B), (0x1F39E, 0x1F3F0), (0x1F3F3, 0x1F3F5), (0x1F3F7, 0x1F4FD),
    (0x1F4FF, 0x1F53D), (0x1F549, 0x1F54E), (0x1F550, 0x1F567), (0x1F56F, 0x1F570), (0x1F573, 0x1F57A),
    (0x1F587, 0x1F587), (0x1F58A, 0x1F58D), (0x1F590, 0x1F590), (0x1F595, 0x1F596), (0x1F5A4, 0x1F5A5),
    (0x1F5A8, 0x1F5A8), (0x1F5B1, 0x1F5B2), (0x1F5BC, 0x1F5BC), (0x1F5C2, 0x1F5C4), (0x1F5D1, 0x1F5D3),
    (0x1F5DC, 0x1F5DE), (0x1F5E1, 0x1F5E1), (0x1F5E3, 0x1F5E3), (0x1F5E8, 0x1F5E8), (0x1F5EF, 0x1F5EF),
    (0x1F5F3, 0x1F5F3), (0x1F5FA, 0x1F64F), (0x1F680, 0x1F6C5), (0x1F6CB, 0x1F6D2), (0x1F6D5, 0x1F6D7),
    (0x1F6DC, 0x1F6E5), (0x1F6E9, 0x1F6E9), (0x1F6EB, 0x1F6EC), (0x1F6F0, 0x1F6F0), (0x1F6F3, 0x1F6FC),
    (0x1F7E0, 0x1F7EB), (0x1F7F0, 0x1F7F0), (0x1F90C, 0x1F93A), (0x1F93C, 0x1F945), (0x1F947, 0x1F9FF),
    (0x1FA70, 0x1FA7C), (0x1FA80, 0x1FA88), (0x1FA90, 0x1FABD), (0x1FABF, 0x1FAC5), (0x1FACE, 0x1FADB),
    (0x1FAE0, 0x1FAE8), (0x1FAF0, 0x1FAF8),
];

/// True if `cp` is an emoji code point dwrite falls back to Segoe UI Emoji.
fn in_emoji(cp: u32) -> bool {
    use core::cmp::Ordering;
    EMOJI.binary_search_by(|&(a, b)| {
        if cp < a { Ordering::Greater } else if cp > b { Ordering::Less } else { Ordering::Equal }
    }).is_ok()
}

/// The range group whose `[first,last]` contains `cp`, if any (the table is sorted and non-overlapping).
fn group_of(cp: u32) -> Option<&'static Group> {
    use core::cmp::Ordering;
    let i = TABLE.binary_search_by(|g| if cp < g.first { Ordering::Greater } else if cp > g.last { Ordering::Less } else { Ordering::Equal }).ok()?;
    Some(&TABLE[i])
}

/// DirectWrite's locale-range inclusion test, narrowed to the locales with
/// scenes: an empty range matches anything; otherwise `range` must be a BCP-47 subtag prefix of the
/// locale or one of its script/parent ancestors. Native resolves the ancestor chain through
/// `GetLocaleInfoEx(LOCALE_SPARENT)`; we hard-code the chains for the tested locales.
/// ponytail: full CLDR likely-subtags / parent resolution isn't reproduced — add a locale's chain
/// here when a scene needs it.
fn locale_matches(range: &str, locale: &str) -> bool {
    if range.is_empty() {
        return true;
    }
    let range = range.to_ascii_lowercase();
    locale_chain(locale).iter().any(|tag| *tag == range || tag.starts_with(&format!("{range}-")))
}

/// A locale's own tag plus its script/parent ancestors, lowercased (closest first).
fn locale_chain(locale: &str) -> Vec<String> {
    let l = locale.to_ascii_lowercase();
    let mut chain = vec![l.clone()];
    let primary = l.split('-').next().unwrap_or("").to_string();
    // zh needs its script inserted (zh-CN → zh-Hans → zh) so a "zh-hans" range matches zh-CN.
    if primary == "zh" {
        let hant = matches!(l.as_str(), "zh-tw" | "zh-hk" | "zh-mo" | "zh-hant");
        chain.push(format!("zh-{}", if hant { "hant" } else { "hans" }));
    }
    if primary != l {
        chain.push(primary);
    }
    chain
}

/// Candidate families for `cp` under `locale`, in the order native tries them: the group's
/// locale-matching lookups concatenated in table order. Empty if `cp` is in no baked group.
pub fn candidates(cp: u32, locale: &str) -> Vec<&'static str> {
    if in_emoji(cp) {
        // Both resource lookups are locale-neutral; Segoe UI Emoji then Segoe UI Symbol.
        return vec!["Segoe UI Emoji", "Segoe UI Symbol"];
    }
    let Some(g) = group_of(cp) else { return Vec::new() };
    let mut out = Vec::new();
    for lk in g.lookups {
        if locale_matches(lk.locale, locale) {
            out.extend_from_slice(lk.families);
        }
    }
    out
}

/// A resolved text span: `[start,end)` bytes to shape with `family` (`None` = the base family).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub family: Option<&'static str>,
}

/// Split `text` into spans each resolved to one family, reproducing `MapCharacters`. `base_covers`
/// reports whether the base family's font has a glyph for a character; `covers(family, ch)` the same
/// for a named fallback family, and first for [`OWN_FAMILY`] (the base family's other faces). A character the
/// base covers stays base (`None`); otherwise the first candidate family that covers it wins; if none does it
/// stays base (its `.notdef`). Consecutive
/// characters resolving to the same family share a span.
pub fn map_characters(
    text: &str,
    locale: &str,
    base_covers: impl Fn(char) -> bool,
    covers: impl Fn(&str, char) -> bool,
) -> Vec<Span> {
    map_characters_at(text, locale, |_, ch| base_covers(ch), |_| 0, |f, _, ch| covers(f, ch))
}

/// The family name a [`Span`] carries for the format's own family: a character its format face lacks is
/// first looked for in another face of that family (the best-matching one that has it), and only then in the
/// fallback list (r25 30471: Times New Roman Bold Italic has no Arabic, so Arabic takes Times New Roman Bold).
pub const OWN_FAMILY: &str = "";

/// [`map_characters`] where the font a character would otherwise use depends on its byte position (a
/// per-range family): `own_covers(byte, ch)` reports whether that font has the character, and
/// `range_of(byte)` identifies the format range a byte is in (native maps each format range on its own).
/// `covers(family, byte, ch)` is asked for [`OWN_FAMILY`] before the fallback list.
pub fn map_characters_at(
    text: &str,
    locale: &str,
    own_covers: impl Fn(usize, char) -> bool,
    range_of: impl Fn(usize) -> usize,
    covers: impl Fn(&str, usize, char) -> bool,
) -> Vec<Span> {
    let resolve = |b: usize, ch: char| -> Option<&'static str> {
        if own_covers(b, ch) {
            return None;
        }
        core::iter::once(OWN_FAMILY).chain(candidates(ch as u32, locale)).find(|f| covers(f, b, ch))
    };
    let mut out: Vec<Span> = Vec::new();
    for (b, ch) in text.char_indices() {
        // A joining character (ZWJ, a variation selector, a skin-tone modifier, an emoji tag, the keycap)
        // stays in its cluster's font when that font has it and the cluster is in the same format range: an
        // emoji ZWJ sequence keeps its ZWJs in the Segoe UI Emoji run (r25 30028), not in a base face that
        // also maps them, while a ZWJ between two ranges keeps the base (r25 30086).
        let joins = matches!(ch as u32, 0x200D | 0xFE00..=0xFE0F | 0x1F3FB..=0x1F3FF | 0xE0020..=0xE007F | 0x20E3);
        let prev = out.last().filter(|s| range_of(s.end - 1) == range_of(b)).and_then(|s| s.family);
        let fam = match prev {
            Some(f) if joins && covers(f, b, ch) => Some(f),
            _ => resolve(b, ch),
        };
        let end = b + ch.len_utf8();
        match out.last_mut() {
            // A family's face is each format range's own, so a span ends at a range boundary.
            Some(s) if s.family == fam && (fam.is_none() || range_of(s.end - 1) == range_of(b)) => s.end = end,
            _ => out.push(Span { start: b, end, family: fam }),
        }
    }
    out
}

/// The order font fallback tries a family's faces in (the WWS `IDWriteFontFamily::GetMatchingFonts` list,
/// whose first face mapping the text is the fallback's face). `faces` are the family's real faces as
/// `(weight, stretch, style)` in font-set order (the system set's is the registry's: [`set_order_key`]).
/// Each takes the simulations the request calls for: bold when `weight` is over 550 and the face is at most
/// 550 and more than 100 lighter (it then counts as the heavier of itself and 700), oblique when a slant is
/// asked of an upright face. Faces rank by squared distance from the request in (1100·stretch, 700·style,
/// 5·(weight − 400)), ties going to the larger dot product with it and then the larger coordinates, after two
/// preferences between faces of the same stretch whose simulations differ: a real slant over a simulated one
/// at the same effective weight, and a real weight over 550 over a simulated bold at the same slant. Those
/// make the order intransitive, so the list is what an insertion sort in set order leaves (r25 30966: Segoe UI
/// at 560 italic takes Bold with a simulated oblique for Hebrew, ahead of Semibold). Returns
/// `(index into faces, DWRITE_FONT_SIMULATIONS)`.
// ponytail: native sorts a family of more than 32 faces with introsort, and drops a variable font's named
// instances whose stretch, weight or style differ from the best one's; neither is modelled (no fallback
// family has that many faces or is variable).
pub fn wws_fallback_order(faces: &[(u16, u16, u8)], weight: u16, stretch: u16, style: u8) -> Vec<(usize, u16)> {
    struct Cand {
        idx: usize,
        weight: i32,
        stretch: u16,
        style: u8,
        sims: u16,
        at: [i32; 3],
    }
    let want = [1100 * (stretch as i32 - 5), 700 * style as i32, 5 * (weight as i32 - 400)];
    let cands = faces.iter().enumerate().map(|(idx, &(w, st, s))| {
        let w = w as i32;
        let sims = u16::from(weight > 550 && w <= 550 && w < weight as i32 - 100) | if style != 0 && s == 0 { 2 } else { 0 };
        let w_eff = if sims & 1 != 0 { w.max(700) } else { w };
        let s_eff = if sims & 2 != 0 { 1 } else { s as i32 };
        Cand { idx, weight: w, stretch: st, style: s, sims, at: [1100 * (st as i32 - 5), 700 * s_eff, 5 * (w_eff - 400)] }
    });
    let better = |a: &Cand, b: &Cand| -> bool {
        if a.sims != b.sims && a.stretch == b.stretch {
            let (a_obl, b_obl) = (a.sims & 2 != 0, b.sims & 2 != 0);
            if a_obl != b_obl {
                let eff = |c: &Cand| if c.sims & 1 != 0 { c.weight.max(700) } else { c.weight };
                if eff(a) == eff(b) {
                    if a_obl {
                        if b.style == 1 {
                            return false;
                        }
                    } else if a.style == 1 {
                        return true;
                    }
                }
            }
            let a_bold = a.sims & 1 != 0;
            if a_bold != (b.sims & 1 != 0) {
                let slant = |c: &Cand, obl: bool| if obl { 1 } else { c.style };
                if slant(a, a_obl) == slant(b, b_obl) {
                    if !a_bold {
                        if a.weight > 550 {
                            return true;
                        }
                    } else if b.weight > 550 {
                        return false;
                    }
                }
            }
        }
        let dist = |c: &Cand| (0..3).map(|i| (c.at[i] - want[i]) * (c.at[i] - want[i])).sum::<i32>();
        let dot = |c: &Cand| (0..3).map(|i| c.at[i] * want[i]).sum::<i32>();
        match (dist(a).cmp(&dist(b)), dot(a).cmp(&dot(b))) {
            (core::cmp::Ordering::Equal, core::cmp::Ordering::Equal) => a.at > b.at,
            (core::cmp::Ordering::Equal, d) => d.is_gt(),
            (d, _) => d.is_lt(),
        }
    };
    let mut out: Vec<Cand> = Vec::new();
    for c in cands {
        // Native's insertion sort: a new best goes to the front outright; otherwise it walks back from the end.
        let mut j = out.len();
        if out.first().is_some_and(|f| better(&c, f)) {
            j = 0;
        } else {
            while j > 0 && better(&c, &out[j - 1]) {
                j -= 1;
            }
        }
        out.insert(j, c);
    }
    out.into_iter().map(|c| (c.idx, c.sims)).collect()
}

/// A face file's place in the system font set, which lists fonts in registry order: by the name the
/// registry gives the file, its first face's English full name, case-insensitively.
pub fn set_order_key(file: &[u8]) -> String {
    skrifa::FontRef::from_index(file, 0)
        .ok()
        .and_then(|f| skrifa::MetadataProvider::localized_strings(&f, skrifa::string::StringId::FULL_NAME).english_or_first().map(|s| s.to_string().to_lowercase()))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wws_fallback_order_matches_native_segoe_ui_lists() {
        // Segoe UI's faces in the system set's order, and the lists native's GetMatchingFonts returned on the
        // reference VM (L light, SL semilight, SB semibold, K black; I italic; +b/+o simulated bold/oblique).
        let names = ["R", "K", "KI", "B", "BI", "I", "L", "LI", "SB", "SBI", "SL", "SLI"];
        let faces = [(400, 5, 0), (900, 5, 0), (900, 5, 2), (700, 5, 0), (700, 5, 2), (400, 5, 2), (300, 5, 0), (300, 5, 2), (600, 5, 0), (600, 5, 2), (350, 5, 0), (350, 5, 2)];
        let list = |w, s| {
            wws_fallback_order(&faces, w, 5, s).into_iter().map(|(i, sims)| format!("{}{}{}", names[i], if sims & 1 != 0 { "+b" } else { "" }, if sims & 2 != 0 { "+o" } else { "" })).collect::<Vec<_>>().join(" ")
        };
        assert_eq!(list(550, 2), "SBI SB+o BI I SLI B+o R+o SL+o LI L+o KI K+o");
        assert_eq!(list(560, 2), "SBI BI B+o KI I+b LI+b SLI+b SB+o K+o R+b+o L+b+o SL+b+o");
        assert_eq!(list(600, 0), "SB B K R+b L+b SL+b SBI BI KI I+b LI+b SLI+b");
        assert_eq!(list(600, 1), "SB+o B+o K+o R+b+o L+b+o SL+b+o SBI BI KI I+b LI+b SLI+b");
    }

    #[test]
    fn candidate_lists_match_table() {
        // en-US Han → the locale-neutral list (zh/ja/ko lookups filtered out): Yu Gothic UI first.
        assert_eq!(candidates(0x65E5, "en-US")[0], "Yu Gothic UI");
        // ko-KR Han → Malgun Gothic first (the "ko" lookup precedes the neutral one).
        assert_eq!(candidates(0x65E5, "ko-KR")[0], "Malgun Gothic");
        // zh-CN Han → Microsoft YaHei UI (zh-CN resolves to the zh-hans lookup).
        assert_eq!(candidates(0x4E2D, "zh-CN")[0], "Microsoft YaHei UI");
        // zh-TW Han → Microsoft JhengHei UI (zh-hant).
        assert_eq!(candidates(0x4E2D, "zh-TW")[0], "Microsoft JhengHei UI");
        // Hangul is locale-neutral → Malgun Gothic for any locale.
        assert_eq!(candidates(0xD55C, "en-US")[0], "Malgun Gothic");
        // Every range has its resource lookups: Armenian, a combining mark (with Latin, Greek and Cyrillic),
        // and an arrow, whose neutral lookup follows the locale-gated CJK ones.
        assert_eq!(candidates(0x0531, "en-US")[0], "Segoe UI");
        assert_eq!(candidates(0x0301, "en-US")[0], "Segoe UI");
        assert_eq!(candidates(0x2192, "en-US")[0], "Segoe UI");
        assert_eq!(candidates(0x2192, "ja-JP")[0], "Yu Gothic UI");
        // Emoji → Segoe UI Emoji then Segoe UI Symbol, for every locale. U+1F357 (🍗, the Calendar
        // special-date glyph) sits in 0x1F324..0x1F393, which the old stub range (0x1F5FA..) missed.
        assert_eq!(candidates(0x1F357, "en-US"), vec!["Segoe UI Emoji", "Segoe UI Symbol"]);
        assert_eq!(candidates(0x1F357, "ja-JP"), vec!["Segoe UI Emoji", "Segoe UI Symbol"]);
        assert_eq!(candidates(0x1F383, "en-US")[0], "Segoe UI Emoji"); // 🎃
        assert_eq!(candidates(0x1F600, "en-US")[0], "Segoe UI Emoji"); // 😀 (old stub range)
        // A non-emoji code point between emoji ranges stays unmapped (e.g. 0x1F322, a reserved gap).
        assert!(!in_emoji(0x1F322) && !in_emoji(0x2605));
    }

    #[test]
    fn map_characters_splits_latin_then_fallback() {
        // "Hi 日本語": Arial covers "Hi " (ASCII), not the Han; the Han resolves to Yu Gothic UI.
        let base = |ch: char| ch.is_ascii();
        let covers = |fam: &str, _ch: char| fam == "Yu Gothic UI";
        let spans = map_characters("Hi 日本語", "en-US", base, covers);
        assert_eq!(spans, vec![
            Span { start: 0, end: 3, family: None },
            Span { start: 3, end: 12, family: Some("Yu Gothic UI") },
        ]);
    }
}
