//! Project conformance meter: drive every captured oracle scene through the **public API**
//! (`Factory` → `Device` → `DeviceContext`, Begin/Clear/…/EndDraw) exactly as the harness drives
//! native d2d1, and compare byte-for-byte against the conformance suite's oracle captures: the raw BGRA
//! pixels, the v2 per-op HRESULT/tag channel, the EndDraw result, and geometry query results.
//!
//! Honesty rules (phase 1E.2):
//!   * **SKIP** means only that a scene uses an op or feature whose `DeviceContext` API does not
//!     exist yet — never a pixel/HRESULT/query mismatch.
//!   * **FAIL** is any mismatch, reported with the first differing pixel + channel (or the metadata
//!     diff). A near-miss is a FAIL, not a skip.
//!   * the suite's `passing.txt` lists the scenes currently byte-exact. `cargo test` stays green as
//!     long as none of those regress; new FAILs (not-yet-exact work) are printed but tolerated.
//!
//! As 1C geometry and 1D brushes landed on main, their ops are dispatched here: gradient and
//! bitmap/image brushes (rectangle fills), ellipses, rounded rectangles, curved path fills, strokes
//! and the `ComputeArea`/`ComputeLength`/`GetBounds`/`FillContainsPoint` query kinds.

use d2d::bitmap::Bitmap;
use d2d::color_context::{ColorContext, ColorTransform, SimpleProfile};
use d2d::effects::builtin::color;
use d2d::effects::{Effect as FxEffect, Input as FxInput, Prop as FxProp};
use d2d::geometry::stroke::{Cap, Join, TransformType};
use d2d::geometry::{self, Geometry, StrokeStyle, DEFAULT_TOLERANCE, IDENTITY};
use d2d::extras::image_source::ImageSource;
use d2d::extras::print;
use d2d::extras::{self, SpriteBatch};
use d2d::text::fallback;
use d2d::text::layout::{self, FontSpan, TextFormat};
use d2d::text::shaping::Font as ShapeFont;
use d2d::text::{FaceSpan, Glyph as TGlyph, GlyphRun as TGlyphRun, RenderingParams};
use d2d::{bitmap_target, CommandList, Device, DeviceContext, DrawingStateBlock, Factory, PixelBrush, SolidColorBrush};
use d2d_reference::{Figure, Op, Scene, Segment, Shape, SvgObject, Value};
use serde::Deserialize;
use skrifa::instance::{LocationRef, Size};
use skrifa::{FontRef, GlyphId, MetadataProvider};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use windows_core::GUID;

// --- text: font resolution + glyph-run building (phase 2D-3) ------------------------------------
// Fonts are the VM's own files (gitignored, shared at ref/windows/Fonts); text scenes skip cleanly
// when the font is absent (e.g. wasm, CI). Family→file covers the corpus (Arial, Times New Roman,
// Courier New, Segoe UI Emoji); the regular face serves any weight/style we have no variant file
// for, so a bold/italic Times scene would FAIL honestly rather than silently substitute.

#[cfg(not(target_arch = "wasm32"))]
fn fonts_dir() -> Option<std::path::PathBuf> {
    load_thai_dictionary();
    if let Ok(p) = std::env::var("D2D_FONTS") {
        return Some(p.into());
    }
    let out = std::process::Command::new("git")
        .args(["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "--path-format=absolute", "--git-common-dir"])
        .output()
        .ok()?;
    let gitdir = std::path::PathBuf::from(String::from_utf8(out.stdout).ok()?.trim());
    let dir = gitdir.parent()?.join("ref/windows/Fonts");
    dir.is_dir().then_some(dir)
}

// DirectWrite's Thai word breaker reads its dictionaries from dwrite.dll's resources. They are not shipped,
// so the breaker is strictly opt-in: only a dwrite.dll named by `D2D_THAI_DICT` loads them, and every other
// run (and wasm) breaks Thai as before.
#[cfg(not(target_arch = "wasm32"))]
fn load_thai_dictionary() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if let Some(dll) = std::env::var_os("D2D_THAI_DICT").and_then(|p| std::fs::read(p).ok()) {
            assert!(d2d::text::thai::set_dictionary_from_dwrite(&dll), "D2D_THAI_DICT has no Thai dictionaries");
        }
    });
}
#[cfg(target_arch = "wasm32")]
fn load_thai_dictionary() {}

// The real faces of each family in ref/windows/Fonts as `(file, weight, italic)`, in file order. Native
// resolves fonts through our drop-in dwrite's system-font matcher (`load_font_sims` →
// `dwrite::resolve_face_sims`); wasm has no filesystem, so it matches against this list instead
// (`font_file_sims`), and `font_map_matches_matcher` keeps the two in step. Bahnschrift is one variable
// file whose named instances span 300-700.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn family_faces(family: &str) -> Option<&'static [(&'static str, u32, u32, bool)]> {
    // `(file, face index, weight, italic)`.
    Some(match family.to_ascii_lowercase().as_str() {
        "arial" => &[("arial.ttf", 0, 400, false), ("arialbd.ttf", 0, 700, false), ("arialbi.ttf", 0, 700, true), ("ariali.ttf", 0, 400, true), ("ariblk.ttf", 0, 900, false)],
        "times new roman" => &[("times.ttf", 0, 400, false), ("timesbd.ttf", 0, 700, false), ("timesbi.ttf", 0, 700, true), ("timesi.ttf", 0, 400, true)],
        "consolas" => &[("consola.ttf", 0, 400, false), ("consolab.ttf", 0, 700, false), ("consolai.ttf", 0, 400, true), ("consolaz.ttf", 0, 700, true)],
        "segoe ui" => &[
            ("segoeui.ttf", 0, 400, false), ("segoeuib.ttf", 0, 700, false), ("segoeuii.ttf", 0, 400, true), ("segoeuil.ttf", 0, 300, false),
            ("segoeuisl.ttf", 0, 350, false), ("segoeuiz.ttf", 0, 700, true), ("seguibl.ttf", 0, 900, false), ("seguibli.ttf", 0, 900, true),
            ("seguili.ttf", 0, 300, true), ("seguisb.ttf", 0, 600, false), ("seguisbi.ttf", 0, 600, true), ("seguisli.ttf", 0, 350, true),
        ],
        "georgia" => &[("georgia.ttf", 0, 400, false), ("georgiab.ttf", 0, 700, false), ("georgiai.ttf", 0, 400, true), ("georgiaz.ttf", 0, 700, true)],
        "bahnschrift" => &[("bahnschrift.ttf", 0, 300, false), ("bahnschrift.ttf", 0, 400, false), ("bahnschrift.ttf", 0, 700, false)],
        "courier new" => &[("cour.ttf", 0, 400, false), ("courbd.ttf", 0, 700, false), ("courbi.ttf", 0, 700, true), ("couri.ttf", 0, 400, true)],
        "segoe ui emoji" => &[("seguiemj.ttf", 0, 400, false)],
        "segoe ui symbol" => &[("seguisym.ttf", 0, 400, false)],
        "cambria" => &[("cambria.ttc", 0, 400, false), ("cambriab.ttf", 0, 700, false), ("cambriai.ttf", 0, 400, true), ("cambriaz.ttf", 0, 700, true)],
        "calibri" => &[
            ("calibri.ttf", 0, 400, false), ("calibrib.ttf", 0, 700, false), ("calibrii.ttf", 0, 400, true), ("calibril.ttf", 0, 300, false),
            ("calibrili.ttf", 0, 300, true), ("calibriz.ttf", 0, 700, true),
        ],
        "lucida console" => &[("lucon.ttf", 0, 400, false)],
        "trebuchet ms" => &[("trebuc.ttf", 0, 400, false), ("trebucbd.ttf", 0, 700, false), ("trebucbi.ttf", 0, 700, true), ("trebucit.ttf", 0, 400, true)],
        "tahoma" => &[("tahoma.ttf", 0, 400, false), ("tahomabd.ttf", 0, 700, false)],
        "verdana" => &[("verdana.ttf", 0, 400, false), ("verdanab.ttf", 0, 700, false), ("verdanai.ttf", 0, 400, true), ("verdanaz.ttf", 0, 700, true)],
        "palatino linotype" => &[("pala.ttf", 0, 400, false), ("palab.ttf", 0, 700, false), ("palabi.ttf", 0, 700, true), ("palai.ttf", 0, 400, true)],
        "impact" => &[("impact.ttf", 0, 400, false)],
        "comic sans ms" => &[("comic.ttf", 0, 400, false), ("comicbd.ttf", 0, 700, false), ("comici.ttf", 0, 400, true), ("comicz.ttf", 0, 700, true)],
        // Font-fallback families: each collection's second face is Yu Gothic UI (YuGothB's third its Semibold).
        "yu gothic ui" => &[("YuGothB.ttc", 1, 700, false), ("YuGothB.ttc", 2, 600, false), ("YuGothL.ttc", 1, 300, false), ("YuGothM.ttc", 1, 400, false), ("YuGothR.ttc", 1, 350, false)],
        "malgun gothic" => &[("malgun.ttf", 0, 400, false), ("malgunbd.ttf", 0, 700, false), ("malgunsl.ttf", 0, 300, false)],
        "leelawadee ui" => &[("LeelaUIb.ttf", 0, 700, false), ("LeelawUI.ttf", 0, 400, false), ("LeelUIsl.ttf", 0, 350, false)],
        "nirmala ui" => &[("Nirmala.ttc", 0, 400, false), ("Nirmala.ttc", 1, 700, false), ("Nirmala.ttc", 2, 350, false)],
        _ => return None,
    })
}
// A family as native's collection holds it: its real faces from `family_faces`, then the faces native
// synthesizes (a bold of the regular in a style group with a lighter face but no 700; an oblique of every
// upright face; a bold oblique of a synthesized upright bold), in native's face order (real first, then
// by weight and style): `(file, face, weight, style, sims)`. `cold` restricts the real faces to that list,
// as [`cold_faces`] does for the native matcher.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn family_entries(family: &str, cold: Option<&[String]>) -> Option<Vec<(&'static str, u32, u32, u32, u32)>> {
    let reals: Vec<_> = family_faces(family)?
        .iter()
        .filter(|f| cold.is_none_or(|c| c.iter().any(|n| n == f.0)))
        .map(|&(f, i, w, it)| (f, i, w, if it { 2 } else { 0 }, 0))
        .collect();
    let (mut sims, mut bold_upright) = (Vec::new(), Vec::new());
    for (k, r) in reals.iter().enumerate() {
        let group = || reals.iter().enumerate().filter(|(_, g)| g.3 == r.3);
        let regular = group().min_by_key(|(_, g)| (g.2 as i32 - 400).abs()).map(|(j, _)| j);
        if regular == Some(k) && group().any(|(_, g)| g.2 < 700) && !group().any(|(_, g)| g.2 == 700) {
            sims.push((r.0, r.1, 700, r.3, 1));
            if r.3 == 0 {
                bold_upright.push(sims.len() - 1);
            }
        }
    }
    sims.extend(reals.iter().filter(|r| r.3 == 0).map(|r| (r.0, r.1, r.2, 1, 2)));
    for b in bold_upright {
        let e = sims[b];
        sims.push((e.0, e.1, 700, 1, 3));
    }
    let mut all = reals;
    all.extend(sims);
    all.sort_by_key(|e| (e.4 != 0, e.2, e.3));
    (!all.is_empty()).then_some(all)
}
// The family's faces `(file, face, sims)` in native's match order for a weight and style: slant first (a
// slanted request takes italic and oblique alike; normal: normal, oblique, italic), then the nearest weight, a
// tie going to the heavier, then the exact style (italic: italic, oblique; oblique: oblique, italic), then the
// face order. The first is the match.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn font_file_candidates(family: &str, weight: u32, style: u32, cold: Option<&[String]>) -> Option<Vec<(&'static str, u32, u32)>> {
    let style_rank = |have: u32| match style {
        2 => [2, 1, 0][have.min(2) as usize],
        1 => [2, 0, 1][have.min(2) as usize],
        _ => have.min(2),
    };
    let mut all = family_entries(family, cold)?;
    let slant_rank = |have: u32| if style == 0 { have.min(2) } else if have == 0 { 2 } else { 0 };
    all.sort_by_key(|e| (slant_rank(e.3), (e.2 as i32 - weight as i32).unsigned_abs(), u32::MAX - e.2, style_rank(e.3)));
    Some(all.into_iter().map(|e| (e.0, e.1, e.4)).collect())
}
// The face (file, face index, DWRITE_FONT_SIMULATIONS) the matcher picks from `family_faces`.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn font_file_sims(family: &str, weight: u32, style: u32, cold: Option<&[String]>) -> Option<(&'static str, u32, u32)> {
    font_file_candidates(family, weight, style, cold)?.first().copied()
}
// A corpus captured with `tools/fuzz/capture.py --cold` resolved every family through a custom font
// collection built from only the faces its `cold-faces.txt` lists (one file name per line, in collection
// order), so native's matcher never saw the others: Arial bold italic there is arialbd with a simulated
// oblique, not arialbi. Replaying such a corpus restricts our matcher to the same faces. wasm has no
// filesystem and replays no corpus.
fn cold_faces() -> Option<&'static [String]> {
    #[cfg(target_arch = "wasm32")]
    return None;
    #[cfg(not(target_arch = "wasm32"))]
    {
        static COLD: std::sync::OnceLock<Option<Vec<String>>> = std::sync::OnceLock::new();
        COLD.get_or_init(|| {
            let list = std::fs::read_to_string(std::path::Path::new(&std::env::var_os("FUZZ_DIFF_DIR")?).join("cold-faces.txt")).ok()?;
            Some(list.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect())
        })
        .as_deref()
    }
}
// Resolve a family + weight/style/stretch to concrete face bytes + face index via our drop-in
// dwrite's verified system-font matcher (GetFirstMatchingFont), so every family and weight picks the
// same face DirectWrite does — not a hand-maintained family→file table. `stretch` defaults to 5
// (normal) for the ops that don't carry it.
#[cfg(not(target_arch = "wasm32"))]
fn load_font(family: &str, weight: u32, style: u32, stretch: u32) -> Option<(Vec<u8>, u32)> {
    load_font_sims(family, weight, style, stretch).map(|(b, f, _)| (b, f))
}
// `load_font` plus the matched face's DWRITE_FONT_SIMULATIONS: an oblique request can match a
// family's simulated-oblique upright face, which DrawText/DrawTextLayout then render slanted.
#[cfg(not(target_arch = "wasm32"))]
fn load_font_sims(family: &str, weight: u32, style: u32, stretch: u32) -> Option<(Vec<u8>, u32, u32)> {
    let dir = fonts_dir()?;
    let (path, face, sims) = match cold_faces() {
        Some(names) => {
            let files: Vec<_> = names.iter().map(|n| dir.join(n)).collect();
            // A listed face we lack can't be replayed: skip the scene rather than match without it.
            if !files.iter().all(|f| f.is_file()) {
                return None;
            }
            // The capture builds its collection from a font set, which groups faces by typographic family.
            dwrite::resolve_face_sims_typographic(&files, family, weight, style, stretch)?
        }
        None => dwrite::resolve_face_sims(&dir, family, weight, style, stretch)?,
    };
    if std::env::var_os("D2D_FACECHECK").is_some() {
        let nat = path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
        match font_file_sims(family, weight, style, cold_faces()) {
            Some((wf, wi, _)) if wf == nat && wi == face => {}
            Some((wf, wi, _)) => eprintln!("FACECHECK MISMATCH {family:?} w={weight} s={style} str={stretch}: native={nat}#{face} wasm={wf}#{wi}"),
            None => eprintln!("FACECHECK WASM-SKIP {family:?} w={weight} s={style} str={stretch}: native={nat}#{face} wasm=None"),
        }
    }
    Some((std::fs::read(path).ok()?, face, sims as u32))
}
// The base face of a text or layout format. Under a cold capture's collection, a family the collection
// lacks lays out through native's system fallback, which gives Segoe UI to Latin and the other scripts Segoe
// UI covers (r25: "NoSuchFont", and "Arial Black", which the font-set collection groups under Arial); the
// per-character fallback then covers the rest. That Segoe UI comes from the system collection, every face
// of it (r25 30027: a weight-900 italic range takes Segoe UI Black Italic, which no cold set holds).
#[cfg(not(target_arch = "wasm32"))]
fn load_text_font_sims(family: &str, weight: u32, style: u32, stretch: u32) -> Option<(Vec<u8>, u32, u32)> {
    load_font_sims(family, weight, style, stretch).or_else(|| {
        let dir = fonts_dir()?;
        let files: Vec<_> = cold_faces()?.iter().map(|n| dir.join(n)).collect();
        if !files.iter().all(|f| f.is_file()) || dwrite::resolve_face_sims_typographic(&files, family, weight, style, stretch).is_some() {
            return None;
        }
        let (path, face, sims) = dwrite::resolve_face_sims(&dir, "Segoe UI", weight, style, stretch)?;
        Some((std::fs::read(path).ok()?, face, sims as u32))
    })
}
#[cfg(target_arch = "wasm32")]
fn load_text_font_sims(family: &str, weight: u32, style: u32, stretch: u32) -> Option<(Vec<u8>, u32, u32)> {
    load_font_sims(family, weight, style, stretch)
}
// wasm has no filesystem: the fonts the corpus uses are baked in at build time by
// scripts/gen-wasm-embed.sh (which resolves the gitignored ref/windows/Fonts). Absent ref/ ⇒ the
// generated table is empty ⇒ text scenes skip, exactly as when fonts_dir() is None on native. The
// embedded table is keyed by the family→file map (face 0), so wasm keeps the static map.
#[cfg(target_arch = "wasm32")]
fn load_font(family: &str, weight: u32, style: u32, stretch: u32) -> Option<(Vec<u8>, u32)> {
    load_font_sims(family, weight, style, stretch).map(|(b, f, _)| (b, f))
}
#[cfg(target_arch = "wasm32")]
fn load_font_sims(family: &str, weight: u32, style: u32, _stretch: u32) -> Option<(Vec<u8>, u32, u32)> {
    const FONTS: &[(&str, &[u8])] = include!("embed/fonts.rs");
    let (want, face, sims) = font_file_sims(family, weight, style, cold_faces())?;
    FONTS.iter().find(|(n, _)| *n == want).map(|&(_, b)| (b.to_vec(), face, sims))
}

// Native's font fallback resolves a family in the system collection (not a cold capture's own set) to its
// face matching the format's weight, style and stretch best among those that map the character, which
// can be simulated.
#[cfg(not(target_arch = "wasm32"))]
fn load_fallback(family: &str, weight: u32, style: u32, stretch: u32, ch: char) -> Option<(Vec<u8>, u32, u32)> {
    let (path, face, sims) = dwrite::resolve_face_sims_covering(&fonts_dir()?, family, weight, style, stretch, ch)?;
    Some((std::fs::read(path).ok()?, face, sims as u32))
}
// A variable face's named instance for a format: DirectWrite's collections list a variable font as its fvar
// named instances, and the matcher ranks stretch first (a narrow request takes the nearest instance no wider,
// a wide one the nearest no narrower), then the nearest weight, the heavier on a tie (r25 30370: Bahnschrift at
// stretch 1 is its Condensed instance). Returns the shaper's user axes (tags packed as DWRITE packs them) and
// the raster's normalized coords; both empty for a static face. Pure font data, so native and wasm agree.
fn instance_axes(bytes: &[u8], face: u32, weight: u32, stretch: u32) -> (Vec<(u32, f32)>, Vec<skrifa::raw::types::F2Dot14>) {
    let Ok(font) = FontRef::from_index(bytes, face) else { return (Vec::new(), Vec::new()) };
    let tags: Vec<skrifa::Tag> = font.axes().iter().map(|a| a.tag()).collect();
    let at = |c: &[f32], t: &[u8; 4], d: f32| tags.iter().position(|x| *x == skrifa::Tag::new(t)).and_then(|i| c.get(i).copied()).unwrap_or(d);
    const PCTS: [f32; 9] = [50.0, 62.5, 75.0, 87.5, 100.0, 112.5, 125.0, 150.0, 200.0];
    let stretch_of = |w: f32| (1..=9u32).min_by_key(|&i| ((PCTS[i as usize - 1] - w).abs() * 1000.0) as i64).unwrap_or(5);
    let stretch_rank = |have: u32| match (stretch <= 5, have <= stretch) {
        (true, true) | (false, false) => (0, have.abs_diff(stretch)),
        _ => (1, have.abs_diff(stretch)),
    };
    let best = font.named_instances().iter().map(|i| i.user_coords().collect::<Vec<f32>>()).min_by_key(|c| {
        let w = at(c, b"wght", 400.0).round() as u32;
        (stretch_rank(stretch_of(at(c, b"wdth", 100.0))), w.abs_diff(weight), u32::MAX - w)
    });
    let axes: Vec<(u32, f32)> = best.map(|c| tags.iter().zip(c).map(|(t, v)| (u32::from_le_bytes(t.to_be_bytes()), v)).collect()).unwrap_or_default();
    let coords = d2d::text::variations::normalized_from_dwrite(bytes, face, &axes);
    if coords.is_empty() { (Vec::new(), Vec::new()) } else { (axes, coords) }
}

// The format's own family for a character its matched face lacks (`fallback::OWN_FAMILY`): the family's
// best-ranked face that maps it, in the layout's own collection (a cold capture's set, else the system's).
fn load_own_family(family: &str, weight: u32, style: u32, stretch: u32, ch: char) -> Option<(Vec<u8>, u32, u32)> {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(names) = cold_faces() {
        let dir = fonts_dir()?;
        let files: Vec<_> = names.iter().map(|n| dir.join(n)).collect();
        let (path, face, sims) = dwrite::resolve_face_sims_covering_typographic(&files, family, weight, style, stretch, ch)?;
        return Some((std::fs::read(path).ok()?, face, sims as u32));
    }
    load_fallback(family, weight, style, stretch, ch)
}
// The family's faces `(file, face, sims)` in native's font-fallback order: the real faces in the system set's
// order, ranked by `wws_fallback_order` with each face's simulations for the request. `key_of` gives a face
// file's set-order key (`set_order_key`); a face it cannot read is left out.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn fallback_order(family: &str, weight: u32, style: u32, stretch: u32, key_of: impl Fn(&str) -> Option<String>) -> Option<Vec<(&'static str, u32, u32)>> {
    let mut reals: Vec<(String, &'static str, u32, u32, bool)> =
        family_faces(family)?.iter().filter_map(|&(f, i, w, it)| Some((key_of(f)?, f, i, w, it))).collect();
    reals.sort_by(|a, b| (&a.0, a.2).cmp(&(&b.0, b.2)));
    let faces: Vec<_> = reals.iter().map(|r| (r.3 as u16, 5, if r.4 { 2 } else { 0 })).collect();
    let order = d2d::text::fallback::wws_fallback_order(&faces, weight as u16, stretch as u16, style as u8);
    Some(order.into_iter().map(|(k, sims)| (reals[k].1, reals[k].2, sims as u32)).collect())
}
// wasm ranks the family's faces as the native fallback does ([`fallback_order`]) and takes the first that maps
// the character, so it names the face and simulations native picks.
#[cfg(target_arch = "wasm32")]
fn load_fallback(family: &str, weight: u32, style: u32, stretch: u32, ch: char) -> Option<(Vec<u8>, u32, u32)> {
    const FONTS: &[(&str, &[u8])] = include!("embed/fonts.rs");
    let bytes = |f: &str| FONTS.iter().find(|(n, _)| *n == f).map(|(_, b)| *b);
    fallback_order(family, weight, style, stretch, |f| bytes(f).map(d2d::text::fallback::set_order_key))?.into_iter().find_map(|(f, face, sims)| {
        let b = bytes(f)?;
        FontRef::from_index(b, face).ok()?.charmap().map(ch)?;
        Some((b.to_vec(), face, sims))
    })
}

/// UTF-16 code-unit offset of a UTF-8 byte offset (fallback spans are byte ranges; a glyph run's
/// `text_pos` — the key `draw_layout` matches faces on — is UTF-16).
fn u16_pos(text: &str, byte: usize) -> u32 {
    text[..byte].encode_utf16().count() as u32
}

/// Build a positioned glyph run from a string the way DirectWrite's layout does for a single run:
/// cmap glyph ids and natural (design-metric) advances scaled by `em/upem` (native's order).
// `coords`: a variable face's instance, whose design advances (whole units, as GetDesignGlyphMetrics reports
// them) the run takes.
fn resolve_glyphs(bytes: &[u8], text: &str, em: f32, coords: &[skrifa::raw::types::F2Dot14]) -> Option<Vec<TGlyph>> {
    let font = FontRef::from_index(bytes, 0).ok()?; // face 0 (also handles a .ttc collection)
    let upem = font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f32;
    let cmap = font.charmap();
    let gm = font.glyph_metrics(Size::unscaled(), LocationRef::new(coords));
    Some(
        text.chars()
            .map(|ch| {
                let id = cmap.map(ch).map(|g| g.to_u32() as u16).unwrap_or(0);
                let adv = gm.advance_width(GlyphId::new(id as u32)).unwrap_or(0.0).round() * em / upem;
                TGlyph { id, advance: adv, offset: [0.0, 0.0] }
            })
            .collect(),
    )
}



#[derive(Deserialize)]
struct OracleOp {
    hr: String,
    tag1: u64,
    tag2: u64,
}
#[derive(Deserialize)]
struct OracleEnd {
    hr: String,
    tag1: u64,
    tag2: u64,
}
#[derive(Deserialize)]
struct OracleQuery {
    op: usize,
    kind: String,
    result: serde_json::Value,
}
#[derive(Deserialize)]
struct Oracle {
    ops: Vec<OracleOp>,
    end_draw: OracleEnd,
    #[serde(default)]
    queries: Vec<OracleQuery>,
}

struct OpRec {
    hr: String,
    tag1: u64,
    tag2: u64,
}
/// A recorded geometry query result, matched to the oracle by op index.
struct QueryRec {
    op: usize,
    kind: String,
    got: serde_json::Value,
}
struct Run {
    pixels: Vec<u8>,
    ops: Vec<OpRec>,
    end: OracleEnd,
    queries: Vec<QueryRec>,
}

fn fmt_hr(code: i32) -> String {
    format!("0x{:08X}", code as u32)
}

// --- scene → d2d::geometry lowering (mirrors the 1C geometry test's builder) -------------------

fn seg(s: &Segment) -> geometry::Segment {
    match *s {
        Segment::Line { to } => geometry::Segment::Line(to),
        Segment::Cubic { p1, p2, to } => geometry::Segment::Cubic(p1, p2, to),
        Segment::Quadratic { control, to } => geometry::Segment::Quadratic(control, to),
        Segment::Arc { to, radius, rotation, sweep, size } => {
            geometry::Segment::Arc { to, radius, rotation, clockwise: sweep == 1, large: size == 1 }
        }
    }
}
fn figure(f: &Figure) -> geometry::Figure {
    geometry::Figure { start: f.start, filled: f.filled, closed: f.closed, segments: f.segments.iter().map(seg).collect(), seg_flags: f.seg_flags.clone() }
}
fn build_geometry(shape: &Shape) -> Geometry {
    match shape {
        Shape::Rect { rect } => Geometry::Rect(*rect),
        Shape::Rounded { rect, radius } => Geometry::Rounded { rect: *rect, radius: *radius },
        Shape::Ellipse { center, radius } => Geometry::Ellipse { center: *center, radius: *radius },
        Shape::Path { fill, figures } => Geometry::Path {
            fill: if *fill == 1 { geometry::FillMode::Winding } else { geometry::FillMode::Alternate },
            figures: figures.iter().map(figure).collect(),
        },
    }
}

// Serialize an operation's output path to the same shape the harness's RecordingSink captures from
// native: fill mode, then per figure the start, begin/end flags, and each segment's type + points.
fn result_stream(g: &Geometry) -> serde_json::Value {
    let Geometry::Path { fill, figures } = g else { return serde_json::Value::Null };
    let fm = match fill {
        geometry::FillMode::Alternate => 0,
        geometry::FillMode::Winding => 1,
    };
    let figs: Vec<_> = figures
        .iter()
        .map(|f| {
            let segs: Vec<_> = f
                .segments
                .iter()
                .map(|s| match s {
                    geometry::Segment::Line(p) => serde_json::json!({"t": "line", "p": p}),
                    geometry::Segment::Cubic(a, b, c) => serde_json::json!({"t": "bezier", "p": [a, b, c]}),
                    geometry::Segment::Quadratic(a, b) => serde_json::json!({"t": "quad", "p": [a, b]}),
                    geometry::Segment::Arc { .. } => serde_json::json!({"t": "arc"}),
                })
                .collect();
            serde_json::json!({
                "start": f.start, "begin": i32::from(!f.filled), "end": i32::from(f.closed), "segments": segs
            })
        })
        .collect();
    serde_json::json!({"fill_mode": fm, "figures": figs})
}

fn cap_of(v: u32) -> Cap {
    match v {
        1 => Cap::Square,
        2 => Cap::Round,
        3 => Cap::Triangle,
        _ => Cap::Flat,
    }
}
fn join_of(v: u32) -> Join {
    match v {
        1 => Join::Bevel,
        2 => Join::Round,
        3 => Join::MiterOrBevel,
        _ => Join::Miter,
    }
}
/// A named `D2D1_DASH_STYLE`'s predefined dash array (stroke-width units), matching d2d-ffi's
/// `dashes_of`. A named style ignores any custom `dashes`; 0 (SOLID/CUSTOM) keeps the custom array.
fn named_dashes(style: u32, custom: &[f32]) -> Vec<f32> {
    match style {
        1 => vec![2.0, 2.0],                     // DASH
        2 => vec![0.0, 2.0],                     // DOT
        3 => vec![2.0, 2.0, 0.0, 2.0],           // DASH_DOT
        4 => vec![2.0, 2.0, 0.0, 2.0, 0.0, 2.0], // DASH_DOT_DOT
        _ => custom.to_vec(),
    }
}
fn transform_type_of(v: u32) -> TransformType {
    match v {
        1 => TransformType::Fixed,
        2 => TransformType::Hairline,
        _ => TransformType::Normal,
    }
}

fn stops_of(stops: &[d2d_reference::Stop]) -> Vec<(f32, [f32; 4])> {
    stops.iter().map(|s| (s.position, s.color)).collect()
}

// --- per-scene resources -----------------------------------------------------------------------

#[derive(Default)]
struct Res<'s> {
    solids: HashMap<&'s str, SolidColorBrush>,
    pbrushes: HashMap<&'s str, PixelBrush>,
    bitmaps: HashMap<&'s str, Bitmap>,
    effects: HashMap<&'s str, FxEffect>,
    geoms: HashMap<&'s str, Geometry>,
    styles: HashMap<&'s str, StrokeStyle>,
    states: HashMap<&'s str, DrawingStateBlock>,
    floods: HashMap<&'s str, [i32; 4]>,
    sprite_batches: HashMap<&'s str, SpriteBatch>,
    inks: HashMap<&'s str, extras::ink::Ink>,
    ink_styles: HashMap<&'s str, extras::ink::InkStyle>,
    svgs: HashMap<&'s str, d2d::svg::Document>,
    // SVG DOM element handles: scene name → (owning doc key, stable node id).
    svg_elems: HashMap<&'s str, (&'s str, u32)>,
    image_sources: HashMap<&'s str, ImageSource>,
    lists: HashMap<&'s str, CommandList>,
    meshes: HashMap<&'s str, extras::mesh::GradientMesh>,
    color_contexts: HashMap<&'s str, ColorContext>,
    // A ColorManagement effect: its transform plus the input-bitmap id (CM is a 1:1 pixel map
    // applied at draw_image time, not an effect-graph node).
    color_mgmt: HashMap<&'s str, (ColorTransform, &'s str)>,
    // ColorManagement effects whose realization native fails (a SIMPLE-profile context): EndDraw
    // returns E_INVALIDARG latched to the draw_image op (cms2-04/05). `cm_fail_tag` is set to that
    // op's tag once the draw runs.
    cm_fail: HashSet<&'s str>,
    cm_fail_tag: Option<u64>,
    // 1C-8 geometry realizations: the baked geometry (+ stroke width/style for a stroked one).
    realizations: HashMap<&'s str, (Geometry, Option<(f32, StrokeStyle)>, f32)>,
    queries: Vec<QueryRec>,
    // Per-op HRESULT overrides: an op native returns a (non-latched) error for, continuing the scene
    // (e.g. CreateEffect of an unregistered CLSID → ERROR_NOT_FOUND, `v2-hresult`). op index → hr string.
    op_hr: HashMap<usize, String>,
    // The context's text rendering params, as the native harness leaves them: `glyph_run` (and the
    // `glyph_bounds` query) SetTextRenderingParams with the op's mode + grid-fit and they PERSIST for
    // every later text draw, recorded ones too; a `text` op with explicit params restores the pinned
    // default after itself.
    text_params: RenderingParams,
}

/// Scene `Value` → effect-graph `Prop` (2B). Parse the CLSID hex into a `GUID`.
fn eff_guid(clsid: &str) -> GUID {
    GUID::from_u128(u128::from_str_radix(&clsid.replace('-', ""), 16).unwrap_or(0))
}

/// A custom vertex-shader effect (3C): loads the VS+PS, builds the vertex buffer (POSITION+TEXCOORD),
/// and binds a vertex-processing draw transform over `inputs` inputs into the `viewport` output rect.
struct VertexEffectImpl {
    vs: Vec<u8>,
    ps: Vec<u8>,
    verts: Vec<u8>,
    inputs: u32,
    viewport: [i32; 4],
}
impl d2d::effects::EffectImpl for VertexEffectImpl {
    fn initialize(&self, _p: &d2d::effects::Props, ctx: &d2d::effects::EffectContext, graph: &mut d2d::effects::TransformGraph) -> Result<(), windows_core::HRESULT> {
        let (vs_id, ps_id) = (GUID::from_u128(0x5e_0001), GUID::from_u128(0x5e_0002));
        ctx.load_vertex_shader(vs_id, &self.vs)?;
        ctx.load_pixel_shader(ps_id, &self.ps)?;
        let elements = vec![
            d2d::effects::custom::VbElement { semantic: "POSITION".into(), index: 0, offset: 0, components: 2 },
            d2d::effects::custom::VbElement { semantic: "TEXCOORD".into(), index: 0, offset: 8, components: 2 },
        ];
        let vb = ctx.create_vertex_buffer(None, &self.verts, 16, elements)?;
        let t = ctx.vertex_shader_transform(vs_id, ps_id, Vec::new(), Vec::new(), vb, self.inputs, Vec::new(), self.viewport)?;
        graph.set_single_transform_node(t)
    }
}
/// A custom effect whose graph is the non-shader built-in transforms border → offset →
/// bounds-adjustment, wired with the multi-node TransformGraph API (the effect-author surface).
struct GraphEffectImpl {
    offset: [i32; 2],
    border: [u32; 2],
    bounds: [i32; 4],
}
impl d2d::effects::EffectImpl for GraphEffectImpl {
    fn initialize(&self, _p: &d2d::effects::Props, ctx: &d2d::effects::EffectContext, graph: &mut d2d::effects::TransformGraph) -> Result<(), windows_core::HRESULT> {
        let border = graph.add_node(ctx.create_border_transform(self.border[0], self.border[1]));
        let offset = graph.add_node(ctx.create_offset_transform(self.offset));
        let bounds = graph.add_node(ctx.create_bounds_adjustment_transform(self.bounds));
        graph.connect_to_effect_input(0, border, 0)?;
        graph.connect_node(border, offset, 0)?;
        graph.connect_node(offset, bounds, 0)?;
        graph.set_output_node(bounds)
    }
}
/// A 2-input custom effect that blends its inputs through a D2D1_BLEND_DESCRIPTION.
struct BlendEffectImpl {
    desc: [u32; 6],
    factor: [f32; 4],
}
impl d2d::effects::EffectImpl for BlendEffectImpl {
    fn initialize(&self, _p: &d2d::effects::Props, ctx: &d2d::effects::EffectContext, graph: &mut d2d::effects::TransformGraph) -> Result<(), windows_core::HRESULT> {
        let d = self.desc;
        let t = ctx.create_blend_desc_transform(2, d[0], d[1], d[2], d[3], d[4], d[5], self.factor)?;
        let blend = graph.add_node(t);
        graph.connect_to_effect_input(0, blend, 0)?;
        graph.connect_to_effect_input(1, blend, 1)?;
        graph.set_output_node(blend)
    }
}
/// A 0-input compute custom effect: loads a cs_5_0 shader and binds it as the single compute
/// transform (the generator writes a fixed-size typed output UAV).
struct ComputeEffectImpl {
    shader_id: GUID,
    dxbc: Vec<u8>,
    cbuffer: Vec<u8>,
    groups: [u32; 3],
}
impl d2d::effects::EffectImpl for ComputeEffectImpl {
    fn initialize(&self, _p: &d2d::effects::Props, ctx: &d2d::effects::EffectContext, graph: &mut d2d::effects::TransformGraph) -> Result<(), windows_core::HRESULT> {
        ctx.load_compute_shader(self.shader_id, &self.dxbc)?;
        let t = ctx.compute_shader_transform(self.shader_id, self.cbuffer.clone(), 0, Vec::new(), self.groups, [0, 0])?;
        graph.set_single_transform_node(t)
    }
}
fn eff_prop(v: &Value) -> FxProp {
    match *v {
        Value::Float(x) => FxProp::F32(x),
        Value::Uint32(x) | Value::Enum(x) => FxProp::U32(x),
        Value::Int32(x) => FxProp::I32(x),
        Value::Bool(x) => FxProp::Bool(x),
        Value::Vector2(x) => FxProp::V2(x),
        Value::Vector3(x) => FxProp::V3(x),
        Value::Vector4(x) => FxProp::V4(x),
        Value::Matrix3x2(x) => FxProp::M32(x),
        Value::Matrix4x4(x) => FxProp::M44(x),
        Value::Matrix5x4(x) => FxProp::M54(x),
        Value::Floats(ref x) => FxProp::Floats(x.as_slice().into()), // e.g. the ConvolveMatrix kernel
        Value::Iunknown(_) => unreachable!("iunknown values appear only on ColorManagement, handled separately"),
    }
}

// --- SVG DOM editing helpers (shared by the `Op::Svg*` arms) ----------------------------------

fn svg_paint_type(n: u32) -> d2d::svg::PaintType {
    use d2d::svg::PaintType::*;
    [None, Color, CurrentColor, Uri, UriNone, UriColor, UriCurrentColor].get(n as usize).copied().unwrap_or(None)
}
fn svg_path_cmd(n: u32) -> d2d::svg::attr::PathCmd {
    use d2d::svg::attr::PathCmd::*;
    [
        ClosePath, MoveAbs, MoveRel, LineAbs, LineRel, CubicAbs, CubicRel, QuadAbs, QuadRel, ArcAbs,
        ArcRel, HorizAbs, HorizRel, VertAbs, VertRel, CubicSmoothAbs, CubicSmoothRel, QuadSmoothAbs,
        QuadSmoothRel,
    ]
    .get(n as usize)
    .copied()
    .unwrap_or(ClosePath)
}
/// A scene `SvgObject` → the core attribute [`d2d::svg::Value`] it sets (the object-attribute path).
fn svg_object_value(o: &SvgObject) -> d2d::svg::Value {
    use d2d::svg::{Units, Value};
    match o {
        SvgObject::Paint { paint_type, color, id } => {
            Value::Paint { ty: svg_paint_type(*paint_type), color: color.unwrap_or([0.0; 4]), id: id.clone() }
        }
        SvgObject::Points { points } => Value::Points(points.clone()),
        SvgObject::Path { commands, data } => {
            Value::Path { cmds: commands.iter().map(|c| svg_path_cmd(*c)).collect(), data: data.clone() }
        }
        SvgObject::Dashes { dashes } => Value::Dashes(dashes.iter().map(|v| (*v, Units::Number)).collect()),
    }
}
/// `GetAttributeValue2` POD bytes as LE u32 words (every field is 4-byte), matching the DLL's encode.
/// Build a `d2d::svg::Value` from a `D2D1_SVG_ATTRIBUTE_POD_TYPE` + its float words — the core side of
/// `SetAttributeValue2`. (Only the all-float POD kinds a scene uses; extend as needed.)
fn svg_pod_value(pod: u32, f: &[f32]) -> d2d::svg::Value {
    use d2d::svg::Value;
    match pod {
        1 => Value::Color([f[0], f[1], f[2], f[3]]),                       // COLOR
        8 => Value::Matrix([f[0], f[1], f[2], f[3], f[4], f[5]]),          // MATRIX
        12 => Value::ViewBox([f[0], f[1], f[2], f[3]]),                    // VIEWBOX
        _ => Value::Float(f[0]),                                          // FLOAT / enum stored as float
    }
}

fn svg_pod_words(d: &d2d::svg::Document, id: u32, name: &str) -> Result<Vec<u32>, String> {
    use d2d::svg::{Units, Value};
    let v = d.attr_value(id, name).ok_or("attr unset")?;
    let f = |x: f32| x.to_bits();
    Ok(match v {
        Value::Float(x) => vec![f(x)],
        Value::Color(c) => c.iter().map(|x| f(*x)).collect(),
        Value::Matrix(m) => m.iter().map(|x| f(*x)).collect(),
        Value::ViewBox(b) => b.iter().map(|x| f(*x)).collect(),
        Value::Length(x, u) => vec![f(x), (u == Units::Percentage) as u32],
        Value::Aspect { defer, align, slice } => vec![defer as u32, align as u32, slice as u32],
        Value::Enum(e) => vec![e as u32],
        _ => return Err("attribute is not a POD type".into()),
    })
}
fn svg_hr(r: Result<(), i32>) -> String {
    r.map_or_else(fmt_hr, |()| "S_OK".into())
}
fn svg_query(
    r: &Res,
    kind: &str,
    doc: Option<&str>,
    elem: Option<&str>,
    name: Option<&str>,
    index: Option<u32>,
    _pod: Option<u32>,
) -> Result<serde_json::Value, String> {
    let name = name.unwrap_or("");
    let docu = r.svgs.get(doc.unwrap_or("")).ok_or("svg doc".to_string());
    let el = elem.and_then(|e| r.svg_elems.get(e).copied());
    let ed = || {
        let (dk, id) = el.ok_or("svg elem".to_string())?;
        Ok::<_, String>((r.svgs.get(dk).ok_or("svg elem doc".to_string())?, id))
    };
    Ok(match kind {
        "serialize" => serde_json::json!(docu?.serialize()),
        "attr" => {
            let (d, id) = ed()?;
            serde_json::json!(d.attr_string(id, name).ok_or("attr unset")?)
        }
        "attr_pod" => {
            let (d, id) = ed()?;
            serde_json::json!(svg_pod_words(d, id, name)?)
        }
        "count" => {
            let (d, id) = ed()?;
            serde_json::json!(d.specified_count(id))
        }
        "name" => {
            let (d, id) = ed()?;
            serde_json::json!(d.specified_name(id, index.unwrap_or(0)).ok_or("name index")?)
        }
        "tag" => {
            let (d, id) = ed()?;
            serde_json::json!(d.tag_of(id).ok_or("tag")?)
        }
        "specified" => {
            let (d, id) = ed()?;
            serde_json::json!(d.is_specified(id, name))
        }
        "has_children" => {
            let (d, id) = ed()?;
            serde_json::json!(d.has_children(id))
        }
        // DOM navigation — tag of the navigated element ("" past the end), and IsTextContent.
        "parent_tag" => {
            let (d, id) = ed()?;
            serde_json::json!(d.parent_id(id).and_then(|p| d.tag_of(p)).unwrap_or_default())
        }
        "first_child_tag" => {
            let (d, id) = ed()?;
            serde_json::json!(d.first_child_id(id).and_then(|c| d.tag_of(c)).unwrap_or_default())
        }
        "last_child_tag" => {
            let (d, id) = ed()?;
            serde_json::json!(d.last_child_id(id).and_then(|c| d.tag_of(c)).unwrap_or_default())
        }
        "next_tag" => {
            let (d, id) = ed()?;
            serde_json::json!(d.parent_id(id).and_then(|p| d.sibling_id(p, id, 1)).and_then(|c| d.tag_of(c)).unwrap_or_default())
        }
        "prev_tag" => {
            let (d, id) = ed()?;
            serde_json::json!(d.parent_id(id).and_then(|p| d.sibling_id(p, id, -1)).and_then(|c| d.tag_of(c)).unwrap_or_default())
        }
        "is_text" => {
            let (d, id) = ed()?;
            serde_json::json!(d.is_text_content(id))
        }
        // Object-valued attribute readback — the stored d2d::svg::Value's fields, matching the native
        // object getters.
        "paint_type" => {
            let (d, id) = ed()?;
            match d.attr_value(id, name) {
                Some(d2d::svg::Value::Paint { ty, .. }) => serde_json::json!(ty as i32),
                _ => return Err("attr not a paint".into()),
            }
        }
        "paint_id" => {
            let (d, id) = ed()?;
            match d.attr_value(id, name) {
                Some(d2d::svg::Value::Paint { id: s, .. }) => serde_json::json!(s),
                _ => return Err("attr not a paint".into()),
            }
        }
        "path_cmds" => {
            let (d, id) = ed()?;
            match d.attr_value(id, name) {
                Some(d2d::svg::Value::Path { cmds, .. }) => serde_json::json!(cmds.iter().map(|c| *c as i32).collect::<Vec<i32>>()),
                _ => return Err("attr not a path".into()),
            }
        }
        "path_data" => {
            let (d, id) = ed()?;
            match d.attr_value(id, name) {
                Some(d2d::svg::Value::Path { data, .. }) => serde_json::json!(data),
                _ => return Err("attr not a path".into()),
            }
        }
        "points" => {
            let (d, id) = ed()?;
            match d.attr_value(id, name) {
                Some(d2d::svg::Value::Points(pts)) => serde_json::json!(pts.iter().flat_map(|p| [p[0], p[1]]).collect::<Vec<f32>>()),
                _ => return Err("attr not points".into()),
            }
        }
        "dashes" => {
            let (d, id) = ed()?;
            match d.attr_value(id, name) {
                Some(d2d::svg::Value::Dashes(ds)) => serde_json::json!(ds.iter().map(|(v, _)| *v).collect::<Vec<f32>>()),
                _ => return Err("attr not dashes".into()),
            }
        }
        "viewport" => {
            let vp = docu?.viewport;
            serde_json::json!([vp[0], vp[1]])
        }
        "text_value" => {
            let (d, id) = ed()?;
            serde_json::json!(d.text_value(id).unwrap_or_default())
        }
        // FindElementById returns S_OK + null for a missing id; the binding surfaces that as Err(0).
        "find_hr" => serde_json::json!(svg_hr(docu?.find_id(name).map(|_| ()).ok_or(0))),
        "attr_hr" => {
            let (d, id) = ed()?;
            serde_json::json!(svg_hr(d.attr_string(id, name).map(|_| ()).ok_or(0x8007_0057u32 as i32)))
        }
        other => return Err(format!("unknown svg query kind {other}")),
    })
}

/// The intrinsic DIP bounds of an image id for `GetImage{Local,World}Bounds` (mirrors the FFI's
/// `image_bounds_dip`): a bitmap's pixel rect scaled to DIPs, an effect's finite output bounds, or the
/// empty rect for an infinite image (command list / flood).
fn image_bounds_dip(r: &Res, image: &str) -> Result<[f32; 4], String> {
    if let Some(e) = r.effects.get(image) {
        Ok(e.intrinsic_bounds().unwrap_or([0.0; 4]))
    } else if let Some(b) = r.bitmaps.get(image) {
        Ok([0.0, 0.0, b.width as f32 * 96.0 / b.dpi.0, b.height as f32 * 96.0 / b.dpi.1])
    } else if let Some(s) = r.image_sources.get(image) {
        let b = &s.bitmap;
        Ok([0.0, 0.0, b.width as f32 * 96.0 / b.dpi.0, b.height as f32 * 96.0 / b.dpi.1])
    } else if r.lists.contains_key(image) || r.floods.contains_key(image) {
        Ok([0.0; 4])
    } else {
        Err("image bounds on unknown image".into())
    }
}

/// Run one op through the public API. `Err(reason)` ⇒ the scene uses a feature with no API yet (SKIP).
fn exec_op<'s>(dc: &mut DeviceContext, factory: &Factory, i: usize, op: &'s Op, r: &mut Res<'s>) -> Result<(), String> {
    match op {
        Op::Clear { color } => dc.clear(Some(*color)),
        Op::Transform { matrix } => dc.set_transform(*matrix),
        Op::Antialias { mode } => dc.set_antialias_mode(*mode),
        Op::Blend { mode } => dc.set_primitive_blend(*mode),
        Op::UnitMode { mode } => dc.set_unit_mode(*mode),
        Op::SetTags { tag1, tag2 } => dc.set_tags(*tag1, *tag2),
        Op::Solid { id, color, opacity, .. } => {
            // A solid brush's transform has no effect on a solid fill, so it is accepted and ignored.
            r.solids.insert(id, dc.create_solid_color_brush(*color, *opacity));
        }
        Op::Linear { id, start, end, stops, gamma, extend, opacity, transform } => {
            let b = dc.create_linear_gradient_brush(*start, *end, &stops_of(stops), *gamma, *extend, opacity.unwrap_or(1.0), *transform);
            r.pbrushes.insert(id, b);
        }
        Op::Radial { id, center, offset, radius, stops, gamma, extend, opacity, transform } => {
            let b = dc.create_radial_gradient_brush(*center, *offset, *radius, &stops_of(stops), *gamma, *extend, opacity.unwrap_or(1.0), *transform);
            r.pbrushes.insert(id, b);
        }
        Op::BitmapBrush { id, bitmap, extend, interpolation, matrix, opacity } => {
            let bm = r.bitmaps.get(bitmap.as_str()).ok_or("bitmap brush of a non-1E bitmap")?;
            r.pbrushes.insert(id, dc.create_bitmap_brush(bm, *extend, *interpolation, *opacity, Some(*matrix)));
        }
        Op::ImageBrush { id, image, source, extend, interpolation, opacity, matrix } => {
            let img = if let Some(e) = r.effects.get(image.as_str()) {
                d2d::BrushImage::Effect(e)
            } else if let Some(l) = r.lists.get(image.as_str()) {
                d2d::BrushImage::CommandList(l)
            } else if let Some(b) = r.bitmaps.get(image.as_str()) {
                d2d::BrushImage::Bitmap(b)
            } else {
                // The image resource is undefined — a command list whose Close failed, or an effect
                // whose SetValue native rejected: native's harness records "undefined resource" and
                // leaves the brush undefined, continuing the scene (fuzz 7770 op15, r16 21185/21017).
                r.op_hr.insert(i, format!("err: undefined resource {image:?}"));
                return Ok(());
            };
            let brush = dc.create_image_brush(img, *source, *extend, *interpolation, *opacity, Some(*matrix));
            r.pbrushes.insert(id, brush);
        }
        Op::Geometry { id, shape } => {
            let g = build_geometry(shape);
            // Native's geometry sink rejects a path arc with a negative radius (E_INVALIDARG at
            // Close) and never creates the geometry, so a later fill sees an undefined resource.
            if g.has_invalid_arc_radius() {
                r.op_hr.insert(i, fmt_hr(0x80070057u32 as i32));
            } else {
                r.geoms.insert(id, g);
            }
        }
        Op::GeometryGroup { id, fill, geometries } => {
            // A child whose creation native rejected (e.g. invalid arc radius) is undefined here;
            // native's harness records "undefined resource" for the first missing child and never
            // creates the group, so mirror that status and leave the group undefined, keep diffing the
            // rest of the scene rather than failing the whole scene as a SKIP.
            if let Some(bad) = geometries.iter().find(|g| !r.geoms.contains_key(g.as_str())) {
                r.op_hr.insert(i, format!("err: undefined resource {bad:?}"));
            } else {
                let children = geometries.iter().map(|g| r.geoms.get(g.as_str()).cloned().unwrap()).collect();
                let fill = if *fill == 1 { geometry::FillMode::Winding } else { geometry::FillMode::Alternate };
                r.geoms.insert(id, Geometry::Group { fill, children });
            }
        }
        Op::GeometryTransform { id, geometry: src, matrix } => {
            let child = r.geoms.get(src.as_str()).cloned().ok_or_else(|| format!("undefined resource {src:?}"))?;
            r.geoms.insert(id, Geometry::Transformed { matrix: *matrix, child: Box::new(child) });
        }
        Op::Stroke { id, caps, join, miter, offset, dashes, transform_type, dash_style } => {
            r.styles.insert(
                id,
                StrokeStyle {
                    start_cap: cap_of(caps[0]),
                    end_cap: cap_of(caps[1]),
                    dash_cap: cap_of(caps[2]),
                    join: join_of(*join),
                    miter_limit: *miter,
                    dash_offset: *offset,
                    dashes: named_dashes(*dash_style, dashes),
                    transform_type: transform_type_of(*transform_type),
                },
            );
        }
        Op::Primitive { shape, brush, width } => {
            // Native's harness looks the brush up first, so an undefined one (an image brush on a list whose
            // Close failed) is "undefined resource" before any shape check (r24 29046).
            if !r.solids.contains_key(brush.as_str()) && !r.pbrushes.contains_key(brush.as_str()) {
                return Err(format!("undefined resource {brush:?}"));
            }
            // A `primitive` builds + draws a transient geometry in one call, so native's geometry-sink
            // validation surfaces at the draw: a path arc with a negative radius (incl. -inf) is
            // rejected with E_INVALIDARG and renders nothing (VM batch-3: 48 primitive cases). Non-finite
            // *points* and +inf/NaN radii are accepted, so this mirrors `has_invalid_arc_radius` exactly
            // — the same check the named-geometry create path uses — not a blanket non-finite reject.
            if !matches!(shape, Shape::Rect { .. }) && build_geometry(shape).has_invalid_arc_radius() {
                r.op_hr.insert(i, fmt_hr(0x80070057u32 as i32));
                return Ok(());
            }
            if let Some(w) = width {
                let g = build_geometry(shape);
                let st = StrokeStyle::default();
                if let (Some(b), Shape::Rect { rect }) = (r.solids.get(brush.as_str()), shape) {
                    dc.draw_rectangle(*rect, b, *w, &st);
                } else if let (Some(pb), Shape::Rect { rect }) = (r.pbrushes.get(brush.as_str()), shape) {
                    dc.draw_rectangle_brush(*rect, pb, *w, &st);
                } else if let Some(b) = r.solids.get(brush.as_str()) {
                    dc.draw_geometry(&g, b, *w, &st);
                } else if let Some(pb) = r.pbrushes.get(brush.as_str()) {
                    dc.draw_geometry_brush(&g, pb, *w, &st);
                } else {
                    return Err(format!("undefined resource {brush:?}"));
                }
            } else if let Some(b) = r.solids.get(brush.as_str()) {
                // Axis-aligned fast path keeps pixel-aligned rects byte-exact; others flatten.
                match shape {
                    Shape::Rect { rect } => dc.fill_rectangle(*rect, b),
                    other => dc.fill_geometry(&build_geometry(other), b),
                }
            } else if let Some(pb) = r.pbrushes.get(brush.as_str()) {
                match shape {
                    Shape::Rect { rect } => dc.fill_rectangle_brush(*rect, pb),
                    other => dc.fill_geometry_brush(&build_geometry(other), pb),
                }
            } else {
                return Err(format!("undefined resource {brush:?}"));
            }
        }
        Op::DrawLine { p0, p1, brush, width } => {
            // DrawLine with no stroke style (the harness passes none).
            if let Some(b) = r.solids.get(brush.as_str()) {
                dc.draw_line(*p0, *p1, b, *width, None);
            } else if let Some(pb) = r.pbrushes.get(brush.as_str()) {
                dc.draw_line_brush(*p0, *p1, pb, *width, None);
            } else {
                return Err(format!("undefined resource {brush:?}"));
            }
        }
        Op::Fill { geometry, brush } => {
            // A geometry whose creation native rejected (e.g. invalid arc radius) is undefined here.
            // Native's d2d-ref SKIPS FillGeometry on a missing geometry — it is a no-op that returns
            // S_OK, not a recorded error (harness/src/render.rs Op::Fill: `if let Some(g) = get { Fill }`;
            // native capture r12 17052 op9/op12, recap 15175 op24/op26 = S_OK). Mirror that: draw nothing
            // and leave the op S_OK, keep diffing the rest of the scene. (DrawGeometryRealization DOES
            // error on a missing geometry — see Op::Realize — but Fill/Draw do not.)
            let Some(geom) = r.geoms.get(geometry.as_str()) else {
                return Ok(());
            };
            if let Some(b) = r.solids.get(brush.as_str()) {
                dc.fill_geometry(geom, b);
            } else if let Some(pb) = r.pbrushes.get(brush.as_str()) {
                match geom {
                    Geometry::Rect(rect) => dc.fill_rectangle_brush(*rect, pb),
                    other => dc.fill_geometry_brush(other, pb),
                }
            } else {
                return Err(format!("undefined resource {brush:?}"));
            }
        }
        Op::Draw { geometry, brush, width, style } => {
            // Undefined geometry (native rejected its creation): native's d2d-ref SKIPS DrawGeometry (a
            // no-op, S_OK), it does NOT record an error (see Op::Fill; native capture 17052 op9, 15175
            // op26 = S_OK). Draw nothing and keep the op S_OK.
            let Some(geom) = r.geoms.get(geometry.as_str()).cloned() else {
                return Ok(());
            };
            let st = match style {
                Some(id) => r.styles.get(id.as_str()).ok_or("undefined stroke style")?.clone(),
                None => StrokeStyle::default(),
            };
            if let Some(b) = r.solids.get(brush.as_str()) {
                dc.draw_geometry(&geom, b, *width, &st);
            } else if let Some(pb) = r.pbrushes.get(brush.as_str()) {
                dc.draw_geometry_brush(&geom, pb, *width, &st);
            } else {
                return Err(format!("undefined resource {brush:?}"));
            }
        }
        Op::Realize { id, geometry, stroke_width, style, tolerance } => {
            let geom = r.geoms.get(geometry.as_str()).ok_or_else(|| format!("undefined resource {geometry:?}"))?.clone();
            // Native rejects CreateFilled/StrokedGeometryRealization with D2DERR_BAD_NUMBER (0x88990011)
            // when the geometry, baked at the current context transform, has a non-finite scanner-workspace
            // extent (a near-f32::MAX / inf coordinate overflows the f32 strip-clip) — see
            // `Geometry::realize_bad_number`. Native leaves the op S_OK for the scene but returns the error
            // for the realize call, so record the per-op HR and skip creating the realization.
            let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
            let st = match style {
                Some(sid) => r.styles.get(sid.as_str()).ok_or("undefined stroke style")?.clone(),
                None => StrokeStyle::default(),
            };
            // A realization whose bounds (or clipped bounds) come out degenerate or NaN bakes empty, S_OK,
            // before anything can fail (`geometry::realization::realization_clips_empty`).
            if geometry::realization::realization_clips_empty(&geom, stroke_width.map(|w| (w, &st))) {
                r.realizations.insert(id.as_str(), (Geometry::Path { fill: geometry::FillMode::Alternate, figures: Vec::new() }, None, tol));
            // A stroked realization's create-time widen can fail too (`Geometry::stroked_realize_bad_number`).
            } else if match stroke_width {
                Some(w) => geom.stroked_realize_bad_number(*w, &st, tol),
                None => geom.realize_bad_number(tol),
            } {
                r.op_hr.insert(i, fmt_hr(0x8899_0011u32 as i32));
            } else {
                let stroke = match stroke_width {
                    Some(w) => {
                        // A pen that voids at create time bakes an empty realization: keep it as a zero
                        // width, which strokes nothing (r13 18223's tolerance 3.4e38).
                        let voids = geometry::stroked_realization_voids(&geom, *w, &st, tol);
                        Some((if voids { 0.0 } else { *w }, st))
                    }
                    None => None,
                };
                r.realizations.insert(id.as_str(), (geom, stroke, tol));
            }
        }
        Op::DrawRealization { realization, brush } => {
            // A realization whose create failed (BAD_NUMBER, fuzz 16683) is undefined: native's harness
            // records the draw as `err: undefined resource` and carries on, as for an undefined brush.
            let (geom, stroke, tol) = r.realizations.get(realization.as_str()).ok_or_else(|| format!("undefined resource {realization:?}"))?;
            if let Some(b) = r.solids.get(brush.as_str()) {
                dc.draw_geometry_realization(geom, stroke.as_ref(), *tol, b);
            } else if let Some(pb) = r.pbrushes.get(brush.as_str()) {
                dc.draw_geometry_realization_brush(geom, stroke.as_ref(), *tol, pb);
            } else {
                return Err(format!("undefined resource {brush:?}"));
            }
        }
        Op::Query { geometry, kind, point, tolerance, world } => {
            let g = r.geoms.get(geometry.as_str()).ok_or_else(|| format!("undefined resource {geometry:?}"))?;
            let (tol, wm) = (tolerance.unwrap_or(DEFAULT_TOLERANCE), world.unwrap_or(IDENTITY));
            // A scanned query the scanner's workspace rejects fails BAD_NUMBER: no result is recorded.
            let bad = match kind.as_str() {
                "area" => g.area_bad_number(&wm),
                "contains_point" => point.is_some_and(|p| g.fill_contains_bad_number(p, &wm, tol)),
                _ => false,
            };
            if bad {
                r.op_hr.insert(i, fmt_hr(0x8899_0011u32 as i32));
                return Ok(());
            }
            let got = match kind.as_str() {
                "area" => serde_json::json!(g.compute_area(&wm, tol)),
                "length" => serde_json::json!(g.compute_length(&wm, tol)),
                "bounds" => serde_json::json!(g.get_bounds(&wm)),
                "contains_point" => {
                    let p = point.ok_or("contains_point without a point")?;
                    serde_json::json!(g.fill_contains(p, &wm, tol))
                }
                other => return Err(format!("unknown query kind {other}")),
            };
            r.queries.push(QueryRec { op: i, kind: kind.clone(), got });
        }
        Op::GeometryOp { id, source, operation, mode, width, style, other, transform, tolerance, world } => {
            let src = r.geoms.get(source.as_str()).ok_or_else(|| format!("undefined resource {source:?}"))?;
            let (tol, wm) = (tolerance.unwrap_or(DEFAULT_TOLERANCE), world.unwrap_or(IDENTITY));
            let result = match operation.as_str() {
                "simplify" => geometry::combine::simplify(src, mode.unwrap_or(0), &wm, tol),
                "combine" => {
                    let n = other.as_ref().ok_or("combine requires an `other` geometry")?;
                    let o = r.geoms.get(n.as_str()).ok_or_else(|| format!("undefined resource {n:?}"))?;
                    let xf = transform.unwrap_or(IDENTITY);
                    // A rejected scan fails the op: no result, and the output geometry stays undefined.
                    if src.combine_bad_number(o, &IDENTITY, &xf) {
                        r.op_hr.insert(i, fmt_hr(0x8899_0011u32 as i32));
                        return Ok(());
                    }
                    geometry::combine::combine_with_geometry(src, o, mode.unwrap_or(0), &xf, tol)
                }
                "outline" if src.outline_bad_number(&wm) => {
                    r.op_hr.insert(i, fmt_hr(0x8899_0011u32 as i32));
                    return Ok(());
                }
                "outline" => geometry::combine::outline(src, &wm, tol),
                "widen" => {
                    let w = width.ok_or("widen requires a `width`")?;
                    let st = match style {
                        Some(id) => r.styles.get(id.as_str()).ok_or("undefined stroke style")?.clone(),
                        None => StrokeStyle::default(),
                    };
                    // Widen's optional worldTransform (ID2D1Geometry::Widen) — the device/elliptical-pen
                    // widen (fuzz 5289); None ⇒ identity, so existing scenes are unchanged.
                    let xf = transform.unwrap_or(IDENTITY);
                    if src.widen_query_bad_number(w, &st, &xf, tol, None) {
                        r.op_hr.insert(i, fmt_hr(0x8899_0011u32 as i32));
                        return Ok(());
                    }
                    src.widen_figures(w, &st, &xf, [1.0, 1.0], tol)
                }
                _ => return Err("geometry op not implemented (scanline region engine)".into()),
            };
            let figures = match &result {
                Geometry::Path { figures, .. } => figures.len(),
                _ => 0,
            };
            let segments = geometry::combine::segment_count(&result);
            let b = result.get_bounds(&IDENTITY);
            r.queries.push(QueryRec {
                op: i,
                kind: format!("geometry_op:{operation}"),
                got: serde_json::json!({"figures": figures, "segments": segments, "bounds": b, "stream": result_stream(&result)}),
            });
            r.geoms.insert(id.as_str(), result);
        }
        Op::Tessellate { geometry, tolerance, world } => {
            let g = r.geoms.get(geometry.as_str()).ok_or_else(|| format!("undefined resource {geometry:?}"))?;
            if g.tessellate_bad_number(&world.unwrap_or(IDENTITY)) {
                r.op_hr.insert(i, fmt_hr(0x8899_0011u32 as i32));
                return Ok(());
            }
            let tris = geometry::tessellate::tessellate(g, &world.unwrap_or(IDENTITY), tolerance.unwrap_or(DEFAULT_TOLERANCE));
            let triangles: Vec<_> = tris.iter().map(|t| serde_json::json!([t[0], t[1], t[2]])).collect();
            r.queries.push(QueryRec {
                op: i,
                kind: "tessellate".into(),
                got: serde_json::json!({ "triangles": triangles }),
            });
        }
        Op::QueryGeom { geometry, kind, other, length, point, width, style, transform, tolerance, world } => {
            let g = r.geoms.get(geometry.as_str()).ok_or_else(|| format!("undefined resource {geometry:?}"))?;
            let (tol, wm) = (tolerance.unwrap_or(DEFAULT_TOLERANCE), world.unwrap_or(IDENTITY));
            let style_of = |s: &Option<String>| -> Result<StrokeStyle, String> {
                match s {
                    Some(id) => Ok(r.styles.get(id.as_str()).ok_or("undefined stroke style")?.clone()),
                    None => Ok(StrokeStyle::default()),
                }
            };
            let got = match kind.as_str() {
                "compare" => {
                    let n = other.as_ref().ok_or("compare requires an `other` geometry")?;
                    let o = r.geoms.get(n.as_str()).ok_or_else(|| format!("undefined resource {n:?}"))?;
                    let xf = transform.unwrap_or(IDENTITY);
                    if g.compare_bad_number(o, &xf) {
                        r.op_hr.insert(i, fmt_hr(0x8899_0011u32 as i32));
                        return Ok(());
                    }
                    serde_json::json!(geometry::combine::compare_with_geometry(g, o, &xf, tol))
                }
                "point_at_length" => {
                    let len = length.ok_or("point_at_length requires a `length`")?;
                    let (p, t) = g
                        .point_at_length(len, &wm, tol)
                        .ok_or("point_at_length on empty geometry")?;
                    serde_json::json!([p[0], p[1], t[0], t[1]])
                }
                "stroke_contains" => {
                    let p = point.ok_or("stroke_contains requires a `point`")?;
                    let w = width.ok_or("stroke_contains requires a `width`")?;
                    // The widener fails the query as it fails a stroke (BAD_NUMBER, no result).
                    let st = style_of(style)?;
                    if g.widen_query_bad_number(w, &st, &wm, tol, Some(p)) {
                        r.op_hr.insert(i, fmt_hr(0x8899_0011u32 as i32));
                        return Ok(());
                    }
                    serde_json::json!(g.stroke_contains(p, w, &st, &wm, tol))
                }
                "widened_bounds" => {
                    let w = width.ok_or("widened_bounds requires a `width`")?;
                    let st = style_of(style)?;
                    if g.widen_query_bad_number(w, &st, &wm, tol, None) {
                        r.op_hr.insert(i, fmt_hr(0x8899_0011u32 as i32));
                        return Ok(());
                    }
                    serde_json::json!(g.widened_bounds(w, &st, &wm, tol))
                }
                _ => return Err("query_geom kind not implemented".into()),
            };
            r.queries.push(QueryRec { op: i, kind: kind.clone(), got });
        }
        // A plain inline bitmap, and `CreateBitmapFromWicBitmap` (a WIC source → premultiplied BGRA),
        // are the same `from_straight_rgba` conversion and share the bitmap namespace.
        // Raw float texels upload verbatim (premultiplied, HDR allowed) as R32G32B32A32_FLOAT, or as
        // R16G16B16A16_FLOAT when the scene names format 10 — as the harness does.
        Op::Bitmap { id, size, data, format, alpha, dpi, .. } if !data.is_empty() => {
            dc.note_texture_upload();
            r.bitmaps.insert(id, Bitmap { width: size[0], height: size[1], premul_bgra: data.clone(), format: d2d::abi::DXGI_FORMAT(format.unwrap_or(87) as i32), alpha: d2d::abi::D2D1_ALPHA_MODE(alpha.unwrap_or(1) as i32), dpi: (dpi.unwrap_or(96.0), dpi.unwrap_or(96.0)) });
        }
        Op::Bitmap { id, size, rgba_f32, format, .. } if !rgba_f32.is_empty() => {
            dc.note_texture_upload();
            let half = *format == Some(d2d::abi::DXGI_FORMAT_R16G16B16A16_FLOAT.0 as u32);
            let premul_bgra = if half {
                rgba_f32.iter().flat_map(|&v| d2d::float::f32_to_f16(v).to_le_bytes()).collect()
            } else {
                rgba_f32.iter().flat_map(|v| v.to_le_bytes()).collect()
            };
            let fmt = if half { d2d::abi::DXGI_FORMAT_R16G16B16A16_FLOAT } else { d2d::abi::DXGI_FORMAT_R32G32B32A32_FLOAT };
            r.bitmaps.insert(id, Bitmap { width: size[0], height: size[1], premul_bgra, format: fmt, alpha: d2d::abi::D2D1_ALPHA_MODE_PREMULTIPLIED, dpi: (96.0, 96.0) });
        }
        // An RGBA bitmap keeps its DPI, as the native harness creates it (WIC bitmaps are 96 dpi).
        Op::Bitmap { id, size, rgba, dpi, .. } => {
            dc.note_texture_upload();
            let mut b = Bitmap::from_straight_rgba(size[0], size[1], rgba);
            b.dpi = (dpi.unwrap_or(96.0), dpi.unwrap_or(96.0));
            r.bitmaps.insert(id, b);
        }
        Op::WicBitmap { id, size, rgba, .. } => {
            dc.note_texture_upload();
            r.bitmaps.insert(id, Bitmap::from_straight_rgba(size[0], size[1], rgba));
        }
        Op::DrawBitmap { bitmap, dest, source, opacity, interpolation, overload: _ } => {
            let b = r.bitmaps.get(bitmap.as_str()).ok_or("bitmap not a 1E inline bitmap")?;
            dc.draw_bitmap(b, *dest, *source, *opacity, *interpolation);
        }
        Op::DrawImage { image, offset, image_rect, interpolation, composite } => {
            if let Some((t, input)) = r.color_mgmt.get(image.as_str()) {
                // ColorManagement output: apply the transform per premultiplied texel of the input
                // bitmap, then draw the result as a bitmap under the requested composite mode.
                // A SIMPLE-profile transform: native latches E_INVALIDARG here at EndDraw (cms2-04/05).
                if r.cm_fail.contains(image.as_str()) {
                    r.cm_fail_tag = Some(i as u64);
                }
                // On a high-precision (Float32) target the colour transform stays in f32: the 8bpc
                // bitmap path below would quantise the result and, drawn onto a linear float target,
                // re-encode it — doubling the curve. Route through the effect graph, which evaluates
                // the transform per float texel at the context precision (the 8bpc targets of
                // cms-01..09 keep the byte-exact bitmap path — they never reach this branch).
                if dc.target_bitmap().is_some_and(|tb| d2d::float::is_highp(tb.desc.format)) {
                    if let Some(e) = r.effects.get(image.as_str()) {
                        e.mark_drawn();
                        dc.draw_image_effect(e, *offset, *interpolation, *composite);
                    }
                } else {
                    let mut out = r.bitmaps.get(*input).ok_or("color management input is not a bitmap")?.clone();
                    for px in out.premul_bgra.chunks_exact_mut(4) {
                        let o = t.apply([px[2] as f32 / 255.0, px[1] as f32 / 255.0, px[0] as f32 / 255.0, px[3] as f32 / 255.0]);
                        let q = |v: f32| (v * 255.0).round_ties_even().clamp(0.0, 255.0) as u8;
                        px[0] = q(o[2]);
                        px[1] = q(o[1]);
                        px[2] = q(o[0]);
                        px[3] = q(o[3]);
                    }
                    dc.draw_image_bitmap(&out, *offset, *image_rect, *interpolation, *composite);
                }
            } else if let Some(list) = r.lists.get(image.as_str()) {
                dc.draw_image_list(list, *offset, *image_rect, *interpolation, *composite);
            } else if let Some(src) = r.floods.get(image.as_str()).filter(|_| !dc.target_bitmap().is_some_and(|t| d2d::float::is_highp(t.desc.format))) {
                // A Flood is an infinite image; `draw_flood` composites it over the whole target with
                // the full composite table (layers-10 and the drawimage flood-mode sweep).
                dc.draw_flood(*src, *offset, *image_rect, *interpolation, *composite);
            } else if let Some(e) = r.effects.get(image.as_str()) {
                e.mark_drawn(); // a HistogramOutput readback becomes valid once drawn (matches native)
                dc.draw_image_effect_rect(e, *offset, *image_rect, *interpolation, *composite);
            } else if let Some(b) = r.bitmaps.get(image.as_str()) {
                dc.draw_image_bitmap(b, *offset, *image_rect, *interpolation, *composite);
            } else {
                // The image id resolves to nothing (e.g. a command list whose recording native rejected,
                // so it was never created) → native's harness records "undefined resource" and draws
                // nothing; mirror that status and keep diffing.
                r.op_hr.insert(i, format!("err: undefined resource {image:?}"));
            }
        }
        Op::ColorContext { id, space, profile, dxgi, simple } => {
            use d2d::abi::*;
            let cc = match space.as_str() {
                "srgb" => ColorContext::from_color_space(D2D1_COLOR_SPACE_SRGB),
                "scrgb" => ColorContext::from_color_space(D2D1_COLOR_SPACE_SCRGB),
                "custom" => ColorContext::from_profile_bytes(profile),
                "dxgi" => ColorContext::from_dxgi_color_space(DXGI_COLOR_SPACE_TYPE(
                    dxgi.ok_or("dxgi color context without a color space")? as i32,
                )),
                "simple" => {
                    let s = simple.as_ref().ok_or("simple color context without a profile")?;
                    ColorContext::from_simple_color_profile(SimpleProfile {
                        red: s.red,
                        green: s.green,
                        blue: s.blue,
                        white: s.white,
                        gamma: s.gamma,
                    })
                }
                other => return Err(format!("unknown color context space {other}")),
            };
            r.color_contexts.insert(id, cc);
        }
        Op::Effect { id, clsid, inputs, properties } if eff_guid(clsid) == d2d::abi::CLSID_D2D1ColorManagement => {
            // ColorManagement: a 1:1 colour transform between two colour contexts. Properties:
            // 0=src ctx, 2=dst ctx (both iunknown), 4=alpha mode; intents (1,3) and quality (5) do
            // not change the sRGB↔scRGB result. Built now, applied at draw_image.
            let ctx = |idx: u32| {
                properties.iter().find(|p| p.index == idx).and_then(|p| match &p.value {
                    Value::Iunknown(c) => r.color_contexts.get(c.as_str()),
                    _ => None,
                })
            };
            let src = ctx(0).ok_or("color management without a source context")?;
            let dst = ctx(2).ok_or("color management without a destination context")?;
            let alpha = properties.iter().find(|p| p.index == 4).map_or(1, |p| match p.value {
                Value::Enum(v) | Value::Uint32(v) => v,
                _ => 1,
            });
            let input = inputs.first().ok_or("color management without an input")?;
            // A SIMPLE-profile context makes native fail realization (EndDraw E_INVALIDARG, cms2-04/05).
            // Carry an identity transform so draw_image still runs, and let the draw latch the failure.
            let t = if src.is_simple() || dst.is_simple() {
                r.cm_fail.insert(id);
                ColorTransform::identity(alpha == 2)
            } else {
                ColorTransform::with_options(src, dst, alpha == 2, [1,3].map(|i| properties.iter().find(|p|p.index == i).map_or(0,|p| match p.value { Value::Enum(v) | Value::Uint32(v) => v, _ => 0 })), properties.iter().find(|p|p.index == 5).map_or(1,|p| match p.value { Value::Enum(v) | Value::Uint32(v) => v, _ => 1 })).ok_or("color management context pair not reproduced")?
            };
            // ColorManagement is a 1:1 transform, so register it as a GRAPH effect under `id` too:
            // that lets it feed ANOTHER effect's input (pdnv5-asifcolorspace: ColorManagement scRGB→sRGB
            // → CrossFade) in FLOAT — the node evaluates the transform in f32, so the pull graph keeps
            // the context (float, for scRGB) precision until the single output quantise, exactly as
            // native materializes a non-linkable effect's intermediate at the context buffer precision.
            // (A pre-realized 8bpc bitmap here quantised the cms before CrossFade, off by ±1 on G/B.)
            // The direct-draw path (draw_image) still goes through `color_mgmt` (8bpc output, byte-exact
            // for cms-01..09), and SIMPLE-profile failure latching is unchanged.
            if let Ok(eff) = dc.create_effect(d2d::abi::CLSID_D2D1ColorManagement) {
                eff.set_cms(Some(t.clone()));
                if let Some(b) = r.bitmaps.get(input.as_str()) {
                    eff.set_input(0, FxInput::Bitmap(Rc::new(b.clone())));
                } else if let Some(e) = r.effects.get(input.as_str()) {
                    eff.set_input(0, FxInput::Effect(e.clone()));
                }
                r.effects.insert(id, eff);
            }
            r.color_mgmt.insert(id, (t, input.as_str()));
        }
        Op::Effect { id, clsid, properties, .. } if eff_guid(clsid) == d2d::abi::CLSID_D2D1BitmapSource => {
            // BitmapSource wraps an IWICBitmapSource (property 0, iunknown) as an image. Native has no
            // image input; bridge the referenced source bitmap as the core effect's input 0. At default
            // (Scale 1:1, no orientation) the output is the source verbatim.
            let src = properties.iter().find(|p| p.index == 0).and_then(|p| match &p.value {
                Value::Iunknown(s) => Some(s.as_str()),
                _ => None,
            }).ok_or("BitmapSource without a WicBitmapSource")?;
            let b = r.bitmaps.get(src).ok_or("BitmapSource source not a bitmap")?.clone();
            let eff = dc.create_effect(d2d::abi::CLSID_D2D1BitmapSource).map_err(|_| "BitmapSource not implemented".to_string())?;
            eff.set_input(0, FxInput::Bitmap(Rc::new(b)));
            // Scale / InterpolationMode / Orientation / … (every property but 0, the source IUnknown).
            for p in properties.iter().filter(|p| p.index != 0) {
                eff.set_value(p.index, eff_prop(&p.value)).map_err(|_| "BitmapSource property rejected".to_string())?;
            }
            r.effects.insert(id, eff);
        }
        Op::Effect { id, clsid, inputs, properties } => {
            // A bare Flood drawn directly keeps 2A's composite-aware draw_flood path (layers-10); the
            // effect graph (2B) also registers every effect so one can feed another (custom-fx-03).
            if clsid == "61c23c20-ae69-4d8e-94cf-50078df638f2" && inputs.is_empty() {
                if let Some(c) = properties.iter().find(|p| p.index == 0).and_then(|p| match &p.value {
                    Value::Vector4(c) => Some(*c),
                    _ => None,
                }) {
                    let q = |v: f32| (v * 255.0).round_ties_even().clamp(0.0, 255.0) as i32;
                    r.floods.insert(id, [q(c[2]), q(c[1]), q(c[0]), q(c[3])]);
                }
            }
            // An unregistered CLSID: native's CreateEffect returns ERROR_NOT_FOUND (0x80070490) to
            // the caller — not latched — and the scene continues (`v2-hresult`). Record that op HR and
            // skip the effect; a later draw_image of it would error, but such scenes don't reference it.
            let eff = match dc.create_effect(eff_guid(clsid)) {
                Ok(e) => e,
                Err(hr) => {
                    r.op_hr.insert(i, fmt_hr(hr.0));
                    return Ok(());
                }
            };
            for (in_idx, input) in inputs.iter().enumerate() {
                let img = if let Some(b) = r.bitmaps.get(input.as_str()) {
                    FxInput::Bitmap(Rc::new(b.clone()))
                } else if let Some(e) = r.effects.get(input.as_str()) {
                    FxInput::Effect(e.clone())
                } else if let Some(list) = r.lists.get(input.as_str()).cloned() {
                    // A command list stays a list: each draw renders it over the rect that draw needs.
                    FxInput::list(list)
                } else {
                    // The input is an undefined resource (e.g. an effect whose property SetValue native
                    // rejected, leaving it undefined): native's harness records "undefined resource" at
                    // this op, creates no effect, and continues (r16 21017 op12 input fx5, 21570 op10/11).
                    r.op_hr.insert(i, format!("err: undefined resource {input:?}"));
                    return Ok(());
                };
                eff.set_input(in_idx as u32, img);
            }
            // A SetValue native rejects (E_INVALIDARG — a mistyped/out-of-range built-in property, e.g.
            // Turbulence Seed-as-uint32 or an EdgeDetection/Sepia AlphaMode of 0) is recorded as this
            // op's HR and the effect is left UNDEFINED; the scene continues and a later draw_image of it
            // reports "undefined resource" (native capture: r16 21017 op7 → E_INVALIDARG, op12 → undefined
            // fx5). Do NOT abort the scene as a SKIP.
            let mut rejected = false;
            for p in properties {
                if let Err(hr) = eff.set_value(p.index, eff_prop(&p.value)) {
                    r.op_hr.insert(i, fmt_hr(hr.0));
                    rejected = true;
                    break;
                }
            }
            if rejected {
                return Ok(());
            }
            r.effects.insert(id, eff);
        }
        Op::SetProperty { target, index, value } => {
            // Change an effect property between draws; this also clears any drawn/realize state, so a
            // later HistogramOutput read invalidates exactly as native does.
            let eff = r.effects.get(target.as_str()).ok_or("set_property: no such effect")?;
            eff.set_value(*index, eff_prop(value)).map_err(|_| "set_property rejected".to_string())?;
        }
        Op::SetInput { target, index, input } => {
            // Rebind an effect's input between draws (ID2D1Effect::SetInput): one effect instance drawn
            // over different inputs across draws, so a per-draw cache keyed on pixel position (the
            // derivative-quad cache) must not reuse the first draw's cached quads.
            let eff = r.effects.get(target.as_str()).ok_or("set_input: no such effect")?.clone();
            let img = if let Some(b) = r.bitmaps.get(input.as_str()) {
                FxInput::Bitmap(Rc::new(b.clone()))
            } else if let Some(e) = r.effects.get(input.as_str()) {
                FxInput::Effect(e.clone())
            } else {
                return Err(format!("set_input: undefined input {input:?}"));
            };
            eff.set_input(*index, img);
        }
        Op::RegisterShaderEffect { clsid, dxbc, cbuffer, cbuffer_bytes, inputs, filter, precision, input_inflate, .. } => {
            // Register a custom effect whose draw transform runs the authored pixel shader (2B-3). A
            // later `effect` op creates it by this CLSID and binds inputs, reusing the arm above.
            let cb: Vec<u8> = if cbuffer_bytes.is_empty() { cbuffer.iter().flat_map(|f| f.to_le_bytes()).collect() } else { cbuffer_bytes.clone() };
            let dxbc = dxbc.clone();
            let id = eff_guid(clsid);
            let n = *inputs;
            let (prec, inflate, filt) = (*precision, *input_inflate, *filter);
            factory
                .register_effect_from_string(id, &d2d_reference::shader_effect_xml(n), Rc::new(move || {
                    Box::new(d2d::effects::PixelShaderEffect { shader_id: id, dxbc: dxbc.clone(), cbuffer: cb.clone(), input_count: n, precision: prec, input_inflate: inflate, filter: filt })
                }))
                .map_err(|e| format!("register shader effect: {e:?}"))?;
        }
        Op::RegisterVertexEffect { clsid, vs_dxbc, ps_dxbc, vertices, inputs, viewport, .. } => {
            // Register a custom effect whose draw transform runs the authored vertex + pixel shaders
            // over a vertex buffer (3C). A later `effect` op creates it, reusing the arm above.
            let (vs, ps) = (vs_dxbc.clone(), ps_dxbc.clone());
            let verts: Vec<u8> = vertices.iter().flat_map(|f| f.to_le_bytes()).collect();
            let id = eff_guid(clsid);
            let (n, vp) = (*inputs, *viewport);
            factory
                .register_effect_from_string(id, &d2d_reference::shader_effect_xml(n), Rc::new(move || Box::new(VertexEffectImpl { vs: vs.clone(), ps: ps.clone(), verts: verts.clone(), inputs: n, viewport: vp })))
                .map_err(|e| format!("register vertex effect: {e:?}"))?;
        }
        Op::RegisterGraphEffect { clsid, offset, border, bounds } => {
            // A later `effect` op creates it by this CLSID and binds the input, reusing the arm above.
            let id = eff_guid(clsid);
            let (off, bor, bnd) = (*offset, *border, *bounds);
            factory
                .register_effect_from_string(id, &d2d_reference::shader_effect_xml(1), Rc::new(move || Box::new(GraphEffectImpl { offset: off, border: bor, bounds: bnd })))
                .map_err(|e| format!("register graph effect: {e:?}"))?;
        }
        Op::RegisterBlendEffect { clsid, desc, factor } => {
            let id = eff_guid(clsid);
            let (d, f) = (*desc, *factor);
            factory
                .register_effect_from_string(id, &d2d_reference::shader_effect_xml(2), Rc::new(move || Box::new(BlendEffectImpl { desc: d, factor: f })))
                .map_err(|e| format!("register blend effect: {e:?}"))?;
        }
        Op::RegisterComputeEffect { clsid, dxbc, cbuffer, groups, .. } => {
            let id = eff_guid(clsid);
            let (dx, cb, g) = (dxbc.clone(), cbuffer.clone(), *groups);
            factory
                .register_effect_from_string(id, &d2d_reference::shader_effect_xml(0), Rc::new(move || Box::new(ComputeEffectImpl { shader_id: id, dxbc: dx.clone(), cbuffer: cb.clone(), groups: g })))
                .map_err(|e| format!("register compute effect: {e:?}"))?;
        }
        Op::LookupTable3D { id, input, extents, data, precision } => {
            let eff = dc.create_effect(d2d::abi::CLSID_D2D1LookupTable3D).map_err(|_| "LookupTable3D not implemented".to_string())?;
            let img = if let Some(b) = r.bitmaps.get(input.as_str()) {
                FxInput::Bitmap(Rc::new(b.clone()))
            } else if let Some(e) = r.effects.get(input.as_str()) {
                FxInput::Effect(e.clone())
            } else {
                return Err("lut3d input is not a bitmap or effect".into());
            };
            eff.set_input(0, img);
            // Mirror native's per-precision stored LUT: quantize each f32 channel to the table's
            // precision and decode it back the way WARP's UNORM sampler does, so the core interpolates
            // the values native holds. UNORM8 decodes as byte·257/65536 (NOT byte/255 — the 0.9999847
            // factor compounds through the trilinear); UNORM16 = u16/65535 (lossless); else 32BPC_FLOAT.
            let prec = (*precision).unwrap_or(5);
            let q = |v: f32| match prec {
                1 => (v.clamp(0.0, 1.0) * 255.0).round() * (257.0 / 65536.0),
                3 => (v.clamp(0.0, 1.0) * 65535.0).round() / 65535.0,
                _ => v,
            };
            let lut_data = data.chunks_exact(4).map(|c| [q(c[0]), q(c[1]), q(c[2]), q(c[3])]).collect();
            let lut = dc.create_lookup_table3d(*extents, lut_data).map_err(|_| "lut3d rejected".to_string())?;
            eff.set_lookup_table(lut);
            r.effects.insert(id, eff);
        }
        Op::SpriteBatch { id, sprites } => {
            let sprites = sprites
                .iter()
                // The native harness passes the source as D2D_RECT_U (`as u32`, saturating).
                .map(|s| extras::Sprite { dest: s.dest, source: s.source.map(|r| r.map(|v| v as u32 as f32)), color: s.color, transform: s.transform })
                .collect();
            r.sprite_batches.insert(id, dc.create_sprite_batch(sprites));
        }
        Op::DrawSpriteBatch { batch, bitmap, interpolation, options } => {
            let sb = r.sprite_batches.get(batch.as_str()).ok_or("undefined sprite batch")?;
            let bm = r.bitmaps.get(bitmap.as_str()).ok_or("sprite batch of a non-1E bitmap")?;
            // As d2d-ffi's DrawSpriteBatch: per-primitive mode latches WRONG_STATE and draws nothing
            // (native's DrawSpriteBatch; r11 16401 op 9).
            if dc.aliased_draw_check() {
                dc.draw_sprite_batch(sb, bm, *interpolation, *options);
            }
        }
        // Phase 2E: parse an SVG document into the mutable DOM, and render it through the same
        // geometry/brush calls native's D2DSvgDocument::Render emits.
        Op::SvgDocument { id, xml, viewport } => {
            let doc = d2d::svg::Document::parse(xml, *viewport).map_err(|_| "svg parse rejected")?;
            r.svgs.insert(id, doc);
        }
        Op::DrawSvgDocument { doc } => {
            r.svgs.get(doc.as_str()).ok_or("draw of undefined svg document")?.render(dc);
        }
        // SVG DOM editing (phase 3A): the same ops the harness drives through COM, replayed on the
        // core `Document` so pixels and query results match the WARP oracle through one source of
        // truth. Element handles name a node by (doc key, stable node id).
        Op::SvgGetRoot { doc, elem } => {
            let id = r.svgs.get(doc.as_str()).ok_or("svg doc")?.root_id().ok_or("no root")?;
            r.svg_elems.insert(elem, (doc, id));
        }
        Op::SvgCreateChild { parent, tag, elem } => {
            let (d, pid) = *r.svg_elems.get(parent.as_str()).ok_or("svg parent")?;
            let id = r.svgs.get_mut(d).unwrap().create_child(pid, tag).map_err(|c| format!("create_child {c:#x}"))?;
            r.svg_elems.insert(elem, (d, id));
        }
        Op::SvgSetAttr { elem, name, value } => {
            let (d, id) = *r.svg_elems.get(elem.as_str()).ok_or("svg elem")?;
            r.svgs.get_mut(d).unwrap().set_attr_string(id, name, value).map_err(|c| format!("set_attr {c:#x}"))?;
        }
        Op::SvgFirstChild { parent, elem } => {
            let (d, pid) = *r.svg_elems.get(parent.as_str()).ok_or("svg parent")?;
            let cid = r.svgs.get(d).unwrap().first_child_id(pid).ok_or("no first child")?;
            r.svg_elems.insert(elem.as_str(), (d, cid));
        }
        Op::SvgSetText { elem, text } => {
            let (d, id) = *r.svg_elems.get(elem.as_str()).ok_or("svg elem")?;
            // SetTextValue on a non-text node rejects with a real HRESULT that native records per-op
            // (E_INVALIDARG for a container like <title>/<desc>/<rect>); record it and keep going rather
            // than aborting the scene as a SKIP.
            if let Err(code) = r.svgs.get_mut(d).unwrap().set_text_value(id, text) {
                r.op_hr.insert(i, fmt_hr(code));
            }
        }
        Op::SvgRemoveAttr { elem, name } => {
            let (d, id) = *r.svg_elems.get(elem.as_str()).ok_or("svg elem")?;
            r.svgs.get_mut(d).unwrap().remove_attr(id, name);
        }
        Op::SvgSetAttrPod { elem, name, pod, floats } => {
            let (d, id) = *r.svg_elems.get(elem.as_str()).ok_or("svg elem")?;
            let v = svg_pod_value(*pod, floats);
            r.svgs.get_mut(d).unwrap().set_attr_value(id, name, v).map_err(|c| format!("set_attr_pod {c:#x}"))?;
        }
        // `mutate` is a harness-only flag (scramble-then-set the native object); the core sets the
        // final value directly either way.
        Op::SvgSetObject { elem, name, object, mutate: _ } => {
            let (d, id) = *r.svg_elems.get(elem.as_str()).ok_or("svg elem")?;
            r.svgs.get_mut(d).unwrap().set_attr_value(id, name, svg_object_value(object)).map_err(|c| format!("set_object {c:#x}"))?;
        }
        Op::SvgAppendChild { parent, child } => {
            let (d, pid) = *r.svg_elems.get(parent.as_str()).ok_or("svg parent")?;
            let (_, cid) = *r.svg_elems.get(child.as_str()).ok_or("svg child")?;
            r.svgs.get_mut(d).unwrap().reparent(pid, cid, None).map_err(|c| format!("append {c:#x}"))?;
        }
        Op::SvgInsertChildBefore { parent, child, reference } => {
            let (d, pid) = *r.svg_elems.get(parent.as_str()).ok_or("svg parent")?;
            let cid = r.svg_elems.get(child.as_str()).ok_or("svg child")?.1;
            let rid = r.svg_elems.get(reference.as_str()).ok_or("svg ref")?.1;
            r.svgs.get_mut(d).unwrap().reparent(pid, cid, Some(rid)).map_err(|c| format!("insert {c:#x}"))?;
        }
        Op::SvgRemoveChild { parent, child } => {
            let (d, pid) = *r.svg_elems.get(parent.as_str()).ok_or("svg parent")?;
            let cid = r.svg_elems.get(child.as_str()).ok_or("svg child")?.1;
            r.svgs.get_mut(d).unwrap().remove_child(pid, cid).map_err(|c| format!("remove {c:#x}"))?;
        }
        Op::SvgReplaceChild { parent, new, old } => {
            let (d, pid) = *r.svg_elems.get(parent.as_str()).ok_or("svg parent")?;
            let nid = r.svg_elems.get(new.as_str()).ok_or("svg new")?.1;
            let oid = r.svg_elems.get(old.as_str()).ok_or("svg old")?.1;
            r.svgs.get_mut(d).unwrap().replace_child(pid, nid, oid).map_err(|c| format!("replace {c:#x}"))?;
        }
        Op::SvgFindById { doc, id, elem } => {
            let nid = r.svgs.get(doc.as_str()).ok_or("svg doc")?.find_id(id).ok_or("id not found")?;
            r.svg_elems.insert(elem, (doc, nid));
        }
        Op::SvgSetViewport { doc, size } => {
            r.svgs.get_mut(doc.as_str()).ok_or("svg doc")?.set_viewport(size[0], size[1]).map_err(|c| format!("viewport {c:#x}"))?;
        }
        Op::SvgDeserialize { doc, xml, elem } => {
            let viewport = r.svgs.get(doc.as_str()).ok_or("svg doc")?.viewport;
            let parsed = d2d::svg::Document::parse(xml, viewport).map_err(|c| format!("deserialize {c:#x}"))?;
            let id = r.svgs.get_mut(doc.as_str()).unwrap().import_root(parsed);
            r.svg_elems.insert(elem, (doc, id));
        }
        Op::SvgSetRoot { doc, elem } => {
            let id = r.svg_elems.get(elem.as_str()).ok_or("svg elem")?.1;
            r.svgs.get_mut(doc.as_str()).ok_or("svg doc")?.set_root(id).map_err(|c| format!("set_root {c:#x}"))?;
        }
        Op::SvgQuery { kind, doc, elem, name, index, pod } => {
            let got = svg_query(r, kind, doc.as_deref(), elem.as_deref(), name.as_deref(), *index, *pod)?;
            r.queries.push(QueryRec { op: i, kind: format!("svg_{kind}"), got });
        }
        // ID2D1SvgGlyphStyle is an FFI-only value holder (it stores ID2D1Brush COM pointers the core
        // crate has no equivalent for), so the core SKIPs it here — the FFI DLL's getters are validated
        // against the oracle by the Wine FFI gate / VM render, not by this core path.
        Op::SvgGlyphStyleProbe => return Err("FFI-only ID2D1SvgGlyphStyle, validated by the FFI gate".into()),
        Op::InkStyle { id, nib_shape, nib_transform } => {
            r.ink_styles.insert(id, dc.create_ink_style(*nib_shape, *nib_transform));
        }
        Op::Ink { id, start, segments } => {
            let mut ink = dc.create_ink(*start);
            let segs: Vec<_> = segments
                .iter()
                .map(|s| extras::ink::InkBezierSegment { p1: s.p1, p2: s.p2, end: s.end })
                .collect();
            ink.add_segments(&segs);
            r.inks.insert(id, ink);
        }
        Op::InkRemoveSegmentsAtEnd { ink, count } => {
            let k = r.inks.get_mut(ink.as_str()).ok_or("undefined ink")?;
            // Over-remove rejects with E_INVALIDARG and leaves the ink unchanged; record it per-op.
            if let Err(code) = k.remove_segments_at_end(*count) {
                r.op_hr.insert(i, fmt_hr(code));
            }
        }
        Op::DrawInk { ink, brush, style } => {
            let k = r.inks.get(ink.as_str()).ok_or("undefined ink")?;
            let st = style.as_ref().map(|s| r.ink_styles.get(s.as_str()).ok_or("undefined ink style")).transpose()?;
            if let Some(b) = r.solids.get(brush.as_str()) {
                dc.draw_ink(k, b, st);
            } else if let Some(pb) = r.pbrushes.get(brush.as_str()) {
                dc.draw_ink_brush(k, pb, st);
            } else {
                return Err(format!("undefined resource {brush:?}"));
            }
        }
        Op::GradientMesh { id, patches } => {
            let patches = patches
                .iter()
                .map(|p| extras::mesh::MeshPatch { points: p.points, colors: p.colors, extend: p.extend })
                .collect();
            r.meshes.insert(id, dc.create_gradient_mesh(patches));
        }
        Op::DrawGradientMesh { mesh } => {
            let gm = r.meshes.get(mesh.as_str()).ok_or("undefined gradient mesh")?;
            dc.draw_gradient_mesh(gm);
        }
        Op::FillMesh { triangles, brush } => {
            // Native requires aliased mode (per-primitive latches D2DERR_WRONG_STATE at EndDraw) and
            // fills each triangle on its own through WARP's D3D triangle rule, for any brush type. The
            // harness resolves the brush first, so an undefined one is "undefined resource" in either
            // mode (r11 16683 op 18).
            if !r.solids.contains_key(brush.as_str()) && !r.pbrushes.contains_key(brush.as_str()) {
                return Err(format!("undefined resource {brush:?}"));
            }
            if dc.aliased_draw_check() {
                if let Some(b) = r.solids.get(brush.as_str()) {
                    for t in triangles {
                        dc.fill_mesh_solid(t, b);
                    }
                } else if let Some(pb) = r.pbrushes.get(brush.as_str()) {
                    for t in triangles {
                        dc.fill_mesh_brush(t, pb);
                    }
                }
            }
        }
        Op::SaveState { id } => {
            let mut block = factory.create_drawing_state_block();
            dc.save_drawing_state(&mut block);
            r.states.insert(id, block);
        }
        Op::RestoreState { id } => {
            let block = *r.states.get(id.as_str()).ok_or("restore before save")?;
            dc.restore_drawing_state(&block);
        }
        Op::PushClip { rect, antialias } => dc.push_axis_aligned_clip(*rect, *antialias),
        Op::PopClip => dc.pop_axis_aligned_clip(),
        Op::PushLayer { bounds, mask, opacity, opacity_brush, mask_antialias, mask_transform, layer_options } => {
            let mask_geom = match mask {
                Some(id) => Some(r.geoms.get(id.as_str()).ok_or_else(|| format!("undefined resource {id:?}"))?),
                None => None,
            };
            let solid = opacity_brush.as_ref().and_then(|id| r.solids.get(id.as_str())).map(|s| s.as_pixel_brush());
            let brush = match opacity_brush {
                Some(id) => Some(r.pbrushes.get(id.as_str()).or(solid.as_ref()).ok_or("undefined layer opacity brush")?),
                None => None,
            };
            dc.push_layer(*bounds, mask_geom, mask_antialias.unwrap_or(0), *mask_transform, *opacity, brush, layer_options.unwrap_or(0));
        }
        Op::PopLayer => dc.pop_layer(),
        Op::FillOpacityMask { bitmap, brush, content, dest, source, overload: _ } => {
            let bm = r.bitmaps.get(bitmap.as_str()).ok_or("opacity mask of a non-1E bitmap")?;
            if let Some(b) = r.solids.get(brush.as_str()) {
                dc.fill_opacity_mask(bm, b, *content, *dest, *source);
            } else if let Some(pb) = r.pbrushes.get(brush.as_str()) {
                dc.fill_opacity_mask_brush(bm, pb, *content, *dest, *source);
            } else {
                return Err(format!("undefined resource {brush:?}"));
            }
        }
        Op::CommandList { id, ops } => {
            // Record the nested ops into a command-list image, then seal it (matches the harness,
            // which SetTarget(list)s, replays the nested ops and Close()s). Resources the nested ops
            // define share this scene's `Res`; replay is driven by a later draw_image.
            // A draw error latched before CreateCommandList is returned by the list's Close (d2d-ffi
            // CreateCommandList's `pending_err`; VM seeds 15026/16026), and the harness then leaves the
            // list undefined (fuzz 7770: op 6's per-primitive FillMesh WRONG_STATE surfaces at op 11).
            let list = dc.create_command_list();
            dc.record_into(&list);
            // The nested replay reuses op indices (j), which collide with this scene's outer op indices
            // in `op_hr`; snapshot and restore around it so a nested status never leaks onto an outer op.
            // If a nested op fails geometry-sink validation (e.g. a path with an invalid arc radius),
            // native's command-list recording fails and surfaces that HRESULT on THIS command_list op
            // (VM batch-5: 28 command_list cases), so propagate the first nested error to op `i`.
            let saved_hr = r.op_hr.clone();
            // Clear the nested indices first so find_map below sees only THIS replay's statuses — a
            // prior command_list (or outer op) at the same index must not leak in (fuzz 2319: a valid
            // nested primitive at index 1 inherited an earlier command_list's E_INVALIDARG otherwise).
            for j in 0..ops.len() {
                r.op_hr.remove(&j);
            }
            // A nested op on an undefined resource aborts native's harness (its `?`): the list is never
            // closed or defined and the command_list op records that error (r23 28317: a list drawing a
            // list whose Close failed).
            let mut undefined = None;
            for (j, nested) in ops.iter().enumerate() {
                dc.set_tags(j as u64, 0); // the harness tags each nested op during recording
                match exec_op(dc, factory, j, nested, r) {
                    Err(e) if e.starts_with("undefined resource") => undefined = Some(format!("err: {e}")),
                    res => res?,
                }
                if let Some(e) = undefined.take().or_else(|| r.op_hr.get(&j).filter(|h| h.starts_with("err: undefined")).cloned()) {
                    undefined = Some(e);
                    break;
                }
                // Native's command-list recording freezes its tag at the FIRST faulting nested op
                // (recording-scoped: the main context stays S_OK, a later top-level op is unaffected) and
                // records nothing more into the list — Close surfaces that op's tag. Stop the loop at the
                // fault so a later sub-op's SetTags cannot overwrite the fault tag (fuzz r20: a nested bad
                // path at tag j -> native tag1=j; we leaked the last sub-op's index). The list is left
                // undefined on a nested fault below, so this only corrects the tag, not pixels.
                if r.op_hr.get(&j).is_some_and(|h| h.starts_with("0x") && h != "0x00000000") {
                    break;
                }
            }
            let nested_err = (0..ops.len()).find_map(|j| {
                r.op_hr.get(&j).filter(|h| h.starts_with("0x") && h.as_str() != "0x00000000").cloned()
            });
            r.op_hr = saved_hr;
            dc.stop_recording();
            if let Some(e) = undefined {
                r.op_hr.insert(i, e);
                return Ok(());
            }
            list.close();
            // When a nested op fails, native's command-list recording fails and the list is never
            // created, so leave it undefined (a later draw_image of it is then "undefined resource",
            // matching native) rather than inserting a list we built anyway.
            // A nested op's own failure wins (r11 16163/16465/16519/16902: nested E_INVALIDARG under a
            // pending WRONG_STATE); else Close returns the draw error latched on the context, whether
            // before CreateCommandList (`pending`, 7770) or by a nested draw (r11 16401's per-primitive
            // FillMesh).
            let latched = dc.current_error();
            if let Some(e) = nested_err {
                r.op_hr.insert(i, e);
            } else if latched.is_err() {
                r.op_hr.insert(i, fmt_hr(latched.0));
            } else if list.unbalanced() {
                // A clip or layer left open in the list fails its Close (clclip probes).
                r.op_hr.insert(i, fmt_hr(0x8899_0016u32 as i32));
            } else {
                r.lists.insert(id, list);
            }
        }
        Op::ImageSource { id, size, rgba, .. } => {
            let is = dc.create_image_source_from_wic(Bitmap::from_straight_rgba(size[0], size[1], rgba));
            r.bitmaps.insert(id, is.bitmap.clone());
            r.image_sources.insert(id, is);
        }
        Op::TransformedImageSource { id, source, orientation, scale, interpolation, .. } => {
            if *scale != [1.0, 1.0] {
                return Err("transformed image source scale != 1 (resample not reproduced)".into());
            }
            let src = r.image_sources.get(source.as_str()).ok_or("undefined image source")?;
            let t = dc.create_transformed_image_source(src, *orientation, *scale, *interpolation);
            r.bitmaps.insert(id, t.bitmap.clone());
            r.image_sources.insert(id, t);
        }
        Op::BlendImage { image, mode, offset, target_rect, interpolation } => {
            // All W3C blend modes are wired (reusing the byte-exact Blend-effect math); only Dissolve
            // (4) is left out (noise-coordinate origin unconfirmed). `target_rect` is BlendImage's
            // `imageRectangle` (a source sub-rect), handled by cropping the image to the integer rect
            // before the blend — the same crop the image brush uses; its top-left maps to `offset`.
            if *mode == 4 {
                return Err("blend-image dissolve (noise coordinate origin unconfirmed)".into());
            }
            let bm = r.bitmaps.get(image.as_str()).ok_or("blend image of a non-bitmap image")?;
            dc.blend_image(bm, *mode, *offset, *target_rect, *interpolation);
        }
        Op::MaxScaleFactor { matrix } => {
            r.queries.push(QueryRec {
                op: i,
                kind: "max_scale".into(),
                got: serde_json::json!(print::compute_maximum_scale_factor(*matrix)),
            });
        }
        // Pure layout query: draws nothing (keeps a text_layout scene's pixels blank, matching the
        // oracle). The byte-exact layout check lives in tests/text_layout, which needs the VM fonts.
        Op::TextLayout { text, font, size, max_width, max_height, weight, style, stretch, locale, measuring, brush, origin, align, para, wrap, reading, flow, trimming, features, inline_objects, ranges, line_spacing, .. } => {
            // CreateTextFormat comes first and rejects a fontSize not `> 0.0` with a per-op E_INVALIDARG,
            // as for Op::Text (r11 16515 op 19's −478); then CreateTextLayout rejects a max width/height
            // that is not `>= 0` (as dwrite-ffi's make_layout). A per-range `SetFontSize` runs the SAME
            // `> 0.0` gate on its argument, independent of whether the range covers any text (the setter
            // validates its float before touching the run), so a range size ≤ 0 / NaN / −inf rejects too
            // (r18 23008 −0.0, 23095 −inf, 23116 −9.99e-21, 23444 −0.0, 23482 −478, 23596 nan, 23779/23085
            // −3.4e38, 23867 −1e9). A range with no size (None) inherits the base and adds no gate.
            if !(*size > 0.0) || [*max_width, *max_height].iter().any(|v| v.is_nan() || *v < 0.0)
                || ranges.iter().any(|rg| rg.size.is_some_and(|s| !(s > 0.0)))
            {
                r.op_hr.insert(i, fmt_hr(0x80070057u32 as i32));
                return Ok(());
            }
            // CreateTextLayout refuses a reading and flow direction that are not perpendicular with
            // DWRITE_E_FLOWDIRECTIONCONFLICTS (VM r25: all 1404 layouts follow this, vertical pairs included).
            if (*reading < 2) != (*flow < 2) {
                r.op_hr.insert(i, fmt_hr(0x8898500Bu32 as i32));
                return Ok(());
            }
            // A GDI measuring mode (1/2) lays the run out through GetGdiCompatibleGlyphMetrics, which
            // rejects a DEVICE em (em · pixelsPerDip · transformScale) reaching 2^15 = 32768 (plain
            // CreateTextLayout, measuring 0, never rejects). The limit is PER LAID-OUT GLYPH, so EMPTY text
            // never trips it — native draws nothing, S_OK (r11 16613 op12: "" at em 1e9 GDI_CLASSIC → S_OK,
            // while r17 22835's non-empty "Ag" em 1e9 · ppd 1.25 = 1.25e9 → E_INVALIDARG). A device em that
            // overflows to non-finite takes the degenerate path and succeeds (gdi_compat_layout_em_rejected).
            // `CreateGdiCompatibleTextLayout` is always handed the IDENTITY transform (the layout box is
            // sized in DIPs; the world transform is applied later at DrawTextLayout), so the device em
            // folds only pixelsPerDip, NOT the context's world transform (r17 21201 op20: a 1e-6 world
            // scale is set, yet native still rejects em 1e9 · ppd 1.25). pixelsPerDip is the x DPI / 96.
            // The em limit is checked for every em the layout actually places a glyph at: the base em
            // (whole run) and each per-range SetFontSize that covers ≥1 character — a huge but finite,
            // positive range size (passing the `> 0.0` gate above) still trips GetGdiCompatibleGlyphMetrics
            // when that range's glyphs are laid out under GDI measuring (r18 23982 GDI_CLASSIC range 3.4e38,
            // 23991 GDI_NATURAL range 1e9). An out-of-text or zero-length range places no glyph, so it is
            // not checked — the per-range analogue of the base `!text.is_empty()` guard. A NON-finite BASE
            // em sends the whole run down the degenerate device-em path (draws nothing, S_OK) before any
            // per-range metrics run, so the per-range check is suppressed there: r18 23095 op15 is base inf
            // with a 1e9 range under GDI_CLASSIC and native returns S_OK, not E_INVALIDARG.
            if matches!(*measuring, 1 | 2) && !text.is_empty() {
                let ppd = dc.get_dpi().0 / 96.0;
                let utf16 = text.encode_utf16().count() as u32;
                let em_bad = |em: f32| d2d::DeviceContext::gdi_compat_layout_em_rejected(em, ppd, 1.0);
                // The base em is validated in CreateGdiCompatibleTextLayout at layout-creation time —
                // independent of the draw transform (fuzz 25217: a huge BASE em rejects even under a
                // singular world transform). A per-range SetFontSize, though, only trips
                // GetGdiCompatibleGlyphMetrics when its glyphs are actually measured at draw time, and a
                // SINGULAR world transform collapses the run so they never are — native then draws
                // nothing and returns S_OK (fuzz 24300/24127: normal base + a 3.4e38 range under a
                // zero-linear transform). So gate the per-range check on a non-singular transform; the
                // base check stays unconditional. Verified against the r18–r20 GDI huge-em captures.
                let xf = dc.get_transform();
                let singular = xf[0] * xf[3] - xf[1] * xf[2] == 0.0;
                // Overlapping SetFontSize calls resolve last-wins per position (DWrite stores a run
                // format), so a huge range em that a LATER range overwrites is never the size any glyph
                // is measured at and does NOT trip the limit — fuzz 26772: a 1e9 range [3,6) fully
                // overwritten by a later 26-em range [0,11) draws fine, S_OK. Fold the ranges onto an
                // effective per-position size (base, then each sized range in call order) and test that,
                // not "any range has a bad size".
                let range_em_bad = !singular && size.is_finite() && {
                    let mut eff = vec![*size; utf16 as usize];
                    for rg in ranges.iter() {
                        if let Some(rs) = rg.size {
                            let start = rg.position.min(utf16) as usize;
                            let end = rg.position.saturating_add(rg.length).min(utf16) as usize;
                            eff[start..end].fill(rs);
                        }
                    }
                    eff.iter().any(|&e| em_bad(e))
                };
                if em_bad(*size) || range_em_bad {
                    r.op_hr.insert(i, fmt_hr(0x80070057u32 as i32));
                    return Ok(());
                }
            }
            // Metrics-only when no brush (pt1/pt2 scenes); else DrawTextLayout the computed layout.
            let Some(bid) = brush else { return Ok(()) };
            let (base_bytes, base_face, base_sims) = load_text_font_sims(font, *weight, *style, *stretch).ok_or("font file not available")?;
            let (axes, coords) = instance_axes(&base_bytes, base_face, *weight, *stretch);
            // This op is DrawTextLayout: the IDWriteTextLayout honours the format's alignment, wrap and
            // reading direction, so pass them through (layout_ex models horizontal + vertical alignment).
            let line_spacing = line_spacing.map_or((0, 0.0, 0.0), |[m, h, b]| (m as u8, h, b));
            let fmt = TextFormat { em: *size, base_rtl: *reading & 1 == 1, align: *align as u8, wrap: *wrap as u8, para: *para as u8, flow: *flow as u8, line_spacing };
            // DrawTextLayout = CreateTextLayout(max_width,max_height) + Draw(origin): the box size is
            // independent of the origin, so pass them separately (not packed into a rect, which cancels
            // the box extent in f32 when origin is non-finite — fuzz 18504's -inf origin). layout_ex
            // wraps to the box; the glyph-run path then collapses an off-canvas origin to the clip origin.
            let o = [origin[0], origin[1]];
            // The IDWriteTextLayout per-call inputs our core's layout_ex supports (trimming, typography
            // feature ranges, inline-object placeholders), mapped from the scene — so DrawTextLayout
            // exercises the engine's implementation, not the format-only path. Per-range character
            // formatting (SetFontSize/Weight/Style/Underline over a sub-range) is NOT modelled by
            // layout_ex, so those scenes still diverge as a genuine engine gap.
            let feats: Vec<layout::Feature> = features.iter().map(|fr| {
                let b = fr.tag.as_bytes();
                layout::Feature { start: fr.position, end: fr.position.saturating_add(fr.length),
                    tag: [b.first().copied().unwrap_or(b' '), b.get(1).copied().unwrap_or(b' '), b.get(2).copied().unwrap_or(b' '), b.get(3).copied().unwrap_or(b' ')], value: fr.value }
            }).collect();
            let inl: Vec<layout::InlineObject> = inline_objects.iter().map(|io| layout::InlineObject {
                start: io.position, end: io.position.saturating_add(io.length), width: io.size[0], height: io.size[1], baseline: io.baseline, sideways: io.supports_sideways }).collect();
            let trim = trimming.as_ref().map(|t| layout::Trimming {
                granularity: t.granularity as u8, delimiter: t.delimiter, delimiter_count: t.delimiter_count, ellipsis: t.ellipsis, upright_sign: 0.0, sign: None });
            // Per-range format runs (item-3 slices A faces + B size): SetFontWeight/Style/Stretch/
            // FontFamilyName and SetFontSize. Flatten the ranges to the effective (family,weight,style,
            // stretch,size) at each UTF-16 position — base, then every covering range in corpus order
            // (later wins) — group contiguous positions that differ from the base, load each face and
            // carry each run's em, and hand the core their byte ranges. Underline/strikethrough are
            // Slice C (decoration-only, no shaping change). A run differing only in size keeps the base
            // face but a per-range em. All range sizes here are finite > 0 (the validation above rejects
            // the op otherwise), so the `Option<f32>` key compares cleanly.
            let utf16_to_byte = |p: u32| -> usize {
                let (mut cu, mut bi) = (0u32, text.len());
                for (b, ch) in text.char_indices() { if cu >= p { bi = b; break; } cu += ch.len_utf16() as u32; }
                if p == 0 { 0 } else { bi }
            };
            let u16len = text.encode_utf16().count() as u32;
            let base_key: (String, u32, u32, u32, Option<f32>) = (font.clone(), *weight, *style, *stretch, None);
            let mut keys = vec![base_key.clone(); u16len as usize];
            for rg in ranges {
                let end = rg.position.saturating_add(rg.length).min(u16len);
                for k in keys.iter_mut().take(end as usize).skip(rg.position.min(u16len) as usize) {
                    if let Some(f) = &rg.family { k.0 = f.clone(); }
                    if let Some(w) = rg.weight { k.1 = w; }
                    if let Some(s) = rg.style { k.2 = s; }
                    if let Some(st) = rg.stretch { k.3 = st; }
                    if let Some(sz) = rg.size { k.4 = Some(sz); }
                }
            }
            let mut face_runs: Vec<(usize, usize, Vec<u8>, u32, Option<f32>, u32, Vec<(u32, f32)>)> = Vec::new();
            let mut p = 0u32;
            while p < u16len {
                let mut q = p + 1;
                while q < u16len && keys[q as usize] == keys[p as usize] { q += 1; }
                let k = &keys[p as usize];
                if *k != base_key {
                    // A range whose family the cold collection lacks falls back to Segoe UI, as the base does.
                    if let Some((bytes, idx, sims)) = load_text_font_sims(&k.0, k.1, k.2, k.3) {
                        let (axes, _) = instance_axes(&bytes, idx, k.1, k.3);
                        face_runs.push((utf16_to_byte(p), utf16_to_byte(q), bytes, idx, k.4, sims, axes));
                    }
                }
                p = q;
            }
            let formats: Vec<d2d::text::FormatFace> = face_runs.iter().map(|(s, e, b, i, em, fs, ax)| (*s, *e, b.as_slice(), *i, *em, *fs, ax.as_slice())).collect();
            // Per-range underline / strikethrough (item-3 slice C): effective bool at each UTF-16
            // position (base off, then each covering range in corpus order — a later SetUnderline(false)
            // clears an earlier true), grouped into byte ranges for the decoration rects.
            let (mut ul, mut stk) = (vec![false; u16len as usize], vec![false; u16len as usize]);
            for rg in ranges {
                let end = rg.position.saturating_add(rg.length).min(u16len);
                for p in rg.position.min(u16len)..end {
                    if let Some(u) = rg.underline { ul[p as usize] = u; }
                    if let Some(s) = rg.strikethrough { stk[p as usize] = s; }
                }
            }
            let to_byte_ranges = |flags: &[bool]| -> Vec<(usize, usize)> {
                let mut out = Vec::new();
                let mut p = 0usize;
                while p < flags.len() {
                    if flags[p] {
                        let mut q = p + 1;
                        while q < flags.len() && flags[q] { q += 1; }
                        out.push((utf16_to_byte(p as u32), utf16_to_byte(q as u32)));
                        p = q;
                    } else { p += 1; }
                }
                out
            };
            let underline = to_byte_ranges(&ul);
            let strikethrough = to_byte_ranges(&stk);
            // A fallback face takes the format at that character: a range's weight, style and stretch (and
            // family, for the own family's other faces).
            let load = |f: &str, b: usize, ch: char| {
                // Past the text no range applies: the base format.
                let k = keys.get(text[..b].encode_utf16().count()).unwrap_or(&base_key);
                match f {
                    d2d::text::fallback::OWN_FAMILY => load_own_family(&k.0, k.1, k.2, k.3, ch),
                    _ => load_fallback(f, k.1, k.2, k.3, ch),
                }
            };
            let ok = if let Some(b) = r.solids.get(bid.as_str()) {
                dc.draw_text_fallback_box(text, &base_bytes, base_face, locale, &fmt, *measuring, o, *max_width, *max_height, b, &r.text_params, false, base_sims, &coords, &axes, &feats, trim, &inl, &formats, &underline, &strikethrough, Some(dc.get_dpi().0 / 96.0), load)
            } else if let Some(pb) = r.pbrushes.get(bid.as_str()) {
                dc.draw_text_fallback_box_brush(text, &base_bytes, base_face, locale, &fmt, *measuring, o, *max_width, *max_height, pb, &r.text_params, false, base_sims, &coords, &axes, &feats, trim, &inl, &formats, &underline, &strikethrough, Some(dc.get_dpi().0 / 96.0), load)
            } else {
                return Err(format!("undefined resource {bid:?}"));
            };
            if !ok {
                return Err("text layout rendering path not implemented".into());
            }
        }
        // Phase 2D-3 text: DrawText (layout → glyph runs → analysis/blend), the DrawGlyphRun
        // primitive, and the DWRITE_TEXT_METRICS query. Fonts load from ref/windows/Fonts; an absent
        // font or an unimplemented rendering path (ClearType/aliased/GDI/colour) skips the scene.
        Op::Text { text, font, size, rect, brush, weight, style, locale, antialias, measuring, options, overload, params } => {
            let rp = params.map_or(r.text_params, |[g, c, gc, l]| RenderingParams { gamma: g, enhanced_contrast: c, grayscale_contrast: gc, cleartype_level: l, ..RenderingParams::default() });
            let rp = RenderingParams { no_snap: options.is_some_and(|o| o & 1 != 0), ..rp }; // D2D1_DRAW_TEXT_OPTIONS_NO_SNAP
            // Native's DrawText builds its layout through `IDWriteFactory::CreateTextFormat`, which
            // rejects a fontSize that is not `> 0.0` (negatives, ±0.0, NaN, −inf) with a PER-OP
            // E_INVALIDARG that is NOT latched — EndDraw stays S_OK. Matches DirectWrite's
            // CreateTextFormat font-size gate (any value not `> 0.0` is rejected). Record the
            // per-op HR and draw nothing, matching fuzz buckets 2/3.
            if !(*size > 0.0) {
                r.op_hr.insert(i, fmt_hr(0x80070057u32 as i32));
                return Ok(());
            }
            // DrawText with system font fallback — the core's `draw_text_fallback` orchestrates it
            // (map each char to the base face or a covering candidate, exactly as MapCharacters does),
            // with the candidate families loaded on disk here. The FFI drives the same core method with
            // a system-collection loader, so the C-API output matches this byte-for-byte.
            let (base_bytes, base_face, base_sims) = load_text_font_sims(font, *weight, *style, 5).ok_or("font file not available")?;
            let (axes, coords) = instance_axes(&base_bytes, base_face, *weight, 5);
            dc.set_text_antialias_mode(*antialias);
            // The "layout" overload is DrawTextLayout of a CreateTextLayout(rect width, rect height) drawn at
            // the rect's top-left: the same box DrawText lays out, but natural measuring, and a negative or
            // NaN width/height fails CreateTextLayout with E_INVALIDARG and draws nothing (DrawText swallows
            // that failure).
            let layout_overload = overload.as_deref() == Some("layout");
            if layout_overload && [rect[2] - rect[0], rect[3] - rect[1]].iter().any(|v| !(*v >= 0.0)) {
                r.op_hr.insert(i, fmt_hr(0x80070057u32 as i32));
                return Ok(());
            }
            let measuring = if layout_overload { &0 } else { measuring };
            let fmt = TextFormat { em: *size, base_rtl: false, align: 0, wrap: 0, para: 0, flow: 0, line_spacing: (0, 0.0, 0.0) };
            let enable_color = options.is_some_and(|o| o & 4 != 0); // D2D1_DRAW_TEXT_OPTIONS_ENABLE_COLOR_FONT
            // D2D1_DRAW_TEXT_OPTIONS_CLIP: native pushes an axis-aligned clip of the layout box (the
            // normalized rect) with the current antialias mode around the draw (DrawTextLayoutHelper).
            let clip = options.is_some_and(|o| o & 2 != 0) && !d2d::text::draw_text_rect_rejected(*rect);
            let clip = clip.then(|| {
                let (o, w, h) = d2d::text::normalize_layout_rect(*rect);
                dc.push_text_clip([o[0], o[1], o[0] + w, o[1] + h])
            });
            let load = |f: &str, _: usize, ch: char| match f {
                d2d::text::fallback::OWN_FAMILY => load_own_family(font, *weight, *style, 5, ch),
                _ => load_fallback(f, *weight, *style, 5, ch),
            };
            let ok = if let Some(b) = r.solids.get(brush.as_str()) {
                dc.draw_text_fallback(text, &base_bytes, base_face, locale, &fmt, *measuring, *rect, b, &rp, enable_color, base_sims, &coords, &axes, &[], None, &[], &[], &[], &[], load)
            } else if let Some(pb) = r.pbrushes.get(brush.as_str()) {
                dc.draw_text_fallback_brush(text, &base_bytes, base_face, locale, &fmt, *measuring, *rect, pb, &rp, enable_color, base_sims, &coords, &axes, &[], None, &[], &[], &[], &[], load)
            } else {
                return Err(format!("undefined resource {brush:?}"));
            };
            if let Some(layer) = clip {
                dc.pop_text_clip(layer);
            }
            if params.is_some() {
                r.text_params = RenderingParams::default();
            }
            if !ok {
                return Err("text rendering path not implemented (cleartype/aliased/gdi/colour)".into());
            }
        }
        Op::GlyphRun { text, font, size, weight, style, baseline, brush, antialias, measuring, rendering_mode, grid_fit, simulations: _ } => {
            // d2d-ref draws the face GetFirstMatchingFont returns, with that face's own simulations; the
            // scene's `simulations` field never reaches the run.
            let (bytes, face, sims) = load_font_sims(font, *weight, *style, 5).ok_or("font file not available")?;
            let (_, coords) = instance_axes(&bytes, face, *weight, 5);
            let mut glyphs = resolve_glyphs(&bytes, text, *size, &coords).ok_or("glyph run font parse")?;
            // d2d-ref takes the advances from GetDesignGlyphMetrics on that face, which for a bold simulation
            // widens every inked glyph by round(upem/50) design units.
            if sims & 1 != 0 {
                let f = FontRef::from_index(&bytes, 0).map_err(|_| "glyph run font parse")?;
                let upem = f.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f32;
                let gm = f.glyph_metrics(Size::unscaled(), LocationRef::default());
                for g in glyphs.iter_mut().filter(|g| gm.bounds(GlyphId::new(g.id as u32)).is_some()) {
                    g.advance += (upem / 50.0).round() * *size / upem;
                }
            }
            dc.set_text_antialias_mode(*antialias);
            r.text_params = RenderingParams { rendering_mode: *rendering_mode, grid_fit: *grid_fit, ..Default::default() };
            let params = r.text_params;
            let run = TGlyphRun {
                font_data: &bytes,
                face_index: face,
                em_size: *size,
                glyphs: &glyphs,
                baseline_origin: *baseline,
                measuring_mode: *measuring,
                is_sideways: false,
                bidi_level: 0,
                // DrawGlyphRun draws at the exact origin — no baseline pixel-snapping (that is a
                // DrawText/layout feature). Native places the run on the sub-pixel grid directly.
                no_snap: true,
                simulations: sims,
                coords: &coords,
            };
            let ok = if let Some(b) = r.solids.get(brush.as_str()) {
                dc.draw_glyph_run(&run, b, &params)
            } else if let Some(pb) = r.pbrushes.get(brush.as_str()) {
                dc.draw_glyph_run_brush(&run, pb, &params)
            } else {
                return Err(format!("undefined resource {brush:?}"));
            };
            if !ok {
                return Err("glyph run rendering path not implemented (cleartype/aliased/gdi)".into());
            }
        }
        // Colour fonts pt2: DrawPaintGlyphRun with the inline COLRv1 font (bypasses layout).
        Op::ColorGlyphRun { font, text, em_size, baseline, palette, color, with_color_support: _ } => {
            let glyphs = resolve_glyphs(font, text, *em_size, &[]).ok_or("colr font parse")?;
            let b = dc.create_solid_color_brush(*color, Some(1.0));
            dc.set_text_antialias_mode(2); // grayscale
            let params = r.text_params;
            let run = TGlyphRun {
                font_data: font,
                face_index: 0,
                em_size: *em_size,
                glyphs: &glyphs,
                baseline_origin: *baseline,
                measuring_mode: 0,
                is_sideways: false,
                bidi_level: 0,
                no_snap: false,
                simulations: 0,
                coords: &[],
            };
            if !dc.draw_color_glyph_run(&run, &b, *palette, &params) {
                return Err("colour glyph run: unsupported paint type".into());
            }
        }
        // Colour bitmap glyphs: DrawColorBitmapGlyphRun with the inline CBDT/sbix font.
        Op::ColorBitmapGlyphRun { font, text, em_size, baseline, glyph_format } => {
            let glyphs = resolve_glyphs(font, text, *em_size, &[]).ok_or("bitmap font parse")?;
            let params = r.text_params;
            let run = TGlyphRun {
                font_data: font,
                face_index: 0,
                em_size: *em_size,
                glyphs: &glyphs,
                baseline_origin: *baseline,
                measuring_mode: 0,
                is_sideways: false,
                bidi_level: 0,
                no_snap: true,
                simulations: 0,
                coords: &[],
            };
            let formats = match glyph_format.as_str() {
                "jpeg" => 0x20,
                "tiff" => 0x40,
                "pbgra" => 0x80,
                _ => 0x10, // png
            };
            if !dc.draw_color_bitmap_glyph_run(&run, formats, &params) {
                return Err("colour bitmap glyph run: no strike".into());
            }
        }
        // OT-SVG glyphs: DrawSvgGlyphRun with the inline `SVG ` font.
        Op::SvgGlyphRun { font, text, em_size, baseline, color } => {
            let glyphs = resolve_glyphs(font, text, *em_size, &[]).ok_or("svg font parse")?;
            let params = r.text_params;
            let run = TGlyphRun {
                font_data: font,
                face_index: 0,
                em_size: *em_size,
                glyphs: &glyphs,
                baseline_origin: *baseline,
                measuring_mode: 0,
                is_sideways: false,
                bidi_level: 0,
                no_snap: true,
                simulations: 0,
                coords: &[],
            };
            if !dc.draw_svg_glyph_run(&run, *color, &params) {
                return Err("svg glyph run: unsupported".into());
            }
        }
        // GetColorBitmapGlyphImage per glyph, then draw the returned image under its glyph transform —
        // the single-glyph image form of ColorBitmapGlyphRun (must match it for s=1).
        Op::ColorBitmapGlyphImage { font, text, em_size, baseline, glyph_format } => {
            let glyphs = resolve_glyphs(font, text, *em_size, &[]).ok_or("bitmap font parse")?;
            let dpi = dc.get_dpi();
            let mut pen_x = baseline[0];
            for g in &glyphs {
                let (bmp, gt) = d2d::text::color::bitmap::color_bitmap_glyph_image(font, 0, g.id, *em_size, 0, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dpi, [pen_x, baseline[1]], false)
                    .ok_or("colour bitmap glyph image: no strike")?;
                pen_x += g.advance;
                // Draw the image under the returned transform, as the caller does with
                // SetTransform(glyphTransform) + DrawImage (source-over, LINEAR).
                let saved = dc.get_transform();
                dc.set_transform(gt);
                dc.draw_image_bitmap(&bmp, [0.0, 0.0], None, 1, 0);
                dc.set_transform(saved);
            }
            let _ = glyph_format;
        }
        // GetSvgGlyphImage: record the SVG glyph run into a command list, then DrawImage it — the
        // command-list form of SvgGlyphRun (must match it).
        Op::SvgGlyphImage { font, text, em_size, baseline, color } => {
            let glyphs = resolve_glyphs(font, text, *em_size, &[]).ok_or("svg font parse")?;
            let params = r.text_params;
            let run = TGlyphRun {
                font_data: font,
                face_index: 0,
                em_size: *em_size,
                glyphs: &glyphs,
                baseline_origin: *baseline,
                measuring_mode: 0,
                is_sideways: false,
                bidi_level: 0,
                no_snap: true,
                simulations: 0,
                coords: &[],
            };
            let list = dc.create_command_list();
            dc.record_into(&list);
            let ok = dc.draw_svg_glyph_run(&run, *color, &params);
            dc.stop_recording();
            if !ok {
                return Err("svg glyph image: unsupported".into());
            }
            list.close();
            dc.draw_image(&list, [0.0, 0.0], 0);
        }
        Op::TextMetrics { text, font, size } => {
            let (bytes, face) = load_font(font, 400, 0, 5).ok_or("font file not available")?;
            let f = ShapeFont::from_bytes(bytes, face).ok_or("font parse")?;
            let fmt = TextFormat { em: *size, base_rtl: false, align: 0, wrap: 0, para: 0, flow: 0, line_spacing: (0, 0.0, 0.0) };
            let m = d2d::text::layout::layout(&f, &fmt, text, f32::MAX, f32::MAX).metrics;
            r.queries.push(QueryRec {
                op: i,
                kind: "text_metrics".into(),
                got: serde_json::json!({ "width": m.width, "widthIncludingTrailingWhitespace": m.width_trailing, "height": m.height }),
            });
        }
        // --- ffirects (this agent): bounds / rect-mapping / copy ---
        Op::GlyphBounds { text, font, size, weight, style, baseline, antialias, measuring, rendering_mode, grid_fit } => {
            let (bytes, face) = load_font(font, *weight, *style, 5).ok_or("font file not available")?;
            let (_, coords) = instance_axes(&bytes, face, *weight, 5);
            let glyphs = resolve_glyphs(&bytes, text, *size, &coords).ok_or("glyph run font parse")?;
            dc.set_text_antialias_mode(*antialias);
            r.text_params = RenderingParams { rendering_mode: *rendering_mode, grid_fit: *grid_fit, ..Default::default() };
            let params = r.text_params;
            let run = TGlyphRun {
                font_data: &bytes,
                face_index: face,
                em_size: *size,
                glyphs: &glyphs,
                baseline_origin: *baseline,
                measuring_mode: *measuring,
                is_sideways: false,
                bidi_level: 0,
                no_snap: false,
                simulations: 0,
                coords: &coords,
            };
            let b = dc.glyph_run_world_bounds(&run, &params);
            r.queries.push(QueryRec { op: i, kind: "glyph_bounds".into(), got: serde_json::json!(b) });
        }
        Op::ImageBounds { image, world } => {
            let dip = image_bounds_dip(r, image)?;
            let b = if *world { dc.image_world_bounds(dip) } else { dc.image_local_bounds(dip) };
            if std::env::var_os("D2D_PRINT_QUERIES").is_some() { println!("QUERY {i} image_bounds {b:?}"); }
            r.queries.push(QueryRec { op: i, kind: "image_bounds".into(), got: serde_json::json!(b) });
        }
        Op::EffectRequiredInput { effect, output, inputs } => {
            let root = r.effects.get(effect.as_str()).ok_or("required-input on unknown effect")?.clone();
            let mut reqs = Vec::with_capacity(inputs.len());
            for ri in inputs {
                let e = r.effects.get(ri.effect.as_str()).ok_or("required-input references unknown effect")?.clone();
                reqs.push((e, ri.index));
            }
            let rects = root.required_input_rectangles(*output, &reqs, dc.get_dpi());
            r.queries.push(QueryRec { op: i, kind: "required_input".into(), got: serde_json::json!(rects) });
        }
        Op::EffectInvalidRects { effect, invalidate: _ } => {
            // A never-realized effect has its whole (infinite) output invalid; the CPU core keeps no
            // render cache to track sub-regions, matching native's one infinite rect.
            r.effects.get(effect.as_str()).ok_or("invalid-rects on unknown effect")?;
            let inf = [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::INFINITY];
            r.queries.push(QueryRec { op: i, kind: "invalid_rects".into(), got: serde_json::json!({"count": 1, "rects": [inf]}) });
        }
        Op::CopyFromBitmap { id, size, src, bgra, point, rect } => {
            let s = r.bitmaps.get(src.as_str()).ok_or_else(|| format!("undefined resource {src:?}"))?.clone();
            let (format, alpha) = if *bgra { (d2d::abi::DXGI_FORMAT_B8G8R8A8_UNORM, d2d::abi::D2D1_ALPHA_MODE_PREMULTIPLIED) } else { (s.format, s.alpha) };
            let bpp = d2d::bitmap::bytes_per_pixel(format).unwrap_or(4);
            let mut d = Bitmap { width: size[0], height: size[1], premul_bgra: vec![0; size[0] as usize * size[1] as usize * bpp], format, alpha, dpi: (96.0, 96.0) };
            if let Err(hr) = d.copy_from_bitmap(&s, *point, *rect) {
                r.op_hr.insert(i, fmt_hr(hr.0));
            }
            r.bitmaps.insert(id, d);
        }
        Op::Readback { rect } => {
            let (w, h) = match (rect, dc.target_bitmap()) {
                (Some(rc), _) => (rc[2].saturating_sub(rc[0]).max(1), rc[3].saturating_sub(rc[1]).max(1)),
                (None, Some(t)) => (t.desc.width, t.desc.height),
                (None, None) => (1, 1),
            };
            let (hr, bytes) = match dc.copy_from_target(*rect) {
                Ok(b) => ("S_OK".to_string(), b),
                Err(e) => (fmt_hr(e.0), Vec::new()),
            };
            r.queries.push(QueryRec { op: i, kind: "readback".into(), got: serde_json::json!({"hr": hr, "size": [w, h], "bytes": bytes}) });
        }
        Op::SetDpi { dpi } => dc.set_dpi(dpi[0], dpi[1]),
        Op::RenderingControls { precision, tile } => dc.set_rendering_controls(*precision, *tile),
        Op::CopyMemoryBitmap { id, size, rgba } => {
            // CopyFromMemory copies the (already premultiplied+swapped) bytes verbatim; the resulting
            // drawable bitmap equals a `from_straight_rgba` upload of the same straight RGBA.
            r.bitmaps.insert(id, Bitmap::from_straight_rgba(size[0], size[1], rgba));
        }
        // census1: resource/state getter readback. Reads the same out-parameters the harness records
        // from native, from the stored core objects. A getter whose value the core bakes away (e.g.
        // gradient-brush parameters) has no arm here, so that scene SKIPs honestly.
        Op::GetProp { kind, target } => {
            let t = target.as_deref().unwrap_or("");
            // census3 getter kinds (gradient-brush params, bitmap/image-brush modes, colour-context
            // profile, geometry sources, effect-property metadata, mesh patches, sprites, path
            // segment, ink segments, device caps) have no `d2d` core counterpart — like the
            // DirectWrite/WIC query kinds the core records nothing (blank pixels) and the drop-in is
            // checked against the captured native oracle by the decisive run.
            const CENSUS3: &[&str] = &[
                "grad_linear", "grad_radial", "brush_extend", "image_brush_extend", "bitmap_opts",
                "colorctx_data", "geom_src", "geom_group_src", "effect_props", "mesh", "sprites",
                "path_seg", "ink_more", "dc_caps",
                "bitmap_brush_set", "image_brush_set", "solid_set", "grad_linear_set", "grad_radial_set",
                "geom_sink_plural",
            ];
            if CENSUS3.contains(&kind.as_str()) {
                return Ok(());
            }
            let cap_u = |c: &Cap| match c { Cap::Flat => 0, Cap::Square => 1, Cap::Round => 2, Cap::Triangle => 3 };
            let got = match kind.as_str() {
                "dc_transform" => serde_json::json!(dc.get_transform()),
                "dc_antialias" => serde_json::json!(dc.get_antialias_mode()),
                "dc_text_antialias" => serde_json::json!(dc.get_text_antialias_mode()),
                "dc_blend" => serde_json::json!(dc.get_primitive_blend()),
                "dc_unit" => serde_json::json!(dc.get_unit_mode()),
                "dc_dpi" => { let (x, y) = dc.get_dpi(); serde_json::json!([x, y]) }
                "dc_tags" => { let (a, b) = dc.get_tags(); serde_json::json!([a, b]) }
                "histogram_output" => {
                    // HistogramOutput reads back only after the effect is drawn + flushed. Before the
                    // first draw native's GetValue returns E_INVALIDARG, latched to this op with no
                    // result; afterwards it returns the value cached at that draw (stale across later
                    // property changes until drawn again). Mirror both.
                    match r.effects.get(t).ok_or("getprop: histogram")?.histogram_cached() {
                        Some(h) => serde_json::json!(h),
                        None => {
                            r.op_hr.insert(i, fmt_hr(0x80070057u32 as i32));
                            return Ok(());
                        }
                    }
                }
                "brush_color" => serde_json::json!(r.solids.get(t).ok_or("getprop: solid")?.color()),
                "brush_opacity" => serde_json::json!(r.solids.get(t).ok_or("getprop: solid")?.opacity()),
                "stroke" => {
                    let s = r.styles.get(t).ok_or("getprop: stroke")?;
                    serde_json::json!({
                        "start_cap": cap_u(&s.start_cap), "end_cap": cap_u(&s.end_cap),
                        "dash_cap": cap_u(&s.dash_cap),
                        "line_join": match s.join { Join::Miter => 0, Join::Bevel => 1, Join::Round => 2, Join::MiterOrBevel => 3 },
                        "miter": s.miter_limit,
                        "dash_style": if s.dashes.is_empty() { 0 } else { 5 },
                        "dash_offset": s.dash_offset, "dash_count": s.dashes.len() as u32, "dashes": s.dashes,
                        "transform_type": match s.transform_type { TransformType::Normal => 0, TransformType::Fixed => 1, TransformType::Hairline => 2 },
                    })
                }
                "geom_rect" => match r.geoms.get(t).ok_or("getprop: geom")? {
                    Geometry::Rect(rect) => serde_json::json!(rect),
                    _ => return Err("geom_rect on non-rectangle".into()),
                },
                "geom_rounded" => match r.geoms.get(t).ok_or("getprop: geom")? {
                    Geometry::Rounded { rect, radius } => serde_json::json!([rect[0], rect[1], rect[2], rect[3], radius[0], radius[1]]),
                    _ => return Err("geom_rounded on non-rounded".into()),
                },
                "geom_ellipse" => match r.geoms.get(t).ok_or("getprop: geom")? {
                    Geometry::Ellipse { center, radius } => serde_json::json!([center[0], center[1], radius[0], radius[1]]),
                    _ => return Err("geom_ellipse on non-ellipse".into()),
                },
                "geom_group" => match r.geoms.get(t).ok_or("getprop: geom")? {
                    Geometry::Group { fill, children } => serde_json::json!([
                        match fill { geometry::FillMode::Alternate => 0, geometry::FillMode::Winding => 1 },
                        children.len() as u32]),
                    _ => return Err("geom_group on non-group".into()),
                },
                "bitmap" => {
                    let b = r.bitmaps.get(t).ok_or("getprop: bitmap")?;
                    serde_json::json!({
                        "size": [b.width as f32 * 96.0 / b.dpi.0, b.height as f32 * 96.0 / b.dpi.1],
                        "pixel_size": [b.width, b.height], "dpi": [b.dpi.0, b.dpi.1],
                        "format": b.format.0, "alpha": b.alpha.0,
                    })
                }
                "colorctx_type" => serde_json::json!(r.color_contexts.get(t).ok_or("getprop: colorctx")?.color_context_type().0),
                "ink" => {
                    let k = r.inks.get(t).ok_or("getprop: ink")?;
                    serde_json::json!({"start": k.start, "segments": k.segments.len() as u32})
                }
                "ink_style" => {
                    let s = r.ink_styles.get(t).ok_or("getprop: ink_style")?;
                    serde_json::json!({"nib_shape": s.nib_shape, "nib_transform": s.nib_transform})
                }
                other => return Err(format!("unknown getprop kind {other}")),
            };
            r.queries.push(QueryRec { op: i, kind: format!("getprop:{kind}"), got });
        }
        // census2 DirectWrite / WIC getter readback (IDWriteTextFormat/RenderingParams/FontFace,
        // IWICBitmapDecoderInfo/Bitmap/Lock/FormatConverter): the `d2d` core models only the layout
        // inputs it reproduces, not these COM objects, so — like `TextLayout` — the core arm records
        // nothing (draws blank, matching the oracle's blank pixels); the getter values are the native
        // ground truth the dwrite-ffi / wic-ffi decisive runs verify the drop-in DLLs against.
        Op::DwriteQuery { .. } | Op::WicQuery { .. } => {}
        Op::DrawMetafile { file, bytes } => {
            // Core DrawGdiMetafile: parse the EMF(+) (inline `bytes`, else the cwd-relative file) and
            // play it at its natural bounds, matching the d2d-ffi facade's DrawGdiMetafile.
            let data = if !bytes.is_empty() { bytes.clone() } else { std::fs::read(file).map_err(|e| e.to_string())? };
            let mf = d2d::metafile::GdiMetafile::parse(data).ok_or("invalid EMF metafile")?;
            let b = mf.bounds();
            dc.draw_gdi_metafile(&mf, b, None);
        }
        _ => return Err("op not implemented".into()),
    }
    Ok(())
}

/// Drive the scene through the public API the way the harness drives native: BeginDraw, an implicit
/// transparent Clear, then `SetTags(i,0)` + the op for each op, then EndDraw. `Err` ⇒ skip.
fn drive(device: &Device, factory: &Factory, scene: &Scene) -> Result<Run, String> {
    let [w, h] = scene.size;
    let mut dc = device.create_device_context();
    dc.set_dpi(scene.dpi, scene.dpi);
    // A high-precision / HDR target (R16G16B16A16_FLOAT=10, R32G32B32A32_FLOAT=2) uses the float
    // pipeline; everything else is the BGRA8 target. Absent format ⇒ BGRA8 (v1/v2 scenes unchanged).
    let target = match scene.format {
        Some(f) if d2d::float::is_highp(d2d::abi::DXGI_FORMAT(f as i32)) => d2d::float::float_target(
            w,
            h,
            scene.dpi,
            d2d::abi::DXGI_FORMAT(f as i32),
            d2d::abi::D2D1_ALPHA_MODE(scene.alpha.unwrap_or(1) as i32),
        ),
        // R8G8B8A8_UNORM is the same 8-bit pipeline as BGRA8, only RGBA byte order; the Target
        // permutes channels from its format, so just stamp the format onto a 4-bpp buffer.
        Some(28) => {
            let mut t = bitmap_target(w, h, scene.dpi);
            t.desc.format = d2d::abi::DXGI_FORMAT_R8G8B8A8_UNORM;
            t
        }
        _ => bitmap_target(w, h, scene.dpi),
    };
    dc.set_target_bitmap(target);
    // SetRenderingControls bufferPrecision: the default intermediate precision for the effect graph
    // (absent ⇒ 8bpc, so existing scenes are unchanged).
    // Absent ⇒ the native default, UNKNOWN (0): no SetRenderingControls, as the harness does.
    dc.set_buffer_precision(scene.buffer_precision.unwrap_or(0));
    dc.begin_draw();
    dc.clear(Some([0.0; 4]));

    let mut r = Res::default();
    let mut ops = Vec::with_capacity(scene.ops.len());
    for (i, op) in scene.ops.iter().enumerate() {
        dc.set_tags(i as u64, 0);
        // Every in-scope op is a void or infallible call ⇒ "S_OK"; a latched draw error surfaces at
        // EndDraw, and a scene that reaches a partially-wired op (e.g. a non-SOURCE_OVER brush fill)
        // will diverge there — honestly, as a FAIL.
        // An op whose brush was never created (its image brush's command list failed Close, …) is
        // native's harness "undefined resource": recorded on the op, and the scene goes on (fuzz 7770).
        match exec_op(&mut dc, factory, i, op, &mut r) {
            Err(e) if e.starts_with("undefined resource") => {
                r.op_hr.insert(i, format!("err: {e}"));
            }
            res => res?,
        }
        let (tag1, tag2) = dc.get_tags();
        let hr = r.op_hr.get(&i).cloned().unwrap_or_else(|| "S_OK".into());
        ops.push(OpRec { hr, tag1, tag2 });
    }
    let (hr, tag1, tag2) = dc.end_draw();
    let pixels = dc.take_target().unwrap().data;
    // A ColorManagement transform with a SIMPLE-profile context: native fails realization at EndDraw
    // with E_INVALIDARG latched to the draw_image op (cms2-04/05); mirror that end result.
    let end = match r.cm_fail_tag {
        Some(tag) => OracleEnd { hr: fmt_hr(0x80070057u32 as i32), tag1: tag, tag2: 0 },
        None => OracleEnd { hr: fmt_hr(hr.0), tag1, tag2 },
    };
    Ok(Run { pixels, ops, end, queries: r.queries })
}

/// Compare one geometry query result with the oracle, mirroring the 1C geometry test's tolerance:
/// exact bits for length/bounds, ≤1 ULP for area (native accumulation order), exact bool for
/// contains_point.
/// Compare two JSON arrays of `[l,t,r,b]` rects, exact f32 bits (ffirects rect-list queries).
fn compare_rect_list(g: &[serde_json::Value], w: &[serde_json::Value], label: &str) -> Option<String> {
    if g.len() != w.len() {
        return Some(format!("{label} count got {} want {}", g.len(), w.len()));
    }
    g.iter().zip(w).enumerate().find_map(|(ri, (gg, ww))| {
        let (ga, wa) = (qf32x4(gg), qf32x4(ww));
        (0..4).find_map(|k| {
            let (a, b) = (ga[k], wa[k]);
            (a.to_bits() != b.to_bits()).then(|| format!("{label}[{ri}][{k}] got {a} want {b}"))
        })
    })
}

/// A recorded f32. JSON has no non-finite numbers, so both recorders write ±∞ and NaN as `null`; any
/// `null` (or missing value) reads back as one NaN, and two of them compare equal.
fn qf32(v: &serde_json::Value) -> f32 {
    v.as_f64().map_or(f32::NAN, |f| f as f32)
}

/// A recorded 4-float result ([`qf32`] per element); a `null` result is four NaNs.
fn qf32x4(v: &serde_json::Value) -> [f32; 4] {
    core::array::from_fn(|k| qf32(&v[k]))
}

fn compare_query(q: &QueryRec, want: &serde_json::Value) -> Option<String> {
    let ulps = |a: f32, b: f32| (a.to_bits() as i64 - b.to_bits() as i64).abs();
    match q.kind.as_str() {
        "length" | "max_scale" => {
            let (g, w) = (qf32(&q.got), qf32(want));
            (g.to_bits() != w.to_bits()).then(|| format!("{} got {g} want {w}", q.kind))
        }
        "area" => {
            let (g, w) = (qf32(&q.got), qf32(want));
            (ulps(g, w) > 1).then(|| format!("area got {g} want {w} ({} ULP)", ulps(g, w)))
        }
        // Triangle-for-triangle, exact f32 bits (native `AddTriangles` stream).
        "tessellate" => {
            let bits = |v: &serde_json::Value| {
                let mut o = Vec::new();
                fn rec(v: &serde_json::Value, o: &mut Vec<u32>) {
                    match v {
                        serde_json::Value::Array(a) => a.iter().for_each(|x| rec(x, o)),
                        serde_json::Value::Number(n) => o.push((n.as_f64().unwrap() as f32).to_bits()),
                        _ => {}
                    }
                }
                rec(v.get("triangles").unwrap_or(v), &mut o);
                o
            };
            let tris = |v: &serde_json::Value| v.get("triangles").and_then(|t| t.as_array()).cloned().unwrap_or_default();
            let (gt, wt) = (tris(&q.got), tris(want));
            if gt.len() != wt.len() {
                return Some(format!("tessellate got {} triangles want {}", gt.len(), wt.len()));
            }
            (0..gt.len()).find_map(|k| (bits(&gt[k]) != bits(&wt[k])).then(|| format!("tessellate tri{k} got {} want {}", gt[k], wt[k])))
        }
        // Both are 4-element f32 results compared bit-exact: `bounds` = the rect; `point_at_length`
        // = ComputePointAtLength's point (x,y) + unit tangent (tx,ty).
        "bounds" | "point_at_length" | "widened_bounds" => {
            let (g, w) = (qf32x4(&q.got), qf32x4(want));
            (0..4).find_map(|k| {
                let (a, b) = (g[k], w[k]);
                (a.to_bits() != b.to_bits()).then(|| format!("{}[{k}] got {a} want {b}", q.kind))
            })
        }
        "contains_point" | "stroke_contains" => {
            let (g, w) = (q.got.as_bool().unwrap(), want.as_bool().unwrap());
            (g != w).then(|| format!("{} got {g} want {w}", q.kind))
        }
        "text_metrics" => ["width", "widthIncludingTrailingWhitespace", "height"].iter().find_map(|k| {
            let (a, b) = (qf32(&q.got[k]), qf32(&want[k]));
            (a.to_bits() != b.to_bits()).then(|| format!("{k} got {a} want {b}"))
        }),
        "compare" => {
            let (g, w) = (q.got.as_i64().unwrap(), want.as_i64().unwrap());
            (g != w).then(|| format!("compare relation got {g} want {w}"))
        }
        // geometry_op:<operation> result: exact figure/segment counts and exact-bit bounds.
        k if k.starts_with("geometry_op:") => {
            let gi = |o: &serde_json::Value, key: &str| o.get(key).and_then(|v| v.as_u64());
            // A non-finite bound (±∞/NaN, e.g. comb-31's empty-fill sentinel) serializes to JSON
            // `null`, losing its sign; map it to one sentinel so both sides compare equal.
            let gf = |o: &serde_json::Value, k: usize| {
                o.get("bounds")
                    .and_then(|v| v.as_array())
                    .map(|a| a[k].as_f64().map_or(0x7fc0_0000, |f| (f as f32).to_bits()))
            };
            if gi(&q.got, "figures") != gi(want, "figures") {
                Some(format!("figures got {:?} want {:?}", gi(&q.got, "figures"), gi(want, "figures")))
            } else if gi(&q.got, "segments") != gi(want, "segments") {
                Some(format!("segments got {:?} want {:?}", gi(&q.got, "segments"), gi(want, "segments")))
            } else if let Some(d) = (0..4).find_map(|k| {
                let (a, b) = (gf(&q.got, k), gf(want, k));
                (a != b).then(|| format!("bounds[{k}] got {a:?} want {b:?}"))
            }) {
                Some(d)
            } else {
                compare_stream(q.got.get("stream"), want.get("stream"))
            }
        }
        // SVG DOM queries record strings / counts / HRESULTs / POD-word arrays — all exact JSON.
        k if k.starts_with("svg_") => (q.got != *want).then(|| format!("{k} got {} want {}", q.got, want)),
        // census1 getter readback: structural JSON, every number compared as exact f32 bits.
        k if k.starts_with("getprop:") => cmp_json_f32(&q.got, want).map(|d| format!("{k} {d}")),
        // ffirects: a single rect [l,t,r,b], exact f32 bits.
        "glyph_bounds" | "image_bounds" => {
            let (g, w) = (qf32x4(&q.got), qf32x4(want));
            (0..4).find_map(|k| {
                let (a, b) = (g[k], w[k]);
                (a.to_bits() != b.to_bits()).then(|| format!("{}[{k}] got {a} want {b}", q.kind))
            })
        }
        // ffirects: a list of rects, exact f32 bits.
        "required_input" => compare_rect_list(q.got.as_array().unwrap(), want.as_array().unwrap(), "required_input"),
        // ffirects: {count, rects:[...]} — exact JSON (an infinite rect serialises to `null` on both
        // sides; finite f32 values compare exactly as serde Numbers).
        "invalid_rects" => (q.got != *want).then(|| format!("invalid_rects got {} want {}", q.got, want)),
        // r24 mid-frame CopyFromRenderTarget + Map: the HRESULT and size, then the bytes exactly.
        "readback" => {
            let (gh, wh) = (q.got.get("hr"), want.get("hr"));
            let bytes = |v: &serde_json::Value| v.get("bytes").and_then(|b| b.as_array()).map(|a| a.iter().map(|x| x.as_i64().unwrap_or(-1)).collect::<Vec<_>>()).unwrap_or_default();
            let (gb, wb) = (bytes(&q.got), bytes(want));
            if gh != wh || q.got.get("size") != want.get("size") {
                Some(format!("readback got {:?} {:?} want {:?} {:?}", gh, q.got.get("size"), wh, want.get("size")))
            } else if gb.len() != wb.len() {
                Some(format!("readback bytes {} vs {}", gb.len(), wb.len()))
            } else {
                let d: Vec<i64> = gb.iter().zip(&wb).map(|(a, b)| (a - b).abs()).filter(|&x| x > 0).collect();
                (!d.is_empty()).then(|| format!("readback bytes n={} maxd={}", d.len(), d.iter().max().unwrap()))
            }
        }
        other => Some(format!("unknown query kind {other}")),
    }
}

// Recursive JSON compare for census1 getter readback: numbers compared as exact f32 bits (native
// serialises f32 out-parameters as f64, so both sides must round-trip to identical f32), structure
// and bool/string/null exact.
fn cmp_json_f32(g: &serde_json::Value, w: &serde_json::Value) -> Option<String> {
    use serde_json::Value::{Array, Number, Object};
    match (g, w) {
        (Number(a), Number(b)) => {
            let (x, y) = (a.as_f64().unwrap() as f32, b.as_f64().unwrap() as f32);
            (x.to_bits() != y.to_bits()).then(|| format!("got {x} want {y}"))
        }
        (Array(a), Array(b)) => {
            if a.len() != b.len() {
                return Some(format!("array len {} vs {}", a.len(), b.len()));
            }
            a.iter().zip(b).find_map(|(x, y)| cmp_json_f32(x, y))
        }
        (Object(a), Object(b)) => {
            if a.len() != b.len() {
                return Some(format!("object keys {} vs {}", a.len(), b.len()));
            }
            a.iter().find_map(|(k, x)| match b.get(k) {
                Some(y) => cmp_json_f32(x, y).map(|d| format!("{k}: {d}")),
                None => Some(format!("missing key {k}")),
            })
        }
        _ => (g != w).then(|| format!("got {g} want {w}")),
    }
}

// Point-by-point diff of an operation's output stream vs the native RecordingSink capture: fill
// mode, figure count, each figure's start/begin/end, and every segment's type and control points
// (exact f32 bits). Pinpoints exactly where output topology/geometry diverges.
fn compare_stream(got: Option<&serde_json::Value>, want: Option<&serde_json::Value>) -> Option<String> {
    let (g, w) = match (got, want) {
        (Some(g), Some(w)) if !w.is_null() => (g, w),
        _ => return None, // oracle predates the recorder; counts/bounds already checked
    };
    let gi = |o: &serde_json::Value, k: &str| o.get(k).and_then(|v| v.as_i64());
    if gi(g, "fill_mode") != gi(w, "fill_mode") {
        return Some(format!("stream fill_mode got {:?} want {:?}", gi(g, "fill_mode"), gi(w, "fill_mode")));
    }
    let (gf, wf) = (g["figures"].as_array()?, w["figures"].as_array()?);
    if gf.len() != wf.len() {
        return Some(format!("stream figure count got {} want {}", gf.len(), wf.len()));
    }
    // All floats compared as exact f32 bits; nested point arrays are flattened.
    fn bits(v: &serde_json::Value) -> Vec<u32> {
        let mut o = Vec::new();
        fn rec(v: &serde_json::Value, o: &mut Vec<u32>) {
            match v {
                serde_json::Value::Array(a) => a.iter().for_each(|x| rec(x, o)),
                serde_json::Value::Number(n) => o.push((n.as_f64().unwrap() as f32).to_bits()),
                _ => {}
            }
        }
        rec(v, &mut o);
        o
    }
    for (fi, (a, b)) in gf.iter().zip(wf).enumerate() {
        if bits(&a["start"]) != bits(&b["start"]) {
            return Some(format!("fig{fi} start got {} want {}", a["start"], b["start"]));
        }
        for k in ["begin", "end"] {
            if gi(a, k) != gi(b, k) {
                return Some(format!("fig{fi} {k} got {:?} want {:?}", gi(a, k), gi(b, k)));
            }
        }
        let (gs, ws) = (a["segments"].as_array()?, b["segments"].as_array()?);
        if gs.len() != ws.len() {
            return Some(format!("fig{fi} segment count got {} want {}", gs.len(), ws.len()));
        }
        for (si, (s, t)) in gs.iter().zip(ws).enumerate() {
            if s.get("t") != t.get("t") {
                return Some(format!("fig{fi} seg{si} type got {:?} want {:?}", s.get("t"), t.get("t")));
            }
            if bits(&s["p"]) != bits(&t["p"]) {
                return Some(format!("fig{fi} seg{si} points got {} want {}", s["p"], t["p"]));
            }
        }
    }
    None
}

/// Compare a run with the oracle; `Err` lists the first mismatch (pixel, metadata, or query).
fn compare(run: &Run, oracle: &Oracle, width: u32, bgra: &[u8]) -> Result<(), String> {
    if run.ops.len() != oracle.ops.len() {
        return Err(format!("{} ops vs oracle {}", run.ops.len(), oracle.ops.len()));
    }
    for (i, (g, w)) in run.ops.iter().zip(&oracle.ops).enumerate() {
        if g.hr != w.hr || g.tag1 != w.tag1 || g.tag2 != w.tag2 {
            return Err(format!("op {i}: got ({},{},{}) want ({},{},{})", g.hr, g.tag1, g.tag2, w.hr, w.tag1, w.tag2));
        }
    }
    if run.end.hr != oracle.end_draw.hr || run.end.tag1 != oracle.end_draw.tag1 || run.end.tag2 != oracle.end_draw.tag2 {
        return Err(format!(
            "end_draw: got ({},{},{}) want ({},{},{})",
            run.end.hr, run.end.tag1, run.end.tag2, oracle.end_draw.hr, oracle.end_draw.tag1, oracle.end_draw.tag2
        ));
    }
    let by_op: HashMap<usize, &OracleQuery> = oracle.queries.iter().map(|q| (q.op, q)).collect();
    for q in &run.queries {
        let want = by_op.get(&q.op).ok_or_else(|| format!("op {}: query result missing from oracle", q.op))?;
        if let Some(diff) = compare_query(q, &want.result) {
            return Err(format!("op {}: {diff}", q.op));
        }
    }
    // An empty oracle bgra means EndDraw failed and native wrote no pixels: the metadata/end_draw
    // check above is the whole comparison (cms2 simple-profile failures).
    if bgra.is_empty() {
        return Ok(());
    }
    if run.pixels.len() != bgra.len() {
        return Err(format!("pixel length {} vs oracle {}", run.pixels.len(), bgra.len()));
    }
    if std::env::var_os("D2D_FULLDIFF").is_some() {
        for (i, (a, b)) in run.pixels.as_chunks::<4>().0.iter().zip(bgra.as_chunks::<4>().0).enumerate() {
            if a != b {
                let p = i as u32;
                eprintln!("DIFF ({},{}) got {} want {}", p % width, p / width, a[0], b[0]);
            }
        }
    }
    let (mut count, mut maxd, mut maxp) = (0u32, 0i32, (0u32, 0u32, [0u8; 4], [0u8; 4]));
    for (i, (a, b)) in run.pixels.as_chunks::<4>().0.iter().zip(bgra.as_chunks::<4>().0).enumerate() {
        if a != b {
            count += 1;
            let d = (0..4).map(|c| (a[c] as i32 - b[c] as i32).abs()).max().unwrap();
            if d > maxd { maxd = d; maxp = (i as u32 % width, i as u32 / width, *a, *b); }
        }
    }
    if count > 0 {
        return Err(format!("n={count} maxd={maxd} @({},{}) got {:?} want {:?}", maxp.0, maxp.1, maxp.2, maxp.3));
    }
    Ok(())
}

// --- cross-platform data source (phase 4A) -----------------------------------------------------
// The driver above is pure compute and runs identically on every target. Only *where the corpus
// comes from* and *which test harness runs it* differ: native reads the repo files; wasm32 (headless
// browser, Node, and the web/ page) has no filesystem, so the same files are baked in at compile time
// via `embed/corpus.rs` (regenerate with scripts/gen-wasm-embed.sh). `report()` is shared by both and the
// web/ runner, so there is exactly one copy of the suite.

/// One oracle scene: the scene JSON, the oracle result JSON, and the raw BGRA pixels.
struct Case {
    name: String,
    scene: String,
    oracle: String,
    bgra: Vec<u8>,
}

#[cfg(not(target_arch = "wasm32"))]
fn cases() -> Vec<Case> {
    // Scenes + oracles live in the conformance suite (the conformance/suite submodule); absent ⇒ no
    // cases, so every scene skips and a public clone passes.
    let Some(suite) = d2d_reference::suite_dir() else { return Vec::new() };
    let oracle = suite.join("oracle/win11-26100");
    let scenes = suite.join("scenes");
    let mut names: Vec<String> = std::fs::read_dir(&oracle)
        .expect("oracle dir")
        .filter_map(|e| {
            let p = e.ok()?.path();
            let stem = p.file_stem()?.to_str()?.to_owned();
            // shader-* scenes belong to shader-ref (their own format, checked by d2d-shader's tests).
            (p.extension()? == "json" && stem != "environment" && stem != "failures" && !stem.starts_with("shader"))
                .then_some(stem)
        })
        .collect();
    names.sort();
    names
        .into_iter()
        .filter_map(|name| {
            Some(Case {
                scene: std::fs::read_to_string(scenes.join(format!("{name}.json"))).ok()?,
                oracle: std::fs::read_to_string(oracle.join(format!("{name}.json"))).ok()?,
                // A scene whose EndDraw fails writes no pixels (e.g. cms2 simple-profile); the oracle
                // JSON still carries the HRESULT, so keep the case with an empty bgra.
                bgra: std::fs::read(oracle.join(format!("{name}.bgra"))).unwrap_or_default(),
                name,
            })
        })
        .collect()
}

#[cfg(target_arch = "wasm32")]
fn cases() -> Vec<Case> {
    const EMBEDDED: &[(&str, &str, &str, &[u8])] = include!("embed/corpus.rs");
    EMBEDDED
        .iter()
        .map(|&(name, scene, oracle, bgra)| Case {
            name: name.to_owned(),
            scene: scene.to_owned(),
            oracle: oracle.to_owned(),
            bgra: bgra.to_vec(),
        })
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn passing_text() -> String {
    d2d_reference::suite_dir()
        .and_then(|s| std::fs::read_to_string(s.join("passing.txt")).ok())
        .unwrap_or_default()
}
#[cfg(target_arch = "wasm32")]
fn passing_text() -> String {
    // Baked from the suite by scripts/gen-wasm-embed.sh; committed empty (public clone ⇒ no ratchet).
    include_str!("embed/passing.txt").to_owned()
}

/// Outcome of running the whole corpus, independent of how it was loaded or displayed.
/// `pub(crate)` so the web/ runner (which includes this file as a module) can read it; several
/// fields are read only there, so allow dead code in the native-test build.
#[allow(dead_code)]
pub(crate) struct Report {
    pub(crate) pass: u32,
    pub(crate) skip: u32,
    pub(crate) fail: u32,
    pub(crate) fails: Vec<String>,
    /// `passing.txt` scenes that now SKIP/FAIL, with the reason — a regression.
    pub(crate) regressed: Vec<String>,
    /// `passing.txt` scenes not seen as a PASS this run.
    pub(crate) missing: Vec<String>,
    /// Hash over every *fully rendered* scene's name + complete pixel buffer, in scene order.
    /// Equal on two platforms ⟺ the CPU renderer produced bit-identical output for the whole
    /// corpus — a full-output cross-platform parity check, not just matching PASS/FAIL verdicts.
    pub(crate) digest: u64,
}

/// Drive every scene through the public API and compare it to the oracle. Pure and platform-free:
/// the native test, the wasm32 test, and the web/ page all call this, so results are identical by
/// construction — any per-platform divergence shows up as a scene flipping PASS→FAIL here.
pub(crate) fn report() -> Report {
    let passing: HashSet<String> = passing_text()
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(String::from)
        .collect();
    let (mut pass, mut skip, mut fail) = (0u32, 0u32, 0u32);
    let mut fails: Vec<String> = Vec::new();
    let mut regressed: Vec<String> = Vec::new();
    let mut passed: HashSet<String> = HashSet::new();
    // Deterministic across targets (byte-defined SipHash); never committed, so version drift is moot.
    let mut hasher = std::hash::DefaultHasher::new();

    for Case { name, scene, oracle, bgra } in cases() {
        // Diagnostic: D2D_ONLY=<name> runs just that scene (fast per-scene debugging).
        if let Some(only) = std::env::var_os("D2D_ONLY") {
            if name != only.to_string_lossy() {
                continue;
            }
        }
        let scene: Scene = match serde_json::from_str(&scene) {
            Ok(s) => s,
            Err(e) => {
                println!("SKIP {name}: no scene ({e})");
                skip += 1;
                continue;
            }
        };
        // A fresh factory per scene, as the oracle captures each scene in its own process: the
        // custom-effect registry is per-factory, so scenes that reuse a CLSID can't leak into each
        // other (`effects/custom.rs`). Factory/device identity is otherwise irrelevant to the output.
        let factory = Factory::new(false);
        let device = factory.create_device();
        // D2D_OURPTS=<scene>: enable the per-glyph hinted-point dump (text::hinted_contours) only for
        // that one scene, so a scene render can be diffed against the native contour capture without
        // the dump firing for every other text scene.
        // Env mutation is unsupported on wasm32 (panics), and the browser conformance runner compiles
        // this file, so the per-scene diagnostic gates are native-only. The wasm parity runner never
        // sets these vars, so gating them out changes no behaviour there.
        #[cfg(not(target_arch = "wasm32"))]
        unsafe {
            if std::env::var("D2D_OURPTS").ok().as_deref() == Some(name.as_str()) {
                std::env::set_var("D2D_OURPTS_GO", "1");
            } else {
                std::env::remove_var("D2D_OURPTS_GO");
            }
            // D2D_GRAYBITS=<scene> enables the grayscale 1-bit grid dump (raster::coverage_box) for
            // just that scene, so it can be diffed against the native bitmap-fill capture.
            if std::env::var("D2D_GRAYBITS").ok().as_deref() == Some(name.as_str()) {
                std::env::set_var("D2D_GRAYBITS_GO", "1");
            } else {
                std::env::remove_var("D2D_GRAYBITS_GO");
            }
            // D2D_CROSS=<scene> enables the scan-crossing dump (raster::scan_bitmap) for just that scene.
            if std::env::var("D2D_CROSS").ok().as_deref() == Some(name.as_str()) {
                std::env::set_var("D2D_CROSS_GO", "1");
            } else {
                std::env::remove_var("D2D_CROSS_GO");
            }
        }
        let run = match drive(&device, &factory, &scene) {
            Ok(r) => r,
            Err(reason) => {
                println!("SKIP {name}: {reason}");
                skip += 1;
                if passing.contains(&name) {
                    regressed.push(format!("{name} (now SKIP: {reason})"));
                }
                continue;
            }
        };
        if std::env::var("D2D_SCENE_DUMP").is_ok_and(|v| v == name) {
            let d = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
            let _ = std::fs::write(format!("{d}/dump-our-{name}.bgra"), &run.pixels);
            let _ = std::fs::write(format!("{d}/dump-nat-{name}.bgra"), &bgra);
        }
        std::hash::Hasher::write(&mut hasher, name.as_bytes());
        std::hash::Hasher::write(&mut hasher, &run.pixels);
        // FFI-vs-core gate support: dump every driven scene's core pixels so the Wine FFI runner can
        // diff the C-API output against the core (any difference there is a C-API plumbing bug).
        if let Some(dir) = std::env::var_os("D2D_CORE_DUMP") {
            let p = std::path::Path::new(&dir).join(format!("{name}.core.bgra"));
            let _ = std::fs::write(p, &run.pixels);
        }
        let oracle: Oracle = serde_json::from_str(&oracle).unwrap();
        match compare(&run, &oracle, scene.size[0], &bgra) {
            Ok(()) => {
                println!("PASS {name}");
                passed.insert(name);
                pass += 1;
            }
            Err(diff) => {
                println!("FAIL {name}: {diff}");
                // Merge-gate severity for failing (not-yet-exact) scenes: the ratchet only guards
                // passing.txt, so a tracked FAIL that gets WORSE (e.g. ±1 → all-black) is invisible.
                // Under D2D_DIFF_STATS, emit a per-scene line the gate can baseline and compare:
                // mismatched BGRA bytes vs the oracle, and the max absolute channel delta. Additive,
                // off by default (every existing run is byte-identical without the env var).
                if std::env::var_os("D2D_DIFF_STATS").is_some() {
                    let n = run.pixels.len().min(bgra.len());
                    let (mut bytes, mut maxd, mut maxi) = (0u64, 0i32, 0usize);
                    for i in 0..n {
                        let d = (run.pixels[i] as i32 - bgra[i] as i32).abs();
                        if d != 0 {
                            bytes += 1;
                            if d > maxd { maxd = d; maxi = i; }
                        }
                    }
                    // A wrong-size render counts its surplus/missing bytes as fully mismatched.
                    bytes += (run.pixels.len() as i64 - bgra.len() as i64).unsigned_abs();
                    let w = scene.size[0] as usize;
                    let (px, py, ch) = ((maxi / 4) % w, (maxi / 4) / w, maxi % 4);
                    println!("DIFF {name} {bytes} {maxd} @({px},{py})ch{ch} got{} want{}", run.pixels.get(maxi).copied().unwrap_or(0), bgra.get(maxi).copied().unwrap_or(0));
                    // D2D_DIFF_SCENE=<name>: list per-pixel diffs for that one scene (additive; the DIFF
                    // line above is unchanged, so gate baselines that parse it are unaffected).
                    if std::env::var("D2D_DIFF_SCENE").ok().as_deref() == Some(name.as_str()) {
                        let h = scene.size[1] as usize;
                        for py in 0..h { for px in 0..w {
                            let i = (py * w + px) * 4;
                            let (gb, ob) = (&run.pixels[i..i + 4.min(run.pixels.len() - i)], bgra.get(i..i + 4).unwrap_or(&[]));
                            if gb != ob { println!("  PX ({px},{py}) got={gb:?} want={ob:?}"); }
                        }}
                    }
                }
                fails.push(format!("{name}: {diff}"));
                fail += 1;
                if passing.contains(&name) {
                    regressed.push(format!("{name} (now FAIL: {diff})"));
                }
            }
        }
    }
    let digest = std::hash::Hasher::finish(&hasher);
    println!("\nconformance: {pass} pass, {skip} skip, {fail} fail; output digest {digest:#018x}");
    let missing = passing.into_iter().filter(|n| !passed.contains(n)).collect();
    Report { pass, skip, fail, fails, regressed, missing, digest }
}

#[cfg(test)]
fn check(r: Report) {
    if !r.fails.is_empty() {
        println!("FAILs:");
        for f in &r.fails {
            println!("  {f}");
        }
    }
    // A scene listed in passing.txt that is not currently a PASS is a regression; that is the only
    // thing that fails the build. Not-yet-exact scenes stay honest FAILs without breaking the suite.
    assert!(
        r.missing.is_empty(),
        "regression against the conformance suite ratchet (passing.txt) — these must still PASS:\n  {}",
        if r.regressed.is_empty() { format!("{:?}", r.missing) } else { r.regressed.join("\n  ") }
    );
}

// With `--features conformance-browser` the wasm32 test runs in a real headless browser instead of
// Node (same code, same corpus); without it, Node. Both are driven by wasm-bindgen-test-runner.
#[cfg(all(test, target_arch = "wasm32", feature = "conformance-browser"))]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg(test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn oracle_conformance() {
    let r = report();
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_test::console_log!("conformance output digest {:#018x}", r.digest);
    check(r);
}

#[cfg(test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn warp_sampler_regressions() {
    let corpus = cases();
    if corpus.is_empty() {
        return; // Public clones and unpopulated wasm embeds have no private oracle corpus.
    }
    for name in [
        "16-text-grayscale",
        // text-05-cleartype-gdi-classic: re-baselined to the .8875 oracle (2026-11-06), an
        // expected-fail until the text owner ports the .8875 ClearType behaviour — out of this
        // byte-exact guard like it is out of passing.txt.
        "text-07-rotated-cleartype",
        "textmodes-04-cleartype-mmm",
        "20-composite",
        "aniso-01-rotate20-magnify3",
        "opacitymask-brush-01-bitmap",
        "opacitymask-brush-02-linear",
        "overload-02-dc-opacitymask",
        "sampling-01-nearest-opacity",
        "sampling-02-nearest-layer-opacity",
        "sampling-03-radial-layer-opacity",
        "bitmaps-05-magnify-hqcubic",
        "fx-transform-11-scale-hqcubic",
        "fx-transform-12-perspective",
        "fx-transform-13-transform3d",
        "fx-sampler-01-scale-up-cubic",
        "fx-sampler-02-scale-down-cubic",
        "fx-sampler-03-affine-rotate-cubic",
        "fx-sampler-04-transform3d-cubic",
        "fx-sampler-05-transform3d-aniso",
        "fx-sampler-06-scale-up-msl",
        "fx-sampler-07-scale-down-msl",
        "fx-sampler-08-affine-rotate-msl",
        "fx-sampler-09-transform3d-minify-msl",
        "fx-sampler-10-affine-rotate-aniso",
        "fx-sampler-11-transform3d-msl-homogeneous",
        "fx-sampler-12-scale-up-aniso",
        "fx-sampler-13-scale-down-aniso",
        "fx-sampler-14-transform3d-magnify-aniso",
        "fx-sampler-15-transform3d-magnify-msl",
        "fx-boundary-horizon-aniso",
        "fx-boundary-horizon-msl",
        "fx-boundary-extreme-aniso",
        "fx-boundary-extreme-msl",
        "fx-boundary-affine-float-nearest",
        "fx-boundary-context-affine-aniso",
        "fx-boundary-context-affine-msl",
        "fx-boundary-fractional-translation-aniso",
        "fx-boundary-fractional-translation-msl",
        "fx-boundary-affine-float-linear",
        "fx-boundary-affine-float-cubic",
        "fx-boundary-affine-float-msl",
        "fx-boundary-affine-float-aniso",
        "aniso-01-rotate20-magnify3",
        "aniso-04-rotate30-minify-half",
        "aniso-05-rotate20-minify-quarter",
        "aniso-06-rotate30-squash-4to1",
        "custom-fx-04-affine-resample",
        "fx-gaussian-border-extend",
        "fx-transform-12-perspective",
        "fx-transform-13-transform3d",
        "fxhs-05-radius3",
        "fxstraighten-01-rotate-linear",
        "fxstraighten-02-maintain-size",
        "pdnv5-sharpenglitch",
        "18-gaussian-blur",
        "13-bitmap-brush",
    ] {
        let case = corpus.iter().find(|c| c.name == name).unwrap();
        let scene: Scene = serde_json::from_str(&case.scene).unwrap();
        let factory = Factory::new(false);
        let device = factory.create_device();
        let run = drive(&device, &factory, &scene).unwrap();
        assert_eq!(run.pixels.len(), case.bgra.len(), "{name}");
        assert_eq!(run.pixels.iter().zip(&case.bgra).position(|(a, b)| a != b), None, "{name}");
    }

}





// Native (Linux, no Wine/VM) differential-fuzz CRASH scan: drive every generated seed through the
// core render under catch_unwind. A panic anywhere in our code is a top-priority cross-site
// fingerprint (it crashes the page). Fast enough for 10^4+ seeds. `FUZZ_N`/`FUZZ_START` size the run
// (default 1000 from 0); gated behind `FUZZ_CRASH_SCAN=1` so the normal suite run is unaffected.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn compare_query_reads_null_as_non_finite() {
    use serde_json::{json, Value};
    let q = |kind: &str, got: Value| QueryRec { op: 0, kind: kind.into(), got };
    // Native's non-finite results arrive as null: null matches null (or a non-finite of ours) and
    // differs from a number, for scalars and 4-float results alike.
    assert_eq!(compare_query(&q("area", Value::Null), &Value::Null), None);
    assert_eq!(compare_query(&q("area", json!(f32::NAN)), &Value::Null), None);
    assert!(compare_query(&q("area", json!(1.5)), &Value::Null).is_some());
    assert!(compare_query(&q("length", Value::Null), &json!(2.0)).is_some());
    assert_eq!(compare_query(&q("max_scale", Value::Null), &Value::Null), None);
    assert_eq!(compare_query(&q("bounds", json!([null, 1.0, 2.0, 3.0])), &json!([null, 1.0, 2.0, 3.0])), None);
    assert_eq!(compare_query(&q("point_at_length", Value::Null), &Value::Null), None);
    assert!(compare_query(&q("bounds", json!([0.0, 1.0, 2.0, 3.0])), &Value::Null).is_some());
    assert_eq!(compare_query(&q("image_bounds", json!([null, null, null, null])), &Value::Null), None);
}

#[test]
fn fuzz_crash_scan() {
    if std::env::var("FUZZ_CRASH_SCAN").is_err() {
        return;
    }
    let n: u64 = std::env::var("FUZZ_N").ok().and_then(|s| s.parse().ok()).unwrap_or(1000);
    let start: u64 = std::env::var("FUZZ_START").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let factory = Factory::new(false);
    let device = factory.create_device();
    // Keep the panic site but silence the default hook's backtrace spew per scene.
    std::panic::set_hook(Box::new(|info| {
        if let Some(l) = info.location() {
            eprintln!("FUZZ-PANIC-AT {}:{}:{}", l.file(), l.line(), l.column());
        }
    }));
    // FUZZ_CRASH_DIR=<dir>: re-drive the captured (minimised) crash scene JSONs in <dir> directly,
    // instead of regenerating from seeds (gen_scene drifts). Reports which still panic on current code.
    if let Some(dir) = std::env::var_os("FUZZ_CRASH_DIR") {
        let mut entries: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("json")).collect();
        entries.sort();
        let mut panics: Vec<String> = Vec::new();
        for p in &entries {
            let name = p.file_stem().unwrap().to_string_lossy().to_string();
            let Ok(scene) = serde_json::from_slice::<Scene>(&std::fs::read(p).unwrap()) else { continue };
            eprintln!("--DRIVE {name}");
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drive(&device, &factory, &scene)));
            if res.is_err() { panics.push(name); } else { eprintln!("OK {name}"); }
        }
        let _ = std::panic::take_hook();
        println!("fuzz_crash_dir: {}/{} still panic: {:?}", panics.len(), entries.len(), panics);
        return;
    }
    // Hang watchdog: a seed that runs longer than FUZZ_HANG_SECS (default 10) is a hang — in wasm
    // (panic=abort, single-threaded) it would wedge the page, so it is as fatal as a crash. The
    // watchdog samples the in-flight seed once a second; if it stops advancing, it prints the seed
    // (so it is attributable even though the main thread can't be unwound) and aborts the process.
    // Resume the scan from the next seed after fixing the hang's root.
    let hang_secs: u64 = std::env::var("FUZZ_HANG_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let cur = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(start));
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        use std::sync::atomic::Ordering;
        let (cur, done) = (cur.clone(), done.clone());
        std::thread::spawn(move || {
            let (mut last, mut since) = (cur.load(Ordering::Relaxed), std::time::Instant::now());
            loop {
                std::thread::sleep(std::time::Duration::from_secs(1));
                if done.load(Ordering::Relaxed) {
                    return;
                }
                let c = cur.load(Ordering::Relaxed);
                if c != last {
                    last = c;
                    since = std::time::Instant::now();
                } else if since.elapsed().as_secs() >= hang_secs {
                    eprintln!("FUZZ-HANG seed {c} (>{hang_secs}s)");
                    std::process::abort();
                }
            }
        });
    }
    let mut panics: Vec<u64> = Vec::new();
    for seed in start..start + n {
        cur.store(seed, std::sync::atomic::Ordering::Relaxed);
        // Record the in-flight seed to FUZZ_SEED_LOG so an un-catchable abort (OOM over the ulimit, or
        // the hang watchdog's process::abort) stays attributable to its scene even though catch_unwind
        // can't recover it. A huge allocation (>~1.5 GB under `ulimit -v`) aborts without the FUZZ-HANG
        // line; a hang prints FUZZ-HANG first — that distinguishes the two abort causes.
        if let Some(p) = std::env::var_os("FUZZ_SEED_LOG") {
            let _ = std::fs::write(&p, seed.to_string());
        }
        let val = d2d_reference::fuzz::gen_scene(seed);
        let Ok(scene) = serde_json::from_value::<Scene>(val) else { continue };
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drive(&device, &factory, &scene)));
        if res.is_err() {
            eprintln!("FUZZ-PANIC seed {seed}");
            panics.push(seed);
        }
    }
    done.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = std::panic::take_hook();
    println!("fuzz_crash_scan: {n} seeds from {start}, {} panic(s): {:?}", panics.len(), panics);
    assert!(panics.is_empty(), "{} panic(s) in fuzz crash scan: {:?}", panics.len(), panics);
}

// Reproduce a crash from a saved minimised scene FILE (the fuzz agent's gen_scene differs from ours,
// so regenerating by seed can't reproduce it). FUZZ_FILE = a dir of fz-*.json; drives each through the
// core and prints its panic site. A pure triage diagnostic, gated off so the normal suite is unaffected.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn fuzz_file_repro() {
    let Some(dir) = std::env::var_os("FUZZ_FILE") else { return };
    let factory = Factory::new(false);
    let device = factory.create_device();
    std::panic::set_hook(Box::new(|info| {
        if let Some(l) = info.location() {
            eprintln!("FILE-PANIC-AT {}:{}:{}", l.file(), l.line(), l.column());
        }
    }));
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("FUZZ_FILE dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    for p in files {
        let Ok(txt) = std::fs::read_to_string(&p) else { continue };
        let Ok(scene) = serde_json::from_str::<Scene>(&txt) else {
            eprintln!("{}: SKIP (parse)", p.file_name().unwrap().to_string_lossy());
            continue;
        };
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drive(&device, &factory, &scene)));
        eprintln!("{}: {}", p.file_name().unwrap().to_string_lossy(), if res.is_err() { "PANIC" } else { "ok" });
    }
}

// EMFPLUS_FILE = path to an EMF (.emf) carrying EMF+ records; dumps the core EMF+ enumerator's output
// (type/flags/dataSize per record) so it can be checked against real GDI+ metafiles. Gated diagnostic.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn emfplus_file_dump() {
    let Some(path) = std::env::var_os("EMFPLUS_FILE") else { return };
    let bytes = std::fs::read(&path).expect("EMFPLUS_FILE");
    let mf = d2d::metafile::GdiMetafile::parse(bytes).expect("valid EMF");
    let stream = mf.emfplus_stream();
    eprintln!("EMFPLUS stream {} bytes", stream.len());
    for r in d2d::metafile::GdiMetafile::emfplus_records(&stream) {
        eprintln!("EMFPLUS {:#06x} flags={:#06x} dataSize={}", r.rtype, r.flags, r.data.len());
    }
    for c in d2d::emfplus::Player::interpret(&mf) {
        eprintln!("CMD {c:?}");
    }
}

// Bisect a hanging fuzz seed: drive gen_scene(FUZZ_TIME_SEED) with the op list truncated to k ops,
// timing each k, to find which op blows up. Throwaway diagnostic, gated by env.
#[test]
fn fuzz_time_one() {
    let full = if let Some(f) = std::env::var_os("FUZZ_TIME_FILE") {
        serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(f).unwrap()).unwrap()
    } else {
        let Some(seed) = std::env::var("FUZZ_TIME_SEED").ok().and_then(|s| s.parse::<u64>().ok()) else { return };
        d2d_reference::fuzz::gen_scene(seed)
    };
    let factory = Factory::new(false);
    let device = factory.create_device();
    let nops = full["ops"].as_array().map(|a| a.len()).unwrap_or(0);
    for k in 1..=nops {
        let mut v = full.clone();
        v["ops"].as_array_mut().unwrap().truncate(k);
        let op_tag = v["ops"][k - 1]["op"].as_str().unwrap_or("?").to_string();
        let Ok(scene) = serde_json::from_value::<Scene>(v) else { continue };
        let t = std::time::Instant::now();
        let run = drive(&device, &factory, &scene);
        let ms = t.elapsed().as_millis();
        // FUZZ_PIXEL=x,y: print that pixel's BGRA after each truncation to bisect which op paints it.
        if let (Some(px), Ok(r)) = (std::env::var("FUZZ_PIXEL").ok(), &run) {
            let xy: Vec<usize> = px.split(',').filter_map(|s| s.parse().ok()).collect();
            if xy.len() == 2 {
                let o = (xy[1] * scene.size[0] as usize + xy[0]) * 4;
                let p = r.pixels.get(o..o + 4).unwrap_or(&[]);
                eprintln!("FPIX k={k} op#{}={op_tag} ({},{})={:?}", k - 1, xy[0], xy[1], p);
            }
            continue;
        }
        eprintln!("FTIME k={k} op#{}={op_tag} {ms}ms", k - 1);
        if ms > 4000 {
            eprintln!("FTIME-SLOW op#{} ({op_tag})", k - 1);
            break;
        }
    }
}

// Render-time benchmark baseline. D2D_BENCH=1 times the whole suite; with D2D_BENCH_CORPUS=<dir>
// (+ FUZZ_START/FUZZ_N) it also times that captured corpus range. Each scene is driven once to warm
// caches, then the minimum wall time over D2D_BENCH_REPS runs is kept (min = least noise). Prints a
// per-category total and the slowest scenes, so profiling can target the hot ones. Output-neutral.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn bench() {
    if std::env::var_os("D2D_BENCH").is_none() {
        return;
    }
    let reps: u32 = std::env::var("D2D_BENCH_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let time_scene = |scene: &Scene| -> Option<u128> {
        let factory = Factory::new(false);
        let device = factory.create_device();
        drive(&device, &factory, scene).ok()?;
        let mut best = u128::MAX;
        for _ in 0..reps {
            let t = std::time::Instant::now();
            let _ = drive(&device, &factory, scene);
            best = best.min(t.elapsed().as_micros());
        }
        Some(best)
    };
    let mut report = |label: &str, mut rows: Vec<(String, u128, [u32; 2])>| {
        rows.sort_by(|a, b| b.1.cmp(&a.1));
        let total: u128 = rows.iter().map(|r| r.1).sum();
        eprintln!("BENCH {label}: {} scenes, total-min {}us ({:.1}ms)", rows.len(), total, total as f64 / 1000.0);
        for (name, us, sz) in rows.iter().take(25) {
            eprintln!("  BENCH {label} {us:>9}us {:>4}x{:<4} {name}", sz[0], sz[1]);
        }
    };

    // D2D_BENCH=scenes times only the explicit scene/corpus dirs (skip the suite's per-scene re-timing).
    if std::env::var("D2D_BENCH").ok().as_deref() != Some("scenes") {
        let suite: Vec<(String, u128, [u32; 2])> = cases()
            .into_iter()
            .filter_map(|Case { name, scene, .. }| {
                let scene: Scene = serde_json::from_str(&scene).ok()?;
                Some((name, time_scene(&scene)?, scene.size))
            })
            .collect();
        report("suite", suite);
    }

    if let Some(dir) = std::env::var_os("D2D_BENCH_CORPUS") {
        let dir = std::path::PathBuf::from(dir);
        let start: u64 = std::env::var("FUZZ_START").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        let n: u64 = std::env::var("FUZZ_N").ok().and_then(|s| s.parse().ok()).unwrap_or(1000);
        let rows: Vec<(String, u128, [u32; 2])> = (start..start + n)
            .filter_map(|seed| {
                let sj = std::fs::read_to_string(dir.join(format!("fz-{seed:010}.scene.json"))).ok()?;
                let scene: Scene = serde_json::from_str(&sj).ok()?;
                Some((format!("fz-{seed}"), time_scene(&scene)?, scene.size))
            })
            .collect();
        report("corpus", rows);
    }

    // D2D_BENCH_SCENES=<dir>: time every *.json scene in a directory (the synthetic photo-size
    // real-use workloads). Each is timed on its own so the slowest category stands out.
    if let Some(dir) = std::env::var_os("D2D_BENCH_SCENES") {
        let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .expect("bench scenes dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
            .collect();
        files.sort();
        let rows: Vec<(String, u128, [u32; 2])> = files
            .iter()
            .filter_map(|p| {
                let scene: Scene = serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()?;
                let name = p.file_stem()?.to_str()?.to_owned();
                Some((name, time_scene(&scene)?, scene.size))
            })
            .collect();
        report("scenes", rows);
    }
}

// Debug: render one arbitrary scene json (D2D_RENDER_SCENE=<path>) and dump our BGRA to D2D_RENDER_OUT
// (default <temp dir>/ours.bgra). Throwaway; lets me diff our output against a native capture off-suite.
#[test]
fn render_one() {
    let Some(p) = std::env::var_os("D2D_RENDER_SCENE") else { return };
    let scene: Scene = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
    let factory = Factory::new(false);
    let device = factory.create_device();
    let run = drive(&device, &factory, &scene).unwrap();
    let out = std::env::var("D2D_RENDER_OUT")
        .unwrap_or_else(|_| std::env::temp_dir().join("ours.bgra").display().to_string());
    std::fs::write(&out, &run.pixels).unwrap();
    eprintln!("render_one: wrote {} bytes to {out}; end_draw {} tag1 {} tag2 {}", run.pixels.len(), run.end.hr, run.end.tag1, run.end.tag2);
    // Optional: dump the query stream (geometry_op/widen figures+segments) for off-suite value diffs.
    if let Ok(qout) = std::env::var("D2D_QUERY_OUT") {
        let qs: Vec<_> = run.queries.iter().map(|q| serde_json::json!({"op": q.op, "kind": q.kind, "result": q.got})).collect();
        std::fs::write(&qout, serde_json::to_string_pretty(&serde_json::json!({"queries": qs})).unwrap()).unwrap();
        eprintln!("render_one: wrote {} queries to {qout}", run.queries.len());
    }
}

// Feed a native hinted contour (captured as JSON: x in device*512, y in device*64
// y-up, on-curve flags) through OUR 8x1 grayscale scan converter and print the resolved coverage grid.
// If this matches native's gray.bin, our scan is byte-exact on native's outline and any render residual
// is the HINTED OUTLINE, not the scan. D2D_SCAN_CONTOUR=<contours.json> D2D_SCAN_PEN=<pen_x>,<pen_y>.
#[test]
fn scan_native_contour() {
    let Some(p) = std::env::var_os("D2D_SCAN_CONTOUR") else { return };
    let (penx, peny) = std::env::var("D2D_SCAN_PEN").ok()
        .and_then(|s| { let v: Vec<f64> = s.split(',').filter_map(|t| t.parse().ok()).collect(); (v.len()==2).then(|| (v[0], v[1])) })
        .unwrap_or((0.0, 0.0));
    let idx: usize = std::env::var("D2D_SCAN_IDX").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let j: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    let c = &j[idx];
    // Mode: D2D_SCAN_MODE override, else the glyph's own `scan_mode` (native's per-glyph dropout mode).
    let mode: u16 = std::env::var("D2D_SCAN_MODE").ok().and_then(|s| s.parse().ok())
        .unwrap_or_else(|| c["scan_mode"].as_u64().unwrap_or(2) as u16);
    // Oversample: D2D_SCAN_SX/SY (default 8x1 grayscale/GDI).
    let sx: i32 = std::env::var("D2D_SCAN_SX").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let sy: i32 = std::env::var("D2D_SCAN_SY").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let xs: Vec<f64> = c["x"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let ys: Vec<f64> = c["y"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let on: Vec<bool> = c["on"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
    let ends: Vec<usize> = c["ends"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    // Split into contours at `ends` (inclusive); device point = (pen_x + x/512, pen_y - y/64) [y-down].
    let mut contours: Vec<(Vec<[f64;2]>, Vec<bool>)> = Vec::new();
    let mut s = 0usize;
    for &e in &ends {
        let pts: Vec<[f64;2]> = (s..=e).map(|i| [penx + xs[i]/512.0, peny - ys[i]/64.0]).collect();
        contours.push((pts, on[s..=e].to_vec()));
        s = e + 1;
    }
    let tex = d2d::text::raster::coverage_box(&contours, mode, sx, sy, (0,0), false);
    eprintln!("SCANCONTOUR dev=[{},{},{},{}] w={} h={}", tex.left, tex.top, tex.right, tex.bottom, tex.width(), tex.height());
    let w = tex.width() as usize;
    for r in 0..tex.height() as usize {
        eprintln!("  y{:>3}: {:?}", tex.top + r as i32, &tex.data[r*w..(r+1)*w]);
    }
}

// M2: run a LINE-only contour (dw_oracle contours.json, x device*512, y device*64, on all true) through
// our both-axis calc_line_bi and print the H and V crossing marks, to diff against the native fsc oracle
// (dw_oracle.py). D2D_BISCAN_CONTOUR=<contours.json> D2D_BISCAN_IDX=<glyph index, default 0>.
#[test]
fn biscan_line_marks() {
    let Some(p) = std::env::var_os("D2D_BISCAN_CONTOUR") else { return };
    let idx: usize = std::env::var("D2D_BISCAN_IDX").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let j: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    let c = &j[idx];
    let xs: Vec<i32> = c["x"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect();
    let ys: Vec<i32> = c["y"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect();
    let on: Vec<bool> = c["on"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
    let ends: Vec<usize> = c["ends"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let (xmn, xmx) = (xs.iter().min().unwrap() >> 6, xs.iter().max().unwrap() >> 6);
    let (ymn, ymx) = (ys.iter().min().unwrap() >> 6, ys.iter().max().unwrap() >> 6);
    let bounds = [ymn - 3, (ymx - ymn) + 7, xmn - 3, (xmx - xmn) + 7];
    let (mut hm, mut vm) = d2d::text::raster::biscan_marks(&xs, &ys, &on, &ends, bounds);
    hm.sort(); vm.sort();
    eprintln!("BISCAN H={} V={}", hm.len(), vm.len());
    eprintln!("BISCAN_H {:?}", hm);
    eprintln!("BISCAN_V {:?}", vm);
}

// Localise a fuzz first-diff pixel to the op that introduces it: render every op-prefix and print the
// pixel value whenever it changes. The op that drives our pixel to its final (divergent) value owns the
// divergence; a pixel that never leaves 0 means we draw nothing where native does (find the covering op).
// D2D_BISECT_SCENE=<scene.json> D2D_BISECT_X/Y=<px>; optional D2D_BISECT_BGRA=<native.bgra> for reference.
#[test]
fn bisect_pixel() {
    let Some(scene_path) = std::env::var_os("D2D_BISECT_SCENE") else { return };
    let json = std::fs::read_to_string(&scene_path).unwrap();
    let x: usize = std::env::var("D2D_BISECT_X").unwrap().parse().unwrap();
    let y: usize = std::env::var("D2D_BISECT_Y").unwrap().parse().unwrap();
    let full: Scene = serde_json::from_str(&json).unwrap();
    let w = full.size[0] as usize;
    let n = full.ops.len();
    let idx = (y * w + x) * 4;
    if let Some(bp) = std::env::var_os("D2D_BISECT_BGRA") {
        let b = std::fs::read(bp).unwrap();
        eprintln!("BISECT native @({x},{y}) = {:?}", &b[idx..idx + 4]);
    }
    eprintln!("BISECT n_ops={n} size={:?}", full.size);
    let mut prev = [0u8; 4];
    for k in 1..=n {
        let mut sc: Scene = serde_json::from_str(&json).unwrap();
        let opdbg = format!("{:?}", &sc.ops[k - 1]);
        let opname: String = opdbg.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
        sc.ops.truncate(k);
        let factory = Factory::new(false);
        let device = factory.create_device();
        let px = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drive(&device, &factory, &sc))) {
            Ok(Ok(run)) if idx + 4 <= run.pixels.len() => {
                [run.pixels[idx], run.pixels[idx + 1], run.pixels[idx + 2], run.pixels[idx + 3]]
            }
            Ok(Ok(_)) => { continue; }
            _ => { eprintln!("  k={k:3} op={opname:16} <render error>"); prev = [255; 4]; continue; }
        };
        if px != prev {
            eprintln!("  k={k:3} op={opname:16} px={px:?}  <- changed");
        }
        prev = px;
    }
    eprintln!("BISECT done; the last changing op before the divergent value is the owner.");
}

// Native differential: for each captured native oracle in FUZZ_DIFF_DIR (fz-<seed>.bgra/.json from
// d2d-ref on the VM), drive gen_scene(seed) through our core and compare, exactly as the suite runner
// does against its oracles. Prints FDIFF <seed>: <first mismatch> per divergence for minimise+bucket.
// Gated by FUZZ_DIFF_DIR; FUZZ_START/FUZZ_N bound the range.
#[test]
fn fuzz_diff() {
    let Some(dir) = std::env::var_os("FUZZ_DIFF_DIR") else { return };
    let dir = std::path::PathBuf::from(dir);
    let start: u64 = std::env::var("FUZZ_START").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let n: u64 = std::env::var("FUZZ_N").ok().and_then(|s| s.parse().ok()).unwrap_or(1000);
    // Each scene drives on its own thread with a deadline, so a renderer hang (an infinite loop on an
    // adversarial input) is reported as FDIFF-HANG and the corpus continues, instead of one hang
    // stalling the whole run. An abandoned thread can't borrow a shared device, so — exactly like the
    // suite runner — each scene gets a fresh factory/device (the output is independent of factory
    // identity, so this changes no digest). FUZZ_DEADLINE_S overrides the 20 s default.
    let deadline = std::time::Duration::from_secs(
        std::env::var("FUZZ_DEADLINE_S").ok().and_then(|s| s.parse().ok()).unwrap_or(20),
    );
    let (mut diffs, mut skips, mut ok) = (Vec::new(), 0u32, 0u32);
    // FUZZ_NOPIX: also compare a scene whose native EndDraw failed (no .bgra written) on its HRESULTs,
    // tags and queries. Off by default, so the gate's comparable set is unchanged.
    let nopix = std::env::var_os("FUZZ_NOPIX").is_some();
    for seed in start..start + n {
        let bgra = match std::fs::read(dir.join(format!("fz-{seed:010}.bgra"))) {
            Ok(b) => b,
            Err(_) if nopix => Vec::new(),
            Err(_) => continue,
        };
        let Ok(oj) = std::fs::read_to_string(dir.join(format!("fz-{seed:010}.json"))) else { continue };
        let Ok(oracle) = serde_json::from_str::<Oracle>(&oj) else { continue };
        // gen_scene drifts as areas are added, so regenerating a seed can yield a different scene than
        // the one that was captured (seed→scene is not stable across versions). Prefer a pinned scene
        // sidecar (fz-<seed>.scene.json, saved at capture time) so the diff is version-safe; only fall
        // back to regenerating when the capture predates sidecars (valid only if gen_scene is unchanged).
        let scene = match std::fs::read_to_string(dir.join(format!("fz-{seed:010}.scene.json"))) {
            Ok(sj) => serde_json::from_str::<Scene>(&sj),
            Err(_) => serde_json::from_value::<Scene>(d2d_reference::fuzz::gen_scene(seed)),
        };
        // A scene that fails to deserialize (e.g. a captured sidecar carries a newer Scene field this
        // binary predates — `deny_unknown_fields`) is NOT comparable here, exactly like a drive error:
        // emit FDIFF-SKIP + count it as a skip so the A/B gate treats it as not-comparable (newly
        // comparable in the newer binary), not as a spurious new divergence. (764ecb3's `ranges` field.)
        let Ok(scene) = scene else {
            println!("FDIFF-SKIP {seed}: scene does not parse");
            skips += 1;
            continue;
        };
        let w = scene.size[0];
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let factory = Factory::new(false);
            let device = factory.create_device();
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drive(&device, &factory, &scene)));
            let _ = tx.send(r);
        });
        let hash = std::env::var_os("FUZZ_HASH").is_some();
        let driven = match rx.recv_timeout(deadline) {
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // A renderer hang (infinite loop) is a divergence: native rendered this scene (its
                // oracle is present), ours never returns. Report it and move on; the worker thread is
                // abandoned (it keeps spinning until the process exits, acceptable for a gate run).
                println!("FDIFF-HANG {seed}");
                diffs.push(seed);
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                // The worker dropped the sender without a value — can't happen (catch_unwind always
                // sends), but handle it as not-comparable rather than unwrap-panicking the run.
                println!("FDIFF-SKIP {seed}: worker vanished");
                skips += 1;
                continue;
            }
            // A panic inside drive is part of the output too: base and candidate must agree. Treat it
            // like a drive error (skip / hashed err marker) rather than aborting the whole corpus.
            Ok(Err(_panic)) => {
                if hash { println!("FHASH {seed} err:panic"); } else { println!("FDIFF-SKIP {seed}: panic"); }
                skips += 1;
                continue;
            }
            Ok(Ok(d)) => d,
        };
        match driven {
            Ok(run) if hash => {
                // Identity gate, corpus side: one full-output digest per scene (pixels + HRESULTs +
                // tags + queries), to compare a base build against a candidate line for line.
                println!("FHASH {seed} {:#018x}", full_digest(&run));
                ok += 1;
            }
            Ok(run) if std::env::var_os("FUZZ_TOTAL").is_some() => {
                let n = run.pixels.len().min(bgra.len());
                let (mut npx, mut maxd) = (0u32, 0i32);
                for i in (0..n).step_by(4) {
                    let dd = (0..4).map(|c| (run.pixels[i + c] as i32 - bgra[i + c] as i32).abs()).max().unwrap();
                    if dd > 0 { npx += 1; maxd = maxd.max(dd); }
                }
                if npx > 0 { println!("FTOTAL {seed} {npx} {maxd}"); diffs.push(seed); } else { ok += 1; }
            }
            Ok(run) => match compare(&run, &oracle, w, &bgra) {
                Ok(()) => ok += 1,
                Err(d) => {
                    println!("FDIFF {seed}: {d}");
                    diffs.push(seed);
                }
            },
            Err(e) if hash => {
                // A drive error is part of the output too: a base and candidate must fail identically.
                println!("FHASH {seed} err:{e}");
                skips += 1;
            }
            Err(e) => {
                println!("FDIFF-SKIP {seed}: {e}");
                skips += 1;
            }
        }
    }
    println!("fuzz_diff: ok={ok} skip={skips} diff={} seeds={:?}", diffs.len(), diffs);
}

// Attribute each pixel-divergent fuzz seed to the op that paints the first mismatching pixel (reuses
// fuzz_diff's captures via FUZZ_DIFF_DIR + the truncate-bisect of fuzz_time_one). For a `text` op it
// also prints font/size/measuring/antialias so the text divergences can be grouped by regime. Gated by
// FUZZ_ATTR; FUZZ_START/FUZZ_N bound the range. Throwaway triage diagnostic.
#[test]
fn fuzz_attr() {
    if std::env::var_os("FUZZ_ATTR").is_none() {
        return;
    }
    let dir = std::path::PathBuf::from(std::env::var_os("FUZZ_DIFF_DIR").expect("FUZZ_DIFF_DIR"));
    let start: u64 = std::env::var("FUZZ_START").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let n: u64 = std::env::var("FUZZ_N").ok().and_then(|s| s.parse().ok()).unwrap_or(1000);
    let factory = Factory::new(false);
    let device = factory.create_device();
    for seed in start..start + n {
        let Ok(bgra) = std::fs::read(dir.join(format!("fz-{seed:010}.bgra"))) else { continue };
        // Prefer the pinned scene sidecar (version-safe) over regenerating, which drifts (see fuzz_diff).
        let full = match std::fs::read_to_string(dir.join(format!("fz-{seed:010}.scene.json"))) {
            Ok(j) => serde_json::from_str::<serde_json::Value>(&j).unwrap_or_else(|_| d2d_reference::fuzz::gen_scene(seed)),
            Err(_) => d2d_reference::fuzz::gen_scene(seed),
        };
        let Ok(scene) = serde_json::from_value::<Scene>(full.clone()) else { continue };
        let Ok(run) = drive(&device, &factory, &scene) else { continue };
        if run.pixels.len() != bgra.len() || bgra.is_empty() {
            continue;
        }
        // first mismatching pixel + max channel delta there
        let w = scene.size[0] as usize;
        let Some((idx, dmax)) = run.pixels.as_chunks::<4>().0.iter().zip(bgra.as_chunks::<4>().0).enumerate()
            .find_map(|(i, (a, b))| (a != b).then(|| (i, (0..4).map(|c| (a[c] as i32 - b[c] as i32).abs()).max().unwrap())))
        else { continue };
        let (px, py) = (idx % w, idx / w);
        // bisect: the last op whose truncation changes pixel idx is its painter. Our clips/layers draw
        // content to an offscreen and composite it back ON POP, so truncating mid-clip would hide a
        // residual drawn inside it (it stays in the offscreen) and the divergence would be mis-blamed on
        // the trailing pop_clip/pop_layer. Append synthetic pops to close whatever is still open at each
        // truncation, so the offscreen content composites into the target and attribution lands on the
        // real content op (gradient, text, mesh, …).
        let nops = full["ops"].as_array().map(|a| a.len()).unwrap_or(0);
        let mut painter = usize::MAX;
        let mut prev: Option<[u8; 4]> = None;
        for k in 1..=nops {
            let mut v = full.clone();
            let arr = v["ops"].as_array_mut().unwrap();
            arr.truncate(k);
            let mut open: Vec<&str> = Vec::new();
            for o in arr.iter() {
                match o["op"].as_str() {
                    Some("push_clip") => open.push("pop_clip"),
                    Some("push_layer") => open.push("pop_layer"),
                    Some("pop_clip" | "pop_layer") => { open.pop(); }
                    _ => {}
                }
            }
            for pop in open.iter().rev() {
                arr.push(serde_json::json!({ "op": pop }));
            }
            let Ok(sc) = serde_json::from_value::<Scene>(v) else { continue };
            let Ok(r) = drive(&device, &factory, &sc) else { continue };
            let cur: [u8; 4] = r.pixels.get(idx * 4..idx * 4 + 4).and_then(|s| s.try_into().ok()).unwrap_or([0; 4]);
            if prev != Some(cur) {
                painter = k - 1;
                prev = Some(cur);
            }
        }
        let op = &full["ops"].get(painter).cloned().unwrap_or(serde_json::Value::Null);
        let tag = op["op"].as_str().unwrap_or("?");
        let extra = if tag == "text" {
            format!(" font={:?} text={:?} size={} meas={} aa={}", op["font"].as_str().unwrap_or(""),
                op["text"].as_str().unwrap_or(""), op["size"], op["measuring"], op["antialias"])
        } else { String::new() };
        println!("ATTR {seed}: op#{painter}={tag} delta={dmax} at ({px},{py}){extra}");
    }
}

// FUZZ_BUCKET: re-bucket each diffing seed as COVERAGE vs SAMPLER. At the first-mismatch pixel, bisect
// the painter op, render the prefix (ops before it) for the pre-op dest D, then un-composite ours and
// native over D (source-over) to the STRAIGHT premul-free source colour. If both straight colours agree
// (<=2/255) the gradient sample is correct and only coverage/compositing differs (COVERAGE); otherwise
// the sampler itself diverges (SAMPLER). FUZZ_SEEDS=csv targets a specific list; else START/N range.
#[test]
fn fuzz_bucket() {
    if std::env::var_os("FUZZ_BUCKET").is_none() {
        return;
    }
    let dir = std::path::PathBuf::from(std::env::var_os("FUZZ_DIFF_DIR").expect("FUZZ_DIFF_DIR"));
    let list: Option<Vec<u64>> = std::env::var("FUZZ_SEEDS").ok().map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect());
    let start: u64 = std::env::var("FUZZ_START").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let n: u64 = std::env::var("FUZZ_N").ok().and_then(|s| s.parse().ok()).unwrap_or(1000);
    let seeds: Vec<u64> = list.unwrap_or_else(|| (start..start + n).collect());
    let factory = Factory::new(false);
    let device = factory.create_device();
    let (mut cov, mut samp, mut incon) = (0u32, 0u32, 0u32);
    // Un-composite a source-over output `o` over dest `d` (both premul BGRA) to straight colour 0..1.
    let unover = |o: [u8; 4], d: [u8; 4]| -> Option<[f32; 3]> {
        let sa = (o[3] as f32 - d[3] as f32) / (1.0 - d[3] as f32 / 255.0);
        if sa < 8.0 { return None; } // too little source to recover a stable hue
        Some(core::array::from_fn(|c| {
            let sp = o[c] as f32 - d[c] as f32 * (1.0 - sa / 255.0);
            (sp / sa).clamp(0.0, 1.0)
        }))
    };
    for seed in seeds {
        let Ok(bgra) = std::fs::read(dir.join(format!("fz-{seed:010}.bgra"))) else { continue };
        let full = match std::fs::read_to_string(dir.join(format!("fz-{seed:010}.scene.json"))) {
            Ok(j) => serde_json::from_str::<serde_json::Value>(&j).unwrap_or_else(|_| d2d_reference::fuzz::gen_scene(seed)),
            Err(_) => d2d_reference::fuzz::gen_scene(seed),
        };
        let Ok(scene) = serde_json::from_value::<Scene>(full.clone()) else { continue };
        let Ok(run) = drive(&device, &factory, &scene) else { continue };
        if run.pixels.len() != bgra.len() || bgra.is_empty() {
            continue;
        }
        let w = scene.size[0] as usize;
        // Anchor on the MAX-diff pixel (not the first mismatch): the first mismatch is usually a tiny
        // edge texel, while the real divergence — a structural coverage/alpha miss or the worst sample —
        // is elsewhere, and classifying/attributing there is what routes correctly.
        let Some((idx, dmax)) = run.pixels.as_chunks::<4>().0.iter().zip(bgra.as_chunks::<4>().0).enumerate()
            .filter_map(|(i, (a, b))| (a != b).then(|| (i, (0..4).map(|c| (a[c] as i32 - b[c] as i32).abs()).max().unwrap())))
            .max_by_key(|&(_, d)| d)
        else { continue };
        let da = (run.pixels[idx * 4 + 3] as i32 - bgra[idx * 4 + 3] as i32).abs();
        let (px, py) = (idx % w, idx / w);
        let nops = full["ops"].as_array().map(|a| a.len()).unwrap_or(0);
        let (mut painter, mut prev) = (usize::MAX, None::<[u8; 4]>);
        for k in 1..=nops {
            let mut v = full.clone();
            let arr = v["ops"].as_array_mut().unwrap();
            arr.truncate(k);
            // Close whatever clip/layer is still open at this truncation so offscreen content
            // composites back and attribution lands on the real content op, not a trailing pop.
            let mut open: Vec<&str> = Vec::new();
            for o in arr.iter() {
                match o["op"].as_str() {
                    Some("push_clip") => open.push("pop_clip"),
                    Some("push_layer") => open.push("pop_layer"),
                    Some("pop_clip" | "pop_layer") => { open.pop(); }
                    _ => {}
                }
            }
            for pop in open.iter().rev() { arr.push(serde_json::json!({ "op": pop })); }
            let Ok(sc) = serde_json::from_value::<Scene>(v) else { continue };
            let Ok(r) = drive(&device, &factory, &sc) else { continue };
            let curp: [u8; 4] = r.pixels.get(idx * 4..idx * 4 + 4).and_then(|s| s.try_into().ok()).unwrap_or([0; 4]);
            if prev != Some(curp) { painter = k - 1; prev = Some(curp); }
        }
        let tag = full["ops"].get(painter).and_then(|o| o["op"].as_str()).unwrap_or("?").to_string();
        // Resolve the painter's brush kind (walk its brush id back to the create op), for routing by
        // brush: a text/primitive/fill carries a brush; draw_bitmap samples its own bitmap.
        let bkind = {
            let painter_op = full["ops"].get(painter);
            if tag == "draw_bitmap" { "bitmap".to_string() }
            else if let Some(bid) = painter_op.and_then(|o| o["brush"].as_str()) {
                full["ops"].as_array().into_iter().flatten()
                    .find(|o| o["id"].as_str() == Some(bid))
                    .and_then(|o| o["op"].as_str()).unwrap_or("?").to_string()
            } else { format!("none/{tag}") }
        };
        // Dest = the prefix (ops 0..painter).
        let d: [u8; 4] = if painter == 0 { [0; 4] } else {
            let mut v = full.clone();
            v["ops"].as_array_mut().unwrap().truncate(painter);
            match serde_json::from_value::<Scene>(v).ok().and_then(|sc| drive(&device, &factory, &sc).ok()) {
                Some(r) => r.pixels.get(idx * 4..idx * 4 + 4).and_then(|s| s.try_into().ok()).unwrap_or([0; 4]),
                None => [0; 4],
            }
        };
        let o: [u8; 4] = run.pixels[idx * 4..idx * 4 + 4].try_into().unwrap();
        let nn: [u8; 4] = bgra[idx * 4..idx * 4 + 4].try_into().unwrap();
        let verdict = match (unover(o, d), unover(nn, d)) {
            (Some(so), Some(sn)) => {
                let md = (0..3).map(|c| (so[c] - sn[c]).abs()).fold(0.0f32, f32::max);
                if md <= 2.0 / 255.0 {
                    cov += 1;
                    format!("COVERAGE straightΔ={:.4}", md)
                } else {
                    samp += 1;
                    format!("SAMPLER straightΔ={:.4} so={:?} sn={:?}", md, so.map(|v| (v * 255.0).round() as i32), sn.map(|v| (v * 255.0).round() as i32))
                }
            }
            _ => { incon += 1; "INCONCLUSIVE(low-alpha)".into() }
        };
        let kind = if da * 2 >= dmax { "STRUCT" } else { "SAMPLE" };
        println!("BUCKET {seed}: op#{painter}={tag} brush={bkind} d={dmax} dA={da} {kind} ({px},{py}) {verdict}");
    }
    println!("fuzz_bucket: COVERAGE={cov} SAMPLER={samp} INCONCLUSIVE={incon}");
}

// FUZZ_DUMP_SEED=<seed>: render our full scene and print, for a window around the first mismatch, the
// BGRA of ours vs the native capture — to eyeball whether a divergence is a shifted glyph (hinting/
// positioning) or a clip/rect edge. Gated; needs FUZZ_DIFF_DIR.
// Write gen_scene(seed) JSON for a seed range into FUZZ_GEN_DIR (FUZZ_GEN_START, FUZZ_GEN_N), each
// round-tripped through read_scene like the standalone fuzz-gen bin. Lives here because the conformance
// test builds d2d_reference cleanly as a dep, sidestepping the standalone-bin resolver panic. Gated.
#[test]
fn fuzz_gen() {
    let Some(dir) = std::env::var_os("FUZZ_GEN_DIR") else { return };
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).unwrap();
    let start: u64 = std::env::var("FUZZ_GEN_START").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let n: u64 = std::env::var("FUZZ_GEN_N").ok().and_then(|s| s.parse().ok()).unwrap_or(1000);
    for seed in start..start + n {
        let scene = d2d_reference::fuzz::gen_scene(seed);
        let path = dir.join(format!("fz-{seed:010}.json"));
        std::fs::write(&path, serde_json::to_string(&scene).unwrap()).unwrap();
        d2d_reference::read_scene(&path).unwrap_or_else(|e| panic!("seed {seed} invalid: {e:#}"));
    }
    eprintln!("fuzz_gen: wrote {n} scenes (seeds {start}..{}) to {}", start + n, dir.display());
}

#[test]
fn fuzz_dump() {
    let Some(sv) = std::env::var("FUZZ_DUMP_SEED").ok().and_then(|s| s.parse::<u64>().ok()) else { return };
    let dir = std::path::PathBuf::from(std::env::var_os("FUZZ_DIFF_DIR").expect("FUZZ_DIFF_DIR"));
    let bgra = std::fs::read(dir.join(format!("fz-{sv:010}.bgra"))).unwrap_or_default();
    // Prefer the version-safe sidecar (as fuzz_diff does) so the dump matches the capture even after
    // gen_scene has drifted; fall back to regenerating only when no sidecar was saved.
    let mut sval: serde_json::Value = match std::fs::read_to_string(dir.join(format!("fz-{sv:010}.scene.json"))) {
        Ok(j) => serde_json::from_str(&j).unwrap(),
        Err(_) => d2d_reference::fuzz::gen_scene(sv),
    };
    // FUZZ_DUMP_TRUNC=k truncates to the first k painter ops — bisect which op paints a pixel.
    if let Some(k) = std::env::var("FUZZ_DUMP_TRUNC").ok().and_then(|s| s.parse::<usize>().ok()) {
        sval["ops"].as_array_mut().unwrap().truncate(k);
    }
    let scene: Scene = serde_json::from_value(sval).unwrap();
    let factory = Factory::new(false);
    let device = factory.create_device();
    let run = drive(&device, &factory, &scene).unwrap();
    if std::env::var_os("FUZZ_DUMP_HR").is_some() {
        let want = std::fs::read_to_string(dir.join(format!("fz-{sv:010}.json"))).ok()
            .and_then(|j| serde_json::from_str::<serde_json::Value>(&j).ok());
        let w = want.as_ref().and_then(|o| o["end_draw"]["hr"].as_str()).unwrap_or("?").to_string();
        println!("HR seed {sv} ours_end={} native_end={w} {}", run.end.hr, if run.end.hr == w { "MATCH" } else { "DIFF" });
    }
    if std::env::var_os("FUZZ_DUMP_QUERIES").is_some() {
        for q in &run.queries { println!("QUERY {} {} {}", q.op, q.kind, q.got); }
    }
    // FUZZ_DUMP_HR prints each op's HRESULT/tags and EndDraw's, the channel compared before pixels.
    if std::env::var_os("FUZZ_DUMP_HR").is_some() {
        for (i, o) in run.ops.iter().enumerate() { println!("OPHR {i} {} {} {}", o.hr, o.tag1, o.tag2); }
        println!("ENDHR {} {} {}", run.end.hr, run.end.tag1, run.end.tag2);
    }
    let w = scene.size[0] as usize;
    let h = scene.size[1] as usize;
    // FUZZ_DUMP_OUT=path writes our BGRA, for a whole-image diff against the capture.
    if let Some(path) = std::env::var_os("FUZZ_DUMP_OUT") {
        std::fs::write(path, &run.pixels).unwrap();
    }
    // Full-image diff stats: how many pixels differ and the worst per-channel delta.
    let (mut ndiff, mut maxd) = (0u32, 0i32);
    for (a, b) in run.pixels.as_chunks::<4>().0.iter().zip(bgra.as_chunks::<4>().0) {
        if a != b {
            ndiff += 1;
            maxd = maxd.max(a.iter().zip(b).map(|(x, y)| (*x as i32 - *y as i32).abs()).max().unwrap_or(0));
        }
    }
    // FUZZ_DUMP_PX="x,y" pins the window (so a truncated render can be inspected at a fixed pixel);
    // otherwise show the first full-scene mismatch.
    let (cx, cy) = if let Some(s) = std::env::var("FUZZ_DUMP_PX").ok() {
        let mut it = s.split(',').map(|v| v.trim().parse::<usize>().unwrap());
        (it.next().unwrap(), it.next().unwrap())
    } else {
        let Some(idx) = run.pixels.as_chunks::<4>().0.iter().zip(bgra.as_chunks::<4>().0).position(|(a, b)| a != b) else {
            println!("no mismatch"); return;
        };
        (idx % w, idx / w)
    };
    println!("seed {sv} size {w}x{h} ndiff={ndiff} maxdelta={maxd} first-mismatch ({cx},{cy})");
    let (x0, x1) = (cx.saturating_sub(4), (cx + 5).min(w));
    for y in cy.saturating_sub(2)..(cy + 3).min(h) {
        let cell = |p: &[u8]| format!("{:3},{:3},{:3},{:3}", p[0], p[1], p[2], p[3]);
        let o: Vec<String> = (x0..x1).map(|x| cell(&run.pixels[(y * w + x) * 4..][..4])).collect();
        let n: Vec<String> = (x0..x1).map(|x| cell(&bgra[(y * w + x) * 4..][..4])).collect();
        println!("y{y:3} OURS {}", o.join(" | "));
        println!("y{y:3} NATV {}", n.join(" | "));
    }
}

// Platform parity: a combined digest of driving gen_scene(seed) for seeds [start, start+n). Runs the
// SAME core on native and wasm (web/ calls this through its wasm export), so a digest mismatch is a
// per-platform divergence (a transcendental or float path the suite never exercised) = a fingerprint.
pub(crate) fn fuzz_digest(start: u64, n: u64) -> u64 {
    let factory = Factory::new(false);
    let device = factory.create_device();
    let mut h = std::hash::DefaultHasher::new();
    for seed in start..start + n {
        std::hash::Hasher::write_u64(&mut h, seed);
        match serde_json::from_value::<Scene>(d2d_reference::fuzz::gen_scene(seed)) {
            Ok(scene) => match drive(&device, &factory, &scene) {
                Ok(run) => std::hash::Hasher::write(&mut h, &run.pixels),
                Err(_) => std::hash::Hasher::write(&mut h, b"ERR"),
            },
            Err(_) => std::hash::Hasher::write(&mut h, b"BAD"),
        }
    }
    std::hash::Hasher::finish(&h)
}

// Benchmark/parity helper: drive one scene from its JSON and return the pixels (empty on failure), so
// the wasm build can time the identical core on the identical scene that the native bench uses.
#[cfg(target_arch = "wasm32")]
pub(crate) fn drive_json_pixels(json: &str) -> Vec<u8> {
    let Ok(scene) = serde_json::from_str::<Scene>(json) else { return Vec::new() };
    let factory = Factory::new(false);
    let device = factory.create_device();
    drive(&device, &factory, &scene).map(|r| r.pixels).unwrap_or_default()
}

// Full per-scene output fingerprint for the identity gate: the pixels, plus every op's HRESULT and
// tags, the final HRESULT and tags, and every recorded query result. Two builds that agree on this
// for every scene produced bit-identical output in every observable way — not merely "no new
// divergence" — which is the guarantee a performance or source-cleanup change must keep. Byte-defined
// hashing, so the value is stable across builds and platforms.
#[cfg(not(target_arch = "wasm32"))]
fn full_digest(run: &Run) -> u64 {
    use std::hash::Hasher;
    let mut h = std::hash::DefaultHasher::new();
    h.write(&run.pixels);
    h.write_usize(run.ops.len());
    for op in &run.ops {
        h.write(op.hr.as_bytes());
        h.write_u64(op.tag1);
        h.write_u64(op.tag2);
    }
    h.write(run.end.hr.as_bytes());
    h.write_u64(run.end.tag1);
    h.write_u64(run.end.tag2);
    h.write_usize(run.queries.len());
    for q in &run.queries {
        h.write_usize(q.op);
        h.write(q.kind.as_bytes());
        h.write(q.got.to_string().as_bytes());
    }
    h.finish()
}

// Identity gate, suite side: print one full-output digest per suite scene, in name order, so a base
// build and a candidate build can be compared line for line. Gated by D2D_SUITE_HASH.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn suite_hash() {
    if std::env::var_os("D2D_SUITE_HASH").is_none() {
        return;
    }
    for Case { name, scene, .. } in cases() {
        let scene: Scene = match serde_json::from_str(&scene) {
            Ok(s) => s,
            Err(_) => {
                println!("SHASH {name} badscene");
                continue;
            }
        };
        let factory = Factory::new(false);
        let device = factory.create_device();
        match drive(&device, &factory, &scene) {
            Ok(run) => println!("SHASH {name} {:#018x}", full_digest(&run)),
            Err(e) => println!("SHASH {name} err:{e}"),
        }
    }
}

// Native side of the fuzz parity check: print the digest for FUZZ_START..+FUZZ_N so it can be diffed
// against the wasm digest (web digest-fuzz). Gated by FUZZ_PARITY.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn fuzz_parity() {
    if std::env::var("FUZZ_PARITY").is_err() {
        return;
    }
    let start: u64 = std::env::var("FUZZ_START").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let n: u64 = std::env::var("FUZZ_N").ok().and_then(|s| s.parse().ok()).unwrap_or(2000);
    println!("FUZZ-PARITY-DIGEST {start} {n} {:#018x}", fuzz_digest(start, n));
}

// The wasm build has no font matcher, only the static `font_file` map, so it must name the same face
// and simulations the native matcher (`load_font_sims`) picks, or the two builds draw different glyphs.
// Checked over the families, weights, styles and stretches the corpus requests.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn font_map_matches_matcher() {
    let Some(dir) = fonts_dir() else { return };
    let mut bad = Vec::new();
    let fams = ["Arial", "Consolas", "Segoe UI", "Times New Roman", "Calibri", "Cambria", "Comic Sans MS", "Courier New",
        "Georgia", "Lucida Console", "Palatino Linotype", "Segoe UI Emoji", "Tahoma", "Trebuchet MS", "Verdana", "Bahnschrift", "Impact",
        "Segoe UI Symbol", "Yu Gothic UI", "Malgun Gothic", "Leelawadee UI", "Nirmala UI"];
    // The whole directory, and the eight-face set the r3-r22 cold captures built (`cold_faces`).
    let cold8: Vec<String> = ["arial", "arialbd", "consola", "consolab", "segoeui", "segoeuib", "times", "timesbd"].map(|f| format!("{f}.ttf")).into();
    for cold in [None, Some(&cold8[..])] {
        for fam in fams {
            for weight in [100, 300, 400, 600, 700, 900] {
                for style in 0..3 {
                    for stretch in 1..10 {
                        let found = match cold {
                            Some(c) => dwrite::resolve_face_sims_in(&c.iter().map(|n| dir.join(n)).collect::<Vec<_>>(), fam, weight, style, stretch),
                            None => dwrite::resolve_face_sims(&dir, fam, weight, style, stretch),
                        };
                        let nat = found.map(|(path, face, sims)| (path.file_name().unwrap().to_string_lossy().into_owned(), face, sims as u32));
                        let map = font_file_sims(fam, weight, style, cold).map(|(f, i, s)| (f.to_string(), i, s));
                        if nat.is_some() && map != nat {
                            bad.push(format!("{fam} w={weight} s={style} str={stretch} cold={}: matcher {nat:?} map {map:?}", cold.is_some()));
                        }
                    }
                }
            }
        }
    }
    // Font fallback's face for a character: the first face in fallback order that maps it, as wasm's
    // `load_fallback` takes it.
    let key_of = |f: &str| std::fs::read(dir.join(f)).ok().map(|b| d2d::text::fallback::set_order_key(&b));
    for fam in fams {
        for weight in [100, 300, 400, 600, 700, 900] {
            for style in 0..3 {
                for ch in ['A', '\u{645}', '\u{5D0}', '\u{E20}', '\u{D55C}', '\u{306E}', '\u{3A9}', '\u{41F}', '\u{1F600}'] {
                    let nat = dwrite::resolve_face_sims_covering(&dir, fam, weight, style, 5, ch).map(|(path, face, sims)| (path.file_name().unwrap().to_string_lossy().into_owned(), face, sims as u32));
                    let map = fallback_order(fam, weight, style, 5, key_of).and_then(|c| {
                        c.into_iter().find(|&(f, i, _)| std::fs::read(dir.join(f)).ok().is_some_and(|b| FontRef::from_index(&b, i).ok().and_then(|ft| ft.charmap().map(ch)).is_some()))
                    });
                    let map = map.map(|(f, i, s)| (f.to_string(), i, s));
                    if nat != map {
                        bad.push(format!("{fam} w={weight} s={style} {ch:?}: matcher {nat:?} map {map:?}"));
                    }
                }
            }
        }
    }
    assert!(bad.is_empty(), "font_file disagrees with the matcher:\n{}", bad.join("\n"));
}
