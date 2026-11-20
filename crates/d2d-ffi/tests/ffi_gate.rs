//! C-API completeness gate (group 5): a static scan of the `win/*` COM implementations that FAILS
//! on a method that silently does nothing — an empty body `{}`, or a draw that only handles a solid
//! brush and drops gradient/bitmap/image ones. Every genuinely-empty method must be in [`NOOP_ALLOW`]
//! with a native-behaviour reason (an advisory hint native also no-ops, or a consciously-deferred
//! decoder). A new stub, or a brush draw that regresses to solid-only, fails this test until it is
//! either implemented or justified — so "empty COM body" can never silently ship.

use std::path::{Path, PathBuf};

/// Empty-body COM methods that are legitimately no-ops, each with why native's behaviour permits it.
/// Keyed by method name (the same hint method on several objects shares one reason).
const NOOP_ALLOW: &[(&str, &str)] = &[
    // Verified no-ops: native's implementation only touches internal cache/hint/eviction state that
    // has no effect on the pixels, HRESULTs or query results we model, so an empty body is exact.
    ("SetCached", "advisory output-caching flag; d2d1 stores it on the render data only (no observable output)"),
    ("SetInstructionCountHint", "shader-linker budget hint: Direct2D stores hint/50 on the render data; our DXBC executor runs the shader directly, so no observable effect"),
    ("ClearResources", "Direct2D's ClearResources evicts the device's cached resources (memory only); we hold no such cache, so nothing to clear"),
];

/// `cast_object_ref::<SolidBrushObj>` sites that are intentionally solid-only. Keyed by `(file,
/// marker)` where `marker` is a distinctive substring of the enclosing method window — NOT a line
/// number, so an edit elsewhere in the file does not break the allowlist.
const SOLID_ONLY_ALLOW: &[(&str, &str, &str)] = &[
    // DrawSvgGlyphRun reads the solid brush only as the SVG `context-fill` foreground colour; an OT-SVG
    // glyph paints with its own explicit fills (the context colour is a fallback), so this is not a
    // brush bail — the run still draws. A gradient/bitmap `context-fill` is an unexercised niche.
    ("context.rs", "context-fill", "DrawSvgGlyphRun context-fill foreground colour, not a brush fill"),
];

fn win_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/win")
}

fn source_files() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(win_dir()).expect("win dir") {
        let p = e.unwrap().path();
        if p.extension().and_then(|x| x.to_str()) == Some("rs") {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            out.push((name, std::fs::read_to_string(&p).unwrap()));
        }
    }
    out.sort();
    out
}

/// Every `fn PascalCase(...) { <body> }` in `src`, as `(name, body_is_empty)`. A COM method is
/// PascalCase (the Windows vtable naming); our own helpers are snake_case and are skipped.
fn com_methods(src: &str) -> Vec<(String, bool)> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(rel) = src[i..].find("fn ") {
        let start = i + rel;
        i = start + 3;
        // Method name.
        let name_start = i;
        while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
            i += 1;
        }
        let name = &src[name_start..i];
        // PascalCase only (first char uppercase); skip whitespace then require '('.
        let mut j = i;
        while j < b.len() && b[j].is_ascii_whitespace() {
            j += 1;
        }
        if name.is_empty() || !name.chars().next().unwrap().is_ascii_uppercase() || j >= b.len() || b[j] != b'(' {
            continue;
        }
        // Balance the signature parens.
        let mut depth = 0i32;
        while j < b.len() {
            match b[j] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            j += 1;
        }
        // Skip to the body `{` (past an optional `-> ReturnType`); a `;` first ⇒ a trait decl, skip.
        j += 1;
        while j < b.len() && b[j] != b'{' && b[j] != b';' {
            j += 1;
        }
        if j >= b.len() || b[j] == b';' {
            continue;
        }
        // Empty iff the next non-whitespace after `{` is `}`.
        let mut k = j + 1;
        while k < b.len() && b[k].is_ascii_whitespace() {
            k += 1;
        }
        out.push((name.to_string(), k < b.len() && b[k] == b'}'));
        i = j + 1;
    }
    out
}

#[test]
fn no_unjustified_empty_com_bodies() {
    let mut offenders = Vec::new();
    for (file, src) in source_files() {
        for (name, empty) in com_methods(&src) {
            if empty && !NOOP_ALLOW.iter().any(|(n, _)| *n == name) {
                offenders.push(format!("{file}: fn {name}() has an empty body and is not in NOOP_ALLOW"));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "empty COM method bodies with no native-behaviour justification (implement them, or add to \
         NOOP_ALLOW with a reason):\n  {}",
        offenders.join("\n  ")
    );
}

#[test]
fn no_unjustified_solid_only_brush_bails() {
    // A brush draw that casts the brush to SolidBrushObj must also handle PixelBrushObj (gradient/
    // bitmap/image), unless the site is allowlisted. Heuristic: for each SolidBrushObj cast, a
    // PixelBrushObj must appear within the same method window; else it is a solid-only bail.
    let mut offenders = Vec::new();
    for (file, src) in source_files() {
        let mut from = 0;
        while let Some(rel) = src[from..].find("cast_object_ref::<SolidBrushObj>") {
            let at = from + rel;
            from = at + 1;
            // Window: the enclosing method, approximated as ±900 chars (methods here are small).
            let lo = at.saturating_sub(900);
            let hi = (at + 900).min(src.len());
            let window = &src[lo..hi];
            if window.contains("PixelBrushObj") {
                continue; // handles non-solid brushes too
            }
            if SOLID_ONLY_ALLOW.iter().any(|(f, marker, _)| *f == file && window.contains(marker)) {
                continue;
            }
            let line = src[..at].bytes().filter(|&c| c == b'\n').count() + 1;
            offenders.push(format!("{file}:{line}: SolidBrushObj cast with no PixelBrushObj branch (solid-only brush bail)"));
        }
    }
    assert!(
        offenders.is_empty(),
        "brush draws that handle only a solid brush (add the PixelBrushObj branch, or allowlist with \
         a reason):\n  {}",
        offenders.join("\n  ")
    );
}
