//! Linux font discovery. Fontconfig supplies configured files; XDG roots supply uncached fonts.
use std::path::{Path, PathBuf};
use std::collections::HashSet;

fn walk(dir: &Path, seen: &mut HashSet<PathBuf>, files: &mut Vec<PathBuf>) {
    let Ok(key) = dir.canonicalize() else { return };
    if !seen.insert(key) { return; }
    let mut entries: Vec<_> = std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() { walk(&p, seen, files); }
        else if p.is_file() { files.push(p); }
    }
}

pub(super) fn paths() -> Vec<PathBuf> {
    let configured = std::process::Command::new("fc-list").arg("--format=%{file}\n").output().ok()
        .filter(|o| o.status.success()).map(|o| o.stdout).unwrap_or_default();
    let mut files: Vec<PathBuf> = String::from_utf8_lossy(&configured).lines().filter(|p| !p.is_empty()).map(PathBuf::from).collect();
    files.sort();
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let data_home = std::env::var_os("XDG_DATA_HOME").filter(|p| !p.is_empty()).map(PathBuf::from)
        .or_else(|| home.as_ref().map(|p| p.join(".local/share")));
    let mut roots = Vec::new();
    if let Some(p) = data_home { roots.push(p.join("fonts")); }
    if let Some(p) = home { roots.push(p.join(".fonts")); }
    let dirs = std::env::var_os("XDG_DATA_DIRS").filter(|p| !p.is_empty()).unwrap_or_else(|| "/usr/local/share:/usr/share".into());
    roots.extend(std::env::split_paths(&dirs).map(|p| p.join("fonts")));
    let mut seen_dirs = HashSet::new();
    for root in roots { walk(&root, &mut seen_dirs, &mut files); }
    let mut seen_files = HashSet::new();
    files.retain(|p| p.canonicalize().ok().is_some_and(|key| seen_files.insert(key)));
    files
}
