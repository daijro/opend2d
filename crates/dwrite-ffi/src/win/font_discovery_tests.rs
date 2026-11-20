

#[test]
fn linux_collection_discovers_fontconfig_and_recursive_xdg_fonts() {
    if std::env::var_os("D2D_DISCOVERY_CHILD").is_some() {
        use windows::Win32::Graphics::DirectWrite::*;
        use windows_core::Interface;
        unsafe {
            let mut raw = std::ptr::null_mut();
            super::DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, &IDWriteFactory::IID, &mut raw).ok().unwrap();
            let factory = IDWriteFactory::from_raw(raw);
            let mut collection = None;
            factory.GetSystemFontCollection(&mut collection, false).unwrap();
            let collection = collection.unwrap();
            let mut names = Vec::new();
            for i in 0..collection.GetFontFamilyCount() {
                let strings = collection.GetFontFamily(i).unwrap().GetFamilyNames().unwrap();
                let mut text = vec![0u16; strings.GetStringLength(0).unwrap() as usize + 1];
                strings.GetString(0, &mut text).unwrap();
                names.push(String::from_utf16_lossy(&text[..text.len()-1]));
            }
            if std::env::var_os("D2D_DISCOVERY_NO_FC").is_none() { assert!(names.iter().any(|n| n == "Arial"), "fontconfig: {names:?}"); }
            assert!(names.iter().any(|n| n == "Times New Roman"), "recursive XDG: {names:?}");
            assert!(names.iter().position(|n| n == "Arial") < names.iter().position(|n| n == "Times New Roman"));

        }
        return;
    }
    use std::os::unix::fs::{symlink, PermissionsExt};
    let Some(fonts) = std::env::var_os("D2D_FONTS") else { return };
    let fonts = std::path::Path::new(&fonts);
    if !fonts.join("arial.ttf").exists() || !fonts.join("times.ttf").exists() { return; }
    let root = std::env::temp_dir().join(format!("d2d-font-discovery-{}", std::process::id()));
    std::fs::create_dir_all(root.join("bin")).unwrap();
    std::fs::create_dir_all(root.join("data/fonts/deep/nested")).unwrap();
    symlink(fonts.join("times.ttf"),root.join("data/fonts/deep/nested/Times.ttf")).unwrap();
    symlink(root.join("data/fonts"),root.join("data/fonts/deep/loop")).unwrap();
    let fc = root.join("bin/fc-list");
    std::fs::write(&fc,format!("#!/bin/sh\nprintf '%s\\n' '{}' '{}'\n",fonts.join("arial.ttf").display(),fonts.join("arial.ttf").display())).unwrap();
    std::fs::set_permissions(&fc,std::fs::Permissions::from_mode(0o755)).unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["linux_collection_discovers_fontconfig_and_recursive_xdg_fonts","--nocapture"])
        .env_remove("D2D_FONTS").env("D2D_DISCOVERY_CHILD","1")
        .env("PATH",root.join("bin")).env("HOME",root.join("home"))
        .env("XDG_DATA_HOME",root.join("data")).env("XDG_DATA_DIRS",root.join("empty"))
        .output().unwrap();
    assert!(output.status.success(),"{}{}",String::from_utf8_lossy(&output.stdout),String::from_utf8_lossy(&output.stderr));
    std::fs::remove_file(fc).unwrap();
    let fallback = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["linux_collection_discovers_fontconfig_and_recursive_xdg_fonts","--nocapture"])
        .env_remove("D2D_FONTS").env("D2D_DISCOVERY_CHILD","1").env("D2D_DISCOVERY_NO_FC","1")
        .env("PATH",root.join("bin")).env("HOME",root.join("home"))
        .env("XDG_DATA_HOME",root.join("data")).env("XDG_DATA_DIRS",root.join("empty"))
        .output().unwrap();
    std::fs::remove_dir_all(root).unwrap();
    assert!(fallback.status.success(),"{}{}",String::from_utf8_lossy(&fallback.stdout),String::from_utf8_lossy(&fallback.stderr));
}
