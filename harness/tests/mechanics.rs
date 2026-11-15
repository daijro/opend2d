use d2d_reference::*;
use std::{fs, path::PathBuf, process::Command};

#[test]
fn every_scene_parses_and_validates() {
    let Some(suite) = suite_dir() else { return };
    let mut count = 0;
    for entry in fs::read_dir(suite.join("scenes")).unwrap() {
        let path = entry.unwrap().path();
        // shader-ref's scenes (shader-*, shader2-*) use its own format, not the scene format.
        let shader_ref = path.file_name().is_some_and(|n| n.to_string_lossy().starts_with("shader"));
        if path.extension().is_some_and(|s| s == "json") && !shader_ref {
            read_scene(&path).unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
            count += 1;
        }
    }
    assert!(count > 0, "no scenes found");
}

#[test]
fn malformed_scene_is_rejected() {
    let dir = scratch("invalid");
    let path = dir.join("scene.json");
    for text in [
        r#"{"version":3,"size":[1,1],"dpi":96,"ops":[]}"#,
        r#"{"version":1,"size":[0,1],"dpi":96,"ops":[]}"#,
        r#"{"version":1,"size":[1,1],"dpi":96,"typo":1,"ops":[]}"#,
        // A top-level frame may end unbalanced, but a command list's may not.
        r#"{"version":1,"size":[1,1],"dpi":96,"ops":[{"op":"command_list","id":"c","ops":[{"op":"pop_clip"}]}]}"#,
        r#"{"version":1,"size":[1,1],"dpi":96,"ops":[{"op":"clear","color":[0,0,0,0],"typo":1}]}"#,
        r#"{"version":1,"size":[1,1],"dpi":96,"ops":[{"op":"bitmap","id":"b","size":[2,1],"rgba":[0,0,0,0]}]}"#,
    ] {
        fs::write(&path, text).unwrap();
        assert!(read_scene(&path).is_err(), "accepted {text}");
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn alpha_preview_and_diff_have_known_values() {
    assert_eq!(
        straight_rgba(&[16, 32, 48, 64, 10, 20, 30, 0]),
        [191, 128, 64, 64, 0, 0, 0, 0]
    );
    let dir = scratch("diff");
    let a = dir.join("a.png");
    let b = dir.join("b.png");
    let diff = dir.join("diff.png");
    write_png(&a, [2, 1], &[0, 10, 20, 255, 100, 120, 140, 128]).unwrap();
    write_png(&b, [2, 1], &[5, 10, 18, 250, 100, 128, 140, 128]).unwrap();
    let run = |x: &PathBuf, y: &PathBuf, raw: bool| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_png-diff"));
        cmd.args([x, y, &diff]);
        if raw {
            cmd.arg("--bgra");
        }
        cmd.output().unwrap()
    };
    let changed = run(&a, &b, false);
    assert_eq!(changed.status.code(), Some(1));
    let stats: serde_json::Value = serde_json::from_slice(&changed.stdout).unwrap();
    assert_eq!(stats["max"], 8);
    assert_eq!(stats["mean"], 2.5);
    assert_eq!(stats["channel_max"], serde_json::json!([5, 8, 2, 5]));
    assert_eq!(stats["changed_pixels"], 2);
    assert_eq!(read_png(&diff).unwrap().1, [5, 5, 5, 255, 0, 8, 0, 255]);
    assert!(run(&a, &a, false).status.success());
    fs::write(a.with_extension("bgra"), [0u8; 8]).unwrap();
    fs::write(b.with_extension("bgra"), [0u8, 0, 0, 0, 0, 0, 1, 0]).unwrap();
    let raw = run(&a, &b, true);
    assert_eq!(raw.status.code(), Some(1));
    let stats: serde_json::Value = serde_json::from_slice(&raw.stdout).unwrap();
    assert_eq!(stats["max"], 1);
    assert_eq!(stats["mean"], 0.125);
    assert_eq!(read_png(&diff).unwrap().1, [0, 0, 0, 255, 1, 0, 0, 255]);
    fs::write(b.with_extension("bgra"), [0u8]).unwrap();
    assert!(!run(&a, &b, true).status.success());
    write_png(&b, [1, 1], &[0; 4]).unwrap();
    assert!(!run(&a, &b, false).status.success());
    fs::remove_dir_all(dir).unwrap();
}

fn scratch(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("d2d-harness-{}-{name}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}
