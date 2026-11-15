"""Use installed Rust sources without changing the read-only toolchain or its lockfile."""
import re
import shutil
import subprocess
from pathlib import Path

root = Path(subprocess.check_output(["rustc", "+1.93", "--print", "sysroot"], text=True).strip())
dest = Path("target/rust-src")
if not dest.exists():
    shutil.copytree(root / "lib/rustlib/src/rust", dest)
# The std lockfile requests uncached memchr 2.7.6; 2.8.3 satisfies its constraint.
pattern = r'\[\[package\]\]\nname = "memchr".*?(?=\n\[\[package\]\]|\Z)'
entry = re.search(pattern, Path("Cargo.lock").read_text(), re.S)
assert entry and 'version = "2.8.3"' in entry[0]
lock = dest / "library/Cargo.lock"
updated, count = re.subn(pattern, entry[0], lock.read_text(), flags=re.S)
assert count == 1
lock.write_text(updated)
