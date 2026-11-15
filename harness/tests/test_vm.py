import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vm


class RemotePaths(unittest.TestCase):
    def test_scoped_paths(self):
        for path in ("harness/scenes/a.json", r"C:\d2d-rs\harness\scenes\a.json"):
            self.assertEqual(vm.remote_path(path), r"C:\d2d-rs\harness\scenes\a.json")

    def test_escape_rejected(self):
        for path in ("../secret", r"C:\Windows", r"C:\d2d-rs-other\a", r"\Windows",
                     r"D:relative", r"\\server\share", "a:stream", "a/../b", "a."):
            with self.subTest(path=path), self.assertRaises(ValueError):
                vm.remote_path(path)


if __name__ == "__main__":
    unittest.main()
