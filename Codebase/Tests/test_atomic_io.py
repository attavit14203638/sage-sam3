"""`atomic_write_text` must match `Path.write_text` byte for byte and never leave a partial file."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parents[1] / "Core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from atomic_io import atomic_write_text  # noqa: E402


class AtomicWriteTests(unittest.TestCase):
    def test_bytes_and_mode_match_write_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, written = Path(tmp) / "reference.json", Path(tmp) / "written.json"
            text = '{"score": 0.5, "name": "tree", "note": "caf\u00e9"}\n'
            reference.write_text(text, encoding="utf-8")
            atomic_write_text(written, text)
            self.assertEqual(written.read_bytes(), reference.read_bytes())
            self.assertEqual(written.stat().st_mode, reference.stat().st_mode)

    def test_overwrites_in_place_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "result.json"
            target.write_text("old")
            atomic_write_text(target, "new")
            self.assertEqual(target.read_text(), "new")
            self.assertEqual(sorted(path.name for path in Path(tmp).iterdir()), ["result.json"])

    def test_failed_write_keeps_the_old_file_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "result.json"
            target.write_text("old")
            with self.assertRaises(TypeError):
                atomic_write_text(target, 123)  # type: ignore[arg-type]
            self.assertEqual(target.read_text(), "old")
            self.assertEqual(sorted(path.name for path in Path(tmp).iterdir()), ["result.json"])

    def test_missing_parent_directory_raises_like_write_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                atomic_write_text(Path(tmp) / "missing" / "result.json", "x")
            self.assertEqual(os.listdir(tmp), [])


if __name__ == "__main__":
    unittest.main(verbosity=1)
