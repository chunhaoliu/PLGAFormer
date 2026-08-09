import importlib.util
import io
import sys
import unittest
from pathlib import Path


class Exp3VisualizeTests(unittest.TestCase):
    def test_configure_console_for_unicode_handles_gbk_stdout(self):
        project_root = Path(__file__).resolve().parents[1]
        module_path = project_root / "experiments" / "exp3_robustness" / "visualize_results.py"
        spec = importlib.util.spec_from_file_location("exp3_visualize_results", module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp936", errors="strict")
        stderr = io.TextIOWrapper(io.BytesIO(), encoding="cp936", errors="strict")
        try:
            sys.stdout = stdout
            sys.stderr = stderr
            module.configure_console_for_unicode()
            print("📂 ✅ ❌ 鲁棒性")
            sys.stdout.flush()
            self.assertIn("utf", (sys.stdout.encoding or "").lower())
            self.assertIn("utf", (sys.stderr.encoding or "").lower())
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    unittest.main()
