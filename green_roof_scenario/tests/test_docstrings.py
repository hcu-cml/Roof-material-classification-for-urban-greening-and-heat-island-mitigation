from __future__ import annotations

import ast
import unittest
from pathlib import Path


class DocstringCoverageTest(unittest.TestCase):
    def test_all_source_functions_have_docstrings(self) -> None:
        """Require docstrings on every function and method in the package."""
        source_root = Path(__file__).parents[1] / "src" / "green_roof_scenario"
        missing: list[str] = []

        for path in sorted(source_root.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if ast.get_docstring(node) is None:
                        missing.append(f"{path.name}:{node.lineno}:{node.name}")

        self.assertEqual(missing, [], "Missing function docstrings:\n" + "\n".join(missing))


if __name__ == "__main__":
    unittest.main()
