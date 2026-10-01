"""Every Core name that another file imports must still exist.

This static check needs no GPU stack. It parses each Core module for its top-level names, then
verifies every `from module import name` and every `module.attribute` use in Core itself
(including imports inside functions), in the suites here, in the notebooks, and, when present, in
the manuscript scripts. A rename or removal that breaks a caller fails here before any run does.
"""

from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path

CODEBASE = Path(__file__).resolve().parents[1]
CORE = CODEBASE / "Core"
CONSUMERS = [
    ("Core", sorted(CORE.glob("*.py"))),
    ("Tests", sorted((CODEBASE / "Tests").glob("*.py"))),
    ("Notebook", sorted((CODEBASE / "Notebook").glob("*.ipynb"))),
    ("Context/Misc_Scripts", sorted((CODEBASE.parent / "Context" / "Misc_Scripts").glob("*.py"))),
]


def top_level_names(statements) -> set[str]:
    names: set[str] = set()
    for node in statements:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
        elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
            names |= top_level_names(node.body + getattr(node, "orelse", []))
            for handler in getattr(node, "handlers", []):
                names |= top_level_names(handler.body)
            names |= top_level_names(getattr(node, "finalbody", []))
    return names


def module_surface() -> dict[str, tuple[set[str], bool]]:
    surface = {}
    for path in sorted(CORE.glob("*.py")):
        tree = ast.parse(path.read_text())
        dynamic = any(isinstance(n, ast.FunctionDef) and n.name == "__getattr__" for n in tree.body)
        surface[path.stem] = (top_level_names(tree.body), dynamic)
    return surface


def source_of(path: Path) -> str:
    if path.suffix != ".ipynb":
        return path.read_text()
    cells = json.loads(path.read_text())["cells"]
    code = "\n\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")
    return re.sub(r"^\s*[%!].*$", "", code, flags=re.M)


class PublicSurfaceTests(unittest.TestCase):
    def test_imported_core_names_exist(self):
        surface = module_surface()
        missing: list[str] = []
        checked = 0
        for group, paths in CONSUMERS:
            for path in paths:
                tree = ast.parse(source_of(path))
                aliases = {}
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in surface:
                        for alias in node.names:
                            if alias.name != "*":
                                checked += 1
                                if alias.name not in surface[node.module][0] and not surface[node.module][1]:
                                    missing.append(f"{group}/{path.name}: from {node.module} import {alias.name}")
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name in surface:
                                aliases[alias.asname or alias.name] = alias.name
                attribute_nodes = [
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases
                ]
                assigned = {(node.value.id, node.attr) for node in attribute_nodes if isinstance(node.ctx, (ast.Store, ast.Del))}
                for node in attribute_nodes:
                    if node.attr.startswith("__") or (node.value.id, node.attr) in assigned:
                        continue
                    module = aliases[node.value.id]
                    checked += 1
                    if node.attr not in surface[module][0] and not surface[module][1]:
                        missing.append(f"{group}/{path.name}: {module}.{node.attr}")
        self.assertGreater(checked, 100, "the surface check looked at suspiciously few names")
        self.assertEqual(missing, [], "\n".join(sorted(set(missing))))
        print(f"checked {checked} imported or accessed Core names")

    def test_hydra_target_modules_exist(self):
        text = (CORE / "config.py").read_text() + (CORE / "prompt_granularity.py").read_text()
        surface = module_surface()
        targets = set(re.findall(r"""["']([a-z_][a-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_.]*)["']""", text))
        local = sorted((module, name) for module, name in targets if module in surface)
        self.assertTrue(local, "no local _target_ strings were found")
        broken = [f"{module}.{name}" for module, name in local if name.split(".")[0] not in surface[module][0]]
        self.assertEqual(broken, [], f"Hydra targets that no longer resolve: {broken}")


if __name__ == "__main__":
    unittest.main(verbosity=1)
