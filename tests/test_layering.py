"""The referee must not know what the players are.

Models import `sentinel_eval.detector`; the harness never imports a model. That
one-way dependency is what lets work items 4, 5 and 6 add LSTM, GRU and TCN
without touching a line of scoring code -- and what stops any architecture being
quietly special-cased.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "src" / "sentinel_eval"
COMPOSITION_ROOT = "cli.py"       # the one module allowed to know about both


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("path", sorted(HARNESS.rglob("*.py")), ids=lambda p: p.name)
def test_the_harness_does_not_import_a_model(path):
    if path.name == COMPOSITION_ROOT:
        return
    offenders = {n for n in _imports(path) if n.startswith("sentinel_models")}
    assert not offenders, f"{path.name} imports {offenders}; only {COMPOSITION_ROOT} may"


def test_the_composition_root_imports_models_lazily():
    """Even the CLI resolves detectors inside functions, not at module import."""
    tree = ast.parse((HARNESS / COMPOSITION_ROOT).read_text())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", "") or ""
            assert not module.startswith("sentinel_models")


def test_models_depend_on_the_published_contract_only():
    for path in (ROOT / "src" / "sentinel_models").rglob("*.py"):
        for name in _imports(path):
            if name.startswith("sentinel_eval"):
                assert name in ("sentinel_eval.detector",), f"{path.name} imports {name}"


VENDORED = ROOT / "third_party"


@pytest.mark.parametrize(
    "path",
    sorted(list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))),
    ids=lambda p: p.name,
)
def test_nothing_imports_the_vendored_source(path):
    """`third_party/telemanom` is evidence, not a dependency.

    It is committed so that every divergence this project claims against the
    published method cites a file a reader can open, rather than a line number
    someone remembers (D53). Importing it would make it a dependency, put an
    unmaintained package on the flight-adjacent path, and -- worse -- let a
    reproduction quietly become a re-export.
    """
    offenders = {n for n in _imports(path) if n.split(".")[0] == "third_party"}
    assert not offenders, f"{path.name} imports {offenders}; it is evidence, never a dependency"


def test_the_vendored_source_is_present_and_pinned():
    """A citation whose original is missing is an assertion with a line number."""
    assert (VENDORED / "telemanom" / "telemanom" / "errors.py").is_file()
    provenance = (VENDORED / "telemanom" / "PROVENANCE.md").read_text()
    assert "2e6c5b6c3558e7835601519b7bdef37c649bdbdc" in provenance
    assert "BSD 3-Clause" in provenance
