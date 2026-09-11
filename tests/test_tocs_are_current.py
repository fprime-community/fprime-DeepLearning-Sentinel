"""The tables of contents are generated, and they still match their documents.

`docs/MODELS.md` carries 369 numbered headings and `docs/DECISIONS.md` 69 entries.
A hand-maintained index of that size drifts, and the drift is invisible until
somebody follows a line that is wrong -- which is exactly how
`docs/TELEMANOM_EXCERPTS.md` came to be missing fifteen locations while claiming
to be generated. So these are regenerated and pinned, and this is the pin.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_tocs as T  # noqa: E402

TARGETS = [("docs/MODELS.md", T.models_toc), ("docs/DECISIONS.md", T.decisions_toc)]


@pytest.mark.parametrize("name,builder", TARGETS)
def test_the_committed_toc_is_what_the_generator_produces(name, builder) -> None:
    path = ROOT / name
    text = path.read_text(encoding="utf-8")
    assert T.BEGIN in text and T.END in text, f"{name} has no generated table of contents"
    body, _ = builder(text)
    committed = text[text.index(T.BEGIN):text.index(T.END)]
    for line in body.splitlines():
        assert line in committed, f"{name}'s table of contents is missing: {line}"


@pytest.mark.parametrize("name,builder", TARGETS)
def test_every_entry_points_at_a_heading_that_exists(name, builder) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")
    body, _ = builder(text)
    assert body.strip(), f"{name}'s table of contents is empty"
    assert body.count("\n") + 1 > 50, f"{name} indexes suspiciously few headings"


def test_the_two_out_of_order_sections_are_reported_not_hidden() -> None:
    """33.6 follows 33.8 and 34.7 follows 34.8, in the document itself.

    Neither is moved: `docs/MODELS.md`'s numbering is cited from hundreds of
    places, and a table of contents is not a reason to break one. What the
    generator must do is **say so**, so a reader who notices is not left thinking
    the index is wrong.
    """
    _, notes = T.models_toc((ROOT / "docs" / "MODELS.md").read_text(encoding="utf-8"))
    joined = "; ".join(notes)
    assert "33.6 follows 33.8" in joined, notes
    assert "34.7 follows 34.8" in joined, notes
    assert "26.6 appears twice" in joined, notes
    text = (ROOT / "docs" / "MODELS.md").read_text(encoding="utf-8")
    assert "the document wins" in text, "the departure is not disclosed in the document"


def test_every_observed_block_is_at_the_same_heading_level() -> None:
    """35.7 sat at `##` where every other OBSERVED block sits at `###`.

    It read as a top-level section in any generated index, which is what made it
    one of the four defects `docs/REORG_PLAN.md` tranche 3 names.
    """
    bad = [line for line in (ROOT / "docs" / "MODELS.md").read_text(encoding="utf-8").splitlines()
           if line.startswith("## ") and "OBSERVED" in line]
    assert bad == [], f"OBSERVED blocks promoted to a top-level heading: {bad}"


def test_regenerating_is_a_no_op() -> None:
    proc = subprocess.run([sys.executable, "scripts/make_tocs.py", "--check"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
