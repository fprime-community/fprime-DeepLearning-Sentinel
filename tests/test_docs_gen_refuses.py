"""The generator refuses to discard prose it did not write.

`docs/DATA.md` is generated from the manifest on every ingest, and it also
carries hand-added prose: section 1's SMAP/MSL block, and three `(!)` riders
recording provenance corrections and a labelling defect. `write_data_md` used to
end in a bare `path.write_text(text)`, so the next ingest would have deleted all
of it and said nothing. The file warned about that in its own text, which is not
a guard -- it is a note asking the next person to remember.

Four behaviours, and the last two are the point: a delimited region survives a
regeneration verbatim, an edit inside one is a person's business and is left
alone, and anything else stops the write instead of being overwritten.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentinel_data import docs_gen as G

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def manifest() -> dict:
    """The committed snapshot. No bucket read, no operation spent."""
    return json.loads((ROOT / "docs" / "manifest.snapshot.json").read_text())


@pytest.fixture
def generated(manifest, tmp_path: Path) -> Path:
    return G.write_data_md(manifest, "fprime-sentinel-data", tmp_path)


def test_a_fresh_generation_is_stamped_and_self_consistent(generated: Path) -> None:
    text = generated.read_text(encoding="utf-8")
    recorded = G._DIGEST_RE.search(text)
    assert recorded, "a generated document must carry its own digest footer"
    assert recorded.group(1) == G._digest(text)


def test_regenerating_over_its_own_output_is_byte_identical(manifest, generated) -> None:
    """Idempotence is what makes the digest usable as a guard at all."""
    before = generated.read_bytes()
    G.write_data_md(manifest, "fprime-sentinel-data", generated.parent)
    assert generated.read_bytes() == before


def test_a_delimited_region_survives_a_regeneration_verbatim(manifest, generated) -> None:
    text = generated.read_text(encoding="utf-8")
    anchor = "## 2. Where it lives"
    assert text.count(anchor) == 1
    block = f"{G.HAND_OPEN}\n\nHand-added prose that must not be lost.\n\n{G.HAND_CLOSE}"
    generated.write_text(G._stamped(text.replace(anchor, f"{anchor}\n\n{block}", 1)),
                         encoding="utf-8")

    G.write_data_md(manifest, "fprime-sentinel-data", generated.parent)

    after = generated.read_text(encoding="utf-8")
    assert "Hand-added prose that must not be lost." in after
    assert after.count(G.HAND_OPEN) == 1 and after.count(G.HAND_CLOSE) == 1
    assert G._DIGEST_RE.search(after).group(1) == G._digest(after)


def test_an_edit_inside_a_delimited_region_does_not_trip_the_guard(manifest, generated) -> None:
    """Those regions are the parts a person is invited to edit.

    If they counted towards the digest, correcting a typo inside one would make
    the next ingest refuse, and the lesson everyone would take is to delete the
    footer.
    """
    text = generated.read_text(encoding="utf-8")
    anchor = "## 2. Where it lives"
    block = f"{G.HAND_OPEN}\n\nFirst wording.\n\n{G.HAND_CLOSE}"
    generated.write_text(G._stamped(text.replace(anchor, f"{anchor}\n\n{block}", 1)),
                         encoding="utf-8")
    generated.write_text(
        generated.read_text(encoding="utf-8").replace("First wording.", "Second wording."),
        encoding="utf-8")

    G.write_data_md(manifest, "fprime-sentinel-data", generated.parent)

    assert "Second wording." in generated.read_text(encoding="utf-8")


def test_an_undelimited_edit_is_refused_and_survives_on_disk(manifest, generated) -> None:
    """The defect this whole module exists for: refuse, do not discard."""
    edited = generated.read_text(encoding="utf-8") + "\nAn undelimited hand edit.\n"
    generated.write_text(edited, encoding="utf-8")

    with pytest.raises(G.HandAuthoredContentWouldBeLost, match="edited since"):
        G.write_data_md(manifest, "fprime-sentinel-data", generated.parent)

    assert "An undelimited hand edit." in generated.read_text(encoding="utf-8")


def test_a_document_with_no_footer_is_refused_rather_than_assumed(manifest, generated) -> None:
    """Without a digest the generator cannot tell generated prose from added."""
    generated.write_text(G._DIGEST_RE.sub("", generated.read_text(encoding="utf-8")),
                         encoding="utf-8")

    with pytest.raises(G.HandAuthoredContentWouldBeLost, match="no generated-digest"):
        G.write_data_md(manifest, "fprime-sentinel-data", generated.parent)


def test_an_unanchorable_region_is_named_rather_than_dropped(manifest, generated) -> None:
    """A region whose preceding line the generator no longer emits."""
    text = generated.read_text(encoding="utf-8")
    block = f"{G.HAND_OPEN}\n\nOrphaned prose.\n\n{G.HAND_CLOSE}"
    generated.write_text(
        G._stamped(f"{text}\n\nA line no generation will ever emit.\n\n{block}\n"),
        encoding="utf-8")

    with pytest.raises(G.HandAuthoredContentWouldBeLost, match="cannot be re-anchored"):
        G.write_data_md(manifest, "fprime-sentinel-data", generated.parent)

    assert "Orphaned prose." in generated.read_text(encoding="utf-8")


def test_the_committed_data_md_is_armed(manifest) -> None:
    """The real file carries a footer and a delimited region, so an ingest is safe."""
    text = (ROOT / "docs" / "DATA.md").read_text(encoding="utf-8")
    assert G._DIGEST_RE.search(text), "docs/DATA.md has no generated-digest footer"
    assert G._DIGEST_RE.search(text).group(1) == G._digest(text)
    assert text.count(G.HAND_OPEN) == 1 and text.count(G.HAND_CLOSE) == 1
    assert "**SMAP/MSL is now ingested too," in text
