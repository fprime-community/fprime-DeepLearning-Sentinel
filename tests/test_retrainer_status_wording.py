"""D83, D84: the retrainer's status is one sentence, and nothing live may say more.

D83 fixed the wording -- *"the chosen retraining implementation, host-verified, not
flight-qualified"* -- and made the case against travel beside it. D84 exports the
component, which makes overclaiming easier: "exported" invites "qualified". Until
now nothing enforced either. Two checks:

1. **No live document calls the retrainer certified, qualified or flight-ready.**
   Every sentence that names the retrainer or OxCaml is read; the only permitted
   uses of those words are the ones the case against itself uses ("not
   flight-qualified", "no qualified compiler", "Ferrocene qualified", "a qualified
   toolchain") and the certification precedent the case against denies.
2. **`master:README.md`'s OPTIONAL retrainer section carries D83's status line
   verbatim and every row of the case against**, because that is the section a
   mission reads before switching it on.

Live documents only: the records (`docs/DECISIONS.md`, `docs/MODELS.md`,
`CHANGELOG.md`, `docs/PHASE5.md`) are never edited, and quote refused wordings on
purpose. Both checks are shown failing on a mutated text below.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

STATUS_LINE = "the chosen retraining implementation, host-verified, not flight-qualified"

#: Live documents on dev, read from disk.
DEV_LIVE = ("README.md", "docs/STATUS.md", "docs/datasets/REPRODUCING.md",
            "fprime/README.md", "fprime/SentinelRef/README.md",
            "fprime/SentinelRetrain/README.md", "fprime/Sentinel/Retrainer/docs/sdd.md",
            "fprime/Sentinel/Monitor/docs/sdd.md")
#: Live documents on master, read with git show.
MASTER_LIVE = ("README.md", "docs/DESIGN.md", "docs/STATUS.md", "docs/EVIDENCE.md",
               "docs/OMISSIONS.md", "fprime/README.md", "fprime/SentinelRetrain/README.md",
               "fprime/Sentinel/Retrainer/docs/sdd.md")

SUBJECT = re.compile(r"retrain|oxcaml", re.I)
OVERCLAIM = re.compile(r"\b(certified|flight[- ]ready|flight[- ]qualified|qualified)\b", re.I)

#: The case against uses these words to DENY the status. Nothing else may use them.
PERMITTED = (
    re.compile(r"not flight[- ]qualified", re.I),
    re.compile(r"no qualified (compiler|toolchain)", re.I),
    re.compile(r"ferrocene(,)? qualified|qualified toolchain in ferrocene|"
               r"qualified (compiler|toolchain) in ferrocene", re.I),
    re.compile(r"not (certified|flight[- ]ready|qualified)", re.I),
)

#: Each row of D83's case against, as a phrase the OPTIONAL section must contain.
CASE_AGAINST = ("flight heritage", "qualified compiler", "garbage-collected",
                "arm64 macOS", "stability", "Ferrocene", "no Rust comparison")


def _sentences(text: str) -> list[str]:
    flat = " ".join(text.split())
    return re.split(r"(?<=[.!?])\s+(?=[A-Z*`(])", flat)


def _overclaims(text: str) -> list[str]:
    out = []
    for sentence in _sentences(text):
        if not SUBJECT.search(sentence):
            continue
        scrubbed = sentence
        for allowed in PERMITTED:
            scrubbed = allowed.sub("", scrubbed)
        if OVERCLAIM.search(scrubbed):
            out.append(sentence[:240])
    return out


def _master(name: str) -> str | None:
    probe = subprocess.run(["git", "rev-parse", "--verify", "master"], cwd=ROOT,
                           capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("no `master` ref in this clone")
    shown = subprocess.run(["git", "show", f"master:{name}"], cwd=ROOT,
                           capture_output=True, text=True)
    return shown.stdout if shown.returncode == 0 else None


@pytest.mark.parametrize("name", DEV_LIVE)
def test_no_dev_document_overclaims_the_retrainer(name: str) -> None:
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} is not on this branch")
    bad = _overclaims(path.read_text(encoding="utf-8"))
    assert not bad, (f"{name} describes the retrainer beyond D83's status line "
                     f"('{STATUS_LINE}'): {bad}")


@pytest.mark.parametrize("name", MASTER_LIVE)
def test_no_master_document_overclaims_the_retrainer(name: str) -> None:
    text = _master(name)
    if text is None:
        pytest.skip(f"master:{name} does not exist")
    bad = _overclaims(text)
    assert not bad, f"master:{name} describes the retrainer beyond D83's status line: {bad}"


def _optional_section(readme: str) -> str | None:
    m = re.search(r"^#{2,4} [^\n]*OPTIONAL[^\n]*\n(.*?)(?=^#{2} |\Z)", readme, re.M | re.S)
    return m.group(0) if m else None


def _section_faults(section: str | None) -> list[str]:
    if section is None:
        return ["no heading labelled OPTIONAL"]
    flat = " ".join(section.split())
    faults = []
    if STATUS_LINE not in flat:
        faults.append(f"D83's status line, verbatim: '{STATUS_LINE}'")
    faults += [f"the case against's row '{row}'" for row in CASE_AGAINST if row not in flat]
    if "not telemetry" not in flat:
        faults.append("that it trains on a drive that is not telemetry (D84 c.3)")
    if "swapped in" not in flat:
        faults.append("that no candidate may be swapped in operationally (D84 c.2)")
    return faults


def test_the_optional_section_carries_the_status_and_the_case_against() -> None:
    readme = _master("README.md")
    assert readme is not None
    faults = _section_faults(_optional_section(readme))
    assert not faults, f"master:README.md's OPTIONAL retrainer section lacks {faults}"


def test_both_checks_can_fail() -> None:
    """Negative directions, on mutated text."""
    assert _overclaims("The OxCaml retrainer is flight-qualified.")
    assert _overclaims("The retrainer is certified for flight.")
    assert _overclaims("Retraining is flight-ready on arm64.")
    assert not _overclaims("OxCaml is the chosen retraining implementation, host-verified, "
                           "not flight-qualified.")
    assert not _overclaims("The case against OxCaml: no qualified compiler, and Ferrocene "
                           "qualified for Rust.")
    good = ("### OPTIONAL: the retrainer\n" + STATUS_LINE + ". No flight heritage, no "
            "qualified compiler, no precedent for a garbage-collected runtime, arm64 macOS "
            "only, no stability promise, Ferrocene, no Rust comparison. It trains on a drive "
            "that is not telemetry; no candidate may be swapped in operationally.\n")
    assert not _section_faults(_optional_section(good))
    assert _section_faults(_optional_section(good.replace("Ferrocene", "Rust")))
    assert _section_faults(_optional_section(good.replace("not flight-qualified", "qualified")))
    assert _section_faults(_optional_section(good.replace("OPTIONAL", "Optional")))
