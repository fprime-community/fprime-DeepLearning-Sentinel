"""Every citation resolves, and the ones that will not yet are listed.

`docs/REORG_PLAN.md` 3 audited six kinds of reference by hand and found two of
them clean -- 629 section citations and 110 repository paths, zero defects. **This
test exists to hold that result, not to discover it**, which is the same reason
`tests/test_no_list.py` exists beside `scripts/check_no_list.py`.

It found one thing on its first run that the hand audit had not: a section
citation to `docs/MODELS.md` 26.31, which has never existed, in a file this
project wrote (`third_party/telemanom/PROVENANCE.md`). Corrected in the same
commit that added the guard.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_references as CR  # noqa: E402


@pytest.fixture(scope="module")
def report():
    return CR.check(master=False)


def test_every_citation_in_the_tree_resolves(report) -> None:
    assert report.breaks == [], (
        "citations that do not resolve:\n  " + "\n  ".join(report.breaks))


def test_the_audit_still_has_something_to_audit(report) -> None:
    """A guard that checks nothing passes for the wrong reason.

    The counts are floors, not exact figures: they move as prose is written, and
    pinning them exactly would make every document edit a test edit. What they
    catch is the regex silently ceasing to match.
    """
    assert report.checked["section"] > 400, report.checked
    assert report.checked["path"] > 400, report.checked
    assert report.checked["line"] > 250, report.checked
    assert report.checked["link"] > 20, report.checked
    assert report.checked["node"] > 0, report.checked


def test_the_planned_list_is_debt_and_not_a_hiding_place(report) -> None:
    """Each entry names the tranche that will write it, and none exists yet.

    An entry that has been written is a stale exemption, and a stale exemption is
    a citation nobody is checking any more.
    """
    for path, why in CR.PLANNED.items():
        assert not (ROOT / path).exists(), (
            f"{path} now exists -- remove it from PLANNED so it is checked again "
            f"(it was waiting on {why})")
        assert "tranche" in why, f"{path}'s exemption does not say who writes it"


# -- D67's master mode --------------------------------------------------------

def test_master_mode_reports_off_branch_paths_rather_than_breaking(report) -> None:
    """D67: `master` carries the component and the evidence and nothing else.

    A citation from `docs/DECISIONS.md` into `scripts/` does not resolve there --
    about 380 do not -- and that is by design. The convention is that any
    off-branch path resolves on `dev` at the named commit, so master mode reports
    those and breaks only on a path absent from **both**.
    """
    master = CR.check(master=True)
    assert master.breaks == report.breaks, (
        "master mode must not invent breaks the dev tree does not have:\n  "
        + "\n  ".join(set(master.breaks) - set(report.breaks)))
    assert len(master.dev_resolving) > 100, (
        f"only {len(master.dev_resolving)} off-branch citations found; D67 counted "
        "about 380, so the curated set is probably being read wrongly")


def test_every_master_prefix_is_a_path_that_exists() -> None:
    """D67's list must name things that are here, not things remembered.

    Two exceptions, both of which D67 states: `LICENSE` is added when a licence is
    selected and it is still "not yet selected"; `docs/datasets/` is written by
    REORG_PLAN tranche 4 and is carried in `PLANNED` until it is.
    """
    planned_prefixes = {p.rsplit("/", 1)[0] + "/" for p in CR.PLANNED if "/" in p}
    missing = [p for p in CR.MASTER_PREFIXES
               if not (ROOT / p).exists()
               and p != "LICENSE"
               and p not in planned_prefixes
               and p not in CR.PLANNED]
    assert missing == [], f"D67 names paths that do not exist: {missing}"


def test_the_three_taught_conventions_are_still_needed() -> None:
    """Each exemption earns its place, or it is dead weight hiding a real break."""
    objective = (ROOT / "Objective.md").read_text(encoding="utf-8")
    assert "14.N" in objective, "the table-row convention no longer applies; remove it"
    assert CR.UPSTREAM_DOC_PREFIXES, "the upstream-docs convention was emptied"
    assert "detector.py" in CR.VENDORED_FILENAMES
    # A fourth, found by the guard flagging this very test file: `docs/DECISIONS.md`
    # has zero numbered headings -- every entry is `## D42.` -- so a bare number
    # after its name is prose, not a citation.
    decisions = (ROOT / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    numbered = [l for l in decisions.splitlines()
                if l.startswith("## ") and l[3:4].isdigit()]
    assert numbered == [], (
        "docs/DECISIONS.md now has numbered sections; the exclusion in "
        "check_references.NO_NUMBERED_SECTIONS is no longer correct")


def test_the_guard_does_not_check_itself() -> None:
    """It quotes citation forms to explain them, so checking them reports itself.

    Found by the guard flagging its own docstring's `tests/test_x.py::test_y`
    example, and this test's quotation of the one dangling citation it caught.
    """
    assert "scripts/check_references.py" in CR.SELF_REFERENTIAL
    assert "tests/test_references_resolve.py" in CR.SELF_REFERENTIAL
    for name in CR.SELF_REFERENTIAL:
        assert (ROOT / name).exists(), f"{name} is exempted and does not exist"


def test_the_checker_runs_clean_from_the_command_line() -> None:
    for args in ([], ["--master"]):
        proc = subprocess.run([sys.executable, "scripts/check_references.py", *args],
                              cwd=ROOT, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "every citation resolves" in proc.stdout
