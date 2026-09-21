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
    """The list must name things that are here, not things remembered.

    The exceptions are declared in `MASTER_ONLY_PREFIXES` and each is stated:
    `LICENSE` is added when a licence is selected and it is still "not yet
    selected"; `docs/DESIGN.md` and `docs/EVIDENCE.md` are D69's customer
    documents and exist on `master` alone. That the list still describes the real
    branch is a separate check, and it is the one nothing made until D69 --
    `tests/test_master_documents_are_current.py` reads `git ls-tree master`.
    """
    planned_prefixes = {p.rsplit("/", 1)[0] + "/" for p in CR.PLANNED if "/" in p}
    missing = [p for p in CR.MASTER_PREFIXES
               if not (ROOT / p).exists()
               and p not in CR.MASTER_ONLY_PREFIXES
               and p not in planned_prefixes
               and p not in CR.PLANNED]
    assert missing == [], f"the curated list names paths that do not exist: {missing}"


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


# ----------------------------------------------------------------------------
# (!) The framework checkout is gitignored, and this checker used to FAIL rather
# than skip when it was absent. Decision 79. The skip has to stay LOUD and
# NARROW, so both halves are asserted here rather than trusted.
# ----------------------------------------------------------------------------

def test_a_citation_into_the_absent_checkout_is_skipped_not_broken(report) -> None:
    """The reason the suite went from 2 failed to 0 on an unbuilt tree.

    `docs/PHASE5.md` cites F's own skills documentation inside `fprime/lib/`. The
    citation is correct and resolves the moment `scripts/fprime_setup.sh` runs. It
    is not a break, and reporting it as one taught readers to ignore a red gate.
    """
    if not CR.checkout_absent():
        pytest.skip("the framework checkout is present; citations into it are checked")
    assert report.skipped, (
        "the checkout is absent but nothing was skipped -- either no citation "
        "points into it any more, or the skip stopped working")
    assert all("fprime/lib/" in line for line in report.skipped), report.skipped
    assert not any("fprime/lib/" in b for b in report.breaks), (
        "a citation into the absent checkout is still being reported as a break:\n  "
        + "\n  ".join(report.breaks))


def test_the_skip_is_narrow(report) -> None:
    """Only `fprime/lib/`, and only while it is absent. Everything else still breaks.

    This is the half that could rot quietly: a widened prefix would silence real
    breaks and nothing would say so.
    """
    assert CR.FPRIME_CHECKOUT_DIR == "fprime/lib/", (
        f"the skip now covers `{CR.FPRIME_CHECKOUT_DIR}`. Widening it silences "
        "citations that no build would ever satisfy -- Decision 79 scoped it to "
        "the rebuildable framework checkout and nothing else.")
    assert CR.checkout_absent() is not (ROOT / "fprime" / "lib").is_dir()


def test_the_skip_is_announced_on_the_command_line(report) -> None:
    """A silent skip is not acceptable: it is the failure mode the symbol guard
    had, where nobody knew a green run had checked nothing."""
    if not CR.checkout_absent():
        pytest.skip("the framework checkout is present; there is nothing to announce")
    for args in ([], ["--master"]):
        proc = subprocess.run([sys.executable, "scripts/check_references.py", *args],
                              cwd=ROOT, capture_output=True, text=True)
        out = proc.stdout
        assert "NOT CHECKED" in out, f"the skip is not announced:\n{out}"
        assert "scripts/fprime_setup.sh" in out, (
            f"the announcement does not name the script that would resolve it:\n{out}")
        assert "SKIP   docs/PHASE5.md:" in out, (
            f"the announcement does not name the citation's file and line:\n{out}")
        assert proc.returncode == 0, f"exit {proc.returncode} in mode {args}:\n{out}"
