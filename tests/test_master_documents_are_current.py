"""Every figure `master` states about `dev` is re-derived from `dev` here (D69).

`master` carries a curated set of documents and those documents quote `dev`:
how many files it tracks, how many scripts it holds, how many tests it runs,
which commit the snapshot was taken from. **Nothing checked any of them.** D69
measured the result: of four such figures, two were stale -- each correct on the
day it was written, each copied onto a branch that never re-derived it -- and a
third went stale one commit later when `docs/MODELS.md` 40 landed. That is not a
proofreading failure; it is the absence of a guard, and this is the guard.

**It compares trees, never ancestry.** `dev` and `master` share no commit at all:
different root commits, `git merge-base dev master` empty, `master..dev` the
whole of `dev` (D69 EVIDENCE). So every check here reads `master` with
`git show` / `git ls-tree` and derives the answer from `dev`'s own working tree.

**The stale figures are pinned, not fixed.** `BASELINE` is a debt register in the
shape `tests/test_documents_are_current.py:97` uses: pinned by exact equality in
*both* directions, so a new drift fails and a repair that leaves the register
stale fails too. D69 consequence 10 keeps the known ones standing until the
customer documents replace them, because a guard proved against a defect that has
already been repaired is a guard proved against nothing --
`test_the_check_actually_catches_things` is where that proof lives.
"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_references as C  # noqa: E402
import make_tocs as T  # noqa: E402

#: The public branch. A ref, not a checkout -- nothing here changes the worktree.
MASTER = "master"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout


def _ref_exists(ref: str) -> bool:
    return subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
                          cwd=ROOT, capture_output=True).returncode == 0


#: A clone that fetched only `dev` has no `master`, and a suite that goes red for
#: that reason is reporting on the clone rather than on the repository -- the same
#: reason `tests/test_flight_build.py:24` skips without the C++ toolchain.
needs_master = pytest.mark.skipif(not _ref_exists(MASTER),
                                  reason=f"no `{MASTER}` ref in this clone")


def _flattened(text: str) -> str:
    """One line, emphasis stripped, whitespace collapsed.

    Same treatment and same reason as `tests/test_documents_are_current.py:214`:
    these figures sit in wrapped table cells, and a pattern anchored to one line
    reads a wrapped claim as *no claim stated*, which is the one failure mode a
    guard may not have.
    """
    return re.sub(r"\s+", " ", text.replace("*", "").replace("`", ""))


def _on_master(path: str) -> str:
    return _flattened(_git("show", f"{MASTER}:{path}"))


# -- the derivations, each from `dev` --------------------------------------

def _dev_tracked_files() -> int:
    return len(_git("ls-files").split())


def _master_tracked_files() -> int:
    return len(_git("ls-tree", "-r", "--name-only", MASTER).split())


def _dev_scripts_files() -> int:
    """Every tracked file under `scripts/`, which is what the figure counted.

    36 was right at `a0dfafb`, where `scripts/` held 36 files and no README. The
    derivation is the one that produced the number, not a tidier one invented now.
    """
    return len(_git("ls-files", "scripts").split())


def _collected_tests() -> int:
    proc = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q",
                           "-p", "no:cacheprovider"],
                          cwd=ROOT, capture_output=True, text=True)
    match = re.search(r"(\d+) tests? collected", proc.stdout)
    assert match, f"could not read a collected count from pytest:\n{proc.stdout[-2000:]}"
    return int(match.group(1))


def _models_index_entries() -> int:
    text = (ROOT / "docs" / "MODELS.md").read_text(encoding="utf-8")
    body, _notes = T.models_toc(text)
    return len([ln for ln in body.splitlines() if ln.startswith("  - [")
                or ln.startswith("- [")])


@dataclass(frozen=True)
class Figure:
    key: str
    document: str
    pattern: str
    group: int
    derive: Callable[[], int]
    what: str


#: What `master` says about `dev`, and how `dev` answers it. A figure absent from
#: this tuple is a figure nothing checks, which is the condition D69 was written
#: about -- so adding a number to a customer document means adding a row here.
FIGURES = (
    Figure("dev_tracked_files", "README.md",
           r"repository's (\d[\d,]*) tracked files, (\d[\d,]*) are here", 1,
           _dev_tracked_files, "files tracked on dev"),
    Figure("master_tracked_files", "README.md",
           r"repository's (\d[\d,]*) tracked files, (\d[\d,]*) are here", 2,
           _master_tracked_files, "files tracked on master"),
    Figure("scripts_files", "README.md",
           r"\| scripts/ \| (\d[\d,]*) scripts", 1,
           _dev_scripts_files, "files under scripts/ on dev"),
    Figure("collected_tests", "README.md",
           r"The (\d[\d,]*)-test suite", 1,
           _collected_tests, "tests collected on dev"),
    Figure("models_index_entries", "README.md",
           r"outcome, (\d[\d,]*) sections", 1,
           _models_index_entries, "generated index entries in docs/MODELS.md"),
)

#: (!) The debt register: `key -> what master states`, measured 2026-09-11.
#: Pinned by exact equality in both directions -- a NEW stale figure fails, and a
#: figure repaired without being removed from here fails too, because a debt list
#: nobody prunes stops describing the debt.
#:
#: **(!) It pins master's side and NOT dev's, and the first draft got that wrong.**
#: Pinning the derived half looked more exact and made the register a liability:
#: `dev_tracked_files` moved 270 -> 271 the moment this file was committed, so
#: every ordinary commit that adds or removes a file on `dev` would turn the suite
#: red for a defect that had not changed. The debt is *master says 227 and 227 is
#: wrong*; `dev` going to 271 is the same debt, not a new one. So the derived side
#: is reported in the failure message and is free to move.
#:
#: **Four, and D69 measured two.** `models_index_entries` went stale when
#: `docs/MODELS.md` 40 landed, which D69.1 records; `collected_tests` went stale
#: when this guard added five tests. Neither needs a rider -- they are the same
#: mechanism D69 was written about, and the register is where a drifting figure is
#: supposed to end up.
#:
#: These stand by D69 consequence 10 until the customer documents replace them.
BASELINE = {
    "dev_tracked_files": 227,
    "scripts_files": 36,
    "collected_tests": 683,
    "models_index_entries": 353,
}

#: Documents `master` holds its own version of, so a byte difference against any
#: `dev` commit is the curation working rather than drift. These are exactly the
#: two paths `git diff dev master` reports as modified rather than deleted.
CURATED_ON_MASTER = {"README.md", "docs/STATUS.md"}

#: (!) Of the paths `master` is supposed to carry unchanged, these do not match
#: the `dev` commit its README names. `a3c2a4a` ported two dataset files from
#: `46165a1` without moving the stated commit, so the branch is a mixture and the
#: citation describes it only approximately. Measured 2026-09-11.
BASELINE_SNAPSHOT_MISMATCH = {
    "docs/datasets/ESA_ADB.md",
    "docs/datasets/REPRODUCING.md",
}

#: On `master` and covered by no `MASTER_PREFIXES` entry, which is harmless
#: because it carries no citations; and declared in `MASTER_PREFIXES` but not yet
#: on the branch, because the licence is not selected (`docs/STATUS.md` 7 item H).
PREFIX_ALLOWANCES = {"uncovered": {".gitignore"}, "unused": {"LICENSE"}}


def survey(figures=FIGURES) -> dict[str, tuple[int, int]]:
    """`key -> (stated, derived)` for every figure whose two sides disagree."""
    documents = {f.document: _on_master(f.document) for f in figures}
    stale = {}
    for figure in figures:
        match = re.search(figure.pattern, documents[figure.document])
        assert match, (
            f"{MASTER}:{figure.document} no longer states {figure.what!r} in the "
            f"form this guard reads ({figure.pattern!r}). Either the figure was "
            "removed -- then remove the row -- or the wording changed and this "
            "guard has gone blind, which is worse than a stale number.")
        stated = int(match.group(figure.group).replace(",", ""))
        derived = figure.derive()
        if stated != derived:
            stale[figure.key] = (stated, derived)
    return stale


@needs_master
def test_every_figure_master_states_about_dev_is_current() -> None:
    found = survey()
    stated = {key: pair[0] for key, pair in found.items()}
    if stated == BASELINE:
        return

    new = {k: found[k] for k in stated if k not in BASELINE}
    fixed = {k: v for k, v in BASELINE.items() if k not in stated}
    moved = {k: (BASELINE[k], stated[k]) for k in stated
             if k in BASELINE and BASELINE[k] != stated[k]}

    assert not new, (
        f"NEW drift on `{MASTER}`, and not permitted: {new} (stated, derived). "
        "A figure a customer document states about `dev` must be re-derivable "
        "from `dev` (D69).")
    assert not moved, (
        f"`{MASTER}` now states something different: {moved} (was, now). The "
        "branch moved under the register; re-measure and update it, with the date.")
    assert not fixed, (
        f"FIXED, so remove them from BASELINE in the same commit: {sorted(fixed)}. "
        "A debt register that outlives its debt stops describing anything.")


@needs_master
def test_the_check_actually_catches_things() -> None:
    """The proof. Run with the register ignored, the guard names the drift.

    D69 consequence 10 leaves the stale figures standing precisely so this can
    exist: a check nobody has watched fail is not known to work
    (`tests/test_no_list.py:18` is the same argument for the LIST guard).
    """
    found = survey()
    assert set(found) == set(BASELINE), (
        f"the guard reports {sorted(found)}; the register pins {sorted(BASELINE)}")
    for key, (stated, derived) in found.items():
        assert stated != derived, f"{key} is not stale, so it proves nothing"
    stated, derived = found["dev_tracked_files"]
    assert derived == _dev_tracked_files()
    assert stated < derived, (
        "the tracked-file figure is pinned as understating dev; if that reversed, "
        "the branch was recut and the register is describing the wrong defect")


@needs_master
def test_the_snapshot_commit_master_names_exists_and_is_named_accurately() -> None:
    """`master` tells a reader which `dev` commit its citations resolve at.

    D67 consequence 5 makes that convention load-bearing for about 380 citations,
    so the commit it names has to be one a reader can check out, and the shared
    files have to be the ones that commit holds -- bar the two `master` writes
    for itself. `a3c2a4a` ported two dataset files from a later `dev` commit
    without moving the stated one, which is what the pinned set records.
    """
    match = re.search(r"development branch at commit ([0-9a-f]{7,40})",
                      _on_master("README.md"))
    assert match, "master:README.md no longer names the dev commit it resolves at"
    cited = match.group(1)
    assert _ref_exists(cited), (
        f"master names dev commit {cited}, which this repository does not hold")

    on_master = set(_git("ls-tree", "-r", "--name-only", MASTER).split())
    at_cited = set(_git("ls-tree", "-r", "--name-only", cited).split())
    differs = {
        path for path in sorted((on_master & at_cited) - CURATED_ON_MASTER)
        if _git("rev-parse", f"{MASTER}:{path}") != _git("rev-parse", f"{cited}:{path}")
    }
    assert differs == BASELINE_SNAPSHOT_MISMATCH, (
        f"paths on `{MASTER}` that differ from the commit it names ({cited}): "
        f"{sorted(differs)}; pinned: {sorted(BASELINE_SNAPSHOT_MISMATCH)}. "
        "Either move the stated commit or update the register.")


@needs_master
def test_master_prefixes_still_describes_the_branch() -> None:
    """`--master` mode classifies by a hand-kept list and never reads the branch.

    `scripts/check_references.py:172-177` resolves existence against `dev`'s
    working tree and decides on-branch membership from `MASTER_PREFIXES`. So the
    list can drift from `master` silently, and a citation would then be reported
    as dev-resolving when it is on the branch, or the reverse. This is the
    comparison that mode does not make (D69 consequence 4).
    """
    paths = _git("ls-tree", "-r", "--name-only", MASTER).split()
    covered = lambda p: any(p == x or p.startswith(x) for x in C.MASTER_PREFIXES)

    uncovered = {p for p in paths if not covered(p)}
    unused = {x for x in C.MASTER_PREFIXES
              if not any(p == x or p.startswith(x) for p in paths)}

    assert uncovered == PREFIX_ALLOWANCES["uncovered"], (
        f"on `{MASTER}` and covered by no MASTER_PREFIXES entry: {sorted(uncovered)}")
    assert unused == PREFIX_ALLOWANCES["unused"], (
        f"declared in MASTER_PREFIXES and absent from `{MASTER}`: {sorted(unused)}")


@needs_master
def test_master_is_a_subset_of_dev_and_adds_nothing() -> None:
    """D67's curation is a subset, and a file appearing only on `master` would
    be a file no guard on `dev` has ever seen -- unreviewed by construction."""
    on_master = set(_git("ls-tree", "-r", "--name-only", MASTER).split())
    on_dev = set(_git("ls-files").split())
    only_master = on_master - on_dev
    assert not only_master, (
        f"present on `{MASTER}` and not on `dev`: {sorted(only_master)}")
