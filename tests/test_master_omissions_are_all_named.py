"""(!) `master`'s omissions table says "Every omission is named." It did not name six.

`master:docs/OMISSIONS.md` opens with:

    **Every omission is named.** A curated branch that quietly drops things is worse
    than an uncurated one, because a reader cannot tell what they are not seeing.

On 2026-09-21 an inspection found 293 tracked paths absent from `master` and six of
them unaccounted for by that table. The one that mattered was **`oxcaml/`** -- the
whole retrainer source, 55 files. `master` discusses OxCaml across six of its own
files, `master:fprime/SentinelRetrain/README.md` tells a reader the deployment needs
the OxCaml switch, and nothing on the branch said the source was not there. The other
five were `docs/manifest.snapshot.json`, `docs/reorg_plan.json`, `.env.example`,
`conftest.py` and `requirements.txt`.

The claim is the thing worth guarding. A table that is merely long is prose; a table
that is provably complete is evidence, and it is the sentence a reviewer would test.

**The branches share no history** (`git merge-base dev master` prints nothing), so this
compares TREES, exactly as `tests/test_master_documents_are_current.py` does. It reads
both branches through `git show` / `git ls-tree` rather than the working tree, so it
does not care which branch is checked out.

A path counts as named when the table mentions it or any ancestor directory of it --
`third_party/telemanom/` covers `third_party/telemanom/LICENSE.txt`, and `src/` covers
everything beneath it. That is the granularity the table already uses.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: (!) D81 moved the table out of `master:README.md` into its own document. The
#: README carries a one-line pointer instead, and the claim -- "every omission is
#: named" -- moved with the table, so this guard follows it. The whole document
#: IS the table now, so there is no section to find; what is asserted instead is
#: that the document exists and still makes the claim.
OMISSIONS = "docs/OMISSIONS.md"


def _git(*args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, f"git {' '.join(args)} failed:\n{proc.stderr}"
    return proc.stdout


def _tracked(ref: str) -> set[str]:
    return {line for line in _git("ls-tree", "-r", "--name-only", ref).splitlines() if line}


def _omissions_table(document: str) -> str:
    assert "|" in document, (
        f"master:{OMISSIONS} carries no table; the omissions table is what this "
        "test guards, and an empty document would make every check below vacuous")
    return document


#: The table names paths in backticks. Parsing them, rather than substring-matching
#: the whole table, is deliberate: `"docs/" in table` is TRUE merely because the
#: table carries `docs/MODELS.md`, which would have silently covered every other
#: file under `docs/`. That is exactly how this test was blind on its first run,
#: and it hid `docs/manifest.snapshot.json` and `docs/reorg_plan.json`.
CODE_SPAN = re.compile(r"`([^`]+)`")


def _declared(table: str) -> set[str]:
    return {s.strip() for s in CODE_SPAN.findall(table)}


def _named(path: str, table: str) -> bool:
    """True when the table declares the path itself, or a DIRECTORY containing it.

    A directory counts only when it is written as one, with a trailing slash.
    """
    declared = _declared(table)
    if path in declared:
        return True
    return any(d.endswith("/") and path.startswith(d) for d in declared)


@pytest.fixture(scope="module")
def absent_and_table() -> tuple[set[str], str]:
    absent = _tracked("dev") - _tracked("master")
    return absent, _omissions_table(_git("show", f"master:{OMISSIONS}"))


def test_every_path_absent_from_master_is_named_in_its_omissions_table(
        absent_and_table) -> None:
    absent, table = absent_and_table
    unnamed = sorted(p for p in absent if not _named(p, table))
    # Collapse to what a reader would add: the top-level entry, or the file itself.
    groups = sorted({p if "/" not in p else p.split("/")[0] + "/" for p in unnamed})
    assert unnamed == [], (
        f"{len(unnamed)} tracked path(s) absent from master are not accounted for by "
        f"its omissions table, which claims every omission is named.\n"
        f"  add rows for: {', '.join(groups)}\n"
        f"  first few: {', '.join(unnamed[:8])}")


def test_the_table_claims_completeness_so_the_check_above_is_the_right_one(
        absent_and_table) -> None:
    """If the claim is ever softened, this test should be reconsidered rather than
    left asserting something the branch no longer promises."""
    _, table = absent_and_table
    assert "Every omission is named" in table, (
        f"master:{OMISSIONS} no longer claims to name every omission. That is "
        "allowed, but then the completeness check above is enforcing a promise the "
        "branch has stopped making -- decide which, and record it.")


def test_the_comparison_can_actually_see_a_difference(absent_and_table) -> None:
    """The positive control. If `_tracked` returned the same set for both refs, or
    an empty one, the check above would pass over nothing and prove nothing."""
    absent, table = absent_and_table
    assert len(absent) > 100, (
        f"only {len(absent)} paths differ between dev and master; the curated branch "
        "is a much smaller subset than that, so this comparison is not working")
    assert not _named("no/such/path/at/all.txt", table), "_named matches anything"
    assert _named("src/sentinel_toolkit/cli.py", table), (
        "_named cannot see `src/`, which the table has always carried")
