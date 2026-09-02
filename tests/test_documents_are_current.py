"""The documents go stale silently, because nothing else in this suite reads one.

Work item 9.5 corrected `baselines._rolling`, re-scored the floor from
headline-cell 3/32 to 25/32, and swept the repository for the figures that had
moved. It corrected every scorecard *table*. It missed the *prose claim*, which
is the sentence a reader actually quotes, and `CHANGELOG.md`'s 0.6.1 entry then
recorded the sweep as complete: "is corrected everywhere it appears with the old
figure beside it." It was not. Nine sentences still assert the retired ratio.

Nothing failed, because no test in this suite opens a `.md` file.
`docs/DECISIONS.md` D15's own words apply: the rule that was supposed to catch
this is a convention, and a convention has no test.

This is that test. It does not check that the documents are *right* -- no test
can -- it checks three things a machine can settle:

  1. a claim `Objective.md` 1.1 has retired does not gain new live occurrences,
  2. every tracked document is ASCII, which is a convention nothing enforced,
  3. a count a document states about this repository matches the repository.

**On the baseline below.** ``BASELINE`` is a debt register, not an allowlist of
things that are fine. It pins where each retired claim currently appears without
a correction on the same line, so the set can shrink but never grow. Two kinds of
entry are in it and they are not the same kind:

  *Preserved.* `docs/MODELS.md` section 4's PREDICTED table and section 21's
  verbatim re-quotation of it, and `CHANGELOG.md`'s 0.3.0 history entry. A
  pre-registration that is edited after its outcome is not a pre-registration,
  and these must never be corrected. They are pinned so they cannot move either.

  *Debt.* The twelve sites the restatement has to fix, held pending review at
  `docs/STATUS.md` section 7 on D39. In order:

      Objective.md          1.1's KEPT block, and section 4.4
      README.md             "What is claimed, and what is retired"
      docs/RESULTS.md       section 7, and section 1a's quotation of the old text
      docs/PHASE1_REPORT.md the work item 4 narrative, and the closing claim
      docs/NARRATIVE.md     section 3
      docs/DECISIONS.md     D23's context block
      docs/MODELS.md        section 8
      whiten.py             the module docstring

  Every figure in those sentences is superseded: the floor is 25/32 and not 3
  (D37), the "28 of 32" is `lstm-telemanom`'s -- a detector disqualified at 22 of
  48 commanded manoeuvres -- and not the flying `gru-quantile`'s 22/32, and D38
  has since measured the flying detector's catches as a strict subset of the
  floor's. When the restatement lands, each corrected line gains a marker and
  drops out of the scan, and its entry here goes with it.

The scan is deliberately narrow. It matches the retired *claims* as prose --
"28 of 32", "finds three", "the floor's 3", "+26 timesteps" -- and not the bare
numerals. `3/32` on its own is not banned: D39 reports a live and unrelated 3/32
(the contextual count), and a dozen correction narratives legitimately write
"3/32 -> 25/32". Banning a numeral would have produced noise and taught the next
reader to ignore this file.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: The retired claims, matched as the prose that carries them.
CLAIMS = {
    # Objective.md 1.1 KEPT: "a forecaster ... finds 28 of 32 headline-cell
    # events where a per-channel statistic finds 3". Falsified by D37, and D38
    # reversed the direction it argued for.
    "ratio": re.compile(r"28 of 32"),
    "three": re.compile(
        r"finds three\b|finds 3\b|floor's 3\b|statistic's 3\b|`rstd`'s 3\b"),
    # Objective.md 1.1 RETIRED: "+26 timesteps of early warning" (D21).
    "lead": re.compile(r"\+26 timesteps|26 timesteps early"),
}

#: A hit on a line carrying one of these is corrected in place, which is the
#: house form: `new (was old, D-ref)`, a `(!)` rider, or prose that names the
#: retirement. See `docs/MODELS.md` 21.2 -- "nothing is overwritten and no
#: number is deleted".
CORRECTED = re.compile(
    r"\(was |\(!\)|RETIRED|[Rr]etired|used to|D37|D38|D39|D40"
    r"|was an artifact|not three|not 3\b|turned out|about arithmetic")

BASELINE = {
    ("CHANGELOG.md", "three"): 1,                    # preserved: 0.3.0 history
    ("Objective.md", "ratio"): 2,
    ("Objective.md", "three"): 1,
    ("README.md", "ratio"): 1,
    ("README.md", "three"): 1,
    ("docs/DECISIONS.md", "lead"): 1,                # preserved: D9's lead table
    ("docs/DECISIONS.md", "ratio"): 1,
    ("docs/DECISIONS.md", "three"): 1,
    ("docs/MODELS.md", "ratio"): 1,
    ("docs/MODELS.md", "three"): 3,                  # 2 preserved (4.x, 21.1)
    ("docs/NARRATIVE.md", "lead"): 1,                # preserved: the narrative
    ("docs/NARRATIVE.md", "ratio"): 1,
    ("docs/NARRATIVE.md", "three"): 1,
    ("docs/PHASE1_REPORT.md", "lead"): 1,            # preserved: the narrative
    ("docs/PHASE1_REPORT.md", "ratio"): 1,
    ("docs/PHASE1_REPORT.md", "three"): 1,
    ("docs/RESULTS.md", "lead"): 1,                  # preserved: section 1 table
    ("docs/RESULTS.md", "ratio"): 1,
    ("docs/RESULTS.md", "three"): 2,                 # 1 preserved (1a's quote)
    ("scripts/threshold_sweep.py", "lead"): 1,       # preserved: a print string
    ("src/sentinel_models/whiten.py", "ratio"): 1,
}


def tracked() -> list[str]:
    """Tracked prose and first-party source. Never the bucket, never `runs/`."""
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout.split()
    return [f for f in out
            if f.endswith(".md")
            or (f.endswith(".py") and f.startswith(("src/", "scripts/", "tests/")))]


def scan() -> dict[tuple[str, str], list[int]]:
    """Uncorrected occurrences of each retired claim, by file, with line numbers."""
    found: dict[tuple[str, str], list[int]] = {}
    for name in tracked():
        if name == "tests/test_documents_are_current.py":
            continue                      # this file quotes every pattern it bans
        text = (ROOT / name).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            if CORRECTED.search(line):
                continue
            for claim, pattern in CLAIMS.items():
                if pattern.search(line):
                    found.setdefault((name, claim), []).append(number)
    return found


def test_no_new_live_occurrence_of_a_retired_claim() -> None:
    """The debt register may shrink. It may not grow, and it may not go stale.

    Equality in both directions on purpose. A new sentence asserting a retired
    claim fails, which is the point. So does a corrected one left in the
    register, because a debt list that outlives its debt is the same failure
    this test exists for.
    """
    found = {key: len(lines) for key, lines in scan().items()}
    if found == BASELINE:
        return

    lines = scan()
    added = {k: lines[k] for k in found if found[k] > BASELINE.get(k, 0)}
    gone = {k: BASELINE[k] for k in BASELINE if found.get(k, 0) < BASELINE[k]}
    report = ["retired-claim occurrences moved."]
    if added:
        report.append("  NEW, and not permitted -- a retired claim stated live:")
        for (name, claim), numbers in sorted(added.items()):
            report.append(f"    {name} [{claim}] lines {numbers}")
        report.append("  Correct it in place in the house form -- the new figure")
        report.append("  with the old in parentheses and its D-reference -- or, if")
        report.append("  it is a pre-registration, leave it and pin it below.")
    if gone:
        report.append("  FIXED, so remove them from BASELINE in the same commit:")
        for (name, claim), was in sorted(gone.items()):
            report.append(f"    {name} [{claim}] was {was}, now {found.get((name, claim), 0)}")
    pytest.fail("\n".join(report))


def test_every_tracked_document_is_ascii() -> None:
    """ASCII-only was a convention with nothing behind it. Now it has this.

    The house typography is `--` for an em dash, `->` for an arrow, `>=`, `+/-`
    and straight quotes. A smart quote pasted in from anywhere renders as
    mojibake in a terminal, which is where these documents are read.
    """
    offenders = []
    for name in tracked():
        if not name.endswith(".md"):
            continue
        for number, line in enumerate(
                (ROOT / name).read_text(encoding="utf-8").splitlines(), 1):
            bad = {c for c in line if c != "\t" and not 0x20 <= ord(c) <= 0x7E}
            if bad:
                offenders.append(f"{name}:{number} {sorted(bad)!r}")
    assert not offenders, "non-ASCII in tracked documents:\n  " + "\n  ".join(offenders)


#: Documents that tell a reader what this suite does *now*. A count here is a
#: claim about the working tree. Counts in `docs/MODELS.md`, `CHANGELOG.md` and
#: `docs/HARNESS.md` are deliberately excluded: those record what passed at a
#: named work item, they are the history, and correcting them would be the one
#: thing this repository never does to a record.
LIVE_COUNT_SITES = ("docs/STATUS.md", "README.md")


def test_the_live_documents_state_the_real_test_count() -> None:
    """A count a document states about this repository must match it.

    `docs/STATUS.md` said 493 and `README.md` said 473 while the suite collected
    505 -- stale since work item 9, and exactly the number a reader checks first
    and loses confidence over. Collected, not run: one subprocess, collection
    only, so this does not recurse.
    """
    proc = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                          cwd=ROOT, capture_output=True, text=True)
    match = re.search(r"(\d+) tests? collected", proc.stdout)
    assert match, f"could not read a collected count from pytest:\n{proc.stdout[-2000:]}"
    collected = int(match.group(1))

    wrong = {}
    for name in LIVE_COUNT_SITES:
        stated = {int(n) for n in
                  re.findall(r"(\d{3,4}) tests", (ROOT / name).read_text(encoding="utf-8"))}
        assert stated, f"{name} no longer states a test count; this test is stale"
        if stated != {collected}:
            wrong[name] = sorted(stated)

    assert not wrong, (
        f"pytest collects {collected}; these documents say otherwise: {wrong}. "
        "Update them -- a reader checks this number first.")
