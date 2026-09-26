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

**On the baseline below.** ``BASELINE`` is a register of where each retired
claim still appears without a correction in its paragraph, pinned so the set can
shrink but never grow. It was 25 entries when this test was written, across ten
sites that asserted a superseded claim as live. **It is now 8, and all 8 are
records that must never change:**

  `docs/MODELS.md` section 4's PREDICTED table and section 21's verbatim
  re-quotation of it; `CHANGELOG.md`'s 0.3.0 entry; `docs/DECISIONS.md` D9's and
  `docs/RESULTS.md` section 1's measured lead tables; `docs/NARRATIVE.md`'s
  account of what was believed at the time; and a print string in a superseded
  sweep. A pre-registration edited after its outcome is not a pre-registration,
  and a record corrected in hindsight is not a record. They are pinned here so
  they cannot move in either direction.

  **The restatement debt is zero.** The ten sites that asserted the claim as
  live were withdrawn on 2026-09-02, each keeping the withdrawn sentence in
  quotation so the record survives. What replaced them is a holding note, not a
  new claim: the headline comparison is withdrawn pending re-measurement under
  work item 9.7, and `docs/RESULTS.md` 6l carries the corrected comparison on
  both channel sets. If a new sentence asserts a retired claim, this test fails
  and names the file and line.

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
    # NOT widened to the bare `28/32`, and the reason is worth keeping.
    # `docs/REORG_PLAN.md` 3b proposed it. Measured before making the change:
    # it matches **30 sites** across RESULTS, MODELS, DECISIONS, Objective,
    # THRESHOLD and a script docstring -- and every one of them is
    # `lstm-telemanom`'s **measured** headline-cell recall, a legitimate `k/n`
    # record that also appears inside a pre-registered band (`docs/MODELS.md`
    # 703's P3 asks for "at or above 25/32"). What D37 and D38 retired is the
    # **comparison** -- 28 of 32 *where a per-channel statistic finds 3* -- and
    # the `three` pattern below is what carries that. Widening here would put
    # 30 standing records into the register, which is the failure this test's
    # own docstring names: a debt list that outlives its debt is one nobody
    # reads. Reported rather than made.
    "ratio": re.compile(r"28 of 32"),
    "three": re.compile(
        r"finds three\b|finds 3\b|floor's 3\b|statistic's 3\b|`rstd`'s 3\b"),
    # Objective.md 1.1 RETIRED: "+26 timesteps of early warning" (D21).
    # Widened 2026-09-10: the claim survives rephrasing. `26-timestep head start`
    # and `a median of 26 timesteps ahead` are the same retired assertion and the
    # original two alternatives matched neither.
    "lead": re.compile(
        r"\+26 timesteps|26 timesteps early|26-timestep|26 timesteps ahead"),
    # D83 c.5, superseded in part by D84 (2026-09-25): the library exports the
    # retrainer OPT-IN, so "nothing else" and "not the retrainer" are no longer
    # true of what a mission CAN adopt -- only of what it gets by default. A live
    # sentence may say the default; it may not say the retrainer cannot be had.
    "noretrainer": re.compile(
        r"exports `Sentinel/Monitor` and nothing else|not the retrainer, not an OCaml runtime"
        r"|inherits none of it|not adopted, not exported"),
}

#: A hit on a line carrying one of these is corrected in place, which is the
#: house form: `new (was old, D-ref)`, a `(!)` rider, or prose that names the
#: retirement. See `docs/MODELS.md` 21.2 -- "nothing is overwritten and no
#: number is deleted".
CORRECTED = re.compile(
    r"\(was |\(!\)|RETIRED|[Rr]etired|WITHDRAWN|[Ww]ithdrawn|used to|D37|D38|D39|D40"
    r"|was an artifact|not three|not 3\b|turned out|about arithmetic")

BASELINE = {
    ("CHANGELOG.md", "three"): 1,             # 0.3.0, the history entry
    # D83.5 and D84 quote D83 c.5's sentence in order to supersede it. Records.
    ("docs/DECISIONS.md", "noretrainer"): 2,
    # Three, since the `lead` pattern was widened 2026-09-10 to catch the claim's
    # rephrasings. All three sit inside decision entries, which are records and
    # are never edited (`docs/DECISIONS.md` header): D9's measured lead table,
    # D9's "only surviving candidate" paragraph, and a blockquote inside a later
    # entry quoting that reasoning back. Pinned as records, not corrected.
    ("docs/DECISIONS.md", "lead"): 3,
    ("docs/MODELS.md", "three"): 3,           # section 4 PREDICTED, and 21's re-quotation
    ("docs/NARRATIVE.md", "lead"): 1,         # the narrative, as it happened
    ("docs/RESULTS.md", "lead"): 1,           # section 1's lead table
    ("scripts/threshold_sweep.py", "lead"): 1,  # a print string in a superseded sweep
}


def tracked() -> list[str]:
    """Tracked prose and first-party source. Never the bucket, never `runs/`."""
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout.split()
    return [f for f in out
            if f.endswith(".md")
            or (f.endswith(".py") and f.startswith(("src/", "scripts/", "tests/")))]


def paragraphs(lines: list[str]) -> list[tuple[int, int]]:
    """Half-open [start, end) spans of consecutive non-blank lines, 0-indexed."""
    spans, start = [], None
    for i, line in enumerate(lines + [""]):
        if line.strip():
            start = i if start is None else start
        elif start is not None:
            spans.append((start, i))
            start = None
    return spans


def scan() -> dict[tuple[str, str], list[int]]:
    """Uncorrected occurrences of each retired claim, by file, with line numbers.

    The exemption is **paragraph-scoped, not line-scoped**. A withdrawal quotes
    the sentence it withdraws, and a quoted sentence wraps: the marker lands on
    one line and the banned phrase on the next. Scoping to the line would put
    every such continuation in the register as debt, and a register full of
    entries that are already fixed is one nobody reads.
    """
    found: dict[tuple[str, str], list[int]] = {}
    for name in tracked():
        if name == "tests/test_documents_are_current.py":
            continue                      # this file quotes every pattern it bans
        lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
        for lo, hi in paragraphs(lines):
            if any(CORRECTED.search(line) for line in lines[lo:hi]):
                continue
            for offset, line in enumerate(lines[lo:hi]):
                for claim, pattern in CLAIMS.items():
                    if pattern.search(line):
                        found.setdefault((name, claim), []).append(lo + offset + 1)
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


def _flattened(path: Path) -> str:
    """The document as one line, emphasis stripped, whitespace collapsed.

    The count wraps. `docs/STATUS.md` once carried it as `**605` on one line and
    `tests**,` on the next, and a pattern anchored to a single line read that as
    *no count stated at all* -- so the guard passed while its subject was wrong,
    which is the one failure mode a guard may not have. Markdown emphasis is
    stripped for the same reason: `**608** tests` and `608 tests` are the same
    claim to a reader and must be the same claim here.
    """
    text = path.read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", text.replace("*", "").replace("`", ""))


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
                  re.findall(r"(\d{3,4})\s+tests", _flattened(ROOT / name))}
        assert stated, f"{name} no longer states a test count; this test is stale"
        if stated != {collected}:
            wrong[name] = sorted(stated)

    assert not wrong, (
        f"pytest collects {collected}; these documents say otherwise: {wrong}. "
        "Update them -- a reader checks this number first.")
