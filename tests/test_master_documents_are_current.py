"""Every figure `master` states is re-derived from `dev`, and every citation resolves (D69).

`master` carries four customer documents -- `README.md`, `docs/DESIGN.md`,
`docs/EVIDENCE.md`, `docs/STATUS.md` -- and they quote this repository: how big
the detector is, how many refusal codes exist, how many of them the F' component
proves degrade safely, what the headline recall was. **Before D69 nothing checked
any of it.** Of the four figures the previous public README stated about `dev`,
two were stale, a third went stale one commit later, and a fourth when the guard
itself added tests. Each had been correct on the day it was written. That is not a
proofreading failure; it is the absence of a guard, and this is the guard.

**It compares trees, never ancestry.** `dev` and `master` share no commit at all:
different root commits, `git merge-base dev master` empty, `master..dev` the whole
of `dev`. So every check here reads `master` with `git show` / `git ls-tree` and
derives the answer from `dev`'s own tree.

**And every derivation is mechanical.** A figure is checked against the source that
produces it -- `sizeof(Detector)` against the assertion in `flight/test/Footprint.cpp`,
the refusal count against the enum in `Status.hpp`, the F' coverage against the loop
bound in the component's own test, the headline against D65's table. Not against
another document's prose, which is how a figure travels while staying wrong.
"""
from __future__ import annotations

import os
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

#: The public branch. A ref, not a checkout -- nothing here touches the worktree.
MASTER = "master"

#: The four documents `master` writes for itself. Two share a name with a `dev`
#: document and differ from it by design; two exist only on `master`.
CUSTOMER_DOCS = ("README.md", "docs/DESIGN.md", "docs/EVIDENCE.md", "docs/STATUS.md")
CURATED_ON_MASTER = {"README.md", "docs/STATUS.md"}
MASTER_ONLY = {"docs/DESIGN.md", "docs/EVIDENCE.md"}


def _git(*args: str) -> str:
    """(!) `GIT_INDEX_FILE` is scrubbed, and that is not defensive programming.

    Building `master`'s tree with plumbing sets `GIT_INDEX_FILE` to a scratch
    index. If it is still exported when this suite runs, `git ls-files` reads
    that index instead of the working one, `on_dev` becomes `master`'s file list,
    and the subset check fails for a reason that has nothing to do with either
    branch. Seen 2026-09-14. A guard that can be made to lie by an environment
    variable is not one to trust.
    """
    env = {k: v for k, v in os.environ.items() if k != "GIT_INDEX_FILE"}
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True, env=env).stdout


def _ref_exists(ref: str) -> bool:
    return subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
                          cwd=ROOT, capture_output=True).returncode == 0


#: A clone that fetched only `dev` has no `master`, and a suite that goes red for
#: that reason reports on the clone rather than on the repository -- the same
#: reason `tests/test_flight_build.py:24` skips without the C++ toolchain.
needs_master = pytest.mark.skipif(not _ref_exists(MASTER),
                                  reason=f"no `{MASTER}` ref in this clone")


def _flattened(text: str) -> str:
    """One line, emphasis stripped, whitespace collapsed.

    Same treatment and same reason as `tests/test_documents_are_current.py:214`:
    these figures sit in wrapped prose and table cells, and a pattern anchored to
    one line reads a wrapped claim as *no claim stated*, which is the one failure
    mode a guard may not have.
    """
    return re.sub(r"\s+", " ", text.replace("*", "").replace("`", ""))


def _on_master(path: str) -> str:
    return _flattened(_git("show", f"{MASTER}:{path}"))


def _dev(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


# -- the derivations, each from the source that produces the figure ----------

def _detector_bytes() -> int:
    """The assertion, not another document. `flight/test/Footprint.cpp` pins it."""
    text = _dev("flight/test/Footprint.cpp")
    return int(re.search(r"checkEqualU32\(detector,\s*(\d+)U", text).group(1))


def _n3_predicted_bytes() -> int:
    text = _dev("flight/test/Footprint.cpp")
    return int(re.search(r"N3\)\s+([\d,]+) B", text).group(1).replace(",", ""))


def _refusal_codes() -> int:
    """`Status.hpp`'s enum, less `OK`. The C++ and Python values are the same set."""
    text = _dev("flight/include/sentinel/Status.hpp")
    names = re.findall(r"^\s+([A-Z_]+)\s*=\s*\d+U,", text, re.M)
    return len([n for n in names if n != "OK"])


def _load_cases() -> int:
    """One `expect(` call site is one load case; `g_cases` counts them at runtime."""
    return len(re.findall(r"^\s+expect\(", _dev("flight/test/RefusalTests.cpp"), re.M))


def _fprime_covered_codes() -> int:
    """The loop bound in the component's own test. **It is 12 since 2026-09-17.**

    `BAD_PARAM_VERSION` arrived with the version-2 rule and was not added to this
    loop, and this docstring used to end "if somebody closes that gap the documents
    must move with it". It was closed at `docs/MODELS.md` 20.13 and they did -- this
    figure is what made them, which is the whole point of deriving rather than
    restating. 20.13 records the larger half: 12 was also `NOT_LOADED` in the FPP
    enum, so the code the loop was missing would have reported as "never read".
    """
    text = _dev("fprime/Sentinel/Monitor/test/ut/MonitorTester.cpp")
    return int(re.search(r"index\s*<\s*(\d+)U", text).group(1))


def _max_channels() -> int:
    text = _dev("flight/include/sentinel/Config.hpp")
    return int(re.search(r"constexpr\s+U32\s+MAX_CHANNELS\s*=\s*(\d+)U", text).group(1))


def _warmup_steps() -> int:
    """`docs/MODEL_FILE.md` is normative for the format, so it is the source here."""
    text = _dev("docs/MODEL_FILE.md")
    return int(re.search(r"warmup_steps\s+U32\s+(\d+)", text).group(1))


def _toolkit_rungs_built() -> int:
    """How many rungs of the acceptance ladder actually run.

    Declared once in `src/sentinel_toolkit/selftest.py` rather than asserted in
    prose on two branches. The third rung is a **compute gate** -- a real
    mission-scale fit, conditional on a measured wall clock -- so it is `False`
    there until one is run, and the public branch's claim follows this number
    rather than somebody's memory of it.
    """
    text = _dev("src/sentinel_toolkit/selftest.py")
    tuple_body = text[text.index("LADDER = ("):text.index("RUNGS_BUILT")]
    return tuple_body.count("True")


def _d65_row(label: str) -> tuple[str, int, int, int, int, int, int]:
    """One row of D65's arms table: rate, TUNE k/n, EVAL k/n, all k/n."""
    text = _dev("docs/DECISIONS.md")
    m = re.search(r"^\s*" + re.escape(label) +
                  r"\s+([\d.]+)%\s+(\d+)/(\d+)\s+(\d+)/(\d+)\s+(\d+)/(\d+)", text, re.M)
    assert m, f"D65's arms table no longer carries a row for {label!r}"
    return (m.group(1),) + tuple(int(g) for g in m.groups()[1:])


def _eval_caught() -> int:
    return _d65_row("arm 2: residual + derivative")[3]


def _eval_total() -> int:
    return _d65_row("arm 2: residual + derivative")[4]


def _frozen_eval() -> int:
    return _d65_row("frozen (stage 4)")[3]


def _population() -> int:
    return _d65_row("arm 2: residual + derivative")[6]


def _alarm_rate_x10000() -> int:
    return int(round(float(_d65_row("frozen (stage 4)")[0]) * 10000))


def _cycle_parameters() -> int:
    """The retraining cycle's parameter count, from the module that declares it.

    `master` states that EX1's gradients were checked at every one of 75,360 indices,
    and that number is the extent of `Deep_f32`'s parameter vector -- the thing the
    exhaustive check ranged over. It was on `master` unguarded until Section 72; a
    number that appears in a customer document and nothing re-derives is exactly the
    condition D69 was written about.

    Read from `oxcaml/retrainer/deep_f32.ml`, which is where the extent is declared,
    rather than from `ModelFile.hpp`'s `maxParameters()`, which is a constexpr
    function this test would have to evaluate rather than read.
    """
    text = _dev("oxcaml/retrainer/deep_f32.ml")
    return int(re.search(r"^let n_params\s*=\s*(\d+)", text, re.M).group(1))


def _ex1_margin_x100() -> int:
    """EX1's EXHAUSTIVE gradient margin, x100, re-derived from the worst ratio.

    `docs/MODELS.md` 64.7 records the exhaustive worst `err/allowed` over all 75,360
    indices and states the margin that follows from it. The margin is `1 / ratio`, so
    this reads the ratio -- the measured quantity -- and does the arithmetic, rather
    than reading back the number the document states about itself.

    It exists because `master` now quotes the margin, and 60.7's sampled **5.3x** was
    wrong by 1.63x for two months before 64 checked every index (`docs/MODELS.md` 60.6b,
    63.6a). A figure that moved once should not be stated anywhere unguarded.
    """
    text = _dev("docs/MODELS.md")
    m = re.search(r"is (0\.\d+), so the margin is [\d.]+x, not", text)
    assert m, ("docs/MODELS.md no longer states EX1's exhaustive worst ratio in the form "
               "this guard reads; either 64.7 was reworded or the figure moved")
    return int(round(100.0 / float(m.group(1))))


@dataclass(frozen=True)
class Figure:
    key: str
    document: str
    pattern: str
    derive: Callable[[], int]
    what: str
    scale: int = 1


#: What the customer documents state, and what produces each answer. **A figure
#: not in this tuple is a figure nothing checks**, which is the condition D69 was
#: written about -- so adding a number to a customer document means adding a row.
FIGURES = (
    Figure("detector_bytes/README", "README.md",
           r"sizeof\(Detector\) asserted exactly, ([\d,]+) B",
           _detector_bytes, "sizeof(Detector)"),
    Figure("detector_bytes/DESIGN", "docs/DESIGN.md",
           r"sizeof\(Detector\) is ([\d,]+) B",
           _detector_bytes, "sizeof(Detector)"),
    Figure("detector_bytes/EVIDENCE", "docs/EVIDENCE.md",
           r"FAILED at ([\d,]+) B",
           _detector_bytes, "sizeof(Detector)"),
    Figure("n3_predicted/EVIDENCE", "docs/EVIDENCE.md",
           r"= ([\d,]+) B \+/- 64",
           _n3_predicted_bytes, "N3's predicted footprint"),
    Figure("cycle_parameters/DESIGN", "docs/DESIGN.md",
           r"checked at every one of ([\d,]+) indices",
           _cycle_parameters, "Deep_f32's parameter extent"),
    Figure("ex1_margin/DESIGN", "docs/DESIGN.md",
           r"margin of ([\d.]+)x over the tolerance model",
           _ex1_margin_x100, "EX1's exhaustive gradient margin", scale=100),
    Figure("refusal_codes/README", "README.md",
           r"exercising all (\d+) refusal codes",
           _refusal_codes, "refusal codes in Status.hpp"),
    Figure("refusal_codes/DESIGN", "docs/DESIGN.md",
           r"OK plus (\d+) refusal codes",
           _refusal_codes, "refusal codes in Status.hpp"),
    Figure("load_cases/README", "README.md",
           r"(\d+) load cases, exercising all",
           _load_cases, "load cases in the refusal suite"),
    Figure("load_cases/DESIGN", "docs/DESIGN.md",
           r"(\d+) load cases, exercising all",
           _load_cases, "load cases in the refusal suite"),
    Figure("fprime_covered/DESIGN", "docs/DESIGN.md",
           r"(\d+) of the 12 proven to degrade",
           _fprime_covered_codes, "refusal codes the F' test covers"),
    Figure("fprime_covered/STATUS", "docs/STATUS.md",
           r"covering (\d+) of the 12 refusal codes",
           _fprime_covered_codes, "refusal codes the F' test covers"),
    Figure("max_channels/DESIGN", "docs/DESIGN.md",
           r"Compile-time maxima: (\d+) channels",
           _max_channels, "MAX_CHANNELS"),
    Figure("warmup_steps/DESIGN", "docs/DESIGN.md",
           r"warmup_steps is ([\d,]+) =",
           _warmup_steps, "warmup_steps"),
    Figure("eval_caught/EVIDENCE", "docs/EVIDENCE.md",
           r"residual fused with the first derivative [\d.]+% \d+/\d+ (\d+)/\d+",
           _eval_caught, "EVAL events the fused arm caught"),
    Figure("eval_total/EVIDENCE", "docs/EVIDENCE.md",
           r"residual fused with the first derivative [\d.]+% \d+/\d+ \d+/(\d+)",
           _eval_total, "EVAL denominator"),
    Figure("frozen_eval/EVIDENCE", "docs/EVIDENCE.md",
           r"published rule \(frozen\) [\d.]+% \d+/\d+ (\d+)/\d+",
           _frozen_eval, "EVAL events the frozen rule caught"),
    Figure("alarm_rate/EVIDENCE", "docs/EVIDENCE.md",
           r"published rule \(frozen\) ([\d.]+)%",
           _alarm_rate_x10000, "the matched alarm rate", scale=10000),
    Figure("population/EVIDENCE", "docs/EVIDENCE.md",
           r"(\d+) labelled contextual anomalies across",
           _population, "the scored population"),
    # The pattern is phrased to survive the count changing, which it did within
    # a day: "tiers 1 and 2" would have stopped matching the moment tier 3 ran,
    # and a guard that stops matching is a guard that has gone blind.
    Figure("toolkit_rungs/README", "README.md",
           r"(\d+) of its \d+ acceptance-ladder rungs have run",
           _toolkit_rungs_built, "acceptance-ladder rungs that run"),
    Figure("toolkit_rungs/STATUS", "docs/STATUS.md",
           r"(\d+) of its \d+ acceptance-ladder rungs have run",
           _toolkit_rungs_built, "acceptance-ladder rungs that run"),
    # (!) AND TWO MORE, BECAUSE THE COMMENT ABOVE DESCRIBED A SITE IT DID NOT
    # COVER. `docs/EVIDENCE.md` section 9 said the toolkit was "pre-registered and
    # unwritten" and `README.md`'s absent-paths table said "at tiers 1 and 2" --
    # both stale, both customer-facing, and both invisible here because neither
    # stated a number to check. Rewritten into the same form on 2026-09-14 and
    # pinned below. A guard that covers three of the four places a figure appears
    # is how the fourth goes stale.
    Figure("toolkit_rungs/EVIDENCE", "docs/EVIDENCE.md",
           r"(\d+) of its \d+ acceptance-ladder rungs have run",
           _toolkit_rungs_built, "acceptance-ladder rungs that run"),
    # A SECOND occurrence in `README.md`, so it needs its own pattern: `_on_master`
    # searches with `re.search`, which returns the first match and would never
    # reach this one. Anchored on the absent-paths table's own wording -- and
    # WITHOUT the `**` that surrounds the figure in the source, because
    # `_flattened` strips emphasis before any pattern sees it.
    Figure("toolkit_rungs/README_absent", "README.md",
           r"the toolkit itself, whose (\d+) of its \d+ acceptance-ladder rungs",
           _toolkit_rungs_built, "acceptance-ladder rungs that run"),
)

#: (!) The debt register, and it is EMPTY, which it has not been before.
#:
#: It held four when the guard was written -- 227 tracked files, 36 scripts, 353
#: index entries, 683 tests -- every one a figure the old public README stated
#: about `dev` and nothing re-derived. **All four died with that README**, and the
#: customer documents state figures that are checked instead of counted.
#:
#: It pins what `master` STATES and not what `dev` derives, and the first draft of
#: this guard got that backwards. Pinning the derived half looked more exact and
#: was a liability: it moved the moment any commit added a file, so ordinary work
#: would turn the suite red for a defect that had not changed.
BASELINE: dict[str, int] = {}

#: On `master`, covered by no `MASTER_PREFIXES` entry, and harmless because it
#: carries no citations; and declared there but not on the branch, because the
#: licence is not selected (`docs/STATUS.md` 7 item H).
PREFIX_ALLOWANCES = {"uncovered": {".gitignore"}, "unused": {"LICENSE"}}


def survey(documents: dict[str, str] | None = None) -> dict[str, tuple[int, int]]:
    """`key -> (stated, derived)` for every figure whose two sides disagree."""
    if documents is None:
        documents = {d: _on_master(d) for d in {f.document for f in FIGURES}}
    stale = {}
    for figure in FIGURES:
        match = re.search(figure.pattern, documents[figure.document])
        assert match, (
            f"{MASTER}:{figure.document} no longer states {figure.what!r} in the "
            f"form this guard reads ({figure.pattern!r}). Either the figure was "
            "removed -- then remove the row -- or the wording changed and this "
            "guard has gone blind, which is worse than a stale number.")
        raw = match.group(1).replace(",", "")
        stated = int(round(float(raw) * figure.scale)) if figure.scale != 1 else int(raw)
        derived = figure.derive()
        if stated != derived:
            stale[figure.key] = (stated, derived)
    return stale


@needs_master
def test_every_figure_master_states_is_current() -> None:
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
        "A figure a customer document states must be re-derivable from the source "
        "that produces it (D69).")
    assert not moved, (
        f"`{MASTER}` now states something different: {moved} (was, now). "
        "Re-measure and update the register, with the date.")
    assert not fixed, (
        f"FIXED, so remove them from BASELINE in the same commit: {sorted(fixed)}. "
        "A debt register that outlives its debt stops describing anything.")


@needs_master
def test_the_check_actually_catches_things() -> None:
    """The proof, and it needs a probe now that the register is empty.

    Every figure is corrupted in turn, in memory, and the guard must report that
    one and only that one. A check nobody has watched fail is not known to work
    (`tests/test_no_list.py:18` is the same argument for the LIST guard).
    """
    clean = {d: _on_master(d) for d in {f.document for f in FIGURES}}
    assert survey(clean) == {}, "master is not clean; fix that before trusting this"

    for figure in FIGURES:
        probe = dict(clean)
        match = re.search(figure.pattern, probe[figure.document])
        span = match.span(1)
        text = probe[figure.document]
        probe[figure.document] = text[:span[0]] + "99999" + text[span[1]:]
        found = survey(probe)
        assert set(found) == {figure.key}, (
            f"corrupting {figure.key} was reported as {sorted(found)}")
        assert found[figure.key][0] != found[figure.key][1]


@needs_master
def test_no_customer_document_says_the_refusal_coverage_is_incomplete() -> None:
    """(!) The figure was guarded and the prose contradicting it was not.

    Until 2026-09-20 `master:docs/DESIGN.md` stated **both** "12 of the 12 proven to
    degrade to Level 1" in its table and, four lines below, "The twelfth is not in that
    loop ... the F' component's unit test still iterates the eleven that existed before
    it". The first is derived by `_fprime_covered_codes` and moved when the gap closed at
    `docs/MODELS.md` 20.13; the second is prose, invisible to the `FIGURES` regex, and
    nobody deleted it. A customer reading that section could not tell which half was true.

    So the register is not enough on its own: a figure that moves can leave a sentence
    behind that says the opposite. This checks the one claim whose figure is already
    derived, and fails if any customer document asserts the coverage is partial while the
    source says it is complete.
    """
    covered = _fprime_covered_codes()
    total = _refusal_codes()
    if covered != total:
        pytest.skip(f"coverage is genuinely partial ({covered} of {total}); the caveat belongs")
    stale = (
        "is not in that loop",
        "still iterates the eleven",
        "the topology-level degradation test does not",
    )
    found = []
    for doc in CUSTOMER_DOCS:
        text = _on_master(doc)
        for phrase in stale:
            if phrase in text:
                found.append(f"{doc}: {phrase!r}")
    assert not found, (
        f"the F' test covers {covered} of {total} refusal codes, and these say otherwise: "
        f"{found}. A figure that moved left a sentence behind."
    )


@needs_master
def test_flight_and_fprime_are_byte_identical_to_dev() -> None:
    """D69 consequence 1 carries both across unchanged, and unchanged is checkable.

    It matters beyond tidiness: until 2026-09-11 `master` carried a lint target
    that could not fail. A public branch whose build differs from the branch the
    evidence was measured on is a public branch nobody can check.
    """
    for tree in ("flight", "fprime"):
        assert _git("rev-parse", f"{MASTER}:{tree}").strip() \
            == _git("rev-parse", f"dev:{tree}").strip(), (
            f"`{tree}/` differs between `{MASTER}` and `dev`")


@needs_master
def test_the_snapshot_commit_master_names_exists_and_is_accurate() -> None:
    """`master` tells a reader which `dev` commit its citations resolve at.

    D67 consequence 5 makes that convention load-bearing for hundreds of
    citations, so the commit it names has to be one a reader can check out, and
    every file `master` is supposed to carry unchanged has to be the one that
    commit holds. The previous public README named a commit two behind the branch,
    because a later commit ported two dataset files without moving it.
    """
    match = re.search(r"points into dev at commit ([0-9a-f]{7,40})",
                      _on_master("README.md"))
    assert match, "master:README.md no longer names the dev commit it resolves at"
    cited = match.group(1)
    assert _ref_exists(cited), (
        f"master names dev commit {cited}, which this repository does not hold")

    on_master = set(_git("ls-tree", "-r", "--name-only", MASTER).split())
    at_cited = set(_git("ls-tree", "-r", "--name-only", cited).split())
    differs = sorted(
        path for path in (on_master & at_cited) - CURATED_ON_MASTER
        if _git("rev-parse", f"{MASTER}:{path}") != _git("rev-parse", f"{cited}:{path}")
    )
    assert not differs, (
        f"paths on `{MASTER}` that differ from the commit it names ({cited}): "
        f"{differs}. Either move the stated commit or curate them deliberately.")

    for doc in CUSTOMER_DOCS:
        assert cited in _on_master(doc), (
            f"{MASTER}:{doc} does not name the snapshot commit its figures resolve at")


@needs_master
def test_every_citation_in_a_master_document_resolves() -> None:
    """(!) `--master` mode never reads `master`, so nothing checked these.

    `scripts/check_references.py:172-177` resolves against `dev`'s working tree and
    classifies by `MASTER_PREFIXES`: it checks **`dev`'s** documents under the
    curated branch's rules. The customer documents are not on `dev`, so no guard
    had ever read a citation in them. This one does, with the same patterns.
    """
    on_master = set(_git("ls-tree", "-r", "--name-only", MASTER).split())
    on_dev = set(_git("ls-files").split())
    breaks = []
    for doc in CUSTOMER_DOCS:
        text = _git("show", f"{MASTER}:{doc}")
        for m in C.REPO_PATH.finditer(text):
            path = m.group(1)
            # (!) The same two exemptions `scripts/check_references.py:211` makes, and
            # this test did not: upstream `nasa/fprime` documentation namespace-collides
            # with this repository's own `docs/`, and `fprime/lib/` is gitignored. Without
            # them a master document could not cite F's user manual, which is exactly what
            # master's integration section has to do to send a reader to the Passive
            # Adapter Pattern.
            if path.startswith(C.UPSTREAM_DOC_PREFIXES) or path in C.FPRIME_CHECKOUT_PATHS:
                continue
            if path not in on_master and path not in on_dev:
                breaks.append(f"{doc}: path `{path}`")
        for m in C.SECTION.finditer(text):
            document, section = m.group(1), m.group(2)
            # (!) A master-only document's sections resolve on MASTER. `docs/DESIGN.md`
            # and `docs/EVIDENCE.md` are not on `dev` at all (MASTER_ONLY), so reading
            # their headings from the dev working tree reported them as existing on
            # neither branch -- which is how this test read a correct citation as broken.
            if document in on_dev:
                available = C.headings(ROOT / document)
            elif document in on_master:
                blob = _git("show", f"{MASTER}:{document}")
                tmp = ROOT / ".pytest_cache" / f"_master_{document.replace('/', '_')}"
                tmp.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(blob, encoding="utf-8")
                available = C.headings(tmp)
            else:
                breaks.append(f"{doc}: section in `{document}`, which is on neither branch")
                continue
            if section not in available:
                breaks.append(f"{doc}: `{document}` {section} -- no such section")
        for m in C.MD_LINK.finditer(text):
            target = m.group(1)
            resolved = str((Path(doc).parent / target)).replace("./", "")
            if resolved not in on_master and target not in on_master:
                breaks.append(f"{doc}: link {target} -- 404 for a reader on `{MASTER}`")
    assert not breaks, "citations that do not resolve:\n  " + "\n  ".join(breaks)


@needs_master
def test_master_prefixes_still_describes_the_branch() -> None:
    """`--master` mode classifies by a hand-kept list and never reads the branch.

    So the list can drift from `master` silently, and a citation would then be
    reported as dev-resolving when it is on the branch, or the reverse. This is
    the comparison that mode does not make (D69 consequence 4).
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
def test_master_adds_only_its_own_customer_documents() -> None:
    """D69 gives `master` four documents of its own and nothing else.

    Anything else appearing only there would be a file no guard on `dev` has ever
    seen -- unreviewed by construction, which is what the curation exists to
    prevent.
    """
    on_master = set(_git("ls-tree", "-r", "--name-only", MASTER).split())
    on_dev = set(_git("ls-files").split())
    assert on_master - on_dev == MASTER_ONLY, (
        f"present only on `{MASTER}`: {sorted(on_master - on_dev)}; "
        f"declared: {sorted(MASTER_ONLY)}")
    for doc in CUSTOMER_DOCS:
        assert doc in on_master, f"{doc} is declared a customer document and is absent"
