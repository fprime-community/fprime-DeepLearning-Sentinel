"""Figures that prose used to protect, now guarded.

`docs/MODELS.md` 52.8's lesson, applied: **a note asking a future reader to
remember something is not a fix.** Three figures in this repository were
protected by prose and by nothing else, and each had already drifted or was
recorded as unguarded when it was found:

- The weight store's size. `docs/MODELS.md` 10.7 calls it a standing gate that
  "the run asserts", but no test under `tests/` asserted it.
- Tracked content, against D64's cap and `docs/MODELS.md` 39's N8 stop.
  `docs/MODELS.md` 47.16's rider records that "no guard re-derives any of
  these numbers", and by then the figure of record had been stale for two sections.
- `docs/INDEX.md`'s row for `docs/MODELS.md`. 47.14 recorded that "nothing
  guards it -- it is prose describing a document, and no test re-derives it",
  declined to close the gap, and the row went on to fall thirteen sections
  behind.

Each test below states the decision it enforces, so that a failure sends the
reader to the entry rather than to this file.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: `docs/MODELS.md` 47.13.1 and 10.7. The store is evidence, and a run that
#: changes it has changed what every earlier figure was measured against.
WEIGHT_STORE = 1313

#: D64. The hard cap on tracked content.
CAP_MIB = 8.0

#: `docs/MODELS.md` 39's N8: under 6 MiB holds, 6 to 7 is no verdict, above the
#: stop is **a stop -- report rather than trimming coverage to fit**. This test
#: is that stop, so that it fires rather than being noticed.
#:
#: **D71 moved it from 7.0 to 7.5**, on D64's own finding that the byte count was
#: never the guard, and because it is documentation rather than vectors that moves
#: the number -- N8 predicted the vectors would fit, and they still do. D64's 8 MiB
#: cap did not move with it.
#:
#: **D77 then took D71's parked alternative 3.** N8's SUBJECT is the vectors;
#: "tracked content" was its INSTRUMENT, correct when the vectors dominated the
#: tree and worse with every megabyte of prose since. So N8's bands are now read
#: against the committed vectors, and total tracked content is guarded by D64's cap
#: alone -- the only figure ever derived about tracked content as a whole.
N8_STOP_MIB = 7.0

#: Every committed test vector lives here. D77 checked: a tracked-file sweep for
#: `.bin`, `.npz`, `.npy`, `.pvec`, `.dat` and `.vec` outside this directory returns
#: nothing, so the directory IS the figure rather than standing in for it.
VECTOR_ROOT = "flight/test/vectors"
VECTOR_SUFFIXES = (".bin", ".npz", ".npy", ".pvec", ".dat", ".vec")


def _tracked_bytes() -> int:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True,
                         capture_output=True)
    names = [n for n in out.stdout.split(b"\0") if n]
    return sum((ROOT / n.decode()).stat().st_size
               for n in names if (ROOT / n.decode()).is_file())


def test_the_weight_store_is_the_size_every_figure_was_measured_against() -> None:
    found = len(list((ROOT / "runs" / "_weights").glob("*.npz")))
    assert found == WEIGHT_STORE, (
        f"the weight store holds {found} .npz, not {WEIGHT_STORE}. It is a standing "
        "gate (docs/MODELS.md 10.7): every figure in docs/RESULTS.md was measured "
        "against this store, and a run that moved it has invalidated them.")


def _vector_bytes() -> int:
    """The committed test vectors, which is what `docs/MODELS.md` 39's N8 measured."""
    out = subprocess.run(["git", "ls-files", "-z", VECTOR_ROOT], cwd=ROOT, check=True,
                         capture_output=True)
    names = [n for n in out.stdout.split(b"\0") if n]
    return sum((ROOT / n.decode()).stat().st_size
               for n in names if (ROOT / n.decode()).is_file())


def _stray_vectors() -> list[str]:
    """Vector-shaped files tracked OUTSIDE `VECTOR_ROOT`.

    D77 rests on the directory being the whole figure. If that stops being true the
    vectors band silently stops measuring the vectors, which is the failure mode D71
    named in the guard it replaced.
    """
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout.split()
    return [n for n in out
            if n.endswith(VECTOR_SUFFIXES) and not n.startswith(VECTOR_ROOT + "/")]


def _check_size_bands(vector_bytes: int, total_bytes: int) -> None:
    """D64's cap on the tree, and N8's bands on the vectors. A pure function.

    Separated from the measurement so that both can be exercised on synthetic byte
    counts. **D71 consequence 5: a gate nobody has seen fail is not known to work**,
    and proving this one by committing 7 MiB of vectors would be the exact opposite
    of what N8 asks for.

    **D77 split the two arguments**, because they guard different things: the cap is
    about the whole tree and N8 is about the vectors. Before the split a megabyte of
    prose could fire a stop whose text says "do not delete evidence to get back under
    the line", which is advice about vectors and unactionable about prose.
    """
    total_mib = total_bytes / (1024 * 1024)
    assert total_mib < CAP_MIB, (
        f"tracked content is {total_mib:.2f} MiB, over D64's {CAP_MIB} MiB cap. D64 "
        "consequence 5: the question is asked again, and the answer may be that prose "
        "belongs somewhere else.")
    vector_mib = vector_bytes / (1024 * 1024)
    assert vector_mib < N8_STOP_MIB, (
        f"the committed vectors are {vector_mib:.2f} MiB, past docs/MODELS.md 39's N8 "
        f"stop at {N8_STOP_MIB} MiB (D77). N8 says report rather than trimming coverage "
        "to fit: bring it to the owner and take a decision, do not delete a tier to get "
        "back under the line.")


def test_tracked_content_is_inside_the_cap_and_the_vectors_below_39s_N8_stop() -> None:
    _check_size_bands(_vector_bytes(), _tracked_bytes())


def test_every_committed_vector_is_where_the_vectors_band_looks() -> None:
    """D77's premise, asserted rather than assumed.

    If a vector is committed outside `VECTOR_ROOT` the band stops measuring the
    thing it is named after, and it does so silently.
    """
    strays = _stray_vectors()
    assert not strays, (
        f"vector-shaped files tracked outside {VECTOR_ROOT}/: {strays}. D77 read N8's "
        "bands against that directory on the finding that it holds every committed "
        "vector. Either move them, or widen VECTOR_ROOT and say so in a rider.")


def test_the_N8_stop_and_the_cap_both_still_fire_above_their_bands() -> None:
    """D71 consequence 5 and D77 consequence 6. Demonstrated rather than assumed.

    `flight/Makefile:85-89` is the precedent this exists against: a lint target
    ran three times, checked no exit status and echoed "clean" unconditionally,
    so it could not fail and nobody knew for months. When a band moves, the guard
    is re-run at the new value in both directions.
    """
    stop = int(N8_STOP_MIB * 1024 * 1024)
    cap = int(CAP_MIB * 1024 * 1024)
    small_total, small_vectors = 1024, 1024

    # Just inside both: passes, as it must.
    _check_size_bands(stop - 1, cap - 1)

    # The vectors band fires on oversized VECTORS.
    with pytest.raises(AssertionError, match="N8 stop"):
        _check_size_bands(stop + 1, small_total)

    # The cap fires on an oversized TREE.
    with pytest.raises(AssertionError, match="cap"):
        _check_size_bands(small_vectors, cap + 1)

    # (!) AND THE POINT OF D77: prose past the OLD 7.5 MiB tracked-content stop no
    # longer fires N8, because N8 is not about prose. A tree of 7.9 MiB holding
    # 1 KiB of vectors is inside the cap and says nothing about coverage.
    _check_size_bands(small_vectors, int(7.9 * 1024 * 1024))


def test_the_index_row_for_models_reaches_the_highest_section() -> None:
    models = (ROOT / "docs" / "MODELS.md").read_text(encoding="utf-8")
    highest = max(int(n) for n in re.findall(r"^## (\d+)\.", models, re.M))
    index = (ROOT / "docs" / "INDEX.md").read_text(encoding="utf-8")
    row = next(ln for ln in index.splitlines() if "](MODELS.md)" in ln)
    named = {int(n) for n in re.findall(r"\d+", " ".join(re.findall(r"\(([\d,\s]+)\)", row)))}
    assert named and max(named) >= highest, (
        f"docs/MODELS.md reaches section {highest}; docs/INDEX.md's row names up to "
        f"{max(named) if named else 'nothing'}. 47.14 recorded this gap and declined to "
        "close it silently; it is guarded now, so extend the row in the same commit as "
        "the section.")


#: `docs/MODELS.md` 54.2b, made a standing rule by D72. Sections below this were
#: written before the rule existed; judging them by it would rewrite the record
#: rather than improve it (D72 consequence 3).
DERIVATION_RULE_FROM = 55


def test_new_pre_registrations_state_where_their_requirements_came_from() -> None:
    """`docs/MODELS.md` 54.2b and D72, and it is a drafting step rather than a citation count.

    Three pre-registrations specified something the target does not permit -- 50.1's
    four-wide tape, 52.4's undimensionable probe, 54.2's incomplete primitive list --
    and all three were written from what had been explored rather than from what the
    flight target requires. The rule permits an honest ``none``: 51's tolerance model
    is derived from the summation error bound and has no flight source.
    """
    text = (ROOT / "docs" / "MODELS.md").read_text(encoding="utf-8")
    heads = [(m.start(), int(m.group(1)), m.group(2))
             for m in re.finditer(r"^## (\d+)\. (.*)$", text, re.M)]
    heads.append((len(text), 10 ** 6, ""))
    missing = []
    for i in range(len(heads) - 1):
        start, number, title = heads[i]
        if number < DERIVATION_RULE_FROM or "Pre-registration" not in title:
            continue
        body = text[start:heads[i + 1][0]]
        if "REQUIREMENTS DERIVED FROM:" not in body:
            missing.append(number)
    assert not missing, (
        f"pre-registration section(s) {missing} carry no 'REQUIREMENTS DERIVED FROM:' line. "
        "docs/MODELS.md 54.2b: derive the section's requirements from the source of record "
        "before committing it, and state where from -- or write 'none' with a reason, which "
        "51's tolerance model would legitimately do.")


#: `docs/MODELS.md` 47.1 row 31 named three figures in `docs/STATUS.md`'s gate
#: block that are "re-derived by **nothing**": the `check_no_list` file count, the
#: selftest ratio, and the three lint configurations. It observed that all three
#: "happen to be correct today" and left them. **The file count then drifted from
#: 104 to 107**, found 2026-09-17 -- which is the failure mode 52.8 names and this
#: module exists for. Two of the three are guarded below.
#:
#: The third, the lint configurations, is deliberately not: `flight/Makefile:90-94`
#: skips the target entirely when `clang-tidy` is absent, so a test asserting three
#: configurations would fail on a machine where the gate itself correctly passes.
GATE_BLOCK = "docs/STATUS.md"


def _gate_block_line(fragment: str) -> str:
    text = (ROOT / GATE_BLOCK).read_text(encoding="utf-8")
    line = next((ln for ln in text.splitlines() if fragment in ln), None)
    assert line is not None, (
        f"{GATE_BLOCK} no longer carries a gate line containing {fragment!r}; "
        "this test is stale, not the document.")
    return line


def test_the_gate_block_states_the_file_count_check_no_list_reports() -> None:
    """`docs/MODELS.md` 47.1 row 31, closed. The figure is re-derived, not recalled."""
    proc = subprocess.run([sys.executable, "scripts/check_no_list.py"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, f"check_no_list is not clean:\n{proc.stdout}{proc.stderr}"
    reported = re.search(r"check_no_list: (\d+) files clean", proc.stdout)
    assert reported, f"could not read a file count from check_no_list:\n{proc.stdout}"

    stated = re.search(r"#\s*(\d+) files", _gate_block_line("check_no_list.py"))
    assert stated, f"{GATE_BLOCK}'s check_no_list gate line states no file count"
    assert int(stated.group(1)) == int(reported.group(1)), (
        f"check_no_list reports {reported.group(1)} files; {GATE_BLOCK} says "
        f"{stated.group(1)}. It drifted from 104 to 107 unguarded once already.")


def test_the_gate_block_states_the_ratio_the_selftest_reports() -> None:
    """The second of 47.1 row 31's three. The oracle's 1.0 is asserted by the selftest
    itself; what was unguarded is the ratio the document quotes back."""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.run([sys.executable, "-m", "sentinel_eval", "selftest"],
                          cwd=ROOT, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, f"the selftest did not pass:\n{proc.stdout}{proc.stderr}"
    reported = re.search(r"(\d+)/(\d+) checks passed", proc.stdout)
    assert reported, f"could not read a ratio from the selftest:\n{proc.stdout}"

    stated = re.search(r"#\s*(\d+)\s*/\s*(\d+)", _gate_block_line("sentinel_eval selftest"))
    assert stated, f"{GATE_BLOCK}'s selftest gate line states no ratio"
    assert stated.groups() == reported.groups(), (
        f"the selftest reports {reported.group(0)}; {GATE_BLOCK} says "
        f"{stated.group(1)}/{stated.group(2)}.")


def test_no_unit_test_artifact_is_committed_outside_the_vector_directory() -> None:
    """(!) A C++ UNIT TEST THAT DIRTIES THE SOURCE TREE, CAUGHT BY NAME.

    Section 73's reload tests name their candidate file by a SHORT path, because
    `FW_CMD_STRING_MAX_SIZE` is 40 and a longer one is truncated at the command
    boundary. A short path is a relative path, and the first version of those
    tests wrote it wherever the runner happened to be -- four `.bin` files in
    `fprime/`, swept into a commit by `git add -A` before anything objected.

    Nothing guarded it. `test_no_local_persistence.py` counts untracked content
    toward D64's cap but does not forbid it, and the files were small. The Python
    side has had this rule since work item 8; the C++ side had it only as a
    convention, and a convention is what this is replacing.

    `flight/test/vectors/` is the one place a committed binary belongs (D77).
    """
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=str(ROOT),
        capture_output=True, text=True, timeout=60).stdout
    binaries = [name for name in tracked.split("\0")
                if name.endswith((".bin", ".npz", ".npy", ".dat"))]
    strays = [name for name in binaries if not name.startswith(VECTOR_ROOT + "/")]
    assert not strays, (
        f"committed binaries outside {VECTOR_ROOT}/: {strays}. A unit test that "
        "writes into the source tree is what this catches; give it an absolute "
        "path under the build cache, or move the process there first.")
