"""Figures that prose used to protect, now guarded.

`docs/MODELS.md` 52.8's lesson, applied: **a note asking a future reader to
remember something is not a fix.** Three figures in this repository were
protected by prose and by nothing else, and each had already drifted or was
recorded as unguarded when it was found:

- The weight store's size. `docs/MODELS.md` 10.7 calls it a standing gate that
  "the run asserts", but no test under `tests/` asserted it.
- Tracked content, against D64's cap and `docs/MODELS.md` 39's N8 stop.
  `docs/MODELS.md` 47.16a records that "no guard re-derives any of these
  numbers", and by then the figure of record had been stale for two sections.
- `docs/INDEX.md`'s row for `docs/MODELS.md`. 47.14 recorded that "nothing
  guards it -- it is prose describing a document, and no test re-derives it",
  declined to close the gap, and the row went on to fall thirteen sections
  behind.

Each test below states the decision it enforces, so that a failure sends the
reader to the entry rather than to this file.
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: `docs/MODELS.md` 47.13.1 and 10.7. The store is evidence, and a run that
#: changes it has changed what every earlier figure was measured against.
WEIGHT_STORE = 1313

#: D64. The hard cap on tracked content.
CAP_MIB = 8.0

#: `docs/MODELS.md` 39's N8: under 6 MiB holds, 6 to 7 is no verdict, above 7 is
#: **a stop -- report rather than trimming coverage to fit**. This test is that
#: stop, so that it fires rather than being noticed.
N8_STOP_MIB = 7.0


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


def test_tracked_content_is_inside_the_cap_and_below_39s_N8_stop() -> None:
    mib = _tracked_bytes() / (1024 * 1024)
    assert mib < CAP_MIB, (
        f"tracked content is {mib:.2f} MiB, over D64's {CAP_MIB} MiB cap.")
    assert mib < N8_STOP_MIB, (
        f"tracked content is {mib:.2f} MiB, past docs/MODELS.md 39's N8 stop at "
        f"{N8_STOP_MIB} MiB. N8 says report rather than trimming coverage to fit: "
        "bring it to the owner and take a decision (raise the stop, split the "
        "bands, or accept it), do not delete evidence to get back under the line.")


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
