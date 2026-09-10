"""Regenerate `docs/TELEMANOM_EXCERPTS.md` section 5 from the citations in the tree.

The section is the navigation index into `third_party/telemanom/`: every location
this repository cites into the published source, and what cites it. It was
hand-maintained, and a hand-maintained index of 64 rows drifts -- `docs/REORG_PLAN.md`
3b found rows missing and "Cited by" columns wrong. This script is the generator
the correction was owed, so the next drift is a diff rather than an audit.

WHAT COUNTS AS A CITATION
-------------------------
A module name from the vendored package followed by a line or line range:
`errors.py:403`, `telemanom/channel.py:69-82`,
`third_party/telemanom/telemanom/errors.py:386-435`. The bare form dominates,
because a document establishes the package once and then cites within it.

(!) `detector.py` IS EXCLUDED FROM THE BARE FORM, AND THE REASON IS MEASURED.
The vendored package has a `detector.py` and so does this repository
(`src/sentinel_eval/detector.py`). Every bare `detector.py:NNN` in the tree --
`docs/MODEL_FILE.md:200`, `docs/MODELS.md:2879`, `:4708`,
`src/sentinel_models/baseline_reference.py:53`, `scripts/make_baseline_vectors.py:37`
and the rest -- resolves to **this repository's** module, not the vendored one.
Matching it bare would put a dozen false rows into an index whose whole value is
that a reader can trust it. So `detector.py` is counted only when written with an
explicit `telemanom/` or `third_party/` prefix. `docs/REORG_PLAN.md:237` records
the one genuine vendored-`detector.py` citation and disambiguates it in prose.

Run: `PYTHONPATH=src .venv/bin/python scripts/telemanom_citations.py --check`
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "third_party" / "telemanom" / "telemanom"
INDEX = ROOT / "docs" / "TELEMANOM_EXCERPTS.md"

#: Cited bare. `detector.py` is deliberately absent -- see the module docstring.
BARE_MODULES = ("channel.py", "errors.py", "modeling.py", "helpers.py")
#: Cited only with an explicit package prefix, because the name is ambiguous.
PREFIXED_ONLY = ("detector.py", "plotting.py", "__init__.py")

_BARE = re.compile(
    r"(?:third_party/telemanom/)?(?:telemanom/)?\b("
    + "|".join(m.replace(".", r"\.") for m in BARE_MODULES)
    + r"):(\d+(?:[-,]\d+)*)")
_PREFIXED = re.compile(
    r"(?:third_party/telemanom/telemanom/|telemanom/)("
    + "|".join(m.replace(".", r"\.") for m in PREFIXED_ONLY)
    + r"):(\d+(?:[-,]\d+)*)")


def tracked_prose() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout.split()
    return [ROOT / f for f in out
            if f.endswith((".md", ".py"))
            and not f.startswith("third_party/")
            and f != "docs/TELEMANOM_EXCERPTS.md"]


def citations() -> dict[str, set[str]]:
    """`telemanom/<module>:<lines>` -> the repository files that cite it."""
    found: dict[str, set[str]] = defaultdict(set)
    for path in tracked_prose():
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT).as_posix()
        for pattern in (_BARE, _PREFIXED):
            for match in pattern.finditer(text):
                found[f"telemanom/{match.group(1)}:{match.group(2)}"].add(rel)
    return dict(found)


def _sort_key(location: str) -> tuple[str, int, int]:
    module, _, lines = location.partition(":")
    first = re.split(r"[-,]", lines)[0]
    last = re.split(r"[-,]", lines)[-1]
    return module, int(first), int(last)


def table(found: dict[str, set[str]]) -> str:
    rows = ["| Location | Cited by |", "|---|---|"]
    for location in sorted(found, key=_sort_key):
        citers = ", ".join(f"`{c}`" for c in sorted(found[location]))
        rows.append(f"| `{location}` | {citers} |")
    return "\n".join(rows)


def overruns(found: dict[str, set[str]]) -> list[str]:
    """Citations whose last line is past the end of the file they name."""
    bad = []
    for location in found:
        module, _, lines = location.partition(":")
        source = VENDORED / Path(module).name
        if not source.exists():
            bad.append(f"{location}: {source} is not in the vendored package")
            continue
        end = int(re.split(r"[-,]", lines)[-1])
        length = len(source.read_text(encoding="utf-8").splitlines())
        if end > length:
            bad.append(f"{location}: line {end} is past {module}'s {length} lines")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report the counts and any overrun, write nothing")
    args = ap.parse_args()

    found = citations()
    citers = {c for v in found.values() for c in v}
    print(f"telemanom_citations: {len(found)} distinct locations "
          f"cited across {len(citers)} files")

    bad = overruns(found)
    for line in bad:
        print(f"  OVERRUN  {line}")

    if not args.check:
        print(table(found))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
