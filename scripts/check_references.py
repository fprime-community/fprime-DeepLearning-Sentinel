"""Every citation in the prose resolves, and it keeps resolving.

`docs/REORG_PLAN.md` 3 audited six kinds of reference by hand and found two of
them clean. **This is the guard that keeps them clean** -- it exists to hold a
result, not to discover one. Run it over the whole tree, or over the curated
public branch with `--master`.

WHAT IT CHECKS

  section     `docs/MODELS.md` 26.18 -> a heading numbered 26.18 exists there
  path        `src/sentinel_eval/detector.py` -> the file is tracked
  line        `telemanom.py:53-58` -> the range is inside the file
  link        [text](../Objective.md) -> the target exists
  node id     `tests/test_x.py::test_y` -> the test exists

(!) THREE CONVENTIONS A NAIVE CHECKER FLAGS WRONGLY, all three measured in
`docs/REORG_PLAN.md` 3 before this existed:

  1. `Objective.md` **14.N** is a TABLE ROW, not a heading. Stated exactly once.
  2. Five cited `docs/...` paths are upstream `nasa/fprime` documentation, which
     namespace-collides with this repository's own `docs/`.
  3. In reverse: `detector.py:167-173` at `scripts/smap_rungs.py:1263` points into
     the **vendored** `detector.py` (254 lines), not `src/sentinel_eval/detector.py`
     (163). Flagging it would be wrong in the other direction.

(!) AND THE `--master` MODE, WHICH D67 REQUIRES.
`master` carries the component and the evidence it works and nothing else, so a
citation from `docs/DECISIONS.md` into `scripts/` does not resolve there -- about
380 of them do not. That is **by design and is not a break**: D67's convention is
that any off-branch path resolves on `dev` at the named commit. So in master mode a
path absent from the curated set but present on `dev` is reported as
**dev-resolving**, and only a path absent from **both** is a break.

Run: `.venv/bin/python scripts/check_references.py [--master] [--verbose]`
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: D67's two lists. Kept here because the guard is what enforces them.
MASTER_PREFIXES = (
    "flight/", "fprime/Sentinel/Monitor/", "fprime/SentinelRef/",
    "fprime/CMakeLists.txt", "fprime/library.cmake", "fprime/settings.ini",
    "README.md", "Objective.md", "docs/STATUS.md", "docs/RESULTS.md",
    "docs/DECISIONS.md", "docs/datasets/", "docs/MODEL_FILE.md",
    "docs/FPRIME.md", "LICENSE",
)

#: Documents whose citations are checked. Source files cite too, and are included.
def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout.split()
    return out


def prose(files: list[str]) -> list[str]:
    return [f for f in files
            if f.endswith(".md")
            or (f.endswith(".py") and f.startswith(("src/", "scripts/", "tests/")))]


# -- the three taught conventions ---------------------------------------------

#: `Objective.md` 14.N, a table row rather than a heading. Exactly one site.
TABLE_ROW_CITATIONS = {("Objective.md", "14.N")}

#: Upstream `nasa/fprime` documentation, which namespace-collides with this
#: repository's own `docs/`. Matched by prefix rather than by an exact list: F'
#: has hundreds of these and an enumerated list would be a second thing to keep
#: right. This repository's own `docs/` is flat -- every file is `docs/NAME.md`
#: or `docs/datasets/NAME.md` -- so a nested path under one of these is upstream.
UPSTREAM_DOC_PREFIXES = ("docs/user-manual/", "docs/how-to/", "docs/reference/",
                         "docs/tutorials/", "docs/documentation/")

#: Files the F' checkout provides. It is gitignored and rebuilt by
#: `scripts/fprime_setup.sh` (docs/FPRIME.md), so citing one is citing something
#: that exists after a build and not in the tree.
FPRIME_CHECKOUT_PATHS = {"fprime/requirements.txt"}

#: (!) Cited today, written by a later tranche of `docs/REORG_PLAN.md`. The debt
#: is listed rather than hidden, and it shrinks as the tranches land -- an empty
#: dict is the goal, and a stale entry here fails the test that pins it.
PLANNED = {
    "docs/datasets/ESA_ADB.md": "REORG_PLAN tranche 4",
    "docs/datasets/SMAP_MSL.md": "REORG_PLAN tranche 4",
    "docs/datasets/OTHERS.md": "REORG_PLAN tranche 4",
    "docs/datasets/REPRODUCING.md": "REORG_PLAN tranche 4",
    "docs/PI_ENVELOPE.md": "REORG_PLAN tranche 4",
}

#: Bare filenames that resolve into the vendored package rather than into this
#: repository, at the sites that cite them.
VENDORED_FILENAMES = {"detector.py", "channel.py", "errors.py", "modeling.py",
                      "helpers.py"}

SECTION = re.compile(
    r"`(docs/[A-Za-z_]+\.md|Objective\.md|CHANGELOG\.md|README\.md)`\s+"
    r"(\d+(?:\.\d+)*[a-z]?)\b")
#: A real extension, not any dotted suffix: `flight_reference.ratios` is a
#: module attribute and was flagged as a missing file until this said so.
FILE_SUFFIXES = ("py", "md", "cpp", "hpp", "json", "txt", "sh", "cmake", "ini",
                 "fpp", "fppi", "yml", "yaml", "toml", "cfg", "csv", "npz",
                 "npy", "bin", "vec", "bvec", "tvec", "dvec", "fvec", "pvec")
REPO_PATH = re.compile(
    r"`((?:docs|src|scripts|tests|flight|fprime|third_party)/[A-Za-z0-9_./+-]+"
    r"\.(?:" + "|".join(FILE_SUFFIXES) + r"))`")
LINE_CITE = re.compile(
    r"`((?:[A-Za-z0-9_./+-]+/)?[A-Za-z0-9_]+\.(?:py|cpp|hpp|md)):"
    r"(\d+)(?:-(\d+))?`")
MD_LINK = re.compile(r"\[[^\]]*\]\((?!https?:)([^)#]+)(?:#[^)]*)?\)")
NODE_ID = re.compile(r"`(tests/[A-Za-z0-9_]+\.py)::([A-Za-z0-9_]+)`")

HEADING = re.compile(r"^#{1,6}\s+(\d+(?:\.\d+)*[a-z]?)[.\s]")
#: `CHANGELOG.md` numbers its sections `## [0.6.1] - date - title`.
VERSION_HEADING = re.compile(r"^#{1,6}\s+\[([0-9][0-9.]*)\]")
#: `docs/REORG_PLAN.md` labels findings `**3b. ...**` in prose rather than as
#: headings, and they are cited as `docs/REORG_PLAN.md` 3b.
BOLD_LABEL = re.compile(r"^\*\*(\d+[a-z]?)\.\s")


def headings(path: Path) -> set[str]:
    out = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        v = VERSION_HEADING.match(line)
        if v:
            out.add(v.group(1))
        b = BOLD_LABEL.match(line)
        if b:
            out.add(b.group(1))
        m = HEADING.match(line)
        if m:
            out.add(m.group(1))
            # `26.18` also answers a citation of `26`
            parts = m.group(1).split(".")
            for i in range(1, len(parts)):
                out.add(".".join(parts[:i]))
    return out


class Report:
    def __init__(self) -> None:
        self.checked = {k: 0 for k in ("section", "path", "line", "link", "node")}
        self.breaks: list[str] = []
        self.dev_resolving: list[str] = []


def check(master: bool = False) -> Report:
    files = tracked()
    on_disk = set(files)
    report = Report()
    heading_cache: dict[str, set[str]] = {}

    def resolves(path: str) -> tuple[bool, bool]:
        """`(exists_somewhere, on_this_branch)`."""
        here = path in on_disk or (ROOT / path).exists()
        if not master:
            return here, here
        return here, any(path == p or path.startswith(p) for p in MASTER_PREFIXES)

    for name in prose(files):
        text = (ROOT / name).read_text(encoding="utf-8", errors="replace")

        for target, number in SECTION.findall(text):
            if (target, number) in TABLE_ROW_CITATIONS:
                continue
            report.checked["section"] += 1
            if target not in heading_cache:
                p = ROOT / target
                heading_cache[target] = headings(p) if p.exists() else set()
            if number not in heading_cache[target]:
                report.breaks.append(f"{name}: `{target}` {number} -- no such section")

        for path in set(REPO_PATH.findall(text)):
            if path.startswith(UPSTREAM_DOC_PREFIXES) or path in FPRIME_CHECKOUT_PATHS:
                continue
            if path in PLANNED:
                continue
            report.checked["path"] += 1
            exists, here = resolves(path)
            if not exists:
                report.breaks.append(f"{name}: `{path}` -- no such file")
            elif not here:
                report.dev_resolving.append(f"{name}: `{path}`")

        for cited, lo, hi in set(LINE_CITE.findall(text)):
            base = Path(cited).name
            if base in VENDORED_FILENAMES and "/" not in cited:
                continue                       # convention 3
            report.checked["line"] += 1
            candidates = [f for f in files if f == cited or Path(f).name == base]
            if len(candidates) != 1:
                continue                       # ambiguous or absent: not this check's call
            target = ROOT / candidates[0]
            last = int(hi or lo)
            length = len(target.read_text(encoding="utf-8", errors="replace").splitlines())
            if last > length:
                report.breaks.append(
                    f"{name}: `{cited}:{lo}` -- past {candidates[0]}'s {length} lines")

        # Markdown links, in Markdown only. A Python slice like `x[args](y)` is
        # not a link, and treating it as one produced five false breaks.
        for link in (set(MD_LINK.findall(text)) if name.endswith(".md") else set()):
            report.checked["link"] += 1
            target = (ROOT / name).parent / link
            if not target.exists():
                report.breaks.append(f"{name}: [{link}] -- no such target")

        for module, test in set(NODE_ID.findall(text)):
            report.checked["node"] += 1
            p = ROOT / module
            if not p.exists():
                report.breaks.append(f"{name}: `{module}::{test}` -- no such module")
            elif f"def {test}" not in p.read_text(encoding="utf-8", errors="replace"):
                report.breaks.append(f"{name}: `{module}::{test}` -- no such test")

    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master", action="store_true",
                    help="D67's curated branch: off-branch paths resolve on dev")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    report = check(master=a.master)
    counts = ", ".join(f"{v} {k}" for k, v in report.checked.items())
    mode = "master" if a.master else "dev"
    print(f"check_references [{mode}]: {counts}")
    if a.master:
        print(f"  {len(report.dev_resolving)} citation(s) resolve on dev, not on master "
              "-- expected, and D67's convention covers them")
        if a.verbose:
            for line in report.dev_resolving[:20]:
                print(f"    dev  {line}")
    for line in report.breaks:
        print(f"  BREAK  {line}")
    if not report.breaks:
        print("  every citation resolves")
    return 1 if report.breaks else 0


if __name__ == "__main__":
    sys.exit(main())
