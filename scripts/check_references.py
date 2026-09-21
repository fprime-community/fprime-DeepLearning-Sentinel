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

#: D69's list, superseding D67's. Kept here because the guard is what enforces it,
#: and checked against the real branch by
#: `tests/test_master_documents_are_current.py` -- this list decides on-branch
#: membership and nothing compared it to `master` until 2026-09-11.
#:
#: D69 took the laboratory record off the branch: `Objective.md`,
#: `docs/DECISIONS.md`, `docs/RESULTS.md` and `docs/NARRATIVE.md` are kept whole on
#: `dev` and are no longer here, so a citation into them is now dev-resolving.
#: `docs/DESIGN.md` and `docs/EVIDENCE.md` exist ONLY on `master`.
MASTER_PREFIXES = (
    "flight/", "fprime/Sentinel/Monitor/", "fprime/SentinelRef/",
    # docs/MODELS.md 65: the retrainer's own deployment, which is where the OCaml
    # runtime lives. D70 consequence 2 makes the separate process a requirement.
    "fprime/SentinelRetrain/",
    "fprime/CMakeLists.txt", "fprime/library.cmake", "fprime/settings.ini",
    # docs/MODELS.md 75: the library's landing page, the one a reader who knows
    # fprime-sensors-reference arrives at. Named individually like the three
    # above, because `fprime/` is not a blanket prefix here.
    "fprime/README.md",
    "README.md", "docs/STATUS.md", "docs/datasets/", "docs/MODEL_FILE.md",
    "docs/FPRIME.md", "docs/PI_ENVELOPE.md", "LICENSE",
    # D69, 2026-09-11: the customer documents. Neither is on `dev`.
    "docs/DESIGN.md", "docs/EVIDENCE.md",
    # (!) D80 carries the ground toolkit's import closure and six build scripts
    # onto `master`. `oxcaml/`, `src/sentinel_toolkit/` and `src/sentinel_export/`
    # move WHOLE, so a prefix is right for them. `src/sentinel_models/`,
    # `src/sentinel_eval/` and `scripts/` move in PART -- the research apparatus
    # stays on `dev` -- so those are named file by file. A blanket `src/` here
    # would report a citation of the scoring harness as resolving on `master`,
    # which is the opposite of what this mode exists to tell a reader.
    "oxcaml/", "src/sentinel_toolkit/", "src/sentinel_export/",
    "requirements-toolkit.txt",
    "src/sentinel_models/__init__.py", "src/sentinel_models/detectors.py",
    "src/sentinel_models/lstm.py", "src/sentinel_models/oscfar.py",
    "src/sentinel_models/reference.py", "src/sentinel_models/telemanom.py",
    "src/sentinel_models/whiten.py", "src/sentinel_models/windows.py",
    "src/sentinel_eval/__init__.py", "src/sentinel_eval/bundle.py",
    "src/sentinel_eval/catalog.py", "src/sentinel_eval/commands.py",
    "src/sentinel_eval/detector.py", "src/sentinel_eval/errors.py",
    "src/sentinel_eval/grid.py", "src/sentinel_eval/labels.py",
    "src/sentinel_eval/normalisation.py", "src/sentinel_eval/read.py",
    "src/sentinel_eval/synthetic.py", "src/sentinel_eval/tasks.py",
    "scripts/oxcaml_setup.sh", "scripts/oxcaml_e1.sh", "scripts/oxcaml_s61.sh",
    "scripts/oxcaml_s72.sh", "scripts/fprime_setup.sh", "scripts/fprime_ref_patch.sh",
)

#: Of `MASTER_PREFIXES`, the entries that do not exist on `dev` at all. `LICENSE`
#: is unselected; the other two are `master`'s own documents.
MASTER_ONLY_PREFIXES = ("LICENSE", "docs/DESIGN.md", "docs/EVIDENCE.md")

#: Documents whose citations are checked. Source files cite too, and are included.
#:
#: (!) UNTRACKED-BUT-NOT-IGNORED FILES ARE INCLUDED, AND THE REASON IS A REAL MISS.
#: This used to be `git ls-files` alone, which meant a brand-new document was not
#: scanned at all until the commit that added it -- so the checker reported "every
#: citation resolves" on a working tree containing two broken ones, and both were
#: found only afterwards. That is the worst possible moment for a guard to be quiet:
#: the commit adding a document is exactly the one a reader would trust. `--others
#: --exclude-standard` adds what a human would call "the new files" and nothing
#: `.gitignore` excludes, so `runs/` and the build trees stay out.
def tracked() -> list[str]:
    def ls(*args: str) -> list[str]:
        return subprocess.run(["git", "ls-files", *args], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.split()
    return ls() + ls("--others", "--exclude-standard")


#: Files that quote citation forms in order to TALK about them rather than to
#: cite: this module's own docstring shows `tests/test_x.py::test_y` and
#: `docs/datasets/NAME.md` as examples, and its test quotes the one dangling
#: citation it found (`docs/MODELS.md` 26.31) as the thing it found. Checking them
#: makes the guard report itself. The same exemption, for the same reason,
#: `scripts/telemanom_citations.py` takes and `tests/test_documents_are_current.py`
#: takes when it skips itself for quoting every pattern it bans.
SELF_REFERENTIAL = ("scripts/check_references.py", "tests/test_references_resolve.py")


def prose(files: list[str]) -> list[str]:
    return [f for f in files
            if f not in SELF_REFERENTIAL
            and (f.endswith(".md")
                 or (f.endswith(".py") and f.startswith(("src/", "scripts/", "tests/"))))]


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

#: (!) THE CHECKOUT ITSELF, WHICH IS GITIGNORED AND MAY SIMPLY NOT BE THERE.
#: The build trees were deleted on 2026-09-21 to reclaim 3.6 GB, and this checker
#: then FAILED on one citation -- `docs/PHASE5.md`'s pointer into F's own skills
#: documentation -- for a reason that has nothing to do with whether the citation
#: is right. It is right, and it resolves the moment the checkout is back.
#:
#: Every other guard that needs a build SKIPS with a message naming the script
#: that would satisfy it. This one had no notion that `fprime/lib/` is rebuildable
#: rather than missing content, so it went red instead, and **a gate that fails
#: for a reason unrelated to what it guards trains people to ignore it**.
#:
#: The rule is deliberately narrow, and both halves matter:
#:   - only paths under this prefix, and
#:   - only while the directory is absent.
#: With the checkout present these citations are checked exactly as before, so a
#: wrong one still BREAKs. The cost is stated rather than hidden: while the
#: checkout is absent, a citation into it that is WRONG is skipped too. That is
#: the same trade every build-dependent skip in this repository already makes.
FPRIME_CHECKOUT_DIR = "fprime/lib/"


def checkout_absent() -> bool:
    """True when the gitignored framework checkout is not on disk."""
    return not (ROOT / FPRIME_CHECKOUT_DIR).is_dir()


def _line_of(text: str, needle: str) -> int:
    """1-based line of the first occurrence, so a skip can be acted on."""
    for n, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return n
    return 0

#: (!) Cited today, written by a later tranche of `docs/REORG_PLAN.md`. The debt
#: is listed rather than hidden, and it shrinks as the tranches land -- a stale
#: entry here fails `tests/test_references_resolve.py`, which asserts that nothing
#: listed exists yet.
#:
#: **Emptied 2026-09-11 by tranche 4**, which wrote all five: the four dataset
#: documents and the reserved Pi envelope. They are checked like everything else
#: now, which is the point of the list being empty rather than absent.
PLANNED: dict[str, str] = {}

#: Bare filenames that resolve into the vendored package rather than into this
#: repository, at the sites that cite them. Convention 3 of the three above.
VENDORED_FILENAMES = {"detector.py", "channel.py", "errors.py", "modeling.py",
                      "helpers.py"}

SECTION = re.compile(
    r"`(docs/(?!DECISIONS\.md)[A-Za-z_]+\.md|Objective\.md|CHANGELOG\.md|README\.md)`\s+"
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
        #: Citations into the absent framework checkout. Reported loudly, never
        #: silently dropped, and they do not change the exit code.
        self.skipped: list[str] = []


def check(master: bool = False) -> Report:
    files = tracked()
    on_disk = set(files)
    report = Report()
    heading_cache: dict[str, set[str]] = {}
    #: Read once: the answer cannot change inside a single run, and both modes
    #: apply the same rule.
    absent_checkout = checkout_absent()

    def resolves(path: str) -> tuple[bool, bool]:
        """`(exists_somewhere, on_this_branch)`.

        (!) A path can exist and not be on `dev`. D69 gave `master` two documents
        of its own, `docs/DESIGN.md` and `docs/EVIDENCE.md`, and a `dev` document
        that cites one is citing something real -- it is simply off this branch,
        which is the same relationship every `runs/` artifact already has.
        """
        here = path in on_disk or (ROOT / path).exists()
        elsewhere = any(path == p or path.startswith(p) for p in MASTER_ONLY_PREFIXES)
        if not master:
            return here or elsewhere, here
        return here or elsewhere, any(path == p or path.startswith(p)
                                      for p in MASTER_PREFIXES)

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
            if path.startswith(FPRIME_CHECKOUT_DIR) and absent_checkout:
                report.skipped.append(f"{name}:{_line_of(text, path)}: `{path}`")
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
    if report.skipped:
        print(f"  {len(report.skipped)} citation(s) into `{FPRIME_CHECKOUT_DIR}` NOT "
              f"CHECKED -- the framework checkout is absent. It is gitignored and "
              f"rebuildable: run `scripts/fprime_setup.sh` and they are checked again.")
        for line in report.skipped:
            print(f"    SKIP   {line}")
    for line in report.breaks:
        print(f"  BREAK  {line}")
    if not report.breaks:
        print("  every citation resolves")
    return 1 if report.breaks else 0


if __name__ == "__main__":
    sys.exit(main())
