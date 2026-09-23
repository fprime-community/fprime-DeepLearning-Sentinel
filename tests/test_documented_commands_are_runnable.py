"""(!) A documented command a stranger cannot run is a broken front door.

`master:README.md`'s Quick start told a first-time reader to run

    python -m venv .venv && .venv/bin/pip install -r requirements-toolkit.txt

and on macOS -- and on most current Linux distributions -- there is no `python` on
PATH, only `python3`. A stranger following the README **failed on the second block, on
their first command**, with `command not found: python`. `dev:README.md` had it right
(`python3.14 -m venv`) the whole time, which is how it survived: the two READMEs are
curated separately (D67, D81) and nothing compared their commands.

It was found by doing what this file now guards: cloning `master` and running every
Quick start block verbatim, 2026-09-23.

**What this checks.** Inside fenced `bash` blocks only -- prose that mentions Python is
not a command -- no line may invoke a bare `python`. `python3`, `python3.14` and
`.venv/bin/python` are all fine, because all three exist by the time the block runs.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Documents that carry commands a reader is expected to run, on either branch.
#: `docs/HARNESS.md` is deliberately absent: its listings describe a CLI's shape
#: rather than a sequence to run, and it is `dev`-only.
DEV_DOCS = ("README.md", "docs/datasets/REPRODUCING.md")
MASTER_DOCS = ("README.md", "docs/datasets/REPRODUCING.md")

#: A bare `python` used as a command: at the start of a line, or after `&&`, `;`,
#: `|` or an environment-variable prefix. `python3`, `python3.14` and any path ending
#: `/python` are all excluded by requiring a non-word, non-`/`, non-digit boundary.
BARE_PYTHON = re.compile(r"(?:^|&&\s*|;\s*|\|\s*|^(?:\w+=\S+\s+)+)python(?![\w./])",
                         re.MULTILINE)


def _bash_blocks(text: str) -> list[str]:
    return re.findall(r"^```bash\n(.*?)^```", text, re.MULTILINE | re.DOTALL)


def _offenders(text: str) -> list[str]:
    out = []
    for block in _bash_blocks(text):
        for line in block.splitlines():
            if BARE_PYTHON.search(line):
                out.append(line.strip())
    return out


@pytest.mark.parametrize("name", DEV_DOCS)
def test_no_dev_document_tells_a_reader_to_run_bare_python(name: str) -> None:
    bad = _offenders((ROOT / name).read_text(encoding="utf-8"))
    assert not bad, (
        f"{name} tells a reader to run `python`, which does not exist on macOS or on "
        f"most current Linux distributions: {bad}. Use `python3` to create the venv "
        "and `.venv/bin/python` afterwards.")


@pytest.mark.parametrize("name", MASTER_DOCS)
def test_no_master_document_tells_a_reader_to_run_bare_python(name: str) -> None:
    """(!) `master` is the branch a stranger actually clones.

    Skips rather than fails where there is no `master` ref, for the same reason the
    other master guards do: a fresh clone of one branch is a legitimate state.
    """
    probe = subprocess.run(["git", "rev-parse", "--verify", "master"],
                           cwd=ROOT, capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("no `master` ref in this clone")
    shown = subprocess.run(["git", "show", f"master:{name}"],
                           cwd=ROOT, capture_output=True, text=True)
    assert shown.returncode == 0, f"master:{name} is not readable"
    bad = _offenders(shown.stdout)
    assert not bad, (
        f"master:{name} tells a reader to run `python`: {bad}. This is the document a "
        "first-time reader follows, and the two READMEs are curated separately, so "
        "`dev` being correct does not make `master` correct.")


def test_the_check_would_catch_a_bare_python() -> None:
    """The positive control. A pattern that matches nothing passes everywhere."""
    assert _offenders("```bash\npython -m venv .venv\n```")
    assert _offenders("```bash\nPYTHONPATH=src python -m sentinel_toolkit selftest\n```")
    assert _offenders("```bash\nmake -C flight test && python foo.py\n```")
    # And the spellings that are fine must NOT be flagged.
    assert not _offenders("```bash\npython3 -m venv .venv\n```")
    assert not _offenders("```bash\npython3.14 -m venv .venv\n```")
    assert not _offenders("```bash\nPYTHONPATH=src .venv/bin/python -m pytest -q\n```")
