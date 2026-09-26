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


def test_the_retrainer_block_regenerates_before_it_builds() -> None:
    """(!) A build that exits 0 having produced nothing, documented as the way to do it.

    Until D84, `Retrainer/CMakeLists.txt` and `SentinelRetrain/CMakeLists.txt` returned
    early when `${OX_OBJ}` was absent, so a build cache generated BEFORE `oxcaml_s61.sh`
    had run registered neither the module nor the deployment. (Since D84 the option
    decides, and ON without the object is an error; the regenerate is still required,
    because the option is set AT generate time.) The README's retrainer block runs
    `oxcaml_s61.sh` and then built straight away against the cache the F' block made --
    so `fprime-util build -p ./SentinelRetrain` printed `ninja: no work to do`, **exited
    0**, and left no binary. Found 2026-09-23 by running the block verbatim and then
    looking for the binary, which is the only thing that would have caught it.

    `docs/MODELS.md` 72.5 recorded the same shape once already: "the build exits 0
    having built nothing". This is that, in the instructions.
    """
    probe = subprocess.run(["git", "rev-parse", "--verify", "master"],
                           cwd=ROOT, capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("no `master` ref in this clone")
    text = subprocess.run(["git", "show", "master:README.md"],
                          cwd=ROOT, capture_output=True, text=True).stdout
    blocks = [b for b in _bash_blocks(text) if "SentinelRetrain" in b]
    assert blocks, "master:README.md no longer shows how to build SentinelRetrain"
    for block in blocks:
        assert "fprime-util generate" in block, (
            "the retrainer block builds SentinelRetrain without regenerating the build "
            "cache first. The deployment is registered only if the OxCaml object exists "
            "at generate time, so this silently produces no binary and still exits 0:\n"
            f"{block}")



# -- D84: the opt-in retrainer's documented path ----------------------------------------

POWERSIM_FPP = ROOT / "fprime" / "SentinelRef" / "PowerSim" / "PowerSim.fpp"
LIBRARY = ROOT / "fprime" / "library.cmake"


def _master_readme() -> str:
    probe = subprocess.run(["git", "rev-parse", "--verify", "master"],
                           cwd=ROOT, capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("no `master` ref in this clone")
    return subprocess.run(["git", "show", "master:README.md"],
                          cwd=ROOT, capture_output=True, text=True).stdout


def _retrainer_block_faults(block: str, channels: int) -> list[str]:
    """What is wrong with one bash block that builds SentinelRetrain. Empty is right."""
    faults = []
    if "fprime-util generate" not in block:
        faults.append("builds without regenerating")
    if "-DSENTINEL_WITH_RETRAINER=ON" not in block:
        faults.append("never switches SENTINEL_WITH_RETRAINER on, so the deployment "
                      "skips itself and the build exits 0 having built nothing")
    shape = re.search(r"oxcaml_shape\.sh --channels (\d+) --predictions (\d+)", block)
    if not shape:
        faults.append("never generates the shape with scripts/oxcaml_shape.sh")
    elif int(shape.group(1)) != channels:
        faults.append(f"generates {shape.group(1)} channels; SentinelRef feeds {channels}")
    asked = re.search(r"-DSENTINEL_RETRAINER_CHANNELS=(\d+)", block)
    if not asked or int(asked.group(1)) != channels:
        faults.append("does not ask F' for the channel count it generated")
    return faults


def _powersim_channels() -> int:
    m = re.search(r"constant POWERSIM_CHANNELS = (\d+)", POWERSIM_FPP.read_text(encoding="utf-8"))
    assert m, "PowerSim.fpp no longer declares POWERSIM_CHANNELS"
    return int(m.group(1))


def test_the_retrainer_block_switches_it_on_at_the_deployments_shape() -> None:
    """D84, and D83.3's lesson one step on: a block that exits 0 and builds nothing."""
    blocks = [b for b in _bash_blocks(_master_readme()) if "SentinelRetrain" in b]
    assert blocks, "master:README.md no longer shows how to build SentinelRetrain"
    channels = _powersim_channels()
    for block in blocks:
        faults = _retrainer_block_faults(block, channels)
        assert not faults, f"master:README.md's retrainer block {faults}:\n{block}"


def test_the_retrainer_block_check_can_fail() -> None:
    """The positive control, one fault at a time."""
    good = ("bash scripts/oxcaml_shape.sh --channels 8 --predictions 10\n"
            "fprime-util generate -f -DSENTINEL_WITH_RETRAINER=ON "
            "-DSENTINEL_RETRAINER_CHANNELS=8\nfprime-util build -p ./SentinelRetrain\n")
    assert not _retrainer_block_faults(good, 8)
    assert _retrainer_block_faults(good.replace("-DSENTINEL_WITH_RETRAINER=ON ", ""), 8)
    assert _retrainer_block_faults(good.replace("--channels 8", "--channels 16"), 8)
    assert _retrainer_block_faults(good.replace("fprime-util generate -f", "true"), 8)
    assert _retrainer_block_faults(good.replace("bash scripts/oxcaml_shape.sh", "true"), 8)


def test_every_option_the_readme_names_exists() -> None:
    """A documented `SENTINEL_*` setting that the library does not read is a typo that
    configures silently -- CMake accepts any -D and ignores what nothing reads."""
    named = set(re.findall(r"\bSENTINEL_[A-Z_]+\b", _master_readme()))
    named = {n for n in named if n.startswith(("SENTINEL_WITH_", "SENTINEL_RETRAINER_"))}
    library = LIBRARY.read_text(encoding="utf-8")
    missing = sorted(n for n in named if n not in library)
    assert not missing, f"master:README.md names {missing}, which fprime/library.cmake never reads"
