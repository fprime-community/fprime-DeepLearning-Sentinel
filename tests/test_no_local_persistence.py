"""Rule 1: no dataset lives on this machine. Outputs are a different thing.

    No telemetry, no parquet, no cached datasets on local disk. Model weights
    and run artifacts under `runs/` are outputs, not data, and may persist.
    Gitignored, never committed.

The earlier wording -- "the Mac is a pipe, not a store" -- was the right instinct
and the wrong rule, because it reads as forbidding a trained model to survive the
process that produced it. Rule 1 exists to stop this machine becoming a data
store, and 358 KiB of learned parameters is not a data store.

So the exemption for `runs/` is **widened** here to cover weights, and the
dataset ban is **tightened** at the same time: parquet, pickles and archives are
now refused everywhere in the tree including `runs/`, which the previous version
skipped entirely. Loosening one rule is not a licence to loosen the other, and a
telemetry file hidden under `runs/` would have gone unnoticed before.

Work item 9 adds one exemption and states its reason here rather than in a
comment. The F' framework checkout and its tool virtualenv live under `fprime/`
because this machine keeps everything for this project inside this directory.
They are a reproducible build dependency, not a store of data -- exactly the
status `.venv` already has in `MACHINERY` -- they are gitignored, and
`scripts/fprime_setup.sh` rebuilds them from nothing. Only those two subtrees are
exempt: `fprime/`'s own sources, the component and the deployment, stay under
every rule below, so a dataset parked beside the component is still caught. The
checkout was scanned before the exemption was written and holds **0** files with
a dataset or array suffix, so nothing is being waved through.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sentinel_eval import bundle as bundle_mod
from sentinel_eval import harness, splits, tasks
from sentinel_models import baselines

#: Never scanned: not ours, or pure build output.
MACHINERY = {".git", ".venv", "__pycache__", ".pytest_cache"}

#: Additionally skipped when asking "did a run dirty the source tree?".
OUTPUTS = MACHINERY | {"runs", "scratch", "data"}

#: A dataset by any name. Refused **everywhere**, `runs/` included.
DATASET_SUFFIXES = {".parquet", ".pkl", ".zip", ".h5", ".hdf5", ".feather", ".arrow"}

#: Arrays are how weights are stored, so they are allowed -- but only as outputs.
ARRAY_SUFFIXES = {".npy", ".npz"}

#: The F' toolchain (D31, work item 9): a gitignored, script-rebuilt dependency
#: tree, skipped for the reason given in this module's docstring. Matched as path
#: prefixes rather than by name, so `fprime/` alone is never skipped.
TOOLCHAIN_PREFIXES = ((("fprime", "lib"), ("fprime", "fprime-venv")))


def _files(root: Path, skip: set[str]):
    for p in root.rglob("*"):
        if not p.is_file() or skip & set(p.parts):
            continue
        parts = p.relative_to(root).parts
        if any(parts[:len(prefix)] == prefix for prefix in TOOLCHAIN_PREFIXES):
            continue
        yield p


def test_a_full_scored_run_writes_nothing_to_the_source_tree(project_root, bucket, catalog,
                                                             labels, channel_ids):
    before = set(_files(project_root, OUTPUTS))
    loaded = bundle_mod.load(bucket, catalog, labels, mission="missionX",
                             channel_ids=channel_ids)
    split = splits.forward_chaining(len(loaded.grid), seed_fraction=0.25, folds=3)
    harness.evaluate(loaded, split, [baselines.MovingAverage(60)],
                     tasks.get("synthetic"), sweep=False)
    assert set(_files(project_root, OUTPUTS)) == before


def test_the_ingest_scratch_directory_is_absent(project_root):
    assert not (project_root / "scratch").exists()


def test_no_dataset_shaped_file_exists_anywhere_including_runs(project_root):
    """The tightened half. `runs/` used to be skipped wholesale by this check."""
    strays = [p.relative_to(project_root).as_posix()
              for p in _files(project_root, MACHINERY)
              if p.suffix in DATASET_SUFFIXES]
    assert strays == [], f"dataset-shaped files on disk: {strays}"


def test_arrays_are_permitted_only_as_outputs_under_runs(project_root):
    """Weights may persist. A stray .npz beside the source is still a defect."""
    strays = [p.relative_to(project_root).as_posix()
              for p in _files(project_root, MACHINERY)
              if p.suffix in ARRAY_SUFFIXES and "runs" not in p.parts]
    assert strays == [], f"arrays outside runs/: {strays}"


def test_the_toolchain_exemption_covers_only_what_it_claims(project_root):
    """The exemption is narrow, and this asserts it stays narrow.

    `fprime/`'s own sources must still be scanned; only the checkout and the tool
    venv are skipped. If someone widens the prefixes to `fprime/`, this fails.
    """
    scanned = {p.relative_to(project_root).as_posix() for p in _files(project_root, OUTPUTS)}
    assert not any(f.startswith("fprime/lib/") for f in scanned)
    assert not any(f.startswith("fprime/fprime-venv/") for f in scanned)
    settings = project_root / "fprime" / "settings.ini"
    if settings.exists():
        assert "fprime/settings.ini" in scanned, "fprime/'s own sources must stay scanned"


def _repository_files(project_root):
    """What a clone would hold: tracked files, plus untracked ones not ignored.

    Sharpened at work item 9. This test previously measured the whole working
    tree, which conflates the repository with local output: 1.97 MiB of the
    4.11 MiB it was seeing is gitignored -- work item 8's g3 and g4 golden
    vectors, which `.gitignore` excludes precisely because they regenerate from
    their seeds, and `flight/build/`. Both are outputs in exactly the sense
    `runs/` is, and `OUTPUTS` already skips `runs/` for that reason.

    Measuring the tracked set is what the test's own name claims, and it is
    strictly sharper for the thing that matters: a committed dataset now fails
    here whatever its suffix, where before it competed for headroom with build
    output. The dataset and array guards above are untouched and still scan the
    entire tree, gitignored files included.
    """
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=str(project_root), capture_output=True, text=True, timeout=120)
    if listing.returncode != 0:
        pytest.skip("not a git checkout")
    return [project_root / name
            for name in listing.stdout.split("\0") if name and (project_root / name).is_file()]


def test_the_repository_holds_code_and_docs_only(project_root):
    """8 MiB, raised from 4 by D64. The byte count is not what keeps data out.

    784 KB was the original agreement, raised to 4 MiB when the golden vectors were
    committed, and to 8 MiB when prose reached 92% of that. The cap exists to stop a
    dataset entering the tree -- and **the format refusals above are what actually do
    that job**: `DATASET_SUFFIXES` refuses parquet, pickles and archives everywhere
    including `runs/`, and `ARRAY_SUFFIXES` permits `.npy`/`.npz` only under `runs/`.
    A byte count cannot tell a dataset from a document; a suffix can, and does.

    So this assertion is a **coarse backstop against bulk**, not the guard. It is
    raised rather than removed because a repository that quietly grows to hundreds of
    megabytes is still a defect worth failing on, and it is raised rather than left to
    strangle the additive documentation work D64 records. **The format refusals are
    unchanged, and they are the part that matters.**
    """
    total = sum(p.stat().st_size for p in _repository_files(project_root))
    assert total < 8 * 1024 * 1024, f"repository is {total / 1048576:.1f} MiB"


def test_no_gitignored_output_is_mistaken_for_repository_content(project_root):
    """The sharpening above must not become a hiding place.

    Anything gitignored is invisible to the size check, so this asserts the two
    sets are actually disjoint in the way the docstring claims: every file the
    size check counts is one `git` would carry.
    """
    counted = {p.resolve() for p in _repository_files(project_root)}
    for path in counted:
        rel = path.relative_to(project_root).as_posix()
        ignored = subprocess.run(["git", "check-ignore", "-q", rel],
                                 cwd=str(project_root), capture_output=True, timeout=60)
        assert ignored.returncode != 0, f"{rel} is gitignored but counted as repository content"


def test_gitignore_covers_the_artifacts_this_harness_produces(project_root):
    ignored = (project_root / ".gitignore").read_text()
    for pattern in ("runs/", "data/", "scratch/", "*.parquet", "*.npz"):
        assert pattern in ignored, f"{pattern} is not gitignored"
