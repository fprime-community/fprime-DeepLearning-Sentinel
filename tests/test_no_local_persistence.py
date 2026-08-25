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
"""
from __future__ import annotations

from pathlib import Path

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


def _files(root: Path, skip: set[str]):
    return (p for p in root.rglob("*") if p.is_file() and not skip & set(p.parts))


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


def test_the_repository_holds_code_and_docs_only(project_root):
    """784 KB was the agreed size. A dataset would show up as megabytes."""
    total = sum(p.stat().st_size for p in _files(project_root, OUTPUTS))
    assert total < 4 * 1024 * 1024, f"working tree is {total / 1048576:.1f} MiB"


def test_gitignore_covers_the_artifacts_this_harness_produces(project_root):
    ignored = (project_root / ".gitignore").read_text()
    for pattern in ("runs/", "data/", "scratch/", "*.parquet", "*.npz"):
        assert pattern in ignored, f"{pattern} is not gitignored"
