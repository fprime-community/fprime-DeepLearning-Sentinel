"""Rule 1: the Mac is a pipe, not a store.

No telemetry, no parquet, no cached datasets on local disk -- ever. A run may
write its own scorecard under `runs/`, which is gitignored, and nothing else.
"""
from __future__ import annotations

from pathlib import Path

from sentinel_eval import bundle as bundle_mod
from sentinel_eval import harness, splits, tasks
from sentinel_models import baselines

IGNORED = {"runs", "scratch", "data", ".git", ".venv", "__pycache__", ".pytest_cache"}


def _snapshot(root: Path) -> set[Path]:
    return {p for p in root.rglob("*")
            if p.is_file() and not IGNORED & set(p.parts)}


def test_a_full_scored_run_writes_nothing_to_the_tree(project_root, bucket, catalog,
                                                      labels, channel_ids):
    before = _snapshot(project_root)
    loaded = bundle_mod.load(bucket, catalog, labels, mission="missionX",
                             channel_ids=channel_ids)
    split = splits.forward_chaining(len(loaded.grid), seed_fraction=0.25, folds=3)
    harness.evaluate(loaded, split, [baselines.MovingAverage(60)],
                     tasks.get("synthetic"), sweep=False)
    assert _snapshot(project_root) == before


def test_the_ingest_scratch_directory_is_absent(project_root):
    assert not (project_root / "scratch").exists()


def test_no_dataset_shaped_file_is_committed_or_staged(project_root):
    strays = [p for p in project_root.rglob("*")
              if p.is_file() and not IGNORED & set(p.parts)
              and p.suffix in {".parquet", ".pkl", ".zip", ".npy", ".npz", ".h5", ".hdf5"}]
    assert strays == []


def test_the_repository_holds_code_and_docs_only(project_root):
    """784 KB was the agreed size. A dataset would show up as megabytes."""
    total = sum(p.stat().st_size for p in project_root.rglob("*")
                if p.is_file() and not IGNORED & set(p.parts))
    assert total < 4 * 1024 * 1024, f"working tree is {total / 1048576:.1f} MiB"


def test_gitignore_covers_the_artifacts_this_harness_produces(project_root):
    ignored = (project_root / ".gitignore").read_text()
    for pattern in ("runs/", "data/", "scratch/", "*.parquet"):
        assert pattern in ignored
