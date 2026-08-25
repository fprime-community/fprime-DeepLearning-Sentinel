"""Shared fixtures. Every test runs against generated data -- zero R2 operations.

The fixture is a fake *bucket*, not a fake bundle: real parquet bytes under real
manifest keys with real SHA-256 digests. So these tests exercise the genuine
catalog, reader, checksum verification, label and grid code paths, and the only
thing that differs from a live run is the transport.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel_eval import bundle as bundle_mod          # noqa: E402
from sentinel_eval import read, splits, synthetic, tasks  # noqa: E402
from sentinel_eval.catalog import Catalog                # noqa: E402
from sentinel_eval.labels import LabelSet                # noqa: E402

STEPS = 8_000


@pytest.fixture(scope="session")
def bucket():
    return synthetic.build(seed=0, n=STEPS)


@pytest.fixture(scope="session")
def catalog(bucket):
    return Catalog.load(bucket)


@pytest.fixture(scope="session")
def labels(bucket, catalog):
    return LabelSet.from_table(read.read_annotation(bucket, catalog, "labels"))


@pytest.fixture(scope="session")
def channel_ids(catalog):
    return tasks.get("synthetic").selection.resolve(catalog)


@pytest.fixture(scope="session")
def loaded(bucket, catalog, labels, channel_ids):
    return bundle_mod.load(bucket, catalog, labels, mission="missionX",
                           channel_ids=channel_ids)


@pytest.fixture(scope="session")
def split(loaded):
    return splits.forward_chaining(len(loaded.grid), seed_fraction=0.25, folds=3)


@pytest.fixture
def project_root():
    return ROOT


@pytest.fixture(autouse=True, scope="session")
def _weights_go_nowhere_real(tmp_path_factory):
    """No test may write into the real weight store.

    Weights now persist under `runs/_weights` (docs/HARNESS.md, Rule 1 stated
    precisely), keyed by content so a stale checkpoint cannot be found. That is
    safe but it is not tidy: a test suite run would otherwise leave fits from
    fixture-scale models in the same directory a real run reads, growing without
    bound and confusing anyone who looks.

    Session-scoped and autouse, so it holds for every test whether or not the
    test knows the store exists.
    """
    from sentinel_models import detectors

    original = detectors.WEIGHT_STORE
    detectors.WEIGHT_STORE = tmp_path_factory.mktemp("weights")
    yield
    detectors.WEIGHT_STORE = original
