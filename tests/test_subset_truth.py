"""A subset's ground truth must agree with the parent's about what was observed.

`bundle.load` folds `~valid.all(axis=1)` into `truth.unscorable`, so *usable*
implies *observed* and `splits.train_mask` never hands a detector a gap.
`Bundle.subset` rebuilt its truth from `labels.truth` alone and never read
`self.valid`, so that fold-in was lost on every subset -- and `m1-ss5`, the
demoted set reported beside the primary on every result, is scored as a subset.

Two consequences, one loud and one quiet. The loud one killed a 75-minute run:
a forecaster sampled a training sequence containing NaN and refused it. The
quiet one is worse: `unscorable` also gates the metrics, so ~1,622 unobserved
timesteps were counted as scorable nominal time -- the free specificity
docs/HARNESS.md section 1 explicitly refuses for gaps and invalid segments.

It survived three work items because the trivial baselines are NaN-tolerant by
construction (`nan_to_num`, `nanstd`) and nothing had ever asked the mask to be
right.
"""
from __future__ import annotations

import numpy as np

from sentinel_eval import bundle as bundle_mod
from sentinel_eval import splits


def _fresh(bucket, catalog, labels, channel_ids, *, gaps_on=(), rows=400, seed=0):
    """A bundle of its own, with scattered unobserved steps on chosen channels.

    Built per test rather than mutating the session fixture: holes punched into a
    shared bundle leak into every test that runs after, and the truth of an
    already-loaded parent is not recomputed when its validity changes, so a
    mutated fixture is doubly misleading.

    The pattern reproduces ESA-ADB's condition -- the staleness guard in
    `sentinel_eval.grid` leaves 1,622 scattered unobserved steps on the six
    group-8 channels of the real primary set.
    """
    loaded = bundle_mod.load(bucket, catalog, labels, mission="missionX",
                             channel_ids=channel_ids)
    holes = np.random.default_rng(seed).choice(
        loaded.values.shape[0], size=rows, replace=False)
    for column in gaps_on:
        loaded.values[holes, column] = np.nan
        loaded.valid[holes, column] = False
    return loaded, np.sort(holes)


def test_a_subset_never_marks_an_unobserved_step_usable(bucket, catalog, labels,
                                                        channel_ids):
    """The shape that killed a run: a training sequence drawn from a gap."""
    loaded, _ = _fresh(bucket, catalog, labels, channel_ids, gaps_on=(0,))
    child = loaded.subset(channel_ids[:3], labels)

    split = splits.forward_chaining(len(child.grid), seed_fraction=0.25, folds=3)
    for fold in split.folds:
        lo, hi = fold.train
        usable = splits.train_mask(fold, child.truth)[lo:hi]
        observed = np.isfinite(child.values[lo:hi]).all(axis=1)
        leaked = int((usable & ~observed).sum())
        assert leaked == 0, f"fold {fold.index} marks {leaked} unobserved steps usable"


def test_the_subset_folds_in_exactly_what_load_would_have(bucket, catalog, labels,
                                                          channel_ids):
    """The invariant, stated directly: subset must match load's own arithmetic."""
    loaded, _ = _fresh(bucket, catalog, labels, channel_ids, gaps_on=(0, 2))
    kept = list(channel_ids[:3])
    child = loaded.subset(kept, labels)

    plain = labels.truth(loaded.grid, "missionX", kept)
    expected = (plain.unscorable | ~child.valid.all(axis=1)) & ~plain.anomaly
    assert np.array_equal(child.truth.unscorable, expected)


def test_a_gap_on_a_channel_the_subset_dropped_stays_scorable(bucket, catalog, labels,
                                                              channel_ids):
    """The fold-in is recomputed, not inherited, and that distinction matters.

    Which timesteps are unobserved depends on which channels were selected. A
    subset that copied the parent's mask would inherit gaps on channels it does
    not carry, and would throw away scorable time for no reason.
    """
    dropped = 4
    loaded, holes = _fresh(bucket, catalog, labels, channel_ids,
                           gaps_on=(dropped,), seed=7)
    assert channel_ids[dropped] not in channel_ids[:3]

    child = loaded.subset(channel_ids[:3], labels)
    clean = holes[~child.truth.anomaly[holes]
                  & ~labels.truth(loaded.grid, "missionX",
                                  list(channel_ids[:3])).unscorable[holes]]
    assert clean.size, "fixture produced no clean gap on the dropped channel"
    assert not child.truth.unscorable[clean].any()


def test_the_subset_carries_its_own_validity_columns(bucket, catalog, labels,
                                                     channel_ids):
    loaded, _ = _fresh(bucket, catalog, labels, channel_ids, gaps_on=(1,))
    child = loaded.subset(channel_ids[:3], labels)
    assert child.valid.shape == (len(child.grid), 3)
    assert np.array_equal(child.valid, loaded.valid[:, :3])


def test_a_bundle_with_no_gaps_is_unchanged_by_the_fold_in(bucket, catalog, labels,
                                                           channel_ids):
    """The fixture has no unobserved steps, so the fix must be a no-op there.

    That is what makes the pre-fix and post-fix numbers comparable at all: the
    correction only moves a result where something was actually unobserved.
    """
    clean, _ = _fresh(bucket, catalog, labels, channel_ids)
    assert clean.valid.all(), "fixture unexpectedly carries gaps"
    child = clean.subset(channel_ids[:3], labels)
    direct = labels.truth(clean.grid, "missionX", list(channel_ids[:3]))
    assert np.array_equal(child.truth.unscorable, direct.unscorable)
