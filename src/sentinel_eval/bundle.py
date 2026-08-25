"""The working set: fetched once, held in memory, scored many times.

Rule 3 exists because operations are the scarce resource, not bytes. Egress is
free; a GET is not. So a run resolves its keys from the manifest, reads each
object exactly once, resamples it onto the shared grid, and releases the raw
series before reading the next -- and every detector and every fold in that run
then scores against the same resident arrays. Three detectors across three folds
cost the same ten operations as one detector on one fold.

Nothing is written to disk on the way. The Mac is a pipe, not a store.

Values pass through :mod:`sentinel_eval.normalisation`, whose only policy is
identity: ESA already min-max scaled within each channel group, and that is the
scaling the cross-channel claim depends on.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from . import normalisation
from .catalog import Catalog, Channel
from .errors import CoverageError, TaskError
from .grid import Grid, difference, resample_zoh
from .labels import LabelSet, Truth
from .read import ObjectSource, read_channel


@dataclass
class Bundle:
    """One mission's channel selection, on one grid, with its ground truth."""

    mission: str
    channels: tuple[Channel, ...]
    grid: Grid
    values: np.ndarray        # float32[T, C], NaN where unobserved
    valid: np.ndarray         # bool[T, C]
    truth: Truth
    provenance: dict

    @property
    def channel_ids(self) -> tuple[str, ...]:
        return tuple(c.channel_id for c in self.channels)

    @property
    def groups(self) -> tuple[int, ...]:
        return tuple(sorted({c.group for c in self.channels if c.group is not None}))

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes + self.valid.nbytes)

    def subset(self, channel_ids: list[str], labels: LabelSet) -> "Bundle":
        """A narrower channel selection, from arrays already in memory.

        `m1-ss5` is a strict subset of `m1-g8.9.10`, so the demoted set costs
        nothing to keep reporting: one twelve-channel load scores both, and the
        ground truth is re-derived here rather than re-read, because which events
        are visible depends on which channels were selected.

        The subset keeps the **parent's grid**, deliberately. The grid is the
        intersection of the selected channels' coverage, so a narrower selection
        would span a slightly wider window; scoring the two sets over different
        time windows would confound "which channels" with "which years". Sharing
        the grid means the pair differs in channels alone.

        Zero R2 operations.
        """
        missing = [c for c in channel_ids if c not in self.channel_ids]
        if missing:
            raise TaskError(
                f"cannot subset to {missing}: not in this bundle ({list(self.channel_ids)})"
            )
        columns = [self.channel_ids.index(c) for c in channel_ids]
        return Bundle(
            mission=self.mission,
            channels=tuple(self.channels[i] for i in columns),
            grid=self.grid,
            values=self.values[:, columns],
            valid=self.valid[:, columns],
            truth=labels.truth(self.grid, self.mission, list(channel_ids)),
            provenance={
                **self.provenance,
                "channels": list(channel_ids),
                "groups": sorted({self.channels[i].group for i in columns
                                  if self.channels[i].group is not None}),
                "derived_from": list(self.channel_ids),
            },
        )

    def describe(self) -> str:
        observed = int(self.valid.all(axis=1).sum())
        return (
            f"  {self.mission}: {len(self.channels)} channels "
            f"{list(self.channel_ids)}\n"
            f"    groups {list(self.groups)}   grid {self.grid.describe()}\n"
            f"    fully observed {observed:,}/{len(self.grid):,} timesteps "
            f"({100 * observed / len(self.grid):.1f}%)   resident "
            f"{self.nbytes / 1048576:.0f} MiB"
        )


def load(
    source: ObjectSource,
    catalog: Catalog,
    labels: LabelSet,
    *,
    mission: str,
    channel_ids: list[str],
    start=None,
    end=None,
    period=None,
    max_hold=None,
    log=None,
) -> Bundle:
    """Read the selection once and put it on a shared grid.

    Categorical channels are refused rather than silently dropped: they are
    stored as strings, arithmetic on them is meaningless, and a caller who asked
    for one has a wrong task definition that should surface here.
    """
    channels = [catalog.channel(mission, cid) for cid in channel_ids]
    categorical = [str(c) for c in channels if not c.numeric]
    if categorical:
        raise TaskError(
            f"categorical channels cannot be forecast: {categorical}. They are stored as "
            f"strings ({channels[0].value_dtype!r} and similar); select numeric channels."
        )

    if period is None:
        period, provenance_kind = catalog.resample_period(mission)
    else:
        period, provenance_kind = np.timedelta64(period), "overridden"

    grid = _grid_for(channels, start, end, period)
    to_difference = set(catalog.monotonic_channels(mission)) & set(channel_ids)

    values = np.empty((len(grid), len(channels)), dtype=np.float32)
    valid = np.zeros((len(grid), len(channels)), dtype=bool)

    for i, channel in enumerate(channels):
        series = read_channel(source, channel)          # one GET per object
        column, ok = resample_zoh(series, grid, max_hold=max_hold)
        del series                                       # release before the next
        if channel.channel_id in to_difference:
            column = difference(column)
            ok = ok & ~np.isnan(column)
        values[:, i] = column
        valid[:, i] = ok
        if log:
            log(f"    {channel.channel_id:<14} {int(ok.sum()):>10,}/{len(grid):,} "
                f"observed{'  [differenced]' if channel.channel_id in to_difference else ''}")

    values = normalisation.apply(values)                  # identity, enforced

    truth = labels.truth(grid, mission, list(channel_ids))
    unobserved = ~valid.all(axis=1)
    truth = dataclasses.replace(
        truth, unscorable=(truth.unscorable | unobserved) & ~truth.anomaly
    )

    return Bundle(
        mission=mission,
        channels=tuple(channels),
        grid=grid,
        values=values,
        valid=valid,
        truth=truth,
        provenance={
            **catalog.provenance(),
            "mission": mission,
            "channels": list(channel_ids),
            "groups": sorted({c.group for c in channels if c.group is not None}),
            "grid_start": str(grid.start),
            "grid_steps": len(grid),
            "grid_period_seconds": int(period / np.timedelta64(1, "s")),
            "grid_period_source": provenance_kind,
            "resampling": catalog.resampling_method,
            "normalisation": normalisation.IDENTITY,
            "differenced": sorted(to_difference),
            "unobserved_steps": int(unobserved.sum()),
        },
    )


def _grid_for(channels: list[Channel], start, end, period) -> Grid:
    """The window every selected channel actually covers.

    Channels start and end at different times, so the grid is their intersection
    unless the caller pins it. Asking for a window a channel does not cover is a
    :class:`~sentinel_eval.errors.CoverageError`, not a column of NaN.
    """
    latest_start = max(c.time_start for c in channels)
    earliest_end = min(c.time_end for c in channels)

    lo = latest_start if start is None else np.datetime64(start, "ns")
    hi = earliest_end if end is None else np.datetime64(end, "ns")

    if start is not None and lo < latest_start:
        raise CoverageError(
            f"requested start {lo} precedes coverage: the latest channel begins {latest_start}"
        )
    if end is not None and hi > earliest_end:
        raise CoverageError(
            f"requested end {hi} exceeds coverage: the earliest channel ends {earliest_end}"
        )
    return Grid.spanning(lo, hi, period)
