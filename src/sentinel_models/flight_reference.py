"""The dynamic threshold as the flight component computes it, in NumPy.

This module is to `flight/src/DynamicThreshold.cpp` what `baseline_reference.py`
is to `flight/src/Baseline.cpp` and `reference.py` is to the GRU core: the NumPy
statement of the arithmetic, transcribed into C++ and pinned against it by golden
vectors. It is **not** a detector, has no registry entry, and nothing in
`sentinel_eval` imports it.

WHY IT EXISTS RATHER THAN CALLING `telemanom.channel_ratios`
------------------------------------------------------------
Two reasons, and the second is the one that matters.

1. `channel_ratios` scores a whole series at once. The flight core sees one tick
   at a time and must decide at the end of each segment with nothing after it.
   Stating the streamed form separately is what makes the two comparable.

2. **The flight rule dilates forward only** (`docs/MODELS.md` 39.5).
   `telemanom._buffered` widens each exceedance by `error_buffer - 1 = 99` steps
   on both sides; the backward half marks timesteps already emitted, and a
   warn-only component cannot go back and say something about a tick it passed in
   silence. Adding an option to `telemanom.py` for that would be a scope change to
   a module every published figure rests on. Restating the rule here is the same
   move `baseline_reference.py` made and for the same reason.

**The faithfulness claim is testable, and `tests/test_flight_reference.py` tests
it**: run this with `forward_only=False` and it reproduces
`telemanom.channel_ratios`' own emission array exactly. Whatever the streamed form
changes, it is not the arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .telemanom import Config, _runs, dynamic_threshold, prune


@dataclass
class Solved:
    """One segment's solve, as the flight core holds it."""

    eps: float
    sequences: list[tuple[int, int]]
    kept: list[bool]


def dilate(mask: np.ndarray, buffer: int, forward_only: bool) -> list[tuple[int, int]]:
    """`telemanom._buffered`, with the backward pad optional.

    With ``forward_only=False`` this is `_buffered` exactly, which is what makes
    the equivalence test possible.
    """
    n = mask.shape[0]
    pad = max(0, buffer - 1)
    back = 0 if forward_only else pad
    merged: list[tuple[int, int]] = []
    for lo, hi in _runs(mask):
        lo, hi = max(0, lo - back), min(n, hi + pad)
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return [(lo, hi) for lo, hi in merged if hi - lo > 1]


def solve_window(window: np.ndarray, config: Config,
                 forward_only: bool) -> Solved:
    """Choose ``eps`` over one window and prune what clears it.

    With ``forward_only=False`` this delegates to `telemanom.dynamic_threshold`
    outright, so the sweep is the published one and not a second copy of it. With
    forward-only dilation the sweep has to be restated, because the sequence
    count and coverage enter its objective and both change.
    """
    if not forward_only:
        eps, sequences = dynamic_threshold(window, config)
        kept = prune(window, sequences, eps, config.pruning_p) if sequences else []
        return Solved(float(eps), sequences, kept)

    mu = float(np.mean(window))
    sigma = float(np.std(window))
    ceiling = mu + config.z_ceiling * (sigma if np.isfinite(sigma) else 0.0)
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma == 0.0:
        return Solved(ceiling, [], [])

    best_score = -np.inf
    best = Solved(ceiling, [], [])
    reach = (float(np.max(window)) - mu) / sigma
    for z in config.z_values[config.z_values <= reach]:
        eps = mu + z * sigma
        above = window >= eps
        if not above.any():
            continue
        remainder = window[~above]
        if remainder.size == 0:
            continue
        sequences = dilate(above, config.error_buffer, forward_only=True)
        if not sequences:
            continue

        d_mu = (mu - float(np.mean(remainder))) / mu if mu else 0.0
        d_sigma = (sigma - float(np.std(remainder))) / sigma
        covered = sum(hi - lo for lo, hi in sequences)
        score = (d_mu + d_sigma) / (len(sequences) ** 2 + covered)
        if score >= best_score:
            best_score = score
            best = Solved(float(eps), sequences,
                          prune(window, sequences, float(eps), config.pruning_p))
    return best


def emissions(e_s: np.ndarray, config: Config,
              forward_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Stream one channel and return `(emission, eps_held)`.

    ``emission[t]`` is True on the tick a segment's solve decided a warning --
    the reference's own flight model (`telemanom.py:420-431`), which is what
    `last_emission` carries and what lead time is measured from.

    ``eps_held[t]`` is the threshold in force at ``t``: the one solved at the end
    of the **previous** segment, because a segment's own eps does not exist until
    the segment is over. The flight core reports `e_s[t] / eps_held[t]` and this
    is the quantity it is held to.
    """
    e_s = np.asarray(e_s, dtype=np.float32)
    steps = e_s.shape[0]
    span, stride = config.error_window, max(1, config.stride)
    emission = np.zeros(steps, dtype=bool)
    eps_held = np.zeros(steps, dtype=np.float64)

    held = 0.0
    for seg_hi in range(stride, steps + 1, stride):
        eps_held[seg_hi - stride:seg_hi] = held
        lo = max(0, seg_hi - stride - span)
        window = e_s[lo:seg_hi]
        offset = max(0, window.shape[0] - stride)

        solved = solve_window(window, config, forward_only)
        held = solved.eps

        if not np.any(window[offset:] >= solved.eps):
            continue
        for keep, (s_lo, s_hi) in zip(solved.kept, solved.sequences):
            if keep and min(s_hi, window.shape[0]) > max(s_lo, offset):
                emission[seg_hi - 1] = True
                break

    if steps % stride:                      # the trailing partial segment
        eps_held[steps - (steps % stride):] = held
    return emission, eps_held
