"""Work item 9.10. Pre-registered in `docs/MODELS.md` 27.

**Twelve arms, one bundle load, cached weights.** Sections 26.29 and 26.30 read
telemanom's source (vendored at `third_party/telemanom/`, D53) and found five
mechanisms this reproduction does not implement, a precision denominator that is a
mixed unit, and a target of **12 false positives** rather than "~91 candidate
ranges". This script ports the detection stack completely as a reference ceiling
and puts the ladder beneath it so every mechanism stays attributable.

    A0  lstm  1a+1b                          prop   gate + ladder base
    A1  lstm  as source (telemanom.py)       prop   gate  -> 44/98, 45/59
    A2  lstm  1a+1b+1c-ii  NOT A REPRODUCTION prop  gate  -> 74/98, 98/221
    A3  gru   as source, swept               prop   gate  -> 0.551, 10/38, 47/100
    A4  gru   1a+1b+1c-ii  NOT A REPRODUCTION prop  the pre-registered comparison
    A5  lstm  1a+1b+1c-ii  NOT A REPRODUCTION pub   the regime control
    L1  lstm  A0 + magnitude conjunct        prop   errors.py:342-343
    L2  lstm  L1 + whole-window bail-out     prop   errors.py:337-340
    L3  lstm  L2 + the find_epsilon guards   prop   errors.py:314-315
    L4  lstm  L3 + the inverse pass          prop   errors.py:132-148
    L5  lstm  L4 + the published window      pub    errors.py:84-93
    F   lstm  the complete port              pub    the reference ceiling
    FG  gru   the complete port              pub    THE NEW 38 BASELINE

**`src/sentinel_models/telemanom.py` is not touched** (D8, 26.21.2). Every
correction lives here, and every mechanism cites the vendored source by line.

    PYTHONPATH=src .venv/bin/python scripts/smap_rungs.py \
        --visibility runs/smap-msl/_forensics/<stage1>.json \
        --stage4     runs/smap-msl/_forensics/<stage4>.json
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel_data import config as C                                  # noqa: E402
from sentinel_eval import ops                                          # noqa: E402
from sentinel_eval.detector import Context, reduce_scores               # noqa: E402
from sentinel_eval.metrics.counts import Count                          # noqa: E402
from sentinel_eval.metrics.ranges import mask_to_ranges                 # noqa: E402
from sentinel_models import detectors as D                             # noqa: E402
from sentinel_models.lstm import Hyper, TelemanomRNN, train as fit_model  # noqa: E402
from sentinel_models.windows import aggregate_predictions, decay        # noqa: E402
import torch                                                            # noqa: E402
from sentinel_models import registry                                   # noqa: E402
from sentinel_models import telemanom                                  # noqa: E402

#: Cells fitted in a worker rather than in the parent, because they refit or
#: are expensive. Everything else is scored from banked weights in-process.
DEFERRED_CELLS = ("published", "rung", "zscore", "published_nc", "zscore_cmd")
MANIFEST_KEY = "_manifest/smap_msl.json"
PRE_REGISTRATION = "docs/MODELS.md 27"

#: telemanom's published constants, read from third_party/telemanom/config.yaml.
BATCH_SIZE = 70            # config.yaml batch_size
WINDOW_SIZE = 30           # config.yaml window_size
L_S = 250                  # config.yaml l_s
N_PREDICTIONS = 10         # config.yaml n_predictions

#: The published training configuration, `docs/MODELS.md` 28.1 T-b to T-h.
PUBLISHED_HYPER = dict(train_batch_size=64, min_delta_absolute=0.0003,
                       validation_split="shuffled", full_epochs=True,
                       restore_best=False)
#: 30.1. One multiplier, **fixed at 1.0 and not swept**, chosen because it is
#: the weakest non-trivial statement of "above the channel's own noise". If the
#: arm needs a different value to work, that is a finding about the form and is
#: reported as one rather than absorbed by a sweep.
DIM_K = 1.0
SD_LIM = 12.0              # errors.py:241
Z_STEP = 0.5               # errors.py:285
Z_FLOOR = 2.5              # errors.py:285


# ===========================================================================
#  The port. Every function is transcribed from
#  third_party/telemanom/telemanom/errors.py at the cited lines.
# ===========================================================================

@dataclass(frozen=True)
class Mech:
    """Which of the source's mechanisms this arm implements.

    The ladder turns them on one at a time, in the order the source applies
    them, so each arm differs from the one below it by exactly one lever -- the
    rule 26.23.2 stated and rung 1c broke.
    """
    magnitude: bool = False     # errors.py:342-343  (e_s > 0.05 * inter_range)
    bailout: bool = False       # errors.py:337-340  whole-window scale check
    guards: bool = False        # errors.py:314-315  len(E_seq)<=5, coverage<0.5
    inverse: bool = False       # errors.py:132-148  the reflected pass
    published_window: bool = False   # errors.py:84-93  adjust_window_size
    clip: bool = True           # errors.py:355-359. False = 1c-ii's unclipped union
    #: 30.1, work item 9.13. Replaces the two candidate filters that are
    #: absolute in the units of the data with `mean(e_s) + DIM_K*sd(e_s)`.
    #: The two that are already scale-free -- `sd_e_s > 0.05*sd_values`, the
    #: coverage cap and the sequence cap -- are left exactly as published,
    #: because changing a rule that is already dimensionless would be a second
    #: lever. D55 is the principle; this is one implementation of it, and 30.2
    #: decides whether it earns adoption.
    dimensionless: bool = False
    #: A multiplier on epsilon. **A deviation from the source**, and the only
    #: one here: the port emits a boolean decision with no dial, so matching it
    #: to another arm's alarm rate (D41, D44) needs one introduced. 1.0 is the
    #: published algorithm exactly; any other value is labelled as a deviation
    #: wherever the arm is reported.
    eps_mult: float = 1.0


#: 27.8 measured arm L1 identical to A0, which is only a statement about the
#: data if the conjunct can bind at all. `tests/test_smap_rungs_port.py` proves
#: it can; these counters say how often it does on the real stream.
CONJUNCT_BOUND = [0]
CONJUNCT_TESTED = [0]

#: 29.1's instrument. Reset per channel; every guard records how often it fired,
#: recorded whatever the outcome so the diagnosis is not a second run's job.
GUARDS = {}


def reset_guards():
    GUARDS.clear()
    GUARDS.update(bailout=0, magnitude=0, seq_cap=0, coverage=0,
                  windows=0, fallback=0)


def _groups(idx: np.ndarray) -> list[tuple[int, int]]:
    """`mit.consecutive_groups`, then `[(g[0], g[-1]) for g in groups]`.

    Inclusive ends, as the source uses them (errors.py:304, :375, :159).
    """
    if idx.size == 0:
        return []
    idx = np.asarray(idx, dtype=np.int64)
    cuts = np.flatnonzero(np.diff(idx) != 1)
    starts = np.concatenate(([0], cuts + 1))
    ends = np.concatenate((cuts, [idx.size - 1]))
    return [(int(idx[a]), int(idx[b])) for a, b in zip(starts, ends)]


def _drop_singletons(seqs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """`if not g[0] == g[-1]` -- errors.py:304, :375, :160."""
    return [(a, b) for a, b in seqs if a != b]


def _buffered(i_anom: np.ndarray, n: int, width: int) -> np.ndarray:
    """Dilate by `arange(1, width)` either side, clip, unique -- errors.py:291-298.

    `compare_to_epsilon` uses `arange(1, error_buffer+1)` (errors.py:347) and
    `find_epsilon` uses `arange(1, error_buffer)` (errors.py:291). The asymmetry
    is upstream's; `width` carries it rather than hiding it.

    Computed by dilating the contiguous runs rather than materialising a
    hundred indices per exceeded timestep. The set is identical -- the same
    argument `src/sentinel_models/telemanom.py:208-224` makes -- and it is the
    difference between seconds and hours over twelve arms.
    """
    if i_anom.size == 0:
        return i_anom
    pad = max(0, width - 1)
    mask = np.zeros(n, dtype=bool)
    mask[i_anom] = True
    for lo, hi in mask_to_ranges(mask):
        mask[max(0, lo - pad):min(n, hi + pad)] = True
    return np.flatnonzero(mask)


def find_epsilon(e_s, mean_e_s, sd_e_s, error_buffer, mech: Mech):
    """errors.py:269-322. Returns the chosen epsilon.

    Note the source uses the FORWARD window's mean and sd on both passes
    (errors.py:286, :319, :322); only `e_s` is reflected.
    """
    epsilon = mean_e_s + SD_LIM * sd_e_s          # errors.py:252-253, the fallback
    max_score = -10000000.0
    chosen = False
    n = e_s.shape[0]
    for z in np.arange(Z_FLOOR, SD_LIM, Z_STEP):
        eps = mean_e_s + sd_e_s * z
        pruned = e_s[e_s < eps]                    # errors.py:288
        i_anom = np.flatnonzero(e_s >= eps)        # errors.py:290
        i_anom = _buffered(i_anom, n, error_buffer)          # errors.py:291-298
        if i_anom.size == 0 or pruned.size == 0:
            continue
        E_seq = _drop_singletons(_groups(i_anom))            # errors.py:302-304
        mean_dec = (mean_e_s - float(np.mean(pruned))) / mean_e_s if mean_e_s else 0.0
        sd_dec = (sd_e_s - float(np.std(pruned))) / sd_e_s if sd_e_s else 0.0
        score = (mean_dec + sd_dec) / (len(E_seq) ** 2 + i_anom.size)   # errors.py:310-311
        # errors.py:314-315. The two guards are deviation 8; L3 turns them on.
        ok = score >= max_score
        if mech.guards:
            if score >= max_score and len(E_seq) > 5:
                GUARDS["seq_cap"] = GUARDS.get("seq_cap", 0) + 1
            if score >= max_score and i_anom.size >= (n * 0.5):
                GUARDS["coverage"] = GUARDS.get("coverage", 0) + 1
            ok = ok and len(E_seq) <= 5 and i_anom.size < (n * 0.5)
        if ok:
            max_score = score
            chosen = True
            epsilon = mean_e_s + z * sd_e_s
    GUARDS["windows"] = GUARDS.get("windows", 0) + 1
    if not chosen:
        # Nothing was admissible, so epsilon stays at telemanom's sd_lim
        # fallback (errors.py:252) -- a bar almost nothing clears.
        GUARDS["fallback"] = GUARDS.get("fallback", 0) + 1
    return float(epsilon)


def compare_to_epsilon(e_s, e_s_fwd, epsilon, y_win, window_num, batch_size,
                       num_to_ignore, prior_anoms, batch_position, error_buffer,
                       mech: Mech, judged_lo: int | None):
    """errors.py:324-384. Returns `(i_anom, E_seq, non_anom_max)`.

    `judged_lo` is this loop's clip point. In the port it is
    `len(e_s) - batch_size` for every window but the first, which is
    errors.py:359 exactly. In the ladder loop it is the judged segment's offset,
    which 26.29.1 established is the same construction.
    """
    n = e_s.shape[0]
    sd_e_s = float(np.std(e_s_fwd))
    mean_e_s = float(np.mean(e_s_fwd))
    sd_values = float(np.std(y_win))
    perc_high, perc_low = np.percentile(y_win, [95, 5])
    inter_range = float(perc_high - perc_low)

    # 30.1. The channel's own noise level, in the channel's own units. Computed
    # from the FORWARD window like every other statistic these guards use, so
    # the arm differs from the source in the CONSTANT and in nothing else.
    dim_cut = mean_e_s + DIM_K * sd_e_s

    # errors.py:337-340. Note the source tests the FORWARD e_s on both passes.
    if mech.bailout:
        # errors.py:337-339 is a THREE-term test and 30.1's table names two of
        # them. Replaced: `max(e_s) > 0.05`, its first row. Left as published:
        # `sd_e_s > 0.05 * sd_values`, its third row, because that is already a
        # ratio -- and ALSO `max(e_s) > 0.05 * inter_range`, the middle term of
        # `big_enough`, which 30.1's table does not list at all.
        #
        # **That omission is the pre-registration's and is not resolved here.**
        # Its second row is `e_s > 0.05 * inter_range`, the conjunct at :343,
        # and the same FORM appears at :338 applied to `max(e_s)`. Reading the
        # table literally leaves :338 alone; reading it as a form-substitution
        # would change it too. The literal reading is taken, because a
        # pre-registration is a literal instrument and moving an unlisted lever
        # is the thing rung 1c's stop fired for (26.26). Recorded in 30.4.
        floor = dim_cut if mech.dimensionless else 0.05
        big_enough = (sd_e_s > 0.05 * sd_values
                      or float(np.max(e_s_fwd)) > 0.05 * inter_range)
        if not big_enough or not float(np.max(e_s_fwd)) > floor:
            GUARDS["bailout"] = GUARDS.get("bailout", 0) + 1
            return np.array([], dtype=np.int64), [], -1000000.0

    above = e_s >= epsilon * mech.eps_mult
    bound = 0
    if mech.magnitude:                                        # errors.py:342-343
        # 30.1 replaces `e_s > 0.05 * inter_range`, which is semi-absolute --
        # a fraction of the VALUES' spread applied to the ERRORS' scale.
        cut = dim_cut if mech.dimensionless else 0.05 * inter_range
        keep = above & (e_s > cut)
        bound = int(above.sum() - keep.sum())   # how often the conjunct BINDS
        if bound:
            GUARDS["magnitude"] = GUARDS.get("magnitude", 0) + 1
        above = keep
    CONJUNCT_BOUND[0] += bound
    CONJUNCT_TESTED[0] += 1
    i_anom = np.flatnonzero(above)
    if i_anom.size == 0:
        return np.array([], dtype=np.int64), [], -1000000.0

    i_anom = _buffered(i_anom, n, error_buffer + 1)            # errors.py:347-353

    # errors.py:355-359. The clip. 26.29.1: this is telemanom's own, and it is
    # what makes the algorithm causal after its opening window.
    if mech.clip:
        if window_num == 0:
            i_anom = i_anom[i_anom >= num_to_ignore]
        else:
            lo = (n - batch_size) if judged_lo is None else judged_lo
            i_anom = i_anom[i_anom >= lo]
    i_anom = np.unique(i_anom)
    if i_anom.size == 0:
        return i_anom, [], -1000000.0

    # errors.py:363-371. 1a's rung and 1b's cross-window accumulator, together.
    window_indices = np.arange(0, n) + batch_position
    adj = i_anom + batch_position
    keep = np.setdiff1d(window_indices, np.append(prior_anoms, adj))
    candidates = np.unique(keep - batch_position)
    non_anom_max = float(np.max(e_s[candidates])) if candidates.size else -1000000.0

    E_seq = _drop_singletons(_groups(i_anom))                 # errors.py:374-375
    return i_anom, E_seq, non_anom_max


def prune_anoms(e_s, E_seq, i_anom, non_anom_max, p):
    """errors.py:386-435. Ties remove EVERY sequence at the rank (errors.py:411)."""
    if not E_seq:
        return np.array([], dtype=np.int64)
    peaks = np.array([float(np.max(e_s[a:b + 1])) for a, b in E_seq])
    ladder = np.append(np.sort(peaks)[::-1], [non_anom_max])   # errors.py:404-405
    remove: list[int] = []
    for i in range(len(ladder) - 1):
        step = ladder[i] - ladder[i + 1]
        rel = step / ladder[i] if ladder[i] else 0.0
        if rel < p:
            remove.extend(np.flatnonzero(peaks == ladder[i]).tolist())
        else:
            remove = []
    kept = [s for j, s in enumerate(E_seq) if j not in set(remove)]
    if not kept:
        return np.array([], dtype=np.int64)
    keep_idx = np.concatenate([np.arange(a, b + 1) for a, b in kept])
    return i_anom[np.isin(i_anom, keep_idx)]                   # errors.py:427-435


def _one_window(e_s_win, y_win, window_num, batch_size, num_to_ignore,
                prior_anoms, batch_position, error_buffer, p, mech, judged_lo):
    """One window's surviving absolute-relative indices: the source's inner loop."""
    mean_e_s = float(np.mean(e_s_win))
    sd_e_s = float(np.std(e_s_win))
    passes = [e_s_win]
    if mech.inverse:                                           # errors.py:249-250
        passes.append(np.asarray([mean_e_s + (mean_e_s - v) for v in e_s_win],
                                 dtype=np.float64))
    found = []
    for series in passes:
        eps = find_epsilon(series, mean_e_s, sd_e_s, error_buffer, mech)
        i_anom, E_seq, non_anom_max = compare_to_epsilon(
            series, e_s_win, eps, y_win, window_num, batch_size, num_to_ignore,
            prior_anoms, batch_position, error_buffer, mech, judged_lo)
        found.append(prune_anoms(series, E_seq, i_anom, non_anom_max, p))
    if not found:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(found))                    # errors.py:147-148


def run_port(e_s_full, y_full, error_buffer, p, tail=0, presliced=False,
             alarm_len=None, dimensionless=False, eps_mult=1.0):
    """errors.py:111-168, `process_batches`, with `adjust_window_size`.

    Operates in telemanom's own coordinates: `y_test` is the raw series offset by
    `l_s`, and the surviving indices are shifted back by `+ l_s` (errors.py:165).
    """
    # 28.1 T-g: `y_test` has `len - l_s - n_predictions` rows (channel.py:55),
    # so the covered raw span is [l_s, len - n_predictions). `tail` carries it.
    if presliced:
        # 29.1: the residual rung already produced the supervised series.
        e_s = np.asarray(e_s_full, dtype=np.float64)
        y = np.asarray(y_full, dtype=np.float64)
        total = int(alarm_len)
    else:
        stop = len(e_s_full) - tail if tail else len(e_s_full)
        e_s = np.asarray(e_s_full[L_S:stop], dtype=np.float64)
        y = np.asarray(y_full[L_S:stop], dtype=np.float64)
        total = len(e_s_full)
    n = y.shape[0]
    alarm = np.zeros(total, dtype=bool)
    eps_at = np.full(total, np.nan)
    if n <= BATCH_SIZE:
        return alarm, 0, WINDOW_SIZE, eps_at

    window_size = WINDOW_SIZE                                  # errors.py:84-93
    n_windows = int((n - BATCH_SIZE * window_size) / BATCH_SIZE)
    while n_windows < 0:
        window_size -= 1
        n_windows = int((n - BATCH_SIZE * window_size) / BATCH_SIZE)
        if window_size == 1 and n_windows < 0:
            return alarm, 0, window_size, eps_at

    num_to_ignore = L_S * 2                                    # errors.py:262-267
    if n < 2500:
        num_to_ignore = L_S
    if n < 1800:
        num_to_ignore = 0

    accumulated = np.array([], dtype=np.int64)
    for i in range(0, n_windows + 1):                          # errors.py:123-130
        prior_idx = i * BATCH_SIZE
        idx = (window_size * BATCH_SIZE) + (i * BATCH_SIZE)
        if i == n_windows:
            idx = n
        # 33.3. `eps_mult` is a LABELLED DEVIATION from the source (28.6): the
        # published algorithm emits a boolean and has no multiplier, and this
        # arm needs one to be brought to another arm's alarm rate at all.
        # 1.0 is the published algorithm exactly.
        full = Mech(magnitude=True, bailout=True, guards=True,
                    inverse=True, published_window=True, clip=True,
                    dimensionless=dimensionless, eps_mult=eps_mult)
        win = e_s[prior_idx:idx]
        eps = find_epsilon(win, float(np.mean(win)), float(np.std(win)),
                           error_buffer, full)
        judged = slice(prior_idx + (num_to_ignore if i == 0 else len(win) - BATCH_SIZE),
                       idx)
        eps_at[L_S + judged.start:L_S + judged.stop] = eps
        survivors = _one_window(win, y[prior_idx:idx], i,
                                BATCH_SIZE, num_to_ignore, accumulated,
                                prior_idx, error_buffer, p, full, None)
        if survivors.size:
            accumulated = np.append(accumulated, survivors + prior_idx)

    if accumulated.size:
        for a, b in _drop_singletons(_groups(np.unique(accumulated))):
            alarm[a + L_S:b + 1 + L_S] = True                  # errors.py:159-166
    return alarm, n_windows + 1, window_size, eps_at


def run_ladder(e_s_full, y_full, cfg, p, mech: Mech):
    """The ladder loop: `channel_ratios`' striding, with mechanisms switchable.

    26.29.1 established this geometry is telemanom's -- a judged segment at the
    end of a trailing reference window. Kept separate from `run_port` so that
    `F` minus `L5` isolates the loop structure from the mechanisms (27.4, F4).
    """
    e_s = np.asarray(e_s_full, dtype=np.float64)
    y = np.asarray(y_full, dtype=np.float64)
    steps = e_s.shape[0]
    span, stride = cfg.error_window, max(1, cfg.stride)
    alarm = np.zeros(steps, dtype=bool)
    accumulated = np.array([], dtype=np.int64)
    windows = 0

    for w, seg_lo in enumerate(range(0, steps, stride)):
        seg_hi = min(seg_lo + stride, steps)
        ref_lo = max(0, seg_lo - span)
        offset = seg_lo - ref_lo
        e_win, y_win = e_s[ref_lo:seg_hi], y[ref_lo:seg_hi]
        if e_win.size == 0:
            continue
        windows += 1
        # `prior_anoms` is compared against `i_anom + batch_position` inside
        # compare_to_epsilon, so it is absolute -- errors.py:367-369.
        survivors = _one_window(e_win, y_win, w, stride, 0,
                                accumulated, ref_lo,
                                cfg.error_buffer, p, mech,
                                offset if mech.clip else None)
        if survivors.size:
            absolute = survivors + ref_lo
            accumulated = np.unique(np.append(accumulated, absolute))

    if accumulated.size:
        for a, b in _drop_singletons(_groups(np.unique(accumulated))):
            alarm[a:b + 1] = True
    return alarm, windows


# ===========================================================================
#  Arms, scoring and the load
# ===========================================================================

LADDER = (
    ("A0", Mech(), "prop", "1a+1b -- the faithful arm (26.29.1)"),
    ("A2", Mech(clip=False), "prop", "1c-ii -- NOT A REPRODUCTION, gate only"),
    ("A5", Mech(clip=False), "pub", "1c-ii at the published window -- regime control"),
    ("L1", Mech(magnitude=True), "prop", "+ magnitude conjunct (errors.py:342-343)"),
    ("L2", Mech(magnitude=True, bailout=True), "prop",
     "+ whole-window bail-out (errors.py:337-340)"),
    ("L3", Mech(magnitude=True, bailout=True, guards=True), "prop",
     "+ the two find_epsilon guards (errors.py:314-315)"),
    ("L4", Mech(magnitude=True, bailout=True, guards=True, inverse=True), "prop",
     "+ the inverse pass (errors.py:132-148)"),
    ("L5", Mech(magnitude=True, bailout=True, guards=True, inverse=True,
                published_window=True), "pub",
     "+ the published window (errors.py:84-93)"),
)
GATES = {"A0": "lstm", "A1": "lstm", "A2": "lstm", "A3": "gru"}


def make_grid(top: float = 50.0) -> np.ndarray:
    return np.unique(np.concatenate([np.geomspace(0.02, 1.0, 60),
                                     np.geomspace(1.0, top, 60), [1.0]]))


#: 33.3's grid for the port dial. Bounded at [1, 6] because the multiplier
#: RAISES epsilon and C1 has to come DOWN from 1.88% to 0.68%.
#:
#: **Two-stage since 2026-09-09 (33.8).** The 24-point geometric grid resolved the
#: curve's shape and NOT its crossing: its neighbours landed at 0.7578% and
#: 0.4465% with the 0.6838% target between them, so the arm was compared 35%
#: quieter than the one it was measured against and M1 could not be adjudicated.
#: The coarse points are kept unchanged so the two curves are comparable, and
#: [1.80, 2.10] is resolved at 0.01 where the crossing actually is.
PORT_GRID = np.unique(np.concatenate([np.geomspace(1.0, 6.0, 24),
                                      np.arange(1.80, 2.101, 0.01)]))


def free_memory_gib() -> float:
    """Free plus inactive pages, in GiB. 31.6's gate."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        size = int(out.split("page size of")[1].split("bytes")[0].strip())
        pages = 0
        for line in out.splitlines():
            if line.startswith(("Pages free", "Pages inactive")):
                pages += int(line.split(":")[1].strip().rstrip("."))
        return pages * size / 1073741824
    except Exception:
        return float("inf")          # unknown is not a reason to refuse to run


def fetch(client, bucket, key):
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def ctx_for(cid, n):
    """Identical to `scripts/smap_stage2.py:59-62` with `commands=None`.

    **The channel id is load-bearing.** It enters `context.channels`, which is
    the second element of the weight-cache key (`detectors.py:421-424`), so a
    different string here orphans every banked fit -- D14's failure exactly.
    """
    return Context(mission="smap-msl", channels=(cid,), groups=(0,),
                   period_seconds=1.0, fold=0, window=(0, n),
                   commands=None, command_ids=())


def ctx_cmd(cid, n, commands):
    """`ctx_for` with the command columns attached -- 28.1 T-f."""
    return Context(mission="smap-msl", channels=(cid,), groups=(0,),
                   period_seconds=1.0, fold=0, window=(0, n),
                   commands=commands, command_ids=())


def _exog(commands) -> np.ndarray:
    """`(T, 2K)` -- each command's impulse and how recently it fired.

    32.1's encoding, and it is the one already in the tree: `windows.decay` is
    the same function `command_features` calls, and computing the trace over the
    whole series is what that function's warm-up prefix approximates. It is exact
    here and bounded because a SMAP/MSL channel is thousands of steps, not the
    14.7M ESA-ADB's windowing was written for (`windows.py:73-88`).
    """
    imp = np.ascontiguousarray(np.asarray(commands, dtype=np.float32))
    return np.concatenate([imp, decay(imp)], axis=1)


def build_published(cid, train, test):
    """Arm T: telemanom's training configuration, its inputs and its aggregation.

    Eight differences at once, deliberately (27/28.2): this is a reproduction arm
    and reproducing a configuration one constant at a time would take eight reads.
    """
    det = registry.build("lstm-telemanom")
    det.hyper = Hyper(cell="lstm", **PUBLISHED_HYPER)          # T-b, T-c, T-d, T-e, T-h
    det.wants_commands = True                                   # T-f
    det.config = telemanom.Config(error_window=BATCH_SIZE * WINDOW_SIZE,
                                  stride=BATCH_SIZE, aggregate="first")   # T-a
    v_tr, v_te = train[:, :1].astype(np.float32), test[:, :1].astype(np.float32)
    c_tr, c_te = train[:, 1:].astype(np.float32), test[:, 1:].astype(np.float32)
    det.fit(v_tr, np.ones(len(v_tr), dtype=bool), ctx_cmd(cid, len(v_tr), c_tr))
    s_ctx = ctx_cmd(cid, len(v_te), c_te)
    e_s = np.asarray(det._smoothed_errors(v_te, s_ctx), dtype=np.float64)[:, 0]
    warm = int(det.warmup_steps)
    D.clear_caches()
    return {"e_s": e_s, "as_source": np.zeros(len(v_te)),
            "values": v_te[:, 0].astype(np.float64),
            "warm": {"prop": warm, "pub": warm},
            "cfg": {"prop": det.config, "pub": det.config}}


def build_published_nocmd(cid, train, test):
    """Arm C1: Arm T with the commands withheld, and nothing else changed.

    32.1. Arm T is already the command-conditioned arm (28.1 T-f,
    `channel.py:63-67`), so the counterfactual is this one and it is the one that
    has to be fitted. Every other published difference -- T-a's aggregation,
    T-b's trainer batch, T-c's absolute `min_delta`, T-d's shuffled split, T-e's
    full epochs, T-h's non-restoration -- is held identical, so the pair differs
    in exactly one lever.
    """
    det = registry.build("lstm-telemanom")
    det.hyper = Hyper(cell="lstm", **PUBLISHED_HYPER)
    det.wants_commands = False                                  # <- the lever
    det.config = telemanom.Config(error_window=BATCH_SIZE * WINDOW_SIZE,
                                  stride=BATCH_SIZE, aggregate="first")
    v_tr, v_te = train[:, :1].astype(np.float32), test[:, :1].astype(np.float32)
    det.fit(v_tr, np.ones(len(v_tr), dtype=bool), ctx_for(cid, len(v_tr)))
    s_ctx = ctx_for(cid, len(v_te))
    e_s = np.asarray(det._smoothed_errors(v_te, s_ctx), dtype=np.float64)[:, 0]
    warm = int(det.warmup_steps)
    D.clear_caches()
    return {"e_s": e_s, "as_source": np.zeros(len(v_te)),
            "values": v_te[:, 0].astype(np.float64),
            "warm": {"prop": warm, "pub": warm},
            "cfg": {"prop": det.config, "pub": det.config}}


def _fit_published(job):
    """One published fit, in its own process. Must be module-level to be picklable.

    `docs/MODELS.md` 28.5: the bundle is loaded once in the parent and the fits
    are pure compute, so fanning them out costs **zero** extra R2 operations.
    Each worker narrows torch to `threads` so four of them do not put sixteen
    threads on ten cores; the fit is seeded per channel (`train` calls
    `configure_determinism(seed + fold)`), so which process runs it cannot change
    the weights.
    """
    cid, train, test, cells = job
    import sentinel_models.lstm as L
    import resource
    L.THREADS = 4
    started, out, err = time.time(), {}, {}
    builders = {"published": build_published, "zscore": build_zscore,
                "published_nc": build_published_nocmd, "zscore_cmd": build_zscore_cmd}
    for cell in cells:
        try:
            out[cell] = builders.get(cell, build_rung)(cid, train, test)
        except Exception as exc:
            err[cell] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    # 32.5. Peak RSS of THIS process, which for a worker includes its own torch
    # import -- the term 28.5's arithmetic left out and the reason that run died.
    # `ru_maxrss` is bytes on macOS and kibibytes on Linux; there is no portable
    # unit and guessing from the magnitude is how a plausible number becomes a
    # wrong one, so the platform decides.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / 1048576.0 if sys.platform == "darwin" else peak / 1024.0
    return cid, out, err, time.time() - started, peak_mb


def raw_errors(det, values, s_ctx):
    """The UNSMOOTHED residual, mirroring `_smoothed_errors` without the EWMA.

    Needed because 29.1's rung changes *what the EWMA runs over*, and
    `_smoothed_errors` has already smoothed by the time it returns.
    """
    filled = det._filled(values)
    steps = filled.shape[0]
    block = det.chunks * det.chunk_steps
    out = np.empty(steps, dtype=np.float64)
    for lo in range(0, steps, block):
        hi = min(lo + block, steps)
        out[lo:hi] = np.abs(filled[lo:hi, 0] - det._forecast(filled, lo, hi)[:, 0])
    return out


def residual_rung(e_raw, n_test, cid, smoothing_window):
    """errors.py:48-64. The supervised region only, then the opening flattened.

    telemanom's `e` exists only where a supervised target exists -- length
    `n - l_s - n_predictions` -- so its EWMA never sees a residual taken against
    padded history, and its first `l_s` smoothed samples are then replaced
    outright by the mean of the first `2*l_s` for every channel but `C-2`.
    """
    e = np.asarray(e_raw[L_S:n_test - N_PREDICTIONS], dtype=np.float64)
    e_s = telemanom.ewma(e.reshape(-1, 1), smoothing_window).ravel().astype(np.float64)
    if cid != "C-2" and e_s.size > L_S:                     # errors.py:62-64
        e_s[:L_S] = float(np.mean(e_s[:min(2 * L_S, e_s.size)]))
    return e_s


def build_rung(cid, train, test):
    """Arm R: Arm T's model and inputs, with the residual built telemanom's way."""
    det = registry.build("lstm-telemanom")
    det.hyper = Hyper(cell="lstm", **PUBLISHED_HYPER)
    det.wants_commands = True
    det.config = telemanom.Config(error_window=BATCH_SIZE * WINDOW_SIZE,
                                  stride=BATCH_SIZE, aggregate="first")
    v_tr, v_te = train[:, :1].astype(np.float32), test[:, :1].astype(np.float32)
    c_tr, c_te = train[:, 1:].astype(np.float32), test[:, 1:].astype(np.float32)
    det.fit(v_tr, np.ones(len(v_tr), dtype=bool), ctx_cmd(cid, len(v_tr), c_tr))
    s_ctx = ctx_cmd(cid, len(v_te), c_te)
    e_raw = raw_errors(det, v_te, s_ctx)
    e_s = residual_rung(e_raw, len(v_te), cid, det.config.smoothing_window)
    warm = int(det.warmup_steps)
    D.clear_caches()
    return {"e_s": e_s, "as_source": np.zeros(len(v_te)),
            "values": v_te[:, 0].astype(np.float64),
            "y_sup": v_te[L_S:len(v_te) - N_PREDICTIONS, 0].astype(np.float64),
            "n_full": len(v_te), "supervised": True,
            "warm": {"prop": warm, "pub": warm},
            "cfg": {"prop": det.config, "pub": det.config}}


#: D56 / `docs/MODELS.md` 31. The same GRU, a head that predicts its own
#: uncertainty, and a statistic that is dimensionless by construction.
ZSCORE_HYPER = dict(cell="gru", head="gaussian")


def _to_torch(weights, hyper, n_channels, n_exogenous):
    """Rebuild the torch module from the arrays `to_weights` produced.

    The study script runs its own forward pass rather than going through
    `reference.forward`, because that module is the flight blueprint and
    `flight/` is transcribed from it. **A design that has not cleared its own
    gate does not get to move the flight path** (D56 consequence 3); the
    transcription is deferred until it does.
    """
    model = TelemanomRNN(n_channels, hyper, n_exogenous)
    with torch.no_grad():
        for i, layer in enumerate(weights.layers):
            getattr(model.rnn, f"weight_ih_l{i}").copy_(torch.from_numpy(layer.w_ih))
            getattr(model.rnn, f"weight_hh_l{i}").copy_(torch.from_numpy(layer.w_hh))
            getattr(model.rnn, f"bias_ih_l{i}").copy_(torch.from_numpy(layer.b_ih))
            getattr(model.rnn, f"bias_hh_l{i}").copy_(torch.from_numpy(layer.b_hh))
        model.head.weight.copy_(torch.from_numpy(weights.head_w))
        model.head.bias.copy_(torch.from_numpy(weights.head_b))
    model.eval()
    return model


def predictive_moments(model, values, window, chunk=1000, exog=None):
    """`mu` and `sigma` per timestep, from warmed chunks started at a zero state.

    The same geometry `ForecastDetector._forecast` uses -- every chunk is
    preceded by `window` real steps -- so nothing peeks. The ten predictions
    covering a timestep are combined by the law of total variance rather than by
    averaging sigmas: `E[var] + E[mu^2] - (E[mu])^2`, which is the variance of the
    mixture the ten of them form.
    """
    steps = values.shape[0]
    # 32.1. The exogenous columns ride alongside the telemetry into the model and
    # never into the target: `mu` and `sigma` are read from channel 0 below,
    # which is the only channel the head predicts.
    series = values if exog is None else np.concatenate(
        [values, np.asarray(exog, dtype=values.dtype)], axis=1)
    mu_out = np.zeros(steps, dtype=np.float64)
    var_out = np.ones(steps, dtype=np.float64)
    for start in range(0, steps, chunk):
        stop = min(start + chunk, steps)
        front = max(0, window - start)
        take = series[max(0, start - window):stop]
        if front:
            take = np.concatenate([np.repeat(take[:1], front, axis=0), take])
        with torch.no_grad():
            y = model(torch.from_numpy(take[None].astype(np.float32)),
                      last_only=False)[0].numpy()
        mus = y[..., 0]                                   # (T, n_pred, C)
        var = np.exp(np.clip(y[..., 1], -10.0, 10.0))
        m = aggregate_predictions(mus, "mean")[:, 0]
        second = aggregate_predictions(var + mus ** 2, "mean")[:, 0]
        take_n = stop - start
        mu_out[start:stop] = m[window:window + take_n]
        var_out[start:stop] = np.maximum(second - m ** 2, 1e-12)[window:window + take_n]
    return mu_out, np.sqrt(var_out)


def build_zscore(cid, train, test):
    """Arm H. Fit on `train`, score `test` with z = |x - mu| / sigma.

    The threshold is D25's rule unchanged -- the 99.9th percentile of the
    smoothed statistic over the mission's own anomaly-masked fitting window,
    which for SMAP/MSL is the whole `train` split. **The nominal rate it produces
    is reported, never targeted** (`docs/HARNESS.md` section 1, and 31.2).
    """
    hyper = Hyper(**ZSCORE_HYPER)
    v_tr = train[:, :1].astype(np.float32)
    v_te = test[:, :1].astype(np.float32)
    weights, report = fit_model(v_tr, np.ones(len(v_tr), dtype=bool), hyper, fold=0)
    model = _to_torch(weights, hyper, 1, 0)
    span = telemanom.Config().smoothing_window                      # 105, unchanged

    def statistic(values):
        mu, sigma = predictive_moments(model, values, hyper.window)
        z = np.abs(values[:, 0].astype(np.float64) - mu) / sigma
        return telemanom.ewma(z.reshape(-1, 1), span).ravel().astype(np.float64), sigma

    s_tr, sigma_tr = statistic(v_tr)
    s_te, sigma_te = statistic(v_te)
    warm = hyper.window + telemanom.Config().error_window
    finite = s_tr[np.isfinite(s_tr)][warm:]
    threshold = float(np.quantile(finite, 0.999)) if finite.size else float("inf")
    cv = float(np.std(sigma_te) / np.mean(sigma_te)) if np.mean(sigma_te) > 0 else 0.0
    return {"e_s": s_te, "as_source": s_te / max(threshold, 1e-12),
            "values": v_te[:, 0].astype(np.float64),
            "threshold": threshold, "sigma_cv": cv,
            "sigma_median": float(np.median(sigma_te)),
            "nll": float(report.best_validation_mse), "objective": report.objective,
            "epochs": int(report.epochs_run), "best_epoch": int(report.best_epoch),
            "nll_improved": bool(report.best_epoch > 0),
            "warm": {"prop": warm, "pub": warm},
            "cfg": {"prop": proportional_config(len(test)), "pub": published_config()}}


# ===========================================================================
#  Work item 9.17 stage A: label-free forecaster selection (docs/MODELS.md 34)
# ===========================================================================

#: 34.2. Thirty-two configurations, varying the forecaster and nothing else.
GRID_CELLS = ("lstm", "gru")
GRID_WINDOWS = (50, 100, 150, 250)
GRID_HIDDEN = ((40, 40), (80, 80))
GRID_EPOCHS = (35, 70)

#: 34.4. Positions are pinned to the LONGEST lookback in the grid so every
#: configuration is scored on identical targets. This is the whole reason the
#: comparison is meaningful; without it each model would choose its own
#: validation set and the MSEs would not be commensurable.
GRID_PIN = max(GRID_WINDOWS)
GRID_HOLDOUT = 0.20                      # 34.4, the chronological tail


def grid_configs():
    """The 32, in a fixed order so the artifact's rows are stable."""
    out = []
    for cell in GRID_CELLS:
        for window in GRID_WINDOWS:
            for hidden in GRID_HIDDEN:
                for epochs in GRID_EPOCHS:
                    out.append(dict(cell=cell, window=window, hidden=hidden,
                                    max_epochs=epochs))
    return out


def grid_subsample(channels, per_craft=8, seed=0):
    """34.3, written before it was run and reproduced from the seed alone.

    No label and no result is consulted: the metric this feeds is label-free
    forecast error, so the draw cannot leak anything a scored arm would.
    """
    chosen = []
    for craft in ("SMAP", "MSL"):
        pool = [c for c in channels if c["spacecraft"] == craft]
        rng = np.random.default_rng(seed)
        take = rng.choice(len(pool), size=min(per_craft, len(pool)), replace=False)
        chosen.extend(pool[int(i)] for i in sorted(take))
    return chosen


def grid_validation_mse(cid, train, cfg):
    """One configuration on one channel: held-out ten-step-ahead squared error.

    34.4. Fitted on the first 80% of the training split and scored on the last
    20%, at positions valid under `GRID_PIN` so every configuration in the grid
    predicts the SAME targets from the same timestamps. Returns the channel's
    mean squared error and the count of positions behind it.

    **No label is read and no anomaly span is consulted.** This is the only
    number 34.1 permits to decide the winner.
    """
    hyper = Hyper(cell=cfg["cell"], window=cfg["window"], hidden=cfg["hidden"],
                  max_epochs=cfg["max_epochs"], **PUBLISHED_HYPER)
    v = train[:, :1].astype(np.float32)               # commands withheld (D60)
    n = len(v)
    cut = int(n * (1.0 - GRID_HOLDOUT))
    fit_on = v[:cut]
    usable = np.ones(len(fit_on), dtype=bool)

    # 34.6's +593 is only meaningful if the fits are BANKED. `fit_model` is
    # `lstm.train`, which does not persist -- the cache lives in
    # `ForecastDetector.fit` -- and that is 31.8's defect, which cost Arm H its
    # reproducibility. Its own namespace, so a grid fit can never be served to a
    # detector or vice versa.
    key = ("wi917-grid-a", hyper.as_dict_key(), cid, GRID_HOLDOUT,
           fit_on.shape, D._sample_digest(fit_on))
    digest = D._digest(key)
    cached = D._load_weights(digest)
    if cached is None:
        weights, rep = fit_model(fit_on, usable, hyper, fold=0)
        report = rep.as_dict()
        D._save_weights(digest, weights, report)
    else:
        weights, report = cached
    model = _to_torch(weights, hyper, 1, 0)

    lo, hi = cut + GRID_PIN, n - N_PREDICTIONS
    if hi <= lo:
        raise ReferenceError(f"held-out window too short: [{lo}, {hi})")
    starts = np.arange(lo, hi)
    # Targets: the next ten steps after each t. Identical for every config.
    targets = np.stack([v[t + 1:t + 1 + N_PREDICTIONS, 0] for t in starts])
    win = cfg["window"]
    hist = np.stack([v[t + 1 - win:t + 1, 0] for t in starts])[:, :, None]
    with torch.no_grad():
        # `last_only=True` already drops the sequence axis, so the output is
        # (batch, n_predictions, n_channels). Indexing [:, -1] here took the
        # LAST PREDICTION instead of the last timestep and collapsed ten
        # horizons into one; the smoke caught it before any grid ran.
        out = model(torch.from_numpy(hist.astype(np.float32)))
    pred = np.asarray(out)[:, :, 0]
    mse = float(np.mean((pred - targets) ** 2))
    n_par = int(sum(a.size for _, a in weights.arrays()))
    D.clear_caches()
    return {"mse": mse, "positions": int(len(starts)), "parameters": n_par,
            "epochs_run": int(report.get("epochs_run", 0)),
            "best_epoch": int(report.get("best_epoch", 0)),
            "stalled": bool(report.get("best_epoch", 0) == 0
                            and report.get("epochs_run", 0) > 1)}


def grid_select(rows):
    """34.4's tie-break, implemented exactly as written and in that order.

    Every clause is a FLIGHT property -- size, then lookback, then cell -- so a
    tie is broken by what is cheaper to fly and never by what scores better on
    anything downstream. The 1% band exists because choosing on the fourth
    decimal of a validation MSE is choosing on noise.
    """
    live = [r for r in rows if not r["disqualified"] and r["primary"] is not None]
    if not live:
        return None, []
    best = min(r["primary"] for r in live)
    tied = [r for r in live if r["primary"] <= best * 1.01]
    order = sorted(tied, key=lambda r: (r["parameters"], r["window"],
                                        0 if r["cell"] == "gru" else 1))
    return order[0], tied


def _fit_grid(job):
    """One (channel, configuration) pair, in its own process."""
    cid, train, cfg, tag = job
    import sentinel_models.lstm as L
    import resource
    L.THREADS = 4
    started = time.time()
    try:
        out, err = grid_validation_mse(cid, train, cfg), None
    except Exception as exc:
        out, err = None, f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / 1048576.0 if sys.platform == "darwin" else peak / 1024.0
    return cid, tag, out, err, time.time() - started, peak_mb


def build_zscore_cmd(cid, train, test):
    """Arm C2: `gru-zscore` with the command columns as exogenous inputs.

    32.1 and D59. 31.8's H4 stop fired because `sigma` collapsed to a per-channel
    constant, and 31.9 named the reason without registering it: a single
    univariate channel gives the likelihood no reason to vary `sigma` **with
    state**. Commands are state. Everything else is Arm H unchanged -- the same
    GRU, the same Gaussian head, the same NLL, D25's threshold rule.

    **It also repairs 31.8's recorded defect.** Arm H fitted through `lstm.train`
    directly while the cache lives in `ForecastDetector.fit`, so its weights were
    never persisted and the arm was reproducible only from its seed. The key
    below is `ForecastDetector.fit`'s own construction (`detectors.py:328-331`),
    so a re-run hits the store rather than refitting.
    """
    hyper = Hyper(**ZSCORE_HYPER)
    v_tr = train[:, :1].astype(np.float32)
    v_te = test[:, :1].astype(np.float32)
    c_tr = np.ascontiguousarray(train[:, 1:].astype(np.float32))
    c_te = np.ascontiguousarray(test[:, 1:].astype(np.float32))
    usable = np.ones(len(v_tr), dtype=bool)

    key = (hyper.as_dict_key(), (cid,), 0, hyper.window, v_tr.shape,
           D._sample_digest(v_tr), D._digest(usable[::997]),
           D._sample_digest(c_tr))
    digest = D._digest(key)
    cached = D._load_weights(digest)
    if cached is None:
        weights, rep = fit_model(v_tr, usable, hyper, fold=0, impulses=c_tr)
        report = rep.as_dict()
        D._save_weights(digest, weights, report)               # 31.8's defect, fixed
    else:
        weights, report = cached

    n_exog = 2 * c_tr.shape[1]                     # impulse + decay, 48 SMAP / 108 MSL
    model = _to_torch(weights, hyper, 1, n_exog)
    span = telemanom.Config().smoothing_window                      # 105, unchanged

    def statistic(values, commands):
        mu, sigma = predictive_moments(model, values, hyper.window,
                                       exog=_exog(commands))
        z = np.abs(values[:, 0].astype(np.float64) - mu) / sigma
        return telemanom.ewma(z.reshape(-1, 1), span).ravel().astype(np.float64), sigma

    s_tr, _ = statistic(v_tr, c_tr)
    s_te, sigma_te = statistic(v_te, c_te)
    warm = hyper.window + telemanom.Config().error_window
    finite = s_tr[np.isfinite(s_tr)][warm:]
    threshold = float(np.quantile(finite, 0.999)) if finite.size else float("inf")
    cv = float(np.std(sigma_te) / np.mean(sigma_te)) if np.mean(sigma_te) > 0 else 0.0
    return {"e_s": s_te, "as_source": s_te / max(threshold, 1e-12),
            "values": v_te[:, 0].astype(np.float64),
            "threshold": threshold, "sigma_cv": cv,
            "sigma_median": float(np.median(sigma_te)),
            "n_exogenous": n_exog,
            "nll": float(report.get("best_validation_mse", float("nan"))),
            "epochs": int(report.get("epochs_run", 0)),
            "best_epoch": int(report.get("best_epoch", 0)),
            "nll_improved": bool(report.get("best_epoch", 0) > 0),
            "warm": {"prop": warm, "pub": warm},
            "cfg": {"prop": proportional_config(len(test)), "pub": published_config()}}


def proportional_config(n: int):
    """D47, unchanged from `scripts/smap_stage2.py:65-78`. The `prop` regime."""
    window = max(1, int(telemanom.SMOOTHING_PERC * n))
    return telemanom.Config(error_window=window,
                            stride=min(telemanom.ERROR_WINDOW_BATCH,
                                       max(1, window // 2)))


def published_config():
    """telemanom's own absolute window. The `pub` regime (errors.py:40-42)."""
    return telemanom.Config(error_window=BATCH_SIZE * WINDOW_SIZE, stride=BATCH_SIZE)


def build(cid: str, cell: str, train, test):
    """Fit on `train`, return the smoothed error series and the as-source score.

    `_smoothed_errors` already computes `|actual - forecast|` and smooths at span
    105, which is telemanom's own `smoothing_window` (errors.py:51-52), so the
    port starts from the same `e_s` the source would. Read directly rather than
    re-derived: no source file changes and no ESA-ADB figure can move (27.1).
    """
    det = registry.build(f"{cell}-telemanom")
    det.config = proportional_config(len(test))
    v_tr = train[:, :1].astype(np.float32)
    v_te = test[:, :1].astype(np.float32)
    det.fit(v_tr, np.ones(len(v_tr), dtype=bool), ctx_for(cid, len(v_tr)))
    s_ctx = ctx_for(cid, len(v_te))
    as_source, _ = reduce_scores(det.score(v_te, None, s_ctx), len(v_te))
    e_s = np.asarray(det._smoothed_errors(v_te, s_ctx), dtype=np.float64)[:, 0]
    warm_prop = int(det.warmup_steps)
    det.config = published_config()
    warm_pub = int(det.warmup_steps)
    D.clear_caches()
    return {"e_s": e_s, "as_source": np.asarray(as_source, dtype=np.float64),
            "values": v_te[:, 0].astype(np.float64),
            "warm": {"prop": warm_prop, "pub": warm_pub},
            "cfg": {"prop": proportional_config(len(test)), "pub": published_config()}}


def score_arm(alarm, scorable, spans):
    """Both accountings, side by side (26.29.2, 27.3).

    ours       range-hits / ranges                 eventwise.py:96,108
    telemanom  matched events / (matched + unmatched ranges)
                                                   detector.py:117-136, :167-173
    """
    fired = np.asarray(alarm, dtype=bool) & np.asarray(scorable, dtype=bool)
    ranges = mask_to_ranges(fired)
    caught, hits = set(), 0
    for lo, hi in ranges:
        matched = [k for k, (a, b) in enumerate(spans)
                   if not (b + 1 <= lo or hi <= a)]
        if matched:
            hits += 1
            caught.update(matched)
    return {"caught": sorted(caught), "n_ranges": len(ranges), "range_hits": hits,
            "tp": len(caught), "fp": len(ranges) - hits, "fired": fired}


def grid_stage_a(args, client, cfg, ledger, budget, man) -> int:
    """Work item 9.17 stage A. Produces a RANKING and nothing else (34.1).

    No label is read, no anomaly span is loaded, and no scored arm is touched.
    If this function ever computes a recall figure, 34.1 voids the arm.
    """
    chosen = grid_subsample(man["channels"])
    configs = grid_configs()
    if args.grid_smoke:
        # 34.6's corners: the cheapest and the dearest of each cell, so the
        # projection is not anchored on one arbitrary point in the middle.
        configs = [dict(cell=c, window=w, hidden=h, max_epochs=e)
                   for c, w, h, e in (("lstm", 50, (40, 40), 35),
                                      ("lstm", 250, (80, 80), 70),
                                      ("gru", 50, (40, 40), 35),
                                      ("gru", 250, (80, 80), 70))]
        chosen = chosen[:1] + chosen[-1:]                   # one SMAP, one MSL
    print(f"  stage A: {len(configs)} configurations x {len(chosen)} channels "
          f"= {len(configs) * len(chosen)} fits")
    print(f"  subsample (seeded rule, 34.3): "
          f"{', '.join(c['channel_id'] for c in chosen)}")

    store = D.WEIGHT_STORE
    before = sum(1 for p in store.iterdir() if p.suffix == ".npz")
    jobs = []
    for ch in chosen:
        keys = {o["split"]: o["key"] for o in ch["objects"]}
        train = np.load(io.BytesIO(fetch(client, cfg.bucket, keys["train"])))
        for i, conf in enumerate(configs):
            jobs.append((ch["channel_id"], train, conf, i))

    if args.workers > 1:
        free = free_memory_gib()
        print(f"  free memory {free:.1f} GiB (gate: 4.0)")
        if free < 4.0:
            print("  ABORT: below the pre-registered 4 GiB gate; not starting the pool.")
            return 7
    started, done = time.time(), []
    if args.workers > 1:
        batch = 4 * args.workers
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for lo in range(0, len(jobs), batch):
                done.extend(pool.map(_fit_grid, jobs[lo:lo + batch]))
                for k in range(lo, min(lo + batch, len(jobs))):
                    jobs[k] = None                      # release the arrays
                print(f"    {min(lo + batch, len(jobs))}/{len(jobs)} fits")
    else:
        done = [_fit_grid(j) for j in jobs]
    elapsed = time.time() - started

    per_conf = collections.defaultdict(list)
    stalls, secs, peaks = collections.defaultdict(list), [], []
    for cid, tag, out, err, sec, peak in done:
        secs.append(sec); peaks.append(peak)
        if err is not None:
            stalls[tag].append({"channel": cid, "error": err})
        else:
            per_conf[tag].append({"channel": cid, **out})

    rows = []
    for i, conf in enumerate(configs):
        got = per_conf.get(i, [])
        n_stall = len(stalls.get(i, []))
        # 34.3: more than half the subsample stalled disqualifies the row.
        dq = n_stall > len(chosen) // 2
        rows.append({
            "index": i, **{k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in conf.items()},
            "channels_fitted": len(got), "stalls": n_stall, "disqualified": dq,
            "parameters": got[0]["parameters"] if got else None,
            # 34.4: primary is the MEAN OF PER-CHANNEL MEANS; pooled is beside it.
            "primary": (float(np.mean([g["mse"] for g in got])) if got and not dq
                        else None),
            "pooled": (float(np.average([g["mse"] for g in got],
                                        weights=[g["positions"] for g in got]))
                       if got and not dq else None),
            "per_channel": got,
        })
    winner, tied = grid_select(rows)

    print()
    print("  rank  cell  l_s  hidden  epochs  params   primary MSE   stalls")
    for r in sorted([r for r in rows if r["primary"] is not None],
                    key=lambda r: r["primary"])[:10]:
        print(f"        {r['cell']:4s} {r['window']:4d}  {r['hidden'][0]:3d}   "
              f"{r['max_epochs']:4d}  {r['parameters']:7d}  {r['primary']:.6e}  "
              f"{r['stalls']:3d}")
    if winner:
        print()
        print(f"  WINNER (34.4): cell={winner['cell']} window={winner['window']} "
              f"hidden={winner['hidden']} epochs={winner['max_epochs']} "
              f"params={winner['parameters']}  primary={winner['primary']:.6e}")
        print(f"  tied within 1%: {len(tied)} configuration(s)")

    after = sum(1 for p in store.iterdir() if p.suffix == ".npz")
    print(f"  weight store {before} -> {after} (+{after - before})")
    if secs:
        med, mean = float(np.median(secs)), float(np.mean(secs))
        full = len(grid_configs()) * len(grid_subsample(man["channels"]))
        print(f"  fits: n={len(secs)} mean {mean:.1f}s median {med:.1f}s "
              f"min {min(secs):.1f}s max {max(secs):.1f}s  spread "
              f"{max(secs)/max(1e-9, min(secs)):.0f}x  |  wall clock {elapsed/60:.1f} min")
        # A TOTAL is n * MEAN, never n * median. On this grid the corners span
        # 40x, so the median is a statistic about a typical fit and says nothing
        # about the sum -- which is 28.5's error in a different costume, where a
        # per-unit figure was multiplied by a count and the total was wrong.
        print(f"  -> the full {full} fits project to {full*mean/3600:.1f} h serial "
              f"from the MEAN, {full*mean/3600/max(1,args.workers):.1f} h at "
              f"{args.workers} workers")
        print(f"     (the median would have said {full*med/3600:.1f} h serial, and a "
              f"total is not n x median)")
        print(f"  peak RSS per worker: max {max(peaks):.0f} MB median "
              f"{float(np.median(peaks)):.0f} MB")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = Path("runs/smap-msl/_forensics") / f"{stamp}-wi917-grid-a.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "generated_utc": stamp, "pre_registration": "docs/MODELS.md 34",
        "smoke": bool(args.grid_smoke),
        "label_free": ("34.1: the winner is chosen ONLY on held-out nominal "
                       "validation error. No label, recall or alarm rate enters."),
        "subsample": [c["channel_id"] for c in chosen],
        "subsample_rule": "34.3: default_rng(0).choice, 8 per spacecraft, manifest order",
        "holdout_fraction": GRID_HOLDOUT, "position_pin": GRID_PIN,
        "n_predictions": N_PREDICTIONS,
        "tie_break": "34.4: within 1% relative, then fewer parameters, then shorter "
                     "lookback, then gru",
        "configurations": rows,
        "winner": (None if winner is None else
                   {k: winner[k] for k in ("cell", "window", "hidden", "max_epochs",
                                           "parameters", "primary", "pooled")}),
        "tied_within_1pc": len(tied),
        "stalls": {str(k): v for k, v in stalls.items()},
        "timing": {"wall_clock_min": elapsed / 60.0, "workers": args.workers,
                   "median_fit_s": float(np.median(secs)) if secs else None,
                   "mean_fit_s": float(np.mean(secs)) if secs else None,
                   "fit_seconds": [round(x, 2) for x in secs],
                   "peak_rss_mb": [round(x, 1) for x in peaks],
                   "peak_rss_mb_max": max(peaks) if peaks else None},
        "weight_store": {"before": before, "after": after},
    }
    out["operations"] = budget.as_dict()
    path.write_text(json.dumps(out, indent=1))          # ARTIFACT BEFORE LEDGER
    print(f"\n  artifact {path}")
    try:
        ops.commit(client, cfg.bucket, ledger, budget)
    except Exception as exc:
        print(f"  (!) LEDGER COMMIT FAILED: {exc}")
        print(f"      {budget.class_a} Class A and {budget.class_b} Class B WERE "
              f"spent and are NOT recorded.")
    print(budget.report())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visibility", required=True, help="the stage 1 artifact")
    ap.add_argument("--stage4", required=True, help="the stage 4 artifact")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N channels")
    ap.add_argument("--acknowledge-tripwire", action="store_true")
    ap.add_argument("--published", action="store_true",
                    help="work item 9.11: add Arm T, the published training "
                         "configuration. THIS REFITS -- see docs/MODELS.md 28.5")
    ap.add_argument("--zscore", action="store_true",
                    help="work item 9.14: add gru-zscore. THIS REFITS (31.6)")
    ap.add_argument("--workers", type=int, default=1, metavar="N",
                    help="parallel published fits; 28.5's measured plan is 4 at "
                         "2 torch threads, ~32 min and ~2.8 GB")
    ap.add_argument("--only", default="", metavar="A,B",
                    help="score only these arms; the timing smoke uses it")
    ap.add_argument("--grid-a", action="store_true",
                    help="work item 9.17 stage A: the 32-configuration grid on the "
                         "16-channel subsample. THIS REFITS -- see docs/MODELS.md 34.6")
    ap.add_argument("--grid-smoke", action="store_true",
                    help="34.6's smoke: 4 corner configurations x 2 channels")
    args = ap.parse_args(argv)

    vis = json.loads(Path(args.visibility).read_text())
    st4 = json.loads(Path(args.stage4).read_text())
    base_rate = float(st4["base_rate"])          # read, never transcribed (27.3)
    in_range = {(v["channel"], v["start"], v["end"]): v["in_range"]
                for v in vis["sequences"]}
    print(f"  stage 4 base rate {base_rate*100:.4f}%  (from {Path(args.stage4).name})")

    cfg = C.load_r2_config()
    client, budget = ops.connect(cfg, acknowledge_tripwire=args.acknowledge_tripwire)
    ledger = ops.load(client, cfg.bucket)
    budget.prior_class_a, budget.prior_class_b = ledger.month_class_a, ledger.month_class_b
    man = json.loads(fetch(client, cfg.bucket, MANIFEST_KEY))
    labels = list(csv.DictReader(io.StringIO(
        fetch(client, cfg.bucket, man["labels_key"]).decode())))

    seqs, seen = {}, set()
    for r in labels:
        cid = r["chan_id"]
        if cid in seen:                      # P-2 counted once, 26.8
            continue
        seen.add(cid)
        classes = [x.strip() for x in r["class"].strip("[]").split(",") if x.strip()]
        seqs[cid] = list(zip(ast.literal_eval(r["anomaly_sequences"]), classes))

    if args.grid_a or args.grid_smoke:
        return grid_stage_a(args, client, cfg, ledger, budget, man)

    channels = man["channels"][: args.limit] if args.limit else man["channels"]
    fit_seconds: list[float] = []
    fit_peaks: list[float] = []
    pending: list[tuple] = []
    store = D.WEIGHT_STORE
    before = sum(1 for p in store.iterdir() if p.suffix == ".npz")

    per_channel, stalls, geometry = {}, {}, []
    print(f"  loading {len(channels)} channels x 2 cells ...")
    for i, ch in enumerate(channels, 1):
        cid, sc = ch["channel_id"], ch["spacecraft"]
        keys = {o["split"]: o["key"] for o in ch["objects"]}
        train = np.load(io.BytesIO(fetch(client, cfg.bucket, keys["train"])))
        test = np.load(io.BytesIO(fetch(client, cfg.bucket, keys["test"])))
        built, failed = {}, {}
        # Which cells this run has to build, derived from the arms it will
        # score. This was a hard-coded map of four arm names; arm G needed a
        # fifth and a map that has to be edited alongside `ARM_TABLE` is a map
        # that goes stale, which is D14's shape of failure. `--only ""` and the
        # four names it already knew resolve identically, so no prior run's
        # cells change.
        asked = {a for a in args.only.split(",") if a}
        if asked:
            cells = tuple(dict.fromkeys(c for n, c, *_ in ARM_TABLE if n in asked))
        elif args.published:
            cells = ("lstm", "gru", "published", "rung")
        elif args.zscore:
            cells = ("zscore",)
        else:
            cells = ("lstm", "gru")
        if args.zscore and "zscore" not in cells:
            cells = tuple(cells) + ("zscore",)
        for cell in [c for c in cells if c not in DEFERRED_CELLS]:
            try:
                built[cell] = build(cid, cell, train, test)
            except Exception as exc:         # D17's guard refuses a channel outright
                failed[cell] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        # A stall is recorded per cell, never silently dropped, and the two
        # populations nest: the LSTM's channels are a subset of the GRU's (26.30.3).
        if failed:
            stalls[cid] = failed
            print(f"    {cid}: STALL -- " + "; ".join(f"{k} {v}" for k, v in failed.items()))
        deferred = [c for c in cells if c in DEFERRED_CELLS]
        if deferred:
            pending.append((cid, train, test, deferred))
        if not built and not deferred:
            continue
        grew = sum(1 for p in store.iterdir() if p.suffix == ".npz")
        # Arm T is the first arm in this study that is SUPPOSED to fit, at a
        # pre-registered +81 (28.5). Everything else must not move the store.
        if grew != before and not args.published:
            print(f"  ABORT: weight store {before} -> {grew} at {cid}; a fit happened.")
            return 6
        per_channel[cid] = {"spacecraft": sc, "n": len(test), "built": built,
                            "train_rows": len(train),
                            "spans": [(int(a), int(b)) for (a, b), _ in seqs.get(cid, [])],
                            "classes": [c for _, c in seqs.get(cid, [])]}
        if not built:
            continue
        any_cell = next(iter(built.values()))
        for regime in ("prop", "pub"):
            c = any_cell["cfg"][regime]
            geometry.append({"channel": cid, "regime": regime,
                             "error_window": c.error_window, "stride": c.stride,
                             "error_buffer": c.error_buffer,
                             "smoothing_window": c.smoothing_window,
                             "windows_per_index": c.error_window / max(1, c.stride) + 1})
        if i % 10 == 0:
            print(f"    {i}/{len(channels)}")

    # -- Arm T's fits, fanned out. Zero extra operations: the bundle is loaded.
    if pending:
        # 31.6's mitigations, carrying 28.7's OOM as a design constraint rather
        # than as a resolution to be careful.
        if args.workers > 1:
            free = free_memory_gib()
            print(f"  free memory {free:.1f} GiB (gate: 4.0)")
            if free < 4.0:
                print("  ABORT: below the pre-registered 4 GiB gate; not starting the pool.")
                return 7
        print(f"  fitting {len(pending)} models on {args.workers} worker(s) ...")
        started, done = time.time(), []
        if args.workers > 1:
            # The parent holds at most one batch of raw arrays at a time, which
            # is the term 28.5's arithmetic left out.
            batch = 4 * args.workers
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for lo in range(0, len(pending), batch):
                    slice_ = pending[lo:lo + batch]
                    done.extend(pool.map(_fit_published, slice_))
                    for k in range(lo, min(lo + batch, len(pending))):
                        pending[k] = None          # release the arrays
                    print(f"    {min(lo + batch, len(pending))}/{len(pending)} fitted")
        else:
            done = [_fit_published(job) for job in pending]
        for cid, out, err, seconds, peak_mb in done:
            fit_seconds.append(seconds)
            fit_peaks.append(peak_mb)
            for cell, message in (err or {}).items():
                stalls.setdefault(cid, {})[cell] = message
                print(f"    {cid}: STALL -- {cell} {message}")
            for cell, built_cell in out.items():
                per_channel[cid]["built"][cell] = built_cell
        per_channel = {c: v for c, v in per_channel.items() if v["built"]}
        print(f"  fits done in {(time.time() - started)/60:.1f} min wall clock")

    after = sum(1 for p in store.iterdir() if p.suffix == ".npz")
    refitting = {a for a in args.only.split(",") if a} & {"T", "C1", "C2", "C2M"}
    if refitting:
        expected = f"32.5 expects +81 per refitting arm ({sorted(refitting)})"
    elif args.published:
        expected = "28.5 expects +78 (3 banked by the smoke)"
    else:
        expected = "27.6 expects +0"
    print(f"  weight store {before} -> {after} (+{after - before}; {expected})")
    if fit_seconds:
        med = float(np.median(fit_seconds))
        print(f"  fits: n={len(fit_seconds)} median {med:.1f}s "
              f"min {min(fit_seconds):.1f}s max {max(fit_seconds):.1f}s "
              f"-> 81 channels projects to {81 * med / 60:.0f} min serial")
        if fit_peaks:
            # 32.5. Measured, per worker process, INCLUDING its torch import.
            pk = max(fit_peaks)
            print(f"  peak RSS per worker: max {pk:.0f} MB "
                  f"median {float(np.median(fit_peaks)):.0f} MB "
                  f"-> {args.workers} workers project to {args.workers * pk / 1024:.1f} GiB "
                  f"against {free_memory_gib():.1f} GiB free now")
    return report(args, per_channel, stalls, geometry, in_range, base_rate,
                  before, after, client, cfg, ledger, budget)


ARM_TABLE = [
    ("A0", "lstm", "ladder", Mech(), "prop", "gate + ladder base: 1a+1b, the faithful arm"),
    ("A1", "lstm", "source", None, "prop", "gate: src/sentinel_models/telemanom.py as-is"),
    ("A2", "lstm", "ladder", Mech(clip=False), "prop", "gate: 1c-ii, NOT A REPRODUCTION"),
    ("A3", "gru", "swept", None, "prop", "gate: stage 4's matched arm"),
    ("A4", "gru", "ladder", Mech(clip=False), "prop", "1c-ii on the GRU"),
    ("A5", "lstm", "ladder", Mech(clip=False), "pub", "1c-ii at the published window"),
    ("L1", "lstm", "ladder", Mech(magnitude=True), "prop",
     "+ magnitude conjunct (errors.py:342-343)"),
    ("L2", "lstm", "ladder", Mech(magnitude=True, bailout=True), "prop",
     "+ whole-window bail-out (errors.py:337-340)"),
    ("L3", "lstm", "ladder", Mech(magnitude=True, bailout=True, guards=True), "prop",
     "+ the two find_epsilon guards (errors.py:314-315)"),
    ("L4", "lstm", "ladder",
     Mech(magnitude=True, bailout=True, guards=True, inverse=True), "prop",
     "+ the inverse pass (errors.py:132-148)"),
    ("L5", "lstm", "ladder",
     Mech(magnitude=True, bailout=True, guards=True, inverse=True,
          published_window=True), "pub", "+ the published window (errors.py:84-93)"),
    ("F", "lstm", "port", None, "pub", "the complete port: the reference ceiling"),
    ("FG", "gru", "port", None, "pub", "the complete port on the GRU: the 38 baseline"),
    ("T", "published", "port", None, "pub",
     "work item 9.11: the published training configuration under Arm F's stack"),
    ("R", "rung", "port", None, "pub",
     "work item 9.12: Arm T plus the residual rung (errors.py:48-64)"),
    ("G", "rung", "port", Mech(dimensionless=True), "pub",
     "work item 9.13: Arm R with the two absolute filters made dimensionless"),
    ("C1", "published_nc", "port", None, "pub",
     "work item 9.15: Arm T with the commands WITHHELD -- D49's counterfactual"),
    ("C2", "zscore_cmd", "source", None, "prop",
     "work item 9.15: gru-zscore WITH the command columns, at D25's threshold"),
    ("C2M", "zscore_cmd", "swept", None, "prop",
     "gru-zscore+cmd swept to stage 4's rate -- a COMPARISON DEVICE, not a calibration"),
    ("C1S", "published_nc", "port-swept", None, "pub",
     "work item 9.16: C1 swept to stage 4's rate -- a LABELLED DEVIATION (28.6), "
     "and the comparison K2 was left owed"),
    ("H", "zscore", "source", None, "prop",
     "work item 9.14: gru-zscore at D25's label-free threshold"),
    ("HM", "zscore", "swept", None, "prop",
     "gru-zscore swept to stage 4's rate -- a COMPARISON DEVICE, not a calibration"),
]


def report(args, per_channel, stalls, geometry, in_range, base_rate,
           before, after, client, cfg, ledger, budget) -> int:
    p = telemanom.PRUNING_P
    cells = tuple(c for c in ("lstm", "gru", "published", "rung", "zscore",
                              "published_nc", "zscore_cmd")
                  if any(c in v["built"] for v in per_channel.values()))

    # -- populations, per cell, each naming the stall set that fixed it (26.30.3)
    events, pops, nominal, scorable = {}, {}, {}, {}
    for cell in cells:
        rows, chans = [], [c for c, v in per_channel.items() if cell in v["built"]]
        for cid in chans:
            v = per_channel[cid]
            for (a, b), cls in zip(v["spans"], v["classes"]):
                rows.append({"channel": cid, "spacecraft": v["spacecraft"],
                             "start": a, "end": b, "class": cls,
                             "in_range": in_range.get((cid, a, b))})
        events[cell] = rows
        pops[cell] = {
            "all": list(range(len(rows))),
            "in_range_contextual": [k for k, e in enumerate(rows)
                                    if e["class"] == "contextual" and e["in_range"]],
            "out_of_range_contextual": [k for k, e in enumerate(rows)
                                        if e["class"] == "contextual"
                                        and e["in_range"] is False],
            "in_range_point": [k for k, e in enumerate(rows)
                               if e["class"] == "point" and e["in_range"]],
        }
        scorable[cell], nominal[cell] = {}, {}
        for regime in ("prop", "pub"):
            sc, nm = {}, 0
            for cid in chans:
                v = per_channel[cid]
                n = v["n"]
                warm = v["built"][cell]["warm"][regime]
                s = np.zeros(n, dtype=bool)
                s[min(warm, n):] = True          # no history bridges train->test, 26.13
                anom = np.zeros(n, dtype=bool)
                for a, b in v["spans"]:
                    anom[a:b + 1] = True
                sc[cid] = s
                nm += int((s & ~anom).sum())
            scorable[cell][regime], nominal[cell][regime] = sc, nm
        print(f"  {cell}: {len(chans)} channels, {len(rows)} sequences, "
              f"{len(pops[cell]['in_range_contextual'])} in-range contextual")

    # -- every arm, one pass -------------------------------------------------
    wanted = {a for a in args.only.split(",") if a} or None
    results, per_ch_rows, diagnostics, sweeps = {}, [], {}, {}
    for name, cell, kind, mech, regime, note in ARM_TABLE:
        if wanted is not None and name not in wanted:
            continue
        chans = [c for c, v in per_channel.items() if cell in v["built"]]
        if not chans:
            continue
        alarms, flagged = {}, 0
        for cid in chans:
            v, b = per_channel[cid], per_channel[cid]["built"][cell]
            if kind == "ladder":
                mask, _ = run_ladder(b["e_s"], b["values"], b["cfg"][regime], p, mech)
            elif kind in ("port", "port-swept"):
                reset_guards()
                dim = bool(mech is not None and mech.dimensionless)
                if b.get("supervised"):
                    mask, _, _, eps_at = run_port(
                        b["e_s"], b["y_sup"], telemanom.ERROR_BUFFER, p,
                        presliced=True, alarm_len=b["n_full"], dimensionless=dim)
                else:
                    mask, _, _, eps_at = run_port(
                        b["e_s"], b["values"], telemanom.ERROR_BUFFER, p,
                        # 28.1 T-g. Every arm built on the PUBLISHED training
                        # configuration carries the published target length.
                        # C1 was omitted here until 2026-09-09 and therefore
                        # differed from Arm T in two levers rather than one --
                        # the commands, which was the point, and the tail, which
                        # was not. Named by cell, so a new published arm cannot
                        # be forgotten the way C1 was.
                        tail=(N_PREDICTIONS
                              if cell in ("published", "published_nc") else 0),
                        dimensionless=dim)
                diagnostics[(name, cid)] = {
                    "guards": dict(GUARDS),
                    "max_e_s": float(np.max(b["e_s"])),
                    "mean_e_s": float(np.mean(b["e_s"])),
                    "sd_e_s": float(np.std(b["e_s"])),
                    "dim_k": DIM_K,
                    "sd_values": float(np.std(b["values"])),
                    "inter_range": float(np.subtract(
                        *np.percentile(b["values"], [95, 5]))),
                    "eps_at": eps_at, "e_s": b["e_s"],
                    "supervised": bool(b.get("supervised")),
                }
            else:                                  # "source" and "swept"
                mask = b["as_source"] >= 1.0
            alarms[cid] = np.asarray(mask, dtype=bool) & scorable[cell][regime][cid]

        if kind == "port-swept":
            # 33.3. The port has no dial, so one is threaded through and
            # LABELLED as a deviation from the source (28.6). Same selection
            # rule as the as_source sweep below, so the two are comparable:
            # among multipliers at or under the target rate, the one catching
            # most events wins.
            curve, best = [], None
            for m in PORT_GRID:
                masks, fl, ct, base = {}, 0, set(), 0
                for cid in chans:
                    v, b = per_channel[cid], per_channel[cid]["built"][cell]
                    mk, _, _, _ = run_port(
                        b["e_s"], b["values"], telemanom.ERROR_BUFFER, p,
                        tail=(N_PREDICTIONS
                              if cell in ("published", "published_nc") else 0),
                        eps_mult=float(m))
                    f = np.asarray(mk, dtype=bool) & scorable[cell][regime][cid]
                    masks[cid] = f
                    anom = np.zeros(v["n"], dtype=bool)
                    for a, bb in v["spans"]:
                        anom[a:bb + 1] = True
                    fl += int((f & ~anom).sum())
                    for k, (a, bb) in enumerate(v["spans"]):
                        if f[a:bb + 1].any():
                            ct.add(base + k)
                    base += len(v["spans"])
                rate = fl / max(1, nominal[cell][regime])
                # 33.8. The CAUGHT SET, not a count of it. A summary retained
                # at lower resolution than the question needs is what left M1
                # unadjudicable and S1/S2 unadjudicable before it (27.8); with
                # the ids kept, any population restriction is computable from
                # the artifact forever without spending another read.
                curve.append({"multiplier": float(m), "nominal_rate": rate,
                              "caught": len(ct),
                              "in_range_contextual": len(
                                  set(ct) & set(pops[cell]["in_range_contextual"])),
                              "caught_ids": sorted(ct)})
                irc = len(set(ct) & set(pops[cell]["in_range_contextual"]))
                flag = "" if rate > base_rate else "  <= target"
                print(f"    {name} mult {m:5.3f}  nominal {100*rate:7.4f}%  "
                      f"caught {len(ct):3d}  in-range contextual {irc:3d}{flag}")
                if rate <= base_rate and (best is None or len(ct) > best[2]):
                    best = (float(m), rate, len(ct), masks)
            sweeps[name] = curve
            if best is None:
                print(f"    {name}: NO multiplier reaches {100*base_rate:.4f}%; "
                      f"the arm has no matched point and none is invented.")
                mult = float(PORT_GRID[-1])
                for cid in chans:
                    b = per_channel[cid]["built"][cell]
                    mk, _, _, _ = run_port(
                        b["e_s"], b["values"], telemanom.ERROR_BUFFER, p,
                        tail=(N_PREDICTIONS
                              if cell in ("published", "published_nc") else 0),
                        eps_mult=mult)
                    alarms[cid] = (np.asarray(mk, dtype=bool)
                                   & scorable[cell][regime][cid])
            else:
                mult, alarms = best[0], best[3]
        elif kind == "swept":
            # `scripts/smap_stage2.py:265-289`, replicated literally: the arm is
            # matched on pooled nominal-step rate, and among multipliers at or
            # under the target the one catching most events wins. Any change to
            # this rule would break gate G1 by construction.
            def at(m):
                fl, ct = 0, set()
                base = 0
                for cid in chans:
                    b, v = per_channel[cid]["built"][cell], per_channel[cid]
                    f = (b["as_source"] >= 1.0 * m) & scorable[cell][regime][cid]
                    anom = np.zeros(v["n"], dtype=bool)
                    for a, bb in v["spans"]:
                        anom[a:bb + 1] = True
                    fl += int((f & ~anom).sum())
                    for k, (a, bb) in enumerate(v["spans"]):
                        if f[a:bb + 1].any():
                            ct.add(base + k)
                    base += len(v["spans"])
                return fl / max(1, nominal[cell][regime]), ct

            best = None
            for m in make_grid():
                rate, ct = at(float(m))
                if rate <= base_rate and (best is None or len(ct) > len(best[2])):
                    best = (float(m), rate, ct)
            mult = best[0] if best else float(make_grid()[-1])
            for cid in chans:
                b = per_channel[cid]["built"][cell]
                alarms[cid] = ((b["as_source"] >= 1.0 * mult)
                               & scorable[cell][regime][cid])
        else:
            mult = 1.0

        tp = fp = hits = ranges = 0
        caught_global: set[int] = set()
        offset = 0
        for cid in chans:
            v = per_channel[cid]
            s = score_arm(alarms[cid], scorable[cell][regime][cid], v["spans"])
            anom = np.zeros(v["n"], dtype=bool)
            for a, b_ in v["spans"]:
                anom[a:b_ + 1] = True
            flagged += int((s["fired"] & ~anom).sum())
            tp += s["tp"]; fp += s["fp"]; hits += s["range_hits"]; ranges += s["n_ranges"]
            caught_global.update(offset + k for k in s["caught"])
            per_ch_rows.append({"channel": cid, "spacecraft": v["spacecraft"],
                                "arm": name, "cell": cell, "regime": regime,
                                "recall": Count(s["tp"], len(v["spans"])).as_dict(),
                                "precision_ours": Count(s["range_hits"],
                                                        s["n_ranges"]).as_dict(),
                                "precision_telemanom": Count(s["tp"],
                                                             s["tp"] + s["fp"]).as_dict(),
                                "ranges": s["n_ranges"],
                                # 27.8: S1 and S2 could not be adjudicated because
                                # the alarm sets were not retained. Ranges are the
                                # set, compactly -- subset and superset checks read
                                # straight off them.
                                "alarm_ranges": mask_to_ranges(s["fired"])})
            offset += len(v["spans"])
        rate = flagged / max(1, nominal[cell][regime])
        results[name] = {
            "cell": cell, "regime": regime, "note": note, "multiplier": mult,
            "nominal_rate": rate, "nominal_steps": nominal[cell][regime],
            "recall": Count(tp, len(events[cell])).as_dict(),
            "precision_ours": Count(hits, ranges).as_dict(),
            "precision_telemanom": Count(tp, tp + fp).as_dict(),
            "ranges": ranges, "false_positives": fp,
            "counts": {k: Count(len(set(v) & caught_global), len(v)).as_dict()
                       for k, v in pops[cell].items()},
            "caught": sorted(caught_global),
        }
        r = results[name]
        print(f"  {name:3s} {cell:4s} {regime:4s}  recall {tp:3d}/{len(events[cell]):3d} "
              f"{100*tp/max(1,len(events[cell])):5.1f}%   ours {hits:3d}/{ranges:3d} "
              f"telemanom {tp:3d}/{tp+fp:3d} {100*tp/max(1,tp+fp):5.1f}%   "
              f"FP {fp:4d}   nominal {rate*100:7.4f}%")

    # -- 29.1's per-event forensics, MSL first --------------------------------
    def classify(arm, cid, a, b, ev_index, caught):
        d = diagnostics.get((arm, cid))
        if d is None:
            return None
        e_s, eps_at = d["e_s"], d["eps_at"]
        # e_s is in supervised coordinates when the rung is on; raw = j + l_s.
        base = L_S if d["supervised"] else 0
        lo, hi = max(0, a - base), min(len(e_s), b + 1 - base)
        if hi <= lo:
            return {"channel": cid, "start": a, "end": b, "class": "out-of-range",
                    "note": "the label span lies outside the supervised region"}
        span = e_s[lo:hi]
        peak = float(np.max(span))
        peak_at = int(np.argmax(span)) + lo + base
        width = b - a + 1
        wide_lo, wide_hi = max(0, lo - width), min(len(e_s), hi + width)
        wide_peak_at = int(np.argmax(e_s[wide_lo:wide_hi])) + wide_lo + base
        eps_window = eps_at[max(0, a):b + 1]
        eps = float(np.nanmedian(eps_window)) if np.isfinite(eps_window).any() else float("nan")
        ratio = peak / eps if eps and np.isfinite(eps) and eps > 0 else float("nan")
        if ev_index in caught:
            label = "caught"
        elif not (a <= wide_peak_at <= b):
            label = "offset"                    # the signal is beside the label
        elif np.isfinite(ratio) and ratio >= 1.0:
            label = "filtered"                  # it crossed and something removed it
        elif np.isfinite(ratio) and ratio >= 0.5:
            label = "threshold"                 # close, and under
        else:
            label = "no-signal"
        return {"channel": cid, "start": a, "end": b, "class": label,
                "peak": peak, "peak_at": peak_at, "wide_peak_at": wide_peak_at,
                "epsilon": eps, "ratio": ratio}

    forensics = {}
    for arm in ("T", "R"):
        if arm not in results:
            continue
        cell = dict((n, c) for n, c, *_ in ARM_TABLE)[arm]
        caught = set(results[arm]["caught"])
        rows, offset = [], 0
        for cid in [c for c in per_channel if cell in per_channel[c]["built"]]:
            v = per_channel[cid]
            for k, (a, b) in enumerate(v["spans"]):
                row = classify(arm, cid, a, b, offset + k, caught)
                if row:
                    row["spacecraft"] = v["spacecraft"]
                    rows.append(row)
            offset += len(v["spans"])
        forensics[arm] = rows
        for sc in ("MSL", "SMAP"):
            tally = collections.Counter(r["class"] for r in rows
                                        if r["spacecraft"] == sc)
            print(f"  forensics {arm} {sc:5s}: " +
                  "  ".join(f"{k} {v}" for k, v in sorted(tally.items())))
    guard_rows = [{"arm": a, "channel": c,
                   **{k: v for k, v in d.items()
                      if k not in ("eps_at", "e_s")}}
                  for (a, c), d in diagnostics.items()]

    # -- the structural checks, per channel (27.4 S1, S2) --------------------
    def _mask(rows, arm, cid, n):
        m = np.zeros(n, dtype=bool)
        for r in rows:
            if r["arm"] == arm and r["channel"] == cid:
                for lo, hi in r["alarm_ranges"]:
                    m[lo:hi] = True
        return m

    structural = {}
    for lower, upper, kind in (("A0", "L1", "subset"), ("L1", "L2", "subset"),
                               ("L2", "L3", "subset"), ("L3", "L4", "superset"),
                               ("A0", "A2", "subset"),
                               # 30.2 G4, structural and a stop: every
                               # replacement is a relaxation, so the
                               # dimensionless arm's alarm set must be a
                               # superset of the absolute arm's on EVERY
                               # channel. A channel where it is not means a
                               # replacement tightened and the forms are wrong.
                               ("R", "G", "superset")):
        if lower not in results or upper not in results:
            continue
        held, total = 0, 0
        for cid, v in per_channel.items():
            if results[lower]["cell"] not in v["built"]:
                continue
            a = _mask(per_ch_rows, lower, cid, v["n"])
            b = _mask(per_ch_rows, upper, cid, v["n"])
            ok = (not (b & ~a).any()) if kind == "subset" else (not (a & ~b).any())
            held += int(ok); total += 1
        structural[f"{upper}_{kind}_of_{lower}"] = [held, total]
        print(f"  structural  {upper} {kind} of {lower}: {held}/{total}")

    print(f"  magnitude conjunct bound on {CONJUNCT_BOUND[0]} indices "
          f"across {CONJUNCT_TESTED[0]} window-passes")

    # -- the artifact, written BEFORE the ledger commit (75cc846) ------------
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = Path("runs/smap-msl/_forensics") / f"{stamp}-wi910-port.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "pre_registration": PRE_REGISTRATION,
        "dataset": "smap-msl/v1", "generated_utc": stamp,
        "split": "the paper's own train/test; not comparable to ESA-ADB folds",
        "not_a_cross_channel_test": "Objective.md 9.2 stands",
        "vendored_source": "third_party/telemanom @ 2e6c5b6c3558e7835601519b7bdef37c649bdbdc",
        "stage1_artifact": args.visibility, "stage4_artifact": args.stage4,
        "stage4_base_rate": base_rate,
        "paper_table2": {"SMAP": {"true": 69, "tp": 59, "fp": 10},
                         "MSL": {"true": 36, "tp": 25, "fp": 2},
                         "Total": {"true": 105, "tp": 84, "fp": 12}},
        "target_scaled_onto_this_population": {"tp": 78, "fp": 11,
                                               "derivation": "MODELS.md 27.4"},
        "populations": {c: {k: len(v) for k, v in pops[c].items()} for c in cells},
        "events": {c: events[c] for c in cells},
        "stalls": stalls, "geometry": geometry,
        "weight_store": {"before": before, "after": after},
        "structural_checks": structural,
        # 33.3: the whole curve, not only the matched point, so the
        # operating point is visible rather than asserted.
        "port_sweeps": sweeps,
        "guard_diagnostics": guard_rows,
        # 32.4 K3 is C2's stop and is adjudicated FIRST, so `sigma_cv` has to
        # reach the artifact for BOTH probabilistic cells. It collected only
        # "zscore" until 2026-09-09, which would have left K3 unadjudicable from
        # the artifact -- 27.8's defect exactly, where S1 and S2 could not be
        # settled because the instrument did not retain what the check needed.
        "zscore_diagnostics": [
            {"channel": c, "cell": cell, "spacecraft": v["spacecraft"],
             **{k: v["built"][cell].get(k) for k in
                ("threshold", "sigma_cv", "sigma_median", "n_exogenous", "nll",
                 "objective", "epochs", "best_epoch", "nll_improved")}}
            for c, v in per_channel.items()
            for cell in ("zscore", "zscore_cmd") if cell in v["built"]],
        "forensics": forensics,
        "magnitude_conjunct": {"indices_removed": CONJUNCT_BOUND[0],
                               "window_passes": CONJUNCT_TESTED[0]},
        "arms": results, "channels": per_ch_rows,
    }
    out["operations"] = budget.as_dict()
    path.write_text(json.dumps(out, indent=1))
    print(f"\n  artifact {path}")

    try:
        ops.commit(client, cfg.bucket, ledger, budget)
        print(); print(budget.report())
    except Exception as failure:               # 75cc846: the artifact is the
        print(f"\n  (!) LEDGER NOT COMMITTED: {failure}")   # expensive thing;
        print("      operations spent and NOT recorded; the artifact above is safe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
