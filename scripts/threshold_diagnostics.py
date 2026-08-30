"""What telemanom's z-selection criterion is actually choosing, and why.

Fixing `min_delta` improved the forecaster about fortyfold and the detection
stack downstream of it collapsed: 182 alarm ranges became 3,548 on the gate set
while recall, headline-cell recall and lead time all held or improved. A worse
model catches fewer events; this one catches more, at the same lead time, firing
twenty times as often. That is a threshold problem, and `scripts/threshold_sweep`
answered it with the wrong instrument -- raising `z_floor` does not tune a
threshold, it takes options away from a selector whose choices we did not like.

So this measures the selector instead of overriding it. Four things, in the
order they should be read:

**M0, the guard counterfactual.** Published telemanom refuses a candidate `z`
unless `len(E_seq) <= 5` **and** `len(i_anom) < len(e_s) * 0.5` -- a 50% coverage
cap that rejects outright any candidate flagging more than half the reference
window. Our `dynamic_threshold` has neither condition. That is invisible while
the forecast is poor and the guards never bind, and decisive once it is good.
This measures how often each would bind, where the selection would move, and
what the alarm count would be -- from the residuals, with no labels consulted.

**D1, what the criterion chooses.** The selected `z` is discarded at
`telemanom.dynamic_threshold`'s return. It is recovered exactly, without
touching that module, as `(eps - mu) / sigma` over the same window. Reported as
a distribution and never as a mean, beside the reachability bound and the number
of candidates that were admissible at all -- which may be the number that
matters and which nobody has looked at.

**D2, whether there is anything to choose between.** The criterion evaluated
across the whole candidate range. A flat or monotone criterion means the range
boundary is doing the work and the selection is decorative.

**D3, whether the residual changed shape.** The same weights on the same data
with only the stopping rule differing, so this is a controlled comparison rather
than a before-and-after. Shape and scale are reported separately, because the
hypothesis under test (docs/DECISIONS.md D17) is precisely that a dimensionless
constant survives rescaling and not reshaping, and conflating the two makes it
untestable.

**Nothing under `src/` is touched.** The guarded variant lives here, and every
window asserts that this file's unguarded selection reproduces the live
`telemanom.dynamic_threshold` byte for byte. Without that assertion the
counterfactual would be a statement about a function nobody scores.

**No labelled anomaly informs anything reported here.** Alarms are counted, not
scored: a mission with no failures can count its own alarms and cannot compute
recall. The one place annotations enter is the gap-and-invalid `scorable` mask,
used only to reproduce the published alarm-range count as a control on this
file's fidelity -- it carries no anomaly information.

Weights come from `runs/_weights`; nothing is refitted. Post-fix weights arrive
through the cache. Pre-fix weights cannot: `Hyper` carried `min_delta` then and
`min_improvement` now, so the key's shape moved and D14's orphaning applies. The
files survive, and are found by content -- `best_epoch == 0` after eleven epochs
is the defect's signature -- never by filename or modification time.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel_data import config as C                                  # noqa: E402
from sentinel_eval import bundle as bundle_mod                         # noqa: E402
from sentinel_eval import ops, read, splits, synthetic, tasks          # noqa: E402
from sentinel_eval.catalog import Catalog                              # noqa: E402
from sentinel_eval.detector import Context                             # noqa: E402
from sentinel_eval.labels import LabelSet                              # noqa: E402
from sentinel_eval.metrics.ranges import mask_to_ranges                # noqa: E402
from sentinel_eval.splits import train_mask                            # noqa: E402
from sentinel_models import detectors as D                             # noqa: E402
from sentinel_models import telemanom                                  # noqa: E402

#: Published telemanom's two admissibility conditions, from `errors.py`,
#: `find_epsilon`: `if score >= max_score and len(E_seq) <= 5 and
#: len(i_anom) < (len(e_s) * 0.5)`. Neither is present in
#: `sentinel_models.telemanom.dynamic_threshold`. Named here rather than inlined
#: because they are the subject of the measurement, not a setting of it.
PUBLISHED_MAX_SEQUENCES = 5
PUBLISHED_MAX_COVERAGE = 0.5

#: The two generations of weights this compares. Distinguished by content:
#: the defect stopped every fit at eleven epochs having kept its first.
PRE_FIX, POST_FIX = "pre-fix", "post-fix"
DEFECT_EPOCHS_RUN, DEFECT_BEST_EPOCH = 11, 0

#: Where D2's sampled tables come from. Stratified by how many candidates a
#: window had, so windows with a real range to choose from are not swamped by
#: the far more numerous windows that had one option or none.
SAMPLE_WINDOWS = 4000


# -- the criterion, re-implemented so its internals can be read ---------------
def candidate_table(e_s: np.ndarray, config: telemanom.Config):
    """Every candidate `z` for one window, with everything the criterion sees.

    Mirrors `telemanom.dynamic_threshold` line for line, and keeps what that
    function discards. The guard columns are computed but **not applied**: which
    candidate wins is decided in :func:`choose`, so one table serves both the
    live rule and the published one.
    """
    mu = float(np.mean(e_s))
    sigma = float(np.std(e_s))
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma == 0.0:
        return mu, sigma, float("nan"), []

    reach = (float(np.max(e_s)) - mu) / sigma
    span = e_s.shape[0]
    rows = []
    candidates = config.z_values
    for z in candidates[candidates <= reach]:
        eps = mu + z * sigma
        above = e_s >= eps
        if not above.any():
            continue
        remainder = e_s[~above]
        if remainder.size == 0:
            continue
        sequences = telemanom._buffered(above, config.error_buffer)
        if not sequences:
            continue
        d_mu = (mu - float(np.mean(remainder))) / mu if mu else 0.0
        d_sigma = (sigma - float(np.std(remainder))) / sigma
        covered = sum(hi - lo for lo, hi in sequences)
        rows.append({
            "z": float(z), "eps": float(eps),
            "n_above": int(above.sum()), "n_seq": len(sequences),
            "covered": int(covered), "d_mu": d_mu, "d_sigma": d_sigma,
            "score": (d_mu + d_sigma) / (len(sequences) ** 2 + covered),
            "seq_guard": len(sequences) <= PUBLISHED_MAX_SEQUENCES,
            "cov_guard": covered < PUBLISHED_MAX_COVERAGE * span,
            "sequences": sequences,
        })
    return mu, sigma, reach, rows


def choose(rows, mu: float, sigma: float, config: telemanom.Config, *, guarded: bool):
    """Pick a candidate. `>=` and ascending z, so ties go to the quieter choice.

    Returns ``(eps, sequences, z, silent)``. ``silent`` is the state published
    telemanom starts in and stays in when nothing qualifies -- eps at
    ``mu + sd_lim*sigma`` and nothing reported. It is tracked separately because
    downstream it is indistinguishable from a genuine selection that happened to
    exceed nothing: both leave every ratio below 1.
    """
    best = None
    for row in rows:
        if guarded and not (row["seq_guard"] and row["cov_guard"]):
            continue
        if best is None or row["score"] >= best["score"]:
            best = row
    if best is None:
        fallback = sigma if np.isfinite(sigma) else 0.0
        return mu + config.z_ceiling * fallback, [], float(config.z_ceiling), True
    return best["eps"], best["sequences"], best["z"], False


class Fidelity:
    """Proof that this file's unguarded selection *is* the scored one.

    Checked on every window rather than a sample: the counterfactual's whole
    value is that it describes the function the harness runs, and a sampled
    equivalence would leave that as an assumption.

    Exact equality, not a tolerance -- the two compute the same expression from
    the same float64 inputs, so anything but a bit-for-bit match is a different
    code path and not rounding. The one concession is that two NaNs agree:
    `_filled` makes a NaN unreachable in a run, and a guard that is wrong about
    the unreachable case is still wrong.
    """

    def __init__(self) -> None:
        self.checked = 0

    def __call__(self, e_s, config, eps_unguarded) -> None:
        reference, _ = telemanom.dynamic_threshold(e_s, config)
        both_nan = reference != reference and eps_unguarded != eps_unguarded
        if not both_nan and reference != eps_unguarded:
            raise SystemExit(
                f"FIDELITY FAILED: this file's unguarded selection is not "
                f"telemanom.dynamic_threshold's.\n"
                f"  reference {reference!r}  reimplementation {eps_unguarded!r}\n"
                f"  Everything downstream describes a function nobody scores. "
                f"Aborting rather than reporting it."
            )
        self.checked += 1


# -- one channel, every window on the real path -------------------------------
def channel_pass(e_s: np.ndarray, config: telemanom.Config, fidelity: Fidelity,
                 collect):
    """`telemanom.channel_ratios`' loop, keeping what it throws away.

    Same span, same stride, same trailing reference window -- including the fact
    that the window handed to the criterion is ``[seg_lo - span, seg_hi)`` and so
    is 2,170 samples with the judged segment inside it, not the 2,100 preceding
    ones the module docstring describes.

    Returns the two alarm masks, live and guarded, and appends one record per
    window to ``collect``.
    """
    steps = e_s.shape[0]
    span, stride = config.error_window, max(1, config.stride)
    alarm = {False: np.zeros(steps, dtype=bool), True: np.zeros(steps, dtype=bool)}

    for seg_lo in range(0, steps, stride):
        seg_hi = min(seg_lo + stride, steps)
        reference_lo = max(0, seg_lo - span)
        window = e_s[reference_lo:seg_hi]
        offset = seg_lo - reference_lo

        mu, sigma, reach, rows = candidate_table(window, config)
        picked = {}
        for guarded in (False, True):
            eps, sequences, z, silent = choose(rows, mu, sigma, config, guarded=guarded)
            picked[guarded] = (eps, z, silent)
            if not sequences:
                continue
            keeps = telemanom.prune(window, sequences, eps, config.pruning_p)
            for keep, (lo, hi) in zip(keeps, sequences):
                lo, hi = max(lo, offset), min(hi, window.shape[0])
                if keep and hi > lo:
                    alarm[guarded][reference_lo + lo:reference_lo + hi] = True

        fidelity(window, config, picked[False][0])
        collect(seg_lo, mu, sigma, reach, rows, picked)

    for guarded in (False, True):
        opening = min(telemanom.EWMA_SETTLE * config.smoothing_window, steps)
        alarm[guarded][:opening] = False
    return alarm[False], alarm[True]


class Records:
    """Per-window diagnostics for one fold and one channel set.

    Fixed-width columns in preallocated arrays: at stride 70 a fold holds ~52,600
    windows per channel and twelve channels of Python dictionaries would be a
    gigabyte of small objects to answer a question about six numbers.
    """

    FIELDS = ("channel", "seg_lo", "mu", "sigma", "reach", "n_candidates",
              "n_admissible", "z_live", "z_guarded", "silent_live", "silent_guarded")

    def __init__(self) -> None:
        self.rows: list[list] = []

    def table(self) -> np.ndarray:
        return np.asarray(self.rows, dtype=np.float64) if self.rows \
            else np.zeros((0, len(self.FIELDS)))


class Stratified:
    """Reservoir sample, stratified by how many candidates a window had.

    An unstratified sample would be almost entirely windows with one option or
    none, which is a fact D1 already reports and is not what D2 asks. Seeded, so
    the sampled tables in the artifact are reproducible.
    """

    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = max(1, capacity)
        self.rng = np.random.default_rng(seed)
        self.seen: dict[int, int] = {}
        self.kept: dict[int, list] = {}

    def offer(self, stratum: int, make) -> None:
        n = self.seen.get(stratum, 0) + 1
        self.seen[stratum] = n
        pool = self.kept.setdefault(stratum, [])
        if len(pool) < self.capacity:
            pool.append(make())
            return
        slot = int(self.rng.integers(0, n))
        if slot < self.capacity:
            pool[slot] = make()

    def all(self) -> list:
        return [row for _, pool in sorted(self.kept.items()) for row in pool]


# -- weights, by content and never by filename --------------------------------
def inspect_store(store: Path) -> list[dict]:
    """Every cached fit, described by what is inside it."""
    out = []
    if not store.is_dir():
        return out
    for path in sorted(store.iterdir()):
        if path.suffix != ".npz":
            continue
        try:
            with np.load(path, allow_pickle=False) as blob:
                report = json.loads(str(blob["report"]))
                n_channels = int(blob["n_channels"])
                n_in = int(blob["l0_w_ih"].shape[1])
                exogenous = (int(blob["n_exogenous"]) if "n_exogenous" in blob.files
                             else n_in - n_channels)
        except Exception:
            continue
        out.append({"digest": path.stem, "n_channels": n_channels,
                    "n_exogenous": exogenous, "report": report})
    return out


def find_pre_fix(store_entries: list[dict], n_channels: int, train_positions: int):
    """The defect-era fit for one channel set and one fold, or None.

    Identified by signature, not by date: the control arm (no exogenous inputs),
    the right width, the same training window -- `train_start_positions` is a
    function of the fold -- and `best_epoch == 0` after eleven epochs, which is
    what the broken stopping rule did to every fit in the project.
    """
    matches = [e for e in store_entries
               if e["n_channels"] == n_channels and e["n_exogenous"] == 0
               and e["report"].get("best_epoch") == DEFECT_BEST_EPOCH
               and e["report"].get("epochs_run") == DEFECT_EPOCHS_RUN
               and e["report"].get("train_start_positions") == train_positions]
    if not matches:
        return None, []
    return matches[0], [m["digest"] for m in matches]


def identical(first: str, second: str) -> bool:
    """Two cached fits hold the same numbers. D14's refit produced duplicates."""
    a, b = D._load_weights(first), D._load_weights(second)
    if a is None or b is None:
        return False
    left = [x for layer in a[0].layers for x in (layer.w_ih, layer.w_hh,
                                                 layer.b_ih, layer.b_hh)]
    right = [x for layer in b[0].layers for x in (layer.w_ih, layer.w_hh,
                                                  layer.b_ih, layer.b_hh)]
    left += [a[0].head_w, a[0].head_b]
    right += [b[0].head_w, b[0].head_b]
    return all(np.array_equal(x, y) for x, y in zip(left, right))


# -- residuals, one forward pass yielding both raw and smoothed ---------------
def residuals(detector, values: np.ndarray, context) -> tuple[np.ndarray, np.ndarray]:
    """`ForecastDetector._smoothed_errors`, keeping the raw residual it discards.

    Identical arithmetic and identical blocking -- the EWMA state is carried
    across blocks exactly as it is there, so this is the same smoothed series the
    harness scores. The signed residual is kept because D3's question is about
    distribution shape, and the absolute value it is folded into has a skew of
    its own whatever the model does.
    """
    filled = detector._filled(values)
    steps, channels = filled.shape
    smoothed = np.empty((steps, channels), dtype=np.float32)
    signed = np.empty((steps, channels), dtype=np.float32)
    state = telemanom.EwmaState()
    block = detector.chunks * detector.chunk_steps

    for lo in range(0, steps, block):
        hi = min(lo + block, steps)
        forecast = detector._forecast(filled, lo, hi)
        signed[lo:hi] = filled[lo:hi] - forecast
        smoothed[lo:hi] = telemanom.ewma(np.abs(signed[lo:hi]),
                                         detector.config.smoothing_window, state)
    return signed, smoothed


def shape(values: np.ndarray, stride: int = 37) -> dict:
    """Scale and shape, kept apart because the hypothesis is about shape alone.

    Moments are exact over the whole series; quantiles, the Hill tail index and
    the normal-quantile agreement come from a strided subsample, which at stride
    37 is ~100,000 points per channel and costs no sort of eleven million.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 100:
        return {}
    n = x.size
    mean = float(x.mean())
    centred = x - mean
    m2 = float(np.mean(centred ** 2))
    sd = float(np.sqrt(m2))
    if sd == 0.0:
        return {"n": n, "mean": mean, "sd": 0.0}
    skew = float(np.mean(centred ** 3) / sd ** 3)
    kurtosis = float(np.mean(centred ** 4) / m2 ** 2) - 3.0

    sample = np.sort(x[::stride])
    q = {p: float(np.quantile(sample, p / 100.0)) for p in (50, 75, 95, 99, 99.9)}
    magnitude = np.sort(np.abs(sample))
    tail = magnitude[magnitude > 0]
    hill = None
    if tail.size > 200:
        k = max(10, int(0.05 * tail.size))
        top = tail[-k:]
        hill = float(np.mean(np.log(top / top[0]))) if top[0] > 0 else None

    # Agreement with a normal of the same mean and sd, as the largest gap
    # between the empirical and normal quantiles in units of sd. Zero is
    # Gaussian; a heavy tail shows up as a large positive number.
    probs = np.linspace(0.001, 0.999, 199)
    empirical = np.quantile(sample, probs)
    normal = mean + sd * np.sqrt(2.0) * _erfinv(2.0 * probs - 1.0)
    qq_max = float(np.max(np.abs(empirical - normal)) / sd)

    return {"n": n, "mean": mean, "sd": sd, "skew": skew, "excess_kurtosis": kurtosis,
            "p50": q[50], "p75": q[75], "p95": q[95], "p99": q[99], "p99.9": q[99.9],
            "p99.9_over_p50": (q[99.9] / q[50]) if q[50] else None,
            "hill_tail_index": hill, "qq_max_gap_sd": qq_max}


def _erfinv(y: np.ndarray) -> np.ndarray:
    """Inverse error function by Newton refinement of a rational start.

    Hand-rolled because scipy is not a dependency of this project and adding one
    to draw a QQ line would be a poor trade. Accurate to ~1e-9 over the range
    used here, which is far tighter than anything read off the result.
    """
    y = np.clip(np.asarray(y, dtype=np.float64), -1 + 1e-12, 1 - 1e-12)
    a = 0.147
    ln = np.log(1 - y * y)
    first = 2 / (np.pi * a) + ln / 2
    x = np.sign(y) * np.sqrt(np.sqrt(first * first - ln / a) - first)
    for _ in range(3):
        err = (2 / np.sqrt(np.pi)) * np.exp(-x * x)
        x -= (np.vectorize(_erf)(x) - y) / err
    return x


def _erf(x: float) -> float:
    import math
    return math.erf(x)


# -- the run ------------------------------------------------------------------
def load(task, args):
    """One bundle, held for everything. Costs what a single scored run costs."""
    if task.id == "synthetic":
        source, client, budget, state = synthetic.build(seed=0, n=args.steps), None, None, None
    else:
        cfg = C.load_r2_config()
        client, budget = ops.connect(cfg, acknowledge_tripwire=args.acknowledge_tripwire)
        ledger = ops.load(client, cfg.bucket)
        budget.prior_class_a, budget.prior_class_b = ledger.month_class_a, ledger.month_class_b
        source, state = read.R2Source(client, cfg.bucket), (cfg, ledger)
    catalog = Catalog.load(source)
    labels = LabelSet.from_table(read.read_annotation(source, catalog, "labels"))
    loaded = bundle_mod.load(source, catalog, labels, mission=task.mission,
                             channel_ids=task.selection.resolve(catalog))
    return loaded, labels, catalog, client, budget, state


def members(primary, catalog, labels, loaded):
    """The paired sets, as `cli.py` builds them: the primary's split, so only
    channels differ. Reporting both is not optional (docs/HARNESS.md section 2)."""
    out = []
    for task_id in tasks.paired_with(primary.id):
        task = tasks.get(task_id)
        if task_id != primary.id:
            task = type(task)(**{**vars(task), "split": primary.split,
                                 "split_params": primary.split_params})
        view = loaded if task_id == primary.id else loaded.subset(
            task.selection.resolve(catalog), labels)
        out.append((task, view))
    return out


def context_for(view, fold, window):
    return Context(mission=view.mission, channels=view.channel_ids, groups=view.groups,
                   period_seconds=view.provenance["grid_period_seconds"],
                   fold=fold.index, window=window,
                   commands=None, command_ids=view.command_ids)


def analyse(view, fold, detector, generation: str, args) -> dict:
    """Everything M0, D1, D2 and D3 need, from one forward pass over one fold."""
    test_lo, test_hi = fold.test
    warmup = min(detector.warmup_steps, test_lo)
    window_lo = test_lo - warmup
    values = view.values[window_lo:test_hi]
    scored = context_for(view, fold, (test_lo, test_hi))

    started = time.perf_counter()
    signed, smoothed = residuals(detector, values, scored)
    forward_seconds = time.perf_counter() - started

    config = detector.config
    fidelity = Fidelity()
    records = Records()
    strata = Stratified(max(1, args.sample_windows // 6))
    steps, channels = smoothed.shape
    live = np.zeros(steps, dtype=bool)
    guarded = np.zeros(steps, dtype=bool)

    started = time.perf_counter()
    for channel in range(channels):
        def collect_and_sample(seg_lo, mu, sigma, reach, rows, picked, ch=channel):
            records.rows.append([
                ch, seg_lo, mu, sigma, reach, len(rows),
                sum(1 for r in rows if r["seq_guard"] and r["cov_guard"]),
                picked[False][1], picked[True][1], picked[False][2], picked[True][2]])
            if rows:
                strata.offer(min(len(rows), 6), lambda: {
                    "channel": ch, "seg_lo": int(seg_lo), "mu": mu, "sigma": sigma,
                    "reach": reach, "z_live": picked[False][1],
                    "z_guarded": picked[True][1],
                    "candidates": [{k: v for k, v in r.items() if k != "sequences"}
                                   for r in rows]})

        a_live, a_guarded = channel_pass(smoothed[:, channel], config, fidelity,
                                         collect_and_sample)
        live |= a_live
        guarded |= a_guarded
    criterion_seconds = time.perf_counter() - started

    # The harness's own alarm accounting, so the live arm reproduces the
    # published count rather than merely resembling it.
    observed = np.isfinite(values).all(axis=1)
    observed &= np.asarray(view.valid[window_lo:test_hi], dtype=bool).all(axis=1)
    scorable = np.asarray(view.truth.scorable[test_lo:test_hi], dtype=bool)
    alarms = {}
    for name, mask in (("live", live), ("guarded", guarded)):
        kept = mask[warmup:] & observed[warmup:]
        alarms[name] = {
            "alarm_ranges_raw": len(mask_to_ranges(kept)),
            "alarm_ranges_scorable": len(mask_to_ranges(kept & scorable)),
            "alarm_steps": int((kept & scorable).sum()),
            "scorable_steps": int(scorable.sum()),
        }

    table = records.table()
    return {
        "generation": generation,
        "fold": fold.index,
        "window": [int(test_lo), int(test_hi)],
        "warmup": int(warmup),
        "windows_examined": int(table.shape[0]),
        "fidelity_windows_checked": fidelity.checked,
        "seconds": {"forward": round(forward_seconds, 1),
                    "criterion": round(criterion_seconds, 1)},
        "training": detector.report,
        "alarms": alarms,
        "d1": d1_summary(table, config),
        "m0": m0_summary(table, alarms),
        "d2": d2_summary(strata.all()),
        "d3": {"residual_signed": shape(signed[warmup:]),
               "residual_absolute": shape(np.abs(signed[warmup:])),
               "smoothed_e_s": shape(smoothed[warmup:]),
               "per_channel_smoothed": [shape(smoothed[warmup:, c])
                                        for c in range(channels)]},
        "_table": table,
    }


def _histogram(values: np.ndarray, config: telemanom.Config) -> dict:
    out = {}
    for z in config.z_values:
        n = int(np.sum(values == float(z)))
        if n:
            out[f"{float(z):.1f}"] = n
    n = int(np.sum(values == float(config.z_ceiling)))
    if n:
        out[f"{float(config.z_ceiling):.1f} (fallback)"] = n
    return out


def _quantiles(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {}
    return {"n": int(finite.size),
            "min": float(finite.min()), "p25": float(np.quantile(finite, 0.25)),
            "median": float(np.median(finite)), "p75": float(np.quantile(finite, 0.75)),
            "max": float(finite.max())}


def d1_summary(table: np.ndarray, config: telemanom.Config) -> dict:
    """What the criterion chose, as a distribution and never as a mean."""
    if table.shape[0] == 0:
        return {}
    z_live, silent = table[:, 7], table[:, 9].astype(bool)
    selected = z_live[~silent]
    counts = np.bincount(table[:, 5].astype(int), minlength=20)
    return {
        "windows": int(table.shape[0]),
        "silent": int(silent.sum()),
        "silent_share": float(silent.mean()),
        "selected_z": _histogram(selected, config),
        "selected_z_at_floor": int(np.sum(selected == float(config.z_floor))),
        "selected_z_at_floor_share": (float(np.mean(selected == float(config.z_floor)))
                                      if selected.size else None),
        "reach": _quantiles(table[:, 4]),
        "candidates_available": {str(i): int(n) for i, n in enumerate(counts) if n},
        "candidates_available_median": float(np.median(table[:, 5])),
        "admissible_under_published_guards_median": float(np.median(table[:, 6])),
    }


def m0_summary(table: np.ndarray, alarms: dict) -> dict:
    """What published telemanom's two conditions would have done. Alarms only."""
    if table.shape[0] == 0:
        return {}
    n_cand, n_adm = table[:, 5], table[:, 6]
    z_live, z_guarded = table[:, 7], table[:, 8]
    silent_live, silent_guarded = table[:, 9].astype(bool), table[:, 10].astype(bool)
    had_choice = n_cand > 0
    moved = (~silent_live) & (~silent_guarded) & (z_live != z_guarded)
    silenced = (~silent_live) & silent_guarded
    shifts = (z_guarded - z_live)[moved]
    return {
        "windows_with_a_candidate": int(had_choice.sum()),
        "windows_where_a_guard_binds": int(np.sum(had_choice & (n_adm < n_cand))),
        "windows_where_a_guard_binds_share": (float(np.mean(n_adm[had_choice] < n_cand[had_choice]))
                                              if had_choice.any() else None),
        "selection_moved": int(moved.sum()),
        "selection_moved_share": (float(moved.sum() / max(1, int((~silent_live).sum())))),
        "selection_silenced": int(silenced.sum()),
        "z_shift": _quantiles(shifts) if shifts.size else {},
        "alarm_ranges_live": alarms["live"]["alarm_ranges_scorable"],
        "alarm_ranges_guarded": alarms["guarded"]["alarm_ranges_scorable"],
    }


def _decomposition(samples: list[dict]) -> dict:
    """The criterion split into its parts, per candidate z.

    D2 establishes *that* the criterion is monotone. This says *which term* makes
    it so, which is the difference between a finding about the selection rule and
    a finding about a stage upstream of it. The denominator is
    ``len(E_seq)**2 + covered`` and ``covered`` is the length of the buffered,
    merged sequences -- so at ``error_buffer = 100`` a single exceeded timestep
    already costs 199. If that term barely moves with z while the numerator
    falls, the criterion is arithmetically forced to prefer the smallest z on
    offer and no residual distribution can rescue it.

    Medians over the sampled windows that had more than one candidate, so a
    window with one option cannot vote on a question about choosing.
    """
    parts: dict[str, dict[str, list]] = {}
    for row in samples:
        if len(row["candidates"]) < 2:
            continue
        for c in row["candidates"]:
            slot = parts.setdefault(f"{c['z']:.1f}",
                                    {"n_above": [], "n_seq": [], "covered": [],
                                     "numerator": [], "denominator": [], "score": []})
            slot["n_above"].append(c["n_above"])
            slot["n_seq"].append(c["n_seq"])
            slot["covered"].append(c["covered"])
            slot["numerator"].append(c["d_mu"] + c["d_sigma"])
            slot["denominator"].append(c["n_seq"] ** 2 + c["covered"])
            slot["score"].append(c["score"])
    return {z: {"n": len(v["score"]),
                **{k: float(np.median(x)) for k, x in v.items()}}
            for z, v in sorted(parts.items(), key=lambda kv: float(kv[0]))}


def d2_summary(samples: list[dict]) -> dict:
    """Is there a peak, or is the range boundary doing all the work."""
    if not samples:
        return {}
    at_min = decreasing = increasing = interior = flat = 0
    margins, multi = [], 0
    curve: dict[str, list[float]] = {}
    for row in samples:
        scores = [c["score"] for c in row["candidates"]]
        best = max(scores)
        chosen = max(i for i, s in enumerate(scores) if s == best)
        if chosen == 0:
            at_min += 1
        elif chosen == len(scores) - 1:
            pass
        else:
            interior += 1
        if len(scores) > 1:
            multi += 1
            diffs = np.diff(scores)
            if np.all(diffs <= 0):
                decreasing += 1
            elif np.all(diffs >= 0):
                increasing += 1
            if best > 0:
                rest = [s for i, s in enumerate(scores) if i != chosen]
                if rest:
                    margins.append(best / max(rest) if max(rest) > 0 else float("inf"))
            if best == min(scores):
                flat += 1
            for c in row["candidates"]:
                curve.setdefault(f"{c['z']:.1f}", []).append(
                    c["score"] / best if best > 0 else 0.0)
    finite = [m for m in margins if np.isfinite(m)]
    return {
        "decomposition_by_z": _decomposition(samples),
        "examples": [samples[i] for i in
                     range(0, len(samples), max(1, len(samples) // 8))][:8],
        "sampled_windows": len(samples),
        "windows_with_more_than_one_candidate": multi,
        "argmax_at_range_minimum": at_min,
        "argmax_at_range_minimum_share": at_min / len(samples),
        "argmax_interior": interior,
        "monotone_decreasing": decreasing,
        "monotone_increasing": increasing,
        "flat": flat,
        "margin_over_next_best": (_quantiles(np.asarray(finite)) if finite else {}),
        "normalised_score_by_z": {z: {"n": len(v), "median": float(np.median(v))}
                                  for z, v in sorted(curve.items(), key=lambda kv: float(kv[0]))},
    }


# -- rendering ----------------------------------------------------------------
def render(results: dict) -> str:
    lines = []
    for set_id, rows in results.items():
        lines.append("")
        lines.append(f"  {set_id}")
        lines.append("")
        lines.append("  M0  THE GUARD COUNTERFACTUAL -- published telemanom's two "
                     "admissibility conditions")
        lines.append("      len(E_seq) <= 5  and  len(i_anom) < len(e_s) * 0.5. "
                     "We have neither.")
        lines.append("      Alarms are COUNTED, never scored: no recall, no precision, "
                     "no F0.5, no")
        lines.append("      lead time. Choosing a threshold by looking at scored results "
                     "is the mistake")
        lines.append("      this investigation exists to correct.")
        lines.append("")
        lines.append("      gen       fold  windows   guard binds   moved   silenced"
                     "   alarm ranges live -> guarded")
        for row in rows:
            m0, alarms = row.get("m0") or {}, row["alarms"]
            if not m0:
                continue
            binds = m0["windows_where_a_guard_binds_share"]
            lines.append(
                f"      {row['generation']:9s} {row['fold']:^4} "
                f"{m0['windows_with_a_candidate']:>8,} "
                f"{(binds if binds is not None else 0):>12.3f} "
                f"{m0['selection_moved_share']:>7.3f} "
                f"{m0['selection_silenced']:>10,} "
                f"{alarms['live']['alarm_ranges_scorable']:>13,} -> "
                f"{alarms['guarded']['alarm_ranges_scorable']:,}")

        lines.append("")
        lines.append("  D1  WHAT THE CRITERION IS CHOOSING -- distributions, not means")
        lines.append("")
        lines.append("      gen       fold   windows    silent   z at floor   "
                     "reach med   candidates med")
        for row in rows:
            d1 = row.get("d1") or {}
            if not d1:
                continue
            share = d1["selected_z_at_floor_share"]
            lines.append(
                f"      {row['generation']:9s} {row['fold']:^4} {d1['windows']:>9,} "
                f"{d1['silent_share']:>9.3f} "
                f"{(share if share is not None else float('nan')):>12.3f} "
                f"{d1['reach'].get('median', float('nan')):>11.2f} "
                f"{d1['candidates_available_median']:>16.1f}")
        for row in rows:
            d1 = row.get("d1") or {}
            if d1:
                lines.append(f"      {row['generation']} fold {row['fold']} selected z: "
                             + "  ".join(f"{k}={v:,}" for k, v in d1["selected_z"].items()))

        lines.append("")
        lines.append("  D2  IS THERE ANYTHING TO CHOOSE BETWEEN")
        lines.append("")
        lines.append("      gen       fold   sampled   >1 cand   argmax at min   "
                     "interior   monotone dec")
        for row in rows:
            d2 = row.get("d2") or {}
            if not d2:
                continue
            lines.append(
                f"      {row['generation']:9s} {row['fold']:^4} "
                f"{d2['sampled_windows']:>9,} "
                f"{d2['windows_with_more_than_one_candidate']:>9,} "
                f"{d2['argmax_at_range_minimum_share']:>15.3f} "
                f"{d2['argmax_interior']:>10,} "
                f"{d2['monotone_decreasing']:>14,}")

        lines.append("")
        lines.append("  D3  DID THE RESIDUAL CHANGE SHAPE -- scale and shape kept apart")
        lines.append("")
        lines.append("      gen       fold   quantity        sd         skew   "
                     "ex.kurtosis   p99.9/p50   QQ gap (sd)")
        for row in rows:
            for name, key in (("signed resid", "residual_signed"),
                              ("smoothed e_s", "smoothed_e_s")):
                s = (row.get("d3") or {}).get(key) or {}
                if not s:
                    continue
                ratio = s.get("p99.9_over_p50")
                lines.append(
                    f"      {row['generation']:9s} {row['fold']:^4}  {name:14s} "
                    f"{s['sd']:.3e} {s.get('skew', float('nan')):>+11.3f} "
                    f"{s.get('excess_kurtosis', float('nan')):>13.2f} "
                    f"{(ratio if ratio is not None else float('nan')):>11.2f} "
                    f"{s.get('qq_max_gap_sd', float('nan')):>13.3f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="m1-g8.9.10")
    parser.add_argument("--folds", default="all",
                        help="comma-separated fold indices, or 'all'")
    parser.add_argument("--generations", default="both",
                        choices=("both", "post-fix", "pre-fix"))
    parser.add_argument("--sample-windows", type=int, default=SAMPLE_WINDOWS)
    parser.add_argument("--steps", type=int, default=20_000,
                        help="synthetic fixture length; ignored for real tasks")
    parser.add_argument("--acknowledge-tripwire", action="store_true")
    args = parser.parse_args(argv)

    primary = tasks.get(args.task)
    if primary.id in ("m1-g3", "m2-ss1"):
        print(f"  REFUSED: {primary.id} is held back. It is run once, at the end, with "
              f"settings already frozen (docs/MODELS.md section 6). A diagnostic is "
              f"exactly the kind of look that spends it.")
        return 2

    print(f"  Threshold-selector diagnostics -- {primary.id}")
    print(f"  Weights are reused, never fitted. This script refuses to train.")
    loaded, labels, catalog, client, budget, state = load(primary, args)

    store = D.WEIGHT_STORE
    entries = inspect_store(store)
    before = sum(1 for p in store.iterdir() if p.suffix == ".npz") if store.is_dir() else 0
    print(f"  weight store: {before} cached fits inspected before anything ran")

    wanted = None if args.folds == "all" else {int(f) for f in args.folds.split(",")}
    generations = ((POST_FIX, PRE_FIX) if args.generations == "both"
                   else (args.generations,))
    results: dict[str, list[dict]] = {}
    tables: dict[str, np.ndarray] = {}
    provenance: list[dict] = []

    for task, view in members(primary, catalog, labels, loaded):
        split = splits.forward_chaining(len(view.grid), **task.split_kwargs)
        rows = results.setdefault(task.id, [])
        for fold in split.folds:
            if wanted is not None and fold.index not in wanted:
                continue
            train_lo, train_hi = fold.train
            usable = train_mask(fold, view.truth)[train_lo:train_hi]
            values = view.values[train_lo:train_hi]
            fitting = context_for(view, fold, fold.train)

            for generation in generations:
                detector = D.ForecastDetector()
                if generation == POST_FIX:
                    key = (detector.hyper.as_dict_key(), fitting.channels, fitting.fold,
                           fitting.window, values.shape, D._sample_digest(values),
                           D._digest(usable[::997]), None)
                    digest = D._digest(key)
                    cached = D._load_weights(digest)
                    if cached is None:
                        print(f"    {task.id} fold {fold.index}: no cached post-fix fit "
                              f"for this key ({digest}). This script does not train; "
                              f"run scripts/fit_folds.py first.")
                        return 3
                    detector.fit(values, usable, fitting)
                    chosen, duplicates = digest, [digest]
                else:
                    reference = next((r for r in rows if r["fold"] == fold.index
                                      and r["generation"] == POST_FIX), None)
                    positions = (reference or {}).get("training", {}).get(
                        "train_start_positions")
                    entry, duplicates = find_pre_fix(entries, len(view.channels), positions)
                    if entry is None:
                        print(f"    {task.id} fold {fold.index}: no defect-era fit on "
                              f"disk for this fold. D3's pre-fix arm is UNAVAILABLE "
                              f"here and is reported empty, not estimated.")
                        provenance.append({"set": task.id, "fold": fold.index,
                                           "generation": PRE_FIX, "digest": None,
                                           "note": "no defect-era fit on disk"})
                        continue
                    if any(not identical(entry["digest"], other)
                           for other in duplicates[1:]):
                        print(f"    {task.id} fold {fold.index}: duplicate defect-era "
                              f"fits disagree. Refusing to pick one.")
                        return 4
                    weights, report = D._load_weights(entry["digest"])
                    if (weights.n_channels != len(view.channels)
                            or weights.window != detector.hyper.window
                            or weights.n_predictions != detector.hyper.n_predictions
                            or weights.n_exogenous != 0):
                        print(f"    {task.id} fold {fold.index}: defect-era fit does not "
                              f"match this view. Refusing to score it.")
                        return 5
                    detector._weights, detector.report = weights, report
                    detector._fit_window, detector._impulses = fold.train, None
                    sample = values[usable] if usable.any() else values
                    with np.errstate(invalid="ignore"):
                        fill = np.nanmean(sample, axis=0).astype(np.float32)
                    detector._fill = np.nan_to_num(fill, nan=0.0)
                    chosen = entry["digest"]

                print(f"    {task.id:12s} fold {fold.index}  {generation:8s}  "
                      f"weights {chosen[:12]}  best_epoch "
                      f"{detector.report.get('best_epoch')}  val MSE "
                      f"{detector.report.get('best_validation_mse'):.3e}", flush=True)
                row = analyse(view, fold, detector, generation, args)
                tables[f"{task.id}__{generation}__fold{fold.index}"] = row.pop("_table")
                rows.append(row)
                provenance.append({"set": task.id, "fold": fold.index,
                                   "generation": generation, "digest": chosen,
                                   "duplicates": duplicates})
                print(f"      {row['windows_examined']:,} windows, "
                      f"{row['fidelity_windows_checked']:,} verified identical to "
                      f"telemanom.dynamic_threshold; "
                      f"{row['seconds']['forward']}s forward, "
                      f"{row['seconds']['criterion']}s criterion", flush=True)

    after = sum(1 for p in store.iterdir() if p.suffix == ".npz") if store.is_dir() else 0
    if after != before:
        print(f"  ABORT: the weight store went from {before} to {after} files. "
              f"Something trained. A cache miss here is silent (docs/DECISIONS.md D16) "
              f"and every number above would be from weights this run invented.")
        return 6
    print(f"\n  weight store unchanged at {after} files -- nothing was fitted")

    print(render(results))


    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    folder = C.PROJECT_ROOT / "runs" / primary.id / "_threshold"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stamp}-diagnostics.json"
    path.write_text(json.dumps({
        "task": primary.id,
        "published_guards": {"max_sequences": PUBLISHED_MAX_SEQUENCES,
                             "max_coverage_fraction": PUBLISHED_MAX_COVERAGE,
                             "source": "khundman/telemanom, telemanom/errors.py, "
                                       "find_epsilon",
                             "present_in_this_repository": False},
        "config": telemanom.Config().as_dict(),
        "sample_windows": args.sample_windows,
        "weights": provenance,
        "operations": budget.as_dict() if budget else None,
        "sets": results,
    }, indent=2, default=str) + "\n")
    print(f"\n  wrote {path.relative_to(C.PROJECT_ROOT)}")

    # Artifact first, ledger second, and the order is load-bearing: a
    # transient RequestTimeTooSkewed on the ledger PutObject once destroyed
    # twenty-five minutes of completed analysis held in memory. The result is
    # the expensive thing; the ledger is bookkeeping and can be retried.
    try:
        if budget is not None:
            cfg, ledger = state
            ops.commit(client, cfg.bucket, ledger, budget)
            print()
            print(budget.report())
    except Exception as failure:
        print(f"\n  (!) LEDGER NOT COMMITTED: {failure}")
        print("      the operations were spent and are NOT recorded; "
              "the artifact above is safe.")

    if tables:
        arrays = folder / f"{stamp}-windows.npz"
        np.savez_compressed(arrays, columns=np.array(Records.FIELDS), **tables)
        print(f"  wrote {arrays.relative_to(C.PROJECT_ROOT)} "
              f"({', '.join(Records.FIELDS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
