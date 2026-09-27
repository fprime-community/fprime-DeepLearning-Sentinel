#!/usr/bin/env python3
"""D86: the rate of change inside the ML, pre-registered in docs/MODELS.md 79.

    PYTHONPATH=src .venv/bin/python scripts/d86_arms.py prep-esa
    PYTHONPATH=src .venv/bin/python scripts/d86_arms.py fit   --dataset {smap,esa} --workers N
    PYTHONPATH=src .venv/bin/python scripts/d86_arms.py score --dataset {smap,esa} --workers N
    PYTHONPATH=src .venv/bin/python scripts/d86_arms.py tune  --dataset {smap,esa}
    PYTHONPATH=src .venv/bin/python scripts/d86_arms.py eval  --dataset {smap,esa}

**Pre-registered in `docs/MODELS.md` 79 before any phase ran.** Research only: nothing
in `flight/`, the Monitor, `reference.py` or `param_version` changes (stop 65).

  fit    trains forecasters only: the joint [x, d] models (both datasets) and the ESA
         univariate models, into `runs/_weights_d86`. No score, no recall.
  score  every term, every channel, written to `runs/d86/<ds>/terms/`. No recall.
  tune   TUNE only. Reproduction gates (SMAP), cuts at the matched rate, r chosen on
         D86.A4's TUNE catches. Writes `runs/d86/<ds>/frozen.json`; EVAL events are not
         loaded. The file is then committed as `tests/fixtures/d86/<ds>_frozen.json`.
  eval   refuses unless `frozen.json` is byte-identical to the committed fixture at HEAD
         and the tracked tree is clean. Scores EVAL once, with per-event attribution.

The 1,313 cached weights in `runs/_weights` are read, never written: a sha256 manifest
of the store is taken before and after every phase and must not move (stop 61). No
wall-clock figure is printed or written (stop 35).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse, hashlib, importlib.util, io, json, subprocess, sys, warnings  # noqa: E401
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="Degrees of freedom")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel_eval import bundle as bundle_mod, read, splits, tasks      # noqa: E402
from sentinel_eval.catalog import Catalog                               # noqa: E402
from sentinel_eval.detector import Context                              # noqa: E402
from sentinel_eval.labels import ANOMALY, LabelSet                      # noqa: E402
from sentinel_models import detectors as D, reference, registry, telemanom  # noqa: E402
from sentinel_models.windows import aggregate_predictions               # noqa: E402
from sentinel_toolkit.statistic import (derivative, motion, slope_mismatch, zstat,  # noqa: E402
                                        zstat_floored)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SR = _load("smap_rungs", "scripts/smap_rungs.py")

# -- docs/MODELS.md 79.4, fixed before any phase ran ------------------------------------
MIRROR = ROOT / "runs" / "_mirror"
OUT = ROOT / "runs" / "d86"
CACHED_STORE = D.WEIGHT_STORE                        # runs/_weights, read only
D86_STORE = ROOT / "runs" / "_weights_d86"
FIXTURES = ROOT / "tests" / "fixtures" / "d86"
SMAP_MANIFEST = "_manifest/smap_msl.json"
VISIBILITY = ROOT / "runs/smap-msl/_forensics/2026-09-03T192723Z-visibility.json"
STAGE4 = ROOT / "runs/smap-msl/_forensics/2026-09-03T212818Z-stage4-ndt.json"
ESA_TASK = "m1-g8.9.10"
ESA_TUNE_FOLDS, ESA_EVAL_FOLDS = (0, 1), (2,)
TUNE = {"A-3", "A-7", "A-8", "C-1", "E-11", "F-7", "G-7", "M-3", "M-4", "P-1", "P-7", "T-8"}
EVAL = {"A-2", "A-4", "A-9", "D-16", "E-1", "E-10", "E-12", "E-13", "F-3", "F-8", "T-1",
        "T-12", "T-13"}
R_GRID = (1, 3, 10)                                  # rate-residual EWMA span; 1 is none
RATE_TOL = 0.02                                      # 37.9: matched within 2%
REPRO = {"A0": (5.288128, 13), "A1": (4.431455, 15), "frozen": (0.550599, 6)}
REPRO_EVAL = {"A0": 17, "A1": 17, "frozen": 4}
CUT_RTOL = 1e-5
PRIOR_EVAL_READS = [                                 # docs/MODELS.md 79.2, finding 1
    "2026-09-10T173015Z-arms", "2026-09-10T182234Z-arms2",
    "2026-09-11T011756Z-departures", "2026-09-11T013152Z-departures2",
    "2026-09-14T212406Z-ablation", "2026-09-14T221354Z-lead"]

#: arm -> the terms it is the maximum of. `r` is substituted once frozen.
ARMS = {
    "frozen": ("frozen",),
    "A0": ("zr", "zd"),
    "A1": ("zd",),
    "A2": ("zr", "zm{rm}"),
    "A3": ("zrg", "zq{r}"),
    "A4": ("zrg", "zq{r}", "zd"),
    "A5": ("zr", "z5"),
    "K1": ("zr", "zk{r}"),
}
#: 79's rider (2026-09-26, before any TUNE number): two primary comparisons. A4 is the
#: rate INSIDE the forecaster; A2 is the rate as post-processing on its OUTPUTS.
PRIMARIES = ("A4", "A2")
TERM_NAMES = {"zr": "z_res (univariate)", "zd": "z_der", "zm": "z_mot", "z5": "z_slope",
              "zrg": "z_res (joint g)", "zq": "z_rate (joint g)", "zk": "z_res unsmoothed",
              "frozen": "telemanom ratio"}


# ======================================================================================
# the store guard (stop 61)
def store_manifest() -> str:
    h = hashlib.sha256()
    files = sorted(p for p in CACHED_STORE.iterdir() if p.is_file())
    for p in files:
        h.update(p.name.encode()); h.update(hashlib.sha256(p.read_bytes()).digest())
    return f"{len(files)}:{h.hexdigest()}"


# ======================================================================================
# forecasting: `detectors._smoothed_errors` / `_forecast`, with the per-origin slope kept
def forecast_parts(det, values):
    """`(agg, one, two)`, each `(T, C)`: the forecast `_forecast` makes, bit for bit, and
    the forecasts OF `t-1` and OF `t` made at origin `t-2` (D86.A5)."""
    filled = det._filled(values)
    steps, channels = filled.shape
    window = det.hyper.window
    agg = np.empty((steps, channels), np.float32)
    one = np.empty((steps, channels), np.float32)
    two = np.empty((steps, channels), np.float32)
    block = det.chunks * det.chunk_steps
    for blo in range(0, steps, block):
        bhi = min(blo + block, steps)
        starts = list(range(blo, bhi, det.chunk_steps))
        lengths = [min(det.chunk_steps, bhi - s) for s in starts]
        span = window + max(lengths)
        batch = np.empty((len(starts), span, channels), np.float32)
        for i, start in enumerate(starts):
            front = max(0, window - start)
            take = filled[start - window + front:start + lengths[i]]
            back = span - front - take.shape[0]
            pieces = [np.repeat(take[:1], front, axis=0)] if front else []
            pieces.append(take)
            if back:
                pieces.append(np.repeat(take[-1:], back, axis=0))
            batch[i] = np.concatenate(pieces) if len(pieces) > 1 else take
        predictions, _ = reference.forward(det._weights, batch)
        for i, start in enumerate(starts):
            n = lengths[i]
            aggregated = aggregate_predictions(predictions[i], det.config.aggregate)
            agg[start:start + n] = aggregated[window:window + n]
            origin = np.arange(window - 2, window - 2 + n)      # origin t-2 for each t
            one[start:start + n] = predictions[i][origin, 0]
            two[start:start + n] = predictions[i][origin, 1]
    return agg, one, two


def _one_thread():
    """One thread per worker: `lstm.train` sets `lstm.THREADS` (4, chosen for the Mac's
    mixed cores) at every call, which would put 4 x workers threads on the box's cores."""
    import torch
    from sentinel_models import lstm
    lstm.THREADS = 1
    torch.set_num_threads(1)


def ewma(err, span):
    """`telemanom.ewma`, except span 1 -- "no smoothing" (79.4) -- which is the identity:
    alpha = 1, and `telemanom.ewma`'s closed form divides by (1 - alpha)**k = 0 there and
    returns NaN (found by the 79.2 smoke, S-1 and E-2)."""
    if span == 1:
        return np.asarray(err, np.float64).copy()
    return np.asarray(telemanom.ewma(np.asarray(err, np.float32)[:, None], span,
                                     telemanom.EwmaState()), np.float64)[:, 0]


def unit_scale(sd):
    """d's scale: sd(dx) on the train split, or 1 where that is 0 -- a training series
    that never moves (16 SMAP/MSL channels; 79.2's rider, fixed before any score)."""
    if not np.isfinite(sd):
        raise RuntimeError("no usable training steps: d's scale is undefined")
    return sd if sd > 0 else 1.0


def rate_column(x, sd):
    return (np.diff(x, prepend=x[0]) / sd).astype(np.float32)


FLOOR_FRACTION = 1e-3                                # MODELS 79.11


def channel_scale(x_train, unit):
    """The channel's training variation, sd of its train-split values; `unit` where that
    is 0 (a channel that never moved in training). MODELS 79.11."""
    x = np.asarray(x_train, np.float64)
    x = x[np.isfinite(x)]
    # "Never moved" is max == min, exactly: `np.std` of a constant can return 1.1e-16
    # rather than 0 (A-1, R-1), which silently defeated the fallback at first.
    if not x.size or np.ptp(x) == 0:
        return float(unit)
    return float(np.std(x))


def terms_for(uni, joint, x, sd, span, smoothing, scale):
    """Every D86 term for one scored series. `x` float64, univariate.

    The reference terms (`zr`, `zd`) are `zstat` exactly as flown. Every NEW term is
    `zstat_floored`, its trailing sd floored at 1e-3 x the channel's training scale in
    x's units, and at that divided by d's scale for the rate stream (MODELS 79.11).
    """
    fx = FLOOR_FRACTION * scale
    fd = fx / sd
    v1 = x.astype(np.float32)[:, None]
    agg, one, two = forecast_parts(uni, v1)
    e = x - agg[:, 0].astype(np.float64)
    e_s = ewma(np.abs(e), smoothing)
    out = {"e_s": e_s, "zr": zstat(e_s, span), "zd": zstat(derivative(x), span),
           **{f"zm{r}": zstat_floored(ewma(motion(x, agg[:, 0]), r), span, fx) for r in R_GRID},
           "z5": zstat_floored(slope_mismatch(x, one[:, 0], two[:, 0]), span, fx)}
    for r in R_GRID:
        out[f"zk{r}"] = zstat_floored(ewma(np.abs(e), r), span, fx)
    if joint is not None:
        v2 = np.stack([x.astype(np.float32), rate_column(x, sd)], axis=1)
        g, _, _ = forecast_parts(joint, v2)
        out["zrg"] = zstat_floored(ewma(np.abs(x - g[:, 0]), smoothing), span, fx)
        for r in R_GRID:
            out[f"zq{r}"] = zstat_floored(ewma(np.abs(v2[:, 1] - g[:, 1]), r), span, fd)
    return out, agg[:, 0]


def load_cached(det, values, usable, ctx):
    """The cached univariate fit, READ ONLY, under the key it was written with (79.2).

    Every file in `runs/_weights` was written by 2026-09-09 under an eight-element key;
    8a35810 (2026-09-21, D76's warm start) appended `init_weights` as a ninth, so today's
    `fit` misses every one of them and would refit. This rebuilds the eight-element key
    exactly as `detectors.fit` did at 8a35810^ -- `Hyper.as_dict_key` and the digests are
    unchanged since -- and loads it. A miss is an error, never a refit.
    """
    values = np.asarray(values)
    usable = np.asarray(usable, dtype=bool) & np.isfinite(values).all(axis=1)
    key = (det.hyper.as_dict_key(), ctx.channels, ctx.fold, ctx.window, values.shape,
           D._sample_digest(values), D._digest(usable[::997]), None)
    D.WEIGHT_STORE = CACHED_STORE
    cached = D._load_weights(D._digest(key))
    if cached is None:
        raise RuntimeError("no cached fit under the pre-D76 key")
    det._weights, det.report = cached
    det._impulses = None
    det._fit_window = ctx.window
    rows = values[usable] if usable.any() else values
    with np.errstate(invalid="ignore"):
        det._fill = np.nan_to_num(np.nanmean(rows, axis=0), nan=0.0).astype(np.float32)
    return det


def fit_into(det, store, values, usable, ctx):
    D.WEIGHT_STORE = store
    try:
        det.fit(values, usable, ctx)
    finally:
        D.WEIGHT_STORE = CACHED_STORE
    return det


# ======================================================================================
# SMAP/MSL
def smap_source():
    src = read.DirSource(MIRROR)
    return src, json.loads(src.get(SMAP_MANIFEST))


def smap_channel(cid):
    src, man = smap_source()
    ch = next(c for c in man["channels"] if c["channel_id"] == cid)
    keys = {o["split"]: o for o in ch["objects"]}
    arrs = {}
    for split, o in keys.items():
        blob = src.get(o["key"])
        if hashlib.sha256(blob).hexdigest() != o["sha256"]:
            raise RuntimeError(f"{o['key']}: mirrored bytes do not match the manifest")
        arrs[split] = np.load(io.BytesIO(blob))
    return arrs["train"], arrs["test"]


def smap_units():
    _, man = smap_source()
    return [c["channel_id"] for c in man["channels"]]


def smap_joint(cid, train, n_test):
    x_tr = train[:, 0].astype(np.float64)
    sd = unit_scale(float(np.std(np.diff(x_tr))))
    joint = registry.build("gru-telemanom"); joint.config = SR.proportional_config(n_test)
    v2 = np.stack([x_tr.astype(np.float32), rate_column(x_tr, sd)], axis=1)
    ctx = Context(mission="smap-msl", channels=(cid, f"{cid}/rate"), groups=(0,),
                  period_seconds=1.0, fold=0, window=(0, len(v2)), commands=None,
                  command_ids=())
    fit_into(joint, D86_STORE, v2, np.ones(len(v2), bool), ctx)
    return joint, sd


def smap_fit_unit(cid):
    _one_thread()
    try:
        train, test = smap_channel(cid)
        smap_joint(cid, train, len(test))
        return cid, "fitted"
    except Exception as exc:
        return cid, f"SKIP {type(exc).__name__}: {str(exc).splitlines()[0][:120]}"


def smap_score_unit(cid):
    _one_thread()
    try:
        train, test = smap_channel(cid)
        uni = registry.build("gru-telemanom"); uni.config = SR.proportional_config(len(test))
        v_tr, v_te = train[:, :1].astype(np.float32), test[:, :1].astype(np.float32)
        load_cached(uni, v_tr, np.ones(len(v_tr), bool), SR.ctx_for(cid, len(v_tr)))
        joint, sd = smap_joint(cid, train, len(test))
        x = v_te[:, 0].astype(np.float64)
        cfg = uni.config
        scale = channel_scale(train[:, 0], unit=1.0)       # SMAP/MSL values lie in [-1, 1]
        t, agg = terms_for(uni, joint, x, sd, cfg.error_window, cfg.smoothing_window, scale)
        # reproduction: the forecast and e_s are the detector's own, bit for bit
        ref = np.asarray(uni._smoothed_errors(v_te, SR.ctx_for(cid, len(v_te))), np.float64)[:, 0]
        if not np.array_equal(ref, t["e_s"]):
            raise RuntimeError("e_s differs from detectors._smoothed_errors")
        t["frozen"] = np.asarray(telemanom.channel_ratios(t["e_s"], cfg), np.float64)
        np.savez(OUT / "smap" / "terms" / f"{cid}.npz", x=x, warmup=uni.warmup_steps,
                 error_window=cfg.error_window, scale=scale, **t)
        return cid, "scored"
    except Exception as exc:
        return cid, f"SKIP {type(exc).__name__}: {str(exc).splitlines()[0][:120]}"


def smap_targets():
    vis = json.loads(VISIBILITY.read_text())
    return [(v["channel"], int(v["start"]), int(v["end"])) for v in vis["sequences"]
            if v["class"] == "contextual" and v["in_range"]]


def smap_labels():
    src, man = smap_source()
    import ast, csv
    rows = list(csv.DictReader(io.StringIO(src.get(man["labels_key"]).decode())))
    spans, seen = {}, set()
    for r in rows:
        if r["chan_id"] in seen:
            continue
        seen.add(r["chan_id"])
        spans[r["chan_id"]] = ast.literal_eval(r["anomaly_sequences"])
    return spans


def smap_per():
    """{cid: dict(steps, nominal, T={term: array})}, `decision_layer_arms` 215-220's masks."""
    spans = smap_labels()
    per = {}
    for p in sorted((OUT / "smap" / "terms").glob("*.npz")):
        z = np.load(p)
        cid = p.stem
        steps = len(z["x"])
        anom = np.zeros(steps, bool)
        for lo, hi in spans.get(cid, []):
            anom[lo:hi + 1] = True
        nominal = ~anom; nominal[: int(z["warmup"])] = False
        per[cid] = dict(steps=steps, nominal=nominal,
                        T={k: z[k] for k in z.files if k.startswith("z") or k == "frozen"})
    return per


# ======================================================================================
# ESA-ADB m1-g8.9.10, one channel at a time
def esa_prep():
    src = read.DirSource(MIRROR)
    catalog = Catalog.load(src)
    labels = LabelSet.from_table(read.read_annotation(src, catalog, "labels"))
    task = tasks.get(ESA_TASK)
    cids = task.selection.resolve(catalog)
    loaded = bundle_mod.load(src, catalog, labels, mission=task.mission, channel_ids=cids)
    T = len(loaded.grid)
    split = splits.forward_chaining(T, **dict(task.split_params))
    (OUT / "esa" / "prep").mkdir(parents=True, exist_ok=True)
    events = []
    full = loaded.truth
    for e in full.of(ANOMALY):
        span = full.spans.get(e.event_id)
        if span is None:
            continue
        fold = next((f.index for f in split.folds if f.test[0] <= span[0] < f.test[1]), None)
        events.append(dict(event_id=e.event_id, start=int(span[0]), end=int(span[1]),
                           fold=fold, cell=e.cell, contextual=e.contextual,
                           channels=[c for c in e.channels if c in cids]))
    for cid in cids:
        view = loaded.subset([cid], labels)
        tr = view.truth
        usable = {f.index: splits.train_mask(f, tr) for f in split.folds}
        segs = {eid: [int(s[0]), int(s[1])] for eid, s in tr.spans.items()}
        np.savez(OUT / "esa" / "prep" / f"{cid}.npz",
                 values=view.values[:, 0], valid=view.valid[:, 0],
                 anomaly=tr.per_channel[:, 0], unscorable=tr.unscorable,
                 **{f"usable{k}": v for k, v in usable.items()})
        (OUT / "esa" / "prep" / f"{cid}.json").write_text(json.dumps(dict(
            channel=cid, segments=segs, groups=list(view.groups),
            period=view.provenance["grid_period_seconds"], mission=view.mission)))
    meta = dict(task=ESA_TASK, channels=cids, steps=T,
                folds=[dict(index=f.index, train=list(f.train), test=list(f.test))
                       for f in split.folds],
                tune_folds=list(ESA_TUNE_FOLDS), eval_folds=list(ESA_EVAL_FOLDS),
                events=events, provenance=dict(loaded.provenance))
    (OUT / "esa" / "prep" / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    n = {s: sum(1 for e in events if e["fold"] in f) for s, f in
         (("TUNE", ESA_TUNE_FOLDS), ("EVAL", ESA_EVAL_FOLDS))}
    print(f"  prepared {len(cids)} channels, {T:,} steps; anomaly events by side {n} "
          f"(counted from labels before any score)")


def esa_meta():
    return json.loads((OUT / "esa" / "prep" / "meta.json").read_text())


def esa_units():
    m = esa_meta()
    return [(c, f["index"]) for c in m["channels"] for f in m["folds"]]


def esa_ctx(cid, fold, lo, hi, info, rate=False):
    chans = (cid, f"{cid}/rate") if rate else (cid,)
    return Context(mission=info["mission"], channels=chans, groups=tuple(info["groups"]),
                   period_seconds=info["period"], fold=fold, window=(lo, hi),
                   commands=None, command_ids=())


def esa_models(cid, fold):
    m = esa_meta()
    f = next(ff for ff in m["folds"] if ff["index"] == fold)
    z = np.load(OUT / "esa" / "prep" / f"{cid}.npz")
    info = json.loads((OUT / "esa" / "prep" / f"{cid}.json").read_text())
    lo, hi = f["train"]
    x = z["values"][lo:hi].astype(np.float32)
    usable = z[f"usable{fold}"][lo:hi]
    uni = registry.build("gru-telemanom")
    fit_into(uni, D86_STORE, x[:, None], usable, esa_ctx(cid, fold, lo, hi, info))
    xf = np.nan_to_num(x.astype(np.float64), nan=float(np.nanmean(x[usable])))
    dx = np.diff(xf, prepend=xf[0])
    ok = usable & np.concatenate([[False], usable[:-1]])
    sd = unit_scale(float(np.std(dx[ok])))
    v2 = np.stack([x, (dx / sd).astype(np.float32)], axis=1)
    v2[~np.isfinite(v2[:, 0]), 1] = np.nan
    joint = registry.build("gru-telemanom")
    fit_into(joint, D86_STORE, v2, usable, esa_ctx(cid, fold, lo, hi, info, rate=True))
    xt = x[usable & np.isfinite(x)].astype(np.float64)
    unit = float(np.max(np.abs(xt))) if xt.size and np.max(np.abs(xt)) > 0 else 1.0
    return uni, joint, sd, z, f, channel_scale(xt, unit)


def esa_fit_unit(unit):
    _one_thread()
    try:
        esa_models(*unit)
        return unit, "fitted"
    except Exception as exc:
        return unit, f"SKIP {type(exc).__name__}: {str(exc).splitlines()[0][:120]}"


def esa_score_unit(unit):
    _one_thread()
    cid, fold = unit
    try:
        uni, joint, sd, z, f, scale = esa_models(cid, fold)
        lo, hi = f["test"]
        warm = uni.warmup_steps
        a = max(0, lo - warm)
        x = z["values"][a:hi].astype(np.float64)
        filled = uni._filled(x.astype(np.float32)[:, None])[:, 0].astype(np.float64)
        cfg = uni.config
        t, _ = terms_for(uni, joint, filled, sd, cfg.error_window, cfg.smoothing_window, scale)
        t["frozen"] = np.asarray(telemanom.channel_ratios(t["e_s"], cfg), np.float64)
        keep = slice(lo - a, None)
        observed = z["valid"][lo:hi] & np.isfinite(z["values"][lo:hi])
        out = {k: np.asarray(v[keep], np.float32) for k, v in t.items()}
        for k in out:
            if k != "e_s":
                out[k][~observed] = -np.inf           # nothing measured is never an alarm
        np.savez(OUT / "esa" / "terms" / f"{cid}__f{fold}.npz", lo=lo, hi=hi,
                 observed=observed, scale=scale, **out)
        return unit, "scored"
    except Exception as exc:
        return unit, f"SKIP {type(exc).__name__}: {str(exc).splitlines()[0][:120]}"


def esa_per():
    """{cid: dict(steps, nominal, side, T)} on the stitched test timeline [folds[0].lo, T)."""
    m = esa_meta()
    t0 = m["folds"][0]["test"][0]
    T = m["steps"]
    side = np.zeros(T - t0, np.int8)                 # 1 TUNE, 2 EVAL
    for f in m["folds"]:
        s = 1 if f["index"] in ESA_TUNE_FOLDS else 2
        side[f["test"][0] - t0:f["test"][1] - t0] = s
    per = {}
    for cid in m["channels"]:
        z = np.load(OUT / "esa" / "prep" / f"{cid}.npz")
        nominal = ~(z["anomaly"][t0:] | z["unscorable"][t0:]) & z["valid"][t0:]
        T_ = {}
        ok = True
        for f in m["folds"]:
            p = OUT / "esa" / "terms" / f"{cid}__f{f['index']}.npz"
            if not p.exists():
                ok = False; break
            zz = np.load(p)
            for k in zz.files:
                if k.startswith("z") or k == "frozen":
                    T_.setdefault(k, np.full(T - t0, -np.inf, np.float32))
                    T_[k][f["test"][0] - t0:f["test"][1] - t0] = zz[k]
        if ok:
            per[cid] = dict(steps=T - t0, nominal=nominal, T=T_, side=side, t0=t0)
    return per


def esa_targets(which):
    """[(event_id, [(cid, lo, hi)], side_range)] for the anomaly events on one side."""
    m = esa_meta()
    t0 = m["folds"][0]["test"][0]
    folds = ESA_TUNE_FOLDS if which == "TUNE" else ESA_EVAL_FOLDS
    rng = [(f["test"][0] - t0, f["test"][1] - t0) for f in m["folds"] if f["index"] in folds]
    lo_side, hi_side = min(r[0] for r in rng), max(r[1] for r in rng)
    segs = {c: json.loads((OUT / "esa" / "prep" / f"{c}.json").read_text())["segments"]
            for c in m["channels"]}
    out = []
    for e in m["events"]:
        if e["fold"] not in folds:
            continue
        parts = [(c, max(segs[c][e["event_id"]][0] - t0, lo_side),
                  min(segs[c][e["event_id"]][1] - t0, hi_side))
                 for c in e["channels"] if e["event_id"] in segs[c]]
        out.append((e["event_id"], [p for p in parts if p[2] > p[1]], e["cell"]))
    return out


# ======================================================================================
# scoring rules shared by both datasets
def arm_terms(arm, r):
    """`r` is {"A4": r, "A2": rm}: A3, A4 and K1 take A4's span, A2 its own (79's rider)."""
    return [t.format(r=r["A4"], rm=r["A2"]) for t in ARMS[arm]]


def arm_score(d, arm, r):
    return np.maximum.reduce([d["T"][t] for t in arm_terms(arm, r)])


def nominal_of(d, which):
    if "side" not in d or which is None:
        return d["nominal"]
    return d["nominal"] & (d["side"] == (1 if which == "TUNE" else 2))


def rate(per, masks, which=None):
    a = sum(int((masks[c] & nominal_of(per[c], which)).sum()) for c in masks)
    n = sum(int(nominal_of(per[c], which).sum()) for c in masks)
    return a / n if n else float("nan")


def solve(per, scores, target, which=None, channels=None):
    """`decision_layer_arms.solve_threshold`'s rule: the (1 - target) pooled nominal quantile
    and its distinct neighbours; at-or-below target x 1.02 preferred, then closest."""
    keys = channels or list(per)
    pool = np.concatenate([scores[c][nominal_of(per[c], which)] for c in keys])
    pool = pool[np.isfinite(pool)]
    q0 = float(np.quantile(pool, 1.0 - target))
    uniq = np.unique(pool)
    i = int(np.searchsorted(uniq, q0))
    cands = np.unique(np.concatenate([uniq[max(0, i - 3): i + 4], [q0, np.nextafter(q0, np.inf)]]))
    best = None
    for thr in cands:
        masks = {c: scores[c] >= thr for c in keys}
        r = rate({c: per[c] for c in keys}, masks, which)
        k = (0 if r <= target * (1 + RATE_TOL) else 1, abs(r - target))
        if best is None or k < best[0]:
            best = (k, float(thr), r)
    return best[1], best[2]


def smap_caught(masks, targets, which):
    return [f"{c}[{lo}]" for c, lo, hi in targets
            if c in which and c in masks and masks[c][lo:hi + 1].any()]


def esa_caught(masks, targets):
    return [eid for eid, parts, _ in targets
            if any(c in masks and masks[c][lo:hi].any() for c, lo, hi in parts)]


def attribution(per, arm, r, cut, windows):
    """For each caught event: the terms that reached the cut at any alarm step inside
    its window(s), and the term(s) at the first alarm step."""
    names = arm_terms(arm, r)
    first, anyof, t_first = None, set(), None
    for c, lo, hi in windows:
        if c not in per:
            continue
        stack = np.stack([per[c]["T"][t][lo:hi] for t in names])
        hits = (stack >= cut)
        alarm = hits.any(axis=0)
        if not alarm.any():
            continue
        idx = np.flatnonzero(alarm)
        anyof |= {names[j] for j in range(len(names)) if hits[j, idx].any()}
        k = int(idx[0]) + lo
        if t_first is None or k < t_first:
            t_first, first = k, sorted(names[j] for j in range(len(names)) if hits[j, idx[0]])
    return dict(first=first, any=sorted(anyof))


# ======================================================================================
def run_pool(fn, units, workers):
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for unit, status in ex.map(fn, units, chunksize=1):
            out[str(unit)] = status
            print(f"    {unit}: {status}", flush=True)
    return out


def limit(units, only):
    """`--only`: a smoke restricted to named units. SMAP smokes may name only channels
    outside TUNE and EVAL (79.2); nothing a smoke writes is read by `tune` or `eval`."""
    if not only:
        return units
    if any(u in TUNE | EVAL for u in only):
        raise SystemExit("  (!) a smoke may not touch a TUNE or EVAL channel")
    return [u for u in units if (u if isinstance(u, str) else u[0]) in only]


def phase_fit(ds, workers, only=None):
    D86_STORE.mkdir(parents=True, exist_ok=True)
    units = limit(smap_units() if ds == "smap" else esa_units(), only)
    fn = smap_fit_unit if ds == "smap" else esa_fit_unit
    status = run_pool(fn, units, workers)
    (OUT / ds).mkdir(parents=True, exist_ok=True)
    (OUT / ds / "fit_status.json").write_text(json.dumps(status, indent=1))


def phase_score(ds, workers, only=None):
    (OUT / ds / "terms").mkdir(parents=True, exist_ok=True)
    units = limit(smap_units() if ds == "smap" else esa_units(), only)
    fn = smap_score_unit if ds == "smap" else esa_score_unit
    status = run_pool(fn, units, workers)
    (OUT / ds / "score_status.json").write_text(json.dumps(status, indent=1))


def target_rate():
    st4 = json.loads(STAGE4.read_text())
    return float(st4["arms"]["gru"]["nominal_rate"]), float(st4["arms"]["gru"]["multiplier"])


def phase_tune(ds):
    target, frozen_mult = target_rate()
    frozen = dict(dataset=ds, section="docs/MODELS.md 79", target_rate=target,
                  r_grid=list(R_GRID), primaries=list(PRIMARIES), cuts={}, rates={}, tune={})
    if ds == "smap":
        per = smap_per()
        targets = [t for t in smap_targets() if t[0] in TUNE]      # EVAL never loaded
        caught = lambda masks: smap_caught(masks, targets, TUNE)   # noqa: E731
        which = None
        frozen["channels_scored"] = len(per)
    else:
        per = esa_per()
        targets = esa_targets("TUNE")
        caught = lambda masks: esa_caught(masks, targets)          # noqa: E731
        which = "TUNE"
        frozen["channels_scored"] = len(per)
        frozen["tune_events"] = len(targets)

    def run(arm, r, cut=None):
        scores = {c: arm_score(per[c], arm, r) for c in per}
        if cut is None:
            cut, rt = solve(per, scores, target, which)
        else:
            rt = rate(per, {c: scores[c] >= cut for c in per}, which)
        masks = {c: scores[c] >= cut for c in per}
        return cut, rt, caught(masks)

    # each primary's span on its OWN TUNE catches; ties to the smaller (79.4 and its rider).
    # The span that is not being chosen is held at 1 -- it does not enter the other's arm.
    r_best, frozen["r_selection"] = {}, {}
    for arm in PRIMARIES:
        by_r = {r: run(arm, {"A4": r, "A2": r}) for r in R_GRID}
        r_best[arm] = max(R_GRID, key=lambda r: (len(by_r[r][2]), -r))
        frozen["r_selection"][arm] = {str(r): dict(cut=v[0], rate=v[1], tune=len(v[2]))
                                      for r, v in by_r.items()}
    frozen["r"] = r_best
    for arm in ARMS:
        fixed = frozen_mult if (arm == "frozen" and ds == "smap") else None
        cut, rt, got = run(arm, r_best, fixed)
        frozen["cuts"][arm], frozen["rates"][arm] = cut, rt
        frozen["tune"][arm] = dict(caught=len(got), of=len(targets), events=got,
                                   attribution={})
        off = abs(rt - target) / target
        frozen["tune"][arm]["rate_matched"] = bool(off <= RATE_TOL) or arm == "frozen"
    if ds == "smap":
        # the TUNE-channels-only cut, REPORTED NOT TARGETED (79.4)
        tchans = [c for c in per if c in TUNE]
        frozen["cuts_tune_channels_only"] = {
            arm: solve(per, {c: arm_score(per[c], arm, r_best) for c in per}, target,
                       channels=tchans)[0] for arm in ARMS if arm != "frozen"}
        rep = {}
        for arm, (cut0, n0) in REPRO.items():
            cut, n = frozen["cuts"][arm], frozen["tune"][arm]["caught"]
            rep[arm] = dict(cut=cut, want_cut=cut0, tune=n, want_tune=n0,
                            held=bool(abs(cut - cut0) <= CUT_RTOL * cut0 + 5e-7 and n == n0))
        frozen["reproduction"] = rep
        if not all(v["held"] for v in rep.values()):
            frozen["status"] = "STOP 62: a reproduction gate failed; nothing is scored on EVAL"
    bad = [a for a in ARMS if not frozen["tune"][a]["rate_matched"]]
    if bad:
        frozen["status"] = f"STOP 67: rate not matched within 2% for {bad}"
    frozen.setdefault("status", "FROZEN")
    for arm in ARMS:
        cut = frozen["cuts"][arm]
        for ev in frozen["tune"][arm]["events"]:
            if ds == "smap":
                c, lo = ev.split("["); lo = int(lo[:-1])
                hi = next(h for cc, ll, h in targets if cc == c and ll == lo)
                win = [(c, lo, hi + 1)]
            else:
                win = next(p for eid, p, _ in targets if eid == ev)
            frozen["tune"][arm]["attribution"][ev] = attribution(per, arm, r_best, cut, win)
    path = OUT / ds / "frozen.json"
    path.write_text(json.dumps(frozen, indent=1) + "\n")
    print(f"  {frozen['status']}; spans {r_best}; wrote {path}")
    for arm in ARMS:
        t = frozen["tune"][arm]
        print(f"    {arm:<7} cut {frozen['cuts'][arm]:10.6f}  rate {100 * frozen['rates'][arm]:.4f}%"
              f"  TUNE {t['caught']}/{t['of']}")
    return 0 if frozen["status"] == "FROZEN" else 3


def phase_eval(ds):
    path = OUT / ds / "frozen.json"
    fixture = f"tests/fixtures/d86/{ds}_frozen.json"
    committed = subprocess.run(["git", "show", f"HEAD:{fixture}"], cwd=ROOT,
                               capture_output=True).stdout
    if not committed or committed != path.read_bytes():
        raise SystemExit(f"  (!) STOP 63: {path} is not the frozen file committed at HEAD "
                         f"({fixture})")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"  (!) STOP 63: the tracked tree is not clean:\n{dirty}")
    frozen = json.loads(committed)
    if frozen["status"] != "FROZEN":
        raise SystemExit(f"  (!) the freeze did not hold: {frozen['status']}")
    r = frozen["r"]
    if ds == "smap":
        per = smap_per()
        all_t = smap_targets()
        targets = [t for t in all_t if t[0] in EVAL]
        caught = lambda masks: smap_caught(masks, targets, EVAL)   # noqa: E731
        which = None
    else:
        per = esa_per()
        targets = esa_targets("EVAL")
        caught = lambda masks: esa_caught(masks, targets)          # noqa: E731
        which = "EVAL"
    result = dict(dataset=ds, frozen_sha256=hashlib.sha256(committed).hexdigest(), r=r,
                  eval_reads=dict(prior_full_reads=PRIOR_EVAL_READS,
                                  this_read="at least the seventh full EVAL read"
                                  if ds == "smap" else "first D86 read of fold 3"),
                  arms={})
    for arm in ARMS:
        cut = frozen["cuts"][arm]
        scores = {c: arm_score(per[c], arm, r) for c in per}
        masks = {c: scores[c] >= cut for c in per}
        got = caught(masks)
        att = {}
        for ev in got:
            if ds == "smap":
                c, lo = ev.split("["); lo = int(lo[:-1])
                hi = next(h for cc, ll, h in targets if cc == c and ll == lo)
                win = [(c, lo, hi + 1)]
            else:
                win = next(p for eid, p, _ in targets if eid == ev)
            att[ev] = attribution(per, arm, r, cut, win)
        result["arms"][arm] = dict(cut=cut, eval_caught=len(got), of=len(targets),
                                   events=got, attribution=att,
                                   rate=rate(per, masks, which),
                                   tune_caught=frozen["tune"][arm]["caught"],
                                   tune_of=frozen["tune"][arm]["of"],
                                   tune_rate=frozen["rates"][arm])
    a0, a1 = set(result["arms"]["A0"]["events"]), set(result["arms"]["A1"]["events"])
    result["primaries"] = {}
    for arm in PRIMARIES:
        got = set(result["arms"][arm]["events"])
        result["primaries"][arm] = dict(
            holds_every_A0_catch=a0 <= got, lost=sorted(a0 - got), gained=sorted(got - a0),
            gained_attribution={e: result["arms"][arm]["attribution"][e]
                                for e in sorted(got - a0)},
            vs_A1=dict(lost=sorted(a1 - got), gained=sorted(got - a1)))
    if ds == "smap":
        result["reproduction_eval"] = {a: dict(eval=result["arms"][a]["eval_caught"], want=n,
                                               held=result["arms"][a]["eval_caught"] == n)
                                       for a, n in REPRO_EVAL.items()}
    out = OUT / ds / "eval.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"  wrote {out}")
    for arm in ARMS:
        a = result["arms"][arm]
        print(f"    {arm:<7} EVAL {a['eval_caught']}/{a['of']}  TUNE {a['tune_caught']}/"
              f"{a['tune_of']}  rate {100 * a['rate']:.4f}%")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phase", choices=["prep-esa", "fit", "score", "tune", "eval"])
    ap.add_argument("--dataset", choices=["smap", "esa"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--only", nargs="+", help="smoke: these units only (never TUNE/EVAL)")
    a = ap.parse_args(argv)
    before = store_manifest()
    print(f"  store {CACHED_STORE}: {before.split(':')[0]} files, manifest {before[-16:]}")
    if a.phase == "prep-esa":
        esa_prep(); rc = 0
    elif a.phase == "fit":
        phase_fit(a.dataset, a.workers, a.only); rc = 0
    elif a.phase == "score":
        phase_score(a.dataset, a.workers, a.only); rc = 0
    elif a.phase == "tune":
        rc = phase_tune(a.dataset)
    else:
        rc = phase_eval(a.dataset)
    after = store_manifest()
    if after != before:
        print(f"  (!) STOP 61: the cached store moved {before} -> {after}")
        return 61
    print(f"  store unmoved ({after[-16:]})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
