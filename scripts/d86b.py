#!/usr/bin/env python3
"""D86b: faults whose rate is normal and whose value is wrong, on the testbed (MODELS 80).

    PYTHONPATH=src .venv/bin/python scripts/d86b.py --work DIR [--k K] [--seeds 1 2 3]

Registered in docs/MODELS.md 80 before any detector saw these runs. The runs are regenerated
from their seeds by LoopSim into --work and deleted afterwards by the caller; only results
are kept (D86.4). `--k` adds D86.A6 with the k frozen by MODELS 79.13's TUNE; without it,
A1, R and A0 are scored. No wall-clock figure is printed (stop 35).
"""
from __future__ import annotations

import argparse, hashlib, json, subprocess, sys  # noqa: E401
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel_export.reader import read_model                          # noqa: E402
from sentinel_models import reference, telemanom                       # noqa: E402
from sentinel_models.windows import aggregate_predictions              # noqa: E402
from sentinel_toolkit.statistic import derivative, zstat               # noqa: E402

LOOPSIM = ROOT / "oxcaml" / "_build" / "shape-c8-p10" / "loopsim"
FLYING = ROOT / "runs" / "d85" / "flying" / "flying.bin"
SEGMENT = ROOT / "runs" / "d85" / "flying" / "flying.bin.segment.npz"
NAMES = ["SolarInput", "ChargeCurrent", "LoadCurrent", "BusVoltage", "CellTemp",
         "RadiatorTemp", "HeaterDuty", "StateOfCharge"]

# -- MODELS 80.2 / 80.3, fixed before any detector ran -----------------------------------
TICKS, SPINUP, ONSET = 100_000, 60_000, 20_000
BASE = ["ticks=100000", "spinup=60000", "balance=0.60,0.92", "retrain=0"]
FAULTS = {
    "a": ["sensor=drift,4,20000,5e-5,1"],
    "b": ["sensor=offset,3,20000,0.4,2000"],
    "c": ["loadshift=20000,450"],
    "d": ["ageing=20000,8000,0,0.6"],
    "e": ["sensor=gain,1,20000,1.05,2000"],
}
WINDOW = 2100                           # flight's ERROR_WINDOW
FLOOR_FRACTION = 1e-3                   # 79.11


def weights_of(spec):
    arrays = dict(spec["arrays"])
    layers = tuple(reference.LayerWeights(arrays[f"l{i}_w_ih"], arrays[f"l{i}_w_hh"],
                                          arrays[f"l{i}_b_ih"], arrays[f"l{i}_b_hh"])
                   for i in range(len(spec["hidden"])))
    return reference.Weights(layers=layers, head_w=arrays["head_w"], head_b=arrays["head_b"],
                             n_channels=spec["n_channels"], window=spec["window"],
                             n_predictions=spec["n_predictions"],
                             n_exogenous=spec["n_exogenous"])


def forecast(w, x):
    """One-step forecasts with the recurrent state CARRIED across the whole run, as flight
    runs it: the forecast OF t is the mean of the predictions of t made at t-1 .. t-10."""
    preds, _ = reference.forward(w, x[None].astype(np.float32))
    return aggregate_predictions(preds[0], "mean").astype(np.float64)


def cusum(u, k):
    """MODELS 79.13: two-sided Page CUSUM, floor at 0 the only reset, none at an alarm."""
    sp = np.zeros_like(u); sm = np.zeros_like(u); a = b = 0.0
    for t, v in enumerate(u):
        a = max(0.0, a + v - k); b = max(0.0, b - v - k)
        sp[t], sm[t] = a, b
    return np.maximum(sp, sm)


def run(work: Path, name: str, seed: int, extra: list[str]) -> np.ndarray:
    out = work / f"{name}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(LOOPSIM), f"out={out}", f"seed={seed}", *BASE, *extra], check=True,
                   capture_output=True)
    x = np.fromfile(out / "trace.f32", dtype=np.float32).reshape(-1, 8).astype(np.float64)
    assert len(x) == TICKS
    return x


def terms(w, spec, x, ref=None, k=None):
    """Per-channel z_res and z_der as flown, and S when `k` is given."""
    f = forecast(w, x)
    e = x - f
    span = int(spec["params"]["ewma_span"])
    zr = np.stack([zstat(np.asarray(telemanom.ewma(np.abs(e[:, c:c + 1]).astype(np.float32),
                                                   span, telemanom.EwmaState()),
                                    np.float64)[:, 0], WINDOW) for c in range(8)], axis=1)
    zd = np.stack([zstat(derivative(x[:, c]), WINDOW) for c in range(8)], axis=1)
    out = {"zr": zr, "zd": zd}
    if k is not None:
        mu, sd, scale = ref
        u = (e - mu) / np.maximum(sd, FLOOR_FRACTION * scale)
        out["S"] = np.stack([cusum(u[:, c], k) for c in range(8)], axis=1)
    return out


def rule(t, name):
    parts = {"A1": ["zd"], "R": ["zr"], "A0": ["zr", "zd"], "A6": ["zr", "zd", "S"]}[name]
    return np.max(np.stack([t[p] for p in parts]), axis=0)       # (T, 8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--k", type=float)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "d86b" / "result.json")
    a = ap.parse_args(argv)
    blob = FLYING.read_bytes()
    status, spec = read_model(blob)
    # The toolkit fitted the flying model on LoopSim's trace columns, which are the plant's
    # published order (PowerPlant.hpp), so channel_i is NAMES[i].
    assert int(status) == 0 and [c["name"] for c in spec["channels"]] == [
        f"channel_{i}" for i in range(8)], (status, spec and spec["channels"])
    w = weights_of(spec)
    warm, cut0 = int(spec["params"]["warmup_steps"]), float(spec["params"]["threshold"])
    ref = None
    if a.k is not None:
        seg = np.load(SEGMENT)["train"].astype(np.float64)
        es = seg - forecast(w, seg)
        es = es[warm:]
        span = seg[warm:]
        scale = np.where(np.ptp(span, axis=0) > 0, span.std(axis=0), 1.0)
        ref = (es.mean(axis=0), es.std(axis=0), scale)
    rules = ["A1", "R", "A0"] + (["A6"] if a.k is not None else [])

    ctrl, fault = {}, {}
    for s in a.seeds:
        ctrl[s] = terms(w, spec, run(a.work, "control", s, []), ref, a.k)
        for f, extra in FAULTS.items():
            fault[(f, s)] = terms(w, spec, run(a.work, f, s, extra), ref, a.k)

    # r*: the flown rule's rate at its own cut on the controls, after warm-up (80.3)
    def score(t, name):
        return rule(t, name).max(axis=1)
    pool = {n: np.concatenate([score(ctrl[s], n)[warm:] for s in a.seeds]) for n in rules}
    r_star = float(np.mean(pool["A0"] >= cut0))
    cuts = {n: float(np.quantile(pool[n], 1.0 - r_star)) for n in rules}
    rates = {n: float(np.mean(pool[n] >= cuts[n])) for n in rules}

    rows = []
    for (f, s), t in sorted(fault.items()):
        row = {"fault": f, "seed": s}
        for n in rules:
            fs, cs = score(t, n), score(ctrl[s], n)
            att = (fs >= cuts[n]) & ~(cs >= cuts[n])
            att[:ONSET] = False
            hit = np.flatnonzero(att)
            if hit.size:
                i = int(hit[0])
                parts = {"A1": ["zd"], "R": ["zr"], "A0": ["zr", "zd"], "A6": ["zr", "zd", "S"]}[n]
                crossed = sorted({f"{p}@{NAMES[c]}" for p in parts for c in range(8)
                                  if t[p][i, c] >= cuts[n]})
                row[n] = dict(caught=True, first_tick=i, lead_from_onset=i - ONSET,
                              attributable_ticks=int(hit.size), crossed=crossed)
            else:
                row[n] = dict(caught=False)
        rows.append(row)
    result = dict(section="docs/MODELS.md 80", model_sha256=hashlib.sha256(blob).hexdigest(),
                  r_star=r_star, flown_cut=cut0, cuts=cuts, rates=rates, k=a.k,
                  seeds=a.seeds, rows=rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"  r* {100 * r_star:.4f}%  cuts {({n: round(c, 4) for n, c in cuts.items()})}")
    for n in rules:
        print(f"  {n:<3} " + "  ".join(
            f"{f}:" + "".join("Y" if r[n]["caught"] else "." for r in rows if r["fault"] == f)
            for f in FAULTS))
    for r in rows:
        print("   ", r["fault"], r["seed"], {n: (r[n]["crossed"][:3], r[n]["lead_from_onset"])
                                           if r[n]["caught"] else "-" for n in rules})
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
